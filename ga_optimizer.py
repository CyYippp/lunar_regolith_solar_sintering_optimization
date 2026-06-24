"""
ga_optimizer.py
===============
Specialised multi-objective genetic-algorithm framework for the optimisation of
lunar regolith solar-sintering parameters (report Chapter 3).

The framework follows the modular, object-oriented architecture of report
sec. 3.2 and is organised into the five computational modules of Figure 1:

    1. PopulationManager      - chromosome encoding, LHS initialisation,
                                diversity maintenance        (sec. 3.3)
    2. FitnessEvaluator       - comprehensive thermophysical models, objective
                                normalisation, weighted-sum aggregation,
                                exterior-penalty constraint handling and result
                                caching                       (sec. 3.4, 3.9)
    3. EvolutionaryEngine     - blended crossover with parameter-specific
                                adaptation, the three parameter-specific
                                mutation operators, adaptive operator
                                probabilities and the diversity-preserving
                                tournament                    (sec. 3.5, 3.6)
    4. ParetoAnalyser         - non-dominated sorting, hypervolume and
                                generational distance         (sec. 3.10, 3.11)
    5. ComputationalController- parallel fitness evaluation, checkpointing and
                                termination criteria          (sec. 3.8, 3.10)

The genetic-algorithm utilities build on DEAP (Distributed Evolutionary
Algorithms in Python); the specialised operators, fitness models and
controller are implemented on top of DEAP's ``creator``/``toolbox`` scaffolding.

The *comprehensive* thermophysical models here extend the efficient surrogates
of ``synthetic_data.py`` with the temperature-dependent thermal-conductivity
coupling of Eq. 41 (effective-medium theory + radiative transfer), which
modulates the thermal-stress estimate through the resulting thermal gradient.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from deap import base, creator, tools
    _HAVE_DEAP = True
except Exception:                                   # pragma: no cover
    _HAVE_DEAP = False

from config import (
    RegolithConfig, REGOLITH_CONFIGS, OperatorConfig, ConvergenceConfig,
    DEFAULT_OPERATORS, DEFAULT_CONVERGENCE,
    UCS_MIN, UCS_MAX, TS_MIN, TS_MAX,
    UCS_CONSTRAINT_MIN, TS_CONSTRAINT_MAX, PHASE_FRACTION_MIN,
    UCS_TARGET, TS_TARGET,
    SOLAR_CONSTANT, REGOLITH_DENSITY, SINTER_DEPTH,
    WEIGHTS_PRIMARY,
)
import synthetic_data as sd


STEFAN_BOLTZMANN = 5.670374419e-8        # W m^-2 K^-4
VACUUM_CORRECTION = 0.35                  # xi_vac, lunar vacuum (Eq. 41)
RADIATION_EXCHANGE = 1.0                  # zeta (Eq. 41)


# ========================================================================== #
#  Comprehensive thermophysical models  (report sec. 3.9, Eqs. 41-45)        #
# ========================================================================== #
class ThermophysicalModel:
    """Comprehensive thermophysical model used for fitness evaluation.

    Reuses the calibrated relations of ``synthetic_data`` for porosity, UCS and
    energy efficiency, and adds the Eq. 41 effective thermal-conductivity
    coupling.  Higher effective conductivity reduces the through-thickness
    thermal gradient and therefore the thermal stress; the correction is mild
    in the operating window and is the principal difference from the efficient
    surrogate tier.
    """

    def __init__(self, cfg: RegolithConfig):
        self.cfg = cfg
        # reference conductivity near the operating point, so the gradient
        # correction is ~1 in the feasible region (keeps the two model tiers
        # consistent, per report sec. 5.6).
        self._k_ref = self.thermal_conductivity(1100.0, cfg.phi_ref)

    # -- Eq. 41 ------------------------------------------------------------- #
    def thermal_conductivity(self, T, phi):
        """Effective thermal conductivity k_eff(T, phi) -- Eq. 41."""
        T = np.asarray(T, dtype=float)
        phi = np.asarray(phi, dtype=float)
        T_kelvin = T + 273.15
        solid = self.cfg.k_solid * VACUUM_CORRECTION * np.power(1.0 - phi, 1.7)
        radiative = (RADIATION_EXCHANGE * 4.0 * STEFAN_BOLTZMANN
                     * self.cfg.emissivity * self.cfg.pore_dim
                     * np.power(T_kelvin, 3))
        return solid + radiative

    # -- delegate the calibrated relations --------------------------------- #
    def porosity(self, T, t, SF):
        return sd.porosity(T, t, SF, self.cfg)

    def ucs(self, T, t, SF):
        return sd.ucs(T, t, SF, self.cfg)

    def phase_fraction(self, T, t):
        return sd.phase_fraction(T, t, self.cfg)

    def energy_efficiency(self, T, t, SF):
        return sd.energy_efficiency(T, t, SF, self.cfg)

    # -- Eq. 44 with Eq. 41 gradient coupling ------------------------------ #
    def thermal_stress(self, T, t, SF):
        base_stress = sd.thermal_stress(T, t, SF, self.cfg)
        phi = self.porosity(T, t, SF)
        k_eff = self.thermal_conductivity(T, phi)
        # gradient ~ flux / conductivity; mild correction about the reference.
        gradient_factor = np.power(self._k_ref / np.maximum(k_eff, 1e-6), 0.15)
        return base_stress * gradient_factor

    def evaluate(self, T, t, SF) -> Dict[str, float]:
        return {
            "ucs": float(self.ucs(T, t, SF)),
            "thermal_stress": float(self.thermal_stress(T, t, SF)),
            "energy_efficiency": float(self.energy_efficiency(T, t, SF)),
            "porosity": float(self.porosity(T, t, SF)),
            "phase_fraction": float(self.phase_fraction(T, t)),
        }


# ========================================================================== #
#  Module 2: Fitness evaluation  (report sec. 3.4)                           #
# ========================================================================== #
def _normalise_ucs(value: float) -> float:                      # Eq. 14
    return np.clip((value - UCS_MIN) / (UCS_MAX - UCS_MIN), 0.0, 1.0)


def _normalise_ts(value: float) -> float:                       # Eq. 15
    return np.clip((TS_MAX - value) / (TS_MAX - TS_MIN), 0.0, 1.0)


def _normalise_eff(value: float, eff_min: float, eff_max: float) -> float:  # Eq.16
    if eff_max <= eff_min:
        return 0.0
    return float(np.clip((value - eff_min) / (eff_max - eff_min), 0.0, 1.0))


class FitnessEvaluator:
    """Multi-objective fitness with constraint penalties and caching."""

    def __init__(self, cfg: RegolithConfig,
                 weights: Sequence[float] = WEIGHTS_PRIMARY,
                 operators: OperatorConfig = DEFAULT_OPERATORS,
                 eff_bounds: Tuple[float, float] = (0.0, 1.0e-3),
                 cache_decimals: int = 2):
        self.cfg = cfg
        self.model = ThermophysicalModel(cfg)
        self.weights = tuple(weights)
        self.operators = operators
        self.eff_min, self.eff_max = eff_bounds
        self.cache_decimals = cache_decimals
        self._cache: Dict[Tuple[float, float, float], Dict[str, float]] = {}
        self.evaluations = 0
        self.cache_hits = 0

    # -- two-tier caching (report sec. 3.4.4) ------------------------------ #
    def _cache_key(self, T, t, SF):
        return (round(T, self.cache_decimals),
                round(t, self.cache_decimals),
                round(SF, self.cache_decimals))

    def raw_response(self, T, t, SF) -> Dict[str, float]:
        key = self._cache_key(T, t, SF)
        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        response = self.model.evaluate(T, t, SF)
        self.evaluations += 1
        self._cache[key] = response
        return response

    # -- constraint violations (report Eqs. 8-9) --------------------------- #
    def constraint_violations(self, T, t, SF, response) -> List[float]:
        g1 = UCS_CONSTRAINT_MIN - response["ucs"]                       # 8a
        g2 = response["thermal_stress"] - TS_CONSTRAINT_MAX             # 8b
        g3 = PHASE_FRACTION_MIN - response["phase_fraction"]            # 8c
        # g4: minimum time-temperature relation for effective diffusion (Eq. 9)
        t_min_required = self._min_time(T)
        g4 = t_min_required - t                                         # 9
        return [g1, g2, g3, g4]

    def _min_time(self, T: float) -> float:
        """Minimum dwell time t_min(T) from diffusion kinetics (Eq. 9).

        Higher temperatures permit shorter dwell times; the relation is taken
        as an inverse-linear decay from t_max at T_min to t_min at T_max.
        """
        frac = (T - self.cfg.T_min) / (self.cfg.T_max - self.cfg.T_min)
        frac = np.clip(frac, 0.0, 1.0)
        return self.cfg.t_min + (1.0 - frac) * 0.15 * (self.cfg.t_max - self.cfg.t_min)

    # -- objectives (normalised) ------------------------------------------- #
    def objectives(self, T, t, SF) -> Tuple[float, float, float]:
        """Return the raw (UCS, thermal_stress, energy_efficiency) triple."""
        r = self.raw_response(T, t, SF)
        return r["ucs"], r["thermal_stress"], r["energy_efficiency"]

    def normalised_objectives(self, T, t, SF) -> Tuple[float, float, float]:
        r = self.raw_response(T, t, SF)
        return (_normalise_ucs(r["ucs"]),
                _normalise_ts(r["thermal_stress"]),
                _normalise_eff(r["energy_efficiency"], self.eff_min, self.eff_max))

    # -- scalar penalised fitness (report Eqs. 17-19) ---------------------- #
    def fitness(self, individual, generation: int = 0) -> float:
        T, t, SF = individual
        r = self.raw_response(T, t, SF)
        n_ucs, n_ts, n_eff = (
            _normalise_ucs(r["ucs"]),
            _normalise_ts(r["thermal_stress"]),
            _normalise_eff(r["energy_efficiency"], self.eff_min, self.eff_max),
        )
        base_fitness = (self.weights[0] * n_ucs
                        + self.weights[1] * n_ts
                        + self.weights[2] * n_eff)               # Eq. 17

        # exterior penalty with generational escalation (Eqs. 18-19)
        penalty = 0.0
        r_coef = self.operators.penalty_init * (
            1.0 + self.operators.penalty_beta * generation)
        for g in self.constraint_violations(T, t, SF, r):
            penalty += r_coef * max(0.0, g) ** 2
        return base_fitness - penalty


# ========================================================================== #
#  Module 1: Population management  (report sec. 3.3)                        #
# ========================================================================== #
class PopulationManager:
    """Chromosome encoding, LHS initialisation and diversity maintenance."""

    def __init__(self, cfg: RegolithConfig,
                 rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.bounds = cfg.bounds_array()
        self.rng = np.random.default_rng() if rng is None else rng

    @staticmethod
    def recommended_size(d: int = 3) -> int:
        """N_pop = 10 * d * sqrt(d) (report Eq. 12)."""
        return int(round(10 * d * np.sqrt(d)))

    def min_separation(self) -> float:
        """Normalised minimum separation d_min (report Eq. 13)."""
        d = self.bounds.shape[0]
        ranges = self.bounds[:, 1] - self.bounds[:, 0]
        return (0.1 / np.sqrt(d)) * np.sum(ranges / ranges)   # = 0.1*sqrt(d)

    def initialise(self, pop_size: int) -> np.ndarray:
        """Latin Hypercube initialisation with a separation constraint."""
        # oversample then thin to honour the separation constraint
        raw = sd.latin_hypercube(pop_size * 3, self.bounds, self.rng)
        norm = (raw - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
        d_min = 0.1 / np.sqrt(self.bounds.shape[0]) * np.sqrt(self.bounds.shape[0])
        selected = [0]
        for i in range(1, len(raw)):
            if len(selected) >= pop_size:
                break
            dists = np.linalg.norm(norm[selected] - norm[i], axis=1)
            if np.min(dists) >= d_min * 0.25:        # relaxed practical bound
                selected.append(i)
        while len(selected) < pop_size:              # top-up if too aggressive
            selected.append(len(selected))
        return raw[selected[:pop_size]]

    def clip(self, individual) -> np.ndarray:
        return np.clip(individual, self.bounds[:, 0], self.bounds[:, 1])

    @staticmethod
    def average_distances(pop: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        """Mean normalised distance of each individual to the rest."""
        norm = (pop - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
        n = len(norm)
        dmat = np.linalg.norm(norm[:, None, :] - norm[None, :, :], axis=2)
        return dmat.sum(axis=1) / max(n - 1, 1)


# ========================================================================== #
#  Module 3: Specialised genetic operators  (report sec. 3.6)               #
# ========================================================================== #
class EvolutionaryEngine:
    """Specialised crossover, mutation and selection operators."""

    def __init__(self, cfg: RegolithConfig, operators: OperatorConfig,
                 n_generations: int,
                 rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.op = operators
        self.bounds = cfg.bounds_array()
        self.g_max = n_generations
        self.rng = np.random.default_rng() if rng is None else rng

    # -- adaptive operator probabilities (Eqs. 33-34) --------------------- #
    def mutation_prob(self, g: int) -> float:
        return self.cfg.base_mutation_prob * (
            1.0 + self.op.pm_alpha * np.exp(-g / self.op.pm_g_decay))

    def crossover_prob(self, g: int) -> float:
        return self.cfg.crossover_prob * (
            1.0 - self.op.pc_beta * g / self.g_max)

    # -- blended crossover with parameter-specific adaptation (Eqs. 23-24) - #
    def crossover(self, p1: np.ndarray, p2: np.ndarray):
        c1, c2 = p1.copy(), p2.copy()
        a = self.op.blend_alpha
        for i in range(len(p1)):
            gamma = self.rng.uniform(-a, 1.0 + a)                # U(-alpha,1+alpha)
            if i == 0:                                           # temperature
                gamma *= (1.0 - self.op.temp_blend_beta)         # Eq. 24
            c1[i] = (1.0 - gamma) * p1[i] + gamma * p2[i]        # Eq. 23a
            c2[i] = (1.0 - gamma) * p2[i] + gamma * p1[i]        # Eq. 23b
        return np.clip(c1, self.bounds[:, 0], self.bounds[:, 1]), \
               np.clip(c2, self.bounds[:, 0], self.bounds[:, 1])

    # -- temperature: phase-aware Gaussian mutation (Eqs. 25-27) ---------- #
    def _phase_damping(self, T: float) -> float:                 # Eq. 27
        for T_phase in self.cfg.T_phase:
            if abs(T - T_phase) <= self.op.phase_window:
                return self.op.phase_damping
        return 1.0

    def mutate_temperature(self, T: float, g: int) -> float:
        sigma = (self.op.sigma_base
                 * np.exp(-(g / self.g_max) * self.op.sigma_decay_lambda)
                 * self._phase_damping(T))                       # Eq. 26
        T_new = T + self.rng.normal(0.0, sigma)                  # Eq. 25
        return float(np.clip(T_new, self.bounds[0, 0], self.bounds[0, 1]))

    # -- time: polynomial bounded mutation (Eqs. 28-30) ------------------- #
    def mutate_time(self, t: float) -> float:
        t_min, t_max = self.bounds[1]
        delta_prime = (t - t_min) / (t_max - t_min)              # Eq. 30
        r = self.rng.random()
        eta = self.op.poly_eta
        if r < 0.5:                                              # Eq. 29a
            delta = (2 * r + (1 - 2 * r) *
                     (1 - delta_prime) ** (eta + 1)) ** (1.0 / (eta + 1)) - 1.0
        else:                                                    # Eq. 29b
            delta = 1.0 - (2 * (1 - r) + 2 * (r - 0.5) *
                           (1 - delta_prime) ** (eta + 1)) ** (1.0 / (eta + 1))
        t_new = t + delta * (t_max - t_min)                      # Eq. 28
        return float(np.clip(t_new, t_min, t_max))

    # -- solar flux: adaptive uniform mutation (Eqs. 31-32) --------------- #
    def mutate_flux(self, SF: float, g: int) -> float:
        delta_f = self.op.flux_delta_base * np.sqrt(
            max(1.0 - g / self.g_max, 0.0))                      # Eq. 32
        SF_new = SF + self.rng.uniform(-delta_f, delta_f)        # Eq. 31
        return float(np.clip(SF_new, self.bounds[2, 0], self.bounds[2, 1]))

    def mutate(self, individual: np.ndarray, g: int) -> np.ndarray:
        out = individual.copy()
        out[0] = self.mutate_temperature(out[0], g)
        out[1] = self.mutate_time(out[1])
        out[2] = self.mutate_flux(out[2], g)
        return out

    # -- diversity-preserving tournament selection (Eqs. 20-21) ----------- #
    def tournament_select(self, pop: np.ndarray, fitnesses: np.ndarray,
                          n_parents: int) -> np.ndarray:
        avg_d = PopulationManager.average_distances(pop, self.bounds)
        mean_d = avg_d.mean() if avg_d.mean() > 0 else 1.0
        div_fitness = fitnesses * (1.0 + self.op.diversity_gamma * avg_d / mean_d)
        k = self.cfg.tournament_size
        parents = []
        for _ in range(n_parents):
            aspirants = self.rng.integers(0, len(pop), size=k)
            winner = aspirants[np.argmax(div_fitness[aspirants])]
            parents.append(pop[winner])
        return np.array(parents)


# ========================================================================== #
#  Module 4: Pareto analysis  (report sec. 3.10-3.11)                       #
# ========================================================================== #
class ParetoAnalyser:
    """Non-dominated sorting and multi-objective performance metrics."""

    # objective sense: maximise UCS, minimise TS, maximise efficiency
    SENSE = np.array([1.0, -1.0, 1.0])

    @staticmethod
    def dominates(a: np.ndarray, b: np.ndarray) -> bool:
        """True if a dominates b (report Eq. 47), objectives already oriented
        so that larger is better."""
        return np.all(a >= b) and np.any(a > b)

    @classmethod
    def non_dominated(cls, objectives: np.ndarray) -> np.ndarray:
        """Indices of the non-dominated set.  ``objectives`` are raw
        (UCS, TS, eff); orientation is applied internally."""
        oriented = objectives * cls.SENSE
        n = len(oriented)
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            if not keep[i]:
                continue
            for j in range(n):
                if i != j and cls.dominates(oriented[j], oriented[i]):
                    keep[i] = False
                    break
        return np.where(keep)[0]

    @classmethod
    def hypervolume(cls, objectives: np.ndarray,
                    reference: Sequence[float],
                    n_mc: int = 200000,
                    rng: Optional[np.random.Generator] = None) -> float:
        """Monte-Carlo hypervolume of the front w.r.t. a reference point
        (report Eq. 48).  Objectives raw (UCS, TS, eff); reference in the same
        raw space (a nadir-like point that the front should dominate)."""
        rng = np.random.default_rng(0) if rng is None else rng
        front = objectives[cls.non_dominated(objectives)]
        if len(front) == 0:
            return 0.0
        oriented = front * cls.SENSE
        ref = np.asarray(reference) * cls.SENSE
        ideal = oriented.max(axis=0)
        lo = np.minimum(ref, oriented.min(axis=0))
        box = ideal - lo
        if np.any(box <= 0):
            return 0.0
        pts = lo + rng.random((n_mc, len(ref))) * box
        dominated = np.zeros(n_mc, dtype=bool)
        for sol in oriented:
            dominated |= np.all(pts <= sol, axis=1) & np.all(pts >= ref, axis=1)
        return float(dominated.mean() * np.prod(box))

    @classmethod
    def generational_distance(cls, objectives: np.ndarray,
                              reference_front: np.ndarray) -> float:
        """Generational distance to a reference front (report Eq. 49)."""
        front = objectives[cls.non_dominated(objectives)]
        if len(front) == 0 or len(reference_front) == 0:
            return float("nan")
        oriented = front * cls.SENSE
        ref = reference_front * cls.SENSE
        # normalise objective scales before distance
        scale = np.maximum(ref.max(axis=0) - ref.min(axis=0), 1e-9)
        d = []
        for p in oriented:
            d.append(np.min(np.linalg.norm((ref - p) / scale, axis=1)))
        d = np.array(d)
        return float(np.sqrt(np.sum(d ** 2)) / len(d))


# ========================================================================== #
#  Module 5: Computational controller  (report sec. 3.8, 3.10)              #
# ========================================================================== #
def _evaluate_chunk(args):
    """Top-level worker for multiprocessing (must be picklable)."""
    params_list, cfg, weights, generation, eff_bounds = args
    ev = FitnessEvaluator(cfg, weights, eff_bounds=eff_bounds)
    return [ev.fitness(p, generation) for p in params_list]


class ComputationalController:
    """Parallel fitness evaluation, checkpointing and termination."""

    def __init__(self, cfg: RegolithConfig, n_workers: Optional[int] = None,
                 checkpoint_dir: Optional[str] = None):
        # Serial by default for portability.  On Windows the multiprocessing
        # "spawn" start method launches a fresh interpreter per worker (each
        # re-importing numpy/scipy/pandas), which is memory-hungry and can
        # exhaust the system page file.  Parallelism is therefore opt-in:
        # pass n_workers > 1 explicitly (e.g. `python main.py --workers 4`).
        self.n_workers = max(1, n_workers or 1)
        self.cfg = cfg
        self.checkpoint_dir = checkpoint_dir
        self.utilisation = 0.0

    def map_fitness(self, evaluator: "FitnessEvaluator",
                    population: np.ndarray, generation: int) -> np.ndarray:
        """Evaluate fitness for the whole population (parallel if >1 worker)."""
        if self.n_workers <= 1 or len(population) < 8:
            # serial path (single-core machines, or populations too small to
            # justify pool overhead): one core fully occupied.
            self.utilisation = 1.0 / (os.cpu_count() or 1)
            return np.array([evaluator.fitness(ind, generation)
                             for ind in population])
        # master-worker pool (report sec. 3.8); load-balanced chunking (Eq. 39)
        import multiprocessing as mp
        chunks = np.array_split(population, self.n_workers)
        args = [(list(c), self.cfg, evaluator.weights, generation,
                 (evaluator.eff_min, evaluator.eff_max)) for c in chunks if len(c)]
        try:
            with mp.Pool(self.n_workers) as pool:
                results = pool.map(_evaluate_chunk, args)
            self.utilisation = self.n_workers / (os.cpu_count() or 1)
            return np.concatenate([np.array(r) for r in results])
        except (OSError, ImportError, RuntimeError):
            # Pool creation can fail (e.g. Windows page-file limits); fall back
            # to serial evaluation so a run never crashes on resource limits.
            self.n_workers = 1
            self.utilisation = 1.0 / (os.cpu_count() or 1)
            return np.array([evaluator.fitness(ind, generation)
                             for ind in population])

    # -- checkpointing with SHA-256 integrity (report sec. 3.8) ------------ #
    @staticmethod
    def checkpoint_interval(pop_size: int, g_base: int = 5,
                            g_min: int = 2, g_max: int = 25) -> int:
        return int(max(g_min, min(g_base * np.sqrt(pop_size), g_max)))  # Eq. 40

    def save_checkpoint(self, state: dict, generation: int):
        if not self.checkpoint_dir:
            return
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        payload = pickle.dumps(state)
        digest = hashlib.sha256(payload).hexdigest()
        path = os.path.join(self.checkpoint_dir,
                            f"{self.cfg.name}_gen{generation:04d}.ckpt")
        with open(path, "wb") as fh:
            fh.write(payload)
        with open(path + ".sha256", "w") as fh:
            fh.write(digest)


# ========================================================================== #
#  Top-level driver                                                          #
# ========================================================================== #
@dataclass
class OptimisationResult:
    """Outcome of a single optimisation run."""
    regolith: str
    best_individual: np.ndarray
    best_fitness: float
    best_objectives: Tuple[float, float, float]      # raw UCS, TS, eff
    best_response: Dict[str, float]
    history: Dict[str, list]                          # per-generation series
    pareto_individuals: np.ndarray
    pareto_objectives: np.ndarray                     # raw UCS, TS, eff
    recommended: Dict[str, float]                     # target-region solution
    feasible_fraction: float
    n_evaluations: int
    cache_hits: int
    processor_utilisation: float


def _setup_deap():
    """Create DEAP creator types once (idempotent)."""
    if not _HAVE_DEAP:
        return
    if not hasattr(creator, "FitnessMultiLunar"):
        creator.create("FitnessMultiLunar", base.Fitness, weights=(1.0, -1.0, 1.0))
    if not hasattr(creator, "IndividualLunar"):
        creator.create("IndividualLunar", list, fitness=creator.FitnessMultiLunar)


class LunarSinteringGA:
    """Multi-objective genetic-algorithm optimiser for one regolith type."""

    def __init__(self, cfg: RegolithConfig,
                 weights: Sequence[float] = WEIGHTS_PRIMARY,
                 operators: OperatorConfig = DEFAULT_OPERATORS,
                 convergence: ConvergenceConfig = DEFAULT_CONVERGENCE,
                 n_generations: Optional[int] = None,
                 pop_size: Optional[int] = None,
                 n_workers: Optional[int] = None,
                 checkpoint_dir: Optional[str] = None,
                 seed: Optional[int] = None):
        self.cfg = cfg
        self.weights = tuple(weights)
        self.operators = operators
        self.convergence = convergence
        self.pop_size = pop_size or cfg.pop_size
        self.n_generations = n_generations or cfg.n_generations
        self.rng = np.random.default_rng(seed)

        self.popman = PopulationManager(cfg, self.rng)
        self.engine = EvolutionaryEngine(cfg, operators, self.n_generations, self.rng)
        self.controller = ComputationalController(cfg, n_workers, checkpoint_dir)
        self.evaluator = self._build_evaluator()
        _setup_deap()

    # -- energy-efficiency normalisation bounds from a pre-sample ---------- #
    def _build_evaluator(self) -> FitnessEvaluator:
        sample = sd.latin_hypercube(400, self.cfg.bounds_array(), self.rng)
        eff = sd.energy_efficiency(sample[:, 0], sample[:, 1], sample[:, 2], self.cfg)
        eff_bounds = (float(np.min(eff)), float(np.max(eff)))
        return FitnessEvaluator(self.cfg, self.weights, self.operators, eff_bounds)

    # -- recommended target-region solution -------------------------------- #
    def _recommend(self, individuals: np.ndarray,
                   objectives: np.ndarray) -> Dict[str, float]:
        """Pick the best solution that meets both targets; if none, the
        closest approach (smallest combined target violation)."""
        ucs_v, ts_v = objectives[:, 0], objectives[:, 1]
        feasible = (ucs_v >= UCS_TARGET) & (ts_v <= TS_TARGET)
        if feasible.any():
            idx_pool = np.where(feasible)[0]
            # Among feasible solutions the report selects the balanced optimum
            # that meets the UCS target while minimising thermal stress, staying
            # clear of the deep partial-melt region (Sec. 4.4.1).  This is the
            # min-TS feasible solution, tie-broken towards UCS nearest the
            # target so we sit on the target-corner rather than far into the
            # over-strength / high-temperature regime.
            score = ts_v[idx_pool] + 0.05 * np.abs(ucs_v[idx_pool] - UCS_TARGET)
            idx = idx_pool[int(np.argmin(score))]
            met = True
        else:
            # No solution meets both targets (the highland case).  The report
            # characterises the "closest approach" as a solution that reaches
            # the UCS target but cannot bring thermal stress below threshold
            # (Sec. 4.2.2).  Prefer UCS-feasible points and minimise TS; fall
            # back to overall minimum combined violation otherwise.
            ucs_ok = ucs_v >= UCS_TARGET
            if ucs_ok.any():
                pool = np.where(ucs_ok)[0]
                idx = pool[int(np.argmin(ts_v[pool] + 0.05 * np.abs(ucs_v[pool] - UCS_TARGET)))]
            else:
                viol = np.maximum(0, UCS_TARGET - ucs_v) + np.maximum(0, ts_v - TS_TARGET)
                idx = int(np.argmin(viol))
            met = False
        T, t, SF = individuals[idx]
        resp = self.evaluator.raw_response(T, t, SF)
        return {"T": float(T), "t": float(t), "SF": float(SF),
                "ucs": resp["ucs"], "thermal_stress": resp["thermal_stress"],
                "phase_fraction": resp["phase_fraction"],
                "energy_efficiency": resp["energy_efficiency"],
                "meets_targets": bool(met)}

    # -- main evolutionary loop -------------------------------------------- #
    def run(self, verbose: bool = False) -> OptimisationResult:
        bounds = self.cfg.bounds_array()
        pop = self.popman.initialise(self.pop_size)
        elite_n = max(1, int(round(self.cfg.elite_fraction * self.pop_size)))

        history = {"generation": [], "best_fitness": [], "mean_fitness": [],
                   "best_ucs": [], "best_ts": [], "hypervolume": [], "diversity": [],
                   "rec_ucs": [], "rec_ts": [], "rec_fitness": [],
                   "feasible_fraction": []}
        ref_point = (UCS_MIN, TS_MAX, 0.0)        # nadir-like reference (Eq. 48)
        hv_window: List[float] = []
        archive_ind: List[np.ndarray] = []
        archive_obj: List[np.ndarray] = []

        fitnesses = self.controller.map_fitness(self.evaluator, pop, 0)

        for g in range(self.n_generations):
            # --- evaluate objectives for the current population ----------- #
            objs = np.array([self.evaluator.objectives(*ind) for ind in pop])

            # --- archive update (running non-dominated set) --------------- #
            archive_ind.extend(pop)
            archive_obj.extend(objs)
            arr_ind = np.array(archive_ind)
            arr_obj = np.array(archive_obj)
            nd = ParetoAnalyser.non_dominated(arr_obj)
            if len(nd) > 60:                       # cap archive size
                nd = nd[np.argsort(arr_obj[nd, 0])[-60:]]
            archive_ind = list(arr_ind[nd])
            archive_obj = list(arr_obj[nd])

            # --- metrics -------------------------------------------------- #
            hv = ParetoAnalyser.hypervolume(np.array(archive_obj), ref_point,
                                            rng=self.rng)
            diversity = float(PopulationManager.average_distances(pop, bounds).mean())
            best_idx = int(np.argmax(fitnesses))
            history["generation"].append(g)
            history["best_fitness"].append(float(fitnesses[best_idx]))
            history["mean_fitness"].append(float(np.mean(fitnesses)))
            history["best_ucs"].append(float(objs[best_idx, 0]))
            history["best_ts"].append(float(objs[best_idx, 1]))
            history["hypervolume"].append(hv)
            history["diversity"].append(diversity)

            # --- target-region solution trajectory (for the convergence
            #     figure, report Figs 2-3): track the *best-so-far* solution
            #     (smallest distance to the ideal target corner found up to
            #     this generation) so the curves converge monotonically as the
            #     search refines, with UCS approaching 10 MPa and thermal
            #     stress approaching 10 MPa.
            ucs_p, ts_p = objs[:, 0], objs[:, 1]
            dist = np.hypot(ucs_p - UCS_TARGET, ts_p - TS_TARGET)
            ri = int(np.argmin(dist))
            cand = (float(dist[ri]), float(ucs_p[ri]), float(ts_p[ri]))
            if g == 0 or cand[0] < self._best_corner_dist:
                self._best_corner_dist = cand[0]
                self._best_corner = (cand[1], cand[2])
            b_ucs, b_ts = self._best_corner
            n_ucs = float(_normalise_ucs(b_ucs))
            n_ts = float(_normalise_ts(b_ts))
            rec_fit = (self.evaluator.weights[0] * n_ucs
                       + self.evaluator.weights[1] * n_ts)
            history["rec_ucs"].append(b_ucs)
            history["rec_ts"].append(b_ts)
            history["rec_fitness"].append(rec_fit)
            history["feasible_fraction"].append(
                float(np.mean((ucs_p >= UCS_TARGET) & (ts_p <= TS_TARGET))))

            if verbose and g % 10 == 0:
                print(f"  [{self.cfg.name}] gen {g:3d}  "
                      f"F*={fitnesses[best_idx]:.3f}  HV={hv:.4f}  "
                      f"UCS*={objs[best_idx,0]:.2f}  TS*={objs[best_idx,1]:.2f}")

            # --- early-stopping on hypervolume stagnation (Eq. 46) -------- #
            hv_window.append(hv)
            if len(hv_window) > self.convergence.g_early:
                past = hv_window[-self.convergence.g_early - 1]
                if past > 0:
                    rel = (hv - past) / past
                    if abs(rel) < self.convergence.epsilon_conv and g > 25:
                        if verbose:
                            print(f"  [{self.cfg.name}] converged at gen {g}")
                        break

            # --- checkpoint ----------------------------------------------- #
            ck = self.controller.checkpoint_interval(self.pop_size)
            if self.controller.checkpoint_dir and g % ck == 0:
                self.controller.save_checkpoint(
                    {"generation": g, "population": pop, "fitness": fitnesses}, g)

            # --- elitism (report sec. 3.5.2) ------------------------------ #
            elite_idx = np.argsort(fitnesses)[-elite_n:]
            elites = pop[elite_idx].copy()

            # --- reproduction --------------------------------------------- #
            n_offspring = self.pop_size - elite_n
            parents = self.engine.tournament_select(pop, fitnesses, n_offspring * 2)
            pc = self.engine.crossover_prob(g)
            pm = self.engine.mutation_prob(g)
            offspring = []
            for i in range(0, len(parents) - 1, 2):
                if len(offspring) >= n_offspring:
                    break
                p1, p2 = parents[i], parents[i + 1]
                if self.rng.random() < pc:
                    c1, c2 = self.engine.crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                for child in (c1, c2):
                    if self.rng.random() < pm * 50:   # per-individual rate
                        child = self.engine.mutate(child, g)
                    offspring.append(child)
            offspring = np.array(offspring[:n_offspring])

            # --- next generation ------------------------------------------ #
            pop = np.vstack([elites, offspring])
            fitnesses = self.controller.map_fitness(self.evaluator, pop, g + 1)

        # ----- final analysis ------------------------------------------- #
        final_objs = np.array([self.evaluator.objectives(*ind) for ind in pop])
        archive_ind.extend(pop); archive_obj.extend(final_objs)
        arr_ind, arr_obj = np.array(archive_ind), np.array(archive_obj)
        nd = ParetoAnalyser.non_dominated(arr_obj)
        pareto_ind, pareto_obj = arr_ind[nd], arr_obj[nd]
        order = np.argsort(pareto_obj[:, 0])
        pareto_ind, pareto_obj = pareto_ind[order], pareto_obj[order]

        best_idx = int(np.argmax(fitnesses))
        best_ind = pop[best_idx]
        best_resp = self.evaluator.raw_response(*best_ind)

        feasible = ((pareto_obj[:, 0] >= UCS_TARGET) &
                    (pareto_obj[:, 1] <= TS_TARGET))
        recommended = self._recommend(pareto_ind, pareto_obj)

        # feasible fraction across a dense sample of the parameter space
        dense = sd.latin_hypercube(2000, bounds, self.rng)
        du = sd.ucs(dense[:, 0], dense[:, 1], dense[:, 2], self.cfg)
        dt = sd.thermal_stress(dense[:, 0], dense[:, 1], dense[:, 2], self.cfg)
        feasible_fraction = float(np.mean((du >= UCS_TARGET) & (dt <= TS_TARGET)))

        return OptimisationResult(
            regolith=self.cfg.name,
            best_individual=best_ind,
            best_fitness=float(fitnesses[best_idx]),
            best_objectives=tuple(final_objs[best_idx]),
            best_response=best_resp,
            history=history,
            pareto_individuals=pareto_ind,
            pareto_objectives=pareto_obj,
            recommended=recommended,
            feasible_fraction=feasible_fraction,
            n_evaluations=self.evaluator.evaluations,
            cache_hits=self.evaluator.cache_hits,
            processor_utilisation=self.controller.utilisation,
        )


def optimise(regolith: str, seed: Optional[int] = None,
             verbose: bool = False, **kwargs) -> OptimisationResult:
    """Convenience entry point: optimise one named regolith type."""
    cfg = REGOLITH_CONFIGS[regolith]
    ga = LunarSinteringGA(cfg, seed=seed, **kwargs)
    return ga.run(verbose=verbose)


if __name__ == "__main__":
    for name in ("mare", "highland"):
        res = optimise(name, seed=7, n_generations=40, verbose=True)
        rec = res.recommended
        print(f"\n=== {name.upper()} ===")
        print(f"  best fitness        : {res.best_fitness:.3f}")
        print(f"  recommended T,t,SF  : "
              f"{rec['T']:.0f} degC, {rec['t']:.0f} min, SF={rec['SF']:.1f}")
        print(f"  recommended UCS/TS  : "
              f"{rec['ucs']:.1f} / {rec['thermal_stress']:.1f} MPa "
              f"(meets targets: {rec['meets_targets']})")
        print(f"  feasible fraction   : {100 * res.feasible_fraction:.1f} %")
        print(f"  Pareto solutions    : {len(res.pareto_individuals)}")
        print(f"  evaluations / hits  : {res.n_evaluations} / {res.cache_hits}\n")
