"""
config.py
=========
Central configuration for the lunar regolith solar-sintering optimisation
framework.

Encodes the parameter bounds (report Table 5), the meta-optimised
genetic-algorithm configuration (report Table 6), the multi-objective
normalisation boundaries (Eqs. 14-16), the physical constraint thresholds
(Eqs. 8-9) and the composition-specific coefficients for the thermophysical
models (Eqs. 41-45).

Where the report states an equation *form* but does not tabulate every
coefficient, the value has been calibrated to the literature ranges cited in
the report and to the headline outcomes of Chapter 4 (mare feasible ~4 %,
highland infeasible with minimum thermal stress ~10.4 MPa, highland requiring
+50-80 degC over mare for equivalent strength, and the mare optimum near
T = 1071 degC / t = 43 min / SF = 2.1 giving UCS ~ 10 MPa, TS ~ 9 MPa).
Such values are flagged ``# calibrated``.

The densification model applies a temperature-dependent activation factor g(T)
to the time/flux terms of the porosity relation.  This reflects the
temperature-dependent rate constant of the exponential densification kinetics
documented by Song et al. [4]; physically, long dwell times cannot densify
regolith held below its effective sintering temperature.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

# --------------------------------------------------------------------------- #
#  Multi-objective normalisation boundaries  (report sec. 3.4.1, Eqs. 14-16)   #
# --------------------------------------------------------------------------- #
UCS_MIN, UCS_MAX = 5.0, 20.0     # MPa  (Indyk & Benaroya [5])
TS_MIN, TS_MAX = 5.0, 20.0       # MPa  (Meurisse et al. [1]); low stress good

# --------------------------------------------------------------------------- #
#  Physical constraint thresholds            (report sec. 3.1.3, Eqs. 8-9)     #
# --------------------------------------------------------------------------- #
UCS_CONSTRAINT_MIN = 3.0         # MPa  minimum required strength      (g1)
TS_CONSTRAINT_MAX = 20.0         # MPa  maximum allowable thermal stress(g2)
PHASE_FRACTION_MIN = 0.15        # -    minimum phase fraction          (g3)

# --------------------------------------------------------------------------- #
#  Target / acceptance criteria for feasibility reporting (Chapter 4)          #
# --------------------------------------------------------------------------- #
UCS_TARGET = 10.0                # MPa  UCS  >= target
TS_TARGET = 10.0                 # MPa  thermal stress <= target

# --------------------------------------------------------------------------- #
#  Energy-efficiency model constants          (report Eq. 45)                  #
# --------------------------------------------------------------------------- #
SOLAR_CONSTANT = 1361.0          # W/m^2 unattenuated lunar irradiance [1]
REGOLITH_DENSITY = 1800.0        # kg/m^3
SINTER_DEPTH = 0.02              # m

# --------------------------------------------------------------------------- #
#  Phase-transformation fraction model        (used for g3 and phase %)        #
# --------------------------------------------------------------------------- #
PHASE_T_SCALE = 40.0             # degC sigmoid width about T_phase[0]
PHASE_TIME_TAU = 60.0            # min  saturation time constant

# --------------------------------------------------------------------------- #
#  Default multi-objective weight vectors     (report sec. 3.4.2, Eq. 17)      #
# --------------------------------------------------------------------------- #
WEIGHTS_PRIMARY = (0.7, 0.3, 0.0)      # (UCS, TS, efficiency)
WEIGHTS_SECONDARY = (0.6, 0.25, 0.15)  # configuration including efficiency


@dataclass
class RegolithConfig:
    """Composition-specific configuration for a single regolith type."""

    name: str

    # ----- parameter bounds (Table 5) -------------------------------------- #
    T_min: float; T_max: float          # degC
    t_min: float; t_max: float          # minutes
    SF_min: float; SF_max: float        # concentration factor

    # ----- critical phase-transformation temperatures (Eq. 27) ------------- #
    T_phase: Tuple[float, float]        # (solid->liquid, partial-melt) degC

    # ----- solidus / liquidus (Table 1) ------------------------------------ #
    solidus: Tuple[float, float]
    liquidus: Tuple[float, float]

    # ----- densification temperature gating g(T) -------------------------- #
    T_onset: float                      # degC where time/flux start to act
    T_full: float                       # degC where g(T) saturates to 1

    # ----- porosity-evolution coefficients (Eq. 42) ----------------------- #
    phi0: float
    alpha_T: float; beta_T: float
    alpha_t: float; beta_t: float
    alpha_S: float; beta_S: float
    phi_floor: float

    # ----- UCS-porosity relationship (Eq. 43) ----------------------------- #
    ucs_ref: float; phi_ref: float; gamma_ucs: float

    # ----- thermal-stress model (Eq. 44) ----------------------------------- #
    ts_base: float
    kappa_T: float; xi_T: float
    kappa_S: float; xi_S: float

    # ----- thermal-conductivity model (Eq. 41) ----------------------------- #
    k_solid: float; emissivity: float; pore_dim: float

    # ----- meta-optimised GA configuration (Table 6) ---------------------- #
    pop_size: int
    n_generations: int
    crossover_prob: float
    base_mutation_prob: float
    tournament_size: int
    elite_fraction: float
    niching_radius: float

    def bounds_array(self) -> np.ndarray:
        """Return the [[lo, hi], ...] bounds matrix for [T, t, SF]."""
        return np.array([[self.T_min, self.T_max],
                         [self.t_min, self.t_max],
                         [self.SF_min, self.SF_max]], dtype=float)


# --------------------------------------------------------------------------- #
#  Mare regolith  -- iron-rich, lower solidus, lower thermal-expansion mismatch#
# --------------------------------------------------------------------------- #
MARE = RegolithConfig(
    name="mare",
    T_min=1000.0, T_max=1300.0,
    t_min=5.0, t_max=60.0,
    SF_min=1.0, SF_max=5.0,
    T_phase=(1100.0, 1250.0),
    solidus=(1050.0, 1100.0),
    liquidus=(1200.0, 1250.0),
    T_onset=1000.0, T_full=1200.0,
    phi0=0.46,
    alpha_T=7.04e-4, beta_T=1.0,    # calibrated
    alpha_t=0.041,   beta_t=0.6,    # calibrated
    alpha_S=0.0278,  beta_S=0.8,    # calibrated
    phi_floor=0.05,
    ucs_ref=10.0, phi_ref=0.27, gamma_ucs=0.085,    # calibrated
    ts_base=4.58, kappa_T=0.075, xi_T=0.9,          # calibrated
    kappa_S=0.45, xi_S=1.0,                         # calibrated
    k_solid=1.6, emissivity=0.9, pore_dim=1.0e-4,
    pop_size=80, n_generations=100,
    crossover_prob=0.80, base_mutation_prob=0.02,
    tournament_size=5, elite_fraction=0.05, niching_radius=0.10,
)

# --------------------------------------------------------------------------- #
#  Highland regolith -- plagioclase-rich, higher solidus, larger expansion     #
#  mismatch.  Thermal-stress floor lifted so all UCS>=10 solutions exceed the  #
#  10 MPa target (report sec. 4.2.3 / 5.5).                                     #
# --------------------------------------------------------------------------- #
HIGHLAND = RegolithConfig(
    name="highland",
    T_min=1050.0, T_max=1350.0,
    t_min=5.0, t_max=60.0,
    SF_min=1.0, SF_max=5.0,
    T_phase=(1150.0, 1300.0),
    solidus=(1150.0, 1200.0),
    liquidus=(1300.0, 1350.0),
    T_onset=1050.0, T_full=1250.0,
    phi0=0.47,
    alpha_T=9.5e-4, beta_T=1.0,     # calibrated
    alpha_t=0.0373, beta_t=0.6,     # calibrated
    alpha_S=0.024,  beta_S=0.8,     # calibrated
    phi_floor=0.05,
    ucs_ref=10.0, phi_ref=0.27, gamma_ucs=0.085,    # calibrated
    ts_base=6.7, kappa_T=0.075, xi_T=0.9,           # calibrated
    kappa_S=0.45, xi_S=1.0,                         # calibrated
    k_solid=1.4, emissivity=0.9, pore_dim=1.0e-4,
    pop_size=100, n_generations=120,
    crossover_prob=0.85, base_mutation_prob=0.03,
    tournament_size=4, elite_fraction=0.05, niching_radius=0.15,
)

REGOLITH_CONFIGS: Dict[str, RegolithConfig] = {"mare": MARE, "highland": HIGHLAND}


# --------------------------------------------------------------------------- #
#  Specialised genetic-operator hyper-parameters (report sec. 3.5-3.7)         #
# --------------------------------------------------------------------------- #
@dataclass
class OperatorConfig:
    blend_alpha: float = 0.5          # alpha in U(-alpha, 1+alpha)   (Eq. 23)
    temp_blend_beta: float = 0.4      # beta_T temperature exploration(Eq. 24)
    sigma_base: float = 50.0          # degC initial step             (Eq. 26)
    sigma_decay_lambda: float = 0.5
    phase_window: float = 50.0        # degC window about T_phase      (Eq. 27)
    phase_damping: float = 0.5
    poly_eta: float = 20.0            # distribution index            (Eq. 29)
    flux_delta_base: float = 1.0      #                               (Eq. 32)
    pm_alpha: float = 0.5             #                               (Eq. 33)
    pm_g_decay: float = 20.0
    pc_beta: float = 0.2              #                               (Eq. 34)
    diversity_gamma: float = 0.25     #                               (Eq. 20)
    penalty_beta: float = 0.05        #                               (Eq. 19)
    penalty_init: float = 10.0


DEFAULT_OPERATORS = OperatorConfig()


@dataclass
class ConvergenceConfig:
    epsilon_conv: float = 1.0e-3      #                               (Eq. 46)
    g_early: int = 10
    checkpoint_interval: int = 10
    n_independent_runs: int = 30      # statistical robustness (Chapter 4)


DEFAULT_CONVERGENCE = ConvergenceConfig()
