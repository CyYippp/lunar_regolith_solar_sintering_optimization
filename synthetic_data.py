"""
synthetic_data.py
=================
Computationally efficient thermophysical surrogate models and Latin Hypercube
validation-dataset generation for the lunar regolith solar-sintering
optimisation framework (report sec. 3.9 / 3.11).

This module is the *lightweight* tier of the report's two-tier modelling
strategy.  It evaluates the porosity (Eq. 42), UCS-porosity (Eq. 43),
thermal-stress (Eq. 44) and energy-efficiency (Eq. 45) relations directly,
without the temperature-dependent thermal-conductivity coupling or result
caching used by the comprehensive models in ``ga_optimizer.py``.  It is used to
generate the reference validation dataset (300 parameter combinations, 150 per
regolith type) and as a fast surrogate for plotting and sensitivity analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    RegolithConfig, REGOLITH_CONFIGS,
    SOLAR_CONSTANT, REGOLITH_DENSITY, SINTER_DEPTH,
    PHASE_T_SCALE, PHASE_TIME_TAU,
)


# --------------------------------------------------------------------------- #
#  Core surrogate models                                                       #
# --------------------------------------------------------------------------- #
def thermal_activation(T, cfg: RegolithConfig):
    """Temperature-dependent densification activation factor g(T) in [0, 1].

    Encodes that the time- and flux-driven densification terms of Eq. 42 only
    become active once the material approaches its effective sintering
    temperature (the temperature-dependent rate constant of Song et al. [4]).
    """
    g = (np.asarray(T, dtype=float) - cfg.T_onset) / (cfg.T_full - cfg.T_onset)
    return np.clip(g, 0.0, 1.0)


def porosity(T, t, SF, cfg: RegolithConfig):
    """Porosity phi(T, t, S) -- report Eq. 42 with temperature gating.

    phi = phi0 - [ a_T (T-Tmin)^bT + g(T)( a_t (t-tmin)^bt + a_S (S-Smin)^bS ) ]
    """
    T = np.asarray(T, dtype=float)
    t = np.asarray(t, dtype=float)
    SF = np.asarray(SF, dtype=float)

    g = thermal_activation(T, cfg)
    term_T = cfg.alpha_T * np.power(np.maximum(T - cfg.T_min, 0.0), cfg.beta_T)
    term_t = cfg.alpha_t * np.power(np.maximum(t - cfg.t_min, 0.0), cfg.beta_t)
    term_S = cfg.alpha_S * np.power(np.maximum(SF - cfg.SF_min, 0.0), cfg.beta_S)

    phi = cfg.phi0 - (term_T + g * (term_t + term_S))
    return np.clip(phi, cfg.phi_floor, cfg.phi0)


def ucs(T, t, SF, cfg: RegolithConfig):
    """Unconfined compressive strength -- report Eq. 43 (exponential, [5]).

    UCS = UCS_ref * exp(-gamma * (phi*100 - phi_ref*100))
    """
    phi = porosity(T, t, SF, cfg)
    value = cfg.ucs_ref * np.exp(
        -cfg.gamma_ucs * (phi * 100.0 - cfg.phi_ref * 100.0)
    )
    return np.maximum(value, 0.0)


def thermal_stress(T, t, SF, cfg: RegolithConfig):
    """Thermal stress sigma_T -- report Eq. 44.

    sigma_T = sigma_base + kappa_T (T-Tmin)^xiT + kappa_S * S^xiS
    """
    T = np.asarray(T, dtype=float)
    SF = np.asarray(SF, dtype=float)
    return (cfg.ts_base
            + cfg.kappa_T * np.power(np.maximum(T - cfg.T_min, 0.0), cfg.xi_T)
            + cfg.kappa_S * np.power(SF, cfg.xi_S))


def phase_fraction(T, t, cfg: RegolithConfig):
    """Phase-transformation fraction Phi(T, t) in [0, 1].

    Logistic onset about the solid->liquid transition temperature T_phase[0]
    multiplied by a saturating dwell-time factor.  Used for constraint g3 and
    for reporting the measured phase change (~16.7 % at the mare optimum).
    """
    T = np.asarray(T, dtype=float)
    t = np.asarray(t, dtype=float)
    onset = 1.0 / (1.0 + np.exp(-(T - cfg.T_phase[0]) / PHASE_T_SCALE))
    dwell = 1.0 - np.exp(-t / PHASE_TIME_TAU)
    return np.clip(onset * dwell, 0.0, 1.0)


def energy_efficiency(T, t, SF, cfg: RegolithConfig):
    """Energy-efficiency ratio eta (MPa.g/kJ) -- report Eq. 45.

    eta = UCS / [ S * t * 60 * S_const / (1000 * rho * d) ]
    """
    strength = ucs(T, t, SF, cfg)
    t = np.asarray(t, dtype=float)
    SF = np.asarray(SF, dtype=float)
    energy_input = (SF * t * 60.0 * SOLAR_CONSTANT) / (
        1000.0 * REGOLITH_DENSITY * SINTER_DEPTH
    )
    energy_input = np.maximum(energy_input, 1.0e-9)
    return strength / energy_input


# --------------------------------------------------------------------------- #
#  Bundled evaluation                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class SinterResponse:
    """Container for the full thermophysical response of a parameter vector."""
    T: float
    t: float
    SF: float
    porosity: float
    ucs: float
    thermal_stress: float
    phase_fraction: float
    energy_efficiency: float


def evaluate(T, t, SF, cfg: RegolithConfig) -> SinterResponse:
    """Evaluate all surrogate responses for a single parameter vector."""
    return SinterResponse(
        T=float(T), t=float(t), SF=float(SF),
        porosity=float(porosity(T, t, SF, cfg)),
        ucs=float(ucs(T, t, SF, cfg)),
        thermal_stress=float(thermal_stress(T, t, SF, cfg)),
        phase_fraction=float(phase_fraction(T, t, cfg)),
        energy_efficiency=float(energy_efficiency(T, t, SF, cfg)),
    )


# --------------------------------------------------------------------------- #
#  Latin Hypercube Sampling                  (report sec. 3.3.2, Eq. 11)       #
# --------------------------------------------------------------------------- #
def latin_hypercube(n_samples: int, bounds: np.ndarray,
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Latin Hypercube Sampling over the supplied [[lo, hi], ...] bounds.

    Implements report Eq. 11:
        x_ij = LB_j + (pi(i, j) + u_ij) / N * (UB_j - LB_j)
    where pi(., j) is an independent random permutation per dimension.
    """
    rng = np.random.default_rng() if rng is None else rng
    d = bounds.shape[0]
    samples = np.empty((n_samples, d))
    for j in range(d):
        perm = rng.permutation(n_samples)        # pi(i, j)
        u = rng.random(n_samples)                # u_ij
        frac = (perm + u) / n_samples
        samples[:, j] = bounds[j, 0] + frac * (bounds[j, 1] - bounds[j, 0])
    return samples


def coverage_efficiency(samples: np.ndarray, bounds: np.ndarray,
                        bins: Optional[int] = None) -> float:
    """Uniformity / coverage metric: fraction of 1-D strata that are occupied.

    Reported in the project as ~92 % for the LHS initialisation (sec. 4.3.4).
    """
    n, d = samples.shape
    bins = n if bins is None else bins
    occupied = 0
    for j in range(d):
        norm = (samples[:, j] - bounds[j, 0]) / (bounds[j, 1] - bounds[j, 0])
        idx = np.clip((norm * bins).astype(int), 0, bins - 1)
        occupied += len(np.unique(idx))
    return occupied / (bins * d)


# --------------------------------------------------------------------------- #
#  Validation-dataset generation                                              #
# --------------------------------------------------------------------------- #
def generate_dataset(cfg: RegolithConfig, n_samples: int = 150,
                     rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
    """Generate a validation dataset for one regolith type."""
    rng = np.random.default_rng() if rng is None else rng
    samples = latin_hypercube(n_samples, cfg.bounds_array(), rng)
    T, t, SF = samples[:, 0], samples[:, 1], samples[:, 2]

    df = pd.DataFrame({
        "regolith": cfg.name,
        "temperature": T,
        "time": t,
        "solar_flux": SF,
        "porosity": porosity(T, t, SF, cfg),
        "ucs": ucs(T, t, SF, cfg),
        "thermal_stress": thermal_stress(T, t, SF, cfg),
        "phase_fraction": phase_fraction(T, t, cfg),
        "energy_efficiency": energy_efficiency(T, t, SF, cfg),
    })
    # composition-weighted thermal conductivity included for completeness
    df["thermal_conductivity"] = (
        cfg.k_solid * np.power(1.0 - df["porosity"], 1.7)
        + 4.0 * 5.67e-8 * cfg.emissivity * cfg.pore_dim
        * np.power(T + 273.15, 3)
    )
    return df


def generate_full_dataset(n_per_type: int = 150,
                          seed: int = 42) -> pd.DataFrame:
    """Generate the complete 300-point validation dataset (report sec. 3.9)."""
    rng = np.random.default_rng(seed)
    frames = [generate_dataset(cfg, n_per_type, rng)
              for cfg in REGOLITH_CONFIGS.values()]
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    data = generate_full_dataset()
    print(f"Generated {len(data)} validation samples "
          f"({data['regolith'].value_counts().to_dict()})")
    for name, cfg in REGOLITH_CONFIGS.items():
        sub = data[data["regolith"] == name]
        feasible = sub[(sub["ucs"] >= 10.0) & (sub["thermal_stress"] <= 10.0)]
        print(f"  {name:8s}: feasible = {len(feasible):3d}/{len(sub)} "
              f"({100 * len(feasible) / len(sub):.1f} %), "
              f"min TS @ UCS>=10 = "
              f"{sub.loc[sub['ucs'] >= 10.0, 'thermal_stress'].min():.2f} MPa")
    data.to_csv("results/validation_dataset.csv", index=False)
    print("Saved -> results/validation_dataset.csv")
