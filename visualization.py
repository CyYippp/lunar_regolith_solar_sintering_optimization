"""
visualization.py
=================

Plotting layer for the lunar-regolith solar-sintering GA framework.  Reproduces
the figure set of the project report (Figs 2-16):

    convergence histories            (Figs 2, 3)        plot_convergence
    convergence comparison           (Fig  4)           plot_convergence_comparison
    parameter sensitivity            (Figs 5, 6)        plot_sensitivity
    Pareto-optimal fronts            (Figs 7, 8)        plot_pareto_front
    parameter correlation matrices   (Figs 9, 10)       plot_correlation_matrix
    process windows                  (Figs 11, 12)      plot_process_window
    parameter combinations           (Figs 13, 14)      plot_parameter_combinations
    parameter influences on UCS      (Figs 15, 16)      plot_ucs_influences

All figures are written as PNG files; nothing is shown interactively so the
module runs unchanged on a head-less machine.

This is module (5) of the framework -- it consumes the ``OptimisationResult``
objects produced by :mod:`ga_optimizer` and the validation dataset produced by
:mod:`synthetic_data`, and performs no optimisation itself.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd

try:
    from scipy.spatial import ConvexHull
    _HAVE_SCIPY = True
except Exception:                                   # pragma: no cover
    _HAVE_SCIPY = False

from config import (RegolithConfig, MARE, HIGHLAND, REGOLITH_CONFIGS,
                    UCS_TARGET, TS_TARGET, UCS_CONSTRAINT_MIN, PHASE_FRACTION_MIN)
import synthetic_data as sd

# --------------------------------------------------------------------------- #
#  House style (matched to the report's figures)                               #
# --------------------------------------------------------------------------- #
NAVY = "#1f3a5f"          # primary data series / "meeting targets"
PINK = "#d98a8a"          # "not meeting targets"
RED = "#d62728"           # thermal-stress target line / trend lines
GREEN = "#2ca02c"         # UCS target line
GREEN_FILL = "#cfe8cf"    # target-region shading
UCS_MARKER = "#11335a"
TS_MARKER = "#d62728"

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.35,
    "font.size": 9,
})

_TITLE = {"mare": "Mare", "highland": "Highland"}


def _save(fig, outdir: str, name: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.savefig(path)
    plt.close(fig)
    return path


def _sub(dataset: pd.DataFrame, cfg_name: str) -> pd.DataFrame:
    return dataset[dataset["regolith"] == cfg_name].copy()


def _reference_point(cfg_name: str) -> Dict[str, float]:
    if cfg_name == "mare":
        return {"T": 1071.0, "t": 43.0, "SF": 2.1}
    return {"T": 1113.0, "t": 58.0, "SF": 2.9}


# --------------------------------------------------------------------------- #
#  Figs 2 / 3 -- convergence histories (two stacked panels)                    #
# --------------------------------------------------------------------------- #
def plot_convergence(result, outdir: str, fname: Optional[str] = None) -> str:
    h = result.history
    gen = np.array(h["generation"])
    name = result.regolith

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 7.2), sharex=True)

    # -- top: objective convergence toward the 10 MPa targets ----------------
    ax1.plot(gen, h["rec_ucs"], "-o", color=UCS_MARKER, ms=4, lw=1.6,
             label="UCS (MPa)")
    ax1.plot(gen, h["rec_ts"], "-s", color=TS_MARKER, ms=4, lw=1.6,
             label="Thermal Stress (MPa)")
    ax1.axhline(UCS_TARGET, color=GREEN, ls="--", lw=1.3,
                label="UCS Target (10.0 MPa)")
    ax1.axhline(TS_TARGET, color=RED, ls="--", lw=1.3,
                label="Stress Target (10.0 MPa)")
    ax1.set_ylabel("Value (MPa)")
    ax1.set_title(f"Optimisation Progress for {_TITLE[name]} Regolith")
    ax1.legend(frameon=True, fontsize=8, loc="upper right")

    # -- bottom: fitness S-curve with the best solution marked ---------------
    fit = np.array(h["rec_fitness"])
    ax2.plot(gen, fit, "-D", color="#3a6ea5", ms=4, lw=1.6,
             label="Fitness Function")
    best_g = int(gen[-1])
    ax2.scatter([best_g], [fit[-1]], s=120, color=RED, zorder=5,
                edgecolor="black", linewidth=0.6,
                label=f"Best Solution (Gen {best_g})")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Fitness Value")
    ax2.set_ylim(0, 1)
    ax2.legend(frameon=True, fontsize=8, loc="lower right")

    fig.tight_layout()
    return _save(fig, outdir, fname or f"fig_convergence_{name}.png")


# --------------------------------------------------------------------------- #
#  Fig 4 -- convergence comparison                                             #
# --------------------------------------------------------------------------- #
def plot_convergence_comparison(results: Dict[str, object], outdir: str,
                                fname: Optional[str] = None) -> str:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 7.2), sharex=True)
    colours = {"mare": "#c1440e", "highland": "#3a6ea5"}
    for name, res in results.items():
        h = res.history
        gen = np.array(h["generation"])
        ax1.plot(gen, h["rec_fitness"], "-o", ms=3, lw=1.6,
                 color=colours.get(name), label=f"{_TITLE[name]} fitness")
        ax2.plot(gen, h["rec_ts"], "-s", ms=3, lw=1.6,
                 color=colours.get(name), label=f"{_TITLE[name]} thermal stress")
    ax1.set_ylabel("Fitness Value")
    ax1.set_ylim(0, 1)
    ax1.set_title("Optimisation Progress Comparison: Mare vs Highland")
    ax1.legend(frameon=True, fontsize=8, loc="lower right")
    ax2.axhline(TS_TARGET, color=RED, ls="--", lw=1.3,
                label="Stress Target (10.0 MPa)")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Thermal Stress (MPa)")
    ax2.legend(frameon=True, fontsize=8, loc="upper right")
    fig.tight_layout()
    return _save(fig, outdir, fname or "fig_convergence_comparison.png")


# --------------------------------------------------------------------------- #
#  Figs 5 / 6 -- parameter sensitivity (2x2 correlation-bar grid)              #
# --------------------------------------------------------------------------- #
def plot_sensitivity(dataset: pd.DataFrame, cfg_name: str, outdir: str,
                     fname: Optional[str] = None) -> str:
    sub = _sub(dataset, cfg_name)
    params = ["temperature", "time", "solar_flux"]
    plabel = {"temperature": "Temperature (\u00b0C)", "time": "Time (min)",
              "solar_flux": "Solar Flux"}
    panels = [
        ("ucs", "Sensitivity: UCS (MPa)", "Positive values increase strength"),
        ("thermal_stress", "Sensitivity: Thermal Stress (MPa)",
         "Negative values are preferable"),
        ("porosity", "Sensitivity: Porosity", "Negative values reduce porosity"),
        ("energy_efficiency", "Sensitivity: Energy Efficiency (MPa/kJ\u00b7g)",
         "Positive values increase efficiency"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    for ax, (resp, title, note) in zip(axes.ravel(), panels):
        corr = {p: float(np.corrcoef(sub[p], sub[resp])[0, 1]) for p in params}
        order = sorted(params, key=lambda p: abs(corr[p]))   # ascending -> top biggest
        vals = [corr[p] for p in order]
        labels = [plabel[p] for p in order]
        y = np.arange(len(order))
        ax.barh(y, vals, color=NAVY, alpha=0.9, height=0.5)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlim(-1.2, 1.2)
        ax.set_xlabel("Sensitivity Coefficient")
        ax.set_title(title)
        for yi, v in zip(y, vals):
            ax.text(v + (0.04 if v >= 0 else -0.04), yi, f"{v:.2f}",
                    va="center", ha="left" if v >= 0 else "right", fontsize=8)
        ax.text(0.97, 0.05, note, transform=ax.transAxes, fontsize=6.5,
                ha="right", va="bottom",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8))
    fig.suptitle(f"Sensitivity Analysis for {_TITLE[cfg_name]} Regolith "
                 f"Solar Sintering", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, outdir, fname or f"fig_sensitivity_{cfg_name}.png")


# --------------------------------------------------------------------------- #
#  Figs 7 / 8 -- Pareto-optimal front (zoomed into the target region)          #
# --------------------------------------------------------------------------- #
def plot_pareto_front(result, dataset: pd.DataFrame, cfg_name: str,
                      outdir: str, fname: Optional[str] = None) -> str:
    sub = _sub(dataset, cfg_name)
    obj = result.pareto_objectives
    ucs, ts = obj[:, 0], obj[:, 1]
    rec = result.recommended

    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    # faint cloud of all parameter combinations
    ax.scatter(sub["ucs"], sub["thermal_stress"], s=12, color="0.6",
               alpha=0.35, edgecolor="none", label="Parameter Combinations")
    # target lines + region
    ax.axhline(TS_TARGET, color=RED, ls="--", lw=1.4,
               label="Thermal Stress Target (10.0 MPa)")
    ax.axvline(UCS_TARGET, color=GREEN, ls="--", lw=1.4,
               label="UCS Target (10.0 MPa)")
    ax.add_patch(plt.Rectangle((UCS_TARGET, 0), 100, TS_TARGET,
                 color=GREEN_FILL, alpha=0.55, zorder=0, label="Target Region"))
    # connected Pareto front within the zoom window
    win = (ucs <= 13) & (ts <= 18)
    fu, ft = ucs[win], ts[win]
    if len(fu) > 1:
        order = np.argsort(fu)
        ax.plot(fu[order], ft[order], "-", color="black", lw=1.4, zorder=3,
                label="Pareto Front")
        ax.scatter(fu, ft, marker="*", s=55, color="black", zorder=4,
                   label="Pareto Optimal Points")
    # recommended / top solution
    ax.scatter([rec["ucs"]], [rec["thermal_stress"]], s=90, color=RED,
               edgecolor="black", linewidth=0.7, zorder=5, label="Top Solution")
    ax.annotate(f"T={rec['T']:.0f}\u00b0C, t={rec['t']:.0f}min, SF={rec['SF']:.1f}",
                xy=(rec["ucs"], rec["thermal_stress"]),
                xytext=(rec["ucs"] + 0.3, rec["thermal_stress"] - 0.8),
                fontsize=7.5, bbox=dict(boxstyle="round", fc="white",
                                        ec="0.6", alpha=0.9))
    ax.set_xlim(5, 13)
    ax.set_ylim(8, 18)
    ax.set_xlabel("Unconfined Compressive Strength (MPa)")
    ax.set_ylabel("Thermal Stress (MPa)")
    ax.set_title(f"Pareto Optimal Front for {_TITLE[cfg_name]} Regolith")
    ax.legend(frameon=True, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    return _save(fig, outdir, fname or f"fig_pareto_{cfg_name}.png")


# --------------------------------------------------------------------------- #
#  Figs 9 / 10 -- parameter correlation matrix (9x9 heat-map)                  #
# --------------------------------------------------------------------------- #
def plot_correlation_matrix(dataset: pd.DataFrame, cfg_name: str, outdir: str,
                            fname: Optional[str] = None) -> str:
    cols = ["temperature", "time", "solar_flux", "porosity", "ucs",
            "thermal_stress", "phase_fraction", "thermal_conductivity",
            "energy_efficiency"]
    nice = ["Temperature", "Time", "Solar Flux", "Porosity", "UCS MPa",
            "Thermal Stress MPa", "Phase Transformation",
            "Thermal Conductivity W/m\u00b7K", "Energy Efficiency MPa/kJ\u00b7g"]
    sub = _sub(dataset, cfg_name)[cols]
    corr = sub.corr().values
    n = len(cols)

    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels(nice, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(nice, fontsize=7)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.5,
                    color="white" if abs(corr[i, j]) > 0.55 else "black")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("correlation")
    ax.set_title(f"Parameter Correlation Matrix for {_TITLE[cfg_name]} Regolith")
    fig.tight_layout()
    return _save(fig, outdir, fname or f"fig_correlation_{cfg_name}.png")


# --------------------------------------------------------------------------- #
#  Figs 11 / 12 -- process window (T-t scatter with feasible hull)             #
# --------------------------------------------------------------------------- #
def plot_process_window(dataset: pd.DataFrame, cfg_name: str, outdir: str,
                        fname: Optional[str] = None) -> str:
    sub = _sub(dataset, cfg_name)
    cfg = REGOLITH_CONFIGS[cfg_name]
    feas = (sub["ucs"] >= UCS_TARGET) & (sub["thermal_stress"] <= TS_TARGET)
    f, nf = sub[feas], sub[~feas]
    rate = 100.0 * feas.mean()

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ax.scatter(nf["temperature"], nf["time"], s=26, color=PINK, alpha=0.7,
               edgecolor="white", linewidth=0.3,
               label="Parameters not meeting targets")
    if len(f):
        ax.scatter(f["temperature"], f["time"], s=42, color=NAVY,
                   edgecolor="white", linewidth=0.4, zorder=4,
                   label="Parameters meeting targets")
        pts = f[["temperature", "time"]].values
        if _HAVE_SCIPY and len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                poly = Polygon(pts[hull.vertices], closed=True,
                               facecolor=GREEN_FILL, edgecolor=GREEN,
                               lw=1.8, alpha=0.6, zorder=3, label="Process Window")
                ax.add_patch(poly)
            except Exception:
                pass
    ax.axvline(cfg.T_min, color=RED, ls="--", lw=1.3,
               label=f"Minimum Temperature Threshold ({cfg.T_min:.0f}\u00b0C)")
    ax.text(0.02, 0.97, f"Success rate: {rate:.1f}%", transform=ax.transAxes,
            ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
    ax.set_xlabel("Temperature (\u00b0C)")
    ax.set_ylabel("Time (min)")
    ax.set_title(f"Process Window for {_TITLE[cfg_name]} Regolith Solar Sintering")
    ax.legend(frameon=True, fontsize=8, loc="upper right")
    fig.tight_layout()
    return _save(fig, outdir, fname or f"fig_process_window_{cfg_name}.png")


# --------------------------------------------------------------------------- #
#  Figs 13 / 14 -- parameter combinations (UCS-TS coloured by temperature)     #
# --------------------------------------------------------------------------- #
def plot_parameter_combinations(dataset: pd.DataFrame, cfg_name: str,
                                outdir: str, fname: Optional[str] = None) -> str:
    sub = _sub(dataset, cfg_name)
    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    ax.add_patch(plt.Rectangle((UCS_TARGET, 0), 100, TS_TARGET,
                 color=GREEN_FILL, alpha=0.55, zorder=0, label="Target Region"))
    ax.axhline(TS_TARGET, color=RED, ls="--", lw=1.4,
               label="Thermal Stress Target (10.0 MPa)")
    ax.axvline(UCS_TARGET, color=GREEN, ls="--", lw=1.4,
               label="UCS Target (10.0 MPa)")
    sizes = 20 + 40 * (sub["solar_flux"] - sub["solar_flux"].min()) / \
        max(1e-9, (sub["solar_flux"].max() - sub["solar_flux"].min()))
    sc = ax.scatter(sub["ucs"], sub["thermal_stress"], c=sub["temperature"],
                    s=sizes, cmap="viridis", alpha=0.8, edgecolor="white",
                    linewidth=0.3, zorder=2)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Temperature (\u00b0C)")
    ax.set_xlim(5, 12.5)
    ax.set_ylim(8, 17)
    ax.set_xlabel("Unconfined Compressive Strength (MPa)")
    ax.set_ylabel("Thermal Stress (MPa)")
    ax.set_title(f"Parameter Combinations for {_TITLE[cfg_name]} "
                 f"Regolith Solar Sintering")
    ax.legend(frameon=True, fontsize=8, loc="upper left")
    fig.tight_layout()
    return _save(fig, outdir, fname or f"fig_param_combinations_{cfg_name}.png")


# --------------------------------------------------------------------------- #
#  Figs 15 / 16 -- parameter influences on UCS (3 stacked scatter panels)      #
# --------------------------------------------------------------------------- #
def plot_ucs_influences(dataset: pd.DataFrame, cfg_name: str, outdir: str,
                        fname: Optional[str] = None) -> str:
    # Restrict to the near-target band (the report zooms its UCS axis to the
    # region around the 10 MPa target rather than the full attainable range).
    full = _sub(dataset, cfg_name)
    sub = full[(full["ucs"] >= 2) & (full["ucs"] <= 14)]
    if len(sub) < 10:                       # safety: keep enough points
        sub = full
    panels = [
        ("temperature", "Temperature (\u00b0C)", "time", "Time (min)",
         "Temperature Influence on UCS", "plasma"),
        ("time", "Time (min)", "temperature", "Temperature (\u00b0C)",
         "Time Influence on UCS", "viridis"),
        ("solar_flux", "Solar Flux", "porosity", "Porosity",
         "Solar Flux Influence on UCS", "cividis"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 11.0))
    for ax, (xcol, xlab, ccol, clab, title, cmap) in zip(axes, panels):
        x = sub[xcol].values
        y = sub["ucs"].values
        ax.axhspan(UCS_TARGET, 14, color=GREEN_FILL, alpha=0.5, zorder=0)
        sc = ax.scatter(x, y, c=sub[ccol], s=34, cmap=cmap, alpha=0.85,
                        edgecolor="white", linewidth=0.3, zorder=2)
        coef = np.polyfit(x, y, 2)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, np.polyval(coef, xs), color=RED, lw=2.0, zorder=3,
                label="Trend Line")
        ax.axhline(UCS_TARGET, color="black", ls="--", lw=1.1,
                   label="Target: 10.0 MPa")
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(clab)
        ax.set_ylim(2, 14)
        ax.set_xlabel(xlab)
        ax.set_ylabel("UCS (MPa)")
        ax.set_title(title)
        ax.legend(frameon=True, fontsize=7.5, loc="lower right")
    fig.suptitle(f"Parameter Influences on UCS for {_TITLE[cfg_name]} Regolith",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return _save(fig, outdir, fname or f"fig_ucs_influence_{cfg_name}.png")


# --------------------------------------------------------------------------- #
#  Orchestration                                                               #
# --------------------------------------------------------------------------- #
def generate_all(results: Dict[str, object], dataset: pd.DataFrame,
                 outdir: str) -> Sequence[str]:
    paths = []
    for name, res in results.items():
        paths.append(plot_convergence(res, outdir))
        paths.append(plot_pareto_front(res, dataset, name, outdir))
    if len(results) > 1:
        paths.append(plot_convergence_comparison(results, outdir))
    for name in dataset["regolith"].unique():
        paths.append(plot_sensitivity(dataset, name, outdir))
        paths.append(plot_correlation_matrix(dataset, name, outdir))
        paths.append(plot_process_window(dataset, name, outdir))
        paths.append(plot_parameter_combinations(dataset, name, outdir))
        paths.append(plot_ucs_influences(dataset, name, outdir))
    return paths


if __name__ == "__main__":
    # Stand-alone smoke test: dataset-only figures (no GA run required).
    df = sd.generate_full_dataset(n_per_type=150, seed=7)
    out = "results/figures"
    for nm in ("mare", "highland"):
        print(plot_sensitivity(df, nm, out))
        print(plot_correlation_matrix(df, nm, out))
        print(plot_process_window(df, nm, out))
        print(plot_parameter_combinations(df, nm, out))
        print(plot_ucs_influences(df, nm, out))
    print("dataset figures written to", out)
