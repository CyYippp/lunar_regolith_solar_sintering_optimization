"""
main.py
=======

Command-line entry point for the lunar-regolith solar-sintering GA framework.

Runs the multi-objective optimisation for the mare and/or highland regolith
types, writes a machine-readable summary and the validation dataset, and
renders the full report figure set.

Examples
--------
    python main.py                      # both regolith types, full run
    python main.py --regolith mare      # mare only
    python main.py --generations 60     # override generation budget
    python main.py --seed 7 --quiet     # reproducible, terse
    python main.py --no-figures         # skip plotting (results only)

Outputs (under ``results/``)
----------------------------
    validation_dataset.csv      300-point LHS validation set
    optimisation_summary.json   per-regolith optima, feasibility, metrics
    figures/*.png               convergence, Pareto, process windows, etc.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np

from config import REGOLITH_CONFIGS, UCS_TARGET, TS_TARGET
import synthetic_data as sd
import ga_optimizer as gao
import visualization as viz

RESULTS_DIR = "results"
FIG_DIR = os.path.join(RESULTS_DIR, "figures")


def _summarise(res) -> Dict:
    rec = res.recommended
    return {
        "regolith": res.regolith,
        "best_fitness": round(res.best_fitness, 4),
        "recommended": {
            "T_degC": round(rec["T"], 1),
            "t_min": round(rec["t"], 1),
            "SF": round(rec["SF"], 2),
            "ucs_MPa": round(rec["ucs"], 2),
            "thermal_stress_MPa": round(rec["thermal_stress"], 2),
            "phase_fraction": round(rec["phase_fraction"], 3),
            "energy_efficiency": round(rec["energy_efficiency"], 4),
            "meets_targets": bool(rec["meets_targets"]),
        },
        "feasible_fraction": round(res.feasible_fraction, 4),
        "n_pareto": int(len(res.pareto_individuals)),
        "n_evaluations": int(res.n_evaluations),
        "cache_hits": int(res.cache_hits),
        "processor_utilisation": round(res.processor_utilisation, 3),
        "targets": {"ucs_min_MPa": UCS_TARGET, "ts_max_MPa": TS_TARGET},
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Multi-objective GA optimiser for lunar regolith "
                    "solar-sintering parameters.")
    ap.add_argument("--regolith", choices=["mare", "highland", "both"],
                    default="both", help="which regolith type(s) to optimise")
    ap.add_argument("--generations", type=int, default=None,
                    help="override the per-type generation budget")
    ap.add_argument("--pop-size", type=int, default=None,
                    help="override the population size")
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed for reproducibility")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel fitness-evaluation workers "
                         "(default 1 = serial; >1 opts into multiprocessing)")
    ap.add_argument("--no-figures", action="store_true",
                    help="skip figure generation")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress per-generation progress")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    verbose = not args.quiet

    # ----- validation dataset (report sec. 3.9) --------------------------- #
    print("Generating 300-point validation dataset ...")
    dataset = sd.generate_full_dataset(n_per_type=150, seed=args.seed)
    dataset.to_csv(os.path.join(RESULTS_DIR, "validation_dataset.csv"),
                   index=False)

    # ----- optimisation runs ---------------------------------------------- #
    names = (["mare", "highland"] if args.regolith == "both"
             else [args.regolith])
    results = {}
    summary = {}
    for name in names:
        print(f"\nOptimising {name} regolith ...")
        res = gao.optimise(
            name, seed=args.seed, verbose=verbose,
            n_generations=args.generations, pop_size=args.pop_size,
            n_workers=args.workers)
        results[name] = res
        summary[name] = _summarise(res)
        rec = res.recommended
        print(f"  -> best fitness {res.best_fitness:.3f}; "
              f"recommended T={rec['T']:.0f} degC, t={rec['t']:.0f} min, "
              f"SF={rec['SF']:.2f}  "
              f"(UCS={rec['ucs']:.1f}, TS={rec['thermal_stress']:.1f} MPa, "
              f"feasible={rec['meets_targets']})")
        print(f"  -> feasible fraction {100 * res.feasible_fraction:.1f} %, "
              f"{len(res.pareto_individuals)} Pareto solutions")

    with open(os.path.join(RESULTS_DIR, "optimisation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to "
          f"{os.path.join(RESULTS_DIR, 'optimisation_summary.json')}")

    # ----- figures --------------------------------------------------------- #
    if not args.no_figures:
        print("\nRendering figures ...")
        for name, res in results.items():
            viz.plot_convergence(res, FIG_DIR)
            viz.plot_pareto_front(res, dataset, name, FIG_DIR)
        if len(results) > 1:
            viz.plot_convergence_comparison(results, FIG_DIR)
        for name in names:
            viz.plot_sensitivity(dataset, name, FIG_DIR)
            viz.plot_correlation_matrix(dataset, name, FIG_DIR)
            viz.plot_process_window(dataset, name, FIG_DIR)
            viz.plot_parameter_combinations(dataset, name, FIG_DIR)
            viz.plot_ucs_influences(dataset, name, FIG_DIR)
        print(f"Figures written to {FIG_DIR}/")

    print("\nDone.")


if __name__ == "__main__":
    main()
