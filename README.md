# Multi-Objective Genetic Algorithm for Lunar Regolith Solar-Sintering

A modular, multi-objective genetic-algorithm framework (built on
[DEAP](https://deap.readthedocs.io)) that optimises the three controllable
parameters of *in-situ* solar sintering of lunar regolith — sintering
temperature `T`, dwell time `t` and solar concentration factor `SF` — against
competing objectives of mechanical strength, thermal-stress integrity and energy
efficiency, subject to manufacturing constraints. Both **mare** (iron-rich) and
**highland** (anorthositic) regolith compositions are treated.

Originally developed as my MEng final-year research project at the University of
Bath; this repository is a clean, runnable rebuild of that framework (see
*Provenance & fidelity*).

## Highlights

* **Five-module GA architecture** on DEAP — population management, fitness
  evaluation, evolutionary operators, Pareto analysis and a parallel controller
  (see below).
* **Problem-specific operators** — blended crossover, temperature-aware Gaussian
  mutation with phase-transition damping, polynomial-bounded time mutation and
  adaptive operator probabilities.
* **Multi-objective machinery** — constrained weighted-sum fitness with an
  escalating exterior penalty, non-dominated sorting, Monte-Carlo hypervolume and
  generational distance.
* **Engineered for use** — reproducible via a single `--seed`, optional
  multiprocessing with automatic serial fallback, SHA-256 checkpointing, two-tier
  response caching, a command-line runner and a passing `pytest` suite.
* **Traceable** — every governing equation in the report is mapped to the
  function that implements it (see *Equation → code map*).

Running `python main.py` reproduces the project's headline findings: a viable
low-temperature process window for mare regolith (~4% of the design space) and
the systematic infeasibility of highland regolith against the same targets.

---

## Quick start

```bash
git clone https://github.com/CyYippp/lunar_regolith_solar_sintering_optimization.git
cd lunar_regolith_solar_sintering_optimization

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # numpy scipy pandas matplotlib seaborn deap

python main.py                     # optimise mare + highland, write figures
python main.py --regolith mare     # one composition only
python main.py --generations 60 --seed 7
python main.py --no-figures        # results only, no plotting
python main.py --workers 4         # opt into parallel fitness evaluation

pytest -q                          # run the regression tests
```

Outputs land under `results/`: `validation_dataset.csv`,
`optimisation_summary.json`, and `figures/*.png`.

---

## Repository structure

```
lunar_regolith_solar_sintering_optimization/
├── config.py            # composition configs, bounds, constants, GA settings
├── synthetic_data.py    # efficient surrogate models + LHS validation dataset
├── ga_optimizer.py      # the full GA framework (5 modules) on DEAP
├── visualization.py     # figure set (convergence, Pareto, process windows, …)
├── main.py              # CLI runner: optimise → summarise → plot
├── requirements.txt
├── tests/
│   └── test_framework.py
└── results/             # generated: dataset, JSON summary, figures/
```

### The five modules

All five live in `ga_optimizer.py`:

1. **Population Manager** — `PopulationManager`: Latin-hypercube initialisation
   with minimum-separation enforcement and the `10·d·√d` sizing heuristic.
2. **Fitness Evaluator** — `FitnessEvaluator`: objective normalisation, weighted
   aggregation, exterior penalty with generational escalation, constraint
   accounting, and two-tier response caching.
3. **Evolutionary Engine** — `EvolutionaryEngine`: the specialised genetic
   operators and adaptive operator probabilities.
4. **Pareto Analyser** — `ParetoAnalyser`: dominance testing, non-dominated
   sorting, Monte-Carlo hypervolume and generational distance.
5. **Computational Controller** — `ComputationalController`: parallel fitness
   mapping with serial fallback and SHA-256 checkpointing.

The driver class `LunarSinteringGA` wires the modules together; `optimise()` is
the one-line entry point.

---

## Equation → code map

| Report | Concept | Location |
|---|---|---|
| Eq. 11 | Latin-hypercube initialisation | `synthetic_data.latin_hypercube`, `PopulationManager.initialise` |
| Eq. 12–13 | Population sizing & min-separation | `PopulationManager.recommended_size`, `.initialise` |
| Eq. 14–16 | Objective normalisation | `FitnessEvaluator.normalised_objectives` |
| Eq. 17 | Weighted-sum aggregation | `FitnessEvaluator.fitness` |
| Eq. 18–19 | Exterior penalty + escalation | `FitnessEvaluator.fitness` |
| Eq. 8–9 | Inequality constraints | `FitnessEvaluator.constraint_violations` |
| Eq. 20–22 | Diversity tournament, elitism, niching | `EvolutionaryEngine.tournament_select`, `LunarSinteringGA.run` |
| Eq. 23–24 | Blended / temperature crossover | `EvolutionaryEngine.crossover` |
| Eq. 25–27 | Temp. Gaussian mutation + phase damping | `EvolutionaryEngine.mutate_temperature` |
| Eq. 28–30 | Polynomial-bounded time mutation | `EvolutionaryEngine.mutate_time` |
| Eq. 31–32 | Uniform flux mutation | `EvolutionaryEngine.mutate_flux` |
| Eq. 33–34 | Adaptive operator probabilities | `EvolutionaryEngine.crossover_prob`, `.mutation_prob` |
| Eq. 38–40 | Parallel mapping + checkpoints | `ComputationalController` |
| Eq. 41 | Effective-medium + radiative conductivity | `ThermophysicalModel` / `synthetic_data` |
| Eq. 42 | Porosity evolution | `synthetic_data.porosity` |
| Eq. 43 | UCS–porosity relationship | `synthetic_data.ucs` |
| Eq. 44 | Thermal stress | `synthetic_data.thermal_stress` |
| Eq. 45 | Energy efficiency (`S=1361 W/m²`) | `synthetic_data.energy_efficiency` |
| Eq. 46–49 | Convergence: early-stop, dominance, HV, GD | `ParetoAnalyser`, `LunarSinteringGA.run` |

Two model tiers exist by design: the **comprehensive** models in
`ga_optimizer.ThermophysicalModel` drive the optimisation, while the **efficient
surrogates** in `synthetic_data.py` generate the validation dataset. They are
calibrated to agree at the reference operating points (pinned by a test).

---

## Provenance & fidelity

This is a clean rebuild of my MEng project framework, reconstructed from the
project report after the original university-hosted repository was lost on
graduation. It is deliberately compact — it reproduces the core framework,
governing equations and headline results in a runnable form, and is not the full
extent of the original submission.

The report gives closed-form model *structures* but not every coefficient, so
the coefficients in `config.py` are calibrated to the literature ranges cited in
the report and to its Chapter 4 results (calibrated values are flagged
`# calibrated`). With these, the framework reproduces the project's physical
findings:

* **Mare optimum** `T ≈ 1071 °C, t ≈ 43 min, SF ≈ 2.1` → UCS ≈ 10.0 MPa,
  thermal stress ≈ 9.0 MPa, phase fraction ≈ 16.7%.
* **Mare feasible fraction** ≈ 3–4% of the design space.
* **Highland infeasible** — no point meets both targets; minimum thermal stress
  at UCS ≥ 10 MPa ≈ 10.4 MPa.
* **Composition penalty** — at equal UCS, highland carries ~1.9–2.5 MPa more
  thermal stress and needs ~50–80 °C higher temperature.

**Densification gating `g(T)`.** One deliberate, physically-motivated addition:
the time/flux densification terms of the porosity model are gated by a
temperature-dependent activation factor
`g(T) = clip((T − T_onset)/(T_full − T_onset), 0, 1)`, mirroring the
temperature-dependent rate constant in the cited sintering kinetics. It prevents
the unphysical artefact of long, low-temperature dwells densifying the compact.

**Fidelity note.** The framework implements the report's equations as documented
rather than back-fitting them to the report's figure axes, so the *physical*
outputs (optimal parameters, UCS/TS values, Pareto fronts, feasibility fractions
and the highland-infeasibility conclusion) match the report, while the absolute
scaling of the weighted-sum *fitness axis* on the convergence plots differs. The
generated figures reproduce the format and qualitative behaviour of the report's
figures from the reconstructed models; they are not data-identical to the
originals.

---

## Notes

* **Reproducibility** — every run takes `--seed`; the GA, LHS and Monte-Carlo
  hypervolume all derive from it.
* **Parallelism** — fitness evaluation runs serially by default for portability;
  `--workers N` opts into multiprocessing, with automatic serial fallback.
* **Recommended operating point** — selected from the converged Pareto set as the
  feasible solution meeting the UCS target at minimum thermal stress; its exact
  `(T, t, SF)` varies slightly between seeds but always lands in the reported
  low-temperature feasible window for mare, and reports infeasibility for
  highland.

## Licence

Released for academic and personal use. No warranty.
