# Multi-Objective Genetic Algorithm Framework for Optimisation of Lunar Regolith Solar-Sintering Parameters

A reconstruction of the Python research code accompanying the MEng final-year
project of the same title (University of Bath). The framework uses a
multi-objective genetic algorithm (built on [DEAP](https://deap.readthedocs.io))
to optimise the three controllable parameters of *in-situ* solar sintering of
lunar regolith — sintering temperature `T`, dwell time `t` and solar
concentration factor `SF` — against competing objectives of mechanical strength,
thermal-stress integrity and energy efficiency, subject to manufacturing
constraints. Both **mare** (iron-rich) and **highland** (anorthositic) regolith
compositions are treated.

> **Provenance & scope.** This is a **reconstruction**, not the original
> project code. The original repository was lost when the author's university
> account was deactivated after graduation, and this codebase has been rebuilt
> from the project report. It is deliberately **smaller and more consolidated**
> than the original submission: the original project was larger, with a more
> extensive directory structure and considerably more code than the five
> modules presented here. This version reproduces the core framework, governing
> equations, and headline results of the report in a compact, runnable form;
> it should not be read as the full extent of the original work.
>
> Where the report specifies an equation *form* but not every numerical
> coefficient, the coefficients have been **calibrated** to published
> regolith-sintering ranges and to the report's own Chapter 4 results.
> Calibrated values are flagged `# calibrated` in `config.py`. The generated
> figures reproduce the *format* and qualitative behaviour of the report's
> figures from the reconstructed models; they are not data-identical to the
> originals. See *Calibration & fidelity* below.

---

## Repository structure

```
lunar-regolith-ga/
├── config.py            # composition configs, bounds, constants, GA settings
├── synthetic_data.py    # efficient surrogate models + LHS validation dataset
├── ga_optimizer.py      # the full GA framework (5 modules) on DEAP
├── visualization.py     # report figure set (convergence, Pareto, windows, …)
├── main.py              # CLI runner: optimise → summarise → plot
├── requirements.txt
├── tests/
│   └── test_framework.py
└── results/             # generated: dataset, JSON summary, figures/
```

### The five modules (report Fig. 1)

All five live in `ga_optimizer.py`:

1. **Population Manager** — `PopulationManager`: Latin-hypercube initialisation
   with minimum-separation enforcement and the `10·d·√d` sizing heuristic.
2. **Fitness Evaluator** — `FitnessEvaluator`: objective normalisation, weighted
   aggregation, exterior penalty with generational escalation, constraint
   accounting, and two-tier response caching.
3. **Evolutionary Engine** — `EvolutionaryEngine`: the specialised genetic
   operators (blended crossover, temperature-aware Gaussian mutation with
   phase-transition damping, polynomial-bounded time mutation, uniform flux
   mutation) and adaptive operator probabilities.
4. **Pareto Analyser** — `ParetoAnalyser`: dominance testing, non-dominated
   sorting, Monte-Carlo hypervolume and generational distance.
5. **Computational Controller** — `ComputationalController`: parallel fitness
   mapping with serial fallback and SHA-256 checkpointing.

The driver class `LunarSinteringGA` wires the modules together; `optimise()` is
the one-line entry point.

---

## Quick start

```bash
# Clone the repository
git clone [https://github.com/CyYippp/lunar_regolith_solar_sintering_optimization.git]
cd lunar_regolith_solar_sintering_optimization

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install the required dependencies
pip install -r requirements.txt        # numpy scipy pandas matplotlib seaborn deap

# Run the optimization script
python main.py                         # optimise mare + highland, write figures
python main.py --regolith mare         # one composition only
python main.py --generations 60 --seed 7
python main.py --no-figures            # results only, no plotting

python synthetic_data.py               # regenerate the 300-point dataset alone
python visualization.py                # dataset/model figures without a GA run
pytest -q                              # regression tests
```

Outputs land under `results/`: `validation_dataset.csv`,
`optimisation_summary.json`, and `figures/*.png`.

---

## Equation → code map

| Report | Concept | Location |
|---|---|---|
| Eq. 11 | Latin-hypercube initialisation | `synthetic_data.latin_hypercube`, `PopulationManager.initialise` |
| Eq. 12–13 | Population sizing & min-separation | `PopulationManager.recommended_size`, `.initialise` |
| Eq. 14–16 | Objective normalisation | `FitnessEvaluator.normalised_objectives` (`_normalise_ucs/_ts/_eff`) |
| Eq. 17 | Weighted-sum aggregation | `FitnessEvaluator.fitness` |
| Eq. 18–19 | Exterior penalty + escalation | `FitnessEvaluator.fitness` (penalty term) |
| Eq. 8–9 | Inequality constraints | `FitnessEvaluator.constraint_violations` |
| Eq. 20–22 | Diversity tournament, elitism, niching | `EvolutionaryEngine.tournament_select`, `LunarSinteringGA.run` |
| Eq. 23–24 | Blended / temperature crossover | `EvolutionaryEngine.crossover` |
| Eq. 25–27 | Temp. Gaussian mutation + phase damping | `EvolutionaryEngine.mutate_temperature` |
| Eq. 28–30 | Polynomial-bounded time mutation | `EvolutionaryEngine.mutate_time` |
| Eq. 31–32 | Uniform flux mutation | `EvolutionaryEngine.mutate_flux` |
| Eq. 33–34 | Adaptive operator probabilities | `EvolutionaryEngine.crossover_prob`, `.mutation_prob` |
| Eq. 38–40 | Parallel mapping + checkpoints | `ComputationalController` |
| Eq. 41 | Effective-medium + radiative conductivity | `ThermophysicalModel` (comprehensive) / `synthetic_data` (surrogate) |
| Eq. 42 | Porosity evolution | `synthetic_data.porosity`, `ThermophysicalModel` |
| Eq. 43 | UCS–porosity relationship | `synthetic_data.ucs` |
| Eq. 44 | Thermal stress | `synthetic_data.thermal_stress` |
| Eq. 45 | Energy efficiency (`S=1361 W/m²`) | `synthetic_data.energy_efficiency` |
| Eq. 46–49 | Convergence: early-stop, dominance, HV, GD | `ParetoAnalyser`, `LunarSinteringGA.run` |

Two model tiers exist by design (report §3): the **comprehensive** models in
`ga_optimizer.ThermophysicalModel` drive the optimisation, while the
**efficient surrogates** in `synthetic_data.py` generate the validation dataset.
They are calibrated to agree at the reference operating points (a regression
test pins this).

---

## Calibration & fidelity

The report gives closed-form model *structures* but not every coefficient. The
coefficients in `config.py` were fitted so that the models reproduce the
report's quantitative findings:

* **Mare optimum** `T = 1071 °C, t = 43 min, SF = 2.1` → UCS ≈ 10.0 MPa,
  thermal stress ≈ 9.0 MPa, phase fraction ≈ 16.7 % (reproduced to 2 d.p.).
* **Mare feasible fraction** ≈ 3–4 % of the design space (report: ~4 %).
* **Highland is infeasible** — 0 % of the space meets both targets; the minimum
  achievable thermal stress at UCS ≥ 10 MPa is ≈ 10.4 MPa (report: 10.4 MPa).
* **Composition penalty** — at equal UCS, highland carries ~1.9–2.5 MPa more
  thermal stress and needs ~50–80 °C higher temperature.

**Densification gating `g(T)`.** One deliberate, physically-motivated addition:
the time/flux densification terms of the porosity model are gated by a
temperature-dependent activation factor
`g(T) = clip((T − T_onset)/(T_full − T_onset), 0, 1)`. This mirrors the
temperature-dependent rate constant in the cited sintering kinetics and prevents
the unphysical artefact of long, low-temperature dwells densifying the compact —
which would otherwise make highland spuriously feasible. Onset/saturation
temperatures are per-composition (`config.py`).

### Known discrepancy — fitness scale

The report's convergence figures show the weighted-sum fitness climbing from
~0.40 to ~0.90. With the normalisation bounds as written in the report
(`UCS, TS ∈ [5, 20] MPa`, Eqs 14–16), the **absolute** fitness at the feasible
optimum sits lower than the reported peak, and the best-fitness curve is
already high at initialisation because the saturating normalisation rewards the
high-strength individuals present in the LHS seed. This code implements the
equations **exactly as documented** (the defensible choice) rather than
back-fitting the normalisation to match the figure's axis. The *physical*
results — optimal parameters, UCS/TS values, Pareto fronts, feasibility
fractions, the highland-infeasibility conclusion — match the report regardless;
only the fitness *axis scaling* differs. The report's stated ordering of
highland fitness above mare fitness is likewise not cleanly recoverable from the
stated equations and appears to be an internal inconsistency in the report.

The convergence figures (`fig_convergence_*`) follow the report's two-panel
layout — objective values (UCS, thermal stress) against the 10 MPa targets on
top, fitness below — and track the **best-so-far target-region solution** each
generation. Because the Latin-hypercube seed already lands inside the small
feasible region, the reconstructed GA converges within a generation or two, so
these curves are flatter than the report's gradual climb. This reflects the
genuine behaviour of the reconstructed search rather than a plotting choice; it
has not been slowed artificially to mimic the report's trajectory.

---

## Notes

* **Reproducibility** — every run takes `--seed`; the GA, LHS and Monte-Carlo
  hypervolume all derive from it.
* **Parallelism** — fitness evaluation runs **serially by default** for
  portability. Opt into multiprocessing with `python main.py --workers 4`;
  if a worker pool cannot be created (e.g. Windows page-file limits) the run
  falls back to serial automatically rather than failing.
* **Stochasticity** — the recommended operating point is selected from the
  converged Pareto set as the feasible solution that meets the UCS target with
  minimum thermal stress; its exact `(T, t, SF)` triple varies slightly between
  seeds, but always lands in the reported low-temperature feasible window for
  mare and reports infeasibility for highland.

## Licence

Released for academic and personal use. No warranty.
