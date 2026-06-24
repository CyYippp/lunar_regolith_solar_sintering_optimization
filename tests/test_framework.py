"""
tests/test_framework.py
=======================

Regression tests pinning the physical behaviour of the framework to the
quantitative findings of the project report.  Run with::

    pytest -q

The tests deliberately assert on physics outcomes (feasibility fractions,
reference operating-point responses, monotonic trends) rather than on exact
floating-point values, so they stay robust to minor coefficient retuning while
still catching regressions in the model structure.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import synthetic_data as sd                       # noqa: E402
import ga_optimizer as gao                         # noqa: E402
from config import (MARE, HIGHLAND, UCS_TARGET, TS_TARGET,                # noqa: E402
                    REGOLITH_CONFIGS)


# --------------------------------------------------------------------------- #
#  Surrogate model -- reference operating points                               #
# --------------------------------------------------------------------------- #
def test_mare_reference_point():
    """Report mare optimum: T=1071, t=43, SF=2.1 -> UCS~10, TS~9, phase~0.167."""
    r = sd.evaluate(1071, 43, 2.1, MARE)
    assert r.ucs == pytest.approx(10.0, abs=0.4)
    assert r.thermal_stress == pytest.approx(9.0, abs=0.4)
    assert r.phase_fraction == pytest.approx(0.167, abs=0.03)


def test_highland_reference_point():
    """Report highland near-optimum reaches UCS target but not the TS target."""
    r = sd.evaluate(1113, 58, 2.9, HIGHLAND)
    assert r.ucs == pytest.approx(10.0, abs=0.6)
    assert r.thermal_stress > TS_TARGET          # highland cannot satisfy TS


# --------------------------------------------------------------------------- #
#  Feasibility landscape                                                        #
# --------------------------------------------------------------------------- #
def test_feasibility_fractions():
    """Mare has a small (~4 %) feasible region; highland has none."""
    data = sd.generate_full_dataset(n_per_type=150, seed=7)
    for name, lo, hi in [("mare", 0.01, 0.10), ("highland", 0.0, 0.0)]:
        sub = data[data["regolith"] == name]
        feas = ((sub["ucs"] >= UCS_TARGET) &
                (sub["thermal_stress"] <= TS_TARGET)).mean()
        assert lo <= feas <= hi, f"{name} feasible fraction {feas:.3f}"


def test_highland_min_ts_at_target():
    """Highland minimum thermal stress at UCS>=10 sits near the reported 10.4 MPa."""
    data = sd.generate_full_dataset(n_per_type=300, seed=3)
    sub = data[data["regolith"] == "highland"]
    min_ts = sub.loc[sub["ucs"] >= UCS_TARGET, "thermal_stress"].min()
    assert min_ts == pytest.approx(10.4, abs=0.6)


# --------------------------------------------------------------------------- #
#  Monotonic / physical trends                                                  #
# --------------------------------------------------------------------------- #
def test_ucs_increases_with_temperature():
    lo = sd.ucs(1050, 40, 2.0, MARE)
    hi = sd.ucs(1150, 40, 2.0, MARE)
    assert hi > lo


def test_porosity_decreases_with_temperature():
    lo = sd.porosity(1050, 40, 2.0, MARE)
    hi = sd.porosity(1200, 40, 2.0, MARE)
    assert hi < lo


def test_thermal_stress_increases_with_temperature():
    lo = sd.thermal_stress(1050, 40, 2.0, MARE)
    hi = sd.thermal_stress(1250, 40, 2.0, MARE)
    assert hi > lo


def test_highland_harder_than_mare():
    """At equal UCS (~10 MPa) highland shows markedly higher thermal stress
    than mare -- the report cites a 1.9-2.5 MPa penalty (Sec. 4.2)."""
    pm = sd.evaluate(1071, 43, 2.1, MARE)       # mare optimum, UCS~10
    ph = sd.evaluate(1113, 58, 2.9, HIGHLAND)   # highland near-optimum, UCS~10
    assert pm.ucs == pytest.approx(ph.ucs, abs=0.6)   # matched strength
    assert ph.thermal_stress > pm.thermal_stress
    assert 1.4 <= (ph.thermal_stress - pm.thermal_stress) <= 3.0


# --------------------------------------------------------------------------- #
#  Latin hypercube sampling                                                     #
# --------------------------------------------------------------------------- #
def test_lhs_within_bounds_and_coverage():
    bounds = MARE.bounds_array()
    rng = np.random.default_rng(0)
    samples = sd.latin_hypercube(50, bounds, rng)
    assert samples.shape == (50, 3)
    assert np.all(samples >= bounds[:, 0]) and np.all(samples <= bounds[:, 1])
    cov = sd.coverage_efficiency(samples, bounds)
    assert cov > 0.8                              # report cites ~92 %


# --------------------------------------------------------------------------- #
#  Comprehensive model parity with surrogate                                    #
# --------------------------------------------------------------------------- #
def test_comprehensive_matches_surrogate_at_reference():
    model = gao.ThermophysicalModel(MARE)
    comp = model.evaluate(1071, 43, 2.1)
    surr = sd.evaluate(1071, 43, 2.1, MARE)
    assert comp["ucs"] == pytest.approx(surr.ucs, abs=0.5)
    assert comp["thermal_stress"] == pytest.approx(surr.thermal_stress, abs=0.5)


# --------------------------------------------------------------------------- #
#  GA end-to-end (short budget)                                                 #
# --------------------------------------------------------------------------- #
def test_ga_mare_finds_feasible_region():
    res = gao.optimise("mare", seed=7, n_generations=20)
    assert res.feasible_fraction > 0.0
    assert res.recommended["meets_targets"] is True
    # recommendation sits in the reported low-temperature window
    assert 1020 <= res.recommended["T"] <= 1120
    assert res.recommended["ucs"] >= UCS_TARGET - 0.5


def test_ga_highland_infeasible():
    res = gao.optimise("highland", seed=7, n_generations=20)
    assert res.feasible_fraction == 0.0
    assert res.recommended["meets_targets"] is False


def test_pareto_set_nonempty_and_nondominated():
    res = gao.optimise("mare", seed=11, n_generations=15)
    obj = res.pareto_objectives
    assert len(obj) > 0
    # verify mutual non-dominance on (UCS up, TS down) projection
    analyser = gao.ParetoAnalyser()
    nd = analyser.non_dominated(obj)
    assert len(nd) == len(obj)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
