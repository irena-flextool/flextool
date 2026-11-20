"""The scaling guard on Benders region subproblems (``_benders._apply_region_autoscale``).

* A region whose limits span 0.5 … 1e18 (hand-derived at HiGHS' default tolerance 1e-7: 18.3 decades,
  more than the 17 of the window [1e-6, 1e11]) stops with :class:`ScalingRangeError` naming the solve
  and the region — also when Layer 3's option apply fails, whose ``except`` must not swallow it.
* The three-region trade fixture solves with no guard error: its cross-region half-flows carry no
  1e12 stand-in any more (an inf capacity is no limit, no row), so nothing trips.
"""
from __future__ import annotations

import polars as pl
import pytest
from polar_high import Param, Problem
from polar_high.autoscale import ScalingConfig, ScalingMode

from flextool.engine import _behaviour_changes
from flextool.engine.input._solve_state import ScalingRangeError
from flextool.engine.solve import _benders
from flextool.run_options import RunOptions
from tests._support.load import load_scenario

_REGIONS = ["region_A", "region_B", "region_C"]


@pytest.fixture(autouse=True)
def _fresh_run(monkeypatch):
    _behaviour_changes.start_run(None)
    monkeypatch.delenv("FLEXTOOL_SCALING_GUARD", raising=False)
    yield
    _behaviour_changes.start_run(None)


def _wide_region() -> Problem:
    """min x0 + x1, x ∈ [0, 10]: rows x0 ≤ 1e18 and x1 ≥ 0.5 (bound / RHS pool 0.5 … 1e18)."""
    pb = Problem()
    idx = pl.DataFrame({"i": [0, 1]})
    x = pb.add_var("x", "i", idx, lower=0.0, upper=10.0)
    i0, i1 = pl.DataFrame({"i": [0]}), pl.DataFrame({"i": [1]})
    pb.add_cstr("cap", over=i0, sense="<=", lhs_terms={"x": x},
                rhs_terms={"r": Param(("i",), pl.DataFrame({"i": [0], "value": [1e18]}))})
    pb.add_cstr("need", over=i1, sense=">=", lhs_terms={"x": x},
                rhs_terms={"r": Param(("i",), pl.DataFrame({"i": [1], "value": [0.5]}))})
    pb.set_objective(Param(("i",), pl.DataFrame({"i": [0, 1], "value": [1.0, 1.0]})) * x, sense="min")
    return pb


_BASIC = ScalingConfig(mode=ScalingMode.BASIC)


def test_region_error_names_the_region():
    guard = {"solve_name": "invest_2030", "run_options": RunOptions.from_env({})}
    with pytest.raises(ScalingRangeError) as exc:
        _benders._apply_region_autoscale(_wide_region(), _BASIC, "region_X", guard=guard)
    text = str(exc.value)
    assert "Solve invest_2030, Benders region region_X cannot be scaled" in text
    assert "span 18.3 decades" in text and "(0.5 … 1e+18)" in text


def test_region_error_escapes_a_failing_layer3_apply(monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("apply failed")

    monkeypatch.setattr(_benders, "apply_scaling", broken)
    guard = {"solve_name": "invest_2030", "run_options": RunOptions.from_env({})}
    with pytest.raises(ScalingRangeError, match="Benders region region_X"):
        _benders._apply_region_autoscale(_wide_region(), _BASIC, "region_X", guard=guard)


def test_region_warning_under_warn_once_per_region(coded_log):
    guard = {"solve_name": "invest_2030",
             "run_options": RunOptions.from_env({"FLEXTOOL_SCALING_GUARD": "warn"})}
    for region in ("region_X", "region_X", "region_Y"):
        _benders._apply_region_autoscale(_wide_region(), _BASIC, region, guard=guard)
    printed = coded_log()
    assert printed.count("Benders region region_X cannot be scaled") == 1
    assert printed.count("Benders region region_Y cannot be scaled") == 1


def test_a_region_whose_readout_fails_says_the_guard_could_not_run(monkeypatch, coded_log):
    """As the monolith: one warning per run (two regions here), and both regions go on unscaled."""
    monkeypatch.delenv("FLEXTOOL_AUTOSCALE_STRICT", raising=False)

    def broken(*a, **k):
        raise RuntimeError("readout failed")

    monkeypatch.setattr(_benders, "detect_ranges", broken)
    guard = {"solve_name": "invest_2030", "run_options": RunOptions.from_env({})}
    for region in ("region_X", "region_Y"):
        assert _benders._apply_region_autoscale(_wide_region(), _BASIC, region, guard=guard) is None
    printed = coded_log()
    assert printed.count("Warning: the scaling guard could not run (the range readout failed); the solves "
                         "go on unchecked [W-0547]") == 1
    assert "cannot be scaled" not in printed


def test_direct_call_without_guard_context_runs_no_guard():
    assert _benders._apply_region_autoscale(_wide_region(), _BASIC, "region_X") is None


@pytest.fixture(scope="module")
def ti_data(scenario_run):
    return load_scenario(scenario_run("lh2_three_region_trade_invest", db_fixture="lh2_trade_invest"))


def test_trade_fixture_regions_do_not_trip(ti_data, monkeypatch):
    seen = []
    real = _benders.run_guard

    def spy(*a, **k):
        out = real(*a, **k)
        seen.append(out)
        return out

    monkeypatch.setattr(_benders, "run_guard", spy)
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")
    res = _benders.solve_benders(ti_data, _REGIONS, max_iters=20, tol=1e-4, solve_name="trade",
                                 run_options=RunOptions.from_env({}))
    assert res.converged
    assert sorted(o.region for o in seen) == _REGIONS
    assert all(o.level != "error" for o in seen), [(o.region, o.level) for o in seen]
    for o in seen:
        b = o.verdict.finding("bound_rhs")
        assert b.numbers["b_hi"] < 1e12, (o.region, b.numbers["b_hi"])
