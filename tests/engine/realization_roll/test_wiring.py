"""The realization roll wired in (design rev 7.1 §2.6, §8.2 T-w1 / T-w3 / T-w4, §8.3 jump = 1): no test seam.

A rolling ``within_solve`` solve is one solve per roll (no realization solve is added); a rolling
``separate_dispatch`` solve is solve (1) then the realization roll, which looks ahead after its jump with the
forecasts made at the jump end.  Models: ``test_split_roll_lp._tw1`` (T-w1) and ``test_trunk_sites._spec``.
Every expected value is derived by hand in the docstrings.
"""
from __future__ import annotations

import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from tests.engine import _solve_copy_models as scm
from tests.engine.realization_roll import test_split_roll_lp as srl
from tests.engine.realization_roll import test_trunk_sites as ts

pack = scm.pack


@pytest.fixture(autouse=True)
def _strict(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _set_solve(spec: dict, name: str, value) -> dict:
    for row in spec["parameter_values"]:
        if row[:3] == ["solve", "st", name]:
            row[3] = value
            return spec
    spec["parameter_values"].append(["solve", "st", name, value, "init"])
    return spec


def _state(step) -> float:
    return srl._value(step.solution.value("v_state"), n="elec", d="p1", t="t01")


@pytest.mark.parametrize(("w_b", "state", "objectives"), [
    (0.5, 1.0, [87_600.0, 0.0]),
    (0.25, 0.0, [76_650.0, 131_400.0])])
def test_within_solve_is_one_solve_per_roll(tmp_path, w_b, state, objectives):
    """T-w1 (jump 1, two rolls): 2 solves, no realization solve (today 4).  Factor 8760 / 2 = 4380 per step.

    R_0 = p1 t01 on R (wind 0) + p1_A / p1_B t02 (forecasts made at t02: A wind 1, B wind 0, weights 1 - w_B, w_B).
    gas covers t01 at 10 and can charge 1 MWh more (2 MW); storing saves 30 at t02 only under B.  w_B 0.5: 15 > 10,
    the realized storage at the end of t01 is 1, cost 20 x 4380 = 87 600; R_1 (t02, deterministic on R) discharges:
    0.  w_B 0.25: 7.5 < 10, storage 0, cost (10 + 0.25 x 30) x 4380 = 76 650; R_1 buys 1 MWh at 30: 131 400.
    No price is asserted (gas is capped: the t01 price lies anywhere in [10, 15])."""
    built, steps = scm.run_spec(srl._tw1(w_b), tmp_path)
    names = [s for s, _, _ in built]
    assert len(names) == 2 and not any("realized_dispatch" in s for s in names), names
    first = list(steps.values())[0]
    assert _state(first) == pytest.approx(state, abs=1e-9)
    assert srl._objectives(steps) == pytest.approx(objectives, rel=1e-9, abs=1e-6)


def test_separate_dispatch_realization_looks_ahead_jump_one(tmp_path):
    """T-w1 as ``separate_dispatch`` (w_B 0.5), jump 1, with forecasts also made at t01 (A wind 0 at t01, 1 at t02;
    B wind 0 at both; weights 0.5 / 0.5) for solve (1) of roll 0.  4 solves: (1) and the realization roll per roll.

    Solve (1) roll 0: both forecasts from t01, storage per forecast (no unit online, so nothing is tied): A buys 1
    MWh at 10; B also stores 1 MWh at 10 for t02 (instead of 30): 0.5 x 10 + 0.5 x 20 = 15 -> 65 700.  Realization
    roll 0: no _fix_in_uc unit, so nothing is pinned: R_0 of T-w1, 87 600, realized storage 1.  Solve (1) roll 1
    (t02, from the realized storage 1): discharge, 0.  Realization roll 1 (t02, deterministic on R): 0."""
    spec = srl._tw1(0.5)
    _set_solve(spec, "dispatch_realization_method", pack("separate_dispatch", "str"))
    _set_solve(spec, "stochastic_branches", srl._fan([
        ("A", {"t01": 0.5, "t02": 0.5}, "no"), ("B", {"t01": 0.5, "t02": 0.5}, "no"),
        ("R", {"t01": 0.0, "t02": 0.0}, "yes")]))
    for row in spec["parameter_values"]:
        if row[:3] == ["profile", "wp", "profile"]:
            row[3] = srl._series({"A": {"t01": {"t01": 0.0, "t02": 1.0}, "t02": {"t02": 1.0}},
                                  "B": {"t01": {"t01": 0.0, "t02": 0.0}, "t02": {"t02": 0.0}},
                                  "R": {"t01": {"t01": 0.0, "t02": 0.0}, "t02": {"t02": 0.0}}})
    built, steps = scm.run_spec(spec, tmp_path)
    names = [s for s, _, _ in built]
    assert len(names) == 4 and ["realized_dispatch" in s for s in names] == [False, True, False, True], names
    assert srl._objectives(steps) == pytest.approx([65_700.0, 87_600.0, 0.0, 0.0], rel=1e-9, abs=1e-6)
    assert _state(list(steps.values())[1]) == pytest.approx(1.0, abs=1e-9)


def test_a_jump_ending_without_forecasts_stops(tmp_path):
    """T-w3: test_trunk_sites' model, jump 1 / horizon 2, forecasts made at t01 and t03 only: roll 0 (t01-t02)
    ends its jump at t02, where no forecast is made."""
    spec = ts._spec(real_at=ts.OVERLAP, forecast_at={"t01": ts.STEPS, "t03": ts.STEPS[2:]}, rolling=(1.0, 2.0))
    with pytest.raises(FlexToolConfigError, match=r"Solve 'st', roll 0 \(time steps t01 to t01\): no forecast of "
                                                  r"stochastic_branches is made at t02, the end of its jump"):
        scm.run_spec(spec, tmp_path)


def test_within_solve_without_a_realized_branch_stops(tmp_path):
    """T-w4 (W3): T-w1 without its ``realized: yes`` row."""
    spec = _set_solve(srl._tw1(0.5), "stochastic_branches",
                      srl._fan([("A", {"t02": 0.5}, "no"), ("B", {"t02": 0.5}, "no")]))
    with pytest.raises(FlexToolConfigError, match=r"Solve 'st' is a rolling stochastic solve with "
                                                  r"dispatch_realization_method within_solve, but its "
                                                  r"stochastic_branches have no branch with realized: yes"):
        scm.run_spec(spec, tmp_path)
