"""Ramp cost: a cost per MW of change in a unit's step-average flow, up and down alike, charged per change.

``ramp_change_eq`` splits the change of a priced flow from its ramp predecessor into ``v_ramp_up`` and
``v_ramp_down``; the objective charges (up + down) × unitsize × ramp_cost with the event weight of a start-up (no
step_duration).  The predecessors are those of the ramp limits: no row at the first step of a solve, after a gap or
at a representative-block start; a continuation roll's first step ramps from the previous roll's realized flow.

Fixtures: ``tests.json`` (coal_plant 500 MW, direct, coal_market -> west; west penalty_up 900 / penalty_down 800
CUR/MWh, so serving the demand is always cheaper than 600 CUR of ramp cost and the coal flow follows the demand)
plus override alternatives, built from JSON under the pytest tmp dir.  The expected values are hand-derived (design
c3_ramp_cost_design.md §7, recomputed independently in its critique).
"""
from __future__ import annotations

import csv

import polars as pl
import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from tests.engine._lp_rows import row_terms, step_of
from tests.engine._unit_db import PLANT, RunCache, b64, build, flow, make_db, run, time_map, utf8

pytestmark = [pytest.mark.solver]

X = 300.0           # MW of demand at west
RC = 2.0            # CUR/MW of change
OUT = "unit__outputNode"
REL = [PLANT, "west"]
ARC = f"{PLANT},coal_market,west,"
TOL = 1e-6
_RUNS = RunCache()


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch) -> None:
    """A missing autoscale registry entry raises instead of falling back to an unscaled LP."""
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _cost(method: str = "ramp_cost", rc: float = RC, cls: str = OUT, rel: list | None = None) -> list:
    rel = rel or REL
    return [[cls, rel, "ramp_method", b64(method, "str")], [cls, rel, "ramp_cost", b64(rc, "float")]]


def _demand(values: dict[str, float]) -> list:
    """West demand ``values`` (MW, positive) at the listed steps, X elsewhere."""
    return [["node", "west", "inflow", time_map({t: -v for t, v in values.items()}, -X)]]


def _solve(tmp_path_factory, alt: str, values: list, chain=("init", "west", "coal"), entities=None) -> dict:
    steps, _, _ = _RUNS.get(tmp_path_factory, alt=alt, values=values, base_chain=list(chain), entities=entities)
    return steps


def _last(steps: dict):
    return list(steps.values())[-1]


def _ramps_dt(tmp_path_factory, alt: str, values: list, chain=("init", "west", "coal"), entities=None
              ) -> tuple[dict, dict[tuple[str, str], float]]:
    """Solve with the CSV outputs; ``(steps, {(period, time): ramps})`` from ``costs__dt.csv``."""
    from flextool.engine import run_chain_from_db
    from flextool.process_outputs.write_outputs import write_outputs

    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=list(chain), entities=entities)
    work = tmp_path_factory.mktemp(f"{alt}_out")
    steps = run_chain_from_db(url, scen, work_folder=work, keep_solutions=True)
    last = _last(steps)
    write_outputs(scenario_name=scen, output_location=str(work), subdir=scen, write_methods=["csv"],
                  fallback_output_location=str(work), raw_output_dir=str(work / "output_raw"),
                  solution=last.solution, solve_name=last.solve_name,
                  solve_steps=[(s.solve_name, s.flex_data, s.effective_solution) for s in steps.values()],
                  flex_data_provider=last.flex_data_provider)
    with open(work / "output_csv" / scen / "costs__dt.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return steps, {(r["period"], r["time"]): float(r["ramps"]) for r in rows}


def _rows(step) -> dict:
    return row_terms(build(step.flex_data), "ramp_change_eq", ARC)


def _realized_ramp_mw(steps: dict) -> dict[tuple[str, str], float]:
    """Realized (d, t) → (v_ramp_up + v_ramp_down) × unitsize of the coal arc over every solve."""
    out: dict = {}
    for s in steps.values():
        real = utf8(s.flex_data.realized_dispatch.rename({"period": "d", "step": "t"})
                    if "period" in s.flex_data.realized_dispatch.columns else s.flex_data.realized_dispatch)
        us = float(utf8(s.flex_data.p_unitsize.frame).filter(pl.col("p") == PLANT)["value"][0])
        for var in ("v_ramp_up", "v_ramp_down"):
            if var not in s.solution._vars:
                continue
            v = (utf8(s.solution.value(var)).filter(pl.col("p") == PLANT)
                 .join(real.select("d", "t"), on=["d", "t"], how="semi"))
            for d, t, x in v.select("d", "t", "value").iter_rows():
                out[(d, t)] = out.get((d, t), 0.0) + x * us
    return out


# ---------------------------------------------------------------------------
# 1. both directions, the first step free


def test_both_directions_charged_first_step_free(tmp_path_factory) -> None:
    """Demand X, X, 0, X (then X) on 1-h steps, ramp_cost 2 CUR/MW on the coal output: the flow follows the demand,
    Δ = 0, −300, +300 at t2–t4, so ramps = 2·300 = 600 at t3 and at t4, 0 elsewhere (also at t1, which has no
    predecessor and no row).  The objective rises by 1200 · inflation_op / period_share over the model without the
    cost."""
    steps, ramps = _ramps_dt(tmp_path_factory, "rc_both", _cost() + _demand({"t0003": 0.0}))
    step = _last(steps)
    assert flow(step, PLANT, "coal_market", "west").to_list()[:4] == pytest.approx([X, X, 0.0, X], abs=TOL)
    assert ramps[("p2020", "t0003")] == pytest.approx(600.0, abs=1e-4)
    assert ramps[("p2020", "t0004")] == pytest.approx(600.0, abs=1e-4)
    assert sum(v for k, v in ramps.items() if k[1] not in ("t0003", "t0004")) == pytest.approx(0.0, abs=1e-4)
    rows = _rows(step)
    assert f"ramp_change_eq[{ARC}p2020,t0001]" not in rows
    assert len(rows) == step.flex_data.dt.height - 1
    without = _last(_solve(tmp_path_factory, "rc_none", _demand({"t0003": 0.0})))
    fd = step.flex_data
    infl = float(utf8(fd.p_inflation_op.frame).filter(pl.col("d") == "p2020")["value"][0])
    share = float(utf8(fd.p_period_share.frame).filter(pl.col("d") == "p2020")["value"][0])
    assert step.solution.obj - without.solution.obj == pytest.approx(1200.0 * infl / share, rel=1e-6)


# ---------------------------------------------------------------------------
# 2. which ramp_method values switch it on


def test_ramp_limit_ignores_the_cost(tmp_path_factory) -> None:
    """ramp_method ramp_limit with a ramp_cost value: limit rows, no ramp_change_eq and no ramp variables."""
    vals = _cost("ramp_limit") + [[OUT, REL, "ramp_speed_up", b64(0.002, "float")],
                                  [OUT, REL, "ramp_speed_down", b64(0.002, "float")]]
    step = _last(_solve(tmp_path_factory, "rc_limit_only", vals))
    pb = build(step.flex_data)
    assert not row_terms(pb, "ramp_change_eq")
    assert row_terms(pb, "ramp_sink_up_constraint", ARC)
    assert "v_ramp_up" not in step.solution._vars


def test_both_builds_limit_and_cost_rows(tmp_path_factory) -> None:
    """ramp_method both with a speed and a cost: the limit rows and one cost row per linked step."""
    vals = _cost("both") + [[OUT, REL, "ramp_speed_up", b64(0.002, "float")],
                            [OUT, REL, "ramp_speed_down", b64(0.002, "float")]]
    step = _last(_solve(tmp_path_factory, "rc_both_method", vals))
    pb = build(step.flex_data)
    assert row_terms(pb, "ramp_sink_up_constraint", ARC)
    cost_rows = row_terms(pb, "ramp_change_eq", ARC)
    assert {step_of(r) for r in cost_rows} == {
        (d, t) for d, t in utf8(step.flex_data.dt_ramp_link).select("d", "t").iter_rows()}


def test_cost_without_speed_and_zero_cost(tmp_path_factory) -> None:
    """ramp_cost without any ramp_speed builds the cost rows (the rows of the first test); a ramp_cost of 0 builds
    none."""
    assert _rows(_last(_solve(tmp_path_factory, "rc_flat", _cost())))
    zero = _last(_solve(tmp_path_factory, "rc_zero", _cost(rc=0.0)))
    assert not row_terms(build(zero.flex_data), "ramp_change_eq")


def test_negative_cost_is_an_error(tmp_path_factory) -> None:
    url, scen = make_db(tmp_path_factory, alt="rc_negative", values=_cost(rc=-1.0),
                        base_chain=["init", "west", "coal"])
    with pytest.raises(FlexToolConfigError, match="ramp_cost is negative"):
        run(tmp_path_factory, url, scen)


# ---------------------------------------------------------------------------
# 3. rolling: a change at a roll boundary is charged once


def test_change_at_roll_boundary_charged_once(tmp_path_factory) -> None:
    """Rolling fullYear dispatch (72-h periods, horizon 36, jump 24), coal alone, no ramp limit: demand X up to
    t0024 and 0 from t0025, the first step of the second roll.  Roll 1 sees the change in its look-ahead (not
    realized); roll 2 ramps from roll 1's realized flow at t0024 (the carry), so the realized ramps of each period
    are 2·300 = 600 once, at t0025, and the carried row has no predecessor column."""
    vals = _cost() + _demand({f"t{i:04d}": 0.0 for i in range(25, 73)}) + [
        ["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", b64(36.0, "float")],
        ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", b64(24.0, "float")]]
    steps, ramps = _ramps_dt(tmp_path_factory, "rc_roll", vals,
                             chain=("init", "west", "coal", "fullYear", "dispatch_fullYear_roll"))
    periods = sorted({d for d, _ in ramps})
    assert periods
    for d in periods:
        assert ramps[(d, "t0025")] == pytest.approx(600.0, abs=1e-4)
        assert sum(v for (dd, _), v in ramps.items() if dd == d) == pytest.approx(600.0, abs=1e-4)
    second = list(steps.values())[1]
    assert second.flex_data.dt_ramp_carry is not None
    row = _rows(second)[f"ramp_change_eq[{ARC}p2020,t0025]"]
    assert set(row) == {f"v_flow[{ARC}p2020,t0025]", f"v_ramp_up[{ARC}p2020,t0025]",
                        f"v_ramp_down[{ARC}p2020,t0025]"}
    realized = _realized_ramp_mw(steps)
    assert realized[("p2020", "t0025")] == pytest.approx(X, abs=1e-4)


# ---------------------------------------------------------------------------
# 4. coarse process: one row per coarse-step boundary, the change reported once


def test_coarse_process_one_row_per_boundary(tmp_path_factory) -> None:
    """west in a 4-h new_stepduration group, so coal_plant has one flow per 4-h coarse step (12 over the 48-h
    2day solve).  Demand X up to t0024 and 0 from t0025, constant inside every coarse step: 11 rows (one per
    boundary, also where Δ = 0), and the reported ramps are rc·X = 600 at t0025, not 4·600 (the change is an
    event at the coarse step's first fine step)."""
    ents = [["group", "four_h", None], ["group__node", ["four_h", "west"], None]]
    vals = _cost() + _demand({f"t{i:04d}": 0.0 for i in range(25, 49)}) + [
        ["group", "four_h", "new_stepduration", b64(4.0, "float")]]
    steps, ramps = _ramps_dt(tmp_path_factory, "rc_coarse", vals, entities=ents)
    step = _last(steps)
    assert step.flex_data.processCoarse is not None
    assert PLANT in set(utf8(step.flex_data.processCoarse)["p"].to_list())
    rows = _rows(step)
    assert len(rows) == 11
    assert {step_of(r)[1] for r in rows} == {f"t{i:04d}" for i in range(5, 48, 4)}
    assert ramps[("p2020", "t0025")] == pytest.approx(600.0, abs=1e-4)
    assert sum(ramps.values()) == pytest.approx(600.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 5. no cost across a gap or a representative-block start


def test_no_cost_across_gap(tmp_path_factory) -> None:
    """5weeks timeset (blocks t0001–4, t0009–12, t0013–16, …): demand X up to t0008 and 0 from t0009, so the flow
    drops across the gap t0004 → t0009.  No row at t0009, no ramp cost anywhere."""
    vals = _cost() + _demand({f"t{i:04d}": 0.0 for i in range(9, 73)})
    steps, ramps = _ramps_dt(tmp_path_factory, "rc_gap", vals, chain=("init", "west", "coal", "5weeks"))
    step = _last(steps)
    rows = {step_of(r) for r in _rows(step)}
    assert ("p2020", "t0009") not in rows and ("p2020", "t0010") in rows
    f = flow(step, PLANT, "coal_market", "west").to_list()
    assert f[3] == pytest.approx(X, abs=TOL) and f[4] == pytest.approx(0.0, abs=TOL)
    assert sum(ramps.values()) == pytest.approx(0.0, abs=1e-4)


def test_no_cost_at_representative_block_start(tmp_path_factory) -> None:
    """Representative timeset with two adjacent 24-step blocks t0001 and t0025: demand X in the first block, 0 in
    the second.  No row at either block start, no ramp cost."""
    rpw = {"type": "map", "index_type": "str", "index_name": "base_period", "rank": 1,
           "data": [["t0001", {"type": "map", "index_type": "str", "index_name": "representative_period",
                               "rank": 1, "data": [["t0001", 0.5], ["t0025", 0.5]]}]]}
    durations = {"index_type": "str", "index_name": "time", "data": [["t0001", 24.0], ["t0025", 24.0]]}
    vals = _cost() + _demand({f"t{i:04d}": 0.0 for i in range(25, 49)}) + [
        ["timeset", "rp2", "timeline", b64("y2020", "str")],
        ["timeset", "rp2", "timeset_duration", b64(durations, "map")],
        ["timeset", "rp2", "representative_period_weights", b64(rpw, "map")],
        ["solve", "y2020_2day_dispatch", "period_timeset",
         b64({"index_type": "str", "index_name": "period", "data": [["p2020", "rp2"]]}, "map")]]
    steps, ramps = _ramps_dt(tmp_path_factory, "rc_rp", vals, entities=[["timeset", "rp2", None]])
    rows = {step_of(r) for r in _rows(_last(steps))}
    assert ("p2020", "t0001") not in rows and ("p2020", "t0025") not in rows
    assert ("p2020", "t0026") in rows
    assert sum(ramps.values()) == pytest.approx(0.0, abs=1e-4)
