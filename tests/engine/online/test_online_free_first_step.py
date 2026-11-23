"""Units online at the first step of a solve (and of a period that does not continue the previous one) are free.

A free start has no online predecessor: no ``online_change`` row, and its start-up and shut-down are 0 (their upper
bound), so the units online there are chosen freely, with no start-up cost (``_online.free_start_rows``).  Before,
the step was linked to the last step of its stretch (a cyclic wrap), so a unit online at the start paid a start-up
when it was offline at the end, and a minimum up time could hold it on for a start-up it never made.  In a rolling
two-stage (stochastic + realization) solve the wrap made the stochastic roll and its realization link their first
step to different steps (the end of the horizon, the end of the jump); free starts are the same in both.

Hand-derived oracles on ``tests.json`` (``coal_plant``: one 500 MW unit, efficiency 0.4, coal 20 €/MWh, so 50 €/MWh
of fuel; ``west``: penalty_up 900, penalty_down 800 €/MWh) and on ``rolling_separate_dispatch.json``.  Databases
are built from JSON under the pytest tmp dir.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import polars as pl
import pytest

from flextool.engine.preprocess.params._ramp_links import online_free_start_cells
from tests.engine._lp_rows import row_terms
from tests.engine._unit_db import PLANT, PLANT_CHAIN, b64, build, make_db, run, time_map, uc, utf8

pytestmark = pytest.mark.solver

SOLVE = "y2020_2day_dispatch"
FLEXTOOL_ROOT = Path(__file__).resolve().parents[3]
ROLLING_FIXTURE = FLEXTOOL_ROOT / "tests" / "fixtures" / "rolling_separate_dispatch.json"


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _steps_of(hours: int) -> list:
    return [["timeset", "2day", "timeset_duration",
             b64({"index_type": "str", "index_name": "time", "data": [["t0001", float(hours)]]}, "map")]]


def _plant(steps: list, var: str, kind: str = "integer") -> list[float]:
    frames = [utf8(s.solution.value(f"{var}_{kind}")).filter(pl.col("p") == PLANT) for s in steps]
    if any("realized_dispatch" in str(getattr(s, "solve_name", "")) for s in steps):
        raise AssertionError("realization steps are not expected here")
    if len(steps) > 1:   # a rolling solve: the realized (jump) steps of each roll
        frames = [f.join(utf8(s.flex_data.realized_dispatch.rename({"period": "d", "step": "t"})).select("d", "t"),
                         on=["d", "t"], how="semi") for f, s in zip(frames, steps)]
    v = pl.concat(frames).sort("d", "t")["value"].to_list()
    return [round(x, 9) + 0.0 for x in v]


def _demand(mw: list[float]) -> list:
    return [["node", "west", "inflow", time_map({f"t{i + 1:04d}": -x for i, x in enumerate(mw)}, 0.0)]]


ROLLING = [["solve", SOLVE, "solve_mode", b64("rolling_window", "str")],
           ["solve", SOLVE, "rolling_solve_horizon", b64(2.0, "float")],
           ["solve", SOLVE, "rolling_solve_jump", b64(2.0, "float")]]


def _first_step_case(tmp_path_factory, startup_cost: float, mode: str):
    values = uc(PLANT, 0.2, "integer_fix_in_uc") + _steps_of(3) + _demand([250.0, 250.0, 0.0]) + [
        ["unit", PLANT, "startup_cost", b64(startup_cost, "float")]] + (ROLLING if mode == "rolling" else [])
    url, scen = make_db(tmp_path_factory, alt=f"free1_{mode}_{int(startup_cost)}", values=values,
                        base_chain=PLANT_CHAIN)
    return list(run(tmp_path_factory, url, scen).values())


@pytest.mark.parametrize("mode", ["single", "rolling"])
def test_first_step_chooses_units_online_without_a_startup(tmp_path_factory, mode) -> None:
    """One binary unit (500 MW, min_load 0.2 = 100 MW), demand 250, 250, 0 MW over 3 h, start-up cost 10 €/MW.

    Hand derivation: online at t1-t2 (250 MW, fuel 50 €/MWh, below the 900 €/MWh shortfall penalty); offline at t3
    (online it would make at least 100 MW of surplus at 800 + 50 €/MWh, and a shut-down is free).  t1 is a free
    start: online there with no start-up, so su = (0, 0, 0), sd = (0, 0, 1), and the objective does not depend on
    the start-up cost.  Before, t1 wrapped to the offline t3 and paid a start-up of 1 unit (10 · 500 €).
    Rolling (horizon = jump = 2 h): roll 0 holds t1-t2 and starts free; roll 1 (t3) carries the units online of t2
    and shuts down: the same commitment as the single solve, so the first-step rule is the same in both."""
    steps = _first_step_case(tmp_path_factory, 10.0, mode)
    assert len(steps) == (2 if mode == "rolling" else 1)
    assert _plant(steps, "v_online") == [1.0, 1.0, 0.0]
    assert _plant(steps, "v_startup") == [0.0, 0.0, 0.0]
    assert _plant(steps, "v_shutdown") == [0.0, 0.0, 1.0]
    flows = pl.concat([utf8(s.solution.value("v_flow")).filter(pl.col("p") == PLANT) for s in steps])
    assert sorted(flows.filter(pl.col("value") > 1e-9)["value"].to_list()) == pytest.approx([0.5, 0.5])
    free = _first_step_case(tmp_path_factory, 0.0, mode)
    assert sum(s.solution.obj for s in steps) == pytest.approx(sum(s.solution.obj for s in free), rel=1e-9)
    fd = steps[0].flex_data
    assert online_free_start_cells(fd.dt, fd.dt_online_wrap, fd.rp_block_first).rows() == [("p2020", "t0001")]
    assert not row_terms(build(fd), "online_change_integer", f"{PLANT},p2020,t0001")


def test_min_uptime_owes_nothing_for_the_first_step(tmp_path_factory) -> None:
    """One binary unit with min_uptime 3 h, min_load 0.2 (100 MW), demand 250, 0, 0, 0 MW.

    Hand derivation: online at t1 (free start: no start-up), so no minimum up time is owed; at t2 it shuts down
    (online it would make at least 100 MW of surplus at 850 €/MWh): on = (1, 0, 0, 0), su = 0, sd = (0, 1, 0, 0),
    and no surplus.  Before, t1 wrapped to the offline t4: a start-up at t1, and minimum_uptime then held the unit
    online at t2-t3 with 100 MW of surplus each (the phantom start-up of the two-stage infeasibility, in its
    smallest form)."""
    values = uc(PLANT, 0.2, "integer_fix_in_uc") + _steps_of(4) + _demand([250.0, 0.0, 0.0, 0.0]) + [
        ["unit", PLANT, "startup_cost", b64(10.0, "float")],
        ["unit", PLANT, "minimum_time_method", b64("min_uptime", "str")],
        ["unit", PLANT, "min_uptime", b64(3.0, "float")]]
    url, scen = make_db(tmp_path_factory, alt="free1_uptime", values=values, base_chain=PLANT_CHAIN)
    steps = list(run(tmp_path_factory, url, scen).values())
    assert _plant(steps, "v_online") == [1.0, 0.0, 0.0, 0.0]
    assert _plant(steps, "v_startup") == [0.0, 0.0, 0.0, 0.0]
    assert _plant(steps, "v_shutdown") == [0.0, 1.0, 0.0, 0.0]
    surplus = utf8(steps[0].solution.value("vq_state_down")).filter(pl.col("n") == "west")
    assert surplus["value"].abs().max() == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Two-stage rolling solves (rolling_separate_dispatch: a stochastic roll and its realization per 24-h jump)


def _enc(value, kind: str) -> list:
    return [base64.b64encode(json.dumps(value).encode()).decode(), kind]


def _dec(value) -> object:
    return json.loads(base64.b64decode(value[0]))


def _leaf(data) -> dict:
    return {"type": "map", "index_type": "str", "data": data}


def _reporter(startup: str):
    """The reported two-stage case: every unit ``startup`` (linear or binary), one forecast (``mid``) plus the
    ``realized: yes`` branch at each roll start, min up / down times 10 h, no start-up cost."""
    def patch(data: dict) -> dict:
        for r in data["parameter_values"]:
            if r[0] == "unit" and r[2] == "startup_method":
                r[3] = _enc(startup, "str")
            elif r[0] == "unit" and r[2] in ("min_uptime", "min_downtime"):
                r[3] = _enc(10.0, "float")
            elif r[0] == "unit" and r[2] == "startup_cost":
                r[3] = _enc(0.0, "float")
            elif r[0] == "solve" and r[1] == "1week_rolling" and r[2] == "stochastic_branches":
                m = _dec(r[3])
                m["data"][0][1]["data"] = [x for x in m["data"][0][1]["data"] if x[0] in ("realized", "mid")]
                r[3] = _enc(m, r[3][1])
        return data
    return patch


def _two_periods(data: dict) -> dict:
    """period1 on t0001-t0012 and period2 on t0025-t0072 (a gap between them: period2 does not continue period1),
    rolled with the fixture's 24-h jump and 48-h horizon: roll 0's jump holds period1 and period2's first 12 hours,
    so period2 starts inside the jump.  One forecast ``mid`` (wind 0.8 × the realized profile) and the ``realized``
    branch at each roll start (t0001, t0037, t0061); binary units with their min up / down times."""
    pv = data["parameter_values"]
    cf = dict(next(_dec(r[3]) for r in pv if r[1] == "wind" and r[2] == "profile" and r[4] == "system")["data"])
    steps = [f"t{i:04d}" for i in list(range(1, 13)) + list(range(25, 73))]
    starts = ["t0001", "t0037", "t0061"]
    branches = {"realized": 1.0, "mid": 0.8}

    def window(a):
        i = steps.index(a)
        return steps[i:i + 48]

    def fan(ts):
        return _leaf([[b, _leaf([[t, _leaf([["yes" if b == "realized" else "no", 1.0]])] for t in ts])]
                      for b in branches])
    out = []
    for r in pv:
        if r[4] == "1week_rolling":
            if r[0] == "timeset":
                continue
            if r[2] == "period_timeset":
                r[3] = _enc({"index_type": "str", "data": [["period1", "ts_a"], ["period2", "ts_b"]]}, "map")
            elif r[2] == "realized_periods":
                r[3] = _enc({"value_type": "str", "data": ["period1", "period2"]}, "array")
            elif r[2] == "stochastic_branches":
                r[3] = _enc({"index_type": "str", "data": [["period1", fan(["t0001"])],
                                                           ["period2", fan(["t0037", "t0061"])]]}, "map")
            elif r[1] == "wind" and r[2] == "profile":
                r[3] = _enc({"index_type": "str", "data": [
                    [b, _leaf([[a, _leaf([[t, min(1.0, cf[t] * k)] for t in window(a)])] for a in starts])]
                    for b, k in branches.items()]}, "map")
        out.append(r)
    data["parameter_values"] = out
    data["entities"] += [["timeset", "ts_a", None], ["timeset", "ts_b", None]]
    for name, dur in (("ts_a", [["t0001", 12.0]]), ("ts_b", [["t0025", 48.0]])):
        data["parameter_values"] += [
            ["timeset", name, "timeline", _enc("y2020", "str"), "1week_rolling"],
            ["timeset", name, "timeset_duration", _enc({"index_type": "str", "data": dur}, "map"), "1week_rolling"]]
    return data


def _run_rolling(tmp_path: Path, patch) -> dict:
    from flextool.engine import run_chain_from_db
    from flextool.update_flextool.db_migration import migrate_database

    tests_dir = FLEXTOOL_ROOT / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    from db_utils import json_to_db  # noqa: E402

    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps(patch(json.loads(ROLLING_FIXTURE.read_text()))))
    url = json_to_db(fixture, tmp_path / "db.sqlite")
    migrate_database(url)
    work = tmp_path / "work"
    work.mkdir()
    assert os.environ.get("FLEXTOOL_AUTOSCALE_STRICT") == "1"
    return run_chain_from_db(url, "rolling_separate_dispatch", work_folder=work, warm=False, keep_solutions=True)


ROLLS = ["1week_rolling_roll_0", "1week_rolling__realized_dispatch_roll_0", "1week_rolling_roll_1",
         "1week_rolling__realized_dispatch_roll_1", "1week_rolling_roll_2", "1week_rolling__realized_dispatch_roll_2"]


def _at(step, var: str, d: str, t: str) -> dict[str, float]:
    v = utf8(step.solution.value(var)).filter((pl.col("d") == d) & (pl.col("t") == t))
    return {p: x for p, x in zip(v["p"], v["value"])}


@pytest.mark.parametrize("startup", ["linear", "integer_fix_in_uc"])
def test_two_stage_rolling_with_min_up_down_times_solves(tmp_path_factory, startup) -> None:
    """The reported case: a rolling two-stage solve (one forecast + realized, 24-h jump, 48-h horizon) whose units
    have 10-h minimum up and down times.  With linear units the first realization roll was infeasible: its first
    step wrapped to the jump end, the stochastic roll's to the horizon end, so the pinned units online implied a
    start-up the stochastic roll never made, inside the minimum up time.  Now all six solves are optimal, and at
    t0001 the start-ups and shut-downs are 0 in the stochastic roll and its realization alike (a free start in both).
    The free start alone made the linear case feasible with its units online still pinned; since then a realization
    decides linear units online again (test_realization_inherits), and only the binary ones stay pinned."""
    var = f"v_online_{'linear' if startup == 'linear' else 'integer'}"
    steps = _run_rolling(tmp_path_factory.mktemp(f"reporter_{startup}"), _reporter(startup))
    assert list(steps) == ROLLS
    assert all(s.solution.optimal for s in steps.values())
    da, rz = steps[ROLLS[0]], steps[ROLLS[1]]
    for step, d in ((da, "period1_mid"), (rz, "period1")):
        for ev in ("v_startup", "v_shutdown"):
            at = _at(step, ev + var[len("v_online"):], d, "t0001")
            assert set(at) == {"coal_plant", "gas_plant"} and all(v == 0.0 for v in at.values()), (d, ev, at)
    if startup == "integer_fix_in_uc":     # the realization keeps the binary units online of the stochastic roll's tied steps
        assert _at(rz, var, "period1", "t0001") == pytest.approx(_at(da, var, "period1_mid", "t0001"), abs=1e-9)


def _links(pb, kind: str, rename: dict[str, str], cells: set) -> dict:
    """``{(p, d, t): sorted v_online columns of its online_change row}`` with the period names mapped by
    ``rename`` (a forecast branch to its period), for the rows at ``cells`` ``(d, t)``."""
    out = {}
    for row, terms in row_terms(pb, f"online_change_{kind}").items():
        p, d, t = row[row.index("[") + 1:-1].split(",")
        d = rename.get(d, d)
        if (d, t) not in cells:
            continue
        cols = []
        for c in terms:
            if c.startswith(f"v_online_{kind}["):
                cp, cd, ct = c[c.index("[") + 1:-1].split(",")
                cols.append((cp, rename.get(cd, cd), ct))
        out[(p, d, t)] = sorted(cols)
    return out


def test_a_period_start_inside_a_jump_links_the_same_in_both_stages(tmp_path_factory) -> None:
    """period2 does not continue period1 and starts inside roll 0's jump (:func:`_two_periods`).  Its first step
    t0025 is a free start in the stochastic roll and in the realization: no online_change row in either, start-ups
    and shut-downs 0.  Every other online_change row of the realization's window reads the same predecessor in both
    stages.  Before, the stochastic roll's t0025 wrapped to the end of its horizon (t0060) and the realization's to
    the end of its jump (t0036): pinned binary units could then owe a start-up the stochastic roll never made."""
    steps = _run_rolling(tmp_path_factory.mktemp("two_periods"), _two_periods)
    assert list(steps) == ROLLS
    assert all(s.solution.optimal for s in steps.values())
    da, rz = steps[ROLLS[0]], steps[ROLLS[1]]
    cells = set(utf8(rz.flex_data.dt.select("d", "t")).rows())
    assert ("period2", "t0025") in cells and ("period2", "t0036") in cells and ("period2", "t0037") not in cells
    rename = {"period1_mid": "period1", "period2_mid": "period2"}
    links_da = _links(build(da.flex_data), "integer", rename, cells)
    links_rz = _links(build(rz.flex_data), "integer", rename, cells)
    assert links_da == links_rz
    for p in ("coal_plant", "gas_plant"):
        assert (p, "period2", "t0025") not in links_rz and (p, "period2", "t0026") in links_rz
        assert links_rz[(p, "period2", "t0026")] == [(p, "period2", "t0025"), (p, "period2", "t0026")]
    for step, d in ((da, "period2_mid"), (rz, "period2")):
        for ev in ("v_startup_integer", "v_shutdown_integer"):
            assert all(v == 0.0 for v in _at(step, ev, d, "t0025").values()), (d, ev)
