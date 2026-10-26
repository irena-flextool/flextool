"""Where the storage chain starts and ends across solves (fix_start, the carry between solves, fix_end).

* ``bind_within_timeblock`` + ``fix_start`` pins the state the first step starts from: the wrap target of the first
  step (the last step of the first timeblock), not the state at the end of the first step.

Databases: ``tests/fixtures/tests.json`` / ``stochastics.json`` plus an override alternative (never a checked-in
sqlite).
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from tests.engine._lp_rows import row_bounds, row_terms
from tests.engine._unit_db import b64, build, make_db, run, time_map, utf8

REPO = Path(__file__).resolve().parents[2]
PIN = "storage_state_start_binding_cyclic_period"
BARE = "storage_state_start_binding"
WIND_BATTERY = ["init", "west", "wind", "battery"]
FULLYEAR_ROLL = ["init", "west", "battery", "wind", "fullYear", "dispatch_fullYear_roll"]
NESTED = ["init", "west", "coal", "coal_invest", "wind", "wind_invest", "battery", "battery_invest",
          "invest_solveSequence_5weeks", "invest_allYears_5weeks", "fullYear", "dispatch_fullYear_roll",
          "storage_fullYear_6h", "invest_24h", "bind_battery_forward", "unlimited_investment"]


def _f(x: float) -> list:
    return b64(x, "float")


def _s(x: str) -> list:
    return b64(x, "str")


def _bare(pb, node: str) -> dict:
    """Rows of the exact family storage_state_start_binding (row_terms is a prefix match)."""
    return {k: v for k, v in row_terms(pb, BARE, f"{node},").items() if k.startswith(f"{BARE}[")}


def _lag_of_first(fd, node: str, col: str = "t_previous_within_timeset") -> set[tuple[str, str]]:
    first = utf8(fd.nodeState_first_dt).filter(pl.col("n") == node).select("d", "t")
    lag = utf8(fd.dtttdt).join(first, on=["d", "t"], how="inner")
    return {(d, t) for d, t in lag.select("d", col).iter_rows()}


def _state_mwh(step, node: str) -> dict[str, float]:
    su = float(utf8(step.flex_data.p_state_unitsize.frame).filter(pl.col("n") == node)["value"][0])
    rows = utf8(step.solution.value("v_state")).filter(pl.col("n") == node)
    return {t: v * su for t, v in rows.select("t", "value").iter_rows()}


# ---------------------------------------------------------------------------
# bind_within_timeblock + fix_start (N8)
# ---------------------------------------------------------------------------


def test_timeblock_fix_start_pins_the_state_the_first_step_starts_from(tmp_path_factory):
    """wind_battery (y2020_2day_dispatch: p2020 t0001–t0048, one timeblock), battery bind_within_timeblock +
    fix_start 0.1 (existing 50 MWh, so 5 MWh), no self-discharge, inverter capacity 0 (the battery has no flows),
    inflow +10 MWh at t0001 and −10 MWh at t0003.

    The first step's state change is S[t0001] − S[t0048] (t0048 = its wrap target in the timeblock), so the start
    state is S[t0048] = 5 and the balance gives S[t0001] = S[t0002] = 5 + 10 = 15, S[t0003..t0048] = 15 − 10 = 5:
    no slack.  Before, the pin sat on S[t0001] = 5, which needs S[t0048] = −5: the balance needed slack."""
    url, scen = make_db(tmp_path_factory, alt="sse_tb", base_chain=WIND_BATTERY, values=[
        ["node", "battery", "self_discharge_loss", _f(0.0)],
        ["node", "battery", "inflow", time_map({"t0001": 10.0, "t0003": -10.0}, 0.0)],
        ["connection", "battery_inverter", "existing", _f(0.0)]])
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    fd = step.flex_data
    assert _lag_of_first(fd, "battery") == {("p2020", "t0048")}
    pb = build(fd)
    assert set(row_terms(pb, PIN, "battery,")) == {f"{PIN}[battery,p2020,t0048]"}
    assert not _bare(pb, "battery")
    lo, hi = row_bounds(pb, PIN, "battery,")[f"{PIN}[battery,p2020,t0048]"]
    su = float(utf8(fd.p_state_unitsize.frame).filter(pl.col("n") == "battery")["value"][0])
    assert lo == pytest.approx(0.1 * 50.0 / su) and hi == pytest.approx(lo)
    state = _state_mwh(step, "battery")
    assert state["t0048"] == pytest.approx(5.0, abs=1e-7)
    assert state["t0001"] == pytest.approx(15.0, abs=1e-7)
    assert state["t0002"] == pytest.approx(15.0, abs=1e-7)
    assert state["t0003"] == pytest.approx(5.0, abs=1e-7)
    for slack in ("vq_state_up", "vq_state_down"):
        rows = utf8(step.solution.value(slack)).filter(pl.col("n") == "battery")
        assert rows["value"].abs().max() == pytest.approx(0.0, abs=1e-9), slack


def test_timeblock_fix_start_rolling_pins_the_end_of_the_rolls_timeblock_part(tmp_path_factory):
    """fullYear_roll with horizon 36 h / jump 24 h: the first roll holds t0001–t0036 of the fullYear timeblock, so its
    first step wraps to t0036 (the end of the timeblock's part in the roll) and fix_start is pinned there.  Later
    rolls are cyclic continuation rolls: no pin (and no carry, #119)."""
    url, scen = make_db(tmp_path_factory, alt="sse_tb_roll", base_chain=FULLYEAR_ROLL, values=[
        ["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", _f(36.0)],
        ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", _f(24.0)]])
    steps = list(run(tmp_path_factory, url, scen).values())
    assert len(steps) > 2
    fd0 = steps[0].flex_data
    assert _lag_of_first(fd0, "battery") == {("p2020", "t0036")}
    pb0 = build(fd0)
    assert set(row_terms(pb0, PIN, "battery,")) == {f"{PIN}[battery,p2020,t0036]"}
    assert not _bare(pb0, "battery")
    for step in steps[1:3]:
        pb = build(step.flex_data)
        assert not row_terms(pb, PIN, "battery,") and not _bare(pb, "battery")


def _stochastic_db(tmp_path_factory, alt: str, values: list) -> tuple[str, str]:
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db
    from tests.engine.test_cyclic_fix_start import _lp_snapshot

    data = json.loads((REPO / "tests" / "fixtures" / "stochastics.json").read_text())
    data = _lp_snapshot()._overlay(data, name=alt, base="2_day_stochastic_dispatch", values=values)
    work = tmp_path_factory.mktemp(alt)
    (work / "f.json").write_text(json.dumps(data))
    url = json_to_db(work / "f.json", work / "f.sqlite")
    migrate_database(url)
    return url, alt


def test_timeblock_fix_start_stochastic_pins_in_the_trunk(tmp_path_factory):
    """2_day_stochastic_dispatch, hydro_reservoir bind_within_timeblock + fix_start 0.5: the timeblock lag stays in
    its period, so each first step of the solve (the trunk's and each forecast branch's) is pinned at the wrap
    target inside its own period, and the solve is optimal."""
    from flextool.engine import run_chain_from_db

    url, scen = _stochastic_db(tmp_path_factory, "sse_tb_stoch", [
        ["node", "hydro_reservoir", "storage_binding_method", "bind_within_timeblock"],
        ["node", "hydro_reservoir", "storage_start_end_method", "fix_start"],
        ["node", "hydro_reservoir", "storage_state_start", 0.5]])
    steps = list(run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp("sse_tb_stoch_w"),
                                   keep_solutions=True).values())
    assert all(s.solution.optimal for s in steps)
    fd = steps[0].flex_data
    first = utf8(fd.nodeState_first_dt).filter(pl.col("n") == "hydro_reservoir").select("d", "t")
    lag = utf8(fd.dtttdt).join(first, on=["d", "t"], how="inner")
    assert lag.height == first.height > 1
    want = {(d, t) for d, t in lag.select("d", "t_previous_within_timeset").iter_rows()}
    assert {d for d, _t in want} == set(first["d"].to_list())
    got = set(row_terms(build(fd), PIN, "hydro_reservoir,"))
    assert got == {f"{PIN}[hydro_reservoir,{d},{t}]" for d, t in want}


def test_timeblock_fix_start_yields_to_a_nested_fix_at_the_wrap_target(tmp_path_factory):
    """Nested invest → storage → dispatch (36 h / 24 h) with nested fix_quantity and the battery
    bind_within_timeblock + fix_start.  The first dispatch roll holds t0001–t0036 of the fullYear timeblock: its wrap
    target is t0036, the step where node_balance_fix_quantity_eq_lower fixes the state to the storage solve's.  The
    upper value wins: the roll has the fix row and no start pin on that variable, and it solves."""
    url, scen = make_db(tmp_path_factory, alt="sse_tb_nested", base_chain=NESTED, values=[
        ["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", _f(36.0)],
        ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", _f(24.0)],
        ["node", "battery", "storage_nested_fix_method", _s("fix_quantity")],
        ["node", "battery", "storage_binding_method", _s("bind_within_timeblock")]])
    steps = run(tmp_path_factory, url, scen)
    roll0 = steps["dispatch_fullYear_roll_roll_0"]
    assert roll0.solution.optimal
    fd = roll0.flex_data
    assert _lag_of_first(fd, "battery") == {("p2020", "t0036")}
    pb = build(fd)
    assert set(row_terms(pb, "node_balance_fix_quantity_eq_lower", "battery,")) == {
        "node_balance_fix_quantity_eq_lower[battery,p2020,t0036]"}
    assert not row_terms(pb, PIN, "battery,")
    assert not _bare(pb, "battery")
    # The storage solve itself (no nested fix above it) keeps its start pin at its own wrap target.
    fd_storage = steps["storage_fullYear_6h"].flex_data
    assert set(row_terms(build(fd_storage), PIN, "battery,")) == {
        f"{PIN}[battery,{d},{t}]" for d, t in _lag_of_first(fd_storage, "battery")}
