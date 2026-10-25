"""``fix_start`` of a cyclic storage node pins the state its first step starts from (#15, FA-3).

A cyclic node's first step wraps to a later step of the solve, so its start state is the state of that step:
the last step of the solve for ``bind_within_solve``, the last step of the first period for
``bind_within_period`` and ``bind_intraperiod_blocks``.  The pinned step comes from the node's own lag (the
state change's), so it is right also when the wrap is not "the last period" (stochastic solves).  Before, every
cyclic node was pinned at the last step of the first period: for ``bind_within_solve`` in a multi-period solve
that is the middle of the horizon.

Databases: ``tests/fixtures/tests.json`` / ``stochastics.json`` plus an override alternative.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import polars as pl
import pytest

from tests.engine._lp_rows import row_bounds, row_terms
from tests.engine._unit_db import b64, build, make_db, run, utf8

REPO = Path(__file__).resolve().parents[2]
CHAIN = ["init", "west", "wind", "battery", "5weeks"]
PIN = "storage_state_start_binding_cyclic_period"


def _step(tmp_path_factory, alt: str, method: str):
    url, scen = make_db(tmp_path_factory, alt=alt, base_chain=CHAIN,
                        values=[["node", "battery", "storage_binding_method", b64(method, "str")]])
    return list(run(tmp_path_factory, url, scen).values())[-1]


def test_within_solve_pins_the_last_step_of_the_solve(tmp_path_factory):
    """y2020_5week (p2020, p2025), battery bind_within_solve + fix_start 0.1: the pin is on the solve's last step
    (p2025, t0040) with RHS 0.1·existing/su; before on (p2020, t0040), mid-horizon."""
    step = _step(tmp_path_factory, "cfs_ws", "bind_within_solve")
    fd = step.flex_data
    assert set(utf8(fd.dt)["d"].to_list()) == {"p2020", "p2025"}
    pb = build(fd)
    rows = row_terms(pb, PIN, "battery,")
    assert set(rows) == {f"{PIN}[battery,p2025,t0040]"}, sorted(rows)
    su = float(utf8(fd.p_state_unitsize.frame).filter(pl.col("n") == "battery")["value"][0])
    existing = float(utf8(fd.p_state_existing_capacity.frame)
                     .filter((pl.col("n") == "battery") & (pl.col("d") == "p2025"))["value"][0])
    lo, hi = row_bounds(pb, PIN, "battery,")[f"{PIN}[battery,p2025,t0040]"]
    assert lo == pytest.approx(0.1 * existing / su) and hi == pytest.approx(lo)
    state = utf8(step.solution.value("v_state")).filter((pl.col("n") == "battery") & (pl.col("d") == "p2025")
                                                       & (pl.col("t") == "t0040"))["value"][0]
    assert state == pytest.approx(0.1 * existing / su, abs=1e-9)


def test_within_period_pins_the_last_step_of_the_first_period(tmp_path_factory):
    """bind_within_period on the same solve: the first period wraps to its own last step (p2020, t0040) — unchanged."""
    pb = build(_step(tmp_path_factory, "cfs_wp", "bind_within_period").flex_data)
    assert set(row_terms(pb, PIN, "battery,")) == {f"{PIN}[battery,p2020,t0040]"}


def _lp_snapshot():
    name = "_lp_snapshot_for_tests"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPO / "tools" / "lp_snapshot.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def test_stochastic_within_solve_pins_the_wrap_of_the_first_step(tmp_path_factory):
    """Stochastic 2-day dispatch, hydro_reservoir bind_within_solve + fix_start 0.5: the pin is on the step the solve's
    first step wraps to in the state change's lag (dtttdt), whatever period that is, and the solve is optimal."""
    from flextool.engine import run_chain_from_db
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    data = json.loads((REPO / "tests" / "fixtures" / "stochastics.json").read_text())
    data = _lp_snapshot()._overlay(data, name="custom", base="2_day_stochastic_dispatch", values=[
        ["node", "hydro_reservoir", "storage_binding_method", "bind_within_solve"],
        ["node", "hydro_reservoir", "storage_start_end_method", "fix_start"],
        ["node", "hydro_reservoir", "storage_state_start", 0.5]])
    work = tmp_path_factory.mktemp("cfs_stoch")
    (work / "f.json").write_text(json.dumps(data))
    url = json_to_db(work / "f.json", work / "f.sqlite")
    migrate_database(url)
    steps = list(run_chain_from_db(url, "custom", work_folder=work / "w", keep_solutions=True).values())
    assert all(s.solution.optimal for s in steps)
    fd = steps[0].flex_data
    first = utf8(fd.nodeState_first_dt).filter(pl.col("n") == "hydro_reservoir")
    lag = utf8(fd.dtttdt).join(first.select("d", "t"), on=["d", "t"], how="inner")
    want = {f"{PIN}[hydro_reservoir,{dp},{tp}]"
            for dp, tp in lag.select("d_previous", "t_previous_within_solve").unique().iter_rows()}
    assert want
    got = set(row_terms(build(fd), PIN, "hydro_reservoir,"))
    assert got == want
    # Here the solve's first step (the trunk) wraps to the last step of every forecast branch: the start state is
    # pinned there, not at the end of the trunk period (the old location).
    periods = utf8(fd.dt)["d"].unique(maintain_order=True).to_list()
    assert all(f",{periods[0]}," not in name for name in got), (sorted(got), periods)


@pytest.mark.parametrize("method,family", [("bind_forward_only", None),
                                            ("bind_within_solve", PIN),
                                            ("bind_within_timeblock", "storage_state_start_binding")])
def test_fix_start_end_pins_the_start(tmp_path_factory, method, family):
    """``storage_start_end_method`` *fix_start_end* fixes the start like *fix_start* (the schema: "fix start and
    end"; the end half is item 133): the node is in storage_fix_start and gets its start term / pin.  Before it was
    in neither, so the storage started from free energy."""
    url, scen = make_db(tmp_path_factory, alt=f"cfse_{method[5:12]}", base_chain=CHAIN, values=[
        ["node", "battery", "storage_binding_method", b64(method, "str")],
        ["node", "battery", "storage_start_end_method", b64("fix_start_end", "str")]])
    fd = list(run(tmp_path_factory, url, scen).values())[-1].flex_data
    assert "battery" in utf8(fd.storage_fix_start)["n"].to_list()
    pb = build(fd)
    if family is not None:
        assert row_terms(pb, family, "battery,"), family
        return
    d0, t0 = utf8(fd.dt).sort("d", "t").row(0)
    name = f"nodeBalance_eq[battery,{d0},{t0}]"
    lo, _hi = row_bounds(pb, "nodeBalance_eq", "battery,")[name]
    start = float(utf8((fd.p_state_start * fd.p_state_existing_capacity).frame)
                  .filter((pl.col("n") == "battery") & (pl.col("d") == d0))["value"][0])
    assert start > 0
    inflow = 0.0 if fd.p_inflow is None else float(
        utf8(fd.p_inflow.frame).filter((pl.col("n") == "battery") & (pl.col("d") == d0) & (pl.col("t") == t0))
        ["value"].sum())
    # the start energy (weighted by the step's self-discharge 1 − β) is in the bound
    assert lo < -inflow - 0.99 * start
