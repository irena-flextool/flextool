"""The reference-value end pin is a share of the built storage capacity, invested storage included (#14, FA-2).

``storage_state_solve_horizon_reference_value`` pins the state at the end of the solve horizon to
``storage_state_reference_value`` × the storage capacity, like the start pins and maxState:
``v_state = rv·(existing + Σ invested·su − Σ divested·su)/su``.  Before, only the existing capacity counted, so new
storage was forced to end empty.  Databases: JSON fixtures plus an override alternative (never a checked-in sqlite).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from tests.engine._lp_rows import row_terms
from tests.engine._unit_db import b64, build, make_db, run, utf8

REPO = Path(__file__).resolve().parents[2]
PIN = "storage_state_solve_horizon_reference_value"


def test_reference_value_counts_invested_storage(tmp_path_factory):
    """wind_battery_invest chain, battery bind_forward_only with use_reference_value 0.4, no existing storage and a
    forced build (invest_min_total 100 MWh): the end pin of each last period holds v_invest_n with −0.4 and the
    solved end state is 0.4 × the invested MWh (before: no invest term, the new battery ended empty)."""
    vals = [["node", "battery", "storage_binding_method", b64("bind_forward_only", "str")],
            ["node", "battery", "storage_solve_horizon_method", b64("use_reference_value", "str")],
            ["node", "battery", "storage_state_reference_value", b64(0.4, "float")],
            ["node", "battery", "existing", b64(0.0, "float")],
            ["node", "battery", "invest_min_total", b64(100.0, "float")]]
    url, scen = make_db(tmp_path_factory, alt="rvi", values=vals,
                        base_chain=["init", "west", "wind", "battery", "battery_invest", "5weeks"])
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    fd = step.flex_data
    su = float(utf8(fd.p_state_unitsize.frame).filter(pl.col("n") == "battery")["value"][0])
    rows = row_terms(build(fd), PIN, "battery,")
    assert rows
    invest = utf8(step.solution.value("v_invest_n")).filter(pl.col("n") == "battery")
    built = {d: v for d, v in invest.select("d", "value").iter_rows()}
    alive = utf8(fd.edd_invest_set).filter(pl.col("e") == "battery")
    state = utf8(step.solution.value("v_state")).filter(pl.col("n") == "battery")
    total_built = 0.0
    for name, cols in rows.items():
        _n, d, t = name[name.index("[") + 1:-1].split(",")
        inv_cols = {c: v for c, v in cols.items() if c.startswith("v_invest_n[battery,")}
        want_periods = set(alive.filter(pl.col("d") == d)["d_invest"].to_list())
        assert {c[c.index(",") + 1:-1] for c in inv_cols} == want_periods, (name, inv_cols)
        for v in inv_cols.values():
            assert v == pytest.approx(-0.4, rel=1e-12)
        in_service = sum(built.get(dd, 0.0) for dd in want_periods)
        total_built = max(total_built, in_service)
        end = state.filter((pl.col("d") == d) & (pl.col("t") == t))["value"][0]
        assert end * su == pytest.approx(0.4 * in_service * su, rel=1e-9, abs=1e-9)
    assert total_built * su >= 100.0 - 1e-6


def test_reference_value_invest_follows_the_branch(tmp_path_factory):
    """Stochastic recourse with non-anticipativity (``stoch_recourse_na``): the end pin of a branch period holds
    v_invest_n of exactly the invest periods in service there (edd_invest_set: the trunk and the branch's own
    periods), never one of a sibling branch."""
    from flextool.engine import run_chain_from_db
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    src = REPO / "tests" / "fixtures" / "stoch_two_period_invest_na.json"
    work = tmp_path_factory.mktemp("rvi_na")
    (work / "f.json").write_text(src.read_text())
    url = json_to_db(work / "f.json", work / "f.sqlite")
    migrate_database(url)
    steps = list(run_chain_from_db(url, "recourse_na", work_folder=work / "w", keep_solutions=True).values())
    checked = 0
    for step in steps:
        fd = step.flex_data
        if fd.edd_invest_set is None:
            continue
        alive = utf8(fd.edd_invest_set).filter(pl.col("e") == "hydro_reservoir")
        for name, cols in row_terms(build(fd), PIN, "hydro_reservoir,").items():
            _n, d, _t = name[name.index("[") + 1:-1].split(",")
            got = {c[c.index(",") + 1:-1] for c in cols if c.startswith("v_invest_n[hydro_reservoir,")}
            want = set(alive.filter(pl.col("d") == d)["d_invest"].to_list())
            assert got == want, (name, got, want)
            # only the branch's own periods and its trunk (``period1`` for ``period1_mid``), never a sibling branch
            assert all(p == d or d.startswith(p) for p in got), (name, got)
            if want:
                checked += 1
    assert checked > 0
