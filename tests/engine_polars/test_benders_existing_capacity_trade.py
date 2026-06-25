"""Benders decomposition of trade connections with FIXED EXISTING capacity.

The Phase-2/3 Benders master originally REQUIRED every cross-region trade
connection to be investment-eligible (a guard in
``_BendersMaster._build_flextool_master`` that raised
``"connection ... has no pd_invest_set period"``).  That blocked exactly the
case the old subgradient path handled: regions that TRADE over pipes of fixed
existing capacity (``existing>0``, NOT investable).  The
``lh2_three_region`` family's pipes are such pipes (``existing=50``).

This module proves the generalised master handles trade connections of ANY
capacity kind in ONE master:

* **Existing-only, load-bearing trade (base ``lh2_three_region``).** No
  investment anywhere; each region's fixed in-region capacity is short, so
  the optimum MUST trade over the existing pipes.  Autarky (no trade) is
  strictly more expensive.  Benders must converge to the monolith optimum
  (BELOW the autarky cost) and RECOVER a non-zero cross-region trade flow
  over the existing pipes — i.e. the existing-capacity ``maxFlow`` bound
  (``v_flow ≤ existing/unitsize``, built natively by ``build_flextool`` with
  no ``v_invest_p`` term) is what makes the recourse feasible.

* **Existing-only pipes + in-region wind INVEST
  (``lh2_three_region_invest``).** The same fixed-capacity pipes, but each
  region may invest in in-region wind.  Benders must converge to the monolith
  optimum, recover the in-region wind investment (spanning all three
  regions), and contribute ZERO invested capacity for the (non-investable)
  pipes — the master builds with no pipe ``v_invest_p`` column and bounds the
  pipe flow by its existing capacity.

Both monolith references are built with ``build_flextool`` + solve, per the
CLAUDE.md "JSON-fixture single source of truth" invariant.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import polars as pl
import pytest

from polar_high import Problem, WarmProblem

from flextool.engine_polars import build_flextool, load_flextool
from flextool.engine_polars import _region_filter
from flextool.engine_polars._benders import solve_benders


FLEXTOOL_ROOT = Path(__file__).resolve().parents[2]
LH2_FIXTURE_JSON = FLEXTOOL_ROOT / "tests" / "fixtures" / "lh2_three_region.json"

REGIONS = ["region_A", "region_B", "region_C"]
TRADE_CONNS = {"pipe_AB", "pipe_BC"}
INVEST_SCENARIO = "lh2_three_region_invest"
INVEST_UNITS = {"wind_A", "wind_B", "wind_C"}


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _autarky_cost(data) -> float:
    """Sum of region recourse costs at f̄=0 (every cross-region half-flow
    pinned to 0) — the no-trade reference.  When this strictly exceeds the
    monolith optimum, the existing-capacity trade is LOAD-BEARING (the
    decomposition cannot collapse to autarky)."""
    splits = _region_filter.split(
        data, regions=REGIONS, benders_uncap_cross_region=True
    )
    warm = []
    for s in splits:
        pb = Problem()
        build_flextool(pb, s.data)
        warm.append(WarmProblem(pb))
    for w in warm:
        w.solve()
    total = 0.0
    for s, w in zip(splits, warm):
        vf = w._p._vars["v_flow"]
        cols: list[int] = []
        for hf in s.half_flows:
            sub = vf.frame.filter(
                (pl.col("p") == hf.virtual_p)
                & (pl.col("source") == hf.virtual_arc_source)
                & (pl.col("sink") == hf.virtual_arc_sink)
            )
            cols += sub["col_id"].to_list()
        if cols:
            dt = [
                tuple(r)
                for r in vf.frame.filter(pl.col("col_id").is_in(cols))
                .select(*vf.dims)
                .iter_rows()
            ]
            w.fix_cols("v_flow", dt, np.zeros(len(dt)))
        total += w.solve().obj
    return float(total)


# ---------------------------------------------------------------------------
# Case 1 — base lh2_three_region: existing-only, LOAD-BEARING trade.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_data(scenario_workdir):
    work = scenario_workdir("lh2_three_region", db_fixture="lh2")
    return load_flextool(work)


@pytest.fixture(scope="module")
def base_monolith(base_data):
    pb = Problem()
    build_flextool(pb, base_data)
    sol = pb.solve()
    assert sol.optimal, "base monolith solve not optimal"
    return sol


@pytest.mark.solver
def test_existing_capacity_trade_is_load_bearing(base_data, base_monolith):
    """Sanity: in the base fixture autarky is strictly MORE expensive than
    the monolith optimum — so trade over the existing pipes is required and
    a decomposition that could not flow over them would be wrong."""
    autarky = _autarky_cost(base_data)
    M = float(base_monolith.obj)
    assert autarky > M * (1 + 1e-4), (
        f"autarky cost {autarky:.6e} not above monolith {M:.6e} — the "
        f"existing-capacity trade is not load-bearing in this fixture; the "
        f"test would not exercise the existing-capacity recourse"
    )


@pytest.mark.solver
def test_benders_existing_only_converges_to_monolith(base_data, base_monolith):
    """``solve_benders`` on the existing-only base fixture converges to the
    monolith optimum (with a valid LB ≤ M ≤ UB sandwich) and RECOVERS a
    non-zero cross-region trade flow over the existing pipes."""
    M = float(base_monolith.obj)
    res = solve_benders(
        base_data, REGIONS, max_iters=40, tol=1e-4,
        monolith_objective=M, master="flextool",
    )
    assert res.converged, f"Benders did not converge (gap={res.gap:.3e})"

    # Valid lower bound: LB ≤ M ≤ best UB, all within tol of M.
    assert res.lower_bound <= M * (1 + 1e-6), (
        f"LB {res.lower_bound:.8e} exceeds monolith {M:.8e} — invalid bound"
    )
    assert res.upper_bound >= M * (1 - 1e-4), (
        f"UB {res.upper_bound:.8e} below monolith {M:.8e}"
    )
    assert abs(res.upper_bound - M) <= 1e-3 * max(1.0, abs(M)), (
        f"Benders UB {res.upper_bound:.8e} != monolith {M:.8e}"
    )

    # The optimum is strictly cheaper than autarky ⇒ the recovered trade
    # flow over the existing pipes is non-zero (the existing-capacity
    # maxFlow bound is exercised).
    autarky = _autarky_cost(base_data)
    assert res.upper_bound < autarky * (1 - 1e-4), (
        f"Benders UB {res.upper_bound:.6e} not below autarky {autarky:.6e} "
        f"— the existing-capacity trade was not used"
    )
    total_flow = sum(
        float(fr["value"].abs().sum()) for fr in res.trade_flow.values()
    )
    assert total_flow > 1e-3, (
        f"recovered trade flow over the existing pipes is ~0 "
        f"({total_flow:.3e}) — existing-capacity trade not recovered"
    )

    # No investment in this fixture ⇒ empty whole-system invest handoff.
    assert res.invest_solution_vars == {}, (
        f"unexpected invest handoff on a no-invest fixture: "
        f"{list(res.invest_solution_vars)}"
    )
    # The existing-only pipes carry no invested capacity.
    for conn in TRADE_CONNS:
        assert abs(res.invest.get(conn, 0.0)) <= 1e-9, (
            f"existing-only pipe {conn} reports invested capacity "
            f"{res.invest.get(conn)} — should be 0"
        )


# ---------------------------------------------------------------------------
# Case 2 — lh2_three_region_invest: existing-only pipes + in-region wind
# INVEST.  Benders recovers the in-region investment over fixed-capacity
# pipes (the scenario the old guard blocked).
# ---------------------------------------------------------------------------


def _build_invest_db(tmp_path: Path) -> str:
    if not LH2_FIXTURE_JSON.exists():
        pytest.skip(f"LH2 JSON fixture not present: {LH2_FIXTURE_JSON}")
    tests_dir = FLEXTOOL_ROOT / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    from db_utils import json_to_db  # noqa: E402
    from flextool.update_flextool.db_migration import migrate_database

    db_path = tmp_path / "lh2_invest.sqlite"
    url = json_to_db(LH2_FIXTURE_JSON, db_path)
    migrate_database(url)
    return url


def _set_single_solve(db_url: str, solve: str) -> None:
    """Override ``model.flexTool.solves`` to a single solve so the snapshot
    carries the invest sets for a direct ``solve_benders`` call."""
    import spinedb_api as api

    with api.DatabaseMapping(db_url) as db:
        arr = {
            "type": "array", "value_type": "str",
            "data": [solve], "index_name": "sequence_index",
        }
        db_value, db_type = api.to_database(arr)
        db.add_update_item(
            "parameter_value",
            entity_class_name="model",
            entity_byname=("flexTool",),
            parameter_definition_name="solves",
            alternative_name=INVEST_SCENARIO,
            value=db_value,
            type=db_type,
        )
        db.commit_session("test: single-solve invest override")


@pytest.fixture(scope="module")
def invest_solve_data(tmp_path_factory):
    """FlexData for the ``lh2_invest`` solve (the Benders invest solve over
    fixed-capacity pipes), snapshotted from a single-solve chain run."""
    from flextool.engine_polars._orchestration import run_chain_from_db

    tmp_path = tmp_path_factory.mktemp("lh2_benders_invest_solve")
    url = _build_invest_db(tmp_path)
    _set_single_solve(url, "lh2_invest")

    parent = tmp_path_factory.mktemp("_root_benders_invest_solve")
    wf = parent / f"work_{INVEST_SCENARIO}"
    wf.mkdir()
    steps = run_chain_from_db(
        input_db_url=url,
        scenario_name=INVEST_SCENARIO,
        work_folder=wf,
        csv_dump=True,
        keep_solutions=True,
    )
    shutil.copy(urlparse(url).path, wf / "tests.sqlite")
    last_step = next(reversed(list(steps.values())))
    provider = getattr(last_step, "flex_data_provider", None)
    if provider is not None:
        provider.snapshot_processed_inputs(wf)
    return load_flextool(wf)


@pytest.mark.solver
def test_benders_invest_over_existing_pipes_matches_monolith(invest_solve_data):
    """Benders on ``lh2_three_region_invest`` (in-region wind invest +
    existing-capacity pipes) converges to the monolith optimum and recovers
    the in-region wind investment; the non-investable pipes contribute no
    invested capacity."""
    data = invest_solve_data
    # The invest sets carry only the in-region wind units — the pipes are
    # NOT invest-eligible (the case the old master guard rejected).
    assert data.pd_invest_set is not None and data.pd_invest_set.height > 0
    pcol = data.pd_invest_set.columns[0]
    in_set = set(data.pd_invest_set[pcol].to_list())
    assert INVEST_UNITS <= in_set, (
        f"expected {INVEST_UNITS} in pd_invest_set, got {sorted(in_set)}"
    )
    assert not (TRADE_CONNS & in_set), (
        f"pipes unexpectedly invest-eligible: {TRADE_CONNS & in_set}"
    )

    pb = Problem()
    build_flextool(pb, data)
    mono = pb.solve()
    assert mono.optimal, "monolith invest solve not optimal"
    M = float(mono.obj)

    res = solve_benders(
        data, REGIONS, max_iters=40, tol=1e-4,
        monolith_objective=M, master="flextool",
    )
    assert res.converged, f"Benders did not converge (gap={res.gap:.3e})"
    assert res.lower_bound <= M * (1 + 1e-6), (
        f"LB {res.lower_bound:.8e} exceeds monolith {M:.8e} — invalid bound"
    )
    assert abs(res.upper_bound - M) <= 1e-3 * max(1.0, abs(M)), (
        f"Benders UB {res.upper_bound:.8e} != monolith {M:.8e}"
    )

    # --- in-region wind investment recovered, matching the monolith --------
    isv = res.invest_solution_vars
    assert "v_invest_p" in isv, (
        f"no v_invest_p in handoff (keys={list(isv)}) — in-region invest "
        f"not recovered"
    )
    inv = isv["v_invest_p"]
    assert inv.columns == ["p", "d", "value"], (
        f"v_invest_p columns {inv.columns} != ['p','d','value']"
    )

    def _nonzero(frame, ent_col):
        return {
            (str(r[ent_col]), str(r["d"])): float(r["value"])
            for r in frame.iter_rows(named=True)
            if abs(float(r["value"])) > 1e-6
        }

    asm = _nonzero(inv, "p")
    mono_p = _nonzero(mono.value("v_invest_p"), "p")
    assert mono_p, "monolith invested nothing — fixture changed?"
    # The investing entities are the in-region wind units (NOT the pipes).
    assert {k[0] for k in asm} == INVEST_UNITS, (
        f"recovered invest entities {set(k[0] for k in asm)} != "
        f"{INVEST_UNITS}"
    )
    assert set(asm) == set(mono_p), (
        f"recovered invest cells {set(asm)} != monolith {set(mono_p)}"
    )
    for cell, mono_v in mono_p.items():
        assert abs(asm[cell] - mono_v) <= 1e-3 * max(1.0, abs(mono_v)), (
            f"v_invest_p{cell}: Benders {asm[cell]:.6e} != monolith "
            f"{mono_v:.6e}"
        )
    # Wind invest spans all three regions (owner-selection exercised).
    assert {k[0] for k in asm} == INVEST_UNITS

    # --- the non-investable pipes contribute no invested capacity ----------
    for conn in TRADE_CONNS:
        assert abs(res.invest.get(conn, 0.0)) <= 1e-9, (
            f"non-investable pipe {conn} reports invested capacity "
            f"{res.invest.get(conn)}"
        )
        assert not any(e == conn for (e, _d) in asm), (
            f"pipe {conn} leaked into the invest handoff"
        )

    # Owner-disjoint handoff: each (entity, d) appears once.
    for name, frame in isv.items():
        ent_col = frame.columns[0]
        keyed = frame.select(ent_col, "d")
        assert keyed.height == keyed.unique().height, (
            f"{name} has duplicate (entity, d) rows"
        )
