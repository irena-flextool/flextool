"""Benders TIER-1 invest->dispatch handoff assembly test.

Proves that ``solve_benders`` assembles a whole-system
``invest_solution_vars`` dict — the UNION of each REGION's owner-selected
in-region investment and the MASTER's cross-region trade-connection
``v_invest_p`` — that matches the MONOLITH's ``v_invest_p`` / ``v_invest_n``
for the invest-bearing ``lh2_three_region_trade_invest`` fixture.

This is the correctness gate for the Phase-4 subgradient-removal plan
(§4.4 GAP-a): the assembled dict has the same shape and the same
p_unitsize-NORMALISED units the subgradient path produced
(``Solution.value(name)`` semantics: ``(entity_col, "d", "value")``), and
its values reconcile to the monolith — so the invest->dispatch chain can be
rewired onto Benders without regressing the handoff.

Both the regions and the master are FlexTool-built, so the master's
``v_invest_p`` for the pipes is normalised IDENTICALLY to the monolith's
(unitsize cancels in the master capacity coupling); the assembled frame
therefore drops straight into ``build_handoff_from_solution`` with no
rescale.  The cross-region trade connections appear in BOTH regions'
membership (each touches an in-region node), so the assembler deliberately
EXCLUDES them from the region invest and sources their value SOLELY from the
master — without that exclusion a region's pinned half-flow model carries a
ZERO invest var for the pipe and would clobber the master's value.
"""
from __future__ import annotations

import polars as pl
import pytest

from polar_high import Problem

from flextool.engine_polars import build_flextool, load_flextool
from flextool.engine_polars._benders import solve_benders

_REGIONS = ["region_A", "region_B", "region_C"]
# The two cross-region pipes the master owns.
_TRADE_CONNS = {"pipe_AB", "pipe_BC"}


@pytest.fixture(scope="module")
def ti_workdir(scenario_workdir):
    return scenario_workdir(
        "lh2_three_region_trade_invest", db_fixture="lh2_trade_invest"
    )


@pytest.fixture(scope="module")
def ti_data(ti_workdir):
    return load_flextool(ti_workdir)


@pytest.fixture(scope="module")
def monolith(ti_data):
    pb = Problem()
    build_flextool(pb, ti_data)
    sol = pb.solve()
    assert sol.optimal, "monolith solve not optimal"
    return sol


@pytest.fixture(scope="module")
def benders_result(ti_data, monolith):
    res = solve_benders(
        ti_data, _REGIONS, max_iters=20, tol=1e-4,
        monolith_objective=monolith.obj, master="flextool",
    )
    assert res.converged, f"Benders did not converge (gap={res.gap:.3e})"
    return res


def _nonzero(frame: pl.DataFrame, ent_col: str) -> dict[tuple, float]:
    """``{(entity, d): value}`` for the materially-nonzero rows."""
    out: dict[tuple, float] = {}
    for r in frame.iter_rows(named=True):
        v = float(r["value"])
        if abs(v) > 1e-9:
            out[(str(r[ent_col]), str(r["d"]))] = v
    return out


# ---------------------------------------------------------------------------
# (1) The result carries an invest_solution_vars dict with the right shape.
# ---------------------------------------------------------------------------


def test_invest_solution_vars_shape(benders_result) -> None:
    isv = benders_result.invest_solution_vars
    assert isinstance(isv, dict)
    # This fixture invests only in the two cross-region pipes (master) and
    # has no in-region unit/storage investment, so only ``v_invest_p`` is
    # present.  Each frame must carry the Solution.value(name) columns.
    assert "v_invest_p" in isv, f"missing v_invest_p (keys={list(isv)})"
    fr = isv["v_invest_p"]
    assert fr.columns == ["p", "d", "value"], (
        f"v_invest_p columns {fr.columns} != ['p','d','value'] "
        f"(must match Solution.value semantics)"
    )
    # Every present frame matches the (entity, d, value) contract.
    for name, frame in isv.items():
        assert frame.columns[-1] == "value"
        assert frame.columns[1] == "d"
        assert frame.height > 0, f"{name} present but empty"


# ---------------------------------------------------------------------------
# (2) The assembled invest reconciles to the monolith (correct units).
# ---------------------------------------------------------------------------


def test_invest_solution_vars_matches_monolith(benders_result, monolith) -> None:
    isv = benders_result.invest_solution_vars

    # MONOLITH ground truth: the nonzero v_invest_p cells (normalised units).
    mono_p = _nonzero(monolith.value("v_invest_p"), "p")
    assert mono_p, "monolith has no v_invest_p — fixture changed?"
    # Sanity: the only invested entities are the two cross-region pipes.
    assert {k[0] for k in mono_p} == _TRADE_CONNS, (
        f"monolith invest entities {set(k[0] for k in mono_p)} != "
        f"{_TRADE_CONNS}"
    )

    assembled_p = _nonzero(isv["v_invest_p"], "p")
    assert set(assembled_p) == set(mono_p), (
        f"assembled v_invest_p cells {set(assembled_p)} != monolith "
        f"{set(mono_p)}"
    )
    for cell, mono_v in mono_p.items():
        got = assembled_p[cell]
        assert abs(got - mono_v) <= 1e-4 * max(1.0, abs(mono_v)), (
            f"v_invest_p{cell}: assembled {got:.8e} != monolith "
            f"{mono_v:.8e} (units mismatch?)"
        )

    # The cross-region pipes are sourced from the MASTER and reconcile to the
    # ``invest`` dict (the master capacity C per connection, summed over
    # invest periods) — same normalisation, independent extraction path.
    for conn in _TRADE_CONNS:
        c = benders_result.invest[conn]
        s = sum(v for (e, _d), v in assembled_p.items() if e == conn)
        assert abs(s - c) <= 1e-6 * max(1.0, abs(c)), (
            f"assembled v_invest_p sum for {conn}={s:.8e} != master C "
            f"{c:.8e}"
        )

    # v_invest_n (node invest) — if the monolith invests in any node, the
    # assembled dict must carry the same owner-de-duplicated cells; this
    # fixture has none, so both must agree (empty).
    if "v_invest_n" in monolith._vars:
        mono_n = _nonzero(monolith.value("v_invest_n"), "n")
        asm_n = _nonzero(isv["v_invest_n"], "n") if "v_invest_n" in isv else {}
        assert set(asm_n) == set(mono_n), (
            f"assembled v_invest_n cells {set(asm_n)} != monolith {set(mono_n)}"
        )


# ---------------------------------------------------------------------------
# (3) Owner-de-duplication: each entity appears at most once across the
#     assembled frame (disjoint region/master partition).
# ---------------------------------------------------------------------------


def test_invest_solution_vars_owner_disjoint(benders_result) -> None:
    for name, frame in benders_result.invest_solution_vars.items():
        ent_col = frame.columns[0]
        keyed = frame.select(ent_col, "d")
        assert keyed.height == keyed.unique().height, (
            f"{name} has duplicate (entity, d) rows — owner-selection "
            f"failed to de-duplicate"
        )


# ---------------------------------------------------------------------------
# (4) progress_callback streams one dict per outer iteration.
# ---------------------------------------------------------------------------


def test_progress_callback_streams_per_iteration(ti_data, monolith) -> None:
    seen: list[dict] = []
    res = solve_benders(
        ti_data, _REGIONS, max_iters=20, tol=1e-4,
        monolith_objective=monolith.obj, master="flextool",
        progress_callback=seen.append,
    )
    assert res.converged
    # One callback per outer iteration up to (and including) convergence.
    assert len(seen) == res.iterations, (
        f"{len(seen)} callbacks != {res.iterations} iterations"
    )
    for i, d in enumerate(seen, start=1):
        assert d["iter"] == i
        for key in ("iter", "lower_bound", "upper_bound", "gap"):
            assert key in d, f"progress dict missing {key!r}"
    # The final callback reports convergence and a gap within tol; its
    # bounds match the returned result (REAL units).
    last = seen[-1]
    assert last["converged"] is True
    assert last["gap"] <= 1e-4
    assert abs(last["lower_bound"] - res.lower_bound) <= 1e-3 * max(
        1.0, abs(res.lower_bound)
    )
