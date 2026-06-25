"""Benders (Option C) Phase 3a — FLEXTOOL-GENERATED master acceptance gate.

Phase 2 proved the HAND-built master converges to the monolith optimum.  Phase
3a SWAPS the hand master for a FlexTool-generated one
(:func:`flextool.engine_polars._region_filter.master_network_data` +
``build_flextool`` + appended ``η_r`` recourse columns), run through the SAME
multi-cut loop on the SAME no-RP 3-region prototype, so the known answer
isolates the reuse-master mechanics (the network-only producer, terminal
omission, native flow / invest / capacity / cost emit) from RP / scale.

GATE (spec ``benders_option_c.md`` §5 sub-stage 3a):

* converges (≤12 iters, tol 1e-4) to ``M = 8.5442474e9`` (rtol 1e-4);
* a VALID lower bound ``LB ≤ M·(1+1e-9)`` (the whole point vs the Lagrangian
  bug);
* recovered invest ``C`` and trade ``f̄`` match the monolith;
* near-PARITY with the Phase-2 hand-master result (the prototype's pipe flow
  cost is 0, so the FlexTool master's invest+flow objective equals the hand
  master's invest-only objective);
* the master ``build_flextool(reduced)`` SOLVE SUCCEEDS with the trade
  terminals OMITTED from balance (the §1.2 unbalanced-terminal claim, executed
  here for the first time).

The loop's Phase-2 self-checks (master kOptimal, LB monotone non-decreasing,
each appended cut SATISFIED at the new master point, finite boundary penalties,
LB ≤ M) run INSIDE ``solve_benders`` and raise on violation — so a green run is
itself the self-check assertion.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polar_high import Problem

from flextool.engine_polars import build_flextool, load_flextool
from flextool.engine_polars import _region_filter
from flextool.engine_polars._benders import solve_benders

_REGIONS = ["region_A", "region_B", "region_C"]
_M_EXPECTED = 8.5442474e9


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


def _arc_sum(sol, p, source, sink) -> float:
    f = sol.value("v_flow").filter(
        (pl.col("p") == p) & (pl.col("source") == source) & (pl.col("sink") == sink)
    )
    return float(f["value"].sum()) if f.height else 0.0


def _invest(sol, p) -> float:
    inv = sol.value("v_invest_p").filter(pl.col("p") == p)
    return float(inv["value"].sum()) if inv.height else 0.0


# ---------------------------------------------------------------------------
# (1) The network-only master producer builds a solvable master with the trade
#     terminals OMITTED from balance (the §1.2 unbalanced-terminal claim).
# ---------------------------------------------------------------------------


def test_master_network_data_builds_and_solves(ti_data) -> None:
    reduced = _region_filter.master_network_data(ti_data, _REGIONS)

    # INCLUDES exactly the cross arcs (pipe_AB / pipe_BC, both directions).
    pss = reduced.process_source_sink
    assert pss is not None and pss.height == 4, (
        f"expected 4 cross arcs, got {None if pss is None else pss.height}"
    )
    assert set(pss["p"].cast(pl.Utf8).unique().to_list()) == {"pipe_AB", "pipe_BC"}

    # OMITS the trade TERMINAL nodes (lh2_A/B/C) from balance.
    nb_nodes = set(reduced.nodeBalance["n"].cast(pl.Utf8).to_list())
    assert not (nb_nodes & {"lh2_A", "lh2_B", "lh2_C"}), (
        f"trade terminals not omitted from nodeBalance: {nb_nodes}"
    )
    # Storage / block / inflow node frames emptied.
    assert reduced.nodeStateBlock is None
    assert reduced.nodeState is None
    # KEEPS the invest set for the cross connections.
    assert set(reduced.pd_invest_set["p"].cast(pl.Utf8).to_list()) == {
        "pipe_AB", "pipe_BC"
    }

    # The §1.2 claim, EXECUTED: build_flextool succeeds on the reduced data
    # with unbalanced terminals; the master skeleton emits v_flow, v_invest_p.
    pb = Problem()
    build_flextool(pb, reduced)
    assert "v_flow" in pb._vars and "v_invest_p" in pb._vars
    sol = pb.solve()
    assert sol.optimal, "network-only master build_flextool solve not optimal"
    # No cost pressure yet (no recourse) → invest 0, flow 0, obj 0.
    assert abs(sol.obj) < 1e-3, f"cut-less master obj should be ~0, got {sol.obj}"


# ---------------------------------------------------------------------------
# (2) The FlexTool master converges to M with a VALID lower bound.
# ---------------------------------------------------------------------------


def test_flextool_master_converges_to_monolith(ti_data, monolith) -> None:
    M = monolith.obj
    assert np.isclose(M, _M_EXPECTED, rtol=1e-3), f"monolith M drifted: {M}"

    res = solve_benders(
        ti_data, _REGIONS, max_iters=20, tol=1e-4,
        monolith_objective=M, master="flextool",
    )

    assert res.converged, (
        f"FlexTool-master Benders did not converge: gap={res.gap:.3e} after "
        f"{res.iterations} iters (LB={res.lower_bound:.6e} "
        f"UB={res.upper_bound:.6e})"
    )
    assert res.iterations <= 12, f"too many iters: {res.iterations}"

    # total_objective (best UB) reconciles to the monolith optimum.
    assert np.isclose(res.total_objective, M, rtol=1e-4), (
        f"FlexTool-master UB {res.total_objective:.8e} != monolith M {M:.8e} "
        f"(LB={res.lower_bound:.8e}, gap={res.gap:.3e}, iters={res.iterations})"
    )

    # VALID lower bound: LB <= M (the whole point vs the Lagrangian bug).
    assert res.lower_bound <= M * (1 + 1e-9), (
        f"FlexTool-master LB {res.lower_bound:.8e} EXCEEDS M {M:.8e} — "
        f"invalid bound"
    )

    # Region recourse + master (invest + flow) cost reconciles to M.
    assert np.isclose(res.upper_bound, M, rtol=1e-4), (
        f"Σ cost_r + master trade cost = {res.upper_bound:.8e} != M {M:.8e}"
    )


# ---------------------------------------------------------------------------
# (3) Recovered pipe investment + A→B / B→C trade match the monolith.
# ---------------------------------------------------------------------------


def test_flextool_master_recovers_invest_and_trade(ti_data, monolith) -> None:
    C_ab_star = _invest(monolith, "pipe_AB")
    C_bc_star = _invest(monolith, "pipe_BC")
    f_ab_star = _arc_sum(monolith, "pipe_AB", "lh2_A", "lh2_B")
    f_bc_star = _arc_sum(monolith, "pipe_BC", "lh2_B", "lh2_C")

    res = solve_benders(
        ti_data, _REGIONS, max_iters=20, tol=1e-4,
        monolith_objective=monolith.obj, master="flextool",
    )
    assert res.converged

    C_ab = res.invest.get("pipe_AB", 0.0)
    C_bc = res.invest.get("pipe_BC", 0.0)
    assert C_ab > 1e-3 and C_bc > 1e-3, f"pipes not invested: {res.invest}"
    assert np.isclose(C_ab, C_ab_star, rtol=2e-2, atol=1e-3), (
        f"pipe_AB invest {C_ab} != monolith {C_ab_star}"
    )
    assert np.isclose(C_bc, C_bc_star, rtol=2e-2, atol=1e-3), (
        f"pipe_BC invest {C_bc} != monolith {C_bc_star}"
    )

    f_ab = float(res.trade_flow[("pipe_AB", "lh2_A", "lh2_B")]["value"].sum())
    f_bc = float(res.trade_flow[("pipe_BC", "lh2_B", "lh2_C")]["value"].sum())
    assert np.isclose(f_ab, f_ab_star, rtol=2e-2, atol=1e-3), (
        f"A→B trade {f_ab} != monolith {f_ab_star}"
    )
    assert np.isclose(f_bc, f_bc_star, rtol=2e-2, atol=1e-3), (
        f"B→C trade {f_bc} != monolith {f_bc_star}"
    )

    # Reverse arcs go to ~0 at the optimum (carried but not omitted).
    f_ba = float(res.trade_flow[("pipe_AB", "lh2_B", "lh2_A")]["value"].sum())
    f_cb = float(res.trade_flow[("pipe_BC", "lh2_C", "lh2_B")]["value"].sum())
    assert abs(f_ba) < 1e-3 and abs(f_cb) < 1e-3, (
        f"reverse trade not ~0: B→A={f_ba}, C→B={f_cb}"
    )


# ---------------------------------------------------------------------------
# (4) FlexTool master ≈ Phase-2 hand master (near-parity; prototype flow cost 0
#     ⇒ the invest+flow FlexTool objective equals the hand invest-only one).
# ---------------------------------------------------------------------------


def test_flextool_master_matches_hand_master(ti_data, monolith) -> None:
    M = monolith.obj
    hand = solve_benders(
        ti_data, _REGIONS, max_iters=20, tol=1e-4,
        monolith_objective=M, master="hand",
    )
    flex = solve_benders(
        ti_data, _REGIONS, max_iters=20, tol=1e-4,
        monolith_objective=M, master="flextool",
    )
    assert hand.converged and flex.converged

    # Both reconcile to M; LB / UB within a tight relative tolerance.
    assert np.isclose(hand.lower_bound, flex.lower_bound, rtol=1e-6), (
        f"LB mismatch: hand={hand.lower_bound:.10e} flex={flex.lower_bound:.10e}"
    )
    assert np.isclose(hand.upper_bound, flex.upper_bound, rtol=1e-6), (
        f"UB mismatch: hand={hand.upper_bound:.10e} flex={flex.upper_bound:.10e}"
    )
    for conn in ("pipe_AB", "pipe_BC"):
        assert np.isclose(
            hand.invest[conn], flex.invest[conn], rtol=1e-6, atol=1e-6
        ), (
            f"invest[{conn}] mismatch: hand={hand.invest[conn]} "
            f"flex={flex.invest[conn]}"
        )
