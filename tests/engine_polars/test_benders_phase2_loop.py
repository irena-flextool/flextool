"""Benders (Option C) Phase 2 — master + multi-cut loop acceptance gate.

Proves the hand-built persistent Benders master + the multi-cut loop
(``flextool.engine_polars._benders.solve_benders``) converges to the MONOLITH
optimum ``M`` on the prototype fixture ``lh2_three_region_trade_invest`` with a
VALID lower bound — the key contrast with the Lagrangian bug, whose autarkic
"bound" 9.071e9 sits ABOVE the true optimum M=8.544e9.

See ``specs/benders_option_c.md`` (Phase 2 design + revised + critique) and the
verified Phase-1 pin/dual contract in ``test_benders_phase1_dual.py``.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polar_high import Problem

from flextool.engine_polars import build_flextool, load_flextool
from flextool.engine_polars._benders import solve_benders
from flextool.engine_polars._lagrangian import solve_lagrangian

_REGIONS = ["region_A", "region_B", "region_C"]
_M_EXPECTED = 8.544247e9


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
    """Solve the whole undecomposed fixture once; expose M, trade f*, invest
    C*."""
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
# (1)+(2) Benders converges to M with a VALID lower bound.
# ---------------------------------------------------------------------------


def test_benders_converges_to_monolith(ti_data, monolith) -> None:
    M = monolith.obj
    assert np.isclose(M, _M_EXPECTED, rtol=1e-3), f"monolith M drifted: {M}"

    res = solve_benders(
        ti_data, _REGIONS, max_iters=20, tol=1e-4, monolith_objective=M
    )

    # Convergence in a small iteration count.
    assert res.converged, (
        f"Benders did not converge: gap={res.gap:.3e} after {res.iterations} "
        f"iters (LB={res.lower_bound:.6e} UB={res.upper_bound:.6e})"
    )
    assert res.iterations <= 12, f"too many iters: {res.iterations}"

    # total_objective (best UB) reconciles to the monolith optimum.
    assert np.isclose(res.total_objective, M, rtol=1e-4), (
        f"Benders UB {res.total_objective:.8e} != monolith M {M:.8e} "
        f"(LB={res.lower_bound:.8e}, gap={res.gap:.3e}, iters={res.iterations})"
    )

    # VALID lower bound: LB <= M (the whole point vs the Lagrangian bug).
    assert res.lower_bound <= M * (1 + 1e-9), (
        f"Benders LB {res.lower_bound:.8e} EXCEEDS M {M:.8e} — invalid bound"
    )

    # Region recourse + master invest reconciles to M (UB restated).
    recon = res.upper_bound
    assert np.isclose(recon, M, rtol=1e-4), (
        f"Σ cost_r + master invest = {recon:.8e} != M {M:.8e}"
    )


# ---------------------------------------------------------------------------
# (2b) Recovered pipe investment + A→C trade match the monolith.
# ---------------------------------------------------------------------------


def test_benders_recovers_invest_and_trade(ti_data, monolith) -> None:
    C_ab_star = _invest(monolith, "pipe_AB")
    C_bc_star = _invest(monolith, "pipe_BC")
    f_ab_star = _arc_sum(monolith, "pipe_AB", "lh2_A", "lh2_B")
    f_bc_star = _arc_sum(monolith, "pipe_BC", "lh2_B", "lh2_C")

    res = solve_benders(
        ti_data, _REGIONS, max_iters=20, tol=1e-4, monolith_objective=monolith.obj
    )
    assert res.converged

    # Recovered invested capacity C ≈ C* (invest is coarser-grained → rtol 1e-2).
    C_ab = res.invest.get("pipe_AB", 0.0)
    C_bc = res.invest.get("pipe_BC", 0.0)
    assert C_ab > 1e-3 and C_bc > 1e-3, f"pipes not invested: {res.invest}"
    assert np.isclose(C_ab, C_ab_star, rtol=2e-2, atol=1e-3), (
        f"pipe_AB invest {C_ab} != monolith {C_ab_star}"
    )
    assert np.isclose(C_bc, C_bc_star, rtol=2e-2, atol=1e-3), (
        f"pipe_BC invest {C_bc} != monolith {C_bc_star}"
    )

    # Recovered forward trade f̄ ≈ f* (sum over the daily grid; rtol 2e-2).
    f_ab = float(res.trade_flow[("pipe_AB", "lh2_A", "lh2_B")]["value"].sum())
    f_bc = float(res.trade_flow[("pipe_BC", "lh2_B", "lh2_C")]["value"].sum())
    assert np.isclose(f_ab, f_ab_star, rtol=2e-2, atol=1e-3), (
        f"A→B trade {f_ab} != monolith {f_ab_star}"
    )
    assert np.isclose(f_bc, f_bc_star, rtol=2e-2, atol=1e-3), (
        f"B→C trade {f_bc} != monolith {f_bc_star}"
    )


# ---------------------------------------------------------------------------
# (3) The CURRENT Lagrangian gives the autarkic ~9.071e9 > M (what Benders fixes).
# ---------------------------------------------------------------------------


def test_lagrangian_is_autarkic_above_monolith(ti_data, ti_workdir, monolith) -> None:
    M = monolith.obj
    lag = solve_lagrangian(
        ti_data, work_dir=ti_workdir, alpha=10.0, max_iters=100, tol=0.5,
        initial_lambda=0.0, min_iters=20,
    )
    # The autarkic Lagrangian total sits clearly ABOVE the true optimum M
    # (weak-duality violation — the bug Benders fixes).
    assert lag.total_objective > M * (1 + 0.01), (
        f"Lagrangian total {lag.total_objective:.6e} did NOT exceed M "
        f"{M:.6e} — the no-trade bug is not reproduced"
    )
    assert np.isclose(lag.total_objective, 9.071132e9, rtol=1e-2), (
        f"Lagrangian autarkic total drifted from ~9.071e9: "
        f"{lag.total_objective:.6e}"
    )
