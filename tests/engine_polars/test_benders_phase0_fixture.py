"""Benders Phase-0 ground-truth fixture tests.

Exercises the ``lh2_three_region_trade_invest`` sibling fixture — a 2-day
(48h) three-region LH2 model whose cross-region pipes are GREENFIELD
investable (``existing=0`` + ``invest_total``).  Region A has cheap wind
on a deterministic 100/50/0 availability profile; region C is a
demand-heavy import sink (its local wind zeroed, coal capped) so the LP
strictly prefers to build pipes A→B→C and ship A's free-wind LH2 to C.

These tests pin the ground truth the brief requires:

* MONOLITH (``build_flextool`` + ``.solve()``): both pipes invested,
  A→B→C trade flow strictly positive, region-C up-slack ≈ 0.

The MONOLITH is the ground-truth optimum ``M`` that Benders must recover
(see ``test_benders_phase2_loop.py``).  The historical contrast — the old
subgradient Lagrangian collapsing trade to autarky ABOVE ``M`` — is no
longer asserted here: the subgradient driver was deleted in Phase 4
Chunk C.

See ``specs/benders_option_c.md`` and
``specs/benders_option_c_fixture_recipe.md``.

NOTE on units: ``v_flow`` / ``v_invest_p`` are normalised by
``p_unitsize`` (1000 for the pipes), so a v_flow value of ``0.1`` is
``0.1 × 1000 × step_duration`` MWh of delivered LH2.  Thresholds below
are in those normalised units (physical: pipe_BC delivers ≈ 5627 LH2 vs
C's 5760 demand over the 2 days; pipes built ≈ 101/117 MW).
"""
from __future__ import annotations

import polars as pl
import pytest

from polar_high import Problem

from flextool.engine_polars import build_flextool, load_flextool


@pytest.fixture(scope="module")
def ti_workdir(scenario_workdir):
    return scenario_workdir(
        "lh2_three_region_trade_invest", db_fixture="lh2_trade_invest"
    )


@pytest.fixture(scope="module")
def ti_data(ti_workdir):
    return load_flextool(ti_workdir)


@pytest.fixture(scope="module")
def ti_monolith(ti_data):
    """Solve the whole (undecomposed) fixture once and expose the
    solution + objective ``M``."""
    pb = Problem()
    build_flextool(pb, ti_data)
    sol = pb.solve()
    return sol


def _arc_flow(sol, p: str, source: str, sink: str) -> float:
    flow = sol.value("v_flow")
    f = flow.filter(
        (pl.col("p") == p)
        & (pl.col("source") == source)
        & (pl.col("sink") == sink)
    )
    return float(f["value"].sum()) if f.height else 0.0


def _invest(sol, p: str) -> float:
    inv = sol.value("v_invest_p")
    row = inv.filter(pl.col("p") == p)
    return float(row["value"].sum()) if row.height else 0.0


def _node_up_slack(sol, n: str) -> float:
    vq = sol.value("vq_state_up")
    s = vq.filter(pl.col("n") == n)
    return float(s["value"].sum()) if s.height else 0.0


# ---------------------------------------------------------------------------
# (b) Monolithic ground truth: pipe invest > 0, trade > 0, C slack ≈ 0.
# ---------------------------------------------------------------------------


def test_monolith_builds_pipes_and_trades(ti_monolith) -> None:
    sol = ti_monolith
    assert sol.optimal

    inv_ab = _invest(sol, "pipe_AB")
    inv_bc = _invest(sol, "pipe_BC")
    assert inv_ab > 1e-3, f"pipe_AB not invested ({inv_ab})"
    assert inv_bc > 1e-3, f"pipe_BC not invested ({inv_bc})"

    # A→B→C trade flow strictly positive (normalised v_flow units).
    f_ab = _arc_flow(sol, "pipe_AB", "lh2_A", "lh2_B")
    f_bc = _arc_flow(sol, "pipe_BC", "lh2_B", "lh2_C")
    assert f_ab > 1e-2, f"no A→B trade ({f_ab})"
    assert f_bc > 1e-2, f"no B→C trade ({f_bc})"

    # Region-C LH2 demand served by import: up-slack ≈ 0.
    slack_c = _node_up_slack(sol, "lh2_C")
    assert slack_c < 1e-3, f"lh2_C up-slack non-zero ({slack_c})"


def test_monolith_objective_finite(ti_monolith) -> None:
    M = ti_monolith.obj
    assert M > 0
    # Sanity band around the empirically verified M ≈ 8.544e9.
    assert 1e9 < M < 1e11, f"objective M out of expected band: {M}"


