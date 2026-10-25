"""Benders Phase-0 ground-truth fixture tests.

Exercises the ``lh2_three_region_trade_invest`` sibling fixture — a 2-day
(48h) three-region LH2 model whose cross-region pipes are GREENFIELD
investable (``existing=0`` + ``invest_total``).  Region A has cheap wind
on a deterministic 100/50/0 availability profile; region C is a
demand-heavy import sink (its local wind zeroed, coal capped) so the LP
strictly prefers to build pipes A→B→C and ship A's free-wind LH2 to C.

These tests pin the monolith ground truth:

* MONOLITH (``build_flextool`` + ``.solve()``): pipe_AB invested and A→B
  trade strictly positive.  Every electrolyser flow now consumes
  electricity at every hour (the daily h2_* / lh2_* nodes sum their hourly
  flows — before, 23 of 24 hours made hydrogen from nothing), so region A's
  surplus no longer reaches C: pipe_BC is not built, B→C trade is 0 and C's
  LH2 demand is served by slack (the fixture is under-sized; see
  ``tests/expected/REGEN_LOG.md``, c15b).

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
are in those normalised units (physical: pipe_AB built ≈ 15.7 MW,
delivering ≈ 753 MWh of LH2 to B over the 2 days).
"""
from __future__ import annotations

import polars as pl
import pytest

from polar_high import Problem

from flextool.engine import build_flextool, load_flextool


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


# ---------------------------------------------------------------------------
# (b) Monolithic ground truth: pipe invest > 0, trade > 0, C slack ≈ 0.
# ---------------------------------------------------------------------------


def test_monolith_builds_pipes_and_trades(ti_monolith) -> None:
    sol = ti_monolith
    assert sol.optimal

    inv_ab = _invest(sol, "pipe_AB")
    inv_bc = _invest(sol, "pipe_BC")
    assert inv_ab > 1e-3, f"pipe_AB not invested ({inv_ab})"
    assert inv_bc < 1e-9, f"pipe_BC invested ({inv_bc}): A's surplus no longer reaches C"

    # A→B trade strictly positive (normalised v_flow units); no B→C trade.
    f_ab = _arc_flow(sol, "pipe_AB", "lh2_A", "lh2_B")
    f_bc = _arc_flow(sol, "pipe_BC", "lh2_B", "lh2_C")
    assert f_ab > 1e-2, f"no A→B trade ({f_ab})"
    assert f_bc < 1e-9, f"B→C trade ({f_bc})"


def test_monolith_objective_finite(ti_monolith) -> None:
    M = ti_monolith.obj
    assert M > 0
    # Sanity band around the empirically verified M ≈ 1.2218e10.
    assert 1e9 < M < 1e11, f"objective M out of expected band: {M}"


