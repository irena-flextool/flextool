"""End-to-end coverage for the Benders decomposition routing + TIER-1
invest handoff THROUGH THE ORCHESTRATOR (production wiring, not a direct
``solve_benders`` call).

Chunk B of the subgradient-removal plan wires the orchestrator's
per-solve decomposition routing onto ``solve_benders``: a solve that
resolves to ``decomposition=benders`` is run through
``_run_benders_solve`` (region discovery + Benders driver + live per-iter
LB/UB logging + the TIER-1 invest→dispatch handoff deposit).

This module drives the ``lh2_three_region_trade_invest`` scenario (a
single Benders solve, ``lh2_trade_invest``, that invests in the two
cross-region trade pipes ``pipe_AB`` / ``pipe_BC``) — the same
Benders-supported, invest-bearing fixture the DIRECT-call test
``test_benders_invest_handoff.py`` proves converges to the monolith.
Here we instead drive it through ``run_chain_from_db`` to prove the
PRODUCTION path:

1. region discovery resolves the three ``benders_regional`` groups,
2. the solve routes through ``solve_benders`` (the step is flagged
   ``is_benders`` and carries a ``SnapshotSolution`` invest carrier),
3. the live per-iteration LB/UB lines + final valid-bound summary reach
   stdout, and
4. the TIER-1 invest handoff is DEPOSITED non-empty (the invested pipes
   carry forward positive capacity in ``handoff.realized_invest``).

After the v62 rename the fixture's authored ``decomposition`` /
``decomposition_method`` migrate to ``benders`` / ``benders_regional`` at
build time, so the orchestrator routes ``lh2_trade_invest`` through the
Benders driver with no CLI flag.

The downstream dispatch-CONSUME leg (the full TIER-1 chain) is proven by
``test_invest_dispatch_chain_consumes_handoff_under_benders`` below on the
``lh2_three_region_invest`` scenario: that scenario invests in IN-REGION
wind units and TRADES over pipes of FIXED EXISTING capacity (NOT
investable).  The Benders master now handles existing-capacity trade
connections (Chunk A2: the over-strict invest-eligibility guard was
relaxed and the FlexTool ``maxFlow`` row bounds the pipe flow by
``existing/unitsize`` natively), so the orchestrator runs the decomposed
invest solve under Benders and the downstream monolithic dispatch
CONSUMES the handed-off in-region wind investment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest


FLEXTOOL_ROOT = Path(__file__).resolve().parents[2]
TI_FIXTURE_JSON = (
    FLEXTOOL_ROOT / "tests" / "fixtures" / "lh2_three_region_trade_invest.json"
)

TI_SCENARIO = "lh2_three_region_trade_invest"
# The two cross-region trade pipes the Benders master invests in.
TRADE_CONNS = {"pipe_AB", "pipe_BC"}

# The in-region-invest / existing-capacity-trade chain fixture.  Pipes are
# fixed existing capacity (``existing=50``, NOT investable); the regions
# invest in in-region wind.  The chain is ``[lh2_invest, lh2_dispatch]``.
LH2_FIXTURE_JSON = FLEXTOOL_ROOT / "tests" / "fixtures" / "lh2_three_region.json"
INVEST_SCENARIO = "lh2_three_region_invest"
INVEST_UNITS = {"wind_A", "wind_B", "wind_C"}


def _build_invest_db(tmp_path: Path) -> str:
    """Materialise the lh2 JSON fixture into a fresh SQLite + migrate
    (build-from-JSON per CLAUDE.md invariant 3).  The v62 migration renames
    the invest scenario's authored ``decomposition=lagrangian`` /
    ``lagrangian_region`` to ``benders`` / ``benders_regional``."""
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


def _build_trade_invest_db(tmp_path: Path) -> str:
    """Materialise the committed trade-invest JSON fixture into a fresh
    SQLite and migrate it to the current schema (build-from-JSON per
    CLAUDE.md invariant 3).  The migration renames the fixture's authored
    ``decomposition=lagrangian`` / ``lagrangian_region`` to
    ``benders`` / ``benders_regional`` (v62)."""
    if not TI_FIXTURE_JSON.exists():
        pytest.skip(f"trade-invest JSON fixture not present: {TI_FIXTURE_JSON}")
    tests_dir = FLEXTOOL_ROOT / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    from db_utils import json_to_db  # noqa: E402
    from flextool.update_flextool.db_migration import migrate_database

    db_path = tmp_path / "lh2_trade_invest.sqlite"
    url = json_to_db(TI_FIXTURE_JSON, db_path)
    migrate_database(url)
    return url


@pytest.mark.solver
def test_benders_routing_and_handoff_through_orchestrator(
    tmp_path, capsys,
) -> None:
    """``run_chain_from_db`` on the trade-invest scenario, routed through
    Benders by the orchestrator:

    1. completes without error;
    2. the solve was ROUTED through the Benders driver (its step is
       flagged ``is_benders`` and carries a ``SnapshotSolution`` invest
       carrier);
    3. the live ``[benders`` per-iter LB/UB lines + final valid-bound
       summary reached stdout (routing + logging wired);
    4. the TIER-1 invest handoff is DEPOSITED non-empty — the invested
       trade pipes carry forward positive capacity (handoff deposit wired).
    """
    from flextool.engine_polars import run_chain_from_db
    from flextool.engine_polars._orchestration import SnapshotSolution

    url = _build_trade_invest_db(tmp_path)
    work_folder = tmp_path / "work"
    work_folder.mkdir()

    steps = run_chain_from_db(
        url, TI_SCENARIO, work_folder=work_folder, keep_solutions=True,
    )
    assert steps, "run_chain_from_db produced no orchestration steps"

    # Single solve in this scenario (``lh2_trade_invest``).
    key = next((k for k in steps if "lh2_trade_invest" in k), None)
    assert key is not None, f"no lh2_trade_invest step in {list(steps)}"
    step = steps[key]

    # --- (2) routed through the Benders driver --------------------------
    assert step.is_benders is True, (
        "step must be flagged is_benders — decomposition=benders routing "
        "did not fire through the orchestrator"
    )
    assert isinstance(step.solution, SnapshotSolution), (
        "Benders step should carry a SnapshotSolution invest carrier, "
        f"got {type(step.solution).__name__}"
    )
    assert step.optimal in (True, False), (
        f"expected a bool convergence flag, got {step.optimal!r}"
    )
    assert step.obj is not None, "Benders step has no objective"

    # --- (3) routing + logging reached stdout ---------------------------
    out = capsys.readouterr().out
    assert "[benders" in out, (
        f"no Benders progress lines in stdout — routing/logging not "
        f"wired:\n{out}"
    )
    assert "start:" in out and "regions" in out, (
        f"Benders start banner missing:\n{out}"
    )
    assert "LB=" in out and "UB=" in out, (
        f"Benders live LB/UB per-iter lines missing:\n{out}"
    )
    assert "lower bound (valid)" in out, (
        f"final valid-lower-bound summary missing:\n{out}"
    )

    # --- (4) the TIER-1 invest handoff was deposited non-empty ----------
    assert step.handoff is not None, (
        "step has no handoff — orchestrator deposit did not fire"
    )
    assert not step.handoff.is_empty(), (
        "step handoff is empty despite a real investing Benders solve"
    )
    ri = step.handoff.realized_invest
    assert ri is not None and ri.height > 0, (
        "step handoff.realized_invest is empty/None"
    )
    val_col = ri.columns[-1]
    ent_col = ri.columns[0]
    invested = ri.filter(pl.col(val_col) > 1.0)
    assert invested.height >= 1, (
        f"invest handoff carries no positive capacity:\n{ri}"
    )
    # The invested entities include the cross-region trade pipes (the
    # master's owned invest, handed forward through the assembled
    # invest_solution_vars).
    invested_ents = set(invested[ent_col].to_list())
    assert invested_ents & TRADE_CONNS, (
        f"expected invested trade pipes {TRADE_CONNS}, got "
        f"{sorted(invested_ents)}"
    )


@pytest.mark.solver
def test_invest_dispatch_chain_consumes_handoff_under_benders(tmp_path) -> None:
    """Full TIER-1 invest→dispatch CONSUME chain, under Benders, over
    FIXED-CAPACITY trade pipes (the scenario the old master guard blocked).

    ``run_chain_from_db`` on ``lh2_three_region_invest`` (chain
    ``[lh2_invest, lh2_dispatch]``):

    1. completes both solves without error;
    2. the ``lh2_invest`` solve ROUTES through Benders (``is_benders``,
       ``SnapshotSolution`` carrier) and DEPOSITS a non-empty handoff with
       positive in-region wind capacity (the master handles the
       existing-capacity pipes natively);
    3. the downstream ``lh2_dispatch`` monolithic solve completes
       (real ``Solution``, optimal);
    4. the dispatch CONSUMES the handed-off investment — its FlexData
       carries positive ``p_entity_invested`` for the invested wind units.
    """
    from flextool.engine_polars import run_chain_from_db
    from flextool.engine_polars._orchestration import SnapshotSolution

    url = _build_invest_db(tmp_path)
    work_folder = tmp_path / "work"
    work_folder.mkdir()

    steps = run_chain_from_db(
        url, INVEST_SCENARIO, work_folder=work_folder, keep_solutions=True,
    )
    assert steps, "run_chain_from_db produced no orchestration steps"

    keys = list(steps.keys())
    invest_key = next((k for k in keys if "lh2_invest" in k), None)
    dispatch_key = next((k for k in keys if "lh2_dispatch" in k), None)
    assert invest_key is not None, f"no lh2_invest step in {keys}"
    assert dispatch_key is not None, f"no lh2_dispatch step in {keys}"

    # --- (2) the Benders invest step deposited a non-empty handoff ---------
    invest_step = steps[invest_key]
    assert invest_step.is_benders is True, (
        "invest step must be flagged is_benders — decomposition=benders "
        "routing did not fire"
    )
    assert isinstance(invest_step.solution, SnapshotSolution), (
        "Benders invest step should carry a SnapshotSolution invest carrier, "
        f"got {type(invest_step.solution).__name__}"
    )
    assert invest_step.handoff is not None, (
        "invest step has no handoff — orchestrator deposit did not fire"
    )
    assert not invest_step.handoff.is_empty(), (
        "invest step handoff is empty despite a real investing Benders solve"
    )
    ri = invest_step.handoff.realized_invest
    assert ri is not None and ri.height > 0, (
        "invest step handoff.realized_invest is empty/None"
    )
    val_col = ri.columns[-1]
    ent_col = ri.columns[0]
    invested = ri.filter(pl.col(val_col) > 1.0)
    assert invested.height >= 1, (
        f"invest handoff carries no positive capacity:\n{ri}"
    )
    # The invested entities are the in-region wind units, spanning ≥2 regions.
    invested_ents = set(invested[ent_col].to_list())
    assert invested_ents & INVEST_UNITS, (
        f"expected invested wind units, got {sorted(invested_ents)}"
    )

    # --- (3) the dispatch step completed with a real Solution --------------
    dispatch_step = steps[dispatch_key]
    assert dispatch_step.solution is not None, "dispatch produced no solution"
    assert not isinstance(dispatch_step.solution, SnapshotSolution), (
        "dispatch should carry a real Solution, not a SnapshotSolution"
    )
    assert dispatch_step.optimal, "dispatch solve not optimal"

    # --- (4) the dispatch CONSUMED the invested capacity -------------------
    fd = dispatch_step.flex_data
    pei = getattr(fd, "p_entity_invested", None)
    assert pei is not None, (
        "dispatch FlexData has no p_entity_invested — the Benders handoff "
        "overlay did not reach the dispatch solve"
    )
    pei_frame = pei.frame if hasattr(pei, "frame") else pei
    pei_val_col = pei_frame.columns[-1]
    pei_ent_col = pei_frame.columns[0]
    dispatch_invested = pei_frame.filter(
        (pl.col(pei_ent_col).is_in(list(INVEST_UNITS)))
        & (pl.col(pei_val_col) > 1.0)
    )
    assert dispatch_invested.height >= 1, (
        "dispatch p_entity_invested has no positive capacity for the "
        f"Benders-invested wind units:\n{pei_frame}"
    )
