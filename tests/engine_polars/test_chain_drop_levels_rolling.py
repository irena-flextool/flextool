"""Diagnostic for drop-levels uniqueness through ``run_chain`` +
``apply_handoff``.

Mirrors flextool's ``test_drop_levels_rolling.py`` (a regression test
for ``Index._join_level on non-unique index is not implemented`` raised
when N consecutive rolls each carry their own ``(solve, entity,
period)`` row into ``ed_invest`` — dropping the ``solve`` level leaves
heavily non-unique indices that crash flextool's downstream MultiIndex
joins).

The polar_high analogue: when a chain runs with realised invest in
many sub-solves, the in-memory ``SolveHandoff.realized_invest``
carriers emitted for each sub-solve must have UNIQUE (entity, period)
keys.  The chain-runner accumulator (``build_handoff_from_solution``
reading ``prior_handoff`` and folding in this-solve's contributions)
is the polar_high code path that, if buggy, would produce per-solve
duplicates analogous to flextool's ``ed_invest`` non-uniqueness.

polar_high doesn't have a ``drop_levels`` post-processor (its outputs
flow through polars frames keyed by stripped dim columns from the
start, so the bug class flextool's test pins is structurally avoided),
but the equivalent uniqueness invariant ON the handoff carriers is
testable and worth pinning as a regression.

Carrier exercised: chain-cumulative invest on a many-period fixture.

Note: per the audit (B3), if this test exercised a feature polar_high
doesn't model, it would be an A-class gap.  The bug class
flextool's drop_levels guards against (per-roll duplicate-index
non-uniqueness) IS structurally absent in polar_high's design — so
this test is the right shape for polar_high's plumbing rather than a
direct port.
"""
from __future__ import annotations

import pytest

from flextool.engine_polars import run_chain_from_db

pytestmark = pytest.mark.solver


# Multi-period fixtures that realise invest in multiple sub-solves —
# the structural shape flextool's drop_levels test guards.  4-solve
# lifetime-renew is the canonical case in the suite (4 distinct
# ``period`` realisations across 4 solves; each solve adds rows from
# its own period to the cumulative carrier).
SCENARIOS = (
    "wind_battery_invest_lifetime_renew_4solve",
    "multi_year",
)


def test_chain_realized_invest_handoff_is_unique_across_rolling_solves(
    scenario_workdir,
) -> None:
    """For every multi-solve invest fixture and every sub-solve, the
    in-memory handoff's ``realized_invest`` and ``realized_existing``
    carriers must have unique (entity, period) keys.

    Flextool's drop_levels test asserts this for the on-disk
    ``ed_invest.csv`` ladder after dropping the ``solve`` level.
    The polar_high chain-runner already strips the solve dim during
    aggregation (see ``build_handoff_from_solution`` — uses
    ``invest_by_ed[(entity, period)]`` keying); this test pins down
    that the equivalent invariant holds on the IN-MEMORY carriers
    polar_high actually produces.

    Δ.12e — the native cascade always threads handoff in-memory between
    solves (the legacy ``use_handoff_overlay`` knob retired with the
    file-symlink driver), so the equivalence between "cold" and
    "overlay" modes the original test asserted is now structural: there
    is only one mode.
    """
    for scenario_name in SCENARIOS:
        work = scenario_workdir(scenario_name)
        db_path = work / "tests.sqlite"
        sols = run_chain_from_db(db_path, scenario_name=scenario_name)

        for sub, step in sols.items():
            h = step.handoff

            ri = h.realized_invest
            if ri is not None:
                assert ri.unique(["entity", "period"]).height == ri.height, (
                    f"{scenario_name}/{sub}: realized_invest has "
                    f"duplicate (entity, period) rows — equivalent of "
                    f"flextool's drop_levels non-unique-index bug.\n"
                    f"{ri}"
                )
                # Levels are exactly (entity, period) — no leftover
                # solve dim (analogue of "droplevel('solve')").
                assert set(ri.columns) >= {"entity", "period", "value"}, (
                    f"{scenario_name}/{sub}: realized_invest "
                    f"columns unexpected: {ri.columns}")

            re = h.realized_existing
            if re is not None:
                assert re.unique(["entity", "period"]).height == re.height, (
                    f"{scenario_name}/{sub}: realized_existing "
                    f"has duplicate (entity, period) rows.\n{re}"
                )

            # divest_cumulative: keyed by entity only; uniqueness
            # is the analogue of flextool's d_realize_invest dedup
            # (one row per entity, never per-solve duplicates).
            dc = h.divest_cumulative
            if dc is not None:
                assert dc.unique(["entity"]).height == dc.height, (
                    f"{scenario_name}/{sub}: divest_cumulative "
                    f"has duplicate entity rows.\n{dc}"
                )
