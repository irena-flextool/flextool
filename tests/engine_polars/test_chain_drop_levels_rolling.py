"""Diagnostic for drop-levels uniqueness through ``run_chain`` +
``apply_handoff``.

Mirrors flextool's ``test_drop_levels_rolling.py`` (a regression test
for ``Index._join_level on non-unique index is not implemented`` raised
when N consecutive rolls each carry their own ``(solve, entity,
period)`` row into ``ed_invest`` — dropping the ``solve`` level leaves
heavily non-unique indices that crash flextool's downstream MultiIndex
joins).

flexpy's analogue: when a chain runs with realised invest in many
sub-solves, the in-memory ``SolveHandoff.realized_invest`` carriers
emitted for each sub-solve must have UNIQUE (entity, period) keys.
The chain-runner accumulator (``build_handoff_from_flexpy`` reading
``prior_handoff`` and folding in this-solve's contributions) is the
flexpy code path that, if buggy, would produce per-solve duplicates
analogous to flextool's ``ed_invest`` non-uniqueness.

flexpy doesn't have a ``drop_levels`` post-processor (its outputs
flow through polars frames keyed by stripped dim columns from the
start, so the bug class flextool's test pins is structurally avoided),
but the equivalent uniqueness invariant ON the handoff carriers is
testable and worth pinning as a regression.

Carrier exercised: chain-cumulative invest on a many-period fixture.

Note: per the audit (B3), if this test exercised a feature flexpy
doesn't model, it would be an A-class gap.  The bug class
flextool's drop_levels guards against (per-roll duplicate-index
non-uniqueness) IS structurally absent in flexpy's design — so
this test is the right shape for flexpy's plumbing rather than a
direct port.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from flextool.engine_polars import run_chain

pytestmark = pytest.mark.solver


DATA = Path(__file__).resolve().parent / "data"


# Multi-period fixtures that realise invest in multiple sub-solves —
# the structural shape flextool's drop_levels test guards.  4-solve
# lifetime-renew is the canonical case in the suite (4 distinct
# ``period`` realisations across 4 solves; each solve adds rows from
# its own period to the cumulative carrier).
SCENARIOS = (
    "work_wind_battery_invest_lifetime_renew_4solve",
    "work_multi_year",
)


def test_chain_realized_invest_handoff_is_unique_across_rolling_solves() -> None:
    """For every multi-solve invest fixture and every sub-solve, the
    in-memory handoff's ``realized_invest`` and ``realized_existing``
    carriers must have unique (entity, period) keys.

    Flextool's drop_levels test asserts this for the on-disk
    ``ed_invest.csv`` ladder after dropping the ``solve`` level.
    flexpy's chain-runner already strips the solve dim during
    aggregation (see ``build_handoff_from_flexpy`` — uses
    ``invest_by_ed[(entity, period)]`` keying); this test pins down
    that the equivalent invariant holds on the IN-MEMORY carriers
    flexpy actually produces.

    Also asserts the ``apply_handoff`` round-trip preserves uniqueness:
    feeding solve N's handoff into solve N+1 via the overlay must not
    introduce duplicate (e, d) rows in the post-overlay FlexData.
    """
    available = [s for s in SCENARIOS if (DATA / s).exists()]
    if not available:
        pytest.skip("no rolling-invest fixtures available")

    for scenario in available:
        work = DATA / scenario
        # Run with default mode (snapshot CSVs as source of truth).
        sols_cold = run_chain(work)
        # Run with handoff-overlay mode (in-memory carriers fully
        # drive solve N+1's prebuilt invest state).  The overlay
        # path is what flextool's drop_levels test is structurally
        # analogous to — flexpy's apply_handoff has to deduplicate
        # the (e, d) → realized_invest map on its way into the
        # next sub-solve's FlexData.
        sols_overlay = run_chain(work, use_handoff_overlay=True)

        for mode_name, sols in (("cold", sols_cold),
                                  ("overlay", sols_overlay)):
            for sub, step in sols.items():
                h = step.handoff

                ri = h.realized_invest
                if ri is not None:
                    assert ri.unique(["entity", "period"]).height == ri.height, (
                        f"{scenario}/{sub}/{mode_name}: realized_invest has "
                        f"duplicate (entity, period) rows — equivalent of "
                        f"flextool's drop_levels non-unique-index bug.\n"
                        f"{ri}"
                    )
                    # Levels are exactly (entity, period) — no leftover
                    # solve dim (analogue of "droplevel('solve')").
                    assert set(ri.columns) >= {"entity", "period", "value"}, (
                        f"{scenario}/{sub}/{mode_name}: realized_invest "
                        f"columns unexpected: {ri.columns}")

                re = h.realized_existing
                if re is not None:
                    assert re.unique(["entity", "period"]).height == re.height, (
                        f"{scenario}/{sub}/{mode_name}: realized_existing "
                        f"has duplicate (entity, period) rows.\n{re}"
                    )

                # divest_cumulative: keyed by entity only; uniqueness
                # is the analogue of flextool's d_realize_invest dedup
                # (one row per entity, never per-solve duplicates).
                dc = h.divest_cumulative
                if dc is not None:
                    assert dc.unique(["entity"]).height == dc.height, (
                        f"{scenario}/{sub}/{mode_name}: divest_cumulative "
                        f"has duplicate entity rows.\n{dc}"
                    )
