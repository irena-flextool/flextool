"""In-memory carrier of state passed between solves.

Home of the canonical :class:`SolveHandoff` dataclass — the typed
record of "what one solve produced that the next solve(s) consume".
Built post-solve by :func:`flextool.engine_polars.input.build_handoff_from_solution`
directly from the polar_high ``Solution`` object; consumed by the next
sub-solve via the orchestrator's iteration-start translator
(:func:`_provider_translators.translate_handoff_to_provider`) which
fans each field into the Provider under a ``handoff/<field>`` key.

Phase 3 of ``specs/provider_consolidation.md`` retired the legacy
``capture_post_solve()`` disk-read constructor — the cascade was
already using ``build_handoff_from_solution`` exclusively; the
capture-from-disk function was dead code and has been removed,
together with the three carrier fields it was the only populator
for (``fix_storage_timesteps``, ``ed_history_realized_first``,
``edd_history``).  Consumers of those fields already fell through
to Provider/CSV reads when the carrier was ``None`` (the universal
state in the cascade path).

Carrier schemas (each is the on-disk equivalent's columns, renamed
for in-memory uniformity — string keys + ``value`` for single-value
carriers, named metric columns for multi-value):

    realized_invest        [entity, period, value]
    realized_existing      [entity, period, value]
    divest_cumulative      [entity, value]
    roll_end_state         [node, value]
    fix_storage_quantity   [node, period, step, p_fix_storage_quantity]
    fix_storage_price      [node, period, step, p_fix_storage_price]
    fix_storage_usage      [node, period, step, p_fix_storage_usage]
    cumulative_co2         [group, period, value]
    cumulative_commodity   [commodity, tier, period, p_ladder_cum_realized_mwh]
    cum_sim_hours          [period, p_ladder_cum_sim_hours]

Phase 4.1a moved ``cumulative_commodity`` and ``cum_sim_hours`` to their
canonical column names (matching the ``solve_data/`` Provider key
schemas) so the iteration-start handoff translator can route the
frames straight through to ``handoff/cumulative_commodity`` /
``handoff/cum_sim_hours`` without a per-iteration rename.

(Δ.1 — ``periods_already_emitted`` was previously listed here; it
moved to ``_output_writer.OutputWriterState`` since it gates writer-
side emission and isn't a true solver-handoff carrier.)

The three ``fix_storage_*`` narrow carriers replaced the legacy
single wide ``fix_storage`` field (retired in Phase 4.1l).  Each
metric is independent, so a separate per-metric carrier matches the
producer/consumer reality without NULL-padding columns.

``realized_invest`` and ``realized_existing`` together cover the two
columns of ``solve_data/p_entity_period_existing_capacity.csv``:
``realized_invest`` is what was *built this solve*, ``realized_existing``
is the resolved existing-capacity history (pre-existing decay +
divest enter this column and aren't reconstructible from
``realized_invest`` alone).
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl



@dataclass
class SolveHandoff:
    """Per-solve output → dependent-solve input carrier.

    Each field is a polars DataFrame in the schema documented in this
    module's docstring, or ``None`` when that carrier kind isn't active
    for this handoff.  Solves are identified by full solve name; the
    handoff represents the *output* of one solve becoming the *input*
    to its dependent (child / next-roll) solves.
    """

    # Realized invest *built this solve* in absolute units (post-unitsize).
    # Producer: any solve with v_invest > 0.
    # Consumer: subsequent solves' preprocessing.
    # File equivalent: ``p_entity_period_invested_capacity`` column of
    #   solve_data/p_entity_period_existing_capacity.csv.
    realized_invest: pl.DataFrame | None = None

    # Realized existing-capacity history per (entity, period) — captures
    # pre-existing decay + divest that ``realized_invest`` doesn't.
    # File equivalent: ``p_entity_period_existing_capacity`` column of
    #   solve_data/p_entity_period_existing_capacity.csv.
    realized_existing: pl.DataFrame | None = None

    # Cumulative divested capacity per entity (scalar, not per-period).
    # Carries pre-existing decay forward across solves.
    # File equivalent: solve_data/p_entity_divested.csv.
    divest_cumulative: pl.DataFrame | None = None

    # End-of-roll storage state for ``bind_forward_only`` carry-over.
    # Producer: prior roll's v_state at its last (d, t).
    # Consumer: next roll's nodeBalance_eq first-timestep term.
    # File equivalent: solve_data/p_roll_continue_state.csv.
    roll_end_state: pl.DataFrame | None = None

    # Upward feedback carrier: nested dispatch sub-solve's realized
    # end-of-horizon v_state routed UPWARD to its parent storage
    # solve's next roll.  When a dispatch sub-solve completes, the
    # parent storage solve's next roll prefers this realized state
    # over its own previously-predicted state — closing the loop
    # between the lower-information storage plan and the higher-
    # information dispatch result.  Always-on for any storage→dispatch
    # nesting; no opt-in flag (per specs/feature_fixes.md §1).
    # Schema mirrors ``roll_end_state``: ``[node, value]``.
    # Producer: dispatch sub-solve's roll_end_state (copy at handoff
    # capture; no separate v_state extraction needed).
    # Consumer: parent storage's continuation-roll
    # ``p_roll_continue_state`` (preferred over sequential prior when
    # available).
    # No file equivalent (in-memory only).
    upward_roll_end_state: pl.DataFrame | None = None

    # Storage quota imposed by parent solve on child solve — narrow
    # per-metric carriers in canonical column schema.  The trio is
    # independent (parent may set quantity-only, price-only, or any
    # combination); each metric travels in its own field.
    # Producer: parent solve's v_state + cost duals.
    # Consumer: child's fix_storage_* constraints.
    # File equivalents: solve_data/fix_storage_{quantity,price,usage}.csv
    # Each schema: ``[node, period, step, p_fix_storage_<metric>]``.
    fix_storage_quantity: pl.DataFrame | None = None
    fix_storage_price: pl.DataFrame | None = None
    fix_storage_usage: pl.DataFrame | None = None

    # Running CO2 totals carried across rolls for cumulative-cap constraint.
    # File equivalent: solve_data/co2_cum_realized_tonnes.csv.
    cumulative_co2: pl.DataFrame | None = None

    # Running per-tier commodity consumption for cumulative ladder pricing.
    # File equivalent: solve_data/commodity_ladder_cumulative.csv (per-tier mwh).
    cumulative_commodity: pl.DataFrame | None = None

    # Running simulated-hour total per period.  Independent 1-D
    # carrier shared by ladder + CO2-cap constraints.
    # File equivalent: solve_data/ladder_cum_sim_hours.csv.
    cum_sim_hours: pl.DataFrame | None = None

    _FIELDS = (
        "realized_invest", "realized_existing", "divest_cumulative",
        "roll_end_state", "upward_roll_end_state",
        "fix_storage_quantity", "fix_storage_price", "fix_storage_usage",
        "cumulative_co2", "cumulative_commodity", "cum_sim_hours",
    )

    def is_empty(self) -> bool:
        """True when no carrier is populated."""
        return all(getattr(self, f) is None for f in self._FIELDS)


# Phase 4.1i — ``write_fix_storage_files_from_handoff`` was retired
# once all readers of ``solve_data/fix_storage_*`` migrated to the
# per-metric ``handoff/*`` Provider keys seeded by the
# iteration-start translator (Phases 4.1f–4.1h).  The wide → narrow
# CSV fan-out has no consumers.


__all__ = [
    "SolveHandoff",
]
