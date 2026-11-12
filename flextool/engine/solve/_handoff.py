"""In-memory carrier of state passed between solves.

Home of the canonical :class:`SolveHandoff` dataclass — the typed
record of "what one solve produced that the next solve(s) consume".
Built post-solve by :func:`flextool.engine.preprocess.flex_data.build_handoff_from_solution`
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
    realized_commission    [entity, period, commission_period, value]
    divest_cumulative      [entity, value]
    realized_divest        [entity, period, value]
    roll_end_state         [node, value]
    roll_end_online        [process, value]
    roll_end_flow          [process, source, sink, period, step, duration, value]
    roll_end_step          [period, step, duration]  (+ end_rank, end_root, end_index, end_branch once routed)
    online_history         [process, period, step, startup, shutdown, duration, hours_to_end]
    fix_storage_quantity   [node, period, step, p_fix_storage_quantity]
    fix_storage_price      [node, period, step, p_fix_storage_price]
    fix_storage_usage      [node, period, step, p_fix_storage_usage]
    cumulative_co2         [group, period, value, represented]
    cumulative_commodity   [commodity, tier, period, p_ladder_cum_realized_mwh]
    cum_sim_hours          [period, p_ladder_cum_sim_hours]

Phase 4.1a moved ``cumulative_commodity`` and ``cum_sim_hours`` to their
canonical column names (matching the ``solve_data/`` Provider key
schemas) so the iteration-start handoff translator can route the
frames straight through to ``handoff/cumulative_commodity`` /
``handoff/cum_sim_hours`` without a per-iteration rename.

(Δ.1 — ``periods_already_emitted`` was previously listed here; it
moved to ``_writer.OutputWriterState`` since it gates writer-
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

    # Realized investments with a construction lead time, chain-cumulative: decision period, the period the
    # capacity starts to operate in (found by the deciding solve on the run calendar) and the capacity
    # (post-unitsize).  ``realized_invest`` / ``realized_existing`` key these by the start of operation; this
    # carrier keeps the decision period, so a later solve knows what comes online and what is in construction in
    # its periods.  Producer: ``build_handoff_from_solution``.  Consumer: ``apply_existing_chain``
    # (``p_entity_carried_commission`` / ``p_entity_carried_in_construction``).  In-memory only.
    realized_commission: pl.DataFrame | None = None

    # Cumulative divested capacity per entity (scalar, not per-period).
    # Carries pre-existing decay forward across solves.
    # File equivalent: solve_data/p_entity_divested.csv.
    divest_cumulative: pl.DataFrame | None = None

    # Capacity retired (post-unitsize) per realized, non-branch decision period, chain-cumulative.  A later solve
    # subtracts a retirement from the periods it still counts in (from its decision period until the end of life
    # of the entity's pre-existing capacity), not from every period.  Producer: ``build_handoff_from_solution``.
    # Consumers: ``apply_existing_chain`` (``p_entity_retired_carried``) and the emitted
    # ``p_entity_all_existing``.
    realized_divest: pl.DataFrame | None = None

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

    # End-of-roll committed online commitment for rolling unit-commitment
    # carry-over — the online analog of ``roll_end_state``.  Carries the
    # realized committed-jump-end ``v_online`` (unit-count space, NOT
    # ×unitsize) per online process so the next roll's ``online_change``
    # first-step row (``dt_online_carry``) sees the
    # prior roll's commitment as its predecessor instead of the
    # meaningless within-roll cyclic wrap.  Without it the startup cost is
    # re-charged at every roll boundary even when the unit stays committed.
    # Producer: prior roll's v_online_{linear,integer} at its last
    # realized (d, t).
    # Consumer: next roll's ``p_roll_continue_online`` overlay (applied at
    # the continuation roll's first step by ``_online.add_constraints``).
    # Schema: ``[process, value]``.  No file equivalent (in-memory only).
    roll_end_online: pl.DataFrame | None = None

    # End-of-roll flow for the ramp carry-in: v_flow (units of unitsize) of every ramp-limited arc at the
    # roll's last realized step, with that step and its duration.  The next roll's first step of a period
    # ramps from it when that step precedes it, adjacent, in the un-rolled solve (``_load_ramp_carry``).
    # Producer: ``flex_data._roll_end_flow``.  Consumer: next roll's ``p_roll_continue_flow`` /
    # ``dt_ramp_carry`` (``model.core._eq_ramp``).  In-memory only.
    # The carry is keyed on ``roll_end_step``: an override of ``roll_end_flow`` without ``roll_end_step`` is
    # ignored, and one whose step differs from ``roll_end_step`` raises (``params._ramp_links.ramp_roll_carry``).
    roll_end_flow: pl.DataFrame | None = None

    # The roll's last realized step in time (with its duration): the step the next roll's first step ramps
    # from (``_load_ramp_carry`` → ``dt_ramp_carry``, also read by the ramp output without ramp limits).
    # Producer: ``flex_data._last_realized_step``.  In-memory only.  The carry routing
    # (``solve._carry_routing.with_end_position``) adds where the step ends on the base timeline (``end_rank``,
    # ``end_root``, ``end_index``, ``end_branch``), so a solve with other step lengths continues it by position; an
    # override without those columns is matched by the step name only.
    roll_end_step: pl.DataFrame | None = None

    # The realized ``v_startup`` / ``v_shutdown`` per ``(process, period, step)`` that a minimum up / down time
    # still covers after this solve: the last realized step and its predecessors on the timeline (along
    # ``dt_ramp_link``), each with ``hours_to_end`` = hours from the start of the step to the end of the last
    # realized step, kept while ``hours_to_end <`` the process's authored max min-up/down time.  Inside a
    # forecasts-only fan the steps are those of the tied forecast.  The history this solve started from is extended
    # when the solve continued from it.  Consumer: the next solve's ``uptime_carry`` / ``downtime_carry``
    # (``_derived.online_mintime_carry_from_source``) at the steps that follow its ``dt_ramp_carry`` (the first steps
    # that continue from this solve's last realized step); ``period`` / ``step`` are for inspection only.  Routed
    # together with ``roll_end_step`` (solve._carry_routing); the consumer raises when the history does not end at
    # the carried step.  Schema: ``[process, period, step, startup, shutdown, duration, hours_to_end]``.
    # No file equivalent (in-memory only).
    online_history: pl.DataFrame | None = None

    # OC1 #1 sub-slice (b) — a separate_dispatch PARENT's committed v_online
    # over the JUMP (the whole realized window), pinned into the realized-
    # dispatch SIBLING so the sibling dispatches realized conditions under
    # the plan's commitment instead of re-deciding it from scratch.  Unlike
    # ``roll_end_online`` (a single jump-end scalar per process for the next
    # ROLL), this carries the FULL per-(process, period, step) committed
    # schedule for the sibling of the SAME solve.  Read from a FORECAST
    # branch (the NA-tied, well-defined value), NOT the degenerate weight-0
    # realized branch.  Unit-count space (NOT ×unitsize).
    # Producer: parent's committed v_online_{linear,integer} over the jump.
    # Consumer: sibling's ``p_fix_online`` overlay (``online_fix_eq``).
    # Schema: ``[process, period, step, value]``.  No file equivalent.
    fix_online: pl.DataFrame | None = None

    # The reserve counterpart of ``fix_online``: the parent's v_reserve on
    # its tied steps (reference forecast, re-keyed to the anchor period),
    # pinned into the realization solve.  Unit-count space (NOT ×unitsize).
    # Producer: parent's v_reserve.  Consumer: realization solve's
    # ``p_fix_reserve`` (``reserve_fix_eq``).
    # Schema: ``[process, reserve, upDown, node, period, step, value]``.
    fix_reserve: pl.DataFrame | None = None

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
    # File equivalent: solve_data/co2_cum_realized_tonnes.csv.  ``value`` is physical tonnes;
    # ``represented`` the same tonnes × years_represented / period_share of the realizing solve.
    cumulative_co2: pl.DataFrame | None = None

    # Running per-tier commodity consumption for cumulative ladder pricing.
    # File equivalent: solve_data/commodity_ladder_cumulative.csv (per-tier mwh).
    cumulative_commodity: pl.DataFrame | None = None

    # Running simulated-hour total per period.  Independent 1-D
    # carrier shared by ladder + CO2-cap constraints.
    # File equivalent: solve_data/ladder_cum_sim_hours.csv.
    cum_sim_hours: pl.DataFrame | None = None

    # The producing solve's ``reduce_unit_detail_method`` (``"none"`` when not reduced) and the processes it left
    # without online variables (every online process under ``online*``, those without a use under ``integers*``;
    # ``preprocess._reduce_detail.online_gone``).  Routed with the roll-forward carriers
    # (``solve._carry_routing.CarryRouter.step_carriers``): a solve that keeps online variables and continues from
    # this producer gets a ramp-carry flow without an online carry for those processes, and starts only their ramp
    # and commitment cold instead of raising (``preprocess._reduce_detail.cold_start_online_flow_carry``).  Not
    # carriers: not in ``_FIELDS``.
    unit_detail: str = "none"
    online_left_out: frozenset = frozenset()

    _FIELDS = (
        "realized_invest", "realized_existing", "realized_commission", "divest_cumulative", "realized_divest",
        "roll_end_state", "upward_roll_end_state", "roll_end_online", "roll_end_flow",
        "roll_end_step", "online_history", "fix_online", "fix_reserve",
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
