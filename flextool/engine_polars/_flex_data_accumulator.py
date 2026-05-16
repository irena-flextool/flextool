"""Phase C — per-sub-solve FlexData accumulator.

This module owns the in-memory carrier that collects the writer-port
``_write_*`` derived frames during one sub-solve's preprocessing pass.
Phase D will consume the accumulator as a ``seed`` to ``load_flextool``
to skip the disk-read path; Phase C only builds the accumulator and
plumbs it forward — CSV emission is unchanged.

Memory discipline (handoff decision #11)
----------------------------------------

The accumulator is **per sub-solve only**.  It is built fresh at the
start of each per-sub-solve preprocessing pass and replaced when the
next sub-solve runs.  The cascade must NOT accumulate Solution +
FlexData across sub-solves; the same applies here — there is no
cascade-wide accumulator dict, only the latest sub-solve's frames.

Design — wrapper-side capture (approach (a) per Phase C handoff)
----------------------------------------------------------------

The 37 ``OK_thin_wrapper`` writers identified in
``specs/phase_b_writer_audit.md`` all funnel their derived frames
through their module's private ``_write(df, path)`` helper before
emitting the CSV.  We monkey-patch that helper for the duration of
:class:`FlexDataAccumulator` (and its context-manager partner
:func:`capture_frames`) so every CSV emission also stashes
``frames[path.name] = df`` for the sub-solve.

The 103 "special-handling" writers (multi-CSV streamed monoliths and
``fh.write()`` row-by-row emitters identified in the same audit) are
left untouched in this phase — Phase C explicitly defers their
adapters.  Downstream consumers in Phase D will read the missing
fields from ``load_flextool``'s disk path.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import polars as pl


# ---------------------------------------------------------------------------
# Writer modules whose ``_write`` helper feeds the 37 thin-wrapper writers.
# Patching these four modules' ``_write`` covers every OK_thin_wrapper entry
# from the Phase B audit (writers in _writer_leaf_sets, _writer_mid_sets,
# _writer_calc_params, _writer_arc_unions).
# ---------------------------------------------------------------------------

_PATCH_MODULES = (
    "flextool.engine_polars._writer_leaf_sets",
    "flextool.engine_polars._writer_mid_sets",
    "flextool.engine_polars._writer_calc_params",
    "flextool.engine_polars._writer_arc_unions",
    "flextool.engine_polars._writer_chain_params",
    "flextool.engine_polars._writer_co2_accumulators",
)


# ---------------------------------------------------------------------------
# Public dataclass.
# ---------------------------------------------------------------------------


@dataclass
class FlexDataAccumulator:
    """Per-sub-solve carrier of derived frames keyed by target CSV name.

    Keys are the basename of the path each writer's ``_write`` helper
    receives (e.g. ``"period_group.csv"`` or
    ``"entity_lifetime_method.csv"``) — the canonical name the CSV
    writes to under ``<work>/solve_data/``.  Phase D will map these
    keys into the equivalent ``FlexData`` fields when wiring the
    cascade to consume the accumulator instead of re-reading from
    disk.

    The accumulator is NOT cascade-wide.  It is built fresh at the
    start of each sub-solve's preprocessing pass and replaced when the
    next sub-solve runs.  Only the latest sub-solve's frames are
    retained.
    """

    solve_name: str | None = None
    frames: dict[str, pl.DataFrame] = field(default_factory=dict)

    def capture(self, path: Path | str, df: pl.DataFrame) -> None:
        """Stash a (path.name → frame) pair.  Overwrites on duplicate key."""
        # Use basename so identical CSV names across runs collide
        # deterministically (only one final frame per CSV target).
        key = Path(path).name
        # Clone to insulate accumulated state from any in-place mutation
        # the writer might do post-_write (none of the 37 thin writers do,
        # but the clone is cheap insurance — polars clone is a view alias).
        self.frames[key] = df

    # Convenience for tests / Phase-D consumers.
    def __contains__(self, key: str) -> bool:
        return key in self.frames

    def get(self, key: str) -> pl.DataFrame | None:
        return self.frames.get(key)

    def keys(self) -> list[str]:
        return list(self.frames.keys())

    # ------------------------------------------------------------------
    # Phase D — seed lookup
    # ------------------------------------------------------------------
    def lookup(self, target: "Path | str") -> "pl.DataFrame | None":
        """Return the captured frame whose target CSV basename matches
        *target*, or ``None`` when this accumulator does not cover the
        file.

        ``target`` may be a full ``Path`` (``<work>/solve_data/foo.csv``)
        or a bare basename (``"foo.csv"`` or ``"foo"`` — the ``.csv``
        suffix is added when missing, to mirror
        :meth:`CsvSource.get`'s call style).

        Phase D's :func:`load_flextool` consumes this method through a
        process-level seed hook installed in
        :mod:`flextool.engine_polars._input_source`.  See
        :func:`_install_seed` / :func:`_seed_lookup` there.
        """
        name = Path(target).name
        if not name.endswith(".csv"):
            name = f"{name}.csv"
        return self.frames.get(name)


# ---------------------------------------------------------------------------
# Context manager — monkey-patches the four writer modules' ``_write``
# helper to capture frames into the supplied accumulator.
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def capture_frames(
    accumulator: FlexDataAccumulator,
) -> Iterator[FlexDataAccumulator]:
    """Patch the participating writers' ``_write`` helper to capture
    derived frames into *accumulator* for the duration of the block.

    The 37 ``OK_thin_wrapper`` writers from the Phase B audit each call
    their module's ``_write(df, path)`` exactly once per emitted CSV.
    By rebinding that name on each module for the lifetime of this
    context, every emission also pushes ``(path.name → df)`` into the
    accumulator.

    The patched ``_write`` still emits the CSV — Phase C is parallel-
    write mode.  CSV byte-identical parity tests stay green; the
    accumulator is purely additive.
    """
    import importlib

    modules = [importlib.import_module(name) for name in _PATCH_MODULES]
    saved: list[tuple[object, object]] = [
        (mod, getattr(mod, "_write")) for mod in modules
    ]
    try:
        for mod, original in saved:
            def _make_wrapped(_orig=original):  # late-bind per module
                def _wrapped(df: pl.DataFrame, path: Path) -> None:
                    accumulator.capture(path, df)
                    _orig(df, path)
                return _wrapped
            setattr(mod, "_write", _make_wrapped())
        yield accumulator
    finally:
        for mod, original in saved:
            setattr(mod, "_write", original)


__all__ = ["FlexDataAccumulator", "capture_frames", "expected_basenames"]


# ---------------------------------------------------------------------------
# Coverage manifest
# ---------------------------------------------------------------------------
#
# Basenames of CSVs that go through one of the patched ``_write`` helpers.
# This list is the public contract Phase D / E-a consumers can read against
# to know which solve_data/*.csv files the accumulator captures in-memory.
#
# When you lift another streamed writer into the canonical
# ``derive_X → _write(derive_X(...), path)`` shape, add its target basename
# below.  The matching test in
# ``tests/engine_polars/test_phase_c_flex_data_accumulator.py`` cross-checks
# the captured-vs-disk frames for each basename present in the cascade run.

_THIN_WRAPPER_BASENAMES: tuple[str, ...] = (
    # _writer_leaf_sets — 27 thin writers
    "period_group.csv",
    "period_node.csv",
    "period_commodity.csv",
    "period_process.csv",
    "entityInvest.csv",
    "entityDivest.csv",
    "group_invest.csv",
    "group_divest.csv",
    "group_co2_price.csv",
    "group_co2_max_period.csv",
    "group_co2_max_total.csv",
    "optional_yes.csv",
    "reserve__upDown__group.csv",
    "group_loss_share.csv",
    "def_optional_yes.csv",
    "process_delayed.csv",
    "process_side.csv",
    "period_solve.csv",
    "time.csv",
    "enable_optional_outputs.csv",
    "nodeState_rp.csv",
    "nodeStateBlock.csv",
    "commodity__tier.csv",
    "tier.csv",
    "timeline.csv",
    "timeline_steps.csv",
    "commodity__tier_ann.csv",
    # _writer_mid_sets — thin writers
    "group_entity.csv",
    "process_delayed__duration.csv",
    "process__sink_nonSync.csv",
    "entity_lifetime_method.csv",
    "process_ct_method.csv",
    "process_startup_method.csv",
    "node_inflow_method.csv",
    "node_storage_binding_method.csv",
    "connection_param.csv",
    "nodegroup_dispatch_node.csv",
    "commodity_node_co2.csv",
    "process__commodity__node.csv",
    # _writer_calc_params — thin writers
    "process_VRE.csv",
    # _writer_arc_unions — thin writers + Phase E-b lifted streamed writers
    # (this group expanded substantially when streamed writers were
    # converted to the canonical derive_X → _write pattern)
    "process_source_sink_param_t.csv",
    "node__TimeParam_in_use.csv",
    "process_source_delayed.csv",
    "process_source_undelayed.csv",
    "process_source_sink_param.csv",
    "process__source__sink__profile__profile_method_connection.csv",
    "process_method_sources_sinks.csv",
    "ed_history_realized_first.csv",
    "process__source__sinkIsNode.csv",
    "process__source__sinkIsNode_2way1var.csv",
    "process__source__sinkIsNode_not2way1var.csv",
    "process__source__sinkIsNode_2way2var.csv",
    "process_source_sink_ramp_method.csv",
    "process_source_sink_coeff_zero.csv",
    "process_source_sink_delayed.csv",
    "process_source_sink_undelayed.csv",
    "p_process_source_sink.csv",
    "nodeGroupDispatch__process_fully_inside.csv",
    # — ramp family + union (Phase E-b promoted)
    "process_source_sink_ramp_limit_source_up.csv",
    "process_source_sink_ramp_limit_sink_up.csv",
    "process_source_sink_ramp_limit_source_down.csv",
    "process_source_sink_ramp_limit_sink_down.csv",
    "process_source_sink_ramp_cost.csv",
    "process_source_sink_ramp.csv",
    # — group_commodity_node co2 (Phase E-b)
    "group_commodity_node_period_co2_total.csv",
    "group_commodity_node_period_co2_period.csv",
    # — param_in_use family (already _write; in audit scope)
    "node__PeriodParam_in_use.csv",
    "process__PeriodParam_in_use.csv",
    "process_TimeParam_in_use.csv",
    "process_source_sourceSinkTimeParam_in_use.csv",
    "process_sink_sourceSinkTimeParam_in_use.csv",
    "process_source_sourceSinkPeriodParam_in_use.csv",
    "process_sink_sourceSinkPeriodParam_in_use.csv",
    # — Phase E-b lifted streamed writers
    "peedt.csv",
    "process__source__sink__param_t.csv",
    "gdt_maxInstantFlow.csv",
    "gdt_minInstantFlow.csv",
    "p_process_delay_weight.csv",
    "gcndt_co2_price.csv",
    "p_flow_min.csv",
    "p_flow_max.csv",
    "p_state_slack_share.csv",
    "p_storage_state_reference_price.csv",
    "ed_history_realized.csv",
    "process__source__sink__profile__profile_method.csv",
    "process_sinkIsNode_2way1var.csv",
    "nodeSelfDischarge.csv",
    "pdt_online_linear.csv",
    "pdt_online_integer.csv",
    # — 12-CSV nodeGroupDispatch family
    "nodeGroupDispatch__process__unit__to_node_Not_in_aggregate.csv",
    "nodeGroupDispatch__process__node__to_unit_Not_in_aggregate.csv",
    "nodeGroupDispatch__group_aggregate__process__unit__to_node.csv",
    "nodeGroupDispatch__group_aggregate__process__node__to_unit.csv",
    "nodeGroupDispatch__process__node__to_connection_Not_in_aggregate.csv",
    "nodeGroupDispatch__process__connection__to_node_Not_in_aggregate.csv",
    "nodeGroupDispatch__connection_Not_in_aggregate.csv",
    "nodeGroupDispatch__group_aggregate__process__connection__to_node.csv",
    "nodeGroupDispatch__group_aggregate__process__node__to_connection.csv",
    "nodeGroupDispatch__group_aggregate_Connection.csv",
    "nodeGroupDispatch__group_aggregate_Unit_to_group.csv",
    "nodeGroupDispatch__group_aggregate_Group_to_unit.csv",
    # — 8-CSV param_t projections + timeParam
    "process__param_t.csv",
    "connection__param__time.csv",
    "connection__param_t.csv",
    "process__source__param_t.csv",
    "process__sink__param_t.csv",
    "process__source__timeParam.csv",
    "process__sink__timeParam.csv",
    "process__timeParam.csv",
    # _writer_chain_params — Phase E-b lifted streamed writers
    "p_entity_pre_existing.csv",
    "p_entity_divest_cumulative_max.csv",
    # — 5-CSV existing chain
    "p_entity_existing_capacity_later_solves.csv",
    "p_entity_all_existing.csv",
    "p_entity_existing_count.csv",
    "p_entity_existing_integer_count.csv",
    "p_entity_previously_invested_capacity.csv",
    # — 4-CSV capacity max chain
    "p_entity_max_capacity.csv",
    "p_entity_max_units.csv",
    "p_entity_invest_cumulative_max.csv",
    "p_entity_dispatch_capacity_max.csv",
    # _writer_co2_accumulators — Phase E-b lifted
    "co2_cum_realized_tonnes.csv",
)


def expected_basenames() -> tuple[str, ...]:
    """Return the basenames the accumulator is expected to capture.

    The list comes from the Phase B writer audit (every ``OK_thin_wrapper``
    entry, plus the streamed writers lifted into the canonical pattern by
    Phase E-b).  Tests use this to cross-check disk-vs-accumulator parity
    without re-enumerating the list inline.
    """
    return _THIN_WRAPPER_BASENAMES
