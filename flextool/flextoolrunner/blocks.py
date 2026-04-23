"""
blocks.py — Per-entity temporal block derivation and overlap set.

Agent 1.1 of the flex-temporal / decomposition refactor.  Purpose:

* Take group-level ``new_stepduration`` (v51 schema) and derive a
  **block** for every node, unit and connection.  A block is an
  identifier that pairs the entity with a particular stepduration
  schedule in the solve.  When no group sets ``new_stepduration`` the
  derivation collapses to a single block named ``"default"``, which is
  bit-identical to pre-v51 behaviour.
* Precompute the cross-resolution **overlap set** — the mapping from
  each (coarse block, coarse timestep) to the (fine block, fine
  timestep) tuples it contains, with the area-fraction the fine step
  contributes to the coarse row.  Agent 1.3's generalised node balance
  consumes this set.
* Emit CSVs in ``solve_data/`` so downstream GMPL / Python layers can
  index by block without re-running the derivation:

  * ``entity_block.csv``            — one row per node.
  * ``process_side_block.csv``      — two rows per process (source + sink).
  * ``block_step_duration.csv``     — per (block, period, step) step
                                       duration, analogous to
                                       ``steps_in_use.csv``.
  * ``overlap_set.csv``             — (period, block_coarse, step_coarse,
                                       block_fine, step_fine, fraction).
  * ``block_step_previous.csv``     — per-block predecessor relations
                                       analogous to ``step_previous.csv``
                                       (Agent 1.4): for each
                                       (block, period, step) row the
                                       corresponding
                                       ``step_previous``,
                                       ``step_previous_within_timeset``,
                                       ``period_previous`` and
                                       ``step_previous_within_solve``.
  * ``block_period_time_first.csv`` — per-block first step of each
                                       period (Agent 1.4).  Analogous
                                       to ``first_timesteps.csv``.
  * ``block_period_time_last.csv``  — per-block last step of each
                                       period (Agent 1.4).  Analogous
                                       to ``last_timesteps.csv``.

The module is **inert for Agent 1.1**: nothing in ``flextool.mod``
consumes these CSVs yet — Agent 1.2 adds the GMPL set/parameter
declarations; Agent 1.3+ wire them into the constraints.

Design notes
------------
* Block naming: an entity in a resolution-group takes that group's
  name; everything else is in ``"default"``.
* Step duration of a block: the resolution-group's
  ``new_stepduration`` if set; else the solve's new_stepduration
  (v50); else the finest step from the solve's timeline.  The
  ``"default"`` block always takes the solve-level value (or the
  raw timeline when unset).
* Per-process block: an explicit ``group__unit`` /
  ``group__connection`` membership in a resolution-group overrides
  both sides to that block.  Otherwise the source-side block follows
  the source node and the sink-side block follows the sink node.
  For direct (1var) methods the two sides agree in practice — when
  they don't the finer side's resolution dominates (Agent 1.2 will
  express this in the model).
* Overlap set: for **aligned subsets** (all coarse stepdurations are
  integer multiples of the finest in the solve, and the finer timeline
  cleanly partitions each coarse row), the fraction is always 1.0 and
  non-matching rows are omitted.  A ``NotImplementedError`` is raised
  for the non-aligned case; the three-region LH2 test case is aligned
  by construction.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from flextool.flextoolrunner.runner_state import FlexToolConfigError

if TYPE_CHECKING:  # pragma: no cover — import cycle guard only
    from flextool.flextoolrunner.solve_config import SolveConfig
    from flextool.flextoolrunner.timeline_config import TimelineConfig


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


DEFAULT_BLOCK = "default"


# Direct (1var) and indirect (nvar) ct_method classification — matches
# ``set method_direct`` / ``set method_indirect`` in flextool_base.dat.
# Users write ct_method values in their DB (constant_efficiency, regular,
# …); the per-(ct, startup, fork) → method resolution happens in
# ``_write_process_method``.  For block derivation we only need the
# single-var vs two-var distinction.  Classification is at the ct_method
# level; an ``exact`` unit with fork_yes becomes ``method_2way_nvar_off``
# — still indirect — so the ct_method → direct/indirect mapping below
# holds regardless of fork.
_CT_METHOD_DIRECT: frozenset[str] = frozenset({
    "constant_efficiency",
    "min_load_efficiency",
    "no_losses_no_variable_cost",
    "variable_cost_only",
    "unidirectional",
})
_CT_METHOD_INDIRECT: frozenset[str] = frozenset({
    "regular",
    "exact",
})


def _is_direct(ct_method: str) -> bool:
    """Return True for 1var-per-process (direct) methods.

    Unknown ct_method values default to ``True`` — i.e. treat them as
    direct — because the block-assignment logic for direct methods is
    strictly more conservative (it picks the finer adjacent block),
    keeping the per-side blocks identical for mis-classified processes.
    """
    return ct_method in _CT_METHOD_DIRECT or ct_method not in _CT_METHOD_INDIRECT


@dataclass
class BlockAssignments:
    """Per-entity block mappings for a single solve.

    Attributes
    ----------
    node_block : dict[str, str]
        node_name → block_name.
    process_block_in : dict[str, str]
        process_name → source-side block_name.
    process_block_out : dict[str, str]
        process_name → sink-side block_name.
    block_step_duration : dict[str, float]
        block_name → hours per step.  ``DEFAULT_BLOCK`` always present.
    """
    node_block: dict[str, str] = field(default_factory=dict)
    process_block_in: dict[str, str] = field(default_factory=dict)
    process_block_out: dict[str, str] = field(default_factory=dict)
    block_step_duration: dict[str, float] = field(default_factory=dict)


@dataclass
class OverlapSet:
    """Cross-resolution overlap rows.

    Each row is ``(period, block_coarse, step_coarse, block_fine,
    step_fine, fraction)``.  Fractions follow the aligned-subsets
    convention: always 1.0 when present, non-matching tuples omitted.
    """
    rows: list[tuple[str, str, str, str, str, float]] = field(default_factory=list)


@dataclass
class BlockTimelines:
    """Per-block (period → [(step, duration_hours)]) timelines.

    Used internally to emit ``block_step_duration.csv`` and to build
    the overlap set.
    """
    per_block: dict[str, dict[str, list[tuple[str, float]]]] = field(default_factory=dict)


@dataclass
class BlockPredecessors:
    """Per-block predecessor relations (Agent 1.4).

    One row per ``(block, period, step)`` in each block's timeline,
    carrying the successor→predecessor mapping analogous to
    ``dtttdt`` / ``step_previous.csv``.  In the degenerate case (only
    the ``"default"`` block) the row set equals the solve's existing
    ``dtttdt`` with the ``"default"`` tag prepended.

    Row schema: ``(block, period, step, step_previous,
    step_previous_within_timeset, period_previous,
    step_previous_within_solve)``.
    """
    rows: list[tuple[str, str, str, str, str, str, str]] = field(default_factory=list)


@dataclass
class BlockBoundaries:
    """Per-block first / last step of each period (Agent 1.4).

    ``first`` and ``last`` are lists of ``(block, period, step)`` rows,
    covering every block in ``BlockAssignments.block_step_duration``.
    In the degenerate case ``first`` and ``last`` for the ``"default"``
    block equal the solve's ``period__time_first`` / ``period__time_last``
    with the ``"default"`` tag prepended.
    """
    first: list[tuple[str, str, str]] = field(default_factory=list)
    last: list[tuple[str, str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_group_membership(
    group_unit: Iterable[tuple[str, str]],
    group_connection: Iterable[tuple[str, str]],
    group_node: Iterable[tuple[str, str]],
    resolution_groups: dict[str, float],
    decomposition_groups: dict[str, str],
) -> None:
    """Reject configurations with ambiguous block / region membership.

    Rules
    -----
    Each entity (node / unit / connection) must be in **at most one**
    resolution-group (a group with ``new_stepduration`` set) and
    **at most one** decomposition-group (a group with
    ``decomposition_method != 'none'``).  Membership in any number of
    regular groups (CO2 caps, reserves, inertia …) is unconstrained.

    Args:
        group_unit: ``(group, unit)`` membership tuples.
        group_connection: ``(group, connection)`` membership tuples.
        group_node: ``(group, node)`` membership tuples.
        resolution_groups: group_name → new_stepduration hours.  Only
            keys matter (the set of resolution-groups).
        decomposition_groups: group_name → decomposition_method value.
            Only groups with a value other than ``"none"`` count as
            decomposition-groups.

    Raises:
        FlexToolConfigError: if any entity is in two or more
            resolution-groups or two or more decomposition-groups.
            The message names the entity and both conflicting groups.
    """
    res_set = set(resolution_groups.keys())
    decomp_set = {g for g, m in decomposition_groups.items() if m and m != "none"}

    # Build entity → list[group_in_resolution / in_decomposition] maps.
    entity_res: dict[str, list[str]] = defaultdict(list)
    entity_decomp: dict[str, list[str]] = defaultdict(list)

    for members in (group_unit, group_connection, group_node):
        for g, e in members:
            if g in res_set:
                entity_res[e].append(g)
            if g in decomp_set:
                entity_decomp[e].append(g)

    errors: list[str] = []
    for e, gs in entity_res.items():
        if len(gs) > 1:
            errors.append(
                f"entity '{e}' is a member of multiple resolution-groups "
                f"({sorted(set(gs))}); each entity may belong to at most "
                f"one group with new_stepduration set."
            )
    for e, gs in entity_decomp.items():
        if len(gs) > 1:
            errors.append(
                f"entity '{e}' is a member of multiple decomposition-groups "
                f"({sorted(set(gs))}); each entity may belong to at most "
                f"one group with decomposition_method != 'none'."
            )
    if errors:
        raise FlexToolConfigError("; ".join(errors))


# ---------------------------------------------------------------------------
# Block derivation
# ---------------------------------------------------------------------------


def _solve_step_duration(
    solve: str,
    solve_config: "SolveConfig | None",
    timeline_config: "TimelineConfig | None",
) -> float:
    """Return the default-block step duration for *solve* in hours.

    Priority (matches existing pre-v51 behaviour):

    1. ``solve.new_stepduration`` (v50 parameter).
    2. The smallest step duration that actually appears in any of the
       solve's timelines.  This catches the pre-v50 case where the
       timeline already encodes the desired resolution.

    When neither is available (unit-test fixture with only the bare
    block dicts) falls back to 1.0.  The return value is only used for
    the ``DEFAULT_BLOCK`` entry in ``block_step_duration.csv`` and the
    identity overlap rows; it does not participate in LP coefficients
    in Agent 1.1.
    """
    if solve_config is not None:
        new_step = solve_config if False else None  # type: ignore
    # Solve-level new_stepduration (v50)
    if timeline_config is not None:
        raw = timeline_config.new_step_durations.get(solve)
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass

    # Fall back to the finest step in the solve's timelines.
    if timeline_config is not None and solve_config is not None:
        timesets_used = solve_config.timesets_used_by_solves.get(solve, [])
        durations: list[float] = []
        for _period, ts in timesets_used:
            tl = timeline_config.timesets__timeline.get(ts)
            if not tl:
                continue
            for step, dur in timeline_config.timelines.get(tl, []):
                try:
                    durations.append(float(dur))
                except (TypeError, ValueError):
                    continue
        if durations:
            return min(durations)
    return 1.0


def _assign_entity_to_group(
    entity: str,
    memberships: Iterable[tuple[str, str]],
    resolution_groups: dict[str, float],
) -> str | None:
    """Return the resolution-group *entity* belongs to, or None."""
    for g, e in memberships:
        if e == entity and g in resolution_groups:
            return g
    return None


def derive_blocks(
    solve: str,
    solve_config: "SolveConfig | None",
    timeline_config: "TimelineConfig | None",
    nodes: Iterable[str],
    units: Iterable[str],
    connections: Iterable[str],
    resolution_groups: dict[str, float],
    group_unit: Iterable[tuple[str, str]],
    group_connection: Iterable[tuple[str, str]],
    group_node: Iterable[tuple[str, str]],
    process_source_sink: Iterable[tuple[str, str, str]],
    process_ct_method: dict[str, str],
) -> BlockAssignments:
    """Compute per-entity block assignments for *solve*.

    Args:
        solve: Solve name (for ``new_stepduration`` lookup).
        solve_config: Live SolveConfig (optional — None in unit tests).
        timeline_config: Live TimelineConfig (optional — None in unit
            tests).
        nodes: Iterable of node names in the solve.
        units: Iterable of unit names in the solve.
        connections: Iterable of connection names in the solve.
        resolution_groups: group_name → hours (``new_stepduration``).
            Used as the set of resolution-groups and for
            ``block_step_duration`` lookup.
        group_unit: ``(group, unit)`` membership rows.
        group_connection: ``(group, connection)`` membership rows.
        group_node: ``(group, node)`` membership rows.
        process_source_sink: ``(process, source_node, sink_node)`` rows.
            A process can appear multiple times for a multi-source or
            multi-sink setup (fork_yes).  The first source/sink seen is
            canonical for block assignment — per-flow blocks are
            derived from node_block in Agent 1.3.
        process_ct_method: process_name → ct_method string (raw
            user-facing values: constant_efficiency, regular, exact,
            …).  Drives direct vs indirect classification.

    Returns:
        BlockAssignments with node_block, process_block_in,
        process_block_out and block_step_duration populated.
    """
    gu = list(group_unit)
    gc = list(group_connection)
    gn = list(group_node)

    # Step-duration table for all blocks seen so far.  Always includes
    # the default block.
    default_step_duration = _solve_step_duration(
        solve, solve_config, timeline_config
    )
    block_step_duration: dict[str, float] = {DEFAULT_BLOCK: default_step_duration}
    for g, dur in resolution_groups.items():
        try:
            block_step_duration[g] = float(dur)
        except (TypeError, ValueError):
            continue

    # --- Node block ----------------------------------------------------
    node_block: dict[str, str] = {}
    for n in nodes:
        g = _assign_entity_to_group(n, gn, resolution_groups)
        node_block[n] = g if g is not None else DEFAULT_BLOCK

    # --- Process block(s) ----------------------------------------------
    # First source and first sink per process (preserve iteration order).
    first_source: dict[str, str] = {}
    first_sink: dict[str, str] = {}
    for p, s, k in process_source_sink:
        first_source.setdefault(p, s)
        first_sink.setdefault(p, k)

    process_block_in: dict[str, str] = {}
    process_block_out: dict[str, str] = {}

    def _finer(a: str, b: str) -> str:
        """Pick the block with the shorter step duration.

        Ties broken in favour of *a* to keep the iteration-order
        determinism callers rely on.
        """
        da = block_step_duration.get(a, default_step_duration)
        db = block_step_duration.get(b, default_step_duration)
        return a if da <= db else b

    for p in list(units) + list(connections):
        memberships = gu if p in first_source or p in first_sink else gu
        # Check explicit unit / connection membership first (same code path —
        # both lists get scanned).
        explicit_block = _assign_entity_to_group(
            p, list(gu) + list(gc), resolution_groups
        )
        if explicit_block is not None:
            process_block_in[p] = explicit_block
            process_block_out[p] = explicit_block
            continue

        src = first_source.get(p)
        snk = first_sink.get(p)
        src_block = node_block.get(src, DEFAULT_BLOCK) if src else DEFAULT_BLOCK
        snk_block = node_block.get(snk, DEFAULT_BLOCK) if snk else DEFAULT_BLOCK

        ct = process_ct_method.get(p, "")
        if _is_direct(ct):
            # One flow variable per process; put it on the finer side.
            finer = _finer(src_block, snk_block)
            process_block_in[p] = finer
            process_block_out[p] = finer
        else:
            # Indirect (nvar) — flows already split by side.
            process_block_in[p] = src_block
            process_block_out[p] = snk_block

    return BlockAssignments(
        node_block=node_block,
        process_block_in=process_block_in,
        process_block_out=process_block_out,
        block_step_duration=block_step_duration,
    )


# ---------------------------------------------------------------------------
# Per-block timelines
# ---------------------------------------------------------------------------


def _build_block_timelines(
    solve: str,
    solve_config: "SolveConfig | None",
    timeline_config: "TimelineConfig | None",
    block_assignments: BlockAssignments,
    active_time_list: dict[str, list] | None = None,
) -> BlockTimelines:
    """Build per-block timelines for *solve*.

    Each block's timeline is a dict of ``period → [(step, duration)]``
    at that block's stepduration.  For the default block this is the
    same list of (timestep, duration) rows the rest of the solve data
    already uses (``steps_in_use.csv``).  For a resolution-group block,
    the timeline is aggregated to that block's step count.

    Aggregation rule
    ----------------
    Every timestep in the default timeline is assigned to the coarse
    block row whose step duration it contributes to.  Currently this
    requires the coarse step duration to be an integer multiple of the
    default step duration (aligned subsets).  The non-aligned case (e.g.
    default=2h, coarse=3h) is surfaced as ``NotImplementedError`` —
    Agent 1.3 will revisit it when the model actually consumes these
    fractions.
    """
    per_block: dict[str, dict[str, list[tuple[str, float]]]] = {}

    # --- Source timeline (default block) ---------------------------------
    # Priority: active_time_list from the orchestration loop (has the
    # trimmed per-solve windowing); otherwise reconstruct from the
    # timeline_config.
    default_dur = block_assignments.block_step_duration[DEFAULT_BLOCK]
    default_periods: dict[str, list[tuple[str, float]]] = {}

    if active_time_list is not None:
        for period, rows in active_time_list.items():
            default_periods[period] = [
                (entry.timestep, float(entry.duration)) for entry in rows
            ]
    elif solve_config is not None and timeline_config is not None:
        for period, ts in solve_config.timesets_used_by_solves.get(solve, []):
            tl = timeline_config.timesets__timeline.get(ts)
            if not tl:
                continue
            tl_rows = timeline_config.timelines.get(tl, [])
            default_periods.setdefault(period, [])
            for step, dur in tl_rows:
                default_periods[period].append((step, float(dur)))
    else:
        # Unit-test mode: no timelines known.  Caller must set them.
        pass

    per_block[DEFAULT_BLOCK] = default_periods

    # --- Aggregate per-resolution-group block ----------------------------
    for block, dur in block_assignments.block_step_duration.items():
        if block == DEFAULT_BLOCK:
            continue
        # Skip blocks that no entity is assigned to (would produce
        # empty rows); consumers still get the duration via
        # block_step_duration.
        if (
            block not in block_assignments.node_block.values()
            and block not in block_assignments.process_block_in.values()
            and block not in block_assignments.process_block_out.values()
        ):
            continue
        block_rows: dict[str, list[tuple[str, float]]] = {}
        for period, rows in default_periods.items():
            aggregated = _aggregate_timeline(rows, coarse_duration=float(dur))
            block_rows[period] = aggregated
        per_block[block] = block_rows

    return BlockTimelines(per_block=per_block)


def _aggregate_timeline(
    rows: list[tuple[str, float]], coarse_duration: float,
) -> list[tuple[str, float]]:
    """Aggregate a fine-resolution timeline to *coarse_duration* hours.

    Emits one coarse row per accumulated block of fine rows whose
    durations sum to exactly ``coarse_duration``.  The coarse row's
    label is the label of the first fine step it covers (matches the
    pattern used by ``create_timeline_from_timestep_duration``).

    Raises ``NotImplementedError`` when a block of fine rows cannot be
    cleanly packed — i.e. the coarse duration is not an integer multiple
    of the fine row it straddles.  The three-region LH2 example is
    aligned-subsets by construction; non-aligned support is deferred.
    """
    out: list[tuple[str, float]] = []
    i = 0
    n = len(rows)
    eps = 1e-9
    while i < n:
        acc = 0.0
        first_label = rows[i][0]
        j = i
        while j < n and acc + float(rows[j][1]) <= coarse_duration + eps:
            acc += float(rows[j][1])
            j += 1
            if abs(acc - coarse_duration) <= eps:
                break
        if abs(acc - coarse_duration) > eps:
            raise NotImplementedError(
                f"Non-aligned subset: fine rows starting at "
                f"'{first_label}' accumulate to {acc}h but coarse "
                f"block stepduration is {coarse_duration}h.  "
                f"Non-integer multiples of the default step are not "
                f"supported yet (aligned-subsets only)."
            )
        out.append((first_label, coarse_duration))
        i = j
    return out


# ---------------------------------------------------------------------------
# Overlap set
# ---------------------------------------------------------------------------


def derive_overlap_set(
    solve: str,
    block_assignments: BlockAssignments,
    block_timelines: BlockTimelines | None = None,
    solve_config: "SolveConfig | None" = None,
    timeline_config: "TimelineConfig | None" = None,
    active_time_list: dict[str, list] | None = None,
) -> OverlapSet:
    """Compute the overlap rows between every pair of blocks in the solve.

    Row schema: ``(period, block_coarse, step_coarse, block_fine,
    step_fine, fraction)``.

    The aligned-subsets case emits fraction = 1.0 for every fine step
    that falls within a coarse step (same calendar moment), and omits
    non-matching rows.  The degenerate case — only the default block
    exists — emits identity rows ``(d, "default", t, "default", t,
    1.0)``.

    Args:
        solve: Solve name.
        block_assignments: From :func:`derive_blocks`.
        block_timelines: Optional precomputed timelines; when ``None``
            this function calls :func:`_build_block_timelines`.
        solve_config / timeline_config: Passed through to the timeline
            builder when *block_timelines* is ``None``.
        active_time_list: Passed through to the timeline builder;
            typically the ``active_time_lists[solve]`` value from the
            orchestration loop.
    """
    if block_timelines is None:
        block_timelines = _build_block_timelines(
            solve=solve,
            solve_config=solve_config,
            timeline_config=timeline_config,
            block_assignments=block_assignments,
            active_time_list=active_time_list,
        )
    per_block = block_timelines.per_block
    default_rows = per_block.get(DEFAULT_BLOCK, {})

    rows: list[tuple[str, str, str, str, str, float]] = []

    # Degenerate case: only the default block exists → identity rows.
    non_default = [b for b in per_block if b != DEFAULT_BLOCK]
    if not non_default:
        for period, pr in default_rows.items():
            for step, _dur in pr:
                rows.append((period, DEFAULT_BLOCK, step, DEFAULT_BLOCK, step, 1.0))
        return OverlapSet(rows=rows)

    # Default-against-default identity rows are always emitted too so
    # that constraints indexed on ``(b_coarse, b_fine) = (default,
    # default)`` have the right coverage.
    for period, pr in default_rows.items():
        for step, _dur in pr:
            rows.append((period, DEFAULT_BLOCK, step, DEFAULT_BLOCK, step, 1.0))

    # For every resolution-group block, emit coarse↔fine rows.  We
    # consider "fine" = default (and any block strictly finer than
    # coarse); "coarse" = the resolution-group block itself.  Every
    # fine step that falls inside a coarse step becomes an overlap row.
    for coarse in non_default:
        coarse_rows = per_block.get(coarse, {})
        fine_rows_per_period = default_rows
        for period, c_rows in coarse_rows.items():
            # Walk the fine timeline in parallel, packing fine rows
            # into the coarse row until its duration is met.
            fine = fine_rows_per_period.get(period, [])
            f_idx = 0
            for coarse_step, coarse_dur in c_rows:
                remaining = float(coarse_dur)
                eps = 1e-9
                while remaining > eps and f_idx < len(fine):
                    fine_step, fine_dur = fine[f_idx]
                    # Aligned-subsets assumption: every fine row is fully
                    # contained in the current coarse row, so we emit the
                    # 1.0 overlap and advance.
                    rows.append(
                        (period, coarse, coarse_step, DEFAULT_BLOCK, fine_step, 1.0)
                    )
                    # Symmetric row: default↔coarse mapping for callers
                    # that query by fine block.
                    rows.append(
                        (period, DEFAULT_BLOCK, fine_step, coarse, coarse_step, 1.0)
                    )
                    remaining -= float(fine_dur)
                    f_idx += 1
                if remaining > eps:
                    raise NotImplementedError(
                        f"overlap_set: coarse step '{coarse_step}' on "
                        f"block '{coarse}' left {remaining}h unmatched "
                        f"at end of fine timeline — non-aligned subsets "
                        f"are not supported yet."
                    )

    # Pairs of resolution-group blocks (coarse vs finer-coarse): emit
    # an overlap row for every shared fine step.  Handles e.g. a 24h
    # block coexisting with a 6h block.  Implemented by intersecting
    # each block's coverage against the default-block steps.
    if len(non_default) >= 2:
        # Build (period, step) → block coverage for each non-default
        # block so we can cross-reference.
        coverage: dict[str, dict[str, dict[str, str]]] = {}
        # coverage[block][period][fine_step] = coarse_step_that_covers_it
        for block in non_default:
            coverage[block] = {}
            b_rows = per_block.get(block, {})
            for period, c_rows in b_rows.items():
                coverage[block][period] = {}
                fine = default_rows.get(period, [])
                f_idx = 0
                for coarse_step, coarse_dur in c_rows:
                    remaining = float(coarse_dur)
                    eps = 1e-9
                    while remaining > eps and f_idx < len(fine):
                        fine_step, fine_dur = fine[f_idx]
                        coverage[block][period][fine_step] = coarse_step
                        remaining -= float(fine_dur)
                        f_idx += 1
        for a in non_default:
            for b in non_default:
                if a == b:
                    continue
                for period, a_cov in coverage[a].items():
                    b_cov = coverage[b].get(period, {})
                    for fine_step, a_step in a_cov.items():
                        b_step = b_cov.get(fine_step)
                        if b_step is None:
                            continue
                        # Identify which is coarser by step-duration.
                        a_dur = block_assignments.block_step_duration.get(a, 0.0)
                        b_dur = block_assignments.block_step_duration.get(b, 0.0)
                        # Only emit coarse→fine pairs to avoid duplication.
                        if a_dur > b_dur:
                            rows.append((period, a, a_step, b, b_step, 1.0))
                        elif a_dur == b_dur and a < b:
                            # Same duration, both cover the same fine
                            # steps — emit the canonical direction.
                            rows.append((period, a, a_step, b, b_step, 1.0))
    # Deduplicate while preserving order: two different fine steps
    # contained in the same coarse pair yields the same (a_step,
    # b_step) pair every iteration.
    seen: set[tuple] = set()
    deduped: list[tuple[str, str, str, str, str, float]] = []
    for row in rows:
        if row in seen:
            continue
        seen.add(row)
        deduped.append(row)
    return OverlapSet(rows=deduped)


# ---------------------------------------------------------------------------
# Per-block predecessors + boundaries (Agent 1.4)
# ---------------------------------------------------------------------------


def derive_block_predecessors(
    solve: str,
    block_assignments: BlockAssignments,
    block_timelines: BlockTimelines,
    default_jump_list: Iterable[tuple] | None = None,
) -> BlockPredecessors:
    """Compute per-block predecessor relations analogous to ``dtttdt``.

    For the ``"default"`` block the predecessor rows are taken straight
    from *default_jump_list* — the orchestration-level
    ``jump_list[solve]`` produced by :func:`make_step_jump` — and
    tagged with the ``"default"`` block label.  This guarantees the
    degenerate case is bit-identical to pre-v51 behaviour: every row
    carries exactly the same ``(step_previous,
    step_previous_within_timeset, period_previous,
    step_previous_within_solve)`` values the GMPL side already
    consumes from ``step_previous.csv``.

    For every non-default block we derive the predecessor relations
    from that block's own aggregated timeline (``block_timelines``).
    The block's timeline is a per-period ordered list of
    ``(step, duration)`` rows; the predecessor of step ``i`` in
    period ``p`` is step ``i - 1`` in the same period (``jump = 1``),
    except the first step of the first period which wraps cyclically
    to the last step of the last period (``jump = -N + 1``), and the
    first step of any subsequent period which points at the last step
    of the previous period.  This mirrors the ``make_step_jump``
    behaviour — block aggregation preserves the simple cyclic pattern
    because the aligned-subsets assumption collapses every block
    internally to one monotone timestep sequence per period.

    Args:
        solve: Solve name (unused for now — kept for symmetry with
            sibling functions).
        block_assignments: From :func:`derive_blocks`.
        block_timelines: From :func:`_build_block_timelines`.
        default_jump_list: Iterable of 7-tuples as produced by
            :func:`flextool.flextoolrunner.timeline_config.make_step_jump`.
            When ``None`` the default-block rows fall back to the
            simple cyclic pattern used for non-default blocks — this
            only matters for unit tests; the orchestration loop always
            passes the real list.

    Returns:
        BlockPredecessors.  ``rows`` preserves iteration order:
        default block first, then resolution-group blocks in insertion
        order.
    """
    rows: list[tuple[str, str, str, str, str, str, str]] = []

    # --- Default block ------------------------------------------------------
    if default_jump_list is not None:
        for entry in default_jump_list:
            # jump_list entries are 7-tuples (period, step, previous,
            # previous_within_timeset, previous_period,
            # previous_within_solve, jump).  Re-project onto the block-
            # tagged 7-tuple (block, period, step, previous,
            # previous_within_timeset, previous_period,
            # previous_within_solve).
            period = entry[0]
            step = entry[1]
            previous = entry[2]
            previous_within_timeset = entry[3]
            previous_period = entry[4]
            previous_within_solve = entry[5]
            rows.append(
                (
                    DEFAULT_BLOCK,
                    period,
                    step,
                    previous,
                    previous_within_timeset,
                    previous_period,
                    previous_within_solve,
                )
            )
    else:
        default_rows = block_timelines.per_block.get(DEFAULT_BLOCK, {})
        rows.extend(_cyclic_block_predecessors(DEFAULT_BLOCK, default_rows))

    # --- Non-default blocks -------------------------------------------------
    for block in block_assignments.block_step_duration:
        if block == DEFAULT_BLOCK:
            continue
        block_rows_per_period = block_timelines.per_block.get(block, {})
        if not block_rows_per_period:
            # Block declared in block_step_duration but no entity is on
            # it — skip to avoid empty period rows.
            continue
        rows.extend(_cyclic_block_predecessors(block, block_rows_per_period))

    return BlockPredecessors(rows=rows)


def _cyclic_block_predecessors(
    block: str,
    per_period_rows: dict[str, list[tuple[str, float]]],
) -> list[tuple[str, str, str, str, str, str, str]]:
    """Cyclic predecessor pattern over a block's aggregated timeline.

    Cycles within the whole block's timeline: the first step of the
    first period wraps to the last step of the last period.  Within a
    period interior steps point at the preceding step (jump=1), and
    the first step of non-first periods points at the last step of
    the previous period (jump across periods).

    The ``step_previous_within_timeset`` column follows the pre-v51
    convention: it equals ``step_previous`` for interior steps
    (rows where the previous step is the immediately adjacent row in
    the same period).  For first-of-period rows — where the jump
    crosses a period boundary — it's pinned at the current period's
    block-last step (matching the ``block_last.timestep`` projection
    used by :func:`make_step_jump`).  In the degenerate case this
    column is populated from the real ``jump_list`` instead of this
    fallback, so the bit-identical promise still holds.
    """
    periods = list(per_period_rows.keys())
    if not periods:
        return []
    first_period = periods[0]
    last_period = periods[-1]

    out: list[tuple[str, str, str, str, str, str, str]] = []
    for pi, period in enumerate(periods):
        rows = per_period_rows[period]
        if not rows:
            continue
        block_last_step = rows[-1][0]
        for si, (step, _dur) in enumerate(rows):
            if si > 0:
                prev_step = rows[si - 1][0]
                prev_within_ts = prev_step
                prev_period = period
                prev_within_solve = prev_step
            else:
                # First step of the period — cross-period predecessor.
                if period == first_period:
                    # Wrap to last period's last step (cyclic).
                    prev_period_rows = per_period_rows[last_period]
                    prev_step = prev_period_rows[-1][0] if prev_period_rows else step
                    prev_period = last_period
                    prev_within_solve = prev_step
                    prev_within_ts = block_last_step
                else:
                    prev_period = periods[pi - 1]
                    prev_rows = per_period_rows[prev_period]
                    prev_step = prev_rows[-1][0] if prev_rows else step
                    prev_within_solve = prev_step
                    prev_within_ts = block_last_step
            out.append(
                (
                    block,
                    period,
                    step,
                    prev_step,
                    prev_within_ts,
                    prev_period,
                    prev_within_solve,
                )
            )
    return out


def derive_block_boundaries(
    block_assignments: BlockAssignments,
    block_timelines: BlockTimelines,
) -> BlockBoundaries:
    """Compute per-block first and last step of each period.

    Mirrors the solve-level ``period__time_first`` /
    ``period__time_last`` sets, but keyed at the block level.  In the
    degenerate (default-block-only) case ``first`` / ``last`` for
    ``"default"`` equal the pre-v51 sets with the block tag prepended.
    """
    first: list[tuple[str, str, str]] = []
    last: list[tuple[str, str, str]] = []
    for block in block_assignments.block_step_duration:
        rows_per_period = block_timelines.per_block.get(block, {})
        for period, rows in rows_per_period.items():
            if not rows:
                continue
            first.append((block, period, rows[0][0]))
            last.append((block, period, rows[-1][0]))
    return BlockBoundaries(first=first, last=last)


# ---------------------------------------------------------------------------
# CSV emission
# ---------------------------------------------------------------------------


def write_block_data(
    block_assignments: BlockAssignments,
    overlap_set: OverlapSet,
    block_timelines: BlockTimelines | None,
    solve_data_dir: Path,
    block_predecessors: BlockPredecessors | None = None,
    block_boundaries: BlockBoundaries | None = None,
) -> None:
    """Emit the four block CSVs into *solve_data_dir*.

    Files:

    * ``entity_block.csv``        ``entity,block`` — one row per node.
    * ``process_side_block.csv``  ``process,side,block`` — two rows per
                                   process (source + sink).
    * ``block_step_duration.csv`` ``block,period,step,step_duration`` —
                                   per-block timeline.  The default
                                   block emits every ``(period, step)``
                                   from the solve's timeline.
    * ``overlap_set.csv``         ``period,block_coarse,step_coarse,
                                   block_fine,step_fine,fraction``.
    """
    solve_data_dir = Path(solve_data_dir)
    solve_data_dir.mkdir(parents=True, exist_ok=True)

    # entity_block.csv --------------------------------------------------
    with open(solve_data_dir / "entity_block.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["entity", "block"])
        for node, block in block_assignments.node_block.items():
            writer.writerow([node, block])

    # process_side_block.csv -------------------------------------------
    with open(solve_data_dir / "process_side_block.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["process", "side", "block"])
        for process, block in block_assignments.process_block_in.items():
            writer.writerow([process, "source", block])
        for process, block in block_assignments.process_block_out.items():
            writer.writerow([process, "sink", block])

    # block_step_duration.csv ------------------------------------------
    with open(solve_data_dir / "block_step_duration.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["block", "period", "step", "step_duration"])
        if block_timelines is not None:
            for block, period_rows in block_timelines.per_block.items():
                for period, rows in period_rows.items():
                    for step, dur in rows:
                        writer.writerow([block, period, step, dur])

    # overlap_set.csv --------------------------------------------------
    with open(solve_data_dir / "overlap_set.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["period", "block_coarse", "step_coarse",
             "block_fine", "step_fine", "fraction"]
        )
        for row in overlap_set.rows:
            writer.writerow(row)

    # block_step_previous.csv (Agent 1.4) -------------------------------
    # Per-block predecessor relations, analogous to step_previous.csv.
    # In the degenerate case the default-block rows match step_previous
    # bit-identically (they're reprojected from the same jump_list).
    with open(solve_data_dir / "block_step_previous.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["block", "period", "step", "step_previous",
             "step_previous_within_timeset", "period_previous",
             "step_previous_within_solve"]
        )
        if block_predecessors is not None:
            for row in block_predecessors.rows:
                writer.writerow(row)

    # block_period_time_first.csv / block_period_time_last.csv (Agent 1.4)
    with open(solve_data_dir / "block_period_time_first.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["block", "period", "step"])
        if block_boundaries is not None:
            for row in block_boundaries.first:
                writer.writerow(row)
    with open(solve_data_dir / "block_period_time_last.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["block", "period", "step"])
        if block_boundaries is not None:
            for row in block_boundaries.last:
                writer.writerow(row)


# ---------------------------------------------------------------------------
# Orchestration hook
# ---------------------------------------------------------------------------


def write_block_data_for_solve(
    solve: str,
    solve_config: "SolveConfig",
    timeline_config: "TimelineConfig",
    work_folder: Path,
    active_time_list: dict[str, list] | None = None,
    default_jump_list: Iterable[tuple] | None = None,
) -> BlockAssignments:
    """End-to-end helper called from the orchestration loop.

    Reads the relevant ``input/`` CSVs (produced earlier in the run by
    ``input_writer.write_input``), derives blocks, builds the overlap
    set and writes all four ``solve_data/`` CSVs.

    Returns the BlockAssignments in case downstream agents want them.
    """
    wf = Path(work_folder)
    inp = wf / "input"

    # Entity iteration ---------------------------------------------------
    nodes: list[str] = []
    units: list[str] = []
    connections: list[str] = []
    ent_csv = inp / "entity.csv"
    # Partition entities by the class files.
    if (inp / "node.csv").exists():
        with open(inp / "node.csv") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    nodes.append(row[0])
    if (inp / "process_unit.csv").exists():
        with open(inp / "process_unit.csv") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    units.append(row[0])
    if (inp / "process_connection.csv").exists():
        with open(inp / "process_connection.csv") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    connections.append(row[0])
    if not nodes and not units and not connections and ent_csv.exists():
        # Fall back to entity.csv when per-class files are empty (rare
        # pre-v42 fixtures).
        with open(ent_csv) as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    nodes.append(row[0])

    # Resolution groups --------------------------------------------------
    resolution_groups: dict[str, float] = {}
    # Read p_group.csv — the universal group-level parameter dump.  We
    # fall back to DB-less defaults when the file is absent (unit tests
    # short-circuit this path via ``derive_blocks`` directly).
    p_group_csv = inp / "p_group.csv"
    if p_group_csv.exists():
        with open(p_group_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("groupParam") == "new_stepduration":
                    try:
                        resolution_groups[row["group"]] = float(row["p_group"])
                    except (TypeError, ValueError):
                        continue
    decomposition_groups: dict[str, str] = {}
    if p_group_csv.exists():
        with open(p_group_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("groupParam") == "decomposition_method":
                    decomposition_groups[row["group"]] = str(row["p_group"])

    # Group memberships --------------------------------------------------
    group_node: list[tuple[str, str]] = []
    group_process: list[tuple[str, str]] = []
    if (inp / "group__node.csv").exists():
        with open(inp / "group__node.csv") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    group_node.append((row[0], row[1]))
    if (inp / "group__process.csv").exists():
        with open(inp / "group__process.csv") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    group_process.append((row[0], row[1]))

    # Split group_process into group_unit / group_connection so the
    # validator can cite the right class (and so the derivation can
    # scan the right CSV).  This is a membership split by entity name.
    unit_set = set(units)
    conn_set = set(connections)
    group_unit = [(g, p) for g, p in group_process if p in unit_set]
    group_connection = [(g, p) for g, p in group_process if p in conn_set]

    # Validation --------------------------------------------------------
    validate_group_membership(
        group_unit, group_connection, group_node,
        resolution_groups, decomposition_groups,
    )

    # Process source/sink ------------------------------------------------
    process_source_sink: list[tuple[str, str, str]] = []
    sources: dict[str, list[str]] = defaultdict(list)
    sinks: dict[str, list[str]] = defaultdict(list)
    if (inp / "process__source.csv").exists():
        with open(inp / "process__source.csv") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    sources[row[0]].append(row[1])
    if (inp / "process__sink.csv").exists():
        with open(inp / "process__sink.csv") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    sinks[row[0]].append(row[1])
    for p in set(list(sources.keys()) + list(sinks.keys())):
        src_list = sources.get(p, [""])
        snk_list = sinks.get(p, [""])
        # Emit one row for the canonical (first) source and first sink
        # pair.  The derivation only needs the first seen.
        process_source_sink.append((p, src_list[0], snk_list[0]))

    # Process ct_method --------------------------------------------------
    process_ct_method: dict[str, str] = {}
    ctm_csv = inp / "process__ct_method.csv"
    if ctm_csv.exists():
        with open(ctm_csv) as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    process_ct_method[row[0]] = row[1]

    # Derive + write ----------------------------------------------------
    block_assignments = derive_blocks(
        solve=solve,
        solve_config=solve_config,
        timeline_config=timeline_config,
        nodes=nodes,
        units=units,
        connections=connections,
        resolution_groups=resolution_groups,
        group_unit=group_unit,
        group_connection=group_connection,
        group_node=group_node,
        process_source_sink=process_source_sink,
        process_ct_method=process_ct_method,
    )

    block_timelines = _build_block_timelines(
        solve=solve,
        solve_config=solve_config,
        timeline_config=timeline_config,
        block_assignments=block_assignments,
        active_time_list=active_time_list,
    )

    overlap = derive_overlap_set(
        solve=solve,
        block_assignments=block_assignments,
        block_timelines=block_timelines,
    )

    block_predecessors = derive_block_predecessors(
        solve=solve,
        block_assignments=block_assignments,
        block_timelines=block_timelines,
        default_jump_list=default_jump_list,
    )
    block_boundaries = derive_block_boundaries(
        block_assignments=block_assignments,
        block_timelines=block_timelines,
    )

    write_block_data(
        block_assignments=block_assignments,
        overlap_set=overlap,
        block_timelines=block_timelines,
        solve_data_dir=wf / "solve_data",
        block_predecessors=block_predecessors,
        block_boundaries=block_boundaries,
    )

    return block_assignments
