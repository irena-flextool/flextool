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

import polars as pl

from flextool.flextoolrunner.runner_state import FlexToolConfigError


# ---------------------------------------------------------------------------
# Single-frame writer (canonical _write helper)
# ---------------------------------------------------------------------------


def _write(df: pl.DataFrame, path: Path) -> None:
    """Emit *df* to *path*, creating parent dirs as needed.

    Wrapped by :func:`flextool.engine_polars._flex_data_accumulator.capture_frames`
    so the eight per-solve block CSVs flow into a Provider in-memory
    instead of disk when the cascade captures.  Outside the context the
    helper writes directly to disk.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)

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
    process_block : dict[str, str]
        process_name → process's unified block (Agent 1.6).  Equals the
        process's explicit resolution-group block if set, else the
        finer of ``process_block_in`` / ``process_block_out``.  Used by
        the GMPL side to index UC / startup / shutdown variables and
        their constraints so that unit commitment lives at the
        process's own temporal resolution.  In the degenerate case
        (all entities on ``DEFAULT_BLOCK``) this equals
        ``DEFAULT_BLOCK`` for every process.
    block_step_duration : dict[str, float]
        block_name → hours per step.  ``DEFAULT_BLOCK`` always present.
    """
    node_block: dict[str, str] = field(default_factory=dict)
    process_block_in: dict[str, str] = field(default_factory=dict)
    process_block_out: dict[str, str] = field(default_factory=dict)
    process_block: dict[str, str] = field(default_factory=dict)
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
    reserve_upDown_group: Iterable[tuple[str, str, str]] | None = None,
    process_reserve_upDown_node: Iterable[tuple[str, str, str, str]] | None = None,
) -> None:
    """Reject configurations with ambiguous block / region membership.

    Rules
    -----
    Each entity (node / unit / connection) must be in **at most one**
    resolution-group (a group with ``new_stepduration`` set) and
    **at most one** decomposition-group (a group with
    ``decomposition_method != 'none'``).  Membership in any number of
    regular groups (CO2 caps, reserves, inertia …) is unconstrained.

    Agent 1.7 additional rule: V1 restricts reserves to the default
    (finest) temporal block.  Any entity that participates in a reserve
    (reserve-group member nodes, or processes listed in
    ``process_reserve_upDown_node``) must therefore NOT sit in a
    resolution-group.  Mixing reserves with coarse-resolution
    participants creates subtle semantic issues around energy vs. power
    aggregation; V1 forbids the configuration outright so the reserve
    constraints in ``flextool.mod`` can stay at the fine ``dt`` index.

    Args:
        group_unit: ``(group, unit)`` membership tuples.
        group_connection: ``(group, connection)`` membership tuples.
        group_node: ``(group, node)`` membership tuples.
        resolution_groups: group_name → new_stepduration hours.  Only
            keys matter (the set of resolution-groups).
        decomposition_groups: group_name → decomposition_method value.
            Only groups with a value other than ``"none"`` count as
            decomposition-groups.
        reserve_upDown_group: ``(reserve, upDown, group)`` rows defining
            active reserves.  When the iterable is empty the reserve
            rule is a no-op.
        process_reserve_upDown_node: ``(process, reserve, upDown, node)``
            rows of processes that participate in reserves.

    Raises:
        FlexToolConfigError: if any entity is in two or more
            resolution-groups, two or more decomposition-groups, or
            is a reserve participant while sitting in a resolution-
            group (Agent 1.7 V1 rule).  The message names the entity
            and both conflicting groups.
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

    # Agent 1.7: reserve-block compatibility.  Collect the set of
    # reserve-participating entities (nodes whose reserve-groups are
    # active, plus the processes and their nodes listed in
    # ``process_reserve_upDown_node``) and reject any entity whose
    # resolution-group is set (entity_res[e] non-empty).
    reserve_groups: set[str] = set()
    if reserve_upDown_group is not None:
        for row in reserve_upDown_group:
            if len(row) >= 3:
                reserve_groups.add(row[2])
    reserve_entities: set[str] = set()
    # Nodes carried into a reserve via group membership.
    for g, n in group_node:
        if g in reserve_groups:
            reserve_entities.add(n)
    # Processes (and their nodes) explicitly listed as reserve
    # participants.
    if process_reserve_upDown_node is not None:
        for row in process_reserve_upDown_node:
            if len(row) >= 4:
                reserve_entities.add(row[0])  # process
                reserve_entities.add(row[3])  # node

    for e in sorted(reserve_entities):
        gs = entity_res.get(e, [])
        if gs:
            errors.append(
                f"entity '{e}' participates in reserves but is a member of "
                f"resolution-group(s) {sorted(set(gs))}; V1 restricts "
                f"reserves to the default (finest) block — move '{e}' out "
                f"of the resolution-group or drop the reserve participation."
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
    """Return the resolution-group *entity* belongs to, or None.

    Linear scan kept for backwards compatibility with callers that pass a
    raw membership iterable.  Hot per-entity-loop callers should build a
    lookup once via :func:`_build_membership_index` instead — the linear
    scan is O(M) per entity and turns the surrounding loop into O(N × M)
    when called inside a `for entity in entities:` block.
    """
    for g, e in memberships:
        if e == entity and g in resolution_groups:
            return g
    return None


def _build_membership_index(
    memberships: Iterable[tuple[str, str]],
    resolution_groups: dict[str, float],
) -> dict[str, str]:
    """Pre-build {entity -> first resolution_group} so per-entity lookups
    in `derive_blocks` go from O(M) linear scan to O(1) dict lookup.

    Preserves :func:`_assign_entity_to_group`'s "first match wins"
    semantics: when an entity appears in multiple resolution-groups, the
    first occurrence in *memberships* iteration order wins, matching the
    behaviour of the linear-scan helper.
    """
    idx: dict[str, str] = {}
    for g, e in memberships:
        if e in idx:
            continue
        if g in resolution_groups:
            idx[e] = g
    return idx


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
    # Pre-build node→group lookup once: was O(|nodes| × |gn|) linear scans
    # per node (quadratic in entity count for models where ~every node is
    # tagged).  Now O(|gn|) once + O(|nodes|) lookups.
    node_membership_index = _build_membership_index(gn, resolution_groups)
    node_block: dict[str, str] = {}
    for n in nodes:
        g = node_membership_index.get(n)
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
    process_block: dict[str, str] = {}

    def _finer(a: str, b: str) -> str:
        """Pick the block with the shorter step duration.

        Ties broken in favour of *a* to keep the iteration-order
        determinism callers rely on.
        """
        da = block_step_duration.get(a, default_step_duration)
        db = block_step_duration.get(b, default_step_duration)
        return a if da <= db else b

    # Pre-build process→group lookup once across both unit and connection
    # membership tables.  Was an O((|units|+|connections|) × (|gu|+|gc|))
    # nested scan with `list(gu) + list(gc)` rebuilt every iteration.
    process_memberships: list[tuple[str, str]] = list(gu) + list(gc)
    process_membership_index = _build_membership_index(
        process_memberships, resolution_groups
    )

    for p in list(units) + list(connections):
        # Check explicit unit / connection membership first (same code path —
        # both lists get scanned).
        explicit_block = process_membership_index.get(p)
        if explicit_block is not None:
            process_block_in[p] = explicit_block
            process_block_out[p] = explicit_block
            process_block[p] = explicit_block
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
            # Agent 1.6: process's unified block for UC / ramp / profile.
            # Direct methods already collapsed both sides to the finer
            # one, so the process block matches.
            process_block[p] = finer
        else:
            # Indirect (nvar) — flows already split by side.
            process_block_in[p] = src_block
            process_block_out[p] = snk_block
            # Agent 1.6: process's unified block for UC / ramp / profile.
            # No explicit process-level resolution-group override, so
            # fall back to the finer of the two sides.
            process_block[p] = _finer(src_block, snk_block)

    return BlockAssignments(
        node_block=node_block,
        process_block_in=process_block_in,
        process_block_out=process_block_out,
        process_block=process_block,
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
            and block not in block_assignments.process_block.values()
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
    #
    # Agent 1.9 fix: also emit *self-identity* rows for every
    # non-default block (e.g. ``(d, daily, t, daily, t, 1.0)``).  The
    # generalised node-balance constraints in ``flextool.mod`` look up
    # ``(d, b_c, tc, b_f, t_f) in overlap`` with ``b_c = b_n`` (node
    # block) and ``b_f`` = the process-side block.  When both the node
    # AND the process side live on the same coarse block (e.g. an H2
    # node and a liquefier flow both at ``daily``) the self-identity
    # row is what carries the contribution.  Agent 1.1 only emitted
    # ``(default, default)`` self-identity which is fine for the
    # degenerate case but silently zeroes out coarse-coarse couplings.
    for coarse in non_default:
        coarse_rows = per_block.get(coarse, {})
        # Self-identity: every (period, step) on the coarse block maps
        # onto itself with fraction 1.0.  Without this row, a
        # daily→daily flow would produce no contribution to the daily
        # node balance.
        for period, c_rows in coarse_rows.items():
            for coarse_step, _coarse_dur in c_rows:
                rows.append(
                    (period, coarse, coarse_step, coarse, coarse_step, 1.0)
                )
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


def _entity_block_frame(block_assignments: BlockAssignments) -> pl.DataFrame:
    rows = list(block_assignments.node_block.items())
    return pl.DataFrame(
        {
            "entity": [r[0] for r in rows],
            "block": [r[1] for r in rows],
        },
        schema={"entity": pl.Utf8, "block": pl.Utf8},
    )


def _process_side_block_frame(
    block_assignments: BlockAssignments,
) -> pl.DataFrame:
    procs: list[str] = []
    sides: list[str] = []
    blocks: list[str] = []
    for process, block in block_assignments.process_block_in.items():
        procs.append(process)
        sides.append("source")
        blocks.append(block)
    for process, block in block_assignments.process_block_out.items():
        procs.append(process)
        sides.append("sink")
        blocks.append(block)
    return pl.DataFrame(
        {"process": procs, "side": sides, "block": blocks},
        schema={"process": pl.Utf8, "side": pl.Utf8, "block": pl.Utf8},
    )


def _process_block_frame(
    block_assignments: BlockAssignments,
) -> pl.DataFrame:
    rows = list(block_assignments.process_block.items())
    return pl.DataFrame(
        {
            "process": [r[0] for r in rows],
            "block": [r[1] for r in rows],
        },
        schema={"process": pl.Utf8, "block": pl.Utf8},
    )


def _block_step_duration_frame(
    block_timelines: BlockTimelines | None,
) -> pl.DataFrame:
    blocks: list[str] = []
    periods: list[str] = []
    steps: list[str] = []
    durs: list[float] = []
    if block_timelines is not None:
        for block, period_rows in block_timelines.per_block.items():
            for period, rows in period_rows.items():
                for step, dur in rows:
                    blocks.append(block)
                    periods.append(period)
                    steps.append(step)
                    durs.append(float(dur))
    return pl.DataFrame(
        {
            "block": blocks, "period": periods,
            "step": steps, "step_duration": durs,
        },
        schema={
            "block": pl.Utf8, "period": pl.Utf8,
            "step": pl.Utf8, "step_duration": pl.Float64,
        },
    )


def _overlap_set_frame(overlap_set: OverlapSet) -> pl.DataFrame:
    rows = overlap_set.rows
    return pl.DataFrame(
        {
            "period": [r[0] for r in rows],
            "block_coarse": [r[1] for r in rows],
            "step_coarse": [r[2] for r in rows],
            "block_fine": [r[3] for r in rows],
            "step_fine": [r[4] for r in rows],
            "fraction": [r[5] for r in rows],
        },
        schema={
            "period": pl.Utf8, "block_coarse": pl.Utf8,
            "step_coarse": pl.Utf8, "block_fine": pl.Utf8,
            "step_fine": pl.Utf8, "fraction": pl.Float64,
        },
    )


def _block_step_previous_frame(
    block_predecessors: BlockPredecessors | None,
) -> pl.DataFrame:
    rows = block_predecessors.rows if block_predecessors is not None else []
    return pl.DataFrame(
        {
            "block": [r[0] for r in rows],
            "period": [r[1] for r in rows],
            "step": [r[2] for r in rows],
            "step_previous": [r[3] for r in rows],
            "step_previous_within_timeset": [r[4] for r in rows],
            "period_previous": [r[5] for r in rows],
            "step_previous_within_solve": [r[6] for r in rows],
        },
        schema={
            "block": pl.Utf8, "period": pl.Utf8, "step": pl.Utf8,
            "step_previous": pl.Utf8,
            "step_previous_within_timeset": pl.Utf8,
            "period_previous": pl.Utf8,
            "step_previous_within_solve": pl.Utf8,
        },
    )


def _block_period_time_frame(
    rows: list[tuple[str, str, str]],
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "block": [r[0] for r in rows],
            "period": [r[1] for r in rows],
            "step": [r[2] for r in rows],
        },
        schema={"block": pl.Utf8, "period": pl.Utf8, "step": pl.Utf8},
    )


def write_block_data(
    block_assignments: BlockAssignments,
    overlap_set: OverlapSet,
    block_timelines: BlockTimelines | None,
    solve_data_dir: Path,
    block_predecessors: BlockPredecessors | None = None,
    block_boundaries: BlockBoundaries | None = None,
) -> None:
    """Emit the eight block CSVs into *solve_data_dir*.

    Each emission flows through :func:`_write` so the cascade's
    ``capture_frames`` monkey-patch routes the frame into the Provider
    rather than disk when active.

    Files:

    * ``entity_block.csv``           ``entity,block``.
    * ``process_side_block.csv``     ``process,side,block``.
    * ``process_block.csv``          ``process,block`` (Agent 1.6 unified).
    * ``block_step_duration.csv``    ``block,period,step,step_duration``.
    * ``overlap_set.csv``            ``period,block_coarse,step_coarse,
                                      block_fine,step_fine,fraction``.
    * ``block_step_previous.csv``    per-block predecessor 7-tuples.
    * ``block_period_time_first.csv``
    * ``block_period_time_last.csv``
    """
    solve_data_dir = Path(solve_data_dir)
    _write(
        _entity_block_frame(block_assignments),
        solve_data_dir / "entity_block.csv",
    )
    _write(
        _process_side_block_frame(block_assignments),
        solve_data_dir / "process_side_block.csv",
    )
    _write(
        _process_block_frame(block_assignments),
        solve_data_dir / "process_block.csv",
    )
    _write(
        _block_step_duration_frame(block_timelines),
        solve_data_dir / "block_step_duration.csv",
    )
    _write(
        _overlap_set_frame(overlap_set),
        solve_data_dir / "overlap_set.csv",
    )
    _write(
        _block_step_previous_frame(block_predecessors),
        solve_data_dir / "block_step_previous.csv",
    )
    first_rows = (
        block_boundaries.first if block_boundaries is not None else []
    )
    last_rows = (
        block_boundaries.last if block_boundaries is not None else []
    )
    _write(
        _block_period_time_frame(first_rows),
        solve_data_dir / "block_period_time_first.csv",
    )
    _write(
        _block_period_time_frame(last_rows),
        solve_data_dir / "block_period_time_last.csv",
    )


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
    # Agent 1.9: decomposition_method is a string enum and lives in its
    # own ``p_group_decomposition.csv`` (separate from the numeric
    # ``p_group.csv``).  Older fixtures may still write the row to
    # ``p_group.csv`` — fall back to it for compatibility.
    p_group_decomp_csv = inp / "p_group_decomposition.csv"
    if p_group_decomp_csv.exists():
        with open(p_group_decomp_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("groupParam") == "decomposition_method":
                    decomposition_groups[row["group"]] = str(row["p_group"])
    if not decomposition_groups and p_group_csv.exists():
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

    # Agent 1.7: reserve membership for block-compatibility check.
    # ``reserve__upDown__group__method.csv`` defines which reserves are
    # active (rows where method != 'no_reserve'); the sibling
    # ``process__reserve__upDown__node.csv`` lists the process/node
    # participants.
    reserve_upDown_group: list[tuple[str, str, str]] = []
    rugm_csv = inp / "reserve__upDown__group__method.csv"
    if rugm_csv.exists():
        with open(rugm_csv) as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 4 and row[3] != "no_reserve":
                    reserve_upDown_group.append((row[0], row[1], row[2]))
    process_reserve_upDown_node: list[tuple[str, str, str, str]] = []
    prun_csv = inp / "process__reserve__upDown__node.csv"
    if prun_csv.exists():
        with open(prun_csv) as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 4:
                    process_reserve_upDown_node.append(
                        (row[0], row[1], row[2], row[3])
                    )

    # Validation --------------------------------------------------------
    validate_group_membership(
        group_unit, group_connection, group_node,
        resolution_groups, decomposition_groups,
        reserve_upDown_group=reserve_upDown_group,
        process_reserve_upDown_node=process_reserve_upDown_node,
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
