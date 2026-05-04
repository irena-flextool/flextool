"""BlockLayout — multi-resolution timestep grouping (Δ.2).

Native ``engine_polars`` port of
``flextool/flextoolrunner/blocks.py`` (1368 LOC).  Produces, in one
pass, every per-solve block-related frame consumed downstream:

* ``entity_block``         — ``(entity, block)``
* ``process_side_block``   — ``(process, side, block)``
* ``process_block``        — ``(process, block)`` — process-unified
* ``block_step_duration``  — ``(block, period, step, step_duration)``
* ``overlap_set``          — ``(period, block_coarse, step_coarse,
  block_fine, step_fine, fraction)``
* ``block_step_previous``  — per-block predecessor 7-tuples
* ``block_period_time_first/_last`` — per-block period boundaries

The algorithm is a 1:1 port of the reference (multi-resolution
alignment is genuinely intricate; the port preserves every quirk of
the original — including the two-block-deep limit and the
aligned-subsets-only assumption).  Internal bookkeeping uses Python
dicts/tuples (the input data is small — O(entities + arcs)) and the
output surface is polars DataFrames so consumers get a uniform shape.

Key decisions:

* **Lazy polars at the rim**: the public ``BlockLayout`` exposes
  ``LazyFrame``-friendly polars frames; consumers ``.collect()`` once
  at LP-build time.
* **Cache per-(b_coarse, b_fine) overlap rows** as the schematic
  recommends — built once during ``derive_overlap_set`` and shared
  across every arc that references the pair.
* **Two-block-deep limit accepted** — the reference assumes one
  resolution-group block + the natural fine block; the port preserves
  this.  Lifting it is a model-design question.
* **No defensive gating** — degenerate single-block fixtures produce
  identity-only overlap rows (matching pre-v51 behaviour); fixtures
  with no resolution groups bypass the multi-block branches entirely.
* **Fail loud** — ambiguous group memberships and non-aligned subsets
  raise ``FlexToolConfigError`` / ``NotImplementedError`` (matching
  the reference).

Reference: ``flextool/flextoolrunner/blocks.py`` (read-only).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

import polars as pl

from flextool.engine_polars._solve_state import FlexToolConfigError

if TYPE_CHECKING:  # pragma: no cover — import cycle guard only
    from flextool.engine_polars._solve_config import SolveConfig
    from flextool.engine_polars._timeline import TimelineConfig


# ---------------------------------------------------------------------------
# Public constants & schema
# ---------------------------------------------------------------------------


DEFAULT_BLOCK = "default"


# Direct (1var) and indirect (nvar) ct_method classification — matches
# ``set method_direct`` / ``set method_indirect`` in flextool_base.dat.
# Classification is at the ct_method level: unknown values default to
# direct (the more conservative choice for block assignment).
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
    """Return ``True`` for 1var-per-process (direct) methods."""
    return (
        ct_method in _CT_METHOD_DIRECT
        or ct_method not in _CT_METHOD_INDIRECT
    )


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

    Mirrors ``flextool/flextoolrunner/blocks.validate_group_membership``
    1:1.  Each entity may belong to at most one resolution-group and
    at most one decomposition-group; reserve participants must NOT
    sit in any resolution-group (V1 reserve-block compatibility rule).

    Raises:
        FlexToolConfigError: on any of the rule violations.
    """
    res_set = set(resolution_groups.keys())
    decomp_set = {g for g, m in decomposition_groups.items() if m and m != "none"}

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

    reserve_groups: set[str] = set()
    if reserve_upDown_group is not None:
        for row in reserve_upDown_group:
            if len(row) >= 3:
                reserve_groups.add(row[2])
    reserve_entities: set[str] = set()
    for g, n in group_node:
        if g in reserve_groups:
            reserve_entities.add(n)
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
# Block derivation primitives
# ---------------------------------------------------------------------------


def _solve_step_duration(
    solve: str,
    solve_config: "SolveConfig | None",
    timeline_config: "TimelineConfig | None",
) -> float:
    """Return the default-block step duration for *solve* in hours.

    Priority (matches ``blocks._solve_step_duration``):

    1. ``timeline.new_step_durations[solve]`` (v50 parameter).
    2. The smallest step duration that actually appears in any of the
       solve's timelines.

    Falls back to ``1.0`` when neither source is available (unit-test
    fixture path).
    """
    if timeline_config is not None:
        raw = timeline_config.new_step_durations.get(solve)
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass

    if timeline_config is not None and solve_config is not None:
        timesets_used = solve_config.timesets_used_by_solves.get(solve, [])
        durations: list[float] = []
        for _period, ts in timesets_used:
            tl = timeline_config.timesets__timeline.get(ts)
            if not tl:
                continue
            for _step, dur in timeline_config.timelines.get(tl, []):
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
    """Return the resolution-group *entity* belongs to, or ``None``."""
    for g, e in memberships:
        if e == entity and g in resolution_groups:
            return g
    return None


def _aggregate_timeline(
    rows: list[tuple[str, float]], coarse_duration: float,
) -> list[tuple[str, float]]:
    """Aggregate a fine-resolution timeline to *coarse_duration* hours.

    Mirrors ``blocks._aggregate_timeline`` 1:1.  Aligned-subsets only:
    raises ``NotImplementedError`` when fine rows can't pack cleanly
    into coarse rows.
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


def _cyclic_block_predecessors(
    block: str,
    per_period_rows: dict[str, list[tuple[str, float]]],
) -> list[tuple[str, str, str, str, str, str, str]]:
    """Cyclic predecessor pattern over a block's aggregated timeline.

    Mirrors ``blocks._cyclic_block_predecessors``.  Wraps cyclically
    on the first step of the first period; first step of subsequent
    periods crosses to the previous period's block-last step.
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
                if period == first_period:
                    prev_period_rows = per_period_rows[last_period]
                    prev_step = (
                        prev_period_rows[-1][0]
                        if prev_period_rows else step
                    )
                    prev_period = last_period
                    prev_within_solve = prev_step
                    prev_within_ts = block_last_step
                else:
                    prev_period = periods[pi - 1]
                    prev_rows = per_period_rows[prev_period]
                    prev_step = (
                        prev_rows[-1][0]
                        if prev_rows else step
                    )
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


# ---------------------------------------------------------------------------
# BlockLayout — public surface
# ---------------------------------------------------------------------------


@dataclass
class BlockLayout:
    """Per-solve block layout — multi-resolution storage.

    Internal bookkeeping (``node_block``, ``process_block_in/_out/_block``,
    ``block_step_duration``) lives as Python dicts because the input
    data is small (O(entities + arcs)) and the algorithm is naturally
    expressed in dict/tuple form.  The output surface (``entity_block``,
    ``process_side_block``, ``block_step_duration_frame``, etc.) exposes
    every frame the LP-build layer needs as a polars DataFrame in the
    same column-naming convention the reference CSVs use.

    Attributes
    ----------
    Internal (matching ``BlockAssignments``):

    node_block : dict[str, str]
        ``node_name → block_name``.
    process_block_in / process_block_out / process_block : dict[str, str]
        Per-process block on each side; ``process_block`` is the
        unified block used for UC / startup / shutdown indices.
    block_step_duration : dict[str, float]
        ``block_name → hours per step``.  ``DEFAULT_BLOCK`` always
        present.

    Frames (every CSV the reference emits):

    entity_block_frame : pl.DataFrame
        Columns ``(entity, block)``.
    process_side_block_frame : pl.DataFrame
        Columns ``(process, side, block)`` — two rows per process.
    process_block_frame : pl.DataFrame
        Columns ``(process, block)`` — one row per process.
    block_step_duration_frame : pl.DataFrame
        Columns ``(block, period, step, step_duration)``.
    overlap_set_frame : pl.DataFrame
        Columns ``(period, block_coarse, step_coarse, block_fine,
        step_fine, fraction)``.
    block_step_previous_frame : pl.DataFrame
        Columns ``(block, period, step, step_previous,
        step_previous_within_timeset, period_previous,
        step_previous_within_solve)``.
    block_period_time_first_frame / _last_frame : pl.DataFrame
        Columns ``(block, period, step)``.
    """

    # --- Internal bookkeeping --------------------------------------------
    node_block: dict[str, str] = field(default_factory=dict)
    process_block_in: dict[str, str] = field(default_factory=dict)
    process_block_out: dict[str, str] = field(default_factory=dict)
    process_block: dict[str, str] = field(default_factory=dict)
    block_step_duration: dict[str, float] = field(default_factory=dict)
    # Per-block (period → [(step, duration)]) timeline.
    per_block_timeline: dict[str, dict[str, list[tuple[str, float]]]] = field(
        default_factory=dict,
    )

    # --- Output frames ----------------------------------------------------
    entity_block_frame: pl.DataFrame = field(default_factory=pl.DataFrame)
    process_side_block_frame: pl.DataFrame = field(default_factory=pl.DataFrame)
    process_block_frame: pl.DataFrame = field(default_factory=pl.DataFrame)
    block_step_duration_frame: pl.DataFrame = field(default_factory=pl.DataFrame)
    overlap_set_frame: pl.DataFrame = field(default_factory=pl.DataFrame)
    block_step_previous_frame: pl.DataFrame = field(default_factory=pl.DataFrame)
    block_period_time_first_frame: pl.DataFrame = field(default_factory=pl.DataFrame)
    block_period_time_last_frame: pl.DataFrame = field(default_factory=pl.DataFrame)

    # ------------------------------------------------------------------
    # Builder
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
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
        decomposition_groups: dict[str, str] | None = None,
        reserve_upDown_group: Iterable[tuple[str, str, str]] | None = None,
        process_reserve_upDown_node: (
            Iterable[tuple[str, str, str, str]] | None
        ) = None,
        active_time_list: dict[str, list] | None = None,
        default_jump_list: Iterable[tuple] | None = None,
        validate: bool = True,
    ) -> "BlockLayout":
        """Compute every per-solve block frame in one pass.

        Parameters mirror flextool's ``write_block_data_for_solve`` /
        ``derive_blocks`` / ``derive_overlap_set`` signatures so the
        port is a drop-in algorithmic equivalent.

        ``validate=False`` skips ``validate_group_membership`` —
        useful when the caller already validated upstream.

        Returns
        -------
        BlockLayout
            Fully populated dataclass.  Frames carry the canonical
            CSV column names so downstream helpers can join directly.

        Raises
        ------
        FlexToolConfigError
            On ambiguous group membership (validation failure).
        NotImplementedError
            On non-aligned-subset coarse blocks (reference behaviour).
        """
        gu = list(group_unit)
        gc = list(group_connection)
        gn = list(group_node)
        decomp = decomposition_groups or {}

        if validate:
            validate_group_membership(
                gu, gc, gn,
                resolution_groups,
                decomp,
                reserve_upDown_group=reserve_upDown_group,
                process_reserve_upDown_node=process_reserve_upDown_node,
            )

        layout = cls()
        layout._derive_blocks(
            solve=solve,
            solve_config=solve_config,
            timeline_config=timeline_config,
            nodes=nodes,
            units=units,
            connections=connections,
            resolution_groups=resolution_groups,
            group_unit=gu,
            group_connection=gc,
            group_node=gn,
            process_source_sink=process_source_sink,
            process_ct_method=process_ct_method,
        )
        layout._build_block_timelines(
            solve=solve,
            solve_config=solve_config,
            timeline_config=timeline_config,
            active_time_list=active_time_list,
        )
        layout._build_overlap_set()
        layout._build_block_predecessors(default_jump_list=default_jump_list)
        layout._build_block_boundaries()
        layout._materialize_frames()
        return layout

    # ------------------------------------------------------------------
    # Phase 1: block tagging (entity + process side)
    # ------------------------------------------------------------------

    def _derive_blocks(
        self,
        *,
        solve: str,
        solve_config,
        timeline_config,
        nodes: Iterable[str],
        units: Iterable[str],
        connections: Iterable[str],
        resolution_groups: dict[str, float],
        group_unit: list[tuple[str, str]],
        group_connection: list[tuple[str, str]],
        group_node: list[tuple[str, str]],
        process_source_sink: Iterable[tuple[str, str, str]],
        process_ct_method: dict[str, str],
    ) -> None:
        """Mirror ``blocks.derive_blocks``."""
        gu = group_unit
        gc = group_connection
        gn = group_node

        default_step_duration = _solve_step_duration(
            solve, solve_config, timeline_config
        )
        block_step_duration: dict[str, float] = {
            DEFAULT_BLOCK: default_step_duration,
        }
        for g, dur in resolution_groups.items():
            try:
                block_step_duration[g] = float(dur)
            except (TypeError, ValueError):
                continue

        # Node block.
        node_block: dict[str, str] = {}
        for n in nodes:
            g = _assign_entity_to_group(n, gn, resolution_groups)
            node_block[n] = g if g is not None else DEFAULT_BLOCK

        # Process source/sink — first source/sink per process.
        first_source: dict[str, str] = {}
        first_sink: dict[str, str] = {}
        for p, s, k in process_source_sink:
            first_source.setdefault(p, s)
            first_sink.setdefault(p, k)

        process_block_in: dict[str, str] = {}
        process_block_out: dict[str, str] = {}
        process_block: dict[str, str] = {}

        def _finer(a: str, b: str) -> str:
            da = block_step_duration.get(a, default_step_duration)
            db = block_step_duration.get(b, default_step_duration)
            return a if da <= db else b

        explicit_membership = list(gu) + list(gc)
        for p in list(units) + list(connections):
            explicit_block = _assign_entity_to_group(
                p, explicit_membership, resolution_groups
            )
            if explicit_block is not None:
                process_block_in[p] = explicit_block
                process_block_out[p] = explicit_block
                process_block[p] = explicit_block
                continue

            src = first_source.get(p)
            snk = first_sink.get(p)
            src_block = (
                node_block.get(src, DEFAULT_BLOCK) if src else DEFAULT_BLOCK
            )
            snk_block = (
                node_block.get(snk, DEFAULT_BLOCK) if snk else DEFAULT_BLOCK
            )

            ct = process_ct_method.get(p, "")
            if _is_direct(ct):
                finer = _finer(src_block, snk_block)
                process_block_in[p] = finer
                process_block_out[p] = finer
                process_block[p] = finer
            else:
                process_block_in[p] = src_block
                process_block_out[p] = snk_block
                process_block[p] = _finer(src_block, snk_block)

        self.node_block = node_block
        self.process_block_in = process_block_in
        self.process_block_out = process_block_out
        self.process_block = process_block
        self.block_step_duration = block_step_duration

    # ------------------------------------------------------------------
    # Phase 2: per-block timelines
    # ------------------------------------------------------------------

    def _build_block_timelines(
        self,
        *,
        solve: str,
        solve_config,
        timeline_config,
        active_time_list: dict[str, list] | None,
    ) -> None:
        """Mirror ``blocks._build_block_timelines``."""
        per_block: dict[str, dict[str, list[tuple[str, float]]]] = {}

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

        per_block[DEFAULT_BLOCK] = default_periods

        for block, dur in self.block_step_duration.items():
            if block == DEFAULT_BLOCK:
                continue
            # Skip blocks no entity is assigned to.
            if (
                block not in self.node_block.values()
                and block not in self.process_block_in.values()
                and block not in self.process_block_out.values()
                and block not in self.process_block.values()
            ):
                continue
            block_rows: dict[str, list[tuple[str, float]]] = {}
            for period, rows in default_periods.items():
                aggregated = _aggregate_timeline(
                    rows, coarse_duration=float(dur),
                )
                block_rows[period] = aggregated
            per_block[block] = block_rows

        self.per_block_timeline = per_block

    # ------------------------------------------------------------------
    # Phase 3: overlap set (cross-resolution)
    # ------------------------------------------------------------------

    def _build_overlap_set(self) -> None:
        """Mirror ``blocks.derive_overlap_set``.

        Caches per-(b_coarse, b_fine) coverage maps so repeated arc
        lookups don't re-walk the fine timeline (schematic
        recommendation — the materialised join can balloon for
        multi-block scenarios with ~30K rows).
        """
        per_block = self.per_block_timeline
        default_rows = per_block.get(DEFAULT_BLOCK, {})

        rows: list[tuple[str, str, str, str, str, float]] = []
        non_default = [b for b in per_block if b != DEFAULT_BLOCK]

        # Degenerate case: only the default block exists.
        if not non_default:
            for period, pr in default_rows.items():
                for step, _dur in pr:
                    rows.append(
                        (period, DEFAULT_BLOCK, step,
                         DEFAULT_BLOCK, step, 1.0)
                    )
            self._overlap_rows = rows
            return

        # Default-against-default identity rows.
        for period, pr in default_rows.items():
            for step, _dur in pr:
                rows.append(
                    (period, DEFAULT_BLOCK, step,
                     DEFAULT_BLOCK, step, 1.0)
                )

        # Cache: coverage[block][period] = {fine_step → coarse_step}.
        coverage: dict[str, dict[str, dict[str, str]]] = {}

        # For every resolution-group block, emit:
        #   * self-identity rows (b, t, b, t, 1.0)
        #   * coarse↔default fine rows
        #   * default↔coarse symmetric rows
        # Build coverage along the way.
        for coarse in non_default:
            coarse_rows = per_block.get(coarse, {})
            coverage[coarse] = {}
            # Self-identity.
            for period, c_rows in coarse_rows.items():
                for coarse_step, _dur in c_rows:
                    rows.append(
                        (period, coarse, coarse_step,
                         coarse, coarse_step, 1.0)
                    )
            for period, c_rows in coarse_rows.items():
                fine = default_rows.get(period, [])
                f_idx = 0
                coverage[coarse].setdefault(period, {})
                for coarse_step, coarse_dur in c_rows:
                    remaining = float(coarse_dur)
                    eps = 1e-9
                    while remaining > eps and f_idx < len(fine):
                        fine_step, fine_dur = fine[f_idx]
                        rows.append(
                            (period, coarse, coarse_step,
                             DEFAULT_BLOCK, fine_step, 1.0)
                        )
                        rows.append(
                            (period, DEFAULT_BLOCK, fine_step,
                             coarse, coarse_step, 1.0)
                        )
                        coverage[coarse][period][fine_step] = coarse_step
                        remaining -= float(fine_dur)
                        f_idx += 1
                    if remaining > eps:
                        raise NotImplementedError(
                            f"overlap_set: coarse step '{coarse_step}' on "
                            f"block '{coarse}' left {remaining}h unmatched "
                            f"at end of fine timeline — non-aligned subsets "
                            f"are not supported yet."
                        )

        # Pairs of resolution-group blocks: emit coarse→fine rows
        # (canonical direction) for blocks of different durations.
        if len(non_default) >= 2:
            for a in non_default:
                for b in non_default:
                    if a == b:
                        continue
                    a_dur = self.block_step_duration.get(a, 0.0)
                    b_dur = self.block_step_duration.get(b, 0.0)
                    # Only emit coarse→fine pairs to avoid duplication.
                    if a_dur > b_dur:
                        emit = True
                    elif a_dur == b_dur and a < b:
                        emit = True
                    else:
                        emit = False
                    if not emit:
                        continue
                    for period, a_cov in coverage[a].items():
                        b_cov = coverage[b].get(period, {})
                        for fine_step, a_step in a_cov.items():
                            b_step = b_cov.get(fine_step)
                            if b_step is None:
                                continue
                            rows.append(
                                (period, a, a_step, b, b_step, 1.0)
                            )

        # Deduplicate while preserving order.
        seen: set[tuple] = set()
        deduped: list[tuple[str, str, str, str, str, float]] = []
        for row in rows:
            if row in seen:
                continue
            seen.add(row)
            deduped.append(row)
        self._overlap_rows = deduped
        # Cache for downstream consumers.
        self._coverage_cache = coverage

    # ------------------------------------------------------------------
    # Phase 4: per-block predecessors + boundaries
    # ------------------------------------------------------------------

    def _build_block_predecessors(
        self, *, default_jump_list: Iterable[tuple] | None,
    ) -> None:
        """Mirror ``blocks.derive_block_predecessors``."""
        rows: list[tuple[str, str, str, str, str, str, str]] = []

        if default_jump_list is not None:
            for entry in default_jump_list:
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
            default_rows = self.per_block_timeline.get(DEFAULT_BLOCK, {})
            rows.extend(_cyclic_block_predecessors(DEFAULT_BLOCK, default_rows))

        for block in self.block_step_duration:
            if block == DEFAULT_BLOCK:
                continue
            block_rows_per_period = self.per_block_timeline.get(block, {})
            if not block_rows_per_period:
                continue
            rows.extend(_cyclic_block_predecessors(block, block_rows_per_period))

        self._predecessor_rows = rows

    def _build_block_boundaries(self) -> None:
        """Mirror ``blocks.derive_block_boundaries``."""
        first: list[tuple[str, str, str]] = []
        last: list[tuple[str, str, str]] = []
        for block in self.block_step_duration:
            rows_per_period = self.per_block_timeline.get(block, {})
            for period, rows in rows_per_period.items():
                if not rows:
                    continue
                first.append((block, period, rows[0][0]))
                last.append((block, period, rows[-1][0]))
        self._boundary_first_rows = first
        self._boundary_last_rows = last

    # ------------------------------------------------------------------
    # Phase 5: materialise polars frames
    # ------------------------------------------------------------------

    def _materialize_frames(self) -> None:
        """Build the polars DataFrame surface from the dict bookkeeping.

        Column names match the canonical CSV writers' headers so
        downstream callers can join directly without renaming.
        """
        # entity_block.csv — one row per node.
        eb_rows = list(self.node_block.items())
        self.entity_block_frame = pl.DataFrame(
            {
                "entity": [r[0] for r in eb_rows],
                "block": [r[1] for r in eb_rows],
            },
            schema={"entity": pl.Utf8, "block": pl.Utf8},
        )

        # process_side_block.csv — two rows per process (source + sink).
        psb_rows: list[tuple[str, str, str]] = []
        for process, block in self.process_block_in.items():
            psb_rows.append((process, "source", block))
        for process, block in self.process_block_out.items():
            psb_rows.append((process, "sink", block))
        self.process_side_block_frame = pl.DataFrame(
            {
                "process": [r[0] for r in psb_rows],
                "side": [r[1] for r in psb_rows],
                "block": [r[2] for r in psb_rows],
            },
            schema={"process": pl.Utf8, "side": pl.Utf8, "block": pl.Utf8},
        )

        # process_block.csv — process-unified block.
        pb_rows = list(self.process_block.items())
        self.process_block_frame = pl.DataFrame(
            {
                "process": [r[0] for r in pb_rows],
                "block": [r[1] for r in pb_rows],
            },
            schema={"process": pl.Utf8, "block": pl.Utf8},
        )

        # block_step_duration.csv — full per-block timeline.
        bsd_rows: list[tuple[str, str, str, float]] = []
        for block, period_rows in self.per_block_timeline.items():
            for period, rows in period_rows.items():
                for step, dur in rows:
                    bsd_rows.append((block, period, step, dur))
        self.block_step_duration_frame = pl.DataFrame(
            {
                "block": [r[0] for r in bsd_rows],
                "period": [r[1] for r in bsd_rows],
                "step": [r[2] for r in bsd_rows],
                "step_duration": [r[3] for r in bsd_rows],
            },
            schema={
                "block": pl.Utf8, "period": pl.Utf8,
                "step": pl.Utf8, "step_duration": pl.Float64,
            },
        )

        # overlap_set.csv — cross-resolution rows.
        ov_rows = self._overlap_rows
        self.overlap_set_frame = pl.DataFrame(
            {
                "period": [r[0] for r in ov_rows],
                "block_coarse": [r[1] for r in ov_rows],
                "step_coarse": [r[2] for r in ov_rows],
                "block_fine": [r[3] for r in ov_rows],
                "step_fine": [r[4] for r in ov_rows],
                "fraction": [r[5] for r in ov_rows],
            },
            schema={
                "period": pl.Utf8, "block_coarse": pl.Utf8,
                "step_coarse": pl.Utf8, "block_fine": pl.Utf8,
                "step_fine": pl.Utf8, "fraction": pl.Float64,
            },
        )

        # block_step_previous.csv — per-block predecessor rows.
        pred_rows = self._predecessor_rows
        self.block_step_previous_frame = pl.DataFrame(
            {
                "block": [r[0] for r in pred_rows],
                "period": [r[1] for r in pred_rows],
                "step": [r[2] for r in pred_rows],
                "step_previous": [r[3] for r in pred_rows],
                "step_previous_within_timeset": [r[4] for r in pred_rows],
                "period_previous": [r[5] for r in pred_rows],
                "step_previous_within_solve": [r[6] for r in pred_rows],
            },
            schema={
                "block": pl.Utf8, "period": pl.Utf8, "step": pl.Utf8,
                "step_previous": pl.Utf8,
                "step_previous_within_timeset": pl.Utf8,
                "period_previous": pl.Utf8,
                "step_previous_within_solve": pl.Utf8,
            },
        )

        # block_period_time_first.csv / _last.csv.
        first_rows = self._boundary_first_rows
        last_rows = self._boundary_last_rows
        self.block_period_time_first_frame = pl.DataFrame(
            {
                "block": [r[0] for r in first_rows],
                "period": [r[1] for r in first_rows],
                "step": [r[2] for r in first_rows],
            },
            schema={"block": pl.Utf8, "period": pl.Utf8, "step": pl.Utf8},
        )
        self.block_period_time_last_frame = pl.DataFrame(
            {
                "block": [r[0] for r in last_rows],
                "period": [r[1] for r in last_rows],
                "step": [r[2] for r in last_rows],
            },
            schema={"block": pl.Utf8, "period": pl.Utf8, "step": pl.Utf8},
        )


__all__ = [
    "DEFAULT_BLOCK",
    "BlockLayout",
    "validate_group_membership",
]
