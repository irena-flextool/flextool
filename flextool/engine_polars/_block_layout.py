"""BlockLayout — multi-resolution timestep grouping (Δ.2).

Native ``engine_polars`` port of
``flextool/engine_polars/_blocks.py`` (1368 LOC).  Produces, in one
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

Reference: ``flextool/engine_polars/_blocks.py`` (read-only).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

import polars as pl

from flextool.engine_polars._axis_enums import (
    get_global_axis_enums,
    rename_to_axis,
    schema_dtype,
)
from flextool.engine_polars._solve_state import FlexToolConfigError


# Substrate handle for the cascade-wide axis enum vocabulary.
# Bare ``None`` here; ``cast_dim`` / ``schema_dtype`` in
# ``_axis_enums`` fall back to ``_LIVE_AXIS_ENUMS_CTX`` (the live
# ContextVar) when this is ``None``, so substrate sites pick up
# activation set by ``load_flextool`` automatically.
_enums: "dict | None" = None

if TYPE_CHECKING:  # pragma: no cover — import cycle guard only
    from flextool.engine_polars._solve_config import SolveConfig
    from flextool.engine_polars._timeline import TimelineConfig


# ---------------------------------------------------------------------------
# Public constants & schema
# ---------------------------------------------------------------------------


DEFAULT_BLOCK = "default"


# Eight per-solve block CSV stems written by
# ``flextool.engine_polars._blocks.write_block_data`` and consumed by
# :meth:`BlockLayout.load_from_solve_data`.  Defined as a module
# constant so the off-cascade workdir bridge can seed only these files
# into the ephemeral Provider (without picking up unrelated workdir
# artefacts like ``solve_progress.csv`` whose ragged shape would fail
# the strict CSV reader).
_BLOCK_CSV_STEMS: tuple[str, ...] = (
    "entity_block",
    "process_side_block",
    "process_block",
    "block_step_duration",
    "overlap_set",
    "block_step_previous",
    "block_period_time_first",
    "block_period_time_last",
)


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

    Mirrors ``flextool/engine_polars/_blocks.validate_group_membership``
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
    # Source-only constructor (Phase 1 of the fast-path multi-block plan)
    # ------------------------------------------------------------------

    @classmethod
    def from_source(
        cls,
        source,
        solve_config: "SolveConfig",
        timeline_config: "TimelineConfig",
        *,
        active_solve: str | None = None,
        validate: bool = True,
    ) -> "BlockLayout":
        """Construct a :class:`BlockLayout` directly from a Spine
        :class:`~flextool.engine_polars._input_source.InputSource`.

        Source-only counterpart to :meth:`load_from_solve_data`: pulls
        every input that :meth:`build` needs from the source / configs
        (no workdir CSV reads), then delegates.  Intended for the fast
        single-solve path where ``solve_data/`` is empty.

        Parameters
        ----------
        source : InputSource
            Scenario-resolved per-(entity_class, parameter_name)
            reader (e.g. :class:`SpineDbReader`,
            :class:`InMemoryReader`).
        solve_config, timeline_config :
            Already-loaded configs.  ``timeline_config`` must have had
            :meth:`TimelineConfig.create_assumptive_parts` /
            :meth:`TimelineConfig.create_timeline_from_timestep_duration`
            applied so :func:`_timeline.get_active_time` succeeds for
            ``active_solve``.
        active_solve : str | None
            Active solve name.  When ``None`` we pick the first solve
            entity in the scenario (single-solve fixtures).
        validate : bool
            Forwarded to :meth:`build` — when ``True`` runs
            :func:`validate_group_membership` on the assembled inputs.

        Returns
        -------
        BlockLayout
            Fully populated, identical surface to what
            :meth:`load_from_solve_data` produces from the slow-path's
            ``solve_data/`` block CSVs (on the same scenario).

        Notes
        -----
        Phase 1 of the multi-block fast-path plan only builds the
        constructor — it is **not** yet wired into ``_fast_load.py``.
        See the audit doc / Phase-2 task for the wiring step.
        """
        # ── 1. Resolve the active solve name ───────────────────────────
        from flextool.engine_polars._projection_params import (
            _try_entities,
            _try_param,
        )

        if active_solve is None:
            solves_param = _try_param(source, "model", "solves")
            if solves_param is not None and solves_param.height > 0:
                val_col = (
                    "value" if "value" in solves_param.columns
                    else solves_param.columns[-1]
                )
                active_solve = str(solves_param[val_col][0])
            else:
                solve_ents = _try_entities(source, "solve")
                if solve_ents is None or solve_ents.height == 0:
                    raise FlexToolConfigError(
                        "BlockLayout.from_source: no active solve could be "
                        "resolved — pass ``active_solve`` explicitly or "
                        "populate ``model.solves`` / the ``solve`` entity "
                        "class in the source."
                    )
                name_col = next(
                    c for c in solve_ents.columns if c != "id"
                )
                active_solve = str(solve_ents[name_col][0])

        # ── 2. Entity universes ────────────────────────────────────────
        units_df = _try_entities(source, "unit")
        units = units_df["name"].to_list() if units_df is not None else []
        conns_df = _try_entities(source, "connection")
        connections = (
            conns_df["name"].to_list() if conns_df is not None else []
        )
        nodes_df = _try_entities(source, "node")
        nodes = nodes_df["name"].to_list() if nodes_df is not None else []

        # ── 3. Group resolution (new_stepduration / decomposition_method)
        resolution_groups: dict[str, float] = {}
        rg_param = _try_param(source, "group", "new_stepduration")
        if rg_param is not None:
            name_col = (
                "name" if "name" in rg_param.columns
                else rg_param.columns[0]
            )
            for row in rg_param.iter_rows(named=True):
                v = row["value"]
                if v is None:
                    continue
                try:
                    resolution_groups[row[name_col]] = float(v)
                except (TypeError, ValueError):
                    continue

        decomposition_groups: dict[str, str] = {}
        dg_param = _try_param(source, "group", "decomposition_method")
        if dg_param is not None:
            name_col = (
                "name" if "name" in dg_param.columns
                else dg_param.columns[0]
            )
            for row in dg_param.iter_rows(named=True):
                v = row["value"]
                if v is None:
                    continue
                decomposition_groups[row[name_col]] = str(v)

        # ── 4. Group memberships ───────────────────────────────────────
        def _pairs(cls_name: str, dim: str) -> list[tuple[str, str]]:
            df = _try_entities(source, cls_name)
            if df is None or df.height == 0:
                return []
            return list(zip(df["group"].to_list(), df[dim].to_list()))

        group_node = _pairs("group__node", "node")
        group_unit = _pairs("group__unit", "unit")
        group_connection = _pairs("group__connection", "connection")

        # ── 5. Reserve membership (optional; absent on most fixtures) ──
        reserve_upDown_group: list[tuple[str, str, str]] = []
        rug_df = _try_entities(source, "reserve__upDown__group")
        if rug_df is not None and rug_df.height > 0:
            # Filter rows whose ``method`` (if defined) is not ``no_reserve``.
            method_param = _try_param(
                source, "reserve__upDown__group", "method",
            )
            excluded: set[tuple[str, str, str]] = set()
            if method_param is not None:
                for row in method_param.iter_rows(named=True):
                    if str(row.get("value", "")) == "no_reserve":
                        excluded.add(
                            (row["reserve"], row["upDown"], row["group"])
                        )
            for r, ud, g in zip(
                rug_df["reserve"].to_list(),
                rug_df["upDown"].to_list(),
                rug_df["group"].to_list(),
            ):
                if (r, ud, g) in excluded:
                    continue
                reserve_upDown_group.append((r, ud, g))

        process_reserve_upDown_node: list[tuple[str, str, str, str]] = []
        for cls_name, p_col in (
            ("reserve__upDown__unit__node", "unit"),
            ("reserve__upDown__connection__node", "connection"),
        ):
            df = _try_entities(source, cls_name)
            if df is None or df.height == 0:
                continue
            for p, r, ud, n in zip(
                df[p_col].to_list(),
                df["reserve"].to_list(),
                df["upDown"].to_list(),
                df["node"].to_list(),
            ):
                process_reserve_upDown_node.append((p, r, ud, n))

        # ── 6. process_source_sink (first source / first sink per process)
        # Built directly from ``unit__inputNode`` / ``unit__outputNode`` /
        # ``connection__node__node`` — matching the slow path's
        # ``process__source.csv`` / ``process__sink.csv`` derivation
        # (input_writer.py emits one row per (process, input-or-output
        # node) pair; first one wins per `derive_blocks`'s usage).
        first_source: dict[str, str] = {}
        first_sink: dict[str, str] = {}
        uin = _try_entities(source, "unit__inputNode")
        if uin is not None:
            for u, n in zip(uin["unit"].to_list(), uin["node"].to_list()):
                first_source.setdefault(u, n)
        uout = _try_entities(source, "unit__outputNode")
        if uout is not None:
            for u, n in zip(uout["unit"].to_list(), uout["node"].to_list()):
                first_sink.setdefault(u, n)
        cnn = _try_entities(source, "connection__node__node")
        if cnn is not None:
            for c, a, b in zip(
                cnn["connection"].to_list(),
                cnn["node_1"].to_list(),
                cnn["node_2"].to_list(),
            ):
                first_source.setdefault(c, a)
                first_sink.setdefault(c, b)
        process_source_sink: list[tuple[str, str, str]] = []
        all_procs = set(first_source) | set(first_sink)
        for p in all_procs:
            process_source_sink.append(
                (p, first_source.get(p, ""), first_sink.get(p, ""))
            )

        # ── 7. process_ct_method ──────────────────────────────────────
        # Slow-path ``process__ct_method.csv`` = union of
        # ``unit.conversion_method`` and ``connection.transfer_method``
        # (input_writer.py:394).
        process_ct_method: dict[str, str] = {}
        cm_unit = _try_param(source, "unit", "conversion_method")
        if cm_unit is not None:
            for row in cm_unit.iter_rows(named=True):
                process_ct_method[row["name"]] = str(row["value"])
        cm_conn = _try_param(source, "connection", "transfer_method")
        if cm_conn is not None:
            for row in cm_conn.iter_rows(named=True):
                process_ct_method[row["name"]] = str(row["value"])

        # ── 8. active_time_list — derived from timeline + solve configs.
        from flextool.engine_polars._timeline import get_active_time

        active_time_list = get_active_time(
            current_solve=active_solve,
            timesets_used_by_solves=(
                solve_config.timesets_used_by_solves
            ),
            timesets=timeline_config.timeset_durations,
            timelines=timeline_config.timelines,
            timesets__timelines=timeline_config.timesets__timeline,
        )

        # ── 9. default_jump_list:
        # Phase-1 scope deliberately defers building this from
        # ``make_step_jump`` (needs ``period__branch`` /
        # ``solve_branch__time_branch_list``, which require additional
        # source plumbing).  Passing ``None`` lets ``build`` fall back
        # to ``_cyclic_block_predecessors`` — on simple single-period
        # non-stochastic fixtures the slow-path's ``step_previous.csv``
        # produces the same rows (verified by the parity test below
        # and by the existing ``test_load_from_solve_data_bridge_*``).
        return cls.build(
            solve=active_solve,
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
            decomposition_groups=decomposition_groups,
            reserve_upDown_group=reserve_upDown_group or None,
            process_reserve_upDown_node=(
                process_reserve_upDown_node or None
            ),
            active_time_list=active_time_list,
            default_jump_list=None,
            validate=validate,
        )

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
            schema={"entity": schema_dtype(_enums, "entity"),
                    "block": pl.Utf8},
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
            schema={"process": schema_dtype(_enums, "process"),
                    "side": pl.Utf8, "block": pl.Utf8},
        )

        # process_block.csv — process-unified block.
        pb_rows = list(self.process_block.items())
        self.process_block_frame = pl.DataFrame(
            {
                "process": [r[0] for r in pb_rows],
                "block": [r[1] for r in pb_rows],
            },
            schema={"process": schema_dtype(_enums, "process"),
                    "block": pl.Utf8},
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
            schema={"block": pl.Utf8,
                    "period": schema_dtype(_enums, "period"),
                    "step": schema_dtype(_enums, "step")},
        )
        self.block_period_time_last_frame = pl.DataFrame(
            {
                "block": [r[0] for r in last_rows],
                "period": [r[1] for r in last_rows],
                "step": [r[2] for r in last_rows],
            },
            schema={"block": pl.Utf8,
                    "period": schema_dtype(_enums, "period"),
                    "step": schema_dtype(_enums, "step")},
        )


    # ------------------------------------------------------------------
    # CSV bridge — load from flextool's solve_data/ CSVs
    # ------------------------------------------------------------------

    @classmethod
    def load_from_solve_data(
        cls,
        solve_data_dir,
        *,
        missing_ok: bool = True,
        provider: "object | None" = None,
    ) -> "BlockLayout":
        """Load a ``BlockLayout`` from a :class:`FlexDataProvider`.

        Step 2.5 Phase A: ``blocks.write_block_data`` is now patched
        into ``_PATCH_MODULES`` so the eight per-solve block frames
        live in the Provider after the cascade's preprocessing pass.
        Callers MUST pass *provider*.  When only a workdir is available
        (off-cascade tests, transitional bridges), use
        :func:`flextool.engine_polars._input_source.seed_provider_from_dir`
        on ``workdir/"solve_data"`` to construct an ephemeral Provider
        first.

        Parameters
        ----------
        solve_data_dir : Path
            Retained for symmetry with the legacy signature; unused
            when *provider* carries every requested key.  The
            classmethod will refuse to read disk directly — the
            workdir-bridge contract is now Provider-only.
        missing_ok : bool
            When True (default), missing Provider keys produce empty
            frames.  When False, raise ``KeyError`` on the first
            missing key.
        provider :
            :class:`FlexDataProvider` to serve frames from.  Required.

        Returns
        -------
        BlockLayout
            Populated layout.  ``per_block_timeline`` is reconstructed
            from ``block_step_duration_frame`` rows.
        """
        if provider is None:
            # Off-cascade bridge path: seed an ephemeral Provider from
            # the workdir CSV directory and serve from that.  Cascade
            # callers always pass *provider* explicitly; this branch is
            # reserved for tests and the legacy ``load_block_bundle``
            # contract.  Centralised here so the Rule 1 audit can stay
            # strict — no bare ``pl.read_csv`` / ``_read_csv_file`` in
            # this module.
            from pathlib import Path as _Path
            from flextool.engine_polars._flex_data_provider import (
                FlexDataProvider,
            )
            from flextool.engine_polars._input_source import (
                seed_provider_from_dir,
            )
            sd_dir = _Path(solve_data_dir)
            provider = FlexDataProvider()
            seed_provider_from_dir(
                provider, sd_dir, "solve_data",
                names=_BLOCK_CSV_STEMS,
            )

        def _read(name: str, schema: dict) -> pl.DataFrame:
            stem = name.removesuffix(".csv")
            k = f"solve_data/{stem}"
            if provider.has(k):
                df = provider.get(k)
                # Coerce dtypes — Provider may carry Utf8 for
                # numeric columns when the writer's _empty_frame
                # shortcut hits.
                for col, dt in schema.items():
                    if col in df.columns and df.schema[col] != dt:
                        df = df.with_columns(
                            pl.col(col).cast(dt, strict=False),
                        )
                return df
            if missing_ok:
                return pl.DataFrame(schema=schema)
            raise KeyError(
                f"BlockLayout.load_from_solve_data: provider has no "
                f"key for '{name}' (tried 'solve_data/{stem}' and "
                f"'{stem}')."
            )

        layout = cls()
        layout.entity_block_frame = _read(
            "entity_block.csv",
            {"entity": pl.Utf8, "block": pl.Utf8},
        )
        layout.process_side_block_frame = _read(
            "process_side_block.csv",
            {"process": pl.Utf8, "side": pl.Utf8, "block": pl.Utf8},
        )
        layout.process_block_frame = _read(
            "process_block.csv",
            {"process": pl.Utf8, "block": pl.Utf8},
        )
        layout.block_step_duration_frame = _read(
            "block_step_duration.csv",
            {
                "block": pl.Utf8, "period": pl.Utf8,
                "step": pl.Utf8, "step_duration": pl.Float64,
            },
        )
        layout.overlap_set_frame = _read(
            "overlap_set.csv",
            {
                "period": pl.Utf8, "block_coarse": pl.Utf8,
                "step_coarse": pl.Utf8, "block_fine": pl.Utf8,
                "step_fine": pl.Utf8, "fraction": pl.Float64,
            },
        )
        layout.block_step_previous_frame = _read(
            "block_step_previous.csv",
            {
                "block": pl.Utf8, "period": pl.Utf8, "step": pl.Utf8,
                "step_previous": pl.Utf8,
                "step_previous_within_timeset": pl.Utf8,
                "period_previous": pl.Utf8,
                "step_previous_within_solve": pl.Utf8,
            },
        )
        layout.block_period_time_first_frame = _read(
            "block_period_time_first.csv",
            {"block": pl.Utf8, "period": pl.Utf8, "step": pl.Utf8},
        )
        layout.block_period_time_last_frame = _read(
            "block_period_time_last.csv",
            {"block": pl.Utf8, "period": pl.Utf8, "step": pl.Utf8},
        )

        # Reconstruct internal bookkeeping dicts from frames so callers
        # can index by entity name.
        if layout.entity_block_frame.height > 0:
            layout.node_block = dict(zip(
                layout.entity_block_frame["entity"].to_list(),
                layout.entity_block_frame["block"].to_list(),
            ))
        if layout.process_side_block_frame.height > 0:
            for row in layout.process_side_block_frame.iter_rows(named=True):
                if row["side"] == "source":
                    layout.process_block_in[row["process"]] = row["block"]
                elif row["side"] == "sink":
                    layout.process_block_out[row["process"]] = row["block"]
        if layout.process_block_frame.height > 0:
            layout.process_block = dict(zip(
                layout.process_block_frame["process"].to_list(),
                layout.process_block_frame["block"].to_list(),
            ))

        # block_step_duration: dict[block → step_duration] uses the
        # first step's duration for each block (every step in a block
        # shares the same duration by construction).
        bsd = layout.block_step_duration_frame
        if bsd.height > 0:
            layout.block_step_duration = dict(
                bsd.group_by("block")
                   .agg(pl.col("step_duration").first())
                   .iter_rows()
            )
            # per_block_timeline reconstruction.
            for row in bsd.iter_rows(named=True):
                blk = row["block"]
                period = row["period"]
                layout.per_block_timeline.setdefault(blk, {}).setdefault(
                    period, [],
                ).append((row["step"], row["step_duration"]))
        return layout

    # ------------------------------------------------------------------
    # Convenience accessors for downstream consumers
    # ------------------------------------------------------------------

    def block_compat(self) -> pl.DataFrame:
        """Return the (bk, b_f) → 1 compatibility set used by the
        block-aware filtering of flow_to_n / flow_from_n.

        The block axis column is named ``bk`` (not ``b``) to disambiguate
        from the branch axis — see the b_collision review note in
        ``schemas/flextool_axis_contract.json``.  ``b_f`` is the
        block-fine companion column (also block vocabulary, but distinct
        column name so it doesn't collide).

        Equivalent to::

            overlap_set_frame
                .rename({"block_coarse": "bk", "block_fine": "b_f"})
                .select("bk", "b_f").unique()

        Cached attribute would help on hot paths; the materialised set
        is small (≤ |blocks|² ≈ tens of rows).
        """
        if self.overlap_set_frame.height == 0:
            return pl.DataFrame(schema={
                "bk": schema_dtype(_enums, "bk"),
                "b_f": schema_dtype(_enums, "b_f"),
            })
        return (
            self.overlap_set_frame
            .pipe(rename_to_axis, {"block_coarse": "bk", "block_fine": "b_f"})
            .select("bk", "b_f").unique()
        )

    def coarse_blocks(self, threshold: float = 1.0) -> list[str]:
        """Return blocks with at least one row of ``step_duration >
        threshold``.

        Mirrors the ``coarse_blocks`` selection in
        ``input.py``'s nodeStateBlock synthesis (lines 2013-2040).
        """
        if self.block_step_duration_frame.height == 0:
            return []
        return (
            self.block_step_duration_frame
            .filter(pl.col("step_duration") > threshold)
            ["block"].unique().to_list()
        )

    def is_empty(self) -> bool:
        """Return True if the layout carries no block data at all
        (every frame is empty).  Useful for the ``missing_ok`` branch
        of the CSV-loading bridge."""
        return (
            self.entity_block_frame.height == 0
            and self.block_step_duration_frame.height == 0
            and self.overlap_set_frame.height == 0
        )


__all__ = [
    "DEFAULT_BLOCK",
    "BlockLayout",
    "validate_group_membership",
]
