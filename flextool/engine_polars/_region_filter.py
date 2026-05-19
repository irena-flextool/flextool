"""Regional filter for Lagrangian decomposition (gaps A4 + A5).

Slices a whole-system :class:`FlexData` into N per-region :class:`FlexData`s.
Cross-region arcs (rows in ``process_source_sink`` whose ``source`` and
``sink`` straddle two regions) are *severed* into two virtual one-way
half-flow arcs, one in each region.

The virtual entities are pure bookkeeping — the half-flow on the export
side and the half-flow on the import side carry the **same flow** at
optimality (the Lagrangian coupling constraint that
:mod:`flextool._lagrangian` prices via λ).  Inside a region's standalone
LP, the half-flow column is just an ordinary ``v_flow`` column with:

* ``p_unitsize = 1`` (same units as the original column);
* ``p_flow_upper`` = original pipe capacity (so dispatch can push flow
  at full pipe capacity even with λ=0);
* the in-region terminal node enters ``flow_to_n`` (export) or
  ``flow_from_n`` (import) so the half-flow contributes to its
  nodeBalance;
* the *virtual* node sits OUTSIDE ``nodeBalance`` so the half-flow's
  other terminal is free (no balance pin, no penalty).

This module is a no-op when the input :class:`FlexData` has no
``decomposition_method=lagrangian_region`` group params.

Public surface
--------------
* :class:`HalfFlow`        — one severed arc; pairs across regions.
* :class:`RegionSplit`     — output of :func:`split` for one region.
* :func:`discover_regions` — returns ``[region_name, ...]`` from a
  whole-system FlexData (reads ``p_group_decomposition.csv`` indirectly
  via the populated ``group_entity`` / ``group_node`` frames).
* :func:`split`            — produces ``list[RegionSplit]``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import dataclasses
from typing import Iterable

import polars as pl

from polar_high import Param

from flextool.engine_polars.input import FlexData
from flextool.engine_polars._axis_enums import (
    cast_dim,
    get_global_axis_enums,
    reset_global_axis_enums,
    schema_dtype,
    set_global_axis_enums,
)


__all__ = [
    "HalfFlow",
    "RegionSplit",
    "discover_regions",
    "split",
    "load_decomposition_method",
    "load_region_membership",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class HalfFlow:
    """One half-flow arc replacing one direction of a cross-region pipe.

    A bi-directional cross-region pipe ``pipe(A→B, B→A)`` produces FOUR
    HalfFlows: ``pipe(A→B)`` becomes an export in A and an import in B;
    ``pipe(B→A)`` becomes an export in B and an import in A.  Each
    coupling pair shares the same ``(original_p, original_source,
    original_sink)`` triple — the :mod:`flextool._lagrangian` coordinator
    pairs export and import on that key.
    """
    region: str
    side: str  # "export" or "import"
    # Original cross-region arc identity (the pairing key).
    original_p: str
    original_source: str
    original_sink: str
    # In-region terminal node — appears in the regional flow_to_n /
    # flow_from_n alongside the half-flow.  For an export this is the
    # original source; for an import it is the original sink.
    in_region_node: str
    # Virtual entities introduced by this half-flow.
    virtual_node: str
    # Virtual connection: the (p, source, sink) row that replaces the
    # original cross-region arc inside the region's frames.
    virtual_p: str
    virtual_arc_source: str
    virtual_arc_sink: str


@dataclass
class RegionSplit:
    """One region's filtered :class:`FlexData` plus coupling metadata."""
    region: str
    data: FlexData
    half_flows: list[HalfFlow] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Decomposition discovery
# ---------------------------------------------------------------------------


def load_decomposition_method(
    work_dir: "Path | str | None" = None,
    *,
    provider: "object | None" = None,
) -> dict[str, str]:
    """Return ``{group: method}`` from
    ``input/p_group_decomposition.csv`` (Step 2.6 Provider-first).

    Resolution order:

    1. *provider* carries ``input/p_group_decomposition`` →
       read from memory.
    2. *provider* is ``None`` AND *work_dir* points at a workdir with
       the file on disk → seed an ephemeral Provider from that
       directory and serve from memory.
    3. Otherwise → ``{}``.

    The whole-system loader doesn't surface the decomposition_method
    parameter in :class:`FlexData` because it's only used by the
    decomposition driver itself; this helper plus the Provider keep
    it out of cascade memory until the driver needs it.
    """
    key = "input/p_group_decomposition"
    df: pl.DataFrame | None = None
    if provider is not None and provider.has(key):
        df = provider.get(key)
    elif provider is None and work_dir is not None:
        # Off-cascade test bridge: seed from disk via the centralised
        # helper so Rule 1 of test_meta_provider_invariants stays clean
        # (no bare ``pl.read_csv`` / ``_read_csv_file`` in this module).
        path = Path(work_dir) / "input" / "p_group_decomposition.csv"
        if not path.exists():
            return {}
        from flextool.engine_polars._flex_data_provider import (
            FlexDataProvider,
        )
        from flextool.engine_polars._input_source import (
            seed_provider_from_dir,
        )
        local = FlexDataProvider()
        seed_provider_from_dir(
            local, Path(work_dir) / "input", "input",
            names=("p_group_decomposition",),
        )
        if local.has(key):
            df = local.get(key)
    if df is None or df.height == 0:
        return {}
    cols = df.columns
    # Expected columns: group, groupParam, p_group
    if "group" not in cols:
        return {}
    val_col = next((c for c in ("p_group", "value") if c in cols), None)
    if val_col is None:
        return {}
    rows = (
        df.filter(pl.col("groupParam") == "decomposition_method")
        if "groupParam" in cols else df
    )
    return {r["group"]: r[val_col] for r in rows.iter_rows(named=True)}


def discover_regions(
    work_dir: "Path | str | None" = None,
    *,
    provider: "object | None" = None,
) -> list[str]:
    """Return the list of group names with
    ``decomposition_method=lagrangian_region``."""
    methods = load_decomposition_method(work_dir, provider=provider)
    return sorted(g for g, m in methods.items() if m == "lagrangian_region")


def load_region_membership(
    data: FlexData, regions: list[str],
) -> dict[str, dict[str, set[str]]]:
    """Return ``{region: {"nodes": set, "processes": set}}`` from
    ``data.group_entity`` / ``data.group_node`` for the given regions.

    Falls back to empty sets when the relevant frames are absent (a
    same-shape stub useful in unit tests).
    """
    out: dict[str, dict[str, set[str]]] = {}
    nodes_by_g: dict[str, set[str]] = {}
    procs_by_g: dict[str, set[str]] = {}
    if data.group_node is not None and data.group_node.height > 0:
        for r in data.group_node.iter_rows(named=True):
            nodes_by_g.setdefault(r["g"], set()).add(r["n"])
    if data.group_entity is not None and data.group_entity.height > 0:
        # group_entity is the union (g, e) — to get processes we'd need
        # to know which e's are processes.  process_unit + connections
        # set is everything in process_source_sink["p"].
        all_procs: set[str] = set()
        if data.process_source_sink is not None:
            all_procs |= set(data.process_source_sink["p"].unique().to_list())
        if data.process_indirect is not None:
            all_procs |= set(data.process_indirect["p"].unique().to_list())
        for r in data.group_entity.iter_rows(named=True):
            if r["e"] in all_procs:
                procs_by_g.setdefault(r["g"], set()).add(r["e"])
    for g in regions:
        out[g] = {
            "nodes": set(nodes_by_g.get(g, set())),
            "processes": set(procs_by_g.get(g, set())),
        }
    return out


# ---------------------------------------------------------------------------
# Helpers for filtering polars frames / Params
# ---------------------------------------------------------------------------


def _is_in_keep(col: str, keep: set[str]) -> pl.Expr:
    """Membership test for ``pl.col(col)`` against *keep* that tolerates
    keep elements outside the column's *original* Enum vocabulary by
    upcasting the column to the live (widened) Enum first.

    ``keep`` is built by the region splitter and may include synthetic
    virtual-entity tokens (``hf_pipe_*`` / ``*__export__*`` /
    ``*__import__*``) that the Spine-DB-derived axis_enums don't
    contain.  ``split()`` widens the global axis_enums ContextVar to
    include those virtual tokens before the filter runs; here we
    upcast the column to that widened vocabulary via
    :func:`cast_dim` (``enums=None`` reads the live ContextVar).  The
    upcast is a strict superset operation (every original value is in
    the wider Enum), the ``is_in`` then succeeds natively, and the
    filter's output retains the widened Enum dtype.
    """
    return cast_dim(pl.col(col), None, col).is_in(list(keep))


def _filter_frame(df: pl.DataFrame | None, col: str,
                  keep: set[str]) -> pl.DataFrame | None:
    if df is None:
        return None
    if col not in df.columns:
        return df
    return df.filter(_is_in_keep(col, keep))


def _filter_frame_multi(df: pl.DataFrame | None,
                        cond_cols: list[tuple[str, set[str]]]) -> pl.DataFrame | None:
    if df is None:
        return None
    out = df
    for col, keep in cond_cols:
        if col in out.columns:
            out = out.filter(_is_in_keep(col, keep))
    return out


def _filter_param(p: Param | None, col: str,
                  keep: set[str]) -> Param | None:
    if p is None:
        return None
    if col not in p.dims:
        return p
    new_frame = p.frame.filter(_is_in_keep(col, keep))
    return Param(p.dims, new_frame, name=p.name)


# ---------------------------------------------------------------------------
# Cross-region classification
# ---------------------------------------------------------------------------


def _classify_arcs(
    pss: pl.DataFrame, region_nodes: dict[str, set[str]],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Tag each (p, source, sink) row with its source-region and
    sink-region, then split into local and cross-region rows.

    Nodes not in any region are "shared" — an arc with a shared endpoint
    is treated as local-to-the-other-region (it stays in that region's
    frames and is not a coupling).
    """
    # Build a node→region map; nodes outside any region map to None.
    node_region: dict[str, str | None] = {}
    for r, ns in region_nodes.items():
        for n in ns:
            node_region[n] = r
    src_r = pss["source"].map_elements(
        lambda n: node_region.get(n), return_dtype=pl.Utf8)
    snk_r = pss["sink"].map_elements(
        lambda n: node_region.get(n), return_dtype=pl.Utf8)
    pss_tagged = pss.with_columns(
        _src_region=src_r, _snk_region=snk_r,
    )
    cross = pss_tagged.filter(
        pl.col("_src_region").is_not_null() &
        pl.col("_snk_region").is_not_null() &
        (pl.col("_src_region") != pl.col("_snk_region"))
    )
    return pss_tagged, cross


def _make_half_flows(cross_arcs: pl.DataFrame) -> dict[str, list[HalfFlow]]:
    """For each cross-region arc, produce two HalfFlow records (one
    per region)."""
    out: dict[str, list[HalfFlow]] = {}
    for r in cross_arcs.iter_rows(named=True):
        p = r["p"]
        s = r["source"]
        k = r["sink"]
        ra = r["_src_region"]
        rb = r["_snk_region"]
        # Naming: replicate flextool's convention loosely — the virtual
        # *node* uses the (p, terminal, region) stem; the virtual
        # *connection* uses the same stem with an ``hf_`` prefix.  We
        # disambiguate per-direction by encoding the original (s, k)
        # pair into the connection id so two-direction pipes don't
        # collide.
        ve_node = f"{p}__{s}__{k}__export__{ra}"
        vi_node = f"{p}__{s}__{k}__import__{rb}"
        ve_conn = f"hf_{p}__{s}__{k}__export__{ra}"
        vi_conn = f"hf_{p}__{s}__{k}__import__{rb}"
        out.setdefault(ra, []).append(HalfFlow(
            region=ra, side="export",
            original_p=p, original_source=s, original_sink=k,
            in_region_node=s,
            virtual_node=ve_node,
            virtual_p=ve_conn,
            virtual_arc_source=s,
            virtual_arc_sink=ve_node,
        ))
        out.setdefault(rb, []).append(HalfFlow(
            region=rb, side="import",
            original_p=p, original_source=s, original_sink=k,
            in_region_node=k,
            virtual_node=vi_node,
            virtual_p=vi_conn,
            virtual_arc_source=vi_node,
            virtual_arc_sink=k,
        ))
    return out


# ---------------------------------------------------------------------------
# Per-region splitter
# ---------------------------------------------------------------------------


def _build_region_data(
    src: FlexData,
    region: str,
    keep_nodes: set[str],
    keep_procs: set[str],
    half_flows: list[HalfFlow],
    cross_arcs_by_pss: set[tuple[str, str, str]],
) -> FlexData:
    """Construct one region's :class:`FlexData` by filtering+rewriting
    the whole-system frames/Params.

    ``keep_nodes``/``keep_procs`` are the in-region+shared sets.
    ``cross_arcs_by_pss`` is the SET of (p, source, sink) tuples to
    REMOVE from this region's process frames (they're being replaced
    by half-flow virtual arcs).
    """
    # Start by shallow-copying the dataclass and clearing fields we'll
    # explicitly rewrite.
    new = dataclasses.replace(src)

    # ---- Filter primary entity sets ----
    new.nodeBalance = _filter_frame(src.nodeBalance, "n", keep_nodes)
    new.nodeBalance_dt = _filter_frame(src.nodeBalance_dt, "n", keep_nodes)
    new.p_inflow = _filter_param(src.p_inflow, "n", keep_nodes)
    new.p_penalty_up = _filter_param(src.p_penalty_up, "n", keep_nodes)
    new.p_penalty_down = _filter_param(src.p_penalty_down, "n", keep_nodes)

    # ---- Filter process_source_sink and arc-side frames ----
    def _drop_cross(df: pl.DataFrame | None) -> pl.DataFrame | None:
        if df is None:
            return None
        if not all(c in df.columns for c in ("p", "source", "sink")):
            return df
        if not cross_arcs_by_pss:
            return df
        # Build a 3-col anti-join key.
        _enums = getattr(src, "_axis_enums", None)
        key_df = pl.DataFrame({
            "p":      [t[0] for t in cross_arcs_by_pss],
            "source": [t[1] for t in cross_arcs_by_pss],
            "sink":   [t[2] for t in cross_arcs_by_pss],
        }, schema={"p": schema_dtype(_enums, "p"),
                   "source": schema_dtype(_enums, "source"),
                   "sink": schema_dtype(_enums, "sink")})
        return df.join(key_df, on=("p", "source", "sink"), how="anti")

    def _filter_arc_by_proc(df: pl.DataFrame | None) -> pl.DataFrame | None:
        df = _drop_cross(df)
        if df is None or "p" not in df.columns:
            return df
        return df.filter(_is_in_keep("p", keep_procs))

    def _filter_param_arc(p: Param | None) -> Param | None:
        if p is None:
            return None
        if "p" not in p.dims:
            return p
        # Drop cross-region rows by triple-key, then filter to keep procs.
        f = p.frame
        if cross_arcs_by_pss and all(c in f.columns
                                     for c in ("p", "source", "sink")):
            _enums = getattr(src, "_axis_enums", None)
            key_df = pl.DataFrame({
                "p":      [t[0] for t in cross_arcs_by_pss],
                "source": [t[1] for t in cross_arcs_by_pss],
                "sink":   [t[2] for t in cross_arcs_by_pss],
            }, schema={"p": schema_dtype(_enums, "p"),
                       "source": schema_dtype(_enums, "source"),
                       "sink": schema_dtype(_enums, "sink")})
            f = f.join(key_df, on=("p", "source", "sink"), how="anti")
        f = f.filter(_is_in_keep("p", keep_procs))
        return Param(p.dims, f, name=p.name)

    new.process_source_sink = _filter_arc_by_proc(src.process_source_sink)
    new.process_source_sink_eff = _filter_arc_by_proc(src.process_source_sink_eff)
    new.process_source_sink_noEff = _filter_arc_by_proc(src.process_source_sink_noEff)
    new.pss_dt = _filter_arc_by_proc(src.pss_dt)
    new.flow_to_n = _filter_arc_by_proc(src.flow_to_n)
    new.flow_from_n = _filter_arc_by_proc(src.flow_from_n)
    new.flow_from_nodeBalance_eff = _filter_arc_by_proc(src.flow_from_nodeBalance_eff)
    new.flow_from_nodeBalance_noEff = _filter_arc_by_proc(src.flow_from_nodeBalance_noEff)
    new.flow_from_commodity_eff = _filter_arc_by_proc(src.flow_from_commodity_eff)
    new.flow_from_commodity_noEff = _filter_arc_by_proc(src.flow_from_commodity_noEff)
    new.flow_to_commodity = _filter_arc_by_proc(src.flow_to_commodity)

    new.p_unitsize = _filter_param(src.p_unitsize, "p", keep_procs)
    new.p_flow_upper = _filter_param_arc(src.p_flow_upper)
    new.p_flow_upper_existing = _filter_param_arc(src.p_flow_upper_existing)
    new.p_slope = _filter_param(src.p_slope, "p", keep_procs)
    new.p_process_existing_count = _filter_param(src.p_process_existing_count, "p", keep_procs)
    new.p_process_availability = _filter_param(src.p_process_availability, "p", keep_procs)

    # ---- Profiles (filter by p) ----
    new.process_profile_upper = _filter_frame(src.process_profile_upper, "p", keep_procs)
    new.process_profile_lower = _filter_frame(src.process_profile_lower, "p", keep_procs)
    new.process_profile_fixed = _filter_frame(src.process_profile_fixed, "p", keep_procs)

    # ---- Storage / nodeState filtered to in-region nodes ----
    new.nodeState = _filter_frame(src.nodeState, "n", keep_nodes)
    new.nodeState_dt = _filter_frame(src.nodeState_dt, "n", keep_nodes)
    new.nodeState_first_dt = _filter_frame(src.nodeState_first_dt, "n", keep_nodes)
    new.storage_bind_within_timeset = _filter_frame(src.storage_bind_within_timeset, "n", keep_nodes)
    new.storage_bind_forward_only = _filter_frame(src.storage_bind_forward_only, "n", keep_nodes)
    new.storage_bind_within_solve = _filter_frame(src.storage_bind_within_solve, "n", keep_nodes)
    new.storage_fix_start = _filter_frame(src.storage_fix_start, "n", keep_nodes)
    new.nodeStateBlock = _filter_frame(src.nodeStateBlock, "n", keep_nodes)
    new.nodeState_last_dt = _filter_frame(src.nodeState_last_dt, "n", keep_nodes)
    new.p_state_upper = _filter_param(src.p_state_upper, "n", keep_nodes)
    new.p_state_unitsize = _filter_param(src.p_state_unitsize, "n", keep_nodes)
    new.p_state_self_discharge = _filter_param(src.p_state_self_discharge, "n", keep_nodes)
    new.p_state_start = _filter_param(src.p_state_start, "n", keep_nodes)
    new.p_state_existing_capacity = _filter_param(src.p_state_existing_capacity, "n", keep_nodes)
    new.p_node_availability = _filter_param(src.p_node_availability, "n", keep_nodes)
    new.p_roll_continue_state = _filter_param(src.p_roll_continue_state, "n", keep_nodes)
    new.p_node_capacity_for_scaling = _filter_param(src.p_node_capacity_for_scaling, "n", keep_nodes)

    # ---- Per-arc block weights (lh2 fixture) ----
    new.arc_sink_block_dt = _filter_arc_by_proc(src.arc_sink_block_dt) \
        if hasattr(src, "arc_sink_block_dt") else None
    new.arc_source_block_dt = _filter_arc_by_proc(src.arc_source_block_dt) \
        if hasattr(src, "arc_source_block_dt") else None
    new.p_arc_sink_weight = _filter_param_arc(getattr(src, "p_arc_sink_weight", None))
    new.p_arc_source_weight = _filter_param_arc(getattr(src, "p_arc_source_weight", None))

    # ---- Drop group_entity / group_node rows referencing dropped entities ----
    if src.group_entity is not None and "e" in src.group_entity.columns:
        keep_e = keep_nodes | keep_procs
        new.group_entity = src.group_entity.filter(
            _is_in_keep("e", keep_e)
        )
    if src.group_node is not None and "n" in src.group_node.columns:
        new.group_node = src.group_node.filter(
            _is_in_keep("n", keep_nodes)
        )
    new.process_unit = _filter_frame(src.process_unit, "p", keep_procs)

    # ---- Inject virtual half-flow arcs ----
    if half_flows:
        new = _inject_half_flows(new, src, half_flows)

    return new


def _inject_half_flows(
    rd: FlexData, src: FlexData, half_flows: list[HalfFlow],
) -> FlexData:
    """Add virtual half-flow connections + virtual arcs into the
    region's frames.  Each half-flow gets:

    * one row in ``process_source_sink`` (and the same partition
      family ``_eff`` or ``_noEff`` as the original arc);
    * a row per (d, t) in ``pss_dt`` covering the same dt frame as
      the source data;
    * a row in ``flow_to_n`` (export: in-region node is sink? no —
      original source is exported FROM the in-region node, so the
      half-flow has source=in_region_node, sink=virtual_node; the
      flow LEAVES the in-region node so it goes into ``flow_from_n``)
      — i.e. only ``flow_from_n`` for export, only ``flow_to_n`` for
      import;
    * unitsize = 1.0 (independent of the original);
    * flow_upper = original arc's flow_upper (so dispatch can max out
      the pipe even with λ=0; subgradient prices the actual flow).
    """
    if not half_flows:
        return rd

    # Capture the original arc rows so we can pull their (d, t) shape and
    # flow_upper Param values.
    orig_pss = src.process_source_sink
    orig_pss_dt = src.pss_dt
    orig_flow_upper = src.p_flow_upper
    orig_flow_upper_existing = src.p_flow_upper_existing
    orig_unitsize = src.p_unitsize
    orig_eff = src.process_source_sink_eff
    orig_noEff = src.process_source_sink_noEff

    # Build new pss / pss_dt / flow_*/upper rows for each half-flow.
    new_pss_rows: list[dict] = []
    new_pss_eff_rows: list[dict] = []
    new_pss_noEff_rows: list[dict] = []
    new_pss_dt_rows: list[dict] = []
    new_flow_to_n_rows: list[dict] = []
    new_flow_from_n_rows: list[dict] = []
    new_flow_from_nb_eff_rows: list[dict] = []
    new_flow_from_nb_noEff_rows: list[dict] = []
    new_unitsize_rows: list[dict] = []
    new_flow_upper_rows: list[dict] = []
    new_flow_upper_existing_rows: list[dict] = []
    new_arc_sink_block_dt_rows: list[dict] = []
    new_arc_sink_block_dt_full = src.arc_sink_block_dt is not None
    new_p_arc_sink_weight_rows: list[dict] = []

    # Classification: inherit from the original arc.  When the original
    # arc is in process_source_sink_eff (with slope p_slope), so is the
    # half-flow — that ensures the source-side nodeBalance loses the
    # right amount of source commodity (source-side flow ×
    # unitsize × slope) which matches the monolithic.  When the original
    # is noEff, the half-flow stays noEff and the source-side loses
    # source-side flow × unitsize (no slope).  Pre-compute per-(p,
    # source, sink) classification.
    orig_eff_keys: set[tuple[str, str, str]] = set()
    if orig_eff is not None:
        for r in orig_eff.iter_rows(named=True):
            orig_eff_keys.add((r["p"], r["source"], r["sink"]))

    new_p_slope_rows: list[dict] = []

    for hf in half_flows:
        # The dt grid for the new arc inherits the source arc's grid.
        # Pull it from the original pss_dt rows for (p_orig, source_orig,
        # sink_orig).
        if orig_pss_dt is not None:
            arc_dt = orig_pss_dt.filter(
                (pl.col("p") == hf.original_p)
                & (pl.col("source") == hf.original_source)
                & (pl.col("sink") == hf.original_sink)
            ).select("d", "t")
        else:
            _enums = getattr(src, "_axis_enums", None)
            arc_dt = pl.DataFrame({"d": [], "t": []},
                                  schema={"d": schema_dtype(_enums, "d"),
                                          "t": schema_dtype(_enums, "t")})

        # Classification inherits from the original arc.
        is_eff = (hf.original_p, hf.original_source, hf.original_sink) in orig_eff_keys

        # process_source_sink rows
        new_pss_rows.append({
            "p": hf.virtual_p,
            "source": hf.virtual_arc_source,
            "sink": hf.virtual_arc_sink,
        })
        if is_eff:
            new_pss_eff_rows.append({
                "p": hf.virtual_p,
                "source": hf.virtual_arc_source,
                "sink": hf.virtual_arc_sink,
            })
            # Inherit p_slope rows from the original (p, d, t).
            if src.p_slope is not None:
                slope_rows = src.p_slope.frame.filter(
                    pl.col("p") == hf.original_p
                ).select("p", "d", "t", "value")
                for r in slope_rows.iter_rows(named=True):
                    new_p_slope_rows.append({
                        "p": hf.virtual_p,
                        "d": r["d"], "t": r["t"],
                        "value": float(r["value"]),
                    })
        else:
            new_pss_noEff_rows.append({
                "p": hf.virtual_p,
                "source": hf.virtual_arc_source,
                "sink": hf.virtual_arc_sink,
            })

        # pss_dt rows
        for r in arc_dt.iter_rows(named=True):
            new_pss_dt_rows.append({
                "p": hf.virtual_p,
                "source": hf.virtual_arc_source,
                "sink": hf.virtual_arc_sink,
                "d": r["d"], "t": r["t"],
            })

        # Flow direction wiring:
        #  * EXPORT: half-flow goes from in_region_node to virtual_node.
        #    Source-side flow leaves in_region_node ⇒ flow_from_n /
        #    flow_from_nodeBalance entry on the in-region node side.
        #    No flow_to_n entry (virtual_node is not in nodeBalance).
        #  * IMPORT: half-flow goes from virtual_node to in_region_node.
        #    Sink-side flow enters in_region_node ⇒ flow_to_n entry on
        #    the in-region node side.  No flow_from_n on the virtual side.
        if hf.side == "export":
            new_flow_from_n_rows.append({
                "p": hf.virtual_p,
                "source": hf.virtual_arc_source,
                "sink": hf.virtual_arc_sink,
                "n": hf.in_region_node,
            })
            # process_source_sink_eff/noEff is the partition used by
            # flow_from_nodeBalance_eff/noEff.  Match the original.
            if is_eff:
                new_flow_from_nb_eff_rows.append({
                    "p": hf.virtual_p,
                    "source": hf.virtual_arc_source,
                    "sink": hf.virtual_arc_sink,
                    "n": hf.in_region_node,
                })
            else:
                new_flow_from_nb_noEff_rows.append({
                    "p": hf.virtual_p,
                    "source": hf.virtual_arc_source,
                    "sink": hf.virtual_arc_sink,
                    "n": hf.in_region_node,
                })
        else:  # import
            new_flow_to_n_rows.append({
                "p": hf.virtual_p,
                "source": hf.virtual_arc_source,
                "sink": hf.virtual_arc_sink,
                "n": hf.in_region_node,
            })

        # unitsize = original (so v_flow numbers match between paired
        # half-flows and original physical flow capacities are
        # preserved).
        if orig_unitsize is not None:
            us_match = orig_unitsize.frame.filter(
                pl.col("p") == hf.original_p)
            us_val = (float(us_match["value"][0]) if us_match.height > 0
                      else 1.0)
        else:
            us_val = 1.0
        new_unitsize_rows.append({"p": hf.virtual_p, "value": us_val})

        # flow_upper inherits from the original arc (per (d, t)).
        if orig_flow_upper is not None:
            cap_rows = orig_flow_upper.frame.filter(
                (pl.col("p") == hf.original_p)
                & (pl.col("source") == hf.original_source)
                & (pl.col("sink") == hf.original_sink)
            ).select("d", "t", "value")
            for r in cap_rows.iter_rows(named=True):
                new_flow_upper_rows.append({
                    "p": hf.virtual_p,
                    "source": hf.virtual_arc_source,
                    "sink": hf.virtual_arc_sink,
                    "d": r["d"], "t": r["t"],
                    "value": float(r["value"]),
                })
        if orig_flow_upper_existing is not None:
            cap_rows = orig_flow_upper_existing.frame.filter(
                (pl.col("p") == hf.original_p)
                & (pl.col("source") == hf.original_source)
                & (pl.col("sink") == hf.original_sink)
            ).select("d", "value")
            for r in cap_rows.iter_rows(named=True):
                new_flow_upper_existing_rows.append({
                    "p": hf.virtual_p,
                    "source": hf.virtual_arc_source,
                    "sink": hf.virtual_arc_sink,
                    "d": r["d"],
                    "value": float(r["value"]),
                })

        # ── arc-block weights (lh2 fixture only) ──
        # For each half-flow, mirror the original arc's weights so the
        # block-aware nodeBalance aggregation includes the half-flow at
        # the right (d, t) granularity.
        if hf.side == "export":
            # Source-side: original arc_source_block_dt row(s).
            if src.arc_source_block_dt is not None:
                src_rows = src.arc_source_block_dt.filter(
                    (pl.col("p") == hf.original_p)
                    & (pl.col("source") == hf.original_source)
                    & (pl.col("sink") == hf.original_sink)
                ).select("d", "b_first", "t", "weight")
                for r in src_rows.iter_rows(named=True):
                    # We'll augment arc_source_block_dt later; collect.
                    new_arc_sink_block_dt_rows.append({
                        "p": hf.virtual_p,
                        "source": hf.virtual_arc_source,
                        "sink": hf.virtual_arc_sink,
                        "d": r["d"], "b_first": r["b_first"],
                        "t": r["t"], "weight": r["weight"],
                        "_side": "source",
                    })
            if src.p_arc_source_weight is not None:
                w_rows = src.p_arc_source_weight.frame.filter(
                    (pl.col("p") == hf.original_p)
                    & (pl.col("source") == hf.original_source)
                    & (pl.col("sink") == hf.original_sink)
                ).select("d", "t", "value")
                for r in w_rows.iter_rows(named=True):
                    new_p_arc_sink_weight_rows.append({
                        "p": hf.virtual_p,
                        "source": hf.virtual_arc_source,
                        "sink": hf.virtual_arc_sink,
                        "d": r["d"], "t": r["t"],
                        "value": float(r["value"]),
                        "_side": "source",
                    })
        else:
            if src.arc_sink_block_dt is not None:
                snk_rows = src.arc_sink_block_dt.filter(
                    (pl.col("p") == hf.original_p)
                    & (pl.col("source") == hf.original_source)
                    & (pl.col("sink") == hf.original_sink)
                ).select("d", "b_first", "t", "weight")
                for r in snk_rows.iter_rows(named=True):
                    new_arc_sink_block_dt_rows.append({
                        "p": hf.virtual_p,
                        "source": hf.virtual_arc_source,
                        "sink": hf.virtual_arc_sink,
                        "d": r["d"], "b_first": r["b_first"],
                        "t": r["t"], "weight": r["weight"],
                        "_side": "sink",
                    })
            if src.p_arc_sink_weight is not None:
                w_rows = src.p_arc_sink_weight.frame.filter(
                    (pl.col("p") == hf.original_p)
                    & (pl.col("source") == hf.original_source)
                    & (pl.col("sink") == hf.original_sink)
                ).select("d", "t", "value")
                for r in w_rows.iter_rows(named=True):
                    new_p_arc_sink_weight_rows.append({
                        "p": hf.virtual_p,
                        "source": hf.virtual_arc_source,
                        "sink": hf.virtual_arc_sink,
                        "d": r["d"], "t": r["t"],
                        "value": float(r["value"]),
                        "_side": "sink",
                    })

    # ---- Concatenate into rd ----
    def _concat(orig: pl.DataFrame | None,
                rows: list[dict],
                schema: dict) -> pl.DataFrame | None:
        if not rows:
            return orig
        new_df = pl.DataFrame(rows, schema=schema)
        if orig is None or orig.height == 0:
            # Need a frame matching the original schema; fall back to new.
            return new_df.select(list(schema.keys()))
        # Coerce types to match orig.
        return pl.concat([orig.select(list(schema.keys())),
                          new_df.select(list(schema.keys()))],
                         how="vertical")

    _enums_loc = getattr(src, "_axis_enums", None)
    _pss_schema = {"p": schema_dtype(_enums_loc, "p"),
                    "source": schema_dtype(_enums_loc, "source"),
                    "sink": schema_dtype(_enums_loc, "sink")}
    _pssn_schema = {**_pss_schema, "n": schema_dtype(_enums_loc, "n")}
    rd.process_source_sink = _concat(
        rd.process_source_sink, new_pss_rows, _pss_schema)
    if new_pss_eff_rows:
        rd.process_source_sink_eff = _concat(
            rd.process_source_sink_eff, new_pss_eff_rows, _pss_schema)
    if new_pss_noEff_rows:
        rd.process_source_sink_noEff = _concat(
            rd.process_source_sink_noEff, new_pss_noEff_rows, _pss_schema)
    rd.pss_dt = _concat(
        rd.pss_dt, new_pss_dt_rows,
        {**_pss_schema,
         "d": schema_dtype(_enums_loc, "d"),
         "t": schema_dtype(_enums_loc, "t")})
    if new_flow_to_n_rows:
        rd.flow_to_n = _concat(
            rd.flow_to_n, new_flow_to_n_rows, _pssn_schema)
    if new_flow_from_n_rows:
        rd.flow_from_n = _concat(
            rd.flow_from_n, new_flow_from_n_rows, _pssn_schema)
    if new_flow_from_nb_eff_rows:
        rd.flow_from_nodeBalance_eff = _concat(
            rd.flow_from_nodeBalance_eff, new_flow_from_nb_eff_rows, _pssn_schema)
    if new_flow_from_nb_noEff_rows:
        rd.flow_from_nodeBalance_noEff = _concat(
            rd.flow_from_nodeBalance_noEff, new_flow_from_nb_noEff_rows, _pssn_schema)

    # Append unitsize Param.
    if rd.p_unitsize is not None and new_unitsize_rows:
        _enums = getattr(src, "_axis_enums", None)
        new_us = pl.DataFrame(new_unitsize_rows,
                              schema={"p": schema_dtype(_enums, "p"),
                                      "value": pl.Float64})
        merged_us = pl.concat([rd.p_unitsize.frame.select("p", "value"),
                               new_us], how="vertical")
        rd.p_unitsize = Param(("p",), merged_us, name=rd.p_unitsize.name)

    # Append p_slope rows for half-flows that are eff-classified.
    if rd.p_slope is not None and new_p_slope_rows:
        _enums = getattr(src, "_axis_enums", None)
        new_sl = pl.DataFrame(new_p_slope_rows,
                              schema={"p": schema_dtype(_enums, "p"),
                                      "d": schema_dtype(_enums, "d"),
                                      "t": schema_dtype(_enums, "t"),
                                      "value": pl.Float64})
        merged_sl = pl.concat([rd.p_slope.frame.select("p", "d", "t", "value"),
                               new_sl], how="vertical")
        rd.p_slope = Param(("p", "d", "t"), merged_sl, name=rd.p_slope.name)

    # Append flow_upper Param rows.
    if rd.p_flow_upper is not None and new_flow_upper_rows:
        _enums = getattr(src, "_axis_enums", None)
        new_fu = pl.DataFrame(new_flow_upper_rows,
                              schema={"p": schema_dtype(_enums, "p"),
                                      "source": schema_dtype(_enums, "source"),
                                      "sink": schema_dtype(_enums, "sink"),
                                      "d": schema_dtype(_enums, "d"),
                                      "t": schema_dtype(_enums, "t"),
                                      "value": pl.Float64})
        merged_fu = pl.concat([rd.p_flow_upper.frame.select(
                                  "p", "source", "sink", "d", "t", "value"),
                               new_fu], how="vertical")
        rd.p_flow_upper = Param(("p", "source", "sink", "d", "t"),
                                merged_fu, name=rd.p_flow_upper.name)
    if rd.p_flow_upper_existing is not None and new_flow_upper_existing_rows:
        _enums = getattr(src, "_axis_enums", None)
        new_fue = pl.DataFrame(new_flow_upper_existing_rows,
                               schema={"p": schema_dtype(_enums, "p"),
                                       "source": schema_dtype(_enums, "source"),
                                       "sink": schema_dtype(_enums, "sink"),
                                       "d": schema_dtype(_enums, "d"),
                                       "value": pl.Float64})
        merged_fue = pl.concat([rd.p_flow_upper_existing.frame.select(
                                    "p", "source", "sink", "d", "value"),
                                new_fue], how="vertical")
        rd.p_flow_upper_existing = Param(
            ("p", "source", "sink", "d"),
            merged_fue, name=rd.p_flow_upper_existing.name)

    # ── p_process_availability and p_process_existing_count ──
    # The maxToSink RHS is multiplied by p_process_availability when
    # populated, and Param×Param is an inner-join so missing half-flow
    # entries collapse to zero RHS.  We must add availability=1.0 and
    # existing_count=1.0 entries so the half-flow's bound stays at the
    # value we set in p_flow_upper_existing.
    if rd.p_process_availability is not None and rd.pss_dt is not None:
        # Add a (p, d, t) row for each (virtual_p, d, t) in pss_dt.
        avail_rows = (rd.pss_dt
                      .filter(cast_dim(pl.col("p"), None, "p").is_in([hf.virtual_p for hf in half_flows]))
                      .select("p", "d", "t")
                      .with_columns(value=pl.lit(1.0)))
        if avail_rows.height > 0:
            merged = pl.concat([rd.p_process_availability.frame.select(
                                    "p", "d", "t", "value"),
                                avail_rows], how="vertical")
            rd.p_process_availability = Param(
                ("p", "d", "t"), merged,
                name=rd.p_process_availability.name)
    if rd.p_process_existing_count is not None and rd.pss_dt is not None:
        # (p, d) row for each virtual half-flow
        ec_rows = (rd.pss_dt
                   .filter(cast_dim(pl.col("p"), None, "p").is_in([hf.virtual_p for hf in half_flows]))
                   .select("p", "d").unique()
                   .with_columns(value=pl.lit(1.0)))
        if ec_rows.height > 0:
            merged = pl.concat([rd.p_process_existing_count.frame.select(
                                    "p", "d", "value"),
                                ec_rows], how="vertical")
            rd.p_process_existing_count = Param(
                ("p", "d"), merged,
                name=rd.p_process_existing_count.name)

    # Append arc-block-weight rows (lh2 fixture).  Half-flows on the
    # source side go to arc_source_block_dt + p_arc_source_weight; on
    # sink side they go to arc_sink_block_dt + p_arc_sink_weight.
    src_block_rows = [r for r in new_arc_sink_block_dt_rows if r["_side"] == "source"]
    snk_block_rows = [r for r in new_arc_sink_block_dt_rows if r["_side"] == "sink"]
    src_w_rows = [r for r in new_p_arc_sink_weight_rows if r["_side"] == "source"]
    snk_w_rows = [r for r in new_p_arc_sink_weight_rows if r["_side"] == "sink"]

    _enums = getattr(src, "_axis_enums", None)
    if rd.arc_source_block_dt is not None and src_block_rows:
        new_df = pl.DataFrame(
            [{k: r[k] for k in ("p", "source", "sink", "d", "b_first", "t", "weight")}
             for r in src_block_rows],
            schema={"p": schema_dtype(_enums, "p"),
                    "source": schema_dtype(_enums, "source"),
                    "sink": schema_dtype(_enums, "sink"),
                    "d": schema_dtype(_enums, "d"),
                    "b_first": schema_dtype(_enums, "b_first"),
                    "t": schema_dtype(_enums, "t"),
                    "weight": pl.Float64})
        rd.arc_source_block_dt = pl.concat([
            rd.arc_source_block_dt.select(*new_df.columns), new_df],
            how="vertical")
    if rd.arc_sink_block_dt is not None and snk_block_rows:
        new_df = pl.DataFrame(
            [{k: r[k] for k in ("p", "source", "sink", "d", "b_first", "t", "weight")}
             for r in snk_block_rows],
            schema={"p": schema_dtype(_enums, "p"),
                    "source": schema_dtype(_enums, "source"),
                    "sink": schema_dtype(_enums, "sink"),
                    "d": schema_dtype(_enums, "d"),
                    "b_first": schema_dtype(_enums, "b_first"),
                    "t": schema_dtype(_enums, "t"),
                    "weight": pl.Float64})
        rd.arc_sink_block_dt = pl.concat([
            rd.arc_sink_block_dt.select(*new_df.columns), new_df],
            how="vertical")
    if rd.p_arc_source_weight is not None and src_w_rows:
        new_df = pl.DataFrame(
            [{k: r[k] for k in ("p", "source", "sink", "d", "t", "value")}
             for r in src_w_rows],
            schema={"p": schema_dtype(_enums, "p"),
                    "source": schema_dtype(_enums, "source"),
                    "sink": schema_dtype(_enums, "sink"),
                    "d": schema_dtype(_enums, "d"),
                    "t": schema_dtype(_enums, "t"),
                    "value": pl.Float64})
        rd.p_arc_source_weight = Param(
            ("p", "source", "sink", "d", "t"),
            pl.concat([rd.p_arc_source_weight.frame.select(*new_df.columns),
                       new_df], how="vertical"),
            name=rd.p_arc_source_weight.name)
    if rd.p_arc_sink_weight is not None and snk_w_rows:
        new_df = pl.DataFrame(
            [{k: r[k] for k in ("p", "source", "sink", "d", "t", "value")}
             for r in snk_w_rows],
            schema={"p": schema_dtype(_enums, "p"),
                    "source": schema_dtype(_enums, "source"),
                    "sink": schema_dtype(_enums, "sink"),
                    "d": schema_dtype(_enums, "d"),
                    "t": schema_dtype(_enums, "t"),
                    "value": pl.Float64})
        rd.p_arc_sink_weight = Param(
            ("p", "source", "sink", "d", "t"),
            pl.concat([rd.p_arc_sink_weight.frame.select(*new_df.columns),
                       new_df], how="vertical"),
            name=rd.p_arc_sink_weight.name)

    # group_entity / group_node — augment with virtual entities under the
    # half-flow's region (so downstream group-aware emitters don't
    # spuriously skip them).  Skip — the group_* sets only matter for
    # group_slack / capacity_margin features which aren't in lh2 fixture.

    return rd


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def split(
    data: FlexData,
    *,
    regions: list[str] | None = None,
    region_membership: dict[str, dict[str, set[str]]] | None = None,
) -> list[RegionSplit]:
    """Slice a whole-system :class:`FlexData` into per-region splits.

    Parameters
    ----------
    data
        Whole-system :class:`FlexData` (output of :func:`load_flextool`).
    regions
        Explicit region list.  When ``None``, derives from
        ``data.group_entity`` / ``data.group_node`` (every group with at
        least one entity is treated as a region).  Callers that have
        access to ``decomposition_method`` from disk should pass an
        explicit list.
    region_membership
        Pre-computed ``{region: {"nodes": ..., "processes": ...}}`` from
        :func:`load_region_membership`.  When omitted we re-derive from
        ``data``.

    Returns
    -------
    list[RegionSplit]
        One per region, in the order given by ``regions``.

    Notes
    -----
    "Shared" entities (in no region) are kept in every region's local
    set — they're typically things like ``coal_market`` (a commodity
    node).  Cross-region arcs are dropped from the *original* process
    frames and replaced by virtual half-flow arcs (one in the source
    region, one in the sink region).
    """
    if regions is None:
        # Derive: every group with at least one membership entity.
        if data.group_node is not None and data.group_node.height > 0:
            regions = sorted(
                set(data.group_node["g"].unique().to_list())
            )
        else:
            regions = []
    if not regions:
        return []

    if region_membership is None:
        region_membership = load_region_membership(data, regions)

    region_nodes = {r: m["nodes"] for r, m in region_membership.items()}
    region_procs = {r: m["processes"] for r, m in region_membership.items()}

    # Identify shared entities (no region).
    all_region_nodes: set[str] = set()
    for ns in region_nodes.values():
        all_region_nodes |= ns
    all_region_procs: set[str] = set()
    for ps in region_procs.values():
        all_region_procs |= ps

    # Whole-set of nodes and processes.
    if data.nodeBalance is not None:
        all_nodes = set(data.nodeBalance["n"].to_list())
    else:
        all_nodes = set()
    if data.process_source_sink is not None:
        all_procs = set(data.process_source_sink["p"].unique().to_list())
    else:
        all_procs = set()

    shared_nodes = all_nodes - all_region_nodes
    shared_procs = all_procs - all_region_procs

    # Classify cross-region arcs.
    if data.process_source_sink is None:
        return [
            RegionSplit(region=r, data=data, half_flows=[])
            for r in regions
        ]

    pss_tagged, cross = _classify_arcs(
        data.process_source_sink, region_nodes,
    )

    half_flows_by_region = _make_half_flows(cross)

    cross_arcs_by_pss: set[tuple[str, str, str]] = set()
    for r in cross.iter_rows(named=True):
        cross_arcs_by_pss.add((r["p"], r["source"], r["sink"]))

    # Phase 4 — virtual half-flow entities ("hf_pipe_*" / "pipe_*__*__*")
    # are created at runtime by ``_make_half_flows``; they are not in the
    # source DB and therefore not in the axis_enums vocabulary built by
    # ``build_axis_enums``.  Downstream filter operations like
    # ``pl.col("p").is_in([...keep_procs incl. virtual_p...])`` raise
    # ``conversion from str to enum failed`` when polars casts the
    # comparison list against the Enum dtype.  Widen the live vocabulary
    # to include the virtual tokens for the duration of the split.
    _virt_p: set[str] = set()
    _virt_n: set[str] = set()
    for _hfs in half_flows_by_region.values():
        for _hf in _hfs:
            _virt_p.add(_hf.virtual_p)
            _virt_n.add(_hf.virtual_node)
    _enums_token = None
    _base_enums = get_global_axis_enums()
    if _base_enums is not None and (_virt_p or _virt_n):
        _ext: dict[str, pl.Enum] = dict(_base_enums)
        _virt_e = _virt_p | _virt_n
        for _axis_name, _new_toks in (
            ("p", _virt_p),
            ("n", _virt_n),
            ("source", _virt_n),
            ("sink", _virt_n),
            ("e", _virt_e),
        ):
            _existing = _ext.get(_axis_name)
            if _existing is None:
                continue
            _existing_cats = list(_existing.categories)
            _existing_set = set(_existing_cats)
            _add = [t for t in _new_toks if t not in _existing_set]
            if _add:
                _ext[_axis_name] = pl.Enum(_existing_cats + _add)
        _enums_token = set_global_axis_enums(_ext)

    try:
        splits: list[RegionSplit] = []
        for r in regions:
            keep_nodes = region_nodes.get(r, set()) | shared_nodes
            keep_procs = region_procs.get(r, set()) | shared_procs
            # Also keep cross-region pipes' original `p` membership in this
            # region IF the in-region terminal is here.  We'll drop the
            # specific (p, source, sink) cross-arc rows below; but we keep
            # the process p in keep_procs so the OTHER direction (back-flow)
            # which has the in-region node as its sink/source is retained.
            # In fact, we add the original cross-region pipe p iff this
            # region has a half-flow involving that p.
            for hf in half_flows_by_region.get(r, []):
                keep_procs.add(hf.original_p)
                keep_procs.add(hf.virtual_p)

            rdata = _build_region_data(
                src=data,
                region=r,
                keep_nodes=keep_nodes,
                keep_procs=keep_procs,
                half_flows=half_flows_by_region.get(r, []),
                cross_arcs_by_pss=cross_arcs_by_pss,
            )
            splits.append(RegionSplit(
                region=r,
                data=rdata,
                half_flows=half_flows_by_region.get(r, []),
            ))
        return splits
    finally:
        if _enums_token is not None:
            reset_global_axis_enums(_enums_token)
