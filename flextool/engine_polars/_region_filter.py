"""Regional filter for Benders decomposition (gaps A4 + A5).

Slices a whole-system :class:`FlexData` into N per-region :class:`FlexData`s.
Cross-region arcs (rows in ``process_source_sink`` whose ``source`` and
``sink`` straddle two regions) are *severed* into two virtual one-way
half-flow arcs, one in each region.

The virtual entities are pure bookkeeping — the half-flow on the export
side and the half-flow on the import side carry the **same flow** at
optimality.  In the Benders scheme that coupling is enforced by the
coordinating master (:mod:`flextool.engine_polars._benders`): the master
holds the inter-regional trade flows and pins each region's forward
cross-region half-flow to its chosen value before solving the region as a
subproblem.  Inside a region's standalone LP, the half-flow column is just
an ordinary ``v_flow`` column with:

* ``p_unitsize = 1`` (same units as the original column);
* ``p_flow_upper`` = original pipe capacity (so dispatch can push flow
  at full pipe capacity; uncapped when the master may pin a positive
  greenfield trade);
* the in-region terminal node enters ``flow_to_n`` (export) or
  ``flow_from_n`` (import) so the half-flow contributes to its
  nodeBalance;
* the *virtual* node sits OUTSIDE ``nodeBalance`` so the half-flow's
  other terminal is free (no balance pin, no penalty).

This module is a no-op when the input :class:`FlexData` has no
``decomposition_method=benders_regional`` group params.

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
from flextool.engine_polars._param_shapes import promote_param_to_dt
from flextool.engine_polars._pdt_join import compute_pss_dt


__all__ = [
    "HalfFlow",
    "RegionSplit",
    "compute_master_hosted_nodes",
    "discover_regions",
    "split",
    "master_network_data",
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
    original_sink)`` triple — the :mod:`flextool.engine_polars._benders`
    coordinator pairs export and import on that key.
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
    ``decomposition_method=benders_regional``."""
    methods = load_decomposition_method(work_dir, provider=provider)
    return sorted(g for g, m in methods.items() if m == "benders_regional")


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


def compute_master_hosted_nodes(
    data: FlexData,
    region_membership: dict[str, dict[str, set[str]]],
) -> set[str]:
    """Return the master-hosted node set: every node carrying a balance
    or state row (``data.nodeBalance`` ∪ ``data.nodeState``) that is in
    NO region's membership.

    Nodes with no balance/state row (pure commodity/market nodes) are
    deliberately NOT included — they keep today's shared-replicate
    semantics (replication is safe for them: there is no balance row to
    duplicate), pinned by the existing region-filter tests.
    """
    balance_state: set[str] = set()
    if data.nodeBalance is not None and data.nodeBalance.height > 0:
        balance_state |= set(data.nodeBalance["n"].to_list())
    if data.nodeState is not None and data.nodeState.height > 0:
        balance_state |= set(data.nodeState["n"].to_list())
    region_all: set[str] = set()
    for m in region_membership.values():
        region_all |= m["nodes"]
    return balance_state - region_all


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
    master_nodes: "frozenset[str] | set[str]" = frozenset(),
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Tag each (p, source, sink) row with its source-region and
    sink-region, then classify into four classes:

    * **local** — everything not in the three classes below (stays in
      its region's frames);
    * **cross-region** — both endpoints in (different) regions; severed
      into an export + import half-flow pair (returned as ``cross``);
    * **region↔master** — one endpoint in a region, the other a
      master-hosted node; severed into exactly ONE half-flow on the
      region side (returned as ``region_master``);
    * **master-local** — at least one master-hosted endpoint and NO
      region endpoint (the other side is master-hosted, shared, or a
      non-node token); dropped from every region and NOT half-flowed —
      the master keeps the whole arc (returned as ``master_local``).

    Nodes not in any region (and not master-hosted) are "shared" — an
    arc with a shared endpoint is treated as local-to-the-other-region
    (it stays in that region's frames and is not a coupling).

    With the default empty ``master_nodes`` the ``region_master`` /
    ``master_local`` frames are empty and ``pss_tagged`` / ``cross``
    are byte-identical to the historical 2-way behaviour.
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
    if not master_nodes:
        empty = pss_tagged.head(0)
        return pss_tagged, cross, empty, empty
    # Vocab-independent membership test (mirror the map_elements-on-raw-
    # strings robustness of the region tagging above: no Enum cast, so a
    # stale global axis-enum vocabulary cannot null the master tokens).
    src_m = pss["source"].cast(pl.Utf8).is_in(list(master_nodes))
    snk_m = pss["sink"].cast(pl.Utf8).is_in(list(master_nodes))
    pss_tagged = pss_tagged.with_columns(_src_master=src_m, _snk_master=snk_m)
    region_master = pss_tagged.filter(
        (pl.col("_src_region").is_not_null() & pl.col("_snk_master"))
        | (pl.col("_src_master") & pl.col("_snk_region").is_not_null())
    )
    master_local = pss_tagged.filter(
        (pl.col("_src_master") | pl.col("_snk_master"))
        & pl.col("_src_region").is_null()
        & pl.col("_snk_region").is_null()
    )
    return pss_tagged, cross, region_master, master_local


def _make_half_flows(
    cross_arcs: pl.DataFrame,
    region_master_arcs: pl.DataFrame | None = None,
) -> dict[str, list[HalfFlow]]:
    """For each cross-region arc, produce two HalfFlow records (one
    per region).

    For each region↔master arc in *region_master_arcs* (master-hosted
    mode), produce exactly ONE HalfFlow — on the region side: an
    export when the region node is the arc's source, an import when it
    is the sink.  Naming reuses the region-side stem of the paired
    convention (the master side is implicit; no master-side virtual
    entity exists — the master keeps the whole original arc).
    """
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
    if region_master_arcs is not None and region_master_arcs.height > 0:
        for r in region_master_arcs.iter_rows(named=True):
            p = r["p"]
            s = r["source"]
            k = r["sink"]
            ra = r["_src_region"]
            rb = r["_snk_region"]
            if ra is not None:
                # Region node is the SOURCE ⇒ export half-flow in ra.
                ve_node = f"{p}__{s}__{k}__export__{ra}"
                ve_conn = f"hf_{p}__{s}__{k}__export__{ra}"
                out.setdefault(ra, []).append(HalfFlow(
                    region=ra, side="export",
                    original_p=p, original_source=s, original_sink=k,
                    in_region_node=s,
                    virtual_node=ve_node,
                    virtual_p=ve_conn,
                    virtual_arc_source=s,
                    virtual_arc_sink=ve_node,
                ))
            else:
                # Region node is the SINK ⇒ import half-flow in rb.
                vi_node = f"{p}__{s}__{k}__import__{rb}"
                vi_conn = f"hf_{p}__{s}__{k}__import__{rb}"
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
# Master-hosted-node validation + region scrubbing
# ---------------------------------------------------------------------------


def _master_local_procs(
    pss: pl.DataFrame, master_local: pl.DataFrame,
) -> set[str]:
    """Processes whose EVERY ``process_source_sink`` arc is master-local
    (critique F3).  Those procs live wholly in the master: regions must
    carry neither their arcs NOR their entity rows (invest sets, annuity
    params, cost rows)."""
    if master_local.height == 0:
        return set()
    ml_triples: set[tuple[str, str, str]] = {
        (r["p"], r["source"], r["sink"])
        for r in master_local.iter_rows(named=True)
    }
    ml_procs = {t[0] for t in ml_triples}
    triples_by_p: dict[str, set[tuple[str, str, str]]] = {}
    for r in pss.iter_rows(named=True):
        if r["p"] in ml_procs:
            triples_by_p.setdefault(r["p"], set()).add(
                (r["p"], r["source"], r["sink"]))
    return {p for p, ts in triples_by_p.items() if ts <= ml_triples}


def _validate_no_straddling_units(
    data: FlexData,
    all_region_nodes: set[str],
    master_nodes: "frozenset[str] | set[str]",
) -> None:
    """Hard-error on any UNIT touching both a region node and a
    master-hosted node (design decision D-a).

    Aggregated PER PROCESS across ALL its ``process_source_sink`` rows —
    NOT per arc: a unit with one master-local arc plus one purely
    in-region arc straddles *as an entity* while having no individually
    straddling arc, and severing any unit arc silently loses conversion
    terms.  Never a silent degrade (the fix_start precedent is the
    anti-pattern this guards against).
    """
    if data.process_source_sink is None:
        return
    units: set[str] = set()
    if data.process_unit is not None and data.process_unit.height > 0:
        units |= set(data.process_unit["p"].to_list())
    if (getattr(data, "process_indirect", None) is not None
            and data.process_indirect.height > 0):
        units |= set(data.process_indirect["p"].to_list())
    if not units:
        return
    endpoints_by_unit: dict[str, set[str]] = {}
    for r in data.process_source_sink.iter_rows(named=True):
        if r["p"] in units:
            endpoints_by_unit.setdefault(r["p"], set()).update(
                (r["source"], r["sink"]))
    for unit in sorted(endpoints_by_unit):
        eps = endpoints_by_unit[unit]
        region_touch = eps & all_region_nodes
        master_touch = eps & master_nodes
        if region_touch and master_touch:
            raise RuntimeError(
                f"split: unit {unit!r} straddles the region/master "
                f"boundary — across its arcs it touches region node(s) "
                f"{sorted(region_touch)} AND master-hosted node(s) "
                f"{sorted(master_touch)}.  A unit cannot be severed "
                f"between a region subproblem and the Benders master.  "
                f"Insert a handover CONNECTION between the region-side "
                f"node and the master-hosted node (the handover-"
                f"connection pattern) so every boundary arc is a "
                f"connection, and keep the unit's arcs wholly on one "
                f"side."
            )


def _user_constraint_sides(
    data: FlexData,
    all_region_nodes: set[str],
    master_nodes: "frozenset[str] | set[str]",
    master_local_procs: set[str],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Aggregate, per user-constraint id (``cn``), the decomposition
    sides ("master" / "region") of every referenced entity across every
    user-constraint frame.

    Shared by the mixed-constraint hard error
    (:func:`_validate_user_constraints`) and the master/region row
    partition (:func:`_master_side_constraint_ids`).  Entities on
    neither side (shared nodes, non-node tokens) are neutral.

    Returns ``(sides_by_cn, refs_by_cn)``; ``refs_by_cn`` carries
    human-readable ``"entity (side)"`` strings for error messages.
    """
    sides_by_cn: dict[str, set[str]] = {}
    refs_by_cn: dict[str, set[str]] = {}

    def _node_side(n: str) -> str | None:
        if n in master_nodes:
            return "master"
        if n in all_region_nodes:
            return "region"
        return None

    def _add(cn: str, entity: str, side: str | None) -> None:
        if side is None:
            return
        sides_by_cn.setdefault(cn, set()).add(side)
        refs_by_cn.setdefault(cn, set()).add(f"{entity} ({side})")

    def _frame_of(obj) -> pl.DataFrame | None:
        if obj is None:
            return None
        f = obj.frame if isinstance(obj, Param) else obj
        return f if f.height > 0 else None

    # Arc-keyed references: (p, source, sink, cn).  Each row contributes
    # the sides of BOTH terminal nodes (a row on a region↔master
    # coupling arc is itself mixed) plus the master-local proc side.
    for fld in ("flow_constraint_idx", "p_flow_constraint_coef"):
        f = _frame_of(getattr(data, fld, None))
        if f is None or "cn" not in f.columns:
            continue
        for r in f.iter_rows(named=True):
            cn = r["cn"]
            _add(cn, r["source"], _node_side(r["source"]))
            _add(cn, r["sink"], _node_side(r["sink"]))
            if r["p"] in master_local_procs:
                _add(cn, r["p"], "master")
    # Node-keyed references: (n, cn).
    for fld in ("p_node_constraint_state_coeff",
                "p_node_constraint_invested_capacity_coeff",
                "p_node_constraint_prebuilt_capacity_coeff"):
        f = _frame_of(getattr(data, fld, None))
        if f is None or "cn" not in f.columns:
            continue
        for r in f.iter_rows(named=True):
            _add(r["cn"], r["n"], _node_side(r["n"]))
    # Process-keyed references: (p, cn).  A process's side is the union
    # of its arc-endpoint node sides (a master-local proc is master; a
    # coupling connection contributes both sides and therefore raises).
    proc_sides: dict[str, set[str]] = {}
    if data.process_source_sink is not None:
        for r in data.process_source_sink.iter_rows(named=True):
            s = proc_sides.setdefault(r["p"], set())
            for n in (r["source"], r["sink"]):
                side = _node_side(n)
                if side is not None:
                    s.add(side)
    for p in master_local_procs:
        proc_sides.setdefault(p, set()).add("master")
    for fld in ("p_process_constraint_invested_capacity_coeff",
                "p_process_constraint_prebuilt_capacity_coeff"):
        f = _frame_of(getattr(data, fld, None))
        if f is None or "cn" not in f.columns:
            continue
        for r in f.iter_rows(named=True):
            for side in proc_sides.get(r["p"], set()):
                _add(r["cn"], r["p"], side)

    return sides_by_cn, refs_by_cn


def _validate_user_constraints(
    data: FlexData,
    all_region_nodes: set[str],
    master_nodes: "frozenset[str] | set[str]",
    master_local_procs: set[str],
) -> None:
    """Hard-error on any user constraint referencing both master-side
    and region-side entities.

    Sides are aggregated per constraint id (``cn``) across every
    user-constraint frame (:func:`_user_constraint_sides`): a constraint
    mixing a master-hosted node (or a master-local process) with a
    region node (or a region process) cannot live whole on either side
    of the decomposition — splitting it would silently lose terms.
    """
    sides_by_cn, refs_by_cn = _user_constraint_sides(
        data, all_region_nodes, master_nodes, master_local_procs)
    for cn in sorted(sides_by_cn):
        if {"region", "master"} <= sides_by_cn[cn]:
            raise RuntimeError(
                f"split: user constraint {cn!r} references both "
                f"master-side and region-side entities: "
                f"{sorted(refs_by_cn[cn])}.  A constraint cannot be "
                f"split between a region subproblem and the Benders "
                f"master — rewrite it to reference entities on one "
                f"side only (e.g. via the handover-connection pattern)."
            )


def _master_side_constraint_ids(
    data: FlexData,
    all_region_nodes: set[str],
    master_nodes: "frozenset[str] | set[str]",
    master_local_procs: set[str],
) -> set[str]:
    """User-constraint ids (``cn``) whose EVERY sided entity reference
    is master-side.  Those constraints live whole in the Benders master:
    :func:`master_network_data` keeps exactly their rows and
    :func:`split` drops them from every region (a region copy would
    degenerate to ``0 sense constant`` once its master-keyed coefficient
    rows are scrubbed).  Assumes :func:`_validate_user_constraints` ran
    (no mixed constraints)."""
    sides_by_cn, _ = _user_constraint_sides(
        data, all_region_nodes, master_nodes, master_local_procs)
    return {cn for cn, sides in sides_by_cn.items() if sides == {"master"}}


#: Group-feature SET fields (``(g,)``): a group present in one of these
#: activates the corresponding group-level constraint family
#: (capacity margin / inertia / non-sync).  Membership is ``group_node``.
_GROUP_FEATURE_SET_FIELDS: tuple[str, ...] = (
    "groupCapacityMargin", "groupInertia", "groupNonSync",
)

#: Group-feature PARAM fields (``(g,)`` / ``(g, d)``) filtered alongside
#: the set fields when partitioning feature groups to the master.
_GROUP_FEATURE_PARAM_FIELDS: tuple[str, ...] = (
    "p_inv_group_cap", "p_group_capacity_for_scaling",
    "pdGroup_capacity_margin",
)


def _master_side_feature_groups(
    data: FlexData,
    all_region_nodes: set[str],
    master_nodes: "frozenset[str] | set[str]",
) -> set[str]:
    """Feature-carrying groups whose member nodes are ALL master-hosted.

    A group in one of the :data:`_GROUP_FEATURE_SET_FIELDS` sets with
    at least one master-hosted member must have EVERY member
    master-hosted — the group constraint sums over its members and
    cannot be enforced whole on either side otherwise (hard error,
    never a silent degrade).  Groups with no master members are left to
    the regions (today's semantics); bare ``group_node`` membership
    rows of non-feature groups are filtered silently.
    """
    feature_gs: set[str] = set()
    for fld in _GROUP_FEATURE_SET_FIELDS:
        f = getattr(data, fld, None)
        if f is not None and f.height > 0:
            feature_gs |= set(f["g"].cast(pl.Utf8).to_list())
    if not feature_gs:
        return set()
    members_by_g: dict[str, set[str]] = {}
    gn = data.group_node
    if gn is not None:
        for r in gn.iter_rows(named=True):
            members_by_g.setdefault(r["g"], set()).add(r["n"])
    master = set(master_nodes)
    kept: set[str] = set()
    for g in sorted(feature_gs):
        members = members_by_g.get(g, set())
        master_hit = members & master
        if not master_hit:
            continue
        if members <= master:
            kept.add(g)
            continue
        raise RuntimeError(
            f"split: feature-carrying group {g!r} straddles the "
            f"region/master boundary — it has master-hosted member "
            f"node(s) {sorted(master_hit)} AND non-master member(s) "
            f"{sorted(members - master)}.  A group-level constraint "
            f"(capacity margin / inertia / non-sync) sums over its "
            f"members and cannot be split between a region subproblem "
            f"and the Benders master — regroup the nodes so every "
            f"member is on one side."
        )
    return kept


def _co2_master_partition(
    data: FlexData,
    master_local_triples: set[tuple[str, str, str]],
    region_master_triples: set[tuple[str, str, str]],
) -> set[str]:
    """Validate the CO2 frames against the master boundary and return
    the CO2-capped groups whose EVERY capped flow is a master-local arc
    (those cap constraints live whole in the master).

    Hard errors (never a silent degrade):

    * a CO2-priced flow row on a region↔master coupling arc — the arc's
      flow lives natively in the master while the region holds the
      half-flow, so neither side can carry the CO2 cost whole;
    * a capped group mixing master-local and region-side flows — the
      shared cap cannot be enforced on either side alone.
    """
    for fld in ("flow_from_co2_priced", "flow_from_co2_priced_noEff"):
        df = getattr(data, fld, None)
        if df is None or df.height == 0:
            continue
        for r in df.iter_rows(named=True):
            t = (r["p"], r["source"], r["sink"])
            if t in region_master_triples:
                raise RuntimeError(
                    f"split: CO2-priced flow {t} sits on a "
                    f"region↔master coupling arc — the CO2 cost "
                    f"cannot be split between a region subproblem and "
                    f"the Benders master.  Keep CO2-priced flows off "
                    f"the handover connections."
                )
    master_flags_by_g: dict[str, list[bool]] = {}
    for fld in ("flow_from_co2_capped", "flow_from_co2_capped_noEff",
                "flow_from_co2_capped_total",
                "flow_from_co2_capped_total_noEff"):
        df = getattr(data, fld, None)
        if df is None or df.height == 0:
            continue
        for r in df.iter_rows(named=True):
            t = (r["p"], r["source"], r["sink"])
            master_flags_by_g.setdefault(r["g"], []).append(
                t in master_local_triples)
    kept: set[str] = set()
    for g in sorted(master_flags_by_g):
        flags = master_flags_by_g[g]
        if all(flags):
            kept.add(g)
        elif any(flags):
            raise RuntimeError(
                f"split: CO2-capped group {g!r} straddles the "
                f"region/master boundary — it caps both master-local "
                f"and region-side flows.  A shared CO2 cap cannot be "
                f"split between a region subproblem and the Benders "
                f"master — regroup so every capped flow is on one side."
            )
    return kept


def _drop_master_rows(
    rd: FlexData,
    master_nodes: "frozenset[str] | set[str]",
    master_procs: set[str],
    master_cns: "frozenset[str] | set[str]" = frozenset(),
    master_groups: "frozenset[str] | set[str]" = frozenset(),
) -> FlexData:
    """Scrub every region frame/Param of master-hosted content: rows
    keyed to a master-hosted node (``n`` axis), a master-local process
    (``p`` axis), either (entity ``e`` axis), an all-master user
    constraint (``cn`` axis) or an all-master feature / CO2-cap group
    (``g`` axis).

    ``keep_nodes`` / ``keep_procs`` filtering already excludes master
    entities from the frames :func:`_build_region_data` filters
    explicitly; this pass additionally covers the frames the splitter
    historically carried through whole under shared-replicate semantics
    (invest/annuity/cost frames, ``process_indirect``, node profiles,
    …) — regions must carry NO rows for master-hosted entities (F3).
    The ``cn`` drop covers the constraint-id-only frames
    (``p_constraint_constant``, ``cdt_eq/le/ge``) whose master-keyed
    coefficient rows are scrubbed by the ``n``/``p`` drops — without it
    a region copy of an all-master constraint degenerates to
    ``0 sense constant``.  The ``g`` drop is the analogous guard for
    all-master group features (a region copy with empty membership
    would still charge the feature's slack penalty).
    Arc rows touching a master node are dropped separately via the
    coupling/master-local triples in ``cross_arcs_by_pss``.
    """
    node_drop = set(master_nodes)
    proc_drop = set(master_procs)
    entity_drop = node_drop | proc_drop
    cn_drop = set(master_cns)
    g_drop = set(master_groups)
    if not (entity_drop or cn_drop or g_drop):
        return rd
    for f in dataclasses.fields(rd):
        v = getattr(rd, f.name)
        if v is None:
            continue
        if isinstance(v, Param):
            frame = v.frame
        elif isinstance(v, pl.DataFrame):
            frame = v
        else:
            continue
        changed = False
        for col, drop in (("n", node_drop), ("p", proc_drop),
                          ("e", entity_drop), ("cn", cn_drop),
                          ("g", g_drop)):
            if drop and col in frame.columns:
                frame = frame.filter(~_is_in_keep(col, drop))
                changed = True
        if not changed:
            continue
        if isinstance(v, Param):
            setattr(rd, f.name, Param(v.dims, frame, name=v.name))
        else:
            setattr(rd, f.name, frame)
    return rd


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
    benders_uncap_cross_region: bool = False,
    *,
    master_hosted_nodes: "frozenset[str] | set[str]" = frozenset(),
    master_local_procs: "set[str] | frozenset[str]" = frozenset(),
    master_cns: "frozenset[str] | set[str]" = frozenset(),
    master_groups: "frozenset[str] | set[str]" = frozenset(),
) -> FlexData:
    """Construct one region's :class:`FlexData` by filtering+rewriting
    the whole-system frames/Params.

    ``keep_nodes``/``keep_procs`` are the in-region+shared sets.
    ``cross_arcs_by_pss`` is the SET of (p, source, sink) tuples to
    REMOVE from this region's process frames (they're being replaced
    by half-flow virtual arcs; master-hosted mode also routes the
    region↔master coupling arcs and the master-local arcs through it).
    ``master_hosted_nodes`` / ``master_local_procs`` / ``master_cns`` /
    ``master_groups`` (master-hosted mode only; empty by default ⇒
    byte-identical path) trigger the :func:`_drop_master_rows` scrub so
    the region carries NO rows for master-hosted entities, all-master
    user constraints or all-master feature/CO2-cap groups.
    """
    # Start by shallow-copying the dataclass and clearing fields we'll
    # explicitly rewrite.
    new = dataclasses.replace(src)

    # ---- Filter primary entity sets ----
    new.nodeBalance = _filter_frame(src.nodeBalance, "n", keep_nodes)
    # Phase E.3: ``nodeBalance_dt`` is no longer materialised on src; the
    # filtered ``new.nodeBalance`` is the only set we need, and
    # ``_pdt_join.compute_nodeBalance_dt(new)`` produces the cross-join
    # on demand downstream.
    new.nodeBalance_dt = None
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
    # Phase E.3: ``pss_dt`` is no longer materialised on src; the filtered
    # ``new.process_source_sink`` is the only set we need, and
    # ``_pdt_join.compute_pss_dt(new)`` produces the cross-join on demand
    # downstream.  Half-flow injection below ALSO needs a pss_dt view; it
    # builds one locally from src for the arc-dt extraction.
    new.pss_dt = None
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
    # Phase E.3: ``nodeState_dt`` is no longer materialised on src; the
    # filtered ``new.nodeState`` is the only set we need, and
    # ``_pdt_join.compute_nodeState_dt(new)`` produces the cross-join
    # on demand downstream.  ``nodeState_first_dt`` is still
    # materialised (small one-row-per-node slice; see ``_load_storage``).
    new.nodeState_dt = None
    new.nodeState_first_dt = _filter_frame(src.nodeState_first_dt, "n", keep_nodes)
    new.storage_bind_within_timeblock = _filter_frame(src.storage_bind_within_timeblock, "n", keep_nodes)
    new.storage_bind_forward_only = _filter_frame(src.storage_bind_forward_only, "n", keep_nodes)
    new.storage_bind_within_solve = _filter_frame(src.storage_bind_within_solve, "n", keep_nodes)
    new.storage_bind_within_solve_blended_weights = _filter_frame(
        getattr(src, "storage_bind_within_solve_blended_weights", None),
        "n", keep_nodes,
    )
    # Phase D — added with the new variant's constraint wiring.
    new.storage_bind_forward_only_blended_weights = _filter_frame(
        getattr(src, "storage_bind_forward_only_blended_weights", None),
        "n", keep_nodes,
    )
    # Phase E — per-period cyclic-closure variant.
    new.storage_bind_within_period_blended_weights = _filter_frame(
        getattr(src, "storage_bind_within_period_blended_weights", None),
        "n", keep_nodes,
    )
    new.storage_fix_start = _filter_frame(src.storage_fix_start, "n", keep_nodes)
    new.nodeStateBlock = _filter_frame(src.nodeStateBlock, "n", keep_nodes)
    new.nodeState_rp = _filter_frame(
        getattr(src, "nodeState_rp", None), "n", keep_nodes,
    )
    # The remaining RP-blended-weights fields (rp_base_period_set,
    # rp_base_chain, rp_base_first, rp_base_last, rp_block_first,
    # p_rp_last_step, rp_base__rep) are solve-data-keyed (period / step
    # / base / rep) — not entity-keyed — so the ``dataclasses.replace``
    # shallow copy above already carries them through unchanged.
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

    # ---- Master-hosted mode: scrub master entities from the frames the
    # splitter otherwise carries through whole (invest/annuity/cost,
    # process_indirect, node profiles, …).  No-op when both sets are
    # empty (the byte-identical default path).
    if (master_hosted_nodes or master_local_procs or master_cns
            or master_groups):
        new = _drop_master_rows(
            new, master_hosted_nodes, set(master_local_procs),
            master_cns=master_cns, master_groups=master_groups)

    # ---- Inject virtual half-flow arcs ----
    if half_flows:
        new = _inject_half_flows(
            new, src, half_flows,
            benders_uncap_cross_region=benders_uncap_cross_region,
        )

    return new


#: Benders mode sentinel for the cross-region half-flow ``maxFlow``
#: capacity.  The real ``f ≤ C·unitsize`` limit lives in the master, so
#: the per-region half-flow must be effectively uncapped.  The largest
#: achievable physical flow is bounded by the connection's
#: ``invest_max_total · unitsize`` (and ``v_flow`` is normalised by
#: unitsize, so in solver units it is bounded by ``invest_max_total``);
#: 1e12 is comfortably ≫ any realistic ``invest_max_total``, so the
#: half-flow's ``maxFlow`` row is structurally slack for any flow the
#: master could pin — it can never bind and therefore cannot leak a dual
#: into the per-region subproblem (Phase-1 Claim 4).
_BENDERS_UNCAP_SENTINEL: float = 1e12


def _inject_half_flows(
    rd: FlexData, src: FlexData, half_flows: list[HalfFlow],
    *,
    benders_uncap_cross_region: bool = False,
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
      the pipe; the master pins the actual flow each iteration).
    """
    if not half_flows:
        return rd

    # Capture the original arc rows so we can pull their (d, t) shape and
    # flow_upper Param values.
    # Phase E.3: ``src.pss_dt`` is no longer materialised; build it on
    # demand from the constituents.  Half-flow injection always touches
    # the full arc-dt grid for the cross-region arcs, so a one-shot build
    # here is fine.
    orig_pss_dt = compute_pss_dt(src)
    orig_flow_upper = src.p_flow_upper
    orig_flow_upper_existing = src.p_flow_upper_existing
    orig_unitsize = src.p_unitsize
    orig_eff = src.process_source_sink_eff

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
                # Benders mode: the master owns the real ``f ≤ C·unitsize``
                # capacity limit, so the per-region half-flow's ``maxFlow``
                # bound must be effectively unbounded — otherwise a
                # greenfield pipe (whose inherited ``existing`` is 0) is
                # pinned to zero trade (the false-convergence bug).  Swap
                # the inherited value for a large sentinel that can never
                # bind.  Default (un-set) keeps today's inherit.
                value = (_BENDERS_UNCAP_SENTINEL
                         if benders_uncap_cross_region
                         else float(r["value"]))
                new_flow_upper_existing_rows.append({
                    "p": hf.virtual_p,
                    "source": hf.virtual_arc_source,
                    "sink": hf.virtual_arc_sink,
                    "d": r["d"],
                    "value": value,
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

    # The widened Enum vocabulary set by ``split()`` lives on the global
    # ContextVar.  ``src._axis_enums`` is the narrower snapshot captured
    # before widening — using it for new-row schemas would null the
    # virtual ``hf_pipe_*`` / virtual-node tokens.  Read from the global
    # first; the live widened vocabulary is a strict superset of the
    # source snapshot, so upcasting existing-frame columns to it is safe.
    _enums_loc = get_global_axis_enums() or getattr(src, "_axis_enums", None)

    def _upcast_dims(frame: pl.DataFrame | None,
                     cols: Iterable[str]) -> pl.DataFrame | None:
        """Re-cast the named dim columns on ``frame`` to the wider Enum
        vocabulary in ``_enums_loc``.  Required before concat against
        new rows built with ``schema_dtype(_enums_loc, axis)`` — both
        sides must agree on the Enum vocabulary or polars raises."""
        if frame is None:
            return None
        exprs = [cast_dim(pl.col(c), _enums_loc, c) for c in cols]
        return frame.with_columns(exprs)

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
        # Upcast orig's dim columns to the wider Enum before concat so
        # both sides agree on the vocabulary.  Value-typed columns
        # (Float64) pass through ``cast_dim`` unchanged.
        orig_w = _upcast_dims(orig, list(schema.keys()))
        return pl.concat([orig_w.select(list(schema.keys())),
                          new_df.select(list(schema.keys()))],
                         how="vertical")

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
    # Phase E.3: ``rd.pss_dt`` is no longer persisted; the half-flow
    # virtual (p, source, sink) rows already appended to
    # ``rd.process_source_sink`` will produce the matching cross-join
    # rows when ``compute_pss_dt(rd)`` runs downstream.  We build a
    # local ``virtual_pss_dt`` (just the half-flow rows) for the
    # availability / existing_count promotion below.
    _virtual_pss_dt_schema = {
        **_pss_schema,
        "d": schema_dtype(_enums_loc, "d"),
        "t": schema_dtype(_enums_loc, "t"),
    }
    if new_pss_dt_rows:
        virtual_pss_dt = pl.DataFrame(
            new_pss_dt_rows, schema=_virtual_pss_dt_schema)
    else:
        virtual_pss_dt = pl.DataFrame(schema=_virtual_pss_dt_schema)
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
        new_us = pl.DataFrame(new_unitsize_rows,
                              schema={"p": schema_dtype(_enums_loc, "p"),
                                      "value": pl.Float64})
        merged_us = pl.concat([_upcast_dims(rd.p_unitsize.frame, ("p",))
                                 .select("p", "value"),
                               new_us], how="vertical")
        rd.p_unitsize = Param(("p",), merged_us, name=rd.p_unitsize.name)

    # Append p_slope rows for half-flows that are eff-classified.
    if rd.p_slope is not None and new_p_slope_rows:
        new_sl = pl.DataFrame(new_p_slope_rows,
                              schema={"p": schema_dtype(_enums_loc, "p"),
                                      "d": schema_dtype(_enums_loc, "d"),
                                      "t": schema_dtype(_enums_loc, "t"),
                                      "value": pl.Float64})
        merged_sl = pl.concat([_upcast_dims(rd.p_slope.frame, ("p", "d", "t"))
                                 .select("p", "d", "t", "value"),
                               new_sl], how="vertical")
        rd.p_slope = Param(("p", "d", "t"), merged_sl, name=rd.p_slope.name)

    # Append flow_upper Param rows.
    if rd.p_flow_upper is not None and new_flow_upper_rows:
        new_fu = pl.DataFrame(new_flow_upper_rows,
                              schema={"p": schema_dtype(_enums_loc, "p"),
                                      "source": schema_dtype(_enums_loc, "source"),
                                      "sink": schema_dtype(_enums_loc, "sink"),
                                      "d": schema_dtype(_enums_loc, "d"),
                                      "t": schema_dtype(_enums_loc, "t"),
                                      "value": pl.Float64})
        merged_fu = pl.concat([_upcast_dims(rd.p_flow_upper.frame,
                                            ("p", "source", "sink", "d", "t"))
                                 .select("p", "source", "sink", "d", "t", "value"),
                               new_fu], how="vertical")
        rd.p_flow_upper = Param(("p", "source", "sink", "d", "t"),
                                merged_fu, name=rd.p_flow_upper.name)
    if rd.p_flow_upper_existing is not None and new_flow_upper_existing_rows:
        new_fue = pl.DataFrame(new_flow_upper_existing_rows,
                               schema={"p": schema_dtype(_enums_loc, "p"),
                                       "source": schema_dtype(_enums_loc, "source"),
                                       "sink": schema_dtype(_enums_loc, "sink"),
                                       "d": schema_dtype(_enums_loc, "d"),
                                       "value": pl.Float64})
        merged_fue = pl.concat([_upcast_dims(rd.p_flow_upper_existing.frame,
                                             ("p", "source", "sink", "d"))
                                  .select("p", "source", "sink", "d", "value"),
                                new_fue], how="vertical")
        rd.p_flow_upper_existing = Param(
            ("p", "source", "sink", "d"),
            merged_fue, name=rd.p_flow_upper_existing.name)

    # ── p_process_availability and p_process_existing_count ──
    # The maxFlow RHS is multiplied by p_process_availability when
    # populated, and Param×Param is an inner-join so missing half-flow
    # entries collapse to zero RHS.  We must add availability=1.0 and
    # existing_count=1.0 entries so the half-flow's bound stays at the
    # value we set in p_flow_upper_existing.
    if rd.p_process_availability is not None and virtual_pss_dt.height > 0:
        # Add a (p, d, t) row for each (virtual_p, d, t) in
        # virtual_pss_dt (Phase E.3: half-flow rows only, no need to
        # filter the whole-region cross-join).
        avail_rows = (virtual_pss_dt
                      .select("p", "d", "t")
                      .with_columns(value=pl.lit(1.0)))
        if avail_rows.height > 0:
            # Phase E.1: p_process_availability dims depend on authored
            # shape — promote to (p, d, t) via virtual_pss_dt's d/t
            # axes so the concat lands at a uniform schema.
            avail_pdt = promote_param_to_dt(
                rd.p_process_availability, virtual_pss_dt).collect()
            merged = pl.concat([_upcast_dims(avail_pdt, ("p", "d", "t"))
                                  .select("p", "d", "t", "value"),
                                avail_rows], how="vertical")
            rd.p_process_availability = Param(
                ("p", "d", "t"), merged,
                name=rd.p_process_availability.name)
    if rd.p_process_existing_count is not None and virtual_pss_dt.height > 0:
        # (p, d) row for each virtual half-flow
        ec_rows = (virtual_pss_dt
                   .filter(cast_dim(pl.col("p"), None, "p").is_in([hf.virtual_p for hf in half_flows]))
                   .select("p", "d").unique()
                   .with_columns(value=pl.lit(1.0)))
        if ec_rows.height > 0:
            merged = pl.concat([_upcast_dims(rd.p_process_existing_count.frame,
                                             ("p", "d"))
                                  .select("p", "d", "value"),
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

    if rd.arc_source_block_dt is not None and src_block_rows:
        new_df = pl.DataFrame(
            [{k: r[k] for k in ("p", "source", "sink", "d", "b_first", "t", "weight")}
             for r in src_block_rows],
            schema={"p": schema_dtype(_enums_loc, "p"),
                    "source": schema_dtype(_enums_loc, "source"),
                    "sink": schema_dtype(_enums_loc, "sink"),
                    "d": schema_dtype(_enums_loc, "d"),
                    "b_first": schema_dtype(_enums_loc, "b_first"),
                    "t": schema_dtype(_enums_loc, "t"),
                    "weight": pl.Float64})
        rd.arc_source_block_dt = pl.concat([
            _upcast_dims(rd.arc_source_block_dt,
                         ("p", "source", "sink", "d", "b_first", "t"))
              .select(*new_df.columns), new_df],
            how="vertical")
    if rd.arc_sink_block_dt is not None and snk_block_rows:
        new_df = pl.DataFrame(
            [{k: r[k] for k in ("p", "source", "sink", "d", "b_first", "t", "weight")}
             for r in snk_block_rows],
            schema={"p": schema_dtype(_enums_loc, "p"),
                    "source": schema_dtype(_enums_loc, "source"),
                    "sink": schema_dtype(_enums_loc, "sink"),
                    "d": schema_dtype(_enums_loc, "d"),
                    "b_first": schema_dtype(_enums_loc, "b_first"),
                    "t": schema_dtype(_enums_loc, "t"),
                    "weight": pl.Float64})
        rd.arc_sink_block_dt = pl.concat([
            _upcast_dims(rd.arc_sink_block_dt,
                         ("p", "source", "sink", "d", "b_first", "t"))
              .select(*new_df.columns), new_df],
            how="vertical")
    if rd.p_arc_source_weight is not None and src_w_rows:
        new_df = pl.DataFrame(
            [{k: r[k] for k in ("p", "source", "sink", "d", "t", "value")}
             for r in src_w_rows],
            schema={"p": schema_dtype(_enums_loc, "p"),
                    "source": schema_dtype(_enums_loc, "source"),
                    "sink": schema_dtype(_enums_loc, "sink"),
                    "d": schema_dtype(_enums_loc, "d"),
                    "t": schema_dtype(_enums_loc, "t"),
                    "value": pl.Float64})
        rd.p_arc_source_weight = Param(
            ("p", "source", "sink", "d", "t"),
            pl.concat([_upcast_dims(rd.p_arc_source_weight.frame,
                                    ("p", "source", "sink", "d", "t"))
                         .select(*new_df.columns),
                       new_df], how="vertical"),
            name=rd.p_arc_source_weight.name)
    if rd.p_arc_sink_weight is not None and snk_w_rows:
        new_df = pl.DataFrame(
            [{k: r[k] for k in ("p", "source", "sink", "d", "t", "value")}
             for r in snk_w_rows],
            schema={"p": schema_dtype(_enums_loc, "p"),
                    "source": schema_dtype(_enums_loc, "source"),
                    "sink": schema_dtype(_enums_loc, "sink"),
                    "d": schema_dtype(_enums_loc, "d"),
                    "t": schema_dtype(_enums_loc, "t"),
                    "value": pl.Float64})
        rd.p_arc_sink_weight = Param(
            ("p", "source", "sink", "d", "t"),
            pl.concat([_upcast_dims(rd.p_arc_sink_weight.frame,
                                    ("p", "source", "sink", "d", "t"))
                         .select(*new_df.columns),
                       new_df], how="vertical"),
            name=rd.p_arc_sink_weight.name)

    # group_entity / group_node — augment with virtual entities under the
    # half-flow's region (so downstream group-aware emitters don't
    # spuriously skip them).  Skip — the group_* sets only matter for
    # group_slack / capacity_margin features which aren't in lh2 fixture.

    return rd


# ---------------------------------------------------------------------------
# Benders network-only master producer (the INVERSE of ``split``)
# ---------------------------------------------------------------------------


#: FlexData fields keyed on a NODE axis ``n`` (node entity sets, storage,
#: inflow, penalties, profiles, …).  The master OMITS every terminal node
#: from balance, so each of these is emptied (set to ``None``) — a node
#: absent from ``nodeBalance`` / ``nodeStateBlock`` gets no balance row
#: (Phase-3 design §1.3; ``model.py`` builds balance only over the
#: populated node sets).
_MASTER_NODE_FIELDS: tuple[str, ...] = (
    "nodeBalance_dt", "nodeBalancePeriod",
    "nodeState", "nodeState_dt", "nodeState_first_dt", "nodeState_last_dt",
    "nodeState_rp", "nodeStateBlock",
    "storage_bind_within_timeblock", "storage_bind_forward_only",
    "storage_bind_within_solve", "storage_bind_within_solve_blended_weights",
    "storage_bind_within_period_blended_weights",
    "storage_bind_forward_only_blended_weights", "storage_fix_start",
    "p_state_upper", "p_state_unitsize", "p_state_self_discharge",
    "p_state_start", "p_state_existing_capacity",
    "storage_use_reference_value", "p_storage_state_reference_value",
    "p_storage_state_reference_price",
    "node_profile_upper", "node_profile_lower", "node_profile_fixed",
    "p_node_availability", "p_roll_continue_state",
    "n_fix_storage_quantity", "ndt_fix_storage_quantity",
    "p_fix_storage_quantity", "n_fix_storage_usage",
    "ndt_fix_storage_usage", "p_fix_storage_usage",
    "p_node_capacity_for_scaling",
    # node-keyed invest/divest sets (the trade nodes are not the master's
    # invest variables — only the cross connections are).
    "nd_invest_set", "nd_divest_set",
    # CO2 / reserve / user-constraint / group features — all in-region
    # recourse, not part of the network-only master.  (The commodity
    # frames + ``p_commodity_price`` are REQUIRED-present by the PROCESSES
    # feature, so they are EMPTIED rather than nulled below.)
    "flow_from_co2_priced", "flow_from_co2_priced_noEff",
    "p_co2_content", "p_co2_price",
    "group_co2_max_period", "flow_from_co2_capped",
    "flow_from_co2_capped_noEff", "p_co2_max_period", "group_d_co2_capped",
    "group_co2_max_total", "flow_from_co2_capped_total",
    "flow_from_co2_capped_total_noEff", "p_co2_max_total",
    "flow_constraint_idx", "p_flow_constraint_coef", "p_constraint_constant",
    "cdt_eq", "cdt_le", "cdt_ge",
    "p_node_constraint_invested_capacity_coeff",
    "p_process_constraint_invested_capacity_coeff",
    "p_node_constraint_state_coeff",
    "p_node_constraint_prebuilt_capacity_coeff",
    "p_process_constraint_prebuilt_capacity_coeff",
    "groupCapacityMargin", "groupInertia", "groupNonSync", "group_node",
    "process_sink_inertia", "process_source_inertia",
    "process_sink_nonSync", "process_group_inside_nonSync",
    "p_inv_group_cap", "p_group_capacity_for_scaling",
    "pdGroup_capacity_margin",
)


# ---------------------------------------------------------------------------
# Master-hosted-mode partition of ``_MASTER_NODE_FIELDS``.  With a
# non-empty ``master_hosted_nodes`` set the master KEEPS master-side
# content instead of nulling everything: each tuple below routes its
# fields through the matching keep-filter in ``master_network_data``.
# Every ``_MASTER_NODE_FIELDS`` member appears in exactly one of these
# groups (pinned by a test) so a schema addition to the legacy tuple
# cannot silently skip the master-hosted path.
# ---------------------------------------------------------------------------

#: ``n``-keyed frames/Params → filter rows to the master-hosted nodes
#: (balance/state/storage/profile/availability/fix/invest-set frames;
#: ``group_node`` bare membership rows are filtered silently the same
#: way, mirroring ``split``).
_MASTER_N_KEYED_FIELDS: tuple[str, ...] = (
    "nodeBalance_dt", "nodeBalancePeriod",
    "nodeState", "nodeState_dt", "nodeState_first_dt", "nodeState_last_dt",
    "nodeState_rp", "nodeStateBlock",
    "storage_bind_within_timeblock", "storage_bind_forward_only",
    "storage_bind_within_solve", "storage_bind_within_solve_blended_weights",
    "storage_bind_within_period_blended_weights",
    "storage_bind_forward_only_blended_weights", "storage_fix_start",
    "p_state_upper", "p_state_unitsize", "p_state_self_discharge",
    "p_state_start", "p_state_existing_capacity",
    "storage_use_reference_value", "p_storage_state_reference_value",
    "p_storage_state_reference_price",
    "node_profile_upper", "node_profile_lower", "node_profile_fixed",
    "p_node_availability", "p_roll_continue_state",
    "n_fix_storage_quantity", "ndt_fix_storage_quantity",
    "p_fix_storage_quantity", "n_fix_storage_usage",
    "ndt_fix_storage_usage", "p_fix_storage_usage",
    "p_node_capacity_for_scaling",
    "nd_invest_set", "nd_divest_set",
    "group_node",
)

#: ``cn``-keyed user-constraint frames/Params → keep rows of the
#: all-master constraint ids (:func:`_master_side_constraint_ids`).
_MASTER_CN_KEYED_FIELDS: tuple[str, ...] = (
    "flow_constraint_idx", "p_flow_constraint_coef",
    "p_constraint_constant", "cdt_eq", "cdt_le", "cdt_ge",
    "p_node_constraint_invested_capacity_coeff",
    "p_process_constraint_invested_capacity_coeff",
    "p_node_constraint_state_coeff",
    "p_node_constraint_prebuilt_capacity_coeff",
    "p_process_constraint_prebuilt_capacity_coeff",
)

#: Arc-keyed CO2-priced flow frames → keep master-local arc rows
#: (coupling-arc rows hard-error in :func:`_co2_master_partition`).
_MASTER_CO2_ARC_FIELDS: tuple[str, ...] = (
    "flow_from_co2_priced", "flow_from_co2_priced_noEff",
)

#: ``g``-keyed CO2-cap frames → keep rows of the all-master capped
#: groups (:func:`_co2_master_partition`).
_MASTER_CO2_GROUP_FIELDS: tuple[str, ...] = (
    "group_co2_max_period", "flow_from_co2_capped",
    "flow_from_co2_capped_noEff", "p_co2_max_period",
    "group_d_co2_capped",
    "group_co2_max_total", "flow_from_co2_capped_total",
    "flow_from_co2_capped_total_noEff", "p_co2_max_total",
)

#: Pure lookup tables consumed by the kept CO2 rows via joins — carried
#: whole (rows for un-kept groups/commodities are inert).
_MASTER_CO2_LOOKUP_FIELDS: tuple[str, ...] = (
    "p_co2_content", "p_co2_price",
)

#: ``p``-keyed group-feature side frames → filter to the master's
#: kept processes (rows for absent flows are inert Where-filters).
_MASTER_GROUP_PROC_FIELDS: tuple[str, ...] = (
    "process_sink_inertia", "process_source_inertia",
    "process_sink_nonSync", "process_group_inside_nonSync",
)


def master_network_data(
    data: FlexData,
    regions: list[str],
    *,
    region_membership: dict[str, dict[str, set[str]]] | None = None,
    master_hosted_nodes: frozenset[str] = frozenset(),
) -> FlexData:
    """Build the Benders MASTER's reduced :class:`FlexData` — the INVERSE
    of :func:`split`.

    Returns a reduced :class:`FlexData` containing ONLY the cross-region
    ``(p, source, sink)`` arcs plus their invest / cost / timeline params,
    with EVERY terminal node OMITTED from node balance (and the
    block / state / inflow frames).  ``build_flextool`` over the result
    generates the master skeleton natively:

    * ``v_flow[conn, source, sink, d, t]`` for every cross arc (built from
      ``process_source_sink × dt``, independent of ``nodeBalance``);
    * ``v_invest_p[conn, d]`` over ``pd_invest_set``;
    * the capacity-tied ``maxFlow`` row (greenfield ⇒ ``flow_upper_rhs=0``
      with ``-v_invest_p`` on the LHS ⇒ ``v_flow ≤ v_invest_p``);
    * the invest annuity cost + (when authored) the connection flow cost.

    Reuses the cross-arc CLASSIFICATION (:func:`_classify_arcs`) — the same
    detection :func:`split` uses — rather than re-deriving it.

    Parameters
    ----------
    data
        Whole-system :class:`FlexData`.
    regions
        The region group names the splitter partitions on.
    region_membership
        Pre-computed membership (see :func:`load_region_membership`); when
        omitted, re-derived from *data*.
    master_hosted_nodes
        Master-hosted node mode (see :func:`compute_master_hosted_nodes`
        / :func:`split`).  With the default EMPTY set the function takes
        today's exact path (explicit early branches — byte-identical).
        Non-empty: the master KEEPS the master-side model instead of
        emptying it — see Notes.

    Notes
    -----
    With the default empty ``master_hosted_nodes``, the reduced FlexData
    KEEPS only cross-arc rows in the process / arc / arc-cost /
    arc-block frames and the cross connections in the invest / unitsize
    / max-units params; EVERY node-keyed frame (balance, state, inflow,
    penalties, storage, profiles, CO2, groups, user constraints) is
    emptied (``None``).

    With a NON-EMPTY ``master_hosted_nodes``:

    * the arc keep-set widens to cross-region ∪ region↔master coupling
      ∪ master-local arcs (and the proc/cost keep-sets follow), so
      master-local connections AND units build natively in the master;
    * ``n``-keyed frames are FILTERED to the master-hosted nodes instead
      of nulled (balance, inflow, penalties, storage/state, profiles,
      availability, fix-storage, node invest sets);
    * the entity-keyed invest frames keep
      ``procs-of-kept-arcs ∪ master_hosted_nodes`` (master storage
      invest needs the ``e``-keyed annuity/lifetime/max-units rows);
    * only the REGION-side endpoints of coupling arcs (and the
      region↔region terminals) are omitted from balance — master-hosted
      endpoints stay balanced;
    * user constraints / CO2 caps / group features whose referenced
      entities are ALL master-side are kept; straddling ones hard-error
      (same validation family as :func:`split`).

    In both modes the solve-data-keyed timeline frames (``dt``,
    ``p_step_duration``, ``p_timestep_weight``, ``p_inflation_op``,
    ``p_period_share``, the RP / block frames) carry through the
    ``dataclasses.replace`` shallow copy unchanged, so the master's
    ``v_flow`` lives on the SAME ``(d, t)`` grid as the region pinned
    half-flows (Phase-3 §3.5 guard (a)).
    """
    if region_membership is None:
        region_membership = load_region_membership(data, regions)
    region_nodes = {r: m["nodes"] for r, m in region_membership.items()}

    new = dataclasses.replace(data)

    if data.process_source_sink is None:
        raise RuntimeError(
            "master_network_data: no process_source_sink — nothing to "
            "decompose"
        )

    # Reuse the cross-region classification (same detection as ``split``).
    # With the default empty master set the extra classification frames
    # are empty and this path is byte-identical to the historical 2-way
    # behaviour.
    master_mode = bool(master_hosted_nodes)
    _pss_tagged, cross, region_master, master_local = _classify_arcs(
        data.process_source_sink, region_nodes,
        master_nodes=master_hosted_nodes,
    )
    cross_keys: set[tuple[str, str, str]] = {
        (r["p"], r["source"], r["sink"]) for r in cross.iter_rows(named=True)
    }
    master_cns: set[str] = set()
    master_groups_feature: set[str] = set()
    master_groups_co2: set[str] = set()
    master_local_keys: set[tuple[str, str, str]] = set()
    if not master_mode:
        # Today's exact path: region↔region cross arcs are the only
        # master content.
        if cross.height == 0:
            raise RuntimeError(
                "master_network_data: no cross-region arcs found"
            )
    else:
        all_region_nodes: set[str] = set()
        for ns in region_nodes.values():
            all_region_nodes |= ns
        overlap = set(master_hosted_nodes) & all_region_nodes
        if overlap:
            raise RuntimeError(
                f"master_network_data: master_hosted_nodes overlap "
                f"region membership: {sorted(overlap)} — a node is "
                f"either master-hosted (in no region group) or in "
                f"exactly one region, never both."
            )
        region_master_keys = {
            (r["p"], r["source"], r["sink"])
            for r in region_master.iter_rows(named=True)
        }
        master_local_keys = {
            (r["p"], r["source"], r["sink"])
            for r in master_local.iter_rows(named=True)
        }
        master_balance_nodes = set()
        for fld in ("nodeBalance", "nodeState"):
            f = getattr(data, fld, None)
            if f is not None and f.height > 0:
                master_balance_nodes |= (
                    set(f["n"].cast(pl.Utf8).to_list())
                    & set(master_hosted_nodes))
        if (not cross_keys and not region_master_keys
                and not master_local_keys and not master_balance_nodes):
            raise RuntimeError(
                "master_network_data: no coupling arcs and no master "
                "content — nothing to host in the Benders master"
            )
        # Same hard-validation family as ``split`` (D-a / R5): never
        # silently mis-partition authored data.  The messages are
        # identical to ``split``'s, so a driver run that already raised
        # there cannot raise differently here.
        _validate_no_straddling_units(
            data, all_region_nodes, master_hosted_nodes)
        _ml_procs = _master_local_procs(
            data.process_source_sink, master_local)
        _validate_user_constraints(
            data, all_region_nodes, master_hosted_nodes, _ml_procs)
        master_cns = _master_side_constraint_ids(
            data, all_region_nodes, master_hosted_nodes, _ml_procs)
        master_groups_feature = _master_side_feature_groups(
            data, all_region_nodes, master_hosted_nodes)
        master_groups_co2 = _co2_master_partition(
            data, master_local_keys, region_master_keys)
        # Arc keep-set = cross ∪ region↔master coupling ∪ master-local
        # (audit 0.D: the coupling-arc-only case is load-bearing).
        cross_keys = cross_keys | region_master_keys | master_local_keys
    cross_procs: set[str] = {k[0] for k in cross_keys}
    # Entity keep-set for the ``e``-keyed invest frames: master-hosted
    # STORAGE invest needs the node entities alongside the connections/
    # units of the kept arcs.
    entity_keep: set[str] = (
        cross_procs | set(master_hosted_nodes) if master_mode
        else cross_procs
    )

    _enums = getattr(data, "_axis_enums", None) or get_global_axis_enums()

    def _keep_cross_triple(df: pl.DataFrame | None) -> pl.DataFrame | None:
        """Keep ONLY the rows whose (p, source, sink) is a cross arc."""
        if df is None:
            return None
        if not all(c in df.columns for c in ("p", "source", "sink")):
            return df
        key_df = pl.DataFrame(
            {
                "p":      [t[0] for t in cross_keys],
                "source": [t[1] for t in cross_keys],
                "sink":   [t[2] for t in cross_keys],
            },
            schema={"p": schema_dtype(_enums, "p"),
                    "source": schema_dtype(_enums, "source"),
                    "sink": schema_dtype(_enums, "sink")},
        )
        return df.join(key_df, on=("p", "source", "sink"), how="semi")

    def _keep_cross_triple_param(p: Param | None) -> Param | None:
        if p is None:
            return None
        if not all(c in p.dims for c in ("p", "source", "sink")):
            # p-keyed but not arc-keyed (e.g. p_unitsize): keep cross procs.
            return _keep_proc_param(p)
        f = _keep_cross_triple(p.frame)
        return Param(p.dims, f, name=p.name)

    def _keep_proc(df: pl.DataFrame | None) -> pl.DataFrame | None:
        if df is None or "p" not in df.columns:
            return df
        return df.filter(_is_in_keep("p", cross_procs))

    def _keep_proc_param(p: Param | None) -> Param | None:
        if p is None:
            return None
        if "p" not in p.dims:
            return p
        return Param(p.dims, p.frame.filter(_is_in_keep("p", cross_procs)),
                     name=p.name)

    def _keep_entity_param(p: Param | None) -> Param | None:
        """Keep only the cross connections (+ master-hosted nodes in
        master mode) on an entity-axis (``e``) param."""
        if p is None:
            return None
        if "e" not in p.dims:
            return p
        return Param(p.dims, p.frame.filter(_is_in_keep("e", entity_keep)),
                     name=p.name)

    def _keep_entity_frame(df):
        """Keep cross connections (+ master-hosted nodes in master mode)
        on an ``e``-axis frame OR Param (some invest sets ship as plain
        DataFrames, others as Params)."""
        if df is None:
            return None
        if isinstance(df, Param):
            return _keep_entity_param(df)
        if "e" not in df.columns:
            return df
        return df.filter(_is_in_keep("e", entity_keep))

    # ---- Process / arc topology: KEEP only cross arcs ----
    new.process_source_sink = _keep_cross_triple(data.process_source_sink)
    new.process_source_sink_eff = _keep_cross_triple(data.process_source_sink_eff)
    new.process_source_sink_noEff = _keep_cross_triple(data.process_source_sink_noEff)
    new.pss_dt = None  # rebuilt on demand from process_source_sink × dt.
    new.process_source_canonical = _keep_proc(data.process_source_canonical)
    new.process_sink_canonical = _keep_proc(data.process_sink_canonical)
    new.flow_to_n = _keep_cross_triple(data.flow_to_n)
    new.flow_from_n = _keep_cross_triple(data.flow_from_n)
    new.flow_from_nodeBalance_eff = _keep_cross_triple(data.flow_from_nodeBalance_eff)
    new.flow_from_nodeBalance_noEff = _keep_cross_triple(data.flow_from_nodeBalance_noEff)
    new.process_unit = _keep_proc(data.process_unit)
    new.process_indirect = _keep_proc(data.process_indirect)
    # Commodity frames + price are REQUIRED-present by the PROCESSES feature.
    # Cross arcs (trade pipes) are not commodity-fed, so keeping only their
    # rows EMPTIES these frames while preserving the schema (not None).
    new.flow_from_commodity_eff = _keep_cross_triple(data.flow_from_commodity_eff)
    new.flow_from_commodity_noEff = _keep_cross_triple(data.flow_from_commodity_noEff)
    new.flow_to_commodity = _keep_cross_triple(data.flow_to_commodity)
    new.p_commodity_price = _keep_proc_param(data.p_commodity_price)

    # ---- Per-arc operating / capacity params ----
    new.p_unitsize = _keep_proc_param(data.p_unitsize)
    new.p_all_entity_unitsize = _keep_entity_param(data.p_all_entity_unitsize)
    new.p_flow_upper = _keep_cross_triple_param(data.p_flow_upper)
    new.p_flow_upper_existing = _keep_cross_triple_param(data.p_flow_upper_existing)
    new.p_arc_max_cap_coef = _keep_cross_triple_param(data.p_arc_max_cap_coef)
    new.p_slope = _keep_proc_param(data.p_slope)
    new.p_process_existing_count = _keep_proc_param(data.p_process_existing_count)
    new.p_process_availability = _keep_proc_param(data.p_process_availability)
    new.pd_neg_cap = _keep_proc_param(data.pd_neg_cap)

    # ---- Profiles (process-keyed) ----
    new.process_profile_upper = _keep_proc(data.process_profile_upper)
    new.process_profile_lower = _keep_proc(data.process_profile_lower)
    new.process_profile_fixed = _keep_proc(data.process_profile_fixed)

    # ---- Per-arc block weights (lh2 fixture; trade nodes are block nodes) ----
    new.arc_sink_block_dt = _keep_cross_triple(getattr(data, "arc_sink_block_dt", None))
    new.arc_source_block_dt = _keep_cross_triple(getattr(data, "arc_source_block_dt", None))
    new.p_arc_sink_weight = _keep_cross_triple_param(getattr(data, "p_arc_sink_weight", None))
    new.p_arc_source_weight = _keep_cross_triple_param(getattr(data, "p_arc_source_weight", None))
    new.p_arc_step_duration_sink = _keep_cross_triple_param(
        getattr(data, "p_arc_step_duration_sink", None))
    new.p_arc_step_duration_source = _keep_cross_triple_param(
        getattr(data, "p_arc_step_duration_source", None))

    # ---- Per-arc / per-process variable-cost frames (flow cost) ----
    new.pssdt_varCost_eff_connection = _keep_cross_triple(
        getattr(data, "pssdt_varCost_eff_connection", None))
    new.pssdt_varCost_eff_unit_source = _keep_cross_triple(
        getattr(data, "pssdt_varCost_eff_unit_source", None))
    new.pssdt_varCost_eff_unit_sink = _keep_cross_triple(
        getattr(data, "pssdt_varCost_eff_unit_sink", None))
    new.pssdt_varCost_noEff = _keep_cross_triple(
        getattr(data, "pssdt_varCost_noEff", None))
    new.p_pssdt_varCost = _keep_cross_triple_param(getattr(data, "p_pssdt_varCost", None))
    new.p_pdt_varCost_source = _keep_proc_param(getattr(data, "p_pdt_varCost_source", None))
    new.p_pdt_varCost_sink = _keep_proc_param(getattr(data, "p_pdt_varCost_sink", None))
    new.p_pdt_varCost_process = _keep_proc_param(getattr(data, "p_pdt_varCost_process", None))

    # ---- Invest params for the cross connections ----
    new.pd_invest_set = _keep_proc(data.pd_invest_set)
    new.pd_divest_set = _keep_proc(data.pd_divest_set)
    new.ed_invest_set = _keep_entity_frame(data.ed_invest_set)
    new.ed_divest_set = _keep_entity_frame(data.ed_divest_set)
    new.edd_invest_set = _keep_entity_frame(data.edd_invest_set)
    new.edd_invest_lookback_set = _keep_entity_frame(data.edd_invest_lookback_set)
    new.edd_divest_active = _keep_entity_frame(data.edd_divest_active)
    new.p_entity_max_units = _keep_entity_param(data.p_entity_max_units)
    new.ed_lifetime_fixed_cost = _keep_entity_param(data.ed_lifetime_fixed_cost)
    new.ed_lifetime_fixed_cost_divest = _keep_entity_param(data.ed_lifetime_fixed_cost_divest)
    new.ed_entity_annual_discounted = _keep_entity_param(data.ed_entity_annual_discounted)
    new.ed_entity_annual_divest_discounted = _keep_entity_param(
        data.ed_entity_annual_divest_discounted)
    new.e_invest_total = _keep_entity_frame(data.e_invest_total)
    new.e_divest_total = _keep_entity_frame(data.e_divest_total)
    new.e_invest_max_total = _keep_entity_frame(data.e_invest_max_total)
    new.e_divest_max_total = _keep_entity_frame(data.e_divest_max_total)
    new.ed_invest_period_set = _keep_entity_frame(data.ed_invest_period_set)
    new.ed_divest_period_set = _keep_entity_frame(data.ed_divest_period_set)
    new.ed_invest_max_period = _keep_entity_frame(data.ed_invest_max_period)
    new.ed_divest_max_period = _keep_entity_frame(data.ed_divest_max_period)
    new.p_entity_previously_invested_capacity = _keep_entity_param(
        data.p_entity_previously_invested_capacity)
    new.p_entity_invested = _keep_entity_param(data.p_entity_invested)
    new.p_entity_divested = _keep_entity_param(data.p_entity_divested)
    new.p_entity_all_existing = _keep_entity_param(data.p_entity_all_existing)
    new.p_ed_fixed_cost = _keep_entity_param(data.p_ed_fixed_cost)

    # ---- Node-keyed / recourse frames ----
    if not master_mode:
        # Today's exact path: drop in-region features that reference the
        # OMITTED terminal nodes (or in-region recourse not part of the
        # network-only master).
        for fld in _MASTER_NODE_FIELDS:
            if hasattr(new, fld):
                setattr(new, fld, None)
    else:
        # Master-hosted mode: KEEP the master-side content.  Every
        # ``_MASTER_NODE_FIELDS`` member is routed through exactly one
        # of the keep-filters below (partition pinned by test).
        def _filter_field(obj, col: str, keep: set[str]):
            """Filter a frame OR Param to ``col ∈ keep`` (no-op on
            ``None`` / missing column, mirroring the region-side
            helpers)."""
            if obj is None:
                return None
            if isinstance(obj, Param):
                return _filter_param(obj, col, keep)
            return _filter_frame(obj, col, keep)

        def _semi_triples(obj, keys: set[tuple[str, str, str]]):
            """Keep only the rows whose (p, source, sink) is in *keys*
            (frame or Param)."""
            if obj is None:
                return None
            frame = obj.frame if isinstance(obj, Param) else obj
            if not all(c in frame.columns for c in ("p", "source", "sink")):
                return obj
            key_df = pl.DataFrame(
                {
                    "p":      [t[0] for t in keys],
                    "source": [t[1] for t in keys],
                    "sink":   [t[2] for t in keys],
                },
                schema={"p": schema_dtype(_enums, "p"),
                        "source": schema_dtype(_enums, "source"),
                        "sink": schema_dtype(_enums, "sink")},
            )
            out = frame.join(key_df, on=("p", "source", "sink"), how="semi")
            if isinstance(obj, Param):
                return Param(obj.dims, out, name=obj.name)
            return out

        master_set = set(master_hosted_nodes)
        for fld in _MASTER_N_KEYED_FIELDS:
            if hasattr(new, fld):
                setattr(new, fld,
                        _filter_field(getattr(data, fld, None), "n",
                                      master_set))
        for fld in _MASTER_CN_KEYED_FIELDS:
            if hasattr(new, fld):
                setattr(new, fld,
                        _filter_field(getattr(data, fld, None), "cn",
                                      master_cns))
        for fld in _MASTER_CO2_ARC_FIELDS:
            if hasattr(new, fld):
                setattr(new, fld,
                        _semi_triples(getattr(data, fld, None),
                                      master_local_keys))
        for fld in _MASTER_CO2_GROUP_FIELDS:
            if hasattr(new, fld):
                setattr(new, fld,
                        _filter_field(getattr(data, fld, None), "g",
                                      master_groups_co2))
        # ``_MASTER_CO2_LOOKUP_FIELDS`` (p_co2_content / p_co2_price):
        # pure lookup tables joined by the kept CO2 rows — carried whole
        # through the shallow copy (rows for un-kept groups are inert,
        # and the PROCESSES feature requires presence).
        for fld in (_GROUP_FEATURE_SET_FIELDS
                    + _GROUP_FEATURE_PARAM_FIELDS):
            if hasattr(new, fld):
                setattr(new, fld,
                        _filter_field(getattr(data, fld, None), "g",
                                      master_groups_feature))
        for fld in _MASTER_GROUP_PROC_FIELDS:
            if hasattr(new, fld):
                obj = getattr(data, fld, None)
                setattr(new, fld,
                        _keep_proc_param(obj) if isinstance(obj, Param)
                        else _keep_proc(obj))

    # ``build_flextool``'s ALWAYS feature requires these four fields to be
    # PRESENT (not None) even when empty (model.py ``ALWAYS``).  Emptying
    # them (head(0)) — rather than nulling — OMITS every terminal node from
    # balance while keeping the build's structural precondition satisfied:
    # the master's ``v_flow`` is then free except for ``maxFlow`` and its
    # own bound (Phase-3 §1.3).
    def _empty_like_frame(df: pl.DataFrame | None) -> pl.DataFrame | None:
        return df.head(0) if df is not None else df

    def _empty_like_param(p: Param | None) -> Param | None:
        if p is None:
            return None
        return Param(p.dims, p.frame.head(0), name=p.name)

    # OMIT every cross-arc TERMINAL node from balance — that is the
    # "unbalanced virtual node" requirement: with the terminals absent the
    # master's trade ``v_flow`` is free except for ``maxFlow`` and its own
    # bound (Phase-3 §1.3).  We do NOT empty ``nodeBalance`` entirely,
    # because ``build_flextool`` requires a non-empty balance-node set to
    # declare the ``vq_state_up/down`` slack vars and the ``nodeBalance_eq``
    # row (a None ``nodeBalance_dt`` crashes ``add_var`` / ``add_cstr``).
    # Instead we keep the NON-terminal nodes: in the reduced master they
    # carry NO arcs (every non-cross arc was dropped) and NO inflow, so
    # their balance collapses to ``slack = 0`` — structurally inert, with
    # zero effect on the trade flow.  This satisfies the build precondition
    # while still omitting exactly the trade terminals.
    terminal_nodes: set[str] = set()
    if not master_mode:
        # Today's exact path: every cross-arc terminal is omitted.
        for k in cross_keys:
            terminal_nodes.add(k[1])
            terminal_nodes.add(k[2])
    else:
        # Omit ONLY the region-side endpoints: both terminals of
        # region↔region cross arcs (as today) plus the region-side
        # endpoint of each region↔master coupling arc.  Master-hosted
        # endpoints stay balanced (that is the whole point of the
        # mode); master-local arc endpoints are all master-side and
        # never omitted.
        for r in cross.iter_rows(named=True):
            terminal_nodes.add(r["source"])
            terminal_nodes.add(r["sink"])
        for r in region_master.iter_rows(named=True):
            terminal_nodes.add(
                r["source"] if r["_src_region"] is not None else r["sink"])

    def _drop_terminal_frame(df: pl.DataFrame | None) -> pl.DataFrame | None:
        if df is None or "n" not in df.columns:
            return df
        return df.filter(~_is_in_keep("n", terminal_nodes))

    def _drop_terminal_param(p: Param | None) -> Param | None:
        if p is None or "n" not in p.dims:
            return p
        return Param(p.dims, p.frame.filter(~_is_in_keep("n", terminal_nodes)),
                     name=p.name)

    new.nodeBalance = _drop_terminal_frame(data.nodeBalance)
    if not master_mode:
        new.p_inflow = _empty_like_param(data.p_inflow)
    else:
        # The master keeps its own inflow; every other balance node in
        # the reduced data stays structurally inert (no arcs, no
        # inflow ⇒ balance collapses to ``slack = 0``, exactly as the
        # legacy all-emptied path).
        new.p_inflow = _filter_param(data.p_inflow, "n",
                                     set(master_hosted_nodes))
    new.p_penalty_up = _drop_terminal_param(data.p_penalty_up)
    new.p_penalty_down = _drop_terminal_param(data.p_penalty_down)

    return new


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def split(
    data: FlexData,
    *,
    regions: list[str] | None = None,
    region_membership: dict[str, dict[str, set[str]]] | None = None,
    benders_uncap_cross_region: bool = False,
    master_hosted_nodes: frozenset[str] = frozenset(),
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
    benders_uncap_cross_region
        Benders mode.  When ``True``, each cross-region virtual half-flow
        is built with an effectively-unbounded ``maxFlow`` capacity (a
        large sentinel) instead of inheriting the original arc's
        ``p_flow_upper_existing``.  In Benders decomposition the TRUE
        capacity limit ``f ≤ C·unitsize`` is enforced in the MASTER, so
        a per-region cap would double-bound the flow and (for greenfield
        cross-region pipes, whose inherited ``existing`` is 0) sever the
        trade arc to zero — the false-convergence bug.  Default ``False``
        preserves today's inherit-from-original behaviour byte-for-byte.
    master_hosted_nodes
        Master-hosted node mode (see
        :func:`compute_master_hosted_nodes`).  With the default empty
        set the split is byte-identical to today's shared-replicate
        behaviour.  Non-empty: the named nodes live in the Benders
        MASTER — they are excluded from the shared-replicate set (no
        region carries them), arcs are classified 4-way
        (:func:`_classify_arcs`), region↔master coupling arcs get
        exactly ONE half-flow on the region side, master-local arcs
        (and the processes ALL of whose arcs are master-local) are
        dropped from every region and NOT half-flowed, and authored
        data that cannot be partitioned (a unit straddling the
        boundary, a user constraint referencing both sides) raises a
        hard error — never a silent degrade.

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

    if master_hosted_nodes:
        overlap = set(master_hosted_nodes) & all_region_nodes
        if overlap:
            raise RuntimeError(
                f"split: master_hosted_nodes overlap region membership: "
                f"{sorted(overlap)} — a node is either master-hosted "
                f"(in no region group) or in exactly one region, never "
                f"both."
            )
        # Master nodes are never shared-replicated: regions must not
        # carry them.
        shared_nodes -= set(master_hosted_nodes)

    # Classify cross-region arcs.
    if data.process_source_sink is None:
        return [
            RegionSplit(region=r, data=data, half_flows=[])
            for r in regions
        ]

    pss_tagged, cross, region_master, master_local = _classify_arcs(
        data.process_source_sink, region_nodes,
        master_nodes=master_hosted_nodes,
    )

    master_local_procs: set[str] = set()
    master_cns: set[str] = set()
    master_groups: set[str] = set()
    if master_hosted_nodes:
        # Hard validation FIRST (D-a): never silently mis-partition
        # authored data.
        _validate_no_straddling_units(
            data, all_region_nodes, master_hosted_nodes)
        master_local_procs = _master_local_procs(
            data.process_source_sink, master_local)
        _validate_user_constraints(
            data, all_region_nodes, master_hosted_nodes,
            master_local_procs)
        # All-master user constraints / feature groups / CO2-cap groups
        # live whole in the master (kept there by
        # ``master_network_data``): regions must drop their rows — a
        # region copy would degenerate (constraint: ``0 sense
        # constant``; feature group: empty membership still charging
        # slack penalty).  The group/CO2 helpers also hard-error on
        # straddling groups (same validation family as the unit /
        # user-constraint checks above).
        master_cns = _master_side_constraint_ids(
            data, all_region_nodes, master_hosted_nodes,
            master_local_procs)
        _ml_triples = {
            (r["p"], r["source"], r["sink"])
            for r in master_local.iter_rows(named=True)
        }
        _rm_triples = {
            (r["p"], r["source"], r["sink"])
            for r in region_master.iter_rows(named=True)
        }
        master_groups = (
            _master_side_feature_groups(
                data, all_region_nodes, master_hosted_nodes)
            | _co2_master_partition(data, _ml_triples, _rm_triples)
        )
        # Master-local procs live wholly in the master (F3): regions
        # carry neither their arcs nor their entity rows.
        shared_procs -= master_local_procs

    half_flows_by_region = _make_half_flows(cross, region_master)

    cross_arcs_by_pss: set[tuple[str, str, str]] = set()
    for r in cross.iter_rows(named=True):
        cross_arcs_by_pss.add((r["p"], r["source"], r["sink"]))
    # Region↔master coupling arcs are replaced by their single-sided
    # half-flow; master-local arcs are dropped outright (the master
    # keeps the whole original arc) — both classes must vanish from
    # every region's process frames.
    for frame in (region_master, master_local):
        for r in frame.iter_rows(named=True):
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
    # Base the widening on the SOURCE's own axis_enums snapshot — that
    # is guaranteed to match the dtypes embedded in ``data``'s frames.
    # The live global ContextVar may have been overwritten by an
    # unrelated ``load_flextool`` call between the lh2 fixture's load
    # and this split (e.g. a sibling test loaded a different DB), in
    # which case widening the live vocabulary would yield an Enum that
    # doesn't contain ``data``'s entity tokens.  Fall back to the live
    # global only when ``data`` lacks its own snapshot.
    _base_enums = (getattr(data, "_axis_enums", None)
                   or get_global_axis_enums())
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
            # Master-local procs are subtracted AFTER the half-flow
            # additions: region membership may name them (e.g. a unit
            # whose every arc moved master-side), but regions must
            # carry no rows for them (F3).  Half-flow originals are
            # never master-local (they have a region-side terminal).
            if master_local_procs:
                keep_procs -= master_local_procs

            rdata = _build_region_data(
                src=data,
                region=r,
                keep_nodes=keep_nodes,
                keep_procs=keep_procs,
                half_flows=half_flows_by_region.get(r, []),
                cross_arcs_by_pss=cross_arcs_by_pss,
                benders_uncap_cross_region=benders_uncap_cross_region,
                master_hosted_nodes=master_hosted_nodes,
                master_local_procs=master_local_procs,
                master_cns=master_cns,
                master_groups=master_groups,
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
