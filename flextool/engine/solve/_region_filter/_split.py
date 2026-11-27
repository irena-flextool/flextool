"""The entry point :func:`split`: a whole-system :class:`FlexData` into per-region :class:`RegionSplit` records."""
from __future__ import annotations

import polars as pl

from flextool.engine.preprocess.flex_data import FlexData
from flextool.engine.input._axis_enums import get_global_axis_enums, reset_global_axis_enums, set_global_axis_enums

from flextool.engine.solve._region_filter._regions import RegionSplit, load_region_membership
from flextool.engine.solve._region_filter._arcs import (
    _classify_arcs,
    _make_half_flows,
    _master_local_procs,
    _unit_names,
    master_units,
)
from flextool.engine.solve._region_filter._master_side import (
    _co2_master_partition,
    _master_side_constraint_ids,
    _master_side_feature_groups,
)
from flextool.engine.solve._region_filter._checks import (
    _validate_cross_region_connections,
    _validate_feature_groups_in_one_region,
    _validate_listed_units,
    _validate_master_units,
    _validate_startup_nodes,
    _validate_user_constraints,
    _warn_integer_region,
)
from flextool.engine.solve._region_filter._region_data import _build_region_data


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
        data that cannot be partitioned (a user constraint referencing
        both sides) raises a hard error — never a silent degrade.

        Whatever this set, a unit whose nodes are on more than one side
        (two regions, or a region and the master-hosted nodes:
        :func:`master_units`) is held whole by the master: its arcs at
        region nodes are region↔master couplings, its other arcs
        master-local, and no region carries its rows.

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
    in_master = master_units(data, region_nodes, master_hosted_nodes)
    _validate_listed_units(data.process_source_sink, region_nodes, region_procs, in_master, _unit_names(data))
    _validate_master_units(data, in_master, region_nodes)
    _validate_startup_nodes(data, region_nodes, region_procs, master_hosted_nodes, in_master)
    # The master's side of the arc classification: its hosted nodes and the units it holds whole (an arc of such a
    # unit at a region node is a region↔master coupling, any other of its arcs master-local).
    master_side = frozenset(master_hosted_nodes) | in_master

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
        master_nodes=master_side,
    )

    master_local_procs: set[str] = set()
    master_cns: set[str] = set()
    master_groups: set[str] = set()
    if master_side:
        master_local_procs = _master_local_procs(
            data.process_source_sink, master_local) | set(in_master)
        _validate_user_constraints(
            data, all_region_nodes, master_hosted_nodes,
            master_local_procs)
        # All-master user constraints / feature groups / CO2-cap groups
        # live whole in the master (kept there by
        # ``master_network_data``): regions must drop their rows — a
        # region copy would degenerate (constraint: ``0 sense
        # constant``; feature group: empty membership still charging
        # slack penalty).  The group/CO2 helpers also hard-error on
        # straddling groups (same validation family as the
        # user-constraint check above).
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
            | _co2_master_partition(data, _ml_triples, _rm_triples,
                                    master_local_procs)
        )
        # Master-local procs live wholly in the master (F3): regions
        # carry neither their arcs nor their entity rows.
        shared_procs -= master_local_procs

    # The master's processes: the arcs between regions and at the master (their investment is the master's).
    master_procs = ({str(p) for frame in (cross, region_master) for p in frame.get_column("p").to_list()}
                    | master_local_procs)
    _validate_feature_groups_in_one_region(data, region_nodes, region_procs, master_hosted_nodes, master_procs)
    _validate_cross_region_connections(data, cross, region_master, in_master)

    _two_way = data.process_source_sink_2way_1var
    one_var_arcs: set[tuple[str, str, str]] = (
        {(str(p), str(src), str(snk)) for p, src, snk in _two_way.select("p", "source", "sink").iter_rows()}
        if _two_way is not None else set())
    half_flows_by_region = _make_half_flows(cross, region_master, one_var_arcs)

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
            # SORTED, never in set order: the Enum category order is the
            # sort key of every LP domain frame (add_var / add_cstr sort
            # before assigning col / row ids), so a set-ordered append made
            # the region subproblem's column / row order depend on the
            # process' PYTHONHASHSEED — a different simplex path, last-bit
            # different cut coefficients and, on a marginal master, a
            # kUnknown in one process but not the next (#153).  The widened
            # vocabulary is re-sorted as a whole (axis vocabularies are
            # lexically sorted, so the virtual tokens sort in with the rest
            # exactly as the Utf8 labels would).
            _add = [t for t in _new_toks if t not in _existing_set]
            if _add:
                _ext[_axis_name] = pl.Enum(sorted(_existing_set | set(_add)))
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
            _warn_integer_region(r, rdata)
            splits.append(RegionSplit(
                region=r,
                data=rdata,
                half_flows=half_flows_by_region.get(r, []),
            ))
        return splits
    finally:
        if _enums_token is not None:
            reset_global_axis_enums(_enums_token)
