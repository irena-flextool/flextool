"""The arcs of the split: each ``process_source_sink`` arc classified by the regions of its ends, the half-flows of
a cross-region arc, the master-local processes and the units (and their arcs) the Benders master holds whole."""
from __future__ import annotations

import polars as pl

from flextool.engine.preprocess.flex_data import FlexData

from flextool.engine.solve._region_filter._regions import HalfFlow, compute_master_hosted_nodes, load_region_membership


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
    one_var_arcs: "set[tuple[str, str, str]] | frozenset" = frozenset(),
) -> dict[str, list[HalfFlow]]:
    """For each cross-region arc, produce two HalfFlow records (one
    per region).

    For each region↔master arc in *region_master_arcs* (master-hosted
    mode), produce exactly ONE HalfFlow — on the region side: an
    export when the region node is the arc's source, an import when it
    is the sink.  Naming reuses the region-side stem of the paired
    convention (the master side is implicit; no master-side virtual
    entity exists — the master keeps the whole original arc).

    A half-flow of an arc in *one_var_arcs* (``process_source_sink_2way_1var``) gets ``one_var``.
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
            one_var=(p, s, k) in one_var_arcs,
        ))
        out.setdefault(rb, []).append(HalfFlow(
            region=rb, side="import",
            original_p=p, original_source=s, original_sink=k,
            in_region_node=k,
            virtual_node=vi_node,
            virtual_p=vi_conn,
            virtual_arc_source=vi_node,
            virtual_arc_sink=k,
            one_var=(p, s, k) in one_var_arcs,
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
                    one_var=(p, s, k) in one_var_arcs,
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
                    one_var=(p, s, k) in one_var_arcs,
                ))
    return out


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


def _unit_names(data: FlexData) -> set[str]:
    """Every unit of the solve (``process_unit`` ∪ ``process_indirect``; connections are in neither)."""
    out: set[str] = set()
    for fld in ("process_unit", "process_indirect"):
        f = getattr(data, fld, None)
        if f is not None and f.height > 0:
            out |= {str(p) for p in f.get_column("p").to_list()}
    return out


def _side_of_node(n: str, node_region: dict[str, str], master_hosted_nodes) -> str | None:
    """The decomposition side of node *n*: its region, ``"the master"`` (master-hosted) or None (a shared node)."""
    if n in node_region:
        return node_region[n]
    return "the master" if n in master_hosted_nodes else None


def master_units(
    data: FlexData,
    region_nodes: dict[str, set[str]],
    master_hosted_nodes: "frozenset[str] | set[str]" = frozenset(),
) -> frozenset[str]:
    """The units that the Benders master holds whole: those with an arc through the unit itself (an input
    ``(u, n, u)``, an output ``(u, u, n)``) whose nodes and start-up input nodes are on more than one side, two
    regions or a region and the master-hosted nodes, whether a region group lists the unit or not.

    Shared nodes (in no region, not master-hosted: commodity nodes) are on no side.  A unit with only direct flows
    (arcs node to node: one input, one output, a single efficiency) is not one of them: it is cut between regions
    like a connection.  The master keeps every arc of such a unit; each arc at a region node is coupled to a
    half-flow in that region.
    """
    pss = data.process_source_sink
    if pss is None or pss.height == 0:
        return frozenset()
    units = _unit_names(data)
    node_region = {n: r for r, ns in region_nodes.items() for n in ns}
    through: set[str] = set()
    sides: dict[str, set[str]] = {}
    for p, source, sink in pss.select(pl.col("p", "source", "sink").cast(pl.Utf8)).iter_rows():
        if p not in units:
            continue
        if p in (source, sink):
            through.add(p)
        for n in (source, sink):
            side = None if n == p else _side_of_node(n, node_region, master_hosted_nodes)
            if side is not None:
                sides.setdefault(p, set()).add(side)
    startup_pn = getattr(data, "process_startup_node", None)
    if startup_pn is not None and startup_pn.height > 0:
        for p, n in startup_pn.select(pl.col("p").cast(pl.Utf8), pl.col("n").cast(pl.Utf8)).iter_rows():
            side = _side_of_node(n, node_region, master_hosted_nodes)
            if p in units and side is not None:
                sides.setdefault(p, set()).add(side)
    return frozenset(p for p in through if len(sides.get(p, ())) > 1)


def master_arc_counts(
    data: FlexData,
    regions: list[str],
    region_membership: dict[str, dict[str, set[str]]] | None = None,
) -> tuple[int, int, int]:
    """``(connections, unit__inputNode arcs, unit__outputNode arcs)`` in the Benders master: the connections with an
    arc in the master, and the input / output node relationships of the units with an arc there (an arc
    ``(u, n, u)`` is an input, ``(u, u, n)`` an output, a unit's direct arc ``(u, a, b)`` one of each).  The master
    holds the arcs between regions, the arcs at its hosted nodes and every arc of :func:`master_units`."""
    pss = data.process_source_sink
    if pss is None or pss.height == 0:
        return 0, 0, 0
    if region_membership is None:
        region_membership = load_region_membership(data, regions)
    region_nodes = {r: m["nodes"] for r, m in region_membership.items()}
    hosted = frozenset(compute_master_hosted_nodes(data, region_membership))
    _, cross, region_master, master_local = _classify_arcs(
        pss, region_nodes, master_nodes=hosted | master_units(data, region_nodes, hosted))
    units = _unit_names(data)
    connections: set[str] = set()
    inputs: set[tuple[str, str]] = set()
    outputs: set[tuple[str, str]] = set()
    for frame in (cross, region_master, master_local):
        for p, source, sink in frame.select(pl.col("p", "source", "sink").cast(pl.Utf8)).iter_rows():
            if p not in units:
                connections.add(p)
                continue
            if sink != p:
                outputs.add((p, sink))
            if source != p:
                inputs.add((p, source))
    return len(connections), len(inputs), len(outputs)
