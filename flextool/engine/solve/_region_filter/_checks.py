"""The checks of a split: units listed in the wrong region, terms of master units, start-up nodes, user
constraints and feature groups across the boundary, severed connections that the half-flows cannot carry, and
the warning on integer columns in a region subproblem."""
from __future__ import annotations

import polars as pl

from flextool.engine.preprocess.flex_data import FlexData
from flextool.engine.model import _dc_power_flow
from flextool.engine.input._solve_state import FlexToolConfigError

from flextool.engine.solve._region_filter._regions import _logger, process_region_by_nodes
from flextool.engine.solve._region_filter._master_side import _limit_groups, _user_constraint_sides


# ---------------------------------------------------------------------------
# Master-hosted-node validation + region scrubbing
# ---------------------------------------------------------------------------


def _validate_listed_units(pss: pl.DataFrame | None, region_nodes: dict[str, set[str]],
                           region_procs: dict[str, set[str]], in_master: "frozenset[str] | set[str]",
                           units: set[str]) -> None:
    """Raise when a region group lists a unit none of whose nodes is in that region (all in other regions): the
    split would keep the unit in the listing region, where its nodes have no balance (its flows would be lost)."""
    if pss is None or pss.height == 0:
        return
    node_region = {n: r for r, ns in region_nodes.items() for n in ns}
    regions_of: dict[str, set[str]] = {}
    for p, source, sink in pss.select(pl.col("p", "source", "sink").cast(pl.Utf8)).iter_rows():
        for n in (source, sink):
            if n in node_region:
                regions_of.setdefault(p, set()).add(node_region[n])
    bad = [f"{p} (listed in {r}, its nodes in {', '.join(sorted(regions_of[p]))})"
           for r, ps in sorted(region_procs.items()) for p in sorted(ps)
           if p in units and p not in in_master and p in regions_of and r not in regions_of[p]]
    if bad:
        raise FlexToolConfigError(
            "Benders decomposition: unit(s) listed in a region group that has none of their nodes: "
            + "; ".join(bad) + ".  A unit belongs to the region of its nodes: list it in that region's group "
            "(group__unit), or remove it from the group (listing units is optional).")


def _validate_master_units(data: FlexData, units: "frozenset[str] | set[str]", region_nodes: dict[str, set[str]],
                           ) -> None:
    """Raise when something that a region subproblem holds reads a term of a master unit other than its flows
    (which the region sees as half-flows): a reserve, a start-up draw from a region node, a capacity margin /
    inertia / non-synchronous group with one of the unit's region nodes, a flowGroup flow limit
    (``_cross_connection_flow_references``), or integer online variables (the master would become a mixed-integer
    problem).  A min-load section (min_load_efficiency) is no such term: it enters a node balance only on the
    efficiency arc of a direct unit (``flow_from_nodeBalance_eff``), and a master unit's arcs go through the unit
    (``noEff``; the section is in its conversion row, in the master)."""
    if not units:
        return
    from flextool.engine.model import _group_slack

    node_region = {n: r for r, ns in region_nodes.items() for n in ns}
    region_nodes_of: dict[str, set[str]] = {}
    pss = data.process_source_sink
    for p, source, sink in pss.select(pl.col("p", "source", "sink").cast(pl.Utf8)).iter_rows():
        if p in units:
            region_nodes_of.setdefault(p, set()).update(n for n in (source, sink) if n in node_region)
    problems: dict[str, list[str]] = {}

    def _names(frame, col: str = "p") -> set[str]:
        return (set() if frame is None or frame.height == 0
                else {str(x) for x in frame.get_column(col).cast(pl.Utf8).to_list()})

    for p in sorted(units & _names(getattr(data, "process_reserve_upDown_node_active", None))):
        problems.setdefault(p, []).append("a reserve: provide it from units of the reserve's region")
    startup_pn = getattr(data, "process_startup_node", None)
    if startup_pn is not None and startup_pn.height > 0:
        for p, n in startup_pn.select(pl.col("p").cast(pl.Utf8), pl.col("n").cast(pl.Utf8)).iter_rows():
            if p in units and n in node_region:
                problems.setdefault(p, []).append(
                    f"start-up energy from node {n!r} of {node_region[n]}: take it from a commodity node")
    if not getattr(data, "relax_integrality", False):
        for p in sorted(units & _names(getattr(data, "process_online_integer", None))):
            problems.setdefault(p, []).append(
                "integer online variables: use startup_method linear, or reduce_unit_detail_method integers")
    gn = data.group_node
    if gn is not None and gn.height > 0:
        members: dict[str, set[str]] = {}
        for g, n in gn.select(pl.col("g").cast(pl.Utf8), pl.col("n").cast(pl.Utf8)).iter_rows():
            members.setdefault(g, set()).add(n)
        for what, set_name, active in (("capacity margin", "groupCapacityMargin", _group_slack.has_capacity_margin),
                                       ("inertia", "groupInertia", _group_slack.has_inertia),
                                       ("non-synchronous", "groupNonSync", _group_slack.has_non_sync)):
            groups = getattr(data, set_name, None)
            if groups is None or groups.height == 0 or not active(data):
                continue
            for g in sorted(_names(groups, "g")):
                for p in sorted(units):
                    if region_nodes_of.get(p, set()) & members.get(g, set()):
                        problems.setdefault(p, []).append(
                            f"{what} group {g!r} at one of its nodes: take the unit's nodes out of the group")
    for ref in (r for r in _cross_connection_flow_references(data, set(units)) if r.startswith("flowGroup")):
        p = ref.rsplit("'", 2)[-2]
        problems.setdefault(p, []).append(ref.replace("connection", "unit") + ": leave the unit out of it")
    if problems:
        raise FlexToolConfigError(
            "Benders decomposition: unit(s) with nodes in more than one region (or in a region and the master) are "
            "solved in the Benders master, and the region subproblems see only their flows, so these cannot be "
            "split: " + "; ".join(f"{p} ({', '.join(why)})" for p, why in sorted(problems.items()))
            + ".  Or put all nodes of the unit in one region.")


def _validate_startup_nodes(data: FlexData, region_nodes: dict[str, set[str]],
                            region_procs: dict[str, set[str]],
                            master_hosted_nodes: "frozenset[str] | set[str]" = frozenset(),
                            in_master: "frozenset[str] | set[str]" = frozenset()) -> None:
    """Hard-error on a unit that takes start-up energy (``process_startup_node``, startup_input_method) from a node
    of a region the unit is not in, or from a master-hosted node: the draw is a term on the unit's units started,
    which live in the unit's own region subproblem, so it cannot enter another region's or the master's balance.
    A unit the master holds whole (*in_master*) has its start-ups in the master (:func:`_validate_master_units`
    refuses its draws from region nodes)."""
    pn = getattr(data, "process_startup_node", None)
    if pn is None or pn.height == 0:
        return
    bad: list[str] = []
    all_region_procs = {q for ps in region_procs.values() for q in ps}
    for p, n in pn.select(pl.col("p").cast(pl.Utf8), pl.col("n").cast(pl.Utf8)).iter_rows():
        if p in in_master:
            continue
        who = f"unit {p!r}" if p in all_region_procs else f"unit {p!r} of no region"
        if n in master_hosted_nodes:
            bad.append(f"{who} takes start-up energy from node {n!r} of the Benders master")
        for r, nodes in region_nodes.items():
            if n in nodes and p not in region_procs.get(r, set()):
                bad.append(f"{who} takes start-up energy from node {n!r} of region {r!r}")
    if bad:
        raise RuntimeError(
            "split: " + "; ".join(sorted(bad)) + ".  A start-up draw is a term on the unit's start-ups, which are "
            "in the unit's own region subproblem (a unit of no region is copied into every region).  Take the "
            "start-up input from a node of the unit's region or a node of no region (a commodity node), take it "
            "through a connection into a node of the unit's region, or bring the node into the unit's region.")


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


#: The side of an entity the Benders master holds (:func:`_validate_feature_groups_in_one_region`).
_MASTER_SIDE = "the Benders master"


def _validate_feature_groups_in_one_region(
    data: FlexData,
    region_nodes: dict[str, set[str]],
    region_procs: "dict[str, set[str]] | None" = None,
    master_hosted_nodes: "frozenset[str] | set[str]" = frozenset(),
    master_procs: "frozenset[str] | set[str]" = frozenset(),
) -> None:
    """A capacity-margin, inertia or non-synchronous group whose nodes lie in more than one region is an error,
    and so is an investment, retirement or cumulative-capacity limit group whose entities do.

    Each region subproblem would get its own copy of the group's row over the group's nodes in that region, so
    the requirement (the whole margin, inertia limit or share) would apply once per region instead of once for the
    group.  Only the families active in the solve are checked (``_group_slack.has_*``: a capacity margin needs
    investment periods; ``reduce_system_detail_method`` may leave inertia and non-synchronous out).  Nodes in no
    region (shared, replicated into every region) do not count.  A limit group (:func:`_limit_groups`: a limit
    value set, not only the method)
    sums over its member entities (``group_entity``): each side would bound its own members' share by the whole
    limit.  An entity's side is where its investment variable is: the master for a master-hosted node and for a
    process with an arc in the master (*master_procs*: connections between regions or at master-hosted nodes, the
    units the master holds whole, the master-local processes), else the region that lists it, else the region of its
    nodes (:func:`process_region_by_nodes`, as the investment owner of an unlisted process).
    """
    from flextool.engine.model import _group_slack

    families = (("capacity margin", "groupCapacityMargin", _group_slack.has_capacity_margin),
                ("inertia", "groupInertia", _group_slack.has_inertia),
                ("non-synchronous", "groupNonSync", _group_slack.has_non_sync))
    region_of: dict[str, str] = {}
    for r, ns in region_nodes.items():
        for n in ns:
            region_of[n] = r
    gn = data.group_node
    members: dict[str, set[str]] = {}
    if gn is not None and gn.height > 0:
        for g, n in gn.select(pl.col("g").cast(pl.Utf8), pl.col("n").cast(pl.Utf8)).iter_rows():
            members.setdefault(g, set()).add(n)
    errors: list[str] = []
    for what, set_name, active in families:
        if not members:
            break
        groups = getattr(data, set_name, None)
        if groups is None or groups.height == 0 or not active(data):
            continue
        for g in sorted({str(x) for x in groups.get_column("g").to_list()}):
            regions = sorted({region_of[n] for n in members.get(g, ()) if n in region_of})
            if len(regions) > 1:
                errors.append(f"{what} group {g!r} has nodes in regions {', '.join(regions)}")
    limit_groups = _limit_groups(data)
    max_errors: list[str] = []
    ge = data.group_entity
    if limit_groups and ge is not None and ge.height > 0:
        entity_side = process_region_by_nodes(data.process_source_sink, region_nodes)
        entity_side.update(region_of)
        for r, ps in (region_procs or {}).items():
            for p in ps:
                entity_side[p] = r
        for e in set(master_hosted_nodes) | set(master_procs):
            entity_side[e] = _MASTER_SIDE
        entity_members: dict[str, set[str]] = {}
        for g, e in ge.select(pl.col("g").cast(pl.Utf8), pl.col("e").cast(pl.Utf8)).iter_rows():
            entity_members.setdefault(g, set()).add(e)
        for g in sorted(limit_groups):
            by_side: dict[str, list[str]] = {}
            for e in sorted(entity_members.get(g, ())):
                if e in entity_side:
                    by_side.setdefault(entity_side[e], []).append(e)
            if len(by_side) > 1:
                sides = sorted(by_side, key=lambda x: (x == _MASTER_SIDE, x))
                text = (f"has entities in regions {', '.join(x for x in sides if x != _MASTER_SIDE)}"
                        + (f" and {_MASTER_SIDE}" if _MASTER_SIDE in by_side else "")
                        + " (" + "; ".join(f"{x}: {', '.join(by_side[x])}" for x in sides) + ")")
                if limit_groups[g] == "min":
                    errors.append(f"investment limit group {g!r} {text}")
                else:
                    max_errors.append(f"investment maximum group {g!r} {text}")
    if errors or max_errors:
        why = []
        if errors:
            why.append("each region would require all of a group's requirement")
        if max_errors:
            why.append("each region could use all of a group's maximum")
        raise RuntimeError(
            "split: " + "; ".join(errors + max_errors) + ". A group's limit cannot be split between Benders "
            f"region subproblems and the master ({'; '.join(why)}). Use one group per region, and one for the "
            "connections between regions and the units with nodes in more than one region (the master's).")


def _validate_cross_region_connections(data: FlexData, cross: pl.DataFrame,
                                       region_master: pl.DataFrame,
                                       in_master: "frozenset[str] | set[str]" = frozenset()) -> None:
    """Raise when a connection severed by the split has online variables, the exact transfer method or reserve.

    A severed connection is replaced by half-flows (other processes) in the regions, so its own rows are gone:
    its online status and direction switch would be free columns without flow rows, and its reserve (limited by
    its flows, connection_reserve_*) would have no channel left.  None of them can be split between two region
    subproblems.  Nor can a DC power-flow line: its dc_flow_eq joins the voltage angles of both end nodes.
    The units the master holds whole (*in_master*, :func:`master_units`) keep their online variables there and
    are checked by :func:`_validate_master_units`.
    """
    procs = {str(p) for frame in (cross, region_master) if frame is not None and frame.height > 0
             for p in frame.get_column("p").to_list()} - set(in_master)
    if not procs:
        return
    # DC power-flow lines: dc_flow_eq joins the voltage angles of both end nodes, which half-flows cannot carry.
    dc_arcs = _dc_power_flow.dc_arcs_frame(data) if _dc_power_flow.has_feature(data) else None
    dc_hit = sorted(procs & ({str(p) for p in dc_arcs.get_column("p").to_list()} if dc_arcs is not None else set()))
    if dc_hit:
        raise FlexToolConfigError(
            "Benders decomposition: DC power-flow line(s) between regions (or between a region and the master): "
            + ", ".join(dc_hit) + ".  The split cannot carry the voltage angles across the cut.  Put both nodes "
            "of the line in one region, or take the nodes out of the dc_power_flow_with_angles group.")
    checks = (("a startup_method (online variables)",
               [getattr(data, "process_online_linear", None), getattr(data, "process_online_integer", None)]),
              ("the exact transfer method", [getattr(data, "process_exact", None)]),
              ("reserve", [getattr(data, "process_reserve_upDown_node_active", None)]))
    problems = []
    for what, frames in checks:
        hit = sorted(procs & {str(p) for f in frames if f is not None and f.height > 0
                              for p in f.get_column("p").to_list()})
        if hit:
            problems.append(f"{', '.join(hit)} ({what})")
    if problems:
        raise FlexToolConfigError(
            "Benders decomposition: connection(s), or unit(s) with one input and one output, between regions (or "
            "between a region and the master) cannot have online variables, the exact transfer method or reserve, "
            "because the split replaces them with half-flows in each region: " + "; ".join(problems) + ".  Remove "
            "these from the connections or units, put both of their nodes in one region, or leave them out of the "
            "solve (reduce_system_detail_method "
            "reserves, reduce_unit_detail_method integers for exact, online for online variables).")
    flow_refs = _cross_connection_flow_references(data, procs)
    if flow_refs:
        raise FlexToolConfigError(
            "Benders decomposition: the flow of a connection between regions (or between a region and the "
            "master) cannot be in a flow limit or a user constraint, because the split replaces the connection "
            "with half-flows in each region, which neither sees: " + "; ".join(flow_refs) + ".  Put both nodes "
            "of the connection in one region, or leave the connection out of the flowGroup limit or the user "
            "constraint.")


#: The flowGroup flow limits (``_cumulative_invest``: maxCumulative_flow_*, maxInstant_flow, ...): Params ``(g, ...)``.
_FLOW_LIMIT_PARAM_FIELDS: tuple[str, ...] = (
    "p_group_max_cumulative_flow", "p_group_min_cumulative_flow",
    "pd_max_cumulative_flow", "pd_min_cumulative_flow",
    "pdt_max_instant_flow", "pdt_min_instant_flow",
)


def _cross_connection_flow_references(data: FlexData, procs: set[str]) -> list[str]:
    """What sees the flow of the severed connections *procs*: ``"flowGroup g (connection p)"`` for a flow-limited
    group with the connection at only one of its member nodes (``group_process_node``; with both ends in the
    group the connection's flow cancels and the limit does not depend on it), ``"user constraint cn (connection
    p)"`` for a ``constraint_flow_coeff`` on it (``flow_constraint_idx``)."""
    out: list[str] = []
    gpn = getattr(data, "group_process_node", None)
    limited: set[str] = set()
    for fld in _FLOW_LIMIT_PARAM_FIELDS:
        frame = getattr(getattr(data, fld, None), "frame", None)
        if frame is not None and frame.height > 0 and "g" in frame.columns:
            limited |= {str(g) for g in frame.get_column("g").cast(pl.Utf8).to_list()}
    if limited and gpn is not None and gpn.height > 0:
        ends: dict[str, set[str]] = {}
        pss = data.process_source_sink
        if pss is not None:
            for p, src, snk in pss.select(pl.col("p").cast(pl.Utf8), pl.col("source").cast(pl.Utf8),
                                          pl.col("sink").cast(pl.Utf8)).iter_rows():
                if p in procs:
                    ends.setdefault(p, set()).update((src, snk))
        members: dict[tuple[str, str], set[str]] = {}
        for g, p, n in gpn.select(pl.col("g").cast(pl.Utf8), pl.col("p").cast(pl.Utf8),
                                  pl.col("n").cast(pl.Utf8)).iter_rows():
            if g in limited and p in procs:
                members.setdefault((g, p), set()).add(n)
        for (g, p), nodes in sorted(members.items()):
            if not ends.get(p, set()) <= nodes:
                out.append(f"flowGroup {g!r} (connection {p!r})")
    fci = getattr(data, "flow_constraint_idx", None)
    if fci is not None and fci.height > 0 and "cn" in fci.columns:
        hits = {(str(cn), str(p)) for p, cn in fci.select(pl.col("p").cast(pl.Utf8),
                                                            pl.col("cn").cast(pl.Utf8)).iter_rows() if p in procs}
        out += [f"user constraint {cn!r} (connection {p!r})" for cn, p in sorted(hits)]
    return out


def _warn_integer_region(region: str, rdata: FlexData) -> None:
    """Warn when a region subproblem has integer columns (whole-unit online status, exact direction switch): the
    Benders cuts use the subproblem's duals, which are not valid for a mixed-integer subproblem.  Quiet when the
    solve's ``reduce_unit_detail_method`` relaxes integers (``relax_integrality``: the integer online columns are
    continuous, and ``process_exact`` is already cleared)."""
    pss = getattr(rdata, "process_source_sink", None)
    in_region = ({str(p) for p in pss.get_column("p").to_list()} if pss is not None and pss.height > 0 else set())
    integer_sets = [getattr(rdata, "process_exact", None)]
    if not getattr(rdata, "relax_integrality", False):
        integer_sets.append(getattr(rdata, "process_online_integer", None))
    names = sorted(in_region & {str(p) for f in integer_sets
                                if f is not None and f.height > 0 for p in f.get_column("p").to_list()})
    if names:
        msg = (f"Benders: region {region} has integer variables (startup_method integer or integer_fix_in_uc, "
               f"or exact transfer method: {', '.join(names)}). Benders cuts come from the duals of the region "
               "subproblem, which are not valid for a mixed-integer subproblem; the decomposition may stop at a "
               "wrong solution. Set reduce_unit_detail_method = integers (or a level that contains it) on the solve "
               "to solve the regions as LPs, or use startup_method linear or linear_fix_in_uc and a non-exact "
               "transfer method in decomposed regions.")
        _logger.warning(f"Warning: {msg}")   # the console handler prints it (once; no extra print)
