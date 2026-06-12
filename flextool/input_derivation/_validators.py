"""SpineDB → Provider pipeline validators.

Each validator consumes a :class:`spinedb_api.DatabaseMapping` (or a
:class:`SpineDBBackend` exposing one as ``._db``) and either:

* raises :class:`FlexToolConfigError` for hard misconfigurations that
  would silently corrupt the solve (e.g. zero ``step_duration``), or
* logs a warning for soft inconsistencies (e.g. ``output_*: yes`` set
  on a group with no members of the required class).

The validators live as their own module because:

1. They are pure code paths with no derivation outputs — they don't
   ``provider.put`` anything.
2. They are called from :func:`flextool.input_derivation.run` early
   enough that a hard failure surfaces before the cascade builds any
   frames against malformed input.
3. They share the :func:`_get_commodity_price_methods` helper but are
   otherwise unrelated to the EAV → tabular materialiser specs in
   :mod:`flextool.input_derivation._specs`.

Pre-Step-2.5 these functions lived in the 2356-LOC ``input_writer.py``
monolith; Step 2.5 item 6 extracted them here.
"""
from __future__ import annotations

import logging

from flextool.engine_polars._solve_state import FlexToolConfigError


__all__ = [
    "validate_timeline_timestep_duration",
    "validate_capacity_margin_groups",
    "validate_ladder_methods",
    "validate_group_output_memberships",
    "validate_connection_node_memberships",
]


def _get_commodity_price_methods(db) -> dict[str, str]:
    """Return ``{commodity: price_method}`` for every commodity whose
    ``price_method`` is set.  Commodities without the param default to
    ``'price'`` in the mod (and do not appear here).
    """
    out: dict[str, str] = {}
    for pv in db.find_parameter_values(
        entity_class_name="commodity",
        parameter_definition_name="price_method",
    ):
        if pv["type"] is None:
            continue
        out[pv["entity_byname"][0]] = str(pv["parsed_value"])
    return out


def validate_timeline_timestep_duration(db) -> None:
    """Raise FlexToolConfigError if any timeline entity is missing its
    ``timestep_duration`` map.  Without it, ``step_duration`` silently
    falls to 0 throughout the model and every time-weighted quantity
    (balances, costs, ramps) collapses to zero.  There is no sensible
    default, so the value must be present on every timeline.
    """
    timelines = [ent["entity_byname"][0]
                 for ent in db.find_entities(entity_class_name="timeline")]
    if not timelines:
        return
    have_duration: set[str] = set()
    for pv in db.find_parameter_values(
        entity_class_name="timeline",
        parameter_definition_name="timestep_duration",
    ):
        if pv["type"] is None:
            continue
        have_duration.add(pv["entity_byname"][0])
    missing = [t for t in timelines if t not in have_duration]
    if missing:
        raise FlexToolConfigError(
            "timeline 'timestep_duration' is not set for: "
            + ", ".join(sorted(missing))
            + ".  Every timeline needs a Map(timestep -> duration_in_hours); "
              "without it all time-weighted quantities collapse to zero."
        )


def validate_capacity_margin_groups(db, logger: logging.Logger) -> None:
    """Storage nodes are excluded from the capacity-margin constraint.
    Raise if any has_capacity_margin group contains *only* storage
    nodes (constraint would have no valid members); warn if a mix is
    present.
    """
    capacity_margin_groups: dict[str, list[str]] = {}
    for pv in db.find_parameter_values(
        entity_class_name="group", parameter_definition_name="has_capacity_margin",
    ):
        if pv["parsed_value"] == "yes":
            capacity_margin_groups[pv["entity_byname"][0]] = []
    if not capacity_margin_groups:
        return
    for ent in db.find_entities(entity_class_name="group__node"):
        g, n = ent["entity_byname"][0], ent["entity_byname"][1]
        if g in capacity_margin_groups:
            capacity_margin_groups[g].append(n)
    storage_nodes: set[str] = set()
    for pv in db.find_parameter_values(
        entity_class_name="node", parameter_definition_name="node_type",
    ):
        if pv["parsed_value"] == "storage":
            storage_nodes.add(pv["entity_byname"][0])
    for g, nodes in capacity_margin_groups.items():
        storage_in_group = [n for n in nodes if n in storage_nodes]
        if storage_in_group and len(storage_in_group) == len(nodes):
            raise FlexToolConfigError(
                f"Capacity margin group '{g}' contains only storage nodes "
                f"({', '.join(storage_in_group)}). The capacity margin constraint "
                f"excludes storage nodes, so this group has no valid nodes."
            )
        elif storage_in_group:
            logger.warning(
                "Capacity margin group '%s' contains storage nodes (%s) which will "
                "be excluded from the capacity margin constraint.",
                g, ', '.join(storage_in_group),
            )


def validate_ladder_methods(db, logger: logging.Logger) -> None:
    """Raise FlexToolConfigError if any commodity declares a ladder
    ``price_method`` but does not have the corresponding ladder parameter
    set.  Runs before the ladder writers so errors name the offending
    commodity and expected parameter.
    """
    methods = _get_commodity_price_methods(db)
    ladder_methods = {"price_ladder_annual", "price_ladder_cumulative"}
    commodities_needing_ladder = {
        c: m for c, m in methods.items() if m in ladder_methods
    }
    if not commodities_needing_ladder:
        return

    # Collect commodities that HAVE each ladder param (non-None, non-empty).
    have_cumulative: set[str] = set()
    have_annual: set[str] = set()
    for pv in db.find_parameter_values(
        entity_class_name="commodity",
        parameter_definition_name="price_ladder_cumulative",
    ):
        if pv["type"] is None:
            continue
        have_cumulative.add(pv["entity_byname"][0])
    for pv in db.find_parameter_values(
        entity_class_name="commodity",
        parameter_definition_name="price_ladder_annual",
    ):
        if pv["type"] is None:
            continue
        have_annual.add(pv["entity_byname"][0])

    for commodity, method in commodities_needing_ladder.items():
        expected_param = method  # parameter name matches method name
        if method == "price_ladder_cumulative" and commodity not in have_cumulative:
            raise FlexToolConfigError(
                f"commodity '{commodity}' has "
                f"price_method='price_ladder_cumulative' but no "
                f"'{expected_param}' value is set.  Add a "
                f"Map(tier -> {{price, quantity}}) on that parameter."
            )
        if method == "price_ladder_annual" and commodity not in have_annual:
            raise FlexToolConfigError(
                f"commodity '{commodity}' has "
                f"price_method='price_ladder_annual' but no "
                f"'{expected_param}' value is set.  Add either a 2d "
                f"Map(tier -> {{price, quantity}}) or a 3d "
                f"Map(period -> Map(tier -> {{price, quantity}}))."
            )


def validate_group_output_memberships(db, logger: logging.Logger) -> None:
    """Warn when a group-level output flag is set but the group lacks
    the membership class required for that output to produce any data.

    Three silent-no-op cases are detected (v58 vocabulary):

    * ``group.print_dispatch: yes`` with no ``group__node`` row
    * ``group.print_indicators: yes`` with no ``group__node`` row
    * ``flowGroup.flow_aggregator`` set to a non-``none`` method with no
      ``flowGroup__unit__node`` **or** ``flowGroup__connection__node`` row

    Additionally, a double-counting guard warns when a single flow arc
    ``(process, node)`` belongs to two or more *dispatch-bound* flowGroups
    (``flow_aggregator`` in ``{dispatch_plots_only, both}``): the arc would
    be summed once per such flowGroup into the nodeGroup dispatch table.

    Only warnings are emitted — a user may deliberately stage a partial
    configuration.
    """
    # Collect groups that are members of the relevant entity classes.
    groups_with_node_members: set[str] = set()
    for ent in db.find_entities(entity_class_name="group__node"):
        byname = ent["entity_byname"]
        if byname:
            groups_with_node_members.add(byname[0])

    flowgroups_with_flow_members: set[str] = set()
    for cls in ("flowGroup__unit__node", "flowGroup__connection__node"):
        for ent in db.find_entities(entity_class_name=cls):
            byname = ent["entity_byname"]
            if byname:
                flowgroups_with_flow_members.add(byname[0])

    # nodeGroup output flags: "yes" on the ``group`` class, need group__node.
    node_checks: list[tuple[str, str, set[str]]] = [
        ("print_dispatch", "group__node", groups_with_node_members),
        ("print_indicators", "group__node", groups_with_node_members),
    ]
    for param_name, required_members, membership_set in node_checks:
        for pv in db.find_parameter_values(
            entity_class_name="group", parameter_definition_name=param_name
        ):
            if pv["type"] is None:
                continue
            if pv["parsed_value"] != "yes":
                continue
            group_name = pv["entity_byname"][0]
            if group_name not in membership_set:
                logger.warning(
                    "Group '%s' has %s: yes but no %s members — output will be empty.",
                    group_name, param_name, required_members,
                )

    # flow_aggregator: an enum METHOD on the ``flowGroup`` class.  Any value
    # other than ``none`` requests dispatch bands and/or standalone
    # indicators, both of which need flowGroup__*__node members to produce
    # any data.
    #
    # We also record which flowGroups request dispatch *bands* — the values
    # ``dispatch_plots_only`` and ``both`` — for the overlap check below.
    _DISPATCH_BOUND = {"dispatch_plots_only", "both"}
    dispatch_bound_flowgroups: set[str] = set()
    for pv in db.find_parameter_values(
        entity_class_name="flowGroup", parameter_definition_name="flow_aggregator"
    ):
        if pv["type"] is None:
            continue
        if pv["parsed_value"] in (None, "none"):
            continue
        group_name = pv["entity_byname"][0]
        if pv["parsed_value"] in _DISPATCH_BOUND:
            dispatch_bound_flowgroups.add(group_name)
        if group_name not in flowgroups_with_flow_members:
            logger.warning(
                "flowGroup '%s' has flow_aggregator: %s but no "
                "flowGroup__unit__node or flowGroup__connection__node members "
                "— output will be empty.",
                group_name, pv["parsed_value"],
            )

    # Double-counting guard: a single flow arc ``(process, node)`` that
    # belongs to two (or more) dispatch-bound flowGroups (flow_aggregator in
    # {dispatch_plots_only, both}) contributes its flow as a band in the
    # nodeGroup dispatch table once per such flowGroup, so the dispatch
    # balance double-counts that arc.  Standalone-only overlap is legitimate
    # (separate aggregator series), so it is excluded here.  Only a warning
    # is emitted — the user may have intended overlapping membership.
    pn_to_aggregators: dict[tuple[str, str], list[str]] = {}
    for cls in ("flowGroup__unit__node", "flowGroup__connection__node"):
        for ent in db.find_entities(entity_class_name=cls):
            byname = ent["entity_byname"]
            if not byname or len(byname) < 3:
                continue
            flowgroup, process, node = byname[0], byname[1], byname[2]
            if flowgroup not in dispatch_bound_flowgroups:
                continue
            pn_to_aggregators.setdefault((process, node), []).append(flowgroup)
    for (process, node), flowgroups in pn_to_aggregators.items():
        if len(flowgroups) >= 2:
            logger.warning(
                "Flow (%s, %s) belongs to multiple flowGroups (%s) in effect "
                "double counting the flow in dispatch plots. Check "
                "flow_aggregator parameter.",
                process, node, ", ".join(sorted(flowgroups)),
            )


def validate_connection_node_memberships(db, logger: logging.Logger) -> None:
    """Warn when a ``connection`` entity has no ``connection__node__node``
    relationship defining its two endpoints.

    A connection's purpose is to transfer between two nodes; without a
    ``connection__node__node`` member it carries no source/sink rows, never
    enters ``process_source_sink``, and contributes nothing to any node
    balance.  If it nonetheless has an ``invest_method`` it still becomes a
    (degenerate, always-zero) investment variable in the LP, surfacing as a
    ``v_invest`` column the output post-processor has to tolerate (see
    ``read_parameters._entity_universe``).  ``db`` here is already
    scenario-filtered, so this fires either because the connection was
    never wired to its nodes, or because one of its endpoint nodes is not
    active in the current scenario (so the ``connection__node__node`` row
    is filtered out while the connection itself survives).  We cannot tell
    the two apart from the filtered set — the connection→node mapping
    lived only in the now-absent ``connection__node__node`` row — so the
    warning names both possibilities and tells the user to check the data.

    Only a warning is emitted: the solve proceeds, treating the connection
    as a no-op.
    """
    connected: set[str] = set()
    for ent in db.find_entities(entity_class_name="connection__node__node"):
        byname = ent["entity_byname"]
        if byname:
            connected.add(byname[0])
    for ent in db.find_entities(entity_class_name="connection"):
        byname = ent["entity_byname"]
        if not byname:
            continue
        name = byname[0]
        if name not in connected:
            logger.warning(
                "Connection '%s' excluded from this scenario: missing one/both endpoint "
                "nodes or its connection__node__node — '%s' not included. Check your data!",
                name, name,
            )
