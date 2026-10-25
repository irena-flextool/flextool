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
from collections.abc import Iterable

import spinedb_api as api

from flextool.engine.input._solve_state import FlexToolConfigError


__all__ = [
    "validate_timeline_timestep_duration",
    "validate_capacity_margin_groups",
    "validate_group_limit_flags",
    "validate_min_load_efficiency_inputs",
    "validate_ladder_methods",
    "validate_group_output_memberships",
    "validate_connection_node_memberships",
    "validate_connection_delay",
    "validate_unit_input_shares",
    "validate_output_min_coeff",
    "validate_node_inflow_not_period",
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


def _pv_rows(src, entity_class: str, parameter: str) -> Iterable:
    """Parameter-value items of ``(entity_class, parameter)``.  ``src`` is a ``SpineDBBackend`` (its bulk
    ``(class, param)`` index, built once per backend, through ``parameter_value_items``) or a ``DatabaseMapping``."""
    if hasattr(src, "parameter_value_items"):
        return src.parameter_value_items(entity_class, parameter)
    return src.find_parameter_values(entity_class_name=entity_class, parameter_definition_name=parameter)


def _parsed_then_evict(pv):
    """Parse ``pv``'s value and drop spinedb-api's cached parse right away (Track A eviction contract, see
    ``SpineDBBackend.parameter_values``), so a validator never keeps every series of a class parsed at once."""
    value = pv["parsed_value"]
    item = getattr(pv, "mapped_item", None)
    if item is not None:
        item._parsed_value = None
    return value


def _map_keys_all_levels(value, out: set[str]) -> None:
    """Add every index key of a (nested) Map to ``out``; Map values that are not Maps are not collected."""
    if not isinstance(value, api.Map):
        return
    for key, sub in zip(value.indexes, value.values):
        out.add(str(key))
        _map_keys_all_levels(sub, out)


_SOLVE_PERIOD_PARAMS = ("realized_periods", "invest_periods", "realized_invest_periods", "fix_storage_periods")


def _period_and_timestep_names(src) -> tuple[set[str], set[str]]:
    """``(periods, timesteps)`` of the scenario: see ``validate_node_inflow_not_period``."""
    periods: set[str] = set()
    for pv in _pv_rows(src, "solve", "period_timeset"):
        if pv["type"] == "map":
            _map_keys_all_levels(_parsed_then_evict(pv), periods)
    for par in _SOLVE_PERIOD_PARAMS:
        for pv in _pv_rows(src, "solve", par):
            if pv["type"] == "array":
                periods.update(str(v) for v in _parsed_then_evict(pv).values)
            elif pv["type"] == "map":
                _map_keys_all_levels(_parsed_then_evict(pv), periods)
    timesteps: set[str] = set()
    for pv in _pv_rows(src, "timeline", "timestep_duration"):
        if pv["type"] in ("map", "time_series"):
            timesteps.update(str(k) for k in _parsed_then_evict(pv).indexes)
    return periods, timesteps


def _keys_all_periods(indexes, periods: set[str], timesteps: set[str]) -> bool:
    """True when ``indexes`` is non-empty, every key is a period name and none is a timestep name.  Stops at the
    first key that is not a period or is a timestep, so a time series exits at its first key."""
    if len(indexes) == 0:
        return False
    for key in indexes:
        name = str(key)
        if name not in periods or name in timesteps:
            return False
    return True


def validate_node_inflow_not_period(src) -> None:
    """Raise FlexToolConfigError when a node's ``inflow`` is keyed by period.

    ``inflow`` is a constant, a time series (1-D map over time) or a stochastic 3-D map (branch, analysis time,
    time); it cannot be period-based (docs/reference.md, 'Timesteps and periods').  A period-keyed value is not
    read by any path: before this check it silently became inflow 0.  Flagged:
      * a 1-D map with index_name 'period';
      * any 2-D map (the schema has no rank-2 inflow; a period x time map is the usual intent);
      * a 1-D map with another index_name whose keys are all period names and none a timestep name.

    Period names: the keys of every ``solve.period_timeset``, the entries of the 1-D period arrays of a solve
    (``realized_periods``, ``invest_periods``, ``realized_invest_periods``, ``fix_storage_periods``) and the index
    keys at every level of their Map variants (never Map values: nested-investment variants map period to a Map).
    Timestep names: the keys of every ``timeline.timestep_duration``.  Both sets are built first (only when some
    inflow is a map), then each map is classified right after its parse, which is evicted at once: the rank is
    read without parsing, rank-1 and rank-2 maps are parsed (rank 2 for its first-level keys only), and only the
    first 3 keys of an offending map are kept.

    ``src`` is the ``SpineDBBackend`` of ``input_derivation.run`` (or a scenario-filtered ``DatabaseMapping``).
    """
    to_rank = api.parameter_value.from_database_to_dimension_count
    inflow_maps = [pv for pv in _pv_rows(src, "node", "inflow") if pv["type"] == "map"]
    if not inflow_maps:
        return
    periods, timesteps = _period_and_timestep_names(src)

    problems: list[str] = []
    for pv in inflow_maps:
        rank = to_rank(pv["value"], pv["type"])
        if rank not in (1, 2):
            continue
        value = _parsed_then_evict(pv)
        indexes = value.indexes
        if rank == 2:
            reason = "a 2-D map"
        elif value.index_name == "period":
            reason = "index_name 'period'"
        elif _keys_all_periods(indexes, periods, timesteps):
            reason = "all map keys are period names"
        else:
            continue
        shown = ", ".join(str(k) for k in indexes[:3]) + (", ..." if len(indexes) > 3 else "")
        problems.append(
            f"node '{pv['entity_byname'][0]}' inflow (alternative '{pv['alternative_name']}') is keyed by period "
            f"({reason}" + (f"; first-level keys: {shown}" if len(indexes) else "") + ")")
    if problems:
        raise FlexToolConfigError(
            "\n".join(problems)
            + "\ninflow is a constant or a time series and cannot be period-based; to vary demand by period use "
              "annual_flow (constant or period) with an inflow_method that scales to it (scale_to_annual_flow, "
              "scale_in_proportion, scale_to_annual_and_peak_flow).")


def validate_capacity_margin_groups(db, logger: logging.Logger) -> None:
    """Storage nodes are excluded from the capacity-margin constraint.
    Raise if any capacity_margin_method='manual' group contains *only*
    storage nodes (constraint would have no valid members); warn if a
    mix is present.
    """
    capacity_margin_groups: dict[str, list[str]] = {}
    for pv in db.find_parameter_values(
        entity_class_name="group", parameter_definition_name="capacity_margin_method",
    ):
        if pv["parsed_value"] == "manual":
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


def _explicit_values(db, entity_class: str, parameter: str) -> dict[str, list[float]]:
    """``{entity: [numeric elements]}`` of the scenario's explicit values of a one-dimensional class's
    ``parameter`` (scalars and every value of a period / time / stochastic-time map)."""
    out: dict[str, list[float]] = {}
    for pv in db.find_parameter_values(entity_class_name=entity_class, parameter_definition_name=parameter):
        if pv["type"] is None:
            continue
        out[pv["entity_byname"][0]] = _numeric_values(pv["parsed_value"])
    return out


def _entities_with_value(db, entity_class: str, parameter: str, value: str) -> set[str]:
    """Entities of ``entity_class`` whose explicit ``parameter`` value is ``value``."""
    return {pv["entity_byname"][0]
            for pv in db.find_parameter_values(entity_class_name=entity_class, parameter_definition_name=parameter)
            if pv["type"] is not None and pv["parsed_value"] == value}


# (limit parameter, method/flag parameter, value that switches the limit on)
_GROUP_LIMIT_FLAGS: tuple[tuple[str, str, str], ...] = (
    ("capacity_margin", "capacity_margin_method", "manual"),
    ("inertia_limit", "has_inertia", "yes"),
    ("non_synchronous_limit", "has_non_synchronous", "yes"),
)

# What a flagged group without the limit gets (the model reads the limit as 0).
_GROUP_LIMIT_ABSENT: dict[str, str] = {
    "capacity_margin": "its capacity must cover the peak net load (margin 0)",
    "inertia_limit": "it has no inertia requirement",
    "non_synchronous_limit": "it reads as 0, so no non-synchronous infeed is allowed (violations are penalised)",
}


def validate_group_limit_flags(db, logger: logging.Logger) -> None:
    """Warn about group limits and their method / flag that do not go together.

    The capacity margin, inertia and non-synchronous constraints are built for the groups whose
    ``capacity_margin_method`` is ``manual`` / ``has_inertia`` is ``yes`` / ``has_non_synchronous``
    is ``yes``, and nowhere else.  Warnings (``db`` is scenario-filtered):

    * a non-zero limit on a group without its method / flag: the limit is ignored;
    * the method / flag on a group with no limit value at all: the model reads the limit as 0.  An
      explicit 0 is a deliberate input and is not reported.
    """
    for limit, flag, on in _GROUP_LIMIT_FLAGS:
        flagged = _entities_with_value(db, "group", flag, on)
        values = _explicit_values(db, "group", limit)
        for g in sorted(values):
            if g not in flagged and any(v != 0.0 for v in values[g]):
                logger.warning(
                    "Group '%s' has %s but %s is not %s: the limit is ignored.", g, limit, flag, on)
        for g in sorted(flagged - set(values)):
            logger.warning(
                "Group '%s' has %s = %s but no %s: %s.", g, flag, on, limit, _GROUP_LIMIT_ABSENT[limit])


def validate_min_load_efficiency_inputs(db, logger: logging.Logger) -> None:
    """Check the conversion curve inputs of ``min_load_efficiency`` units.

    Errors (:class:`FlexToolConfigError`), for units with ``conversion_method = min_load_efficiency``, over every
    value of a scalar or a period / time / stochastic-time map:

    * ``efficiency_at_min_load`` not above 0 (the fuel use at minimum load would be infinite);
    * ``min_load`` not below 1 (the curve between minimum and full load would have no length).

    Warning: a unit with a ``min_load`` value above 0 anywhere (folded as ``process_minload`` does) and no
    ``efficiency_at_min_load``.  Such a unit uses its ``efficiency`` at minimum load too (a flat curve: the fuel
    use is output / efficiency, as with constant_efficiency, while the minimum load still applies).  Cells that an
    ``efficiency_at_min_load`` map leaves out get the efficiency in the same way, without a warning.  ``db`` is
    scenario-filtered.
    """
    mle = _entities_with_value(db, "unit", "conversion_method", "min_load_efficiency")
    if not mle:
        return
    min_load = _explicit_values(db, "unit", "min_load")
    eta_min = _explicit_values(db, "unit", "efficiency_at_min_load")
    errors: list[str] = []
    for unit in sorted(mle):
        if any(v <= 0.0 for v in eta_min.get(unit, [])):
            errors.append(f"unit '{unit}': efficiency_at_min_load {_fmt(eta_min[unit])} must be above 0")
        if any(v >= 1.0 for v in min_load.get(unit, [])):
            errors.append(f"unit '{unit}': min_load {_fmt(min_load[unit])} must be below 1 for a "
                          "min_load_efficiency unit (use constant_efficiency for a unit that only runs at full load)")
    if errors:
        raise FlexToolConfigError("min_load_efficiency inputs: " + "; ".join(errors) + ".")
    with_min_load = {u for u, vals in min_load.items() if any(v > 0.0 for v in vals)}
    for unit in sorted((mle & with_min_load) - set(eta_min)):
        logger.warning(
            "Unit '%s' uses min_load_efficiency with min_load > 0 but has no efficiency_at_min_load: its "
            "efficiency is used at minimum load too (a flat curve; the minimum load still applies).", unit)


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


def validate_connection_delay(db, logger: logging.Logger) -> None:
    """Warn once per connection that has a non-zero ``delay``.

    Connection delay is not implemented: no connection term shifts the
    flow in time (the delayed input term exists only in the indirect
    conversion of units).  The delay is therefore ignored everywhere
    (``_specs`` delay tables, ``_process_method`` and the engine readers
    read ``unit.delay`` only), so the connection is modelled exactly as
    without a delay.  Before, the delay still marked the connection
    delayed, which dropped its efficiency losses and made a ``regular``
    connection one-way.  ``db`` is scenario-filtered; a delay of 0 (or a
    map of zeros) changes nothing and is not reported.
    """
    for pv in db.find_parameter_values(
        entity_class_name="connection", parameter_definition_name="delay",
    ):
        if pv["type"] is None:
            continue
        if not any(v != 0.0 for v in _numeric_values(pv["parsed_value"])):
            continue
        logger.warning(
            "Connection '%s' has a delay: connection delay is not "
            "implemented yet and is ignored (the connection is modelled "
            "without a delay). Use a unit with a delay for a delayed flow.",
            pv["entity_byname"][0],
        )


def validate_connection_startup(db, provider) -> None:
    """Raise when a connection in a ``dc_power_flow_with_angles`` group has a ``startup_method``.

    A DC power-flow line's flow is set by the voltage angles (``dc_flow_eq``); an offline line would still tie
    the angles, and switching it needs line-switching constraints FlexTool does not have.  Reads the DC line
    set from the Provider (``input/connection_dc_power_flow``, after :func:`derive_dc_power_flow`).  The model
    builder raises the same error for data built without the input derivation
    (``model.core._validate_no_online_dc_lines``).
    """
    from flextool.engine.model.core import DC_ONLINE_MESSAGE

    if not provider.has("input/connection_dc_power_flow"):
        return
    dc_frame = provider.get("input/connection_dc_power_flow")
    if dc_frame is None or dc_frame.height == 0:
        return
    dc = {str(x) for x in dc_frame.get_column(dc_frame.columns[0]).to_list()}
    hit = sorted(
        pv["entity_byname"][0]
        for pv in db.find_parameter_values(entity_class_name="connection",
                                           parameter_definition_name="startup_method")
        if pv["type"] is not None and str(pv["parsed_value"]) != "no_startup" and pv["entity_byname"][0] in dc)
    if hit:
        raise FlexToolConfigError(DC_ONLINE_MESSAGE.format(names=", ".join(hit)))


def validate_group_increase_reserve_ratio(db, logger: logging.Logger) -> None:
    """Warn once per reserve group with an ``increase_reserve_ratio`` but no dynamic ``reserve_method``.

    The group ratio adds the demand of the group's nodes to the dynamic requirement only
    (``model._reserve.group_demand_requirement``); with any other method (or none) it is ignored.
    ``db`` is scenario-filtered; a ratio of 0 changes nothing and is not reported.
    """
    from flextool.engine.preprocess._reserve_methods import RESERVE_DYNAMIC_METHODS

    methods = {
        tuple(pv["entity_byname"]): pv["parsed_value"]
        for pv in db.find_parameter_values(
            entity_class_name="reserve__upDown__group", parameter_definition_name="reserve_method",
        )
    }
    for pv in db.find_parameter_values(
        entity_class_name="reserve__upDown__group", parameter_definition_name="increase_reserve_ratio",
    ):
        if pv["type"] is None:
            continue
        if not any(v != 0.0 for v in _numeric_values(pv["parsed_value"])):
            continue
        key = tuple(pv["entity_byname"])
        if methods.get(key) in RESERVE_DYNAMIC_METHODS:
            continue
        logger.warning(
            "increase_reserve_ratio of reserve group '%s' is used only with a dynamic reserve_method "
            "(dynamic_only, timeseries_and_dynamic, dynamic_and_large_failure, all); it is ignored.",
            "__".join(key),
        )


def _numeric_values(parsed) -> list[float]:
    """Every numeric element of a parsed parameter value (a float, or the
    values of a Map / TimeSeries / array).  Non-numeric values give []."""
    if parsed is None or isinstance(parsed, (str, bool)):
        return []
    if isinstance(parsed, (int, float)):
        return [float(parsed)]
    values = getattr(parsed, "values", None)
    if values is None:
        return []
    out: list[float] = []
    for v in values:
        out.extend(_numeric_values(v))
    return out


def _arc_values(db, entity_class: str, parameter: str
                ) -> dict[tuple[str, str], list[float]]:
    """``{(unit, node): [numeric elements]}`` of the scenario's explicit
    values of ``entity_class.parameter``."""
    out: dict[tuple[str, str], list[float]] = {}
    for pv in db.find_parameter_values(entity_class_name=entity_class,
                                       parameter_definition_name=parameter):
        if pv["type"] is None:
            continue
        vals = _numeric_values(pv["parsed_value"])
        if vals:
            out[(pv["entity_byname"][0], pv["entity_byname"][1])] = vals
    return out


def _arcs(db, entity_class: str) -> dict[str, list[str]]:
    """``{unit: [nodes]}`` of the scenario's ``entity_class`` entities."""
    out: dict[str, list[str]] = {}
    for ent in db.find_entities(entity_class_name=entity_class):
        unit, node = ent["entity_byname"][0], ent["entity_byname"][1]
        out.setdefault(unit, []).append(node)
    return out


def _process_methods(provider) -> dict[str, str]:
    """``{process: method}`` from ``derive_process_method``'s output."""
    if provider is None or not provider.has("input/process_method"):
        return {}
    df = provider.get("input/process_method")
    if df is None or df.height == 0:
        return {}
    return dict(zip(df.get_column("process").to_list(),
                    df.get_column("method").to_list()))


def validate_unit_input_shares(db, provider, logger: logging.Logger) -> None:
    """Check ``unit__inputNode.input_share_min`` / ``input_share_max``.

    Runs once per scenario (after ``derive_process_method``).  Only inputs
    with ``conversion_flow_coeff ≠ 0`` count ("the unit's input" is their
    flow × conversion_flow_coeff).  Errors (:class:`FlexToolConfigError`):

    * ``input_share_min`` outside [0, 1];
    * ``input_share_max`` below 0;
    * ``input_share_min`` above the same input's ``input_share_max``: the
      input must supply at least that share of the unit's current input but
      may supply at most the smaller share of its full-load input, so the
      unit could not run at full load;
    * the ``input_share_min`` of a unit's inputs summing above 1 (the unit
      could never run);
    * ``input_share_min`` on a unit that has an input with a negative
      ``conversion_flow_coeff`` (the unit's input would no longer be an
      energy total).

    Warnings: ``input_share_max`` above 1 (same effect as 1);
    ``input_share_min`` on an input with ``conversion_flow_coeff = 0`` or
    on a unit with fewer than two inputs with ``conversion_flow_coeff ≠ 0``
    (no effect).
    """
    inputs = _arcs(db, "unit__inputNode")
    if not inputs:
        return
    conv = _arc_values(db, "unit__inputNode", "conversion_flow_coeff")
    share_min = _arc_values(db, "unit__inputNode", "input_share_min")
    share_max = _arc_values(db, "unit__inputNode", "input_share_max")
    if not share_min and not share_max:
        return
    errors: list[str] = []
    for unit in sorted(inputs):
        nodes = inputs[unit]
        conv_of = {n: conv.get((unit, n), [1.0]) for n in nodes}
        live = [n for n in nodes if any(v != 0.0 for v in conv_of[n])]
        for n in nodes:
            mins = share_min.get((unit, n), [])
            maxs = share_max.get((unit, n), [])
            if any(v < 0.0 or v > 1.0 for v in mins):
                errors.append(
                    f"unit '{unit}' input '{n}': input_share_min "
                    f"{_fmt(mins)} is outside [0, 1].  If this database was "
                    "migrated during development, the value may predate the "
                    "input_share_min meaning (it was carried over from the "
                    "old input-side capacity_min_coeff); re-author or "
                    "delete it")
            if any(v < 0.0 for v in maxs):
                errors.append(
                    f"unit '{unit}' input '{n}': input_share_max "
                    f"{_fmt(maxs)} is negative")
            elif any(v > 1.0 for v in maxs):
                logger.warning(
                    "Unit '%s' input '%s': input_share_max %s is above 1; "
                    "it has the same effect as 1 (no limit).",
                    unit, n, _fmt(maxs))
            if mins and maxs and max(mins) > min(maxs) + 1e-12:
                errors.append(
                    f"unit '{unit}' input '{n}' must supply at least "
                    f"{max(mins):g} of the unit's input (input_share_min) "
                    f"but may supply at most {min(maxs):g} of its full-load "
                    "input (input_share_max): the unit could not run at "
                    "full load")
            if any(v > 0.0 for v in mins):
                if n not in live:
                    logger.warning(
                        "Unit '%s' input '%s': input_share_min has no "
                        "effect on an input with conversion_flow_coeff = 0.",
                        unit, n)
                elif len(live) < 2:
                    logger.warning(
                        "Unit '%s' input '%s': input_share_min has no "
                        "effect on a unit with a single input.", unit, n)
        live_mins = [max(share_min.get((unit, n), [0.0])) for n in live]
        if len(live) >= 2 and sum(live_mins) > 1.0 + 1e-9:
            errors.append(
                f"unit '{unit}': input_share_min of its inputs sums to "
                f"{sum(live_mins):g} > 1; the unit could never run")
        if (len(live) >= 2 and any(v > 0.0 for v in live_mins)
                and any(min(conv_of[n]) < 0.0 for n in live)):
            neg = [n for n in live if min(conv_of[n]) < 0.0]
            errors.append(
                f"unit '{unit}': input_share_min is not defined for a unit "
                f"with a negative input conversion_flow_coeff "
                f"({', '.join(neg)})")
    if errors:
        raise FlexToolConfigError(
            "Invalid input shares:\n  " + "\n  ".join(errors))


def _fmt(values: list[float]) -> str:
    return (f"{values[0]:g}" if len(values) == 1
            else "[" + ", ".join(f"{v:g}" for v in values) + "]")


def validate_output_min_coeff(db, provider, logger: logging.Logger) -> None:
    """Check ``unit__outputNode.capacity_min_coeff`` against the per-unit
    minimum load.

    The minimum load is once per unit on the sum of its outputs
    (``Σ outputs ≥ min_load × online capacity``) and carries no
    coefficient; ``capacity_min_coeff`` is an optional extra floor on one
    output of a multi-output unit.  Runs once per scenario.

    * Error: an explicit ``capacity_min_coeff`` outside [0, 1].
    * Warning: a unit with online variables and ``min_load > 0`` whose
      explicit output coefficients are all below 1 (0 included — the old
      way to lower or switch off the minimum load).
    * Warning: a multi-output online unit whose per-output floors
      ``Σ_k capacity_min_coeff_k · min_load > 1`` or
      ``capacity_min_coeff_k · min_load > capacity_max_coeff_k`` may be
      unable to run while online (static values only; availability and
      time-varying values are not checked).
    * Warning: an online min-load unit whose outputs all have
      ``conversion_flow_coeff = 0`` has no output measure and no floor.
    """
    outputs = _arcs(db, "unit__outputNode")
    if not outputs:
        return
    cmin = _arc_values(db, "unit__outputNode", "capacity_min_coeff")
    bad = [f"unit '{u}' output '{n}': {_fmt(v)}"
           for (u, n), v in sorted(cmin.items())
           if any(x < 0.0 or x > 1.0 for x in v)]
    if bad:
        raise FlexToolConfigError(
            "capacity_min_coeff must lie in [0, 1]:\n  " + "\n  ".join(bad))
    min_load: dict[str, float] = {}
    for pv in db.find_parameter_values(entity_class_name="unit",
                                       parameter_definition_name="min_load"):
        vals = _numeric_values(pv["parsed_value"])
        if vals:
            min_load[pv["entity_byname"][0]] = max(vals)
    if not min_load:
        return
    from flextool.input_derivation._method_constants import (
        METHOD_LP, METHOD_MIP,
    )
    methods = _process_methods(provider)
    online = {p for p, mth in methods.items() if mth in METHOD_LP | METHOD_MIP}
    cmax = _arc_values(db, "unit__outputNode", "capacity_max_coeff")
    conv = _arc_values(db, "unit__outputNode", "conversion_flow_coeff")
    for unit in sorted(outputs):
        ml = min_load.get(unit, 0.0)
        if ml <= 0.0 or unit not in online:
            continue
        nodes = outputs[unit]
        explicit = [max(cmin[(unit, n)]) for n in nodes if (unit, n) in cmin]
        if explicit and max(explicit) < 1.0:
            logger.warning(
                "Unit '%s': capacity_min_coeff no longer lowers the unit's "
                "minimum load (the floor is min_load x online capacity on "
                "the sum of the outputs); set min_load lower (min_load = 0 "
                "switches it off) instead, or remove the coefficient.", unit)
        live = [n for n in nodes
                if any(v != 0.0 for v in conv.get((unit, n), [1.0]))]
        if not live:
            logger.warning(
                "Unit '%s': all outputs have conversion_flow_coeff = 0, so "
                "the unit has no minimum-load floor.", unit)
            continue
        if len(live) < 2:
            continue
        floors = {n: max(cmin.get((unit, n), [0.0])) for n in live}
        total = sum(floors.values()) * ml
        over_cap = [n for n in live
                    if floors[n] * ml > min(cmax.get((unit, n), [1.0])) + 1e-12]
        if total > 1.0 + 1e-9 or over_cap:
            logger.warning(
                "Unit '%s': its per-output floors (capacity_min_coeff x "
                "min_load) %s; the unit may be unable to run while online.",
                unit,
                (f"sum to {total:g} of the capacity" if total > 1.0 + 1e-9
                 else "exceed capacity_max_coeff on "
                 + ", ".join(over_cap)))
