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
    "validate_unit_input_shares",
    "validate_output_min_coeff",
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
