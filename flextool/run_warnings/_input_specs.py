"""The input-side warning codes (W-0100..W-0499) of the registry :mod:`flextool.run_warnings.registry`: the checks
of the input derivation (``flextool/input_derivation``) and the plots.  Same rules as the registry's own codes
(code order, a short lowercase ``message``, a ``body`` with **How to fix:**); the registry merges them.

* W-01xx groups, flowGroups and reserves
* W-02xx units and connections
* W-03xx nodes, profiles and the time / period keys of values
* W-04xx commodities, DC power flow and plots

The console line names the entity; the values behind each case (the parameter, the alternative, the key, the
nodes) go to the debug log (``--debug``), as ``W-0123 <name>: <details>``.
"""
from __future__ import annotations

from flextool.run_warnings.registry import NOTE, WARNING, WarningSpec

_GROUPS = "Groups and outputs"
_UNITS = "Units and connections"
_STARTUP = "Start-up inputs"
_NODES = "Nodes"
_PROFILES = "Profiles"
_KEYS = "Time series and period keys"
_VALUES = "Input values"
_UOM = "Units of measure"
_COMMODITIES = "Commodities"
_DC = "DC power flow"
_PLOTS = "Plots"


def _w(code: str, message: str, title: str, topic: str, body: str, severity: str = WARNING) -> WarningSpec:
    return WarningSpec(code=code, message=message, title=title, topic=topic, body=body, severity=severity)


INPUT_SPECS: tuple[WarningSpec, ...] = (
    # -- groups, flowGroups, reserves (W-01xx) ----------------------------------------------------------------
    _w("W-0100", "capacity-margin group has storage nodes with demand; not covered",
       "Storage node with demand in a capacity-margin group", _GROUPS,
       "A group with `capacity_margin_method` `absolute` or `relative_of_peak_demand` has a storage node with "
       "demand (a negative `inflow`, e.g. a heat node with a tank). The capacity margin covers the demand of the "
       "group's nodes without a storage state, so that demand is left out of the margin. The debug log names the "
       "nodes.\n\n"
       "**How to fix:** model the demand on a balance node of the group, connected to the storage node, if the "
       "margin should cover it."),
    _w("W-0101", "group has {limit} without its method or flag; ignored",
       "Group limit without its method or flag", _GROUPS,
       "The capacity margin, inertia and non-synchronous constraints are built only for the groups that switch "
       "them on: `capacity_margin_absolute` needs `capacity_margin_method: absolute`, `capacity_margin_relative` "
       "needs `capacity_margin_method: relative_of_peak_demand`, `inertia_limit` needs `has_inertia: yes` and "
       "`non_synchronous_limit` needs `has_non_synchronous: yes`. The group has a non-zero limit without it, so "
       "the limit is ignored.\n\n"
       "**How to fix:** set the method or flag, or remove the limit."),
    _w("W-0102", "group has no {limit} for its method or flag (read as 0)",
       "Group method or flag without its limit", _GROUPS,
       "The group switches on a capacity margin, inertia or non-synchronous constraint (see W-0101 for the pairs) "
       "but has no value for its limit, so the model reads the limit as 0: a capacity margin of 0 (the capacity "
       "must cover the demand), no inertia requirement, or no non-synchronous infeed allowed (violations are "
       "penalised). An explicit 0 is a deliberate input and is not reported.\n\n"
       "**How to fix:** give the limit a value, or switch the method or flag off."),
    _w("W-0103", "group has {limit} but its co2_method has no such cap; ignored",
       "CO2 cap without a capping co2_method", _GROUPS,
       "The group has `co2_max_period` or `co2_max_total` (any value, 0 included), but its `co2_method` "
       "(explicit, or the default) has no period or total cap, so the limit is ignored. The debug log gives the "
       "alternative and the method.\n\n"
       "**How to fix:** use a `co2_method` with `period` (for `co2_max_period`) or `total` (for `co2_max_total`) "
       "in it, or remove the limit."),
    _w("W-0104", "one-variable connection crosses a non-synchronous group boundary",
       "One-variable connection at the edge of a non-synchronous group", _GROUPS,
       "The non-synchronous limit counts the flows of a connection with one node in the group both ways: the "
       "flow into the group as infeed, the flow out of it as supply. A one-variable connection "
       "(`no_losses_no_variable_cost`, and the lines of a `dc_power_flow_with_angles` group, which use it "
       "internally) can flow both ways at once at no cost, so the optimiser can loosen the limit by up to the "
       "line capacity. The case is named as (connection, group). `is_DC` connections are not reported.\n\n"
       "**How to fix:** use `transfer_method: exact` for the connection (a DC power-flow line: check the results "
       "for simultaneous flows both ways)."),
    _w("W-0105", "group's member nodes have different units of measure",
       "Group of nodes in different units of measure", _UOM,
       "The member nodes of the group have different `unit_of_measure` values, so the group's sums are labelled "
       "`Mixed units of measure!`. `unit_of_measure` is a label only: FlexTool converts nothing. The debug log "
       "lists the nodes by unit.\n\n"
       "**How to fix:** keep nodes of one quantity in a group, or accept the mixed label."),
    _w("W-0106", "group with MW features has member nodes not in MWh",
       "Power group feature on nodes that are not in MWh", _UOM,
       "The group has reserves, inertia, a non-synchronous limit or a capacity margin. These features are in MW "
       "(inertia in MW·s) and assume power nodes, but some member nodes have a `unit_of_measure` that is not MWh "
       "(or `MWh_<suffix>`). The debug log names the features and the nodes.\n\n"
       "**How to fix:** leave the non-power nodes out of the group, or check their `unit_of_measure`."),
    _w("W-0107", "group capacity limit sums members in different units",
       "Group capacity limit over members in different units", _UOM,
       "The group has an `invest_*`, `retire_*` or `cumulative_*` capacity limit, and its members' capacities are "
       "in different units, so the limit adds them up as one number. The debug log lists the members by unit.\n\n"
       "**How to fix:** split the group so that each limit covers members of one unit."),
    _w("W-0108", "flowGroup's flows are at nodes with different units of measure",
       "flowGroup of flows in different units of measure", _UOM,
       "The flows of the flowGroup are at nodes with different `unit_of_measure` values, so its sums are labelled "
       "`Mixed units of measure!`. The debug log lists the nodes by unit.\n\n"
       "**How to fix:** keep flows of one quantity in a flowGroup, or accept the mixed label."),
    _w("W-0109", "increase_reserve_ratio without a dynamic reserve_method; ignored",
       "Reserve ratio without a dynamic reserve method", _GROUPS,
       "The reserve group (`reserve__upDown__group`) has an `increase_reserve_ratio`, which adds the demand of the "
       "group's nodes to the dynamic reserve requirement only. Its `reserve_method` is not dynamic, so the ratio "
       "is ignored.\n\n"
       "**How to fix:** use a dynamic `reserve_method` (`dynamic_only`, `timeseries_and_dynamic`, "
       "`dynamic_and_large_failure` or `all`), or remove the ratio."),

    # -- units and connections (W-02xx) -----------------------------------------------------------------------
    _w("W-0200", "min_load_efficiency unit has no efficiency_at_min_load; flat curve",
       "min_load_efficiency without efficiency_at_min_load", _UNITS,
       "The unit has `conversion_method: min_load_efficiency` and a `min_load` above 0, but no "
       "`efficiency_at_min_load`. Its `efficiency` is used at minimum load too: a flat curve (fuel use is output "
       "/ efficiency, as with `constant_efficiency`), while the minimum load still applies.\n\n"
       "**How to fix:** give `efficiency_at_min_load`, or use `constant_efficiency`."),
    _w("W-0201", "unit has piecewise_linear_efficiency but another conversion_method",
       "piecewise_linear_efficiency not used", _UNITS,
       "The unit has a `piecewise_linear_efficiency` curve, but its `conversion_method` is not "
       "`piecewise_linear`, so the curve is not used. The debug log gives the method.\n\n"
       "**How to fix:** set `conversion_method: piecewise_linear`, or remove the curve."),
    _w("W-0202", "efficiency not used by a piecewise_linear unit (its curve gives it)",
       "Efficiency on a piecewise_linear unit", _UNITS,
       "A `piecewise_linear` unit gets its efficiency from `piecewise_linear_efficiency`; its `efficiency` / "
       "`efficiency_at_min_load` are not used. The debug log names the parameters.\n\n"
       "**How to fix:** remove them (or keep them for another alternative that uses another method)."),
    _w("W-0203", "min_load is below the first point of the piecewise_linear curve",
       "min_load below the first point of a rising piecewise curve", _UNITS,
       "The unit's curve rises from its first point to the second (it burns fuel at zero output, so it gets "
       "online variables), and its `min_load` (none = 0) is below the first point of the curve. Between them the "
       "first segment is extended, so the unit can be online at zero output while burning fuel. The debug log "
       "gives the period, the efficiency at `min_load` and the fuel per MW online.\n\n"
       "**How to fix:** raise `min_load` to the first point of the curve, or start the curve at `min_load`."),
    _w("W-0204", "piecewise_linear input can be met by burning above the curve",
       "piecewise_linear input that can burn above its curve", _UNITS,
       "A multi-line `piecewise_linear` unit's rows are \"at least\": the input is at least what the curve needs. "
       "Something on the input (an upward reserve, a flowGroup minimum, a profile or user constraint on the "
       "input, a ramp limit, inertia counted from the flow) or a negative price or `co2_content` of the input's "
       "commodity can make it worth burning more input than the curve needs. The case is named as (unit, input "
       "node); the debug log names the item. A post-solve check reports the time steps where it happened.\n\n"
       "**How to fix:** move the item to the output where possible; check the post-solve warning."),
    _w("W-0205", "existing of a {cls} is not a whole number of units (integer online)",
       "Fractional unit count with integer online variables", _UNITS,
       "With `virtual_unitsize` set, `existing` is the number of units (capacity = existing × "
       "virtual_unitsize). The entity has integer online variables (`startup_method: binary`), which count whole "
       "units, so a count such as 2.5 cannot all be online. Results are otherwise unchanged. The debug log gives "
       "the counts and the unit size.\n\n"
       "**How to fix:** give `existing` as a whole number of units, or change `virtual_unitsize`."),
    _w("W-0206", "start-up input has no startup_energy; start-ups draw nothing",
       "Start-up input without startup_energy", _STARTUP,
       "The `unit__inputNode` arc has a `startup_input_method` that draws start-up energy, but no "
       "`startup_energy` (or only 0), so the start-ups take nothing from the node.\n\n"
       "**How to fix:** give `startup_energy`, or set `startup_input_method: just_input`."),
    _w("W-0207", "parameter has no effect on a startup_only arc (no flow)",
       "Flow parameter on a startup_only arc", _STARTUP,
       "A `startup_only` arc has no flow, so flow parameters on it have no effect. The debug log names them.\n\n"
       "**How to fix:** remove them, or use `startup_input_method: startup_and_input`."),
    _w("W-0208", "start-up node has no balance and no commodity; the draw is free",
       "Start-up node without balance or commodity", _STARTUP,
       "The start-up input's node is of type `commodity` but is not a node of a commodity, so the start-up "
       "energy is taken from nowhere and costs nothing.\n\n"
       "**How to fix:** link the node to a commodity (`commodity__node`), or make it a balance node."),
    _w("W-0209", "unit has only startup_only inputs and no profile",
       "Unit with only start-up inputs", _STARTUP,
       "Every input of the unit is `startup_only` and the unit has no profile, so its output costs only the "
       "start-up energy.\n\n"
       "**How to fix:** give the unit an input with flow (`just_input` or `startup_and_input`), or a profile."),
    _w("W-0210", "part-load curve has no input: every input is startup_only",
       "Part-load curve without an input", _STARTUP,
       "The unit is `min_load_efficiency` or `piecewise_linear`, but every input is `startup_only`, so its "
       "part-load section (or curve) has no input to draw from.\n\n"
       "**How to fix:** give the unit an input with flow (`just_input` or `startup_and_input`)."),
    _w("W-0211", "start-up input is not in the arc's constraint or flowGroup",
       "Start-up input outside the arc's constraints and flowGroups", _STARTUP,
       "The arc is `startup_and_input` and has a `constraint_flow_coeff` or is in a flowGroup. Those count only "
       "the flow of the arc, not the start-up input. The debug log names the constraints and flowGroups.\n\n"
       "**How to fix:** nothing if that is what you mean; otherwise model the start-up draw on its own arc."),
    _w("W-0212", "input_share_max above 1 (same as 1)",
       "input_share_max above 1", _UNITS,
       "`input_share_max` is a share of the unit's input; a value above 1 has the same effect as 1 (no limit). "
       "The case is named as (unit, input node).\n\n"
       "**How to fix:** use a share between 0 and 1, or remove the value."),
    _w("W-0213", "input_share_min has no effect on this input",
       "input_share_min without effect", _UNITS,
       "`input_share_min` has no effect on an input with `conversion_flow_coeff` 0, nor on a unit with a single "
       "input (with a non-zero coefficient). The debug log says which. The case is named as (unit, input node).\n\n"
       "**How to fix:** remove the value, or give the unit the inputs the share is meant for."),
    _w("W-0214", "capacity_min_coeff no longer lowers the minimum load",
       "capacity_min_coeff below 1 on an online unit", _UNITS,
       "The minimum load is once per unit on the sum of its outputs (min_load × online capacity) and carries no "
       "coefficient; `capacity_min_coeff` is an extra floor on one output. The unit's output coefficients are all "
       "below 1 (0 included), the old way to lower or switch off the minimum load.\n\n"
       "**How to fix:** set `min_load` lower instead (`min_load: 0` switches it off), or remove the coefficient."),
    _w("W-0215", "per-output floors may stop the online unit from running",
       "Per-output floors above the capacity", _UNITS,
       "The per-output floors of the multi-output unit (`capacity_min_coeff` × `min_load`) sum above 1, or exceed "
       "an output's `capacity_max_coeff`, so the unit may be unable to run while online. Only static values are "
       "checked. The debug log gives the details.\n\n"
       "**How to fix:** lower the `capacity_min_coeff` values or `min_load`."),
    _w("W-0216", "unit uses no input: all its outputs have conversion_flow_coeff 0",
       "Unit whose outputs all have conversion_flow_coeff 0", _UNITS,
       "Every output of the unit has `conversion_flow_coeff` 0, so the unit uses no input and the untied outputs "
       "are free, up to `max_flow_for_unconstrained_variables`. The debug log names the free outputs.\n\n"
       "**How to fix:** give an output a non-zero `conversion_flow_coeff`, or tie the flow with a user constraint."),
    _w("W-0217", "unit {side} with conversion_flow_coeff 0 has nothing to limit its flow",
       "Unlimited flow with conversion_flow_coeff 0", _UNITS,
       "An arc with `conversion_flow_coeff` 0 is out of the conversion equation and of every capacity, ramp and "
       "minimum-load limit. No user constraint (`constraint_flow_coeff`), flowGroup maximum or profile limit "
       "ties it, so an input can take any amount from its node for free (a dump load), or an output can put any "
       "amount into its node for free (a free source), up to `max_flow_for_unconstrained_variables`. A flowGroup "
       "`min_instant_flow` / `min_cumulative_flow` bounds it only from below. The case is named as (unit, "
       "node).\n\n"
       "**How to fix:** tie the flow with a user constraint, or give it a non-zero `conversion_flow_coeff`."),
    _w("W-0218", "capacity coefficient has no effect on a zero-coefficient output",
       "Capacity coefficient on an output with conversion_flow_coeff 0", _UNITS,
       "`capacity_max_coeff` / `capacity_min_coeff` on an output with `conversion_flow_coeff` 0 have no effect: "
       "the flow is outside the capacity limits. The debug log names the parameter.\n\n"
       "**How to fix:** remove the coefficient, or give the output a non-zero `conversion_flow_coeff`."),
    _w("W-0219", "connection has no connection__node__node in the scenario; left out",
       "Connection without its nodes", _UNITS,
       "The connection has no `connection__node__node` in this scenario: it was never linked to its nodes, or "
       "one of its nodes is not in the scenario. It transfers nothing; an `invest_method` still gives it an "
       "investment variable that is always 0.\n\n"
       "**How to fix:** add the `connection__node__node` entity, or include both nodes in the scenario (or "
       "leave the connection out)."),
    _w("W-0220", "connection delay is not implemented; ignored",
       "Connection delay", _UNITS,
       "A connection `delay` is not implemented: the connection is modelled without a delay.\n\n"
       "**How to fix:** use a unit with a `delay` for a delayed flow, or remove the delay."),
    _w("W-0221", "connection joins nodes with different units of measure",
       "Connection between nodes in different units of measure", _UOM,
       "The two nodes of the connection have different `unit_of_measure` values. Its flow, losses and capacity "
       "are labelled `Mixed units of measure!`; a connection moves one quantity. The debug log lists the nodes "
       "by unit.\n\n"
       "**How to fix:** check that this is intended (a unit converts between quantities)."),
    _w("W-0222", "connection efficiency above 1 creates energy",
       "Connection efficiency above 1", _VALUES,
       "A connection `efficiency` above 1 creates energy on the way. The debug log gives the value, the "
       "alternative and the key.\n\n"
       "**How to fix:** use an efficiency of at most 1 (0.98 for 2 % losses)."),
    _w("W-0223", "negative existing capacity of a {cls}",
       "Negative existing capacity", _VALUES,
       "A negative `existing` is supported, but more often a typing mistake. A unit with a negative capacity is "
       "an \"anti-energy\" unit (its flows run the other way); a connection gets a negative transfer capacity "
       "and a node a negative storage capacity. Combined with a profile, `min_load`, storage state limits or "
       "invest / retire limits it can make the problem infeasible. The debug log gives the value, the "
       "alternative and the key.\n\n"
       "**How to fix:** if you did not mean it, use 0 or more."),
    _w("W-0224", "virtual_unitsize of a {cls} is not above 0; ignored",
       "virtual_unitsize not above 0", _VALUES,
       "`virtual_unitsize` must be above 0; the value is ignored (the entity is one unit of its capacity).\n\n"
       "**How to fix:** give a size above 0, or remove the value."),
    _w("W-0225", "life extension of a {cls} without lifetime_technical; never extended",
       "Life extension without lifetime_technical", _VALUES,
       "The entity has `life_extension_cost` and `life_extension_years` but no `lifetime_technical`: its "
       "capacity never reaches an end of life, so it is never extended.\n\n"
       "**How to fix:** give `lifetime_technical`, or remove the life extension."),
    _w("W-0226", "availability of a {cls} above 1.5 looks like a percentage",
       "Availability that looks like a percentage", _VALUES,
       "`availability` is a share (0.95, not 95); values above 1.5 look like percentages. The debug log gives the "
       "value, the alternative and the key.\n\n"
       "**How to fix:** give the availability as a share."),
    _w("W-0227", "process has no process method for its settings; left out",
       "No process method", _UNITS,
       "The combination of the process's conversion, start-up and fork methods has no process method, so the "
       "process is left out. This should not happen with valid data; the debug log gives the combination.\n\n"
       "**How to fix:** check the process's `conversion_method` and `startup_method`, and report the case."),
    _w("W-0228", "process method resolves to not_applicable",
       "Process method not applicable", _UNITS,
       "The combination of the process's conversion, start-up and fork methods is not applicable. The debug log "
       "gives the combination.\n\n"
       "**How to fix:** check the process's `conversion_method` and `startup_method`."),
    _w("W-0229", "startup_method set to linear: minimum_time_method needs it",
       "startup_method set to linear for a minimum time", _UNITS,
       "The process has a `minimum_time_method` (`min_uptime`, `min_downtime` or `both`), which needs online "
       "variables, and `startup_method: no_startup`; it is run with `startup_method: linear`.\n\n"
       "**How to fix:** nothing; set `startup_method` yourself to choose `linear` or `binary`.",
       severity=NOTE),
    _w("W-0230", "startup_method set to linear: piecewise curve burns fuel at zero output",
       "startup_method set to linear for a rising piecewise curve", _UNITS,
       "The unit's `piecewise_linear_efficiency` curve burns fuel at zero output (its efficiency rises from the "
       "first point to the second), which needs online variables, so the unit is run with "
       "`startup_method: linear`. The debug log gives the points.\n\n"
       "**How to fix:** nothing; set `startup_method` yourself to choose `linear` or `binary`.",
       severity=NOTE),

    # -- nodes, profiles, time and period keys (W-03xx) -------------------------------------------------------
    _w("W-0300", "node has no units or connections; only slack can meet its inflow",
       "Node with inflow but no units or connections", _NODES,
       "The node has an `inflow` but no unit (`unit__inputNode` / `unit__outputNode`) or connection "
       "(`connection__node__node`), so only the penalty slack (`penalty_up` / `penalty_down`) can meet the "
       "inflow. The debug log gives the alternative of the inflow.\n\n"
       "**How to fix:** connect the node, or remove its inflow (or the node)."),
    _w("W-0301", "node has no units, connections or inflow; it does nothing",
       "Node without units, connections or inflow", _NODES,
       "The node has no unit, no connection and no inflow; it does nothing in this scenario.\n\n"
       "**How to fix:** connect the node, or leave it out of the scenario."),
    _w("W-0302", "node's units and connections only take energy out of it",
       "Node that nothing supplies", _NODES,
       "A `balance` / `balance_within_period` node with no positive inflow, whose units and connections only "
       "take energy out of it. Nothing brings energy in, so those flows stay at 0, or only the penalty slack "
       "(`penalty_up`) can supply them.\n\n"
       "**How to fix:** add a unit or connection that supplies it, give it a positive inflow, or make it a "
       "commodity node."),
    _w("W-0303", "node's units and connections only bring energy into it",
       "Node that nothing drains", _NODES,
       "A `balance` / `balance_within_period` node with no negative inflow (demand), whose units and connections "
       "only bring energy into it. Nothing takes energy out, so those flows stay at 0, or only the penalty slack "
       "(`penalty_down`) can absorb them.\n\n"
       "**How to fix:** add a unit or connection that takes from it, or give it a negative inflow."),
    _w("W-0304", "non-storage node has an invest_method; it cannot invest",
       "invest_method on a node that is not a storage node", _NODES,
       "Node investment and retirement are storage capacity: only a node with `node_type: storage` can invest or "
       "retire. The node's `invest_method` is ignored. The debug log gives the method and the node type.\n\n"
       "**How to fix:** remove the `invest_method`, or make the node a storage node."),
    _w("W-0305", "node's unit_of_measure looks like a rate, not an amount",
       "unit_of_measure that names a rate", _UOM,
       "`unit_of_measure` is the amount unit (m3, MWh), not the rate (m3/h, MW); with a rate the flows would be "
       "labelled wrong. The debug log suggests the amount unit.\n\n"
       "**How to fix:** set the amount unit (e.g. `m3` for `m3/h`)."),
    _w("W-0306", "profile link ({cls}) has no profile_method; the profile is not applied",
       "Profile link without profile_method", _PROFILES,
       "A `node__profile` / `connection__profile` link without an authored `profile_method` gets the default "
       "`none`, so the profile is not applied. An explicit `none` is not reported.\n\n"
       "**How to fix:** set `profile_method` `upper_limit`, `lower_limit` or `fixed`, or remove the link."),
    _w("W-0307", "profile above 1.5 limits a unit flow; it looks like a percentage",
       "Profile that looks like a percentage", _PROFILES,
       "The profile limits a unit flow (`upper_limit` or `fixed`) and has values above 1.5. A profile is a share "
       "of the capacity (0.35, not 35). The debug log gives the value, the alternative and the key.\n\n"
       "**How to fix:** give the profile as a share."),
    _w("W-0308", "time series of {parameter} on a timeline the scenario does not use",
       "Time series on another scenario's timeline", _KEYS,
       "Every key of the value is a time step of a timeline that is not one of this scenario's (another "
       "alternative's `timestep_duration`), so the value is not used: plain data of another scenario. The debug "
       "log gives the alternative, the keys and the timeline.\n\n"
       "**How to fix:** nothing if that is intended; otherwise include the timeline's alternative in the "
       "scenario, or re-key the value."),
    _w("W-0309", "time keys of {parameter} are not on any timeline; not used",
       "Time-series keys that are not time steps", _KEYS,
       "Some keys of the values are not time steps of any timeline of the scenario (hourly data on a 3 h "
       "timeline, another year's stamps, another letter case). The readers join a time series with the solve's "
       "time steps, so those values are dropped. The debug log gives the number of keys, the first of them and "
       "the timelines.\n\n"
       "**How to fix:** re-key the values to the timeline's time steps."),
    _w("W-0310", "profile has no value for some solve periods; 0 there",
       "Profile period map without some periods", _KEYS,
       "The profile is a period map without a value for some periods of the solves that run, so the profile is "
       "0 in those periods. The debug log names the periods.\n\n"
       "**How to fix:** add the periods to the map, or give the profile a constant value."),
    _w("W-0311", "map keys are neither periods nor time steps; value not used",
       "Map keys that are neither periods nor time steps", _KEYS,
       "The map keys of the value look like time steps but are neither periods nor time steps of the model, so "
       "the value is not used. The debug log gives the alternative and the keys.\n\n"
       "**How to fix:** key the map by period (p2020, ...) or by the time steps of a timeline."),
    _w("W-0312", "map keys are not periods of this scenario; values dropped",
       "Map keys that are not periods", _KEYS,
       "Keys of a period map are not periods. This is an error when the whole database is read; here only the "
       "scenario's periods were known, so it is a warning and the values with those keys are dropped. The debug "
       "log lists the values and the closest periods.\n\n"
       "**How to fix:** rename the keys to periods of the model (or add the periods to "
       "`model.periods_available`)."),

    # -- commodities, DC power flow, plots (W-04xx) -----------------------------------------------------------
    _w("W-0400", "price ladder of the commodity cannot be read; skipped",
       "Unreadable price ladder", _COMMODITIES,
       "The commodity's `price_ladder_cumulative` / `price_ladder_annual` is not a map of the expected shape "
       "(tier → {price, quantity}, or period → tier → {price, quantity} for the annual ladder), or no model "
       "periods were found to spread a 2-level annual ladder over. The ladder is skipped. The debug log gives the "
       "reason.\n\n"
       "**How to fix:** give the ladder in the documented shape."),
    _w("W-0401", "commodity is used by nodes with different units of measure",
       "Commodity of nodes in different units of measure", _UOM,
       "The `commodity__node` nodes of the commodity have different `unit_of_measure` values. Its price and "
       "`co2_content` are per the unit of each node, so one value cannot fit them all. The debug log lists the "
       "nodes by unit.\n\n"
       "**How to fix:** use one commodity per unit of measure."),
    _w("W-0402", "reference_node is not a member of its DC power-flow group; ignored",
       "DC power flow reference node outside the group", _DC,
       "The group's `reference_node` is not a member node of the group, so the reference node is chosen "
       "automatically (the node with the largest existing capacity of each connected part). The case is named as "
       "(group, node).\n\n"
       "**How to fix:** name a member node of the group, or remove `reference_node`."),
    _w("W-0403", "reference node of a DC power-flow network chosen automatically",
       "DC power flow reference node chosen automatically", _DC,
       "The group has no `reference_node`, so each connected part of its network gets the node with the largest "
       "existing capacity as its reference (voltage angle 0). The case is named as (group, node).\n\n"
       "**How to fix:** nothing; set `reference_node` to choose it yourself.",
       severity=NOTE),
    _w("W-0404", "line of a DC power-flow group has no or zero reactance; no susceptance",
       "DC power-flow line without reactance", _DC,
       "A line of a `dc_power_flow_with_angles` group needs a non-zero `reactance` for its susceptance; without "
       "it the line gets no power-flow equation. The debug log says which.\n\n"
       "**How to fix:** give the connection a non-zero `reactance`."),
    _w("W-0405", "line of a DC power-flow group has zero existing capacity; big-M needed",
       "DC power-flow line without existing capacity", _DC,
       "A DC power-flow line with zero existing capacity needs big-M constraints. The debug log gives the "
       "group's `candidate_precapacity_to_avoid_big_m`.\n\n"
       "**How to fix:** give the line a small existing capacity (e.g. the "
       "`candidate_precapacity_to_avoid_big_m` value)."),
    _w("W-0406", "plot settings name results this run did not write; not plotted",
       "Plot of results that were not written", _PLOTS,
       "The plot settings ask for result tables that the run did not write (an output that is switched off, or "
       "a feature the model does not have), so those plots are skipped.\n\n"
       "**How to fix:** nothing, or remove the plots from the plot settings.",
       severity=NOTE),
)
