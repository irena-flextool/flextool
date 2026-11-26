"""Data version 79: the schema batch."""
import json

from spinedb_api import from_database, Map, to_database

from flextool.update_flextool.db_migration.common import _commit_step, _ensure_parameter_group, _pdef_or_none
from flextool.update_flextool.db_migration.v50_v55 import _STORAGE_BINDING_METHOD_DESCRIPTION_V55
from flextool.update_flextool.db_migration.v77 import _v77_set_marker, _v77_union_types


def _v79_texts() -> dict[tuple[str, str], str]:
    """Descriptions written by v79 (``constraint_online_coeff``)."""
    t: dict[tuple[str, str], str] = {}
    for cls, what, needs in (
            ("unit", "units", "startup_method linear or binary, or a minimum_time_method"),
            ("connection", "units of the connection", "startup_method linear or binary")):
        t[(cls, "constraint_online_coeff")] = (
            "[unit of the constraint per unit online] A map of coefficients (Index: constraint name, value: "
            f"coefficient). Puts the number of {what} online in each time step on the left side of the user "
            "constraint. One unit is virtual_unitsize; without it, the largest existing capacity, else 1000 MW "
            f"(set virtual_unitsize when the capacity is invested). Needs online variables ({needs}); a "
            "coefficient without them is an error. With linear startup the number is continuous. For MW online, "
            "use the unit size (MW per unit) as the coefficient.")
    return t


def _v79_constraint_online_coeff(db) -> None:
    """``constraint_online_coeff`` on unit and connection (part of v79).

    A new optional ``1d_map`` (index: constraint name) in the ``constraint``
    group, no default and no value list: the number of units online in each
    time step as a term of the user constraint.  No data is rewritten.
    Idempotent: an existing definition is updated to the v79 fields.
    """
    texts = _v79_texts()
    for cls in ("unit", "connection"):
        if not db.get_item("entity_class", name=cls):
            continue
        _ensure_parameter_group(db, "constraint", "a36784", 70)
        fields = {
            "parameter_type_list": ("1d_map",),
            "parameter_group_name": "constraint",
            "description": texts[(cls, "constraint_online_coeff")],
        }
        pdef = _pdef_or_none(db, cls, "constraint_online_coeff")
        if pdef is None:
            db.add_parameter_definition(entity_class_name=cls, name="constraint_online_coeff", **fields)
        elif any(pdef[k] != v for k, v in fields.items()):
            db.update_parameter_definition(id=pdef["id"], **fields)


#: Descriptions written by v79: ``ramp_cost`` became functional (a cost per MW of change of the flow).
_V79_RAMP_DESCRIPTIONS: dict[tuple[str, str], str] = {
    **{(cls, "ramp_method"): (
        "[choice] Choice of ramp method. ramp_limit limits the ramp speed. ramp_cost charges every change in the "
        "flow between consecutive time steps. both adds the limit as well.")
       for cls in ("unit__inputNode", "unit__outputNode")},
    **{(cls, "ramp_cost"): (
        "[CUR/MW] Cost per MW of change in the step-average flow, up or down, between a time step and the step "
        "before it. Charged per change, independent of step length. Needs ramp_method ramp_cost or both. Constant. "
        "On a constant_efficiency unit with one input and one output, an input-side ramp cost prices the change "
        "of the unit's output flow.")
       for cls in ("unit__inputNode", "unit__outputNode")},
}


def _v79_ramp_cost_descriptions(db) -> None:
    """``ramp_method`` and ``ramp_cost`` descriptions on unit__inputNode / unit__outputNode (part of v79).

    Description-only: ``ramp_cost`` is now charged (``ramp_change_eq``), so the "not functional" text goes.
    No data is rewritten.  Definitions a database does not have are skipped.  Idempotent: a rerun finds every
    description already in its v79 form.
    """
    for (cls, name), text in sorted(_V79_RAMP_DESCRIPTIONS.items()):
        pdef = _pdef_or_none(db, cls, name)
        if pdef is None or (pdef["description"] or "") == text:
            continue
        db.update_parameter_definition(id=pdef["id"], description=text)


#: v79: the two solve-level detail reductions; values in the fixed word order of the design.
REDUCE_UNIT_DETAIL_METHODS: tuple[str, ...] = (
    "none", "integers", "integers_ramps", "integers_min_limits", "integers_ramps_min_limits", "online",
    "online_ramps")
REDUCE_SYSTEM_DETAIL_METHODS: tuple[str, ...] = (
    "none", "reserves", "inertia", "non_synchronous", "inertia_non_synchronous",
    "reserves_inertia_non_synchronous", "dc_power_flow", "reserves_inertia_non_synchronous_dc_power_flow")
_V79_REDUCE_DESCRIPTIONS: dict[str, str] = {
    "reduce_unit_detail_method": (
        "[choice] Detail of units and connections that this solve leaves out, to make it faster (typically an "
        "investment solve). none (default): all detail. integers: binary startup_method is solved as continuous "
        "and exact connections behave as regular. integers_ramps: integers, and no ramp limits or ramp costs. "
        "integers_min_limits: integers, and no min_load, capacity_min_coeff floors or minimum up / down times. "
        "integers_ramps_min_limits: integers, no ramps and no min limits. Under the integers levels, a unit or "
        "connection whose online variables nothing in the solve still uses (no start-up cost, min_load, "
        "capacity_min_coeff floor, minimum up / down time, min_load_efficiency curve, constraint_online_coeff in "
        "a user constraint, reserve, inertia or ramp limit left) has none, as with startup_method no_startup; "
        "the run log names them. online: everything of "
        "integers_min_limits, and no online variables: no start-up costs, user constraints that count units "
        "online are left out whole, min_load_efficiency units use their full-load efficiency (less fuel than "
        "the curve at part load), and ramp limits apply to the capacity. Reserves are then limited by built "
        "capacity, not online capacity, and inertia is counted from flows. online_ramps: online, and no ramp limits or ramp costs. Never adds detail."),
    "reduce_system_detail_method": (
        "[choice] System detail that this solve leaves out, to make it faster. none (default): all detail. "
        "reserves: no reserve requirements. inertia: no inertia limits. non_synchronous: no non-synchronous "
        "share limits. dc_power_flow: the lines of dc_power_flow_with_angles groups are lossless transport "
        "limited by their capacity (no angles). A combination leaves out each named part. The capacity margin "
        "is always kept."),
}


def _v79_reduce_detail_methods(db) -> None:
    """``solve.reduce_unit_detail_method`` and ``solve.reduce_system_detail_method`` (part of v79).

    Two scalar choices (type ``str``) with their value lists (``reduce_unit_detail_methods`` /
    ``reduce_system_detail_methods``), default ``none``, group ``solve_advanced``.  No data is rewritten.  Idempotent: an existing definition is
    updated to the v79 fields, and value-list entries already present are not added again.
    """
    if not db.get_item("entity_class", name="solve"):
        return
    for list_name, values in (("reduce_unit_detail_methods", REDUCE_UNIT_DETAIL_METHODS),
                              ("reduce_system_detail_methods", REDUCE_SYSTEM_DETAIL_METHODS)):
        if not db.get_item("parameter_value_list", name=list_name):
            db.add_parameter_value_list(name=list_name)
        items = db.get_items("list_value", parameter_value_list_name=list_name)
        present = {x["parsed_value"] for x in items}
        index = max((x["index"] for x in items), default=-1) + 1
        for value in values:
            if value in present:
                continue
            val, typ = to_database(value)
            db.add_list_value(parameter_value_list_name=list_name, value=val, type=typ, index=index)
            index += 1
    _ensure_parameter_group(db, "solve_advanced", "b56f6f", 87)
    default_val, default_type = to_database("none")
    for name, list_name in (("reduce_unit_detail_method", "reduce_unit_detail_methods"),
                            ("reduce_system_detail_method", "reduce_system_detail_methods")):
        fields = {"default_value": default_val, "default_type": default_type,
                  "parameter_value_list_name": list_name, "parameter_group_name": "solve_advanced",
                  "description": _V79_REDUCE_DESCRIPTIONS[name]}
        pdef = _pdef_or_none(db, "solve", name)
        if pdef is None:
            db.add_parameter_definition(entity_class_name="solve", name=name, parameter_type_list=("str",), **fields)
        elif any(pdef[k] != v for k, v in fields.items()) or tuple(pdef["parameter_type_list"] or ()) != ("str",):
            db.update_parameter_definition(id=pdef["id"], parameter_type_list=("str",), **fields)


#: v79: ``group.inertia_limit`` and ``group.non_synchronous_limit`` also take a time or period + time map.
_V79_GROUP_LIMIT_TYPES: tuple[str, ...] = ("float", "1d_map", "2d_map")
_V79_GROUP_LIMIT_DESCRIPTIONS: dict[tuple[str, str], str] = {
    ("group", "inertia_limit"): (
        "[MWs] Minimum synchronous inertia in the group. Used only when has_inertia is yes. No value (or 0) means "
        "no requirement. Constant, period, time or period+time. A time map must give a value for every time step "
        "of the solve (a missing time step is an error). Over a coarse time step the row uses the step mean."),
    ("group", "non_synchronous_limit"): (
        "[-] e.g. 0.8 means 80%. Largest share of non-synchronous supply in the group, measured against the real "
        "demand. Flows between the group's nodes do not count. Used only when has_non_synchronous is yes; no "
        "value means 0. Constant, period, time or period+time. A time map must give a value for every time step "
        "of the solve (a missing time step is an error). Over a coarse time step the row uses the step mean."),
}


def _v79_group_limits_by_time(db) -> None:
    """``group.inertia_limit`` and ``group.non_synchronous_limit`` per time step (part of v79).

    Each type list is widened to its union with ``float`` / ``1d_map`` / ``2d_map`` (:func:`_v77_union_types`) and
    the description written.  No value is rewritten: a constant or period map stays valid.  Definitions a database
    does not have are skipped.  Idempotent: a rerun finds every definition already in its v79 form.
    """
    for (cls, name), text in sorted(_V79_GROUP_LIMIT_DESCRIPTIONS.items()):
        pdef = _pdef_or_none(db, cls, name)
        if pdef is None:
            continue
        fields: dict = {}
        union = _v77_union_types(pdef["parameter_type_list"], _V79_GROUP_LIMIT_TYPES)
        if tuple(pdef["parameter_type_list"] or ()) != union:
            fields["parameter_type_list"] = union
        if (pdef["description"] or "") != text:
            fields["description"] = text
        if fields:
            db.update_parameter_definition(id=pdef["id"], **fields)


#: v79: ``energy_margin_adder`` is an energy per time step of the timeline, summed over a new_stepduration step.
_V79_ENERGY_MARGIN_ADDER_DESCRIPTION = (
    "[MWh] per time step of the timeline. Extra demand (negative inflow) added to every time step of the "
    "investment solve, when `energy_margin_method` = `inflow_adder`. The value is not a total that is divided over "
    "the steps: each time step gets the full value. On a timeline of 24 h time steps, a margin of 10 MW needs a "
    "value of 240. When the investment solve uses `new_stepduration`, a solve step gets the sum of the values of "
    "the time steps it combines (a constant or period value counts once per time step). Dispatch solves are not "
    "affected. 0 = off. Constant, period or period+time.")


#: v79 (C-6): ``unit.efficiency`` and ``efficiency_at_min_load`` may be above 1 (heat-pump COP, m³ per MWh).
_V79_UNIT_EFFICIENCY_DESCRIPTIONS = {
    "efficiency": (
        "[-] Efficiency of the unit at full load: outputs / inputs, each flow weighted by its conversion_flow_coeff "
        "(default 1, giving plain sum of outputs / sum of inputs). May be above 1: a heat pump with COP 3 has 3; a "
        "unit whose output is measured in another unit of measure (e.g. m³ of water per MWh of electricity) uses "
        "that ratio. A min_load_efficiency unit can have different efficiency below full load (see "
        "efficiency_at_min_load). No effect on a unit without inputs or without outputs. Constant, period, time, "
        "period+time or stochastic time."),
    "efficiency_at_min_load": (
        "[-] Output / input at minimum load, like efficiency (e.g. 0.4; may be above 1). Efficiency of a "
        "min_load_efficiency unit at its minimum load. Fuel use is a straight line from minimum load (this efficiency) to full load (efficiency). "
        "Has no effect when min_load is 0: the efficiency at zero output is not defined. No value: efficiency "
        "applies at minimum load too. Constant, period, time or period+time."),
}

#: v79 (C-10): the user-constraint capacity coefficients say how their map is indexed, like their siblings.
_V79_CONSTRAINT_COEFF_SENTENCE = "A map of coefficients (Index: constraint name, value: coefficient). "
_V79_CONSTRAINT_COEFF_PARAMETERS: tuple[tuple[str, str], ...] = tuple(
    (cls, f"constraint_{kind}_coeff")
    for cls in ("unit", "node", "connection")
    for kind in ("capacity", "cumulative_pre_built_capacity", "in_construction_capacity", "invested_capacity",
                 "retired_capacity"))
#: v79 (C-10): a doubled word in the ``timeset.timeset_duration`` description.
_V79_TIMESET_DURATION_TYPO = ("name of the the timestep", "name of the timestep")


def _v79_constraint_coeff_description(description: str) -> str:
    """*description* with the index sentence after its unit bracket (unchanged when it has the sentence)."""
    if _V79_CONSTRAINT_COEFF_SENTENCE.strip() in description:
        return description
    end = description.find("] ") + 2 if description.startswith("[") and "] " in description else 0
    return description[:end] + _V79_CONSTRAINT_COEFF_SENTENCE + description[end:]


def _v79_solve_step_series(db) -> None:
    """A solve's ``new_stepduration`` combines every time series by its kind (part of v79).

    No value is rewritten.  The ``node.energy_margin_adder`` description is rewritten (an energy per time step of
    the timeline, summed over a new_stepduration step), and ``unit.efficiency`` / ``efficiency_at_min_load`` say
    they may be above 1 (C-6).  The 15 user-constraint capacity coefficients say they are a map of coefficients
    indexed by constraint name, and a doubled word leaves the ``timeset_duration`` description (C-10).
    Behaviour-change markers, only on a database whose
    results change: ``solve_step_series_combined`` when some solve has a ``new_stepduration`` value, and
    ``energy_margin_adder_per_original_step`` when also some node has an ``energy_margin_adder`` value (the
    marker value lists the solves / nodes).  Idempotent: a rerun finds the description and the markers written.
    """
    pdef = _pdef_or_none(db, "node", "energy_margin_adder")
    if pdef is not None and (pdef["description"] or "") != _V79_ENERGY_MARGIN_ADDER_DESCRIPTION:
        db.update_parameter_definition(id=pdef["id"], description=_V79_ENERGY_MARGIN_ADDER_DESCRIPTION)
    for name, text in _V79_UNIT_EFFICIENCY_DESCRIPTIONS.items():
        pdef = _pdef_or_none(db, "unit", name)
        if pdef is not None and (pdef["description"] or "") != text:
            db.update_parameter_definition(id=pdef["id"], description=text)
    for cls, name in _V79_CONSTRAINT_COEFF_PARAMETERS:
        pdef = _pdef_or_none(db, cls, name)
        if pdef is not None:
            text = _v79_constraint_coeff_description(pdef["description"] or "")
            if text != (pdef["description"] or ""):
                db.update_parameter_definition(id=pdef["id"], description=text)
    pdef = _pdef_or_none(db, "timeset", "timeset_duration")
    if pdef is not None and _V79_TIMESET_DURATION_TYPO[0] in (pdef["description"] or ""):
        db.update_parameter_definition(id=pdef["id"], description=pdef["description"].replace(
            *_V79_TIMESET_DURATION_TYPO))
    solves = sorted({pv["entity_byname"][0] for pv in db.find_parameter_values(
        entity_class_name="solve", parameter_definition_name="new_stepduration")
        if from_database(pv["value"], pv["type"]) is not None})
    if solves:
        _v77_set_marker(db, "solve_step_series_combined", json.dumps({"since": 79, "solves": solves}))
        nodes = sorted({pv["entity_byname"][0] for pv in db.find_parameter_values(
            entity_class_name="node", parameter_definition_name="energy_margin_adder")
            if _v79_adder_has_value(from_database(pv["value"], pv["type"]))})
        if nodes:
            _v77_set_marker(db, "energy_margin_adder_per_original_step",
                            json.dumps({"since": 79, "solves": solves, "nodes": nodes}))


#: v79: the capacity margin counts firm and VRE capacity derated by availability, and storage-backed units,
#: connections and consumption by their flow; ``share_of_peak`` sets the margin as a share of the peak demand.
_V79_CAPACITY_MARGIN_DESCRIPTIONS: dict[tuple[str, str], str] = {
    ("group", "capacity_margin_method"): (
        "[choice] Whether the group needs a capacity margin in investment solves, and how the margin is given. "
        "none (default): no margin. manual: capacity_margin MW. share_of_peak: capacity_margin_share times the "
        "group's peak demand in the period. At every time step of an investment period, the capacity of the "
        "group's firm units (capacity x availability x capacity_max_coeff) and VRE units (capacity x profile x "
        "availability) must cover the group's demand plus the margin. Units fed from a storage node, "
        "connections and the units consuming from the group count by their actual flow, so exports add to the "
        "requirement; connections and lossy units inside the group are left out (their losses are not "
        "required). Units fed from fuel or commodity nodes count at capacity: the fuel is assumed to last."),
    ("group", "capacity_margin"): (
        "[MW] Capacity the group needs above its demand at every time step of an investment period. Used only "
        "with capacity_margin_method manual. No value means 0 (the demand must still be covered). Constant or "
        "period."),
    ("group", "capacity_margin_share"): (
        "[-] e.g. 0.15 means 15%. Capacity margin as a share of the group's peak demand: the largest sum of the "
        "demand of the group's nodes over the period's full time series (before representative periods and "
        "new_stepduration). Used only with capacity_margin_method share_of_peak. No value means 0. Constant or "
        "period."),
}


def _v79_capacity_margin(db) -> None:
    """The capacity margin counts by entity class and gets a ``share_of_peak`` method (part of v79).

    ``share_of_peak`` joins the ``capacity_margin_methods`` value list, ``group.capacity_margin_share`` (float or
    1d_map, no default, group ``investment``) is added, and the ``capacity_margin_method`` / ``capacity_margin``
    descriptions are rewritten.  No value is rewritten.  Behaviour-change marker ``capacity_margin_counts_by_flow``
    (``{"since": 79, "groups": [...]}``) only on a database that has a group with ``capacity_margin_method``
    ``manual``: its results change.  Idempotent: a rerun finds the value, the definition, the descriptions and the
    marker written.
    """
    if not db.get_item("entity_class", name="group"):
        return
    if db.get_item("parameter_value_list", name="capacity_margin_methods"):
        items = db.get_items("list_value", parameter_value_list_name="capacity_margin_methods")
        if "share_of_peak" not in {x["parsed_value"] for x in items}:
            val, typ = to_database("share_of_peak")
            db.add_list_value(parameter_value_list_name="capacity_margin_methods", value=val, type=typ,
                              index=max((x["index"] for x in items), default=-1) + 1)
    for (cls, name), text in sorted(_V79_CAPACITY_MARGIN_DESCRIPTIONS.items()):
        pdef = _pdef_or_none(db, cls, name)
        if name == "capacity_margin_share":
            fields = {"parameter_type_list": ("float", "1d_map"), "parameter_group_name": "investment",
                      "description": text}
            if pdef is None:
                db.add_parameter_definition(entity_class_name=cls, name=name, **fields)
            elif any((tuple(pdef[k] or ()) if k == "parameter_type_list" else pdef[k]) != v
                     for k, v in fields.items()):
                db.update_parameter_definition(id=pdef["id"], **fields)
        elif pdef is not None and (pdef["description"] or "") != text:
            db.update_parameter_definition(id=pdef["id"], description=text)
    groups = sorted({pv["entity_byname"][0] for pv in db.find_parameter_values(
        entity_class_name="group", parameter_definition_name="capacity_margin_method")
        if from_database(pv["value"], pv["type"]) == "manual"})
    if groups:
        _v77_set_marker(db, "capacity_margin_counts_by_flow", json.dumps({"since": 79, "groups": groups}))


#: v79: ``constraint.constraint_period_method`` -- a user constraint per time step or as one total per period.
CONSTRAINT_PERIOD_METHODS: tuple[str, ...] = ("per_timestep", "period_total")
_V79_CONSTRAINT_PERIOD_METHOD_DESCRIPTION = (
    "[choice] How the user constraint is applied in time. per_timestep (default): a row for every time step (a "
    "constraint with only capacity terms has one row per period). period_total: one row per period over the sum "
    "of the time steps, each step's terms times its duration and weight, annualised (divided by the period's share "
    "of a year): flows count in MWh per year and units online in unit-hours per year; capacity terms and their "
    "existing capacity are multiplied by the hours of the year, and constant is per year. A state coefficient is "
    "not allowed. In a rolling solve the row is written only in a roll that realizes the whole period; otherwise "
    "the solve leaves it out with a warning. Constant.")


def _v79_constraint_period_method(db) -> None:
    """``constraint.constraint_period_method`` (part of v79).

    A scalar choice (type ``str``) with the value list ``constraint_period_methods`` (``per_timestep``,
    ``period_total``), default ``per_timestep`` (the rows of before), group ``constraint``.  No data is rewritten.
    Idempotent: an existing definition is updated to the v79 fields, and value-list entries already present are not
    added again.
    """
    if not db.get_item("entity_class", name="constraint"):
        return
    list_name = "constraint_period_methods"
    if not db.get_item("parameter_value_list", name=list_name):
        db.add_parameter_value_list(name=list_name)
    items = db.get_items("list_value", parameter_value_list_name=list_name)
    present = {x["parsed_value"] for x in items}
    index = max((x["index"] for x in items), default=-1) + 1
    for value in CONSTRAINT_PERIOD_METHODS:
        if value in present:
            continue
        val, typ = to_database(value)
        db.add_list_value(parameter_value_list_name=list_name, value=val, type=typ, index=index)
        index += 1
    _ensure_parameter_group(db, "constraint", "a36784", 70)
    default_val, default_type = to_database("per_timestep")
    fields = {"default_value": default_val, "default_type": default_type, "parameter_value_list_name": list_name,
              "parameter_group_name": "constraint", "description": _V79_CONSTRAINT_PERIOD_METHOD_DESCRIPTION}
    pdef = _pdef_or_none(db, "constraint", "constraint_period_method")
    if pdef is None:
        db.add_parameter_definition(entity_class_name="constraint", name="constraint_period_method",
                                    parameter_type_list=("str",), **fields)
    elif any(pdef[k] != v for k, v in fields.items()) or tuple(pdef["parameter_type_list"] or ()) != ("str",):
        db.update_parameter_definition(id=pdef["id"], parameter_type_list=("str",), **fields)


#: v79: an explicit discount_rate 0 is 0 % (the straight-line annuity), no longer 5 %; lifetime is required where
#: the annuity or the lifetime window uses it.
_V79_DISCOUNT_RATE_DESCRIPTION = (
    "[-] e.g. 0.05 means 5% per year. Discount rate for investments (WACC). Reflects the financing cost and risk "
    "premium for this technology. When the model inflation_rate > 0, this should be a nominal rate. When "
    "inflation_rate = 0, this should be a real rate. Used to annualize investment costs over the lifetime: "
    "annuity = invest_cost x r / (1 - (1 + r)^-lifetime). 0 gives the straight-line annuity invest_cost / "
    "lifetime. No value means 0.05. Must be 0 or above. Constant or period.")
_V79_LIFETIME_REQUIRED = (
    " Required when the {what} invests or retires with a cost (invest_cost, salvage_value), and when "
    "lifetime_method is reinvest_choice or no_investment (also for existing capacity without investment).")
_V79_LIFETIME_DESCRIPTIONS: dict[str, str] = {
    cls: (
        f"[years] Operating lifetime. It starts when the {cap} starts to operate: with a construction lead time, "
        "after the construction. Used with the interest rate to calculate the annuity. What happens at the end of "
        f"the lifetime depends on lifetime_method: reinvest_automatic keeps the {cap} to the end of the model "
        "horizon, reinvest_choice retires it, no_investment retires it once."
        + _V79_LIFETIME_REQUIRED.format(what=what) + " Constant or period.")
    for cls, cap, what in (("unit", "capacity", "unit"), ("connection", "capacity", "connection"),
                           ("node", "storage capacity", "node"))}


def _v79_numbers(value) -> list[float]:
    """Every number of a parsed value (a constant, or the leaves of a map / time series)."""
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    values = getattr(value, "values", None)
    if values is None:
        return []
    out: list[float] = []
    for v in values:
        out.extend(_v79_numbers(v))
    return out


def _v79_discount_rate_zero(db) -> None:
    """``discount_rate`` 0 is 0 % (part of v79).

    Before v79 an explicit ``discount_rate`` of 0 or below was read as 5 %; now 0 gives the straight-line annuity
    ``invest_cost / lifetime`` and a negative value is an input error.  No value is rewritten.  The
    ``discount_rate`` and ``lifetime`` descriptions of unit / connection / node are rewritten.  Behaviour-change
    marker ``discount_rate_zero_is_zero`` (``{"since": 79, "entities": [...]}``) only on a database where an
    entity that can invest or retire (an ``invest_method`` other than ``not_allowed``) has an explicit
    ``discount_rate`` element ≤ 0: its results change.  Idempotent: a rerun finds the descriptions and the marker
    written.
    """
    entities: set[str] = set()
    for cls in ("unit", "connection", "node"):
        if not db.get_item("entity_class", name=cls):
            continue
        pdef = _pdef_or_none(db, cls, "discount_rate")
        if pdef is not None and (pdef["description"] or "") != _V79_DISCOUNT_RATE_DESCRIPTION:
            db.update_parameter_definition(id=pdef["id"], description=_V79_DISCOUNT_RATE_DESCRIPTION)
        pdef = _pdef_or_none(db, cls, "lifetime")
        if pdef is not None and (pdef["description"] or "") != _V79_LIFETIME_DESCRIPTIONS[cls]:
            db.update_parameter_definition(id=pdef["id"], description=_V79_LIFETIME_DESCRIPTIONS[cls])
        investing = {pv["entity_byname"][0] for pv in db.find_parameter_values(
            entity_class_name=cls, parameter_definition_name="invest_method")
            if from_database(pv["value"], pv["type"]) not in (None, "not_allowed")}
        for pv in db.find_parameter_values(entity_class_name=cls, parameter_definition_name="discount_rate"):
            name = pv["entity_byname"][0]
            if name in investing and any(v <= 0.0 for v in _v79_numbers(from_database(pv["value"], pv["type"]))):
                entities.add(name)
    if entities:
        _v77_set_marker(db, "discount_rate_zero_is_zero", json.dumps({"since": 79, "entities": sorted(entities)}))


def _v79_method_allows(method: str, word: str) -> bool:
    """Whether ``invest_method`` *method* allows investing (*word* ``invest``) or retiring (``retire``)."""
    return method == "cumulative_limits" or word in method.split("_")


def _v79_invest_without_cost(db) -> None:
    """Investment (retirement) offered also without ``invest_cost`` (``salvage_value``) (part of v79).

    Before v79 an entity got its investment variable in a period only when its ``invest_cost`` there was not 0
    (retirement: ``salvage_value``), or a user constraint had a capacity coefficient on it; now every period of
    an ``invest_method`` that allows it is offered.  No value is rewritten.  Behaviour-change marker
    ``invest_without_cost_offered`` (``{"since": 79, "entities": [...]}``) only on a database where a unit /
    connection / node whose ``invest_method`` allows investing has no ``invest_cost`` or one with a 0 element,
    or one that allows retiring has no ``salvage_value`` or one with a 0 element (any alternative).  The run's
    notice names only the (entity, period) pairs the old rule left out.  Idempotent: a rerun finds the marker.
    """
    entities: set[str] = set()
    for cls in ("unit", "connection", "node"):
        if not db.get_item("entity_class", name=cls):
            continue
        methods: dict[str, set[str]] = {}
        for pv in db.find_parameter_values(entity_class_name=cls, parameter_definition_name="invest_method"):
            value = from_database(pv["value"], pv["type"])
            if isinstance(value, str):
                methods.setdefault(pv["entity_byname"][0], set()).add(value)
        for word, cost in (("invest", "invest_cost"), ("retire", "salvage_value")):
            allowed = {e for e, ms in methods.items() if any(_v79_method_allows(m, word) for m in ms)}
            if not allowed:
                continue
            costed: set[str] = set()
            zero: set[str] = set()
            for pv in db.find_parameter_values(entity_class_name=cls, parameter_definition_name=cost):
                name = pv["entity_byname"][0]
                costed.add(name)
                if any(v == 0.0 for v in _v79_numbers(from_database(pv["value"], pv["type"]))):
                    zero.add(name)
            entities |= {e for e in allowed if e not in costed or e in zero}
    if entities:
        _v77_set_marker(db, "invest_without_cost_offered", json.dumps({"since": 79, "entities": sorted(entities)}))


#: v79: ``unit__inputNode.startup_input_method`` -- the input arc also, or only, supplies the energy a start-up uses.
STARTUP_INPUT_METHODS: tuple[str, ...] = ("just_input", "startup_and_input", "startup_only")
_V79_STARTUP_INPUT_DESCRIPTIONS: dict[str, str] = {
    "startup_input_method": (
        "[choice] What the input arc supplies. just_input (default): the normal input flow. startup_and_input: "
        "the normal input flow, and each start-up of the unit also takes startup_energy x MW started from this "
        "node. startup_only: only the start-up energy; the arc has no flow and takes no part in the conversion, "
        "the capacity, input shares or ramps. The start-up input can be fuel (oil to start a coal unit), heat "
        "(from the tank of a CSP plant) or electricity. Needs online variables (startup_method linear or binary, "
        "or a minimum_time_method). The start-up input is priced as the node's commodity, counts in CO2 prices "
        "and caps and is dropped with the start-ups by reduce_unit_detail_method online. It is not part of "
        "constraint_flow_coeff, flow groups or the capacity margin."),
    "startup_energy": (
        "[MWh per MW started] Energy taken from this input node per MW of capacity started up (start-ups x "
        "unit size). Used with startup_input_method startup_and_input or startup_only. Constant or period."),
}


def _v79_startup_input(db) -> None:
    """``unit__inputNode.startup_input_method`` and ``startup_energy`` (part of v79).

    ``startup_input_method``: a scalar choice (type ``str``) with the value list ``startup_input_methods``
    (``just_input``, ``startup_and_input``, ``startup_only``), default ``just_input`` (the arc of before).
    ``startup_energy``: float or ``1d_map`` (period), no default.  Both in group ``tech_advanced``.  No data is
    rewritten.  Idempotent: an existing definition is updated to the v79 fields, and value-list entries already
    present are not added again.
    """
    if not db.get_item("entity_class", name="unit__inputNode"):
        return
    list_name = "startup_input_methods"
    if not db.get_item("parameter_value_list", name=list_name):
        db.add_parameter_value_list(name=list_name)
    items = db.get_items("list_value", parameter_value_list_name=list_name)
    present = {x["parsed_value"] for x in items}
    index = max((x["index"] for x in items), default=-1) + 1
    for value in STARTUP_INPUT_METHODS:
        if value in present:
            continue
        val, typ = to_database(value)
        db.add_list_value(parameter_value_list_name=list_name, value=val, type=typ, index=index)
        index += 1
    _ensure_parameter_group(db, "tech_advanced", "d9925c", 35)
    default_val, default_type = to_database("just_input")
    specs = (
        ("startup_input_method", ("str",), {"default_value": default_val, "default_type": default_type,
                                            "parameter_value_list_name": list_name}),
        ("startup_energy", ("float", "1d_map"), {}),
    )
    for name, types, extra in specs:
        fields = {**extra, "parameter_group_name": "tech_advanced",
                  "description": _V79_STARTUP_INPUT_DESCRIPTIONS[name]}
        pdef = _pdef_or_none(db, "unit__inputNode", name)
        if pdef is None:
            db.add_parameter_definition(entity_class_name="unit__inputNode", name=name, parameter_type_list=types,
                                        **fields)
        elif any(pdef[k] != v for k, v in fields.items()) or tuple(pdef["parameter_type_list"] or ()) != types:
            db.update_parameter_definition(id=pdef["id"], parameter_type_list=types, **fields)


#: ``node.unit_of_measure`` description (v79 part 11).  The only parameter description that explains the per-kW
#: cost rule for a node whose unit is not MWh (owner decision); the cost parameters keep their own wording.
V79_NODE_UNIT_OF_MEASURE_DESCRIPTION = (
    "[text] Unit of measure of this node's amounts, e.g. m3, t, MWh_th (default MWh). A label only: FlexTool "
    "converts nothing. Flows into and out of the node, and the capacity of a unit whose output is this node (of a"
    " connection when both of its nodes have this unit), are rates in this unit per hour (MWh → MW, m3 → m3/h); "
    "storage state, storage capacity and inflow per time step are amounts (m3); annual results are per year "
    "(m3/a); prices and penalty_up/down are CUR per this unit. The per-kW costs (invest_cost, fixed_cost, "
    "salvage_value) of a unit whose capacity is in this unit, and the per-kWh costs of this node's storage, are "
    "per 0.001 of the rate or amount: 500 CUR per m3/h of capacity is entered as 0.5, and 20 CUR per m3 of "
    "storage as 0.02. An efficiency into or out of this node is output units per input unit (e.g. m3/MWh).")


def _v79_node_unit_of_measure(db) -> None:
    """``node.unit_of_measure`` (part of v79).

    A constant string (type ``str``, no value list), default ``MWh``, group ``basics``: the unit of measure of the
    node's amounts.  Labels only (the results show the node's flows in this unit per hour); the engine does not
    read it.  No value is rewritten.  Idempotent: an existing definition is updated to the v79 fields.
    """
    if not db.get_item("entity_class", name="node"):
        return
    _ensure_parameter_group(db, "basics", "6fa8c7", 10)
    default_val, default_type = to_database("MWh")
    types = ("str",)
    fields = {"default_value": default_val, "default_type": default_type, "parameter_value_list_name": None,
              "parameter_group_name": "basics", "description": V79_NODE_UNIT_OF_MEASURE_DESCRIPTION}
    pdef = _pdef_or_none(db, "node", "unit_of_measure")
    if pdef is None:
        db.add_parameter_definition(entity_class_name="node", name="unit_of_measure", parameter_type_list=types,
                                    **{k: v for k, v in fields.items() if v is not None})
    elif any(pdef[k] != v for k, v in fields.items()) or tuple(pdef["parameter_type_list"] or ()) != types:
        db.update_parameter_definition(id=pdef["id"], parameter_type_list=types, **fields)


def _v79_rp_texts() -> dict[tuple[str, str], str]:
    """Descriptions written by v79 part 12: representative periods work in stochastic solves; blended methods bind
    forward only in a rolling solve."""
    return {
        ("node", "storage_binding_method"): (
            "[choice] " + _STORAGE_BINDING_METHOD_DESCRIPTION_V55 + " In a stochastic solve each forecast branch uses the "
            "representative periods of its period, and the blended_weights methods work per branch. In a "
            "rolling_window solve on representative periods the blended_weights methods are not applied: those "
            "nodes bind forward only (bind_forward_only), with a warning."),
        ("timeset", "representative_period_weights"): (
            "[-] Representative-period weighting map (rank-2 nested Map: base period start timestep -> representative "
            "period start timestep -> weight).  Created by the representative-periods preprocess tool; declares how "
            "much each representative period stands in for each base period.  Its presence enables RP-aware "
            "storage-binding ('*_blended_weights'); its absence silently downgrades those methods to the non-RP "
            "behaviour.  Mutually exclusive with 'timeset_weights' on the same timeset. The forecast branches of a "
            "stochastic solve use the representative periods of their period."),
    }


def _v79_rp_stochastic_descriptions(db) -> None:
    """Representative periods in stochastic and rolling solves (part of v79): the ``node.storage_binding_method``
    and ``timeset.representative_period_weights`` descriptions no longer say that representative periods do not
    work with stochastic branches (:func:`_v79_rp_texts`).  No value is rewritten.  Idempotent."""
    for (cls, name), text in _v79_rp_texts().items():
        if not db.get_item("entity_class", name=cls):
            continue
        pdef = _pdef_or_none(db, cls, name)
        if pdef is not None and (pdef["description"] or "") != text:
            db.update_parameter_definition(id=pdef["id"], description=text)


def _v79_solver_precommand_texts() -> dict[tuple[str, str], str]:
    """Descriptions written by v79 part 14: FlexTool never runs ``solve.solver_precommand``."""
    return {
        ("solve", "solver_precommand"): (
            "[text] Not used: FlexTool never runs this command. Kept only so that older databases still read."),
        ("solve", "solver_arguments"): (
            "[text] Map of HiGHS solver options (option name -> value), layered on top of the floor values in "
            "solver_config/highs.opt and below any CLI overrides (e.g. --solver-time-limit). Use this for ad-hoc "
            "HiGHS knobs that do not have a dedicated FlexTool parameter (solver_mip_gap)."),
    }


def _v79_solver_precommand_descriptions(db) -> None:
    """``solve.solver_precommand`` is not run (part of v79): its description says so, and the
    ``solve.solver_arguments`` description no longer lists it as a dedicated parameter
    (:func:`_v79_solver_precommand_texts`).  No value is rewritten.  Idempotent."""
    for (cls, name), text in _v79_solver_precommand_texts().items():
        if not db.get_item("entity_class", name=cls):
            continue
        pdef = _pdef_or_none(db, cls, name)
        if pdef is not None and (pdef["description"] or "") != text:
            db.update_parameter_definition(id=pdef["id"], description=text)


def _v79_stochastic_storage_paths(db) -> None:
    """Storage in stochastic solves follows each path (part of v79): plain ``bind_within_solve`` closes one loop
    per path (every forecast branch ends at the level the solve starts from) and ``bind_forward_only`` with a free
    start starts all the forecasts of a fan in the solve's first period at the same level.  No value is rewritten.
    Behaviour-change marker ``stochastic_storage_paths_tied`` (``{"since": 79, "solves": [...]}``) only on a
    database with a ``stochastic_branches`` value on some solve and a storage node whose results can change (the
    run names the nodes that get the new rows):

    * ``storage_binding_method`` ``bind_within_solve`` (in any alternative), or
    * ``bind_forward_only`` (stored in some alternative, or the default when none is stored) with a start that is
      not fixed somewhere: no ``storage_start_end_method`` stored (the default ``fix_nothing``) or a stored value
      other than ``fix_start`` / ``fix_start_end`` in some alternative.  A node whose every stored
      ``storage_start_end_method`` is ``fix_start`` or ``fix_start_end`` already starts every forecast from the
      same value and is left out.

    The marker is broader than the change: a free-start ``bind_forward_only`` node changes only when a fan opens in
    the solve's first period (a fan opening later continues from the trunk as before), and the migration does not
    read where the fans open or which alternatives a scenario combines.  Idempotent: a rerun finds the marker
    written.
    """
    if not (db.get_item("entity_class", name="solve") and db.get_item("entity_class", name="node")):
        return
    solves = sorted({pv["entity_byname"][0] for pv in db.find_parameter_values(
        entity_class_name="solve", parameter_definition_name="stochastic_branches")
        if from_database(pv["value"], pv["type"]) is not None})
    if not solves:
        return

    def stored(param: str) -> dict[str, set]:
        out: dict[str, set] = {}
        for pv in db.find_parameter_values(entity_class_name="node", parameter_definition_name=param):
            out.setdefault(pv["entity_byname"][0], set()).add(from_database(pv["value"], pv["type"]))
        return out

    storage = {pv["entity_byname"][0] for pv in db.find_parameter_values(
        entity_class_name="node", parameter_definition_name="node_type")
        if from_database(pv["value"], pv["type"]) == "storage"}
    methods = stored("storage_binding_method")
    starts = stored("storage_start_end_method")
    fixed_start = {"fix_start", "fix_start_end"}

    def changes(n: str) -> bool:
        if "bind_within_solve" in methods.get(n, set()):
            return True
        # A stored null is the default (bind_forward_only / fix_nothing).
        forward_only = n not in methods or bool(methods[n] & {"bind_forward_only", None})
        free_start = n not in starts or bool(starts[n] - fixed_start)
        return forward_only and free_start

    if any(changes(n) for n in storage):
        _v77_set_marker(db, "stochastic_storage_paths_tied", json.dumps({"since": 79, "solves": solves}))


#: v79: the solve parameters that split a solve when given as a 2-D map.
_V79_SPLIT_PERIOD_PARAMETERS = ("invest_periods", "realized_invest_periods", "realized_periods", "fix_storage_periods")


def _v79_period_names(value) -> list[str]:
    """The period names of a parsed 1-D period parameter: an array's values, a map's indexes."""
    if isinstance(value, Map):
        return [str(d) for d in value.indexes]
    values = getattr(value, "values", None)
    return [str(d) for d in values] if values is not None else []


def _v79_map_out_of_period_order(value, periods: list[str]) -> bool:
    """A ``years_represented`` map (periods with years above 0) that lists the ``periods`` out of their order, or
    lists a period outside them before one of them (it moves that period's start in the map count)."""
    if not isinstance(value, Map) or not periods:
        return False
    keys = [str(d) for d, v in zip(value.indexes, value.values)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0]
    used = [d for d in keys if d in periods]
    if used != [d for d in periods if d in used]:
        return True
    last_used = max((i for i, d in enumerate(keys) if d in periods), default=0)
    return any(d not in periods for d in keys[:last_used])


def _v79_discount_from_run_start(db) -> None:
    """Every solve discounts its investments and its operations from the start of the run (part of v79).

    No value is rewritten.  Behaviour-change marker ``chained_solve_discount_from_run_start``
    (``{"since": 79, "solves": [...]}``) only on a database with an ``inflation_rate`` value other than 0 (with 0
    every discount base gives the same results) and a solve whose results can change:

    * the solves of a ``model.solves`` list of two or more solves (a later solve starts after the run start),
    * a solve with ``contains_solves`` (its contained solves run after it, also from a later period),
    * a solve with a 2-D period map (``invest_periods`` / ``realized_invest_periods`` / ``realized_periods`` /
      ``fix_storage_periods``: it splits into solves that start later),
    * a solve with a fractional ``years_represented`` (it counted as a whole year for the later periods),
    * a solve whose ``years_represented`` map does not list its periods in the solve's period order, or lists a
      period the solve does not use before one it does (the investment walk counted the years in the period
      order of the solve, the operations in the map order; both now count in the map order, which also moves a
      single solve).  The period order is approximated by ``realized_periods`` then ``invest_periods`` (and also
      with ``period_timeset`` after them); either order flags the solve.

    The marker is broader than the change: a run whose solves all start at the run start with whole years keeps
    its results, and the migration does not read which alternatives a scenario combines.  Idempotent: a rerun
    finds the marker written.
    """
    if not (db.get_item("entity_class", name="solve") and db.get_item("entity_class", name="model")):
        return
    if not any(v != 0.0 for pv in db.find_parameter_values(entity_class_name="model",
                                                           parameter_definition_name="inflation_rate")
               for v in _v79_numbers(from_database(pv["value"], pv["type"]))):
        return
    solves: set[str] = set()
    for pv in db.find_parameter_values(entity_class_name="model", parameter_definition_name="solves"):
        value = from_database(pv["value"], pv["type"])
        names = [str(n) for n in getattr(value, "values", [])] if value is not None else []
        if len(names) >= 2:
            solves.update(names)
    for pv in db.find_parameter_values(entity_class_name="solve", parameter_definition_name="contains_solves"):
        if from_database(pv["value"], pv["type"]) is not None:
            solves.add(pv["entity_byname"][0])
    for name in _V79_SPLIT_PERIOD_PARAMETERS:
        for pv in db.find_parameter_values(entity_class_name="solve", parameter_definition_name=name):
            value = from_database(pv["value"], pv["type"])
            if isinstance(value, Map) and any(isinstance(v, Map) for v in value.values):
                solves.add(pv["entity_byname"][0])
    order: dict[str, dict[str, list[str]]] = {}
    for name in ("realized_periods", "invest_periods", "period_timeset"):
        for pv in db.find_parameter_values(entity_class_name="solve", parameter_definition_name=name):
            seen = order.setdefault(pv["entity_byname"][0], {}).setdefault(name, [])
            for d in _v79_period_names(from_database(pv["value"], pv["type"])):
                if d not in seen:
                    seen.append(d)
    for pv in db.find_parameter_values(entity_class_name="solve", parameter_definition_name="years_represented"):
        value = from_database(pv["value"], pv["type"])
        if any(v != int(v) for v in _v79_numbers(value)):
            solves.add(pv["entity_byname"][0])
        by_name = order.get(pv["entity_byname"][0], {})
        realized_invest = [*by_name.get("realized_periods", []), *by_name.get("invest_periods", [])]
        if any(_v79_map_out_of_period_order(value, list(dict.fromkeys(periods)))
               for periods in (realized_invest, [*realized_invest, *by_name.get("period_timeset", [])])):
            solves.add(pv["entity_byname"][0])
    if solves:
        _v77_set_marker(db, "chained_solve_discount_from_run_start", json.dumps({"since": 79, "solves": sorted(solves)}))


def _migrate_v79_schema_batch(db) -> None:
    """All schema changes of database version 79 (v78 -> v79), in one step.

    1. ``constraint_online_coeff`` on unit and connection (:func:`_v79_constraint_online_coeff`).
    2. ``ramp_method`` / ``ramp_cost`` descriptions: ``ramp_cost`` is charged (:func:`_v79_ramp_cost_descriptions`).
    3. ``solve.reduce_unit_detail_method`` / ``reduce_system_detail_method`` (:func:`_v79_reduce_detail_methods`).
    4. ``group.inertia_limit`` / ``non_synchronous_limit`` per time step (:func:`_v79_group_limits_by_time`).
    5. A solve's ``new_stepduration`` combines every time series by its kind: descriptions and the
       behaviour-change markers ``solve_step_series_combined`` / ``energy_margin_adder_per_original_step``
       (:func:`_v79_solve_step_series`).
    6. The capacity margin counts firm and VRE capacity derated by availability and the other entities by their
       flow; ``share_of_peak`` and ``group.capacity_margin_share``; the behaviour-change marker
       ``capacity_margin_counts_by_flow`` (:func:`_v79_capacity_margin`).
    7. ``constraint.constraint_period_method`` (``per_timestep`` / ``period_total``)
       (:func:`_v79_constraint_period_method`).
    8. ``discount_rate`` 0 is 0 % (straight-line annuity), no longer 5 %: the ``discount_rate`` / ``lifetime``
       descriptions and the behaviour-change marker ``discount_rate_zero_is_zero`` (:func:`_v79_discount_rate_zero`).
    9. ``unit__inputNode.startup_input_method`` (``just_input`` / ``startup_and_input`` / ``startup_only``) and
       ``startup_energy`` [MWh per MW started] (:func:`_v79_startup_input`).
    10. An ``invest_method`` that allows investing (retiring) offers the variable also where ``invest_cost``
        (``salvage_value``) is missing or 0: the behaviour-change marker ``invest_without_cost_offered``
        (:func:`_v79_invest_without_cost`).
    11. ``node.unit_of_measure`` (``str``, default ``MWh``, group ``basics``): the unit of measure of the node's
        amounts, a label only (:func:`_v79_node_unit_of_measure`).
    12. Representative periods work in stochastic solves and bind forward only in rolling solves: the
        ``node.storage_binding_method`` / ``timeset.representative_period_weights`` descriptions
        (:func:`_v79_rp_stochastic_descriptions`).
    13. Storage in stochastic solves follows each path (plain ``bind_within_solve`` closes one loop per path, a
        free-start ``bind_forward_only`` fan in the first period starts at one level): the behaviour-change marker
        ``stochastic_storage_paths_tied`` (:func:`_v79_stochastic_storage_paths`).
    14. ``solve.solver_precommand`` is never run: its and the ``solver_arguments`` descriptions
        (:func:`_v79_solver_precommand_descriptions`).
    15. Every solve discounts from the start of the run: the behaviour-change marker
        ``chained_solve_discount_from_run_start`` (:func:`_v79_discount_from_run_start`).

    No value is rewritten.  Each part is idempotent, so a rerun of the step changes nothing.
    """
    _v79_constraint_online_coeff(db)
    _v79_ramp_cost_descriptions(db)
    _v79_reduce_detail_methods(db)
    _v79_group_limits_by_time(db)
    _v79_solve_step_series(db)
    _v79_capacity_margin(db)
    _v79_constraint_period_method(db)
    _v79_discount_rate_zero(db)
    _v79_startup_input(db)
    _v79_invest_without_cost(db)
    _v79_node_unit_of_measure(db)
    _v79_rp_stochastic_descriptions(db)
    _v79_stochastic_storage_paths(db)
    _v79_solver_precommand_descriptions(db)
    _v79_discount_from_run_start(db)
    _commit_step(db, "v79: constraint_online_coeff, ramp_cost, reduce_*_detail_method, time-varying group "
                     "limits, time series combined over a solve's new_stepduration step, capacity margin by "
                     "entity class and share_of_peak, constraint_period_method, discount_rate 0 is 0 %, "
                     "startup_input_method and startup_energy, investment offered without invest_cost, "
                     "node.unit_of_measure, representative periods in stochastic and rolling solves, storage "
                     "along each stochastic path, solver_precommand not run, every solve discounted from the run "
                     "start")


def _v79_adder_has_value(value) -> bool:
    """An ``energy_margin_adder`` value that adds demand: not null and not a constant 0 (a map counts)."""
    if value is None:
        return False
    return not (isinstance(value, (int, float)) and float(value) == 0.0)
