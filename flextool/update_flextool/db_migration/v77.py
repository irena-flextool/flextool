"""Data version 77: the schema batch, the storage start / end rules and the unit-count texts."""
import json
from dataclasses import dataclass

from spinedb_api import from_database, Map, SpineDBAPIError, to_database

from flextool.update_flextool.db_migration.common import (
    _attach_list_defaults,
    _commit_step,
    _describe_pv,
    _detach_list_defaults,
    _ensure_parameter_group,
    _pdef_or_none,
    _remove_items_bounded,
    add_value_list_manual,
    BEHAVIOUR_CHANGE_MARKER_PREFIX,
)
from flextool.update_flextool.db_migration.v50_v55 import _STORAGE_BINDING_METHOD_DESCRIPTION_V55
from flextool.update_flextool.db_migration.v70_v76 import (
    _V70_INPUT_SHARE_MAX_DESCRIPTION,
    _V70_INPUT_SHARE_MIN_DESCRIPTION,
)


# ─── v77: schema batch ─────────────────────────────────────────────────

# Value-type order used when a type list is widened (the union keeps the
# definition's existing types and adds the target ones; this order makes
# the stored tuple independent of which side a type came from).
_V77_TYPE_ORDER = (
    "float", "str", "bool", "duration", "date_time", "array", "time_pattern",
    "time_series", "1d_map", "2d_map", "3d_map", "4d_map", "5d_map",
)

_V77_PERIOD_TYPES = ("float", "1d_map")

# Target type lists (v77 §1.A).  Each is unioned with the definition's
# current list, so a type a DB already allows is never dropped (e.g.
# unit.efficiency keeps its stochastic 3d_map).
_V77_TYPE_TARGETS: dict[tuple[str, str], tuple[str, ...]] = {
    **{(cls, f"{kind}_{bound}_period"): _V77_PERIOD_TYPES
       for cls in ("unit", "node", "connection")
       for kind in ("invest", "retire")
       for bound in ("max", "min")},
    ("group", "invest_max_period"): _V77_PERIOD_TYPES,
    ("group", "invest_min_period"): _V77_PERIOD_TYPES,
    ("group", "co2_max_period"): _V77_PERIOD_TYPES,
    ("unit", "efficiency"): ("float", "1d_map", "2d_map", "3d_map"),
    ("unit", "min_load"): ("float", "1d_map", "2d_map", "3d_map"),
    ("connection", "efficiency"): ("float", "1d_map", "2d_map"),
    ("unit", "efficiency_at_min_load"): ("float", "1d_map", "2d_map"),
    ("unit", "startup_cost"): _V77_PERIOD_TYPES,
    ("connection", "startup_cost"): _V77_PERIOD_TYPES,
    ("profile", "profile"): ("float", "1d_map", "2d_map", "3d_map"),
    ("node", "self_discharge_loss"): ("float", "1d_map", "2d_map"),
    ("group", "capacity_margin_method"): ("str",),
    ("node", "energy_margin_method"): ("str",),
    ("node", "penalty_method"): ("str",),
    ("solve", "dispatch_realization_method"): ("str",),
    ("solve", "scaling"): ("str",),
    ("solve", "stochastic_invest_method"): ("str",),
    ("node", "energy_margin_multiplier"): ("float",),
    ("node", "energy_margin_adder"): ("float", "1d_map", "2d_map"),
    ("solve", "shared_invest_periods"): ("array",),
    ("solve", "shared_operation_periods"): ("array",),
    ("constraint", "constant"): _V77_PERIOD_TYPES,
}


def _unit_count_existing_text(cls: str) -> str:
    """Description of ``existing``: with virtual_unitsize it is the number of units."""
    if cls == "node":
        head, typical = "[MWh or count] Existing storage capacity or number of storage units.", "MWh"
    else:
        head, typical = "[MW or count] Existing capacity or number of units.", "MW"
    return (
        f"{head} If virtual_unitsize is given, then existing is the number of units (capacity = existing × "
        f"virtual_unitsize). Otherwise existing is the existing capacity of a single unit (typically in "
        f"{typical}). Constant or period.")


def _unit_count_vu_text(cls: str) -> str:
    """Description of ``virtual_unitsize``; nodes are in MWh and have no startups."""
    unit = "MWh" if cls == "node" else "MW"
    startups = "" if cls == "node" else "Binary startups commit whole units of this size. "
    return (
        f"[{unit}] Size of one unit. With it, existing is the number of units. {startups}Investments and "
        f"retirements are continuous; their limits are in {unit}. No value: the existing capacity, or 1000 {unit} "
        "without it.")


# Descriptions FlexTool has shipped for ``existing`` / ``virtual_unitsize`` before the
# unit-count texts (FlexTool 4.0 - 4.1 schema and the earlier v77 texts).  A database
# already at v77 has one of these unless the user edited it.
_SUPERSEDED_UNIT_COUNT_TEXTS: dict[tuple[str, str], tuple[str, ...]] = {
    ("unit", "existing"): (
        "[MW] Existing capacity. Constant or Period",
        "[MW] Existing capacity. Constant or period.",
        "[MW] Capacity that already exists. With virtual_unitsize, the number of units is existing / "
        "virtual_unitsize. Constant or period.",
    ),
    ("connection", "existing"): (
        "[MW] Existing capacity. Constant or Period",
        "[MW] Existing capacity. Constant or period.",
        "[MW] Capacity that already exists. With virtual_unitsize, the number of units is existing / "
        "virtual_unitsize. Constant or period.",
    ),
    ("node", "existing"): (
        "[MWh] Existing storage capacity. Constant or Period",
        "[MWh] Existing storage capacity. Constant or period.",
        "[MWh] Storage capacity that already exists. With virtual_unitsize, the number of units is existing / "
        "virtual_unitsize. Constant or period.",
    ),
    ("unit", "virtual_unitsize"): (
        "[MW] Size of a single unit - used for integer investments (lumped investments), minimum loads and "
        "start-up costs. If not given, assumed from the existing unit capacity.",
        "[MW] Size of one unit. Investments, retirements and binary startups move whole units of this size. "
        "No value: the existing capacity, or 1000 MW without it.",
    ),
    ("connection", "virtual_unitsize"): (
        "[MW] Size of single connection - used for integer (lumped) investments.",
        "[MW] Size of one unit. Investments, retirements and binary startups move whole units of this size. "
        "No value: the existing capacity, or 1000 MW without it.",
    ),
    ("node", "virtual_unitsize"): (
        "[MWh] Size of a single storage unit - used for integer investments (lumped investments). If not "
        "given, assumed from the existing storage capacity.",
        "[MWh] Size of one unit. Investments, retirements and binary startups move whole units of this size. "
        "No value: the existing capacity, or 1000 MWh without it.",
    ),
}


def _refresh_unit_count_texts(db) -> None:
    """Rewrite superseded ``existing`` / ``virtual_unitsize`` descriptions of a database
    already at the current version.  Only a description equal to one FlexTool shipped
    (:data:`_SUPERSEDED_UNIT_COUNT_TEXTS`) is replaced; an edited one is kept.  Idempotent."""
    for (cls, name), old_texts in _SUPERSEDED_UNIT_COUNT_TEXTS.items():
        pdef = _pdef_or_none(db, cls, name)
        if pdef is None or pdef["description"] not in old_texts:
            continue
        text = _unit_count_existing_text(cls) if name == "existing" else _unit_count_vu_text(cls)
        db.update_parameter_definition(id=pdef["id"], description=text)


def _v77_texts() -> dict[tuple[str, str], str]:
    """Descriptions written by v77 (owner-reviewed texts, 2026-11-03)."""
    t: dict[tuple[str, str], str] = {}
    for cls in ("unit", "connection"):
        t[(cls, "invest_max_period")] = (
            "[MW] Largest allowed capacity coming online in each period. Existing capacity and retirements do not "
            "change it. With a construction lead time, the investment is decided earlier. Constant or period.")
        t[(cls, "invest_min_period")] = (
            "[MW] Smallest required capacity coming online in each period. Existing capacity and retirements do not "
            "change it. With a construction lead time, the investment is decided earlier. Constant or period.")
        t[(cls, "retire_max_period")] = (
            "[MW] Largest allowed retirement in each period. Counts only what is retired in that period; existing "
            "capacity and investments do not change it. Constant or period.")
        t[(cls, "retire_min_period")] = (
            "[MW] Smallest required retirement in each period. Counts only what is retired in that period; existing "
            "capacity and investments do not change it. Constant or period.")
        t[(cls, "invest_max_total")] = (
            "[MW] Largest allowed total investment over all solves. Every investment decision counts, also "
            "reinvestments and investments that are no longer in service. Retirements and existing capacity do not "
            "reduce it. To limit the capacity in service, use cumulative_max_capacity. Constant.")
        t[(cls, "invest_min_total")] = (
            "[MW] Smallest required total investment over all solves. Every investment decision counts, also "
            "reinvestments and investments that are no longer in service. Retirements and existing capacity do not "
            "reduce it. To limit the capacity in service, use cumulative_max_capacity. Constant.")
        t[(cls, "startup_cost")] = "[CUR/MW] Cost of starting up one MW of capacity. Constant or period."
        t[(cls, "constraint_capacity_coeff")] = (
            f"Puts the {cls}'s capacity in service in each period on the left side of the user constraint (existing "
            "plus investments still in service minus retirements). Coefficient per MW. Capacity in construction is "
            "not included.")
        t[(cls, "constraint_retired_capacity_coeff")] = (
            "Puts the capacity retired in each period on the left side of the user constraint. Coefficient per MW.")
        t[(cls, "constraint_in_construction_capacity_coeff")] = (
            "Puts the capacity in construction in each period on the left side of the user constraint. Capacity is "
            "in construction from the period of the investment decision until it comes online. It is zero without "
            "a construction lead time. Coefficient per MW.")
    t[("node", "invest_max_period")] = (
        "[MWh] Largest allowed storage capacity coming online in each period. Existing capacity and retirements do "
        "not change it. With a construction lead time, the investment is decided earlier. Constant or period.")
    t[("node", "invest_min_period")] = (
        "[MWh] Smallest required storage capacity coming online in each period. Existing capacity and retirements "
        "do not change it. With a construction lead time, the investment is decided earlier. Constant or period.")
    t[("node", "retire_max_period")] = (
        "[MWh] Largest allowed retirement of storage capacity in each period. Counts only what is retired in that "
        "period; existing capacity and investments do not change it. Constant or period.")
    t[("node", "retire_min_period")] = (
        "[MWh] Smallest required retirement of storage capacity in each period. Counts only what is retired in that "
        "period; existing capacity and investments do not change it. Constant or period.")
    t[("node", "invest_max_total")] = (
        "[MWh] Largest allowed total investment over all solves. Every investment decision counts, also "
        "reinvestments and investments that are no longer in service. Retirements and existing capacity do not "
        "reduce it. To limit the capacity in service, use cumulative_max_capacity. Constant.")
    t[("node", "invest_min_total")] = (
        "[MWh] Smallest required total investment over all solves. Every investment decision counts, also "
        "reinvestments and investments that are no longer in service. Retirements and existing capacity do not "
        "reduce it. To limit the capacity in service, use cumulative_max_capacity. Constant.")
    for cls, unit in (("unit", "[MW]"), ("connection", "[MW]"), ("node", "[MWh]")):
        t[(cls, "retire_max_total")] = (
            f"{unit} Largest allowed total retirement over all solves. Investments and existing capacity do not "
            "change it. Constant.")
        t[(cls, "retire_min_total")] = (
            f"{unit} Smallest required total retirement over all solves. Investments and existing capacity do not "
            "change it. Constant.")
        t[(cls, "constraint_cumulative_pre_built_capacity_coeff")] = (
            "Puts the capacity in service at the start of each period on the left side of the user constraint. "
            "This is existing capacity plus capacity that came online in earlier periods, minus earlier "
            "retirements. Capacity coming online in the period itself is in the invested term. Coefficient per "
            f"{unit[1:-1]}.")
        t[(cls, "constraint_invested_capacity_coeff")] = (
            "Puts the capacity that comes online in each period on the left side of the user constraint. Without "
            "a construction lead time, this is the capacity invested in that period. Coefficient per "
            f"{unit[1:-1]}.")
    t[("node", "constraint_capacity_coeff")] = (
        "Puts the node's storage capacity in service in each period on the left side of the user constraint "
        "(existing plus investments still in service minus retirements). Coefficient per MWh. Capacity in "
        "construction is not included.")
    t[("node", "constraint_retired_capacity_coeff")] = (
        "Puts the storage capacity retired in each period on the left side of the user constraint. Coefficient per "
        "MWh.")
    t[("node", "constraint_in_construction_capacity_coeff")] = (
        "Puts the storage capacity in construction in each period on the left side of the user constraint. Capacity "
        "is in construction from the period of the investment decision until it comes online. It is zero without a "
        "construction lead time. Coefficient per MWh.")
    t[("group", "invest_max_period")] = (
        "[MW or MWh] Largest allowed capacity coming online in each period, summed over the group's units and "
        "connections (MW) and storage nodes (MWh). With a construction lead time, the investment is decided "
        "earlier. Constant or period.")
    t[("group", "invest_min_period")] = (
        "[MW or MWh] Smallest required capacity coming online in each period, summed over the group's units and "
        "connections (MW) and storage nodes (MWh). With a construction lead time, the investment is decided "
        "earlier. Constant or period.")
    t[("group", "invest_max_total")] = (
        "[MW or MWh] Largest allowed total investment of the group over all solves. Every investment decision "
        "counts; retirements do not reduce it. Constant.")
    t[("group", "invest_min_total")] = (
        "[MW or MWh] Smallest required total investment of the group over all solves. Every investment decision "
        "counts; retirements do not reduce it. Constant.")
    t[("group", "retire_max_period")] = (
        "[MW or MWh] Largest allowed retirement in each period, summed over the group's members. Needs a retire "
        "…_period option in the group's invest_method. Constant or period.")
    t[("group", "retire_min_period")] = (
        "[MW or MWh] Smallest required retirement in each period, summed over the group's members. Needs a retire "
        "…_period option in the group's invest_method. Constant or period.")
    t[("group", "retire_max_total")] = (
        "[MW or MWh] Largest allowed total retirement of the group over all solves. Needs a retire …_total option "
        "in the group's invest_method. Constant.")
    t[("group", "retire_min_total")] = (
        "[MW or MWh] Smallest required total retirement of the group over all solves. Needs a retire …_total "
        "option in the group's invest_method. Constant.")
    t[("group", "invest_method")] = (
        "How the group limits investment and retirement of its members. not_active: the group sets no limits. "
        "invest / retire / invest_retire: which decisions the limits apply to. _period uses the …_period limits, "
        "_total the …_total limits, _period_total both. no_limit: no limits even if values are set. "
        "cumulative_limits: uses the cumulative capacity limits.")
    t[("group", "co2_max_period")] = (
        "[tCO2/a] Largest annual CO2 emissions in each period. In a rolling solve, each roll may use the part of "
        "the limit its horizon reaches, minus what earlier rolls already emitted in that period. Constant or "
        "period.")
    t[("group", "co2_max_total")] = (
        "[tCO2] Largest CO2 emissions over the whole model horizon. Each period counts its annual emissions times "
        "its years_represented. Rolling solves and chains of solves pass the remaining budget on. Each stochastic "
        "forecast gets the whole remaining budget. Constant.")
    t[("unit", "efficiency")] = (
        "[factor] Efficiency of the unit at full load: outputs / inputs, each flow weighted by its "
        "conversion_flow_coeff (default 1, giving plain sum of outputs / sum of inputs). A min_load_efficiency "
        "unit can have different efficiency below full load (see efficiency_at_min_load). No effect on a unit "
        "without inputs or without outputs. Constant, period, time, period+time or stochastic time.")
    t[("unit", "min_load")] = (
        "[0-1] Minimum load of the unit as a share of its online capacity. Applies only if the unit has online "
        "variables. No value means 0. Constant, period, time, period+time or stochastic time.")
    t[("unit", "efficiency_at_min_load")] = (
        "[e.g. 0.4 means 40%] Efficiency of a min_load_efficiency unit at its minimum load. Fuel use is a straight "
        "line from minimum load (this efficiency) to full load (efficiency). Has no effect when min_load is 0: the "
        "efficiency at zero output is not defined. No value: efficiency applies at minimum load too. Constant, "
        "period, time or period+time.")
    t[("connection", "efficiency")] = (
        "[factor, typically 0-1] Efficiency of the connection. Constant, period, time or period+time.")
    t[("profile", "profile")] = (
        "[factor, typically 0-1] Profile values, e.g. the availability of wind power (capacity factor). Constant, "
        "period, time, period+time or stochastic time.")
    t[("unit", "startup_method")] = (
        "Whether the unit has online variables. linear: any share of the capacity can be online. binary: whole "
        "units only (slower to solve). Online status links to the previous time step; that link stops at a gap in "
        "the timeline.")
    t[("connection", "startup_method")] = (
        "Gives the connection online capacity in both directions. Flows and reserves are limited by the online "
        "capacity, and startup_cost is paid per MW brought online. No minimum load or minimum up/down time. Not "
        "allowed for dc_power_flow_with_angles lines.")
    t[("unit", "min_uptime")] = (
        "[hours] Minimum time the unit stays online after a start. Needs minimum_time_method. The look-back stops "
        "at a gap in the timeline and at the first step of a solve. Constant.")
    t[("unit", "min_downtime")] = (
        "[hours] Minimum time the unit stays offline after a stop. Needs minimum_time_method. The look-back stops "
        "at a gap in the timeline and at the first step of a solve. Constant.")
    for cls in ("unit__inputNode", "unit__outputNode"):
        t[(cls, "ramp_speed_up")] = (
            "[share of capacity per minute] Largest increase of the flow between two time steps. No limit across a "
            "gap in the timeline. Constant.")
        t[(cls, "ramp_speed_down")] = (
            "[share of capacity per minute] Largest decrease of the flow between two time steps. No limit across a "
            "gap in the timeline. Constant.")
        t[(cls, "ramp_method")] = (
            "Choice of ramp method. ramp_limit limits the ramp speed. ramp_cost is not functional yet.")
        t[(cls, "conversion_flow_coeff")] = (
            "[factor, default 1] Fuel-equivalent energy per unit of this flow. It enters the unit's conversion "
            "equation, not the node balance. Applies to simple units too. 0 takes the flow out of the capacity, ramp "
            "and minimum load limits.")
    for cls, unit, cost_unit, what in (("unit", "[MW]", "[CUR/kW]", "capacity"),
                                       ("connection", "[MW]", "[CUR/kW]", "capacity"),
                                       ("node", "[MWh]", "[CUR/kWh]", "storage capacity")):
        t[(cls, "construction_lead_time")] = (
            f"[years] Time from the investment decision until the {what} starts to operate. The start is rounded "
            "to the start of a period (see construction_lead_time_method). Investments whose rounded start is "
            "after the end of the solve or the model horizon are not offered, with a warning. Constant or period. "
            "Default 0.")
        t[(cls, "construction_lead_time_method")] = (
            "How the start of operation (decision year plus construction_lead_time) is rounded to the start of a "
            f"period. immediate: no lead time, the {what} operates from the decision period. closest_seam: the "
            "nearest period start; a tie goes to the later one. previous_seam: the period start before. "
            "next_seam: the period start after. Without a value: immediate when there is no lead time, else "
            "closest_seam.")
        t[(cls, "invest_cost")] = (
            f"{cost_unit} Overnight investment cost of new {what}. It is paid as an annuity over the years the "
            f"{what} operates. Construction time costs with their interest must be included here. Constant or "
            f"period: the value of the period when the {what} is built (comes online) is used.")
        t[(cls, "fixed_cost")] = (
            f"{cost_unit} Annual fixed cost{' for storage' if cls == 'node' else ''}. New {what} pays it only "
            "while it operates. Constant or period.")
        t[(cls, "lifetime")] = (
            f"[years] Operating lifetime. It starts when the {what} starts to operate: with a construction lead "
            "time, after the construction. Used with the interest rate to calculate the annuity. What happens at "
            "the end of the lifetime depends on lifetime_method: reinvest_automatic keeps the "
            f"{what} to the end of the model horizon, reinvest_choice retires it, no_investment retires it once. "
            "Constant or period.")
    t[("node", "invest_forced")] = (
        "[MWh] Forces exactly this much new storage capacity to come online in the period, overriding the "
        "investment optimisation (as setting invest_min and invest_max equal). Constant or period.")
    t[("solve", "invest_periods")] = (
        "Array of periods in which investments can be decided. With a construction lead time, the capacity comes "
        "online later.")
    for cls, unit in (("unit", "[MW]"), ("connection", "[MW]"), ("node", "[MWh]")):
        t[(cls, "virtual_unitsize")] = _unit_count_vu_text(cls)
        t[(cls, "existing")] = _unit_count_existing_text(cls)
    t[("connection", "is_DC")] = (
        "Whether the connection is DC. Its flow into a group with a non-synchronous limit counts as "
        "non-synchronous, in both directions. Default no.")
    t[("group", "has_inertia")] = (
        "Whether the group has an inertia constraint. Without yes, inertia_limit is ignored.")
    t[("group", "inertia_limit")] = (
        "[MWs] Minimum synchronous inertia in the group. Used only when has_inertia is yes. No value means no "
        "requirement. Constant or period.")
    t[("group", "has_non_synchronous")] = (
        "Whether the group has a non-synchronous share limit. Without yes, non_synchronous_limit is ignored.")
    t[("group", "non_synchronous_limit")] = (
        "[share, e.g. 0.8] Largest share of non-synchronous supply in the group, measured against the real demand. "
        "Flows between the group's nodes do not count. Used only when has_non_synchronous is yes; no value means "
        "0. Constant or period.")
    t[("group", "capacity_margin_method")] = (
        "Whether the group needs a capacity margin in investment solves. none: no margin. manual: the group needs "
        "capacity_margin MW above its peak net load.")
    t[("group", "capacity_margin")] = (
        "[MW] Capacity the group needs above its peak net load. Used only with capacity_margin_method manual, and "
        "only in investment periods. No value means 0. Constant or period.")
    t[("reserve__upDown__unit__node", "max_share")] = (
        "[factor] Largest share of the unit's built capacity at this node that can be held as this reserve. At an "
        "input node, the capacity is the unit's full-load input there. Without a value the unit offers no "
        "reserve. Constant.")
    t[("reserve__upDown__connection__node", "max_share")] = (
        "[factor] Largest share of the connection's built capacity that can be held as this reserve at this node. "
        "Constant.")
    t[("reserve__upDown__unit__node", "increase_reserve_ratio")] = (
        "[factor] The reserve requirement grows by this ratio times the flow between the unit and the node, for "
        "upward and downward reserve alike. At an input node, the flow is what the unit draws. Constant.")
    t[("reserve__upDown__connection__node", "increase_reserve_ratio")] = (
        "[factor] The reserve requirement grows by this ratio times the flow between the connection and the node, "
        "in either direction, for upward and downward reserve alike. Constant.")
    t[("reserve__upDown__group", "increase_reserve_ratio")] = (
        "[factor] Only for dynamic reserve methods. The requirement grows by this ratio times the demand of the "
        "group's nodes in each step (MW, real demand). Constant.")
    t[("reserve__upDown__unit__node", "large_failure_ratio")] = (
        "[factor] With a large_failure method, the reserve must cover the loss of this unit's flow times this "
        "ratio. Its own reserve does not count. Constant.")
    t[("reserve__upDown__connection__node", "large_failure_ratio")] = (
        "[factor] With a large_failure method, the reserve must cover the loss of this connection's flow times "
        "this ratio. Its own reserve does not count. Constant.")
    t[("reserve__upDown__group", "reserve_method")] = (
        "Choice of reserve method: no_reserve, timeseries_only, dynamic_only, large_failure_only or their "
        "combinations. Dynamic and large_failure requirements need no separate reservation.")
    t[("group", "new_stepduration")] = (
        "[hours] Balances the energy of the group's nodes once per step of this length instead of every time step, "
        "e.g. daily hydrogen next to hourly electricity. Within the step, energy can move freely between time "
        "steps, and a storage node has one state per step. A unit or connection whose nodes are all in such "
        "groups also uses the longer step: it has one flow, units online and reserve per step, so the model gets "
        "smaller. Its limits, and rows that combine only such units and connections, apply to their average over "
        "the step, and ramp limits apply between steps. Units and connections with a node outside these groups "
        "keep the solve's time step, and so do those with a delay (an info line names them). Listing units or "
        "connections has no effect. Must be a multiple of the solve's step.")
    t[("timeset", "timeset_weights")] = (
        "Weight of each time step (map, index time). Weights how much each step counts in costs, energy and demand "
        "scaling. Normalised per period. Forecast branches use the weights of their period. Other shapes are an "
        "error. Cannot be set on a timeset that also has representative_period_weights (error).")
    t[("timeset", "representative_period_weights")] = (
        "Representative-period weighting map (rank-2 nested Map: base period start timestep -> representative "
        "period start timestep -> weight).  Created by the representative-periods preprocess tool; declares how "
        "much each representative period stands in for each base period.  Its presence enables RP-aware "
        "storage-binding ('*_blended_weights'); its absence silently downgrades those methods to the non-RP "
        "behaviour.  Mutually exclusive with 'timeset_weights' on the same timeset. Not allowed in a stochastic "
        "solve.")
    t[("node", "storage_binding_method")] = (
        _STORAGE_BINDING_METHOD_DESCRIPTION_V55 + " The blended_weights methods do not work with stochastic branches.")
    t[("node", "self_discharge_loss")] = (
        "[e.g. 0.01 means 1% per hour] Share of the stored energy lost each hour. Constant, period, time or "
        "period+time.")
    t[("node", "storage_state_start")] = (
        "[0-1] Storage state at the start of the first solve, as a share of the storage capacity. Needed by "
        "fix_start and fix_start_end. Constant.")
    t[("node", "storage_state_end")] = (
        "[0-1] Storage state at the end of the last solve, as a share of the storage capacity. Needed by fix_end "
        "and fix_start_end. Constant.")
    t[("flowGroup", "max_cumulative_flow")] = (
        "[MW] Largest allowed average flow into the group. A constant gives one limit per solve; a period map "
        "gives one limit per period. Average flow times the weighted hours gives the energy limit. Constant or "
        "period.")
    t[("flowGroup", "min_cumulative_flow")] = (
        "[MW] Smallest required average flow into the group. A constant gives one limit per solve; a period map "
        "gives one limit per period. Average flow times the weighted hours gives the energy limit. Constant or "
        "period.")
    t[("commodity", "price_ladder_annual")] = (
        "Stepped supply curve for price_method price_ladder_annual. Map of tier → price and quantity (optionally "
        "per period). Quantity is per year; a shorter solve gets its share by the hours it covers. quantity = inf "
        "can be used to mark an unlimited last tier.")
    t[("solve", "stochastic_branches")] = (
        "Forecast branches of the solve (map: period, branch, analysis_time, realized → probability weight). "
        "Weights are normalised per start time. At most one branch per start time may be realized: yes. That "
        "branch is not a weighted forecast; it adds a realization solve over the committed time steps. Without "
        "it, the solve reports investments only.")
    t[("solve", "stochastic_invest_method")] = (
        "How stochastic branches invest. none (default): one shared investment plan for all branches. recourse: "
        "each branch also invests at and after its branching period.")
    t[("solve", "dispatch_realization_method")] = (
        "How the committed part of a stochastic solve is realized. within_solve (default): a realized: yes branch "
        "gets a realization solve automatically. separate_dispatch: always runs a separate realized dispatch "
        "solve; needs a realized: yes branch (error otherwise).")
    t[("solve", "shared_operation_periods")] = (
        "Periods where operations are the same in every stochastic branch. No value: the committed time steps. "
        "Empty array: none.")
    t[("solve", "shared_invest_periods")] = (
        "Periods where investments are the same in every stochastic branch (with recourse). No value or empty "
        "array: none.")
    t[("solve", "scaling")] = (
        "Numerical scaling of this solve's LP. full (default): FlexTool rescales variables, constraints and costs "
        "and sets the solver's scaling. basic: no per-quantity rescaling. solver_only: only the solver's own "
        "scaling. off: no scaling at all. Try basic or solver_only if a solve fails numerically. The --scaling run "
        "option overrides it.")
    t[("model", "output_horizon")] = (
        "Debug flag. yes writes one raw parquet file per LP variable and constraint, with every row of every solve "
        "and a realized column. Normal outputs do not change.")
    t[("node", "energy_margin_multiplier")] = (
        "[factor] Multiplier applied to this node's inflow in the investment solve only, when energy_margin_method "
        "= 'inflow_multiplier'. Offsets representative-period VRE optimism (rep-period average VRE availability "
        "exceeds the annual average). 1 = off. e.g. 1.1 builds ~10% more capacity to serve the true annual demand "
        "under the true (lower) annual VRE. Constant.")
    t[("node", "energy_margin_adder")] = (
        "[MWh] Extra demand (negative inflow) added to this node in the investment solve only, when "
        "energy_margin_method = 'inflow_adder'. Raises adequacy headroom without changing dispatch demand. 0 = "
        "off. Constant, period or period+time.")
    t[("constraint", "constant")] = (
        "Right-hand side of the user constraint. A period map gives each period its own value; it must cover every "
        "period of the solve. Constant or period.")
    return t


_V77_CONSTRAINT_CLASS_DESCRIPTION = (
    "A constraint with only capacity terms is written once per period, otherwise once per time step."
)

_V77_GROUP_INVEST_METHODS = (
    "not_active", "invest_no_limit", "invest_period", "invest_total", "invest_period_total",
    "retire_no_limit", "retire_period", "retire_total", "retire_period_total",
    "invest_retire_no_limit", "invest_retire_period", "invest_retire_total",
    "invest_retire_period_total", "cumulative_limits",
)


_V77_UNCONDITIONAL_MARKERS = ("user_constraint_prebuilt_nets_retirements", "dc_power_flow_lines_beside_dc_links",
                              "price_ladder_periods_from_solve_periods", "construction_lead_time_commissioning",
                              "coarse_process_flows", "chained_solve_costs_from_run_start",
                              "stochastic_branch_period_years", "retirement_of_existing_capacity_only",
                              "ramp_branch_point_shutdown_once", "acceptance_ceiling_follows_tolerance")

# Definitions the schema no longer has but old databases may still carry.
_V77_STALE_DEFINITIONS = (
    ("solve", "discount_years"),
    ("unit", "max_relative_capacity"),
    ("node", "max_relative_capacity"),
    ("connection", "max_relative_capacity"),
    ("unit__inputNode", "max_relative_capacity"),
    ("unit__outputNode", "max_relative_capacity"),
    ("node", "storage_include_solve_fix_method"),
    ("unit__inputNode", "capacity_max_coeff"),
    ("unit__inputNode", "capacity_min_coeff"),
)


def _v77_union_types(existing, target) -> tuple[str, ...]:
    """``existing ∪ target`` in :data:`_V77_TYPE_ORDER` order (unknown
    names last, in first-seen order)."""
    seen: list[str] = []
    for name in (*(existing or ()), *target):
        if name not in seen:
            seen.append(name)
    order = {n: i for i, n in enumerate(_V77_TYPE_ORDER)}
    return tuple(sorted(seen, key=lambda n: (order.get(n, len(order)), seen.index(n))))


def _v77_types_and_descriptions(db) -> None:
    """Widen the type lists of §1.A (union with the current list) and write
    the v77 descriptions.  Definitions a DB does not have are skipped."""
    texts = _v77_texts()
    for (cls, name), target in _V77_TYPE_TARGETS.items():
        pdef = _pdef_or_none(db, cls, name)
        if pdef is None:
            continue
        union = _v77_union_types(pdef["parameter_type_list"], target)
        if tuple(pdef["parameter_type_list"] or ()) != union:
            db.update_parameter_definition(id=pdef["id"], parameter_type_list=union)
    for (cls, name), text in texts.items():
        pdef = _pdef_or_none(db, cls, name)
        if pdef is None or pdef["description"] == text:
            continue
        db.update_parameter_definition(id=pdef["id"], description=text)
    try:
        constraint_class = db.item(db.mapped_table("entity_class"), name="constraint")
    except SpineDBAPIError:
        constraint_class = None
    if constraint_class and constraint_class["description"] != _V77_CONSTRAINT_CLASS_DESCRIPTION:
        db.update_item("entity_class", id=constraint_class["id"], description=_V77_CONSTRAINT_CLASS_DESCRIPTION)


def _v77_new_definitions(db) -> None:
    """Group retirement limits and the user-constraint capacity, retired and
    in-construction coefficients (default None, no value list)."""
    texts = _v77_texts()
    new_defs: list[tuple[str, str, tuple[str, ...], str, str, int]] = []
    if db.get_item("entity_class", name="group"):
        for bound in ("max", "min"):
            new_defs.append(("group", f"retire_{bound}_period", _V77_PERIOD_TYPES, "retirement", "a3b08c", 25))
            new_defs.append(("group", f"retire_{bound}_total", ("float",), "retirement", "a3b08c", 25))
    for cls in ("unit", "node", "connection"):
        if not db.get_item("entity_class", name=cls):
            continue
        for name in ("constraint_capacity_coeff", "constraint_retired_capacity_coeff",
                     "constraint_in_construction_capacity_coeff"):
            new_defs.append((cls, name, ("1d_map",), "constraint", "a36784", 70))
    for cls, name, types, group, color, priority in new_defs:
        _ensure_parameter_group(db, group, color, priority)
        fields = {
            "parameter_type_list": types,
            "parameter_group_name": group,
            "description": texts[(cls, name)],
        }
        pdef = _pdef_or_none(db, cls, name)
        if pdef is None:
            db.add_parameter_definition(entity_class_name=cls, name=name, **fields)
        elif any(pdef[k] != v for k, v in fields.items()):
            db.update_parameter_definition(id=pdef["id"], **fields)


def _v77_group_invest_methods(db) -> None:
    """Give ``group.invest_method`` its own value list, where ``not_active``
    replaces ``not_allowed``.

    A value bound to a list is stored as a reference to one list member, and
    spinedb_api refuses to switch the list of a definition that has values.
    So: read every group value, add the new list, remove the values, switch
    the definition's list and default, and write every value again
    (``not_allowed`` → ``not_active``, others unchanged) so each reference
    resolves in the new list.
    """
    pdef = _pdef_or_none(db, "group", "invest_method")
    if pdef is None:
        return
    existing_members = {
        lv["parsed_value"] for lv in db.get_items("list_value", parameter_value_list_name="group_invest_methods")
    }
    missing = [m for m in _V77_GROUP_INVEST_METHODS if m not in existing_members]
    if missing:
        add_value_list_manual(db, [["group_invest_methods", m] for m in missing])
    default_val, default_type = to_database("not_active")
    if pdef["parameter_value_list_name"] == "group_invest_methods":
        if from_database(pdef["default_value"], pdef["default_type"]) != "not_active":
            db.update_parameter_definition(id=pdef["id"], default_value=default_val, default_type=default_type)
        return
    values = [
        (tuple(pv["entity_byname"]), pv["alternative_name"], pv["parsed_value"], pv["id"])
        for pv in db.get_items("parameter_value", entity_class_name="group",
                               parameter_definition_name="invest_method")
    ]
    if values:
        db.remove_items("parameter_value", *[v[3] for v in values])
    db.update_parameter_definition(
        id=pdef["id"], parameter_value_list_name="group_invest_methods",
        default_value=default_val, default_type=default_type)
    renamed = []
    for byname, alternative, parsed, _id in values:
        new_value = "not_active" if parsed == "not_allowed" else parsed
        if new_value != parsed:
            renamed.append(f"group {'/'.join(byname)} [{alternative}]")
        val, typ = to_database(new_value)
        db.add_parameter_value(
            entity_class_name="group", entity_byname=byname, parameter_definition_name="invest_method",
            alternative_name=alternative, value=val, type=typ)
    if renamed:
        print("v77: group invest_method not_allowed is now called not_active for: " + ", ".join(renamed))


def _v77_behaviour_markers(db) -> None:
    """Record the behaviour changes this DB predates (database metadata,
    not alternative-scoped)."""
    for change_id in _V77_UNCONDITIONAL_MARKERS:
        name = BEHAVIOUR_CHANGE_MARKER_PREFIX + change_id
        if not db.get_item("metadata", name=name, value="77"):
            db.add_item("metadata", name=name, value="77")


def _v77_drop_stale_definitions(db) -> None:
    """Drop definitions the schema no longer has; list every value they held.

    ``unit__inputNode.capacity_{max,min}_coeff`` were renamed
    ``input_share_{max,min}`` in v70.  A DB that carries the old name has a
    collided rename (both names present) or never had the new one:

    * new name present: ``capacity_max_coeff`` values move to
      ``input_share_max`` where that (entity, alternative) has no value of its
      own; ``capacity_min_coeff`` values are deleted (their old meaning is
      unrelated to the input share, as in v70); then the old definition goes.
    * new name absent: the definition is renamed as v70 does (values of
      ``capacity_max_coeff`` kept, those of ``capacity_min_coeff`` deleted).

    The ``solve_mode`` value ``single_shot`` (the engine runs it as
    ``single_solve``) becomes ``single_solve`` and leaves the value list.
    """
    zero_val, zero_type = to_database(0.0)
    renames = {
        ("unit__inputNode", "capacity_max_coeff"): ("input_share_max", _V70_INPUT_SHARE_MAX_DESCRIPTION, None),
        ("unit__inputNode", "capacity_min_coeff"): ("input_share_min", _V70_INPUT_SHARE_MIN_DESCRIPTION,
                                                    (zero_val, zero_type)),
    }
    for cls, name in _V77_STALE_DEFINITIONS:
        pdef = _pdef_or_none(db, cls, name)
        if pdef is None:
            continue
        values = list(db.find_parameter_values(entity_class_name=cls, parameter_definition_name=name))
        keep_values = name == "capacity_max_coeff"
        rename = renames.get((cls, name))
        if rename is not None and _pdef_or_none(db, cls, rename[0]) is None:
            new_name, description, default = rename
            if not keep_values and values:
                print(f"v77: deleted the old {cls} {name} values (now {new_name}, a different meaning; "
                      "re-author if needed):")
                for pv in values:
                    print("  " + _describe_pv(pv))
                _remove_items_bounded(db, "parameter_value", [pv["id"] for pv in values],
                                      f"v77: removed the {cls}.{name} values")
            fields = {"description": description}
            if default is not None:
                fields["default_value"], fields["default_type"] = default
            db.update_parameter_definition(id=pdef["id"], name=new_name, **fields)
            continue
        moved: list = []
        if rename is not None and keep_values and values:
            taken = {(tuple(pv["entity_byname"]), pv["alternative_name"])
                     for pv in db.find_parameter_values(entity_class_name=cls,
                                                        parameter_definition_name=rename[0])}
            for pv in values:
                key = (tuple(pv["entity_byname"]), pv["alternative_name"])
                if key in taken:
                    continue
                db.add_parameter_value(
                    entity_class_name=cls, entity_byname=key[0], parameter_definition_name=rename[0],
                    alternative_name=key[1], value=pv["value"], type=pv["type"])
                moved.append(pv)
        if moved:
            print(f"v77: moved {cls} {name} values to {rename[0]} (its name since v70):")
            for pv in moved:
                print("  " + _describe_pv(pv))
        moved_ids = {pv["id"] for pv in moved}
        dropped = [pv for pv in values if pv["id"] not in moved_ids]
        if dropped:
            print(f"v77: {cls}.{name} is no longer a FlexTool parameter; deleted its values:")
            for pv in dropped:
                print("  " + _describe_pv(pv))
        _remove_items_bounded(db, "parameter_definition", [pdef["id"]], f"v77: removed {cls}.{name}")
    # The value list of the dropped storage_include_solve_fix_method, when nothing else uses it.
    if db.get_item("parameter_value_list", name="storage_include_solve_fix_method") and not any(
            pd["parameter_value_list_name"] == "storage_include_solve_fix_method"
            for pd in db.get_items("parameter_definition")):
        db.remove_item("parameter_value_list",
                       db.get_item("parameter_value_list", name="storage_include_solve_fix_method")["id"])

    shots = [pv for pv in db.find_parameter_values(entity_class_name="solve", parameter_definition_name="solve_mode")
             if pv["parsed_value"] == "single_shot"]
    if shots:
        print("v77: solve_mode single_shot is now single_solve (it already ran as one) for:")
        for pv in shots:
            print("  " + _describe_pv(pv))
        val, typ = to_database("single_solve")
        for pv in shots:
            db.update_parameter_value(id=pv["id"], value=val, type=typ)
    pending = _detach_list_defaults(db, "solve_mode", {"single_shot": "single_solve"})
    for lv in db.get_items("list_value", parameter_value_list_name="solve_mode"):
        if lv["parsed_value"] == "single_shot":
            db.remove_items("list_value", lv["id"])
    _attach_list_defaults(db, pending)


_V77_STORAGE_START_END_DESCRIPTION = (
    "Whether the storage state is fixed at the start and/or end of the whole model. fix_nothing (default): not "
    "fixed. fix_start, fix_end and fix_start_end use storage_state_start and storage_state_end, which must then "
    "have a value.")
_V77_FIXES = {"start": frozenset(("fix_start", "fix_start_end")), "end": frozenset(("fix_end", "fix_start_end"))}
_V77_NO_SCENARIO = "(all alternatives)"


@dataclass(frozen=True)
class StorageFixPlan:
    """What the version 77 upgrade does to the explicit storage fix methods without their value.

    ``methods``: ``(node, alternative, old method, new method)`` rewrites; ``values``: ``(node, alternative,
    parameter)`` gets the value 0; ``undecided``: ``(node, alternative, method, scenarios)`` left for the person
    (the run-time error names them).  ``records``: ``(node, alternative, old method, what was written)``."""
    methods: tuple = ()
    values: tuple = ()
    undecided: tuple = ()
    records: tuple = ()


def _v77_winner(alternatives, table, node):
    for alternative in reversed(alternatives):
        if (node, alternative) in table:
            return alternative
    return None


def _v77_storage_behaviour(scenario_alts, node, part, methods, values, *, new: bool, off=frozenset()):
    """How *node* starts (``part="start"``) or ends in one scenario: ``"free"``, ``("own", value)`` or, under the
    version 77 rule (*new*), ``"error"`` for a fix without its value.  Before version 77 such a fix did nothing
    when no node of the scenario had a value for that part, else it fixed the node at 0.  *off*: the nodes
    switched off in the scenario (their values are not read)."""
    winner = _v77_winner(scenario_alts, methods, node)
    if winner is None or methods[(node, winner)] not in _V77_FIXES[part]:
        return "free"
    own = _v77_winner(scenario_alts, values, node)
    if own is not None:
        return ("own", values[(node, own)])
    if new:
        return "error"
    anyone = any(_v77_winner(scenario_alts, values, other) is not None
                 for other in {k[0] for k in values} - set(off))
    return ("own", 0.0) if anyone else "free"


def classify_storage_fix(methods, starts, ends, scenarios, inactive=frozenset()) -> StorageFixPlan:
    """Decide, per stored ``storage_start_end_method`` value, what the version 77 upgrade writes.

    *methods* ``{(node, alternative): method}`` (stored values only), *starts* / *ends* ``{(node, alternative):
    value}`` of ``storage_state_start`` / ``storage_state_end``, *scenarios* ``{scenario: [alternative, ...]}``
    lowest rank first (the last one wins), *inactive* ``{(scenario, node)}`` nodes switched off in a scenario
    (their values are not read there).  A database without scenarios is passed as one scenario of all
    alternatives.

    For a value v = (n, A) let W(v) be the scenarios where A gives n's method.  Separately for the start and the
    end (when the method fixes it): (1) n's own value present in every scenario of W(v) → kept; (2) missing, and
    no node has a value in every scenario (the fix did nothing) → the part is dropped from the method
    (fix_start → fix_nothing, fix_start_end → fix_end, ...); (3) missing, and some node has a value in every
    scenario (the fix pinned n at 0) → the value 0 is added for n in A; (4) otherwise → left (undecided).  Then
    every scenario is re-resolved with all rewrites applied; a node whose start or end would change in any
    scenario has its rewrites dropped (undecided).  So the rewrites keep every result unchanged.
    """
    tables = {"start": starts, "end": ends}
    params = {"start": "storage_state_start", "end": "storage_state_end"}
    nodes = sorted({k[0] for k in methods})
    off = {s: frozenset(n for sc, n in inactive if sc == s) for s in scenarios}

    def scen_of(node, alternative):
        return [s for s, alts in scenarios.items()
                if (s, node) not in inactive and alternative in alts and _v77_winner(alts, methods, node) == alternative]

    candidates: dict[tuple, tuple] = {}
    undecided: dict[tuple, tuple] = {}
    for (node, alternative), method in sorted(methods.items()):
        where = scen_of(node, alternative)
        if not where:
            continue
        drop: list[str] = []
        add: list[str] = []
        left = False
        for part in ("start", "end"):
            if method not in _V77_FIXES[part]:
                continue
            table = tables[part]
            own = [_v77_winner(scenarios[s], table, node) is not None for s in where]
            if all(own):
                continue
            old = [_v77_storage_behaviour(scenarios[s], node, part, methods, table, new=False, off=off[s])
                   for s in where]
            if not any(own) and all(b == "free" for b in old):
                drop.append(part)
            elif not any(own) and all(b == ("own", 0.0) for b in old):
                add.append(part)
            else:
                left = True
        if left:
            undecided[(node, alternative)] = (method, tuple(where))
        elif drop or add:
            keep = [p for p in ("start", "end") if method in _V77_FIXES[p] and p not in drop]
            new_method = {("start", "end"): "fix_start_end", ("start",): "fix_start",
                          ("end",): "fix_end"}.get(tuple(keep), "fix_nothing")
            candidates[(node, alternative)] = (method, new_method, tuple(add), tuple(where))

    while True:
        new_methods = dict(methods)
        new_tables = {p: dict(t) for p, t in tables.items()}
        for (node, alternative), (_old, new_method, add, _w) in candidates.items():
            new_methods[(node, alternative)] = new_method
            for part in add:
                new_tables[part][(node, alternative)] = 0.0
        changed: dict[str, set] = {}
        for s, alts in scenarios.items():
            for node in nodes:
                if (s, node) in inactive:
                    continue
                for part in ("start", "end"):
                    before = _v77_storage_behaviour(alts, node, part, methods, tables[part], new=False, off=off[s])
                    after = _v77_storage_behaviour(alts, node, part, new_methods, new_tables[part], new=True,
                                                   off=off[s])
                    if after != before and after != "error":
                        changed.setdefault(node, set()).add(s)
        hit = [k for k in candidates if k[0] in changed]
        if not hit:
            break
        for key in hit:
            method, _new, _add, where = candidates.pop(key)
            undecided[key] = (method, tuple(sorted(set(where) | changed[key[0]])))

    rewrites, values, records = [], [], []
    for (node, alternative), (old, new_method, add, _w) in sorted(candidates.items()):
        written = []
        if new_method != old:
            rewrites.append((node, alternative, old, new_method))
            written.append(new_method)
        for part in add:
            values.append((node, alternative, params[part]))
            written.append(f"{params[part]} = 0")
        records.append((node, alternative, old, " and ".join(written)))
    return StorageFixPlan(
        methods=tuple(rewrites), values=tuple(values),
        undecided=tuple((n, a, m, tuple(sorted(w))) for (n, a), (m, w) in sorted(undecided.items())),
        records=tuple(records))


def _v77_default_behaviour(scenario_alts, node, part, methods, values, default, off=frozenset()):
    """How *node* starts (``part="start"``) or ends in one scenario when a node without a stored method takes
    *default*: ``"free"``, ``("own", value)`` or ``"error"`` (a fix without its value under the version 77 rule,
    i.e. *default* ``fix_nothing``).  Before version 77 (any other *default*) such a fix pinned the node at 0
    when some node of the scenario had a value for that part, else it did nothing."""
    winner = _v77_winner(scenario_alts, methods, node)
    method = methods[(node, winner)] if winner is not None else default
    if method not in _V77_FIXES[part]:
        return "free"
    own = _v77_winner(scenario_alts, values, node)
    if own is not None:
        return ("own", values[(node, own)])
    if default == "fix_nothing":
        return "error"
    anyone = any(_v77_winner(scenario_alts, values, other) is not None
                 for other in {k[0] for k in values} - set(off))
    return ("own", 0.0) if anyone else "free"


def classify_storage_default(methods, starts, ends, scenarios, old_default, inactive=frozenset()) -> tuple:
    """The ``(node, alternative)`` where the version 77 upgrade writes the old default *old_default* as an
    explicit ``storage_start_end_method``, so that the node keeps its results under the new default
    ``fix_nothing``.

    Arguments as :func:`classify_storage_fix`.  A candidate is a node that has no stored method in a scenario
    but has its own value there for a part the old default fixes; the write goes into the alternative that
    gives that value.  A node's writes are kept only when, in every scenario where a written value gives the
    node's method, the node starts and ends as it did under the old default (so a write never overrides a
    stored method of a lower alternative that behaved differently).  Nodes without an own value are left: they
    start free now and the run names them (``storage_default_fix_nothing``).
    """
    if old_default not in _V77_FIXES["start"] | _V77_FIXES["end"]:
        return ()
    tables = {"start": starts, "end": ends}
    off = {s: frozenset(n for sc, n in inactive if sc == s) for s in scenarios}
    parts = [p for p in ("start", "end") if old_default in _V77_FIXES[p]]
    candidates: dict[str, set] = {}
    for s, alts in scenarios.items():
        for node in sorted({k[0] for p in parts for k in tables[p]}):
            if (s, node) in inactive or _v77_winner(alts, methods, node) is not None:
                continue
            own = _v77_winner(alts, tables[parts[0]], node)
            if own is not None:
                candidates.setdefault(node, set()).add(own)
    writes = []
    for node, alternatives in sorted(candidates.items()):
        new_methods = dict(methods)
        new_methods.update({(node, a): old_default for a in alternatives})
        keep = True
        for s, alts in scenarios.items():
            if (s, node) in inactive or _v77_winner(alts, new_methods, node) not in alternatives:
                continue
            for part in ("start", "end"):
                before = _v77_default_behaviour(alts, node, part, methods, tables[part], old_default, off[s])
                after = _v77_default_behaviour(alts, node, part, new_methods, tables[part], "fix_nothing", off[s])
                if after != before:
                    keep = False
        if keep:
            writes.extend((node, a) for a in sorted(alternatives))
    return tuple(writes)


def _v77_storage_inputs(db):
    """``classify_storage_fix`` arguments read from *db*."""
    def stored(name):
        return {(pv["entity_name"], pv["alternative_name"]): pv["parsed_value"]
                for pv in db.find_parameter_values(entity_class_name="node", parameter_definition_name=name)}
    methods = stored("storage_start_end_method") if _pdef_or_none(db, "node", "storage_start_end_method") else {}
    starts = stored("storage_state_start") if _pdef_or_none(db, "node", "storage_state_start") else {}
    ends = stored("storage_state_end") if _pdef_or_none(db, "node", "storage_state_end") else {}
    scenarios = _scenario_alternatives(db)
    inactive = _inactive_in_scenarios(db, "node", {k[0] for k in (*methods, *starts, *ends)}, scenarios)
    return methods, starts, ends, scenarios, inactive


def _scenario_alternatives(db) -> dict[str, list[str]]:
    """``{scenario: [alternative, ...]}``, lowest rank first (the last one wins).  A database without scenarios is
    read as one pseudo-scenario of all alternatives (:data:`_V77_NO_SCENARIO`)."""
    scenarios: dict[str, list[str]] = {}
    for scen in db.get_items("scenario"):
        rows = sorted(db.get_items("scenario_alternative", scenario_name=scen["name"]), key=lambda r: r["rank"])
        scenarios[scen["name"]] = [r["alternative_name"] for r in rows]
    if not scenarios:
        scenarios[_V77_NO_SCENARIO] = [a["name"] for a in db.get_items("alternative")]
    return scenarios


def _inactive_in_scenarios(db, entity_class: str, entities, scenarios) -> frozenset:
    """``{(scenario, entity)}``: the *entities* of *entity_class* switched off in a scenario (``entity_alternative``
    of the highest alternative that has one, else the class's ``active_by_default``)."""
    klass = db.get_item("entity_class", name=entity_class)
    by_default = bool(klass["active_by_default"]) if klass else True
    activity = {(ea["entity_byname"][0], ea["alternative_name"]): bool(ea["active"])
                for ea in db.get_items("entity_alternative", entity_class_name=entity_class)}
    inactive = set()
    for s, alts in scenarios.items():
        for entity in entities:
            winner = _v77_winner(alts, activity, entity)
            if not (activity[(entity, winner)] if winner is not None else by_default):
                inactive.add((s, entity))
    return frozenset(inactive)


def _v77_set_marker(db, change_id: str, value: str) -> None:
    name = BEHAVIOUR_CHANGE_MARKER_PREFIX + change_id
    existing = [md for md in db.get_items("metadata") if md["name"] == name]
    if existing:
        if existing[0]["value"] != value:
            db.update_item("metadata", id=existing[0]["id"], value=value)
        return
    db.add_item("metadata", name=name, value=value)


def _v77_storage_fix_rewrite(db) -> None:
    """Rewrite the explicit storage fix methods that have no value to fix at, where results stay unchanged
    (:func:`classify_storage_fix`), and record them in the marker ``storage_fix_method_rewritten``
    (``{"since": 77, "rewrites": [[node, alternative, old, written], ...], "undecided": [[node, alternative,
    method, [scenario, ...]], ...]}``)."""
    methods, starts, ends, scenarios, inactive = _v77_storage_inputs(db)
    if not any(m in _V77_FIXES["start"] | _V77_FIXES["end"] for m in methods.values()):
        return
    plan = classify_storage_fix(methods, starts, ends, scenarios, inactive)
    ids = {(pv["entity_name"], pv["alternative_name"]): pv["id"]
           for pv in db.find_parameter_values(entity_class_name="node",
                                              parameter_definition_name="storage_start_end_method")}
    for node, alternative, _old, new_method in plan.methods:
        val, typ = to_database(new_method)
        db.update_parameter_value(id=ids[(node, alternative)], value=val, type=typ)
    zero_val, zero_type = to_database(0.0)
    for node, alternative, param in plan.values:
        db.add_parameter_value(entity_class_name="node", entity_byname=(node,), parameter_definition_name=param,
                               alternative_name=alternative, value=zero_val, type=zero_type)
    if plan.records:
        print("v77: storage_start_end_method without its storage_state_start / storage_state_end is now an error; "
              "rewritten to what it did before (results unchanged):")
        for node, alternative, old, written in plan.records:
            print(f"  node {node} [{alternative}]: {old} -> {written}")
    if plan.undecided:
        print("v77: left for you to decide (scenarios used the value differently; the run stops with an error "
              "naming them):")
        for node, alternative, method, where in plan.undecided:
            print(f"  node {node} [{alternative}]: {method} (scenarios {', '.join(where)})")
    _v77_record_storage_rewrites(db, plan.records, [[n, a, m, list(w)] for n, a, m, w in plan.undecided])


def _v77_record_storage_rewrites(db, records, undecided=None) -> None:
    """Merge *records* ``(node, alternative, old, written)`` into the marker ``storage_fix_method_rewritten``;
    *undecided* replaces its undecided list (None keeps the stored one).  Written when either list is
    non-empty."""
    _merge_marker_lists(db, "storage_fix_method_rewritten", 77, "rewrites", records, undecided)


def _merge_marker_lists(db, change_id: str, since: int, key: str, records, undecided=None) -> None:
    """Merge *records* into the list *key* of the marker *change_id* (``{"since": since, key: [...], "undecided":
    [...]}``); *undecided* replaces its undecided list (None keeps the stored one).  Written when either list is
    non-empty."""
    name = BEHAVIOUR_CHANGE_MARKER_PREFIX + change_id
    previous = [md for md in db.get_items("metadata") if md["name"] == name]
    stored: dict = {}
    if previous:
        try:
            stored = json.loads(previous[0]["value"])
        except (TypeError, ValueError):
            stored = {}
        if not isinstance(stored, dict):
            stored = {}
    merged = [list(r) for r in stored.get(key, [])]
    for record in records:
        if list(record) not in merged:
            merged.append(list(record))
    if undecided is None:
        undecided = [list(u) for u in stored.get("undecided", [])]
    if merged or undecided:
        _v77_set_marker(db, change_id, json.dumps({"since": since, key: merged, "undecided": undecided}))


def _v77_storage_default_explicit(db) -> None:
    """Write the old default ``storage_start_end_method`` explicitly on the nodes that relied on it with their
    own start (or end) value (:func:`classify_storage_default`), so they keep their results under the new
    default ``fix_nothing``; record them in the marker ``storage_fix_method_rewritten`` (old method ``default
    <old default>``).  Runs before :func:`_v77_storage_default` changes the default."""
    pdef = _pdef_or_none(db, "node", "storage_start_end_method")
    if pdef is None or pdef["default_value"] is None:
        return
    default = from_database(pdef["default_value"], pdef["default_type"])
    if default not in _V77_FIXES["start"] | _V77_FIXES["end"]:
        return
    methods, starts, ends, scenarios, inactive = _v77_storage_inputs(db)
    writes = classify_storage_default(methods, starts, ends, scenarios, default, inactive)
    if not writes:
        return
    val, typ = to_database(default)
    for node, alternative in writes:
        db.add_parameter_value(entity_class_name="node", entity_byname=(node,),
                               parameter_definition_name="storage_start_end_method", alternative_name=alternative,
                               value=val, type=typ)
    print(f"v77: the default storage_start_end_method is now fix_nothing; {default} written on the nodes that "
          "used the old default with their own start value (results unchanged):")
    for node, alternative in writes:
        print(f"  node {node} [{alternative}]: default {default} -> {default}")
    _v77_record_storage_rewrites(db, [(node, alternative, f"default {default}", default)
                                      for node, alternative in writes])


def _v77_storage_default(db) -> None:
    """``node.storage_start_end_method``: default ``fix_nothing`` (from a default that fixed something) and the
    version 77 description.  A default that fixed something writes the marker ``storage_default_fix_nothing``
    (value ``77``; for an old default other than ``fix_start`` the JSON ``{"since": 77, "old_default": ...}``).
    A None default stays None (it already fixed nothing)."""
    pdef = _pdef_or_none(db, "node", "storage_start_end_method")
    if pdef is None:
        return
    if pdef["description"] != _V77_STORAGE_START_END_DESCRIPTION:
        db.update_parameter_definition(id=pdef["id"], description=_V77_STORAGE_START_END_DESCRIPTION)
    default = (from_database(pdef["default_value"], pdef["default_type"])
               if pdef["default_value"] is not None else None)
    if default not in _V77_FIXES["start"] | _V77_FIXES["end"]:
        return
    _v77_set_marker(db, "storage_default_fix_nothing",
                    "77" if default == "fix_start" else json.dumps({"since": 77, "old_default": default}))
    val, typ = to_database("fix_nothing")
    db.update_parameter_definition(id=pdef["id"], default_value=val, default_type=typ)


def _v77_map_key_paths(value, prefix=()):
    """Every key path (outer to inner) to a leaf of a parsed Map value."""
    if not isinstance(value, Map):
        yield prefix
        return
    for key, child in zip(value.indexes, value.values):
        yield from _v77_map_key_paths(child, (*prefix, str(key)))


def _v77_ladder_tier_marker(db) -> None:
    """Write the marker ``ladder_tier_labels_as_written`` when a price ladder has a tier label that earlier
    releases read as another tier: a label that is not a whole number written plainly (``01`` was read as
    ``1``, ``a`` was dropped).  The tier is the level above price / quantity."""
    commodities: set[str] = set()
    for name in ("price_ladder_annual", "price_ladder_cumulative"):
        for pv in db.find_parameter_values(entity_class_name="commodity", parameter_definition_name=name):
            if pv["type"] != "map":
                continue
            for path in _v77_map_key_paths(from_database(pv["value"], pv["type"])):
                if len(path) < 2:
                    continue
                tier = path[-2]
                try:
                    plain = str(int(tier)) == tier
                except ValueError:
                    plain = False
                if not plain:
                    commodities.add(pv["entity_byname"][0])
    if commodities:
        _v77_set_marker(db, "ladder_tier_labels_as_written", json.dumps({"since": 77, "commodities":
                                                                          sorted(commodities)}))


def _v77_unnamed_map_depth(value) -> int | None:
    """The depth of a parsed Map value whose levels all lack an index name (``x`` / empty), else None."""
    depth = 0
    cur = value
    while isinstance(cur, Map):
        if (cur.index_name or "x").strip().lower() != "x":
            return None
        depth += 1
        if not cur.values:
            break
        cur = cur.values[0]
    return depth or None


def _v77_unnamed_mixed_depth_marker(db) -> None:
    """Write the marker ``unnamed_maps_read_per_depth`` when some parameter has maps without index names at
    one level on some entities and at more levels on others: earlier releases read such one-level maps into
    the deeper maps' first level (a time map lost its values).  The run names the entities whose one-level
    map is now read over time."""
    depths: dict[tuple[str, str], set[int]] = {}
    for pv in db.get_items("parameter_value"):
        if pv["type"] != "map":
            continue
        depth = _v77_unnamed_map_depth(from_database(pv["value"], pv["type"]))
        if depth is not None:
            depths.setdefault((pv["entity_class_name"], pv["parameter_definition_name"]), set()).add(depth)
    mixed = sorted(f"{cls}.{name}" for (cls, name), ds in depths.items() if 1 in ds and max(ds) > 1)
    if mixed:
        _v77_set_marker(db, "unnamed_maps_read_per_depth", json.dumps({"since": 77, "parameters": mixed}))


def _migrate_v77_schema_batch(db) -> None:
    """Schema batch v77 (v76 -> v77).

    * Type lists widened to what the engine reads (period maps for the
      investment / retirement / CO2 period limits and startup costs, period +
      time maps for efficiencies, min_load, profiles and self-discharge, a
      period map for the user-constraint constant, and the missing type
      lists of ten definitions).  Each list is the union with the current
      one, so no allowed type is dropped.
    * Descriptions rewritten (owner-reviewed texts).
    * New: group ``retire_{max,min}_{period,total}`` and the unit / node /
      connection ``constraint_capacity_coeff``,
      ``constraint_retired_capacity_coeff`` and
      ``constraint_in_construction_capacity_coeff``.
    * ``group.invest_method`` gets the list ``group_invest_methods``:
      ``not_allowed`` is ``not_active`` there (also the new default).
    * Behaviour-change markers (database metadata) for the changes this DB
      predates (the price-ladder tier-label marker only when a ladder has a
      tier label that is not a plain whole number; the unnamed-map marker
      only when a parameter has unnamed maps of one and of more levels).
    * Stale definitions dropped (their values listed), ``single_shot`` →
      ``single_solve``.
    * Storage start / end: explicit fix methods without their value are
      rewritten to what they did; nodes that used the old default
      ``fix_start`` with their own start value get it written explicitly;
      the default becomes ``fix_nothing`` (results unchanged except for
      default nodes without an own value, which a notice names).

    Idempotent: re-running finds nothing to change.
    """
    _v77_types_and_descriptions(db)
    _v77_new_definitions(db)
    _v77_group_invest_methods(db)
    _v77_behaviour_markers(db)
    _v77_ladder_tier_marker(db)
    _v77_unnamed_mixed_depth_marker(db)
    _v77_drop_stale_definitions(db)
    _v77_storage_fix_rewrite(db)
    _v77_storage_default_explicit(db)
    _v77_storage_default(db)
    _commit_step(db,
        "v77: schema batch — widened type lists, owner-reviewed descriptions, group retirement limits, "
        "user-constraint capacity coefficients, group_invest_methods (not_active), behaviour-change markers, "
        "stale definitions dropped, single_shot -> single_solve."
    )
