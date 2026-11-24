"""The behaviour-change notices (``flextool.engine._behaviour_changes``) as codes of the registry
:mod:`flextool.run_warnings.registry` (W-0600..W-0699): per change its code, the short console text, the
description of the change and how to silence the notice.  The single source of these texts: ``_behaviour_changes``
reads its ``summary`` / ``silence`` from here, and the registry turns each entry into a ``Note:`` code whose
documentation is the description.  Add a new change at the end with the next free code.
"""
from __future__ import annotations

from flextool.run_warnings.registry import NOTE, WarningSpec

MARKER_PREFIX = "flextool_behaviour_change:"
_DELETE_MARKER = "Delete the marker {marker} in the database's Metadata tab once checked."


def _delete_marker(change_id: str) -> str:
    return _DELETE_MARKER.format(marker=MARKER_PREFIX + change_id)


#: ``{change id: (code, console text, summary, silence)}``.
BEHAVIOUR_TEXTS: dict[str, tuple[str, str, str, str]] = {
    "storage_default_fix_nothing": (
        "W-0600", "storage starts free by default (fix_nothing)",
        ("The default storage_start_end_method is now fix_nothing: a storage node without its own "
         "storage_start_end_method starts free. Before, when some node had a storage_state_start, it was "
         "fixed at the start: at its own storage_state_start, or at 0 when only other nodes had one "
         "(results change)."),
        ("Set storage_start_end_method on these nodes (fix_nothing keeps the new behaviour)."),
    ),
    "storage_fix_method_rewritten": (
        "W-0601", "the upgrade rewrote storage_start_end_method",
        ("The database upgrade rewrote storage_start_end_method on these nodes, so results are unchanged."
         " A fix_start / fix_end / fix_start_end without storage_state_start / storage_state_end is now "
         "an error. The default is now fix_nothing, so a node that used the old default fix_start with "
         "its own storage_state_start now has fix_start written."),
        _delete_marker("storage_fix_method_rewritten"),
    ),
    "user_constraint_prebuilt_nets_retirements": (
        "W-0602", "pre-built capacity term subtracts retirements",
        ("The pre-built capacity term of a user constraint "
         "(constraint_cumulative_pre_built_capacity_coeff) now subtracts the capacity retired in earlier "
         "periods of the same solve too (before: only retirements of earlier solves of a chain), so a "
         "single solve and a chain of solves agree."),
        _delete_marker("user_constraint_prebuilt_nets_retirements"),
    ),
    "dc_power_flow_lines_beside_dc_links": (
        "W-0603", "DC power-flow lines are the group's connections",
        ("DC power-flow lines are the connections inside dc_power_flow_with_angles groups (other than "
         "is_DC connections). Before, a model with any is_DC connection used the is_DC connections "
         "instead: the group's lines had no angle constraints and the reference node could move (results "
         "change)."),
        _delete_marker("dc_power_flow_lines_beside_dc_links"),
    ),
    "ladder_tier_labels_as_written": (
        "W-0604", "price-ladder tier labels are kept as written",
        ("Price-ladder tier labels are now kept as written: tiers such as 01 or a are their own tiers. "
         "Before, a label was read as a whole number, so 01 became 1 (such tiers were bought at no cost, "
         "and 1 and 01 were merged) and tiers like a were dropped (results change)."),
        _delete_marker("ladder_tier_labels_as_written"),
    ),
    "unnamed_maps_read_per_depth": (
        "W-0605", "maps without index names are read per depth",
        ("Maps without index names that have different depths within one parameter are now read per "
         "depth: a one-level map keyed by time steps next to a period x time map is read over time. "
         "Before, its time steps were read as periods and its values were lost (results change)."),
        _delete_marker("unnamed_maps_read_per_depth"),
    ),
    "price_ladder_periods_from_solve_periods": (
        "W-0606", "a ladder without periods applies in every period",
        ("A price_ladder_annual given without periods now applies in every period of a solve that has no "
         "period_timeset (the model's only timeset is used). Before, when the model had no "
         "periods_available either, the ladder was skipped with a warning (results change)."),
        _delete_marker("price_ladder_periods_from_solve_periods"),
    ),
    "coarse_process_flows": (
        "W-0607", "processes of coarse nodes have coarse flows",
        ("A unit or connection whose balance nodes are all in coarse new_stepduration groups now has one "
         "flow per coarse step of its nodes (before: one per time step of the solve), with its "
         "availability, profiles and efficiency averaged over the coarse step and its ramp limits between"
         " coarse steps; a unit with online variables decides its units online, start-ups and shut-downs "
         "once per coarse step, counts its minimum up / down time in whole coarse steps and takes its "
         "start-up / shut-down capacity reduction over the coarse step; a reserve provider holds one "
         "reserve per coarse step, the requirement checked at every time step; an exact connection "
         "chooses its direction and the nodes of a DC network whose lines are all coarse their angle once"
         " per coarse step; a capacity-margin, inertia, non-synchronous, flow-group instant-flow, "
         "user-constraint, storage-use tie or DC power-flow row whose units and connections are all "
         "coarse is averaged over the coarse step, and one that mixes them with units at the solve's step"
         " sees the coarse flows constant (results change where, within a coarse step, a cost, price, "
         "efficiency, minimum load, demand, inflow, profile or row limit varies, two or more limits vary,"
         " ramp limits apply, the unit has online variables or provides reserves, or a row mixes coarse "
         "and solve-step flows)."),
        _delete_marker("coarse_process_flows"),
    ),
    "construction_lead_time_commissioning": (
        "W-0608", "lead-time capacity counts when it comes online",
        ("With a construction lead time, 'invested' and the per-period investment limits count capacity "
         "coming online in the period; annuity and fixed cost are paid only while it operates, at the "
         "invest_cost of the period it comes online; the start of operation uses the whole model's "
         "periods; investments starting after the solve or model horizon are not offered (results "
         "change)."),
        _delete_marker("construction_lead_time_commissioning"),
    ),
    "chained_solve_costs_from_run_start": (
        "W-0609", "costs of a run of solves count from its start",
        ("In a run of several solves the reported costs are now discounted from the start of the run "
         "(before: from the start of each solve); the investment and fixed cost of new capacity cover its"
         " operating years in the whole run and are reported once, in the period it starts operating, by "
         "the solve that realizes that period (before: in the decision period, over the deciding solve's "
         "periods); the fixed cost of capacity carried from an earlier solve or retired is no longer "
         "charged again as existing capacity.  The printed objectives stayed discounted from each solve's"
         " start until chained_solve_discount_from_run_start (results change)."),
        _delete_marker("chained_solve_costs_from_run_start"),
    ),
    "stochastic_branch_period_years": (
        "W-0610", "stochastic branch periods count from their year",
        ("In a stochastic solve whose stochastic_invest_method is not recourse, the branch periods of the"
         " stochastic branches now count from the year of their period. Before, every branch period "
         "counted as the first year of the solve: capacity invested in a later period, or starting to "
         "operate in a later period after a construction lead time, was missing from the branches of that"
         " period, and lifetimes ended late in the branches (results change)."),
        _delete_marker("stochastic_branch_period_years"),
    ),
    "ramp_branch_point_shutdown_once": (
        "W-0611", "branch-point ramp counts the shut-down once",
        ("At a stochastic branch point (a step continued by several mid-horizon branches) the downward "
         "ramp row of each branch's first step now counts the shut-down share of the units online at the "
         "branch point once. Before, it was multiplied by the number of branches squared, so units "
         "shutting down there could drop their flow faster than the ramp limit allows (results change)."),
        _delete_marker("ramp_branch_point_shutdown_once"),
    ),
    "retirement_of_existing_capacity_only": (
        "W-0612", "only existing capacity can be retired",
        ("Only capacity that existed before the run can be retired, and a retirement counts until that "
         "capacity's end of life (reinvest_choice / no_reinvestment: the run's first year plus "
         "lifetime_technical, in every solve of a chain and once with stochastic branches in the first "
         "period; before, a later solve counted it from its own first period and branches multiplied it; "
         "reinvest_automatic: the end of the run). It saves each period's fixed cost of the existing "
         "capacity in that window, with the factor that fixed cost is charged with (its net salvage value"
         " is unchanged: an annuity over payback_time). Periods without existing capacity have no "
         "retirement option. Before, a retirement counted in every later period (past the end of life the"
         " capacity went negative: an infeasible or zero retirement), chained solves subtracted it from "
         "every period, and its fixed-cost saving was counted over the technical lifetime from the "
         "retirement (results change)."),
        _delete_marker("retirement_of_existing_capacity_only"),
    ),
    "acceptance_ceiling_follows_tolerance": (
        "W-0613", "near-optimal acceptance follows the tolerance",
        ("A solve HiGHS could not certify as optimal (status Unknown) is now accepted as near-optimal "
         "when its largest relative primal infeasibility is within max(1e-6, 10 x its "
         "primal_feasibility_tolerance), and its primal-dual objective error is within the accepted gap. "
         "Before, the limit was 1e-6 whatever the tolerance, so a solve run with a looser tolerance (e.g."
         " 1e-3) stopped the run without results; it now writes results."),
        _delete_marker("acceptance_ceiling_follows_tolerance"),
    ),
    "solve_step_series_combined": (
        "W-0614", "new_stepduration combines series by their kind",
        ("A solve with new_stepduration now combines every time series over the time steps of each new "
         "step by the kind of its value: prices, costs, limits, availability, min_load, penalties, "
         "reservations, self-discharge, time-step weights and profiles take the duration-weighted mean, "
         "efficiencies the harmonic mean (exact fuel use at a constant flow), inflow the sum and "
         "storage_state_reference_value the value at the end of the step. Before, these took the value of"
         " the step's first time step; only time-map profiles and inflow were combined (results change). "
         "A stochastic (branch) inflow used to take its first time step's value once; it is now summed "
         "like a deterministic inflow, so it grows by the number of time steps each new step combines, "
         "also where it is constant within the step."),
        _delete_marker("solve_step_series_combined"),
    ),
    "energy_margin_adder_per_original_step": (
        "W-0615", "energy_margin_adder is per timeline step",
        ("energy_margin_adder is now an energy per time step of the timeline: when the investment solve "
         "has new_stepduration, a solve step adds the sum of the values of the time steps it combines (a "
         "constant or period value counts once per time step). Before, a solve step added the value of "
         "its first time step once, so a coarse investment solve added less demand (results change). To "
         "keep the old demand, divide the value by the number of time steps a solve step combines."),
        _delete_marker("energy_margin_adder_per_original_step"),
    ),
    "capacity_margin_counts_by_flow": (
        "W-0616", "capacity margin counts derated capacity and flows",
        ("The capacity margin now counts firm and VRE units at their capacity derated by availability "
         "(and capacity_max_coeff), units fed from a storage node, connections and the units consuming "
         "from the group by their actual flow; energy dumped through the downward balance slack does not "
         "count. Before, availability was ignored, storage-backed units counted at full capacity and "
         "connections not at all. Exports now add to the requirement; connections and lossy units inside "
         "the group are left out (results change). In an investment -> dispatch chain the investment "
         "solve's shortfall and its penalty are now reported (slack__capacity_margin__d, the costs) "
         "instead of 0, so the total cost includes the penalty; capacity_margin__d explains each solve's "
         "margin."),
        _delete_marker("capacity_margin_counts_by_flow"),
    ),
    "discount_rate_zero_is_zero": (
        "W-0617", "a discount_rate of 0 is 0 %",
        ("A discount_rate of 0 is now 0 %: the annuity of an investment is invest_cost / payback_time "
         "(net_salvage_value / payback_time for a retirement). Before, a discount_rate of 0 or below was "
         "read as 5 %, so these investments cost more per year (results change). A negative discount_rate"
         " is now an input error, and an investing or retiring entity with a cost needs a "
         "lifetime_technical or payback_time (before: 20 years)."),
        _delete_marker("discount_rate_zero_is_zero"),
    ),
    "invest_without_cost_offered": (
        "W-0618", "investments without invest_cost are offered",
        ("An invest_method that allows investing (retiring) now offers the investment (retirement) in "
         "every investment period, also where invest_cost (net_salvage_value) is missing or 0: that "
         "capacity has no annuity, its fixed_cost still counts. Before, such a period had no investment "
         "(retirement) variable unless a user constraint had a capacity coefficient on the entity, so a 0"
         " in an invest_cost map blocked investment in that period (results change). To forbid investment"
         " in a period, set invest_max_period 0 there under a *_period invest_method (the other methods "
         "do not use invest_max_period), or use not_allowed. With neither invest_cost nor fixed_cost the "
         "invested amount is arbitrary."),
        _delete_marker("invest_without_cost_offered"),
    ),
    "stochastic_storage_paths_tied": (
        "W-0619", "stochastic bind_within_solve closes each path",
        ("In a stochastic solve, bind_within_solve now closes one loop per path: the last step of every "
         "forecast branch ends at the level the solve's first step starts from, and with forecasts "
         "opening in the solve's first period each forecast's first step continues from the end of its "
         "own path. Before, only the branch listed last closed the loop and the others ended free (a fan "
         "in the first period looped each forecast within its first period). With bind_forward_only and a"
         " free start (no fix_start, no carried state), the forecasts opening in the solve's first period"
         " now start at the same level; before, each chose its own (results change). With "
         "storage_start_end_method fix_start the start of each forecast is the storage_state_start share "
         "of that forecast's own storage capacity, so where the capacity differs by forecast (investment "
         "or divestment per forecast) the starting energy differs. The upgrade marks every stochastic "
         "database with a storage node on bind_within_solve, or on bind_forward_only without a fixed "
         "start; such a bind_forward_only node changes only when a fan opens in the solve's first period."),
        _delete_marker("stochastic_storage_paths_tied"),
    ),
    "chained_solve_discount_from_run_start": (
        "W-0620", "a run of solves discounts from its start",
        ("With a social_discount_rate other than 0, every solve now discounts its investments and its "
         "operations from the start of the run, as one solve over the same periods does. Before, a later "
         "solve of a run of several solves (also a split solve of a 2-D period map, a rolling or nested "
         "solve starting after the run start) discounted its operations, and when its years_represented "
         "held only its own periods also its investments, from its own start; and a fractional "
         "years_represented counted as a whole year in the discounting of the periods after it; a "
         "years_represented map that listed the periods out of the solve's period order (or an unused "
         "period before a used one) discounted the investments in the period order and the operations in "
         "the map order, also in a single solve, and now both in the map order. Where only the "
         "investments were discounted from the run start (hand-written solves with the whole "
         "years_represented map), the balance between investment and operating costs was wrong and "
         "decisions change; otherwise the objectives, investment duals and CO2 cap prices of later solves"
         " are now on the run's discount base, and decisions can still change through absolute MIP gaps, "
         "objective targets or bounds and terms without a discount factor (results change)."),
        _delete_marker("chained_solve_discount_from_run_start"),
    ),
    "rolling_cyclic_storage_anchored": (
        "W-0621", "rolling bind_within_solve storage is anchored",
        ("In a rolling_window solve without representative periods, a storage node with "
         "storage_binding_method bind_within_solve (also bind_within_solve_blended_weights, which such a "
         "solve uses as bind_within_solve) now continues each roll from the level realised by the "
         "previous roll, and each roll ends its horizon at least at the level it started from. The "
         "roll(s) whose horizon reaches the end of the solve end at the level the first roll started "
         "from, so the year (the solve) returns the energy it took. An upper-level fix_quantity or "
         "fix_usage at the end of a roll's horizon replaces these end conditions (fix_price keeps the "
         "closure and only values the end), and fix_end wins at the last step of the model. "
         "bind_within_period (also bind_within_period_blended_weights, and bind_within_timeblock, which "
         "such a solve uses as bind_within_period) and bind_intraperiod_blocks do the same per period "
         "where a roll starts inside one: the roll continues it from the carried level and the roll(s) "
         "holding its end close it at the level it started from. Before, each roll cycled on itself from "
         "a free level (or the blocks of a period held in part chained forward without closing it), so "
         "the level jumped at roll boundaries and the rolls could use energy that was never produced "
         "(results change). The closing level is the start level in MWh, at most the storage capacity of "
         "the closing period."),
        _delete_marker("rolling_cyclic_storage_anchored"),
    ),
    "nested_storage_start_borrowed": (
        "W-0622", "nested storage starts from the upper level",
        ("A storage node of a nested lower-level solve whose end an upper-level solve fixes "
         "(storage_nested_fix_method fix_quantity, fix_usage or fix_price) and that has no state realised"
         " before the lower solve now starts from the level the upper solve planned for the same moment, "
         "when it chains forward (bind_forward_only, or bind_within_solve in a rolling solve under "
         "fix_quantity or fix_usage, and bind_within_period (also bind_within_timeblock) there when the "
         "first roll cuts the period) and fix_start does not apply (no fix_start, or the lower solve "
         "starts after the start of the model). Before, it started free or from fix_start, so the lower "
         "solve could start fuller than the upper plan and keep the difference (results change)."),
        _delete_marker("nested_storage_start_borrowed"),
    ),
    "nested_storage_fix_cumulative": (
        "W-0623", "nested fix_usage is capped from the period start",
        ("A lower-level solve under an upper-level storage_nested_fix_method fix_usage is now capped by "
         "the upper solve's usage from the start of the period (within the instance nested in that upper "
         "solve) to the end of its horizon, less what its earlier rolls realised, with an upper step that"
         " the span cuts counted pro rata to its hours. Before, each roll was capped by the upper usage "
         "of every upper step it touched, counted whole and from its own start, so the realised rolls "
         "could use energy the upper solve never planned. Under fix_quantity a lower horizon that ends "
         "inside an upper step now takes the upper level interpolated at that moment instead of the level"
         " at the end of the step (results change)."),
        _delete_marker("nested_storage_fix_cumulative"),
    ),
    "rp_storage_chain_pinned": (
        "W-0624", "representative-period storage chain start / end set",
        ("In a representative-period solve, storage_state_start, storage_state_end and "
         "use_reference_value of a *_blended_weights storage node now set the stored level at the start "
         "and the end of its base-period chain (the chronological level the representative blocks add up "
         "to), not a state inside the first or last representative block, which only placed the block's "
         "free offset. The level a bind_forward_only_blended_weights chain starts from is now within the "
         "storage capacity, and a bind_within_solve_blended_weights or bind_within_period_blended_weights"
         " node takes no use_reference_value (its cycle closes on itself). A solve with representative "
         "periods that follows another solve starts the chain of a bind_forward_only_blended_weights node"
         " from the realised level, and hands on its chain level at the end of its realised period. "
         "Before, a forward chain started and ended at free levels whatever these parameters said, so an "
         "investment solve could use stored energy that was never produced (results change)."),
        _delete_marker("rp_storage_chain_pinned"),
    ),
    "storage_end_matches_start": (
        "W-0625", "forward storage ends at least at its start",
        ("A storage node that chains forward (bind_forward_only, also the default, and "
         "bind_forward_only_blended_weights) and whose start the solve decides now ends the model at "
         "least at that start: a free start (storage_start_end_method fix_nothing or fix_end) ends at "
         "least at the level it started from, and a fix_start start on storage the solve can invest in "
         "returns at least the storage_state_start share of the capacity it builds in the first period. "
         "This holds in a solve that has both the start of the model and its last step (a single solve, "
         "an investment solve over the whole horizon, every representative-period solve) and, with a free"
         " start (fix_nothing), over the rolls of a rolling solve (the first roll's horizon ends at least"
         " at its start, the rolls reaching the end of the model at least at that level). Before, the "
         "solve could start full and end empty, or build storage only for its start energy (results "
         "change). fix_start on capacity the solve cannot change keeps a free end."),
        _delete_marker("storage_end_matches_start"),
    ),
    "storage_reference_start": (
        "W-0626", "use_reference_value storage starts at the reference",
        ("A storage node with storage_solve_horizon_method use_reference_value and no fix_start now "
         "starts from its storage_state_reference_value at the first step (the model's first step in the "
         "first solve). Before, its start was free and only the matching end bounded it, so a reference "
         "series that falls during the year did not drain the storage; now the existing storage gives the"
         " drop (results change). With fix_start, storage_state_start still wins."),
        _delete_marker("storage_reference_start"),
    ),
    "annuity_over_payback_time": (
        "W-0627", "investment cost is levelled over the lifetime",
        ("With lifetime_method reinvest_choice or no_reinvestment the investment cost is now the level "
         "yearly cost of the plant over its lifetime_technical (the annuity over payback_time with "
         "discount_rate, its present value spread evenly over the lifetime), paid for each year the "
         "capacity is in service within the horizon. Before, the annuity was paid in every period that "
         "started before the end of the lifetime, also where the capacity was already gone (results "
         "change). The reported investment costs follow; the fixed cost runs over the same years."),
        _delete_marker("annuity_over_payback_time"),
    ),
    "fixed_cost_per_operating_period": (
        "W-0628", "new capacity pays each period's fixed_cost",
        ("New capacity now pays the fixed_cost of each period it operates in, as existing capacity does. "
         "In FlexTool 4.5.0 an investment paid the fixed_cost of the period it came online in every "
         "period it operated, so where fixed_cost differs between periods (a period map) the fixed cost "
         "of new capacity changes (results change)."),
        _delete_marker("fixed_cost_per_operating_period"),
    ),
    "annuity_timing": (
        "W-0629", "annuity sized for discount_offset_investment",
        ("The yearly payment of the investment annuity (and of the yearly salvage credits) is now sized "
         "for the moment discount_offset_investment places it in the year: the annuity times (1 + "
         "discount_rate)^(discount_offset_investment - 1), so the loan taken when the capacity comes "
         "online is worth invest_cost at the discount_rate. Before, every offset used the year-end "
         "annuity, so at the default offset 0 (payments at the start of each year) an investment cost "
         "invest_cost times (1 + discount_rate). Investments are now cheaper by one year of interest at "
         "offset 0 (results change); offset 1 is unchanged."),
        _delete_marker("annuity_timing"),
    ),
    "net_salvage_once": (
        "W-0630", "net_salvage_value is counted once",
        ("The net_salvage_value is now counted once, as a cash flow when the capacity stops operating. A "
         "retirement of existing capacity receives the whole net_salvage_value of the retirement period "
         "at its start (the investment in existing capacity is sunk; before: the salvage annuity over "
         "lifetime_technical, inside the horizon). With lifetime_method reinvest_choice or "
         "no_reinvestment the net_salvage_value at the end of life lowers the yearly cost of new capacity"
         " (levelled over lifetime_technical at discount_rate; before: no effect), existing capacity "
         "holds it as a level yearly credit (a constant that is not in the reported objective) and a "
         "retirement gives the credits back. Retiring long-lived capacity late in the horizon can now "
         "pay, since the years after the horizon are not seen (results change)."),
        _delete_marker("net_salvage_once"),
    ),
    "built_capacity_retirable": (
        "W-0631", "built capacity can be retired",
        ("With lifetime_method reinvest_choice or no_reinvestment and an invest_method that allows "
         "investment and retirement (invest_retire_* or cumulative_limits), the capacity the model builds"
         " can now be retired too (before: only existing capacity). A retired MW pays its loan in full at"
         " the start of the retirement period, receives the net_salvage_value then and saves its fixed "
         "cost from then on; retirement limits and user-constraint retirement terms count it, and a chain"
         " of solves carries each retirement with its cohort (results change)."),
        _delete_marker("built_capacity_retirable"),
    ),
    "automatic_replacement_cost": (
        "W-0632", "automatic replacement priced as a new plant",
        ("With lifetime_method reinvest_automatic and a lifetime_technical, capacity is now replaced at "
         "the end of its life by a new plant, at the invest_cost, discount_rate, payback_time, "
         "lifetime_technical and net_salvage_value of the period the replacement comes online, and "
         "replaced again at its own end of life to the end of the horizon. Before, it paid the annuity of"
         " its first plant for ever, without its net_salvage_value. Existing capacity reaches its end of "
         "life and is replaced the same way (free without an invest_cost), and reinvest_automatic "
         "capacity, built or existing, can be retired when its invest_method allows it: retiring it also "
         "avoids every replacement still to come (results change)."),
        _delete_marker("automatic_replacement_cost"),
    ),
    "end_of_life_closest_seam": (
        "W-0633", "end of life rounds to the closest period start",
        ("With lifetime_method reinvest_choice or no_reinvestment, the end of life (start of operation "
         "plus lifetime_technical; existing capacity from the model start) is now rounded to the closest "
         "period start (end_of_life_method closest_seam, the new default; a tie goes to the later start)."
         " Before, it was rounded to the next period start, so the capacity operated to the end of the "
         "period its life ended in. Capacity whose life ends in the first half of a period is now out of "
         "service, without fixed cost, for that whole period (results change)."),
        ("Set end_of_life_method on these entities: next_seam keeps the old rounding, closest_seam the "
         "new."),
    ),
    "minimum_time_method_written": (
        "W-0634", "the upgrade wrote minimum_time_method",
        ("minimum_time_method now decides whether min_uptime and min_downtime apply: a unit without it (none, "
         "the default) has no minimum up or down time. Before, a time applied whenever it was set. The "
         "database upgrade wrote minimum_time_method (both, min_uptime or min_downtime, and none where a unit "
         "is kept offline) on these units, so their results are unchanged. Units the upgrade could not decide "
         "got nothing written: in the scenarios it named, their times no longer apply (results change), and warning W-0231 names them at "
         "each run. Set minimum_time_method on them where the times should apply."),
        _delete_marker("minimum_time_method_written"),
    ),
    "rolling_within_solve_one_solve": (
        "W-0635", "rolling within_solve realizes the jump in one solve",
        ("A rolling stochastic solve with dispatch_realization_method within_solve now solves each roll once. Over "
         "the jump it dispatches the data of the realized: yes branch once, under the period's own name. After the "
         "jump it looks ahead with each forecast made at the end of the jump, with its probability. Before, each "
         "roll solved the forecasts made at its start, and a separate realization solve then dispatched the jump "
         "alone, without looking ahead. Results change: units online, storage levels and prices in the jump, and, "
         "with timeset weights that are not equal, the costs reported for the jump. Such a solve now needs a "
         "realized: yes branch and forecasts made at the end of each jump; without them the run stops with an "
         "error naming the roll."),
        _delete_marker("rolling_within_solve_one_solve"),
    ),
    "separate_dispatch_realization_looks_ahead": (
        "W-0636", "separate_dispatch realization looks ahead",
        ("In a rolling stochastic solve with dispatch_realization_method separate_dispatch, the realization solve "
         "now looks ahead after its jump with each forecast made at the end of the jump and its probability; "
         "before, it dispatched the jump alone. In rolling and single solves, the first solve now shares across "
         "the forecasts only the units online of units whose startup_method ends in _fix_in_uc, the only "
         "decisions the realization keeps, and the start level of a storage that starts free; before, it also "
         "shared the other units online, reserves and the storage use of the stochastic nodes. Results change: units online, storage levels, prices and, with "
         "timeset weights that are not equal, the costs reported for the jump. A rolling solve without forecasts "
         "made at the start of each roll and at the end of each jump now stops with an error naming the roll and "
         "the time step."),
        _delete_marker("separate_dispatch_realization_looks_ahead"),
    ),
}

#: The changes whose upgrade keeps the results (the upgrade wrote values that keep the old behaviour); every
#: other entry changes results.  The GUI's "database upgraded" message lists them under their own heading.
RESULTS_UNCHANGED = frozenset({"storage_fix_method_rewritten", "minimum_time_method_written"})

BEHAVIOUR_SPECS: tuple[WarningSpec, ...] = tuple(
    WarningSpec(
        code=code,
        message="changed in {version}: " + text,
        title=f"Behaviour change: {change_id}",
        topic="Behaviour changes",
        severity=NOTE,
        body=(f"{summary}\n\nThe notice is printed for a database upgraded from an earlier FlexTool, once per run "
              "(the debug log names every affected entity and gives the full notice).\n\n"
              f"**How to fix:** check the results of the affected entities. {silence}"),
    )
    for change_id, (code, text, summary, silence) in BEHAVIOUR_TEXTS.items()
)
