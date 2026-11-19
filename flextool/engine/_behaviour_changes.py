"""Behaviour-change notices.

When a FlexTool release changes what an existing model does, the database upgrade writes a marker into the
upgraded database (metadata ``flextool_behaviour_change:<id>``, visible in the database editor's Metadata tab).
While the marker is present, each run prints one notice per change naming the entities whose results the change
affects.  Deleting the marker stops the notice.  New databases (and the repository's templates, examples and test
fixtures) carry no markers, so they never print notices.

:data:`BEHAVIOUR_CHANGES` is the registry.  ``since`` / ``published`` are the FlexTool version and its publication
date (``D.M.YYYY``, as in the CHANGELOG ``## Release X (D.M.YYYY)`` heading); both stay ``None`` until the release
that ships the change fills them in (``tests/engine/test_behaviour_change_registry.py`` refuses a release that
leaves one empty).

A run calls :func:`start_run` with the markers of its input database (:func:`read_markers`).  :func:`notice`
prints with ``print(..., flush=True)`` and announces each entity of a change once per run, so rolls and repeated
solves of the same run do not repeat it.  :func:`warn_once` gives other run-time warnings the same once-per-run
scope.

``gui_solver_defaults`` is not a database change: its marker is the ``solver_settings_revision`` of a GUI
project's settings.yaml, and the GUI shows the notice once when it updates an older file
(``flextool.gui.settings_io``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import metadata
from typing import Iterable, Mapping

MARKER_PREFIX = "flextool_behaviour_change:"


@dataclass(frozen=True)
class BehaviourChange:
    id: str
    summary: str
    silence: str
    since: str | None = None
    published: str | None = None


_DELETE_MARKER = "Delete the marker {marker} in the database's Metadata tab once checked."

BEHAVIOUR_CHANGES: dict[str, BehaviourChange] = {
    change.id: change for change in (
        BehaviourChange(
            id="storage_default_fix_nothing",
            summary=("The default storage_start_end_method is now fix_nothing: a storage node without its own "
                     "storage_start_end_method starts free. Before, when some node had a storage_state_start, it "
                     "was fixed at the start: at its own storage_state_start, or at 0 when only other nodes had "
                     "one (results change)."),
            silence="Set storage_start_end_method on these nodes (fix_nothing keeps the new behaviour).",
            since="4.2.0",
            published="7.11.2026",
        ),
        BehaviourChange(
            id="storage_fix_method_rewritten",
            summary=("The database upgrade rewrote storage_start_end_method on these nodes, so results are "
                     "unchanged. A fix_start / fix_end / fix_start_end without storage_state_start / "
                     "storage_state_end is now an error. The default is now fix_nothing, so a node that used the "
                     "old default fix_start with its own storage_state_start now has fix_start written."),
            silence=_DELETE_MARKER.format(marker=MARKER_PREFIX + "storage_fix_method_rewritten"),
            since="4.2.0",
            published="7.11.2026",
        ),
        BehaviourChange(
            id="user_constraint_prebuilt_nets_retirements",
            summary=("The pre-built capacity term of a user constraint "
                     "(constraint_cumulative_pre_built_capacity_coeff) now subtracts the capacity retired in "
                     "earlier periods of the same solve too (before: only retirements of earlier solves of a "
                     "chain), so a single solve and a chain of solves agree."),
            silence=_DELETE_MARKER.format(marker=MARKER_PREFIX + "user_constraint_prebuilt_nets_retirements"),
            since="4.2.0",
            published="7.11.2026",
        ),
        BehaviourChange(
            id="dc_power_flow_lines_beside_dc_links",
            summary=("DC power-flow lines are the connections inside dc_power_flow_with_angles groups (other than "
                     "is_DC connections). Before, a model with any is_DC connection used the is_DC connections "
                     "instead: the group's lines had no angle constraints and the reference node could move "
                     "(results change)."),
            silence=_DELETE_MARKER.format(marker=MARKER_PREFIX + "dc_power_flow_lines_beside_dc_links"),
            since="4.2.0",
            published="7.11.2026",
        ),
        BehaviourChange(
            id="ladder_tier_labels_as_written",
            summary=("Price-ladder tier labels are now kept as written: tiers such as 01 or a are their own "
                     "tiers. Before, a label was read as a whole number, so 01 became 1 (such tiers were bought "
                     "at no cost, and 1 and 01 were merged) and tiers like a were dropped (results change)."),
            silence=_DELETE_MARKER.format(marker=MARKER_PREFIX + "ladder_tier_labels_as_written"),
            since="4.2.0",
            published="7.11.2026",
        ),
        BehaviourChange(
            id="unnamed_maps_read_per_depth",
            summary=("Maps without index names that have different depths within one parameter are now read "
                     "per depth: a one-level map keyed by time steps next to a period x time map is read over "
                     "time. Before, its time steps were read as periods and its values were lost (results "
                     "change)."),
            silence=_DELETE_MARKER.format(marker=MARKER_PREFIX + "unnamed_maps_read_per_depth"),
            since="4.2.0",
            published="7.11.2026",
        ),
        BehaviourChange(
            id="price_ladder_periods_from_solve_periods",
            summary=("A price_ladder_annual given without periods now applies in every period of a solve that "
                     "has no period_timeset (the model's only timeset is used). Before, when the model had no "
                     "periods_available either, the ladder was skipped with a warning (results change)."),
            silence=_DELETE_MARKER.format(marker=MARKER_PREFIX + "price_ladder_periods_from_solve_periods"),
            since="4.2.0",
            published="7.11.2026",
        ),
        BehaviourChange(
            id="coarse_process_flows",
            summary=("A unit or connection whose balance nodes are all in coarse new_stepduration groups now has "
                     "one flow per coarse step of its nodes (before: one per time step of the solve), with its "
                     "availability, profiles and efficiency averaged over the coarse step and its ramp limits "
                     "between coarse steps; a unit with online variables decides its units online, start-ups "
                     "and shut-downs once per coarse step, counts its minimum up / down time in whole coarse "
                     "steps and takes its start-up / shut-down capacity reduction over the coarse step; a "
                     "reserve provider holds one reserve per coarse step, the requirement checked at every "
                     "time step; an exact connection chooses its direction and the nodes of a DC network whose "
                     "lines are all coarse their angle once per coarse step; a capacity-margin, inertia, non-synchronous, "
                     "flow-group instant-flow, user-constraint, storage-use tie or DC power-flow row whose "
                     "units and connections are all coarse is averaged over the coarse step, and one that mixes "
                     "them with units at the solve's step sees the coarse flows constant (results change where, "
                     "within a coarse step, a cost, price, efficiency, minimum load, demand, inflow, profile or "
                     "row limit varies, two or more limits vary, ramp limits apply, the unit has online "
                     "variables or provides reserves, or a row mixes coarse and solve-step flows)."),
            silence=_DELETE_MARKER.format(marker=MARKER_PREFIX + "coarse_process_flows"),
            since="4.2.0",
            published="7.11.2026",
        ),
        BehaviourChange(
            id="construction_lead_time_commissioning",
            summary=("With a construction lead time, 'invested' and the per-period investment limits count "
                     "capacity coming online in the period; annuity and fixed cost are paid only while it "
                     "operates, at the invest_cost of the period it comes online; the start of operation uses "
                     "the whole model's periods; investments starting after the solve or model horizon are not "
                     "offered (results change)."),
            silence=_DELETE_MARKER.format(marker=MARKER_PREFIX + "construction_lead_time_commissioning"),
            since="4.2.0",
            published="7.11.2026",
        ),
        BehaviourChange(
            id="chained_solve_costs_from_run_start",
            summary=("In a run of several solves the reported costs are now discounted from the start of the "
                     "run (before: from the start of each solve); the investment and fixed cost of new capacity "
                     "cover its operating years in the whole run and are reported once, in the period it starts "
                     "operating, by the solve that realizes that period (before: in the decision period, over "
                     "the deciding solve's periods); the fixed cost of capacity carried from an earlier solve or "
                     "retired is no longer charged again as existing capacity.  The printed objectives stayed "
                     "discounted from each solve's start until chained_solve_discount_from_run_start (results "
                     "change)."),
            silence=_DELETE_MARKER.format(marker=MARKER_PREFIX + "chained_solve_costs_from_run_start"),
            since="4.2.0",
            published="7.11.2026",
        ),
        BehaviourChange(
            id="stochastic_branch_period_years",
            summary=("In a stochastic solve whose stochastic_invest_method is not recourse, the branch periods of "
                     "the stochastic branches now count from the year of their period. Before, every branch period "
                     "counted as the first year of the solve: capacity invested in a later period, or starting to "
                     "operate in a later period after a construction lead time, was missing from the branches of "
                     "that period, and lifetimes ended late in the branches (results change)."),
            silence=_DELETE_MARKER.format(marker=MARKER_PREFIX + "stochastic_branch_period_years"),
            since="4.2.0",
            published="7.11.2026",
        ),
        BehaviourChange(
            id="ramp_branch_point_shutdown_once",
            summary=("At a stochastic branch point (a step continued by several mid-horizon branches) the "
                     "downward ramp row of each branch's first step now counts the shut-down share of the "
                     "units online at the branch point once. Before, it was multiplied by the number of "
                     "branches squared, so units shutting down there could drop their flow faster than the "
                     "ramp limit allows (results change)."),
            silence=_DELETE_MARKER.format(marker=MARKER_PREFIX + "ramp_branch_point_shutdown_once"),
            since="4.2.0",
            published="7.11.2026",
        ),
        BehaviourChange(
            id="retirement_of_existing_capacity_only",
            summary=("Only capacity that existed before the run can be retired, and a retirement counts until that "
                     "capacity's end of life (reinvest_choice / no_reinvestment: the run's first year plus "
                     "lifetime_technical, "
                     "in every solve of a chain and once with stochastic branches in the first period; before, "
                     "a later solve counted it from its own first period and branches multiplied it; "
                     "reinvest_automatic: the end of the run). It saves each "
                     "period's fixed cost of the existing capacity in that window, with the factor that fixed cost "
                     "is charged with (its net salvage value is unchanged: an annuity over payback_time). Periods "
                     "without existing capacity have no retirement option. Before, a retirement counted in every later "
                     "period (past the end of life the capacity went negative: an infeasible or zero retirement), "
                     "chained solves subtracted it from every period, and its fixed-cost saving was counted over "
                     "the technical lifetime from the retirement (results change)."),
            silence=_DELETE_MARKER.format(marker=MARKER_PREFIX + "retirement_of_existing_capacity_only"),
            since="4.2.0",
            published="7.11.2026",
        ),
        BehaviourChange(
            id="acceptance_ceiling_follows_tolerance",
            summary=("A solve HiGHS could not certify as optimal (status Unknown) is now accepted as near-optimal "
                     "when its largest relative primal infeasibility is within max(1e-6, 10 x its "
                     "primal_feasibility_tolerance), and its primal-dual objective error is within the accepted "
                     "gap. Before, the limit was 1e-6 whatever the tolerance, so a solve run with a looser "
                     "tolerance (e.g. 1e-3) stopped the run without results; it now writes results."),
            silence=_DELETE_MARKER.format(marker=MARKER_PREFIX + "acceptance_ceiling_follows_tolerance"),
            since="4.2.0",
            published="7.11.2026",
        ),
        BehaviourChange(
            id="solve_step_series_combined",
            summary=("A solve with new_stepduration now combines every time series over the time steps of each "
                     "new step by the kind of its value: prices, costs, limits, availability, min_load, penalties, "
                     "reservations, self-discharge, time-step weights and profiles take the duration-weighted "
                     "mean, efficiencies the harmonic mean (exact fuel use at a constant flow), inflow the sum and "
                     "storage_state_reference_value the value at the end of the step. Before, these took the value "
                     "of the step's first time step; only time-map profiles and inflow were combined "
                     "(results change). A stochastic (branch) inflow used to take its first time step's value "
                     "once; it is now summed like a deterministic inflow, so it grows by the number of time steps "
                     "each new step combines, also where it is constant within the step."),
            silence=_DELETE_MARKER.format(marker=MARKER_PREFIX + "solve_step_series_combined"),
            since="4.4.0",
            published="13.11.2026",
        ),
        BehaviourChange(
            id="energy_margin_adder_per_original_step",
            summary=("energy_margin_adder is now an energy per time step of the timeline: when the investment "
                     "solve has new_stepduration, a solve step adds the sum of the values of the time steps it "
                     "combines (a constant or period value counts once per time step). Before, a solve step "
                     "added the value of its first time step once, so a coarse investment solve added less "
                     "demand (results change). To keep the old demand, divide the value by the number of time "
                     "steps a solve step combines."),
            silence=_DELETE_MARKER.format(marker=MARKER_PREFIX + "energy_margin_adder_per_original_step"),
            since="4.4.0",
            published="13.11.2026",
        ),
        BehaviourChange(
            id="capacity_margin_counts_by_flow",
            summary=("The capacity margin now counts firm and VRE units at their capacity derated by availability "
                     "(and capacity_max_coeff), units fed from a storage node, connections and the units "
                     "consuming from the group by their actual flow; energy dumped through the downward balance "
                     "slack does not count. Before, availability was ignored, storage-backed units counted at full capacity and "
                     "connections not at all. Exports now add to the requirement; connections and lossy units "
                     "inside the group are left out (results change). In an investment -> dispatch chain the investment solve's "
                     "shortfall and its penalty are now reported (slack__capacity_margin__d, the costs) instead of "
                     "0, so the total cost includes the penalty; capacity_margin__d explains each solve's margin."),
            silence=_DELETE_MARKER.format(marker=MARKER_PREFIX + "capacity_margin_counts_by_flow"),
            since="4.4.0",
            published="13.11.2026",
        ),
        BehaviourChange(
            id="discount_rate_zero_is_zero",
            summary=("A discount_rate of 0 is now 0 %: the annuity of an investment is invest_cost / payback_time "
                     "(net_salvage_value / payback_time for a retirement). Before, a discount_rate of 0 or below was "
                     "read as 5 %, so these investments cost more per year (results change). A negative "
                     "discount_rate is now an input error, and an investing or retiring entity with a cost "
                     "needs a lifetime_technical or payback_time (before: 20 years)."),
            silence=_DELETE_MARKER.format(marker=MARKER_PREFIX + "discount_rate_zero_is_zero"),
            since="4.4.0",
            published="13.11.2026",
        ),
        BehaviourChange(
            id="invest_without_cost_offered",
            summary=("An invest_method that allows investing (retiring) now offers the investment (retirement) "
                     "in every investment period, also where invest_cost (net_salvage_value) is missing or 0: that "
                     "capacity has no annuity, its fixed_cost still counts. Before, such a period had no "
                     "investment (retirement) variable unless a user constraint had a capacity coefficient on "
                     "the entity, so a 0 in an invest_cost map blocked investment in that period (results "
                     "change). To forbid investment in a period, set invest_max_period 0 there under a *_period "
                     "invest_method (the other methods do not use invest_max_period), or use not_allowed. With "
                     "neither invest_cost nor fixed_cost the invested amount is arbitrary."),
            silence=_DELETE_MARKER.format(marker=MARKER_PREFIX + "invest_without_cost_offered"),
            since="4.4.0",
            published="13.11.2026",
        ),
        BehaviourChange(
            id="stochastic_storage_paths_tied",
            summary=("In a stochastic solve, bind_within_solve now closes one loop per path: the last step of "
                     "every forecast branch ends at the level the solve's first step starts from, and with "
                     "forecasts opening in the solve's first period each forecast's first step continues from the "
                     "end of its own path. Before, only the branch listed last closed the loop and the others "
                     "ended free (a fan in the first period looped each forecast within its first period). With "
                     "bind_forward_only and a free start (no fix_start, no carried state), the forecasts opening "
                     "in the solve's first period now start at the same level; before, each chose its own "
                     "(results change). With storage_start_end_method fix_start the start of each forecast is "
                     "the storage_state_start share of that forecast's own storage capacity, so where the "
                     "capacity differs by forecast (investment or divestment per forecast) the starting energy "
                     "differs. The upgrade marks every stochastic database with a storage node on "
                     "bind_within_solve, or on bind_forward_only without a fixed start; such a "
                     "bind_forward_only node changes only when a fan opens in the solve's first period."),
            silence=_DELETE_MARKER.format(marker=MARKER_PREFIX + "stochastic_storage_paths_tied"),
            since="4.4.0",
            published="13.11.2026",
        ),
        BehaviourChange(
            id="chained_solve_discount_from_run_start",
            summary=("With a social_discount_rate other than 0, every solve now discounts its investments and its "
                     "operations from the start of the run, as one solve over the same periods does. Before, a "
                     "later solve of a run of several solves (also a split solve of a 2-D period map, a rolling "
                     "or nested solve starting after the run start) discounted its operations, and when its "
                     "years_represented held only its own periods also its investments, from its own start; and "
                     "a fractional years_represented counted as a whole year in the discounting of the periods "
                     "after it; a years_represented map that listed the periods out of the solve's period order "
                     "(or an unused period before a used one) discounted the investments in the period order and "
                     "the operations in the map order, also in a single solve, and now both in the map order. "
                     "Where only the investments were discounted from the run start (hand-written "
                     "solves with the whole years_represented map), the balance between investment and "
                     "operating costs was wrong and decisions change; otherwise the objectives, investment "
                     "duals and CO2 cap prices of later solves are now on the run's discount base, and "
                     "decisions can still change through absolute MIP gaps, objective targets or bounds and "
                     "terms without a discount factor (results change)."),
            silence=_DELETE_MARKER.format(marker=MARKER_PREFIX + "chained_solve_discount_from_run_start"),
            since="4.4.0",
            published="13.11.2026",
        ),
        BehaviourChange(
            id="rolling_cyclic_storage_anchored",
            summary=("In a rolling_window solve without representative periods, a storage node with "
                     "storage_binding_method bind_within_solve (also bind_within_solve_blended_weights, which such a "
                     "solve uses as bind_within_solve) now continues each roll from the level realised by the "
                     "previous roll, and each roll ends its horizon at least at the level it started from. The "
                     "roll(s) whose horizon reaches the end of the solve end at the level the first roll started "
                     "from, so the year (the solve) returns the energy it took. An upper-level fix_quantity or "
                     "fix_usage at the end of a roll's horizon replaces these end conditions (fix_price keeps the "
                     "closure and only values the end), and fix_end wins at the last step of the model. "
                     "bind_within_period (also bind_within_period_blended_weights, and bind_within_timeblock, which "
                     "such a solve uses as bind_within_period) and bind_intraperiod_blocks do the same per period "
                     "where a roll starts inside one: "
                     "the roll continues it from the carried level and the roll(s) holding its end close it at the "
                     "level it started from. Before, each roll cycled on itself from a free level (or the blocks of "
                     "a period held in part chained forward without closing it), so the level jumped at roll "
                     "boundaries and the rolls could use energy that was never produced (results change). The "
                     "closing level is the start level in MWh, at most the storage capacity of the closing period."),
            silence=_DELETE_MARKER.format(marker=MARKER_PREFIX + "rolling_cyclic_storage_anchored"),
        ),
        BehaviourChange(
            id="nested_storage_start_borrowed",
            summary=("A storage node of a nested lower-level solve whose end an upper-level solve fixes "
                     "(storage_nested_fix_method fix_quantity, fix_usage or fix_price) and that has no state "
                     "realised before the lower solve now starts from the level the upper solve planned for the "
                     "same moment, when it chains forward (bind_forward_only, or bind_within_solve in a rolling "
                     "solve under fix_quantity or fix_usage, and bind_within_period (also bind_within_timeblock) "
                     "there when the first roll cuts the period) and fix_start does not apply (no fix_start, or the "
                     "lower solve starts after the start of the model). Before, it started free or from fix_start, "
                     "so the lower solve could start fuller than the upper plan and keep the difference (results "
                     "change)."),
            silence=_DELETE_MARKER.format(marker=MARKER_PREFIX + "nested_storage_start_borrowed"),
        ),
        BehaviourChange(
            id="nested_storage_fix_cumulative",
            summary=("A lower-level solve under an upper-level storage_nested_fix_method fix_usage is now capped by "
                     "the upper solve's usage from the start of the period (within the instance nested in that "
                     "upper solve) to the end of its horizon, less what its earlier rolls realised, with an upper "
                     "step that the span cuts counted pro rata to its hours. Before, each roll was capped by the "
                     "upper usage of every upper step it touched, counted whole and from its own start, so the "
                     "realised rolls could use energy the upper solve never planned. Under fix_quantity a lower "
                     "horizon that ends inside an upper step now takes the upper level interpolated at that moment "
                     "instead of the level at the end of the step (results change)."),
            silence=_DELETE_MARKER.format(marker=MARKER_PREFIX + "nested_storage_fix_cumulative"),
        ),
        BehaviourChange(
            id="rp_storage_chain_pinned",
            summary=("In a representative-period solve, storage_state_start, storage_state_end and "
                     "use_reference_value of a *_blended_weights storage node now set the stored level at the start "
                     "and the end of its base-period chain (the chronological level the representative blocks add up "
                     "to), not a state inside the first or last representative block, which only placed the block's "
                     "free offset. The level a bind_forward_only_blended_weights chain starts from is now within the "
                     "storage capacity, and a bind_within_solve_blended_weights or "
                     "bind_within_period_blended_weights node takes no use_reference_value (its cycle closes on "
                     "itself). A solve with representative periods that follows another solve starts the chain of a "
                     "bind_forward_only_blended_weights node from the realised level, and hands on its chain level "
                     "at the end of its realised period. Before, a forward chain started and ended at free levels "
                     "whatever these parameters "
                     "said, so an investment solve could use stored energy that was never produced (results "
                     "change)."),
            silence=_DELETE_MARKER.format(marker=MARKER_PREFIX + "rp_storage_chain_pinned"),
        ),
        BehaviourChange(
            id="storage_end_matches_start",
            summary=("A storage node that chains forward (bind_forward_only, also the default, and "
                     "bind_forward_only_blended_weights) and whose start the solve decides now ends the model at "
                     "least at that start: a free start (storage_start_end_method fix_nothing or fix_end) ends at "
                     "least at the level it started from, and a fix_start start on storage the solve can invest in "
                     "returns at least the storage_state_start share of the capacity it builds in the first period. "
                     "This holds in a solve that has both the start of the model and its last step (a single solve, "
                     "an investment solve over the whole horizon, every representative-period solve) and, with a "
                     "free start (fix_nothing), over the rolls of a rolling solve (the first roll's horizon ends at "
                     "least at its start, the rolls reaching the end of the model at least at that level). Before, the "
                     "solve could start full and end empty, or build storage only for its start energy (results "
                     "change). fix_start on capacity the solve cannot change keeps a free end."),
            silence=_DELETE_MARKER.format(marker=MARKER_PREFIX + "storage_end_matches_start"),
        ),
        BehaviourChange(
            id="storage_reference_start",
            summary=("A storage node with storage_solve_horizon_method use_reference_value and no fix_start now "
                     "starts from its storage_state_reference_value at the first step (the model's first step in "
                     "the first solve). Before, its start was free and only the matching end bounded it, so a "
                     "reference series that falls during the year did not drain the storage; now the existing "
                     "storage gives the drop (results change). With fix_start, storage_state_start still wins."),
            silence=_DELETE_MARKER.format(marker=MARKER_PREFIX + "storage_reference_start"),
        ),
        BehaviourChange(
            id="annuity_over_payback_time",
            summary=("With lifetime_method reinvest_choice or no_reinvestment the investment cost is now the level "
                     "yearly cost of the plant over its lifetime_technical (the annuity over payback_time with "
                     "discount_rate, its present value spread evenly over the lifetime), paid for each year the "
                     "capacity is in service within the horizon. Before, the annuity was paid in every period that "
                     "started before the end of the lifetime, also where the capacity was already gone (results "
                     "change). The reported investment costs follow; the fixed cost runs over the same years."),
            silence=_DELETE_MARKER.format(marker=MARKER_PREFIX + "annuity_over_payback_time"),
        ),
        BehaviourChange(
            id="fixed_cost_per_operating_period",
            summary=("New capacity now pays the fixed_cost of each period it operates in, as existing capacity does. "
                     "In FlexTool 4.5.0 an investment paid the fixed_cost of the period it came online in every "
                     "period it operated, so where fixed_cost differs between periods (a period map) the fixed cost "
                     "of new capacity changes (results change)."),
            silence=_DELETE_MARKER.format(marker=MARKER_PREFIX + "fixed_cost_per_operating_period"),
        ),
        BehaviourChange(
            id="annuity_timing",
            summary=("The yearly payment of the investment annuity (and of the yearly salvage credits) is now "
                     "sized for the moment discount_offset_investment places it in the year: the annuity "
                     "times (1 + discount_rate)^(discount_offset_investment - 1), so the loan taken when the "
                     "capacity comes online is worth invest_cost at the discount_rate. Before, every offset used "
                     "the year-end annuity, so at the default offset 0 (payments at the start of each year) an "
                     "investment cost invest_cost times (1 + discount_rate). Investments are now cheaper by one "
                     "year of interest at offset 0 (results change); offset 1 is unchanged."),
            silence=_DELETE_MARKER.format(marker=MARKER_PREFIX + "annuity_timing"),
        ),
        BehaviourChange(
            id="net_salvage_once",
            summary=("The net_salvage_value is now counted once, as a cash flow when the capacity stops operating. A "
                     "retirement of existing capacity receives the whole net_salvage_value of the retirement period "
                     "at its start (the investment in existing capacity is sunk; before: the salvage annuity over "
                     "lifetime_technical, inside the horizon). With lifetime_method reinvest_choice or "
                     "no_reinvestment the net_salvage_value at the end of life lowers the yearly cost of new "
                     "capacity (levelled over lifetime_technical at discount_rate; before: no effect), existing "
                     "capacity holds it as a level yearly credit (a constant that is not in the reported "
                     "objective) and a retirement gives the credits back. Retiring long-lived "
                     "capacity late in the horizon can now pay, since the years after the horizon are not seen "
                     "(results change)."),
            silence=_DELETE_MARKER.format(marker=MARKER_PREFIX + "net_salvage_once"),
        ),
        BehaviourChange(
            id="built_capacity_retirable",
            summary=("With lifetime_method reinvest_choice or no_reinvestment and an invest_method that allows "
                     "investment and retirement (invest_retire_* or cumulative_limits), the capacity the model "
                     "builds can now be retired too (before: only existing capacity). A retired MW pays its loan "
                     "in full at the start of the retirement period, receives the net_salvage_value then and saves "
                     "its fixed cost from then on; retirement limits and user-constraint retirement terms count "
                     "it, and a chain of solves carries each retirement with its cohort (results change)."),
            silence=_DELETE_MARKER.format(marker=MARKER_PREFIX + "built_capacity_retirable"),
        ),
        BehaviourChange(
            id="automatic_replacement_cost",
            summary=("With lifetime_method reinvest_automatic and a lifetime_technical, capacity is now replaced at "
                     "the end of its life by a new plant, at the invest_cost, discount_rate, payback_time, "
                     "lifetime_technical and net_salvage_value of the period the replacement comes online, and "
                     "replaced again at its own end of life to the end of the horizon. Before, it paid the annuity "
                     "of its first plant for ever, without its net_salvage_value. Existing capacity reaches its end "
                     "of life and is replaced the same way (free without an invest_cost), and reinvest_automatic "
                     "capacity, built or existing, can be retired when its invest_method allows it: retiring it "
                     "also avoids every replacement still to come (results change)."),
            silence=_DELETE_MARKER.format(marker=MARKER_PREFIX + "automatic_replacement_cost"),
        ),
        BehaviourChange(
            id="end_of_life_closest_seam",
            summary=("With lifetime_method reinvest_choice or no_reinvestment, the end of life (start of operation "
                     "plus lifetime_technical; existing capacity from the model start) is now rounded to the "
                     "closest period start (end_of_life_method closest_seam, the new default; a tie goes to the "
                     "later start). Before, it was rounded to the next period start, so the capacity operated to "
                     "the end of the period its life ended in. Capacity whose life ends in the first half of a "
                     "period is now out of service, without fixed cost, for that whole period (results change)."),
            silence="Set end_of_life_method on these entities: next_seam keeps the old rounding, closest_seam the new.",
        ),
        BehaviourChange(
            id="gui_solver_defaults",
            summary=("Runs started from the FlexTool GUI now use the same solver settings as terminal and Spine "
                     "Toolbox runs: presolve is on (before: choose), and the MIP gap of each solve comes from its "
                     "solver_mip_gap, else mip_rel_gap in solver_config/highs.opt (before: 0.001 for every "
                     "solve; the GUI no longer has a MIP-gap setting). Results of MIP solves (integer "
                     "investments, online variables) can change within the gap; an LP keeps its optimal "
                     "objective value."),
            silence=("This notice is shown once. To keep the old behaviour, choose presolve 'Choose' in the "
                     "Solver options dialog and set solver_mip_gap on the solves in the database."),
            since="4.2.0",
            published="7.11.2026",
        ),
    )
}

_markers: dict[str, str] = {}
_announced: set[tuple[str, str]] = set()
_warned: set = set()


def read_markers(db) -> dict[str, str]:
    """``{change id: marker value}`` of the behaviour-change markers of an open ``DatabaseMapping``."""
    out: dict[str, str] = {}
    for md in db.get_items("metadata"):
        name = str(md["name"])
        if name.startswith(MARKER_PREFIX):
            out[name[len(MARKER_PREFIX):]] = str(md["value"])
    return out


def read_markers_from_url(db_url: str) -> dict[str, str]:
    """:func:`read_markers` of the database at *db_url* (a read-only look at its metadata table).

    A database that cannot be opened gives ``{}``: the run reports that itself when it reads the model.
    """
    from spinedb_api import SpineDBAPIError
    from flextool.update_flextool.db_handle import open_database_mapping

    try:
        with open_database_mapping(str(db_url)) as db:
            return read_markers(db)
    except SpineDBAPIError:
        return {}


def start_run(markers: Mapping[str, str] | None) -> None:
    """Begin a run: remember the input database's markers and forget what earlier runs announced."""
    _markers.clear()
    _markers.update(dict(markers or {}))
    _announced.clear()
    _warned.clear()


def warn_once(key, text: str) -> bool:
    """Print *text* unless a message with *key* was printed earlier in this run (rolls and repeated solves of a
    run print it once).  Returns True when it printed."""
    if key in _warned:
        return False
    _warned.add(key)
    print(text, flush=True)
    return True


def warn_once_each(kind: str, names: list[str], text_of) -> bool:
    """Print ``text_of(fresh)`` for the ``names`` not named in a ``kind`` message earlier in this run (each name
    once per run, however the solves group them).  Returns True when it printed."""
    fresh = [n for n in names if (kind, n) not in _warned]
    if not fresh:
        return False
    _warned.update((kind, n) for n in fresh)
    print(text_of(fresh), flush=True)
    return True


def complete_solve_of(solve: str | None) -> str:
    """The complete solve a sub-solve name belongs to: a roll ``<solve>_roll_<k>`` counts as ``<solve>``."""
    import re
    return re.sub(r"_roll_\d+$", "", solve) if solve else ""


def storages_named(names: Iterable[str], noun: str = "storage node") -> str:
    """``"1 storage node (A)"`` / ``"12 storage nodes (A, ...)"``: how many and the first only (owner rule for the
    storage messages: one console line per message type and solve, never a line or a list per node)."""
    names = list(dict.fromkeys(str(n) for n in names))
    count = len(names)
    first = names[0] if names else ""
    return f"{count} {noun}{'' if count == 1 else 's'} ({first}{', ...' if count > 1 else ''})"


def storage_message(kind: str, solve: str | None, names: Iterable[str], text_of, details: str = "") -> bool:
    """Print a storage warning or notice: one line per message ``kind`` and complete solve (the rolls of a rolling
    solve count as one), ``text_of(named)`` where ``named`` is :func:`storages_named` of ``names`` (how many and
    the first).  ``details`` (the values per node) go to the log at debug level, not to the console.  Returns True
    when it printed."""
    import logging
    names = list(dict.fromkeys(str(n) for n in names))
    if not names:
        return False
    key = ("storage_message", kind, complete_solve_of(solve))
    if key in _warned:
        return False
    _warned.add(key)
    print(text_of(storages_named(names)), flush=True)
    if details:
        logging.getLogger("flextool.engine.storage").debug(details)
    return True


def mark_once(key) -> bool:
    """Record *key* for this run without printing (a message that goes to the log, once per run).  Returns True
    the first time in a run, False after that; :func:`start_run` forgets it."""
    if key in _warned:
        return False
    _warned.add(key)
    return True


def already_warned(key) -> bool:
    """Whether :func:`warn_once` printed a message with *key* earlier in this run."""
    return key in _warned


def has_marker(change_id: str) -> bool:
    return change_id in _markers


def marker_value(change_id: str) -> str | None:
    return _markers.get(change_id)


def marker_json(change_id: str) -> dict:
    """The marker value of *change_id* parsed as a JSON object; ``{}`` when absent or not JSON."""
    raw = _markers.get(change_id)
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _installed_version() -> str:
    try:
        return metadata.version("flextool")
    except metadata.PackageNotFoundError:
        return "unknown"


def version_phrase(change: BehaviourChange) -> str:
    """``FlexTool 4.2.0 (published 1.12.2026)``, or the not-yet-published form for an unreleased change."""
    if change.since is None:
        return f"the FlexTool release after {_installed_version()} (not yet published)"
    if change.published is None:
        return f"FlexTool {change.since}"
    return f"FlexTool {change.since} (published {change.published})"


def notice_parts(change_id: str, kind: str, names: Iterable[str], detail: str = "") -> list[str]:
    """The notice of *change_id* in its separate items: what changed, the affected entities, the *detail* (when
    given) and how to silence it.  :func:`notice_text` joins them into the printed line."""
    change = BEHAVIOUR_CHANGES[change_id]
    parts = [f"Behaviour changed in {version_phrase(change)}: {change.summary}",
             f"Affected {kind}: {', '.join(names)}."]
    if detail:
        parts.append(detail)
    parts.append(change.silence)
    return parts


def notice_text(change_id: str, kind: str, names: Iterable[str], detail: str = "") -> str:
    return " ".join(notice_parts(change_id, kind, names, detail))


def display_text(notices: Iterable[Iterable[str]]) -> str:
    """Notices (each as :func:`notice_parts`) for a dialog: each notice its own paragraph with its items on
    their own lines, a blank line between notices."""
    return "\n\n".join("\n".join(parts) for parts in notices)


def unannounced(change_id: str, names: Iterable[str]) -> list[str]:
    """The *names* (de-duplicated, in order) that :func:`notice` has not yet announced for *change_id* in this
    run, so that a caller can build its ``detail`` from the fresh names only."""
    return [n for n in dict.fromkeys(str(n) for n in names) if (change_id, n) not in _announced]


def notice(change_id: str, kind: str, names: Iterable[str], detail: str = "", solve: str | None = None) -> bool:
    """Print the notice of *change_id* for the *names* not yet announced in this run.

    Fires only while the input database carries the change's marker.  Returns True when a line was printed.
    With *solve* (the storage notices): once per change and complete solve, naming how many entities and the first
    only (:func:`storages_named`); the full list goes to the log at debug level.
    """
    if change_id not in BEHAVIOUR_CHANGES:
        raise KeyError(f"unknown behaviour change {change_id!r}")
    if change_id not in _markers:
        return False
    if solve is not None:
        names = list(dict.fromkeys(str(n) for n in names))
        key = ("notice", change_id, complete_solve_of(solve))
        if not names:
            return False
        import logging
        if key in _warned:
            fresh = [n for n in names if (change_id, n) not in _announced]
            if fresh:
                _announced.update((change_id, n) for n in fresh)
                logging.getLogger("flextool.engine.storage").debug(
                    "Behaviour change %s in solve '%s', also affected %s: %s", change_id,
                    complete_solve_of(solve), kind, ", ".join(fresh))
            return False
        _warned.add(key)
        _announced.update((change_id, n) for n in names)
        affected = names[0] if len(names) == 1 else f"{len(names)} {kind} ({names[0]}, ...)"
        print(notice_text(change_id, kind, [affected], detail), flush=True)
        logging.getLogger("flextool.engine.storage").debug(
            "Behaviour change %s in solve '%s', affected %s: %s", change_id, complete_solve_of(solve), kind,
            ", ".join(names))
        return True
    fresh = unannounced(change_id, names)
    if not fresh:
        return False
    _announced.update((change_id, n) for n in fresh)
    print(notice_text(change_id, kind, fresh, detail), flush=True)
    return True
