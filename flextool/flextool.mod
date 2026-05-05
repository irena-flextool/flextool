# © International Renewable Energy Agency 2018-2022

#The FlexTool is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License
#as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

#The FlexTool is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
#without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

#You should have received a copy of the GNU Lesser General Public License along with the FlexTool.
#If not, see <https://www.gnu.org/licenses/>.

#Authors:
# Juha Kiviluoma, VTT Technical Research Centre of Finland (2017-2025), Nodal-Tools Ltd (2025-2026)
# Arttu Tupala, VTT Technical Research Centre of Finland (2023-2026)

param datetime0 := gmtime();

#########################
# Fundamental sets of the model
set entity 'e - contains both nodes and processes';
set process 'p - Particular activity that transfers, converts or stores commodities' within entity;
set node 'n - Any location where a balance needs to be maintained' within entity;
set group 'g - Any group of entities that have a set of common constraints';
set commodity 'c - Stuff that is being processed';
set period_time '(d, t) - Time steps in the time periods of the whole timeline' dimen 2;
set period__time_first within period_time;
set period__time_last within period_time;
set solve_period_timeset '(solve, d, tb) - All solve, period, timeset combinations in the model instance' dimen 3;
set solve_period '(solve, d) - Time periods in the solves to extract periods that can be found in the full data' dimen 2;  # Migrated to Python (preprocessing/simple_projections.py).
set period_capacity;  # Periods for which capacities have been already been output
set period_solve 'picking up periods from solve_period';  # Migrated to Python (preprocessing/simple_projections.py).
set solve_current 'current solve name' dimen 1;
set period_from_model dimen 1;
set period_from_period_time;  # Migrated to Python (preprocessing/per_solve_sets.py).
set period 'd - Time periods in the current solve';  # Migrated to Python (preprocessing/per_solve_sets.py).
set period_first dimen 1 within period;
set period_last dimen 1 within period;
set branch_all dimen 1;
set time_branch_all dimen 1;
set period__branch dimen 2 within {period, period};
set branch;  # Migrated to Python (preprocessing/per_solve_sets.py).
set period__year dimen 2;
set year 'y - Years for discount calculations';  # Migrated to Python (preprocessing/per_solve_sets.py).
set timeline__timestep__duration dimen 3;
set time 't - Time steps in the current timelines';  # Migrated to Python (preprocessing/simple_projections.py).
set timeset__timeline dimen 2;
set timeline;  # Migrated to Python (preprocessing/simple_projections.py).
set period__timeline dimen 2;  # Migrated to Python (preprocessing/per_solve_sets.py).
set method 'm - Type of process that transfers, converts or stores commodities';
set upDown 'upward and downward directions for some variables';
set ct_method;
set ct_method_constant within ct_method;
set ct_method_regular within ct_method;
set startup_method;
set startup_method_no within startup_method;
set fork_method;
set fork_method_yes within fork_method;
set fork_method_no within fork_method;
set reserve_method;
set ramp_method;
set ramp_limit_method within ramp_method;
set ramp_cost_method within ramp_method;
set profile;
set profile_method;
set debug 'flags to output debugging and test results';
set test_dt 'a shorter set of time steps for printing out test results' dimen 2;
set test_t dimen 1;
set model 'dummy set because has to load a table';

set constraint 'user defined greater than, less than or equality constraints between inputs and outputs';
set sense 'sense of user defined constraints';
set sense_greater_than within sense;
set sense_less_than within sense;
set sense_equal within sense;

set commodityParam;
set commodityPeriodParam within commodityParam;
set commodityTimeParam within commodityParam;
set nodeParam;
set nodeParam_def1 within nodeParam;
set nodePeriodParam;
set nodePeriodParamRequired within nodePeriodParam;
set nodePeriodParamInvest within nodePeriodParam;
set nodeTimeParam within nodeParam;
set nodeTimeParamRequired within nodeParam;
set processParam;
set processParam_def1 within processParam;
set processPeriodParam;
set processTimeParam within processParam;
set processPeriodParamRequired within processPeriodParam;
set processPeriodParamInvest within processPeriodParam;
set processTimeParamRequired within processParam;
set sourceSinkParam;
set sourceSinkTimeParam within sourceSinkParam;
set sourceSinkTimeParamRequired within sourceSinkParam;
set sourceSinkPeriodParam within sourceSinkParam;
set sourceSinkPeriodParamRequired within sourceSinkParam;
set reserveParam;
set reserveTimeParam within reserveParam;
set groupParam;
set groupPeriodParam;
set groupTimeParam within groupParam;

set exclude_entity_outputs;
set def_optional_outputs dimen 2;
set optional_outputs dimen 2;
set optional_yes;  # Migrated to Python (preprocessing/simple_projections.py); loaded via table data IN below.
set def_optional_yes;  # Migrated to Python (preprocessing/simple_projections.py).
set enable_optional_outputs;  # Migrated to Python (preprocessing/simple_projections.py).

set reserve__upDown__group__method dimen 4;
set reserve__upDown__group dimen 3;  # Migrated to Python (preprocessing/simple_projections.py); loaded via table data IN below.
set reserve 'r - Categories for the reservation of capacity_existing';  # Migrated to Python (preprocessing/reserve_method_partitions.py).
set reserve__upDown__group__reserveParam__time dimen 5 within {reserve, upDown, group, reserveTimeParam, time};

set group__param dimen 2 within {group, groupParam};
set group__param__period dimen 3; # within {group, groupPeriodParam, periodAll};
set group__param__time dimen 3 within {group, groupTimeParam, time};
set node__param__period dimen 3; # within {node, nodePeriodParam, periodAll};
set commodity__param__period dimen 3; # within {commodity, commodityPeriodParam, periodAll};
set commodity__param__time dimen 3; # within {commodity, commodityTimeParam, time};
set process__param__period dimen 3; # within {process, processPeriodParam, periodAll};

# period_* sets migrated to Python (preprocessing/period_param_sets.py).
# Each is the projection of the period column out of the corresponding
# pd_<class>.csv file written by input_writer.write_parameter.
set period_group 'picking up periods from group data';
set period_node 'picking up periods from node data';
set period_commodity 'picking up periods from commodity data';
set period_process 'picking up periods from process data';

set periodAll 'd - Time periods in data (including those currently in use)';  # Migrated to Python (preprocessing/per_solve_sets.py).


param p_group {g in group, groupParam} default 0;
param pd_group {g in group, groupPeriodParam, d in periodAll} default 0;
param pt_group {g in group, groupTimeParam, time} default 0;
param p_group__process {g in group, p in process, groupParam};


#Method collections use in the model (abstracted methods)
set method_1var_off within method;
set method_1way_off within method;
set method_1way_LP within method;
set method_1way_MIP within method;
set method_2way_off within method;
set method_1way_1var_on within method;
set method_1way_nvar_on within method;
set method_1way within method;
set method_1way_1var within method;
set method_2way_1var within method;
set method_2way within method;
set method_2way_2var within method;
set method_2way_nvar within method;
set method_1way_on within method;
set method_2way_on within method;
set method_1var within method;
set method_nvar within method;
set method_off within method;
set method_on within method;
set method_LP within method;
set method_MIP within method;
set method_direct within method;
set method_indirect within method;
set method_1var_per_way within method;

set invest_method 'methods available for investments';
set invest_method_not_allowed 'method for denying investments' within invest_method;
set divest_method_not_allowed 'method for denying divestments' within invest_method;
set lifetime_method 'methods available for end of lifetime behavior';
set lifetime_method_default within lifetime_method;
set co2_method 'methods available for co2 price and limits';
set co2_price_method within co2_method;
set co2_max_period_method within co2_method;
set co2_max_total_method within co2_method;
set price_method 'methods available for commodity price/ladder';
set entity__invest_method 'the investment method applied to an entity' dimen 2 within {entity, invest_method};
set entityDivest;  # Migrated to Python (preprocessing/invest_method_sets.py).
set entityInvest;  # Migrated to Python (preprocessing/invest_method_sets.py).
set entity__lifetime_method_read dimen 2 within {entity, lifetime_method};
set entity__lifetime_method 'the lifetime method applied to an entity' dimen 2;  # Migrated to Python (preprocessing/method_with_fallback_sets.py).
set group__invest_method 'the investment method applied to a group' dimen 2 within {group, invest_method};
set group_invest;  # Migrated to Python (preprocessing/invest_method_sets.py).
set group_divest;  # Migrated to Python (preprocessing/invest_method_sets.py).
set group__co2_method 'the investment method applied to a group' dimen 2 within {group, co2_method};
set group_co2_price;       # Migrated to Python (preprocessing/co2_method_sets.py).
set group_co2_max_period;  # Migrated to Python (preprocessing/co2_method_sets.py).
set group_co2_max_total;   # Migrated to Python (preprocessing/co2_method_sets.py).
set node_type 'enum universe of node_type values';
param p_node_type {n in node} symbolic in node_type, default 'balance';
# node-type partitions migrated to Python (preprocessing/node_type_sets.py).
# Default 'balance' on p_node_type is materialized in Python — every node
# in input/node.csv is assigned its explicit type or 'balance' before the
# four sets below are written to solve_data/.
set nodeCommodity;
set nodeBalance;
set nodeState;
set nodeBalancePeriod;
set inflow_method 'method for scaling the inflow';
set inflow_method_default within inflow_method;
set node__inflow_method_read 'method for scaling the inflow applied to a node' within {node, inflow_method};
set node__inflow_method dimen 2 within {node, inflow_method};  # Migrated to Python (preprocessing/method_with_fallback_sets.py).
set storage_binding_method 'methods for binding storage state between periods';
set storage_binding_method_default within storage_binding_method;
set node__storage_binding_method_read within {node, storage_binding_method};
set node__storage_binding_method dimen 2 within {node, storage_binding_method};  # Migrated to Python (preprocessing/method_with_fallback_sets.py).
set storage_start_end_method 'method to fix start and/or end value of storage in a model run';
set node__storage_start_end_method within {node, storage_start_end_method};
set storage_solve_horizon_method 'methods to set reference value or price for the end of horizon storage state';
set node__storage_solve_horizon_method within {node, storage_solve_horizon_method};
set storage_nested_fix_method 'methods to set the storage value for lower level solves';
set node__storage_nested_fix_method within {node, storage_nested_fix_method};

# Representative period sets and parameters (populated only when RP weights are used)
set rp_base__rep 'weight matrix (base_period_start, rep_period_start)' dimen 2;
set rp_base_chain 'chronological chain of base periods (current, previous)' dimen 2;
set rp_base_first 'first base period start timestep' dimen 1;
set rp_base_last 'last base period start timestep' dimen 1;
set rp_block_first 'first timestep of each RP block (period, step)' dimen 2;
set rp_block_last 'last timestep of each RP block (period, step)' dimen 2;
set rp_base_period;  # Migrated to Python (preprocessing/per_solve_sets.py).
set rp_rep_period;   # Migrated to Python (preprocessing/per_solve_sets.py).
set nodeState_rp;  # Migrated to Python (preprocessing/simple_projections.py::write_node_state_subsets).

# Intraperiod-blocks sets (bind_intraperiod_blocks storage binding method).
# Block = maximal contiguous run of active timesteps in the timeline. The Python
# runner emits period_block_time.csv (one row per active step with its block's
# first-step label) and period_block_succ.csv (cyclic successor of each block
# within its period).
set period_block_time 'active timestep tagged with its block (period, block_first, step)' dimen 3;
set period_block_succ 'cyclic block successor within a period (period, block_first, block_first_next)' dimen 3;
set period_block dimen 2;  # Migrated to Python (preprocessing/per_solve_sets.py).
set nodeStateBlock;  # Migrated to Python (preprocessing/simple_projections.py::write_node_state_subsets).

# Temporal-resolution block abstraction (Agents 1.1/1.2): per-entity resolution
# classes.  A node's balance equation is emitted at the node's block; a
# process's flow variables are emitted at their adjacent node's block (one
# per side, derived in Python by flextoolrunner/blocks.py).  In the degenerate
# case (no group has new_stepduration set) everything maps to block "default".
# Declarations are inert in this agent — Agents 1.3+ consume them.
set node__block 'per-node block assignment' dimen 2;
set process_side 'source/sink side label';  # Migrated to Python (preprocessing/simple_projections.py).
set process__side__block 'per-process per-side block assignment' dimen 3;
# Agent 1.6: per-process unified block used by UC (online/startup/shutdown),
# ramp and profile constraints.  Equals the process's explicit resolution-
# group block if set, else the finer of (process_block_in,
# process_block_out).  Every process has exactly one row.  In the degenerate
# case all rows carry ``'default'``, so UC / ramp / profile constraints
# indexed via this set reduce to the pre-v51 fine-grid domain.
set process__block 'per-process unified block (finer-of-sides, or group override)' dimen 2;
# block__period__step companion set carries the keys of block_step_duration
# so the parameter can be iterated without relying on a default-0 sentinel.
set block__period__step 'active (block, period, step) triples' dimen 3;
# Overlap fraction M_{b_coarse, t_coarse; b_fine, t_fine} from Gao/
# Morales-España 2025.  Used by Agent 1.3's generalized balance equation to
# aggregate flows across resolutions.  In the degenerate (single-block) case
# every (d, t, b, t, b) row carries fraction 1.0 — identity.
set overlap 'overlap tuples (period, block_coarse, step_coarse, block_fine, step_fine)' dimen 5;
# Per-block predecessor relations (Agent 1.4). Analogous to dtttdt but
# keyed at the block level — (block, period, step, step_previous,
# step_previous_within_timeset, period_previous, step_previous_within_solve).
# In the degenerate case the 'default' rows are the same tuples as dtttdt,
# so storage state-transition terms using block_dtttdt on the node's block
# collapse to the pre-v51 form bit-identically.
set block_dtttdt 'per-block predecessor relations' dimen 7;
# Per-block first / last step of each period (Agent 1.4). In the
# degenerate case ('default' block) these match period__time_first /
# period__time_last exactly.
set block__period__time_first 'per-block period first step' dimen 3;
set block__period__time_last 'per-block period last step' dimen 3;
# Block set derived from the union of blocks referenced in the four CSVs.
set block 'temporal-resolution classes for nodes and processes';  # Migrated to Python (preprocessing/per_solve_sets.py).

set node__profile__profile_method dimen 3 within {node,profile,profile_method};
set group_node 'member nodes of a particular group' dimen 2 within {group, node};
set group_process 'member processes of a particular group' dimen 2 within {group, process};
set group_process_node 'process__nodes of a particular group' dimen 3 within {group, process, node};
set group_entity dimen 2;  # Migrated to Python (preprocessing/union_sets.py).
set groupInertia 'node groups with an inertia constraint' within group;
set groupNonSync 'node groups with a non-synchronous constraint' within group;
set groupCapacityMargin 'node groups with a capacity margin' within group;
set nodeGroupIndicators 'groups that output node-group indicators' within group;
set flowGroupIndicators 'groups that output flow-group indicators' within group;
set nodeGroupDispatch 'groups that will output the node-group dispatch table' within group;
set nodeGroupDispatch_node 'dispatch groups with node members';  # Migrated to Python (preprocessing/structural_filters.py).
set flowAggregator 'groups that aggregate flows into a node-group dispatch' within group;

set group__loss_share_type dimen 2;
set group_loss_share 'group that share the loss of load (upward penalty)';  # Migrated to Python (preprocessing/simple_projections.py).

set process_unit 'processes that are unit' within process;
set process_connection 'processes that are connections' within process;
set process__ct_method_read dimen 2 within {process, ct_method};
set process__ct_method dimen 2 within {process, ct_method};  # Migrated to Python (preprocessing/method_with_fallback_sets.py).
set process__startup_method_read dimen 2 within {process, startup_method} default {p in process, 'no_startup'} ;
set process__startup_method dimen 2 within {process, startup_method};  # Migrated to Python (preprocessing/method_with_fallback_sets.py).
set process_node_ramp_method dimen 3 within {process, node, ramp_method};
set methods dimen 4;
set process__profile__profile_method dimen 3 within {process, profile, profile_method};
set process__node__profile__profile_method dimen 4 within {process, node, profile, profile_method};
set process_source dimen 2 within {process, entity};
set process_sink dimen 2 within {process, entity};

set process__sink_nonSync_unit dimen 2 within {process, node};
set process_nonSync_connection dimen 1 within {process};
# Migrated to Python (preprocessing/nonsync_sets.py); original derivation
# replaced with this forward declaration so it precedes its table data IN
# reader. See flextool.mod:2017+ comment for the original mod logic.
set process__group_inside_group_nonSync dimen 2;

set node_dc_power_flow 'nodes participating in DC power flow' within node;
set connection_dc_power_flow 'connections with DC power flow angle constraints' within process_connection;
set node_reference_angle 'reference bus nodes (angle fixed to 0)' within node;

set process_reserve_upDown_node dimen 4;
set process_node_flow_constraint dimen 3 within {process, node, constraint};
set process_capacity_constraint_invested dimen 2 within {process, constraint};
set node_capacity_constraint_invested dimen 2 within {node, constraint};
set process_capacity_constraint_prebuilt dimen 2 within {process, constraint};
set node_capacity_constraint_prebuilt dimen 2 within {node, constraint};
set node_state_constraint dimen 2 within {node, constraint};
set constraint__sense dimen 2 within {constraint, sense};
set commodity_node dimen 2 within {commodity, node};
# Tier indices used by the commodity price ladder.  The sets are split
# per price_method so the two parameter shapes can be read independently
# from their own CSVs:
#   commodity__tier_cum → tiers for price_ladder_cumulative (one price/
#                         quantity per tier, period-agnostic)
#   commodity__tier_ann → tiers for price_ladder_annual (price/quantity
#                         per tier, per period)
# `tier` is their union — the old single-ladder users of the tier set
# (e.g. `v_trade {(c,n,d,i) in cndi_ladder}`) still index across both.
set commodity__tier_cum dimen 2;
# commodity__tier__period_ann carries the raw (c, tier, period) triples
# read from commodity_ladder_annual.csv.  The annual tier membership set
# is derived from it — GMPL dedupes projections via `setof`, so the CSV
# reader can safely emit multiple rows per (c, tier) without triggering
# a "duplicate tuple" error.
set commodity__tier__period_ann dimen 3;
set commodity__tier_ann dimen 2;  # Migrated to Python (preprocessing/simple_projections.py::write_simple_setof_projections).
set commodity__tier dimen 2;  # Migrated to Python (preprocessing/simple_projections.py::write_commodity_tier_sets).
set tier;                     # Migrated to Python (preprocessing/simple_projections.py::write_commodity_tier_sets).

set dt dimen 2 within period_time;
param dt_jump {(d, t) in dt};
set dtttdt dimen 6;
set dtt dimen 3;  # Migrated to Python (preprocessing/per_solve_sets.py).
set period_invest dimen 1 within period;
set d_realize_invest dimen 1 within period;
set period_with_history dimen 1 within periodAll;
param p_period_from_solve{period_with_history};
set time_in_use;    # Migrated to Python (preprocessing/per_solve_sets.py).
set period_in_use;  # Migrated to Python (preprocessing/per_solve_sets.py).
set period_first_of_solve dimen 1 within period;

set dt_realize_dispatch_input dimen 2 within period_time;
set dt_realize_dispatch dimen 2;  # Migrated to Python (preprocessing/per_solve_sets.py).
set d_realized_period;            # Migrated to Python (preprocessing/per_solve_sets.py).
set realized_period__time_last dimen 2 within period_time;
set d_realize_dispatch_or_invest;  # Migrated to Python (preprocessing/per_solve_sets.py).
#dt_complete is the timesteps of the whole rolling_window set, not just single roll. For single_solve it is the same as dt
set dt_complete dimen 2 within period_time;
set complete_time_in_use;  # Migrated to Python (preprocessing/per_solve_sets.py).
param complete_step_duration{(d, t) in dt_complete};
set timeline_steps dimen 2;  # Migrated to Python (preprocessing/simple_projections.py).
param p_timeline_step_duration{timeline_steps};
param p_timeline_duration_in_years{tl in timeline};  # Migrated to Python (preprocessing/period_calculated_params.py).
table data IN 'CSV' 'solve_data/p_timeline_duration_in_years.csv' : [timeline], p_timeline_duration_in_years~value;

set dt_fix_storage_timesteps dimen 2 within period_time;
set d_fix_storage_period;  # Migrated to Python (preprocessing/per_solve_sets.py).
set ndt_fix_storage_price dimen 3 within  {node, period_solve, time};
set ndt_fix_storage_quantity dimen 3 within  {node, period_solve, time};
set ndt_fix_storage_usage dimen 3 within  {node, period_solve, time};
set n_fix_storage_quantity;  # Migrated to Python (preprocessing/per_solve_sets.py).
set n_fix_storage_price;     # Migrated to Python (preprocessing/per_solve_sets.py).
set n_fix_storage_usage;     # Migrated to Python (preprocessing/per_solve_sets.py).
set dtt_timeline_matching dimen 3 within {period,time,time};

param p_fix_storage_price {node, period_solve, time};
param p_fix_storage_quantity {node, period_solve, time};
param p_fix_storage_usage {node, period_solve, time};
param p_roll_continue_state {node};

set startTime dimen 1 within time;
set startNext dimen 1 within time;
set modelParam;

set process__param dimen 2 within {process, processParam};
set process__param__time dimen 3 within {process, processTimeParam, time};
set process__param_t dimen 2;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/process__param_t.csv' : process__param_t <- [process, param];
set profile_param dimen 1 within {profile};
set profile_param__time dimen 2 within {profile, time};

set connection__param dimen 2;  # Migrated to Python (preprocessing/structural_filters.py).
set connection__param__time dimen 3;  # Migrated to Python (preprocessing/process_arc_unions.py).
set connection__param_t     dimen 2;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/connection__param__time.csv' : connection__param__time <- [connection, param, time];
table data IN 'CSV' 'solve_data/connection__param_t.csv'     : connection__param_t     <- [connection, param];
set process__source__param dimen 3 within {process_source, sourceSinkParam};
set process__source__param__time dimen 4 within {process_source, sourceSinkTimeParam, time};
set process__source__param__period dimen 4 within {process_source, sourceSinkPeriodParam, period};
set process__source__param_t dimen 3;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/process__source__param_t.csv' : process__source__param_t <- [process, source, param];
set process__sink__param dimen 3 within {process_sink, sourceSinkParam};
set process__sink__param__time dimen 4 within {process_sink, sourceSinkTimeParam, time};
set process__sink__param_t dimen 3;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/process__sink__param_t.csv' : process__sink__param_t <- [process, sink, param];
set process__sink__param__period dimen 4 within {process_sink, sourceSinkPeriodParam, period};

set node__param dimen 2 within {node, nodeParam};
set node__param__time dimen 3 within {node, nodeTimeParam, time};
set node__time_inflow dimen 2 within {node, time};

param p_model {modelParam};
param p_nested_model {modelParam};
param p_commodity {c in commodity, commodityParam} default 0;
param pd_commodity {c in commodity, commodityPeriodParam, d in periodAll} default 0;
param pt_commodity {c in commodity, commodityTimeParam, time} default 0;
param p_commodity_price_method {c in commodity} symbolic in price_method, default 'price';
param p_commodity_unitsize {c in commodity} default 1.0;
# Ladder prices and quantities are split per ladder method.  Defaults are
# a very large sentinel treated as "infinite" (see
# ladder_tier_cap_infinite_cum / _ann below and the 1e30 sentinel
# emitted by flextoolrunner/input_writer).
# GMPL's CSV reader rejects literal 'inf' / 'Infinity', so the Python side
# writes 1e30 for user-provided +Infinity values and these defaults match.
param p_ladder_cum_price    {(c, i) in commodity__tier_cum} default 0;
param p_ladder_cum_quantity {(c, i) in commodity__tier_cum} default 1e30;
# Annual ladder parameters carry a period dimension — they are read
# over (commodity, tier, period) triples from commodity_ladder_annual.csv;
# the writer expands 1d-map inputs across all model periods, so the CSV
# always has a period column.
param p_ladder_ann_price    {(c, i) in commodity__tier_ann, d in periodAll} default 0;
param p_ladder_ann_quantity {(c, i) in commodity__tier_ann, d in periodAll} default 1e30;

# Per-period rolling accumulators for price_ladder_{annual,cumulative} tiers.
# Rewritten at the end of each rolling solve by the Python cumulative-handoff
# writer (flextool/process_outputs/cumulative_handoffs.py) as:
#   p_ladder_cum_realized_mwh[c, i, d] : realized sim-MWh of tier i of commodity
#       c that fell into period d across all prior rolls.  Zero on the first
#       solve (header-only seed).
#   p_ladder_cum_sim_hours[d]          : realized sim-hours of period d across
#       all prior rolls.  Zero on the first solve.
# Together with the current roll's horizon these let the LP compute, per
# period, the fraction of that period "filled" so far and cap v_trade on a
# rolling-partition basis (see ladder_tier_cap_annual_roll and
# ladder_tier_cap_cumulative_roll below).
# Indexed over periodAll (not period_in_use) because the prior roll's CSV
# may contain rows for periods that are not in the current roll's solve
# window — the CSV reader rejects out-of-domain rows, and periodAll is
# the superset that covers every period the model knows about.
param p_ladder_cum_realized_mwh {c in commodity, i in tier, d in periodAll} default 0;
param p_ladder_cum_sim_hours {d in periodAll} default 0;

# Per-period rolling accumulator for the model-wide CO2 cap (co2_max_total).
# Stored in tonnes (post-/1000 scaling, matching the mod's RHS convention).
# Written at the end of each rolling solve by
# cumulative_handoffs.write_co2_rolling_accumulators.  Reuses
# p_ladder_cum_sim_hours (same period-level realized-hours accumulator)
# plus the shared f_d_k[d] for the RHS partition.
param p_co2_cum_realized_tonnes {g in group, d in periodAll} default 0;

# Commodity-ladder derived sets (used by v_trade and the ladder constraints).
# 'price' is the default scalar-price behaviour; the two ladder methods route
# into v_trade / tier caps and replace the pdtCommodity price objective term.
# Computed in Python (flextool/flextoolrunner/preprocessing/commodity_ladder_sets.py)
# and loaded via table data IN below — see solve_data/commodity_with_ladder*.csv.
set commodity_with_ladder;
set commodity_with_ladder_annual;
set commodity_with_ladder_cumulative;

param p_node {node, nodeParam} default 0;
param pd_node {node, nodePeriodParam, periodAll} default 0;
param pt_node {node, nodeTimeParam, time} default 0;
param pt_node_inflow {node, time} default 0;

param p_process_source {(p, source) in process_source, sourceSinkParam} default 0;
param pt_process_source {(p, source) in process_source, sourceSinkTimeParam, time} default 0;
param pd_process_source {(p, source) in process_source, sourceSinkPeriodParam, period} default 0;
param p_process_sink {(p, sink) in process_sink, sourceSinkParam} default 0;
param pt_process_sink {(p, sink) in process_sink, sourceSinkTimeParam, time} default 0;
param pd_process_sink {(p, sink) in process_sink, sourceSinkPeriodParam, period} default 0;

param p_process_source_flow_coefficient {(p, source) in process_source} default 1;
param p_process_sink_flow_coefficient {(p, sink) in process_sink} default 1;
param p_process_source_max_capacity_coefficient {(p, source) in process_source} default 1;
param p_process_sink_max_capacity_coefficient {(p, sink) in process_sink} default 1;
param p_process_source_min_capacity_coefficient {(p, source) in process_source} default 1;
param p_process_sink_min_capacity_coefficient {(p, sink) in process_sink} default 1;

param p_profile {profile};
param pt_profile {profile, time};

param reserveParam_defaults{rp in reserveParam}:= (if rp == 'reliability' then 1 else if rp == 'penalty_reserve' then 5000 else 0);
param p_reserve_upDown_group {reserve, upDown, group, rp in reserveParam} default reserveParam_defaults[rp];
param pt_reserve_upDown_group {reserve, upDown, group, reserveTimeParam, time};
param p_process_reserve_upDown_node {p in process, r in reserve, ud in upDown, n in node, rp in reserveParam} default reserveParam_defaults[rp];

param p_process {process, processParam} default 0;
param pd_process {process, processPeriodParam, periodAll} default 0;
param pt_process {process, processTimeParam, time} default 0;
param p_connection_susceptance {p in connection_dc_power_flow};

param p_constraint_constant {constraint} default 0;
param p_process_node_constraint_flow_coefficient {process, node, constraint};
param p_process_constraint_invested_capacity_coefficient {process, constraint};
param p_node_constraint_invested_capacity_coefficient {node, constraint};
param p_process_constraint_prebuilt_capacity_coefficient {process, constraint};
param p_node_constraint_prebuilt_capacity_coefficient {node, constraint};
param p_node_constraint_state_coefficient {node, constraint};
param step_duration{(d, t) in dt};

# Block-aware step durations (Agent 1.2): step_duration per (block, period,
# step).  Indexed on the companion set block__period__step populated alongside
# the parameter from solve_data/block_step_duration.csv — matching the dt/
# step_duration read idiom.
param block_step_duration {(b, d, t) in block__period__step};
# Overlap fraction per (period, block_coarse, step_coarse, block_fine,
# step_fine).  Read from solve_data/overlap_set.csv; rows not present default
# to 0.  In the degenerate single-block solve the identity rows carry 1.0.
param p_overlap {overlap} default 0;
param p_hole_multiplier {solve_current} default 1;
# Agent 5 (LP-scaling): per-solve opt-in for automatic row scaling.
# When the solve's p_use_row_scaling is 1, node_capacity_for_scaling
# and group_capacity_for_scaling below are computed from connected-unit
# unitsizes (power-of-10 rounded) instead of staying at 1.  Default 0
# keeps the pre-Agent-5 behaviour bit-for-bit.
param p_use_row_scaling {solve_current} default 0;

# Representative period parameters
param p_rp_weight{rp_base__rep} default 0;
param p_rp_last_step{rp_rep_period} symbolic;
param p_rp_cost_weight{(d,t) in dt} default 1;

param p_years_represented{d in period, y in year} default 1;
param p_years_from_solve{d in period, y in year} default 0;
param p_discount_years{d in period} default 0;
param p_inflation_rate{model} default 0;
param p_inflation_offset_investment{model} default 0;    # Inflation offset for investment annuity (assumes investments at the start of the year unless other value is given)
param p_inflation_offset_operations{model} default 0.5;  # Inflation offset for operational costs (assumes costs on average at the middle of the year unless other value is given)
param p_max_flow_for_unconstrained_variables{model} default 1000000;  # MW cap on variables with no other upper bound (invest_no_limit, zero-coefficient flows, infinite tiers)

param p_entity_divested {e in entity : e in entityDivest};
set ed_history_realized_read dimen 2 within {e in entity, d in period_with_history};
param p_entity_period_existing_capacity {e in entity, d in period_with_history};
param p_entity_period_invested_capacity {e in entity, d in period_with_history};

####
# Delayed flows
####
set delay_duration dimen 1;
set process_delay_weighted__delay_duration dimen 2 within {process, delay_duration};
set dtt__delay_duration dimen 4 within {period, time, time, delay_duration};
param p_process_delay_weighted {process, delay_duration};
set process_delay_single__delay_duration dimen 2 within {process, delay_duration};

####
#Stochastic sets and params
####
set groupStochastic dimen 1 within {group};
set solve_branch__time_branch dimen 2 within {branch_all, time_branch_all};
param p_branch_weight_input {b in branch} default 1;
#normalize the branches with the same starting time to add up to 1
param pd_branch_weight {d in period_in_use};  # Migrated to Python (preprocessing/period_calculated_params.py).
param pdt_branch_weight {(d,t) in dt};        # Migrated to Python (preprocessing/period_calculated_params.py).
table data IN 'CSV' 'solve_data/pd_branch_weight.csv'  : [period],          pd_branch_weight~value;
table data IN 'CSV' 'solve_data/pdt_branch_weight.csv' : [period, time],    pdt_branch_weight~value;

set dt_non_anticipativity dimen 2;  # Migrated to Python (preprocessing/per_solve_sets.py).

#stochastic versions of timeseries
set process__param__branch__time dimen 5 within {process, processTimeParam, time_branch_all, time, time};
set profile__branch__time dimen 4 within {profile, time_branch_all, time, time};
set process__source__param__branch__time dimen 6 within {process_source, sourceSinkTimeParam, time_branch_all, time, time};
set process__sink__param__branch__time dimen 6 within {process_sink, sourceSinkTimeParam, time_branch_all, time, time};
set node__param__branch__time dimen 5 within {node, nodeTimeParam, time_branch_all, time, time};
set node__branch__time_inflow dimen 4 within {node, time_branch_all, time, time};
set reserve__upDown__group__reserveParam__branch__time dimen 7 within {reserve, upDown, group, reserveTimeParam, time_branch_all, time, time};

param pbt_node {node, nodeTimeParam, time_branch_all, time, time} default 0;
param pbt_node_inflow {node, time_branch_all, time, time} default 0;
param pbt_process_source {(p, source) in process_source, sourceSinkTimeParam, time_branch_all, time, time} default 0;
param pbt_process_sink {(p, sink) in process_sink, sourceSinkTimeParam, time_branch_all, time, time} default 0;
param pbt_profile {profile, time_branch_all, time, time} default 0;
param pbt_reserve_upDown_group {reserve, upDown, group, reserveTimeParam, time_branch_all, time, time} default 0;
param pbt_process {process, processTimeParam, time_branch_all, time, time} default 0;

# Agent 12 (LP-scaling): scale_the_objective / scale_the_state are
# Python-driven per solve (flextoolrunner/solve_writers.py writes
# solve_data/scale_the_objective.csv and solve_data/scale_the_state.csv
# from the Agent-8 ScaleTable on every solve).  The CSVs use the simple
# layout ``key,value`` with a single data row; the key is arbitrary.
# The ``default`` clauses below are the legacy fallbacks that were
# previously hardcoded in flextool_base.dat — they apply only when the
# CSV is empty / absent (e.g. AMPL invoked outside the Python harness).
set _scale_obj_keys dimen 1 default {};
param _scale_obj_from_csv {_scale_obj_keys} default 0;
param scale_the_objective := (if card(_scale_obj_keys) > 0
    then sum {k in _scale_obj_keys} _scale_obj_from_csv[k]
    else 1e-6);
# scale_the_state was an unused twin of scale_the_objective —
# declared, written by Python in solve_data/scale_the_state.csv,
# but no constraint or other expression ever read its value.
# Removed in batch 69 along with its _scale_state_* helper sets.

set param_costs dimen 1;
param costs_discounted {param_costs} default 0;
set param_co2 dimen 1;
param model_co2 {param_co2} default 0;

set class_paramName_default dimen 2;
param default_value {class_paramName_default};
set phase;

#########################
# Read data
#table data IN 'CSV' '.csv' :  <- [];
table data IN 'CSV' 'solve_data/glpsol_phase.csv' : phase <- [phase];

# Domain sets
table data IN 'CSV' 'input/commodity.csv' : commodity <- [commodity];
table data IN 'CSV' 'input/constraint__sense.csv' : constraint <- [constraint];
table data IN 'CSV' 'input/debug.csv': debug <- [debug];
table data IN 'CSV' 'input/entity.csv': entity <- [entity];
table data IN 'CSV' 'input/group.csv' : group <- [group];
table data IN 'CSV' 'input/node.csv' : node <- [node];
table data IN 'CSV' 'input/p_node_type.csv' : [node], p_node_type;
table data IN 'CSV' 'input/groupInertia.csv' : groupInertia <- [groupInertia];
table data IN 'CSV' 'input/groupNonSync.csv' : groupNonSync <- [groupNonSync];
table data IN 'CSV' 'input/groupCapacityMargin.csv' : groupCapacityMargin <- [groupCapacityMargin];
table data IN 'CSV' 'input/nodeGroupIndicators.csv' : nodeGroupIndicators <- [nodeGroupIndicators];
table data IN 'CSV' 'input/flowGroupIndicators.csv' : flowGroupIndicators <- [flowGroupIndicators];
table data IN 'CSV' 'input/nodeGroupDispatch.csv' : nodeGroupDispatch <- [nodeGroupDispatch];
table data IN 'CSV' 'input/flowAggregator.csv' : flowAggregator <- [flowAggregator];
table data IN 'CSV' 'input/process.csv': process <- [process];
table data IN 'CSV' 'input/profile.csv': profile <- [profile];
table data IN 'CSV' 'input/optional_outputs.csv': optional_outputs <- [output, value];
table data IN 'CSV' 'input/exclude_entity_outputs.csv': exclude_entity_outputs <- [value];
table data IN 'CSV' 'input/groupIncludeStochastics.csv' : groupStochastic <- [group];
table data IN 'CSV' 'input/periods_available.csv': period_from_model <- [period_from_model];

# Single dimension membership sets
table data IN 'CSV' 'input/process_connection.csv': process_connection <- [process_connection];
table data IN 'CSV' 'input/process_nonSync_connection.csv': process_nonSync_connection <- [process];
table data IN 'CSV' 'input/node_dc_power_flow.csv' : node_dc_power_flow <- [node];
table data IN 'CSV' 'input/connection_dc_power_flow.csv' : connection_dc_power_flow <- [process];
table data IN 'CSV' 'input/node_reference_angle.csv' : node_reference_angle <- [node];
table data IN 'CSV' 'input/p_connection_susceptance.csv' : [process], p_connection_susceptance;
table data IN 'CSV' 'input/process_unit.csv': process_unit <- [process_unit];

# Multi dimension membership sets
table data IN 'CSV' 'input/commodity__node.csv' : commodity_node <- [commodity,node];
table data IN 'CSV' 'input/entity__invest_method.csv' : entity__invest_method <- [entity,invest_method];
table data IN 'CSV' 'input/group__invest_method.csv' : group__invest_method <- [group,invest_method];
table data IN 'CSV' 'input/entity__lifetime_method.csv' : entity__lifetime_method_read <- [entity,lifetime_method];
table data IN 'CSV' 'input/group__co2_method.csv' : group__co2_method <- [group,co2_method];
table data IN 'CSV' 'input/group__loss_share_type.csv' : group__loss_share_type <- [group,loss_share_type];
table data IN 'CSV' 'input/node__inflow_method.csv' : node__inflow_method_read <- [node,inflow_method];
table data IN 'CSV' 'input/node__storage_binding_method.csv' : node__storage_binding_method_read <- [node,storage_binding_method];
table data IN 'CSV' 'input/node__storage_start_end_method.csv' : node__storage_start_end_method <- [node,storage_start_end_method];
table data IN 'CSV' 'input/node__storage_solve_horizon_method.csv' : node__storage_solve_horizon_method <- [node,storage_solve_horizon_method];
table data IN 'CSV' 'input/node__storage_nested_fix_method.csv' : node__storage_nested_fix_method <- [node,storage_nested_fix_method];
table data IN 'CSV' 'input/node__profile__profile_method.csv' : node__profile__profile_method <- [node,profile,profile_method];
table data IN 'CSV' 'input/group__node.csv' : group_node <- [group,node];
table data IN 'CSV' 'input/group__process.csv' : group_process <- [group,process];
table data IN 'CSV' 'input/group__process__node.csv' : group_process_node <- [group,process,node];
table data IN 'CSV' 'input/p_process_node_constraint_flow_coefficient.csv' : process_node_flow_constraint <- [process, node, constraint];
table data IN 'CSV' 'input/p_process_constraint_invested_capacity_coefficient.csv' : process_capacity_constraint_invested <- [process, constraint];
table data IN 'CSV' 'input/p_node_constraint_invested_capacity_coefficient.csv' : node_capacity_constraint_invested <- [node, constraint];
table data IN 'CSV' 'input/p_process_constraint_cumulative_pre_built_capacity_coefficient.csv' : process_capacity_constraint_prebuilt <- [process, constraint];
table data IN 'CSV' 'input/p_node_constraint_cumulative_pre_built_capacity_coefficient.csv' : node_capacity_constraint_prebuilt <- [node, constraint];
table data IN 'CSV' 'input/p_node_constraint_state_coefficient.csv' : node_state_constraint <- [node, constraint];
table data IN 'CSV' 'input/p_process_delay_weighted.csv' : process_delay_weighted__delay_duration <- [process, delay_duration];
table data IN 'CSV' 'input/constraint__sense.csv' : constraint__sense <- [constraint, sense];
table data IN 'CSV' 'input/p_process.csv' : process__param <- [process, processParam];
table data IN 'CSV' 'input/p_profile.csv' : profile_param <- [profile];
table data IN 'CSV' 'input/p_node.csv' : node__param <- [node, nodeParam];
table data IN 'CSV' 'input/pd_node.csv' : node__param__period <- [node, nodeParam, period];
table data IN 'CSV' 'input/pd_process.csv' : process__param__period <- [process, processParam, period];
table data IN 'CSV' 'input/p_group.csv' : group__param <- [group, groupParam];
table data IN 'CSV' 'input/pd_group.csv' : group__param__period <- [group, groupParam, period];
table data IN 'CSV' 'input/process__ct_method.csv' : process__ct_method_read <- [process,ct_method];
table data IN 'CSV' 'input/process__node__ramp_method.csv' : process_node_ramp_method <- [process,node,ramp_method];
table data IN 'CSV' 'input/process__reserve__upDown__node.csv' : process_reserve_upDown_node <- [process,reserve,upDown,node];
table data IN 'CSV' 'input/process__sink.csv' : process_sink <- [process,sink];
table data IN 'CSV' 'input/process__source.csv' : process_source <- [process,source];
table data IN 'CSV' 'input/process__sink_nonSync_unit.csv' : process__sink_nonSync_unit <- [process,sink];
table data IN 'CSV' 'input/process__startup_method.csv' : process__startup_method_read <- [process,startup_method];

set process_min_uptime 'processes with minimum uptime constraint' within process;
table data IN 'CSV' 'input/process_min_uptime.csv' : process_min_uptime <- [process_min_uptime];

set process_min_downtime 'processes with minimum downtime constraint' within process;
table data IN 'CSV' 'input/process_min_downtime.csv' : process_min_downtime <- [process_min_downtime];

table data IN 'CSV' 'input/process__profile__profile_method.csv' : process__profile__profile_method <- [process,profile,profile_method];
table data IN 'CSV' 'input/process__node__profile__profile_method.csv' : process__node__profile__profile_method <- [process,node,profile,profile_method];
table data IN 'CSV' 'input/reserve__upDown__group__method.csv' : reserve__upDown__group__method <- [reserve,upDown,group,method];
table data IN 'CSV' 'input/timesets_in_use.csv' : solve_period_timeset <- [solve,period,timesets];
table data IN 'CSV' 'input/timesets__timeline.csv' : timeset__timeline <- [timesets,timeline];
table data IN 'CSV' 'solve_data/solve_current.csv' : solve_current <- [solve];
table data IN 'CSV' 'input/p_process_source.csv' : process__source__param <- [process, source, sourceSinkParam];
table data IN 'CSV' 'input/p_process_sink.csv' : process__sink__param <- [process, sink, sourceSinkParam];
table data IN 'CSV' 'input/pd_commodity.csv' : commodity__param__period <- [commodity, commodityParam, period];
table data IN 'CSV' 'input/timeline.csv' : timeline__timestep__duration <- [timeline,timestep,duration];
table data IN 'CSV' 'solve_data/delay_duration.csv' : delay_duration <- [delay_duration];
table data IN 'CSV' 'solve_data/dtt__delay_duration.csv' : dtt__delay_duration <- [period,time_source,time_sink,delay_duration];
table data IN 'CSV' 'input/process_delay_single.csv' : process_delay_single__delay_duration <- [process,delay_duration];

# Parameters for model data.
table data IN 'CSV' 'input/p_commodity.csv' : [commodity, commodityParam], p_commodity;
table data IN 'CSV' 'input/pd_commodity.csv' : [commodity, commodityParam, period], pd_commodity;
table data IN 'CSV' 'input/p_commodity_price_method.csv' : [commodity], p_commodity_price_method;
table data IN 'CSV' 'input/p_commodity_unitsize.csv' : [commodity], p_commodity_unitsize;
table data IN 'CSV' 'input/commodity_ladder_cumulative.csv' : commodity__tier_cum <- [commodity, tier], p_ladder_cum_price~price, p_ladder_cum_quantity~quantity;
# Annual ladder: the CSV has one row per (commodity, tier, period); read
# it into the raw triples set and the per-period params together.  The
# derived commodity__tier_ann set projects out the period dimension via
# `setof` (see above), avoiding duplicate-tuple errors.
table data IN 'CSV' 'input/commodity_ladder_annual.csv' : commodity__tier__period_ann <- [commodity, tier, period], p_ladder_ann_price~price, p_ladder_ann_quantity~quantity;
table data IN 'CSV' 'input/p_group__process.csv' : [group, process, groupParam], p_group__process;
table data IN 'CSV' 'input/p_group.csv' : [group, groupParam], p_group;
table data IN 'CSV' 'input/pd_group.csv' : [group, groupParam, period], pd_group;
table data IN 'CSV' 'input/p_node.csv' : [node, nodeParam], p_node;
table data IN 'CSV' 'input/pd_node.csv' : [node, nodeParam, period], pd_node;
table data IN 'CSV' 'input/p_process_node_constraint_flow_coefficient.csv' : [process, node, constraint], p_process_node_constraint_flow_coefficient;
table data IN 'CSV' 'input/p_process_constraint_invested_capacity_coefficient.csv' : [process, constraint], p_process_constraint_invested_capacity_coefficient;
table data IN 'CSV' 'input/p_node_constraint_invested_capacity_coefficient.csv' : [node, constraint], p_node_constraint_invested_capacity_coefficient;
table data IN 'CSV' 'input/p_process_constraint_cumulative_pre_built_capacity_coefficient.csv' : [process, constraint], p_process_constraint_prebuilt_capacity_coefficient;
table data IN 'CSV' 'input/p_node_constraint_cumulative_pre_built_capacity_coefficient.csv' : [node, constraint], p_node_constraint_prebuilt_capacity_coefficient;
table data IN 'CSV' 'input/p_node_constraint_state_coefficient.csv' : [node, constraint], p_node_constraint_state_coefficient;
table data IN 'CSV' 'input/p_process__reserve__upDown__node.csv' : [process, reserve, upDown, node, reserveParam], p_process_reserve_upDown_node;
table data IN 'CSV' 'input/p_process_sink.csv' : [process, sink, sourceSinkParam], p_process_sink;
table data IN 'CSV' 'input/p_process_sink_flow_coefficient.csv' : [process, sink], p_process_sink_flow_coefficient;
table data IN 'CSV' 'input/p_process_sink_max_capacity_coefficient.csv' : [process, sink], p_process_sink_max_capacity_coefficient;
table data IN 'CSV' 'input/p_process_sink_min_capacity_coefficient.csv' : [process, sink], p_process_sink_min_capacity_coefficient;
table data IN 'CSV' 'input/p_process_source.csv' : [process, source, sourceSinkParam], p_process_source;
table data IN 'CSV' 'input/p_process_source_flow_coefficient.csv' : [process, source], p_process_source_flow_coefficient;
table data IN 'CSV' 'input/p_process_source_max_capacity_coefficient.csv' : [process, source], p_process_source_max_capacity_coefficient;
table data IN 'CSV' 'input/p_process_source_min_capacity_coefficient.csv' : [process, source], p_process_source_min_capacity_coefficient;
table data IN 'CSV' 'input/p_process_delay_weighted.csv' : [process, delay_duration], p_process_delay_weighted;
table data IN 'CSV' 'input/p_constraint_constant.csv' : [constraint], p_constraint_constant;
table data IN 'CSV' 'input/p_process.csv' : [process, processParam], p_process;
table data IN 'CSV' 'input/pd_process.csv' : [process, processParam, period], pd_process;
table data IN 'CSV' 'input/pd_process_source.csv' : process__source__param__period <- [process, source, sourceSinkPeriodParam, period], pd_process_source~pd_process_source;
table data IN 'CSV' 'input/pd_process_sink.csv' : process__sink__param__period <- [process, sink, sourceSinkPeriodParam, period], pd_process_sink~pd_process_sink;
table data IN 'CSV' 'input/p_profile.csv' : [profile], p_profile;
table data IN 'CSV' 'input/p_reserve__upDown__group.csv' : [reserve, upDown, group, reserveParam], p_reserve_upDown_group;
table data IN 'CSV' 'input/timeline.csv' : [timeline,timestep], p_timeline_step_duration~duration;
table data IN 'CSV' 'solve_data/p_discount_years.csv' : [period], p_discount_years~param;
table data IN 'CSV' 'solve_data/p_years_represented.csv' : period__year <- [period,years_from_solve], p_years_represented~p_years_represented, p_years_from_solve~p_years_from_solve;
table data IN 'CSV' 'input/p_inflation_rate.csv' : model <- [model];
table data IN 'CSV' 'input/p_inflation_rate.csv' : [model], p_inflation_rate;
table data IN 'CSV' 'input/p_inflation_offset_operations.csv' : [model], p_inflation_offset_operations;
table data IN 'CSV' 'input/p_inflation_offset_investment.csv' : [model], p_inflation_offset_investment;
table data IN 'CSV' 'input/p_max_flow_for_unconstrained_variables.csv' : [model], p_max_flow_for_unconstrained_variables;
table data IN 'CSV' 'input/default_values.csv' : class_paramName_default <-[class, paramName];
table data IN 'CSV' 'input/default_values.csv' : [class,paramName], default_value;

#Timeseries parameters, Timestep values are in solve_data as they might be averaged for the solve
table data IN 'CSV' 'solve_data/pt_commodity.csv' : commodity__param__time <- [commodity, commodityParam, time], pt_commodity~pt_commodity;
table data IN 'CSV' 'solve_data/pt_group.csv' : group__param__time <- [group, groupParam, time], pt_group~pt_group;
table data IN 'CSV' 'solve_data/pt_node.csv' : node__param__time <- [node, nodeParam, time], pt_node~pt_node;
table data IN 'CSV' 'solve_data/pt_node_inflow.csv' : node__time_inflow <- [node, time], pt_node_inflow~pt_node_inflow;
table data IN 'CSV' 'solve_data/pt_process.csv' : process__param__time <- [process, processParam, time], pt_process~pt_process;
table data IN 'CSV' 'solve_data/pt_profile.csv' : profile_param__time <- [profile, time], pt_profile~pt_profile;
table data IN 'CSV' 'solve_data/pt_reserve__upDown__group.csv' : reserve__upDown__group__reserveParam__time <- [reserve, upDown, group, reserveParam, time], pt_reserve_upDown_group~pt_reserve_upDown_group;
table data IN 'CSV' 'solve_data/pt_process_source.csv' : process__source__param__time <- [process, source, sourceSinkTimeParam, time], pt_process_source~pt_process_source;
table data IN 'CSV' 'solve_data/pt_process_sink.csv' : process__sink__param__time <- [process, sink, sourceSinkTimeParam, time], pt_process_sink~pt_process_sink;

# Parameters from the solve loop
table data IN 'CSV' 'solve_data/solve_hole_multiplier.csv' : [solve], p_hole_multiplier;
table data IN 'CSV' 'solve_data/p_use_row_scaling.csv' : [solve], p_use_row_scaling;
# Agent 12 (LP-scaling): Python-driven scale_the_objective / scale_the_state.
# These CSVs are emitted by flextoolrunner/solve_writers.py on every solve
# from the Agent-8 ScaleTable analyser.  The default 1e-6 / 1 clauses on
# the param declarations apply only if the CSVs are empty or absent.
table data IN 'CSV' 'solve_data/scale_the_objective.csv' : _scale_obj_keys <- [key], _scale_obj_from_csv~value;
table data IN 'CSV' 'solve_data/steps_in_use.csv' : dt <- [period, step], step_duration~step_duration;
table data IN 'CSV' 'solve_data/steps_in_timeline.csv' : period_time <- [period,step];
table data IN 'CSV' 'solve_data/first_timesteps.csv' : period__time_first <- [period,step];
table data IN 'CSV' 'solve_data/last_timesteps.csv' : period__time_last <- [period,step];
table data IN 'CSV' 'solve_data/last_realized_timestep.csv' : realized_period__time_last <- [period,step];
table data IN 'CSV' 'solve_data/step_previous.csv' : dtttdt <- [period, time, previous, previous_within_timeset, previous_period, previous_within_solve];
table data IN 'CSV' 'solve_data/step_previous.csv' : [period, time], dt_jump~jump;
table data IN 'CSV' 'solve_data/period_with_history.csv' : period_with_history <- [period], p_period_from_solve~param;
table data IN 'CSV' 'solve_data/realized_invest_periods_of_current_solve.csv' : d_realize_invest <- [period];
table data IN 'CSV' 'solve_data/invest_periods_of_current_solve.csv' : period_invest <- [period];
table data IN 'CSV' 'solve_data/p_model.csv' : [modelParam], p_model;

# Clear some files in the first solve
if p_model["solveFirst"] == 1 and 'read' in phase then {
  printf "entity,p_entity_invested\n" > "solve_data/p_entity_invested.csv";
  printf "entity,p_entity_divested\n" > "solve_data/p_entity_divested.csv";
  printf "entity,period,p_entity_period_existing_capacity,p_entity_period_invested_capacity\n" > "solve_data/p_entity_period_existing_capacity.csv";
  printf "period,step,node,p_fix_storage_price\n" > "solve_data/fix_storage_price.csv";
  printf "period,step,node,p_fix_storage_quantity\n" > "solve_data/fix_storage_quantity.csv";
  printf "period,step,node,p_fix_storage_usage\n" > "solve_data/fix_storage_usage.csv";
  printf "node, p_roll_continue_state\n" > "solve_data/p_roll_continue_state.csv";
  printf 'param_costs,costs_discounted\n' > "solve_data/costs_discounted.csv";
  printf 'param_co2\n' > "solve_data/co2.csv";
  printf 'period\n' > 'solve_data/period_capacity.csv';
  printf '' >> "solve_data/costs_discounted.csv";  # To close the previous file and update the changes
}

# Further parameters from the solve loop
table data IN 'CSV' 'solve_data/p_nested_model.csv' : [modelParam], p_nested_model;
table data IN 'CSV' 'solve_data/realized_dispatch.csv' : dt_realize_dispatch_input <- [period, step];
table data IN 'CSV' 'solve_data/fix_storage_timesteps.csv' : dt_fix_storage_timesteps <- [period, step];
table data IN 'CSV' 'solve_data/fix_storage_price.csv' : ndt_fix_storage_price <- [node, period, step], p_fix_storage_price~p_fix_storage_price;
table data IN 'CSV' 'solve_data/fix_storage_quantity.csv' : ndt_fix_storage_quantity <- [node, period, step], p_fix_storage_quantity~p_fix_storage_quantity;
table data IN 'CSV' 'solve_data/fix_storage_usage.csv' : ndt_fix_storage_usage <- [node, period, step], p_fix_storage_usage~p_fix_storage_usage;
table data IN 'CSV' 'solve_data/timeline_matching_map.csv' : dtt_timeline_matching <- [period, step, upper_step];

# Representative period data (empty files when not using RP weights)
table data IN 'CSV' 'solve_data/rp_weights.csv' : rp_base__rep <- [base_start, rep_start], p_rp_weight~weight;
table data IN 'CSV' 'solve_data/rp_base_chain.csv' : rp_base_chain <- [base_start, prev_base_start];
table data IN 'CSV' 'solve_data/rp_base_first.csv' : rp_base_first <- [base_start];
table data IN 'CSV' 'solve_data/rp_base_last.csv' : rp_base_last <- [base_start];
table data IN 'CSV' 'solve_data/rp_block_first.csv' : rp_block_first <- [period, step];
table data IN 'CSV' 'solve_data/rp_block_last.csv' : rp_block_last <- [period, step];
table data IN 'CSV' 'solve_data/rp_block_start_last.csv' : [rep_start], p_rp_last_step~last_step;
table data IN 'CSV' 'solve_data/rp_cost_weight.csv' : [period, time], p_rp_cost_weight~weight;
table data IN 'CSV' 'solve_data/period_block_time.csv' : period_block_time <- [period, block_first, step];
table data IN 'CSV' 'solve_data/period_block_succ.csv' : period_block_succ <- [period, block_first, block_first_next];
# Temporal-resolution block CSVs (Agent 1.1 writes; Agent 1.2 reads inert).
# Consumed by generalized balance / flow / storage constraints in Agents 1.3+.
table data IN 'CSV' 'solve_data/entity_block.csv' :
    node__block <- [entity, block];
table data IN 'CSV' 'solve_data/process_side_block.csv' :
    process__side__block <- [process, side, block];
# Agent 1.6: per-process unified block for UC / ramp / profile.
table data IN 'CSV' 'solve_data/process_block.csv' :
    process__block <- [process, block];
table data IN 'CSV' 'solve_data/block_step_duration.csv' :
    block__period__step <- [block, period, step], block_step_duration ~ step_duration;
table data IN 'CSV' 'solve_data/overlap_set.csv' :
    overlap <- [period, block_coarse, step_coarse, block_fine, step_fine],
    p_overlap ~ fraction;
# Per-block predecessor + boundary tables (Agent 1.4).  Consumers:
# nodeBalance_eq state-transition terms (block_dtttdt), and the seven
# storage constraints indexed at the node's block's first / last step.
table data IN 'CSV' 'solve_data/block_step_previous.csv' :
    block_dtttdt <- [block, period, step, step_previous,
        step_previous_within_timeset, period_previous,
        step_previous_within_solve];
table data IN 'CSV' 'solve_data/block_period_time_first.csv' :
    block__period__time_first <- [block, period, step];
table data IN 'CSV' 'solve_data/block_period_time_last.csv' :
    block__period__time_last <- [block, period, step];
table data IN 'CSV' 'solve_data/steps_complete_solve.csv' : dt_complete <- [period, step];
table data IN 'CSV' 'solve_data/steps_complete_solve.csv' : [period, step], complete_step_duration;
table data IN 'CSV' 'solve_data/p_roll_continue_state.csv' : [node], p_roll_continue_state;
# Rolling per-period accumulators for price_ladder_* tiers.  Written at
# the end of each roll by cumulative_handoffs.write_ladder_rolling_accumulators.
# Header-only CSVs on the first solve (seeded by Python) → zero rows loaded
# → both params default to 0 → the caps collapse to their single-solve form
# (f_d_k[d] = horizon_hours_d / (share_of_year[d] * 8760) = 1.0 for a full
# single solve, and cum_realized_mwh = 0).
table data IN 'CSV' 'solve_data/ladder_cum_realized_mwh.csv'
    : [commodity, tier, period], p_ladder_cum_realized_mwh;
table data IN 'CSV' 'solve_data/ladder_cum_sim_hours.csv'
    : [period], p_ladder_cum_sim_hours;
# Rolling per-period accumulator for co2_max_total.  Same mechanism as the
# ladder accumulators above — header-only seed on first solve, rewritten
# per roll by cumulative_handoffs.write_co2_rolling_accumulators.
table data IN 'CSV' 'solve_data/co2_cum_realized_tonnes.csv'
    : [group, period], p_co2_cum_realized_tonnes;
table data IN 'CSV' 'solve_data/branch_all.csv' : branch_all <- [branch];
table data IN 'CSV' 'solve_data/time_branch_all.csv' : time_branch_all <- [time_branch];
table data IN 'CSV' 'solve_data/period__branch.csv' : period__branch <- [period, branch];
table data IN 'CSV' 'solve_data/solve_branch_weight.csv' : [branch], p_branch_weight_input;
table data IN 'CSV' 'solve_data/solve_branch__time_branch.csv' : solve_branch__time_branch <- [period, branch];
table data IN 'CSV' 'solve_data/period_first.csv' : period_first <- [period];
table data IN 'CSV' 'solve_data/period_last.csv' : period_last <- [period];
table data IN 'CSV' 'solve_data/period_first_of_solve.csv' : period_first_of_solve <- [period];

set uptime_lookback dimen 5;
table data IN 'CSV' 'solve_data/uptime_lookback.csv' :
    uptime_lookback <- [process, period, time, period_back, time_back];

set downtime_lookback dimen 5;
table data IN 'CSV' 'solve_data/downtime_lookback.csv' :
    downtime_lookback <- [process, period, time, period_back, time_back];

# Stochastic input data
table data IN 'CSV' 'solve_data/pbt_node.csv' : node__param__branch__time <- [node, nodeParam, branch, time_start, time], pbt_node~pbt_node;
table data IN 'CSV' 'solve_data/pbt_node_inflow.csv' : node__branch__time_inflow <- [node, branch, time_start, time], pbt_node_inflow~pbt_node_inflow;
table data IN 'CSV' 'solve_data/pbt_process_sink.csv' : process__sink__param__branch__time <- [process, sink, sourceSinkTimeParam, branch, time_start, time], pbt_process_sink~pbt_process_sink;
table data IN 'CSV' 'solve_data/pbt_process_source.csv' : process__source__param__branch__time <-  [process, source, sourceSinkTimeParam, branch, time_start, time], pbt_process_source~pbt_process_source;
table data IN 'CSV' 'solve_data/pbt_process.csv' : process__param__branch__time <- [process, processParam, branch, time_start, time], pbt_process~pbt_process;
table data IN 'CSV' 'solve_data/pbt_profile.csv' : profile__branch__time <- [profile, branch, time_start, time], pbt_profile~pbt_profile;
table data IN 'CSV' 'solve_data/pbt_reserve__upDown__group.csv' : reserve__upDown__group__reserveParam__branch__time <- [reserve, upDown, group, reserveParam, branch, time_start, time], pbt_reserve_upDown_group~pbt_reserve_upDown_group;

# After rolling forward the investment model
table data IN 'CSV' 'solve_data/p_entity_divested.csv' : [entity], p_entity_divested;
table data IN 'CSV' 'solve_data/p_entity_period_existing_capacity.csv' : ed_history_realized_read <- [entity, period];
table data IN 'CSV' 'solve_data/p_entity_period_existing_capacity.csv' : [entity, period], p_entity_period_existing_capacity;
table data IN 'CSV' 'solve_data/p_entity_period_existing_capacity.csv' : [entity, period], p_entity_period_invested_capacity;

# Reading results from previous solves
table data IN 'CSV' 'solve_data/costs_discounted.csv' : [param_costs], costs_discounted;
table data IN 'CSV' 'solve_data/co2.csv' : [param_co2], model_co2~model_wide;
table data IN 'CSV' 'solve_data/period_capacity.csv' : period_capacity <- [period];

# Migrated derived sets (Python preprocessing — see
# flextool/flextoolrunner/preprocessing/<module>.py for each family).
table data IN 'CSV' 'solve_data/commodity_with_ladder.csv' : commodity_with_ladder <- [commodity];
table data IN 'CSV' 'solve_data/commodity_with_ladder_annual.csv' : commodity_with_ladder_annual <- [commodity];
table data IN 'CSV' 'solve_data/commodity_with_ladder_cumulative.csv' : commodity_with_ladder_cumulative <- [commodity];
# L0 batch 1 — period_param_sets, invest_method_sets, co2_method_sets, simple_projections
table data IN 'CSV' 'solve_data/period_group.csv' : period_group <- [period];
table data IN 'CSV' 'solve_data/period_node.csv' : period_node <- [period];
table data IN 'CSV' 'solve_data/period_commodity.csv' : period_commodity <- [period];
table data IN 'CSV' 'solve_data/period_process.csv' : period_process <- [period];
table data IN 'CSV' 'solve_data/entityInvest.csv' : entityInvest <- [entity];
table data IN 'CSV' 'solve_data/entityDivest.csv' : entityDivest <- [entity];
table data IN 'CSV' 'solve_data/group_invest.csv' : group_invest <- [group];
table data IN 'CSV' 'solve_data/group_divest.csv' : group_divest <- [group];
table data IN 'CSV' 'solve_data/group_co2_price.csv' : group_co2_price <- [group];
table data IN 'CSV' 'solve_data/group_co2_max_period.csv' : group_co2_max_period <- [group];
table data IN 'CSV' 'solve_data/group_co2_max_total.csv' : group_co2_max_total <- [group];
table data IN 'CSV' 'solve_data/optional_yes.csv' : optional_yes <- [output];
table data IN 'CSV' 'solve_data/reserve__upDown__group.csv' : reserve__upDown__group <- [reserve, upDown, group];
table data IN 'CSV' 'solve_data/group_loss_share.csv' : group_loss_share <- [group];
# L0 batch 2 — node_type_sets, method_with_fallback_sets, nonsync_sets
table data IN 'CSV' 'solve_data/nodeCommodity.csv' : nodeCommodity <- [node];
table data IN 'CSV' 'solve_data/nodeBalance.csv' : nodeBalance <- [node];
table data IN 'CSV' 'solve_data/nodeState.csv' : nodeState <- [node];
table data IN 'CSV' 'solve_data/nodeBalancePeriod.csv' : nodeBalancePeriod <- [node];
table data IN 'CSV' 'solve_data/entity__lifetime_method.csv' : entity__lifetime_method <- [entity, lifetime_method];
table data IN 'CSV' 'solve_data/process__ct_method.csv' : process__ct_method <- [process, ct_method];
table data IN 'CSV' 'solve_data/process__startup_method.csv' : process__startup_method <- [process, startup_method];
table data IN 'CSV' 'solve_data/process__group_inside_group_nonSync.csv' : process__group_inside_group_nonSync <- [process, group];
# L0 batch 3 — union_sets, entity_total_caps. group_entity is declared at L287
# so its reader belongs here; process_delayed__duration and e_*_total are
# declared later (L950, L1825+) so their readers live alongside the declarations.
table data IN 'CSV' 'solve_data/group_entity.csv' : group_entity <- [group, entity];
# L0 batch 4 — readers for sets whose declarations precede this block.
table data IN 'CSV' 'solve_data/def_optional_yes.csv' : def_optional_yes <- [output];
table data IN 'CSV' 'solve_data/reserve.csv' : reserve <- [reserve];
table data IN 'CSV' 'solve_data/process_side.csv' : process_side <- [side];
table data IN 'CSV' 'solve_data/nodeGroupDispatch_node.csv' : nodeGroupDispatch_node <- [group];
table data IN 'CSV' 'solve_data/commodity__tier_ann.csv' : commodity__tier_ann <- [commodity, tier];
table data IN 'CSV' 'solve_data/connection__param.csv' : connection__param <- [process, processParam];
table data IN 'CSV' 'solve_data/solve_period.csv' : solve_period <- [solve, period];
table data IN 'CSV' 'solve_data/timeline.csv' : timeline <- [timeline];
table data IN 'CSV' 'solve_data/timeline_steps.csv' : timeline_steps <- [timeline, step];
# L0 batch 5 — node-method fallbacks (declarations precede this block).
table data IN 'CSV' 'solve_data/node__inflow_method.csv' : node__inflow_method <- [node, inflow_method];
table data IN 'CSV' 'solve_data/node__storage_binding_method.csv' : node__storage_binding_method <- [node, storage_binding_method];
# L0 batch 6 — final write_input-scope sets.
table data IN 'CSV' 'solve_data/period_solve.csv' : period_solve <- [period];
table data IN 'CSV' 'solve_data/time.csv' : time <- [time];
table data IN 'CSV' 'solve_data/enable_optional_outputs.csv' : enable_optional_outputs <- [output];
table data IN 'CSV' 'solve_data/nodeState_rp.csv' : nodeState_rp <- [node];
table data IN 'CSV' 'solve_data/nodeStateBlock.csv' : nodeStateBlock <- [node];
table data IN 'CSV' 'solve_data/commodity__tier.csv' : commodity__tier <- [commodity, tier];
table data IN 'CSV' 'solve_data/tier.csv' : tier <- [tier];
# Per-solve preprocessing (preprocessing/per_solve_sets.py via the
# orchestration hook). Filenames use a `_set` suffix to avoid clashing
# with mod's `if p_model['solveFirst']` printf-to-CSV blocks for the
# same-named files (which write a different schema for output use).
table data IN 'CSV' 'solve_data/branch_set.csv' : branch <- [branch];
table data IN 'CSV' 'solve_data/year_set.csv' : year <- [year];
table data IN 'CSV' 'solve_data/period_from_period_time_set.csv' : period_from_period_time <- [period];
table data IN 'CSV' 'solve_data/period_in_use_set.csv' : period_in_use <- [period];
table data IN 'CSV' 'solve_data/time_in_use_set.csv' : time_in_use <- [time];
table data IN 'CSV' 'solve_data/complete_time_in_use_set.csv' : complete_time_in_use <- [time];
table data IN 'CSV' 'solve_data/rp_base_period_set.csv' : rp_base_period <- [period];
table data IN 'CSV' 'solve_data/rp_rep_period_set.csv' : rp_rep_period <- [period];
table data IN 'CSV' 'solve_data/period_block_set.csv' : period_block <- [period, block_first];
table data IN 'CSV' 'solve_data/dtt_set.csv' : dtt <- [period, time, time_previous];
table data IN 'CSV' 'solve_data/d_fix_storage_period_set.csv' : d_fix_storage_period <- [period];
table data IN 'CSV' 'solve_data/n_fix_storage_quantity_set.csv' : n_fix_storage_quantity <- [node];
table data IN 'CSV' 'solve_data/n_fix_storage_price_set.csv' : n_fix_storage_price <- [node];
table data IN 'CSV' 'solve_data/n_fix_storage_usage_set.csv' : n_fix_storage_usage <- [node];
# L0 batch 8 — additional per-solve sets (block, periodAll, conditional/union variants).
table data IN 'CSV' 'solve_data/period_set.csv' : period <- [period];
table data IN 'CSV' 'solve_data/period__timeline_set.csv' : period__timeline <- [period, timeline];
table data IN 'CSV' 'solve_data/periodAll_set.csv' : periodAll <- [period];
table data IN 'CSV' 'solve_data/block_set.csv' : block <- [block];
table data IN 'CSV' 'solve_data/dt_realize_dispatch_set.csv' : dt_realize_dispatch <- [period, time];
table data IN 'CSV' 'solve_data/d_realized_period_set.csv' : d_realized_period <- [period];
table data IN 'CSV' 'solve_data/d_realize_dispatch_or_invest_set.csv' : d_realize_dispatch_or_invest <- [period];
table data IN 'CSV' 'solve_data/dt_non_anticipativity_set.csv' : dt_non_anticipativity <- [period, time];
# pdt_uptime, pdt_downtime, dtdt_next are declared at L1124+ — readers
# co-located with the declarations below to avoid forward-reference.

#check
set ed_history_realized_first dimen 2;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/ed_history_realized_first.csv' : ed_history_realized_first <- [entity, period];
set ed_history_realized dimen 2;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/ed_history_realized.csv' : ed_history_realized <- [entity, period];

set process_delayed__duration dimen 2;  # Migrated to Python (preprocessing/union_sets.py).
table data IN 'CSV' 'solve_data/process_delayed__duration.csv' : process_delayed__duration <- [process, delay_duration];
set process_delayed;  # Migrated to Python (preprocessing/simple_projections.py).
table data IN 'CSV' 'solve_data/process_delayed.csv' : process_delayed <- [process];

# process_method is now resolved in Python (input_writer.py) and read from CSV
set process_method dimen 2 within {process, method};
table data IN 'CSV' 'input/process_method.csv' : process_method <- [process,method];
set process__profileProcess__toSink__profile__profile_method dimen 5;  # Migrated to Python (preprocessing/process_method_sets.py).
table data IN 'CSV' 'solve_data/process__profileProcess__toSink__profile__profile_method.csv' : process__profileProcess__toSink__profile__profile_method <- [process_outer, process, sink, profile, profile_method];
set process__profileProcess__toSink dimen 3;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/process__profileProcess__toSink.csv' : process__profileProcess__toSink <- [process_outer, process, sink];
set process__source__toProfileProcess__profile__profile_method dimen 5;  # Migrated to Python (preprocessing/process_method_sets.py).
table data IN 'CSV' 'solve_data/process__source__toProfileProcess__profile__profile_method.csv' : process__source__toProfileProcess__profile__profile_method <- [process, source, process_aux, profile, profile_method];
set process__source__toProfileProcess dimen 3;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/process__source__toProfileProcess.csv' : process__source__toProfileProcess <- [process, source, process_aux];
set process_profile;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/process_profile.csv' : process_profile <- [process];
set process_source_toProcess dimen 3;  # Migrated to Python (preprocessing/process_arc_unions.py).
set process_process_toSink dimen 3;    # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/process_source_toProcess.csv' : process_source_toProcess <- [process, source, process_aux];
table data IN 'CSV' 'solve_data/process_process_toSink.csv' : process_process_toSink <- [process_outer, process, sink];
set process_sink_toProcess dimen 3;  # Migrated to Python (preprocessing/process_method_sets.py).
table data IN 'CSV' 'solve_data/process_sink_toProcess.csv' : process_sink_toProcess <- [process, sink, process_aux];
set process_process_toSource dimen 3;  # Migrated to Python (preprocessing/process_method_sets.py).
table data IN 'CSV' 'solve_data/process_process_toSource.csv' : process_process_toSource <- [process_outer, process, source];
set process_source_toSink dimen 3;  # Migrated to Python (preprocessing/process_method_sets.py).
table data IN 'CSV' 'solve_data/process_source_toSink.csv' : process_source_toSink <- [process, source, sink];
set process_source_toProcess_direct dimen 3;  # Migrated to Python (preprocessing/process_method_sets.py).
table data IN 'CSV' 'solve_data/process_source_toProcess_direct.csv' : process_source_toProcess_direct <- [process, source, process_aux];
set process_process_toSink_direct dimen 3;  # Migrated to Python (preprocessing/process_method_sets.py).
table data IN 'CSV' 'solve_data/process_process_toSink_direct.csv' : process_process_toSink_direct <- [process_outer, process, sink];
set process_sink_toProcess_direct dimen 3;  # Migrated to Python (preprocessing/process_method_sets.py).
table data IN 'CSV' 'solve_data/process_sink_toProcess_direct.csv' : process_sink_toProcess_direct <- [process, sink, process_aux];
set process_process_toSource_direct dimen 3;  # Migrated to Python (preprocessing/process_method_sets.py).
table data IN 'CSV' 'solve_data/process_process_toSource_direct.csv' : process_process_toSource_direct <- [process_outer, process, source];
set process_sink_toSource dimen 3;  # Migrated to Python (preprocessing/process_method_sets.py).
table data IN 'CSV' 'solve_data/process_sink_toSource.csv' : process_sink_toSource <- [process, sink, source];
set process__source__sink__profile__profile_method_direct dimen 5;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/process__source__sink__profile__profile_method_direct.csv' : process__source__sink__profile__profile_method_direct <- [process, source, sink, profile, profile_method];
set process_process_toSink_noConversion dimen 3;  # Migrated to Python (preprocessing/process_method_sets.py).
table data IN 'CSV' 'solve_data/process_process_toSink_noConversion.csv' : process_process_toSink_noConversion <- [process_outer, process, sink];
set process_source_toProcess_noConversion dimen 3;  # Migrated to Python (preprocessing/process_method_sets.py).
table data IN 'CSV' 'solve_data/process_source_toProcess_noConversion.csv' : process_source_toProcess_noConversion <- [process, source, process_aux];

set process_source_sink dimen 3;                # Migrated to Python (preprocessing/process_arc_unions.py).
set process_source_sink_alwaysProcess dimen 3;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/process_source_sink.csv' : process_source_sink <- [process, source, sink];
table data IN 'CSV' 'solve_data/process_source_sink_alwaysProcess.csv' : process_source_sink_alwaysProcess <- [process, source, sink];

set process_method_sources_sinks dimen 6;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/process_method_sources_sinks.csv' : process_method_sources_sinks <- [process, method, orig_source, orig_sink, always_source, always_sink];

set process_source_sink_noEff dimen 3;  # Migrated to Python (preprocessing/process_arc_unions.py).
set process_source_sink_eff dimen 3;    # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/process_source_sink_noEff.csv' : process_source_sink_noEff <- [process, source, sink];
table data IN 'CSV' 'solve_data/process_source_sink_eff.csv' : process_source_sink_eff <- [process, source, sink];

set process__source__sink__profile__profile_method_connection dimen 5;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/process__source__sink__profile__profile_method_connection.csv' : process__source__sink__profile__profile_method_connection <- [process, source, sink, profile, profile_method];
set process__source__sink__profile__profile_method dimen 5;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/process__source__sink__profile__profile_method.csv' : process__source__sink__profile__profile_method <- [process, source, sink, profile, profile_method];

set process__source__sinkIsNode dimen 3;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/process__source__sinkIsNode.csv' : process__source__sinkIsNode <- [process, source, sink];

set process_online_linear 'processes with an online status using linear variable';  # Migrated to Python (preprocessing/process_method_sets.py).
set process_online_integer 'processes with an online status using integer variable';  # Migrated to Python (preprocessing/process_method_sets.py).
table data IN 'CSV' 'solve_data/process_online_linear.csv' : process_online_linear <- [process];
table data IN 'CSV' 'solve_data/process_online_integer.csv' : process_online_integer <- [process];
set process_online;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/process_online.csv' : process_online <- [process];

# Timestep sets that have at least one lookback entry (for min up/downtime constraint indexing)
set pdt_uptime dimen 3;    # Migrated to Python (preprocessing/per_solve_sets.py).
set pdt_downtime dimen 3;  # Migrated to Python (preprocessing/per_solve_sets.py).
table data IN 'CSV' 'solve_data/pdt_uptime_set.csv' : pdt_uptime <- [process, period, time];
table data IN 'CSV' 'solve_data/pdt_downtime_set.csv' : pdt_downtime <- [process, period, time];

# Next-timestep mapping (inverse of dtttdt) for shutdown tightening
set dtdt_next dimen 4;  # Migrated to Python (preprocessing/per_solve_sets.py).
table data IN 'CSV' 'solve_data/dtdt_next_set.csv' : dtdt_next <- [period_prev, time_prev_solve, period, time];

set peedt dimen 5;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/peedt.csv' : peedt <- [process, source, sink, period, time];

set process_source_undelayed dimen 2;  # Migrated to Python (preprocessing/process_arc_unions.py).
set process_source_delayed   dimen 2;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/process_source_undelayed.csv' : process_source_undelayed <- [process, source];
table data IN 'CSV' 'solve_data/process_source_delayed.csv'   : process_source_delayed   <- [process, source];
set process_source_sink_undelayed dimen 3;  # Migrated to Python (preprocessing/process_arc_unions.py).
set process_source_sink_delayed   dimen 3;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/process_source_sink_undelayed.csv' : process_source_sink_undelayed <- [process, source, sink];
table data IN 'CSV' 'solve_data/process_source_sink_delayed.csv'   : process_source_sink_delayed   <- [process, source, sink];

param p_process_delay_weight {(p, td) in process_delayed__duration};  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/p_process_delay_weight.csv' : [process, delay_duration], p_process_delay_weight~value;

param pdCommodity {c in commodity, param in commodityPeriodParam, d in period_in_use};  # Migrated to Python (preprocessing/entity_period_calc_params.py).
table data IN 'CSV' 'solve_data/pdCommodity.csv' : [commodity, param, period], pdCommodity~value;

param pdtCommodity {c in commodity, param in commodityTimeParam, (d, t) in dt};  # Migrated to Python (preprocessing/entity_period_calc_params.py).
table data IN 'CSV' 'solve_data/pdtCommodity.csv' : [commodity, param, period, time], pdtCommodity~value;

param pdGroup {g in group, param in groupPeriodParam, d in period_in_use};  # Migrated to Python (preprocessing/entity_period_calc_params.py).
table data IN 'CSV' 'solve_data/pdGroup.csv' : [group, param, period], pdGroup~value;

param pdtGroup {g in group, param in groupTimeParam, (d, t) in dt};  # Migrated to Python (preprocessing/entity_period_calc_params.py).
table data IN 'CSV' 'solve_data/pdtGroup.csv' : [group, param, period, time], pdtGroup~value;

# Reserve method partitions migrated to Python (preprocessing/reserve_method_partitions.py).
set reserve__upDown__group__method_timeseries dimen 4;
set reserve__upDown__group__method_dynamic dimen 4;
set reserve__upDown__group__method_n_1 dimen 4;
table data IN 'CSV' 'solve_data/reserve__upDown__group__method_timeseries.csv' : reserve__upDown__group__method_timeseries <- [reserve, upDown, group, method];
table data IN 'CSV' 'solve_data/reserve__upDown__group__method_dynamic.csv' : reserve__upDown__group__method_dynamic <- [reserve, upDown, group, method];
table data IN 'CSV' 'solve_data/reserve__upDown__group__method_n_1.csv' : reserve__upDown__group__method_n_1 <- [reserve, upDown, group, method];

set process__method_indirect dimen 2;  # Migrated to Python (preprocessing/process_method_sets.py).
table data IN 'CSV' 'solve_data/process__method_indirect.csv' : process__method_indirect <- [process, method];

set process__sourceIsNode__sink_1way_noSinkOrMoreThan1Source dimen 3;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/process__sourceIsNode__sink_1way_noSinkOrMoreThan1Source.csv' : process__sourceIsNode__sink_1way_noSinkOrMoreThan1Source <- [process, source, sink];
set process__source__sinkIsNode_not2way1var dimen 3;  # Migrated to Python (preprocessing/process_arc_unions.py).
set process__source__sinkIsNode_2way1var    dimen 3;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/process__source__sinkIsNode_not2way1var.csv' : process__source__sinkIsNode_not2way1var <- [process, source, sink];
table data IN 'CSV' 'solve_data/process__source__sinkIsNode_2way1var.csv'    : process__source__sinkIsNode_2way1var    <- [process, source, sink];
set process_sinkIsNode_2way1var dimen 1;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/process_sinkIsNode_2way1var.csv' : process_sinkIsNode_2way1var <- [process];
set process__source__sinkIsNode_2way2var dimen 3;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/process__source__sinkIsNode_2way2var.csv' : process__source__sinkIsNode_2way2var <- [process, source, sink];

set gdt_maxInstantFlow dimen 3;  # Migrated to Python (preprocessing/process_arc_unions.py).
set gdt_minInstantFlow dimen 3;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/gdt_maxInstantFlow.csv' : gdt_maxInstantFlow <- [group, period, time];
table data IN 'CSV' 'solve_data/gdt_minInstantFlow.csv' : gdt_minInstantFlow <- [group, period, time];

set process__source__timeParam dimen 3;  # Migrated to Python (preprocessing/process_arc_unions.py).
set process__sink__timeParam   dimen 3;  # Migrated to Python (preprocessing/process_arc_unions.py).
set process__timeParam         dimen 2;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/process__source__timeParam.csv' : process__source__timeParam <- [process, source, param];
table data IN 'CSV' 'solve_data/process__sink__timeParam.csv'   : process__sink__timeParam   <- [process, sink, param];
table data IN 'CSV' 'solve_data/process__timeParam.csv'         : process__timeParam         <- [process, param];

set process__source__sink__param dimen 4;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/process__source__sink__param.csv' : process__source__sink__param <- [process, source, sink, param];
set process__source__sink__param_t dimen 4;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/process__source__sink__param_t.csv' : process__source__sink__param_t <- [process, source, sink, param];


set process_source_sink_param_t dimen 4;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/process_source_sink_param_t.csv' : process_source_sink_param_t <- [process, source, sink, param];

set process__source__sink__ramp_method dimen 4;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/process__source__sink__ramp_method.csv' : process__source__sink__ramp_method <- [process, source, sink, ramp_method];

set node__PeriodParam_in_use dimen 2;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/node__PeriodParam_in_use.csv' : node__PeriodParam_in_use <- [node, param];

set node__TimeParam_in_use dimen 2;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/node__TimeParam_in_use.csv' : node__TimeParam_in_use <- [node, param];

param pdNode {(n, param) in node__PeriodParam_in_use, d in period_with_history};  # Migrated to Python (preprocessing/entity_period_calc_params.py).
table data IN 'CSV' 'solve_data/pdNode.csv' : [node, param, period], pdNode~value;

param pdtNode {(n, param) in node__TimeParam_in_use, (d, t) in dt};  # Migrated to Python (preprocessing/entity_period_calc_params.py).
table data IN 'CSV' 'solve_data/pdtNode.csv' : [node, param, period, time], pdtNode~value;

param ptNode_inflow {n in node, t in time};  # Migrated to Python.
table data IN 'CSV' 'solve_data/ptNode_inflow.csv' : [node, time], ptNode_inflow~value;
set nodeSelfDischarge dimen 1;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/nodeSelfDischarge.csv' : nodeSelfDischarge <- [node];

set process__PeriodParam_in_use dimen 2;  # Migrated to Python (preprocessing/process_arc_unions.py).
set process_TimeParam_in_use dimen 2;     # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/process__PeriodParam_in_use.csv' : process__PeriodParam_in_use <- [process, param];
table data IN 'CSV' 'solve_data/process_TimeParam_in_use.csv' : process_TimeParam_in_use <- [process, param];

param pdProcess {(p, param) in process__PeriodParam_in_use, d in period_with_history};  # Migrated to Python (preprocessing/entity_period_calc_params.py).
table data IN 'CSV' 'solve_data/pdProcess.csv' : [process, param, period], pdProcess~value;
param pdtProcess {(p, param) in process_TimeParam_in_use, (d,t) in dt} default 0;  # Migrated to Python (preprocessing/entity_period_calc_params.py).  Sparse CSV: writer skips zero-valued rows; mod's `default 0` substitutes.
table data IN 'CSV' 'solve_data/pdtProcess.csv' : [process, param, period, time], pdtProcess~value;
param pdtProfile {p in profile, (d,t) in dt};  # Migrated to Python (preprocessing/entity_period_calc_params.py).
table data IN 'CSV' 'solve_data/pdtProfile.csv' : [profile, period, time], pdtProfile~value;

param p_entity_unitsize {e in entity};  # Migrated to Python (preprocessing/entity_period_calc_params.py).
table data IN 'CSV' 'solve_data/p_entity_unitsize.csv' : [entity], p_entity_unitsize~value;

param edEntity_lifetime {e in entity, d in period_with_history};  # Migrated to Python (preprocessing/entity_period_calc_params.py).
table data IN 'CSV' 'solve_data/edEntity_lifetime.csv' : [entity, period], edEntity_lifetime~value;

param pProcess_source_sink {(p, source, sink, param) in process__source__sink__param};  # Migrated to Python (preprocessing/entity_period_calc_params.py).
table data IN 'CSV' 'solve_data/pProcess_source_sink.csv' : [process, source, sink, param], pProcess_source_sink~value;

set process_source_sourceSinkTimeParam_in_use dimen 3;  # Migrated to Python.
set process_sink_sourceSinkTimeParam_in_use dimen 3;    # Migrated to Python.
set process_source_sourceSinkPeriodParam_in_use dimen 3;  # Migrated to Python.
set process_sink_sourceSinkPeriodParam_in_use dimen 3;    # Migrated to Python.
table data IN 'CSV' 'solve_data/process_source_sourceSinkTimeParam_in_use.csv' : process_source_sourceSinkTimeParam_in_use <- [process, source, param];
table data IN 'CSV' 'solve_data/process_sink_sourceSinkTimeParam_in_use.csv' : process_sink_sourceSinkTimeParam_in_use <- [process, sink, param];
table data IN 'CSV' 'solve_data/process_source_sourceSinkPeriodParam_in_use.csv' : process_source_sourceSinkPeriodParam_in_use <- [process, source, param];
table data IN 'CSV' 'solve_data/process_sink_sourceSinkPeriodParam_in_use.csv' : process_sink_sourceSinkPeriodParam_in_use <- [process, sink, param];

param pdtProcess_source {(p, source, param) in process_source_sourceSinkTimeParam_in_use, (d, t) in dt} default 0;  # Migrated to Python (preprocessing/entity_period_calc_params.py).  Sparse CSV: writer skips zero-valued rows; mod's `default 0` substitutes.
table data IN 'CSV' 'solve_data/pdtProcess_source.csv' : [process, source, param, period, time], pdtProcess_source~value;

param pdtProcess_sink {(p, sink, param) in process_sink_sourceSinkTimeParam_in_use, (d, t) in dt} default 0;  # Migrated to Python (preprocessing/entity_period_calc_params.py).  Sparse CSV: writer skips zero-valued rows; mod's `default 0` substitutes.
table data IN 'CSV' 'solve_data/pdtProcess_sink.csv' : [process, sink, param, period, time], pdtProcess_sink~value;

param pdtProcess_source_sink {(p, source, sink, param) in process__source__sink__param_t, (d, t) in dt};  # Migrated to Python (preprocessing/entity_period_calc_params.py).
table data IN 'CSV' 'solve_data/pdtProcess_source_sink.csv' : [process, source, sink, param, period, time], pdtProcess_source_sink~value;


param pdtReserve_upDown_group {(r, ud, g) in reserve__upDown__group, param in reserveTimeParam, (d,t) in dt};  # Migrated to Python (preprocessing/reserve_calc_params.py).
table data IN 'CSV' 'solve_data/pdtReserve_upDown_group.csv' : [reserve, upDown, group, param, period, time], pdtReserve_upDown_group~value;
set process_reserve_upDown_node_active dimen 4;  # Migrated to Python (preprocessing/reserve_calc_params.py).
table data IN 'CSV' 'solve_data/process_reserve_upDown_node_active.csv' : process_reserve_upDown_node_active <- [process, reserve, upDown, node];
set prundt dimen 6;  # Migrated to Python (preprocessing/reserve_calc_params.py).
table data IN 'CSV' 'solve_data/prundt.csv' : prundt <- [process, reserve, upDown, node, period, time];
set pdt_online_linear  dimen 3;  # Migrated to Python (preprocessing/process_arc_unions.py).
set pdt_online_integer dimen 3;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/pdt_online_linear.csv'  : pdt_online_linear  <- [process, period, time];
table data IN 'CSV' 'solve_data/pdt_online_integer.csv' : pdt_online_integer <- [process, period, time];

param hours_in_period{d in period_in_use};  # Migrated to Python (preprocessing/period_calculated_params.py).
table data IN 'CSV' 'solve_data/hours_in_period.csv' : [period], hours_in_period~value;
param period_share_of_year{d in period_in_use};  # Migrated to Python (preprocessing/period_calculated_params.py).
table data IN 'CSV' 'solve_data/period_share_of_year.csv' : [period], period_share_of_year~value;
param p_years_d{d in period_with_history};            # Migrated to Python (preprocessing/period_calculated_params.py).
param p_years_represented_d{d in periodAll};          # Migrated to Python (preprocessing/period_calculated_params.py).
table data IN 'CSV' 'solve_data/p_years_d.csv' : [period], p_years_d~value;
table data IN 'CSV' 'solve_data/p_years_represented_d_calc.csv' : [period], p_years_represented_d~value;

param complete_hours_in_period{d in period_in_use};       # Migrated to Python (preprocessing/period_calculated_params.py).
param complete_period_share_of_year{d in period_in_use};  # Migrated to Python (preprocessing/period_calculated_params.py).
table data IN 'CSV' 'solve_data/complete_hours_in_period.csv' : [period], complete_hours_in_period~value;
table data IN 'CSV' 'solve_data/complete_period_share_of_year_calc.csv' : [period], complete_period_share_of_year~value;

# Rolling "fraction of period d filled" — central to the rolling-aware
# ladder caps below.  Numerator = sim-hours of period d already realized
# across prior rolls (p_ladder_cum_sim_hours[d]) plus sim-hours of d seen
# in the current roll's horizon.  Denominator = full sim-hours of d for
# one representative year (share_of_year * 8760).  On a single solve
# covering all of period d this is exactly 1.0 so the rolling caps reduce
# to their pre-refactor form bit-for-bit.
param f_d_k {d in period_in_use};  # Migrated to Python (preprocessing/period_calculated_params.py).
table data IN 'CSV' 'solve_data/f_d_k.csv' : [period], f_d_k~value;

param period_share_of_annual_flow {n in node, d in period_in_use};   # Migrated to Python.
param period_flow_annual_multiplier {n in node, d in period_in_use};  # Migrated to Python.
param orig_flow_sum {n in node, d in period_in_use};                  # Migrated to Python.
param period_flow_proportional_multiplier {n in node, d in period_in_use};  # Migrated to Python.
param new_peak_sign{n in node, d in period_in_use};                   # Migrated to Python.
param old_peak_max{n in node, d in period_in_use};                    # Migrated to Python.
param old_peak_min{n in node, d in period_in_use};                    # Migrated to Python.
param old_peak_sign{n in node, d in period_in_use};                   # Migrated to Python.
param old_peak{n in node, d in period_in_use};                        # Migrated to Python.
table data IN 'CSV' 'solve_data/period_share_of_annual_flow.csv' : [node, period], period_share_of_annual_flow~value;
table data IN 'CSV' 'solve_data/period_flow_annual_multiplier.csv' : [node, period], period_flow_annual_multiplier~value;
table data IN 'CSV' 'solve_data/orig_flow_sum.csv' : [node, period], orig_flow_sum~value;
table data IN 'CSV' 'solve_data/period_flow_proportional_multiplier.csv' : [node, period], period_flow_proportional_multiplier~value;
table data IN 'CSV' 'solve_data/new_peak_sign.csv' : [node, period], new_peak_sign~value;
table data IN 'CSV' 'solve_data/old_peak_max.csv' : [node, period], old_peak_max~value;
table data IN 'CSV' 'solve_data/old_peak_min.csv' : [node, period], old_peak_min~value;
table data IN 'CSV' 'solve_data/old_peak_sign.csv' : [node, period], old_peak_sign~value;
table data IN 'CSV' 'solve_data/old_peak.csv' : [node, period], old_peak~value;
printf ('Checking: if the sign of new peak inflow is the same as the sign ');
printf ('of the peak inflow in the original inflow time series\n');
check {n in node, d in period_in_use : (n, 'scale_to_annual_and_peak_flow') in node__inflow_method && pdNode[n, 'annual_flow', d] && pdNode[n, 'peak_inflow', d]} new_peak_sign[n, d] = old_peak_sign[n, d];

param new_peak_divided_by_old_peak {n in node, d in period_in_use};            # Migrated to Python.
param new_peak_divide_by_old_peak_sum_inflow {n in node, d in period_in_use};  # Migrated to Python.
param new_peak_inflow_sum {n in node, d in period_in_use};                     # Migrated to Python.
param new_old_multiplier {n in node, d in period_in_use};                      # Migrated to Python.
param new_old_slope {n in node, d in period_in_use};                           # Migrated to Python.
param new_old_section {n in node, d in period_in_use};                         # Migrated to Python.
table data IN 'CSV' 'solve_data/new_peak_divided_by_old_peak.csv' : [node, period], new_peak_divided_by_old_peak~value;
table data IN 'CSV' 'solve_data/new_peak_divide_by_old_peak_sum_inflow.csv' : [node, period], new_peak_divide_by_old_peak_sum_inflow~value;
table data IN 'CSV' 'solve_data/new_peak_inflow_sum.csv' : [node, period], new_peak_inflow_sum~value;
table data IN 'CSV' 'solve_data/new_old_multiplier.csv' : [node, period], new_old_multiplier~value;
table data IN 'CSV' 'solve_data/new_old_slope.csv' : [node, period], new_old_slope~value;
table data IN 'CSV' 'solve_data/new_old_section.csv' : [node, period], new_old_section~value;
param pdtNodeInflow {n in node, (d, t) in dt : (n, 'no_inflow') not in node__inflow_method};  # Migrated to Python (preprocessing/entity_period_calc_params.py).
table data IN 'CSV' 'solve_data/pdtNodeInflow.csv' : [node, period, time], pdtNodeInflow~value;
# Agent 5b (LP-scaling): row scalers for node and group, computed from
# connected-unit unitsizes when the solve opts in via
# p_use_row_scaling == 1; otherwise held at 1 (pre-Agent-5 behaviour).
#
# The formula rounds to the nearest power of 10 so structurally-
# identical entities share the same scaler (preserves HiGHS symmetry
# detection).  When a node has no connected processes (pure demand
# node), the max absolute inflow across timesteps is used as a
# fallback; if that is also zero the scaler collapses to 1.  Final
# value is clamped to [1e-6, 1e9] so no pathological input produces a
# nonsense scaler.
param _node_cap_unitsize_sum {n in node, d in period_in_use};       # Migrated to Python.
param _node_cap_inflow_fallback {n in node, d in period_in_use};     # Migrated to Python (preprocessing/node_inflow_scaling_params.py).
param _node_cap_raw {n in node, d in period_in_use};                # Migrated to Python.
param _node_cap_pow10 {n in node, d in period_in_use};              # Migrated to Python.
param node_capacity_for_scaling{n in node, d in period_in_use};     # Migrated to Python.
param _group_cap_raw {g in group, d in period_in_use};              # Migrated to Python.
param _group_cap_pow10 {g in group, d in period_in_use};            # Migrated to Python.
param group_capacity_for_scaling{g in group, d in period_in_use};   # Migrated to Python.
table data IN 'CSV' 'solve_data/_node_cap_unitsize_sum.csv' : [node, period], _node_cap_unitsize_sum~value;
table data IN 'CSV' 'solve_data/_node_cap_inflow_fallback.csv' : [node, period], _node_cap_inflow_fallback~value;
table data IN 'CSV' 'solve_data/_node_cap_raw.csv' : [node, period], _node_cap_raw~value;
table data IN 'CSV' 'solve_data/_node_cap_pow10.csv' : [node, period], _node_cap_pow10~value;
table data IN 'CSV' 'solve_data/node_capacity_for_scaling.csv' : [node, period], node_capacity_for_scaling~value;
table data IN 'CSV' 'solve_data/_group_cap_raw.csv' : [group, period], _group_cap_raw~value;
table data IN 'CSV' 'solve_data/_group_cap_pow10.csv' : [group, period], _group_cap_pow10~value;
table data IN 'CSV' 'solve_data/group_capacity_for_scaling.csv' : [group, period], group_capacity_for_scaling~value;
# Agent 5c (LP-scaling): precomputed reciprocals of the row scalers
# above.  Multiplying every term of a balance or group-aggregation
# constraint by the reciprocal divides the whole row by its scaler,
# compressing the matrix coefficients to O(1) relative to the row.  In
# Mode A (flag = 0) both scalers are 1, so these reciprocals are 1 and
# every `* inv_*_cap[.]` factor in the constraints collapses to a
# no-op at AMPL parse time.
param inv_node_cap{n in node, d in period_in_use};   # Migrated to Python.
param inv_group_cap{g in group, d in period_in_use}; # Migrated to Python.
table data IN 'CSV' 'solve_data/inv_node_cap.csv' : [node, period], inv_node_cap~value;
table data IN 'CSV' 'solve_data/inv_group_cap.csv' : [group, period], inv_group_cap~value;
# p_inflation, p_infl_offset_investment, p_infl_offset_operations
# were dead scalars — declared from input/p_inflation_rate.csv etc.
# but never used in any constraint. Inflation factors that the LP
# actually consumes are computed in Python preprocessing
# (period_calculated_params.py:read_p_inflation_factors). Removed
# in batch 69. The model-indexed source params at L542-544 remain
# loaded but are now also unreferenced — left in place for
# documentation; deletable in a future cleanup if the `set model`
# loader (L757) is repointed to p_max_flow_for_unconstrained_variables.csv.
param p_unconstrained_flow_cap := (if sum{m in model} 1 then max{m in model} p_max_flow_for_unconstrained_variables[m] else 1000000);
# Inflation offsets are fractions of the represented window p_years_represented[d, y]:
#   0   = beginning of the window
#   0.5 = midway through the window
#   1   = end of the window
# Defaults: investment annuity 0 (start-of-year — "paid when built"),
# operational costs 0.5 (mid-year average).  User-provided values on the
# model entity override the defaults via p_inflation_offset_*.
param p_years_until_dispatch{(d, y) in period__year};  # Migrated to Python (preprocessing/period_calculated_params.py).
param p_years_until_invest{(d, y) in period__year};    # Migrated to Python (preprocessing/period_calculated_params.py).
table data IN 'CSV' 'solve_data/p_years_until_dispatch.csv' : [period, year], p_years_until_dispatch~value;
table data IN 'CSV' 'solve_data/p_years_until_invest.csv' : [period, year], p_years_until_invest~value;
param p_inflation_factor_investment_yearly{d in period};        # Migrated to Python.
param p_inflation_factor_operations_yearly{d in period_in_use};  # Migrated to Python.
table data IN 'CSV' 'solve_data/p_inflation_factor_investment_yearly.csv' : [period], p_inflation_factor_investment_yearly~value;
table data IN 'CSV' 'solve_data/p_inflation_factor_operations_yearly.csv' : [period], p_inflation_factor_operations_yearly~value;

# Check for division by zero
printf 'Checking: node lifetime > 0, if the node is investing or divesting';
check {e in (entityInvest union entityDivest), d in period_invest : e in node} pdNode[e, 'lifetime', d] > 0;
printf 'Checking: process lifetime > 0, if the process is investing or divesting';
check {e in (entityInvest union entityDivest), d in period_invest : e in process} pdProcess[e, 'lifetime', d] > 0;

# discount_rate defaults to 0.05 and lifetime to 20 when no value is recorded
# on the entity.  The DB parameter definition's default_value is advisory
# only (no mechanism to cascade it into p_node / p_process), so the defaults
# are applied inline below to avoid 0 / 0 in the annuity factor
# r / (1 - 1/(1+r)^n).  The lifetime fallback is a belt-and-braces backup
# to the check above; the check is authoritative for investing/divesting
# entities.

param ed_entity_annual{e in entityInvest, d in period_invest};  # Migrated to Python (preprocessing/entity_annual_calc_params.py).
table data IN 'CSV' 'solve_data/ed_entity_annual.csv' : [entity, period], ed_entity_annual~value;

param ed_entity_annual_discounted{e in entityInvest, d in period_invest};  # Migrated to Python.
table data IN 'CSV' 'solve_data/ed_entity_annual_discounted.csv' : [entity, period], ed_entity_annual_discounted~value;

param ed_entity_annual_divest{e in entityDivest, d in period_invest};            # Migrated to Python.
param ed_entity_annual_divest_discounted{e in entityDivest, d in period_invest};  # Migrated to Python.
table data IN 'CSV' 'solve_data/ed_entity_annual_divest.csv' : [entity, period], ed_entity_annual_divest~value;
table data IN 'CSV' 'solve_data/ed_entity_annual_divest_discounted.csv' : [entity, period], ed_entity_annual_divest_discounted~value;

param ed_fixed_cost{e in entity, d in period_with_history};  # Migrated to Python (preprocessing/entity_period_calc_params.py).
table data IN 'CSV' 'solve_data/ed_fixed_cost.csv' : [entity, period], ed_fixed_cost~value;

param ed_lifetime_fixed_cost{e in entity, d in period_with_history};  # Migrated to Python.
table data IN 'CSV' 'solve_data/ed_lifetime_fixed_cost.csv' : [entity, period], ed_lifetime_fixed_cost~value;

param ed_lifetime_fixed_cost_divest{e in entityDivest, d in period_invest};  # Migrated to Python.
table data IN 'CSV' 'solve_data/ed_lifetime_fixed_cost_divest.csv' : [entity, period], ed_lifetime_fixed_cost_divest~value;


set process_minload;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/process_minload.csv' : process_minload <- [process];

param pdtConversion_rate{p in process, (d, t) in dt};  # Migrated to Python (preprocessing/entity_period_calc_params.py).
table data IN 'CSV' 'solve_data/pdtConversion_rate.csv' : [process, period, time], pdtConversion_rate~value;

param pdtProcess_section{p in process_minload, (d, t) in dt};  # Migrated to Python (preprocessing/entity_period_calc_params.py).
table data IN 'CSV' 'solve_data/pdtProcess_section.csv' : [process, period, time], pdtProcess_section~value;

param pdtProcess_slope{p in process, (d, t) in dt};  # Migrated to Python (preprocessing/entity_period_calc_params.py).
table data IN 'CSV' 'solve_data/pdtProcess_slope.csv' : [process, period, time], pdtProcess_slope~value;

param pdtProcess__source__sink__dt_varCost {(p, source, sink) in process_source_sink, (d, t) in dt};  # Migrated to Python (preprocessing/entity_period_calc_params.py).
table data IN 'CSV' 'solve_data/pdtProcess__source__sink__dt_varCost.csv' : [process, source, sink, period, time], pdtProcess__source__sink__dt_varCost~value;

param pdtProcess__source__sink__dt_varCost_alwaysProcess {(p, source, sink) in process_source_sink_alwaysProcess, (d, t) in dt};  # Migrated to Python (preprocessing/entity_period_calc_params.py).
table data IN 'CSV' 'solve_data/pdtProcess__source__sink__dt_varCost_alwaysProcess.csv' : [process, source, sink, period, time], pdtProcess__source__sink__dt_varCost_alwaysProcess~value;
set pssdt_varCost_noEff dimen 5;  # Migrated to Python (preprocessing/entity_period_calc_params.py).
set pssdt_varCost_eff_unit_source dimen 5;  # Migrated to Python.
set pssdt_varCost_eff_unit_sink dimen 5;    # Migrated to Python.
set pssdt_varCost_eff_connection dimen 5;   # Migrated to Python.
table data IN 'CSV' 'solve_data/pssdt_varCost_noEff.csv' : pssdt_varCost_noEff <- [process, source, sink, period, time];
table data IN 'CSV' 'solve_data/pssdt_varCost_eff_unit_source.csv' : pssdt_varCost_eff_unit_source <- [process, source, sink, period, time];
table data IN 'CSV' 'solve_data/pssdt_varCost_eff_unit_sink.csv' : pssdt_varCost_eff_unit_sink <- [process, source, sink, period, time];
table data IN 'CSV' 'solve_data/pssdt_varCost_eff_connection.csv' : pssdt_varCost_eff_connection <- [process, source, sink, period, time];
set ed_invest dimen 2;            # Migrated to Python (preprocessing/invest_divest_sets.py).
set ed_invest_period dimen 2;     # Migrated to Python.
set e_invest_total;               # Migrated to Python (preprocessing/invest_total_sets.py).
set ed_invest_cumulative dimen 2; # Migrated to Python.
set edd_history_choice dimen 3;   # Migrated to Python.
set edd_history_automatic dimen 3;# Migrated to Python.
set edd_history_no_investment dimen 3;  # Migrated to Python.
set edd_history dimen 3;          # Migrated to Python.
set edd_history_invest dimen 3;   # Migrated to Python.
set edd_invest dimen 3;           # Migrated to Python.
set pd_invest dimen 2;            # Migrated to Python.
set nd_invest dimen 2;            # Migrated to Python.
set ed_divest dimen 2;            # Migrated to Python.
set ed_divest_period dimen 2;     # Migrated to Python.
set e_divest_total;               # Migrated to Python (preprocessing/invest_total_sets.py).
set pd_divest dimen 2;            # Migrated to Python.
set nd_divest dimen 2;            # Migrated to Python.
set gd_invest dimen 2;            # Migrated to Python.
set gd_invest_period dimen 2;     # Migrated to Python.
set g_invest_total;               # Migrated to Python (preprocessing/invest_total_sets.py).
set gd_divest dimen 2;            # Migrated to Python.
set gd_divest_period dimen 2;     # Migrated to Python.
table data IN 'CSV' 'solve_data/ed_invest.csv' : ed_invest <- [entity, period];
table data IN 'CSV' 'solve_data/ed_invest_period.csv' : ed_invest_period <- [entity, period];
table data IN 'CSV' 'solve_data/ed_invest_cumulative.csv' : ed_invest_cumulative <- [entity, period];
table data IN 'CSV' 'solve_data/edd_history_choice.csv' : edd_history_choice <- [entity, period_history, period];
table data IN 'CSV' 'solve_data/edd_history_automatic.csv' : edd_history_automatic <- [entity, period_history, period];
table data IN 'CSV' 'solve_data/edd_history_no_investment.csv' : edd_history_no_investment <- [entity, period_history, period];
table data IN 'CSV' 'solve_data/edd_history.csv' : edd_history <- [entity, period_history, period];
table data IN 'CSV' 'solve_data/edd_history_invest.csv' : edd_history_invest <- [entity, period_history, period];
table data IN 'CSV' 'solve_data/edd_invest.csv' : edd_invest <- [entity, period_history, period];
table data IN 'CSV' 'solve_data/pd_invest.csv' : pd_invest <- [process, period];
table data IN 'CSV' 'solve_data/nd_invest.csv' : nd_invest <- [node, period];
table data IN 'CSV' 'solve_data/ed_divest.csv' : ed_divest <- [entity, period];
table data IN 'CSV' 'solve_data/ed_divest_period.csv' : ed_divest_period <- [entity, period];
table data IN 'CSV' 'solve_data/pd_divest.csv' : pd_divest <- [process, period];
table data IN 'CSV' 'solve_data/nd_divest.csv' : nd_divest <- [node, period];
table data IN 'CSV' 'solve_data/gd_invest.csv' : gd_invest <- [group, period];
table data IN 'CSV' 'solve_data/gd_invest_period.csv' : gd_invest_period <- [group, period];
table data IN 'CSV' 'solve_data/gd_divest.csv' : gd_divest <- [group, period];
table data IN 'CSV' 'solve_data/gd_divest_period.csv' : gd_divest_period <- [group, period];
set g_divest_total;       # Migrated to Python (preprocessing/invest_total_sets.py).
set g_invest_cumulative;  # Migrated to Python (preprocessing/invest_total_sets.py).
table data IN 'CSV' 'solve_data/e_invest_total.csv' : e_invest_total <- [entity];
table data IN 'CSV' 'solve_data/e_divest_total.csv' : e_divest_total <- [entity];
table data IN 'CSV' 'solve_data/g_invest_total.csv' : g_invest_total <- [group];
table data IN 'CSV' 'solve_data/g_divest_total.csv' : g_divest_total <- [group];
table data IN 'CSV' 'solve_data/g_invest_cumulative.csv' : g_invest_cumulative <- [group];

# For entities with lifetime_method = 'no_investment', v_invest is allowed
# only within the first-period lifetime window. After p_years_d[period_first]
# + lifetime the asset retires once and the model is forbidden from
# rebuilding — the fix_v_invest_no_investment_eq constraint below pins
# v_invest[e, d] = 0 for these (e, d).
set ed_invest_forbidden_no_investment dimen 2;  # Migrated to Python (preprocessing/invest_divest_sets.py).
table data IN 'CSV' 'solve_data/ed_invest_forbidden_no_investment.csv'
    : ed_invest_forbidden_no_investment <- [entity, period];

# e_*_total params migrated to Python (preprocessing/entity_total_caps.py).
# Each is the sum of p_process[e, paramName] + p_node[e, paramName] (one
# of the two contributes per entity since process/node are disjoint).
# Computed during write_input and loaded back via table data IN below.
param e_invest_max_total{e in entityInvest};
param e_divest_max_total{e in entityDivest};
param e_invest_min_total{e in entityInvest};
param e_divest_min_total{e in entityDivest};
table data IN 'CSV' 'solve_data/e_invest_max_total.csv' : [entity], e_invest_max_total~value;
table data IN 'CSV' 'solve_data/e_divest_max_total.csv' : [entity], e_divest_max_total~value;
table data IN 'CSV' 'solve_data/e_invest_min_total.csv' : [entity], e_invest_min_total~value;
table data IN 'CSV' 'solve_data/e_divest_min_total.csv' : [entity], e_divest_min_total~value;

param ed_invest_max_period{(e, d) in ed_invest};         # Migrated to Python.
param ed_divest_max_period{(e, d) in ed_divest};         # Migrated to Python.
param ed_invest_min_period{(e, d) in ed_invest};         # Migrated to Python.
param ed_divest_min_period{(e, d) in ed_divest};         # Migrated to Python.
param ed_cumulative_max_capacity{(e, d) in ed_invest};   # Migrated to Python.
param ed_cumulative_min_capacity{(e, d) in ed_invest};   # Migrated to Python.
table data IN 'CSV' 'solve_data/ed_invest_max_period.csv' : [entity, period], ed_invest_max_period~value;
table data IN 'CSV' 'solve_data/ed_divest_max_period.csv' : [entity, period], ed_divest_max_period~value;
table data IN 'CSV' 'solve_data/ed_invest_min_period.csv' : [entity, period], ed_invest_min_period~value;
table data IN 'CSV' 'solve_data/ed_divest_min_period.csv' : [entity, period], ed_divest_min_period~value;
table data IN 'CSV' 'solve_data/ed_cumulative_max_capacity.csv' : [entity, period], ed_cumulative_max_capacity~value;
table data IN 'CSV' 'solve_data/ed_cumulative_min_capacity.csv' : [entity, period], ed_cumulative_min_capacity~value;

set process_source_sink_ramp_limit_source_up   dimen 3;  # Migrated to Python (preprocessing/process_arc_unions.py).
set process_source_sink_ramp_limit_sink_up     dimen 3;  # Migrated to Python (preprocessing/process_arc_unions.py).
set process_source_sink_ramp_limit_source_down dimen 3;  # Migrated to Python (preprocessing/process_arc_unions.py).
set process_source_sink_ramp_limit_sink_down   dimen 3;  # Migrated to Python (preprocessing/process_arc_unions.py).
set process_source_sink_ramp_cost              dimen 3;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/process_source_sink_ramp_limit_source_up.csv'   : process_source_sink_ramp_limit_source_up   <- [process, source, sink];
table data IN 'CSV' 'solve_data/process_source_sink_ramp_limit_sink_up.csv'     : process_source_sink_ramp_limit_sink_up     <- [process, source, sink];
table data IN 'CSV' 'solve_data/process_source_sink_ramp_limit_source_down.csv' : process_source_sink_ramp_limit_source_down <- [process, source, sink];
table data IN 'CSV' 'solve_data/process_source_sink_ramp_limit_sink_down.csv'   : process_source_sink_ramp_limit_sink_down   <- [process, source, sink];
table data IN 'CSV' 'solve_data/process_source_sink_ramp_cost.csv'              : process_source_sink_ramp_cost              <- [process, source, sink];
set process_source_sink_ramp dimen 3;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/process_source_sink_ramp.csv' : process_source_sink_ramp <- [process, source, sink];
# The 4 process_source_sink_dtttdt_ramp_limit_*_{up,down} sets that
# used to live here were unused (the ramp_*_constraint families at
# L3468-3543 inline the same filter on block_dtttdt) — removed in the
# batch 68 cleanup along with their CSV writers and table-data-IN.

param p_process_reserve_upDown_node_reliability {(p, r, ud, n) in process_reserve_upDown_node_active};  # Migrated to Python (preprocessing/reserve_calc_params.py).
table data IN 'CSV' 'solve_data/p_process_reserve_upDown_node_reliability.csv' : [process, reserve, upDown, node], p_process_reserve_upDown_node_reliability~value;

set process_reserve_upDown_node_increase_reserve_ratio dimen 4;  # Migrated to Python.
table data IN 'CSV' 'solve_data/process_reserve_upDown_node_increase_reserve_ratio.csv' : process_reserve_upDown_node_increase_reserve_ratio <- [process, reserve, upDown, node];
set process_reserve_upDown_node_large_failure_ratio dimen 4;  # Migrated to Python.
table data IN 'CSV' 'solve_data/process_reserve_upDown_node_large_failure_ratio.csv' : process_reserve_upDown_node_large_failure_ratio <- [process, reserve, upDown, node];
set process_large_failure;  # Migrated to Python.
table data IN 'CSV' 'solve_data/process_large_failure.csv' : process_large_failure <- [process];

# Morales-Espana startup/shutdown capacity reduction parameters
# If ramp_speed > 0: reduction = max(0, 1 - min_load - ramp_speed * 60 * step_duration)
# If no ramp: reduction = 0 (unit can reach full power instantly)
param p_startup_cap_reduction_sink {(p, sink) in process_sink, (d, t) in dt : p in process_online};  # Migrated to Python (preprocessing/entity_period_calc_params.py).
table data IN 'CSV' 'solve_data/p_startup_cap_reduction_sink.csv' : [process, sink, period, time], p_startup_cap_reduction_sink~value;

param p_shutdown_cap_reduction_sink {(p, sink) in process_sink, (d, t) in dt : p in process_online};  # Migrated to Python.
table data IN 'CSV' 'solve_data/p_shutdown_cap_reduction_sink.csv' : [process, sink, period, time], p_shutdown_cap_reduction_sink~value;

param p_startup_cap_reduction_source {(p, source) in process_source, (d, t) in dt : p in process_online};  # Migrated to Python.
table data IN 'CSV' 'solve_data/p_startup_cap_reduction_source.csv' : [process, source, period, time], p_startup_cap_reduction_source~value;

param p_shutdown_cap_reduction_source {(p, source) in process_source, (d, t) in dt : p in process_online};  # Migrated to Python.
table data IN 'CSV' 'solve_data/p_shutdown_cap_reduction_source.csv' : [process, source, period, time], p_shutdown_cap_reduction_source~value;

set gcndt_co2_price                          dimen 5;  # Migrated to Python (preprocessing/process_arc_unions.py).
set group_commodity_node_period_co2_period   dimen 4;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/gcndt_co2_price.csv'                          : gcndt_co2_price                          <- [group, commodity, node, period, time];
table data IN 'CSV' 'solve_data/group_commodity_node_period_co2_period.csv'   : group_commodity_node_period_co2_period   <- [group, commodity, node, period];

set group_commodity_node_period_co2_total dimen 3;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/group_commodity_node_period_co2_total.csv' : group_commodity_node_period_co2_total <- [group, commodity, node];

# Commodity-ladder index sets.  cnd_ladder lists (commodity, node, period)
# triples that need period-level v_trade variables; cndi_ladder_* adds the
# tier index, split per ladder method so each reads from its own tier set.
# cndi_ladder is the union used for v_trade declaration and the
# per-(c, n, d) balance row.
set cnd_ladder dimen 3;       # Migrated to Python (preprocessing/per_solve_sets.py).
set cndi_ladder_cum dimen 4;  # Migrated to Python (preprocessing/per_solve_sets.py).
set cndi_ladder_ann dimen 4;  # Migrated to Python (preprocessing/per_solve_sets.py).
set cndi_ladder dimen 4;      # Migrated to Python (preprocessing/per_solve_sets.py).
table data IN 'CSV' 'solve_data/cnd_ladder_set.csv' : cnd_ladder <- [commodity, node, period];
table data IN 'CSV' 'solve_data/cndi_ladder_cum_set.csv' : cndi_ladder_cum <- [commodity, node, period, tier];
table data IN 'CSV' 'solve_data/cndi_ladder_ann_set.csv' : cndi_ladder_ann <- [commodity, node, period, tier];
table data IN 'CSV' 'solve_data/cndi_ladder_set.csv' : cndi_ladder <- [commodity, node, period, tier];
set ci_ladder_cumulative dimen 2;  # Migrated to Python (preprocessing/invest_total_sets.py).
table data IN 'CSV' 'solve_data/ci_ladder_cumulative.csv' : ci_ladder_cumulative <- [commodity, tier];

set process__commodity__node dimen 3;  # Migrated to Python (preprocessing/structural_filters.py).
table data IN 'CSV' 'solve_data/process__commodity__node.csv' : process__commodity__node <- [process, commodity, node];

set commodity_node_co2 dimen 2;  # Migrated to Python (preprocessing/structural_filters.py).
table data IN 'CSV' 'solve_data/commodity_node_co2.csv' : commodity_node_co2 <- [commodity, node];

set process__commodity__node_co2 dimen 3;  # Migrated to Python (preprocessing/process_arc_unions.py).
set process_co2;                           # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/process__commodity__node_co2.csv' : process__commodity__node_co2 <- [process, commodity, node];
table data IN 'CSV' 'solve_data/process_co2.csv' : process_co2 <- [process];

set process__sink_nonSync dimen 2;  # Migrated to Python (preprocessing/nonsync_sets.py).
table data IN 'CSV' 'solve_data/process__sink_nonSync.csv' : process__sink_nonSync <- [process, sink];

  #|| sum{(p,m) in process_method: m in method_2way_1var} 1
  # Declaration moved to the top of the model alongside other process-related
  # sets so it precedes its `table data IN` reader.

set nodeGroupDispatch__process_fully_inside dimen 2;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/nodeGroupDispatch__process_fully_inside.csv' : nodeGroupDispatch__process_fully_inside <- [group, process];
set nodeGroupDispatch__process__unit__to_node_Not_in_aggregate         dimen 4;  # Migrated to Python (preprocessing/process_arc_unions.py).
set nodeGroupDispatch__process__node__to_unit_Not_in_aggregate         dimen 4;  # Migrated to Python (preprocessing/process_arc_unions.py).
set nodeGroupDispatch__group_aggregate__process__unit__to_node         dimen 5;  # Migrated to Python (preprocessing/process_arc_unions.py).
set nodeGroupDispatch__group_aggregate__process__node__to_unit         dimen 5;  # Migrated to Python (preprocessing/process_arc_unions.py).
set nodeGroupDispatch__process__node__to_connection_Not_in_aggregate   dimen 4;  # Migrated to Python (preprocessing/process_arc_unions.py).
set nodeGroupDispatch__process__connection__to_node_Not_in_aggregate   dimen 4;  # Migrated to Python (preprocessing/process_arc_unions.py).
set nodeGroupDispatch__connection_Not_in_aggregate                     dimen 2;  # Migrated to Python (preprocessing/process_arc_unions.py).
set nodeGroupDispatch__group_aggregate__process__connection__to_node   dimen 5;  # Migrated to Python (preprocessing/process_arc_unions.py).
set nodeGroupDispatch__group_aggregate__process__node__to_connection   dimen 5;  # Migrated to Python (preprocessing/process_arc_unions.py).
set nodeGroupDispatch__group_aggregate_Connection                      dimen 2;  # Migrated to Python (preprocessing/process_arc_unions.py).
set nodeGroupDispatch__group_aggregate_Unit_to_group                   dimen 2;  # Migrated to Python (preprocessing/process_arc_unions.py).
set nodeGroupDispatch__group_aggregate_Group_to_unit                   dimen 2;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/nodeGroupDispatch__process__unit__to_node_Not_in_aggregate.csv'         : nodeGroupDispatch__process__unit__to_node_Not_in_aggregate         <- [group, process, unit, node];
table data IN 'CSV' 'solve_data/nodeGroupDispatch__process__node__to_unit_Not_in_aggregate.csv'         : nodeGroupDispatch__process__node__to_unit_Not_in_aggregate         <- [group, process, node, unit];
table data IN 'CSV' 'solve_data/nodeGroupDispatch__group_aggregate__process__unit__to_node.csv'         : nodeGroupDispatch__group_aggregate__process__unit__to_node         <- [group, group_aggregate, unit, source, sink];
table data IN 'CSV' 'solve_data/nodeGroupDispatch__group_aggregate__process__node__to_unit.csv'         : nodeGroupDispatch__group_aggregate__process__node__to_unit         <- [group, group_aggregate, unit, source, sink];
table data IN 'CSV' 'solve_data/nodeGroupDispatch__process__node__to_connection_Not_in_aggregate.csv'   : nodeGroupDispatch__process__node__to_connection_Not_in_aggregate   <- [group, process, node, connection];
table data IN 'CSV' 'solve_data/nodeGroupDispatch__process__connection__to_node_Not_in_aggregate.csv'   : nodeGroupDispatch__process__connection__to_node_Not_in_aggregate   <- [group, process, connection, node];
table data IN 'CSV' 'solve_data/nodeGroupDispatch__connection_Not_in_aggregate.csv'                     : nodeGroupDispatch__connection_Not_in_aggregate                     <- [group, connection];
table data IN 'CSV' 'solve_data/nodeGroupDispatch__group_aggregate__process__connection__to_node.csv'   : nodeGroupDispatch__group_aggregate__process__connection__to_node   <- [group, group_aggregate, connection, source, sink];
table data IN 'CSV' 'solve_data/nodeGroupDispatch__group_aggregate__process__node__to_connection.csv'   : nodeGroupDispatch__group_aggregate__process__node__to_connection   <- [group, group_aggregate, connection, source, sink];
table data IN 'CSV' 'solve_data/nodeGroupDispatch__group_aggregate_Connection.csv'                      : nodeGroupDispatch__group_aggregate_Connection                      <- [group, group_aggregate];
table data IN 'CSV' 'solve_data/nodeGroupDispatch__group_aggregate_Unit_to_group.csv'                   : nodeGroupDispatch__group_aggregate_Unit_to_group                   <- [group, group_aggregate];
table data IN 'CSV' 'solve_data/nodeGroupDispatch__group_aggregate_Group_to_unit.csv'                   : nodeGroupDispatch__group_aggregate_Group_to_unit                   <- [group, group_aggregate];

param p_positive_inflow{n in node, (d,t) in dt: (n, 'no_inflow') not in node__inflow_method};  # Migrated to Python (preprocessing/entity_period_calc_params.py).
table data IN 'CSV' 'solve_data/p_positive_inflow.csv' : [node, period, time], p_positive_inflow~value;

param p_negative_inflow{n in node, (d,t) in dt};  # Migrated to Python (preprocessing/entity_period_calc_params.py).
table data IN 'CSV' 'solve_data/p_negative_inflow.csv' : [node, period, time], p_negative_inflow~value;

param p_entity_pre_existing {e in entity, d in period_in_use};  # Migrated to Python (preprocessing/entity_period_calc_params.py).
table data IN 'CSV' 'solve_data/p_entity_pre_existing.csv' : [entity, period], p_entity_pre_existing~value;
param p_entity_existing_capacity_later_solves {e in entity, d in period_in_use};  # Migrated to Python (preprocessing/entity_period_calc_params.py).
table data IN 'CSV' 'solve_data/p_entity_existing_capacity_later_solves.csv' : [entity, period], p_entity_existing_capacity_later_solves~value;

param p_entity_all_existing {e in entity, d in period_in_use};  # Migrated to Python (preprocessing/entity_period_calc_params.py).
table data IN 'CSV' 'solve_data/p_entity_all_existing.csv' : [entity, period], p_entity_all_existing~value;

param p_entity_existing_count {e in entity, d in period_in_use};  # Migrated to Python (preprocessing/entity_period_calc_params.py).
table data IN 'CSV' 'solve_data/p_entity_existing_count.csv' : [entity, period], p_entity_existing_count~value;

param p_entity_existing_integer_count {e in entity, d in period_in_use};  # Migrated to Python (preprocessing/entity_period_calc_params.py).
table data IN 'CSV' 'solve_data/p_entity_existing_integer_count.csv' : [entity, period], p_entity_existing_integer_count~value;

param p_entity_previously_invested_capacity {e in entity, d in period_in_use};  # Migrated to Python (preprocessing/entity_period_calc_params.py).
table data IN 'CSV' 'solve_data/p_entity_previously_invested_capacity.csv' : [entity, period], p_entity_previously_invested_capacity~value;

param p_entity_max_capacity {e in entity, d in period_in_use};  # Migrated to Python (preprocessing/entity_period_calc_params.py).
table data IN 'CSV' 'solve_data/p_entity_max_capacity.csv' : [entity, period], p_entity_max_capacity~value;

param p_entity_max_units {e in entity, d in period};  # Migrated to Python (preprocessing/entity_period_calc_params.py).
table data IN 'CSV' 'solve_data/p_entity_max_units.csv' : [entity, period], p_entity_max_units~value;

param p_entity_invest_cumulative_max {e in entityInvest, d in period_in_use};  # Migrated to Python (preprocessing/entity_period_calc_params.py).
table data IN 'CSV' 'solve_data/p_entity_invest_cumulative_max.csv' : [entity, period], p_entity_invest_cumulative_max~value;

# Symmetric building block for divestments: cumulative upper bound on
# v_divest summed by dispatch period d.  Currently not consumed by the
# dispatch UBs below (those take the max-alive case = zero optional
# divest); kept for parity and for future tightening via forced-divest
# floors.
param p_entity_divest_cumulative_max {e in entityDivest, d in period_in_use};  # Migrated to Python (preprocessing/entity_period_calc_params.py).
table data IN 'CSV' 'solve_data/p_entity_divest_cumulative_max.csv' : [entity, period], p_entity_divest_cumulative_max~value;

# Upper bound on alive capacity at dispatch period d (existing + cumulative
# invest ceiling).  Drives dispatch variable UBs: v_flow (via p_flow_max /
# p_flow_min), v_ramp, v_reserve, v_state, v_state_rp_start, v_online_*,
# v_startup_*, v_shutdown_*.  v_invest / v_divest keep their own per-
# invest-period scalar UB on p_entity_max_units — that is the correct
# per-tranche cap for those decision variables.
param p_entity_dispatch_capacity_max {e in entity, d in period_in_use};  # Migrated to Python (preprocessing/entity_period_calc_params.py).
table data IN 'CSV' 'solve_data/p_entity_dispatch_capacity_max.csv' : [entity, period], p_entity_dispatch_capacity_max~value;

set process_source_coeff_zero dimen 2;  # Migrated to Python (preprocessing/structural_filters.py).
set process_sink_coeff_zero dimen 2;    # Migrated to Python (preprocessing/structural_filters.py).
table data IN 'CSV' 'solve_data/process_source_coeff_zero.csv' : process_source_coeff_zero <- [process, source];
table data IN 'CSV' 'solve_data/process_sink_coeff_zero.csv' : process_sink_coeff_zero <- [process, sink];
set process_source_sink_coeff_zero dimen 3;  # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/process_source_sink_coeff_zero.csv' : process_source_sink_coeff_zero <- [process, source, sink];

param p_flow_max{(p, source, sink, d, t) in peedt};                  # Migrated to Python (preprocessing/process_arc_unions.py).
param p_flow_min{(p, source, sink, d, t) in peedt} default 0;        # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/p_flow_max.csv' : [process, source, sink, period, time], p_flow_max~value;
table data IN 'CSV' 'solve_data/p_flow_min.csv' : [process, source, sink, period, time], p_flow_min~value;

set process_VRE;  # Migrated to Python (preprocessing/process_method_sets.py).
table data IN 'CSV' 'solve_data/process_VRE.csv' : process_VRE <- [process];

param p_state_slack_share{(g,n) in group_node, (d,t) in dt: g in group_loss_share};  # Migrated to Python (preprocessing/process_arc_unions.py).
param p_storage_state_reference_price{n in nodeState, d in period_in_use};           # Migrated to Python (preprocessing/process_arc_unions.py).
table data IN 'CSV' 'solve_data/p_state_slack_share.csv'              : [group, node, period, time], p_state_slack_share~value;
table data IN 'CSV' 'solve_data/p_storage_state_reference_price.csv'  : [node, period],              p_storage_state_reference_price~value;

param d_obj default 0;
param d_flow {(p, source, sink, d, t) in peedt} default 0;
param d_flow_1_or_2_variable {(p, source, sink, d, t) in peedt} default 0;
param d_flowInvest {(p, d) in pd_invest} default 0;
param d_reserve_upDown_node {(p, r, ud, n, d, t) in prundt} default 0;
param dq_reserve {(r, ud, ng) in reserve__upDown__group, (d, t) in dt} default 0;

#########################
# Variable declarations
var v_flow {(p, source, sink, d, t) in peedt} >= p_flow_min[p, source, sink, d, t], <= p_flow_max[p, source, sink, d, t];
param p_angle_lower{n in node_dc_power_flow};  # Migrated to Python (preprocessing/dc_angle_bounds.py).
param p_angle_upper{n in node_dc_power_flow};  # Migrated to Python (preprocessing/dc_angle_bounds.py).
table data IN 'CSV' 'solve_data/p_angle_lower.csv' : [node], p_angle_lower~value;
table data IN 'CSV' 'solve_data/p_angle_upper.csv' : [node], p_angle_upper~value;
var v_angle {n in node_dc_power_flow, (d, t) in dt} >= p_angle_lower[n], <= p_angle_upper[n];
var v_ramp {(p, source, sink) in process_source_sink_ramp, (d, t) in dt} >= -p_entity_dispatch_capacity_max[p, d] / p_entity_unitsize[p], <= p_entity_dispatch_capacity_max[p, d] / p_entity_unitsize[p];
var v_reserve {(p, r, ud, n, d, t) in prundt : sum{(r, ud, g) in reserve__upDown__group} 1 } >= 0, <= p_entity_dispatch_capacity_max[p, d] / p_entity_unitsize[p];
var v_state {n in nodeState, (d, t) in dt} >= 0, <= p_entity_dispatch_capacity_max[n, d] / p_entity_unitsize[n];
# Inter-period storage state for representative period method (Paper: σ^{inter}_{s,d})
var v_state_inter {n in nodeState_rp, b in rp_base_period} >= 0;
# Starting state of each representative period (Paper: σ^{intra,0}_{s,r})
var v_state_rp_start {n in nodeState_rp, (d, t) in rp_block_first} >= 0, <= p_entity_dispatch_capacity_max[n, d] / p_entity_unitsize[n];
# Agent 1.6: UC / startup / shutdown variables live at the process's
# own block (process__block).  Declaring the variable at the process's
# block timeline means fewer columns in the coarse case and matches
# Gao's pattern where commitment is aggregated together with flow.
# Variable column index stays ``[p, d, t]`` so all cross-referencing
# constraints (maxToSink, ramp, reserve, …) keep their existing
# dereferences; the indexing set restricts ``(p, d, t)`` to the
# process's block's active timesteps.  In the degenerate case
# (process_block[p] = 'default') block__period__step at 'default'
# covers every fine (d, t), so this reduces to the pre-v51 declaration
# bit-identically.
#
# V1 limitation (see Agent 1.6 check stanzas below): UC processes are
# required to have matched source and sink blocks, i.e.
# process_block_in = process_block_out = process_block.  That keeps
# Agent 1.5's capacity-bound references to v_online_* at b_out / b_in
# valid 1-to-1 without overlap aggregation.
set p_online_dt
    'UC variable indexing: (process, period, step) at the process''s block'
    dimen 3;  # Migrated to Python (preprocessing/per_solve_sets.py).
table data IN 'CSV' 'solve_data/p_online_dt_set.csv' : p_online_dt <- [process, period, step];
var v_online_linear {(p, d, t) in p_online_dt : p in process_online_linear}
    >= 0, <= p_entity_dispatch_capacity_max[p, d] / p_entity_unitsize[p];
var v_startup_linear {(p, d, t) in p_online_dt : p in process_online_linear}
    >= 0, <= p_entity_dispatch_capacity_max[p, d] / p_entity_unitsize[p];
var v_shutdown_linear {(p, d, t) in p_online_dt : p in process_online_linear}
    >= 0, <= p_entity_dispatch_capacity_max[p, d] / p_entity_unitsize[p];
var v_online_integer {(p, d, t) in p_online_dt : p in process_online_integer}
    >= 0, <= p_entity_dispatch_capacity_max[p, d] / p_entity_unitsize[p], integer;
var v_startup_integer {(p, d, t) in p_online_dt : p in process_online_integer}
    >= 0, <= p_entity_dispatch_capacity_max[p, d] / p_entity_unitsize[p];
var v_shutdown_integer {(p, d, t) in p_online_dt : p in process_online_integer}
    >= 0, <= p_entity_dispatch_capacity_max[p, d] / p_entity_unitsize[p];
var v_invest {(e, d) in ed_invest} >= 0, <= p_entity_max_units[e, d];
var v_divest {(e, d) in ed_divest} >= 0, <= p_entity_max_units[e, d];
# Slack variables (see flextool/SLACK_CONVENTION.md).  Single-variable
# form with the user-supplied penalty itself acting as the valve that
# keeps the slack quiescent on well-posed inputs while still absorbing
# pathological inputs without false infeasibility.  Output CSV writes
# the single-variable value directly, un-scaled by the row scaler.
var vq_state_up   {n in (nodeBalance union nodeBalancePeriod), (d, t) in dt} >= 0;
var vq_state_down {n in (nodeBalance union nodeBalancePeriod), (d, t) in dt} >= 0;
var vq_reserve {(r, ud, ng) in reserve__upDown__group, (d, t) in dt} >= 0, <= 1;
var vq_inertia {g in groupInertia, (d, t) in dt} >= 0, <= 1;
var vq_non_synchronous {g in groupNonSync, (d, t) in dt} >= 0;
var vq_capacity_margin {g in groupCapacityMargin, d in period_invest} >= 0;
# vq_state_up_group has no objective term of its own: it is tied by the
# group_loss_share_constraint equality to vq_state_up (which is
# penalised in the objective), so the group slack is penalised
# indirectly through its sibling.
var vq_state_up_group {g in group_loss_share, (d,t) in dt} >= 0;

# Commodity-ladder period-level trade variable.  Column unit is
# MWh / p_commodity_unitsize[c] (matches entity-unitsize convention
# used by v_flow / v_state).  No t index and no branch index — v_trade
# is a period-level decision; stochastic branches pool into a single
# un-branched trade.  Defined only for commodities using a ladder
# price method; upper bounds are imposed by the tier-cap constraints
# below.
var v_trade {(c, n, d, i) in cndi_ladder} >= 0;

#########################
## Data checks
if p_model["solveFirst"] == 1 && 'read' in phase then {
  printf "!!! Data checks\n";
  printf 'Checking: Eff. data for 1 variable conversions directly from source to sink (and possibly back).';
  printf' Efficiency should always be !=0 \n';
  check {(p, m) in process_method, (d,t) in dt : m in method_1var && not (p, 'none') in process__ct_method } pdtProcess[p, 'efficiency', d, t] != 0 ;

  printf 'Checking: Efficiency data for 1-way conversions with an online variable.';
  printf 'Efficiency should always be !=0\n';
  check {(p, m) in process_method, (d,t) in dt : m in method_1way_on} pdtProcess[p, 'efficiency', d, t] != 0;

  printf 'Checking: Efficiency data for 2-way linear conversions without online variables.';
  printf 'Efficiency should always be !=0\n';
  check {(p, m) in process_method, (d,t) in dt : m in method_2way_off} pdtProcess[p, 'efficiency', d, t] != 0;

  printf 'Checking: Min load efficiency should be greater than zero\n';
  check {p in process_minload, (d,t) in dt} pdtProcess[p, 'efficiency_at_min_load', d, t] > 0;

  printf 'Checking: Min load should be less than 1\n';
  check {p in process_minload, (d,t) in dt} pdtProcess[p, 'min_load', d, t] < 1.0;

  printf 'Checking: Invalid combinations between conversion/transfer methods and the startup method\n';
  check {(p, m) in process_method} : m != 'not_applicable';

  printf 'Checking: Is there a timeline connected to a timeset\n';
  check sum{(tb, tl) in timeset__timeline} 1 > 0;

  printf 'Checking: Are discount factors set in models with investments and multiple periods\n';
  check {d in period_in_use : d not in period_first && (sum{(e, d) in ed_invest} 1 || sum{(e, d) in ed_divest} 1)} : p_discount_years[d] != 0;

  printf 'Checking: The nodes with scaling methods should have the inflow parameter set\n';
  check {n in node, d in period_in_use: (n, 'scale_to_annual_flow') in node__inflow_method || (n, 'scale_to_annual_and_peak_flow') in node__inflow_method ||
          (n, 'scale_to_annual_and_peak_flow') in node__inflow_method}:
    sum{(d, t) in dt_complete} ptNode_inflow[n, t] != 0;

  printf 'Checking: Availability conflicts with storage constraints\n';
  check {n in nodeState, (d,t) in period__time_first: (n, 'fix_start') in node__storage_start_end_method || (n, 'fix_start_end') in node__storage_start_end_method}:
    p_node[n,'storage_state_start'] <= pdtNode[n, 'availability', d, t];
  check {n in nodeState, (d,t) in period__time_last: (n, 'fix_end') in node__storage_start_end_method || (n, 'fix_start_end') in node__storage_start_end_method}:
    p_node[n,'storage_state_end'] <= pdtNode[n, 'availability', d, t];

  check {n in nodeState, (d,t) in (period__time_first union period__time_last): ((n, 'fix_start') in node__storage_start_end_method || (n, 'fix_start_end') in node__storage_start_end_method)
  && ((n, 'bind_within_solve') in node__storage_binding_method || (n, 'bind_within_period') in node__storage_binding_method || (n, 'bind_intraperiod_blocks') in node__storage_binding_method)}:
    p_node[n,'storage_state_start'] <= pdtNode[n, 'availability', d, t];
  check {n in nodeState, (d,t) in (period__time_first union period__time_last): ((n, 'fix_start_end') in node__storage_start_end_method || (n, 'fix_end') in node__storage_start_end_method)
  && ((n, 'bind_within_solve') in node__storage_binding_method || (n, 'bind_within_period') in node__storage_binding_method || (n, 'bind_intraperiod_blocks') in node__storage_binding_method)}:
    p_node[n,'storage_state_end'] <= pdtNode[n, 'availability', d, t];

  check {n in nodeState, (d,t,t_previous,t_previous_within_timeset,d_previous,t_previous_within_solve) in dtttdt:
  ((n, 'fix_start_end') in node__storage_start_end_method || (n, 'fix_end') in node__storage_start_end_method)
  && (n, 'bind_within_timeset') in node__storage_binding_method
  && dt_jump[d,t] != 1}:
    p_node[n,'storage_state_end'] <= pdtNode[n, 'availability', d, t] && p_node[n,'storage_state_end'] <= pdtNode[n,'availability', d, t_previous];

  check {n in nodeState, (d,t,t_previous,t_previous_within_timeset,d_previous,t_previous_within_solve) in dtttdt:
  ((n, 'fix_start') in node__storage_start_end_method || (n, 'fix_start_end') in node__storage_start_end_method)
  && (n, 'bind_within_timeset') in node__storage_binding_method
  && dt_jump[d,t] != 1}:
    (p_node[n,'storage_state_start'] <= pdtNode[n, 'availability', d, t] && p_node[n,'storage_state_start'] <= pdtNode[n,'availability', d, t_previous]);

  check {n in nodeState, (d,t) in period__time_last:
    (n, 'use_reference_value') in node__storage_solve_horizon_method
    && (n, 'fix_end') not in node__storage_start_end_method
    && (n, 'fix_start_end') not in node__storage_start_end_method
    && (n, 'bind_within_solve') not in node__storage_binding_method
    && (n, 'bind_within_period') not in node__storage_binding_method
    && (n, 'bind_within_timeset') not in node__storage_binding_method}:
    pdtNode[n,'storage_state_reference_value', d, t] <= pdtNode[n, 'availability', d, t];

  # VERY SLOW, therefore commented out for now
  # printf 'Checking: transfer_method no_losses_no_variable_cost ';
  # printf 'is not allowed to a group with non-synchronous constraint\n';
  # check {g in groupNonSync, (p,source,sink) in process_source_sink:
  #   (((p,source) in process_source && (g,source) in group_node)
  #   || ((p,sink) in process_sink && (g,sink) in group_node))
  #   && (p,g) not in process__group_inside_group_nonSync}:
  #     sum{(p, m) in process_method : m in method_2way_1var} 1 < 1;

  printf 'Checking: transfer_method no_losses_no_variable_cost ';
  printf 'is not allowed to have other_operational_cost\n';
  check {(p,m) in process_method, (d,t) in dt: m in method_2way_1var}:
    pdtProcess[p, 'other_operational_cost', d, t] = 0;

  printf 'Checking: node not in more than one loss of load sharing group\n';
  check {n in node}:
    sum{(g,n) in group_node: g in group_loss_share} 1 < 2;

  printf 'Checking: Groups with investment constraints have entities that can be invested in\n';
  check {g in group_invest, d in period_invest}:
    sum{(g,e) in group_entity: e in entityInvest } 1 > 0;

  printf'Checking: If stochastic timeseries data given, ';
  printf'the realized branch is set for the period in stochastic_branches ';
  printf'and the period start time is a branch start time is the timeseries ';
  printf'for process timeseries \n';
  check {(d,t) in period__time_first: exists{(p, param, tb, ts, t2) in process__param__branch__time} 1}:
    exists{(p, param, tb, t, t) in process__param__branch__time, (d2,tb) in solve_branch__time_branch: (d2,d) in period__branch} 1;
  printf'Checking: If stochastic timeseries data given, ';
  printf'the realized branch is set for the period in stochastic_branches ';
  printf'and the period start time is a branch start time is the timeseries ';
  printf'for process_inputNode timeseries\n';
  check {(d,t) in period__time_first: exists{(p, source, param, tb, ts, t2) in process__source__param__branch__time} 1}:
    exists{(p, source, param, tb, t, t) in process__source__param__branch__time, (d2,tb) in solve_branch__time_branch: (d2,d) in period__branch} 1;
  printf'Checking: If stochastic timeseries data given, ';
  printf'the realized branch is set for the period in stochastic_branches ';
  printf'and the period start time is a branch start time is the timeseries ';
  printf'for process_outputNode timeseries\n';
  check {(d,t) in period__time_first: exists{(p, sink, param, tb, ts, t2) in process__sink__param__branch__time} 1}:
    exists{(p, sink, param, tb, t, t) in process__sink__param__branch__time, (d2,tb) in solve_branch__time_branch: (d2,d) in period__branch} 1;
  printf'Checking: If stochastic timeseries data given, ';
  printf'the realized branch is set for the period in stochastic_branches ';
  printf'and the period start time is a branch start time is the timeseries ';
  printf'for Node inflow timeseries\n';
  check{(d,t) in period__time_first: exists{(n, tb, ts, t2) in node__branch__time_inflow} 1 }:
    exists{(n, tb, t, t) in node__branch__time_inflow, (d2,tb) in solve_branch__time_branch: (d2,d) in period__branch} 1;
  printf'Checking: If stochastic timeseries data given, ';
  printf'the realized branch is set for the period in stochastic_branches ';
  printf'and the period start time is a branch start time is the timeseries ';
  printf'for Node timeseries\n';
  check {(d,t) in period__time_first: exists{(n, param, tb, ts, t2) in node__param__branch__time} 1}:
    exists{(n, param, tb, t, t) in node__param__branch__time, (d2,tb) in solve_branch__time_branch: (d2,d) in period__branch} 1;
  printf'Checking: If stochastic timeseries data given, ';
  printf'the realized branch is set for the period in stochastic_branches ';
  printf'and the period start time is a branch start time is the timeseries ';
  printf'for profile timeseries\n';
  check {(d,t) in period__time_first: exists{(p, tb, ts, t2) in profile__branch__time} 1}:
    exists{(p, tb, t, t) in profile__branch__time, (d2,tb) in solve_branch__time_branch: (d2,d) in period__branch} 1;
  printf'Checking: If stochastic timeseries data given, ';
  printf'the realized branch is set for the period in stochastic_branches ';
  printf'and the period start time is a branch start time is the timeseries ';
  printf'for reserve timeseries\n';
  check {(d,t) in period__time_first: exists{(r, ud, g, param, tb, ts, t2) in reserve__upDown__group__reserveParam__branch__time} 1}:
    exists{(r, ud, g, param, tb, t, t) in reserve__upDown__group__reserveParam__branch__time, (d2,tb) in solve_branch__time_branch: (d2,d) in period__branch} 1;
  printf'Checking: Existing capacity is less than cumulative_max_capacity\n';
  check {(e, d) in ed_invest_cumulative}:
    p_entity_all_existing[e, d] <= ed_cumulative_max_capacity[e, d];
  printf 'Checking: Delayed flows must be one-way:  ';
  check {p in process_delayed} sum{(p, m) in process_method : m in method_1way} 1 > 0;

  # Agent 1.6 V1 limitation: unit-commitment processes (v_online_* present)
  # must have their source and sink flows on the same temporal block as the
  # process's unified block.  The cross-referencing capacity / ramp / reserve
  # constraints from Agents 1.5 / 1.6 read v_online_*[p, d, t] at the side's
  # block timesteps — if the process sits at a finer block than one of the
  # sides, the fine-grid online value has no 1-to-1 counterpart at the
  # coarser side and overlap aggregation would be required.  V1 forbids this
  # configuration outright.  In the degenerate case (everything on
  # 'default') the check is trivially satisfied.
  printf 'Checking: UC (online) processes must have matched source/sink blocks (Agent 1.6 V1 limitation)\n';
  check {p in process_online, (p, 'source', b_in) in process__side__block,
         (p, 'sink', b_out) in process__side__block,
         (p, b_proc) in process__block} :
    b_in == b_proc && b_out == b_proc;

  # -----------------------------------------------------------------------
  # Agent 1.7: structural block-invariant checks.
  #
  # In the degenerate ('default'-only) case every check below is trivially
  # satisfied because node__block / process__block / process__side__block
  # each carry exactly one row per entity at 'default'.  These guard the
  # CSVs produced by flextool/flextoolrunner/blocks.py against silent
  # corruption (e.g. a hand-edited solve_data/entity_block.csv) and stop
  # the LP build before the block-aware sums misbehave.
  # -----------------------------------------------------------------------

  # 1. Every node must appear in node__block exactly once.
  printf 'Checking: Every node is in exactly one block (node__block)\n';
  check {n in node} : sum{(n, b) in node__block} 1 == 1;

  # 2a. Every process must appear in process__block exactly once.
  printf 'Checking: Every process is in exactly one unified block (process__block)\n';
  check {p in process} : sum{(p, b) in process__block} 1 == 1;

  # 2b. Every process must have exactly one source-side and one sink-side
  # entry in process__side__block.
  printf 'Checking: Every process has a source-side block (process__side__block)\n';
  check {p in process} : sum{(p, 'source', b) in process__side__block} 1 == 1;
  printf 'Checking: Every process has a sink-side block (process__side__block)\n';
  check {p in process} : sum{(p, 'sink', b) in process__side__block} 1 == 1;

  # 3. block_step_duration must be strictly positive everywhere it's
  # declared.  The parameter is declared over block__period__step so this
  # loop covers every used (block, period, step) triple.
  printf 'Checking: block_step_duration is strictly positive\n';
  check {(b, d, t) in block__period__step} : block_step_duration[b, d, t] > 0;

  # 4. p_overlap must be in [0, 1].  Rows not present default to 0 so we
  # only need to bound the explicit entries.
  printf 'Checking: p_overlap is in [0, 1]\n';
  check {(d, bc, tc, bf, tf) in overlap} :
    p_overlap[d, bc, tc, bf, tf] >= 0 && p_overlap[d, bc, tc, bf, tf] <= 1;

  # -----------------------------------------------------------------------
  # Agent 1.7 reserve-block compatibility (V1).
  #
  # Reserves are intrinsically short-term (intra-hour in practice).
  # Mixing reserves with coarse-resolution participants creates subtle
  # semantic issues around energy vs. power aggregation.  V1 requires
  # every reserve participant (reserve-group member nodes + reserve-
  # participating processes) to sit on the default block so the reserve
  # balance / reserve_process constraints can stay at the fine ``dt``
  # index without overlap-aggregation gymnastics.  The Python-side
  # validate_group_membership enforces the same rule at input-write time;
  # duplicating the check here guarantees a solve-time failure even when
  # the block CSVs were hand-edited.
  # -----------------------------------------------------------------------
  printf 'Checking: reserve-group member nodes must sit on the default block (Agent 1.7 V1)\n';
  check {(r, ud, g) in reserve__upDown__group, (g, n) in group_node,
         (n, b) in node__block} : b == 'default';
  printf 'Checking: reserve-participating processes must sit on the default block (Agent 1.7 V1)\n';
  check {(p, r, ud, n) in process_reserve_upDown_node,
         (p, b) in process__block} : b == 'default';
  printf 'Checking: reserve-participating process nodes must sit on the default block (Agent 1.7 V1)\n';
  check {(p, r, ud, n) in process_reserve_upDown_node,
         (n, b) in node__block} : b == 'default';
}

param setup := gmtime();
printf 'Timer - Setup: %ss\n', setup - datetime0;
# Per-solve mod-phase timings — truncated on each invocation (single
# glpsol process per solve), so the file always reflects the current
# solve.  orchestration.py reads it after solver.run() and feeds the
# rows into the unified TimingRecorder (timings.csv).
param mod_phases_file symbolic := 'solve_data/mod_phases.csv';
printf 'phase,seconds\n' > mod_phases_file;
printf '%s,%s\n', 'setup', setup - datetime0 >> mod_phases_file;

printf("Constraint generation:\n");

minimize total_cost:
( + sum {(c, n) in commodity_node, (d, t) in dt : c not in commodity_with_ladder}
    (+ pdtCommodity[c, 'price', d, t]
	  * (
		  # Buying a commodity (increases the objective function)
		  + sum {(p, n, sink) in process_source_sink_noEff }
			( + v_flow[p, n, sink, d, t] * p_entity_unitsize[p] )
		  + sum {(p, n, sink) in process_source_sink_eff } (
			  + v_flow[p, n, sink, d, t] * p_entity_unitsize[p]
				  * pdtProcess_slope[p, d, t]
				  * (if p in process_unit then p_process_sink_flow_coefficient[p, sink] / p_process_source_flow_coefficient[p, n] else 1)
			  + (if (p, 'min_load_efficiency') in process__ct_method then
				  + ( + (if p in process_online_linear then v_online_linear[p, d, t])
				      + (if p in process_online_integer then v_online_integer[p, d, t])
					)
					* pdtProcess_section[p, d, t]
					* p_entity_unitsize[p]
				)
			)
		  # Selling to a commodity node (decreases objective function if price is positive)
		  - sum {(p, source, n) in process_source_sink } (
			  + v_flow[p, source, n, d, t] * p_entity_unitsize[p]
			)
		)
	  * step_duration[d, t] * p_rp_cost_weight[d, t] * p_inflation_factor_operations_yearly[d] / complete_period_share_of_year[d] * pdt_branch_weight[d,t]
	)
  # Commodity-ladder price term.  v_trade * p_commodity_unitsize is the
  # branch-weighted realized-timeline MWh purchased in period d (set by
  # commodity_ladder_balance below), so multiplying by
  # p_inflation_factor_operations_yearly / complete_period_share_of_year
  # matches the pdtCommodity annualization exactly: when a single-tier
  # ladder uses price_method='price_ladder_annual' with the same price as
  # pdtCommodity['price'] and unitsize=1, the objective contribution is
  # bit-identical to the legacy term.
  + sum {(c, n, d, i) in cndi_ladder_cum}
      ( + p_ladder_cum_price[c, i]
          * v_trade[c, n, d, i]
          * p_commodity_unitsize[c]
          * p_inflation_factor_operations_yearly[d] / complete_period_share_of_year[d]
      )
  + sum {(c, n, d, i) in cndi_ladder_ann}
      ( + p_ladder_ann_price[c, i, d]
          * v_trade[c, n, d, i]
          * p_commodity_unitsize[c]
          * p_inflation_factor_operations_yearly[d] / complete_period_share_of_year[d]
      )
  + sum {(g, c, n, d, t) in gcndt_co2_price}
    (+ p_commodity[c, 'co2_content'] * pdtGroup[g, 'co2_price', d, t]
	  * (
		  # Paying for CO2 (increases the objective function)
		  + sum {(p, n, sink) in process_source_sink_noEff }
			( + v_flow[p, n, sink, d, t] * p_entity_unitsize[p] )
		  + sum {(p, n, sink) in process_source_sink_eff } (
			  + v_flow[p, n, sink, d, t] * p_entity_unitsize[p]
				  * pdtProcess_slope[p, d, t]
				  * (if p in process_unit then p_process_sink_flow_coefficient[p, sink] / p_process_source_flow_coefficient[p, n] else 1)
			  + (if (p, 'min_load_efficiency') in process__ct_method then
				  + ( + (if p in process_online_linear then v_online_linear[p, d, t])
				      + (if p in process_online_integer then v_online_integer[p, d, t])
					)
					* pdtProcess_section[p, d, t]
					* p_entity_unitsize[p]
				)
			)
		  # Receiving credits for removing CO2 (decreases the objective function)
		  - sum {(p, source, n) in process_source_sink } (
			  + v_flow[p, source, n, d, t] * p_entity_unitsize[p]
			)
	    )
	  * step_duration[d, t] * p_rp_cost_weight[d, t] * p_inflation_factor_operations_yearly[d] / complete_period_share_of_year[d] * pdt_branch_weight[d,t]
	)
  + sum {(p, d, t) in pdt_online_linear}
      ( + v_startup_linear[p, d, t] * pdProcess[p, 'startup_cost', d]
	      * p_entity_unitsize[p]
		  * p_rp_cost_weight[d, t] * p_inflation_factor_operations_yearly[d] / complete_period_share_of_year[d] * pdt_branch_weight[d,t]
	  )
  + sum {(p, d, t) in pdt_online_integer}
      ( + v_startup_integer[p, d, t] * pdProcess[p, 'startup_cost', d]
	      * p_entity_unitsize[p]
		  * p_rp_cost_weight[d, t] * p_inflation_factor_operations_yearly[d] / complete_period_share_of_year[d] * pdt_branch_weight[d,t]
	  )
  + sum {(p, source, sink, d, t) in pssdt_varCost_noEff}
    ( + pdtProcess__source__sink__dt_varCost[p, source, sink, d, t]
	    * v_flow[p, source, sink, d, t] * p_entity_unitsize[p]
        * step_duration[d, t] * p_rp_cost_weight[d, t] * p_inflation_factor_operations_yearly[d] / complete_period_share_of_year[d] * pdt_branch_weight[d,t]
	)
  + sum {(p, source, sink, d, t) in pssdt_varCost_eff_unit_source}
    ( - pdtProcess_source[p, source, 'other_operational_cost', d, t]
	    *
	    ( + v_flow[p, source, sink, d, t] * p_entity_unitsize[p]
		    * pdtProcess_slope[p, d, t]
			* (if p in process_unit then p_process_sink_flow_coefficient[p, sink] / p_process_source_flow_coefficient[p, source] else 1)
          + ( if (p, 'min_load_efficiency') in process__ct_method then
	          + ( + (if p in process_online_linear then v_online_linear[p, d, t])
			      + (if p in process_online_integer then v_online_integer[p, d, t])
			    )
			    * pdtProcess_section[p, d, t]
			    * p_entity_unitsize[p]
			)
		)
        * step_duration[d, t] * p_rp_cost_weight[d, t] * p_inflation_factor_operations_yearly[d] / complete_period_share_of_year[d] * pdt_branch_weight[d,t]
	)
  + sum {(p, source, sink, d, t) in pssdt_varCost_eff_unit_sink}
    ( + pdtProcess_sink[p, sink, 'other_operational_cost', d, t]
	    * v_flow[p, source, sink, d, t] * p_entity_unitsize[p]
        * step_duration[d, t] * p_rp_cost_weight[d, t] * p_inflation_factor_operations_yearly[d] / complete_period_share_of_year[d] * pdt_branch_weight[d,t]
	)
  + sum {(p, source, sink, d, t) in pssdt_varCost_eff_connection}
    ( + pdtProcess[p, 'other_operational_cost', d, t]
 	   * v_flow[p, source, sink, d, t] * p_entity_unitsize[p]
       * step_duration[d, t] * p_rp_cost_weight[d, t] * p_inflation_factor_operations_yearly[d] / complete_period_share_of_year[d] * pdt_branch_weight[d,t]
	)
#  + sum {(p, source, sink, m) in process__source__sink__ramp_method, (d, t) in dt : m in ramp_cost_method}
#    ( + v_ramp[p, source, sink, d, t] * p_entity_unitsize[p] * pProcess_source_sink[p, source, sink, 'ramp_cost'] ) * step_duration[d, t] * p_inflation_factor_operations_yearly[d] / complete_period_share_of_year[d]
  + sum {g in groupInertia, (d, t) in dt} pdt_branch_weight[d,t] * vq_inertia[g, d, t] * pdGroup[g, 'inertia_limit', d]
                                            * pdGroup[g, 'penalty_inertia', d] * step_duration[d, t] * p_rp_cost_weight[d, t] * p_inflation_factor_operations_yearly[d] / complete_period_share_of_year[d]
  + sum {g in groupNonSync, (d, t) in dt} pdt_branch_weight[d,t] * vq_non_synchronous[g, d, t] * group_capacity_for_scaling[g, d]
                                            * pdGroup[g, 'penalty_non_synchronous', d] * step_duration[d, t] * p_rp_cost_weight[d, t] * p_inflation_factor_operations_yearly[d] / complete_period_share_of_year[d]
  + sum {n in nodeBalance union nodeBalancePeriod, (d, t) in dt} pdt_branch_weight[d,t] * vq_state_up[n, d, t] * node_capacity_for_scaling[n, d]
                                            * pdtNode[n, 'penalty_up', d, t] * step_duration[d, t] * p_rp_cost_weight[d, t] * p_inflation_factor_operations_yearly[d] / complete_period_share_of_year[d]
  + sum {n in nodeBalance union nodeBalancePeriod, (d, t) in dt} pdt_branch_weight[d,t] * vq_state_down[n, d, t] * node_capacity_for_scaling[n, d]
                                            * pdtNode[n, 'penalty_down', d, t] * step_duration[d, t] * p_rp_cost_weight[d, t] * p_inflation_factor_operations_yearly[d] / complete_period_share_of_year[d]
  + sum {(r, ud, ng) in reserve__upDown__group, (d, t) in dt} pdt_branch_weight[d,t] * vq_reserve[r, ud, ng, d, t]  * pdtReserve_upDown_group[r, ud, ng, 'reservation', d, t]
                                            * p_reserve_upDown_group[r, ud, ng, 'penalty_reserve'] * step_duration[d, t] * p_rp_cost_weight[d, t] * p_inflation_factor_operations_yearly[d] / complete_period_share_of_year[d]
  - sum {n in nodeState, (d, t) in period__time_last : (n, 'use_reference_price') in node__storage_solve_horizon_method && d in period_last}
    (+ p_storage_state_reference_price[n,d]
        * v_state[n, d, t] * p_entity_unitsize[n]
		 * p_rp_cost_weight[d, t] * p_inflation_factor_operations_yearly[d] / complete_period_share_of_year[d] * pdt_branch_weight[d,t]
    )
  + sum {e in entity, d in period_in_use}  # This is constant term and will be dropped by the solver. Here for completeness.
    + p_entity_all_existing[e, d]
      * ed_fixed_cost[e, d]
	  * p_inflation_factor_operations_yearly[d] * pd_branch_weight[d]
  + sum {(e, d) in ed_invest}
    # Currently investment happens only on the realized branch and the rest get them as existing.
    # Only one period investment is supported with stochastics
    # The branch weight should be added if this is changed.
      + v_invest[e, d]
        * p_entity_unitsize[e]
        * ed_entity_annual_discounted[e, d]
  + sum {(e, d) in ed_invest}
      + v_invest[e, d]
          * p_entity_unitsize[e]
          * ed_lifetime_fixed_cost[e, d]  # This includes all years until end of lifetime, inflation adjusted
  - sum {(e, d) in ed_divest}
      + v_divest[e, d]
          * p_entity_unitsize[e]
          * ed_lifetime_fixed_cost_divest[e, d]
  # Salvage / retirement value of divested capacity.  Variable in v_divest,
  # so the solver must see it when choosing whether to divest.  The sign of
  # ed_entity_annual_divest_discounted follows salvage_value: positive
  # salvage (scrap value) reduces the objective; negative salvage
  # (decommissioning cost) increases it.  Currently investment/divest is
  # not branched, so no pdt_branch_weight factor — see also the comment on
  # the invest term above.
  - sum {(e, d) in ed_divest}
      + v_divest[e, d]
          * p_entity_unitsize[e]
          * ed_entity_annual_divest_discounted[e, d]
  + sum {g in groupCapacityMargin, d in period_invest}
    + vq_capacity_margin[g, d] * group_capacity_for_scaling[g, d]
	  * pdGroup[g, 'penalty_capacity_margin', d]
	  * p_inflation_factor_operations_yearly[d]
) * scale_the_objective
;
param total_obj_cost := gmtime();
printf 'Timer - Objective: %ss\n', total_obj_cost - setup;
printf '%s,%s\n', 'total_obj_cost', total_obj_cost - setup >> mod_phases_file;

# ---------------------------------------------------------------------------
# Generalized node balance equations (Agent 1.3 — M-matrix / overlap-set
# aggregation, per Gao & Morales-España 2025).
#
# Each node n lives on its own temporal-resolution block b_n = node__block[n].
# Each process p has a per-side block b_f = process__side__block[p, side]
# which may differ from b_n when an indirect-conversion process straddles a
# fine-resolution node and a coarse-resolution node (e.g. hourly elec + daily
# H2).  The generalized balance at node n, period d, node-timestep t_n
# aggregates every contributing flow v_flow[...,d,t_f] via the overlap
# fraction M_{b_n,t_n;b_f,t_f} = p_overlap[d, b_n, t_n, b_f, t_f] and weights
# it by the flow-side stepduration block_step_duration[b_f, d, t_f].  In the
# degenerate single-block case (every node and every process-side mapped to
# "default"), overlap carries identity rows (1.0 on the diagonal, 0
# elsewhere) and block_step_duration['default', d, t] == step_duration[d, t]
# (same float value in the CSV), so each block-aware sum collapses to its
# pre-v51 form and the LP is bit-identical to Agent 1.2's baseline.
#
# For non-state terms the time index on the LHS (t_n) is the node's own
# block's time; process-flow variables are keyed at (d, t_f) on the process-
# side block.  Agent 1.4 generalises state-transition predecessor logic to
# use block_dtttdt (tagged by the node's block b_n) instead of the
# fine-timeline dtttdt: in the degenerate case ('default'-only) the tagged
# rows equal dtttdt exactly so the LP stays bit-identical; for a node sitting
# on a coarser block (daily storage on an hourly electricity network) the
# state transitions, self-discharge and inflow terms all evaluate at the
# coarser grid.  Agent 5c row-scaling (inv_node_cap[n, d]) is preserved on
# every RHS term.
# ---------------------------------------------------------------------------

# Energy balance in each node
# Agent 5c: every term on LHS and RHS is multiplied by inv_node_cap[n, d]
# (row scaler).  For slack terms the pre-multiplier
# `* node_capacity_for_scaling[n, d]` cancels with `* inv_node_cap[n, d]`,
# collapsing the slack column coefficient to `step_duration`.  For every
# other term this introduces a `/ node_cap` factor that compresses the
# matrix coefficient range in Mode B.  In Mode A both scalers are 1 so
# inv_node_cap = 1 and this is a structural no-op.
s.t. nodeBalance_eq {c in solve_current, n in nodeBalance, (n, bn) in node__block,
    (bn, d, t, t_previous, t_previous_within_timeset, d_previous, t_previous_within_solve) in block_dtttdt
    : n not in nodeStateBlock} :
  + (if n in nodeState && (n, 'bind_forward_only') in node__storage_binding_method && not ((bn, d, t) in block__period__time_first && d in period_first_of_solve) then (v_state[n, d, t] -  v_state[n, d_previous, t_previous_within_solve]) * p_entity_unitsize[n] / p_hole_multiplier[c] * inv_node_cap[n, d] )
  + (if n in nodeState && (n, 'bind_within_solve') in node__storage_binding_method && (n, 'fix_start_end') not in node__storage_start_end_method then (v_state[n, d, t] -  v_state[n, d_previous, t_previous_within_solve]) * p_entity_unitsize[n]  / p_hole_multiplier[c] * inv_node_cap[n, d] )
  + (if n in nodeState && (n, 'bind_within_period') in node__storage_binding_method && (n, 'fix_start_end') not in node__storage_start_end_method then (v_state[n, d, t] -  v_state[n, d, t_previous]) * p_entity_unitsize[n]  / p_hole_multiplier[c] * inv_node_cap[n, d] )
  + (if n in nodeState && (n, 'bind_within_timeset') in node__storage_binding_method && (n, 'fix_start_end') not in node__storage_start_end_method then (v_state[n, d, t] -  v_state[n, d, t_previous_within_timeset]) * p_entity_unitsize[n] * inv_node_cap[n, d] )
  # bind_using_blended_weights: within RP, NOT at first timestep — standard intra-period state tracking
  + (if n in nodeState_rp && not ((d, t) in rp_block_first) then (v_state[n, d, t] - v_state[n, d, t_previous_within_timeset]) * p_entity_unitsize[n] * inv_node_cap[n, d] )
  # bind_using_blended_weights: at first timestep of RP — state change from free starting variable
  + (if n in nodeState_rp && (d, t) in rp_block_first then (v_state[n, d, t] - v_state_rp_start[n, d, t]) * p_entity_unitsize[n] * inv_node_cap[n, d] )
  + (if n in nodeState && (bn, d, t) in block__period__time_first && d in period_first_of_solve && not p_nested_model['solveFirst'] then (v_state[n,d,t] * p_entity_unitsize[n] - p_roll_continue_state[n]) * inv_node_cap[n, d])
  + (if n in nodeState && (n, 'bind_forward_only') in node__storage_binding_method && (bn, d, t) in block__period__time_first && d in period_first_of_solve && p_nested_model['solveFirst']
    && ((n, 'fix_start') in node__storage_start_end_method || (n, 'fix_start_end') in node__storage_start_end_method)
    then (+ v_state[n,d,t] * p_entity_unitsize[n] - p_node[n,'storage_state_start'] *
          (+ p_entity_all_existing[n, d]
          + sum {(n, d_invest, d) in edd_invest} v_invest[n, d_invest] * p_entity_unitsize[n]
          - sum {(n, d_divest) in pd_divest : p_years_d[d_divest] <= p_years_d[d]} v_divest[n, d_divest] * p_entity_unitsize[n]
	  )) * inv_node_cap[n, d])
  =
  # n is sink — block-aware aggregation.  For each incoming flow, sum over
  # the flow-side's timesteps t_f that overlap the node's timestep t at
  # node-block b_n (degenerate: single identity row at t_f = t, fraction 1.0).
  + sum {(p, source, n) in process_source_sink, (n, b_n) in node__block,
         (p, 'sink', b_f) in process__side__block,
         (d, b_n, t, b_f, t_f) in overlap} (
      + p_overlap[d, b_n, t, b_f, t_f]
        * v_flow[p, source, n, d, t_f] * p_entity_unitsize[p]
        * block_step_duration[b_f, d, t_f] * inv_node_cap[n, d]
	)
# It would be nice to have single variable delay, but not yet implemented (it would need post-processing and fixing the term below)
#  + sum {(p, source, n) in process_source_sink_delayed} (
#      + sum {(d, t, t_, td) in dtt__delay_duration : p_process_delay_weight[p, td]}
#        ( + v_flow[p, source, n, d, t_] * p_entity_unitsize[p] * step_duration[d, t_]
#  	          * p_process_delay_weight[p, td]
#        )
#    )
  # n is source (with efficiency conversion) — block-aware.
  - sum {(p, n, sink) in process_source_sink_eff, (n, b_n) in node__block,
         (p, 'source', b_f) in process__side__block,
         (d, b_n, t, b_f, t_f) in overlap} (
      ( + v_flow[p, n, sink, d, t_f] * p_entity_unitsize[p]
            * pdtProcess_slope[p, d, t_f]
            * (if p in process_unit then p_process_sink_flow_coefficient[p, sink] / p_process_source_flow_coefficient[p, n] else 1)
        + (if (p, 'min_load_efficiency') in process__ct_method then
                + ( + (if p in process_online_linear then v_online_linear[p, d, t_f])
                    + (if p in process_online_integer then v_online_integer[p, d, t_f])
                  )
                  * pdtProcess_section[p, d, t_f]
                  * p_entity_unitsize[p]
          )
      ) * p_overlap[d, b_n, t, b_f, t_f]
        * block_step_duration[b_f, d, t_f] * inv_node_cap[n, d]
    )
  # n is source (no efficiency) — block-aware.
  - sum {(p, n, sink) in process_source_sink_noEff, (n, b_n) in node__block,
         (p, 'source', b_f) in process__side__block,
         (d, b_n, t, b_f, t_f) in overlap}
    ( + v_flow[p, n, sink, d, t_f] * p_entity_unitsize[p]
        * p_overlap[d, b_n, t, b_f, t_f]
        * block_step_duration[b_f, d, t_f] * inv_node_cap[n, d]
    )
  # Inflow — aggregated from the default ('fine') block to the node's
  # block bn via the overlap set.  pdtNodeInflow is in energy units per
  # step (not power) so the sum collects fine-block energy contributions
  # without a step_duration factor.  In the degenerate case (bn =
  # 'default') overlap carries a single identity row (t_f = t, fraction
  # 1.0) so the sum reduces to the original single term.
  + (if (n, 'no_inflow') not in node__inflow_method then
      sum {(d, bn, t, 'default', t_f) in overlap}
        p_overlap[d, bn, t, 'default', t_f]
          * pdtNodeInflow[n, d, t_f] * inv_node_cap[n, d])
  # Self-discharge at the node's block — block_step_duration[bn, d, t]
  # replaces step_duration[d, t] so a coarser node experiences a single
  # coarse-step decay per row.  In the degenerate case the two values are
  # identical.
  - (if n in nodeSelfDischarge then
      + v_state[n, d, t]
	    * (-1 + (1 + pdtNode[n, 'self_discharge_loss', d, t]) ** block_step_duration[bn, d, t])
		  * p_entity_unitsize[n] * inv_node_cap[n, d]
    )
  + vq_state_up[n, d, t] * block_step_duration[bn, d, t]
  - vq_state_down[n, d, t] * block_step_duration[bn, d, t]
;

# Energy balance within period in each node
# Agent 5c: every term scaled by inv_node_cap[n, d] (see nodeBalance_eq note).
# Block-aware (Agent 1.3): the period-level balance sums every flow
# contribution through overlap × block_step_duration[b_f, d, t_f] so that a
# process on a finer/coarser block than the node still contributes the
# correct energy volume to the period total.  In the degenerate case this
# reduces to summing v_flow * step_duration over every (d, t) in dt.
s.t. nodeBalancePeriod_eq {c in solve_current, n in nodeBalancePeriod, d in period_in_use : n not in nodeState} :
  0
  =
  # n is sink — aggregate across every coarse-row × fine-row overlap in d.
  + sum {(p, source, n) in process_source_sink, (n, b_n) in node__block,
         (p, 'sink', b_f) in process__side__block,
         (d, b_n, tc, b_f, t_f) in overlap} (
      + p_overlap[d, b_n, tc, b_f, t_f]
        * v_flow[p, source, n, d, t_f] * p_entity_unitsize[p]
        * block_step_duration[b_f, d, t_f] * inv_node_cap[n, d]
	)
  # n is source (with efficiency conversion).
  - sum {(p, n, sink) in process_source_sink_eff, (n, b_n) in node__block,
         (p, 'source', b_f) in process__side__block,
         (d, b_n, tc, b_f, t_f) in overlap} (
      ( + v_flow[p, n, sink, d, t_f] * p_entity_unitsize[p]
            * pdtProcess_slope[p, d, t_f]
            * (if p in process_unit then p_process_sink_flow_coefficient[p, sink] / p_process_source_flow_coefficient[p, n] else 1)
        + (if (p, 'min_load_efficiency') in process__ct_method then
                + ( + (if p in process_online_linear then v_online_linear[p, d, t_f])
                    + (if p in process_online_integer then v_online_integer[p, d, t_f])
                  )
                  * pdtProcess_section[p, d, t_f]
                  * p_entity_unitsize[p]
          )
      ) * p_overlap[d, b_n, tc, b_f, t_f]
        * block_step_duration[b_f, d, t_f] * inv_node_cap[n, d]
    )
  # n is source (no efficiency).
  - sum {(p, n, sink) in process_source_sink_noEff, (n, b_n) in node__block,
         (p, 'source', b_f) in process__side__block,
         (d, b_n, tc, b_f, t_f) in overlap}
    ( + v_flow[p, n, sink, d, t_f] * p_entity_unitsize[p]
        * p_overlap[d, b_n, tc, b_f, t_f]
        * block_step_duration[b_f, d, t_f] * inv_node_cap[n, d]
    )
  + (if (n, 'no_inflow') not in node__inflow_method then sum{(d, t) in dt} pdtNodeInflow[n, d, t] * inv_node_cap[n, d])
  + sum {(d, t) in dt} vq_state_up[n, d, t] * step_duration[d, t]
  - sum {(d, t) in dt} vq_state_down[n, d, t] * step_duration[d, t]
;

# Within-block constancy of v_state for nodeStateBlock (bind_intraperiod_blocks).
# In the step_previous relation, rows with t_previous_within_timeset = t_previous
# are interior-of-block rows (jump=1). State is pinned constant across them; the
# nodeBalanceBlock_eq constraint below handles the per-block state transitions.
s.t. stateConstantWithinBlock_eq {n in nodeStateBlock,
    (d, t, t_previous, t_previous_within_timeset, d_previous, t_previous_within_solve) in dtttdt
    : t_previous_within_timeset = t_previous} :
  v_state[n, d, t] = v_state[n, d, t_previous]
;

# Per-block energy balance for nodeStateBlock: state transition over the block
# equals the net flow summed over the block's timesteps. Cyclic within period
# via period_block_succ.
# Agent 5c: every term scaled by inv_node_cap[n, d] (see nodeBalance_eq note).
# Block-aware (Agent 1.3): flows are aggregated through the M-matrix /
# overlap-set so that a process at a different temporal-resolution block than
# the node still contributes the correct energy volume to the intraperiod
# block's balance.  In the degenerate case this reduces to the pre-v51 sum
# over `period_block_time` at `step_duration[d, t]`.  The intraperiod-block
# grouping (period_block_time / period_block_succ) here refers to the
# `bind_intraperiod_blocks` state-transition grouping — this is **orthogonal**
# to the temporal-resolution block (node__block / process__side__block)
# plumbing used by the flow aggregation below.  To disambiguate we keep
# `b_first` (intraperiod-block label) distinct from `b_n` (temporal-
# resolution block of node n) and `b_f` (temporal-resolution block of the
# process-side flow variable).
s.t. nodeBalanceBlock_eq {c in solve_current, n in nodeStateBlock,
    (d, b_first) in period_block
    : (n, 'fix_start_end') not in node__storage_start_end_method} :
  sum {(d, b_first, b_next) in period_block_succ}
    (v_state[n, d, b_next] - v_state[n, d, b_first]) * p_entity_unitsize[n] / p_hole_multiplier[c] * inv_node_cap[n, d]
  =
  # n is sink: flows into n — aggregate each flow contributor via the
  # overlap set, keyed at the node-block timesteps that fall within this
  # intraperiod block b_first.
  + sum {(p, source, n) in process_source_sink, (n, b_n) in node__block,
         (p, 'sink', b_f) in process__side__block,
         (d, b_first, t) in period_block_time,
         (d, b_n, t, b_f, t_f) in overlap} (
      + p_overlap[d, b_n, t, b_f, t_f]
        * v_flow[p, source, n, d, t_f] * p_entity_unitsize[p]
        * block_step_duration[b_f, d, t_f] * inv_node_cap[n, d]
    )
  # n is source: flows out via efficiency
  - sum {(p, n, sink) in process_source_sink_eff, (n, b_n) in node__block,
         (p, 'source', b_f) in process__side__block,
         (d, b_first, t) in period_block_time,
         (d, b_n, t, b_f, t_f) in overlap} (
      ( + v_flow[p, n, sink, d, t_f] * p_entity_unitsize[p]
            * pdtProcess_slope[p, d, t_f]
            * (if p in process_unit then p_process_sink_flow_coefficient[p, sink] / p_process_source_flow_coefficient[p, n] else 1)
        + (if (p, 'min_load_efficiency') in process__ct_method then
            ( + (if p in process_online_linear then v_online_linear[p, d, t_f])
              + (if p in process_online_integer then v_online_integer[p, d, t_f])
            )
            * pdtProcess_section[p, d, t_f] * p_entity_unitsize[p]
          )
      ) * p_overlap[d, b_n, t, b_f, t_f]
        * block_step_duration[b_f, d, t_f] * inv_node_cap[n, d]
    )
  # n is source: flows out, no efficiency conversion
  - sum {(p, n, sink) in process_source_sink_noEff, (n, b_n) in node__block,
         (p, 'source', b_f) in process__side__block,
         (d, b_first, t) in period_block_time,
         (d, b_n, t, b_f, t_f) in overlap} (
      + v_flow[p, n, sink, d, t_f] * p_entity_unitsize[p]
        * p_overlap[d, b_n, t, b_f, t_f]
        * block_step_duration[b_f, d, t_f] * inv_node_cap[n, d]
    )
  # Inflow (pdtNodeInflow is already in energy units per step, not power)
  + (if (n, 'no_inflow') not in node__inflow_method then
      sum {(d, b_first, t) in period_block_time} pdtNodeInflow[n, d, t] * inv_node_cap[n, d])
  # Self-discharge
  - (if n in nodeSelfDischarge then
      sum {(d, b_first, t) in period_block_time}
        v_state[n, d, t]
          * (-1 + (1 + pdtNode[n, 'self_discharge_loss', d, t]) ** step_duration[d, t])
          * p_entity_unitsize[n] * inv_node_cap[n, d]
    )
  # Penalty slacks (still per timestep so the solver can locate infeasibilities)
  + sum {(d, b_first, t) in period_block_time}
      vq_state_up[n, d, t] * step_duration[d, t]
  - sum {(d, b_first, t) in period_block_time}
      vq_state_down[n, d, t] * step_duration[d, t]
;

param balance := gmtime();
printf 'Timer - Balance: %ss\n', balance - total_obj_cost;
printf '%s,%s\n', 'balance', balance - total_obj_cost >> mod_phases_file;

s.t. reserveBalance_timeseries_eq {(r, ud, ng, r_m) in reserve__upDown__group__method_timeseries, (d, t) in dt} :
  + sum {(p, r, ud, n) in process_reserve_upDown_node_active
	      : ( sum{(p, m) in process_method : m not in method_1var_per_way} 1   ## not 1var_per_way and source; not 1var_per_way and sink; 1var_per_way and sink
		        || (p, n) in process_sink
			)
		    && (ng, n) in group_node
		    && (r, ud, ng) in reserve__upDown__group}
	    ( v_reserve[p, r, ud, n, d, t] * p_entity_unitsize[p]
	      * p_process_reserve_upDown_node_reliability[p, r, ud, n]
	    )
  + sum {(p, r, ud, n) in process_reserve_upDown_node_active                   ## 1var_per_way and source
		  : ( sum{(p, m) in process_method : m in method_1var_per_way} 1
		        && (p, n) in process_source
			)
		    && (ng, n) in group_node
		    && (r, ud, ng) in reserve__upDown__group}
	    ( v_reserve[p, r, ud, n, d, t] * p_entity_unitsize[p]
	      * p_process_reserve_upDown_node_reliability[p, r, ud, n]
	      * pdtProcess_slope[p, d, t]
		)
  + vq_reserve[r, ud, ng, d, t] * pdtReserve_upDown_group[r, ud, ng, 'reservation', d, t]
  >=
  + pdtReserve_upDown_group[r, ud, ng, 'reservation', d, t]
;

s.t. reserveBalance_dynamic_eq{(r, ud, ng, r_m) in reserve__upDown__group__method_dynamic, (d, t) in dt} :
  + sum {(p, r, ud, n) in process_reserve_upDown_node_active
	      : ( sum{(p, m) in process_method : m not in method_1var_per_way} 1   ## not 1var_per_way and source; not 1var_per_way and sink; 1var_per_way and sink
		        || (p, n) in process_sink
			)
		    && (ng, n) in group_node
		    && (r, ud, ng) in reserve__upDown__group}
	    ( v_reserve[p, r, ud, n, d, t] * p_entity_unitsize[p]
	      * p_process_reserve_upDown_node_reliability[p, r, ud, n]
	    )
  + sum {(p, r, ud, n) in process_reserve_upDown_node_active
		  : ( sum{(p, m) in process_method : m in method_1var_per_way} 1       ## 1var_per_way and source
		        && (p, n) in process_source
			)
		    && (ng, n) in group_node
		    && (r, ud, ng) in reserve__upDown__group}
	    ( v_reserve[p, r, ud, n, d, t] * p_entity_unitsize[p]
	      * p_process_reserve_upDown_node_reliability[p, r, ud, n]
	      * pdtProcess_slope[p, d, t]
		)
  + vq_reserve[r, ud, ng, d, t] * pdtReserve_upDown_group[r, ud, ng, 'reservation', d, t]
  >=
  + sum {(p, r, ud, n) in process_reserve_upDown_node_increase_reserve_ratio : (ng, n) in group_node
          && (r, ud, ng) in reserve__upDown__group}
	   ( + sum{(p, source, n) in process_source_sink} v_flow[p, source, n, d, t] * p_entity_unitsize[p] * p_process_reserve_upDown_node[p, r, ud, n, 'increase_reserve_ratio']
	     + sum{(p, n, sink) in process_source_sink_noEff} v_flow[p, n, sink, d, t] * p_entity_unitsize[p] * p_process_reserve_upDown_node[p, r, ud, n, 'increase_reserve_ratio']
	     + sum{(p, n, sink) in process_source_sink_eff} v_flow[p, n, sink, d, t] * p_entity_unitsize[p] * p_process_reserve_upDown_node[p, r, ud, n, 'increase_reserve_ratio']
	                                                    * pdtProcess_slope[p, d, t]
	   )
  + sum {(n, ng) in group_node : p_reserve_upDown_group[r, ud, ng, 'increase_reserve_ratio']}
	   (pdtNodeInflow[n, d, t] * p_reserve_upDown_group[r, ud, ng, 'increase_reserve_ratio'])/ step_duration[d, t]
;

s.t. reserveBalance_up_n_1_eq{(r, 'up', ng, r_m) in reserve__upDown__group__method_n_1, p_n_1 in process_large_failure, (d, t) in dt :
  sum{(p_n_1, sink) in process_sink : (ng, sink) in group_node} 1 } :
  + sum {(p, r, 'up', n) in process_reserve_upDown_node_active
	      : p <> p_n_1
		    && ( sum{(p, m) in process_method : m not in method_1var_per_way} 1  ## not 1var_per_way and source; not 1var_per_way and sink; 1var_per_way and sink
		         || (p, n) in process_sink
			   )
		    && (ng, n) in group_node
		    && (r, 'up', ng) in reserve__upDown__group
		}
	    ( v_reserve[p, r, 'up', n, d, t] * p_entity_unitsize[p]
	      * p_process_reserve_upDown_node_reliability[p, r, 'up', n]
	    )
  + sum {(p, r, 'up', n) in process_reserve_upDown_node_active
		  : p <> p_n_1
 		    && ( sum{(p, m) in process_method : m in method_1var_per_way} 1      ## 1var_per_way and source
		         && (p, n) in process_source
			   )
		    && (ng, n) in group_node
		    && (r, 'up', ng) in reserve__upDown__group
		}
	    ( v_reserve[p, r, 'up', n, d, t] * p_entity_unitsize[p]
	      * p_process_reserve_upDown_node_reliability[p, r, 'up', n]
	      * pdtProcess_slope[p, d, t]
		)
  + vq_reserve[r, 'up', ng, d, t] * pdtReserve_upDown_group[r, 'up', ng, 'reservation', d, t]
  >=
  + sum{(p_n_1, source, n) in process_source_sink : (ng, n) in group_node}
      + v_flow[p_n_1, source, n, d, t] * p_entity_unitsize[p_n_1]
	    * p_process_reserve_upDown_node[p_n_1, r, 'up', n, 'large_failure_ratio']
;

s.t. reserveBalance_down_n_1_eq{(r, 'down', ng, r_m) in reserve__upDown__group__method_n_1, p_n_1 in process_large_failure, (d, t) in dt :
  sum{(p_n_1, source) in process_source : (ng, source) in group_node} 1 } :
  + sum {(p, r, 'down', n) in process_reserve_upDown_node_active
	      : p <> p_n_1
		    && ( sum{(p, m) in process_method : m not in method_1var_per_way} 1  ## not 1var_per_way and source; not 1var_per_way and sink; 1var_per_way and sink
		         || (p, n) in process_sink
			   )
		    && (ng, n) in group_node
		    && (r, 'down', ng) in reserve__upDown__group
		}
	    ( v_reserve[p, r, 'down', n, d, t] * p_entity_unitsize[p]
	      * p_process_reserve_upDown_node[p, r, 'down', n, 'reliability']
	    )
  + sum {(p, r, 'down', n) in process_reserve_upDown_node_active
		  : p <> p_n_1
 		    && ( sum{(p, m) in process_method : m in method_1var_per_way} 1      ## 1var_per_way and source
		         && (p, n) in process_source
			   )
		    && (ng, n) in group_node
		    && (r, 'down', ng) in reserve__upDown__group
		}
	    ( v_reserve[p, r, 'down', n, d, t] * p_entity_unitsize[p]
	      * p_process_reserve_upDown_node[p, r, 'down', n, 'reliability']
	      * pdtProcess_slope[p, d, t]
		)
  + vq_reserve[r, 'down', ng, d, t] * pdtReserve_upDown_group[r, 'down', ng, 'reservation', d, t]
  >=
  + sum{(p_n_1, n, sink) in process_source_sink_noEff : (ng, n) in group_node}
      + v_flow[p_n_1, n, sink, d, t] * p_entity_unitsize[p_n_1]
	    * p_process_reserve_upDown_node[p_n_1, r, 'down', n, 'large_failure_ratio']
  + sum{(p_n_1, n, sink) in process_source_sink_eff : (ng, n) in group_node }
      + v_flow[p_n_1, n, sink, d, t] * p_entity_unitsize[p_n_1]
	    * p_process_reserve_upDown_node[p_n_1, r, 'down', n, 'large_failure_ratio']
	    * pdtProcess_slope[p_n_1, d, t]
;
param reserves := gmtime();
printf 'Timer - Reserves: %ss\n', reserves - balance;
printf '%s,%s\n', 'reserves', reserves - balance >> mod_phases_file;

# Indirect efficiency conversion - there is more than one variable. Direct conversion does not have an equation - it's directly in the nodeBalance_eq.
# Block-aware (Agent 1.5): the equation is emitted at the sink-side block
# b_out's timeline.  Source-side flows (at source-side block b_in) are
# aggregated through the overlap set ``b_c = b_out, b_fn = b_in``.  The
# sink-side and section/v_online terms live at the anchor block b_out.
# In the degenerate case (b_in = b_out = 'default') the identity overlap
# rows collapse the source-side sum to a single term at t_out = t,
# reducing the equation bit-identically to the pre-v51 power-balance
# form.  This is a power-balance equation (no block_step_duration
# scaling) — at each anchor step the source-side power × coefficient
# equals sink-side power × slope × coefficient (+ section × v_online).
# V1 limitation (inherited from Agent 1.3): both sides on the same
# non-default block has no self-identity overlap row; b_in coarser than
# b_out without a default side would also miss overlap rows.  The common
# V1 configurations (both default, or one side default) are covered.
s.t. conversion_indirect {(p, m) in process__method_indirect,
    (p, 'source', b_in) in process__side__block,
    (p, 'sink', b_out) in process__side__block,
    (b_out, d, t) in block__period__step} :
  + sum {(p, source) in process_source_undelayed,
         (d, b_out, t, b_in, t_f) in overlap}
    ( + p_overlap[d, b_out, t, b_in, t_f]
        * v_flow[p, source, p, d, t_f] * p_entity_unitsize[p]
  	      * p_process_source_flow_coefficient[p, source]
	)
  # Delayed-source contribution: kept at the anchor timeline t.  Block-
  # aware support for delays is out of scope for V1; in the degenerate
  # case (b_in = b_out = 'default') this reproduces the pre-v51 form.
  + sum {(p, source) in process_source_delayed}
      + sum {(d, t_, t, td) in dtt__delay_duration : (p, td) in process_delayed__duration}
        ( + v_flow[p, source, p, d, t_] * p_entity_unitsize[p]
  	          * p_process_source_flow_coefficient[p, source]
  	          * p_process_delay_weight[p, td]
	    )
  =
  + sum {(p, sink) in process_sink : p_process_sink_flow_coefficient[p,sink] != 0}
    ( + v_flow[p, p, sink, d, t] * p_entity_unitsize[p]
          * p_process_sink_flow_coefficient[p, sink]
    )
	  * pdtProcess_slope[p, d, t]
  + (if (p, 'min_load_efficiency') in process__ct_method then
			( + (if p in process_online_linear then v_online_linear[p, d, t])
			  + (if p in process_online_integer then v_online_integer[p, d, t])
			)
            * pdtProcess_section[p, d, t] * p_entity_unitsize[p])
;

# Agent 1.6: profile constraints are block-aware.
#
# Flow profile bounds (upper / lower / fixed) iterate at the process's
# own block (``process__block``) so the profile values are aggregated
# to the same resolution used by the UC / capacity constraints.
# State profile bounds iterate at the node's block (``node__block``)
# because v_state lives at the node's temporal resolution.  In the
# degenerate case every process / node sits on ``'default'`` and
# ``block__period__step`` at 'default' matches ``dt`` row-for-row.
s.t. profile_flow_upper_limit {(p, source, sink, f, 'upper_limit') in process__source__sink__profile__profile_method,
    (p, b_p) in process__block,
    (b_p, d, t) in block__period__step} :
  + ( + v_flow[p, source, sink, d, t]
      + sum{(p, r, 'up', sink) in process_reserve_upDown_node_active} v_reserve[p, r, 'up', sink, d, t]
	) * 1000
  <=
  + pdtProfile[f, d, t]
    * ( + p_entity_existing_count[p, d]
        + sum {(p, d_invest, d) in edd_invest} v_invest[p, d_invest]
        - sum {(p, d_divest) in pd_divest : p_years_d[d_divest] <= p_years_d[d]} v_divest[p, d_divest]
	  )
      * pdtProcess[p, 'availability', d, t]
      * 1000
;

s.t. profile_flow_lower_limit {(p, source, sink, f, 'lower_limit') in process__source__sink__profile__profile_method,
    (p, b_p) in process__block,
    (b_p, d, t) in block__period__step} :
  + ( + v_flow[p, source, sink, d, t]
      - sum{(p, r, 'down', sink) in process_reserve_upDown_node_active} v_reserve[p, r, 'down', sink, d, t]
    ) * 1000
  >=
  + pdtProfile[f, d, t]
    * ( + p_entity_existing_count[p, d]
        + sum {(p, d_invest, d) in edd_invest} v_invest[p, d_invest]
        - sum {(p, d_divest) in pd_divest : p_years_d[d_divest] <= p_years_d[d]} v_divest[p, d_divest]
	  )
      * pdtProcess[p, 'availability', d, t]
      * 1000
;

s.t. profile_flow_fixed {(p, source, sink, f, 'fixed') in process__source__sink__profile__profile_method,
    (p, b_p) in process__block,
    (b_p, d, t) in block__period__step} :
  + v_flow[p, source, sink, d, t] * 1000
  =
  + pdtProfile[f, d, t]
    * ( + p_entity_existing_count[p, d]
        + sum {(p, d_invest, d) in edd_invest} v_invest[p, d_invest]
        - sum {(p, d_divest) in pd_divest : p_years_d[d_divest] <= p_years_d[d]} v_divest[p, d_divest]
	  )
      * pdtProcess[p, 'availability', d, t]
      * 1000
;

s.t. profile_state_upper_limit {(n, f, 'upper_limit') in node__profile__profile_method,
    (n, b_n) in node__block,
    (b_n, d, t) in block__period__step} :
  + v_state[n, d, t] * 1000
  <=
  + pdtProfile[f, d, t]
    * ( + p_entity_existing_count[n, d]
        + sum {(n, d_invest, d) in edd_invest} v_invest[n, d_invest]
        - sum {(n, d_divest) in pd_divest : p_years_d[d_divest] <= p_years_d[d]} v_divest[n, d_divest]
	  )
      * pdtNode[n, 'availability', d, t]
      * 1000
;

s.t. profile_state_lower_limit {(n, f, 'lower_limit') in node__profile__profile_method,
    (n, b_n) in node__block,
    (b_n, d, t) in block__period__step} :
  + v_state[n, d, t] * 1000
  >=
  + pdtProfile[f, d, t]
    * ( + p_entity_existing_count[n, d]
        + sum {(n, d_invest, d) in edd_invest} v_invest[n, d_invest]
        - sum {(n, d_divest) in pd_divest : p_years_d[d_divest] <= p_years_d[d]} v_divest[n, d_divest]
	  )
      * pdtNode[n, 'availability', d, t]
      * 1000
;

s.t. profile_state_fixed {(n, f, 'fixed') in node__profile__profile_method,
    (n, b_n) in node__block,
    (b_n, d, t) in block__period__step} :
  + v_state[n, d, t] * 1000
  =
  + pdtProfile[f, d, t]
    * ( + p_entity_existing_count[n, d]
        + sum {(n, d_invest, d) in edd_invest} v_invest[n, d_invest]
        - sum {(n, d_divest) in pd_divest : p_years_d[d_divest] <= p_years_d[d]} v_divest[n, d_divest]
	  )
      * pdtNode[n, 'availability', d, t]
      * 1000
;

# storage_state_start_binding: pin the storage state so that
# storage_state_start × capacity means "state just before TS01 of the solve
# horizon" — consistent with how bind_forward_only handles it inline.
#
# For bind_within_period and bind_within_solve (single cyclic chain per
# period / solve): the balance at TS01 is cyclic, with t_previous wrapping
# to TS_last. So pinning v_state[TS_last of period_first] equals "state
# before TS01 of period_first" through the cyclic wrap, and TS01 inflow is
# absorbed correctly into v_state[TS01] = start × cap + flows_TS01.
#
# For bind_within_timeset and bind_intraperiod_blocks the cycle is
# per-block, not per-period, so TS_last of period_first sits inside the
# last block rather than the first — the old TS_first pin is kept for
# those methods (bind_intraperiod_blocks already routes TS01 flows
# correctly via nodeBalanceBlock_eq; bind_within_timeset has the same
# first-block absorption bug but needs a per-block fix, out of scope for
# this patch).
#
# (bind_forward_only handles this inline inside nodeBalance_eq via an
#  explicit v_state[TS01] − start × cap term; it is excluded from both
#  constraints below via the filters.)

# Agent 1.4: the seven storage constraints below now index v_state at the
# node's own block (b_n).  In the degenerate case (b_n = 'default') the
# tagged block__period__time_first / block__period__time_last sets match
# period__time_first / period__time_last exactly, so the LP is
# bit-identical to pre-v51.  For a coarser-block storage node the pin
# lands on the node's block boundary instead of the fine-grid one.
s.t. storage_state_start_binding_cyclic_period {n in nodeState, (n, b_n) in node__block,
     (b_n, d, t) in block__period__time_last
     : p_nested_model['solveFirst']
	 && ((n, 'bind_within_period') in node__storage_binding_method
	     || (n, 'bind_within_solve') in node__storage_binding_method)
	 && d in period_first
	 && ((n, 'fix_start') in node__storage_start_end_method || (n, 'fix_start_end') in node__storage_start_end_method)} :
  + v_state[n, d, t] * p_entity_unitsize[n]
  =
  + p_node[n,'storage_state_start']
    * ( + p_entity_all_existing[n, d]
        + sum {(n, d_invest, d) in edd_invest} v_invest[n, d_invest] * p_entity_unitsize[n]
        - sum {(n, d_divest) in pd_divest : p_years_d[d_divest] <= p_years_d[d]} v_divest[n, d_divest] * p_entity_unitsize[n]
	  )
;

s.t. storage_state_start_binding {n in nodeState, (n, b_n) in node__block,
     (b_n, d, t) in block__period__time_first
     : p_nested_model['solveFirst']
	 && (n, 'bind_forward_only') not in node__storage_binding_method
	 && (n, 'bind_within_period') not in node__storage_binding_method
	 && (n, 'bind_within_solve') not in node__storage_binding_method
	 && d in period_first
	 && ((n, 'fix_start') in node__storage_start_end_method || (n, 'fix_start_end') in node__storage_start_end_method)} :
  + v_state[n, d, t] * p_entity_unitsize[n]
  =
  + p_node[n,'storage_state_start']
    * ( + p_entity_all_existing[n, d]
        + sum {(n, d_invest, d) in edd_invest} v_invest[n, d_invest] * p_entity_unitsize[n]
        - sum {(n, d_divest) in pd_divest : p_years_d[d_divest] <= p_years_d[d]} v_divest[n, d_divest] * p_entity_unitsize[n]
	  )
;

s.t. storage_state_end {n in nodeState, (n, b_n) in node__block,
     (b_n, d, t) in block__period__time_last
     : p_nested_model['solveLast']
	 && d in period_last
	 && ((n, 'fix_end') in node__storage_start_end_method || (n, 'fix_start_end') in node__storage_start_end_method)
   && sum{(d2,d) in period__branch, (n,d2,t2) in ndt_fix_storage_quantity: (d, t, t2) in dtt_timeline_matching} 1 = 0
   && sum{(d2,d) in period__branch, (n,d2,t2) in ndt_fix_storage_price: (d, t, t2) in dtt_timeline_matching} 1 = 0
   && sum{(d2,d) in period__branch, (n,d2,t2) in ndt_fix_storage_usage: (d, t, t2) in dtt_timeline_matching} 1 = 0} :
  + v_state[n, d, t] * p_entity_unitsize[n]
  =
  + p_node[n,'storage_state_end']
    * ( + p_entity_all_existing[n, d]
        + sum {(n, d_invest, d) in edd_invest} v_invest[n, d_invest] * p_entity_unitsize[n]
        - sum {(n, d_divest) in pd_divest : p_years_d[d_divest] <= p_years_d[d]} v_divest[n, d_divest] * p_entity_unitsize[n]
	  )
;

#Storage state fix quantity for timesteps
s.t. node_balance_fix_quantity_eq_lower {n in n_fix_storage_quantity, (n, b_n) in node__block,
      (b_n, d, t) in block__period__time_last, (d2,d) in period__branch:
      d in period_last && sum{(n,d2,t2) in ndt_fix_storage_quantity: (d, t, t2) in dtt_timeline_matching} 1}:
  + v_state[n,d,t]* p_entity_unitsize[n]
  =
  + sum{(d, t, t2) in dtt_timeline_matching} p_fix_storage_quantity[n,d2,t2]
;

#Storage usage fix for timesteps

s.t. storage_usage_fix{n in n_fix_storage_usage, (n, b_n) in node__block,
      (b_n, d, t) in block__period__time_last, (d2,d) in period__branch:
      d in period_last && sum{(n,d2,t2) in ndt_fix_storage_usage: (d, t, t2) in dtt_timeline_matching} 1}:
  # n is sink
  - sum {(p, source, n) in process_source_sink, (d,t3) in dt} (
      + v_flow[p, source, n, d, t3] * p_entity_unitsize[p] * step_duration[d, t3]
  )
  # n is source
  + sum {(p, n, sink) in process_source_sink_eff, (d,t3) in dt} (
      + v_flow[p, n, sink, d, t3] * p_entity_unitsize[p]
        * pdtProcess_slope[p, d, t3]
      * (if p in process_unit then p_process_sink_flow_coefficient[p, sink] / p_process_source_flow_coefficient[p, n] else 1)
      + (if (p, 'min_load_efficiency') in process__ct_method then
        + ( + (if p in process_online_linear then v_online_linear[p, d, t3])
            + (if p in process_online_integer then v_online_integer[p, d, t3])
        )
          * pdtProcess_section[p, d, t3]
        * p_entity_unitsize[p]
    )
    ) * step_duration[d, t3]
  + sum {(p, n, sink) in process_source_sink_noEff, (d,t3) in dt}
    ( + v_flow[p, n, sink, d, t3] * p_entity_unitsize[p]
    ) * step_duration[d, t3]
  <=
  + sum{(n,d2,t2) in ndt_fix_storage_usage: exists{(d, t3, t2) in dtt_timeline_matching} 1} p_fix_storage_usage[n,d2,t2]
;

s.t. storage_state_solve_horizon_reference_value {n in nodeState, (n, b_n) in node__block,
     (b_n, d, t) in block__period__time_last
     : d in period_last
	 && ((n, 'use_reference_value') in node__storage_solve_horizon_method
   && (n, 'fix_end') not in node__storage_start_end_method
   && (n, 'fix_start_end') not in node__storage_start_end_method
   && (n, 'bind_within_solve') not in node__storage_binding_method
   && (n, 'bind_within_period') not in node__storage_binding_method
   && (n, 'bind_within_timeset') not in node__storage_binding_method
   && (n, 'bind_intraperiod_blocks') not in node__storage_binding_method
   && sum{(d2,d) in period__branch, (d, t, t2) in dtt_timeline_matching: n in n_fix_storage_price} 1 = 0
   && sum{(d2,d) in period__branch, (d, t, t2) in dtt_timeline_matching: n in n_fix_storage_quantity} 1 = 0
   && sum{(d2,d) in period__branch, (d, t, t2) in dtt_timeline_matching: n in n_fix_storage_usage} 1 = 0)} :
  + v_state[n, d, t] * p_entity_unitsize[n]
  =
  + pdtNode[n,'storage_state_reference_value', d, t]
    * ( + p_entity_all_existing[n, d]
        + sum {(n, d_invest, d) in edd_invest} v_invest[n, d_invest] * p_entity_unitsize[n]
        - sum {(n, d_divest) in pd_divest : p_years_d[d_divest] <= p_years_d[d]} v_divest[n, d_divest] * p_entity_unitsize[n]
	  )
;

s.t. constraint_greater_than {(c, 'greater_than') in constraint__sense, (d, t) in dt} :
  + sum {(p, source, sink) in process_source_sink : (p, source, c) in process_node_flow_constraint}
    ( + v_flow[p, source, sink, d, t] * p_entity_unitsize[p]
	      * p_process_node_constraint_flow_coefficient[p, source, c]
	)
  + sum {(p, source, sink) in process_source_sink : (p, sink, c) in process_node_flow_constraint}
    ( + v_flow[p, source, sink, d, t] * p_entity_unitsize[p]
	      * p_process_node_constraint_flow_coefficient[p, sink, c]
	)
  + sum {(n, c) in node_state_constraint}
    ( + v_state[n, d, t]
	      * p_node_constraint_state_coefficient[n, c]
		  * p_entity_unitsize[n]
	)
  + sum {(n, c) in node_capacity_constraint_invested : (n, d) in nd_invest}
    ( v_invest[n, d]
      * p_node_constraint_invested_capacity_coefficient[n, c]
      * p_entity_unitsize[n]
    )
  + sum {(p, c) in process_capacity_constraint_invested : (p, d) in pd_invest}
    ( v_invest[p, d]
      * p_process_constraint_invested_capacity_coefficient[p, c]
      * p_entity_unitsize[p]
    )
  + sum {(n, c) in node_capacity_constraint_prebuilt}
    ( ( + p_entity_all_existing[n, d] / p_entity_unitsize[n]
        + sum{(n, d_invest, d) in edd_invest : p_years_d[d_invest] < p_years_d[d]} v_invest[n, d_invest]
      )
      * p_node_constraint_prebuilt_capacity_coefficient[n, c]
      * p_entity_unitsize[n]
    )
  + sum {(p, c) in process_capacity_constraint_prebuilt}
    ( ( + p_entity_all_existing[p, d] / p_entity_unitsize[p]
        + sum{(p, d_invest, d) in edd_invest : p_years_d[d_invest] < p_years_d[d]} v_invest[p, d_invest]
      )
      * p_process_constraint_prebuilt_capacity_coefficient[p, c]
      * p_entity_unitsize[p]
    )
  >=
  + p_constraint_constant[c]
;

s.t. process_constraint_less_than {(c, 'less_than') in constraint__sense, (d, t) in dt} :
  + sum {(p, source, sink) in process_source_sink : (p, source, c) in process_node_flow_constraint}
    ( + v_flow[p, source, sink, d, t] * p_entity_unitsize[p]
	      * p_process_node_constraint_flow_coefficient[p, source, c]
	)
  + sum {(p, source, sink) in process_source_sink : (p, sink, c) in process_node_flow_constraint}
    ( + v_flow[p, source, sink, d, t] * p_entity_unitsize[p]
	      * p_process_node_constraint_flow_coefficient[p, sink, c]
	)
  + sum {(n, c) in node_state_constraint}
    ( + v_state[n, d, t]
	      * p_node_constraint_state_coefficient[n, c]
		  * p_entity_unitsize[n]
	)
  + sum {(n, c) in node_capacity_constraint_invested : (n, d) in nd_invest}
    ( v_invest[n, d]
      * p_node_constraint_invested_capacity_coefficient[n, c]
      * p_entity_unitsize[n]
    )
  + sum {(p, c) in process_capacity_constraint_invested : (p, d) in pd_invest}
    ( v_invest[p, d]
      * p_process_constraint_invested_capacity_coefficient[p, c]
      * p_entity_unitsize[p]
    )
  + sum {(n, c) in node_capacity_constraint_prebuilt}
    ( ( + p_entity_all_existing[n, d] / p_entity_unitsize[n]
        + sum{(n, d_invest, d) in edd_invest : p_years_d[d_invest] < p_years_d[d]} v_invest[n, d_invest]
      )
      * p_node_constraint_prebuilt_capacity_coefficient[n, c]
      * p_entity_unitsize[n]
    )
  + sum {(p, c) in process_capacity_constraint_prebuilt}
    ( ( + p_entity_all_existing[p, d] / p_entity_unitsize[p]
        + sum{(p, d_invest, d) in edd_invest : p_years_d[d_invest] < p_years_d[d]} v_invest[p, d_invest]
      )
      * p_process_constraint_prebuilt_capacity_coefficient[p, c]
      * p_entity_unitsize[p]
    )
  <=
  + p_constraint_constant[c]
;

s.t. process_constraint_equal {(c, 'equal') in constraint__sense, (d, t) in dt} :
  + sum {(p, source, sink) in process_source_sink : (p, source, c) in process_node_flow_constraint}
    ( + v_flow[p, source, sink, d, t] * p_entity_unitsize[p]
	      * p_process_node_constraint_flow_coefficient[p, source, c]
	)
  + sum {(p, source, sink) in process_source_sink : (p, sink, c) in process_node_flow_constraint}
    ( + v_flow[p, source, sink, d, t] * p_entity_unitsize[p]
	      * p_process_node_constraint_flow_coefficient[p, sink, c]
	)
  + sum {(n, c) in node_state_constraint}
    ( + v_state[n, d, t]
	      * p_node_constraint_state_coefficient[n, c]
		  * p_entity_unitsize[n]
	)
  + sum {(n, c) in node_capacity_constraint_invested : (n, d) in nd_invest}
    ( v_invest[n, d]
      * p_node_constraint_invested_capacity_coefficient[n, c]
      * p_entity_unitsize[n]
    )
  + sum {(p, c) in process_capacity_constraint_invested : (p, d) in pd_invest}
    ( v_invest[p, d]
      * p_process_constraint_invested_capacity_coefficient[p, c]
      * p_entity_unitsize[p]
    )
  + sum {(n, c) in node_capacity_constraint_prebuilt}
    ( ( + p_entity_all_existing[n, d] / p_entity_unitsize[n]
        + sum{(n, d_invest, d) in edd_invest : p_years_d[d_invest] < p_years_d[d]} v_invest[n, d_invest]
      )
      * p_node_constraint_prebuilt_capacity_coefficient[n, c]
      * p_entity_unitsize[n]
    )
  + sum {(p, c) in process_capacity_constraint_prebuilt}
    ( ( + p_entity_all_existing[p, d] / p_entity_unitsize[p]
        + sum{(p, d_invest, d) in edd_invest : p_years_d[d_invest] < p_years_d[d]} v_invest[p, d_invest]
      )
      * p_process_constraint_prebuilt_capacity_coefficient[p, c]
      * p_entity_unitsize[p]
    )
  =
  + p_constraint_constant[c]
;


s.t. maxState {n in nodeState, (n, b_n) in node__block,
    (b_n, d, t) in block__period__step} :
  + v_state[n, d, t] * p_entity_unitsize[n]
  <=
  + (
      + p_entity_all_existing[n, d]
      + sum {(n, d_invest, d) in edd_invest} v_invest[n, d_invest] * p_entity_unitsize[n]
      - sum {(n, d_divest) in pd_divest : p_years_d[d_divest] <= p_years_d[d]} v_divest[n, d_divest] * p_entity_unitsize[n]
    )
	* pdtNode[n, 'availability', d, t]
;

# Representative period inter-period storage balance (Paper Eq 2i-2j)
# Net change in inter-period state = weighted sum of intra-period net changes
s.t. rp_inter_period_balance
    {n in nodeState_rp, (b, b_prev) in rp_base_chain} :
  v_state_inter[n, b] - v_state_inter[n, b_prev]
  =
  sum{(b, r) in rp_base__rep, d in period_in_use : (d, r) in rp_block_first}
  (
    p_rp_weight[b, r]
    * ( v_state[n, d, p_rp_last_step[r]] - v_state_rp_start[n, d, r] )
    * p_entity_unitsize[n]
  )
;

# Cyclic constraint (Paper Eq 2k): first base period wraps to last
s.t. rp_inter_period_cyclic
    {n in nodeState_rp, b_first in rp_base_first, b_last in rp_base_last} :
  v_state_inter[n, b_first] - v_state_inter[n, b_last]
  =
  sum{(b_first, r) in rp_base__rep, d in period_in_use : (d, r) in rp_block_first}
  (
    p_rp_weight[b_first, r]
    * ( v_state[n, d, p_rp_last_step[r]] - v_state_rp_start[n, d, r] )
    * p_entity_unitsize[n]
  )
;

# Inter-period state upper bound (must not exceed storage capacity)
s.t. rp_inter_period_max_state
    {n in nodeState_rp, b in rp_base_period, d in period_in_use} :
  v_state_inter[n, b] * p_entity_unitsize[n]
  <=
  + p_entity_all_existing[n, d]
  + sum {(n, d_invest, d) in edd_invest} v_invest[n, d_invest] * p_entity_unitsize[n]
  - sum {(n, d_divest) in pd_divest : p_years_d[d_divest] <= p_years_d[d]} v_divest[n, d_divest] * p_entity_unitsize[n]
;

# Block-aware (Agent 1.5): capacity bounds emit once per (d, t) in the
# sink-side block b_out's timeline — v_flow, startup/shutdown
# tightening terms, v_online and availability all evaluate at that
# block's timesteps.  Degenerate case (b_out = 'default') reduces to
# the pre-v51 fine-grid indexing exactly.
s.t. maxToSink {(p, source, sink) in process__source__sinkIsNode,
    (p, 'sink', b_out) in process__side__block,
    (b_out, d, t) in block__period__step
    : p_process_sink_flow_coefficient[p, sink]} :
  + v_flow[p, source, sink, d, t] * p_entity_unitsize[p]
  + sum {r in reserve : (p, r, 'up', sink) in process_reserve_upDown_node_active} v_reserve[p, r, 'up', sink, d, t] * p_entity_unitsize[p]
  <=
  + ( if p not in process_online then
      + ( + p_entity_all_existing[p, d]
          + sum {(p, d_invest, d) in edd_invest} v_invest[p, d_invest] * p_entity_unitsize[p]
          - sum {(p, d_divest) in pd_divest : p_years_d[d_divest] <= p_years_d[d]} v_divest[p, d_divest] * p_entity_unitsize[p]
	    )
		* pdtProcess[p, 'availability', d, t]
		* p_process_sink_max_capacity_coefficient[p, sink]
	)
  + ( if p in process_online_linear then
      + p_process_sink_max_capacity_coefficient[p, sink]
        * v_online_linear[p, d, t]
		* p_entity_unitsize[p]
        * pdtProcess[p, 'availability', d, t]
    )
  + ( if p in process_online_integer then
      + p_process_sink_max_capacity_coefficient[p, sink]
        * v_online_integer[p, d, t]
		* p_entity_unitsize[p]
        * pdtProcess[p, 'availability', d, t]
    )
  # Morales-Espana startup tightening: reduce max output in startup timestep
  - ( if p in process_online_linear && p_startup_cap_reduction_sink[p, sink, d, t] then
      + p_startup_cap_reduction_sink[p, sink, d, t]
        * p_process_sink_max_capacity_coefficient[p, sink]
        * v_startup_linear[p, d, t]
        * p_entity_unitsize[p]
    )
  - ( if p in process_online_integer && p_startup_cap_reduction_sink[p, sink, d, t] then
      + p_startup_cap_reduction_sink[p, sink, d, t]
        * p_process_sink_max_capacity_coefficient[p, sink]
        * v_startup_integer[p, d, t]
        * p_entity_unitsize[p]
    )
  # Morales-Espana shutdown tightening: reduce max output in pre-shutdown timestep
  - sum{(d, t, d2, t2) in dtdt_next} (
      + ( if p in process_online_linear && p_shutdown_cap_reduction_sink[p, sink, d, t] then
          + p_shutdown_cap_reduction_sink[p, sink, d, t]
            * p_process_sink_max_capacity_coefficient[p, sink]
            * v_shutdown_linear[p, d2, t2]
            * p_entity_unitsize[p]
        )
      + ( if p in process_online_integer && p_shutdown_cap_reduction_sink[p, sink, d, t] then
          + p_shutdown_cap_reduction_sink[p, sink, d, t]
            * p_process_sink_max_capacity_coefficient[p, sink]
            * v_shutdown_integer[p, d2, t2]
            * p_entity_unitsize[p]
        )
    )
;

s.t. minToSink {(p, source, sink) in process__source__sinkIsNode_not2way1var,
    (p, 'sink', b_out) in process__side__block,
    (b_out, d, t) in block__period__step
    : p_process_sink_flow_coefficient[p, sink]} :
  + v_flow[p, source, sink, d, t] >= 0
;

s.t. minToSink_minload {(p, source, sink) in process__source__sinkIsNode_not2way1var,
    (p, 'sink', b_out) in process__side__block,
    (b_out, d, t) in block__period__step
    : p_process_sink_flow_coefficient[p, sink] && p in process_online} :
  + sum{(p, source, sink2) in process__source__sinkIsNode_not2way1var} v_flow[p, source, sink2, d, t]
  >=
  + (if p in process_online_linear then v_online_linear[p, d, t] * p_process[p, 'min_load'] * p_process_sink_min_capacity_coefficient[p, sink] else 0)
  + (if p in process_online_integer then v_online_integer[p, d, t] * p_process[p, 'min_load'] * p_process_sink_min_capacity_coefficient[p, sink] else 0)
;

# Block-aware (Agent 1.5): source-side capacity bounds iterate the
# source-side block b_in's timeline.  Degenerate (b_in = 'default')
# reproduces the pre-v51 fine-grid domain.
s.t. maxFromSource {(p, source, sink) in process__sourceIsNode__sink_1way_noSinkOrMoreThan1Source,
    (p, 'source', b_in) in process__side__block,
    (b_in, d, t) in block__period__step
    : p_process_source_flow_coefficient[p, source]} :
  + v_flow[p, source, sink, d, t] * p_entity_unitsize[p]
  <=
  + ( if p not in process_online then
      + ( + p_entity_all_existing[p, d]
          + sum {(p, d_invest, d) in edd_invest} v_invest[p, d_invest] * p_entity_unitsize[p]
          - sum {(p, d_divest) in pd_divest : p_years_d[d_divest] <= p_years_d[d]} v_divest[p, d_divest] * p_entity_unitsize[p]
	    )
	    * pdtProcess[p, 'availability', d, t]
		* p_process_source_max_capacity_coefficient[p, source]
	)
  + ( if p in process_online_linear then
      + v_online_linear[p, d, t]
		* p_entity_unitsize[p]
        * pdtProcess[p, 'availability', d, t]
		* p_process_source_max_capacity_coefficient[p, source]
	)
  + ( if p in process_online_integer then
      + v_online_integer[p, d, t]
		* p_entity_unitsize[p]
        * pdtProcess[p, 'availability', d, t]
		* p_process_source_max_capacity_coefficient[p, source]
    )
  # Morales-Espana startup tightening: reduce max output in startup timestep
  - ( if p in process_online_linear && p_startup_cap_reduction_source[p, source, d, t] then
      + p_startup_cap_reduction_source[p, source, d, t]
        * p_process_source_max_capacity_coefficient[p, source]
        * v_startup_linear[p, d, t]
        * p_entity_unitsize[p]
    )
  - ( if p in process_online_integer && p_startup_cap_reduction_source[p, source, d, t] then
      + p_startup_cap_reduction_source[p, source, d, t]
        * p_process_source_max_capacity_coefficient[p, source]
        * v_startup_integer[p, d, t]
        * p_entity_unitsize[p]
    )
  # Morales-Espana shutdown tightening: reduce max output in pre-shutdown timestep
  - sum{(d, t, d2, t2) in dtdt_next} (
      + ( if p in process_online_linear && p_shutdown_cap_reduction_source[p, source, d, t] then
          + p_shutdown_cap_reduction_source[p, source, d, t]
            * p_process_source_max_capacity_coefficient[p, source]
            * v_shutdown_linear[p, d2, t2]
            * p_entity_unitsize[p]
        )
      + ( if p in process_online_integer && p_shutdown_cap_reduction_source[p, source, d, t] then
          + p_shutdown_cap_reduction_source[p, source, d, t]
            * p_process_source_max_capacity_coefficient[p, source]
            * v_shutdown_integer[p, d2, t2]
            * p_entity_unitsize[p]
        )
    )
;

# Force source flows from 1-way processes with more than 1 source to be at least 0 (conversion equation does not do it)
s.t. minFromSource {(p, source, sink) in process__sourceIsNode__sink_1way_noSinkOrMoreThan1Source,
    (p, 'source', b_in) in process__side__block,
    (b_in, d, t) in block__period__step
    : p_process_source_flow_coefficient[p, source]} :
  + v_flow[p, source, sink, d, t] * p_process_source_flow_coefficient[p, source] >= 0
;

s.t. minFromSource_minload {(p, source, sink) in process__sourceIsNode__sink_1way_noSinkOrMoreThan1Source,
    (p, 'source', b_in) in process__side__block,
    (b_in, d, t) in block__period__step
    : p_process_source_flow_coefficient[p, source] && p in process_online} :
  +  sum{(p, source2, sink) in process__sourceIsNode__sink_1way_noSinkOrMoreThan1Source} v_flow[p, source2, sink, d, t] * p_process_source_min_capacity_coefficient[p, source]
  >=
  + (if p in process_online_linear then v_online_linear[p, d, t] * p_process[p, 'min_load'] else 0)
  + (if p in process_online_integer then v_online_integer[p, d, t] * p_process[p, 'min_load'] else 0)
;

# Special equation to limit the 1variable connection on the negative transfer
# Block-aware (Agent 1.5): operates on v_flow at the sink-side block b_out.
s.t. minToSink_1var {(p, source, sink) in process__source__sinkIsNode_2way1var,
    (p, 'sink', b_out) in process__side__block,
    (b_out, d, t) in block__period__step
    : p_process_sink_flow_coefficient[p, sink]} :
  + v_flow[p, source, sink, d, t] * p_entity_unitsize[p]
  >=
  - ( if p not in process_online then
      + p_process_sink_max_capacity_coefficient[p, sink]
        * ( + p_entity_all_existing[p, d]
            + sum {(p, d_invest, d) in edd_invest} v_invest[p, d_invest] * p_entity_unitsize[p]
            - sum {(p, d_divest) in pd_divest : p_years_d[d_divest] <= p_years_d[d]} v_divest[p, d_divest] * p_entity_unitsize[p]
	      )
	)
  - ( if p in process_online_linear then
      + p_process_sink_max_capacity_coefficient[p, sink]
        * v_online_linear[p, d, t]
		* p_entity_unitsize[p]
    )
  - ( if p in process_online_integer then
      + p_process_sink_max_capacity_coefficient[p, sink]
        * v_online_integer[p, d, t]
		* p_entity_unitsize[p]
    )
;

# Special equations for the method with 2 variables presenting a direct 2way connection between source and sink (without the process)
# Block-aware (Agent 1.5): flow direction is sink→source so the variable
# lives at the source-side block b_in.
s.t. maxToSource {(p, sink, source) in process_sink_toSource,
    (p, 'source', b_in) in process__side__block,
    (b_in, d, t) in block__period__step
    : p_process_source_flow_coefficient[p, source]} :
  + v_flow[p, sink, source, d, t] * p_entity_unitsize[p]
  + sum {r in reserve : (p, r, 'up', source) in process_reserve_upDown_node_active} v_reserve[p, r, 'up', source, d, t] * p_entity_unitsize[p]
  <=
  + ( if p not in process_online then
      + p_process_source_max_capacity_coefficient[p, source]
        * ( + p_entity_all_existing[p, d]
            + sum {(p, d_invest, d) in edd_invest} v_invest[p, d_invest] * p_entity_unitsize[p]
            - sum {(p, d_divest) in pd_divest : p_years_d[d_divest] <= p_years_d[d]} v_divest[p, d_divest] * p_entity_unitsize[p]
	      )
		  * pdtProcess[p, 'availability', d, t]
	)
  + ( if p in process_online_linear then
      + p_process_source_max_capacity_coefficient[p, source]
        * v_online_linear[p, d, t]
		* p_entity_unitsize[p]
        * pdtProcess[p, 'availability', d, t]
    )
  + ( if p in process_online_integer then
      + p_process_source_max_capacity_coefficient[p, source]
        * (
		    + p_entity_existing_integer_count[p, d]
            + sum {(p, d_invest, d) in edd_invest} v_invest[p, d_invest]
			- v_online_integer[p, d, t]   # Using binary online variable as a switch between directions
		  )
		* p_entity_unitsize[p]
        * pdtProcess[p, 'availability', d, t]
    )
;

s.t. minToSource {(p, source, sink) in process__source__sinkIsNode_2way2var,
    (p, 'source', b_in) in process__side__block,
    (b_in, d, t) in block__period__step
    : p_process_source_flow_coefficient[p, source]} :
  + v_flow[p, sink, source, d, t]
  >=
  + (if p in process_online_linear then v_online_linear[p, d, t] * p_process[p, 'min_load'] * p_process_source_min_capacity_coefficient[p, source] else 0)
  + (if p in process_online_integer then v_online_integer[p, d, t] * p_process[p, 'min_load'] * p_process_source_min_capacity_coefficient[p, source] else 0)
;

# DC power flow: flow on connection equals susceptance * angle difference
# Block-aware (Agent 1.5): emit at the sink-side block b_out's timeline.
# In V1 DC power flow scenarios the participating nodes sit at the default
# (hourly) electricity grid, so b_out = 'default' and this reduces to the
# pre-v51 fine-grid domain.  v_angle stays on the fine timeline; a
# non-default b_out would need angle variables redefined on the block —
# out of scope for V1.
s.t. dc_flow_eq {p in connection_dc_power_flow,
                 (p, source, sink) in process_source_toSink,
                 (p, 'sink', b_out) in process__side__block,
                 (b_out, d, t) in block__period__step
                 : source in node_dc_power_flow && sink in node_dc_power_flow} :
  v_flow[p, source, sink, d, t] * p_entity_unitsize[p]
  =
  p_connection_susceptance[p] * (v_angle[source, d, t] - v_angle[sink, d, t])
;

# Agent 1.6: UC constraints are emitted on the process's own block
# (``process__block``) rather than the solve-wide fine ``dt``.  The
# startup/shutdown equalities use ``block_dtttdt`` (Agent 1.4) at the
# process's block so the predecessor lookup is block-consistent; the
# min-up/min-down equations retain their ``pdt_uptime`` / ``pdt_downtime``
# domain but are filtered through the ``p_online_dt`` helper set so the
# rows line up with the timesteps at which ``v_online_*`` / ``v_startup_*``
# actually exist.  In the degenerate case (process_block = 'default')
# the filter is a no-op and every constraint reduces to its pre-v51 form.
s.t. maxOnline {(p, d, t) in p_online_dt} :
  + (if p in process_online_linear then v_online_linear[p, d, t])
  + (if p in process_online_integer then v_online_integer[p, d, t])
  <=
  + p_entity_all_existing[p, d] / p_entity_unitsize[p]
  + sum {(p, d_invest, d) in edd_invest} v_invest[p, d_invest]
  - sum {(p, d_divest) in pd_divest : p_years_d[d_divest] <= p_years_d[d]} v_divest[p, d_divest]
;

s.t. online__startup_linear {p in process_online_linear,
    (p, b) in process__block,
    (b, d, t, t_previous, t_previous_within_timeset, d_previous, t_previous_within_solve) in block_dtttdt} :
  + v_startup_linear[p, d, t]
  >=
  + v_online_linear[p, d, t]
  - v_online_linear[p, d_previous, t_previous_within_solve]
;

s.t. online__startup_integer {p in process_online_integer,
    (p, b) in process__block,
    (b, d, t, t_previous, t_previous_within_timeset, d_previous, t_previous_within_solve) in block_dtttdt} :
  + v_startup_integer[p, d, t]
  >=
  + v_online_integer[p, d, t]
  - v_online_integer[p, d_previous, t_previous_within_solve]
;

s.t. maxStartup {(p, d, t) in p_online_dt} :
  + (if p in process_online_linear then v_startup_linear[p, d, t])
  + (if p in process_online_integer then v_startup_integer[p, d, t])
  <=
  + p_entity_all_existing[p, d] / p_entity_unitsize[p]
  + sum {(p, d_invest, d) in edd_invest} v_invest[p, d_invest]
  - sum {(p, d_divest) in pd_divest : p_years_d[d_divest] <= p_years_d[d]} v_divest[p, d_divest]
;

s.t. online__shutdown_linear {p in process_online_linear,
    (p, b) in process__block,
    (b, d, t, t_previous, t_previous_within_timeset, d_previous, t_previous_within_solve) in block_dtttdt} :
  + v_shutdown_linear[p, d, t]
  >=
  - v_online_linear[p, d, t]
  + v_online_linear[p, d_previous, t_previous_within_solve]
;

s.t. online__shutdown_integer {p in process_online_integer,
    (p, b) in process__block,
    (b, d, t, t_previous, t_previous_within_timeset, d_previous, t_previous_within_solve) in block_dtttdt} :
  + v_shutdown_integer[p, d, t]
  >=
  - v_online_integer[p, d, t]
  + v_online_integer[p, d_previous, t_previous_within_solve]
;

s.t. maxShutdown {(p, d, t) in p_online_dt} :
  + (if p in process_online_linear then v_shutdown_linear[p, d, t])
  + (if p in process_online_integer then v_shutdown_integer[p, d, t])
  <=
  + p_entity_all_existing[p, d] / p_entity_unitsize[p]
  + sum {(p, d_invest, d) in edd_invest} v_invest[p, d_invest]
  - sum {(p, d_divest) in pd_divest : p_years_d[d_divest] <= p_years_d[d]} v_divest[p, d_divest]
;

# Minimum downtime: if a unit shut down within the last min_downtime hours, it must still be offline.
# Agent 1.6: restrict (d, t) to timesteps at which v_online_* is defined
# (i.e. the process's block).  In the degenerate case this filter is a
# no-op.  pdt_downtime / downtime_lookback are computed on the fine
# solve timeline; rows that target a (d2, t2) outside the process's
# block timeline remain on the fine grid — V1 requires matched side
# blocks for UC so this is consistent (the fine timeline of the
# process's block equals the matching-side timeline).
s.t. minimum_downtime {(p, d, t) in pdt_downtime : p in process_online
                        && (p, d, t) in p_online_dt} :
  + p_entity_all_existing[p, d] / p_entity_unitsize[p]
  + sum {(p, d_invest, d) in edd_invest} v_invest[p, d_invest]
  - sum {(p, d_divest) in pd_divest : p_years_d[d_divest] <= p_years_d[d]} v_divest[p, d_divest]
  - (if p in process_online_linear then v_online_linear[p, d, t])
  - (if p in process_online_integer then v_online_integer[p, d, t])
  >=
  + sum{(p, d, t, d2, t2) in downtime_lookback : (p, d2, t2) in p_online_dt}
    ( + (if p in process_online_linear then v_shutdown_linear[p, d2, t2])
      + (if p in process_online_integer then v_shutdown_integer[p, d2, t2])
    )
;

# Minimum uptime: if a unit started up within the last min_uptime hours, it must still be online
s.t. minimum_uptime {(p, d, t) in pdt_uptime : p in process_online
                      && (p, d, t) in p_online_dt} :
  + (if p in process_online_linear then v_online_linear[p, d, t])
  + (if p in process_online_integer then v_online_integer[p, d, t])
  >=
  + sum{(p, d, t, d2, t2) in uptime_lookback : (p, d2, t2) in p_online_dt}
    ( + (if p in process_online_linear then v_startup_linear[p, d2, t2])
      + (if p in process_online_integer then v_startup_integer[p, d2, t2])
    )
;

# Agent 1.6: ramp constraints are block-aware.
#
# ``ramp_up_variable`` defines v_ramp from two consecutive v_flow values.
# Since v_flow / v_ramp are declared on the fine ``(d, t) in dt`` grid we
# stick with ``dtttdt`` for the predecessor linkage — the equality links
# a flow at t to the flow at t_previous, and the block-aware capacity
# checks downstream already gate where non-zero values are permitted.
# In the degenerate case this reduces exactly to the pre-v51 equation.
#
# The four ramp-limit constraints live at the flow's own side's block —
# source-side ramps at ``process_block_in`` (b_in), sink-side at
# ``process_block_out`` (b_out).  Iteration uses ``block_dtttdt`` so the
# predecessor is consistent with that block's own timeline; the existing
# ramp-set filters (ramp_speed × 60 < step_duration && dt_jump == 1) are
# inlined here so we do not re-derive the Cartesian set on a mixed-grid
# domain.  In the degenerate case ``block_dtttdt`` tagged with 'default'
# equals ``dtttdt`` and the inlined filter matches the original.
s.t. ramp_up_variable {(p, source, sink) in process_source_sink_ramp, (d, t, t_previous, t_previous_within_timeset, d_previous, t_previous_within_solve) in dtttdt} :
  + v_ramp[p, source, sink, d, t]
  =
  + v_flow[p, source, sink, d, t]  * step_duration[d, t]
  - v_flow[p, source, sink, d, t_previous] * step_duration[d, t]
;

s.t. ramp_source_up_constraint {(p, source, sink) in process_source_sink_ramp_limit_source_up,
    (p, 'source', b_in) in process__side__block,
    (b_in, d, t, t_previous, t_previous_within_timeset, d_previous, t_previous_within_solve) in block_dtttdt
    : p_process_source[p, source, 'ramp_speed_up'] * 60 < step_duration[d, t]
      && dt_jump[d, t] == 1} :
  + v_ramp[p, source, sink, d, t] * p_entity_unitsize[p]
  + sum {r in reserve : (p, r, 'up', source) in process_reserve_upDown_node_active}
         (v_reserve[p, r, 'up', source, d, t] * p_entity_unitsize[p] / step_duration[d, t])
  <=
  + p_process_source[p, source, 'ramp_speed_up']
    * 60 * step_duration[d, t]
	* p_process_source_max_capacity_coefficient[p, source]
    * ( + p_entity_all_existing[p, d]
	    + sum {(p, d_invest, d) in edd_invest} v_invest[p, d_invest] * p_entity_unitsize[p]
        - sum {(p, d_divest) in pd_divest : p_years_d[d_divest] <= p_years_d[d]} v_divest[p, d_divest] * p_entity_unitsize[p]
	  )
  + ( if p in process_online_linear then v_startup_linear[p, d, t] * p_entity_unitsize[p] )  # To make sure that units can startup despite ramp limits.
  + ( if p in process_online_integer then v_startup_integer[p, d, t] * p_entity_unitsize[p] )  # To make sure that units can startup despite ramp limits.
;

s.t. ramp_sink_up_constraint {(p, source, sink) in process_source_sink_ramp_limit_sink_up,
    (p, 'sink', b_out) in process__side__block,
    (b_out, d, t, t_previous, t_previous_within_timeset, d_previous, t_previous_within_solve) in block_dtttdt
    : p_process_sink[p, sink, 'ramp_speed_up'] * 60 < step_duration[d, t]
      && dt_jump[d, t] == 1} :
  + v_ramp[p, source, sink, d, t] * p_entity_unitsize[p]
  + sum {r in reserve : (p, r, 'up', sink) in process_reserve_upDown_node_active}
         (v_reserve[p, r, 'up', sink, d, t] * p_entity_unitsize[p] / step_duration[d, t])
  <=
  + p_process_sink[p, sink, 'ramp_speed_up']
    * 60 * step_duration[d, t]
	* p_process_sink_max_capacity_coefficient[p, sink]
    * ( + p_entity_all_existing[p, d]
	    + sum {(p, d_invest, d) in edd_invest} v_invest[p, d_invest] * p_entity_unitsize[p]
        - sum {(p, d_divest) in pd_divest : p_years_d[d_divest] <= p_years_d[d]} v_divest[p, d_divest] * p_entity_unitsize[p]
	  )
  + ( if p in process_online_linear then v_startup_linear[p, d, t] * p_entity_unitsize[p] )  # To make sure that units can startup despite ramp limits.
  + ( if p in process_online_integer then v_startup_integer[p, d, t] * p_entity_unitsize[p] )  # To make sure that units can startup despite ramp limits.
;

s.t. ramp_source_down_constraint {(p, source, sink) in process_source_sink_ramp_limit_source_down,
    (p, 'source', b_in) in process__side__block,
    (b_in, d, t, t_previous, t_previous_within_timeset, d_previous, t_previous_within_solve) in block_dtttdt
    : p_process_source[p, source, 'ramp_speed_down'] * 60 < step_duration[d, t]
      && dt_jump[d, t] == 1} :
  + v_ramp[p, source, sink, d, t] * p_entity_unitsize[p]
  + sum {r in reserve : (p, r, 'down', source) in process_reserve_upDown_node_active}
         (v_reserve[p, r, 'down', source, d, t] * p_entity_unitsize[p] / step_duration[d, t])
  >=
  - p_process_sink[p, source, 'ramp_speed_down']
    * 60 * step_duration[d, t]
	* p_process_source_max_capacity_coefficient[p, source]
    * ( + p_entity_all_existing[p, d]
	    + sum {(p, d_invest, d) in edd_invest} v_invest[p, d_invest] * p_entity_unitsize[p]
        - sum {(p, d_divest) in pd_divest : p_years_d[d_divest] <= p_years_d[d]} v_divest[p, d_divest] * p_entity_unitsize[p]
	  )
  - ( if p in process_online_linear then v_shutdown_linear[p, d, t] * p_entity_unitsize[p] )  # To make sure that units can shutdown despite ramp limits.
  - ( if p in process_online_integer then v_shutdown_integer[p, d, t] * p_entity_unitsize[p] )  # To make sure that units can shutdown despite ramp limits.
;

s.t. ramp_sink_down_constraint {(p, source, sink) in process_source_sink_ramp_limit_sink_down,
    (p, 'sink', b_out) in process__side__block,
    (b_out, d, t, t_previous, t_previous_within_timeset, d_previous, t_previous_within_solve) in block_dtttdt
    : p_process_sink[p, sink, 'ramp_speed_down'] * 60 < step_duration[d, t]
      && dt_jump[d, t] == 1} :
  + v_ramp[p, source, sink, d, t] * p_entity_unitsize[p]
  + sum {r in reserve : (p, r, 'down', sink) in process_reserve_upDown_node_active}
         (v_reserve[p, r, 'down', sink, d, t] * p_entity_unitsize[p] / step_duration[d, t])
  >=
  - p_process_sink[p, sink, 'ramp_speed_down']
    * 60 * step_duration[d, t]
	* p_process_sink_max_capacity_coefficient[p, sink]
    * ( + p_entity_all_existing[p, d]
	    + sum {(p, d_invest, d) in edd_invest} v_invest[p, d_invest] * p_entity_unitsize[p]
        - sum {(p, d_divest) in pd_divest : p_years_d[d_divest] <= p_years_d[d]} v_divest[p, d_divest] * p_entity_unitsize[p]
	  )
  - ( if p in process_online_linear then v_shutdown_linear[p, d, t] * p_entity_unitsize[p] )  # To make sure that units can shutdown despite ramp limits.
  - ( if p in process_online_integer then v_shutdown_integer[p, d, t] * p_entity_unitsize[p] )  # To make sure that units can shutdown despite ramp limits.
;

s.t. reserve_process_upward{(p, r, 'up', n, d, t) in prundt} :
  + v_reserve[p, r, 'up', n, d, t] * p_entity_unitsize[p]
  <=
  ( if p in process_online then
      + ( + (if p in process_online_linear then v_online_linear[p, d, t])
	      + (if p in process_online_integer then v_online_integer[p, d, t])
		)
	    * p_process_reserve_upDown_node[p, r, 'up', n, 'max_share']
		* p_entity_unitsize[p]
    else
      + p_process_reserve_upDown_node[p, r, 'up', n, 'max_share']
        * (
            + p_entity_all_existing[p, d]
            + sum {(p, d_invest, d) in edd_invest} v_invest[p, d_invest] * p_entity_unitsize[p]
            - sum {(p, d_divest) in pd_divest : p_years_d[d_divest] <= p_years_d[d]} v_divest[p, d_divest] * p_entity_unitsize[p]
          )
    	* ( if (sum{(p, prof, 'upper_limit') in process__profile__profile_method} 1) then
	          ( + sum{(p, prof, 'upper_limit') in process__profile__profile_method} pdtProfile[prof, d, t] )
	        else 1
	      )
  )
;

s.t. reserve_process_downward{(p, r, 'down', n, d, t) in prundt} :
  + v_reserve[p, r, 'down', n, d, t] * p_entity_unitsize[p]
  <=
  + p_process_reserve_upDown_node[p, r, 'down', n, 'max_share']
    * ( + sum{(p, source, n) in process_source_sink} v_flow[p, source, n, d, t] * p_entity_unitsize[p]
        - ( + p_entity_all_existing[p, d]
            + sum {(p, d_invest, d) in edd_invest} v_invest[p, d_invest] * p_entity_unitsize[p]
            - sum {(p, d_divest) in pd_divest : p_years_d[d_divest] <= p_years_d[d]} v_divest[p, d_divest] * p_entity_unitsize[p]
          ) * ( if (sum{(p, prof, 'lower_limit') in process__profile__profile_method} 1) then
	              ( + sum{(p, prof, 'lower_limit') in process__profile__profile_method} pdtProfile[prof, d, t] )
	          )
	  )
;

s.t. maxInvestGroup_entity_period {(g, d) in gd_invest_period} :
  + sum{(g, e) in group_entity : (e, d) in ed_invest} v_invest[e, d] * p_entity_unitsize[e]
  <=
  + pdGroup[g, 'invest_max_period', d]
;

s.t. maxDivestGroup_entity_period {(g, d) in gd_divest_period} :
  + sum{(g, e) in group_entity : (e, d) in ed_divest} v_divest[e, d] * p_entity_unitsize[e]
  <=
  + pdGroup[g, 'retire_max_period', d]
;

s.t. minInvestGroup_entity_period {(g, d) in gd_invest_period} :
  + sum{(g, e) in group_entity : (e, d) in ed_invest} v_invest[e, d] * p_entity_unitsize[e]
  >=
  + pdGroup[g, 'invest_min_period', d]
;

s.t. minDivestGroup_entity_period {(g, d) in gd_divest_period} :
  + sum{(g, e) in group_entity : (e, d) in ed_divest} v_divest[e, d] * p_entity_unitsize[e]
  >=
  + pdGroup[g, 'retire_min_period', d]
;

s.t. maxInvestGroup_entity_total {g in g_invest_total, d in period_invest} :
  + sum{(g, e) in group_entity, d_invest in period_in_use : (e, d_invest, d) in edd_invest} v_invest[e, d_invest] * p_entity_unitsize[e]
  + sum{(g, e) in group_entity} p_entity_previously_invested_capacity[e, d]
  <=
  + p_group[g, 'invest_max_total']
;

s.t. maxInvestGroup_entity_cumulative {g in g_invest_cumulative, d in period_invest : p_group[g, 'cumulative_max_capacity']} :
  + sum{(g, e) in group_entity, d_invest in period_in_use : (e, d_invest, d) in edd_invest} v_invest[e, d_invest] * p_entity_unitsize[e]
  + sum{(g, e) in group_entity} p_entity_previously_invested_capacity[e, d]
#  - sum{(g, e) in group_entity, d_divest in period_in_use : e in entityDivest} v_divest[e, d_divest] * p_entity_unitsize[e]
#  - sum{(g, e) in group_entity : e in entityDivest} (if not p_model['solveFirst'] then p_entity_divested[e])
  + sum{(g, e) in group_entity} p_entity_all_existing[e, d]
  <=
  + p_group[g, 'cumulative_max_capacity']
;

s.t. minInvestGroup_entity_cumulative {g in g_invest_cumulative , d in period_invest: p_group[g, 'cumulative_min_capacity']} :
  + sum{(g, e) in group_entity, d_invest in period_in_use : (e, d_invest, d) in edd_invest} v_invest[e, d_invest] * p_entity_unitsize[e]
  + sum{(g, e) in group_entity} p_entity_previously_invested_capacity[e, d]
  - sum{(g, e) in group_entity, d_divest in period_in_use : e in entityDivest} v_divest[e, d_divest] * p_entity_unitsize[e]
  - sum{(g, e) in group_entity : e in entityDivest} (if not p_model['solveFirst'] then p_entity_divested[e])
  + sum{(g, e) in group_entity} p_entity_all_existing[e, d]
  <=
  + p_group[g, 'cumulative_min_capacity']
;

s.t. maxDivestGroup_entity_total {g in g_divest_total} :
  + sum{(g, e) in group_entity, d in period_in_use : e in entityDivest} v_divest[e, d] * p_entity_unitsize[e]
  + sum{(g, e) in group_entity : e in entityDivest} (if not p_model['solveFirst'] then p_entity_divested[e])
  <=
  + p_group[g, 'retire_max_total']
;

s.t. minInvestGroup_entity_total {g in g_invest_total, d in period_invest} :
  + sum{(g, e) in group_entity, d_invest in period_in_use : (e, d_invest, d) in edd_invest} v_invest[e, d_invest] * p_entity_unitsize[e]
  + sum{(g, e) in group_entity} p_entity_previously_invested_capacity[e, d]
  >=
  + p_group[g, 'invest_min_total']
;

s.t. minDivestGroup_entity_total {g in g_divest_total} :
  + sum{(g, e) in group_entity, d in period_in_use : e in entityDivest} v_divest[e, d] * p_entity_unitsize[e]
  + sum{(g, e) in group_entity : e in entityDivest} (if not p_model['solveFirst'] then p_entity_divested[e])
  >=
  + p_group[g, 'retire_min_total']
;

s.t. maxInvest_entity_period {(e, d) in ed_invest_period} :  # Covers both processes and nodes
  + v_invest[e, d] * p_entity_unitsize[e]
  <=
  + ed_invest_max_period[e, d]
;

# lifetime_method = no_investment: pin v_invest to zero past the
# first-period lifetime window. The model is free to invest during the
# first lifetime (subject to invest_method and invest_max_period caps);
# after that the asset retires once and cannot be rebuilt.
s.t. fix_v_invest_no_investment_eq {(e, d) in ed_invest_forbidden_no_investment} :
  v_invest[e, d] = 0
;

s.t. maxDivest_entity_period {(e, d) in ed_divest_period} :  # Covers both processes and nodes
  + v_divest[e, d] * p_entity_unitsize[e]
  <=
  + ed_divest_max_period[e, d]
;

s.t. minInvest_entity_period {(e, d)  in ed_invest_period} :  # Covers both processes and nodes
  + v_invest[e, d] * p_entity_unitsize[e]
  >=
  + ed_invest_min_period[e, d]
;

s.t. minDivest_entity_period {(e, d)  in ed_divest_period} :  # Covers both processes and nodes
  + v_divest[e, d] * p_entity_unitsize[e]
  >=
  + ed_divest_min_period[e, d]
;

s.t. maxInvest_entity_total {e in e_invest_total, d in period_invest} :  # Covers both processes and nodes
  + sum{(e, d_invest, d) in edd_invest} v_invest[e, d_invest] * p_entity_unitsize[e]
  + p_entity_previously_invested_capacity[e, d]
  <=
  + e_invest_max_total[e]
;

s.t. maxDivest_entity_total {e in e_divest_total} :  # Covers both processes and nodes
  + sum{(e, d) in ed_divest} v_divest[e, d] * p_entity_unitsize[e]
  + (if not p_model['solveFirst'] then p_entity_divested[e])
  <=
  + e_divest_max_total[e]
;

s.t. minInvest_entity_total {e in e_invest_total, d in period_invest} :  # Covers both processes and nodes
  + sum{(e, d_invest, d) in edd_invest} v_invest[e, d_invest] * p_entity_unitsize[e]
  + p_entity_previously_invested_capacity[e, d]
  >=
  + e_invest_min_total[e]
;

s.t. minDivest_entity_total {e in e_divest_total} :  # Covers both processes and nodes
  + sum{(e, d) in ed_divest} v_divest[e, d] * p_entity_unitsize[e]
  + (if not p_model['solveFirst'] then p_entity_divested[e])
  >=
  + e_divest_min_total[e]
;

s.t. maxCumulative_capacity {(e, d) in ed_invest_cumulative} :
  + p_entity_all_existing[e, d]
  + sum{(e, d_invest, d) in edd_invest} v_invest[e, d_invest] * p_entity_unitsize[e]
  - (if (e, d) in ed_divest then v_divest[e, d] * p_entity_unitsize[e])
  <=
  + ed_cumulative_max_capacity[e, d]
;

s.t. minCumulative_capacity {(e, d) in ed_invest_cumulative : ed_cumulative_min_capacity[e, d]} :
  + p_entity_all_existing[e, d]
  + sum{(e, d_invest, d) in edd_invest} v_invest[e, d_invest] * p_entity_unitsize[e]
  - (if (e, d) in ed_divest then v_divest[e, d] * p_entity_unitsize[e])
  >=
  + ed_cumulative_min_capacity[e, d]
;

# Commodity-ladder balance link: the sum over tiers of v_trade (scaled by
# p_commodity_unitsize) equals the branch-weighted realized-timeline MWh
# that the processes buy from / sell to the (c, n) commodity node during
# period d.  The aggregation mirrors the pdtCommodity price term (noEff
# flow, eff flow × slope, and the min_load_efficiency online section) so
# that a single-tier ∞-quantity ladder at the same price is bit-identical
# to the legacy objective term.
s.t. commodity_ladder_balance {(c, n, d) in cnd_ladder} :
  + sum {(c, n, d, i) in cndi_ladder} v_trade[c, n, d, i] * p_commodity_unitsize[c]
  =
  # Buying: flows INTO n (n is sink of a process)
  + sum {(p, n, sink) in process_source_sink_noEff, (d, t) in dt}
      ( + v_flow[p, n, sink, d, t] * p_entity_unitsize[p] )
        * step_duration[d, t] * p_rp_cost_weight[d, t] * pdt_branch_weight[d, t]
  + sum {(p, n, sink) in process_source_sink_eff, (d, t) in dt}
      ( + v_flow[p, n, sink, d, t] * p_entity_unitsize[p]
            * pdtProcess_slope[p, d, t]
            * (if p in process_unit then p_process_sink_flow_coefficient[p, sink] / p_process_source_flow_coefficient[p, n] else 1)
        + (if (p, 'min_load_efficiency') in process__ct_method then
            + ( + (if p in process_online_linear then v_online_linear[p, d, t])
                + (if p in process_online_integer then v_online_integer[p, d, t])
              )
              * pdtProcess_section[p, d, t]
              * p_entity_unitsize[p]
          )
      ) * step_duration[d, t] * p_rp_cost_weight[d, t] * pdt_branch_weight[d, t]
  # Selling: flows OUT of n (n is source of a process)
  - sum {(p, source, n) in process_source_sink, (d, t) in dt}
      ( + v_flow[p, source, n, d, t] * p_entity_unitsize[p] )
        * step_duration[d, t] * p_rp_cost_weight[d, t] * pdt_branch_weight[d, t]
;

# Rolling-aware ANNUAL tier cap for price_ladder_annual commodities.
# p_ladder_ann_quantity[c, i, d] is user-provided per-year MWh for
# period d.  Across a rolling run that partitions one period into several
# rolls, each roll is allowed its share f_d_k[d] of the annual cap,
# minus whatever prior rolls already realized into d
# (p_ladder_cum_realized_mwh[c, i, d]).
#
# On a single solve covering all of period d the accumulators are zero
# and f_d_k[d] = 1.0, reducing the RHS to p_ladder_ann_quantity — the
# pre-refactor form bit-for-bit.  When a prior roll overspent within
# d (cum_realized > f_d_k * cap), the RHS would go negative and LHS >= 0
# is infeasible; the overspent-override below handles that case.
s.t. ladder_tier_cap_annual_roll
    {c in commodity_with_ladder_annual, d in period_in_use, (c, i) in commodity__tier_ann
     : p_ladder_ann_quantity[c, i, d] < 1e29
       && f_d_k[d] * p_ladder_ann_quantity[c, i, d] >= p_ladder_cum_realized_mwh[c, i, d]} :
  + sum {(c, n) in commodity_node} v_trade[c, n, d, i] * p_commodity_unitsize[c]
  <=
  + p_ladder_ann_quantity[c, i, d] * f_d_k[d]
  - p_ladder_cum_realized_mwh[c, i, d]
;

# Annual overspent override — whenever f_d_k[d] × cap < prior realized
# MWh for (c, i, d), the main cap's RHS would be negative.  Force
# v_trade = 0 for that period/tier on every commodity_node to keep the
# LP feasible.  The filter predicate on the main constraint already
# ensures these two constraints' activation regions are disjoint.
s.t. ladder_tier_cap_annual_overspent
    {c in commodity_with_ladder_annual, (c, n) in commodity_node, d in period_in_use, (c, i) in commodity__tier_ann
     : p_ladder_ann_quantity[c, i, d] < 1e29
       && f_d_k[d] * p_ladder_ann_quantity[c, i, d] < p_ladder_cum_realized_mwh[c, i, d]} :
  + v_trade[c, n, d, i]
  <=
  + 0
;

# Rolling-aware CUMULATIVE tier cap for price_ladder_cumulative commodities.
# The cap is one total across the whole model horizon (not per-year).
# LHS is current-roll v_trade × unitsize summed over periods and nodes
# (realized-timeline MWh); RHS partitions the cap by the sum of f_d_k[d]
# over periods in this roll's view, minus MWh already realized across
# ALL prior rolls (the p_ladder_cum_realized_mwh sum ranges over
# periodAll, not period_in_use, because prior rolls may have realized
# periods that are not in this roll's window).  On a single full solve
# the sum of f_d_k reduces to the fraction of the model horizon
# covered by dt, matching the pre-refactor years_represented /
# period_share derivation in a non-binding regime.
s.t. ladder_tier_cap_cumulative_roll
    {(c, i) in ci_ladder_cumulative
     : p_ladder_cum_quantity[c, i] < 1e29
       && (sum {d in period_in_use} f_d_k[d]) * p_ladder_cum_quantity[c, i]
          >= sum {d in periodAll} p_ladder_cum_realized_mwh[c, i, d]} :
  + sum {(c, n) in commodity_node, d in period_in_use}
      v_trade[c, n, d, i] * p_commodity_unitsize[c]
  <=
  + p_ladder_cum_quantity[c, i] * (sum {d in period_in_use} f_d_k[d])
  - sum {d in periodAll} p_ladder_cum_realized_mwh[c, i, d]
;

# Cumulative overspent override — mirrors the annual overspent branch.
# When prior realized MWh summed across all periods exceeds the running
# cap allotment, force v_trade = 0 on every (c, n, d, i) with tier i in
# the cumulative ladder.  The filter predicates keep the main cap and
# this override on disjoint activation regions.
s.t. ladder_tier_cap_cumulative_overspent
    {c in commodity_with_ladder_cumulative, (c, n) in commodity_node, d in period_in_use, (c, i) in commodity__tier_cum
     : p_ladder_cum_quantity[c, i] < 1e29
       && (sum {d2 in period_in_use} f_d_k[d2]) * p_ladder_cum_quantity[c, i]
          < sum {d2 in periodAll} p_ladder_cum_realized_mwh[c, i, d2]} :
  + v_trade[c, n, d, i]
  <=
  + 0
;

# Infinite-tier bound.  When a tier has its quantity parameter set to
# +Infinity (1e30 sentinel) the variable is otherwise unbounded; cap it
# with the global p_max_flow_for_unconstrained_variables (MW) times the
# realized-timeline hours (complete_period_share_of_year[d] * 8760).
# Split per ladder method so each reads its own quantity parameter.
# TODO decision #1: a tighter bound based on sum of connected
# process p_flow_max × p_entity_unitsize would be preferable but requires
# a commodity__node→process connectivity set that is not precomputed;
# postponed to a follow-up commit.
s.t. ladder_tier_cap_infinite_cum {(c, n, d, i) in cndi_ladder_cum
    : p_ladder_cum_quantity[c, i] >= 1e29} :
  + v_trade[c, n, d, i] * p_commodity_unitsize[c]
  <=
  + p_unconstrained_flow_cap * 8760 * complete_period_share_of_year[d]
;

s.t. ladder_tier_cap_infinite_ann {(c, n, d, i) in cndi_ladder_ann
    : p_ladder_ann_quantity[c, i, d] >= 1e29} :
  + v_trade[c, n, d, i] * p_commodity_unitsize[c]
  <=
  + p_unconstrained_flow_cap * 8760 * complete_period_share_of_year[d]
;

s.t. maxCumulative_flow_solve {g in group : p_group[g, 'max_cumulative_flow']} :
  # LHS: total physical flow over the full solve horizon.  step_duration * p_rp_cost_weight
  # / complete_period_share_of_year mirrors the objective's annualization; * p_years_represented_d
  # then sums each period's annual flow across the multi-year horizon.
  + sum{(g, p, n) in group_process_node, (d, t) in dt} (
      # n is sink
      + sum {(p, source, n) in process_source_sink} (
          + v_flow[p, source, n, d, t] * p_entity_unitsize[p]
	    )
      # n is source
      - sum {(p, n, sink) in process_source_sink_eff } (
          + v_flow[p, n, sink, d, t] * p_entity_unitsize[p]
	           * pdtProcess_slope[p, d, t]
          + (if (p, 'min_load_efficiency') in process__ct_method then
		      + ( + (if p in process_online_linear then v_online_linear[p, d, t])
				  + (if p in process_online_integer then v_online_integer[p, d, t])
				)
		    	* pdtProcess_section[p, d, t]
			    * p_entity_unitsize[p]
		    )
        )
      - sum {(p, n, sink) in process_source_sink_noEff } (
          + v_flow[p, n, sink, d, t] * p_entity_unitsize[p]
		)
	) * step_duration[d, t] * p_rp_cost_weight[d, t]
	  * p_years_represented_d[d] / complete_period_share_of_year[d]
	<=
  # RHS: avg_MW * total horizon hours across all periods and represented years.
  + p_group[g, 'max_cumulative_flow']
      * sum {d in period_in_use} (p_years_represented_d[d] * 8760)
;

s.t. minCumulative_flow_solve {g in group : p_group[g, 'min_cumulative_flow']} :
  # LHS: total physical flow over the full solve horizon.  step_duration * p_rp_cost_weight
  # / complete_period_share_of_year mirrors the objective's annualization; * p_years_represented_d
  # then sums each period's annual flow across the multi-year horizon.
  + sum{(g, p, n) in group_process_node, (d, t) in dt} (
      # n is sink
      + sum {(p, source, n) in process_source_sink} (
          + v_flow[p, source, n, d, t] * p_entity_unitsize[p]
	    )
      # n is source
      - sum {(p, n, sink) in process_source_sink_eff } (
          + v_flow[p, n, sink, d, t] * p_entity_unitsize[p]
	           * pdtProcess_slope[p, d, t]
          + (if (p, 'min_load_efficiency') in process__ct_method then
		      + ( + (if p in process_online_linear then v_online_linear[p, d, t])
				  + (if p in process_online_integer then v_online_integer[p, d, t])
				)
		    	* pdtProcess_section[p, d, t]
			    * p_entity_unitsize[p]
		    )
        )
      - sum {(p, n, sink) in process_source_sink_noEff } (
          + v_flow[p, n, sink, d, t] * p_entity_unitsize[p]
		)
	) * step_duration[d, t] * p_rp_cost_weight[d, t]
	  * p_years_represented_d[d] / complete_period_share_of_year[d]
	>=
  # RHS: avg_MW * total horizon hours across all periods and represented years.
  + p_group[g, 'min_cumulative_flow']
      * sum {d in period_in_use} (p_years_represented_d[d] * 8760)
;

s.t. maxCumulative_flow_period {g in group, d in period_in_use : pdGroup[g, 'max_cumulative_flow', d]} :
  # LHS: annualized flow within the period.  step_duration * p_rp_cost_weight
  # / complete_period_share_of_year mirrors the objective's annualization so the LHS
  # matches the annual flow volumes the solver sees as per-year costs.
  + sum{(g, p, n) in group_process_node, (d, t) in dt} (
      # n is sink
      + sum {(p, source, n) in process_source_sink} (
          + v_flow[p, source, n, d, t] * p_entity_unitsize[p]
	    )
      # n is source
      - sum {(p, n, sink) in process_source_sink_eff } (
          + v_flow[p, n, sink, d, t] * p_entity_unitsize[p]
	           * pdtProcess_slope[p, d, t]
          + (if (p, 'min_load_efficiency') in process__ct_method then
			  + ( + (if p in process_online_linear then v_online_linear[p, d, t])
			      + (if p in process_online_integer then v_online_integer[p, d, t])
				)
		    	* pdtProcess_section[p, d, t]
			    * p_entity_unitsize[p]
		    )
        )
      - sum {(p, n, sink) in process_source_sink_noEff } (
          + v_flow[p, n, sink, d, t] * p_entity_unitsize[p]
		)
	)  * step_duration[d, t] * p_rp_cost_weight[d, t]
	   / complete_period_share_of_year[d]
	<=
  # RHS: avg_MW * annual hours (per-period limit applied to one year's worth of flow).
  + pdGroup[g, 'max_cumulative_flow', d]
      * 8760
;

s.t. minCumulative_flow_period {g in group, d in period_in_use : pdGroup[g, 'min_cumulative_flow', d]} :
  # LHS: annualized flow within the period.  step_duration * p_rp_cost_weight
  # / complete_period_share_of_year mirrors the objective's annualization so the LHS
  # matches the annual flow volumes the solver sees as per-year costs.
  + sum{(g, p, n) in group_process_node, (d, t) in dt} (
      # n is sink
      + sum {(p, source, n) in process_source_sink} (
          + v_flow[p, source, n, d, t] * p_entity_unitsize[p]
	    )
      # n is source
      - sum {(p, n, sink) in process_source_sink_eff } (
          + v_flow[p, n, sink, d, t] * p_entity_unitsize[p]
	           * pdtProcess_slope[p, d, t]
          + (if (p, 'min_load_efficiency') in process__ct_method then
			  + ( + (if p in process_online_linear then v_online_linear[p, d, t])
			      + (if p in process_online_integer then v_online_integer[p, d, t])
				)
		        * pdtProcess_section[p, d, t]
			   	* p_entity_unitsize[p]
		    )
        )
      - sum {(p, n, sink) in process_source_sink_noEff } (
          + v_flow[p, n, sink, d, t] * p_entity_unitsize[p]
		)
	) * step_duration[d, t] * p_rp_cost_weight[d, t]
	  / complete_period_share_of_year[d]
	>=
  # RHS: avg_MW * annual hours (per-period limit applied to one year's worth of flow).
  + pdGroup[g, 'min_cumulative_flow', d]
      * 8760
;

s.t. maxInstant_flow {(g, d, t) in gdt_maxInstantFlow} :
  + sum{(g, p, n) in group_process_node} (
      # n is sink
      + sum {(p, source, n) in process_source_sink} (
          + v_flow[p, source, n, d, t] * p_entity_unitsize[p]
	    )
      # n is source
      - sum {(p, n, sink) in process_source_sink_eff } (
          + v_flow[p, n, sink, d, t] * p_entity_unitsize[p]
	           * pdtProcess_slope[p, d, t]
          + (if (p, 'min_load_efficiency') in process__ct_method then
			  + ( + (if p in process_online_linear then v_online_linear[p, d, t])
			      + (if p in process_online_integer then v_online_integer[p, d, t])
				)
		    	* pdtProcess_section[p, d, t]
			    * p_entity_unitsize[p]
		    )
        )
      - sum {(p, n, sink) in process_source_sink_noEff }
          + v_flow[p, n, sink, d, t] * p_entity_unitsize[p]
	)
	<=
  + pdtGroup[g, 'max_instant_flow', d, t]
;

s.t. minInstant_flow {(g, d, t) in gdt_minInstantFlow} :
  + sum{(g, p, n) in group_process_node} (
      # n is sink
      + sum {(p, source, n) in process_source_sink} (
          + v_flow[p, source, n, d, t] * p_entity_unitsize[p]
	    )
      # n is source
      - sum {(p, n, sink) in process_source_sink_eff } (
          + v_flow[p, n, sink, d, t] * p_entity_unitsize[p]
	           * pdtProcess_slope[p, d, t]
          + (if (p, 'min_load_efficiency') in process__ct_method then
			  + ( + (if p in process_online_linear then v_online_linear[p, d, t])
			      + (if p in process_online_integer then v_online_integer[p, d, t])
				)
	    	    * pdtProcess_section[p, d, t]
		    	* p_entity_unitsize[p]
		    )
        )
      - sum {(p, n, sink) in process_source_sink_noEff }
          + v_flow[p, n, sink, d, t] * p_entity_unitsize[p]
	)
	>=
  + pdtGroup[g, 'min_instant_flow', d, t]
;

s.t. inertia_constraint {g in groupInertia, (d, t) in dt} :
  + sum {(p, source, sink) in process_source_sink : (p, source) in process_source && (g, source) in group_node && p_process_source[p, source, 'inertia_constant']}
    ( + (if p in process_online_linear then v_online_linear[p, d, t])
      + (if p in process_online_integer then v_online_integer[p, d, t])
	  + (if p not in process_online then v_flow[p, source, sink, d, t])
	) * p_process_source[p, source, 'inertia_constant'] * p_entity_unitsize[p]
  + sum {(p, source, sink) in process_source_sink : (p, sink) in process_sink && (g, sink) in group_node && p_process_sink[p, sink, 'inertia_constant']}
    ( + (if p in process_online_linear then v_online_linear[p, d, t])
      + (if p in process_online_integer then v_online_integer[p, d, t])
	  + (if p not in process_online then v_flow[p, source, sink, d, t])
    ) * p_process_sink[p, sink, 'inertia_constant'] * p_entity_unitsize[p]
  + vq_inertia[g, d, t] * pdGroup[g, 'inertia_limit', d]
  >=
  + pdGroup[g, 'inertia_limit', d]
;

s.t. co2_max_period{g in group_co2_max_period, d in period_in_use} :
  + sum{(g, c, n, d) in group_commodity_node_period_co2_period }
    (
      + p_commodity[c, 'co2_content'] / 1000
        * (
            # CO2 increases.  step_duration * p_rp_cost_weight / complete_period_share_of_year
            # mirrors the objective's annualization so the LHS matches the
            # flow volumes the solver sees as full-year costs.
            + sum {(p, n, sink) in process_source_sink_noEff, (d, t) in dt }
              ( + v_flow[p, n, sink, d, t] * p_entity_unitsize[p]
                  * step_duration[d, t] * p_rp_cost_weight[d, t] )
            + sum {(p, n, sink) in process_source_sink_eff, (d, t) in dt }
              ( + v_flow[p, n, sink, d, t] * p_entity_unitsize[p]
                  * step_duration[d, t] * p_rp_cost_weight[d, t]
                  * pdtProcess_slope[p, d, t]
                  * (if p in process_unit then p_process_sink_flow_coefficient[p, sink] / p_process_source_flow_coefficient[p, n] else 1)
                + ( if (p, 'min_load_efficiency') in process__ct_method then
                    + ( + (if p in process_online_linear then v_online_linear[p, d, t])
                        + (if p in process_online_integer then v_online_integer[p, d, t])
                      )
                      * pdtProcess_section[p, d, t]
                      * p_entity_unitsize[p]
                  )
              )
            # CO2 removals
            - sum {(p, source, n) in process_source_sink, (d, t) in dt }
              ( + v_flow[p, source, n, d, t] * p_entity_unitsize[p]
                  * step_duration[d, t] * p_rp_cost_weight[d, t] )
          ) / complete_period_share_of_year[d]
    )
  <=
  + pdGroup[g, 'co2_max_period', d] / 1000
;

# Rolling-aware model-wide CO2 cap.  Mirrors the ladder_tier_cap_cumulative_roll
# pattern (commit f78d2de Bug #1 fix): the LHS is the current roll's physical
# window emissions (step_duration * p_rp_cost_weight only — no year projection),
# and the RHS partitions the user-declared horizon cap by the sum of f_d_k[d]
# over periods in this roll's view, minus realized-timeline tonnes already
# emitted across ALL prior rolls (p_co2_cum_realized_tonnes sum ranges over
# periodAll because prior rolls may have realized periods that are not in this
# roll's window).  Single full solve of a one-period scenario: f_d_k[d] = 1.0
# and the accumulator is zero, so the RHS reduces to p_group['co2_max_total']
# / 1000 matching the pre-refactor form in physical tonnes.  (For multi-period
# single solves, the new form caps per-period emissions at cap × f_d_k[d]
# partitioned across periods — a semantic change from the old year-projected
# form; mirrors the ladder cumulative cap's design.)
#
# No overspent-override variant exists for CO2 (no slack/penalty variable is
# declared for this constraint, unlike the ladder's *_overspent branches
# that force v_trade = 0 to absorb an overspend).  If a prior rolling run
# accidentally overspends the cap, the next roll's RHS goes negative and the
# LP becomes infeasible.  Rolling CO2 cap users should choose the cap loose
# enough that a single roll cannot overspend; a future follow-up could add a
# slack/penalty variable if a softer override is needed.
s.t. co2_max_total
    {g in group_co2_max_total
     : (sum {d in period_in_use} f_d_k[d]) * p_group[g, 'co2_max_total'] / 1000
        >= sum {d in periodAll} p_co2_cum_realized_tonnes[g, d]} :
  + sum{(g, c, n) in group_commodity_node_period_co2_total }
    (
      + p_commodity[c, 'co2_content'] / 1000
        * (
            # CO2 increases.  step_duration * p_rp_cost_weight gives the
            # physical sim-window MWh of flow; multiplied by co2_content/1000
            # yields sim-window tonnes.  No year-projection (years_represented
            # / period_share) — that is folded into the RHS via f_d_k[d].
            + sum {(p, n, sink) in process_source_sink_noEff, (d, t) in dt }
              ( + v_flow[p, n, sink, d, t] * p_entity_unitsize[p]
                  * step_duration[d, t] * p_rp_cost_weight[d, t] )
            + sum {(p, n, sink) in process_source_sink_eff, (d, t) in dt }
              ( + v_flow[p, n, sink, d, t] * p_entity_unitsize[p]
                  * step_duration[d, t] * p_rp_cost_weight[d, t]
                  * pdtProcess_slope[p, d, t]
                  * (if p in process_unit then p_process_sink_flow_coefficient[p, sink] / p_process_source_flow_coefficient[p, n] else 1)
                + ( if (p, 'min_load_efficiency') in process__ct_method then
                    + ( + (if p in process_online_linear then v_online_linear[p, d, t])
                        + (if p in process_online_integer then v_online_integer[p, d, t])
                      )
                      * pdtProcess_section[p, d, t]
                      * p_entity_unitsize[p]
				  )
              )
          # CO2 removals
            - sum {(p, source, n) in process_source_sink, (d, t) in dt }
              ( + v_flow[p, source, n, d, t] * p_entity_unitsize[p]
                  * step_duration[d, t] * p_rp_cost_weight[d, t]
              )
          )
    )
  <=
  + p_group[g, 'co2_max_total'] / 1000 * (sum {d in period_in_use} f_d_k[d])
  - sum {d in periodAll} p_co2_cum_realized_tonnes[g, d]
;

s.t. non_sync_constraint{g in groupNonSync, (d, t) in dt} :
# Sum all incoming non-synchronous flows to the group nodes (and possibly decrease them with penalty)
# Agent 5c: every term scaled by inv_group_cap[g, d].  Slack term's
# * group_capacity_for_scaling cancels with inv_group_cap, leaving just
# step_duration on the slack column.
  # Include incoming non-sync flows if they come from outside of the node group (and ignore sync and non-sync flows within the node group)
  + sum {(p, source, sink) in process_source_sink : (g, sink) in group_node && (p, sink) in process__sink_nonSync && (p, g) not in process__group_inside_group_nonSync}
    ( + v_flow[p, source, sink, d, t]
	    * p_entity_unitsize[p]
		* step_duration[d, t]
		* inv_group_cap[g, d] )
  # Assumes that exogenous inflows are always non-synchronous (there is no separate parameter for this)
  + sum {(g, n) in group_node} p_positive_inflow[n,d,t] * inv_group_cap[g, d]
  - vq_non_synchronous[g, d, t] * step_duration[d, t]
  <=
# Sum all outgoing flows from the group nodes and multiply that with the non-sync limit
  ( + sum {(p, source, sink) in process_source_sink_noEff : (g, source) in group_node && (p,g) not in process__group_inside_group_nonSync}
      ( + v_flow[p, source, sink, d, t]
		  * p_entity_unitsize[p]
	  ) * step_duration[d, t]
    + sum {(p, source, sink) in process_source_sink_eff: (g, source) in group_node && (p,g) not in process__group_inside_group_nonSync}
      ( + v_flow[p, source, sink, d, t]
	      * pdtProcess_slope[p, d, t]
	      * ( if p in process_unit then p_process_sink_flow_coefficient[p, sink] / p_process_source_flow_coefficient[p, source]
	          else 1 )
  	    + ( if (p, 'min_load_efficiency') in process__ct_method then
	        + ( + (if p in process_online_linear then v_online_linear[p, d, t])
	            + (if p in process_online_integer then v_online_integer[p, d, t])
	          )
		      * pdtProcess_section[p, d, t]
	      )
        # -(if (p,g) in process__group_inside_group_nonSync then v_flow[p, source, sink, d, t] else 0)
	  )	* p_entity_unitsize[p]
		* step_duration[d, t]
    # Add exogenous outflow (demand)
    + sum {(g, n) in group_node} -p_negative_inflow[n,d,t]
  ) * pdGroup[g, 'non_synchronous_limit', d] * inv_group_cap[g, d]
;

s.t. capacityMargin {g in groupCapacityMargin, (d, t, t_previous, t_previous_within_timeset, d_previous, t_previous_within_solve) in dtttdt : d in period_invest} :
  # Agent 5c: every term scaled by inv_group_cap[g, d].  Slack term's
  # * group_capacity_for_scaling cancels with inv_group_cap.
  # profile limited units producing to a node in the group (based on available capacity)
  + sum {(p, source, sink, f, m) in process__source__sink__profile__profile_method
         : m = 'upper_limit' || m = 'fixed'
           && (p, sink) in process_sink
		   && (g, sink) in group_node
		   && sink not in nodeState
		   && p in process_unit
		}
    ( + pdtProfile[f, d, t]
        * ( + p_entity_all_existing[p, d]
            + sum {(p, d_invest, d) in edd_invest} v_invest[p, d_invest] * p_entity_unitsize[p]
            - sum {(p, d_divest) in pd_divest : p_years_d[d_divest] <= p_years_d[d]} v_divest[p, d_divest] * p_entity_unitsize[p]
	      )
    ) * inv_group_cap[g, d]
  # capacity limited units producing to a node in the group (based on available capacity)
  + sum {(p, source, sink) in process_source_sink
         : (p, sink) in process_sink
	       && sum {(p, source, sink, f, m) in process__source__sink__profile__profile_method : m = 'upper_limit' || m = 'fixed'} 1 = 0
		   && (p, sink) in process_sink
  		   && (g, sink) in group_node
		   && sink not in nodeState
		   && p in process_unit
		}
	(
      + ( + p_entity_all_existing[p, d]
          + sum {(p, d_invest, d) in edd_invest} v_invest[p, d_invest] * p_entity_unitsize[p]
          - sum {(p, d_divest) in pd_divest : p_years_d[d_divest] <= p_years_d[d]} v_divest[p, d_divest] * p_entity_unitsize[p]
        )
	) * inv_group_cap[g, d]
  # profile or capacity limited units consuming from a node in the group (as they consume in any given time step)
  - sum {(p, source, sink) in process_source_sink
         : (p, source) in process_source
		   && (g, source) in group_node
		   && source not in nodeState
		   && p in process_unit
		}
    ( + if (p, source, sink) in process_source_sink_eff then
        ( + v_flow[p, source, sink, d, t] * p_entity_unitsize[p]
	          * pdtProcess_slope[p, d, t]
		      * p_process_sink_flow_coefficient[p, sink] / p_process_source_flow_coefficient[p, source]
          + (if (p, 'min_load_efficiency') in process__ct_method then
			  + ( + (if p in process_online_linear then v_online_linear[p, d, t])
			      + (if p in process_online_integer then v_online_integer[p, d, t])
				)
			    * pdtProcess_section[p, d, t]
			    * p_entity_unitsize[p]
	  	    )
	    )
	  + if (p, source, sink) in process_source_sink_noEff then
        ( + v_flow[p, source, sink, d, t] * p_entity_unitsize[p]
        )
	) * inv_group_cap[g, d]
  + vq_capacity_margin[g, d]
  >=
  + sum {(g, n) in group_node : n not in nodeState}
    ( - (if (n, 'no_inflow') not in node__inflow_method then pdtNodeInflow[n, d, t] / step_duration[d, t])
  ) * inv_group_cap[g, d]
  + pdGroup[g, 'capacity_margin', d] * inv_group_cap[g, d]
;

s.t. group_loss_share_constraint{(g,n) in group_node, (d,t) in dt: g in group_loss_share && n in nodeBalance}:
  # Agent 5c: row scaler for this per-group constraint is group_cap,
  # so every term is multiplied by inv_group_cap[g, d].  The LHS
  # `vq_state_up_* * node_cap` retains a residual factor
  # node_cap[n] / group_cap[g] — physically meaningful: a shortfall at
  # node n occupies that fraction of the group's aggregate capacity
  # when converted into the group-level slack share.  The RHS slack
  # term's * group_cap cancels with inv_group_cap.
  + vq_state_up[n,d,t] * node_capacity_for_scaling[n, d] * inv_group_cap[g, d]
  =
  + p_state_slack_share[g,n,d,t] * vq_state_up_group[g,d,t];

s.t. non_anticipativity_storage_use{n in nodeState, (d,b) in period__branch, (d,t) in dt_non_anticipativity:
      d != b && b in period_in_use && exists{(g,n) in group_node: g in groupStochastic} 1}:
        # n is sink
        + sum {(p, source, n) in process_source_sink} (
            + v_flow[p, source, n, d, t] * p_entity_unitsize[p] * step_duration[d, t]
        )
        # n is source
        - sum {(p, n, sink) in process_source_sink_eff } (
            + v_flow[p, n, sink, d, t] * p_entity_unitsize[p]
              * pdtProcess_slope[p, d, t]
            * (if p in process_unit then p_process_sink_flow_coefficient[p, sink] / p_process_source_flow_coefficient[p, n] else 1)
            + (if (p, 'min_load_efficiency') in process__ct_method then
              + ( + (if p in process_online_linear then v_online_linear[p, d, t])
                  + (if p in process_online_integer then v_online_integer[p, d, t])
              )
                * pdtProcess_section[p, d, t]
              * p_entity_unitsize[p]
          )
          ) * step_duration[d, t]
        - sum {(p, n, sink) in process_source_sink_noEff}
          ( + v_flow[p, n, sink, d, t] * p_entity_unitsize[p]
          ) * step_duration[d, t]

        =
          # n is sink
        + sum {(p, source, n) in process_source_sink} (
            + v_flow[p, source, n, b, t] * p_entity_unitsize[p] * step_duration[b, t]
        )
        # n is source
        - sum {(p, n, sink) in process_source_sink_eff } (
            + v_flow[p, n, sink, b, t] * p_entity_unitsize[p]
              * pdtProcess_slope[p, b, t]
            * (if p in process_unit then p_process_sink_flow_coefficient[p, sink] / p_process_source_flow_coefficient[p, n] else 1)
            + (if (p, 'min_load_efficiency') in process__ct_method then
              + ( + (if p in process_online_linear then v_online_linear[p, b, t])
                  + (if p in process_online_integer then v_online_integer[p, b, t])
              )
                * pdtProcess_section[p, b, t]
              * p_entity_unitsize[p]
          )
          ) * step_duration[b, t]
        - sum {(p, n, sink) in process_source_sink_noEff}
          ( + v_flow[p, n, sink, b, t] * p_entity_unitsize[p]
          ) * step_duration[b, t]
        ;

s.t. non_anticipativity_online_integer{p in process_online_integer, (d,b) in period__branch, (d,t) in dt_non_anticipativity: b in period_in_use}:
  + v_online_integer[p,d,t]
  =
  + v_online_integer[p,b,t]
;
s.t. non_anticipativity_online_linear{p in process_online_linear, (d,b) in period__branch, (d,t) in dt_non_anticipativity: b in period_in_use}:
  + v_online_linear[p,d,t]
  =
  + v_online_linear[p,b,t]
;
s.t. non_anticipativity_reserve{(p, r, ud, n) in process_reserve_upDown_node_active, (d,b) in period__branch, (d,t) in dt_non_anticipativity: sum{(r, ud, g) in reserve__upDown__group} 1 && b in period_in_use}:
  + v_reserve[p, r, ud, n, d, t]
  =
  + v_reserve[p, r, ud, n, b, t]
;
param rest := gmtime();
printf "Timer - Rest: %ss\n", rest - reserves;
printf "Timer - Total constraints: %ss\n\n", rest - setup;
printf '%s,%s\n', 'rest', rest - reserves >> mod_phases_file;
printf '%s,%s\n', 'total_constraints', rest - setup >> mod_phases_file;

# ==========================================================
# Outputs that do not depend on solver values — emitted in
# phase 1 so glpsol phase 3 can be retired for HiGHS runs.
# Static (write-once) blocks write to input/; per-solve
# blocks write to solve_data/.  See ARCHITECTURE.md
# "Solver outputs: folder layout".
#
# Gated to phase == 'read' so multi-solve runs don't double-
# append: glpsol re-processes this file in phase 3 (``-r``)
# too, and without the guard every per-solve printf would
# fire twice.
# ==========================================================
if 'read' in phase then {

# Parameters with (d, t) dimensions
# Write step_duration
if p_model["solveFirst"] == 1 then {
  printf "solve,period,time,value" > "solve_data/p_step_duration.csv";
}
for {s in solve_current, (d, t) in dt_realize_dispatch} {
    printf "\n%s,%s,%s,%.8g", s, d, t, step_duration[d, t] >> "solve_data/p_step_duration.csv";
}

# Write p_rp_cost_weight (representative-period cost weight, used for annualization
# of counts like startups that aren't already carrying step_duration × rp scaling).
if p_model["solveFirst"] == 1 then {
  printf "solve,period,time,value" > "solve_data/p_rp_cost_weight.csv";
}
for {s in solve_current, (d, t) in dt_realize_dispatch} {
    printf "\n%s,%s,%s,%.8g", s, d, t, p_rp_cost_weight[d, t] >> "solve_data/p_rp_cost_weight.csv";
}

# Write p_flow_min — wide-format dump for read_parameters.py;
# retargeted to solve__p_flow_min.csv to break the path collision
# with the LONG-format CSV that Python preprocessing writes for
# mod's table-data-IN (see preprocessing/process_arc_unions.py).
if p_model["solveFirst"] == 1 then {
  printf "solve,period,time" > "solve_data/solve__p_flow_min.csv";
  for {(p, source, sink) in process_source_sink} {printf ",%s", p >> "solve_data/solve__p_flow_min.csv";}
  printf "\n,," >> "solve_data/solve__p_flow_min.csv";
  for {(p, source, sink) in process_source_sink} {printf ",%s", source >> "solve_data/solve__p_flow_min.csv";}
  printf "\n,," >> "solve_data/solve__p_flow_min.csv";
  for {(p, source, sink) in process_source_sink} {printf ",%s", sink >> "solve_data/solve__p_flow_min.csv";}
}
for {s in solve_current, (d, t) in dt_realize_dispatch} {
    printf "\n%s,%s,%s", s, d, t >> "solve_data/solve__p_flow_min.csv";
    for {(p, source, sink) in process_source_sink} {
        printf ",%.8g", p_flow_min[p, source, sink, d, t] >> "solve_data/solve__p_flow_min.csv";
    }
}

# Write p_flow_max — same path-collision retarget as p_flow_min above.
if p_model["solveFirst"] == 1 then {
  printf "solve,period,time" > "solve_data/solve__p_flow_max.csv";
  for {(p, source, sink) in process_source_sink} {printf ",%s", p >> "solve_data/solve__p_flow_max.csv";}
  printf "\n,," >> "solve_data/solve__p_flow_max.csv";
  for {(p, source, sink) in process_source_sink} {printf ",%s", source >> "solve_data/solve__p_flow_max.csv";}
  printf "\n,," >> "solve_data/solve__p_flow_max.csv";
  for {(p, source, sink) in process_source_sink} {printf ",%s", sink >> "solve_data/solve__p_flow_max.csv";}
}
for {s in solve_current, (d, t) in dt_realize_dispatch} {
    printf "\n%s,%s,%s", s, d, t >> "solve_data/solve__p_flow_max.csv";
    for {(p, source, sink) in process_source_sink} {
        printf ",%.8g", p_flow_max[p, source, sink, d, t] >> "solve_data/solve__p_flow_max.csv";
    }
}

# Write pdtProcess_slope (post-solve realized values, wide format).
# Mod's input now reads solve_data/pdtProcess_slope.csv (long) written by
# Python preprocessing, so this output goes to solve__pdtProcess_slope.csv
# (read_parameters.py:47 + cumulative_handoffs.py:449 read here).
if p_model["solveFirst"] == 1 then {
  printf "solve,period,time" > "solve_data/solve__pdtProcess_slope.csv";
  for {p in process} {printf ",%s", p >> "solve_data/solve__pdtProcess_slope.csv";}
}
for {s in solve_current, (d, t) in dt_realize_dispatch} {
    printf "\n%s,%s,%s", s, d, t >> "solve_data/solve__pdtProcess_slope.csv";
    for {p in process} {
        printf ",%.8g", pdtProcess_slope[p, d, t] >> "solve_data/solve__pdtProcess_slope.csv";
    }
}

# Write pdtProcess_section (post-solve realized values, wide format).
# Mod's input now reads solve_data/pdtProcess_section.csv (long) written by
# Python preprocessing, so this output goes to solve__pdtProcess_section.csv
# (read_parameters.py:48 reads here).
if p_model["solveFirst"] == 1 then {
  printf "solve,period,time" > "solve_data/solve__pdtProcess_section.csv";
  for {p in process_minload} {printf ",%s", p >> "solve_data/solve__pdtProcess_section.csv";}
}
for {s in solve_current, (d, t) in dt_realize_dispatch} {
    printf "\n%s,%s,%s", s, d, t >> "solve_data/solve__pdtProcess_section.csv";
    for {p in process_minload} {
        printf ",%.8g", pdtProcess_section[p, d, t] >> "solve_data/solve__pdtProcess_section.csv";
    }
}

# Write pdtProcess (availability)
if p_model["solveFirst"] == 1 then {
  printf "solve,period,time" > "solve_data/pdtProcess_availability.csv";
  for {p in process} {printf ",%s", p >> "solve_data/pdtProcess_availability.csv";}
}
for {s in solve_current, (d, t) in dt_realize_dispatch} {
    printf "\n%s,%s,%s", s, d, t >> "solve_data/pdtProcess_availability.csv";
    for {p in process} {
        printf ",%.8g", pdtProcess[p, 'availability', d, t] >> "solve_data/pdtProcess_availability.csv";
    }
}

# Write pdtProcess_source_sink_varCost
if p_model["solveFirst"] == 1 then {
  printf "solve,period,time" > "solve_data/pdtProcess_source_sink_varCost.csv";
  for {(p, source, sink) in process_source_sink_alwaysProcess} {printf ",%s", p >> "solve_data/pdtProcess_source_sink_varCost.csv";}
  printf "\n,," >> "solve_data/pdtProcess_source_sink_varCost.csv";
  for {(p, source, sink) in process_source_sink_alwaysProcess} {printf ",%s", source >> "solve_data/pdtProcess_source_sink_varCost.csv";}
  printf "\n,," >> "solve_data/pdtProcess_source_sink_varCost.csv";
  for {(p, source, sink) in process_source_sink_alwaysProcess} {printf ",%s", sink >> "solve_data/pdtProcess_source_sink_varCost.csv";}
}
for {s in solve_current, (d, t) in dt_realize_dispatch} {
    printf "\n%s,%s,%s", s, d, t >> "solve_data/pdtProcess_source_sink_varCost.csv";
    for {(p, source, sink) in process_source_sink_alwaysProcess} {
        printf ",%.8g", pdtProcess__source__sink__dt_varCost_alwaysProcess[p, source, sink, d, t] >> "solve_data/pdtProcess_source_sink_varCost.csv";
    }
}

# Write pdtNode (self_discharge_loss, penalty_up, penalty_down)
if p_model["solveFirst"] == 1 then {
  printf "solve,period,time" > "solve_data/pdtNode_self_discharge_loss.csv";
  for {n in nodeSelfDischarge} {printf ",%s", n >> "solve_data/pdtNode_self_discharge_loss.csv";}
}
for {s in solve_current, (d, t) in dt_realize_dispatch} {
    printf "\n%s,%s,%s", s, d, t >> "solve_data/pdtNode_self_discharge_loss.csv";
    for {n in nodeSelfDischarge} {
        printf ",%.8g", pdtNode[n, 'self_discharge_loss', d, t] >> "solve_data/pdtNode_self_discharge_loss.csv";
    }
}

if p_model["solveFirst"] == 1 then {
  printf "solve,period,time" > "solve_data/pdtNode_penalty_up.csv";
  for {n in nodeBalance union nodeBalancePeriod} {printf ",%s", n >> "solve_data/pdtNode_penalty_up.csv";}
}
for {s in solve_current, (d, t) in dt_realize_dispatch} {
    printf "\n%s,%s,%s", s, d, t >> "solve_data/pdtNode_penalty_up.csv";
    for {n in nodeBalance union nodeBalancePeriod} {
        printf ",%.8g", pdtNode[n, 'penalty_up', d, t] >> "solve_data/pdtNode_penalty_up.csv";
    }
}

if p_model["solveFirst"] == 1 then {
  printf "solve,period,time" > "solve_data/pdtNode_penalty_down.csv";
  for {n in nodeBalance union nodeBalancePeriod} {printf ",%s", n >> "solve_data/pdtNode_penalty_down.csv";}
}
for {s in solve_current, (d, t) in dt_realize_dispatch} {
    printf "\n%s,%s,%s", s, d, t >> "solve_data/pdtNode_penalty_down.csv";
    for {n in nodeBalance union nodeBalancePeriod} {
        printf ",%.8g", pdtNode[n, 'penalty_down', d, t] >> "solve_data/pdtNode_penalty_down.csv";
    }
}

# Write pdtNodeInflow (post-solve realized values, wide solve×period×time × node).
# Mod's input now reads solve_data/pdtNodeInflow.csv (long format) written
# by Python preprocessing, so this output goes to solve__pdtNodeInflow.csv
# to avoid the dual-writer path collision (read_parameters.py:54 reads here).
if p_model["solveFirst"] == 1 then {
  printf "solve,period,time" > "solve_data/solve__pdtNodeInflow.csv";
  for {n in node} {printf ",%s", n >> "solve_data/solve__pdtNodeInflow.csv";}
}
for {s in solve_current, (d, t) in dt_realize_dispatch} {
    printf "\n%s,%s,%s", s, d, t >> "solve_data/solve__pdtNodeInflow.csv";
    for {n in node} {
        printf ",%.8g", pdtNodeInflow[n, d, t] >> "solve_data/solve__pdtNodeInflow.csv";
    }
}

# Write pdtCommodity (price)
if p_model["solveFirst"] == 1 then {
  printf "solve,period,time" > "solve_data/pdtCommodity_price.csv";
  for {c in commodity} {printf ",%s", c >> "solve_data/pdtCommodity_price.csv";}
}
for {s in solve_current, (d, t) in dt_realize_dispatch} {
    printf "\n%s,%s,%s", s, d, t >> "solve_data/pdtCommodity_price.csv";
    for {c in commodity} {
        printf ",%.8g", pdtCommodity[c, 'price', d, t] >> "solve_data/pdtCommodity_price.csv";
    }
}

# Write pdtGroup (co2_price)
if p_model["solveFirst"] == 1 then {
  printf "solve,period,time" > "solve_data/pdtGroup_co2_price.csv";
  for {g in group_co2_price} {printf ",%s", g >> "solve_data/pdtGroup_co2_price.csv";}
}
for {s in solve_current, (d, t) in dt_realize_dispatch} {
    printf "\n%s,%s,%s", s, d, t >> "solve_data/pdtGroup_co2_price.csv";
    for {g in group_co2_price} {
        printf ",%.8g", pdtGroup[g, 'co2_price', d, t] >> "solve_data/pdtGroup_co2_price.csv";
    }
}

# Write pdtReserve_upDown_group (reservation)
if p_model["solveFirst"] == 1 then {
  printf "solve,period,time" > "solve_data/pdtReserve_upDown_group_reservation.csv";
  for {(r, ud, ng) in reserve__upDown__group} {printf ",%s", r >> "solve_data/pdtReserve_upDown_group_reservation.csv";}
  printf "\n,," >> "solve_data/pdtReserve_upDown_group_reservation.csv";
  for {(r, ud, ng) in reserve__upDown__group} {printf ",%s", ud >> "solve_data/pdtReserve_upDown_group_reservation.csv";}
  printf "\n,," >> "solve_data/pdtReserve_upDown_group_reservation.csv";
  for {(r, ud, ng) in reserve__upDown__group} {printf ",%s", ng >> "solve_data/pdtReserve_upDown_group_reservation.csv";}
}
for {s in solve_current, (d, t) in dt_realize_dispatch} {
    printf "\n%s,%s,%s", s, d, t >> "solve_data/pdtReserve_upDown_group_reservation.csv";
    for {(r, ud, ng) in reserve__upDown__group} {
        printf ",%.8g", pdtReserve_upDown_group[r, ud, ng, 'reservation', d, t] >> "solve_data/pdtReserve_upDown_group_reservation.csv";
    }
}

# Write pdtProfile (post-solve realized values, wide format).
# Mod's input now reads solve_data/pdtProfile.csv (long) written by Python
# preprocessing, so this output goes to solve__pdtProfile.csv to avoid
# the dual-writer collision (read_parameters.py:58 reads here).
if p_model["solveFirst"] == 1 then {
  printf "solve,period,time" > "solve_data/solve__pdtProfile.csv";
  for {f in profile} {printf ",%s", f >> "solve_data/solve__pdtProfile.csv";}
}
for {s in solve_current, (d, t) in dt_realize_dispatch} {
    printf "\n%s,%s,%s", s, d, t >> "solve_data/solve__pdtProfile.csv";
    for {f in profile} {
        printf ",%.8g", pdtProfile[f, d, t] >> "solve_data/solve__pdtProfile.csv";
    }
}

# Write p_years_from_start_d
if p_model["solveFirst"] == 1 then {
  printf "solve,period,value" > "solve_data/p_years_from_start_d.csv";
}
for {s in solve_current, d in d_realize_dispatch_or_invest} {
    printf "\n%s,%s,%.8g", s, d, p_years_d[d] >> "solve_data/p_years_from_start_d.csv";
}

# Write p_years_represented_d
if p_model["solveFirst"] == 1 then {
  printf "solve,period,value" > "solve_data/p_years_represented_d.csv";
}
for {s in solve_current, d in d_realize_dispatch_or_invest} {
    printf "\n%s,%s,%.8g", s, d, p_years_represented_d[d] >> "solve_data/p_years_represented_d.csv";
}

# Write p_entity_max_units (post-solve realized values, wide format).
# Mod's input now reads solve_data/p_entity_max_units.csv (long) written by
# Python preprocessing, so this output goes to solve__p_entity_max_units.csv
# (read_parameters.py:61 reads here).
if p_model["solveFirst"] == 1 then {
  printf "solve,period" > "solve_data/solve__p_entity_max_units.csv";
  for {e in entity} {printf ",%s", e >> "solve_data/solve__p_entity_max_units.csv";}
}
for {s in solve_current, d in d_realize_dispatch_or_invest} {
    printf "\n%s,%s", s, d >> "solve_data/solve__p_entity_max_units.csv";
    for {e in entity} {
        printf ",%.8g", p_entity_max_units[e, d] >> "solve_data/solve__p_entity_max_units.csv";
    }
}

# Write p_entity_all_existing (post-solve realized values, wide format).
# Mod's input now reads solve_data/p_entity_all_existing.csv (long) written by
# Python preprocessing, so this output goes to solve__p_entity_all_existing.csv
# (handoff_writers._load_p_entity_all_existing + read_parameters.py:62 read here).
if p_model["solveFirst"] == 1 then {
  printf "solve,period" > "solve_data/solve__p_entity_all_existing.csv";
  for {e in entity} {printf ",%s", e >> "solve_data/solve__p_entity_all_existing.csv";}
}
for {s in solve_current, d in d_realize_dispatch_or_invest} {
    printf "\n%s,%s", s, d >> "solve_data/solve__p_entity_all_existing.csv";
    for {e in entity} {
        printf ",%.8g", p_entity_all_existing[e, d] >> "solve_data/solve__p_entity_all_existing.csv";
    }
}

# Write p_entity_pre_existing
if p_model["solveFirst"] == 1 then {
  printf "solve,period" > "solve_data/solve__p_entity_pre_existing.csv";
  for {e in entity} {printf ",%s", e >> "solve_data/solve__p_entity_pre_existing.csv";}
}
for {s in solve_current, d in d_realize_dispatch_or_invest} {
    printf "\n%s,%s", s, d >> "solve_data/solve__p_entity_pre_existing.csv";
    for {e in entity} {
        printf ",%.8g", p_entity_pre_existing[e, d] >> "solve_data/solve__p_entity_pre_existing.csv";
    }
}

# Write pdProcess_startup_cost
if p_model["solveFirst"] == 1 then {
  printf "solve,period" > "solve_data/pdProcess_startup_cost.csv";
  for {p in process_online} {printf ",%s", p >> "solve_data/pdProcess_startup_cost.csv";}
}
for {s in solve_current, d in d_realized_period} {
    printf "\n%s,%s", s, d >> "solve_data/pdProcess_startup_cost.csv";
    for {p in process_online} {
        printf ",%.8g", pdProcess[p, 'startup_cost', d] >> "solve_data/pdProcess_startup_cost.csv";
    }
}

# Write fixed costs
if p_model["solveFirst"] == 1 then {
  printf "solve,period" > "solve_data/solve__ed_fixed_cost.csv";
  for {e in entity} {printf ",%s", e >> "solve_data/solve__ed_fixed_cost.csv";}
}
for {s in solve_current, d in d_realized_period} {
    printf "\n%s,%s", s, d >> "solve_data/solve__ed_fixed_cost.csv";
    for {e in entity} {
        printf ",%.8g", ed_fixed_cost[e, d] >> "solve_data/solve__ed_fixed_cost.csv";
    }
}

if p_model["solveFirst"] == 1 then {
  printf "solve,period" > "solve_data/solve__ed_lifetime_fixed_cost.csv";
  for {e in entity} {printf ",%s", e >> "solve_data/solve__ed_lifetime_fixed_cost.csv";}
}
for {s in solve_current, d in d_realized_period} {
    printf "\n%s,%s", s, d >> "solve_data/solve__ed_lifetime_fixed_cost.csv";
    # Header has all entities (see line above); write 0 for entities not in
    # ed_invest so the column alignment matches the header.
    for {e in entity} {
        printf ",%.8g",
            (if (e, d) in ed_invest then ed_lifetime_fixed_cost[e, d] else 0)
            >> "solve_data/solve__ed_lifetime_fixed_cost.csv";
    }
}

if p_model["solveFirst"] == 1 then {
  printf "solve,period" > "solve_data/solve__ed_lifetime_fixed_cost_divest.csv";
  for {e in entity} {printf ",%s", e >> "solve_data/solve__ed_lifetime_fixed_cost_divest.csv";}
}
for {s in solve_current, d in d_realized_period} {
    printf "\n%s,%s", s, d >> "solve_data/solve__ed_lifetime_fixed_cost_divest.csv";
    # Header has all entities; write 0 for entities not in ed_divest so the
    # column alignment matches the header.
    for {e in entity} {
        printf ",%.8g",
            (if (e, d) in ed_divest then ed_lifetime_fixed_cost_divest[e, d] else 0)
            >> "solve_data/solve__ed_lifetime_fixed_cost_divest.csv";
    }
}

# Write pdNode annual_flow
if p_model["solveFirst"] == 1 then {
  printf "solve,period" > "solve_data/pdNode_annual_flow.csv";
  for {(n, 'annual_flow') in node__PeriodParam_in_use} {printf ",%s", n >> "solve_data/pdNode_annual_flow.csv";}
}
for {s in solve_current, d in d_realized_period} {
    printf "\n%s,%s", s, d >> "solve_data/pdNode_annual_flow.csv";
    for {(n, 'annual_flow') in node__PeriodParam_in_use} {
        printf ",%.8g", pdNode[n, 'annual_flow', d] >> "solve_data/pdNode_annual_flow.csv";
    }
}

# Write pdGroup (penalty_inertia, penalty_non_synchronous, penalty_capacity_margin, inertia_limit, capacity_margin)
if p_model["solveFirst"] == 1 then {
  printf "solve,period" > "solve_data/pdGroup_penalty_inertia.csv";
  for {g in groupInertia} {printf ",%s", g >> "solve_data/pdGroup_penalty_inertia.csv";}
}
for {s in solve_current, d in d_realized_period} {
    printf "\n%s,%s", s, d >> "solve_data/pdGroup_penalty_inertia.csv";
    for {g in groupInertia} {
        printf ",%.8g", pdGroup[g, 'penalty_inertia', d] >> "solve_data/pdGroup_penalty_inertia.csv";
    }
}

if p_model["solveFirst"] == 1 then {
  printf "solve,period" > "solve_data/pdGroup_penalty_non_synchronous.csv";
  for {g in groupNonSync} {printf ",%s", g >> "solve_data/pdGroup_penalty_non_synchronous.csv";}
}
for {s in solve_current, d in d_realized_period} {
    printf "\n%s,%s", s, d >> "solve_data/pdGroup_penalty_non_synchronous.csv";
    for {g in groupNonSync} {
        printf ",%.8g", pdGroup[g, 'penalty_non_synchronous', d] >> "solve_data/pdGroup_penalty_non_synchronous.csv";
    }
}

if p_model["solveFirst"] == 1 then {
  printf "solve,period" > "solve_data/pdGroup_penalty_capacity_margin.csv";
  for {g in groupCapacityMargin} {printf ",%s", g >> "solve_data/pdGroup_penalty_capacity_margin.csv";}
}
for {s in solve_current, d in d_realize_invest} {
    printf "\n%s,%s", s, d >> "solve_data/pdGroup_penalty_capacity_margin.csv";
    for {g in groupCapacityMargin} {
        printf ",%.8g", pdGroup[g, 'penalty_capacity_margin', d] >> "solve_data/pdGroup_penalty_capacity_margin.csv";
    }
}

if p_model["solveFirst"] == 1 then {
  printf "solve,period" > "solve_data/pdGroup_inertia_limit.csv";
  for {g in groupInertia} {printf ",%s", g >> "solve_data/pdGroup_inertia_limit.csv";}
}
for {s in solve_current, d in d_realized_period} {
    printf "\n%s,%s", s, d >> "solve_data/pdGroup_inertia_limit.csv";
    for {g in groupInertia} {
        printf ",%.8g", pdGroup[g, 'inertia_limit', d] >> "solve_data/pdGroup_inertia_limit.csv";
    }
}

if p_model["solveFirst"] == 1 then {
  printf "solve,period" > "solve_data/pdGroup_capacity_margin.csv";
  for {g in groupCapacityMargin} {printf ",%s", g >> "solve_data/pdGroup_capacity_margin.csv";}
}
for {s in solve_current, d in d_realize_invest} {
    printf "\n%s,%s", s, d >> "solve_data/pdGroup_capacity_margin.csv";
    for {g in groupCapacityMargin} {
        printf ",%.8g", pdGroup[g, 'capacity_margin', d] >> "solve_data/pdGroup_capacity_margin.csv";
    }
}

# Write ed_entity_annuity
if p_model["solveFirst"] == 1 then {
  printf "solve,period" > "solve_data/ed_entity_annuity.csv";
  for {e in entityInvest} {printf ",%s", e >> "solve_data/ed_entity_annuity.csv";}
}
for {s in solve_current, d in d_realize_invest} {
    printf "\n%s,%s", s, d >> "solve_data/ed_entity_annuity.csv";
    for {e in entityInvest} {
        printf ",%.8g", (if (e, d) in ed_invest then ed_entity_annual[e, d] else 0) >> "solve_data/ed_entity_annuity.csv";
    }
}

# Write ed_entity_annual_discounted
if p_model["solveFirst"] == 1 then {
  printf "solve,period" > "solve_data/solve__ed_entity_annual_discounted.csv";
  for {e in entityInvest} {printf ",%s", e >> "solve_data/solve__ed_entity_annual_discounted.csv";}
}
for {s in solve_current, d in d_realize_invest} {
    printf "\n%s,%s", s, d >> "solve_data/solve__ed_entity_annual_discounted.csv";
    for {e in entityInvest} {
        printf ",%.8g", (if (e, d) in ed_invest then ed_entity_annual_discounted[e, d] else 0) >> "solve_data/solve__ed_entity_annual_discounted.csv";
    }
}

# Write ed_entity_annual_divest_discounted
if p_model["solveFirst"] == 1 then {
  printf "solve,period" > "solve_data/solve__ed_entity_annual_divest_discounted.csv";
  for {e in entityDivest} {printf ",%s", e >> "solve_data/solve__ed_entity_annual_divest_discounted.csv";}
}
for {s in solve_current, d in d_realize_invest} {
    printf "\n%s,%s", s, d >> "solve_data/solve__ed_entity_annual_divest_discounted.csv";
    for {e in entityDivest} {
        printf ",%.8g", (if (e, d) in ed_divest then ed_entity_annual_divest_discounted[e, d] else 0) >> "solve_data/solve__ed_entity_annual_divest_discounted.csv";
    }
}

# Write p_inflation_factor_operations_yearly
if p_model["solveFirst"] == 1 then {
  printf "solve,period,value" > "solve_data/solve__p_inflation_factor_operations_yearly.csv";
}
for {s in solve_current, d in d_realized_period} {
    printf "\n%s,%s,%.12g", s, d, p_inflation_factor_operations_yearly[d] >> "solve_data/solve__p_inflation_factor_operations_yearly.csv";
}

# Write p_inflation_factor_investment_yearly
if p_model["solveFirst"] == 1 then {
  printf "solve,period,value" > "solve_data/solve__p_inflation_factor_investment_yearly.csv";
}
for {s in solve_current, d in d_realize_invest} {
    printf "\n%s,%s,%.12g", s, d, p_inflation_factor_investment_yearly[d] >> "solve_data/solve__p_inflation_factor_investment_yearly.csv";
}

# Write node_capacity_for_scaling
if p_model["solveFirst"] == 1 then {
  printf "solve,period" > "solve_data/solve__node_capacity_for_scaling.csv";
  for {n in node} {printf ",%s", n >> "solve_data/solve__node_capacity_for_scaling.csv";}
}
for {s in solve_current, d in d_realized_period} {
    printf "\n%s,%s", s, d >> "solve_data/solve__node_capacity_for_scaling.csv";
    for {n in node} {
        printf ",%.12g", node_capacity_for_scaling[n, d] >> "solve_data/solve__node_capacity_for_scaling.csv";
    }
}

# Write group_capacity_for_scaling
if p_model["solveFirst"] == 1 then {
  printf "solve,period" > "solve_data/solve__group_capacity_for_scaling.csv";
  for {g in group} {printf ",%s", g >> "solve_data/solve__group_capacity_for_scaling.csv";}
}
for {s in solve_current, d in d_realized_period} {
    printf "\n%s,%s", s, d >> "solve_data/solve__group_capacity_for_scaling.csv";
    for {g in group} {
        printf ",%.12g", group_capacity_for_scaling[g, d] >> "solve_data/solve__group_capacity_for_scaling.csv";
    }
}

# Write complete_period_share_of_year
if p_model["solveFirst"] == 1 then {
  printf "solve,period,value" > "solve_data/complete_period_share_of_year.csv";
}
for {s in solve_current, d in d_realized_period} {
    printf "\n%s,%s,%.12g", s, d, complete_period_share_of_year[d] >> "solve_data/complete_period_share_of_year.csv";
}

# Parameters and sets without d or (d,t) dimensions written only at first solve

if p_model['solveFirst'] then {
  # Write p_node
  printf "param" > "solve_data/p_node.csv";
  for {n in node} printf ",%s", n >> "solve_data/p_node.csv";
  for {param in nodeParam} {
      printf "\n%s", param >> "solve_data/p_node.csv";
      for {n in node} {
        printf ",%.8g", p_node[n, param] >> "solve_data/p_node.csv";
      }
  }

  # Write p_unit
  printf "param" > "input/p_unit.csv";
  for {p in process_unit} printf ",%s", p >> "input/p_unit.csv";
  for {param in processParam} {
      printf "\n%s", param >> "input/p_unit.csv";
      for {p in process_unit} {
        printf ",%.8g", p_process[p, param] >> "input/p_unit.csv";
      }
  }

  # Write p_connection
  printf "param" > "input/p_connection.csv";
  for {p in process_connection} printf ",%s", p >> "input/p_connection.csv";
  for {param in processParam} {
      printf "\n%s", param >> "input/p_connection.csv";
      for {p in process_connection} {
        printf ",%.8g", p_process[p, param] >> "input/p_connection.csv";
      }
  }

  # Write p_entity_unitsize
  printf "entity" > "input/p_entity_unitsize.csv";
  for {e in entity} printf ",%s", e >> "input/p_entity_unitsize.csv";
  printf "\nvalue" >> "input/p_entity_unitsize.csv";
  for {e in entity} {
    printf ",%.8g", p_entity_unitsize[e] >> "input/p_entity_unitsize.csv";
  }

  # Write p_process_source
  printf "process" > "solve_data/p_process_source.csv";
  for {(p, sr) in process_source} printf ",%s", p >> "solve_data/p_process_source.csv";
  printf "\nsource" >> "solve_data/p_process_source.csv";
  for {(p, sr) in process_source} printf ",%s", sr >> "solve_data/p_process_source.csv";
  for {param in sourceSinkParam} {
      printf "\n%s", param >> "solve_data/p_process_source.csv";
      for {(p, sr) in process_source} {
          printf ",%.8g", p_process_source[p, sr, param] >> "solve_data/p_process_source.csv";
      }
  }

  # Write p_process_sink
  printf "process" > "solve_data/p_process_sink.csv";
  for {(p, sink) in process_sink} printf ",%s", p >> "solve_data/p_process_sink.csv";
  printf "\nsink" >> "solve_data/p_process_sink.csv";
  for {(p, sink) in process_sink} printf ",%s", sink >> "solve_data/p_process_sink.csv";
  for {param in sourceSinkParam} {
      printf "\n%s", param >> "solve_data/p_process_sink.csv";
      for {(p, sink) in process_sink} {
          printf ",%.8g", p_process_sink[p, sink, param] >> "solve_data/p_process_sink.csv";
      }
  }

  # Write p_process_sink_flow_coefficient
  printf "process" > "solve_data/p_process_sink_flow_coefficient.csv";
  for {(p, sink) in process_sink} printf ",%s", p >> "solve_data/p_process_sink_flow_coefficient.csv";
  printf "\nsink" >>  "solve_data/p_process_sink_flow_coefficient.csv";
  for {(p, sink) in process_sink} printf ",%s", sink >> "solve_data/p_process_sink_flow_coefficient.csv";
  printf "\nvalue" >> "solve_data/p_process_sink_flow_coefficient.csv";
  for {(p, sink) in process_sink} {
    printf ",%.8g", p_process_sink_flow_coefficient[p, sink] >> "solve_data/p_process_sink_flow_coefficient.csv";
  }

  # Write p_process_source_flow_coefficient
  printf "process" > "solve_data/p_process_source_flow_coefficient.csv";
  for {(p, sr) in process_source} printf ",%s", p >> "solve_data/p_process_source_flow_coefficient.csv";
  printf "\nsource" >> "solve_data/p_process_source_flow_coefficient.csv";
  for {(p, sr) in process_source} printf ",%s", sr >> "solve_data/p_process_source_flow_coefficient.csv";
  printf "\nvalue" >> "solve_data/p_process_source_flow_coefficient.csv";
  for {(p, sr) in process_source} {
    printf ",%.8g", p_process_source_flow_coefficient[p, sr] >> "solve_data/p_process_source_flow_coefficient.csv";
  }

  # Write p_process_sink_max_capacity_coefficient
  printf "process" > "solve_data/p_process_sink_max_capacity_coefficient.csv";
  for {(p, sink) in process_sink} printf ",%s", p >> "solve_data/p_process_sink_max_capacity_coefficient.csv";
  printf "\nsink" >>  "solve_data/p_process_sink_max_capacity_coefficient.csv";
  for {(p, sink) in process_sink} printf ",%s", sink >> "solve_data/p_process_sink_max_capacity_coefficient.csv";
  printf "\nvalue" >> "solve_data/p_process_sink_max_capacity_coefficient.csv";
  for {(p, sink) in process_sink} {
    printf ",%.8g", p_process_sink_max_capacity_coefficient[p, sink] >> "solve_data/p_process_sink_max_capacity_coefficient.csv";
  }

  # Write p_process_sink_min_capacity_coefficient
  printf "process" > "solve_data/p_process_sink_min_capacity_coefficient.csv";
  for {(p, sink) in process_sink} printf ",%s", p >> "solve_data/p_process_sink_min_capacity_coefficient.csv";
  printf "\nsink" >>  "solve_data/p_process_sink_min_capacity_coefficient.csv";
  for {(p, sink) in process_sink} printf ",%s", sink >> "solve_data/p_process_sink_min_capacity_coefficient.csv";
  printf "\nvalue" >> "solve_data/p_process_sink_min_capacity_coefficient.csv";
  for {(p, sink) in process_sink} {
    printf ",%.8g", p_process_sink_min_capacity_coefficient[p, sink] >> "solve_data/p_process_sink_min_capacity_coefficient.csv";
  }

  # Write p_process_source_max_capacity_coefficient
  printf "process" > "solve_data/p_process_source_max_capacity_coefficient.csv";
  for {(p, sr) in process_source} printf ",%s", p >> "solve_data/p_process_source_max_capacity_coefficient.csv";
  printf "\nsource" >> "solve_data/p_process_source_max_capacity_coefficient.csv";
  for {(p, sr) in process_source} printf ",%s", sr >> "solve_data/p_process_source_max_capacity_coefficient.csv";
  printf "\nvalue" >> "solve_data/p_process_source_max_capacity_coefficient.csv";
  for {(p, sr) in process_source} {
    printf ",%.8g", p_process_source_max_capacity_coefficient[p, sr] >> "solve_data/p_process_source_max_capacity_coefficient.csv";
  }

  # Write p_process_source_min_capacity_coefficient
  printf "process" > "solve_data/p_process_source_min_capacity_coefficient.csv";
  for {(p, sr) in process_source} printf ",%s", p >> "solve_data/p_process_source_min_capacity_coefficient.csv";
  printf "\nsource" >> "solve_data/p_process_source_min_capacity_coefficient.csv";
  for {(p, sr) in process_source} printf ",%s", sr >> "solve_data/p_process_source_min_capacity_coefficient.csv";
  printf "\nvalue" >> "solve_data/p_process_source_min_capacity_coefficient.csv";
  for {(p, sr) in process_source} {
    printf ",%.8g", p_process_source_min_capacity_coefficient[p, sr] >> "solve_data/p_process_source_min_capacity_coefficient.csv";
  }

  # Write p_commodity_co2_content
  printf "commodity" > "input/p_commodity_co2_content.csv";
  for {c in commodity} printf ",%s", c >> "input/p_commodity_co2_content.csv";
  printf "\nvalue" >> "input/p_commodity_co2_content.csv";
  for {c in commodity} {
    printf ",%.8g", p_commodity[c, 'co2_content'] >> "input/p_commodity_co2_content.csv";
  }
  printf "\n" >> "input/p_commodity_co2_content.csv";

  # Write p_reserve_upDown_group_penalty
  printf "reserve" > "input/p_reserve_upDown_group_penalty.csv";
  for {(r, ud, ng) in reserve__upDown__group} printf ",%s", r >> "input/p_reserve_upDown_group_penalty.csv";
  printf "\nupDown" >> "input/p_reserve_upDown_group_penalty.csv";
  for {(r, ud, ng) in reserve__upDown__group} printf ",%s", ud >> "input/p_reserve_upDown_group_penalty.csv";
  printf "\ngroup" >> "input/p_reserve_upDown_group_penalty.csv";
  for {(r, ud, ng) in reserve__upDown__group} printf ",%s", ng >> "input/p_reserve_upDown_group_penalty.csv";
  printf "\nvalue" >> "input/p_reserve_upDown_group_penalty.csv";
  for {(r, ud, ng) in reserve__upDown__group} {
    printf ",%.8g", p_reserve_upDown_group[r, ud, ng, 'penalty_reserve'] >> "input/p_reserve_upDown_group_penalty.csv";
  }


  # Sets needed for entity_all_capacity
  # entity - all entities (ordered)

  # period - all periods (ordered)
  if p_model["solveFirst"] == 1 then printf "solve,period\n" > "solve_data/period.csv";
  for {s in solve_current, d in period} {
      printf "%s,%s\n", s, d >> "solve_data/period.csv";
  }

  # entityInvest - entities that can invest
  printf "entity\n" > "solve_data/entityInvest.csv";
  for {e in entityInvest} {
      printf "%s\n", e >> "solve_data/entityInvest.csv";
  }

  # entityDivest - entities that can divest
  printf "entity\n" > "solve_data/entityDivest.csv";
  for {e in entityDivest} {
      printf "%s\n", e >> "solve_data/entityDivest.csv";
  }


  # Sets for online methods
  # process_online - processes with online variables
  printf "process\n" > "solve_data/process_online.csv";
  for {p in process_online} {
      printf "%s\n", p >> "solve_data/process_online.csv";
  }

  # process_online_linear - processes with linear online variables
  printf "process\n" > "solve_data/process_online_linear.csv";
  for {p in process_online_linear} {
      printf "%s\n", p >> "solve_data/process_online_linear.csv";
  }

  # process_online_integer - processes with integer online variables
  printf "process\n" > "solve_data/process_online_integer.csv";
  for {p in process_online_integer} {
      printf "%s\n", p >> "solve_data/process_online_integer.csv";
  }

  # Sets needed for flow calculations
  # Process topology sets
  printf "process,source,sink\n" > "solve_data/process_source_sink.csv";
  for {(p, source, sink) in process_source_sink} {
      printf "%s,%s,%s\n", p, source, sink >> "solve_data/process_source_sink.csv";
  }

  # process_source_sink_noEff / _eff splits — used by the Python CO2
  # rolling-accumulator writer to classify flows into "no-eff" and
  # "simple-eff" branches (matches the mod's co2_max_total LHS split).
  printf "process,source,sink\n" > "solve_data/process_source_sink_noEff.csv";
  for {(p, source, sink) in process_source_sink_noEff} {
      printf "%s,%s,%s\n", p, source, sink >> "solve_data/process_source_sink_noEff.csv";
  }
  printf "process,source,sink\n" > "solve_data/process_source_sink_eff.csv";
  for {(p, source, sink) in process_source_sink_eff} {
      printf "%s,%s,%s\n", p, source, sink >> "solve_data/process_source_sink_eff.csv";
  }

  printf "process,method,orig_source,orig_sink,always_source,always_sink\n" > "solve_data/process_method_sources_sinks.csv";
  for {(p, m, orig_source, orig_sink, always_source, always_sink) in process_method_sources_sinks} {
      printf "%s,%s,%s,%s,%s,%s\n", p, m, orig_source, orig_sink, always_source, always_sink >> "solve_data/process_method_sources_sinks.csv";
  }

  # Process profile set
  printf "process\n" > "solve_data/process_profile.csv";
  for {p in process_profile} {
      printf "%s\n", p >> "solve_data/process_profile.csv";
  }

  # Process process__node__profile__profile_method set
  # Process method sets
  printf "process,method\n" > "solve_data/process__ct_method.csv";
  for {(p, m) in process__ct_method} {
      printf "%s,%s\n", p, m >> "solve_data/process__ct_method.csv";
  }

  # Process unit and connection sets
  printf "process\n" > "solve_data/process_unit.csv";
  for {p in process_unit} {
      printf "%s\n", p >> "solve_data/process_unit.csv";
  }
  printf "process\n" > "solve_data/process_connection.csv";
  for {p in process_connection} {
      printf "%s\n", p >> "solve_data/process_connection.csv";
  }

  # Method type sets
  printf "method\n" > "solve_data/method_1var_per_way.csv";
  for {m in method_1var_per_way} {
      printf "%s\n", m >> "solve_data/method_1var_per_way.csv";
  }

  printf "method\n" > "solve_data/method_nvar.csv";
  for {m in method_nvar} {
      printf "%s\n", m >> "solve_data/method_nvar.csv";
  }

  # Node-related sets — the three membership files
  # (set_nodeState, set_nodeBalance, set_nodeBalancePeriod) are no longer
  # written from the mod.  read_sets.py derives all three from
  # input/p_node_type.csv directly.

  printf "node\n" > "solve_data/nodeSelfDischarge.csv";
  for {n in nodeSelfDischarge} {
      printf "%s\n", n >> "solve_data/nodeSelfDischarge.csv";
  }

  printf "node,method\n" > "solve_data/node__storage_binding_method.csv";
  for {(n, m) in node__storage_binding_method} {
      printf "%s,%s\n", n, m >> "solve_data/node__storage_binding_method.csv";
  }

  printf "node,method\n" > "solve_data/node__storage_start_end_method.csv";
  for {(n, m) in node__storage_start_end_method} {
      printf "%s,%s\n", n, m >> "solve_data/node__storage_start_end_method.csv";
  }

  printf "node,method\n" > "solve_data/node__inflow_method.csv";
  for {(n, m) in node__inflow_method} {
      printf "%s,%s\n", n, m >> "solve_data/node__inflow_method.csv";
  }

  printf "node,method\n" > "solve_data/node__storage_nested_fix_method.csv";
  for {(n, m) in node__storage_nested_fix_method} {
      printf "%s,%s\n", n, m >> "solve_data/node__storage_nested_fix_method.csv";
  }

  # Process-related sets
  printf "process\n" > "solve_data/process_connection.csv";
  for {c in process_connection} {
      printf "%s\n", c >> "solve_data/process_connection.csv";
  }

  printf "process,source\n" > "solve_data/process_source.csv";
  for {(p, source) in process_source} {
      printf "%s,%s\n", p, source >> "solve_data/process_source.csv";
  }

  printf "process,sink\n" > "solve_data/process_sink.csv";
  for {(p, sink) in process_sink} {
      printf "%s,%s\n", p, sink >> "solve_data/process_sink.csv";
  }

  printf "process,node\n" > "solve_data/process_VRE.csv";
  for {(p, n) in process_sink : p in process_VRE} {
      printf "%s,%s\n", p, n >> "solve_data/process_VRE.csv";
  }

  printf "process,source,sink,profile,method\n" > "solve_data/process__source__sink__profile__profile_method.csv";
  for {(p, source, sink, f, m) in process__source__sink__profile__profile_method} {
      printf "%s,%s,%s,%s,%s\n", p, source, sink, f, m >> "solve_data/process__source__sink__profile__profile_method.csv";
  }

  # Commodity-related sets
  printf "commodity,node\n" > "solve_data/commodity_node.csv";
  for {(c, n) in commodity_node} {
      printf "%s,%s\n", c, n >> "solve_data/commodity_node.csv";
  }

  printf "commodity,node\n" > "solve_data/commodity_node_co2.csv";
  for {(c, n) in commodity_node_co2} {
      printf "%s,%s\n", c, n >> "solve_data/commodity_node_co2.csv";
  }

  printf "process,commodity,node\n" > "solve_data/process__commodity__node.csv";
  for {(p, c, n) in process__commodity__node} {
      printf "%s,%s,%s\n", p, c, n >> "solve_data/process__commodity__node.csv";
  }

  printf "process,commodity,node\n" > "solve_data/process__commodity__node_co2.csv";
  for {(p, c, n) in process__commodity__node_co2} {
      printf "%s,%s,%s\n", p, c, n >> "solve_data/process__commodity__node_co2.csv";
  }

  # Group-related sets
  printf "group\n" > "solve_data/group_co2_price.csv";
  for {g in group_co2_price} {
      printf "%s\n", g >> "solve_data/group_co2_price.csv";
  }

  printf "group\n" > "solve_data/group_co2_limit.csv";
  for {g in group_co2_max_period union group_co2_max_total} {
      printf "%s\n", g >> "solve_data/group_co2_limit.csv";
  }

  # The 12 nodeGroupDispatch__* derived-set printfs that used to live
  # here (mod batch 62) and the nodeGroupDispatch__process_fully_inside
  # printf (batch 33) were redundant — Python preprocessing
  # (preprocessing/process_arc_unions.py) writes the same files via
  # write_node_group_dispatch_sets/_fully_inside before mod runs.
  # Removed in batch 68.

  printf "group,process\n" > "solve_data/group_process.csv";
  for {(g, p) in group_process} {
      printf "%s,%s\n", g, p >> "solve_data/group_process.csv";
  }

  printf "group,node\n" > "solve_data/group_node.csv";
  for {(g, n) in group_node} {
      printf "%s,%s\n", g, n >> "solve_data/group_node.csv";
  }

  printf "group,process,node\n" > "solve_data/group_process_node.csv";
  for {(g, p, n) in group_process_node} {
      printf "%s,%s,%s\n", g, p, n >> "solve_data/group_process_node.csv";
  }

  # upDown set
  printf "updown\n" > "solve_data/upDown.csv";
  for {ud in upDown} {
      printf "%s\n", ud >> "solve_data/upDown.csv";
  }

  # Optional output flag
  printf "flag\n" > "solve_data/enable_optional_outputs.csv";
  for {flag in enable_optional_outputs} {
      printf "%s\n", flag >> "solve_data/enable_optional_outputs.csv";
  }
}

# Sets with period and/or time dimensions
# period_invest - periods where investment can occur
if p_model["solveFirst"] == 1 then printf "solve,period\n" > "solve_data/d_realize_dispatch_or_invest.csv";
for {s in solve_current, d in d_realize_dispatch_or_invest} {
    printf "%s,%s\n", s, d >> "solve_data/d_realize_dispatch_or_invest.csv";
}

if p_model["solveFirst"] == 1 then printf "solve,period\n" > "solve_data/d_realize_invest.csv";
for {s in solve_current, d in d_realize_invest} {
    printf "%s,%s\n", s, d >> "solve_data/d_realize_invest.csv";
}

# Timeline breaks: timesteps where the timeline has a discontinuity.
# Each row marks a timestep where dt_jump != 1 (excluding the very first
# timestep of each period, which always has a non-1 jump from wrap-around).
# The plotting code inserts a NaN gap BEFORE these timesteps.
if p_model["solveFirst"] == 1 then printf "period,time\n" > "solve_data/timeline_breaks.csv";
for {s in solve_current, (d, t) in dt_realize_dispatch: dt_jump[d, t] != 1 && (d, t) not in period__time_first} {
    printf "%s,%s\n", d, t >> "solve_data/timeline_breaks.csv";
}

# ed_invest - (entity, period) pairs where investment occurs
if p_model["solveFirst"] == 1 then printf "solve,entity,period\n" > "solve_data/solve__ed_invest.csv";
for {s in solve_current, (e, d) in ed_invest : d in d_realize_invest} {
    printf "%s,%s,%s\n", s, e, d >> "solve_data/solve__ed_invest.csv";
}

# ed_divest - (entity, period) pairs where divestment occurs
if p_model["solveFirst"] == 1 then printf "solve,entity,period\n" > "solve_data/solve__ed_divest.csv";
for {s in solve_current, (e, d) in ed_divest} {
    printf "%s,%s,%s\n", s, e, d >> "solve_data/solve__ed_divest.csv";
}

# edd_invest - (entity, d_invest, d) triplets showing which investments apply to which periods
if p_model["solveFirst"] == 1 then printf "solve,entity,d_invest,d\n" > "solve_data/solve__edd_invest.csv";
for {s in solve_current, (e, d_invest, d) in edd_invest} {
    printf "%s,%s,%s,%s\n", s, e, d_invest, d >> "solve_data/solve__edd_invest.csv";
}

# Write p_nested_model (solveFirst)
if p_model["solveFirst"] == 1 then printf "solve,param,value" > "solve_data/p_nested_model_dump.csv";
for {s in solve_current, param_name in {"solveFirst"}} {
    printf "\n%s,%s,%.8g", s, param_name, p_nested_model[param_name] >> "solve_data/p_nested_model_dump.csv";
}

# d_realized_period - subset of periods
if p_model["solveFirst"] == 1 then printf "solve,period\n" > "solve_data/d_realized_period.csv";
for {s in solve_current, d in d_realized_period} {
    printf "%s,%s\n", s, d >> "solve_data/d_realized_period.csv";
}

# dt_realize_dispatch - (period, time) pairs
if p_model["solveFirst"] == 1 then printf "solve,period,time\n" > "solve_data/dt_realize_dispatch.csv";
for {s in solve_current, (d, t) in dt_realize_dispatch} {
    printf "%s,%s,%s\n", s, d, t >> "solve_data/dt_realize_dispatch.csv";
}

if p_model["solveFirst"] == 1 then printf "solve,period,time\n" > "solve_data/dt.csv";
for {s in solve_current, (d, t) in dt_realize_dispatch} {
    printf "%s,%s,%s\n", s, d, t >> "solve_data/dt.csv";
}

if p_model["solveFirst"] == 1 then printf "solve,period,time,t_previous\n" > "solve_data/dtt.csv";
for {s in solve_current, (d, t, t_previous) in dtt : (d, t) in dt_realize_dispatch} {
    printf "%s,%s,%s,%s\n", s, d, t, t_previous >> "solve_data/dtt.csv";
}

if p_model["solveFirst"] == 1 then printf "solve,period,time,t_previous,t_previous_within_timeset,d_previous,t_previous_within_solve\n" > "solve_data/dtttdt.csv";
for {s in solve_current, (d, t, t_previous, t_previous_within_timeset, d_previous, t_previous_within_solve) in dtttdt : (d, t) in dt_realize_dispatch} {
    printf "%s,%s,%s,%s,%s,%s,%s\n", s, d, t, t_previous, t_previous_within_timeset, d_previous, t_previous_within_solve >> "solve_data/dtttdt.csv";
}

if p_model["solveFirst"] == 1 then printf "solve,period,time\n" > "solve_data/period__time_first.csv";
for {s in solve_current, (d, t) in period__time_first} {
    printf "%s,%s,%s\n", s, d, t >> "solve_data/period__time_first.csv";
}

if p_model["solveFirst"] == 1 then printf "solve,period\n" > "solve_data/solve__period_first.csv";
for {s in solve_current, d in period_first_of_solve} {
    printf "%s,%s\n", s, d >> "solve_data/solve__period_first.csv";
}

if p_model["solveFirst"] == 1 then printf "solve,period\n" > "solve_data/period_in_use.csv";
for {s in solve_current, d in period_in_use} {
    printf "%s,%s\n", s, d >> "solve_data/period_in_use.csv";
}

if p_model["solveFirst"] == 1 then printf "solve,period,time\n" > "solve_data/dt_fix_storage_timesteps.csv";
for {s in solve_current, (d, t) in dt_fix_storage_timesteps} {
    printf "%s,%s,%s\n", s, d, t >> "solve_data/dt_fix_storage_timesteps.csv";
}

# Group-entity mapping for investment groups (needed by Python post-processing)
if p_model["solveFirst"] == 1 then {
  printf "group,entity\n" > "input/group_entity_invest.csv";
  for {(g, e) in group_entity : g in group_invest}
    { printf "%s,%s\n", g, e >> "input/group_entity_invest.csv"; }
}

} # end: if 'read' in phase

solve;

param r_solution := gmtime();
printf "Timer - Read solution: %ss\n", r_solution - rest;
printf '%s,%s\n', 'r_solution', r_solution - rest >> mod_phases_file;

printf("\n\nOutputs:");

# Write objective value
if p_model["solveFirst"] == 1 then {
  printf "solve,objective" > "output_raw/v_obj.csv";
}
for {s in solve_current} {
    printf "\n%s,%.10g", s, total_cost.val / scale_the_objective >> "output_raw/v_obj.csv";
}


# Agent 1.8: block-aware expansion.  Every fine (d, t) row resolves the
# variable at the coarse step ``tc`` that covers ``t`` on the variable's
# own resolution block (``process__block[p]`` for process-indexed
# variables, ``node__block[n]`` for node-indexed variables).  The
# sum-over-overlap pattern picks the unique ``(b_c, tc)`` pair whose
# default-block fine step equals ``t`` — in the degenerate case every
# node/process maps to 'default' and overlap carries identity rows
# ``(d, 'default', t, 'default', t, 1.0)``, so each sum collapses to
# ``v_foo[..., d, t].val`` and output CSVs are bit-identical to the
# pre-Agent-1.8 state.  For non-degenerate (LH2-style) scenarios each
# coarse value is broadcast across all fine timesteps it covers, so the
# CSV stays rectangular at the finest resolution (design rule: "print
# all at finest resolution and drop the block dimension").

# Write v_flow
if p_model["solveFirst"] == 1 then {
  printf "solve,period,time" > "output_raw/v_flow.csv";
  for {(p, source, sink) in process_source_sink} {printf ",%s", p >> "output_raw/v_flow.csv";}
  printf "\n,," >> "output_raw/v_flow.csv";
  for {(p, source, sink) in process_source_sink} {printf ",%s", source >> "output_raw/v_flow.csv";}
  printf "\n,," >> "output_raw/v_flow.csv";
  for {(p, source, sink) in process_source_sink} {printf ",%s", sink >> "output_raw/v_flow.csv";}
}
for {s in solve_current, (d, t) in dt_realize_dispatch} {
    printf "\n%s,%s,%s", s, d, t >> "output_raw/v_flow.csv";
    for {(p, source, sink) in process_source_sink} {
        printf ",%.6g",
            sum {(p, b_p) in process__block,
                 (d, b_c, tc, b_f, tf) in overlap
                 : b_c = b_p and b_f = 'default' and tf = t}
                v_flow[p, source, sink, d, tc].val
            >> "output_raw/v_flow.csv";
    }
}

# Write v_ramp
if p_model["solveFirst"] == 1 then {
  printf "solve,period,time" > "output_raw/v_ramp.csv";
  for {(p, source, sink) in process_source_sink_ramp} {printf ",%s", p >> "output_raw/v_ramp.csv";}
  printf "\n,," >> "output_raw/v_ramp.csv";
  for {(p, source, sink) in process_source_sink_ramp} {printf ",%s", source >> "output_raw/v_ramp.csv";}
  printf "\n,," >> "output_raw/v_ramp.csv";
  for {(p, source, sink) in process_source_sink_ramp} {printf ",%s", sink >> "output_raw/v_ramp.csv";}
}
for {s in solve_current, (d, t) in dt_realize_dispatch} {
    printf "\n%s,%s,%s", s, d, t >> "output_raw/v_ramp.csv";
    for {(p, source, sink) in process_source_sink_ramp} {
        printf ",%.6g",
            sum {(p, b_p) in process__block,
                 (d, b_c, tc, b_f, tf) in overlap
                 : b_c = b_p and b_f = 'default' and tf = t}
                v_ramp[p, source, sink, d, tc].val
            >> "output_raw/v_ramp.csv";
    }
}

# Write v_reserve.  Agent 1.7 V1 constraint pins every reserve-
# participating process/node to the default block, so expansion here is
# a no-op (the sum reduces to v_reserve[..., d, t].val directly).  The
# overlap-based lookup is retained for consistency with the rest of the
# variables and to remain correct when that V1 restriction is lifted.
if p_model["solveFirst"] == 1 then {
  printf "solve,period,time" > "output_raw/v_reserve.csv";
  for {(p, r, ud, n) in process_reserve_upDown_node_active} {printf ",%s", p >> "output_raw/v_reserve.csv";}
  printf "\n,," >> "output_raw/v_reserve.csv";
  for {(p, r, ud, n) in process_reserve_upDown_node_active} {printf ",%s", r >> "output_raw/v_reserve.csv";}
  printf "\n,," >> "output_raw/v_reserve.csv";
  for {(p, r, ud, n) in process_reserve_upDown_node_active} {printf ",%s", ud >> "output_raw/v_reserve.csv";}
  printf "\n,," >> "output_raw/v_reserve.csv";
  for {(p, r, ud, n) in process_reserve_upDown_node_active} {printf ",%s", n >> "output_raw/v_reserve.csv";}
}
for {s in solve_current, (d, t) in dt_realize_dispatch} {
    printf "\n%s,%s,%s", s, d, t >> "output_raw/v_reserve.csv";
    for {(p, r, ud, n) in process_reserve_upDown_node_active} {
        printf ",%.6g",
            sum {(p, b_p) in process__block,
                 (d, b_c, tc, b_f, tf) in overlap
                 : b_c = b_p and b_f = 'default' and tf = t
                   and (p, r, ud, n, d, tc) in prundt}
                v_reserve[p, r, ud, n, d, tc].val
            >> "output_raw/v_reserve.csv";
    }
}

# Write v_state
if p_model["solveFirst"] == 1 then {
  printf "solve,period,time" > "output_raw/v_state.csv";
  for {n in nodeState} {printf ",%s", n >> "output_raw/v_state.csv";}
}
for {s in solve_current, (d, t) in dt_realize_dispatch} {
    printf "\n%s,%s,%s", s, d, t >> "output_raw/v_state.csv";
    for {n in nodeState} {
        printf ",%.6g",
            sum {(n, b_n) in node__block,
                 (d, b_c, tc, b_f, tf) in overlap
                 : b_c = b_n and b_f = 'default' and tf = t}
                v_state[n, d, tc].val
            >> "output_raw/v_state.csv";
    }
}

# Write v_online_linear.  Agent 1.6 redeclared v_online_* at
# p_online_dt (= process__block timesteps), so the overlap-based
# lookup is required whenever the process block is not 'default'.
if p_model["solveFirst"] == 1 then {
  printf "solve,period,time" > "output_raw/v_online_linear.csv";
  for {p in process_online_linear} {printf ",%s", p >> "output_raw/v_online_linear.csv";}
}
for {s in solve_current, (d, t) in dt_realize_dispatch} {
    printf "\n%s,%s,%s", s, d, t >> "output_raw/v_online_linear.csv";
    for {p in process_online_linear} {
        printf ",%.6g",
            sum {(p, b_p) in process__block,
                 (d, b_c, tc, b_f, tf) in overlap
                 : b_c = b_p and b_f = 'default' and tf = t
                   and (p, d, tc) in p_online_dt}
                v_online_linear[p, d, tc].val
            >> "output_raw/v_online_linear.csv";
    }
}

# Write v_startup_linear
if p_model["solveFirst"] == 1 then {
  printf "solve,period,time" > "output_raw/v_startup_linear.csv";
  for {p in process_online_linear} {printf ",%s", p >> "output_raw/v_startup_linear.csv";}
}
for {s in solve_current, (d, t) in dt_realize_dispatch} {
    printf "\n%s,%s,%s", s, d, t >> "output_raw/v_startup_linear.csv";
    for {p in process_online_linear} {
        printf ",%.6g",
            sum {(p, b_p) in process__block,
                 (d, b_c, tc, b_f, tf) in overlap
                 : b_c = b_p and b_f = 'default' and tf = t
                   and (p, d, tc) in p_online_dt}
                v_startup_linear[p, d, tc].val
            >> "output_raw/v_startup_linear.csv";
    }
}

# Write v_shutdown_linear
if p_model["solveFirst"] == 1 then {
  printf "solve,period,time" > "output_raw/v_shutdown_linear.csv";
  for {p in process_online_linear} {printf ",%s", p >> "output_raw/v_shutdown_linear.csv";}
}
for {s in solve_current, (d, t) in dt_realize_dispatch} {
    printf "\n%s,%s,%s", s, d, t >> "output_raw/v_shutdown_linear.csv";
    for {p in process_online_linear} {
        printf ",%.6g",
            sum {(p, b_p) in process__block,
                 (d, b_c, tc, b_f, tf) in overlap
                 : b_c = b_p and b_f = 'default' and tf = t
                   and (p, d, tc) in p_online_dt}
                v_shutdown_linear[p, d, tc].val
            >> "output_raw/v_shutdown_linear.csv";
    }
}

# Write v_online_integer
if p_model["solveFirst"] == 1 then {
  printf "solve,period,time" > "output_raw/v_online_integer.csv";
  for {p in process_online_integer} {printf ",%s", p >> "output_raw/v_online_integer.csv";}
}
for {s in solve_current, (d, t) in dt_realize_dispatch} {
    printf "\n%s,%s,%s", s, d, t >> "output_raw/v_online_integer.csv";
    for {p in process_online_integer} {
        printf ",%.6g",
            sum {(p, b_p) in process__block,
                 (d, b_c, tc, b_f, tf) in overlap
                 : b_c = b_p and b_f = 'default' and tf = t
                   and (p, d, tc) in p_online_dt}
                v_online_integer[p, d, tc].val
            >> "output_raw/v_online_integer.csv";
    }
}

# Write v_startup_integer
if p_model["solveFirst"] == 1 then {
  printf "solve,period,time" > "output_raw/v_startup_integer.csv";
  for {p in process_online_integer} {printf ",%s", p >> "output_raw/v_startup_integer.csv";}
}
for {s in solve_current, (d, t) in dt_realize_dispatch} {
    printf "\n%s,%s,%s", s, d, t >> "output_raw/v_startup_integer.csv";
    for {p in process_online_integer} {
        printf ",%.6g",
            sum {(p, b_p) in process__block,
                 (d, b_c, tc, b_f, tf) in overlap
                 : b_c = b_p and b_f = 'default' and tf = t
                   and (p, d, tc) in p_online_dt}
                v_startup_integer[p, d, tc].val
            >> "output_raw/v_startup_integer.csv";
    }
}

# Write v_shutdown_integer
if p_model["solveFirst"] == 1 then {
  printf "solve,period,time" > "output_raw/v_shutdown_integer.csv";
  for {p in process_online_integer} {printf ",%s", p >> "output_raw/v_shutdown_integer.csv";}
}
for {s in solve_current, (d, t) in dt_realize_dispatch} {
    printf "\n%s,%s,%s", s, d, t >> "output_raw/v_shutdown_integer.csv";
    for {p in process_online_integer} {
        printf ",%.6g",
            sum {(p, b_p) in process__block,
                 (d, b_c, tc, b_f, tf) in overlap
                 : b_c = b_p and b_f = 'default' and tf = t
                   and (p, d, tc) in p_online_dt}
                v_shutdown_integer[p, d, tc].val
            >> "output_raw/v_shutdown_integer.csv";
    }
}

# Write v_angle (DC power flow voltage angles).  V1 DC power flow
# scenarios pin node_dc_power_flow nodes to the default block so the
# overlap lookup collapses to identity; retained for symmetry with the
# other writers.
if p_model["solveFirst"] == 1 then {
  printf "solve,period,time" > "output_raw/v_angle.csv";
  for {n in node_dc_power_flow} {printf ",%s", n >> "output_raw/v_angle.csv";}
}
for {s in solve_current, (d, t) in dt_realize_dispatch} {
    printf "\n%s,%s,%s", s, d, t >> "output_raw/v_angle.csv";
    for {n in node_dc_power_flow} {
        printf ",%.8g",
            sum {(n, b_n) in node__block,
                 (d, b_c, tc, b_f, tf) in overlap
                 : b_c = b_n and b_f = 'default' and tf = t}
                v_angle[n, d, tc].val
            >> "output_raw/v_angle.csv";
    }
}

# Write v_invest (only period dimension)
if p_model["solveFirst"] == 1 then {
  printf "solve,period" > "output_raw/v_invest.csv";
  for {e in entityInvest} {printf ",%s", e >> "output_raw/v_invest.csv";}
}
for {s in solve_current, d in d_realize_invest} {
    printf "\n%s,%s", s, d >> "output_raw/v_invest.csv";
    for {e in entityInvest} {
        printf ",%.6g", (if (e, d) in ed_invest then v_invest[e, d].val else 0) >> "output_raw/v_invest.csv";
    }
}

# Write v_divest (only period dimension)
if p_model["solveFirst"] == 1 then {
  printf "solve,period" > "output_raw/v_divest.csv";
  for {e in entityDivest} {printf ",%s", e >> "output_raw/v_divest.csv";}
}
for {s in solve_current, d in d_realize_invest} {
    printf "\n%s,%s", s, d >> "output_raw/v_divest.csv";
    for {e in entityDivest} {
        printf ",%.6g", (if (e, d) in ed_divest then v_divest[e, d].val else 0) >> "output_raw/v_divest.csv";
    }
}

# Write vq_state_up.  Agent 9b: multiply by node_capacity_for_scaling[n, d]
# to un-scale the row-scaling division (Agent 5c).  In Mode A
# node_cap = 1 so this is a no-op; in Mode B the variable lives in
# "fraction of node_cap" units and this factor recovers the absolute
# CSV magnitude.  Agent 1.8: slack lives in the balance equation which
# is emitted at the node's block; expand via the overlap set so fine-t
# rows carry the coarse slack value.  Degenerate (block='default'):
# identity sum, bit-identical.
if p_model["solveFirst"] == 1 then {
  printf "solve,period,time" > "output_raw/vq_state_up.csv";
  for {n in (nodeBalance union nodeBalancePeriod)} {printf ",%s", n >> "output_raw/vq_state_up.csv";}
}
for {s in solve_current, (d, t) in dt_realize_dispatch} {
    printf "\n%s,%s,%s", s, d, t >> "output_raw/vq_state_up.csv";
    for {n in (nodeBalance union nodeBalancePeriod)} {
        printf ",%.6g",
            sum {(n, b_n) in node__block,
                 (d, b_c, tc, b_f, tf) in overlap
                 : b_c = b_n and b_f = 'default' and tf = t}
                vq_state_up[n, d, tc].val * node_capacity_for_scaling[n, d]
            >> "output_raw/vq_state_up.csv";
    }
}

# Write vq_state_down.  Agent 9b: * node_capacity_for_scaling — see
# vq_state_up note above.  Agent 1.8: block expansion via node__block.
if p_model["solveFirst"] == 1 then {
  printf "solve,period,time" > "output_raw/vq_state_down.csv";
  for {n in (nodeBalance union nodeBalancePeriod)} {printf ",%s", n >> "output_raw/vq_state_down.csv";}
}
for {s in solve_current, (d, t) in dt_realize_dispatch} {
    printf "\n%s,%s,%s", s, d, t >> "output_raw/vq_state_down.csv";
    for {n in (nodeBalance union nodeBalancePeriod)} {
        printf ",%.6g",
            sum {(n, b_n) in node__block,
                 (d, b_c, tc, b_f, tf) in overlap
                 : b_c = b_n and b_f = 'default' and tf = t}
                vq_state_down[n, d, tc].val * node_capacity_for_scaling[n, d]
            >> "output_raw/vq_state_down.csv";
    }
}

# Write vq_reserve.  Agent 1.7 V1: reserve participants pinned to the
# default block, so no expansion is effectively needed — the overlap
# lookup collapses to identity.  Kept at fine (d, t) for output.
if p_model["solveFirst"] == 1 then {
  printf "solve,period,time" > "output_raw/vq_reserve.csv";
  for {(r, ud, ng) in reserve__upDown__group} {printf ",%s", r >> "output_raw/vq_reserve.csv";}
  printf "\n,," >> "output_raw/vq_reserve.csv";
  for {(r, ud, ng) in reserve__upDown__group} {printf ",%s", ud >> "output_raw/vq_reserve.csv";}
  printf "\n,," >> "output_raw/vq_reserve.csv";
  for {(r, ud, ng) in reserve__upDown__group} {printf ",%s", ng >> "output_raw/vq_reserve.csv";}
}
for {s in solve_current, (d, t) in dt_realize_dispatch} {
    printf "\n%s,%s,%s", s, d, t >> "output_raw/vq_reserve.csv";
    for {(r, ud, ng) in reserve__upDown__group} {
        printf ",%.6g", vq_reserve[r, ud, ng, d, t].val >> "output_raw/vq_reserve.csv";
    }
}

# Write vq_inertia.  Agent 1.7 V1: groupInertia nodes pinned to the
# default block, so no expansion needed (identity overlap).
if p_model["solveFirst"] == 1 then {
  printf "solve,period,time" > "output_raw/vq_inertia.csv";
  for {g in groupInertia} {printf ",%s", g >> "output_raw/vq_inertia.csv";}
}
for {s in solve_current, (d, t) in dt_realize_dispatch} {
    printf "\n%s,%s,%s", s, d, t >> "output_raw/vq_inertia.csv";
    for {g in groupInertia} {
        printf ",%.6g", vq_inertia[g, d, t].val >> "output_raw/vq_inertia.csv";
    }
}

# Write vq_non_synchronous.  Agent 9b: * group_capacity_for_scaling[g, d]
# to un-scale the row-scaling division applied in non_sync_constraint
# (Agent 5c).  Mode A: group_cap = 1, no effect; Mode B: recovers the
# absolute CSV magnitude.  Agent 1.7 V1: groupNonSync members pinned to
# the default block, so no block expansion needed.
if p_model["solveFirst"] == 1 then {
  printf "solve,period,time" > "output_raw/vq_non_synchronous.csv";
  for {g in groupNonSync} {printf ",%s", g >> "output_raw/vq_non_synchronous.csv";}
}
for {s in solve_current, (d, t) in dt_realize_dispatch} {
    printf "\n%s,%s,%s", s, d, t >> "output_raw/vq_non_synchronous.csv";
    for {g in groupNonSync} {
        printf ",%.6g", vq_non_synchronous[g, d, t].val * group_capacity_for_scaling[g, d] >> "output_raw/vq_non_synchronous.csv";
    }
}

# Write vq_capacity_margin.  Agent 9b: * group_capacity_for_scaling[g, d]
# to un-scale the row-scaling division applied in capacityMargin
# constraint (Agent 5c).  No t dimension.  Mode A: group_cap = 1, no
# effect.
if p_model["solveFirst"] == 1 then {
  printf "solve,period" > "output_raw/vq_capacity_margin.csv";
  for {g in groupCapacityMargin} {printf ",%s", g >> "output_raw/vq_capacity_margin.csv";}
}
for {s in solve_current, d in d_realize_invest} {
    printf "\n%s,%s", s, d >> "output_raw/vq_capacity_margin.csv";
    for {g in groupCapacityMargin} {
        printf ",%.6g", vq_capacity_margin[g, d].val * group_capacity_for_scaling[g, d] >> "output_raw/vq_capacity_margin.csv";
    }
}

# Write vq_state_up_group.  Agent 9b: * group_capacity_for_scaling[g, d]
# to un-scale the row-scaling division applied in
# group_loss_share_constraint (Agent 5c).  Mode A: group_cap = 1, no
# effect.  group_loss_share_constraint is emitted at every fine (d, t),
# so vq_state_up_group needs no block expansion.
if p_model["solveFirst"] == 1 then {
  printf "solve,period,time" > "output_raw/vq_state_up_group.csv";
  for {g in group_loss_share} {printf ",%s", g >> "output_raw/vq_state_up_group.csv";}
}
for {s in solve_current, (d, t) in dt_realize_dispatch} {
    printf "\n%s,%s,%s", s, d, t >> "output_raw/vq_state_up_group.csv";
    for {g in group_loss_share} {
        printf ",%.6g", vq_state_up_group[g, d, t].val * group_capacity_for_scaling[g, d] >> "output_raw/vq_state_up_group.csv";
    }
}

# Write v_dual_node_balance
if p_model["solveFirst"] == 1 then {
  printf "solve,period,time" > "output_raw/v_dual_node_balance.csv";
  for {n in nodeBalance} {printf ",%s", n >> "output_raw/v_dual_node_balance.csv";}
}
for {s in solve_current, (d, t, t_previous, t_previous_within_timeset, d_previous, t_previous_within_solve) in dtttdt} {
    printf "\n%s,%s,%s", s, d, t >> "output_raw/v_dual_node_balance.csv";
    for {n in nodeBalance} {
        # Agent 9b: divide by node_capacity_for_scaling[n, d] to un-scale
        # the row division (Agent 5c).  The .dual value of a row-scaled
        # constraint is node_cap * original_dual; dividing by node_cap
        # recovers the user-facing nodal price (EUR / MWh).  Mode A:
        # node_cap = 1, no effect.
        # Agent 1.4: nodeBalance_eq gained a leading `bn` subscript (the
        # node's temporal-resolution block).  In the degenerate case
        # every node is in 'default' so the dual lookup resolves to the
        # single block row.  For rows at (d, t) outside the node's
        # block's timeline the constraint is not emitted — sum over
        # block_dtttdt filtered by (n, bn) picks only the row that
        # actually exists.
        printf ",%.6g",
          (if n not in nodeStateBlock then
             -sum {(n, bn) in node__block
                   : (bn, d, t, t_previous, t_previous_within_timeset, d_previous, t_previous_within_solve) in block_dtttdt}
                nodeBalance_eq[s, n, bn, d, t, t_previous, t_previous_within_timeset, d_previous, t_previous_within_solve].dual
           else
             -sum {(d, b, t) in period_block_time} nodeBalanceBlock_eq[s, n, d, b].dual)
          / p_inflation_factor_operations_yearly[d]
          / scale_the_objective
          / node_capacity_for_scaling[n, d] >> "output_raw/v_dual_node_balance.csv";
        }
}

# Write v_dual_reserve_balance
param fn_group_reserve_dual__dt symbolic := "output_raw/v_dual_reserve__upDown__group__period__t.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf 'solve,period,time' > fn_group_reserve_dual__dt;
    for {(r, ud, g) in reserve__upDown__group}
      { printf ',%s', r >> fn_group_reserve_dual__dt; }
    printf '\n,,' >> fn_group_reserve_dual__dt;
    for {(r, ud, g) in reserve__upDown__group}
      { printf ',%s', ud >> fn_group_reserve_dual__dt; }
    printf '\n,,' >> fn_group_reserve_dual__dt;
    for {(r, ud, g) in reserve__upDown__group}
      { printf ',%s', g >> fn_group_reserve_dual__dt; }
  }
for {s in solve_current, (d, t) in dt_realize_dispatch}
  {
    printf '\n%s,%s,%s', s, d, t >> fn_group_reserve_dual__dt;
    for {(r, ud, g) in reserve__upDown__group}
      {
    for {(r, ud, g, r_m) in reserve__upDown__group__method : r_m <> 'no_reserve'}
        printf ',%.8g', ( if ud = 'up' then
		                    max(( if (r, ud, g, r_m) in reserve__upDown__group__method_timeseries then reserveBalance_timeseries_eq[r, ud, g, r_m, d, t].dual else 0 ),
							    ( if (r, ud, g, r_m) in reserve__upDown__group__method_dynamic    then reserveBalance_dynamic_eq[r, ud, g, r_m, d, t].dual else 0 ),
							    ( if (r, ud, g, r_m) in reserve__upDown__group__method_n_1
								     and card({p_n_1 in process_large_failure : sum{(p_n_1, sink) in process_sink : (g, sink) in group_node} 1}) > 0
								  then max{p_n_1 in process_large_failure : sum{(p_n_1, sink) in process_sink : (g, sink) in group_node} 1} reserveBalance_up_n_1_eq[r, g, r_m, p_n_1, d, t].dual else 0 )
							   )
						  else
		                    max(( if (r, ud, g, r_m) in reserve__upDown__group__method_timeseries then reserveBalance_timeseries_eq[r, ud, g, r_m, d, t].dual else 0 ),
							    ( if (r, ud, g, r_m) in reserve__upDown__group__method_dynamic    then reserveBalance_dynamic_eq[r, ud, g, r_m, d, t].dual else 0 ),
							    ( if (r, ud, g, r_m) in reserve__upDown__group__method_n_1
								     and card({p_n_1 in process_large_failure : sum{(p_n_1, source) in process_source : (g, source) in group_node} 1}) > 0
								  then max{p_n_1 in process_large_failure : sum{(p_n_1, source) in process_source : (g, source) in group_node} 1} reserveBalance_down_n_1_eq[r, g, r_m, p_n_1, d, t].dual else 0 )
							   )
						) / p_inflation_factor_operations_yearly[d] * complete_period_share_of_year[d]
		    >> fn_group_reserve_dual__dt;
      }
  }

# Write v_dual_invest_unit
for {i in 1..1 : p_model['solveFirst']}
  { printf 'solve,period' > "output_raw/v_dual_invest_unit.csv";
    for {e in entityInvest : e in process_unit}
	  { printf ',%s', e >> "output_raw/v_dual_invest_unit.csv"; }
  }
for {s in solve_current, d in d_realize_invest : 'yes' not in exclude_entity_outputs}
  { printf '\n%s,%s', s, d >> "output_raw/v_dual_invest_unit.csv";
    for {e in entityInvest : e in process_unit}
      {
	    printf ',%.8g', (if (e, d) in ed_invest then v_invest[e, d].dual) >> "output_raw/v_dual_invest_unit.csv";
      }
  }

# Write v_dual_invest_connection
for {i in 1..1 : p_model['solveFirst']}
  { printf 'solve,period' > "output_raw/v_dual_invest_connection.csv";
    for {e in entityInvest : e in process_connection}
	  { printf ',%s', e >> "output_raw/v_dual_invest_connection.csv"; }
  }
for {s in solve_current, d in d_realize_invest : 'yes' not in exclude_entity_outputs}
  { printf '\n%s,%s', s, d >> "output_raw/v_dual_invest_connection.csv";
    for {e in entityInvest : e in process_connection}
      {
	    printf ',%.8g', (if (e, d) in ed_invest then v_invest[e, d].dual) >> "output_raw/v_dual_invest_connection.csv";
      }
  }

# Write v_dual_invest_node
for {i in 1..1 : p_model['solveFirst']}
  { printf 'solve,period' > "output_raw/v_dual_invest_node.csv";
    for {e in entityInvest : e in node}
	  { printf ',%s', e >> "output_raw/v_dual_invest_node.csv"; }
  }
for {s in solve_current, d in d_realize_invest : 'yes' not in exclude_entity_outputs}
  { printf '\n%s,%s', s, d >> "output_raw/v_dual_invest_node.csv";
    for {e in entityInvest : e in node}
      {
	    printf ',%.8g', (if (e, d) in ed_invest then v_invest[e, d].dual) >> "output_raw/v_dual_invest_node.csv";
      }
  }

# Investment constraint duals - raw output per constraint type
# Entity-level constraint duals
for {i in 1..1 : p_model['solveFirst']}
  { printf 'solve,period' > "output_raw/v_dual_maxInvest_period.csv";
    for {e in entityInvest : (e, 'invest_period') in entity__invest_method || (e, 'invest_period_total') in entity__invest_method
                          || (e, 'invest_retire_period') in entity__invest_method || (e, 'invest_retire_period_total') in entity__invest_method}
      { printf ',%s', e >> "output_raw/v_dual_maxInvest_period.csv"; }
  }
for {s in solve_current, d in d_realize_invest : 'yes' not in exclude_entity_outputs}
  { printf '\n%s,%s', s, d >> "output_raw/v_dual_maxInvest_period.csv";
    for {e in entityInvest : (e, 'invest_period') in entity__invest_method || (e, 'invest_period_total') in entity__invest_method
                          || (e, 'invest_retire_period') in entity__invest_method || (e, 'invest_retire_period_total') in entity__invest_method}
      { printf ',%.8g', (if (e, d) in ed_invest_period then maxInvest_entity_period[e, d].dual / scale_the_objective else 0)
                        >> "output_raw/v_dual_maxInvest_period.csv"; }
  }

for {i in 1..1 : p_model['solveFirst']}
  { printf 'solve,period' > "output_raw/v_dual_maxInvest_total.csv";
    for {e in e_invest_total}
	  { printf ',%s', e >> "output_raw/v_dual_maxInvest_total.csv"; }
  }
for {s in solve_current, d in d_realize_invest : 'yes' not in exclude_entity_outputs}
  { printf '\n%s,%s', s, d >> "output_raw/v_dual_maxInvest_total.csv";
    for {e in e_invest_total}
      { printf ',%.8g', maxInvest_entity_total[e, d].dual / scale_the_objective >> "output_raw/v_dual_maxInvest_total.csv"; }
  }

for {i in 1..1 : p_model['solveFirst']}
  { printf 'solve,period' > "output_raw/v_dual_maxCumulative.csv";
    for {e in entityInvest : (e, 'cumulative_limits') in entity__invest_method}
	  { printf ',%s', e >> "output_raw/v_dual_maxCumulative.csv"; }
  }
for {s in solve_current, d in d_realize_invest : 'yes' not in exclude_entity_outputs}
  { printf '\n%s,%s', s, d >> "output_raw/v_dual_maxCumulative.csv";
    for {e in entityInvest : (e, 'cumulative_limits') in entity__invest_method}
      { printf ',%.8g', (if (e, d) in ed_invest_cumulative then maxCumulative_capacity[e, d].dual / scale_the_objective else 0)
                        >> "output_raw/v_dual_maxCumulative.csv"; }
  }

# Group-level investment constraint duals
for {i in 1..1 : p_model['solveFirst']}
  { printf 'solve,period' > "output_raw/v_dual_maxInvestGroup_period.csv";
    for {g in group_invest : (g, 'invest_period') in group__invest_method || (g, 'invest_period_total') in group__invest_method
                          || (g, 'invest_retire_period') in group__invest_method || (g, 'invest_retire_period_total') in group__invest_method}
	  { printf ',%s', g >> "output_raw/v_dual_maxInvestGroup_period.csv"; }
  }
for {s in solve_current, d in d_realize_invest : 'yes' not in exclude_entity_outputs}
  { printf '\n%s,%s', s, d >> "output_raw/v_dual_maxInvestGroup_period.csv";
    for {g in group_invest : (g, 'invest_period') in group__invest_method || (g, 'invest_period_total') in group__invest_method
                          || (g, 'invest_retire_period') in group__invest_method || (g, 'invest_retire_period_total') in group__invest_method}
      { printf ',%.8g', (if (g, d) in gd_invest_period then maxInvestGroup_entity_period[g, d].dual / scale_the_objective else 0)
                        >> "output_raw/v_dual_maxInvestGroup_period.csv"; }
  }

for {i in 1..1 : p_model['solveFirst']}
  { printf 'solve,period' > "output_raw/v_dual_maxInvestGroup_total.csv";
    for {g in g_invest_total}
	  { printf ',%s', g >> "output_raw/v_dual_maxInvestGroup_total.csv"; }
  }
for {s in solve_current, d in d_realize_invest : 'yes' not in exclude_entity_outputs}
  { printf '\n%s,%s', s, d >> "output_raw/v_dual_maxInvestGroup_total.csv";
    for {g in g_invest_total}
      { printf ',%.8g', maxInvestGroup_entity_total[g, d].dual / scale_the_objective >> "output_raw/v_dual_maxInvestGroup_total.csv"; }
  }

for {i in 1..1 : p_model['solveFirst']}
  { printf 'solve,period' > "output_raw/v_dual_maxInvestGroup_cumulative.csv";
    for {g in g_invest_cumulative}
	  { printf ',%s', g >> "output_raw/v_dual_maxInvestGroup_cumulative.csv"; }
  }
for {s in solve_current, d in d_realize_invest : 'yes' not in exclude_entity_outputs}
  { printf '\n%s,%s', s, d >> "output_raw/v_dual_maxInvestGroup_cumulative.csv";
    for {g in g_invest_cumulative}
      { printf ',%.8g', (if pdGroup[g, 'cumulative_max_capacity', d] then maxInvestGroup_entity_cumulative[g, d].dual / scale_the_objective else 0)
                        >> "output_raw/v_dual_maxInvestGroup_cumulative.csv"; }
  }

# CO2 emission-cap duals. Both sides of co2_max_period and co2_max_total are
# divided by 1000 (Mt instead of t), so the raw dual is Currency/Mt. Downstream
# processing DIVIDES by 1000 (because Δ(scaled-RHS) = Δ(raw-RHS)/1000) and also
# divides by the operational inflation factor to report nominal Currency/tCO2.
for {i in 1..1 : p_model['solveFirst']}
  { printf 'solve,period' > "output_raw/v_dual_co2_max_period.csv";
    for {g in group_co2_max_period}
      { printf ',%s', g >> "output_raw/v_dual_co2_max_period.csv"; }
  }
for {s in solve_current, d in period_in_use}
  { printf '\n%s,%s', s, d >> "output_raw/v_dual_co2_max_period.csv";
    for {g in group_co2_max_period}
      { printf ',%.8g', co2_max_period[g, d].dual / scale_the_objective
                        >> "output_raw/v_dual_co2_max_period.csv"; }
  }

for {i in 1..1 : p_model['solveFirst']}
  { printf 'solve' > "output_raw/v_dual_co2_max_total.csv";
    for {g in group_co2_max_total}
      { printf ',%s', g >> "output_raw/v_dual_co2_max_total.csv"; }
  }
for {s in solve_current}
  { printf '\n%s', s >> "output_raw/v_dual_co2_max_total.csv";
    for {g in group_co2_max_total}
      { printf ',%.8g', co2_max_total[g].dual / scale_the_objective
                        >> "output_raw/v_dual_co2_max_total.csv"; }
  }

param w_raw := gmtime();
printf "Timer - Write raw output: %ss\n", w_raw - r_solution;
printf '%s,%s\n', 'w_raw', w_raw - r_solution >> mod_phases_file;

# hours_in_realized_period and realized_period_share_of_year removed —
# unused in mod (no readers); the Python post-processor recomputes both
# directly from step_duration in process_outputs/calc_capacity_flows.py.

param entity_all_capacity{e in entity, d in period} :=
  + p_entity_all_existing[e, d]
  + sum {(e, d_invest, d) in edd_invest} v_invest[e, d_invest].val * p_entity_unitsize[e]
  - sum {(e, d_divest) in ed_divest : p_years_d[d_divest] <= p_years_d[d]} v_divest[e, d_divest].val * p_entity_unitsize[e]
;

# Write entity_all_capacity (existing + cumulative invest - divest, computed by the model)
if p_model["solveFirst"] == 1 then {
  printf "solve,period" > "output_raw/entity_all_capacity.csv";
  for {e in entity} {printf ",%s", e >> "output_raw/entity_all_capacity.csv";}
}
for {s in solve_current, d in d_realize_dispatch_or_invest} {
    printf "\n%s,%s", s, d >> "output_raw/entity_all_capacity.csv";
    for {e in entity} {
        printf ",%.8g", entity_all_capacity[e, d] >> "output_raw/entity_all_capacity.csv";
    }
}

param r_process_Online__dt{p in process_online, (d, t) in dt} :=
  + (if p in process_online_linear then v_online_linear[p, d, t].val)
  + (if p in process_online_integer then v_online_integer[p, d, t].val);

param r_process__source__sink_Flow__dt{(p, source, sink) in process_source_sink_alwaysProcess, (d, t) in dt} :=
  + sum {(p, m) in process_method : m in method_1var_per_way}
    ( + sum {(p, source, sink2) in process_source_toSink}
        ( + v_flow[p, source, sink2, d, t].val * p_entity_unitsize[p]
	          * pdtProcess_slope[p, d, t]
	  		  * (if p in process_unit then p_process_sink_flow_coefficient[p, sink2] / p_process_source_flow_coefficient[p, source] else 1)
          + (if (p, 'min_load_efficiency') in process__ct_method then
			  + r_process_Online__dt[p, d, t]
			    * pdtProcess_section[p, d, t] * p_entity_unitsize[p])
	    )
      + sum {(p, source2, sink) in process_source_toSink}
          + v_flow[p, source2, sink, d, t].val * p_entity_unitsize[p]
      + sum {(p, source, sink2) in process_sink_toSource}
        ( + v_flow[p, source, sink2, d, t].val * p_entity_unitsize[p]
	          * pdtProcess_slope[p, d, t]
          + (if (p, 'min_load_efficiency') in process__ct_method then
			  + ( + (if p in process_online_linear then v_online_linear[p, d, t])
			      + (if p in process_online_integer then v_online_integer[p, d, t])
				)
	            * pdtProcess_section[p, d, t] * p_entity_unitsize[p])
	    )
      + sum {(p, source2, sink) in process_sink_toSource}
          + v_flow[p, source2, sink, d, t].val * p_entity_unitsize[p]
      + (if (p, source, sink) in process__profileProcess__toSink then
	      + v_flow[p, source, sink, d, t].val * p_entity_unitsize[p])
      + (if (p, source, sink) in process__source__toProfileProcess then
	      + v_flow[p, source, sink, d, t].val * p_entity_unitsize[p])
	  + (if (p, source, sink) in process_process_toSink then
	      + v_flow[p, source, sink, d, t].val * p_entity_unitsize[p])
	  + (if (p, source, sink) in process_source_toProcess then
	      + v_flow[p, source, sink, d, t].val * p_entity_unitsize[p])
   )
  + sum {(p, m) in process_method : m in method_nvar} (
      + v_flow[p, source, sink, d, t].val * p_entity_unitsize[p]
	)
  + sum {(p, source, sink2) in process_source_sink : (p, 'method_2way_1var_off') in process_method && (p, source, sink) in process_source_toProcess_direct} (
      + v_flow[p, source, sink2, d, t].val * p_entity_unitsize[p]
	)
  + sum {(p, source2, sink) in process_source_sink : (p, 'method_2way_1var_off') in process_method && (p, source, sink) in process_process_toSink_direct} (
      + v_flow[p, source2, sink, d, t].val * p_entity_unitsize[p]
    )
;

param r_storage_usage_dt{(n,'fix_usage') in node__storage_nested_fix_method, (d, t) in dt_fix_storage_timesteps}:=
    + sum{(p, n, sink) in process_source_sink_alwaysProcess} r_process__source__sink_Flow__dt[p, n, sink, d, t] * step_duration[d,t]
    - sum{(p, source, n) in process_source_sink_alwaysProcess} r_process__source__sink_Flow__dt[p, source, n, d, t] * step_duration[d,t]
    ;


#### Transfer variables to next solve
param fn_entity_period_existing_capacity symbolic := "solve_data/p_entity_period_existing_capacity.csv";
printf 'entity,period,p_entity_period_existing_capacity,p_entity_period_invested_capacity\n' > fn_entity_period_existing_capacity;
for {(e, d) in ed_history_realized union {e in entity, d in d_realize_invest}}
  {
    printf '%s,%s,%.8g,%.8g\n', e, d,
	  + (if p_model['solveFirst'] && d in period_first then p_entity_pre_existing[e, d])
	  + (if not p_model['solveFirst'] && (e, d) in ed_history_realized then p_entity_period_existing_capacity[e, d])
	  + (if (e, d) in ed_invest && d in d_realize_invest then ( v_invest[e, d].val ) * p_entity_unitsize[e]),
	  + (if not p_model['solveFirst'] && (e, d) in ed_history_realized  then p_entity_period_invested_capacity[e, d])
	  + (if (e, d) in ed_invest && d in d_realize_invest then ( v_invest[e, d].val ) * p_entity_unitsize[e])
	>> fn_entity_period_existing_capacity;
  }

printf 'Transfer divestments to the next solve...\n';
param fn_entity_divested symbolic := "solve_data/p_entity_divested.csv";
printf 'entity,p_entity_divested\n' > fn_entity_divested;
for {e in entityDivest}
  {
    printf '%s,%.8g\n', e,
	  + (if not p_model['solveFirst'] then p_entity_divested[e] else 0)
	  + sum {(e, d_divest) in ed_divest} v_divest[e, d_divest].val * p_entity_unitsize[e]
	>> fn_entity_divested;
  }

printf 'Write node state quantity for fixed timesteps ..\n';
param fn_fix_quantity_nodeState__dt symbolic := "solve_data/fix_storage_quantity.csv";
# Truncate + header only when this solve will write rows — otherwise
# preserve the prior solve's handoff content.  The canonical header is
# also written by the phase-1 init (above mod:638) on the very first
# solve, so a first solve with no fix_storage still produces a
# well-formed header-only file for the next solve's read at mod:652.
for {i in 1..1 : exists{(d,t) in dt_fix_storage_timesteps} 1}
  { printf 'period,step,node,p_fix_storage_quantity\n' > fn_fix_quantity_nodeState__dt;
  }
for {(n,'fix_quantity') in node__storage_nested_fix_method, (d, t) in dt_fix_storage_timesteps}
  {
    printf '%s,%s,%s,%.8g\n', d, t, n, v_state[n, d, t].val * p_entity_unitsize[n]>> fn_fix_quantity_nodeState__dt;
  }

printf 'Write node state price for fixed timesteps ..\n';
param fn_fix_price_nodeState__dt symbolic := "solve_data/fix_storage_price.csv";
for {i in 1..1 : exists{(d,t) in dt_fix_storage_timesteps} 1}
  { printf 'period,step,node,p_fix_storage_price\n' > fn_fix_price_nodeState__dt;
  }
for {c in solve_current, (n,'fix_price') in node__storage_nested_fix_method, (d, t, t_previous, t_previous_within_timeset, d_previous, t_previous_within_solve) in dtttdt: (d, t) in dt_fix_storage_timesteps}
  {
    # Agent 9b: / node_capacity_for_scaling to un-scale the row scaling
    # on nodeBalance_eq (Agent 5c).  Mode A: node_cap = 1, no effect.
    # Agent 1.4: sum over the node's block rows of block_dtttdt so the
    # dual lookup works for both degenerate 'default' and non-default
    # nodes — exactly one row matches at each (n, d, t) in the default
    # case, bit-identical to the old single-dual lookup.
    printf '%s,%s,%s,%.8g\n', d, t, n,
      -sum {(n, bn) in node__block
            : (bn, d, t, t_previous, t_previous_within_timeset, d_previous, t_previous_within_solve) in block_dtttdt}
         nodeBalance_eq[c, n, bn, d, t, t_previous, t_previous_within_timeset, d_previous, t_previous_within_solve].dual
      / p_inflation_factor_operations_yearly[d] * complete_period_share_of_year[d] / scale_the_objective / node_capacity_for_scaling[n, d] >> fn_fix_price_nodeState__dt;
  }

printf 'Write node state usage for fixed timesteps ..\n';
param fn_fix_usage_nodeState__dt symbolic := "solve_data/fix_storage_usage.csv";
for {i in 1..1 : exists{(d,t) in dt_fix_storage_timesteps} 1}
  { printf 'period,step,node,p_fix_storage_usage\n' > fn_fix_usage_nodeState__dt;
  }
for {(n,'fix_usage') in node__storage_nested_fix_method, (d, t) in dt_fix_storage_timesteps}
  {
    printf '%s,%s,%s,%.8g\n', d, t, n, r_storage_usage_dt[n,d,t] >> fn_fix_usage_nodeState__dt;
  }
for {s in solve_current}
  printf '%s\n', s >> "solve_data/a_test.csv";
for {(d, t) in realized_period__time_last}
  printf '%s,%s\n', d, t >> "solve_data/a_test.csv";
printf '\n' >> "solve_data/a_test.csv";

for {s in solve_current}
  printf '%s\n', s >> "solve_data/aa_test.csv";
for {(d, t) in dt_realize_dispatch}
  printf '%s,%s\n', d, t >> "solve_data/aa_test.csv";
printf '\n' >> "solve_data/aa_test.csv";

for {s in solve_current}
  printf '%s\n', s >> "solve_data/aaa_test.csv";
for {(d, t, t_previous, t_previous_within_timeset, d_previous, t_previous_within_solve) in dtttdt}
  printf '%s,%s,%s,%s,%s,%s\n',d, t, t_previous, t_previous_within_timeset, d_previous, t_previous_within_solve >> "solve_data/aaa_test.csv";
printf '\n' >> "solve_data/aaa_test.csv";

printf 'Write node state last timestep ..\n';
param fn_p_roll_continue_state symbolic := "solve_data/p_roll_continue_state.csv";
# write over only if in a dispatch roll, storage solve should not create this
for {n in nodeState, (d, t) in realized_period__time_last}
  {
    printf 'node,p_roll_continue_state\n' > fn_p_roll_continue_state;
  }
for {n in nodeState, (d, t) in realized_period__time_last}
  {
    printf '%s,%.8g\n', n, v_state[n, d, t].val * p_entity_unitsize[n]  >> fn_p_roll_continue_state;
  }

param fn_period_capacity symbolic := "solve_data/period_capacity.csv";
printf 'period\n' > fn_period_capacity;
for {d in period_capacity union d_realize_dispatch_or_invest}
  { printf '%s\n', d >> fn_period_capacity; }


#### Write out results
printf 'Write unit capacity results...\n';
param fn_unit_capacity symbolic := "solve_data/unit_capacity__period.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf 'unit,solve,period,existing,invested,divested,total\n' > fn_unit_capacity; }  # Clear the file on the first solve
for {s in solve_current, p in process_unit, d in d_realize_dispatch_or_invest: 'yes' not in exclude_entity_outputs && d not in period_capacity}
  {
    printf '%s,%s,%s,%.8g,%.8g,%.8g,%.8g\n', p, s, d,
	        p_entity_all_existing[p, d],
			(if (p, d) in pd_invest then v_invest[p, d].val * p_entity_unitsize[p] else 0),
			(if (p, d) in pd_divest then v_divest[p, d].val * p_entity_unitsize[p] else 0),
			entity_all_capacity[p, d]
	>> fn_unit_capacity;
  }

printf 'Write connection capacity results...\n';
param fn_connection_capacity symbolic := "solve_data/connection_capacity__period.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf 'connection,solve,period,existing,invested,divested,total\n' > fn_connection_capacity; }  # Clear the file on the first solve
for {s in solve_current, p in process_connection, d in d_realize_dispatch_or_invest: 'yes' not in exclude_entity_outputs && d not in period_capacity}
  {
    printf '%s,%s,%s,%.8g,%.8g,%.8g,%.8g\n', p, s, d,
	        p_entity_all_existing[p, d],
			(if (p, d) in pd_invest then v_invest[p, d].val * p_entity_unitsize[p] else 0),
			(if (p, d) in pd_divest then v_divest[p, d].val * p_entity_unitsize[p] else 0),
			+ entity_all_capacity[p, d]
	>> fn_connection_capacity;
  }

printf 'Write node/storage capacity results...\n';
param fn_node_capacity symbolic := "solve_data/node_capacity__period.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf 'node,solve,period,existing,invested,divested,total\n' > fn_node_capacity; }  # Clear the file on the first solve
for {s in solve_current, e in nodeState, d in d_realize_dispatch_or_invest: 'yes' not in exclude_entity_outputs && d not in period_capacity}
  {
    printf '%s,%s,%s,%.8g,%.8g,%.8g,%.8g\n', e, s, d,
	        p_entity_all_existing[e, d],
			(if (e, d) in ed_invest then v_invest[e, d].val * p_entity_unitsize[e] else 0),
			(if (e, d) in ed_divest then v_divest[e, d].val * p_entity_unitsize[e] else 0),
			+ entity_all_capacity[e, d]
	 >> fn_node_capacity;
  }

param w_capacity := gmtime();
printf "Timer - Write capacity: %ss\n", w_capacity - w_raw;
printf '%s,%s\n', 'w_capacity', w_capacity - w_raw >> mod_phases_file;

#display period_first;
#display period;
#display period__time_first;
#display period_last;
#display branch;
#display branch_all;
#display period__branch;
#display solve_branch__time_branch;
#display {d in period_in_use}: pd_branch_weight[d];
#display {d in period_in_use}: complete_period_share_of_year[d];
#display {p in process, (d,t) in dt}: pdtProcess_slope[p, d, t];
#display {n in nodeBalance union nodeBalancePeriod, b in branch, (b,t) in dt, (b, ts) in period__time_first}: pbt_node_inflow[n, b, ts, t];
#display {(g, n) in group_node, (d,t) in dt : g in groupStochastic}: pdtNodeInflow[n, d, t];
#display {(p,n,f,m) in process__node__profile__profile_method, (d,t) in dt}: pdtProfile[f,d,t];
#display {(p,n,p2) in process_source_sink, (d,t) in dt: n in nodeState} v_flow[p,n,p2,d,t];
#display {(p,p2,n) in process_source_sink, (d,t) in dt: n in nodeState} v_flow[p,p2,n,d,t];
#display {n in nodeState, (d,t) in dt}: v_state[n,d,t];
#display dtttdt;
#display {(p, r, ud, n, d, t) in prundt : sum{(r, ud, g) in reserve__upDown__group} 1 } : v_reserve[p, r, ud, n, d, t].dual / p_entity_unitsize[p];
#display {(p, source, sink) in process_source_sink_alwaysProcess, (d, t) in dt : (d, t) in test_dt}: r_process__source__sink_Flow__dt[p, source, sink, d, t];
#display {p in process, (d, t) in dt : (d, t) in test_dt}: r_cost_process_other_operational_cost_dt[p, d, t];
#display {(p, source, sink, d, t) in peedt : (d, t) in test_dt}: v_flow[p, source, sink, d, t].val;
#display {(p, source, sink, d, t) in peedt : (d, t) in test_dt}: v_flow[p, source, sink, d, t].dual;
#display {p in process_online, (d, t) in dt : (d, t) in test_dt} : r_process_Online__dt[p, d, t];
#display {n in nodeState, (d, t) in dt : (d, t) in test_dt}: v_state[n, d, t].val;
#display {n in nodeBalance union nodeBalancePeriod, (d, t) in dt : (d, t) in test_dt}: pdtNodeInflow[n, d, t];
#display {n in nodeBalance union nodeBalancePeriod, (d, t) in dt : (d, t) in test_dt}: pdtNode[n, 'penalty_up', d, t];
#display {n in nodeState, (d, t) in dt : (d, t) in test_dt}: pdtNode[n, 'availability', d, t];
#display {n in nodeState, (d, t) in dt : (d, t) in test_dt}: v_state[n, d, t].val * p_entity_unitsize[n];
#display {(p, r, ud, n, d, t) in prundt : (d, t) in test_dt}: v_reserve[p, r, ud, n, d, t].val * p_entity_unitsize[p];
#display {(r, ud, ng) in reserve__upDown__group, (d, t) in test_dt}: vq_reserve[r, ud, ng, d, t].val * pdtReserve_upDown_group[r, ud, ng, 'reservation', d, t];
#display {n in (nodeBalance union nodeBalancePeriod), (d, t) in dt : (d, t) in test_dt}: vq_state_up[n, d, t].val * node_capacity_for_scaling[n, d];
#display {n in (nodeBalance union nodeBalancePeriod), (d, t) in dt : (d, t) in test_dt}: vq_state_down[n, d, t].val * node_capacity_for_scaling[n, d];
#display {g in groupInertia, (d, t) in dt : (d, t) in test_dt}: inertia_constraint[g, d, t].dual;
#display {c in solve_current, n in nodeBalance, (d, t, t_previous, t_previous_within_timeset, d_previous, t_previous_within_solve) in dtttdt : (d, t) in test_dt}: -nodeBalance_eq[n, d, t, t_previous, t_previous_within_timeset, d_previous, t_previous_within_solve].dual / p_inflation_factor_operations_yearly[d] * complete_period_share_of_year[d] / scale_the_objective;
#display {(r, ud, g, r_m) in reserve__upDown__group__method_timeseries, (d, t) in dt : (d, t) in test_dt}: reserveBalance_timeseries_eq[r, ud, g, r_m, d, t].dual;
#display {(p, source, sink) in process_source_sink, (d, t) in dt : (d, t) in test_dt && (p, sink) in process_sink}: maxToSink[p, source, sink, d, t].ub;
#display {(p, sink, source) in process_sink_toSource, (d, t) in dt : (d, t) in test_dt}: maxToSource[p, sink, source, d, t].ub;
#display {(p, m) in process_method, (d, t) in dt : (d, t) in test_dt && m in method_indirect} conversion_indirect[p, m, d, t].ub;
#display {(p, source, sink, f, m) in process__source__sink__profile__profile_method, (d, t) in dt : (d, t) in test_dt && m = 'lower_limit'}: profile_flow_lower_limit[p, source, sink, f, d, t].dual;
#display {(p, sink) in process_sink, param in sourceSinkTimeParam, (d, t) in test_dt}: ptProcess_sink[p, sink, param, t];
display v_invest, v_divest, solve_current, total_cost;
#display {(p, source, sink) in process_source_sink, (d, t) in test_dt}: pdtProcess__source__sink__dt_varCost[p, source, sink, d, t];
#display test_dt;
#display {n in nodeState, (d,t) in period__time_last: (n, 'use_reference_value') in node__storage_solve_horizon_method}: pdtNode[n,'storage_state_reference_value',d,t];
end;
