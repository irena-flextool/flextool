# © International Renewable Energy Agency 2018-2022

#The FlexTool is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License
#as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

#The FlexTool is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
#without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

#You should have received a copy of the GNU Lesser General Public License along with the FlexTool.  
#If not, see <https://www.gnu.org/licenses/>.

#Author: Juha Kiviluoma (2017-2022), VTT Technical Research Centre of Finland

param datetime0 := gmtime();
display datetime0;

#########################
# Fundamental sets of the model
set entity 'e - contains both nodes and processes';
set process 'p - Particular activity that transfers, converts or stores commodities' within entity;
set processUnit 'Unit processes' within process;
set node 'n - Any location where a balance needs to be maintained' within entity;
set group 'g - Any group of entities that have a set of common constraints';
set commodity 'c - Stuff that is being processed';
set period_time '(d, t) - Time steps in the time periods of the whole timeline' dimen 2;
set period__time_first within period_time;
set period__time_last within period_time;
set solve_period_timeblockset '(solve, d, tb) - All solve, period, timeblockset combinations in the model instance' dimen 3;
set solve_period '(solve, d) - Time periods in the solves to extract periods that can be found in the full data' := setof {(s, d, tb) in solve_period_timeblockset} (s, d);
set period_solve 'picking up periods from solve_period' := setof {(s,d) in solve_period} (d);
set solve_current 'current solve name' dimen 1;
set period 'd - Time periods in the current solve' := setof {(d, t) in period_time} (d);
set period_first := {d in period : sum{d2 in period : d2 <= d} 1 = 1};
set period_last  := {d in period : sum{d2 in period : d2 <= d} 1 = card(period)};
set timeline__timestep__duration dimen 3;
set time 't - Time steps in the current timelines' := setof {(tl, t, duration) in timeline__timestep__duration} (t); 
set timeblockset__timeline dimen 2;
set timeline := setof{(tb, tl) in timeblockset__timeline} (tl);
set period__timeline := {d in period, tl in timeline : sum{(s, d, tb) in solve_period_timeblockset : s in solve_current && (tb, tl) in timeblockset__timeline} 1};
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
set nodeParam;
set nodePeriodParam;
set nodeTimeParam within nodeParam;
set processParam;
set processPeriodParam;
set processTimeParam within processParam;
set sourceSinkParam;
set sourceSinkTimeParam within sourceSinkParam;
set reserveParam;
set reserveTimeParam within reserveParam;
set groupParam;
set groupPeriodParam;
set groupTimeParam within groupParam;

set reserve__upDown__group__method dimen 4;
set reserve__upDown__group := setof {(r, ud, g, m) in reserve__upDown__group__method : m <> 'no_reserve'} (r, ud, g);
set reserve 'r - Categories for the reservation of capacity_existing' := setof {(r, ud, ng, r_m) in reserve__upDown__group__method} (r);
set reserve__upDown__group__reserveParam__time dimen 5 within {reserve, upDown, group, reserveTimeParam, time};

set group__param dimen 2 within {group, groupParam};
set group__param__period dimen 3; # within {group, groupPeriodParam, periodAll};
set node__param__period dimen 3; # within {node, nodePeriodParam, periodAll};
set commodity__param__period dimen 3; # within {commodity, commodityPeriodParam, periodAll};
set process__param__period dimen 3; # within {process, processPeriodParam, periodAll};

set period_group 'picking up periods from group data' := setof {(n, param, d) in group__param__period} (d);
set period_node 'picking up periods from node data' := setof {(n, param, d) in node__param__period} (d);
set period_commodity 'picking up periods from commodity data' := setof {(n, param, d) in commodity__param__period} (d);
set period_process 'picking up periods from process data' := setof {(n, param, d) in process__param__period} (d);

set periodAll 'd - Time periods in data (including those currently in use)' := period_group union period_node union period_commodity union period_process union period_solve;


param p_group {g in group, groupParam} default 0;
param pd_group {g in group, groupPeriodParam, d in periodAll} default 0;
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
set entity__invest_method 'the investment method applied to an entity' dimen 2 within {entity, invest_method};
set entityDivest := setof {(e, m) in entity__invest_method : m not in divest_method_not_allowed} (e);
set entityInvest := setof {(e, m) in entity__invest_method : m not in invest_method_not_allowed} (e);
param investableEntities := sum{e in entityInvest} 1;
set group__invest_method 'the investment method applied to a group' dimen 2 within {group, invest_method};
set group_invest := setof {(g, m) in group__invest_method : m not in invest_method_not_allowed} (g);
set group_divest := setof {(g, m) in group__invest_method : m not in divest_method_not_allowed} (g);
set nodeBalance 'nodes that maintain a node balance' within node;
set nodeState 'nodes that have a state' within node;
set inflow_method 'method for scaling the inflow';
set inflow_method_default within inflow_method;
set node__inflow_method_read 'method for scaling the inflow applied to a node' within {node, inflow_method};
set node__inflow_method dimen 2 within {node, inflow_method} :=
    {n in node, m in inflow_method : (n, m) in node__inflow_method_read || (sum{(n, m2) in node__inflow_method_read} 1 = 0 && m in inflow_method_default)};
set storage_binding_method 'methods for binding storage state between periods';
set storage_binding_method_default within storage_binding_method;
set node__storage_binding_method_read within {node, storage_binding_method};
set node__storage_binding_method dimen 2 within {node, storage_binding_method} :=
    {n in node, m in storage_binding_method : (n, m) in node__storage_binding_method_read || (sum{(n, m2) in node__storage_binding_method_read} 1 = 0 && m in storage_binding_method_default)};
set storage_start_end_method 'method to fix start and/or end value of storage in a model run';
set node__storage_start_end_method within {node, storage_start_end_method};
set storage_solve_horizon_method 'methods to set reference value or price for the end of horizon storage state';
set node__storage_solve_horizon_method within {node, storage_solve_horizon_method};
set node__profile__profile_method dimen 3 within {node,profile,profile_method};
set group_node 'member nodes of a particular group' dimen 2 within {group, node};
set group_process 'member processes of a particular group' dimen 2 within {group, process};
set group_process_node 'process__nodes of a particular group' dimen 3 within {group, process, node};
set group_entity := group_process union group_node;
set groupInertia 'node groups with an inertia constraint' within group;
set groupNonSync 'node groups with a non-synchronous constraint' within group;
set groupCapacityMargin 'node groups with a capacity margin' within group;
set groupOutput 'groups that will output aggregated results' within group;
set process_unit 'processes that are unit' within process;
set process_connection 'processes that are connections' within process;
set process__ct_method_read dimen 2 within {process, ct_method};
set process__ct_method dimen 2 within {process, ct_method} := 
    {p in process, m in ct_method 
	   : (p, m) in process__ct_method_read 
	   || (sum{(p, m2) in process__ct_method_read} 1 = 0 && p in process_connection && m in ct_method_regular)
	   || (sum{(p, m2) in process__ct_method_read} 1 = 0 && p in process_unit && m in ct_method_constant)};
set process__startup_method_read dimen 2 within {process, startup_method} default {p in process, 'no_startup'} ;
set process__startup_method dimen 2 within {process, startup_method}:=
    {p in process, m in startup_method : (p, m) in process__startup_method_read || (sum{(p, m2) in process__startup_method_read} 1 = 0 && m in startup_method_no)};
set process_node_ramp_method dimen 3 within {process, node, ramp_method};
set methods dimen 4; 
set process__profile__profile_method dimen 3 within {process, profile, profile_method};
set process__node__profile__profile_method dimen 4 within {process, node, profile, profile_method};
set process_source dimen 2 within {process, entity};
set process_sink dimen 2 within {process, entity};

set process__sink_nonSync_unit dimen 2 within {process, node};
set process_nonSync_connection dimen 1 within {process};

set process_reserve_upDown_node dimen 4;
set process_node_flow_constraint dimen 3 within {process, node, constraint};
set process_capacity_constraint dimen 2 within {process, constraint};
set node_capacity_constraint dimen 2 within {node, constraint};
set node_state_constraint dimen 2 within {node, constraint};
set constraint__sense dimen 2 within {constraint, sense};
set commodity_node dimen 2 within {commodity, node}; 

set dt dimen 2 within period_time;
set dtttdt dimen 6;
set period_invest dimen 1 within period;
set period_realized dimen 1 within period;


set startTime dimen 1 within time;
set startNext dimen 1 within time;
param startNext_index := sum{t in time, t_startNext in startNext : t <= t_startNext} 1;
set modelParam;

set process__param dimen 2 within {process, processParam};
set process__param__time dimen 3 within {process, processTimeParam, time};
set process__param_t := setof {(p, param, t) in process__param__time} (p, param);

set connection__param := {(p, param) in process__param : p in process_connection};
set connection__param__time := { (p, param, t) in process__param__time : (p in process_connection)};
set connection__param_t := setof {(connection, param, t) in connection__param__time} (connection, param);
set process__source__param dimen 3 within {process_source, sourceSinkParam};
set process__source__param__time dimen 4 within {process_source, sourceSinkTimeParam, time};
set process__source__param_t := setof {(p, source, param, t) in process__source__param__time} (p, source, param);
set process__sink__param dimen 3 within {process_sink, sourceSinkParam};
set process__sink__param__time dimen 4 within {process_sink, sourceSinkTimeParam, time};
set process__sink__param_t := setof {(p, sink, param, t) in process__sink__param__time} (p, sink, param);

set node__param__time dimen 3 within {node, nodeTimeParam, time};

param p_model {modelParam};
param p_commodity {c in commodity, commodityParam} default 0;
param pd_commodity {c in commodity, commodityPeriodParam, d in periodAll} default 0;

param p_node {node, nodeParam} default 0;
param pd_node {node, nodePeriodParam, periodAll} default 0;
param pt_node {node, nodeTimeParam, time} default 0;

param p_process_source {(p, source) in process_source, sourceSinkParam} default 0;
param pt_process_source {(p, source) in process_source, sourceSinkTimeParam, time} default 0;
param p_process_sink {(p, sink) in process_sink, sourceSinkParam} default 0;
param pt_process_sink {(p, sink) in process_sink, sourceSinkTimeParam, time} default 0;

param p_process_source_coefficient {(p, source) in process_source} default 1;
param p_process_sink_coefficient {(p, sink) in process_sink} default 1;

param pt_profile {profile, time};

param p_reserve_upDown_group {reserve, upDown, group, reserveParam} default 0;
param pt_reserve_upDown_group {reserve, upDown, group, reserveTimeParam, time};
param p_process_reserve_upDown_node {process, reserve, upDown, node, reserveParam} default 0;

param p_process {process, processParam} default 0;
param pd_process {process, processPeriodParam, periodAll} default 0;
param pt_process {process, processTimeParam, time} default 0;

param p_constraint_constant {constraint};
param p_process_node_constraint_flow_coefficient {process, node, constraint};
param p_process_constraint_capacity_coefficient {process, constraint};
param p_node_constraint_capacity_coefficient {node, constraint};
param p_node_constraint_state_coefficient {node, constraint};
param penalty_up {n in nodeBalance};
param penalty_down {n in nodeBalance};
param step_duration{(d, t) in dt};

param p_timeline_duration_in_years{timeline};
param p_discount_years{d in period} default 0;
param p_discount_rate{model} default 0.05;
param p_discount_offset_investment{model} default 0;    # Calculate investment cost discounting while assuming they are made at the begining of the year (unless other value is given)
param p_discount_offset_operations{model} default 0.5;  # Calculate operational costs assuming they are on average taking place at the middle of the year (unless other value is given)

param p_entity_invested {e in entity : e in entityInvest};
param p_entity_divested {e in entity : e in entityInvest};


#########################
# Read data
#table data IN 'CSV' '.csv' :  <- [];
# Domain sets
table data IN 'CSV' 'input/commodity.csv' : commodity <- [commodity];
table data IN 'CSV' 'input/constraint__sense.csv' : constraint <- [constraint];
table data IN 'CSV' 'input/debug.csv': debug <- [debug];
table data IN 'CSV' 'input/entity.csv': entity <- [entity];
table data IN 'CSV' 'input/group.csv' : group <- [group];
table data IN 'CSV' 'input/node.csv' : node <- [node];
table data IN 'CSV' 'input/nodeBalance.csv' : nodeBalance <- [nodeBalance];
table data IN 'CSV' 'input/nodeState.csv' : nodeState <- [nodeState];
table data IN 'CSV' 'input/groupInertia.csv' : groupInertia <- [groupInertia];
table data IN 'CSV' 'input/groupNonSync.csv' : groupNonSync <- [groupNonSync];
table data IN 'CSV' 'input/groupCapacityMargin.csv' : groupCapacityMargin <- [groupCapacityMargin];
table data IN 'CSV' 'input/groupOutput.csv' : groupOutput <- [groupOutput];
table data IN 'CSV' 'input/process.csv': process <- [process];
table data IN 'CSV' 'input/profile.csv': profile <- [profile];

# Single dimension membership sets
table data IN 'CSV' 'input/process_connection.csv': process_connection <- [process_connection];
table data IN 'CSV' 'input/process_nonSync_connection.csv': process_nonSync_connection <- [process];
table data IN 'CSV' 'input/process_unit.csv': process_unit <- [process_unit];

# Multi dimension membership sets
table data IN 'CSV' 'input/commodity__node.csv' : commodity_node <- [commodity,node];
table data IN 'CSV' 'input/entity__invest_method.csv' : entity__invest_method <- [entity,invest_method];
table data IN 'CSV' 'input/group__invest_method.csv' : group__invest_method <- [group,invest_method];
table data IN 'CSV' 'input/node__inflow_method.csv' : node__inflow_method_read <- [node,inflow_method];
table data IN 'CSV' 'input/node__storage_binding_method.csv' : node__storage_binding_method_read <- [node,storage_binding_method];
table data IN 'CSV' 'input/node__storage_start_end_method.csv' : node__storage_start_end_method <- [node,storage_start_end_method];
table data IN 'CSV' 'input/node__storage_solve_horizon_method.csv' : node__storage_solve_horizon_method <- [node,storage_solve_horizon_method];
table data IN 'CSV' 'input/node__profile__profile_method.csv' : node__profile__profile_method <- [node,profile,profile_method];
table data IN 'CSV' 'input/group__node.csv' : group_node <- [group,node];
table data IN 'CSV' 'input/group__process.csv' : group_process <- [group,process];
table data IN 'CSV' 'input/group__process__node.csv' : group_process_node <- [group,process,node];
table data IN 'CSV' 'input/p_process_node_constraint_flow_coefficient.csv' : process_node_flow_constraint <- [process, node, constraint];
table data IN 'CSV' 'input/p_process_constraint_capacity_coefficient.csv' : process_capacity_constraint <- [process, constraint];
table data IN 'CSV' 'input/p_node_constraint_capacity_coefficient.csv' : node_capacity_constraint <- [node, constraint];
table data IN 'CSV' 'input/p_node_constraint_state_coefficient.csv' : node_state_constraint <- [node, constraint];
table data IN 'CSV' 'input/constraint__sense.csv' : constraint__sense <- [constraint, sense];
table data IN 'CSV' 'input/p_process.csv' : process__param <- [process, processParam];
table data IN 'CSV' 'input/pd_node.csv' : node__param__period <- [node, nodeParam, period];
table data IN 'CSV' 'input/pt_node.csv' : node__param__time <- [node, nodeParam, time];
table data IN 'CSV' 'input/pd_process.csv' : process__param__period <- [process, processParam, period];
table data IN 'CSV' 'input/pt_process.csv' : process__param__time <- [process, processParam, time];
table data IN 'CSV' 'input/p_group.csv' : group__param <- [group, groupParam];
table data IN 'CSV' 'input/pd_group.csv' : group__param__period <- [group, groupParam, period];
table data IN 'CSV' 'input/process__ct_method.csv' : process__ct_method_read <- [process,ct_method];
table data IN 'CSV' 'input/process__node__ramp_method.csv' : process_node_ramp_method <- [process,node,ramp_method];
table data IN 'CSV' 'input/process__reserve__upDown__node.csv' : process_reserve_upDown_node <- [process,reserve,upDown,node];
table data IN 'CSV' 'input/process__sink.csv' : process_sink <- [process,sink];
table data IN 'CSV' 'input/process__source.csv' : process_source <- [process,source];
table data IN 'CSV' 'input/process__sink_nonSync_unit.csv' : process__sink_nonSync_unit <- [process,sink];
table data IN 'CSV' 'input/process__startup_method.csv' : process__startup_method_read <- [process,startup_method];
table data IN 'CSV' 'input/process__profile__profile_method.csv' : process__profile__profile_method <- [process,profile,profile_method];
table data IN 'CSV' 'input/process__node__profile__profile_method.csv' : process__node__profile__profile_method <- [process,node,profile,profile_method];
table data IN 'CSV' 'input/reserve__upDown__group__method.csv' : reserve__upDown__group__method <- [reserve,upDown,group,method];
table data IN 'CSV' 'input/pt_reserve__upDown__group.csv' : reserve__upDown__group__reserveParam__time <- [reserve, upDown, group, reserveParam, time];
table data IN 'CSV' 'input/timeblocks_in_use.csv' : solve_period_timeblockset <- [solve,period,timeblocks];
table data IN 'CSV' 'input/timeblocks__timeline.csv' : timeblockset__timeline <- [timeblocks,timeline];
table data IN 'CSV' 'solve_data/solve_current.csv' : solve_current <- [solve];
table data IN 'CSV' 'input/p_process_source.csv' : process__source__param <- [process, source, sourceSinkParam];
table data IN 'CSV' 'input/pt_process_source.csv' : process__source__param__time <- [process, source, sourceSinkTimeParam, time];
table data IN 'CSV' 'input/p_process_sink.csv' : process__sink__param <- [process, sink, sourceSinkParam];
table data IN 'CSV' 'input/pt_process_sink.csv' : process__sink__param__time <- [process, sink, sourceSinkTimeParam, time];
table data IN 'CSV' 'input/pd_commodity.csv' : commodity__param__period <- [commodity, commodityParam, period];
table data IN 'CSV' 'input/timeline.csv' : timeline__timestep__duration <- [timeline,timestep,duration];

# Parameters for model data
table data IN 'CSV' 'input/p_commodity.csv' : [commodity, commodityParam], p_commodity;
table data IN 'CSV' 'input/pd_commodity.csv' : [commodity, commodityParam, period], pd_commodity;
table data IN 'CSV' 'input/p_group__process.csv' : [group, process, groupParam], p_group__process;
table data IN 'CSV' 'input/p_group.csv' : [group, groupParam], p_group;
table data IN 'CSV' 'input/pd_group.csv' : [group, groupParam, period], pd_group;
table data IN 'CSV' 'input/p_node.csv' : [node, nodeParam], p_node;
table data IN 'CSV' 'input/pd_node.csv' : [node, nodeParam, period], pd_node;
table data IN 'CSV' 'input/pt_node.csv' : [node, nodeParam, time], pt_node;
table data IN 'CSV' 'input/p_process_node_constraint_flow_coefficient.csv' : [process, node, constraint], p_process_node_constraint_flow_coefficient;
table data IN 'CSV' 'input/p_process_constraint_capacity_coefficient.csv' : [process, constraint], p_process_constraint_capacity_coefficient;
table data IN 'CSV' 'input/p_node_constraint_capacity_coefficient.csv' : [node, constraint], p_node_constraint_capacity_coefficient;
table data IN 'CSV' 'input/p_node_constraint_state_coefficient.csv' : [node, constraint], p_node_constraint_state_coefficient;
table data IN 'CSV' 'input/p_process__reserve__upDown__node.csv' : [process, reserve, upDown, node, reserveParam], p_process_reserve_upDown_node;
table data IN 'CSV' 'input/p_process_sink.csv' : [process, sink, sourceSinkParam], p_process_sink;
table data IN 'CSV' 'input/pt_process_sink.csv' : [process, sink, sourceSinkTimeParam, time], pt_process_sink;
table data IN 'CSV' 'input/p_process_sink_coefficient.csv' : [process, sink], p_process_sink_coefficient;
table data IN 'CSV' 'input/p_process_source.csv' : [process, source, sourceSinkParam], p_process_source;
table data IN 'CSV' 'input/p_process_source_coefficient.csv' : [process, source], p_process_source_coefficient;
table data IN 'CSV' 'input/pt_process_source.csv' : [process, source, sourceSinkTimeParam, time], pt_process_source;
table data IN 'CSV' 'input/p_constraint_constant.csv' : [constraint], p_constraint_constant;
table data IN 'CSV' 'input/p_process.csv' : [process, processParam], p_process;
table data IN 'CSV' 'input/pd_process.csv' : [process, processParam, period], pd_process;
table data IN 'CSV' 'input/pt_process.csv' : [process, processParam, time], pt_process;
table data IN 'CSV' 'input/pt_profile.csv' : [profile, time], pt_profile;
table data IN 'CSV' 'input/p_reserve__upDown__group.csv' : [reserve, upDown, group, reserveParam], p_reserve_upDown_group;
table data IN 'CSV' 'input/pt_reserve__upDown__group.csv' : [reserve, upDown, group, reserveParam, time], pt_reserve_upDown_group;
table data IN 'CSV' 'input/timeline_duration_in_years.csv' : [timeline], p_timeline_duration_in_years;
table data IN 'CSV' 'solve_data/p_discount_years.csv' : [period], p_discount_years;
table data IN 'CSV' 'input/p_discount_rate.csv' : model <- [model];
table data IN 'CSV' 'input/p_discount_rate.csv' : [model], p_discount_rate;
# Parameters from the solve loop
table data IN 'CSV' 'solve_data/steps_in_use.csv' : dt <- [period, step];
table data IN 'CSV' 'solve_data/steps_in_use.csv' : [period, step], step_duration;
table data IN 'CSV' 'solve_data/steps_in_timeline.csv' : period_time <- [period,step];
table data IN 'CSV' 'solve_data/first_timesteps.csv' : period__time_first <- [period,step];
table data IN 'CSV' 'solve_data/last_timesteps.csv' : period__time_last <- [period,step];
table data IN 'CSV' 'solve_data/step_previous.csv' : dtttdt <- [period, time, previous, previous_within_block, previous_period, previous_within_solve];
table data IN 'CSV' 'solve_data/realized_periods_of_current_solve.csv' : period_realized <- [period];
table data IN 'CSV' 'solve_data/invest_periods_of_current_solve.csv' : period_invest <- [period];
table data IN 'CSV' 'input/p_model.csv' : [modelParam], p_model;

# After rolling forward the investment model
table data IN 'CSV' 'solve_data/p_entity_invested.csv' : [entity], p_entity_invested;


set process__fork_method_yes dimen 2 within {process, fork_method} := 
    {p in process, m in fork_method 
	  : (sum{(p, source) in process_source} 1 > 1 || sum{(p, sink) in process_sink} 1 > 1) && m in fork_method_yes};
set process__fork_method_no dimen 2 within {process, fork_method} := 
    {p in process, m in fork_method 
	  : (sum{(p, source) in process_source} 1 < 2 && sum{(p, sink) in process_sink} 1 < 2) && m in fork_method_no};
set process__fork_method := process__fork_method_yes union process__fork_method_no;
set process_ct_startup_fork_method := 
    { p in process, m1 in ct_method, m2 in startup_method, m3 in fork_method, m in method
	    : (m1, m2, m3, m) in methods
	    && (p, m1) in process__ct_method
	    && (p, m2) in process__startup_method
		&& (p, m3) in process__fork_method
	};
set process_method := setof {(p, m1, m2, m3, m) in process_ct_startup_fork_method} (p, m);
set process__profileProcess__toSink__profile__profile_method :=
    { p in process, (p2, sink, f, fm) in process__node__profile__profile_method
	    :  p = p2
		&& (p, sink) in process_sink
	    && (sum{(p, m) in process_method : m in method_indirect} 1
		    || sum{(p, source) in process_source} 1 < 1)
	};
set process__profileProcess__toSink := setof {(p, p2, sink, f, m) in process__profileProcess__toSink__profile__profile_method} (p, p2, sink);
set process__source__toProfileProcess__profile__profile_method :=
    { (p, source) in process_source, (p2, source, f, fm) in process__node__profile__profile_method
	    :  p = p2
	    && (sum{(p, m) in process_method : m in method_indirect} 1
		    || sum{(p, sink) in process_sink} 1 < 1)
	};
set process__source__toProfileProcess := setof {(p, source, p2, f, m) in process__source__toProfileProcess__profile__profile_method} (p, source, p2);
set process_profile := setof {(p, source, p2) in process__source__toProfileProcess} (p) union setof {(p, p2, sink) in process__profileProcess__toSink} (p);
set process_source_toProcess := 
    { (p, source) in process_source, p2 in process 
	    :  p = p2 
	    && (p2, source) in process_source 
	    && sum{(p, m) in process_method : m in method_indirect} 1
	};
set process_process_toSink := 
    { p in process, (p2, sink) in process_sink
	    :  p = p2 
	    && (p, sink) in process_sink 
	    && sum{(p, m) in process_method : m in method_indirect} 1
	};
set process_sink_toProcess := 
    { sink in node, p in process, p2 in process 
	    :  p = p2 
	    && (p, sink) in process_sink 
	    && (p2, sink) in process_sink 
	    && sum{(p, m) in process_method : m in method_2way_nvar} 1
	};
set process_process_toSource := 
    { p in process, (p2, source) in process_source
	    :  p = p2 
	    && (p, source) in process_source
	    && sum{(p, m) in process_method : m in method_2way_nvar} 1
	};
set process_source_toSink := 
    { (p, source) in process_source, sink in node
	    :  (p, sink) in process_sink
        && sum{(p, m) in process_method : m in method_direct} 1
	};
set process_source_toProcess_direct :=
    { (p, source) in process_source, p2 in process
	    :  p = p2
        && sum{(p, m) in process_method : m in method_direct} 1
	};
set process_process_toSink_direct :=
    { p in process, p2 in process, sink in node
	    :  p = p2
		&& (p, sink) in process_sink
        && sum{(p, m) in process_method : m in method_direct} 1
	};
set process_sink_toProcess_direct := 
	{ (p, sink) in process_sink, p2 in process
	    :  p = p2
		&& (p, sink) in process_sink
	    && sum{(p, m) in process_method : m in method_2way_2var} 1
	};
set process_process_toSource_direct := 
	{ p in process, p2 in process, source in node
	    :  p = p2
		&& (p, source) in process_source
	    && sum{(p, m) in process_method : m in method_2way_2var} 1
	};
set process_sink_toSource := 
	{ (p, sink) in process_sink, source in node
	    :  (p, source) in process_source
	    && (p, sink) in process_sink
	    && sum{(p, m) in process_method : m in method_2way_2var} 1
	};
set process__source__sink__profile__profile_method_direct :=
    { (p, source, sink) in process_source_toSink, f in profile, fm in profile_method
	    :  sum{(p, m) in process_method : m in method_direct} 1
		&& ( (p, source, f, fm) in process__node__profile__profile_method
		     || (p, sink, f, fm) in process__node__profile__profile_method
		   )
	};

set process_source_sink := 
    process_source_toSink union    # Direct 1-variable
	process_sink_toSource union    # Direct 1-variable, but the other way
	process_source_toProcess union # First step for indirect (from source to process)
	process_process_toSink union   # Second step for indirect (from process to sink)
	process_sink_toProcess union   # Add the 'wrong' direction in 2-way processes with multiple inputs/outputs
	process_process_toSource union # Add the 'wrong' direction in 2-way processes with multiple inputs/outputs
	process__profileProcess__toSink union   # Add profile based inputs to process
	process__source__toProfileProcess;	   # Add profile based inputs to process	

set process_source_sink_alwaysProcess :=
    process_source_toProcess_direct union  # Direct 1-variable, but showing the process in between
	process_process_toSink_direct union
	process_sink_toProcess_direct union
	process_process_toSource_direct union
	process_source_toProcess union # First step for indirect (from source to process)
	process_process_toSink union   # Second step for indirect (from process to sink)
	process_sink_toProcess union   # Add the 'wrong' direction in 2-way processes with multiple inputs/outputs
	process_process_toSource union # Add the 'wrong' direction in 2-way processes with multiple inputs/outputs
	process__profileProcess__toSink union   # Add profile based inputs to process
	process__source__toProfileProcess;	   # Add profile based inputs to process	

set process_source_sink_noEff :=
	process_source_toProcess union # First step for indirect (from source to process)
	process_process_toSink union   # Second step for indirect (from process to sink)
	process_sink_toProcess union   # Add the 'wrong' direction in 2-way processes with multiple inputs/outputs
	process_process_toSource union # Add the 'wrong' direction in 2-way processes with multiple inputs/outputs
	process__profileProcess__toSink union   # Add profile based inputs to process
	process__source__toProfileProcess;	   # Add profile based inputs to process	

set process_source_sink_eff :=
    process_source_toSink union    # Direct 1-variable
	process_sink_toSource;         # Direct 1-variable, but the other way

set process__source__sink__profile__profile_method_connection :=
    { (p, sink, source) in process_source_sink, f in profile, m in profile_method
	    : (p, f, m) in process__profile__profile_method
	};
set process__source__sink__profile__profile_method :=
    process__profileProcess__toSink__profile__profile_method union
	process__source__toProfileProcess__profile__profile_method union
	process__source__sink__profile__profile_method_connection union
	process__source__sink__profile__profile_method_direct
;

set process__source__sink_isNodeSink := {(p, source, sink) in process_source_sink : (p, sink) in process_sink};

set process_online 'processes with an online status' := setof {(p, m) in process_method : m in method_LP} p;

set peedt := {(p, source, sink) in process_source_sink, (d, t) in dt};


param pdCommodity {c in commodity, param in commodityPeriodParam, d in period} := 
        + if (c, param, d) in commodity__param__period
		  then pd_commodity[c, param, d]
		  else p_commodity[c, param];

param pdGroup {g in group, param in groupPeriodParam, d in period} :=
        + if (g, param, d) in group__param__period
		  then pd_group[g, param, d]
		  else if (g, param) in group__param 
		  then p_group[g, param]
		  else 0;

set reserve__upDown__group__method_timeseries := {(r, ud, ng, r_m) in reserve__upDown__group__method 
                                                       : r_m = 'timeseries_only'
													   || r_m = 'timeseries_and_dynamic' 
													   || r_m = 'timeseries_and_large_failure' 
													   || r_m = 'all'};
set reserve__upDown__group__method_dynamic := {(r, ud, ng, r_m) in reserve__upDown__group__method 
                                                       : r_m = 'dynamic_only'
													   || r_m = 'timeseries_and_dynamic' 
													   || r_m = 'dynamic_and_large_failure' 
													   || r_m = 'all'};
set reserve__upDown__group__method_n_1 := {(r, ud, ng, r_m) in reserve__upDown__group__method
                                                       : r_m = 'large_failure_only' 
													   || r_m = 'timeseries_and_large_failure' 
													   || r_m = 'dynamic_and_large_failure' 
													   || r_m = 'all'};

set process__method_indirect := {(p, m) in process_method : m in method_indirect};

set process__source__sink_isNodeSink_no2way := {(p, source, sink) in process_source_sink : (p, sink) in process_sink 
	                                       && sum{(p,m) in process_method : m not in method_2way_1var} 1};
set process__source__sink_isNodeSink_yes2way := {(p, source, sink) in process_source_sink : (p, sink) in process_sink 
	                                       && sum{(p,m) in process_method : m in method_2way_1var} 1};
set process_isNodeSink_yes2way := setof {(p, source, sink) in process__source__sink_isNodeSink_yes2way} p;
set process__source__sink_isNodeSink_2way_2var := {(p, source, sink) in process_source_sink : (p, sink) in process_sink 
	                                       && sum{(p,m) in process_method : m in method_2way_2var } 1};

set gdt_maxInstantFlow := {g in group, (d, t) in dt : pdGroup[g, 'max_instant_flow', d]};
set gdt_minInstantFlow := {g in group, (d, t) in dt : pdGroup[g, 'min_instant_flow', d]};
		  
param pdNode {n in node, param in nodePeriodParam, d in period} :=
        + if (n, param, d) in node__param__period
		  then pd_node[n, param, d]
		  else p_node[n, param];
param ptNode {n in node, param in nodeTimeParam, t in time} :=
        + if (n, param, t) in node__param__time
		  then pt_node[n, param, t]
		  else p_node[n, param];
set nodeSelfDischarge :=  {n in nodeState : sum{(d, t) in dt : ptNode[n, 'self_discharge_loss', t]} 1};
		  

set process__source__timeParam := 
    { (p, source) in process_source, param in sourceSinkTimeParam
	    :  (p, source, param) in process__source__param
	    || (p, source, param) in process__source__param_t
	};

set process__sink__timeParam :=
    { (p, sink) in process_sink, param in sourceSinkTimeParam
	    :  (p, sink, param) in process__sink__param
	    || (p, sink, param) in process__sink__param_t
	};

set process__timeParam :=
    { p in process, param in sourceSinkTimeParam
	   :  ((p, param) in process__param && p in process_connection)
	   || ((p, param) in process__param_t && p in process_connection)
	}; 

set process__source__sink__param :=
    { (p, source, sink) in process_source_sink, param in sourceSinkParam
	    :  (p, source, param) in process__source__param
	    || (p, sink, param) in process__sink__param
	    || ((p, param) in process__param && p in process_connection)
	};
set process__source__sink__param_t :=
    { (p, source, sink) in process_source_sink, param in sourceSinkTimeParam
	    :  (p, source, param) in process__source__param
	    || (p, source, param) in process__source__param_t
	    || (p, sink, param) in process__sink__param
	    || (p, sink, param) in process__sink__param_t
	    || ((p, param) in process__param && p in process_connection)
	    || ((p, param) in process__param_t && p in process_connection)
	};


param setup1 := gmtime() - datetime0;
display setup1;

set process_source_sink_param_t := {(p, source, sink) in process_source_sink_eff, param in processTimeParam : (p, param) in process__param_t};

set process__source__sink__ramp_method :=
    { (p, source, sink) in process_source_sink, m in ramp_method
	    :  (p, source, m) in process_node_ramp_method
		|| (p, sink, m) in process_node_ramp_method
	};

param pdProcess {p in process, param in processPeriodParam, d in period} :=
        + if (p, param, d) in process__param__period
		  then pd_process[p, param, d]
		  else if (p, param) in process__param
		  then p_process[p, param]
		  else 0;
param ptProcess {p in process, param in processTimeParam, t in time} :=
        + if (p, param, t) in process__param__time
		  then pt_process[p, param, t]
		  else if (p, param) in process__param
		  then p_process[p, param]
		  else 0;

param p_entity_unitsize {e in entity} := 
        + if e in process 
		  then ( if p_process[e, 'virtual_unitsize']
                 then p_process[e, 'virtual_unitsize'] 
		         else if e in process && p_process[e, 'existing']
			          then p_process[e, 'existing']
					  else 1
			   )			 
          else if e in node 
		  then ( if p_node[e, 'virtual_unitsize'] 
                 then p_node[e, 'virtual_unitsize'] 
		         else if e in node && p_node[e, 'existing']
		              then p_node[e, 'existing']
					  else 1
			   );

param pProcess_source_sink {(p, source, sink, param) in process__source__sink__param} :=
		+ if (p, source, param) in process__source__param
		  then p_process_source[p, source, param]
		  else if (p, sink, param) in process__sink__param
		  then p_process_sink[p, sink, param]
		  else 0;

param ptProcess_source {(p, source) in process_source, param in sourceSinkTimeParam, t in time} :=  # : sum{d in period : (d, t) in dt} 1
        + if (p, source, param, t) in process__source__param__time
		  then pt_process_source[p, source, param, t]
		  else if (p, source, param) in process__source__param
		  then p_process_source[p, source, param]
		  else 0;
        
param ptProcess_sink {(p, sink) in process_sink, param in sourceSinkTimeParam, t in time} :=  #  : sum{d in period : (d, t) in dt} 1
        + if (p, sink, param, t) in process__sink__param__time
		  then pt_process_sink[p, sink, param, t]
		  else if (p, sink, param) in process__sink__param
		  then p_process_sink[p, sink, param]
		  else 0;

param ptProcess_source_sink {(p, source, sink, param) in process__source__sink__param_t, t in time} := #  : sum{d in period : (d, t) in dt} 1
        + if (p, sink, param, t) in process__sink__param__time
		  then pt_process_sink[p, sink, param, t]
          else if (p, source, param, t) in process__source__param__time
		  then pt_process_source[p, source, param, t]
		  else if (p, param, t) in connection__param__time
		  then pt_process[p, param, t]
		  else if (p, source, param) in process__source__param
		  then p_process_source[p, source, param]
		  else if (p, sink, param) in process__sink__param
		  then p_process_sink[p, sink, param]
		  else if (p, param) in connection__param
		  then p_process[p, param]
		  else 0;


param ptReserve_upDown_group {(r, ud, g) in reserve__upDown__group, param in reserveTimeParam, t in time} :=
        + if (r, ud, g, param, t) in reserve__upDown__group__reserveParam__time
		  then pt_reserve_upDown_group[r, ud, g, param, t]
		  else p_reserve_upDown_group[r, ud, g, param];
set process_reserve_upDown_node_active := {(p, r, ud, n) in process_reserve_upDown_node : sum{(r, ud, g) in reserve__upDown__group} 1};
set prundt := {(p, r, ud, n) in process_reserve_upDown_node_active, (d, t) in dt};
set pdt_online := {p in process_online, (d, t) in dt : pdProcess[p, 'startup_cost', d]};

param hours_in_period{d in period} := sum {(d, t) in dt} (step_duration[d, t]);
param hours_in_solve := sum {(d, t) in dt} (step_duration[d, t]);
param period_share_of_year{d in period} := hours_in_period[d] / 8760;
param solve_share_of_year := hours_in_solve / 8760;

param period_share_of_annual_flow {n in node, d in period : (n, 'scale_to_annual_flow') in node__inflow_method && pdNode[n, 'annual_flow', d]} := 
        abs(sum{(d, t) in dt} (ptNode[n, 'inflow', t])) / pdNode[n, 'annual_flow', d];
param period_flow_annual_multiplier {n in node, d in period : (n, 'scale_to_annual_flow') in node__inflow_method && pdNode[n, 'annual_flow', d]} := 
        period_share_of_year[d] / period_share_of_annual_flow[n, d];
param period_flow_proportional_multiplier {n in node, d in period : (n, 'scale_in_proportion') in node__inflow_method && pdNode[n, 'annual_flow', d]} :=
        pdNode[n, 'annual_flow', d] / (abs(sum{t in time} (ptNode[n, 'inflow', t])) / sum{(d, tl) in period__timeline} p_timeline_duration_in_years[tl]);
param pdtNodeInflow {n in node, (d, t) in dt : (n, 'no_inflow') not in node__inflow_method}  := 
        + ptNode[n, 'inflow', t] *
        ( if (n, 'scale_to_annual_flow') in node__inflow_method && pdNode[n, 'annual_flow', d] then
		    + period_flow_annual_multiplier[n, d]
		  else if (n, 'scale_in_proportion') in node__inflow_method && pdNode[n, 'annual_flow', d] then
		    + period_flow_proportional_multiplier[n, d]
		  else 1
		);

set period__period_next := {d in period, dNext in period : 1 + sum{d2 in period : d2 <=d} 1 = sum{dNext2 in period : dNext2 <=dNext} 1};
param p_disc_rate := (if sum{m in model} 1 then max{m in model} p_discount_rate[m] else 0.05);
param p_disc_offset_investment := (if sum{m in model} 1 then max{m in model} p_discount_offset_investment[m] else 0);
param p_disc_offset_operations := (if sum{m in model} 1 then max{m in model} p_discount_offset_operations[m] else 0.5);
param p_discount_factor_investment{d in period} := 1/(1 + p_disc_rate) ^ (p_discount_years[d] + p_disc_offset_investment);
param p_discount_factor_operations{d in period} := 1/(1 + p_disc_rate) ^ (p_discount_years[d] + p_disc_offset_operations);
param p_discount_in_perpetuity_investment{d in period} := (if p_disc_rate then (1/(1+p_disc_rate)^(p_discount_years[d]+p_disc_offset_investment))/p_disc_rate else 1);
param p_discount_in_perpetuity_operations{d in period} := (if p_disc_rate then (1/(1+p_disc_rate)^(p_discount_years[d]+p_disc_offset_operations))/p_disc_rate else 1);
param p_discount_with_perpetuity_investment{d in period} := (if d not in period_last 
                                                             then sum{(d, dNext) in period__period_next} 
															    (+ p_discount_in_perpetuity_investment[d] 
																 - p_discount_in_perpetuity_investment[dNext]) 
															 else p_discount_in_perpetuity_investment[d]);
param p_discount_with_perpetuity_operations{d in period} := 
  ( if d not in period_last 
    then sum{(d, dNext) in period__period_next} 
	          ( + p_discount_in_perpetuity_operations[d] 
			    - p_discount_in_perpetuity_operations[dNext]) 
    else p_discount_in_perpetuity_operations[d]);
param ed_entity_annual{e in entityInvest, d in period_invest} :=
        + sum{m in invest_method : (e, m) in entity__invest_method && e in node && m not in invest_method_not_allowed}
          ( + (pdNode[e, 'invest_cost', d] * 1000 * ( pdNode[e, 'interest_rate', d] 
			  / (1 - (1 / (1 + pdNode[e, 'interest_rate', d])^pdNode[e, 'lifetime', d] ) ) ))
			+ pdNode[e, 'fixed_cost', d] * 1000
		  )
        + sum{m in invest_method : (e, m) in entity__invest_method && e in process && m not in invest_method_not_allowed}
		  (
            + (pdProcess[e, 'invest_cost', d] * 1000 * ( pdProcess[e, 'interest_rate', d] 
			  / (1 - (1 / (1 + pdProcess[e, 'interest_rate', d])^pdProcess[e, 'lifetime', d] ) ) ))
			+ pdProcess[e, 'fixed_cost', d] * 1000
		  )
; 			
param ed_entity_annual_divest{e in entityDivest, d in period_invest} :=
        + sum{m in invest_method : (e, m) in entity__invest_method && e in node && m not in divest_method_not_allowed}
          ( + (pdNode[e, 'salvage_value', d] * 1000 * ( pdNode[e, 'interest_rate', d] 
			  / (1 - (1 / (1 + pdNode[e, 'interest_rate', d])^pdNode[e, 'lifetime', d] ) ) ))
			+ pdNode[e, 'fixed_cost', d] * 1000
		  )
        + sum{m in invest_method : (e, m) in entity__invest_method && e in process && m not in divest_method_not_allowed}
		  (
            + (pdProcess[e, 'salvage_value', d] * 1000 * ( pdProcess[e, 'interest_rate', d] 
			  / (1 - (1 / (1 + pdProcess[e, 'interest_rate', d])^pdProcess[e, 'lifetime', d] ) ) ))
			+ pdProcess[e, 'fixed_cost', d] * 1000
		  )
; 			


set process_minload := {p in process : (p, 'min_load_efficiency') in process__ct_method};
param ptProcess_section{p in process_minload, t in time} := 
        + 1 / ptProcess[p, 'efficiency', t] 
    	- ( 1 / ptProcess[p, 'efficiency', t] - ptProcess[p, 'min_load', t] / ptProcess[p, 'efficiency_at_min_load', t] ) 
			    / (1 - ptProcess[p, 'min_load', t])
		; 
param ptProcess_slope{p in process, t in time} := 
        1 / ptProcess[p, 'efficiency', t] 
		- (if p in process_minload then ptProcess_section[p, t] else 0);

#         	          * (if (p, 'min_load_efficiency') in process__ct_method then ptProcess_slope[p, t] else 1 / ptProcess[p, 'efficiency', t])
#		              * (if p in process_unit then p_process_sink_coefficient[p, sink] / p_process_source_coefficient[p, n] else 1)

param w_calc_slope := gmtime() - datetime0 - setup1;
display w_calc_slope;

param ptProcess__source__sink__t_varCost {(p, source, sink) in process_source_sink, t in time : sum{d in period : (d, t) in dt} 1} :=
  + (if (p, source) in process_source then ptProcess_source[p, source, 'variable_cost', t])
#      * (if (p, source, sink) in process_source_sink_eff
#	        then (if (p, 'min_load_efficiency') in process__ct_method then ptProcess_slope[p, t] else 1 / ptProcess[p, 'efficiency', t])
#			else 1
#		)	
  + (if (p, sink) in process_sink then ptProcess_sink[p, sink, 'variable_cost', t])
  + (if (p, source, sink) in process_source_sink then ptProcess[p, 'variable_cost', t])
;

param ptProcess__source__sink__t_varCost_alwaysProcess {(p, source, sink) in process_source_sink_alwaysProcess, t in time : sum{d in period : (d, t) in dt} 1} :=
  + (if (p, source) in process_source then ptProcess_source[p, source, 'variable_cost', t])
  + (if (p, sink) in process_sink then ptProcess_sink[p, sink, 'variable_cost', t])
  + (if (p, source, sink) in process_source_sink_alwaysProcess 
        && ((p, sink) in process_sink || (p, sink) in process_source)
	 then ptProcess[p, 'variable_cost', t])
;

set pssdt_varCost_noEff := {(p, source, sink) in process_source_sink_noEff, (d, t) in dt : ptProcess__source__sink__t_varCost[p, source, sink, t]};
set pssdt_varCost_eff := {(p, source, sink) in process_source_sink_eff, (d, t) in dt : (p, source) in process_source && ptProcess_source[p, source, 'variable_cost', t]};

set ed_invest := {e in entityInvest, d in period_invest : ed_entity_annual[e, d] || sum{(e, c) in process_capacity_constraint} 1 || sum{(e, c) in node_capacity_constraint} 1 };
set ed_invest_period := {(e, d) in ed_invest : (e, 'invest_period') in entity__invest_method || (e, 'invest_period_total') in entity__invest_method 
                                               || (e, 'invest_retire_period') in entity__invest_method || (e, 'invest_retire_period_total') in entity__invest_method};
set e_invest_total := {e in entityInvest : (e, 'invest_total') in entity__invest_method || (e, 'invest_period_total') in entity__invest_method 
                                               || (e, 'invest_retire_total') in entity__invest_method || (e, 'invest_retire_period_total') in entity__invest_method};
set pd_invest := {(p, d) in ed_invest : p in process};
set nd_invest := {(n, d) in ed_invest : n in node};
set ed_divest := {e in entityDivest, d in period_invest : ed_entity_annual_divest[e, d] || sum{(e, c) in process_capacity_constraint} 1 || sum{(e, c) in node_capacity_constraint} 1 };
set ed_divest_period := {(e, d) in ed_invest : (e, 'retire_period') in entity__invest_method || (e, 'retire_period_total') in entity__invest_method 
                                               || (e, 'invest_retire_period') in entity__invest_method || (e, 'invest_retire_period_total') in entity__invest_method};
set e_divest_total := {e in entityDivest : (e, 'retire_total') in entity__invest_method || (e, 'retire_period_total') in entity__invest_method 
                                               || (e, 'invest_retire_total') in entity__invest_method || (e, 'invest_retire_period_total') in entity__invest_method};
set pd_divest := {(p, d) in ed_divest : p in process};
set nd_divest := {(n, d) in ed_divest : n in node};

set gd_invest := {g in group_invest, d in period_invest : sum{(g, e) in group_entity : (e, d) in ed_invest} 1};
set gd_invest_period := {(g, d) in gd_invest : (g, 'invest_period') in group__invest_method || (g, 'invest_period_total') in group__invest_method 
                                               || (g, 'invest_retire_period') in group__invest_method || (g, 'invest_retire_period_total') in group__invest_method};
set g_invest_total := {g in group_invest : (g, 'invest_total') in group__invest_method || (g, 'invest_period_total') in group__invest_method 
                                               || (g, 'invest_retire_total') in group__invest_method || (g, 'invest_retire_period_total') in group__invest_method};
set gd_divest := {g in group_invest, d in period_invest : sum{(g, e) in group_entity : (e, d) in ed_invest} 1};
set gd_divest_period := {(g, d) in gd_invest : (g, 'retire_period') in group__invest_method || (g, 'retire_period_total') in group__invest_method 
                                               || (g, 'invest_retire_period') in group__invest_method || (g, 'invest_retire_period_total') in group__invest_method};
set g_divest_total := {g in group_divest : (g, 'retire_total') in group__invest_method || (g, 'retire_period_total') in group__invest_method 
                                               || (g, 'invest_retire_total') in group__invest_method || (g, 'invest_retire_period_total') in group__invest_method};

param e_invest_max_total{e in entityInvest} :=
  + (if e in process then p_process[e, 'invest_max_total'])
  + (if e in node then p_node[e, 'invest_max_total'])
;  

param e_divest_max_total{e in entityDivest} :=
  + (if e in process then p_process[e, 'retire_max_total'])
  + (if e in node then p_node[e, 'retire_max_total'])
;  

param e_invest_min_total{e in entityInvest} :=
  + (if e in process then p_process[e, 'invest_min_total'])
  + (if e in node then p_node[e, 'invest_min_total'])
;  

param e_divest_min_total{e in entityDivest} :=
  + (if e in process then p_process[e, 'retire_min_total'])
  + (if e in node then p_node[e, 'retire_min_total'])
;  

param ed_invest_max_period{(e, d) in ed_invest} :=
  + (if e in process then pdProcess[e, 'invest_max_period', d])
  + (if e in node then pdNode[e, 'invest_max_period', d])
;  

param ed_divest_max_period{(e, d) in ed_divest} :=
  + (if e in process then pdProcess[e, 'retire_max_period', d])
  + (if e in node then pdNode[e, 'retire_max_period', d])
;  

param ed_invest_min_period{(e, d) in ed_invest} :=
  + (if e in process then pdProcess[e, 'invest_min_period', d])
  + (if e in node then pdNode[e, 'invest_min_period', d])
;  

param ed_divest_min_period{(e, d) in ed_divest} :=
  + (if e in process then pdProcess[e, 'retire_min_period', d])
  + (if e in node then pdNode[e, 'retire_min_period', d])
;  

set process_source_sink_ramp_limit_up :=
    {(p, source, sink) in process_source_sink
	    : ( sum{(p, source, m) in process_node_ramp_method : m in ramp_limit_method} 1
		    && p_process_source[p, source, 'ramp_speed_up'] > 0
		  ) || 
		  ( sum{(p, sink, m) in process_node_ramp_method : m in ramp_limit_method} 1
		    && p_process_sink[p, sink, 'ramp_speed_up'] > 0
		  )
	};
set process_source_sink_ramp_limit_down :=
    {(p, source, sink) in process_source_sink
	    : ( sum{(p, source, m) in process_node_ramp_method : m in ramp_limit_method} 1
		    && p_process_source[p, source, 'ramp_speed_down'] > 0
		  ) ||
		  ( sum{(p, sink, m) in process_node_ramp_method : m in ramp_limit_method} 1
		    && p_process_sink[p, sink, 'ramp_speed_down'] > 0
		  )
	};
set process_source_sink_ramp_cost :=
    {(p, source, sink) in process_source_sink
	    : sum{(p, source, m) in process_node_ramp_method : m in ramp_cost_method} 1
		  || sum{(p, sink, m) in process_node_ramp_method : m in ramp_cost_method} 1
	};
set process_source_sink_ramp :=
    process_source_sink_ramp_limit_up 
	union process_source_sink_ramp_limit_down 
	union process_source_sink_ramp_cost;

set process_source_sink_dt_ramp_up :=
        {(p, source, sink) in process_source_sink_ramp_limit_up, (d, t) in dt :
 		    p_process[p, 'ramp_speed_up'] * 60 < step_duration[d, t]
        };
set process_source_sink_dt_ramp_down :=
        {(p, source, sink) in process_source_sink_ramp_limit_down, (d, t) in dt :
 		    p_process[p, 'ramp_speed_down'] * 60 < step_duration[d, t]
		};
set process_source_sink_dt_ramp :=
        {(p, source, sink) in process_source_sink_ramp, (d, t) in dt :
		    (p, source, sink) in process_source_sink_ramp_cost
		    || (p, source, sink, d, t) in process_source_sink_dt_ramp_down
            || (p, source, sink, d, t) in process_source_sink_dt_ramp_up
        };

set process_source_sink_dtttdt_ramp_up := {(p, source, sink) in process_source_sink_ramp_limit_up, (d, t, t_previous, t_previous_within_block, d_previous, t_previous_within_solve) in dtttdt
		: (p, source, sink, d, t) in process_source_sink_dt_ramp};

set process_source_sink_dtttdt_ramp_down := {(p, source, sink) in process_source_sink_ramp_limit_down, (d, t, t_previous, t_previous_within_block, d_previous, t_previous_within_solve) in dtttdt
		: (p, source, sink, d, t) in process_source_sink_dt_ramp};

set process_reserve_upDown_node_increase_reserve_ratio :=
        {(p, r, ud, n) in process_reserve_upDown_node_active :
		    p_process_reserve_upDown_node[p, r, ud, n, 'increase_reserve_ratio'] > 0
		};
set process_reserve_upDown_node_large_failure_ratio :=
        {(p, r, ud, n) in process_reserve_upDown_node_active :
		    p_process_reserve_upDown_node[p, r, ud, n, 'large_failure_ratio'] > 0
		};
set process_large_failure := setof {(p, r, ud, n) in process_reserve_upDown_node_large_failure_ratio} p;

set group_commodity_node_period_co2 :=
        {g in group, (c, n) in commodity_node, d in period : 
		    (g, n) in group_node 
			&& p_commodity[c, 'co2_content'] 
			&& pdGroup[g, 'co2_price', d]
		};

set gcndt_co2 := {(g, c, n, d) in group_commodity_node_period_co2, t in time : (d, t) in dt};

set process__commodity__node := {p in process, (c, n) in commodity_node : (p, n) in process_source || (p, n) in process_sink};

set commodity_node_co2 :=
        {(c, n) in commodity_node : 
			p_commodity[c, 'co2_content'] 
		};

set process__commodity__node_co2 := {p in process, (c, n) in commodity_node_co2 : (p, n) in process_source || (p, n) in process_sink};
set process_co2 := setof{(p, c, n) in process__commodity__node_co2} p;

set process__sink_nonSync :=
        {p in process, sink in node :
		       ( (p, sink) in process_sink && (p, sink) in process__sink_nonSync_unit )
			|| ( (p, sink) in process_sink && p in process_nonSync_connection )
			|| ( (p, sink) in process_source && p in process_nonSync_connection )  
	    };
param p_entity_all_existing {e in entity} :=
        + (if e in process then p_process[e, 'existing'])
        + (if e in node then p_node[e, 'existing'])
		+ (if not p_model['solveFirst'] && e in entityInvest then p_entity_invested[e])
;


set process_VRE := {p in process_unit : not (sum{(p, source) in process_source} 1)
                                        && (sum{(p, n, prof, m) in process__node__profile__profile_method : m = 'upper_limit'} 1)};

param d_obj default 0;
param d_flow {(p, source, sink, d, t) in peedt} default 0;
param d_flow_1_or_2_variable {(p, source, sink, d, t) in peedt} default 0;
param d_flowInvest {(p, d) in pd_invest} default 0;
param d_reserve_upDown_node {(p, r, ud, n, d, t) in prundt} default 0;
param dq_reserve {(r, ud, ng) in reserve__upDown__group, (d, t) in dt} default 0;

#########################
# Variable declarations
var v_flow {(p, source, sink, d, t) in peedt};
var v_ramp {(p, source, sink, d, t) in peedt};
var v_reserve {(p, r, ud, n, d, t) in prundt : sum{(r, ud, g) in reserve__upDown__group} 1 } >= 0;
var v_state {n in nodeState, (d, t) in dt} >= 0;
var v_online_linear {p in process_online,(d, t) in dt} >=0;
var v_startup_linear {p in process_online, (d, t) in dt} >=0;
var v_shutdown_linear {p in process_online, (d, t) in dt} >=0;
var v_invest {(e, d) in ed_invest} >= 0;
var v_divest {(e, d) in ed_divest} >= 0;
var vq_state_up {n in nodeBalance, (d, t) in dt} >= 0;
var vq_state_down {n in nodeBalance, (d, t) in dt} >= 0;
var vq_reserve {(r, ud, ng) in reserve__upDown__group, (d, t) in dt} >= 0;
var vq_inertia {g in groupInertia, (d, t) in dt} >= 0;
var vq_non_synchronous {g in groupNonSync, (d, t) in dt} >= 0;
var vq_capacity_margin {g in groupCapacityMargin, d in period_invest} >= 0;

#########################
## Data checks 
printf 'Checking: Eff. data for 1 variable conversions directly from source to sink (and possibly back)\n';
check {(p, m) in process_method, t in time : m in method_1var} ptProcess[p, 'efficiency', t] != 0 ;

printf 'Checking: Efficiency data for 1-way conversions with an online variable\n';
check {(p, m) in process_method, t in time : m in method_1way_on} ptProcess[p, 'efficiency', t] != 0;

printf 'Checking: Efficiency data for 2-way linear conversions without online variables\n';
check {(p, m) in process_method, t in time : m in method_2way_off} ptProcess[p, 'efficiency', t] != 0;

printf 'Checking: Invalid combinations between conversion/transfer methods and the startup method\n';
check {(p, ct_m, s_m, f_m, m) in process_ct_startup_fork_method} : not (p, ct_m, s_m, f_m, 'not_applicable') in process_ct_startup_fork_method;

printf 'Checking: Is there a timeline connected to a timeblockset\n';
check sum{(tb, tl) in timeblockset__timeline} 1 > 0;

printf 'Checking: Are discount factors set in models with investments and multiple periods\n';
check {d in period : d not in period_first && (sum{(e, d) in ed_invest} 1 || sum{(e, d) in ed_divest} 1)} : p_discount_years[d] != 0;

printf 'Checking: Does a node with has_storage also have has_balance set to yes\n';
check {n in nodeState} : n in nodeBalance;

param setup2 := gmtime() - datetime0 - setup1 - w_calc_slope;
display setup2;
minimize total_cost:
      + sum {(c, n) in commodity_node, (d, t) in dt} 
	      pdCommodity[c, 'price', d]
	      * (
		      # Buying a commodity (increases the objective function)
	          + sum {(p, n, sink) in process_source_sink_noEff } 
			    ( + v_flow[p, n, sink, d, t] )
	          + sum {(p, n, sink) in process_source_sink_eff } (
			      + v_flow[p, n, sink, d, t]
         	          * (if (p, 'min_load_efficiency') in process__ct_method then ptProcess_slope[p, t] else 1 / ptProcess[p, 'efficiency', t])
		              * (if p in process_unit then p_process_sink_coefficient[p, sink] / p_process_source_coefficient[p, n] else 1)
                  + (if (p, 'min_load_efficiency') in process__ct_method then 
	                  + v_online_linear[p, d, t] 
			              * ptProcess_section[p, t]
				          * p_entity_unitsize[p]
					)	  
				)		  
			  # Selling to a commodity node (decreases objective function if price is positive)
	          - sum {(p, source, n) in process_source_sink } (
			      + v_flow[p, source, n, d, t]
				)  
		    ) * step_duration[d, t] * p_discount_with_perpetuity_operations[d] / period_share_of_year[d]
	  + sum {(g, c, n, d, t) in gcndt_co2} 
	      p_commodity[c, 'co2_content'] * pdGroup[g, 'co2_price', d] 
	      * (
		      # Paying for CO2 (increases the objective function)
			  + sum {(p, n, sink) in process_source_sink_noEff } 
			    ( + v_flow[p, n, sink, d, t] )
	          + sum {(p, n, sink) in process_source_sink_eff } (
			      + v_flow[p, n, sink, d, t]
         	          * (if (p, 'min_load_efficiency') in process__ct_method then ptProcess_slope[p, t] else 1 / ptProcess[p, 'efficiency', t])
		              * (if p in process_unit then p_process_sink_coefficient[p, sink] / p_process_source_coefficient[p, n] else 1)
                  + (if (p, 'min_load_efficiency') in process__ct_method then 
	                  + v_online_linear[p, d, t] 
			              * ptProcess_section[p, t]
				          * p_entity_unitsize[p]
					)	  
				)		  
			  # Receiving credits for removing CO2 (decreases the objective function)
	          - sum {(p, source, n) in process_source_sink } (
			      + v_flow[p, source, n, d, t]
				)  
			) * step_duration[d, t] * p_discount_with_perpetuity_operations[d] / period_share_of_year[d]
 	 + sum {(p, d, t) in pdt_online} (v_startup_linear[p, d, t] * pdProcess[p, 'startup_cost', d] * p_entity_unitsize[p]) * step_duration[d, t] * p_discount_with_perpetuity_operations[d] / period_share_of_year[d]
     + sum {(p, source, sink, d, t) in pssdt_varCost_noEff}
       ( + ptProcess__source__sink__t_varCost[p, source, sink, t]
	       * v_flow[p, source, sink, d, t]
       ) * step_duration[d, t] * p_discount_with_perpetuity_operations[d] / period_share_of_year[d]
     + sum {(p, source, sink, d, t) in pssdt_varCost_eff}
	   ( + ptProcess_source[p, source, 'variable_cost', t]
	       * v_flow[p, source, sink, d, t] 
           	       * (if (p, 'min_load_efficiency') in process__ct_method then ptProcess_slope[p, t] else 1 / ptProcess[p, 'efficiency', t])
                   * (if p in process_unit then p_process_sink_coefficient[p, sink] / p_process_source_coefficient[p, source] else 1)
               + (if (p, 'min_load_efficiency') in process__ct_method then 
	               + v_online_linear[p, d, t] 
   			          * ptProcess_section[p, t]
			          * p_entity_unitsize[p]
    			 )	  
	   ) * step_duration[d, t] * p_discount_with_perpetuity_operations[d] / period_share_of_year[d]
     + sum {(p, source, sink, d, t) in pssdt_varCost_eff}
	   ( + ptProcess_sink[p, sink, 'variable_cost', t]
	       * v_flow[p, source, sink, d, t] 
	   ) * step_duration[d, t] * p_discount_with_perpetuity_operations[d] / period_share_of_year[d]
     + sum {(p, source, sink, d, t) in pssdt_varCost_eff}
	   ( + ptProcess[p, 'variable_cost', t]
	       * v_flow[p, source, sink, d, t] 
       ) * step_duration[d, t] * p_discount_with_perpetuity_operations[d] / period_share_of_year[d]
#      + sum {(p, source, sink, m) in process__source__sink__ramp_method, (d, t) in dt : m in ramp_cost_method}
#        ( + v_ramp[p, source, sink, d, t] * pProcess_source_sink[p, source, sink, 'ramp_cost'] ) * step_duration[d, t] * p_discount_with_perpetuity_operations[d] / period_share_of_year[d]
      + sum {g in groupInertia, (d, t) in dt} vq_inertia[g, d, t] * pdGroup[g, 'penalty_inertia', d] * step_duration[d, t] * p_discount_with_perpetuity_operations[d] / period_share_of_year[d]
      + sum {g in groupNonSync, (d, t) in dt} vq_non_synchronous[g, d, t] * pdGroup[g, 'penalty_non_synchronous', d] * step_duration[d, t] * p_discount_with_perpetuity_operations[d] / period_share_of_year[d]
      + sum {n in nodeBalance, (d, t) in dt} vq_state_up[n, d, t] * ptNode[n, 'penalty_up', t] * step_duration[d, t] * p_discount_with_perpetuity_operations[d] / period_share_of_year[d]
      + sum {n in nodeBalance, (d, t) in dt} vq_state_down[n, d, t] * ptNode[n, 'penalty_down', t] * step_duration[d, t] * p_discount_with_perpetuity_operations[d] / period_share_of_year[d]
      + sum {(r, ud, ng) in reserve__upDown__group, (d, t) in dt} vq_reserve[r, ud, ng, d, t] * p_reserve_upDown_group[r, ud, ng, 'penalty_reserve'] * step_duration[d, t] * p_discount_with_perpetuity_operations[d] / period_share_of_year[d]
  - sum {n in nodeState, (d, t) in period__time_last : (n, 'use_reference_price') in node__storage_solve_horizon_method && d in period_last}
      + pdNode[n, 'storage_state_reference_price', d]
        * v_state[n, d, t]
		* step_duration[d, t] * p_discount_with_perpetuity_operations[d] / period_share_of_year[d]
  + sum {e in entity, d in period}  # This is constant term and will be dropped by the solver. Here for completeness.
    + p_entity_all_existing[e]
      * ( + (if e in node then pdNode[e, 'fixed_cost', d] * 1000)
	      + (if e in process then pdProcess[e, 'fixed_cost', d] * 1000)
		)
	  * p_discount_with_perpetuity_investment[d]
  + sum {(e, d) in ed_invest} 
    + v_invest[e, d]
      * p_entity_unitsize[e]
      * ed_entity_annual[e, d]
	  * p_discount_with_perpetuity_investment[d]
  - sum {(e, d) in ed_divest} 
    + v_divest[e, d]
      * p_entity_unitsize[e]
      * ed_entity_annual_divest[e, d]
	  * p_discount_with_perpetuity_investment[d]
  + sum {g in groupCapacityMargin, d in period_invest}
    + vq_capacity_margin[g, d]
	  * pdGroup[g, 'penalty_capacity_margin', d]
	  * p_discount_with_perpetuity_investment[d]
;
param w_total_cost := gmtime() - datetime0 - setup1 - w_calc_slope - setup2;
display w_total_cost;

# Energy balance in each node  
s.t. nodeBalance_eq {n in nodeBalance, (d, t, t_previous, t_previous_within_block, d_previous, t_previous_within_solve) in dtttdt} :
  + (if n in nodeState && (n, 'bind_forward_only') in node__storage_binding_method && not ((d, t) in period__time_first && d in period_first) then (v_state[n, d, t] -  v_state[n, d_previous, t_previous_within_solve]))
  + (if n in nodeState && (n, 'bind_within_solve') in node__storage_binding_method then (v_state[n, d, t] -  v_state[n, d_previous, t_previous_within_solve]))
  + (if n in nodeState && (n, 'bind_within_period') in node__storage_binding_method then (v_state[n, d, t] -  v_state[n, d, t_previous]))
  + (if n in nodeState && (n, 'bind_within_timeblock') in node__storage_binding_method then (v_state[n, d, t] -  v_state[n, d, t_previous_within_block]))
  =
  # n is sink
  + sum {(p, source, n) in process_source_sink} (
      + v_flow[p, source, n, d, t]
	)  
  # n is source
  - sum {(p, n, sink) in process_source_sink_eff } ( 
      + v_flow[p, n, sink, d, t] 
	      * (if (p, 'min_load_efficiency') in process__ct_method then ptProcess_slope[p, t] else 1 / ptProcess[p, 'efficiency', t])
		  * (if p in process_unit then p_process_sink_coefficient[p, sink] / p_process_source_coefficient[p, n] else 1)
      + (if (p, 'min_load_efficiency') in process__ct_method then 
	        + v_online_linear[p, d, t]
			    * ptProcess_section[p, t]
				* p_entity_unitsize[p]
		)
    )		
  - sum {(p, n, sink) in process_source_sink_noEff} 
    ( + v_flow[p, n, sink, d, t] 
    )
  + (if (n, 'no_inflow') not in node__inflow_method then pdtNodeInflow[n, d, t])
  - (if n in nodeSelfDischarge then 
      + v_state[n, d, t] 
	      * ptNode[n, 'self_discharge_loss', t] 
		  * step_duration[d, t])
  + vq_state_up[n, d, t]
  - vq_state_down[n, d, t]
;
param balance := gmtime() - datetime0 - setup1 - w_calc_slope - setup2 - w_total_cost;
display balance;

s.t. reserveBalance_timeseries_eq {(r, ud, ng, r_m) in reserve__upDown__group__method_timeseries, (d, t) in dt} :
  + sum {(p, r, ud, n) in process_reserve_upDown_node_active 
	      : ( sum{(p, m) in process_method : m not in method_1var_per_way} 1   ## not 1var_per_way and source; not 1var_per_way and sink; 1var_per_way and sink 
		        || (p, n) in process_sink
			)
		    && (ng, n) in group_node 
		    && (r, ud, ng) in reserve__upDown__group} 
	    ( v_reserve[p, r, ud, n, d, t] 
	      * p_process_reserve_upDown_node[p, r, ud, n, 'reliability']
	    )
  + sum {(p, r, ud, n) in process_reserve_upDown_node_active                   ## 1var_per_way and source  
		  : ( sum{(p, m) in process_method : m in method_1var_per_way} 1 
		        && (p, n) in process_source
			)
		    && (ng, n) in group_node 
		    && (r, ud, ng) in reserve__upDown__group} 
	    ( v_reserve[p, r, ud, n, d, t] 
	      * p_process_reserve_upDown_node[p, r, ud, n, 'reliability']
	      * (if (p, 'min_load_efficiency') in process__ct_method then ptProcess_slope[p, t] else 1 / ptProcess[p, 'efficiency', t])
		)
  + vq_reserve[r, ud, ng, d, t]
  >=
  + ptReserve_upDown_group[r, ud, ng, 'reservation', t]
;

s.t. reserveBalance_dynamic_eq{(r, ud, ng, r_m) in reserve__upDown__group__method_dynamic, (d, t) in dt} :
  + sum {(p, r, ud, n) in process_reserve_upDown_node_active 
	      : ( sum{(p, m) in process_method : m not in method_1var_per_way} 1   ## not 1var_per_way and source; not 1var_per_way and sink; 1var_per_way and sink
		        || (p, n) in process_sink
			)
		    && (ng, n) in group_node 
		    && (r, ud, ng) in reserve__upDown__group} 
	    ( v_reserve[p, r, ud, n, d, t] 
	      * p_process_reserve_upDown_node[p, r, ud, n, 'reliability']
	    )
  + sum {(p, r, ud, n) in process_reserve_upDown_node_active               
		  : ( sum{(p, m) in process_method : m in method_1var_per_way} 1       ## 1var_per_way and source
		        && (p, n) in process_source
			)
		    && (ng, n) in group_node 
		    && (r, ud, ng) in reserve__upDown__group} 
	    ( v_reserve[p, r, ud, n, d, t] 
	      * p_process_reserve_upDown_node[p, r, ud, n, 'reliability']
	      * (if (p, 'min_load_efficiency') in process__ct_method then ptProcess_slope[p, t] else 1 / ptProcess[p, 'efficiency', t])
		)
  + vq_reserve[r, ud, ng, d, t]
  >=
  + sum {(p, r, ud, n) in process_reserve_upDown_node_increase_reserve_ratio : (ng, n) in group_node 
          && (r, ud, ng) in reserve__upDown__group}
	   ( + sum{(p, source, n) in process_source_sink} v_flow[p, source, n, d, t] * p_process_reserve_upDown_node[p, r, ud, n, 'increase_reserve_ratio']
	     + sum{(p, n, sink) in process_source_sink_noEff} v_flow[p, n, sink, d, t] * p_process_reserve_upDown_node[p, r, ud, n, 'increase_reserve_ratio']
	     + sum{(p, n, sink) in process_source_sink_eff} v_flow[p, n, sink, d, t] * p_process_reserve_upDown_node[p, r, ud, n, 'increase_reserve_ratio']
	                                                    * (if (p, 'min_load_efficiency') in process__ct_method then ptProcess_slope[p, t] else 1 / ptProcess[p, 'efficiency', t])
	   )
  + sum {(n, ng) in group_node : p_reserve_upDown_group[r, ud, ng, 'increase_reserve_ratio']}
	   (pdtNodeInflow[n, d, t] * p_reserve_upDown_group[r, ud, ng, 'increase_reserve_ratio'])
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
	    ( v_reserve[p, r, 'up', n, d, t] 
	      * p_process_reserve_upDown_node[p, r, 'up', n, 'reliability']
	    )
  + sum {(p, r, 'up', n) in process_reserve_upDown_node_active                 
		  : p <> p_n_1
 		    && ( sum{(p, m) in process_method : m in method_1var_per_way} 1      ## 1var_per_way and source
		         && (p, n) in process_source
			   )
		    && (ng, n) in group_node 
		    && (r, 'up', ng) in reserve__upDown__group
		} 
	    ( v_reserve[p, r, 'up', n, d, t] 
	      * p_process_reserve_upDown_node[p, r, 'up', n, 'reliability']
	      * (if (p, 'min_load_efficiency') in process__ct_method then ptProcess_slope[p, t] else 1 / ptProcess[p, 'efficiency', t])
		)
  + vq_reserve[r, 'up', ng, d, t]
  >=
  + sum{(p_n_1, source, n) in process_source_sink : (ng, n) in group_node} 
      + v_flow[p_n_1, source, n, d, t] 
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
	    ( v_reserve[p, r, 'down', n, d, t] 
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
	    ( v_reserve[p, r, 'down', n, d, t] 
	      * p_process_reserve_upDown_node[p, r, 'down', n, 'reliability']
	      * (if (p, 'min_load_efficiency') in process__ct_method then ptProcess_slope[p, t] else 1 / ptProcess[p, 'efficiency', t])
		)
  + vq_reserve[r, 'down', ng, d, t]
  >=
  + sum{(p_n_1, n, sink) in process_source_sink_noEff : (ng, n) in group_node} 
      + v_flow[p_n_1, n, sink, d, t] 
	    * p_process_reserve_upDown_node[p_n_1, r, 'down', n, 'large_failure_ratio']
  + sum{(p_n_1, n, sink) in process_source_sink_eff : (ng, n) in group_node } 
      + v_flow[p_n_1, n, sink, d, t] 
	    * p_process_reserve_upDown_node[p_n_1, r, 'down', n, 'large_failure_ratio']
	    * (if (p_n_1, 'min_load_efficiency') in process__ct_method then ptProcess_slope[p_n_1, t] else 1 / ptProcess[p_n_1, 'efficiency', t])
;
param reserves := gmtime() - datetime0 - setup1 - w_calc_slope - setup2 - w_total_cost - balance;
display reserves;

# Indirect efficiency conversion - there is more than one variable. Direct conversion does not have an equation - it's directly in the nodeBalance_eq.
s.t. conversion_indirect {(p, m) in process__method_indirect, (d, t) in dt} :
  + sum {source in entity : (p, source) in process_source} 
    ( + v_flow[p, source, p, d, t] 
  	      * p_process_source_coefficient[p, source]
	)
  =
  + sum {sink in entity : (p, sink) in process_sink} 
    ( + v_flow[p, p, sink, d, t] 
	      * p_process_sink_coefficient[p, sink]
	)
	  * (if (p, 'min_load_efficiency') in process__ct_method then ptProcess_slope[p, t] else 1 / ptProcess[p, 'efficiency', t])
  + (if (p, 'min_load_efficiency') in process__ct_method then v_online_linear[p, d, t] * ptProcess_section[p, t] * p_entity_unitsize[p])
;
param indirect := gmtime() - datetime0 - setup1 - w_calc_slope - setup2 - w_total_cost - balance - reserves;
display indirect;

s.t. profile_flow_upper_limit {(p, source, sink, f, 'upper_limit') in process__source__sink__profile__profile_method, (d, t) in dt} :
  + ( + v_flow[p, source, sink, d, t]
      + sum{(p, r, 'up', sink) in process_reserve_upDown_node} v_reserve[p, r, 'up', sink, d, t]
	)
  	* 
	  ( if (p, source) in process_source then p_process_source_coefficient[p, source]
	    else if (p, sink) in process_sink then p_process_sink_coefficient[p, sink]
		else 1
      )
  <=
  + pt_profile[f, t]
    * ( + p_entity_all_existing[p]
        + sum {(p, d_invest) in pd_invest : d_invest <= d} v_invest[p, d_invest] * p_entity_unitsize[p]
        - sum {(p, d_invest) in pd_divest : d_invest <= d} v_divest[p, d_invest] * p_entity_unitsize[p]
	  )
;

s.t. profile_flow_lower_limit {(p, source, sink, f, 'lower_limit') in process__source__sink__profile__profile_method, (d, t) in dt} :
  + ( + v_flow[p, source, sink, d, t] 
      - sum{(p, r, 'down', sink) in process_reserve_upDown_node} v_reserve[p, r, 'down', sink, d, t]
    )
    * ( if (p, source) in process_source then p_process_source_coefficient[p, source]
        else if (p, sink) in process_sink then p_process_sink_coefficient[p, sink]
		else 1
	  )
  >=
  + pt_profile[f, t]
    * ( + p_entity_all_existing[p]
        + sum {(p, d_invest) in pd_invest : d_invest <= d} v_invest[p, d_invest] * p_entity_unitsize[p]
        - sum {(p, d_invest) in pd_divest : d_invest <= d} v_divest[p, d_invest] * p_entity_unitsize[p]
	  )
;

s.t. profile_flow_fixed {(p, source, sink, f, 'fixed') in process__source__sink__profile__profile_method, (d, t) in dt} :
  + ( + v_flow[p, source, sink, d, t] 
  	      * ( if (p, source) in process_source then p_process_source_coefficient[p, source]
			  else if (p, sink) in process_sink then p_process_sink_coefficient[p, sink]
			  else 1
			)
	)
  =
  + pt_profile[f, t]
    * ( + p_entity_all_existing[p]
        + sum {(p, d_invest) in pd_invest : d_invest <= d} v_invest[p, d_invest] * p_entity_unitsize[p]
        - sum {(p, d_invest) in pd_divest : d_invest <= d} v_divest[p, d_invest] * p_entity_unitsize[p]
	  )
;

s.t. profile_state_upper_limit {(n, f, 'upper_limit') in node__profile__profile_method, (d, t) in dt} :
  + v_state[n, d, t] 
  <=
  + pt_profile[f, t]
    * ( + p_entity_all_existing[n]
        + sum {(n, d_invest) in pd_invest : d_invest <= d} v_invest[n, d_invest] * p_entity_unitsize[n]
        - sum {(n, d_invest) in pd_divest : d_invest <= d} v_divest[n, d_invest] * p_entity_unitsize[n]
	  )
;

s.t. profile_state_lower_limit {(n, f, 'lower_limit') in node__profile__profile_method, (d, t) in dt} :
  + v_state[n, d, t] 
  >=
  + pt_profile[f, t]
    * ( + p_entity_all_existing[n]
        + sum {(n, d_invest) in pd_invest : d_invest <= d} v_invest[n, d_invest] * p_entity_unitsize[n]
        - sum {(n, d_invest) in pd_divest : d_invest <= d} v_divest[n, d_invest] * p_entity_unitsize[n]
	  )
;

s.t. profile_state_fixed {(n, f, 'fixed') in node__profile__profile_method, (d, t) in dt} :
  + v_state[n, d, t] 
  =
  + pt_profile[f, t]
    * ( + p_entity_all_existing[n]
        + sum {(n, d_invest) in pd_invest : d_invest <= d} v_invest[n, d_invest] * p_entity_unitsize[n]
        - sum {(n, d_invest) in pd_divest : d_invest <= d} v_divest[n, d_invest] * p_entity_unitsize[n]
	  )
;

s.t. storage_state_start {n in nodeState, (d, t) in period__time_first
     : p_model['solveFirst'] 
	 && d in period_first 
	 && ((n, 'fix_start') in node__storage_start_end_method || (n, 'fix_start_end') in node__storage_start_end_method)} :
  + v_state[n, d, t]
  =
  + p_node[n,'storage_state_start']
    * ( + p_entity_all_existing[n]
        + sum {(n, d_invest) in pd_invest : d_invest <= d} v_invest[n, d_invest] * p_entity_unitsize[n]
        - sum {(n, d_invest) in pd_divest : d_invest <= d} v_divest[n, d_invest] * p_entity_unitsize[n]
	  )
;

s.t. storage_state_end {n in nodeState, (d, t) in period__time_last 
     : p_model['solveLast'] 
	 && d in period_last 
	 && ((n, 'fix_end') in node__storage_start_end_method || (n, 'fix_start_end') in node__storage_start_end_method)} :
  + v_state[n, d, t]
  =
  + p_node[n,'storage_state_end']
    * ( + p_entity_all_existing[n]
        + sum {(n, d_invest) in pd_invest : d_invest <= d} v_invest[n, d_invest] * p_entity_unitsize[n]
        - sum {(n, d_invest) in pd_divest : d_invest <= d} v_divest[n, d_invest] * p_entity_unitsize[n]
	  )
;

s.t. storage_state_solve_horizon_reference_value {n in nodeState, (d, t) in period__time_last
     : d in period_last
	 && (n, 'use_reference_value') in node__storage_solve_horizon_method} :
  + v_state[n, d, t]
  =
  + p_node[n,'storage_state_reference_value']
    * ( + p_entity_all_existing[n]
        + sum {(n, d_invest) in pd_invest : d_invest <= d} v_invest[n, d_invest] * p_entity_unitsize[n]
        - sum {(n, d_invest) in pd_divest : d_invest <= d} v_divest[n, d_invest] * p_entity_unitsize[n]
	  )
;

s.t. constraint_greater_than {(c, 'greater_than') in constraint__sense, (d, t) in dt} :
  + sum {(p, source, sink) in process_source_sink : (p, source, c) in process_node_flow_constraint}
    ( + v_flow[p, source, sink, d, t]
	      * p_process_node_constraint_flow_coefficient[p, source, c]
	)
  + sum {(p, source, sink) in process_source_sink : (p, sink, c) in process_node_flow_constraint}
    ( + v_flow[p, source, sink, d, t]
	      * p_process_node_constraint_flow_coefficient[p, sink, c]
	)
  + sum {(n, c) in node_state_constraint}
    ( + v_state[n, d, t]
	      * p_node_constraint_state_coefficient[n, c]
	)
  + sum {(n, c) in node_capacity_constraint : d in period_invest}
    ( ( + (if (n, d) in ed_invest then v_invest[n, d])
	    - (if (n, d) in ed_divest then v_divest[n, d])
	  )
	  * p_node_constraint_capacity_coefficient[n, c]
	)
  + sum {(p, c) in process_capacity_constraint : d in period_invest}
    ( ( + (if (p, d) in ed_invest then v_invest[p, d])
	    - (if (p, d) in ed_divest then v_divest[p, d])
	  )
	  * p_process_constraint_capacity_coefficient[p, c]
	)
  >=
  + p_constraint_constant[c]
;
	
s.t. process_constraint_less_than {(c, 'lesser_than') in constraint__sense, (d, t) in dt} :
  + sum {(p, source, sink) in process_source_sink : (p, source, c) in process_node_flow_constraint}
    ( + v_flow[source, source, p, d, t]
	      * p_process_node_constraint_flow_coefficient[p, source, c]
	)
  + sum {(p, source, sink) in process_source_sink : (p, sink, c) in process_node_flow_constraint}
    ( + v_flow[p, source, sink, d, t]
	      * p_process_node_constraint_flow_coefficient[p, sink, c]
	)
  + sum {(n, c) in node_state_constraint}
    ( + v_state[n, d, t]
	      * p_node_constraint_state_coefficient[n, c]
	)
  + sum {(n, c) in node_capacity_constraint : d in period_invest}
    ( ( + (if (n, d) in ed_invest then v_invest[n, d])
	    - (if (n, d) in ed_divest then v_divest[n, d])
	  )
	  * p_node_constraint_capacity_coefficient[n, c]
	)
  + sum {(p, c) in process_capacity_constraint : d in period_invest}
    ( ( + (if (p, d) in ed_invest then v_invest[p, d])
	    - (if (p, d) in ed_divest then v_divest[p, d])
	  )
	  * p_process_constraint_capacity_coefficient[p, c]
	)
  <=
  + p_constraint_constant[c]
;

s.t. process_constraint_equal {(c, 'equal') in constraint__sense, (d, t) in dt} :
  + sum {(p, source, sink) in process_source_sink : (p, source, c) in process_node_flow_constraint}
    ( + v_flow[p, source, sink, d, t]
	      * p_process_node_constraint_flow_coefficient[p, source, c]
	)
  + sum {(p, source, sink) in process_source_sink : (p, sink, c) in process_node_flow_constraint}
    ( + v_flow[p, source, sink, d, t]
	      * p_process_node_constraint_flow_coefficient[p, sink, c]
	)
  + sum {(n, c) in node_state_constraint}
    ( + v_state[n, d, t]
	      * p_node_constraint_state_coefficient[n, c]
	)
  + sum {(n, c) in node_capacity_constraint : d in period_invest}
    ( ( + (if (n, d) in ed_invest then v_invest[n, d])
	    - (if (n, d) in ed_divest then v_divest[n, d])
	  )
	  * p_node_constraint_capacity_coefficient[n, c]
	)
  + sum {(p, c) in process_capacity_constraint : d in period_invest}
    ( ( + (if (p, d) in ed_invest then v_invest[p, d])
	    - (if (p, d) in ed_divest then v_divest[p, d])
	  )
	  * p_process_constraint_capacity_coefficient[p, c]
	)
  =
  + p_constraint_constant[c]
;

s.t. maxState {n in nodeState, (d, t) in dt} :
  + v_state[n, d, t]
  <=
  + p_entity_all_existing[n]
  + sum {(p, d_invest) in pd_invest : d_invest <= d} v_invest[p, d_invest] * p_entity_unitsize[n]
  - sum {(p, d_invest) in pd_divest : d_invest <= d} v_divest[p, d_invest] * p_entity_unitsize[n]
;

s.t. maxToSink {(p, source, sink) in process__source__sink_isNodeSink, (d, t) in dt} :
  + v_flow[p, source, sink, d, t]
  + sum {r in reserve : (p, r, 'up', sink) in process_reserve_upDown_node_active} v_reserve[p, r, 'up', sink, d, t]
  <=
  + ( if p not in process_online then
      + p_process_sink_coefficient[p, sink]
        * ( + p_entity_all_existing[p]
            + sum {(p, d_invest) in pd_invest : d_invest <= d} v_invest[p, d_invest] * p_entity_unitsize[p]
            - sum {(p, d_invest) in pd_divest : d_invest <= d} v_divest[p, d_invest] * p_entity_unitsize[p]
	      )	
	)
  + ( if p in process_online then
      + p_process_sink_coefficient[p, sink]
        * v_online_linear[p, d, t]
		* p_entity_unitsize[p]
    ) 
;

s.t. minToSink {(p, source, sink) in process__source__sink_isNodeSink_no2way, (d, t) in dt} :
  + v_flow[p, source, sink, d, t]
  >=
  + (if p in process_online then v_online_linear[p, d, t] * p_process[p, 'min_load'] * p_entity_unitsize[p] else 0)
;

# Special equation to limit the 1variable connection on the negative transfer
s.t. minToSink_1var {(p, source, sink) in process__source__sink_isNodeSink_yes2way, (d, t) in dt} :
  + v_flow[p, source, sink, d, t]
  >=
  - ( if p not in process_online then
      + p_process_sink_coefficient[p, sink] 
        * ( + p_entity_all_existing[p]
            + sum {(p, d_invest) in pd_invest : d_invest <= d} v_invest[p, d_invest] * p_entity_unitsize[p]
            - sum {(p, d_invest) in pd_divest : d_invest <= d} v_divest[p, d_invest] * p_entity_unitsize[p]
	      )	
	)
  - ( if p in process_online then
      + p_process_sink_coefficient[p, sink]
        * v_online_linear[p, d, t] 
		* p_entity_unitsize[p]
    )  
;

# Special equations for the method with 2 variables presenting a direct 2way connection between source and sink (without the process)
s.t. maxToSource {(p, sink, source) in process_sink_toSource, (d, t) in dt} :
  + v_flow[p, sink, source, d, t]
  + sum {r in reserve : (p, r, 'up', source) in process_reserve_upDown_node_active} v_reserve[p, r, 'up', source, d, t]
  <=
  + ( if p not in process_online then
      + p_process_source_coefficient[p, source] 
        * ( + p_entity_all_existing[p]
            + sum {(p, d_invest) in pd_invest : d_invest <= d} v_invest[p, d_invest] * p_entity_unitsize[p]
            - sum {(p, d_invest) in pd_divest : d_invest <= d} v_divest[p, d_invest] * p_entity_unitsize[p]
	      )	
	)
  + ( if p in process_online then
      + p_process_sink_coefficient[p, sink]
        * v_online_linear[p, d, t] 
		* p_entity_unitsize[p]
    )  
;

s.t. minToSource {(p, source, sink) in process__source__sink_isNodeSink_2way_2var, (d, t) in dt} :
  + v_flow[p, sink, source, d, t]
  >=
  + (if p in process_online then v_online_linear[p, d, t] * p_process[p, 'min_load'] * p_entity_unitsize[p] else 0)
;

s.t. maxOnline {p in process_online, (d, t) in dt} :
  + v_online_linear[p, d, t]
  <=
  + p_entity_all_existing[p] / p_entity_unitsize[p]
  + sum {(p, d_invest) in pd_invest : d_invest <= d} v_invest[p, d_invest]
  - sum {(p, d_invest) in pd_divest : d_invest <= d} v_divest[p, d_invest]
;

s.t. online__startup_linear {p in process_online, (d, t, t_previous, t_previous_within_block, d_previous, t_previous_within_solve) in dtttdt} :
  + v_startup_linear[p, d, t]
  >=
  + v_online_linear[p, d, t] 
  - v_online_linear[p, d_previous, t_previous_within_solve]
;

s.t. maxStartup {p in process_online, (d, t) in dt} :
  + v_startup_linear[p, d, t]
  <=
  + p_entity_all_existing[p] / p_entity_unitsize[p]
  + sum {(p, d_invest) in pd_invest : d_invest <= d} v_invest[p, d_invest]
;

s.t. online__shutdown_linear {p in process_online, (d, t, t_previous, t_previous_within_block, d_previous, t_previous_within_solve) in dtttdt} :
  + v_shutdown_linear[p, d, t]
  >=
  - v_online_linear[p, d, t] 
  + v_online_linear[p, d_previous, t_previous_within_solve]
;

s.t. maxShutdown {p in process_online, (d, t) in dt} :
  + v_shutdown_linear[p, d, t]
  <=
  + p_entity_all_existing[p] / p_entity_unitsize[p]
  + sum {(p, d_invest) in pd_invest : d_invest <= d} v_invest[p, d_invest]
;

#s.t. minimum_downtime {p in process_online, t : p_process[u,'min_downtime'] >= step_duration[t]} :
#  + v_online_linear[p, d, t]
#  <=
#  + p_entity_all_existing[p] / p_entity_unitsize[p]
#  + sum {(p, d_invest) in pd_invest : d_invest <= d} [p, d_invest]
#   - sum {(p, d_invest) in pd_divest : d_invest <= d} v_divest[p, d_invest]
#  - sum{(d, t_) in dt : t_ > t && t_ <= t + p_process[u,'min_downtime'] / time_period_duration} (
#      + v_startup_linear[g, n, u, t_]
#	)
#;

# Minimum operational time
#s.t. minimum_uptime {(g, n, u, t) in gnut : u in unit_online && p_unittype[u,'min_uptime_h'] >= time_period_duration / 60 && t >= p_unittype[u,'min_uptime_h'] * 60 #/ time_period_duration} :
#  + v_online[g, n, u, t]
#  >=
#  + sum{t_ in time_in_use : t_ > t - 1 - p_unittype[u,'min_uptime_h'] * 60 / time_period_duration && t_ < t} (
#      + v_startup_linear[g, n, u, t_]
#	)
#;

s.t. ramp_up_variable {(p, source, sink) in process_source_sink_ramp, (d, t, t_previous, t_previous_within_block, d_previous, t_previous_within_solve) in dtttdt} :
  + v_ramp[p, source, sink, d, t]
  >=
  + v_flow[p, source, sink, d, t]
  - v_flow[p, source, sink, d, t_previous]
;

s.t. ramp_up_constraint {(p, source, sink, d, t, t_previous, t_previous_within_block, d_previous, t_previous_within_solve) in process_source_sink_dtttdt_ramp_up} :
  + v_ramp[p, source, sink, d, t]
  + sum {r in reserve : (p, r, 'up', sink) in process_reserve_upDown_node_active} 
         (v_reserve[p, r, 'up', sink, d, t] / step_duration[d, t])
  <=
  + p_process[p, 'ramp_speed_up']
    * 60
	* step_duration[d, t]
	* ( + if (p, sink) in process_sink then p_process_sink_coefficient[p, sink]
        + if (p, source) in process_source then p_process_source_coefficient[p, source]
      )
    * ( + p_entity_all_existing[p]
	    + ( if (p, d) in ed_invest then v_invest[p, d] * p_entity_unitsize[p] )
		- ( if (p, d) in ed_divest then v_divest[p, d] * p_entity_unitsize[p] )
	  )
  + ( if p in process_online then v_startup_linear[p, d, t] * p_entity_unitsize[p] )  # To make sure that units can startup despite ramp limits.
;

s.t. ramp_down_constraint {(p, source, sink, d, t, t_previous, t_previous_within_block, d_previous, t_previous_within_solve) in process_source_sink_dtttdt_ramp_down} :
  + v_ramp[p, source, sink, d, t]
  + sum {r in reserve : (p, r, 'down', sink) in process_reserve_upDown_node_active} 
         (v_reserve[p, r, 'down', sink, d, t] / step_duration[d, t])
  >=
  - p_process[p, 'ramp_speed_down']
    * 60
	* step_duration[d, t]
	* ( + if (p, sink) in process_sink then p_process_sink_coefficient[p, sink]
        + if (p, source) in process_source then p_process_source_coefficient[p, source]
      )
    * ( + p_entity_all_existing[p]
	    + ( if (p, d) in ed_invest then v_invest[p, d] * p_entity_unitsize[p] )
		- ( if (p, d) in ed_divest then v_divest[p, d] * p_entity_unitsize[p] )
	  )
  - ( if p in process_online then v_shutdown_linear[p, d, t] * p_entity_unitsize[p] )  # To make sure that units can shutdown despite ramp limits.
;

s.t. reserve_process_upward{(p, r, 'up', n, d, t) in prundt} :
  + v_reserve[p, r, 'up', n, d, t]
  <=
  ( if p in process_online then
      + v_online_linear[p, d, t] 
	    * p_process_reserve_upDown_node[p, r, 'up', n, 'max_share']
		* p_entity_unitsize[p]
    else
      + p_process_reserve_upDown_node[p, r, 'up', n, 'max_share'] 
        * (
            + p_entity_all_existing[p]
            + sum {(p, d_invest) in pd_invest : d_invest <= d} v_invest[p, d_invest] * p_entity_unitsize[p]
            - sum {(p, d_invest) in pd_divest : d_invest <= d} v_divest[p, d_invest] * p_entity_unitsize[p]
          )
    	* ( if (sum{(p, prof, 'upper_limit') in process__profile__profile_method} 1) then
	          ( + sum{(p, prof, 'upper_limit') in process__profile__profile_method} pt_profile[prof, t] )
	        else 1
	      )
  )
;

s.t. reserve_process_downward{(p, r, 'down', n, d, t) in prundt} :
  + v_reserve[p, r, 'down', n, d, t]
  <=
  + p_process_reserve_upDown_node[p, r, 'down', n, 'max_share']
    * ( + sum{(p, source, n) in process_source_sink} v_flow[p, source, n, d, t]
        - ( + p_entity_all_existing[p]
            + sum {(p, d_invest) in pd_invest : d_invest <= d} v_invest[p, d_invest] * p_entity_unitsize[p]
            - sum {(p, d_invest) in pd_divest : d_invest <= d} v_divest[p, d_invest] * p_entity_unitsize[p]
          ) * ( if (sum{(p, prof, 'lower_limit') in process__profile__profile_method} 1) then
	              ( + sum{(p, prof, 'lower_limit') in process__profile__profile_method} pt_profile[prof, t] )
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
  + sum{(g, e) in group_entity : (e, d) in ed_divest} v_invest[e, d] * p_entity_unitsize[e]
  >=
  + pdGroup[g, 'retire_min_period', d]
;

s.t. maxInvestGroup_entity_total {g in g_invest_total} :
  + sum{(g, e) in group_entity, d in period : e in entityInvest} v_invest[e, d] * p_entity_unitsize[e]
  + sum{(g, e) in group_entity : e in entityInvest} (if not p_model['solveFirst'] then p_entity_invested[e])
  <=
  + p_group[g, 'invest_max_total']
;

s.t. maxDivestGroup_entity_total {g in g_divest_total} :
  + sum{(g, e) in group_entity, d in period : e in entityDivest} v_divest[e, d] * p_entity_unitsize[e]
  + sum{(g, e) in group_entity : e in entityDivest} (if not p_model['solveFirst'] then p_entity_divested[e])
  <=
  + p_group[g, 'retire_max_total']
;

s.t. minInvestGroup_entity_total {g in g_invest_total} :
  + sum{(g, e) in group_entity, d in period : e in entityInvest} v_invest[e, d] * p_entity_unitsize[e]
  + sum{(g, e) in group_entity : e in entityInvest} (if not p_model['solveFirst'] then p_entity_invested[e])
  >=
  + p_group[g, 'invest_min_total']
;

s.t. minDivestGroup_entity_total {g in g_divest_total} :
  + sum{(g, e) in group_entity, d in period : e in entityDivest} v_divest[e, d] * p_entity_unitsize[e]
  + sum{(g, e) in group_entity : e in entityDivest} (if not p_model['solveFirst'] then p_entity_divested[e])
  >=
  + p_group[g, 'retire_min_total']
;

s.t. maxInvest_entity_period {(e, d) in ed_invest_period} :  # Covers both processes and nodes
  + v_invest[e, d] * p_entity_unitsize[e] 
  <= 
  + ed_invest_max_period[e, d]
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

s.t. maxInvest_entity_total {e in e_invest_total} :  # Covers both processes and nodes
  + sum{(e, d) in ed_invest} v_invest[e, d] * p_entity_unitsize[e] 
  + (if not p_model['solveFirst'] then p_entity_invested[e])
  <= 
  + e_invest_max_total[e]
;

s.t. maxDivest_entity_total {e in e_divest_total} :  # Covers both processes and nodes
  + sum{(e, d) in ed_divest} v_divest[e, d] * p_entity_unitsize[e] 
  + (if not p_model['solveFirst'] then p_entity_divested[e])
  <= 
  + e_divest_max_total[e]
;

s.t. minInvest_entity_total {e in e_invest_total} :  # Covers both processes and nodes
  + sum{(e, d) in ed_invest} v_invest[e, d] * p_entity_unitsize[e] 
  + (if not p_model['solveFirst'] then p_entity_invested[e])
  >= 
  + e_invest_min_total[e]
;

s.t. minDivest_entity_total {e in e_divest_total} :  # Covers both processes and nodes
  + sum{(e, d) in ed_divest} v_divest[e, d] * p_entity_unitsize[e] 
  + (if not p_model['solveFirst'] then p_entity_divested[e])
  >= 
  + e_divest_min_total[e]
;

s.t. maxCumulative_flow_solve {g in group : p_group[g, 'max_cumulative_flow']} :
  + sum{(g, p, n) in group_process_node, (d, t) in dt} (
      # n is sink
      + sum {(p, source, n) in process_source_sink} (
          + v_flow[p, source, n, d, t]
	    )  
      # n is source
      - sum {(p, n, sink) in process_source_sink_eff } ( 
          + v_flow[p, n, sink, d, t] 
	           * (if (p, 'min_load_efficiency') in process__ct_method then ptProcess_slope[p, t] else 1 / ptProcess[p, 'efficiency', t])
          + (if (p, 'min_load_efficiency') in process__ct_method then 
	           + v_online_linear[p, d, t] 
		    	    * ptProcess_section[p, t]
			    	* p_entity_unitsize[p]
		    )
        )		
      - sum {(p, n, sink) in process_source_sink_noEff } (
          + v_flow[p, n, sink, d, t] 
		)
	)
	<=
  + p_group[g, 'max_cumulative_flow'] 
      * hours_in_solve
;

s.t. minCumulative_flow_solve {g in group : p_group[g, 'min_cumulative_flow']} :
  + sum{(g, p, n) in group_process_node, (d, t) in dt} (
      # n is sink
      + sum {(p, source, n) in process_source_sink} (
          + v_flow[p, source, n, d, t]
	    )  
      # n is source
      - sum {(p, n, sink) in process_source_sink_eff } ( 
          + v_flow[p, n, sink, d, t] 
	           * (if (p, 'min_load_efficiency') in process__ct_method then ptProcess_slope[p, t] else 1 / ptProcess[p, 'efficiency', t])
          + (if (p, 'min_load_efficiency') in process__ct_method then 
	           + v_online_linear[p, d, t] 
		    	    * ptProcess_section[p, t]
			    	* p_entity_unitsize[p]
		    )
        )		
      - sum {(p, n, sink) in process_source_sink_noEff } (
          + v_flow[p, n, sink, d, t] 
		)  
	)
	>=
  + p_group[g, 'min_cumulative_flow'] 
      * hours_in_solve
;

s.t. maxCumulative_flow_period {g in group, d in period : pd_group[g, 'max_cumulative_flow', d]} :
  + sum{(g, p, n) in group_process_node, (d, t) in dt} (
      # n is sink
      + sum {(p, source, n) in process_source_sink} (
          + v_flow[p, source, n, d, t]
	    )  
      # n is source
      - sum {(p, n, sink) in process_source_sink_eff } ( 
          + v_flow[p, n, sink, d, t] 
	           * (if (p, 'min_load_efficiency') in process__ct_method then ptProcess_slope[p, t] else 1 / ptProcess[p, 'efficiency', t])
          + (if (p, 'min_load_efficiency') in process__ct_method then 
	           + v_online_linear[p, d, t] 
		    	    * ptProcess_section[p, t]
			    	* p_entity_unitsize[p]
		    )
        )		
      - sum {(p, n, sink) in process_source_sink_noEff } (
          + v_flow[p, n, sink, d, t] 
		)
	)   
	<=
  + pd_group[g, 'max_cumulative_flow', d] 
      * hours_in_period[d]
;

s.t. minCumulative_flow_period {g in group, d in period : pd_group[g, 'min_cumulative_flow', d]} :
  + sum{(g, p, n) in group_process_node, (d, t) in dt} (
      # n is sink
      + sum {(p, source, n) in process_source_sink} (
          + v_flow[p, source, n, d, t]
	    )  
      # n is source
      - sum {(p, n, sink) in process_source_sink_eff } ( 
          + v_flow[p, n, sink, d, t] 
	           * (if (p, 'min_load_efficiency') in process__ct_method then ptProcess_slope[p, t] else 1 / ptProcess[p, 'efficiency', t])
          + (if (p, 'min_load_efficiency') in process__ct_method then 
	           + v_online_linear[p, d, t] 
		    	    * ptProcess_section[p, t]
			    	* p_entity_unitsize[p]
		    )
        )		
      - sum {(p, n, sink) in process_source_sink_noEff } (
          + v_flow[p, n, sink, d, t] 
		)
	)
	>=
  + pd_group[g, 'min_cumulative_flow', d] 
      * hours_in_period[d]
;

s.t. maxInstant_flow {(g, d, t) in gdt_maxInstantFlow} :
  + sum{(g, p, n) in group_process_node} (
      # n is sink
      + sum {(p, source, n) in process_source_sink} (
          + v_flow[p, source, n, d, t]
	    )  
      # n is source
      - sum {(p, n, sink) in process_source_sink_eff } ( 
          + v_flow[p, n, sink, d, t] 
	           * (if (p, 'min_load_efficiency') in process__ct_method then ptProcess_slope[p, t] else 1 / ptProcess[p, 'efficiency', t])
          + (if (p, 'min_load_efficiency') in process__ct_method then 
	           + v_online_linear[p, d, t] 
		    	    * ptProcess_section[p, t]
			    	* p_entity_unitsize[p]
		    )
        )		
      - sum {(p, n, sink) in process_source_sink_noEff } 
          + v_flow[p, n, sink, d, t] 
	)
	<=
  + pdGroup[g, 'max_instant_flow', d] 
;

s.t. minInstant_flow {(g, d, t) in gdt_minInstantFlow} :
  + sum{(g, p, n) in group_process_node} (
      # n is sink
      + sum {(p, source, n) in process_source_sink} (
          + v_flow[p, source, n, d, t]
	    )  
      # n is source
      - sum {(p, n, sink) in process_source_sink_eff } ( 
          + v_flow[p, n, sink, d, t] 
	           * (if (p, 'min_load_efficiency') in process__ct_method then ptProcess_slope[p, t] else 1 / ptProcess[p, 'efficiency', t])
          + (if (p, 'min_load_efficiency') in process__ct_method then 
	           + v_online_linear[p, d, t] 
		    	    * ptProcess_section[p, t]
			    	* p_entity_unitsize[p]
		    )
        )		
      - sum {(p, n, sink) in process_source_sink_noEff } 
          + v_flow[p, n, sink, d, t] 
	)
	>=
  + pdGroup[g, 'min_instant_flow', d] 
;

s.t. inertia_constraint {g in groupInertia, (d, t) in dt} :
  + sum {(p, source, sink) in process_source_sink : (p, source) in process_source && (g, source) in group_node && p_process_source[p, source, 'inertia_constant']} 
    ( + (if p in process_online then v_online_linear[p, d, t]) 
	  + (if p not in process_online then v_flow[p, source, sink, d, t])
	) * p_process_source[p, source, 'inertia_constant']
  + sum {(p, source, sink) in process_source_sink : (p, sink) in process_sink && (g, sink) in group_node && p_process_sink[p, sink, 'inertia_constant']} 
    ( + (if p in process_online then v_online_linear[p, d, t]) 
	  + (if p not in process_online then v_flow[p, source, sink, d, t])
    ) * p_process_sink[p, sink, 'inertia_constant']
  + vq_inertia[g, d, t]
  >=
  + pdGroup[g, 'inertia_limit', d]
;

s.t. non_sync_constraint{g in groupNonSync, (d, t) in dt} :
  + sum {(p, source, sink) in process_source_sink : (p, sink) in process__sink_nonSync && (g, sink) in group_node}
    ( + v_flow[p, source, sink, d, t] )
  - vq_non_synchronous[g, d, t]
  <=
  ( + sum {(p, source, sink) in process_source_sink : (p, source) in process_source && (g, source) in group_node} 
        + v_flow[p, source, sink, d, t]
    + sum {(g, n) in group_node} ptNode[n, 'inflow', t]
  ) * pdGroup[g, 'non_synchronous_limit', d]
;

s.t. capacityMargin {g in groupCapacityMargin, (d, t, t_previous, t_previous_within_block, d_previous, t_previous_within_solve) in dtttdt : d in period_invest} :
  # profile limited units producing to a node in the group (based on available capacity)
  + sum {(p, source, sink, f, m) in process__source__sink__profile__profile_method 
         : m = 'upper_limit' || m = 'fixed'
           && (p, sink) in process_sink
		   && (g, sink) in group_node
		   && p in process_unit
		} 
    ( + pt_profile[f, t]
        * ( + p_entity_all_existing[p]
            + sum {(p, d_invest) in pd_invest : d_invest <= d} v_invest[p, d_invest] * p_entity_unitsize[p]
            - sum {(p, d_invest) in pd_divest : d_invest <= d} v_divest[p, d_invest] * p_entity_unitsize[p]
	      )
    )
  # capacity limited units producing to a node in the group (based on available capacity)
  + sum {(p, source, sink) in process_source_sink 
         : (p, sink) in process_sink 
	       && sum {(p, source, sink, f, m) in process__source__sink__profile__profile_method : m = 'upper_limit' || m = 'fixed'} 1 = 0
		   && (p, sink) in process_sink
  		   && (g, sink) in group_node 
		   && p in process_unit
		} 
	(
      + ( + p_entity_all_existing[p]
          + sum {(p, d_invest) in pd_invest : d_invest <= d} v_invest[p, d_invest] * p_entity_unitsize[p]
          - sum {(p, d_invest) in pd_divest : d_invest <= d} v_divest[p, d_invest] * p_entity_unitsize[p]
        )
	)
  # profile or capacity limited units consuming from a node in the group (as they consume in any given time step)
  - sum {(p, source, sink) in process_source_sink 
         : (p, source) in process_source
		   && (g, source) in group_node
		   && p in process_unit
		} 
    ( + if (p, source, sink) in process_source_sink_eff then 
        ( + v_flow[p, source, sink, d, t] 
	          * (if (p, 'min_load_efficiency') in process__ct_method then ptProcess_slope[p, t] else 1 / ptProcess[p, 'efficiency', t])
		      * p_process_sink_coefficient[p, sink] / p_process_source_coefficient[p, source]
          + (if (p, 'min_load_efficiency') in process__ct_method then 
	            + v_online_linear[p, d, t]
			        * ptProcess_section[p, t]
				    * p_entity_unitsize[p]
	  	    )
	    )
	  + if (p, source, sink) in process_source_sink_noEff then
        ( + v_flow[p, source, sink, d, t] 
        )
	)
  + vq_capacity_margin[g, d]
  >=
  + sum {(g, n) in group_node} 
    ( - (if (n, 'no_inflow') not in node__inflow_method then pdtNodeInflow[n, d, t])
      + (if n in nodeState && (n, 'bind_forward_only') in node__storage_binding_method && ((d, t) not in period__time_first && d in period_first) then (v_state[n, d, t] -  v_state[n, d_previous, t_previous_within_solve]))
      + (if n in nodeState && (n, 'bind_within_solve') in node__storage_binding_method then (v_state[n, d, t] -  v_state[n, d_previous, t_previous_within_solve]))
      + (if n in nodeState && (n, 'bind_within_period') in node__storage_binding_method then (v_state[n, d, t] -  v_state[n, d, t_previous]))
      + (if n in nodeState && (n, 'bind_within_timeblock') in node__storage_binding_method then (v_state[n, d, t] -  v_state[n, d, t_previous_within_block]))
	)
  + pdGroup[g, 'capacity_margin', d]
;
param rest := gmtime() - datetime0 - setup1 - w_calc_slope - setup2 - w_total_cost - balance - reserves - indirect;
display rest;

solve;

param w_solve := gmtime() - datetime0 - setup1 - w_calc_slope - setup2 - w_total_cost - balance - reserves - indirect - rest;
display w_solve;

param entity_all_capacity{e in entity, d in period_realized} :=
  + p_entity_all_existing[e]
  + sum {(e, d2) in ed_invest : d2 <= d} v_invest[e, d2].val * p_entity_unitsize[e]
  - sum {(e, d2) in ed_divest : d2 <= d} v_divest[e, d2].val * p_entity_unitsize[e]
;

param r_process_source_sink_flow_dt{(p, source, sink) in process_source_sink_alwaysProcess, (d, t) in dt} :=
  + sum {(p, m) in process_method : m in method_1var_per_way}
    ( + sum {(p, source, sink2) in process_source_toSink} 
        ( + v_flow[p, source, sink2, d, t].val 
	          * (if (p, 'min_load_efficiency') in process__ct_method then ptProcess_slope[p, t] else 1 / ptProcess[p, 'efficiency', t])
	  		  * (if p in process_unit then p_process_sink_coefficient[p, sink2] / p_process_source_coefficient[p, source] else 1)
          + (if (p, 'min_load_efficiency') in process__ct_method then v_online_linear[p, d, t] * ptProcess_section[p, t] * p_entity_unitsize[p])
	    )
      + sum {(p, source2, sink) in process_source_toSink} 
          + v_flow[p, source2, sink, d, t].val 
      + sum {(p, source, sink2) in process_sink_toSource} 
        ( + v_flow[p, source, sink2, d, t].val 
	          * (if (p, 'min_load_efficiency') in process__ct_method then ptProcess_slope[p, t] else 1 / ptProcess[p, 'efficiency', t])
          + (if (p, 'min_load_efficiency') in process__ct_method then v_online_linear[p, d, t] * ptProcess_section[p, t] * p_entity_unitsize[p])
	    )
      + sum {(p, source2, sink) in process_sink_toSource} 
          + v_flow[p, source2, sink, d, t].val 
      + (if (p, source, sink) in process__profileProcess__toSink then 
	      + v_flow[p, source, sink, d, t].val)
      + (if (p, source, sink) in process__source__toProfileProcess then 
	      + v_flow[p, source, sink, d, t].val)
   )
  + sum {(p, m) in process_method : m not in method_1var_per_way} (
      + v_flow[p, source, sink, d, t].val 
	)
;

param r_process_source_sink_ramp_dt{(p, source, sink) in process_source_sink_alwaysProcess, (d, t) in dt} :=
  + r_process_source_sink_flow_dt[p, source, sink, d, t]
  - sum{(d, t, t_previous, t_previous_within_block, d_previous, t_previous_within_solve) in dtttdt} r_process_source_sink_flow_dt[p, source, sink, d, t_previous]
;

param r_node_ramp_dt{n in nodeBalance, (d, t) in dt} :=
  + sum {(p, n, sink) in process_source_sink_alwaysProcess} r_process_source_sink_ramp_dt[p, n, sink, d, t]
  + sum {(p, source, n) in process_source_sink_alwaysProcess} -r_process_source_sink_ramp_dt[p, source, n, d, t]
;

param r_process_source_sink_flow_d{(p, source, sink) in process_source_sink_alwaysProcess, d in period} :=
  + sum {(d, t) in dt} ( r_process_source_sink_flow_dt [p, source, sink, d, t] * step_duration[d, t] )
;
param r_process_source_flow_d{(p, source) in process_source, d in period_realized} := 
  + sum {(p, source, sink) in process_source_sink_alwaysProcess} r_process_source_sink_flow_d[p, source, sink, d]
;
param r_process_sink_flow_d{(p, sink) in process_sink, d in period_realized} := 
  + sum {(p, source, sink) in process_source_sink_alwaysProcess} r_process_source_sink_flow_d[p, source, sink, d]
;

param r_nodeState_change_dt{n in nodeState, (d, t_previous) in dt} := sum {(d, t, t_previous, t_previous_within_block, d_previous, t_previous_within_solve) in dtttdt} (
      + (if (n, 'bind_forward_only') in node__storage_binding_method && ((d, t) not in period__time_first && d in period_first) then (v_state[n, d, t] -  v_state[n, d_previous, t_previous_within_solve]))
      + (if (n, 'bind_within_solve') in node__storage_binding_method then (v_state[n, d, t] -  v_state[n, d_previous, t_previous_within_solve]))
      + (if (n, 'bind_within_period') in node__storage_binding_method then (v_state[n, d, t] -  v_state[n, d, t_previous]))
      + (if (n, 'bind_within_timeblock') in node__storage_binding_method then (v_state[n, d, t] -  v_state[n, d, t_previous_within_block]))
);
param r_nodeState_change_d{n in nodeState, d in period} := sum {(d, t) in dt} r_nodeState_change_dt[n, d, t];
param r_selfDischargeLoss_dt{n in nodeSelfDischarge, (d, t) in dt} := v_state[n, d, t] * ptNode[n, 'self_discharge_loss', t] * step_duration[d, t];
param r_selfDischargeLoss_d{n in nodeSelfDischarge, d in period} := sum{(d, t) in dt} r_selfDischargeLoss_dt[n, d, t];

param r_cost_commodity_dt{(c, n) in commodity_node, (d, t) in dt} := 
  + step_duration[d, t] 
      * pdCommodity[c, 'price', d] 
      * ( + sum{(p, n, sink) in process_source_sink_alwaysProcess}
              + r_process_source_sink_flow_dt[p, n, sink, d, t]
		  - sum{(p, source, n) in process_source_sink_alwaysProcess}	  
              + r_process_source_sink_flow_dt[p, source, n, d, t]
	    )
;

param r_process_commodity_d{(p, c, n) in process__commodity__node, d in period} :=
 + sum{(p, n, sink) in process_source_sink_alwaysProcess}
      + r_process_source_sink_flow_d[p, n, sink, d]
 - sum{(p, source, n) in process_source_sink_alwaysProcess}	  
      + r_process_source_sink_flow_d[p, source, n, d]
;

param r_process_emissions_co2_dt{(p, c, n) in process__commodity__node_co2, (d, t) in dt} := 
  + step_duration[d, t]
      * p_commodity[c, 'co2_content'] 
      * ( + sum{(p, n, sink) in process_source_sink_alwaysProcess}
              + r_process_source_sink_flow_dt[p, n, sink, d, t]
	      - sum{(p, source, n) in process_source_sink_alwaysProcess}	  
              + r_process_source_sink_flow_dt[p, source, n, d, t]
        )
;	  

param r_process_emissions_co2_d{(p, c, n) in process__commodity__node_co2, d in period} :=
  + sum{t in time : (d, t) in dt} r_process_emissions_co2_dt[p, c, n, d, t];

param r_emissions_co2_dt{(c, n) in commodity_node_co2, (d, t) in dt} :=
  + sum{(p, c, n) in process__commodity__node_co2} r_process_emissions_co2_dt[p, c, n, d, t];

param r_emissions_co2_d{(c, n) in commodity_node_co2, d in period} :=
  + sum{t in time : (d, t) in dt} r_emissions_co2_dt[c, n, d, t];

param r_cost_co2_dt{(g, c, n, d, t) in gcndt_co2} := 
  + r_emissions_co2_dt[c, n, d, t] 
    * pdGroup[g, 'co2_price', d]
;	  

param r_cost_process_variable_cost_dt{p in process, (d, t) in dt} :=
  + step_duration[d, t]
      * sum{(p, source, sink) in process_source_sink_alwaysProcess}
          + ptProcess__source__sink__t_varCost_alwaysProcess[p, source, sink, t]
	          * r_process_source_sink_flow_dt[p, source, sink, d, t]
#	  * ( + sum {(p, source, sink, 'variable_cost') in process__source__sink__param_t}
#	        ( + sum{(p, n, sink) in process_source_sink_alwaysProcess : (p, sink) in process_sink}
#  			      + ptProcess_source_sink[p, source, sink, 'variable_cost', t]
#		              * r_process_source_sink_flow_dt[p, n, sink, d, t]
#	          + sum{(p, source, n) in process_source_sink_alwaysProcess : (p, source) in process_source}
#  			      + ptProcess_source_sink[p, source, sink, 'variable_cost', t]
#		              * r_process_source_sink_flow_dt[p, source, n, d, t]
#			)
#		)
;
#param r_cost_process_ramp_cost_dt{p in process, (d, t) in dt :
#  sum {(p, source, sink, m) in process__source__sink__ramp_method : m in ramp_cost_method} 1 } :=
#  + step_duration[d, t]
#	  * sum {(p, source, sink, m) in process__source__sink__ramp_method : m in ramp_cost_method} 
#	      + pProcess_source_sink[p, source, sink, 'ramp_cost']
#              * v_ramp[p, source, sink, d, t].val
#;
param r_cost_startup_dt{p in process, (d, t) in dt : p in process_online && pdProcess[p, 'startup_cost', d]} :=
  (v_startup_linear[p, d, t] * pdProcess[p, 'startup_cost', d]);

param r_costPenalty_nodeState_upDown_dt{n in nodeBalance, ud in upDown, (d, t) in dt} :=
  + (if ud = 'up'   then step_duration[d, t] * vq_state_up[n, d, t] * ptNode[n, 'penalty_up', t])
  + (if ud = 'down' then step_duration[d, t] * vq_state_down[n, d, t] * ptNode[n, 'penalty_down', t]) ;

param r_penalty_nodeState_upDown_d{n in nodeBalance, ud in upDown, d in period} :=
  + sum {(d, t) in dt : ud = 'up'} step_duration[d, t] * vq_state_up[n, d, t]
  + sum {(d, t) in dt : ud = 'down'} step_duration[d, t] * vq_state_down[n, d, t] ;

param r_costPenalty_inertia_dt{g in groupInertia, (d, t) in dt} :=
  + step_duration[d, t]
      * vq_inertia[g, d, t] 
	  * pdGroup[g, 'penalty_inertia', d]
;

param r_costPenalty_non_synchronous_dt{g in groupNonSync, (d, t) in dt} :=
  + step_duration[d, t]
      * vq_non_synchronous[g, d, t] 
	  * pdGroup[g, 'penalty_non_synchronous', d]
;

param r_costPenalty_capacity_margin_d{g in groupCapacityMargin, d in period_invest} :=
  + vq_capacity_margin[g, d]
      * pdGroup[g, 'penalty_capacity_margin', d]
;

param r_costPenalty_reserve_upDown_dt{(r, ud, ng) in reserve__upDown__group, (d, t) in dt} :=
  + step_duration[d, t]
      * (
          + vq_reserve[r, ud, ng, d, t] * p_reserve_upDown_group[r, ud, ng, 'penalty_reserve']
	    )
;
		
param r_cost_entity_invest_d{(e, d) in ed_invest} :=
  + v_invest[e, d]
      * p_entity_unitsize[e]
      * ed_entity_annual[e, d]
	  * p_discount_in_perpetuity_investment[d]
;

param r_cost_entity_divest_d{(e, d) in ed_divest} :=
  - v_divest[e, d]
      * p_entity_unitsize[e]
      * ed_entity_annual_divest[e, d]
	  * p_discount_in_perpetuity_investment[d]
;

param r_costOper_dt{(d, t) in dt} :=
  + sum{(c, n) in commodity_node} r_cost_commodity_dt[c, n, d, t]
  + sum{(g, c, n, d, t) in gcndt_co2} r_cost_co2_dt[g, c, n, d, t]
  + sum{p in process} r_cost_process_variable_cost_dt[p, d, t]
#  + sum{(p, source, sink, m) in process__source__sink__ramp_method : m in ramp_cost_method}
#      + r_cost_process_ramp_cost_dt[p, d, t]
  + sum{p in process_online : pdProcess[p, 'startup_cost', d]} r_cost_startup_dt[p, d, t]
;

param r_costPenalty_dt{(d, t) in dt} :=
  + sum{n in nodeBalance, ud in upDown} r_costPenalty_nodeState_upDown_dt[n, ud, d, t]
  + sum{g in groupInertia} r_costPenalty_inertia_dt[g, d, t]
  + sum{g in groupNonSync} r_costPenalty_non_synchronous_dt[g, d, t]
  + sum{(r, ud, ng) in reserve__upDown__group} r_costPenalty_reserve_upDown_dt[r, ud, ng, d, t]
;

param r_costOper_and_penalty_dt{(d,t) in dt} :=
  + r_costOper_dt[d, t]
  + r_costPenalty_dt[d, t]
;

param r_cost_process_variable_cost_d{p in process, d in period} := sum{(d, t) in dt} r_cost_process_variable_cost_dt[p, d, t];
param r_cost_co2_d{d in period} := sum{(g, c, n, d, t) in gcndt_co2} r_cost_co2_dt[g, c, n, d, t];
param r_cost_commodity_d{d in period} := sum{(c, n) in commodity_node, (d, t) in dt} r_cost_commodity_dt[c, n, d, t];
param r_cost_variable_d{d in period} := sum{p in process} r_cost_process_variable_cost_d[p, d];
#param r_cost_ramp_d{d in period} := sum{(p, source, sink, m) in process__source__sink__ramp_method, (d, t) in dt : m in ramp_cost_method} r_cost_process_ramp_cost_dt[p, d, t];
param r_cost_startup_d{d in period} := sum{p in process_online, (d, t) in dt : pdProcess[p, 'startup_cost', d]} r_cost_startup_dt[p, d, t];

param r_costPenalty_nodeState_upDown_d{n in nodeBalance, ud in upDown, d in period} := sum{(d, t) in dt} r_costPenalty_nodeState_upDown_dt[n, ud, d, t];
param r_costPenalty_inertia_d{g in groupInertia, d in period} := sum{(d, t) in dt} r_costPenalty_inertia_dt[g, d, t];
param r_costPenalty_non_synchronous_d{g in groupNonSync, d in period} := sum{(d, t) in dt} r_costPenalty_non_synchronous_dt[g, d, t];
param r_costPenalty_reserve_upDown_d{(r, ud, ng) in reserve__upDown__group, d in period} := sum{(d, t) in dt} r_costPenalty_reserve_upDown_dt[r, ud, ng, d, t];

param r_costOper_d{d in period} := sum{(d, t) in dt} r_costOper_dt[d, t] * step_duration[d, t];
param r_costPenalty_d{d in period} := sum{(d, t) in dt} r_costPenalty_dt[d, t] * step_duration[d, t] + sum{g in groupCapacityMargin} r_costPenalty_capacity_margin_d[g, d];
param r_costOper_and_penalty_d{d in period} := + r_costOper_d[d] + r_costPenalty_d[d];

param r_costInvestUnit_d{d in period} :=
  + sum{(e, d) in ed_invest : e in process_unit} r_cost_entity_invest_d[e, d]
;
param r_costDivestUnit_d{d in period} :=
  + sum{(e, d) in ed_divest : e in process_unit} r_cost_entity_divest_d[e, d]
;
param r_costInvestConnection_d{d in period} :=
  + sum{(e, d) in ed_invest : e in process_connection} r_cost_entity_invest_d[e, d]
;
param r_costDivestConnection_d{d in period} :=
  + sum{(e, d) in ed_divest : e in process_connection} r_cost_entity_divest_d[e, d]
;
param r_costInvestState_d{d in period} :=
  + sum{(e, d) in ed_invest : e in nodeState} r_cost_entity_invest_d[e, d]
;
param r_costDivestState_d{d in period} :=
  + sum{(e, d) in ed_divest : e in nodeState} r_cost_entity_divest_d[e, d]
;

param r_costInvest_d{d in period} := r_costInvestUnit_d[d] + r_costInvestConnection_d[d] + r_costInvestState_d[d];
param r_costDivest_d{d in period} := r_costDivestUnit_d[d] + r_costDivestConnection_d[d] + r_costDivestState_d[d];

param pdNodeInflow{n in node, d in period} := sum{(d, t) in dt} pdtNodeInflow[n, d, t];

param potentialVREgen{(p, n) in process_sink, d in period_realized : p in process_VRE} :=
  + sum{(p, source, n, f, m) in process__source__sink__profile__profile_method, (d, t) in dt : m = 'upper_limit'} 
      + pt_profile[f, t] * entity_all_capacity[p, d];

printf 'Transfer investments to the next solve...\n';
param fn_entity_invested symbolic := "solve_data/p_entity_invested.csv";
printf 'entity,p_entity_invested\n' > fn_entity_invested;
for {e in entity: e in entityInvest} 
  {
    printf '%s,%.8g\n', e, 
	  + (if not p_model['solveFirst'] then p_entity_invested[e] else 0)
	  + sum {(e, d_invest) in ed_invest} v_invest[e, d_invest].val * p_entity_unitsize[e]
	>> fn_entity_invested;
  }

printf 'Write unit capacity results...\n';
param fn_unit_capacity symbolic := "output/unit_capacity__period.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf 'unit,solve,period,existing,invested,divested,total\n' > fn_unit_capacity; }  # Clear the file on the first solve
for {s in solve_current, p in process_unit, d in period_realized}
  {
    printf '%s,%s,%s,%.8g,%.8g,%.8g,%.8g\n', p, s, d, 
	        p_entity_all_existing[p], 
			(if (p, d) in pd_invest then v_invest[p, d].val * p_entity_unitsize[p] else 0), 
			(if (p, d) in pd_divest then v_divest[p, d].val * p_entity_unitsize[p] else 0), 
			entity_all_capacity[p, d]
#			+ (if entity_all_capacity[p, d] then sum{(p, sink) in process_sink} (r_process_sink_flow_d[p, sink, d]) / entity_all_capacity[p,d] else 0),
#			+ r_cost_process_variable_cost_d[p, d]
	>> fn_unit_capacity;
  }

printf 'Write connection capacity results...\n';
param fn_connection_capacity symbolic := "output/connection_capacity__period.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf 'connection,solve,period,existing,invested,divested,total\n' > fn_connection_capacity; }  # Clear the file on the first solve
for {s in solve_current, p in process_connection, d in period_realized}
  {
    printf '%s,%s,%s,%.8g,%.8g,%.8g,%.8g\n', p, s, d, 
	        p_entity_all_existing[p],
			(if (p, d) in pd_invest then v_invest[p, d].val * p_entity_unitsize[p] else 0),
			(if (p, d) in pd_divest then v_divest[p, d].val * p_entity_unitsize[p] else 0),
			+ p_entity_all_existing[p] 
			+ (if (p, d) in pd_invest then v_invest[p, d].val * p_entity_unitsize[p] else 0)
			- (if (p, d) in pd_divest then v_divest[p, d].val * p_entity_unitsize[p] else 0)
	>> fn_connection_capacity;
  }

printf 'Write node/storage capacity results...\n';
param fn_node_capacity symbolic := "output/node_capacity__period.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf 'node,solve,period,existing,invested,divested,total\n' > fn_node_capacity; }  # Clear the file on the first solve
for {s in solve_current, e in nodeState, d in period_realized}
  {
    printf '%s,%s,%s,%.8g,%.8g,%.8g,%.8g\n', e, s, d, 
	        p_entity_all_existing[e],
			(if (e, d) in ed_invest then v_invest[e, d].val * p_entity_unitsize[e] else 0),
			(if (e, d) in ed_divest then v_divest[e, d].val * p_entity_unitsize[e] else 0),
			+ p_entity_all_existing[e]
			+ (if (e, d) in ed_invest then v_invest[e, d].val * p_entity_unitsize[e] else 0)
			- (if (e, d) in ed_divest then v_divest[e, d].val * p_entity_unitsize[e] else 0)
	 >> fn_node_capacity;
  }


printf 'Write summary results...\n';
param fn_summary symbolic := "output/summary_solve.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf '"Diagnostic results from all solves. Output at (UTC): %s"', time2str(gmtime(), "%FT%TZ") > fn_summary; }
for {s in solve_current} { printf '\n\n"Solve",%s\n', s >> fn_summary; }
printf '"Total cost obj. function (M CUR)",%.12g,"Minimized total system cost as ', (total_cost.val / 1000000) >> fn_summary;
printf 'given by the solver (includes all penalty costs)"\n' >> fn_summary;
printf '"Total cost (calculated) full horizon (M CUR)",%.12g,', sum{d in period} 
           ( + r_costOper_and_penalty_d[d] * p_discount_with_perpetuity_operations[d] / period_share_of_year[d] 
		     + r_costInvest_d[d]
			 + r_costDivest_d[d]
		   ) / 1000000 >> fn_summary;
printf '"Annualized operational, penalty and investment costs"\n' >> fn_summary;
printf '"Total cost (calculated) realized periods (M CUR)",%.12g\n', sum{d in period_realized} 
           ( + r_costOper_and_penalty_d[d] * p_discount_with_perpetuity_operations[d] / period_share_of_year[d] 
		     + r_costInvest_d[d]
			 + r_costDivest_d[d]
		   ) / 1000000 >> fn_summary;
printf '"Operational costs for realized periods (M CUR)",%.12g\n', sum{d in period_realized} 
           + r_costOper_d[d] * p_discount_with_perpetuity_operations[d] / period_share_of_year[d] / 1000000>> fn_summary;
printf '"Investment costs for realized periods (M CUR)",%.12g\n', sum{d in period_realized} 
           + r_costInvest_d[d] / 1000000 >> fn_summary;
printf '"Retirement costs (negative salvage value) for realized periods (M CUR)",%.12g\n', sum{d in period_realized} 
           + r_costDivest_d[d] / 1000000 >> fn_summary;
printf '"Penalty (slack) costs for realized periods (M CUR)",%.12g\n', sum{d in period_realized} 
           + r_costPenalty_d[d] * p_discount_with_perpetuity_operations[d] / period_share_of_year[d] / 1000000 >> fn_summary;
printf '\nPeriod' >> fn_summary;
for {d in period}
  { printf ',%s', d >> fn_summary; }
printf '\n"Time in use in years"' >> fn_summary;
for {d in period}
  { printf ',%.12g', period_share_of_year[d] >> fn_summary; }
printf '\n"Operational discount factor"' >> fn_summary;
for {d in period}
  { printf ',%.12g', p_discount_in_perpetuity_operations[d] >> fn_summary; }
printf '\n"Investment discount factor"' >> fn_summary;
for {d in period}
  { printf ',%.12g', p_discount_in_perpetuity_investment[d] >> fn_summary; }
printf '\n' >> fn_summary;

printf '\nEmissions\n' >> fn_summary;
printf '"CO2 (Mt)",%.6g,"System-wide annualized CO2 emissions for all periods"\n', sum{(c, n) in commodity_node_co2, d in period} (r_emissions_co2_d[c, n, d] / period_share_of_year[d]) / 1000000 >> fn_summary;
printf '"CO2 (Mt)",%.6g,"System-wide annualized CO2 emissions for realized periods"\n', sum{(c, n) in commodity_node_co2, d in period_realized} (r_emissions_co2_d[c, n, d] / period_share_of_year[d]) / 1000000 >> fn_summary;

printf '\n"Slack variables (creating or removing energy/matter, creating inertia, ' >> fn_summary;
printf 'changing non-synchronous generation to synchronous)"\n' >> fn_summary;
for {n in nodeBalance}
  {  
    for {d in period : r_penalty_nodeState_upDown_d[n, 'up', d]}
      {
	    printf 'Created, %s, %s, %.5g\n', n, d, r_penalty_nodeState_upDown_d[n, 'up', d] >> fn_summary;
      }
  }

for {n in nodeBalance}
  {  
    for {d in period : r_penalty_nodeState_upDown_d[n, 'down', d]}
      {
	    printf 'Removed, %s, %s, %.5g\n', n, d, r_penalty_nodeState_upDown_d[n, 'down', d] >> fn_summary;
      }
  }

for {g in groupInertia}
  {
    for {d in period : r_costPenalty_inertia_d[g, d]}
	  {
        printf 'Inertia, %s, %s, %.5g\n', g, d, r_costPenalty_inertia_d[g, d] / pdGroup[g, 'penalty_inertia', d] >> fn_summary;
	  }
  }

for {g in groupNonSync}
  {
    for {d in period : r_costPenalty_non_synchronous_d[g, d]}
	  {
        printf 'NonSync, %s, %s, %.5g\n', g, d, r_costPenalty_non_synchronous_d[g, d] / pdGroup[g, 'penalty_non_synchronous', d] >> fn_summary;
	  }
  }

for {g in groupCapacityMargin}
  {
    for {d in period_invest : r_costPenalty_capacity_margin_d[g, d]}
	  {
        printf 'CapMargin, %s, %s, %.5g\n', g, d, r_costPenalty_capacity_margin_d[g, d] / pdGroup[g, 'penalty_capacity_margin', d] >> fn_summary;
	  }
  }


printf 'Write group results for nodes...\n';
param fn_groupNode__d symbolic := "output/group_node__period.csv";
for {i in 1..1 : p_model['solveFirst']}
  { 
    printf 'group,solve,period,"sum of annualized inflows [MWh]","VRE share [\% of annual inflow]",' > fn_groupNode__d;
	printf '"curtailed VRE share, [\% of annual inflow]","upward slack [\% of annual inflow]",' >> fn_groupNode__d;
	printf '"downward slack [\% of annual inflow]"\n' >> fn_groupNode__d;
  }
for {g in groupOutput, s in solve_current, d in period_realized : sum{(g, n) in group_node} pdNodeInflow[n, d]}
  {
    printf '%s,%s,%s,%.8g,%.8g,%.8g,%.8g,%.8g\n', g, s, d 
       , sum{(g, n) in group_node} pdNodeInflow[n, d] / period_share_of_year[d]
       , ( sum{(p, source, n) in process_source_sink_alwaysProcess : (g, n) in group_node && p in process_VRE && (p, n) in process_sink} 
	             r_process_source_sink_flow_d[p, source, n, d]  
		 ) / ( - sum{(g, n) in group_node} pdNodeInflow[n, d] ) * 100	   
	   , ( + sum{(p, n) in process_sink : p in process_VRE} potentialVREgen[p, n, d]
	       - sum{(p, source, n) in process_source_sink_alwaysProcess : (g, n) in group_node && p in process_VRE && (p, n) in process_sink} 
		         r_process_source_sink_flow_d[p, source, n, d] 
		 ) / ( - sum{(g, n) in group_node} pdNodeInflow[n, d] ) * 100
	  , ( sum{(g, n) in group_node} r_penalty_nodeState_upDown_d[n, 'up', d] ) 
	    / ( - sum{(g, n) in group_node} pdNodeInflow[n, d] ) * 100
	  , ( sum{(g, n) in group_node} r_penalty_nodeState_upDown_d[n, 'down', d] ) 
	    / ( - sum{(g, n) in group_node} pdNodeInflow[n, d] ) * 100
	>> fn_groupNode__d;
  }


printf 'Write cost summary for realized periods...\n';
param fn_summary_cost symbolic := "output/costs__period.csv";
for {i in 1..1 : p_model['solveFirst']}
  { 
    printf 'solve,period,total,"unit investment","connection investment","storage investment",' > fn_summary_cost;
    printf 'commodity,CO2,"variable cost",starts,"upward penalty","downward penalty","inertia penalty",' >> fn_summary_cost;
	printf '"non-synchronous penalty","capacity margin penalty","upward reserve penalty",' >> fn_summary_cost;
	printf '"downward reserve penalty"\n' >> fn_summary_cost;
  }
for {s in solve_current, d in period_realized}
  { 
    printf '%s,%s,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n', 
      s, d,
	  (r_costOper_and_penalty_d[d] / period_share_of_year[d] + r_costInvest_d[d]) / 1000000,
      r_costInvestUnit_d[d] / 1000000,
      r_costInvestConnection_d[d] / 1000000,
      r_costInvestState_d[d] / 1000000,
	  r_cost_commodity_d[d] / period_share_of_year[d] / 1000000,
	  r_cost_co2_d[d] / period_share_of_year[d] / 1000000,
	  r_cost_variable_d[d] / period_share_of_year[d] / 1000000,
	  r_cost_startup_d[d] / period_share_of_year[d] / 1000000,
	  sum{n in nodeBalance} (r_costPenalty_nodeState_upDown_d[n, 'up', d] / period_share_of_year[d]) / 1000000,
	  sum{n in nodeBalance} (r_costPenalty_nodeState_upDown_d[n, 'down', d] / period_share_of_year[d]) / 1000000,
	  sum{g in groupInertia} (r_costPenalty_inertia_d[g, d] / period_share_of_year[d]) / 1000000,
	  sum{g in groupNonSync} (r_costPenalty_non_synchronous_d[g, d] / period_share_of_year[d]) / 1000000,
	  sum{g in groupCapacityMargin} (r_costPenalty_capacity_margin_d[g, d] / period_share_of_year[d]) / 1000000,
	  sum{(r, ud, ng) in reserve__upDown__group : ud = 'up'} (r_costPenalty_reserve_upDown_d[r, ud, ng, d] / period_share_of_year[d]) / 1000000,
	  sum{(r, ud, ng) in reserve__upDown__group : ud = 'down'} (r_costPenalty_reserve_upDown_d[r, ud, ng, d] / period_share_of_year[d]) / 1000000
	>> fn_summary_cost;
  } 

printf 'Write cost for realized periods and t...\n';
param fn_summary_cost_dt symbolic := "output/costs__period__t.csv";
for {i in 1..1 : p_model['solveFirst']}
  { 
    printf 'solve,period,time,commodity,CO2,variable,starts,"upward slack penalty","downward slack penalty",' > fn_summary_cost_dt;
	printf '"inertia slack penalty","non-synchronous slack penalty","upward reserve slack penalty",' >> fn_summary_cost_dt;
	printf '""downward reserves slack penalty"\n' >> fn_summary_cost_dt;
  }
for {s in solve_current, (d, t) in dt : d in period_realized}
  { 
    printf '%s,%s,%s,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n', 
      s, d, t,
	  sum{(c, n) in commodity_node} r_cost_commodity_dt[c, n, d, t],
	  sum{(g, c, n, d, t) in gcndt_co2} r_cost_co2_dt[g, c, n, d, t],
	  sum{p in process} r_cost_process_variable_cost_dt[p, d, t],
	  sum{p in process_online : pdProcess[p, 'startup_cost', d]} r_cost_startup_dt[p, d, t],
	  sum{n in nodeBalance} (r_costPenalty_nodeState_upDown_dt[n, 'up', d, t]),
	  sum{n in nodeBalance} (r_costPenalty_nodeState_upDown_dt[n, 'down', d, t]),
	  sum{g in groupInertia} (r_costPenalty_inertia_dt[g, d, t]),
	  sum{g in groupNonSync} (r_costPenalty_non_synchronous_dt[g, d, t]),
	  sum{(r, ud, ng) in reserve__upDown__group : ud = 'up'} (r_costPenalty_reserve_upDown_dt[r, ud, ng, d, t]),
	  sum{(r, ud, ng) in reserve__upDown__group : ud = 'down'} (r_costPenalty_reserve_upDown_dt[r, ud, ng, d, t])
	>> fn_summary_cost_dt;
  } 

printf 'Write unit__outputNode flow for periods...\n';
param fn_unit__sinkNode__d symbolic := "output/unit__outputNode__period.csv";
for {i in 1..1 : p_model['solveFirst']}
  { 
    printf 'solve,period' > fn_unit__sinkNode__d;  # Print the header on the first solve
	for {(u, sink) in process_sink : u in process_unit} printf ',%s', u >> fn_unit__sinkNode__d;
	printf '\n,' >> fn_unit__sinkNode__d;
	for {(u, sink) in process_sink : u in process_unit} printf ',%s', sink >> fn_unit__sinkNode__d;
  }
for {s in solve_current, d in period_realized}
  {
	printf '\n%s,%s', s, d >> fn_unit__sinkNode__d;
    for {(u, sink) in process_sink : u in process_unit}
      { printf ',%.8g', r_process_sink_flow_d[u, sink, d] / period_share_of_year[d] >> fn_unit__sinkNode__d; }
  } 

printf 'Write unit__outputNode flow for time...\n';
param fn_unit__sinkNode__dt symbolic := "output/unit__outputNode__period__t.csv";
for {i in 1..1 : p_model['solveFirst']}
  {
    printf 'solve,period,time' > fn_unit__sinkNode__dt;  # Print the header on the first solve
	for {(u, source, sink) in process_source_sink_alwaysProcess : (u, sink) in process_sink && u in process_unit} printf ',%s', u >> fn_unit__sinkNode__dt;
	printf '\n,,' >> fn_unit__sinkNode__dt;
	for {(u, source, sink) in process_source_sink_alwaysProcess : (u, sink) in process_sink && u in process_unit} printf ',%s', sink >> fn_unit__sinkNode__dt;
  }
for {s in solve_current, (d, t) in dt : d in period_realized}
  {
	printf '\n%s,%s,%s', s, d, t >> fn_unit__sinkNode__dt;
    for {(u, source, sink) in process_source_sink_alwaysProcess : (u, sink) in process_sink && u in process_unit}
      { printf ',%.8g', r_process_source_sink_flow_dt[u, source, sink, d, t] >> fn_unit__sinkNode__dt; }
  } 

printf 'Write unit__inputNode flow for periods...\n';
param fn_unit__sourceNode__d symbolic := "output/unit__inputNode__period.csv";
for {i in 1..1 : p_model['solveFirst']}
  {
    printf 'solve,period' > fn_unit__sourceNode__d;  # Print the header on the first solve
	for {(u, source) in process_source : u in process_unit} printf ',%s', u >> fn_unit__sourceNode__d;
	printf '\n,' >> fn_unit__sourceNode__d;
	for {(u, source) in process_source : u in process_unit} printf ',%s', source >> fn_unit__sourceNode__d;
  }
for {s in solve_current, d in period_realized}
  {
	printf '\n%s,%s', s, d >> fn_unit__sourceNode__d;
    for {(u, source) in process_source : u in process_unit}
      { printf ',%.8g', r_process_source_flow_d[u, source, d] / period_share_of_year[d] >> fn_unit__sourceNode__d; }
  } 

printf 'Write unit__inputNode flow for time...\n';
param fn_unit__sourceNode__dt symbolic := "output/unit__inputNode__period__t.csv";
for {i in 1..1 : p_model['solveFirst']}
  {
    printf 'solve,period,time' > fn_unit__sourceNode__dt;  # Print the header on the first solve
	for {(u, source, sink) in process_source_sink_alwaysProcess : (u, source) in process_source && u in process_unit} printf ',%s', u >> fn_unit__sourceNode__dt;
	printf '\n,,' >> fn_unit__sourceNode__dt;
	for {(u, source, sink) in process_source_sink_alwaysProcess : (u, source) in process_source && u in process_unit} printf ',%s', source >> fn_unit__sourceNode__dt;
  }
for {s in solve_current, (d, t) in dt : d in period_realized}
  {
	printf '\n%s,%s,%s', s, d, t >> fn_unit__sourceNode__dt;
    for {(u, source, sink) in process_source_sink_alwaysProcess : (u, source) in process_source && u in process_unit}
      { printf ',%.8g', r_process_source_sink_flow_dt[u, source, sink, d, t] >> fn_unit__sourceNode__dt; }
  } 

printf 'Write connection flow for periods...\n';
param fn_connection__d symbolic := "output/connection__period.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf 'solve,period' > fn_connection__d;  # Print the header on the first solve
	for {(c, input, output) in process_source_sink : c in process_connection && (c, output) in process_sink} printf ',%s', c >> fn_connection__d;
	printf '\n,' >> fn_connection__d;
	for {(c, input, output) in process_source_sink : c in process_connection && (c, output) in process_sink} printf ',%s', input >> fn_connection__d;
	printf '\n,' >> fn_connection__d;
	for {(c, input, output) in process_source_sink : c in process_connection && (c, output) in process_sink} printf ',%s', output >> fn_connection__d;
  }
for {s in solve_current, d in period_realized}
  {
	printf '\n%s,%s', s, d >> fn_connection__d;
    for {(c, input, output) in process_source_sink : c in process_connection && (c, output) in process_sink}
	  { printf ',%.8g', (r_process_source_sink_flow_d[c, c, output, d] - r_process_source_sink_flow_d[c, c, input, d]) / period_share_of_year[d] >> fn_connection__d; }
  }

printf 'Write connection flow for time...\n';
param fn_connection__dt symbolic := "output/connection__period__t.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf 'solve,period,time' > fn_connection__dt;  # Print the header on the first solve
	for {(c, input, output) in process_source_sink : c in process_connection && (c, output) in process_sink} printf ',%s', c >> fn_connection__dt;
	printf '\n,,' >> fn_connection__dt;
	for {(c, input, output) in process_source_sink : c in process_connection && (c, output) in process_sink} printf ',%s', input >> fn_connection__dt;
	printf '\n,,' >> fn_connection__dt;
	for {(c, input, output) in process_source_sink : c in process_connection && (c, output) in process_sink} printf ',%s', output >> fn_connection__dt;
  }
for {s in solve_current, (d, t) in dt : d in period_realized}
  {
	printf '\n%s,%s,%s', s, d, t >> fn_connection__dt;
    for {(c, input, output) in process_source_sink : c in process_connection && (c, output) in process_sink}
	  { printf ',%.8g', r_process_source_sink_flow_dt[c, c, output, d, t] - r_process_source_sink_flow_dt[c, c, input, d, t] >> fn_connection__dt; }
  }

printf 'Write ramps from units over time...\n';
param fn_unit_ramp__sinkNode__dt symbolic := "output/unit_ramp__outputNode__dt.csv";
for {i in 1..1 : p_model['solveFirst']}
  {
    printf 'solve,period,time' > fn_unit_ramp__sinkNode__dt;  # Print the header on the first solve
	for {(u, source, sink) in process_source_sink_alwaysProcess : (u, sink) in process_sink && u in process_unit} printf ',%s', u >> fn_unit_ramp__sinkNode__dt;
	printf '\n,,' >> fn_unit_ramp__sinkNode__dt;
	for {(u, source, sink) in process_source_sink_alwaysProcess : (u, sink) in process_sink && u in process_unit} printf ',%s', sink >> fn_unit_ramp__sinkNode__dt;
  }
for {s in solve_current, (d, t) in dt : d in period_realized}
  {
	printf '\n%s,%s,%s', s, d, t >> fn_unit_ramp__sinkNode__dt;
	for {(u, source, sink) in process_source_sink_alwaysProcess : (u, sink) in process_sink && u in process_unit}
      { printf ',%.8g', r_process_source_sink_ramp_dt[u, source, sink, d, t] >> fn_unit_ramp__sinkNode__dt; }
  } 

param fn_unit_ramp__sourceNode__dt symbolic := "output/unit_ramp__inputNode__dt.csv";
for {i in 1..1 : p_model['solveFirst']}
  {
    printf 'solve,period,time' > fn_unit_ramp__sourceNode__dt;  # Print the header on the first solve
	for {(u, source, sink) in process_source_sink_alwaysProcess : (u, source) in process_source && u in process_unit} printf ',%s', u >> fn_unit_ramp__sourceNode__dt;
	printf '\n,,' >> fn_unit_ramp__sourceNode__dt;
	for {(u, source, sink) in process_source_sink_alwaysProcess : (u, source) in process_source && u in process_unit} printf ',%s', source >> fn_unit_ramp__sourceNode__dt;
  }
for {s in solve_current, (d, t) in dt : d in period_realized}
  {
	printf '\n%s,%s,%s', s, d, t >> fn_unit_ramp__sourceNode__dt;
	for {(u, source, sink) in process_source_sink_alwaysProcess : (u, source) in process_source && u in process_unit}
      { printf ',%.8g', r_process_source_sink_ramp_dt[u, source, sink, d, t] >> fn_unit_ramp__sourceNode__dt; }
  } 

printf 'Write reserve from processes over time...\n';
param fn_process__reserve__upDown__node__dt symbolic := "output/process__reserve__upDown__node__period__t.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf 'solve,period,time' > fn_process__reserve__upDown__node__dt;   # Print the header on the first solve
    for  {(p, r, ud, n) in process_reserve_upDown_node_active} printf ',%s', p >> fn_process__reserve__upDown__node__dt;
	printf '\n,,' >> fn_process__reserve__upDown__node__dt;
    for  {(p, r, ud, n) in process_reserve_upDown_node_active} printf ',%s', r >> fn_process__reserve__upDown__node__dt;
	printf '\n,,' >> fn_process__reserve__upDown__node__dt;
    for  {(p, r, ud, n) in process_reserve_upDown_node_active} printf ',%s', ud >> fn_process__reserve__upDown__node__dt;
	printf '\n,,' >> fn_process__reserve__upDown__node__dt;
    for  {(p, r, ud, n) in process_reserve_upDown_node_active} printf ',%s', n >> fn_process__reserve__upDown__node__dt;
  }
for {s in solve_current, (d, t) in dt : d in period_realized}
  {
	printf '\n%s,%s,%s', s, d, t >> fn_process__reserve__upDown__node__dt;
	for {(p, r, ud, n) in process_reserve_upDown_node_active}
	  { printf ',%.8g', v_reserve[p, r, ud, n, d, t].val >> fn_process__reserve__upDown__node__dt; }
  }

printf 'Write average reserve from processes during periods...\n';
param fn_process__reserve__upDown__node__d symbolic := "output/process__reserve__upDown__node__period_average.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf 'solve,period' > fn_process__reserve__upDown__node__d;   # Print the header on the first solve
    for  {(p, r, ud, n) in process_reserve_upDown_node_active} printf ',%s', p >> fn_process__reserve__upDown__node__d;
	printf '\n,' >> fn_process__reserve__upDown__node__d;
    for  {(p, r, ud, n) in process_reserve_upDown_node_active} printf ',%s', r >> fn_process__reserve__upDown__node__d;
	printf '\n,' >> fn_process__reserve__upDown__node__d;
    for  {(p, r, ud, n) in process_reserve_upDown_node_active} printf ',%s', ud >> fn_process__reserve__upDown__node__d;
	printf '\n,' >> fn_process__reserve__upDown__node__d;
    for  {(p, r, ud, n) in process_reserve_upDown_node_active} printf ',%s', n >> fn_process__reserve__upDown__node__d;
  }
for {s in solve_current, d in period_realized}
  {
	printf '\n%s,%s', s, d >> fn_process__reserve__upDown__node__d;
	for {(p, r, ud, n) in process_reserve_upDown_node_active}
	  { printf ',%.8g', sum{(d, t) in dt} (v_reserve[p, r, ud, n, d, t].val * step_duration[d, t]) / hours_in_period[d] >> fn_process__reserve__upDown__node__d; }
  }

printf 'Write online status of units over time...\n';
param fn_unit_online__dt symbolic := "output/unit_online__period__t.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf 'solve,period,time' > fn_unit_online__dt; 
    for {p in process_unit  : p in process_online}
      { printf ',%s', p >> fn_unit_online__dt; }
  }  # Print the header on the first solve
for {s in solve_current, (d, t) in dt}
  {
    printf '\n%s,%s,%s', s, d, t >> fn_unit_online__dt;
	for {p in process_unit : p in process_online}
	  {
	    printf ',%.8g', v_online_linear[p, d, t].val >> fn_unit_online__dt;
	  }
  }

printf 'Write average unit online during periods...\n';
param fn_unit_online__d symbolic := "output/unit_online__period_average.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf 'solve,period' > fn_unit_online__d; 
    for {p in process_unit  : p in process_online}
      { printf ',%s', p >> fn_unit_online__d; }
  }  # Print the header on the first solve
for {s in solve_current, d in period_realized}
  {
    printf '\n%s,%s', s, d >> fn_unit_online__d;
	for {p in process_unit : p in process_online}
	  {
	    printf ',%.8g', sum{(d, t) in dt} (v_online_linear[p, d, t].val * step_duration[d, t]) / hours_in_period[d] >> fn_unit_online__d;
	  }
  }

printf 'Write unit startups for periods...\n';
param fn_unit_startup__d symbolic := "output/unit_startup__period.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf 'solve,period' > fn_unit_startup__d; 
    for {p in process_unit  : p in process_online}
      { printf ',%s', p >> fn_unit_startup__d; }
  }  # Print the header on the first solve
for {s in solve_current, d in period_realized}
  {
    printf '\n%s,%s', s, d >> fn_unit_startup__d;
	for {p in process_unit : p in process_online}
	  {
	    printf ',%.8g', sum{(d, t) in dt} v_startup_linear[p, d, t].val >> fn_unit_startup__d;
	  }
  }

 
printf 'Write node results for periods...\n';
param fn_node__d symbolic := "output/node__period.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf 'node,solve,period,inflow,"from units","from connections","to units","to connections",' > fn_node__d;
    printf '"state change","self discharge","upward slack","downward slack"\n' >> fn_node__d; }  # Print the header on the first solve
for {n in node, s in solve_current, d in period_realized}
  {
    printf '%s,%s,%s,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g\n'
		, n, s, d
        , (if (n, 'no_inflow') not in node__inflow_method then sum{(d, t) in dt : d in period_realized} pdtNodeInflow[n, d, t])
	    , sum{(p, source, n) in process_source_sink_alwaysProcess : p in process_unit} r_process_source_sink_flow_d[p, source, n, d]
	    , sum{(p, source, n) in process_source_sink_alwaysProcess : p in process_connection} r_process_source_sink_flow_d[p, source, n, d]
  	    , sum{(p, n, sink) in process_source_sink_alwaysProcess : p in process_unit} -r_process_source_sink_flow_d[p, n, sink, d]
  	    , sum{(p, n, sink) in process_source_sink_alwaysProcess : p in process_connection} -r_process_source_sink_flow_d[p, n, sink, d]
	    , (if n in nodeState then r_nodeState_change_d[n, d] else 0)
        , (if n in nodeSelfDischarge then r_selfDischargeLoss_d[n, d] else 0)
	    , sum{ud in upDown : ud = 'up' && n in nodeBalance} r_penalty_nodeState_upDown_d[n, ud, d]
	    , sum{ud in upDown : ud = 'down' && n in nodeBalance} -r_penalty_nodeState_upDown_d[n, ud, d]
	  >> fn_node__d;
  }

printf 'Write process CO2 results for periods...\n';
param fn_process_co2__d symbolic := "output/process__period_co2.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf 'process,solve,period,"CO2 [Mt]"\n' > fn_process_co2__d; }  # Print the header on the first solve 
for {p in process_co2, s in solve_current, d in period_realized}
  {
    printf '%s,%s,%s,%.8g\n'
		, p, s, d
        , sum{(p, c, n) in process__commodity__node_co2} r_process_emissions_co2_d[p, c, n, d] / 1000000
	  >> fn_process_co2__d;
  }

printf 'Write node results for time...\n';
param fn_node__dt symbolic := "output/node__period__t.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf 'node,solve,period,time,inflow,"from units","from connections","to units","to connections",' > fn_node__dt;
    printf '"state","self discharge","upward slack","downward slack"\n' >> fn_node__dt; }  # Print the header on the first solve
for {n in node, s in solve_current, (d, t) in dt : d in period_realized}
  {
    printf '%s,%s,%s,%s,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g\n'
		, n, s, d, t
        , (if (n, 'no_inflow') not in node__inflow_method then pdtNodeInflow[n, d, t])
	    , sum{(p, source, n) in process_source_sink_alwaysProcess : p in process_unit} r_process_source_sink_flow_dt[p, source, n, d, t]
	    , sum{(p, source, n) in process_source_sink_alwaysProcess : p in process_connection} r_process_source_sink_flow_dt[p, source, n, d, t]
  	    , sum{(p, n, sink) in process_source_sink_alwaysProcess : p in process_unit} -r_process_source_sink_flow_dt[p, n, sink, d, t]
  	    , sum{(p, n, sink) in process_source_sink_alwaysProcess : p in process_connection} -r_process_source_sink_flow_dt[p, n, sink, d, t]
	    , (if n in nodeState then v_state[n, d, t].val else 0)
        , (if n in nodeSelfDischarge then r_selfDischargeLoss_dt[n, d, t] else 0)
	    , (if n in nodeBalance then vq_state_up[n, d, t].val else 0)
	    , (if n in nodeBalance then -vq_state_down[n, d, t].val else 0)
	  >> fn_node__dt;
  }

printf 'Write nodal prices for time...\n';
param fn_nodal_prices__dt symbolic := "output/node_prices__period__t.csv";
for {i in 1..1 : p_model['solveFirst']}
  {
    printf 'solve,period,time' > fn_nodal_prices__dt;
    for {n in nodeBalance}
      { printf ',%s', n >> fn_nodal_prices__dt; }
  }
for {s in solve_current, (d, t, t_previous, t_previous_within_block, d_previous, t_previous_within_solve) in dtttdt : d in period_realized}
  {
    printf '\n%s,%s,%s', s, d, t >> fn_nodal_prices__dt;
    for {n in nodeBalance}
	  {
	    printf ',%8g', -nodeBalance_eq[n, d, t, t_previous, t_previous_within_block, d_previous, t_previous_within_solve].dual / p_discount_with_perpetuity_operations[d] * period_share_of_year[d] >> fn_nodal_prices__dt;
      }
  }

printf 'Write node state for time..\n';
param fn_nodeState__dt symbolic := "output/node_state__period__t.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf 'solve,period,time' > fn_nodeState__dt;
    for {n in nodeState}
      { printf ',%s', n >> fn_nodeState__dt; }
  }
for {s in solve_current, (d, t) in dt : d in period_realized}
  { printf '\n%s,%s,%s', s, d, t >> fn_nodeState__dt;
    for {n in nodeState} 
      {
	    printf ',%.8g', v_state[n, d, t].val >> fn_nodeState__dt;
      }
  }

printf 'Write marginal value for investment entities...\n';
param fn_unit_invested_marginal symbolic := "output/unit_invest_marginal__period.csv";
for {i in 1..1 : p_model['solveFirst']} 
  { printf 'solve,period' > fn_unit_invested_marginal;
    for {e in entityInvest : e in process_unit}
	  { printf ',%s', e >> fn_unit_invested_marginal; }
  }
for {s in solve_current, d in period_invest}
  { printf '\n%s,%s', s, d >> fn_unit_invested_marginal;
    for {e in entityInvest : e in process_unit} 
      {
	    printf ',%.8g', (if (e, d) in ed_invest then v_invest[e, d].dual) >> fn_unit_invested_marginal;
      }
  }

param fn_connection_invested_marginal symbolic := "output/connection_invest_marginal__period.csv";
for {i in 1..1 : p_model['solveFirst']} 
  { printf 'solve,period' > fn_connection_invested_marginal;
    for {e in entityInvest : e in process_connection}
	  { printf ',%s', e >> fn_connection_invested_marginal; }
  }
for {s in solve_current, d in period_invest}
  { printf '\n%s,%s', s, d >> fn_connection_invested_marginal;
    for {e in entityInvest : e in process_connection} 
      {
	    printf ',%.8g', (if (e, d) in ed_invest then v_invest[e, d].dual) >> fn_connection_invested_marginal;
      }
  }

param fn_node_invested_marginal symbolic := "output/node_invest_marginal__period.csv";
for {i in 1..1 : p_model['solveFirst']} 
  { printf 'solve,period' > fn_node_invested_marginal;
    for {e in entityInvest : e in node}
	  { printf ',%s', e >> fn_node_invested_marginal; }
  }
for {s in solve_current, d in period_invest}
  { printf '\n%s,%s', s, d >> fn_node_invested_marginal;
    for {e in entityInvest : e in node} 
      {
	    printf ',%.8g', (if (e, d) in ed_invest then v_invest[e, d].dual) >> fn_node_invested_marginal;
      }
  }


param r_node_ramproom_units_up_dt{n in nodeBalance, (d, t) in dt} := 
          + sum{(u, source, n) in process_source_sink_alwaysProcess : u in process_unit && u not in process_VRE} ( 
              + p_process_sink_coefficient[u, n]
                * ( if u in process_online 
			  	    then v_online_linear[u, d, t] * p_entity_unitsize[u]
			  		else entity_all_capacity[u, d]
		          )
			  - r_process_source_sink_flow_dt[u, source, n, d, t]
			  - sum{(u, r, ud, n) in process_reserve_upDown_node_active : ud = 'up'}
				     v_reserve[u, r, ud, n, d, t].val
			)
		  + sum{(u, n, sink) in process_source_sink_alwaysProcess : u in process_unit && u not in process_VRE} ( 
			  + r_process_source_sink_flow_dt[u, n, sink, d, t]
			  - sum{(u, r, ud, n) in process_reserve_upDown_node_active : ud = 'up'}
				     v_reserve[u, r, ud, n, d, t].val
			);

param r_node_ramproom_units_down_dt{n in nodeBalance, (d, t) in dt} := 
          - sum{(u, source, n) in process_source_sink_alwaysProcess : u in process_unit && u not in process_VRE} ( 
			  + r_process_source_sink_flow_dt[u, source, n, d, t]
			  - sum{(u, r, ud, n) in process_reserve_upDown_node_active : ud = 'down'}
				     v_reserve[u, r, ud, n, d, t].val
			)
		  - sum{(u, n, sink) in process_source_sink_alwaysProcess : u in process_unit && u not in process_VRE} ( 
              + p_process_source_coefficient[u, n]
                * ( if u in process_online 
			  	    then v_online_linear[u, d, t] * p_process[u, 'min_load'] * p_entity_unitsize[u]
					else entity_all_capacity[u, d]
		          )
			  - r_process_source_sink_flow_dt[u, n, sink, d, t]
			  - sum{(u, r, ud, n) in process_reserve_upDown_node_active : ud = 'down'}
				     v_reserve[u, r, ud, n, d, t].val
			);  

param r_node_ramproom_VRE_up_dt{n in nodeBalance, (d, t) in dt} := 
          + r_node_ramproom_units_up_dt[n, d, t]
		  + sum{(u, source, n) in process_source_sink_alwaysProcess : u in process_unit && u in process_VRE} ( 
              + p_process_sink_coefficient[u, n]
                * ( if u in process_online 
			  	    then v_online_linear[u, d, t] * p_entity_unitsize[u]
			  		else entity_all_capacity[u, d]
		          )
				* sum{(u, source, n, f, 'upper_limit') in process__source__sink__profile__profile_method} pt_profile[f, t]
			  - r_process_source_sink_flow_dt[u, source, n, d, t]
			  - sum{(u, r, ud, n) in process_reserve_upDown_node_active : ud = 'up'}
				     v_reserve[u, r, ud, n, d, t].val
			)
		  + sum{(u, n, sink) in process_source_sink_alwaysProcess : u in process_unit && u in process_VRE} ( 
			  + r_process_source_sink_flow_dt[u, n, sink, d, t]
			  - sum{(u, r, ud, n) in process_reserve_upDown_node_active : ud = 'up'}
				     v_reserve[u, r, ud, n, d, t].val
			);

param r_node_ramproom_VRE_down_dt{n in nodeBalance, (d, t) in dt} := 
          + r_node_ramproom_units_down_dt[n, d, t]
          - sum{(u, source, n) in process_source_sink_alwaysProcess : u in process_unit && u in process_VRE} ( 
			  + r_process_source_sink_flow_dt[u, source, n, d, t]
			  - sum{(u, r, ud, n) in process_reserve_upDown_node_active : ud = 'down'}
				     v_reserve[u, r, ud, n, d, t].val
			)
		  - sum{(u, n, sink) in process_source_sink_alwaysProcess : u in process_unit && u in process_VRE} ( 
              + p_process_source_coefficient[u, n]
                * ( if u in process_online 
			  	    then v_online_linear[u, d, t] * p_process[u, 'min_load'] * p_entity_unitsize[u]
					else entity_all_capacity[u, d]
		          )
				* sum{(u, n, sink, f, 'upper_limit') in process__source__sink__profile__profile_method} pt_profile[f, t]
			  - r_process_source_sink_flow_dt[u, n, sink, d, t]
			  - sum{(u, r, ud, n) in process_reserve_upDown_node_active : ud = 'down'}
				     v_reserve[u, r, ud, n, d, t].val
			);  

param r_node_ramproom_connections_up_dt{n in nodeBalance, (d, t) in dt} :=  
          + r_node_ramproom_VRE_up_dt[n, d, t]
          + sum{(u, source, n) in process_source_sink_alwaysProcess : (u, n) in process_sink && u in process_connection && u not in process_VRE} ( 
              + p_process_sink_coefficient[u, n]
                * ( if u in process_online 
			  	    then v_online_linear[u, d, t] * p_entity_unitsize[u]
			  		else entity_all_capacity[u, d]
		          )
			  - r_process_source_sink_flow_dt[u, source, n, d, t]
			  - sum{(u, r, ud, n) in process_reserve_upDown_node_active : ud = 'up'}
				     v_reserve[u, r, ud, n, d, t].val
			)
		  + sum{(u, n, sink) in process_source_sink_alwaysProcess : u in process_connection && u not in process_VRE} ( 
			  + r_process_source_sink_flow_dt[u, n, sink, d, t]
			  - sum{(u, r, ud, n) in process_reserve_upDown_node_active : ud = 'up'}
				     v_reserve[u, r, ud, n, d, t].val
			);

param r_node_ramproom_connections_down_dt{n in nodeBalance, (d, t) in dt} := 
          + r_node_ramproom_VRE_down_dt[n, d, t]
          - sum{(u, source, n) in process_source_sink_alwaysProcess : (u, n) in process_sink && u in process_connection && u not in process_VRE && u not in process_isNodeSink_yes2way} ( 
			  + r_process_source_sink_flow_dt[u, source, n, d, t]
			  - sum{(u, r, ud, n) in process_reserve_upDown_node_active : ud = 'down'}
				     v_reserve[u, r, ud, n, d, t].val
			)
          - sum{(u, source, n) in process_source_sink_alwaysProcess : u in process_connection && u not in process_VRE && u in process_isNodeSink_yes2way} ( 
              + p_process_sink_coefficient[u, n]
                * ( if u in process_online 
			  	    then v_online_linear[u, d, t] * p_process[u, 'min_load'] * p_entity_unitsize[u]
					else entity_all_capacity[u, d]
		          )
			  + r_process_source_sink_flow_dt[u, source, n, d, t]
			  - sum{(u, r, ud, n) in process_reserve_upDown_node_active : ud = 'down'}
				     v_reserve[u, r, ud, n, d, t].val
			)
		  - sum{(u, n, sink) in process_source_sink_alwaysProcess : (u, n) in process_source && u in process_connection && u not in process_VRE && u not in process_isNodeSink_yes2way} ( 
              + p_process_source_coefficient[u, n]
                * ( if u in process_online 
			  	    then v_online_linear[u, d, t] * p_process[u, 'min_load'] * p_entity_unitsize[u]
					else entity_all_capacity[u, d]
		          )
			  - r_process_source_sink_flow_dt[u, n, sink, d, t]
			  - sum{(u, r, ud, n) in process_reserve_upDown_node_active : ud = 'down'}
				     v_reserve[u, r, ud, n, d, t].val
			)
;  

printf 'Write node ramps for time...\n';
param fn_node_ramp__dt symbolic := "output/node_ramp__period__t.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf 'node,solve,period,time,"ramp","+connections_up","+VRE_up","units_up",' > fn_node_ramp__dt;
    printf '"units_down","+VRE_down","+connections_down"' >> fn_node_ramp__dt; }  # Print the header on the first solve
for {n in nodeBalance, s in solve_current, (d, t) in dt : d in period_realized}
  {
    printf '\n%s,%s,%s,%s,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g'
		, n, s, d, t
		, r_node_ramp_dt[n, d, t]
		, r_node_ramproom_connections_up_dt[n, d, t]
		, r_node_ramproom_VRE_up_dt[n, d, t]
        , r_node_ramproom_units_up_dt[n, d, t]
		, r_node_ramproom_units_down_dt[n, d, t]
		, r_node_ramproom_VRE_down_dt[n, d, t]
		, r_node_ramproom_connections_down_dt[n, d, t]
	  >> fn_node_ramp__dt;
  }

printf 'Write group inertia over time...\n';
param fn_group_inertia__dt symbolic := "output/group_inertia__period__t.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf 'solve,period,time' > fn_group_inertia__dt;
    for {g in groupInertia}
	  { printf ',%s', g >> fn_group_inertia__dt; }
  }
for {s in solve_current, (d, t) in dt : d in period_realized}
  {
    printf '\n%s,%s,%s', s, d, t >> fn_group_inertia__dt;
	for {g in groupInertia}
	  { printf ',%.8g' 
		  , + sum {(p, source, sink) in process_source_sink : (p, source) in process_source && (g, source) in group_node && p_process_source[p, source, 'inertia_constant']} 
              ( + (if p in process_online then v_online_linear[p, d, t]) 
	            + (if p not in process_online then v_flow[p, source, sink, d, t])
	          ) * p_process_source[p, source, 'inertia_constant']
            + sum {(p, source, sink) in process_source_sink : (p, sink) in process_sink && (g, sink) in group_node && p_process_sink[p, sink, 'inertia_constant']} 
              ( + (if p in process_online then v_online_linear[p, d, t]) 
	            + (if p not in process_online then v_flow[p, source, sink, d, t])
              ) * p_process_sink[p, sink, 'inertia_constant']
  		  >> fn_group_inertia__dt;
	  }
  }

printf 'Write reserve slack variables over time...\n';
param fn_group_reserve_slack__dt symbolic := "output/slack__reserve__upDown__group__period__t.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf 'solve,period,time' > fn_group_reserve_slack__dt; 
    for {(r, ud, g) in reserve__upDown__group}
      { printf ',%s', r >> fn_group_reserve_slack__dt; }
    printf '\n,,' >> fn_group_reserve_slack__dt;
    for {(r, ud, g) in reserve__upDown__group}
      { printf ',%s', ud >> fn_group_reserve_slack__dt; }
    printf '\n,,' >> fn_group_reserve_slack__dt;
    for {(r, ud, g) in reserve__upDown__group}
      { printf ',%s', g >> fn_group_reserve_slack__dt; }
  }
for {s in solve_current, (d, t) in dt : d in period_realized}
  {
    printf '\n%s,%s,%s', s, d, t >> fn_group_reserve_slack__dt;
    for {(r, ud, g) in reserve__upDown__group}
      {
        printf ',%.8g', vq_reserve[r, ud, g, d, t].val
		    >> fn_group_reserve_slack__dt;
      }
  }

printf 'Write non-synchronous slack variables over time...\n';
param fn_group_nonsync_slack__dt symbolic := "output/slack__nonsync_group__period__t.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf 'solve,period,time' > fn_group_nonsync_slack__dt; 
    for {g in groupNonSync}
      { printf ',%s', g >> fn_group_nonsync_slack__dt; }
  }
for {s in solve_current, (d, t) in dt : d in period_realized}
  {
    for {g in groupNonSync}
      {
        printf '\n%s,%s,%s,%.8g'
	        , s, d, t
		    , vq_non_synchronous[g, d, t].val
		    >> fn_group_nonsync_slack__dt;
      }
  }

printf 'Write inertia slack variables over time...\n';
param fn_group_inertia_slack__dt symbolic := "output/slack__inertia_group__period__t.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf 'solve,period,time' > fn_group_inertia_slack__dt; 
    for {g in groupInertia}
      { printf ',%s', g >> fn_group_inertia_slack__dt; }
  }
for {s in solve_current, (d, t) in dt : d in period_realized}
  {
    for {g in groupInertia}
      {
        printf '\n%s,%s,%s,%.8g'
	        , s, d, t
		    , vq_inertia[g, d, t].val
		    >> fn_group_inertia_slack__dt;
      }
  }

printf 'Write capacity margin slack for periods with investments...\n';
param fn_group_capmargin_slack__d symbolic := "output/slack__capacity_margin__period.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf 'solve,period' > fn_group_capmargin_slack__d; 
    for {g in groupCapacityMargin}
      { printf ',%s', g >> fn_group_capmargin_slack__d; }
  }
for {s in solve_current, d in period_invest : d in period_realized}
  {
    for {g in groupCapacityMargin}
      {
        printf '\n%s,%s,%.8g'
	        , s, d
		    , vq_capacity_margin[g, d].val
		    >> fn_group_capmargin_slack__d;
      }
  }

param resultFile symbolic := "output/result.csv";

printf 'Upward slack for node balance\n' > resultFile;
for {n in nodeBalance, (d, t) in dt}
  {
    printf '%s,%s,%s,%.8g\n', n, d, t, vq_state_up[n, d, t].val >> resultFile;
  }

printf '\nDownward slack for node balance\n' >> resultFile;
for {n in nodeBalance, (d, t) in dt}
  {
    printf '%s,%s,%s,%.8g\n', n, d, t, vq_state_down[n, d, t].val >> resultFile;
  }

printf '\nReserve upward slack variable\n' >> resultFile;
for {(r, ud, ng) in reserve__upDown__group, (d, t) in dt}
  {
    printf '%s,%s,%s,%s,%s,%.8g\n', r, ud, ng, d, t, vq_reserve[r, ud, ng, d, t].val >> resultFile;
  }

printf '\nInvestments\n' >> resultFile;
for {(e, d_invest) in ed_invest} {
  printf '%s,%s,%.8g\n', e, d_invest, v_invest[e, d_invest].val * p_entity_unitsize[e] >> resultFile;
}

printf '\nDivestments\n' >> resultFile;
for {(e, d_invest) in ed_divest} {
  printf '%s,%s,%.8g\n', e, d_invest, v_divest[e, d_invest].val * p_entity_unitsize[e] >> resultFile;
}


### UNIT TESTS ###
param unitTestFile symbolic := "tests/unitTests.txt";
printf (if sum{d in debug} 1 then '%s --- ' else ''), time2str(gmtime(), "%FT%TZ") > unitTestFile;
for {d in debug} {
  printf '%s  ', d >> unitTestFile;
}
printf (if sum{d in debug} 1 then '\n\n' else '') >> unitTestFile;

## Objective test
printf (if (sum{d in debug} 1 && total_cost.val <> d_obj) 
        then 'Objective value test fails. Model value: %.8g, test value: %.8g\n' else ''), total_cost.val, d_obj >> unitTestFile;

## Testing flows from and to node
for {n in node : 'method_1way_1var' in debug || 'mini_system' in debug} {
  printf 'Testing incoming flows of node %s\n', n >> unitTestFile;
  for {(p, source, n, d, t) in peedt} {
    printf (if v_flow[p, source, n, d, t].val <> d_flow[p, source, n, d, t] 
	        then 'Test fails at %s, %s, %s, %s, %s, model value: %.8g, test value: %.8g\n' else ''),
			    p, source, n, d, t, v_flow[p, source, n, d, t].val, d_flow[p, source, n, d, t] >> unitTestFile;
  }
  printf 'Testing outgoing flows of node %s\n', n >> unitTestFile;
  for {(p, n, sink, d, t) in peedt : sum{(p, m) in process_method : m = 'method_1var' || m = 'method_2way_2var'} 1 } {
    printf (if -v_flow[p, n, sink, d, t].val / ptProcess[p, 'efficiency', t] <> d_flow_1_or_2_variable[p, n, sink, d, t]
	        then 'Test fails at %s, %s, %s, %s, %s, model value: %.8g, test value: %.8g\n' else ''),
	            p, n, sink, d, t, -v_flow[p, n, sink, d, t].val / ptProcess[p, 'efficiency', t], d_flow_1_or_2_variable[p, n, sink, d, t] >> unitTestFile;
  }
  for {(p, n, sink, d, t) in peedt : sum{(p, m) in process_method : m in method && (m <> 'method_1var' || m <> 'method_2way_2var')} 1 } {
    printf (if -v_flow[p, n, sink, d, t].val <> d_flow[p, n, sink, d, t] 
	        then 'Test fails at %s, %s, %s, %s, %s, model value: %.8g, test value: %.8g\n' else ''),
	            p, n, sink, d, t, -v_flow[p, n, sink, d, t].val, d_flow[p, n, sink, d, t] >> unitTestFile;
  }
  printf '\n' >> unitTestFile;
}  

## Testing reserves
for {(p, r, ud, n, d, t) in prundt} {
  printf (if v_reserve[p, r, ud, n, d, t].val <> d_reserve_upDown_node[p, r, ud, n, d, t]
          then 'Reserve test fails at %s, %s, %s, %s, %s, %s. Model value: %.8g, test value: %.8g\n' else ''),
		      p, r, ud, n, d, t, v_reserve[p, r, ud, n, d, t].val, d_reserve_upDown_node[p, r, ud, n, d, t] >> unitTestFile;
}
for {(r, ud, ng) in reserve__upDown__group, (d, t) in dt} {
  printf (if vq_reserve[r, ud, ng, d, t].val <> dq_reserve[r, ud, ng, d, t]
          then 'Reserve slack variable test fails at %s, %s, %s, %s, %s. Model value: %.8g, test value: %.8g\n' else ''),
		      r, ud, ng, d, t, vq_reserve[r, ud, ng, d, t].val, dq_reserve[r, ud, ng, d, t] >> unitTestFile;
}

## Testing investments
#for {(p, n, d_invest) in ped_invest : 'invest_source_to_sink' in debug} {
#  printf 'Testing investment decisions of %s %s %s\n', p, n, d_invest >> unitTestFile;
#  printf (if v_flowInvest[p, n, d_invest].val <> d_flowInvest[p, n, d_invest]
#          then 'Test fails at %s, %s, %s, model value: %.8g, test value: %.8g\n' else ''),
#		      p, n, d_invest, v_flowInvest[p, n, d_invest].val, d_flowInvest[p, n, d_invest] >> unitTestFile;
#}
printf (if sum{d in debug} 1 then '\n\n' else '') >> unitTestFile;	  

#display {(p, source, sink) in process_source_sink_alwaysProcess, (d, t) in dt : (d, t) in test_dt}: r_process_source_sink_flow_dt[p, source, sink, d, t];
#display {p in process, (d, t) in dt : (d, t) in test_dt}: r_cost_process_variable_cost_dt[p, d, t];
#display {(p, source, sink, d, t) in peedt : (d, t) in test_dt}: v_flow[p, source, sink, d, t].val;
#display {(p, source, sink, d, t) in peedt : (d, t) in test_dt}: v_flow[p, source, sink, d, t].ub;
#display {p in process_online, (d, t) in dt : (d, t) in test_dt} : v_online_linear[p, d, t].val;
#display {n in nodeState, (d, t) in dt : (d, t) in test_dt}: v_state[n, d, t].val;
#display {(p, r, ud, n, d, t) in prundt : (d, t) in test_dt}: v_reserve[p, r, ud, n, d, t].val;
#display {(r, ud, ng) in reserve__upDown__group, (d, t) in test_dt}: vq_reserve[r, ud, ng, d, t].val;
#display {n in nodeBalance, (d, t) in dt : (d, t) in test_dt}: vq_state_up[n, d, t].val;
#display {n in nodeBalance, (d, t) in dt : (d, t) in test_dt}: vq_state_down[n, d, t].val;
#display {g in groupInertia, (d, t) in dt : (d, t) in test_dt}: inertia_constraint[g, d, t].dual;
#display {n in nodeBalance, (d, t, t_previous, t_previous_within_block, d_previous, t_previous_within_solve) in dtttdt : (d, t) in test_dt}: -nodeBalance_eq[n, d, t, t_previous, t_previous_within_block, d_previous, t_previous_within_solve].dual / p_discount_with_perpetuity_operations[d] * period_share_of_year[d];
#display {(p, source, sink) in process_source_sink, (d, t) in dt : (d, t) in test_dt && (p, sink) in process_sink}: maxToSink[p, source, sink, d, t].ub;
#display {(p, sink, source) in process_sink_toSource, (d, t) in dt : (d, t) in test_dt}: maxToSource[p, sink, source, d, t].ub;
#display {(p, m) in process_method, (d, t) in dt : (d, t) in test_dt && m in method_indirect} conversion_indirect[p, m, d, t].ub;
#display {(p, source, sink, f, m) in process__source__sink__profile__profile_method, (d, t) in dt : (d, t) in test_dt && m = 'lower_limit'}: profile_flow_lower_limit[p, source, sink, f, d, t].dual;
display v_invest, v_divest;
#display {(e, d) in ed_invest} : v_invest[e, d].dual;
end;
