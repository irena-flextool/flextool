# © International Renewable Energy Agency 2018-2022

#The FlexTool is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License
#as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

#The FlexTool is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
#without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

#You should have received a copy of the GNU Lesser General Public License along with the FlexTool.  
#If not, see <https://www.gnu.org/licenses/>.

#Author: Juha Kiviluoma (2017-2022), VTT Technical Research Centre of Finland

#########################
# Fundamental sets of the model
set entity 'e - contains both nodes and processes';
set process 'p - Particular activity that transfers, converts or stores commodities' within entity;
set processUnit 'Unit processes' within process;
set processTransfer 'Transfer processes' within process;
set node 'n - Any location where a balance needs to be maintained' within entity;
set group 'g - Any group of entities that have a set of common constraints';
set commodity 'c - Stuff that is being processed';
set reserve__upDown__group dimen 3;
set reserve__upDown__group__method dimen 4;
set reserve 'r - Categories for the reservation of capacity_existing' := setof {(r, ud, ng, r_m) in reserve__upDown__group__method} (r);
set period_time '(d, t) - Time steps in the time periods of the timelines in use' dimen 2;
set solve_period '(solve, d) - Time periods in the solves in order to extract periods that can be found in data' dimen 2;
set periodAll 'd - Time periods in data (including those currently in use)' := setof {(s, d) in solve_period} (d);
set period 'd - Time periods in the current solve' := setof {(d, t) in period_time} (d);
set time 't - Time steps in the current timelines'; 
set method 'm - Type of process that transfers, converts or stores commodities';
set upDown 'upward and downward directions for some variables';
set ct_method;
set startup_method;
set reserve_method;
set ramp_method;
set ramp_limit_method within ramp_method;
set ramp_cost_method within ramp_method;
set profile;
set profile_method;
set debug 'flags to output debugging and test results';

set constraint 'user defined greater than, less than or equality constraints between inputs and outputs';
set sense 'sense of user defined constraints';
set sense_greater_than within sense;
set sense_less_than within sense;
set sense_equal within sense;

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
set method_area;


set invest_method 'methods available for investments';
set invest_method_not_allowed 'method for denying investments' within invest_method;
set entity__invest_method 'the investment method applied to an entity' dimen 2 within {entity, invest_method};
set entityInvest := setof {(e, m) in entity__invest_method : m not in invest_method_not_allowed} (e);
set nodeBalance 'nodes that maintain a node balance' within node;
set nodeState 'nodes that have a state' within node;
set inflow_method 'method for scaling the inflow';
set node__inflow_method 'method for scaling the inflow applied to a node' within {node, inflow_method};
set group_node 'member nodes of a particular group' dimen 2 within {group, node};
set group_process 'member processes of a particular group' dimen 2 within {group, process};
set group_process_node 'process__nodes of a particular group' dimen 3 within {group, process, node};
set group_entity := group_process union group_node;
set groupInertia 'node groups with an inertia constraint' within group;
set groupNonSync 'node groups with a non-synchronous constraint' within group;
set groupCapacityMargin 'node groups with a capacity margin' within group;
set process_unit 'processes that are unit' within process;
set process_connection 'processes that are connections' within process;
set process_ct_method dimen 2 within {process, ct_method};
set process_startup_method dimen 2 within {process, startup_method};
set process_node_ramp_method dimen 3 within {process, node, ramp_method};
set methods dimen 3; 
set process__profile__profile_method dimen 3 within {process, profile, profile_method};
set process__node__profile__profile_method dimen 4 within {process, node, profile, profile_method};
set process_source dimen 2 within {process, entity};
set process_sink dimen 2 within {process, entity};
set process_ct_startup_method := 
    { p in process, m1 in ct_method, m2 in startup_method, m in method
	    : (m1, m2, m) in methods
	    && (p, m1) in process_ct_method
	    && (p, m2) in process_startup_method 
	};
set process_method := setof {(p, m1, m2, m) in process_ct_startup_method} (p, m);
set process_source_toProcess := 
    { p in process, source in node, p2 in process 
	    :  p = p2 
	    && (p, source) in process_source 
	    && (p2, source) in process_source 
	    && sum{(p, m) in process_method : m in method_indirect} 1
	};
set process_process_toSink := 
    { p in process, p2 in process, sink in node 
	    :  p = p2 
	    && (p, sink) in process_sink 
	    && (p2, sink) in process_sink 
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
    { p in process, p2 in process, source in node 
	    :  p = p2 
	    && (p, source) in process_source
	    && (p2, source) in process_source
	    && sum{(p, m) in process_method : m in method_2way_nvar} 1
	};
set process_source_toSink := 
    { p in process, source in node, sink in node
	    :  (p, source) in process_source
	    && (p, sink) in process_sink
        && sum{(p, m) in process_method : m in method_direct} 1
	};
set process_sink_toSource := 
	{ p in process, sink in node, source in node
	    :  (p, source) in process_source
	    && (p, sink) in process_sink
	    && sum{(p, m) in process_method : m in method_2way_2var} 1
	};
set process__profileProcess__toSink__profile__profile_method :=
    { p in process, p2 in process, sink in node, f in profile, m in profile_method
	    :  p = p2
		&& (p, sink) in process_sink
		&& (p2, sink, f, m) in process__node__profile__profile_method
	};
set process__profileProcess__toSink := setof {(p, p2, sink, f, m) in process__profileProcess__toSink__profile__profile_method} (p, p2, sink);
set process__source__toProfileProcess__profile__profile_method :=
    { p in process, source in node, p2 in process, f in profile, m in profile_method
	    :  p = p2
		&& (p, source) in process_source
		&& (p2, source, f, m) in process__node__profile__profile_method
	};
set process__source__toProfileProcess := setof {(p, source, p2, f, m) in process__source__toProfileProcess__profile__profile_method} (p, source, p2);
set process_source_sink := 
    process_source_toSink union 
	process_sink_toSource union   
	process_source_toProcess union 
	process_process_toSink union 
	process_sink_toProcess union   # Add the 'wrong' direction in 2-way processes with multiple inputs/outputs
	process_process_toSource union # Add the 'wrong' direction in 2-way processes with multiple inputs/outputs
	process__profileProcess__toSink union   # Add profile based inputs to process
	process__source__toProfileProcess;	   # Add profile based inputs to process	
set process_online 'processes with an online status' := setof {(p, m) in process_method : m in method_LP} p;
set process__source__sink__profile__profile_method_connection :=
    { (p, sink, source) in process_source_sink, f in profile, m in profile_method
	    : (p, f, m) in process__profile__profile_method
	};
set process__source__sink__profile__profile_method :=
    process__profileProcess__toSink__profile__profile_method union
	process__source__toProfileProcess__profile__profile_method union
	process__source__sink__profile__profile_method_connection
;

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

set process_reserve_upDown_node dimen 4;
set process_node_constraint dimen 3 within {process, node, constraint};
set constraint__sense dimen 2 within {constraint, sense};
set commodity_node dimen 2 within {commodity, node}; 
set connection_variable_cost dimen 1 within process;
set unit__sourceNode_variable_cost dimen 2 within process_source;
set unit__sinkNode_variable_cost dimen 2 within process_sink;

set dt dimen 2 within period_time;
set dttt dimen 4;
set period_invest dimen 1 within period;
set period_realized dimen 1 within period;
set peedt := {(p, source, sink) in process_source_sink, (d, t) in period_time};
set prundt := {(p, r, ud, n) in process_reserve_upDown_node, (d, t) in period_time};

set startTime dimen 1 within time;
set startNext dimen 1 within time;
param startNext_index := sum{t in time, t_startNext in startNext : t <= t_startNext} 1;
set modelParam;
param p_model {modelParam};

set commodity__param__period dimen 3 within {commodity, commodityPeriodParam, periodAll};
param p_commodity {c in commodity, commodityParam};
param pd_commodity {c in commodity, commodityPeriodParam, d in periodAll} default 0;
param pdCommodity {c in commodity, param in commodityPeriodParam, d in period} := 
        + if (c, param, d) in commodity__param__period
		  then pd_commodity[c, param, d]
		  else p_commodity[c, param];

param p_group__process {g in group, p in process, groupParam};

set group__param dimen 2 within {group, groupParam};
set group__param__period dimen 3 within {group, groupPeriodParam, periodAll};
param p_group {g in group, groupParam} default 0;
param pd_group {g in group, groupPeriodParam, d in periodAll} default 0;
param pdGroup {g in group, param in groupPeriodParam, d in period} :=
        + if (g, param, d) in group__param__period
		  then pd_group[g, param, d]
		  else if (g, param) in group__param 
		  then p_group[g, param]
		  else 0;
		  
set node__param__period dimen 3 within {node, nodePeriodParam, periodAll};
set node__param__time dimen 3 within {node, nodeTimeParam, time};
param p_node {node, nodeParam} default 0;
param pd_node {node, nodePeriodParam, periodAll} default 0;
param pt_node {node, nodeTimeParam, time} default 0;
param pdNode {n in node, param in nodePeriodParam, d in period} :=
        + if (n, param, d) in node__param__period
		  then pd_node[n, param, d]
		  else p_node[n, param];
param ptNode {n in node, param in nodeTimeParam, t in time} :=
        + if (n, param, t) in node__param__time
		  then pt_node[n, param, t]
		  else p_node[n, param];

set process__param dimen 2 within {process, processParam};
set process__param__period dimen 3 within {process, processPeriodParam, periodAll};
set process__param__time dimen 3 within {process, processTimeParam, time};
set process__param_t := setof {(p, param, t) in process__param__time} (p, param);

set connection__param dimen 2 within {process, processParam};
set connection__param__time dimen 3 within {process, sourceSinkTimeParam, time};
set connection__param_t := setof {(connection, param, t) in connection__param__time} (connection, param);
set process__source__param dimen 3 within {process_source, sourceSinkParam};
set process__source__param__time dimen 4 within {process_source, sourceSinkTimeParam, time};
set process__source__param_t := setof {(p, source, param, t) in process__source__param__time} (p, source, param);
set process__sink__param dimen 3 within {process_sink, sourceSinkParam};
set process__sink__param__time dimen 4 within {process_sink, sourceSinkTimeParam, time};
set process__sink__param_t := setof {(p, sink, param, t) in process__sink__param__time} (p, sink, param);

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

set process__source__sink__ramp_method :=
    { (p, source, sink) in process_source_sink, m in ramp_method
	    :  (p, source, m) in process_node_ramp_method
		|| (p, sink, m) in process_node_ramp_method
	};

param p_process {process, processParam} default 0;
param pd_process {process, processPeriodParam, periodAll} default 0;
param pt_process {process, processTimeParam, time} default 0;
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
        + if e in process && p_process[e, 'virtual_unitsize']
          then p_process[e, 'virtual_unitsize'] 
		  else if e in process && p_process[e, 'existing']
			   then p_process[e, 'existing']
			   else if e in process && p_process[e, 'invest_forced']
			        then p_process[e, 'invest_forced']
			        else 1
        + if e in node && p_node[e, 'virtual_unitsize'] 
          then p_node[e, 'virtual_unitsize'] 
		  else if e in node && p_node[e, 'existing']
		       then p_node[e, 'existing']
		       else if e in node && p_node[e, 'invest_forced']
		            then p_node[e, 'invest_forced']
		            else 1;

param p_process_source {(p, source) in process_source, sourceSinkParam} default 0;
param pt_process_source {(p, source) in process_source, sourceSinkTimeParam, time} default 0;
param p_process_sink {(p, sink) in process_sink, sourceSinkParam} default 0;
param pt_process_sink {(p, sink) in process_sink, sourceSinkTimeParam, time} default 0;
param pProcess_source_sink {(p, source, sink, param) in process__source__sink__param} :=
		+ if (p, source, param) in process__source__param
		  then p_process_source[p, source, param]
		  else if (p, sink, param) in process__sink__param
		  then p_process_sink[p, sink, param]
		  else 0;
param ptProcess_source_sink {(p, source, sink, param) in process__source__sink__param_t, t in time} :=
        + if (p, sink, param, t) in process__sink__param__time
		  then pt_process_sink[p, sink, param, t]
          else if (p, source, param, t) in process__source__param__time
		  then pt_process_source[p, source, param, t]
		  else if (p, source, param) in process__source__param
		  then p_process_source[p, source, param]
		  else if (p, sink, param) in process__sink__param
		  then p_process_sink[p, sink, param]
		  else 0;

param p_process_source_coefficient {(p, source) in process_source} := 
    + if (p_process_source[p, source, 'coefficient']) 
	  then p_process_source[p, source, 'coefficient'] 
	  else 1;
param p_process_sink_coefficient {(p, sink) in process_sink} := 
    + if (p_process_sink[p, sink, 'coefficient']) 
	  then p_process_sink[p, sink, 'coefficient'] 
	  else 1;

param pt_profile {profile, time};

set reserve__upDown__group__reserveParam__time dimen 5 within {reserve, upDown, group, reserveTimeParam, time};
param p_reserve_upDown_group {reserve, upDown, group, reserveParam};
param pt_reserve_upDown_group {reserve, upDown, group, reserveTimeParam, time};
param ptReserve_upDown_group {(r, ud, g) in reserve__upDown__group, param in reserveTimeParam, t in time} :=
        + if (r, ud, g, param, t) in reserve__upDown__group__reserveParam__time
		  then pt_reserve_upDown_group[r, ud, g, param, t]
		  else p_reserve_upDown_group[r, ud, g, param];
param p_process_reserve_upDown_node {process, reserve, upDown, node, reserveParam} default 0;

param p_constraint_constant {constraint};
param p_process_node_constraint_coefficient {process, node, constraint};
param penalty_up {n in nodeBalance};
param penalty_down {n in nodeBalance};
param step_duration{(d, t) in dt};
param hours_in_solve := sum {(d, t) in dt} (step_duration[d, t]);
param solve_share_of_year := hours_in_solve / 8760;
param solve_share_of_annual_flow {n in node, d in period : (n, 'scale_to_annual_flow') in node__inflow_method && pdNode[n, 'annual_flow', d]} := 
        abs(sum{(d, t) in dt} (ptNode[n, 'inflow', t])) / pdNode[n, 'annual_flow', d];
param annual_flow_multiplier {n in node, d in period : (n, 'scale_to_annual_flow') in node__inflow_method && pdNode[n, 'annual_flow', d]} := 
        solve_share_of_year / solve_share_of_annual_flow[n, d];
param step_period{(d, t) in dt} := 0;
param ed_entity_annual{e in entityInvest, d in period_invest} :=
        + sum{m in invest_method : (e, m) in entity__invest_method && e in node && m not in invest_method_not_allowed}
          ( + (pdNode[e, 'invest_cost', d] * 1000 * ( pdNode[e, 'interest_rate', d] 
			  / (1 - (1 / (1 + pdNode[e, 'interest_rate', d])^pdNode[e, 'lifetime', d] ) ) ))
			+ pdNode[e, 'fixed_cost', d]
		  )
        + sum{m in invest_method : (e, m) in entity__invest_method && e in process && m not in invest_method_not_allowed}
		  (
            + (pdProcess[e, 'invest_cost', d] * 1000 * ( pdProcess[e, 'interest_rate', d] 
			  / (1 - (1 / (1 + pdProcess[e, 'interest_rate', d])^pdProcess[e, 'lifetime', d] ) ) ))
			+ pdProcess[e, 'fixed_cost', d]
		  )
; 			
#param pt_process_source_sink_variable_cost{(p, source, sink) in process_source_sink_variable_cost, t in time} :=
#        + (if (p, source) in unit__sourceNode_variable_cost then ptProcess_source[p, source, 'variable_cost', t])
#        + (if (p, sink) in unit__sinkNode_variable_cost then ptProcess_sink[p, sink, 'variable_cost', t])
#		+ (if p in connection_variable_cost then ptProcess[p, 'variable_cost', t])
#;

set ed_invest := {e in entityInvest, d in period_invest : ed_entity_annual[e, d]};
set pd_invest := {(p, d) in ed_invest : p in process};
set nd_invest := {(n, d) in ed_invest : n in node};
set ed_divest := ed_invest;

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

set process_reserve_upDown_node_increase_reserve_ratio :=
        {(p, r, ud, n) in process_reserve_upDown_node :
		    p_process_reserve_upDown_node[p, r, ud, n, 'increase_reserve_ratio'] > 0
		};

set group_commodity_node_period_CO2 :=
        {g in group, (c, n) in commodity_node, d in period : 
		    (g, n) in group_node 
			&& p_commodity[c, 'co2_content'] 
			&& pdGroup[g, 'co2_price', d]
		};

set process__sink_nonSync_unit dimen 2 within {process, node};
set process_nonSync_connection dimen 1 within {process};
set process__sink_nonSync :=
        {p in process, sink in node :
		       ( (p, sink) in process_sink && (p, sink) in process__sink_nonSync_unit )
			|| ( (p, sink) in process_sink && p in process_nonSync_connection )
			|| ( (p, sink) in process_source && p in process_nonSync_connection )  
	    };

param p_entity_invested {e in entity : e in entityInvest};
param p_entity_all_existing {e in entity} :=
        + (if e in process then p_process[e, 'existing'])
		+ (if e in process then p_process[e, 'invest_forced'])
        + (if e in node then p_node[e, 'existing'])
		+ (if e in node then p_node[e, 'invest_forced'])
		+ (if not p_model['solveFirst'] && e in entityInvest then p_entity_invested[e])
;

param d_obj default 0;
param d_flow {(p, source, sink, d, t) in peedt} default 0;
param d_flow_1_or_2_variable {(p, source, sink, d, t) in peedt} default 0;
param d_flowInvest {(p, d) in pd_invest} default 0;
param d_reserve_upDown_node {(p, r, ud, n, d, t) in prundt} default 0;
param dq_reserve {(r, ud, ng) in reserve__upDown__group, (d, t) in dt} default 0;

#########################
# Read data
#table data IN 'CSV' '.csv' :  <- [];

# Domain sets
table data IN 'CSV' 'commodity.csv' : commodity <- [commodity];
table data IN 'CSV' 'connection_variable_cost.csv' : connection_variable_cost <- [process];
table data IN 'CSV' 'constraint__sense.csv' : constraint <- [constraint];
table data IN 'CSV' 'debug.csv': debug <- [debug];
table data IN 'CSV' 'entity.csv': entity <- [entity];
table data IN 'CSV' 'group.csv' : group <- [group];
table data IN 'CSV' 'node.csv' : node <- [node];
table data IN 'CSV' 'nodeBalance.csv' : nodeBalance <- [nodeBalance];
table data IN 'CSV' 'nodeState.csv' : nodeState <- [nodeState];
table data IN 'CSV' 'groupInertia.csv' : groupInertia <- [groupInertia];
table data IN 'CSV' 'groupNonSync.csv' : groupNonSync <- [groupNonSync];
table data IN 'CSV' 'groupCapacityMargin.csv' : groupCapacityMargin <- [groupCapacityMargin];
table data IN 'CSV' 'process.csv': process <- [process];
table data IN 'CSV' 'profile.csv': profile <- [profile];
table data IN 'CSV' 'timeline.csv' : time <- [timestep];

# Single dimension membership sets
table data IN 'CSV' 'process_connection.csv': process_connection <- [process_connection];
table data IN 'CSV' 'process_nonSync_connection.csv': process_nonSync_connection <- [process];
table data IN 'CSV' 'process_unit.csv': process_unit <- [process_unit];

# Multi dimension membership sets
table data IN 'CSV' 'commodity__node.csv' : commodity_node <- [commodity,node];
table data IN 'CSV' 'entity__invest_method.csv' : entity__invest_method <- [entity,invest_method];
table data IN 'CSV' 'node__inflow_method.csv' : node__inflow_method <- [node,inflow_method];
table data IN 'CSV' 'group__node.csv' : group_node <- [group,node];
table data IN 'CSV' 'group__process.csv' : group_process <- [group,process];
table data IN 'CSV' 'group__process__node.csv' : group_process_node <- [group,process,node];
table data IN 'CSV' 'p_process_node_constraint_coefficient.csv' : process_node_constraint <- [process, node, constraint];
table data IN 'CSV' 'constraint__sense.csv' : constraint__sense <- [constraint, sense];
table data IN 'CSV' 'p_process.csv' : process__param <- [process, processParam];
table data IN 'CSV' 'pd_node.csv' : node__param__period <- [node, nodeParam, period];
table data IN 'CSV' 'pt_node.csv' : node__param__time <- [node, nodeParam, time];
table data IN 'CSV' 'pd_process.csv' : process__param__period <- [process, processParam, period];
table data IN 'CSV' 'pt_process.csv' : process__param__time <- [process, processParam, time];
table data IN 'CSV' 'p_group.csv' : group__param <- [group, groupParam];
table data IN 'CSV' 'pd_group.csv' : group__param__period <- [group, groupParam, period];
table data IN 'CSV' 'process__ct_method.csv' : process_ct_method <- [process,ct_method];
table data IN 'CSV' 'process__node__ramp_method.csv' : process_node_ramp_method <- [process,node,ramp_method];
table data IN 'CSV' 'process__reserve__upDown__node.csv' : process_reserve_upDown_node <- [process,reserve,upDown,node];
table data IN 'CSV' 'process__sink.csv' : process_sink <- [process,sink];
table data IN 'CSV' 'process__source.csv' : process_source <- [process,source];
table data IN 'CSV' 'process__sink_nonSync_unit.csv' : process__sink_nonSync_unit <- [process,sink];
table data IN 'CSV' 'process__startup_method.csv' : process_startup_method <- [process,startup_method];
table data IN 'CSV' 'process__profile__profile_method.csv' : process__profile__profile_method <- [process,profile,profile_method];
table data IN 'CSV' 'process__node__profile__profile_method.csv' : process__node__profile__profile_method <- [process,node,profile,profile_method];
table data IN 'CSV' 'reserve__upDown__group.csv' : reserve__upDown__group <- [reserve,upDown,group];
table data IN 'CSV' 'reserve__upDown__group__method.csv' : reserve__upDown__group__method <- [reserve,upDown,group,method];
table data IN 'CSV' 'pt_reserve__upDown__group.csv' : reserve__upDown__group__reserveParam__time <- [reserve, upDown, group, reserveParam, time];
table data IN 'CSV' 'timeblocks_in_use.csv' : solve_period <- [solve,period];
table data IN 'CSV' 'p_process_source.csv' : process__source__param <- [process, source, sourceSinkParam];
table data IN 'CSV' 'pt_process_source.csv' : process__source__param__time <- [process, source, sourceSinkTimeParam, time];
table data IN 'CSV' 'p_process_sink.csv' : process__sink__param <- [process, sink, sourceSinkParam];
table data IN 'CSV' 'pt_process_sink.csv' : process__sink__param__time <- [process, sink, sourceSinkTimeParam, time];
table data IN 'CSV' 'pd_commodity.csv' : commodity__param__period <- [commodity, commodityParam, period];

# Parameters for model data
table data IN 'CSV' 'p_commodity.csv' : [commodity, commodityParam], p_commodity;
table data IN 'CSV' 'pd_commodity.csv' : [commodity, commodityParam, period], pd_commodity;
table data IN 'CSV' 'p_group__process.csv' : [group, process, groupParam], p_group__process;
table data IN 'CSV' 'p_group.csv' : [group, groupParam], p_group;
table data IN 'CSV' 'pd_group.csv' : [group, groupParam, period], pd_group;
table data IN 'CSV' 'p_node.csv' : [node, nodeParam], p_node;
table data IN 'CSV' 'pd_node.csv' : [node, nodeParam, period], pd_node;
table data IN 'CSV' 'pt_node.csv' : [node, nodeParam, time], pt_node;
table data IN 'CSV' 'p_process_node_constraint_coefficient.csv' : [process, node, constraint], p_process_node_constraint_coefficient;
table data IN 'CSV' 'p_process__reserve__upDown__node.csv' : [process, reserve, upDown, node, reserveParam], p_process_reserve_upDown_node;
table data IN 'CSV' 'p_process_sink.csv' : [process, sink, sourceSinkParam], p_process_sink;
table data IN 'CSV' 'pt_process_sink.csv' : [process, sink, sourceSinkTimeParam, time], pt_process_sink;
table data IN 'CSV' 'p_process_source.csv' : [process, source, sourceSinkParam], p_process_source;
table data IN 'CSV' 'pt_process_source.csv' : [process, source, sourceSinkTimeParam, time], pt_process_source;
table data IN 'CSV' 'p_constraint_constant.csv' : [constraint], p_constraint_constant;
table data IN 'CSV' 'p_process.csv' : [process, processParam], p_process;
table data IN 'CSV' 'pd_process.csv' : [process, processParam, period], pd_process;
table data IN 'CSV' 'pt_process.csv' : [process, processParam, time], pt_process;
table data IN 'CSV' 'pt_profile.csv' : [profile, time], pt_profile;
table data IN 'CSV' 'p_reserve__upDown__group.csv' : [reserve, upDown, group, reserveParam], p_reserve_upDown_group;
table data IN 'CSV' 'pt_reserve__upDown__group.csv' : [reserve, upDown, group, reserveParam, time], pt_reserve_upDown_group;

# Parameters from the solve loop
table data IN 'CSV' 'steps_in_use.csv' : dt <- [period, step];
table data IN 'CSV' 'steps_in_use.csv' : [period, step], step_duration;
table data IN 'CSV' 'steps_in_timeline.csv' : period_time <- [period,step];
table data IN 'CSV' 'step_previous.csv' : dttt <- [period, time, previous, previous_within_block];
table data IN 'CSV' 'realized_periods_of_current_solve.csv' : period_realized <- [period];
table data IN 'CSV' 'invest_periods_of_current_solve.csv' : period_invest <- [period];
table data IN 'CSV' 'p_model.csv' : [modelParam], p_model;

# After rolling forward the investment model
table data IN 'CSV' 'p_entity_invested.csv' : [entity], p_entity_invested;


#########################
# Variable declarations
var v_flow {(p, source, sink, d, t) in peedt};
var v_ramp {(p, source, sink, d, t) in peedt};
var v_reserve {(p, r, ud, n, d, t) in prundt} >= 0;
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

display method_2way;

#########################
## Data checks 
printf 'Checking: Eff. data for 1 variable conversions directly from source to sink (and possibly back)\n';
check {(p, m) in process_method, t in time : m in method_1var} ptProcess[p, 'efficiency', t] != 0 ;

printf 'Checking: Efficiency data for 1-way conversions with an online variable\n';
check {(p, m) in process_method, t in time : m in method_1way_on} ptProcess[p, 'efficiency', t] != 0;

printf 'Checking: Efficiency data for 2-way linear conversions without online variables\n';
check {(p, m) in process_method, t in time : m in method_2way_off} ptProcess[p, 'efficiency', t] != 0;

minimize total_cost:
  + sum {(d, t) in dt}
    (
      + sum {(c, n) in commodity_node} pdCommodity[c, 'price', d]
	      * (
	          + sum {(p, n, sink) in process_source_sink : sum{(p, m) in process_method : m in method_1var_per_way} 1 } 
			         v_flow[p, n, sink, d, t] / ptProcess[p, 'efficiency', t]
	          + sum {(p, n, sink) in process_source_sink : sum{(p, m) in process_method : m not in method_1var_per_way} 1 } 
			         v_flow[p, n, sink, d, t]
	          + sum {(p, source, n) in process_source_sink} v_flow[p, source, n, d, t]
		    )
	  + sum {(g, c, n, d) in group_commodity_node_period_CO2} p_commodity[c, 'co2_content'] * pdGroup[g, 'co2_price', d] 
	      * (
	          + sum {(p, n, sink) in process_source_sink : sum{(p, m) in process_method : m in method_1var_per_way} 1 } 
			         v_flow[p, n, sink, d, t] / ptProcess[p, 'efficiency', t]
	          + sum {(p, n, sink) in process_source_sink : sum{(p, m) in process_method : m not in method_1var_per_way} 1 } 
			         v_flow[p, n, sink, d, t]
			)
	  + sum {p in process_online} v_startup_linear[p, d, t] * pdProcess[p, 'startup_cost', d]
	  + sum {(p, source, sink, 'variable_cost') in process__source__sink__param_t} 
	      + v_flow[p, source, sink, d, t] 
		    * ptProcess_source_sink[p, source, sink, 'variable_cost', t]
      + sum {(p, source, sink, m) in process__source__sink__ramp_method : m in ramp_cost_method}
          + v_ramp[p, source, sink, d, t] * pProcess_source_sink[p, source, sink, 'ramp_cost']
      + sum {g in groupInertia} vq_inertia[g, d, t] * pdGroup[g, 'penalty_inertia', d]
      + sum {g in groupNonSync} vq_non_synchronous[g, d, t] * pdGroup[g, 'penalty_non_synchronous', d]
      + sum {n in nodeBalance} vq_state_up[n, d, t] * ptNode[n, 'penalty_up', t]
      + sum {n in nodeBalance} vq_state_down[n, d, t] * ptNode[n, 'penalty_down', t]
      + sum {(r, ud, ng) in reserve__upDown__group} vq_reserve[r, ud, ng, d, t] * p_reserve_upDown_group[r, ud, ng, 'penalty_reserve']
	) * step_duration[d, t]
  + sum {(e, d) in ed_invest} v_invest[e, d]
    * p_entity_unitsize[e]
    * ed_entity_annual[e, d]
;

# Energy balance in each node  
s.t. nodeBalance_eq {n in nodeBalance, (d, t, t_previous, t_previous_within_block) in dttt} :
  + (if n in nodeState then (v_state[n, d, t] -  v_state[n, d, t_previous]))
  =
  + sum {(p, source, n) in process_source_sink} v_flow[p, source, n, d, t]
  + (if (n, 'no_inflow') not in node__inflow_method then 
      + ptNode[n, 'inflow', t] *
        ( if (n, 'scale_to_annual_flow') in node__inflow_method && pdNode[n, 'annual_flow', d] then
		    + annual_flow_multiplier[n, d]
		  else 1)
	)
  + vq_state_up[n, d, t]
  - sum {(p, n, sink) in process_source_sink : sum{(p, m) in process_method : m in method_1var_per_way} 1 } (
       + v_flow[p, n, sink, d, t] / ptProcess[p, 'efficiency', t]
    )		
  - sum {(p, n, sink) in process_source_sink : sum{(p, m) in process_method : m not in method_1var_per_way} 1 } (
       + v_flow[p, n, sink, d, t]
    )		
  - (if ptNode[n, 'self_discharge_loss', t] then v_state[n, d, t_previous] * ptNode[n, 'self_discharge_loss', t] * step_duration[d, t])
  - vq_state_down[n, d, t]
;

s.t. reserveBalance_timeseries_eq {(r, ud, ng, r_m) in reserve__upDown__group__method, (d, t) in dt : r_m = 'timeseries_only' || r_m = 'both'} :
  + sum {(p, r, ud, n) in process_reserve_upDown_node 
	      : sum{(p, m) in process_method : m in method_1var_per_way} 1
		  && (ng, n) in group_node 
		  && (r, ud, ng) in reserve__upDown__group} 
	    ( v_reserve[p, r, ud, n, d, t] 
	      * p_process_reserve_upDown_node[p, r, ud, n, 'reliability']
	    )
  + sum {(p, r, ud, n) in process_reserve_upDown_node  
		  : sum{(p, m) in process_method : m not in method_1var_per_way} 1
		  && (ng, n) in group_node 
		  && (r, ud, ng) in reserve__upDown__group} 
	    ( v_reserve[p, r, ud, n, d, t] 
	      * p_process_reserve_upDown_node[p, r, ud, n, 'reliability']
		  / ptProcess[p, 'efficiency', t]
		)
  + vq_reserve[r, ud, ng, d, t]
  >=
  + ptReserve_upDown_group[r, ud, ng, 'reservation', t]
;

s.t. reserveBalance_dynamic_eq{(r, ud, ng, r_m) in reserve__upDown__group__method, (d, t) in dt : r_m = 'dynamic_only' || r_m = 'both'} :
  + sum {(p, r, ud, n) in process_reserve_upDown_node 
	      : sum{(p, m) in process_method : m in method_1var_per_way} 1
		  && (ng, n) in group_node 
		  && (r, ud, ng) in reserve__upDown__group} 
	    ( v_reserve[p, r, ud, n, d, t] 
	      * p_process_reserve_upDown_node[p, r, ud, n, 'reliability']
	    )
  + sum {(p, r, ud, n) in process_reserve_upDown_node  
		  : sum{(p, m) in process_method : m not in method_1var_per_way} 1
		  && (ng, n) in group_node 
		  && (r, ud, ng) in reserve__upDown__group} 
	    ( v_reserve[p, r, ud, n, d, t] 
	      * p_process_reserve_upDown_node[p, r, ud, n, 'reliability']
		  / ptProcess[p, 'efficiency', t]
		)
  + vq_reserve[r, ud, ng, d, t]
  >=
  + sum {(p, r, ud, n) in process_reserve_upDown_node_increase_reserve_ratio : (ng, n) in group_node 
          && (r, ud, ng) in reserve__upDown__group}
	   (v_reserve[p, r, ud, n, d, t] * p_process_reserve_upDown_node[p, r, ud, n, 'increase_reserve_ratio'])
;

# Indirect efficiency conversion - there is more than one variable. Direct conversion does not have an equation - it's in the nodeBalance_eq.
s.t. conversion_indirect {(p, m) in process_method, (d, t) in dt : m in method_indirect} :
  + sum {source in entity : (p, source) in process_source} 
    ( + v_flow[p, source, p, d, t] 
  	      * p_process_source_coefficient[p, source]
	)
	* ptProcess[p, 'efficiency', t]
  =
  + sum {sink in entity : (p, sink) in process_sink} 
    ( + v_flow[p, p, sink, d, t] 
	      * p_process_sink_coefficient[p, sink]
	)
;

display process__source__sink__profile__profile_method;
s.t. profile_upper_limit {(p, source, sink, f, m) in process__source__sink__profile__profile_method, (d, t) in dt : m = 'upper_limit'} :
  + ( + v_flow[p, source, sink, d, t] 
  	      * ( if (p, source) in process_source then p_process_source_coefficient[p, source]
			  else if (p, sink) in process_sink then p_process_sink_coefficient[p, sink]
			  else 1
			)
	)
  <=
  + pt_profile[f, t]
    * ( + ( if p not in process_online then
              + p_entity_all_existing[p]
              + sum {(p, d_invest) in pd_invest : d_invest <= d} v_invest[p, d_invest] * p_entity_unitsize[p]
#              - sum {(p, d_invest) in pd_divest : d_invest <= d} v_divest[p, d_invest] * p_entity_unitsize[p]
	      )
        + ( if p in process_online then
              + v_online_linear[p, d, t]
	      )
      )
;

s.t. profile_lower_limit {(p, source, sink, f, m) in process__source__sink__profile__profile_method, (d, t) in dt : m = 'lower_limit'} :
  + ( + v_flow[p, source, sink, d, t] 
  	      * ( if (p, source) in process_source then p_process_source_coefficient[p, source]
			  else if (p, sink) in process_sink then p_process_sink_coefficient[p, sink]
			  else 1
			)
	)
  >=
  + pt_profile[f, t]
    * ( + ( if p not in process_online then
              + p_entity_all_existing[p]
              + sum {(p, d_invest) in pd_invest : d_invest <= d} v_invest[p, d_invest] * p_entity_unitsize[p]
#              - sum {(p, d_invest) in pd_divest : d_invest <= d} v_divest[p, d_invest] * p_entity_unitsize[p]
	      )
        + ( if p in process_online then
              + v_online_linear[p, d, t]
	      )
	  )
;

s.t. profile_fixed_limit {(p, source, sink, f, m) in process__source__sink__profile__profile_method, (d, t) in dt : m = 'fixed'} :
  + ( + v_flow[p, source, sink, d, t] 
  	      * ( if (p, source) in process_source then p_process_source_coefficient[p, source]
			  else if (p, sink) in process_sink then p_process_sink_coefficient[p, sink]
			  else 1
			)
	)
  =
  + pt_profile[f, t]
    * ( + ( if p not in process_online then
              + p_entity_all_existing[p]
              + sum {(p, d_invest) in pd_invest : d_invest <= d} v_invest[p, d_invest] * p_entity_unitsize[p]
#              - sum {(p, d_invest) in pd_divest : d_invest <= d} v_divest[p, d_invest] * p_entity_unitsize[p]
	      )
        + ( if p in process_online then
              + v_online_linear[p, d, t]
	    )
	  )
;

s.t. process_constraint_greater_than {(c, s) in constraint__sense, (d, t) in dt 
     : s in sense_greater_than} :
  + sum {(p, source, sink) in process_source_sink : (p, source, c) in process_node_constraint}
    ( + v_flow[p, source, sink, d, t]
	      * p_process_node_constraint_coefficient[p, source, c]
	)
  + sum {(p, source, sink) in process_source_sink : (p, sink, c) in process_node_constraint}
    ( + v_flow[p, source, sink, d, t]
	      * p_process_node_constraint_coefficient[p, sink, c]
	)
  >=
  + p_constraint_constant[c]
;
	
s.t. process_constraint_less_than {(c, s) in constraint__sense, (d, t) in dt 
     : s in sense_less_than} :
  + sum {(p, source, sink) in process_source_sink : (p, source, c) in process_node_constraint}
    ( + v_flow[source, source, p, d, t]
	      * p_process_node_constraint_coefficient[p, source, c]
	)
  + sum {(p, source, sink) in process_source_sink : (p, sink, c) in process_node_constraint}
    ( + v_flow[p, source, sink, d, t]
	      * p_process_node_constraint_coefficient[p, sink, c]
	)
  <=
  + p_constraint_constant[c]
;

s.t. process_constraint_equal {(c, s) in constraint__sense, (d, t) in dt 
     : s in sense_equal} :
  + sum {(p, source, sink) in process_source_sink : (p, source, c) in process_node_constraint}
    ( + v_flow[p, source, sink, d, t]
	      * p_process_node_constraint_coefficient[p, source, c]
	)
  + sum {(p, source, sink) in process_source_sink : (p, sink, c) in process_node_constraint}
    ( + v_flow[p, source, sink, d, t]
	      * p_process_node_constraint_coefficient[p, sink, c]
	)
  =
  + p_constraint_constant[c]
;


s.t. maxToSink {(p, source, sink) in process_source_sink, (d, t) in dt 
     : (p, sink) in process_sink } :
  + v_flow[p, source, sink, d, t]
  + sum {r in reserve : (p, r, 'up', sink) in process_reserve_upDown_node} v_reserve[p, r, 'up', sink, d, t]
  <=
  + ( if p not in process_online then
      + p_process_sink_coefficient[p, sink]
        * ( + p_entity_all_existing[p]
            + sum {(p, d_invest) in pd_invest : d_invest <= d} v_invest[p, d_invest] * p_entity_unitsize[p]
#            - sum {(p, d_invest) in pd_divest : d_invest <= d} v_divest[p, d_invest] * p_entity_unitsize[p]
	      )	
	)
  + ( if p in process_online then
      + p_process_sink_coefficient[p, sink]
        * v_online_linear[p, d, t]
    )  
;

s.t. minToSink {(p, source, sink) in process_source_sink, (d, t) in dt
     : (p, sink) in process_sink 
	 && sum{(p,m) in process_method : m not in method_2way_1var} 1 
} :
  + v_flow[p, source, sink, d, t]
  >=
  + 0
;

# Special equation to limit the 1variable connection on the negative transfer
s.t. minToSink_1var {(p, source, sink) in process_source_sink, (d, t) in dt
     : (p, sink) in process_sink 
	 && sum{(p,m) in process_method : m in method_2way_1var} 1 
} :
  + v_flow[p, source, sink, d, t]
  >=
  - p_process_sink_coefficient[p, sink] 
    * ( + p_entity_all_existing[p]
        + sum {(p, d_invest) in pd_invest : d_invest <= d} v_invest[p, d_invest] * p_entity_unitsize[p]
#        - sum {(p, d_invest) in pd_divest : d_invest <= d} v_divest[p, d_invest] * p_entity_unitsize[p]
	  )	

;

# Special equations for the method with 2 variables presenting 2way connection between source and sink (without the process)
s.t. maxToSource {(p, source, sink) in process_source_sink, (d, t) in dt
     : (p, source) in process_source
	 && sum{(p,m) in process_method : m in method_2way_2var } 1 
} :
  + v_flow[p, sink, source, d, t]
  + sum {r in reserve : (p, r, 'up', source) in process_reserve_upDown_node} v_reserve[p, r, 'up', source, d, t]
  <=
  + p_process_source_coefficient[p, source] 
    * ( + p_entity_all_existing[p]
        + sum {(p, d_invest) in pd_invest : d_invest <= d} v_invest[p, d_invest] * p_entity_unitsize[p]
#        - sum {(p, d_invest) in pd_divest : d_invest <= d} v_divest[p, d_invest] * p_entity_unitsize[p]
	  )	
;

s.t. minToSource {(p, source, sink) in process_source_sink, (d, t) in dt
     : (p, sink) in process_sink 
	 && sum{(p,m) in process_method : m in method_2way_2var } 1 
} :
  + v_flow[p, sink, source, d, t]
  >=
  + 0
;

s.t. maxOnline {p in process_online, (d, t) in dt} :
  + v_online_linear[p, d, t]
  <=
  + p_entity_all_existing[p] / p_entity_unitsize[p]
  + sum {(p, d_invest) in pd_invest : d_invest <= d} v_invest[p, d_invest]
#   - sum {(p, d_invest) in pd_divest : d_invest <= d} v_divest[p, d_invest]
;

s.t. online__startup_linear {p in process_online, (d, t, t_previous, t_previous_within_block) in dttt} :
  + v_startup_linear[p, d, t]
  >=
  + v_online_linear[p, d, t] 
  - v_online_linear[p, d, t_previous]
;

s.t. online__shutdown_linear {p in process_online, (d, t, t_previous, t_previous_within_block) in dttt} :
  + v_shutdown_linear[p, d, t]
  >=
  - v_online_linear[p, d, t] 
  + v_online_linear[p, d, t_previous]
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

s.t. ramp {(p, source, sink) in process_source_sink_ramp, (d, t, t_previous, t_previous_within_block) in dttt} :
  + v_ramp[p, source, sink, d, t]
  =
  + v_flow[p, source, sink, d, t]
  - v_flow[p, source, sink, d, t_previous]
;

s.t. ramp_up {(p, source, sink) in process_source_sink_ramp_limit_up, (d, t, t_previous, t_previous_within_block) in dttt
		: (p, source, sink, d, t) in process_source_sink_dt_ramp} :
  + v_ramp[p, source, sink, d, t]
  + sum {r in reserve : (p, r, 'up', sink) in process_reserve_upDown_node} 
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
#		- ( if (p, d) in ed_divest then v_divest[p, d] * p_entity_unitsize[p] )
	  )
  + ( if p in process_online then v_startup_linear[p, d, t] * p_entity_unitsize[p] )  # To make sure that units can startup despite ramp limits.
;

s.t. ramp_down {(p, source, sink) in process_source_sink_ramp_limit_down, (d, t, t_previous, t_previous_within_block) in dttt
		: (p, source, sink, d, t) in process_source_sink_dt_ramp} :
  + v_ramp[p, source, sink, d, t]
  + sum {r in reserve : (p, r, 'down', sink) in process_reserve_upDown_node} 
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
#		- ( if (p, d) in ed_divest then v_divest[p, d] * p_entity_unitsize[p] )
	  )
  - ( if p in process_online then v_shutdown_linear[p, d, t] * p_entity_unitsize[p] )  # To make sure that units can shutdown despite ramp limits.
;

s.t. reserve_process_upward{(p, r, ud, n, d, t) in prundt : ud = 'up'} :
  + v_reserve[p, r, ud, n, d, t]
  <=
  ( if p in process_online then
      + v_online_linear[p, n, t] * p_process_reserve_upDown_node[p, r, ud, n, 'max_share']
    else
      + p_process_reserve_upDown_node[p, r, ud, n, 'max_share'] 
        * (
            + p_entity_all_existing[p]
            + sum {(p, d_invest) in pd_invest : d_invest <= d} v_invest[p, d_invest] * p_entity_unitsize[p]
#            - sum {(p, d_invest) in pd_divest : d_invest <= d} v_divest[p, d_invest] * p_entity_unitsize[p]
          )
    	* ( if (sum{(p, prof, m) in process__profile__profile_method : m = 'upper_limit'} 1) then
	          ( + sum{(p, prof, m) in process__profile__profile_method : m = 'upper_limit'} pt_profile[prof, t] )
	        else 1
	      )
  )
;

s.t. reserve_process_downward{(p, r, ud, n, d, t) in prundt : ud = 'down'} :
  + v_reserve[p, r, ud, n, d, t]
  <=
  + p_process_reserve_upDown_node[p, r, ud, n, 'max_share']
    * ( + sum{(p, source, n) in process_source_sink} v_flow[p, source, n, d, t]
        - ( + p_entity_all_existing[p]
            + sum {(p, d_invest) in pd_invest : d_invest <= d} v_invest[p, d_invest] * p_entity_unitsize[p]
#            - sum {(p, d_invest) in pd_divest : d_invest <= d} v_divest[p, d_invest] * p_entity_unitsize[p]
          ) * ( if (sum{(p, prof, m) in process__profile__profile_method : m = 'lower_limit'} 1) then
	              ( + sum{(p, prof, m) in process__profile__profile_method : m = 'lower_limit'} pt_profile[prof, t] )
	          )
	  )		  
;

s.t. maxInvestGroup_entity_period {g in group, d in period_invest : pdGroup[g, 'invest_max_period', d] } :
  + sum{(g, e) in group_entity : (e, d) in ed_invest} v_invest[e, d] * p_entity_unitsize[e]
  <=
  + pdGroup[g, 'invest_max_period', d]
;

s.t. minInvestGroup_entity_period {g in group, d in period_invest : pdGroup[g, 'invest_min_period', d] } :
  + sum{(g, e) in group_entity : (e, d) in ed_invest} v_invest[e, d] * p_entity_unitsize[e]
  <=
  + pdGroup[g, 'invest_min_period', d]
;

s.t. maxInvestGroup_entity_total {g in group : p_group[g, 'invest_max_total'] } :
  + sum{(g, e) in group_entity, d in period : (e, d) in ed_invest}
    (
      + v_invest[e, d]
      + (if not p_model['solveFirst'] && e in entityInvest then p_entity_invested[e])
	)
  <=
  + p_group[g, 'invest_max_total']
;

s.t. maxInvest_entity_period {(e, d)  in ed_invest} :  # Covers both processes and nodes
  + v_invest[e, d] * p_entity_unitsize[e] 
  <= 
  + sum{(e, m) in entity__invest_method : m not in invest_method_not_allowed && (e,d) in pd_invest} pdProcess[e, 'invest_max_period', d]
  + sum{(e, m) in entity__invest_method : m not in invest_method_not_allowed && (e,d) in nd_invest} pdNode[e, 'invest_max_period', d]
;

s.t. inertia_constraint {g in groupInertia, (d, t) in dt} :
  + sum {(p, source, sink) in process_source_sink : (p, source) in process_source && (g, source) in group_node && p_process_source[p, source, 'inertia_constant']} 
      + ( + (if p in process_online then v_online_linear[p, d, t]) 
	      + (if p not in process_online then v_flow[p, source, sink, d, t])
	    ) * p_process_source[p, source, 'inertia_constant']
  + sum {(p, source, sink) in process_source_sink : (p, sink) in process_sink && (g, sink) in group_node && p_process_sink[p, sink, 'inertia_constant']} 
      + ( + (if p in process_online then v_online_linear[p, d, t]) 
	      + (if p not in process_online then v_flow[p, source, sink, d, t])
        ) * p_process_sink[p, sink, 'inertia_constant']
  + vq_inertia[g, d, t]
  >=
  pdGroup[g, 'inertia_limit', d]
;

s.t. non_sync_constraint{g in groupNonSync, (d, t) in dt} :
  + sum {(p, source, sink) in process_source_sink : (p, sink) in process_sink && (g, sink) in group_node && (p, sink) in process__sink_nonSync}
      + v_flow[p, source, sink, d, t]
  <=
  ( + sum {(p, source, sink) in process_source_sink : (p, source) in process_source && (g, source) in group_node} 
        + v_flow[p, source, sink, d, t]
    + sum {(g, n) in group_node} ptNode[n, 'inflow', t]
    + vq_non_synchronous[g, d, t]
  ) * pdGroup[g, 'non_synchronous_limit', d]
;



solve;

param entity_all_capacity{e in entity, d in period_realized} :=
  + p_entity_all_existing[e]
  + sum {(e, d_invest) in ed_invest : d <= d_invest} v_invest[e, d_invest].val * p_entity_unitsize[e]
;

param r_process_source_sink_flow_d{(p, source, sink) in process_source_sink, d in period_realized} :=
  + sum {(c,m) in process_method, (d, t) in dt : m not in method_2way_2var} v_flow[p, source, sink, d, t]
  + sum {(c,m) in process_method, (d, t) in dt : m in method_2way_2var} v_flow[p, source, sink, d, t] / ptProcess[c, 'efficiency', t]
;
param r_process_source_flow_d{(p, source) in process_source, d in period_realized} := 
  + sum {(p, source, sink) in process_source_sink, (d, t) in dt} r_process_source_sink_flow_d[p, source, sink, d]
;
param r_process_sink_flow_d{(p, sink) in process_sink, d in period_realized} := 
  + sum {(p, source, sink) in process_source_sink, (d, t) in dt} r_process_source_sink_flow_d[p, source, sink, d]
;

param r_cost_commodity{(c, n) in commodity_node, (d, t) in dt} := 
  + step_duration[d, t] * pdCommodity[c, 'price', d] 
      * ( 
	      + sum {(p, n, sink) in process_source_sink : sum{(p, m) in process_method : m in method_1var_per_way} 1 } 
		         v_flow[p, n, sink, d, t].val / ptProcess[p, 'efficiency', t]
	      + sum {(p, n, sink) in process_source_sink : sum{(p, m) in process_method : m not in method_1var_per_way} 1 } 
		         v_flow[p, n, sink, d, t].val
	      + sum {(p, source, n) in process_source_sink} v_flow[p, source, n, d, t].val
		)
;

param r_cost_process_variable_cost{p in process, (d, t) in dt} :=
  + step_duration[d, t]
	  * sum {(p, source, sink, 'variable_cost') in process__source__sink__param_t} 
	         v_flow[p, source, sink, d, t].val * ptProcess_source_sink[p, source, sink, 'variable_cost', t]
;

param r_costPenalty_nodeState{n in nodeBalance, (d, t) in dt} :=
  + step_duration[d, t]
      * (
          + vq_state_up[n, d, t] * ptNode[n, 'penalty_up', t]
          + vq_state_down[n, d, t] * ptNode[n, 'penalty_down', t]
		)
;

param r_costPenalty_reserve_upDown{(r, ud, ng) in reserve__upDown__group, (d, t) in dt} :=
  + step_duration[d, t]
      * (
          + vq_reserve[r, ud, ng, d, t] * p_reserve_upDown_group[r, ud, ng, 'penalty_reserve']
	    )
;
		
param r_cost_entity_invest{(e, d) in ed_invest} :=
  + v_invest[e, d]
      * p_entity_unitsize[e]
      * ed_entity_annual[e, d]
;

param r_cost_dt{(d, t) in dt} :=
  + sum{(c, n) in commodity_node} r_cost_commodity[c, n, d, t]
  + sum{p in process} r_cost_process_variable_cost[p, d, t]
;

param r_costPenalty_dt{(d, t) in dt} :=
  + sum{n in nodeBalance} r_costPenalty_nodeState[n, d, t]
  + sum{(r, ud, ng) in reserve__upDown__group} r_costPenalty_reserve_upDown[r, ud, ng, d, t]
;

param r_cost_and_penalty_dt{(d,t) in dt} :=
  + r_cost_dt[d, t]
  + r_costPenalty_dt[d, t]
;


printf 'Transfer investments to the next solve...\n';
param fn_entity_invested symbolic := "p_entity_invested.csv";
printf 'entity,p_entity_invested\n' > fn_entity_invested;
for {e in entity: e in entityInvest} 
  {
    printf '%s,%.8g\n', e, 
	  + (if not p_model['solveFirst'] then p_entity_invested[e] else 0)
	  + sum {d_invest in period_invest} v_invest[e, d_invest].val * p_entity_unitsize[e]
	>> fn_entity_invested;
  }

printf 'Write process investment results...\n';
param fn_process_investment symbolic := "r_process_investment__d.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf 'process,period,invested\n' > fn_process_investment; }  # Clear the file on the first solve
for {(p, d) in pd_invest : d in period_realized && d in period_invest}
  {
    printf '%s, %s, %.8g\n', p, d, v_invest[p, d].val * p_entity_unitsize[p] >> fn_process_investment;
  }


printf 'Write unit__sinkNode flow for periods...\n';
param fn_unit__sinkNode__d symbolic := "r_unit__sinkNode__d.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf 'unit,node,period,flow\n' > fn_unit__sinkNode__d; }  # Print the header on the first solve
for {u in process_unit, d in period_realized}
  {
    for {(u, sink) in process_sink}
      {
        printf '%s, %s, %s, %.8g\n', u, sink, d, r_process_sink_flow_d[u, sink, d] >> fn_unit__sinkNode__d;
      }
  } 

printf 'Write unit__sinkNode flow for time...\n';
param fn_unit__sinkNode__dt symbolic := "r_unit__sinkNode__dt.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf 'unit,node,period,time,flow\n' > fn_unit__sinkNode__dt; }  # Print the header on the first solve
for {(u, m) in process_method, (d, t) in dt : d in period_realized && u in process_unit}
  {
    for {(u, sink) in process_sink}
      {
        printf '%s, %s, %s, %s, %.8g\n', u, sink, d, t
		   , ( if m not in method_1var_per_way then v_flow[u, u, sink, d, t].val
		       else sum{(u, source, sink) in process_source_sink} v_flow[u, source, sink, d, t].val
		     )
		   >> fn_unit__sinkNode__dt;
      }
  } 

printf 'Write unit__sourceNode flow for periods...\n';
param fn_unit__sourceNode__d symbolic := "r_unit__sourceNode__d.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf 'unit,node,period,flow\n' > fn_unit__sourceNode__d; }  # Print the header on the first solve
for {u in process_unit, d in period_realized}
  {
    for {(u, source) in process_source}
      {
        printf '%s, %s, %s, %.8g\n', u, source, d, r_process_source_flow_d[u, source, d] >> fn_unit__sourceNode__d;
      }
  } 

printf 'Write unit__sourceNode flow for time...\n';
param fn_unit__sourceNode__dt symbolic := "r_unit__sourceNode__dt.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf 'unit,node,period,time,flow\n' > fn_unit__sourceNode__dt; }  # Print the header on the first solve
for {(u, m) in process_method, (d, t) in dt : d in period_realized && u in process_unit}
  {
    for {(u, source) in process_source}
      {
        printf '%s, %s, %s, %s, %.8g\n', u, source, d, t
		   , ( if m not in method_1var_per_way then v_flow[u, source, u, d, t].val
		       else sum{(u, source, sink) in process_source_sink} v_flow[u, source, sink, d, t].val
		     )
		   >> fn_unit__sourceNode__dt;
      }
  } 

printf 'Write connection flow for periods...\n';
param fn_connection__d symbolic := "r_connection__d.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf 'connection,source,sink,period,flow\n' > fn_connection__d; }  # Print the header on the first solve
for {(c, source, sink) in process_source_sink, d in period_realized : c in process_connection}
  {
    printf '%s, %s, %s, %s, %.8g\n', c, source, sink, d, r_process_source_sink_flow_d[c, source, sink, d] >> fn_connection__d;
  } 

printf 'Write connection flow for time...\n';
param fn_connection__dt symbolic := "r_connection__dt.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf 'connection,source,sink,period,time,flow\n' > fn_connection__dt; }  # Print the header on the first solve
for {(c, source, sink) in process_source_sink, (d, t) in dt 
       : d in period_realized 
	     && sum{(c,m) in process_method : m in method_2way} 1
		 && (c, source) in process_source
	}
  {
    printf '%s, %s, %s, %s, %s, %.8g\n', c, source, sink, d, t, 
	  + v_flow[c, source, sink, d, t].val 
	  - ( if sum{(c,m) in process_method : m in method_2way_2var} 1 
	      then v_flow[c, sink, source, d, t].val / ptProcess[c, 'efficiency', t]
		) >> fn_connection__dt;
  } 

printf 'Write reserve from processes over time...\n';
param fn_process__reserve__upDown__node__dt symbolic := "r_process__reserve__upDown__node__dt.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf 'process,reserve,upDown,node,period,time,reservation\n' > fn_process__reserve__upDown__node__dt; }  # Print the header on the first solve
for {(p, r, ud, n) in process_reserve_upDown_node, (d, t) in dt}
  {
    printf '%s, %s, %s, %s, %s, %s, %.8g\n', p, r, ud, n, d, t, v_reserve[p, r, ud, n, d, t].val >> fn_process__reserve__upDown__node__dt;
  }

printf 'Write online status of units over time...\n';
param fn_unit_online__dt symbolic := "r_unit_online__dt.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf 'unit,period,time,online\n' > fn_unit_online__dt; }  # Print the header on the first solve
for {p in process_unit, (d, t) in dt : p in process_online}
  {
    printf '%s, %s, %s, %.8g\n', p, d, t, v_online_linear[p, d, t].val >> fn_unit_online__dt;
  }
 

printf 'Write node results for periods...\n';
param fn_node__d symbolic := "r_node__d.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf 'node,period,exogenous_flow,state_change,inflow,outflow,penalty\n' > fn_node__d; }  # Print the header on the first solve
for {n in node, d in period_realized : d not in period_invest}
  {
    printf '%s, %s, %.8g, %.8g, %.8g, %.8g, %.8g\n'
		, n, d
		, (if (n, 'scale_to_annual_flow') in node__inflow_method then sum {t in time : (d, t) in dt} ptNode[n, 'inflow', t] else 0)
        , (if n in nodeState then sum {(d, t, t_previous, t_previous_within_block) in dttt}
		      (v_state[n, d, t] -  v_state[n, d, t_previous]) else 0)
        , sum {(p, source, n) in process_source_sink, t in time : (d, t) in dt} v_flow[p, source, n, d, t]
        , - sum {(p, n, sink) in process_source_sink, t in time : (d, t) in dt && sum{(p, m) in process_method : m in method_1var_per_way} 1 }
		     ( + v_flow[p, n, sink, d, t] / ptProcess[p, 'efficiency', t] )
          - sum {(p, n, sink) in process_source_sink, t in time : (d, t) in dt && sum{(p, m) in process_method : m not in method_1var_per_way} 1 }
		     ( + v_flow[p, n, sink, d, t] )
        , (if n in nodeBalance then sum {t in time : (d, t) in dt} (
		      + vq_state_up[n, d, t]
              - vq_state_down[n, d, t]
          ) else 0)
		>> fn_node__d;
  }

printf 'Write group inertia over time...\n';
param fn_group_inertia__dt symbolic := "r_group_inertia__dt.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf 'group,period,inertia,penalty_variable\n' > fn_group_inertia__dt; }
for {g in groupInertia, (d, t) in dt : d in period_realized}
  {
    printf '%s, %s, %s, %.8g, %.8g\n'
	    , g, d, t
		, + sum {(p, source, sink) in process_source_sink : (p, source) in process_source && (g, source) in group_node && p_process_source[p, source, 'inertia_constant']} 
              + ( + (if p in process_online then v_online_linear[p, d, t]) 
	              + (if p not in process_online then v_flow[p, source, sink, d, t])
	            ) * p_process_source[p, source, 'inertia_constant']
          + sum {(p, source, sink) in process_source_sink : (p, sink) in process_sink && (g, sink) in group_node && p_process_sink[p, sink, 'inertia_constant']} 
              + ( + (if p in process_online then v_online_linear[p, d, t]) 
	              + (if p not in process_online then v_flow[p, source, sink, d, t])
                ) * p_process_sink[p, sink, 'inertia_constant']
		, vq_inertia[g, d, t]
		>> fn_group_inertia__dt;
  }


param resultFile symbolic := "result.csv";

printf 'Upward slack for node balance\n' > resultFile;
for {n in nodeBalance, (d, t) in dt}
  {
    printf '%s, %s, %s, %.8g\n', n, d, t, vq_state_up[n, d, t].val >> resultFile;
  }

printf '\nDownward slack for node balance\n' >> resultFile;
for {n in nodeBalance, (d, t) in dt}
  {
    printf '%s, %s, %s, %.8g\n', n, d, t, vq_state_down[n, d, t].val >> resultFile;
  }

printf '\nReserve upward slack variable\n' >> resultFile;
for {(r, ud, ng) in reserve__upDown__group, (d, t) in dt}
  {
    printf '%s, %s, %s, %s, %s, %.8g\n', r, ud, ng, d, t, vq_reserve[r, ud, ng, d, t].val >> resultFile;
  }

printf '\nFlow variables\n' >> resultFile;
for {(p, source, sink) in process_source_sink, (d, t) in dt}
  {
    printf '%s, %s, %s, %s, %s, %.8g\n', p, source, sink, d, t, v_flow[p, source, sink, d, t].val >> resultFile;
  }

printf '\nInvestments\n' >> resultFile;
for {(e, d_invest) in ed_invest} {
  printf '%s, %s, %.8g\n', e, d_invest, v_invest[e, d_invest].val * p_entity_unitsize[e] >> resultFile;
}

printf '\nDivestments\n' >> resultFile;
for {(e, d_invest) in ed_divest} {
  printf '%s, %s, %.8g\n', e, d_invest, v_divest[e, d_invest].val * p_entity_unitsize[e] >> resultFile;
}


printf '\nNode balances\n' >> resultFile;
for {n in node} {
  printf '\n%s\nPeriod, Time', n >> resultFile;
  printf (if (n, 'scale_to_annual_flow') in node__inflow_method then ', %s' else ''), n >> resultFile;
  for {(p, source, n) in process_source_sink} {
    printf ', %s->', source >> resultFile;
  }
  for {(p, n, sink) in process_source_sink : sum{(p, m) in process_method : m in method_1var_per_way} 1 } {
    printf ', ->%s', sink >> resultFile;
  }
  for {(p, n, sink) in process_source_sink : sum{(p, m) in process_method : m not in method_1var_per_way} 1 } {
    printf ', ->%s->', sink >> resultFile;
  }
  printf '\n' >> resultFile;
  for {(d, t) in dt} {
    printf '%s, %s', d, t >> resultFile;
	printf (if (n, 'scale_to_annual_flow') in node__inflow_method then ', %.8g' else ''), ptNode[n, 'inflow', t] >> resultFile; 
    for {(p, source, n) in process_source_sink} {
      printf ', %.8g', v_flow[p, source, n, d, t].val >> resultFile;
	}
    for {(p, n, sink) in process_source_sink : sum{(p, m) in process_method : m in method_1var_per_way} 1 } {
      printf ', %.8g', -v_flow[p, n, sink, d, t].val / ptProcess[p, 'efficiency', t] >> resultFile;
	}
    for {(p, n, sink) in process_source_sink : sum{(p, m) in process_method : m not in method_1var_per_way} 1 } {
      printf ', %.8g', -v_flow[p, n, sink, d, t].val >> resultFile;
	}
    printf '\n' >> resultFile;
  }
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


end;
