# © International Renewable Energy Agency 2018-2021

#The FlexTool is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License
#as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

#The FlexTool is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
#without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

#You should have received a copy of the GNU Lesser General Public License along with the FlexTool.  
#If not, see <https://www.gnu.org/licenses/>.

#Author: Juha Kiviluoma (2017-2021), VTT Technical Research Centre of Finland

#########################
# Fundamental sets of the model
set entity 'e - contains both nodes and processes';
set process 'p - Particular activity that transfers, converts or stores commodities' within entity;
set processUnit 'Unit processes' within process;
set processTransfer 'Transfer processes' within process;
set node 'n - Any location where a balance needs to be maintained' within entity;
set nodeGroup 'ng - Any group of nodes that have a set of common constraints';
set commodity 'c - Stuff that is being processed';
set reserve_upDown_nodeGroup dimen 3;
set reserve 'r - Categories for the reservation of capacity_existing' := setof {(r, ud, ng) in reserve_upDown_nodeGroup} (r);
set period_time '(d, t) - Time steps in the time periods of the timelines in use' dimen 2;
set period 'd - Time periods in the current timelines' := setof {(d, t) in period_time} (d);
set time 't - Time steps in the current timelines'; 
set method 'm - Type of process that transfers, converts or stores commodities';
set upDown 'upward and downward directions for some variables';
set ct_method;
set startup_method;
set ramp_method;
set ramp_limit_method within ramp_method;
set ramp_cost_method within ramp_method;
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
set nodeInflow 'nodes that have an inflow' within node;
set nodeGroup_node 'member nodes of a particular nodeGroup' dimen 2 within {nodeGroup, node};
set process_unit 'processes that are unit' within process;
set process_connection 'processes that are connections' within process;
set process_ct_method dimen 2 within {process, ct_method};
set process_startup_method dimen 2 within {process, startup_method};
set process_ramp_method dimen 2 within {process, ramp_method};
set methods dimen 3; 
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
set process_source_sink := 
    process_source_toSink union 
	process_sink_toSource union   
	process_source_toProcess union 
	process_process_toSink union 
	process_sink_toProcess union  # Add the 'wrong' direction in 2-way processes with multiple inputs/outputs
	process_process_toSource;     # Add the 'wrong' direction in 2-way processes with multiple inputs/outputs
set process_online 'processes with an online status' := setof {(p, m) in process_method : m in method_LP} p;

set process_reserve_upDown_node dimen 4;
set process_node_constraint dimen 3 within {process, node, constraint};
set process_constraint_sense dimen 3 within {process, constraint, sense};
set commodity_node dimen 2 within {commodity, node}; 
set connection_variable_cost dimen 1 within process;
set unit__sourceNode_variable_cost dimen 2 within process_source;
set unit__sinkNode_variable_cost dimen 2 within process_sink;
set process_source_sink_variable_cost :=
    { (p, source, sink) in process_source_sink 
	    :  (p, source) in unit__sourceNode_variable_cost 
	    || (p, sink) in unit__sinkNode_variable_cost 
	    || p in connection_variable_cost
	};

set commodityParam;
set nodeParam;
set processParam;
set sourceSinkParam;
set reserveParam;

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

param p_node {n in node, nodeParam} default 0;
param pt_node {n in node, nodeParam, t in time};
param pt_commodity {(c, n) in commodity_node, commodityParam, t in time};
param p_process {process, processParam} default 0;
param pt_process {process, processParam, time} default 0;
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
param p_process_sink {(p, sink) in process_sink, sourceSinkParam} default 0;
param p_process_source_coefficient {(p, source) in process_source} := 
    + if (p_process_source[p, source, 'coefficient']) 
	  then p_process_source[p, source, 'coefficient'] 
	  else 1;
param p_process_sink_coefficient {(p, sink) in process_sink} := 
    + if (p_process_sink[p, sink, 'coefficient']) 
	  then p_process_sink[p, sink, 'coefficient'] 
	  else 1;
param p_process_source_flow_unitsize {(p, source) in process_source} := 
    + if (p_process_source[p, source, 'flow_unitsize']) 
	  then p_process_source[p, source, 'flow_unitsize'] 
	  else 1;
param p_process_sink_flow_unitsize {(p, sink) in process_sink} := 
    + if (p_process_sink[p, sink, 'flow_unitsize']) 
	  then p_process_sink[p, sink, 'flow_unitsize'] 
	  else 1;
param p_inflow {n in nodeInflow, t in time};
param p_reserve_upDown_nodeGroup {reserve, upDown, nodeGroup, reserveParam};
param pt_reserve_upDown_nodeGroup {reserve, upDown, nodeGroup, reserveParam, time};
param p_process_reserve_upDown_node {process, reserve, upDown, node, reserveParam} default 0;
param p_process_constraint_constant {process, constraint};
param p_process_node_constraint_coefficient {process, node, constraint};
param pq_up {n in nodeBalance};
param pq_down {n in nodeBalance};
param step_duration{(d, t) in dt};
param step_period{(d, t) in dt} := 0;
param ed_entity_annual{e in entityInvest, d in period_invest} :=
        + sum{m in invest_method : (e, m) in entity__invest_method && e in node && m = 'one_cost'}
            (p_node[e, 'invest_cost'] * 1000 * ( p_node[e, 'interest_rate'] / (1 - (1 / (1 + p_node[e, 'interest_rate'])^p_node[e, 'lifetime'] ) ) ))
        + sum{m in invest_method : (e, m) in entity__invest_method && e in process && m = 'one_cost'}
            (p_process[e, 'invest_cost'] * 1000 * ( p_process[e, 'interest_rate'] / (1 - (1 / (1 + p_process[e, 'interest_rate'])^p_process[e, 'lifetime'] ) ) ))
; 			
param p_process_source_sink_variable_cost{(p, source, sink) in process_source_sink_variable_cost} :=
        + (if (p, source) in unit__sourceNode_variable_cost then p_process_source[p, source, 'variable_cost'])
        + (if (p, sink) in unit__sinkNode_variable_cost then p_process_sink[p, sink, 'variable_cost'])
		+ (if p in connection_variable_cost then p_process[p, 'variable_cost'])
;

set ed_invest := {e in entityInvest, d in period_invest : ed_entity_annual[e, d]};
set pd_invest := {(p, d) in ed_invest : p in process};
set nd_invest := {(n, d) in ed_invest : n in node};
set ed_divest := ed_invest;

set process_source_sink_ramp_limit_up :=
    {(p, source, sink) in process_source_sink
	    : sum{(p, m) in process_ramp_method : m in ramp_limit_method} 1
		&& p_process[p, 'ramp_speed_up'] > 0
	};
set process_source_sink_ramp_limit_down :=
    {(p, source, sink) in process_source_sink
	    : sum{(p, m) in process_ramp_method : m in ramp_limit_method} 1
		&& p_process[p, 'ramp_speed_down'] > 0
	};
set process_source_sink_ramp_cost :=
    {(p, source, sink) in process_source_sink
	    : sum{(p, m) in process_ramp_method : m in ramp_cost_method} 1
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

param p_process_invested {p in process : p in entityInvest};
param p_process_all_existing {p in process} :=
        + p_process[p, 'existing']
		+ p_process[p, 'invest_forced']
		+ (if not p_model['solveFirst'] && p in entityInvest then p_process_invested[p])
;

param d_obj default 0;
param d_flow {(p, source, sink, d, t) in peedt} default 0;
param d_flow_1_or_2_variable {(p, source, sink, d, t) in peedt} default 0;
param d_flowInvest {(p, d) in pd_invest} default 0;
param d_reserve_upDown_node {(p, r, ud, n, d, t) in prundt} default 0;
param dq_reserve {(r, ud, ng) in reserve_upDown_nodeGroup, (d, t) in dt} default 0;

#########################
# Read parameter data (no time series yet)
table data IN 'CSV' 'entity.csv': entity <- [entity];
table data IN 'CSV' 'process.csv': process <- [process];
table data IN 'CSV' 'process_unit.csv': process_unit <- [process_unit];
table data IN 'CSV' 'process_connection.csv': process_connection <- [process_connection];
table data IN 'CSV' 'node.csv' : node <- [node];
table data IN 'CSV' 'nodeGroup.csv' : nodeGroup <- [nodeGroup];
table data IN 'CSV' 'commodity.csv' : commodity <- [commodity];
table data IN 'CSV' 'timeline.csv' : time <- [timestep];
table data IN 'CSV' 'steps_in_timeline.csv' : period_time <- [period,step];
table data IN 'CSV' 'reserve__upDown__nodeGroup.csv' : reserve_upDown_nodeGroup <- [reserve,upDown,nodeGroup];
table data IN 'CSV' 'commodity__node.csv' : commodity_node <- [commodity,node];

table data IN 'CSV' 'nodeBalance.csv' : nodeBalance <- [nodeBalance];
table data IN 'CSV' 'nodeState.csv' : nodeState <- [nodeState];
table data IN 'CSV' 'nodeInflow.csv' : nodeInflow <- [nodeInflow];
table data IN 'CSV' 'nodeGroup__node.csv': nodeGroup_node <- [nodeGroup,node];
table data IN 'CSV' 'process__ct_method.csv' : process_ct_method <- [process,ct_method];
table data IN 'CSV' 'process__startup_method.csv' : process_startup_method <- [process,startup_method];
table data IN 'CSV' 'process__ramp_method.csv' : process_ramp_method <- [process,ramp_method];
table data IN 'CSV' 'process__source.csv' : process_source <- [process,source];
table data IN 'CSV' 'process__sink.csv' : process_sink <- [process,sink];
table data IN 'CSV' 'process__reserve__upDown__node.csv' : process_reserve_upDown_node <- [process,reserve,upDown,node];
table data IN 'CSV' 'entity__invest_method.csv' : entity__invest_method <- [entity,invest_method];
table data IN 'CSV' 'connection_variable_cost.csv' : connection_variable_cost <- [process];
table data IN 'CSV' 'unit__sourceNode_variable_cost.csv' : unit__sourceNode_variable_cost <- [process,source];
table data IN 'CSV' 'unit__sinkNode_variable_cost.csv' : unit__sinkNode_variable_cost <- [process,sink];

table data IN 'CSV' 'p_node.csv' : [node, nodeParam], p_node;
table data IN 'CSV' 'pt_node.csv' : [node, nodeParam, time], pt_node;
table data IN 'CSV' 'pt_commodity.csv' : [commodity, node, commodityParam, time], pt_commodity;
table data IN 'CSV' 'p_process.csv' : [process, processParam], p_process;
table data IN 'CSV' 'pt_process.csv' : [process, processParam, time], pt_process;
table data IN 'CSV' 'p_process_source.csv' : [process, source, sourceSinkParam], p_process_source;
table data IN 'CSV' 'p_process_sink.csv' : [process, sink, sourceSinkParam], p_process_sink;
table data IN 'CSV' 'p_reserve__upDown__nodeGroup.csv' : [reserve, upDown, nodeGroup, reserveParam], p_reserve_upDown_nodeGroup;
table data IN 'CSV' 'pt_reserve__upDown__nodeGroup.csv' : [reserve, upDown, nodeGroup, reserveParam, time], pt_reserve_upDown_nodeGroup;
table data IN 'CSV' 'p_process__reserve__upDown__node.csv' : [process, reserve, upDown, node, reserveParam], p_process_reserve_upDown_node;
table data IN 'CSV' 'p_process_invested.csv' : [process], p_process_invested;

table data IN 'CSV' 'constraint.csv' : constraint <- [constraint];
table data IN 'CSV' 'p_process_constraint_constant.csv' : process_constraint_sense <- [process, constraint, sense];
table data IN 'CSV' 'p_process_constraint_constant.csv' : [process, constraint], p_process_constraint_constant;
table data IN 'CSV' 'p_process_node_constraint_coefficient.csv' : process_node_constraint <- [process, node, constraint];
table data IN 'CSV' 'p_process_node_constraint_coefficient.csv' : [process, node, constraint], p_process_node_constraint_coefficient;

#table data IN 'CSV' '.csv' :  <- [];

table data IN 'CSV' 'debug.csv' : debug <- [debug];
table data IN 'CSV' 'steps_in_use.csv' : dt <- [period, step];
table data IN 'CSV' 'steps_in_use.csv' : [period, step], step_duration;
table data IN 'CSV' 'step_previous.csv' : dttt <- [period, time, previous, previous_within_block];
table data IN 'CSV' 'realized_periods_of_current_solve.csv' : period_realized <- [period];
table data IN 'CSV' 'invest_periods_of_current_solve.csv' : period_invest <- [period];
#table data IN 'CSV' 'solve_start.csv' : startTime <- [start];
#table data IN 'CSV' 'solve_startNext.csv' : startNext <- [startNext];
table data IN 'CSV' 'p_model.csv' : [modelParam], p_model;

#########################
# Variable declarations
var v_flow {(p, source, sink, d, t) in peedt};
var v_ramp {(p, source, sink, d, t) in peedt};
var v_reserve {(p, r, ud, n, d, t) in prundt} >= 0;
var v_state {n in nodeState, (d, t) in dt} >= 0;
var v_online_linear {p in process_online,(d, t) in dt} >=0;
var v_startup_linear {p in process_online, (d, t) in dt} >=0;
var v_invest {(e, d) in ed_invest} >= 0;
var v_divest {(e, d) in ed_divest} >= 0;
var vq_state_up {n in nodeBalance, (d, t) in dt} >= 0;
var vq_state_down {n in nodeBalance, (d, t) in dt} >= 0;
var vq_reserve {(r, ud, ng) in reserve_upDown_nodeGroup, (d, t) in dt} >= 0;

display sense_greater_than, process_constraint_sense, process_node_constraint, p_process_node_constraint_coefficient;

#########################
## Data checks 
printf 'Checking: Data for 1 variable conversions directly from source to sink (and possibly back)\n';
check {(p, m) in process_method, t in time : m in method_1var} pt_process[p, 'efficiency', t] != 0 ;

printf 'Checking: Data for 1-way conversions with an online variable\n';
check {(p, m) in process_method, t in time : m in method_1way_on} pt_process[p, 'efficiency', t] != 0;

printf 'Checking: Data for 2-way linear conversions without online variables\n';
check {(p, m) in process_method, t in time : m in method_2way_off} pt_process[p, 'efficiency', t] != 0;

minimize total_cost:
  + sum {(d, t) in dt}
    (
      + sum {(c, n) in commodity_node} pt_commodity[c, n, 'price', t]
          * ( 
	          + sum {(p, n, sink) in process_source_sink : sum{(p, m) in process_method : m in method_1var_per_way} 1 } 
			         v_flow[p, n, sink, d, t] / pt_process[p, 'efficiency', t]
	          + sum {(p, n, sink) in process_source_sink : sum{(p, m) in process_method : m not in method_1var_per_way} 1 } 
			         v_flow[p, n, sink, d, t]
	          + sum {(p, source, n) in process_source_sink} v_flow[p, source, n, d, t]
		    )
	  + sum {p in process_online} v_startup_linear[p, d, t] * p_process[p, 'startup_cost']
	  + sum {(p, source, sink) in process_source_sink_variable_cost} 
	      + v_flow[p, source, sink, d, t] 
		    * p_process_source_sink_variable_cost[p, source, sink]
      + sum {(p, source, sink) in process_source_sink_ramp_cost}
          + v_ramp[p, source, sink, d, t] * p_process[p, 'ramp_cost']
      + sum {n in nodeBalance} vq_state_up[n, d, t] * p_node[n, 'pq_up']
      + sum {n in nodeBalance} vq_state_down[n, d, t] * p_node[n, 'pq_down']
      + sum {(r, ud, ng) in reserve_upDown_nodeGroup} vq_reserve[r, ud, ng, d, t] * p_reserve_upDown_nodeGroup[r, ud, ng, 'penalty_reserve']
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
  + (if n in nodeInflow then pt_node[n, 'inflow', t])
  + vq_state_up[n, d, t]
  - sum {(p, n, sink) in process_source_sink : sum{(p, m) in process_method : m in method_1var_per_way} 1 } (
       + v_flow[p, n, sink, d, t] / pt_process[p, 'efficiency', t]
    )		
  - sum {(p, n, sink) in process_source_sink : sum{(p, m) in process_method : m not in method_1var_per_way} 1 } (
       + v_flow[p, n, sink, d, t]
    )		
  - vq_state_down[n, d, t]
;

s.t. reserveBalance_eq {(r, ud, ng) in reserve_upDown_nodeGroup, (d, t) in dt} :
  + sum {(p, r, ud, n) in process_reserve_upDown_node_increase_reserve_ratio : (ng, n) in nodeGroup_node 
          && (r, ud, ng) in reserve_upDown_nodeGroup}
	   (v_reserve[p, r, ud, n, d, t] * p_process_reserve_upDown_node[p, r, ud, n, 'increase_reserve_ratio'])
  + pt_reserve_upDown_nodeGroup[r, ud, ng, 'reservation', t]
  =
  + vq_reserve[r, ud, ng, d, t]
  + sum {(p, r, ud, n) in process_reserve_upDown_node 
	      : sum{(p, m) in process_method : m in method_1var_per_way} 1
		  && (ng, n) in nodeGroup_node 
		  && (r, ud, ng) in reserve_upDown_nodeGroup} 
	    ( v_reserve[p, r, ud, n, d, t] 
	      * p_process_reserve_upDown_node[p, r, ud, n, 'reliability']
	    )
  + sum {(p, r, ud, n) in process_reserve_upDown_node  
		  : sum{(p, m) in process_method : m not in method_1var_per_way} 1
		  && (ng, n) in nodeGroup_node 
		  && (r, ud, ng) in reserve_upDown_nodeGroup} 
	    ( v_reserve[p, r, ud, n, d, t] 
	      * p_process_reserve_upDown_node[p, r, ud, n, 'reliability']
		  / pt_process[p, 'efficiency', t]
		)
;

s.t. conversion_indirect {(p, m) in process_method, (d, t) in dt : m in method_indirect} :
  + sum {source in entity : (p, source) in process_source} 
    ( + v_flow[p, source, p, d, t] 
  	      * p_process_source_coefficient[p, source]
	)
	* pt_process[p, 'efficiency', t]
  =
  + sum {sink in entity : (p, sink) in process_sink} 
    ( + v_flow[p, p, sink, d, t] 
	      * p_process_sink_coefficient[p, sink]
	)
;

s.t. process_constraint_greater_than {(p, m) in process_method, c in constraint, s in sense, (d, t) in dt 
     : m in method_area && (p, c, s) in process_constraint_sense && s in sense_greater_than} :
  + sum {source in entity : (p, source) in process_source && (p, source, c) in process_node_constraint}
    ( + v_flow[p, source, p, d, t]
	      * p_process_node_constraint_coefficient[p, source, c]
	)
  + sum {sink in entity : (p, sink) in process_sink && (p, sink, c) in process_node_constraint}
    ( + v_flow[p, p, sink, d, t]
	      * p_process_node_constraint_coefficient[p, sink, c]
	)
  >=
  + ( if p in process_online then
      + p_process_constraint_constant[p, c]
        * v_online_linear[p, d, t]
    )
;
	
s.t. process_constraint_less_than {(p, m) in process_method, c in constraint, s in sense, (d, t) in dt 
     : m in method_area && (p, c, s) in process_constraint_sense && s in sense_less_than} :
  + sum {source in entity : (p, source) in process_source && (p, source, c) in process_node_constraint}
    ( + v_flow[p, source, p, d, t]
	      * p_process_node_constraint_coefficient[p, source, c]
	)
  + sum {sink in entity : (p, sink) in process_sink && (p, sink, c) in process_node_constraint}
    ( + v_flow[p, p, sink, d, t]
	      * p_process_node_constraint_coefficient[p, sink, c]
	)
  <=
  + p_process_constraint_constant[p, c]
;

s.t. process_constraint_equal {(p, m) in process_method, c in constraint, s in sense, (d, t) in dt 
     : m in method_area && (p, c, s) in process_constraint_sense && s in sense_equal} :
  + sum {source in entity : (p, source) in process_source && (p, source, c) in process_node_constraint}
    ( + v_flow[p, source, p, d, t]
	      * p_process_node_constraint_coefficient[p, source, c]
	)
  + sum {sink in entity : (p, sink) in process_sink && (p, sink, c) in process_node_constraint}
    ( + v_flow[p, p, sink, d, t]
	      * p_process_node_constraint_coefficient[p, sink, c]
	)
  =
  + p_process_constraint_constant[p, c]
;


s.t. maxToSink {(p, source, sink) in process_source_sink, (d, t) in dt 
     : (p, sink) in process_sink
} :
  + v_flow[p, source, sink, d, t]
  + sum {r in reserve : (p, r, 'up', sink) in process_reserve_upDown_node} v_reserve[p, r, 'up', sink, d, t]
  <=
  + ( if p not in process_online then
      + p_process_sink_flow_unitsize[p, sink]
        * ( + p_process_all_existing[p]
            + sum {(p, d_invest) in pd_invest : d_invest <= d} v_invest[p, d_invest] * p_entity_unitsize[p]
#            - sum {(p, d_invest) in pd_divest : d_invest <= d} v_divest[p, d_invest] * p_entity_unitsize[p]
	      )	
	)
  + ( if p in process_online then
      + p_process_sink_flow_unitsize[p, sink]
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
	 && sum{(p,m) in process_method : m in method_1way_1var} 1 
} :
  + v_flow[p, source, sink, d, t]
  >=
  - p_process_sink_flow_unitsize[p, sink] 
    * ( + p_process_all_existing[p]
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
  + p_process_source_flow_unitsize[p, source] 
    * ( + p_process_all_existing[p]
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
  + p_process_all_existing[p] / p_entity_unitsize[p]
  + sum {(p, d_invest) in pd_invest : d_invest <= d} v_invest[p, d_invest]
#   - sum {(p, d_invest) in pd_divest : d_invest <= d} v_divest[p, d_invest]
;

s.t. online__startup_Linear {p in process_online, (d, t, t_previous, t_previous_within_block) in dttt} :
  + v_startup_linear[p, d, t]
  >=
  + v_online_linear[p, d, t] 
  - v_online_linear[p, d, t_previous]
;

#s.t. minimum_downtime {p in process_online, t : p_process[u,'min_downtime'] >= step_duration[t]} :
#  + v_online_linear[p, d, t]
#  <=
#  + p_process_all_existing[p] / p_entity_unitsize[p]
#  + sum {(p, d_invest) in pd_invest : d_invest <= d} v_invest[p, d_invest]
#   - sum {(p, d_invest) in pd_divest : d_invest <= d} v_divest[p, d_invest]
#  - sum{(d, t_) in dt : t_ > t && t_ <= t + p_process[u,'min_downtime'] / time_period_duration} (
#      + v_startup[g, n, u, t_]
#	)
#;

# Minimum operational time
#s.t. minimum_uptime {(g, n, u, t) in gnut : u in unit_online && p_unittype[u,'min_uptime_h'] >= time_period_duration / 60 && t >= p_unittype[u,'min_uptime_h'] * 60 #/ time_period_duration} :
#  + v_online[g, n, u, t]
#  >=
#  + sum{t_ in time_in_use : t_ > t - 1 - p_unittype[u,'min_uptime_h'] * 60 / time_period_duration && t_ < t} (
#      + v_startup[g, n, u, t_]
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
    * ( + p_process_all_existing[p]
	    + ( if (p, d) in ed_invest then v_invest[p, d] )
#		- ( if (p, d) in ed_divest then v_divest[p, d] )
	  )
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
    * ( + p_process_all_existing[p]
	    + ( if (p, d) in ed_invest then v_invest[p, d] )
#		- ( if (p, d) in ed_divest then v_divest[p, d] )
	  )
;

s.t. maxInvest {(e, d)  in ed_invest} :
  + v_invest[e, d] * p_entity_unitsize[e] 
  <= 
  + sum{(e, m) in entity__invest_method : m not in invest_method_not_allowed && (e,d) in pd_invest} p_process[e, 'invest_max_total']
  + sum{(e, m) in entity__invest_method : m not in invest_method_not_allowed && (e,d) in nd_invest} p_node[e, 'invest_max_total']
;

solve;

param process_all_capacity{p in process, d in period_realized} :=
  + p_process_all_existing[p]
  + sum {(p, d_invest) in pd_invest : d <= d_invest} v_invest[p, d_invest].val * p_entity_unitsize[p]
;

param process_source_flow{(p, source) in process_source, d in period_realized} :=
  + sum {(p, source, sink) in process_source_sink, (d, t) in dt} v_flow[p, source, sink, d, t]
;
param process_sink_flow{(p, sink) in process_sink, d in period_realized} :=
  + sum {(p, source, sink) in process_source_sink, (d, t) in dt} v_flow[p, source, sink, d, t]
;

param r_cost_commodity{(c, n) in commodity_node, (d, t) in dt} := 
  + step_duration[d, t] * pt_commodity[c, n, 'price', t] 
      * ( 
	      + sum {(p, n, sink) in process_source_sink : sum{(p, m) in process_method : m in method_1var_per_way} 1 } 
		         v_flow[p, n, sink, d, t].val / pt_process[p, 'efficiency', t]
	      + sum {(p, n, sink) in process_source_sink : sum{(p, m) in process_method : m not in method_1var_per_way} 1 } 
		         v_flow[p, n, sink, d, t].val
	      + sum {(p, source, n) in process_source_sink} v_flow[p, source, n, d, t].val
		)
;

param r_cost_process_variable_cost{p in process, (d, t) in dt} :=
  + step_duration[d, t]
	  * sum {(p, source, sink) in process_source_sink_variable_cost} 
	         v_flow[p, source, sink, d, t].val * p_process_source_sink_variable_cost[p, source, sink]
;

param r_costPenalty_nodeState{n in nodeBalance, (d, t) in dt} :=
  + step_duration[d, t]
      * (
          + vq_state_up[n, d, t] * p_node[n, 'pq_up']
          + vq_state_down[n, d, t] * p_node[n, 'pq_down']
		)
;

param r_costPenalty_reserve_upDown{(r, ud, ng) in reserve_upDown_nodeGroup, (d, t) in dt} :=
  + step_duration[d, t]
      * (
          + vq_reserve[r, ud, ng, d, t] * p_reserve_upDown_nodeGroup[r, ud, ng, 'penalty_reserve']
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
  + sum{(r, ud, ng) in reserve_upDown_nodeGroup} r_costPenalty_reserve_upDown[r, ud, ng, d, t]
;

param r_cost_and_penalty_dt{(d,t) in dt} :=
  + r_cost_dt[d, t]
  + r_costPenalty_dt[d, t]
;



printf 'Transfer investments to the next solve...\n';
param fn_process_invested symbolic := "p_process_invested.csv";
printf 'process,p_process_invested\n' > fn_process_invested;
for {p in process : p in entityInvest} 
  {
    printf '%s,%.8g\n', p, 
	  + (if not p_model['solveFirst'] then p_process_invested[p])
	  + sum {d_invest in period_invest} v_invest[p, d_invest].val * p_entity_unitsize[p]
	>> fn_process_invested;
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
        printf '%s, %s, %s, %.8g\n', u, sink, d, process_sink_flow[u, sink, d] >> fn_unit__sinkNode__d;
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
        printf '%s, %s, %s, %.8g\n', u, source, d, process_source_flow[u, source, d] >> fn_unit__sourceNode__d;
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
 
#printf 'Write unit__node flow for time steps...\n';
#param fn_unit__node__dt symbolic := "r_unit__node__dt.csv";
#for {i in 1..1 : p_model['solveFirst']}
#  { printf 'unit,node,period,t,flow\n' > fn_unit__node__dt; }  # Print the header on the first solve
#for {p in process, (d, t) in dt : d in period_realized && d not in period_invest}
#  {
#    for {(p, source) in process_source}
#      {
#        printf '%s, %s, %s, %s, %.8g\n', p, source, d, t, v_flow[p, source, d, t].val >> fn_unit__node__d;
#      }
#    for {(p, sink) in process_sink}
#      {
#        printf '%s, %s, %s, %s, %.8g\n', p, sink, d, t, v_flow[p, source, d, t].val >> fn_unit__node__d;
#      }
#  }  

  
printf 'Write node results for periods...\n';
param fn_node__d symbolic := "r_node__d.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf 'node,period,exogenous_flow,state_change,inflow,outflow,dummy\n' > fn_node__d; }  # Print the header on the first solve
for {n in node, d in period_realized : d not in period_invest}
  {
    printf '%s, %s, %.8g, %.8g, %.8g, %.8g, %.8g\n'
		, n, d
		, (if n in nodeInflow then sum {t in time : (d, t) in dt} pt_node[n, 'inflow', t] else 0)
        , (if n in nodeState then sum {(d, t, t_previous, t_previous_within_block) in dttt}
		      (v_state[n, d, t] -  v_state[n, d, t_previous]) else 0)
        , sum {(p, source, n) in process_source_sink, t in time : (d, t) in dt} v_flow[p, source, n, d, t]
        , - sum {(p, n, sink) in process_source_sink, t in time : (d, t) in dt && sum{(p, m) in process_method : m in method_1var_per_way} 1 }
		     ( + v_flow[p, n, sink, d, t] / pt_process[p, 'efficiency', t] )
          - sum {(p, n, sink) in process_source_sink, t in time : (d, t) in dt && sum{(p, m) in process_method : m not in method_1var_per_way} 1 }
		     ( + v_flow[p, n, sink, d, t] )
        , (if n in nodeBalance then sum {t in time : (d, t) in dt} (
		      + vq_state_up[n, d, t]
              - vq_state_down[n, d, t]
          ) else 0)
		>> fn_node__d;
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
for {(r, ud, ng) in reserve_upDown_nodeGroup, (d, t) in dt}
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
  printf '\n%s\nNode', n >> resultFile;
  printf (if n in nodeInflow then ', %s' else ''), n >> resultFile;
  for {(p, source, n) in process_source_sink} {
    printf ', %s s ', source >> resultFile;
  }
  for {(p, n, sink) in process_source_sink : sum{(p, m) in process_method : m in method_1var_per_way} 1 } {
    printf ', %s = ', sink >> resultFile;
  }
  for {(p, n, sink) in process_source_sink : sum{(p, m) in process_method : m not in method_1var_per_way} 1 } {
    printf ', %s <> ', sink >> resultFile;
  }
  printf '\n' >> resultFile;
  for {(d, t) in dt} {
    printf '%s, %s', d, t >> resultFile;
	printf (if n in nodeInflow then ', %.8g' else ''), pt_node[n, 'inflow', t] >> resultFile; 
    for {(p, source, n) in process_source_sink} {
      printf ', %.8g', v_flow[p, source, n, d, t].val >> resultFile;
	}
    for {(p, n, sink) in process_source_sink : sum{(p, m) in process_method : m in method_1var_per_way} 1 } {
      printf ', %.8g', -v_flow[p, n, sink, d, t].val / pt_process[p, 'efficiency', t] >> resultFile;
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
    printf (if -v_flow[p, n, sink, d, t].val / pt_process[p, 'efficiency', t] <> d_flow_1_or_2_variable[p, n, sink, d, t]
	        then 'Test fails at %s, %s, %s, %s, %s, model value: %.8g, test value: %.8g\n' else ''),
	            p, n, sink, d, t, -v_flow[p, n, sink, d, t].val / pt_process[p, 'efficiency', t], d_flow_1_or_2_variable[p, n, sink, d, t] >> unitTestFile;
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
for {(r, ud, ng) in reserve_upDown_nodeGroup, (d, t) in dt} {
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
