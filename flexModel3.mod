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
set reserve 'r - Categories for the reservation of capacity_existing';
set period_time '(d, t) - Time steps in the time periods of the timelines in use' dimen 2;
set period 'd - Time periods in the current timelines' := setof {(d, t) in period_time} (d);
set time 't - Time steps in the current timelines' := setof {(d, t) in period_time} (t); 
set method 'm - Type of process that transfers, converts or stores commodities';
set ct_method;
set startup_method;
set debug 'flags to output debugging and test results';


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
set entity__invest_method 'the investment method applied to an entity' dimen 2 within {entity, invest_method};
set entityInvest := setof {(e, m) in entity__invest_method} (e);
set nodeBalance 'nodes that maintain a node balance' within node;
set nodeState 'nodes that have a state' within node;
set nodeInflow 'nodes that have an inflow' within node;
set nodeGroup_node 'member nodes of a particular nodeGroup' dimen 2 within {nodeGroup, node};
set process_ct_method dimen 2 within {process, ct_method};
set process_startup_method dimen 2 within {process, startup_method};
#set process_method dimen 2 within {process, method};
set methods dimen 3; 
set process_source dimen 2 within {process, entity};
set process_sink dimen 2 within {process, entity};
set process_ct_startup_method := {
    p in process, m1 in ct_method, m2 in startup_method, m in method
	: (m1, m2, m) in methods
	&& (p, m1) in process_ct_method
	&& (p, m2) in process_startup_method };
set process_method := setof {(p, m1, m2, m) in process_ct_startup_method} (p, m);
set process_source_toProcess := {
    p in process, source in node, p2 in process 
	:  p = p2 
	&& (p, source) in process_source 
	&& (p2, source) in process_source 
#	&& (p, m) in process_method;
	&& sum{(p, m) in process_method 
	         : m in method_indirect} 1};
set process_process_toSink := {
	p in process, p2 in process, sink in node 
	:  p = p2 
	&& (p, sink) in process_sink 
	&& (p2, sink) in process_sink 
#	&& (p, 'method_var') in process_method;
	&& sum{(p, m) in process_method 
	        : m in method_indirect} 1};
set process_sink_toProcess := {
    sink in node, p in process, p2 in process 
	:  p = p2 
	&& (p, sink) in process_sink 
	&& (p2, sink) in process_sink 
	&& sum{(p, m) in process_method 
	         : m in method_2way_nvar} 1};
set process_process_toSource := {
    p in process, p2 in process, source in node 
	:  p = p2 
	&& (p, source) in process_source
	&& (p2, source) in process_source
	&& sum{(p, m) in process_method 
	        : m in method_2way_nvar} 1};
set process_source_toSink := {
    p in process, source in node, sink in node
	:  (p, source) in process_source
	&& (p, sink) in process_sink
    && sum{(p, m) in process_method 
	       : m in method_direct} 1};
set process_sink_toSource := {
    p in process, sink in node, source in node
	:  (p, source) in process_source
	&& (p, sink) in process_sink
	&& sum{(p, m) in process_method 
	       : m in method_2way_2var} 1};
set process_source_sink := 
    process_source_toSink union 
	process_sink_toSource union   
	process_source_toProcess union 
	process_process_toSink union 
	process_sink_toProcess union  # Add the 'wrong' direction in 2-way processes with multiple inputs/outputs
	process_process_toSource;     # Add the 'wrong' direction in 2-way processes with multiple inputs/outputs

set reserve_nodeGroup dimen 2;
set process_reserve_source_sink dimen 4;
set process_reserve_source dimen 3;
set process_reserve_sink dimen 3;
set commodity_node dimen 2; 

set commodityParam;
set nodeParam;
set processParam;
set sourceSinkParam;
set reserveParam;

set dt dimen 2 within period_time;
set period_invest dimen 1 within period;
set peedt := {(p, source, sink) in process_source_sink, (d, t) in period_time};
set preedt := {(p, r, source, sink) in process_reserve_source_sink, (d, t) in period_time};

display period_time;

set startTime dimen 1 within time;
set startNext dimen 1 within time;
param startNext_index := sum{t in time, t_startNext in startNext : t <= t_startNext} 1;
set modelParam;
param p_model {modelParam};

param p_node {n in node, nodeParam};
param pt_node {n in node, nodeParam, t in time};
param pt_commodity {(c, n) in commodity_node, commodityParam, t in time};
param p_process {process, processParam} default 0;
param pt_process {process, processParam, time} default 0;
param p_process_source {(p, source) in process_source, sourceSinkParam} default 0;
param p_process_sink {(p, sink) in process_sink, sourceSinkParam} default 0;
param p_process_source_coefficient {(p, source) in process_source} := if (p_process_source[p, source, 'coefficient']) then p_process_source[p, source, 'coefficient'] else 1;
param p_process_sink_coefficient {(p, sink) in process_sink} := if (p_process_sink[p, sink, 'coefficient']) then p_process_sink[p, sink, 'coefficient'] else 1;
param p_process_source_flow_unitsize {(p, source) in process_source} := if (p_process_source[p, source, 'flow_unitsize']) then p_process_source[p, source, 'flow_unitsize'] else 1;
param p_process_sink_flow_unitsize {(p, sink) in process_sink} := if (p_process_sink[p, sink, 'flow_unitsize']) then p_process_sink[p, sink, 'flow_unitsize'] else 1;
param p_inflow {n in nodeInflow, t in time};
param p_reserve {r in reserve, ng in nodeGroup, reserveParam};
param pt_reserve {r in reserve, ng in nodeGroup, reserveParam, t in time};
param pq_up {n in nodeBalance};
param pq_down {n in nodeBalance};
param pq_reserve {(r, ng) in reserve_nodeGroup};
param step_jump{d in period, t in time};
param step_duration{d in period, t in time};
param pd_entity_annual{e in entityInvest, d in period_invest} :=
        + sum{m in invest_method : (e, m) in entity__invest_method && e in node && m = 'one_cost'}
            (p_node[e, 'invest_cost'] * ( p_node[e, 'interest_rate'] / (1 - (1 / (1 + p_node[e, 'invest_cost'])^p_node[e, 'lifetime'] ) ) ))
        + sum{m in invest_method : (e, m) in entity__invest_method && e in process && m = 'one_cost'}
            (p_process[e, 'invest_cost'] * ( p_process[e, 'interest_rate'] / (1 - (1 / (1 + p_process[e, 'invest_cost'])^p_process[e, 'lifetime'] ) ) ))
; 			
#       p_node[e, 'invest_cost'] * ( p_node[e, 'interest_rate'] / (1 - (1 / (1 + p_node[e, 'invest_cost'])^p_node[e, 'lifetime'] ) ) );

set ed_invest := {e in entityInvest, d in period_invest : pd_entity_annual[e, d]};
set pd_invest := {(p, d) in ed_invest : p in process};
set nd_invest := {(n, d) in ed_invest : n in node};
#    p in process, d in period_invest
#	:  p_process[p, 'invest_cost'] 
#	|| pt_process[p, 'invest_cost', d] };
#set nd_invest := {
#    n in node, d in period_invest
#	:  n in entityInvest
#	|| pd_node[n, 'invest_cost', d] };
#set ed_invest := pd_invest union nd_invest;
set ed_divest := ed_invest;

display p_model;
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
param d_reserve {(p, r, source, sink, d, t) in preedt} default 0;
param dq_reserve_up {(r, ng) in reserve_nodeGroup, (d, t) in dt} default 0;

#########################
# Read parameter data (no time series yet)
table data IN 'CSV' 'entity.csv': entity <- [entity];
table data IN 'CSV' 'process.csv': process <- [process];
table data IN 'CSV' 'node.csv' : node <- [node];
table data IN 'CSV' 'nodeGroup.csv' : nodeGroup <- [nodeGroup];
table data IN 'CSV' 'commodity.csv' : commodity <- [commodity];
table data IN 'CSV' 'reserve.csv' : reserve <- [reserve];
table data IN 'CSV' 'steps_in_timeline.csv' : period_time <- [period,step];
table data IN 'CSV' 'reserve__nodeGroup.csv' : reserve_nodeGroup <- [reserve,nodeGroup];
table data IN 'CSV' 'commodity__node.csv' : commodity_node <- [commodity,node];

table data IN 'CSV' 'nodeBalance.csv' : nodeBalance <- [nodeBalance];
table data IN 'CSV' 'nodeState.csv' : nodeState <- [nodeState];
table data IN 'CSV' 'nodeInflow.csv' : nodeInflow <- [nodeInflow];
table data IN 'CSV' 'nodeGroup__node.csv': nodeGroup_node <- [nodeGroup,node];
table data IN 'CSV' 'process__ct_method.csv' : process_ct_method <- [process,ct_method];
table data IN 'CSV' 'process__startup_method.csv' : process_startup_method <- [process,startup_method];
table data IN 'CSV' 'process__source.csv' : process_source <- [process,source];
table data IN 'CSV' 'process__sink.csv' : process_sink <- [process,sink];
table data IN 'CSV' 'process__reserve__source__sink.csv' : process_reserve_source_sink <- [process,reserve,source,sink];
table data IN 'CSV' 'entity__invest_method.csv' : entity__invest_method <- [entity,invest_method];

table data IN 'CSV' 'p_node.csv' : [node, nodeParam], p_node;
table data IN 'CSV' 'pt_node.csv' : [node, nodeParam, time], pt_node;
table data IN 'CSV' 'pt_commodity.csv' : [commodity, node, commodityParam, time], pt_commodity;
table data IN 'CSV' 'p_process.csv' : [process, processParam], p_process;
table data IN 'CSV' 'pt_process.csv' : [process, processParam, time], pt_process;
table data IN 'CSV' 'p_process_source.csv' : [process, source, sourceSinkParam], p_process_source;
table data IN 'CSV' 'p_process_sink.csv' : [process, sink, sourceSinkParam], p_process_sink;
table data IN 'CSV' 'pt_reserve.csv' : [reserve, nodeGroup, reserveParam, time], pt_reserve;
table data IN 'CSV' 'p_process_invested.csv' : [process], p_process_invested;

#table data IN 'CSV' '.csv' :  <- [];

table data IN 'CSV' 'debug.csv' : debug <- [debug];
table data IN 'CSV' 'step_jump.csv' : [period, time], step_jump;
table data IN 'CSV' 'steps_in_use.csv' : dt <- [period, step];
table data IN 'CSV' 'steps_in_use.csv' : [period, step], step_duration;
table data IN 'CSV' 'realized_period.csv' : period_invest <- [period];
#table data IN 'CSV' 'solve_start.csv' : startTime <- [start];
#table data IN 'CSV' 'solve_startNext.csv' : startNext <- [startNext];
table data IN 'CSV' 'p_model.csv' : [modelParam], p_model;

display p_model;
#display pt_entity_annual, entityInvest, p_process_source_flow_unitsize, p_process_sink_flow_unitsize;
#########################
# Variable declarations
var v_flow {(p, source, sink, d, t) in peedt};
var v_reserve {(p, r, source, sink, d, t) in preedt} >= 0;
var v_state {n in nodeState, (d, t) in dt} >= 0;
var v_online {p in process, (d, t) in dt} >=0;
var v_invest {(e, d) in ed_invest} >= 0;
var v_divest {(e, d) in ed_divest} >= 0;
var vq_state_up {n in nodeBalance, (d, t) in dt} >= 0;
var vq_state_down {n in nodeBalance, (d, t) in dt} >= 0;
var vq_reserve_up {(r, ng) in reserve_nodeGroup, (d, t) in dt} >= 0;


#########################
## Data checks 
printf 'Checking: Data for 1 variable conversions directly from source to sink (and possibly back)\n';
check {(p, m) in process_method, t in time : m in method_1var} pt_process[p, 'efficiency', t] != 0 ;

printf 'Checking: Data for 1-way conversions with an online variable\n';
check {(p, m) in process_method, t in time : m in method_1way_on} pt_process[p, 'efficiency', t] != 0;

printf 'Checking: Data for 2-way linear conversions without online variables\n';
check {(p, m) in process_method, t in time : m in method_2way_off} pt_process[p, 'efficiency', t] != 0;

display ed_invest, pd_invest;
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
      + sum {n in nodeBalance} vq_state_up[n, d, t] * p_node[n, 'pq_up']
      + sum {n in nodeBalance} vq_state_down[n, d, t] * p_node[n, 'pq_down']
      + sum {(r, ng) in reserve_nodeGroup} vq_reserve_up[r, ng, d, t] * p_reserve[r, ng, 'pq_reserve']
	) * step_duration[d, t]
  + sum {(e, d) in ed_invest} v_invest[e, d]
    * pd_entity_annual[e, d]
;

# Energy balance in each node  
s.t. nodeBalance_eq {n in nodeBalance, (d, t) in dt} :
  + (if n in nodeState then (v_state[n, d, t] -  v_state[n, d, t-step_jump[d,t]]))
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

s.t. reserveBalance_eq {(r, ng) in reserve_nodeGroup, (d, t) in dt} :
  + sum {(p, r, source, n) in process_reserve_source_sink : (ng, n) in nodeGroup_node 
          && (r, ng) in reserve_nodeGroup} 
	   v_reserve[p, r, source, n, d, t]
  + pt_reserve[r, ng, 'reservation', t]
  =
  + vq_reserve_up[r, ng, d, t]
  + sum {(p, r, n, sink) in process_reserve_source_sink 
	      : sum{(p, m) in process_method : m in method_1var_per_way} 1
		  && (ng, n) in nodeGroup_node 
		  && (r, ng) in reserve_nodeGroup} 
	   v_reserve[p, r, n, sink, d, t]
  + sum {(p, r, n, sink) in process_reserve_source_sink  
		  : sum{(p, m) in process_method : m not in method_1var_per_way} 1
		  && (ng, n) in nodeGroup_node 
		  && (r, ng) in reserve_nodeGroup} 
	   v_reserve[p, r, n, sink, d, t] / pt_process[sink, 'efficiency', t]
#  + vq_reserve_down[r, ng, d, t]
;

s.t. conversion_equality_constraint {(p, m) in process_method, (d, t) in dt : m in method_indirect} :
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

s.t. maxToSink {(p, source, sink) in process_source_sink, (d, t) in dt : (p, sink) in process_sink} :
  + v_flow[p, source, sink, d, t]
  + sum {r in reserve : (p, r, source, sink) in process_reserve_source_sink} v_reserve[p, r, source, sink, d, t]
  <=
  + p_process_sink_flow_unitsize[p, sink] 
    * ( + p_process_all_existing[p]
	    #+ p_process[p, 'existing']
	    #+ p_process[p, 'invest_forced']
        + sum {(p, d_invest) in pd_invest : d_invest <= d} v_invest[p, d_invest]
#        - sum {(p, d_invest) in pd_divest : d_invest <= d} v_divest[p, d_invest]
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
    * ( + p_process[p, 'existing']
	    + p_process[p, 'invest_forced']
        + sum {(p, d_invest) in pd_invest : d_invest <= d} v_invest[p, d_invest]
#        - sum {(p, d_invest) in pd_divest : d_invest <= d} v_divest[p, d_invest]
	  )	

;

# Special equations for the method with 2 variables presenting 2way connection between source and sink (without the process)
s.t. maxToSource {(p, source, sink) in process_source_sink, (d, t) in dt
     : (p, source) in process_source
	 && sum{(p,m) in process_method : m in method_2way_2var } 1 
} :
  + v_flow[p, sink, source, d, t]
  + sum {r in reserve : (p, r, sink, source) in process_reserve_source_sink} v_reserve[p, r, sink, source, d, t]
  <=
  + p_process_source_flow_unitsize[p, source] 
    * ( + p_process[p, 'existing']
	    + p_process[p, 'invest_forced']
        + sum {(p, d_invest) in pd_invest : d_invest <= d} v_invest[p, d_invest]
#        - sum {(p, d_invest) in pd_divest : d_invest <= d} v_divest[p, d_invest]
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


solve;


param process_all_capacity{p in process, d in period_invest} :=
  + p_process_all_existing[p]
  + sum {(p, d_invest) in pd_invest : d <= d_invest} v_invest[p, d_invest].val
;

param process_source_produce{(p, source) in process_source, d in period} :=
  + sum {(p, source, sink) in process_source_sink, (d, t) in dt} v_flow[p, source, sink, d, t]
;
param process_sink_produce{(p, sink) in process_sink, d in period} :=
  + sum {(p, source, sink) in process_source_sink, (d, t) in dt} v_flow[p, source, sink, d, t]
;

display process_all_capacity, startTime, startNext;

printf 'Transfer investments to the next solve...\n';
param fn_process_invested symbolic := "p_process_invested.csv";
printf 'process,p_process_invested\n' > fn_process_invested;
for {p in process : p in entityInvest} 
  {
    printf '%s,%.8g\n', p, 
	  + (if not p_model['solveFirst'] then p_process_invested[p])
	  + sum {d_invest in period_invest} v_invest[p, d_invest].val
	>> fn_process_invested;
  }

printf 'Write process investment results...\n';
param fn_process_investment symbolic := "r_process_investment.csv";
for {i in 1..1 : p_model['solveFirst']}
  { printf '' > fn_process_investment; }  # Clear the file on the first solve
for {(p, d_invest) in pd_invest}
  {
    printf '%s, %s, %.8g\n', p, d_invest, v_invest[p, d_invest].val >> fn_process_investment;
  }


printf 'Write unit__node results...\n';
param fn_unit__node symbolic := "r_unit__node.csv";
printf 'Unit,node,period,produce\n' > fn_unit__node;
for {p in process, d in period}
  {
    for {(p, source) in process_source}
      {
        printf '%s, %s, %s, %.8g\n', p, source, d, process_source_produce[p, source, d] >> fn_unit__node;
      }
    for {(p, sink) in process_sink}
      {
        printf '%s, %s, %s, %.8g\n', p, sink, d, process_sink_produce[p, sink, d] >> fn_unit__node;
      }
  }  
  


param resultFile symbolic := "result.csv";

printf 'Upward slack for node balance\n' >> resultFile;
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
for {(r, ng) in reserve_nodeGroup, (d, t) in dt}
  {
    printf '%s, %s, %s, %s, %.8g\n', r, ng, d, t, vq_reserve_up[r, ng, d, t].val >> resultFile;
  }

printf '\nFlow variables\n' >> resultFile;
for {(p, source, sink) in process_source_sink, (d, t) in dt}
  {
    printf '%s, %s, %s, %s, %s, %.8g\n', p, source, sink, d, t, v_flow[p, source, sink, d, t].val >> resultFile;
  }

printf '\nInvestments\n' >> resultFile;
for {(e, d_invest) in ed_invest} {
  printf '%s, %s, %.8g\n', e, d_invest, v_invest[e, d_invest].val >> resultFile;
}

printf '\nDivestments\n' >> resultFile;
for {(e, d_invest) in ed_divest} {
  printf '%s, %s, %.8g\n', e, d_invest, v_divest[e, d_invest].val >> resultFile;
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
for {(p, r, source, sink, d, t) in preedt} {
  printf (if v_reserve[p, r, source, sink, d, t].val <> d_reserve[p, r, source, sink, d, t]
          then 'Reserve test fails at %s, %s, %s, %s, %s, %s. Model value: %.8g, test value: %.8g\n' else ''),
		      p, r, source, sink, d, t, v_reserve[p, r, source, sink, d, t].val, d_reserve[p, r, source, sink, d, t] >> unitTestFile;
}
for {(r, ng) in reserve_nodeGroup, (d, t) in dt} {
  printf (if vq_reserve_up[r, ng, d, t].val <> dq_reserve_up[r, ng, d, t]
          then 'Reserve slack variable test fails at %s, %s, %s, %s. Model value: %.8g, test value: %.8g\n' else ''),
		      r, ng, d, t, vq_reserve_up[r, ng, d, t].val, dq_reserve_up[r, ng, d, t] >> unitTestFile;
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
