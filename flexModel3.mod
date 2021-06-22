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
set time 't - Time steps in the data files'; 
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

set steps_in_use := {t in time};
set step_invest dimen 1 within time;
set peet := {(p, source, sink) in process_source_sink, t in steps_in_use};
set preet := {(p, r, source, sink) in process_reserve_source_sink, t in steps_in_use};

param p_node {n in node, nodeParam};
param pt_node {n in node, nodeParam, t in steps_in_use};
param pt_commodity {(c, n) in commodity_node, commodityParam, t in steps_in_use};
param p_process {process, processParam} default 0;
param pt_process {process, processParam, steps_in_use} default 0;
param p_process_source {(p, source) in process_source, sourceSinkParam} default 0;
param p_process_sink {(p, sink) in process_sink, sourceSinkParam} default 0;
param p_process_source_coefficient {(p, source) in process_source} := if (p_process_source[p, source, 'coefficient']) then p_process_source[p, source, 'coefficient'] else 1;
param p_process_sink_coefficient {(p, sink) in process_sink} := if (p_process_sink[p, sink, 'coefficient']) then p_process_sink[p, sink, 'coefficient'] else 1;
param p_process_source_flow_unitsize {(p, source) in process_source} := if (p_process_source[p, source, 'flow_unitsize']) then p_process_source[p, source, 'flow_unitsize'] else 1;
param p_process_sink_flow_unitsize {(p, sink) in process_sink} := if (p_process_sink[p, sink, 'flow_unitsize']) then p_process_sink[p, sink, 'flow_unitsize'] else 1;
param p_inflow {n in nodeInflow, t in steps_in_use};
param p_reserve {r in reserve, ng in nodeGroup, reserveParam};
param pt_reserve {r in reserve, ng in nodeGroup, reserveParam, t in steps_in_use};
param pq_up {n in nodeBalance};
param pq_down {n in nodeBalance};
param pq_reserve {(r, ng) in reserve_nodeGroup};
param step_jump{t in time};
param step_duration{t in time};
param pt_entity_annual{e in entityInvest, t in step_invest} := 
        + sum{m in invest_method : (e, m) in entity__invest_method && e in node && m = 'one_cost'}
            (p_node[e, 'invest_cost'] * ( p_node[e, 'interest_rate'] / (1 - (1 / (1 + p_node[e, 'invest_cost'])^p_node[e, 'lifetime'] ) ) ))
        + sum{m in invest_method : (e, m) in entity__invest_method && e in process && m = 'one_cost'}
            (p_process[e, 'invest_cost'] * ( p_process[e, 'interest_rate'] / (1 - (1 / (1 + p_process[e, 'invest_cost'])^p_process[e, 'lifetime'] ) ) ))
; 			
#       p_node[e, 'invest_cost'] * ( p_node[e, 'interest_rate'] / (1 - (1 / (1 + p_node[e, 'invest_cost'])^p_node[e, 'lifetime'] ) ) );

set et_invest := {e in entityInvest, t in step_invest : pt_entity_annual[e, t]};
set pt_invest := {(p, t) in et_invest : p in process};
set nt_invest := {(n, t) in et_invest : n in node};
#    p in process, t in step_invest 
#	:  p_process[p, 'invest_cost'] 
#	|| pt_process[p, 'invest_cost', t] };
#set nt_invest := {
#    n in node, t in step_invest 
#	:  n in entityInvest
#	|| pt_node[n, 'invest_cost', t] };
#set et_invest := pt_invest union nt_invest;
set et_divest := et_invest;

param d_obj;
param d_flow {(p, source, sink, t) in peet} default 0;
param d_flow_1_or_2_variable {(p, source, sink, t) in peet} default 0;
param d_flowInvest {(p, t) in pt_invest} default 0;
param d_reserve {(p, r, source, sink, t) in preet} default 0;
param dq_reserve_up {(r, ng) in reserve_nodeGroup, t in steps_in_use} default 0;

#########################
# Read parameter data (no time series yet)
table data IN 'CSV' 'entity.csv': entity <- [entity];
table data IN 'CSV' 'process.csv': process <- [process];
table data IN 'CSV' 'node.csv' : node <- [node];
table data IN 'CSV' 'nodeGroup.csv' : nodeGroup <- [nodeGroup];
table data IN 'CSV' 'commodity.csv' : commodity <- [commodity];
table data IN 'CSV' 'reserve.csv' : reserve <- [reserve];
table data IN 'CSV' 'steps.csv' : time <- [step];
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

#table data IN 'CSV' '.csv' :  <- [];

table data IN 'CSV' 'debug.csv' : debug <- [debug];
table data IN 'CSV' 'step_jump.csv' : [time], step_jump;
table data IN 'CSV' 'step_duration.csv' : [time], step_duration;
table data IN 'CSV' 'step_invest.csv' : step_invest <- [step_invest];

display pt_entity_annual, entityInvest, step_invest;
#########################
# Variable declarations
var v_flow {(p, source, sink, t) in peet};
var v_reserve {(p, r, source, sink, t) in preet} >= 0;
var v_state {n in nodeState, t in steps_in_use} >= 0;
var v_online {p in process, t in steps_in_use} >=0;
var v_invest {(e, t) in et_invest} >= 0;
var v_divest {(e, t) in et_divest} >= 0;
var vq_state_up {n in nodeBalance, t in steps_in_use} >= 0;
var vq_state_down {n in nodeBalance, t in steps_in_use} >= 0;
var vq_reserve_up {(r, ng) in reserve_nodeGroup, t in steps_in_use} >= 0;

display process_method;
#########################
## Data checks 
printf 'Checking: Data for 1 variable conversions directly from source to sink (and possibly back)\n';
check {(p, m) in process_method, t in steps_in_use : m in method_1var} pt_process[p, 'efficiency', t] != 0 ;

printf 'Checking: Data for 1-way conversions with an online variable\n';
check {(p, m) in process_method, t in steps_in_use : m in method_1way_on} pt_process[p, 'efficiency', t] != 0;

printf 'Checking: Data for 2-way linear conversions without online variables\n';
#check {(p, m) in process_method, t in steps_in_use : m in method_2way_off} pt_process[p, 'efficiency', t] != 0;

display commodity_node, nodeBalance, reserve_nodeGroup, et_invest;
minimize total_cost: 
  + sum {t in steps_in_use}
    (
      + sum {(c, n) in commodity_node} pt_commodity[c, n, 'price', t] 
          * ( 
	          + sum {(p, n, sink) in process_source_sink : sum{(p, m) in process_method : m in method_1var_per_way} 1 } 
			         v_flow[p, n, sink, t] / pt_process[p, 'efficiency', t]
	          + sum {(p, n, sink) in process_source_sink : sum{(p, m) in process_method : m not in method_1var_per_way} 1 } 
			         v_flow[p, n, sink, t]
	          + sum {(p, source, n) in process_source_sink} v_flow[p, source, n, t]
		    )
      + sum {n in nodeBalance} vq_state_up[n, t] * p_node[n, 'pq_up']
      + sum {n in nodeBalance} vq_state_down[n, t] * p_node[n, 'pq_down']
      + sum {(r, ng) in reserve_nodeGroup} vq_reserve_up[r, ng, t] * p_reserve[r, ng, 'pq_reserve']
	) * step_duration[t]
  + sum {(e, t) in et_invest} v_invest[e, t] 
    * pt_entity_annual[e, t]
;

# Energy balance in each node  
s.t. nodeBalance_eq {n in nodeBalance, t in steps_in_use} :
  + (if n in nodeState then (v_state[n, t] -  v_state[n, t-1]))
  =
  + sum {(p, source, n) in process_source_sink} v_flow[p, source, n, t]
  + (if n in nodeInflow then pt_node[n, 'inflow', t])
  + vq_state_up[n, t]
  - sum {(p, n, sink) in process_source_sink : sum{(p, m) in process_method : m in method_1var_per_way} 1 } (
       + v_flow[p, n, sink, t] / pt_process[p, 'efficiency', t]
    )		
  - sum {(p, n, sink) in process_source_sink : sum{(p, m) in process_method : m not in method_1var_per_way} 1 } (
       + v_flow[p, n, sink, t]
    )		
  - vq_state_down[n, t]
;

s.t. reserveBalance_eq {(r, ng) in reserve_nodeGroup, t in steps_in_use} :
  + sum {(p, r, source, n) in process_reserve_source_sink : (ng, n) in nodeGroup_node 
          && (r, ng) in reserve_nodeGroup} 
	   v_reserve[p, r, source, n, t]
  + pt_reserve[r, ng, 'reservation', t]
  =
  + vq_reserve_up[r, ng, t]
  + sum {(p, r, n, sink) in process_reserve_source_sink 
	      : sum{(p, m) in process_method : m in method_1var_per_way} 1
		  && (ng, n) in nodeGroup_node 
		  && (r, ng) in reserve_nodeGroup} 
	   v_reserve[p, r, n, sink, t]
  + sum {(p, r, n, sink) in process_reserve_source_sink  
		  : sum{(p, m) in process_method : m not in method_1var_per_way} 1
		  && (ng, n) in nodeGroup_node 
		  && (r, ng) in reserve_nodeGroup} 
	   v_reserve[p, r, n, sink, t] / pt_process[sink, 'efficiency', t]
#  + vq_reserve_down[r, ng, t]
;

s.t. conversion_equality_constraint {(p, m) in process_method, t in steps_in_use : m in method_indirect} :
  + sum {source in entity : (p, source) in process_source} 
    ( + v_flow[p, source, p, t] 
  	      * p_process_source_coefficient[p, source]
	)
	* pt_process[p, 'efficiency', t]
  =
  + sum {sink in entity : (p, sink) in process_sink} 
    ( + v_flow[p, p, sink, t] 
	      * p_process_sink_coefficient[p, sink]
	)
;
display process_source_sink, process_sink, process_source;
s.t. maxToSink {(p, source, sink) in process_source_sink, t in steps_in_use : (p, sink) in process_sink} :
  + v_flow[p, source, sink, t]
  + sum {r in reserve : (p, r, source, sink) in process_reserve_source_sink} v_reserve[p, r, source, sink, t]
  <=
  + p_process_sink_flow_unitsize[p, sink] 
    * ( + p_process[p, 'existing']
        + sum {(p, t_invest) in pt_invest : t_invest <= t} v_invest[p, t_invest]
#        - sum {(p, t_invest) in pt_divest : t_invest <= t} v_divest[p, t_invest]
	  )	
;

s.t. minToSink {(p, source, sink) in process_source_sink, t in steps_in_use 
     : (p, sink) in process_sink 
	 && sum{(p,m) in process_method : m not in method_2way_1var} 1 } :
  + v_flow[p, source, sink, t]
  >=
  + 0
;

# Special equations for the method with 2 variables presenting 2way connection between source and sink (without the process)
s.t. maxToSource {(p, source, sink) in process_source_sink, t in steps_in_use 
     : (p, source) in process_sink 
	 && sum{(p,m) in process_method : m in method_2way_2var} 1 } :
  + v_flow[p, sink, source, t]
  + sum {r in reserve : (p, r, sink, source) in process_reserve_source_sink} v_reserve[p, r, sink, source, t]
  <=
  + p_process_source_flow_unitsize[p, sink] 
    * ( + p_process[p, 'existing']
        + sum {(p, t_invest) in pt_invest : t_invest <= t} v_invest[p, t_invest]
#        - sum {(p, t_invest) in pt_divest : t_invest <= t} v_divest[p, t_invest]
	  )	
;

s.t. minToSource {(p, source, sink) in process_source_sink, t in steps_in_use 
     : (p, sink) in process_sink 
	 && sum{(p,m) in process_method : m in method_2way_2var } 1 } :
  + v_flow[p, sink, source, t]
  >=
  + 0
;


solve;


param process_MW{p in process, t_invest in step_invest} := 
  + p_process[p, 'existing']
  + p_process[p, 'invest_forced']
  + sum {(p, t) in pt_invest : t <= t_invest} v_invest[p, t].val
;

param process_produce{(p, source) in process_source, t_invest in step_invest} :=
  + sum {(p, source, sink) in process_source_sink, t in steps_in_use} v_flow[p, source, sink, t]
;

display process_MW, process_produce;
#  ( + (if (g,n,u) not in (gnu_convertOutput union gnu_output2) then abs(p_unit[g,n,u,'capacity_MW']))
#  );

printf 'Write unit results...\n';
param fn_unit symbolic := "units.csv";
printf 'Unit,Time,"Capacity (MW)","Produce (MWh)","Consume (MWh)"' > fn_unit;
printf ',"Curtail (MWh)","Utilization (\%)","Max. ramp up (p.u.)","Max. ramp down (p.u.)"' >> fn_unit;
printf ',"Reserve provision (\%)"\n' >> fn_unit;
	
for {p in process, t_invest in step_invest}
  {
    printf '%s, %s, %.8g\n', p, t_invest, process_MW[p, t_invest] >> fn_unit;
  }
  


param resultFile symbolic := "result.csv";

printf 'Upward slack for node balance\n' > resultFile;
for {n in nodeBalance, t in steps_in_use}
  {
    printf '%s, %s, %.8g\n', n, t, vq_state_up[n, t].val >> resultFile;
  }

printf '\nDownward slack for node balance\n' >> resultFile;
for {n in nodeBalance, t in steps_in_use}
  {
    printf '%s, %s, %.8g\n', n, t, vq_state_down[n, t].val >> resultFile;
  }

printf '\nReserve upward slack variable\n' >> resultFile;
for {(r, ng) in reserve_nodeGroup, t in steps_in_use} 
  {
    printf '%s, %s, %s, %.8g\n', r, ng, t, vq_reserve_up[r, ng, t].val >> resultFile;
  }

printf '\nFlow variables\n' >> resultFile;
for {(p, source, sink) in process_source_sink, t in steps_in_use}
  {
    printf '%s, %s, %s, %s, %.8g\n', p, source, sink, t, v_flow[p, source, sink, t].val >> resultFile;
  }

printf '\nInvestments\n' >> resultFile;
for {(e, t_invest) in et_invest} {
  printf '%s, %s, %.8g\n', e, t_invest , v_invest[e, t_invest].val >> resultFile;
}

printf '\nDivestments\n' >> resultFile;
for {(e, t_invest) in et_divest} {
  printf '%s, %s, %.8g\n', e, t_invest , v_divest[e, t_invest].val >> resultFile;
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
  for {t in steps_in_use} {
    printf '%s', t >> resultFile;
	printf (if n in nodeInflow then ', %.8g' else ''), pt_node[n, 'inflow', t] >> resultFile; 
    for {(p, source, n) in process_source_sink} {
      printf ', %.8g', v_flow[p, source, n, t].val >> resultFile;
	}
    for {(p, n, sink) in process_source_sink : sum{(p, m) in process_method : m in method_1var_per_way} 1 } {
      printf ', %.8g', -v_flow[p, n, sink, t].val / pt_process[p, 'efficiency', t] >> resultFile;
	}
    for {(p, n, sink) in process_source_sink : sum{(p, m) in process_method : m not in method_1var_per_way} 1 } {
      printf ', %.8g', -v_flow[p, n, sink, t].val >> resultFile;
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
  for {(p, source, n, t) in peet} {
    printf (if v_flow[p, source, n, t].val <> d_flow[p, source, n, t] 
	        then 'Test fails at %s, %s, %s, %s, model value: %.8g, test value: %.8g\n' else ''),
			    p, source, n, t, v_flow[p, source, n, t].val, d_flow[p, source, n, t] >> unitTestFile;
  }
  printf 'Testing outgoing flows of node %s\n', n >> unitTestFile;
  for {(p, n, sink, t) in peet : sum{(p, m) in process_method : m = 'method_1var' || m = 'method_2way_2var'} 1 } {
    printf (if -v_flow[p, n, sink, t].val / pt_process[p, 'efficiency', t] <> d_flow_1_or_2_variable[p, n, sink, t] 
	        then 'Test fails at %s, %s, %s, %s, model value: %.8g, test value: %.8g\n' else ''),
	            p, n, sink, t, -v_flow[p, n, sink, t].val / pt_process[p, 'efficiency', t], d_flow_1_or_2_variable[p, n, sink, t] >> unitTestFile;
  }
  for {(p, n, sink, t) in peet : sum{(p, m) in process_method : m in method && (m <> 'method_1var' || m <> 'method_2way_2var')} 1 } {
    printf (if -v_flow[p, n, sink, t].val <> d_flow[p, n, sink, t] 
	        then 'Test fails at %s, %s, %s, %s, model value: %.8g, test value: %.8g\n' else ''),
	            p, n, sink, t, -v_flow[p, n, sink, t].val, d_flow[p, n, sink, t] >> unitTestFile;
  }
  printf '\n' >> unitTestFile;
}  

## Testing reserves
for {(p, r, source, sink, t) in preet} {
  printf (if v_reserve[p, r, source, sink, t].val <> d_reserve[p, r, source, sink, t]
          then 'Reserve test fails at %s, %s, %s, %s, %s. Model value: %.8g, test value: %.8g\n' else ''),
		      p, r, source, sink, t, v_reserve[p, r, source, sink, t].val, d_reserve[p, r, source, sink, t] >> unitTestFile;
}
for {(r, ng) in reserve_nodeGroup, t in steps_in_use} {
  printf (if vq_reserve_up[r, ng, t].val <> dq_reserve_up[r, ng, t]
          then 'Reserve slack variable test fails at %s, %s, %s. Model value: %.8g, test value: %.8g\n' else ''),
		      r, ng, t, vq_reserve_up[r, ng, t].val, dq_reserve_up[r, ng, t] >> unitTestFile;
}

## Testing investments
#for {(p, n, t_invest) in pet_invest : 'invest_source_to_sink' in debug} {
#  printf 'Testing investment decisions of %s %s %s\n', p, n, t_invest >> unitTestFile;
#  printf (if v_flowInvest[p, n, t_invest].val <> d_flowInvest[p, n, t_invest]
#          then 'Test fails at %s, %s, %s, model value: %.8g, test value: %.8g\n' else ''),
#		      p, n, t_invest, v_flowInvest[p, n, t_invest].val, d_flowInvest[p, n, t_invest] >> unitTestFile;
#}
printf (if sum{d in debug} 1 then '\n\n' else '') >> unitTestFile;	  


end;
