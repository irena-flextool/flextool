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
set node 'n - Any location where a balance needs to be maintained' within entity;
set nodeGroup 'ng - Any group of nodes that have a set of common constraints';
set commodity 'c - Stuff that is being processed';
set reserve 'r - Categories for the reservation of capacity_existing';
set time 't - Time steps in the data files'; 
set method 'm - Type of process that transfers, converts or stores commodities';
set debug 'flags to output debugging and test results';

#Individual methods
set method_1way_1variable within method;
set method_2way_1variable within method;
set method_2way_2variable within method;
set method_1way_offline within method;
set method_1way_online within method;
set method_2way_offline within method;
set method_2way_online within method;
set method_2way_online_exclusive within method;

#Method collections
set method_1variable;
set method_1way within method;
set method_2way within method;
set method_online within method;
set method_offline within method;
set method_sum_flow within method;
set method_sum_flow_2way within method;


set entity__invest_method 'the investment method applied to an entity' dimen 2 within {entity, invest_method};
set entityInvest 'nodes, units and node__nodes that can be invested in' setof {(e, m) in entity__invest_method0 (e);
set nodeBalance 'nodes that maintain a node balance' within node;
set nodeState 'nodes that have a state' within node;
set nodeInflow 'nodes that have an inflow' within node;
set nodeGroup_node 'member nodes of a particular nodeGroup' dimen 2 within {nodeGroup, node};
set process_method dimen 2 within {process, method};
set process_source dimen 2 within {process, entity};
set process_sink dimen 2 within {process, entity};
set process_source_toProcess := {
    p in process, source in node, p2 in process 
	:  p = p2 
	&& (p, source) in process_source 
	&& (p2, source) in process_source 
	&& sum{(p, m) in process_method 
	         : m in method_sum_flow} 1};
set process_process_toSink := {
	p in process, p2 in process, sink in node 
	:  p = p2 
	&& (p, sink) in process_sink 
	&& (p2, sink) in process_sink 
	&& sum{(p, m) in process_method 
	        : m in method_sum_flow} 1};
set process_sink_toProcess := {
    sink in node, p in process, p2 in process 
	:  p = p2 
	&& (p, sink) in process_sink 
	&& (p2, sink) in process_sink 
	&& sum{(p, m) in process_method 
	         : m in method_sum_flow_2way} 1};
set process_process_toSource := {
    p in process, p2 in process, source in node 
	:  p = p2 
	&& (p, source) in process_source
	&& (p2, source) in process_source
	&& sum{(p, m) in process_method 
	        : m in method_sum_flow_2way} 1};
set process_source_toSink := {
    p in process, source in node, sink in node
	:  (p, source) in process_source
	&& (p, sink) in process_sink
    && sum{(p, m) in process_method 
	       : m in method_1variable union method_2way_2variable} 1};
set process_sink_toSource := {
    p in process, sink in node, source in node
	:  (p, source) in process_source
	&& (p, sink) in process_sink
	&& sum{(p, m) in process_method 
	       : m in method_2way_2variable} 1};
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
set reserveParam;

set time_in_use := {t in time};
set time_invest dimen 1 within time;
set peet := {(p, source, sink) in process_source_sink, t in time_in_use};
set preet := {(p, r, source, sink) in process_reserve_source_sink, t in time_in_use};

param p_node {n in node, nodeParam};
param pt_node {n in node, nodeParam, t in time_in_use};
param pt_commodity {(c, n) in commodity_node, commodityParam, t in time_in_use};
param p_process {process, processParam} default 0;
param pt_process {process, processParam, time_in_use} default 0;
param p_process_source {(p, source) in process_source, processParam} default 0;
param p_process_sink {(p, sink) in process_sink, processParam} default 0;
param p_inflow {n in nodeInflow, t in time_in_use};
param p_reserve {r in reserve, ng in nodeGroup, reserveParam};
param pt_reserve {r in reserve, ng in nodeGroup, reserveParam, t in time_in_use};
param pq_up {n in nodeBalance};
param pq_down {n in nodeBalance};
param pq_reserve {(r, ng) in reserve_nodeGroup};
param t_jump{t in time};
param t_duration{t in time};
param p_entity_annual{e in entityInvest, t in time_invest} := 
        + sum{m in invest_method : (e, m) in entity__invest_method && m = 'one_cost'}
            1
; 			
#        if invest_method[e] = 'one_cost' 
#		then 1 else 2;
#       p_node[e, 'invest_cost'] * ( p_node[e, 'interest_rate'] / (1 - (1 / (1 + p_node[e, 'invest_cost'])^p_node[e, 'lifetime'] ) ) );

set pt_invest := {
    p in process, t in time_invest 
	:  p_process[p, 'invest_cost'] 
	|| pt_process[p, 'invest_cost', t] };
set nt_invest := {
    n in node, t in time_invest 
	:  n in entityInvest
	|| pt_node[n, 'invest_cost', t] };
set et_invest := pt_invest union nt_invest;
set et_divest := et_invest;

param d_obj;
param d_flow {(p, source, sink, t) in peet} default 0;
param d_flow_1_or_2_variable {(p, source, sink, t) in peet} default 0;
param d_flowInvest {(p, t) in pt_invest} default 0;
param d_reserve {(p, r, source, sink, t) in preet} default 0;
param dq_reserve_up {(r, ng) in reserve_nodeGroup, t in time_in_use} default 0;

#########################
# Read parameter data (no time series yet)
table data IN 'CSV' 'entity.csv': entity <- [entity];
table data IN 'CSV' 'process.csv': process <- [process];
table data IN 'CSV' 'node.csv' : node <- [node];
table data IN 'CSV' 'nodeGroup.csv' : nodeGroup <- [nodeGroup];
table data IN 'CSV' 'commodity.csv' : commodity <- [commodity];
table data IN 'CSV' 'reserve.csv' : reserve <- [reserve];
table data IN 'CSV' 'time.csv' : time <- [time];

table data IN 'CSV' 'entityInvest.csv' : entityInvest <- [entityInvest];
table data IN 'CSV' 'nodeBalance.csv' : nodeBalance <- [nodeBalance];
table data IN 'CSV' 'nodeState.csv' : nodeState <- [nodeState];
table data IN 'CSV' 'nodeInflow.csv' : nodeInflow <- [nodeInflow];
table data IN 'CSV' 'nodeGroup__node.csv': nodeGroup_node <- [nodeGroup,node];
table data IN 'CSV' 'process__method.csv' : process_method <- [process,method];
table data IN 'CSV' 'process__source.csv' : process_source <- [process,source];
table data IN 'CSV' 'process__sink.csv' : process_sink <- [process,sink];

table data IN 'CSV' 'p_node.csv' : [node, nodeParam], p_node;
table data IN 'CSV' 'pt_node.csv' : [node, nodeParam, time], pt_node;
table data IN 'CSV' 'pt_commodity.csv' : [commodity, node, commodityParam, time], pt_commodity;
table data IN 'CSV' 'p_process.csv' : [process, processParam], p_process;
table data IN 'CSV' 'pt_process.csv' : [process, processParam, time], pt_process;
table data IN 'CSV' 'p_process_source.csv' : [process, source, processParam], p_process_source;
table data IN 'CSV' 'p_process_sink.csv' : [process, sink, processParam], p_process_sink;
table data IN 'CSV' 'pt_reserve.csv' : [reserve, nodeGroup, reserveParam, time], pt_reserve;

#table data IN 'CSV' '.csv' :  <- [];

table data IN 'CSV' 't_jump.csv' : [time], t_jump;
table data IN 'CSV' 't_duration.csv' : [time], t_duration;
table data IN 'CSV' 'time_invest.csv' : time_invest <- [time_invest];


display et_invest, p_entity_annual;

#########################
# Variable declarations
var v_flow {(p, source, sink, t) in peet};
var v_reserve {(p, r, source, sink, t) in preet} >= 0;
var v_state {n in nodeState, t in time_in_use} >= 0;
var v_online {p in process, t in time_in_use} >=0;
var v_invest {(e, t) in et_invest} >= 0;
var v_divest {(e, t) in et_divest} >= 0;
var vq_state_up {n in nodeBalance, t in time_in_use} >= 0;
var vq_state_down {n in nodeBalance, t in time_in_use} >= 0;
var vq_reserve_up {(r, ng) in reserve_nodeGroup, t in time_in_use} >= 0;

display pt_process;
display process_method;
#########################
## Data checks 
printf 'Checking: Data for 1 variable conversions directly from source to sink (and possibly back)\n';
check {(p, m) in process_method, t in time_in_use : m in method_1variable} pt_process[p, 'efficiency', t] != 0 ;

printf 'Checking: Data for 1-way conversions with an online variable\n';
check {(p, m) in process_method, t in time_in_use : m in method_1way_online} pt_process[p, 'efficiency', t] != 0;
for {(p, m) in process_method : m in method_1way_online} {
  check {(p, s) in process_source} p_process_source[p, s, 'coefficient'] > -1e15;
  check {(p, s) in process_sink} p_process_sink[p, s, 'coefficient'] > -1e15;
}

printf 'Checking: Data for 2-way linear conversions without online variables\n';
check {(p, m) in process_method, t in time_in_use : m in method_2way_offline} pt_process[p, 'efficiency', t] != 0;
for {(p, m) in process_method : m in method_2way_offline} {
  check {(p, s) in process_source} p_process_source[p, s, 'coefficient'] > -1e15;
  check {(p, s) in process_sink} p_process_sink[p, s, 'coefficient'] > -1e15;
}


minimize total_cost: 
  + sum {t in time_in_use}
    (
      + sum {(c, n) in commodity_node} pt_commodity[c, n, 'price', t] 
          * ( 
	          + sum {(p, n, sink) in process_source_sink : sum{(p, m) in process_method : m in method_1variable union method_2way_2variable} 1 } v_flow[p, n, sink, t] / pt_process[p, 'efficiency', t]
	          + sum {(p, n, sink) in process_source_sink : sum{(p, m) in process_method : m in method diff (method_1variable union method_2way_2variable)} 1 } v_flow[p, n, sink, t]
	          + sum {(p, source, n) in process_source_sink} v_flow[p, source, n, t]
		    )
      + sum {n in nodeBalance} vq_state_up[n, t] * p_node[n, 'pq_up']
      + sum {n in nodeBalance} vq_state_down[n, t] * p_node[n, 'pq_down']
      + sum {(r, ng) in reserve_nodeGroup} vq_reserve_up[r, ng, t] * p_reserve[r, ng, 'pq_reserve']
	) * t_duration[t]
  + sum {(e, t) in et_invest} v_invest[e, t] 
##  * pt_invest_annual[e, t]
;
	  

# Energy balance in each node  
s.t. nodeBalance_eq {n in nodeBalance, t in time_in_use} :
  + (if n in nodeState then (v_state[n, t] -  v_state[n, t-1]))
  =
  + sum {(p, source, n) in process_source_sink} v_flow[p, source, n, t]
  + (if n in nodeInflow then p_inflow[n, t])
  + vq_state_up[n, t]
  - sum {(p, n, sink) in process_source_sink : sum{(p, m) in process_method : m in method_1variable union method_2way_2variable} 1 } (
       + v_flow[p, n, sink, t] / pt_process[p, 'efficiency', t]
    )		
  - sum {(p, n, sink) in process_source_sink : sum{(p, m) in process_method : m in method diff (method_1variable union method_2way_2variable)} 1 } (
       + v_flow[p, n, sink, t]
    )		
  - vq_state_down[n, t]
;

s.t. reserveBalance_eq {r in reserve, ng in nodeGroup, t in time_in_use : (r, ng) in reserve_nodeGroup} :
  + sum {(p, r, source, n) in process_reserve_source_sink : (ng, n) in nodeGroup_node 
          && (r, ng) in reserve_nodeGroup} 
	   v_reserve[p, r, source, n, t]
  + pt_reserve[r, ng, 'reservation', t]
  =
  + vq_reserve_up[r, ng, t]
  + sum {(p, r, n, sink) in process_reserve_source_sink : not (p, 'simple_method_1way') in process_method 
		  && (ng, n) in nodeGroup_node 
		  && (r, ng) in reserve_nodeGroup} 
	   v_reserve[p, r, n, sink, t]
  + sum {(p, r, n, sink) in process_reserve_source_sink :     (p, 'simple_method_1way') in process_method 
		  && (ng, n) in nodeGroup_node 
		  && (r, ng) in reserve_nodeGroup} 
	   v_reserve[p, r, n, sink, t] / pt_process[sink, 'efficiency', t]
#  + vq_reserve_down[r, ng, t]
;

s.t. conversion_equality_constraint {(p, m) in process_method, t in time_in_use : m in method_sum_flow} :
  + sum {source in entity : (p, source) in process_source} 
    ( + v_flow[p, source, p, t] 
  	      * p_process_source[p, source, 'coefficient']
	)
	* pt_process[p, 'efficiency', t]
  =
  + sum {sink in entity : (p, sink) in process_sink} 
    ( + v_flow[p, p, sink, t] 
	      * p_process_sink[p, sink, 'coefficient']
	)
;

s.t. maxToSink {(p, source, sink) in process_source_sink, t in time_in_use : (p, sink) in process_sink} :
  + v_flow[p, source, sink, t]
  + sum {r in reserve : (p, r, source, sink) in process_reserve_source_sink} v_reserve[p, r, source, sink, t]
  <=
  + p_process_sink[p, sink, 'capacity_existing']
#  + sum {(p, sink, t_invest) in pet_invest : t_invest <= t} v_flowInvest[p, sink, t_invest]
#  - sum {(p, sink, t_invest) in pet_divest : t_invest <= t} v_flowDivest[p, sink, t_invest]
;

s.t. minToSink {(p, source, sink) in process_source_sink, t in time_in_use : (p, sink) in process_sink && sum{(p,m) in process_method : m in method diff method_2way_1variable } 1 } :
  + v_flow[p, source, sink, t]
  >=
  + 0
;

display process_source_sink, process_method, process_source, process_sink;
# Special equations for the method with 2 variables presenting 2way connection between source and sink (without the process)
s.t. maxToSource {(p, source, sink) in process_source_sink, t in time_in_use : (p, source) in process_sink && sum{(p,m) in process_method : m in method_2way_2variable } 1 } :
  + v_flow[p, sink, source, t]
  + sum {r in reserve : (p, r, sink, source) in process_reserve_source_sink} v_reserve[p, r, sink, source, t]
  <=
  + p_process_sink[p, source, 'capacity_existing']
##  + sum {(p, source, t_invest) in et_invest : t_invest <= t} v_flowInvest[p, source, t_invest]
##  - sum {(p, source, t_invest) in et_divest : t_invest <= t} v_flowDivest[p, source, t_invest]
;

s.t. minToSource {(p, source, sink) in process_source_sink, t in time_in_use : (p, sink) in process_sink && sum{(p,m) in process_method : m in method_2way_2variable } 1 } :
  + v_flow[p, sink, source, t]
  >=
  + 0
;


solve;


param resultFile symbolic := "result.csv";

printf 'Upward slack for node balance\n' > resultFile;
for {n in nodeBalance, t in time_in_use}
  {
    printf '%s, %s, %.8g\n', n, t, vq_state_up[n, t].val >> resultFile;
  }

printf '\nDownward slack for node balance\n' >> resultFile;
for {n in nodeBalance, t in time_in_use}
  {
    printf '%s, %s, %.8g\n', n, t, vq_state_down[n, t].val >> resultFile;
  }

printf '\nReserve upward slack variable\n' >> resultFile;
for {r in reserve, ng in nodeGroup, t in time_in_use} 
  {
    printf '%s, %s, %s, %.8g\n', r, ng, t, vq_reserve_up[r, ng, t].val >> resultFile;
  }

printf '\nFlow variables\n' >> resultFile;
for {(p, source, sink) in process_source_sink, t in time_in_use}
  {
    printf '%s, %s, %s, %s, %.8g\n', p, source, sink, t, v_flow[p, source, sink, t].val >> resultFile;
  }

printf '\nFlow investments\n' >> resultFile;
#for {(p, n, t_invest) in pet_invest} {
#  printf '%s, %s, %s, %.8g\n', p, n, t_invest , v_flowInvest[p, n, t_invest].val >> resultFile;
#}
  


printf '\nNode balance\n' >> resultFile;
for {n in node} {
  printf '\n%s\nNode', n >> resultFile;
  printf (if n in nodeInflow then ', %s' else ''), n >> resultFile;
  for {(p, source, n) in process_source_sink} {
    printf ', %s', source >> resultFile;
  }
  for {(p, n, sink) in process_source_sink : sum{(p, m) in process_method : m in method_1variable union method_2way_2variable} 1 } {
    printf ', %s', sink >> resultFile;
  }
  for {(p, n, sink) in process_source_sink : sum{(p, m) in process_method : m in method diff (method_1variable union method_2way_2variable)} 1 } {
    printf ', %s', sink >> resultFile;
  }
  printf '\n' >> resultFile;
  for {t in time_in_use} {
    printf '%s', t >> resultFile;
	printf (if n in nodeInflow then ', %.8g' else ''), p_inflow[n, t] >> resultFile; 
    for {(p, source, n) in process_source_sink} {
      printf ', %.8g', v_flow[p, source, n, t].val >> resultFile;
	}
    for {(p, n, sink) in process_source_sink : sum{(p, m) in process_method : m in method_1variable union method_2way_2variable} 1 } {
      printf ', %.8g', -v_flow[p, n, sink, t].val / pt_process[p, 'efficiency', t] >> resultFile;
	}
    for {(p, n, sink) in process_source_sink : sum{(p, m) in process_method : m in method diff (method_1variable union method_2way_2variable)} 1 } {
      printf ', %.8g', -v_flow[p, n, sink, t].val >> resultFile;
	}
    printf '\n' >> resultFile;
  }
}


### UNIT TESTS ###
param unitTestFile symbolic := "tests/unitTests.txt";
printf (if sum{d in debug} 1 then '%s --- ' else ''), time2str(gmtime(), "%FT%TZ") >> unitTestFile;
for {d in debug} {
  printf '%s  ', d >> unitTestFile;
}
printf (if sum{d in debug} 1 then '\n\n' else '') >> unitTestFile;

## Objective test
printf (if (sum{d in debug} 1 && total_cost.val <> d_obj) 
        then 'Objective value test fails. Model value: %.8g, test value: %.8g\n' else ''), total_cost.val, d_obj >> unitTestFile;

## Testing flows from and to node
for {n in node : 'method_1way_1variable' in debug || 'mini_system' in debug} {
  printf 'Testing incoming flows of node %s\n', n >> unitTestFile;
  for {(p, source, n, t) in peet} {
    printf (if v_flow[p, source, n, t].val <> d_flow[p, source, n, t] 
	        then 'Test fails at %s, %s, %s, %s, model value: %.8g, test value: %.8g\n' else ''),
			    p, source, n, t, v_flow[p, source, n, t].val, d_flow[p, source, n, t] >> unitTestFile;
  }
  printf 'Testing outgoing flows of node %s\n', n >> unitTestFile;
  for {(p, n, sink, t) in peet : sum{(p, m) in process_method : m in method_1variable union method_2way_2variable} 1 } {
    printf (if -v_flow[p, n, sink, t].val / pt_process[p, 'efficiency', t] <> d_flow_1_or_2_variable[p, n, sink, t] 
	        then 'Test fails at %s, %s, %s, %s, model value: %.8g, test value: %.8g\n' else ''),
	            p, n, sink, t, -v_flow[p, n, sink, t].val / pt_process[p, 'efficiency', t], d_flow_1_or_2_variable[p, n, sink, t] >> unitTestFile;
  }
  for {(p, n, sink, t) in peet : sum{(p, m) in process_method : m in method diff (method_1variable union method_2way_2variable)} 1 } {
    printf (if -v_flow[p, n, sink, t].val <> d_flow[p, n, sink, t] 
	        then 'Test fails at %s, %s, %s, %s, model value: %.8g, test value: %.8g\n' else ''),
	            p, n, sink, t, -v_flow[p, n, sink, t].val, d_flow[p, n, sink, t] >> unitTestFile;
  }
  printf '\n' >> unitTestFile;
}  

display reserve_nodeGroup;
## Testing reserves
for {(p, r, source, sink, t) in preet} {
  printf (if v_reserve[p, r, source, sink, t].val <> d_reserve[p, r, source, sink, t]
          then 'Reserve test fails at %s, %s, %s, %s, %s. Model value: %.8g, test value: %.8g\n' else ''),
		      p, r, source, sink, t, v_reserve[p, r, source, sink, t].val, d_reserve[p, r, source, sink, t] >> unitTestFile;
}
for {(r, ng) in reserve_nodeGroup, t in time_in_use} {
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
