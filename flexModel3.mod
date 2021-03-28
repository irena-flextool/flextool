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
set reserve 'r - Categories for the reservation of capacity';
set time 't - Time steps in the data files'; 
set method 'm - Type of process that transfers, converts or stores commodities';
set method__direct;
set method__indirect;

set nodeBalance 'nodes that maintain a node balance';
set nodeState 'nodes that have a state';
set nodeInflow 'nodes that have an inflow';
#set process_method := {p in process, m in method};
#set commodity_node {c in commodity, n in node} dimen 2; 
set process_method dimen 2;
set read_process_source_sink dimen 3;
set process_source := setof {(p, source, sink) in read_process_source_sink} (p, source);
set process_sink := setof {(p, source, sink) in read_process_source_sink} (p, sink);
set process_source_toProcess := {p in process, source in node, p2 in process : p = p2 && (p, source) in process_source && (p2, source) in process_source && sum{(p, m) in process_method : m in method__indirect} 1};
set process_process_toSink := {p in process, p2 in process, sink in node : p = p2 && (p, sink) in process_sink && (p2, sink) in process_sink && sum{(p, m) in process_method : m in method__indirect} 1};
set process_source_toSink := {(p, source, sink) in read_process_source_sink : sum{(p, m) in process_method : m in method__direct} 1};
set process_source_sink := process_source_toSink union process_source_toProcess union process_process_toSink;

set reserve_node dimen 2;
set process_reserve_source_sink dimen 4;
set commodity_node dimen 2; 

set commodityParam;
#set nodeParam;
set processParam;

set time_in_use := {t in time};
set peet := {(p, source, sink) in process_source_sink, t in time_in_use};
set preet := {(p, r, source, sink) in process_reserve_source_sink, t in time_in_use};

param p_commodity {(c, n) in commodity_node, commodityParam, t in time_in_use};
param p_process {process, processParam, t in time_in_use} default 0;
param p_process_source {(p, source) in process_source, processParam} default 0;
param p_process_sink {(p, sink) in process_sink, processParam} default 0;
param p_process_source_sink {(p, source, sink) in process_source_sink, processParam};
param p_inflow {n in nodeInflow, time};
param pq_up {n in nodeBalance};
param pq_down {n in nodeBalance};

var v_flow {(p, source, sink, t) in peet} >= 0;
var v_reserve {(p, r, source, sink, t) in preet} >= 0;
var v_state {n in nodeState, t in time_in_use} >= 0;
var vq_up {n in nodeBalance, t in time_in_use} >= 0;
var vq_down {n in nodeBalance, t in time_in_use} >= 0;
var vq_reserve_up {(r, n) in reserve_node, t in time_in_use} >= 0;
var vq_reserve_down {(r, n) in reserve_node, t in time_in_use} >= 0;


minimize total_cost: 
  + sum {(c, n) in commodity_node, t in time_in_use} p_commodity[c, n, 'price', t] 
      * ( 
	      + sum {(p, n, sink) in process_source_sink} v_flow[p, n, sink, t]
	      + sum {(p, source, n) in process_source_sink} v_flow[p, source, n, t]
		)
  + sum {n in nodeBalance, t in time_in_use} vq_up[n, t] * pq_up[n]
  + sum {n in nodeBalance, t in time_in_use} vq_down[n, t] * pq_down[n]
;
	  

# Energy balance in each node  
s.t. nodeBalance_eq {n in nodeBalance, t in time_in_use} :
  + (if n in nodeState then (v_state[n, t] -  v_state[n, t-1]))

  + sum {(p, source, n) in process_source_sink} v_flow[p, source, n, t]
  + (if n in nodeInflow then p_inflow[n, t])
  + vq_up[n, t]
  =
  + sum {(p, n, sink) in process_source_sink : not (p, 'simple_1way') in process_method} v_flow[p, n, sink, t]
  + sum {(p, n, sink) in process_source_sink :     (p, 'simple_1way') in process_method} v_flow[p, n, sink, t] / p_process[p, 'efficiency', t]
  + vq_down[n, t]
;

s.t. reserveBalance_eq {r in reserve, n in node, t in time_in_use : (r, n) in reserve_node} :
  + sum {(p, r, source, n) in process_reserve_source_sink : (r, n) in reserve_node} v_reserve[p, r, source, n, t]
  + (if n in nodeInflow then p_inflow[n, t])
  + vq_reserve_up[r, n, t]
  =
  + sum {(p, r, n, sink) in process_reserve_source_sink : not (p, 'simple_1way') in process_method && (r, n) in reserve_node} v_reserve[p, r, n, sink, t]
  + sum {(p, r, n, sink) in process_reserve_source_sink :     (p, 'simple_1way') in process_method && (r, n) in reserve_node} v_reserve[p, r, n, sink, t] / p_process[sink, 'efficiency', t]
  + vq_reserve_down[r, n, t]
;

display process_source_toProcess, process_source_toSink, process_process_toSink, process_source_sink, process_sink;

s.t. conversion_equality_constraint {(p, m) in process_method, t in time_in_use : (p, 'complex_1way') in process_method || (p, 'losses_2way') in process_method} :
  + sum {source in entity : (p, source) in process_source} 
    ( + v_flow[p, source, p, t] 
  	      * p_process_source[p, source, 'coefficient']
	)
	* p_process[p, 'efficiency', t]
  =
  + sum {sink in entity : (p, sink) in process_sink} 
    ( + v_flow[p, p, sink, t] 
	      * p_process_sink[p, sink, 'coefficient']
	)
;

s.t. maxToSink {(p, source, sink) in process_source_sink, t in time_in_use : (p, sink) in process_sink && p_process_sink[p, sink, 'capacity']} :
  + v_flow[p, source, sink, t]
  + sum {r in reserve : (p, r, source, sink) in process_reserve_source_sink} v_reserve[p, r, source, sink, t]
  <=
  + p_process_sink[p, sink, 'capacity']
;

s.t. minToSink {p in process, source in entity, sink in entity, t in time_in_use : (p, source, sink) in process_source_sink} :
  + v_flow[p, source, sink, t]
  >=
  + 0
;

solve;

for {(p, source, sink) in process_source_sink, t in time_in_use}
  {
    printf '%s with %s to %s at %s was %.8g\n', p, source, sink, t, v_flow[p, source, sink, t].val;
  }



data;

set method := simple_1way lossless_2way complex_1way losses_2way;
set method__direct within method := simple_1way lossless_2way;
set method__indirect within method := complex_1way losses_2way;

set commodityParam := price;
#set nodeParam := ;
set processParam := coefficient efficiency capacity;


set entity := gas_node coal_node elecA elec_lossless elec_losses gas_turbine coal_unit transfer_A_Lossless transfer_A_Losses transfer_Losses_A;
set process := gas_turbine coal_unit transfer_A_Lossless transfer_A_Losses transfer_Losses_A;
set node := gas_node coal_node elecA elec_lossless elec_losses;
set commodity := gas coal elec;
set reserve := FCR_up;
set time := t01 t02 t03;

set nodeBalance := elecA elec_losses elec_lossless;
set nodeState := ;
set nodeInflow := elecA elec_losses elec_lossless;
set process_method := 
  (gas_turbine, complex_1way) 
  (coal_unit, simple_1way) 
  (transfer_A_Lossless, lossless_2way) 
  (transfer_A_Losses, losses_2way) 
  (transfer_Losses_A, losses_2way)
;
set reserve_node := ('FCR_up', 'elecA');
set read_process_source_sink := 
  (gas_turbine, gas_node, elecA) 
  (coal_unit, coal_node, elecA)
  (transfer_A_Lossless, elecA, elec_lossless) 
  (transfer_A_Losses, elecA, elec_losses) 
;
set process_reserve_source_sink := 
  (gas_turbine, FCR_up, gas_turbine, elecA)
  (coal_unit, FCR_up, coal_node, elecA)
  (transfer_A_Losses, FCR_up, transfer_A_Losses, elec_losses) 
;
set commodity_node := ('gas', 'gas_node') ('coal', 'coal_node');

param p_commodity := 
["gas", "gas_node", "price", "t01"] 10,
["gas", "gas_node", "price", "t02"] 11,
["gas", "gas_node", "price", "t03"] 12,
["coal", "coal_node", "price", "t01"] 5,
["coal", "coal_node", "price", "t02"] 6,
["coal", "coal_node", "price", "t03"] 7
;

param p_process :=
["gas_turbine", "efficiency", "t01"] 0.4
["gas_turbine", "efficiency", "t02"] 0.4
["gas_turbine", "efficiency", "t03"] 0.4
["coal_unit", "efficiency", "t01"] 0.3
["coal_unit", "efficiency", "t02"] 0.3
["coal_unit", "efficiency", "t03"] 0.3
[transfer_A_Losses, efficiency, t01] 0.98
[transfer_A_Losses, efficiency, t02] 0.98
[transfer_A_Losses, efficiency, t03] 0.98
;

param p_process_source := 
[gas_turbine, gas_node, coefficient] 1
[coal_unit, coal_node, coefficient] 1
[transfer_A_Lossless, elecA, coefficient] 1
[transfer_A_Losses, elecA, coefficient] 1
;

param p_process_sink :=
[gas_turbine, elecA, coefficient] 1
[coal_unit, elecA, coefficient] 1
[transfer_A_Lossless, elec_lossless, coefficient] 1 
[transfer_A_Losses, elec_losses, coefficient] 1
[gas_turbine, elecA, capacity] 50
[coal_unit, elecA, capacity] 30
[transfer_A_Lossless, elec_lossless, capacity] 10 
[transfer_A_Losses, elec_losses, capacity] 10
;

param p_inflow :=
[elecA, t01] -50
[elecA, t02] -60
[elecA, t03] -70
[elec_losses, t01] -20
[elec_losses, t02] -20
[elec_losses, t03] -20
[elec_lossless, t01] -20
[elec_lossless, t02] -20
[elec_lossless, t03] -20
;

param pq_up := 
"elecA" 10000,
"elec_losses" 12000,
"elec_lossless" 14000,
;

param pq_down := 
"elecA" 5000,
"elec_losses" 12000,
"elec_lossless" 14000,
;

end;

  

