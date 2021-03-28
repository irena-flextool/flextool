# © International Renewable Energy Agency 2018-2020

#The FlexTool is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License
#as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

#The FlexTool is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
#without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

#You should have received a copy of the GNU Lesser General Public License along with the FlexTool.  
#If not, see <https://www.gnu.org/licenses/>.

#Author: Juha Kiviluoma (2017-2020), VTT Technical Research Centre of Finland

#########################
# Fundamental sets of the model
set grid 'g - Forms of energy endogenously presented in the model';
set node 'n - Nodes where different types of energy are produced, transferred, converted and consumed';
set nodeGroup 'ng - Groups of nodes to which apply common constraints';
set unit 'u - Unit types, which also serve as units when defined in a specific node in a grid';
set unitGroup 'ug - Groupd of nodes to which apply common constraints';
set time 't - Time steps in the data files'; 
set oneMember 'pseudo set used for printing results';
set unit_ts_param 'time series parameters for units';

#########################
# Sets for specific energy resource types
set fuel 'Fuels';
set cf_profile 'Available capacity factor profiles (0-1)';
set inflow 'Inflows of energy into units/storages using inflows';
set gnuug_output1 dimen 4; # {grid, node, unit, unitGroup}
set gnu_all dimen 3;
set gnuFuel dimen 4; # {grid, node, unit, fuel}
set gnuCfprofile dimen 4; # {grid, node, unit, cf_profile}
set gnuInflow dimen 4; # {grid, node, unit, inflow}
set gnuGrid2Node2 dimen 5; # {input_grid, input_node, unit, output_grid, output_node}
set gnuOutputGrid2Node2 dimen 5; # {grid, node, unit, grid, node}
set gnn_all dimen 3;
set gnng dimen 3; # {grid, node, nodeGroup}


#########################
# Flags that define model behaviour
set masterParams 'Model level parameters';
table data IN 'CSV' 'masterParams.csv': masterParams <- [masterParams];
param p_master {masterParams};
table data IN 'CSV' 'p_master.csv': [masterParams], p_master;
param use_capacity_margin 'Flag: Additional capacity margin over the peak demand in the investment model time series' := p_master['use_capacity_margin'];
param use_inertia_limit 'Flag: Inertia limit - there has to be enough inertia in the nodeGroup at all times' := p_master['use_inertia_limit'];
param use_ramps 'A flag to indicate whether ramp constraints are in use or not' := p_master['use_ramps'];
param use_online 'A flag to indicate whether online variables (startups and min.loads) are in use' := p_master['use_online'];
param use_non_synchronous 'A flag to indicate whether to use non-synchronous constraint' := p_master['use_non_synchronous'];
param mode_invest 'A flag to indicate that investments are to be optimized' := p_master['mode_invest'];
param mode_dispatch 'A flag to indicate that dispatch is to be optimized' := p_master['mode_dispatch'];  # 0 = no dispatch, 1 = only dispatch, 2 = dispatch after investment solve
param decs_round_invest 'Number of decimals in the investment results' default 3;
param loss_of_load_penalty 'Penalty for violating the energy balance constraint' := p_master['loss_of_load_penalty'];
param curtailment_penalty 'Cost for curtailing variable generation' := p_master['curtailment_penalty'];
param loss_of_reserves_penalty 'Penalty for violating the reserve constraints' := p_master['loss_of_reserves_penalty'];
param lack_of_inertia_penalty 'Penalty for violating the reserve constraints' := p_master['lack_of_inertia_penalty'];
param lack_of_capacity_penalty 'Penalty for violating the capacity margin constraint' := p_master['lack_of_capacity_penalty'];

#########################
# Time sets and parameters defining the model temporal structure
param time_in_years 'The length of the time series in years (not time_in_use, but the whole time series)' := p_master['time_in_years'];
param time_period_duration 'The duration of a single time period in minutes' := p_master['time_period_duration'];
param p_time_dispatch {time} default 0;
param p_time_jump_dispatch {time};
param p_time_invest {time} default 0;
param p_time_jump_invest {time};
param p_time_jump {t in time} integer := (if not mode_invest then p_time_jump_dispatch[t] else p_time_jump_invest[t]);
set time_in_use := {t in time : (not mode_invest && p_time_dispatch[t]) || (mode_invest && p_time_invest[t])};
set time_in_use_ramp := {t in time_in_use : p_time_jump[t] = 1};
set time_start := {t in time_in_use : sum{t2 in time_in_use : t2 <= t} 1 = 1};
set time_end := {t in time_in_use : sum{t2 in time_in_use : t2 <= t} 1 = card(time_in_use)};
param time_in_use_in_years 'Share of year presented by time_in_use' := (sum {t in time_in_use} 1) / (sum {t2 in time} 1) * time_in_years;
param scale_to_annual 'Multiplier for scaling MWh/h results to annual MWh results' := (time_period_duration / 60) / time_in_use_in_years;
param reserve_duration 'For how long reserve needs to able to serve - affects how much storages need to have stored energy' := p_master['reserve_duration'];

#########################
# Sets that define possible parameters for the sets containing the actual data
set unitTypeParams 'Unit type parameters';
set unitParams 'Specific unit parameters';
set nodeNodeParams 'Parameters for transfer';
set fuelParams 'Fuel parameters';
set gridNodeParams 'gridNode parameters';
set unitGroupParams 'unitGroup parameters';
set nodeGroupParams 'nodeGroup parameters';
set emission 'Emissions';
set unitParamsSeparatePrint within unitParams;
set nodeNodeParamsSeparatePrint within nodeNodeParams;
#set investSlackParams := max_MW max_MWh min_MW minMWh;

#########################
# Main data items
#param p_unit {(g,n,u) in gnu_all, up in unitParams} default 0;
param p_unit {grid, node, unit, unitParams} default 0;
param p_unitGroup{unitGroup, unitGroupParams} default 0;
param p_unittype{unit, unitTypeParams} default 0;
param p_node{grid, node, gridNodeParams} default 0;
param p_nodeGroup{nodeGroup, nodeGroupParams} default 0;
param p_nodeNode{grid, node, node, nodeNodeParams} default 0;
param p_fuel{fuel, fuelParams};
param co2_cost := p_master['co2_cost'];
param print_duration := p_master['print_duration'];
param print_durationRamp  := p_master['print_durationRamp'];
param print_unit_results  := p_master['print_unit_results'];
param p_gridNodeGridNode{grid, node, grid, node, nodeNodeParams};

#########################
# Time series and related parameters
param ts_energy 'Time series for energy demand' {grid, node, time} default 0;
param ts_import 'Import/export time series' {grid, node, time} default 0;
param ts_cf 'Capacity factors for some generation technologies (mainly wind and PV)' {cf_profile, time};
param ts_inflow 'Inflow time series for some generation technologies (hydro)' {inflow, time};
param ts_reserve_n 'Reserve requirement time series for nodes' {node, time} default 0;
param ts_reserve_ng 'Reserve requirement time series for nodeGroups' {nodeGroup, time} default 0;
param ts_unit 'Time series to control the behaviour of specific units' {grid, node, unit, unit_ts_param, time} default 0;
param ts_time 'Available time periods' {time};


#########################
# Read parameter data (no time series yet)
table data IN 'CSV' 'grid.csv': grid <- [grid];
table data IN 'CSV' 'node.csv': node <- [node];
table data IN 'CSV' 'gnng.csv': gnng <- [grid,node,nodeGroup];
table data IN 'CSV' 'gridNodeParams.csv' : gridNodeParams <- [gridNodeParams];
table data IN 'CSV' 'p_node.csv' : [grid,node,gridNodeParams], p_node;
table data IN 'CSV' 'unitTypeParams.csv' : unitTypeParams <- [unitTypeParams];
table data IN 'CSV' 'unit.csv' : unit <- [unit];
table data IN 'CSV' 'p_unittype.csv' : [unit,unitTypeParams], p_unittype;
table data IN 'CSV' 'p_unit.csv' : [grid,node,unit,unitParams], p_unit;
table data IN 'CSV' 'unitParams.csv' : unitParams <- [unitParams];
table data IN 'CSV' 'gnuug_output1.csv': gnuug_output1 <- [grid,node,unit,unitGroup];
table data IN 'CSV' 'gnu_all.csv': gnu_all <- [grid,node,unit];
table data IN 'CSV' 'gnuFuel.csv': gnuFuel <- [grid,node,unit,fuel];
table data IN 'CSV' 'gnuGrid2Node2.csv': gnuGrid2Node2 <- [input_grid,input_node,unit,output_grid,output_node];
table data IN 'CSV' 'gnuOutputGrid2Node2.csv': gnuOutputGrid2Node2 <- [grid,node,unit,output2_grid,output2_node];
table data IN 'CSV' 'gnuCfprofile.csv': gnuCfprofile <- [grid,node,unit,cf_profile];
table data IN 'CSV' 'gnuInflow.csv': gnuInflow <- [grid,node,unit,inflow];
table data IN 'CSV' 'inflow.csv': inflow <- [inflow];
table data IN 'CSV' 'cf_profile.csv': cf_profile <- [cf_profile];
table data IN 'CSV' 'fuelParams.csv': fuelParams <- [fuelParams];
table data IN 'CSV' 'fuel.csv': fuel <- [fuel];
table data IN 'CSV' 'p_fuel.csv': [fuel,fuelParams], p_fuel;
table data IN 'CSV' 'gnn_all.csv': gnn_all <- [grid,node,node2];
table data IN 'CSV' 'nodeNodeParams.csv': nodeNodeParams <- [nodeNodeParams];
table data IN 'CSV' 'p_nodeNode.csv' : [grid,node,node2,nodeNodeParams], p_nodeNode;
table data IN 'CSV' 'nodeGroupParams.csv': nodeGroupParams <- [nodeGroupParams];
table data IN 'CSV' 'nodeGroup.csv': nodeGroup <- [nodeGroup];
table data IN 'CSV' 'p_nodeGroup.csv': [nodeGroup,nodeGroupParams], p_nodeGroup;
table data IN 'CSV' 'unitGroupParams.csv': unitGroupParams <- [unitGroupParams];
table data IN 'CSV' 'unitGroup.csv': unitGroup <- [unitGroup];
table data IN 'CSV' 'p_unitGroup.csv': [unitGroup,unitGroupParams], p_unitGroup;


#########################
# Combanitorial sets
set gn := {g in grid, n in node : 
     p_node[g, n, 'demand_MWh'] 
  || sum{u in unit} p_unit[g, n, u, 'capacity_MW'] 
  || sum{u in unit} p_unit[g, n, u, 'storage_MWh'] 
  || sum{u in unit} p_unit[g, n, u, 'max_invest_MW'] 
  || sum{u in unit} p_unit[g, n, u, 'max_invest_MWh'] 
  || sum{u in unit} p_unit[g, n, u, 'invested_capacity_MW'] 
  || sum{u in unit} p_unit[g, n, u, 'invested_storage_MWh'] 
  || sum {(g,n,n2) in gnu_all} p_nodeNode[g, n, n2, 'cap.rightward_MW'] 
  || sum {(g,n,n2) in gnu_all} p_nodeNode[g, n2, n, 'cap.rightward_MW'] 
  || sum {(g,n,n2) in gnu_all} p_nodeNode[g, n, n2, 'cap.leftward_MW'] 
  || sum {(g,n,n2) in gnu_all} p_nodeNode[g, n2, n, 'cap.leftward_MW'] 
  || sum {(g,n,n2) in gnu_all} p_nodeNode[g, n, n2, 'invested_capacity_MW'] 
  || sum {(g,n,n2) in gnu_all} p_nodeNode[g, n2, n, 'invested_capacity_MW']
  || sum {(g,n,n2) in gnu_all} p_nodeNode[g, n, n2, 'max_invest_MW'] 
  || sum {(g,n,n2) in gnu_all} p_nodeNode[g, n2, n, 'max_invest_MW']
};
set gn_demand := {(g,n) in gn : p_node[g,n,'demand_MWh']};
set gn_import := {(g,n) in gn : p_node[g,n,'import_MWh']};
set gn_reserve := {(g,n) in gn : not mode_invest && (p_node[g,n,'use_dynamic_reserve'] || p_node[g,n,'use_ts_reserve'])};
set ng_reserve := {ng in nodeGroup : not mode_invest && (p_nodeGroup[ng,'use_dynamic_reserve'] || p_nodeGroup[ng,'use_ts_reserve'])};
set gng := {g in grid, ng in nodeGroup : sum{(g,n,ng) in gnng} 1};
set gnu := {(g, n) in gn, u in unit : p_unit[g, n, u, 'capacity_MW'] 
                                   || p_unit[g,n,u,'invested_capacity_MW']
								   || p_unit[g, n, u, 'max_invest_MW'] && mode_invest
								   || p_unit[g, n, u, 'storage_MWh'] 
								   || p_unit[g,n,u,'invested_storage_MWh'] 
								   || p_unit[g,n,u,'max_invest_MWh'] && mode_invest
								   || sum{(g,n,u,ug) in gnuug_output1} p_unitGroup[ug,'max_invest_MW'] && mode_invest
								   || sum{(g,n,u,ug) in gnuug_output1} p_unitGroup[ug,'max_invest_MWh'] && mode_invest
								   || sum{(g,n,u,ug) in gnuug_output1} p_unitGroup[ug,'min_invest_MW'] && mode_invest
								   || sum{(g,n,u,ug) in gnuug_output1} p_unitGroup[ug,'min_invest_MWh'] && mode_invest
			};
set gnu_output2_all := setof {(g_output1, n_output1, u, g, n) in gnuOutputGrid2Node2} (g,n,u);
set gnu_output2 := setof {(g_output1, n_output1, u, g, n) in gnuOutputGrid2Node2 : (g_output1,n_output1,u) in gnu} (g,n,u);
set gnu_output1 := setof {(g, n, u, g_output2, n_output2) in gnuOutputGrid2Node2 : (g,n,u) in gnu} (g,n,u);
set gnut := {(g, n, u) in gnu, t in time_in_use};
set gnut_all := {(g, n, u) in gnu_all, t in time_in_use};
set gnt := {(g, n) in gn, t in time_in_use};
set ngt := {ng in nodeGroup, t in time_in_use};
set gnu_convertOutput 'Units that convert between energy grids' := setof {(g2, n2, u, g, n) in gnuGrid2Node2} (g, n, u);
set gnu_convertInput 'Units that convert between energy grids' := setof {(g, n, u, g2, n2) in gnuGrid2Node2} (g, n, u);
set gnu_inputs_outputs := {(g,n,u) in gnu union gnu_output2 union gnu_convertInput};
set gnu_all_inputs_outputs := {(g,n,u) in gnu_all union gnu_output2_all union gnu_convertInput};
set gnngu := {(g, n, ng) in gnng, u in unit : (g,n,u) in gnu};
set gnuug_output2 :=  {(g,n,u) in gnu_output2, ug in unitGroup : sum{(g_output1,n_output1,u,ug) in gnuug_output1} 1};
set gnuug_convertInput := {(g,n,u) in gnu_convertInput, ug in unitGroup : sum{(g_output,n_output,u,ug) in gnuug_output1} 1};
set gnuug := {(g,n,u) in gnu union gnu_output2 union gnu_convertInput, ug in unitGroup : (g,n,u,ug) in gnuug_output1 union gnuug_output2 union gnuug_convertInput};
set gnt_reserve := {(g, n, t) in gnt : ((ts_reserve_n[n, t] && p_node[g, n, 'use_ts_reserve']) || p_node[g, n, 'use_dynamic_reserve']) && not mode_invest};
set ngt_reserve := {(ng, t) in ngt : ((ts_reserve_ng[ng, t] && p_nodeGroup[ng, 'use_ts_reserve']) || p_nodeGroup[ng, 'use_dynamic_reserve']) && not mode_invest};
set gnuft := {(g, n, u) in gnu, f in fuel, t in time_in_use : (g, n, u, f) in gnuFuel && (g,n,u) in gnu};
set gnn := { (g, n, n2) in gnn_all:    p_nodeNode[g, n, n2, 'cap.rightward_MW'] 
                                    || p_nodeNode[g, n, n2, 'cap.leftward_MW'] 
  		                            || p_nodeNode[g, n, n2, 'max_invest_MW'] 
						            || p_nodeNode[g, n, n2, 'invested_capacity_MW']
           };
#set nodeNode 'Existing transmission links and links with investment possibility' := 
#  { n in node, n2 in node: sum{(g, n, n2) in gnu_all} p_nodeNode[g, n, n2, 'cap.rightward_MW'] 
#                        || sum{(g, n, n2) in gnu_all} p_nodeNode[g, n, n2, 'cap.leftward_MW'] 
#					    || sum{(g, n, n2) in gnu_all} p_nodeNode[g, n, n2, 'max_invest_MW'] 
#						|| sum{(g, n, n2) in gnu_all} p_nodeNode[g, n, n2, 'invested_capacity_MW']
#  };
		   
set gnnt := {(g, n, n2) in gnn, t in time_in_use};
set gnuGrid2Node2Inverse := {g in grid, n in node, u in unit, g2 in grid, n2 in node : (g2,n2,u,g,n) in gnuGrid2Node2};
set gnuGrid2Node2UnitGroup := {g in grid, n in node, u in unit, g2 in grid, n2 in node, ug in unitGroup : (g,n,u,g2,n2) in gnuGrid2Node2 && (g2,n2,u,ug) in gnuug};
set gnuGrid2Node2InverseUnitGroup := {g in grid, n in node, u in unit, g2 in grid, n2 in node, ug in unitGroup : (g2,n2,u,g,n) in gnuGrid2Node2 && (g,n,u,ug) in gnuug};
set gnu_bidirectional := {(g,n) in gn, u in unit : (g,n,u) in gnu || sum{(g2,n2,u,g,n) in gnuGrid2Node2} 1};
set gug := {g in grid, ug in unitGroup : sum{(g,n,u,ug) in gnuug} 1};

param p_averageDemand{(g,n) in gn_demand} := p_node[g,n,"demand_MWh"] / 8760;
param p_scaleDemand{(g,n) in gn_demand} := p_averageDemand[g,n] / ((sum{t in time_in_use} ts_energy[g, n, t]) / (sum{t in time_in_use} 1));
param p_averageImport{(g,n) in gn_import} := p_node[g,n,"import_MWh"] / 8760;
param p_scaleImport{(g,n) in gn_import} := p_averageImport[g,n] / ((sum{t in time_in_use} ts_import[g, n, t]) / (sum{t in time_in_use} 1));
param p_scaleImportFull{(g,n) in gn_import} := p_averageImport[g,n] / ((sum{t in time} ts_import[g, n, t]) / (sum{t in time} 1));
param p_sumCf{cf in cf_profile} := sum{t in time_in_use} ts_cf[cf, t];
param p_scaleCf{cf in cf_profile} := (sum{t2 in time} ts_cf[cf, t2] / (if p_sumCf[cf] then p_sumCf[cf] else 1)) * (time_in_use_in_years / time_in_years);
param ts_cf_scaled{cf in cf_profile, t in time_in_use} := ts_cf[cf, t] * p_scaleCf[cf];
param p_scaleInflow{i in inflow} := (sum{t in time_in_use} ts_inflow[i, t] / sum{t2 in time} ts_inflow[i, t2]) * (time_in_years / time_in_use_in_years);
param p_nodeNodeEff{g in grid, n in node, n2 in node : (g, n, n2) in gnn || (g, n2, n) in gnn} := 1 - p_nodeNode[g, n, n2, 'loss'] - p_nodeNode[g, n2, n, 'loss'];  # Loss has been defined only in one direction, so no double loss should result
param p_unitMaxOutput2Ratio{(g,n,u) in gnu_output1} default 0;

param p_unittypeSection{u in unit} := 
            (if p_unittype[u, "eff_at_min_load"] && use_online && not mode_invest then 
                1 / p_unittype[u, "efficiency"] - ( 1 / p_unittype[u, "efficiency"] - p_unittype[u,"min_load"] / p_unittype[u, "eff_at_min_load"] ) / (1 - p_unittype[u, "min_load"]) 
             else 
                0            
            ); 
param p_unittypeSlope{u in unit} := (if p_unittype[u, "efficiency"] then 1 / p_unittype[u, "efficiency"] - p_unittypeSection[u] else 1);
param p_unittypeRampUp{u in unit} := (if p_unittype[u, "ramp_up_p.u._per_min"] then p_unittype[u, "ramp_up_p.u._per_min"] * time_period_duration else 1);
param p_unittypeRampDown{u in unit} := (if p_unittype[u, "ramp_down_p.u._per_min"] then p_unittype[u, "ramp_down_p.u._per_min"] * time_period_duration else 1);


#########################
# Sets for units with specific constraints
set unit_startup 'Units with start-up costs' within unit := {u in unit : (p_unittype[u, 'startup_cost'] || p_unittype[u, 'min_downtime_h'] || p_unittype[u, 'min_uptime_h']) && use_online && not mode_invest };
set unit_minload 'Units that have a min. load limit' within unit := {u in unit : p_unittype[u, 'min_load'] && use_online && not mode_invest };
set unit_online 'Units with an online variable' within unit := {u in unit : (p_unittype[u, 'min_load'] || sum{(g,n,u) in gnu} p_unit[g,n,u,'use_min_online'] || p_unittype[u,'min_uptime_h'] || p_unittype[u,'min_downtime_h']) && use_online && not mode_invest };
set unit_storage 'Units with a storage' within unit := {u in unit : 
                      sum {(g, n) in gn} p_unit[g,n,u,'storage_MWh'] 
                   || sum {(g, n) in gn} p_unit[g,n,u,'invested_storage_MWh'] 
                   || sum {(g,n) in gn} p_unit[g,n,u,'max_invest_MWh'] 
                   || (sum {(g,n) in gn} p_unit[g,n,u,'max_invest_MW'] && p_unittype[u,'fixed_kW_per_kWh_ratio'])
                 };
set gnu_all_storage 'Units that could have storage in some scenarios' := {(g,n,u) in gnu_all :
					  p_unit[g,n,u,'storage_MWh'] 
                   || p_unit[g,n,u,'invested_storage_MWh'] 
                   || p_unit[g,n,u,'max_invest_MWh'] 
                   || p_unittype[u,'fixed_kW_per_kWh_ratio']
				   || p_unittype[u,'inv.cost_per_kWh']
				   || (p_unittype[u,'eff_charge'] && p_unittype[u,'efficiency'])
				 };
set unit_demand_increase 'Units that can increase demand via v_charge, but not generate' within unit := {u in unit : p_unittype[u,'efficiency'] = 0 && p_unittype[u,'eff_charge'] > 0};
set gnu_storage 'Units with a storage' := {(g, n, u) in gnu : u in unit_storage};
set gnu_storage_kW_per_kWh 'Units with a fixed ratio between kW and kWh investments' := {(g,n,u) in gnu : p_unittype[u, 'fixed_kW_per_kWh_ratio']};
set gnu_storage_charging 'Units with a storage that can be charged' := {(g, n, u) in gnu : u in unit_storage && p_unittype[u, 'eff_charge']};
set gnu_demand_increase 'Units that can increase demand' := {(g,n,u) in gnu : u in unit_demand_increase};
set gnu_fuel 'Units using a commercial fuel' := setof {(g, n, u, f) in gnuFuel : (g, n, u) in gnu} (g, n, u);
set gnu_inflow 'Units using an inflow of energy with absolute quantities' := setof {(g, n, u, i) in gnuInflow : (g, n, u) in gnu} (g, n, u);
set gnu_inflow_noStorage 'Units using an inflow of energy with absolute quantities without storage possibility' := (gnu_inflow diff gnu_storage);
set gnu_spill 'Units that can use the v_spill variable' := (gnu_storage inter gnu_inflow);
set gnu_cf 'Units using a flow of energy with relative quantities (0-1)' := setof {(g, n, u, c) in gnuCfprofile : (g,n,u) in gnu} (g, n, u);
set gnu_flow 'Inflow units without storage and units using CF without storage' := (gnu_inflow_noStorage union gnu_cf);
set gnu_output2_eq 'Units that have an equality constraint for output2 in relation to output1' := {(g,n,u) in gnu_output1 : p_unit[g,n,u,'output2_eq_coeff'] || p_unit[g,n,u,'output2_eq_constant']};
set gnu_output2_lt 'Units that have a less than constraint for output2 in relation to output1' := {(g,n,u) in gnu_output1 : p_unit[g,n,u,'output2_lt_coeff'] || p_unit[g,n,u,'output2_lt_constant']};
set gnu_output2_gt 'Units that have a greater than constraint for output2 in relation to output1' := {(g,n,u) in gnu_output1 : p_unit[g,n,u,'output2_gt_coeff'] || p_unit[g,n,u,'output2_gt_constant']};
set gnu_gen 'Units with v_gen variable (precluding output2)' := (gnu diff gnu_convertOutput diff gnu_demand_increase);
set gnu_capacityAvailability 'Units that use availability in the capacity margin' := gnu_output2 union gnu_gen diff gnu_cf diff gnu_demand_increase diff gnu_inflow_noStorage diff (gnu_storage diff gnu_spill);
set gnu_capacityGen 'Units that use generation in the capacity margin' := gnu_gen diff gnu_capacityAvailability;
set gnut_gen_output2 := {(g, n, u) in gnu_gen union gnu_output2, t in time_in_use};
set gnu_invest := {(g, n, u) in gnu :  (p_unit[g, n, u, 'max_invest_MW'] && mode_invest)
   								    || (p_unit[g,n,u,'max_invest_MWh'] && mode_invest)
                                    || (sum{(g,n,u,ug) in gnuug_output1} p_unitGroup[ug,'max_invest_MW'] && mode_invest)
								    || (sum{(g,n,u,ug) in gnuug_output1} p_unitGroup[ug,'max_invest_MWh'] && mode_invest)
								    || (sum{(g,n,u,ug) in gnuug_output1} p_unitGroup[ug,'min_invest_MW'] && mode_invest)
								    || (sum{(g,n,u,ug) in gnuug_output1} p_unitGroup[ug,'min_invest_MWh'] && mode_invest)
				  }; # && (p_unit[g,n,u,'max_invest_MW'] || p_unit[g,n,u,'max_invest_MWh'])};
set gnn_invest := {(g, n, n2) in gnn : mode_invest && p_nodeNode[g, n, n2, 'inv.cost_per_kW'] && p_nodeNode[g, n, n2, 'max_invest_MW']};
set gnu_invested := {(g, n, u) in gnu}; # && (p_unit[g,n,u,'max_invest_MW'] || p_unit[g,n,u,'max_invest_MWh'])};
set gnn_invested := {(g, n, n2) in gnn : p_nodeNode[g, n, n2, 'inv.cost_per_kW']};
#set gnu_reserve := {(g, n, u) in gnu : p_unittype[u, 'max_reserve'] && (g, n, u) not in gnu_flow && ((g,n) in gn_reserve || sum{(g, n, ng) in gnng} ng_reserve[ng]) && not mode_invest};
set gnu_reserve := {g in grid, n in node, u in unit : p_unittype[u, 'max_reserve'] && (((g,n,u) in gnu_gen) || (g,n,u) in gnu_convertInput) && ((g,n) in gn_reserve || sum{(g,n,ng) in gnng : ng in ng_reserve} 1) && not mode_invest};
set gnu_reserve_output := (gnu_reserve diff gnu_convertInput);
set gnu_reserve_input := (gnu_reserve diff gnu);
set ug2n2 := setof {(g, n, u, g2, n2) in gnuGrid2Node2} (u, g2, n2);
set ucf_profile := setof {(g, n, u, c) in gnuCfprofile} (u, c);
set uinflow := setof {(g, n, u, i) in gnuInflow} (u, i);
set gn_flow := setof {(g,n,u) in gnu_flow} (g,n);
set unit_rampUp 'Units that have ramp up constraints' within unit := {u in unit : use_ramps && ((u not in unit_storage && p_unittypeRampUp[u] < 1) || (u in unit_storage && p_unittypeRampUp[u] < 2))};
set unit_rampDown 'Units that have ramp down constraints' within unit := {u in unit : use_ramps && ((u not in unit_storage && p_unittypeRampDown[u] < 1) || (u in unit_storage && p_unittypeRampDown[u] < 2))};
set unit_flow 'Units using either CF or inflow time series' := setof {(g,n,u) in gnu_flow} (u);
set ngu := setof {(g,n,ng,u) in gnngu} (ng,u);
set ngu_VRE := {(ng,u) in ngu : u in unit_flow && u not in unit_storage};
param ts_inflowUnit{(g,n,u) in gnu_inflow, t in time_in_use} := sum{i in inflow : (g,n,u,i) in gnuInflow} (ts_inflow[i, t] * (if p_unit[g,n,u,'inflow_multiplier'] then p_unit[g,n,u,'inflow_multiplier'] else 1) / p_scaleInflow[i]);


param p_unit_max_invest_MW{(g,n,u) in gnu_invest} :=
         ( if p_unit[g,n,u,'max_invest_MW'] then
             p_unit[g,n,u,'max_invest_MW']
           else if p_unit[g,n,u,'max_invest_MWh'] && p_unittype[u,'fixed_kW_per_kWh_ratio'] then
             p_unit[g,n,u,'max_invest_MWh'] * p_unittype[u,'fixed_kW_per_kWh_ratio']
           else if sum{(g,n,u,ug) in gnuug_output1} p_unitGroup[ug,'max_invest_MW'] then
		     10e100
		   else
             0
         );
param p_unit_max_invest_MWh{(g,n,u) in gnu_invest : (g,n,u) in gnu_storage} :=
         ( if p_unit[g,n,u,'max_invest_MWh'] then
             p_unit[g,n,u,'max_invest_MWh']
           else if p_unit[g,n,u,'max_invest_MW'] && p_unittype[u,'fixed_kW_per_kWh_ratio'] then
             p_unit[g,n,u,'max_invest_MW'] / p_unittype[u,'fixed_kW_per_kWh_ratio']
           else if sum{(g,n,u,ug) in gnuug_output1} p_unitGroup[ug,'max_invest_MWh'] then
		     10e100
           else
             0
         );

param gnu_investCost_I{(g,n,u) in gnu} default 0;
param gnn_investCost_I{(g,n,n2) in gnn} default 0;

param resfolder symbolic := "results";
param import{(g,n,t) in gnt : (g,n) in gn_import} := ts_import[g,n,t] * p_scaleImport[g,n];
param demand{(g,n,t) in gnt : (g,n) in gn_demand} := ts_energy[g,n,t] * p_scaleDemand[g, n];
param p_scaleDemandForCapacityAdequacy{(g,n) in gn_demand} := max{t in time} ts_energy[g,n,t] / max{t2 in time_in_use} ts_energy[g,n,t2];
param demand_gt{g in grid, t in time_in_use} := sum{n in node : (g,n) in gn_demand} demand[g,n,t];
param pos_demand{g in grid, t in time_in_use} := 
      1 + card({t_ in time_in_use : demand_gt[g,t_] > demand_gt[g,t] or demand_gt[g,t_] = demand_gt[g,t] and t_ > t});
param ind_demand{g in grid, t__ in 1..card(time_in_use)} := sum{t in time_in_use : pos_demand[g,t] = t__} t;

param rampTimePeriods := sum{t in time_in_use : p_time_jump[t] = 1} 1 integer;
param demandRamp{(g,n) in gn_demand, t in time_in_use_ramp} := demand[g,n,t] - demand[g,n,t-p_time_jump[t]];
param demandRamp_gt{g in grid, t in time_in_use_ramp} := sum{n in node : (g,n) in gn_demand} demandRamp[g,n,t];
param pos_demandRamp{g in grid, t in time_in_use_ramp} := 
      1 + card({t_ in time_in_use_ramp : demandRamp_gt[g,t_] > demandRamp_gt[g,t] or demandRamp_gt[g,t_] = demandRamp_gt[g,t] and t_ > t});
param ind_demandRamp{g in grid, t__ in 1..rampTimePeriods} := sum{t in time_in_use_ramp : pos_demandRamp[g,t] = t__} t;


#########################
# Variables
var v_gen 'Generation (or reduced consumption)' {(g, n, u, t) in gnut_gen_output2}  >= 0;
var v_online 'Online status of units with online variable' {(g, n, u, t) in gnut : u in unit_online} >= 0;
var v_startup 'Start-up of units that have an online variable' {(g, n, u, t) in gnut : u in unit_startup} >= 0;
var v_fuelUse 'Fuel consumption in units consuming fuel' {(g, n, u, t) in gnut : (g, n, u) in gnu_fuel} >= 0;
var v_reserve 'Upward reserve procurement by units' {g in grid, n in node, u in unit, t in time_in_use : (g,n,u) in gnu_reserve} >= 0;
var v_state 'State variable for storage units' {(g, n, u, t) in gnut : (g, n, u) in gnu_storage}  >= 0;
var v_charge 'Charging variable for storages' {(g, n, u, t) in gnut : (g, n, u) in gnu_storage_charging || (g,n,u) in gnu_demand_increase} >= 0;
var v_spill 'Spill variable for storages' {(g, n, u, t) in gnut : (g, n, u) in gnu_spill} >= 0; 
var v_transfer 'Transfer of energy from node n to node n2 or from n2 to n as negative value' {g in grid, n in node, n2 in node, t in time : (g, n, n2, t) in gnnt};
var v_transferRightward 'Transfer of energy from node n to node n2' {g in grid, n in node, n2 in node, t in time : (g, n, n2, t) in gnnt} >= 0;
var v_transferLeftward 'Transfer of energy from node n2 to node n' {g in grid, n in node, n2 in node, t in time : (g, n, n2, t) in gnnt} >= 0;
var v_convert 'Convert energy from input grid/node to output grid/node' {(g, n, u, g2, n2) in gnuGrid2Node2, t in time_in_use} >=0;
var v_invest 'Investment into generation (or reduced consumption) in MW' {(g, n, u) in gnu_invest : (g, n, u) not in gnu_convertOutput} >= 0, <= p_unit_max_invest_MW[g,n,u];
var v_investStorage 'Investment into storage capacity in MWh' {(g, n, u) in gnu_invest : (g, n, u) in gnu_storage} >= 0, <= p_unit_max_invest_MWh[g,n,u];
var v_investTransfer 'Investment into transfer capacity between nodes in MW' {(g, n, n2) in gnn_invest} >= 0, <= p_nodeNode[g, n, n2, 'max_invest_MW'];
var v_investConvert 'Investment into conversion capacity between two forms of energy' {(g2, n2, u, g, n) in gnuGrid2Node2 : (g, n, u) in gnu_invest} >= 0, <= p_unit_max_invest_MW[g,n,u];
var v_slack 'Dummy variable to indicate violations of the energy balance equation' {(g, n, t) in gnt} >=0;
var v_reserveSlack_n 'Dummy variable to indicate violations in the node reserve constraint' {(g,n,t) in gnt_reserve} >=0;
var v_reserveSlack_ng 'Dummy variable to indicate violations in the nodeGroup reserve constraint' {(ng,t) in ngt_reserve} >=0;
var v_capacitySlack 'Dummy variable to indicate violations of the capacity margin equation' {(g,n,t) in gnt : use_capacity_margin && mode_invest} >=0;
var v_inertiaSlack 'Dummy variable to indicate violations of the inertia constraint' {(ng,t) in ngt : use_inertia_limit && p_nodeGroup[ng, 'inertia_limit_MWs']} >=0;
var v_demandSlack 'Dummy variable to indicate a need to increase demand to maintain node energy balance' {(g,n,t) in gnt} >=0;

#########################
# Dummy sets for the time series
set I dimen 3; # dummy set for csv
set J dimen 3; # dummy set for csv
set K dimen 2; # dummy set for csv
set L dimen 2; # dummy set for csv
set M dimen 2; # dummy set for csv
set N dimen 2; # dummy set for csv
set O dimen 5; # dummy set for csv

#########################
# read csv files

table infile7 IN 'CSV' 'ts_time.csv':
  time <- [time_in_integer], p_time_dispatch~in_use, p_time_jump_dispatch~time_jump, p_time_invest~in_use_invest, p_time_jump_invest~time_jump_invest;
table infile0 IN 'CSV' 'ts_energy.csv':
  I <- [grid,node,time], ts_energy;
table infile1 IN 'CSV' 'ts_import.csv':
  J <- [grid,node,time], ts_import;
table infile2 IN 'CSV' 'ts_cf.csv':
  K <- [cf_profile,time], ts_cf;
table infile3 IN 'CSV' 'ts_inflow.csv':
  L <- [inflow,time], ts_inflow;
table infile4 IN 'CSV' 'ts_reserve_n.csv':
  M <- [node,time], ts_reserve_n;
table infile5 IN 'CSV' 'ts_reserve_ng.csv':
  N <- [nodeGroup,time], ts_reserve_ng;
table infile6 IN 'CSV' 'ts_unit.csv':
  O <- [grid,node,unit,unit_ts_param,time], ts_unit;


#########################
# Data checks
printf 'Do all units have a unit_type (unittype names in unit_type and unit sheets should match)?\n';
check {(g,n,u) in gnu}: u in unit;
printf 'Do all units have a unitGroup (names in unitGroup and unit sheets should match)?\n';
check {(g,n,u,ug) in gnuug_output1}: ug in unitGroup;
printf 'Are node names written correctly (node names in gridNode and nodeNode sheet)?\n';
check {(g,n,n2) in gnn}: n in node;
printf 'Are node names written correctly (node names in gridNode and nodeNode sheet)?\n';
check {(g,n,n2) in gnn}: n2 in node;


##########################
# More data checks
printf 'Is there at least one selected time period in ts_time?\n';
check sum{t in time_in_use} 1 > 0;
printf 'Is min_generation always smaller than max_generation?\n';
check {(g,n,u) in gnu_gen, t in time_in_use : p_unit[g,n,u,'use_min_generation'] && p_unit[g,n,u,'use_max_generation']} ts_unit[g,n,u,'max_generation',t] - ts_unit[g,n,u,'min_generation',t] >= 0;
printf 'Is min_online always smaller than max_generation?\n';
check {(g,n,u) in gnu_gen, t in time_in_use : p_unit[g,n,u,'use_min_online'] && p_unit[g,n,u,'use_max_generation']} ts_unit[g,n,u,'max_generation',t] - ts_unit[g,n,u,'min_online',t] >= 0;


#########################
# Equations

# Objective function
minimize cost :
  + (  
      + sum {(g, n, u, t) in gnut : (g, n, u) in gnu_gen} v_gen[g,n,u,t] * p_unittype[u, 'O&M_cost_per_MWh']
      + sum {(g, n, u, t) in gnut : (g, n, u) in gnu_demand_increase} v_charge[g,n,u,t] * p_unittype[u, 'O&M_cost_per_MWh']
      + sum {(g, n, u, t) in gnut : (g, n, u) in gnu_convertOutput} sum{(g2,n2,u,g,n) in gnuGrid2Node2} (v_convert[g2,n2,u,g,n,t] * p_unittype[u, 'O&M_cost_per_MWh'])
      + sum {(g, n, u, f, t) in gnuft} v_fuelUse[g,n,u,t] * p_fuel[f, 'fuel_price_per_MWh']
      + sum {(g, n, u, f, t) in gnuft} v_fuelUse[g,n,u,t] * p_fuel[f, 'CO2_content_t_per_MWh'] * co2_cost
      + sum {(g, n, u, t) in gnut : u in unit_startup} v_startup[g,n,u,t] * p_unittype[u, 'startup_cost']
      + sum {(g, n, t) in gnt} v_slack[g, n, t] * loss_of_load_penalty 
      + sum {(g, n, t) in gnt_reserve} v_reserveSlack_n[g,n,t] *  loss_of_reserves_penalty
      + sum {(ng, t) in ngt_reserve} v_reserveSlack_ng[ng,t] *  loss_of_reserves_penalty
      + sum {(g, n, u, t) in gnut : (g,n,u) in gnu_flow} -v_gen[g,n,u,t] * curtailment_penalty
      + sum {(g, n, t) in gnt : use_capacity_margin && mode_invest} v_capacitySlack[g,n,t] * lack_of_capacity_penalty
      + sum {(ng, t) in ngt : use_inertia_limit && p_nodeGroup[ng, 'inertia_limit_MWs']} v_inertiaSlack[ng,t] * lack_of_inertia_penalty
	  + sum {(g, n, t) in gnt} v_demandSlack[g,n,t] * loss_of_load_penalty
    ) * scale_to_annual
  + sum {(g, n, u) in gnu} (p_unit[g,n,u,'capacity_MW'] + p_unit[g,n,u,'invested_capacity_MW']) * p_unittype[u, 'fixed_cost_per_kW_per_year'] * 1000
  + sum {(g, n, u) in gnu_invest : (g, n, u) not in gnu_convertOutput} v_invest[g,n,u] * (p_unittype[u, 'inv.cost_per_kW'] *  p_unittype[u, 'annuity'] + p_unittype[u, 'fixed_cost_per_kW_per_year']) * 1000
  + sum {(g2, n2, u, g, n) in gnuGrid2Node2 : (g, n, u) in gnu_invest} v_investConvert[g2,n2,u,g,n] * (p_unittype[u, 'inv.cost_per_kW'] *  p_unittype[u, 'annuity'] + p_unittype[u, 'fixed_cost_per_kW_per_year']) * 1000
  + sum {(g, n, u) in gnu} p_unit[g,n,u,'invested_capacity_MW'] * p_unittype[u, 'inv.cost_per_kW'] * 1000 * p_unittype[u, 'annuity']
  + sum {(g, n, u) in gnu_invest : u in unit_storage} v_investStorage[g,n,u] * p_unittype[u, 'inv.cost_per_kWh'] * 1000 * p_unittype[u, 'annuity']
  + sum {(g, n, u) in gnu_storage} p_unit[g,n,u,'invested_storage_MWh'] * p_unittype[u, 'inv.cost_per_kWh'] * 1000 * p_unittype[u, 'annuity']
  + sum {(g, n, n2) in gnn_invest } v_investTransfer[g, n, n2] * p_nodeNode[g, n, n2, 'inv.cost_per_kW'] * 1000 * p_nodeNode[g, n, n2, 'annuity']
  + sum {(g, n, n2) in gnn } p_nodeNode[g,n,n2,'invested_capacity_MW'] * p_nodeNode[g, n, n2, 'inv.cost_per_kW'] * 1000 * p_nodeNode[g, n, n2, 'annuity']
;  

# Energy balance in each node  
s.t. nodeBalance {(g, n, t) in gnt} :
  + sum {(g, n, u) in gnu_gen} v_gen[g, n, u, t] 
  + sum {(input_g, input_n, u, g, n) in gnuGrid2Node2} v_convert[input_g, input_n, u, g, n, t] * (if p_unit[g,n,u,'use_efficiency_time_series'] then ts_unit[g,n,u,'efficiency',t] else p_unittype[u, 'conversion_eff'])
  + sum {(g_output1, n_output1, u, g, n) in gnuOutputGrid2Node2 : (g_output1, n_output1, u) in gnu_output1} v_gen[g,n,u,t]
  + (if (g,n) in gn_import then import[g, n, t])
  + v_slack[g, n, t]
  = 
  + sum {(g,n,u) in gnu_storage_charging} v_charge[g, n, u, t]
  + sum {(g,n,u) in gnu_demand_increase} v_charge[g, n, u, t]
  + sum {n2 in node : (g,n,n2) in gnn} (v_transfer[g,n,n2,t] * p_nodeNodeEff[g,n,n2] + v_transferRightward[g,n,n2,t] * p_nodeNode[g,n,n2,'loss'])  # Applies when the current node n is on the left side of the connection
  - sum {n2 in node : (g,n2,n) in gnn} (v_transfer[g,n2,n,t] - v_transferRightward[g,n2,n,t] * p_nodeNode[g,n2,n,'loss'])  # Applies when the current node 'n' is on the right side of the connection
  + sum {(g, n, u, g_output, n_output) in gnuGrid2Node2} v_convert[g, n, u, g_output, n_output, t]
  + (if (g,n) in gn_demand then demand[g,n,t])
  + v_demandSlack[g,n,t]
;

# Generation constraints
s.t. upwardLimitNotOnline {(g, n, u, t) in gnut : (g, n, u) in gnu_gen && u not in unit_online} :
  + v_gen[g, n, u, t]
  + (if (g,n,u) in gnu_reserve_output then v_reserve[g,n,u,t])
  - (if (g,n,u) in gnu_storage_charging then v_charge[g,n,u,t])
  <=
  + (if (g,n,u) in gnu_cf then sum{(g,n,u,cf) in gnuCfprofile} ts_cf_scaled[cf,t] 
	 else if p_unit[g,n,u,'use_max_generation'] then ts_unit[g,n,u,'max_generation',t]
	 else 1
	) 
	 * (
	      + p_unit[g,n,u,'capacity_MW'] 
		  + p_unit[g,n,u,'invested_capacity_MW']
		  + (if (g,n,u) in gnu_invest then v_invest[g, n, u])
	    )
;

s.t. upwardLimitOnline {(g, n, u, t) in gnut : (g, n, u) in gnu_gen && u in unit_online} :
  + v_gen[g, n, u, t]
  + (if (g, n, u) in gnu_reserve_output then v_reserve[g, n, u, t])
  - (if (g,n,u) in gnu_storage_charging then v_charge[g, n, u, t])
  <=
  + v_online[g, n, u, t]
;

s.t. upwardLimitInflowWithoutStorage {(g, n, u, t) in gnut : (g, n, u) in gnu_inflow_noStorage} :
  + v_gen[g, n, u, t]
  + (if (g, n, u) in gnu_reserve_output then v_reserve[g, n, u, t])
  <=
  + (if (g,n,u) in gnu_inflow then ts_inflowUnit[g,n,u,t])
;

s.t. fueluse {(g, n, u, t) in gnut : (g, n, u) in gnu_fuel && (g,n,u) in gnu_gen union gnu_output2} :
  + v_fuelUse[g, n, u, t]
  =
  + v_gen[g, n, u, t] * (if p_unit[g,n,u,'use_efficiency_time_series'] then 1/ts_unit[g,n,u,'efficiency',t] else p_unittypeSlope[u])  
  + (if u in unit_online && p_unittype[u, "eff_at_min_load"] then v_online[g, n, u, t] * p_unittypeSection[u])
  + sum{(g_output1, n_output1, u, g, n) in gnuOutputGrid2Node2 : (g_output1, n_output1, u) in gnu_output1 && p_unit[g,n,u,'fueluse_increase_eq_x_output2']} v_gen[g, n, u, t] * p_unit[g,n,u,'fueluse_increase_eq_x_output2']
;

s.t. minimum_load {(g, n, u, t) in gnut : (g, n, u) in gnu_gen && u in unit_minload && u in unit_online} :
  + v_gen[g, n, u, t]
  >=
  + v_online[g, n, u, t] * p_unittype[u, 'min_load']
;

s.t. fix_unit_generation {(g,n,u,t) in gnut : (g,n,u) in gnu_gen && p_unit[g,n,u,'fix_unit_generation']} :
  + v_gen[g,n,u,t]
  =
  + ts_unit[g,n,u,'fix_generation',t] * 
    ( + p_unit[g,n,u,'capacity_MW'] 
      + p_unit[g,n,u,'invested_capacity_MW']
      + (if (g,n,u) in gnu_invest then v_invest[g, n, u])
	)
;


# Output2 unit constraints
s.t. output2_equals_output1_times_x {(g,n,u,g_output2,n_output2) in gnuOutputGrid2Node2, t in time_in_use : (g,n,u) in gnu_output2_eq} :
  + v_gen[g_output2,n_output2,u,t]
  =
  + v_gen[g,n,u,t] * p_unit[g,n,u,'output2_eq_coeff']
  + p_unit[g,n,u,'output2_eq_constant'] *
    ( + p_unit[g,n,u,'capacity_MW'] 
      + p_unit[g,n,u,'invested_capacity_MW']
      + (if (g,n,u) in gnu_invest then v_invest[g,n,u])
	)
;

s.t. output2_greater_than_output1_times_x {(g,n,u,g_output2,n_output2) in gnuOutputGrid2Node2, t in time_in_use : (g,n,u) in gnu_output2_gt} :
  + v_gen[g_output2,n_output2,u,t]
  >=
  + v_gen[g,n,u,t] * p_unit[g,n,u,'output2_gt_coeff']
  + p_unit[g,n,u,'output2_gt_constant'] *
    ( + p_unit[g,n,u,'capacity_MW'] 
      + p_unit[g,n,u,'invested_capacity_MW']
      + (if (g,n,u) in gnu_invest then v_invest[g,n,u])
	)
;

s.t. output2_less_than_output1_times_x {(g,n,u,g_output2,n_output2) in gnuOutputGrid2Node2, t in time_in_use : (g,n,u) in gnu_output2_lt} :
  + v_gen[g_output2,n_output2,u,t]
  <=
  + v_gen[g,n,u,t] * p_unit[g,n,u,'output2_lt_coeff']
  + p_unit[g,n,u,'output2_lt_constant'] *
    ( + p_unit[g,n,u,'capacity_MW'] 
      + p_unit[g,n,u,'invested_capacity_MW']
      + (if (g,n,u) in gnu_invest then v_invest[g,n,u])
	)
;

s.t. upwardLimitOutput2 {(g_output1,n_output1,u,g,n) in gnuOutputGrid2Node2, t in time_in_use : (g_output1, n_output1, u) in gnu_output1 && p_unit[g_output1,n_output1,u,'output2_max_capacity_ratio']} :
  + v_gen[g, n, u, t]
# not defined  + (if (g, n, u) in gnu_reserve then v_reserve[g, n, u, t])
# not defined  - (if (g,n,u) in gnu_storage_charging then v_charge[g, n, u, t])
  <=
  + p_unit[g_output1,n_output1,u,'output2_max_capacity_ratio'] *  
			      ( + p_unit[g_output1,n_output1,u,'capacity_MW'] 
                    + p_unit[g_output1,n_output1,u,'invested_capacity_MW']
                    + (if (g_output1,n_output1,u) in gnu_invest then v_invest[g_output1, n_output1, u])
		          )
;

# Conversion unit constraints
s.t. convertUpwardLimit {(g2, n2, u, g, n) in gnuGrid2Node2, t in time_in_use} :
  + v_convert[g2, n2, u, g, n, t]
  + (if (g,n,u) in gnu_reserve_output then v_reserve[g,n,u,t])
  <=
  + (if p_unit[g,n,u,'use_max_generation'] then ts_unit[g,n,u,'max_generation',t] else 1)
    * (
        + p_unit[g, n, u, 'capacity_MW']
        + p_unit[g, n, u, 'invested_capacity_MW']
        + (if (g, n, u) in gnu_invest then v_investConvert[g2,n2,u,g,n])
	  )
;

s.t. max_reserve_convertInput {(g, n, u, g2, n2) in gnuGrid2Node2, t in time_in_use : (g,n,u) in gnu_reserve_input} :
  + v_reserve[g, n, u, t]
  <=
  + v_convert[g, n, u, g2, n2, t] * p_unittype[u, 'max_reserve']
;

s.t. max_reserve_convertOutput {(g2, n2, u, g, n) in gnuGrid2Node2, t in time_in_use : (g,n,u) in gnu_reserve_output} :
  + v_reserve[g, n, u, t]
  <=
  + p_unittype[u,'max_reserve']
    * (if p_unit[g,n,u,'use_max_generation'] then ts_unit[g,n,u,'max_generation',t] else 1) 
	    * (
            + p_unit[g, n, u, 'capacity_MW']
            + p_unit[g, n, u, 'invested_capacity_MW']
            + (if (g, n, u) in gnu_invest then v_investConvert[g2,n2,u,g,n])
	      )
;

s.t. convert_minimum_load {(g, n, u, t) in gnut : (g, n, u) in gnu_convertOutput && u in unit_minload && u in unit_online} :
  + sum{(g2,n2,u,g,n) in gnuGrid2Node2} v_convert[g2, n2, u, g, n, t]
  >=
  + v_online[g, n, u, t] * p_unittype[u, 'min_load']
;

# Storage balance and constraints
s.t. storageBalance {(g, n, u, t) in gnut : (g, n, u) in gnu_storage && (t not in time_start || t in time_start && not (p_unit[g,n,u,'storage_start'] && p_unit[g,n,u,'storage_finish']))} :
  + v_state[g, n, u, t]
  =
  + v_state[g, n, u, t - p_time_jump[t]]
  - v_gen[g, n, u, t - p_time_jump[t]] 
      / p_unittype[u, 'efficiency'] * time_period_duration / 60
  + (if (g,n,u) in gnu_storage_charging then v_charge[g, n, u, t - p_time_jump[t]]
      * p_unittype[u, 'eff_charge'] * time_period_duration / 60
    )
  + (if (g,n,u) in gnu_inflow then ts_inflowUnit[g,n,u,t-p_time_jump[t]])
  - (if (g,n,u) in gnu_spill then v_spill[g, n, u, t - p_time_jump[t]]  * time_period_duration / 60)
  - (if p_unittype[u,'self_discharge_loss'] then v_state[g, n, u, t - p_time_jump[t]] * p_unittype[u,'self_discharge_loss'] * time_period_duration / 60)
;

s.t. stateLimit {(g, n, u, t) in gnut : (g, n, u) in gnu_storage} :
  + v_state[g, n, u, t]
  <=
  + p_unit[g,n,u,'storage_MWh']
  + p_unit[g,n,u,'invested_storage_MWh']
  + (if (g,n,u) in gnu_invest then v_investStorage[g, n, u])
;

s.t. stateStart {(g, n, u, t) in gnut : (g, n, u) in gnu_storage && t in time_start && p_unit[g,n,u,'storage_start']} :
  + v_state[g,n,u,t]
  =
  + p_unit[g,n,u,'storage_start']
      * ( + p_unit[g, n, u, 'storage_MWh']
          + p_unit[g, n, u, 'invested_storage_MWh']
          + (if (g,n,u) in gnu_invest then v_investStorage[g, n, u])
        )
;

s.t. stateEnd {(g, n, u, t) in gnut : (g, n, u) in gnu_storage && t in time_end && p_unit[g,n,u,'storage_finish']} :
  + v_state[g,n,u,t]
  =
  + p_unit[g,n,u,'storage_finish']
      * ( + p_unit[g, n, u, 'storage_MWh']
          + p_unit[g, n, u, 'invested_storage_MWh']
          + (if (g,n,u) in gnu_invest then v_investStorage[g, n, u])
        )
;

s.t. min_online_ts {(g,n,u,t) in gnut : p_unit[g,n,u,'use_min_online'] && u in unit_online} :
  + v_online[g,n,u,t]
  >=
  + ts_unit[g,n,u,'min_online',t]
    * (
        + p_unit[g, n, u, 'capacity_MW']
        + p_unit[g, n, u, 'invested_capacity_MW']
        + (if (g, n, u) in gnu_invest then v_invest[g,n,u])
        + (sum{(g2,n2,u,g,n) in gnuGrid2Node2} v_investConvert[g2,n2,u,g,n])
	  )
;

s.t. min_generation_ts {(g,n,u,t) in gnut : p_unit[g,n,u,'use_min_generation']} :
  + v_gen[g,n,u,t]
  >=
  + ts_unit[g,n,u,'min_generation',t]
    * (
        + p_unit[g, n, u, 'capacity_MW']
        + p_unit[g, n, u, 'invested_capacity_MW']
        + (if (g, n, u) in gnu_invest then v_invest[g,n,u])
	  )
;

s.t. min_generation_ts_convert {(g_input, n_input, u, g, n) in gnuGrid2Node2, t in time_in_use : p_unit[g,n,u,'use_min_generation']} :
  + v_convert[g_input,n_input,u,g,n,t]
  >=
  + ts_unit[g,n,u,'min_generation',t]
    * (
        + p_unit[g, n, u, 'capacity_MW']
        + p_unit[g, n, u, 'invested_capacity_MW']
        + (if (g,n,u) in gnu_invest then v_investConvert[g_input,n_input,u,g,n])
	  )
;


s.t. chargeLimitNotOnline {(g, n, u, t) in gnut : ((g, n, u) in gnu_storage_charging || (g,n,u) in gnu_demand_increase) && (g, n, u) not in gnu_flow && u not in unit_online} :
  + v_charge[g, n, u, t]
  <=
  + abs(p_unit[g, n, u, 'capacity_MW'])
  + abs(p_unit[g, n, u, 'invested_capacity_MW'])
  + (if (g,n,u) in gnu_invest then v_invest[g, n, u])
;

s.t. chargeLimitOnline {(g, n, u, t) in gnut : ((g, n, u) in gnu_storage_charging || (g,n,u) in gnu_demand_increase) && (g, n, u) not in gnu_flow && u in unit_online} :
  + v_charge[g, n, u, t]
  <=
  + v_online[g, n, u, t]
;

s.t. genLimitStoragesNotOnline {(g, n, u, t) in gnut : ((g, n, u) in gnu_storage_charging) && (g, n, u) not in gnu_flow && u not in unit_online} :
  + v_gen[g, n, u, t]
  <=
  + abs(p_unit[g, n, u, 'capacity_MW'])
  + abs(p_unit[g, n, u, 'invested_capacity_MW'])
  + (if (g,n,u) in gnu_invest then v_invest[g, n, u])
;

s.t. genLimitStoragesOnline {(g, n, u, t) in gnut : ((g, n, u) in gnu_storage_charging) && (g, n, u) not in gnu_flow && u in unit_online} :
  + v_gen[g, n, u, t]
  <=
  + v_online[g, n, u, t]
;

s.t. reservesFromDemandIncrease {(g,n,u,t) in gnut: (g,n,u) in gnu_demand_increase && (g,n,u) in gnu_reserve_output} :
  + v_reserve[g,n,u,t]
  <=
  + v_charge[g,n,u,t] * p_unittype[u, 'max_reserve']
;

s.t. investStorage_kW_per_kWh {(g, n, u) in gnu_storage_kW_per_kWh : (g, n, u) in gnu_invest} :
  + v_invest[g, n, u]
  =
  + v_investStorage[g, n, u] * p_unittype[u, 'fixed_kW_per_kWh_ratio']
;

# Ramp constraints
s.t. ramp_up {(g, n, u, t) in gnut : use_ramps && (g, n, u) in gnu_gen union gnu_demand_increase && u in unit_rampUp && p_time_jump[t] = 1} :
  + v_gen[g, n, u, t]
  - (if ((g,n,u) in gnu_storage_charging || (g,n,u) in gnu_demand_increase) then v_charge[g, n, u, t])
  + (if (g,n,u) in gnu_reserve_output then v_reserve[g,n,u,t])
  <=
  + v_gen[g, n, u, t-p_time_jump[t]] 
  - (if ((g,n,u) in gnu_storage_charging || (g,n,u) in gnu_demand_increase) then v_charge[g, n, u, t-p_time_jump[t]])
  + p_unittypeRampUp[u] 
      * ( p_unit[g,n,u,'capacity_MW'] + p_unit[g,n,u,'invested_capacity_MW'] + (if (g,n,u) in gnu_invest then v_invest[g,n,u]) )
;

s.t. ramp_down {(g, n, u, t) in gnut : use_ramps && (g, n, u) in gnu_gen union gnu_demand_increase && u in unit_rampDown && p_time_jump[t] = 1} :
  + v_gen[g, n, u, t]
  - (if ((g,n,u) in gnu_storage_charging || (g,n,u) in gnu_demand_increase) then v_charge[g, n, u, t])
  >=
  + v_gen[g, n, u, t-p_time_jump[t]] 
  - (if ((g,n,u) in gnu_storage_charging || (g,n,u) in gnu_demand_increase) then v_charge[g, n, u, t-p_time_jump[t]])
  - p_unittypeRampDown[u]
      * ( p_unit[g,n,u,'capacity_MW'] + p_unit[g,n,u,'invested_capacity_MW'] + (if (g,n,u) in gnu_invest then v_invest[g,n,u]) )
;

s.t. ramp_up_convert {(g, n, u, t) in gnut : use_ramps && (g, n, u) in gnu_convertOutput && u in unit_rampUp && p_time_jump[t] = 1} :
  + sum{(g_input,n_input,u,g,n) in gnuGrid2Node2} v_convert[g_input, n_input, u, g, n, t]
  <=
  + sum{(g_input,n_input,u,g,n) in gnuGrid2Node2} v_convert[g_input, n_input, u, g, n, t-p_time_jump[t]] 
  + p_unittypeRampUp[u] 
      * ( p_unit[g,n,u,'capacity_MW'] + p_unit[g,n,u,'invested_capacity_MW'] + (if (g,n,u) in gnu_invest then v_invest[g,n,u]) )
;

s.t. ramp_down_convert {(g, n, u, t) in gnut : use_ramps && (g, n, u) in gnu_convertOutput && u in unit_rampDown && p_time_jump[t] = 1} :
  + sum{(g_input,n_input,u,g,n) in gnuGrid2Node2} v_convert[g_input, n_input, u, g, n, t]
  >=
  + sum{(g_input,n_input,u,g,n) in gnuGrid2Node2} v_convert[g_input, n_input, u, g, n, t-p_time_jump[t]] 
  - p_unittypeRampDown[u] 
      * ( p_unit[g,n,u,'capacity_MW'] + p_unit[g,n,u,'invested_capacity_MW'] + (if (g,n,u) in gnu_invest then v_invest[g,n,u]) )
;

# Reserve constraints
s.t. reserveNeedStatic_n {(g, n, t) in gnt_reserve : mode_dispatch && p_node[g, n, 'use_ts_reserve']} :
  + sum{u in unit : (g, n, u) in gnu_reserve} v_reserve[g, n, u, t]
  + v_reserveSlack_n[g,n,t]
  >=
  + ts_reserve_n[n, t] 
;

s.t. reserveNeedStatic_ng {(ng, t) in ngt_reserve : mode_dispatch && p_nodeGroup[ng, 'use_ts_reserve']} :
  + sum{(g, n, u) in gnu_reserve : (g,n,ng) in gnng} v_reserve[g, n, u, t]
  + v_reserveSlack_ng[ng,t]
  >=
  + ts_reserve_ng[ng, t]
;

s.t. reserveNeedDynamic_n {(g, n, t) in gnt_reserve : mode_dispatch && p_node[g, n, 'use_dynamic_reserve'] && sum{(g,n,u) in gnu_flow} p_unit[g,n,u,'reserve_increase_ratio']} :
  + sum{u in unit : (g, n, u) in gnu_reserve} v_reserve[g, n, u, t]
  + v_reserveSlack_n[g,n,t]
  >=
  + sum{(g,n,u) in gnu_flow} (v_gen[g,n,u,t] * p_unit[g, n, u, 'reserve_increase_ratio'])
;

s.t. reserveNeedDynamic_ng {(ng, t) in ngt_reserve : mode_dispatch && p_nodeGroup[ng, 'use_dynamic_reserve'] && sum{(g,n,u) in gnu_flow : (g,n,ng) in gnng} p_unit[g,n,u,'reserve_increase_ratio']} :
  + sum{(g, n, u) in gnu_reserve : (g,n,ng) in gnng} v_reserve[g, n, u, t]
  + v_reserveSlack_ng[ng,t]
  >=
  + sum{(g,n,u) in gnu_flow : (g,n,ng) in gnng} (v_gen[g,n,u,t] * p_unit[g, n, u, 'reserve_increase_ratio'])
;

s.t. reserveConstraintOnline {(g, n, u, t) in gnut : (g, n, u) in gnu_gen inter gnu_reserve && u in unit_online} :
  + v_reserve[g, n, u, t]
  <=
  + v_online[g, n, u, t] * p_unittype[u, 'max_reserve']
;

s.t. reserveConstraintNotOnline {(g, n, u, t) in gnut : (g, n, u) in gnu_gen inter gnu_reserve && u not in unit_online} :
  + v_reserve[g, n, u, t]
  <=
  + (if not p_unit[g,n,u,'use_max_generation'] then 
	  + p_unittype[u, 'max_reserve']
		* (
          + p_unit[g, n, u, 'capacity_MW'] 
          + p_unit[g, n, u, 'invested_capacity_MW'] 
          + (if (g,n,u) in gnu_invest then v_invest[g, n, u])
        )
	 else
	  + ts_unit[g,n,u,'max_generation',t]
	    * p_unittype[u, 'max_reserve']
		* (
          + p_unit[g, n, u, 'capacity_MW'] 
          + p_unit[g, n, u, 'invested_capacity_MW'] 
          + (if (g,n,u) in gnu_invest then v_invest[g, n, u])
        )
	)
;

s.t. storageReserveConstraint {(g, n, u, t) in gnut : (g, n, u) in gnu_reserve && (g,n,u) in gnu_storage} :
  + v_reserve[g, n, u, t] 
  <=
  + v_state[g, n, u, t] / reserve_duration 
;

# Capacity margin for investment phase for nodes
s.t. capacityMarginNode {(g, n, t) in gnt : use_capacity_margin && p_node[g, n, 'capacity_margin_MW'] && mode_invest} :
  + sum {(g,n,u) in gnu_capacityAvailability} (p_unittype[u, 'availability']
       * ( + p_unit[g, n, u, 'capacity_MW']
           + p_unit[g, n, u, 'invested_capacity_MW']        
           + (if (g,n,u) in gnu_invest then v_invest[g, n, u])
        ))
  + sum {(g,n,u) in gnu_capacityGen} v_gen[g, n, u, t]
  + sum {(g,n2,n) in gnn} (v_transfer[g,n2,n,t] - v_transferRightward[g,n2,n,t] * p_nodeNode[g,n2,n,'loss'])
  + sum {(g2, n2, u, g, n) in gnuGrid2Node2} v_convert[g2, n2, u, g, n, t] * (if p_unit[g,n,u,'use_efficiency_time_series'] then ts_unit[g,n,u,'efficiency',t] else p_unittype[u, 'conversion_eff'])
  + sum {(g_output1, n_output1, u, g, n) in gnuOutputGrid2Node2 : (g_output1, n_output1, u) in gnu_output1} v_gen[g,n,u,t]
  + (if (g,n) in gn_import then ts_import[g,n,t] * p_scaleImportFull[g,n])
  + v_capacitySlack[g, n, t]
  + v_slack[g, n, t]
  >= 
  + sum {u in unit : (g, n, u) in gnu_storage_charging || (g,n,u) in gnu_demand_increase} v_charge[g, n, u, t] 
  + sum {(g,n,n2,t) in gnnt} (v_transfer[g,n,n2,t] - v_transferLeftward[g,n,n2,t] * p_nodeNode[g,n,n2,'loss'])
  + sum {(g, n, u, g2, n2) in gnuGrid2Node2} v_convert[g, n, u, g2, n2, t]
  + (if (g,n) in gn_demand then demand[g,n,t] * p_scaleDemandForCapacityAdequacy[g,n])
  + p_node[g, n, 'capacity_margin_MW']
;

# Capacity margin for investment phase for nodeGroups
s.t. capacityMarginNodeGroup {(ng, t) in ngt : use_capacity_margin && p_nodeGroup[ng, 'capacity_margin_MW'] && mode_invest} :
  + sum {(g, n, u) in gnu_capacityAvailability : (g, n, ng) in gnng} 
      + (p_unittype[u, 'availability']
         * ( + p_unit[g, n, u, 'capacity_MW']
             + p_unit[g, n, u, 'invested_capacity_MW']        
             + (if (g,n,u) in gnu_invest then v_invest[g, n, u])
           )
		)
  + sum {(g,n,u) in gnu_capacityGen : (g, n, ng) in gnng} v_gen[g, n, u, t]
  + sum {(g,n2,n) in gnn : (g, n, ng) in gnng && (g, n2, ng) not in gnng} (v_transfer[g,n2,n,t] - v_transferRightward[g,n2,n,t] * p_nodeNode[g,n2,n,'loss'])
  + sum {(g2,n2,u,g,n) in gnuGrid2Node2 : (g,n,ng) in gnng} v_convert[g2, n2, u, g, n, t] * (if p_unit[g,n,u,'use_efficiency_time_series'] then ts_unit[g,n,u,'efficiency',t] else p_unittype[u, 'conversion_eff'])
  + sum {(g_output1, n_output1, u, g, n) in gnuOutputGrid2Node2 : (g,n,ng) in gnng && (g_output1, n_output1, u) in gnu_output1} v_gen[g,n,u,t]
  + sum {(g,n,ng) in gnng : (g,n) in gn_import} ts_import[g,n,t] * p_scaleImportFull[g,n]
  + sum {(g,n,ng) in gnng} v_capacitySlack[g, n, t]
  + sum {(g,n,ng) in gnng} v_slack[g, n, t]
  >= 
  + sum {(g,n,u) in gnu_all : (g,n,ng) in gnng && ((g,n,u) in gnu_storage_charging || (g,n,u) in gnu_demand_increase)} v_charge[g, n, u, t] 
  + sum {(g,n,n2) in gnn : (g, n, ng) in gnng && (g, n2, ng) not in gnng} (v_transfer[g,n,n2,t] - v_transferLeftward[g,n,n2,t] * p_nodeNode[g,n,n2,'loss'])
  + sum {(g,n,u,g2,n2) in gnuGrid2Node2 : (g,n,ng) in gnng} v_convert[g, n, u, g2, n2, t]
  + sum {(g,n,ng) in gnng : (g,n) in gn_demand} demand[g,n,t] * p_scaleDemandForCapacityAdequacy[g,n]
  + p_nodeGroup[ng, 'capacity_margin_MW']
;

# Online variable
s.t. startup {(g, n, u, t) in gnut : u in unit_startup} :
  + v_startup[g, n, u, t-p_time_jump[t]]
  >=
  + v_online[g, n, u, t]
  - v_online[g, n, u, t-p_time_jump[t]]
;


s.t. onlineLimit {(g, n, u, t) in gnut : u in unit_online} :
  + v_online[g, n, u, t]
  <=
  + (if (g, n, u) not in gnu_cf then
      + (if p_unit[g,n,u,'use_max_generation'] then ts_unit[g,n,u,'max_generation',t] else 1)
		* (
            + abs(p_unit[g, n, u, 'capacity_MW'])
            + abs(p_unit[g, n, u, 'invested_capacity_MW'])
            + (if (g,n,u) in gnu_invest then v_invest[g, n, u])
		  )
    )
  + sum {(g, n, u, c) in gnuCfprofile} ts_cf_scaled[c, t]
      * ( + p_unit[g, n, u, 'capacity_MW']
          + p_unit[g,n,u,'invested_capacity_MW']       
          + (if (g,n,u) in gnu_invest then v_invest[g, n, u])
        )
;

# Minimum downtime
s.t. minimum_downtime {(g, n, u, t) in gnut : u in unit_online && p_unittype[u,'min_downtime_h'] >= time_period_duration / 60 && t >= p_unittype[u,'min_downtime_h'] * 60 / time_period_duration} :
  + v_online[g, n, u, t]
  <=
  + abs(p_unit[g, n, u, 'capacity_MW'])
  + abs(p_unit[g, n, u, 'invested_capacity_MW'])
  + (if (g,n,u) in gnu_invest then v_invest[g, n, u])
  - sum{t_ in time_in_use : t_ > t && t_ <= t + p_unittype[u,'min_downtime_h'] * 60 / time_period_duration} (
      + v_startup[g, n, u, t_]
	)
;

# Minimum operational time
s.t. minimum_uptime {(g, n, u, t) in gnut : u in unit_online && p_unittype[u,'min_uptime_h'] >= time_period_duration / 60 && t >= p_unittype[u,'min_uptime_h'] * 60 / time_period_duration} :
  + v_online[g, n, u, t]
  >=
  + sum{t_ in time_in_use : t_ > t - 1 - p_unittype[u,'min_uptime_h'] * 60 / time_period_duration && t_ < t} (
      + v_startup[g, n, u, t_]
	)
;

# Transfer constraints 
s.t. transferLimit_rightward {(g, n, n2, t) in gnnt} :
  + v_transfer[g, n, n2, t]
  <=
  + p_nodeNode[g,n,n2,'cap.rightward_MW']
  + p_nodeNode[g,n,n2,'invested_capacity_MW']
  + (if (g, n, n2) in gnn_invest then v_investTransfer[g, n, n2])
;

s.t. transferLimit_leftward {(g, n, n2, t) in gnnt} :
  + v_transfer[g, n, n2, t]
  >=
  - p_nodeNode[g, n, n2, 'cap.leftward_MW']
  - p_nodeNode[g,n,n2,'invested_capacity_MW']
  - (if (g, n, n2) in gnn_invest then v_investTransfer[g, n, n2])
;

s.t. transferAbsolute {(g,n,n2,t) in gnnt} :
  + v_transferRightward[g,n,n2,t]
  - v_transferLeftward[g,n,n2,t]
  =
  + v_transfer[g,n,n2,t]
;

s.t. transferRightwardConstraint {(g,n,n2,t) in gnnt} :
  + v_transferRightward[g,n,n2,t]
  <=
  + p_nodeNode[g,n,n2,'cap.rightward_MW']
  + p_nodeNode[g,n,n2,'invested_capacity_MW']
  + (if (g, n, n2) in gnn_invest then v_investTransfer[g, n, n2])
;

s.t. transferLeftwardConstraint {(g,n,n2,t) in gnnt} :
  - v_transferLeftward[g,n,n2,t]
  >=
  - p_nodeNode[g, n, n2, 'cap.leftward_MW']
  - p_nodeNode[g,n,n2,'invested_capacity_MW']
  - (if (g, n, n2) in gnn_invest then v_investTransfer[g, n, n2])
;

# Investment limits for unitGroups
s.t. maximum_MW_unitGroup {ug in unitGroup : mode_invest && p_unitGroup[ug, 'max_invest_MW']} :
  + sum{(g,n,u,ug) in gnuug_output1} v_invest[g,n,u]
  <=
  + p_unitGroup[ug, 'max_invest_MW']
#  + v_investSlack[ug, 'max_MW']
;

s.t. minimum_MW_unitGroup {ug in unitGroup : mode_invest && p_unitGroup[ug, 'min_invest_MW']} :
  + sum{(g,n,u,ug) in gnuug_output1} v_invest[g,n,u]
  >=
  + p_unitGroup[ug, 'min_invest_MW']
#  - v_investSlack[ug, 'min_MW']
;

s.t. maximum_MWh_unitGroup {ug in unitGroup : mode_invest && p_unitGroup[ug, 'max_invest_MWh']} :
  + sum{(g,n,u,ug) in gnuug_output1 : (g,n,u) in gnu_storage} v_investStorage[g,n,u]
  <=
  + p_unitGroup[ug, 'max_invest_MWh']
#  + v_investSlack[ug, 'max_MWh']
;

s.t. minimum_MWh_unitGroup {ug in unitGroup : mode_invest && p_unitGroup[ug, 'min_invest_MWh']} :
  + sum{(g,n,u,ug) in gnuug_output1 : (g,n,u) in gnu_storage} v_investStorage[g,n,u]
  >=
  + p_unitGroup[ug, 'min_invest_MWh']
#  - v_investSlack[ug, 'min_MWh']
;


# Constraint for the instantenous share of non-synchronous generation for nodes
s.t. non_synchronousLimit {(g, n, t) in gnt : use_non_synchronous && p_node[g, n, 'non_synchronous_share']} :
  + sum {(g, n, u) in gnu_gen : p_unittype[u, 'non_synchronous']} v_gen[g, n, u, t] 
  + sum {(g2, n2, u, g, n) in gnuGrid2Node2 : p_unittype[u, 'non_synchronous']} 
      v_convert[g2, n2, u, g, n, t] * (if p_unit[g,n,u,'use_efficiency_time_series'] then ts_unit[g,n,u,'efficiency',t] else p_unittype[u, 'conversion_eff'])
  + sum {n2 in node : (g,n2,n) in gnn && p_nodeNode[g, n2, n, 'HVDC']} 
        v_transferRightward[g,n2,n,t] * p_nodeNodeEff[g,n2,n]
  + sum {n2 in node : (g,n,n2) in gnn && p_nodeNode[g, n, n2, 'HVDC']} 
        v_transferLeftward[g,n,n2,t] * p_nodeNodeEff[g,n,n2]
  <= 
  (
    + sum {(g, n, u) in gnu_storage_charging} v_charge[g, n, u, t]
    + sum {(g,n,u) in gnu_demand_increase} v_charge[g, n, u, t] 
    + sum {n2 in node : (g,n,n2) in gnn} (v_transfer[g,n,n2,t] * p_nodeNodeEff[g,n,n2] + v_transferRightward[g,n,n2,t] * p_nodeNode[g,n,n2,'loss'])  # Applies when the current node n is on the left side of the connection
    - sum {n2 in node : (g,n2,n) in gnn} (v_transfer[g,n2,n,t] - v_transferRightward[g,n2,n,t] * p_nodeNode[g,n2,n,'loss'])  # Applies when the current node 'n' is on the right side of the connection
    + sum {(g, n, u, g2, n2) in gnuGrid2Node2} v_convert[g, n, u, g2, n2, t]
    + (if (g,n) in gn_demand then demand[g,n,t])
    - v_slack[g,n,t]
    - (if (g,n) in gn_import then import[g, n, t])
  ) * p_node[g, n, 'non_synchronous_share']
;

# Constraint for the instantenous share of non-synchronous generation for nodeGroups
s.t. non_synchronousLimitNodeGroup {(ng, t) in ngt : use_non_synchronous && p_nodeGroup[ng, 'non_synchronous_share']} :
  + sum {(g, n, u) in gnu_gen : (g, n, ng) in gnng &&  p_unittype[u, 'non_synchronous']} v_gen[g, n, u, t] 
  + sum {(g2, n2, u, g, n) in gnuGrid2Node2 : p_unittype[u, 'non_synchronous'] && (g, n, ng) in gnng} 
      v_convert[g2, n2, u, g, n, t] * (if p_unit[g,n,u,'use_efficiency_time_series'] then ts_unit[g,n,u,'efficiency',t] else p_unittype[u, 'conversion_eff'])
  + sum {(g,n2,n) in gnn : p_nodeNode[g, n2, n, 'HVDC'] && (g, n, ng) in gnng && (g, n2, ng) not in gnng} 
        v_transferRightward[g,n2,n,t] * p_nodeNodeEff[g,n2,n]
  + sum {(g,n,n2) in gnn : p_nodeNode[g, n, n2, 'HVDC'] && (g, n, ng) in gnng && (g, n2, ng) not in gnng} 
        v_transferLeftward[g,n,n2,t] * p_nodeNodeEff[g,n,n2]
  <= 
  (
    + sum {(g, n, u) in gnu_storage_charging : (g, n, ng) in gnng} v_charge[g, n, u, t]
    + sum {(g,n,u) in gnu_demand_increase : (g, n, ng) in gnng} v_charge[g, n, u, t] 
    + sum {(g,n,n2) in gnn : (g, n, ng) in gnng && (g, n2, ng) not in gnng} (v_transfer[g,n,n2,t] * p_nodeNodeEff[g,n,n2] + v_transferRightward[g,n,n2,t] * p_nodeNode[g,n,n2,'loss'])  # Applies when the current node n is on the left side of the connection
    - sum {(g,n2,n) in gnn : (g, n, ng) in gnng && (g, n2, ng) not in gnng} (v_transfer[g,n2,n,t] - v_transferRightward[g,n2,n,t] * p_nodeNode[g,n2,n,'loss'])  # Applies when the current node 'n' is on the right side of the connection
    + sum {(g, n, u, g2, n2) in gnuGrid2Node2 : (g, n, ng) in gnng} v_convert[g, n, u, g2, n2, t]
    + sum {(g, n, ng) in gnng : (g,n) in gn_demand} demand[g,n,t]
    - sum {(g, n, ng) in gnng} v_slack[g,n,t]
    - sum {(g, n, ng) in gnng : (g,n) in gn_import} import[g, n, t]
  ) * p_nodeGroup[ng, 'non_synchronous_share']
;


# Constraint for minimum inertia in node groups
s.t. inertiaNodeGroup {(ng, t) in ngt : use_inertia_limit && p_nodeGroup[ng, 'inertia_limit_MWs']} :
  + sum {(g,n,ng,u) in gnngu : (g,n,u) in gnu_gen && u in unit_online && p_unittype[u, 'inertia_constant_MWs_per_MW']} 
      + v_online[g,n,u,t] * p_unittype[u, 'inertia_constant_MWs_per_MW']
  + sum {(g,n,ng,u) in gnngu : (g,n,u) in gnu_gen && u not in unit_online && p_unittype[u, 'inertia_constant_MWs_per_MW']} 
      + v_gen[g,n,u,t] * p_unittype[u, 'inertia_constant_MWs_per_MW']
  + v_inertiaSlack[ng,t]
  >=
  p_nodeGroup[ng, 'inertia_limit_MWs']
;	  


#########################
# Outputs to the console

display time_in_use_in_years, mode_invest, scale_to_annual;

printf "\n############# WARNINGS ###############\n";
printf {u in unit : p_unittype[u, 'availability'] = 0}: "Unittype %s has availability of 0 - no contribution to capacity margin!\n", u;
printf {(g,n,u) in gnu : (g,n,u) not in gnu_storage && (g,n,u) not in gnu_fuel && (g,n,u) not in gnu_flow && (g,n,u) not in gnu_demand_increase && (g,n,u) not in gnu_convertOutput}: "%s, %s, %s is not constrained by fuel, cf, or inflow!\n", g,n,u;
printf {u in unit_storage : p_unittype[u,'inv.cost_per_kW'] && p_unittype[u,'inv.cost_per_kWh'] && p_unittype[u,'fixed_kW_per_kWh_ratio']}: "%s should define only two of inv. cost per kW, inv. cost per kWh and fixed kW per kWh ratio!\n", u;
printf {ug in unitGroup : mode_invest && p_unitGroup[ug, 'max_invest_MW'] && p_unitGroup[ug, 'max_invest_MW'] < p_unitGroup[ug, 'min_invest_MW']}: "UnitGroup %s has higher min_invest_MW than max_invest_MW!\n", ug;
printf {ug in unitGroup : mode_invest && p_unitGroup[ug, 'max_invest_MWh'] && p_unitGroup[ug, 'max_invest_MWh'] < p_unitGroup[ug, 'min_invest_MWh']}: "UnitGroup %s has higher min_invest_MWh than max_invest_MWh!\n", ug;
printf {ug in unitGroup : not sum{(g,n,u,ug) in gnuug} 1}: "Unitgroup %s has no members!\n", ug;
printf {(g,n,u) in gnu : not sum{(g,n,u,ug) in gnuug} 1}: "Unit %s, %s, %s has no unitgroup!\n", g, n, u;
printf {cf in cf_profile : not p_sumCf[cf]}: "Cf profile %s has only zeroes in the selected time series!\n", cf;
printf {(g,n,u) in gnu_flow : p_unit[g,n,u,'fix_unit_generation']}: "Unit %s, %s, %s with inflow or cf has fixed generation as well - possible infeasibilities!\n", g, n, u;
printf {(g,n,u) in gnu_flow : p_unit[g,n,u,'use_min_generation']}: "Unit %s, %s, %s with inflow or cf has minimum generation as well - possible infeasibilities!\n", g, n, u;
printf {(g,n,u) in gnu_flow : p_unit[g,n,u,'use_min_online']}: "Unit %s, %s, %s with inflow or cf has min. online time series as well - possible infeasibilities!\n", g, n, u;
printf {(g,n,u) in gnu_gen : p_unit[g,n,u,'fix_unit_generation'] && p_unit[g,n,u,'use_max_generation']}: "Unit %s, %s, %s has both fix unit generation and use max generation - possible infeasibilities!\n", g, n, u;
printf {(g,n,u) in gnu_gen : p_unit[g,n,u,'fix_unit_generation'] && p_unit[g,n,u,'use_min_generation']}: "Unit %s, %s, %s has both fix unit generation and use min generation - possible infeasibilities!\n", g, n, u;
#printf {(g,n,u,t) in gnut : p_unit[g,n,u,'use_min_online'] && p_unit[g,n,u,'use_max_online'] && u in unit_online}: 
printf "######################################\n\n";


#########################
# Solve

solve;


##############################################
# Post-processing results for output
printf 'Post-processing results...\n';

param unitMW_output2temp{(g,n,u) in gnu_output1} :=
( if   p_unit[g,n,u,'output2_max_capacity_ratio']
  then p_unit[g,n,u,'output2_max_capacity_ratio'] * 
      ( + p_unit[g,n,u,'capacity_MW']
	    + p_unit[g,n,u,'invested_capacity_MW'] 
	    + (if (g,n,u) in gnu_invest then v_invest[g,n,u])
	  )
  else 0
);

param unitMW_output2temp2{(g,n,u) in gnu_output1} :=
( if   (g,n,u) in gnu_output2_eq 
  then ( if p_unit[g,n,u,'output2_eq_coeff'] >= 0
         then p_unit[g,n,u,'output2_eq_coeff'] * (p_unit[g,n,u,'capacity_MW'] + p_unit[g,n,u,'invested_capacity_MW'] + if (g,n,u) in gnu_invest then v_invest[g,n,u] else 0) + p_unit[g,n,u,'output2_eq_constant']
		 else p_unit[g,n,u,'output2_eq_constant']
	   )
  else ( if (g,n,u) in gnu_output2_lt
         then ( if p_unit[g,n,u,'output2_lt_coeff'] >= 0
		        then p_unit[g,n,u,'output2_lt_coeff'] * (p_unit[g,n,u,'capacity_MW'] + p_unit[g,n,u,'invested_capacity_MW'] + if (g,n,u) in gnu_invest then v_invest[g,n,u] else 0) + p_unit[g,n,u,'output2_lt_constant']
				else p_unit[g,n,u,'output2_lt_constant']
			  )
		 else 0
	   )	  
);

param unitMW_output2{(g,n,u,g_output2,n_output2) in gnuOutputGrid2Node2} :=
  ( if unitMW_output2temp[g,n,u] && unitMW_output2temp2[g,n,u]
    then min(unitMW_output2temp[g,n,u], unitMW_output2temp2[g,n,u] * (p_unit[g,n,u,'capacity_MW'] + p_unit[g,n,u,'invested_capacity_MW'] + if (g,n,u) in gnu_invest then v_invest[g,n,u] else 0))
	else ( if unitMW_output2temp[g,n,u]
	       then unitMW_output2temp[g,n,u]
		   else ( if unitMW_output2temp2[g,n,u]
		          then unitMW_output2temp2[g,n,u] * (p_unit[g,n,u,'capacity_MW'] + p_unit[g,n,u,'invested_capacity_MW'] + if (g,n,u) in gnu_invest then v_invest[g,n,u] else 0)
				  else 0
				)  
		 )
  );		 

param unitMW_convertFrom{(g_input,n_input,u) in gnu_convertInput} :=
    + sum{(g_input,n_input,u,g,n) in gnuGrid2Node2} (p_unit[g,n,u,'capacity_MW'] + p_unit[g,n,u,'invested_capacity_MW']) 
    + sum{(g_input,n_input,u,g,n) in gnuGrid2Node2 : (g,n,u) in gnu_invest}  v_investConvert[g_input,n_input,u,g,n].val 
;
 
param unitMW{(g,n,u) in gnu_all union gnu_output2 union gnu_convertInput} := 
  ( + (if (g,n,u) not in (gnu_convertOutput union gnu_output2) then abs(p_unit[g,n,u,'capacity_MW']))
    + (if (g,n,u) not in (gnu_convertOutput union gnu_output2) then abs(p_unit[g,n,u,'invested_capacity_MW']))
    + (if (g,n,u) in gnu_invest && not (g,n,u) in (gnu_convertOutput union gnu_output2) then v_invest[g,n,u].val)
	+ (if (g,n,u) in gnu_convertInput then unitMW_convertFrom[g,n,u])
    + (if (g,n,u) in gnu_convertOutput then 
		( + p_unit[g,n,u,'capacity_MW'] + p_unit[g,n,u,'invested_capacity_MW']
		  + sum{(g_input, n_input, u, g, n) in gnuGrid2Node2 : (g,n,u) in gnu_invest} v_investConvert[g_input,n_input,u,g,n].val
	    ) * (if p_unit[g,n,u,'use_efficiency_time_series'] then max{t in time_in_use} ts_unit[g,n,u,'efficiency',t] else p_unittype[u, 'conversion_eff']) 
	  )  
	+ (if (g,n,u) in gnu_output2 then sum{(g_output1,n_output1,u,g,n) in gnuOutputGrid2Node2} unitMW_output2[g_output1,n_output1,u,g,n])
  );

param unitMWh{(g,n,u) in gnu_storage} := 
  ( + abs(p_unit[g,n,u,'storage_MWh'])
    + abs(p_unit[g,n,u,'invested_storage_MWh'])
    + (if (g,n,u) in gnu_invest then v_investStorage[g,n,u].val)
  );

param unitVREnodeShare{(g,n,u,t) in gnut : (g,n,u) in gnu_cf && (g,n,u) not in gnu_storage && unitMW[g,n,u]} :=
  if sum {(g,n,u2,c) in gnuCfprofile : (g, n, u2) not in gnu_storage} ts_cf[c, t] * unitMW[g,n,u2] then
        ( sum {(g,n,u,c) in gnuCfprofile : (g, n, u) not in gnu_storage} ts_cf[c, t] * unitMW[g,n,u] * p_scaleCf[c]) 
      / ( sum {(g,n,u2,c) in gnuCfprofile : (g,n,u2) not in gnu_storage} ts_cf[c, t] * unitMW[g,n,u2] * p_scaleCf[c])  
  else 0
;

param unitVREnodeGroupShare{(ng,u) in ngu_VRE, t in time_in_use : sum{(g,n,ng) in gnng : (g,n,u) in gnu_cf && (g,n,u) not in gnu_storage && unitMW[g,n,u]} 1 } :=
  if sum {(g,n,u2,c) in gnuCfprofile : (g,n,ng) in gnng && (g, n, u2) not in gnu_storage} ts_cf[c, t] * unitMW[g,n,u2] then
        ( sum {(g,n,u,c) in gnuCfprofile : (g,n,ng) in gnng && (g, n, u) not in gnu_storage} ts_cf[c, t] * unitMW[g,n,u] * p_scaleCf[c]) 
      / ( sum {(g,n,u2,c) in gnuCfprofile : (g,n,ng) in gnng && (g,n,u2) not in gnu_storage} ts_cf[c, t] * unitMW[g,n,u2] * p_scaleCf[c])  
  else 0
;

param unitCurtailT{(g,n,u,t) in gnut : (g,n,u) in gnu_flow && unitMW[g,n,u]} :=
    + sum{(g,n,u,cf) in gnuCfprofile} (ts_cf_scaled[cf,t] * unitMW[g,n,u] - v_gen[g,n,u,t].val)
	+ (if (g,n,u) in gnu_inflow_noStorage then (if ts_inflowUnit[g,n,u,t] <= unitMW[g,n,u] then ts_inflowUnit[g,n,u,t] - v_gen[g,n,u,t].val else unitMW[g,n,u] - v_gen[g,n,u,t].val));
param unitCurtail{(g,n,u) in gnu_flow : unitMW[g,n,u]} := (sum{t in time_in_use} unitCurtailT[g,n,u,t]) * scale_to_annual;
param unitChargeGenLeakageT{(g,n,u,t) in gnut : (g,n,u) in gnu_storage_charging} :=
  ( if v_charge[g,n,u,t].val && v_gen[g,n,u,t].val then 
      + v_charge[g,n,u,t].val
  );
param unitChargeGenLeakage{(g,n,u) in gnu_storage_charging} := sum{t in time_in_use} unitChargeGenLeakageT[g,n,u,t] * scale_to_annual;
param unitGen{(g,n,u) in gnu_gen union gnu_output2 diff gnu_convertOutput diff gnu_demand_increase} := sum{t in time_in_use} v_gen[g,n,u,t].val * scale_to_annual - (if (g,n,u) in gnu_storage_charging then unitChargeGenLeakage[g,n,u]);
param unitCharge{(g,n,u) in gnu_storage_charging union gnu_demand_increase} := sum{t in time_in_use} v_charge[g,n,u,t].val * scale_to_annual - (if (g,n,u) in gnu_storage_charging then unitChargeGenLeakage[g,n,u]);
param unitConvertFrom{(g,n,u,g_output,n_output) in gnuGrid2Node2} := sum{t in time_in_use} (if (g,n,u,g_output,n_output) in gnuGrid2Node2 then v_convert[g,n,u,g_output,n_output,t].val) * scale_to_annual;
param unitConvertTo{(g_input,n_input,u,g,n) in gnuGrid2Node2} := sum{t in time_in_use} (v_convert[g_input,n_input,u,g,n,t].val * (if p_unit[g,n,u,'use_efficiency_time_series'] then ts_unit[g,n,u,'efficiency',t] else p_unittype[u, 'conversion_eff'])) * scale_to_annual;
param unitSpill{(g,n,u) in gnu_spill} := sum{t in time_in_use} (if (g,n,u) in gnu_spill then v_spill[g, n, u, t].val) * scale_to_annual;
param unitFuel{(g,n,u) in gnu} := (sum{(g,n,u,f,t) in gnuft} v_fuelUse[g,n,u,t].val) * scale_to_annual;
param unitStartup{(g,n,u) in gnu} := sum{t in time_in_use} (if u in unit_startup then v_startup[g,n,u,t].val) * scale_to_annual;
param unitReserveVRET{(g,n,u,t) in gnut : (g,n,u) in gnu_flow && unitMW[g,n,u]} := v_reserve[g,n,u,t]; 


param reserveDemand_nt{(g,n) in gn_reserve, t in time_in_use} :=
  ( + (if p_node[g, n, 'use_ts_reserve'] then ts_reserve_n[n, t])
    + sum{(u, c) in ucf_profile : (g, n, u, c) in gnuCfprofile && (g, n, u) not in gnu_storage && p_node[g, n, 'use_dynamic_reserve']} ts_cf[c, t] * p_unit[g, n, u, 'reserve_increase_ratio'] * unitMW[g,n,u]  
    + sum{(g,n,u) in gnu_inflow_noStorage} (v_gen[g,n,u,t] * p_unit[g,n,u,'reserve_increase_ratio'])
  );
  
param reserveDemandAnnual_n{(g,n) in gn_reserve} :=
  ( + sum{t in time_in_use : p_node[g, n, 'use_ts_reserve']} ts_reserve_n[n, t] 
    + sum{t in time_in_use, (u, c) in ucf_profile : (g, n, u, c) in gnuCfprofile && (g, n, u) not in gnu_storage && p_node[g, n, 'use_dynamic_reserve']} ts_cf[c, t] * p_unit[g, n, u, 'reserve_increase_ratio'] * unitMW[g,n,u]  
  ) * scale_to_annual
;

param reserveDemandAnnual_ng{ng in ng_reserve} :=
  ( + sum{t in time_in_use : p_nodeGroup[ng, 'use_ts_reserve']} ts_reserve_ng[ng, t] 
    + sum{t in time_in_use, (u, c) in ucf_profile, (g,n,ng) in gnng : (g, n, u, c) in gnuCfprofile && (g, n, u) not in gnu_storage && p_nodeGroup[ng, 'use_dynamic_reserve']} ts_cf[c, t] * p_unit[g, n, u, 'reserve_increase_ratio'] * unitMW[g,n,u]  
  ) * scale_to_annual
;

param inflowNoStorageCurtailedT{(g,n,t) in gnt} :=
  + sum{(g,n,u) in gnu_inflow_noStorage} (if ts_inflowUnit[g,n,u,t] <= unitMW[g,n,u] then ts_inflowUnit[g,n,u,t] - v_gen[g,n,u,t].val else unitMW[g,n,u] - v_gen[g,n,u,t].val)
  + sum{(g,n,u,cf) in gnuCfprofile : (g,n,u) in gnu_cf} (ts_cf_scaled[cf,t] * unitMW[g,n,u] - v_gen[g,n,u,t].val);
param inflowNoStorageCurtailed{(g,n,u) in gnu_flow} := 
  + unitGen[g,n,u] - unitCurtail[g,n,u];

param gnu_investCost{(g,n,u) in gnu_all} :=
  + (p_unit[g, n, u, 'invested_capacity_MW'] 
  + (if (g,n,u) in gnu_invest && not (g,n,u) in gnu_convertOutput then v_invest[g,n,u].val)) 
      * p_unittype[u, 'inv.cost_per_kW'] / 1000 * p_unittype[u, 'annuity']
  + (if (g,n,u) in gnu_invest && (g,n,u) in gnu_convertOutput then sum{(g2, n2, u, g, n) in gnuGrid2Node2} v_investConvert[g2, n2, u, g, n].val * p_unittype[u, 'inv.cost_per_kW'] / 1000 * p_unittype[u, 'annuity'] else 0)
  + (p_unit[g, n, u, 'invested_storage_MWh'] + (if (g,n,u) in gnu_invest && (g,n,u) in gnu_storage then v_investStorage[g,n,u].val)) * p_unittype[u, 'inv.cost_per_kWh'] / 1000 * p_unittype[u, 'annuity']
  + (if mode_dispatch = 2 && (g,n,u) in gnu_invest then gnu_investCost_I[g,n,u])
;
param gnn_investCost{(g,n,n2) in gnn} :=
  + (p_nodeNode[g,n,n2,'invested_capacity_MW'] + (if (g,n,n2) in gnn_invest then v_investTransfer[g, n, n2].val else 0)) * p_nodeNode[g, n, n2, 'inv.cost_per_kW'] / 1000 * p_nodeNode[g, n, n2, 'annuity']
  + (if mode_dispatch = 2 && (g,n,n2) in gnn_invest then gnn_investCost_I[g,n,n2])
;

param transferMW{(g,n,n2) in gnn} :=
  ( + ( p_nodeNode[g,n,n2,'cap.rightward_MW'] + p_nodeNode[g,n,n2,'cap.leftward_MW'] ) / 2
    + p_nodeNode[g,n,n2,'invested_capacity_MW']
    + (if (g,n,n2) in gnn_invest then v_investTransfer[g,n,n2].val else 0)
  );
param transferRightward_MW{(g,n,n2) in gnn} :=
  ( + p_nodeNode[g,n,n2,'cap.rightward_MW'] 
    + p_nodeNode[g,n,n2,'invested_capacity_MW']
    + (if (g,n,n2) in gnn_invest then v_investTransfer[g,n,n2].val else 0)
  );
param transferLeftward_MW{(g,n,n2) in gnn} :=
  ( + p_nodeNode[g,n,n2,'cap.leftward_MW'] 
    + p_nodeNode[g,n,n2,'invested_capacity_MW']
    + (if (g,n,n2) in gnn_invest then v_investTransfer[g,n,n2].val else 0)
  );
#param transferLeakageT{(g,n,n2,t) in gnnt} := (if v_transfer[g,n,n2,t]
param transfer_TWh{(g,n,n2) in gnn} := (sum{t in time_in_use} abs(v_transfer[g,n,n2,t].val)) / 1000000 * scale_to_annual;
param p_transferRightward{(g,n,n2,t) in gnnt} :=  if v_transfer[g,n,n2,t] > 0 then v_transfer[g,n,n2,t].val;
param p_transferLeftward{(g,n,n2,t) in gnnt} :=  if v_transfer[g,n,n2,t] < 0 then -v_transfer[g,n,n2,t].val;
param transferRightward_TWh{(g,n,n2) in gnn} := (sum{t in time_in_use} p_transferRightward[g,n,n2,t]) / 1000000 * scale_to_annual;
param transferLeftward_TWh{(g,n,n2) in gnn} := (sum{t in time_in_use} p_transferLeftward[g,n,n2,t]) / 1000000 * scale_to_annual;
param p_transfer_in{(g,n,t) in gnt} := + sum{n2 in node : (g,n,n2) in gnn} (p_transferLeftward[g,n,n2,t] * p_nodeNodeEff[g,n,n2])
                                       + sum{n2 in node : (g,n2,n) in gnn} (p_transferRightward[g,n2,n,t] * p_nodeNodeEff[g,n2,n]);
param p_transfer_out{(g,n,t) in gnt} := + sum{n2 in node : (g,n,n2) in gnn} p_transferRightward[g,n,n2,t] 
                                        + sum{n2 in node : (g,n2,n) in gnn} p_transferLeftward[g,n2,n,t];
param transfer_in_TWh{(g,n) in gn}  := sum{t in time_in_use} p_transfer_in[g,n,t] / 1000000 * scale_to_annual;
param transfer_out_TWh{(g,n) in gn} := sum{t in time_in_use} p_transfer_out[g,n,t] / 1000000 * scale_to_annual;

param transfer_losses_TWh{(g,n,n2) in gnn} := sum{t in time_in_use} (v_transfer[g,n,n2,t] - 2 * v_transferRightward[g,n,n2,t]) * p_nodeNode[g,n,n2,'loss'] / 1000000 * scale_to_annual;
#param transfer_losses_actual_TWh{(g,n,n2) in gnn} := -sum{t in time_in_use} (p_transferRightward[g,n,n2,t] + p_transferLeftward[g,n,n2,t]) * p_nodeNode[g,n,n2,'loss'] / 1000000 * scale_to_annual;
#param transferTWh_leftward{(g,n,n2) in gnn} := (sum{t in time_in_use} v_transferLeftward[g,n,n2,t].val) * scale_to_annual / 1000000;

param netLoad{(g,n,t) in gnt : (g,n) in gn_demand} := ( 
    + demand[g,n,t]
    - sum {(g,n,u) in gnu_flow} v_gen[g,n,u,t].val
    - (if (g,n) in gn_import then import[g, n, t])
   );
param netLoad_gt{g in grid, t in time_in_use} := sum{n in node : (g,n) in gn_demand} netLoad[g,n,t];
param pos_netLoad{g in grid, t in time_in_use} := 
      1 + card({t_ in time_in_use : netLoad_gt[g,t_] > netLoad_gt[g,t] or netLoad_gt[g,t_] = netLoad_gt[g,t] and t_ > t});
param ind_netLoad{g in grid, t__ in 1..card(time_in_use)} := sum{t in time_in_use: pos_netLoad[g,t] = t__} t;

param netLoadRamp{(g,n,t) in gnt : (g,n) in gn_demand && p_time_jump[t] = 1} := 
	netLoad[g,n,t] - netLoad[g,n,t-1];
param netLoadRamp_gt{g in grid, t in time_in_use_ramp} := sum{n in node : (g,n) in gn_demand} netLoadRamp[g,n,t];
param pos_netLoadRamp{g in grid, t in time_in_use_ramp} := 
      1 + card({t_ in time_in_use_ramp : netLoadRamp_gt[g,t_] > netLoadRamp_gt[g,t] or netLoadRamp_gt[g,t_] = netLoadRamp_gt[g,t] and t_ > t});
param ind_netLoadRamp{g in grid, t__ in 1..rampTimePeriods} := sum{t in time_in_use_ramp: pos_netLoadRamp[g,t] = t__} t;
param netLoadRamp_4h{(g,n,t) in gnt : (g,n) in gn_demand && p_time_jump[t] = 1 && p_time_jump[t-1] = 1 && p_time_jump[t-2] = 1 && p_time_jump[t-3] = 1} :=
    netLoad[g,n,t] - netLoad[g,n,t-4];

param p_unitRampUpMW{(g,n,u) in gnu_inputs_outputs} := (if p_unittypeRampUp[u] <= 1 then p_unittypeRampUp[u] else 1) * unitMW[g,n,u];
param p_unitRampDownMW{(g,n,u) in gnu_inputs_outputs} := (if p_unittypeRampDown[u] <= 1 then p_unittypeRampDown[u] else 1) * unitMW[g,n,u];
param p_unitRampUpMW_4h{(g,n,u) in gnu_inputs_outputs} := (if p_unittypeRampUp[u] * 4 <= 1 then p_unittypeRampUp[u] else 1) * unitMW[g,n,u];
param p_unitRampDownMW_4h{(g,n,u) in gnu_inputs_outputs} := (if p_unittypeRampDown[u] * 4 <= 1 then p_unittypeRampDown[u] else 1) * unitMW[g,n,u];

param rampRoomUp{(g,n,t) in gnt} :=
  ( + sum{ (g,n,u) in gnu_inputs_outputs : (g,n,u) not in gnu_flow } (
        + (if p_unitRampUpMW[g,n,u] <= (if (g,n,u) in gnu union gnu_output1 then 
                                            + (if (g,n,u) in gnu_demand_increase then 0 else unitMW[g,n,u])
                                                - (
                                                     + (if (g,n,u) in gnu_gen diff gnu_flow then v_gen[g,n,u,t].val) 
                                                     - (if ((g,n,u) in gnu_storage_charging || (g,n,u) in gnu_demand_increase) then v_charge[g,n,u,t].val)
                                                  )  
                                            + sum{(g,n,u,g2,n2) in gnuGrid2Node2} v_convert[g,n,u,g2,n2,t].val
                                        else
                                            + sum{(g,n,u,g2,n2) in gnuGrid2Node2Inverse} ((unitMW_convertFrom[g,n,u] - v_convert[g2,n2,u,g,n,t].val) * (if p_unit[g,n,u,'use_efficiency_time_series'] then ts_unit[g,n,u,'efficiency',t] else p_unittype[u, 'conversion_eff']))
                                        )
           then p_unitRampUpMW[g,n,u] 
           else (if (g,n,u) in gnu union gnu_output1 then 
                   + (if (g,n,u) in gnu_demand_increase then 0 else unitMW[g,n,u])
                       - (
                           + (if (g,n,u) in gnu_gen diff gnu_flow then v_gen[g,n,u,t]) 
                           - (if ((g,n,u) in gnu_storage_charging || (g,n,u) in gnu_demand_increase) then v_charge[g,n,u,t])
                         )
                   + sum{(g,n,u,g2,n2) in gnuGrid2Node2} v_convert[g,n,u,g2,n2,t].val
                 else
                   + sum{(g,n,u,g2,n2) in gnuGrid2Node2Inverse} ((unitMW_convertFrom[g,n,u] - v_convert[g2,n2,u,g,n,t].val) * (if p_unit[g,n,u,'use_efficiency_time_series'] then ts_unit[g,n,u,'efficiency',t] else p_unittype[u, 'conversion_eff']))
                 )
          )
        # Could include reserve also, but depends if the reserves can be used for 'ramping reserve' 
      )
  );

param rampRoomUpVG{(g,n,t) in gnt} := 
   + rampRoomUp[g,n,t] 
   + sum{(g,n,u) in gnu_inflow_noStorage} 
       ( if p_unitRampUpMW[g,n,u] <= ts_inflowUnit[g,n,u,t]
         then p_unitRampUpMW[g,n,u] - v_gen[g,n,u,t].val 
         else ts_inflowUnit[g,n,u,t] - v_gen[g,n,u,t].val
       )
   + sum{(g,n,u) in gnu_cf} 
       ( if p_unitRampUpMW[g,n,u] <= sum{(g,n,u,cf) in gnuCfprofile} ts_cf_scaled[cf,t] * unitMW[g,n,u]
         then p_unitRampUpMW[g,n,u] - v_gen[g,n,u,t].val 
         else sum{(g,n,u,cf) in gnuCfprofile} ts_cf_scaled[cf,t] * unitMW[g,n,u] - v_gen[g,n,u,t].val
       );

param rampRoomUpTransfer{(g,n,t) in gnt} := 
   + rampRoomUpVG[g,n,t]
   + sum{n2 in node : (g,n,n2) in gnn} (
       if rampRoomUpVG[g,n2,t] <= transferMW[g,n,n2] + v_transfer[g,n,n2,t].val
       then rampRoomUpVG[g,n2,t]
       else transferMW[g,n,n2] + v_transfer[g,n,n2,t].val
     )
   + sum{n2 in node : (g,n2,n) in gnn} (
       if rampRoomUpVG[g,n2,t] <= transferMW[g,n2,n] - v_transfer[g,n2,n,t].val
       then rampRoomUpVG[g,n2,t]
       else transferMW[g,n2,n] - v_transfer[g,n2,n,t].val
     )
;

param rampRoomDown{(g,n,t) in gnt} :=
  ( - sum{ (g,n,u) in gnu_inputs_outputs : (g,n,u) not in gnu_flow } (
        + (if p_unitRampDownMW[g,n,u] <= (if (g,n,u) in gnu union gnu_output1 then 
                                            + (if (g,n,u) in gnu_gen diff gnu_flow then v_gen[g,n,u,t].val) 
                                            + (if ((g,n,u) in gnu_storage_charging || (g,n,u) in gnu_demand_increase) then unitMW[g,n,u] - v_charge[g,n,u,t].val)
                                            + sum{(g,n,u,g2,n2) in gnuGrid2Node2} (unitMW[g,n,u] - v_convert[g,n,u,g2,n2,t].val)
                                        else
                                            + sum{(g,n,u,g2,n2) in gnuGrid2Node2Inverse} (v_convert[g2,n2,u,g,n,t].val * (if p_unit[g,n,u,'use_efficiency_time_series'] then ts_unit[g,n,u,'efficiency',t] else p_unittype[u, 'conversion_eff']))
                                        )
           then p_unitRampDownMW[g,n,u] 
           else (if (g,n,u) in gnu union gnu_output1 then 
                   + (if (g,n,u) in gnu_gen diff gnu_flow then v_gen[g,n,u,t]) 
                   + (if ((g,n,u) in gnu_storage_charging || (g,n,u) in gnu_demand_increase) then unitMW[g,n,u] - v_charge[g,n,u,t])
                   + sum{(g,n,u,g2,n2) in gnuGrid2Node2} (unitMW[g,n,u] - v_convert[g,n,u,g2,n2,t].val)
                 else
                   + sum{(g_input,n_input,u,g,n) in gnuGrid2Node2} (v_convert[g_input,n_input,u,g,n,t].val * (if p_unit[g,n,u,'use_efficiency_time_series'] then ts_unit[g,n,u,'efficiency',t] else p_unittype[u, 'conversion_eff']))
                 )
          )
        # Could include reserve also, but depends if the reserves can be used for 'ramping reserve' 
      )
  );

param rampRoomDownVG{(g,n,t) in gnt} := 
  + rampRoomDown[g,n,t] 
  - sum {(g,n,u) in gnu_flow} (if p_unitRampDownMW[g,n,u] <= v_gen[g,n,u,t].val then p_unitRampDownMW[g,n,u] else v_gen[g,n,u,t].val)
;

param rampRoomDownTransfer{(g,n,t) in gnt} := 
   + rampRoomDownVG[g,n,t]
   - sum{n2 in node : (g,n,n2) in gnn} (
       if -rampRoomDownVG[g,n2,t] <= transferMW[g,n,n2] - v_transfer[g,n,n2,t].val
       then -rampRoomDownVG[g,n2,t]
       else transferMW[g,n,n2] - v_transfer[g,n,n2,t].val
     )
   - sum{n2 in node : (g,n2,n) in gnn} (
       if -rampRoomDownVG[g,n2,t] <= transferMW[g,n2,n] + v_transfer[g,n2,n,t].val
       then -rampRoomDownVG[g,n2,t]
       else transferMW[g,n2,n] + v_transfer[g,n2,n,t].val
     )
;

param rampRoomUp_4h{(g,n,t) in gnt} :=
  ( + sum{ (g,n,u) in gnu_inputs_outputs : (g,n,u) not in gnu_flow } (
        + (if p_unitRampUpMW_4h[g,n,u] <= (if (g,n,u) in gnu union gnu_output1 then 
                                            + (if (g,n,u) in gnu_demand_increase then 0 else unitMW[g,n,u])
                                                - (
                                                    + (if (g,n,u) in gnu_gen diff gnu_flow then v_gen[g,n,u,t].val) 
                                                    - (if ((g,n,u) in gnu_storage_charging || (g,n,u) in gnu_demand_increase) then v_charge[g,n,u,t].val)
                                                  ) 
                                            + sum{(g,n,u,g2,n2) in gnuGrid2Node2} v_convert[g,n,u,g2,n2,t].val
                                        else
                                            + sum{(g,n,u,g2,n2) in gnuGrid2Node2Inverse} ((unitMW_convertFrom[g,n,u] - v_convert[g2,n2,u,g,n,t].val) * (if p_unit[g,n,u,'use_efficiency_time_series'] then ts_unit[g,n,u,'efficiency',t] else p_unittype[u, 'conversion_eff']))
                                        )
           then p_unitRampUpMW_4h[g,n,u] 
           else (if (g,n,u) in gnu union gnu_output1 then 
                   + (if (g,n,u) in gnu_demand_increase then 0 else unitMW[g,n,u])
                       - (
                           + (if (g,n,u) in gnu_gen diff gnu_flow then v_gen[g,n,u,t]) 
                           - (if ((g,n,u) in gnu_storage_charging || (g,n,u) in gnu_demand_increase) then v_charge[g,n,u,t])
                         )
                   + sum{(g,n,u,g2,n2) in gnuGrid2Node2} v_convert[g,n,u,g2,n2,t].val
                 else
                   + sum{(g,n,u,g2,n2) in gnuGrid2Node2Inverse} ((unitMW_convertFrom[g,n,u] - v_convert[g2,n2,u,g,n,t].val) * (if p_unit[g,n,u,'use_efficiency_time_series'] then ts_unit[g,n,u,'efficiency',t] else p_unittype[u, 'conversion_eff']))
                 )
          )
        # Could include reserve also, but depends if the reserves can be used for 'ramping reserve' 
      )
  );

param rampRoomUpVG_4h{(g,n,t) in gnt} := 
   + rampRoomUp_4h[g,n,t] 
   + sum{(g,n,u) in gnu_cf} 
       ( if p_unitRampUpMW[g,n,u] <= sum{(g,n,u,cf) in gnuCfprofile} ts_cf_scaled[cf, t] * unitMW[g,n,u] 
         then p_unitRampUpMW[g,n,u] - v_gen[g,n,u,t].val 
         else sum{(g,n,u,cf) in gnuCfprofile} ts_cf_scaled[cf, t] * unitMW[g,n,u] - v_gen[g,n,u,t].val
       )
   + sum{(g,n,u) in gnu_inflow_noStorage} 
       ( if p_unitRampUpMW[g,n,u] <= ts_inflowUnit[g,n,u,t] 
         then p_unitRampUpMW[g,n,u] - v_gen[g,n,u,t].val 
         else ts_inflowUnit[g,n,u,t] - v_gen[g,n,u,t].val
       );

param rampRoomUpTransfer_4h{(g,n,t) in gnt} := 
   + rampRoomUpVG_4h[g,n,t]
   + sum{n2 in node : (g,n,n2) in gnn} (
       if rampRoomUpVG_4h[g,n2,t] <= transferMW[g,n,n2] + v_transfer[g,n,n2,t].val
       then rampRoomUpVG_4h[g,n2,t]
       else transferMW[g,n,n2] + v_transfer[g,n,n2,t].val
     )
   + sum{n2 in node : (g,n2,n) in gnn} (
       if rampRoomUpVG_4h[g,n2,t] <= transferMW[g,n2,n] - v_transfer[g,n2,n,t].val
       then rampRoomUpVG_4h[g,n2,t]
       else transferMW[g,n2,n] - v_transfer[g,n2,n,t].val
     )
;

param rampRoomDown_4h{(g,n,t) in gnt} :=
  ( - sum{ (g,n,u) in gnu_inputs_outputs : (g,n,u) not in gnu_flow } (
        + (if p_unitRampDownMW_4h[g,n,u] <= (if (g,n,u) in gnu union gnu_output1 then 
                                            + (if (g,n,u) in gnu_gen diff gnu_flow then v_gen[g,n,u,t].val) 
                                            + (if ((g,n,u) in gnu_storage_charging || (g,n,u) in gnu_demand_increase) then unitMW[g,n,u] - v_charge[g,n,u,t].val)
                                            + sum{(g,n,u,g2,n2) in gnuGrid2Node2} (unitMW[g,n,u] - v_convert[g,n,u,g2,n2,t].val)
                                        else
                                            + sum{(g,n,u,g2,n2) in gnuGrid2Node2Inverse} (v_convert[g2,n2,u,g,n,t].val * (if p_unit[g,n,u,'use_efficiency_time_series'] then ts_unit[g,n,u,'efficiency',t] else p_unittype[u, 'conversion_eff']))
                                        )
           then p_unitRampDownMW_4h[g,n,u] 
           else (if (g,n,u) in gnu union gnu_output1 then 
                   + (if (g,n,u) in gnu_gen diff gnu_flow then v_gen[g,n,u,t]) 
                   + (if ((g,n,u) in gnu_storage_charging || (g,n,u) in gnu_demand_increase) then unitMW[g,n,u] - v_charge[g,n,u,t].val)
                   + sum{(g,n,u,g2,n2) in gnuGrid2Node2} (unitMW[g,n,u] - v_convert[g,n,u,g2,n2,t].val)
                 else
                   + sum{(g,n,u,g2,n2) in gnuGrid2Node2Inverse} (v_convert[g2,n2,u,g,n,t].val * (if p_unit[g,n,u,'use_efficiency_time_series'] then ts_unit[g,n,u,'efficiency',t] else p_unittype[u, 'conversion_eff']))
                 )
          )
        # Could include reserve also, but depends if the reserves can be used for 'ramping reserve' 
      )
  );

param rampRoomDownVG_4h{(g,n,t) in gnt} := 
  + rampRoomDown_4h[g,n,t] 
  - sum {(g,n,u) in gnu_flow} (if p_unitRampDownMW[g,n,u] <= v_gen[g,n,u,t].val then p_unitRampDownMW[g,n,u] else v_gen[g,n,u,t].val)
;

param rampRoomDownTransfer_4h{(g,n,t) in gnt} := 
   + rampRoomDownVG_4h[g,n,t]
   - sum{n2 in node : (g,n,n2) in gnn} (
       if -rampRoomDownVG_4h[g,n2,t] <= transferMW[g,n,n2] - v_transfer[g,n,n2,t].val
       then -rampRoomDownVG_4h[g,n2,t]
       else transferMW[g,n,n2] - v_transfer[g,n,n2,t].val
     )
   - sum{n2 in node : (g,n2,n) in gnn} (
       if -rampRoomDownVG_4h[g,n2,t] <= transferMW[g,n2,n] + v_transfer[g,n2,n,t].val
       then -rampRoomDownVG_4h[g,n2,t] 
       else transferMW[g,n2,n] + v_transfer[g,n2,n,t].val
     )
;

param rampUpLossOfLoad{g in grid, t in time_in_use : sum{(g,n) in gn_demand} 1 && p_time_jump[t] = 1} :=
   + (if sum{(g,n,t) in gnt} rampRoomUpVG[g,n,t-p_time_jump[t]] < sum{(g,n,t) in gnt} netLoadRamp[g,n,t] then sum{(g,n,t) in gnt} (netLoadRamp[g,n,t] - rampRoomUpVG[g,n,t-p_time_jump[t]]))
;

param rampDownCurtail{(g,n,t) in gnt : sum{(g,n,u) in gnu_flow} unitGen[g,n,u] && p_time_jump[t] = 1} :=
   + (if rampRoomDown[g,n,t-p_time_jump[t]] > netLoadRamp[g,n,t] then (if sum{(g,n,u) in gnu_flow} unitCurtailT[g,n,u,t-p_time_jump[t]] > 0 then sum{(g,n,u) in gnu_flow} unitCurtailT[g,n,u,t-p_time_jump[t]]))
;

param gnu_operationCost{(g,n,u) in gnu} :=
    + unitMW[g,n,u] * p_unittype[u, 'fixed_cost_per_kW_per_year'] / 1000 
    + (if (g,n,u) in gnu_gen then unitGen[g,n,u] * p_unittype[u, 'O&M_cost_per_MWh'] / 1000000)
	+ (if (g,n,u) in gnu_demand_increase then unitCharge[g,n,u] * p_unittype[u, 'O&M_cost_per_MWh'] / 1000000)
	+ (if (g,n,u) in gnu_convertOutput then sum{(g2,n2,u,g,n) in gnuGrid2Node2} unitConvertFrom[g2,n2,u,g,n] * p_unittype[u, 'O&M_cost_per_MWh'] / 1000000)
	+ sum{(g,n,u,f) in gnuFuel : (g,n,u) in gnu_fuel} unitFuel[g,n,u] * p_fuel[f, 'fuel_price_per_MWh'] / 1000000
    + sum{(g,n,u,f) in gnuFuel : (g,n,u) in gnu_fuel} unitFuel[g,n,u] * p_fuel[f, 'CO2_content_t_per_MWh'] * co2_cost / 1000000
    + (if (g,n,u) in gnu && u in unit_startup then unitStartup[g,n,u] * p_unittype[u, 'startup_cost'] / 1000000)
;

param gn_reserveSlack{(g,n,t) in gnt} :=
    + (if (g,n,t) in gnt_reserve then v_reserveSlack_n[g, n, t].val)
	+ sum {(g,n,ng) in gnng : ng in ng_reserve} v_reserveSlack_ng[ng, t].val
	    / (sum {(g,n2,ng) in gnng : ng in ng_reserve} 1)
;

param gn_inertiaSlack{(g,n,t) in gnt} :=
	+ sum {(g,n,ng) in gnng : p_nodeGroup[ng, 'inertia_limit_MWs'] && use_inertia_limit} v_inertiaSlack[ng, t].val
	    / (sum {(g,n2,ng) in gnng : p_nodeGroup[ng, 'inertia_limit_MWs'] && use_inertia_limit} 1)
;

param gn_penaltyCost{(g,n) in gn} :=
#    + sum {(g,n,t) in gnt : (g,n) in gn_flow} (inflowNoStorageCurtailedT[g,n,t]) * curtailment_penalty / 1000000 * scale_to_annual
    + sum {(g,n,t) in gnt} v_slack[g, n, t] * loss_of_load_penalty / 1000000 * scale_to_annual
    + sum {(g,n,t) in gnt} v_demandSlack[g, n, t] * loss_of_load_penalty / 1000000 * scale_to_annual
    + sum {(g,n,t) in gnt} gn_reserveSlack[g, n, t] * loss_of_reserves_penalty / 1000000 * scale_to_annual
    + sum {(ng,t) in ngt : (g,n,ng) in gnng && use_inertia_limit && p_nodeGroup[ng, 'inertia_limit_MWs']} v_inertiaSlack[ng, t] * lack_of_inertia_penalty / (sum{(g,n2,ng) in gnng} 1) / 1000000 * scale_to_annual
    + sum {(g,n,t) in gnt : use_capacity_margin && mode_invest} v_capacitySlack[g, n, t] * lack_of_capacity_penalty / 1000000 * scale_to_annual
;

#########################
# Start the files that are used to print color codes
param fn_colors{g in grid} symbolic := resfolder & "\color_codes_" & g & ".dat";
for {g in grid}	printf '#Color codes for %s\n', g > fn_colors[g];
param fn_colors_no_grid symbolic := resfolder & "\color_codes_no_grid.dat";
printf '#Color codes for items not dependent on the grid\n' > fn_colors_no_grid;

#########################
# Outputs to result csv files

table transfers_over_time {(g, n, n2, t) in gnnt} OUT 'CSV' resfolder & 'transfers_t_' & (if mode_invest then "I.csv" else "D.csv"): 
  g~grid, n~node, n2~node, t~time, 
  - v_transfer[g,n,n2,t].val * p_nodeNodeEff[g,n,n2] ~rightward_MWh
;

table unit_investment_results {(g, n, u) in gnu_all} OUT 'CSV' resfolder & 'units_invest_' & (if mode_invest then "I.csv" else "D.csv"): 
  g~grid, n~node, u~unit, 
  (if (g, n, u) in gnu_invest then (if not (g,n,u) in gnu_convertOutput then v_invest[g, n, u].val else sum{(g2, n2, u, g, n) in gnuGrid2Node2} v_investConvert[g2, n2, u, g, n].val))~invest_MW,
  (if (g, n, u) in gnu_invest then (if not (g,n,u) in gnu_convertOutput then v_invest[g, n, u].dual / 1000 else sum{(g2, n2, u, g, n) in gnuGrid2Node2} v_investConvert[g2, n2, u, g, n].dual / 1000))~invest_dual_CUR_per_kW,
  (if (g, n, u) in gnu_invest && u in unit_storage then v_investStorage[g, n, u].val else 0)~invest_MWh,
  (if (g, n, u) in gnu_invest && u in unit_storage then v_investStorage[g, n, u].dual / 1000 else 0)~invest_dual_CUR_per_kWh
;

table transfers {(g, n, n2) in gnn} OUT 'CSV' resfolder & 'transfers_invest_' & (if mode_invest then "I.csv" else "D.csv"): 
  g~grid, n~node, n2~node, 
  (if (g,n,n2) in gnn_invest then v_investTransfer[g, n, n2].val else 0)~invest_MW,
  (if (g,n,n2) in gnn_invest then v_investTransfer[g, n, n2].dual / 1000 else 0)~invest_dual_CUR_per_kW
;


param print_event_n{(g,n,t) in gnt : (g,n) in gn_demand} := 
  + v_slack[g,n,t]
  + (if use_capacity_margin && mode_invest then v_capacitySlack[g,n,t])
  + (if (g,n) in gn_reserve then v_reserveSlack_n[g,n,t])
  + sum{(g,n,ng) in gnng : ng in ng_reserve} v_reserveSlack_ng[ng,t]
  + sum{(g,n,u) in gnu_spill} v_spill[g,n,u,t]
  + gn_inertiaSlack[g,n,t]
;
  
#param print_event_ng{(ng,t) in ngt} := 
#  + sum{(g,n,ng) in gnng} v_slack[g,n,t]
#  + sum{(g,n,ng) in gnng} (if use_capacity_margin && mode_invest then v_capacitySlack[g,n,t])
#  + (if ng in ng_reserve then v_reserveSlack_ng[ng,t])
#  + sum{(g,n,u) in gnu_spill : (g,n,ng) in gnng} v_spill[g,n,u,t]
#;

#########################
# Print out events
printf 'Write potential events over time...\n';
param fn_events symbolic := resfolder & "\events_" & (if mode_invest then "I.csv" else "D.csv");

printf 'Grid,Node,Time,' > fn_events;
printf '"Capacity inadequacy (MW)","Reserve inadequacy node (MW)","Reserve inadequacy nodeGroup (MW)",' >> fn_events;
printf '"Inertia inadequacy (MWs)","Loss of load (MW)","Excess load (MW)","Demand (MW)",' >> fn_events;
printf '"Imports (MW)","Non-VRE generation (MW)","VRE generation (MW)",' >> fn_events;
printf '"Inflow generation (MW)","Consume (MW)","Transfer in (MW)","Transfer out (MW)",' >> fn_events;
printf '"Converted (MW)","Curtailed (MW)","Spilled (MW)","Actual net load (MW)","Net load ramp (MWh/h)",' >> fn_events;
printf '"Precalculated reserve requirement (MWh/h)","Dynamic reserve requirement (MWh/h)",' >> fn_events;
printf '"Reserve conventional (MWh/h)","Reserve VRE node (MWh/h)",' >> fn_events;
printf '"Price CUR/MWh",' >> fn_events;
printf '"Non synchronous shadow value (CUR/MW)"\n' >> fn_events;

for {(g,n) in gn_demand}
  {
    for {t in time_in_use}
      {
       printf (if p_time_jump[t] > 1 then '\n' else '') >> fn_events;
       printf '%s,%s,%s,', g, n, t >> fn_events;
       for {x in oneMember : print_event_n[g,n,t]}
         { 
           printf '%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,', 
             + (if use_capacity_margin && mode_invest then v_capacitySlack[g, n, t]),
             + (if (g,n) in gn_reserve then v_reserveSlack_n[g, n, t]),
			 + sum{(g,n,ng) in gnng : ng in ng_reserve} v_reserveSlack_ng[ng,t],
			 + gn_inertiaSlack[g,n,t],
             + v_slack[g, n, t],
             + v_demandSlack[g, n, t],
             + demand[g, n, t], 
             + (if (g,n) in gn_import then import[g, n, t]),
             + sum{(g, n, u) in gnu_gen union gnu_output2 diff gnu_flow} v_gen[g, n, u, t].val,
             + (sum {(g, n, u) in gnu_cf} v_gen[g,n,u,t].val),
             + (sum {(g, n, u) in gnu_inflow_noStorage} v_gen[g,n,u,t].val),
             - (sum{u in unit : (g, n, u) in gnu_storage_charging || (g,n,u) in gnu_demand_increase} v_charge[g, n, u, t]),
             + p_transfer_in[g,n,t],
             - p_transfer_out[g,n,t],
             + ( + sum {(g2, n2, u, g, n) in gnuGrid2Node2} v_convert[g2, n2, u, g, n, t] * (if p_unit[g,n,u,'use_efficiency_time_series'] then ts_unit[g,n,u,'efficiency',t] else p_unittype[u, 'conversion_eff'])
                 - sum {(g, n, u, g2, n2) in gnuGrid2Node2} v_convert[g, n, u, g2, n2, t]
               ),
             - (if (g,n) in gn_flow then inflowNoStorageCurtailedT[g,n,t]),
             - (sum {(g,n,u) in gnu_spill} v_spill[g, n, u, t].val)
             >> fn_events;
           printf '%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g', 
             + netLoad[g,n,t],
             + (if p_time_jump[t] = 1 then netLoadRamp[g,n,t]),
             + ts_reserve_n[n, t] * p_node[g, n, 'use_ts_reserve'],
             + ( sum {(u, c) in ucf_profile : (g, n, u) in gnu_cf && p_unit[g, n, u, 'reserve_increase_ratio'] && p_node[g, n, 'use_dynamic_reserve'] && unitMW[g,n,u]} ts_cf[c, t] * p_unit[g, n, u, 'reserve_increase_ratio'] * unitMW[g,n,u]
               ) * p_node[g, n, 'use_dynamic_reserve'],
             + ( sum{(g,n,u) in gnu_reserve : (g,n,u) not in gnu_flow && unitMW[g,n,u]} v_reserve[g, n, u, t] ), 
             + ( sum{(g,n,u) in gnu_reserve : (g,n,u) in gnu_flow && unitMW[g,n,u]} v_reserve[g,n,u,t] ),
             + nodeBalance[g,n,t].dual * (60 / time_period_duration),
             + (if use_non_synchronous && p_node[g,n,'non_synchronous_share'] then non_synchronousLimit[g,n,t].dual * (60 / time_period_duration))
           >> fn_events;
         }
       printf '\n' >> fn_events; 
      }
    printf '\n' >> fn_events;
  }

#########################
# Print out summary of results
printf 'Write a summary of results...\n';
param fn_summary symbolic := resfolder & "\summary_" & (if mode_invest then "I.csv" else "D.csv");
printf '"Total cost obj. function (M CUR)",%.12g,,"Minimized total system cost as ', (cost.val / 1000000) > fn_summary;
printf 'given by the solver (includes all penalty costs and curtailment payment ' >> fn_summary;
printf 'for VRE generation not curtailed)"\n' >> fn_summary;
printf '"Total cost calculated (M CUR)",%.12g,,', (sum{(g,n,u) in gnu} gnu_operationCost[g,n,u] + sum{(g,n,u) in gnu_all} gnu_investCost[g,n,u] + sum{(g,n) in gn} gn_penaltyCost[g,n] + sum{(g,n,n2) in gnn} gnn_investCost[g,n,n2]) >> fn_summary;
printf '"Total cost calculated from variables and cost parameters"\n' >> fn_summary;
printf '"...Operational cost of units (M CUR)",%.12g,,\n', sum{(g,n,u) in gnu} gnu_operationCost[g,n,u] >> fn_summary;
printf '"...Investment cost of units (M CUR)",%.12g,,\n', sum{(g,n,u) in gnu_all} gnu_investCost[g,n,u] >> fn_summary;
printf '"...Investment cost for transfers (M CUR)",%.12g,,\n', sum{(g,n,n2) in gnn} gnn_investCost[g,n,n2] >> fn_summary;
printf '"...Penalty costs (M CUR)",%.12g,,\n', sum{(g,n) in gn} gn_penaltyCost[g,n] >> fn_summary;
printf '"...Curtailment payments (M CUR)",%.12g,,\n\n', sum {(g,n,t) in gnt : (g,n) in gn_flow} (inflowNoStorageCurtailedT[g,n,t]) * curtailment_penalty / 1000000 * scale_to_annual >> fn_summary;
printf '"Time in use in years",%.12g,,"The amount of time ', time_in_use_in_years >> fn_summary;
printf 'selected by the in_use or in_use_invest in the ts_time sheet of the input data"\n' >> fn_summary;
printf '"Full time series in years",%.12g,,"The selected plus ', time_in_years >> fn_summary;
printf 'non-selected time defined in the ts_time sheet"\n' >> fn_summary;

printf '\nEmissions\n' >> fn_summary;
printf '"CO2 (Mt)",%.6g,,"System-wide annualized CO2 emissions"\n\n', sum {(g,n,u,f) in gnuFuel : (g,n,u) in gnu} unitFuel[g,n,u] * p_fuel[f, 'CO2_content_t_per_MWh'] / 1000000 >> fn_summary;

for {g in grid}
  {
	printf '"General results",%s\n', g >> fn_summary;
	printf '"VRE share (\% of annual demand)",' >> fn_summary;
	printf (if sum{(g,n) in gn_demand} 1 then '%.4g' else ''), (sum{(g,n) in gn_demand} (100 * (sum{(g,n,u) in gnu_flow} (unitGen[g,n,u])))) / sum{(g,n) in gn_demand} (p_node[g,n,'demand_MWh']) >> fn_summary;
#    printf '"Max non-synchronous share (\% of annual demand)",%.4g\n', (if sum{(g,n) in gn_demand} 1 then (100 * max{t in time_in_use} ((sum{(g,n,u) in gnu : p_unittype[u, 'non_synchronous']} v_gen[g,n,u,t].val) / (sum{(g,n) in gn_demand} demand[g,n,t])))) >> fn_summary;
	printf ',,"Energy share of VRE"\n"Loss of load (\% of annual demand)",' >> fn_summary;
    printf (if sum{(g,n) in gn_demand} 1 then '%.4g' else ''), 100 * (sum{(g,n,t) in gnt : (g,n) in gn_demand} v_slack[g,n,t].val) / (sum{(g,n,t) in gnt : (g,n) in gn_demand} demand[g,n,t]) >> fn_summary;
	printf ',,"Share of unserved energy"\n" -> ramp up constrained (\% of annual demand)",' >> fn_summary;
    printf (if use_ramps && sum{(g,n) in gn_demand} 1 then '%.4g' else ''),  100 * (sum{t in time_in_use : p_time_jump[t] = 1} rampUpLossOfLoad[g,t]) / (sum{(g,n,t) in gnt : (g,n) in gn_demand} demand[g,n,t]) >> fn_summary;
	printf ',,"Unserved energy caused by upward ramp limitations"\n' >> fn_summary;
	printf '"Excess load (\% of annual demand)",' >> fn_summary;
    printf (if sum{(g,n) in gn_demand} 1 then '%.4g' else ''), 100 * (sum{(g,n,t) in gnt : (g,n) in gn_demand} v_demandSlack[g,n,t].val) / (sum{(g,n,t) in gnt : (g,n) in gn_demand} demand[g,n,t]) >> fn_summary;
	printf ',,"Additional demand caused by minimum generation constraints"\n' >> fn_summary;
	printf '"Insufficient reserves (\% of reserve demand)",' >> fn_summary;
    printf (if sum{(g,n) in gn_reserve} reserveDemandAnnual_n[g,n] || sum{(g,ng) in gng : ng in ng_reserve} reserveDemandAnnual_ng[ng] 
	        then '%.4g' else '')
			    , 100 * (sum{(g,n,t) in gnt} gn_reserveSlack[g,n,t]) 
				  / ((sum{(g,n) in gn_reserve} reserveDemandAnnual_n[g,n] + sum{ng in ng_reserve} reserveDemandAnnual_ng[ng]) / scale_to_annual)  >> fn_summary;
	printf ',,"Share of unserved reserve"\n"Insufficient inertia (\% of inertia demand)",' >> fn_summary;
    printf (if use_inertia_limit && sum{(ng,t) in ngt : (g,ng) in gng} p_nodeGroup[ng, 'inertia_limit_MWs'] then '%.4g' else '')
	            , (sum{(ng,t) in ngt : (g,ng) in gng && use_inertia_limit && p_nodeGroup[ng, 'inertia_limit_MWs']} (v_inertiaSlack[ng,t].val / (p_nodeGroup[ng, 'inertia_limit_MWs']))) / sum{t in time_in_use} 1  >> fn_summary;
	printf ',,"Share of unserved inertia (out of total required MWs)"\n' >> fn_summary;
	printf '"Curtailment (\% of VRE gen.)",' >> fn_summary;
    printf (if (sum{(g,n,u) in gnu_flow : unitMW[g,n,u]} unitGen[g,n,u]) then '%.4g' else '')
	            , 100 * (sum{(g,n,u) in gnu_flow : unitMW[g,n,u]} unitCurtail[g,n,u]) 
				      / (sum{(g,n,u) in gnu_flow : unitMW[g,n,u]} unitGen[g,n,u]) >> fn_summary;
	printf ',,"Share of curtailed VRE out of total available VRE"\n' >> fn_summary;
	printf '" -> ramp down constrained (\% of VRE gen.)",' >> fn_summary;
    printf (if use_ramps && sum{(g,n,u) in gnu_flow : (g,n) in gn_demand} unitGen[g,n,u] then '%.4g' else '')
	            , 100 * (sum{(g,n,t) in gnt : (sum{(g,n,u) in gnu_flow} unitGen[g,n,u]) && p_time_jump[t] = 1} rampDownCurtail[g,n,t]) / (sum{(g,n,u) in gnu_flow} unitGen[g,n,u]) >> fn_summary;
	printf ',,"VRE curtailed to decrease downward ramp so that rest of the system manages to ramp up"\n' >> fn_summary;
	printf '"Peak load (MW)",' >> fn_summary;
	printf (if sum{(g,n) in gn_demand} 1 then '%.6g' else ''), max{t in time_in_use} demand_gt[g,t] >> fn_summary;
	printf ',,"Highest demand in the modelled time series"\n"Peak net load (MW)",' >> fn_summary;
	printf (if sum{(g,n) in gn_demand} 1 then '%.6g' else ''), max{t in time_in_use} netLoad_gt[g,t] >> fn_summary;
	printf ',,"Highest demand minus VRE generation and import/export time series"\n' >> fn_summary;

    printf '\n"Flexibility issues",%s\n', g >> fn_summary;
    printf '"Loss of load (max MW)",%.6g,,"Highest unserved demand in the results"\n', (if sum{(g,n) in gn_demand} 1 then max{(g,n,t) in gnt} v_slack[g,n,t].val) >> fn_summary;
    printf '"Excess load (max MW)",%.6g,,"Highest forced demand increase to avoid model infeasibility"\n', (if sum{(g,n) in gn_demand} 1 then -max{(g,n,t) in gnt} v_demandSlack[g,n,t].val) >> fn_summary;
    printf '"Reserve inadequacy (max MW)",%.6g,,"Highest deficit of upward reserve"\n', (if mode_dispatch then max{t in time_in_use} sum{(g,n) in gn} gn_reserveSlack[g,n,t]) >> fn_summary;
    printf '"Insufficient inertia (TWs/a)",%.6g,,"Highest deficit in the inertia"\n', (sum{(ng,t) in ngt : (g,ng) in gng && use_inertia_limit && p_nodeGroup[ng, 'inertia_limit_MWs']} v_inertiaSlack[ng,t].val) * scale_to_annual / 1000000  >> fn_summary;
    printf '"Curtailment (max MW)",%.6g,,"Highest curtailment of VRE"\n', (if sum{(g,n) in gn_flow} 1 then max{t in time_in_use} sum{(g,n,t) in gnt : (g,n) in gn_flow} inflowNoStorageCurtailedT[g,n,t]) >> fn_summary;
    printf '"Curtailment (TWh/a)",%.6g,,"Annualized curtailment"\n', sum{(g,n,t) in gnt : (g,n) in gn_flow} (inflowNoStorageCurtailedT[g,n,t]) * scale_to_annual / 1000000 >> fn_summary;
	printf '"Model leakage (TWh/a)",%.6g,,"How much energy the model has disposed through fictitious losses"\n', sum{(g,n,u) in gnu_storage_charging} unitChargeGenLeakage[g,n,u] * ((1 - 1/p_unittype[u,'efficiency']) + (1 - p_unittype[u,'eff_charge'])) / 1000000 >> fn_summary;
    printf '"Capacity inadequacy (max MW)",%.6g,,"Highest capacity deficit in the investment mode"\n', (if use_capacity_margin && mode_invest && sum{(g,n) in gn_demand} 1 then max{(g,n,t) in gnt} v_capacitySlack[g,n,t].val) >> fn_summary;
    printf '"Spill (TWh/a)",%.6g,,"Annualized energy spill from reservoirs"\n', sum{(g,n,u) in gnu_spill} unitSpill[g,n,u] / 1000000 >> fn_summary;
    
    printf '\n"Energy balance",%s\n', g >> fn_summary;
    printf '"Demand (TWh)",%.6g,,"Annualized unflexible demand"\n', sum{(g,n,t) in gnt : (g,n) in gn_demand} -demand[g,n,t] / 1000000 * scale_to_annual >> fn_summary;
    printf '"Consume (TWh)",%.6g,,"Annualized demand from units with controllable demand\n', sum{(g,n,u) in gnu_demand_increase} -unitCharge[g,n,u] / 1000000 >> fn_summary;
    printf '"Loss of load (TWh)",%.6g,,"Annualized unserved demand"\n', (sum{(g,n,t) in gnt} v_slack[g,n,t].val) * scale_to_annual / 1000000 >> fn_summary;
    printf '"Excess load (TWh)",%.6g,,"Annualized increase of demand to keep the model feasible"\n', (sum{(g,n,t) in gnt} -v_demandSlack[g,n,t].val) * scale_to_annual / 1000000 >> fn_summary;
    printf '"Generation, fuel based (TWh)",%.6g,,"Annualized generation from units using fuels"\n', sum{(g,n,u) in gnu_gen union gnu_output2 diff gnu_storage diff gnu_flow} unitGen[g,n,u] / 1000000 >> fn_summary;
    printf '"Generation, VRE inc. river hydro (TWh)",%.6g,,"Annualized generation from VRE units"\n', sum{(g,n,u) in gnu_flow} unitGen[g,n,u] / 1000000 >> fn_summary;
    printf '"Discharge, inc. reserv. hydro (TWh)",%.6g,,"Annualized generation from reservoir hydro units"\n', sum{(g,n,u) in gnu_storage} unitGen[g,n,u] / 1000000 >> fn_summary;
    printf '"Charge (TWh)",%.6g,,"Annualized charging of storages"\n', sum{(g,n,u) in gnu_storage_charging} -unitCharge[g,n,u] / 1000000 >> fn_summary;
    printf '"Convert (TWh)",%.6g,,"Annualized conversion of energy from the node or to the node"\n', (sum{(g_input,n_input,u,g,n) in gnuGrid2Node2} unitConvertTo[g_input,n_input,u,g,n] - sum{(g,n,u,g_output,n_output) in gnuGrid2Node2} unitConvertFrom[g,n,u,g_output,n_output]) / 1000000 >> fn_summary;
    printf '"Import (TWh)",%.6g,,"Annualized import/export from the time series"\n', sum{(g,n) in gn} p_node[g,n,'import_MWh'] / 1000000 >> fn_summary;
    printf '"Transfer losses (TWh)",%.6g,,"Annualized losses in the energy transfers between the nodes"\n', sum{(g,n,n2) in gnn} transfer_losses_TWh[g,n,n2] >> fn_summary;

    printf '\n"Costs",%s\n', g >> fn_summary;
    printf '"Cost operations (M CUR)",%.6g\n', sum {(g,n,u) in gnu}
          ( + (if (g,n,u) in gnu_gen then unitGen[g,n,u] * p_unittype[u, 'O&M_cost_per_MWh'] / 1000000 )
            + (if (g,n,u) in gnu_demand_increase then unitCharge[g,n,u] * p_unittype[u, 'O&M_cost_per_MWh'] / 1000000 )
            + (if (g,n,u) in gnu_convertOutput then sum{(g_input,n_input,u,g,n) in gnuGrid2Node2} unitConvertFrom[g_input,n_input,u,g,n] * p_unittype[u, 'O&M_cost_per_MWh'] / 1000000 )
            + sum {(g,n,u,f) in gnuFuel : (g,n,u) in gnu} unitFuel[g,n,u] * p_fuel[f, 'fuel_price_per_MWh'] / 1000000 
            + sum {(g,n,u,f) in gnuFuel : (g,n,u) in gnu} unitFuel[g,n,u] * p_fuel[f, 'CO2_content_t_per_MWh'] * co2_cost / 1000000
            + (if u in unit_startup then unitStartup[g,n,u] * p_unittype[u, 'startup_cost'] / 1000000 )
          ) >> fn_summary;
    printf '"Cost investments (M CUR)",%.6g\n', 
          ( + sum {(g,n,u) in gnu} gnu_investCost[g,n,u]
            + sum {(g,n,n2) in gnn} gnn_investCost[g,n,n2]
          ) >> fn_summary;
    printf '"Fixed annual costs (M CUR)",%.6g\n', 
          ( + sum {(g,n,u) in gnu} unitMW[g,n,u] * p_unittype[u, 'fixed_cost_per_kW_per_year'] / 1000
          ) >> fn_summary;
    printf '"Cost loss of load (M CUR)",%.6g\n', sum {(g,n,t) in gnt} v_slack[g, n, t] * loss_of_load_penalty / 1000000 * scale_to_annual >> fn_summary;
    printf '"Cost excess load (M CUR)",%.6g\n', sum {(g,n,t) in gnt} -v_demandSlack[g, n, t] * loss_of_load_penalty / 1000000 * scale_to_annual >> fn_summary;
    printf '"Cost curtailment (M CUR)",%.6g\n', sum {(g,n,t) in gnt : (g,n) in gn_flow} (inflowNoStorageCurtailedT[g,n,t]) * curtailment_penalty / 1000000 * scale_to_annual >> fn_summary;
    printf '"Cost of insufficient reserves (M CUR)",%.6g\n', sum{(g,n) in gn_reserve} (sum{(g,n,t) in gnt} gn_reserveSlack[g,n,t]) * loss_of_reserves_penalty / 1000000 * scale_to_annual >> fn_summary;
    printf '"Cost of insufficient inertia (M CUR)",%.6g\n', sum{(ng,t) in ngt : (g,ng) in gng && use_inertia_limit && p_nodeGroup[ng, 'inertia_limit_MWs']} v_inertiaSlack[ng,t].val * lack_of_inertia_penalty / 1000000 * scale_to_annual >> fn_summary;
    printf '"Cost of insufficient capacity (M CUR)",%.6g\n', (if use_capacity_margin && mode_invest then (sum{(g,n,t) in gnt} v_capacitySlack[g,n,t].val) * lack_of_capacity_penalty / 1000000 * scale_to_annual) >> fn_summary;

	printf 	( if sum{(g,n,u) in gnu_all_inputs_outputs} 1 then '\n"Unit type","Utilization (\%)"\n' else '' ) >> fn_summary;
		for {u in unit : sum{(g,n,u) in gnu_all_inputs_outputs} 1}
		{ 
			printf '%s,%.4g\n', u, (if (sum{(g,n,u) in gnu_gen union gnu_output2} unitMW[g,n,u]) then (sum{(g,n,u) in gnu union gnu_output2} ( unitGen[g,n,u] + (if (g,n,u) in gnu_storage_charging union gnu_demand_increase then unitCharge[g,n,u]) + sum{(g_input,n_input,u,g,n) in gnuGrid2Node2} unitConvertTo[g_input,n_input,u,g,n])) 
																										/ (sum{(g,n,u) in gnu union gnu_output2} unitMW[g,n,u]) / 87.60 else 0) 
				+ (if sum{(g,n,u) in gnu_convertOutput} unitMW[g,n,u] then sum{(g_input,n_input,u,g,n) in gnuGrid2Node2} (unitConvertTo[g_input,n_input,u,g,n] / unitMW[g,n,u]) / 87.60)
				+ (if sum{(g,n,u) in gnu_convertInput} unitMW_convertFrom[g,n,u] then sum{(g,n,u,g_output,n_output) in gnuGrid2Node2} (unitConvertFrom[g,n,u,g_output,n_output] / unitMW_convertFrom[g,n,u]) / 87.60)
				+ (if sum{(g,n,u) in gnu_demand_increase} unitMW[g,n,u]	then sum{(g,n,u) in gnu_demand_increase} (unitCharge[g,n,u] / unitMW[g,n,u]) / 87.60)
																										>> fn_summary;
		}

    printf ( if sum{(g,n,u) in gnu_all_inputs_outputs} 1 then '\n"Unit type","Capacity (MW)"\n' else '' ) >> fn_summary;
    for {u in unit : sum{(g,n,u) in gnu_all_inputs_outputs} 1 }
      { 
        printf '%s,%.6g\n', u, + sum{(g,n,u) in gnu} (if (g,n,u) in gnu_demand_increase then -unitMW[g,n,u] else unitMW[g,n,u])
							   + sum{(g,n,u) in gnu_output2} unitMW[g,n,u]
							   + sum{(g,n,u) in gnu_convertInput} unitMW_convertFrom[g,n,u] >> fn_summary;
      }

    printf ( if sum{(g,n,u) in gnu_all_storage} 1 then '\n"Unit type","Capacity (MWh)"\n' else '' ) >> fn_summary;
    for {u in unit : sum{(g,n,u) in gnu_all_storage} 1 }
      { 
        printf '%s,%.6g\n', u, + sum{(g,n,u) in gnu_storage} unitMWh[g,n,u] >> fn_summary;
      }

    printf ( if sum{(g,n,u) in gnu_all_inputs_outputs} 1 then '\n"Unit type","Energy (TWh)"\n' else '' ) >> fn_summary;
    for {u in unit : sum{(g,n,u) in gnu_all_inputs_outputs} 1}
      { 
        printf '%s,%.6g\n', u, ( + sum{(g,n,u) in gnu_gen union gnu_output2} unitGen[g,n,u] 
								 - sum{(g,n,u) in gnu_storage_charging union gnu_demand_increase} unitCharge[g,n,u]
                                 - sum{(g,n,u,g_output,n_output) in gnuGrid2Node2} (unitConvertFrom[g,n,u,g_output,n_output])
                                 + sum{(g_input,n_input,u,g,n) in gnuGrid2Node2} (unitConvertTo[g_input,n_input,u,g,n])
                               ) /1000000 >> fn_summary;
      }

    printf ( if sum{(g,n,n2) in gnn} 1 then '\n"Transfer","Utilization (\%)"\n' else '' ) >> fn_summary;
    for {(g,n,n2) in gnn}
      {
        printf '"%s - %s",%.4g\n', n, n2, (if transferRightward_MW[g,n,n2] then transferRightward_TWh[g,n,n2] / transferRightward_MW[g,n,n2] / 8760 * 100000000) >> fn_summary;
        printf '"%s - %s",%.4g\n', n2, n, (if transferLeftward_MW[g,n,n2] then transferLeftward_TWh[g,n,n2] / transferLeftward_MW[g,n,n2] / 8760 * 100000000) >> fn_summary;
      }

    printf ( if sum{(g,n,n2) in gnn} 1 then '\n"Transfer","Capacity (MW)"\n' else '' ) >> fn_summary;
    for {(g,n,n2) in gnn}
      {
        printf '"%s - %s",%.6g\n', n, n2, transferRightward_MW[g,n,n2] >> fn_summary;
        printf '"%s - %s",%.6g\n', n2, n, transferLeftward_MW[g,n,n2] >> fn_summary;
      }

    printf ( if sum{(g,n,n2) in gnn} 1 then '\n"Transfer","Energy (TWh)"\n' else '' ) >> fn_summary;
    for {(g,n,n2) in gnn}
      {
        printf '"%s - %s",%.6g\n', n, n2, transferRightward_TWh[g,n,n2] >> fn_summary;
        printf '"%s - %s",%.6g\n', n2, n, transferLeftward_TWh[g,n,n2] >> fn_summary;
      }
    printf '\n\n', g >> fn_summary;
  } 
   
#########################
# Print out unit results from the model
printf 'Write unit results...\n';
param fn_unit{g in grid} symbolic := resfolder & "\units_" & g & (if mode_invest then "_I.csv" else "_D.csv");
for {g in grid}
  {
    printf 'Node,Unit,"Capacity (MW)","Generation (MWh)","Charge/consume (MWh)","Convert (MWh)"' > fn_unit[g];
    printf ',"Curtail/spill (MWh)","Utilization (\%)","Max. ramp up (p.u.)","Max. ramp down (p.u.)"' >> fn_unit[g];
	printf ',"Reserve provision (\%)"' >> fn_unit[g];
	
    for {(g,n) in gn_demand}
      {
        for {(g,n,u) in gnu_inputs_outputs : unitMW[g,n,u]}
          {
            printf '\n%s,%s,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g', n, u,
                ( + (if (g,n,u) in gnu_gen union gnu_output2 union gnu_convertOutput then unitMW[g,n,u] else 0)
   				  + (if (g,n,u) in gnu_demand_increase then -unitMW[g,n,u] else 0)
				  + (if (g,n,u) in gnu_convertInput then -unitMW[g,n,u] else 0)
				),
                (if (g,n,u) in gnu_gen union gnu_output2 then unitGen[g,n,u] else 0),
                (if (g,n,u) in gnu_storage_charging union gnu_demand_increase then unitCharge[g,n,u] else 0),
				(if (g,n,u) in gnu_convertOutput then sum{(g_input,n_input,u,g,n) in gnuGrid2Node2} unitConvertTo[g_input,n_input,u,g,n] else 0)
                + (if (g,n,u) in gnu_convertInput then sum{(g,n,u,g_output,n_output) in gnuGrid2Node2} -unitConvertFrom[g,n,u,g_output,n_output] else 0),
                (if (g,n,u) in gnu_spill then unitSpill[g,n,u] + (if (g,n,u) in gnu_flow && unitMW[g,n,u] then unitCurtail[g,n,u]) else 0),
                ( + if (g,n,u) in gnu_gen union gnu_output2 then unitGen[g,n,u] / unitMW[g,n,u] else 0
				  + if (g,n,u) in gnu_storage_charging union gnu_demand_increase then unitCharge[g,n,u] / unitMW[g,n,u] else 0
				  + if (g,n,u) in gnu_convertInput then sum{(g,n,u,g_output,n_output) in gnuGrid2Node2} unitConvertFrom[g,n,u,g_output,n_output] / unitMW[g,n,u] else 0
				  + if (g,n,u) in gnu_convertOutput then sum{(g_input,n_input,u,g,n) in gnuGrid2Node2} unitConvertTo[g_input,n_input,u,g,n] / unitMW[g,n,u] else 0
				) / 87.60,
                ( + if (g,n,u) in gnu_gen union gnu_output2 diff gnu_storage_charging then max{t in time_in_use} (v_gen[g,n,u,t].val - v_gen[g,n,u,t-p_time_jump[t]].val) / unitMW[g,n,u] else 0
				  + if (g,n,u) in gnu_storage_charging then max{t in time_in_use} (v_gen[g,n,u,t].val - v_charge[g,n,u,t].val - (v_gen[g,n,u,t-p_time_jump[t]].val - v_charge[g,n,u,t-p_time_jump[t]].val))  / unitMW[g,n,u] else 0
				  + if (g,n,u) in gnu_demand_increase then max{t in time_in_use} (-v_charge[g,n,u,t].val - (-v_charge[g,n,u,t-p_time_jump[t]].val))  / unitMW[g,n,u] else 0
				  + if (g,n,u) in gnu_convertInput then max{t in time_in_use} sum{(g,n,u,g_output,n_output) in gnuGrid2Node2} (-v_convert[g,n,u,g_output,n_output,t].val - (-v_convert[g,n,u,g_output,n_output,t-p_time_jump[t]].val)) / unitMW[g,n,u] else 0
				  + if (g,n,u) in gnu_convertOutput then max{t in time_in_use} sum{(g_input,n_input,u,g,n) in gnuGrid2Node2} (v_convert[g_input,n_input,u,g,n,t].val - v_convert[g_input,n_input,u,g,n,t-p_time_jump[t]].val) * (if p_unit[g,n,u,'use_efficiency_time_series'] then ts_unit[g,n,u,'efficiency',t] else p_unittype[u, 'conversion_eff']) / unitMW[g,n,u] else 0
				),
                ( + if (g,n,u) in gnu_gen union gnu_output2 diff gnu_storage_charging then min{t in time_in_use} (v_gen[g,n,u,t].val - v_gen[g,n,u,t-p_time_jump[t]].val) / unitMW[g,n,u] else 0
				  + if (g,n,u) in gnu_storage_charging then min{t in time_in_use} (v_gen[g,n,u,t].val - v_charge[g,n,u,t].val - (v_gen[g,n,u,t-p_time_jump[t]].val - v_charge[g,n,u,t-p_time_jump[t]].val))  / unitMW[g,n,u] else 0
				  + if (g,n,u) in gnu_demand_increase then min{t in time_in_use} (-v_charge[g,n,u,t].val - (-v_charge[g,n,u,t-p_time_jump[t]].val))  / unitMW[g,n,u] else 0
				  + if (g,n,u) in gnu_convertInput then min{t in time_in_use} sum{(g,n,u,g_output,n_output) in gnuGrid2Node2} (-v_convert[g,n,u,g_output,n_output,t].val - (-v_convert[g,n,u,g_output,n_output,t-p_time_jump[t]].val)) / unitMW[g,n,u] else 0
				  + if (g,n,u) in gnu_convertOutput then min{t in time_in_use} sum{(g_input,n_input,u,g,n) in gnuGrid2Node2} (v_convert[g_input,n_input,u,g,n,t].val - v_convert[g_input,n_input,u,g,n,t-p_time_jump[t]].val) * (if p_unit[g,n,u,'use_efficiency_time_series'] then ts_unit[g,n,u,'efficiency',t] else p_unittype[u, 'conversion_eff']) / unitMW[g,n,u] else 0
				),
                sum{t in time_in_use : (g,n,u) in gnu_reserve && unitMW[g,n,u]} ((v_reserve[g,n,u,t].val + (if (g,n,u) in gnu_flow then unitReserveVRET[g,n,u,t])) / unitMW[g,n,u]) / 87.60
              >> fn_unit[g];
          }
        printf '\n%s,"loss of load",,%.8g,,,,,%.8g,%.8g,', n, 
            sum{t in time_in_use} v_slack[g,n,t].val * scale_to_annual, 
            (max{t in time_in_use} (v_slack[g,n,t].val - v_slack[g,n,t-p_time_jump[t]].val)) / max{t in time_in_use} demand[g,n,t],
            (min{t in time_in_use} (v_slack[g,n,t].val - v_slack[g,n,t-p_time_jump[t]].val)) / max{t in time_in_use} demand[g,n,t]
          >> fn_unit[g];
        printf '\n%s,"excess load",,%.8g,,,,,%.8g,%.8g,', n, 
            sum{t in time_in_use} -v_demandSlack[g,n,t].val * scale_to_annual, 
            (max{t in time_in_use} (-v_demandSlack[g,n,t].val + v_demandSlack[g,n,t-p_time_jump[t]].val)) / max{t in time_in_use} demand[g,n,t],
            (min{t in time_in_use} (-v_demandSlack[g,n,t].val + v_demandSlack[g,n,t-p_time_jump[t]].val)) / max{t in time_in_use} demand[g,n,t]
          >> fn_unit[g];
      }
  }


#########################
# Print out annual node results
printf 'Write annual node results...\n';
param fn_node symbolic := resfolder & "\node_" & (if mode_invest then "I.csv" else "D.csv");
printf '"Grid","Node","Loss of load (TWh)","Excess load (TWh)","Curtailed (TWh)","Spill (TWh)",' > fn_node;
printf '"Uncurtailed net load (TWh)","Generation non-VRE (TWh)","Charge (TWh)","Transfer in (TWh)",' >> fn_node;
printf '"Transfer out (TWh)","Converted (TWh)","Required reserve (TWh)","Reserves non-VRE (TWh)",' >> fn_node;
printf '"Reserves VRE (TWh)"' >> fn_node;
printf "\nnode" >> fn_colors_no_grid;   	# Print color codes to a separate file
for {(g, n) in gn}
  {
	printf ' %d', p_node[g, n, 'color_in_results'] >> fn_colors_no_grid;  	# Print color codes to a separate file
	printf '\n%s,%s,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g',
		g,
		n,
		+ sum{t in time_in_use} (v_slack[g, n, t]) * scale_to_annual / 1000000,
		+ sum{t in time_in_use} -(v_demandSlack[g, n, t]) * scale_to_annual / 1000000,
		+ sum{(g,n,u) in gnu_flow : unitMW[g,n,u]} unitCurtail[g,n,u] / 1000000,
		+ sum{(g,n,u) in gnu_spill} unitSpill[g,n,u] / 1000000,
		+ sum{(g,n,t) in gnt : (g,n) in gn_demand} netLoad[g,n,t] * scale_to_annual / 1000000,
		+ sum{u in unit : (g, n, u) in gnu_gen union gnu_output2 diff gnu_flow} unitGen[g, n, u] / 1000000,
		- (sum{u in unit_storage : (g, n, u) in gnu_storage_charging union gnu_demand_increase} unitCharge[g, n, u]) / 1000000,
		+ transfer_in_TWh[g,n],
		+ transfer_out_TWh[g,n],
		+ (sum{(g_input,n_input,u,g,n) in gnuGrid2Node2} unitConvertTo[g_input,n_input,u,g,n] 
		    - sum{(g,n,u,g_output,n_output) in gnuGrid2Node2} unitConvertFrom[g,n,u,g_output,n_output]) / 1000000,	
		+ (if (g,n) in gn_reserve then reserveDemandAnnual_n[g,n] / 1000000 else 0),
		+ (sum{(g,n,u,t) in gnut : (g,n,u) in gnu_reserve && (g,n,u) not in gnu_flow} v_reserve[g, n, u, t]) * scale_to_annual / 1000000,
		+ (sum{(g,n,u,t) in gnut : (g,n,u) in gnu_reserve && (g,n,u) in gnu_flow && unitMW[g,n,u]} unitReserveVRET[g,n,u,t]) * scale_to_annual / 1000000
	  >> fn_node;
  }


#########################
# Print out transfers from the model
printf 'Write transfers...\n';
param fn_transfer{g in grid} symbolic := resfolder & "\transfers_" & g & (if mode_invest then "_I.csv" else "_D.csv");
for {g in grid : sum{(g,n,n2) in gnn} abs(transferMW[g,n,n2])}
  {
    printf '"Node left","Node right","Rightward (TWh)","Leftward (TWh)","Utilization rightward (\%)",' > fn_transfer[g];
    printf '"Utilization leftward (\%)","Max. ramp rightward (p.u.)","Max. ramp leftward (p.u.)"' >> fn_transfer[g];
    for {(g,n,n2) in gnn : transferMW[g,n,n2]}
      {
        printf '\n%s,%s,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g', n, n2,
            + transferRightward_TWh[g,n,n2],
            + transferLeftward_TWh[g,n,n2],
            + (if transferRightward_MW[g,n,n2] then transferRightward_TWh[g,n,n2] / transferRightward_MW[g,n,n2] / 8760 * 100000000 else 0),
            + (if transferLeftward_MW[g,n,n2] then transferLeftward_TWh[g,n,n2] / transferLeftward_MW[g,n,n2] / 8760 * 100000000 else 0),
            + max{t in time_in_use} (v_transfer[g,n,n2,t].val - v_transfer[g,n,n2,t-p_time_jump[t]]) / transferMW[g,n,n2],
            - min{t in time_in_use} (v_transfer[g,n,n2,t].val - v_transfer[g,n,n2,t-p_time_jump[t]]) / transferMW[g,n,n2]
          >> fn_transfer[g];
      }
  }

#########################
# Print out generation by unitGroup from the model
printf 'Write generation by unitGroup over time...\n';
param fn_genUnitGroup{g in grid} symbolic := resfolder & "\genUnitGroup_" & g & (if mode_invest then "_I.csv" else "_D.csv");
for {g in grid}
  {
    printf 'Time,Demand+exp.-imp.' > fn_genUnitGroup[g];
	printf "\ngenUnitGroup 16777215" >> fn_colors[g];   	# Print color codes to a separate file
    printf ',"Loss of load"' >> fn_genUnitGroup[g];
    printf ' 255' >> fn_colors[g];
    for {(g, ug) in gug : not sum {(g,n,u,ug) in gnuug_convertInput} 1 && not sum {(g,n,u,ug) in gnuug} p_unittype[u, 'eff_charge']} printf ',"%s"', ug >> fn_genUnitGroup[g];
    for {(g, ug) in gug : not sum {(g,n,u,ug) in gnuug_convertInput} 1 && not sum {(g,n,u,ug) in gnuug} p_unittype[u, 'eff_charge']} printf ' %d', p_unitGroup[ug, 'color_in_results'] >> fn_colors[g];
    printf ',"Curtailed"' >> fn_genUnitGroup[g];
    printf ' 16764159' >> fn_colors[g];
    printf ',"Excess load"' >> fn_genUnitGroup[g];
    printf ' 8420607' >> fn_colors[g];
    for {(g, ug) in gug : sum {(g,n,u,ug) in gnuug_convertInput} 1} printf ',%s', ug >> fn_genUnitGroup[g];
    for {(g, ug) in gug : sum {(g,n,u,ug) in gnuug_convertInput} 1} printf ' %d', p_unitGroup[ug, 'color_in_results'] >> fn_colors[g];
    for {(g, ug) in gug : sum {(g,n,u,ug) in gnuug} p_unittype[u, 'eff_charge'] && not sum {(g,n,u,ug) in gnuug} p_unittype[u, 'efficiency']} printf ',"%s"', ug >> fn_genUnitGroup[g];
    for {(g, ug) in gug : sum {(g,n,u,ug) in gnuug} p_unittype[u, 'eff_charge'] && not sum {(g,n,u,ug) in gnuug} p_unittype[u, 'efficiency']} printf ' %d', p_unitGroup[ug, 'color_in_results'] >> fn_colors[g];
    for {(g, ug) in gug : sum {(g,n,u,ug) in gnuug} p_unittype[u, 'eff_charge'] && sum {(g,n,u,ug) in gnuug} p_unittype[u, 'efficiency']} printf ',"%s"', ug >> fn_genUnitGroup[g];
    for {(g, ug) in gug : sum {(g,n,u,ug) in gnuug} p_unittype[u, 'eff_charge'] && sum {(g,n,u,ug) in gnuug} p_unittype[u, 'efficiency']} printf ' %d', p_unitGroup[ug, 'color_in_results'] >> fn_colors[g];
    printf '\n' >> fn_genUnitGroup[g];
    for {t in time_in_use}
      {
        printf (if p_time_jump[t] > 1 then '\n' else '') >> fn_genUnitGroup[g];
        printf '%s,%.8g', t, sum{(g,n) in gn_demand} demand[g,n,t] - sum{(g,n) in gn_import} import[g,n,t] >> fn_genUnitGroup[g];
        printf ',%.8g', sum{(g,n) in gn} v_slack[g,n,t] >> fn_genUnitGroup[g];
        for {(g, ug) in gug : not sum {(g,n,u,ug) in gnuug_convertInput} 1 && not sum {(g,n,u,ug) in gnuug} p_unittype[u, 'eff_charge']} 
          {
            printf ',%.8g', sum{(g,n,u,ug) in gnuug : not (g,n,u,ug) in gnuug_convertInput && not p_unittype[u, 'eff_charge']} 
              (
                + (if (g,n,u) in gnu_gen union gnu_output2 then v_gen[g,n,u,t].val) 
				+ (if (g,n,u) in gnu_flow && unitMW[g,n,u] then unitCurtailT[g,n,u,t])
                + (if (g,n,u) in gnu_demand_increase then -v_charge[g,n,u,t].val)
				+ (if (g,n,u) in gnu_storage_charging then v_gen[g,n,u,t].val - v_charge[g,n,u,t].val)
              ) 
              + sum{(g,n,u,g2,n2,ug) in gnuGrid2Node2InverseUnitGroup : not p_unittype[u, 'eff_charge']} v_convert[g2,n2,u,g,n,t].val * (if p_unit[g,n,u,'use_efficiency_time_series'] then ts_unit[g,n,u,'efficiency',t] else p_unittype[u, 'conversion_eff']) >> fn_genUnitGroup[g];
          }
        printf ',%.8g', -sum{(g,n) in gn_flow} (inflowNoStorageCurtailedT[g,n,t]) >> fn_genUnitGroup[g];
        printf ',%.8g', -sum{(g,n) in gn} v_demandSlack[g,n,t] >> fn_genUnitGroup[g];
        for {(g, ug) in gug : sum {(g,n,u,ug) in gnuug_convertInput} 1} 
          {
            printf ',%.8g', 
                - sum{(g,n,u,output_g,output_n) in gnuGrid2Node2 : (g,n,u,ug) in gnuug_convertInput} v_convert[g,n,u,output_g,output_n,t].val >> fn_genUnitGroup[g];
		  }
        for {(g, ug) in gug : sum {(g,n,u,ug) in gnuug} p_unittype[u, 'eff_charge'] && not sum {(g,n,u,ug) in gnuug} p_unittype[u, 'efficiency']} 
          {
            printf ',%.8g', sum{(g,n,u,ug) in gnuug : (g,n,u) in gnu_demand_increase} 
               ( - v_charge[g,n,u,t].val ) >> fn_genUnitGroup[g];
          }
        for {(g, ug) in gug : sum {(g,n,u,ug) in gnuug} p_unittype[u, 'eff_charge'] && sum {(g,n,u,ug) in gnuug} p_unittype[u, 'efficiency']} 
          {
            printf ',%.8g', sum{(g,n,u,ug) in gnuug : (g,n,u) in gnu_storage_charging} 
              (
                + v_gen[g,n,u,t].val
				- v_charge[g,n,u,t].val
              ) >> fn_genUnitGroup[g]; 
          }
        printf '\n' >> fn_genUnitGroup[g];
      }
  }


#########################
# Print out generation by unit from the model
printf 'Write generation by unit over time...\n';
param fn_genUnit{g in grid} symbolic := resfolder & "\genUnit_" & g & (if mode_invest then "_I.csv" else "_D.csv");
for {g in grid}
  {
    printf 'Time,Demand+exp.-imp.' > fn_genUnit[g];
	printf "\ngenUnit 16777215" >> fn_colors[g];   	# Print color codes to a separate file
#    for {(g,n) in gn} printf ',"Curtailed"' >> fn_genUnit[g];
#    for {(g,n) in gn} printf ' 16764159' >> fn_colors[g];
    for {(g,n) in gn} printf ',"Excess load"' >> fn_genUnit[g];
    for {(g,n) in gn} printf ' 8420607' >> fn_colors[g];
    for {(g,n) in gn}
      {
       for {(g,n,u) in gnu_gen union gnu_output2 diff gnu_flow diff gnu_storage_charging diff gnu_convertOutput} printf ',"%s"', u >> fn_genUnit[g];
       for {(g,n,u) in gnu_gen union gnu_output2 diff gnu_flow diff gnu_storage_charging diff gnu_convertOutput} printf ' %d', p_unit[g, n, u, 'color_in_results'] >> fn_colors[g];
       for {(g,n,u,g2,n2) in gnuGrid2Node2Inverse} printf ',"%s"', u >> fn_genUnit[g];
       for {(g,n,u,g2,n2) in gnuGrid2Node2Inverse} printf ' %d', p_unit[g, n, u, 'color_in_results'] >> fn_colors[g];
       for {(g,n,u) in gnu_storage_charging} printf ',"%s_discharge"', u >> fn_genUnit[g];
       for {(g,n,u) in gnu_storage_charging} printf ' %d', p_unit[g, n, u, 'color_in_results'] >> fn_colors[g];
       for {(g,n,u) in gnu_inflow_noStorage} printf ',"%s"', u >> fn_genUnit[g];
       for {(g,n,u) in gnu_inflow_noStorage} printf ' %d', p_unit[g, n, u, 'color_in_results'] >> fn_colors[g];
       for {(g,n,u) in gnu_cf} printf ',"%s"', u >> fn_genUnit[g];
       for {(g,n,u) in gnu_cf} printf ' %d', p_unit[g, n, u, 'color_in_results'] >> fn_colors[g];
      }
    for {(g,n) in gn} printf ',"Loss of load"' >> fn_genUnit[g];
    for {(g,n) in gn} printf ' 255' >> fn_colors[g];
    for {(g,n) in gn}
      {
       for {(g,n,u,g2,n2) in gnuGrid2Node2} printf ',"%s"', u >> fn_genUnit[g];
       for {(g,n,u,g2,n2) in gnuGrid2Node2} printf ' %d', p_unit[g2, n2, u, 'color_in_results'] >> fn_colors[g];
       for {(g,n,u) in gnu_demand_increase} printf ',"%s"', u >> fn_genUnit[g];
       for {(g,n,u) in gnu_demand_increase} printf ' %d', p_unit[g, n, u, 'color_in_results'] >> fn_colors[g];
       for {(g,n,u) in gnu_storage_charging} printf ',"%s_charge"', u >> fn_genUnit[g];
       for {(g,n,u) in gnu_storage_charging} printf ' %d', p_unit[g, n, u, 'color_in_results'] >> fn_colors[g];
      }
    printf '\n,All' >> fn_genUnit[g];
    for {(g,n) in gn} printf ',"%s"', n >> fn_genUnit[g];
    for {(g,n) in gn}
      {
       for {(g,n,u) in gnu_gen union gnu_output2 diff gnu_flow diff gnu_storage_charging diff gnu_convertOutput} printf ',%s', n >> fn_genUnit[g];
       for {(g,n,u,g2,n2) in gnuGrid2Node2Inverse} printf ',%s', n >> fn_genUnit[g];
       for {(g,n,u) in gnu_storage_charging} printf ',%s', n >> fn_genUnit[g];
       for {(g,n,u) in gnu_inflow_noStorage} printf ',%s', n >> fn_genUnit[g];
       for {(g,n,u) in gnu_cf} printf ',%s', n >> fn_genUnit[g];
      }
    for {(g,n) in gn} printf ',%s', n >> fn_genUnit[g];
    for {(g,n) in gn}
      {
       for {(g,n,u,g2,n2) in gnuGrid2Node2} printf ',%s', n >> fn_genUnit[g];
       for {(g,n,u) in gnu_demand_increase} printf ',%s', n >> fn_genUnit[g];
       for {(g,n,u) in gnu_storage_charging} printf ',%s', n >> fn_genUnit[g];
      }
    
    printf '\n' >> fn_genUnit[g];
    for {t in time_in_use}
      {
        printf (if p_time_jump[t] > 1 then '\n' else '') >> fn_genUnit[g];
        printf '%s,%.8g', t, sum{(g,n) in gn_demand} demand[g,n,t] - sum{(g,n) in gn_import} import[g,n,t] >> fn_genUnit[g];
#        for {(g,n) in gn} printf ',%.8g', (if (g,n) in gn_flow then -(inflowNoStorageCurtailedT[g,n,t])) >> fn_genUnit[g];
        for {(g,n) in gn} printf ',%.8g', -v_demandSlack[g,n,t] >> fn_genUnit[g];
        for {(g,n) in gn}
          {
            for {(g,n,u) in gnu_gen union gnu_output2 diff gnu_flow diff gnu_storage_charging diff gnu_convertOutput}
              { printf ',%.8g', (if (g,n,u) in gnu_gen union gnu_output2 && (g,n,u) not in gnu_inflow_noStorage then v_gen[g,n,u,t].val) >> fn_genUnit[g]; }
            for {(g,n,u,g2,n2) in gnuGrid2Node2Inverse} 
              { printf ',%.8g', v_convert[g2,n2,u,g,n,t].val * (if p_unit[g,n,u,'use_efficiency_time_series'] then ts_unit[g,n,u,'efficiency',t] else p_unittype[u, 'conversion_eff']) >> fn_genUnit[g]; }
            for {(g,n,u) in gnu_storage_charging} { printf ',%.8g', v_gen[g,n,u,t].val >> fn_genUnit[g]; }
            for {(g,n,u) in gnu_inflow_noStorage} { printf ',%.8g', v_gen[g,n,u,t].val >> fn_genUnit[g]; }
            for {(g,n,u) in gnu_cf} { printf ',%.8g', v_gen[g,n,u,t].val >> fn_genUnit[g]; }
#            for {(g,n,u) in gnu_inflow_noStorage} { printf ',%.8g', v_gen[g,n,u,t].val + unitCurtailT[g,n,u,t] >> fn_genUnit[g]; }
#            for {(g,n,u) in gnu_cf} { printf ',%.8g', v_gen[g,n,u,t].val + unitCurtailT[g,n,u,t] >> fn_genUnit[g]; }
          }
        for {(g,n) in gn} printf ',%.8g', v_slack[g,n,t] >> fn_genUnit[g];
        for {(g,n) in gn}
          {
            for {(g,n,u,g2,n2) in gnuGrid2Node2} 
              { printf ',%.8g', -v_convert[g,n,u,g2,n2,t].val >> fn_genUnit[g]; }
            for {(g,n,u) in gnu_demand_increase}
              { printf ',%.8g', (if ((g,n,u) in gnu_storage_charging || (g,n,u) in gnu_demand_increase) then -v_charge[g,n,u,t].val)
                  >> fn_genUnit[g]; }
            for {(g,n,u) in gnu_storage_charging}
              {
                printf ',%.8g', 
                  (
                    - (if ((g,n,u) in gnu_storage_charging) then v_charge[g,n,u,t].val)
                  ) >> fn_genUnit[g];
              }
          }
        printf '\n' >> fn_genUnit[g];
      }
  }




#########################
# Print out storage content from the model
printf 'Write storage contents over time...\n';
param fn_state{g in grid} symbolic := resfolder & "\storageContent_" & g & (if mode_invest then "_I.csv" else "_D.csv");
for {g in grid}
  {
	printf "\nstorage" >> fn_colors[g];   	# Print color codes to a separate file
    printf 'Time' > fn_state[g];
    for {(g,n) in gn}
      {
        for {(g,n,u) in gnu_storage}
          {
            printf ' %d', p_unit[g, n, u, 'color_in_results'] >> fn_colors[g];
            printf ',"%s"', u >> fn_state[g];
          }
      }
    printf '\n' >> fn_state[g];
    for {(g,n) in gn}
      {
        for {(g,n,u) in gnu_storage}
          {
            printf ',"%s"', n >> fn_state[g];
          }
      }
    printf '\n' >> fn_state[g];
    for {t in time_in_use}
      {
        printf (if p_time_jump[t] > 1 then '\n' else '') >> fn_state[g];
        printf '%s', t >> fn_state[g];
        for {(g,n) in gn}
          {
            for {(g,n,u) in gnu_storage}
              {
                printf ',%.8g', 
                  (
                    + v_state[g,n,u,t].val 
                  ) >> fn_state[g];
              }
          }
        printf '\n' >> fn_state[g];
      }
  }

#########################
# Print out online by unit from the model
printf 'Write online status by unit over time...\n';
param fn_onlineUnit{g in grid} symbolic := resfolder & "\onlineUnit_" & g & (if mode_invest then "_I.csv" else "_D.csv");
for {g in grid : sum{(g,n,u) in gnu : u in unit_online} 1}
  {
	printf "\nonline" >> fn_colors[g];   	# Print color codes to a separate file
    printf 'Time' > fn_onlineUnit[g];
    for {(g,n) in gn}
      {
        for {(g,n,u) in gnu : u in unit_online}
          {
            printf ',"%s"', u >> fn_onlineUnit[g];
			printf ' %d', p_unit[g, n, u, 'color_in_results'] >> fn_colors[g];
          }
      }
    printf '\n' >> fn_onlineUnit[g];
    for {(g,n) in gn}
      {
        for {(g,n,u) in gnu : u in unit_online}
          {
            printf ',"%s"', n >> fn_onlineUnit[g];
          }
      }
    printf '\n' >> fn_onlineUnit[g];
    for {t in time_in_use}
      {
        printf (if p_time_jump[t] > 1 then '\n' else '') >> fn_onlineUnit[g];
        printf '%s', t >> fn_onlineUnit[g];
        for {(g,n) in gn}
          {
            for {(g,n,u) in gnu : u in unit_online}
              {
                printf ',%.8g', 
                  (
                    + v_online[g,n,u,t] 
                  ) >> fn_onlineUnit[g];
              }
          }
        printf '\n' >> fn_onlineUnit[g];
      }
  }


#########################
# Print out inertia provision by units from the model
printf 'Write inertia of units over time...\n';
param fn_inertiaUnit{g in grid} symbolic := resfolder & "\inertiaUnit_" & g & (if mode_invest then "_I.csv" else "_D.csv");
for {g in grid : use_inertia_limit && sum{(g,ng) in gng} p_nodeGroup[ng, 'inertia_limit_MWs']}
  {
	printf "\ninertia" >> fn_colors[g];   	# Print color codes to a separate file
    printf 'Time' > fn_inertiaUnit[g];
    for {(g,n) in gn}
      {
        for {(g,n,u) in gnu : p_unittype[u, 'inertia_constant_MWs_per_MW']}
          {
            printf ',"%s"', u >> fn_inertiaUnit[g];
			printf ' %d', p_unit[g, n, u, 'color_in_results'] >> fn_colors[g];
          }
      }
    printf '\n' >> fn_inertiaUnit[g];
    for {(g,n) in gn}
      {
        for {(g,n,u) in gnu : p_unittype[u, 'inertia_constant_MWs_per_MW']}
          {
            printf ',"%s"', n >> fn_inertiaUnit[g];
          }
      }
    printf '\n' >> fn_inertiaUnit[g];
    for {t in time_in_use}
      {
        printf (if p_time_jump[t] > 1 then '\n' else '') >> fn_inertiaUnit[g];
        printf '%s', t >> fn_inertiaUnit[g];
        for {(g,n) in gn}
          {
            for {(g,n,u) in gnu : p_unittype[u, 'inertia_constant_MWs_per_MW']}
              {
                printf ',%.8g', 
                  (
                    + (if (g,n,u) in gnu_gen && u in unit_online then v_online[g,n,u,t] * p_unittype[u, 'inertia_constant_MWs_per_MW'])
					+ (if (g,n,u) in gnu_gen && u not in unit_online then v_gen[g,n,u,t] * p_unittype[u, 'inertia_constant_MWs_per_MW'])
                  ) >> fn_inertiaUnit[g];
              }
          }
        printf '\n' >> fn_inertiaUnit[g];
      }
  }


#########################
# Print out reserve provision by unit from the model
printf 'Write reserve provision by unit over time...\n';
param fn_reserveUnit{g in grid} symbolic := resfolder & "\reserveUnit_" & g & (if mode_invest then "_I.csv" else "_D.csv");
for {g in grid : sum{(g,n) in gn_reserve} 1 || sum{(g,ng) in gng : ng in ng_reserve} 1}
  {
    printf 'Time' > fn_reserveUnit[g];
    for {(g,n) in gn}
      {
        for {(g,n,u) in gnu_reserve}
          {
            printf ',"%s"', u >> fn_reserveUnit[g];
          }
      }
    printf '\n' >> fn_reserveUnit[g];
    for {(g,n) in gn}
      {
        for {(g,n,u) in gnu_reserve}
          {
            printf ',"%s"', n >> fn_reserveUnit[g];
          }
        printf ',"%s"', n >> fn_reserveUnit[g];
      }
	for {(g,ng) in gng : ng in ng_reserve}
	  {
		printf ',"%s"', ng >> fn_reserveUnit[g];
	  }
    printf '\n' >> fn_reserveUnit[g];
    for {t in time_in_use}
      {
        printf (if p_time_jump[t] > 1 then '\n' else '') >> fn_reserveUnit[g];
        printf '%s', t >> fn_reserveUnit[g];
        for {(g,n) in gn}
          {
            for {(g,n,u) in gnu_reserve}
              {
                printf ',%.8g', v_reserve[g,n,u,t] >> fn_reserveUnit[g];
              }
          }
        printf '\n' >> fn_reserveUnit[g];
      }
  }
  
#########################
# Print out gnt from the model
printf 'Write node results over time...\n';
param fn_gnt{(g,n) in gn} symbolic := resfolder & "\node_t_" & g & "_" & n & (if mode_invest then "_I.csv" else "_D.csv");
for {(g,n) in gn : p_node[g,n,'print_results']}
  {
    printf 'Time, Demand (MW), Charge/consume (MW), Import (MW), Non-VRE generation (MW), ' > fn_gnt[g,n];
    printf 'VRE generation (MW), Inflow generation (MW), Transfer in (MW), Transfer out (MW), ' >> fn_gnt[g,n];
    printf 'Convert (MW), Curtailed (MW), Loss of load (MW), Excess load (MW), Spill (MW), ' >> fn_gnt[g,n];
    printf 'Net load (MW), Net load ramp (MWh/h), Precalculated reserve need (MWh/h), ' >> fn_gnt[g,n];
    printf 'Dynamic reserve need (MWh/h), Non-VRE reserve (MWh/h), VRE reserve (MWh/h), ' >> fn_gnt[g,n];
    printf 'Weighted price (CUR/MWh)\n' >> fn_gnt[g,n];
    for {t in time_in_use}
      {
        printf (if p_time_jump[t] > 1 then '\n' else '') >> fn_gnt[g,n];
        printf '%s', t >> fn_gnt[g,n];
        printf ', %.8g', (if (g,n) in gn_demand then -demand[g,n,t]) >> fn_gnt[g,n];
        printf ', %.8g', -(sum{u in unit : (g, n, u) in gnu_storage_charging || (g,n,u) in gnu_demand_increase} v_charge[g, n, u, t]) >> fn_gnt[g,n];
        printf ', %.8g', (if (g,n) in gn_import then import[g,n,t]) >> fn_gnt[g,n];
        printf ', %.8g', (sum{(g, n, u) in gnu_gen union gnu_output2 : (g,n,u) not in gnu_flow} v_gen[g, n, u, t]) >> fn_gnt[g,n]; 
        printf ', %.8g', (sum{(g, n, u) in gnu_cf} v_gen[g,n,u,t]) >> fn_gnt[g,n];
        printf ', %.8g', (sum{(g, n, u) in gnu_inflow_noStorage} v_gen[g,n,u,t]) >> fn_gnt[g,n];
        printf ', %.8g', p_transfer_in[g,n,t] >> fn_gnt[g,n];
        printf ', %.8g', - p_transfer_out[g,n,t] >> fn_gnt[g,n];
        printf ', %.8g', ( + sum {(input_g, input_n, u, g, n) in gnuGrid2Node2} v_convert[input_g, input_n, u, g, n, t] * (if p_unit[g,n,u,'use_efficiency_time_series'] then ts_unit[g,n,u,'efficiency',t] else p_unittype[u, 'conversion_eff'])
                           - sum {(g, n, u, output_g, output_n) in gnuGrid2Node2} v_convert[g, n, u, output_g, output_n, t]
                         ) >> fn_gnt[g,n];
        printf ', %.8g', v_slack[g, n, t].val >> fn_gnt[g,n];
        printf ', %.8g', v_demandSlack[g, n, t].val >> fn_gnt[g,n];
        printf ', %.8g', (sum{(g,n,u) in gnu_flow : unitMW[g,n,u]} unitCurtailT[g,n,u,t]) >> fn_gnt[g,n];
        printf ', %.8g', (sum{u in unit : (g, n, u) in gnu_spill} v_spill[g, n, u, t]) >> fn_gnt[g,n];
        printf ', %.8g', (if (g,n) in gn_demand then netLoad[g,n,t]) >> fn_gnt[g,n];
        printf ', %.8g', (if (g,n) in gn_demand && p_time_jump[t] = 1 then netLoadRamp[g,n,t]) >> fn_gnt[g,n];
        printf ', %.8g', ts_reserve_n[n, t] * p_node[g, n, 'use_ts_reserve'] >> fn_gnt[g,n];
        printf ', %.8g', ( sum {(u, c) in ucf_profile : (g, n, u, c) in gnuCfprofile && (g, n, u) not in gnu_storage} ts_cf[c, t] * p_unit[g, n, u, 'reserve_increase_ratio'] * unitMW[g,n,u] )
                           * p_node[g, n, 'use_dynamic_reserve'] >> fn_gnt[g,n];
        printf ', %.8g', ( sum{(g,n,u) in gnu_reserve : (g,n,u) not in gnu_flow} v_reserve[g, n, u, t] ) >> fn_gnt[g,n];
        printf ', %.8g', ( sum{(g,n,u) in gnu_reserve : (g,n,u) in gnu_flow && unitMW[g,n,u]} unitReserveVRET[g,n,u,t] ) >> fn_gnt[g,n];
        printf ', %.8g\n', nodeBalance[g,n,t].dual / scale_to_annual >> fn_gnt[g,n];
      }
  }
        
#########################
# Print out gt from the model
printf 'Write grid results over time...\n';
param fn_gt{g in grid} symbolic := resfolder & "\grid_t_" & g & (if mode_invest then "_I.csv" else "_D.csv");
for {g in grid}
  {
    printf 'Time, Demand (MW), Charge/consume (MW), Import (MW), Non-VRE generation (MW), ' > fn_gt[g];
    printf 'VRE Generation (MW), Inflow generation (MW), Transfer losses (MW), Convert (MW), ' >> fn_gt[g];
    printf 'Curtailed (MW), Loss of load (MW), Excess load (MW), Spill (MW), Net load (MW), ' >> fn_gt[g];
    printf 'Net load ramp (MWh/h), Precalculated reserve need (MWh/h), Dynamic reserve need (MWh/h), ' >> fn_gt[g];
    printf 'Non-VRE reserve (MWh/h), VRE reserve (MWh/h), Weighted price (CUR/MWh)\n' >> fn_gt[g];
    for {t in time_in_use}
      {
        printf '%s', t >> fn_gt[g];
        printf ', %.8g', - sum{(g,n) in gn_demand} demand[g, n, t] >> fn_gt[g];
        printf ', %.8g', -(sum{(g,n,u) in gnu : (g,n,u) in gnu_storage_charging || (g,n,u) in gnu_demand_increase} v_charge[g, n, u, t]) >> fn_gt[g];
        printf ', %.8g', sum{(g,n) in gn_import} import[g, n, t] >> fn_gt[g];
        printf ', %.8g', (sum{(g,n,u) in gnu_gen union gnu_output2 diff gnu_flow} v_gen[g, n, u, t]) >> fn_gt[g];
        printf ', %.8g', (sum {(g,n,u) in gnu_cf} v_gen[g,n,u,t].val) >> fn_gt[g];
        printf ', %.8g', sum {(g,n,u) in gnu_inflow_noStorage} v_gen[g,n,u,t].val >> fn_gt[g];
        printf ', %.8g', ( + sum{(g, n, n2) in gnn} abs(v_transfer[g, n, n2, t] * p_nodeNode[g, n, n2, 'loss']) 
                         ) >> fn_gt[g];
        printf ', %.8g', ( + sum {(g2, n2, u, g, n) in gnuGrid2Node2} v_convert[g2, n2, u, g, n, t] * (if p_unit[g,n,u,'use_efficiency_time_series'] then ts_unit[g,n,u,'efficiency',t] else p_unittype[u, 'conversion_eff'])
                           - sum {(g, n, u, g2, n2) in gnuGrid2Node2} v_convert[g, n, u, g2, n2, t]
                         ) >> fn_gt[g];
        printf ', %.8g', sum{(g,n) in gn_flow} -(inflowNoStorageCurtailedT[g,n,t]) >> fn_gt[g];
        printf ', %.8g', sum{(g,n) in gn} v_slack[g, n, t].val >> fn_gt[g];
        printf ', %.8g', sum{(g,n) in gn} v_demandSlack[g, n, t].val >> fn_gt[g];
        printf ', %.8g', (sum{(g,n,u) in gnu_spill} v_spill[g, n, u, t]) >> fn_gt[g];
        printf ', %.8g', sum{(g,n) in gn_demand} netLoad[g, n, t] >> fn_gt[g];
        printf ', %.8g', sum{(g,n) in gn_demand : p_time_jump[t] = 1} ( netLoad[g,n,t] - netLoad[g,n,t-p_time_jump[t]] ) >> fn_gt[g];
        printf ', %.8g', sum{(g,n) in gn_reserve : p_node[g, n, 'use_ts_reserve']} ts_reserve_n[n, t] + sum{(g,ng) in gng : p_nodeGroup[ng, 'use_ts_reserve']} ts_reserve_ng[ng,t] >> fn_gt[g];
        printf ', %.8g', sum{(g,n) in gn_reserve} ( sum {(g, n, u, c) in gnuCfprofile : (g, n, u) not in gnu_storage} ts_cf[c, t] * p_unit[g, n, u, 'reserve_increase_ratio'] * unitMW[g,n,u] ) 
                           * p_node[g, n, 'use_dynamic_reserve'] >> fn_gt[g];
        printf ', %.8g', ( sum{(g,n,u) in gnu_reserve : (g,n,u) not in gnu_flow} v_reserve[g, n, u, t] ) >> fn_gt[g];
        printf ', %.8g', ( sum{(g,n,u) in gnu_reserve : (g,n,u) in gnu_flow && unitMW[g,n,u]} unitReserveVRET[g,n,u,t] ) >> fn_gt[g];
        printf ', %.8g\n', if sum{(g,n) in gn_demand} demand[g,n,t] then 
                             (sum{(g,n) in gn_demand} (demand[g,n,t] * nodeBalance[g,n,t].dual / scale_to_annual))
                               / 
                             (sum{(g,n) in gn_demand} demand[g,n,t])
                           >> fn_gt[g];
      }
  }
 
#########################
# Print out annualized costs from the model
printf 'Write annualized costs...\n';
param fn_costs symbolic := resfolder & "\costs_" & (if mode_invest then "I.csv" else "D.csv");
printf 'Grid,Node,Unit,"Inv. cost (M CUR)","Fixed cost (M CUR)","O&M (M CUR)","Fuel (M CUR)","CO2 (M CUR)",' > fn_costs;
printf 'Startups (M CUR),Curtailment penalty (M CUR),Loss of load penalty (M CUR),' >> fn_costs;
printf 'Excess load penalty (M CUR),Loss of reserves penalty (M CUR),' >> fn_costs;
printf 'Lack of inertia penalty (M CUR),Lack of capacity penalty (M CUR)\n' >> fn_costs;

for {(g,n) in gn}
  {
    for {(g,n,u) in gnu_all}
      {
        printf '"%s","%s","%s"', g, n, u >> fn_costs;
        printf ',%.8g', gnu_investCost[g,n,u] >> fn_costs; 
        printf ',%.8g', unitMW[g,n,u] * p_unittype[u, 'fixed_cost_per_kW_per_year'] / 1000 >> fn_costs; 
        printf ',%.8g', (if (g,n,u) in gnu_gen then unitGen[g,n,u] * p_unittype[u, 'O&M_cost_per_MWh'] / 1000000)
                          + (if (g,n,u) in gnu_demand_increase then unitCharge[g,n,u] * p_unittype[u, 'O&M_cost_per_MWh'] / 1000000)
                          + (if (g,n,u) in gnu_convertOutput then sum{(g2,n2,u,g,n) in gnuGrid2Node2} unitConvertFrom[g2,n2,u,g,n] * p_unittype[u, 'O&M_cost_per_MWh'] / 1000000) >> fn_costs;
        printf ',%.8g', sum{(g,n,u,f) in gnuFuel : (g,n,u) in gnu} unitFuel[g,n,u] * p_fuel[f, 'fuel_price_per_MWh'] / 1000000 >> fn_costs;
        printf ',%.8g', sum{(g,n,u,f) in gnuFuel : (g,n,u) in gnu} unitFuel[g,n,u] * p_fuel[f, 'CO2_content_t_per_MWh'] * co2_cost / 1000000 >> fn_costs;
        printf ',%.8g,0,0,0,0\n', (if (g,n,u) in gnu && u in unit_startup then unitStartup[g,n,u] * p_unittype[u, 'startup_cost'] / 1000000) >> fn_costs;
      }
    printf '"%s","%s","Penalties"', g, n >> fn_costs;
    printf ',0,0,0,0,0,0,%.8g,%.8g,%.8g,%.8g,%.8g\n', 
        sum {(g,n,t) in gnt : (g,n) in gn_flow} (inflowNoStorageCurtailedT[g,n,t]) * curtailment_penalty / 1000000 * scale_to_annual,
        sum {(g,n,t) in gnt} v_slack[g, n, t] * loss_of_load_penalty / 1000000 * scale_to_annual,
        sum {(g,n,t) in gnt} v_demandSlack[g, n, t] * loss_of_load_penalty / 1000000 * scale_to_annual,
        sum {(g,n,t) in gnt} gn_reserveSlack[g,n,t] * loss_of_reserves_penalty / 1000000 * scale_to_annual,
        sum {(g,n,t) in gnt} gn_inertiaSlack[g,n,t] * lack_of_inertia_penalty / 1000000 * scale_to_annual,
        sum {(g,n,t) in gnt : use_capacity_margin && mode_invest} v_capacitySlack[g, n, t] * lack_of_capacity_penalty / 1000000 * scale_to_annual
      >> fn_costs;
    for {(g, n, n2) in gnn_all}
      {
        printf '"%s","%s","%s"', g, n, n2 >> fn_costs;
        printf ',%.8g\n', (if (g,n,n2) in gnn then gnn_investCost[g,n,n2] else 0) >> fn_costs;
      }
  }


#########################
# Print out annualized costs for unitGroups
printf 'Write annualized costs for unitGroups...\n';
param fn_costs_unitGroup symbolic := resfolder & "\costs_unitGroup_" & (if mode_invest then "I.csv" else "D.csv");
printf 'Unit,"Total cost (M CUR)"\n' > fn_costs_unitGroup;
printf '"Transfer invest.",%.8g\n', sum {(g, n, n2) in gnn} (if mode_invest || mode_dispatch = 1 then gnn_investCost[g,n,n2] else gnn_investCost_I[g,n,n2]) >> fn_costs_unitGroup;
printf "\ncosts_unitGroup 167" >> fn_colors_no_grid;   	# Print color codes to a separate file
for {ug in unitGroup}
  {
	printf ' %d', p_unitGroup[ug, 'color_in_results'] >> fn_colors_no_grid;
    printf '"%s"', ug >> fn_costs_unitGroup;
    printf ',%.8g\n', 
         sum {(g,n,u,ug) in gnuug_output1 }
          ( + (if (g,n,u) in gnu then gnu_investCost[g,n,u])
            + (if (g,n,u) in gnu then unitMW[g,n,u] * p_unittype[u, 'fixed_cost_per_kW_per_year'] / 1000)
            + (if (g,n,u) in gnu_gen then unitGen[g,n,u] * p_unittype[u, 'O&M_cost_per_MWh'] / 1000000 )
            + (if (g,n,u) in gnu_demand_increase then unitCharge[g,n,u] * p_unittype[u, 'O&M_cost_per_MWh'] / 1000000 )
            + (if (g,n,u) in gnu_convertOutput then sum{(g2,n2,u,g,n) in gnuGrid2Node2} unitConvertFrom[g2,n2,u,g,n] * p_unittype[u, 'O&M_cost_per_MWh'] / 1000000 )
            + sum {(g,n,u,f) in gnuFuel : (g,n,u) in gnu} unitFuel[g,n,u] * p_fuel[f, 'fuel_price_per_MWh'] / 1000000
            + sum {(g,n,u,f) in gnuFuel : (g,n,u) in gnu} unitFuel[g,n,u] * p_fuel[f, 'CO2_content_t_per_MWh'] * co2_cost / 1000000
            + (if (g,n,u) in gnu && u in unit_startup then unitStartup[g,n,u] * p_unittype[u, 'startup_cost'] / 1000000)
          )
      >> fn_costs_unitGroup;
  }
printf " 16764159 255 8420607 6684876 3342458 6684927" >> fn_colors_no_grid;   	# Print color codes to a separate file
printf '"Curtailments",%.8g\n', sum {(g,n,t) in gnt : (g,n) in gn_flow} (inflowNoStorageCurtailedT[g,n,t]) * curtailment_penalty / 1000000 * scale_to_annual >> fn_costs_unitGroup;
printf '"Loss of load",%.8g\n', sum {(g,n,t) in gnt} v_slack[g, n, t] * loss_of_load_penalty / 1000000 * scale_to_annual >> fn_costs_unitGroup;
printf '"Excess load",%.8g\n', sum {(g,n,t) in gnt} v_demandSlack[g, n, t] * loss_of_load_penalty / 1000000 * scale_to_annual >> fn_costs_unitGroup;
printf '"Loss of reserves",%.8g\n', (sum {(g,n,t) in gnt} gn_reserveSlack[g, n, t]) * loss_of_reserves_penalty / 1000000 * scale_to_annual >> fn_costs_unitGroup;
printf '"Lack of inertia",%.8g\n', sum{(ng,t) in ngt : use_inertia_limit && p_nodeGroup[ng, 'inertia_limit_MWs']} v_inertiaSlack[ng,t].val * lack_of_inertia_penalty / 1000000 * scale_to_annual >> fn_costs_unitGroup;
printf '"Lack of capacity",%.8g\n', sum {(g,n,t) in gnt : use_capacity_margin && mode_invest} v_capacitySlack[g, n, t] * lack_of_capacity_penalty / 1000000 * scale_to_annual >> fn_costs_unitGroup;


#########################
# Print out operational costs for each time period
printf 'Write operational costs for each time period...\n';
param fn_costs_time symbolic := resfolder & "\costs_t_" & (if mode_invest then "I.csv" else "D.csv");
printf 'Time,"O&M (CUR)","Fuel (CUR)","CO2 (CUR)","Startups (CUR)","Loss of load penalty (CUR)",' > fn_costs_time;
printf '"Excess load penalty (CUR)","Curtailment penalty (CUR)"\n' >> fn_costs_time;
printf '"Lack of reserve penalty (CUR)","Lack of inertia penalty (CUR)"\n' >> fn_costs_time;
for {t in time_in_use}
 {  
   printf '%s, %.8g', t, sum{(g,n,u) in gnu_gen} v_gen[g,n,u,t].val * p_unittype[u, 'O&M_cost_per_MWh']
                           + sum {(g,n,u) in gnu_demand_increase} v_charge[g,n,u,t] * p_unittype[u, 'O&M_cost_per_MWh'] 
                           + sum{(g,n,u,g2,n2) in gnuGrid2Node2} v_convert[g,n,u,g2,n2,t] * p_unittype[u, 'O&M_cost_per_MWh'] >> fn_costs_time;
   printf ', %.8g', sum {(g, n, u, f) in gnuFuel : (g, n, u) in gnu} v_fuelUse[g,n,u,t] * p_fuel[f, 'fuel_price_per_MWh'] >> fn_costs_time;
   printf ', %.8g', sum {(g, n, u, f) in gnuFuel : (g, n, u) in gnu} v_fuelUse[g,n,u,t] * p_fuel[f, 'CO2_content_t_per_MWh'] * co2_cost >> fn_costs_time;
   printf ', %.8g', sum {(g, n, u) in gnu : u in unit_startup} v_startup[g,n,u,t] * p_unittype[u, 'startup_cost'] >> fn_costs_time;
   printf ', %.8g', sum {(g, n) in gn} v_slack[g, n, t] * loss_of_load_penalty >> fn_costs_time;
   printf ', %.8g', sum {(g, n) in gn} v_demandSlack[g, n, t] * loss_of_load_penalty >> fn_costs_time;
   printf ', %.8g', sum {(g, n) in gn_flow} inflowNoStorageCurtailedT[g,n,t] * curtailment_penalty >> fn_costs_time;
   printf ', %.8g', sum {(g, n) in gn} gn_reserveSlack[g, n, t] * loss_of_reserves_penalty >> fn_costs_time;
   printf ', %.8g', sum {(g, n) in gn} gn_inertiaSlack[g, n, t] * lack_of_inertia_penalty >> fn_costs_time;
   printf '\n' >> fn_costs_time;
 }
 
#########################
# Print out duration curves
printf 'Write duration curves...\n';
param fn_duration{g in grid} symbolic := resfolder & "\duration_" & g & (if mode_invest then "_I.csv" else "_D.csv");
for {g in grid : print_duration}
  {
    printf 'Time,"Demand (MWh)","Net load (MWh)"' > fn_duration[g];
    for {u in unit}
      { 
        printf ',%s', u >> fn_duration[g];
      }
    printf '\nNon-VRE,,' >> fn_duration[g];
    for {u in unit}
      { 
        printf ',%.8g', (if u not in unit_flow && u not in unit_demand_increase then sum{(g,n,u) in gnu} unitMW[g,n,u]) >> fn_duration[g];
      }
    printf '\nVRE and storage,,' >> fn_duration[g];
    for {u in unit}
      { 
        printf ',%.8g', (if u in unit_flow then sum{(g,n,u) in gnu} unitMW[g,n,u]) >> fn_duration[g];
      }
    printf '\n' >> fn_duration[g];
    printf{t__ in 1..card(time_in_use)} '%s,%.8g,%.8g\n', t__, 
        demand_gt[g,ind_demand[g,t__]],
        netLoad_gt[g,ind_netLoad[g,t__]]
      >> fn_duration[g];
  }


#########################
# Print out ramp duration curves
printf 'Write ramp duration curves...\n';
param fn_durationRamp{g in grid} symbolic := resfolder & "\durationRamp_" & g & (if mode_invest then "_I.csv" else "_D.csv");
for {g in grid : print_durationRamp}
  {
    printf 'Time,"Demand ramp (MWh/h)","Net load ramp (MWh/h)"' > fn_durationRamp[g];
    for {u in unit}
      { 
        printf ',%s', u >> fn_durationRamp[g];
      }
    printf '\nNon-VRE,,' >> fn_durationRamp[g];
    for {u in unit} { printf ',%.8g', (if u not in unit_flow then sum{(g,n,u) in gnu} p_unitRampUpMW[g,n,u]) >> fn_durationRamp[g]; }
    printf '\nVRE and storage,,' >> fn_durationRamp[g];
    for {u in unit} printf ',%.8g', (if u in unit_flow then sum{(g,n,u) in gnu} p_unitRampUpMW[g,n,u]) >> fn_durationRamp[g];
    printf '\n' >> fn_durationRamp[g];
    printf{t__ in 1..rampTimePeriods} '%s,%.8g,%.8g\n', t__, 
        demandRamp_gt[g,ind_demandRamp[g,t__]],
        netLoadRamp_gt[g,ind_netLoadRamp[g,t__]]
      >> fn_durationRamp[g];
  }
        

#########################
# Print out ramp capabilities 1h
printf 'Write ramp capabilities one time step...\n';
param fn_rampRoom_1h{(g, n) in gn} symbolic := resfolder & "\rampRoom_1h_" & g & "_" & n & (if mode_invest then "_I.csv" else "_D.csv");
for {(g,n) in gn : p_node[g,n,'print_results'] && (g,n) in gn_demand && use_ramps}
  {
    printf 'Time,"+upward 1h transfers","+upward 1h VRE","Upward 1h non-VRE","Net load ramp",' > fn_rampRoom_1h[g,n];
    printf '"Downward 1h non-VRE","-downward 1h VRE","-downward 1h transfers"\n' >> fn_rampRoom_1h[g,n];
    for {t in time_in_use : p_time_jump[t] = 1}
      {
        printf '%s,', t >> fn_rampRoom_1h[g,n];
        printf '%.8g,', rampRoomUpTransfer[g,n,t-p_time_jump[t]] >> fn_rampRoom_1h[g,n];
        printf '%.8g,', rampRoomUpVG[g,n,t-p_time_jump[t]] >> fn_rampRoom_1h[g,n];
        printf '%.8g,', rampRoomUp[g,n,t-p_time_jump[t]] >> fn_rampRoom_1h[g,n];
        printf '%.8g,', netLoadRamp[g,n,t] >> fn_rampRoom_1h[g,n];
        printf '%.8g,', rampRoomDown[g,n,t-p_time_jump[t]] >> fn_rampRoom_1h[g,n];
        printf '%.8g,', rampRoomDownVG[g,n,t-p_time_jump[t]] >> fn_rampRoom_1h[g,n];
        printf '%.8g', rampRoomDownTransfer[g,n,t-p_time_jump[t]] >> fn_rampRoom_1h[g,n];
        printf '\n' >> fn_rampRoom_1h[g,n];
      }
  }
        

#########################
# Print out ramp capabilities 4h
printf 'Write ramp capabilities 4h...\n';
param fn_rampRoom_4h{(g, n) in gn} symbolic := resfolder & "\rampRoom_4h_" & g & "_" & n & (if mode_invest then "_I.csv" else "_D.csv");
for {(g,n) in gn : p_node[g,n,'print_results'] && (g,n) in gn_demand && use_ramps}
  {
    printf 'Time,"+upward 4h transfers","+upward 4h VRE","Upward 4h non-VRE","4h net load ramp",' > fn_rampRoom_4h[g,n];
    printf '"Downward 4h non-VRE","-downward 4h VRE","-downward 4h transfers"\n' >> fn_rampRoom_4h[g,n];
    for {t in time_in_use : p_time_jump[t] = 1 && p_time_jump[t-1] = 1 && p_time_jump[t-2] = 1 && p_time_jump[t-3] = 1}
      {
        printf '%s,', t >> fn_rampRoom_4h[g,n];
        printf '%.8g,', rampRoomUpTransfer_4h[g,n,t-p_time_jump[t-p_time_jump[t-p_time_jump[t-p_time_jump[t]]]]] >> fn_rampRoom_4h[g,n];
        printf '%.8g,', rampRoomUpVG_4h[g,n,t-p_time_jump[t-p_time_jump[t-p_time_jump[t-p_time_jump[t]]]]] >> fn_rampRoom_4h[g,n];
        printf '%.8g,', rampRoomUp_4h[g,n,t-p_time_jump[t-p_time_jump[t-p_time_jump[t-p_time_jump[t]]]]] >> fn_rampRoom_4h[g,n];
        printf '%.8g,', netLoadRamp_4h[g,n,t] >> fn_rampRoom_4h[g,n];
        printf '%.8g,', rampRoomDown_4h[g,n,t-p_time_jump[t-p_time_jump[t-p_time_jump[t-p_time_jump[t]]]]] >> fn_rampRoom_4h[g,n];
        printf '%.8g,', rampRoomDownVG_4h[g,n,t-p_time_jump[t-p_time_jump[t-p_time_jump[t-p_time_jump[t]]]]] >> fn_rampRoom_4h[g,n];
        printf '%.8g', rampRoomDownTransfer_4h[g,n,t-p_time_jump[t-p_time_jump[t-p_time_jump[t-p_time_jump[t]]]]] >> fn_rampRoom_4h[g,n];
        printf '\n' >> fn_rampRoom_4h[g,n];
      }
  }


#########################
# Print out ramp capabilities 1h for grids
printf 'Write ramp capabilities 1h for grids...\n';
param fn_rampRoom_1h_grid{g in grid} symbolic := resfolder & "\rampRoom_1h_" & g & (if mode_invest then "_I.csv" else "_D.csv");
for {g in grid : sum{(g,n) in gn_demand} 1 && use_ramps}
  {
    printf 'Time,"+upward 1h VRE","Upward 1h ramping","Net load ramp",' > fn_rampRoom_1h_grid[g];
    printf '"Downward 1h ramping","-downward 1h VRE"\n' >> fn_rampRoom_1h_grid[g];
    for {t in time_in_use : p_time_jump[t] = 1}
      {
        printf '%s,', t >> fn_rampRoom_1h_grid[g];
        printf '%.8g,', sum{(g,n) in gn} rampRoomUpVG[g,n,t-p_time_jump[t]] >> fn_rampRoom_1h_grid[g];
        printf '%.8g,', sum{(g,n) in gn} rampRoomUp[g,n,t-p_time_jump[t]] >> fn_rampRoom_1h_grid[g];
        printf '%.8g,', netLoadRamp_gt[g,t] >> fn_rampRoom_1h_grid[g];
        printf '%.8g,', sum{(g,n) in gn} rampRoomDown[g,n,t-p_time_jump[t]] >> fn_rampRoom_1h_grid[g];
        printf '%.8g,', sum{(g,n) in gn} rampRoomDownVG[g,n,t-p_time_jump[t]] >> fn_rampRoom_1h_grid[g];
        printf '\n' >> fn_rampRoom_1h_grid[g];
      }
  }
  
  
#########################
# Print out ramp capabilities 4h for grids
printf 'Write ramp capabilities 4h for grids...\n';
param fn_rampRoom_4h_grid{g in grid} symbolic := resfolder & "\rampRoom_4h_" & g & (if mode_invest then "_I.csv" else "_D.csv");
for {g in grid : sum{(g,n) in gn_demand} 1 && use_ramps}
  {
    printf 'Time,"+upward 4h VRE","Upward 4h ramping","4h net load ramp",' > fn_rampRoom_4h_grid[g];
    printf '"Downward 4h ramping","-downward 4h VRE"\n' >> fn_rampRoom_4h_grid[g];
    for {t in time_in_use : p_time_jump[t] = 1 && p_time_jump[t-1] = 1 && p_time_jump[t-2] = 1 && p_time_jump[t-3] = 1}
      {
        printf '%s,', t >> fn_rampRoom_4h_grid[g];
        printf '%.8g,', sum{(g,n) in gn} rampRoomUpVG_4h[g,n,t-p_time_jump[t-p_time_jump[t-p_time_jump[t-p_time_jump[t]]]]] >> fn_rampRoom_4h_grid[g];
        printf '%.8g,', sum{(g,n) in gn} rampRoomUp_4h[g,n,t-p_time_jump[t-p_time_jump[t-p_time_jump[t-p_time_jump[t]]]]] >> fn_rampRoom_4h_grid[g];
        printf '%.8g,', sum{(g,n) in gn_demand} netLoadRamp_4h[g,n,t] >> fn_rampRoom_4h_grid[g];
        printf '%.8g,', sum{(g,n) in gn} rampRoomDown_4h[g,n,t-p_time_jump[t-p_time_jump[t-p_time_jump[t-p_time_jump[t]]]]] >> fn_rampRoom_4h_grid[g];
        printf '%.8g,', sum{(g,n) in gn} rampRoomDownVG_4h[g,n,t-p_time_jump[t-p_time_jump[t-p_time_jump[t-p_time_jump[t]]]]] >> fn_rampRoom_4h_grid[g];
        printf '\n' >> fn_rampRoom_4h_grid[g];
      }
  }
 


#########################
# Outputs for running a dispatch model after investments
printf 'Write outputs for the dispatch model...\n';
param fn_invested symbolic := resfolder & "\investments_" & (if mode_invest then "I.dat" else "D.dat");
printf "data;\n\n" > fn_invested;
printf "param mode_invest := 0;\n" >> fn_invested;
printf "param mode_dispatch := 2;\n\n" >> fn_invested;
printf "param p_unit :=\n" >> fn_invested;
printf {(g, n, u) in gnu, up in unitParams : up not in unitParamsSeparatePrint && p_unit[g, n, u, up]} '["%s", "%s", "%s", "%s"] %.9g,\n', g, n, u, up, p_unit[g, n, u, up] >> fn_invested;
printf {(g, n, u) in gnu, up in unitParams : up = 'invested_capacity_MW'} '["%s", "%s", "%s", "%s"] %.9g,\n', g, n, u, up, 
  + p_unit[g, n, u, up] 
  + (if (g,n,u) in gnu_invest && not (g,n,u) in gnu_convertOutput then round(v_invest[g, n, u],decs_round_invest)) 
  + (if (g,n,u) in gnu_invest && (g,n,u) in gnu_convertOutput then sum {(g2,n2,u,g,n) in gnuGrid2Node2} round(v_investConvert[g2,n2,u,g,n],decs_round_invest)) 
  >> fn_invested;
printf {(g, n, u) in gnu_storage, up in unitParams : up = 'invested_storage_MWh'} '["%s", "%s", "%s", "%s"] %.9g,\n', g, n, u, up, 
  + p_unit[g, n, u, up] 
  + (if (g,n,u) in gnu_invest then round(v_investStorage[g, n, u],decs_round_invest)) 
  >> fn_invested;
printf {(g, n, u) in gnu, up in unitParams : up = 'max_invest_MW'} '["%s", "%s", "%s", "%s"] %.9g,\n', g, n, u, up, 
  + p_unit[g, n, u, up] 
  #- p_unit[g,n,u,'invested_capacity_MW']
  - (if (g,n,u) in gnu_invest && not (g,n,u) in gnu_convertOutput then round(v_invest[g, n, u],decs_round_invest)) 
  - (if (g,n,u) in gnu_invest && (g,n,u) in gnu_convertOutput then sum {(g2,n2,u,g,n) in gnuGrid2Node2} round(v_investConvert[g2,n2,u,g,n],decs_round_invest)) 
  >> fn_invested;
printf {(g, n, u) in gnu_storage, up in unitParams : up = 'max_invest_MWh'} '["%s", "%s", "%s", "%s"] %.9g,\n', g, n, u, up, 
  + p_unit[g, n, u, up] 
  #- p_unit[g, n, u, 'invested_storage_MWh'] 
  - (if (g,n,u) in gnu_invest then round(v_investStorage[g, n, u],decs_round_invest)) 
  >> fn_invested;
printf ";\n\n" >> fn_invested;
printf "param p_nodeNode :=\n" >> fn_invested;
printf {(g, n, n2) in gnn, nnp in nodeNodeParams : nnp not in nodeNodeParamsSeparatePrint && p_nodeNode[g, n, n2, nnp]} '["%s", "%s", "%s", "%s"] %.9g,\n', g, n, n2, nnp, p_nodeNode[g, n, n2, nnp] >> fn_invested;
printf {(g, n, n2) in gnn, nnp in nodeNodeParams : nnp = 'cap.rightward_MW'} '["%s", "%s", "%s", "%s"] %.9g,\n', g, n, n2, nnp, 
  + p_nodeNode[g, n, n2, nnp] 
  >> fn_invested;
printf {(g, n, n2) in gnn, nnp in nodeNodeParams : nnp = 'cap.leftward_MW'} '["%s", "%s", "%s", "%s"] %.9g,\n', g, n, n2, nnp, 
  + p_nodeNode[g, n, n2, nnp] 
  >> fn_invested;
printf {(g, n, n2) in gnn, nnp in nodeNodeParams : nnp = 'invested_capacity_MW'} '["%s", "%s", "%s", "%s"] %.9g,\n', g, n, n2, nnp, 
  + p_nodeNode[g, n, n2, 'invested_capacity_MW'] 
  + (if (g,n,n2) in gnn_invest then round(v_investTransfer[g, n, n2],decs_round_invest)) 
  >> fn_invested;
printf {(g, n, n2) in gnn, nnp in nodeNodeParams : nnp = 'max_invest_MW'} '["%s", "%s", "%s", "%s"] %.9g,\n', g, n, n2, nnp, 
  + p_nodeNode[g, n, n2, nnp] 
  #- p_nodeNode[g, n, n2, 'invested_capacity_MW'] 
  - (if (g,n,n2) in gnn_invest then round(v_investTransfer[g, n, n2],decs_round_invest)) 
  >> fn_invested;
printf ";\n\n" >> fn_invested;
printf "param gnu_investCost_I :=\n" >> fn_invested;
printf {(g,n,u) in gnu} '["%s", "%s", "%s"] %.9g,\n', g, n, u, gnu_investCost[g,n,u] >> fn_invested;
printf ";\n\n" >> fn_invested;
printf "param gnn_investCost_I :=\n" >> fn_invested;
printf {(g,n,n2) in gnn} '["%s", "%s", "%s"] %.9g,\n', g, n, n2, gnn_investCost[g,n,n2] >> fn_invested;
printf ";\n\nend;\n" >> fn_invested;


end;
