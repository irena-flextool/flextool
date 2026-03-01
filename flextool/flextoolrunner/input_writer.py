"""
input_writer.py — Write input/ CSV files from the database.

Entry point: write_input(input_db_url, scenario_name, logger)
All write_entity / write_parameter / write_default_values calls are internal helpers.
"""
import copy
import logging
import os
import sys
import time

import spinedb_api as api
from spinedb_api import DatabaseMapping


def write_input(input_db_url: str, scenario_name: str | None, logger: logging.Logger) -> None:
    if scenario_name:
        scen_config = api.filters.scenario_filter.scenario_filter_config(scenario_name)
    with DatabaseMapping(input_db_url) as db:
        #it is faster to fetch all now than fetching multiple times
        db.fetch_all("entity")
        db.fetch_all("parameter_value")
        if scenario_name:
            api.filters.scenario_filter.scenario_filter_from_dict(db, scen_config)
        if not os.path.exists("input"):
            os.makedirs("input", exist_ok=True)
        write_default_values(db, [("node", "penalty_up"), ("node", "penalty_down")],
                             "class,paramName,default_value", "input/default_values.csv",
                             filter_in_type=["float", "str", "bool"])
        write_parameter(db, [("commodity", "price")], "commodity,commodityParam,time,pt_commodity",
                        "input/pdt_commodity.csv", filter_in_type=["1d_map"], param_print=True)
        write_parameter(db, [("commodity", "price"), ("commodity", "co2_content")], "commodity,commodityParam,p_commodity",
                        "input/p_commodity.csv", filter_in_type=["float", "str"], param_print=True)
        write_entity(db, ["commodity"], "commodity", "input/commodity.csv")
        write_entity(db, ["commodity__node"], "commodity,node", "input/commodity__node.csv")
        write_parameter(db, [("constraint", "sense")], "constraint,sense", "input/constraint__sense.csv")
        write_parameter(db, [("constraint", "constant")], "constraint,p_constraint_constant",
                        "input/p_constraint_constant.csv")
        write_parameter(db, [("model", "debug")], "debug", "input/debug.csv")
        write_entity(db, ["node", "unit", "connection"], "entity", "input/entity.csv")
        write_parameter(db, [("node", "invest_method"), ("unit", "invest_method"), ("connection", "invest_method")],
                        "entity,invest_method", "input/entity__invest_method.csv")
        write_parameter(db, [("node", "lifetime_method"), ("unit", "lifetime_method"),
                             ("connection", "lifetime_method")], "entity,lifetime_method",
                        "input/entity__lifetime_method.csv")
        write_entity(db, ["group"], "group", "input/group.csv")
        write_parameter(db, [("group", "co2_method")], "group,co2_method", "input/group__co2_method.csv")
        write_parameter(db, [("group", "invest_method")], "group,invest_method", "input/group__invest_method.csv")
        write_parameter(db, [("group", "loss_share_type")], "group,loss_share_type",
                        "input/group__loss_share_type.csv")
        write_entity(db, ["group__node"], "group,node", "input/group__node.csv")
        write_entity(db, ["group__unit", "group__connection"], "group,process", "input/group__process.csv")
        write_entity(db, ["group__unit__node", "group__connection__node"], "group,process,node",
                     "input/group__process__node.csv")
        write_parameter(db, [("group", "has_capacity_margin")], "groupCapacityMargin",
                        "input/groupCapacityMargin.csv", filter_in_value="yes", no_value=True)
        write_parameter(db, [("group", "include_stochastics")], "group", "input/groupIncludeStochastics.csv",
                        filter_in_value="yes", no_value=True)
        write_parameter(db, [("group", "has_inertia")], "groupInertia", "input/groupInertia.csv",
                        filter_in_value="yes", no_value=True)
        write_parameter(db, [("group", "output_node_flows")], "groupOutputNodeFlows",
                        "input/groupOutputNodeFlows.csv", filter_in_value="yes", no_value=True)
        write_parameter(db, [("group", "output_aggregate_flows")], "groupOutputAggregateFlows",
                        "input/groupOutputAggregateFlows.csv", filter_in_value="yes", no_value=True)
        write_parameter(db, [("model", "exclude_entity_outputs")], "value", "input/exclude_entity_outputs.csv")
        write_parameter(db, [("model", "solves")], "model,solve", "input/model__solve.csv")
        write_parameter(db, [("model", "periods_available")], "model,period_from_model", "input/periods_available.csv")
        write_entity(db, ["node"], "node", "input/node.csv")
        write_parameter(db, [("node", "constraint_capacity_coefficient")],
                        "node,constraint,p_node_constraint_capacity_coefficient",
                        "input/p_node_constraint_capacity_coefficient.csv")
        write_parameter(db, [("node", "constraint_state_coefficient")],
                        "node,constraint,p_node_constraint_state_coefficient",
                        "input/p_node_constraint_state_coefficient.csv")
        write_parameter(db, [("node", "has_balance")], "nodeBalance", "input/nodeBalance.csv",
                        filter_in_value="yes", no_value=True)
        write_parameter(db, [("node", "inflow_method")], "node,inflow_method", "input/node__inflow_method.csv")
        write_parameter(db, [("node", "node_type")], "node,node_type", "input/node__node_type.csv")
        write_parameter(db, [("node", "profile_method")], "node,profile,profile_method",
                        "input/node__profile__profile_method.csv")
        write_parameter(db, [("node", "has_storage")], "nodeState", "input/nodeState.csv", filter_in_value="yes",
                        no_value=True)
        write_parameter(db, [("node", "storage_binding_method")], "node,storage_binding_method",
                        "input/node__storage_binding_method.csv")
        write_parameter(db, [("node", "storage_nested_fix_method")], "node,storage_nested_fix_method",
                        "input/node__storage_nested_fix_method.csv")
        write_parameter(db, [("node", "storage_solve_horizon_method")], "node,storage_solve_horizon_method",
                        "input/node__storage_solve_horizon_method.csv")
        write_parameter(db, [("node", "storage_start_end_method")], "node,storage_start_end_method",
                        "input/node__storage_start_end_method.csv")
        write_parameter(db, [("node", "penalty_down"), ("node", "self_discharge_loss"), ("node", "availability"),
                             ("node", "storage_state_reference_value")], "node,nodeParam,time,pt_node",
                        "input/pt_node.csv", filter_in_type=["1d_map", "array", "time_series"],
                        filter_out_index="period", param_print=True)
        write_parameter(db, [("node", "penalty_down"), ("node", "self_discharge_loss"), ("node", "availability"),
                             ("node", "storage_state_reference_value")],
                        "node,nodeParam,branch,time_start,time,pt_node", "input/pbt_node.csv",
                        filter_in_type=["3d_map"], param_print=True)
        write_parameter(db, [("node", "inflow")], "node,time,pt_node_inflow", "input/pt_node_inflow.csv",
                        filter_in_type=["1d_map", "array", "time_series"], filter_out_index="period")
        write_parameter(db, [("node", "inflow")], "node,branch,time_start,time,pbt_node_inflow",
                        "input/pbt_node_inflow.csv", filter_in_type=["3d_map"])
        write_parameter(db, [("node", "annual_flow"),
                             ("node", "peak_inflow"),
                             ("node", "invest_forced"),
                             ("node", "invest_max_period"),
                             ("node", "invest_min_period"),
                             ("node", "retire_forced"),
                             ("node", "retire_max_period"),
                             ("node", "retire_min_period"),
                             ("node", "invest_cost"),
                             ("node", "salvage_value"),
                             ("node", "interest_rate"),
                             ("node", "lifetime"),
                             ("node", "fixed_cost"),
                             ("node", "storage_state_reference_price"),
                             ("node", "availability"),
                             ("node", "penalty_up"),
                             ("node", "penalty_down"),
                             ("node", "cumulative_max_capacity"),
                             ("node", "cumulative_min_capacity"),
                             ("node", "self_discharge_loss"),
                             ("node", "existing"),
                             ("node", "storage_state_reference_value")], "node,nodeParam,period,pd_node",
                        "input/pd_node.csv", filter_in_type=["1d_map"], filter_out_index="time", param_print=True)
        write_entity(db, ["unit", "connection"], "process", "input/process.csv")
        write_entity(db, ["connection"], "process_connection", "input/process_connection.csv")
        write_parameter(db, [("unit__outputNode", "coefficient")], "process,sink,p_process_sink_coefficient",
                        "input/p_process_sink_coefficient.csv", filter_in_type=["float", "str", "bool"])
        write_parameter(db, [("unit__inputNode", "coefficient")], "process,source,p_process_source_coefficient",
                        "input/p_process_source_coefficient.csv", filter_in_type=["float", "str", "bool"])
        write_parameter(db, [("connection", "is_DC")], "process", "input/process_nonSync_connection.csv",
                        filter_in_value="yes", no_value=True)
        write_entity(db, ["unit"], "process_unit", "input/process_unit.csv")
        write_parameter(db, [("unit__outputNode", "other_operational_cost")], "process,sink,sourceSinkTimeParam,time,pt_process_sink",
                        "input/pt_process_sink.csv", filter_in_type=["1d_map"], filter_out_index="period", param_print=True)
        write_parameter(db, [("unit__outputNode", "other_operational_cost")], "process,sink,sourceSinkPeriodParam,period,pd_process_sink",
                        "input/pd_process_sink.csv", filter_in_type=["1d_map"], filter_out_index="time", param_print=True)
        write_parameter(db, [("unit__outputNode", "other_operational_cost")],
                        "process,sink,sourceSinkTimeParam,branch,time_start,time,pbt_process_sink", "input/pbt_process_sink.csv",
                        filter_in_type=["3d_map"], param_print=True)
        write_parameter(db, [("unit__inputNode", "other_operational_cost")],
                        "process,source,sourceSinkTimeParam,time,pt_process_source", "input/pt_process_source.csv",
                        filter_in_type=["1d_map"], filter_out_index="period", param_print=True)
        write_parameter(db, [("unit__inputNode", "other_operational_cost")],
                        "process,source,sourceSinkPeriodParam,period,pd_process_source", "input/pd_process_source.csv",
                        filter_in_type=["1d_map"], filter_out_index="time", param_print=True)
        write_parameter(db, [("unit__inputNode", "other_operational_cost")],
                        "process,source,sourceSinkTimeParam,branch,time_start,time,pbt_process_source", "input/pbt_process_source.csv",
                        filter_in_type=["3d_map"], param_print=True)
        write_parameter(db, [("connection__profile", "profile_method")], "process,profile,profile_method",
                        "input/process__profile__profile_method.csv")
        write_parameter(db, [("unit__outputNode", "ramp_method"), ("unit__inputNode", "ramp_method")],
                        "process,node,ramp_method", "input/process__node__ramp_method.csv")
        write_parameter(db, [("unit", "startup_method"), ("connection", "startup_method")],
                        "process,startup_method", "input/process__startup_method.csv")
        write_parameter(db, [("unit", "conversion_method"), ("connection", "transfer_method")], "process,ct_method",
                        "input/process__ct_method.csv")
        write_entity(db, ["reserve__upDown__unit__node", "reserve__upDown__connection__node"], "process,reserve,upDown,node",
                        "input/process__reserve__upDown__node.csv", entity_dimens=[[2,0,1,3], [2,0,1,3]])
        write_parameter(db, [("profile", "profile")], "profile,time,pt_profile", "input/pt_profile.csv",
                        filter_in_type=["1d_map"], filter_out_index="period")
        write_parameter(db, [("profile", "profile")], "profile,branch,time_start,time,pbt_profile", "input/pbt_profile.csv",
                        filter_in_type=["3d_map"])
        write_parameter(db, [("profile", "profile")], "profile,period,pd_profile", "input/pd_profile.csv",
                        filter_in_type=["1d_map"], filter_out_index="time")
        write_parameter(db, [("profile", "profile")], "profile,p_profile", "input/p_profile.csv",
                        filter_in_type=["float", "str", "bool"])
        write_entity(db, ["profile"], "profile", "input/profile.csv")
        write_parameter(db, [("reserve__upDown__group", "increase_reserve_ratio"),
                             ("reserve__upDown__group", "penalty_reserve"),
                             ("reserve__upDown__group", "reservation")],
                        "reserve,upDown,group,reserveParam,p_reserve_upDown_group",
                        "input/p_reserve__upDown__group.csv", filter_in_type=["float", "str", "bool"], param_print=True)
        write_parameter(db, [("reserve__upDown__group", "reservation")],
                        "reserve,upDown,group,reserveParam,time,pt_reserve_upDown_group",
                        "input/pt_reserve__upDown__group.csv", filter_in_type=["1d_map"], filter_out_index="period", param_print=True)
        write_parameter(db, [("reserve__upDown__group", "reservation")],
                        "reserve,upDown,group,reserveParam,branch,time_start,time,pbt_reserve_upDown_group",
                        "input/pbt_reserve__upDown__group.csv", filter_in_type=["3d_map"], param_print=True)
        write_parameter(db, [("reserve__upDown__group", "reserve_method")], "reserve,upDown,group,method",
                        "input/reserve__upDown__group__method.csv")
        write_parameter(db, [("solve", "solver")], "solve,solver", "input/solver.csv")
        write_parameter(db, [("solve", "timeline_hole_multiplier")], "solve,p_hole_multiplier",
                        "input/solve_hole_multiplier.csv")
        write_parameter(db, [("solve", "solver_precommand")], "solve,solver_precommand",
                        "input/solver_precommand.csv")
        write_parameter(db, [("solve", "solver_arguments")], "solve,arguments", "input/solver_arguments.csv")
        write_parameter(db, [("solve", "highs_method"),
                             ("solve", "highs_parallel"),
                             ("solve", "highs_presolve"),
                             ("solve", "solve_mode")],
                        "param,solve,value", "input/solve_mode.csv", param_print=True, param_loc = 0)
        write_parameter(db, [("solve", "contains_solves")], "solve,include_solve",
                        "input/solve__contains_solve.csv")
        write_parameter(db, [("solve", "realized_periods")], "solve,roll,period",
                        "input/solve__realized_period_2d_map.csv", filter_in_type=["2d_map"], no_value=True)
        write_parameter(db, [("solve", "fix_storage_periods")], "solve,roll,period",
                        "input/solve__fix_storage_period_2d_map.csv", filter_in_type=["2d_map"], no_value=True)
        write_parameter(db, [("solve", "invest_periods")], "solve,roll,period",
                        "input/solve__invest_period_2d_map.csv", filter_in_type=["2d_map"], no_value=True)
        write_parameter(db, [("solve", "realized_periods")], "solve,period", "input/solve__realized_period.csv",
                        filter_in_type=["array", "1d_map"])
        write_parameter(db, [("solve", "realized_invest_periods")], "solve,invest_realized_period",
                        "input/solve__realized_invest_period.csv", filter_in_type=["array", "1d_map"])
        write_parameter(db, [("solve", "realized_invest_periods")], "solve,roll,period",
                        "input/solve__realized_invest_period_2d_map.csv", filter_in_type=["2d_map"], no_value=True)
        write_parameter(db, [("solve", "fix_storage_periods")], "solve,period",
                        "input/solve__fix_storage_period.csv", filter_in_type=["array", "1d_map"])
        write_parameter(db, [("solve", "invest_periods")], "solve,period", "input/solve__invest_period.csv",
                        filter_in_type=["array", "1d_map"])
        write_parameter(db, [("timeline", "timestep_duration")], "timeline,timestep,duration", "input/timeline.csv")
        write_parameter(db, [("unit", "efficiency"),
                             ("unit", "efficiency_at_min_load"),
                             ("unit", "min_load"),
                             ("unit", "availability"),
                             ("connection", "efficiency"),
                             ("connection", "efficiency_at_min_load"),
                             ("connection", "min_load"),
                             ("connection", "other_operational_cost"),
                             ("connection", "availability"),
                            ],
                        "process,processParam,time,pt_process", "input/pt_process.csv", filter_in_type=["1d_map"],
                        filter_out_index="period", param_print=True)
        write_entity(db, ["unit__inputNode", "connection__node__node"], "process,source", "input/process__source.csv",
                     entity_dimens=[[0,1], [0,1]])
        write_parameter(db, [("unit__outputNode", "is_non_synchronous")], "process,sink",
                        "input/process__sink_nonSync_unit.csv", filter_in_value="yes", no_value=True)
        write_entity(db, ["unit__outputNode", "connection__node__node"], "process,sink", "input/process__sink.csv",
                     entity_dimens=[[0,1], [0,2]])
        write_parameter(db, [("unit__node__profile", "profile_method")], "process,node,profile,profile_method",
                        "input/process__node__profile__profile_method.csv")
        write_parameter(db, [("unit__inputNode", "inertia_constant"),
                             ("unit__inputNode", "other_operational_cost"),
                             ("unit__inputNode", "ramp_cost"),
                             ("unit__inputNode", "ramp_speed_down"),
                             ("unit__inputNode", "ramp_speed_up")], "process,source,sourceSinkParam,p_process_source",
                        "input/p_process_source.csv", filter_in_type=["float"], param_print=True)
        write_parameter(db, [("unit__outputNode", "inertia_constant"),
                             ("unit__outputNode", "other_operational_cost"),
                             ("unit__outputNode", "ramp_cost"),
                             ("unit__outputNode", "ramp_speed_down"),
                             ("unit__outputNode", "ramp_speed_up")], "process,sink,sourceSinkParam,p_process_sink",
                        "input/p_process_sink.csv", filter_in_type=["float"], param_print=True)
        write_parameter(db, [("reserve__upDown__unit__node", "increase_reserve_ratio"),
                             ("reserve__upDown__unit__node", "large_failure_ratio"),
                             ("reserve__upDown__unit__node", "max_share"),
                             ("reserve__upDown__unit__node", "reliability"),
                             ("reserve__upDown__connection__node", "increase_reserve_ratio"),
                             ("reserve__upDown__connection__node", "large_failure_ratio"),
                             ("reserve__upDown__connection__node", "max_share"),
                             ("reserve__upDown__connection__node", "reliability")
                             ],
                        "process,reserve,upDown,node,reserveParam,p_process_reserve_upDown_node",
                        "input/p_process__reserve__upDown__node.csv",
                        filter_in_type=["float", "str", "bool"], param_print=True, dimens = [1, 2, 0, 3])
        write_parameter(db, [("unit__outputNode", "constraint_flow_coefficient"),
                             ("unit__inputNode", "constraint_flow_coefficient"),
                             ("connection__node", "constraint_flow_coefficient")],
                        "process,node,constraint,p_process_node_constraint_flow_coefficient",
                        "input/p_process_node_constraint_flow_coefficient.csv", filter_in_type=["1d_map"])
        write_parameter(db, [("unit", "constraint_capacity_coefficient"),
                             ("connection", "constraint_capacity_coefficient")],
                        "process,constraint,p_process_constraint_capacity_coefficient",
                        "input/p_process_constraint_capacity_coefficient.csv", filter_in_type=["1d_map"])
        write_parameter(db, [("unit", "delay"),
                             ("connection", "delay")],
                        "process,delay_duration,p_process_delay_weighted",
                        "input/p_process_delay_weighted.csv", filter_in_type=["1d_map"])
        write_parameter(db, [("unit", "delay"),
                             ("connection", "delay")],
                        "process,delay_duration",
                        "input/process_delay_single.csv", filter_in_type=["str", "float"])
        write_parameter(db, [("unit", "availability"),
                             ("unit", "cumulative_max_capacity"),
                             ("unit", "cumulative_min_capacity"),
                             ("unit", "efficiency"),
                             ("unit", "efficiency_at_min_load"),
                             ("unit", "existing"),
                             ("unit", "fixed_cost"),
                             ("unit", "interest_rate"),
                             ("unit", "invest_cost"),
                             ("unit", "invest_max_total"),
                             ("unit", "invest_min_total"),
                             ("unit", "lifetime"),
                             ("unit", "min_downtime"),
                             ("unit", "min_load"),
                             ("unit", "min_uptime"),
                             ("unit", "retire_max_total"),
                             ("unit", "retire_min_total"),
                             ("unit", "salvage_value"),
                             ("unit", "startup_cost"),
                             ("unit", "virtual_unitsize"),
                             ("connection", "availability"),
                             ("connection", "cumulative_max_capacity"),
                             ("connection", "cumulative_min_capacity"),
                             ("connection", "efficiency"),
                             ("connection", "existing"),
                             ("connection", "fixed_cost"),
                             ("connection", "interest_rate"),
                             ("connection", "invest_cost"),
                             ("connection", "invest_max_total"),
                             ("connection", "invest_min_total"),
                             ("connection", "lifetime"),
                             ("connection", "other_operational_cost"),
                             ("connection", "retire_max_total"),
                             ("connection", "retire_min_total"),
                             ("connection", "salvage_value"),
                             ("connection", "startup_cost"),
                             ("connection", "virtual_unitsize")
                            ],
                        "process,processParam,p_process", "input/p_process.csv",
                        filter_in_type=["float", "str", "bool"], param_print=True)
        write_parameter(db, [("node", "annual_flow"),
                             ("node", "availability"),
                             ("node", "cumulative_max_capacity"),
                             ("node", "cumulative_min_capacity"),
                             ("node", "existing"),
                             ("node", "fixed_cost"),
                             ("node", "inflow"),
                             ("node", "interest_rate"),
                             ("node", "invest_cost"),
                             ("node", "invest_forced"),
                             ("node", "invest_max_total"),
                             ("node", "invest_min_total"),
                             ("node", "lifetime"),
                             ("node", "peak_inflow"),
                             ("node", "penalty_down"),
                             ("node", "penalty_up"),
                             ("node", "retire_max_total"),
                             ("node", "retire_min_total"),
                             ("node", "salvage_value"),
                             ("node", "self_discharge_loss"),
                             ("node", "storage_state_end"),
                             ("node", "storage_state_reference_price"),
                             ("node", "storage_state_reference_value"),
                             ("node", "storage_state_start"),
                             ("node", "storate_state_end"),
                             ("node", "virtual_unitsize")
                            ], "node,nodeParam,p_node", "input/p_node.csv",
                        filter_in_type=["float", "str", "bool"], param_print=True)
        write_parameter(db, [("group__unit", "groupParam"), ("group__connection", "groupParam")],
                        "group,process,groupParam,p_group_process_s",
                        "input/p_group__process.csv", param_print=True)
        write_parameter(db, [("group", "groupParam"),
                             ("group", "capacity_margin"),
                             ("group", "co2_max_total"),
                             ("group", "co2_price"),
                             ("group", "inertia_limit"),
                             ("group", "invest_max_total"),
                             ("group", "invest_min_total"),
                             ("group", "invest_max_cumulative"),
                             ("group", "invest_min_cumulative"),
                             ("group", "max_cumulative_flow"),
                             ("group", "max_instant_flow"),
                             ("group", "min_cumulative_flow"),
                             ("group", "min_instant_flow"),
                             ("group", "non_synchronous_limit"),
                             ("group", "penalty_capacity_margin"),
                             ("group", "penalty_inertia"),
                             ("group", "penalty_non_synchronous"),
                            ], "group,groupParam,p_group", "input/p_group.csv",
                        filter_in_type=["float", "str", "bool"], param_print=True)
        write_parameter(db, [("unit", "invest_forced"),
                             ("unit", "invest_max_period"),
                             ("unit", "invest_min_period"),
                             ("unit", "retire_forced"),
                             ("unit", "retire_max_period"),
                             ("unit", "retire_min_period"),
                             ("unit", "invest_cost"),
                             ("unit", "salvage_value"),
                             ("unit", "interest_rate"),
                             ("unit", "lifetime"),
                             ("unit", "fixed_cost"),
                             ("unit", "existing"),
                             ("unit", "cumulative_max_capacity"),
                             ("unit", "cumulative_min_capacity"),
                             ("connection", "invest_forced"),
                             ("connection", "invest_max_period"),
                             ("connection", "invest_min_period"),
                             ("connection", "retire_forced"),
                             ("connection", "retire_max_period"),
                             ("connection", "retire_min_period"),
                             ("connection", "invest_cost"),
                             ("connection", "salvage_value"),
                             ("connection", "interest_rate"),
                             ("connection", "lifetime"),
                             ("connection", "fixed_cost"),
                             ("connection", "other_operational_cost"),
                             ("connection", "existing"),
                             ("connection", "cumulative_max_capacity"),
                             ("connection", "cumulative_min_capacity"),
                            ],
                        "process,processParam,period,pd_process", "input/pd_process.csv", filter_in_type=["1d_map"],
                        filter_out_index="time", param_print=True)
        write_parameter(db, [("model", "discount_rate")], "model,p_discount_rate", "input/p_discount_rate.csv")
        write_parameter(db, [("model", "discount_offset_operations")], "model,p_discount_offset_operations",
                        "input/p_discount_offset_operations.csv")
        write_parameter(db, [("model", "discount_offset_investment")], "model,p_discount_offset_investment",
                        "input/p_discount_offset_investment.csv")
        write_parameter(db, [("group", "co2_max_period"),
                             ("group", "co2_price"),
                             ("group", "inertia_limit"),
                             ("group", "invest_max_period"),
                             ("group", "invest_min_period"),
                             ("group", "invest_min_total"),
                             ("group", "max_cumulative_flow"),
                             ("group", "max_instant_flow"),
                             ("group", "min_cumulative_flow"),
                             ("group", "min_instant_flow"),
                             ("group", "non_synchronous_limit"),
                             ("group", "penalty_capacity_margin"),
                             ("group", "penalty_inertia"),
                             ("group", "penalty_non_synchronous"),
                             ], "group,groupParam,time,pt_group", "input/pdt_group.csv",
                        filter_in_type=["1d_map"], param_print=True)
        write_parameter(db, [("unit", "efficiency"),
                             ("unit", "efficiency_at_min_load"),
                             ("unit", "min_load"),
                             ("unit", "availability"),
                             ("connection", "efficiency"),
                             ("connection", "efficiency_at_min_load"),
                             ("connection", "min_load"),
                             ("connection", "other_operational_cost"),
                             ("connection", "availability")
                            ],
                        "process,processParam,branch,time_start,time,pbt_process", "input/pbt_process.csv",
                        filter_in_type=["3d_map"], param_print=True)
        write_parameter(db, [("model", "exclude_entity_outputs"),
                             ("model", "output_connection__node__node_flow_t"),
                             ("model", "output_connection_flow_separate"),
                             ("model", "output_horizon"),
                             ("model", "output_ramp_envelope"),
                             ("model", "output_unit__node_flow_t"),
                             ("model", "output_unit__node_ramp_t"),
                            ], "output,value", "input/optional_outputs.csv", param_print=True, no_entity=True)
        write_parameter(db, [("group", "output_results")], "groupOutput", "input/groupOutput.csv",
                        filter_in_value="yes", no_value=True)
        write_parameter(db, [("group", "has_non_synchronous")], "groupNonSync", "input/groupNonSync.csv",
                        filter_in_value="yes", no_value=True)
        write_default_values(db, [("model", "version")], "version", "input/db_version.csv",
                        filter_in_type=["float", "str", "bool"], only_value=True)


def write_entity(
    db,
    cl: list[str],
    header: str,
    filename: str,
    entity_dimens: list[list[int]] | None = None,
) -> None:
    entities = []
    for (i, ent_class) in enumerate(cl):
        class_entity_dimens = None
        if entity_dimens:
            class_entity_dimens = entity_dimens[i]
        for entity in db.find_entities(entity_class_name=ent_class):
            if class_entity_dimens is None:
                entities.append(','.join(entity["entity_byname"]))
            else:
                entity_dim = []
                for x in class_entity_dimens:
                    entity_dim.append(entity["entity_byname"][x])
                entities.append(','.join(entity_dim))

    with open(filename, 'w') as realfile:
        realfile.write(header + "\n")
        for entity in entities:
            realfile.write(entity + "\n")


def write_parameter(
    db,
    cl_pars: list[tuple[str, str]],
    header: str,
    filename: str,
    filter_in_type: list[str] | None = None,
    filter_out_index: str | None = None,
    filter_in_value: str | None = None,
    no_value: bool = False,
    param_print: bool = False,
    dimens: list[int] | None = None,
    param_loc: int | None = None,
    no_entity: bool | None = None,
) -> None:
    # interpret map dimensionality and map into map for later comparisons
    type_filter_map_dim = []
    if filter_in_type:
        map_found = False
        for type_filter in filter_in_type:
            if type_filter in ["1d_map", "2d_map", "3d_map", "4d_map", "5d_map"]:
                if map_found:
                    logging.error("Trying to have two different dimensionalities in the same parameter to be written out")
                    sys.exit(-1)
                map_found = True
                type_filter_map_dim = int(type_filter[0])
                filter_in_type.remove(type_filter)
        if map_found:
            filter_in_type.append("map")
    params = []
    for cl_par in cl_pars:
        params = params + db.find_parameter_values(entity_class_name=cl_par[0],
                                                    parameter_definition_name=cl_par[1])
    with open(filename, 'w') as realfile:
        realfile.write(header + "\n")
        for param in params:
            # This filter ensures that the parameter is of required type (skip to next if not)
            if filter_in_type and param["type"] not in filter_in_type:
                continue

            entity_byname = param["entity_byname"]
            if dimens:
                temp_entity_byname = [None] * len(entity_byname)
                for i, dimen in enumerate(dimens):
                    temp_entity_byname[dimen] = entity_byname[i]
                entity_byname = temp_entity_byname


            if param_print:
                if param_loc is not None:
                    time.sleep(0.1)
                    collect = []
                    for (i, byname) in enumerate(entity_byname):
                        if i == param_loc:
                            collect.append(param["parameter_definition_name"])
                        collect.append(byname)
                    first_cols = ','.join(collect)
                else:
                    if no_entity:
                        first_cols = param["parameter_definition_name"]
                    else:
                        first_cols = ','.join(entity_byname) + ',' + param["parameter_definition_name"]
            else:
                first_cols = ','.join(entity_byname)
            if param["type"] == "map":
                # If the first parameter index contains filter_out_index, then skip the parameter (maybe should be extended to other indexes)
                if filter_out_index and param["parsed_value"].index_name == filter_out_index:
                    continue
                # Check that map dimensionality matches with filter requirement (if not, then skip)
                if filter_in_type and type_filter_map_dim != api.parameter_value.from_database_to_dimension_count(param["value"], param["type"]):
                    continue
                value = param["parsed_value"]
                indexes = []
                if api.parameter_value.from_database_to_dimension_count(param["value"], param["type"]) <= 1:
                    result = [str(ind) for ind in value.indexes]
                    # Doing a zip, since there can be multiple rows in the map
                    result = list(zip(result, [str(v) for v in value.values]))
                    for res in result:
                        if no_value:
                            realfile.write(first_cols + ',' + res[0] + '\n')
                        else:
                            realfile.write(first_cols + ',' + ','.join(res) + '\n')
                else:
                    flat_map = api.convert_map_to_table(value)
                    for (i, index) in enumerate(flat_map):
                        if no_value:
                            realfile.write(first_cols + ',' + ','.join(index[:-1]) + '\n')
                        else:
                            index[-1] = str(index[-1])
                            realfile.write(first_cols + ',' + ','.join(index) + '\n')
            elif param["type"] == "array" or param["type"] == "time_series":
                for row in param["parsed_value"].values:
                    realfile.write(','.join(entity_byname) + ',' + row + '\n')
            elif param["type"] == "str" or param["type"] == "float" or param["type"] == "bool":
                # Filter based on values: only if the value is found, then data is written
                if filter_in_value and param["parsed_value"] != filter_in_value:
                    continue
                if no_value:
                    realfile.write(first_cols + '\n')
                else:
                    realfile.write(first_cols + ',' + str(param["parsed_value"]) + '\n')
            else:
                if not filter_in_type:
                    filter_in_type = ["bool", "str", "float", "array", "time_series", "map"]
                logging.error(f"Input data found in a parameter not of supported type."+
                              f"\nEntity: {','.join(entity_byname)}"+
                              f"\nParameter: {param['parameter_definition_name']}"+
                              f"\nSupported types: {filter_in_type}"+
                              f"\nParameter type: {param['type']}")
                sys.exit(-1)


def flatten_map(mapList: list, indexes: list) -> tuple[list, list]:
    result = []
    j = 0
    for (i, subMap) in enumerate(mapList):
        parent_index = indexes.pop(i + j)
        for (k, child_index) in enumerate(list(subMap.indexes)):
            comb_index = copy.deepcopy(parent_index)
            comb_index.extend([child_index])
            indexes.insert(i + j, comb_index)
            if any(isinstance(x, api.Map) for x in subMap.values):
                (result, indexes) = flatten_map(subMap.values, [indexes[i + j]])
            else:
                result.append(subMap.values[k])
            j = j + 1
        # del indexes[i + j]
        j = j - 1
    return (result, indexes)


def write_default_values(
    db,
    cl_pars: list[tuple[str, str]],
    header: str,
    filename: str,
    filter_in_type: list[str] | None = None,
    only_value: bool = False,
) -> None:
    param_defs = []
    definitions = db.find_parameter_definitions()#entity_class_name=cl_par[0], name=cl_par[1])
    for cl_par in cl_pars:
        for definition in definitions:
            if definition["entity_class_name"] == cl_par[0] and definition["name"] == cl_par[1]:
                param_defs.append(definition)
    with open(filename, 'w') as realfile:
        realfile.write(header + "\n")
        for param in param_defs:
            # This filter ensures that the parameter is of required type (skip to next if not)
            if filter_in_type and param["default_type"] not in filter_in_type:
                continue

            if param["default_type"] == "str" or param["default_type"] == "float" or param["default_type"] == "bool":
                if only_value:
                    realfile.write(str(api.from_database(param["default_value"], param["default_type"])) + '\n')
                else:
                    realfile.write(param["entity_class_name"] + "," + param["name"] + ","
                               + str(api.from_database(param["default_value"], param["default_type"])) + '\n')
            else:
                logging.error("Default_value found in a parameter definition not of supported default type"+
                              "\nParameter: " + param["parameter_definition_name"])
                sys.exit(-1)
