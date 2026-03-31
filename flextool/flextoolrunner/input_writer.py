"""
input_writer.py — Write input/ CSV files from the database.

Entry point: write_input(input_db_url, scenario_name, logger)
All write_entity / write_parameter / write_default_values calls are internal helpers.
"""
import logging
import os
from pathlib import Path
from typing import NamedTuple

import spinedb_api as api
from spinedb_api import DatabaseMapping

from flextool.flextoolrunner.runner_state import FlexToolConfigError


# ---------------------------------------------------------------------------
# Spec definitions: each entry maps 1-to-1 to a former call in write_input().
# ---------------------------------------------------------------------------

class EntitySpec(NamedTuple):
    """Spec for a write_entity() call."""
    classes: list[str]
    header: str
    filename: str
    entity_dimens: list[list[int]] | None = None


_ENTITY_SPECS: list[EntitySpec] = [
    EntitySpec(["commodity"], "commodity", "input/commodity.csv"),
    EntitySpec(["commodity__node"], "commodity,node", "input/commodity__node.csv"),
    EntitySpec(["node", "unit", "connection"], "entity", "input/entity.csv"),
    EntitySpec(["group"], "group", "input/group.csv"),
    EntitySpec(["group__node"], "group,node", "input/group__node.csv"),
    EntitySpec(["group__unit", "group__connection"], "group,process", "input/group__process.csv"),
    EntitySpec(["group__unit__node", "group__connection__node"], "group,process,node",
               "input/group__process__node.csv"),
    EntitySpec(["node"], "node", "input/node.csv"),
    EntitySpec(["unit", "connection"], "process", "input/process.csv"),
    EntitySpec(["connection"], "process_connection", "input/process_connection.csv"),
    EntitySpec(["unit"], "process_unit", "input/process_unit.csv"),
    EntitySpec(["reserve__upDown__unit__node", "reserve__upDown__connection__node"],
               "process,reserve,upDown,node", "input/process__reserve__upDown__node.csv",
               entity_dimens=[[2, 0, 1, 3], [2, 0, 1, 3]]),
    EntitySpec(["profile"], "profile", "input/profile.csv"),
    EntitySpec(["unit__inputNode", "connection__node__node"], "process,source",
               "input/process__source.csv", entity_dimens=[[0, 1], [0, 1]]),
    EntitySpec(["unit__outputNode", "connection__node__node"], "process,sink",
               "input/process__sink.csv", entity_dimens=[[0, 1], [0, 2]]),
]


_PARAMETER_SPECS: list[dict] = [
    # --- commodity ---
    {
        "cl_pars": [("commodity", "price")],
        "header": "commodity,commodityParam,time,pt_commodity",
        "filename": "input/pdt_commodity.csv",
        "filter_in_type": ["1d_map"],
        "param_print": True,
    },
    {
        "cl_pars": [("commodity", "price"), ("commodity", "co2_content")],
        "header": "commodity,commodityParam,p_commodity",
        "filename": "input/p_commodity.csv",
        "filter_in_type": ["float", "str"],
        "param_print": True,
    },
    # --- constraint ---
    {
        "cl_pars": [("constraint", "sense")],
        "header": "constraint,sense",
        "filename": "input/constraint__sense.csv",
    },
    {
        "cl_pars": [("constraint", "constant")],
        "header": "constraint,p_constraint_constant",
        "filename": "input/p_constraint_constant.csv",
    },
    # --- model (debug) ---
    {
        "cl_pars": [("model", "debug")],
        "header": "debug",
        "filename": "input/debug.csv",
    },
    # --- entity invest/lifetime methods ---
    {
        "cl_pars": [("node", "invest_method"), ("unit", "invest_method"), ("connection", "invest_method")],
        "header": "entity,invest_method",
        "filename": "input/entity__invest_method.csv",
    },
    {
        "cl_pars": [("node", "lifetime_method"), ("unit", "lifetime_method"),
                    ("connection", "lifetime_method")],
        "header": "entity,lifetime_method",
        "filename": "input/entity__lifetime_method.csv",
    },
    # --- group parameters ---
    {
        "cl_pars": [("group", "co2_method")],
        "header": "group,co2_method",
        "filename": "input/group__co2_method.csv",
    },
    {
        "cl_pars": [("group", "invest_method")],
        "header": "group,invest_method",
        "filename": "input/group__invest_method.csv",
    },
    {
        "cl_pars": [("group", "loss_share_type")],
        "header": "group,loss_share_type",
        "filename": "input/group__loss_share_type.csv",
    },
    {
        "cl_pars": [("group", "has_capacity_margin")],
        "header": "groupCapacityMargin",
        "filename": "input/groupCapacityMargin.csv",
        "filter_in_value": "yes",
        "no_value": True,
    },
    {
        "cl_pars": [("group", "include_stochastics")],
        "header": "group",
        "filename": "input/groupIncludeStochastics.csv",
        "filter_in_value": "yes",
        "no_value": True,
    },
    {
        "cl_pars": [("group", "has_inertia")],
        "header": "groupInertia",
        "filename": "input/groupInertia.csv",
        "filter_in_value": "yes",
        "no_value": True,
    },
    {
        "cl_pars": [("group", "output_node_flows")],
        "header": "groupOutputNodeFlows",
        "filename": "input/groupOutputNodeFlows.csv",
        "filter_in_value": "yes",
        "no_value": True,
    },
    {
        "cl_pars": [("group", "output_aggregate_flows")],
        "header": "groupOutputAggregateFlows",
        "filename": "input/groupOutputAggregateFlows.csv",
        "filter_in_value": "yes",
        "no_value": True,
    },
    # --- model parameters ---
    {
        "cl_pars": [("model", "exclude_entity_outputs")],
        "header": "value",
        "filename": "input/exclude_entity_outputs.csv",
    },
    {
        "cl_pars": [("model", "solves")],
        "header": "model,solve",
        "filename": "input/model__solve.csv",
    },
    {
        "cl_pars": [("model", "periods_available")],
        "header": "model,period_from_model",
        "filename": "input/periods_available.csv",
    },
    # --- node parameters ---
    {
        "cl_pars": [("node", "constraint_capacity_coefficient")],
        "header": "node,constraint,p_node_constraint_capacity_coefficient",
        "filename": "input/p_node_constraint_capacity_coefficient.csv",
    },
    {
        "cl_pars": [("node", "constraint_state_coefficient")],
        "header": "node,constraint,p_node_constraint_state_coefficient",
        "filename": "input/p_node_constraint_state_coefficient.csv",
    },
    {
        "cl_pars": [("node", "has_balance")],
        "header": "nodeBalance",
        "filename": "input/nodeBalance.csv",
        "filter_in_value": "yes",
        "no_value": True,
    },
    {
        "cl_pars": [("node", "inflow_method")],
        "header": "node,inflow_method",
        "filename": "input/node__inflow_method.csv",
    },
    {
        "cl_pars": [("node", "node_type")],
        "header": "node,node_type",
        "filename": "input/node__node_type.csv",
    },
    {
        "cl_pars": [("node", "profile_method")],
        "header": "node,profile,profile_method",
        "filename": "input/node__profile__profile_method.csv",
    },
    {
        "cl_pars": [("node", "has_storage")],
        "header": "nodeState",
        "filename": "input/nodeState.csv",
        "filter_in_value": "yes",
        "no_value": True,
    },
    {
        "cl_pars": [("node", "storage_binding_method")],
        "header": "node,storage_binding_method",
        "filename": "input/node__storage_binding_method.csv",
    },
    {
        "cl_pars": [("node", "storage_nested_fix_method")],
        "header": "node,storage_nested_fix_method",
        "filename": "input/node__storage_nested_fix_method.csv",
    },
    {
        "cl_pars": [("node", "storage_solve_horizon_method")],
        "header": "node,storage_solve_horizon_method",
        "filename": "input/node__storage_solve_horizon_method.csv",
    },
    {
        "cl_pars": [("node", "storage_start_end_method")],
        "header": "node,storage_start_end_method",
        "filename": "input/node__storage_start_end_method.csv",
    },
    {
        "cl_pars": [("node", "penalty_down"), ("node", "self_discharge_loss"), ("node", "availability"),
                    ("node", "storage_state_reference_value")],
        "header": "node,nodeParam,time,pt_node",
        "filename": "input/pt_node.csv",
        "filter_in_type": ["1d_map", "array", "time_series"],
        "filter_out_index": "period",
        "param_print": True,
    },
    {
        "cl_pars": [("node", "penalty_down"), ("node", "self_discharge_loss"), ("node", "availability"),
                    ("node", "storage_state_reference_value")],
        "header": "node,nodeParam,branch,time_start,time,pt_node",
        "filename": "input/pbt_node.csv",
        "filter_in_type": ["3d_map"],
        "param_print": True,
    },
    {
        "cl_pars": [("node", "inflow")],
        "header": "node,time,pt_node_inflow",
        "filename": "input/pt_node_inflow.csv",
        "filter_in_type": ["1d_map", "array", "time_series"],
        "filter_out_index": "period",
    },
    {
        "cl_pars": [("node", "inflow")],
        "header": "node,branch,time_start,time,pbt_node_inflow",
        "filename": "input/pbt_node_inflow.csv",
        "filter_in_type": ["3d_map"],
    },
    {
        "cl_pars": [("node", "annual_flow"),
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
                    ("node", "storage_state_reference_value")],
        "header": "node,nodeParam,period,pd_node",
        "filename": "input/pd_node.csv",
        "filter_in_type": ["1d_map"],
        "filter_out_index": "time",
        "param_print": True,
    },
    # --- process parameters ---
    {
        "cl_pars": [("unit__outputNode", "coefficient")],
        "header": "process,sink,p_process_sink_coefficient",
        "filename": "input/p_process_sink_coefficient.csv",
        "filter_in_type": ["float", "str", "bool"],
    },
    {
        "cl_pars": [("unit__inputNode", "coefficient")],
        "header": "process,source,p_process_source_coefficient",
        "filename": "input/p_process_source_coefficient.csv",
        "filter_in_type": ["float", "str", "bool"],
    },
    {
        "cl_pars": [("connection", "is_DC")],
        "header": "process",
        "filename": "input/process_nonSync_connection.csv",
        "filter_in_value": "yes",
        "no_value": True,
    },
    {
        "cl_pars": [("unit__outputNode", "other_operational_cost")],
        "header": "process,sink,sourceSinkTimeParam,time,pt_process_sink",
        "filename": "input/pt_process_sink.csv",
        "filter_in_type": ["1d_map"],
        "filter_out_index": "period",
        "param_print": True,
    },
    {
        "cl_pars": [("unit__outputNode", "other_operational_cost")],
        "header": "process,sink,sourceSinkPeriodParam,period,pd_process_sink",
        "filename": "input/pd_process_sink.csv",
        "filter_in_type": ["1d_map"],
        "filter_out_index": "time",
        "param_print": True,
    },
    {
        "cl_pars": [("unit__outputNode", "other_operational_cost")],
        "header": "process,sink,sourceSinkTimeParam,branch,time_start,time,pbt_process_sink",
        "filename": "input/pbt_process_sink.csv",
        "filter_in_type": ["3d_map"],
        "param_print": True,
    },
    {
        "cl_pars": [("unit__inputNode", "other_operational_cost")],
        "header": "process,source,sourceSinkTimeParam,time,pt_process_source",
        "filename": "input/pt_process_source.csv",
        "filter_in_type": ["1d_map"],
        "filter_out_index": "period",
        "param_print": True,
    },
    {
        "cl_pars": [("unit__inputNode", "other_operational_cost")],
        "header": "process,source,sourceSinkPeriodParam,period,pd_process_source",
        "filename": "input/pd_process_source.csv",
        "filter_in_type": ["1d_map"],
        "filter_out_index": "time",
        "param_print": True,
    },
    {
        "cl_pars": [("unit__inputNode", "other_operational_cost")],
        "header": "process,source,sourceSinkTimeParam,branch,time_start,time,pbt_process_source",
        "filename": "input/pbt_process_source.csv",
        "filter_in_type": ["3d_map"],
        "param_print": True,
    },
    {
        "cl_pars": [("connection__profile", "profile_method")],
        "header": "process,profile,profile_method",
        "filename": "input/process__profile__profile_method.csv",
    },
    {
        "cl_pars": [("unit__outputNode", "ramp_method"), ("unit__inputNode", "ramp_method")],
        "header": "process,node,ramp_method",
        "filename": "input/process__node__ramp_method.csv",
    },
    {
        "cl_pars": [("unit", "startup_method"), ("connection", "startup_method")],
        "header": "process,startup_method",
        "filename": "input/process__startup_method.csv",
    },
    {
        "cl_pars": [("unit", "conversion_method"), ("connection", "transfer_method")],
        "header": "process,ct_method",
        "filename": "input/process__ct_method.csv",
    },
    # --- profile ---
    {
        "cl_pars": [("profile", "profile")],
        "header": "profile,time,pt_profile",
        "filename": "input/pt_profile.csv",
        "filter_in_type": ["1d_map"],
        "filter_out_index": "period",
    },
    {
        "cl_pars": [("profile", "profile")],
        "header": "profile,branch,time_start,time,pbt_profile",
        "filename": "input/pbt_profile.csv",
        "filter_in_type": ["3d_map"],
    },
    {
        "cl_pars": [("profile", "profile")],
        "header": "profile,period,pd_profile",
        "filename": "input/pd_profile.csv",
        "filter_in_type": ["1d_map"],
        "filter_out_index": "time",
    },
    {
        "cl_pars": [("profile", "profile")],
        "header": "profile,p_profile",
        "filename": "input/p_profile.csv",
        "filter_in_type": ["float", "str", "bool"],
    },
    # --- reserve ---
    {
        "cl_pars": [("reserve__upDown__group", "increase_reserve_ratio"),
                    ("reserve__upDown__group", "penalty_reserve"),
                    ("reserve__upDown__group", "reservation")],
        "header": "reserve,upDown,group,reserveParam,p_reserve_upDown_group",
        "filename": "input/p_reserve__upDown__group.csv",
        "filter_in_type": ["float", "str", "bool"],
        "param_print": True,
    },
    {
        "cl_pars": [("reserve__upDown__group", "reservation")],
        "header": "reserve,upDown,group,reserveParam,time,pt_reserve_upDown_group",
        "filename": "input/pt_reserve__upDown__group.csv",
        "filter_in_type": ["1d_map"],
        "filter_out_index": "period",
        "param_print": True,
    },
    {
        "cl_pars": [("reserve__upDown__group", "reservation")],
        "header": "reserve,upDown,group,reserveParam,branch,time_start,time,pbt_reserve_upDown_group",
        "filename": "input/pbt_reserve__upDown__group.csv",
        "filter_in_type": ["3d_map"],
        "param_print": True,
    },
    {
        "cl_pars": [("reserve__upDown__group", "reserve_method")],
        "header": "reserve,upDown,group,method",
        "filename": "input/reserve__upDown__group__method.csv",
    },
    # --- solve ---
    {
        "cl_pars": [("solve", "solver")],
        "header": "solve,solver",
        "filename": "input/solver.csv",
    },
    {
        "cl_pars": [("solve", "timeline_hole_multiplier")],
        "header": "solve,p_hole_multiplier",
        "filename": "input/solve_hole_multiplier.csv",
    },
    {
        "cl_pars": [("solve", "solver_precommand")],
        "header": "solve,solver_precommand",
        "filename": "input/solver_precommand.csv",
    },
    {
        "cl_pars": [("solve", "solver_arguments")],
        "header": "solve,arguments",
        "filename": "input/solver_arguments.csv",
    },
    {
        "cl_pars": [("solve", "highs_method"),
                    ("solve", "highs_parallel"),
                    ("solve", "highs_presolve"),
                    ("solve", "solve_mode")],
        "header": "param,solve,value",
        "filename": "input/solve_mode.csv",
        "param_print": True,
        "param_loc": 0,
    },
    {
        "cl_pars": [("solve", "contains_solves")],
        "header": "solve,include_solve",
        "filename": "input/solve__contains_solve.csv",
    },
    {
        "cl_pars": [("solve", "realized_periods")],
        "header": "solve,roll,period",
        "filename": "input/solve__realized_period_2d_map.csv",
        "filter_in_type": ["2d_map"],
        "no_value": True,
    },
    {
        "cl_pars": [("solve", "fix_storage_periods")],
        "header": "solve,roll,period",
        "filename": "input/solve__fix_storage_period_2d_map.csv",
        "filter_in_type": ["2d_map"],
        "no_value": True,
    },
    {
        "cl_pars": [("solve", "invest_periods")],
        "header": "solve,roll,period",
        "filename": "input/solve__invest_period_2d_map.csv",
        "filter_in_type": ["2d_map"],
        "no_value": True,
    },
    {
        "cl_pars": [("solve", "realized_periods")],
        "header": "solve,period",
        "filename": "input/solve__realized_period.csv",
        "filter_in_type": ["array", "1d_map"],
    },
    {
        "cl_pars": [("solve", "realized_invest_periods")],
        "header": "solve,invest_realized_period",
        "filename": "input/solve__realized_invest_period.csv",
        "filter_in_type": ["array", "1d_map"],
    },
    {
        "cl_pars": [("solve", "realized_invest_periods")],
        "header": "solve,roll,period",
        "filename": "input/solve__realized_invest_period_2d_map.csv",
        "filter_in_type": ["2d_map"],
        "no_value": True,
    },
    {
        "cl_pars": [("solve", "fix_storage_periods")],
        "header": "solve,period",
        "filename": "input/solve__fix_storage_period.csv",
        "filter_in_type": ["array", "1d_map"],
    },
    {
        "cl_pars": [("solve", "invest_periods")],
        "header": "solve,period",
        "filename": "input/solve__invest_period.csv",
        "filter_in_type": ["array", "1d_map"],
    },
    # --- timeline ---
    {
        "cl_pars": [("timeline", "timestep_duration")],
        "header": "timeline,timestep,duration",
        "filename": "input/timeline.csv",
    },
    # --- process time-series / period / branch ---
    {
        "cl_pars": [("unit", "efficiency"),
                    ("unit", "efficiency_at_min_load"),
                    ("unit", "min_load"),
                    ("unit", "availability"),
                    ("connection", "efficiency"),
                    ("connection", "efficiency_at_min_load"),
                    ("connection", "min_load"),
                    ("connection", "other_operational_cost"),
                    ("connection", "availability")],
        "header": "process,processParam,time,pt_process",
        "filename": "input/pt_process.csv",
        "filter_in_type": ["1d_map"],
        "filter_out_index": "period",
        "param_print": True,
    },
    {
        "cl_pars": [("unit__outputNode", "is_non_synchronous")],
        "header": "process,sink",
        "filename": "input/process__sink_nonSync_unit.csv",
        "filter_in_value": "yes",
        "no_value": True,
    },
    {
        "cl_pars": [("unit__node__profile", "profile_method")],
        "header": "process,node,profile,profile_method",
        "filename": "input/process__node__profile__profile_method.csv",
    },
    {
        "cl_pars": [("unit__inputNode", "inertia_constant"),
                    ("unit__inputNode", "other_operational_cost"),
                    ("unit__inputNode", "ramp_cost"),
                    ("unit__inputNode", "ramp_speed_down"),
                    ("unit__inputNode", "ramp_speed_up")],
        "header": "process,source,sourceSinkParam,p_process_source",
        "filename": "input/p_process_source.csv",
        "filter_in_type": ["float"],
        "param_print": True,
    },
    {
        "cl_pars": [("unit__outputNode", "inertia_constant"),
                    ("unit__outputNode", "other_operational_cost"),
                    ("unit__outputNode", "ramp_cost"),
                    ("unit__outputNode", "ramp_speed_down"),
                    ("unit__outputNode", "ramp_speed_up")],
        "header": "process,sink,sourceSinkParam,p_process_sink",
        "filename": "input/p_process_sink.csv",
        "filter_in_type": ["float"],
        "param_print": True,
    },
    {
        "cl_pars": [("reserve__upDown__unit__node", "increase_reserve_ratio"),
                    ("reserve__upDown__unit__node", "large_failure_ratio"),
                    ("reserve__upDown__unit__node", "max_share"),
                    ("reserve__upDown__unit__node", "reliability"),
                    ("reserve__upDown__connection__node", "increase_reserve_ratio"),
                    ("reserve__upDown__connection__node", "large_failure_ratio"),
                    ("reserve__upDown__connection__node", "max_share"),
                    ("reserve__upDown__connection__node", "reliability")],
        "header": "process,reserve,upDown,node,reserveParam,p_process_reserve_upDown_node",
        "filename": "input/p_process__reserve__upDown__node.csv",
        "filter_in_type": ["float", "str", "bool"],
        "param_print": True,
        "dimens": [1, 2, 0, 3],
    },
    {
        "cl_pars": [("unit__outputNode", "constraint_flow_coefficient"),
                    ("unit__inputNode", "constraint_flow_coefficient"),
                    ("connection__node", "constraint_flow_coefficient")],
        "header": "process,node,constraint,p_process_node_constraint_flow_coefficient",
        "filename": "input/p_process_node_constraint_flow_coefficient.csv",
        "filter_in_type": ["1d_map"],
    },
    {
        "cl_pars": [("unit", "constraint_capacity_coefficient"),
                    ("connection", "constraint_capacity_coefficient")],
        "header": "process,constraint,p_process_constraint_capacity_coefficient",
        "filename": "input/p_process_constraint_capacity_coefficient.csv",
        "filter_in_type": ["1d_map"],
    },
    {
        "cl_pars": [("unit", "delay"), ("connection", "delay")],
        "header": "process,delay_duration,p_process_delay_weighted",
        "filename": "input/p_process_delay_weighted.csv",
        "filter_in_type": ["1d_map"],
    },
    {
        "cl_pars": [("unit", "delay"), ("connection", "delay")],
        "header": "process,delay_duration",
        "filename": "input/process_delay_single.csv",
        "filter_in_type": ["str", "float"],
    },
    # --- p_process (scalars) ---
    {
        "cl_pars": [("unit", "availability"),
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
                    ("connection", "virtual_unitsize")],
        "header": "process,processParam,p_process",
        "filename": "input/p_process.csv",
        "filter_in_type": ["float", "str", "bool"],
        "param_print": True,
    },
    # --- p_node (scalars) ---
    {
        "cl_pars": [("node", "annual_flow"),
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
                    ("node", "virtual_unitsize")],
        "header": "node,nodeParam,p_node",
        "filename": "input/p_node.csv",
        "filter_in_type": ["float", "str", "bool"],
        "param_print": True,
    },
    # --- group__process ---
    {
        "cl_pars": [("group__unit", "groupParam"), ("group__connection", "groupParam")],
        "header": "group,process,groupParam,p_group_process_s",
        "filename": "input/p_group__process.csv",
        "param_print": True,
    },
    # --- p_group (scalars) ---
    {
        "cl_pars": [("group", "groupParam"),
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
                    ("group", "penalty_non_synchronous")],
        "header": "group,groupParam,p_group",
        "filename": "input/p_group.csv",
        "filter_in_type": ["float", "str", "bool"],
        "param_print": True,
    },
    # --- pd_process (period maps) ---
    {
        "cl_pars": [("unit", "invest_forced"),
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
                    ("connection", "cumulative_min_capacity")],
        "header": "process,processParam,period,pd_process",
        "filename": "input/pd_process.csv",
        "filter_in_type": ["1d_map"],
        "filter_out_index": "time",
        "param_print": True,
    },
    # --- model discount ---
    {
        "cl_pars": [("model", "discount_rate")],
        "header": "model,p_discount_rate",
        "filename": "input/p_discount_rate.csv",
    },
    {
        "cl_pars": [("model", "discount_offset_operations")],
        "header": "model,p_discount_offset_operations",
        "filename": "input/p_discount_offset_operations.csv",
    },
    {
        "cl_pars": [("model", "discount_offset_investment")],
        "header": "model,p_discount_offset_investment",
        "filename": "input/p_discount_offset_investment.csv",
    },
    # --- pdt_group (time maps) ---
    {
        "cl_pars": [("group", "co2_max_period"),
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
                    ("group", "penalty_non_synchronous")],
        "header": "group,groupParam,time,pt_group",
        "filename": "input/pdt_group.csv",
        "filter_in_type": ["1d_map"],
        "param_print": True,
    },
    # --- pbt_process (branch maps) ---
    {
        "cl_pars": [("unit", "efficiency"),
                    ("unit", "efficiency_at_min_load"),
                    ("unit", "min_load"),
                    ("unit", "availability"),
                    ("connection", "efficiency"),
                    ("connection", "efficiency_at_min_load"),
                    ("connection", "min_load"),
                    ("connection", "other_operational_cost"),
                    ("connection", "availability")],
        "header": "process,processParam,branch,time_start,time,pbt_process",
        "filename": "input/pbt_process.csv",
        "filter_in_type": ["3d_map"],
        "param_print": True,
    },
    # --- optional outputs ---
    {
        "cl_pars": [("model", "exclude_entity_outputs"),
                    ("model", "output_connection__node__node_flow_t"),
                    ("model", "output_connection_flow_separate"),
                    ("model", "output_horizon"),
                    ("model", "output_ramp_envelope"),
                    ("model", "output_unit__node_flow_t"),
                    ("model", "output_unit__node_ramp_t")],
        "header": "output,value",
        "filename": "input/optional_outputs.csv",
        "param_print": True,
        "no_entity": True,
    },
    {
        "cl_pars": [("group", "output_results")],
        "header": "groupOutput",
        "filename": "input/groupOutput.csv",
        "filter_in_value": "yes",
        "no_value": True,
    },
    {
        "cl_pars": [("group", "has_non_synchronous")],
        "header": "groupNonSync",
        "filename": "input/groupNonSync.csv",
        "filter_in_value": "yes",
        "no_value": True,
    },
]


_DEFAULT_VALUES_SPECS: list[dict] = [
    {
        "cl_pars": [("node", "penalty_up"), ("node", "penalty_down")],
        "header": "class,paramName,default_value",
        "filename": "input/default_values.csv",
        "filter_in_type": ["float", "str", "bool"],
    },
    {
        "cl_pars": [("model", "version")],
        "header": "version",
        "filename": "input/db_version.csv",
        "filter_in_type": ["float", "str", "bool"],
        "only_value": True,
    },
]


# ---------------------------------------------------------------------------
# Methods mapping — must stay in sync with flextool_base.dat ``set methods``
# ---------------------------------------------------------------------------

# (ct_method, startup_method, fork_method) -> method
METHODS_MAPPING: dict[tuple[str, str, str], str] = {
    ("constant_efficiency", "no_startup", "fork_no"): "method_1way_1var_off",
    ("constant_efficiency", "no_startup", "fork_yes"): "method_1way_nvar_off",
    ("constant_efficiency", "linear", "fork_no"): "method_1way_1var_LP",
    ("constant_efficiency", "linear", "fork_yes"): "method_1way_nvar_LP",
    ("constant_efficiency", "binary", "fork_no"): "method_1way_1var_MIP",
    ("constant_efficiency", "binary", "fork_yes"): "method_1way_nvar_MIP",
    ("no_losses_no_variable_cost", "no_startup", "fork_no"): "method_2way_1var_off",
    ("no_losses_no_variable_cost", "no_startup", "fork_yes"): "method_2way_nvar_off",
    ("variable_cost_only", "no_startup", "fork_no"): "method_2way_2var_off",
    ("variable_cost_only", "no_startup", "fork_yes"): "method_2way_nvar_off",
    ("regular", "no_startup", "fork_no"): "method_2way_2var_exclude",
    ("regular", "no_startup", "fork_yes"): "not_applicable",
    ("exact", "no_startup", "fork_no"): "method_2way_2var_MIP_exclude",
    ("exact", "no_startup", "fork_yes"): "not_applicable",
    ("min_load_efficiency", "no_startup", "fork_no"): "not_applicable",
    ("min_load_efficiency", "no_startup", "fork_yes"): "not_applicable",
    ("min_load_efficiency", "linear", "fork_no"): "method_1way_1var_LP",
    ("min_load_efficiency", "linear", "fork_yes"): "method_1way_nvar_LP",
    ("min_load_efficiency", "binary", "fork_no"): "method_1way_1var_MIP",
    ("min_load_efficiency", "binary", "fork_yes"): "method_1way_nvar_MIP",
    ("none", "no_startup", "fork_no"): "method_1way_1var_off",
    ("none", "no_startup", "fork_yes"): "method_1way_nvar_off",
    ("none", "linear", "fork_no"): "method_1way_1var_LP",
    ("none", "linear", "fork_yes"): "method_1way_nvar_LP",
    ("none", "binary", "fork_no"): "method_1way_1var_MIP",
    ("none", "binary", "fork_yes"): "method_1way_nvar_MIP",
}


def _write_process_method(db, wf: Path, logger: logging.Logger) -> None:
    """Resolve (ct_method, startup_method, fork_method) -> method for each
    process and write ``input/process_method.csv``.

    This replaces the GMPL set computation that was formerly in flextool.mod
    (process__fork_method_yes/no, process_ct_startup_fork_method, process_method).
    """

    # --- Collect ct_method per process ---
    ct_method_map: dict[str, str] = {}
    for cl, par in [("unit", "conversion_method"), ("connection", "transfer_method")]:
        for pv in db.find_parameter_values(entity_class_name=cl, parameter_definition_name=par):
            if pv["type"] is None:
                continue
            process_name = pv["entity_byname"][0]
            ct_method_map[process_name] = str(pv["parsed_value"])

    # --- Collect startup_method per process ---
    startup_method_map: dict[str, str] = {}
    for cl in ["unit", "connection"]:
        for pv in db.find_parameter_values(entity_class_name=cl, parameter_definition_name="startup_method"):
            if pv["type"] is None:
                continue
            process_name = pv["entity_byname"][0]
            startup_method_map[process_name] = str(pv["parsed_value"])

    # --- Collect sources and sinks per process ---
    source_counts: dict[str, int] = {}
    for ent_class, dim_idx in [("unit__inputNode", [0, 1]), ("connection__node__node", [0, 1])]:
        for entity in db.find_entities(entity_class_name=ent_class):
            process_name = entity["entity_byname"][dim_idx[0]]
            source_counts[process_name] = source_counts.get(process_name, 0) + 1

    sink_counts: dict[str, int] = {}
    for ent_class, dim_idx in [("unit__outputNode", [0, 1]), ("connection__node__node", [0, 2])]:
        for entity in db.find_entities(entity_class_name=ent_class):
            process_name = entity["entity_byname"][dim_idx[0]]
            sink_counts[process_name] = sink_counts.get(process_name, 0) + 1

    # --- Collect delayed processes ---
    delayed_processes: set[str] = set()
    for cl in ["unit", "connection"]:
        for pv in db.find_parameter_values(entity_class_name=cl, parameter_definition_name="delay"):
            if pv["type"] is None:
                continue
            delayed_processes.add(pv["entity_byname"][0])

    # --- Collect all processes and which class they belong to ---
    all_processes: dict[str, str] = {}  # process_name -> "unit" or "connection"
    for entity in db.find_entities(entity_class_name="unit"):
        all_processes[entity["entity_byname"][0]] = "unit"
    for entity in db.find_entities(entity_class_name="connection"):
        all_processes[entity["entity_byname"][0]] = "connection"

    # --- Resolve method for each process ---
    rows: list[tuple[str, str]] = []
    for process_name, process_class in all_processes.items():
        # ct_method defaults must match flextool_base.dat:
        #   ct_method_constant (units) = "constant_efficiency"
        #   ct_method_regular (connections) = "regular"
        if process_name in ct_method_map:
            ct = ct_method_map[process_name]
        elif process_class == "connection":
            ct = "regular"
        else:
            ct = "constant_efficiency"

        # startup_method: default "no_startup"
        startup = startup_method_map.get(process_name, "no_startup")

        # fork_method: fork_yes if >1 source OR >1 sink OR delayed
        n_sources = source_counts.get(process_name, 0)
        n_sinks = sink_counts.get(process_name, 0)
        is_delayed = process_name in delayed_processes
        fork = "fork_yes" if (n_sources > 1 or n_sinks > 1 or is_delayed) else "fork_no"

        key = (ct, startup, fork)
        method = METHODS_MAPPING.get(key)
        if method is None:
            logger.warning(
                "process_method: no mapping for process '%s' with "
                "(ct_method=%s, startup_method=%s, fork_method=%s) — skipping",
                process_name, ct, startup, fork,
            )
            continue
        if method == "not_applicable":
            logger.warning(
                "process_method: method resolves to 'not_applicable' for "
                "process '%s' (ct_method=%s, startup_method=%s, fork_method=%s)",
                process_name, ct, startup, fork,
            )
        rows.append((process_name, method))

    # --- Write CSV ---
    filepath = wf / "input" / "process_method.csv"
    with open(filepath, "w") as f:
        f.write("process,method\n")
        for process_name, method in rows:
            f.write(f"{process_name},{method}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def write_input(input_db_url: str, scenario_name: str | None, logger: logging.Logger, work_folder: Path | None = None) -> None:
    wf = work_folder if work_folder is not None else Path.cwd()
    if scenario_name:
        scen_config = api.filters.scenario_filter.scenario_filter_config(scenario_name)
    with DatabaseMapping(input_db_url) as db:
        #it is faster to fetch all now than fetching multiple times
        db.fetch_all("entity")
        db.fetch_all("parameter_value")
        if scenario_name:
            api.filters.scenario_filter.scenario_filter_from_dict(db, scen_config)
        os.makedirs(wf / "input", exist_ok=True)

        for spec in _DEFAULT_VALUES_SPECS:
            prefixed_spec = dict(spec)
            prefixed_spec["filename"] = str(wf / spec["filename"])
            write_default_values(db, **prefixed_spec)

        for spec in _ENTITY_SPECS:
            write_entity(db, spec.classes, spec.header, str(wf / spec.filename),
                         entity_dimens=spec.entity_dimens)

        for spec in _PARAMETER_SPECS:
            prefixed_spec = dict(spec)
            prefixed_spec["filename"] = str(wf / spec["filename"])
            write_parameter(db, **prefixed_spec)

        _write_process_method(db, wf, logger)


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
        # Work on a local copy to avoid mutating the caller's list
        filter_in_type = list(filter_in_type)
        map_found = False
        for type_filter in filter_in_type:
            if type_filter in ["1d_map", "2d_map", "3d_map", "4d_map", "5d_map"]:
                if map_found:
                    message = "Trying to have two different dimensionalities in the same parameter to be written out"
                    logging.error(message)
                    raise FlexToolConfigError(message)
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
            # Skip parameters with None type (value cleared in an alternative)
            if param["type"] is None:
                continue
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
                message = (f"Input data found in a parameter not of supported type."
                           f"\nEntity: {','.join(entity_byname)}"
                           f"\nParameter: {param['parameter_definition_name']}"
                           f"\nSupported types: {filter_in_type}"
                           f"\nParameter type: {param['type']}")
                logging.error(message)
                raise FlexToolConfigError(message)



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
                message = ("Default_value found in a parameter definition not of supported default type"
                           "\nParameter: " + param["parameter_definition_name"])
                logging.error(message)
                raise FlexToolConfigError(message)
