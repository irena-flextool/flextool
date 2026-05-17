"""
input_writer.py — Write input/ CSV files from the database.

Entry point: write_input(input_db_url, scenario_name, logger, *, provider, ...)
All write_entity / write_parameter / write_default_values calls are internal helpers.
"""
from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import NamedTuple, TYPE_CHECKING

import spinedb_api as api
from spinedb_api import DatabaseMapping

if TYPE_CHECKING:
    from flextool.engine_polars._flex_data_provider import FlexDataProvider

from flextool.flextoolrunner.runner_state import FlexToolConfigError
from flextool.flextoolrunner.precision import format_scalar_for_csv


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
    {
        "cl_pars": [("commodity", "price_method")],
        "header": "commodity,p_commodity_price_method",
        "filename": "input/p_commodity_price_method.csv",
    },
    {
        "cl_pars": [("commodity", "unitsize")],
        "header": "commodity,p_commodity_unitsize",
        "filename": "input/p_commodity_unitsize.csv",
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
        "cl_pars": [("group", "output_nodeGroup_dispatch")],
        "header": "nodeGroupDispatch",
        "filename": "input/nodeGroupDispatch.csv",
        "filter_in_value": "yes",
        "no_value": True,
    },
    {
        "cl_pars": [("group", "flow_aggregator")],
        "header": "flowAggregator",
        "filename": "input/flowAggregator.csv",
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
        "cl_pars": [("node", "constraint_invested_capacity_coefficient")],
        "header": "node,constraint,p_node_constraint_invested_capacity_coefficient",
        "filename": "input/p_node_constraint_invested_capacity_coefficient.csv",
    },
    {
        "cl_pars": [("node", "constraint_cumulative_pre_built_capacity_coefficient")],
        "header": "node,constraint,p_node_constraint_prebuilt_capacity_coefficient",
        "filename": "input/p_node_constraint_cumulative_pre_built_capacity_coefficient.csv",
    },
    {
        "cl_pars": [("node", "constraint_state_coefficient")],
        "header": "node,constraint,p_node_constraint_state_coefficient",
        "filename": "input/p_node_constraint_state_coefficient.csv",
    },
    {
        "cl_pars": [("node", "node_type")],
        "header": "node,p_node_type",
        "filename": "input/p_node_type.csv",
    },
    {
        "cl_pars": [("node", "inflow_method")],
        "header": "node,inflow_method",
        "filename": "input/node__inflow_method.csv",
    },
    {
        "cl_pars": [("node__profile", "profile_method")],
        "header": "node,profile,profile_method",
        "filename": "input/node__profile__profile_method.csv",
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
                    ("node", "discount_rate"),
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
        "cl_pars": [("unit__outputNode", "flow_coefficient")],
        "header": "process,sink,p_process_sink_flow_coefficient",
        "filename": "input/p_process_sink_flow_coefficient.csv",
        "filter_in_type": ["float", "str", "bool"],
    },
    {
        "cl_pars": [("unit__inputNode", "flow_coefficient")],
        "header": "process,source,p_process_source_flow_coefficient",
        "filename": "input/p_process_source_flow_coefficient.csv",
        "filter_in_type": ["float", "str", "bool"],
    },
    {
        "cl_pars": [("unit__outputNode", "max_capacity_coefficient")],
        "header": "process,sink,p_process_sink_max_capacity_coefficient",
        "filename": "input/p_process_sink_max_capacity_coefficient.csv",
        "filter_in_type": ["float", "str", "bool"],
    },
    {
        "cl_pars": [("unit__outputNode", "min_capacity_coefficient")],
        "header": "process,sink,p_process_sink_min_capacity_coefficient",
        "filename": "input/p_process_sink_min_capacity_coefficient.csv",
        "filter_in_type": ["float", "str", "bool"],
    },
    {
        "cl_pars": [("unit__inputNode", "max_capacity_coefficient")],
        "header": "process,source,p_process_source_max_capacity_coefficient",
        "filename": "input/p_process_source_max_capacity_coefficient.csv",
        "filter_in_type": ["float", "str", "bool"],
    },
    {
        "cl_pars": [("unit__inputNode", "min_capacity_coefficient")],
        "header": "process,source,p_process_source_min_capacity_coefficient",
        "filename": "input/p_process_source_min_capacity_coefficient.csv",
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
        "cl_pars": [("unit", "constraint_invested_capacity_coefficient"),
                    ("connection", "constraint_invested_capacity_coefficient")],
        "header": "process,constraint,p_process_constraint_invested_capacity_coefficient",
        "filename": "input/p_process_constraint_invested_capacity_coefficient.csv",
        "filter_in_type": ["1d_map"],
    },
    {
        "cl_pars": [("unit", "constraint_cumulative_pre_built_capacity_coefficient"),
                    ("connection", "constraint_cumulative_pre_built_capacity_coefficient")],
        "header": "process,constraint,p_process_constraint_prebuilt_capacity_coefficient",
        "filename": "input/p_process_constraint_cumulative_pre_built_capacity_coefficient.csv",
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
                    ("unit", "discount_rate"),
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
                    ("connection", "discount_rate"),
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
                    ("node", "discount_rate"),
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
                    ("group", "penalty_non_synchronous"),
                    # Agent 1.9: v51 group-level new_stepduration is
                    # numeric so it flows through the normal p_group
                    # numeric channel.  The companion
                    # ``decomposition_method`` is a *string* enum and
                    # would break ``p_group``'s numeric type — it lives
                    # in its own ``input/p_group_decomposition.csv``
                    # written below (consumed only by ``blocks.py``).
                    ("group", "new_stepduration")],
        "header": "group,groupParam,p_group",
        "filename": "input/p_group.csv",
        "filter_in_type": ["float", "str", "bool"],
        "param_print": True,
    },
    # Agent 1.9: separate file for the (string) decomposition_method
    # so the numeric p_group channel stays clean.  ``blocks.py``
    # falls back to this file when the row isn't found in
    # ``p_group.csv``.
    {
        "cl_pars": [("group", "decomposition_method")],
        "header": "group,groupParam,p_group",
        "filename": "input/p_group_decomposition.csv",
        "filter_in_type": ["str"],
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
                    ("unit", "discount_rate"),
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
                    ("connection", "discount_rate"),
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
    # --- model inflation ---
    {
        "cl_pars": [("model", "inflation_rate")],
        "header": "model,p_inflation_rate",
        "filename": "input/p_inflation_rate.csv",
    },
    {
        "cl_pars": [("model", "inflation_offset_operations")],
        "header": "model,p_inflation_offset_operations",
        "filename": "input/p_inflation_offset_operations.csv",
    },
    {
        "cl_pars": [("model", "inflation_offset_investment")],
        "header": "model,p_inflation_offset_investment",
        "filename": "input/p_inflation_offset_investment.csv",
    },
    {
        "cl_pars": [("model", "max_flow_for_unconstrained_variables")],
        "header": "model,p_max_flow_for_unconstrained_variables",
        "filename": "input/p_max_flow_for_unconstrained_variables.csv",
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
        "cl_pars": [("group", "output_nodeGroup_indicators")],
        "header": "nodeGroupIndicators",
        "filename": "input/nodeGroupIndicators.csv",
        "filter_in_value": "yes",
        "no_value": True,
    },
    {
        "cl_pars": [("group", "output_flowGroup_indicators")],
        "header": "flowGroupIndicators",
        "filename": "input/flowGroupIndicators.csv",
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
    ("unidirectional", "no_startup", "fork_no"): "method_1way_1var_off",
    ("unidirectional", "no_startup", "fork_yes"): "method_1way_nvar_off",
    ("unidirectional", "linear", "fork_no"): "method_1way_1var_LP",
    ("unidirectional", "linear", "fork_yes"): "method_1way_nvar_LP",
    ("unidirectional", "binary", "fork_no"): "method_1way_1var_MIP",
    ("unidirectional", "binary", "fork_yes"): "method_1way_nvar_MIP",
}


def _write_process_method(
    db,
    wf: Path,
    logger: logging.Logger,
    ct_method_overrides: dict[str, str] | None = None,
) -> None:
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

    # Apply group-level ct_method overrides (from DC PF and other group transfer_methods)
    if ct_method_overrides:
        for process_name, override_method in ct_method_overrides.items():
            ct_method_map[process_name] = override_method

    # --- Collect startup_method per process ---
    startup_method_map: dict[str, str] = {}
    for cl in ["unit", "connection"]:
        for pv in db.find_parameter_values(entity_class_name=cl, parameter_definition_name="startup_method"):
            if pv["type"] is None:
                continue
            process_name = pv["entity_byname"][0]
            startup_method_map[process_name] = str(pv["parsed_value"])

    # --- Collect minimum_time_method per process ---
    minimum_time_method_map: dict[str, str] = {}
    for pv in db.find_parameter_values(
        entity_class_name="unit", parameter_definition_name="minimum_time_method"
    ):
        if pv["type"] is None:
            continue
        process_name = pv["entity_byname"][0]
        minimum_time_method_map[process_name] = str(pv["parsed_value"])

    # --- Override startup_method if minimum_time_method requires online variables ---
    for process_name, mtm in minimum_time_method_map.items():
        if mtm in ("min_uptime", "min_downtime", "both"):
            current_startup = startup_method_map.get(process_name, "no_startup")
            if current_startup == "no_startup":
                startup_method_map[process_name] = "linear"
                logger.info(
                    "Process '%s': startup_method overridden to 'linear' "
                    "because minimum_time_method='%s' requires online variables",
                    process_name, mtm,
                )

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

    # --- Write process_min_uptime.csv and process_min_downtime.csv ---
    # Collect min_uptime and min_downtime values to determine which processes have nonzero values
    min_uptime_values: dict[str, float] = {}
    min_downtime_values: dict[str, float] = {}
    for pv in db.find_parameter_values(
        entity_class_name="unit", parameter_definition_name="min_uptime"
    ):
        if pv["type"] is not None and pv["parsed_value"]:
            min_uptime_values[pv["entity_byname"][0]] = float(pv["parsed_value"])
    for pv in db.find_parameter_values(
        entity_class_name="unit", parameter_definition_name="min_downtime"
    ):
        if pv["type"] is not None and pv["parsed_value"]:
            min_downtime_values[pv["entity_byname"][0]] = float(pv["parsed_value"])

    # Process is in process_min_uptime if minimum_time_method is min_uptime or both, AND min_uptime > 0
    process_min_uptime = []
    process_min_downtime = []
    for process_name, mtm in minimum_time_method_map.items():
        if mtm in ("min_uptime", "both") and min_uptime_values.get(process_name, 0) > 0:
            process_min_uptime.append(process_name)
        if mtm in ("min_downtime", "both") and min_downtime_values.get(process_name, 0) > 0:
            process_min_downtime.append(process_name)

    with open(wf / "input/process_min_uptime.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["process_min_uptime"])
        for p in sorted(process_min_uptime):
            writer.writerow([p])

    with open(wf / "input/process_min_downtime.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["process_min_downtime"])
        for p in sorted(process_min_downtime):
            writer.writerow([p])


def _tier_sort_key(t: str) -> tuple[int, str]:
    """Stable sort by integer tier when possible, else by string."""
    try:
        return (0, f"{int(t):020d}")
    except ValueError:
        return (1, t)


def _quantity_sentinel(quantity: str) -> str:
    """GMPL's CSV reader rejects 'inf'/'Infinity'.  Convert user-facing
    infinite quantities into the 1e30 sentinel the mod interprets as the
    unbounded tail tier (see ladder_tier_cap_infinite_cum / _ann)."""
    try:
        q_float = float(quantity)
    except ValueError:
        q_float = float("inf")
    if q_float == float("inf") or q_float >= 1e30:
        return "1e30"
    return quantity


def _get_commodity_price_methods(db) -> dict[str, str]:
    """Return ``{commodity: price_method}`` for every commodity whose
    ``price_method`` is set.  Commodities without the param default to
    ``'price'`` in the mod (and do not appear here).
    """
    out: dict[str, str] = {}
    for pv in db.find_parameter_values(
        entity_class_name="commodity",
        parameter_definition_name="price_method",
    ):
        if pv["type"] is None:
            continue
        out[pv["entity_byname"][0]] = str(pv["parsed_value"])
    return out


def _collect_periods(db, wf: Path) -> list[str]:
    """Return the model's period list (for 1d-map → per-period expansion
    of ``price_ladder_annual``).

    Reads the periods from ``input/periods_available.csv`` which
    ``write_parameter`` already emitted for the ``model.periods_available``
    parameter.  Falls back to scanning ``model.periods_available`` values
    from the DB if the CSV is empty (e.g. when periods come exclusively
    from ``period_timeset``).  Periods from ``period_timeset`` are not
    available at writer-run time, so when the CSV is empty and no
    ``periods_available`` is set we return an empty list — the annual
    writer then emits no rows for 1d ladders (and the preflight already
    caught the "price_ladder_annual set but empty" case).
    """
    periods: list[str] = []
    seen: set[str] = set()
    csv_path = wf / "input" / "periods_available.csv"
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if not row:
                    continue
                # File layout: "model,period_from_model"; take column 1.
                p = row[-1].strip()
                if p and p not in seen:
                    periods.append(p)
                    seen.add(p)
    if periods:
        return periods
    # Fallback 1: pull periods_available direct from the DB as a map /
    # array.  Values are period names.
    for pv in db.find_parameter_values(
        entity_class_name="model",
        parameter_definition_name="periods_available",
    ):
        if pv["type"] is None:
            continue
        val = pv["parsed_value"]
        try:
            flat = api.convert_map_to_table(val)
        except Exception:
            flat = []
        for entry in flat:
            for c in (str(x) for x in entry):
                if c and c not in seen:
                    periods.append(c)
                    seen.add(c)
    if periods:
        return periods
    # Fallback 2: scan solve.period_timeset map indexes.  For typical
    # test setups this is where periods live.
    for pv in db.find_parameter_values(
        entity_class_name="solve",
        parameter_definition_name="period_timeset",
    ):
        if pv["type"] is None:
            continue
        val = pv["parsed_value"]
        try:
            flat = api.convert_map_to_table(val)
        except Exception:
            flat = []
        for entry in flat:
            # Row layout: [period, timeset] for a simple Map.
            if len(entry) >= 1:
                p = str(entry[0])
                if p and p not in seen:
                    periods.append(p)
                    seen.add(p)
    return periods


def _validate_timeline_timestep_duration(db) -> None:
    """Raise FlexToolConfigError if any timeline entity is missing its
    ``timestep_duration`` map.  Without it, ``step_duration`` silently
    falls to 0 throughout the model and every time-weighted quantity
    (balances, costs, ramps) collapses to zero.  There is no sensible
    default, so the value must be present on every timeline.
    """
    timelines = [ent["entity_byname"][0]
                 for ent in db.find_entities(entity_class_name="timeline")]
    if not timelines:
        return
    have_duration: set[str] = set()
    for pv in db.find_parameter_values(
        entity_class_name="timeline",
        parameter_definition_name="timestep_duration",
    ):
        if pv["type"] is None:
            continue
        have_duration.add(pv["entity_byname"][0])
    missing = [t for t in timelines if t not in have_duration]
    if missing:
        raise FlexToolConfigError(
            "timeline 'timestep_duration' is not set for: "
            + ", ".join(sorted(missing))
            + ".  Every timeline needs a Map(timestep -> duration_in_hours); "
              "without it all time-weighted quantities collapse to zero."
        )


def _validate_ladder_methods(db, logger: logging.Logger) -> None:
    """Raise FlexToolConfigError if any commodity declares a ladder
    ``price_method`` but does not have the corresponding ladder parameter
    set.  Runs before the ladder writers so errors name the offending
    commodity and expected parameter.
    """
    methods = _get_commodity_price_methods(db)
    ladder_methods = {"price_ladder_annual", "price_ladder_cumulative"}
    commodities_needing_ladder = {
        c: m for c, m in methods.items() if m in ladder_methods
    }
    if not commodities_needing_ladder:
        return

    # Collect commodities that HAVE each ladder param (non-None, non-empty).
    have_cumulative: set[str] = set()
    have_annual: set[str] = set()
    for pv in db.find_parameter_values(
        entity_class_name="commodity",
        parameter_definition_name="price_ladder_cumulative",
    ):
        if pv["type"] is None:
            continue
        have_cumulative.add(pv["entity_byname"][0])
    for pv in db.find_parameter_values(
        entity_class_name="commodity",
        parameter_definition_name="price_ladder_annual",
    ):
        if pv["type"] is None:
            continue
        have_annual.add(pv["entity_byname"][0])

    for commodity, method in commodities_needing_ladder.items():
        expected_param = method  # parameter name matches method name
        if method == "price_ladder_cumulative" and commodity not in have_cumulative:
            raise FlexToolConfigError(
                f"commodity '{commodity}' has "
                f"price_method='price_ladder_cumulative' but no "
                f"'{expected_param}' value is set.  Add a "
                f"Map(tier -> {{price, quantity}}) on that parameter."
            )
        if method == "price_ladder_annual" and commodity not in have_annual:
            raise FlexToolConfigError(
                f"commodity '{commodity}' has "
                f"price_method='price_ladder_annual' but no "
                f"'{expected_param}' value is set.  Add either a 1d "
                f"Map(tier -> {{price, quantity}}) or a 2d "
                f"Map(tier -> Map(period -> {{price, quantity}}))."
            )


def _iter_flat_ladder_rows(
    value,
    commodity: str,
    logger: logging.Logger,
) -> list[list]:
    """Flatten a Spine map ladder value.  Returns the raw list-of-lists
    from ``convert_map_to_table`` or an empty list on failure.
    """
    try:
        return api.convert_map_to_table(value)
    except Exception as exc:
        logger.warning(
            "Could not flatten ladder for commodity '%s': %s",
            commodity, exc,
        )
        return []


def _write_commodity_ladder_cumulative(
    db, wf: Path, logger: logging.Logger,
) -> None:
    """Emit ``input/commodity_ladder_cumulative.csv`` with columns
    ``commodity, tier, price, quantity`` — one row per (commodity, tier).

    Only the ``commodity.price_ladder_cumulative`` parameter is consulted
    (always a 2d map: ``Map(tier -> {price, quantity})`` — 2d in Spine's
    counting because ``{price, quantity}`` is a second index layer).  The
    ``price_method`` filter happens mod-side via the
    ``commodity_with_ladder_cumulative`` set.
    """
    filepath = wf / "input" / "commodity_ladder_cumulative.csv"
    rows: list[tuple[str, int, str, str]] = []

    for pv in db.find_parameter_values(
        entity_class_name="commodity",
        parameter_definition_name="price_ladder_cumulative",
    ):
        if pv["type"] is None:
            continue
        if pv["type"] != "map":
            logger.warning(
                "commodity.price_ladder_cumulative on '%s' has type %s "
                "(expected nested 1d map); skipping.",
                pv["entity_byname"][0], pv["type"],
            )
            continue
        commodity = pv["entity_byname"][0]
        flat = _iter_flat_ladder_rows(pv["parsed_value"], commodity, logger)
        per_tier: dict[str, dict[str, str]] = {}
        for entry in flat:
            # Expected layout: [tier_idx, facet, value] (length 3).
            if len(entry) < 3:
                continue
            tier_str = str(entry[0])
            facet = str(entry[1])
            val = entry[-1]
            per_tier.setdefault(tier_str, {})[facet] = str(val)

        for tier_str in sorted(per_tier.keys(), key=_tier_sort_key):
            facets = per_tier[tier_str]
            price = facets.get("price", "0")
            quantity = _quantity_sentinel(facets.get("quantity", "inf"))
            try:
                tier_int = int(tier_str)
            except ValueError:
                logger.warning(
                    "commodity.price_ladder_cumulative tier on '%s' is not "
                    "an integer ('%s'); skipping tier.",
                    commodity, tier_str,
                )
                continue
            rows.append((commodity, tier_int, price, quantity))

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["commodity", "tier", "price", "quantity"])
        for row in rows:
            writer.writerow(row)


def _write_commodity_ladder_annual(
    db, wf: Path, logger: logging.Logger,
) -> None:
    """Emit ``input/commodity_ladder_annual.csv`` with columns
    ``commodity, period, tier, price, quantity`` — one row per
    (commodity, period, tier).

    Reads ``commodity.price_ladder_annual``.  Auto-detects the map depth:

    * 2d form (Spine 2d_map): ``Map(tier -> {price, quantity})`` — the
      same (price, quantity) is expanded across every model period.
    * 3d form (Spine 3d_map): ``Map(period -> Map(tier -> {price,
      quantity}))`` — per-period rows are kept as-is.
    """
    filepath = wf / "input" / "commodity_ladder_annual.csv"
    rows: list[tuple[str, int, str, str, str]] = []
    periods_cache: list[str] | None = None

    for pv in db.find_parameter_values(
        entity_class_name="commodity",
        parameter_definition_name="price_ladder_annual",
    ):
        if pv["type"] is None:
            continue
        if pv["type"] != "map":
            logger.warning(
                "commodity.price_ladder_annual on '%s' has type %s "
                "(expected nested map); skipping.",
                pv["entity_byname"][0], pv["type"],
            )
            continue
        commodity = pv["entity_byname"][0]
        flat = _iter_flat_ladder_rows(pv["parsed_value"], commodity, logger)
        if not flat:
            continue

        # Depth detection via the flat table row length.  Spine's Map
        # dimension count matches the flat row length: 2d_map yields
        # length-3 rows, 3d_map yields length-4 rows.
        #   2d_map: [tier, facet, value]                      → len 3
        #   3d_map: [period, tier, facet, value]              → len 4
        max_len = max((len(row) for row in flat), default=0)
        if max_len == 3:
            # 2d_map → expand across all model periods.
            per_tier: dict[str, dict[str, str]] = {}
            for entry in flat:
                if len(entry) < 3:
                    continue
                tier_str = str(entry[0])
                facet = str(entry[1])
                val = entry[-1]
                per_tier.setdefault(tier_str, {})[facet] = str(val)
            if periods_cache is None:
                periods_cache = _collect_periods(db, wf)
            if not periods_cache:
                logger.warning(
                    "commodity.price_ladder_annual on '%s' is 2d_map but "
                    "no model periods were available for expansion; "
                    "skipping.", commodity,
                )
                continue
            for period in periods_cache:
                for tier_str in sorted(per_tier.keys(), key=_tier_sort_key):
                    facets = per_tier[tier_str]
                    price = facets.get("price", "0")
                    quantity = _quantity_sentinel(facets.get("quantity", "inf"))
                    try:
                        tier_int = int(tier_str)
                    except ValueError:
                        logger.warning(
                            "commodity.price_ladder_annual tier on '%s' is "
                            "not an integer ('%s'); skipping tier.",
                            commodity, tier_str,
                        )
                        continue
                    rows.append(
                        (commodity, period, tier_int, price, quantity)
                    )
        elif max_len >= 4:
            # 3d_map → per-period.  Flat row layout [period, tier,
            # facet, value] — Spine nests Map(period -> Map(tier ->
            # {price, quantity})).
            per_period_tier: dict[tuple[str, str], dict[str, str]] = {}
            for entry in flat:
                if len(entry) < 4:
                    continue
                period = str(entry[0])
                tier_str = str(entry[1])
                facet = str(entry[2])
                val = entry[-1]
                per_period_tier.setdefault(
                    (period, tier_str), {}
                )[facet] = str(val)

            def _sort_key(k: tuple[str, str]) -> tuple:
                return (k[0], _tier_sort_key(k[1]))

            for (period, tier_str) in sorted(
                per_period_tier.keys(), key=_sort_key,
            ):
                facets = per_period_tier[(period, tier_str)]
                price = facets.get("price", "0")
                quantity = _quantity_sentinel(facets.get("quantity", "inf"))
                try:
                    tier_int = int(tier_str)
                except ValueError:
                    logger.warning(
                        "commodity.price_ladder_annual tier on '%s' is not "
                        "an integer ('%s'); skipping tier.",
                        commodity, tier_str,
                    )
                    continue
                rows.append(
                    (commodity, period, tier_int, price, quantity)
                )
        else:
            logger.warning(
                "commodity.price_ladder_annual on '%s' has unexpected "
                "flattened shape (max row length %d); skipping.",
                commodity, max_len,
            )
            continue

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["commodity", "period", "tier", "price", "quantity"]
        )
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def write_input(
    input_db_url: str,
    scenario_name: str | None,
    logger: logging.Logger,
    work_folder: Path | None = None,
    precision_digits: int = 0,
    *,
    provider: "FlexDataProvider",
) -> None:
    """Populate ``work_folder/input/`` + ``work_folder/solve_data/`` CSVs
    from *input_db_url*.

    Step 2.5 thread-through
    -----------------------

    *provider* is the cascade-input
    :class:`flextool.engine_polars._flex_data_provider.FlexDataProvider`.
    Phases 2-4 of Step 2.5 progressively replace the disk-emitting spec
    loops (``_DEFAULT_VALUES_SPECS`` / ``_ENTITY_SPECS`` /
    ``_PARAMETER_SPECS``) with
    :class:`flextool.spinedb_backend.SpineDBBackend` materialisers whose
    frames land in *provider* — without any CSV touching ``input/``.

    Until those phases complete, this Phase 1 wiring simply accepts the
    parameter (it is required, not optional — there is no disk
    fallback).  Callers that don't have a Provider must construct an
    ephemeral one; ``None`` is rejected.
    """
    if provider is None:  # pragma: no cover — explicit guard
        raise TypeError(
            "write_input requires a FlexDataProvider; pass an ephemeral "
            "provider for one-shot CSV-only callers (Step 2.5 contract).",
        )
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

        # Step 2.5 Phase 2 — _DEFAULT_VALUES_SPECS materialised by the
        # SpineDBBackend; frames land in the cascade-input Provider
        # under their canonical input/<stem> key, replacing the legacy
        # ``input/<stem>.csv`` disk write (deleted at this site).
        from flextool.spinedb_backend import SpineDBBackend
        _backend_default = SpineDBBackend.__new__(SpineDBBackend)
        _backend_default._db = db                              # type: ignore[attr-defined]
        _backend_default._api = api                            # type: ignore[attr-defined]
        _backend_default._precision_digits = precision_digits  # type: ignore[attr-defined]
        for spec in _DEFAULT_VALUES_SPECS:
            _frame = _backend_default.parameter_defaults(
                cl_pars=spec["cl_pars"],
                header=spec["header"],
                filter_in_type=spec.get("filter_in_type"),
                only_value=spec.get("only_value", False),
            )
            # Canonical Provider key — matches ``_provider_key`` in
            # :mod:`flextool.engine_polars._writer_provider_io`:
            # ``"<parent_dir_name>/<stem>"``.  E.g. ``input/default_values``.
            _spec_path = Path(spec["filename"])
            _key = (
                f"{_spec_path.parent.name}/{_spec_path.stem}"
                if _spec_path.parent.name else _spec_path.stem
            )
            provider.put(_key, _frame)

        # Step 2.5 Phase 3 — _ENTITY_SPECS migrated to SpineDBBackend.entities.
        # The 16 entity-class frames now flow through the cascade-input
        # Provider under the canonical input/<stem> key.  No disk writes
        # at this site; write_entity has been deleted.
        for spec in _ENTITY_SPECS:
            _entity_frame = _backend_default.entities(
                classes=spec.classes,
                header=spec.header,
                entity_dimens=spec.entity_dimens,
            )
            _spec_path = Path(spec.filename)
            _key = (
                f"{_spec_path.parent.name}/{_spec_path.stem}"
                if _spec_path.parent.name else _spec_path.stem
            )
            provider.put(_key, _entity_frame)

        _validate_timeline_timestep_duration(db)

        # Step 2.5-E Phase E — _PARAMETER_SPECS materialised by the
        # SpineDBBackend.  Each spec's frame lands in the cascade-input
        # Provider under the canonical ``input/<stem>`` key; the legacy
        # ``write_parameter`` disk write is deleted at this site.
        for spec in _PARAMETER_SPECS:
            _spec_kwargs = {
                k: v for k, v in spec.items() if k != "filename"
            }
            _frame = _backend_default.parameter_values(**_spec_kwargs)
            _spec_path = Path(spec["filename"])
            _key = (
                f"{_spec_path.parent.name}/{_spec_path.stem}"
                if _spec_path.parent.name else _spec_path.stem
            )
            provider.put(_key, _frame)

        # Step 2.5-F Phase B — DC power flow derivation moved to
        # ``flextool.input_derivation._dc_power_flow``.  The derivation
        # reads via the SpineDBBackend, populates the cascade-input
        # Provider with the four DC PF frames, and returns the
        # ``ct_method_overrides`` dict for the immediate-caller
        # convenience (also stored on the Provider under
        # ``derived/ct_method_overrides`` for cross-derivation use).
        from flextool.input_derivation._dc_power_flow import (
            derive_dc_power_flow,
        )
        ct_method_overrides = derive_dc_power_flow(
            _backend_default, provider, logger,
        )
        _write_process_method(db, wf, logger, ct_method_overrides=ct_method_overrides)
        _validate_ladder_methods(db, logger)
        _write_commodity_ladder_cumulative(db, wf, logger)
        _write_commodity_ladder_annual(db, wf, logger)
        # Migrated from flextool.mod:468-470 — commodity_with_ladder*
        # filtered subsets used to be derived inside MathProg via setof
        # filters on p_commodity_price_method. Computed in Python for
        # cheaper matrix generation; loaded back via table data IN.
        from flextool.flextoolrunner.preprocessing import (
            commodity_ladder_sets,
            period_param_sets,
            invest_method_sets,
            co2_method_sets,
            simple_projections,
            node_type_sets,
            method_with_fallback_sets,
            nonsync_sets,
        )
        input_dir = wf / "input"
        solve_data_dir = wf / "solve_data"
        commodity_ladder_sets.write_commodity_ladder_sets(
            _get_commodity_price_methods(db), solve_data_dir,
        )
        # L0 Batch 1: simple projections / filters of already-written
        # input/*.csv tables. Each function reads a CSV that the spec-
        # driven write_parameter loop above produced and writes a
        # solve_data/*.csv that flextool.mod loads via table data IN.
        period_param_sets.write_period_param_sets(input_dir, solve_data_dir)
        invest_method_sets.write_invest_method_sets(input_dir, solve_data_dir)
        co2_method_sets.write_co2_method_sets(input_dir, solve_data_dir)
        simple_projections.write_optional_yes(input_dir, solve_data_dir)
        simple_projections.write_reserve_upDown_group(input_dir, solve_data_dir)
        simple_projections.write_group_loss_share(input_dir, solve_data_dir)
        # L0 Batch 2: harder operation types — defaults flow, joining
        # with global-empty fallback, quadratic-style joining.
        node_type_sets.write_node_type_sets(input_dir, solve_data_dir)
        method_with_fallback_sets.write_entity_lifetime_method(input_dir, solve_data_dir)
        method_with_fallback_sets.write_process_ct_method(input_dir, solve_data_dir)
        method_with_fallback_sets.write_process_startup_method(input_dir, solve_data_dir)
        method_with_fallback_sets.write_node_inflow_method(input_dir, solve_data_dir)
        method_with_fallback_sets.write_node_storage_binding_method(input_dir, solve_data_dir)
        nonsync_sets.write_process_group_inside_group_nonsync(input_dir, solve_data_dir)
        nonsync_sets.write_process__sink_nonSync(input_dir, solve_data_dir)
        # L0 Batch 3: union sets + first calculated-param migration.
        from flextool.flextoolrunner.preprocessing import (
            union_sets, entity_total_caps,
        )
        union_sets.write_group_entity(input_dir, solve_data_dir)
        union_sets.write_process_delayed__duration(input_dir, solve_data_dir)
        entity_total_caps.write_entity_total_caps(input_dir, solve_data_dir)
        # L0 Batch 4: bulk simple/method-driven sets — process_*_to_*
        # family, profile-method joins, reserve-method partitions,
        # structural filters, and the remaining trivial setof projections.
        # All upstream sources live in input/ so these run at write_input
        # time alongside the earlier batches.
        from flextool.flextoolrunner.preprocessing import (
            process_method_sets,
            reserve_method_partitions,
            structural_filters,
        )
        process_method_sets.write_process_method_projections(input_dir, solve_data_dir)
        process_method_sets.write_process_VRE(input_dir, solve_data_dir)
        process_method_sets.write_process_arc_method_joins(input_dir, solve_data_dir)
        process_method_sets.write_process_profile_method_joins(input_dir, solve_data_dir)
        reserve_method_partitions.write_reserve_partitions(input_dir, solve_data_dir)
        structural_filters.write_connection_param(input_dir, solve_data_dir)
        structural_filters.write_nodegroup_dispatch_node(input_dir, solve_data_dir)
        structural_filters.write_commodity_node_co2(input_dir, solve_data_dir)
        structural_filters.write_process__commodity__node(input_dir, solve_data_dir)
        structural_filters.write_process_coeff_zero_sets(input_dir, solve_data_dir)
        simple_projections.write_def_optional_yes(input_dir, solve_data_dir)
        simple_projections.write_process_delayed(input_dir, solve_data_dir)
        simple_projections.write_process_side(solve_data_dir)
        simple_projections.write_simple_setof_projections(input_dir, solve_data_dir)
        # L0 batch 6: late projections that depend on already-Python-driven
        # solve_data CSVs (must run AFTER the calls above).
        simple_projections.write_period_solve(solve_data_dir)
        simple_projections.write_time_set(input_dir, solve_data_dir)
        simple_projections.write_enable_optional_outputs(solve_data_dir)
        simple_projections.write_node_state_subsets(solve_data_dir)
        simple_projections.write_commodity_tier_sets(input_dir, solve_data_dir)
        # L0 batch 9: DC angle bounds (calculated per-DC-node param).
        from flextool.flextoolrunner.preprocessing import dc_angle_bounds, invest_total_sets
        dc_angle_bounds.write_dc_angle_bounds(input_dir, solve_data_dir)
        # L1 batch 10: invest/divest *_total filters + cumulative ladder index.
        invest_total_sets.write_invest_total_sets(input_dir, solve_data_dir)
        invest_total_sets.write_ci_ladder_cumulative(input_dir, solve_data_dir)
        # L1 batch 11: process arc unions + co2/group set.
        from flextool.flextoolrunner.preprocessing import process_arc_unions
        process_arc_unions.write_process_arc_unions(input_dir, solve_data_dir)
        process_arc_unions.write_group_commodity_node_period_co2_total(input_dir, solve_data_dir)
        # L1 batch 12: *_in_use sets driven by per-class param taxonomy.
        process_arc_unions.write_param_in_use_sets(input_dir, solve_data_dir)

        # Validate capacity margin groups: storage nodes are excluded from capacity margin
        capacity_margin_groups: dict[str, list[str]] = {}
        for pv in db.find_parameter_values(entity_class_name="group", parameter_definition_name="has_capacity_margin"):
            if pv["parsed_value"] == "yes":
                capacity_margin_groups[pv["entity_byname"][0]] = []

        if capacity_margin_groups:
            # Get nodes in each group
            for ent in db.find_entities(entity_class_name="group__node"):
                group_name = ent["entity_byname"][0]
                node_name = ent["entity_byname"][1]
                if group_name in capacity_margin_groups:
                    capacity_margin_groups[group_name].append(node_name)

            # Get storage nodes
            storage_nodes: set[str] = set()
            for pv in db.find_parameter_values(entity_class_name="node", parameter_definition_name="node_type"):
                if pv["parsed_value"] == "storage":
                    storage_nodes.add(pv["entity_byname"][0])

            # Check each capacity margin group
            for group_name, nodes in capacity_margin_groups.items():
                storage_in_group = [n for n in nodes if n in storage_nodes]
                if storage_in_group and len(storage_in_group) == len(nodes):
                    raise FlexToolConfigError(
                        f"Capacity margin group '{group_name}' contains only storage nodes "
                        f"({', '.join(storage_in_group)}). The capacity margin constraint "
                        f"excludes storage nodes, so this group has no valid nodes."
                    )
                elif storage_in_group:
                    logger.warning(
                        "Capacity margin group '%s' contains storage nodes (%s) which will "
                        "be excluded from the capacity margin constraint.",
                        group_name, ', '.join(storage_in_group),
                    )

        _validate_group_output_memberships(db, logger)


def write_input_for_region(
    input_db_url: str,
    scenario_name: str | None,
    logger: logging.Logger,
    region_group: str,
    output_dir: Path,
    work_folder: Path | None = None,
    precision_digits: int = 0,
) -> dict:
    """Write a self-contained ``input_region_<region>/`` directory for one
    decomposition region's standalone GMPL solve (Agent 3.1).

    This is a thin wrapper around :func:`write_input` plus the regional
    filter in :mod:`flextool.flextoolrunner.region_filter`.  The monolithic
    ``input/`` is produced first (exactly as in a normal run), then the
    filter copies and filters each CSV into *output_dir*, synthesising
    virtual import/export nodes and half-flow connections for every
    cross-region process.

    Parameters
    ----------
    input_db_url, scenario_name, logger, work_folder, precision_digits
        Same semantics as :func:`write_input`.
    region_group
        The group name whose ``decomposition_method`` is ``lagrangian_region``.
        Must exist in the database.
    output_dir
        Destination directory, typically
        ``work_folder / "input_region_<region>"``.  Created if missing.

    Returns a dict: ``{"region": ..., "half_flows": [...], "kept_nodes": ...,
    "kept_units": ..., "kept_connections": ...}``.
    """
    from flextool.flextoolrunner import region_filter

    wf = work_folder if work_folder is not None else Path.cwd()
    # Produce the full input/ directory first — this is the staging area.
    # Construct an ephemeral cascade-input Provider; the region wrapper
    # only consumes the on-disk staging area, so the Provider goes
    # unused here.  Step 2.5 contract requires the Provider be present
    # even when the caller doesn't consume it.
    from flextool.engine_polars._flex_data_provider import FlexDataProvider
    _ephemeral_provider = FlexDataProvider()
    write_input(
        input_db_url,
        scenario_name,
        logger,
        work_folder=work_folder,
        precision_digits=precision_digits,
        provider=_ephemeral_provider,
    )
    all_regions = region_filter.discover_decomposition_regions_from_db(input_db_url)
    if region_group not in all_regions:
        raise FlexToolConfigError(
            f"Region '{region_group}' is not declared with "
            f"decomposition_method='lagrangian_region' in the database. "
            f"Available regions: {sorted(all_regions) or '(none)'}"
        )
    result = region_filter.build_region_directory(
        input_dir=wf / "input",
        output_dir=Path(output_dir),
        region=region_group,
        all_regions=all_regions,
    )
    region_filter.write_region_coupling_manifest(
        work_folder=wf,
        results=[result],
    )
    return result


def _validate_group_output_memberships(db, logger: logging.Logger) -> None:
    """Warn when a group-level output flag is ``yes`` but the group lacks
    the membership class required for that output to produce any data.

    Four silent-no-op cases are detected:

    * ``output_nodeGroup_dispatch: yes`` with no ``group__node`` row
    * ``output_nodeGroup_indicators: yes`` with no ``group__node`` row
    * ``output_flowGroup_indicators: yes`` with no ``group__unit__node``
      **or** ``group__connection__node`` row
    * ``flow_aggregator: yes`` with no ``group__unit__node`` **or**
      ``group__connection__node`` row

    Only warnings are emitted — a user may deliberately stage a partial
    configuration.
    """
    # Collect groups that are members of the relevant entity classes.
    groups_with_node_members: set[str] = set()
    for ent in db.find_entities(entity_class_name="group__node"):
        byname = ent["entity_byname"]
        if byname:
            groups_with_node_members.add(byname[0])

    groups_with_flow_members: set[str] = set()
    for cls in ("group__unit__node", "group__connection__node"):
        for ent in db.find_entities(entity_class_name=cls):
            byname = ent["entity_byname"]
            if byname:
                groups_with_flow_members.add(byname[0])

    # (parameter_name, required_membership_description, membership_set)
    checks: list[tuple[str, str, set[str]]] = [
        ("output_nodeGroup_dispatch", "group__node", groups_with_node_members),
        ("output_nodeGroup_indicators", "group__node", groups_with_node_members),
        (
            "output_flowGroup_indicators",
            "group__unit__node or group__connection__node",
            groups_with_flow_members,
        ),
        (
            "flow_aggregator",
            "group__unit__node or group__connection__node",
            groups_with_flow_members,
        ),
    ]
    for param_name, required_members, membership_set in checks:
        for pv in db.find_parameter_values(
            entity_class_name="group", parameter_definition_name=param_name
        ):
            if pv["type"] is None:
                continue
            if pv["parsed_value"] != "yes":
                continue
            group_name = pv["entity_byname"][0]
            if group_name not in membership_set:
                logger.warning(
                    "Group '%s' has %s: yes but no %s members — output will be empty.",
                    group_name, param_name, required_members,
                )


# Step 2.5-E Phase E — write_parameter() deleted; its body lives in
# flextool.spinedb_backend.SpineDBBackend.parameter_values(), which
# returns a polars DataFrame for placement into the cascade-input
# Provider.  No disk write at this site.
#
# Step 2.5 Phase 2 — write_default_values() deleted; its body lives in
# flextool.spinedb_backend.SpineDBBackend.parameter_defaults().
