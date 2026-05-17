"""Specification tables + validators for the SpineDB → Provider pipeline.

This module holds the canonical EAV → tabular materialiser
specifications (``_ENTITY_SPECS``, ``_PARAMETER_SPECS``,
``_DEFAULT_VALUES_SPECS``), the ``METHODS_MAPPING`` lookup, plus a
handful of validators that gate :func:`flextool.input_derivation.run`.

Pre-Step-2.5 this content lived in
``flextool.flextoolrunner.input_writer``; that 2356-LOC monolith was
torn down in items 13-18 and its content split into:

* :mod:`flextool.input_derivation` — the run-time entry point;
* this module — declarative specs + validators;
* :mod:`flextool.spinedb_backend` — the EAV → polars materialisers
  (``SpineDBBackend.entities`` / ``parameter_values`` /
  ``parameter_defaults``);
* :mod:`flextool.engine_polars._writer_*` — the native polars writers
  that consume the materialised frames and emit derived ``solve_data/``
  frames.
"""
from __future__ import annotations

import logging
from typing import NamedTuple

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


def _validate_capacity_margin_groups(db, logger: logging.Logger) -> None:
    """Storage nodes are excluded from the capacity-margin constraint.
    Raise if any has_capacity_margin group contains *only* storage
    nodes (constraint would have no valid members); warn if a mix is
    present.
    """
    capacity_margin_groups: dict[str, list[str]] = {}
    for pv in db.find_parameter_values(
        entity_class_name="group", parameter_definition_name="has_capacity_margin",
    ):
        if pv["parsed_value"] == "yes":
            capacity_margin_groups[pv["entity_byname"][0]] = []
    if not capacity_margin_groups:
        return
    for ent in db.find_entities(entity_class_name="group__node"):
        g, n = ent["entity_byname"][0], ent["entity_byname"][1]
        if g in capacity_margin_groups:
            capacity_margin_groups[g].append(n)
    storage_nodes: set[str] = set()
    for pv in db.find_parameter_values(
        entity_class_name="node", parameter_definition_name="node_type",
    ):
        if pv["parsed_value"] == "storage":
            storage_nodes.add(pv["entity_byname"][0])
    for g, nodes in capacity_margin_groups.items():
        storage_in_group = [n for n in nodes if n in storage_nodes]
        if storage_in_group and len(storage_in_group) == len(nodes):
            raise FlexToolConfigError(
                f"Capacity margin group '{g}' contains only storage nodes "
                f"({', '.join(storage_in_group)}). The capacity margin constraint "
                f"excludes storage nodes, so this group has no valid nodes."
            )
        elif storage_in_group:
            logger.warning(
                "Capacity margin group '%s' contains storage nodes (%s) which will "
                "be excluded from the capacity margin constraint.",
                g, ', '.join(storage_in_group),
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
