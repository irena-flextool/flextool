"""FlexTool LP model: builds the FlexTool LP from FlexData ``d`` into a polar_high Problem ``m``.

:func:`build_flextool` (``_build``) is the table of contents: ``_prepare`` (flags, data checks, index sets),
``_declare_variables``, ``_shared_expressions``, then one step per equation family in emit order; the
objective is the sum of the terms the ``_obj_*`` / ``objective_terms`` steps return.  Every step takes the
``_Build`` context ``b`` (``_context``).  The core model that every run uses lives in this package as
``_eq_<family>`` / ``_obj_<family>``, one module per equation family: ``_prepare``, ``_variables``,
``_rp_storage``, ``_node_balance``, ``_flows``, ``_ramp``, ``_invest``, ``_conversion``, ``_co2``,
``_user_constraints``, ``_storage_state``, ``_profiles``, ``_objective``; ``_storage_links`` holds the storage
helpers several of them read.  This module re-exports the public names only.  An optional feature that brings
its own variables lives in its own module of ``model/`` with ``add_variables(b)`` / ``add_constraints(b)`` /
``objective_terms(b)``.  Shared helpers: ``model/_common``.

Each row, objective term and variable has an ``EQ.`` / ``OBJ.`` / ``VAR.`` header with its formula;
grep ``EQ. ─── <name>`` to find an equation.

Variables, in declaration (column) order.  p process, n node, d period, t step, b base period, g group,
r reserve, ud up/down, c commodity, i price tier.  "× unitsize" converts to MW / MWh.
  v_flow[p,source,sink,d,t]            flow on a process arc; × unitsize = MW (a coarse process: one column per
                                       coarse step, read on every step through _coarse_flows.alias)
  vq_state_up / _down[n,d,t]           balance slack (shortfall / surplus), MW
  v_state[n,d,t]                       storage state; × state_unitsize = MWh
  v_state_inter[n,d,b]                 representative-period storage state at base-period boundaries of d
  v_state_rp_start[n,d,t]              free start state of each representative block
  v_{online,startup,shutdown}_{linear,integer}[p,d,t]
                                       units (connections) online / starting / stopping, count of unitsize
                                                                                              [_online]
  v_{invest,divest}_{p,n}[p|n,d]       capacity built / retired in d, count of unitsize
  v_divest_built_{p,n}[p|n,c,d], v_divested_built_{p,n}[p|n,d]
                                       model-built cohort c retired in d, retired so far     [_retire_built]
  v_extend_{p,n}[p|n,c,d], v_extended_{p,n}[p|n,d]
                                       cohort c kept beyond its end of life from d, extended in service [_extend]
  v_reserve[p,r,ud,n,d,t]              reserve held by p for n; × unitsize = MW               [_reserve]
  vq_reserve[r,ud,g,d,t]               reserve shortfall, fraction of the requirement scale   [_reserve]
  v_reserve_total[r,ud,g,d,t]          total reserve of a group with n-1 rows, MW             [_reserve]
  v_flow_back[p,source,sink,d,t]       reverse flow of 2-way 1-var and DC arcs; × unitsize = MW
  v_exact_forward[p,d,t]               exact connection set to its forward direction, binary   [_exact]
  v_angle[n,d,t]                       voltage angle, rad in [−π, π]                          [_dc_power_flow]
                                       (units online, start-ups, reserves, directions, angles and the inertia /
                                       non-synchronous slacks of coarse entities: one column per coarse step)
  v_trade[c,n,d,i]                     commodity bought in tier i; × commodity unitsize = MWh [_commodity_ladder]
  vq_capacity_margin[g,d], vq_inertia[g,d,t], vq_non_synchronous[g,d,t]
                                       group floor slacks, fraction of the group's scale      [_group_slack]

Equation families, in emit order (module of model/ in brackets when not this package):
  Balance & storage   rp_inter_period_{balance,cyclic,fan_start,max_state}, maxState_rp_start;
                      rp_inter_period_{start_fix,start_carry,end_fix,start_max_state,start_min_state,end_match,start_match,
                      reference_value} (the ends of the base-period chain); nodeBalance_eq;
                      nodeBalanceCoarse_eq, stateConstantCoarse_eq (coarse new_stepduration groups);
                      nodeBalancePeriod_eq; node_balance_fix_quantity_eq_lower; node_storage_usage_fix_le;
                      stateConstantWithinBlock_eq, nodeBalanceBlock_eq (bind_intraperiod_blocks)
  Flows & capacity    maxFlow, maxFlow_negCap, maxFlow_back; maxOutputSum(_negCap);
                      dc_reference_angle_eq, dc_flow_eq [_dc_power_flow]
                      (the arc-local families of coarse processes — maxFlow, maxOutputSum, conversion_indirect,
                      conversion_piecewise, minInputShare and profile_flow_* — are written once per coarse step as
                      <family>_coarse [_coarse_flows])
  Units               max{Online,Startup,Shutdown}, online_change, maxFlow_online[_shutdown[_branch]],
                      maxFlow_back_online, maxOutputSum_online, maxInputFlow_online[_shutdown[_branch]],
                      minimum_{uptime,downtime}, online_fix_eq [_online];
                      maxFlow_exact_{forward,reverse} (exact connections) [_exact];
                      minFlow_{minload,output_floor,reserve,reserve_arc} [_unit_floors];
                      ramp_{sink,source}_{up,down}_constraint; ramp_change_eq (ramp cost)
  Investment          max{Invest,Divest}_var_bound[_n], _entity_period_{p,n}, _entity_total[_path][_n];
                      maxRetired_existing_{p,n}; divestedBuilt_{p,n}, maxDivest_cohort_{p,n} [_retire_built];
                      extendedCapacity_{p,n}, maxExtend_cohort_{p,n} [_extend]
  Conversion          conversion_indirect (delayed inputs from [_delay]); conversion_piecewise (part-load curves of
                      piecewise_linear units); minInputShare
  Emissions & user    co2_max_period, co2_max_total; process_constraint_{equal,less_than,greater_than}[_period]
  Storage state       maxState; profile_state_{upper_limit,lower_limit,fixed};
                      storage_state_start_binding_cyclic_period; storage_within_solve_path_end,
                      storage_state_fan_start (stochastic fans); storage_split_path_end (split realization
                      rolls); storage_state_end_binding;
                      storage_state_roll_{anchor,close}[_share] (rolling cyclic storage, per instance or
                      period); storage_state_roll_match (rolling forward chains);
                      storage_state_end_match[_share] (forward chains whose start the solve decides);
                      storage_state_solve_horizon_reference_value
  Profiles            profile_flow_{upper_limit,lower_limit,fixed}
  Objective           balance slack penalty, commodity cost and sales, CO2 price, variable cost, startup cost,
                      ramp cost, storage reference price, invest/divest annuities, existing fixed cost (optional
                      constant),
                      divest_built_cost [_retire_built], extend_cost [_extend]
  Feature modules     capacityMargin, inertia_constraint, non_sync_constraint + slack penalties [_group_slack];
                      reserveBalance_*, reserve_total_upper, reserve_process_*, connection_reserve_*,
                      reserve_storage_*
                      + shortfall penalty [_reserve];
                      investment floors, cumulative, group and flow-group limits [_cumulative_invest];
                      commodity_ladder_balance, ladder_tier_cap_* + tier costs [_commodity_ladder]
  Stochastic ties     non_anticipativity_* (dispatch, then investment) [_non_anticipativity]
"""

from flextool.engine.model.core._context import (
    ALWAYS, CAPACITY_MARGIN, CO2_CAP, CO2_CAP_TOTAL, DIVEST, DIVEST_BUILT, DIVEST_N, EXTEND, INDIRECT, INVEST,
    MIN_LOAD, MINLOAD_EFF, OBJ_COEF_ZERO_REL_THRESHOLD, ONLINE, PIECEWISE, PROCESSES, PROFILES, RAMP, RAMP_COST,
    STARTUP_COST_INTEGER, STARTUP_COST_LINEAR, STORAGE, USER_CSTR,
)
from flextool.engine.model.core._build import build_flextool
from flextool.engine.model.core._prepare import DC_ONLINE_MESSAGE, STORAGE_START_END_MESSAGE
from flextool.engine.model.core._ramp import ramp_step_links

__all__ = [
    "build_flextool", "OBJ_COEF_ZERO_REL_THRESHOLD", "DC_ONLINE_MESSAGE", "STORAGE_START_END_MESSAGE",
    "ramp_step_links",
    "ALWAYS", "CAPACITY_MARGIN", "CO2_CAP", "CO2_CAP_TOTAL", "DIVEST", "DIVEST_BUILT", "DIVEST_N", "EXTEND",
    "INDIRECT", "INVEST", "MIN_LOAD", "MINLOAD_EFF", "ONLINE", "PIECEWISE", "PROCESSES", "PROFILES", "RAMP",
    "RAMP_COST", "STARTUP_COST_INTEGER", "STARTUP_COST_LINEAR", "STORAGE", "USER_CSTR",
]
