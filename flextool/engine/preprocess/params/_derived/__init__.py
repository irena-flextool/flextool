"""Derived parameters: the FlexData fields computed from the input source and the solve's frames, in seven
batches that ``flex_data._db_overrides`` applies in order (:func:`apply_derived_a` … :func:`apply_derived_g`,
plus :func:`apply_synthetic_invest_sets` and :func:`apply_delay_params` for a synthetic sub-solve).

Each helper takes an :class:`flextool._input_source.InputSource` plus
any already-computed sibling frames it depends on (foundational order
established in §7.3 Phase Γ.3a) and returns the eager
``Param`` / ``DataFrame`` shape consumed by downstream model-build.

Lazy-evaluation pattern
-----------------------
``InputSource.parameter()`` collects at the source boundary.  Helpers
compose multi-call results via :class:`pl.LazyFrame` and ``.collect()``
once per public function.

Per-solve scope
---------------
Several Params (``dt``, ``p_step_duration``, ``p_timestep_weight``,
``pdt_branch_weight``, ``pd_branch_weight``) are per-active-solve.  The
helpers accept an explicit ``active_solve`` argument; the
:func:`apply_derived_a` integration entrypoint reads
``solve_data/solve_current.csv`` from the workdir to pick the active
solve, mirroring the established hand-off used by the CSV path.

Modules (this package re-exports the public names; private helpers are imported from their module):

* ``_scope`` — Provider access, the ``source`` readers, the active solve, the recourse flag; ``_unitsize`` — the
  unit size cascades; ``_periods`` — the solve's periods, ``period_in_use`` and branch periods.
* ``_time`` — Γ.3.A time and weighting, :func:`apply_derived_a`; ``_inflation`` — inflation and the year cascade.
* ``_process_method`` — the process method classification; ``_process_flows`` — flow, conversion and minimum
  load params, :func:`apply_derived_b`; ``_arc_capacity`` — per-arc capacity and the indirect input-arc limit.
* ``_existing_capacity`` — existing counts and capacity, :func:`apply_derived_d`; ``_invest_sets`` — the
  investment sets and ``p_entity_max_units``; ``_invest_apply`` — the recourse anchor scope,
  :func:`apply_derived_c` and :func:`apply_synthetic_invest_sets`.
* ``_lookbacks`` — minimum up / down time; ``_scaling_inflow`` — groups, row scaling and inflow; ``_reserves``
  — reserve activity and ``prundt``; ``_delay`` — unit delays.
* ``_active_steps``, ``_dtttdt``, ``_storage`` — Γ.3.E storage block algebra, :func:`apply_derived_e`.
* ``_handoff_state`` — the rolling-handoff investment state, :func:`apply_derived_f`; ``_ladder`` — the commodity
  price ladder, :func:`apply_derived_g`.
"""
from __future__ import annotations

from flextool.engine.preprocess.params._derived._unitsize import UNITSIZE_DEFAULT, unitsize_cascade
from flextool.engine.preprocess.params._derived._periods import run_start_period
from flextool.engine.preprocess.params._derived._inflation import (
    p_inflation_op_from_source,
    p_inflation_op_multi_year_from_source,
)
from flextool.engine.preprocess.params._derived._process_method import (
    process_indirect_set,
    process_input_flows,
    process_output_flows,
    process_indirect_dt,
    one_variable_efficiency_ignored,
    apply_one_variable_efficiency,
)
from flextool.engine.preprocess.params._derived._delay import (
    p_process_delay_weight_from_source,
    process_delayed__duration_from_weight,
    apply_delay_params,
    dtt__delay_duration_from_source,
)
from flextool.engine.preprocess.params._derived._arc_capacity import (
    process_source_sink_uncapped_from_source,
    p_arc_max_cap_coef_from_source,
    wire_arc_capacity_params,
    p_flow_upper_from_source,
    p_indirect_input_cap_from_source,
)
from flextool.engine.preprocess.params._derived._reserves import (
    process_reserve_upDown_node_active_from_source,
    prundt_from_source,
)
from flextool.engine.preprocess.params._derived._existing_capacity import (
    p_process_existing_count_from_source,
    p_flow_upper_existing_from_source,
    p_entity_all_existing_from_source,
    D_PUBLIC_FIELDS,
    apply_derived_d,
)
from flextool.engine.preprocess.params._derived._time import (
    dt_and_step_duration_from_source,
    p_period_share_from_source,
    p_window_fraction_from_shares,
    periods_of_dt,
    bookkeeping_periods,
    p_years_represented_d_from_source,
    solve_has_rp_timeset,
    p_timestep_weight_from_source,
    pd_branch_weight_from_source,
    pdt_branch_weight_from_source,
    DERIVED_A_FIELDS,
    apply_derived_a,
    pd_branch_weight_full_from_source,
    pdt_branch_weight_full_from_source,
)
from flextool.engine.preprocess.params._derived._process_flows import (
    flow_to_n,
    flow_from_n,
    p_flow_constraint_coef_from_source,
    p_slope_from_source,
    p_min_load_pdt_from_source,
    process_minload_from_p_min_load,
    DERIVED_B_FIELDS,
    apply_derived_b,
    p_section_from_source,
)
from flextool.engine.preprocess.params._derived._invest_sets import (
    INVEST_METHODS_INVEST_NO_LIMIT,
    ed_invest_set_from_source,
    ed_divest_set_from_source,
    retirable_existing_pairs,
    p_entity_max_units_from_source,
)
from flextool.engine.preprocess.params._derived._lookbacks import (
    uptime_lookback_from_source,
    downtime_lookback_from_source,
    apply_mintime_lookbacks,
    pdt_uptime_set_from_lookback,
    pdt_downtime_set_from_lookback,
    online_mintime_carry_from_source,
)
from flextool.engine.preprocess.params._derived._scaling_inflow import (
    process_group_inside_nonSync_from_flex,
    p_group_capacity_for_scaling_from_source,
    p_inv_group_cap_from_source,
    p_node_capacity_for_scaling_from_source,
    p_positive_inflow_from_inflow,
    p_negative_inflow_from_inflow,
    real_inflow,
    pdtNodeInflow_per_step_from_inflow,
)
from flextool.engine.preprocess.params._derived._handoff_state import (
    p_entity_previously_invested_capacity_from_workdir,
    p_entity_invested_from_workdir,
    p_entity_divested_from_handoff,
    apply_derived_f,
)
from flextool.engine.preprocess.params._derived._invest_apply import (
    DERIVED_C_FIELDS,
    zero_cost_invest_notice,
    apply_derived_c,
    apply_synthetic_invest_sets,
)
from flextool.engine.preprocess.params._derived._dtttdt import (
    dtttdt_from_source,
    dtttdt_with_jump_from_source,
    period_block_family_from_source,
    dtttdt_forward_only_from_dtttdt,
    dtttdt_within_solve_from_dtttdt,
    read_period_first_of_solve,
    dtttdt_block_interior_from_dtttdt,
    dtttdt_block_lag_from_dtttdt,
)
from flextool.engine.preprocess.params._derived._storage import (
    nodeStateBlock_from_source,
    p_state_existing_capacity_from_source,
    p_state_upper_from_source,
    storage_use_reference_value_from_source,
    p_roll_continue_state_from_handoff,
    p_fix_storage_quantity_from_handoff,
    dtt_timeline_matching_from_workdir,
    period_branch_from_source,
    E_PUBLIC_FIELDS,
    apply_derived_e,
)
from flextool.engine.preprocess.params._derived._ladder import (
    p_f_d_k_from_source,
    p_ladder_cum_realized_mwh_from_workdir,
    G_PUBLIC_FIELDS,
    apply_derived_g,
)

__all__ = [
    "UNITSIZE_DEFAULT",
    "unitsize_cascade",
    "run_start_period",
    "p_inflation_op_from_source",
    "p_inflation_op_multi_year_from_source",
    "process_indirect_set",
    "process_input_flows",
    "process_output_flows",
    "process_indirect_dt",
    "one_variable_efficiency_ignored",
    "apply_one_variable_efficiency",
    "p_process_delay_weight_from_source",
    "process_delayed__duration_from_weight",
    "apply_delay_params",
    "dtt__delay_duration_from_source",
    "process_source_sink_uncapped_from_source",
    "p_arc_max_cap_coef_from_source",
    "wire_arc_capacity_params",
    "p_flow_upper_from_source",
    "p_indirect_input_cap_from_source",
    "process_reserve_upDown_node_active_from_source",
    "prundt_from_source",
    "p_process_existing_count_from_source",
    "p_flow_upper_existing_from_source",
    "p_entity_all_existing_from_source",
    "D_PUBLIC_FIELDS",
    "apply_derived_d",
    "dt_and_step_duration_from_source",
    "p_period_share_from_source",
    "p_window_fraction_from_shares",
    "periods_of_dt",
    "bookkeeping_periods",
    "p_years_represented_d_from_source",
    "solve_has_rp_timeset",
    "p_timestep_weight_from_source",
    "pd_branch_weight_from_source",
    "pdt_branch_weight_from_source",
    "DERIVED_A_FIELDS",
    "apply_derived_a",
    "pd_branch_weight_full_from_source",
    "pdt_branch_weight_full_from_source",
    "flow_to_n",
    "flow_from_n",
    "p_flow_constraint_coef_from_source",
    "p_slope_from_source",
    "p_min_load_pdt_from_source",
    "process_minload_from_p_min_load",
    "DERIVED_B_FIELDS",
    "apply_derived_b",
    "p_section_from_source",
    "INVEST_METHODS_INVEST_NO_LIMIT",
    "ed_invest_set_from_source",
    "ed_divest_set_from_source",
    "retirable_existing_pairs",
    "p_entity_max_units_from_source",
    "uptime_lookback_from_source",
    "downtime_lookback_from_source",
    "apply_mintime_lookbacks",
    "pdt_uptime_set_from_lookback",
    "pdt_downtime_set_from_lookback",
    "online_mintime_carry_from_source",
    "process_group_inside_nonSync_from_flex",
    "p_group_capacity_for_scaling_from_source",
    "p_inv_group_cap_from_source",
    "p_node_capacity_for_scaling_from_source",
    "p_positive_inflow_from_inflow",
    "p_negative_inflow_from_inflow",
    "real_inflow",
    "pdtNodeInflow_per_step_from_inflow",
    "p_entity_previously_invested_capacity_from_workdir",
    "p_entity_invested_from_workdir",
    "p_entity_divested_from_handoff",
    "apply_derived_f",
    "DERIVED_C_FIELDS",
    "zero_cost_invest_notice",
    "apply_derived_c",
    "apply_synthetic_invest_sets",
    "dtttdt_from_source",
    "dtttdt_with_jump_from_source",
    "period_block_family_from_source",
    "dtttdt_forward_only_from_dtttdt",
    "dtttdt_within_solve_from_dtttdt",
    "read_period_first_of_solve",
    "dtttdt_block_interior_from_dtttdt",
    "dtttdt_block_lag_from_dtttdt",
    "nodeStateBlock_from_source",
    "p_state_existing_capacity_from_source",
    "p_state_upper_from_source",
    "storage_use_reference_value_from_source",
    "p_roll_continue_state_from_handoff",
    "p_fix_storage_quantity_from_handoff",
    "dtt_timeline_matching_from_workdir",
    "period_branch_from_source",
    "E_PUBLIC_FIELDS",
    "apply_derived_e",
    "p_f_d_k_from_source",
    "p_ladder_cum_realized_mwh_from_workdir",
    "G_PUBLIC_FIELDS",
    "apply_derived_g",
]
