from types import SimpleNamespace
from pathlib import Path
import pandas as pd

def read_variables(output_dir):
    """
    Read all variable CSV files into a namespace.
    
    Returns:
        SimpleNamespace: Namespace with variables as attributes
    """
    output_path = Path(output_dir)
    v = SimpleNamespace()
    
    # Variables with (solve, period, time) index

    v.obj = pd.read_csv(output_path / 'v_obj.csv', header=[0], index_col=[0]).astype(float)    
    v.flow = pd.read_csv(output_path / 'v_flow.csv', header=[0, 1, 2], index_col=[0, 1, 2]).astype(float)
    v.ramp = pd.read_csv(output_path / 'v_ramp.csv', header=[0, 1, 2], index_col=[0, 1, 2]).astype(float)
    v.reserve = pd.read_csv(output_path / 'v_reserve.csv', header=[0, 1, 2, 3], index_col=[0, 1, 2]).astype(float)
    v.state = pd.read_csv(output_path / 'v_state.csv', index_col=[0, 1, 2]).astype(float)
    v.online_linear = pd.read_csv(output_path / 'v_online_linear.csv', index_col=[0, 1, 2]).astype(float)
    v.startup_linear = pd.read_csv(output_path / 'v_startup_linear.csv', index_col=[0, 1, 2]).astype(float)
    v.shutdown_linear = pd.read_csv(output_path / 'v_shutdown_linear.csv', index_col=[0, 1, 2]).astype(float)
    v.online_integer = pd.read_csv(output_path / 'v_online_integer.csv', index_col=[0, 1, 2]).astype(float)
    v.startup_integer = pd.read_csv(output_path / 'v_startup_integer.csv', index_col=[0, 1, 2]).astype(float)
    v.shutdown_integer = pd.read_csv(output_path / 'v_shutdown_integer.csv', index_col=[0, 1, 2]).astype(float)
    v.q_state_up = pd.read_csv(output_path / 'vq_state_up.csv', index_col=[0, 1, 2]).astype(float)
    v.q_state_down = pd.read_csv(output_path / 'vq_state_down.csv', index_col=[0, 1, 2]).astype(float)
    v.q_reserve = pd.read_csv(output_path / 'vq_reserve.csv', header=[0, 1, 2], index_col=[0, 1, 2]).astype(float)
    v.q_inertia = pd.read_csv(output_path / 'vq_inertia.csv', index_col=[0, 1, 2]).astype(float)
    v.q_non_synchronous = pd.read_csv(output_path / 'vq_non_synchronous.csv', index_col=[0, 1, 2]).astype(float)
    v.q_state_up_group = pd.read_csv(output_path / 'vq_state_up_group.csv', index_col=[0, 1, 2]).astype(float)
    v.q_capacity_margin = pd.read_csv(output_path / 'vq_capacity_margin.csv', index_col=[0, 1]).astype(float)
    v.invest = pd.read_csv(output_path / 'v_invest.csv', index_col=[0, 1]).astype(float)
    v.divest = pd.read_csv(output_path / 'v_divest.csv', index_col=[0, 1]).astype(float)
    v.dual_node_balance = pd.read_csv(output_path / 'v_dual_node_balance.csv', index_col=[0, 1, 2]).astype(float)
    v.dual_reserve_balance = pd.read_csv(output_path / 'v_dual_reserve__upDown__group__period__t.csv', header=[0, 1, 2], index_col=[0, 1, 2]).astype(float)
    v.dual_invest_unit = pd.read_csv(output_path / 'v_dual_invest_unit.csv', index_col=[0, 1]).astype(float)
    v.dual_invest_connection = pd.read_csv(output_path / 'v_dual_invest_connection.csv', index_col=[0, 1]).astype(float)
    v.dual_invest_node = pd.read_csv(output_path / 'v_dual_invest_node.csv', index_col=[0, 1]).astype(float)

    v.flow.index.names = ['solve', 'period', 'time']    
    v.ramp.index.names = ['solve', 'period', 'time']
    v.reserve.index.names = ['solve', 'period', 'time']
    v.state.index.names = ['solve', 'period', 'time']
    v.online_linear.index.names = ['solve', 'period', 'time']
    v.startup_linear.index.names = ['solve', 'period', 'time']
    v.shutdown_linear.index.names = ['solve', 'period', 'time']
    v.online_integer.index.names = ['solve', 'period', 'time']
    v.startup_integer.index.names = ['solve', 'period', 'time']
    v.shutdown_integer.index.names = ['solve', 'period', 'time']
    v.q_state_up.index.names = ['solve', 'period', 'time']
    v.q_state_down.index.names = ['solve', 'period', 'time']
    v.q_reserve.index.names = ['solve', 'period', 'time']
    v.q_inertia.index.names = ['solve', 'period', 'time']
    v.q_non_synchronous.index.names = ['solve', 'period', 'time']
    v.q_state_up_group.index.names = ['solve', 'period', 'time']
    v.q_capacity_margin.index.names = ['solve', 'period']
    v.invest.index.names = ['solve', 'period']
    v.divest.index.names = ['solve', 'period']
    v.dual_node_balance.index.names = ['solve', 'period', 'time']
    v.dual_reserve_balance.index.names = ['solve', 'period', 'time']
    v.dual_invest_unit.index.names = ['solve', 'period']
    v.dual_invest_connection.index.names = ['solve', 'period']
    v.dual_invest_node.index.names = ['solve', 'period']

    # Create multi-index for variables with single header row
    v.state.columns.name = 'node'
    v.online_linear.columns.name = 'process'
    v.startup_linear.columns.name = 'process'
    v.shutdown_linear.columns.name = 'process'
    v.online_integer.columns.name = 'process'
    v.startup_integer.columns.name = 'process'
    v.shutdown_integer.columns.name = 'process'
    v.q_state_up.columns.name = 'node'
    v.q_state_down.columns.name = 'node'
    v.q_inertia.columns.name = 'group'
    v.q_non_synchronous.columns.name = 'group'
    v.q_state_up_group.columns.name = 'group'
    v.q_capacity_margin.columns.name = 'group'
    v.invest.columns.name = 'entity'
    v.divest.columns.name = 'entity'
    v.dual_node_balance.columns.name = 'node'
    v.dual_invest_unit.columns.name = 'unit'
    v.dual_invest_connection.columns.name = 'connection'
    v.dual_invest_node.columns.name = 'node'

    # Add multi-index to variables with multiple header rows (this multi-index creation works also when the dataframe is empty)
    v.flow.columns = pd.MultiIndex.from_tuples(
        [(col[0], col[1], col[2]) for col in v.flow.columns],
        names=['process', 'source', 'sink']
    )
    v.ramp.columns = pd.MultiIndex.from_tuples(
        [(col[0], col[1], col[2]) for col in v.ramp.columns],
        names=['process', 'source', 'sink']
    )
    v.reserve.columns = pd.MultiIndex.from_tuples(
        [(col[0], col[1], col[2], col[3]) for col in v.reserve.columns],
        names=['process', 'reserve', 'updown', 'node']
    )
    v.q_reserve.columns = pd.MultiIndex.from_tuples(
        [(col[0], col[1], col[2]) for col in v.q_reserve.columns],
        names=['reserve', 'updown', 'node_group']
    )
    v.dual_reserve_balance.columns = pd.MultiIndex.from_tuples(
        [(col[0], col[1], col[2]) for col in v.dual_reserve_balance.columns],
        names=['reserve', 'updown', 'node_group']
    )

    return v


def read_parameters(output_dir):
    """
    Read all parameter CSV files into a namespace.
    
    Returns:
        SimpleNamespace: Namespace with parameters as attributes
    """
    output_path = Path(output_dir)
    p = SimpleNamespace()

    # Parameters that have only one row of data
    p.node = pd.read_csv(output_path / 'p_node.csv', index_col=0).astype(float)
    p.entity_unitsize = pd.read_csv(output_path / 'p_entity_unitsize.csv', index_col=0).loc['value'].astype(float)
    # These could be empty of data, so they'll return an empty series in that case
    df = pd.read_csv(output_path / 'p_commodity_co2_content.csv', header=[0], index_col=0).astype(float)
    p.commodity_co2_content = df.loc['value'] if 'value' in df.index else pd.Series(dtype=float)
    df = pd.read_csv(output_path / 'p_process_sink_coefficient.csv', header=[0, 1], index_col=0).astype(float)
    p.process_sink_coefficient = df.loc['value'] if 'value' in df.index else pd.Series(dtype=float)
    df = pd.read_csv(output_path / 'p_process_source_coefficient.csv', header=[0, 1], index_col=0).astype(float)
    p.process_source_coefficient = df.loc['value'] if 'value' in df.index else pd.Series(dtype=float)
    df = pd.read_csv(output_path / 'p_reserve_upDown_group_penalty.csv', header=[0, 1, 2], index_col=0).astype(float)
    p.reserve_upDown_group_penalty = df.loc['value'] if 'value' in df.index else pd.Series(dtype=float)

    # Parameters with (period, time) index and multi-dimensional columns
    p.step_duration = pd.read_csv(output_path / 'p_step_duration.csv', index_col=[0, 1, 2])['value'].astype(float)
    p.flow_min = pd.read_csv(output_path / 'p_flow_min.csv', header=[0, 1, 2], index_col=[0, 1, 2]).astype(float)
    p.flow_max = pd.read_csv(output_path / 'p_flow_max.csv', header=[0, 1, 2], index_col=[0, 1, 2]).astype(float)
    p.process_source = pd.read_csv(output_path / 'p_process_source.csv', header=[0, 1], index_col=[0]).astype(float)
    p.process_sink = pd.read_csv(output_path / 'p_process_sink.csv', header=[0, 1], index_col=0).astype(float)
    p.process_slope = pd.read_csv(output_path / 'pdtProcess_slope.csv', index_col=[0, 1, 2]).astype(float)
    p.process_section = pd.read_csv(output_path / 'pdtProcess_section.csv', index_col=[0, 1, 2]).astype(float)
    p.process_availability = pd.read_csv(output_path / 'pdtProcess_availability.csv', index_col=[0, 1, 2]).astype(float)
    p.process_source_sink_varCost = pd.read_csv(output_path / 'pdtProcess_source_sink_varCost.csv', header=[0, 1, 2], index_col=[0, 1, 2]).astype(float)
    p.node_self_discharge_loss = pd.read_csv(output_path / 'pdtNode_self_discharge_loss.csv', index_col=[0, 1, 2]).astype(float)
    p.node_penalty_up = pd.read_csv(output_path / 'pdtNode_penalty_up.csv', index_col=[0, 1, 2]).astype(float)
    p.node_penalty_down = pd.read_csv(output_path / 'pdtNode_penalty_down.csv', index_col=[0, 1, 2]).astype(float)
    p.node_inflow = pd.read_csv(output_path / 'pdtNodeInflow.csv', index_col=[0, 1, 2]).astype(float)
    p.commodity_price = pd.read_csv(output_path / 'pdtCommodity_price.csv', index_col=[0, 1, 2]).astype(float)
    p.group_co2_price = pd.read_csv(output_path / 'pdtGroup_co2_price.csv', index_col=[0, 1, 2]).astype(float)
    p.reserve_upDown_group_reservation = pd.read_csv(output_path / 'pdtReserve_upDown_group_reservation.csv', header=[0, 1, 2], index_col=[0, 1, 2]).astype(float)
    p.profile = pd.read_csv(output_path / 'pdtProfile.csv', index_col=[0, 1, 2]).astype(float)
    p.years_from_start_d = pd.read_csv(output_path / 'p_years_from_start_d.csv', index_col=[0, 1])['value'].astype(float)
    p.years_represented_d = pd.read_csv(output_path / 'p_years_represented_d.csv', index_col=[0, 1])['value'].astype(float)
    p.entity_max_units = pd.read_csv(output_path / 'p_entity_max_units.csv', index_col=[0, 1]).astype(float)
    p.entity_all_existing = pd.read_csv(output_path / 'p_entity_all_existing.csv', index_col=[0, 1]).astype(float)
    p.entity_pre_existing = pd.read_csv(output_path / 'p_entity_pre_existing.csv', index_col=[0, 1]).astype(float)
    p.process_startup_cost = pd.read_csv(output_path / 'pdProcess_startup_cost.csv', index_col=[0, 1]).astype(float)
    p.entity_fixed_cost = pd.read_csv(output_path / 'ed_fixed_cost.csv', index_col=[0, 1]).astype(float)
    p.entity_lifetime_fixed_cost = pd.read_csv(output_path / 'ed_lifetime_fixed_cost.csv', index_col=[0, 1]).astype(float)
    p.entity_lifetime_fixed_cost_divest = pd.read_csv(output_path / 'ed_lifetime_fixed_cost_divest.csv', index_col=[0, 1]).astype(float)
    p.node_annual_flow = pd.read_csv(output_path / 'pdNode_annual_flow.csv', index_col=[0, 1]).astype(float)
    p.group_penalty_inertia = pd.read_csv(output_path / 'pdGroup_penalty_inertia.csv', index_col=[0, 1]).astype(float)
    p.group_penalty_non_synchronous = pd.read_csv(output_path / 'pdGroup_penalty_non_synchronous.csv', index_col=[0, 1]).astype(float)
    p.group_penalty_capacity_margin = pd.read_csv(output_path / 'pdGroup_penalty_capacity_margin.csv', index_col=[0, 1]).astype(float)
    p.group_inertia_limit = pd.read_csv(output_path / 'pdGroup_inertia_limit.csv', index_col=[0, 1]).astype(float)
    p.group_capacity_margin = pd.read_csv(output_path / 'pdGroup_capacity_margin.csv', index_col=[0, 1]).astype(float)
    p.entity_annuity = pd.read_csv(output_path / 'ed_entity_annuity.csv', index_col=[0, 1]).astype(float)
    p.entity_annual_discounted = pd.read_csv(output_path / 'ed_entity_annual_discounted.csv', index_col=[0, 1]).astype(float)
    p.entity_annual_divest_discounted = pd.read_csv(output_path / 'ed_entity_annual_divest_discounted.csv', index_col=[0, 1]).astype(float)
    p.discount_factor_operations_yearly = pd.read_csv(output_path / 'p_discount_factor_operations_yearly.csv', index_col=[0, 1])['value'].astype(float)
    p.discount_factor_investment_yearly = pd.read_csv(output_path / 'p_discount_factor_investment_yearly.csv', index_col=[0, 1])['value'].astype(float)
    p.node_capacity_for_scaling = pd.read_csv(output_path / 'node_capacity_for_scaling.csv', index_col=[0, 1]).astype(float)
    p.group_capacity_for_scaling = pd.read_csv(output_path / 'group_capacity_for_scaling.csv', index_col=[0, 1]).astype(float)
    p.complete_period_share_of_year = pd.read_csv(output_path / 'complete_period_share_of_year.csv', index_col=[0, 1])['value'].astype(float)
    p.nested_model = pd.read_csv(output_path / 'p_nested_model.csv', dtype={'param': str, 'value': float}).set_index('param')

    # Parameters with multiple row and header indexes (read_csv does not interpret these when there are multiple header rows)
    p.flow_min.index.names = ['solve', 'period', 'time']
    p.flow_max.index.names = ['solve', 'period', 'time']
    p.process_source_sink_varCost.index.names = ['solve', 'period', 'time']
    p.reserve_upDown_group_reservation.index.names = ['solve', 'period', 'time']

    # Create multi-index for data with more than one header row (will be missing when there is no data)
    if p.process_source.empty:
        p.process_source.columns = pd.MultiIndex.from_arrays([[],[]], names = list(pd.read_csv(output_path / 'p_process_source.csv', nrows=2, header=None)[0]))
    if p.process_sink.empty:
        p.process_sink.columns = pd.MultiIndex.from_arrays([[],[]], names = list(pd.read_csv(output_path / 'p_process_sink.csv', nrows=2, header=None)[0]))
    p.process_source_sink_varCost.columns = pd.MultiIndex.from_tuples(
        [(col[0], col[1], col[2]) for col in p.process_source_sink_varCost.columns],
        names=['process', 'source', 'sink']
    )
    p.flow_min.columns = pd.MultiIndex.from_tuples(
        [(col[0], col[1], col[2]) for col in p.flow_min.columns],
        names=['process', 'source', 'sink']
    )
    p.flow_max.columns = pd.MultiIndex.from_tuples(
        [(col[0], col[1], col[2]) for col in p.flow_max.columns],
        names=['process', 'source', 'sink']
    )
    p.reserve_upDown_group_reservation.columns = pd.MultiIndex.from_tuples(
        [(col[0], col[1], col[2]) for col in p.reserve_upDown_group_reservation.columns],
        names=['reserve', 'updown', 'node_group']
    )
    # These have only one row of data and are therefore interpreted as Series by pd.read_csv (using index instead of columns)
    p.process_sink_coefficient.index = pd.MultiIndex.from_tuples(
        [(col[0], col[1]) for col in p.process_sink_coefficient.index],
        names=['process', 'sink']
    )
    p.process_source_coefficient.index = pd.MultiIndex.from_tuples(
        [(col[0], col[1]) for col in p.process_source_coefficient.index],
        names=['process', 'source']
    )
    p.reserve_upDown_group_penalty.index = pd.MultiIndex.from_tuples(
        [(col[0], col[1], col[2]) for col in p.reserve_upDown_group_penalty.index],
        names=['reserve', 'updown', 'node_group']
    )

    # Create a multi-index for those that have only one index in the header rows (i.e. columns)
    p.process_slope.columns.name = 'process'
    p.process_section.columns.name = 'process'
    p.process_availability.columns.name = 'process'
    p.node_self_discharge_loss.columns.name = 'node'
    # p.years_represented_d.columns.name = 'period'
    p.node_penalty_up.columns.name = 'node'
    p.node_penalty_down.columns.name = 'node'
    p.node_inflow.columns.name = 'node'
    p.commodity_price.columns.name = 'commodity'
    p.group_co2_price.columns.name = 'group'
    p.profile.columns.name = 'profile'
    p.entity_max_units.columns.name = 'entity'
    p.entity_all_existing.columns.name = 'entity'
    p.entity_pre_existing.columns.name = 'entity'
    p.process_startup_cost.columns.name = 'process'
    p.entity_fixed_cost.columns.name = 'entity'
    p.entity_lifetime_fixed_cost.columns.name = 'entity'
    p.entity_lifetime_fixed_cost_divest.columns.name = 'entity'
    p.node_annual_flow.columns.name = 'node'
    p.group_penalty_inertia.columns.name = 'group'
    p.group_penalty_non_synchronous.columns.name = 'group'
    p.group_penalty_capacity_margin.columns.name = 'group'
    p.group_inertia_limit.columns.name = 'group'
    p.group_capacity_margin.columns.name = 'group'
    p.entity_annuity.columns.name = 'entity'
    p.entity_annual_discounted.columns.name = 'entity'
    p.entity_annual_divest_discounted.columns.name = 'entity'
    p.entity_unitsize.name = 'entity'
    p.node_capacity_for_scaling.columns.name = 'node'
    p.group_capacity_for_scaling.columns.name = 'group'
    p.node.columns.name = 'node'

    return p


def read_sets(output_dir):
    """
    Read set definitions from CSV files into a namespace.
    Simple sets are stored as pandas Index for fast O(1) membership testing.
    Tuple sets are stored as DataFrames for vectorized operations.
    
    Returns:
        SimpleNamespace: Namespace with sets as attributes
    """
    output_path = Path(output_dir)
    s = SimpleNamespace()

    # Process and entity sets    
    s.entity = pd.read_csv(output_path / 'set_entity.csv').set_index(['entity']).index
    s.entityInvest = pd.read_csv(output_path / 'set_entityInvest.csv').set_index(['entity']).index
    s.entityDivest = pd.read_csv(output_path / 'set_entityDivest.csv').set_index(['entity']).index
    s.process_online = pd.read_csv(output_path / 'set_process_online.csv').set_index(['process']).index
    s.process_online_integer = pd.read_csv(output_path / 'set_process_online_integer.csv').set_index(['process']).index
    s.process_online_linear = pd.read_csv(output_path / 'set_process_online_linear.csv').set_index(['process']).index

    # Tuple sets - store as DataFrames for vectorized filtering and operations
    
    # Tuple sets that need filtering - keep as DataFrame
    s.period = pd.read_csv(output_path / 'set_period.csv').set_index(['solve', 'period']).index
    s.d_realized_period = pd.read_csv(output_path / 'set_d_realized_period.csv').set_index(['solve', 'period']).index
    s.d_realize_invest = pd.read_csv(output_path / 'set_d_realize_invest.csv').set_index(['solve', 'period']).index
    s.dt_realize_dispatch = pd.MultiIndex.from_frame(pd.read_csv(output_path / 'set_dt_realize_dispatch.csv'))
    s.d_realize_dispatch_or_invest = pd.MultiIndex.from_frame(pd.read_csv(output_path / 'set_d_realize_dispatch_or_invest.csv'))
    s.dt = pd.MultiIndex.from_frame(pd.read_csv(output_path / 'set_dt.csv'))
    s.ed_invest = pd.MultiIndex.from_frame(pd.read_csv(output_path / 'set_ed_invest.csv'))
    s.ed_divest = pd.MultiIndex.from_frame(pd.read_csv(output_path / 'set_ed_divest.csv'))
    s.edd_invest = pd.MultiIndex.from_frame(pd.read_csv(output_path / 'set_edd_invest.csv'))
    s.process__node__profile__profile_method = pd.MultiIndex.from_frame(pd.read_csv(output_path / 'set_process__node__profile__profile_method.csv'))

    # Process topology sets
    s.process_source_sink = pd.read_csv(output_path / 'set_process_source_sink.csv').set_index(['process', 'source', 'sink']).index
    s.process_method_sources_sinks = pd.read_csv(output_path / 'set_process_method_sources_sinks.csv').set_index(['process', 'method', 'orig_source', 'orig_sink', 'always_source', 'always_sink']).index
   
    # Process method sets
    s.process_method = pd.read_csv(output_path / 'set_process_method.csv').set_index(['process', 'method']).index
    s.process__ct_method = pd.read_csv(output_path / 'set_process__ct_method.csv').set_index(['process', 'method']).index

     # Method sets
    df = pd.read_csv(output_path / 'set_method_1var_per_way.csv')
    s.method_1var_per_way = pd.Index(df.iloc[:, 0])
    
    df = pd.read_csv(output_path / 'set_method_nvar.csv')
    s.method_nvar = pd.Index(df.iloc[:, 0])

    # Time-related sets
    s.dtt = pd.MultiIndex.from_frame(pd.read_csv(output_path / 'set_dtt.csv'))
    s.dtttdt = pd.MultiIndex.from_frame(pd.read_csv(output_path / 'set_dtttdt.csv'))
    s.period__time_first = pd.MultiIndex.from_frame(pd.read_csv(output_path / 'set_period__time_first.csv'))
    s.period_first_of_solve = pd.MultiIndex.from_frame(pd.read_csv(output_path / 'set_period_first_of_solve.csv'))
    s.period_in_use = pd.MultiIndex.from_frame(pd.read_csv(output_path / 'set_period_in_use.csv'))
    s.dt_fix_storage_timesteps = pd.MultiIndex.from_frame(pd.read_csv(output_path / 'set_dt_fix_storage_timesteps.csv'))

    # Node-related sets
    df = pd.read_csv(output_path / 'set_nodeState.csv')
    s.node_state = pd.Index(df.iloc[:, 0])
    df = pd.read_csv(output_path / 'set_nodeBalance.csv')
    s.node_balance = pd.Index(df.iloc[:, 0])
    df = pd.read_csv(output_path / 'set_nodeBalancePeriod.csv')
    s.node_balance_period = pd.Index(df.iloc[:, 0])
    df = pd.read_csv(output_path / 'set_nodeSelfDischarge.csv')
    s.node_self_discharge = pd.Index(df.iloc[:, 0])
    s.node__storage_binding_method = pd.read_csv(output_path / 'set_node__storage_binding_method.csv').set_index(['node', 'method']).index
    s.node__storage_start_end_method = pd.read_csv(output_path / 'set_node__storage_start_end_method.csv').set_index(['node', 'method']).index
    s.node__inflow_method = pd.read_csv(output_path / 'set_node__inflow_method.csv').set_index(['node', 'method']).index
    s.node__storage_nested_fix_method = pd.read_csv(output_path / 'set_node__storage_nested_fix_method.csv').set_index(['node', 'method']).index

    # Process-related sets
    s.process = pd.read_csv(output_path / 'set_process.csv').set_index(['process']).index
    s.node = pd.read_csv(output_path / 'set_node.csv').set_index(['node']).index
    s.process_source = pd.read_csv(output_path / 'set_process_source.csv').set_index(['process', 'source']).index
    s.process_sink = pd.read_csv(output_path / 'set_process_sink.csv').set_index(['process', 'sink']).index
    s.process_VRE = pd.read_csv(output_path / 'set_process_VRE.csv').set_index(['process', 'node']).index
    s.process__source__sink__profile__profile_method = pd.read_csv(output_path / 'set_process__source__sink__profile__profile_method.csv')

    # Process type sets
    s.process_unit = pd.read_csv(output_path / 'set_process_unit.csv').set_index(['process']).index
    s.process_connection = pd.read_csv(output_path / 'set_process_connection.csv').set_index(['process']).index
    s.process_profile = pd.read_csv(output_path / 'set_process_profile.csv').set_index(['process']).index

    # Commodity-related sets
    s.commodity_node = pd.read_csv(output_path / 'set_commodity_node.csv').set_index(['commodity', 'node']).index
    s.commodity_node_co2 = pd.read_csv(output_path / 'set_commodity_node_co2.csv').set_index(['commodity', 'node']).index
    s.process__commodity__node = pd.read_csv(output_path / 'set_process__commodity__node.csv').set_index(['process', 'commodity', 'node']).index
    s.process__commodity__node_co2 = pd.read_csv(output_path / 'set_process__commodity__node_co2.csv').set_index(['process', 'commodity', 'node']).index
    s.group_co2_price = pd.read_csv(output_path / 'set_group_co2_price.csv').set_index(['group']).index
    s.group_co2_limit = pd.read_csv(output_path / 'set_group_co2_limit.csv').set_index(['group']).index

    # Group-related sets
    df = pd.read_csv(output_path / 'set_groupInertia.csv')
    s.groupInertia = pd.Index(df.iloc[:, 0])
    df = pd.read_csv(output_path / 'set_groupNonSync.csv')
    s.groupNonSync = pd.Index(df.iloc[:, 0])
    df = pd.read_csv(output_path / 'set_groupCapacityMargin.csv')
    s.groupCapacityMargin = pd.Index(df.iloc[:, 0])
    df = pd.read_csv(output_path / 'set_groupOutputNodeFlows.csv')
    s.outputNodeGroup_does_generic_flows = pd.Index(df.iloc[:, 0])
    df = pd.read_csv(output_path / 'set_groupOutput_node.csv')
    s.outputNodeGroup_does_specified_flows_node = pd.Index(df.iloc[:, 0])
    df = pd.read_csv(output_path / 'set_groupOutput_process.csv')
    s.outputNodeGroup_does_specified_flows_process = pd.Index(df.iloc[:, 0])
    df = pd.read_csv(output_path / 'set_groupOutput.csv')
    s.outputNodeGroup_does_specified_flows = pd.Index(df.iloc[:, 0])
    s.outputNodeGroup__connection_Not_in_aggregate = pd.read_csv(output_path / 'set_group_output__connection_Not_in_aggregate.csv').set_index(['group', 'connection']).index
    s.outputNodeGroup__process__unit__to_node_Not_in_aggregate = pd.read_csv(output_path / 'set_group_output__process__unit__to_node_Not_in_aggregate.csv').set_index(['group', 'process', 'unit', 'node']).index
    s.outputNodeGroup__process__node__to_unit_Not_in_aggregate = pd.read_csv(output_path / 'set_group_output__process__node__to_unit_Not_in_aggregate.csv').set_index(['group', 'process', 'node', 'unit']).index
    s.outputNodeGroup__process__connection__to_node_Not_in_aggregate = pd.read_csv(output_path / 'set_group_output__process__connection__to_node_Not_in_aggregate.csv').set_index(['group', 'process', 'connection', 'node']).index
    s.outputNodeGroup__process__node__to_connection_Not_in_aggregate = pd.read_csv(output_path / 'set_group_output__process__node__to_connection_Not_in_aggregate.csv').set_index(['group', 'process', 'node', 'connection']).index
    s.outputNodeGroup__processGroup_Unit_to_group = pd.read_csv(output_path / 'set_group_output__group_aggregate_Unit_to_group.csv').set_index(['group', 'group_aggregate']).index
    s.outputNodeGroup__processGroup__process__unit__to_node = pd.read_csv(output_path / 'set_group_output__group_aggregate__process__unit__to_node.csv').set_index(['group', 'group_aggregate', 'unit', 'source', 'sink']).index
    s.outputNodeGroup__processGroup__process__unit__to_node.names = ['group', 'group_aggregate', 'process', 'unit', 'node']
    s.outputNodeGroup__processGroup_Group_to_unit = pd.read_csv(output_path / 'set_group_output__group_aggregate_Group_to_unit.csv').set_index(['group', 'group_aggregate']).index
    s.outputNodeGroup__processGroup__process__node__to_unit = pd.read_csv(output_path / 'set_group_output__group_aggregate__process__node__to_unit.csv').set_index(['group', 'group_aggregate', 'unit', 'source', 'sink']).index
    s.outputNodeGroup__processGroup__process__node__to_unit.names = ['group', 'group_aggregate', 'process', 'node', 'unit']
    s.outputNodeGroup__processGroup_Connection = pd.read_csv(output_path / 'set_group_output__group_aggregate_Connection.csv').set_index(['group', 'group_aggregate']).index
    s.outputNodeGroup__processGroup__process__connection__to_node = pd.read_csv(output_path / 'set_group_output__group_aggregate__process__connection__to_node.csv').set_index(['group', 'group_aggregate', 'connection', 'source', 'sink']).index
    s.outputNodeGroup__processGroup__process__connection__to_node.names = ['group', 'group_aggregate', 'process', 'connection', 'node']
    s.outputNodeGroup__processGroup__process__node__to_connection = pd.read_csv(output_path / 'set_group_output__group_aggregate__process__node__to_connection.csv').set_index(['group', 'group_aggregate', 'connection', 'source', 'sink']).index
    s.outputNodeGroup__processGroup__process__node__to_connection.names = ['group', 'group_aggregate', 'process', 'node', 'connection']
    s.outputNodeGroup__process_fully_inside = pd.read_csv(output_path / 'set_group_output__process_fully_inside.csv').set_index(['group', 'process']).index
    s.group_node = pd.read_csv(output_path / 'set_group_node.csv').set_index(['group', 'node']).index
    s.group_process = pd.read_csv(output_path / 'set_group_process.csv').set_index(['group', 'process']).index
    s.group_process_node = pd.read_csv(output_path / 'set_group_process_node.csv').set_index(['group', 'process', 'node']).index

    # upDown set
    df = pd.read_csv(output_path / 'set_upDown.csv')
    s.upDown = pd.Index(df.iloc[:, 0])

    # Optional output flags
    df = pd.read_csv(output_path / 'set_enable_optional_outputs.csv')
    s.enable_optional_outputs = set(df.iloc[:, 0])

    return s