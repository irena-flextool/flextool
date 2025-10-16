from types import SimpleNamespace
from pathlib import Path
import pandas as pd

def read_variables(output_dir='output_raw'):
    """
    Read all variable CSV files into a namespace.
    
    Returns:
        SimpleNamespace: Namespace with variables as attributes
    """
    output_path = Path(output_dir)
    v = SimpleNamespace()
    
    # Variables with (solve, period, time) index
    
    v.flow = pd.read_csv(output_path / 'v_flow.csv', header=[0, 1, 2], index_col=[0, 1, 2])
    v.ramp = pd.read_csv(output_path / 'v_ramp.csv', header=[0, 1, 2], index_col=[0, 1, 2])
    v.reserve = pd.read_csv(output_path / 'v_reserve.csv', header=[0, 1, 2, 3], index_col=[0, 1, 2])
    v.state = pd.read_csv(output_path / 'v_state.csv', index_col=[0, 1, 2])
    v.online_linear = pd.read_csv(output_path / 'v_online_linear.csv', index_col=[0, 1, 2])
    v.startup_linear = pd.read_csv(output_path / 'v_startup_linear.csv', index_col=[0, 1, 2])
    v.shutdown_linear = pd.read_csv(output_path / 'v_shutdown_linear.csv', index_col=[0, 1, 2])
    v.online_integer = pd.read_csv(output_path / 'v_online_integer.csv', index_col=[0, 1, 2])
    v.startup_integer = pd.read_csv(output_path / 'v_startup_integer.csv', index_col=[0, 1, 2])
    v.shutdown_integer = pd.read_csv(output_path / 'v_shutdown_integer.csv', index_col=[0, 1, 2])
    v.q_state_up = pd.read_csv(output_path / 'vq_state_up.csv', index_col=[0, 1, 2])
    v.q_state_down = pd.read_csv(output_path / 'vq_state_down.csv', index_col=[0, 1, 2])
    v.q_reserve = pd.read_csv(output_path / 'vq_reserve.csv', header=[0, 1, 2], index_col=[0, 1, 2])
    v.q_inertia = pd.read_csv(output_path / 'vq_inertia.csv', index_col=[0, 1, 2])
    v.q_non_synchronous = pd.read_csv(output_path / 'vq_non_synchronous.csv', index_col=[0, 1, 2])
    v.q_state_up_group = pd.read_csv(output_path / 'vq_state_up_group.csv', index_col=[0, 1, 2])
    v.q_capacity_margin = pd.read_csv(output_path / 'vq_capacity_margin.csv', index_col=[0, 1])
    v.invest = pd.read_csv(output_path / 'v_invest.csv', index_col=[0, 1], header=0)
    v.divest = pd.read_csv(output_path / 'v_divest.csv', index_col=[0, 1])

    # Create multi-index for variables with single header row
    #v.flow.columns = pd.MultiIndex.from_product([['process', 'source', 'sink'], v.flow.columns])
    #v.ramp.columns = pd.MultiIndex.from_product([['process', 'source', 'sink'], v.ramp.columns])
    #v.reserve.columns = pd.MultiIndex.from_product([['process', 'reserve', 'updown', 'node'], v.reserve.columns])
    v.state.columns = pd.MultiIndex.from_product([['node'], v.state.columns])
    v.online_linear.columns = pd.MultiIndex.from_product([['process'], v.online_linear.columns])
    v.startup_linear.columns = pd.MultiIndex.from_product([['process'], v.startup_linear.columns])
    v.shutdown_linear.columns = pd.MultiIndex.from_product([['process'], v.shutdown_linear.columns])
    v.online_integer.columns = pd.MultiIndex.from_product([['process'], v.online_integer.columns])
    v.startup_integer.columns = pd.MultiIndex.from_product([['process'], v.startup_integer.columns])
    v.shutdown_integer.columns = pd.MultiIndex.from_product([['process'], v.shutdown_integer.columns])
    v.q_state_up.columns = pd.MultiIndex.from_product([['node'], v.q_state_up.columns])
    v.q_state_down.columns = pd.MultiIndex.from_product([['node'], v.q_state_down.columns])
    #v.q_reserve.columns = pd.MultiIndex.from_product([['reserve', 'updown', 'node_group'], v.q_reserve.columns])
    v.q_inertia.columns = pd.MultiIndex.from_product([['group'], v.q_inertia.columns])
    v.q_non_synchronous.columns = pd.MultiIndex.from_product([['group'], v.q_non_synchronous.columns])
    v.q_state_up_group.columns = pd.MultiIndex.from_product([['group'], v.q_state_up_group.columns])
    v.q_capacity_margin.columns = pd.MultiIndex.from_product([['group'], v.q_capacity_margin.columns])
    v.invest.columns = pd.MultiIndex.from_product([['entity'], v.invest.columns])
    v.divest.columns = pd.MultiIndex.from_product([['entity'], v.divest.columns])

    # Add multi-index to variables with multiple header rows
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


    return v


def read_parameters(output_dir='output_raw'):
    """
    Read all parameter CSV files into a namespace.
    
    Returns:
        SimpleNamespace: Namespace with parameters as attributes
    """
    output_path = Path(output_dir)
    p = SimpleNamespace()
    
    # Parameters with (period, time) index and multi-dimensional columns
    p.step_duration = pd.read_csv(output_path / 'p_step_duration.csv', dtype={'value': float}, index_col=[0, 1, 2])
    p.flow_min = pd.read_csv(output_path / 'p_flow_min.csv', dtype={'value': float}, header=[0, 1, 2], index_col=[0, 1, 2])
    p.flow_max = pd.read_csv(output_path / 'p_flow_max.csv', dtype={'value': float}, header=[0, 1, 2], index_col=[0, 1, 2])
    p.process_slope = pd.read_csv(output_path / 'pdtProcess_slope.csv', dtype={'value': float}, index_col=[0, 1, 2])
    p.process_section = pd.read_csv(output_path / 'pdtProcess_section.csv', dtype={'value': float}, index_col=[0, 1, 2])
    p.process_availability = pd.read_csv(output_path / 'pdtProcess_availability.csv', dtype={'value': float}, index_col=[0, 1, 2])
    p.process_source_sink_varCost = pd.read_csv(output_path / 'pdtProcess_source_sink_varCost.csv', dtype={'value': float}, header=[0, 1, 2], index_col=[0, 1, 2])
    p.node_self_discharge_loss = pd.read_csv(output_path / 'pdtNode_self_discharge_loss.csv', dtype={'value': float}, index_col=[0, 1, 2])
    p.node_penalty_up = pd.read_csv(output_path / 'pdtNode_penalty_up.csv', dtype={'value': float}, index_col=[0, 1, 2])
    p.node_penalty_down = pd.read_csv(output_path / 'pdtNode_penalty_down.csv', dtype={'value': float}, index_col=[0, 1, 2])
    p.node_inflow = pd.read_csv(output_path / 'pdtNodeInflow.csv', dtype={'value': float}, index_col=[0, 1, 2])
    p.commodity_price = pd.read_csv(output_path / 'pdtCommodity_price.csv', dtype={'value': float}, index_col=[0, 1, 2])
    p.group_co2_price = pd.read_csv(output_path / 'pdtGroup_co2_price.csv', dtype={'value': float}, index_col=[0, 1, 2])
    p.reserve_upDown_group_reservation = pd.read_csv(output_path / 'pdtReserve_upDown_group_reservation.csv', dtype={'value': float}, header=[0, 1, 2], index_col=[0, 1, 2])
    p.profile = pd.read_csv(output_path / 'pdtProfile.csv', dtype={'value': float}, index_col=[0, 1, 2])
    p.years_d = pd.read_csv(output_path / 'p_years_d.csv', dtype={'value': float}, index_col=[0, 1])
    p.entity_max_units = pd.read_csv(output_path / 'p_entity_max_units.csv', dtype={'value': float}, index_col=[0, 1])
    p.entity_all_existing = pd.read_csv(output_path / 'p_entity_all_existing.csv', dtype={'value': float}, index_col=[0, 1])
    p.entity_unitsize = pd.read_csv(output_path / 'p_entity_unitsize.csv', dtype={'value': float}, index_col=0)
    p.process_startup_cost = pd.read_csv(output_path / 'pdProcess_startup_cost.csv', dtype={'value': float}, index_col=[0, 1])
    p.process_fixed_cost = pd.read_csv(output_path / 'pdProcess_fixed_cost.csv', dtype={'value': float}, index_col=[0, 1])
    p.node_fixed_cost = pd.read_csv(output_path / 'pdNode_fixed_cost.csv', dtype={'value': float}, index_col=[0, 1])
    p.node_annual_flow = pd.read_csv(output_path / 'pdNode_annual_flow.csv', dtype={'value': float}, index_col=[0, 1])
    p.group_penalty_inertia = pd.read_csv(output_path / 'pdGroup_penalty_inertia.csv', dtype={'value': float}, index_col=[0, 1])
    p.group_penalty_non_synchronous = pd.read_csv(output_path / 'pdGroup_penalty_non_synchronous.csv', dtype={'value': float}, index_col=[0, 1])
    p.group_penalty_capacity_margin = pd.read_csv(output_path / 'pdGroup_penalty_capacity_margin.csv', dtype={'value': float}, index_col=[0, 1])
    p.group_inertia_limit = pd.read_csv(output_path / 'pdGroup_inertia_limit.csv', dtype={'value': float}, index_col=[0, 1])
    p.group_capacity_margin = pd.read_csv(output_path / 'pdGroup_capacity_margin.csv', dtype={'value': float}, index_col=[0, 1])
    p.entity_annual_discounted = pd.read_csv(output_path / 'ed_entity_annual_discounted.csv', dtype={'value': float}, index_col=[0, 1])
    p.entity_annual_divest_discounted = pd.read_csv(output_path / 'ed_entity_annual_divest_discounted.csv', dtype={'value': float}, index_col=[0, 1])
    p.discount_factor_operations_yearly = pd.read_csv(output_path / 'p_discount_factor_operations_yearly.csv', dtype={'value': float}, index_col=[0, 1])
    p.discount_factor_investment_yearly = pd.read_csv(output_path / 'p_discount_factor_investment_yearly.csv', dtype={'value': float}, index_col=[0, 1])
    p.node_capacity_for_scaling = pd.read_csv(output_path / 'node_capacity_for_scaling.csv', dtype={'value': float}, index_col=[0, 1])
    p.group_capacity_for_scaling = pd.read_csv(output_path / 'group_capacity_for_scaling.csv', dtype={'value': float}, index_col=[0, 1])
    p.complete_period_share_of_year = pd.read_csv(output_path / 'complete_period_share_of_year.csv', dtype={'value': float}, index_col=[0, 1])
    p.node = pd.read_csv(output_path / 'p_node.csv', dtype={'value': float}, index_col=0)
    p.process_sink_coefficient = pd.read_csv(output_path / 'p_process_sink_coefficient.csv', dtype={'value': float}, header=[0, 1], index_col=0)
    p.process_source_coefficient = pd.read_csv(output_path / 'p_process_source_coefficient.csv', dtype={'value': float}, header=[0, 1], index_col=0)
    p.commodity_co2_content = pd.read_csv(output_path / 'p_commodity_co2_content.csv', dtype={'value': float}, index_col=0)
    p.reserve_upDown_group_penalty = pd.read_csv(output_path / 'p_reserve_upDown_group_penalty.csv', dtype={'value': float}, header=[0, 1, 2], index_col=0)
    p.nested_model = pd.read_csv(output_path / 'p_nested_model.csv', dtype={'param': str, 'value': float}).set_index('param')
    p.roll_continue_state = pd.read_csv('solve_data/p_roll_continue_state.csv', dtype={'node': str, 'value': float}).set_index('node')

    # Create multi-index for data with more than one header row
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
    p.process_sink_coefficient.columns = pd.MultiIndex.from_tuples(
        [(col[0], col[1]) for col in p.process_sink_coefficient.columns],
        names=['process', 'sink']
    )
    p.process_source_coefficient.columns = pd.MultiIndex.from_tuples(
        [(col[0], col[1]) for col in p.process_source_coefficient.columns],
        names=['process', 'source']
    )
    p.reserve_upDown_group_penalty.columns = pd.MultiIndex.from_tuples(
        [(col[0], col[1], col[2]) for col in p.reserve_upDown_group_penalty.columns],
        names=['reserve', 'updown', 'node_group']
    )

    # Create a multi-index for those that have only one index in the header rows (i.e. columns)
    p.process_slope.columns = pd.MultiIndex.from_product([['process'], p.process_slope.columns])
    p.process_section.columns = pd.MultiIndex.from_product([['process'], p.process_section.columns])
    p.process_availability.columns = pd.MultiIndex.from_product([['process'], p.process_availability.columns])
    p.node_self_discharge_loss.columns = pd.MultiIndex.from_product([['node'], p.node_self_discharge_loss.columns])
    p.node_penalty_up.columns = pd.MultiIndex.from_product([['node'], p.node_penalty_up.columns])
    p.node_penalty_down.columns = pd.MultiIndex.from_product([['node'], p.node_penalty_down.columns])
    p.node_inflow.columns = pd.MultiIndex.from_product([['node'], p.node_inflow.columns])
    p.commodity_price.columns = pd.MultiIndex.from_product([['commodity'], p.commodity_price.columns])
    p.group_co2_price.columns = pd.MultiIndex.from_product([['group'], p.group_co2_price.columns])
    p.profile.columns = pd.MultiIndex.from_product([['profile'], p.profile.columns])
    p.entity_max_units.columns = pd.MultiIndex.from_product([['entity'], p.entity_max_units.columns])
    p.entity_all_existing.columns = pd.MultiIndex.from_product([['entity'], p.entity_all_existing.columns])
    p.entity_unitsize.columns = pd.MultiIndex.from_product([['entity'], p.entity_unitsize.columns])
    p.process_startup_cost.columns = pd.MultiIndex.from_product([['process'], p.process_startup_cost.columns])
    p.process_fixed_cost.columns = pd.MultiIndex.from_product([['process'], p.process_fixed_cost.columns])
    p.node_fixed_cost.columns = pd.MultiIndex.from_product([['node'], p.node_fixed_cost.columns])
    p.node_annual_flow.columns = pd.MultiIndex.from_product([['node'], p.node_annual_flow.columns])
    p.group_penalty_inertia.columns = pd.MultiIndex.from_product([['group'], p.group_penalty_inertia.columns])
    p.group_penalty_non_synchronous.columns = pd.MultiIndex.from_product([['group'], p.group_penalty_non_synchronous.columns])
    p.group_penalty_capacity_margin.columns = pd.MultiIndex.from_product([['group'], p.group_penalty_capacity_margin.columns])
    p.group_inertia_limit.columns = pd.MultiIndex.from_product([['group'], p.group_inertia_limit.columns])
    p.group_capacity_margin.columns = pd.MultiIndex.from_product([['group'], p.group_capacity_margin.columns])
    p.entity_annual_discounted.columns = pd.MultiIndex.from_product([['entity'], p.entity_annual_discounted.columns])
    p.entity_annual_divest_discounted.columns = pd.MultiIndex.from_product([['entity'], p.entity_annual_divest_discounted.columns])
    p.node_capacity_for_scaling.columns = pd.MultiIndex.from_product([['node'], p.node_capacity_for_scaling.columns])
    p.group_capacity_for_scaling.columns = pd.MultiIndex.from_product([['group'], p.group_capacity_for_scaling.columns])
    p.node.columns = pd.MultiIndex.from_product([['node'], p.node.columns])
    p.commodity_co2_content.columns = pd.MultiIndex.from_product([['commodity'], p.commodity_co2_content.columns])

    return p


def read_sets(output_dir='output_raw'):
    """
    Read set definitions from CSV files into a namespace.
    Simple sets are stored as pandas Index for fast O(1) membership testing.
    Tuple sets are stored as DataFrames for vectorized operations.
    
    Returns:
        SimpleNamespace: Namespace with sets as attributes
    """
    output_path = Path(output_dir)
    s = SimpleNamespace()
    
    # Simple sets (single column) - store as pandas Index
    # Index maintains order and provides O(1) membership testing
    simple_set_mapping = [
        ('d_realized_period', 'd_realized_period'),
        ('entity', 'entity'),
        ('period', 'period'),
        ('entityInvest', 'entity_invest'),
        ('entityDivest', 'entity_divest'),
        ('period_invest', 'period_invest'),
        ('process_online', 'process_online'),
        ('process_online_linear', 'process_online_linear'),
        ('process_online_integer', 'process_online_integer')
    ]
    
    for file_name, attr_name in simple_set_mapping:
        df = pd.read_csv(output_path / f'set_{file_name}.csv')
        setattr(s, attr_name, pd.Index(df.iloc[:, 0]))
    
    # Tuple sets - store as DataFrames for vectorized filtering and operations
    
    # Tuple sets that need filtering - keep as DataFrame
    s.dt_realize_dispatch = pd.MultiIndex.from_frame(pd.read_csv(output_path / 'set_dt_realize_dispatch.csv'))
    s.dt = pd.MultiIndex.from_frame(pd.read_csv(output_path / 'set_dt.csv'))
    s.ed_invest = pd.MultiIndex.from_frame(pd.read_csv(output_path / 'set_ed_invest.csv'))
    s.ed_divest = pd.MultiIndex.from_frame(pd.read_csv(output_path / 'set_ed_divest.csv'))
    s.edd_invest = pd.MultiIndex.from_frame(pd.read_csv(output_path / 'set_edd_invest.csv'))

    # Process topology sets
    s.process_method_sources_sinks = pd.read_csv(output_path / 'set_process_method_sources_sinks.csv')
    # s.process_source_sink_alwaysProcess = pd.read_csv(output_path / 'set_process_source_sink_alwaysProcess.csv') 
    # s.process_source_toSink = pd.read_csv(output_path / 'set_process_source_toSink.csv').set_index(['process', 'source', 'sink']).index
    # s.process_sink_toSource = pd.read_csv(output_path / 'set_process_sink_toSource.csv').set_index(['process', 'source', 'sink']).index
    # s.process__profileProcess__toSink = pd.read_csv(output_path / 'set_process__profileProcess__toSink.csv').set_index(['process', 'source', 'sink']).index
    # s.process__source__toProfileProcess = pd.read_csv(output_path / 'set_process__source__toProfileProcess.csv').set_index(['process', 'source', 'sink']).index
    # s.process_process_toSink = pd.read_csv(output_path / 'set_process_process_toSink.csv').set_index(['process', 'source', 'sink']).index
    # s.process_source_toProcess = pd.read_csv(output_path / 'set_process_source_toProcess.csv').set_index(['process', 'source', 'sink']).index
    # s.process_source_toProcess_direct = pd.read_csv(output_path / 'set_process_source_toProcess_direct.csv').set_index(['process', 'source', 'sink']).index
    # s.process_process_toSink_direct = pd.read_csv(output_path / 'set_process_process_toSink_direct.csv').set_index(['process', 'source', 'sink']).index
   
    # Process method sets
    s.process_method = pd.read_csv(output_path / 'set_process_method.csv').set_index(['process', 'method']).index
    s.process__ct_method = pd.read_csv(output_path / 'set_process__ct_method.csv').set_index(['process', 'method']).index

    # Process type sets
    df = pd.read_csv(output_path / 'set_process_unit.csv')
    s.process_unit = pd.Index(df.iloc[:, 0])
    df = pd.read_csv(output_path / 'set_process_connection.csv')
    s.process_connection = pd.Index(df.iloc[:, 0])
    df = pd.read_csv(output_path / 'set_process_profile.csv')
    s.process_profile = pd.Index(df.iloc[:, 0])

     # Method sets
    df = pd.read_csv(output_path / 'set_method_1var_per_way.csv')
    s.method_1var_per_way = pd.Index(df.iloc[:, 0])
    
    df = pd.read_csv(output_path / 'set_method_nvar.csv')
    s.method_nvar = pd.Index(df.iloc[:, 0])

    # Time-related sets
    s.dtt = pd.read_csv(output_path / 'set_dtt.csv')
    s.dtttdt = pd.read_csv(output_path / 'set_dtttdt.csv')
    s.period__time_first = pd.read_csv(output_path / 'set_period__time_first.csv')
    df = pd.read_csv(output_path / 'set_period_first_of_solve.csv')
    s.period_first_of_solve = pd.Index(df.iloc[:, 0])
    df = pd.read_csv(output_path / 'set_period_in_use.csv')
    s.period_in_use = pd.Index(df.iloc[:, 0])
    s.dt_fix_storage_timesteps = pd.read_csv(output_path / 'set_dt_fix_storage_timesteps.csv')

    # Node-related sets
    df = pd.read_csv(output_path / 'set_nodeState.csv')
    s.nodeState = pd.Index(df.iloc[:, 0])
    df = pd.read_csv(output_path / 'set_nodeBalance.csv')
    s.nodeBalance = pd.Index(df.iloc[:, 0])
    df = pd.read_csv(output_path / 'set_nodeBalancePeriod.csv')
    s.nodeBalancePeriod = pd.Index(df.iloc[:, 0])
    df = pd.read_csv(output_path / 'set_nodeSelfDischarge.csv')
    s.nodeSelfDischarge = pd.Index(df.iloc[:, 0])
    s.node__storage_binding_method = pd.read_csv(output_path / 'set_node__storage_binding_method.csv').set_index(['node', 'method']).index
    s.node__storage_start_end_method = pd.read_csv(output_path / 'set_node__storage_start_end_method.csv').set_index(['node', 'method']).index
    s.node__inflow_method = pd.read_csv(output_path / 'set_node__inflow_method.csv').set_index(['node', 'method']).index
    s.node__storage_nested_fix_method = pd.read_csv(output_path / 'set_node__storage_nested_fix_method.csv').set_index(['node', 'method']).index

    # Process-related sets
    df = pd.read_csv(output_path / 'set_process.csv')
    s.process = pd.Index(df.iloc[:, 0])
    df = pd.read_csv(output_path / 'set_node.csv')
    s.node = pd.Index(df.iloc[:, 0])
    df = pd.read_csv(output_path / 'set_process_connection.csv')
    s.process_connection = pd.Index(df.iloc[:, 0])
    s.process_source = pd.read_csv(output_path / 'set_process_source.csv').set_index(['process', 'source']).index
    s.process_sink = pd.read_csv(output_path / 'set_process_sink.csv').set_index(['process', 'sink']).index
    s.process_VRE = pd.read_csv(output_path / 'set_process_VRE.csv')
    s.process__source__sink__profile__profile_method = pd.read_csv(output_path / 'set_process__source__sink__profile__profile_method.csv')

    # Commodity-related sets
    s.commodity_node = pd.read_csv(output_path / 'set_commodity_node.csv').set_index(['commodity', 'node']).index
    s.commodity_node_co2 = pd.read_csv(output_path / 'set_commodity_node_co2.csv').set_index(['commodity', 'node']).index
    s.process__commodity__node = pd.read_csv(output_path / 'set_process__commodity__node.csv')
    s.process__commodity__node_co2 = pd.read_csv(output_path / 'set_process__commodity__node_co2.csv')
    s.gcndt_co2_price = pd.read_csv(output_path / 'set_gcndt_co2_price.csv')

    # Group-related sets
    df = pd.read_csv(output_path / 'set_groupInertia.csv')
    s.groupInertia = pd.Index(df.iloc[:, 0])
    df = pd.read_csv(output_path / 'set_groupNonSync.csv')
    s.groupNonSync = pd.Index(df.iloc[:, 0])
    df = pd.read_csv(output_path / 'set_groupCapacityMargin.csv')
    s.groupCapacityMargin = pd.Index(df.iloc[:, 0])
    df = pd.read_csv(output_path / 'set_groupOutputNodeFlows.csv')
    s.groupOutputNodeFlows = pd.Index(df.iloc[:, 0])
    s.group_output__connection_Not_in_aggregate = pd.read_csv(output_path / 'set_group_output__connection_Not_in_aggregate.csv')
    s.group_output__process__connection__to_node_Not_in_aggregate = pd.read_csv(output_path / 'set_group_output__process__connection__to_node_Not_in_aggregate.csv')
    s.group_output__process__node__to_connection_Not_in_aggregate = pd.read_csv(output_path / 'set_group_output__process__node__to_connection_Not_in_aggregate.csv')
    s.group_output__group_aggregate_Unit_to_group = pd.read_csv(output_path / 'set_group_output__group_aggregate_Unit_to_group.csv')
    s.group_output__group_aggregate__process__unit__to_node = pd.read_csv(output_path / 'set_group_output__group_aggregate__process__unit__to_node.csv')
    s.group_output__group_aggregate_Group_to_unit = pd.read_csv(output_path / 'set_group_output__group_aggregate_Group_to_unit.csv')
    s.group_output__group_aggregate__process__node__to_unit = pd.read_csv(output_path / 'set_group_output__group_aggregate__process__node__to_unit.csv')
    s.group_output__group_aggregate_Connection = pd.read_csv(output_path / 'set_group_output__group_aggregate_Connection.csv')
    s.group_output__group_aggregate__process__connection__to_node = pd.read_csv(output_path / 'set_group_output__group_aggregate__process__connection__to_node.csv')
    s.group_output__group_aggregate__process__node__to_connection = pd.read_csv(output_path / 'set_group_output__group_aggregate__process__node__to_connection.csv')
    s.group_output__process_fully_inside = pd.read_csv(output_path / 'set_group_output__process_fully_inside.csv').set_index(['group', 'process']).index
    s.group_node = pd.read_csv(output_path / 'set_group_node.csv').set_index(['group', 'node']).index

    # upDown set
    df = pd.read_csv(output_path / 'set_upDown.csv')
    s.upDown = pd.Index(df.iloc[:, 0])

    # Optional output flags
    df = pd.read_csv(output_path / 'set_enable_optional_outputs.csv')
    s.enable_optional_outputs = set(df.iloc[:, 0])

    return s