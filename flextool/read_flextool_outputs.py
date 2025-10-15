from types import SimpleNamespace
from pathlib import Path
import pandas as pd

def read_variables(output_dir='output'):
    """
    Read all variable CSV files into a namespace.
    
    Returns:
        SimpleNamespace: Namespace with variables as attributes
    """
    output_path = Path(output_dir)
    v = SimpleNamespace()
    
    # Variables with (period, time) index and multi-dimensional columns
    
    # v_flow: 3-level column MultiIndex (process, source, sink)
    df = pd.read_csv(output_path / 'v_flow_pivot.csv', header=[0, 1, 2], index_col=[0, 1])
    df.columns = pd.MultiIndex.from_tuples(
        [(col[0], col[1], col[2]) for col in df.columns],
        names=['process', 'source', 'sink']
    )
    df.index.names = ['period', 'time']
    v.flow = df
    
    # v_ramp: 3-level column MultiIndex (process, source, sink)
    df = pd.read_csv(output_path / 'v_ramp_pivot.csv', header=[0, 1, 2], index_col=[0, 1])
    df.columns = pd.MultiIndex.from_tuples(
        [(col[0], col[1], col[2]) for col in df.columns],
        names=['process', 'source', 'sink']
    )
    df.index.names = ['period', 'time']
    v.ramp = df
    
    # v_reserve: 4-level column MultiIndex (process, reserve, updown, node)
    df = pd.read_csv(output_path / 'v_reserve_pivot.csv', header=[0, 1, 2, 3], index_col=[0, 1])
    df.columns = pd.MultiIndex.from_tuples(
        [(col[0], col[1], col[2], col[3]) for col in df.columns],
        names=['process', 'reserve', 'updown', 'node']
    )
    df.index.names = ['period', 'time']
    v.reserve = df
    
    # v_state: 1-level column index (node)
    df = pd.read_csv(output_path / 'v_state_pivot.csv', index_col=[0, 1])
    df.index.names = ['period', 'time']
    v.state = df
    
    # v_online_linear: 1-level column index (process)
    df = pd.read_csv(output_path / 'v_online_linear_pivot.csv', index_col=[0, 1])
    df.index.names = ['period', 'time']
    v.online_linear = df
    
    # v_startup_linear: 1-level column index (process)
    df = pd.read_csv(output_path / 'v_startup_linear_pivot.csv', index_col=[0, 1])
    df.index.names = ['period', 'time']
    v.startup_linear = df
    
    # v_shutdown_linear: 1-level column index (process)
    df = pd.read_csv(output_path / 'v_shutdown_linear_pivot.csv', index_col=[0, 1])
    df.index.names = ['period', 'time']
    v.shutdown_linear = df
    
    # v_online_integer: 1-level column index (process)
    df = pd.read_csv(output_path / 'v_online_integer_pivot.csv', index_col=[0, 1])
    df.index.names = ['period', 'time']
    v.online_integer = df
    
    # v_startup_integer: 1-level column index (process)
    df = pd.read_csv(output_path / 'v_startup_integer_pivot.csv', index_col=[0, 1])
    df.index.names = ['period', 'time']
    v.startup_integer = df
    
    # v_shutdown_integer: 1-level column index (process)
    df = pd.read_csv(output_path / 'v_shutdown_integer_pivot.csv', index_col=[0, 1])
    df.index.names = ['period', 'time']
    v.shutdown_integer = df
    
    # vq_state_up: 1-level column index (node)
    df = pd.read_csv(output_path / 'vq_state_up_pivot.csv', index_col=[0, 1])
    df.index.names = ['period', 'time']
    v.q_state_up = df
    
    # vq_state_down: 1-level column index (node)
    df = pd.read_csv(output_path / 'vq_state_down_pivot.csv', index_col=[0, 1])
    df.index.names = ['period', 'time']
    v.q_state_down = df
    
    # vq_reserve: 3-level column MultiIndex (reserve, updown, node_group)
    df = pd.read_csv(output_path / 'vq_reserve_pivot.csv', header=[0, 1, 2], index_col=[0, 1])
    df.columns = pd.MultiIndex.from_tuples(
        [(col[0], col[1], col[2]) for col in df.columns],
        names=['reserve', 'updown', 'node_group']
    )
    df.index.names = ['period', 'time']
    v.q_reserve = df
    
    # vq_inertia: 1-level column index (group)
    df = pd.read_csv(output_path / 'vq_inertia_pivot.csv', index_col=[0, 1])
    df.index.names = ['period', 'time']
    v.q_inertia = df
    
    # vq_non_synchronous: 1-level column index (group)
    df = pd.read_csv(output_path / 'vq_non_synchronous_pivot.csv', index_col=[0, 1])
    df.index.names = ['period', 'time']
    v.q_non_synchronous = df
    
    # vq_state_up_group: 1-level column index (group)
    df = pd.read_csv(output_path / 'vq_state_up_group_pivot.csv', index_col=[0, 1])
    df.index.names = ['period', 'time']
    v.q_state_up_group = df
    
    # Variables with only period index
    
    # v_invest: 1-level column index (entity)
    df = pd.read_csv(output_path / 'v_invest_pivot.csv', index_col=0)
    df.index.name = 'period'
    v.invest = df
    
    # v_divest: 1-level column index (entity)
    df = pd.read_csv(output_path / 'v_divest_pivot.csv', index_col=0)
    df.index.name = 'period'
    v.divest = df
    
    # vq_capacity_margin: 1-level column index (group)
    df = pd.read_csv(output_path / 'vq_capacity_margin_pivot.csv', index_col=0)
    df.index.name = 'period'
    v.q_capacity_margin = df
    
    return v


def read_parameters(output_dir='output'):
    """
    Read all parameter CSV files into a namespace.
    
    Returns:
        SimpleNamespace: Namespace with parameters as attributes
    """
    output_path = Path(output_dir)
    p = SimpleNamespace()
    
    # Parameters with (period, time) index and multi-dimensional columns
    
    # step_duration: single value per (period, time)
    df = pd.read_csv(output_path / 'p_step_duration_pivot.csv', dtype={'value': float})
    df = df.set_index(['period', 'time'])
    p.step_duration = df['value']
    
    # p_flow_min: 3-level column MultiIndex (process, source, sink)
    df = pd.read_csv(output_path / 'p_flow_min_pivot.csv', header=[0, 1, 2], index_col=[0, 1])
    df.columns = pd.MultiIndex.from_tuples(
        [(col[0], col[1], col[2]) for col in df.columns],
        names=['process', 'source', 'sink']
    )
    df.index.names = ['period', 'time']
    p.flow_min = df
    
    # p_flow_max: 3-level column MultiIndex (process, source, sink)
    df = pd.read_csv(output_path / 'p_flow_max_pivot.csv', header=[0, 1, 2], index_col=[0, 1])
    df.columns = pd.MultiIndex.from_tuples(
        [(col[0], col[1], col[2]) for col in df.columns],
        names=['process', 'source', 'sink']
    )
    df.index.names = ['period', 'time']
    p.flow_max = df
    
    # pdtProcess_slope: 1-level column index (process)
    df = pd.read_csv(output_path / 'pdtProcess_slope_pivot.csv', index_col=[0, 1])
    df.index.names = ['period', 'time']
    p.process_slope = df
    
    # pdtProcess_section: 1-level column index (process)
    df = pd.read_csv(output_path / 'pdtProcess_section_pivot.csv', index_col=[0, 1])
    df.index.names = ['period', 'time']
    p.process_section = df
    
    # pdtProcess_availability: 1-level column index (process)
    df = pd.read_csv(output_path / 'pdtProcess_availability_pivot.csv', index_col=[0, 1])
    df.index.names = ['period', 'time']
    p.process_availability = df
    
    # pdtProcess_source_sink_varCost: 3-level column MultiIndex (process, source, sink)
    df = pd.read_csv(output_path / 'pdtProcess_source_sink_varCost_pivot.csv', header=[0, 1, 2], index_col=[0, 1])
    df.columns = pd.MultiIndex.from_tuples(
        [(col[0], col[1], col[2]) for col in df.columns],
        names=['process', 'source', 'sink']
    )
    df.index.names = ['period', 'time']
    p.process_source_sink_varCost = df
    
    # pdtNode_self_discharge_loss: 1-level column index (node)
    df = pd.read_csv(output_path / 'pdtNode_self_discharge_loss_pivot.csv', index_col=[0, 1])
    df.index.names = ['period', 'time']
    p.node_self_discharge_loss = df
    
    # pdtNode_penalty_up: 1-level column index (node)
    df = pd.read_csv(output_path / 'pdtNode_penalty_up_pivot.csv', index_col=[0, 1])
    df.index.names = ['period', 'time']
    p.node_penalty_up = df
    
    # pdtNode_penalty_down: 1-level column index (node)
    df = pd.read_csv(output_path / 'pdtNode_penalty_down_pivot.csv', index_col=[0, 1])
    df.index.names = ['period', 'time']
    p.node_penalty_down = df
    
    # pdtNodeInflow: 1-level column index (node)
    df = pd.read_csv(output_path / 'pdtNodeInflow_pivot.csv', index_col=[0, 1])
    df.index.names = ['period', 'time']
    p.node_inflow = df
    
    # pdtCommodity_price: 1-level column index (commodity)
    df = pd.read_csv(output_path / 'pdtCommodity_price_pivot.csv', index_col=[0, 1])
    df.index.names = ['period', 'time']
    p.commodity_price = df
    
    # pdtGroup_co2_price: 1-level column index (group)
    df = pd.read_csv(output_path / 'pdtGroup_co2_price_pivot.csv', index_col=[0, 1])
    df.index.names = ['period', 'time']
    p.group_co2_price = df
    
    # pdtReserve_upDown_group_reservation: 3-level column MultiIndex (reserve, updown, node_group)
    df = pd.read_csv(output_path / 'pdtReserve_upDown_group_reservation_pivot.csv', header=[0, 1, 2], index_col=[0, 1])
    df.columns = pd.MultiIndex.from_tuples(
        [(col[0], col[1], col[2]) for col in df.columns],
        names=['reserve', 'updown', 'node_group']
    )
    df.index.names = ['period', 'time']
    p.reserve_upDown_group_reservation = df
    
    # pdtProfile: 1-level column index (profile)
    df = pd.read_csv(output_path / 'pdtProfile_pivot.csv', index_col=[0, 1])
    df.index.names = ['period', 'time']
    p.profile = df
    
    # Parameters with only period index
    
    # p_years_d: single value per period
    df = pd.read_csv(output_path / 'p_years_d_pivot.csv', index_col=0, header=None, names=['period', 'value'])
    p.years_d = df['value']
    
    # p_entity_max_units: 1-level column index (entity)
    df = pd.read_csv(output_path / 'p_entity_max_units_pivot.csv', index_col=0)
    df.index.name = 'period'
    p.entity_max_units = df
    
    # p_entity_all_existing: 1-level column index (entity)
    df = pd.read_csv(output_path / 'p_entity_all_existing_pivot.csv', index_col=0)
    df.index.name = 'period'
    p.entity_all_existing = df
    
    # pdProcess_startup_cost: 1-level column index (process)
    df = pd.read_csv(output_path / 'pdProcess_startup_cost_pivot.csv', index_col=0)
    df.index.name = 'period'
    p.process_startup_cost = df
    
    # pdProcess_fixed_cost: 1-level column index (process)
    df = pd.read_csv(output_path / 'pdProcess_fixed_cost_pivot.csv', index_col=0)
    df.index.name = 'period'
    p.process_fixed_cost = df
    
    # pdNode_fixed_cost: 1-level column index (node)
    df = pd.read_csv(output_path / 'pdNode_fixed_cost_pivot.csv', index_col=0)
    df.index.name = 'period'
    p.node_fixed_cost = df
    
    # pdNode_annual_flow: 1-level column index (node)
    df = pd.read_csv(output_path / 'pdNode_annual_flow_pivot.csv', index_col=0)
    df.index.name = 'period'
    p.node_annual_flow = df
    
    # pdGroup parameters
    df = pd.read_csv(output_path / 'pdGroup_penalty_inertia_pivot.csv', index_col=0)
    df.index.name = 'period'
    p.group_penalty_inertia = df
    
    df = pd.read_csv(output_path / 'pdGroup_penalty_non_synchronous_pivot.csv', index_col=0)
    df.index.name = 'period'
    p.group_penalty_non_synchronous = df
    
    df = pd.read_csv(output_path / 'pdGroup_penalty_capacity_margin_pivot.csv', index_col=0)
    df.index.name = 'period'
    p.group_penalty_capacity_margin = df
    
    df = pd.read_csv(output_path / 'pdGroup_inertia_limit_pivot.csv', index_col=0)
    df.index.name = 'period'
    p.group_inertia_limit = df
    
    df = pd.read_csv(output_path / 'pdGroup_capacity_margin_pivot.csv', index_col=0)
    df.index.name = 'period'
    p.group_capacity_margin = df
    
    # ed_entity_annual_discounted: 1-level column index (entity)
    df = pd.read_csv(output_path / 'ed_entity_annual_discounted_pivot.csv', index_col=0)
    df.index.name = 'period'
    p.entity_annual_discounted = df
    
    # ed_entity_annual_divest_discounted: 1-level column index (entity)
    df = pd.read_csv(output_path / 'ed_entity_annual_divest_discounted_pivot.csv', index_col=0)
    df.index.name = 'period'
    p.entity_annual_divest_discounted = df
    
    # p_discount_factor_operations_yearly: single value per period
    df = pd.read_csv(output_path / 'p_discount_factor_operations_yearly_pivot.csv', index_col=0, header=None, names=['period', 'value'])
    p.discount_factor_operations_yearly = df['value']
    
    # node_capacity_for_scaling: 1-level column index (node)
    df = pd.read_csv(output_path / 'node_capacity_for_scaling_pivot.csv', index_col=0)
    df.index.name = 'period'
    p.node_capacity_for_scaling = df
    
    # group_capacity_for_scaling: 1-level column index (group)
    df = pd.read_csv(output_path / 'group_capacity_for_scaling_pivot.csv', index_col=0)
    df.index.name = 'period'
    p.group_capacity_for_scaling = df
    
    # complete_period_share_of_year: single value per period
    df = pd.read_csv(output_path / 'complete_period_share_of_year_pivot.csv', index_col=0, header=None, names=['period', 'value'])
    p.complete_period_share_of_year = df['value']
    
    # Parameters without d or (d,t) dimensions - tree format
    
    # p_node
    df = pd.read_csv(output_path / 'p_node.csv')
    p.node = df.set_index(['node', 'param'])

    # p_entity_unitsize
    df = pd.read_csv(output_path / 'p_entity_unitsize.csv', dtype={'entity': str, 'value': float})
    p.entity_unitsize = df.set_index('entity')
    
    # p_process_sink_coefficient
    df = pd.read_csv(output_path / 'p_process_sink_coefficient.csv', dtype={'process': str, 'sink': str, 'value': float})
    p.process_sink_coefficient = df.set_index(['process', 'sink'])
    
    # p_process_source_coefficient
    df = pd.read_csv(output_path / 'p_process_source_coefficient.csv', dtype={'process': str, 'source': str, 'value': float})
    p.process_source_coefficient = df.set_index(['process', 'source'])
    
    # p_commodity_co2_content
    df = pd.read_csv(output_path / 'p_commodity_co2_content.csv', dtype={'commodity': str, 'value': float})
    p.commodity_co2_content = df.set_index('commodity')
    
    # p_reserve_upDown_group_penalty
    df = pd.read_csv(output_path / 'p_reserve_upDown_group_penalty.csv', dtype={'reserve': str, 'updown': str, 'node_group': str, 'value': float})
    p.reserve_upDown_group_penalty = df.set_index(['reserve', 'updown', 'node_group'])
    
    # p_nested_model
    df = pd.read_csv(output_path / 'p_nested_model.csv', dtype={'param': str, 'value': float})
    p.nested_model = df.set_index('param')
    
    # p_roll_continue_state
    df = pd.read_csv(output_path / 'p_roll_continue_state.csv', dtype={'node': str, 'value': float})
    p.roll_continue_state = df.set_index('node')
    
    return p


def read_sets(output_dir='output'):
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
    s.dt_realize_dispatch = pd.read_csv(output_path / 'set_dt_realize_dispatch.csv')
    s.dt = pd.read_csv(output_path / 'set_dt.csv')
    s.ed_invest = pd.read_csv(output_path / 'set_ed_invest.csv')
    s.ed_divest = pd.read_csv(output_path / 'set_ed_divest.csv')
    s.edd_invest = pd.read_csv(output_path / 'set_edd_invest.csv')
 
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