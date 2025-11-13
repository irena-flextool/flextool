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

    # Create multi-index for variables with single header row
    v.state.columns = pd.MultiIndex.from_product([v.state.columns], names=['node'])
    v.online_linear.columns = pd.MultiIndex.from_product([v.online_linear.columns], names=['process'])
    v.startup_linear.columns = pd.MultiIndex.from_product([v.startup_linear.columns], names=['process'])
    v.shutdown_linear.columns = pd.MultiIndex.from_product([v.shutdown_linear.columns], names=['process'])
    v.online_integer.columns = pd.MultiIndex.from_product([v.online_integer.columns], names=['process'])
    v.startup_integer.columns = pd.MultiIndex.from_product([v.startup_integer.columns], names=['process'])
    v.shutdown_integer.columns = pd.MultiIndex.from_product([v.shutdown_integer.columns], names=['process'])
    v.q_state_up.columns = pd.MultiIndex.from_product([v.q_state_up.columns], names=['node'])
    v.q_state_down.columns = pd.MultiIndex.from_product([v.q_state_down.columns], names=['node'])
    v.q_inertia.columns = pd.MultiIndex.from_product([v.q_inertia.columns], names=['group'])
    v.q_non_synchronous.columns = pd.MultiIndex.from_product([v.q_non_synchronous.columns], names=['group'])
    v.q_state_up_group.columns = pd.MultiIndex.from_product([v.q_state_up_group.columns], names=['group'])
    v.q_capacity_margin.columns = pd.MultiIndex.from_product([v.q_capacity_margin.columns], names=['group'])
    v.invest.columns = pd.MultiIndex.from_product([v.invest.columns], names=['entity'])
    v.divest.columns = pd.MultiIndex.from_product([v.divest.columns], names=['entity'])
    v.dual_node_balance.columns = pd.MultiIndex.from_product([v.dual_node_balance.columns], names=['node'])
    v.dual_invest_unit.columns = pd.MultiIndex.from_product([v.dual_invest_unit.columns], names=['unit'])
    v.dual_invest_connection.columns = pd.MultiIndex.from_product([v.dual_invest_connection.columns], names=['connection'])
    v.dual_invest_node.columns = pd.MultiIndex.from_product([v.dual_invest_node.columns], names=['node'])

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
    p.process_source = pd.read_csv(output_path / 'p_process_source.csv', header=[0, 1], index_col=0).astype(float)
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
    p.process_startup_cost = pd.read_csv(output_path / 'pdProcess_startup_cost.csv', index_col=[0, 1]).astype(float)
    p.process_fixed_cost = pd.read_csv(output_path / 'pdProcess_fixed_cost.csv', index_col=[0, 1]).astype(float)
    p.node_fixed_cost = pd.read_csv(output_path / 'pdNode_fixed_cost.csv', index_col=[0, 1]).astype(float)
    p.node_annual_flow = pd.read_csv(output_path / 'pdNode_annual_flow.csv', index_col=[0, 1]).astype(float)
    p.group_penalty_inertia = pd.read_csv(output_path / 'pdGroup_penalty_inertia.csv', index_col=[0, 1]).astype(float)
    p.group_penalty_non_synchronous = pd.read_csv(output_path / 'pdGroup_penalty_non_synchronous.csv', index_col=[0, 1]).astype(float)
    p.group_penalty_capacity_margin = pd.read_csv(output_path / 'pdGroup_penalty_capacity_margin.csv', index_col=[0, 1]).astype(float)
    p.group_inertia_limit = pd.read_csv(output_path / 'pdGroup_inertia_limit.csv', index_col=[0, 1]).astype(float)
    p.group_capacity_margin = pd.read_csv(output_path / 'pdGroup_capacity_margin.csv', index_col=[0, 1]).astype(float)
    p.entity_annual_discounted = pd.read_csv(output_path / 'ed_entity_annual_discounted.csv', index_col=[0, 1]).astype(float)
    p.entity_annual_divest_discounted = pd.read_csv(output_path / 'ed_entity_annual_divest_discounted.csv', index_col=[0, 1]).astype(float)
    p.discount_factor_operations_yearly = pd.read_csv(output_path / 'p_discount_factor_operations_yearly.csv', index_col=[0, 1])['value'].astype(float)
    p.discount_factor_investment_yearly = pd.read_csv(output_path / 'p_discount_factor_investment_yearly.csv', index_col=[0, 1])['value'].astype(float)
    p.node_capacity_for_scaling = pd.read_csv(output_path / 'node_capacity_for_scaling.csv', index_col=[0, 1]).astype(float)
    p.group_capacity_for_scaling = pd.read_csv(output_path / 'group_capacity_for_scaling.csv', index_col=[0, 1]).astype(float)
    p.complete_period_share_of_year = pd.read_csv(output_path / 'complete_period_share_of_year.csv', index_col=[0, 1])['value'].astype(float)
    p.nested_model = pd.read_csv(output_path / 'p_nested_model.csv', dtype={'param': str, 'value': float}).set_index('param')
    p.roll_continue_state = pd.read_csv('solve_data/p_roll_continue_state.csv', dtype={'node': str, 'value': float}).set_index('node')

    # Parameters with multiple row and header indexes (read_csv does not interpret these when there are multiple header rows)
    p.flow_min.index.names = ['solve', 'period', 'time']
    p.flow_max.index.names = ['solve', 'period', 'time']
    p.process_source_sink_varCost.index.names = ['solve', 'period', 'time']
    p.reserve_upDown_group_reservation.index.names = ['solve', 'period', 'time']

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
    #p.process_sink_coefficient.columns = pd.MultiIndex.from_tuples(
    #    [(col[0], col[1]) for col in p.process_sink_coefficient.columns],
    #    names=['process', 'sink']
    #)
    #p.process_source_coefficient.columns = pd.MultiIndex.from_tuples(
    #    [(col[0], col[1]) for col in p.process_source_coefficient.columns],
    #    names=['process', 'source']
    #)
    #p.reserve_upDown_group_penalty.columns = pd.MultiIndex.from_tuples(
    #    [(col[0], col[1], col[2]) for col in p.reserve_upDown_group_penalty.columns],
    #    names=['reserve', 'updown', 'node_group']
    #)

    # Create a multi-index for those that have only one index in the header rows (i.e. columns)
    p.process_slope.columns = pd.MultiIndex.from_product([p.process_slope.columns], names=['process'])
    p.process_section.columns = pd.MultiIndex.from_product([p.process_section.columns], names=['process'])
    p.process_availability.columns = pd.MultiIndex.from_product([p.process_availability.columns], names=['process'])
    p.node_self_discharge_loss.columns = pd.MultiIndex.from_product([p.node_self_discharge_loss.columns], names=['node'])
    p.node_penalty_up.columns = pd.MultiIndex.from_product([p.node_penalty_up.columns], names=['node'])
    p.node_penalty_down.columns = pd.MultiIndex.from_product([p.node_penalty_down.columns], names=['node'])
    p.node_inflow.columns = pd.MultiIndex.from_product([p.node_inflow.columns], names=['node'])
    p.commodity_price.columns = pd.MultiIndex.from_product([p.commodity_price.columns], names=['commodity'])
    p.group_co2_price.columns = pd.MultiIndex.from_product([p.group_co2_price.columns], names=['group'])
    p.profile.columns = pd.MultiIndex.from_product([p.profile.columns], names=['profile'])
    p.entity_max_units.columns = pd.MultiIndex.from_product([p.entity_max_units.columns], names=['entity'])
    p.entity_all_existing.columns = pd.MultiIndex.from_product([p.entity_all_existing.columns], names=['entity'])
    p.process_startup_cost.columns = pd.MultiIndex.from_product([p.process_startup_cost.columns], names=['process'])
    p.process_fixed_cost.columns = pd.MultiIndex.from_product([p.process_fixed_cost.columns], names=['process'])
    p.node_fixed_cost.columns = pd.MultiIndex.from_product([p.node_fixed_cost.columns], names=['node'])
    p.node_annual_flow.columns = pd.MultiIndex.from_product([p.node_annual_flow.columns], names=['node'])
    p.group_penalty_inertia.columns = pd.MultiIndex.from_product([p.group_penalty_inertia.columns], names=['group'])
    p.group_penalty_non_synchronous.columns = pd.MultiIndex.from_product([p.group_penalty_non_synchronous.columns], names=['group'])
    p.group_penalty_capacity_margin.columns = pd.MultiIndex.from_product([p.group_penalty_capacity_margin.columns], names=['group'])
    p.group_inertia_limit.columns = pd.MultiIndex.from_product([p.group_inertia_limit.columns], names=['group'])
    p.group_capacity_margin.columns = pd.MultiIndex.from_product([p.group_capacity_margin.columns], names=['group'])
    p.entity_annual_discounted.columns = pd.MultiIndex.from_product([p.entity_annual_discounted.columns], names=['entity'])
    p.entity_annual_divest_discounted.columns = pd.MultiIndex.from_product([p.entity_annual_divest_discounted.columns], names=['entity'])
    p.node_capacity_for_scaling.columns = pd.MultiIndex.from_product([p.node_capacity_for_scaling.columns], names=['node'])
    p.group_capacity_for_scaling.columns = pd.MultiIndex.from_product([p.group_capacity_for_scaling.columns], names=['group'])
    p.node.columns = pd.MultiIndex.from_product([p.node.columns], names=['node'])

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

    # Process topology sets
    s.process_source_sink = pd.read_csv(output_path / 'set_process_source_sink.csv')
    s.process_method_sources_sinks = pd.read_csv(output_path / 'set_process_method_sources_sinks.csv')
   
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
    s.group_co2_price = pd.read_csv(output_path / 'set_group_co2_price.csv')

    # Group-related sets
    df = pd.read_csv(output_path / 'set_groupInertia.csv')
    s.groupInertia = pd.Index(df.iloc[:, 0])
    df = pd.read_csv(output_path / 'set_groupNonSync.csv')
    s.groupNonSync = pd.Index(df.iloc[:, 0])
    df = pd.read_csv(output_path / 'set_groupCapacityMargin.csv')
    s.groupCapacityMargin = pd.Index(df.iloc[:, 0])
    df = pd.read_csv(output_path / 'set_groupOutputNodeFlows.csv')
    s.groupOutputNodeFlows = pd.Index(df.iloc[:, 0])
    df = pd.read_csv(output_path / 'set_groupOutput_node.csv')
    s.groupOutput_node = pd.Index(df.iloc[:, 0])
    df = pd.read_csv(output_path / 'set_groupOutput_process.csv')
    s.groupOutput_process = pd.Index(df.iloc[:, 0])
    df = pd.read_csv(output_path / 'set_groupOutput.csv')
    s.groupOutput = pd.Index(df.iloc[:, 0])
    s.group_output__connection_Not_in_aggregate = pd.read_csv(output_path / 'set_group_output__connection_Not_in_aggregate.csv')
    s.group_output__process__unit__to_node_Not_in_aggregate = pd.read_csv(output_path / 'set_group_output__process__unit__to_node_Not_in_aggregate.csv')
    s.group_output__process__node__to_unit_Not_in_aggregate = pd.read_csv(output_path / 'set_group_output__process__node__to_unit_Not_in_aggregate.csv')
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
    s.group_process = pd.read_csv(output_path / 'set_group_process.csv').set_index(['group', 'process']).index
    s.group_process_node = pd.read_csv(output_path / 'set_group_process_node.csv').set_index(['group', 'process', 'node']).index

    # upDown set
    df = pd.read_csv(output_path / 'set_upDown.csv')
    s.upDown = pd.Index(df.iloc[:, 0])

    # Optional output flags
    df = pd.read_csv(output_path / 'set_enable_optional_outputs.csv')
    s.enable_optional_outputs = set(df.iloc[:, 0])

    return s