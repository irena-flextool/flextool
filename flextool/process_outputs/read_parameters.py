from types import SimpleNamespace
from pathlib import Path
import pandas as pd


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
    p.rp_cost_weight = pd.read_csv(output_path / 'p_rp_cost_weight.csv', index_col=[0, 1, 2])['value'].astype(float)
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
    p.entity_all_capacity = pd.read_csv(output_path / 'entity_all_capacity.csv', index_col=[0, 1]).astype(float)
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
    p.inflation_factor_operations_yearly = pd.read_csv(output_path / 'p_inflation_factor_operations_yearly.csv', index_col=[0, 1])['value'].astype(float)
    p.inflation_factor_investment_yearly = pd.read_csv(output_path / 'p_inflation_factor_investment_yearly.csv', index_col=[0, 1])['value'].astype(float)
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
    p.entity_all_capacity.columns.name = 'entity'
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
