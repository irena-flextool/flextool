from types import SimpleNamespace
from pathlib import Path
import pandas as pd


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
