from types import SimpleNamespace
from pathlib import Path
import pandas as pd


def read_sets(output_dir):
    """
    Read set definitions from CSV files into a namespace.
    Simple sets are stored as pandas Index for fast O(1) membership testing.
    Tuple sets are stored as DataFrames for vectorized operations.

    ``output_dir`` points at ``work_folder/output_raw``.  All resolved-set
    files (``entity.csv``, ``period.csv``, ``node__storage_binding_method.csv``,
    etc.) live in the sibling ``solve_data/`` folder, written there by the
    GMPL ``printf`` blocks in ``flextool.mod``.  Raw user-input parameter
    files (``p_node_type.csv``, ``node_dc_power_flow.csv``,
    ``connection_dc_power_flow.csv``, ``p_connection_susceptance.csv``)
    still come from ``input/``.

    Returns:
        SimpleNamespace: Namespace with sets as attributes
    """
    output_path = Path(output_dir)
    work_folder = output_path.parent
    input_path = work_folder / 'input'
    solve_data_path = work_folder / 'solve_data'
    s = SimpleNamespace()

    # Process and entity sets (write-once)
    s.entity = pd.read_csv(input_path / 'entity.csv').set_index(['entity']).index
    s.entityInvest = pd.read_csv(solve_data_path / 'entityInvest.csv').set_index(['entity']).index
    s.entityDivest = pd.read_csv(solve_data_path / 'entityDivest.csv').set_index(['entity']).index
    s.process_online = pd.read_csv(solve_data_path / 'process_online.csv').set_index(['process']).index
    s.process_online_integer = pd.read_csv(solve_data_path / 'process_online_integer.csv').set_index(['process']).index
    s.process_online_linear = pd.read_csv(solve_data_path / 'process_online_linear.csv').set_index(['process']).index

    # Tuple sets - store as DataFrames for vectorized filtering and operations

    # Tuple sets that need filtering - keep as DataFrame
    s.period = pd.read_csv(solve_data_path / 'period.csv').set_index(['solve', 'period']).index
    s.d_realized_period = pd.read_csv(solve_data_path / 'd_realized_period.csv').set_index(['solve', 'period']).index
    s.d_realize_invest = pd.read_csv(solve_data_path / 'd_realize_invest.csv').set_index(['solve', 'period']).index
    s.dt_realize_dispatch = pd.MultiIndex.from_frame(pd.read_csv(solve_data_path / 'dt_realize_dispatch.csv'))
    s.d_realize_dispatch_or_invest = pd.MultiIndex.from_frame(pd.read_csv(solve_data_path / 'd_realize_dispatch_or_invest.csv'))
    s.dt = pd.MultiIndex.from_frame(pd.read_csv(solve_data_path / 'dt.csv'))
    s.ed_invest = pd.MultiIndex.from_frame(pd.read_csv(solve_data_path / 'solve__ed_invest.csv'))
    s.ed_divest = pd.MultiIndex.from_frame(pd.read_csv(solve_data_path / 'solve__ed_divest.csv'))
    s.edd_invest = pd.MultiIndex.from_frame(pd.read_csv(solve_data_path / 'solve__edd_invest.csv'))
    s.process__node__profile__profile_method = pd.MultiIndex.from_frame(pd.read_csv(input_path / 'process__node__profile__profile_method.csv'))

    # Process topology sets (write-once)
    s.process_source_sink = pd.read_csv(solve_data_path / 'process_source_sink.csv').set_index(['process', 'source', 'sink']).index
    s.process_method_sources_sinks = pd.read_csv(solve_data_path / 'process_method_sources_sinks.csv').set_index(['process', 'method', 'orig_source', 'orig_sink', 'always_source', 'always_sink']).index

    # Process method sets (write-once)
    s.process_method = pd.read_csv(input_path / 'process_method.csv').set_index(['process', 'method']).index
    s.process__ct_method = pd.read_csv(solve_data_path / 'process__ct_method.csv').set_index(['process', 'method']).index

     # Method sets (write-once)
    df = pd.read_csv(solve_data_path / 'method_1var_per_way.csv')
    s.method_1var_per_way = pd.Index(df.iloc[:, 0])

    df = pd.read_csv(solve_data_path / 'method_nvar.csv')
    s.method_nvar = pd.Index(df.iloc[:, 0])

    # Time-related sets (per-solve)
    s.dtt = pd.MultiIndex.from_frame(pd.read_csv(solve_data_path / 'dtt.csv'))
    s.dtttdt = pd.MultiIndex.from_frame(pd.read_csv(solve_data_path / 'dtttdt.csv'))
    s.period__time_first = pd.MultiIndex.from_frame(pd.read_csv(solve_data_path / 'period__time_first.csv'))
    s.period_first_of_solve = pd.MultiIndex.from_frame(pd.read_csv(solve_data_path / 'solve__period_first.csv'))
    s.period_in_use = pd.MultiIndex.from_frame(pd.read_csv(solve_data_path / 'period_in_use.csv'))
    s.dt_fix_storage_timesteps = pd.MultiIndex.from_frame(pd.read_csv(solve_data_path / 'dt_fix_storage_timesteps.csv'))

    # Node-related sets — derived from per-node p_node_type.
    # Nodes absent from p_node_type.csv take the mod's default 'balance'
    # (flextool.mod: `param p_node_type ... default 'balance'`).
    all_nodes = pd.read_csv(input_path / 'node.csv')['node']
    nt_df = pd.read_csv(input_path / 'p_node_type.csv')
    nt_map = dict(zip(nt_df['node'], nt_df['p_node_type']))
    node_types = {n: nt_map.get(n, 'balance') for n in all_nodes}
    s.node_state           = pd.Index([n for n, t in node_types.items() if t == 'storage'])
    s.node_balance         = pd.Index([n for n, t in node_types.items() if t in ('balance', 'storage')])
    s.node_balance_period  = pd.Index([n for n, t in node_types.items() if t == 'balance_within_period'])
    s.node_commodity       = pd.Index([n for n, t in node_types.items() if t == 'commodity'])
    df = pd.read_csv(solve_data_path / 'nodeSelfDischarge.csv')
    s.node_self_discharge = pd.Index(df.iloc[:, 0])
    s.node__storage_binding_method = pd.read_csv(solve_data_path / 'node__storage_binding_method.csv').set_index(['node', 'method']).index
    s.node__storage_start_end_method = pd.read_csv(solve_data_path / 'node__storage_start_end_method.csv').set_index(['node', 'method']).index
    s.node__inflow_method = pd.read_csv(solve_data_path / 'node__inflow_method.csv').set_index(['node', 'method']).index
    s.node__storage_nested_fix_method = pd.read_csv(solve_data_path / 'node__storage_nested_fix_method.csv').set_index(['node', 'method']).index

    # Process-related sets (write-once)
    s.process = pd.read_csv(input_path / 'process.csv').set_index(['process']).index
    s.node = pd.read_csv(input_path / 'node.csv').set_index(['node']).index
    s.process_source = pd.read_csv(solve_data_path / 'process_source.csv').set_index(['process', 'source']).index
    s.process_sink = pd.read_csv(solve_data_path / 'process_sink.csv').set_index(['process', 'sink']).index
    s.process_VRE = pd.read_csv(solve_data_path / 'process_VRE.csv').set_index(['process', 'node']).index
    s.process__source__sink__profile__profile_method = pd.read_csv(solve_data_path / 'process__source__sink__profile__profile_method.csv')

    # Process type sets (write-once)
    s.process_unit = pd.read_csv(solve_data_path / 'process_unit.csv').set_index(['process']).index
    s.process_connection = pd.read_csv(solve_data_path / 'process_connection.csv').set_index(['process']).index
    s.process_profile = pd.read_csv(solve_data_path / 'process_profile.csv').set_index(['process']).index

    # Commodity-related sets (write-once)
    s.commodity_node = pd.read_csv(solve_data_path / 'commodity_node.csv').set_index(['commodity', 'node']).index
    s.commodity_node_co2 = pd.read_csv(solve_data_path / 'commodity_node_co2.csv').set_index(['commodity', 'node']).index
    s.process__commodity__node = pd.read_csv(solve_data_path / 'process__commodity__node.csv').set_index(['process', 'commodity', 'node']).index
    s.process__commodity__node_co2 = pd.read_csv(solve_data_path / 'process__commodity__node_co2.csv').set_index(['process', 'commodity', 'node']).index
    s.group_co2_price = pd.read_csv(solve_data_path / 'group_co2_price.csv').set_index(['group']).index
    s.group_co2_limit = pd.read_csv(solve_data_path / 'group_co2_limit.csv').set_index(['group']).index

    # Group-related sets (write-once)
    df = pd.read_csv(input_path / 'groupInertia.csv')
    s.groupInertia = pd.Index(df.iloc[:, 0])
    df = pd.read_csv(input_path / 'groupNonSync.csv')
    s.groupNonSync = pd.Index(df.iloc[:, 0])
    df = pd.read_csv(input_path / 'groupCapacityMargin.csv')
    s.groupCapacityMargin = pd.Index(df.iloc[:, 0])
    df = pd.read_csv(input_path / 'nodeGroupDispatch.csv')
    s.nodeGroupDispatch = pd.Index(df.iloc[:, 0])
    df = pd.read_csv(input_path / 'nodeGroupIndicators.csv')
    s.nodeGroupIndicators = pd.Index(df.iloc[:, 0])
    df = pd.read_csv(input_path / 'flowGroupIndicators.csv')
    s.flowGroupIndicators = pd.Index(df.iloc[:, 0])
    s.nodeGroupDispatch__connection_Not_in_aggregate = pd.read_csv(solve_data_path / 'nodeGroupDispatch__connection_Not_in_aggregate.csv').set_index(['group', 'connection']).index
    s.nodeGroupDispatch__process__unit__to_node_Not_in_aggregate = pd.read_csv(solve_data_path / 'nodeGroupDispatch__process__unit__to_node_Not_in_aggregate.csv').set_index(['group', 'process', 'unit', 'node']).index
    s.nodeGroupDispatch__process__node__to_unit_Not_in_aggregate = pd.read_csv(solve_data_path / 'nodeGroupDispatch__process__node__to_unit_Not_in_aggregate.csv').set_index(['group', 'process', 'node', 'unit']).index
    s.nodeGroupDispatch__process__connection__to_node_Not_in_aggregate = pd.read_csv(solve_data_path / 'nodeGroupDispatch__process__connection__to_node_Not_in_aggregate.csv').set_index(['group', 'process', 'connection', 'node']).index
    s.nodeGroupDispatch__process__node__to_connection_Not_in_aggregate = pd.read_csv(solve_data_path / 'nodeGroupDispatch__process__node__to_connection_Not_in_aggregate.csv').set_index(['group', 'process', 'node', 'connection']).index
    s.nodeGroupDispatch__processGroup_Unit_to_group = pd.read_csv(solve_data_path / 'nodeGroupDispatch__group_aggregate_Unit_to_group.csv').set_index(['group', 'group_aggregate']).index
    s.nodeGroupDispatch__processGroup__process__unit__to_node = pd.read_csv(solve_data_path / 'nodeGroupDispatch__group_aggregate__process__unit__to_node.csv').set_index(['group', 'group_aggregate', 'unit', 'source', 'sink']).index
    s.nodeGroupDispatch__processGroup__process__unit__to_node.names = ['group', 'group_aggregate', 'process', 'unit', 'node']
    s.nodeGroupDispatch__processGroup_Group_to_unit = pd.read_csv(solve_data_path / 'nodeGroupDispatch__group_aggregate_Group_to_unit.csv').set_index(['group', 'group_aggregate']).index
    s.nodeGroupDispatch__processGroup__process__node__to_unit = pd.read_csv(solve_data_path / 'nodeGroupDispatch__group_aggregate__process__node__to_unit.csv').set_index(['group', 'group_aggregate', 'unit', 'source', 'sink']).index
    s.nodeGroupDispatch__processGroup__process__node__to_unit.names = ['group', 'group_aggregate', 'process', 'node', 'unit']
    s.nodeGroupDispatch__processGroup_Connection = pd.read_csv(solve_data_path / 'nodeGroupDispatch__group_aggregate_Connection.csv').set_index(['group', 'group_aggregate']).index
    s.nodeGroupDispatch__processGroup__process__connection__to_node = pd.read_csv(solve_data_path / 'nodeGroupDispatch__group_aggregate__process__connection__to_node.csv').set_index(['group', 'group_aggregate', 'connection', 'source', 'sink']).index
    s.nodeGroupDispatch__processGroup__process__connection__to_node.names = ['group', 'group_aggregate', 'process', 'connection', 'node']
    s.nodeGroupDispatch__processGroup__process__node__to_connection = pd.read_csv(solve_data_path / 'nodeGroupDispatch__group_aggregate__process__node__to_connection.csv').set_index(['group', 'group_aggregate', 'connection', 'source', 'sink']).index
    s.nodeGroupDispatch__processGroup__process__node__to_connection.names = ['group', 'group_aggregate', 'process', 'node', 'connection']
    s.nodeGroupDispatch__process_fully_inside = pd.read_csv(solve_data_path / 'nodeGroupDispatch__process_fully_inside.csv').set_index(['group', 'process']).index
    s.group_node = pd.read_csv(solve_data_path / 'group_node.csv').set_index(['group', 'node']).index
    s.group_process = pd.read_csv(solve_data_path / 'group_process.csv').set_index(['group', 'process']).index
    s.group_process_node = pd.read_csv(solve_data_path / 'group_process_node.csv').set_index(['group', 'process', 'node']).index

    # upDown set (write-once)
    df = pd.read_csv(solve_data_path / 'upDown.csv')
    s.upDown = pd.Index(df.iloc[:, 0])

    # Optional output flags (write-once)
    df = pd.read_csv(solve_data_path / 'enable_optional_outputs.csv')
    s.enable_optional_outputs = set(df.iloc[:, 0])

    # DC power flow sets (read from input/ directory, sibling of output_raw/)
    # These may not exist or may be empty when no DC PF is configured.
    node_dc_pf_path = input_path / 'node_dc_power_flow.csv'
    if node_dc_pf_path.exists():
        df = pd.read_csv(node_dc_pf_path)
        s.node_dc_power_flow = pd.Index(df.iloc[:, 0]) if not df.empty else pd.Index([], dtype=str)
    else:
        s.node_dc_power_flow = pd.Index([], dtype=str)

    conn_dc_pf_path = input_path / 'connection_dc_power_flow.csv'
    if conn_dc_pf_path.exists():
        df = pd.read_csv(conn_dc_pf_path)
        s.connection_dc_power_flow = pd.Index(df.iloc[:, 0]) if not df.empty else pd.Index([], dtype=str)
    else:
        s.connection_dc_power_flow = pd.Index([], dtype=str)

    susceptance_path = input_path / 'p_connection_susceptance.csv'
    if susceptance_path.exists():
        df = pd.read_csv(susceptance_path)
        if not df.empty and len(df.columns) >= 2:
            s.connection_susceptance = df.set_index(df.columns[0])[df.columns[1]]
        else:
            s.connection_susceptance = pd.Series(dtype=float)
    else:
        s.connection_susceptance = pd.Series(dtype=float)

    return s
