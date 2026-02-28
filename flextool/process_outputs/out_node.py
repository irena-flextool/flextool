import pandas as pd


def node_summary(par, s, v, r, debug):
    """Node balance summaries for periods and timesteps"""
    results = []

    categories = ['From units', 'From connections', 'Loss of load', 'To units', 'To connections', 'Self discharge', 'Excess load', 'Inflow']

    balanced_nodes = s.node_balance.union(s.node_balance_period)
    if debug:
        nodes = s.node
    else:
        nodes = balanced_nodes.difference(s.node_state)
    nodes_sink = s.node.copy().intersection(nodes)
    nodes_sink.name = 'sink'
    nodes_source = s.node.copy().intersection(nodes)
    nodes_source.name = 'source'

    # 1. Timestep-level node summary
    node_dt = pd.DataFrame(index=s.dt_realize_dispatch, columns=pd.MultiIndex.from_product([nodes, categories], names=['node', 'category']), dtype=float)

    # From units
    from_units = r.flow_dt[s.process_unit.join(r.flow_dt.columns).join(nodes_sink, how='inner')].T.groupby('sink').sum().T
    from_units_cols = node_dt.columns[
                        node_dt.columns.get_level_values('node').isin(from_units.columns)
                        & node_dt.columns.get_level_values('category').isin(['From units'])]
    node_dt[from_units_cols] = from_units[from_units_cols.get_level_values('node')]

    # From connections
    from_connections = r.flow_dt[s.process_connection.join(r.flow_dt.columns).join(nodes_sink, how='inner')].T.groupby('sink').sum().T
    from_connections_cols = node_dt.columns[
                        node_dt.columns.get_level_values('node').isin(from_connections.columns)
                        & node_dt.columns.get_level_values('category').isin(['From connections'])]
    node_dt[from_connections_cols] = from_connections[from_connections_cols.get_level_values('node')]

    # Upward slack
    upward_slack_data = v.q_state_up.loc[:, v.q_state_up.columns.get_level_values('node').isin(balanced_nodes.intersection(nodes))].clip(lower=0)
    upward_slack_data = upward_slack_data.mul(par.node_capacity_for_scaling[upward_slack_data.columns], axis=1)
    upward_slack_cols = node_dt.columns[
                        node_dt.columns.get_level_values('node').isin(upward_slack_data.columns.get_level_values('node'))
                        & node_dt.columns.get_level_values('category').isin(['Loss of load'])]
    node_dt[upward_slack_cols] = upward_slack_data[upward_slack_cols.get_level_values('node')]

    # To units (negative)
    to_units = -r.flow_dt[s.process_unit.join(r.flow_dt.columns).join(nodes_source, how='inner')].T.groupby('source').sum().T
    to_units_cols = node_dt.columns[
                        node_dt.columns.get_level_values('node').isin(to_units.columns)
                        & node_dt.columns.get_level_values('category').isin(['To units'])]
    node_dt[to_units_cols] = to_units[to_units_cols.get_level_values('node')]

    # To connections (negative)
    to_connections = -r.flow_dt[s.process_connection.join(r.flow_dt.columns).join(nodes_source, how='inner')].T.groupby('source').sum().T
    to_connections_cols = node_dt.columns[
                        node_dt.columns.get_level_values('node').isin(to_connections.columns)
                        & node_dt.columns.get_level_values('category').isin(['To connections'])]
    node_dt[to_connections_cols] = to_connections[to_connections_cols.get_level_values('node')]

    # Self discharge (negative)
    self_discharge = -r.self_discharge_loss_dt[r.self_discharge_loss_dt.columns.intersection(s.node_self_discharge.intersection(nodes))]
    self_discharge_cols = node_dt.columns[
                        node_dt.columns.get_level_values('node').isin(self_discharge.columns)
                        & node_dt.columns.get_level_values('category').isin(['Self discharge'])]
    node_dt[self_discharge_cols] = self_discharge[self_discharge_cols.get_level_values('node')]

    # Downward slack (negative)
    downward_slack_data = (-v.q_state_down.loc[:, v.q_state_down.columns.get_level_values('node').isin(balanced_nodes.intersection(nodes))]).clip(upper=0)
    downward_slack_data = downward_slack_data.mul(par.node_capacity_for_scaling[downward_slack_data.columns], axis=1)
    downward_slack_cols = node_dt.columns[
                        node_dt.columns.get_level_values('node').isin(downward_slack_data.columns.get_level_values('node'))
                        & node_dt.columns.get_level_values('category').isin(['Excess load'])]
    node_dt[downward_slack_cols] = downward_slack_data[downward_slack_cols.get_level_values('node')]

    # Inflow
    inflow_cols = node_dt.columns[
                    node_dt.columns.get_level_values('node').isin(par.node_inflow.columns.intersection(nodes))
                    & node_dt.columns.get_level_values('category').isin(['Inflow'])]
    node_dt[inflow_cols] = par.node_inflow[inflow_cols.get_level_values('node')]

    # Fill any remaining NaN values with 0
    node_dt = node_dt.fillna(0.0)

    results.append((node_dt, 'node_dt_ep'))

    # 2. Period-level node summary
    node_d = node_dt.groupby('period').sum().div(par.complete_period_share_of_year, axis=0, level=1)

    results.append((node_d, 'node_d_ep'))

    return results


def node_additional_results(par, s, v, r, debug):
    """Additional node results: prices, state, and slacks"""
    results = []

    # 1. Nodal prices
    results.append((v.dual_node_balance[s.node_balance.difference(s.node_state)], 'node_prices_dt_e'))

    # 2. Node state
    node_state = v.state.mul(par.entity_unitsize[s.node_state], level="node")
    results.append((node_state, 'node_state_dt_e'))

    # 3. Node upward slack
    upward_slack = v.q_state_up.mul(par.node_capacity_for_scaling[s.node_balance.union(s.node_balance_period)], level=0).clip(lower=0)
    results.append((upward_slack, 'node_slack_up_dt_e'))
    upward_slack_d = upward_slack.groupby(level='period').sum().div(par.complete_period_share_of_year, axis=0, level=1)
    results.append((upward_slack_d, 'node_slack_up_d_e'))

    # 4. Node downward slack
    downward_slack = v.q_state_down.mul(par.node_capacity_for_scaling[s.node_balance.union(s.node_balance_period)], level=0).clip(lower=0)
    results.append((downward_slack, 'node_slack_down_dt_e'))
    downward_slack_d = downward_slack.groupby(level='period').sum().div(par.complete_period_share_of_year, axis=0, level=1)
    results.append((downward_slack_d, 'node_slack_down_d_e'))

    return results
