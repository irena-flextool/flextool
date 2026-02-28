import pandas as pd


def compute_connection_flows(par, s, v, r) -> None:
    """Compute connection flow quantities and expose from_conn/to_conn for group aggregations."""
    step_duration = par.step_duration

    # Filter to connection processes and clip negatives
    conn_flows = r.flow_dt[
        r.flow_dt.columns[r.flow_dt.columns.get_level_values('process').isin(s.process_connection)]
    ].clip(lower=0.0)

    # Split into the four directional flows present in one connection
    conn_to_left = conn_flows[conn_flows.columns[conn_flows.columns.droplevel('source').isin(s.process_source)]]
    left_to_conn = conn_flows[conn_flows.columns[conn_flows.columns.droplevel('sink').isin(s.process_source)]]
    conn_to_right = conn_flows[conn_flows.columns[conn_flows.columns.droplevel('source').isin(s.process_sink)]]
    right_to_conn = conn_flows[conn_flows.columns[conn_flows.columns.droplevel('sink').isin(s.process_sink)]]

    conn_to_left.columns = conn_to_left.columns.droplevel('source')
    left_to_conn.columns = left_to_conn.columns.droplevel('sink')
    conn_to_right.columns = conn_to_right.columns.droplevel('source')
    right_to_conn.columns = right_to_conn.columns.droplevel('sink')

    conn_to_left.columns.names = ['process', 'node']
    left_to_conn.columns.names = ['process', 'node']
    conn_to_right.columns.names = ['process', 'node']
    right_to_conn.columns.names = ['process', 'node']

    r.connection_dt = conn_to_right.droplevel('node', axis=1).sub(
        conn_to_left.droplevel('node', axis=1), axis=1
    )
    r.connection_losses_dt = (
        r.connection_dt
        .sub(right_to_conn.droplevel('node', axis=1))
        .sub(left_to_conn.droplevel('node', axis=1))
    )
    r.connection_to_left_node__dt = conn_to_left.sub(left_to_conn, axis=1)
    r.connection_to_right_node__dt = conn_to_right.sub(right_to_conn, axis=1)
    r.connection_to_left_node__d = (
        r.connection_to_left_node__dt.groupby('period').sum()
        .div(par.complete_period_share_of_year, axis=0)
    )
    r.connection_to_right_node__d = (
        r.connection_to_right_node__dt.groupby('period').sum()
        .div(par.complete_period_share_of_year, axis=0)
    )

    # Expose combined directional flows for use by calc_group_flows
    r.from_conn = pd.concat([conn_to_left, conn_to_right], axis=1)  # columns: ['process', 'node']
    r.to_conn = pd.concat([left_to_conn, right_to_conn], axis=1)    # columns: ['process', 'node']

    # connection_d
    r_conn_weighted = r.connection_dt.mul(step_duration, axis=0)
    if not r_conn_weighted.empty:
        r.connection_d = r_conn_weighted[
            r_conn_weighted.index.get_level_values('period').isin(s.d_realized_period)
        ].groupby(level='period').sum()
    else:
        r.connection_d = pd.DataFrame(index=s.d_realized_period)

    # connection_losses_d
    r_conn_losses_weighted = r.connection_losses_dt.mul(step_duration, axis=0)
    if not r_conn_losses_weighted.empty:
        r.connection_losses_d = r_conn_losses_weighted[
            r_conn_losses_weighted.index.get_level_values('period').isin(s.d_realized_period)
        ].groupby(level='period').sum()
    else:
        r.connection_losses_d = pd.DataFrame(index=s.d_realized_period)
