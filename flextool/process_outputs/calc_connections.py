import pandas as pd


def compute_connection_flows(par, s, v, r) -> None:
    """Compute connection flow quantities and expose from_conn/to_conn for group aggregations."""
    step_duration = par.step_duration

    # Identify method_2way_1var connections (single signed flow variable).
    # These include DC power flow connections (no_losses_no_variable_cost).
    conn_1var: set[str] = set()
    for proc, method in s.process_method:
        if proc in s.process_connection and method == 'method_2way_1var_off':
            conn_1var.add(proc)

    # Filter to connection processes.
    # For 2var connections: clip to non-negative (each direction is a separate variable).
    # For 1var connections: keep sign (positive = left→right, negative = right→left).
    all_conn_flows = r.flow_dt[
        r.flow_dt.columns[r.flow_dt.columns.get_level_values('process').isin(s.process_connection)]
    ]
    cols_1var = [c for c in all_conn_flows.columns if c[0] in conn_1var]
    cols_2var = [c for c in all_conn_flows.columns if c[0] not in conn_1var]
    parts = []
    if cols_1var:
        parts.append(all_conn_flows[cols_1var])  # keep sign
    if cols_2var:
        parts.append(all_conn_flows[cols_2var].clip(lower=0.0))  # clip
    conn_flows = pd.concat(parts, axis=1) if len(parts) > 1 else parts[0] if parts else pd.DataFrame()

    # Split into the four directional flows present in one connection.
    # For 2var: both forward and reverse entries exist in process_source_sink.
    # For 1var: only one entry exists — left_to_conn and conn_to_right match,
    #           conn_to_left and right_to_conn are empty. The signed flow value
    #           naturally gives correct net flow and directional results.
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
        conn_to_left.droplevel('node', axis=1), fill_value=0
    )
    # Losses = total_out - total_in (negative when energy is lost in connections)
    r.connection_losses_dt = (
        conn_to_left.droplevel('node', axis=1)
        .add(conn_to_right.droplevel('node', axis=1), fill_value=0)
        .sub(left_to_conn.droplevel('node', axis=1), fill_value=0)
        .sub(right_to_conn.droplevel('node', axis=1), fill_value=0)
    )
    r.connection_to_left_node__dt = conn_to_left.sub(left_to_conn, fill_value=0)
    r.connection_to_right_node__dt = conn_to_right.sub(right_to_conn, fill_value=0)
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

    # DC power flow: angle differences per connection
    if not v.angle.empty and not s.connection_dc_power_flow.empty:
        angle_diff_cols: dict[str, pd.Series] = {}
        for conn in s.connection_dc_power_flow:
            # Find source and sink nodes for this connection from process_source_sink
            pss_for_conn = s.process_source_sink[
                s.process_source_sink.get_level_values('process') == conn
            ]
            if pss_for_conn.empty:
                continue
            source_node = pss_for_conn.get_level_values('source')[0]
            sink_node = pss_for_conn.get_level_values('sink')[0]
            if source_node in v.angle.columns and sink_node in v.angle.columns:
                angle_diff_cols[conn] = v.angle[source_node] - v.angle[sink_node]
        if angle_diff_cols:
            r.connection_angle_diff_dt = pd.DataFrame(angle_diff_cols)
            r.connection_angle_diff_dt.columns.name = 'connection'
        else:
            r.connection_angle_diff_dt = pd.DataFrame()
    else:
        r.connection_angle_diff_dt = pd.DataFrame()

    # Store angle_dt for output (may be empty)
    r.angle_dt = v.angle
