import pandas as pd


def compute_capacity_and_flows(par, s, v, r) -> None:
    """Compute capacity, process online state, and flow-related quantities."""
    step_duration = par.step_duration
    unitsize = par.entity_unitsize

    # hours_in_realized_period
    hours_in_realized_period = step_duration.groupby(level='period').sum()
    hours_in_realized_period = hours_in_realized_period.reindex(s.d_realized_period)
    r.hours_in_realized_period = hours_in_realized_period
    r.realized_period_share_of_year = hours_in_realized_period / 8760

    s.nb = s.node_balance.union(s.node_balance_period)

    # entity_all_capacity: read directly from model output (existing + cumulative invest - divest)
    # The model uses edd_invest set to correctly track which investments apply to which periods,
    # handling both single-solve and multi-solve scenarios.
    r.entity_all_capacity = par.entity_all_capacity.copy()
    # Drop solve level from index — downstream code expects (period,) only
    if r.entity_all_capacity.index.nlevels > 1:
        r.entity_all_capacity = r.entity_all_capacity.droplevel('solve')
    r.entity_all_capacity.columns.name = 'process'  # Required for level=0 matching in VRE calculations

    # process_online_dt
    r.process_online_dt = v.online_linear.add(v.online_integer, fill_value=0)

    # flow_dt: applies unit-size scaling and slope/section transformations
    s.process_source_sink_alwaysProcess = s.process_method_sources_sinks.droplevel(['method', 'orig_source', 'orig_sink'])
    s.process_source_sink_alwaysProcess.names = ['process', 'source', 'sink']

    r.flow_dt = pd.DataFrame(
        index=s.dt_realize_dispatch,
        columns=s.process_source_sink_alwaysProcess,
        dtype=float
    )
    slope = par.process_slope
    section = par.process_section

    for row in s.process_method_sources_sinks:
        p = row[0]
        method = row[1]
        orig_source, orig_sink = row[2], row[3]
        always_source, always_sink = row[4], row[5]

        flow_val = v.flow[[(p, orig_source, orig_sink)]] * unitsize[p]

        if (method in s.method_1var_per_way and
                p not in s.process_profile and
                orig_source == always_source and orig_sink != always_sink):
            flow_val = flow_val.mul(slope[p], axis=0)
            if p in s.process_unit:
                flow_val /= (par.process_sink_coefficient.loc[p, orig_sink] *
                             par.process_source_coefficient.loc[p, orig_source])
            if (p, 'min_load_efficiency') in s.process__ct_method:
                flow_val = flow_val.add(r.process_online_dt[p] * section[p] * unitsize[p], axis=0)
        r.flow_dt[p, always_source, always_sink] = flow_val

    # ramp_dtt
    current_idx = s.dtt.droplevel('t_previous')
    previous_idx = s.dtt.droplevel('time')
    r.ramp_dtt = pd.DataFrame(
        r.flow_dt.reindex(current_idx).values - r.flow_dt.reindex(previous_idx).values,
        index=s.dtt,
        columns=r.flow_dt.columns,
        dtype=float
    )

    # flow_d (with step_duration), process_source_flow_d, process_sink_flow_d
    r.flow_d = r.flow_dt.mul(step_duration, axis=0).groupby('period').sum()
    r.process_source_flow_d = (
        r.flow_d.T.groupby(level=['process', 'source']).sum().T
        .reindex(columns=s.process_source, fill_value=0.0)
    )
    r.process_sink_flow_d = (
        r.flow_d.T.groupby(level=['process', 'sink']).sum().T
        .reindex(columns=s.process_sink, fill_value=0.0)
    )
