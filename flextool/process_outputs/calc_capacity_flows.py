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
    # Drop solve level from index — downstream code expects (period,) only.
    # For rolling scenarios, keep only the last solve per period.
    if r.entity_all_capacity.index.nlevels > 1:
        r.entity_all_capacity = r.entity_all_capacity.droplevel('solve')
        r.entity_all_capacity = r.entity_all_capacity[~r.entity_all_capacity.index.duplicated(keep='last')]
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

        # method_2way_1var_off arcs carry a single signed flow
        # ``v_flow - v_flow_back``: the forward variable ``v_flow >= 0``
        # plus a non-negative reverse auxiliary ``v_flow_back``.  Subtract
        # the back auxiliary so the reported flow is the true signed net
        # (negative = sink->source), matching the engine's nodeBalance.
        flow_back = getattr(v, "flow_back", None)
        if (flow_back is not None and not flow_back.empty
                and (p, orig_source, orig_sink) in flow_back.columns):
            # ``flow_back`` shares ``v.flow``'s (solve, period, time) row
            # index, so its column lines up element-wise with ``flow_val``
            # (which is ``v.flow[[col]] * unitsize``).  Subtract by raw
            # values to stay agnostic to row-index level differences.
            back_vals = (flow_back[(p, orig_source, orig_sink)] * unitsize[p]).values
            flow_val = flow_val.sub(back_vals, axis=0)

        # A direct unit's single v_flow is its output; the input it draws is output × slope (+ online × section),
        # as in nodeBalance_eq (balance_flow_terms: source_eff / source_section).  A profile on the unit limits that
        # same v_flow, so a unit with a profile converts like any other.
        if (method in s.method_1var_per_way and
                orig_source == always_source and orig_sink != always_sink):
            # ``slope[p]`` already carries the direct unit's output/input conversion_flow_coeff
            # ratio (#10 / Q12 — folded at the source by ``p_slope_from_source`` /
            # ``_derive_conversion_trio``), so no separate coefficient division is applied here.
            flow_val = flow_val.mul(slope[p], axis=0)
            if (p, 'min_load_efficiency') in s.process__ct_method:
                online_p = r.process_online_dt[p]
                # A solve whose reduce_unit_detail_method left out online variables has no units online (NaN)
                # and a section folded to 0 (_reduce_detail): its fuel is output × slope alone.  Only its steps
                # are filled; a missing online cell of any other solve stays NaN.
                left_out = getattr(s, 'reduced_cells', None)
                if left_out:
                    online_p = online_p.reindex(flow_val.index)
                    online_p = online_p.mask(left_out.step_mask('online', online_p.index) & online_p.isna(), 0.0)
                flow_val = flow_val.add(online_p * section[p] * unitsize[p], axis=0)
        r.flow_dt[p, always_source, always_sink] = flow_val

    # ramp_dt: flow change from the step each step's ramp is measured from (s.ramp_link: the adjacent
    # predecessor, or the previous roll's last realized step at a roll boundary) — on the realized flows, so
    # the change is in the reported flow's own units.  Empty (NaN) at steps without one.
    r.ramp_dt = pd.DataFrame(index=r.flow_dt.index, columns=r.flow_dt.columns, dtype=float)
    if len(s.ramp_link) > 0:
        current_idx = s.ramp_link.droplevel(['period_previous', 'time_previous'])
        previous_idx = s.ramp_link.droplevel(['period', 'time']).set_names(['period', 'time'])
        change = pd.DataFrame(
            r.flow_dt.reindex(current_idx).values - r.flow_dt.reindex(previous_idx).values,
            index=current_idx, columns=r.flow_dt.columns, dtype=float)
        r.ramp_dt = change.reindex(r.flow_dt.index)

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
