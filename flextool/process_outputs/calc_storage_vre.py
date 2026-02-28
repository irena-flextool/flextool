import pandas as pd


def compute_storage_and_vre(par, s, v, r) -> None:
    """Compute storage state changes, self-discharge, node inflow, VRE potential, and storage usage."""
    step_duration = par.step_duration
    unitsize = par.entity_unitsize

    # node_state_change_dt (four storage binding methods)
    current_idx = s.dtt.droplevel('t_previous')
    prev_period_idx = (
        s.dtttdt
        .droplevel(['time', 't_previous_within_timeset', 'd_previous', 't_previous_within_solve'])
        .set_names(['period', 'time'])
    )
    prev_timeset_idx = (
        s.dtttdt
        .droplevel(['time', 't_previous', 'd_previous', 't_previous_within_solve'])
        .set_names(['period', 'time'])
    )
    prev_forward_only_idx = (
        s.dtttdt
        .droplevel(['period', 'time', 't_previous', 't_previous_within_timeset'])
        .set_names(['period', 'time'])
    )
    exclude_idx = s.period__time_first[
        s.period__time_first.get_level_values('period').isin(s.period_first_of_solve)
    ]

    r_state_change = pd.DataFrame(0.0, index=s.dt_realize_dispatch, columns=s.node_state, dtype=float)
    for n in s.node_state:
        if n not in v.state.columns:
            continue

        state_change = pd.Series(0.0, index=current_idx)
        v_current = v.state[n].squeeze()
        v_prev_period = pd.Series(v.state[n].squeeze().reindex(prev_period_idx).values, index=current_idx)
        v_prev_timeblock = pd.Series(v.state[n].squeeze().reindex(prev_timeset_idx).values, index=current_idx)
        v_forward = pd.Series(v.state[n].squeeze().reindex(prev_forward_only_idx).values, index=current_idx)

        if (n, 'bind_forward_only') in s.node__storage_binding_method:
            mask = ~current_idx.isin(exclude_idx)
            state_change += ((v_current - v_forward) * unitsize[n]).where(mask, 0)

        if (n, 'bind_within_solve') in s.node__storage_binding_method:
            state_change += (v_current - v_forward) * unitsize[n]

        if (n, 'bind_within_period') in s.node__storage_binding_method:
            state_change += (v_current - v_prev_period) * unitsize[n]

        if (n, 'bind_within_timeset') in s.node__storage_binding_method:
            state_change += (v_current - v_prev_timeblock) * unitsize[n]

        r_state_change[n] = state_change

    r.node_state_change_dt = r_state_change
    r.node_state_change_d = r.node_state_change_dt.groupby(level='period').sum()

    # self_discharge_loss
    r.self_discharge_loss_dt = (
        v.state[par.node_self_discharge_loss.columns]
        .mul(par.node_self_discharge_loss, axis=1, level=0)
        .mul(unitsize[par.node_self_discharge_loss.columns], axis='columns', level=0)
    )
    r.self_discharge_loss_d = r.self_discharge_loss_dt.mul(step_duration, axis=0).groupby('period').sum()

    # node_inflow_d
    r.node_inflow_d = par.node_inflow.groupby('period').sum().div(par.complete_period_share_of_year, axis=0)

    # potentialVREgen
    vre_with_sink = s.process_VRE[s.process_VRE.isin(s.process_sink)]
    vre_node_profile = vre_with_sink.join(s.process__node__profile__profile_method)
    vre_node_profile_upper = vre_node_profile[
        vre_node_profile.get_level_values('profile_method').isin(['upper_limit'])
    ]
    vre_profiles_in_use = par.profile[vre_node_profile_upper.get_level_values('profile').unique()]
    profile_level_of_vre_node_profile_upper = vre_node_profile_upper.get_level_values('profile')
    vre_processes_in_use = vre_profiles_in_use[profile_level_of_vre_node_profile_upper]
    vre_processes_in_use.columns = vre_node_profile_upper
    r.potentialVREgen_dt = (
        vre_processes_in_use
        .mul(par.process_availability)
        .mul(r.entity_all_capacity, axis=1, level=0)
        .droplevel(axis=1, level=['profile', 'profile_method'])
    )
    r.potentialVREgen_d = r.potentialVREgen_dt.groupby('period').sum()

    # storage_usage_dt (for nested model storage fixing)
    dt_fix_idx = s.dt_fix_storage_timesteps
    r_storage_usage = {}
    for n in s.node:
        if (n, 'fix_usage') in s.node__storage_nested_fix_method:
            usage = pd.Series(0.0, index=dt_fix_idx)
            for col in r.flow_dt.columns:
                p, source, sink = col
                if source == n:
                    usage += (r.flow_dt[col].reindex(dt_fix_idx, fill_value=0) *
                              par.step_duration.reindex(dt_fix_idx, fill_value=0))
                if sink == n:
                    usage -= (r.flow_dt[col].reindex(dt_fix_idx, fill_value=0) *
                              par.step_duration.reindex(dt_fix_idx, fill_value=0))
            r_storage_usage[n] = usage
    r.storage_usage_dt = (
        pd.DataFrame(r_storage_usage, dtype=float)
        if r_storage_usage
        else pd.DataFrame(0.0, index=dt_fix_idx, columns=[])
    )
