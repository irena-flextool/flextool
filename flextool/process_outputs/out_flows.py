import pandas as pd


def unit_outputNode(par, s, v, r, debug):
    """Unit output node flow for periods and time"""

    results = []

    if r.flow_dt.empty:
        return results

    # Calculate timestep-level results first
    result_multi_dt = pd.DataFrame(index=s.dt_realize_dispatch, columns=pd.MultiIndex.from_tuples([], names=['unit', 'node']))

    # Filter columns: unit processes that have sinks
    unit_sink_cols = r.flow_dt.columns[
        r.flow_dt.columns.get_level_values(0).isin(s.process_unit) &
        r.flow_dt.columns.to_series().apply(lambda col: (col[0], col[2]) in s.process_sink)
    ]

    for col in unit_sink_cols:
        u, source, sink = col
        result_multi_dt[(u, sink)] = r.flow_dt[col]

    # Return timestep results
    results.append((result_multi_dt, 'unit_outputNode_dt_ee'))

    # Aggregate to period level
    result_multi_d = result_multi_dt.groupby(level='period').sum()

    # Divide by period shares to annualize
    result_multi_d = result_multi_d.div(par.complete_period_share_of_year, axis=0)

    # Return period results
    results.append((result_multi_d, 'unit_outputNode_d_ee'))

    return results


def unit_inputNode(par, s, v, r, debug):
    """Unit input node flow for periods and time"""

    results = []

    if r.flow_dt.empty:
        return results

    # Calculate timestep-level results first
    result_multi_dt = pd.DataFrame(index=s.dt_realize_dispatch, columns=pd.MultiIndex.from_tuples([], names=['unit', 'node']))

    # Filter columns: unit processes that have sources
    unit_source_cols = r.flow_dt.columns[
        r.flow_dt.columns.get_level_values(0).isin(s.process_unit) &
        r.flow_dt.columns.to_series().apply(lambda col: (col[0], col[1]) in s.process_source)
    ]

    for col in unit_source_cols:
        u, source, sink = col
        result_multi_dt[(u, source)] = -r.flow_dt[col]

    # Return timestep results
    results.append((result_multi_dt, 'unit_inputNode_dt_ee'))

    # Aggregate to period level
    result_multi_d = result_multi_dt.groupby(level='period').sum()

    # Divide by period shares to annualize
    result_multi_d = result_multi_d.div(par.complete_period_share_of_year, axis=0)

    # Return period results
    results.append((result_multi_d, 'unit_inputNode_d_ee'))

    return results


def unit_cf_outputNode(par, s, v, r, debug):
    """Unit capacity factors by output node for periods"""
    complete_hours = par.complete_period_share_of_year * 8760
    unit_cols = r.process_sink_flow_d.columns[r.process_sink_flow_d.columns.get_level_values(0).isin(s.process_unit)]
    unit_capacity = r.entity_all_capacity[unit_cols.droplevel(1).unique()].rename_axis('process', axis=1)
    unit_capacity.columns = unit_capacity.columns.get_level_values(0)
    results = r.process_sink_flow_d[unit_cols].div(unit_capacity, level=0).div(complete_hours, axis=0)
    results.columns.names = ['unit', 'sink']
    return results, 'unit_outputs_cf_d_ee'


def unit_cf_inputNode(par, s, v, r, debug):
    """Unit capacity factors by input node for periods"""
    # !!! This should account for efficiency losses in direct conversion units (but it does not)
    complete_hours = par.complete_period_share_of_year * 8760
    unit_source = r.process_source_flow_d.columns[r.process_source_flow_d.columns.get_level_values(0).isin(s.process_unit)]
    unit_capacity = r.entity_all_capacity[unit_source.droplevel(1).unique()].rename_axis('process', axis=1)
    unit_capacity.columns = unit_capacity.columns.get_level_values(0)
    results = r.process_source_flow_d[unit_source].div(unit_capacity, level=0).div(complete_hours, axis=0)
    results.columns.names = ['unit', 'source']
    return results, 'unit_inputs_cf_d_ee'


def unit_VRE_curtailment_and_potential(par, s, v, r, debug):
    """Unit VRE curtailment and potential for both periods and timesteps"""

    results = []
    vre_processes = s.process_VRE.unique()

    # Timestep-level curtailment (absolute values) - calculate first
    if not r.flow_dt.empty and not r.potentialVREgen_dt.empty:
        curtail_dt = pd.DataFrame(index=s.dt_realize_dispatch, columns=pd.MultiIndex.from_tuples([], names=['unit', 'sink']))
        potential_dt = pd.DataFrame(index=s.dt_realize_dispatch, columns=pd.MultiIndex.from_tuples([], names=['unit', 'sink']))

        for col in r.flow_dt.columns:
            u, source, sink = col
            if u in vre_processes and (u, sink) in s.process_sink and (u, sink) in r.potentialVREgen_dt.columns:
                curtail_dt[u, sink] = r.potentialVREgen_dt[(u, sink)] - r.flow_dt[col]
                potential_dt[u, sink] = r.potentialVREgen_dt[(u, sink)]

        results.append((curtail_dt, 'unit_curtailment_outputNode_dt_ee'))
        results.append((potential_dt, 'unit_VRE_potential_outputNode_dt_ee'))

        # Calculate curtailment share at timestep level
        curtail_share_dt = (curtail_dt / potential_dt).where(potential_dt != 0, 0)
        results.append((curtail_share_dt, 'unit_curtailment_share_outputNode_dt_ee'))

        # Aggregate to period level
        curtail_period = curtail_dt.groupby(level='period').sum()
        potential_period = potential_dt.groupby(level='period').sum()

        # Calculate curtailment share at period level
        curtail_share_period = (curtail_period / potential_period).where(potential_period != 0, 0)

        results.append((curtail_period, 'unit_curtailment_outputNode_d_ee'))
        results.append((curtail_share_period, 'unit_curtailment_share_outputNode_d_ee'))
        results.append((potential_period, 'unit_VRE_potential_outputNode_d_ee'))

    return results


def unit_ramps(par, s, v, r, debug):
    """Unit ramps by input and output nodes for timesteps"""
    results = []
    if r.ramp_dtt.empty:
        return results

    # Output node ramps
    process_sink_ramp_output = s.process_sink[s.process_sink.get_level_values(0).isin(s.process_unit)]
    pss_ramp_output = r.ramp_dtt.columns[r.ramp_dtt.columns.droplevel(1).isin(process_sink_ramp_output)]
    ramp_output = r.ramp_dtt[pss_ramp_output].droplevel('t_previous')
    ramp_output.columns = ramp_output.columns.droplevel(1)  # Remove 'source' from (process, source, sink) to get (unit, source)
    ramp_output.columns.names = ['unit', 'sink']
    results.append((ramp_output, 'unit_ramp_outputs_dt_ee'))

    # Input node ramps
    process_source_ramp_input = s.process_source[s.process_source.get_level_values(0).isin(s.process_unit)]
    pss_ramp_input = r.ramp_dtt.columns[r.ramp_dtt.columns.droplevel(2).isin(process_source_ramp_input)]
    ramp_input = r.ramp_dtt[pss_ramp_input].droplevel('t_previous')
    ramp_input.columns = ramp_input.columns.droplevel(2)  # Remove 'sink' from (process, source, sink) to get (unit, source)
    ramp_input.columns.names = ['unit', 'source']
    results.append((ramp_input, 'unit_ramp_inputs_dt_ee'))

    return results


def unit_online_and_startup(par, s, v, r, debug):
    """Unit online status and startups for timesteps and periods"""
    results = []

    # 1. Online units dt
    online_units_dt = r.process_online_dt[s.process_unit.intersection(s.process_online)]
    results.append((online_units_dt, 'unit_online_dt_e'))

    # 2. Average online status at period level (weighted by step_duration)
    complete_hours = par.complete_period_share_of_year * 8760
    online_units_d = online_units_dt.mul(par.step_duration, axis=0).groupby('period').sum().div(complete_hours, axis=0)
    results.append((online_units_d, 'unit_online_average_d_e'))

    # 3. Startups aggregated to period level
    startup_units_d = r.process_startup_dt[s.process_unit.intersection(s.process_online)].groupby('period').sum()
    results.append((startup_units_d, 'unit_startup_d_e'))

    return results
