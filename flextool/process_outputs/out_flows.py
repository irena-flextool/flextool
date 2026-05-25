import pandas as pd


# Threshold below which LP residuals on flow variables are treated as exact
# zero.  HiGHS's primal feasibility tolerance is 1e-8 (see HiGHS options),
# so any output below 1e-9 is conclusively below solver precision and
# represents zero.  Without this clip the wide-format flow CSVs round-trip
# through ``pd.read_csv`` as ``float64`` while integer-zero goldens load
# as ``int64``, causing ``pd.testing.assert_frame_equal`` to fail on dtype
# even when values match under ``round_for_comparison``'s 4-decimal
# threshold.
#
# Apply ONLY to the wide output frames destined for CSV write — never to
# ``r.flow_dt`` itself, because curtailment derives ``potentialVRE - flow``
# and small physically-meaningful curtailment values could otherwise be
# masked.
_FLOW_RESIDUAL_TOLERANCE: float = 1e-9


def _clip_flow_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """Snap sub-femto LP residuals to exactly zero.

    Returns a frame where every cell with ``|x| < 1e-9`` is replaced with
    ``0.0``.  Empty / all-NaN frames are returned unchanged.
    """
    if df.empty:
        return df
    return df.mask(df.abs() < _FLOW_RESIDUAL_TOLERANCE, 0.0)


def _indirect_units(s) -> set:
    """Return set of units classified as indirect (2-arc conversion form).

    An indirect unit has BOTH a row where ``source == p`` and a row where
    ``sink == p`` in ``process_source_sink`` — i.e. the topology was split
    into separate input-side and output-side arcs.  Direct-method and
    profile/noConversion units have at most one of those, never both.

    This mirrors v3.32.0's ordering convention for the wide-pivot output
    CSVs (``unit__outputNode__dt.csv`` etc.): direct-method units listed
    first (in alphabetical order), then indirect units.  v3.32.0 inherited
    the partition from the legacy ``write_process_arc_unions`` union order
    (process_arc_unions.py L248-263) which placed ``*_direct.csv`` sets
    ahead of the indirect/noConversion ones.
    """
    if s.process_source_sink is None or len(s.process_source_sink) == 0:
        return set()
    procs = s.process_source_sink.get_level_values('process')
    srcs = s.process_source_sink.get_level_values('source')
    snks = s.process_source_sink.get_level_values('sink')
    as_source = {p for p, src in zip(procs, srcs) if src == p}
    as_sink = {p for p, snk in zip(procs, snks) if snk == p}
    return as_source & as_sink


def _sort_unit_node_columns(df: pd.DataFrame, indirect: set) -> pd.DataFrame:
    """Reorder a (unit, node) MultiIndex column frame to match v3.32.0.

    Partition: direct units first, indirect units last.  Alphabetical by
    (unit, node) within each partition.
    """
    if df.empty or df.columns.empty:
        return df
    sorted_cols = sorted(
        df.columns,
        key=lambda c: (c[0] in indirect, c[0], c[1]),
    )
    return df.reindex(columns=pd.MultiIndex.from_tuples(
        sorted_cols, names=df.columns.names,
    ))


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

    # Reorder columns to match v3.32.0's direct-then-indirect / alphabetical
    # ordering — replaces the upstream ``flow_dt.columns`` order which is
    # non-deterministic at HEAD (polars ``unique()`` after sort).
    indirect = _indirect_units(s)
    result_multi_dt = _sort_unit_node_columns(result_multi_dt, indirect)

    # Snap sub-femto HiGHS LP residuals to exact zero before emitting so the
    # CSV round-trip preserves the int dtype of zero-valued columns.
    result_multi_dt = _clip_flow_residuals(result_multi_dt)

    # Return timestep results
    results.append((result_multi_dt, 'unit_outputNode_dt_ee'))

    # Aggregate to period level.  v_flow is in MW (MWh/h) — multiply by
    # step_duration to get MWh per step, then sum to per-period MWh.
    result_multi_d = (
        result_multi_dt.mul(par.step_duration, axis=0)
        .groupby(level='period').sum()
    )

    # Divide by period shares to annualize
    result_multi_d = result_multi_d.div(par.complete_period_share_of_year, axis=0)
    result_multi_d = _clip_flow_residuals(result_multi_d)

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

    # Reorder columns to match v3.32.0's direct-then-indirect / alphabetical
    # ordering (see ``unit_outputNode`` for rationale).
    indirect = _indirect_units(s)
    result_multi_dt = _sort_unit_node_columns(result_multi_dt, indirect)

    # Snap sub-femto HiGHS LP residuals to exact zero (see ``unit_outputNode``).
    result_multi_dt = _clip_flow_residuals(result_multi_dt)

    # Return timestep results
    results.append((result_multi_dt, 'unit_inputNode_dt_ee'))

    # Aggregate to period level.  v_flow is MW — multiply by step_duration
    # to get MWh per step before summing.
    result_multi_d = (
        result_multi_dt.mul(par.step_duration, axis=0)
        .groupby(level='period').sum()
    )

    # Divide by period shares to annualize
    result_multi_d = result_multi_d.div(par.complete_period_share_of_year, axis=0)
    result_multi_d = _clip_flow_residuals(result_multi_d)

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

        # Aggregate to period level.  curtail_dt / potential_dt are MW —
        # multiply by step_duration to get MWh per step before summing.
        curtail_period = (
            curtail_dt.mul(par.step_duration, axis=0)
            .groupby(level='period').sum()
        )
        potential_period = (
            potential_dt.mul(par.step_duration, axis=0)
            .groupby(level='period').sum()
        )

        # Calculate curtailment share at period level (ratio — step_duration cancels)
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

    # 3. Startups annualized to period level.  Raw startups only cover modeled
    # timesteps; weight each by p_rp_cost_weight and divide by the period share
    # of a year so the value is an annual count (same convention as flows).
    units_online = s.process_unit.intersection(s.process_online)
    startup_weighted = r.process_startup_dt[units_online].mul(par.rp_cost_weight, axis=0)
    startup_units_d = (
        startup_weighted.groupby('period').sum()
        .div(par.complete_period_share_of_year, axis=0)
    )
    results.append((startup_units_d, 'unit_startup_d_e'))

    return results
