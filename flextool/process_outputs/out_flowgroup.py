"""Output functions for flow group results.

Groups that have ``output_flowGroup_indicators = yes`` appear in
``s.flowGroupIndicators``.  Their member flows are listed in
``s.group_process_node`` — each tuple ``(group, process, node)`` identifies
one unit-flow or connection-flow leg that should be summed into the group.

The metrics emitted here are intentionally minimal (a stub); richer
direction-aware or technology-aware metrics are future work.
"""
from __future__ import annotations

import pandas as pd


def _member_legs(s, g: str) -> pd.MultiIndex:
    """Return the ``(process, node)`` legs belonging to group ``g``.

    ``s.group_process_node`` is a ``(group, process, node)`` MultiIndex;
    filtering by the ``group`` level gives the flows aggregated into this
    group.  The returned index drops the ``group`` level so callers can
    match against ``r.flow_dt``/``r.from_conn``/``r.to_conn`` columns
    (which are ``(process, node)`` after de-duplication).
    """
    gpn = s.group_process_node
    mask = gpn.get_level_values('group') == g
    return gpn[mask].droplevel('group').unique()


def flowGroup_indicators(par, s, v, r, debug):
    """Flow-group indicator results by period.

    For each group in ``s.flowGroupIndicators``, sum the unsigned magnitudes
    of all member unit- and connection-flows, weight by step duration to get
    energy per period, and divide by period hours for the average power.

    Returns a list ``[(frame, 'flowGroup_gd_p')]`` keyed by
    ``(group, period)`` with columns ``cumulative_flow`` [MWh] and
    ``average_flow`` [MW].  A group with no matched member flows gets
    zeros rather than being dropped.
    """
    results: list[tuple[pd.DataFrame, str]] = []

    if not list(s.flowGroupIndicators) or s.dt_realize_dispatch.empty:
        return results

    step_dur = par.step_duration  # Series indexed by (period, time), MW→MWh weight
    dt_index = s.dt_realize_dispatch

    # Pre-compute lookups of member columns.  ``r.flow_dt`` is keyed by
    # (process, source, sink) — a unit flow touches a group's (process,
    # node) membership whenever the node is the source *or* the sink.
    # ``r.from_conn``/``r.to_conn`` are already keyed by (process, node),
    # so membership is a direct column pick.
    flow_cols = r.flow_dt.columns
    flow_sink_pairs = None
    flow_source_pairs = None
    if not r.flow_dt.empty:
        flow_sink_pairs = pd.MultiIndex.from_arrays(
            [flow_cols.get_level_values('process'), flow_cols.get_level_values('sink')],
            names=['process', 'node'],
        )
        flow_source_pairs = pd.MultiIndex.from_arrays(
            [flow_cols.get_level_values('process'), flow_cols.get_level_values('source')],
            names=['process', 'node'],
        )

    rows: list[dict] = []
    # Period hours = share of year * 8760 (same convention used by
    # out_flows.unit_online_and_startup for period-hour normalization).
    period_hours = par.complete_period_share_of_year.mul(8760.0)

    for g in s.flowGroupIndicators:
        legs = _member_legs(s, g)  # MultiIndex[(process, node)]

        # Accumulate |flow| (MW) at each timestep.
        magnitude_dt = pd.Series(0.0, index=dt_index)

        if len(legs) and flow_sink_pairs is not None:
            # Unit flows — pick every (process, source, sink) column whose
            # (process, sink) or (process, source) is a member leg.
            sink_hits = flow_sink_pairs.isin(legs)
            source_hits = flow_source_pairs.isin(legs)
            unit_mask = (
                flow_cols.get_level_values('process').isin(s.process_unit)
                & (sink_hits | source_hits)
            )
            if unit_mask.any():
                magnitude_dt = magnitude_dt.add(
                    r.flow_dt.loc[:, unit_mask].abs().sum(axis=1), fill_value=0.0,
                )

            # Connection flows — from_conn and to_conn are already keyed by
            # (process, node).  Pick every column whose key is a member leg.
            if not r.from_conn.empty:
                conn_legs_from = r.from_conn.columns.intersection(legs)
                if len(conn_legs_from):
                    magnitude_dt = magnitude_dt.add(
                        r.from_conn[conn_legs_from].abs().sum(axis=1), fill_value=0.0,
                    )
            if not r.to_conn.empty:
                conn_legs_to = r.to_conn.columns.intersection(legs)
                if len(conn_legs_to):
                    magnitude_dt = magnitude_dt.add(
                        r.to_conn[conn_legs_to].abs().sum(axis=1), fill_value=0.0,
                    )

        # Weight MW → MWh per step, then sum to period level.
        energy_dt = magnitude_dt.mul(step_dur, fill_value=0.0)
        cumulative_d = energy_dt.groupby(level='period').sum()
        # Reindex to the full realized-period set so groups with no matched
        # flows still produce a row per realized period.
        cumulative_d = cumulative_d.reindex(s.d_realized_period, fill_value=0.0)

        hours_d = period_hours.reindex(cumulative_d.index)
        average_d = cumulative_d.div(hours_d.where(hours_d != 0, pd.NA))
        average_d = average_d.fillna(0.0)

        for period in cumulative_d.index:
            rows.append({
                'group': g,
                'period': period,
                'cumulative_flow': float(cumulative_d.loc[period]),
                'average_flow': float(average_d.loc[period]),
            })

    if not rows:
        return results

    frame = pd.DataFrame(rows).set_index(['group', 'period'])[
        ['cumulative_flow', 'average_flow']
    ]
    frame.columns.name = 'parameter'
    results.append((frame, 'flowGroup_gd_p'))
    return results
