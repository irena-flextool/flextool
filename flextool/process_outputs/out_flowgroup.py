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

from flextool.process_outputs._annualize import annualize_dt_to_d


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
    """Flow-group indicator results by period and a signed timewise series.

    For each group in ``s.flowGroupIndicators`` two quantities are built per
    timestep:

    * ``magnitude_dt`` — the UNSIGNED sum of all member unit- and
      connection-flow magnitudes (``.abs()``).  It is annualised like the
      flows (× step duration × timestep weight, Σ over the period, ÷ the
      period's share of a year) to the annual energy; the average power is
      that energy over 8760 h.
    * ``net_dt`` — the SIGNED net flow into the group's nodes (into-node
      ``+``, out-of-node ``−``; matches ``calc_group_flows``/
      ``calc_connections``).  A unit leg whose ``node`` is the flow's *sink*
      counts ``+`` (delivery into the node); whose ``node`` is the *source*
      counts ``−`` (withdrawal).  Connection ``from_conn`` (into node) counts
      ``+`` and ``to_conn`` (out of node) counts ``−``.  No ``.abs()`` and no
      step-duration weighting — it is MW per timestep.

    Returns a list of two ``(frame, name)`` tuples:

    * ``flowGroup_gd_p`` — keyed ``(group, period)`` with columns
      ``cumulative_flow`` [MWh/a, annualised] and ``average_flow`` [MW].  A group with no
      matched member flows gets zeros rather than being dropped.
    * ``flowGroup_gd_t`` — keyed ``(group, period, time)`` with the single
      column ``net_flow`` [MW], the signed per-timestep net flow.
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
    net_rows: list[dict] = []

    for g in s.flowGroupIndicators:
        legs = _member_legs(s, g)  # MultiIndex[(process, node)]

        # Accumulate |flow| (MW) at each timestep (UNSIGNED magnitude path).
        magnitude_dt = pd.Series(0.0, index=dt_index)
        # Accumulate signed net flow (MW) into the group's nodes at each
        # timestep (into-node +, out-of-node −).
        net_dt = pd.Series(0.0, index=dt_index)

        if len(legs) and flow_sink_pairs is not None:
            # Unit flows — pick every (process, source, sink) column whose
            # (process, sink) or (process, source) is a member leg.
            sink_hits = flow_sink_pairs.isin(legs)
            source_hits = flow_source_pairs.isin(legs)
            is_unit = flow_cols.get_level_values('process').isin(s.process_unit)
            unit_mask = is_unit & (sink_hits | source_hits)
            if unit_mask.any():
                magnitude_dt = magnitude_dt.add(
                    r.flow_dt.loc[:, unit_mask].abs().sum(axis=1), fill_value=0.0,
                )

            # Signed: node==sink (unit delivers into the node) counts +,
            # node==source (unit withdraws from the node) counts −.
            unit_into = is_unit & sink_hits
            unit_out = is_unit & source_hits
            if unit_into.any():
                net_dt = net_dt.add(
                    r.flow_dt.loc[:, unit_into].sum(axis=1), fill_value=0.0,
                )
            if unit_out.any():
                net_dt = net_dt.sub(
                    r.flow_dt.loc[:, unit_out].sum(axis=1), fill_value=0.0,
                )

            # Connection flows — from_conn and to_conn are already keyed by
            # (process, node).  Pick every column whose key is a member leg.
            if not r.from_conn.empty:
                conn_legs_from = r.from_conn.columns.intersection(legs)
                if len(conn_legs_from):
                    magnitude_dt = magnitude_dt.add(
                        r.from_conn[conn_legs_from].abs().sum(axis=1), fill_value=0.0,
                    )
                    # from_conn flows into the node → +.
                    net_dt = net_dt.add(
                        r.from_conn[conn_legs_from].sum(axis=1), fill_value=0.0,
                    )
            if not r.to_conn.empty:
                conn_legs_to = r.to_conn.columns.intersection(legs)
                if len(conn_legs_to):
                    magnitude_dt = magnitude_dt.add(
                        r.to_conn[conn_legs_to].abs().sum(axis=1), fill_value=0.0,
                    )
                    # to_conn flows out of the node → −.
                    net_dt = net_dt.sub(
                        r.to_conn[conn_legs_to].sum(axis=1), fill_value=0.0,
                    )

        # MW → MWh per step, × timestep weight, Σ over the period, ÷ the
        # period's share of a year: annual energy (MWh/a), like the flows.
        cumulative_d = annualize_dt_to_d(
            magnitude_dt.to_frame('v'), par.timestep_weight,
            par.complete_period_share_of_year, step_dur)['v']
        # Reindex to the full realized-period set so groups with no matched
        # flows still produce a row per realized period.
        cumulative_d = cumulative_d.reindex(s.d_realized_period, fill_value=0.0)
        # Average power over the year: the annual energy over 8760 h (the
        # weighted mean of the sampled MW).
        average_d = cumulative_d.div(8760.0).fillna(0.0)

        for period in cumulative_d.index:
            rows.append({
                'group': g,
                'period': period,
                'cumulative_flow': float(cumulative_d.loc[period]),
                'average_flow': float(average_d.loc[period]),
            })

        # Signed per-timestep net flow (MW) — one row per realized (period,
        # time).  No step-duration weighting; this is instantaneous power.
        for (period, time), value in net_dt.items():
            net_rows.append({
                'group': g,
                'period': period,
                'time': time,
                'net_flow': float(value),
            })

    if not rows:
        return results

    frame = pd.DataFrame(rows).set_index(['group', 'period'])[
        ['cumulative_flow', 'average_flow']
    ]
    frame.columns.name = 'parameter'
    results.append((frame, 'flowGroup_gd_p'))

    if net_rows:
        net_frame = pd.DataFrame(net_rows).set_index(
            ['group', 'period', 'time'])[['net_flow']]
        net_frame.columns.name = 'parameter'
        results.append((net_frame, 'flowGroup_gd_t'))

    return results
