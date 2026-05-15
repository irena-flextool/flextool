import pandas as pd


def unit_capacity(par, s, v, r, debug):
    """Unit capacity by period"""

    # Get all periods and filter to process_unit entities.  Sort
    # periods deterministically: the nested cascade unions
    # ``d_realize_dispatch_or_invest`` across rolls and the invest
    # parent step's invest_periods come from a Polars ``.unique()``
    # call whose row order is hash-partitioned and unstable across
    # processes (see read_sets.py invest_periods_list build).  An
    # explicit sort here is the row-ordering guarantee the period-
    # indexed CSV writers depend on.
    if hasattr(s.d_realize_dispatch_or_invest, 'get_level_values') \
            and 'period' in (s.d_realize_dispatch_or_invest.names or []):
        periods = sorted(set(
            s.d_realize_dispatch_or_invest.get_level_values('period')
        ))
    else:
        periods = sorted(set(s.d_realize_dispatch_or_invest))
    processes = list(s.process_unit)

    # Create base dataframe with all combinations (unit, period order)
    index = pd.MultiIndex.from_product([processes, periods], names=['unit', 'period'])
    results = pd.DataFrame(index=index)
    results.columns.name = 'parameter'

    # Existing capacity - filter to process_unit only.  Uses
    # ``par.entity_all_existing`` which read_parameters_multi has
    # filtered per-step to each solve's own realized periods (so the
    # nested-cascade parent invest step's pre_existing baseline wins
    # over the children's later_existing on shared (entity, period)
    # cells — see the comment block in
    # ``read_parameters.py:read_parameters_multi``).
    existing = par.entity_all_existing[processes].unstack()
    results['existing'] = existing

    # Invested capacity - default to None, overwrite if data exists
    results['invested'] = pd.Series(dtype=float)
    if not v.invest.empty and len(v.invest.columns) > 0:
        ed_unit_invest = s.ed_invest[s.ed_invest.get_level_values('entity').isin(s.process_unit)]
        # Filter to (entity, period) pairs that actually have a v_invest value
        # in this solve.  Without the per-solve solve__ed_invest filter that
        # v3.32.0 used, s.ed_invest is the full possible-pairs union; restrict
        # here to avoid KeyErrors on never-realized pairs (e.g. (e, p2025)
        # when only p2020 is realized in the current solve).
        unstacked_invest = v.invest.unstack()
        ed_unit_invest = ed_unit_invest.intersection(unstacked_invest.index)
        unit_invest = ed_unit_invest.get_level_values('entity').unique()
        results['invested'] = unstacked_invest[ed_unit_invest] * par.entity_unitsize[unit_invest]

    # Divested capacity - default to None, overwrite if data exists
    results['divested'] = pd.Series(dtype=float)
    if not v.divest.empty and len(v.divest.columns) > 0:
        ed_unit_divest = s.ed_divest[s.ed_divest.get_level_values('entity').isin(s.process_unit)]
        unstacked_divest = v.divest.unstack()
        ed_unit_divest = ed_unit_divest.intersection(unstacked_divest.index)
        unit_divest = ed_unit_divest.get_level_values('entity').unique()
        results['divested'] = unstacked_divest[ed_unit_divest] * par.entity_unitsize[unit_divest]

    # Total capacity - filter to process_unit only
    total = r.entity_all_capacity[processes].unstack()
    results['total'] = total
    results = results[['existing', 'invested', 'divested', 'total']]
    return results, 'unit_capacity_ed_p'


def connection_capacity(par, s, v, r, debug):
    """Connection capacity by period"""

    # Get all periods and filter to process_connection entities.
    # See unit_capacity above for the deterministic-period-order rationale.
    if hasattr(s.d_realize_dispatch_or_invest, 'get_level_values') \
            and 'period' in (s.d_realize_dispatch_or_invest.names or []):
        periods = sorted(set(
            s.d_realize_dispatch_or_invest.get_level_values('period')
        ))
    else:
        periods = sorted(set(s.d_realize_dispatch_or_invest))
    connections = list(s.process_connection)

    # Create base dataframe with all combinations (connection, period order)
    index = pd.MultiIndex.from_product([connections, periods], names=['connection', 'period'])
    results = pd.DataFrame(index=index)
    results.columns.name = 'parameter'

    # Existing capacity - filter to process_connection only.  See
    # unit_capacity above for the read_parameters_multi per-step
    # filter that ensures correct dedup semantics on nested cascades.
    existing = par.entity_all_existing[connections].unstack()
    results['existing'] = existing

    # Invested capacity - default to empty, overwrite if data exists
    results['invested'] = pd.Series(dtype=float)
    if not v.invest.empty and len(v.invest.columns) > 0:
        ed_conn_invest = s.ed_invest[s.ed_invest.get_level_values('entity').isin(s.process_connection)]
        # See unit_capacity above for why we intersect with v.invest's columns.
        unstacked_invest = v.invest.unstack()
        ed_conn_invest = ed_conn_invest.intersection(unstacked_invest.index)
        conn_invest = ed_conn_invest.get_level_values('entity').unique()
        results['invested'] = unstacked_invest[ed_conn_invest] * par.entity_unitsize[conn_invest]

    # Divested capacity - default to empty, overwrite if data exists
    results['divested'] = pd.Series(dtype=float)
    if not v.divest.empty and len(v.divest.columns) > 0:
        ed_conn_divest = s.ed_divest[s.ed_divest.get_level_values('entity').isin(s.process_connection)]
        unstacked_divest = v.divest.unstack()
        ed_conn_divest = ed_conn_divest.intersection(unstacked_divest.index)
        conn_divest = ed_conn_divest.get_level_values('entity').unique()
        results['divested'] = unstacked_divest[ed_conn_divest] * par.entity_unitsize[conn_divest]  # was: results['invested']

    # Total capacity - filter to process_connection only
    results['total'] = r.entity_all_capacity[connections].unstack()

    # Reorder columns
    results = results[['existing', 'invested', 'divested', 'total']]

    return results, 'connection_capacity_ed_p'


def node_capacity(par, s, v, r, debug):
    """Node capacity by period"""

    # Get all periods and filter to node_state entities.
    # See unit_capacity above for the deterministic-period-order rationale.
    if hasattr(s.d_realize_dispatch_or_invest, 'get_level_values') \
            and 'period' in (s.d_realize_dispatch_or_invest.names or []):
        periods = sorted(set(
            s.d_realize_dispatch_or_invest.get_level_values('period')
        ))
    else:
        periods = sorted(set(s.d_realize_dispatch_or_invest))
    nodes = list(s.node_state)

    # Create base dataframe with all combinations (node, period order)
    index = pd.MultiIndex.from_product([nodes, periods], names=['node', 'period'])
    results = pd.DataFrame(index=index)
    results.columns.name = 'parameter'

    # Existing capacity - filter to node_state only.  See unit_capacity
    # above for the read_parameters_multi per-step filter.
    if nodes:
        existing = par.entity_all_existing[nodes].unstack()
        results['existing'] = existing
    else:
        results['existing'] = pd.Series(dtype=float)

    # Invested capacity - default to empty, overwrite if data exists
    results['invested'] = pd.Series(dtype=float)
    if not v.invest.empty and len(v.invest.columns) > 0:
        ed_node_invest = s.ed_invest[s.ed_invest.get_level_values('entity').isin(s.node)]
        # See unit_capacity above for why we intersect with v.invest's columns.
        unstacked_invest = v.invest.unstack()
        ed_node_invest = ed_node_invest.intersection(unstacked_invest.index)
        node_invest = ed_node_invest.get_level_values('entity').unique()
        results['invested'] = unstacked_invest[ed_node_invest] * par.entity_unitsize[node_invest]

    # Divested capacity - default to empty, overwrite if data exists
    results['divested'] = pd.Series(dtype=float)
    if not v.divest.empty and len(v.divest.columns) > 0:
        ed_node_divest = s.ed_divest[s.ed_divest.get_level_values('entity').isin(s.node)]
        unstacked_divest = v.divest.unstack()
        ed_node_divest = ed_node_divest.intersection(unstacked_divest.index)
        node_divest = ed_node_divest.get_level_values('entity').unique()
        results['divested'] = unstacked_divest[ed_node_divest] * par.entity_unitsize[node_divest]  # was: v.invest, results['invested']

    # Total capacity - filter to node_state only
    if nodes:
        results['total'] = r.entity_all_capacity[nodes].unstack()
    else:
        results['total'] = pd.Series(dtype=float)

    results = results[['existing', 'invested', 'divested', 'total']]
    return results, 'node_capacity_ed_p'
