import pandas as pd


def _invest_columns(par, s, v, entities) -> tuple[pd.Series, pd.Series]:
    """``(invested, in_construction)`` in MW (MWh for storage) by ``(entity, period)`` for ``entities``.

    ``invested`` is the capacity that comes online in the period, ``in_construction`` the capacity decided but
    not yet operating.  Both are sums of the realized investment decisions ``v_invest[e, period_invest]`` ×
    unit size over the solve's lead-time sets ``edd_invest_commission`` / ``edd_invest_in_construction``
    (``(entity, period_invest, period)``).  Without a construction lead time those sets are empty: a decision
    comes online in its own period and nothing is in construction.  Over combined solves the decisions are the
    unioned realized ``v_invest``, each counted once.  A cell without a decision is empty.
    """
    empty = pd.Series(dtype=float)
    if v.invest.empty or len(v.invest.columns) == 0:
        return empty, empty
    ed = s.ed_invest[s.ed_invest.get_level_values('entity').isin(entities)]
    # Restrict to (entity, period) pairs that have a v_invest value in the
    # solve: s.ed_invest is the full possible-pairs union, so a never-
    # realized pair (e.g. (e, p2025) when only p2020 is realized in the
    # current solve) would raise a KeyError.
    unstacked = v.invest.unstack()
    ed = ed.intersection(unstacked.index)
    decision = unstacked[ed] * par.entity_unitsize[ed.get_level_values('entity').unique()]
    commission = s.edd_invest_commission
    if len(commission) == 0:
        return decision, empty
    decided = decision.rename('value').rename_axis(['entity', 'period_invest']).reset_index()

    def _by_period(sets: pd.MultiIndex) -> pd.Series:
        rows = sets[sets.get_level_values('entity').isin(entities)].to_frame(index=False)
        if rows.empty:
            return empty
        joined = rows.merge(decided, on=['entity', 'period_invest'], how='inner')
        return joined.groupby(['entity', 'period'], sort=False)['value'].sum(min_count=1)

    return _by_period(commission), _by_period(s.edd_invest_in_construction)


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
    # ``par.entity_pre_existing`` (the static pre-invest baseline), NOT
    # ``entity_all_existing`` (the cumulative chain-sum).  On a multi-
    # solve handoff (e.g. an invest solve feeding a dispatch solve) the
    # handoff folds the prior solve's ``v_invest`` into
    # ``entity_all_existing``; reading that here would misreport newly
    # invested capacity under "existing" in the dispatch solve.  The
    # invested capacity is reported in the ``invested`` column (this
    # solve's / roll's ``v_invest``) and the ``total`` column carries
    # the cumulative sum; ``existing`` stays at the original baseline.
    existing = par.entity_pre_existing[processes].unstack()
    results['existing'] = existing

    # Capacity coming online (invested) and in construction
    results['invested'], results['in_construction'] = _invest_columns(par, s, v, s.process_unit)

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
    results = results[['existing', 'invested', 'in_construction', 'divested', 'total']]
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

    # Existing capacity - filter to process_connection only.  Uses
    # ``entity_pre_existing`` (pre-invest baseline); see unit_capacity
    # above for why carried-forward invest must not land here.
    existing = par.entity_pre_existing[connections].unstack()
    results['existing'] = existing

    # Capacity coming online (invested) and in construction
    results['invested'], results['in_construction'] = _invest_columns(par, s, v, s.process_connection)

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
    results = results[['existing', 'invested', 'in_construction', 'divested', 'total']]

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

    # Existing capacity - filter to node_state only.  Uses
    # ``entity_pre_existing`` (pre-invest baseline); see unit_capacity
    # above for why carried-forward invest must not land here.
    if nodes:
        existing = par.entity_pre_existing[nodes].unstack()
        results['existing'] = existing
    else:
        results['existing'] = pd.Series(dtype=float)

    # Capacity coming online (invested) and in construction
    results['invested'], results['in_construction'] = _invest_columns(par, s, v, s.node)

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

    results = results[['existing', 'invested', 'in_construction', 'divested', 'total']]
    return results, 'node_capacity_ed_p'
