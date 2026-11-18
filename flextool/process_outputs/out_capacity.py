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


def _with_built_divested(divested: pd.Series, par, v, members, names) -> pd.Series:
    """``divested`` ((entity, period) MW of the existing capacity retired in the period) plus the model-built
    capacity of ``members`` retired then (``v.divest_built`` × unitsize, read_highs_solution.write_v_divest_built),
    on the index level ``names`` of the capacity table; ``divested`` unchanged without built retirements of these
    entities."""
    built = v.divest_built
    cols = [e for e in built.columns if e in set(members)] if not built.empty else []
    if not cols:
        return divested
    mw = built[cols].mul(par.entity_unitsize[cols]).unstack()
    mw.index = mw.index.set_names(list(names))
    if divested.empty:
        return mw
    return divested.add(mw, fill_value=0.0)


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
    results['divested'] = _with_built_divested(results['divested'], par, v, s.process_unit, results.index.names)

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
    results['divested'] = _with_built_divested(results['divested'], par, v, s.process_connection, results.index.names)

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
    results['divested'] = _with_built_divested(results['divested'], par, v, s.node, results.index.names)

    # Total capacity - filter to node_state only
    if nodes:
        results['total'] = r.entity_all_capacity[nodes].unstack()
    else:
        results['total'] = pd.Series(dtype=float)

    results = results[['existing', 'invested', 'in_construction', 'divested', 'total']]
    return results, 'node_capacity_ed_p'


#: The columns of the retirement tables: the capacity leaving service at the start of a period, by cause.
RETIREMENT_CAUSES = ("end_of_life", "economic", "replaced", "extended")
_COHORT_ROWS = ("cohort", "kind", "period_online", "period_end_of_life", "period")
_CLASSES = (("unit", "process_unit"), ("connection", "process_connection"), ("node", "node_state"))


def _realized_periods(s) -> list[str]:
    """The periods of the capacity tables (``d_realize_dispatch_or_invest``), sorted as they sort them."""
    periods = s.d_realize_dispatch_or_invest
    if hasattr(periods, 'get_level_values') and 'period' in (periods.names or []):
        periods = periods.get_level_values('period')
    return sorted({str(d) for d in periods})


def retirement_causes(cohorts, periods: list[str]):
    """``[e, d, end_of_life, economic, replaced, extended]`` (MW, MWh) of the cohort capacity ``cohorts`` (the
    union of :func:`flextool.engine.output._cohort_capacity.read_cohort_capacity` in the realized ``periods``):

        economic[e,d]    = Σ_c retired_in_period[c,d]
        extended[e,d]    = Σ_{c extension, d_on(c) = d} (capacity + retired)[c,d]
        replaced[e,d]    = Σ_{c replacement, d_on(c) = d} capacity[c,d]
        end_of_life[e,d] = Σ_k max(0, N[k,d⁻] − N[k,d] − econ[k,d]) − extended[e,d]

    k a block of capacity (a cohort; the plants of a reinvest_automatic chain are one block, the chain's root),
    N[k,d] its capacity in d (0 when out of service), econ[k,d] its part retired at the start of d, d⁻ the realized
    period before d (a block in service there).  A block that leaves service (its end of life, a fall of a
    period-map ``existing``) is end of life, less the capacity a life extension keeps; a rise of a period-map
    ``existing`` is no cause.  The first realized period has no end of life."""
    import polars as pl

    rows = cohorts.filter(pl.col("d").is_in(periods))
    if rows.height == 0:
        return pl.DataFrame(schema={"e": pl.Utf8, "d": pl.Utf8, **{c: pl.Float64 for c in RETIREMENT_CAUSES}})
    order = {d: i for i, d in enumerate(periods)}
    block = pl.when(pl.col("cohort").str.starts_with("gen:")).then(
        pl.col("cohort").str.strip_prefix("gen:").str.replace(r":\d+$", "")).otherwise(pl.col("cohort"))
    held = (rows.with_columns(block=block)
                .group_by("e", "block", "d").agg(pl.col("capacity").sum(), pl.col("retired_in_period").sum()))
    before = (held.with_columns(_i=pl.col("d").replace_strict(order, return_dtype=pl.Int64) + 1)
                  .filter(pl.col("_i") < len(periods))
                  .select("e", "block", d=pl.col("_i").replace_strict(dict(enumerate(periods)), return_dtype=pl.Utf8),
                          before=pl.col("capacity")))
    leaving = (before.join(held, on=["e", "block", "d"], how="left")
                     .select("e", "d", leaving=(pl.col("before") - pl.col("capacity").fill_null(0.0)
                                                - pl.col("retired_in_period").fill_null(0.0)).clip(lower_bound=0.0))
                     .group_by("e", "d").agg(pl.col("leaving").sum()))
    starts = pl.col("d_on") == pl.col("d")
    by_period = (rows.group_by("e", "d").agg(
        economic=pl.col("retired_in_period").sum(),
        extended=pl.when((pl.col("kind") == "extension") & starts)
                   .then(pl.col("capacity") + pl.col("retired")).otherwise(0.0).sum(),
        replaced=pl.when((pl.col("kind") == "replacement") & starts).then(pl.col("capacity")).otherwise(0.0).sum()))
    grid = rows.select("e").unique().join(pl.DataFrame({"d": periods}, schema={"d": pl.Utf8}), how="cross")
    return (grid.join(by_period, on=["e", "d"], how="left").join(leaving, on=["e", "d"], how="left")
                .with_columns(pl.col("economic", "extended", "replaced", "leaving").fill_null(0.0))
                .with_columns(end_of_life=(pl.col("leaving") - pl.col("extended")).clip(lower_bound=0.0) + 0.0)
                .with_columns(_o=pl.col("d").replace_strict(order, return_dtype=pl.Int64))
                .sort("e", "_o")
                .select("e", "d", *RETIREMENT_CAUSES))


def cohort_capacity(par, s, v, r, debug):
    """Capacity by cohort and retirements by cause, per class (unit / connection / node):

    * ``<class>_cohort_capacity_ed_p`` — rows (entity, cohort, kind, period_online, period_end_of_life, period), one
      column ``capacity``: each cohort's capacity in service in the period
      (``output._cohort_capacity.cohort_capacity``), a reinvest_automatic chain under the plant in service; summed
      over the cohorts it is the ``total`` of the capacity table;
    * ``<class>_retirement_ed_p`` — rows (entity, period), columns :data:`RETIREMENT_CAUSES`
      (:func:`retirement_causes`).

    Over the periods of the capacity tables; none without a cohort table."""
    import polars as pl

    frame = getattr(v, 'cohort_capacity', None)
    if frame is None or frame.height == 0:
        return []
    periods = _realized_periods(s)
    rows = frame.filter(pl.col("d").is_in(periods))
    causes = retirement_causes(frame, periods)
    results = []
    for name, members in _CLASSES:
        entities = [str(e) for e in getattr(s, members)]
        cap = (rows.filter(pl.col("e").is_in(entities))
                   .select(pl.col("e").alias(name), "cohort", "kind", pl.col("d_on").alias("period_online"),
                           pl.col("d_eol").fill_null("").alias("period_end_of_life"), pl.col("d").alias("period"),
                           "capacity")
                   .to_pandas())
        cap = cap.set_index([name, *_COHORT_ROWS])
        cap.columns.name = 'parameter'
        results.append((cap, f'{name}_cohort_capacity_ed_p'))
        gone = causes.filter(pl.col("e").is_in(entities)).rename({"e": name, "d": "period"}).to_pandas()
        gone = gone.set_index([name, 'period'])
        gone.columns.name = 'cause'
        results.append((gone, f'{name}_retirement_ed_p'))
    return results
