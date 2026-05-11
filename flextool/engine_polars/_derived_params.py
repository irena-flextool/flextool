"""Γ.3.A — Foundational Derived Param helpers.

This module covers the first batch of Derived Params per the
``audit/db_direct_param_map.md`` §3 deep-dive:

* ``§3.1`` Time / weighting:
  ``dt`` / ``p_step_duration``, ``p_rp_cost_weight``,
  ``p_inflation_op``, ``p_period_share``.
* ``§3.2`` Nodes:
  ``p_inflow`` (scaling step on top of node.inflow Map).
* ``§3.6`` Profiles:
  ``p_profile_value`` (multi-tier alternative cascade),
  ``p_process_existing_count``.
* ``§3.18`` Stochastic:
  ``pdt_branch_weight`` / ``pd_branch_weight``.

Each helper takes an :class:`flextool._input_source.InputSource` plus
any already-computed sibling frames it depends on (foundational order
established in §7.3 Phase Γ.3a) and returns the eager
``Param`` / ``DataFrame`` shape consumed by downstream model-build.

Lazy-evaluation pattern
-----------------------
``InputSource.parameter()`` collects at the source boundary.  Helpers
compose multi-call results via :class:`pl.LazyFrame` and ``.collect()``
once per public function.

Per-solve scope
---------------
Several Params (``dt``, ``p_step_duration``, ``p_rp_cost_weight``,
``pdt_branch_weight``, ``pd_branch_weight``) are per-active-solve.  The
helpers accept an explicit ``active_solve`` argument; the
:func:`apply_derived_a` integration entrypoint reads
``solve_data/solve_current.csv`` from the workdir to pick the active
solve, mirroring the established hand-off used by the CSV path.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from polar_high import Param

from ._input_source import _read_csv_file

if TYPE_CHECKING:
    from flextool.engine_polars._input_source import InputSource
    from flextool.engine_polars._solve_context import SolveContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _try_param(source: "InputSource", entity_class: str,
                parameter_name: str) -> pl.DataFrame | None:
    """Return ``source.parameter(...)`` or ``None`` if class unknown /
    parameter unknown / empty.  Mirrors ``_projection_params._try_param``.
    """
    try:
        df = source.parameter(entity_class, parameter_name)
    except KeyError:
        return None
    if df.height == 0:
        return None
    return df


def _try_param_explicit(source: "InputSource", entity_class: str,
                         parameter_name: str) -> pl.DataFrame | None:
    """Like :func:`_try_param` but suppresses default-broadcast rows.

    Mirrors flextool's ``p_unit.get(name, None)`` semantic: returns only
    entities with a parameter_value row in the active scenario.  Falls
    back to the public :meth:`InputSource.parameter` when the source
    has no ``parameter_explicit`` method (legacy custom InputSource
    implementations).
    """
    fn = getattr(source, "parameter_explicit", None) or source.parameter
    try:
        df = fn(entity_class, parameter_name)
    except KeyError:
        return None
    if df.height == 0:
        return None
    return df


def _try_entities(source: "InputSource", entity_class: str) -> pl.DataFrame | None:
    try:
        df = source.entities(entity_class)
    except KeyError:
        return None
    if df.height == 0:
        return None
    return df


def _scalar_default(source: "InputSource", entity_class: str,
                     parameter_name: str, fallback: Any) -> Any:
    """Look up the scalar Spine default; fall back to *fallback* when
    the source returns ``None`` (None-default policy).
    """
    try:
        d = source.parameter_default(entity_class, parameter_name)
    except KeyError:
        return fallback
    return fallback if d is None else d


def _read_active_solve(workdir: Path) -> str | None:
    """Read ``solve_data/solve_current.csv`` and return the active solve
    name, or ``None`` when the file is absent / empty.
    """
    p = Path(workdir) / "solve_data" / "solve_current.csv"
    if not p.exists():
        return None
    df = _read_csv_file(p)
    if df.height == 0:
        return None
    col = df.columns[0]
    return df[col][0]


def _solve_in_spine(source: "InputSource",
                      active_solve: str | None) -> bool:
    """Return True iff ``active_solve`` appears as a row in Spine's
    ``solve`` entity class.

    Δ.18 — synthetic per-sub-solve names (e.g. ``invest_5weeks_p2020``
    for nested-multi-invest fixtures) don't exist in Spine; the per-solve
    override chain (``apply_derived_a`` etc.) returns None for every
    helper that takes ``active_solve`` as a key.  Detecting this upfront
    lets the loader keep the snapshot-CSV-loaded values as authoritative.

    Probes the ``solve`` entity table directly — relying on a per-solve
    parameter (e.g. ``invest_periods``) was incorrect because some solves
    have empty ``invest_periods`` (dispatch-only) yet are valid solves.
    """
    if active_solve is None:
        return False
    try:
        ents = source.entities("solve")
    except KeyError:
        return False
    if ents is None or ents.height == 0:
        return False
    name_col = "name" if "name" in ents.columns else ents.columns[0]
    return ents.filter(pl.col(name_col) == active_solve).height > 0


def _ctx_read(
    ctx: "SolveContext | None",
    workdir: Path | str | None,
    name: str,
    *,
    kind: str = "solve_data",
) -> "pl.DataFrame | None":
    """Δ.12a — single funnel for workdir CSV reads inside derived helpers.

    When ``ctx`` is supplied, the call routes through
    :meth:`SolveContext.read_csv` which caches by absolute path so
    repeated reads of the same file (e.g. ``period_in_use_set.csv``,
    ``period__branch.csv``) hit memory.  When ``ctx`` is None falls
    back to direct ``_read_csv_file`` against the workdir — preserves
    pre-Δ.12a behaviour for callers that haven't been wired up yet.

    Returns ``None`` if the file is absent (matches the existing
    pre-existence-check pattern most helper sites use).
    """
    if ctx is not None:
        return ctx.read_csv(name, kind=kind)
    if workdir is None:
        return None
    if kind == "solve_data":
        path = Path(workdir) / "solve_data" / name
    elif kind == "input":
        path = Path(workdir) / "input" / name
    else:
        path = Path(workdir) / name
    if not path.exists():
        return None
    try:
        return _read_csv_file(path)
    except pl.exceptions.NoDataError:
        return pl.DataFrame()


# ---------------------------------------------------------------------------
# §3.1.1 — dt + p_step_duration
# ---------------------------------------------------------------------------


def dt_and_step_duration_from_source(
    source: "InputSource",
    active_solve: str,
    workdir: Path | None = None,
    *,
    ctx: "SolveContext | None" = None,
) -> tuple[pl.DataFrame, Param] | None:
    """Compute the per-(d, t) ``dt`` set and ``p_step_duration`` Param
    for the active solve.

    Algorithm (audit §3.1.1):

      1. ``solve.realized_periods`` → list of periods active in this solve.
      2. ``solve.period_timeset[active_solve, period]`` → timeset for each.
      3. ``timeset.timeline[ts]`` → which timeline supplies step_duration.
      4. ``timeset.timeset_duration[ts]`` is a 1d_map ``(start_step → count)``
         describing one or more contiguous timestep blocks.
      5. For each (start_step, count) emit *count* consecutive timesteps
         starting at ``start_step`` in the timeline's step ordering.
      6. ``step_duration = timeline.timestep_duration[timeline_name, step]``.

    Per-solve Params; ``active_solve`` is required.

    Returns ``None`` if the active solve has no realized periods (e.g.
    the solve doesn't exist in the source) — caller falls back to CSV.
    """
    # Build the period_in_use set.  When the workdir is supplied AND
    # ``solve_data/period_in_use_set.csv`` exists, prefer that — it's the
    # authoritative set INCLUDING any stochastic-branch periods (mirrors
    # flextool's per_solve_sets.py:95-101 which writes branch periods on
    # top of realised + invest periods).  Otherwise fall back to the
    # union of realized_periods + invest_periods.
    realized_p = None
    if ctx is not None and ctx.period_in_use.height > 0:
        realized_p = ctx.period_in_use.lazy().select("d").unique()
    elif workdir is not None:
        piu = _ctx_read(ctx, workdir, "period_in_use_set.csv")
        if piu is not None and piu.height > 0 and "period" in piu.columns:
            realized_p = (piu.lazy()
                            .select(pl.col("period").alias("d"))
                            .unique())
    if realized_p is None:
        parts: list[pl.LazyFrame] = []
        for ec, par in (("solve", "realized_periods"),
                         ("solve", "invest_periods")):
            df = _try_param(source, ec, par)
            if df is None:
                continue
            parts.append(df.lazy()
                           .filter(pl.col("name") == active_solve)
                           .select(pl.col("value").alias("d")))
        if not parts:
            return None
        realized_p = pl.concat(parts).unique()

    p_ts = _try_param(source, "solve", "period_timeset")
    if p_ts is None:
        return None
    # period_timeset has: name (solve), <period_col>, value (timeset).
    # Discover the period column (the source picks it from the Map's
    # index_name; flexpy schema tolerates either 'period' or 'x').
    p_ts_cols = p_ts.columns
    period_col = next((c for c in ("period", "x") if c in p_ts_cols), None)
    if period_col is None:
        return None
    pt = (p_ts.lazy()
              .filter(pl.col("name") == active_solve)
              .select(pl.col(period_col).alias("d"),
                      pl.col("value").alias("ts")))
    # Broadcast timesets to stochastic-branch periods: when a period
    # appears in period_in_use_set without a direct period_timeset row,
    # use the anchor's timeset via ``period__branch.csv`` (anchor → branch
    # map; flextool's per_solve_sets.py:65-95 does this implicitly by
    # writing per-branch steps_in_use rows under the anchor's timeset).
    pb_raw = _ctx_read(ctx, workdir, "period__branch.csv") if workdir is not None or ctx is not None else None
    if pb_raw is not None and pb_raw.height > 0:
        pb_lf = (pb_raw.lazy()
                    .rename({"period": "anchor", "branch": "d"})
                    .filter(pl.col("anchor") != pl.col("d")))
        # Map branch d → anchor's timeset.
        anchor_ts = (pt.rename({"d": "anchor"})
                        .join(pb_lf, on="anchor", how="inner")
                        .select("d", "ts"))
        pt = pl.concat([pt, anchor_ts]).unique()

    ts_timeline = _try_param(source, "timeset", "timeline")
    if ts_timeline is None:
        return None
    ttl = ts_timeline.lazy().select(pl.col("name").alias("ts"),
                                      pl.col("value").alias("timeline"))

    ts_dur = _try_param(source, "timeset", "timeset_duration")
    if ts_dur is None:
        return None
    # timeset_duration: name=timeset, <step_col>, value=count.
    dur_cols = ts_dur.columns
    step_col = next((c for c in ("t", "x", "step", "timestep")
                       if c in dur_cols and c != "name" and c != "value"),
                     None)
    if step_col is None:
        return None
    blocks = (ts_dur.lazy()
                    .select(pl.col("name").alias("ts"),
                            pl.col(step_col).alias("start_step"),
                            pl.col("value").cast(pl.Float64).alias("count")))

    tl_dur = _try_param(source, "timeline", "timestep_duration")
    if tl_dur is None:
        return None
    # timeline.timestep_duration: name=timeline, <step_col>, value=duration.
    tl_cols = tl_dur.columns
    tl_step_col = next((c for c in ("t", "step", "timestep", "x")
                          if c in tl_cols and c != "name" and c != "value"),
                        None)
    if tl_step_col is None:
        return None
    tl = (tl_dur.lazy()
                .select(pl.col("name").alias("timeline"),
                        pl.col(tl_step_col).alias("t"),
                        pl.col("value").cast(pl.Float64).alias("step_duration")))
    # Determine step rank within each timeline (sort by t lex order, which
    # matches t0001 < t0002 < ... in flextool's canonical naming).
    tl = tl.sort(["timeline", "t"]) \
            .with_columns(rank=pl.col("t").cum_count().over("timeline"))

    # Build (period, ts, timeline, start_step, count) frame, then expand.
    realized_p_with_ts = realized_p.join(pt, on="d", how="inner")
    pst = (realized_p_with_ts
            .join(ttl, on="ts", how="inner")
            .join(blocks, on="ts", how="inner"))

    # For each (period, timeline, start_step, count), emit the timeline
    # rows whose rank ∈ [start_rank, start_rank + count - 1].  Resolve
    # start_rank by joining timeline ranks on (timeline, t=start_step).
    start_ranks = (pst.join(
        tl.select(pl.col("timeline"),
                  pl.col("t").alias("start_step"),
                  pl.col("rank").alias("start_rank")),
        on=["timeline", "start_step"],
        how="left",
    ))
    # Drop blocks whose start_step isn't in the timeline (mis-spec).
    start_ranks = start_ranks.filter(pl.col("start_rank").is_not_null())

    # Cross-join with tl on timeline, then filter rank-in-window.
    expanded = (start_ranks
                  .join(tl, on="timeline", how="inner")
                  .filter((pl.col("rank") >= pl.col("start_rank"))
                          & (pl.col("rank") < pl.col("start_rank")
                              + pl.col("count").cast(pl.Int64))))
    out = (expanded
            .select("d", pl.col("t"),
                    pl.col("step_duration").alias("value"))
            .unique()
            .sort("d", "t")
            .collect())
    if out.height == 0:
        return None
    dt = out.select("d", "t")
    step_dur = Param(("d", "t"), out)
    return dt, step_dur


# ---------------------------------------------------------------------------
# §3.1.4 — p_period_share
# ---------------------------------------------------------------------------


def p_period_share_from_source(
    source: "InputSource",
    dt: pl.DataFrame,
    step_dur: Param,
) -> Param | None:
    """``p_period_share[d] = sum_t step_duration[d,t] / 8760`` — the
    fraction of a full year covered by this period in the active solve.

    Audit §3.1.4: built from ``dt`` and ``p_step_duration``.
    """
    if dt is None or step_dur is None:
        return None
    sd = step_dur.frame
    out = (sd.lazy()
              .group_by("d")
              .agg((pl.col("value").cast(pl.Float64).sum() / 8760.0)
                   .alias("value"))
              .sort("d")
              .collect())
    if out.height == 0:
        return None
    return Param(("d",), out)


# ---------------------------------------------------------------------------
# §3.1.3 — p_inflation_op
# ---------------------------------------------------------------------------


def p_inflation_op_from_source(
    source: "InputSource",
    dt: pl.DataFrame,
    active_solve: str | None = None,
) -> Param | None:
    """Period-level inflation factor for operations.

    Audit §3.1.3 says the formula compounds ``(1 + inflation_rate)^...``;
    the actual flextool implementation
    (``period_calculated_params.py:280-322``) computes
    ``sum_y p_years_represented[d,y] * (1+rate)^until_op[d,y]``
    where ``until_op[d, y] = base[d,y] + pyr[d,y] * offset_op``,
    ``base[d, y] = sum_{y'<y} pyr[d, y']``.

    Γ.8.E coverage:
      * ``model.inflation_rate ∈ {None, 0}`` (the default) — formula
        collapses to ``ops_factor[d] = Σ_y pyr[d, y] * 1 =
        years_represented[solve, d]`` since each period has a single
        flat scalar in the 1D-Map shape exposed by InputSource (each
        scalar IS the total summed over the period's year-set).
      * Multi-year-per-period (``years_represented`` set per solve,
        rate=0) — handled here, mirrors the regression that
        ``work_fullYear_roll`` exposes when CSV retirement Step 3 lands.

    Non-trivial-rate (``rate != 0``) and ``inflation_offset`` cascade
    are handled by :func:`p_inflation_op_multi_year_from_source`
    (Γ.3.C, applied in :func:`apply_derived_c`).

    Returns ``None`` only when no solve / no dt is available; the
    helper otherwise produces the canonical frame so CSV retirement
    won't reintroduce the simple-1.0 regression.
    """
    if dt is None:
        return None
    rate = _try_param(source, "model", "inflation_rate")
    rate_v = 0.0
    if rate is not None and rate.height > 0:
        rate_v = float(rate["value"][0])
    if rate_v != 0.0:
        # Non-trivial rate handled in the multi-year cascade helper
        # (which runs later in apply_derived_c).  Returning None
        # here lets the simple-default 1.0 stay in flex_data; the
        # later helper overlays the correct factor.
        return None

    # Rate = 0.  ``ops_factor[d] = Σ_y pyr[d, y]``.  The 1D Map shape
    # InputSource exposes for ``solve.years_represented`` already
    # collapses the inner year axis into a per-period scalar — so
    # that scalar IS the sum we need.
    #
    # Rolling solves: ``active_solve`` in the workdir is the per-roll
    # name (e.g. ``dispatch_fullYear_roll_roll_71``) but the DB has
    # ``years_represented`` keyed by the parent solve
    # (``dispatch_fullYear_roll``).  flextool's preprocessing copies
    # the parent's value down to every roll; flexpy mirrors that here
    # by trying the roll name first, then the parent name (strip
    # ``_roll_<N>`` suffix).
    yr = None
    if active_solve is not None:
        yr_raw = _try_param(source, "solve", "years_represented")
        if yr_raw is not None and "period" in yr_raw.columns:
            candidate_names = [active_solve]
            import re
            parent = re.sub(r"_roll_\d+$", "", active_solve)
            if parent != active_solve:
                candidate_names.append(parent)
            yr_lf = None
            for cand in candidate_names:
                lf = (yr_raw.lazy()
                            .filter(pl.col("name") == cand)
                            .select(pl.col("period").alias("d"),
                                    pl.col("value").cast(pl.Float64).alias("yr_total")))
                if lf.collect().height > 0:
                    yr_lf = lf
                    break
            if yr_lf is not None:
                d_unique_lf = dt.lazy().select("d").unique()
                yr_joined = (d_unique_lf.join(yr_lf, on="d", how="left")
                                         .with_columns(
                                             value=pl.col("yr_total").fill_null(1.0)
                                         )
                                         .select("d", "value")
                                         .sort("d"))
                yr_collected = yr_joined.collect()
                if yr_collected.height > 0:
                    yr = yr_collected

    if yr is None:
        # Fallback: trivial 1.0 per realized period (no
        # ``years_represented`` row → defaults to 1 year per period).
        yr = (dt.lazy()
                .select("d").unique()
                .with_columns(value=pl.lit(1.0).cast(pl.Float64))
                .sort("d")
                .collect())
    if yr.height == 0:
        return None
    return Param(("d",), yr)


# ---------------------------------------------------------------------------
# §3.1.2 — p_rp_cost_weight
# ---------------------------------------------------------------------------


def p_rp_cost_weight_from_source(
    source: "InputSource",
    dt: pl.DataFrame,
    active_solve: str | None = None,
) -> Param | None:
    """Per-(d, t) representative-period cost weight.

    Audit §3.1.2: defaults to 1.0 per (d, t); explicit overrides come
    from solve-level ``timeset_weights`` (flat per-step weighting,
    non-RP) and ``representative_period_weights`` (nested base→rep
    Map, RP scenarios) in flextool.

    Algorithm (mirrors flextool's
    :func:`flextool.flextoolrunner.solve_writers.write_timeset_cost_weight`
    for the non-RP path and
    :func:`flextool.flextoolrunner.solve_writers.write_rp_data` for the
    RP path):

    Non-RP path (``timeset_weights`` set on a timeset):
      1. For each ``(period, timeset)`` pair from
         ``solve.period_timeset`` filtered to *active_solve*, look up
         ``timeset.timeset_weights[timeset]`` — a flat
         ``{timestep: weight}`` mapping.
      2. For each active timestep ``t`` in the period, emit
         ``raw[t] = weights.get(t, 0.0)``.
      3. Normalise to sum-to-n_active over the period's active steps:
         ``scale = n_active / sum(raw)``, then per-(d, t)
         ``value = raw[t] * scale``.  This makes a uniform input
         reproduce the trivial default 1.0 per step.
      4. Periods without a ``timeset_weights`` row keep the trivial
         default 1.0.

    RP path (``representative_period_weights`` set):
      Currently deferred — RP fixtures don't appear in the regression
      set this helper unblocks.  The CSV-loaded ``rp_cost_weight.csv``
      already encodes the canonical weights; the helper emits ``None``
      via the empty-frame fallback so the CSV value survives.

    The ``active_solve`` argument was added in Γ.8.E so the helper can
    filter ``solve.period_timeset`` to the right solve when running in
    multi-solve cascades.  Backward-compat: ``None`` falls back to a
    no-op (returns the dense 1.0 default) for any caller still on the
    Γ.3.A signature.
    """
    if dt is None:
        return None

    # Build the dense default as the baseline; any per-(d, t) override
    # from ``timeset_weights`` will replace it via a left-join.
    default_lf = (dt.lazy()
                    .select("d", "t")
                    .with_columns(value=pl.lit(1.0).cast(pl.Float64))
                    .sort("d", "t"))

    # Discover ``solve.period_timeset`` and ``timeset.timeset_weights``;
    # if either is unavailable / empty, fall through to the trivial
    # default.
    if active_solve is None:
        out = default_lf.collect()
        return Param(("d", "t"), out) if out.height > 0 else None

    pt = _try_param(source, "solve", "period_timeset")
    if pt is None:
        out = default_lf.collect()
        return Param(("d", "t"), out) if out.height > 0 else None
    pt_cols = pt.columns
    period_col = next((c for c in ("period", "x") if c in pt_cols), None)
    if period_col is None:
        out = default_lf.collect()
        return Param(("d", "t"), out) if out.height > 0 else None
    period_timeset_lf = (pt.lazy()
                            .filter(pl.col("name") == active_solve)
                            .select(pl.col(period_col).alias("d"),
                                    pl.col("value").alias("ts")))

    tw = _try_param_explicit(source, "timeset", "timeset_weights")
    if tw is None or tw.height == 0:
        out = default_lf.collect()
        return Param(("d", "t"), out) if out.height > 0 else None
    # ``timeset_weights`` shape: name (timeset), x (timestep), value (weight).
    tw_cols = tw.columns
    step_col = next((c for c in ("x", "time", "step") if c in tw_cols), None)
    if step_col is None:
        out = default_lf.collect()
        return Param(("d", "t"), out) if out.height > 0 else None
    weights_lf = (tw.lazy()
                    .select(pl.col("name").alias("ts"),
                            pl.col(step_col).alias("t"),
                            pl.col("value").cast(pl.Float64).alias("w_raw")))

    # Per-(d, t) raw weight: lookup via period_timeset → timeset_weights.
    # Periods whose timeset has weights produce non-null ``w_raw``;
    # periods without keep null and fall back to the trivial default 1.0.
    dt_lf = dt.lazy().select("d", "t")
    joined = (dt_lf
              .join(period_timeset_lf, on="d", how="left")
              .join(weights_lf, on=["ts", "t"], how="left"))

    # Per-period sum of raw weights and active-step count for scaling.
    # A period without any ``timeset_weights`` row sums to null/0 — the
    # downstream branch leaves ``value=1.0`` for those rows.
    by_period = (joined
                 .group_by("d")
                 .agg(pl.col("w_raw").fill_null(0.0).sum().alias("w_sum"),
                      pl.len().alias("n_active")))

    # Periods with weights: scale = n_active / w_sum, value = w_raw * scale.
    # Periods without weights: value = 1.0 (default).
    out_lf = (joined
              .join(by_period, on="d", how="left")
              .with_columns(
                  value=pl.when(
                      pl.col("w_sum").is_not_null() & (pl.col("w_sum") > 0)
                  ).then(
                      pl.col("w_raw").fill_null(0.0)
                      * pl.col("n_active").cast(pl.Float64)
                      / pl.col("w_sum")
                  ).otherwise(pl.lit(1.0))
              )
              .select("d", "t", "value")
              .sort("d", "t"))

    out = out_lf.collect()
    if out.height == 0:
        return None
    return Param(("d", "t"), out)


# ---------------------------------------------------------------------------
# §3.18.1 — pdt_branch_weight / pd_branch_weight
# ---------------------------------------------------------------------------


def pd_branch_weight_from_source(
    source: "InputSource",
    dt: pl.DataFrame,
) -> Param | None:
    """Per-period branch weight.

    Audit §3.18.1: defaults to 1.0; in deterministic single-branch
    scenarios the file is absent / silent.  The full multi-branch
    derivation (read ``period__branch.weight``, normalise across
    siblings) needs a ``period__branch`` relationship class that the
    current Spine schema doesn't expose directly to ``InputSource`` —
    it's downstream of ``solve.stochastic_branches``.

    For Γ.3.A: emit the trivial 1.0 default per realized period; this
    matches every non-stochastic fixture.  Stochastic fixtures retain
    their CSV-computed value (deferred to Batch C/D).
    """
    if dt is None:
        return None
    out = (dt.lazy()
             .select("d").unique()
             .with_columns(value=pl.lit(1.0))
             .sort("d")
             .collect())
    if out.height == 0:
        return None
    return Param(("d",), out)


def pdt_branch_weight_from_source(
    source: "InputSource",
    dt: pl.DataFrame,
    pd_bw: Param | None = None,
) -> Param | None:
    """Per-(d, t) branch weight.

    Audit §3.18.1: defaults to 1.0 per (d, t).  The full multi-branch
    derivation is deferred (same rationale as :func:`pd_branch_weight_from_source`).
    """
    if dt is None:
        return None
    out = (dt.lazy()
             .with_columns(value=pl.lit(1.0))
             .select("d", "t", "value")
             .sort("d", "t")
             .collect())
    if out.height == 0:
        return None
    return Param(("d", "t"), out)


# ---------------------------------------------------------------------------
# §3.2.1 — p_inflow (scaling step over node.inflow)
# ---------------------------------------------------------------------------


def p_inflow_from_source(
    source: "InputSource",
    dt: pl.DataFrame,
    step_dur: Param,
) -> Param | None:
    """Per-(n, d, t) node inflow with scaling applied.

    Audit §3.2.1 algorithm:

    1. ``p_inflow_raw`` = unrolled ``node.inflow`` (Map / TimeSeries /
       3d_map for stochastic).  Source already produces this.
    2. ``inflow_method`` decides the scaling:
       * ``use_original`` — pass-through.
       * ``no_inflow`` — drop.
       * ``scale_to_annual_flow`` — scale so sum_t (raw * step_dur) =
         annual_flow * period_share.
       * ``scale_to_peak_inflow`` / ``use_peak_inflow`` — scale so
         max_t raw' = peak_inflow * availability.
       * ``scale_to_annual_and_peak`` — combination.

    For Γ.3.A: cover the **default ``use_original``** path which
    requires no scaling.  This handles all non-time-scaled inflow
    fixtures (most of the suite).  The scaling cascade for
    ``scale_to_*`` methods is deferred to Batch B; fixtures that
    activate it retain their CSV-loaded value.

    Returns ``None`` if the helper can't safely override (any
    non-default ``inflow_method`` present, or the source frame's shape
    doesn't match what the simple path expects).
    """
    if dt is None:
        return None
    method = _try_param(source, "node", "inflow_method")
    if method is not None and method.height > 0:
        # Any non-`use_original` (and non-default) method → scaling path.
        # We don't yet implement scaling — defer to CSV.
        non_trivial = method.filter(
            ~pl.col("value").is_in(["use_original"])
        )
        if non_trivial.height > 0:
            return None
    raw = _try_param(source, "node", "inflow")
    if raw is None:
        return None
    cols = raw.columns
    # Expected 1d-period-keyed Map(t) → cols ['name', 'period', 't', 'value']
    # or 3d_map ['name','period','branch','t','value'] (stochastic).
    if "branch" in cols:
        # Stochastic — defer (Batch C scope).
        return None
    if not {"name", "period", "t", "value"}.issubset(cols):
        return None
    out = (raw.lazy()
              .select(pl.col("name").alias("n"),
                      pl.col("period").alias("d"),
                      pl.col("t"),
                      pl.col("value").cast(pl.Float64))
              # Restrict to the active solve's (d, t) — matches CSV-side
              # which only emits rows for periods/timesteps in dt.
              .join(dt.lazy(), on=["d", "t"], how="inner")
              .sort("n", "d", "t")
              .collect())
    if out.height == 0:
        return None
    return Param(("n", "d", "t"), out)


# ---------------------------------------------------------------------------
# §3.6.2 — p_process_existing_count
# ---------------------------------------------------------------------------


def _read_p_entity_all_existing_csv(workdir: Path | None
                                       ) -> pl.DataFrame | None:
    """Γ.6.D — direct read of ``solve_data/p_entity_all_existing.csv``.

    Returns ``[e, d, value]`` or None when the CSV is absent.  Used by
    helpers that need the chained existing capacity (incl. multi-solve
    handoff) without recomputing it from raw ``entity.existing``.
    """
    if workdir is None:
        return None
    p = Path(workdir) / "solve_data" / "p_entity_all_existing.csv"
    if not p.exists():
        return None
    try:
        df = _read_csv_file(p)
    except Exception:
        return None
    if (df.height == 0 or "entity" not in df.columns
            or "period" not in df.columns or "value" not in df.columns):
        return None
    return (df.rename({"entity": "e", "period": "d"})
              .with_columns(value=pl.col("value")
                                       .cast(pl.Float64, strict=False)
                                       .fill_null(0.0))
              .select("e", "d", "value"))


def p_process_existing_count_from_source(
    source: "InputSource",
    dt: pl.DataFrame,
    active_solve: str | None = None,
    workdir: Path | None = None,
) -> Param | None:
    """Per-(p, d) existing unit count = ``existing / unitsize``.

    Audit §3.6.2 / canonical unitsize cascade per
    ``entity_period_calc_params.py:186-202``::

        unitsize = virtual_unitsize (if explicitly set and non-zero)
                  OR existing       (if non-zero)
                  OR 1000.0

    The "explicitly set" check uses ``source.parameter_explicit(...)``
    which returns only entities with rows in ``parameter_value`` —
    Spine's default-broadcast rows are suppressed so the cascade can
    distinguish "absent in JSON" from "set to the schema default".

    For multi-period scenarios with no investment (default fixture
    coverage), ``existing`` does not vary with ``d`` — the result is a
    cross-product against the period set.

    Returns ``None`` when no entity has an explicit existing capacity.
    """
    if dt is None:
        return None

    # Γ.6.D — prefer the canonical chained ``p_entity_all_existing.csv``
    # when available (carries multi-solve handoff state and lifetime
    # gate already integrated).  We still need the unitsize cascade per
    # entity to convert capacity → unit count.
    pae_csv = _read_p_entity_all_existing_csv(workdir)
    if pae_csv is not None and pae_csv.height > 0:
        # Restrict to process-side entities (unit ∪ connection).
        proc_set, _, _ = _entity_classes_lookup(source)
        per_proc = pae_csv.filter(pl.col("e").is_in(list(proc_set)))
        if per_proc.height > 0:
            us_lf = _entity_unitsize_lf(source)
            df = (per_proc.lazy()
                     .rename({"e": "p"})
                     .join(us_lf.rename({"e": "p"}), on="p", how="left")
                     .with_columns(us=pl.col("us").fill_null(1000.0))
                     .with_columns(value=pl.col("value") / pl.col("us"))
                     .select("p", "d", "value")
                     .sort("p", "d")
                     .collect())
            if df.height > 0:
                return Param(("p", "d"), df)

    parts: list[pl.LazyFrame] = []
    for cls in ("unit", "connection"):
        ex = _try_param_explicit(source, cls, "existing")
        if ex is None:
            continue
        ex_lf = ex.lazy().select(
            pl.col("name").alias("p"),
            pl.col("value").cast(pl.Float64).alias("existing"),
        )
        us = _try_param_explicit(source, cls, "virtual_unitsize")
        if us is not None and us.height > 0:
            us_lf = us.lazy().select(
                pl.col("name").alias("p"),
                pl.col("value").cast(pl.Float64).alias("us_raw"),
            )
            ex_lf = (ex_lf.join(us_lf, on="p", how="left")
                            .with_columns(
                                us_raw=pl.col("us_raw").fill_null(0.0)))
        else:
            ex_lf = ex_lf.with_columns(us_raw=pl.lit(0.0))
        # Apply the canonical cascade: virtual_unitsize → existing → 1000.
        ex_lf = ex_lf.with_columns(
            us=pl.when(pl.col("us_raw") != 0.0)
                  .then(pl.col("us_raw"))
                  .when(pl.col("existing") != 0.0)
                  .then(pl.col("existing"))
                  .otherwise(pl.lit(1000.0))
        )
        parts.append(ex_lf)
    if not parts:
        return None
    base = pl.concat(parts)
    periods = dt.lazy().select("d").unique()
    out = (base.join(periods, how="cross")
                .with_columns(value=pl.col("existing") / pl.col("us"))
                .select("p", "d", "value")
                .sort("p", "d")
                .collect())
    if out.height == 0:
        return None
    # Γ.6.D — apply lifetime gate to the existing-count cascade.  Mirrors
    # ``entity_period_calc_params.write_p_entity_existing_chain`` line
    # 1571-1582: ``p_entity_existing_count = all_existing / unitsize``
    # where ``all_existing`` carries the lifetime gate from
    # ``p_entity_pre_existing``.  For ``reinvest_choice`` / ``no_investment``
    # entities past expiry, zero out the row.
    expired = _lifetime_expired_pairs(
        source, active_solve, workdir,
        methods=("reinvest_choice", "no_investment"))
    if expired:
        expired_lf = pl.DataFrame(
            [(e, d) for (e, d) in expired],
            schema=["p", "d"], orient="row").lazy()
        out = (out.lazy()
                  .join(expired_lf.with_columns(_expired=pl.lit(True)),
                          on=["p", "d"], how="left")
                  .with_columns(
                      value=pl.when(pl.col("_expired").fill_null(False))
                                .then(0.0)
                                .otherwise(pl.col("value"))
                  )
                  .select("p", "d", "value")
                  .sort("p", "d")
                  .collect())
    return Param(("p", "d"), out)


# ---------------------------------------------------------------------------
# §3.6.1 — p_profile_value (alternative cascade)
# ---------------------------------------------------------------------------


def p_profile_value_from_source(
    source: "InputSource",
    dt: pl.DataFrame,
) -> Param | None:
    """Per-(f, d, t) profile value, resolved across the alternative
    cascade.

    Audit §3.6.1 algorithm:
        ``profile.profile`` parameter cascades scalar → period → time-of-
        period → branch.  Each tier is the next narrower index space.

    Implementation strategy (single-collect lazy chain):

    1. Pull ``source.parameter("profile", "profile")``.  The source has
       already collapsed alternative ranks into a single per-entity
       view.  The shape depends on the value's runtime type:
        - scalar → ``[name, value]`` (broadcast over (d, t) of dt).
        - 1d_map(period) → ``[name, period, value]``
          (broadcast over t of dt for matching periods).
        - time_series → ``[name, t, value]``
          (broadcast over d of dt for matching t).
        - 2d_map(period, t) → ``[name, period, t, value]`` (direct).
        - 3d_map(period, branch, t) → ``[name, period, branch, t, value]``
          (stochastic — drops branch dim downstream via period_branch).

    The cascade is per-entity, not within an entity: each profile picks
    one tier based on the alternative chain's resolution.  The returned
    frame is ``[f, d, t, value]`` with one row per (profile × dt).

    Stochastic 3d_map case is deferred (Batch C scope) — caller falls
    back to CSV when the source returns a branch-dim'd profile frame.
    """
    if dt is None:
        return None
    raw = _try_param(source, "profile", "profile")
    if raw is None:
        return None
    cols = raw.columns
    if "branch" in cols:
        # Stochastic 3d_map (named branch column) — deferred.
        return None
    # Δ.7: dropped the ``_check_canonical_keys`` predicate (per the
    # Δ.6 close-stanza TODO #5).  The Δ.7 ``_derived_profile`` module
    # owns the canonical cascade; this legacy helper is preserved as a
    # ``parameter()``-only fast path that handles the deterministic
    # period/t/scalar cases via column-name detection (still useful as
    # a fallback when the new cascade can't be wired — eg. ad-hoc
    # callers in the test suite).  Generic ``x`` / ``i`` keys are now
    # tolerated and routed through the time-axis branch by the column
    # detector below.
    dt_lf = dt.lazy()
    # Detect tier by which columns are present.
    has_period = "period" in cols
    has_t = "t" in cols
    if has_period and has_t:
        # 2d_map: direct (f, d, t, value).
        out = (raw.lazy()
                  .select(pl.col("name").alias("f"),
                          pl.col("period").alias("d"),
                          pl.col("t"),
                          pl.col("value").cast(pl.Float64))
                  .join(dt_lf, on=["d", "t"], how="inner")
                  .sort("f", "d", "t")
                  .collect())
    elif has_period:
        # 1d_map(period): broadcast over t of dt for matching periods.
        out = (raw.lazy()
                  .select(pl.col("name").alias("f"),
                          pl.col("period").alias("d"),
                          pl.col("value").cast(pl.Float64))
                  .join(dt_lf, on="d", how="inner")
                  .select("f", "d", "t", "value")
                  .sort("f", "d", "t")
                  .collect())
    elif has_t:
        # time_series: broadcast over d for matching t.
        out = (raw.lazy()
                  .select(pl.col("name").alias("f"),
                          pl.col("t"),
                          pl.col("value").cast(pl.Float64))
                  .join(dt_lf, on="t", how="inner")
                  .select("f", "d", "t", "value")
                  .sort("f", "d", "t")
                  .collect())
    else:
        # Scalar — broadcast over the full dt × profiles grid.
        out = (raw.lazy()
                  .select(pl.col("name").alias("f"),
                          pl.col("value").cast(pl.Float64))
                  .join(dt_lf, how="cross")
                  .select("f", "d", "t", "value")
                  .sort("f", "d", "t")
                  .collect())
    if out.height == 0:
        return None
    return Param(("f", "d", "t"), out)


# ---------------------------------------------------------------------------
# Catalog (used by parity tests)
# ---------------------------------------------------------------------------

DERIVED_A_FIELDS = (
    "dt",
    "p_step_duration",
    "p_period_share",
    "p_inflation_op",
    "p_rp_cost_weight",
    "pd_branch_weight",
    "pdt_branch_weight",
    "p_inflow",
    "p_process_existing_count",
    "p_profile_value",
)


# ---------------------------------------------------------------------------
# Integration entrypoint
# ---------------------------------------------------------------------------


def apply_derived_a(
    flex_data: object,
    source: "InputSource",
    workdir: Path,
    *,
    ctx: "SolveContext | None" = None,
) -> None:
    """Apply Γ.3.A foundational Derived Params, mutating ``flex_data``
    in place.

    Δ.12b — assignment is unconditional for the helpers that are
    authoritative producers (``p_inflation_op``, ``p_rp_cost_weight``,
    ``pd_branch_weight``, ``pdt_branch_weight``, ``p_penalty_up``,
    ``p_penalty_down``).  ``p_inflow`` and ``p_process_existing_count``
    retain a conditional assignment because the helpers have known
    incomplete coverage (``inflow_method=scale_to_*`` is deferred to
    Batch B; ``p_process_existing_count`` skips when no processes have
    explicit existing capacity in the source) — see TODO at each call
    site for the helper-extension scope.

    Dependency order:
        dt / p_step_duration → p_period_share, p_inflation_op,
        p_rp_cost_weight, pd_branch_weight, pdt_branch_weight, p_inflow
        → p_process_existing_count, p_profile_value (high-risk, last).
    """
    active_solve = ctx.solve_name if ctx is not None else _read_active_solve(workdir)
    if active_solve is None:
        # Single-solve fixtures may omit solve_current.csv; fall back to
        # the FlexData's already-loaded dt (no override possible).
        return

    # 1. dt + p_step_duration -------------------------------------------
    dt_step = dt_and_step_duration_from_source(source, active_solve,
                                                   workdir, ctx=ctx)
    if dt_step is None:
        # Without dt we can't derive the dependent Params either; bail.
        return
    dt_db, step_dur_db = dt_step
    dt_csv = getattr(flex_data, "dt", None)
    if dt_db is not None:
        flex_data.dt = dt_db
        flex_data.p_step_duration = step_dur_db
        usable_dt = dt_db
        sd_for_share = step_dur_db
    else:
        usable_dt = dt_csv
        sd_for_share = getattr(flex_data, "p_step_duration", None)

    if usable_dt is None:
        return

    # 2. p_period_share — None when sd_for_share is None (no step duration).
    if sd_for_share is not None:
        flex_data.p_period_share = p_period_share_from_source(
            source, usable_dt, sd_for_share)

    # 3. p_inflation_op — Δ.12b: unconditional.  None == no inflation
    #    declared; the multi-year cascade in apply_derived_c overlays
    #    this with the full per-period inflation later.
    flex_data.p_inflation_op = p_inflation_op_from_source(
        source, usable_dt, active_solve)

    # 4. p_rp_cost_weight — Δ.12b: unconditional.  None == no
    #    timeset.timeset_weights declared (default 1.0 broadcast handled
    #    inside the helper).
    flex_data.p_rp_cost_weight = p_rp_cost_weight_from_source(
        source, usable_dt, active_solve)

    # 5. pd_branch_weight + pdt_branch_weight — Δ.12b: unconditional.
    # Γ.3.A's simple-1.0 helpers; Γ.3.G's full multi-branch cascade
    # overlays them later via apply_branch_cluster.
    pd_bw = pd_branch_weight_from_source(source, usable_dt)
    flex_data.pd_branch_weight = pd_bw
    flex_data.pdt_branch_weight = pdt_branch_weight_from_source(
        source, usable_dt, pd_bw)

    # 6. p_inflow -------------------------------------------------------
    # Δ.12c-fix2 gap #2 close — full scaling cascade for
    # ``inflow_method ∈ {scale_to_*, scale_in_proportion, use_original}``
    # ported into ``_inflow_scaling.py`` (mirrors flextool's
    # ``preprocessing/node_inflow_scaling_params.py`` +
    # ``entity_period_calc_params.write_pdtNodeInflow``).  The scaling
    # helper returns None on stochastic 3d_map shapes (caller's
    # branch 1/2 fold-in) — fall through to the simpler
    # ``p_inflow_from_source`` (Γ.3.A use_original path) for
    # non-scaling fixtures.  The legacy seed-loaded CSV value remains
    # the safety net only for stochastic inflow (deferred to Δ.13+).
    #
    # Δ.13: derive per-solve aggregates natively when possible so the
    # scaling helper can avoid reading workdir CSVs for cpsoy / p_tdy /
    # period_timeline / dt_complete.  Returns None on synthetic
    # rolling/nested sub-solves whose names aren't in Spine — caller's
    # workdir-CSV path takes over.
    from ._inflow_scaling import apply_p_inflow_with_scaling
    from ._per_solve_sets import derive_per_solve_aggregates
    per_solve_aggs = derive_per_solve_aggregates(source, active_solve)
    scaled = apply_p_inflow_with_scaling(flex_data, source, workdir,
                                          usable_dt,
                                          per_solve_aggs=per_solve_aggs)
    if not scaled and sd_for_share is not None:
        inflow = p_inflow_from_source(source, usable_dt, sd_for_share)
        if inflow is not None:
            flex_data.p_inflow = inflow

    # 7. p_process_existing_count ---------------------------------------
    # Δ.12c-fix gap #4: the helper handles two paths:
    # (1) ``p_entity_all_existing.csv`` (workdir-derived, carries
    #     multi-solve handoff state and lifetime gate);
    # (2) explicit ``unit.existing`` / ``connection.existing`` cascade
    #     (existing / unitsize per (p, d)).
    # Returns ``None`` for fixtures with neither path active (pure-invest
    # without initial capacity).  Keep conditional so the seed Param
    # (which always emits a row per process for ``base_cap_pd`` from the
    # CSV preprocessing) survives in that degenerate case.
    ec = p_process_existing_count_from_source(
        source, usable_dt, active_solve, workdir)
    if ec is not None:
        flex_data.p_process_existing_count = ec

    # 8. p_profile_value (Δ.7 cluster C — profile cascade) ------------
    # Δ.7 lifts the helper into ``_derived_profile.apply_profile_cascade``
    # — the new module ports flextool's full 5-branch cascade lazily and
    # drops the Δ.5/Δ.6 canonical-keys gate.  We delegate here so the
    # apply_derived_a entry stays the single integration point.
    # Δ.12b: defensive try/except removed — apply_profile_cascade is
    # parity-bound (cluster C tests gate it).
    from ._derived_profile import apply_profile_cascade
    apply_profile_cascade(flex_data, source, workdir)

    # 9. p_penalty_up / p_penalty_down (Δ.10 cluster F) ----------------
    # Sentinel-default scalar broadcast over (n, d, t) restricted to
    # nodeBalance nodes.  Mirrors ``input.py:_load_node`` lines 695-700
    # (the slice from ``pdtNode.csv``).
    # Δ.12b: unconditional — penalty helpers always produce a Param
    # when nodeBalance is non-empty (sentinel default applied inside).
    from ._derived_arithmetic import (
        p_penalty_up_from_source,
        p_penalty_down_from_source,
    )
    nb_df = getattr(flex_data, "nodeBalance", None)
    flex_data.p_penalty_up = p_penalty_up_from_source(source, nb_df, usable_dt)
    flex_data.p_penalty_down = p_penalty_down_from_source(
        source, nb_df, usable_dt)


# ---------------------------------------------------------------------------
# Internal: frame-equal guard for safe overlay
# ---------------------------------------------------------------------------


def _frame_equal_sorted(a, b, keys: tuple[str, ...]) -> bool:
    """Return True iff frames *a* and *b* are equal after sort by *keys*
    and ``Float64`` cast on the value column where present.  Lenient
    on dtype mismatches — the bar is *value parity*, not metadata
    parity.
    """
    fa = a.frame if hasattr(a, "frame") else a
    fb = b.frame if hasattr(b, "frame") else b
    if fa is None or fb is None:
        return False
    cols_a = set(fa.columns)
    cols_b = set(fb.columns)
    if cols_a != cols_b:
        return False
    sort_cols = [c for c in keys if c in cols_a]
    extra = [c for c in fa.columns if c not in sort_cols]
    fa = fa.sort(sort_cols + extra) if sort_cols else fa
    fb = fb.sort(sort_cols + extra) if sort_cols else fb
    if "value" in cols_a:
        fa = fa.with_columns(value=pl.col("value").cast(pl.Float64,
                                                          strict=False))
        fb = fb.with_columns(value=pl.col("value").cast(pl.Float64,
                                                          strict=False))
    return fa.equals(fb)


def _param_matches(csv_val: object | None, db_val: object | None,
                    tol: float = 1e-9) -> bool:
    """Return True iff CSV and DB Params have matching frames within
    *tol* on the value column.  Conservative: requires both non-None
    and same column set.  Used as the safety gate before overlaying.
    """
    if csv_val is None or db_val is None:
        return False
    fa = csv_val.frame if hasattr(csv_val, "frame") else csv_val
    fb = db_val.frame if hasattr(db_val, "frame") else db_val
    cols_a = set(fa.columns)
    cols_b = set(fb.columns)
    if cols_a != cols_b:
        return False
    if fa.height != fb.height:
        return False
    keys = [c for c in fa.columns if c != "value"]
    fa = fa.sort(keys) if keys else fa
    fb = fb.sort(keys) if keys else fb
    if "value" in cols_a:
        a_v = fa["value"].cast(pl.Float64, strict=False).fill_null(0.0)
        b_v = fb["value"].cast(pl.Float64, strict=False).fill_null(0.0)
        diff = (a_v - b_v).abs().max()
        if diff is None or diff > tol:
            return False
        # Compare keys row-by-row.
        for k in keys:
            if not fa[k].equals(fb[k]):
                return False
        return True
    # No value column — pure set equality.
    return fa.equals(fb)


# ===========================================================================
# Γ.3.B — Process topology + reclassified method-derived helpers
# ===========================================================================
#
# Per the audit doc §3.3 / §3.5 / §3.10 (and the Γ.2 reclassification list),
# every Param built on top of flextool's "internal method" classification
# must derive that classification from the (conversion_method,
# startup_method, fork_method) triplet — see
# ``flextool/flextoolrunner/input_writer.py:929-963``.  This batch ports
# that mapping plus the dependent topology / cap / slope / varCost helpers.
#
# Each helper is gated on a frame-equal precheck against the CSV-loaded
# value before being written to ``flex_data`` (``apply_derived_b``).
# A mismatch leaves the CSV value in place — never an LP corruption surface.

# (ct_method, startup_method, fork_method) → internal method.
# Mirrors flextool/flextoolrunner/input_writer.py::METHODS_MAPPING:929-963.
_METHODS_MAPPING: dict[tuple[str, str, str], str] = {
    ("constant_efficiency", "no_startup", "fork_no"): "method_1way_1var_off",
    ("constant_efficiency", "no_startup", "fork_yes"): "method_1way_nvar_off",
    ("constant_efficiency", "linear", "fork_no"): "method_1way_1var_LP",
    ("constant_efficiency", "linear", "fork_yes"): "method_1way_nvar_LP",
    ("constant_efficiency", "binary", "fork_no"): "method_1way_1var_MIP",
    ("constant_efficiency", "binary", "fork_yes"): "method_1way_nvar_MIP",
    ("no_losses_no_variable_cost", "no_startup", "fork_no"): "method_2way_1var_off",
    ("no_losses_no_variable_cost", "no_startup", "fork_yes"): "method_2way_nvar_off",
    ("variable_cost_only", "no_startup", "fork_no"): "method_2way_2var_off",
    ("variable_cost_only", "no_startup", "fork_yes"): "method_2way_nvar_off",
    ("regular", "no_startup", "fork_no"): "method_2way_2var_exclude",
    ("regular", "no_startup", "fork_yes"): "not_applicable",
    ("exact", "no_startup", "fork_no"): "method_2way_2var_MIP_exclude",
    ("exact", "no_startup", "fork_yes"): "not_applicable",
    ("min_load_efficiency", "no_startup", "fork_no"): "not_applicable",
    ("min_load_efficiency", "no_startup", "fork_yes"): "not_applicable",
    ("min_load_efficiency", "linear", "fork_no"): "method_1way_1var_LP",
    ("min_load_efficiency", "linear", "fork_yes"): "method_1way_nvar_LP",
    ("min_load_efficiency", "binary", "fork_no"): "method_1way_1var_MIP",
    ("min_load_efficiency", "binary", "fork_yes"): "method_1way_nvar_MIP",
    ("none", "no_startup", "fork_no"): "method_1way_1var_off",
    ("none", "no_startup", "fork_yes"): "method_1way_nvar_off",
    ("none", "linear", "fork_no"): "method_1way_1var_LP",
    ("none", "linear", "fork_yes"): "method_1way_nvar_LP",
    ("none", "binary", "fork_no"): "method_1way_1var_MIP",
    ("none", "binary", "fork_yes"): "method_1way_nvar_MIP",
    ("unidirectional", "no_startup", "fork_no"): "method_1way_1var_off",
    ("unidirectional", "no_startup", "fork_yes"): "method_1way_nvar_off",
    ("unidirectional", "linear", "fork_no"): "method_1way_1var_LP",
    ("unidirectional", "linear", "fork_yes"): "method_1way_nvar_LP",
    ("unidirectional", "binary", "fork_no"): "method_1way_1var_MIP",
    ("unidirectional", "binary", "fork_yes"): "method_1way_nvar_MIP",
}


# Method-category subsets — mirror
# flextool/flextoolrunner/preprocessing/_method_constants.py.
_METHOD_DIRECT = frozenset((
    "method_1way_1var_off", "method_1way_1var_LP", "method_1way_1var_MIP",
    "method_2way_1var_off", "method_2way_2var_off",
    "method_2way_2var_exclude", "method_2way_2var_MIP_exclude",
))
_METHOD_INDIRECT = frozenset((
    "method_1way_nvar_off", "method_1way_nvar_LP", "method_1way_nvar_MIP",
    "method_2way_nvar_off",
))
_METHOD_2WAY_2VAR = frozenset((
    "method_2way_2var_off", "method_2way_2var_exclude",
    "method_2way_2var_MIP_exclude",
))
_METHOD_1WAY_1VAR = frozenset((
    "method_1way_1var_off", "method_1way_1var_LP", "method_1way_1var_MIP",
))
_METHOD_LP = frozenset(("method_1way_1var_LP", "method_1way_nvar_LP"))
_METHOD_MIP = frozenset((
    "method_1way_1var_MIP", "method_1way_nvar_MIP",
    "method_2way_2var_MIP_exclude",
))


def _classify_process_method(source: "InputSource") -> pl.DataFrame:
    """Build the per-process internal-method classification.

    Returns a DataFrame with schema ``[p, ct, startup, fork, method,
    klass]``:

      * ``ct`` — user-facing ``conversion_method`` / ``transfer_method``.
      * ``startup`` — user-facing ``startup_method`` (default ``no_startup``,
        promoted to ``linear`` when ``minimum_time_method ∈ {min_uptime,
        min_downtime, both}``; mirrors input_writer.py:1294-1304).
      * ``fork`` — ``fork_yes`` when |sources|>1 OR |sinks|>1 OR delayed,
        else ``fork_no`` (input_writer.py:1350-1354).
      * ``method`` — internal method per :data:`_METHODS_MAPPING`; rows
        whose triplet maps to ``not_applicable`` or has no mapping are
        dropped.
      * ``klass`` — ``"unit"`` or ``"connection"`` (the entity class of
        the row).

    Algorithm (single-collect lazy chain):

      1. Pull all units + connections.
      2. Pull ``conversion_method`` (units), ``transfer_method``
         (connections), ``startup_method`` (units + connections),
         ``minimum_time_method`` (units), ``delay`` (units + connections).
      3. Count input/output arcs per process.
      4. Apply defaults (``constant_efficiency`` for units,
         ``regular`` for connections; ``no_startup``).
      5. Apply minimum_time → linear-startup override.
      6. Resolve fork from arc counts + delayed-set membership.
      7. Map ``(ct, startup, fork) → method`` via the polars when/then
         chain built from :data:`_METHODS_MAPPING`.
      8. Drop ``not_applicable`` rows; ``not_applicable`` is also returned
         when the triplet has no mapping (defensive).
    """
    # Step 1 — process universe.
    units = _try_entities(source, "unit")
    conns = _try_entities(source, "connection")
    parts: list[pl.LazyFrame] = []
    if units is not None:
        parts.append(units.lazy().select(
            pl.col("name").alias("p"),
            pl.lit("unit").alias("klass"),
        ))
    if conns is not None:
        parts.append(conns.lazy().select(
            pl.col("name").alias("p"),
            pl.lit("connection").alias("klass"),
        ))
    if not parts:
        return pl.DataFrame(schema={
            "p": pl.Utf8, "ct": pl.Utf8, "startup": pl.Utf8,
            "fork": pl.Utf8, "method": pl.Utf8, "klass": pl.Utf8,
        })
    base = pl.concat(parts).unique()

    # Step 2 — pull method-shaping params.
    cm_unit = _try_param(source, "unit", "conversion_method")
    cm_conn = _try_param(source, "connection", "transfer_method")
    cm_parts: list[pl.LazyFrame] = []
    if cm_unit is not None:
        cm_parts.append(cm_unit.lazy().select(
            pl.col("name").alias("p"),
            pl.col("value").alias("ct"),
        ))
    if cm_conn is not None:
        cm_parts.append(cm_conn.lazy().select(
            pl.col("name").alias("p"),
            pl.col("value").alias("ct"),
        ))
    cm_lazy = pl.concat(cm_parts) if cm_parts else None

    sm_unit = _try_param(source, "unit", "startup_method")
    sm_conn = _try_param(source, "connection", "startup_method")
    sm_parts: list[pl.LazyFrame] = []
    if sm_unit is not None:
        sm_parts.append(sm_unit.lazy().select(
            pl.col("name").alias("p"),
            pl.col("value").alias("startup_raw"),
        ))
    if sm_conn is not None:
        sm_parts.append(sm_conn.lazy().select(
            pl.col("name").alias("p"),
            pl.col("value").alias("startup_raw"),
        ))
    sm_lazy = pl.concat(sm_parts) if sm_parts else None

    mtm = _try_param(source, "unit", "minimum_time_method")
    mtm_lazy = (mtm.lazy().select(
        pl.col("name").alias("p"),
        pl.col("value").alias("mtm"),
    )) if mtm is not None else None

    delay_unit = _try_param(source, "unit", "delay")
    delay_conn = _try_param(source, "connection", "delay")
    delay_parts: list[pl.LazyFrame] = []
    if delay_unit is not None:
        delay_parts.append(delay_unit.lazy().select(
            pl.col("name").alias("p"),
        ).unique())
    if delay_conn is not None:
        delay_parts.append(delay_conn.lazy().select(
            pl.col("name").alias("p"),
        ).unique())
    delay_lazy = (pl.concat(delay_parts).unique()
                    .with_columns(is_delayed=pl.lit(True))) \
                  if delay_parts else None

    # Step 3 — input/output arc counts per process.
    src_acc: list[pl.LazyFrame] = []
    sink_acc: list[pl.LazyFrame] = []
    uin = _try_entities(source, "unit__inputNode")
    if uin is not None:
        src_acc.append(uin.lazy().select(pl.col("unit").alias("p")))
    uout = _try_entities(source, "unit__outputNode")
    if uout is not None:
        sink_acc.append(uout.lazy().select(pl.col("unit").alias("p")))
    cnn = _try_entities(source, "connection__node__node")
    if cnn is not None:
        src_acc.append(cnn.lazy().select(pl.col("connection").alias("p")))
        sink_acc.append(cnn.lazy().select(pl.col("connection").alias("p")))

    src_counts = (pl.concat(src_acc).group_by("p").agg(
                    pl.len().alias("n_src"))) if src_acc else None
    sink_counts = (pl.concat(sink_acc).group_by("p").agg(
                    pl.len().alias("n_sink"))) if sink_acc else None

    # Step 4 — assemble lazy frame with defaults.
    lf = base.lazy()
    if cm_lazy is not None:
        lf = lf.join(cm_lazy, on="p", how="left")
    else:
        lf = lf.with_columns(ct=pl.lit(None).cast(pl.Utf8))
    # Defaults: unit → constant_efficiency, connection → regular.
    lf = lf.with_columns(
        ct=pl.when(pl.col("ct").is_not_null()).then(pl.col("ct"))
            .when(pl.col("klass") == "unit").then(pl.lit("constant_efficiency"))
            .otherwise(pl.lit("regular"))
    )
    if sm_lazy is not None:
        lf = lf.join(sm_lazy, on="p", how="left")
    else:
        lf = lf.with_columns(startup_raw=pl.lit(None).cast(pl.Utf8))
    lf = lf.with_columns(
        startup=pl.col("startup_raw").fill_null("no_startup")
    ).drop("startup_raw")
    # Step 5 — minimum_time_method overrides startup → linear when active.
    if mtm_lazy is not None:
        lf = lf.join(mtm_lazy, on="p", how="left")
        lf = lf.with_columns(
            startup=pl.when(
                (pl.col("startup") == "no_startup")
                & pl.col("mtm").is_in(["min_uptime", "min_downtime", "both"])
            ).then(pl.lit("linear"))
             .otherwise(pl.col("startup"))
        ).drop("mtm")
    # Step 6 — fork resolution.
    if src_counts is not None:
        lf = lf.join(src_counts, on="p", how="left")
    else:
        lf = lf.with_columns(n_src=pl.lit(0))
    if sink_counts is not None:
        lf = lf.join(sink_counts, on="p", how="left")
    else:
        lf = lf.with_columns(n_sink=pl.lit(0))
    lf = lf.with_columns(
        n_src=pl.col("n_src").fill_null(0),
        n_sink=pl.col("n_sink").fill_null(0),
    )
    if delay_lazy is not None:
        lf = lf.join(delay_lazy, on="p", how="left")
        lf = lf.with_columns(
            is_delayed=pl.col("is_delayed").fill_null(False))
    else:
        lf = lf.with_columns(is_delayed=pl.lit(False))
    lf = lf.with_columns(
        fork=pl.when((pl.col("n_src") > 1) | (pl.col("n_sink") > 1)
                      | pl.col("is_delayed"))
              .then(pl.lit("fork_yes"))
              .otherwise(pl.lit("fork_no"))
    )

    # Step 7 — map (ct, startup, fork) → method via when/then chain.
    expr = pl.lit("not_applicable")
    for (ct, st, fk), method in _METHODS_MAPPING.items():
        expr = (pl.when((pl.col("ct") == ct)
                        & (pl.col("startup") == st)
                        & (pl.col("fork") == fk))
                  .then(pl.lit(method))
                  .otherwise(expr))
    lf = lf.with_columns(method=expr)

    # Step 8 — drop not_applicable; project final columns.
    out = (lf.filter(pl.col("method") != "not_applicable")
              .select("p", "ct", "startup", "fork", "method", "klass")
              .sort("p")
              .collect())
    return out


# ---------------------------------------------------------------------------
# Method-classifier-keyed Param sets (reclassified Projections, §1.3, §1.5)
# ---------------------------------------------------------------------------


def process_indirect_set(source: "InputSource",
                          classified: pl.DataFrame | None = None
                          ) -> pl.DataFrame:
    """Units with internal method ∈ INDIRECT family.  Schema ``[p]``.

    Reclassified-Derived equivalent of the §1.5 ``process_indirect``
    Projection: filter classified processes (units only) to the
    ``method_*_nvar_*`` family (CHP / extraction patterns).  flextool's
    preprocessing only emits unit names here; connections are skipped.
    """
    if classified is None:
        classified = _classify_process_method(source)
    if classified.height == 0:
        return pl.DataFrame(schema={"p": pl.Utf8})
    return (classified.lazy()
              .filter((pl.col("klass") == "unit")
                      & pl.col("method").is_in(list(_METHOD_INDIRECT)))
              .select("p")
              .sort("p")
              .collect())


def _zero_flow_coef_pairs(source: "InputSource",
                              kind: str
                              ) -> pl.DataFrame:
    """Read ``unit__inputNode.flow_coefficient`` (kind='source') or
    ``unit__outputNode.flow_coefficient`` (kind='sink') and return the
    ``(p, source)`` / ``(p, sink)`` pairs whose coefficient is zero.
    flextool drops these from the conversion_indirect LHS / RHS
    (``input.py:_load_indirect`` lines 942-944 / 968-970).
    """
    if kind == "source":
        ec = "unit__inputNode"
        col = "source"
    else:
        ec = "unit__outputNode"
        col = "sink"
    df = _try_param(source, ec, "flow_coefficient")
    if df is None or df.height == 0:
        return pl.DataFrame(schema={"p": pl.Utf8, col: pl.Utf8})
    cols = df.columns
    rename: dict[str, str] = {}
    for c in cols:
        if c == "unit":
            rename[c] = "p"
        elif c == "node":
            rename[c] = col
    out = (df.lazy().rename(rename)
              .with_columns(pl.col("value").cast(pl.Float64, strict=False))
              .filter(pl.col("value") == 0.0)
              .select("p", col))
    return out.unique().collect()


def process_input_flows(source: "InputSource",
                          pss: pl.DataFrame,
                          classified: pl.DataFrame | None = None
                          ) -> pl.DataFrame:
    """Indirect-process input arcs.

    pss filtered to ``sink == p AND p ∈ process_indirect``.  Drops arcs
    whose ``flow_coefficient == 0`` (mirrors
    ``input.py:_load_indirect:942-944``).  Schema ``[p, source, sink]``.
    """
    pi = process_indirect_set(source, classified)
    if pss is None or pss.height == 0 or pi.height == 0:
        return pl.DataFrame(schema={"p": pl.Utf8, "source": pl.Utf8,
                                      "sink": pl.Utf8})
    out_lf = (pss.lazy()
                .filter(pl.col("sink") == pl.col("p"))
                .join(pi.lazy(), on="p", how="inner"))
    zero_src = _zero_flow_coef_pairs(source, "source")
    if zero_src.height > 0:
        out_lf = out_lf.join(zero_src.lazy(), on=["p", "source"], how="anti")
    return out_lf.sort("p", "source", "sink").collect()


def process_output_flows(source: "InputSource",
                           pss: pl.DataFrame,
                           classified: pl.DataFrame | None = None
                           ) -> pl.DataFrame:
    """Indirect-process output arcs (mirror of input).

    Drops arcs whose ``flow_coefficient == 0`` (mirror of
    ``input.py:_load_indirect:968-970``).
    """
    pi = process_indirect_set(source, classified)
    if pss is None or pss.height == 0 or pi.height == 0:
        return pl.DataFrame(schema={"p": pl.Utf8, "source": pl.Utf8,
                                      "sink": pl.Utf8})
    out_lf = (pss.lazy()
                .filter(pl.col("source") == pl.col("p"))
                .join(pi.lazy(), on="p", how="inner"))
    zero_sink = _zero_flow_coef_pairs(source, "sink")
    if zero_sink.height > 0:
        out_lf = out_lf.join(zero_sink.lazy(), on=["p", "sink"], how="anti")
    return out_lf.sort("p", "source", "sink").collect()


def process_indirect_dt(source: "InputSource",
                          dt: pl.DataFrame,
                          classified: pl.DataFrame | None = None
                          ) -> pl.DataFrame:
    """``process_indirect × dt`` cross-product.  Schema ``[p, d, t]``."""
    pi = process_indirect_set(source, classified)
    if pi.height == 0 or dt is None or dt.height == 0:
        return pl.DataFrame(schema={"p": pl.Utf8, "d": pl.Utf8,
                                      "t": pl.Utf8})
    return (pi.lazy()
              .join(dt.lazy(), how="cross")
              .sort("p", "d", "t")
              .collect())


# ---------------------------------------------------------------------------
# §3.3.1 — flow_to_n / flow_from_n (block-aware filter on top of pss)
# ---------------------------------------------------------------------------


def flow_to_n(source: "InputSource",
              pss: pl.DataFrame) -> pl.DataFrame:
    """Augment pss with ``n = sink``.  Schema ``[p, source, sink, n]``.

    Block-compatibility filtering (the ``process_side_block.csv`` /
    ``entity_block.csv`` / ``overlap_set.csv`` triple-join in
    input.py:713-762) is OUT OF SCOPE for the DB-direct overlay —
    those auxiliary tables are themselves preprocessed Derived
    artefacts.  The default behaviour is "no block filtering" which
    is correct on every fixture except those activating non-default
    blocks (``lh2_three_region``, ``5weeks_battery_intraperiod_blocks``
    — explicitly deferred to Batch C in the task spec).
    """
    if pss is None or pss.height == 0:
        return pl.DataFrame(schema={"p": pl.Utf8, "source": pl.Utf8,
                                      "sink": pl.Utf8, "n": pl.Utf8})
    return (pss.lazy()
              .with_columns(n=pl.col("sink"))
              .select("p", "source", "sink", "n")
              .sort("p", "source", "sink", "n")
              .collect())


def flow_from_n(source: "InputSource",
                pss: pl.DataFrame) -> pl.DataFrame:
    """Augment pss with ``n = source``.  Schema ``[p, source, sink, n]``."""
    if pss is None or pss.height == 0:
        return pl.DataFrame(schema={"p": pl.Utf8, "source": pl.Utf8,
                                      "sink": pl.Utf8, "n": pl.Utf8})
    return (pss.lazy()
              .with_columns(n=pl.col("source"))
              .select("p", "source", "sink", "n")
              .sort("p", "source", "sink", "n")
              .collect())


# ---------------------------------------------------------------------------
# §3.3.3 — p_flow_upper_existing (existing/unitsize per arc)
# ---------------------------------------------------------------------------


def _flow_upper_existing_from_chained_csv(source: "InputSource",
                                              pss: pl.DataFrame,
                                              workdir: Path | None,
                                              ) -> "Param | None":
    """Γ.6.D — build ``p_flow_upper_existing`` from the canonical
    ``solve_data/p_entity_all_existing.csv`` when it exists, divided by
    the per-entity unitsize cascade and cross-joined with ``pss``.

    Returns None when the CSV is absent (caller falls back to the
    raw-existing path).
    """
    pae_csv = _read_p_entity_all_existing_csv(workdir)
    if pae_csv is None or pae_csv.height == 0:
        return None
    proc_set, _, _ = _entity_classes_lookup(source)
    per_proc = pae_csv.filter(pl.col("e").is_in(list(proc_set)))
    if per_proc.height == 0:
        return None
    us_lf = _entity_unitsize_lf(source)
    base = (per_proc.lazy()
              .rename({"e": "p"})
              .join(us_lf.rename({"e": "p"}), on="p", how="left")
              .with_columns(us=pl.col("us").fill_null(1000.0))
              .with_columns(value=pl.col("value") / pl.col("us"))
              .select("p", "d", "value"))
    if pss is None or pss.height == 0:
        return None
    out = (pss.lazy()
              .join(base, on="p", how="inner")
              .select("p", "source", "sink", "d", "value")
              .sort("p", "source", "sink", "d")
              .collect())
    if out.height == 0:
        return None
    return Param(("p", "source", "sink", "d"), out)


def p_flow_upper_existing_from_source(source: "InputSource",
                                         pss: pl.DataFrame,
                                         active_solve: str | None = None,
                                         workdir: Path | None = None,
                                         ) -> "Param | None":
    """Cross-join ``base_cap_pd = existing_capacity / unitsize`` per
    (p, d) with ``pss`` to produce per-arc structural existing bound.

    ``base_cap_pd`` is sourced from ``unit/connection.existing`` +
    ``unit/connection.virtual_unitsize`` parameters at the source
    boundary; default unitsize is 1.0.  The CSV path uses the
    preprocessed ``p_entity_period_existing_capacity.csv`` which
    bakes in the lifetime-cumulative chain — for fixtures without
    multi-period invest (single-period) the value collapses to plain
    ``existing``.

    Schema: ``Param[(p, source, sink, d), value]``.

    Returns ``None`` when the source has no usable ``existing`` data —
    caller falls back to CSV value.
    """
    # Γ.6.D — prefer the canonical ``p_entity_all_existing.csv`` when
    # available (carries chained existing + lifetime gate from the
    # multi-solve handoff).
    chained = _flow_upper_existing_from_chained_csv(source, pss, workdir)
    if chained is not None:
        return chained

    # Source `existing` lives on unit / connection / node; we want
    # process-side only.  The Spine classes carry a 1d_map (period → value).
    # Use ``parameter_explicit`` so default-broadcast rows don't pollute
    # the cap frame; entities absent from ``existing`` have no cap.
    parts: list[pl.LazyFrame] = []
    e_unit = _try_param_explicit(source, "unit", "existing")
    if e_unit is not None:
        parts.append(e_unit.lazy().select(
            pl.col("name").alias("p"),
            pl.col(_index_col(e_unit, "period")).alias("d"),
            pl.col("value").cast(pl.Float64).alias("cap"),
        ) if _index_col(e_unit, "period") in e_unit.columns
            else _broadcast_existing_to_pd(e_unit, source, workdir))
    e_conn = _try_param_explicit(source, "connection", "existing")
    if e_conn is not None:
        parts.append(e_conn.lazy().select(
            pl.col("name").alias("p"),
            pl.col(_index_col(e_conn, "period")).alias("d"),
            pl.col("value").cast(pl.Float64).alias("cap"),
        ) if _index_col(e_conn, "period") in e_conn.columns
            else _broadcast_existing_to_pd(e_conn, source, workdir))
    if not parts:
        return None

    cap = pl.concat(parts).unique()

    # Unitsize cascade per ``entity_period_calc_params.py:186-202``.
    # Use ``parameter_explicit`` to mirror flextool's ``p_unit.get(name,
    # None)`` semantic: only entities with an explicit JSON row count as
    # "set"; default-broadcast rows fall through to the cascade.
    us_parts: list[pl.LazyFrame] = []
    for cls in ("unit", "connection"):
        us_df = _try_param_explicit(source, cls, "virtual_unitsize")
        if us_df is None or us_df.height == 0:
            continue
        us_parts.append(us_df.lazy().select(
            pl.col("name").alias("p"),
            pl.col("value").cast(pl.Float64).alias("us"),
        ))
    if us_parts:
        us = pl.concat(us_parts).unique()
        cap = cap.join(us, on="p", how="left")
    else:
        cap = cap.with_columns(us=pl.lit(None).cast(pl.Float64))
    # flextool's unitsize cascade (mod L1279 / entity_period_calc_params
    # _params.py:159-198): virtual_unitsize → existing → 1000.
    cap = cap.with_columns(
        us=pl.when(pl.col("us").is_not_null() & (pl.col("us") != 0.0))
              .then(pl.col("us"))
              .when(pl.col("cap").is_not_null() & (pl.col("cap") != 0.0))
              .then(pl.col("cap"))
              .otherwise(pl.lit(1000.0))
    )

    base = (cap.with_columns(value=pl.col("cap") / pl.col("us"))
               .select("p", "d", "value"))

    if pss is None or pss.height == 0:
        return None
    out = (pss.lazy()
              .join(base, on="p", how="inner")
              .select("p", "source", "sink", "d", "value")
              .sort("p", "source", "sink", "d")
              .collect())
    if out.height == 0:
        return None
    # Γ.6.D — lifetime gate.  CSV path's p_flow_upper_existing pulls
    # from ``p_entity_period_existing_capacity.csv`` which inherits the
    # lifetime-gated existing chain.  We mirror by zeroing rows whose
    # (p, d) is past the ``reinvest_choice`` / ``no_investment`` expiry.
    expired = _lifetime_expired_pairs(
        source, active_solve, workdir,
        methods=("reinvest_choice", "no_investment"))
    if expired:
        expired_lf = pl.DataFrame(
            [(e, d) for (e, d) in expired],
            schema=["p", "d"], orient="row").lazy()
        out = (out.lazy()
                  .join(expired_lf.with_columns(_expired=pl.lit(True)),
                          on=["p", "d"], how="left")
                  .with_columns(
                      value=pl.when(pl.col("_expired").fill_null(False))
                                .then(0.0)
                                .otherwise(pl.col("value"))
                  )
                  .select("p", "source", "sink", "d", "value")
                  .sort("p", "source", "sink", "d")
                  .collect())
    return Param(("p", "source", "sink", "d"), out)


def _index_col(df: pl.DataFrame, candidate: str) -> str:
    """Probe for ``candidate`` in df.columns; if missing, return the
    first non-(name, value) column.  Helps when the source emits the
    period index under an alternative name (``x``, ``period``).
    """
    if candidate in df.columns:
        return candidate
    for c in df.columns:
        if c not in ("name", "value"):
            return c
    return candidate  # caller will trip a clear error


def _broadcast_existing_to_pd(df: pl.DataFrame,
                                 source: "InputSource",
                                 workdir: Path | None = None,
                                 ) -> pl.LazyFrame:
    """Fallback: when ``existing`` is a scalar (not 1d_map), broadcast
    over ``period_in_use``.

    flextool's per_solve_sets.py:95-101 defines ``period_in_use`` as the
    union of realised + invest periods plus stochastic-branch siblings;
    ``p_flow_upper_existing`` is keyed on (p, d) for d ∈ period_in_use,
    so we mirror that union here.  When ``workdir`` is provided we read
    the canonical ``period_in_use_set.csv`` (includes stochastic
    branches); otherwise fall back to the realised + invest union.
    """
    if workdir is not None:
        piu_path = Path(workdir) / "solve_data" / "period_in_use_set.csv"
        if piu_path.exists():
            piu_df = _read_csv_file(piu_path)
            if piu_df.height > 0 and "period" in piu_df.columns:
                pd_lf = (piu_df.lazy()
                            .select(pl.col("period").alias("d"))
                            .unique())
                return (df.lazy()
                          .select(pl.col("name").alias("p"),
                                   pl.col("value").cast(pl.Float64).alias("cap"))
                          .join(pd_lf, how="cross")
                          .select("p", "d", "cap"))
    parts: list[pl.LazyFrame] = []
    for par in ("realized_periods", "invest_periods"):
        rp = _try_param(source, "solve", par)
        if rp is None:
            continue
        parts.append(rp.lazy().select(pl.col("value").alias("d")))
    if not parts:
        return pl.DataFrame(schema={
            "p": pl.Utf8, "d": pl.Utf8, "cap": pl.Float64,
        }).lazy()
    pd_lf = pl.concat(parts).unique()
    return (df.lazy()
              .select(pl.col("name").alias("p"),
                       pl.col("value").cast(pl.Float64).alias("cap"))
              .join(pd_lf, how="cross")
              .select("p", "d", "cap"))


# ---------------------------------------------------------------------------
# §3.5.1 — p_flow_constraint_coef
# ---------------------------------------------------------------------------


def p_flow_constraint_coef_from_source(source: "InputSource",
                                          pss: pl.DataFrame,
                                          ) -> "Param | None":
    """User-defined per-arc (p, source, sink, c) flow_coefficient.

    Algorithm (audit §3.5.1, input.py:1005-1019):
      1. Pull unit__inputNode / unit__outputNode /
         connection__node.constraint_flow_coefficient.  Each row carries
         ``(process, node, constraint, coef)``.
      2. Inner-join with pss on (p, n=source) → source-leg rows.
      3. Inner-join with pss on (p, n=sink) → sink-leg rows.
      4. Concat + group_by(p, source, sink, c).sum.

    Returns ``None`` when no flow_coefficient rows exist.
    """
    if pss is None or pss.height == 0:
        return None
    parts: list[pl.LazyFrame] = []
    for entity_class in ("unit__inputNode", "unit__outputNode",
                          "connection__node"):
        df = _try_param(source, entity_class, "constraint_flow_coefficient")
        if df is None:
            continue
        # Schema: <process_col>, <node_col>, <constraint_index_col>, value.
        # constraint_flow_coefficient is a 1d_map[constraint] → coef.
        cols = df.columns
        # Disambiguate process / node columns by entity_class layout.
        p_col = ("unit" if entity_class.startswith("unit")
                  else "connection")
        n_col = "node"
        c_col = next((c for c in cols
                       if c not in (p_col, n_col, "value")), None)
        if c_col is None:
            continue
        parts.append(df.lazy().select(
            pl.col(p_col).alias("p"),
            pl.col(n_col).alias("n"),
            pl.col(c_col).alias("c"),
            pl.col("value").cast(pl.Float64).alias("coef"),
        ))
    if not parts:
        return None
    coef = pl.concat(parts)

    src_match = (pss.lazy()
        .join(coef, left_on=["p", "source"], right_on=["p", "n"],
                how="inner")
        .select("p", "source", "sink", "c", "coef"))
    sink_match = (pss.lazy()
        .join(coef, left_on=["p", "sink"], right_on=["p", "n"],
                how="inner")
        .select("p", "source", "sink", "c", "coef"))
    joined = (pl.concat([src_match, sink_match], how="vertical")
                .group_by(["p", "source", "sink", "c"])
                .agg(pl.col("coef").sum().alias("value"))
                .sort("p", "source", "sink", "c")
                .collect())
    if joined.height == 0:
        return None
    return Param(("p", "source", "sink", "c"), joined)


# ---------------------------------------------------------------------------
# §3.10.1 — p_pssdt_varCost
# ---------------------------------------------------------------------------


def p_pssdt_varCost_from_source(source: "InputSource",
                                   pss: pl.DataFrame,
                                   dt: pl.DataFrame,
                                   ) -> "Param | None":
    """Long-format (p, source, sink, d, t, value) other_operational_cost
    aggregate.  Mirrors flextool's
    ``write_pdtProcess__source__sink__dt_varCost_pair`` (audit §3.10.1).

    The CSV path collects three separate cost streams:
      1. ``unit__inputNode.other_operational_cost`` per arc-side source.
      2. ``unit__outputNode.other_operational_cost`` per arc-side sink.
      3. ``unit/connection.other_operational_cost`` per process.

    flextool's preprocessor sums them per (p, source, sink, d, t).  Each
    Map dimension can be: scalar, 1d_map(period), time_series(t),
    or 2d_map(period × t).  Filtering out value=0 keeps the row count
    aligned with the CSV's "drop zero coefficients" pass.

    Returns ``None`` when no varCost rows are non-zero on any side.
    """
    if pss is None or pss.height == 0 or dt is None or dt.height == 0:
        return None

    contributions: list[pl.LazyFrame] = []

    def _broadcast_to_psd_dt(df: pl.DataFrame, side: str) -> pl.LazyFrame | None:
        """Broadcast a (entity, [period], [t], value) frame to
        (p, source, sink, d, t, value) by joining onto pss × dt.

        ``side ∈ {'source', 'sink', 'process'}`` — controls how the
        entity column is matched against pss's (p, source, sink) tuple.
        """
        cols = df.columns
        if side == "source":
            ent_p_col = "unit"
            ent_n_col = "node"
        elif side == "sink":
            ent_p_col = "unit"
            ent_n_col = "node"
        elif side == "process":
            ent_p_col = "name"
            ent_n_col = None
        else:
            return None

        # Detect period / t index columns.
        non_idx = {ent_p_col, "value"}
        if ent_n_col is not None:
            non_idx.add(ent_n_col)
        idx_cols = [c for c in cols if c not in non_idx]
        has_period = any(c in idx_cols for c in ("period", "d"))
        has_t = any(c in idx_cols for c in ("t", "time", "step"))
        period_col = next((c for c in ("period", "d") if c in idx_cols), None)
        t_col = next((c for c in ("t", "time", "step") if c in idx_cols), None)

        # Build base lazyframe and rename keys.
        select_exprs = [pl.col(ent_p_col).alias("p"),
                         pl.col("value").cast(pl.Float64).alias("v")]
        if ent_n_col is not None:
            select_exprs.append(pl.col(ent_n_col).alias("n"))
        if period_col is not None:
            select_exprs.append(pl.col(period_col).alias("d"))
        if t_col is not None:
            select_exprs.append(pl.col(t_col).alias("t"))
        lf = df.lazy().select(*select_exprs).filter(pl.col("v") != 0.0)

        # Match against pss to get (p, source, sink) tuples.
        if side == "source":
            # n binds source.
            psk = (pss.lazy()
                .join(lf, left_on=["p", "source"], right_on=["p", "n"],
                       how="inner"))
        elif side == "sink":
            psk = (pss.lazy()
                .join(lf, left_on=["p", "sink"], right_on=["p", "n"],
                       how="inner"))
        else:
            # process-level, no n match.
            psk = pss.lazy().join(lf, on="p", how="inner")

        # Broadcast over dt according to which dims are present.
        dt_lf = dt.lazy()
        if has_period and has_t:
            return (psk.join(dt_lf, on=["d", "t"], how="inner")
                       .select("p", "source", "sink", "d", "t", "v"))
        if has_period:
            return (psk.join(dt_lf, on="d", how="inner")
                       .select("p", "source", "sink", "d", "t", "v"))
        if has_t:
            return (psk.join(dt_lf, on="t", how="inner")
                       .select("p", "source", "sink", "d", "t", "v"))
        # Scalar — broadcast over full dt.
        return (psk.join(dt_lf, how="cross")
                   .select("p", "source", "sink", "d", "t", "v"))

    # 1) unit__inputNode.other_operational_cost — source-side
    df = _try_param(source, "unit__inputNode", "other_operational_cost")
    if df is not None:
        c = _broadcast_to_psd_dt(df, "source")
        if c is not None:
            contributions.append(c)

    # 2) unit__outputNode.other_operational_cost — sink-side
    df = _try_param(source, "unit__outputNode", "other_operational_cost")
    if df is not None:
        c = _broadcast_to_psd_dt(df, "sink")
        if c is not None:
            contributions.append(c)

    # 3) unit / connection.other_operational_cost — process-level
    for ec in ("unit", "connection"):
        df = _try_param(source, ec, "other_operational_cost")
        if df is not None:
            c = _broadcast_to_psd_dt(df, "process")
            if c is not None:
                contributions.append(c)

    if not contributions:
        return None

    summed = (pl.concat(contributions, how="vertical")
                .group_by(["p", "source", "sink", "d", "t"])
                .agg(pl.col("v").sum().alias("value"))
                .filter(pl.col("value") != 0.0)
                .sort("p", "source", "sink", "d", "t")
                .collect())
    if summed.height == 0:
        return None
    return Param(("p", "source", "sink", "d", "t"), summed)


# ---------------------------------------------------------------------------
# §3.3.4 — p_slope (efficiency-curve slope)
# ---------------------------------------------------------------------------


def p_slope_from_source(source: "InputSource",
                          dt: pl.DataFrame,
                          classified: pl.DataFrame | None = None,
                          ) -> "Param | None":
    """Conversion-curve slope per (p, d, t).

    For ``constant_efficiency`` ct_method: slope = 1 / efficiency.
    For ``min_load_efficiency`` ct_method: slope is linearised between
    (min_load, efficiency_at_min_load) and (1.0, efficiency); per
    audit §3.3.4 / entity_period_calc_params.py:1209-1306.

    Time-varying efficiency Maps are broadcast to the (d, t) grid;
    scalar efficiency is broadcast to the full grid.

    Returns ``None`` when no usable efficiency data is available.
    """
    if dt is None or dt.height == 0:
        return None
    if classified is None:
        classified = _classify_process_method(source)
    if classified.height == 0:
        return None
    # Pull efficiency / efficiency_at_min_load / min_load from units +
    # transfer connections (efficiency lives on both unit and connection
    # though connection.efficiency may be implicit via transfer_method).
    eff_unit = _try_param(source, "unit", "efficiency")
    eff_conn = _try_param(source, "connection", "efficiency")
    eff_at_min = _try_param(source, "unit", "efficiency_at_min_load")
    minload = _try_param(source, "unit", "min_load")

    # Build a per-(p, d, t) efficiency lazyframe — union unit + connection.
    eff_lfs: list[pl.LazyFrame] = []
    if eff_unit is not None:
        e = _broadcast_param_to_dt(eff_unit, dt, value_col_alias="eta")
        if e is not None:
            eff_lfs.append(e)
    if eff_conn is not None:
        e = _broadcast_param_to_dt(eff_conn, dt, value_col_alias="eta")
        if e is not None:
            eff_lfs.append(e)
    # Default-fill for processes in the classifier but missing from the
    # explicit efficiency rows.  flextool's default is 1.0 (constant_eff
    # ct_method baseline) — broadcast to the full classified × dt grid
    # then left-join with explicit rows so explicit overrides win.
    classified_dt = (classified.lazy()
        .select("p", "ct", "klass")
        .join(dt.lazy(), how="cross")
        .select("p", "d", "t", "ct", "klass"))
    if eff_lfs:
        explicit = pl.concat(eff_lfs).unique(subset=["p", "d", "t"])
        eff_lf = (classified_dt
                    .join(explicit, on=["p", "d", "t"], how="left")
                    .with_columns(eta=pl.col("eta").fill_null(1.0)))
    else:
        eff_lf = classified_dt.with_columns(eta=pl.lit(1.0))

    # constant_efficiency branch: slope = 1 / eta.
    base = eff_lf  # already has p, d, t, ct, klass, eta

    # min_load_efficiency branch (when applicable): slope =
    #   (eta - min_load * eta_min) / ((1 - min_load) * eta * eta_min).
    # Linearisation: between (min_load, eta_min) and (1.0, eta), slope
    # of input/output (Δin/Δout) ≈ as in entity_period_calc_params.
    if eff_at_min is not None and minload is not None:
        eta_min_lf = _broadcast_param_to_dt(eff_at_min, dt,
                                              value_col_alias="eta_min")
        ml_lf = (minload.lazy().select(
            pl.col("name").alias("p"),
            pl.col("value").cast(pl.Float64).alias("min_load"),
        ))
        if eta_min_lf is not None:
            base = (base.join(eta_min_lf, on=["p", "d", "t"], how="left")
                         .join(ml_lf, on="p", how="left"))
        else:
            base = (base
                .with_columns(eta_min=pl.lit(None).cast(pl.Float64))
                .join(ml_lf, on="p", how="left"))
    else:
        base = (base
            .with_columns(eta_min=pl.lit(None).cast(pl.Float64),
                            min_load=pl.lit(None).cast(pl.Float64)))

    # slope expression: branch on ct.
    base = base.with_columns(
        slope=pl.when(pl.col("ct") == "min_load_efficiency")
                .then(
                    # Δinput / Δoutput between the two anchor points.
                    # input(min_load) = min_load / eta_min,
                    # input(1.0)      = 1.0 / eta.
                    # slope = (1/eta - min_load/eta_min) / (1.0 - min_load).
                    (1.0 / pl.col("eta")
                       - pl.col("min_load") / pl.col("eta_min"))
                    / (1.0 - pl.col("min_load"))
                )
              .otherwise(1.0 / pl.col("eta"))
    )
    out = (base.filter(pl.col("slope").is_not_null())
                .select("p", "d", "t",
                         pl.col("slope").alias("value"))
                .sort("p", "d", "t")
                .collect())
    if out.height == 0:
        return None
    return Param(("p", "d", "t"), out)


def _broadcast_param_to_dt(df: pl.DataFrame,
                              dt: pl.DataFrame,
                              value_col_alias: str = "value",
                              ) -> pl.LazyFrame | None:
    """Broadcast a Spine Map / scalar parameter onto the (d, t) grid.

    Accepts:
      * scalar: broadcast over full dt.
      * 1d_map(period): broadcast over t for each period.
      * time_series(t): broadcast over d for each t.
      * 2d_map(period × t): keep both dims.

    Returns a lazyframe with schema ``[p, d, t, <value_col_alias>]``,
    or ``None`` when df is empty.
    """
    if df is None or df.height == 0:
        return None
    cols = df.columns
    has_period = any(c in cols for c in ("period", "d"))
    has_t = any(c in cols for c in ("t", "time", "step"))
    period_col = next((c for c in ("period", "d") if c in cols), None)
    t_col = next((c for c in ("t", "time", "step") if c in cols), None)
    dt_lf = dt.lazy()
    base = df.lazy().select(
        pl.col("name").alias("p"),
        *([pl.col(period_col).alias("d")] if period_col else []),
        *([pl.col(t_col).alias("t")] if t_col else []),
        pl.col("value").cast(pl.Float64).alias(value_col_alias),
    )
    if has_period and has_t:
        return base.join(dt_lf, on=["d", "t"], how="inner") \
                    .select("p", "d", "t", value_col_alias)
    if has_period:
        return base.join(dt_lf, on="d", how="inner") \
                    .select("p", "d", "t", value_col_alias)
    if has_t:
        return base.join(dt_lf, on="t", how="inner") \
                    .select("p", "d", "t", value_col_alias)
    return base.join(dt_lf, how="cross") \
                .select("p", "d", "t", value_col_alias)


# ===========================================================================
# Γ.3.B integration entrypoint
# ===========================================================================


DERIVED_B_FIELDS = (
    "p_flow_upper_existing",
    "p_flow_constraint_coef",
    "p_pssdt_varCost",
    "p_slope",
)


def apply_derived_b(
    flex_data: object,
    source: "InputSource",
    workdir: Path,
    *,
    ctx: "SolveContext | None" = None,
) -> None:
    """Apply Γ.3.B Derived Params, mutating ``flex_data`` in place.

    The classifier (``_classify_process_method``) is the single
    dependency for the reclassified ``process_indirect`` / ``flow_to_n``
    family; it's computed once per call and threaded through the
    per-Param helpers as a kwarg.

    Δ.3 replaced the previous ``derived_overrides_b`` dict-return;
    Δ.4 deleted the deprecated wrapper alias.
    """
    # Active solve (used for existing-period broadcast in some helpers).
    active_solve = ctx.solve_name if ctx is not None else _read_active_solve(workdir)

    # Build the classifier once.  Δ.12b: hard errors here would
    # indicate a malformed source — let them propagate instead of
    # silently producing an empty schema.
    classified = _classify_process_method(source)

    # Topology — already-computed CSV-side process_source_sink (a plain
    # DataFrame on FlexData; see input.py:252).
    pss_frame = getattr(flex_data, "process_source_sink", None)
    dt_csv = getattr(flex_data, "dt", None)

    # process_indirect / process_input_flows / process_output_flows /
    # process_indirect_dt.  Δ.12b: helpers produce or raise.  ``None``
    # / empty frame == "no indirect units" — same semantic the seed
    # produces.  We keep the ``height > 0`` filter on the assignment
    # so the SET-frame contract (None or non-empty) is preserved.
    if classified.height > 0:
        pi_db = process_indirect_set(source, classified)
        if pi_db is not None and pi_db.height > 0:
            flex_data.process_indirect = pi_db
        else:
            flex_data.process_indirect = None

        if pss_frame is not None and pss_frame.height > 0:
            pif_db = process_input_flows(source, pss_frame, classified)
            flex_data.process_input_flows = (
                pif_db if pif_db is not None and pif_db.height > 0
                else None)

            pof_db = process_output_flows(source, pss_frame, classified)
            flex_data.process_output_flows = (
                pof_db if pof_db is not None and pof_db.height > 0
                else None)

        if dt_csv is not None and dt_csv.height > 0:
            pid_db = process_indirect_dt(source, dt_csv, classified)
            flex_data.process_indirect_dt = (
                pid_db if pid_db is not None and pid_db.height > 0
                else None)

    # flow_to_n / flow_from_n — Δ.9 closes the Δ.3 gap: the lazy
    # cluster E port now applies the block-aware filter on the source-
    # driven path, mirroring ``input.py::_load_process_topology`` lines
    # 728-782.  Single-block fixtures are no-ops; multi-block fixtures
    # (work_lh2_three_region) drop arc rows incompatible with the
    # destination node's block.
    # Δ.12b: bundle load failure is non-fatal (workdir without block
    # CSVs => no block filtering).  Helper exceptions propagate.
    if pss_frame is not None and pss_frame.height > 0:
        from flextool.engine_polars._derived_block import (
            flow_to_n_block_filtered,
            flow_from_n_block_filtered,
            flow_from_nodeBalance_seed,
            load_block_bundle,
        )
        try:
            bundle = load_block_bundle(workdir)
        except Exception:
            bundle = None
        ftn_db = flow_to_n_block_filtered(pss_frame, bundle)
        ffn_db = flow_from_n_block_filtered(pss_frame, bundle)
        if ftn_db is not None and ftn_db.height > 0:
            flex_data.flow_to_n = ftn_db
        if ffn_db is not None and ffn_db.height > 0:
            flex_data.flow_from_n = ffn_db

        # ─── Δ.27 flow_from_nodeBalance_{eff,noEff} ─────────────────
        # Source-side nodeBalance topology — native port of the
        # inline derivation in ``input.py::_load_storage`` lines
        # 1658-1705.  Without these, source-side flows from
        # nodeBalance nodes (e.g. transmission discharge into a
        # balance node) don't appear in nodeBalance_eq and the LP
        # decouples on multi-arc fixtures
        # (``work_lh2_three_region``).  ``apply_derived_e`` step 4c
        # is now an idempotent overlay: it re-applies the same
        # block-compat filter when bundle data is present, but the
        # seed produced here already includes the filter so the
        # overlay is a no-op.
        nb_for_ffnb = getattr(flex_data, "nodeBalance", None)
        pss_eff_frame = getattr(flex_data, "process_source_sink_eff", None)
        pss_noEff_frame = getattr(flex_data, "process_source_sink_noEff",
                                    None)
        if (nb_for_ffnb is not None and nb_for_ffnb.height > 0
                and pss_eff_frame is not None
                and pss_eff_frame.height > 0):
            ffnb_eff = flow_from_nodeBalance_seed(
                pss_eff_frame, nb_for_ffnb, bundle)
            if ffnb_eff is not None and ffnb_eff.height > 0:
                flex_data.flow_from_nodeBalance_eff = ffnb_eff
        if (nb_for_ffnb is not None and nb_for_ffnb.height > 0
                and pss_noEff_frame is not None
                and pss_noEff_frame.height > 0):
            ffnb_noEff = flow_from_nodeBalance_seed(
                pss_noEff_frame, nb_for_ffnb, bundle)
            if ffnb_noEff is not None and ffnb_noEff.height > 0:
                flex_data.flow_from_nodeBalance_noEff = ffnb_noEff

    # ─── §3.3.3 p_flow_upper_existing ──────────────────────────────────
    # Δ.12b: unconditional when pss is non-empty.  Helper returns None
    # for fixtures without entity-existing-capacity.
    if pss_frame is not None and pss_frame.height > 0:
        flex_data.p_flow_upper_existing = p_flow_upper_existing_from_source(
            source, pss_frame, active_solve, workdir)

    # ─── §3.5.1 p_flow_constraint_coef ─────────────────────────────────
    # Δ.12b: unconditional when pss is non-empty.
    if pss_frame is not None and pss_frame.height > 0:
        flex_data.p_flow_constraint_coef = p_flow_constraint_coef_from_source(
            source, pss_frame)

    # ─── §3.10.1 p_pssdt_varCost ───────────────────────────────────────
    # Δ.12b: unconditional when (pss, dt) are non-empty.
    if (pss_frame is not None and pss_frame.height > 0
            and dt_csv is not None and dt_csv.height > 0):
        flex_data.p_pssdt_varCost = p_pssdt_varCost_from_source(
            source, pss_frame, dt_csv)

    # ─── §3.3.4 p_slope ────────────────────────────────────────────────
    # Δ.12b: unconditional when (dt, classified) are non-empty.
    if dt_csv is not None and dt_csv.height > 0 and classified.height > 0:
        flex_data.p_slope = p_slope_from_source(source, dt_csv, classified)

    # ─── §F.1 p_unitsize  (Δ.10 cluster F) ─────────────────────────────
    # Per-process unitsize cascade restricted to processes appearing in
    # ``pss``.  Mirrors ``input.py:800-825``'s
    # ``unitsize_long.filter(p ∈ pss["p"].unique())``.
    # Δ.12b: unconditional when pss is non-empty.
    if pss_frame is not None and pss_frame.height > 0:
        from ._derived_arithmetic import p_unitsize_from_source
        flex_data.p_unitsize = p_unitsize_from_source(source, pss_frame)

    # ─── §F.1b p_all_entity_unitsize  (scaling family — all entities) ────
    # Unfiltered entity unitsize covering processes + connections + nodes.
    # Used by the scaling analyzer (scaling.py:analyze_solve) to compute the
    # full entity-unitsize spread including node entries.
    # Note: _entity_unitsize_lf returns all entities (unit ∪ node ∪ connection)
    # with the cascade: virtual_unitsize OR existing OR 1000.0.
    _all_us_lf = _entity_unitsize_lf(source)
    _all_us_df = _all_us_lf.rename({"us": "value"}).collect()
    if _all_us_df.height > 0:
        from polar_high import Param as _Param
        flex_data.p_all_entity_unitsize = _Param(("e",), _all_us_df)

    # ─── §F.4 p_process_source_flow_coef / p_process_sink_flow_coef ────
    # (Δ.10 cluster F).  Mirrors ``input.py:_load_indirect`` lines
    # 950-1002.  Anti-joins zero-coef rows out of the indirect-process
    # input/output flow sets and emits a Param keyed on (p, source) /
    # (p, sink) when any non-default, non-zero coef remains.
    # Δ.12b: helper exceptions propagate; the anti-join inner gate
    # (only when zero-rows exist) is preserved as a structural filter.
    from ._derived_arithmetic import (
        p_process_source_flow_coef_from_source,
        p_process_sink_flow_coef_from_source,
    )
    pif = getattr(flex_data, "process_input_flows", None)
    pof = getattr(flex_data, "process_output_flows", None)
    z_src, p_src_coef = p_process_source_flow_coef_from_source(source, pif)
    if z_src is not None and z_src.height > 0 and pif is not None \
            and pif.height > 0:
        new_pif = pif.join(z_src, on=["p", "source"], how="anti")
        if new_pif.height < pif.height:
            flex_data.process_input_flows = new_pif
    # TODO(Δ.12b helper-fix): p_process_source_flow_coef_from_source
    # returns None when no zero-coef rows exist in the source — but the
    # seed-side _load_indirect builds the coefficient Param from a
    # different code path (process__commodity__node_flow_coefficient.csv).
    # Keep the conditional assignment until the helpers converge on a
    # single producer.
    if p_src_coef is not None:
        flex_data.p_process_source_flow_coef = p_src_coef

    z_sink, p_sink_coef = p_process_sink_flow_coef_from_source(source, pof)
    if z_sink is not None and z_sink.height > 0 and pof is not None \
            and pof.height > 0:
        new_pof = pof.join(z_sink, on=["p", "sink"], how="anti")
        if new_pof.height < pof.height:
            flex_data.process_output_flows = new_pof
    if p_sink_coef is not None:
        flex_data.p_process_sink_flow_coef = p_sink_coef


# ===========================================================================
# Γ.3.C — invest/divest + online/UC + group slack + existing fixed cost
# ===========================================================================
#
# Per the audit doc §3.7 / §3.8 / §3.11 / §3.12.  Each helper composes a
# lazy chain over ``InputSource`` reads and ``.collect()``s once at the
# boundary.  Each overlay is gated on a frame-equal pre-check against the
# CSV-loaded value (``apply_derived_c``) so multi-year invest cascades
# / scaling-active fixtures fall back to the CSV value instead of being
# corrupted by a simple-path approximation.
#
# Integration: ``apply_derived_c`` runs after Γ.3.A + Γ.3.B; see
# ``input.py::load_flextool``.

# ---------------------------------------------------------------------------
# §3.7.0 — Helpers (active-solve scope, period_invest, p_years_d, etc.)
# ---------------------------------------------------------------------------


def _resolve_synthetic_solve(source: "InputSource",
                                  active_solve: str | None,
                                  ) -> tuple[str, str] | None:
    """Recognise a synthetic per-sub-solve name of the form
    ``<base>_<anchor>`` and return ``(base, anchor)``.

    Δ.19 — flextool's orchestrator synthesises per-period sub-solve
    names at runtime by joining a Spine ``solve`` name with one of its
    ``invest_periods`` anchor keys (see ``_solve_config.py:618-660`` —
    ``periods_to_tuples`` walks the outer Map and calls
    ``duplicate_solve(<base>, <base>_<anchor>)``).  These synthetic
    names don't exist as rows in Spine; the per-solve override chain
    would otherwise return None for every ``solve``-keyed parameter.

    Algorithm: try every suffix split (rightmost underscore first) and
    return the first ``(base, anchor)`` where ``base`` is a Spine
    ``solve`` entity.  ``base`` may itself contain underscores
    (``invest_5weeks_p2020`` splits as ``base='invest_5weeks'``,
    ``anchor='p2020'``).

    Returns ``None`` when no split matches a Spine solve.
    """
    if active_solve is None or "_" not in active_solve:
        return None
    try:
        ents = source.entities("solve")
    except KeyError:
        return None
    if ents is None or ents.height == 0:
        return None
    name_col = "name" if "name" in ents.columns else ents.columns[0]
    spine_solves = set(ents[name_col].to_list())
    parts = active_solve.split("_")
    for i in range(len(parts) - 1, 0, -1):
        base = "_".join(parts[:i])
        anchor = "_".join(parts[i:])
        if base in spine_solves:
            return base, anchor
    return None


def _solve_periods(source: "InputSource", active_solve: str,
                    parameter_name: str) -> list[str] | None:
    """Read the period-list-valued ``solve.<parameter>`` for the active solve.

    ``solve.realized_periods`` is stored as a Spine Array
    (``[name, i (int), value=period]``); ``solve.invest_periods`` is
    stored either as an Array (single-anchor solves) or as a 2D Map
    (nested multi-invest solves: ``[name, x=anchor, i=period,
    value="yes"]``).  This helper returns the period list regardless of
    shape.

    Δ.19 — when ``active_solve`` doesn't appear in the param table,
    fall back to the synthetic ``<base>_<anchor>`` recognition: for
    Map-shaped params filter ``name==base AND x==anchor`` and return
    the ``i`` values; for Array-shaped params return ``[anchor]``
    (the single realised period of the synthetic sub-solve).
    """
    if active_solve is None:
        return None
    df = _try_param(source, "solve", parameter_name)
    if df is None:
        return None
    is_map = "x" in df.columns
    period_col = "i" if is_map else "value"
    sub = df.filter(pl.col("name") == active_solve)
    if sub.height == 0:
        # Synthetic fallback — split active_solve as <base>_<anchor>.
        resolved = _resolve_synthetic_solve(source, active_solve)
        if resolved is None:
            return None
        base, anchor = resolved
        if is_map:
            sub = df.filter((pl.col("name") == base)
                              & (pl.col("x") == anchor))
            if sub.height == 0:
                return None
            return sub[period_col].cast(pl.Utf8, strict=False).to_list()
        # Array param: synthetic sub-solve realises only its anchor.
        # Confirm the anchor is in base's array; if not, return None.
        base_sub = df.filter(pl.col("name") == base)
        if base_sub.height == 0:
            return None
        base_periods = set(base_sub["value"].cast(pl.Utf8, strict=False).to_list())
        return [anchor] if anchor in base_periods else None
    # Non-synthetic path.  For Map params filter to the (most-common)
    # anchor's row set; in practice the non-synthetic Map case doesn't
    # surface (orchestrator always synthesises sub-solves), but if it
    # ever did we'd return the union of inner periods deterministically.
    if "i" in sub.columns and not is_map:
        sub = sub.sort("i")
    if is_map:
        return sub[period_col].cast(pl.Utf8, strict=False).unique(maintain_order=True).to_list()
    return sub["value"].to_list()


def _entityInvest_set(source: "InputSource") -> pl.LazyFrame:
    """Entities (units + nodes + connections) whose
    ``invest_method`` ∉ ``_INVEST_NOT_ALLOWED``.  Returns a lazy frame
    with column ``e``.  Mirrors flextool's ``entityInvest`` set.
    """
    return _entity_invest_filter(source, _INVEST_METHODS_INVEST_ALLOWED)


def _entityDivest_set(source: "InputSource") -> pl.LazyFrame:
    """Entities whose ``invest_method`` ∉ ``_DIVEST_NOT_ALLOWED``.
    Mirrors flextool's ``entityDivest`` set.
    """
    return _entity_invest_filter(source, _INVEST_METHODS_DIVEST_ALLOWED)


# Mirror flextool/flextool_base.dat:211-212 (cf. entity_annual_calc_params.py).
_INVEST_NOT_ALLOWED: frozenset[str] = frozenset((
    "not_allowed", "retire_period", "retire_total", "retire_no_limit",
))
_DIVEST_NOT_ALLOWED: frozenset[str] = frozenset((
    "not_allowed", "invest_period", "invest_total", "invest_no_limit",
))
_INVEST_METHODS_ALL: frozenset[str] = frozenset((
    "not_allowed", "invest_no_limit", "invest_period", "invest_total",
    "invest_period_total",
    "retire_no_limit", "retire_period", "retire_total", "retire_period_total",
    "invest_retire_no_limit", "invest_retire_period",
    "invest_retire_total", "invest_retire_period_total",
    "cumulative_limits",
))
_INVEST_METHODS_INVEST_ALLOWED: frozenset[str] = (
    _INVEST_METHODS_ALL - _INVEST_NOT_ALLOWED
)
_INVEST_METHODS_DIVEST_ALLOWED: frozenset[str] = (
    _INVEST_METHODS_ALL - _DIVEST_NOT_ALLOWED
)

# Mirror ``invest_total_sets.py:25-32``.
_INVEST_METHODS_INVEST_TOTAL: frozenset[str] = frozenset((
    "invest_total", "invest_period_total",
    "invest_retire_total", "invest_retire_period_total",
))
_INVEST_METHODS_DIVEST_TOTAL: frozenset[str] = frozenset((
    "retire_total", "retire_period_total",
    "invest_retire_total", "invest_retire_period_total",
))
# Mirror ``invest_divest_sets.py:33-40``.
_INVEST_METHODS_INVEST_PERIOD: frozenset[str] = frozenset((
    "invest_period", "invest_period_total",
    "invest_retire_period", "invest_retire_period_total",
))
_INVEST_METHODS_DIVEST_PERIOD: frozenset[str] = frozenset((
    "retire_period", "retire_period_total",
    "invest_retire_period", "invest_retire_period_total",
))


def _entity_invest_method(source: "InputSource") -> pl.LazyFrame | None:
    """Pull the per-entity invest_method as a (e, method) lazy frame.

    Spine source: ``unit.invest_method`` / ``node.invest_method`` /
    ``connection.invest_method``.  Returns None when no entity has an
    explicit method (default is ``not_allowed``).
    """
    parts: list[pl.LazyFrame] = []
    for ec in ("unit", "node", "connection"):
        df = _try_param(source, ec, "invest_method")
        if df is None:
            continue
        parts.append(df.lazy().select(
            pl.col("name").alias("e"),
            pl.col("value").alias("method"),
        ))
    if not parts:
        return None
    return pl.concat(parts).unique()


def _entity_invest_filter(source: "InputSource",
                           allowed: frozenset[str]) -> pl.LazyFrame:
    """Lazy frame ``[e]`` of entities whose ``invest_method ∈ allowed``.
    """
    eim = _entity_invest_method(source)
    if eim is None:
        return pl.DataFrame(schema={"e": pl.Utf8}).lazy()
    return (eim.filter(pl.col("method").is_in(list(allowed)))
                .select("e").unique())


def _has_capacity_constraint_invest_set(source: "InputSource"
                                          ) -> pl.LazyFrame:
    """Set of entities with any ``constraint_invested_capacity_coefficient``
    or ``constraint_pre_built_capacity_coefficient`` row.  Mirrors
    flextool's ``_has_capacity_constraint_invest`` predicate
    (``invest_divest_sets.py:172-174``).  Returns a lazy frame ``[e]``.
    """
    parts: list[pl.LazyFrame] = []
    for ec in ("unit", "node", "connection"):
        for pname in ("constraint_invested_capacity_coefficient",
                       "constraint_pre_built_capacity_coefficient"):
            df = _try_param(source, ec, pname)
            if df is None:
                continue
            # Spine reader returns the entity name column as ``name``.
            if "name" in df.columns:
                parts.append(df.lazy().select(
                    pl.col("name").alias("e")).unique())
    if not parts:
        return pl.DataFrame(schema={"e": pl.Utf8}).lazy()
    return pl.concat(parts).unique()


def _eea_pairs(source: "InputSource",
                entityInvest_lf: pl.LazyFrame,
                period_invest: list[str]) -> pl.LazyFrame:
    """Compute non-zero ``ed_entity_annual`` pairs (e, d).

    Algorithm (audit §3.7.5/6, ``entity_annual_calc_params.py:177-198``):
      annuity[e, d] = #methods_for_e * invest_value[e, d] * 1000 * r
                       / (1 - 1/(1+r)^n)
    where r = max(0.05, discount_rate); n = max(20, lifetime).

    For ``ed_invest`` we only need the *non-zero* predicate, so we can
    short-circuit: a pair (e, d) has non-zero annuity iff
    ``invest_cost[e] != 0`` AND ``methods_for_e ∩ INVEST_ALLOWED`` is
    non-empty.  Same for ``ed_divest`` with ``salvage_value`` and the
    divest-allowed method intersection.  This is an exact mirror of
    flextool's predicate at ``invest_divest_sets.py:177-180``.
    """
    return _eea_predicate(source, entityInvest_lf, period_invest,
                            cost_param="invest_cost",
                            allowed=_INVEST_METHODS_INVEST_ALLOWED)


def _eead_pairs(source: "InputSource",
                  entityDivest_lf: pl.LazyFrame,
                  period_invest: list[str]) -> pl.LazyFrame:
    return _eea_predicate(source, entityDivest_lf, period_invest,
                            cost_param="salvage_value",
                            allowed=_INVEST_METHODS_DIVEST_ALLOWED)


def _eea_predicate(source: "InputSource",
                     entity_lf: pl.LazyFrame,
                     period_invest: list[str],
                     cost_param: str,
                     allowed: frozenset[str]) -> pl.LazyFrame:
    """Predicate frame ``[e, d]`` where the per-method annuity sum is
    structurally non-zero, given:

      * Some method m ∈ allowed exists for e (``entity__invest_method``).
      * ``cost_param`` (per-(e[, d])) is non-zero.

    Used as the ``eea != 0`` short-circuit in ``ed_invest``/``ed_divest``
    composition.
    """
    if not period_invest:
        return pl.DataFrame(schema={"e": pl.Utf8, "d": pl.Utf8}).lazy()
    eim = _entity_invest_method(source)
    if eim is None:
        return pl.DataFrame(schema={"e": pl.Utf8, "d": pl.Utf8}).lazy()
    has_method = (eim.filter(pl.col("method").is_in(list(allowed)))
                       .select("e").unique())
    cost_lf = _per_entity_period_cost(source, cost_param, period_invest)
    if cost_lf is None:
        return pl.DataFrame(schema={"e": pl.Utf8, "d": pl.Utf8}).lazy()
    return (has_method.join(entity_lf, on="e", how="inner")
                       .join(cost_lf, on="e", how="inner")
                       .filter(pl.col("v") != 0.0)
                       .select("e", "d").unique())


def _per_entity_period_cost(source: "InputSource",
                              parameter_name: str,
                              period_invest: list[str]) -> pl.LazyFrame | None:
    """Read ``unit/node/connection.<parameter_name>`` and broadcast to
    (e, d, v) over the supplied period list.  Handles scalar, 1d_map(period)
    or 2d_map shapes — returns one row per (e, d) at the end.

    For the predicate use-case we collapse across periods by taking the
    per-entity max over d (any non-zero in any period qualifies), but to
    keep parity with flextool's per-(e, d) test we keep per-(e, d).
    """
    parts: list[pl.LazyFrame] = []
    for ec in ("unit", "node", "connection"):
        df = _try_param(source, ec, parameter_name)
        if df is None:
            continue
        cols = df.columns
        # Detect shape: (name, value), (name, period, value), or
        # (name, ..., value).
        if "period" in cols:
            parts.append(df.lazy().select(
                pl.col("name").alias("e"),
                pl.col("period").alias("d"),
                pl.col("value").cast(pl.Float64).alias("v"),
            ))
        else:
            # Broadcast scalar across period_invest.
            pi_lf = pl.LazyFrame({"d": period_invest})
            parts.append(df.lazy().select(
                pl.col("name").alias("e"),
                pl.col("value").cast(pl.Float64).alias("v"),
            ).join(pi_lf, how="cross"))
    if not parts:
        return None
    return pl.concat(parts).unique()


# ---------------------------------------------------------------------------
# §3.7.1 — ed_invest_set / ed_divest_set
# ---------------------------------------------------------------------------


def _read_period_first(source: "InputSource",
                          active_solve: str | None,
                          workdir: Path | None) -> list[str]:
    """Return ``period_first`` for the active solve (Γ.6.D).

    Mirrors flextool's ``solve_writers.write_periods`` (line 216-226):
    the first period of the first timeset on the active solve, plus any
    stochastic-branch siblings.

    Resolution order:
      1. ``solve_data/period_first.csv`` if present (canonical).
      2. First entry of ``solve.realized_periods`` (non-stochastic
         fallback for tests that don't have the CSV).

    Returns ``[]`` when no usable source.
    """
    if workdir is not None:
        p = Path(workdir) / "solve_data" / "period_first.csv"
        if p.exists():
            try:
                df = _read_csv_file(p)
                if df.height > 0 and "period" in df.columns:
                    return df["period"].cast(pl.Utf8, strict=False).to_list()
            except Exception:
                pass
    if active_solve is None:
        return []
    realized = _solve_periods(source, active_solve, "realized_periods") or []
    invest = _solve_periods(source, active_solve, "invest_periods") or []
    seq = realized or invest
    return [seq[0]] if seq else []


def _lifetime_gate_sums(source: "InputSource",
                            active_solve: str | None,
                            workdir: Path | None,
                            ) -> dict[str, float]:
    """Per-entity ``life_sum`` for the ``reinvest_choice`` /
    ``no_investment`` lifetime gate (Γ.6.D).

    Algorithm (mirror
    ``preprocessing/entity_period_calc_params.py:315-319`` &
    ``invest_divest_sets.py:343-347``)::

        life_sum(e) = sum_{d_first in period_first}
                       (p_years_d[d_first] + edEntity_lifetime[e, d_first])

    Only entities whose ``lifetime_method`` is ``reinvest_choice`` or
    ``no_investment`` are returned; other methods don't trigger the
    expiry gate so callers can safely treat ``e ∉ map`` as "no gate".
    """
    period_first = _read_period_first(source, active_solve, workdir)
    if not period_first:
        return {}
    # p_years_d for those periods.
    pyd_lf = _p_years_d_lf(source, active_solve, workdir)
    pyd: dict[str, float] = {}
    if pyd_lf is not None:
        pyd_df = pyd_lf.collect()
        for r in pyd_df.iter_rows(named=True):
            pyd[str(r["d"])] = float(r["yr"])
    # edEntity_lifetime: per-(e, d) lifetime, broadcast scalar over periods.
    per_p = _per_entity_period_value(source, "lifetime") or {}
    scalar = _per_entity_scalar(source, "lifetime")
    process_set, node_set, _ = _entity_classes_lookup(source)
    # Lifetime-method classification.
    all_e = _all_entities(source)
    elm = _entity_lifetime_methods(source, all_e)

    out: dict[str, float] = {}
    for e, methods in elm.items():
        if not (("reinvest_choice" in methods)
                or ("no_investment" in methods)):
            continue
        if e not in process_set and e not in node_set:
            # Connection without is_DC / non-process — no lifetime.
            out[e] = 0.0
            continue
        s = 0.0
        for d_first in period_first:
            life = _resolve_pdX(per_p, scalar, e, d_first)
            s += pyd.get(d_first, 0.0) + life
        out[e] = s
    return out


def _lifetime_expired_pairs(source: "InputSource",
                                active_solve: str | None,
                                workdir: Path | None,
                                methods: tuple[str, ...] = (
                                    "reinvest_choice", "no_investment"),
                                ) -> set[tuple[str, str]]:
    """Set of (e, d) pairs whose pre-existing capacity has expired
    (Γ.6.D).

    For each entity whose ``lifetime_method`` is in ``methods``, return
    the periods d in ``period_in_use`` where ``p_years_d[d] >= life_sum(e)``.
    These are the rows where:

      * ``p_entity_pre_existing[e, d]`` should evaluate to 0.
      * ``ed_invest_forbidden_no_investment`` adds (e, d) to the
        forbidden set (only ``no_investment`` is forbidden — but the
        gate is shared with ``reinvest_choice`` for the existing-cap
        cascade).

    Only ``no_investment`` entries actually pin ``v_invest = 0`` —
    callers filtering ``ed_invest_set`` should pass ``methods=("no_investment",)``.
    """
    sums = _lifetime_gate_sums(source, active_solve, workdir)
    if not sums:
        return set()
    # Restrict to entities whose method is in the requested filter.
    all_e = _all_entities(source)
    elm = _entity_lifetime_methods(source, all_e)
    keep_e = {e for e, ms in elm.items() if any(m in ms for m in methods)}
    sums = {e: v for e, v in sums.items() if e in keep_e}
    if not sums:
        return set()

    pyd_lf = _p_years_d_lf(source, active_solve, workdir)
    if pyd_lf is None:
        return set()
    pyd_df = pyd_lf.collect()
    pyd: dict[str, float] = {str(r["d"]): float(r["yr"])
                                for r in pyd_df.iter_rows(named=True)}
    piu = _period_in_use_set(source, active_solve, workdir)
    if not piu:
        return set()
    out: set[tuple[str, str]] = set()
    for e, s in sums.items():
        for d in piu:
            if pyd.get(d, 0.0) >= s:
                out.add((e, d))
    return out


def ed_invest_forbidden_no_investment_from_source(
        source: "InputSource",
        active_solve: str | None,
        workdir: Path | None,
        ed_invest: pl.DataFrame | None = None,  # noqa: ARG001 — kept for back-compat
        ) -> pl.DataFrame | None:
    """Audit §3.7 — ``ed_invest`` rows whose ``no_investment`` lifetime
    window has already ended.

    Mirrors :func:`flextool/preprocessing/invest_divest_sets.py
    :write_ed_invest_forbidden_no_investment` — used in
    ``fix_v_invest_no_investment_eq`` (mod L3930) to pin ``v_invest = 0``
    after the lifetime expires.

    Computed against the *unfiltered* ed_invest cross-product (so the
    output matches flextool's CSV regardless of whether
    :func:`ed_invest_set_from_source` has already anti-joined the
    forbidden tuples).  We rebuild the universe locally:
    ``entityInvest × period_invest`` then keep only the (e, d) pairs
    where:

      * ``e`` has ``lifetime_method == 'no_investment'``,
      * ``yr[d] >= life_sum(e)``.
    """
    period_invest = _solve_periods(source, active_solve, "invest_periods")
    if not period_invest:
        return pl.DataFrame(schema={"e": pl.Utf8, "d": pl.Utf8})
    forbidden = _lifetime_expired_pairs(
        source, active_solve, workdir, methods=("no_investment",))
    if not forbidden:
        return pl.DataFrame(schema={"e": pl.Utf8, "d": pl.Utf8})
    # Universe = entityInvest × period_invest (canonical's source set,
    # before the forbidden filter is applied).
    ei_lf = _entityInvest_set(source)
    ei_df = ei_lf.collect() if ei_lf is not None else pl.DataFrame(
        schema={"e": pl.Utf8})
    if ei_df.height == 0:
        return pl.DataFrame(schema={"e": pl.Utf8, "d": pl.Utf8})
    ei = set(ei_df["e"].to_list())
    rows = [(e, d) for (e, d) in forbidden
              if e in ei and d in period_invest]
    if not rows:
        return pl.DataFrame(schema={"e": pl.Utf8, "d": pl.Utf8})
    return pl.DataFrame(rows, schema=["e", "d"], orient="row").sort("e", "d")


def ed_invest_set_from_source(source: "InputSource",
                                active_solve: str | None,
                                workdir: Path | None = None,
                                ) -> pl.DataFrame | None:
    """Compute the (entity, period) pairs where invest is allowed.

    Algorithm (audit §3.7.1, ``invest_divest_sets.py:171-183``):

        ed_invest = { (e, d) :
                       e ∈ entityInvest, d ∈ period_invest_of_solve,
                       (eea[e, d] != 0  OR  e has capacity constraint) }

    For the simple case (no per-entity-period invest_cost Maps, no
    capacity constraints), this reduces to the cross-product of
    entityInvest × period_invest, intersected with the eea-non-zero set.

    Γ.6.D: ``ed_invest_forbidden_no_investment`` rows (entities with
    ``lifetime_method = no_investment`` whose lifetime window has
    expired) are anti-joined out.  Mirrors
    ``invest_divest_sets.py:write_ed_invest_forbidden_no_investment``
    consumed by ``fix_v_invest_no_investment_eq`` (mod L3930).
    """
    period_invest = _solve_periods(source, active_solve, "invest_periods")
    if not period_invest:
        # Empty invest_periods → empty ed_invest (the dispatch-only solve case).
        return pl.DataFrame(schema={"e": pl.Utf8, "d": pl.Utf8})

    ei_lf = _entityInvest_set(source)
    has_cc = _has_capacity_constraint_invest_set(source)
    eea_pred = _eea_pairs(source, ei_lf, period_invest)

    # Cross ei × period_invest, then keep where (eea != 0) OR (e in has_cc).
    pi_lf = pl.LazyFrame({"d": period_invest})
    cross = ei_lf.join(pi_lf, how="cross")
    has_cc_marked = (cross.join(has_cc, on="e", how="inner")
                            .select("e", "d"))
    out = (pl.concat([eea_pred.select("e", "d"),
                       has_cc_marked])
              .unique()
              .sort("e", "d")
              .collect())
    # Γ.6.D: drop pairs forbidden by the no_investment lifetime gate.
    forbidden = _lifetime_expired_pairs(
        source, active_solve, workdir, methods=("no_investment",))
    if forbidden and out.height > 0:
        keep = [(e, d) for (e, d) in out.iter_rows()
                  if (e, d) not in forbidden]
        if not keep:
            out = pl.DataFrame(schema={"e": pl.Utf8, "d": pl.Utf8})
        else:
            out = pl.DataFrame(keep, schema=["e", "d"], orient="row")
    return out


def ed_divest_set_from_source(source: "InputSource",
                                active_solve: str | None,
                                ) -> pl.DataFrame | None:
    """Mirror of ``ed_invest_set`` for divest.  Algorithm
    (``invest_divest_sets.py:185-193``):

        ed_divest = { (e, d) :
                       e ∈ entityDivest, d ∈ period_invest_of_solve,
                       (eead[e, d] != 0  OR  e has capacity constraint) }
    """
    period_invest = _solve_periods(source, active_solve, "invest_periods")
    if not period_invest:
        return pl.DataFrame(schema={"e": pl.Utf8, "d": pl.Utf8})
    ed_lf = _entityDivest_set(source)
    has_cc = _has_capacity_constraint_invest_set(source)
    eead_pred = _eead_pairs(source, ed_lf, period_invest)

    pi_lf = pl.LazyFrame({"d": period_invest})
    cross = ed_lf.join(pi_lf, how="cross")
    has_cc_marked = (cross.join(has_cc, on="e", how="inner")
                            .select("e", "d"))
    out = (pl.concat([eead_pred.select("e", "d"),
                       has_cc_marked])
              .unique()
              .sort("e", "d")
              .collect())
    return out


# ---------------------------------------------------------------------------
# §3.7.2 / §3.7.3 — edd_invest_set / edd_invest_lookback / edd_divest_active
# ---------------------------------------------------------------------------


def _p_years_d_lf(source: "InputSource",
                    active_solve: str,
                    workdir: Path | None = None,
                    ) -> pl.LazyFrame | None:
    """Build per-period ``p_years_d`` for the active solve.

    Resolution order (Γ.6.D):
      1. ``solve_data/p_years_d.csv`` from workdir, if present — the
         canonical post-preprocessing CSV that already encodes the
         cumulative year offset.
      2. ``solve_data/period_with_history.csv`` (column ``param`` holds
         the cumulative year value).
      3. ``solve.years_from_start`` 1d_map(period) on the active solve
         (rare; not in current SpineDB schema but kept as a safety net).
      4. ``solve.years_represented`` — cumulative-sum the per-period
         year-count to derive the offset (matches
         ``_p_discount_years_from_source``).
      5. Integer-indexed fallback ``[0, 1, 2, ...]`` over period_in_use.

    Returns lazy frame ``[d, yr]`` or None when nothing usable exists.
    """
    # 1 — workdir CSV (authoritative).
    if workdir is not None:
        p = Path(workdir) / "solve_data" / "p_years_d.csv"
        if p.exists():
            try:
                df = _read_csv_file(p)
                if df.height > 0 and "period" in df.columns:
                    val_col = ("value" if "value" in df.columns
                                else df.columns[-1])
                    return df.lazy().select(
                        pl.col("period").alias("d"),
                        pl.col(val_col).cast(pl.Float64, strict=False)
                                          .alias("yr"),
                    )
            except Exception:
                pass
        p2 = Path(workdir) / "solve_data" / "period_with_history.csv"
        if p2.exists():
            try:
                df = _read_csv_file(p2)
                if df.height > 0 and "period" in df.columns:
                    val_col = ("param" if "param" in df.columns
                                else df.columns[-1])
                    return df.lazy().select(
                        pl.col("period").alias("d"),
                        pl.col(val_col).cast(pl.Float64, strict=False)
                                          .alias("yr"),
                    )
            except Exception:
                pass
    # 2 — explicit years_from_start (rare; not present in current schema).
    yfs = _try_param(source, "solve", "years_from_start")
    if yfs is not None and "period" in yfs.columns:
        sub = yfs.filter(pl.col("name") == active_solve)
        if sub.height > 0:
            return sub.lazy().select(
                pl.col("period").alias("d"),
                pl.col("value").cast(pl.Float64).alias("yr"),
            )
    # 3 — cumulative-sum from years_represented.
    yrp = _try_param(source, "solve", "years_represented")
    if (yrp is not None and "period" in yrp.columns
            and active_solve is not None):
        sub = yrp.filter(pl.col("name") == active_solve)
        if sub.height > 0:
            # Map iteration order — the SpineDbReader returns rows in DB
            # storage order which mirrors the canonical Map index order.
            cum = 0.0
            rows: list[tuple[str, float]] = []
            for r in sub.iter_rows(named=True):
                d = str(r["period"])
                rows.append((d, cum))
                try:
                    cum += float(r["value"])
                except Exception:
                    pass
            if rows:
                return pl.LazyFrame({
                    "d": [r[0] for r in rows],
                    "yr": [r[1] for r in rows],
                }).lazy()
    # 4 — fallback to integer-indexed period_in_use.
    periods = _period_in_use_set(source, active_solve)
    if periods:
        return pl.LazyFrame({
            "d": periods,
            "yr": [float(i) for i in range(len(periods))],
        }).lazy()
    return None


def _edEntity_lifetime_lf(source: "InputSource",
                            ed_invest_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Build ``edEntity_lifetime[(e, d)]`` from ``unit/node/connection.lifetime``.

    Default lifetime = 0 (no lifetime); we keep the value as-is.  When
    lifetime is a scalar, broadcast across all (e, d) in ed_invest.
    """
    parts: list[pl.LazyFrame] = []
    for ec in ("unit", "node", "connection"):
        df = _try_param(source, ec, "lifetime")
        if df is None:
            continue
        cols = df.columns
        if "period" in cols:
            parts.append(df.lazy().select(
                pl.col("name").alias("e"),
                pl.col("period").alias("d"),
                pl.col("value").cast(pl.Float64).alias("life"),
            ))
        else:
            parts.append(df.lazy().select(
                pl.col("name").alias("e"),
                pl.col("value").cast(pl.Float64).alias("life"),
            ).join(ed_invest_lf.select("d").unique(), how="cross"))
    if not parts:
        return ed_invest_lf.with_columns(life=pl.lit(0.0))
    return (ed_invest_lf
              .join(pl.concat(parts).unique(), on=["e", "d"], how="left")
              .with_columns(life=pl.col("life").fill_null(0.0)))


def _period_in_use_set(source: "InputSource",
                          active_solve: str | None,
                          workdir: Path | None = None,
                          *,
                          ctx: "SolveContext | None" = None) -> list[str]:
    """Compute the canonical ``period_in_use`` set for the active solve.

    Mirrors flextool's ``preprocessing/per_solve_sets.py:95-101``:
    distinct periods in ``solve_data/steps_in_use.csv`` (a.k.a. the
    realised ``dt`` set).  When ``ctx`` is supplied, the typed
    :pyattr:`SolveContext.period_in_use` frame is consulted first — same
    semantics as the workdir CSV path but served from cache.  When
    ``workdir`` is provided AND the file is present (no ctx), the
    authoritative CSV is read directly.

    Falls back to ``solve.realized_periods ∪ solve.invest_periods`` when
    no in-memory / workdir source is available — sufficient for non-
    stochastic fixtures.
    """
    if ctx is not None:
        piu = ctx.period_in_use
        if piu.height > 0:
            return piu["d"].cast(pl.Utf8, strict=False).to_list()
    if workdir is not None:
        p = Path(workdir) / "solve_data" / "period_in_use_set.csv"
        if p.exists():
            df = _read_csv_file(p)
            if df.height > 0 and "period" in df.columns:
                return df["period"].cast(pl.Utf8, strict=False).to_list()
    if active_solve is None:
        return []
    realized = _solve_periods(source, active_solve, "realized_periods") or []
    invest = _solve_periods(source, active_solve, "invest_periods") or []
    seen: dict[str, None] = {}
    for d in realized + invest:
        seen.setdefault(d, None)
    return list(seen.keys())


def edd_invest_lookback_set_from_source(source: "InputSource",
                                            active_solve: str | None,
                                            ed_invest: pl.DataFrame | None,
                                            workdir: Path | None = None,
                                            ) -> pl.DataFrame | None:
    """Build the strict-lookback (e, d_invest, d) tuples used by the
    user-constraint LHS prebuilt-capacity term (mod L2885-2898).

    Algorithm (audit §3.7.3, input.py:1248-1256 + canonical
    ``invest_divest_sets.py:227-270`` lifetime filter):

      lookback = { (e, d_invest, d) :
                    (e, d_invest) ∈ ed_invest,
                    d in dt periods,
                    yr[d_invest] < yr[d],
                    yr[d] < yr[d_invest] + lifetime[e, d_invest]   (Γ.6.D)
                                            (only when entity's
                                             lifetime_method limits the
                                             window — reinvest_choice /
                                             no_investment) }

    Δ.7 consolidation: delegates to the lazy port
    :func:`._derived_existing.edd_invest_lookback_set_lf` which
    consumes :func:`._derived_walks.period_walk_iterator` with the
    new ``STRICT_LOOKBACK_*`` modes.  The previous eager
    ``for r in out.iter_rows`` lifetime gate is replaced with a fully
    lazy join + filter on the shared walker.
    """
    if ed_invest is None or ed_invest.height == 0:
        return pl.DataFrame(schema={
            "e": pl.Utf8, "d_invest": pl.Utf8, "d": pl.Utf8,
        })
    pyd_lf = _p_years_d_lf(source, active_solve, workdir)
    if pyd_lf is None:
        return None
    periods = _period_in_use_set(source, active_solve, workdir)
    if not periods:
        return pl.DataFrame(schema={
            "e": pl.Utf8, "d_invest": pl.Utf8, "d": pl.Utf8,
        })
    # Δ.7 — delegate to the lazy port consuming the shared walker.
    from ._derived_existing import edd_invest_lookback_set_lf
    ed_invest_lf = ed_invest.lazy().select("e", "d")
    return edd_invest_lookback_set_lf(
        source, active_solve, ed_invest_lf, periods, workdir).collect()


def edd_divest_active_from_source(source: "InputSource",
                                      active_solve: str | None,
                                      pd_divest: pl.DataFrame | None,
                                      ) -> pl.DataFrame | None:
    """Build the active-divest (p, d_divest, d) tuples — see audit §3.7.3.

    pd_divest ⊆ ed_divest with ``p ∈ process``; the active set further
    filters d_divest ≤ d using year ordering.
    """
    if pd_divest is None or pd_divest.height == 0:
        return pl.DataFrame(schema={
            "p": pl.Utf8, "d_divest": pl.Utf8, "d": pl.Utf8,
        })
    pyd_lf = _p_years_d_lf(source, active_solve)
    if pyd_lf is None:
        return None
    periods = _period_in_use_set(source, active_solve)
    if not periods:
        return pl.DataFrame(schema={
            "p": pl.Utf8, "d_divest": pl.Utf8, "d": pl.Utf8,
        })
    period_lf = pl.LazyFrame({"d": periods})
    pdd_lf = pd_divest.lazy().rename({"d": "d_divest"})
    yr_div = pyd_lf.rename({"d": "d_divest", "yr": "yr_divest"})
    yr_d = pyd_lf.rename({"yr": "yr"})
    out = (pdd_lf
              .join(period_lf, how="cross")
              .join(yr_div, on="d_divest", how="inner")
              .join(yr_d, on="d", how="inner")
              .filter(pl.col("yr_divest") <= pl.col("yr"))
              .select("p", "d_divest", "d")
              .sort("p", "d_divest", "d")
              .collect())
    return out


# ---------------------------------------------------------------------------
# §3.7.4 — p_entity_max_units
# ---------------------------------------------------------------------------


def _entity_unitsize_lf(source: "InputSource") -> pl.LazyFrame:
    """Per-entity unitsize cascade across unit + node + connection,
    mirroring ``entity_period_calc_params.py:159-202``::

        unitsize = virtual_unitsize (if explicitly set non-zero)
                  OR existing       (if explicitly set non-zero)
                  OR 1000.0

    Returns a lazy frame ``[e, us]`` covering every entity in
    ``unit ∪ node ∪ connection``.  Entities with neither
    ``virtual_unitsize`` nor ``existing`` get the canonical fallback
    ``1000.0``.

    For ``existing`` 1d_map(period) values, the per-period MAX is used
    as the cascade input (matches flextool's flat ``p_unit/p_node`` Map
    -agnostic read; the cascade only checks "non-zero", and the per-
    period MAX is non-zero iff any period is).
    """
    base_parts: list[pl.LazyFrame] = []
    us_parts: list[pl.LazyFrame] = []
    ex_parts: list[pl.LazyFrame] = []
    for ec in ("unit", "node", "connection"):
        ents = _try_entities(source, ec)
        if ents is None or ents.height == 0:
            continue
        base_parts.append(ents.lazy().select(
            pl.col("name").alias("e")))
        us = _try_param_explicit(source, ec, "virtual_unitsize")
        if us is not None and us.height > 0:
            us_parts.append(us.lazy().select(
                pl.col("name").alias("e"),
                pl.col("value").cast(pl.Float64).alias("vu"),
            ))
        ex = _try_param_explicit(source, ec, "existing")
        if ex is not None and ex.height > 0:
            if "period" in ex.columns:
                ex_parts.append(ex.lazy()
                    .select(pl.col("name").alias("e"),
                              pl.col("value").cast(pl.Float64).alias("v"))
                    .group_by("e")
                    .agg(pl.col("v").max().alias("ex")))
            else:
                ex_parts.append(ex.lazy().select(
                    pl.col("name").alias("e"),
                    pl.col("value").cast(pl.Float64).alias("ex"),
                ))
    if not base_parts:
        return pl.LazyFrame(schema={"e": pl.Utf8, "us": pl.Float64})
    base = pl.concat(base_parts).unique()
    if us_parts:
        vu_lf = pl.concat(us_parts).unique(subset=["e"], keep="last")
        base = base.join(vu_lf, on="e", how="left")
    else:
        base = base.with_columns(vu=pl.lit(None, dtype=pl.Float64))
    if ex_parts:
        ex_lf = pl.concat(ex_parts).unique(subset=["e"], keep="last")
        base = base.join(ex_lf, on="e", how="left")
    else:
        base = base.with_columns(ex=pl.lit(None, dtype=pl.Float64))
    return base.with_columns(
        us=pl.when(pl.col("vu").fill_null(0.0) != 0.0)
              .then(pl.col("vu"))
              .when(pl.col("ex").fill_null(0.0) != 0.0)
              .then(pl.col("ex"))
              .otherwise(pl.lit(1000.0))
    ).select("e", "us")


def _entity_class_membership(source: "InputSource") -> dict[str, set[str]]:
    """Return ``{"unit": {...}, "node": {...}, "connection": {...}}``."""
    out: dict[str, set[str]] = {}
    for ec in ("unit", "node", "connection"):
        ents = _try_entities(source, ec)
        out[ec] = set() if ents is None else set(ents["name"].to_list())
    return out


def _entity_method_pairs(source: "InputSource") -> set[tuple[str, str]]:
    """Mirror flextool's ``entity__invest_method`` 2-col CSV — return
    ``{(e, method)}``.  Uses ``parameter`` (not _explicit) since
    ``invest_method`` has a non-broadcasting "not_allowed" default that
    SpineDbReader doesn't synthesise.
    """
    out: set[tuple[str, str]] = set()
    for ec in ("unit", "node", "connection"):
        df = _try_param(source, ec, "invest_method")
        if df is None:
            continue
        for e, m in df.select("name", "value").iter_rows():
            out.add((str(e), str(m)))
    return out


def _e_explicit_param(source: "InputSource", parameter_name: str
                       ) -> pl.LazyFrame:
    """Union explicit per-entity scalar parameter across unit + node +
    connection — returns ``[e, value]`` lazy frame (empty if no rows).
    """
    parts: list[pl.LazyFrame] = []
    for ec in ("unit", "node", "connection"):
        df = _try_param_explicit(source, ec, parameter_name)
        if df is None or df.height == 0:
            continue
        parts.append(df.lazy().select(
            pl.col("name").alias("e"),
            pl.col("value").cast(pl.Float64),
        ))
    if not parts:
        return pl.LazyFrame(schema={"e": pl.Utf8, "value": pl.Float64})
    return pl.concat(parts).unique(subset=["e"], keep="last")


def _ed_explicit_period_param(source: "InputSource", parameter_name: str
                               ) -> pl.LazyFrame:
    """Per-(e, d) 1d_map(period) explicit values across unit + node +
    connection.  Returns ``[e, d, value]`` lazy frame.

    Handles both canonical (``period`` named index) and fallback (``x``
    generic index) Map shapes — SpineDbReader uses ``x`` when
    Map.index_name is empty / generic and the period interpretation is
    inferred from value content.
    """
    parts: list[pl.LazyFrame] = []
    for ec in ("unit", "node", "connection"):
        df = _try_param_explicit(source, ec, parameter_name)
        if df is None or df.height == 0:
            continue
        period_col: str | None = None
        for cand in ("period", "x"):
            if cand in df.columns:
                period_col = cand
                break
        if period_col is None:
            continue
        parts.append(df.lazy().select(
            pl.col("name").alias("e"),
            pl.col(period_col).alias("d"),
            pl.col("value").cast(pl.Float64),
        ))
    if not parts:
        return pl.LazyFrame(schema={"e": pl.Utf8, "d": pl.Utf8,
                                       "value": pl.Float64})
    return pl.concat(parts).unique(subset=["e", "d"], keep="last")


def p_entity_max_units_from_source(source: "InputSource",
                                       ed_invest: pl.DataFrame | None,
                                       active_solve: str | None = None,
                                       workdir: Path | None = None,
                                       p_entity_all_existing: "Param | None" = None,
                                       ) -> "Param | None":
    """Per-(e, d) maximum new-unit count = ``max_capacity / unitsize``.

    Mirrors flextool's ``entity_period_calc_params.py:1718-1761``:
      ``p_entity_max_capacity[e, d]`` is the per-(entity, period)
      capacity ceiling factoring in existing capacity, per-period and
      total invest caps, and the ``invest_no_limit`` blanket cap.
      ``p_entity_max_units[e, d] = max_capacity / unitsize`` (or 0 when
      unitsize is 0).  Caller filters value > 0.

    The returned Param mirrors the CSV-loader: only rows where
    ``value > 0`` after the divide.

    The ``p_entity_all_existing`` argument lets the caller pass the
    CSV-loaded lifetime cascade frame (which factors in
    ``previously_invested`` and lifetime-cumulative chains for
    multi-period invest fixtures).  When ``None`` we fall back to
    :func:`p_entity_all_existing_from_source` — sufficient for the
    no-prior-handoff single-year case.
    """
    # Gate: both ed_invest AND ed_divest must be empty for the
    # corresponding CSV ``_load_invest`` to blank-out.  We get
    # ``ed_invest`` here; the caller checks ed_divest analogously.
    if ed_invest is None or ed_invest.height == 0:
        return None

    piu = _period_in_use_set(source, active_solve, workdir)
    if not piu:
        return None
    period_set = set(piu)

    # Entities universe ∪ classes.
    cls_members = _entity_class_membership(source)
    entities = sorted(cls_members["unit"] | cls_members["node"]
                      | cls_members["connection"])
    if not entities:
        return None

    # Per-(e, method) pairs.
    method_pairs = _entity_method_pairs(source)
    has_no_limit_e = {e for (e, m) in method_pairs if m == "invest_no_limit"}
    methods_by_e: dict[str, set[str]] = {}
    for (e, m) in method_pairs:
        methods_by_e.setdefault(e, set()).add(m)

    # entityInvest = entities with any allowed-invest method.
    entityInvest = {e for e, ms in methods_by_e.items()
                    if ms - _INVEST_NOT_ALLOWED}
    # e_invest_total = subset filtered by INVEST_TOTAL methods.
    e_invest_total = {e for e in entityInvest
                       if methods_by_e.get(e, set()) & _INVEST_METHODS_INVEST_TOTAL}

    # Unitsize cascade per entity.
    us_lf = _entity_unitsize_lf(source)
    us_df = us_lf.collect()
    us_map: dict[str, float] = dict(us_df.rows())

    # Existing capacity per (e, d).  Prefer the supplied lifetime cascade
    # frame (CSV-loaded ``p_entity_all_existing``), fall back to the
    # raw-existing helper.
    if p_entity_all_existing is not None:
        existing_map: dict[tuple[str, str], float] = {
            (e, d): v
            for e, d, v in p_entity_all_existing.frame.iter_rows()
        }
    else:
        pae_param = p_entity_all_existing_from_source(source, active_solve,
                                                         workdir=workdir)
        if pae_param is None:
            existing_map = {}
        else:
            existing_map = {(e, d): v
                            for e, d, v in pae_param.frame.iter_rows()}

    # invest_max_period[e, d]: explicit per-period values, default 0.
    imp_df = _ed_explicit_period_param(source, "invest_max_period").collect()
    imp_map: dict[tuple[str, str], float] = {(e, d): v
        for e, d, v in imp_df.iter_rows()}
    # ed_invest_period: rows in ed_invest_pairs whose method ∈ INVEST_PERIOD.
    ed_invest_period = {(e, d) for (e, d) in ed_invest.iter_rows()
                         if methods_by_e.get(e, set()) & _INVEST_METHODS_INVEST_PERIOD}

    # invest_max_total[e]: explicit values, default 0.
    eim_df = _e_explicit_param(source, "invest_max_total").collect()
    eim_map: dict[str, float] = dict(eim_df.iter_rows())

    # cumulative_max_capacity per (e, d): explicit ed map, broadcast-able.
    cmc_df = _ed_explicit_period_param(source, "cumulative_max_capacity").collect()
    cmc_map: dict[tuple[str, str], float] = {(e, d): v
        for e, d, v in cmc_df.iter_rows()}
    # ed_invest_cumulative: ed_invest pairs with cumulative_limits method.
    ed_invest_cumulative = {(e, d) for (e, d) in ed_invest.iter_rows()
                             if "cumulative_limits" in methods_by_e.get(e, set())}

    # p_max_flow_for_unconstrained_variables: per-model max → take max.
    p_unc = 1000000.0  # default per ``entity_period_calc_params.py:1689``.
    df_unc = _try_param(source, "model", "p_max_flow_for_unconstrained_variables")
    if df_unc is not None and df_unc.height > 0:
        try:
            p_unc = max(p_unc, float(df_unc["value"].max()))
        except Exception:
            pass

    # Compute max_capacity per (e, d in period_in_use) and divide by unitsize.
    rows: list[tuple[str, str, float]] = []
    for e in entities:
        us = us_map.get(e, 1000.0)
        in_total = e in e_invest_total
        has_no_limit = e in has_no_limit_e
        for d in piu:
            if d not in period_set:
                continue  # defensive; piu is the period_in_use list
            if (e, d) in ed_invest_cumulative:
                v = cmc_map.get((e, d), 0.0)
            else:
                v = existing_map.get((e, d), 0.0)
                in_period = (e, d) in ed_invest_period
                imp = imp_map.get((e, d), 0.0)
                eim = eim_map.get(e, 0.0)
                if in_period and not in_total:
                    v += imp
                if in_total and not in_period:
                    v += eim
                if in_period and in_total:
                    v += max(imp, eim)
                if has_no_limit:
                    v += p_unc
            mu = (v / us) if us != 0.0 else 0.0
            if mu > 0.0:
                rows.append((e, d, float(mu)))
    if not rows:
        return None
    out = pl.DataFrame(rows, schema=["e", "d", "value"], orient="row").sort("e", "d")
    return Param(("e", "d"), out)


# ---------------------------------------------------------------------------
# §3.7.5/6 — Multi-year inflation cascade  (default-rate path only)
# ---------------------------------------------------------------------------
#
# Replaces the simple Γ.3.A ``p_inflation_op`` when ``model.inflation_rate
# != 0`` AND the active solve carries non-trivial ``years_represented`` /
# ``period_timeset``.  The full cascade is documented at
# ``period_calculated_params.py:280-322``: for each period d, sum across
# (d, y) ∈ years_for_period, weighted by ``years_represented[d, y]`` of
# ``(1+r)^{-(base[d, y] + pyr * offset_op)}`` where base counts cumulative
# years across the GLOBAL year universe.
#
# Γ.3.C scope: produce the cascade for the simplest non-trivial case —
# every period has exactly one represented year and ``years_from_start``
# is supplied as an absolute year offset.  This collapses to
# ``inflation_op[d] = (1+r)^{-(yr[d] + offset_op)}``.  More general shapes
# (multi-year-per-period, missing offset) gate-fail and the CSV value
# survives.


def p_inflation_op_multi_year_from_source(source: "InputSource",
                                             active_solve: str,
                                             dt: pl.DataFrame,
                                             ) -> "Param | None":
    """Multi-year inflation factor for the default 1-year-per-period case.

    Returns None when:
      * ``model.inflation_rate == 0`` (Γ.3.A's simple path covers it).
      * The active solve has no ``years_from_start`` parameter.
      * Any period has > 1 represented year (defer to CSV).
    """
    rate_df = _try_param(source, "model", "inflation_rate")
    rate_v = 0.0
    if rate_df is not None and rate_df.height > 0:
        rate_v = float(rate_df["value"][0])
    if rate_v == 0.0:
        return None  # Γ.3.A's trivial path applies.
    offset_df = _try_param(source, "model", "inflation_offset_operations")
    offset_v = 0.5
    if offset_df is not None and offset_df.height > 0:
        offset_v = float(offset_df["value"][0])

    yfs = _try_param(source, "solve", "years_from_start")
    if yfs is None or "period" not in yfs.columns:
        return None
    yfs_sub = yfs.filter(pl.col("name") == active_solve)
    if yfs_sub.height == 0:
        return None
    # Multi-year-per-period guard: if ``years_represented`` Map has
    # multiple (period, year) rows for any period, defer.
    yr = _try_param(source, "solve", "years_represented")
    if yr is not None:
        yr_sub = yr.filter(pl.col("name") == active_solve)
        if yr_sub.height > 0:
            counts = (yr_sub.group_by("period").agg(pl.len().alias("n")))
            if counts["n"].max() and counts["n"].max() > 1:
                return None

    factor = 1.0 + rate_v
    if dt is None or dt.height == 0:
        return None
    yfs_lf = yfs_sub.lazy().select(
        pl.col("period").alias("d"),
        pl.col("value").cast(pl.Float64).alias("yr"),
    )
    out = (dt.lazy().select("d").unique()
              .join(yfs_lf, on="d", how="left")
              .with_columns(yr=pl.col("yr").fill_null(0.0))
              .with_columns(
                  value=pl.lit(factor).pow(-(pl.col("yr") + offset_v))
              )
              .select("d", "value")
              .sort("d")
              .collect())
    if out.height == 0:
        return None
    return Param(("d",), out)


# ---------------------------------------------------------------------------
# §3.8.1 — p_section
# ---------------------------------------------------------------------------


def p_section_from_source(source: "InputSource",
                            dt: pl.DataFrame,
                            classified: pl.DataFrame | None = None,
                            ) -> "Param | None":
    """Section term (y-intercept) for ``min_load_efficiency`` linearisation.

    For ``min_load_efficiency``:
        section = (input(min_load) - input(1.0) * min_load) / (1 - min_load)
                = (min_load / eta_min - min_load * (1.0 / eta)) / (1 - min_load)
                = min_load * (1/eta_min - 1/eta) / (1 - min_load)

    Other ct_methods → no section row.
    """
    if dt is None or dt.height == 0:
        return None
    if classified is None:
        classified = _classify_process_method(source)
    if classified.height == 0:
        return None
    # Only min_load_efficiency rows get a section.
    mle = (classified.lazy()
              .filter(pl.col("ct") == "min_load_efficiency")
              .select("p"))
    eff_unit = _try_param(source, "unit", "efficiency")
    eff_at_min = _try_param(source, "unit", "efficiency_at_min_load")
    minload = _try_param(source, "unit", "min_load")
    if eff_unit is None or eff_at_min is None or minload is None:
        return None

    eta_lf = _broadcast_param_to_dt(eff_unit, dt, value_col_alias="eta")
    eta_min_lf = _broadcast_param_to_dt(eff_at_min, dt,
                                          value_col_alias="eta_min")
    if eta_lf is None or eta_min_lf is None:
        return None
    ml_lf = (minload.lazy().select(
        pl.col("name").alias("p"),
        pl.col("value").cast(pl.Float64).alias("min_load"),
    ))

    # Mirror flextool's rounding (entity_period_calc_params.py:1283-1291):
    #   cr = round(1/eta, 6)
    #   sec = cr - round((cr - ml * (1/eta_min)) / (1 - ml), 6)
    out = (mle.join(eta_lf, on="p", how="inner")
                .join(eta_min_lf, on=["p", "d", "t"], how="inner")
                .join(ml_lf, on="p", how="inner")
                .with_columns(
                    cr=(1.0 / pl.col("eta")).round(6),
                    inv_em=1.0 / pl.col("eta_min"),
                    denom=1.0 - pl.col("min_load"),
                )
                .with_columns(
                    rounded=((pl.col("cr") - pl.col("min_load")
                                 * pl.col("inv_em"))
                              / pl.col("denom")).round(6)
                )
                .with_columns(section=pl.col("cr") - pl.col("rounded"))
                .filter(pl.col("section").is_not_null())
                .select("p", "d", "t", pl.col("section").alias("value"))
                .sort("p", "d", "t")
                .collect())
    if out.height == 0:
        return None
    return Param(("p", "d", "t"), out)


# ---------------------------------------------------------------------------
# §3.3.5 — p_flow_upper  (Δ.26)
# ---------------------------------------------------------------------------
#
# Native port of flextool's
# ``preprocessing/process_arc_unions.py:write_p_flow_max`` (file:line
# `process_arc_unions.py:1469-1624` for `write_p_flow_max`).  Mirrors
# flextool.mod L1661-1677:
#
#     p_flow_max{(p, source, sink, d, t) in peedt} :=
#       if (p, source, sink) in process_source_sink_coeff_zero
#       then p_unconstrained_flow_cap
#       else (
#         if exists{(p, m) in process__method_indirect} 1
#            AND (p, source) in process_source
#         then ( if (p, 'min_load_efficiency') in process__ct_method
#                then pdtProcess_slope[p,d,t] + pdtProcess_section[p,d,t]
#                else pdtProcess_slope[p,d,t]
#              )
#              * (p_entity_dispatch_capacity_max[p,d] / p_entity_unitsize[p])
#              / p_process_source_max_capacity_coefficient[p, source]
#         else (p_entity_dispatch_capacity_max[p,d] / p_entity_unitsize[p])
#       )
#       * (if (p, sink) in process_sink
#          then p_process_sink_max_capacity_coefficient[p, sink] else 1)
#
# Slow-path consumer: ``input.py::_load_process_topology`` reads
# ``solve_data/p_flow_max.csv`` (preprocessed CSV).  Δ.26 ports the
# derivation natively as a polars-lazy helper, wired into
# :func:`apply_derived_c` so the fast single-solve path produces the
# correct ``p_flow_upper`` Param without any preprocessing.


def _arc_max_capacity_coef_lf(source: "InputSource",
                                  side: str,
                                  ) -> pl.LazyFrame:
    """Per-(p, node) ``max_capacity_coefficient`` from the relationship-class
    parameter on ``unit__inputNode`` / ``unit__outputNode``.

    ``side='source'`` → reads ``unit__inputNode.max_capacity_coefficient``
    and returns ``[p, source, coef]``; ``side='sink'`` → reads
    ``unit__outputNode.max_capacity_coefficient`` and returns
    ``[p, sink, coef]``.

    Connections don't carry a ``max_capacity_coefficient`` parameter in
    the canonical Spine schema (the .mod's per-arc coef is unit-only;
    connections always default to 1 on the Coefficient cascade — see
    flextool.mod L686-687).  We mirror that by emitting only unit-arc
    rows; downstream callers fill the default 1.0 via left-join.
    """
    if side == "source":
        ec = "unit__inputNode"
        node_alias = "source"
    else:
        ec = "unit__outputNode"
        node_alias = "sink"
    df = _try_param(source, ec, "max_capacity_coefficient")
    if df is None or df.height == 0:
        return pl.LazyFrame(
            schema={"p": pl.Utf8, node_alias: pl.Utf8, "coef": pl.Float64})
    cols = df.columns
    rename: dict[str, str] = {}
    for c in cols:
        if c == "unit":
            rename[c] = "p"
        elif c == "node":
            rename[c] = node_alias
    return (df.lazy().rename(rename)
              .with_columns(pl.col("value").cast(pl.Float64).alias("coef"))
              .select("p", node_alias, "coef")
              .unique(subset=["p", node_alias], keep="last"))


def _process_source_pairs_lf(source: "InputSource") -> pl.LazyFrame:
    """``process_source`` set ``[p, source]`` — union of unit input arcs
    and connection's first node.  Mirrors flextool.mod L686 + the
    expansion in ``input_writer.write_topology``."""
    parts: list[pl.LazyFrame] = []
    uin = _try_entities(source, "unit__inputNode")
    if uin is not None and uin.height > 0:
        parts.append(uin.lazy().select(
            pl.col("unit").alias("p"),
            pl.col("node").alias("source"),
        ))
    cnn = _try_entities(source, "connection__node__node")
    if cnn is not None and cnn.height > 0:
        parts.append(cnn.lazy().select(
            pl.col("connection").alias("p"),
            pl.col("node_1").alias("source"),
        ))
    if not parts:
        return pl.LazyFrame(schema={"p": pl.Utf8, "source": pl.Utf8})
    return pl.concat(parts).unique()


def _process_sink_pairs_lf(source: "InputSource") -> pl.LazyFrame:
    """``process_sink`` set ``[p, sink]`` — union of unit output arcs
    and connection's second node.  Mirrors flextool.mod L687."""
    parts: list[pl.LazyFrame] = []
    uout = _try_entities(source, "unit__outputNode")
    if uout is not None and uout.height > 0:
        parts.append(uout.lazy().select(
            pl.col("unit").alias("p"),
            pl.col("node").alias("sink"),
        ))
    cnn = _try_entities(source, "connection__node__node")
    if cnn is not None and cnn.height > 0:
        parts.append(cnn.lazy().select(
            pl.col("connection").alias("p"),
            pl.col("node_2").alias("sink"),
        ))
    if not parts:
        return pl.LazyFrame(schema={"p": pl.Utf8, "sink": pl.Utf8})
    return pl.concat(parts).unique()


def _process_source_sink_coeff_zero_lf(source: "InputSource",
                                            pss: pl.DataFrame,
                                            ) -> pl.LazyFrame:
    """``process_source_sink_coeff_zero`` set: rows of pss whose source-
    or sink-side ``max_capacity_coefficient`` is explicitly zero.

    Mirrors flextool.mod L2219-2220 +
    ``preprocessing/process_arc_unions.py:write_process_source_sink_coeff_zero``
    (file:line `process_arc_unions.py:1006-1027`).
    """
    if pss is None or pss.height == 0:
        return pl.LazyFrame(schema={
            "p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    src_zero = (_arc_max_capacity_coef_lf(source, "source")
                  .filter(pl.col("coef") == 0.0)
                  .select("p", "source"))
    sink_zero = (_arc_max_capacity_coef_lf(source, "sink")
                   .filter(pl.col("coef") == 0.0)
                   .select("p", "sink"))
    src_match = (pss.lazy()
                   .join(src_zero, on=["p", "source"], how="inner")
                   .select("p", "source", "sink"))
    sink_match = (pss.lazy()
                    .join(sink_zero, on=["p", "sink"], how="inner")
                    .select("p", "source", "sink"))
    return pl.concat([src_match, sink_match]).unique()


def p_flow_upper_from_source(source: "InputSource",
                                pss: pl.DataFrame,
                                dt: pl.DataFrame,
                                p_slope: "Param | None",
                                p_section: "Param | None",
                                p_unitsize: "Param | None",
                                p_entity_max_units: "Param | None",
                                p_entity_all_existing: "Param | None",
                                classified: pl.DataFrame | None = None,
                                ) -> "Param | None":
    """§3.3.5 — per-(p, source, sink, d, t) structural max flow.

    Native port of flextool's
    ``preprocessing/process_arc_unions.py:write_p_flow_max``
    (file:line `process_arc_unions.py:1469-1624`).  Returns a Param
    keyed on ``(p, source, sink, d, t)`` whose values match
    ``solve_data/p_flow_max.csv`` written by the slow path.

    Inputs:

    ``pss`` — ``process_source_sink`` set; the (p, source, sink) index
    of the output Param.  Cross-joined with ``dt`` to form ``peedt``.

    ``p_slope`` / ``p_section`` — per-(p, d, t) slope / section terms
    from :func:`p_slope_from_source` / :func:`p_section_from_source`.
    Used in the indirect-method branch.

    ``p_unitsize`` — per-(p,) unitsize cascade from
    :func:`_derived_arithmetic.p_unitsize_from_source`.  Caller may
    also pass the per-entity cascade keyed on ``(e,)``; we accept either
    column name.

    ``p_entity_max_units`` — per-(e, d) ``max_capacity / unitsize`` from
    :func:`p_entity_max_units_from_source`.  Provides
    ``p_entity_dispatch_capacity_max[p, d] / p_entity_unitsize[p]``
    directly, so we don't need to recompute the dispatch cap inline.

    ``p_entity_all_existing`` — fallback when ``p_entity_max_units`` is
    absent (no invest configured); when both invest sets are empty
    flextool's preprocessing still emits ``p_flow_max`` rows valued at
    ``existing/unitsize`` per arc.  This matches the slow path's
    behaviour in the no-invest case.

    Returns ``None`` when no usable inputs exist (empty pss or dt) —
    caller leaves the Param at its empty placeholder.
    """
    if pss is None or pss.height == 0 or dt is None or dt.height == 0:
        return None

    # ── 1. Resolve the (p, d) cap-per-unitsize term ─────────────────
    # Prefer p_entity_max_units (= dispatch_capacity_max / unitsize per
    # (e, d)).  When absent (no invest in fixture), derive it from
    # p_entity_all_existing / p_unitsize.
    if p_entity_max_units is not None and p_entity_max_units.frame.height > 0:
        cap_per_unit_lf = (p_entity_max_units.frame.lazy()
            .rename({"e": "p"})
            .select("p", "d", pl.col("value").alias("cap_per_unit")))
    else:
        # Derive from existing / unitsize.  We only need rows that join
        # against pss, so we filter inline.
        if p_entity_all_existing is None or p_entity_all_existing.frame.height == 0:
            return None
        # p_unitsize may come keyed on (p,) (post-Δ.4b override) or (e,)
        # (raw cascade); accept both.
        if p_unitsize is None or p_unitsize.frame.height == 0:
            us_df = _entity_unitsize_lf(source).rename({"e": "p", "us": "us"}).collect()
        else:
            us_cols = p_unitsize.frame.columns
            us_key = "p" if "p" in us_cols else "e"
            us_df = p_unitsize.frame.rename({us_key: "p", "value": "us"})
        cap_per_unit_lf = (
            p_entity_all_existing.frame.lazy()
                .rename({"e": "p", "value": "cap"})
                .join(us_df.lazy(), on="p", how="left")
                .with_columns(
                    cap_per_unit=pl.when((pl.col("us").is_not_null())
                                          & (pl.col("us") != 0.0))
                                    .then(pl.col("cap") / pl.col("us"))
                                    .otherwise(0.0))
                .select("p", "d", "cap_per_unit")
        )

    # ── 2. Indirect-method partition ────────────────────────────────
    # Needs (p, source) ∈ process_source AND p ∈ process__method_indirect.
    if classified is None:
        classified = _classify_process_method(source)
    indirect_p = (classified.lazy()
                    .filter((pl.col("klass") == "unit")
                            & pl.col("method").is_in(list(_METHOD_INDIRECT)))
                    .select("p")
                    .unique())
    process_source_lf = _process_source_pairs_lf(source)
    process_sink_lf = _process_sink_pairs_lf(source)

    # min_load_efficiency rows (units only).
    min_load_p = (classified.lazy()
                    .filter(pl.col("ct") == "min_load_efficiency")
                    .select("p")
                    .unique())

    # ── 3. Coefficient frames ───────────────────────────────────────
    src_coef_lf = _arc_max_capacity_coef_lf(source, "source")
    sink_coef_lf = _arc_max_capacity_coef_lf(source, "sink")

    # ── 4. p_unconstrained_flow_cap ────────────────────────────────
    p_unc = 1_000_000.0
    df_unc = _try_param(source, "model", "p_max_flow_for_unconstrained_variables")
    if df_unc is not None and df_unc.height > 0:
        try:
            p_unc = max(p_unc, float(df_unc["value"].max()))
        except Exception:
            pass

    # ── 5. coeff_zero set — rows that get the unconstrained value ─
    coeff_zero_lf = _process_source_sink_coeff_zero_lf(source, pss)

    # ── 6. Build peedt = pss × dt with cap-per-unit and indirect tags ─
    pss_lf = pss.lazy().select("p", "source", "sink")
    base = (pss_lf
        .join(dt.lazy(), how="cross")  # adds d, t
        .join(cap_per_unit_lf, on=["p", "d"], how="left")
        .with_columns(
            cap_per_unit=pl.col("cap_per_unit").fill_null(0.0))
        # Tag indirect arcs.
        .join(indirect_p.with_columns(_is_indirect=pl.lit(True)),
                on="p", how="left")
        .with_columns(
            _is_indirect=pl.col("_is_indirect").fill_null(False))
        # Tag (p, source) ∈ process_source.
        .join(process_source_lf.with_columns(_has_source=pl.lit(True)),
                on=["p", "source"], how="left")
        .with_columns(
            _has_source=pl.col("_has_source").fill_null(False))
        # Tag min_load_efficiency processes.
        .join(min_load_p.with_columns(_has_min_load=pl.lit(True)),
                on="p", how="left")
        .with_columns(
            _has_min_load=pl.col("_has_min_load").fill_null(False))
        # Source-side max_capacity_coefficient (default 1.0).
        .join(src_coef_lf.rename({"coef": "src_coef"}),
                on=["p", "source"], how="left")
        .with_columns(
            src_coef=pl.col("src_coef").fill_null(1.0))
        # Sink-side max_capacity_coefficient (default 1.0); also tag
        # (p, sink) ∈ process_sink (the .mod multiplies by sink_coef
        # only when (p, sink) ∈ process_sink, defaulting to 1 outside).
        .join(process_sink_lf.with_columns(_has_sink=pl.lit(True)),
                on=["p", "sink"], how="left")
        .with_columns(
            _has_sink=pl.col("_has_sink").fill_null(False))
        .join(sink_coef_lf.rename({"coef": "sink_coef"}),
                on=["p", "sink"], how="left")
        .with_columns(
            sink_coef=pl.col("sink_coef").fill_null(1.0))
        # Mark coeff_zero rows.
        .join(coeff_zero_lf.with_columns(_coeff_zero=pl.lit(True)),
                on=["p", "source", "sink"], how="left")
        .with_columns(
            _coeff_zero=pl.col("_coeff_zero").fill_null(False))
    )

    # ── 7. Slope / section join (only relevant for indirect arcs) ───
    if p_slope is not None and p_slope.frame.height > 0:
        slope_lf = (p_slope.frame.lazy()
                      .select("p", "d", "t",
                                pl.col("value").cast(pl.Float64).alias("slope")))
        base = base.join(slope_lf, on=["p", "d", "t"], how="left")
    else:
        base = base.with_columns(slope=pl.lit(None, dtype=pl.Float64))
    if p_section is not None and p_section.frame.height > 0:
        section_lf = (p_section.frame.lazy()
                        .select("p", "d", "t",
                                  pl.col("value").cast(pl.Float64).alias("section")))
        base = base.join(section_lf, on=["p", "d", "t"], how="left")
    else:
        base = base.with_columns(section=pl.lit(None, dtype=pl.Float64))

    # ── 8. Compute the formula. ─────────────────────────────────────
    # eff_term used when indirect: slope (+ section iff min_load_efficiency).
    # base_unconstrained: cap_per_unit / src_coef * eff_term.
    # base_constrained:   cap_per_unit.
    # Multiply by sink_coef when (p, sink) ∈ process_sink, else *1.
    base = base.with_columns(
        eff_term=pl.when(pl.col("_has_min_load"))
                    .then(pl.col("slope").fill_null(0.0)
                            + pl.col("section").fill_null(0.0))
                    .otherwise(pl.col("slope").fill_null(0.0)),
    ).with_columns(
        # Avoid div-by-zero on src_coef; flextool's CSV reader uses
        # default 1.0 → src_coef=1 when missing (fill_null above).  But
        # explicit 0 in src_coef would crash; rows with src_coef=0
        # are also (p,source) ∈ coeff_zero so they're handled by the
        # _coeff_zero branch.  Guard with a when-then-otherwise.
        _safe_src_coef=pl.when(pl.col("src_coef") == 0.0)
                          .then(1.0)
                          .otherwise(pl.col("src_coef")),
    ).with_columns(
        _indirect_branch=(pl.col("eff_term")
                            * pl.col("cap_per_unit")
                            / pl.col("_safe_src_coef")),
        _direct_branch=pl.col("cap_per_unit"),
    ).with_columns(
        _branch_value=pl.when(pl.col("_is_indirect") & pl.col("_has_source"))
                          .then(pl.col("_indirect_branch"))
                          .otherwise(pl.col("_direct_branch")),
    ).with_columns(
        # sink-coef multiplier — only when (p, sink) ∈ process_sink.
        _sink_factor=pl.when(pl.col("_has_sink"))
                          .then(pl.col("sink_coef"))
                          .otherwise(1.0),
    ).with_columns(
        value=pl.when(pl.col("_coeff_zero"))
                  .then(pl.lit(p_unc))
                  .otherwise(pl.col("_branch_value") * pl.col("_sink_factor")),
    )

    out = (base
        .select("p", "source", "sink", "d", "t", "value")
        .sort("p", "source", "sink", "d", "t")
        .collect())
    if out.height == 0:
        return None
    return Param(("p", "source", "sink", "d", "t"), out)


# ---------------------------------------------------------------------------
# §3.8.2 / §3.8.3 — pdt_uptime_set / pdt_downtime_set / lookbacks
# ---------------------------------------------------------------------------


def _ordered_steps_from_dt(dt: pl.DataFrame) -> list[tuple[str, str, float]]:
    """Return [(d, t, step_duration)] in a stable (d, t) order.  Step
    duration defaults to 1.0 when not present (caller may supply
    ``p_step_duration`` separately).
    """
    if dt is None:
        return []
    df = dt.sort("d", "t")
    if "step_duration" in df.columns:
        return [(d, t, float(s)) for d, t, s in zip(
            df["d"].to_list(), df["t"].to_list(),
            df["step_duration"].to_list())]
    return [(d, t, 1.0) for d, t in zip(df["d"].to_list(), df["t"].to_list())]


def _build_lookback_rows(processes_min: list[tuple[str, float]],
                            ordered_steps: list[tuple[str, str, float]],
                            ) -> list[tuple[str, str, str, str, str]]:
    """Mirror flextool's ``_write_lookback_csv`` (minimum_time.py:122-169).

    For each (process, min_time) and each (d, t):
      * Always include the current step (d, t, d, t).
      * Walk backwards through the ordered timeline accumulating
        durations; include each predecessor while accumulated < min_time.

    The simple linear timeline (single-period or sequential periods with
    no gaps) suffices for our covered fixtures.  Gap detection (jump != 1)
    is not modelled — fixtures that activate it gate-fail.
    """
    rows: list[tuple[str, str, str, str, str]] = []
    n = len(ordered_steps)
    if n == 0:
        return rows
    # Build index from (d, t) → position.
    idx_by_pos: dict[tuple[str, str], int] = {
        (d, t): i for i, (d, t, _) in enumerate(ordered_steps)
    }
    for proc, min_time in sorted(processes_min):
        if min_time <= 0:
            continue
        for i, (d, t, _) in enumerate(ordered_steps):
            rows.append((proc, d, t, d, t))
            accumulated = 0.0
            j = i - 1
            while j >= 0:
                d_prev, t_prev, dur_prev = ordered_steps[j]
                accumulated += dur_prev
                if accumulated < min_time:
                    rows.append((proc, d, t, d_prev, t_prev))
                    j -= 1
                else:
                    break
    return rows


def _process_min_time(source: "InputSource",
                        parameter_name: str) -> list[tuple[str, float]]:
    """Pull (entity, value) for ``unit.<parameter_name>`` (and connection
    when applicable) — value > 0 only.  Returns a list of (proc, min_time)
    pairs.
    """
    out: dict[str, float] = {}
    for ec in ("unit", "connection"):
        df = _try_param(source, ec, parameter_name)
        if df is None:
            continue
        for r in df.iter_rows(named=True):
            try:
                v = float(r["value"])
            except (TypeError, ValueError):
                continue
            if v > 0:
                out[r["name"]] = v
    return list(out.items())


def uptime_lookback_from_source(source: "InputSource",
                                    dt: pl.DataFrame | None,
                                    p_step_duration: object | None = None,
                                    ) -> pl.DataFrame | None:
    """Build the ``uptime_lookback`` set ``(p, d, t, d_back, t_back)``.

    Algorithm: ``minimum_time.py:_write_lookback_csv:122-169``.
    """
    if dt is None or dt.height == 0:
        return None
    procs = _process_min_time(source, "min_uptime")
    if not procs:
        return None
    ordered = _step_order_with_duration(dt, p_step_duration)
    if not ordered:
        return None
    rows = _build_lookback_rows(procs, ordered)
    if not rows:
        return None
    return pl.DataFrame(rows, schema=["p", "d", "t", "d_back", "t_back"],
                          orient="row").sort("p", "d", "t",
                                                "d_back", "t_back")


def downtime_lookback_from_source(source: "InputSource",
                                       dt: pl.DataFrame | None,
                                       p_step_duration: object | None = None,
                                       ) -> pl.DataFrame | None:
    """Build the ``downtime_lookback`` set."""
    if dt is None or dt.height == 0:
        return None
    procs = _process_min_time(source, "min_downtime")
    if not procs:
        return None
    ordered = _step_order_with_duration(dt, p_step_duration)
    if not ordered:
        return None
    rows = _build_lookback_rows(procs, ordered)
    if not rows:
        return None
    return pl.DataFrame(rows, schema=["p", "d", "t", "d_back", "t_back"],
                          orient="row").sort("p", "d", "t",
                                                "d_back", "t_back")


def _step_order_with_duration(dt: pl.DataFrame,
                                  p_step_duration: object | None
                                  ) -> list[tuple[str, str, float]]:
    """Return (d, t, duration) tuples in dt's natural order."""
    if dt is None or dt.height == 0:
        return []
    if "step_duration" in dt.columns:
        df = dt.sort("d", "t")
        return [(d, t, float(s)) for d, t, s in zip(
            df["d"].to_list(), df["t"].to_list(),
            df["step_duration"].to_list())]
    sd = (p_step_duration.frame
            if p_step_duration is not None
                and hasattr(p_step_duration, "frame")
            else p_step_duration)
    if sd is None:
        df = dt.sort("d", "t")
        return [(d, t, 1.0) for d, t in zip(
            df["d"].to_list(), df["t"].to_list())]
    df = (dt.lazy()
            .join(sd.lazy(), on=["d", "t"], how="left")
            .with_columns(value=pl.col("value").fill_null(1.0))
            .sort("d", "t")
            .collect())
    return [(d, t, float(v)) for d, t, v in zip(
        df["d"].to_list(), df["t"].to_list(), df["value"].to_list())]


def pdt_uptime_set_from_lookback(uptime_lookback: pl.DataFrame | None
                                     ) -> pl.DataFrame | None:
    """Project ``(p, d, t, d_back, t_back) → (p, d, t)`` distinct."""
    if uptime_lookback is None or uptime_lookback.height == 0:
        return None
    return (uptime_lookback.lazy()
              .select("p", "d", "t").unique()
              .sort("p", "d", "t")
              .collect())


def pdt_downtime_set_from_lookback(downtime_lookback: pl.DataFrame | None
                                       ) -> pl.DataFrame | None:
    if downtime_lookback is None or downtime_lookback.height == 0:
        return None
    return (downtime_lookback.lazy()
              .select("p", "d", "t").unique()
              .sort("p", "d", "t")
              .collect())


# ---------------------------------------------------------------------------
# §3.12.1 — process_group_inside_nonSync
# ---------------------------------------------------------------------------


def process_group_inside_nonSync_from_source(source: "InputSource"
                                                  ) -> pl.DataFrame | None:
    """Process×group set: processes belonging to a group nested inside a
    non-sync group.  Algorithm (audit §3.12.1, ``nonsync_sets.py:90-152``):

      1. Pull non-sync groups (``group.<some_flag>`` or hand-curated).
      2. Walk ``group__group`` nesting transitively.
      3. For each (process, group) ∈ ``unit__inputNode → node ∈ group``,
         emit if group is in the closure.

    Defaults to None (empty) — fixtures without nested non-sync groups
    surface this as a None overlay (no-op in load).
    """
    # Spine source has no direct entity class for ``process__group_inside_group_nonSync``
    # — flextool builds it from ``group__group`` + ``group__node`` + the
    # process arc topology.  Without a covered fixture we return None
    # and rely on the CSV value (always empty for our scenarios).
    return None


# ---------------------------------------------------------------------------
# §3.12.2 — p_inv_group_cap / p_group_capacity_for_scaling
# ---------------------------------------------------------------------------


def p_group_capacity_for_scaling_from_source(source: "InputSource",
                                                  active_solve: str | None,
                                                  workdir: Path | None = None,
                                                  ) -> "Param | None":
    """Per-(g, d) row scaling factor.  Algorithm (audit §3.12.2,
    ``lp_scaling_params.py:91-244``): pow10-clamped per-group capacity
    proxy when scaling is active, else 1.0 across all (g, d).

    For Γ.3.C we cover the **scaling-inactive default**: emit 1.0 across
    all (group, period) pairs (including stochastic-branch siblings when
    workdir-provided ``period_in_use_set.csv`` is read).  Active scaling
    (rare; activated via ``solve.use_row_scaling``) returns None and the
    CSV value survives.
    """
    period_in_use = _period_in_use_set(source, active_solve, workdir)
    if not period_in_use:
        return None
    groups_df = _try_entities(source, "group")
    if groups_df is None or groups_df.height == 0:
        return None
    # Detect scaling active: solve.use_row_scaling truthy.
    urs = _try_param(source, "solve", "use_row_scaling")
    scaling_active = False
    if urs is not None and active_solve is not None:
        sub = urs.filter(pl.col("name") == active_solve)
        if sub.height > 0:
            try:
                scaling_active = float(sub["value"][0]) >= 0.5
            except (TypeError, ValueError):
                pass
    if scaling_active:
        return None  # defer; CSV path handles pow10 cascade
    out = (groups_df.lazy().select(pl.col("name").alias("g"))
              .join(pl.LazyFrame({"d": period_in_use}), how="cross")
              .with_columns(value=pl.lit(1.0))
              .sort("g", "d")
              .collect())
    if out.height == 0:
        return None
    return Param(("g", "d"), out)


def p_inv_group_cap_from_source(source: "InputSource",
                                    active_solve: str | None,
                                    workdir: Path | None = None,
                                    ) -> "Param | None":
    """Reciprocal of ``p_group_capacity_for_scaling``.  When scaling is
    inactive (default), this is 1.0 per (g, d).
    """
    gcs = p_group_capacity_for_scaling_from_source(source, active_solve,
                                                       workdir)
    if gcs is None:
        return None
    out = (gcs.frame.lazy()
              .with_columns(value=1.0 / pl.col("value"))
              .sort("g", "d")
              .collect())
    return Param(("g", "d"), out)


def p_node_capacity_for_scaling_from_source(source: "InputSource",
                                                  active_solve: str | None,
                                                  workdir: Path | None = None,
                                                  ) -> "Param | None":
    """Per-(n, d) row scaling factor for slack-penalty terms in the
    objective (``vq_up``/``vq_down`` × ``p_penalty`` × this factor).

    Δ.29 — tactical default-1.0 path matching
    :func:`p_group_capacity_for_scaling_from_source`.  Algorithm
    (audit §3.12.2, ``lp_scaling_params.py:91-244``): pow10-clamped
    per-node capacity proxy when ``solve.use_row_scaling`` is active,
    else 1.0 across all (nodeBalance, period_in_use) pairs.

    For Δ.29 we cover the **scaling-inactive default**: emit 1.0 across
    nodeBalance × period_in_use.  Active scaling (rare; gated on
    ``solve.use_row_scaling``) returns None and the CSV value survives.

    The slow path's :func:`_load_node_capacity_for_scaling` reads
    ``solve_data/node_capacity_for_scaling.csv`` and filters to
    nodeBalance via ``df.join(nb, on='n', how='inner')``; we replicate
    that filter inline here.

    TODO(future): port the full pow10 cascade in
    ``preprocessing/lp_scaling_params.py`` for fixtures with
    ``solve.use_row_scaling=1``.  For now the helper bails (returns
    None) when scaling is active so the seed CSV survives.
    """
    period_in_use = _period_in_use_set(source, active_solve, workdir)
    if not period_in_use:
        return None
    # Detect scaling active: solve.use_row_scaling truthy.
    urs = _try_param(source, "solve", "use_row_scaling")
    scaling_active = False
    if urs is not None and active_solve is not None:
        sub = urs.filter(pl.col("name") == active_solve)
        if sub.height > 0:
            try:
                scaling_active = float(sub["value"][0]) >= 0.5
            except (TypeError, ValueError):
                pass
    if scaling_active:
        return None  # defer; CSV path handles pow10 cascade
    # nodeBalance projection — same set as the slow path's filter.
    from ._projection_params import nodeBalance as _nodeBalance_proj
    nb = _nodeBalance_proj(source)
    if nb is None or nb.height == 0:
        return None
    out = (nb.lazy()
              .join(pl.LazyFrame({"d": period_in_use}), how="cross")
              .with_columns(value=pl.lit(1.0))
              .sort("n", "d")
              .collect())
    if out.height == 0:
        return None
    return Param(("n", "d"), out)


# ---------------------------------------------------------------------------
# §3.12.3 — p_positive_inflow / p_negative_inflow
# ---------------------------------------------------------------------------


def p_positive_inflow_from_inflow(p_inflow: object | None) -> "Param | None":
    """Positive component (clip-low at 0) of ``p_inflow``."""
    if p_inflow is None:
        return None
    fr = p_inflow.frame if hasattr(p_inflow, "frame") else p_inflow
    if fr is None or fr.height == 0:
        return None
    out = (fr.lazy()
            .with_columns(value=pl.when(pl.col("value") > 0.0)
                                  .then(pl.col("value"))
                                  .otherwise(0.0))
            .sort("n", "d", "t")
            .collect())
    return Param(("n", "d", "t"), out)


def p_negative_inflow_from_inflow(p_inflow: object | None) -> "Param | None":
    """Negative component (clip-high at 0) of ``p_inflow``."""
    if p_inflow is None:
        return None
    fr = p_inflow.frame if hasattr(p_inflow, "frame") else p_inflow
    if fr is None or fr.height == 0:
        return None
    out = (fr.lazy()
            .with_columns(value=pl.when(pl.col("value") < 0.0)
                                  .then(pl.col("value"))
                                  .otherwise(0.0))
            .sort("n", "d", "t")
            .collect())
    return Param(("n", "d", "t"), out)


# ---------------------------------------------------------------------------
# §3.12.4 — pdtNodeInflow_per_step
# ---------------------------------------------------------------------------


def pdtNodeInflow_per_step_from_inflow(p_inflow: object | None,
                                          p_step_duration: object | None,
                                          ) -> "Param | None":
    """``p_inflow / p_step_duration`` per (n, d, t).  Used by
    capacity_margin RHS.  Mirrors the .mod's
    ``pdtNodeInflow / step_duration`` divider.

    Note: ``p_inflow`` here is flextool's ``pdtNodeInflow`` shape — a
    long-format (n, d, t, value) frame already integrated over the step.
    Dividing by ``step_duration`` returns the per-hour rate.
    """
    if p_inflow is None or p_step_duration is None:
        return None
    inflow_fr = (p_inflow.frame if hasattr(p_inflow, "frame") else p_inflow)
    sd_fr = (p_step_duration.frame if hasattr(p_step_duration, "frame")
              else p_step_duration)
    if (inflow_fr is None or inflow_fr.height == 0
            or sd_fr is None or sd_fr.height == 0):
        return None
    sd_lf = sd_fr.lazy().select(
        "d", "t", pl.col("value").cast(pl.Float64).alias("dur"))
    out = (inflow_fr.lazy()
              .join(sd_lf, on=["d", "t"], how="inner")
              .with_columns(value=pl.col("value") / pl.col("dur"))
              .select("n", "d", "t", "value")
              .sort("n", "d", "t")
              .collect())
    if out.height == 0:
        return None
    return Param(("n", "d", "t"), out)


# ===========================================================================
# Γ.3.C integration entrypoint
# ===========================================================================


DERIVED_C_FIELDS = (
    # §3.7
    "ed_invest_set", "ed_divest_set",
    "edd_invest_lookback_set", "edd_divest_active",
    "p_entity_max_units",
    # §3.8
    "p_section",
    "uptime_lookback", "downtime_lookback",
    "pdt_uptime_set", "pdt_downtime_set",
    # §3.12
    "process_group_inside_nonSync",
    "p_group_capacity_for_scaling", "p_inv_group_cap",
    "p_node_capacity_for_scaling",
    "p_positive_inflow", "p_negative_inflow",
    "pdtNodeInflow_per_step",
    # §3.1.3 multi-year (extends Γ.3.A)
    "p_inflation_op",
)


def _set_eq_sorted(a: pl.DataFrame | None,
                     b: pl.DataFrame | None,
                     keys: tuple[str, ...]) -> bool:
    """Frame equality on a *set-shaped* DataFrame (no value column).

    Sort by ``keys`` then call ``.equals``.  Either being None or empty
    → equal iff the other is None / empty.
    """
    a_empty = a is None or a.height == 0
    b_empty = b is None or b.height == 0
    if a_empty and b_empty:
        return True
    if a_empty or b_empty:
        return False
    cols_a = set(a.columns)
    cols_b = set(b.columns)
    if cols_a != cols_b:
        return False
    sk = [k for k in keys if k in cols_a]
    return a.sort(sk).equals(b.sort(sk))


def apply_derived_c(
    flex_data: object,
    source: "InputSource",
    workdir: Path,
    *,
    ctx: "SolveContext | None" = None,
) -> None:
    """Apply Γ.3.C Derived Params, mutating ``flex_data`` in place.

    Order:
      1. Multi-year inflation (extends Γ.3.A's default-rate path).
      2. §3.12 group slack (smallest, foundational for cap_margin).
      3. §3.8 online/UC.
      4. §3.7 invest/divest cascade.
      5. §3.11 (capped by §3.7.5/6 — no separate Params).

    Δ.3 replaced the previous ``derived_overrides_c`` dict-return;
    Δ.4 deleted the deprecated wrapper alias.
    """
    active_solve = ctx.solve_name if ctx is not None else _read_active_solve(workdir)
    dt_csv = getattr(flex_data, "dt", None)
    sd_csv = getattr(flex_data, "p_step_duration", None)

    # Δ.12b — assignment is unconditional except for fields with
    # documented helper-coverage gaps (multi-year cascade extends
    # apply_derived_a's path; fall-throughs noted inline).

    # ─── Multi-year inflation cascade ─────────────────────────────────
    # TODO(Δ.12b helper-fix): the multi-year helper EXTENDS rather than
    # replaces apply_derived_a's p_inflation_op (single-year path) — when
    # the multi-year cascade isn't applicable (single-year fixture) the
    # helper returns None and we want to retain apply_derived_a's value.
    # Keep the conditional assignment.
    if active_solve is not None and dt_csv is not None:
        infl_my = p_inflation_op_multi_year_from_source(
            source, active_solve, dt_csv)
        if infl_my is not None:
            flex_data.p_inflation_op = infl_my

    # ─── §3.12 group slack ────────────────────────────────────────────
    # TODO(Δ.12b helper-fix): the helpers cover the scaling-inactive
    # default path; scaling-active fixtures (``solve.use_row_scaling``)
    # require the pow10 cascade in ``lp_scaling_params.py:91-244`` which
    # the seed-side ``_group_slack`` reads directly from
    # ``solve_data/group_capacity_for_scaling.csv`` /
    # ``solve_data/inv_group_cap.csv``.  Keep the conditional assignment.
    gcs_db = p_group_capacity_for_scaling_from_source(
        source, active_solve, workdir)
    if gcs_db is not None:
        flex_data.p_group_capacity_for_scaling = gcs_db
    igc_db = p_inv_group_cap_from_source(source, active_solve, workdir)
    if igc_db is not None:
        flex_data.p_inv_group_cap = igc_db
    # Δ.29 — tactical default-1.0 path for p_node_capacity_for_scaling.
    # The proper fix is the lp_scaling_params pow10 cascade
    # (``preprocessing/lp_scaling_params.py:91-244``); until that lands,
    # default to 1.0 broadcast across (nodeBalance, period_in_use).
    # Without this default, the fast path leaves the field None for
    # fixtures with no preprocessed CSV → the slack penalty becomes
    # the only sink for unmet demand without the row-scaling factor,
    # but more critically when the pow10 cascade is active CSV-side and
    # the fast path emits None, the LP misses the scaling on penalty
    # terms (penalty becomes a free source of energy).  Default 1.0
    # produces the same numeric coefficient as the None-branch in
    # ``model.py:2349-2356`` but exposes the field uniformly so future
    # consumers (e.g. region filter, warm cascade) see a populated
    # frame.
    ncs_db = p_node_capacity_for_scaling_from_source(
        source, active_solve, workdir)
    if ncs_db is not None:
        flex_data.p_node_capacity_for_scaling = ncs_db

    # Inflow derivatives — these consume the (CSV-loaded or override-set)
    # p_inflow.  Helpers return None when p_inflow is None.
    p_inflow_csv = getattr(flex_data, "p_inflow", None)
    flex_data.p_positive_inflow = p_positive_inflow_from_inflow(p_inflow_csv)
    flex_data.p_negative_inflow = p_negative_inflow_from_inflow(p_inflow_csv)
    flex_data.pdtNodeInflow_per_step = pdtNodeInflow_per_step_from_inflow(
        p_inflow_csv, sd_csv)

    # process_group_inside_nonSync — empty for all our covered fixtures.
    flex_data.process_group_inside_nonSync = (
        process_group_inside_nonSync_from_source(source))

    # ─── §3.8 online / UC ─────────────────────────────────────────────
    classified = _classify_process_method(source)
    if dt_csv is not None and classified is not None and classified.height > 0:
        flex_data.p_section = p_section_from_source(source, dt_csv, classified)

    # uptime / downtime lookbacks + projected pdt_*_set.
    ulb_db = uptime_lookback_from_source(source, dt_csv, sd_csv)
    if ulb_db is not None and ulb_db.height > 0:
        flex_data.uptime_lookback = ulb_db
        flex_data.pdt_uptime_set = pdt_uptime_set_from_lookback(ulb_db)

    dlb_db = downtime_lookback_from_source(source, dt_csv, sd_csv)
    if dlb_db is not None and dlb_db.height > 0:
        flex_data.downtime_lookback = dlb_db
        flex_data.pdt_downtime_set = pdt_downtime_set_from_lookback(dlb_db)

    # ─── §3.7 invest / divest ─────────────────────────────────────────
    # Δ.12b — set frames (None or non-empty); keep height>0 as a
    # structural filter to preserve the SET-frame contract.
    ed_inv_db = ed_invest_set_from_source(source, active_solve, workdir)
    if ed_inv_db is not None and ed_inv_db.height > 0:
        flex_data.ed_invest_set = ed_inv_db

    ed_div_db = ed_divest_set_from_source(source, active_solve)
    if ed_div_db is not None and ed_div_db.height > 0:
        flex_data.ed_divest_set = ed_div_db

    # Δ.17c Gap D — pd/nd_invest_set, pd/nd_divest_set, edd_invest_set:
    # partition the (e, d) frames to (process, d) / (node, d) by
    # entity-class membership.  Mirrors the user's MathProg snippets
    # for ``e_invest_total`` / ``e_divest_total`` (already wired in
    # apply_projection_params) and ``invest_divest_sets.py:215-217``.
    #
    # We use the ed_*_set values just produced (or, when this run
    # contributes nothing, fall back to the seed-loaded ed_*_set on
    # flex_data).  The partition helpers are pure projections; they're
    # cheap so re-running on every invocation is fine.
    from flextool.engine_polars._derived_existing import (
        pd_invest_set_lf as _pd_invest_lf,
        nd_invest_set_lf as _nd_invest_lf,
        pd_divest_set_lf as _pd_divest_lf,
        nd_divest_set_lf as _nd_divest_lf,
        ed_invest_period_set_lf as _ed_invest_period_lf,
        ed_divest_period_set_lf as _ed_divest_period_lf,
    )
    ed_inv_for_partition = (
        ed_inv_db if (ed_inv_db is not None and ed_inv_db.height > 0)
        else getattr(flex_data, "ed_invest_set", None))
    ed_div_for_partition = (
        ed_div_db if (ed_div_db is not None and ed_div_db.height > 0)
        else getattr(flex_data, "ed_divest_set", None))
    if ed_inv_for_partition is not None and ed_inv_for_partition.height > 0:
        ed_inv_lf = ed_inv_for_partition.lazy()
        pd_inv_db = _pd_invest_lf(source, ed_inv_lf).collect()
        if pd_inv_db.height > 0:
            flex_data.pd_invest_set = pd_inv_db
        nd_inv_db = _nd_invest_lf(source, ed_inv_lf).collect()
        if nd_inv_db.height > 0:
            flex_data.nd_invest_set = nd_inv_db
        # Per-period invest cap subset — entities whose invest_method
        # carries a period cap (``maxInvest_entity_period`` index).
        ed_inv_period_db = _ed_invest_period_lf(source, ed_inv_lf).collect()
        if ed_inv_period_db.height > 0:
            flex_data.ed_invest_period_set = ed_inv_period_db
    if ed_div_for_partition is not None and ed_div_for_partition.height > 0:
        ed_div_lf = ed_div_for_partition.lazy()
        pd_div_db = _pd_divest_lf(source, ed_div_lf).collect()
        if pd_div_db.height > 0:
            flex_data.pd_divest_set = pd_div_db
        nd_div_db = _nd_divest_lf(source, ed_div_lf).collect()
        if nd_div_db.height > 0:
            flex_data.nd_divest_set = nd_div_db
        ed_div_period_db = _ed_divest_period_lf(source, ed_div_lf).collect()
        if ed_div_period_db.height > 0:
            flex_data.ed_divest_period_set = ed_div_period_db

    # Γ.6.D — ed_invest_forbidden_no_investment.  Built off the
    # (possibly-overridden) ed_invest_set so the helper sees the same
    # frame the LP downstream consumes.
    ed_invest_for_forbidden = (
        ed_inv_db if ed_inv_db is not None
        else getattr(flex_data, "ed_invest_set", None))
    forbidden_db = ed_invest_forbidden_no_investment_from_source(
        source, active_solve, workdir, ed_invest_for_forbidden)
    if forbidden_db is not None:
        flex_data.ed_invest_forbidden_no_investment = (
            forbidden_db if forbidden_db.height > 0 else None)

    # Dispatch-only gate: when neither ed_invest nor ed_divest carries
    # any (e, d) row, mirror ``input.py::_load_invest`` blank-out by
    # nulling every invest-cascade Param.
    #
    # Δ.18 — gate must consult the *effective* invest/divest sets (the
    # seed-loaded value persists when the override returned empty, e.g.
    # for synthetic per-sub-solve names that don't exist in Spine).  Using
    # ed_inv_db / ed_div_db alone fired the blank-out spuriously on
    # per-sub-solve snapshot fixtures, nulling p_entity_max_units etc.
    # even though the snapshot CSV had genuine invest activity.
    inv_empty = ed_inv_for_partition is None or ed_inv_for_partition.height == 0
    div_empty = ed_div_for_partition is None or ed_div_for_partition.height == 0
    if inv_empty and div_empty:
        for blank_field in (
            "e_invest_total", "e_divest_total",
            "e_invest_max_total", "e_divest_max_total",
            "e_invest_min_total", "e_divest_min_total",
            "p_entity_max_units",
            "ed_invest_period_set", "ed_divest_period_set",
            "ed_invest_max_period", "ed_divest_max_period",
        ):
            setattr(flex_data, blank_field, None)
        return

    # edd_invest_lookback uses the (possibly-overlaid) ed_invest_set.
    ed_inv_used = ed_inv_db if (ed_inv_db is not None and ed_inv_db.height > 0) \
                   else getattr(flex_data, "ed_invest_set", None)
    try:
        eil_db = edd_invest_lookback_set_from_source(
            source, active_solve, ed_inv_used, workdir)
    except Exception:
        eil_db = None
    if eil_db is not None and eil_db.height > 0:
        flex_data.edd_invest_lookback_set = eil_db

    # Δ.17c Gap D — edd_invest_set: union of edd_history walks intersected
    # with ed_invest_set on (e, d_history).  Mirror of
    # ``invest_divest_sets.py:267-270``.
    if ed_inv_used is not None and ed_inv_used.height > 0:
        from flextool.engine_polars._derived_existing import (
            edd_invest_set_lf as _edd_invest_lf,
        )
        period_in_use = _period_in_use_set(source, active_solve, workdir,
                                              ctx=ctx)
        period_with_history = (_read_period_with_history(workdir)
                                  or list(period_in_use))
        try:
            edd_inv_db = _edd_invest_lf(
                source, active_solve, ed_inv_used.lazy(),
                period_with_history, period_in_use, workdir).collect()
        except Exception:
            edd_inv_db = None
        if edd_inv_db is not None and edd_inv_db.height > 0:
            flex_data.edd_invest_set = edd_inv_db

    pd_div_used = getattr(flex_data, "pd_divest_set", None)
    try:
        edda_db = edd_divest_active_from_source(
            source, active_solve, pd_div_used)
    except Exception:
        edda_db = None
    if edda_db is not None and edda_db.height > 0:
        flex_data.edd_divest_active = edda_db

    # p_entity_max_units uses ed_invest plus the per-period-in-use grid
    # and the unitsize cascade.
    try:
        pae_for_max_units = getattr(flex_data, "p_entity_all_existing", None)
        pemu_db = p_entity_max_units_from_source(
            source, ed_inv_used, active_solve, workdir=workdir,
            p_entity_all_existing=pae_for_max_units)
    except Exception:
        pemu_db = None
    if pemu_db is not None:
        flex_data.p_entity_max_units = pemu_db




# =============================================================================
# Γ.3.D — final derived-Param batch.  Narrow, conservative scope:
#
#   * §3.11 existing-entity fixed cost — ``p_entity_all_existing``
#     (sum of pre_existing + previously_invested per (e, d)).
#   * §3.16 DC PF — ``node_reference_angle`` (per-component reference
#     pick: explicit ``group.reference_node`` else largest existing).
#   * §3.13 reserve cleanup — narrow gate-on-equality overlay for the
#     `_increase_reserve_ratio` / `_large_failure_ratio` Projection sets
#     when the underlying SpineDB reserve relationships are present.
#
# Storage / lifetime cascade / handoff state / ladder / delay / multi-
# branch Params are deferred to Γ.3.E; their algorithms either depend
# on per-solve state not in Spine, or require a fully-faithful port of
# flextool preprocessing's coarse-block synthesis (input.py:1985-2126)
# whose risk profile exceeds the time budget for this batch.  See
# ``progress.md`` for the deferred-list bookkeeping.
# =============================================================================


def p_entity_all_existing_from_source(source: "InputSource",
                                        active_solve: str | None,
                                        workdir: Path | None = None,
                                        ) -> "Param | None":
    """§3.11 — sum of pre-existing capacity per (entity, period).

    First-solve (or single-solve) collapses to:

        p_entity_all_existing[e, d] = pre_existing[e, d]

    where ``pre_existing[e, d]`` is the lifetime-gated existing
    capacity from
    ``preprocessing/entity_period_calc_params.py:write_p_entity_pre_existing``:

      * ``reinvest_automatic`` → ``entity.existing[e, d]``.
      * ``reinvest_choice`` / ``no_investment`` → ``entity.existing[e, d]``
        only while ``p_years_d[d] < life_sum(e)``; 0 thereafter.
      * Any other ``lifetime_method`` → 0.

    Chain-run handoff (Γ.6.D): when ``solve_data/p_entity_period_existing_capacity.csv``
    is present in ``workdir`` it carries the chained existing capacity
    summed across the entity's history (see
    ``entity_period_calc_params.py:1463-1543``).  We prefer that CSV
    when it has explicit non-zero rows for ``period_in_use`` periods
    that aren't covered by ``entity.existing`` — captures the
    multi-solve state passed via ``apply_handoff``.
    """
    piu = _period_in_use_set(source, active_solve, workdir)
    if not piu:
        return None

    # Γ.6.D — chain-run handoff: prefer the canonical per-solve
    # ``solve_data/p_entity_all_existing.csv`` when present.  This CSV
    # carries the lifetime-cumulative + previously-invested chain
    # already integrated by the chain runner / flextool's preprocessing
    # for sub-solves N>1 (see
    # ``entity_period_calc_params.write_p_entity_existing_chain``).
    # When the CSV is absent (single-solve fixtures, or DB-direct sweep
    # tooling without preprocessing), fall back to deriving from
    # ``entity.existing`` + lifetime gate.
    if workdir is not None:
        pae_path = Path(workdir) / "solve_data" / "p_entity_all_existing.csv"
        if pae_path.exists():
            try:
                df = _read_csv_file(pae_path)
                if df.height > 0 and "entity" in df.columns \
                        and "period" in df.columns and "value" in df.columns:
                    df2 = (df.rename({"entity": "e", "period": "d"})
                              .with_columns(value=pl.col("value")
                                                       .cast(pl.Float64,
                                                                strict=False)
                                                       .fill_null(0.0))
                              .select("e", "d", "value")
                              .sort("e", "d"))
                    if df2.height > 0:
                        return Param(("e", "d"), df2)
            except Exception:
                pass
    # Enumerate all entities across unit / node / connection.
    ent_names: list[str] = []
    for ec in ("unit", "node", "connection"):
        ents = _try_entities(source, ec)
        if ents is None or ents.height == 0:
            continue
        for n in ents["name"].to_list():
            ent_names.append(str(n))
    ent_names = list(dict.fromkeys(ent_names))
    if not ent_names:
        return None
    # Build the (e, d) grid with default 0.0.
    grid = (pl.LazyFrame({"e": ent_names})
              .join(pl.LazyFrame({"d": piu}), how="cross")
              .with_columns(value=pl.lit(0.0)))
    # Read explicit existing rows.
    parts: list[pl.LazyFrame] = []
    for ec in ("unit", "node", "connection"):
        df = _try_param(source, ec, "existing")
        if df is None or df.height == 0:
            continue
        cols = df.columns
        if "period" in cols:
            parts.append(df.lazy().select(
                pl.col("name").alias("e"),
                pl.col("period").alias("d"),
                pl.col("value").cast(pl.Float64).alias("ex"),
            ))
        else:
            base = df.lazy().select(
                pl.col("name").alias("e"),
                pl.col("value").cast(pl.Float64).alias("ex"),
            )
            parts.append(base.join(pl.LazyFrame({"d": piu}), how="cross")
                              .select("e", "d", "ex"))
    if parts:
        explicit = pl.concat(parts).unique(subset=["e", "d"], keep="last")
        merged = (grid.join(explicit, on=["e", "d"], how="left")
                       .with_columns(value=pl.coalesce(pl.col("ex"),
                                                            pl.col("value")))
                       .select("e", "d", "value"))
    else:
        merged = grid.select("e", "d", "value")
    out = merged.sort("e", "d").collect()
    if out.height == 0:
        return None
    # Γ.6.D — apply the lifetime gate.  Mirrors
    # ``entity_period_calc_params.write_p_entity_pre_existing`` (line
    # 320-348): for entities with ``lifetime_method`` ∈ {reinvest_choice,
    # no_investment}, zero out existing past ``life_sum(e)`` expiry.
    expired = _lifetime_expired_pairs(
        source, active_solve, workdir,
        methods=("reinvest_choice", "no_investment"))
    # Also need to zero out entities whose lifetime_method is none of
    # the three allowed values (default fallback handles this since
    # _entity_lifetime_methods returns "reinvest_automatic" by default,
    # which is allowed — only explicit unrecognised values would fall
    # through, and flextool's own writer drops them too).
    if expired:
        expired_lf = pl.DataFrame(
            list(expired), schema=["e", "d"], orient="row").lazy()
        out = (out.lazy()
                  .join(expired_lf.with_columns(_expired=pl.lit(True)),
                          on=["e", "d"], how="left")
                  .with_columns(
                      value=pl.when(pl.col("_expired").fill_null(False))
                                .then(0.0)
                                .otherwise(pl.col("value"))
                  )
                  .select("e", "d", "value")
                  .sort("e", "d")
                  .collect())
    return Param(("e", "d"), out)


# ---------------------------------------------------------------------------
# §3.16 — DC power flow reference angle pick
# ---------------------------------------------------------------------------


def _bfs_components(adj: dict[str, set[str]],
                      seed: set[str]) -> list[set[str]]:
    """Return the connected components of an undirected graph (adj is a
    set-of-neighbours dict) restricted to seed nodes.
    """
    seen: set[str] = set()
    comps: list[set[str]] = []
    for n in sorted(seed):
        if n in seen:
            continue
        comp: set[str] = set()
        stack = [n]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.add(x)
            for nb in adj.get(x, ()):
                if nb not in seen and nb in seen:
                    pass
                if nb not in seen:
                    stack.append(nb)
        comps.append(comp)
    return comps


def node_reference_angle_from_source(source: "InputSource",
                                        ) -> pl.DataFrame | None:
    """§3.16 — per-component DC PF reference node selection.

    Algorithm (mirrors ``input_writer.py:_write_dc_power_flow_data``):

      1. Build the undirected graph from ``connection__node__node``
         restricted to connections with ``is_DC == 'yes'`` and to
         endpoints that are nodes in the DC-PF subnetwork (the
         endpoints of any DC connection).
      2. Per connected component:
         a. If any node belongs to a group with explicit
            ``reference_node = node-name``, pick that node.
         b. Else pick the node with the largest ``existing`` capacity
            (ties → lexicographic tiebreak).

    Returns a single-column ``[n]`` DataFrame, or ``None`` when the
    fixture has no DC PF connections.
    """
    is_dc = _try_param(source, "connection", "is_DC")
    if is_dc is None:
        return None
    dc_conns = is_dc.filter(pl.col("value") == "yes")["name"].unique().to_list()
    if not dc_conns:
        return None
    cnn = _try_entities(source, "connection__node__node")
    if cnn is None or cnn.height == 0:
        return None
    cols = cnn.columns
    # Identify the connection col + two node cols.
    if "connection" in cols:
        conn_col = "connection"
    else:
        conn_col = cols[0]
    node_cols = [c for c in cols if c != conn_col][:2]
    if len(node_cols) < 2:
        return None
    cnn_dc = cnn.filter(pl.col(conn_col).is_in(dc_conns))
    if cnn_dc.height == 0:
        return None
    # Build adjacency.
    adj: dict[str, set[str]] = {}
    for row in cnn_dc.iter_rows(named=True):
        a = row[node_cols[0]]
        b = row[node_cols[1]]
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    seed = set(adj.keys())
    if not seed:
        return None
    comps = _bfs_components(adj, seed)
    # Per-component pick.
    # Preferred: explicit group.reference_node within the component.
    grp_ref_df = _try_param(source, "group", "reference_node")
    grp_ref_map: dict[str, str] = {}
    if grp_ref_df is not None:
        for row in grp_ref_df.iter_rows(named=True):
            grp_ref_map[row["name"]] = row["value"]
    explicit_refs: set[str] = set(grp_ref_map.values())

    # Existing capacity per node.
    ex_df = _try_param(source, "node", "existing")
    ex_map: dict[str, float] = {}
    if ex_df is not None:
        for row in ex_df.iter_rows(named=True):
            v = row["value"]
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            # If multi-period (period column present), keep max across
            # periods so the "largest existing" picks something stable.
            cur = ex_map.get(row["name"])
            if cur is None or fv > cur:
                ex_map[row["name"]] = fv

    picks: list[str] = []
    for comp in comps:
        # 1. Explicit group reference inside component.
        explicit = sorted(comp & explicit_refs)
        if explicit:
            picks.append(explicit[0])
            continue
        # 2. Largest existing tiebreak by name.
        ranked = sorted(comp,
                         key=lambda n: (-ex_map.get(n, 0.0), n))
        if ranked:
            picks.append(ranked[0])
    if not picks:
        return None
    return pl.DataFrame({"n": sorted(set(picks))})


# ---------------------------------------------------------------------------
# §3.13 reserve helpers (narrow gate-on-equality overlays)
# ---------------------------------------------------------------------------


def _reserve_active_relationships(source: "InputSource"
                                       ) -> pl.DataFrame | None:
    """Union the unit-side and connection-side reserve relationships
    into a single ``(p, r, ud, n)`` frame.  Returns None when no reserve
    relationships exist on the source.
    """
    parts: list[pl.LazyFrame] = []
    for ec in ("reserve__upDown__unit__node",
               "reserve__upDown__connection__node"):
        ents = _try_entities(source, ec)
        if ents is None or ents.height == 0:
            continue
        cols = ents.columns
        # Schema: [reserve, upDown, unit_or_connection, node]
        # Column names from SpineDbReader use the dim class names.
        rename = {}
        for c in cols:
            if c == "reserve":
                rename[c] = "r"
            elif c == "upDown":
                rename[c] = "ud"
            elif c == "node":
                rename[c] = "n"
            elif c in ("unit", "connection"):
                rename[c] = "p"
        if "r" not in [rename.get(c, c) for c in cols] \
                or "ud" not in [rename.get(c, c) for c in cols]:
            continue
        parts.append(ents.lazy().rename(rename).select("p", "r", "ud", "n"))
    if not parts:
        return None
    return pl.concat(parts).unique().sort("p", "r", "ud", "n").collect()


def process_reserve_upDown_node_active_from_source(source: "InputSource",
                                                       ) -> pl.DataFrame | None:
    """§3.13.x — Projection of (p, r, ud, n) tuples that are active.

    The simple case is the union of the unit-side / connection-side
    reserve relationships filtered to those with non-zero
    ``reliability``.  When ``reliability`` is missing on the source,
    fall back to the relationship membership (default reliability=1.0
    per the Spine schema).
    """
    rel = _reserve_active_relationships(source)
    if rel is None or rel.height == 0:
        return None
    # If reliability is parametrised, keep rows with reliability > 0.
    rel_dfs: list[pl.LazyFrame] = []
    for ec in ("reserve__upDown__unit__node",
               "reserve__upDown__connection__node"):
        df = _try_param(source, ec, "reliability")
        if df is None or df.height == 0:
            continue
        cols = df.columns
        rename: dict[str, str] = {}
        for c in cols:
            if c == "reserve":
                rename[c] = "r"
            elif c == "upDown":
                rename[c] = "ud"
            elif c == "node":
                rename[c] = "n"
            elif c in ("unit", "connection"):
                rename[c] = "p"
        rel_dfs.append(df.lazy().rename(rename).select(
            "p", "r", "ud", "n",
            pl.col("value").cast(pl.Float64).alias("rel"),
        ))
    if not rel_dfs:
        return rel  # Defaults to "all relationships active".
    rel_lf = (pl.concat(rel_dfs)
                  .filter(pl.col("rel") > 0)
                  .select("p", "r", "ud", "n"))
    out = rel_lf.unique().sort("p", "r", "ud", "n").collect()
    return out if out.height > 0 else None


# ---------------------------------------------------------------------------
# Public — Γ.3.D field list (for selective scope-checking).
# ---------------------------------------------------------------------------


D_PUBLIC_FIELDS: tuple[str, ...] = (
    "p_entity_all_existing",
    "node_reference_angle",
    "process_reserve_upDown_node_active",
)


def apply_derived_d(
    flex_data: object,
    source: "InputSource",
    workdir: Path,
    *,
    ctx: "SolveContext | None" = None,
) -> None:
    """Apply Γ.3.D Derived Params, mutating ``flex_data`` in place.

    Scope (Tier-1 selected items):
      * §3.11 ``p_entity_all_existing`` (simple existing-only path).
      * §3.16 ``node_reference_angle`` (DC PF reference pick).
      * §3.13 ``process_reserve_upDown_node_active`` (reliability>0 set).

    Δ.12b — assignment is unconditional; helpers are authoritative
    producers.  ``None`` is the explicit "feature inactive" signal
    (no DC-PF reference angle / no reserve relationships /
    no chained existing capacity).  Hard exceptions propagate.
    """
    active_solve = ctx.solve_name if ctx is not None else _read_active_solve(workdir)

    # ─── §3.11 p_entity_all_existing — Δ.12b: unconditional ──────────
    flex_data.p_entity_all_existing = p_entity_all_existing_from_source(
        source, active_solve, workdir)

    # Δ.12c — ``apply_existing_chain`` was previously invoked here, but
    # it consumes the handoff carriers (``p_entity_previously_invested_capacity``
    # / ``p_entity_divested``) which are populated by ``apply_derived_f``.
    # Calling it here meant the chain-summation read stale (or seed-only)
    # carriers on the first iteration.  Moved to ``_apply_db_overrides``
    # so it runs strictly after ``apply_derived_f``.

    # ─── §3.16 node_reference_angle — Δ.12b: unconditional, with
    #     CSV-fallback (Δ.16).  Some fixtures (e.g. ``work_dc_power_flow``
    #     / ``case14.sqlite``) ship a pre-computed
    #     ``input/node_reference_angle.csv`` but don't populate the
    #     ``connection.is_DC`` parameter the source-side derivation
    #     consumes.  Preserve the CSV-loaded value when the source path
    #     produces nothing.
    derived_ref = node_reference_angle_from_source(source)
    if derived_ref is not None:
        flex_data.node_reference_angle = derived_ref

    # ─── §3.13 process_reserve_upDown_node_active — Δ.12b: unconditional
    flex_data.process_reserve_upDown_node_active = (
        process_reserve_upDown_node_active_from_source(source))

    # ─── §3.3.5 p_flow_upper (Δ.26) ───────────────────────────────────
    # Per-(p, source, sink, d, t) structural max flow.  Native port of
    # ``preprocessing/process_arc_unions.py:write_p_flow_max`` (file:line
    # `process_arc_unions.py:1469-1624`).  Wired in apply_derived_d so
    # ``p_entity_all_existing`` (set immediately above) is available;
    # ``p_entity_max_units`` from apply_derived_c is also picked up
    # when the fixture has invest activity.
    pss_for_upper = getattr(flex_data, "process_source_sink", None)
    dt_for_upper = getattr(flex_data, "dt", None)
    if (pss_for_upper is not None and pss_for_upper.height > 0
            and dt_for_upper is not None and dt_for_upper.height > 0):
        slope_for_upper = getattr(flex_data, "p_slope", None)
        section_for_upper = getattr(flex_data, "p_section", None)
        unitsize_for_upper = getattr(flex_data, "p_unitsize", None)
        max_units_for_upper = getattr(flex_data, "p_entity_max_units", None)
        existing_for_upper = getattr(flex_data, "p_entity_all_existing", None)
        try:
            pfu_db = p_flow_upper_from_source(
                source,
                pss_for_upper,
                dt_for_upper,
                p_slope=slope_for_upper,
                p_section=section_for_upper,
                p_unitsize=unitsize_for_upper,
                p_entity_max_units=max_units_for_upper,
                p_entity_all_existing=existing_for_upper,
            )
        except Exception:
            pfu_db = None
        if pfu_db is not None:
            flex_data.p_flow_upper = pfu_db




# ===========================================================================
# Γ.3.E — Storage block algebra (audit §3.9)
#
# Eight sub-sections covering the multi-resolution timeline algebra used by
# storage state, intra-period blocks, arc weighting, and rolling-handoff.
# No defensive gating — these helpers either produce the canonical frame or
# the parity test fails loudly.  See task spec for the architectural shift.
# ===========================================================================


def _expand_branch_periods(period_order: list[str],
                              workdir: Path | None,
                              ) -> list[str]:
    """Given a period_order coming from solve.realized_periods +
    period_timeset, add stochastic-branch periods read from
    ``solve_data/period__branch.csv`` (anchor → branch map), filtered
    to those that actually appear in the canonical ``period_in_use_set``
    or ``steps_in_use.csv`` (excludes realized-output-only branches).

    For each anchor in period_order, append all unique in-use branches
    whose anchor is the anchor — preserving anchor's relative position.
    """
    if workdir is None:
        return list(period_order)
    p = Path(workdir) / "solve_data" / "period__branch.csv"
    if not p.exists():
        return list(period_order)
    try:
        df = _read_csv_file(p)
    except Exception:
        return list(period_order)
    if df.height == 0:
        return list(period_order)
    in_use: set[str] = set()
    piu_path = Path(workdir) / "solve_data" / "period_in_use_set.csv"
    if piu_path.exists():
        try:
            piu = _read_csv_file(piu_path)
            if piu.height > 0:
                in_use = set(piu["period"].to_list())
        except Exception:
            in_use = set()
    df = df.rename({"period": "anchor", "branch": "br"})
    anchor_to_branches: dict[str, list[str]] = {}
    for row in df.iter_rows(named=True):
        anchor_to_branches.setdefault(row["anchor"], []).append(row["br"])
    out: list[str] = []
    seen: set[str] = set()
    for d in period_order:
        if d not in seen:
            out.append(d)
            seen.add(d)
        for br in anchor_to_branches.get(d, []):
            if br in seen:
                continue
            if in_use and br not in in_use:
                continue
            out.append(br)
            seen.add(br)
    return out


def _dt_period_active_steps_from_workdir(
    source: "InputSource",
    workdir: Path,
) -> dict | None:
    """Δ.12c-fix2 gap #5 — preprocessing-only-timeline fallback.

    Build ``per_period`` / ``period_order`` directly from the workdir's
    preprocessing artefacts (``solve_data/steps_in_use.csv`` +
    ``solve_data/period_in_use_set.csv``) when the source's
    ``solve.period_timeset`` filter for the active solve is empty.

    This handles **rolling-horizon sub-solves** and **nested-invest
    sub-solves** whose synthetic solve names (e.g.
    ``dispatch_fullYear_roll_roll_71``) aren't in the Spine DB — only
    the parent solve is.  flextool's preprocessing pipeline writes
    ``steps_in_use.csv`` per sub-solve; we leverage that CSV plus the
    Spine ``timeline.timestep_duration`` for the rank/timeline lookup.

    Returns ``None`` when the workdir doesn't have the required
    preprocessing artefacts (caller stays on the seed-CSV path).
    """
    sd = workdir / "solve_data"
    siu_path = sd / "steps_in_use.csv"
    if not siu_path.exists():
        return None
    try:
        siu = _read_csv_file(siu_path)
    except Exception:
        return None
    if siu.height == 0:
        return None
    period_col = next((c for c in ("period", "d") if c in siu.columns), None)
    step_col = next((c for c in ("step", "t", "time")
                       if c in siu.columns), None)
    if period_col is None or step_col is None:
        return None
    siu = siu.select(pl.col(period_col).alias("d"),
                       pl.col(step_col).alias("t")).unique()
    # Period order: prefer per_solve_sets.py's ``period_in_use_set.csv``
    # (canonical insertion order); fall back to insertion order in
    # steps_in_use.csv.
    period_order: list[str] = []
    seen: set[str] = set()
    piu_path = sd / "period_in_use_set.csv"
    if piu_path.exists():
        try:
            piu_df = _read_csv_file(piu_path)
            if piu_df.height > 0 and "period" in piu_df.columns:
                for d in piu_df["period"].to_list():
                    if d and d not in seen:
                        period_order.append(d)
                        seen.add(d)
        except Exception:
            pass
    for d in siu["d"].to_list():
        if d and d not in seen:
            period_order.append(d)
            seen.add(d)
    if not period_order:
        return None
    # Timeline lookup: per period, the timeline is whichever timeline
    # contains the period's first step (per the Spine
    # ``timeline.timestep_duration`` rank).  When multiple timelines
    # match (rare; only for ambiguous step naming) pick the first.
    tl_dur = _try_param(source, "timeline", "timestep_duration")
    if tl_dur is None:
        return None
    tl_step_col = next((c for c in ("t", "step", "timestep", "x")
                          if c in tl_dur.columns
                          and c not in ("name", "value")),
                         None)
    if tl_step_col is None:
        return None
    tl_lf = (tl_dur.lazy()
                    .select(pl.col("name").alias("timeline"),
                            pl.col(tl_step_col).alias("t"),
                            pl.col("value").cast(pl.Float64)
                                            .alias("step_duration"))
                    .sort("timeline", "t")
                    .with_columns(rank=pl.col("t").cum_count()
                                                  .over("timeline")
                                                  .cast(pl.Int64)))
    # Pick a single timeline per period via the first step of that period.
    # Mirrors flextool's preprocessing: a period maps to exactly one
    # timeset, which maps to exactly one timeline.  When the timeline
    # is ambiguous (multiple timelines contain the same step name —
    # shouldn't happen with t0001..t8760 naming), prefer the timeline
    # whose first matching step has the smallest rank (i.e. earliest
    # step in the timeline).
    siu_e = siu.lazy().join(tl_lf, on="t", how="inner").collect()
    if siu_e.height == 0:
        return None
    timeline_per_period: dict[str, str] = {}
    per_period: dict[str, list[tuple[str, int]]] = {}
    block_starts_per_period: dict[str, list[str]] = {}
    for d in period_order:
        rows = siu_e.filter(pl.col("d") == d)
        if rows.height == 0:
            per_period[d] = []
            block_starts_per_period[d] = []
            continue
        # When multiple timelines match, pick the timeline with the
        # smallest sum-of-ranks for this period (== smallest first step).
        if rows["timeline"].n_unique() > 1:
            best_tl = (rows.group_by("timeline")
                           .agg(pl.col("rank").min().alias("min_rk"))
                           .sort("min_rk")
                           .select("timeline").to_series()[0])
            rows = rows.filter(pl.col("timeline") == best_tl)
        rows_sorted = rows.sort("rank")
        timeline_per_period[d] = rows_sorted["timeline"][0]
        per_period[d] = list(zip(rows_sorted["t"].to_list(),
                                    rows_sorted["rank"].to_list()))
        # Block starts: indices where the rank gap to the previous step
        # is >1.  Mirrors ``make_period_block``'s detection rule.
        ts = rows_sorted["t"].to_list()
        rks = rows_sorted["rank"].to_list()
        block_firsts: list[str] = [ts[0]] if ts else []
        for j in range(1, len(rks)):
            if rks[j] - rks[j - 1] > 1:
                block_firsts.append(ts[j])
        block_starts_per_period[d] = block_firsts
    return dict(
        per_period=per_period,
        timeline_for_period=timeline_per_period,
        period_order=period_order,
        block_starts_per_period=block_starts_per_period,
    )


def _dt_period_active_steps(source: "InputSource",
                              active_solve: str,
                              workdir: Path | None = None,
                              ) -> dict | None:
    """Build the per-period active-step structure used by every storage
    block-algebra helper.

    Returns a dict ``{"per_period": {d -> [(t, global_rank), ...]},
    "timeline_for_period": {d -> timeline_name},
    "period_order": [d in dispatch order],
    "block_starts_per_period": {d -> [t_block_first ordered by rank]}}``,
    where ``global_rank`` is the timeline rank (cum_count over the
    timeline).  The block-first list is derived from
    ``timeset.timeset_duration`` block start_steps cross-joined with
    ``period_timeset`` for the active solve.

    Returns ``None`` when the active solve has no realized periods or any
    of the required Spine classes are absent.

    Δ.12c-fix2 gap #5 — when the source-side path returns no rows for the
    active solve (e.g. rolling-horizon / nested-invest synthetic solves
    whose names aren't in Spine), falls back to
    :func:`_dt_period_active_steps_from_workdir` which reads
    ``steps_in_use.csv`` directly.  The fallback uses the Spine
    ``timeline.timestep_duration`` for rank/timeline assignment.
    """
    p_ts = _try_param(source, "solve", "period_timeset")
    if p_ts is None:
        # Try preprocessing-only fallback before bailing.
        if workdir is not None:
            return _dt_period_active_steps_from_workdir(source, workdir)
        return None
    period_col = next((c for c in ("period", "x")
                        if c in p_ts.columns), None)
    if period_col is None:
        return None
    pt_eager_full = (p_ts
                       .filter(pl.col("name") == active_solve)
                       .rename({period_col: "d"})
                       .select("d", pl.col("value").alias("ts")))
    if pt_eager_full.height == 0:
        # Active solve isn't in the source's period_timeset (synthetic
        # rolling-horizon / nested-invest sub-solve).  Try preprocessing-
        # only fallback.
        if workdir is not None:
            wd_result = _dt_period_active_steps_from_workdir(source, workdir)
            if wd_result is not None:
                return wd_result
        # No fallback available — caller will bail.
        # (We continue with the empty pt_eager_full, which produces
        # an empty anchor_to_ts and the function returns None below.)
    # Anchor → timeset map for the active solve.
    anchor_to_ts: dict[str, str] = dict(
        zip(pt_eager_full["d"].to_list(),
              pt_eager_full["ts"].to_list()))
    # Expand to include branch periods (workdir-aware — period__branch
    # is preprocessing-only data, no Spine class in v25).  Filter on
    # period_in_use_set.csv when present to exclude realized-output-only
    # branches (e.g. period1_realized).
    in_use: set[str] = set()
    if workdir is not None:
        piu_path = Path(workdir) / "solve_data" / "period_in_use_set.csv"
        if piu_path.exists():
            try:
                piu_df = _read_csv_file(piu_path)
                if piu_df.height > 0:
                    in_use = set(piu_df["period"].to_list())
            except Exception:
                in_use = set()
        pb_path = Path(workdir) / "solve_data" / "period__branch.csv"
        if pb_path.exists():
            try:
                pbf = _read_csv_file(pb_path)
            except Exception:
                pbf = None
            if pbf is not None and pbf.height > 0:
                for row in pbf.iter_rows(named=True):
                    anchor = row["period"]
                    branch = row["branch"]
                    if in_use and branch not in in_use:
                        continue
                    if (anchor in anchor_to_ts
                            and branch not in anchor_to_ts):
                        anchor_to_ts[branch] = anchor_to_ts[anchor]
    if not anchor_to_ts:
        return None
    pt_lf = pl.LazyFrame({
        "d": list(anchor_to_ts.keys()),
        "ts": list(anchor_to_ts.values()),
    })
    ts_timeline = _try_param(source, "timeset", "timeline")
    if ts_timeline is None:
        return None
    ttl_lf = ts_timeline.lazy().select(pl.col("name").alias("ts"),
                                          pl.col("value").alias("timeline"))
    ts_dur = _try_param(source, "timeset", "timeset_duration")
    if ts_dur is None:
        return None
    step_col = next((c for c in ("t", "x", "step", "timestep")
                      if c in ts_dur.columns and c not in ("name", "value")),
                     None)
    if step_col is None:
        return None
    blocks_lf = (ts_dur.lazy()
                       .select(pl.col("name").alias("ts"),
                               pl.col(step_col).alias("start_step"),
                               pl.col("value").cast(pl.Int64).alias("count")))
    tl_dur = _try_param(source, "timeline", "timestep_duration")
    if tl_dur is None:
        return None
    tl_step_col = next((c for c in ("t", "step", "timestep", "x")
                          if c in tl_dur.columns and c not in ("name", "value")),
                         None)
    if tl_step_col is None:
        return None
    tl_lf = (tl_dur.lazy()
                    .select(pl.col("name").alias("timeline"),
                            pl.col(tl_step_col).alias("t"),
                            pl.col("value").cast(pl.Float64)
                                            .alias("step_duration"))
                    .sort("timeline", "t")
                    .with_columns(rank=pl.col("t").cum_count()
                                                  .over("timeline")
                                                  .cast(pl.Int64)))
    # Period order: prefer realized_periods.i ordering when present, else
    # fall back to period_timeset's natural order.  ``period_in_use_set``
    # lists ALL solve periods (realized + non-realized + branch siblings).
    realized = _try_param(source, "solve", "realized_periods")
    explicit_order: list[str] = []
    if realized is not None:
        realized_p = (realized
                       .filter(pl.col("name") == active_solve)
                       .sort("i"))
        explicit_order = realized_p["value"].to_list()
    pt_eager = pt_lf.select("d").unique().collect()
    pt_periods = pt_eager["d"].to_list()
    seen_o: set[str] = set(explicit_order)
    period_order: list[str] = list(explicit_order)
    for d in pt_periods:
        if d not in seen_o:
            period_order.append(d)
            seen_o.add(d)
    # Workdir-aware branch expansion + reorder by period_in_use_set
    # insertion order (canonical flextool active_time_list ordering).
    if workdir is not None:
        period_order = _expand_branch_periods(period_order, workdir)
        piu_path = Path(workdir) / "solve_data" / "period_in_use_set.csv"
        if piu_path.exists():
            try:
                piu_df = _read_csv_file(piu_path)
                if piu_df.height > 0:
                    canonical = piu_df["period"].to_list()
                    # Re-order period_order to match canonical, keeping
                    # only periods present in BOTH (and any extra
                    # period_order entries appended at the end for
                    # robustness — doesn't matter when canonical covers).
                    in_canonical = [p for p in canonical
                                       if p in seen_o
                                       or p in set(period_order)]
                    extras = [p for p in period_order
                                if p not in set(in_canonical)]
                    period_order = in_canonical + extras
            except Exception:
                pass
    if not period_order:
        return None
    realized_clean = pl.LazyFrame({"d": period_order})
    # Tie everything together to find each (d, t, rank).
    realized_with_ts = realized_clean.join(pt_lf, on="d", how="inner")
    pst = (realized_with_ts
            .join(ttl_lf, on="ts", how="inner")
            .join(blocks_lf, on="ts", how="inner"))
    starts = pst.join(
        tl_lf.select("timeline",
                       pl.col("t").alias("start_step"),
                       pl.col("rank").alias("start_rank")),
        on=["timeline", "start_step"], how="inner",
    )
    # Expand: for each (d, ts, timeline, start_step, count), emit each
    # rank in [start_rank, start_rank + count - 1].
    expanded = (starts
                  .join(tl_lf, on="timeline", how="inner")
                  .filter((pl.col("rank") >= pl.col("start_rank"))
                          & (pl.col("rank") < pl.col("start_rank")
                              + pl.col("count")))
                  .select("d", "t", "rank", "timeline", "start_step")
                  .unique())
    expanded_e = expanded.collect()
    if expanded_e.height == 0:
        return None
    timeline_per_period: dict[str, str] = {}
    per_period: dict[str, list[tuple[str, int]]] = {}
    block_starts_per_period: dict[str, list[str]] = {}
    # Group by period; within period, sort by rank to get (t, rank) order.
    for d in period_order:
        rows = expanded_e.filter(pl.col("d") == d)
        if rows.height == 0:
            per_period[d] = []
            block_starts_per_period[d] = []
            continue
        rows_sorted = rows.sort("rank")
        per_period[d] = list(zip(rows_sorted["t"].to_list(),
                                    rows_sorted["rank"].to_list()))
        timelines = rows_sorted["timeline"].unique().to_list()
        # Single timeline per period under the single-timeset-per-(solve,
        # period) rule of period_timeset.  Pick the first.
        timeline_per_period[d] = timelines[0]
        # Block starts: distinct start_step values, sorted by their rank
        # in the timeline (== rank of the first-occurrence row).
        first_occ = (rows_sorted
                       .group_by("start_step")
                       .agg(pl.col("rank").min().alias("rk"))
                       .sort("rk"))
        block_starts_per_period[d] = first_occ["start_step"].to_list()
    return dict(
        per_period=per_period,
        timeline_for_period=timeline_per_period,
        period_order=period_order,
        block_starts_per_period=block_starts_per_period,
    )


# ---------------------------------------------------------------------------
# §3.9.1 — dtttdt
# ---------------------------------------------------------------------------


def dtttdt_from_source(source: "InputSource",
                          active_solve: str | None,
                          workdir: Path | None = None,
                          ) -> pl.DataFrame | None:
    """Build the canonical ``(d, t, t_previous, t_previous_within_timeset,
    d_previous, t_previous_within_solve)`` dispatch-step lag frame.

    Mirrors ``timeline_config.make_step_jump`` (default-block path):

      * Cyclic within solve: first step of the first realized period
        wraps to the last step of the last realized period.
      * Cyclic within timeset: the ``t_previous_within_timeset`` column
        is the block's last step for first-of-period rows, else
        identical to ``t_previous`` for jump=1 interior steps; for
        jump>1 transitions (block boundaries) it pins on the prior
        block's last step.
      * Within-period predecessor: previous step in the same period
        cyclically (first wraps to block_last of the period).
    """
    info = _dt_period_active_steps(source, active_solve, workdir)
    if info is None:
        return None
    per_period = info["per_period"]
    period_order = info["period_order"]
    if not period_order:
        return None
    # Per-period anchor lookup: branches whose anchor != self use a
    # per-branch self-loop for the period-wrap predecessor.  Read
    # period__branch.csv (workdir-aware) to discover which periods are
    # branches off a non-self anchor.
    branch_anchor: dict[str, str] = {}
    if workdir is not None:
        pb_path = Path(workdir) / "solve_data" / "period__branch.csv"
        if pb_path.exists():
            try:
                pbf = _read_csv_file(pb_path)
            except Exception:
                pbf = None
            if pbf is not None and pbf.height > 0:
                pb_rows = list(zip(pbf["period"].to_list(),
                                      pbf["branch"].to_list()))
                anchors_with_self: set[str] = set()
                for a, b in pb_rows:
                    if a == b:
                        anchors_with_self.add(a)
                for a, b in pb_rows:
                    if b != a and a in anchors_with_self:
                        branch_anchor[b] = a
    out_rows: list[tuple[str, str, str, str, str, str]] = []
    n_periods = len(period_order)
    # Find within-solve predecessor for first-of-period rows.
    for pi, period in enumerate(period_order):
        steps = per_period.get(period) or []
        if not steps:
            continue
        # Determine the predecessor period for the j=0 wrap row.
        # Mirrors ``timeline_config.make_step_jump``:
        #   * branch (non-anchor) period → self-wrap;
        #   * first anchor period (== first_period_name in flextool's
        #     reversed walk) → wrap to last_period_name (last in
        #     active_time_list iteration order, == period_order[-1]);
        #   * non-first anchor period → wrap to the period at
        #     period_order[pi - 1].
        if period in branch_anchor:
            wrap_period = period
        else:
            if pi == 0:
                wrap_period = period_order[-1]
            else:
                wrap_period = period_order[pi - 1]
        prev_solve_steps = per_period.get(wrap_period) or []
        prev_solve_last_t = prev_solve_steps[-1][0] if prev_solve_steps \
                            else steps[-1][0]
        # Within-period: cyclic over current period.
        block_last_t = steps[-1][0]
        n = len(steps)
        # Track running block_last_t as we walk forward from j=0 to j=n-1.
        # In flextool's reversed loop, block_last initialises to len-1 and
        # decreases on jump>1 transitions.  Equivalent forward pass: the
        # block_last for index j is the index of the next jump>1 boundary
        # to the right (or the last index n-1 if none).
        # Pre-compute for each j, the block_last index (forward sweep).
        block_last_idx_for: list[int] = [n - 1] * n
        # Find boundaries: indices where ranks[j] - ranks[j-1] > 1.
        # block_last_idx_for[k] = block_last for indices in current block.
        # We iterate from the right, tracking the current block's last index.
        cur_last = n - 1
        for k in range(n - 1, 0, -1):
            t_k, rk_k = steps[k]
            t_prev, rk_prev = steps[k - 1]
            block_last_idx_for[k] = cur_last
            if rk_k - rk_prev > 1:
                # Boundary — next k's block_last is k-1.
                cur_last = k - 1
        block_last_idx_for[0] = cur_last
        for j in range(n):
            t_cur, rk_cur = steps[j]
            if j == 0:
                # First step of the period — cross-period wrap.  Use
                # wrap_period computed above (handles branch self-wrap
                # vs anchor cyclic-chain).
                prev_p = wrap_period
                prev_p_steps = per_period.get(prev_p) or []
                prev_t = prev_p_steps[-1][0] if prev_p_steps else t_cur
                # Block's last step in current period — specifically the
                # last step of the block CONTAINING the current step
                # (block_last_idx_for[0]).
                bli0 = block_last_idx_for[0]
                prev_within_ts = steps[bli0][0]
                prev_within_solve = prev_solve_last_t
                d_prev = prev_p
            else:
                t_prev, rk_prev = steps[j - 1]
                prev_t = t_prev
                jump = rk_cur - rk_prev
                if jump > 1:
                    # Block boundary — pin within_ts to the last step of
                    # the block CONTAINING the current step (j), per
                    # ``make_step_jump``'s reverse-walk semantics.  The
                    # outgoing-block last index would be (j-1) but the
                    # canonical algorithm pins it to active_time[block_last]
                    # tracked in reverse-iteration order which equals the
                    # last index of the block containing j.
                    bli = block_last_idx_for[j]
                    prev_within_ts = steps[bli][0]
                else:
                    prev_within_ts = t_prev
                d_prev = period
                prev_within_solve = t_prev
            out_rows.append((period, t_cur, prev_t, prev_within_ts,
                              d_prev, prev_within_solve))
    if not out_rows:
        return None
    df = pl.DataFrame(
        out_rows,
        schema=["d", "t", "t_previous", "t_previous_within_timeset",
                  "d_previous", "t_previous_within_solve"],
        orient="row",
    )
    return df.sort("d", "t")


# ---------------------------------------------------------------------------
# §3.9.3 — period_block / period_block_succ / period_block_time
# ---------------------------------------------------------------------------


def period_block_family_from_source(source: "InputSource",
                                       active_solve: str | None,
                                       workdir: Path | None = None,
                                       ) -> dict | None:
    """Build the ``period_block_set`` / ``period_block_succ`` /
    ``period_block_time`` frames mirrored on
    ``timeline_config.make_period_block`` AND the multi-resolution
    synthesis branch in ``input.py:1985-2126``.

    Two paths:

      1. **Default (timeset-block detection)**: per realized period,
         sort active steps by rank.  A new block starts when the rank
         gap to the previous step exceeds 1.  This matches
         ``make_period_block`` exactly.

      2. **Multi-resolution synthesis**: when the workdir's
         ``entity_block.csv`` assigns coarse blocks (sd>1) to one or
         more nodeBalance entities AND multiple distinct blocks are in
         use, REPLACE the block decomposition with the daily-aggregated
         coarse-block structure: per-period block_firsts come from the
         coarse block's step list in ``block_step_duration.csv``;
         ``period_block_time`` comes from ``overlap_set.csv`` (rows
         where ``b_coarse=coarse, b_fine=default``).

    Returns dict with the three frames or None when source insufficient.
    """
    info = _dt_period_active_steps(source, active_solve, workdir)
    if info is None:
        return None
    per_period = info["per_period"]
    period_order = info["period_order"]
    pb_rows: list[tuple[str, str]] = []         # (d, b_first)
    pbs_rows: list[tuple[str, str, str]] = []   # (d, b_first, b_next)
    pbt_rows: list[tuple[str, str, str]] = []   # (d, b_first, t)
    for period in period_order:
        steps = per_period.get(period) or []
        if not steps:
            continue
        block_firsts: list[str] = [steps[0][0]]
        cur_b = steps[0][0]
        for j, (t, rk) in enumerate(steps):
            if j > 0 and rk - steps[j - 1][1] > 1:
                cur_b = t
                block_firsts.append(cur_b)
            pbt_rows.append((period, cur_b, t))
        for i, b in enumerate(block_firsts):
            pb_rows.append((period, b))
            nxt = block_firsts[(i + 1) % len(block_firsts)]
            pbs_rows.append((period, b, nxt))
    period_block = (pl.DataFrame(pb_rows, schema=["d", "b_first"],
                                    orient="row").unique()
                    if pb_rows else None)
    period_block_succ = (pl.DataFrame(pbs_rows,
                                          schema=["d", "b_first", "b_next"],
                                          orient="row")
                          if pbs_rows else None)
    period_block_time = (pl.DataFrame(pbt_rows,
                                          schema=["d", "b_first", "t"],
                                          orient="row").unique()
                          if pbt_rows else None)

    # Multi-resolution synthesis (input.py:1985-2126 mirror).  Δ.2:
    # consume frames via BlockLayout.load_from_solve_data instead of
    # re-reading the CSVs at the call site.
    if workdir is not None:
        from flextool.engine_polars._block_layout import BlockLayout
        sd = Path(workdir) / "solve_data"
        bl = BlockLayout.load_from_solve_data(sd)
        eb = bl.entity_block_frame
        bsd = bl.block_step_duration_frame
        if eb.height > 0 and bsd.height > 0:
            distinct_blocks = bsd["block"].unique().to_list()
            if len(distinct_blocks) >= 2:
                coarse = bl.coarse_blocks(threshold=1.0)
                if coarse:
                    non_default_nodes = eb.filter(
                        pl.col("block").is_in(coarse))
                    if non_default_nodes.height > 0:
                        coarse_use = non_default_nodes["block"] \
                                      .unique().to_list()
                        bsd_c = bsd.filter(
                            pl.col("block").is_in(coarse_use))
                        new_pb = (bsd_c
                            .rename({"period": "d", "step": "b_first"})
                            .select("d", "b_first").unique())
                        # period_block_succ: cyclic per (block, period).
                        succ_rows: list[tuple[str, str, str]] = []
                        bsd_sorted = bsd_c.rename(
                            {"period": "d", "step": "b_first"}
                            ).sort("block", "d", "b_first")
                        for (blk, dval), grp in bsd_sorted.group_by(
                                ["block", "d"], maintain_order=True):
                            bfs = grp["b_first"].to_list()
                            n = len(bfs)
                            for i in range(n):
                                succ_rows.append(
                                    (dval, bfs[i], bfs[(i + 1) % n]))
                        new_pbs = (pl.DataFrame(
                            succ_rows,
                            schema=["d", "b_first", "b_next"],
                            orient="row")
                            if succ_rows else None)
                        new_pbt = None
                        ov = bl.overlap_set_frame
                        if ov.height > 0:
                            ov = ov.rename({
                                "period": "d",
                                "block_coarse": "b",
                                "step_coarse": "b_first",
                                "block_fine": "b_fine",
                                "step_fine": "t",
                            })
                            ov_keep = ov.filter(
                                pl.col("b").is_in(coarse_use)
                                & (pl.col("b_fine") == "default"))
                            if ov_keep.height > 0:
                                new_pbt = ov_keep.select(
                                    "d", "b_first", "t").unique()
                        if new_pb is not None and new_pb.height > 0:
                            period_block = new_pb
                        if new_pbs is not None and new_pbs.height > 0:
                            period_block_succ = new_pbs
                        if new_pbt is not None and new_pbt.height > 0:
                            period_block_time = new_pbt

    return dict(
        period_block=period_block,
        period_block_succ=period_block_succ,
        period_block_time=period_block_time,
    )


# ---------------------------------------------------------------------------
# §3.9.2 — nodeStateBlock multi-resolution synthesis
# ---------------------------------------------------------------------------


def _node_storage_binding_method_with_fallback(source: "InputSource"
                                                       ) -> pl.DataFrame | None:
    """Per-node storage binding method with the
    ``method_with_fallback_sets.write_node_storage_binding_method``
    fallback rule applied.

    Mirrors flextool's preprocessing: explicit rows are kept verbatim;
    every node lacking an explicit row gets the default
    ``bind_forward_only``.  Returns ``[n, method]`` frame.
    """
    nodes = _try_entities(source, "node")
    if nodes is None or nodes.height == 0:
        return None
    explicit = _try_param(source, "node", "storage_binding_method")
    if explicit is None:
        explicit_rows = pl.DataFrame(schema={"n": pl.Utf8,
                                              "method": pl.Utf8})
    else:
        explicit_rows = (explicit
            .rename({"name": "n", "value": "method"})
            .select("n", "method"))
    explicit_n = set(explicit_rows["n"].to_list())
    fallback_n = [n for n in nodes["name"].to_list() if n not in explicit_n]
    fb = pl.DataFrame({"n": fallback_n,
                         "method": ["bind_forward_only"] * len(fallback_n)}) \
            if fallback_n else pl.DataFrame(
                schema={"n": pl.Utf8, "method": pl.Utf8})
    out = pl.concat([explicit_rows, fb]).unique().sort("n", "method")
    return out


def _coarse_blocks_from_source(source: "InputSource"
                                  ) -> tuple[set[str], dict[str, str]] | None:
    """Identify *coarse* blocks (block_step_duration > 1) and the
    per-entity block assignment from the source.

    Returns ``(coarse_blocks, entity_block)``: a set of coarse block
    names and a dict ``entity -> block_name`` for entities whose block
    assignment is non-default.  Returns ``None`` when the schema is
    incomplete.

    Currently the multi-resolution synthesis is driven primarily by
    flextool's preprocessing CSVs (``entity_block.csv`` /
    ``block_step_duration.csv`` / ``overlap_set.csv``) that are NOT yet
    on the Spine schema in a way that flexpy's Param helpers can read.
    The Spine layer exposes ``solve.contains_solves`` and
    ``timeset.timeset_duration``; the per-block (``daily_group``,
    ``hourly_group``, ``default``) infrastructure is the responsibility
    of ``preprocessing/blocks.py`` running on the canonical
    ``solve_data/`` CSVs.

    For the DB-direct port, the synthesis ALSO falls back to reading
    the on-disk CSVs from the workdir — see the wrapper
    ``nodeStateBlock_from_source`` which is workdir-aware.  This
    function returns only what's discoverable on the Spine schema.
    """
    # Probe for any source-level info about blocks.  In current schemas
    # (v25-v28) we don't have ``block`` as a queryable entity class with
    # ``step_duration`` as a parameter — return None.
    return None


def nodeStateBlock_from_source(source: "InputSource",
                                  workdir: Path | None,
                                  ) -> pl.DataFrame | None:
    """Synthesise the ``nodeStateBlock`` set per audit §3.9.2.

    Two contributing branches (matching ``input.py:_load_storage``):

      1. **Explicit method**: nodes whose
         ``storage_binding_method == 'bind_intraperiod_blocks'`` join
         the set directly.
      2. **Multi-resolution synthesis**: when the workdir's
         ``entity_block.csv`` assigns *coarse* blocks (any
         ``block_step_duration > 1``) AND the scenario has multiple
         distinct blocks AND the entity is on the nodeBalance side, that
         entity is folded into ``nodeStateBlock`` so the daily-aggregation
         balance fires.

    The flextool preprocessing emits the ``entity_block.csv``,
    ``block_step_duration.csv`` and ``nodeStateBlock.csv`` to
    ``solve_data/``; the Spine schema in v25-v28 does not yet expose
    these as queryable entity classes.  We therefore read the
    workdir-side CSVs as the source of truth for the synthesis.

    Returns ``None`` if neither branch yields any rows (most fixtures).
    """
    rows: list[str] = []
    # Branch 1: explicit bind_intraperiod_blocks method on Spine schema.
    sbm = _try_param(source, "node", "storage_binding_method")
    if sbm is not None:
        intraperiod = (sbm.lazy()
                          .filter(pl.col("value")
                                  == "bind_intraperiod_blocks")
                          .select(pl.col("name").alias("n"))
                          .collect())
        if intraperiod.height > 0:
            rows.extend(intraperiod["n"].to_list())
    # Branch 2: multi-resolution synthesis via in-memory BlockLayout
    # (Δ.2: consolidated through ``BlockLayout.load_from_solve_data``).
    if workdir is not None:
        from flextool.engine_polars._block_layout import BlockLayout
        sd = Path(workdir) / "solve_data"
        bl = BlockLayout.load_from_solve_data(sd)
        eb = bl.entity_block_frame
        bsd = bl.block_step_duration_frame
        if eb.height > 0 and bsd.height > 0:
            distinct_blocks = bsd["block"].unique().to_list()
            if len(distinct_blocks) >= 2:
                coarse = bl.coarse_blocks(threshold=1.0)
                if coarse:
                    # Filter to nodeBalance nodes — read node entities
                    # from source (we don't have nodeBalance set on
                    # Spine, but every node-entity is an
                    # entity_block candidate here).
                    nodes = _try_entities(source, "node")
                    if nodes is not None:
                        node_set = set(nodes["name"].to_list())
                        picked = (eb
                                   .filter(pl.col("block").is_in(coarse))
                                   .filter(pl.col("entity").is_in(node_set))
                                   ["entity"].unique().to_list())
                        rows.extend(picked)
    if not rows:
        return None
    out = (pl.DataFrame({"n": sorted(set(rows))})
              .unique().sort("n"))
    if out.height == 0:
        return None
    return out


# ---------------------------------------------------------------------------
# §3.9.4 — arc_sink_block_dt / arc_source_block_dt + weights
# ---------------------------------------------------------------------------


def arc_block_dt_from_source(source: "InputSource",
                                 workdir: Path | None,
                                 nodeStateBlock_df: pl.DataFrame | None,
                                 period_block_time_df: pl.DataFrame | None,
                                 pss: pl.DataFrame | None,
                                 ) -> dict | None:
    """Build per-arc daily-block aggregation frames.

    For each arc (p, source, sink) participating in nodeStateBlock on its
    sink (or source) side, project to ``(p, source, sink, d, b_first, t,
    weight)`` with weight = block_step_duration on the arc-side block.

    Inputs (workdir-side CSVs as source-of-truth, since block schema is
    on the preprocessing side):

      * ``solve_data/process_side_block.csv`` — (process, side, block).
      * ``solve_data/block_step_duration.csv`` — (block, period, step,
        step_duration).
      * ``period_block_time`` — (d, b_first, t) computed Param.
      * ``pss`` — (p, source, sink) topology.
      * ``nodeStateBlock_df`` — set of nodes that pull this aggregation.

    Returns dict with ``arc_sink_block_dt``, ``arc_source_block_dt``,
    ``p_arc_sink_weight``, ``p_arc_source_weight``.  Any of these may be
    ``None`` when the corresponding side has no rows.
    """
    if (workdir is None or nodeStateBlock_df is None
            or period_block_time_df is None or pss is None
            or pss.height == 0 or nodeStateBlock_df.height == 0
            or period_block_time_df.height == 0):
        return None
    # Δ.2: consume block frames via BlockLayout instead of reading
    # process_side_block.csv + block_step_duration.csv directly.
    from flextool.engine_polars._block_layout import BlockLayout
    sd = Path(workdir) / "solve_data"
    bl = BlockLayout.load_from_solve_data(sd)
    if (bl.process_side_block_frame.height == 0
            or bl.block_step_duration_frame.height == 0):
        return None
    psb = bl.process_side_block_frame.rename(
        {"process": "p", "block": "b_f"})
    bsd_arc = bl.block_step_duration_frame.rename(
        {"block": "b_f", "period": "d", "step": "t",
          "step_duration": "weight"})
    nsb_set = nodeStateBlock_df["n"].unique()
    pbt = period_block_time_df

    out: dict[str, object | None] = {
        "arc_sink_block_dt": None,
        "arc_source_block_dt": None,
        "p_arc_sink_weight": None,
        "p_arc_source_weight": None,
    }

    psb_sink = psb.filter(pl.col("side") == "sink").select("p", "b_f")
    sink_arcs = (pss
        .filter(pl.col("sink").is_in(nsb_set))
        .join(psb_sink, on="p", how="inner"))
    if sink_arcs.height > 0:
        ab = (sink_arcs
            .join(bsd_arc, on="b_f", how="inner")
            .join(pbt, on=["d", "t"], how="inner")
            .select("p", "source", "sink", "d", "b_first", "t", "weight")
            .unique())
        if ab.height > 0:
            out["arc_sink_block_dt"] = ab
            wf = (ab.select("p", "source", "sink", "d", "t", "weight")
                     .unique()
                     .rename({"weight": "value"}))
            out["p_arc_sink_weight"] = Param(
                ("p", "source", "sink", "d", "t"), wf)

    psb_src = psb.filter(pl.col("side") == "source").select("p", "b_f")
    src_arcs = (pss
        .filter(pl.col("source").is_in(nsb_set))
        .join(psb_src, on="p", how="inner"))
    if src_arcs.height > 0:
        ab = (src_arcs
            .join(bsd_arc, on="b_f", how="inner")
            .join(pbt, on=["d", "t"], how="inner")
            .select("p", "source", "sink", "d", "b_first", "t", "weight")
            .unique())
        if ab.height > 0:
            out["arc_source_block_dt"] = ab
            wf = (ab.select("p", "source", "sink", "d", "t", "weight")
                     .unique()
                     .rename({"weight": "value"}))
            out["p_arc_source_weight"] = Param(
                ("p", "source", "sink", "d", "t"), wf)
    return out


# ---------------------------------------------------------------------------
# §3.9.5 — p_state_existing_capacity / p_state_upper
# ---------------------------------------------------------------------------


def p_state_existing_capacity_from_source(source: "InputSource",
                                                active_solve: str | None,
                                                nodeState_df: pl.DataFrame
                                                  | None,
                                                workdir: Path | None = None,
                                                ) -> "Param | None":
    """``p_state_existing_capacity`` per (n, d) — node existing capacity
    restricted to nodes carrying state (``nodeState`` set).

    Algorithm: take ``p_entity_all_existing`` for nodes only, restrict to
    nodes in ``nodeState_df``.  Falls back to ``node.existing`` directly
    when the chained Param isn't available.  Broadcasts onto the full
    period set including stochastic-branch siblings (workdir-aware).
    """
    if nodeState_df is None or nodeState_df.height == 0:
        return None
    state_n = set(nodeState_df["n"].to_list())
    # Full period set (realized + invest + stochastic-branch siblings).
    piu_base = _period_in_use_set(source, active_solve) if active_solve \
                else []
    full_periods = _expand_branch_periods(piu_base, workdir)
    pae = p_entity_all_existing_from_source(source, active_solve, workdir)
    if pae is not None and pae.frame.height > 0:
        base = (pae.frame.lazy()
                  .rename({"e": "n"})
                  .filter(pl.col("n").is_in(list(state_n)))
                  .select("n", "d", "value")
                  .collect())
        # Broadcast each (n, d) to its branch siblings sharing the
        # anchor's existing value.
        if full_periods and len(full_periods) > base["d"].n_unique():
            anchor_branches = _expand_branch_periods(
                base["d"].unique().to_list(), workdir)
            # Build (anchor → list of d) mapping
            extra_rows: list[tuple[str, str, float]] = []
            covered = set(base["d"].to_list())
            for d in anchor_branches:
                if d in covered:
                    continue
                # Find anchor whose row to copy: any anchor it expanded
                # from.  Walk period__branch.csv to find anchor mapping.
                anchor_for_d = _branch_anchor(d, workdir)
                if anchor_for_d is None:
                    continue
                rows_for_anchor = base.filter(
                    pl.col("d") == anchor_for_d)
                for r in rows_for_anchor.iter_rows(named=True):
                    extra_rows.append((r["n"], d, r["value"]))
            if extra_rows:
                add = pl.DataFrame(
                    extra_rows, schema=["n", "d", "value"], orient="row")
                base = pl.concat([base, add])
        df = base.sort("n", "d")
        if df.height > 0:
            return Param(("n", "d"), df)
    # Fallback: read node.existing directly.
    ex = _try_param(source, "node", "existing")
    if ex is None or ex.height == 0:
        return None
    cols = ex.columns
    if "period" in cols:
        df = (ex.lazy()
                .rename({"name": "n", "period": "d"})
                .filter(pl.col("n").is_in(list(state_n)))
                .select("n", "d", pl.col("value").cast(pl.Float64))
                .sort("n", "d")
                .collect())
    else:
        if not full_periods:
            return None
        df = (ex.lazy()
                .rename({"name": "n"})
                .filter(pl.col("n").is_in(list(state_n)))
                .select("n", pl.col("value").cast(pl.Float64))
                .join(pl.LazyFrame({"d": full_periods}), how="cross")
                .select("n", "d", "value")
                .sort("n", "d")
                .collect())
    if df.height == 0:
        return None
    return Param(("n", "d"), df)


def _branch_anchor(branch: str, workdir: Path | None) -> str | None:
    """Return the anchor period whose ``period__branch.csv`` row maps
    to *branch*.  None if no mapping found.
    """
    if workdir is None:
        return None
    p = Path(workdir) / "solve_data" / "period__branch.csv"
    if not p.exists():
        return None
    try:
        df = _read_csv_file(p)
    except Exception:
        return None
    if df.height == 0:
        return None
    cand = df.filter(pl.col("branch") == branch)
    if cand.height > 0:
        return cand["period"][0]
    return None


def _node_unitsize_lf(source: "InputSource") -> pl.LazyFrame:
    """Per-node unitsize lazy frame following the flextool preprocess
    formula (entity_period_calc_params.py:186-202): use
    ``virtual_unitsize`` if explicitly set non-zero, else ``existing``
    if explicitly set non-zero, else 1000.0.

    Uses :func:`_try_param_explicit` so default-broadcast rows from
    SpineDbReader are suppressed — only entities with a parameter_value
    row count as "explicitly set".  This mirrors flextool's
    ``p_node.get(name, None)`` semantic: an absent row is *not*
    equivalent to a row with the schema default value.

    Returns a lazy frame with ``[n, us]`` for every node entity.
    """
    us = _try_param_explicit(source, "node", "virtual_unitsize")
    ex = _try_param_explicit(source, "node", "existing")
    nodes = _try_entities(source, "node")
    if nodes is None or nodes.height == 0:
        return pl.LazyFrame(schema={"n": pl.Utf8, "us": pl.Float64})
    base = nodes.lazy().select(pl.col("name").alias("n"))
    if us is not None and us.height > 0:
        us_lf = us.lazy().select(pl.col("name").alias("n"),
                                    pl.col("value").cast(pl.Float64)
                                                     .alias("vu"))
        base = base.join(us_lf, on="n", how="left")
    else:
        base = base.with_columns(vu=pl.lit(None, dtype=pl.Float64))
    # Existing has no default (schema-None policy); rows present == explicit.
    if ex is not None and ex.height > 0:
        ex_cols = ex.columns
        if "period" in ex_cols:
            ex_lf = (ex.lazy()
                       .select(pl.col("name").alias("n"),
                                 pl.col("value").cast(pl.Float64))
                       .group_by("n")
                       .agg(pl.col("value").max().alias("ex")))
        else:
            ex_lf = ex.lazy().select(pl.col("name").alias("n"),
                                       pl.col("value").cast(pl.Float64)
                                                        .alias("ex"))
        base = base.join(ex_lf, on="n", how="left")
    else:
        base = base.with_columns(ex=pl.lit(None, dtype=pl.Float64))
    # Apply formula: virtual_unitsize (explicit non-zero)
    #                OR existing (explicit non-zero)
    #                OR 1000.
    base = base.with_columns(
        us=pl.when(pl.col("vu").fill_null(0.0) != 0.0)
             .then(pl.col("vu"))
             .when(pl.col("ex").fill_null(0.0) != 0.0)
             .then(pl.col("ex"))
             .otherwise(pl.lit(1000.0))
    ).select("n", "us")
    return base


def p_state_upper_from_source(source: "InputSource",
                                  state_existing_capacity: "Param | None",
                                  nodeState_df: pl.DataFrame | None,
                                  ) -> "Param | None":
    """``p_state_upper[n, d] = state_existing_capacity[n, d] /
    p_entity_unitsize[n]``, with the unitsize fallback chain
    ``virtual_unitsize OR existing OR 1000``.  Mirrors
    ``entity_period_calc_params.py:186-202`` and ``input.py:1755-1772``.
    """
    if (state_existing_capacity is None
            or state_existing_capacity.frame.height == 0
            or nodeState_df is None or nodeState_df.height == 0):
        return None
    cap = state_existing_capacity.frame
    us_lf = _node_unitsize_lf(source)
    df = (cap.lazy()
              .join(us_lf, on="n", how="left")
              .with_columns(us=pl.col("us").fill_null(1000.0))
              .with_columns(value=pl.col("value").cast(pl.Float64)
                                                   / pl.col("us"))
              .select("n", "d", "value")
              .sort("n", "d")
              .collect())
    if df.height == 0:
        return None
    return Param(("n", "d"), df)


# ---------------------------------------------------------------------------
# §3.9.6 — storage_use_reference_value (multi-method exclusion chain)
# ---------------------------------------------------------------------------


def storage_use_reference_value_from_source(source: "InputSource",
                                                  nodeStateBlock_df: pl.DataFrame
                                                    | None,
                                                  ) -> pl.DataFrame | None:
    """``storage_use_reference_value`` — nodes with horizon method
    ``use_reference_value`` after multi-method exclusion.

    Mirror of input.py:1880-1899.  Filter
    ``node.storage_solve_horizon_method == 'use_reference_value'``, then
    anti-join against:

      * ``storage_start_end_method ∈ {fix_end, fix_start_end}``
      * ``storage_binding_method ∈ {bind_within_solve, bind_within_timeset,
                                      bind_intraperiod_blocks}``
      * ``nodeStateBlock`` (synthesised + explicit).
    """
    sshm = _try_param(source, "node", "storage_solve_horizon_method")
    if sshm is None:
        return None
    base = (sshm.lazy()
                 .filter(pl.col("value") == "use_reference_value")
                 .select(pl.col("name").alias("n"))
                 .unique()
                 .collect())
    if base.height == 0:
        return None
    # Exclusion sets.
    sse = _try_param(source, "node", "storage_start_end_method")
    excl_lists: list[list[str]] = []
    if sse is not None:
        e = (sse.lazy()
                 .filter(pl.col("value").is_in(["fix_end", "fix_start_end"]))
                 .select(pl.col("name").alias("n"))
                 .collect())
        if e.height > 0:
            excl_lists.append(e["n"].to_list())
    sbm_full = _node_storage_binding_method_with_fallback(source)
    if sbm_full is not None:
        b = (sbm_full.lazy()
                  .filter(pl.col("method").is_in([
                      "bind_within_solve", "bind_within_timeset",
                      "bind_intraperiod_blocks"]))
                  .select("n")
                  .collect())
        if b.height > 0:
            excl_lists.append(b["n"].to_list())
    if nodeStateBlock_df is not None and nodeStateBlock_df.height > 0:
        excl_lists.append(nodeStateBlock_df["n"].to_list())
    excl = set()
    for lst in excl_lists:
        excl.update(lst)
    out = base.filter(~pl.col("n").is_in(list(excl)))
    if out.height == 0:
        return None
    return out.unique().sort("n")


# ---------------------------------------------------------------------------
# §3.9.7 — p_roll_continue_state / p_fix_storage_quantity (rolling handoff)
# ---------------------------------------------------------------------------


def p_roll_continue_state_from_workdir(workdir: Path | None
                                            ) -> "Param | None":
    """Read the rolling-handoff ``p_roll_continue_state`` from the
    on-disk CSV (written by a prior solve via
    ``fn_p_roll_continue_state``).  No Spine source — pure handoff
    artefact per audit §3.9.7.
    """
    if workdir is None:
        return None
    p = Path(workdir) / "solve_data" / "p_roll_continue_state.csv"
    if not p.exists():
        return None
    df = _read_csv_file(p)
    df.columns = [c.strip() for c in df.columns]
    if df.height == 0:
        return None
    df = (df.rename({"node": "n", "p_roll_continue_state": "value"})
            .with_columns(value=pl.col("value").cast(pl.Float64))
            .select("n", "value"))
    return Param(("n",), df)


def p_fix_storage_quantity_from_workdir(workdir: Path | None
                                              ) -> "Param | None":
    """Read the rolling-handoff ``p_fix_storage_quantity`` from
    ``solve_data/fix_storage_quantity.csv``.
    """
    if workdir is None:
        return None
    p = Path(workdir) / "solve_data" / "fix_storage_quantity.csv"
    if not p.exists():
        return None
    df = _read_csv_file(p)
    if df.height == 0:
        return None
    df = (df.rename({"period": "d", "step": "t", "node": "n",
                       "p_fix_storage_quantity": "value"})
            .with_columns(value=pl.col("value").cast(pl.Float64))
            .select("n", "d", "t", "value"))
    return Param(("n", "d", "t"), df)


# ---------------------------------------------------------------------------
# §3.9.8 — dtt_timeline_matching / period_branch
# ---------------------------------------------------------------------------


def dtt_timeline_matching_from_workdir(workdir: Path | None
                                              ) -> pl.DataFrame | None:
    """Per-solve timeline matching map ``(d, t, t_upper)`` between a
    sub-solve's fine timesteps and the upper-level coarse timesteps.
    Read from ``solve_data/timeline_matching_map.csv``.
    """
    if workdir is None:
        return None
    p = Path(workdir) / "solve_data" / "timeline_matching_map.csv"
    if not p.exists():
        return None
    df = _read_csv_file(p)
    if df.height == 0:
        return None
    return (df
            .rename({"period": "d", "step": "t", "upper_step": "t_upper"})
            .select("d", "t", "t_upper")
            .unique())


def period_branch_from_source(source: "InputSource",
                                workdir: Path | None,
                                ) -> pl.DataFrame | None:
    """``period_branch`` rolling-handoff helper — (d_upper, d) anchor →
    branch map per audit §3.9.8.

    Spine source: ``period__branch`` 1-dim entity class is currently a
    flat name list with no values; the canonical per-solve mapping is
    written by ``solve_writers`` to ``solve_data/period__branch.csv``
    with columns ``(period, branch)``.  Read that file when present
    (workdir-aware path).
    """
    if workdir is not None:
        p = Path(workdir) / "solve_data" / "period__branch.csv"
        if p.exists():
            df = _read_csv_file(p)
            if df.height > 0:
                return (df
                        .rename({"period": "d", "branch": "d_upper"})
                        .select("d_upper", "d")
                        .unique())
    return None


# ---------------------------------------------------------------------------
# §3.9 helper compose: dtttdt-derived auxiliary frames
# ---------------------------------------------------------------------------


def dtttdt_forward_only_from_dtttdt(dtttdt: pl.DataFrame | None
                                          ) -> pl.DataFrame | None:
    """``dtttdt`` sorted by (d, t), .slice(1) — drops the first wrap row
    so bind_forward_only nodes don't carry a state-change term across
    the very first step of the first period (mod:2188).
    """
    if dtttdt is None or dtttdt.height == 0:
        return None
    out = dtttdt.sort("d", "t").slice(1)
    return out if out.height > 0 else None


def dtttdt_block_interior_from_dtttdt(dtttdt: pl.DataFrame | None,
                                          period_block_time: pl.DataFrame
                                              | None = None,
                                          ) -> pl.DataFrame | None:
    """Interior-of-block dtttdt rows.

    Two paths matching input.py's branching:

      1. **Default** (timeset-block decomposition only): keep dtttdt
         rows where ``t_previous_within_timeset == t_previous``
         (jump=1 interior, not a boundary).  Mirrors input.py:1959-1968.

      2. **Synthesised (multi-resolution)**: when *period_block_time*
         is the synthesised daily-block table (multiple b_first per
         period), rebuild block_interior from it: for each (d, b_first),
         consecutive sorted t's give (d, t, t_previous=t_prev) — i.e.
         intra-day predecessor pairs.  Mirrors input.py:2076-2092.

    The caller distinguishes by passing the (possibly synthesised)
    ``period_block_time`` frame.  When it has multiple b_first values
    per period, branch 2 fires.
    """
    if dtttdt is None or dtttdt.height == 0:
        return None
    multi_res = False
    if period_block_time is not None and period_block_time.height > 0:
        # Detect multi-resolution: more than one distinct b_first per d.
        nb = (period_block_time
              .group_by("d")
              .agg(pl.col("b_first").n_unique().alias("nb"))
              ["nb"].max())
        if nb is not None and nb > 1:
            multi_res = True
    if multi_res and period_block_time is not None:
        rows: list[tuple[str, str, str]] = []
        pbt_sorted = period_block_time.sort("d", "b_first", "t")
        for (dval, bf), grp in pbt_sorted.group_by(
                ["d", "b_first"], maintain_order=True):
            ts = grp["t"].to_list()
            for i in range(1, len(ts)):
                rows.append((dval, ts[i], ts[i - 1]))
        if not rows:
            return None
        return (pl.DataFrame(rows, schema=["d", "t", "t_previous"],
                                orient="row").unique())
    if "t_previous_within_timeset" not in dtttdt.columns:
        return None
    out = (dtttdt
            .filter(pl.col("t_previous_within_timeset")
                    == pl.col("t_previous"))
            .select("d", "t", "t_previous"))
    return out if out.height > 0 else None


# ---------------------------------------------------------------------------
# Public — Γ.3.E field list (selective scope-checking).
# ---------------------------------------------------------------------------


E_PUBLIC_FIELDS: tuple[str, ...] = (
    "dtttdt",
    "dtttdt_forward_only",
    "dtttdt_block_interior",
    "nodeStateBlock",
    "period_block",
    "period_block_succ",
    "period_block_time",
    "arc_sink_block_dt",
    "arc_source_block_dt",
    "p_arc_sink_weight",
    "p_arc_source_weight",
    "p_state_upper",
    "p_state_existing_capacity",
    "storage_use_reference_value",
    "p_roll_continue_state",
    "p_fix_storage_quantity",
    "dtt_timeline_matching",
    "period_branch",
)


def apply_derived_e(
    flex_data: object,
    source: "InputSource",
    workdir: Path,
    *,
    ctx: "SolveContext | None" = None,
) -> None:
    """Apply Γ.3.E storage block algebra (§3.9), mutating ``flex_data``
    in place.

    Order — dependency-driven:
      1. dtttdt + auxiliary lag frames (§3.9.1).
      2. period_block / _succ / _time (§3.9.3).
      3. nodeStateBlock multi-resolution synthesis (§3.9.2).
      4. arc_sink_block_dt / arc_source_block_dt + weights (§3.9.4).
      4b. nodeState_last_dt + flow_from_nodeBalance block filter (Δ.9).
      5. p_state_existing_capacity / p_state_upper (§3.9.5).
      6. storage_use_reference_value (§3.9.6).
      7. p_roll_continue_state / p_fix_storage_quantity (§3.9.7).
      8. dtt_timeline_matching / period_branch (§3.9.8).

    Δ.3 replaced the previous ``derived_overrides_e`` dict-return;
    Δ.4 deleted the deprecated wrapper alias.
    Δ.9 wires every cluster E consumer onto a single ``BlockBundle``
    instance (replacing four separate ``BlockLayout.load_from_solve_data``
    calls per solve) and lands the previously-CSV-only
    ``nodeState_last_dt`` + ``flow_from_nodeBalance_*`` block filter on
    the source-driven path.
    """
    from flextool.engine_polars._derived_block import (
        load_block_bundle,
        nodeState_last_dt_lf,
        flow_from_nodeBalance_block_filtered,
    )

    active_solve = ctx.solve_name if ctx is not None else _read_active_solve(workdir)
    nodeState_df = getattr(flex_data, "nodeState", None)
    has_state = (nodeState_df is not None
                  and getattr(nodeState_df, "height", 0) > 0)
    # Δ.9 — single BlockBundle per solve (replaces 4× CSV reads).
    # Δ.12b — bundle-load failure is non-fatal (workdir without block
    # CSVs is legitimate for fixtures without explicit blocks).
    try:
        block_bundle = load_block_bundle(workdir)
    except Exception:
        block_bundle = None

    # Δ.12b — assignment is unconditional for storage-only Params and
    # the rolling-handoff carriers.  ``None`` is the explicit
    # "feature inactive / no handoff state" signal.

    # 1. dtttdt -------------------------------------------------------
    # Δ.12c-fix2 gap #5 close — ``_dt_period_active_steps`` now falls
    # back to reading ``solve_data/steps_in_use.csv`` for synthetic
    # rolling-horizon / nested-invest sub-solves whose names aren't in
    # the Spine DB.  The fallback uses ``timeline.timestep_duration``
    # for rank/timeline assignment.  Conditional assignment is retained
    # because fixtures genuinely lacking timeline data on the source AND
    # the workdir CSV (very rare; not present in the engine_polars
    # corpus) still need to fall through to the seed-loaded value.
    dtttdt_db = dtttdt_from_source(source, active_solve, workdir)
    if dtttdt_db is not None:
        flex_data.dtttdt = dtttdt_db

    # 2. period_block family (storage-only). --------------------------
    pb_family = None
    if has_state:
        pb_family = period_block_family_from_source(
            source, active_solve, workdir)
    period_block_time_for_arc: pl.DataFrame | None = None
    if pb_family is not None:
        flex_data.period_block = pb_family["period_block"]
        flex_data.period_block_succ = pb_family["period_block_succ"]
        flex_data.period_block_time = pb_family["period_block_time"]
        period_block_time_for_arc = pb_family["period_block_time"]

    # 1b. dtttdt-derived lag frames (storage-only). ------------------
    if has_state:
        fwd = dtttdt_forward_only_from_dtttdt(dtttdt_db)
        if fwd is not None:
            sbm_full = _node_storage_binding_method_with_fallback(source)
            if (sbm_full is not None
                    and sbm_full.filter(
                        pl.col("method") == "bind_forward_only").height > 0):
                flex_data.dtttdt_forward_only = fwd
        flex_data.dtttdt_block_interior = dtttdt_block_interior_from_dtttdt(
            dtttdt_db, period_block_time_for_arc)

    # 3. nodeStateBlock synthesis (storage-only) ----------------------
    nsb_db = None
    if has_state:
        nsb_db = nodeStateBlock_from_source(source, workdir)
        flex_data.nodeStateBlock = nsb_db

    # 4. arc block weights --------------------------------------------
    # NB: FlexData attribute is ``process_source_sink`` (not ``pss``);
    # the prior ``getattr(..., "pss", ...)`` always returned None and
    # the override never fired — local seeds in input.py:3060+ were the
    # sole producer.  Δ.17b Gap A: fixed; eff/noEff fallbacks also use
    # canonical names.
    pss = getattr(flex_data, "process_source_sink", None)
    if pss is None:
        pss = (getattr(flex_data, "process_source_sink_eff", None)
               or getattr(flex_data, "process_source_sink_noEff", None))
    if (nsb_db is not None and period_block_time_for_arc is not None
            and pss is not None):
        arc = arc_block_dt_from_source(
            source, workdir, nsb_db, period_block_time_for_arc, pss)
        if arc is not None:
            for k in ("arc_sink_block_dt", "arc_source_block_dt",
                       "p_arc_sink_weight", "p_arc_source_weight"):
                setattr(flex_data, k, arc.get(k))

    # 4b. nodeState_last_dt — Δ.9: previously only set by ``input.py``'s
    # ``_load_storage`` from the CSV path.  The lazy port lifts the
    # algorithm onto the in-memory ``BlockBundle`` so source-driven
    # parity holds for multi-block fixtures.
    if has_state and block_bundle is not None:
        nsld = nodeState_last_dt_lf(nodeState_df, block_bundle).collect()
        if nsld is not None and nsld.height > 0:
            flex_data.nodeState_last_dt = nsld

    # 4c. flow_from_nodeBalance block filter — Δ.9: mirrors
    # ``input.py::_load_storage`` lines 1664-1699.  Drops arc rows
    # whose source block doesn't overlap the destination node's block.
    # Δ.27: ``apply_derived_b`` is now the authoritative producer of the
    # ``flow_from_nodeBalance_{eff,noEff}`` seed AND already applies the
    # block-compat filter when a bundle is loadable from the workdir.
    # This re-application is therefore an idempotent safety net for the
    # case where ``apply_derived_b`` ran without a bundle (workdir's
    # block CSVs not yet written) but ``apply_derived_e`` later resolves
    # one — re-filter so the source-driven path matches the slow path.
    if block_bundle is not None and block_bundle.has_block_data():
        for fld in ("flow_from_nodeBalance_eff",
                     "flow_from_nodeBalance_noEff"):
            cur = getattr(flex_data, fld, None)
            if cur is None or getattr(cur, "height", 0) == 0:
                continue
            filtered = flow_from_nodeBalance_block_filtered(cur, block_bundle)
            if filtered is not None and filtered.height > 0:
                setattr(flex_data, fld, filtered)

    # 5. p_state_existing_capacity / p_state_upper --------------------
    if has_state:
        pse = p_state_existing_capacity_from_source(
            source, active_solve, nodeState_df, workdir)
        flex_data.p_state_existing_capacity = pse
        flex_data.p_state_upper = p_state_upper_from_source(
            source, pse, nodeState_df)
        # §F.2 p_state_unitsize (Δ.10 cluster F) — per-node unitsize
        # restricted to nodeState.
        from ._derived_arithmetic import p_state_unitsize_from_source
        flex_data.p_state_unitsize = p_state_unitsize_from_source(
            source, nodeState_df)

    # 6. storage_use_reference_value (storage-only) -------------------
    if has_state:
        flex_data.storage_use_reference_value = (
            storage_use_reference_value_from_source(source, nsb_db))

    # 7. rolling-handoff (storage-only — read-from-disk handoff) ------
    if has_state:
        flex_data.p_roll_continue_state = p_roll_continue_state_from_workdir(workdir)
        flex_data.p_fix_storage_quantity = p_fix_storage_quantity_from_workdir(workdir)
        flex_data.dtt_timeline_matching = dtt_timeline_matching_from_workdir(workdir)
        flex_data.period_branch = period_branch_from_source(source, workdir)




# ===========================================================================
# Γ.3.F — Lifetime cascade + handoff state + multi-year inflation cascade
# ===========================================================================
#
# This batch lands the §3.1.3 multi-year inflation cascade in full
# (replacing Γ.3.A's simple-1-year-per-period subcase), the §3.7.5/6
# discounted lifetime-fixed-cost / annual-discounted-invest annuity
# families, and the §3.7.7/8 rolling-handoff Param read side.  It is
# the dependency-final piece of the §3.7 + §3.11 cohort.
#
# Per the architectural shift (no defensive feature gating), helpers
# return ``None`` only when the underlying source is missing or the
# fixture's invest/lifetime configuration is degenerate (no
# entityInvest, no lifetime_method, etc.).  When data is present, the
# helper produces the canonical frame; the parity test catches any
# discrepancy.
#
# Authoritative algorithm references:
#   * §3.1.3 (multi-year cascade): preprocessing/period_calculated_params.py:200-322.
#   * §3.7.5/6 (lifetime + annual_discounted): preprocessing/entity_annual_calc_params.py:105-348.
#   * §3.7.7/8 (handoff state): preprocessing/entity_period_calc_params.py:1525-1600.


# ---------------------------------------------------------------------------
# §3.1.3 — multi-year inflation cascade (full version)
# ---------------------------------------------------------------------------


_LIFETIME_METHOD_DEFAULT_FALLBACK = "reinvest_automatic"


def _solve_inflation_inputs(source: "InputSource") -> tuple[float, float, float]:
    """Read the three scalar inflation inputs from ``model``.

    Returns ``(rate, offset_invest, offset_ops)``.  Mirrors flextool's
    ``_scalar_max`` over ``p_inflation_*.csv``:
        * rate              → 0.0 when no explicit Spine value
        * offset_invest     → 0.0 when no explicit Spine value
        * offset_operations → 0.5 when no explicit Spine value

    flextool's input_writer skips Spine *defaults* (only writes
    explicit parameter rows to ``input/p_inflation_*.csv``), so the
    downstream cascade reads ``_scalar_max(... default)`` and falls back
    to the constant defaults documented above.  We mirror that: when
    the SpineDbReader returns only the broadcast-default row (no
    explicit per-entity row), we fall back to flextool's defaults.

    We can't tell from a returned frame whether a value came from
    explicit storage or default broadcast; we use ``parameter_default``
    plus row count to detect the all-default case.  When the default
    matches every row's value AND no explicit per-class row exists,
    treat as "no explicit" (use flextool's CSV default).
    """
    def _explicit_max(par: str, default: float) -> float:
        df = _try_param(source, "model", par)
        if df is None or df.height == 0:
            return default
        # Detect "all rows = Spine default broadcast" case.  When the
        # Spine reader's default-fill kicked in, every row's value
        # equals the parameter_default and the CSV writer would have
        # produced an empty file.  Fall back to flextool's constant.
        try:
            spine_default = source.parameter_default("model", par)
        except KeyError:
            spine_default = None
        if spine_default is not None:
            try:
                spine_default_f = float(spine_default)
            except (ValueError, TypeError):
                spine_default_f = None
            if spine_default_f is not None:
                # If every row equals the default, treat as no explicit data.
                vals = df["value"].cast(pl.Float64, strict=False).to_list()
                if vals and all(
                        (v is not None and abs(v - spine_default_f) < 1e-12)
                        for v in vals):
                    return default
        try:
            return float(df["value"].cast(pl.Float64).max())
        except Exception:
            return default

    rate = _explicit_max("inflation_rate", 0.0)
    off_inv = _explicit_max("inflation_offset_investment", 0.0)
    off_ops = _explicit_max("inflation_offset_operations", 0.5)
    return rate, off_inv, off_ops


def _years_for_period_from_source(source: "InputSource",
                                    active_solve: str | None,
                                    period_set: list[str],
                                    ) -> dict[str, list[tuple[str, float]]]:
    """Reproduce ``solve_writers.write_years_represented`` to derive the
    per-period ``(year_label, width)`` list flextool's preprocessing
    builds in ``solve_data/p_years_represented.csv``.

    Algorithm (mirror of solve_writers.py:112-144):
      For each (period, R) in solve.years_represented[active_solve],
      with R > 0:
        rows = ceil(R), remaining = R, year_count starts at solve-global 0.
        For i in range(rows):
          width = min(1.0, remaining)
          emit (period, str(year_count), width)
          year_count += width; remaining -= width

    Note: flextool also expands into stochastic branches
    (``period__branch``).  We don't model that here — for non-stochastic
    fixtures the branch loop is a no-op.

    Returns a dict ``{period: [(year_label, width), ...]}``.  When the
    parameter is absent or active_solve has no rows, default to one row
    per period in the order supplied (width=1, year_label = "0", "1", …).
    """
    yr_p = _try_param(source, "solve", "years_represented")
    out: dict[str, list[tuple[str, float]]] = {}
    if active_solve is None or yr_p is None or "period" not in yr_p.columns:
        # Default fallback — one year per period, width 1, integer-indexed.
        for i, d in enumerate(period_set):
            out[d] = [(str(int(i)), 1.0)]
        return out

    # Rolling solves: ``active_solve`` is the per-roll name
    # (``<parent>_roll_<N>``) but the DB has the parameter on the parent
    # solve.  flextool's preprocessing copies the parent value down to
    # each roll.  Try the active_solve first, then the parent name.
    sub = yr_p.filter(pl.col("name") == active_solve)
    if sub.height == 0:
        import re
        parent = re.sub(r"_roll_\d+$", "", active_solve)
        if parent != active_solve:
            sub = yr_p.filter(pl.col("name") == parent)
    if sub.height == 0:
        for i, d in enumerate(period_set):
            out[d] = [(str(int(i)), 1.0)]
        return out

    # Iterate in CSV order — flextool walks the Spine Map order, which
    # the Spine reader presents row-by-row.  We sort by period to match
    # the canonical ordering when periods aren't already sorted in the
    # source — flextool's solve_writers iterates the dict in insertion
    # order, which mirrors the periodAll set ordering.  For our purposes
    # the period_set arg gives the canonical order.
    rows_by_d: dict[str, float] = {}
    for r in sub.iter_rows(named=True):
        rows_by_d[str(r["period"])] = float(r["value"])

    import math
    year_count = 0.0
    for d in period_set:
        R = rows_by_d.get(d, 0.0)
        if R <= 0:
            out[d] = []
            continue
        rows = int(math.ceil(R))
        remaining = R
        emitted: list[tuple[str, float]] = []
        for _i in range(rows):
            width = min(1.0, remaining)
            # Mirror solve_writers' int-cast path: when year_count is an
            # integer (R = whole numbers) we emit ``str(int(yc))``
            # literals; with fractional widths flextool writes the
            # float-stringified value.  Emulate the same:
            if year_count == float(int(year_count)):
                lbl = str(int(year_count))
            else:
                lbl = str(year_count)
            emitted.append((lbl, width))
            year_count += width
            remaining -= width
        out[d] = emitted
    return out


def _periodAll_from_source(source: "InputSource",
                              active_solve: str | None,
                              workdir: Path | None = None) -> list[str]:
    """Compute the periodAll set for the active solve.

    Mirrors flextool's ``periodAll_set.csv`` shape.  When ``workdir``
    is provided AND the per-solve preprocessing has already written
    the file, prefer that (it's the authoritative cumulative-history
    union for chained solves).  Falls back to the simple
    realized ∪ invest union from the source when the file is absent.
    """
    if workdir is not None:
        p = Path(workdir) / "solve_data" / "periodAll_set.csv"
        if p.exists():
            df = _read_csv_file(p)
            if df.height > 0 and "period" in df.columns:
                return df["period"].cast(pl.Utf8, strict=False).to_list()
    seen: dict[str, None] = {}
    if active_solve is None:
        return []
    for par in ("realized_periods", "invest_periods"):
        ps = _solve_periods(source, active_solve, par) or []
        for d in ps:
            seen.setdefault(d, None)
    return list(seen.keys())


def _inflation_yearly_from_source(source: "InputSource",
                                     active_solve: str | None,
                                     workdir: Path | None = None,
                                     ) -> tuple[dict[str, float],
                                                 dict[str, float]] | None:
    """Compute ``p_inflation_factor_investment_yearly[d]`` and
    ``p_inflation_factor_operations_yearly[d]`` per
    ``period_calculated_params.py:200-322``.

    Returns ``(invest_factor, ops_factor)`` dicts keyed on period.
    Returns ``None`` only when the periodAll set is empty.

    Algorithm:
      1. ``inflation_rate``, ``inflation_offset_invest/ops`` from model.
      2. For each period ``d`` ∈ periodAll, build ``years_for_period[d]``
         via :func:`_years_for_period_from_source`.
      3. Build the GLOBAL year set = union of all year labels across
         all periods, sort numerically (or lexically when non-numeric).
      4. For each period ``d``:
         a. Walk the global year set computing
            ``global_pos[y2] = sum_{y' < y2 in global order} pyr[d, y']``
            (inner accumulator restricted to the period's row).
         b. For each y in years_for_period[d]:
              base = global_pos[y]
              until_inv = base + pyr[d, y] * offset_invest
              until_ops = base + pyr[d, y] * offset_ops
         c. ``inv_factor[d] = sum_y pyr[d, y] * (1 + rate)^(-until_inv)``
            ``ops_factor[d] = sum_y pyr[d, y] * (1 + rate)^(-until_ops)``
            When ``sum_y pyr[d, y] == 0`` → factor = 1.0.
    """
    period_universe = _periodAll_from_source(source, active_solve, workdir)
    if not period_universe:
        return None
    rate, off_inv, off_ops = _solve_inflation_inputs(source)
    one_plus_inflation_inv = (1.0 / (1.0 + rate)) if rate != -1.0 else 1.0
    yfp = _years_for_period_from_source(source, active_solve,
                                          period_universe)

    # Global year set — union of all year labels.
    global_year_set: dict[str, None] = {}
    for years_list in yfp.values():
        for y, _w in years_list:
            global_year_set.setdefault(y, None)
    try:
        sorted_global_years = sorted(global_year_set.keys(),
                                       key=lambda y: float(y))
    except ValueError:
        sorted_global_years = sorted(global_year_set.keys())

    inflation_invest: dict[str, float] = {}
    inflation_ops: dict[str, float] = {}

    # Build a fast lookup: pyr_lookup[(d, y)] = width.
    pyr_lookup: dict[tuple[str, str], float] = {}
    for d, years in yfp.items():
        for y, w in years:
            pyr_lookup[(d, y)] = w

    for d in period_universe:
        years_for_d = yfp.get(d, [])
        try:
            sorted_d_years = sorted(years_for_d, key=lambda yw: float(yw[0]))
        except ValueError:
            sorted_d_years = sorted(years_for_d, key=lambda yw: yw[0])

        # global_pos[y2] = sum_{y' in global, y' < y2} pyr[d, y'] (default 1)
        cumulative = 0.0
        global_pos: dict[str, float] = {}
        for y2 in sorted_global_years:
            global_pos[y2] = cumulative
            cumulative += pyr_lookup.get((d, y2), 1.0)

        sum_p_years_for_d = sum(w for _y, w in sorted_d_years)
        if sum_p_years_for_d > 0:
            inv_factor = 0.0
            ops_factor = 0.0
            for y, w in sorted_d_years:
                base = global_pos.get(y, 0.0)
                until_inv = base + w * off_inv
                until_ops = base + w * off_ops
                inv_factor += w * (one_plus_inflation_inv ** until_inv)
                ops_factor += w * (one_plus_inflation_inv ** until_ops)
            inflation_invest[d] = inv_factor
            inflation_ops[d] = ops_factor
        else:
            inflation_invest[d] = 1.0
            inflation_ops[d] = 1.0

    return inflation_invest, inflation_ops


def p_inflation_op_full_cascade_from_source(source: "InputSource",
                                                active_solve: str | None,
                                                dt: pl.DataFrame,
                                                workdir: Path | None = None,
                                                ) -> "Param | None":
    """Full multi-year inflation cascade for ``p_inflation_op``.

    Replaces both Γ.3.A's trivial path (rate=0) and Γ.3.C's
    simple-1-year-per-period subcase: walks all (d, y) ∈ years_for_period
    summing ``pyr * (1+rate)^until_op[d, y]``.  Domain is
    ``period_in_use`` (mirrors flextool.mod L1551 — operations factor is
    declared over period_in_use).
    """
    factors = _inflation_yearly_from_source(source, active_solve, workdir)
    if factors is None:
        return None
    _inv, ops = factors
    period_in_use = _period_in_use_set(source, active_solve, workdir)
    if not period_in_use:
        return None
    rows = [(d, ops.get(d, 1.0)) for d in period_in_use]
    df = pl.DataFrame(rows, schema=["d", "value"], orient="row")
    if df.height == 0:
        return None
    return Param(("d",), df.sort("d"))


# ---------------------------------------------------------------------------
# §3.7.7/8 — handoff state (rolling solve-chain feeds)
# ---------------------------------------------------------------------------


def _read_handoff_e_d_from_workdir(workdir: Path,
                                       name: str) -> "Param | None":
    """Read a (entity, period, value) handoff CSV; mirror the loader's
    zero-filter (input.py:1316-1328).  Returns None when the file is
    absent / empty / all-zero.
    """
    p = Path(workdir) / "solve_data" / f"{name}.csv"
    if not p.exists():
        return None
    df = _read_csv_file(p)
    if df.height == 0:
        return None
    df = (df.rename({"entity": "e", "period": "d"})
              .with_columns(value=pl.col("value")
                                       .cast(pl.Float64, strict=False)
                                       .fill_null(0.0))
              .filter(pl.col("value") != 0.0)
              .select("e", "d", "value"))
    if df.height == 0:
        return None
    return Param(("e", "d"), df)


def _read_handoff_e_from_workdir(workdir: Path, name: str,
                                    value_col: str | None = None
                                    ) -> "Param | None":
    """Wide-format (entity, p_entity_invested) → Param ((e,), value)."""
    p = Path(workdir) / "solve_data" / f"{name}.csv"
    if not p.exists():
        return None
    df = _read_csv_file(p)
    if df.height == 0:
        return None
    if value_col is None:
        non = [c for c in df.columns if c != "entity"]
        if not non:
            return None
        value_col = non[0]
    df = (df.rename({"entity": "e", value_col: "value"})
              .with_columns(value=pl.col("value")
                                       .cast(pl.Float64, strict=False)
                                       .fill_null(0.0))
              .filter(pl.col("value") != 0.0)
              .select("e", "value"))
    if df.height == 0:
        return None
    return Param(("e",), df)


def p_entity_previously_invested_capacity_from_workdir(
        workdir: Path) -> "Param | None":
    """Per-solve handoff carrier (§3.7.7).  Read-side mirror of the
    chain runner's write-side: ``solve_data/p_entity_previously_invested_capacity.csv``
    holds the cumulative prior-solve invest summed across the entity's
    historical periods.  In single-solve fixtures this CSV is
    header-only (or all-zero), and the Param collapses to ``None``.

    For chain runs, the chain runner is responsible for writing the
    fresh CSV between sub-solves (or for calling :func:`apply_handoff`
    with the in-memory carrier).  This helper only reads from disk —
    it doesn't manufacture state.
    """
    return _read_handoff_e_d_from_workdir(
        workdir, "p_entity_previously_invested_capacity")


def p_entity_invested_from_workdir(workdir: Path) -> "Param | None":
    """Per-entity scalar of cumulative prior-solve invest (§3.7.8).

    Read from ``solve_data/p_entity_invested.csv`` (a 2-column
    ``entity, p_entity_invested`` wide CSV).  Single-solve fixtures
    leave this header-only; reads as ``None``.
    """
    return _read_handoff_e_from_workdir(workdir, "p_entity_invested")


def p_entity_divested_from_workdir(workdir: Path) -> "Param | None":
    """Per-entity scalar of cumulative prior-solve divest (§3.7.8)."""
    return _read_handoff_e_from_workdir(workdir, "p_entity_divested")


# ---------------------------------------------------------------------------
# §3.7.5/6 — Lifetime fixed cost + entity annual discounted (annuity)
# ---------------------------------------------------------------------------


_INVEST_NOT_ALLOWED_F: frozenset[str] = frozenset((
    "not_allowed", "retire_period", "retire_total", "retire_no_limit",
))
_DIVEST_NOT_ALLOWED_F: frozenset[str] = frozenset((
    "not_allowed", "invest_period", "invest_total", "invest_no_limit",
))


def _annuity(invest_value: float, discount_rate: float,
              lifetime: float) -> float:
    """Mirror :func:`flextool/preprocessing/entity_annual_calc_params._annuity`.

    annuity = invest_value * 1000 * r / (1 - 1/(1+r)^n) with mod's
    ≤ 0 fallbacks: r → 0.05, n → 20.
    """
    r = discount_rate if discount_rate > 0 else 0.05
    n = lifetime if lifetime > 0 else 20.0
    if r == 0:
        return 0.0
    return invest_value * 1000.0 * r / (1.0 - (1.0 / (1.0 + r)) ** n)


def _entity_classes_lookup(source: "InputSource"
                              ) -> tuple[set[str], set[str], set[str]]:
    """Return ``(process_set, node_set, connection_set)`` of entity names.

    Mirrors flextool's input/process.csv = unit ∪ connection (see
    ``input_writer.py:42``); the distinct ``unit`` and ``connection``
    classes both feed into the shared ``process_set`` consumed by
    ``entity_period_calc_params._ed_value`` and the annuity sums.
    The ``connection_set`` is kept separate so callers can resolve
    connection-only attributes (e.g. ``is_DC``) when needed.
    """
    def _set(ec: str) -> set[str]:
        df = _try_entities(source, ec)
        if df is None:
            return set()
        return set(df["name"].to_list())
    unit_s = _set("unit")
    conn_s = _set("connection")
    process_s = unit_s | conn_s
    return process_s, _set("node"), conn_s


def _per_entity_period_value(source: "InputSource",
                                  parameter_name: str,
                                  ) -> dict[tuple[str, str], float] | None:
    """Read ``unit/node/connection.<parameter_name>`` and return a
    ``(entity, period) → value`` dict.  Scalar shapes broadcast to all
    periods downstream; this helper itself only emits explicit rows.

    A separate ``_get_value(e, d)`` helper layers the broadcast.
    """
    out: dict[tuple[str, str], float] = {}
    for ec in ("unit", "node", "connection"):
        df = _try_param(source, ec, parameter_name)
        if df is None:
            continue
        cols = df.columns
        if "period" in cols:
            for r in df.iter_rows(named=True):
                try:
                    out[(str(r["name"]), str(r["period"]))] = \
                        float(r["value"])
                except Exception:
                    continue
    return out if out else None


def _per_entity_scalar(source: "InputSource",
                          parameter_name: str,
                          ) -> dict[str, float]:
    """Per-entity scalar (no period dim) from
    unit/node/connection.<parameter_name>.  Returns ``{}`` when none.
    """
    out: dict[str, float] = {}
    for ec in ("unit", "node", "connection"):
        df = _try_param(source, ec, parameter_name)
        if df is None:
            continue
        cols = df.columns
        if "period" in cols:
            continue
        for r in df.iter_rows(named=True):
            try:
                out[str(r["name"])] = float(r["value"])
            except Exception:
                continue
    return out


def _entity_invest_methods(source: "InputSource"
                              ) -> dict[str, list[str]]:
    """Return ``entity → [invest_method, ...]`` from
    ``entity__invest_method.csv``.  Spine source: per-entity-class
    ``invest_method`` Map, scalar-valued; flextool keeps the per-entity
    list (flextool.mod allows multi-method but every concrete model
    has 1).
    """
    out: dict[str, list[str]] = {}
    for ec in ("unit", "node", "connection"):
        df = _try_param(source, ec, "invest_method")
        if df is None:
            continue
        for r in df.iter_rows(named=True):
            out.setdefault(str(r["name"]), []).append(str(r["value"]))
    return out


def _entity_lifetime_methods(source: "InputSource",
                                  all_entities: list[str],
                                  ) -> dict[str, list[str]]:
    """Return ``entity → [lifetime_method, ...]`` for every entity.

    Default (when an entity has no explicit lifetime_method row) is
    ``["reinvest_automatic"]`` per
    ``preprocessing/method_with_fallback_sets.py::_LIFETIME_METHOD_DEFAULT``.
    """
    explicit: dict[str, list[str]] = {}
    for ec in ("unit", "node", "connection"):
        df = _try_param(source, ec, "lifetime_method")
        if df is None:
            continue
        for r in df.iter_rows(named=True):
            explicit.setdefault(str(r["name"]), []).append(str(r["value"]))

    out: dict[str, list[str]] = {}
    for e in all_entities:
        if e in explicit:
            out[e] = list(explicit[e])
        else:
            out[e] = [_LIFETIME_METHOD_DEFAULT_FALLBACK]
    return out


def _all_entities(source: "InputSource") -> list[str]:
    """Union of unit + node + connection entity names, in flextool's
    canonical iteration order (concatenation, dedup with first-wins).
    """
    seen: dict[str, None] = {}
    for ec in ("unit", "node", "connection"):
        df = _try_entities(source, ec)
        if df is None:
            continue
        for n in df["name"].to_list():
            seen.setdefault(str(n), None)
    return list(seen.keys())


def _resolve_pdX(per_period: dict[tuple[str, str], float] | None,
                    scalar: dict[str, float],
                    e: str,
                    d: str) -> float:
    """Resolve `pdX[e, param, d]` mirroring flextool's PdLookup:
       1. (e, d) explicit row → that value.
       2. else (e, ·) scalar  → broadcast.
       3. else                → 0.0 (mod default).
    """
    if per_period is not None:
        v = per_period.get((e, d))
        if v is not None:
            return v
    return scalar.get(e, 0.0)


def _ed_fixed_cost_mapping(source: "InputSource"
                              ) -> dict[tuple[str, str], float]:
    """Read ``solve_data/ed_fixed_cost.csv`` shape — i.e. the
    per-(e, d) ``fixed_cost`` value flextool prebuilds.  We reproduce
    it here from raw Spine ``unit/node/connection.fixed_cost`` (Map or
    scalar).  Default 0.0.
    """
    per = _per_entity_period_value(source, "fixed_cost") or {}
    scalar = _per_entity_scalar(source, "fixed_cost")
    if not per and not scalar:
        return {}
    # Materialise scalar broadcast lazily via _resolve_pdX at the
    # consumer; here we just merge for direct (e, d) cases.
    out = dict(per)
    return out


def _ed_lifetime_mapping(source: "InputSource",
                            entities: list[str],
                            period_with_history: list[str],
                            ) -> dict[tuple[str, str], float]:
    """Reproduce ``solve_data/edEntity_lifetime.csv``.

    Algorithm (mirror of
    ``preprocessing/entity_period_calc_params.py:140-147``): for each
    (e, d) ∈ entity × period_with_history, emit
    ``pdProcess[e, lifetime, d]`` if e is a process (unit ∪ connection),
    ``pdNode[e, lifetime, d]`` if e is a node, else 0.

    Scalar ``lifetime`` is broadcast over the period axis (PdLookup
    behaviour).  Returns ``{(e, d) → value}``.
    """
    process_set, node_set, _conn_set = _entity_classes_lookup(source)
    per_p = _per_entity_period_value(source, "lifetime") or {}
    scalar = _per_entity_scalar(source, "lifetime")
    out: dict[tuple[str, str], float] = {}
    for e in entities:
        if e not in process_set and e not in node_set:
            for d in period_with_history:
                out[(e, d)] = 0.0
            continue
        for d in period_with_history:
            v = _resolve_pdX(per_p, scalar, e, d)
            out[(e, d)] = v
    return out


def _per_method_annuity(e: str, d: str,
                            cost_param_name: str,
                            cost_per: dict[tuple[str, str], float] | None,
                            cost_scalar: dict[str, float],
                            disc_per: dict[tuple[str, str], float] | None,
                            disc_scalar: dict[str, float],
                            life_per: dict[tuple[str, str], float] | None,
                            life_scalar: dict[str, float],
                            methods_for_e: list[str],
                            disallowed: frozenset[str],
                            unit_set: set[str],
                            node_set: set[str],
                            conn_set: set[str]) -> float:
    """Sum-over-allowed-methods annuity reproducing
    ``entity_annual_calc_params._per_method_annuity_invest/divest``.

    Each method m ∈ methods_for_e \\ disallowed contributes
    ``_annuity(cost_value(e, d), discount_rate(e, d), lifetime(e, d))``.
    """
    v = 0.0
    if e not in unit_set and e not in node_set and e not in conn_set:
        return 0.0
    for m in methods_for_e:
        if m in disallowed:
            continue
        cost = _resolve_pdX(cost_per, cost_scalar, e, d)
        disc = _resolve_pdX(disc_per, disc_scalar, e, d)
        life = _resolve_pdX(life_per, life_scalar, e, d)
        v += _annuity(cost, disc, life)
    return v


def _p_discount_years_from_source(source: "InputSource",
                                       active_solve: str | None,
                                       ) -> dict[str, float]:
    """Reproduce ``solve_data/p_discount_years.csv`` —
    ``solve_writers.write_period_years``.

    Algorithm: for each period in solve.years_represented[active_solve]
    (iteration order = Spine Map order = canonical period_set order),
    emit ``(period, year_count)`` then ``year_count += R[d]``.

    Periods absent from years_represented (or with R=0) get value 0.0
    via the consumer's ``.get(d, 0.0)`` default.

    Returns ``{period: cumulative_years}`` dict.
    """
    out: dict[str, float] = {}
    if active_solve is None:
        return out
    yrp = _try_param(source, "solve", "years_represented")
    if yrp is None or "period" not in yrp.columns:
        return out
    sub = yrp.filter(pl.col("name") == active_solve)
    if sub.height == 0:
        return out
    # Spine Map iteration order — preserve the row order from the source.
    # The Spine reader returns rows in DB-storage order which mirrors the
    # canonical Map index order.
    year_count = 0.0
    for r in sub.iter_rows(named=True):
        d = str(r["period"])
        out[d] = year_count
        try:
            year_count += float(r["value"])
        except Exception:
            pass
    return out


def _read_period_with_history(workdir: Path | None) -> list[str]:
    """Read ``solve_data/period_with_history.csv`` from disk if present.

    For multi-solve chain runs flextool's per-solve preprocessing
    (``orchestration.py::main`` block at L262) writes this CSV with the
    accumulated period universe from all prior solves.  When the file
    is absent (e.g. tests using only the input/ side), the caller's
    fallback to ``period_in_use`` is used.

    Returns ``[period, ...]`` in CSV row order.
    """
    if workdir is None:
        return []
    p = Path(workdir) / "solve_data" / "period_with_history.csv"
    if not p.exists():
        return []
    df = _read_csv_file(p)
    if df.height == 0 or "period" not in df.columns:
        return []
    return df["period"].cast(pl.Utf8, strict=False).to_list()


def ed_entity_annual_family_from_source(source: "InputSource",
                                            active_solve: str | None,
                                            ed_invest: pl.DataFrame | None,
                                            ed_divest: pl.DataFrame | None,
                                            workdir: Path | None = None,
                                            ) -> dict[str, "Param | None"]:
    """Compute ``ed_entity_annual``, ``ed_entity_annual_discounted``,
    ``ed_entity_annual_divest``, ``ed_entity_annual_divest_discounted``
    in a single pass over (e, d) ∈ ed_invest / ed_divest.

    Returns a dict keyed by FlexData attribute name.  Only the two
    discounted Params are exposed via :class:`FlexData` (the non-
    discounted ones are intermediates flextool feeds straight into the
    discounted sum); the dict is shaped that way too.
    """
    factors = _inflation_yearly_from_source(source, active_solve, workdir)
    period_in_use = _period_in_use_set(source, active_solve, workdir)
    period_invest = _solve_periods(source, active_solve, "invest_periods") or []

    cost_invest = _per_entity_period_value(source, "invest_cost")
    cost_invest_scalar = _per_entity_scalar(source, "invest_cost")
    cost_div = _per_entity_period_value(source, "salvage_value")
    cost_div_scalar = _per_entity_scalar(source, "salvage_value")
    disc_per = _per_entity_period_value(source, "discount_rate")
    disc_scalar = _per_entity_scalar(source, "discount_rate")
    life_per = _per_entity_period_value(source, "lifetime")
    life_scalar = _per_entity_scalar(source, "lifetime")

    methods = _entity_invest_methods(source)
    process_set, node_set, conn_set = _entity_classes_lookup(source)

    p_discount_years_lookup = _p_discount_years_from_source(
        source, active_solve)

    inv_factor = ops_factor = None
    if factors is not None:
        inv_factor, ops_factor = factors

    # Build ed_invest list
    ed_inv_list: list[tuple[str, str]] = []
    if ed_invest is not None and ed_invest.height > 0:
        for r in ed_invest.iter_rows(named=True):
            ed_inv_list.append((str(r["e"]), str(r["d"])))
    ed_div_list: list[tuple[str, str]] = []
    if ed_divest is not None and ed_divest.height > 0:
        for r in ed_divest.iter_rows(named=True):
            ed_div_list.append((str(r["e"]), str(r["d"])))

    # entityInvest set = unique e values from ed_invest_list
    entityInvest_set = sorted({e for (e, _d) in ed_inv_list})
    entityDivest_set = sorted({e for (e, _d) in ed_div_list})

    # Lifetime methods (with reinvest_automatic default)
    all_entities = _all_entities(source)
    elm = _entity_lifetime_methods(source, all_entities)

    # period_with_history (history union).  Multi-solve chain runs
    # accumulate periods across prior solves; we read the per-solve
    # CSV when present (it's written by flextool's orchestration at
    # L262).  For single-solve fixtures the file equals period_in_use.
    period_with_history = _read_period_with_history(workdir) \
                              or list(period_in_use)
    edEntity_lifetime = _ed_lifetime_mapping(
        source, all_entities, period_with_history)

    # ed_entity_annual_discounted: per (e in entityInvest, d in period_invest)
    rows_ann_disc: list[tuple[str, str, float]] = []
    for e in entityInvest_set:
        elm_set = frozenset(elm.get(e, ()))
        is_choice_or_no_invest = (
            "reinvest_choice" in elm_set or "no_investment" in elm_set
        )
        is_automatic = "reinvest_automatic" in elm_set
        for d in period_invest:
            ann = _per_method_annuity(
                e, d, "invest_cost",
                cost_invest, cost_invest_scalar,
                disc_per, disc_scalar,
                life_per, life_scalar,
                methods.get(e, ()),
                _INVEST_NOT_ALLOWED_F,
                process_set, node_set, conn_set,
            )
            disc = 0.0
            pdy_d = p_discount_years_lookup.get(d, 0.0)
            life = edEntity_lifetime.get((e, d), 0.0)
            if is_choice_or_no_invest:
                s = 0.0
                for d_all in period_in_use:
                    pdy = p_discount_years_lookup.get(d_all, 0.0)
                    if pdy >= pdy_d and pdy < pdy_d + life:
                        s += (inv_factor.get(d_all, 1.0)
                              if inv_factor is not None else 1.0)
                disc += ann * s
            if is_automatic:
                s = 0.0
                for d_all in period_in_use:
                    pdy = p_discount_years_lookup.get(d_all, 0.0)
                    if pdy >= pdy_d:
                        s += (inv_factor.get(d_all, 1.0)
                              if inv_factor is not None else 1.0)
                disc += ann * s
            rows_ann_disc.append((e, d, disc))

    # ed_entity_annual_divest_discounted: per (e in entityDivest, d in period_invest)
    rows_div_disc: list[tuple[str, str, float]] = []
    for e in entityDivest_set:
        for d in period_invest:
            ann = _per_method_annuity(
                e, d, "salvage_value",
                cost_div, cost_div_scalar,
                disc_per, disc_scalar,
                life_per, life_scalar,
                methods.get(e, ()),
                _DIVEST_NOT_ALLOWED_F,
                process_set, node_set, conn_set,
            )
            disc = 0.0
            pdy_d = p_discount_years_lookup.get(d, 0.0)
            # Divest discount window uses raw lifetime (per-class
            # broadcast) — entity_annual_calc_params.py:266-285.
            life = _resolve_pdX(life_per, life_scalar, e, d)
            if e in node_set or e in process_set:
                s = 0.0
                for d_all in period_in_use:
                    pdy = p_discount_years_lookup.get(d_all, 0.0)
                    if pdy >= pdy_d and pdy < pdy_d + life:
                        s += (inv_factor.get(d_all, 1.0)
                              if inv_factor is not None else 1.0)
                disc = ann * s
            rows_div_disc.append((e, d, disc))

    # ed_lifetime_fixed_cost: per (e in entity, d in period_with_history)
    rows_lfc: list[tuple[str, str, float]] = []
    # ed_fixed_cost = fixed_cost * 1000 per
    # entity_period_calc_params.py:149-156, but only for processes
    # (unit ∪ connection) and nodes — non-classified entities get 0.
    fc_per_raw = _per_entity_period_value(source, "fixed_cost")
    fc_scalar_raw = _per_entity_scalar(source, "fixed_cost")
    fc_per = {k: v * 1000.0 for k, v in fc_per_raw.items()} \
              if fc_per_raw is not None else None
    fc_scalar = {k: v * 1000.0 for k, v in fc_scalar_raw.items()}
    for e in all_entities:
        elm_set = frozenset(elm.get(e, ()))
        is_choice_or_no_invest = (
            "reinvest_choice" in elm_set or "no_investment" in elm_set
        )
        is_automatic = "reinvest_automatic" in elm_set
        for d in period_with_history:
            fc = _resolve_pdX(fc_per, fc_scalar, e, d)
            v = 0.0
            pdy_d = p_discount_years_lookup.get(d, 0.0)
            life = edEntity_lifetime.get((e, d), 0.0)
            if is_choice_or_no_invest:
                s = 0.0
                for d_all in period_in_use:
                    pdy = p_discount_years_lookup.get(d_all, 0.0)
                    if pdy >= pdy_d and pdy < pdy_d + life:
                        s += (ops_factor.get(d_all, 1.0)
                              if ops_factor is not None else 1.0)
                v += fc * s
            if is_automatic:
                s = 0.0
                for d_all in period_in_use:
                    pdy = p_discount_years_lookup.get(d_all, 0.0)
                    if pdy >= pdy_d:
                        s += (ops_factor.get(d_all, 1.0)
                              if ops_factor is not None else 1.0)
                v += fc * s
            rows_lfc.append((e, d, v))

    # ed_lifetime_fixed_cost_divest: per (e in entityDivest, d in period_invest)
    # NB: mod L1651 uses INVESTMENT-yearly inflation here, not operations
    # — asymmetric vs the non-divest variant.
    rows_lfcd: list[tuple[str, str, float]] = []
    for e in entityDivest_set:
        for d in period_invest:
            fc = _resolve_pdX(fc_per, fc_scalar, e, d)
            v = 0.0
            pdy_d = p_discount_years_lookup.get(d, 0.0)
            life = _resolve_pdX(life_per, life_scalar, e, d)
            s = 0.0
            for d_all in period_in_use:
                pdy = p_discount_years_lookup.get(d_all, 0.0)
                if pdy >= pdy_d and pdy < pdy_d + life:
                    s += (inv_factor.get(d_all, 1.0)
                          if inv_factor is not None else 1.0)
            v = fc * s
            rows_lfcd.append((e, d, v))

    def _to_param(rows: list[tuple[str, str, float]]) -> "Param | None":
        # Mirror the loader's filter (input.py:1268): drop value==0 rows.
        nonzero = [(e, d, v) for (e, d, v) in rows if v != 0.0]
        if not nonzero:
            return None
        df = pl.DataFrame(nonzero,
                            schema=["e", "d", "value"], orient="row")
        return Param(("e", "d"), df.sort("e", "d"))

    # _load_invest short-circuits all four cost frames to None when
    # both ed_invest_set and ed_divest_set are empty (input.py:1116-1135).
    # Mirror that — these Params only feed v_invest / v_divest objective
    # terms, so omitting them when no such variables exist is correct.
    no_invest = (ed_invest is None or ed_invest.height == 0) and \
                 (ed_divest is None or ed_divest.height == 0)
    if no_invest:
        return {
            "ed_entity_annual_discounted": None,
            "ed_entity_annual_divest_discounted": None,
            "ed_lifetime_fixed_cost": None,
            "ed_lifetime_fixed_cost_divest": None,
        }

    return {
        "ed_entity_annual_discounted": _to_param(rows_ann_disc),
        "ed_entity_annual_divest_discounted": _to_param(rows_div_disc),
        "ed_lifetime_fixed_cost": _to_param(rows_lfc),
        "ed_lifetime_fixed_cost_divest": _to_param(rows_lfcd),
    }


# ---------------------------------------------------------------------------
# Integration: apply_derived_f
# ---------------------------------------------------------------------------


def apply_derived_f(
    flex_data: object,
    source: "InputSource",
    workdir: Path,
    *,
    ctx: "SolveContext | None" = None,
) -> None:
    """Apply Γ.3.F lifetime cascade family + handoff state + multi-year
    inflation cascade, mutating ``flex_data`` in place.

    Order:
      1. Cluster A (Δ.5): ``p_inflation_op``, ``p_ed_fixed_cost``,
         ``ed_entity_annual_discounted``, ``ed_entity_annual_divest_discounted``,
         ``ed_lifetime_fixed_cost``, ``ed_lifetime_fixed_cost_divest``
         — delegated to :func:`._derived_npv.apply_npv`, the lazy-polars
         port.  Replaces the eager ``ed_entity_annual_family_from_source``
         and ``p_inflation_op_full_cascade_from_source`` paths.
      2. ``p_entity_previously_invested_capacity`` / ``p_entity_invested``
         / ``p_entity_divested`` (handoff state).

    Δ.3 replaced the previous ``derived_overrides_f`` dict-return;
    Δ.4 deleted the deprecated wrapper alias; Δ.5 ports the lifetime
    cascade family to lazy polars.
    """
    from . import _derived_npv

    # 1. Cluster A (lazy NPV / inflation / fixed cost cascade).
    _derived_npv.apply_npv(flex_data, source, workdir)

    # 2. Handoff state -----------------------------------------------
    # Δ.12b — unconditional assignment.  Each helper returns ``None``
    # when the corresponding handoff CSV is missing / header-only
    # (single-solve fixtures, or first-of-chain) — ``None`` is the
    # explicit "no prior-solve state" signal.
    flex_data.p_entity_previously_invested_capacity = (
        p_entity_previously_invested_capacity_from_workdir(workdir))
    flex_data.p_entity_invested = p_entity_invested_from_workdir(workdir)
    flex_data.p_entity_divested = p_entity_divested_from_workdir(workdir)




# ===========================================================================
# Γ.3.G — Residual Derived Params (audit §3.13/3.15/3.17/3.18 + §3.6.1 cascade)
#
# Final batch.  Closes the Derived class:
#
#   * §3.17 commodity ladder (`p_f_d_k`, `p_ladder_cum_realized_mwh`)
#   * §3.13 reserves remainder (`prundt`)
#   * §3.15 delay (`dtt__delay_duration`, `p_process_delay_weight`)
#   * §3.18 multi-branch normalisation (full cascade for both
#     ``pdt_branch_weight`` and ``pd_branch_weight``)
#   * §3.6.1 stochastic 3d_map profile cascade (branch-keyed alternative
#     resolution for ``p_profile_value``)
#
# No defensive gating — helpers either produce the canonical frame or
# return None when the structural source data is genuinely absent (e.g.
# a fixture without commodity-ladder rows).  Architectural shift per the
# task spec: the CSV path is the debug oracle, not a runtime fallback;
# parity test failures are the right signal.
# ===========================================================================


# ---------------------------------------------------------------------------
# §3.17 — Commodity ladder
# ---------------------------------------------------------------------------


def p_f_d_k_from_source(
    source: "InputSource",
    active_solve: str | None,
    workdir: Path | None,
) -> "Param | None":
    """Per-period "fraction realised this solve" for the price-ladder
    feature (audit §3.17.1).

    Algorithm — mirror of
    ``preprocessing/period_calculated_params.py:324-344``::

        f_d_k[d] = (p_ladder_cum_sim_hours[d] + sum_t step_duration[d, t])
                 / (complete_period_share_of_year[d] * 8760)

    Domain: ``period_in_use``.  Inputs:

    * ``solve_data/steps_in_use.csv`` — per-(d, t) step duration sum.
    * ``solve_data/ladder_cum_sim_hours.csv`` — handoff carrier; default
      0.0 when row absent.
    * ``solve_data/complete_period_share_of_year_calc.csv`` — per-period
      hour share of the year (already-computed §3.1.4 frame).

    Returns ``None`` unless the price-ladder feature is active (i.e. at
    least one commodity has ``price_method ∈ {price_ladder_annual,
    price_ladder_cumulative}`` — flextool emits ``f_d_k.csv`` always
    but ``flexpy/_commodity_ladder.py:load`` only consumes it when
    ladder commodities exist; the parity test mirrors that semantic).
    """
    if workdir is None:
        return None
    # Gate on ladder activity — mirrors flexpy's CSV-side behaviour
    # where the loader only emits p_f_d_k when there are ladder
    # commodities.
    sd = Path(workdir) / "solve_data"
    cwl_path = sd / "commodity_with_ladder.csv"
    if cwl_path.exists():
        cwl = _read_csv_file(cwl_path)
        if cwl.height == 0:
            return None
    else:
        # Fall back to source data: any commodity with a non-default
        # price_method counts as ladder-active.
        method = _try_param(source, "commodity", "price_method")
        if method is None or method.height == 0:
            return None
        ladder = method.filter(pl.col("value") != "price")
        if ladder.height == 0:
            return None
    siu = sd / "steps_in_use.csv"
    if not siu.exists():
        return None
    df = _read_csv_file(siu)
    if df.height == 0 or "period" not in df.columns:
        return None
    sum_step = (df.lazy()
                  .with_columns(pl.col("step_duration")
                                  .cast(pl.Float64, strict=False))
                  .group_by("period")
                  .agg(pl.col("step_duration").sum().alias("sum_step"))
                  .rename({"period": "d"}))
    # complete_period_share_of_year[d]
    csy_path = sd / "complete_period_share_of_year_calc.csv"
    if not csy_path.exists():
        # Try the non-_calc variant (some fixtures only emit the latter).
        csy_path = sd / "complete_period_share_of_year.csv"
    if not csy_path.exists():
        return None
    csy = (_read_csv_file(csy_path).lazy()
              .rename({"period": "d"})
              .with_columns(pl.col("value").cast(pl.Float64, strict=False)
                                  .alias("share")))
    # ladder_cum_sim_hours — default 0.0 when absent.
    cum_path = sd / "ladder_cum_sim_hours.csv"
    if cum_path.exists():
        cum_raw = _read_csv_file(cum_path)
        if cum_raw.height > 0 and "p_ladder_cum_sim_hours" in cum_raw.columns:
            cum_lf = (cum_raw.lazy()
                       .rename({"period": "d",
                                "p_ladder_cum_sim_hours": "cum"})
                       .with_columns(pl.col("cum")
                                       .cast(pl.Float64, strict=False)))
        else:
            cum_lf = None
    else:
        cum_lf = None
    # period_in_use as the output domain
    piu_path = sd / "period_in_use_set.csv"
    if not piu_path.exists():
        return None
    piu_lf = (_read_csv_file(piu_path).lazy()
                .rename({"period": "d"})
                .select("d"))
    out_lf = (piu_lf
                .join(sum_step.select("d", "sum_step"), on="d", how="left")
                .join(csy.select("d", "share"), on="d", how="left"))
    if cum_lf is not None:
        out_lf = out_lf.join(cum_lf.select("d", "cum"), on="d", how="left")
        out_lf = out_lf.with_columns(pl.col("cum").fill_null(0.0))
    else:
        out_lf = out_lf.with_columns(cum=pl.lit(0.0))
    out_lf = (out_lf
                .with_columns(pl.col("sum_step").fill_null(0.0))
                .with_columns(value=(pl.col("cum") + pl.col("sum_step"))
                                       / (pl.col("share") * 8760.0))
                .select("d", "value"))
    out = out_lf.collect()
    if out.height == 0:
        return None
    return Param(("d",), out)


def p_ladder_cum_realized_mwh_from_workdir(
    workdir: Path | None,
) -> "Param | None":
    """Per-(c, i, d) cumulative realized MWh handoff carrier
    (audit §3.17.2).

    This is a pure rolling-handoff state — written by the chain runner
    after each solve summing ``v_trade`` per tier into the next solve's
    starting accumulator.  Single-solve fixtures emit a header-only CSV
    (returns None).
    """
    if workdir is None:
        return None
    rel_path = Path(workdir) / "solve_data" / "ladder_cum_realized_mwh.csv"
    if not rel_path.exists():
        return None
    raw = _read_csv_file(rel_path)
    if raw.height == 0:
        return None
    value_col = ("p_ladder_cum_realized_mwh"
                 if "p_ladder_cum_realized_mwh" in raw.columns else "value")
    if value_col not in raw.columns:
        return None
    out = (raw.rename({"commodity": "c", "tier": "i", "period": "d",
                         value_col: "value"})
              .with_columns(pl.col("i").cast(pl.Utf8))
              .with_columns(pl.col("value").cast(pl.Float64, strict=False))
              .select("c", "i", "d", "value"))
    if out.height == 0:
        return None
    return Param(("c", "i", "d"), out)


# ---------------------------------------------------------------------------
# §3.13 — prundt (residual reserve coverage)
# ---------------------------------------------------------------------------


def prundt_from_source(
    source: "InputSource",
    active_solve: str | None,
    dt: pl.DataFrame | None,
) -> pl.DataFrame | None:
    """Cross-product of active reserve relationships × dt
    (audit §3.13.1, mirror of
    ``reserve_calc_params.py:write_process_reserve_upDown_node_active_and_prundt:279-336``).

    flextool's algorithm:

    * Build ``process_reserve_upDown_node_active`` = (p, r, ud, n) tuples
      where ``sum_{(r,ud,g) ∈ reserve__upDown__group, (d,t) ∈ dt}
      pdtReserve_upDown_group[r, ud, g, 'reservation', d, t] != 0``.
    * ``prundt`` = ``process_reserve_upDown_node_active × dt``.

    For the simple non-stochastic case where the
    ``reserve__upDown__group.reservation`` is a scalar/Map: any non-zero
    reservation (anywhere on the (r, ud) cohort) makes every (p, r, ud, n)
    tuple in the matching reserve relationship active — so we can use
    the active-relationship cross-product directly.

    Returns the (p, r, ud, n, d, t) frame.  When dt or active relationships
    are absent, returns None.
    """
    if dt is None or dt.height == 0:
        return None
    pruna = process_reserve_upDown_node_active_from_source(source)
    if pruna is None or pruna.height == 0:
        return None
    # Filter to active reserves: groups whose reservation > 0 anywhere.
    res = _try_param(source, "reserve__upDown__group", "reservation")
    if res is not None and res.height > 0:
        # Take (r, ud) pairs that have any non-zero reservation entry.
        rename: dict[str, str] = {}
        for c in res.columns:
            if c == "reserve":
                rename[c] = "r"
            elif c == "upDown":
                rename[c] = "ud"
        res_lf = (res.lazy().rename(rename)
                     .with_columns(pl.col("value")
                                     .cast(pl.Float64, strict=False)))
        # Sum over the value column per (r, ud) — non-zero ⇒ active.
        active_ru = (res_lf
                       .filter(pl.col("value").fill_null(0.0) != 0.0)
                       .select("r", "ud").unique())
        # Restrict pruna to active (r, ud) pairs.
        pruna_lf = pruna.lazy().join(active_ru, on=["r", "ud"], how="inner")
    else:
        # No reservation parameter at all — fall through with the full
        # active-relationship set; dispatch will be a no-op when no
        # reservation is set, but the index is structurally correct.
        pruna_lf = pruna.lazy()
    out = (pruna_lf
              .join(dt.lazy().select("d", "t"), how="cross")
              .select("p", "r", "ud", "n", "d", "t")
              .sort("p", "r", "ud", "n", "d", "t")
              .collect())
    if out.height == 0:
        return None
    return out


# ---------------------------------------------------------------------------
# §3.15 — Delay
# ---------------------------------------------------------------------------


def _delay_distributions_from_source(
    source: "InputSource",
) -> tuple[dict[tuple[str, str], float],
            set[tuple[str, str]],
            set[str]] | None:
    """Read ``unit.delay`` + ``connection.delay`` from the source and
    classify into:

    * ``weighted`` — (p, td) → weight when value is a 1d_map.  These are
      multi-duration distributions.
    * ``single`` — (p, td) when value is a scalar.  Single delay duration.
    * ``processes`` — set of all delayed processes (union of both).

    Returns ``None`` if no delays are defined.

    flextool's input writer (input_writer.py:638-648) emits the same
    distinction via ``filter_in_type``: 1d_map → ``p_process_delay_weighted``,
    scalar → ``process_delay_single``.
    """
    weighted: dict[tuple[str, str], float] = {}
    single: set[tuple[str, str]] = set()
    processes: set[str] = set()
    found_any = False
    for ec in ("unit", "connection"):
        df = _try_param(source, ec, "delay")
        if df is None or df.height == 0:
            continue
        found_any = True
        cols = df.columns
        # Distinguish 1d_map shape (entity, key, value — three cols) from
        # scalar shape (entity, value — two cols).  The SpineDbReader
        # uses generic key names for 1d_map (typically `constraint` or
        # similar from the shape registry); we detect by column count.
        if len(cols) >= 3:
            # 1d_map: name, <key>, value.
            key_col = next((c for c in cols
                              if c not in ("name", "value")),
                            None)
            if key_col is None:
                continue
            for row in df.iter_rows(named=True):
                p = row["name"]
                kv = row[key_col]
                v = row["value"]
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                # The CSV writer formats the delay duration via
                # repr(float(x)), so the string key matches when we
                # round-trip via float.
                try:
                    td = repr(float(kv))
                except (TypeError, ValueError):
                    td = str(kv)
                weighted[(p, td)] = fv
                processes.add(p)
        else:
            # Scalar entry — value IS the delay duration.
            for row in df.iter_rows(named=True):
                p = row["name"]
                v = row["value"]
                if v is None:
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                # 0 and negative durations don't activate delay.
                if fv <= 0.0:
                    continue
                td = repr(fv)
                single.add((p, td))
                processes.add(p)
    if not found_any:
        return None
    return weighted, single, processes


def p_process_delay_weight_from_source(
    source: "InputSource",
) -> "Param | None":
    """Per-(p, td) delay weight — audit §3.15.2.

    Algorithm — mirror of
    ``preprocessing/process_arc_unions.py:write_p_process_delay_weight:2105-2141``::

        p_process_delay_weight[p, td] = 1.0 if (p, td) ∈ process_delay_single
                                         else p_process_delay_weighted[p, td]

    The set ``process_delayed__duration`` is the union of the two sources;
    every row in that set produces an output value.

    Returns ``None`` when no delays are defined (no fixture rows on
    either source).
    """
    info = _delay_distributions_from_source(source)
    if info is None:
        return None
    weighted, single, _ = info
    rows: list[tuple[str, float, float]] = []
    # Order: weighted entries first, then single-only (mirrors flextool's
    # union which preserves first-occurrence order across the two sources).
    seen: set[tuple[str, str]] = set()
    for (p, td), v in weighted.items():
        rows.append((p, float(td), 1.0 if (p, td) in single else v))
        seen.add((p, td))
    for (p, td) in single:
        if (p, td) not in seen:
            rows.append((p, float(td), 1.0))
    if not rows:
        return None
    out = pl.DataFrame(rows, schema={"p": pl.Utf8, "td": pl.Float64,
                                       "value": pl.Float64},
                       orient="row")
    return Param(("p", "td"), out)


def dtt__delay_duration_from_source(
    source: "InputSource",
    dt: pl.DataFrame | None,
) -> pl.DataFrame | None:
    """Per-(d, t_source, t_sink, td) delay shift map — audit §3.15.1.

    Algorithm — mirror of
    ``flextoolrunner/solve_writers.py:write_delayed_durations:742-771``.
    For each delay duration ``td`` and each (d, t_source) in dt, produce
    one row (d, t_source, t_sink, td) where ``t_sink = t_source + td``
    timesteps within the period.  Wraps at period boundaries::

        if k + td_int < len(time_steps):
            t_sink = time_steps[k + td_int]
        else:
            t_sink = time_steps[k - len(time_steps) + td_int]

    flextool casts ``td`` through ``int(float(delay_duration))`` so the
    csv-string "4.0" becomes shift index 4.

    Inputs: ``dt`` (period-time index) + the union of all delay durations
    from ``unit.delay`` and ``connection.delay``.

    Returns ``None`` when no delays are defined or dt is empty.
    """
    if dt is None or dt.height == 0:
        return None
    info = _delay_distributions_from_source(source)
    if info is None:
        return None
    weighted, single, _ = info
    # Distinct delay duration set (string keys, as written by the CSV).
    durations: set[str] = set()
    for (_, td) in weighted:
        durations.add(td)
    for (_, td) in single:
        durations.add(td)
    if not durations:
        return None
    # Build per-period ordered timestep lists from dt.
    dt_sorted = (dt.lazy().sort("d", "t")
                    .group_by("d", maintain_order=True)
                    .agg(pl.col("t").alias("steps"))
                    .collect())
    rows: list[tuple[str, str, str, float]] = []
    for row in dt_sorted.iter_rows(named=True):
        d = row["d"]
        steps: list[str] = list(row["steps"])
        n = len(steps)
        if n == 0:
            continue
        for td in durations:
            try:
                td_int = int(float(td))
                td_float = float(td)
            except (TypeError, ValueError):
                continue
            for k, t_src in enumerate(steps):
                if k + td_int < n:
                    t_sink = steps[k + td_int]
                else:
                    t_sink = steps[k - n + td_int]
                rows.append((d, t_src, t_sink, td_float))
    if not rows:
        return None
    out = pl.DataFrame(rows, schema={"d": pl.Utf8, "t_source": pl.Utf8,
                                       "t_sink": pl.Utf8, "td": pl.Float64},
                       orient="row")
    return out


# ---------------------------------------------------------------------------
# §3.18 — Multi-branch normalisation (full cascade)
# ---------------------------------------------------------------------------


def _read_period_branch_pairs(workdir: Path | None
                                  ) -> list[tuple[str, str]]:
    """Read ``solve_data/period__branch.csv`` as a list of (d2, b)
    pairs — d2 is the parent period, b is the sibling branch.
    """
    if workdir is None:
        return []
    p = Path(workdir) / "solve_data" / "period__branch.csv"
    if not p.exists():
        return []
    df = _read_csv_file(p)
    if df.height == 0:
        return []
    cols = df.columns
    if "period" in cols and "branch" in cols:
        d_col, b_col = "period", "branch"
    elif len(cols) >= 2:
        d_col, b_col = cols[0], cols[1]
    else:
        return []
    return [(str(r[d_col]), str(r[b_col])) for r in df.iter_rows(named=True)]


def _read_solve_branch_weights(workdir: Path | None
                                   ) -> dict[str, float]:
    """Read ``solve_data/solve_branch_weight.csv`` as branch → weight."""
    out: dict[str, float] = {}
    if workdir is None:
        return out
    p = Path(workdir) / "solve_data" / "solve_branch_weight.csv"
    if not p.exists():
        return out
    df = _read_csv_file(p)
    if df.height == 0:
        return out
    cols = df.columns
    b_col = "branch" if "branch" in cols else cols[0]
    v_col = "value" if "value" in cols else (
        cols[1] if len(cols) > 1 else None)
    if v_col is None:
        return out
    for r in df.iter_rows(named=True):
        b = str(r[b_col])
        try:
            out[b] = float(r[v_col])
        except (TypeError, ValueError):
            continue
    return out


def _read_first_timesteps(workdir: Path | None
                              ) -> dict[str, str]:
    """Read ``solve_data/first_timesteps.csv`` as period → first step."""
    out: dict[str, str] = {}
    if workdir is None:
        return out
    p = Path(workdir) / "solve_data" / "first_timesteps.csv"
    if not p.exists():
        return out
    df = _read_csv_file(p)
    if df.height == 0:
        return out
    cols = df.columns
    d_col = "period" if "period" in cols else cols[0]
    s_col = ("step" if "step" in cols else
             ("time" if "time" in cols else cols[1]))
    for r in df.iter_rows(named=True):
        out[str(r[d_col])] = str(r[s_col])
    return out


def pd_branch_weight_full_from_source(
    source: "InputSource",
    active_solve: str | None,
    dt: pl.DataFrame | None,
    workdir: Path | None,
) -> "Param | None":
    """Per-period branch weight — full multi-branch cascade.

    Δ.8 consolidation: delegates to the lazy port
    :func:`._derived_branch.pd_branch_weight_param`.  Behaviour
    preserved: defaults to 1.0 when no ``period__branch`` rows exist
    (deterministic fixtures), normalises across siblings otherwise.

    See :mod:`._derived_branch` for the algorithm reference.
    """
    from flextool.engine_polars._derived_branch import pd_branch_weight_param
    out = pd_branch_weight_param(workdir, source, active_solve)
    if out is not None:
        return out
    # Match the historical "no period_in_use, but dt present →
    # deterministic 1.0 per realised period" fallback path.  This
    # keeps parity with the eager helper for chain-runner-less single
    # solves where the workdir CSVs may be absent but dt is built.
    if dt is None or dt.height == 0:
        return None
    df = (dt.lazy()
              .select("d").unique()
              .with_columns(value=pl.lit(1.0))
              .sort("d")
              .collect())
    if df.height == 0:
        return None
    return Param(("d",), df)


def pdt_branch_weight_full_from_source(
    source: "InputSource",
    active_solve: str | None,
    dt: pl.DataFrame | None,
    workdir: Path | None,
) -> "Param | None":
    """Per-(d, t) branch weight — full multi-branch cascade.

    Δ.8 consolidation: delegates to the lazy port
    :func:`._derived_branch.pdt_branch_weight_param`.  Behaviour
    preserved: dense over ``dt`` (every (d, t) gets a value;
    denominator-zero rows fall through to 1.0).

    See :mod:`._derived_branch` for the algorithm reference.
    """
    from flextool.engine_polars._derived_branch import pdt_branch_weight_param
    return pdt_branch_weight_param(workdir, source, active_solve, dt)


def apply_synthetic_invest_sets(flex_data: object,
                                     source: "InputSource",
                                     active_solve: str,
                                     synthetic: tuple[str, str],
                                     workdir: "Path | None" = None) -> None:
    """Δ.19 — populate the invest-set FlexData fields for a synthetic
    ``<base>_<anchor>`` solve, replacing the matching disk reads in
    ``_invest_seeds.py``.

    The synthetic-aware :func:`_solve_periods` returns the right period
    list for the anchor (Map-shaped ``invest_periods`` filtered to the
    synthetic anchor); the eager helpers in this module and the lazy
    partition helpers in :mod:`._derived_existing` are pure derivations
    on top of that period list so they work as-is.  Fields populated
    (mirrors the seed names in ``_invest_seeds.py``):

      * ``ed_invest_set`` / ``ed_divest_set``
      * ``pd_invest_set`` / ``nd_invest_set`` (process / node partition
        of ``ed_invest_set``)
      * ``pd_divest_set`` / ``nd_divest_set`` (process / node partition
        of ``ed_divest_set``)
      * ``edd_invest_set`` (history-extended triple; requires workdir
        for ``period_with_history.csv`` + ``period_in_use_set.csv``).
      * ``ed_invest_forbidden_no_investment`` (no-investment lifetime
        gate; needed by ``apply_existing_chain``).

    The per-period cap subsets (``ed_invest_period_set`` /
    ``ed_divest_period_set``) and the NPV cost cascade
    (``ed_lifetime_fixed_cost`` etc.) are NOT touched — they stay on
    the ``_invest_seeds.py`` / ``_load_invest`` CSV-seed path because
    those values bake in multi-year discounting and the per-sub-solve
    filter doesn't compose cleanly with the existing derivation
    (deferred per the Gap D dispatch).
    """
    base, anchor = synthetic
    # ed_invest_set / ed_divest_set — eager helpers use _solve_periods
    # which is synthetic-aware (Δ.19); pass the synthetic name through.
    # Γ.6.D forbidden filter is applied internally by the helper.
    ed_inv = ed_invest_set_from_source(source, active_solve, workdir=workdir)
    if ed_inv is not None and ed_inv.height > 0:
        flex_data.ed_invest_set = ed_inv
    ed_div = ed_divest_set_from_source(source, active_solve)
    if ed_div is not None and ed_div.height > 0:
        flex_data.ed_divest_set = ed_div

    # ed_invest_forbidden_no_investment — derived from the lifetime
    # gate; consumed by apply_existing_chain.
    try:
        forbidden = ed_invest_forbidden_no_investment_from_source(
            source, active_solve, workdir, ed_inv)
    except Exception:  # pragma: no cover — defensive
        forbidden = None
    if forbidden is not None and forbidden.height > 0:
        flex_data.ed_invest_forbidden_no_investment = forbidden

    # pd/nd_invest_set, pd/nd_divest_set partitions — pure entity-class
    # projections, no solve dependency beyond the input ed_*_set frame.
    from flextool.engine_polars._derived_existing import (
        pd_invest_set_lf as _pd_invest_lf,
        nd_invest_set_lf as _nd_invest_lf,
        pd_divest_set_lf as _pd_divest_lf,
        nd_divest_set_lf as _nd_divest_lf,
        edd_invest_set_lf as _edd_invest_lf,
    )
    if ed_inv is not None and ed_inv.height > 0:
        ed_inv_lf = ed_inv.lazy()
        pd_inv = _pd_invest_lf(source, ed_inv_lf).collect()
        if pd_inv.height > 0:
            flex_data.pd_invest_set = pd_inv
        nd_inv = _nd_invest_lf(source, ed_inv_lf).collect()
        if nd_inv.height > 0:
            flex_data.nd_invest_set = nd_inv
    if ed_div is not None and ed_div.height > 0:
        ed_div_lf = ed_div.lazy()
        pd_div = _pd_divest_lf(source, ed_div_lf).collect()
        if pd_div.height > 0:
            flex_data.pd_divest_set = pd_div
        nd_div = _nd_divest_lf(source, ed_div_lf).collect()
        if nd_div.height > 0:
            flex_data.nd_divest_set = nd_div

    # edd_invest_set — history-extended triple.  Uses workdir CSV for
    # period_with_history + period_in_use_set (both anchor-specific in
    # the synthetic-solve snapshot).
    if (ed_inv is not None and ed_inv.height > 0
            and workdir is not None):
        period_in_use = _period_in_use_set(source, active_solve, workdir)
        period_with_history = (_read_period_with_history(workdir)
                                  or list(period_in_use))
        try:
            edd_inv = _edd_invest_lf(
                source, active_solve, ed_inv.lazy(),
                period_with_history, period_in_use, workdir).collect()
        except Exception:  # pragma: no cover — defensive
            edd_inv = None
        if edd_inv is not None and edd_inv.height > 0:
            flex_data.edd_invest_set = edd_inv


# ---------------------------------------------------------------------------
# Public — Γ.3.G field list (for selective scope-checking).
# ---------------------------------------------------------------------------


G_PUBLIC_FIELDS: tuple[str, ...] = (
    "p_f_d_k",
    "p_ladder_cum_realized_mwh",
    "prundt",
    "dtt__delay_duration",
    "p_process_delay_weight",
    "pd_branch_weight",
    "pdt_branch_weight",
)


def apply_derived_g(
    flex_data: object,
    source: "InputSource",
    workdir: Path,
    *,
    ctx: "SolveContext | None" = None,
) -> None:
    """Apply Γ.3.G residual Derived Params, mutating ``flex_data``
    in place.

    Scope:
      * §3.17 commodity ladder — ``p_f_d_k``,
        ``p_ladder_cum_realized_mwh``.
      * §3.13 reserves — ``prundt``.
      * §3.15 delay — ``dtt__delay_duration``, ``p_process_delay_weight``.
      * §3.18 multi-branch normalisation — full cascade for
        ``pd_branch_weight`` and ``pdt_branch_weight``.

    Δ.3 replaced the previous ``derived_overrides_g`` dict-return;
    Δ.4 deleted the deprecated wrapper alias.
    """
    active_solve = ctx.solve_name if ctx is not None else _read_active_solve(workdir)
    dt = getattr(flex_data, "dt", None)

    # Δ.12b — assignment is unconditional; helpers are authoritative
    # producers.  ``None`` is the explicit "feature inactive" signal
    # (no commodity-ladder commodities, no reserve relationships,
    # no delayed processes, etc.).  Hard exceptions propagate.

    # ─── §3.17.1 p_f_d_k ───────────────────────────────────────────
    flex_data.p_f_d_k = p_f_d_k_from_source(source, active_solve, workdir)

    # ─── §3.17.2 p_ladder_cum_realized_mwh ─────────────────────────
    flex_data.p_ladder_cum_realized_mwh = (
        p_ladder_cum_realized_mwh_from_workdir(workdir))

    # ─── §3.13.1 prundt ────────────────────────────────────────────
    flex_data.prundt = prundt_from_source(source, active_solve, dt)

    # ─── §3.15.1 dtt__delay_duration ───────────────────────────────
    flex_data.dtt__delay_duration = dtt__delay_duration_from_source(source, dt)

    # ─── §3.15.2 p_process_delay_weight ────────────────────────────
    flex_data.p_process_delay_weight = p_process_delay_weight_from_source(source)

    # ─── §3.18 cluster D — multi-branch normalisation + non-anticipativity ──
    # Δ.8 consolidation: delegate the full cluster D port to
    # ``_derived_branch.apply_branch_cluster``.  The helper covers
    # ``pd_branch_weight``, ``pdt_branch_weight``, and (already
    # CSV-loaded by ``input.py``) ``period_branch_full``,
    # ``period_in_use_set``, ``dt_non_anticipativity`` — the latter
    # three are kept as overrides so the lazy port becomes the single
    # producer once the CSV cascade retires (Δ.12).  R-O6: the helper
    # never touches ``invest_periods`` / ``v_invest``.
    # Δ.12b: defensive try/except removed; apply_branch_cluster is
    # parity-bound (cluster D tests gate it).
    from flextool.engine_polars._derived_branch import apply_branch_cluster
    apply_branch_cluster(flex_data, source, workdir, active_solve)


