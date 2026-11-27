"""Γ.3.A — time and weighting (§3.1, §3.18): ``dt`` / ``p_step_duration``, ``p_period_share``,
``p_window_fraction``, the bookkeeping periods, ``p_years_represented_d``, ``p_timestep_weight``, the branch
weights, and :func:`apply_derived_a`."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from polar_high import Param

from flextool.engine.input._axis_enums import (
    alias_to_axis,
    cast_dim,
    cast_frame_axes,
    get_global_axis_enums,
    rename_to_axis,
)
from flextool.engine.input._constants import HOURS_PER_YEAR
from flextool.engine.preprocess.params._branch import (
    inherit_active_branch_periods,
    period_branch_pairs_lf,
    period_in_use_set_lf,
)
from flextool.engine.preprocess.params._per_solve_sets import (
    solve_period_timeset_frame,
    timeset_duration_lf,
    timeset_timeline_lf,
)
from flextool.engine.input._solve_step_aggregation import aggregate_if_active
from flextool.engine.input._param_shapes import resolve_param_shape

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource
    from flextool.engine.preprocess._solve_context import SolveContext

from flextool.engine.preprocess.params._derived._scope import (
    _ctx_read,
    _labels_of_pairs,
    _provider_has_key,
    _provider_read,
    _read_active_solve,
    _try_param,
    _try_param_explicit,
)
from flextool.engine.preprocess.params._derived._inflation import (
    _years_for_period_from_source,
    p_inflation_op_from_source,
)
from flextool.engine.preprocess.params._derived._existing_capacity import p_process_existing_count_from_source


# ---------------------------------------------------------------------------
# §3.1.1 — dt + p_step_duration
# ---------------------------------------------------------------------------


def dt_and_step_duration_from_source(
    source: "InputSource",
    active_solve: str,
    workdir: Path | None = None,
    *,
    ctx: "SolveContext | None" = None,
    provider: "object | None" = None,
) -> tuple[pl.DataFrame, Param] | None:
    """Compute the per-(d, t) ``dt`` set and ``p_step_duration`` Param
    for the active solve.

    Algorithm (audit §3.1.1):

      1. ``solve.realized_periods`` → list of periods active in this solve.
      2. ``solve.period_timeset[active_solve, period]`` → timeset for each
         (or the Rule 1 / 5b default, :func:`solve_period_timeset_frame`).
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
        piu = _ctx_read(ctx, workdir, "period_in_use_set.csv", provider=provider)
        if piu is not None and piu.height > 0 and "period" in piu.columns:
            realized_p = (piu.lazy()
                            .select(alias_to_axis("period", "d"))
                            .unique())
    if realized_p is None:
        parts: list[pl.LazyFrame] = []
        _live = get_global_axis_enums()
        for ec, par in (("solve", "realized_periods"),
                         ("solve", "invest_periods")):
            df = _try_param(source, ec, par)
            if df is None:
                continue
            parts.append(df.lazy()
                           .filter(pl.col("name") == active_solve)
                           .select(alias_to_axis("value", "d")))
        if not parts:
            return None
        realized_p = pl.concat(parts).unique()

    # period → timeset of the active solve: ``solve.period_timeset``, or the
    # Rule 1 / 5b default of ``create_assumptive_parts`` (item #175).
    pt_frame = solve_period_timeset_frame(source, active_solve)
    if pt_frame is None:
        return None
    pt = pt_frame.lazy().select(alias_to_axis("d", "d"), pl.col("ts"))
    # Broadcast timesets to stochastic-branch periods: when a period
    # appears in period_in_use_set without a direct period_timeset row,
    # use the anchor's timeset via ``period__branch.csv`` (anchor → branch
    # map; flextool's per_solve_sets.py:65-95 does this implicitly by
    # writing per-branch steps_in_use rows under the anchor's timeset).
    pb_raw = _ctx_read(ctx, workdir, "period__branch.csv", provider=provider) if workdir is not None or ctx is not None else None
    if pb_raw is not None and pb_raw.height > 0:
        pb_lf = (pb_raw.lazy()
                    .pipe(rename_to_axis, {"period": "anchor", "branch": "d"})
                    # Cross-axis-by-value compare: "anchor" and "d"
                    # both resolve to the canonical d (period) enum, so
                    # the Enum vocabularies match.  Cast to Utf8 here
                    # explicitly so the equality test stays robust if
                    # the vocabularies ever diverge (e.g. branch axis
                    # gains a stochastic vocabulary distinct from d).
                    .filter(pl.col("anchor").cast(pl.Utf8)
                            != pl.col("d").cast(pl.Utf8)))
        # Map branch d → anchor's timeset.  Pre-snap both branches'
        # dim-column dtypes to the canonical Enum so the concat below
        # sees identical schemas (Polars's UNION schema-resolution
        # otherwise drops one side to Utf8, triggering a SchemaError
        # at the downstream join).
        anchor_ts = (pt.rename({"d": "anchor"})
                        .join(pb_lf, on="anchor", how="inner")
                        .select("d", "ts"))
        _live = get_global_axis_enums()
        # Materialise both halves eagerly before concat so Polars's
        # lazy union schema-resolver doesn't see two large optimiser
        # contexts at once — that path triggers an internal
        # str-vs-Enum mis-resolution on the ``d`` column when both
        # halves transitively reference ``period_timeset`` with cast
        # nodes.  Eager concat + re-cast yields the canonical Enum
        # dtype reliably; the upstream cost is negligible (pt + anchor
        # are O(num_periods) rows).
        pt_e = pt.collect()
        at_e = anchor_ts.collect()
        if _live is not None:
            pt_e = cast_frame_axes(pt_e, _live)
            at_e = cast_frame_axes(at_e, _live)
        pt = pl.concat([pt_e, at_e]).unique().lazy()
        # A period label (the original name of a period that opens a fan) is in period_in_use_set for the
        # period-level bookkeeping but has no dispatch: it gets no (d, t) rows.
        labels = _labels_of_pairs(pb_raw["period"].to_list(), pb_raw["branch"].to_list())
        if labels:
            realized_p = realized_p.filter(~pl.col("d").cast(pl.Utf8).is_in(sorted(labels)))

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
                        alias_to_axis(tl_step_col, "t"),
                        pl.col("value").cast(pl.Float64).alias("step_duration")))
    # Determine step rank within each timeline (sort by t lex order, which
    # matches t0001 < t0002 < ... in flextool's canonical naming).
    tl = tl.sort(["timeline", "t"]) \
            .with_columns(rank=pl.col("t").cum_count().over("timeline"))

    # ``timeset.timeline`` / ``timeset.timeset_duration``, through the
    # shared single-timeline defaults (item #171, Rules 2/3 — see
    # ``_per_solve_sets.timeset_timeline_lf`` / ``timeset_duration_lf``).
    # ``timesets`` is the set of timesets the active solve's (branch-
    # broadcast) ``period_timeset`` actually names.
    timesets = pt.select("ts").unique()
    ttl = timeset_timeline_lf(source, timesets, tl)
    if ttl is None:
        return None

    ts_dur_lf = timeset_duration_lf(source, timesets, tl)
    if ts_dur_lf is None:
        return None
    # ``start_step`` joins against ``tl.t`` which is t-Enum under
    # activation.  The shared helper returns Utf8; cast explicitly to
    # t-Enum and alias to "start_step" (caller-side cast, per design).
    _live_enums = get_global_axis_enums()
    blocks = ts_dur_lf.select(
        pl.col("ts"),
        cast_dim(pl.col("start_step"), _live_enums, "t").alias("start_step"),
        pl.col("count"),
    )

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
    """``p_period_share[d] = sum_{t in dt[d]} step_duration[d,t] / 8760`` — the
    fraction of a full year covered by this period in the active solve.

    Audit §3.1.4: built from ``dt`` and ``p_step_duration``, over the steps of
    ``dt`` only (``step_dur`` rows outside ``dt`` are ignored; the keys are
    matched as strings so Enum and Utf8 key columns align).
    """
    if dt is None or step_dur is None:
        return None
    keys = dt.select(pl.col("d").cast(pl.Utf8).alias("_d"),
                     pl.col("t").cast(pl.Utf8).alias("_t")).unique()
    sd = (step_dur.frame
          .with_columns(pl.col("d").cast(pl.Utf8).alias("_d"),
                        pl.col("t").cast(pl.Utf8).alias("_t"))
          .join(keys, on=["_d", "_t"], how="semi", maintain_order="left")
          .drop("_d", "_t"))
    out = (sd.lazy()
              .group_by("d")
              .agg((pl.col("value").cast(pl.Float64).sum() / HOURS_PER_YEAR)
                   .alias("value"))
              .sort("d")
              .collect())
    if out.height == 0:
        return None
    return Param(("d",), out)


def p_window_fraction_from_shares(
    dt: pl.DataFrame,
    step_dur: Param,
    period_share: Param,
) -> Param | None:
    """``p_window_fraction[d] = (Σ_{t in dt[d]} step_duration[d,t] / 8760) / p_period_share[d]``.

    The share of each period's complete-solve hours that the sub-solve's own ``dt`` covers (the
    window sum runs over the steps of ``dt`` only, see :func:`p_period_share_from_source`).  In a
    roll of a rolling solve ``p_period_share`` is the complete (parent) solve's share, so the
    fraction is the roll's window over the complete solve, per period.  A value within 1e-9
    (relative) of 1 is snapped to exactly 1.0, so a sub-solve that covers its periods whole keeps
    the ``co2_max_period`` RHS bit for bit.  A dt period without a ``p_period_share`` row raises: the
    share is inconsistent with dt (a loader bug, or a reloaded workdir whose share file lacks the
    period), and the operational costs (``op_factor``) and the cap's RHS of that period would be lost.
    """
    window = p_period_share_from_source(None, dt, step_dur)
    if window is None:
        return None
    share = period_share.frame.select(
        pl.col("d").cast(pl.Utf8).alias("_d"), pl.col("value").cast(pl.Float64).alias("_share"))
    out = (window.frame.with_columns(pl.col("d").cast(pl.Utf8).alias("_d"))
           .join(share, on="_d", how="left"))
    missing = out.filter(pl.col("_share").is_null())["_d"].to_list()
    if missing:
        raise ValueError(
            f"p_window_fraction: p_period_share has no row for dt periods {missing}; the period share (the "
            f"complete solve's share in a sub-solve, complete_period_share_of_year_calc.csv in a reloaded workdir) "
            f"must cover every dt period, else that period's operational costs and co2_max_period cap lose "
            f"their weight.")
    out = out.with_columns(
        value=pl.when((pl.col("value") - pl.col("_share")).abs() <= 1e-9 * pl.col("value").abs())
        .then(pl.lit(1.0, dtype=pl.Float64))
        .otherwise(pl.col("value") / pl.col("_share")))
    return Param(("d",), out.select("d", "value").sort("d"))


# ---------------------------------------------------------------------------
# p_years_represented_d — per-period R width sum
# ---------------------------------------------------------------------------


def periods_of_dt(dt: "pl.DataFrame | None") -> list[str]:
    """The distinct periods of ``dt`` (column ``d``) as strings, in dt order."""
    if dt is None or dt.height == 0:
        return []
    return [str(x) for x in dt["d"].unique(maintain_order=True).to_list()]


def bookkeeping_periods(dt: "pl.DataFrame | None", workdir, *, ctx=None,
                        provider: "object | None" = None) -> list[str]:
    """The periods of ``dt`` (in dt order) followed by the solve's period labels (sorted).

    A period label (the original name of a period that opens a stochastic fan) has no (d, t) rows but is a period
    of the solve for period-level bookkeeping (years, investment costs).
    """
    dt_periods = periods_of_dt(dt)
    pb = period_branch_pairs_lf(workdir, ctx=ctx, provider=provider).collect()
    labels = _labels_of_pairs(pb["d"].to_list(), pb["b"].to_list())
    if not labels:
        return dt_periods
    piu = {str(d) for d in period_in_use_set_lf(workdir, ctx=ctx, provider=provider).collect()["d"].to_list()}
    return dt_periods + sorted(d for d in labels if d in piu and d not in dt_periods)


def p_years_represented_d_from_source(
    source: "InputSource",
    active_solve: str | None,
    period_set: list[str],
    *,
    provider: "object | None" = None,
) -> Param | None:
    """Per-period ``years_represented_d``: sum of per-year widths
    bound to each period (the ``R`` value flextool's
    ``solve.years_represented`` Map carries — e.g. 5.0 for a 5-year
    investment period).

    ``period_set`` is the periods of the solve's ``dt``, in dt order.
    With ``provider``, the widths come from the canonical
    ``solve_data/p_years_represented.csv``, which gives every stochastic
    branch period its anchor period's years and, on a rolling or
    synthetic sub-solve, the parent solve's years; the source arm is the
    fallback (see ``_years_for_period_from_source``).

    Returns ``None`` when no period set is available; otherwise emits
    one row per period.
    """
    if not period_set:
        return None
    yfp = _years_for_period_from_source(source, active_solve, period_set,
                                        provider=provider)
    rows = []
    for d in period_set:
        widths = yfp.get(d, [])
        # Empty period — emit 0.0 (matches legacy ``sum(())``); upstream
        # consumers (out_costs) multiply by emissions so a 0 period
        # contributes nothing.  This hits periods explicitly set to R=0
        # (and periods the canonical CSV omits for that reason); the
        # source arm gives a period absent from ``years_represented`` a
        # single (label, 1.0) row.
        s = sum(w for _y, w in widths)
        rows.append({"d": str(d), "value": float(s)})
    out = pl.DataFrame(rows, schema={"d": pl.Utf8, "value": pl.Float64})
    if out.height == 0:
        return None
    return Param(("d",), out)


# ---------------------------------------------------------------------------
# §3.1.2 — p_timestep_weight
# ---------------------------------------------------------------------------


def solve_has_rp_timeset(source: "InputSource", active_solve: str | None) -> bool:
    """True when a timeset in ``active_solve``'s ``period_timeset`` has ``representative_period_weights``.

    Such a solve's per-step weights come from the writer's ``solve_data/timestep_weight`` frame
    (``_solve_writers._compute_rp_frames``): the folded representative-period weights of its rp periods and
    the ``timeset_weights`` rows of its other periods, so no derived value may replace them.
    """
    if active_solve is None:
        return False
    rpw = _try_param_explicit(source, "timeset", "representative_period_weights")
    if rpw is None or rpw.height == 0 or "name" not in rpw.columns:
        return False
    pt = solve_period_timeset_frame(source, active_solve)
    if pt is None:
        return False
    active_ts = set(pt.get_column("ts").cast(pl.Utf8).to_list())
    rp_ts = {str(n) for n in rpw.get_column("name").to_list()}
    return bool(active_ts & rp_ts)


def p_timestep_weight_from_source(
    source: "InputSource",
    dt: pl.DataFrame,
    active_solve: str | None = None,
) -> Param | None:
    """Per-(d, t) representative-period cost weight.

    Audit §3.1.2: defaults to 1.0 per (d, t); explicit overrides come
    from solve-level ``timeset_weights`` (flat per-step weighting,
    non-RP) and ``representative_period_weights`` (nested base→rep
    Map, RP scenarios) in flextool.

    Algorithm (mirrors
    :func:`._solve_writers.emit_timeset_cost_weight` for the non-RP
    path and :func:`._solve_writers.emit_rp_data` for the RP path):

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
      The scale is per period name.  A split realization roll is a
      synthetic sub-solve (no derived pass), so its emitted weights, one
      scale over the label's jump plus a tail, are the ones used.

    RP path (``representative_period_weights`` set):
      Representative-period weights are folded into
      ``solve_data/timestep_weight.csv`` by the writer
      (:func:`._solve_writers._compute_rp_frames`,
      ``w_r[rep]=Σ_base·n_rp/n_base``) and loaded into
      ``flex_data.p_timestep_weight`` by the CSV loader
      (``flex_data/_load_topology.py`` ``_load_time``).  When the ACTIVE solve's
      ``period_timeset`` references a timeset carrying
      ``representative_period_weights``, this helper returns ``None`` so
      the call site (guarded) PRESERVES that folded CSV value instead of
      clobbering it with the dense 1.0 default.  This keeps a single
      source of truth (``_compute_rp_frames``) shared by every consumer
      — objective, outputs, inflow scaling — rather than re-folding the
      weights against ``source`` here.

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
    # Defensive re-cast: ensure d/t are canonical Enum on the Param.
    default_lf = (dt.lazy()
                    .select(alias_to_axis("d", "d"), alias_to_axis("t", "t"))
                    .with_columns(value=pl.lit(1.0).cast(pl.Float64))
                    .sort("d", "t"))

    # Discover ``solve.period_timeset`` and ``timeset.timeset_weights``;
    # if either is unavailable / empty, fall through to the trivial
    # default.
    if active_solve is None:
        out = default_lf.collect()
        return Param(("d", "t"), out) if out.height > 0 else None

    # period → timeset of the active solve: ``solve.period_timeset``, or the Rule 1 / 5b default
    # (item #175) -- so the sole timeset's ``timeset_weights`` apply without an authored mapping.
    pt = solve_period_timeset_frame(source, active_solve)
    if pt is None:
        out = default_lf.collect()
        return Param(("d", "t"), out) if out.height > 0 else None
    # A stochastic branch period is a copy of its anchor period: it takes the anchor's timeset (and so
    # its timeset_weights) through the active branch-anchor pairs, like every period-keyed input.
    period_timeset = inherit_active_branch_periods(
        pt.select(pl.col("d").cast(pl.Utf8), pl.col("ts")),
        group_keys=(), period_col="d")
    period_timeset_lf = period_timeset.lazy().select(alias_to_axis("d", "d"), pl.col("ts"))

    # ``timeset.timeset_weights``: a ``time`` Map (index_name ``time`` / ``t``, or the silent ``x``),
    # resolved and validated by the shape registry; any other shape raises.
    resolved_tw = resolve_param_shape(source, "timeset", "timeset_weights")
    tw = resolved_tw.frame if resolved_tw is not None else None
    if tw is not None and tw.height > 0:
        # A solve with new_stepduration: one weight per solve step, the step mean (_solve_step_aggregation).
        tw = aggregate_if_active(
            tw.select("name", resolved_tw.time_index_column, pl.col("value").cast(pl.Float64)),
            ("timeset", "timeset_weights"), time_col=resolved_tw.time_index_column, steps=dt.get_column("t"))
    if tw is None or tw.height == 0:
        # No flat ``timeset_weights`` — but this may be an RP solve.
        # ``representative_period_weights`` are folded into
        # ``solve_data/timestep_weight.csv`` by the writer
        # (``_solve_writers._compute_rp_frames``) and already loaded
        # into ``flex_data.p_timestep_weight`` by the CSV loader.  If the
        # ACTIVE solve's ``period_timeset`` references a timeset carrying
        # RP weights, return None so the guarded call site preserves that
        # folded value instead of clobbering it with the dense 1.0
        # default.  Filter to the active timesets so a stray RP param on
        # an UNUSED timeset cannot suppress the legitimate dense default
        # for a genuinely non-RP solve.
        rpw = _try_param_explicit(
            source, "timeset", "representative_period_weights")
        if rpw is not None and rpw.height > 0 and "name" in rpw.columns:
            active_ts = (period_timeset_lf
                         .select(pl.col("ts"))
                         .unique()
                         .collect()
                         .get_column("ts")
                         .to_list())
            rp_ts = set(rpw.get_column("name").to_list())
            if any(ts in rp_ts for ts in active_ts):
                return None
        out = default_lf.collect()
        return Param(("d", "t"), out) if out.height > 0 else None
    # Resolved ``timeset_weights`` frame: name (timeset), t (timestep), value (weight).
    weights_lf = (tw.lazy()
                    .select(pl.col("name").alias("ts"),
                            alias_to_axis(resolved_tw.time_index_column, "t"),
                            pl.col("value").cast(pl.Float64).alias("w_raw")))

    # Per-(d, t) raw weight: lookup via period_timeset → timeset_weights.
    # Periods whose timeset has weights produce non-null ``w_raw``;
    # periods without keep null and fall back to the trivial default 1.0.
    dt_lf = dt.lazy().select("d", "t")
    # Align the CSV-fresh helper frames to dt's dim dtypes (the loader
    # cast dt to Enum end-of-load).
    from flextool.engine.input._axis_enums import align_join_dtypes
    dt_lf, period_timeset_lf = align_join_dtypes(
        dt_lf, period_timeset_lf, ["d"])
    dt_lf, weights_lf = align_join_dtypes(dt_lf, weights_lf, ["t"])
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
    """Per-period branch weight — dense 1.0 baseline (superseded later).

    Returns the trivial 1.0 default per realized period.  This is the
    baseline assigned in ``apply_derived_a``; it is the final value only
    for deterministic / non-stochastic fixtures (where it matches the
    .mod's ``param pd_branch_weight`` default).

    For stochastic fixtures it is SUPERSEDED downstream: ``apply_derived_g``
    calls ``_branch.apply_branch_cluster``, whose
    ``pd_branch_weight_lf`` reads ``solve_branch_weight`` /
    ``period__branch`` and writes the real sibling-normalised weights
    (each branch's input weight divided by the sum across siblings that
    share the same first-step).  That later pass overwrites the dense 1.0
    set here whenever stochastic branches are active.
    """
    if dt is None:
        return None
    # Defensive re-cast: ensure d is canonical Enum on the Param.
    out = (dt.lazy()
             .select(alias_to_axis("d", "d")).unique()
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
    """Per-(d, t) branch weight — dense 1.0 baseline (superseded later).

    Returns the trivial 1.0 default per (d, t).  Like
    :func:`pd_branch_weight_from_source`, this is the ``apply_derived_a``
    baseline; it is the final value only for deterministic / non-stochastic
    fixtures.  For stochastic fixtures it is SUPERSEDED by the real
    sibling-normalised weights computed in ``apply_derived_g``
    (``_branch.apply_branch_cluster`` → ``pdt_branch_weight_lf``).
    """
    if dt is None:
        return None
    # Defensive re-cast: ensure d/t are canonical Enum on the Param.
    out = (dt.lazy()
             .with_columns(value=pl.lit(1.0))
             .select(alias_to_axis("d", "d"),
                     alias_to_axis("t", "t"),
                     "value")
             .sort("d", "t")
             .collect())
    if out.height == 0:
        return None
    return Param(("d", "t"), out)


# ---------------------------------------------------------------------------
# Catalog (used by parity tests)
# ---------------------------------------------------------------------------

DERIVED_A_FIELDS = (
    "dt",
    "p_step_duration",
    "p_period_share",
    "p_inflation_op",
    "p_timestep_weight",
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
    provider: "object | None" = None,
) -> None:
    """Apply Γ.3.A foundational Derived Params, mutating ``flex_data``
    in place.

    Δ.12b — assignment is unconditional for the helpers that are
    authoritative producers (``p_inflation_op``, ``p_timestep_weight``,
    ``pd_branch_weight``, ``pdt_branch_weight``, ``p_penalty_up``,
    ``p_penalty_down``).  ``p_inflow`` and ``p_process_existing_count``
    retain a conditional assignment because the helpers have known
    incomplete coverage (``inflow_method=scale_to_*`` is deferred to
    Batch B; ``p_process_existing_count`` skips when no processes have
    explicit existing capacity in the source) — see TODO at each call
    site for the helper-extension scope.

    Dependency order:
        dt / p_step_duration → p_period_share, p_inflation_op,
        p_timestep_weight, pd_branch_weight, pdt_branch_weight, p_inflow
        → p_process_existing_count, p_profile_value (high-risk, last).
    """
    active_solve = ctx.solve_name if ctx is not None else _read_active_solve(workdir, provider=provider)
    if active_solve is None:
        # Single-solve fixtures may omit solve_current.csv; fall back to
        # the FlexData's already-loaded dt (no override possible).
        return

    # 1. dt + p_step_duration -------------------------------------------
    # When ``solve.new_stepduration`` is set for the active solve, the
    # legacy ``TimelineConfig.create_timeline_from_timestep_duration``
    # has already rewritten the timeline (e.g. collapsing 8760 hourly
    # rows into one 8760 h row) and the legacy per-solve preprocessing
    # has materialised that rewrite into ``solve_data/steps_in_use.csv``.
    # The DB-direct ``dt_and_step_duration_from_source`` reads the raw
    # ``timeline.timestep_duration`` rows and would re-introduce the
    # full hourly grid, undoing the aggregation.  Skip the DB override
    # in that case and trust the CSV-loaded values.
    dt_csv = getattr(flex_data, "dt", None)
    new_sd_df = _try_param(source, "solve", "new_stepduration")
    skip_dt_override = False
    if new_sd_df is not None and "name" in new_sd_df.columns:
        sel = new_sd_df.filter(pl.col("name") == active_solve)
        if sel.height > 0 and "value" in sel.columns:
            v = sel["value"][0]
            if v is not None:
                skip_dt_override = True
    if skip_dt_override:
        usable_dt = dt_csv
        sd_for_share = getattr(flex_data, "p_step_duration", None)
    else:
        dt_step = dt_and_step_duration_from_source(source, active_solve,
                                                       workdir, ctx=ctx,
                                                       provider=provider)
        if dt_step is None:
            # Without dt we can't derive the dependent Params either; bail.
            return
        dt_db, step_dur_db = dt_step
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
        # The share was just built from this solve's own dt, so the solve covers all of it.
        if flex_data.p_period_share is not None:
            flex_data.p_window_fraction = Param(
                ("d",), flex_data.p_period_share.frame.with_columns(value=pl.lit(1.0, dtype=pl.Float64)))

    # 2b. p_years_represented_d — per-period R width sum (e.g. 5.0 for a
    # 5-year invest period) over the periods of the solve's dt and its
    # period labels (a label has no dt rows but keeps its years for the
    # period-level costs); consumed by ``process_outputs.read_parameters``
    # to scale annualised per-period costs / CO2 to absolute totals.  The
    # provider arm reads the canonical CSV, which carries the anchor's
    # years on every stochastic branch period.
    flex_data.p_years_represented_d = p_years_represented_d_from_source(
        source, active_solve, bookkeeping_periods(usable_dt, workdir, ctx=ctx, provider=provider),
        provider=provider)

    # 3. p_inflation_op — Δ.12b: unconditional.  None == no inflation
    #    declared; the multi-year cascade in apply_derived_c overlays
    #    this with the full per-period inflation later.
    flex_data.p_inflation_op = p_inflation_op_from_source(
        source, usable_dt, active_solve)

    # 4. p_timestep_weight — GUARDED (narrow inversion of Δ.12b for THIS
    #    helper).  None now means "no override — keep the CSV-loaded
    #    value".  On RP solves the helper returns None so the folded
    #    ``solve_data/timestep_weight.csv`` weights survive (the loader
    #    always populates a non-None Param, at minimum the dense 1.0
    #    default, so arithmetic stays safe).  Non-RP ``timeset_weights``
    #    solves still take the helper's computed Param unchanged.
    #    An rp solve keeps the writer's frame even when timeset_weights exist: it holds the rp periods'
    #    folded weights and the other periods' timeset_weights rows (solve_has_rp_timeset).
    tsw_new = p_timestep_weight_from_source(
        source, usable_dt, active_solve)
    if tsw_new is not None and not solve_has_rp_timeset(source, active_solve):
        flex_data.p_timestep_weight = tsw_new

    # 5. pd_branch_weight + pdt_branch_weight — Δ.12b: unconditional.
    # Γ.3.A's simple-1.0 helpers; Γ.3.G's full multi-branch cascade
    # overlays them later via apply_branch_cluster.
    pd_bw = pd_branch_weight_from_source(source, usable_dt)
    flex_data.pd_branch_weight = pd_bw
    flex_data.pdt_branch_weight = pdt_branch_weight_from_source(
        source, usable_dt, pd_bw)

    # 6. p_inflow -------------------------------------------------------
    # Native derivation pathways are DISABLED.  ``apply_p_inflow_with_scaling``
    # (and the ``p_inflow_from_source`` fallback) overrode the seed value
    # loaded by ``_load_node`` from ``solve_data/pdtNodeInflow.csv``, but
    # for aggregated-timestep solves (e.g. ``lt_rp`` with
    # ``step_duration = 8760``) the fallback joined raw per-hour source
    # values against ``dt`` and picked the per-hour rate as if it were
    # MWh-per-timestep — collapsing annual demand by a factor of
    # ``step_duration`` and yielding a structurally-wrong LP.
    #
    # The seed value produced by flextool's
    # ``preprocessing/node_inflow_scaling_params.py`` +
    # ``entity_period_calc_params.write_pdtNodeInflow`` is the known-good
    # input.  Until the native cascade is proven correct across all
    # ``inflow_method`` × ``step_duration`` combinations, the seed
    # survives unchanged.  Re-enable per-pathway only after parity is
    # established (see commit message for the disable for context).

    # 7. p_process_existing_count ---------------------------------------
    # Δ.12c-fix gap #4: the helper handles two paths:
    # (1) ``p_entity_all_existing.csv`` (workdir-derived, carries
    #     multi-solve handoff state and lifetime gate);
    # (2) ``p_entity_pre_existing`` (existing × virtual_unitsize) for the
    #     explicit ``unit.existing`` / ``connection.existing``; count = capacity / unitsize per (p, d).
    # Returns ``None`` for fixtures with neither path active (pure-invest
    # without initial capacity).  Keep conditional so the seed Param
    # (which always emits a row per process for ``base_cap_pd`` from the
    # CSV preprocessing) survives in that degenerate case.
    ec = p_process_existing_count_from_source(
        source, usable_dt, active_solve, workdir, provider=provider)
    if ec is not None:
        flex_data.p_process_existing_count = ec

    # 8. p_profile_value (Δ.7 cluster C — profile cascade) ------------
    # Δ.7 lifts the helper into ``_profiles.apply_profile_cascade``
    # — the new module ports flextool's full 5-branch cascade lazily and
    # drops the Δ.5/Δ.6 canonical-keys gate.  We delegate here so the
    # apply_derived_a entry stays the single integration point.
    # Δ.12b: defensive try/except removed — apply_profile_cascade is
    # parity-bound (cluster C tests gate it).
    from flextool.engine.preprocess.params._profiles import apply_profile_cascade
    apply_profile_cascade(flex_data, source, workdir, provider=provider)

    # 9. p_penalty_up / p_penalty_down (Δ.10 cluster F) ----------------
    # Dense over (nodeBalance ∪ nodeBalancePeriod) × usable_dt: the authored value (scalar, period,
    # time or period × time map, mixed across nodes), else the schema default.  A None result
    # (unclassifiable map, or an empty node set / dt) keeps the seed (emit pdtNode); a stochastic-time
    # map is resolved by the producer's own per-solve fold, not left to the seed.
    from flextool.engine.preprocess.params._arithmetic import apply_penalty_params
    apply_penalty_params(flex_data, source, usable_dt)


# ---------------------------------------------------------------------------
# §3.18 — Multi-branch normalisation (full cascade)
# ---------------------------------------------------------------------------


def _read_period_branch_pairs(workdir: Path | None,
                                  *,
                                  provider: "object | None" = None,
                                  ) -> list[tuple[str, str]]:
    """Read ``solve_data/period__branch.csv`` as a list of (d2, b)
    pairs — d2 is the parent period, b is the sibling branch.
    """
    if workdir is None:
        return []
    p = Path(workdir) / "solve_data" / "period__branch.csv"
    if not _provider_has_key(provider, p):  # Phase E-h — seed-aware
        return []
    df = _provider_read(provider, p)
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


def _read_solve_branch_weights(workdir: Path | None,
                                   *,
                                   provider: "object | None" = None,
                                   ) -> dict[str, float]:
    """Read ``solve_data/solve_branch_weight.csv`` as branch → weight."""
    out: dict[str, float] = {}
    if workdir is None:
        return out
    p = Path(workdir) / "solve_data" / "solve_branch_weight.csv"
    if not _provider_has_key(provider, p):  # Phase E-h — seed-aware
        return out
    df = _provider_read(provider, p)
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


def _read_first_timesteps(workdir: Path | None,
                              *,
                              provider: "object | None" = None,
                              ) -> dict[str, str]:
    """Read ``solve_data/first_timesteps.csv`` as period → first step."""
    out: dict[str, str] = {}
    if workdir is None:
        return out
    p = Path(workdir) / "solve_data" / "first_timesteps.csv"
    if not _provider_has_key(provider, p):  # Phase E-h — seed-aware
        return out
    df = _provider_read(provider, p)
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
    :func:`._branch.pd_branch_weight_param`.  Behaviour
    preserved: defaults to 1.0 when no ``period__branch`` rows exist
    (deterministic fixtures), normalises across siblings otherwise.

    See :mod:`._branch` for the algorithm reference.
    """
    from flextool.engine.preprocess.params._branch import pd_branch_weight_param
    out = pd_branch_weight_param(workdir, source, active_solve)
    if out is not None:
        return out
    # Match the historical "no period_in_use, but dt present →
    # deterministic 1.0 per realised period" fallback path.  This
    # keeps parity with the eager helper for chain-runner-less single
    # solves where the workdir CSVs may be absent but dt is built.
    if dt is None or dt.height == 0:
        return None
    # Defensive re-cast: ensure d is canonical Enum on the Param.
    df = (dt.lazy()
              .select(alias_to_axis("d", "d")).unique()
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
    :func:`._branch.pdt_branch_weight_param`.  Behaviour
    preserved: dense over ``dt`` (every (d, t) gets a value;
    denominator-zero rows fall through to 1.0).

    See :mod:`._branch` for the algorithm reference.
    """
    from flextool.engine.preprocess.params._branch import pdt_branch_weight_param
    return pdt_branch_weight_param(workdir, source, active_solve, dt)
