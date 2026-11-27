"""Γ.3.E — the per-period active steps of the solve (the structure every storage and step-lag frame is
built on)."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from flextool.engine.input._axis_enums import alias_to_axis, axis_lazyframe, cast_dim, rename_to_axis
from flextool.engine.preprocess.params._branch import period_branch_pairs_lf
from flextool.engine.preprocess.params._per_solve_sets import (
    solve_period_timeset_frame,
    timeset_duration_lf,
    timeset_timeline_lf,
)

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource
    from flextool.engine.preprocess._solve_context import SolveContext

from flextool.engine.preprocess.params._derived._scope import (
    _labels_of_pairs,
    _provider_has_key,
    _provider_read,
    _try_param,
)
from flextool.engine.preprocess.params._derived._periods import _expand_branch_periods


# ===========================================================================
# Γ.3.E — Storage block algebra (audit §3.9)
#
# Eight sub-sections covering the multi-resolution timeline algebra used by
# storage state, intra-period blocks, arc weighting, and rolling-handoff.
# No defensive gating — these helpers either produce the canonical frame or
# the parity test fails loudly.  See task spec for the architectural shift.
# ===========================================================================


def _dt_period_active_steps_from_workdir(
    source: "InputSource",
    workdir: Path,
    *,
    ctx: "SolveContext | None" = None,
    provider: "object | None" = None,
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

    Path B Cat B: when ``ctx`` is provided, the in-memory
    :pyattr:`SolveContext.steps_in_use` and
    :pyattr:`SolveContext.period_in_use` frames are consulted first.
    """
    siu: pl.DataFrame | None = None
    if ctx is not None:
        siu_ctx = ctx.steps_in_use
        if siu_ctx.height > 0:
            # Canonical schema [d, t, step_duration]; select the (d, t)
            # subset to match the legacy local frame.
            siu = siu_ctx.select("d", "t").unique()
    if siu is None:
        sd = workdir / "solve_data"
        siu_path = sd / "steps_in_use.csv"
        if not _provider_has_key(provider, siu_path):  # Phase E-j — seed-aware
            return None
        try:
            siu_raw = _provider_read(provider, siu_path)
        except Exception:
            return None
        if siu_raw.height == 0:
            return None
        period_col = next((c for c in ("period", "d") if c in siu_raw.columns),
                            None)
        step_col = next((c for c in ("step", "t", "time")
                           if c in siu_raw.columns), None)
        if period_col is None or step_col is None:
            return None
        siu = siu_raw.select(alias_to_axis(period_col, "d"),
                                alias_to_axis(step_col, "t")).unique()
    if siu.height == 0:
        return None
    sd = workdir / "solve_data"
    # Period order: prefer per_solve_sets.py's ``period_in_use_set.csv``
    # (canonical insertion order); fall back to insertion order in
    # steps_in_use.csv.
    period_order: list[str] = []
    seen: set[str] = set()
    piu_ctx_df = ctx.period_in_use if ctx is not None else None
    if piu_ctx_df is not None and piu_ctx_df.height > 0:
        for d in piu_ctx_df["d"].to_list():
            if d and d not in seen:
                period_order.append(d)
                seen.add(d)
    else:
        piu_path = sd / "period_in_use_set.csv"
        if _provider_has_key(provider, piu_path):  # Phase E-j — seed-aware
            try:
                piu_df = _provider_read(provider, piu_path)
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
    # A period label is in period_in_use_set for the period-level bookkeeping but has no active steps (except the
    # label of a split realization roll, which holds the jump).
    pb_wd = period_branch_pairs_lf(workdir, ctx=ctx, provider=provider).collect()
    labels = _labels_of_pairs(pb_wd["d"].to_list(), pb_wd["b"].to_list()) - set(siu["d"].cast(pl.Utf8).to_list())
    period_order = [d for d in period_order if str(d) not in labels]
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
                            alias_to_axis(tl_step_col, "t"),
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
    step_space = _to_solve_step_space(per_period, timeline_per_period, workdir, provider)
    for d, steps in per_period.items():
        if not steps:
            continue
        # Block starts: indices where the rank gap to the previous step
        # is >1.  Mirrors ``make_period_block``'s detection rule.
        block_firsts: list[str] = [steps[0][0]]
        for j in range(1, len(steps)):
            if steps[j][1] - steps[j - 1][1] > 1:
                block_firsts.append(steps[j][0])
        block_starts_per_period[d] = block_firsts
    return dict(
        per_period=per_period,
        timeline_for_period=timeline_per_period,
        period_order=period_order,
        block_starts_per_period=block_starts_per_period,
        step_space=step_space,
    )


def _solve_step_positions(timeline_for_period: dict[str, str], workdir: "Path | None",
                          provider: "object | None") -> "dict[tuple[str, str], int] | None":
    """``{(timeline, t): 1-based position}`` of every step in the solve's own timeline of each original timeline.

    Read from ``solve_data/steps_in_timeline`` (every step of each period's timeline, in timeline order; under a
    solve-level ``new_stepduration`` the aggregated timeline, whose gap chunks keep the gaps between timeset
    blocks).  The longest list per timeline wins (stochastic branch rows list only their active steps).  ``None``
    when the frame is absent."""
    if workdir is None:
        return None
    path = Path(workdir) / "solve_data" / "steps_in_timeline.csv"
    if not _provider_has_key(provider, path):
        return None
    try:
        df = _provider_read(provider, path)
    except Exception:
        return None
    if df.height == 0 or len(df.columns) < 2:
        return None
    steps_of: dict[str, list[str]] = {}
    for d, t in df.select(pl.col(df.columns[0]).cast(pl.Utf8), pl.col(df.columns[1]).cast(pl.Utf8)).iter_rows():
        steps_of.setdefault(d, []).append(t)
    best: dict[str, list[str]] = {}
    for d, steps in steps_of.items():
        tl = timeline_for_period.get(d)
        if tl is not None and len(steps) > len(best.get(tl, ())):
            best[tl] = steps
    return {(tl, t): i + 1 for tl, steps in best.items() for i, t in enumerate(steps)}


def _to_solve_step_space(per_period: dict[str, list[tuple[str, int]]], timeline_for_period: dict[str, str],
                         workdir: "Path | None", provider: "object | None") -> bool:
    """Re-rank ``per_period`` in place by position in the solve's own timeline (:func:`_solve_step_positions`).

    The rank difference of two steps is then ``make_step_jump``'s ``jump``: 1 exactly for steps adjacent on the
    timeline, also for aggregated ``new_stepduration`` steps, and > 1 across a gap.  Without the solve's timeline
    (or when a step is missing from it) the raw timeline ranks stay; returns whether the re-rank was applied."""
    positions = _solve_step_positions(timeline_for_period, workdir, provider)
    if not positions:
        return False
    reranked: dict[str, list[tuple[str, int]]] = {}
    for d, steps in per_period.items():
        tl = timeline_for_period.get(d)
        new = [(t, positions.get((tl, str(t)))) for t, _rk in steps]
        if any(rk is None for _t, rk in new):
            return False
        reranked[d] = sorted(new, key=lambda r: r[1])
    per_period.update(reranked)
    return True


def _dt_period_active_steps(source: "InputSource",
                              active_solve: str,
                              workdir: Path | None = None,
                              *,
                              ctx: "SolveContext | None" = None,
                              provider: "object | None" = None,
                              ) -> dict | None:
    """Build the per-period active-step structure used by every storage
    block-algebra helper.

    Returns a dict ``{"per_period": {d -> [(t, global_rank), ...]},
    "timeline_for_period": {d -> timeline_name},
    "period_order": [d in dispatch order],
    "block_starts_per_period": {d -> [t_block_first ordered by rank]},
    "step_space": bool}``, where ``global_rank`` is the step's position in
    the solve's own timeline (:func:`_to_solve_step_space`; ``step_space``
    True), else the raw timeline rank (cum_count over the timeline).  The
    block-first list is derived from
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
    # Prefer the workdir fallback when ``steps_in_use.csv`` (or
    # ``ctx.steps_in_use``) is available: that frame reflects the
    # post-``new_stepduration`` reality (e.g. invest_24h's 3 timesteps/
    # period after 24h aggregation), whereas the source path expands the
    # timeset over the raw hourly timeline.  When ``new_stepduration`` is
    # set, the source path would otherwise return the wrong (hourly)
    # active-step list, breaking storage state-coupling rows in the LP.
    steps_in_use_available = False
    if ctx is not None:
        try:
            steps_in_use_available = ctx.steps_in_use.height > 0
        except Exception:
            steps_in_use_available = False
    if not steps_in_use_available and workdir is not None:
        steps_in_use_available = (
            _provider_has_key(provider,   # Phase E-j — seed-aware
                Path(workdir) / "solve_data" / "steps_in_use.csv")
        )
    if steps_in_use_available and workdir is not None:
        wd_result = _dt_period_active_steps_from_workdir(
            source, workdir, ctx=ctx, provider=provider)
        if wd_result is not None:
            return wd_result

    # period → timeset of the active solve: ``solve.period_timeset``, or the
    # Rule 1 / 5b default (item #175).
    pt_frame = solve_period_timeset_frame(source, active_solve)
    if pt_frame is None:
        # No mapping for the active solve (synthetic rolling-horizon /
        # nested-invest sub-solve): preprocessing-only fallback.
        if workdir is not None:
            return _dt_period_active_steps_from_workdir(
                source, workdir, ctx=ctx, provider=provider)
        return None
    pt_eager_full = pt_frame.pipe(rename_to_axis, {"d": "d"})
    # Anchor → timeset map for the active solve.
    anchor_to_ts: dict[str, str] = dict(
        zip(pt_eager_full["d"].to_list(),
              pt_eager_full["ts"].to_list()))
    # Expand to include branch periods (workdir-aware — period__branch
    # is preprocessing-only data, no Spine class in v25).  Filter on
    # period_in_use_set.csv when present to exclude realized-output-only
    # branches (e.g. period1_realized).
    in_use: set[str] = set()
    pbf: pl.DataFrame | None = None
    # Path B Cat B: prefer typed ctx accessors over CSV reads.
    if ctx is not None:
        piu_ctx = ctx.period_in_use
        if piu_ctx.height > 0:
            in_use = set(piu_ctx["d"].to_list())
        pb_ctx = ctx.period_branch
        if pb_ctx.height > 0:
            # Match the legacy iteration shape via a local rename.
            pbf = pb_ctx.pipe(rename_to_axis, {"d_anchor": "period", "b": "branch"})
    if workdir is not None:
        if not in_use:
            piu_path = Path(workdir) / "solve_data" / "period_in_use_set.csv"
            if _provider_has_key(provider, piu_path):  # Phase E-j — seed-aware
                try:
                    piu_df = _provider_read(provider, piu_path)
                    if piu_df.height > 0:
                        in_use = set(piu_df["period"].to_list())
                except Exception:
                    in_use = set()
        if pbf is None:
            pb_path = Path(workdir) / "solve_data" / "period__branch.csv"
            if _provider_has_key(provider, pb_path):  # Phase E-j — seed-aware
                try:
                    pbf = _provider_read(provider, pb_path)
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
        # A period label has no dispatch: it has no active steps.
        for label in _labels_of_pairs(pbf["period"].to_list(), pbf["branch"].to_list()):
            anchor_to_ts.pop(label, None)
    if not anchor_to_ts:
        return None
    # Use axis_lazyframe so ``d`` carries the global Enum vocabulary
    # when active — otherwise the downstream join against
    # ``realized_clean`` (which IS axis_lazyframe-built) trips on
    # ``SchemaError: d: enum on left does not match d: str on right``.
    pt_lf = axis_lazyframe({
        "d": list(anchor_to_ts.keys()),
        "ts": list(anchor_to_ts.values()),
    })
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
                            alias_to_axis(tl_step_col, "t"),
                            pl.col("value").cast(pl.Float64)
                                            .alias("step_duration"))
                    .sort("timeline", "t")
                    .with_columns(rank=pl.col("t").cum_count()
                                                  .over("timeline")
                                                  .cast(pl.Int64)))

    # ``timeset.timeline`` / ``timeset.timeset_duration``, through the
    # shared single-timeline defaults (item #171, Rules 2/3 — see
    # ``_per_solve_sets.timeset_timeline_lf`` / ``timeset_duration_lf``).
    timesets_lf = pt_lf.select("ts").unique()
    ttl_lf = timeset_timeline_lf(source, timesets_lf, tl_lf)
    if ttl_lf is None:
        return None

    ts_dur_lf = timeset_duration_lf(source, timesets_lf, tl_lf)
    if ts_dur_lf is None:
        return None
    # ``start_step`` lives on the ``t`` axis; cast to the canonical Enum
    # so the downstream join against ``tl_lf`` (whose ``t``/``start_step``
    # is cast to Enum at line 6204+) doesn't trip on Utf8-vs-Enum.  The
    # shared helper returns Utf8/Float64; this caller casts both itself.
    blocks_lf = ts_dur_lf.select(
        pl.col("ts"),
        cast_dim(pl.col("start_step"), None, "t").alias("start_step"),
        pl.col("count").cast(pl.Int64),
    )
    # Period order: prefer realized_periods.i ordering when present, else
    # fall back to period_timeset's natural order.  ``period_in_use_set``
    # lists ALL solve periods (realized + non-realized + branch siblings).
    realized = _try_param(source, "solve", "realized_periods")
    explicit_order: list[str] = []
    if realized is not None:
        # ``realized_periods`` is Array-shaped; the index column name is
        # the Spine ``index_name`` when set, else ``_array_index`` (see
        # ``_spinedb_reader._unroll_index_columns_from_value`` — Array
        # types use ``_array_index`` to avoid collision with the tier
        # axis ``i``).  Use the second-to-last column (between ``name``
        # and ``value``) rather than hardcoding ``"i"`` — same shape-
        # tolerant rule as ``_solve_periods``.
        order_col = (realized.columns[-2]
                       if len(realized.columns) >= 2 else "i")
        realized_p = (realized
                       .filter(pl.col("name") == active_solve)
                       .sort(order_col))
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
    if workdir is not None or ctx is not None:
        period_order = _expand_branch_periods(period_order, workdir, ctx=ctx, provider=provider)
        # Path B Cat B: prefer ctx.period_in_use over CSV read.
        canonical: list[str] | None = None
        if ctx is not None:
            piu_ctx = ctx.period_in_use
            if piu_ctx.height > 0:
                canonical = piu_ctx["d"].to_list()
        if canonical is None and workdir is not None:
            piu_path = Path(workdir) / "solve_data" / "period_in_use_set.csv"
            if _provider_has_key(provider, piu_path):  # Phase E-j — seed-aware
                try:
                    piu_df = _provider_read(provider, piu_path)
                    if piu_df.height > 0:
                        canonical = piu_df["period"].to_list()
                except Exception:
                    canonical = None
        if canonical is not None:
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
    if not period_order:
        return None
    realized_clean = axis_lazyframe({"d": period_order})
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
    step_space = _to_solve_step_space(per_period, timeline_per_period, workdir, provider)
    return dict(
        per_period=per_period,
        timeline_for_period=timeline_per_period,
        period_order=period_order,
        block_starts_per_period=block_starts_per_period,
        step_space=step_space,
    )
