"""Δ.13 — Per-solve sets derived natively from InputSource.

Native polars-lazy port of the workdir-CSV-producing fragments of
``flextool/flextoolrunner/preprocessing/per_solve_sets.py`` and
``flextool/flextoolrunner/preprocessing/period_calculated_params.py``
that the override chain currently depends on.

Scope
-----

This module produces — directly from an :class:`InputSource` (and an
optional active solve name) — the same per-solve aggregates that
flextool's preprocessing emits to ``solve_data/*.csv``.  Specifically:

* ``period_in_use`` — the ``[d]`` set of periods active in the current
  solve.  Mirrors ``per_solve_sets.write_per_solve_sets`` line 95-101
  (``setof d from dt`` projected from ``steps_in_use.csv``).
* ``dt_complete`` — the ``[d, t]`` set of complete-time-in-use pairs.
  Mirrors ``solve_writers.write_active_timelines`` for
  ``steps_complete_solve.csv``.
* ``period__timeline`` — the ``[d, timeline]`` mapping.  Mirrors
  ``per_solve_sets.write_per_solve_sets`` line 205-225 + the timeset →
  timeline lookup.
* ``p_timeline_duration_in_years`` — the ``[timeline, value]`` per-
  timeline year fraction (``sum_t step_duration / 8760``).  Mirrors
  ``period_calculated_params.write_period_calculated_params`` line
  151-154.
* ``complete_period_share_of_year`` — the ``[d, value]`` per-period
  year fraction, restricted to dt_complete.  Mirrors
  ``period_calculated_params.write_period_calculated_params`` line
  201-204.

The module's authoritative caller is :func:`derive_per_solve_aggregates`
which returns a typed :class:`PerSolveAggregates` dataclass.  Two
downstream consumers benefit:

1. :mod:`flextool.engine_polars._inflow_scaling` — used to drop the
   workdir-CSV reads in ``_timeline_aggregates`` and ``_dt_complete_lf``.
2. :mod:`flextool.engine_polars._derived_params` —
   ``_dt_period_active_steps_from_workdir`` can fall through to native
   derivation when the workdir CSVs are absent.

The override chain becomes self-sufficient: when a fixture has a
``solve.period_timeset`` filter for the active solve in the source DB,
the helpers no longer need ``runner.write_input``'s ``solve_data/``
output to compute their domain.

Architecture invariants preserved
---------------------------------

* **Lazy throughout** — every internal frame is a
  :class:`polars.LazyFrame`; the public dataclass holds eager
  :class:`polars.DataFrame` for caller convenience but every
  computation funnels through one ``.collect()`` at the rim.
* **None default → skip entirely** — the public function returns
  ``None`` when the source lacks ``solve.period_timeset`` /
  ``timeline.timestep_duration``; callers fall through to the workdir
  path when present, or the existing legacy CSV fallback.
* **No defensive gating** — helpers fail loudly when frames have
  unexpected schemas.

Reference: ``flextool/flextoolrunner/preprocessing/per_solve_sets.py``
(read-only mirror of the procedural reference).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from ._axis_enums import alias_to_axis

if TYPE_CHECKING:
    from flextool.engine_polars._input_source import InputSource


__all__ = [
    "PerSolveAggregates",
    "derive_per_solve_aggregates",
]


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class PerSolveAggregates:
    """Native per-solve aggregates derived from an :class:`InputSource`.

    All frames are eagerly materialised
    (:class:`polars.DataFrame`) for caller convenience; internally the
    derivation is lazy with one ``.collect()`` per field at the rim.

    Attributes
    ----------
    period_in_use : pl.DataFrame
        ``[d]`` distinct periods active in the current solve.
    dt_complete : pl.DataFrame
        ``[d, t]`` complete-time-in-use pairs (every (d, t) over the
        period's full timeline range, not just dt).
    period_timeline : pl.DataFrame
        ``[d, timeline]`` mapping — one row per (period, timeline_name).
    p_timeline_duration_in_years : pl.DataFrame
        ``[timeline, value]`` — ``sum_t step_duration[tl, t] / 8760``.
    complete_period_share_of_year : pl.DataFrame
        ``[d, value]`` — period's full-timeline coverage as a year
        fraction (``sum_t step_duration / 8760`` over dt_complete).
    """

    period_in_use: pl.DataFrame
    dt_complete: pl.DataFrame
    period_timeline: pl.DataFrame
    p_timeline_duration_in_years: pl.DataFrame
    complete_period_share_of_year: pl.DataFrame


# ---------------------------------------------------------------------------
# Internal source helpers
# ---------------------------------------------------------------------------


def _try_param(source: "InputSource", entity_class: str,
               parameter_name: str) -> pl.DataFrame | None:
    try:
        df = source.parameter(entity_class, parameter_name)
    except KeyError:
        return None
    if df.height == 0:
        return None
    return df


def _timeline_step_duration_lf(source: "InputSource"
                                ) -> pl.LazyFrame | None:
    """Lazy ``[timeline, t, step_duration]`` from
    ``timeline.timestep_duration``.

    Returns ``None`` when the parameter is absent or has an unexpected
    schema.  The Spine source emits the index column under
    ``t`` / ``step`` / ``timestep`` / ``x`` depending on the underlying
    Map index_name; the helper auto-discovers the column.
    """
    tl_dur = _try_param(source, "timeline", "timestep_duration")
    if tl_dur is None:
        return None
    cols = tl_dur.columns
    step_col = next(
        (c for c in ("t", "step", "timestep", "x")
         if c in cols and c not in ("name", "value")),
        None,
    )
    if step_col is None:
        return None
    return (tl_dur.lazy().select(
        pl.col("name").alias("timeline"),
        alias_to_axis(step_col, "t"),
        pl.col("value").cast(pl.Float64).alias("step_duration"),
    ))


def _period_timeset_lf(source: "InputSource", active_solve: str
                        ) -> pl.LazyFrame | None:
    """Lazy ``[d, ts]`` for the active solve from ``solve.period_timeset``.

    Returns ``None`` when the parameter is absent OR the filter for the
    active solve is empty (synthetic rolling-horizon / nested-invest
    sub-solves whose names aren't in Spine — caller handles None).
    """
    p_ts = _try_param(source, "solve", "period_timeset")
    if p_ts is None:
        return None
    period_col = next((c for c in ("period", "x") if c in p_ts.columns),
                      None)
    if period_col is None:
        return None
    lf = (p_ts.lazy()
              .filter(pl.col("name") == active_solve)
              .select(alias_to_axis(period_col, "d"),
                      pl.col("value").alias("ts")))
    if lf.collect().height == 0:
        return None
    return lf


def _timeset_duration_lf(source: "InputSource") -> pl.LazyFrame | None:
    """Lazy ``[ts, start_step, count]`` from ``timeset.timeset_duration``.

    Mirrors the (start, count) Map semantics: each ts has zero or more
    contiguous timestep blocks, each starting at ``start_step`` and
    spanning ``count`` consecutive timesteps in the timeline's lex order.
    """
    ts_dur = _try_param(source, "timeset", "timeset_duration")
    if ts_dur is None:
        return None
    cols = ts_dur.columns
    step_col = next(
        (c for c in ("t", "x", "step", "timestep")
         if c in cols and c not in ("name", "value")),
        None,
    )
    if step_col is None:
        return None
    return (ts_dur.lazy()
                  .select(pl.col("name").alias("ts"),
                          pl.col(step_col).alias("start_step"),
                          pl.col("value").cast(pl.Float64).alias("count")))


def _timeset_timeline_lf(source: "InputSource") -> pl.LazyFrame | None:
    """Lazy ``[ts, timeline]`` from ``timeset.timeline``."""
    ts_tl = _try_param(source, "timeset", "timeline")
    if ts_tl is None:
        return None
    return (ts_tl.lazy()
                 .select(pl.col("name").alias("ts"),
                         pl.col("value").alias("timeline")))


# ---------------------------------------------------------------------------
# Per-period active-step expansion (mirrors flextool's get_active_time)
# ---------------------------------------------------------------------------


def _expand_active_steps_lf(
    pt_lf: pl.LazyFrame,        # [d, ts]
    ts_tl_lf: pl.LazyFrame,     # [ts, timeline]
    ts_dur_lf: pl.LazyFrame,    # [ts, start_step, count]
    tl_lf: pl.LazyFrame,        # [timeline, t, step_duration]
) -> pl.LazyFrame:
    """Expand (d, ts) → (d, t, step_duration) by walking the timeline
    starting at each block's ``start_step`` for ``count`` steps.

    Mirrors :func:`flextool.engine_polars._timeline.get_active_time`'s
    procedural expansion: for each timeset block, find the rank of
    ``start_step`` within the timeline's lex-ordered timestep list,
    then emit timesteps with ranks in ``[start_rank, start_rank+count)``.
    """
    # Rank timeline timesteps in lex order (matches t0001 < t0002 < ...).
    tl_ranked = (tl_lf.sort("timeline", "t")
                       .with_columns(rank=pl.col("t").cum_count()
                                                      .over("timeline")
                                                      .cast(pl.Int64)))
    # Resolve start_rank by joining timeline ranks on (timeline, t=start_step).
    blocks = (pt_lf
                .join(ts_tl_lf, on="ts", how="inner")
                .join(ts_dur_lf, on="ts", how="inner")
                .join(tl_ranked.select(
                    pl.col("timeline"),
                    pl.col("t").alias("start_step"),
                    pl.col("rank").alias("start_rank")),
                    on=["timeline", "start_step"], how="left")
                .filter(pl.col("start_rank").is_not_null()))
    # Cross-join with timeline ranks, then filter rank-in-window.
    expanded = (blocks
                  .join(tl_ranked, on="timeline", how="inner")
                  .filter((pl.col("rank") >= pl.col("start_rank"))
                          & (pl.col("rank") < pl.col("start_rank")
                              + pl.col("count").cast(pl.Int64))))
    return expanded.select("d", "timeline", "t", "step_duration").unique()


# ---------------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------------


def derive_per_solve_aggregates(
    source: "InputSource",
    active_solve: str,
) -> PerSolveAggregates | None:
    """Build per-solve aggregates natively from *source*.

    Returns ``None`` when:
      * ``timeline.timestep_duration`` is absent;
      * ``solve.period_timeset`` for *active_solve* is empty (synthetic
        rolling/nested sub-solves whose names aren't in Spine — caller
        falls through to the workdir path);
      * ``timeset.timeset_duration`` or ``timeset.timeline`` is absent.

    Otherwise returns a fully-populated :class:`PerSolveAggregates`.

    The derivation is lazy until each field's eager ``.collect()`` —
    five collects total per call.  Caller is expected to materialise
    once per per-solve iteration and reuse across the override chain
    (``_inflow_scaling`` + ``_dt_period_active_steps``).
    """
    pt_lf = _period_timeset_lf(source, active_solve)
    if pt_lf is None:
        return None
    ts_dur_lf = _timeset_duration_lf(source)
    if ts_dur_lf is None:
        return None
    ts_tl_lf = _timeset_timeline_lf(source)
    if ts_tl_lf is None:
        return None
    tl_lf = _timeline_step_duration_lf(source)
    if tl_lf is None:
        return None

    # Expand to (d, t, timeline, step_duration) at full lazy.
    expanded_lf = _expand_active_steps_lf(pt_lf, ts_tl_lf, ts_dur_lf, tl_lf)

    # ── period_in_use[d] ─────────────────────────────────────────────
    period_in_use = (expanded_lf.select("d").unique().sort("d").collect())
    if period_in_use.height == 0:
        # No active periods for this solve — caller falls through.
        return None

    # ── dt_complete[d, t] ────────────────────────────────────────────
    # ``complete_active_time_lists = get_active_time(complete_solve_name)``
    # — the ACTIVE expansion of the period's timeset blocks (NOT the
    # full timeline).  For non-rolling solves this equals dt; for
    # rolling solves it's the parent solve's full-year expansion.  Both
    # are produced by the same expand-blocks-to-timesteps walk.
    period_tl_lf = expanded_lf.select("d", "timeline").unique()
    dt_complete = (expanded_lf.select("d", "t").unique()
                                .sort("d", "t").collect())

    # ── period_timeline[d, timeline] ─────────────────────────────────
    period_timeline = (period_tl_lf.sort("d", "timeline").collect())

    # ── p_timeline_duration_in_years[timeline] ───────────────────────
    # = sum_t step_duration[timeline, t] / 8760
    p_tdy = (tl_lf.group_by("timeline")
                   .agg((pl.col("step_duration").sum() / 8760.0)
                        .alias("value"))
                   .sort("timeline")
                   .collect())

    # ── complete_period_share_of_year[d] ─────────────────────────────
    # Mirrors ``period_calculated_params.py`` lines 186-204:
    #   complete_hours_in_period[d] = sum_{(d2, t) ∈ dt_complete,
    #                                       (d2, d) ∈ period__branch}
    #                                  complete_step_duration[d2, t]
    # For deterministic (non-stochastic) solves the period__branch is
    # the diagonal — (d, d) for every d — so the sum collapses to
    # ``sum_t complete_step_duration[d, t]``.  Multi-branch stochastic
    # extension is out of scope for the current dispatch; the
    # PerSolveAggregates dataclass already exposes the per-(d, d) frame
    # which downstream branch-aware helpers can fold themselves.
    cpsoy = (expanded_lf
               .group_by("d").agg((pl.col("step_duration").sum() / 8760.0)
                                  .alias("value"))
               .sort("d")
               .collect())

    return PerSolveAggregates(
        period_in_use=period_in_use,
        dt_complete=dt_complete,
        period_timeline=period_timeline,
        p_timeline_duration_in_years=p_tdy,
        complete_period_share_of_year=cpsoy,
    )
