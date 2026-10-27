"""Per-solve sets derived natively from InputSource.

Produces — directly from an :class:`InputSource` (and an optional
active solve name) — the per-solve aggregates that preprocessing
emits to ``solve_data/*.csv``.  Specifically:

* ``period_in_use`` — the ``[d]`` set of periods active in the current
  solve (``setof d from dt`` projected from ``steps_in_use.csv``).
* ``dt_complete`` — the ``[d, t]`` set of complete-time-in-use pairs,
  matching ``steps_complete_solve.csv``.
* ``period__timeline`` — the ``[d, timeline]`` mapping (timeset →
  timeline lookup).
* ``p_timeline_duration_in_years`` — the ``[timeline, value]`` per-
  timeline year fraction (``sum_t step_duration / 8760``).
* ``complete_period_share_of_year`` — the ``[d, value]`` per-period
  year fraction, restricted to dt_complete.

The authoritative caller is :func:`derive_per_solve_aggregates`
which returns a typed :class:`PerSolveAggregates` dataclass.  A
downstream consumer benefits:

* :mod:`flextool.engine.preprocess.params._derived` —
  ``_dt_period_active_steps_from_workdir`` can fall through to native
  derivation when the workdir CSVs are absent.

When a fixture has a ``solve.period_timeset`` filter for the active
solve in the source DB, the helpers no longer need ``solve_data/``
CSV output to compute their domain.

Architecture invariants
-----------------------

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
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from flextool.engine.input._axis_enums import alias_to_axis
from flextool.engine.input._constants import HOURS_PER_YEAR

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource


__all__ = [
    "PerSolveAggregates",
    "derive_per_solve_aggregates",
    "timeset_duration_lf",
    "timeset_timeline_lf",
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


def timeset_duration_lf(
    source: "InputSource",
    timesets: pl.LazyFrame,
    tl_lf: pl.LazyFrame,
) -> pl.LazyFrame | None:
    """Lazy ``[ts, start_step, count]`` from ``timeset.timeset_duration``,
    with the single-timeline "Rule 2" default applied (item #171).

    Mirrors the (start, count) Map semantics: each ts has zero or more
    contiguous timestep blocks, each starting at ``start_step`` and
    spanning ``count`` consecutive timesteps in the timeline's lex order.

    **Rule 2 default.** When the *scenario* authors no ``timeset.
    timeset_duration`` value at all (not merely "none for this timeset" —
    the guard is scenario-wide, matching ``input/_timeline.py``'s own
    Rule 2) and ``tl_lf`` names exactly one timeline, every timeset in
    *timesets* gets a single block covering the whole timeline:
    ``(first_step, step_count)``.

    ``first_step`` here is the **lexicographic minimum** step of the sole
    timeline (the same ``t0001 < t0002 < ...`` rank order every DB-side
    helper uses, see :func:`_expand_active_steps_lf`) — not the timeline's
    *authored-first* step, which is what ``_timeline.py``'s CSV-side Rule 2
    uses. Both blocks cover every rank of the timeline, so the *set* of
    steps the block expands to is identical either way; only the literal
    ``start_step`` label differs, and no caller reads that label on its
    own once the block has been expanded to steps.

    Returns ``None`` when *source* has an authored ``timeset.
    timeset_duration`` with no recognisable step column, or when no value
    is authored at all and the scenario does not have exactly one
    timeline (nothing to default to).
    """
    ts_dur = _try_param(source, "timeset", "timeset_duration")
    if ts_dur is not None:
        cols = ts_dur.columns
        step_col = next(
            (c for c in ("t", "x", "step", "timestep")
             if c in cols and c not in ("name", "value")),
            None,
        )
        if step_col is None:
            return None
        return (ts_dur.lazy()
                      .select(pl.col("name").cast(pl.Utf8).alias("ts"),
                              pl.col(step_col).cast(pl.Utf8).alias("start_step"),
                              pl.col("value").cast(pl.Float64).alias("count")))
    # Rule 2: no timeset_duration value anywhere in the scenario.
    tl_sorted = tl_lf.select("timeline", "t").sort("timeline", "t").collect()
    timeline_names = tl_sorted.select("timeline").unique()
    if timeline_names.height != 1:
        return None
    sole_timeline = timeline_names.item(0, "timeline")
    tl_sole = tl_sorted.filter(pl.col("timeline") == sole_timeline)
    first_step = str(tl_sole.row(0)[1])
    step_count = float(tl_sole.height)
    return (timesets.select(pl.col("ts").cast(pl.Utf8))
                     .unique()
                     .with_columns(pl.lit(first_step).alias("start_step"),
                                   pl.lit(step_count).alias("count")))


def timeset_timeline_lf(
    source: "InputSource",
    timesets: pl.LazyFrame,
    tl_lf: pl.LazyFrame,
) -> pl.LazyFrame | None:
    """Lazy ``[ts, timeline]`` from ``timeset.timeline``, with the
    single-timeline "Rule 3" default applied (item #171).

    **Rule 3 default.** When ``tl_lf`` names exactly one timeline, every
    timeset in *timesets* that has no authored ``timeset.timeline`` row
    is mapped to that sole timeline. With several timelines a missing row
    stays missing — ``_timeline.py``'s CSV-side Rule 3 raises
    ``FlexToolConfigError`` first in that situation, so this helper does
    not need to (it is read by DB-side readers only after the CSV path
    would already have failed loudly for the same model).

    Unlike :func:`timeset_duration_lf`'s Rule 2, this default is applied
    **per timeset** (a timeset with its own authored row keeps it; only
    timesets with no row at all are defaulted) — mirroring
    ``_timeline.py``'s own per-timeset Rule 3 loop.

    Returns ``None`` only when *source* has no ``timeset.timeline``
    parameter at all AND the scenario does not have exactly one timeline.
    """
    ts_tl = _try_param(source, "timeset", "timeline")
    authored = (
        ts_tl.lazy().select(pl.col("name").cast(pl.Utf8).alias("ts"),
                             pl.col("value").cast(pl.Utf8).alias("timeline"))
        if ts_tl is not None
        else pl.DataFrame(
            {"ts": [], "timeline": []},
            schema={"ts": pl.Utf8, "timeline": pl.Utf8},
        ).lazy()
    )
    timeline_names = tl_lf.select("timeline").unique().collect()
    if timeline_names.height != 1:
        return authored if ts_tl is not None else None
    sole_timeline = str(timeline_names.item(0, "timeline"))
    missing = (timesets.select(pl.col("ts").cast(pl.Utf8))
                        .unique()
                        .join(authored, on="ts", how="anti")
                        .with_columns(pl.lit(sole_timeline).alias("timeline")))
    return pl.concat([authored, missing]).unique()


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

    Mirrors :func:`flextool.engine.input._timeline.get_active_time`'s
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
      * ``timeset.timeset_duration`` or ``timeset.timeline`` is absent AND
        the scenario does not have exactly one timeline (the single-
        timeline "Rule 2" / "Rule 3" defaults of item #171 — see
        :func:`timeset_duration_lf` / :func:`timeset_timeline_lf`).

    Otherwise returns a fully-populated :class:`PerSolveAggregates`.

    The derivation is lazy until each field's eager ``.collect()`` —
    five collects total per call.  Caller is expected to materialise
    once per per-solve iteration and reuse across the override chain
    (``_dt_period_active_steps``).
    """
    pt_lf = _period_timeset_lf(source, active_solve)
    if pt_lf is None:
        return None
    tl_lf = _timeline_step_duration_lf(source)
    if tl_lf is None:
        return None
    timesets_lf = pt_lf.select("ts").unique()
    ts_dur_lf = timeset_duration_lf(source, timesets_lf, tl_lf)
    if ts_dur_lf is None:
        return None
    ts_tl_lf = timeset_timeline_lf(source, timesets_lf, tl_lf)
    if ts_tl_lf is None:
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
                   .agg((pl.col("step_duration").sum() / HOURS_PER_YEAR)
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
               .group_by("d").agg((pl.col("step_duration").sum()
                                    / HOURS_PER_YEAR)
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
