"""Placing a reported cost event in a period: the periods an event can be reported in (:func:`report_years`), the
period an event's year falls in (:func:`event_period`) and the stacking of the cost lines (:func:`stack_lines`).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from flextool.engine.preprocess.params._discount import run_years

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource


def report_years(source: "InputSource",
                 active_solve: str | None,
                 period_in_use: list[str],
                 workdir: Path | None,
                 provider: "object | None") -> dict[str, float]:
    """``{period: year from the start of the run}`` of the periods the reported costs can place an event in: the run's
    periods (:func:`._discount.run_years`), or the solve's periods with its history (:func:`._cohorts.cohort_years`)
    when the run is the solve."""
    run = run_years(source, active_solve, list(period_in_use), workdir, provider=provider)
    if run is not None:
        return dict(run[0])
    from flextool.engine.preprocess.params._cohorts import cohort_years
    return dict(cohort_years(source, active_solve, workdir, provider=provider))


def event_period(frame: pl.DataFrame, y_col: str, years: dict[str, float], out: str = "d_report") -> pl.DataFrame:
    """``frame`` with ``out`` (Utf8): the period of ``years`` that starts at the year ``frame[y_col]`` (within
    ``YEAR_TOL``; the first such period in ``years``' order), else the last period of ``years`` (the latest year):
    the period an end of life is reported in, the last period for a life that ends at or past the end of the
    horizon (the salvage credit of its years inside the horizon)."""
    from flextool.engine.preprocess.params._run_calendar import YEAR_TOL
    if not years:
        return frame.with_columns(pl.lit(None, dtype=pl.Utf8).alias(out))
    order = list(years)
    latest = max(years.values())
    last = next(d for d in order if abs(years[d] - latest) <= YEAR_TOL)
    yr = pl.DataFrame({"_p": order, "_y": [float(years[d]) for d in order], "_i": list(range(len(order)))},
                      schema={"_p": pl.Utf8, "_y": pl.Float64, "_i": pl.Int64})
    wanted = frame.select(pl.col(y_col).cast(pl.Float64).alias("_want")).unique().filter(pl.col("_want").is_finite())
    hit = (wanted.join(yr, how="cross").filter((pl.col("_y") - pl.col("_want")).abs() <= YEAR_TOL)
                 .sort("_i").unique(subset=["_want"], keep="first").select("_want", "_p"))
    return (frame.with_columns(pl.col(y_col).cast(pl.Float64).alias("_want"))
                 .join(hit, on="_want", how="left", maintain_order="left")
                 .with_columns(pl.coalesce("_p", pl.lit(last, dtype=pl.Utf8)).alias(out))
                 .drop("_want", "_p"))


def stack_lines(frame: pl.DataFrame, keys: tuple[str, ...], parts: list[tuple[str, pl.Expr, pl.Expr]]
                ) -> pl.DataFrame:
    """``[*keys, line, d_report, value]`` (labels Utf8, :func:`._cohorts.as_names`) of ``frame``: for each ``(line,
    value, report)`` of ``parts`` the amount ``value`` reported in the period ``report``, summed per key, line and
    report period; rows of 0 left out."""
    from flextool.engine.preprocess.params._cohorts import as_names
    schema = {**{k: pl.Utf8 for k in keys}, "line": pl.Utf8, "d_report": pl.Utf8, "value": pl.Float64}
    if frame.height == 0 or not parts:
        return pl.DataFrame(schema=schema)
    named = as_names(frame, [k for k in keys if frame.schema[k] != pl.Utf8])
    stacked = pl.concat([named.select(*keys, line=pl.lit(line, dtype=pl.Utf8), d_report=report.cast(pl.Utf8),
                                      value=value.cast(pl.Float64))
                         for line, value, report in parts], how="vertical_relaxed")
    return (stacked.filter(pl.col("value").fill_null(0.0) != 0.0)
                   .group_by(*keys, "line", "d_report", maintain_order=True).agg(pl.col("value").sum())
                   .select(*schema))
