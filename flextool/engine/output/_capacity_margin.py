"""The capacity margin of each solve with investment periods: its per-solve tables and its run-log line.

At the per-roll write hook (:func:`flextool.engine.output._writer.write_outputs_for_solve`), while the solution is
live, :func:`write_capacity_margin_report` evaluates the solve's capacity-margin rows
(:func:`flextool.engine.model._group_slack.capacity_margin_report`) and

* writes ``output_raw/capacity_margin_report__<solve>.parquet`` (one row per group and investment period, at its
  binding step) and ``output_raw/capacity_margin_entity__<solve>.parquet`` (each unit's, connection's and node's
  contribution at that step).  They are kept per solve, over every investment period of the solve (look-ahead and
  forecast branch periods included), not filtered to the realized periods: :func:`read_capacity_margin_parts`
  concatenates them in solve order without dedup, so an invest solve's shortfall stays under its own name when a
  dispatch solve realizes the period;
* prints one line per group and period whose shortfall is above ``1e-6 · group_capacity_for_scaling``::

      capacity margin shortfall: solve <s>, group <g>, period <d>: <X> MW at <t> (penalty <Y> M CUR)

  The solve name tells a look-ahead period of one investment solve from the same period of the next.

  ``print(flush=True)``: the solve site's logger output does not reach the console.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from polar_high import Solution

_logger = logging.getLogger(__name__)

REPORT_PREFIX = "capacity_margin_report"
ENTITY_PREFIX = "capacity_margin_entity"
#: A shortfall below this share of the group's scaling capacity is solver noise: no run-log line.
SHORTFALL_TOLERANCE = 1e-6


def shortfall_lines(summary: pl.DataFrame, gcs: "pl.DataFrame | None") -> list[str]:
    """The run-log lines of the rows of ``summary`` (:func:`capacity_margin_report`) whose shortfall is above
    ``SHORTFALL_TOLERANCE · gcs`` (``gcs`` ``(group, period, gcs)``; 1 where it has no row)."""
    if summary.height == 0:
        return []
    rows = summary
    if gcs is not None and gcs.height > 0:
        rows = rows.join(gcs, on=["group", "period"], how="left")
    else:
        rows = rows.with_columns(gcs=pl.lit(None, dtype=pl.Float64))
    rows = rows.filter(pl.col("shortfall") > SHORTFALL_TOLERANCE * pl.col("gcs").fill_null(1.0))
    return [f"capacity margin shortfall: solve {r['solve']}, group {r['group']}, period {r['period']}: "
            f"{r['shortfall']:.6g} MW at "
            f"{r['binding_time']} (penalty {r['penalty_cost']:.6g} M CUR)"
            for r in rows.iter_rows(named=True)]


def write_capacity_margin_report(sol: "Solution", flex_data, *, output_dir: Path, solve_name: str,
                                 scale_the_objective: "float | None" = None) -> "tuple[pl.DataFrame, pl.DataFrame] | None":
    """Evaluate, persist and print the capacity margin of one solve (module docstring); returns ``(summary,
    entity)`` with a leading ``solve`` column, or None when the solve has no capacity margin."""
    from flextool.engine.model._group_slack import capacity_margin_report
    from flextool.engine.output._horizon_dump import _duals_valid

    if flex_data is None or sol is None:
        return None
    report = capacity_margin_report(flex_data, sol, scale_the_objective=scale_the_objective,
                                    duals_valid=_duals_valid(sol))
    if report is None:
        return None
    summary, entity = (frame.select(pl.lit(solve_name, dtype=pl.Utf8).alias("solve"), pl.all()) for frame in report)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.write_parquet(output_dir / f"{REPORT_PREFIX}__{solve_name}.parquet")
    entity.write_parquet(output_dir / f"{ENTITY_PREFIX}__{solve_name}.parquet")
    gcs_param = getattr(flex_data, "p_group_capacity_for_scaling", None)
    gcs = None
    if gcs_param is not None:
        frame = gcs_param.frame
        frame = frame.collect() if isinstance(frame, pl.LazyFrame) else frame
        gcs = frame.select(pl.col("g").cast(pl.Utf8).alias("group"), pl.col("d").cast(pl.Utf8).alias("period"),
                           pl.col("value").cast(pl.Float64).alias("gcs"))
    for line in shortfall_lines(summary, gcs):
        print(line, flush=True)
    return summary, entity


def read_capacity_margin_parts(output_dir: Path, prefix: str, solve_order: "dict[str, int] | None" = None
                               ) -> "pl.DataFrame | None":
    """The per-solve parquets ``<prefix>__<solve>.parquet`` of ``output_dir`` concatenated in solve order (no
    dedup: each solve keeps its own rows); None without any."""
    parts = sorted(Path(output_dir).glob(f"{prefix}__*.parquet"))
    if not parts:
        return None
    frames = [pl.read_parquet(p) for p in parts]
    frames = [f for f in frames if f.height > 0]
    if not frames:
        return None
    order = solve_order or {}
    frames.sort(key=lambda f: order.get(str(f["solve"][0]), -1))
    return pl.concat(frames, how="vertical_relaxed")
