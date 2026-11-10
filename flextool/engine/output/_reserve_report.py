"""The reserve requirement of each reserve group: its per-solve tables and the run-log line of a broken check.

At the per-roll write hook (:func:`flextool.engine.output._writer.write_outputs_for_solve`), while the solution is
live, :func:`write_reserve_report` evaluates the solve's reserve requirement rows
(:func:`flextool.engine.model._reserve.reserve_requirement_report`) and

* writes ``output_raw/reserve_group__<solve>.parquet`` (one row per reserve, direction, group and step: the
  requirement of each method, the binding row, the reserve provided and counted, the shortfall) and
  ``output_raw/reserve_group_provider__<solve>.parquet`` (each provider's counted reserve), filtered to the steps
  the solve realizes.  :func:`read_reserve_report_parts` unions them over the solves, a step taken from the last
  solve that realizes it (as the other per-step outputs);
* prints one line per group step whose shortfall does not match its largest deficit (the row evaluation and the LP
  disagree), at most :data:`MAX_CHECK_LINES` per solve and then a count of the rest::

      reserve requirement check: solve <s>, reserve <r> <ud>, group <g>, step (<d>, <t>): deficit <X> MW, shortfall <Y> MW

  ``print(flush=True)``: the solve site's logger output does not reach the console.  An accepted near-optimal
  solve (an interior point without crossover) can miss the tolerance without being wrong: its lines say so and
  are logged as a warning, not an error.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from polar_high import Solution

_logger = logging.getLogger(__name__)

REPORT_PREFIX = "reserve_group"
PROVIDER_PREFIX = "reserve_group_provider"
#: The row keys of each raw frame below the solve; the step is (period, time).
REPORT_KEYS = ("reserve", "updown", "node_group", "period", "time")
PROVIDER_KEYS = ("reserve", "updown", "node_group", "process", "node", "period", "time")


#: The provider frame of the steps a solve realizes, the one the variable writer filters by
#: (:func:`flextool.process_outputs.read_highs_solution._load_realized_set`).
REALIZED_DISPATCH_KEY = "solve_data/realized_dispatch"


def _realized_steps(flex_data, provider=None) -> "pl.DataFrame | None":
    """``(period, time)`` of the steps the solve realizes, Utf8: the provider's ``solve_data/realized_dispatch``
    (as the variable writer; empty for a forecast fan solve that realizes no step), else
    ``flex_data.realized_dispatch``; both are keyed ``(period, step)``.  None when neither says (every step is
    kept)."""
    rd = provider.get(REALIZED_DISPATCH_KEY) if provider is not None and provider.has(REALIZED_DISPATCH_KEY) else None
    if rd is None:
        rd = getattr(flex_data, "realized_dispatch", None)
    if rd is None:
        return None
    if rd.height == 0:
        return pl.DataFrame(schema={"period": pl.Utf8, "time": pl.Utf8})
    time_col = "step" if "step" in rd.columns else "time"
    return rd.select(pl.col("period").cast(pl.Utf8), pl.col(time_col).cast(pl.Utf8).alias("time")).unique()


#: The most broken-check lines one solve prints; the rest are counted in one closing line.
MAX_CHECK_LINES = 10


def check_lines(violations: pl.DataFrame, solve_name: str, *, near_optimal: bool) -> list[str]:
    """The run-log lines of the broken checks ``violations`` (:func:`reserve_requirement_report`): the first
    :data:`MAX_CHECK_LINES` group steps, then ``... and N more group steps`` for the rest."""
    note = " (accepted near-optimal solve)" if near_optimal else ""
    lines = [f"reserve requirement check: solve {solve_name}, reserve {r['reserve']} {r['updown']}, group "
             f"{r['node_group']}, step ({r['period']}, {r['time']}): deficit {max(r['deficit'], 0.0):.6g} MW, "
             f"shortfall {r['shortfall']:.6g} MW{note}"
             for r in violations.head(MAX_CHECK_LINES).iter_rows(named=True)]
    if violations.height > MAX_CHECK_LINES:
        lines.append(f"reserve requirement check: solve {solve_name}: ... and "
                     f"{violations.height - MAX_CHECK_LINES} more group steps{note}")
    return lines


def write_reserve_report(sol: "Solution", flex_data, *, output_dir: Path, solve_name: str, provider=None
                         ) -> "tuple[pl.DataFrame, pl.DataFrame] | None":
    """Evaluate, persist and check the reserve requirement of one solve (module docstring); returns ``(summary,
    provider)`` with a leading ``solve`` column, filtered to the realized steps, or None when the solve has no
    reserve requirement."""
    from flextool.engine.model._reserve import reserve_requirement_report

    if flex_data is None or sol is None:
        return None
    report = reserve_requirement_report(flex_data, sol)
    if report is None:
        return None
    steps = _realized_steps(flex_data, provider)
    summary, provider, violations = report
    if steps is not None:
        summary, provider, violations = (frame.join(steps, on=["period", "time"], how="semi")
                                         for frame in (summary, provider, violations))
    summary, provider = (frame.select(pl.lit(solve_name, dtype=pl.Utf8).alias("solve"), pl.all())
                         for frame in (summary, provider))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.write_parquet(output_dir / f"{REPORT_PREFIX}__{solve_name}.parquet")
    provider.write_parquet(output_dir / f"{PROVIDER_PREFIX}__{solve_name}.parquet")
    near_optimal = not getattr(sol, "optimal", True)
    for line in check_lines(violations, solve_name, near_optimal=near_optimal):
        print(line, flush=True)
        (_logger.warning if near_optimal else _logger.error)(line)
    return summary, provider


def read_reserve_report_parts(output_dir: Path, prefix: str, solve_order: "dict[str, int] | None" = None
                              ) -> "pl.DataFrame | None":
    """The per-solve parquets ``<prefix>__<solve>.parquet`` of ``output_dir`` unioned over the solves: each
    ``(period, time)`` step from the last solve (in ``solve_order``, else file order) that has it, as the other
    per-step outputs (``drop_levels``, keep last).  None without any row."""
    parts = sorted(Path(output_dir).glob(f"{prefix}__*.parquet"))
    frames = [f for f in (pl.read_parquet(p) for p in parts) if f.height > 0]
    if not frames:
        return None
    order = solve_order or {}
    frames.sort(key=lambda f: order.get(str(f["solve"][0]), -1))
    union = pl.concat([f.with_columns(__part=pl.lit(i, dtype=pl.Int32)) for i, f in enumerate(frames)],
                      how="vertical_relaxed")
    owner = union.group_by("period", "time").agg(pl.col("__part").max())
    return union.join(owner, on=["period", "time", "__part"], how="semi").drop("__part")
