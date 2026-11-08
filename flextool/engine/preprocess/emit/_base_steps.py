"""The original time steps behind each time step of a solve with ``new_stepduration``.

``TimelineConfig.create_averaged_timeseries`` writes ``solve_data/new_step_base_steps``: one row per original time
step of the coarsened timeline, ``time`` (the original step), ``new_step`` (the solve's time step that combines it;
its label is its first original step) and ``value`` (how many original steps that new step combines).  A solve
without ``new_stepduration`` gets an empty frame, so a later uncoarsened solve never reads an earlier solve's
rows.  An absent or empty frame means every solve step is one original step.

``inflow`` and ``peak_inflow`` are energy per original time step; a coarsened solve step carries the sum of its
original steps, so the inflow scaling divides by (or multiplies with) the number of steps it combines.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from flextool.engine.preprocess._provider_io import _provider_key

NEW_STEP_BASE_STEPS = "new_step_base_steps.csv"
NEW_STEP_BASE_STEPS_SCHEMA = {"time": pl.Utf8, "new_step": pl.Utf8, "value": pl.Float64}


def read_new_step_base_steps(solve_data_dir: Path, *, provider) -> tuple[dict[str, float], frozenset[str]]:
    """``({solve step: number of original steps it combines}, original steps absorbed into another step)``.

    Both are empty when the solve has no ``new_stepduration``.  The absorbed steps are the original steps that are
    not a solve step themselves (their energy is in the solve step that combines them).
    """
    df = provider.get(_provider_key(solve_data_dir / NEW_STEP_BASE_STEPS)) if provider is not None else None
    if df is None or df.height == 0:
        return {}, frozenset()
    df = df.select(pl.col("time").cast(pl.Utf8), pl.col("new_step").cast(pl.Utf8),
                   pl.col("value").cast(pl.Float64))
    n_of = {t: float(v) for t, s, v in df.iter_rows() if t == s}
    absorbed = frozenset(t for t, s, _v in df.iter_rows() if t != s)
    return n_of, absorbed
