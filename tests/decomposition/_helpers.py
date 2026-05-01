"""Helpers for Tier 8 obj-decomposition parity tests.

The decomposition assertion is:

    sum(costs_discounted.csv numeric values)  ~=  parse_summary_obj(summary_solve.csv)

both in M CUR (millions of currency units).

Units note
----------
``flextool/process_outputs/out_costs.py`` divides every entry of the
``costs_discounted_d_p`` / ``costs_discounted_p_`` table by ``to_millions
= 1_000_000`` before writing. ``summary_solve.csv``'s row
"Total cost (calculated) full horizon (M CUR)" is also in millions
(parsed by ``_parse_summary_solve_objective`` in ``test_scenarios.py``).
No unit conversion is required in the helper.

Format note
-----------
Despite the spec hint that multi-period scenarios may use period-named
columns (``p2020``, ``p2025``, …), every golden in ``tests/expected``
emits a single numeric column named ``0`` — the writer collapses
``costs_discounted_p_`` to a one-column "horizon total" series. The
helper sums *all* numeric columns so it works regardless of layout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_TEST_DIR = Path(__file__).resolve().parent.parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

# Re-export the canonical objective parser so tests share one source of truth.
from test_scenarios import _parse_summary_solve_objective  # noqa: E402


def parse_costs_discounted(path: Path) -> float:
    """Sum all numeric values in ``costs_discounted.csv``.

    The first column is the category label; every remaining column is a
    period (or the single ``0`` column in non-period layouts). Returns the
    grand total in M CUR.
    """
    df = pd.read_csv(path)
    # All columns except the category label are numeric value columns.
    if df.columns[0] != "category":
        # Defensive: if the writer's header ever changes, error loudly.
        raise ValueError(
            f"Unexpected header in {path}: expected first column 'category', "
            f"got {df.columns[0]!r}"
        )
    value_cols = df.columns[1:]
    total = 0.0
    for col in value_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        total += float(series.fillna(0.0).sum())
    return total


def parse_summary_obj(path: Path) -> float:
    """Thin wrapper around ``_parse_summary_solve_objective`` (M CUR)."""
    return _parse_summary_solve_objective(path)
