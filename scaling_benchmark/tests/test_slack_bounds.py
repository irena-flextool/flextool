"""Unit tests for ``flextool.flextoolrunner.slack_bounds._read_periods_in_use``.

Agent 3b's original helper (commit ``1af169b``) treated the first column
of ``solve_data/set_period_in_use.csv``'s header line (``solve``) as a
period value and leaked the literal string ``"solve"`` into
``p_state_slack_k_rel.csv``, causing glpsol to reject the second solve
of multi-solve scenarios with ``p_state_slack_k_rel[..., solve] out of
domain``.  Agent 14's fix switches the parser to ``csv.DictReader``
keyed on the ``period`` column and optionally filters by ``solve``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make ``flextool`` importable when running the tests in-place.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flextool.flextoolrunner.slack_bounds import _read_periods_in_use


def _write_csv(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


def test_read_periods_in_use_single_solve_with_header(tmp_path: Path) -> None:
    """Basic single-solve CSV with the ``solve,period`` header."""
    csv_path = tmp_path / "set_period_in_use.csv"
    _write_csv(csv_path, [
        "solve,period",
        "base,2025",
        "base,2030",
    ])
    steps = tmp_path / "steps_in_use.csv"  # does not exist

    periods = _read_periods_in_use(csv_path, steps)

    assert periods == ["2025", "2030"]
    assert "solve" not in periods


def test_read_periods_in_use_multi_solve_filters_by_solve(tmp_path: Path) -> None:
    """Multi-solve accumulated CSV: filter to the current solve only."""
    csv_path = tmp_path / "set_period_in_use.csv"
    _write_csv(csv_path, [
        "solve,period",
        "first,2025",
        "first,2030",
        "second,2035",
        "second,2040",
    ])
    steps = tmp_path / "steps_in_use.csv"

    periods_first  = _read_periods_in_use(csv_path, steps, solve="first")
    periods_second = _read_periods_in_use(csv_path, steps, solve="second")

    assert periods_first  == ["2025", "2030"]
    assert periods_second == ["2035", "2040"]
    # The regression — the literal header token must never leak in.
    assert "solve" not in periods_first
    assert "solve" not in periods_second


def test_read_periods_in_use_unknown_solve_returns_empty(tmp_path: Path) -> None:
    """Unknown solve filters everything out; falls back only if the file is
    missing, so here we get an empty list from the CSV and fall through to
    the empty ``steps_in_use.csv`` fallback."""
    csv_path = tmp_path / "set_period_in_use.csv"
    _write_csv(csv_path, [
        "solve,period",
        "first,2025",
    ])
    steps = tmp_path / "steps_in_use.csv"

    periods = _read_periods_in_use(csv_path, steps, solve="not_a_solve")

    assert periods == []


def test_read_periods_in_use_no_solve_arg_dedupes_in_order(tmp_path: Path) -> None:
    """Without a ``solve`` filter, return every unique period in first-seen order."""
    csv_path = tmp_path / "set_period_in_use.csv"
    _write_csv(csv_path, [
        "solve,period",
        "first,2025",
        "first,2030",
        "second,2030",  # duplicate period — must be dropped
        "second,2035",
    ])
    steps = tmp_path / "steps_in_use.csv"

    periods = _read_periods_in_use(csv_path, steps)

    assert periods == ["2025", "2030", "2035"]
    assert "solve" not in periods


def test_read_periods_in_use_falls_back_to_steps_in_use(tmp_path: Path) -> None:
    """When ``set_period_in_use.csv`` is absent, derive periods from
    ``steps_in_use.csv`` in first-seen order."""
    csv_path = tmp_path / "set_period_in_use.csv"  # does not exist
    steps = tmp_path / "steps_in_use.csv"
    _write_csv(steps, [
        "period,step,step_duration",
        "2025,t01,1.0",
        "2025,t02,1.0",
        "2030,t01,1.0",
    ])

    periods = _read_periods_in_use(csv_path, steps)

    assert periods == ["2025", "2030"]


def test_read_periods_in_use_empty_csv_returns_empty(tmp_path: Path) -> None:
    """A CSV with just the header yields no periods (and no ``solve`` leak)."""
    csv_path = tmp_path / "set_period_in_use.csv"
    _write_csv(csv_path, ["solve,period"])
    steps = tmp_path / "steps_in_use.csv"  # does not exist

    periods = _read_periods_in_use(csv_path, steps)

    assert periods == []
