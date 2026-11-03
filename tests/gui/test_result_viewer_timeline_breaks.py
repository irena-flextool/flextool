"""The result viewer reads the timeline break steps as ``(period, time)`` pairs through the shared reader.

The loaders are called on a bare ``ResultViewer`` (no Tk root: only ``_project_path`` and the cache are used).
"""
from __future__ import annotations

import pandas as pd

from flextool.gui.result_viewer import ResultViewer
from flextool.plot_outputs.format_helpers import write_timeline_breaks


def _viewer(project) -> ResultViewer:
    rv = ResultViewer.__new__(ResultViewer)
    rv._project_path = project
    rv._break_times_cache = {}
    return rv


def _write(project, scenario: str, rows) -> None:
    d = project / "output_parquet" / scenario
    d.mkdir(parents=True)
    write_timeline_breaks(d, pd.MultiIndex.from_tuples(rows, names=["period", "time"]))


def test_scenario_breaks_are_period_time_pairs(tmp_path):
    """The per-scenario file yields (period, time) pairs, not the period column."""
    _write(tmp_path, "A", [("p2025", "t0049")])
    assert list(_viewer(tmp_path)._load_break_times("A")) == [("p2025", "t0049")]


def test_scenario_without_breaks_or_file_gives_none(tmp_path):
    _write(tmp_path, "A", [])
    rv = _viewer(tmp_path)
    assert rv._load_break_times("A") is None
    assert rv._load_break_times("missing") is None


def test_comparison_breaks_follow_the_selected_scenarios(tmp_path):
    """The comparison union is cached per selection: changing the selection does not serve the previous union."""
    _write(tmp_path, "A", [("p2020", "t0049")])
    _write(tmp_path, "B", [("p2025", "t0001")])
    rv = _viewer(tmp_path)
    assert sorted(rv._load_comparison_break_times(["A", "B"])) == [("p2020", "t0049"), ("p2025", "t0001")]
    assert list(rv._load_comparison_break_times(["B"])) == [("p2025", "t0001")]
    assert list(rv._load_comparison_break_times(["B", "A"])) == list(rv._load_comparison_break_times(["A", "B"]))
