"""Tests for PlotPlan JSON serialization of time-axis labels.

Chunk A of the plot-plan refactor drops these fields from the on-disk
JSON — ``load_plot_plan`` reconstructs them from ``processed_df.index``.
Old-format JSON files that still carry the keys must continue to load.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from flextool.plot_outputs.config import PlotConfig
from flextool.plot_outputs.plan import (
    PlotPlan,
    _compute_time_plan,
    _extract_time_labels,
    load_plot_plan,
    save_plot_plan,
)


def _make_time_plan(with_period: bool = True) -> PlotPlan:
    sub_vals = ["A", "B"]
    items = [f"item_{i}" for i in range(3)]
    columns = pd.MultiIndex.from_product(
        [sub_vals, items], names=["subplot", "line"],
    )
    times = [f"t{i:02d}" for i in range(6)]
    if with_period:
        periods = ["p1", "p1", "p1", "p2", "p2", "p2"]
        index = pd.MultiIndex.from_arrays(
            [periods, times], names=["period", "t"],
        )
    else:
        index = pd.Index(times, name="t")
    df = pd.DataFrame(
        np.arange(len(index) * len(columns), dtype=float).reshape(
            len(index), len(columns)
        ),
        index=index,
        columns=columns,
    )
    cfg = PlotConfig(
        map_dimensions_for_plots=["d", "u", "l"],
        max_items_per_plot=10,
        max_subplots_per_file=10,
        subplots_per_row=1,
        legend="right",
    )
    plan = _compute_time_plan(
        df, "time_test", cfg,
        fm_stack_levels=[], fm_subplot_levels=[0], fm_line_levels=[1],
        axis_bounds=None, plot_rows=(0, len(index)),
    )
    assert plan is not None
    return plan


class TestExtractTimeLabelsHelper:
    def test_multiindex_with_period_level(self):
        index = pd.MultiIndex.from_arrays(
            [["p1", "p1", "p2"], ["t1", "t2", "t3"]],
            names=["period", "t"],
        )
        df = pd.DataFrame({"x": [1, 2, 3]}, index=index)
        times, periods, max_len = _extract_time_labels(df)
        assert times == ["t1", "t2", "t3"]
        assert periods == ["p1", "p1", "p2"]
        assert max_len == 2

    def test_multiindex_with_solve_period_level(self):
        index = pd.MultiIndex.from_arrays(
            [["p1", "long_name"], ["t1", "t2"]],
            names=["Solve_Period", "t"],  # case-insensitive match
        )
        df = pd.DataFrame({"x": [1, 2]}, index=index)
        times, periods, max_len = _extract_time_labels(df)
        assert times == ["t1", "t2"]
        assert periods == ["p1", "long_name"]
        assert max_len == len("long_name")

    def test_multiindex_without_period_level(self):
        index = pd.MultiIndex.from_arrays(
            [["s1", "s1"], ["t1", "t2"]],
            names=["scenario", "t"],
        )
        df = pd.DataFrame({"x": [1, 2]}, index=index)
        times, periods, max_len = _extract_time_labels(df)
        assert times == ["t1", "t2"]
        assert periods is None
        assert max_len == 0

    def test_plain_index(self):
        df = pd.DataFrame({"x": [1, 2, 3]}, index=pd.Index(["a", "b", "c"], name="t"))
        times, periods, max_len = _extract_time_labels(df)
        assert times == ["a", "b", "c"]
        assert periods is None
        assert max_len == 0


class TestRoundTrip:
    def test_save_load_preserves_time_fields(self, tmp_path: Path):
        plan = _make_time_plan(with_period=True)
        original_times = list(plan.time_index_values)
        original_periods = list(plan.period_labels)
        original_max_len = plan.max_period_label_len

        save_plot_plan(plan, tmp_path, "rk", "cfg")
        loaded = load_plot_plan(tmp_path, "rk", "cfg")
        assert loaded is not None
        assert loaded.time_index_values == original_times
        assert loaded.period_labels == original_periods
        assert loaded.max_period_label_len == original_max_len

    def test_save_load_preserves_time_fields_no_period(self, tmp_path: Path):
        plan = _make_time_plan(with_period=False)
        original_times = list(plan.time_index_values)
        assert plan.period_labels is None

        save_plot_plan(plan, tmp_path, "rk", "cfg")
        loaded = load_plot_plan(tmp_path, "rk", "cfg")
        assert loaded is not None
        assert loaded.time_index_values == original_times
        assert loaded.period_labels is None
        assert loaded.max_period_label_len == 0

    def test_new_json_does_not_contain_dropped_keys(self, tmp_path: Path):
        plan = _make_time_plan(with_period=True)
        save_plot_plan(plan, tmp_path, "rk", "cfg")
        json_path = tmp_path / "rk__cfg_plan.json"
        data = json.loads(json_path.read_text())
        assert "time_index_values" not in data
        assert "period_labels" not in data
        assert "max_period_label_len" not in data


class TestLegacyJsonCompat:
    def test_old_format_json_with_fields_still_loads(self, tmp_path: Path):
        # Build a plan, save it, then surgically add the dropped keys to
        # the JSON to simulate an old-format file on disk.
        plan = _make_time_plan(with_period=True)
        save_plot_plan(plan, tmp_path, "rk", "old")
        json_path = tmp_path / "rk__old_plan.json"
        data = json.loads(json_path.read_text())
        # Hand-craft "legacy" values so we can verify they win over the
        # reconstruction path.
        legacy_times = ["legacy_t0", "legacy_t1", "legacy_t2",
                        "legacy_t3", "legacy_t4", "legacy_t5"]
        legacy_periods = ["LP1", "LP1", "LP1", "LP2", "LP2", "LP2"]
        data["time_index_values"] = legacy_times
        data["period_labels"] = legacy_periods
        data["max_period_label_len"] = 3
        json_path.write_text(json.dumps(data))

        loaded = load_plot_plan(tmp_path, "rk", "old")
        assert loaded is not None
        assert loaded.time_index_values == legacy_times
        assert loaded.period_labels == legacy_periods
        assert loaded.max_period_label_len == 3


class TestBarChartNoTimeFields:
    def test_bar_chart_time_fields_remain_none_after_roundtrip(self, tmp_path: Path):
        # Build a minimal bar plan by hand — _compute_bar_plan requires
        # more setup than we need for this serialization check.
        df = pd.DataFrame(
            {"x": [1.0, 2.0, 3.0]},
            index=pd.Index(["a", "b", "c"], name="bar"),
        )
        plan = PlotPlan(
            chart_type="bar",
            plot_name="bar_test",
            total_file_count=1,
            processed_df=df,
            effective_plot_specs=[(None, {"rows": [["a"], ["b"], ["c"]], "cols": None})],
            file_batches=[[0]],
            layout_type="bar",
            layout_params={
                "bar_label_width": 1.0,
                "group_label_width": 0.0,
                "total_label_width": 1.0,
                "legend_width": 2.0,
                "legend_height": 1.0,
                "base_bar_length": 4.0,
                "value_axis_width": 1.0,
            },
        )
        save_plot_plan(plan, tmp_path, "rk", "bar")
        loaded = load_plot_plan(tmp_path, "rk", "bar")
        assert loaded is not None
        assert loaded.chart_type == "bar"
        assert loaded.time_index_values is None
        assert loaded.period_labels is None
        assert loaded.max_period_label_len == 0
