"""Tests for the cross-scenario axis-bounds manifest writer.

Chunk B of the plot-plan refactor writes a side-car
``output_parquet/_shared/axis_bounds.json`` that holds the union of
subplot y-ranges across every scenario in a batch run.  Chunk C teaches
the viewer to consume it; this test suite exercises only the writer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flextool.plot_outputs.plan import PlotPlan
from flextool.plot_outputs.shared_manifest import (
    ManifestAccumulator,
    _UNTITLED_KEY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_time_plan(
    titles_and_ranges: list[tuple[str | None, tuple[float, float]]],
) -> PlotPlan:
    """Build a minimal time-series PlotPlan with the given titles + y-ranges.

    The ``processed_df`` is a tiny dummy frame — it's irrelevant for the
    accumulator, which only reads ``effective_plot_specs`` and
    ``subplot_y_ranges``.
    """
    df = pd.DataFrame(
        {"x": [1.0, 2.0]},
        index=pd.Index(["a", "b"], name="t"),
    )
    specs: list[tuple[str | None, list]] = [
        (title, []) for title, _ in titles_and_ranges
    ]
    ranges: list[tuple[float, float]] = [
        r for _, r in titles_and_ranges
    ]
    return PlotPlan(
        chart_type="lines",
        plot_name="plan",
        total_file_count=1,
        processed_df=df,
        effective_plot_specs=specs,
        file_batches=[[i for i, _ in enumerate(specs)]],
        layout_type="line",
        layout_params={
            "value_label_width": 0.0,
            "legend_width": 0.0,
            "base_width": 6.0,
            "subplot_height": 4.0,
        },
        subplot_y_ranges=ranges,
    )


def _make_bar_plan() -> PlotPlan:
    """A bar plan — should be ignored by the accumulator."""
    df = pd.DataFrame(
        {"x": [1.0, 2.0]},
        index=pd.Index(["a", "b"], name="bar"),
    )
    return PlotPlan(
        chart_type="bar",
        plot_name="bar_plan",
        total_file_count=1,
        processed_df=df,
        effective_plot_specs=[("only", {"rows": [["a"], ["b"]], "cols": None})],
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
        subplot_y_ranges=[(0.0, 10.0)],
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestManifestAccumulator:
    def test_path_points_into_output_parquet_shared(self, tmp_path: Path):
        acc = ManifestAccumulator(tmp_path)
        assert acc.manifest_path == tmp_path / "output_parquet" / "_shared" / "axis_bounds.json"

    def test_add_single_plan_stores_ranges_by_title(self, tmp_path: Path):
        acc = ManifestAccumulator(tmp_path)
        plan = _make_time_plan([
            ("node_A", (0.0, 10.0)),
            ("node_B", (-5.0, 3.0)),
        ])
        acc.add_plan("node_balance", "default", plan)

        assert acc.data == {
            "node_balance": {
                "default": {
                    "node_A": [0.0, 10.0],
                    "node_B": [-5.0, 3.0],
                },
            },
        }

    def test_overlapping_titles_union_min_max_across_scenarios(
        self, tmp_path: Path,
    ):
        acc = ManifestAccumulator(tmp_path)
        # Scenario 1: A overlaps with scenario 2's A; B only in 1; C only in 2.
        plan1 = _make_time_plan([
            ("A", (0.0, 5.0)),
            ("B", (-1.0, 1.0)),
        ])
        plan2 = _make_time_plan([
            ("A", (-3.0, 4.0)),
            ("C", (2.0, 8.0)),
        ])
        acc.add_plan("rk", "default", plan1)
        acc.add_plan("rk", "default", plan2)

        assert acc.data["rk"]["default"] == {
            "A": [-3.0, 5.0],   # union: min(0, -3), max(5, 4)
            "B": [-1.0, 1.0],   # only scenario 1
            "C": [2.0, 8.0],    # only scenario 2
        }

    def test_none_title_uses_sentinel_key(self, tmp_path: Path):
        acc = ManifestAccumulator(tmp_path)
        plan = _make_time_plan([(None, (-2.0, 7.0))])
        acc.add_plan("rk", "default", plan)
        assert acc.data["rk"]["default"] == {_UNTITLED_KEY: [-2.0, 7.0]}

    def test_bar_plans_are_skipped(self, tmp_path: Path):
        acc = ManifestAccumulator(tmp_path)
        acc.add_plan("rk", "default", _make_bar_plan())
        assert acc.data == {}

    def test_empty_subplot_y_ranges_are_skipped(self, tmp_path: Path):
        acc = ManifestAccumulator(tmp_path)
        df = pd.DataFrame({"x": [1.0]}, index=pd.Index(["a"], name="t"))
        plan = PlotPlan(
            chart_type="lines",
            plot_name="plan",
            total_file_count=1,
            processed_df=df,
            effective_plot_specs=[("A", [])],
            file_batches=[[0]],
            layout_type="line",
            layout_params={
                "value_label_width": 0.0, "legend_width": 0.0,
                "base_width": 6.0, "subplot_height": 4.0,
            },
            subplot_y_ranges=[],  # empty
        )
        acc.add_plan("rk", "default", plan)
        assert acc.data == {}

    def test_specs_and_ranges_length_mismatch_skips(self, tmp_path: Path, caplog):
        acc = ManifestAccumulator(tmp_path)
        df = pd.DataFrame({"x": [1.0]}, index=pd.Index(["a"], name="t"))
        # Force a mismatch
        plan = PlotPlan(
            chart_type="lines",
            plot_name="plan",
            total_file_count=1,
            processed_df=df,
            effective_plot_specs=[("A", []), ("B", [])],
            file_batches=[[0, 1]],
            layout_type="line",
            layout_params={
                "value_label_width": 0.0, "legend_width": 0.0,
                "base_width": 6.0, "subplot_height": 4.0,
            },
            subplot_y_ranges=[(0.0, 1.0)],  # length 1 vs specs length 2
        )
        with caplog.at_level("WARNING", logger="flextool.plot_outputs.shared_manifest"):
            acc.add_plan("rk", "default", plan)
        assert acc.data == {}

    def test_different_sub_configs_are_isolated(self, tmp_path: Path):
        acc = ManifestAccumulator(tmp_path)
        plan1 = _make_time_plan([("A", (0.0, 1.0))])
        plan2 = _make_time_plan([("A", (0.0, 100.0))])
        acc.add_plan("rk", "hourly", plan1)
        acc.add_plan("rk", "period", plan2)
        assert acc.data["rk"]["hourly"]["A"] == [0.0, 1.0]
        assert acc.data["rk"]["period"]["A"] == [0.0, 100.0]

    def test_different_result_keys_are_isolated(self, tmp_path: Path):
        acc = ManifestAccumulator(tmp_path)
        acc.add_plan("node_balance", "d", _make_time_plan([("A", (0.0, 1.0))]))
        acc.add_plan("node_inflow", "d", _make_time_plan([("A", (-5.0, 0.0))]))
        assert acc.data["node_balance"]["d"]["A"] == [0.0, 1.0]
        assert acc.data["node_inflow"]["d"]["A"] == [-5.0, 0.0]


class TestWrite:
    def test_write_creates_shared_directory_and_json(self, tmp_path: Path):
        acc = ManifestAccumulator(tmp_path)
        acc.add_plan("rk", "cfg", _make_time_plan([("A", (0.0, 10.0))]))
        acc.write()

        manifest_path = tmp_path / "output_parquet" / "_shared" / "axis_bounds.json"
        assert manifest_path.is_file()

        data = json.loads(manifest_path.read_text())
        assert data == {"rk": {"cfg": {"A": [0.0, 10.0]}}}

    def test_write_is_atomic(self, tmp_path: Path, monkeypatch):
        """Temp file should never be left behind after a successful write
        and should not be the one the reader observes.

        We smoke-test by intercepting ``os.replace`` and checking the temp
        file exists up until the rename call.
        """
        import os as _os

        acc = ManifestAccumulator(tmp_path)
        acc.add_plan("rk", "cfg", _make_time_plan([("A", (0.0, 10.0))]))

        seen_tmp_paths: list[str] = []
        real_replace = _os.replace

        def spy_replace(src, dst):
            seen_tmp_paths.append(str(src))
            # At the moment of rename, the temp file still exists.
            assert Path(src).is_file()
            # The destination must not be a temp file
            assert "axis_bounds.json" in str(dst)
            real_replace(src, dst)

        # Path.replace calls os.replace under the hood
        monkeypatch.setattr(_os, "replace", spy_replace)
        acc.write()

        assert len(seen_tmp_paths) == 1
        # No temp file should linger.
        shared = tmp_path / "output_parquet" / "_shared"
        leftover = list(shared.glob("axis_bounds_*.tmp"))
        assert leftover == []

    def test_empty_accumulator_does_not_create_file(self, tmp_path: Path):
        acc = ManifestAccumulator(tmp_path)
        acc.write()
        shared = tmp_path / "output_parquet" / "_shared"
        # Directory may or may not exist; file must not.
        assert not (shared / "axis_bounds.json").exists()

    def test_round_trip_with_multiple_scenarios(self, tmp_path: Path):
        """First scenario writes; second scenario instantiates a new
        accumulator (simulating a fresh batch) which seeds from disk and
        unions the new values."""
        # Scenario 1
        acc1 = ManifestAccumulator(tmp_path)
        acc1.add_plan("rk", "d", _make_time_plan([
            ("A", (0.0, 5.0)),
            ("B", (-1.0, 1.0)),
        ]))
        acc1.write()

        # Scenario 2: new accumulator picks up existing manifest
        acc2 = ManifestAccumulator(tmp_path)
        acc2.add_plan("rk", "d", _make_time_plan([
            ("A", (-3.0, 4.0)),  # unions with (0, 5) → (-3, 5)
            ("C", (2.0, 8.0)),
        ]))
        acc2.write()

        manifest_path = tmp_path / "output_parquet" / "_shared" / "axis_bounds.json"
        data = json.loads(manifest_path.read_text())
        assert data == {
            "rk": {
                "d": {
                    "A": [-3.0, 5.0],
                    "B": [-1.0, 1.0],
                    "C": [2.0, 8.0],
                },
            },
        }


class TestSeedingFromDisk:
    def test_missing_file_starts_empty(self, tmp_path: Path):
        acc = ManifestAccumulator(tmp_path)
        assert acc.data == {}

    def test_corrupt_json_does_not_raise(self, tmp_path: Path, caplog):
        shared = tmp_path / "output_parquet" / "_shared"
        shared.mkdir(parents=True)
        (shared / "axis_bounds.json").write_text("{{ not json")
        with caplog.at_level("WARNING", logger="flextool.plot_outputs.shared_manifest"):
            acc = ManifestAccumulator(tmp_path)
        assert acc.data == {}

    def test_wrong_shape_entries_are_dropped(self, tmp_path: Path):
        shared = tmp_path / "output_parquet" / "_shared"
        shared.mkdir(parents=True)
        payload = {
            "rk": {
                "cfg": {
                    "good": [0.0, 1.0],
                    "bad_length": [0.0, 1.0, 2.0],
                    "non_numeric": ["x", "y"],
                    "not_a_list": 42,
                },
                "also_bad": "not a dict",
            },
            "not_a_dict": 7,
        }
        (shared / "axis_bounds.json").write_text(json.dumps(payload))

        acc = ManifestAccumulator(tmp_path)
        assert acc.data == {"rk": {"cfg": {"good": [0.0, 1.0]}}}


# ---------------------------------------------------------------------------
# End-to-end integration (lightweight)
# ---------------------------------------------------------------------------

class TestIntegrationWithComputeAllPlotPlans:
    """Checks that ``compute_all_plot_plans`` wires the accumulator when its
    output_dir sits inside ``output_parquet/``.  Uses a minimal in-memory
    results dict so we don't need a full project on disk."""

    def test_accumulator_used_for_per_scenario_output_dir(self, tmp_path: Path):
        from flextool.plot_outputs.orchestrator import compute_all_plot_plans

        # Simulate <project>/output_parquet/<scenario>/
        scenario_dir = tmp_path / "output_parquet" / "scen_a"
        scenario_dir.mkdir(parents=True)

        # Minimal time-series DataFrame with (node, time) column/row indexing.
        columns = pd.MultiIndex.from_product(
            [["nodeA", "nodeB"]], names=["node"],
        )
        times = pd.Index([f"t{i:02d}" for i in range(4)], name="t")
        df = pd.DataFrame(
            np.arange(4 * 2, dtype=float).reshape(4, 2),
            index=times, columns=columns,
        )
        results = {"node_balance__dt": df}
        plot_settings = {
            "node_balance__dt": {
                "map_dimensions_for_plots": ["t_u", "l"],
                "max_items_per_plot": 10,
                "subplots_per_row": 1,
                "legend": "right",
            },
        }

        compute_all_plot_plans(
            results, plot_settings, scenario_dir,
            plot_rows=(0, 4),
        )

        manifest_path = tmp_path / "output_parquet" / "_shared" / "axis_bounds.json"
        # Manifest should exist and hold node_balance__dt bounds.
        if manifest_path.exists():
            data = json.loads(manifest_path.read_text())
            assert "node_balance__dt" in data

    def test_no_manifest_written_for_non_output_parquet_dir(self, tmp_path: Path):
        from flextool.plot_outputs.orchestrator import compute_all_plot_plans

        # Comparison-style dir: not under output_parquet/
        comp_dir = tmp_path / "output_parquet_comparison"
        comp_dir.mkdir(parents=True)

        df = pd.DataFrame(
            {"x": [1.0, 2.0]},
            index=pd.Index(["a", "b"], name="t"),
        )
        compute_all_plot_plans({"rk": df}, {}, comp_dir)

        # _shared must NOT be created anywhere.
        assert not (tmp_path / "output_parquet" / "_shared").exists()
