"""Tests for the cross-scenario axis-bounds manifest.

Option B schema:

    {
      "<result_key>": {
        "<sub_config>": {
          "<subplot_title>": {
            "<scenario_name>": [min, max],
            ...
          }
        }
      }
    }

The writer stores per-scenario slices; the reader unions over a
caller-supplied set of active scenarios (or over all scenarios when the
set is ``None``).
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
    apply_manifest_to_plan,
    load_axis_bounds_manifest,
    remove_scenario_from_manifest,
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
# Unit tests — accumulator
# ---------------------------------------------------------------------------

class TestManifestAccumulator:
    def test_path_points_into_output_parquet_shared(self, tmp_path: Path):
        acc = ManifestAccumulator(tmp_path)
        assert acc.manifest_path == tmp_path / "output_parquet" / "_shared" / "axis_bounds.json"

    def test_add_single_plan_stores_ranges_by_scenario(self, tmp_path: Path):
        acc = ManifestAccumulator(tmp_path)
        plan = _make_time_plan([
            ("node_A", (0.0, 10.0)),
            ("node_B", (-5.0, 3.0)),
        ])
        acc.add_plan("node_balance", "default", plan, "scen1")

        assert acc.data == {
            "node_balance": {
                "default": {
                    "node_A": {"scen1": [0.0, 10.0]},
                    "node_B": {"scen1": [-5.0, 3.0]},
                },
            },
        }

    def test_two_scenarios_stored_side_by_side(self, tmp_path: Path):
        acc = ManifestAccumulator(tmp_path)
        plan1 = _make_time_plan([
            ("A", (0.0, 5.0)),
            ("B", (-1.0, 1.0)),
        ])
        plan2 = _make_time_plan([
            ("A", (-3.0, 4.0)),
            ("C", (2.0, 8.0)),
        ])
        acc.add_plan("rk", "default", plan1, "s1")
        acc.add_plan("rk", "default", plan2, "s2")

        # Each scenario keeps its own entry; no union at write time.
        assert acc.data["rk"]["default"] == {
            "A": {"s1": [0.0, 5.0], "s2": [-3.0, 4.0]},
            "B": {"s1": [-1.0, 1.0]},
            "C": {"s2": [2.0, 8.0]},
        }

    def test_replaying_same_scenario_replaces_its_slice(self, tmp_path: Path):
        """Second add_plan call with the same scenario name should
        replace that scenario's entries (not union)."""
        acc = ManifestAccumulator(tmp_path)

        # Seed disk with old values for two scenarios.
        acc.add_plan(
            "rk", "default",
            _make_time_plan([("A", (0.0, 5.0)), ("B", (-2.0, 2.0))]),
            "s1",
        )
        acc.add_plan(
            "rk", "default",
            _make_time_plan([("A", (10.0, 20.0))]),
            "s2",
        )
        acc.write()

        # New accumulator: simulate a fresh batch re-running scenario s1
        # with different bounds and dropping subplot B entirely.
        acc2 = ManifestAccumulator(tmp_path)
        acc2.add_plan(
            "rk", "default",
            _make_time_plan([("A", (100.0, 200.0))]),
            "s1",
        )
        # s1 now holds only the new A entry — old B slice is gone.
        # s2 is untouched.
        assert acc2.data["rk"]["default"]["A"] == {
            "s1": [100.0, 200.0],
            "s2": [10.0, 20.0],
        }
        assert "B" not in acc2.data["rk"]["default"]

    def test_none_title_uses_sentinel_key(self, tmp_path: Path):
        acc = ManifestAccumulator(tmp_path)
        plan = _make_time_plan([(None, (-2.0, 7.0))])
        acc.add_plan("rk", "default", plan, "s1")
        assert acc.data["rk"]["default"] == {_UNTITLED_KEY: {"s1": [-2.0, 7.0]}}

    def test_bar_plans_are_skipped(self, tmp_path: Path):
        acc = ManifestAccumulator(tmp_path)
        acc.add_plan("rk", "default", _make_bar_plan(), "s1")
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
        acc.add_plan("rk", "default", plan, "s1")
        assert acc.data == {}

    def test_specs_and_ranges_length_mismatch_skips(self, tmp_path: Path, caplog):
        acc = ManifestAccumulator(tmp_path)
        df = pd.DataFrame({"x": [1.0]}, index=pd.Index(["a"], name="t"))
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
            subplot_y_ranges=[(0.0, 1.0)],
        )
        with caplog.at_level("WARNING", logger="flextool.plot_outputs.shared_manifest"):
            acc.add_plan("rk", "default", plan, "s1")
        assert acc.data == {}

    def test_different_sub_configs_are_isolated(self, tmp_path: Path):
        acc = ManifestAccumulator(tmp_path)
        plan1 = _make_time_plan([("A", (0.0, 1.0))])
        plan2 = _make_time_plan([("A", (0.0, 100.0))])
        acc.add_plan("rk", "hourly", plan1, "s1")
        acc.add_plan("rk", "period", plan2, "s1")
        assert acc.data["rk"]["hourly"]["A"] == {"s1": [0.0, 1.0]}
        assert acc.data["rk"]["period"]["A"] == {"s1": [0.0, 100.0]}

    def test_different_result_keys_are_isolated(self, tmp_path: Path):
        acc = ManifestAccumulator(tmp_path)
        acc.add_plan(
            "node_balance", "d",
            _make_time_plan([("A", (0.0, 1.0))]),
            "s1",
        )
        acc.add_plan(
            "node_inflow", "d",
            _make_time_plan([("A", (-5.0, 0.0))]),
            "s1",
        )
        assert acc.data["node_balance"]["d"]["A"] == {"s1": [0.0, 1.0]}
        assert acc.data["node_inflow"]["d"]["A"] == {"s1": [-5.0, 0.0]}

    def test_empty_scenario_name_is_skipped(self, tmp_path: Path, caplog):
        acc = ManifestAccumulator(tmp_path)
        with caplog.at_level("WARNING", logger="flextool.plot_outputs.shared_manifest"):
            acc.add_plan(
                "rk", "d",
                _make_time_plan([("A", (0.0, 1.0))]),
                "",
            )
        assert acc.data == {}


class TestWrite:
    def test_write_creates_shared_directory_and_json(self, tmp_path: Path):
        acc = ManifestAccumulator(tmp_path)
        acc.add_plan("rk", "cfg", _make_time_plan([("A", (0.0, 10.0))]), "s1")
        acc.write()

        manifest_path = tmp_path / "output_parquet" / "_shared" / "axis_bounds.json"
        assert manifest_path.is_file()

        data = json.loads(manifest_path.read_text())
        assert data == {"rk": {"cfg": {"A": {"s1": [0.0, 10.0]}}}}

    def test_write_is_atomic(self, tmp_path: Path, monkeypatch):
        """Temp file should never be left behind after a successful write."""
        import os as _os

        acc = ManifestAccumulator(tmp_path)
        acc.add_plan("rk", "cfg", _make_time_plan([("A", (0.0, 10.0))]), "s1")

        seen_tmp_paths: list[str] = []
        real_replace = _os.replace

        def spy_replace(src, dst):
            seen_tmp_paths.append(str(src))
            assert Path(src).is_file()
            assert "axis_bounds.json" in str(dst)
            real_replace(src, dst)

        monkeypatch.setattr(_os, "replace", spy_replace)
        acc.write()

        assert len(seen_tmp_paths) == 1
        shared = tmp_path / "output_parquet" / "_shared"
        leftover = list(shared.glob("axis_bounds_*.tmp"))
        assert leftover == []

    def test_empty_accumulator_does_not_create_file(self, tmp_path: Path):
        acc = ManifestAccumulator(tmp_path)
        acc.write()
        shared = tmp_path / "output_parquet" / "_shared"
        assert not (shared / "axis_bounds.json").exists()

    def test_round_trip_preserves_other_scenarios(self, tmp_path: Path):
        """Scenario 1 writes; a fresh batch re-running scenario 2 keeps
        scenario 1's slice intact."""
        acc1 = ManifestAccumulator(tmp_path)
        acc1.add_plan(
            "rk", "d",
            _make_time_plan([("A", (0.0, 5.0)), ("B", (-1.0, 1.0))]),
            "s1",
        )
        acc1.write()

        acc2 = ManifestAccumulator(tmp_path)
        acc2.add_plan(
            "rk", "d",
            _make_time_plan([("A", (-3.0, 4.0)), ("C", (2.0, 8.0))]),
            "s2",
        )
        acc2.write()

        manifest_path = tmp_path / "output_parquet" / "_shared" / "axis_bounds.json"
        data = json.loads(manifest_path.read_text())
        assert data == {
            "rk": {
                "d": {
                    "A": {"s1": [0.0, 5.0], "s2": [-3.0, 4.0]},
                    "B": {"s1": [-1.0, 1.0]},
                    "C": {"s2": [2.0, 8.0]},
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
                    "good": {"s1": [0.0, 1.0]},
                    "bad_scenario_bounds": {"s2": [0.0, 1.0, 2.0]},
                    "non_numeric_scenario_bounds": {"s3": ["x", "y"]},
                    "not_a_dict": 42,
                    "scenario_int_key": {5: [0.0, 1.0]},  # non-string scenario
                },
                "also_bad": "not a dict",
            },
            "not_a_dict": 7,
        }
        (shared / "axis_bounds.json").write_text(json.dumps(payload))

        acc = ManifestAccumulator(tmp_path)
        # Only the well-formed entry survives.  JSON coerces int keys to
        # strings, so "scenario_int_key" actually survives — but the
        # inner scenario name becomes "5" (JSON has no integer keys).
        assert acc.data == {
            "rk": {
                "cfg": {
                    "good": {"s1": [0.0, 1.0]},
                    "scenario_int_key": {"5": [0.0, 1.0]},
                },
            },
        }

    def test_legacy_flat_entries_are_dropped_with_warning(
        self, tmp_path: Path, caplog,
    ):
        """Old-format manifest (value is a 2-tuple/list instead of a dict)
        should be treated as missing — the next write rewrites it in the
        new schema."""
        shared = tmp_path / "output_parquet" / "_shared"
        shared.mkdir(parents=True)
        payload = {
            "rk": {
                "cfg": {
                    "A": [0.0, 10.0],            # legacy
                    "B": {"s1": [-1.0, 1.0]},    # new
                },
            },
        }
        (shared / "axis_bounds.json").write_text(json.dumps(payload))

        with caplog.at_level(
            "WARNING", logger="flextool.plot_outputs.shared_manifest",
        ):
            acc = ManifestAccumulator(tmp_path)
        # Legacy A is silently dropped; new B survives.
        assert acc.data == {"rk": {"cfg": {"B": {"s1": [-1.0, 1.0]}}}}


# ---------------------------------------------------------------------------
# remove_scenario_from_manifest
# ---------------------------------------------------------------------------

class TestRemoveScenarioFromManifest:
    def test_missing_file_returns_false(self, tmp_path: Path):
        assert remove_scenario_from_manifest(tmp_path, "s1") is False

    def test_missing_scenario_name_returns_false(self, tmp_path: Path):
        assert remove_scenario_from_manifest(tmp_path, "") is False

    def test_removes_scenario_across_subplots_and_keys(self, tmp_path: Path):
        """Removing a scenario should strip all its entries while
        leaving other scenarios' slices intact."""
        acc = ManifestAccumulator(tmp_path)
        acc.add_plan(
            "rk", "d",
            _make_time_plan([("A", (0.0, 5.0)), ("B", (-1.0, 1.0))]),
            "s1",
        )
        acc.add_plan(
            "rk", "d",
            _make_time_plan([("A", (-3.0, 4.0)), ("C", (2.0, 8.0))]),
            "s2",
        )
        acc.add_plan(
            "other", "d",
            _make_time_plan([("X", (0.0, 1.0))]),
            "s1",
        )
        acc.write()

        # Removing s1 should leave s2's entries (A=[−3,4], C=[2,8]) and
        # drop "other/d/X" (was only s1) and B (only s1).
        changed = remove_scenario_from_manifest(tmp_path, "s1")
        assert changed is True

        data = load_axis_bounds_manifest(tmp_path)
        assert data == {
            "rk": {
                "d": {
                    "A": {"s2": [-3.0, 4.0]},
                    "C": {"s2": [2.0, 8.0]},
                },
            },
        }

    def test_noop_when_scenario_absent(self, tmp_path: Path):
        acc = ManifestAccumulator(tmp_path)
        acc.add_plan(
            "rk", "d", _make_time_plan([("A", (0.0, 1.0))]), "s1",
        )
        acc.write()
        assert remove_scenario_from_manifest(tmp_path, "unknown") is False

    def test_removes_file_when_empty_after_strip(self, tmp_path: Path):
        """If removing the only scenario leaves an empty dict, the
        manifest file itself should be deleted."""
        acc = ManifestAccumulator(tmp_path)
        acc.add_plan(
            "rk", "d", _make_time_plan([("A", (0.0, 1.0))]), "s1",
        )
        acc.write()

        manifest_path = tmp_path / "output_parquet" / "_shared" / "axis_bounds.json"
        assert manifest_path.is_file()

        assert remove_scenario_from_manifest(tmp_path, "s1") is True
        assert not manifest_path.exists()


# ---------------------------------------------------------------------------
# End-to-end integration (lightweight)
# ---------------------------------------------------------------------------

class TestIntegrationWithComputeAllPlotPlans:
    """Checks that ``compute_all_plot_plans`` wires the accumulator and
    uses the output-dir folder name as the scenario key."""

    def test_accumulator_used_for_per_scenario_output_dir(self, tmp_path: Path):
        from flextool.plot_outputs.orchestrator import compute_all_plot_plans

        scenario_dir = tmp_path / "output_parquet" / "scen_a"
        scenario_dir.mkdir(parents=True)

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
        if manifest_path.exists():
            data = json.loads(manifest_path.read_text())
            # The scenario key should be the folder name.
            assert "node_balance__dt" in data
            sub_map = data["node_balance__dt"]
            for subplot_map in sub_map.values():
                for scen_map in subplot_map.values():
                    assert "scen_a" in scen_map

    def test_no_manifest_written_for_non_output_parquet_dir(self, tmp_path: Path):
        from flextool.plot_outputs.orchestrator import compute_all_plot_plans

        comp_dir = tmp_path / "output_parquet_comparison"
        comp_dir.mkdir(parents=True)

        df = pd.DataFrame(
            {"x": [1.0, 2.0]},
            index=pd.Index(["a", "b"], name="t"),
        )
        compute_all_plot_plans({"rk": df}, {}, comp_dir)

        assert not (tmp_path / "output_parquet" / "_shared").exists()


# ---------------------------------------------------------------------------
# Reader API tests
# ---------------------------------------------------------------------------

class TestLoadAxisBoundsManifest:
    def test_missing_file_returns_none(self, tmp_path: Path):
        assert load_axis_bounds_manifest(tmp_path) is None

    def test_valid_file_parses(self, tmp_path: Path):
        shared = tmp_path / "output_parquet" / "_shared"
        shared.mkdir(parents=True)
        payload = {"rk": {"cfg": {"A": {"s1": [0.0, 10.0]}}}}
        (shared / "axis_bounds.json").write_text(json.dumps(payload))

        data = load_axis_bounds_manifest(tmp_path)
        assert data == payload

    def test_malformed_json_returns_none(self, tmp_path: Path, caplog):
        shared = tmp_path / "output_parquet" / "_shared"
        shared.mkdir(parents=True)
        (shared / "axis_bounds.json").write_text("not json { ")
        with caplog.at_level(
            "WARNING", logger="flextool.plot_outputs.shared_manifest",
        ):
            data = load_axis_bounds_manifest(tmp_path)
        assert data is None

    def test_non_object_top_level_returns_none(self, tmp_path: Path):
        shared = tmp_path / "output_parquet" / "_shared"
        shared.mkdir(parents=True)
        (shared / "axis_bounds.json").write_text(json.dumps([1, 2, 3]))
        assert load_axis_bounds_manifest(tmp_path) is None

    def test_accepts_str_path(self, tmp_path: Path):
        shared = tmp_path / "output_parquet" / "_shared"
        shared.mkdir(parents=True)
        (shared / "axis_bounds.json").write_text("{}")
        assert load_axis_bounds_manifest(str(tmp_path)) == {}


class TestApplyManifestToPlan:
    def _manifest(self, **per_subplot: dict[str, list[float]]) -> dict:
        return {"rk": {"cfg": dict(per_subplot)}}

    def test_active_none_unions_over_all_scenarios(self):
        plan = _make_time_plan([
            ("A", (0.0, 1.0)),
            ("B", (-1.0, 1.0)),
        ])
        manifest = self._manifest(
            A={"s1": [-5.0, 5.0], "s2": [0.0, 10.0]},
            B={"s1": [-2.0, 2.0]},
        )
        changed = apply_manifest_to_plan(
            plan, manifest, "rk", "cfg", active_scenarios=None,
        )
        assert changed is True
        # A unions s1+s2: min(-5,0)=-5, max(5,10)=10.  B: s1 only.
        assert plan.subplot_y_ranges == [(-5.0, 10.0), (-2.0, 2.0)]

    def test_active_subset_filters_to_those_scenarios(self):
        plan = _make_time_plan([("A", (0.0, 1.0))])
        manifest = self._manifest(
            A={"s1": [-5.0, 5.0], "s2": [0.0, 10.0], "s3": [-100.0, 100.0]},
        )
        changed = apply_manifest_to_plan(
            plan, manifest, "rk", "cfg", active_scenarios={"s1", "s2"},
        )
        assert changed is True
        # Only s1 and s2 contribute: min(-5,0)=-5, max(5,10)=10.  s3 is excluded.
        assert plan.subplot_y_ranges == [(-5.0, 10.0)]

    def test_active_single_scenario(self):
        plan = _make_time_plan([("A", (0.0, 1.0))])
        manifest = self._manifest(
            A={"a": [-5.0, 5.0], "b": [-100.0, 100.0]},
        )
        changed = apply_manifest_to_plan(
            plan, manifest, "rk", "cfg", active_scenarios={"a"},
        )
        assert changed is True
        assert plan.subplot_y_ranges == [(-5.0, 5.0)]

    def test_empty_active_set_is_noop(self):
        plan = _make_time_plan([("A", (0.0, 1.0))])
        manifest = self._manifest(A={"s1": [-5.0, 5.0]})
        changed = apply_manifest_to_plan(
            plan, manifest, "rk", "cfg", active_scenarios=set(),
        )
        assert changed is False
        assert plan.subplot_y_ranges == [(0.0, 1.0)]

    def test_active_scenarios_all_missing_from_manifest_leaves_plan_unchanged(self):
        plan = _make_time_plan([("A", (0.0, 1.0))])
        manifest = self._manifest(A={"s1": [-5.0, 5.0]})
        changed = apply_manifest_to_plan(
            plan, manifest, "rk", "cfg", active_scenarios={"unknown"},
        )
        assert changed is False
        assert plan.subplot_y_ranges == [(0.0, 1.0)]

    def test_missing_manifest_is_noop(self):
        plan = _make_time_plan([("A", (0.0, 1.0))])
        original = list(plan.subplot_y_ranges)
        assert apply_manifest_to_plan(plan, None, "rk", "cfg") is False
        assert plan.subplot_y_ranges == original

    def test_missing_result_key_is_noop(self):
        plan = _make_time_plan([("A", (0.0, 1.0))])
        original = list(plan.subplot_y_ranges)
        manifest = {"other_rk": {"cfg": {"A": {"s1": [-5.0, 5.0]}}}}
        assert apply_manifest_to_plan(plan, manifest, "rk", "cfg") is False
        assert plan.subplot_y_ranges == original

    def test_missing_sub_config_is_noop(self):
        plan = _make_time_plan([("A", (0.0, 1.0))])
        original = list(plan.subplot_y_ranges)
        manifest = {"rk": {"other_cfg": {"A": {"s1": [-5.0, 5.0]}}}}
        assert apply_manifest_to_plan(plan, manifest, "rk", "cfg") is False
        assert plan.subplot_y_ranges == original

    def test_subset_override_keeps_untouched_subplots(self):
        plan = _make_time_plan([
            ("A", (0.0, 1.0)),
            ("B", (-1.0, 1.0)),
            ("C", (10.0, 20.0)),
        ])
        # Only B is in the manifest — A and C keep their original ranges.
        manifest = self._manifest(B={"s1": [-100.0, 100.0]})
        changed = apply_manifest_to_plan(plan, manifest, "rk", "cfg")
        assert changed is True
        assert plan.subplot_y_ranges == [
            (0.0, 1.0),
            (-100.0, 100.0),
            (10.0, 20.0),
        ]

    def test_extra_manifest_entries_are_ignored(self):
        plan = _make_time_plan([("A", (0.0, 1.0))])
        manifest = self._manifest(
            A={"s1": [-1.0, 2.0]},
            Z={"s1": [0.0, 999.0]},
        )
        changed = apply_manifest_to_plan(plan, manifest, "rk", "cfg")
        assert changed is True
        assert plan.subplot_y_ranges == [(-1.0, 2.0)]

    def test_none_title_uses_sentinel(self):
        plan = _make_time_plan([(None, (0.0, 1.0))])
        manifest = {"rk": {"cfg": {_UNTITLED_KEY: {"s1": [-3.0, 4.0]}}}}
        changed = apply_manifest_to_plan(plan, manifest, "rk", "cfg")
        assert changed is True
        assert plan.subplot_y_ranges == [(-3.0, 4.0)]

    def test_bar_plan_is_skipped(self):
        plan = _make_bar_plan()
        original = list(plan.subplot_y_ranges)
        manifest = self._manifest(only={"s1": [-100.0, 100.0]})
        changed = apply_manifest_to_plan(plan, manifest, "rk", "cfg")
        assert changed is False
        assert plan.subplot_y_ranges == original

    def test_no_change_when_ranges_match_returns_false(self):
        plan = _make_time_plan([("A", (0.0, 10.0))])
        manifest = self._manifest(A={"s1": [0.0, 10.0]})
        changed = apply_manifest_to_plan(plan, manifest, "rk", "cfg")
        assert changed is False
        assert plan.subplot_y_ranges == [(0.0, 10.0)]

    def test_malformed_bounds_are_skipped(self):
        plan = _make_time_plan([
            ("A", (0.0, 1.0)),
            ("B", (-1.0, 1.0)),
        ])
        manifest = self._manifest(
            A={"s1": [0.0, 1.0, 2.0]},   # wrong length
            B={"s1": ["x", "y"]},         # non-numeric
        )
        changed = apply_manifest_to_plan(plan, manifest, "rk", "cfg")
        assert changed is False
        assert plan.subplot_y_ranges == [(0.0, 1.0), (-1.0, 1.0)]

    def test_legacy_flat_schema_entry_is_skipped(self):
        """If the manifest on disk is still in the legacy flat schema
        (value is a 2-element list rather than a per-scenario dict), the
        reader should treat it as missing and not crash."""
        plan = _make_time_plan([("A", (0.0, 1.0))])
        manifest = {"rk": {"cfg": {"A": [-5.0, 5.0]}}}
        changed = apply_manifest_to_plan(plan, manifest, "rk", "cfg")
        assert changed is False
        assert plan.subplot_y_ranges == [(0.0, 1.0)]

    def test_empty_effective_plot_specs_is_noop(self):
        df = pd.DataFrame({"x": [1.0]}, index=pd.Index(["a"], name="t"))
        plan = PlotPlan(
            chart_type="lines",
            plot_name="plan",
            total_file_count=1,
            processed_df=df,
            effective_plot_specs=[],
            file_batches=[],
            layout_type="line",
            layout_params={
                "value_label_width": 0.0, "legend_width": 0.0,
                "base_width": 6.0, "subplot_height": 4.0,
            },
            subplot_y_ranges=[],
        )
        manifest = {"rk": {"cfg": {"A": {"s1": [0.0, 1.0]}}}}
        assert apply_manifest_to_plan(plan, manifest, "rk", "cfg") is False

    def test_pads_short_subplot_y_ranges(self):
        """Defensive: if subplot_y_ranges is shorter than
        effective_plot_specs (legacy plan file), the override should still
        work without IndexError."""
        plan = _make_time_plan([
            ("A", (0.0, 1.0)),
            ("B", (-1.0, 1.0)),
        ])
        plan.subplot_y_ranges = [(0.0, 1.0)]  # deliberately short
        manifest = self._manifest(B={"s1": [-5.0, 5.0]})
        changed = apply_manifest_to_plan(plan, manifest, "rk", "cfg")
        assert changed is True
        assert len(plan.subplot_y_ranges) == 2
        assert plan.subplot_y_ranges[1] == (-5.0, 5.0)
