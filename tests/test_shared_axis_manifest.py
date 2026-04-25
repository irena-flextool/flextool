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
    def test_path_points_into_output_parquet(self, tmp_path: Path):
        acc = ManifestAccumulator(tmp_path)
        assert acc.manifest_path == tmp_path / "output_parquet" / "_axis_bounds.json"

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
    def test_write_creates_axis_bounds_json(self, tmp_path: Path):
        acc = ManifestAccumulator(tmp_path)
        acc.add_plan("rk", "cfg", _make_time_plan([("A", (0.0, 10.0))]), "s1")
        acc.write()

        manifest_path = tmp_path / "output_parquet" / "_axis_bounds.json"
        assert manifest_path.is_file()
        # The manifest lives directly under output_parquet/, not in a
        # ``_shared`` subfolder.
        assert manifest_path.parent == tmp_path / "output_parquet"
        assert not (tmp_path / "output_parquet" / "_shared").exists()

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
            assert "_axis_bounds.json" in str(dst)
            real_replace(src, dst)

        monkeypatch.setattr(_os, "replace", spy_replace)
        acc.write()

        assert len(seen_tmp_paths) == 1
        parquet = tmp_path / "output_parquet"
        leftover = list(parquet.glob("axis_bounds_*.tmp"))
        assert leftover == []

    def test_empty_accumulator_does_not_create_file(self, tmp_path: Path):
        acc = ManifestAccumulator(tmp_path)
        acc.write()
        parquet = tmp_path / "output_parquet"
        assert not (parquet / "_axis_bounds.json").exists()

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

        manifest_path = tmp_path / "output_parquet" / "_axis_bounds.json"
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
        parquet = tmp_path / "output_parquet"
        parquet.mkdir(parents=True)
        (parquet / "_axis_bounds.json").write_text("{{ not json")
        with caplog.at_level("WARNING", logger="flextool.plot_outputs.shared_manifest"):
            acc = ManifestAccumulator(tmp_path)
        assert acc.data == {}

    def test_wrong_shape_entries_are_dropped(self, tmp_path: Path):
        parquet = tmp_path / "output_parquet"
        parquet.mkdir(parents=True)
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
        (parquet / "_axis_bounds.json").write_text(json.dumps(payload))

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
        parquet = tmp_path / "output_parquet"
        parquet.mkdir(parents=True)
        payload = {
            "rk": {
                "cfg": {
                    "A": [0.0, 10.0],            # legacy
                    "B": {"s1": [-1.0, 1.0]},    # new
                },
            },
        }
        (parquet / "_axis_bounds.json").write_text(json.dumps(payload))

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

        manifest_path = tmp_path / "output_parquet" / "_axis_bounds.json"
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

        manifest_path = tmp_path / "output_parquet" / "_axis_bounds.json"
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

        assert not (
            tmp_path / "output_parquet" / "_axis_bounds.json"
        ).exists()


# ---------------------------------------------------------------------------
# Reader API tests
# ---------------------------------------------------------------------------

class TestLoadAxisBoundsManifest:
    def test_missing_file_returns_none(self, tmp_path: Path):
        assert load_axis_bounds_manifest(tmp_path) is None

    def test_valid_file_parses(self, tmp_path: Path):
        parquet = tmp_path / "output_parquet"
        parquet.mkdir(parents=True)
        payload = {"rk": {"cfg": {"A": {"s1": [0.0, 10.0]}}}}
        (parquet / "_axis_bounds.json").write_text(json.dumps(payload))

        data = load_axis_bounds_manifest(tmp_path)
        assert data == payload

    def test_malformed_json_returns_none(self, tmp_path: Path, caplog):
        parquet = tmp_path / "output_parquet"
        parquet.mkdir(parents=True)
        (parquet / "_axis_bounds.json").write_text("not json { ")
        with caplog.at_level(
            "WARNING", logger="flextool.plot_outputs.shared_manifest",
        ):
            data = load_axis_bounds_manifest(tmp_path)
        assert data is None

    def test_non_object_top_level_returns_none(self, tmp_path: Path):
        parquet = tmp_path / "output_parquet"
        parquet.mkdir(parents=True)
        (parquet / "_axis_bounds.json").write_text(json.dumps([1, 2, 3]))
        assert load_axis_bounds_manifest(tmp_path) is None

    def test_accepts_str_path(self, tmp_path: Path):
        parquet = tmp_path / "output_parquet"
        parquet.mkdir(parents=True)
        (parquet / "_axis_bounds.json").write_text("{}")
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

    # ------------------------------------------------------------------
    # Subset-filter regression: re-applying with a narrower active set
    # should shrink the union, not leave the previous wide union in place.
    # This catches the class of bug "uncheck a scenario but y-axis doesn't
    # shrink" — i.e. that the filter doesn't silently degrade to
    # ``active_scenarios=None`` (union over all) when a caller passes a
    # real subset.
    # ------------------------------------------------------------------

    def test_successive_applies_narrow_as_active_set_shrinks(self):
        """Plan is mutated in place by each call.  The second call with a
        narrower *active_scenarios* must recompute the union from scratch
        (not build on the prior wide result) so the y-range contracts."""
        plan = _make_time_plan([("A", (0.0, 10.0))])
        manifest = self._manifest(A={
            "s1": [-5.0, 5.0],
            "s2": [0.0, 10.0],
            "s3": [-100.0, 100.0],
        })

        # All three scenarios active — wide union.
        apply_manifest_to_plan(
            plan, manifest, "rk", "cfg",
            active_scenarios={"s1", "s2", "s3"},
        )
        assert plan.subplot_y_ranges == [(-100.0, 100.0)]

        # Drop s3 — union must shrink to s1+s2 (not stay at s3's wide range).
        apply_manifest_to_plan(
            plan, manifest, "rk", "cfg",
            active_scenarios={"s1", "s2"},
        )
        assert plan.subplot_y_ranges == [(-5.0, 10.0)], (
            "Expected y-range to shrink when s3 is removed from the active "
            "set; got a stale union which indicates the filter wasn't "
            "applied on re-render."
        )

        # Drop s2 too — s1 alone.
        apply_manifest_to_plan(
            plan, manifest, "rk", "cfg",
            active_scenarios={"s1"},
        )
        assert plan.subplot_y_ranges == [(-5.0, 5.0)]

    def test_subset_strictly_narrower_than_full_union(self):
        """Given a manifest with 3 scenarios where one has an obviously
        wider bound than the others, filtering to the two narrower
        scenarios must yield a result strictly inside the full union.

        This is the core user-visible contract: checking only a subset
        must not leak the unchecked scenarios' bounds into the y-axis.
        """
        plan = _make_time_plan([("A", (0.0, 0.0))])
        manifest = self._manifest(A={
            "narrow_a": [-5.0, 5.0],
            "narrow_b": [0.0, 10.0],
            "wide": [-1000.0, 1000.0],
        })

        # Full union includes the wide scenario.
        apply_manifest_to_plan(
            plan, manifest, "rk", "cfg", active_scenarios=None,
        )
        full_union = plan.subplot_y_ranges[0]
        assert full_union == (-1000.0, 1000.0)

        # Reset and apply the narrower subset.
        plan.subplot_y_ranges = [(0.0, 0.0)]
        apply_manifest_to_plan(
            plan, manifest, "rk", "cfg",
            active_scenarios={"narrow_a", "narrow_b"},
        )
        subset_union = plan.subplot_y_ranges[0]

        # The subset union must be strictly inside the full union.
        assert subset_union[0] > full_union[0]
        assert subset_union[1] < full_union[1]
        assert subset_union == (-5.0, 10.0)


# ---------------------------------------------------------------------------
# Viewer wiring regression: make sure the path
#   ``_get_axis_active_scenarios`` → ``_apply_axis_manifest`` →
#   ``apply_manifest_to_plan``
# actually filters by the checked-scenarios set.  Earlier bugs in this
# area have been silent: the filter looked correct in isolation, but the
# viewer wired ``active_scenarios`` up in a way that effectively always
# passed ``None`` (union over all scenarios).  These tests exercise the
# ResultViewer methods directly (no Tk window) to guard against that
# class of regression.
# ---------------------------------------------------------------------------

class TestViewerActiveScenarioWiring:
    """Verify that the viewer passes the checked subset through to
    ``apply_manifest_to_plan``.  We avoid opening a real Tk toplevel by
    binding the relevant bound methods onto a SimpleNamespace that mimics
    ``ResultViewer`` just enough for ``_apply_axis_manifest`` to run.
    """

    def _make_viewer_shim(
        self,
        project_path: Path,
        mode: str,
        checked_subdirs: list[str],
    ):
        """Return a minimal object that exposes the viewer's
        ``_get_axis_active_scenarios`` / ``_apply_axis_manifest`` /
        ``_get_axis_manifest`` / ``_scan_scenarios`` methods.

        The real methods are grabbed off the class and bound to a
        SimpleNamespace that carries just the attributes they read.
        """
        from types import SimpleNamespace
        from flextool.gui.result_viewer import ResultViewer
        from flextool.gui.data_models import ProjectSettings

        settings = ProjectSettings()
        settings.checked_executed_scenarios = [
            f"0|{name}" for name in checked_subdirs
        ]
        # bare_output_owners empty → resolve_source_number falls back to
        # the legacy parse, which gives ``(0, <subdir>)`` for subdirs
        # without a numeric suffix.

        # tkinter.StringVar-alike: just needs a .get() method.
        mode_var = SimpleNamespace(get=lambda: mode)

        shim = SimpleNamespace(
            _project_path=Path(project_path),
            _settings=settings,
            _mode=mode_var,
            _axis_manifest=None,
            _axis_manifest_mtime=0.0,
        )
        # Bind the unbound methods from the class onto our shim so
        # ``self`` resolves to ``shim``.
        shim._scan_scenarios = ResultViewer._scan_scenarios.__get__(shim)
        shim._get_axis_manifest = (
            ResultViewer._get_axis_manifest.__get__(shim)
        )
        shim._get_axis_active_scenarios = (
            ResultViewer._get_axis_active_scenarios.__get__(shim)
        )
        shim._apply_axis_manifest = (
            ResultViewer._apply_axis_manifest.__get__(shim)
        )
        return shim

    def test_single_mode_filters_to_checked_subset(self, tmp_path: Path):
        """The full wiring: with a subset checked in single mode, the
        viewer must narrow the plan's y-range to the union over that
        subset (not the union over all scenarios in the manifest)."""
        # Create scenario output folders and a manifest with three
        # scenarios; one has a wide range that must drop out of the union
        # when unchecked.
        for name in ("s1", "s2", "s3"):
            (tmp_path / "output_parquet" / name).mkdir(parents=True)

        acc = ManifestAccumulator(tmp_path)
        acc.add_plan(
            "rk", "cfg", _make_time_plan([("A", (-5.0, 5.0))]), "s1",
        )
        acc.add_plan(
            "rk", "cfg", _make_time_plan([("A", (0.0, 10.0))]), "s2",
        )
        acc.add_plan(
            "rk", "cfg", _make_time_plan([("A", (-1000.0, 1000.0))]), "s3",
        )
        acc.write()

        # User has checked only s1 and s2.
        shim = self._make_viewer_shim(
            tmp_path, mode="single", checked_subdirs=["s1", "s2"],
        )

        # Sanity: _scan_scenarios + _get_axis_active_scenarios.
        assert sorted(shim._scan_scenarios()) == ["s1", "s2"]
        assert shim._get_axis_active_scenarios() == {"s1", "s2"}

        # Apply manifest through the viewer path.
        plan = _make_time_plan([("A", (0.0, 0.0))])
        shim._apply_axis_manifest(plan, "rk", "cfg")

        # Must be the narrower union of s1+s2 only — NOT the full union
        # which would include s3's wide [-1000, 1000] range.
        assert plan.subplot_y_ranges == [(-5.0, 10.0)], (
            "Viewer wiring didn't filter by the checked-subset; got "
            f"{plan.subplot_y_ranges} instead of (-5.0, 10.0).  This is "
            "the user-reported bug: unchecking a scenario didn't shrink "
            "the y-axis."
        )

    def test_checked_subset_excludes_wider_scenario(self, tmp_path: Path):
        """Regression: checking only scenarios with narrow ranges must
        not pick up a third scenario's wider range from the manifest."""
        for name in ("narrow_a", "narrow_b", "wide"):
            (tmp_path / "output_parquet" / name).mkdir(parents=True)

        acc = ManifestAccumulator(tmp_path)
        acc.add_plan(
            "rk", "cfg",
            _make_time_plan([("A", (-5.0, 5.0))]), "narrow_a",
        )
        acc.add_plan(
            "rk", "cfg",
            _make_time_plan([("A", (0.0, 10.0))]), "narrow_b",
        )
        acc.add_plan(
            "rk", "cfg",
            _make_time_plan([("A", (-1000.0, 1000.0))]), "wide",
        )
        acc.write()

        shim = self._make_viewer_shim(
            tmp_path, mode="single",
            checked_subdirs=["narrow_a", "narrow_b"],
        )

        plan = _make_time_plan([("A", (0.0, 0.0))])
        shim._apply_axis_manifest(plan, "rk", "cfg")

        lo, hi = plan.subplot_y_ranges[0]
        assert lo > -1000.0 and hi < 1000.0, (
            f"Unchecked 'wide' scenario still leaked into y-axis: "
            f"got ({lo}, {hi}), expected strictly inside (-1000, 1000)."
        )
        assert (lo, hi) == (-5.0, 10.0)

    def test_unchecking_shrinks_axis_on_successive_apply(self, tmp_path: Path):
        """Mirror the user's live workflow: initial render with all
        checked, then uncheck one, then call ``_apply_axis_manifest``
        again on the same plan.  The cached plan object gets its
        subplot_y_ranges mutated each time — the second call must
        recompute the union, not preserve the first call's wide one.
        """
        for name in ("s1", "s2", "s3"):
            (tmp_path / "output_parquet" / name).mkdir(parents=True)

        acc = ManifestAccumulator(tmp_path)
        acc.add_plan("rk", "cfg", _make_time_plan([("A", (-5.0, 5.0))]), "s1")
        acc.add_plan("rk", "cfg", _make_time_plan([("A", (0.0, 10.0))]), "s2")
        acc.add_plan(
            "rk", "cfg", _make_time_plan([("A", (-100.0, 100.0))]), "s3",
        )
        acc.write()

        shim = self._make_viewer_shim(
            tmp_path, mode="single",
            checked_subdirs=["s1", "s2", "s3"],
        )
        plan = _make_time_plan([("A", (0.0, 0.0))])

        # First render: all three checked.
        shim._apply_axis_manifest(plan, "rk", "cfg")
        assert plan.subplot_y_ranges == [(-100.0, 100.0)]

        # User unchecks s3 — simulate by updating the settings list.
        shim._settings.checked_executed_scenarios = ["0|s1", "0|s2"]
        # Note: the axis-manifest mtime cache must not prevent the shim
        # from picking up the narrower active set; the active set is
        # derived from settings on every call.

        # Next render: must shrink.
        shim._apply_axis_manifest(plan, "rk", "cfg")
        assert plan.subplot_y_ranges == [(-5.0, 10.0)], (
            "Unchecking s3 between renders didn't shrink the y-axis — "
            "the filter treats the new active set as if nothing changed."
        )

    def test_comparison_mode_returns_none_for_active(self, tmp_path: Path):
        """In comparison mode the viewer returns ``None`` from
        ``_get_axis_active_scenarios`` — we don't want it to silently
        start returning a bare set derived from ``_scan_scenarios`` and
        accidentally filter the comparison-mode plan (whose ranges are
        already the combined df's ranges)."""
        (tmp_path / "output_parquet" / "s1").mkdir(parents=True)
        shim = self._make_viewer_shim(
            tmp_path, mode="comparison", checked_subdirs=["s1"],
        )
        assert shim._get_axis_active_scenarios() is None

    def test_apply_axis_manifest_skips_when_active_is_none(
        self, tmp_path: Path,
    ):
        """When ``_get_axis_active_scenarios`` returns ``None`` (comparison
        mode or an internal error), ``_apply_axis_manifest`` must NOT
        forward ``None`` to :func:`apply_manifest_to_plan` — that would
        union over every scenario in the manifest and defeat the
        checked-subset filter for any accidental cross-mode call.

        The defensive path is an explicit early return; we verify by
        preparing a manifest that *would* widen the plan if the filter
        were bypassed, and asserting the plan is left untouched.
        """
        for name in ("s1", "s2"):
            (tmp_path / "output_parquet" / name).mkdir(parents=True)

        acc = ManifestAccumulator(tmp_path)
        acc.add_plan("rk", "cfg", _make_time_plan([("A", (-999.0, 999.0))]), "s1")
        acc.add_plan("rk", "cfg", _make_time_plan([("A", (0.0, 1.0))]), "s2")
        acc.write()

        # Comparison mode → active is None → should skip.
        shim = self._make_viewer_shim(
            tmp_path, mode="comparison", checked_subdirs=["s1", "s2"],
        )
        plan = _make_time_plan([("A", (0.0, 0.0))])
        shim._apply_axis_manifest(plan, "rk", "cfg")

        # Plan untouched — not unioned over everyone.
        assert plan.subplot_y_ranges == [(0.0, 0.0)], (
            "_apply_axis_manifest unexpectedly forwarded None to "
            "apply_manifest_to_plan, which unioned over every manifest "
            "scenario and widened the plan."
        )
