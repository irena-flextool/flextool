"""Tests for the plot pipeline: prepare_plot_data() and plot_dict_of_dataframes().

Verifies that:
- prepare_plot_data() produces Figures from synthetic DataFrames
- plot_dict_of_dataframes() writes PNG files to disk
- Both bar and time-series chart types work end-to-end
- File splitting produces the expected number of files
- PlotPlan save/load roundtrip produces valid figures
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from flextool.plot_outputs.config import PlotConfig
from flextool.plot_outputs.orchestrator import plot_dict_of_dataframes, prepare_plot_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bar_df(n_rows: int = 5, n_cols: int = 3) -> pd.DataFrame:
    """Create a DataFrame suitable for bar charts: period index, entity columns."""
    rng = np.random.default_rng(42)
    index = pd.Index([f"p{i}" for i in range(n_rows)], name="period")
    columns = pd.MultiIndex.from_arrays(
        [[f"entity_{i}" for i in range(n_cols)]],
        names=["entity"],
    )
    return pd.DataFrame(rng.random((n_rows, n_cols)) * 100, index=index, columns=columns)


def _make_time_df(n_rows: int = 168, n_cols: int = 4) -> pd.DataFrame:
    """Create a DataFrame suitable for time-series line charts."""
    rng = np.random.default_rng(42)
    index = pd.Index(range(n_rows), name="time")
    columns = pd.MultiIndex.from_arrays(
        [[f"node_{i}" for i in range(n_cols)]],
        names=["node"],
    )
    return pd.DataFrame(rng.random((n_rows, n_cols)) * 50, index=index, columns=columns)


def _make_time_multi_df(n_rows: int = 168, n_entities: int = 4, n_scenarios: int = 2) -> pd.DataFrame:
    """Create a DataFrame with two column levels (scenario, entity) for subplot tests."""
    rng = np.random.default_rng(42)
    index = pd.Index(range(n_rows), name="time")
    tuples = [
        (f"scen_{s}", f"entity_{e}")
        for s in range(n_scenarios)
        for e in range(n_entities)
    ]
    columns = pd.MultiIndex.from_tuples(tuples, names=["scenario", "entity"])
    n_cols = len(tuples)
    return pd.DataFrame(rng.random((n_rows, n_cols)) * 50, index=index, columns=columns)


# ---------------------------------------------------------------------------
# Tests for prepare_plot_data
# ---------------------------------------------------------------------------

class TestPreparePlotData:
    """Test that prepare_plot_data produces Figures for various chart types."""

    def test_bar_chart_produces_figure(self):
        """A simple bar config produces at least one Figure."""
        df = _make_bar_df()
        cfg = PlotConfig(
            plot_name="Test bar",
            map_dimensions_for_plots=["d_e", "s_b"],
        )
        figures, total = prepare_plot_data(df, cfg, plot_name="Test bar")
        assert total >= 1
        assert len(figures) >= 1
        _name, fig = figures[0]
        assert fig is not None

    def test_line_chart_produces_figure(self):
        """A simple line config produces at least one Figure."""
        df = _make_time_df()
        cfg = PlotConfig(
            plot_name="Test lines",
            map_dimensions_for_plots=["t_e", "t_l"],
        )
        figures, total = prepare_plot_data(df, cfg, plot_name="Test lines")
        assert total >= 1
        assert len(figures) >= 1
        _name, fig = figures[0]
        assert fig is not None

    def test_stacked_area_produces_figure(self):
        """A stacked area config produces at least one Figure."""
        df = _make_time_df()
        cfg = PlotConfig(
            plot_name="Test stack",
            map_dimensions_for_plots=["t_e", "t_s"],
        )
        figures, total = prepare_plot_data(df, cfg, plot_name="Test stack")
        assert total >= 1
        assert len(figures) >= 1

    def test_only_file_index_returns_single_figure(self):
        """With only_file_index, at most one Figure is returned."""
        df = _make_time_df(n_cols=20)
        cfg = PlotConfig(
            plot_name="Test split",
            map_dimensions_for_plots=["t_e", "t_l"],
            max_items_per_plot=5,
        )
        # Get total count
        _, total = prepare_plot_data(df, cfg, plot_name="Test split")
        assert total > 1, "Expected multiple file splits"

        # Request specific page
        figures, total2 = prepare_plot_data(
            df, cfg, plot_name="Test split", only_file_index=0
        )
        assert total2 == total
        assert len(figures) == 1

    def test_empty_dataframe_returns_empty(self):
        """An empty DataFrame produces no Figures."""
        df = pd.DataFrame()
        cfg = PlotConfig(
            plot_name="Empty",
            map_dimensions_for_plots=["d_e", "s_b"],
        )
        figures, total = prepare_plot_data(df, cfg, plot_name="Empty")
        assert figures == []
        assert total == 0

    def test_subplot_config(self):
        """A config with 'u' (subplot) dimension works."""
        df = _make_time_multi_df()
        cfg = PlotConfig(
            plot_name="Test subplots",
            map_dimensions_for_plots=["t_se", "t_ul"],
        )
        figures, total = prepare_plot_data(df, cfg, plot_name="Test subplots")
        assert total >= 1
        assert len(figures) >= 1


# ---------------------------------------------------------------------------
# Tests for plot_dict_of_dataframes (end-to-end PNG writing)
# ---------------------------------------------------------------------------

class TestPlotDictOfDataframes:
    """Test that plot_dict_of_dataframes writes PNG files to disk."""

    def test_bar_chart_writes_png(self, tmp_path):
        """Bar chart config writes at least one PNG."""
        plot_dir = tmp_path / "plots"
        plot_dir.mkdir()

        results = {"test_d_e": _make_bar_df()}
        settings = {
            "test_d_e": {
                "plot_name": "1.0 Test bar",
                "map_dimensions_for_plots": ["d_e", "s_b"],
            },
        }
        plot_dict_of_dataframes(
            results, str(plot_dir), settings,
            active_settings=["default"],
            plot_rows=(0, 167),
        )
        pngs = list(plot_dir.glob("*.png"))
        assert len(pngs) >= 1, f"Expected PNG files, got: {list(plot_dir.iterdir())}"

    def test_line_chart_writes_png(self, tmp_path):
        """Line chart config writes at least one PNG."""
        plot_dir = tmp_path / "plots"
        plot_dir.mkdir()

        results = {"test_dt_e": _make_time_df()}
        settings = {
            "test_dt_e": {
                "plot_name": "1.0 Test lines",
                "map_dimensions_for_plots": ["t_e", "t_l"],
            },
        }
        plot_dict_of_dataframes(
            results, str(plot_dir), settings,
            active_settings=["default"],
            plot_rows=(0, 167),
        )
        pngs = list(plot_dir.glob("*.png"))
        assert len(pngs) >= 1

    def test_stacked_area_writes_png(self, tmp_path):
        """Stacked area config writes at least one PNG."""
        plot_dir = tmp_path / "plots"
        plot_dir.mkdir()

        results = {"test_dt_e": _make_time_df()}
        settings = {
            "test_dt_e": {
                "plot_name": "1.0 Test stack",
                "map_dimensions_for_plots": ["t_e", "t_s"],
            },
        }
        plot_dict_of_dataframes(
            results, str(plot_dir), settings,
            active_settings=["default"],
            plot_rows=(0, 167),
        )
        pngs = list(plot_dir.glob("*.png"))
        assert len(pngs) >= 1

    def test_multiple_results_write_multiple_pngs(self, tmp_path):
        """Multiple result keys each produce their own PNG(s)."""
        plot_dir = tmp_path / "plots"
        plot_dir.mkdir()

        results = {
            "bars_d_e": _make_bar_df(),
            "lines_dt_e": _make_time_df(),
        }
        settings = {
            "bars_d_e": {
                "plot_name": "1.0 Bars",
                "map_dimensions_for_plots": ["d_e", "s_b"],
            },
            "lines_dt_e": {
                "plot_name": "2.0 Lines",
                "map_dimensions_for_plots": ["t_e", "t_l"],
            },
        }
        plot_dict_of_dataframes(
            results, str(plot_dir), settings,
            active_settings=["default"],
            plot_rows=(0, 167),
        )
        pngs = list(plot_dir.glob("*.png"))
        assert len(pngs) >= 2

    def test_named_config(self, tmp_path):
        """Named sub-config (non-default) works."""
        plot_dir = tmp_path / "plots"
        plot_dir.mkdir()

        results = {"test_d_e": _make_bar_df()}
        settings = {
            "test_d_e": {
                "default": {
                    "plot_name": "1.0 Default bars",
                    "map_dimensions_for_plots": ["d_e", "s_b"],
                },
                "alt": {
                    "plot_name": "1.0 Alt bars",
                    "map_dimensions_for_plots": ["d_e", "s_b"],
                    "bar_orientation": "vertical",
                },
            },
        }
        plot_dict_of_dataframes(
            results, str(plot_dir), settings,
            active_settings=["default", "alt"],
            plot_rows=(0, 167),
        )
        pngs = list(plot_dir.glob("*.png"))
        assert len(pngs) >= 2

    def test_file_splitting(self, tmp_path):
        """Many columns with small max_items_per_plot produces multiple files."""
        plot_dir = tmp_path / "plots"
        plot_dir.mkdir()

        results = {"test_dt_e": _make_time_df(n_cols=15)}
        settings = {
            "test_dt_e": {
                "plot_name": "1.0 Split test",
                "map_dimensions_for_plots": ["t_e", "t_l"],
                "max_items_per_plot": 5,
            },
        }
        plot_dict_of_dataframes(
            results, str(plot_dir), settings,
            active_settings=["default"],
            plot_rows=(0, 167),
        )
        pngs = list(plot_dir.glob("*.png"))
        assert len(pngs) >= 3, f"Expected 3+ files for 15 cols / 5 items, got {len(pngs)}"


# ---------------------------------------------------------------------------
# Tests for PlotPlan save/load roundtrip
# ---------------------------------------------------------------------------

class TestPlotPlan:
    """Test PlotPlan save/load and figure building."""

    def test_save_load_roundtrip_bar(self, tmp_path):
        """Bar chart plan saves and loads correctly."""
        from flextool.plot_outputs.plan import (
            compute_plot_plans_for_result, load_plot_plan, build_figure_from_plan,
        )

        df = _make_bar_df()
        settings = {
            "test_d_e": {
                "plot_name": "1.0 Plan test",
                "map_dimensions_for_plots": ["d_e", "s_b"],
            },
        }
        plan_dir = tmp_path / "plans"
        compute_plot_plans_for_result(
            df, "test_d_e", settings, plan_dir,
            plot_rows=(0, 167),
        )
        assert plan_dir.exists()

        plan = load_plot_plan(plan_dir, "test_d_e", "default")
        assert plan is not None
        assert plan.total_file_count >= 1

        fig = build_figure_from_plan(plan, 0)
        assert fig is not None

    def test_save_load_roundtrip_lines(self, tmp_path):
        """Line chart plan saves and loads correctly."""
        from flextool.plot_outputs.plan import (
            compute_plot_plans_for_result, load_plot_plan, build_figure_from_plan,
        )

        df = _make_time_df()
        settings = {
            "test_dt_e": {
                "plot_name": "1.0 Line plan",
                "map_dimensions_for_plots": ["t_e", "t_l"],
            },
        }
        plan_dir = tmp_path / "plans"
        compute_plot_plans_for_result(
            df, "test_dt_e", settings, plan_dir,
            plot_rows=(0, 167),
        )

        plan = load_plot_plan(plan_dir, "test_dt_e", "default")
        assert plan is not None

        fig = build_figure_from_plan(plan, 0)
        assert fig is not None

    def test_plan_missing_returns_none(self, tmp_path):
        """Loading a non-existent plan returns None."""
        from flextool.plot_outputs.plan import load_plot_plan

        plan = load_plot_plan(tmp_path, "nonexistent", "default")
        assert plan is None
