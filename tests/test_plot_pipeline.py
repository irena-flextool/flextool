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

from flextool.plot_outputs.config import PlotConfig
from flextool.plot_outputs.orchestrator import (
    _apply_dimension_rules,
    plot_dict_of_dataframes,
    prepare_plot_data,
)


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

class TestPeriodWeightingRules:
    """The 'y' (period-weighted sum) and 'z' (period-weighted average) rules.

    Missing ``years_represented`` means every period is one year (unit
    weights) — a valid choice, not a degradation: 'y' must collapse to a
    plain sum and 'z' to a plain mean, with NO warning.  Explicit weights
    must weight both.
    """

    @staticmethod
    def _period_df() -> pd.DataFrame:
        # Two periods, one entity column. period is the collapsed row level.
        df = pd.DataFrame(
            {"eA": [10.0, 20.0]},
            index=pd.Index(["p1", "p2"], name="period"),
        )
        df.columns.name = "entity"
        return df

    def _collapse(self, rules: str, period_weights=None) -> float:
        cfg = PlotConfig(
            plot_name="weight test",
            map_dimensions_for_plots=["d_e", rules],
        )
        out = _apply_dimension_rules(
            self._period_df(), cfg, (0, 100), period_weights=period_weights,
        )
        assert out is not None
        return float(out[0].values.sum())

    def test_y_without_weights_is_plain_sum(self):
        # weighted sum with unit weights == plain sum
        assert self._collapse("y_b") == 10.0 + 20.0

    def test_z_without_weights_is_plain_mean(self):
        # The bug being fixed: 'z' used to fall through to a SUM (30) when
        # no weights were present; with unit weights it must be the mean.
        assert self._collapse("z_b") == (10.0 + 20.0) / 2

    def test_y_with_weights_is_weighted_sum(self):
        w = pd.Series({"p1": 2.0, "p2": 3.0})
        assert self._collapse("y_b", w) == 10.0 * 2 + 20.0 * 3

    def test_z_with_weights_is_weighted_average(self):
        w = pd.Series({"p1": 2.0, "p2": 3.0})
        assert self._collapse("z_b", w) == (10.0 * 2 + 20.0 * 3) / (2 + 3)

    def test_no_warning_when_weights_absent(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="flextool.plot_outputs.orchestrator"):
            self._collapse("z_b")
        assert not any(
            "years_represented" in r.getMessage() for r in caplog.records
        )


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
# Tests for period ordering on bar axis (horizontal vs vertical)
# ---------------------------------------------------------------------------

class TestPeriodOrdering:
    """Verify that p-variant bar plots render periods in reading order.

    Horizontal bars: first period at the top (highest y-position).
    Vertical bars: first period at the left (lowest x-position).
    """

    @staticmethod
    def _period_df(periods: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {"value": list(range(1, len(periods) + 1))},
            index=pd.Index(periods, name="period"),
        )

    def test_horizontal_periods_top_to_bottom(self):
        from flextool.plot_outputs.plot_bars import build_bar_figures

        periods = ["y2019", "y2020", "y2030", "y2040", "y2050"]
        df = self._period_df(periods)
        figs, _ = build_bar_figures(
            df, "test", "",
            stack_levels=[], expand_axis_levels=[],
            bar_orientation="horizontal", base_bar_length=4,
        )
        ax = figs[0][1].axes[0]
        pairs = sorted(
            zip([t.get_text() for t in ax.get_yticklabels()], ax.get_yticks()),
            key=lambda x: -x[1],
        )
        ordered_labels = [lbl for lbl, _ in pairs]
        assert ordered_labels == periods

    def test_vertical_periods_left_to_right(self):
        from flextool.plot_outputs.plot_bars import build_bar_figures

        periods = ["y2019", "y2020", "y2030", "y2040", "y2050"]
        df = self._period_df(periods)
        figs, _ = build_bar_figures(
            df, "test", "",
            stack_levels=[], expand_axis_levels=[],
            bar_orientation="vertical", base_bar_length=4,
        )
        ax = figs[0][1].axes[0]
        pairs = sorted(
            zip([t.get_text() for t in ax.get_xticklabels()], ax.get_xticks()),
            key=lambda x: x[1],
        )
        ordered_labels = [lbl for lbl, _ in pairs]
        assert ordered_labels == periods

    def test_vertical_expand_axis_groups_left_to_right(self):
        """Groups should be placed left-to-right; periods within each group
        should also be left-to-right."""
        from flextool.plot_outputs.plot_bars import build_bar_figures

        periods = ["y2019", "y2020", "y2030"]
        cols = pd.MultiIndex.from_product(
            [["scA", "scB"], ["e1", "e2"]], names=["scen", "item"],
        )
        df = pd.DataFrame(
            np.arange(12).reshape(3, 4) + 1.0,
            index=pd.Index(periods, name="period"),
            columns=cols,
        )
        figs, _ = build_bar_figures(
            df, "test", "",
            stack_levels=[1], expand_axis_levels=[0],
            bar_orientation="vertical", base_bar_length=4,
        )
        ax = figs[0][1].axes[0]
        # Tick labels in ascending x order
        pairs = sorted(
            zip([t.get_text() for t in ax.get_xticklabels()], ax.get_xticks()),
            key=lambda x: x[1],
        )
        ordered_labels = [lbl for lbl, _ in pairs]
        # The expand-group level (scen) is folded into each tick label as
        # "<group> | <bar>" (group first). Bar (period) portion reads
        # left-to-right within each group; the first group (scA) is left-most.
        group_parts = [lbl.split(" | ")[0] for lbl in ordered_labels]
        bar_parts = [lbl.split(" | ")[1] for lbl in ordered_labels]
        assert bar_parts == periods + periods
        assert group_parts[:3] == ["scA"] * 3
        assert group_parts[3:] == ["scB"] * 3

    def test_horizontal_expand_axis_groups_top_to_bottom(self):
        from flextool.plot_outputs.plot_bars import build_bar_figures

        periods = ["y2019", "y2020", "y2030"]
        cols = pd.MultiIndex.from_product(
            [["scA", "scB"], ["e1", "e2"]], names=["scen", "item"],
        )
        df = pd.DataFrame(
            np.arange(12).reshape(3, 4) + 1.0,
            index=pd.Index(periods, name="period"),
            columns=cols,
        )
        figs, _ = build_bar_figures(
            df, "test", "",
            stack_levels=[1], expand_axis_levels=[0],
            bar_orientation="horizontal", base_bar_length=4,
        )
        ax = figs[0][1].axes[0]
        pairs = sorted(
            zip([t.get_text() for t in ax.get_yticklabels()], ax.get_yticks()),
            key=lambda x: -x[1],
        )
        ordered_labels = [lbl for lbl, _ in pairs]
        # The expand-group level (scen) is now folded into each tick label as
        # "<group> | <bar>" (group first, outer→inner) instead of a separate
        # label to the left of the axis. The bar (period) portion still reads
        # top-to-bottom within each group, and the first group (scA) is on top.
        group_parts = [lbl.split(" | ")[0] for lbl in ordered_labels]
        bar_parts = [lbl.split(" | ")[1] for lbl in ordered_labels]
        assert bar_parts == periods + periods
        assert group_parts[:3] == ["scA"] * 3
        assert group_parts[3:] == ["scB"] * 3


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

    def test_color_hints_survive_save_load_roundtrip(self, tmp_path):
        """Stage 3.4: color_category / color_entity_class round-trip on disk."""
        from flextool.plot_outputs.plan import (
            PlotPlan, save_plot_plan, load_plot_plan,
        )

        plan = PlotPlan(
            chart_type='stack',
            plot_name='p',
            total_file_count=1,
            processed_df=pd.DataFrame(
                {'coal': [1.0, 2.0]},
                index=pd.Index([0, 1], name='time'),
            ),
            effective_plot_specs=[(None, ['coal'])],
            file_batches=[[0]],
            shared_color_map={'coal': (1.0, 0.0, 0.0)},
            color_category='node_flows',
            color_entity_class='unit',
        )
        plan_dir = tmp_path / "plans"
        save_plot_plan(plan, plan_dir, "rk", "default")

        loaded = load_plot_plan(plan_dir, "rk", "default")
        assert loaded is not None
        assert loaded.color_category == 'node_flows'
        assert loaded.color_entity_class == 'unit'

    def test_old_plan_without_color_hints_loads_with_none(self, tmp_path):
        """Backward compat: a plan JSON predating the color hints loads with
        ``color_category`` / ``color_entity_class`` defaulting to None (no
        crash)."""
        import json
        from flextool.plot_outputs.plan import (
            PlotPlan, save_plot_plan, load_plot_plan,
        )

        plan = PlotPlan(
            chart_type='lines',
            plot_name='p',
            total_file_count=1,
            processed_df=pd.DataFrame(
                {'a': [1.0, 2.0]},
                index=pd.Index([0, 1], name='time'),
            ),
            effective_plot_specs=[(None, ['a'])],
            file_batches=[[0]],
        )
        plan_dir = tmp_path / "plans"
        save_plot_plan(plan, plan_dir, "rk", "default")

        # Strip the new keys to emulate an older on-disk plan file.
        json_path = plan_dir / "rk__default_plan.json"
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        meta.pop("color_category", None)
        meta.pop("color_entity_class", None)
        json_path.write_text(json.dumps(meta), encoding="utf-8")

        loaded = load_plot_plan(plan_dir, "rk", "default")
        assert loaded is not None
        assert loaded.color_category is None
        assert loaded.color_entity_class is None

    def test_max_items_per_subplot_column(self, tmp_path):
        """Bar plan respects max_items_per_subplot_column."""
        from flextool.plot_outputs.plan import (
            compute_plot_plans_for_result, load_plot_plan,
        )

        # 3 subplots (one per scenario) each with 15 bar rows.
        # subplots_per_row=1 → single column, 3 grid rows.
        # Column cumulative after row 1: 15, row 2: 30, row 3: 45.
        # With limit=40, rows 1-2 fit (30 ≤ 40), row 3 spills to file 2.
        rng = np.random.default_rng(99)
        n_rows = 15
        tuples = [
            (f"scen_{s}", f"entity_{e}")
            for s in range(3)
            for e in range(4)
        ]
        columns = pd.MultiIndex.from_tuples(tuples, names=["scenario", "entity"])
        index = pd.Index([f"p{i}" for i in range(n_rows)], name="period")
        df = pd.DataFrame(rng.random((n_rows, len(tuples))) * 100,
                          index=index, columns=columns)

        # d_se = 1 row level (period) + 2 col levels (scenario, entity)
        # b_us = bar axis + subplot + stack  (3 rules for 3 levels)
        settings = {
            "test_col_limit": {
                "plot_name": "Column limit test",
                "map_dimensions_for_plots": ["d_se", "b_us"],
                "max_items_per_subplot_column": 40,
                "subplots_per_row": 1,
                "max_subplots_per_file": 20,
                "max_items_per_plot": 40,
            },
        }
        plan_dir = tmp_path / "plans"
        compute_plot_plans_for_result(
            df, "test_col_limit", settings, plan_dir,
            plot_rows=(0, 167),
        )

        plan = load_plot_plan(plan_dir, "test_col_limit", "default")
        assert plan is not None
        # Should be split into 2 files: first with 2 subplots (30 ≤ 40),
        # second with 1 subplot (the third that would push to 45 > 40).
        assert plan.total_file_count == 2
        assert len(plan.file_batches[0]) == 2
        assert len(plan.file_batches[1]) == 1


# ---------------------------------------------------------------------------
# Color-template kwargs forwarding through the batch render path
# ---------------------------------------------------------------------------


class TestColorTemplateForwardedToBuildFigures:
    """The batch render path in orchestrator.prepare_plot_data must forward
    ``color_template``/``category``/``entity_class`` into the underlying
    build_*_figures functions so batch-rendered PNGs pick up template colors
    just like the plan-based path.
    """

    def _spy(self, monkeypatch):
        """Install a spy on build_shared_color_map in plot_lines + plot_bars.

        Returns a list that will receive the kwargs each call used.
        """
        from flextool.plot_outputs import plot_lines, plot_bars
        from flextool.plot_outputs.legend_helpers import build_shared_color_map as real

        calls: list[dict] = []

        def _spy_fn(labels, *, color_template=None, category=None,
                    entity_class=None):
            calls.append({
                "labels": list(labels),
                "color_template": color_template,
                "category": category,
                "entity_class": entity_class,
            })
            return real(
                labels,
                color_template=color_template,
                category=category,
                entity_class=entity_class,
            )

        monkeypatch.setattr(plot_lines, "build_shared_color_map", _spy_fn)
        monkeypatch.setattr(plot_bars, "build_shared_color_map", _spy_fn)
        return calls

    def test_line_chart_forwards_kwargs(self, monkeypatch):
        """build_line_figures receives color_template + category kwargs."""
        calls = self._spy(monkeypatch)
        df = _make_time_df(n_cols=4)
        cfg = PlotConfig(
            plot_name="Test line colors",
            map_dimensions_for_plots=["t_e", "t_l"],
            legend="shared",
            color_category="costs",
        )
        figures, total = prepare_plot_data(df, cfg, plot_name="Test line colors")
        assert total >= 1
        assert len(calls) >= 1, "build_shared_color_map should have been called"
        last = calls[-1]
        assert last["category"] == "costs"
        assert last["entity_class"] is None
        # color_template is the loaded dict — may be {} if template file
        # isn't present, but it MUST have been passed through (not None
        # where the caller did not pass it).  With the default repo layout
        # schemas/default_plot_settings.yaml *may* exist; either way the value
        # should be a dict (not None) because orchestrator forwards the
        # load_color_template() result.
        assert isinstance(last["color_template"], dict)

    def test_stack_chart_forwards_kwargs(self, monkeypatch):
        """build_stack_figures receives entity_class kwarg."""
        calls = self._spy(monkeypatch)
        df = _make_time_df(n_cols=4)
        cfg = PlotConfig(
            plot_name="Test stack colors",
            map_dimensions_for_plots=["t_e", "t_s"],
            legend="shared",
            color_entity_class="group",
        )
        figures, total = prepare_plot_data(df, cfg, plot_name="Test stack colors")
        assert total >= 1
        assert len(calls) >= 1
        last = calls[-1]
        assert last["entity_class"] == "group"
        assert last["category"] is None
        assert isinstance(last["color_template"], dict)

    def test_bar_chart_forwards_kwargs(self, monkeypatch):
        """build_bar_figures receives color_template + category kwargs.

        Uses a stacked bar config so the shared color map codepath
        triggers (bar build uses shared colors only when stack_levels or
        grouped_bar_levels are present).
        """
        calls = self._spy(monkeypatch)
        # 2-level column index so we have a stack dimension
        rng = np.random.default_rng(42)
        n_rows = 5
        tuples = [(f"scen_{s}", f"entity_{e}") for s in range(2) for e in range(3)]
        columns = pd.MultiIndex.from_tuples(tuples, names=["scenario", "entity"])
        index = pd.Index([f"p{i}" for i in range(n_rows)], name="period")
        df = pd.DataFrame(rng.random((n_rows, len(tuples))) * 100,
                          index=index, columns=columns)
        cfg = PlotConfig(
            plot_name="Test bar colors",
            map_dimensions_for_plots=["d_se", "s_bs"],  # bar + stack
            legend="shared",
            color_category="costs",
        )
        figures, total = prepare_plot_data(df, cfg, plot_name="Test bar colors")
        assert total >= 1
        assert len(calls) >= 1
        last = calls[-1]
        assert last["category"] == "costs"
        assert isinstance(last["color_template"], dict)

    def test_no_color_config_still_forwards_none_category(self, monkeypatch):
        """With neither color_category nor color_entity_class set, the
        build_*_figures call still receives None for both (default behaviour
        preserved — palette-only coloring)."""
        calls = self._spy(monkeypatch)
        df = _make_time_df(n_cols=4)
        cfg = PlotConfig(
            plot_name="Test default colors",
            map_dimensions_for_plots=["t_e", "t_l"],
            legend="shared",
        )
        figures, total = prepare_plot_data(df, cfg, plot_name="Test default colors")
        assert total >= 1
        assert len(calls) >= 1
        last = calls[-1]
        assert last["category"] is None
        assert last["entity_class"] is None

    def test_template_colors_applied_to_figure(self, monkeypatch, tmp_path):
        """End-to-end: with a custom template file pointing at a known
        color, the resulting figure's legend patches should include that
        exact color."""
        # Build a tiny color template yaml pointing ``Solar`` at a known
        # distinctive color.
        import yaml
        from flextool.plot_outputs import color_template as ct
        tmpl_path = tmp_path / "colors.yaml"
        tmpl_path.write_text(
            yaml.safe_dump(
                {"category": {"costs": {"node_0": "#123456"}}}
            ),
            encoding="utf-8",
        )
        ct._clear_cache()
        # Patch load_color_template (used by orchestrator) to read our temp
        # template instead of the default.
        from flextool.plot_outputs import orchestrator as orch
        monkeypatch.setattr(
            orch, "load_color_template",
            lambda path=None: ct.load_color_template(tmpl_path),
        )

        df = _make_time_df(n_cols=4)
        cfg = PlotConfig(
            plot_name="Test template applied",
            map_dimensions_for_plots=["t_e", "t_l"],
            legend="shared",
            color_category="costs",
        )
        figures, total = prepare_plot_data(
            df, cfg, plot_name="Test template applied"
        )
        assert total >= 1
        # Collect all line colors across subplots of the first figure.
        _, fig = figures[0]
        found = False
        target_rgb = (0x12 / 255.0, 0x34 / 255.0, 0x56 / 255.0)
        for ax in fig.get_axes():
            for line in ax.get_lines():
                c = line.get_color()
                # Matplotlib may return a string or tuple.
                import matplotlib.colors as mcolors
                rgb = mcolors.to_rgb(c)
                if all(abs(a - b) < 1e-6 for a, b in zip(rgb, target_rgb)):
                    found = True
                    break
            if found:
                break
        assert found, (
            "Expected at least one line colored with the template color #123456 "
            "for label 'node_0' under category 'costs'."
        )


class TestColorHintBuildsMapWithoutSharedLegend:
    """Regression: preset category/entity colors must apply regardless of
    legend layout.

    The defect: ``shared_color_map`` was built ONLY under
    ``legend == 'shared'``.  The default costs config uses ``legend: right``
    and entity-tagged plots commonly use ``legend: all`` — so an explicit
    ``color_category`` / ``color_entity_class`` hint never reached
    ``build_shared_color_map`` and the plot fell back to the tab10 palette.

    Fix: build the color map whenever the legend is shared OR a color hint
    is present; leave non-hinted non-shared plots untouched (palette).
    """

    def _write_template(self, tmp_path):
        import yaml
        tmpl = tmp_path / "plot_settings.yaml"
        tmpl.write_text(
            yaml.safe_dump(
                {
                    "scenarios": {},
                    "categories": {"costs": {"node_0": "#123456"}},
                    "entities": {"unit": {"node_0": "#abcdef"}},
                }
            ),
            encoding="utf-8",
        )
        from flextool.plot_outputs import color_template as ct
        ct._clear_cache()
        return tmpl

    def test_color_category_applies_with_right_legend(self, tmp_path):
        """A ``color_category`` plot with the default (non-shared) ``right``
        legend gets its preset category color, not the palette."""
        from flextool.plot_outputs.plan import compute_live_plan
        tmpl = self._write_template(tmp_path)
        df = _make_time_df(n_cols=4)
        cfg = PlotConfig(
            plot_name="costs",
            map_dimensions_for_plots=["t_e", "t_l"],
            color_category="costs",
        )
        # Sanity: this is the non-shared layout the bug missed.
        assert cfg.legend == "right"
        plan = compute_live_plan(df, cfg, "costs", color_path=tmpl)
        assert plan is not None
        assert plan.shared_color_map is not None, (
            "color_category plot with legend=right must build a color map"
        )
        assert plan.shared_color_map.get("node_0") == (
            0x12 / 255.0, 0x34 / 255.0, 0x56 / 255.0,
        )

    def test_color_entity_class_applies_with_all_legend(self, tmp_path):
        """A ``color_entity_class`` plot with ``legend: all`` (non-shared)
        gets its preset entity color, not the palette."""
        from flextool.plot_outputs.plan import compute_live_plan
        tmpl = self._write_template(tmp_path)
        df = _make_time_df(n_cols=4)
        cfg = PlotConfig(
            plot_name="unit flows",
            map_dimensions_for_plots=["t_e", "t_l"],
            legend="all",
            color_entity_class="unit",
        )
        plan = compute_live_plan(df, cfg, "unit flows", color_path=tmpl)
        assert plan is not None
        assert plan.shared_color_map is not None, (
            "color_entity_class plot with legend=all must build a color map"
        )
        assert plan.shared_color_map.get("node_0") == (
            0xab / 255.0, 0xcd / 255.0, 0xef / 255.0,
        )

    def test_color_category_applies_to_stacked_bars_with_right_legend(self, tmp_path):
        """The bar-plan build gate honors the hint under a non-shared legend
        too (stacked bars)."""
        from flextool.plot_outputs.plan import compute_live_plan
        tmpl = self._write_template(tmp_path)
        df = pd.DataFrame(
            {"node_0": [10.0, 20.0], "node_1": [11.0, 22.0]},
            index=pd.Index(["catA", "catB"], name="category"),
        )
        df.columns.name = "entity"
        cfg = PlotConfig(
            plot_name="bars",
            map_dimensions_for_plots=["d_e", "s_b"],
            color_category="costs",
        )
        assert cfg.legend == "right"
        plan = compute_live_plan(df, cfg, "bars", color_path=tmpl)
        assert plan is not None
        assert plan.shared_color_map is not None, (
            "color_category stacked-bar plot with legend=right must build a map"
        )

    def test_non_hinted_non_shared_plot_builds_no_map(self, tmp_path):
        """Guard the byte-identical path: a plot with NO color hint and a
        non-shared legend must NOT build a shared color map (palette only)."""
        from flextool.plot_outputs.plan import compute_live_plan
        tmpl = self._write_template(tmp_path)
        df = _make_time_df(n_cols=4)
        cfg = PlotConfig(
            plot_name="plain",
            map_dimensions_for_plots=["t_e", "t_l"],
            # no color_category / color_entity_class; legend defaults to right
        )
        plan = compute_live_plan(df, cfg, "plain", color_path=tmpl)
        assert plan is not None
        assert plan.shared_color_map is None, (
            "non-hinted non-shared plot must stay on the per-subplot palette"
        )


class TestSingleColLevelStackUnstack:
    """Regression for the single-col-level stack-then-unstack quirk.

    When ``_apply_dimension_rules`` stacks the only column level into the
    row index (because a ``b/t/i`` rule lands on it), pandas collapses the
    result to a Series; the code wraps it back via ``to_frame()`` which
    introduces a default-named placeholder column.  If a subsequent
    ``unstack`` adds a real col level, the placeholder lingers — leaving
    the processed_df with one more column level than the rules string
    accounts for, which silently strips downstream ``s``/``g``/``u`` etc.
    rules from rendering.

    The fix tags the placeholder with a sentinel level name and drops it
    once a real col level is alongside.
    """

    def test_stack_then_unstack_preserves_single_target_col_level(self):
        """A 1-row × 1-col config like ``[d_s, s_b]`` must produce a
        clean 1-row × 1-col processed shape, not a phantom extra level.

        Mirrors costs_discounted_p_ (category × scenario): bars per
        scenario, categories rendered as stacks within each bar.
        """
        from flextool.plot_outputs.orchestrator import _apply_dimension_rules

        df = pd.DataFrame(
            {"scenA": [10.0, 20.0, 30.0], "scenB": [11.0, 22.0, 33.0]},
            index=pd.Index(["catA", "catB", "catC"], name="category"),
        )
        df.columns.name = "scenario"

        cfg = PlotConfig(
            plot_name="t",
            map_dimensions_for_plots=["d_s", "s_b"],
        )
        result = _apply_dimension_rules(df, cfg, (0, len(df)))
        assert result is not None
        processed_df, rules, chart_type, _, _ = result

        # row=scenario (1 level), col=category (1 level) — no placeholder.
        assert processed_df.index.nlevels == 1
        assert processed_df.columns.nlevels == 1
        assert processed_df.index.names == ["scenario"]
        assert processed_df.columns.names == ["category"]
        # Rules string must match the new level count (1+1=2 chars).
        assert len(rules) == 2

    def test_compute_live_plan_keeps_stack_level(self):
        """End-to-end: the surviving col level retains its stack rule.

        With the fix, ``_compute_bar_plan`` sees ``stack_levels=[0]`` and
        renders categories as visible stacks; without the fix, the
        phantom level shifted positions and stack_levels was empty.
        """
        from flextool.plot_outputs.plan import compute_live_plan

        df = pd.DataFrame(
            {"scenA": [10.0, 20.0, 30.0], "scenB": [11.0, 22.0, 33.0]},
            index=pd.Index(["catA", "catB", "catC"], name="category"),
        )
        df.columns.name = "scenario"

        cfg = PlotConfig(
            plot_name="costs",
            map_dimensions_for_plots=["d_s", "s_b"],
        )
        plan = compute_live_plan(df, cfg, "costs")
        assert plan is not None
        assert plan.chart_type == "bar"
        # The 's' rule on category (now in col after the unstack) must
        # be picked up as a stack level.
        assert plan.stack_levels == [0]

    def test_no_unstack_keeps_placeholder_when_safe(self):
        """If only ``stack`` runs (no real col added by unstack), the
        sentinel-tagged level stays — dropping it would leave 0 col
        levels, which the figure builder doesn't expect.

        This case isn't a real shipped config, but the safeguard
        protects against future ones.
        """
        from flextool.plot_outputs.orchestrator import _apply_dimension_rules

        df = pd.DataFrame(
            {"scenA": [10.0, 20.0]},
            index=pd.Index(["catA", "catB"], name="category"),
        )
        df.columns.name = "scenario"

        cfg = PlotConfig(
            plot_name="t",
            map_dimensions_for_plots=["d_s", "b_b"],   # 'b' on row d → stays; 'b' on col → stack to row
        )
        result = _apply_dimension_rules(df, cfg, (0, len(df)))
        # The function may return None or a valid frame — what matters
        # is that we don't crash with a level mismatch.
        if result is not None:
            processed_df, *_ = result
            assert processed_df.columns.nlevels >= 1


class TestResolveSharedAxisBounds:
    """Regression: comparison-mode shared bounds must sum only over stack_levels.

    Previously the resolver did sum(axis=1) over all non-subplot column levels,
    which for plots with expand_axis (e.g. node_d_ep) inflated the bound by
    the cardinality of those levels. The visible stacked bar is the sum over
    stack_levels only; other levels are distinct bars/series.
    """

    def _build_df(self, n_nodes=3, n_cats=3, n_periods=2, n_scen=4, seed=0):
        import numpy as np
        rng = np.random.default_rng(seed)
        nodes = [f'N{i}' for i in range(n_nodes)]
        cats = [f'C{i}' for i in range(n_cats)]
        periods = [f'P{i}' for i in range(n_periods)]
        cols = pd.MultiIndex.from_product(
            [nodes, cats, periods], names=['node', 'category', 'period']
        )
        return pd.DataFrame(
            rng.uniform(10, 100, size=(n_scen, len(cols))), columns=cols
        )

    def test_sums_only_over_stack_level(self):
        from flextool.plot_outputs.orchestrator import _resolve_shared_axis_bounds
        df = self._build_df()
        expected_max = float(
            df.T.groupby(level=['node', 'period']).sum().T.max().max()
        )
        bounds = _resolve_shared_axis_bounds(
            df, 'shared', stack_levels=[1], subplot_levels=[2],
            always_include_zero=True,
        )
        assert bounds is not None
        assert abs(bounds[0][1] - expected_max) < 1e-6, (
            f"Shared max {bounds[0][1]:.4f} should equal per-bar stack max "
            f"{expected_max:.4f}, not be inflated by extra column levels."
        )

    def test_buggy_over_sum_would_inflate_by_expand_cardinality(self):
        """Sanity check: the old bug inflated by ~n_nodes; fix removes it."""
        from flextool.plot_outputs.orchestrator import _resolve_shared_axis_bounds
        df = self._build_df(n_nodes=5)
        bounds = _resolve_shared_axis_bounds(
            df, 'shared', stack_levels=[1], subplot_levels=[2],
            always_include_zero=True,
        )
        old_buggy_max = float('-inf')
        for p in df.columns.get_level_values('period').unique():
            sub = df.xs(p, level='period', axis=1)
            old_buggy_max = max(
                old_buggy_max, float(sub.clip(lower=0).sum(axis=1).max())
            )
        assert bounds[0][1] < old_buggy_max * 0.5, (
            "Fixed bound must be substantially narrower than the old "
            "sum-everything behavior."
        )

    def test_no_subplot_level_still_correct(self):
        from flextool.plot_outputs.orchestrator import _resolve_shared_axis_bounds
        df = self._build_df()
        expected_max = float(
            df.T.groupby(level=['node', 'period']).sum().T.max().max()
        )
        bounds = _resolve_shared_axis_bounds(
            df, 'shared', stack_levels=[1], subplot_levels=[],
            always_include_zero=True,
        )
        assert abs(bounds[0][1] - expected_max) < 1e-6

    def test_passthrough_for_non_shared_bounds(self):
        from flextool.plot_outputs.orchestrator import _resolve_shared_axis_bounds
        df = self._build_df()
        explicit = [(-10.0, 50.0)]
        assert _resolve_shared_axis_bounds(
            df, explicit, stack_levels=[1], subplot_levels=[2],
            always_include_zero=True,
        ) is explicit


# ---------------------------------------------------------------------------
# Issue B: x-axis pinned to requested duration regardless of data length
# ---------------------------------------------------------------------------

class TestXLimPinnedToDuration:
    """Renderer keeps the x-axis at the requested duration even when the
    underlying data is shorter — switching scenarios with different model
    horizons should not change the visible time window.
    """

    def _make_short_time_df(self, n_rows: int = 50) -> pd.DataFrame:
        """Time-series DataFrame with `n_rows` rows (single-level index)."""
        index = pd.Index([f"t{i:03d}" for i in range(n_rows)], name="time")
        columns = pd.MultiIndex.from_arrays(
            [["nodeA", "nodeB"]], names=["node"],
        )
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            rng.random((n_rows, 2)) * 10.0, index=index, columns=columns,
        )

    def test_build_lines_figure_pins_xlim_to_expected_length(self):
        """Direct call to _build_lines_figure with expected_x_length=100."""
        from flextool.plot_outputs.plot_lines import (
            _build_lines_figure, _compute_line_layout,
        )
        df = self._make_short_time_df(50)
        time_index = df.index.astype(str)
        effective_plots = [(None, df)]
        layout = _compute_line_layout(
            effective_plots, ["node"], "right", 1, 6.0, 4.0, "1,.0f",
        )
        fig = _build_lines_figure(
            effective_plots, "test", [], ["node"], time_index,
            subplots_per_row=1, legend_position="right",
            xlabel=None, ylabel=None,
            axis_bounds=None, axis_tick_format="1,.0f",
            always_include_zero_in_axis=True,
            layout=layout,
            shared_color_map=None,
            period_labels=None,
            expected_x_length=100,
        )
        ax = fig.axes[0]
        lo, hi = ax.get_xlim()
        # Half-step pad on either side of integer positions 0..99.
        assert abs(lo - (-0.5)) < 1e-6, f"xlim lower={lo}"
        assert abs(hi - 99.5) < 1e-6, f"xlim upper={hi}"
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_build_lines_figure_no_pin_when_expected_length_none(self):
        """Without expected_x_length, matplotlib's default auto-scale wins."""
        from flextool.plot_outputs.plot_lines import (
            _build_lines_figure, _compute_line_layout,
        )
        df = self._make_short_time_df(50)
        time_index = df.index.astype(str)
        effective_plots = [(None, df)]
        layout = _compute_line_layout(
            effective_plots, ["node"], "right", 1, 6.0, 4.0, "1,.0f",
        )
        fig = _build_lines_figure(
            effective_plots, "test", [], ["node"], time_index,
            subplots_per_row=1, legend_position="right",
            xlabel=None, ylabel=None,
            axis_bounds=None, axis_tick_format="1,.0f",
            always_include_zero_in_axis=True,
            layout=layout,
            shared_color_map=None,
            period_labels=None,
            expected_x_length=None,
        )
        ax = fig.axes[0]
        lo, hi = ax.get_xlim()
        # Auto-scale: stretches to data length 50 with matplotlib's small pad.
        # Specifically, hi should not approach 100 (the duration we'd pass when
        # the pin is active).
        assert hi < 60, f"unexpectedly wide xlim hi={hi}"
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_build_figure_from_plan_pins_xlim_via_plot_rows(self):
        """Plan-based render path: passing plot_rows=(0, 100) against 50-row
        data sets xlim to (-0.5, 99.5) — the renderer-side fix for the
        viewer's duration setting.
        """
        from flextool.plot_outputs.plan import (
            compute_live_plan, build_figure_from_plan,
        )
        df = self._make_short_time_df(50)
        cfg = PlotConfig(
            plot_name="duration-pin",
            map_dimensions_for_plots=["t_e", "t_l"],
            legend="right",
        )
        plan = compute_live_plan(df, cfg, "duration-pin")
        assert plan is not None
        # Request a duration much wider than the 50-row data we built.
        fig = build_figure_from_plan(plan, file_index=0, plot_rows=(0, 100))
        assert fig is not None
        ax = fig.axes[0]
        lo, hi = ax.get_xlim()
        assert abs(hi - 99.5) < 1e-6, f"xlim hi={hi} not pinned to duration"
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestRebuildPlanColorMapParity:
    """Stage 3.4: in-place ``rebuild_plan_color_map`` must match a full
    ``compute_live_plan`` recompute for a color/order-only template edit.

    The viewer's "Colors, order..." save path rebuilds only a cached
    plan's ``shared_color_map`` (new colors + new file order) from the
    edited template using the plan's stored ``color_category`` /
    ``color_entity_class`` hints — without recomputing dimension rules,
    layout, or the processed DataFrame.  This proves that shortcut yields
    the SAME color map (keys order + values) as the slow path.
    """

    def _write_template(self, path, mapping):
        import yaml
        path.write_text(
            yaml.safe_dump({"entities": {"node": mapping}}, sort_keys=False),
            encoding="utf-8",
        )

    def test_inplace_matches_full_recompute_stack(self, tmp_path):
        from flextool.plot_outputs import color_template as ct
        from flextool.plot_outputs.plan import (
            compute_live_plan, rebuild_plan_color_map,
        )

        df = _make_time_df(n_rows=24, n_cols=4)  # node_0..node_3
        cfg = PlotConfig(
            plot_name="recolor-parity",
            map_dimensions_for_plots=["t_e", "t_s"],  # stack on node level
            legend="shared",
            color_entity_class="node",
        )

        # Initial template: a couple of nodes pinned, in some order.
        tmpl = tmp_path / "plot_settings.yaml"
        self._write_template(tmpl, {
            "node_0": "#ff0000",
            "node_2": "#0000ff",
        })
        ct._clear_cache()
        plan = compute_live_plan(df, cfg, "recolor-parity", color_path=tmpl)
        assert plan is not None
        assert plan.shared_color_map
        assert plan.color_entity_class == "node"

        # Edit the template: CHANGE a color AND CHANGE the order (node_2
        # listed first now, node_1 newly pinned, node_0 recolored).
        self._write_template(tmpl, {
            "node_2": "#00ff00",   # was blue → green, and now first
            "node_1": "#abcdef",   # newly listed
            "node_0": "#101010",   # recolored
        })
        ct._clear_cache()

        # In-place rebuild on the already-cached plan.
        new_map = rebuild_plan_color_map(plan, color_path=tmpl)
        assert new_map is plan.shared_color_map

        # Full recompute from scratch with the edited template.
        ct._clear_cache()
        fresh = compute_live_plan(df, cfg, "recolor-parity", color_path=tmpl)
        assert fresh is not None

        # Key ORDER and VALUES must be identical.
        assert list(plan.shared_color_map.keys()) == list(
            fresh.shared_color_map.keys()
        )
        assert plan.shared_color_map == fresh.shared_color_map
        # Sanity: the edit actually took effect (node_2 → green, first key).
        assert list(plan.shared_color_map.keys())[0] == "node_2"
        assert plan.shared_color_map["node_2"] == (0.0, 1.0, 0.0)

    def test_rebuild_returns_none_without_color_map(self, tmp_path):
        """A plan with no ``shared_color_map`` (non-shared legend) is left
        untouched and the helper returns None for a graceful fallback."""
        from flextool.plot_outputs.plan import PlotPlan, rebuild_plan_color_map

        plan = PlotPlan(
            chart_type='lines',
            plot_name='p',
            total_file_count=1,
            processed_df=pd.DataFrame({'a': [1.0]}),
            effective_plot_specs=[(None, ['a'])],
            file_batches=[[0]],
            shared_color_map=None,
        )
        assert rebuild_plan_color_map(plan, color_path=None) is None
        assert plan.shared_color_map is None
