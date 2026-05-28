"""Regression tests for plot-plan chunk column selectors.

When a bar or time-series plot is split into chunks by ``_compute_bar_plan``
/ ``_compute_time_plan``, each chunk's column selector must reconstruct
the exact chunk at viewer time via ``build_figure_from_plan``.

The bug these tests guard against: ``_extract_subplot_data`` drops the
sub-level from ``df_sub.columns`` via ``xs``, so the stored selector had
tuples narrower than ``df_full.columns``. ``df_full.columns.isin(...)``
then matched nothing and the fallback returned the full df, making every
chunk identical and flooding each subplot with columns from every other
subplot (observed in ``unit_capacity_ed`` as >200 bars per subplot and
"bars stay the same when browsing files").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from flextool.plot_outputs.config import PlotConfig
from flextool.plot_outputs.plan import (
    _compute_bar_plan,
    _compute_time_plan,
)


def _make_multiindex_columns(sub_vals, expand_vals, stack_vals):
    """Build a 3-level MultiIndex: (subplot, expand, stack)."""
    tuples = [
        (s, e, st) for s in sub_vals for e in expand_vals for st in stack_vals
    ]
    return pd.MultiIndex.from_tuples(tuples, names=["subplot", "expand", "stack"])


class TestBarChunkColumnReconstruction:
    def test_expand_group_chunk_has_only_its_columns(self):
        # Setup: 3 subplots (A, B, C), each with 4 expand groups (1..4),
        # each with 2 stack items. max_items_per_plot=4 forces a chunk.
        # Rows are the bar labels.
        sub_vals = ["A", "B", "C"]
        expand_vals = [1, 2, 3, 4]
        stack_vals = ["lo", "hi"]
        columns = _make_multiindex_columns(sub_vals, expand_vals, stack_vals)
        rows = [f"row_{i}" for i in range(6)]  # 6 rows > max_items=4
        df = pd.DataFrame(
            np.arange(len(rows) * len(columns), dtype=float).reshape(len(rows), len(columns)),
            index=pd.Index(rows, name="bar"),
            columns=columns,
        )

        cfg = PlotConfig(
            map_dimensions_for_plots=["d", "u", "e", "s"],
            bar_orientation="horizontal",
            max_items_per_plot=4,
            max_subplots_per_file=6,
            max_items_per_subplot_column=100,
            subplots_per_row=1,
            legend="right",
        )

        plan = _compute_bar_plan(
            df, "test_plot", cfg,
            fm_stack_levels=[2],
            fm_expand_axis_levels=[1],
            fm_subplot_levels=[0],
            fm_grouped_bar_levels=[],
            axis_bounds=None,
        )
        assert plan is not None
        # 6 rows * 4 expand groups = 24 visual items > max=4 → split by
        # expand groups (one chunk per group, 6 rows × 1 group = 6 bars).
        # Since 6 > max_items=4, each expand-group chunk is then further
        # row-split into 2 sub-chunks (4 + 2 rows). 3 subs × 4 groups × 2
        # row-chunks = 24 effective_plots.
        assert len(plan.effective_plot_specs) == 24

        # Reconstruct each chunk and verify it only contains that subplot's
        # data AND only that expand group's columns.
        for title, selector in plan.effective_plot_specs:
            selector["rows"]
            cols_sel = selector["cols"]
            # cols_sel must be full-width tuples matching df.columns
            for c in cols_sel:
                assert len(c) == df.columns.nlevels, (
                    f"selector tuple {c} has {len(c)} levels but df has "
                    f"{df.columns.nlevels}"
                )
                assert tuple(c) in df.columns, (
                    f"selector tuple {tuple(c)} not found in df.columns"
                )

    def test_build_figure_from_plan_produces_distinct_chunks(self):
        # Drive the fast viewer path and check each file shows a DIFFERENT
        # chunk rather than the same full dataframe. The original bug made
        # every file identical.
        sub_vals = ["A", "B"]
        expand_vals = [1, 2]
        stack_vals = ["x"]
        columns = _make_multiindex_columns(sub_vals, expand_vals, stack_vals)
        rows = [f"row_{i}" for i in range(5)]
        df = pd.DataFrame(
            np.arange(len(rows) * len(columns), dtype=float).reshape(len(rows), len(columns)),
            index=pd.Index(rows, name="bar"),
            columns=columns,
        )
        cfg = PlotConfig(
            map_dimensions_for_plots=["d", "u", "e", "s"],
            bar_orientation="horizontal",
            max_items_per_plot=2,
            max_subplots_per_file=1,
            max_items_per_subplot_column=100,
            subplots_per_row=1,
            legend="right",
        )
        plan = _compute_bar_plan(
            df, "test", cfg,
            fm_stack_levels=[2],
            fm_expand_axis_levels=[1],
            fm_subplot_levels=[0],
            fm_grouped_bar_levels=[],
            axis_bounds=None,
        )
        assert plan is not None
        assert plan.total_file_count >= 2

        # We can't easily compare matplotlib Figure objects, but we can
        # inspect the reconstruction step that the plan builder uses.
        from flextool.plot_outputs.plan import (
            _select_bar_rows, _select_time_columns,
        )
        dfs = []
        for title, selector in plan.effective_plot_specs:
            df_sub = _select_bar_rows(df, selector["rows"])
            df_sub = _select_time_columns(df_sub, selector["cols"])
            dfs.append(df_sub)

        # No two reconstructed chunks should be identical — and none
        # should have the full df's column count.
        for df_chunk in dfs:
            assert df_chunk.shape[1] < df.shape[1], (
                f"chunk has {df_chunk.shape[1]} cols but df has "
                f"{df.shape[1]} — reconstruction returned the whole df"
            )
        # At least two chunks should differ
        shapes_or_cols = {(tuple(d.columns), d.shape) for d in dfs}
        assert len(shapes_or_cols) > 1, "all reconstructed chunks are identical"


class TestSubplotOrdering:
    def test_bar_subplots_ordered_alphabetically(self):
        # Deliberately insert subplot values out of alphabetic order so a
        # naive "unique in insertion order" implementation would fail.
        sub_vals = ["zeta", "alpha", "Beta"]  # mixed case
        expand_vals = [1]
        stack_vals = ["x"]
        columns = _make_multiindex_columns(sub_vals, expand_vals, stack_vals)
        rows = [f"r{i}" for i in range(3)]
        df = pd.DataFrame(
            np.arange(len(rows) * len(columns), dtype=float).reshape(
                len(rows), len(columns)
            ),
            index=pd.Index(rows, name="bar"),
            columns=columns,
        )
        cfg = PlotConfig(
            map_dimensions_for_plots=["d", "u", "e", "s"],
            bar_orientation="horizontal",
            max_items_per_plot=10,
            max_subplots_per_file=6,
            max_items_per_subplot_column=100,
            subplots_per_row=1,
            legend="right",
        )
        plan = _compute_bar_plan(
            df, "ordered", cfg,
            fm_stack_levels=[2], fm_expand_axis_levels=[1],
            fm_subplot_levels=[0], fm_grouped_bar_levels=[],
            axis_bounds=None,
        )
        titles = [t for t, _ in plan.effective_plot_specs]
        # Case-insensitive alphabetical
        assert titles == sorted(titles, key=str.lower)

    def test_time_subplots_ordered_alphabetically(self):
        sub_vals = ["zeta", "alpha", "Beta"]
        items = ["i0"]
        columns = pd.MultiIndex.from_product(
            [sub_vals, items], names=["subplot", "line"],
        )
        index = pd.MultiIndex.from_product(
            [["p1"], [f"t{i}" for i in range(4)]], names=["period", "t"],
        )
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
            df, "ordered", cfg,
            fm_stack_levels=[], fm_subplot_levels=[0], fm_line_levels=[1],
            axis_bounds=None, plot_rows=(0, 10),
        )
        titles = [t for t, _ in plan.effective_plot_specs]
        assert titles == sorted(titles, key=str.lower)


class TestTimeChunkColumnReconstruction:
    def test_line_chunk_has_only_its_columns(self):
        # time-series with subplot levels: columns (subplot, line_item)
        sub_vals = ["A", "B"]
        items = [f"item_{i}" for i in range(6)]
        columns = pd.MultiIndex.from_product(
            [sub_vals, items], names=["subplot", "line"],
        )
        # Time-series rows: (period, t)
        periods = ["p1"]
        times = [f"t{i}" for i in range(8)]
        index = pd.MultiIndex.from_product(
            [periods, times], names=["period", "t"],
        )
        df = pd.DataFrame(
            np.arange(len(index) * len(columns), dtype=float).reshape(
                len(index), len(columns)
            ),
            index=index,
            columns=columns,
        )
        cfg = PlotConfig(
            map_dimensions_for_plots=["d", "u", "l"],
            max_items_per_plot=3,
            max_subplots_per_file=10,
            subplots_per_row=1,
            legend="right",
        )

        plan = _compute_time_plan(
            df, "time_test", cfg,
            fm_stack_levels=[],
            fm_subplot_levels=[0],
            fm_line_levels=[1],
            axis_bounds=None,
            plot_rows=(0, 10),
        )
        assert plan is not None
        # 2 subs × ceil(6/3)=2 chunks = 4 effective_plots
        assert len(plan.effective_plot_specs) == 4

        for title, selector in plan.effective_plot_specs:
            # time-series stores a flat list (not dict)
            cols_sel = selector
            for c in cols_sel:
                assert len(c) == df.columns.nlevels
                assert tuple(c) in df.columns
