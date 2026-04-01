"""Tests for the bar layout computation system.

Verifies that _compute_bar_layout produces correct BarLayoutParams
for various DataFrame shapes and configuration options.
"""

import pandas as pd
import numpy as np
import pytest

from flextool.plot_outputs.plot_bars import _compute_bar_layout
from flextool.plot_outputs.subplot_helpers import BarLayoutParams


def _make_multiindex_df(
    level_tuples: list[tuple[str, ...]],
    level_names: list[str],
    n_rows: int = 3,
    row_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Create a synthetic DataFrame with MultiIndex columns for testing.

    Parameters
    ----------
    level_tuples : list of tuples
        Each tuple defines one column in the MultiIndex.
    level_names : list of str
        Names for each level of the MultiIndex.
    n_rows : int
        Number of rows in the DataFrame.
    row_labels : list of str or None
        Row index labels. Defaults to ['p1', 'p2', ...].

    Returns
    -------
    pd.DataFrame
        DataFrame with random data and the specified MultiIndex columns.
    """
    columns = pd.MultiIndex.from_tuples(level_tuples, names=level_names)
    if row_labels is None:
        row_labels = [f"p{i + 1}" for i in range(n_rows)]
    index = pd.Index(row_labels, name="period")
    rng = np.random.default_rng(42)
    data = rng.random((n_rows, len(level_tuples)))
    return pd.DataFrame(data, index=index, columns=columns)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_subplot_no_layout() -> None:
    """Simple DataFrame, no expand_axis / stack / grouped_bar levels.

    Expects bar_label_width > 0, group_label_width == 0,
    total_label_width == bar_label_width, and legend_width == 0.
    """
    df = pd.DataFrame(
        np.random.default_rng(0).random((3, 2)),
        index=pd.Index(["p1", "p2", "p3"], name="period"),
        columns=pd.Index(["colA", "colB"], name="item"),
    )
    effective_plots: list[tuple[str | None, pd.DataFrame]] = [(None, df)]

    layout = _compute_bar_layout(
        effective_plots=effective_plots,
        df=df,
        expand_axis_levels=[],
        expand_axis_level_names=[],
        stack_levels=[],
        stack_level_names=[],
        grouped_bar_levels=[],
        grouped_bar_level_names=[],
        legend_position="right",
        subplots_per_row=3,
        base_bar_length=4.0,
    )

    assert isinstance(layout, BarLayoutParams)
    assert layout.bar_label_width > 0
    assert layout.group_label_width == 0
    assert layout.total_label_width == layout.bar_label_width
    assert layout.legend_width == 0


def test_bar_label_width_scales_with_longest_label() -> None:
    """Longer bar labels should produce a wider bar_label_width."""
    short_labels = ["A", "B"]
    long_labels = ["very_long_label_name", "another_long_one"]

    df_short = pd.DataFrame(
        np.random.default_rng(1).random((2, 2)),
        index=pd.Index(short_labels, name="period"),
        columns=pd.Index(["x", "y"], name="item"),
    )
    df_long = pd.DataFrame(
        np.random.default_rng(2).random((2, 2)),
        index=pd.Index(long_labels, name="period"),
        columns=pd.Index(["x", "y"], name="item"),
    )

    eff_short: list[tuple[str | None, pd.DataFrame]] = [("short", df_short)]
    eff_long: list[tuple[str | None, pd.DataFrame]] = [("long", df_long)]

    layout_short = _compute_bar_layout(
        effective_plots=eff_short,
        df=df_short,
        expand_axis_levels=[],
        expand_axis_level_names=[],
        stack_levels=[],
        stack_level_names=[],
        grouped_bar_levels=[],
        grouped_bar_level_names=[],
        legend_position="right",
        subplots_per_row=3,
        base_bar_length=4.0,
    )

    layout_long = _compute_bar_layout(
        effective_plots=eff_long,
        df=df_long,
        expand_axis_levels=[],
        expand_axis_level_names=[],
        stack_levels=[],
        stack_level_names=[],
        grouped_bar_levels=[],
        grouped_bar_level_names=[],
        legend_position="right",
        subplots_per_row=3,
        base_bar_length=4.0,
    )

    assert layout_long.bar_label_width > layout_short.bar_label_width


def test_group_label_width_with_expand_axis() -> None:
    """When expand_axis_levels is set, group_label_width should be positive
    and total_label_width should equal bar_label_width + group_label_width.
    """
    index = pd.MultiIndex.from_tuples(
        [("scA", "n1"), ("scA", "n2"), ("scB", "n1"), ("scB", "n2")],
        names=["scenario", "node"],
    )
    df = pd.DataFrame(
        np.random.default_rng(3).random((3, 4)),
        index=pd.Index(["p1", "p2", "p3"], name="period"),
        columns=index,
    )

    effective_plots: list[tuple[str | None, pd.DataFrame]] = [(None, df)]

    layout = _compute_bar_layout(
        effective_plots=effective_plots,
        df=df,
        expand_axis_levels=[0],
        expand_axis_level_names=["scenario"],
        stack_levels=[],
        stack_level_names=[],
        grouped_bar_levels=[],
        grouped_bar_level_names=[],
        legend_position="right",
        subplots_per_row=3,
        base_bar_length=4.0,
    )

    assert layout.group_label_width > 0
    assert layout.total_label_width == layout.bar_label_width + layout.group_label_width


def test_legend_width_with_stacked_bars() -> None:
    """With stack_levels, legend_position='all', and n_cols > 1,
    legend_width should be positive.
    """
    columns = pd.MultiIndex.from_tuples(
        [("coal", "s1"), ("coal", "s2"), ("gas", "s1"), ("gas", "s2")],
        names=["stack_col", "sub"],
    )
    df = pd.DataFrame(
        np.random.default_rng(4).random((3, 4)),
        index=pd.Index(["p1", "p2", "p3"], name="period"),
        columns=columns,
    )

    # Need at least 3 effective_plots so that n_cols > 1 with subplots_per_row=2
    effective_plots: list[tuple[str | None, pd.DataFrame]] = [
        ("plot1", df),
        ("plot2", df),
        ("plot3", df),
    ]

    layout = _compute_bar_layout(
        effective_plots=effective_plots,
        df=df,
        expand_axis_levels=[],
        expand_axis_level_names=[],
        stack_levels=[0],
        stack_level_names=["stack_col"],
        grouped_bar_levels=[],
        grouped_bar_level_names=[],
        legend_position="all",
        subplots_per_row=2,
        base_bar_length=4.0,
    )

    assert layout.legend_width > 0


def test_legend_width_positive_when_right_position() -> None:
    """With legend_position='right' and stack levels, legend_width should
    still be positive (it measures the legend size for placement).
    """
    columns = pd.MultiIndex.from_tuples(
        [("coal", "s1"), ("coal", "s2"), ("gas", "s1"), ("gas", "s2")],
        names=["stack_col", "sub"],
    )
    df = pd.DataFrame(
        np.random.default_rng(5).random((3, 4)),
        index=pd.Index(["p1", "p2", "p3"], name="period"),
        columns=columns,
    )

    effective_plots: list[tuple[str | None, pd.DataFrame]] = [
        ("plot1", df),
        ("plot2", df),
        ("plot3", df),
    ]

    layout = _compute_bar_layout(
        effective_plots=effective_plots,
        df=df,
        expand_axis_levels=[],
        expand_axis_level_names=[],
        stack_levels=[0],
        stack_level_names=["stack_col"],
        grouped_bar_levels=[],
        grouped_bar_level_names=[],
        legend_position="right",
        subplots_per_row=2,
        base_bar_length=4.0,
    )

    assert layout.legend_width > 0


def test_layout_params_consistent_across_batches() -> None:
    """Layout computed once for ALL effective_plots should yield consistent
    values regardless of which batch the plots belong to.
    """
    index = pd.MultiIndex.from_tuples(
        [("scA", "n1"), ("scA", "n2"), ("scB", "n1"), ("scB", "n2")],
        names=["scenario", "node"],
    )
    df = pd.DataFrame(
        np.random.default_rng(6).random((3, 4)),
        index=pd.Index(["p1", "p2", "p3"], name="period"),
        columns=index,
    )

    # 6 effective_plots (could be split into 2 batches of 3)
    all_effective_plots: list[tuple[str | None, pd.DataFrame]] = [
        (f"plot{i}", df) for i in range(6)
    ]
    batch_1 = all_effective_plots[:3]
    batch_2 = all_effective_plots[3:]

    common_kwargs = dict(
        df=df,
        expand_axis_levels=[0],
        expand_axis_level_names=["scenario"],
        stack_levels=[],
        stack_level_names=[],
        grouped_bar_levels=[],
        grouped_bar_level_names=[],
        legend_position="right",
        subplots_per_row=3,
        base_bar_length=4.0,
    )

    # Compute layout once for all plots (the correct approach)
    layout_all = _compute_bar_layout(effective_plots=all_effective_plots, **common_kwargs)

    # Verify that key measurements are positive
    assert layout_all.bar_label_width > 0
    assert layout_all.group_label_width > 0
    assert layout_all.total_label_width > 0

    # Verify internal consistency
    assert layout_all.total_label_width == layout_all.bar_label_width + layout_all.group_label_width

    # Compute layout separately for each batch and verify they match
    # (since all subplots share the same DataFrame structure, the label
    # widths should be identical)
    layout_b1 = _compute_bar_layout(effective_plots=batch_1, **common_kwargs)
    layout_b2 = _compute_bar_layout(effective_plots=batch_2, **common_kwargs)

    assert layout_b1.bar_label_width == layout_b2.bar_label_width
    assert layout_b1.group_label_width == layout_b2.group_label_width
    assert layout_b1.total_label_width == layout_b2.total_label_width
    assert layout_b1.bar_label_width == layout_all.bar_label_width
    assert layout_b1.group_label_width == layout_all.group_label_width
    assert layout_b1.total_label_width == layout_all.total_label_width
