"""Draw/legend order follows the shared color map's key order.

Stack and stacked-bar charts already ordered their series by
``shared_color_map`` key order (file order from ``plot_settings.yaml``).
These tests pin the same behaviour for **line** charts (trace/legend order)
and **grouped bars** (side-by-side category order), which previously drew in
raw-dataframe order / alphabetical order and so ignored the picker's
reordering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - matplotlib always present in CI
    plt = None


def test_line_trace_order_follows_color_map() -> None:
    """Line traces (and thus the legend) are drawn in color-map key order,
    not the raw dataframe column order."""
    from flextool.plot_outputs.plot_lines import (
        _build_lines_figure, _compute_line_layout,
    )
    # Native column order is nodeB, nodeA — the OPPOSITE of the map order.
    index = pd.Index([f"t{i:03d}" for i in range(5)], name="time")
    columns = pd.MultiIndex.from_arrays([["nodeB", "nodeA"]], names=["node"])
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.random((5, 2)) * 10.0, index=index, columns=columns)
    effective_plots = [(None, df)]
    # Map lists nodeA first — the reordered file order.
    shared_color_map = {"nodeA": (1.0, 0.0, 0.0), "nodeB": (0.0, 0.0, 1.0)}

    layout = _compute_line_layout(
        effective_plots, ["node"], "right", 1, 6.0, 4.0, "1,.0f",
    )
    fig = _build_lines_figure(
        effective_plots, "test", [], ["node"], index.astype(str),
        subplots_per_row=1, legend_position="right",
        xlabel=None, ylabel=None,
        axis_bounds=None, axis_tick_format="1,.0f",
        always_include_zero_in_axis=True,
        layout=layout,
        shared_color_map=shared_color_map,
        period_labels=None,
        expected_x_length=None,
    )
    ax = fig.axes[0]
    drawn = [ln.get_label() for ln in ax.get_lines()]
    plt.close(fig)
    # Draw order tracks the map, not the raw column order (nodeB, nodeA).
    assert drawn == ["nodeA", "nodeB"], drawn


def test_line_trace_order_native_when_no_color_map() -> None:
    """Without a shared color map, line order stays the raw column order."""
    from flextool.plot_outputs.plot_lines import (
        _build_lines_figure, _compute_line_layout,
    )
    index = pd.Index([f"t{i:03d}" for i in range(5)], name="time")
    columns = pd.MultiIndex.from_arrays([["nodeB", "nodeA"]], names=["node"])
    df = pd.DataFrame(
        np.ones((5, 2)), index=index, columns=columns,
    )
    effective_plots = [(None, df)]
    layout = _compute_line_layout(
        effective_plots, ["node"], "right", 1, 6.0, 4.0, "1,.0f",
    )
    fig = _build_lines_figure(
        effective_plots, "test", [], ["node"], index.astype(str),
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
    drawn = [ln.get_label() for ln in ax.get_lines()]
    plt.close(fig)
    assert drawn == ["nodeB", "nodeA"], drawn


def test_grouped_bar_order_follows_color_map() -> None:
    """Grouped (side-by-side) bar categories follow the color-map key order,
    not alphabetical."""
    from flextool.plot_outputs.plot_bars_detail import _plot_grouped_bars

    # Native category order B, A, C; alphabetical would be A, B, C.
    cols = pd.MultiIndex.from_tuples(
        [("B",), ("A",), ("C",)], names=["scenario"],
    )
    df = pd.DataFrame(
        np.array([[10.0, 20.0, 30.0]]),
        index=pd.Index(["p1"], name="period"),
        columns=cols,
    )
    # Map lists C, A, B — the reordered file order.
    shared_color_map = {
        "C": (1.0, 0.0, 0.0), "A": (0.0, 1.0, 0.0), "B": (0.0, 0.0, 1.0),
    }
    fig, ax = plt.subplots(figsize=(3, 2))
    _plot_grouped_bars(
        ax,
        df,
        all_bars=[(None, "p1")],
        expand_axis_level_names=[],
        grouped_bar_level_names=["scenario"],
        bar_orientation="horizontal",
        value_fmt=None,
        shared_color_map=shared_color_map,
    )
    labels = ax.get_legend_handles_labels()[1]
    plt.close(fig)
    # Category order tracks the map (C, A, B), not alphabetical (A, B, C).
    assert labels == ["C", "A", "B"], labels


def test_grouped_bar_order_alphabetical_absent_from_map_go_last() -> None:
    """A category missing from the map keeps its position AFTER the mapped
    ones (defensive fallback), rather than jumping to the front."""
    from flextool.plot_outputs.plot_bars_detail import _plot_grouped_bars

    cols = pd.MultiIndex.from_tuples(
        [("B",), ("A",), ("Z",)], names=["scenario"],
    )
    df = pd.DataFrame(
        np.array([[10.0, 20.0, 30.0]]),
        index=pd.Index(["p1"], name="period"),
        columns=cols,
    )
    # Z is not in the map → it should trail A, B (which are ordered A, B).
    shared_color_map = {"A": (0.0, 1.0, 0.0), "B": (0.0, 0.0, 1.0)}
    fig, ax = plt.subplots(figsize=(3, 2))
    _plot_grouped_bars(
        ax,
        df,
        all_bars=[(None, "p1")],
        expand_axis_level_names=[],
        grouped_bar_level_names=["scenario"],
        bar_orientation="horizontal",
        value_fmt=None,
        shared_color_map=shared_color_map,
    )
    labels = ax.get_legend_handles_labels()[1]
    plt.close(fig)
    assert labels == ["A", "B", "Z"], labels
