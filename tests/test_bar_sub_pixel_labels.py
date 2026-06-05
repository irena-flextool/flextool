"""Regression tests: sub-pixel bars must still get their value labels.

When a bar's value is tiny relative to the largest bar in the same chart
(e.g. comparing two scenarios where one is ~120x the other), the small
bar falls below the 1-display-pixel width threshold. The bar itself stays
invisibly thin, but its numeric value LABEL must still be drawn — the
label is the whole point of showing that data exists at that category.

These tests exercise the draw functions directly and count the value-label
Text artists (matplotlib's ``ax.bar_label`` emits one Text per bar). Prior
to the fix, sub-pixel bars were dropped from the drawn container and so got
no label; the asserts below would have seen only the single largest bar's
label.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from flextool.plot_outputs.plot_bars_detail import (  # noqa: E402
    _autoscale_value_range,
    _min_visible_data_width,
    _plot_grouped_bars,
    _plot_simple_bars,
)

# One huge value plus two values that are far below the 1-pixel threshold in
# a small figure. The 120000:10:5 ratio guarantees the small bars are
# sub-pixel (verified by _assert_small_bars_subpixel).
_VALUES = np.array([120000.0, 10.0, 5.0])


def _assert_small_bars_subpixel() -> None:
    """Guard: the two small values really are below the draw threshold.

    If a future geometry change made them visible, the regression these
    tests protect would no longer be exercised — fail loudly so the test is
    updated rather than silently passing for the wrong reason.
    """
    fig, ax = plt.subplots(figsize=(3, 2))
    lo, hi = _autoscale_value_range(_VALUES)
    ax.set_xlim(lo, hi)
    fig.canvas.draw()
    threshold = _min_visible_data_width(ax, "x")
    plt.close(fig)
    n_small_subpixel = int((np.abs(_VALUES[1:]) < threshold).sum())
    assert n_small_subpixel == 2, (
        f"expected both small bars sub-pixel, got {n_small_subpixel} "
        f"(threshold={threshold})"
    )


def test_simple_bars_subpixel_values_still_labeled() -> None:
    """Every non-zero simple bar gets a label, even sub-pixel ones."""
    _assert_small_bars_subpixel()
    df = pd.DataFrame(
        _VALUES.reshape(-1, 1),
        index=pd.Index(["p1", "p2", "p3"], name="period"),
        columns=pd.Index(["x"], name="item"),
    )
    fig, ax = plt.subplots(figsize=(3, 2))
    _plot_simple_bars(
        ax,
        df,
        all_bars=[(None, "p1"), (None, "p2"), (None, "p3")],
        expand_axis_level_names=[],
        bar_orientation="horizontal",
        value_fmt="dynamic",
    )
    fig.canvas.draw()
    texts = [t.get_text() for t in ax.texts]
    plt.close(fig)
    # One label per non-zero bar — including the two sub-pixel ones.
    assert len(texts) == 3, f"expected 3 value labels, got {texts}"


def test_grouped_bars_subpixel_values_still_labeled() -> None:
    """Every non-zero grouped bar gets a label, even sub-pixel ones."""
    _assert_small_bars_subpixel()
    cols = pd.MultiIndex.from_tuples(
        [("A",), ("B",), ("C",)], names=["scenario"]
    )
    df = pd.DataFrame(
        _VALUES.reshape(1, -1),
        index=pd.Index(["p1"], name="period"),
        columns=cols,
    )
    fig, ax = plt.subplots(figsize=(3, 2))
    _plot_grouped_bars(
        ax,
        df,
        all_bars=[(None, "p1")],
        expand_axis_level_names=[],
        grouped_bar_level_names=["scenario"],
        bar_orientation="horizontal",
        value_fmt="dynamic",
    )
    fig.canvas.draw()
    texts = [t.get_text() for t in ax.texts]
    plt.close(fig)
    # One label per non-zero grouped category — including the sub-pixel ones.
    assert len(texts) == 3, f"expected 3 value labels, got {texts}"


def test_zero_value_bars_are_not_labeled() -> None:
    """A genuinely zero bar produces no label (only non-zero data labeled)."""
    df = pd.DataFrame(
        np.array([[100.0], [0.0], [5.0]]),
        index=pd.Index(["p1", "p2", "p3"], name="period"),
        columns=pd.Index(["x"], name="item"),
    )
    fig, ax = plt.subplots(figsize=(3, 2))
    _plot_simple_bars(
        ax,
        df,
        all_bars=[(None, "p1"), (None, "p2"), (None, "p3")],
        expand_axis_level_names=[],
        bar_orientation="horizontal",
        value_fmt="dynamic",
    )
    fig.canvas.draw()
    texts = [t.get_text() for t in ax.texts]
    plt.close(fig)
    assert len(texts) == 2, f"expected 2 value labels (zero skipped), got {texts}"
