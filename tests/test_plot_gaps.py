"""Line and stack plots draw an empty result row and a timeline-gap row as a gap, not as 0.

Asserted on the figure's data (line y-data, the stacked areas' polygons), not on pixels.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from flextool.plot_outputs.config import PlotConfig
from flextool.plot_outputs.format_helpers import insert_timeline_breaks
from flextool.plot_outputs.orchestrator import _apply_dimension_rules
from flextool.plot_outputs.plot_lines import (
    _build_lines_figure,
    _build_stack_figure,
    _compute_line_layout,
)

#: Rows 0-5 of one period; row 2 is empty in every column; a timeline break at t5.
_TIMES = [f"t{i}" for i in range(6)]


def _frame() -> pd.DataFrame:
    """Two nodes with two units each (the unit level is summed in the drawing); a break before t5."""
    index = pd.MultiIndex.from_tuples([("p1", t) for t in _TIMES], names=["period", "time"])
    columns = pd.MultiIndex.from_tuples(
        [("n1", "u1"), ("n1", "u2"), ("n2", "u1"), ("n2", "u2")], names=["node", "unit"])
    values = np.arange(24, dtype=float).reshape(6, 4) + 1.0
    values[2, :] = np.nan
    df = pd.DataFrame(values, index=index, columns=columns)
    df.loc[("p1", "t4"), ("n2", "u2")] = np.nan  # one empty cell in a row with values: the sum is the rest
    breaks = pd.MultiIndex.from_tuples([("p1", "t5")], names=["period", "time"])
    out = insert_timeline_breaks(df, breaks)
    assert len(out) == 9  # 3 gap rows before t5
    return out


#: Positions of the drawn rows that must be gaps: the empty row t2 and the 3 inserted gap rows.
_GAP_POSITIONS = [2, 5, 6, 7]


def _build(builder, df: pd.DataFrame):
    effective_plots = [(None, df)]
    layout = _compute_line_layout(effective_plots, ["node"], "right", 1, 6.0, 4.0, "1,.0f")
    time_index = df.index.get_level_values("time").astype(str)
    return builder(
        effective_plots, "test", [], ["node"], time_index,
        subplots_per_row=1, legend_position="right", xlabel=None, ylabel=None,
        axis_bounds=None, axis_tick_format="1,.0f", always_include_zero_in_axis=True,
        layout=layout, shared_color_map=None, period_labels=None, expected_x_length=None,
    )


def test_line_plot_draws_empty_and_gap_rows_as_nan() -> None:
    df = _frame()
    fig = _build(_build_lines_figure, df)
    lines = {ln.get_label(): np.asarray(ln.get_ydata(), dtype=float) for ln in fig.axes[0].get_lines()}
    assert set(lines) == {"n1", "n2"}
    for y in lines.values():
        assert np.isnan(y[_GAP_POSITIONS]).all(), y
        assert not np.isnan(np.delete(y, _GAP_POSITIONS)).any(), y
    # Hand sums: n1 at t0 = 1 + 2; n2 at t4 = 19 (its other cell is empty).
    assert lines["n1"][0] == 3.0
    assert lines["n2"][4] == 19.0


def test_stack_plot_draws_nothing_at_empty_and_gap_rows() -> None:
    df = _frame()
    fig = _build(_build_stack_figure, df)
    ax = fig.axes[0]
    covered: set[int] = set()
    for coll in ax.collections:
        for path in coll.get_paths():
            covered.update(int(round(x)) for x in path.vertices[:, 0])
    assert covered == {0, 1, 3, 4, 8}
    # The whole time axis stays in view.
    lo, hi = ax.get_xlim()
    assert lo <= 0 and hi >= 8


def test_dimension_rule_sum_keeps_an_all_empty_cell_empty_in_a_time_chart() -> None:
    """An 'm' (sum) column level: all-empty → NaN in a time chart; a bar chart keeps 0 (its frames are zero-filled)."""
    df = _frame().droplevel("period")
    df = df[~df.index.duplicated(keep="first")]  # drop the gap rows: this is the frame before breaks are inserted
    time_cfg = PlotConfig(map_dimensions_for_plots=["t_ee", "tlm"])
    out, _rules, chart_type, *_ = _apply_dimension_rules(df, time_cfg, (0, len(df)))
    assert chart_type == "time"
    assert out.loc[("t2",)].isna().all()
    assert out.loc[("t4",), "n2"].item() == 19.0
