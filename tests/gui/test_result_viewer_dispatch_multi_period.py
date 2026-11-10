"""The result viewer draws the dispatch of a multi-period scenario period after period (item 181).

Drives the real ``ResultViewer._display_dispatch`` / ``_display_node_dispatch`` bodies on a lightweight
stand-in for the rest of the viewer, with in-memory 2-period results.  The Start/Duration window is
positional, so it runs across the periods: 4 rows (y2030 t1, t2, y2050 t1, t2), not 2 merged ones.

Must run under ``xvfb-run -a`` (creates a real Tk root).
"""
import tkinter as tk

import numpy as np
import pandas as pd
import pytest

from flextool.gui.result_viewer import ResultViewer
from flextool.scenario_comparison.data_models import DispatchMappings, TimeSeriesResults

pytestmark = pytest.mark.gui

SCEN = "s"
NODE = "n"
GROUP = "g"
STEPS = [("y2030", "t1"), ("y2030", "t2"), ("y2050", "t1"), ("y2050", "t2")]


def _frame(key, values, names):
    cols = pd.MultiIndex.from_tuples([(SCEN, *key)], names=["scenario", *names])
    idx = pd.MultiIndex.from_tuples(STEPS, names=["period", "time"])
    return pd.DataFrame([[v] for v in values], index=idx, columns=cols)


class _Canvas:
    def __init__(self):
        self.figure = None
        self.message = None

    def display_figure(self, fig):
        self.figure = fig

    def show_message(self, msg):
        self.message = msg


class _FakeViewer:
    """What ``_display_dispatch`` / ``_display_node_dispatch`` touch, with real Tk variables."""

    def __init__(self, root, project_path):
        res = TimeSeriesResults()
        res.unit_outputNode_dt_ee = _frame(("pv", NODE), [10.0, 20.0, 100.0, 200.0], ["e0", "e1"])
        res.node_inflow__dt = _frame((NODE,), [-11.0, -20.0, -103.0, -200.0], ["node"])
        m = DispatchMappings()
        m.dispatch_groups = pd.DataFrame({"group": [GROUP], "scenario": [SCEN]}).set_index("scenario")
        m.group_node = pd.DataFrame({"group": [GROUP], "node": [NODE], "scenario": [SCEN]}).set_index("scenario")
        m.not_in_aggregate_unit_to_node = pd.DataFrame(
            {"group": [GROUP], "process": ["pv"], "unit": ["pv"], "node": [NODE], "scenario": [SCEN]}
        ).set_index("scenario")
        self._dispatch_results = res
        self._dispatch_mappings = m
        self._dispatch_data_tag = SCEN
        self._project_path = project_path
        self._start_var = tk.IntVar(root, value=0)
        self._duration_var = tk.IntVar(root, value=10)
        self._dispatch_ylims = {}
        self._dispatch_columns = {}
        self._plot_canvas = _Canvas()
        self.time_range = None

    def _maybe_invalidate_dispatch_order_on_settings_change(self):
        pass

    def _load_dispatch_data(self, scenario):
        return True

    def _update_time_range(self, n):
        self.time_range = n

    def _load_break_times(self, scenario):
        return pd.MultiIndex.from_tuples([("y2050", "t1")], names=["period", "time"])

    def _update_file_nav(self):
        pass

    def _resolve_dispatch_template(self):
        return ResultViewer._resolve_dispatch_template(self)


@pytest.fixture()
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"No display available: {exc}")
    root.withdraw()
    yield root
    root.destroy()


def _demand_line(fig):
    ax = fig.axes[0]
    line = next(ln for ln in ax.get_lines() if ln.get_label() == "Demand")
    return np.asarray(line.get_ydata(), dtype=float)


@pytest.mark.parametrize("node_view", [False, True], ids=["group", "node"])
def test_viewer_dispatch_runs_across_the_periods(tk_root, tmp_path, node_view):
    fake = _FakeViewer(tk_root, tmp_path)
    if node_view:
        ResultViewer._display_node_dispatch(fake, SCEN, NODE)
    else:
        ResultViewer._display_dispatch(fake, SCEN, GROUP)
    assert fake._plot_canvas.message is None
    assert fake.time_range == 4
    y = _demand_line(fake._plot_canvas.figure)
    # 4 steps plus the 3 gap rows before y2050 (it does not continue y2030)
    assert len(y) == 7
    assert y[[0, 1, 5, 6]].tolist() == [11, 20, 103, 200]
    assert np.isnan(y[2:5]).all()
