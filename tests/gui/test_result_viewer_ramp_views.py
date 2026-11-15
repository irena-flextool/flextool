"""The result viewer's dispatch-tree item shows Dispatch / Ramp envelope / Ramp frequency.

Drives the real ``ResultViewer._display_dispatch_iid`` / ``_display_ramp`` bodies on a lightweight stand-in for
the rest of the viewer, with in-memory envelope tables of one scenario (oracle v2 Case 6 entity A = the Case 1
node: rows t1 and t3 filled, t2 and t4 empty).  The group and the node of the dispatch tree reach the same views.

Must run under ``xvfb-run -a`` (creates a real Tk root).
"""
import math
import tkinter as tk

import numpy as np
import pandas as pd
import pytest

from flextool.gui.result_viewer import (
    _DISPATCH_GROUP_PREFIX,
    _DISPATCH_NODE_PREFIX,
    _DISPATCH_VIEWS,
    ResultViewer,
)
from flextool.scenario_comparison.data_models import TimeSeriesResults

pytestmark = pytest.mark.gui

SCEN = "s"
BANDS = ("ramp_net_load", "ramp_demand", "units_up", "storage_up", "vre_up", "connections_up",
         "units_down", "storage_down", "vre_down", "connections_down")
ROWS = pd.MultiIndex.from_tuples([("p", "t1"), ("p", "t2"), ("p", "t3"), ("p", "t4")], names=["period", "time"])
ENTITY_A = [[35, 20, 65, 30, 10, 20, -30, -7, -5, -40], [math.nan] * 10,
            [-50, -20, 60, 3, 0, 25, -50, -35, -30, -35], [math.nan] * 10]


def _table(entity: str, level: str) -> pd.DataFrame:
    cols = pd.MultiIndex.from_product([[SCEN], [entity], BANDS], names=["scenario", level, "band"])
    return pd.DataFrame(ENTITY_A, index=ROWS, columns=cols, dtype=float)


def _param(name: str, values) -> pd.DataFrame:
    cols = pd.MultiIndex.from_tuples([(SCEN, name)], names=["scenario", "param"])
    return pd.DataFrame({cols[0]: values}, index=ROWS)


class _Canvas:
    def __init__(self):
        self.figure = None
        self.message = None

    def display_figure(self, fig):
        self.figure = fig

    def show_message(self, msg):
        self.message = msg


class _FakeViewer:
    """What ``_display_dispatch_iid`` / ``_display_ramp`` touch, with real Tk variables."""

    def __init__(self, root, project_path, view):
        res = TimeSeriesResults()
        res.node_ramp_envelope_dt_ep = _table("n", "node")
        res.nodeGroup_ramp_envelope_dt_gp = _table("G", "group")
        res.timestep_hours_dt = _param("timestep_hours", [100.0, 100.0, 50.0, 50.0])
        res.transition_hours_dt = _param("transition_hours", [1.0, math.nan, 1.0, math.nan])
        self._dispatch_results = res
        self._dispatch_data_tag = SCEN
        self._project_path = project_path
        self._start_var = tk.IntVar(root, value=0)
        self._duration_var = tk.IntVar(root, value=10)
        self._dispatch_view = tk.StringVar(root, value=view)
        self._dispatch_node_iids = {f"{_DISPATCH_NODE_PREFIX}0_0_n": "n"}
        self._last_dispatch_iid = None
        self._plot_canvas = _Canvas()
        self.time_range = None
        self.dispatch_calls = []

    def _dispatch_view_name(self):
        return ResultViewer._dispatch_view_name(self)

    def _display_ramp(self, scenario, kind, entity, view):
        return ResultViewer._display_ramp(self, scenario, kind, entity, view)

    def _display_dispatch(self, scenario, group):
        self.dispatch_calls.append(("group", group))

    def _display_node_dispatch(self, scenario, node):
        self.dispatch_calls.append(("node", node))

    def _load_dispatch_data(self, scenario):
        return True

    def _update_time_range(self, n):
        self.time_range = n

    def _load_break_times(self, scenario):
        return None

    def _update_file_nav(self):
        pass


@pytest.fixture()
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"No display available: {exc}")
    root.withdraw()
    yield root
    root.destroy()


IIDS = {"group": f"{_DISPATCH_GROUP_PREFIX}G", "node": f"{_DISPATCH_NODE_PREFIX}0_0_n"}


@pytest.mark.parametrize("kind", ["group", "node"])
def test_dispatch_view_keeps_the_dispatch(tk_root, tmp_path, kind):
    fake = _FakeViewer(tk_root, tmp_path, _DISPATCH_VIEWS[0])
    ResultViewer._display_dispatch_iid(fake, SCEN, IIDS[kind])
    assert fake.dispatch_calls == [(kind, "G" if kind == "group" else "n")]
    assert fake._plot_canvas.figure is None


@pytest.mark.parametrize("kind", ["group", "node"])
def test_ramp_envelope_view(tk_root, tmp_path, kind):
    """The envelope over the window: the net-load line 35, gap, −50, gap; the time range is the 4 rows."""
    fake = _FakeViewer(tk_root, tmp_path, "Ramp envelope")
    ResultViewer._display_dispatch_iid(fake, SCEN, IIDS[kind])
    assert fake._plot_canvas.message is None and not fake.dispatch_calls
    assert fake.time_range == 4
    ax = fake._plot_canvas.figure.axes[0]
    net = next(ln for ln in ax.get_lines() if ln.get_label() == "Net load ramp").get_ydata()
    assert net[0] == 35.0 and math.isnan(net[1]) and net[2] == -50.0
    assert "change toward the next step" in ax.get_ylabel()


@pytest.mark.parametrize("kind", ["group", "node"])
def test_ramp_frequency_view(tk_root, tmp_path, kind):
    """Case 6 entity A: 21 bins of 5 MW/h over ±52.5; room lines at +88 and −82 MW/h."""
    fake = _FakeViewer(tk_root, tmp_path, "Ramp frequency")
    ResultViewer._display_dispatch_iid(fake, SCEN, IIDS[kind])
    assert fake._plot_canvas.message is None
    ax = fake._plot_canvas.figure.axes[0]
    steps = [p for p in ax.patches if p.get_label() in ("net load", "demand")]
    assert len(steps) == 2
    verticals = sorted(float(np.asarray(ln.get_xdata())[0]) for ln in ax.get_lines())
    assert verticals == [-82.0, 88.0]
    assert ax.get_xlabel() == "Ramp rate toward the next step (MW/h)"


def test_missing_envelope_shows_a_message(tk_root, tmp_path):
    fake = _FakeViewer(tk_root, tmp_path, "Ramp envelope")
    fake._dispatch_results.node_ramp_envelope_dt_ep = None
    ResultViewer._display_dispatch_iid(fake, SCEN, IIDS["node"])
    assert fake._plot_canvas.figure is None
    assert "run the scenario again" in fake._plot_canvas.message
