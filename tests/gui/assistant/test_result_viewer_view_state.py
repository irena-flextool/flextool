"""The result viewer reports its view to the AI assistant (P2 §13.2 test 11, §6.3).

``ResultViewer.view_state()`` runs against a light stand-in with the viewer's real widgets (mode, scenario list,
plot tree, comparison tree, time range): single mode with a plotted series at hand, comparison mode, dispatch mode;
a duration box that cannot be read gives ``time_steps = None`` and leaves the other fields; a destroyed viewer
gives ``None``.  The values the viewer shows are set here; the expected state is written out by hand.

Run headless:  xvfb-run -a python -m pytest tests/gui/assistant --run-gui
"""
from __future__ import annotations

import tkinter as tk
from types import SimpleNamespace
from tkinter import ttk

import pytest

from flextool.assistant.context import ViewerState
from flextool.gui.plot_config_reader import PlotEntry, PlotVariant
from flextool.gui.result_viewer import ResultViewer

pytestmark = pytest.mark.gui


class _FakeViewer(tk.Frame):
    """The parts of the viewer ``view_state`` reads, with the viewer's own methods."""

    view_state = ResultViewer.view_state
    _VIEW_READ_ERRORS = ResultViewer._VIEW_READ_ERRORS
    _COMP_CHECK_ON = ResultViewer._COMP_CHECK_ON
    _COMP_CHECK_OFF = ResultViewer._COMP_CHECK_OFF
    _get_selected_scenarios = ResultViewer._get_selected_scenarios
    _get_comparison_scenarios = ResultViewer._get_comparison_scenarios
    _get_active_variant = ResultViewer._get_active_variant
    _view_entry = ResultViewer._view_entry
    _view_dispatch_item = ResultViewer._view_dispatch_item
    _view_time_label = ResultViewer._view_time_label

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self._mode = tk.StringVar(self, value="single")
        self._scenario_listbox = tk.Listbox(self, selectmode="extended", exportselection=False)
        for name in ("base", "high_wind", "low_hydro"):
            self._scenario_listbox.insert("end", name)
        self._plot_tree = ttk.Treeview(self)
        self._comp_tree = ttk.Treeview(self, columns=("check", "name"), show="headings")
        for name, on in (("base", True), ("high_wind", False), ("low_hydro", True)):
            self._comp_tree.insert("", "end", values=(self._COMP_CHECK_ON if on else self._COMP_CHECK_OFF, name))
        self._tree_entry_map: dict[str, PlotEntry] = {}
        group = self._plot_tree.insert("", "end", iid="group_0", text="0")
        entry = PlotEntry(number="0.1", short_name="Loss of load", full_name="Loss of load (upward slack)",
                          variants=[PlotVariant("d", "node_slack_up_d_e", "default", "Loss of load, per period"),
                                    PlotVariant("t", "node_slack_up_dt_e", "default", "Loss of load, per time")])
        self._plot_tree.insert(group, "end", iid="entry_0.1", text="0.1")
        self._tree_entry_map["entry_0.1"] = entry
        self._active_variant = "t"
        self._start_var = tk.IntVar(self, value=24)
        self._duration_var = tk.IntVar(self, value=168)
        self._last_dispatch_iid: str | None = None
        self._dispatch_node_iids: dict[str, str] = {}
        self._live_plan = None
        self._live_plan_key: tuple = ("", "", "")


@pytest.fixture
def viewer(tk_root):
    fake = _FakeViewer(tk_root)
    yield fake
    try:
        fake.destroy()
    except tk.TclError:
        pass


def test_single_mode_with_the_plotted_series(viewer):
    viewer._scenario_listbox.selection_set(1)
    viewer._plot_tree.selection_set("entry_0.1")
    viewer._live_plan = SimpleNamespace(time_index_values=[f"t{i:04d}" for i in range(1, 200)],
                                        period_labels=["p2030"] * 199)
    viewer._live_plan_key = ("high_wind", "node_slack_up_dt_e", "default")
    viewer.update()
    assert viewer.view_state() == ViewerState(
        mode="single", scenarios=("high_wind",), plot_entry="0.1", plot_name="Loss of load (upward slack)",
        variant="t", result_key="node_slack_up_dt_e", sub_config="default", time_start=24, time_steps=168,
        time_start_label="p2030 t0025", dispatch_item=None, comparison_checked=("base", "low_hydro"))

    viewer._live_plan_key = ("base", "node_slack_up_dt_e", "default")     # a plan of another scenario
    assert viewer.view_state().time_start_label is None
    viewer._live_plan = None                                              # no series at hand: position only
    state = viewer.view_state()
    assert (state.time_start, state.time_steps, state.time_start_label) == (24, 168, None)


def test_comparison_mode(viewer):
    viewer._mode.set("comparison")
    viewer._plot_tree.selection_set("entry_0.1")
    viewer._active_variant = "d"
    viewer._live_plan = SimpleNamespace(time_index_values=["t0001", "t0002"], period_labels=None)
    viewer._live_plan_key = ("_comparison_merged", "node_slack_up_d_e", "default", ("base", "low_hydro"))
    viewer._start_var.set(1)
    viewer.update()
    assert viewer.view_state() == ViewerState(
        mode="comparison", scenarios=("base", "low_hydro"), plot_entry="0.1",
        plot_name="Loss of load (upward slack)", variant="d", result_key="node_slack_up_d_e", sub_config="default",
        time_start=1, time_steps=168, time_start_label="t0002", dispatch_item=None,
        comparison_checked=("base", "low_hydro"))


def test_dispatch_mode(viewer):
    viewer._mode.set("dispatch")
    viewer._scenario_listbox.selection_set(0)
    viewer._dispatch_node_iids["dispatchnode_0_1_north"] = "north"
    viewer._last_dispatch_iid = "dispatchnode_0_1_north"
    state = viewer.view_state()
    assert (state.mode, state.scenarios, state.plot_entry, state.dispatch_item, state.time_start_label) == (
        "dispatch", ("base",), None, "north", None)
    viewer._last_dispatch_iid = "dispatch_electricity"
    assert viewer.view_state().dispatch_item == "electricity"


def test_a_duration_being_edited_reads_as_none(viewer):
    viewer._plot_tree.selection_set("entry_0.1")
    viewer.setvar(str(viewer._duration_var), "")           # the box is empty while the user types
    state = viewer.view_state()
    assert state.time_steps is None
    assert (state.time_start, state.plot_entry, state.variant) == (24, "0.1", "t")


def test_a_destroyed_viewer_gives_none(tk_root):
    fake = _FakeViewer(tk_root)
    fake.destroy()
    assert fake.view_state() is None
