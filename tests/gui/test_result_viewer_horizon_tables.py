"""The result viewer draws the node-group horizon tables (``nodeGroup_g_p``, ``nodeGroup_VRE_share_g``) in comparison
mode with their 'sum_periods' configs from the shipped ``default_plots.yaml``.

Drives the real ``ResultViewer._display_comparison`` / ``_display_comparison_from_single`` bodies on a stand-in for
the rest of the viewer (as ``test_result_viewer_recolor``), against per-scenario parquets written as
``write_outputs`` writes them.  Must run under ``xvfb-run -a`` (creates a real Tk root).
"""
from __future__ import annotations

import tkinter as tk

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from flextool._resources import package_data_path  # noqa: E402
from flextool.gui.result_viewer import ResultViewer  # noqa: E402
from flextool.lean_parquet import write_lean_parquet  # noqa: E402

pytestmark = pytest.mark.gui

SCENARIOS = ("A", "B")


@pytest.fixture()
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"No display available: {exc}")
    root.withdraw()
    yield root
    root.destroy()


class _Var:
    def __init__(self, v):
        self._v = v

    def get(self):
        return self._v


class _Canvas:
    def __init__(self):
        self.figure = None
        self.message = None

    def display_figure(self, fig):
        self.figure = fig

    def show_message(self, text):
        self.message = text


class _Viewer(ResultViewer):
    """Skips the heavy ``__init__``; the comparison display bodies are the real ones."""

    def __init__(self, root, project_path):
        self._root = root
        self._project_path = project_path
        self._single_config_path = package_data_path("schemas/default_plots.yaml")
        self._yaml_cache = {}
        self._plot_canvas = _Canvas()
        self._live_plan = None
        self._live_plan_key = None
        self._break_times_cache = {}
        self._comparison_break_cache = {}
        self._start_var = _Var(0)
        self._duration_var = _Var(24)
        self._file_index = 0
        self._file_count = 1

    def _get_comparison_viewer_scenarios(self):
        return list(SCENARIOS)

    def _get_comparison_scenarios(self):
        return list(SCENARIOS)

    def _load_comparison_break_times(self, viewer_scenarios=None):
        return None

    def _apply_axis_manifest(self, plan, result_key, sub_config):
        return None

    def _update_time_range(self, n):
        return None

    def _update_file_nav(self):
        return None


class _Variant:
    def __init__(self, result_key):
        self.result_key = result_key
        self.sub_config = "sum_periods"
        self.full_name = result_key


def _write_project(project, share_by_scenario):
    for scenario, share in share_by_scenario.items():
        d = project / "output_parquet" / scenario
        d.mkdir(parents=True)
        g_p = pd.DataFrame(
            {"1. Loss of load share": [0.05], "2. VRE share of demand": [share], "3. Excess load share": [0.0],
             "4. Curtailed VRE of demand": [0.07], "5. Inflow (horizon total)": [5.0e6],
             "6. Curtailed VRE of potential VRE": [0.07]},
            index=pd.Index(["electricity"], name="group"))
        g_p.columns.name = "parameter"
        vre = pd.DataFrame({"electricity": [share]}, index=pd.Index(["total"], name="horizon"))
        vre.columns.name = "group"
        for name, df in (("nodeGroup_g_p", g_p), ("nodeGroup_VRE_share_g", vre)):
            write_lean_parquet(pd.concat({scenario: df}, axis=1, names=["scenario"]), d / f"{name}.parquet")


@pytest.mark.parametrize("result_key,n_axes,first_xlabel", [
    ("nodeGroup_g_p", 6, "percent (horizon total, energy ratio)"),
    ("nodeGroup_VRE_share_g", 1, "percent (horizon total, energy ratio)"),
])
def test_viewer_draws_horizon_tables_in_comparison(tk_root, tmp_path, result_key, n_axes, first_xlabel):
    _write_project(tmp_path, {"A": 0.93, "B": 0.95})
    viewer = _Viewer(tk_root, tmp_path)
    viewer._display_comparison(None, _Variant(result_key))
    fig = viewer._plot_canvas.figure
    assert fig is not None, viewer._plot_canvas.message
    axes = [ax for ax in fig.get_axes() if ax.has_data()]
    assert len(axes) == n_axes
    assert axes[0].get_xlabel() == first_xlabel
    # The subplot of the MWh inflow keeps its own label (the YAML list is one label per subplot).
    if result_key == "nodeGroup_g_p":
        assert axes[4].get_xlabel() == "MWh (horizon total)"
