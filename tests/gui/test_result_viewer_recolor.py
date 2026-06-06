"""Regression test for the result-viewer recolor-on-save flow.

Reproduces and guards the bug where editing an entity color in the
"Colors, order..." editor and saving did NOT recolor the currently
displayed single-mode plot, because the viewer was showing a *legacy*
on-disk plot-plan (written before plans carried
``color_category`` / ``color_entity_class`` hints — these deserialize as
``None``).  An in-place ``rebuild_plan_color_map`` on such a plan ignores
the edited ``plot_settings.yaml`` ``entities`` section and silently
reverts to palette colors, so the displayed colors never change.

The fix forces a from-scratch plan recompute (bypassing the stale disk
plan) when the cached/disk plan has no usable color hints — see
``ResultViewer._on_change_colors`` / ``_force_plan_recompute``.

This test drives the REAL ``ResultViewer._display_from_parquet`` and
``_on_change_colors`` method bodies against on-disk parquet + plan
fixtures, using a lightweight stand-in for the heavy parts of the viewer
(the same approach as ``test_result_viewer_dispatch_tree``).  It captures
the actual displayed figure's fill colors before and after the edit.

Must run under ``xvfb-run -a`` (creates a real Tk root).
"""

import threading
import tkinter as tk

import matplotlib
matplotlib.use("Agg")

import json
import numpy as np
import pandas as pd
import pytest
import yaml

from flextool.gui.result_viewer import ResultViewer

SCEN = "S1"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"No display available: {exc}")
    root.withdraw()
    yield root
    root.destroy()


class _FakeVar:
    def __init__(self, v):
        self._v = v

    def get(self):
        return self._v

    def set(self, v):
        self._v = v


class _FakeCanvas:
    def __init__(self):
        self.figure = None
        self.message = None

    def display_figure(self, fig):
        self.figure = fig

    def show_message(self, text):
        self.message = text


class _SyncExecutor:
    """Run submitted work inline so the (single-threaded) test stays on the
    Tk main thread — avoids the cross-thread ``after`` fragility while still
    exercising the real ``_build_figure_from_plan_async`` body."""

    def submit(self, fn, *a, **k):
        fn(*a, **k)


class _RecolorViewer(ResultViewer):
    """Subclass that skips the heavy ``__init__`` but reuses the real
    ``_display_from_parquet`` / ``_on_change_colors`` / async-build bodies."""

    def __init__(self, root, project_path, plot_cfg, settings_path):
        self._root = root
        self._project_path = project_path
        self._plot_cfg = plot_cfg
        self._settings_path = settings_path
        self._plot_canvas = _FakeCanvas()
        self._executor = _SyncExecutor()
        self._render_gen = 0
        self._live_plan = None
        self._live_plan_key = ("", "", "")
        self._force_plan_recompute = False
        self._figure_cache = {}
        self._figure_cache_lock = threading.Lock()
        self._parquet_cache_key = ("", "")
        self._parquet_cache_df = None
        self._break_times_cache = {}
        self._start_var = _FakeVar(0)
        self._duration_var = _FakeVar(24)
        self._file_index = 0
        self._file_count = 1

    # Single-mode replot dispatch (mirrors the real _trigger_replot path).
    _replot_target = None

    def _trigger_replot(self):
        scen, entry, variant = self._replot_target
        self._display_from_parquet(scen, entry, variant)

    # With the synchronous executor everything runs on the main thread, so
    # after(0, cb) fires the callback inline exactly as Tk eventually would.
    def after(self, ms, func=None, *args):
        if func is None:
            return None
        func(*args)
        return None

    # Real PlotConfig from our in-memory config dict.
    def _load_plot_config(self, result_key, sub_config):
        from flextool.plot_outputs.config import PlotConfig

        raw = dict(self._plot_cfg[result_key])
        name = raw.pop("plot_name", None)
        return PlotConfig(plot_name=name, **raw)

    # No-op heavy helpers not under test.
    def _apply_axis_manifest(self, plan, result_key, sub_config):
        return None

    def _update_time_range(self, n):
        return None

    def _update_file_nav(self):
        return None

    def _schedule_placeholder(self, generation, text):
        return None

    def _cancel_placeholder(self):
        return None

    def _prefetch_adjacent(self, *a, **k):
        return None

    def _get_axis_manifest(self):
        return None

    def _get_axis_active_scenarios(self):
        return None


class _Variant:
    result_key = "node_state_dt_e"
    sub_config = "default"
    full_name = "Node state"


def _write_template(path, entities_node):
    path.write_text(
        yaml.safe_dump({"entities": {"node": entities_node}}, sort_keys=False)
    )


def _collect_fill_colors(fig):
    """Rounded RGBA face colors of stacked-area collections + bar patches."""
    colors = set()
    for ax in fig.get_axes():
        for coll in ax.collections:
            try:
                fc = coll.get_facecolor()
            except Exception:  # noqa: BLE001
                continue
            for c in np.atleast_2d(fc):
                colors.add(tuple(round(float(x), 4) for x in c))
        for patch in ax.patches:
            try:
                colors.add(tuple(round(float(x), 4) for x in patch.get_facecolor()))
            except Exception:  # noqa: BLE001
                pass
    return colors


def _make_project(tmp_path, *, legacy_plan, empty_initial):
    """Build a project dir with a node-state parquet + a pre-computed disk
    plan.  ``legacy_plan`` strips the color hints from the saved plan json
    (simulating a pre-stage-3.4 on-disk plan).  ``empty_initial`` leaves the
    project ``plot_settings.yaml`` with no entity colors at compute time."""
    from flextool.plot_outputs import color_template as ct
    from flextool.plot_outputs.plan import compute_plot_plans_for_result

    proj = tmp_path / "proj"
    pdir = proj / "output_parquet" / SCEN
    pdir.mkdir(parents=True)

    nodes = ("node_a", "node_b", "node_c")
    rng = np.random.default_rng(1)
    index = pd.Index(range(24), name="time")
    df = pd.DataFrame(
        rng.random((24, len(nodes))) * 50.0,
        index=index,
        columns=pd.MultiIndex.from_arrays([list(nodes)], names=["node"]),
    )

    # Parquet as the viewer reads it: 'scenario' as the top column level.
    disk_df = df.copy()
    disk_df.columns = pd.MultiIndex.from_tuples(
        [(SCEN, n) for n in nodes], names=["scenario", "node"]
    )
    disk_df.to_parquet(pdir / "node_state_dt_e.parquet")

    settings_path = proj / "plot_settings.yaml"
    if empty_initial:
        _write_template(settings_path, {})
    else:
        _write_template(
            settings_path,
            {"node_a": "#ff0000", "node_b": "#00ff00", "node_c": "#0000ff"},
        )

    plot_cfg = {
        "node_state_dt_e": {
            "plot_name": "Node state",
            "map_dimensions_for_plots": ["t_e", "t_s"],
            "legend": "shared",
            "color_entity_class": "node",
        }
    }

    ct._clear_cache()
    plan_dir = pdir / "plot_plans"
    compute_plot_plans_for_result(
        df, "node_state_dt_e", plot_cfg, plan_dir,
        plot_rows=(0, 24), color_path=settings_path,
    )
    assert (plan_dir / "node_state_dt_e__default_plan.json").is_file()

    if legacy_plan:
        jp = plan_dir / "node_state_dt_e__default_plan.json"
        meta = json.loads(jp.read_text())
        meta.pop("color_category", None)
        meta.pop("color_entity_class", None)
        jp.write_text(json.dumps(meta))

    return proj, settings_path, plot_cfg


def _simulate_edit_and_save(viewer, settings_path, new_node_colors, monkeypatch):
    """Edit the project template, then invoke the REAL _on_change_colors
    post-save logic with the modal editor stubbed (saved=True)."""
    _write_template(settings_path, new_node_colors)

    import flextool.gui.dialogs.plot_settings_editor as pse_mod
    import flextool.gui.project_utils as pu_mod

    class _FakeEditor:
        def __init__(self, parent, path):
            self.saved = True

    monkeypatch.setattr(pse_mod, "PlotSettingsEditor", _FakeEditor)
    monkeypatch.setattr(pu_mod, "seed_plot_settings", lambda p: settings_path)
    viewer._on_change_colors()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# Edited colors → rounded RGB the figure should display.
_EDITED = {
    "node_a": "#202020",  # 0.1255
    "node_b": "#303030",  # 0.1882
    "node_c": "#404040",  # 0.2510
}
_EDITED_RGB = {
    (0.1255, 0.1255, 0.1255, 1.0),
    (0.1882, 0.1882, 0.1882, 1.0),
    (0.251, 0.251, 0.251, 1.0),
}


@pytest.mark.parametrize(
    "legacy_plan,empty_initial",
    [
        (False, False),  # fresh hinted disk plan (in-place rebuild path)
        (True, False),   # legacy disk plan, colors set at compute time
        (True, True),    # legacy disk plan, no initial colors — user symptom
    ],
)
def test_recolor_updates_displayed_plot(
    tk_root, tmp_path, monkeypatch, legacy_plan, empty_initial
):
    proj, settings_path, plot_cfg = _make_project(
        tmp_path, legacy_plan=legacy_plan, empty_initial=empty_initial
    )

    viewer = _RecolorViewer(tk_root, proj, plot_cfg, settings_path)
    variant = _Variant()
    viewer._replot_target = (SCEN, None, variant)

    # Initial render.
    viewer._display_from_parquet(SCEN, None, variant)
    assert viewer._plot_canvas.figure is not None, viewer._plot_canvas.message

    # Edit colors to a distinctive set and "save".
    _simulate_edit_and_save(viewer, settings_path, _EDITED, monkeypatch)
    assert viewer._plot_canvas.figure is not None, viewer._plot_canvas.message

    after = _collect_fill_colors(viewer._plot_canvas.figure)

    # The displayed figure must now contain exactly the EDITED entity colors.
    assert _EDITED_RGB.issubset(after), (
        f"displayed colors did not update to the edit "
        f"(legacy_plan={legacy_plan}, empty_initial={empty_initial}): {sorted(after)}"
    )
    # And the live plan now carries the edited colors (so subsequent
    # in-place rebuilds work too).
    assert viewer._live_plan is not None
    cm = viewer._live_plan.shared_color_map
    assert cm is not None
    rounded = {tuple(round(float(x), 4) for x in v) for v in cm.values()}
    assert rounded == {
        (0.1255, 0.1255, 0.1255),
        (0.1882, 0.1882, 0.1882),
        (0.251, 0.251, 0.251),
    }
