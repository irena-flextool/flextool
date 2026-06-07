"""Tests for the non-modal color/order picker (Stage 6.2).

Covers:
* ``PlotSettingsPicker`` builds a Notebook with one tab per present
  section, each tab's Treeview populated with the right names, and a
  composite swatch ``PhotoImage`` kept alive (not GC'd) per row.
* ``_write`` round-trips the working dict to byte-valid YAML; Apply writes
  + invokes ``on_apply``; Save-and-exit writes + invokes + closes; Cancel
  restores the on-open file text + invokes ``on_apply``.
* ``ResultViewer._on_change_colors`` seeds a project ``plot_settings.yaml``
  when absent, never overwrites an existing one, edits only the project
  copy (never the bundled package file), and opens the picker non-modally
  with ``_apply_color_settings`` as the ``on_apply`` callback.
* ``ResultViewer._apply_color_settings`` clears the cache and re-renders /
  rebuilds the color map (the reusable recolor body).
* The PNG settings ``PlotDialog`` opens the picker with NO ``on_apply``.

All Tk widgets are constructed under a withdrawn root; run headless via
``xvfb-run -a``.
"""

from __future__ import annotations

import tkinter as tk
import types
from pathlib import Path
from tkinter import ttk

import pytest
import yaml

try:
    import matplotlib
    matplotlib.use("Agg")
except Exception:
    pass


@pytest.fixture()
def tk_root():
    """Create a withdrawn Tk root; skip if no display is available."""
    try:
        root = tk.Tk()
        root.withdraw()
        yield root
        root.destroy()
    except tk.TclError:
        pytest.skip("No display available")


def _bundled_default() -> Path:
    from flextool._resources import package_data_path
    return package_data_path("schemas/default_plot_settings.yaml")


# ---------------------------------------------------------------------------
#  PlotSettingsPicker — tabs, trees, swatches
# ---------------------------------------------------------------------------


_SAMPLE = {
    "scenarios": {"S1": "#1f77b4", "S2": "#ff7f0e"},
    "categories": {
        "costs": {"co2": "#4d4d4d"},
        "dispatch": {"Charge": "lime"},  # named color
    },
    "entities": {
        "unit": {
            "coal": "#212121",
            "chp": {"color": "#E64A19", "neg_color": "#9c3010"},
        },
        "node": {"n1": "#4FC3F7"},
    },
}


def _make_picker(tk_root, tmp_path, data=None, on_apply=None):
    from flextool.gui.dialogs.plot_settings_picker import PlotSettingsPicker

    f = tmp_path / "plot_settings.yaml"
    f.write_text(
        yaml.safe_dump(data if data is not None else _SAMPLE, sort_keys=False),
        encoding="utf-8",
    )
    picker = PlotSettingsPicker(tk_root, f, on_apply=on_apply)
    return picker, f


def _tab_titles(picker) -> list[str]:
    nb = picker._notebook
    return [nb.tab(tid, "text") for tid in nb.tabs()]


def _tree_in_tab(picker, index):
    nb = picker._notebook
    frame = nb.nametowidget(nb.tabs()[index])
    for child in frame.winfo_children():
        if isinstance(child, ttk.Treeview):
            return child
    raise AssertionError("no Treeview in tab")


def _row_names(tree) -> list[str]:
    return [tree.item(iid, "text") for iid in tree.get_children("")]


class TestPickerBuild:
    def test_tabs_only_for_present_sections(self, tk_root, tmp_path):
        picker, _ = _make_picker(tk_root, tmp_path)
        # entities: unit, node (group/connection absent) → categories: costs,
        # dispatch (node_flows/nodegroup_flows absent) → scenarios.
        assert _tab_titles(picker) == [
            "unit", "node", "costs", "dispatch", "scenarios",
        ]

    def test_empty_sections_skipped(self, tk_root, tmp_path):
        data = {
            "scenarios": {},
            "categories": {"costs": {}},
            "entities": {"unit": {}, "node": {"n1": "#abcdef"}},
        }
        picker, _ = _make_picker(tk_root, tmp_path, data=data)
        # Only the non-empty node entities tab survives.
        assert _tab_titles(picker) == ["node"]

    def test_rows_populated_with_names(self, tk_root, tmp_path):
        picker, _ = _make_picker(tk_root, tmp_path)
        titles = _tab_titles(picker)
        unit = _tree_in_tab(picker, titles.index("unit"))
        assert _row_names(unit) == ["coal", "chp"]
        scen = _tree_in_tab(picker, titles.index("scenarios"))
        assert _row_names(scen) == ["S1", "S2"]
        costs = _tree_in_tab(picker, titles.index("costs"))
        assert _row_names(costs) == ["co2"]

    def test_swatches_created_and_referenced(self, tk_root, tmp_path):
        picker, _ = _make_picker(tk_root, tmp_path)
        # One swatch per row: 2 unit + 1 node + 1 costs + 1 dispatch + 2 scen.
        assert len(picker._swatches) == 7
        # Every swatch is a live PhotoImage (not GC'd).
        for img in picker._swatches:
            assert isinstance(img, tk.PhotoImage)
            assert img.width() > 0 and img.height() > 0

    def test_composite_vs_single_swatch_width(self, tk_root, tmp_path):
        """Entity with neg_color → double-width composite; bare entity →
        single box."""
        from flextool.gui.dialogs.plot_settings_picker import _SWATCH_W

        picker, _ = _make_picker(tk_root, tmp_path)
        titles = _tab_titles(picker)
        unit = _tree_in_tab(picker, titles.index("unit"))
        coal_iid, chp_iid = unit.get_children("")

        def _img_width(iid) -> int:
            name = unit.item(iid, "image")
            if isinstance(name, (list, tuple)):
                name = name[0]
            return int(tk_root.tk.call(name, "cget", "-width"))

        assert _img_width(coal_iid) == _SWATCH_W       # bare → single box
        assert _img_width(chp_iid) == _SWATCH_W * 2    # neg_color → composite

    def test_non_modal_no_grab(self, tk_root, tmp_path):
        """The picker must not grab input (usable alongside the viewer)."""
        picker, _ = _make_picker(tk_root, tmp_path)
        # The picker must not be the current grab.
        assert picker.grab_current() in (None, "")


# ---------------------------------------------------------------------------
#  PlotSettingsPicker — Apply / Save / Cancel + on_apply wiring
# ---------------------------------------------------------------------------


class TestPickerButtons:
    def test_apply_writes_roundtrip_and_calls_on_apply(self, tk_root, tmp_path):
        calls = []
        picker, f = _make_picker(
            tk_root, tmp_path, on_apply=lambda: calls.append(1),
        )
        picker._on_apply_clicked()
        # on_apply fired; window stayed open.
        assert calls == [1]
        assert picker.winfo_exists()
        # File round-trips equal to the working dict.
        assert yaml.safe_load(f.read_text(encoding="utf-8")) == picker._data
        assert picker._data == _SAMPLE

    def test_save_and_exit_writes_and_closes(self, tk_root, tmp_path):
        calls = []
        picker, f = _make_picker(
            tk_root, tmp_path, on_apply=lambda: calls.append(1),
        )
        picker._on_save_exit()
        assert calls == [1]
        assert not picker.winfo_exists()
        assert yaml.safe_load(f.read_text(encoding="utf-8")) == _SAMPLE

    def test_cancel_restores_original_and_calls_on_apply(self, tk_root, tmp_path):
        calls = []
        picker, f = _make_picker(
            tk_root, tmp_path, on_apply=lambda: calls.append(1),
        )
        original = f.read_text(encoding="utf-8")
        # Simulate a prior Apply that changed the file on disk.
        f.write_text("scenarios:\n  X: '#000000'\n", encoding="utf-8")
        picker._on_cancel()
        # Original on-open text restored byte-for-byte; on_apply (revert) fired.
        assert f.read_text(encoding="utf-8") == original
        assert calls == [1]
        assert not picker.winfo_exists()

    def test_no_on_apply_is_fine(self, tk_root, tmp_path):
        """Picker opened with no callback (PNG dialog) just writes."""
        picker, f = _make_picker(tk_root, tmp_path, on_apply=None)
        picker._on_apply_clicked()  # must not raise
        assert yaml.safe_load(f.read_text(encoding="utf-8")) == _SAMPLE


# ---------------------------------------------------------------------------
#  PlotDialog — shared "Colors, order..." button opens the picker
# ---------------------------------------------------------------------------


def _iter_buttons(widget):
    for child in widget.winfo_children():
        if isinstance(child, ttk.Button):
            yield child
        yield from _iter_buttons(child)


class TestPlotDialogColorsButton:
    def test_button_seeds_and_opens_picker_no_preview(
        self, tk_root, tmp_path, monkeypatch,
    ):
        """The dialog-level "Colors, order..." button seeds the project
        ``plot_settings.yaml`` and opens the picker on that project copy
        with NO ``on_apply`` (the batch dialog has no live preview)."""
        from flextool.gui.dialogs.plot_dialog import PlotDialog
        from flextool.gui.data_models import ProjectSettings

        project = tmp_path / "proj"
        project.mkdir()
        assert not (project / "plot_settings.yaml").exists()

        opened = {}

        class _FakePicker:
            def __init__(self, parent, path, on_apply=None):
                opened["parent"] = parent
                opened["path"] = Path(path)
                opened["on_apply"] = on_apply

        monkeypatch.setattr(
            "flextool.gui.dialogs.plot_settings_picker.PlotSettingsPicker",
            _FakePicker,
        )

        captured = {}

        def drive():
            dlg = captured["dialog"]
            buttons = [
                b for b in _iter_buttons(dlg)
                if str(b.cget("text")) == "Colors, order..."
            ]
            captured["button_count"] = len(buttons)
            if buttons:
                buttons[0].invoke()
            dlg._on_ok()

        class _Probe(PlotDialog):
            def __init__(self, parent, project_path, settings):
                captured["dialog"] = self
                parent.after(0, drive)
                super().__init__(parent, project_path, settings)

        _Probe(tk_root, project, ProjectSettings())

        assert captured["button_count"] == 1
        seeded = project / "plot_settings.yaml"
        assert seeded.is_file()
        assert seeded.read_bytes() == _bundled_default().read_bytes()
        assert opened["path"] == seeded
        # PNG batch dialog → no live preview.
        assert opened["on_apply"] is None

    def test_dispatch_config_editor_is_gone(self):
        """The old ``DispatchConfigEditor`` and its handler are removed."""
        from flextool.gui.dialogs import plot_dialog

        assert not hasattr(plot_dialog, "DispatchConfigEditor")
        assert not hasattr(plot_dialog._PlotSection, "_on_edit_dispatch_config")


# ---------------------------------------------------------------------------
#  _on_change_colors — seeding + opens picker non-modally with on_apply
# ---------------------------------------------------------------------------


def _make_stub_viewer(project_path: Path, live_plan=None):
    """A minimal stand-in carrying just what ``_on_change_colors`` and
    ``_apply_color_settings`` touch.  Binds the real unbound methods."""
    from flextool.gui.result_viewer import ResultViewer

    stub = types.SimpleNamespace()
    stub._project_path = project_path
    stub._live_plan = live_plan
    stub.calls = []
    stub._clear_figure_cache = lambda: stub.calls.append("clear_figure_cache")
    stub._clear_prefetched_figures = lambda: stub.calls.append(
        "clear_prefetched_figures"
    )
    stub._trigger_replot = lambda: stub.calls.append("trigger_replot")
    stub._on_change_colors = types.MethodType(
        ResultViewer._on_change_colors, stub,
    )
    stub._apply_color_settings = types.MethodType(
        ResultViewer._apply_color_settings, stub,
    )
    return stub


class TestOnChangeColorsSeeding:
    def test_seeds_project_file_and_opens_picker_with_callback(
        self, tk_root, tmp_path, monkeypatch,
    ):
        import flextool.gui.result_viewer as rv

        project = tmp_path / "proj"
        project.mkdir()
        assert not (project / "plot_settings.yaml").exists()

        opened = {}

        class _FakePicker:
            def __init__(self, parent, path, on_apply=None):
                opened["path"] = Path(path)
                opened["on_apply"] = on_apply

        monkeypatch.setattr(
            "flextool.gui.dialogs.plot_settings_picker.PlotSettingsPicker",
            _FakePicker,
        )

        stub = _make_stub_viewer(project)
        stub._on_change_colors()

        seeded = project / "plot_settings.yaml"
        assert seeded.is_file(), "project plot_settings.yaml must be seeded"
        assert seeded.read_bytes() == _bundled_default().read_bytes()
        assert opened["path"] == seeded
        assert opened["path"] != _bundled_default()
        # Picker gets the viewer's reusable recolor body as on_apply.
        assert opened["on_apply"] == stub._apply_color_settings
        # Opening alone does not recolor (that happens on Apply).
        assert stub.calls == []
        assert rv is not None

    def test_does_not_overwrite_existing_file(self, tk_root, tmp_path, monkeypatch):
        project = tmp_path / "proj"
        project.mkdir()
        existing = project / "plot_settings.yaml"
        custom = "categories:\n  costs:\n    mine: '#010203'\n"
        existing.write_text(custom, encoding="utf-8")

        class _FakePicker:
            def __init__(self, parent, path, on_apply=None):
                pass

        monkeypatch.setattr(
            "flextool.gui.dialogs.plot_settings_picker.PlotSettingsPicker",
            _FakePicker,
        )

        stub = _make_stub_viewer(project)
        stub._on_change_colors()

        assert existing.read_text(encoding="utf-8") == custom


class TestApplyColorSettings:
    def test_clears_cache_and_rerenders_when_no_live_plan(
        self, tk_root, tmp_path, monkeypatch,
    ):
        from flextool.plot_outputs import color_template

        project = tmp_path / "proj"
        project.mkdir()
        (project / "plot_settings.yaml").write_text(
            "entities:\n  node:\n    n1: '#abcdef'\n", encoding="utf-8",
        )

        cleared = []
        monkeypatch.setattr(
            color_template, "_clear_cache", lambda: cleared.append(True),
        )

        # No live plan cached → full clear + recompute fallback.
        stub = _make_stub_viewer(project, live_plan=None)
        stub._apply_color_settings()

        assert cleared == [True]
        assert stub.calls == ["clear_figure_cache", "trigger_replot"]

    def test_rebuilds_color_map_in_place(self, tk_root, tmp_path, monkeypatch):
        """A cached live plan with hints recolors IN PLACE: plan identity
        preserved, only prefetched figures dropped."""
        from flextool.plot_outputs import color_template
        from flextool.plot_outputs.plan import PlotPlan
        import pandas as pd

        project = tmp_path / "proj"
        project.mkdir()
        (project / "plot_settings.yaml").write_text(
            "entities:\n  unit:\n    coal: '#00ff00'\n", encoding="utf-8",
        )
        color_template._clear_cache()

        plan = PlotPlan(
            chart_type='stack',
            plot_name='p',
            total_file_count=1,
            processed_df=pd.DataFrame({'coal': [1.0]}),
            effective_plot_specs=[(None, ['coal'])],
            file_batches=[[0]],
            shared_color_map={'coal': (1.0, 0.0, 0.0)},  # old red
            color_entity_class='unit',
        )

        stub = _make_stub_viewer(project, live_plan=plan)
        stub._apply_color_settings()

        assert stub._live_plan is plan
        assert plan.shared_color_map == {'coal': (0.0, 1.0, 0.0)}
        assert stub.calls == ["clear_prefetched_figures", "trigger_replot"]
