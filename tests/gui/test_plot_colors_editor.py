"""Tests for the Stage-2 "Change colors" feature.

Covers:
* ``ResultViewer._on_change_colors`` seeds a project ``plot_settings.yaml``
  when absent, never overwrites an existing one, edits only the project
  copy (never the bundled package file), and on save clears the
  color-template cache and re-renders.
* ``PlotColorsEditor`` validates YAML and refuses to save broken syntax.

All Tk widgets are constructed under a withdrawn root; run headless via
``xvfb-run -a``.
"""

from __future__ import annotations

import tkinter as tk
import types
from pathlib import Path

import pytest

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
    return package_data_path("schemas/default_colors.yaml")


# ---------------------------------------------------------------------------
#  PlotColorsEditor — YAML validation
# ---------------------------------------------------------------------------


class TestPlotColorsEditorValidation:
    def test_valid_yaml_saves_and_sets_saved(self, tk_root, tmp_path):
        from flextool.gui.dialogs.plot_colors_editor import PlotColorsEditor

        f = tmp_path / "plot_settings.yaml"
        f.write_text("category:\n  costs:\n    a: '#111111'\n", encoding="utf-8")

        captured = {}

        def drive():
            ed = captured["editor"]
            ed._text.delete("1.0", "end")
            ed._text.insert("1.0", "category:\n  costs:\n    a: '#abcdef'\n")
            ed._on_save()

        # PlotColorsEditor blocks on wait_window in __init__, so we must
        # mutate+save from an after() callback. Stash the instance via a
        # subclass that records itself before entering the modal loop.
        class _Probe(PlotColorsEditor):
            def __init__(self, parent, path):
                captured["editor"] = self
                parent.after(0, drive)
                super().__init__(parent, path)

        ed = _Probe(tk_root, f)
        assert ed.saved is True
        assert "#abcdef" in f.read_text(encoding="utf-8")

    def test_invalid_yaml_refused(self, tk_root, tmp_path, monkeypatch):
        from flextool.gui.dialogs import plot_colors_editor
        from flextool.gui.dialogs.plot_colors_editor import PlotColorsEditor

        f = tmp_path / "plot_settings.yaml"
        original = "category:\n  costs:\n    a: '#111111'\n"
        f.write_text(original, encoding="utf-8")

        # Swallow the error dialog (no live display interaction).
        errors = []
        monkeypatch.setattr(
            plot_colors_editor.messagebox, "showerror",
            lambda *a, **k: errors.append((a, k)),
        )

        captured = {}

        def drive():
            ed = captured["editor"]
            ed._text.delete("1.0", "end")
            # Broken YAML: unbalanced brackets / bad indentation mapping.
            ed._text.insert("1.0", "category: [unterminated\n  : :\n")
            ed._on_save()
            # Save must be refused → still open; close it ourselves.
            ed._on_cancel()

        class _Probe(PlotColorsEditor):
            def __init__(self, parent, path):
                captured["editor"] = self
                parent.after(0, drive)
                super().__init__(parent, path)

        ed = _Probe(tk_root, f)
        assert ed.saved is False
        assert errors, "expected an Invalid-YAML error dialog"
        # File on disk is untouched.
        assert f.read_text(encoding="utf-8") == original


# ---------------------------------------------------------------------------
#  _on_change_colors — seeding + cache-clear/re-render
# ---------------------------------------------------------------------------


def _make_stub_viewer(project_path: Path):
    """A minimal stand-in carrying just what ``_on_change_colors`` touches.

    Avoids constructing the full (heavy) ResultViewer while exercising the
    real unbound method.
    """
    from flextool.gui.result_viewer import ResultViewer

    stub = types.SimpleNamespace()
    stub._project_path = project_path
    stub.calls = []
    stub._clear_figure_cache = lambda: stub.calls.append("clear_figure_cache")
    stub._trigger_replot = lambda: stub.calls.append("trigger_replot")
    # Bind the real (unbound) method to the stub.
    stub._on_change_colors = types.MethodType(
        ResultViewer._on_change_colors, stub,
    )
    return stub


class TestOnChangeColorsSeeding:
    def test_seeds_project_file_when_absent(self, tk_root, tmp_path, monkeypatch):
        import flextool.gui.result_viewer as rv

        project = tmp_path / "proj"
        project.mkdir()
        assert not (project / "plot_settings.yaml").exists()

        # Patch the editor to a no-op that reports "not saved" so we only
        # test the seeding branch (no cache clear / replot expected).
        opened = {}

        class _FakeEditor:
            def __init__(self, parent, path):
                opened["path"] = Path(path)
                self.saved = False

        monkeypatch.setattr(
            "flextool.gui.dialogs.plot_colors_editor.PlotColorsEditor",
            _FakeEditor,
        )

        stub = _make_stub_viewer(project)
        stub._on_change_colors()

        seeded = project / "plot_settings.yaml"
        assert seeded.is_file(), "project plot_settings.yaml must be seeded"
        # Seeded from the bundled default (byte-identical copy).
        assert seeded.read_bytes() == _bundled_default().read_bytes()
        # Editor was opened on the PROJECT copy, never the bundled file.
        assert opened["path"] == seeded
        assert opened["path"] != _bundled_default()
        # Not saved → no re-render.
        assert stub.calls == []
        assert rv is not None  # import sanity

    def test_does_not_overwrite_existing_file(self, tk_root, tmp_path, monkeypatch):
        project = tmp_path / "proj"
        project.mkdir()
        existing = project / "plot_settings.yaml"
        custom = "category:\n  costs:\n    mine: '#010203'\n"
        existing.write_text(custom, encoding="utf-8")

        class _FakeEditor:
            def __init__(self, parent, path):
                self.saved = False

        monkeypatch.setattr(
            "flextool.gui.dialogs.plot_colors_editor.PlotColorsEditor",
            _FakeEditor,
        )

        stub = _make_stub_viewer(project)
        stub._on_change_colors()

        # User content preserved (not clobbered by the bundled default).
        assert existing.read_text(encoding="utf-8") == custom

    def test_save_clears_cache_and_rerenders(self, tk_root, tmp_path, monkeypatch):
        from flextool.plot_outputs import color_template

        project = tmp_path / "proj"
        project.mkdir()

        class _FakeEditor:
            def __init__(self, parent, path):
                self.saved = True

        monkeypatch.setattr(
            "flextool.gui.dialogs.plot_colors_editor.PlotColorsEditor",
            _FakeEditor,
        )

        cleared = []
        monkeypatch.setattr(
            color_template, "_clear_cache",
            lambda: cleared.append(True),
        )

        stub = _make_stub_viewer(project)
        stub._on_change_colors()

        # On save: template cache cleared, figures invalidated, replot fired.
        assert cleared == [True]
        assert stub.calls == ["clear_figure_cache", "trigger_replot"]
