"""The project export and import dialogs (C-9): tick a run, the total follows, Export writes the package; Import
reads its manifest and suggests ``<Name>_2`` while ``<Name>`` is taken.

Run headless:  xvfb-run -a python -m pytest tests/gui/test_project_package.py --run-gui
"""
from __future__ import annotations

import time
import tkinter as tk
from pathlib import Path

import pytest

from flextool.gui.dialogs.project_package_dialog import ExportDialog, ImportDialog, format_size
from tests.test_project_package_cli import db_template, workspace  # noqa: F401  (fixtures)

pytestmark = pytest.mark.gui


@pytest.fixture()
def root():
    try:
        window = tk.Tk()
    except tk.TclError:
        pytest.skip("No display available")
    window.withdraw()
    yield window
    window.destroy()


def _wait(root: tk.Tk, done, timeout: float = 60.0) -> None:
    end = time.monotonic() + timeout
    while not done():
        assert time.monotonic() < end, "the dialog did not finish"
        root.update()
        time.sleep(0.02)


def test_export_then_import_suggests_free_name(root: tk.Tk, workspace: dict) -> None:  # noqa: F811
    busy = [True]
    out_dir: Path = workspace["root"] / "out"
    dlg = ExportDialog(root, workspace["project"], jobs_running=lambda: busy[0],
                       solver_config_dir=workspace["config"], target_dir=out_dir)
    try:
        assert "0 of 2 result runs" in dlg.total_label.cget("text")
        base = dlg.total_bytes()
        dlg.checks.set_checked("A_1", True)
        run_bytes = dlg._runs["A_1"].bytes
        axis = (workspace["project"] / "output_parquet" / "_axis_bounds.json").stat().st_size
        assert dlg.total_bytes() == base + run_bytes + axis
        assert f"Total: {format_size(base + run_bytes + axis)} uncompressed, 1 of 2" in dlg.total_label.cget("text")
        assert dlg.stale_label.cget("text") == ""
        dlg.checks.set_checked("B_2", True)
        assert "B_2" in dlg.stale_label.cget("text")
        dlg.checks.set_checked("B_2", False)

        dlg.export_button.invoke()                      # refused while jobs run
        assert "Wait for the runs to finish" in dlg.status_label.cget("text") and dlg.manifest is None
        busy[0] = False
        dlg.export_button.invoke()
        _wait(root, lambda: dlg.manifest is not None or "failed" in dlg.status_label.cget("text"))
        assert dlg.manifest is not None, dlg.status_label.cget("text")
        assert dlg.manifest.results == ["A_1"] and dlg.zip_path.is_file()
        zip_path = dlg.zip_path
    finally:
        dlg.destroy()

    imp = ImportDialog(root, workspace["projects"], zip_path=zip_path, solver_config_dir=workspace["config"])
    try:
        assert imp.name_entry.get() == "Proj_2"
        assert str(imp.import_button.cget("state")) == "normal"
        imp.name_entry.delete(0, "end")
        imp.name_entry.insert(0, "proj")                # taken, case-insensitively
        assert str(imp.import_button.cget("state")) == "disabled"
        assert "exists already" in imp.name_note.cget("text")
        imp.name_entry.delete(0, "end")
        imp.name_entry.insert(0, "Proj_2")
        imp.import_button.invoke()
        _wait(root, lambda: not imp.winfo_exists() or "failed" in imp.name_note.cget("text"))
        assert imp.result == "Proj_2", imp.name_note.cget("text")
        assert (workspace["projects"] / "Proj_2" / "output_parquet" / "A_1" / "unit_flow_dt.parquet").is_file()
    finally:
        if imp.winfo_exists():
            imp.destroy()
