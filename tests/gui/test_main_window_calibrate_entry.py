"""Headless wiring test for the main-window 'Calibrate investments' entry point.

Proves ``MainWindow._open_calibrate_dialog`` collects the checked available
scenarios, resolves each to its input-DB url (mirroring
``ExecutionManager.add_jobs``), and constructs a live ``CalibrateDialog`` with
the injected dependencies — without building a full ``MainWindow`` (a heavy
``tk.Tk`` subclass). The handler is bound to a real Tk root carrying stubbed
collaborators; its undecorated body is invoked via ``__wrapped__`` so a real
exception surfaces as a test failure instead of being swallowed by
``safe_callback``.

Fixture DB is built from the schema JSON (CLAUDE.md invariant #3 — never read a
checked-in ``.sqlite``).

Run headless (no Tk on the live display):
    xvfb-run -a ~/venv-spi/bin/python -m pytest \
        tests/gui/test_main_window_calibrate_entry.py -q
"""
from __future__ import annotations

import tkinter as tk
from pathlib import Path

import pytest
from spinedb_api import Array, DatabaseMapping, import_data

from flextool._resources import package_data_path
from flextool.gui import main_window as mw
from flextool.gui.data_models import ProjectSettings, ScenarioInfo
from flextool.gui.dialogs.calibrate_dialog import CalibrateDialog
from flextool.update_flextool import initialize_database

SCENARIO = "calib_scen"
ALT = "calib_base"
MODEL = "flexModel"
INVEST_SOLVE = "invest_solve"
DISPATCH_SOLVE = "dispatch_solve"


def _build_db(db_path: str) -> None:
    """Schema-complete DB: model.solves = [invest, dispatch]; one scenario."""
    initialize_database(
        str(package_data_path("schemas/spinedb_schema.json")), db_path
    )
    url = f"sqlite:///{db_path}"
    entities = [
        ("model", MODEL, None),
        ("solve", INVEST_SOLVE, None),
        ("solve", DISPATCH_SOLVE, None),
    ]
    parameter_values = [
        ("model", MODEL, "solves", Array([INVEST_SOLVE, DISPATCH_SOLVE]), ALT),
        ("solve", INVEST_SOLVE, "invest_periods", Array(["y2050"]), ALT),
    ]
    with DatabaseMapping(url) as db:
        _, errors = import_data(
            db,
            alternatives=[(ALT, "calibrate fixture")],
            scenarios=[(SCENARIO, True, "calibrate scenario")],
            scenario_alternatives=[(SCENARIO, ALT)],
            entities=entities,
            parameter_values=parameter_values,
        )
        if errors:
            raise RuntimeError(f"fixture import errors: {errors[:5]}")
        db.commit_session("calibrate fixture")


@pytest.fixture()
def tk_root():
    try:
        root = tk.Tk()
        root.withdraw()
        yield root
        root.destroy()
    except tk.TclError:
        pytest.skip("No display available")


class _AvailMgr:
    def __init__(self, scenarios: list[ScenarioInfo]) -> None:
        self._scenarios = scenarios

    def get_checked_scenarios(self, _tree):  # noqa: ANN001
        return list(self._scenarios)


class _ExecMgr:
    def __init__(self, source_path: Path) -> None:
        self._source_path = source_path

    def _resolve_source_path(self, _source_name: str) -> Path:
        return self._source_path


def _make_host(tk_root, tmp_path: Path, monkeypatch, checked):
    """Bind the handler's collaborators onto a real Tk root acting as `self`."""
    db_file = tmp_path / "mysource.sqlite"
    _build_db(str(db_file))
    monkeypatch.setattr(mw, "get_projects_dir", lambda: tmp_path)

    host = tk_root
    host.avail_scenario_mgr = _AvailMgr(checked)
    host.current_project = "proj"
    host.available_tree = None
    host.execution_mgr = _ExecMgr(db_file)
    host.project_settings = ProjectSettings()
    host._calibrate_dialog = None
    host._ensure_execution_mgr = lambda: None
    return host


def test_handler_builds_dialog_for_checked_scenario(
    tk_root, tmp_path: Path, monkeypatch
):
    scen = ScenarioInfo(
        name=SCENARIO, source_number=1, source_name="mysource.sqlite"
    )
    host = _make_host(tk_root, tmp_path, monkeypatch, [scen])

    # Call the undecorated body so a real error is not swallowed by
    # safe_callback (which would leave _calibrate_dialog None and fail below).
    mw.MainWindow._open_calibrate_dialog.__wrapped__(host)

    dialog = host._calibrate_dialog
    try:
        assert isinstance(dialog, CalibrateDialog)
        # The resolved db_url fed a real read: both solves enumerated.
        assert dialog._solve_order == [INVEST_SOLVE, DISPATCH_SOLVE]
        # Non-xlsx source → action buttons live.
        assert str(dialog._run_button.cget("state")) == "normal"
        assert str(dialog._rp_button.cget("state")) == "normal"
    finally:
        if dialog is not None:
            dialog.destroy()


def test_handler_noops_when_nothing_checked(
    tk_root, tmp_path: Path, monkeypatch
):
    host = _make_host(tk_root, tmp_path, monkeypatch, [])

    mw.MainWindow._open_calibrate_dialog.__wrapped__(host)

    assert host._calibrate_dialog is None


def test_handler_marks_xlsx_source(tk_root, tmp_path: Path, monkeypatch):
    scen = ScenarioInfo(
        name=SCENARIO, source_number=1, source_name="book.xlsx"
    )
    host = _make_host(tk_root, tmp_path, monkeypatch, [scen])

    mw.MainWindow._open_calibrate_dialog.__wrapped__(host)

    dialog = host._calibrate_dialog
    try:
        assert isinstance(dialog, CalibrateDialog)
        # xlsx-backed scenario is flagged, so the run/RP actions are disabled.
        assert dialog._scenarios[0].is_xlsx is True
        assert str(dialog._run_button.cget("state")) == "disabled"
    finally:
        if dialog is not None:
            dialog.destroy()
