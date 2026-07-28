"""Headless smoke test for the Calibrate-investments dialog.

Proves the dialog constructs and populates against a schema-JSON fixture DB
(CLAUDE.md invariant #3 — never read a checked-in ``.sqlite``) with an
investment solve + a dispatch solve: the solve checklist shows both, with the
investment solve pre-checked and the dispatch solve unchecked, and ``_flush``
persists the resulting selection.

Run headless (no Tk on the live display):
    xvfb-run -a ~/venv-spi/bin/python -m pytest tests/gui/test_calibrate_dialog.py -q
"""
from __future__ import annotations

import sys
import tkinter as tk
import types
from pathlib import Path

import pytest
from spinedb_api import Array, DatabaseMapping, import_data

from flextool._resources import package_data_path
from flextool.gui.data_models import ProjectSettings
from flextool.gui.dialogs.calibrate_dialog import CalibrateDialog
from flextool.update_flextool import initialize_database

SCENARIO = "calib_scen"
ALT = "calib_base"
MODEL = "flexModel"
INVEST_SOLVE = "invest_solve"
DISPATCH_SOLVE = "dispatch_solve"


def _build_db(db_path: str) -> str:
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
    return url


@pytest.fixture()
def tk_root():
    try:
        root = tk.Tk()
        root.withdraw()
        yield root
        root.destroy()
    except tk.TclError:
        pytest.skip("No display available")


def test_dialog_populates_solve_checklist(tk_root, tmp_path: Path):
    url = _build_db(str(tmp_path / "calib.sqlite"))
    scenario = types.SimpleNamespace(name=SCENARIO, is_xlsx=False)
    settings = ProjectSettings()
    saved = {"n": 0}

    dialog = CalibrateDialog(
        tk_root,
        scenarios=[scenario],
        project_path=tmp_path,
        settings=settings,
        execution_mgr=None,
        python_exe=sys.executable,
        resolve_db_url=lambda _sc: url,
        save_settings=lambda: saved.__setitem__("n", saved["n"] + 1),
    )
    try:
        # Union checklist shows both solves, in model.solves order.
        assert dialog._solve_order == [INVEST_SOLVE, DISPATCH_SOLVE]
        # Investment solve pre-checked; dispatch solve unchecked.
        assert dialog._solve_vars[INVEST_SOLVE].get() is True
        assert dialog._solve_vars[DISPATCH_SOLVE].get() is False

        # Buttons enabled (a non-xlsx scenario is provided).
        assert str(dialog._run_button.cget("state")) == "normal"
        assert str(dialog._rp_button.cget("state")) == "normal"

        # A flush persists the checked selection back to settings.
        dialog._flush()
        assert settings.calib_selected_solves == [INVEST_SOLVE]
        assert saved["n"] >= 1
    finally:
        dialog.destroy()


def test_dialog_disabled_for_xlsx_scenario(tk_root, tmp_path: Path):
    url = _build_db(str(tmp_path / "calib.sqlite"))
    scenario = types.SimpleNamespace(name=SCENARIO, is_xlsx=True)
    settings = ProjectSettings()

    dialog = CalibrateDialog(
        tk_root,
        scenarios=[scenario],
        project_path=tmp_path,
        settings=settings,
        execution_mgr=None,
        python_exe=sys.executable,
        resolve_db_url=lambda _sc: url,
        save_settings=lambda: None,
    )
    try:
        # xlsx-backed scenario disables both action buttons with a reason.
        assert str(dialog._run_button.cget("state")) == "disabled"
        assert str(dialog._rp_button.cget("state")) == "disabled"
        assert "xlsx" in dialog._reason_var.get()
        # An xlsx scenario contributes no solves (it is skipped).
        assert dialog._solve_order == []
    finally:
        dialog.destroy()
