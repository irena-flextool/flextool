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


def test_dialog_cli_previews_and_rp_naming(tk_root, tmp_path: Path):
    """Both CLI previews populate at open; the RP name encodes solve/n_rp/len."""
    url = _build_db(str(tmp_path / "calib.sqlite"))
    scenario = types.SimpleNamespace(name=SCENARIO, is_xlsx=False)
    settings = ProjectSettings()
    settings.calib_rp_n_rp = 40
    settings.calib_rp_period_length = 54
    settings.calib_selected_solves = [INVEST_SOLVE]
    # "File outputs" choices: csv + excel on, plot + spinedb off. The calib
    # preview must regenerate exactly these from the final parquet.
    settings.auto_generate_scen_plots = False
    settings.auto_generate_scen_excels = True
    settings.auto_generate_scen_csvs = True
    settings.auto_generate_comp_spinedb = False

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
        rp_text = dialog._rp_cli_text.get("1.0", "end").strip()
        calib_text = dialog._calib_cli_text.get("1.0", "end").strip()
        # RP preview is non-empty and carries the scenario-keyed alt name.
        assert "flextool.representative_periods.preprocess" in rp_text
        assert f"{SCENARIO}_rp_40rp_54h" in rp_text
        assert "--alternative-description" in rp_text
        # Calibration preview is non-empty and points at the scenario.
        assert "flextool.calibrate" in calib_text
        assert SCENARIO in calib_text
        # The final-outputs flag mirrors the File-outputs settings (excel + csv,
        # in the run path's order), so the calibrated results match a normal run.
        assert "--final-write-methods excel csv" in calib_text
        assert "--skip-final-outputs" not in calib_text

        # The add-to-scenario checkbox mirrors the (default True) setting.
        assert dialog._var_add_to_scenario.get() is True

        # A second (post-launch) allocation would advance to _2 once the first
        # name is reserved; simulate the reservation the launch performs.
        # Plan tuple: (scenario, name, db_url, applicable_solves, alt, desc).
        first = dialog._rp_plan(commit=True)
        assert first[0][3] == [INVEST_SOLVE]
        assert first[0][4] == f"{SCENARIO}_rp_40rp_54h"
        second = dialog._rp_plan(commit=False)
        assert second[0][4] == f"{SCENARIO}_rp_40rp_54h_2"
    finally:
        dialog.destroy()


def _build_db_with_solves(
    db_path: str, scenario: str, solves: list[str], *, invest: bool = True
) -> str:
    """Schema-complete DB: one scenario running exactly *solves*.

    ``invest`` controls whether each solve carries ``invest_periods`` (an
    investment solve, pre-checked in the dialog) or not (a dispatch solve).
    """
    initialize_database(
        str(package_data_path("schemas/spinedb_schema.json")), db_path
    )
    url = f"sqlite:///{db_path}"
    entities = [("model", MODEL, None)]
    pvs = [("model", MODEL, "solves", Array(solves), ALT)]
    for sv in solves:
        entities.append(("solve", sv, None))
        if invest:
            pvs.append(("solve", sv, "invest_periods", Array(["y2050"]), ALT))
    with DatabaseMapping(url) as db:
        _, errors = import_data(
            db,
            alternatives=[(ALT, "fixture")],
            scenarios=[(scenario, True, "scenario")],
            scenario_alternatives=[(scenario, ALT)],
            entities=entities,
            parameter_values=pvs,
        )
        if errors:
            raise RuntimeError(f"fixture import errors: {errors[:5]}")
        db.commit_session("fixture")
    return url


def test_rp_plan_filters_solves_per_scenario(tk_root, tmp_path: Path):
    """A union-checklist solve that a scenario lacks never reaches that scenario."""
    url_a = _build_db_with_solves(str(tmp_path / "a.sqlite"), "scenA", ["lt"])
    # scenB runs only a DISPATCH solve (like the reported 'coal' scenario), so
    # it is not auto-selected and the invest solve 'lt' does not belong to it.
    url_b = _build_db_with_solves(
        str(tmp_path / "b.sqlite"), "scenB", ["op"], invest=False
    )
    sc_a = types.SimpleNamespace(name="scenA", is_xlsx=False)
    sc_b = types.SimpleNamespace(name="scenB", is_xlsx=False)
    urls = {"scenA": url_a, "scenB": url_b}
    settings = ProjectSettings()
    # Union checklist is {lt, op}; select ONLY lt (which belongs to scenA).
    settings.calib_selected_solves = ["lt"]

    dialog = CalibrateDialog(
        tk_root,
        scenarios=[sc_a, sc_b],
        project_path=tmp_path,
        settings=settings,
        execution_mgr=None,
        python_exe=sys.executable,
        resolve_db_url=lambda sc: urls[sc.name],
        save_settings=lambda: None,
    )
    try:
        assert set(dialog._solve_order) == {"lt", "op"}
        plan = dialog._rp_plan(commit=False)
        # Only scenA is planned (it runs 'lt'); scenB is skipped (no 'lt').
        assert [row[1] for row in plan] == ["scenA"]
        assert plan[0][3] == ["lt"]  # applicable solves — no 'op' leaked in
        # The RP preview never mentions scenB.
        rp_text = dialog._rp_cli_text.get("1.0", "end")
        assert "scenB" not in rp_text
        assert "scenA" in rp_text
    finally:
        dialog.destroy()


def test_solve_checklist_bounded_and_scrollable(tk_root, tmp_path: Path):
    """A long solve list is bounded in a scrollable canvas, all still selectable."""
    solves = [f"invest_{i:02d}" for i in range(12)]
    url = _build_db_with_solves(str(tmp_path / "many.sqlite"), "scenA", solves)
    dialog = CalibrateDialog(
        tk_root,
        scenarios=[types.SimpleNamespace(name="scenA", is_xlsx=False)],
        project_path=tmp_path,
        settings=ProjectSettings(),
        execution_mgr=None,
        python_exe=sys.executable,
        resolve_db_url=lambda _sc: url,
        save_settings=lambda: None,
    )
    try:
        # Every solve is present and independently selectable despite the box
        # showing only a few rows at a time.
        assert dialog._solve_order == solves
        assert set(dialog._solve_vars) == set(solves)
        # The bounded, scrollable containers exist.
        assert isinstance(dialog._solve_canvas, tk.Canvas)
        assert isinstance(dialog._body_canvas, tk.Canvas)
        # The solve canvas is capped near _MAX_SOLVE_ROWS rows, not 12.
        dialog.update_idletasks()
        assert int(dialog._solve_canvas.cget("height")) < 12 * 40
        # The wheel handler routes without error for both X11 and delta events.
        dialog._on_wheel(types.SimpleNamespace(widget=dialog, num=5, delta=0))
        dialog._on_wheel(
            types.SimpleNamespace(widget=dialog._solve_canvas, num=0, delta=-120)
        )
    finally:
        dialog.destroy()


def test_dialog_preview_does_not_create_dirs(tk_root, tmp_path: Path):
    """Opening the dialog (which renders the calib preview) creates no work/ dir."""
    url = _build_db(str(tmp_path / "calib.sqlite"))
    scenario = types.SimpleNamespace(name=SCENARIO, is_xlsx=False)
    dialog = CalibrateDialog(
        tk_root,
        scenarios=[scenario],
        project_path=tmp_path,
        settings=ProjectSettings(),
        execution_mgr=None,
        python_exe=sys.executable,
        resolve_db_url=lambda _sc: url,
        save_settings=lambda: None,
    )
    try:
        assert not (tmp_path / "work").exists()
    finally:
        dialog.destroy()


def test_launch_reveals_execution_window(tk_root, tmp_path: Path, monkeypatch):
    """Both action buttons open/raise the execution window on launch."""
    import flextool.gui.dialogs.calibrate_dialog as cd

    # Stub the actual launchers so no subprocess/thread work happens.
    monkeypatch.setattr(cd, "launch_rp_jobs", lambda *a, **k: None)
    monkeypatch.setattr(cd, "launch_calibration_jobs", lambda *a, **k: None)

    url = _build_db(str(tmp_path / "calib.sqlite"))
    scenario = types.SimpleNamespace(name=SCENARIO, is_xlsx=False)
    settings = ProjectSettings()
    settings.calib_selected_solves = [INVEST_SOLVE]
    revealed = {"n": 0}

    dialog = CalibrateDialog(
        tk_root,
        scenarios=[scenario],
        project_path=tmp_path,
        settings=settings,
        execution_mgr=object(),
        python_exe=sys.executable,
        resolve_db_url=lambda _sc: url,
        save_settings=lambda: None,
        show_execution_window=lambda: revealed.__setitem__("n", revealed["n"] + 1),
    )
    try:
        dialog._on_create_rp()
        assert revealed["n"] == 1
        dialog._on_run()
        assert revealed["n"] == 2
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
