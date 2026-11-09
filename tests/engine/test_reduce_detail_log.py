"""Run log, run status and warm reuse of solves with a ``reduce_unit_detail_method`` / ``reduce_system_detail_method``.

Models built from JSON (``tests/fixtures/tests.json``; never a checked-in ``.sqlite``):

* ``dispatch_fullYear_roll`` cut to the first 36 h of p2020, rolls of 12 h / 12 h (three rolls); ``coal_plant``
  with a linear ``startup_method`` (no min load, no ramps); levels ``online`` and ``reserves``.  The fixture's
  reserve groups ``primary`` and ``n_1`` (upward, group ``electricity``) have no requirement, but coal_plant holds
  them, so both are left out.
* ``sys_full`` of ``test_reduce_system_detail``: unit ``g1`` (linear startup, min load 0.5, inertia constant 2)
  holds the upward and downward reserve ``rsv`` of group ``electricity``, which also has an inertia and a
  non-synchronous limit.

Expected lines are written from the design (solve_relaxation_design.md §5.4) and the entity data above, not from
the code's output.
"""
from __future__ import annotations

import pytest

from tests.engine._unit_db import PLANT, b64, fullyear_roll_first_hours, make_db

pytestmark = pytest.mark.solver


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch) -> None:
    """A missing autoscale registry entry raises instead of falling back to an unscaled LP."""
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _f(x: float) -> list:
    return b64(float(x), "float")


def _s(x: str) -> list:
    return b64(x, "str")


ROLL = "dispatch_fullYear_roll"
ROLL_LINE = (f"Solve {ROLL}: reduce_unit_detail_method = online: online variables and start-up costs left out for "
             f"1 unit ({PLANT}). reduce_system_detail_method = reserves: reserves left out for 2 reserve groups "
             "(n_1__up__electricity, primary__up__electricity).")


def test_info_line_once_per_solve_and_run_status_on_every_roll(tmp_path_factory, capsys) -> None:
    """Three rolls of one reduced solve: the run log has the info line once, the run status entry of every roll
    carries both levels and the line, and each roll leaves a marker for its empty online and reserve cells."""
    from flextool.engine import run_chain_from_db
    from flextool.process_outputs.reduced_cells import read_reduced_markers
    from flextool.run_record import RunRecorder, set_active

    vals = fullyear_roll_first_hours(36.0) + [
        ["solve", ROLL, "rolling_solve_horizon", _f(12.0)], ["solve", ROLL, "rolling_solve_jump", _f(12.0)],
        ["solve", ROLL, "reduce_unit_detail_method", _s("online")],
        ["solve", ROLL, "reduce_system_detail_method", _s("reserves")],
        ["unit", PLANT, "startup_method", _s("linear")]]
    url, scen = make_db(tmp_path_factory, alt="rdl_roll", values=vals,
                        base_chain=["init", "west", "coal", "fullYear", ROLL])
    rec = RunRecorder()
    rec.folder = tmp_path_factory.mktemp("rdl_status")
    rec._status = {"schema": "flextool.run_status", "schema_version": 1, "run_id": rec.run_id, "state": "running",
                   "updated_at": None, "outputs": {"parquet_state": "none"}, "solves": [], "solves_omitted": 0,
                   "warnings": [], "warnings_omitted": 0}
    set_active(rec)
    try:
        wf = tmp_path_factory.mktemp("rdl_roll_w")
        steps = run_chain_from_db(url, scen, work_folder=wf, keep_solutions=True)
    finally:
        set_active(None)
    assert len(steps) == 3, list(steps)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "reduce_unit_detail_method =" in ln]
    assert lines == [ROLL_LINE]
    solves = rec._status["solves"]
    assert len(solves) == 3, solves
    for entry in solves:
        assert entry["reduced"] == {"unit": "online", "system": "reserves", "summary": ROLL_LINE}, entry
    assert read_reduced_markers(wf / "output_raw") == {name: frozenset({"online", "reserves"}) for name in steps}


def test_summary_line_names_what_is_left_out_and_the_interactions(tmp_path_factory) -> None:
    """On sys_full, online with the system detail kept: g1 loses its online variables and min load; its reserve now
    comes from built capacity, its downward reserve reaches down to zero flow, and the inertia of electricity is
    counted from flows.  With reserves, inertia and non-synchronous left out too, the system part names them and
    no interaction is left.  A level with nothing to reduce says so."""
    from flextool.engine.preprocess._reduce_detail import ReduceReport, apply_reduce_detail, summary_line
    from tests.engine.test_reduce_system_detail import _solve, _sys_full

    fd = _solve(tmp_path_factory, "rs_full", *_sys_full()).flex_data
    unit = ("reduce_unit_detail_method = online: online variables and start-up costs left out for 1 unit (g1); "
            "min load left out for 1 unit (g1).")
    _, rep = apply_reduce_detail(fd, "online", "none")
    assert summary_line(rep, fd, "s") == (
        f"Solve s: {unit} The reserves of 1 unit (g1) now come from built capacity, not online capacity; the "
        "downward reserve of 1 unit (g1) now reaches down to zero flow, not to min load; inertia of 1 group "
        "(electricity) is now counted from flows, not from units online.")
    _, rep = apply_reduce_detail(fd, "online", "reserves_inertia_non_synchronous")
    assert summary_line(rep, fd, "s") == (
        f"Solve s: {unit} reduce_system_detail_method = reserves_inertia_non_synchronous: reserves left out for 2 "
        "reserve groups (rsv__down__electricity, rsv__up__electricity); inertia limits left out for 1 group "
        "(electricity); non-synchronous limits left out for 1 group (electricity).")
    assert summary_line(ReduceReport(unit="none", system="dc_power_flow"), fd, "s") == (
        "Solve s: reduce_system_detail_method = dc_power_flow: nothing to reduce.")

