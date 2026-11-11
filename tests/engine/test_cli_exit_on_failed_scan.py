"""A run whose final optimality scan fails exits 1, does not write the output-info database and records the failure.

``main`` is driven in-process with ``run_chain_from_db`` replaced by a stub that returns one Benders step without a
feasible incumbent (``obj=None``): ``_scan_cascade_optimality`` rejects it, so no outputs are written.  The stub
also makes the run-record calls the real solve loop and Benders driver make for such a solve (start, Benders
result, ``solver.run`` returning 0 -- a Benders solve without an incumbent is rejected by the scan, not by the
solve).  The output-info database named on the command line does not exist beforehand, and only the output-info
write would create it, so its absence shows the write was skipped.
"""
from __future__ import annotations

import logging
import os
import sys
from types import SimpleNamespace

import pytest

import flextool.engine
from flextool import run_record
from flextool.cli import cmd_run_flextool


@pytest.fixture
def restore_env():
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def _benders_step_without_incumbent() -> SimpleNamespace:
    return SimpleNamespace(
        optimal=False, near_optimal=False, is_benders=True, obj=None, solve_name="y2030",
        benders_gap=None, benders_tol=1e-4, benders_iterations=50,
        flex_data=None, solution=None, flex_data_provider=None,
    )


def test_failed_scan_exits_1_without_output_info_write(restore_env, monkeypatch, tmp_path):
    def stub(input_db_url, scenario_name, **kwargs):
        rec = run_record.active()
        rec.solve_started("y2030", "y2030", 0, "Solve start: y2030, 1/1")
        rec.solve_benders(converged=False, has_incumbent=False, objective=None, gap=None, tol=1e-4,
                          iterations=50)
        rec.solve_finished(0)
        return {"y2030": _benders_step_without_incumbent()}

    monkeypatch.setattr(flextool.engine, "run_chain_from_db", stub)
    db = tmp_path / "input.sqlite"
    db.write_bytes(b"")
    info_db = tmp_path / "run_info.sqlite"
    monkeypatch.setattr(sys, "argv", [
        "cmd_run_flextool", f"sqlite:///{db}", f"sqlite:///{info_db}", "--scenario-name", "base",
        "--work-folder", str(tmp_path / "work"), "--output-location", str(tmp_path / "out"),
        "--output-subdir", "base_1",
    ])
    handlers_before = list(logging.getLogger().handlers)
    stdout_before = sys.stdout
    with pytest.raises(SystemExit) as exc:
        cmd_run_flextool.main()
    assert exc.value.code == 1
    assert not info_db.exists()
    folder = tmp_path / "out" / "output_parquet" / "base_1"
    assert not list(folder.glob("*.parquet"))

    status = run_record.read_status_file(folder)
    assert status["state"] == "failed" and status["exit_code"] == 1
    assert status["verdict"] == "no_incumbent"
    assert status["error"]["type"] == "BendersNoIncumbent"
    assert status["error"]["message"] == "decomposed solve 'y2030' found no feasible solution in 50 iterations"
    [solve] = status["solves"]
    assert (solve["kind"], solve["state"], solve["acceptance"]) == ("benders", "rejected", "no_incumbent")
    assert status["outputs"]["parquet_state"] == "none"
    assert status["launcher"] == "direct" and status["log_capture"] == "python"
    assert (folder / run_record.RUN_LOG_FILE).read_text(encoding="utf-8").startswith(
        f"# FlexTool run {status['run_id']} started ")

    # Nothing of the run is left in the process (a Spine Toolbox console runs the next scenario in it).
    assert run_record.active() is run_record.NULL_RECORDER
    assert not [h for h in logging.getLogger().handlers if h not in handlers_before]
    assert sys.stdout is stdout_before
