"""``_run_status.json`` through a run: start, solves, finish; the ``previous`` carry rules; foreign runs; retries."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest

from flextool.run_options import RunOptions
from flextool.run_record import RunRecorder, read_status_file, worst_verdict
from flextool.run_record import _owner
from flextool.run_record._owner import RUN_LOG_FILE, VERDICT_ORDER, write_status_atomic


@pytest.fixture
def recorders():
    started: list[RunRecorder] = []
    yield started
    for rec in started:
        rec.close()


def _start(recorders, folder: Path, environ: dict | None = None) -> RunRecorder:
    run_options = RunOptions.from_env(environ or {})
    rec = RunRecorder.start(
        parquet_dir=folder,
        input={"db_url": f"sqlite:///{folder.parent / 'input.sqlite'}?spinedbfilter=x", "scenario": "base"},
        output={"location": str(folder.parent.parent), "subdir": folder.name, "work_folder": "/work",
                "write_methods": ["parquet"]},
        options=run_options.summary_lines(),
        argv=["cmd_run_flextool", "input.sqlite"],
        log_capture="python",
        launcher="direct",
        run_options=run_options,
    )
    recorders.append(rec)
    return rec


def _succeed(rec: RunRecorder) -> None:
    rec.solve_started("y2030", "y2030", 0, "Solve start: y2030, 1/1")
    rec.solve_classified(model_status="kOptimal", acceptance="optimal", message="", objective=12.5, warm_used=False)
    rec.solve_finished(0)
    rec.outputs_cleared()
    rec.outputs_written()
    rec.finish("succeeded", exit_code=0, error=None)


def _fail(rec: RunRecorder) -> None:
    rec.solve_started("y2030", "y2030", 0, None)
    rec.solve_classified(model_status="kInfeasible", acceptance="rejected",
                         message="solve for y2030 is infeasible", objective=None, warm_used=False)
    rec.solve_finished(1)
    rec.finish("failed", exit_code=1, error={"type": "FlexToolSolveError", "message": "Error: 1"})


def test_start_solves_finish(tmp_path, recorders):
    folder = tmp_path / "out" / "output_parquet" / "base_1"
    (tmp_path / "out").mkdir()
    db = tmp_path / "out" / "output_parquet" / "input.sqlite"
    rec = _start(recorders, folder, environ={"FLEXTOOL_THREADS": "3", "FLEXTOOL_HIGHS_PRESOLVE": "off"})
    running = read_status_file(folder)
    assert running["state"] == "running" and running["run_id"] == rec.run_id
    assert running["schema"] == "flextool.run_status" and running["schema_version"] == 1
    assert (folder / RUN_LOG_FILE).read_text(encoding="utf-8") == (
        f"# FlexTool run {rec.run_id} started {running['started_at']}\n"
    )
    # The query of the URL is not part of the database path.
    assert running["input"]["db_path"] == str(db.resolve())
    assert running["options"] == ["threads=3 (env FLEXTOOL_THREADS)", "presolve=off (env FLEXTOOL_HIGHS_PRESOLVE)"]
    assert running["options_resolved"]["threads"] == {"value": 3, "source": "env", "env": "FLEXTOOL_THREADS"}
    assert running["options_resolved"]["scaling"] == {"value": "full", "source": "default", "env": "FLEXTOOL_SCALING"}
    # FLEXTOOL_THREADS=3 with no --highs-threads: HiGHS gets threads=3.
    threads = running["threads"]
    assert (threads["budget"], threads["source"], threads["highs"]) == (3, "env", 3)
    assert running["outputs"]["parquet_state"] == "none"

    rec.note_input(db_version=77, alternatives=["base", "dry"])
    rec.note_solver("highs")
    _succeed(rec)
    done = read_status_file(folder)
    assert done["state"] == "succeeded" and done["exit_code"] == 0 and done["error"] is None
    assert done["verdict"] == "optimal"
    assert done["solver"] == "highs"
    assert (done["input"]["db_version"], done["input"]["alternatives"]) == (77, ["base", "dry"])
    assert done["outputs"]["parquet_state"] == "written" and done["outputs"]["by"] == "run"
    [solve] = done["solves"]
    assert solve["name"] == "y2030" and solve["parent"] is None and solve["index"] == 0
    assert (solve["kind"], solve["state"], solve["model_status"], solve["acceptance"]) == (
        "lp", "accepted", "kOptimal", "optimal")
    assert solve["objective"] == 12.5 and solve["log_anchor"] == "Solve start: y2030, 1/1"
    assert done["finished_at"] is not None and done["wall_time_s"] >= 0
    # finish is idempotent: a second call changes nothing.
    rec.finish("failed", exit_code=1, error={"type": "X", "message": "late"})
    assert read_status_file(folder)["state"] == "succeeded"


def test_failed_run_reports_the_acceptance_message(tmp_path, recorders):
    folder = tmp_path / "base"
    _fail(_start(recorders, folder))
    status = read_status_file(folder)
    assert status["state"] == "failed" and status["verdict"] == "infeasible"
    assert status["error"] == {"type": "FlexToolSolveError", "message": "solve for y2030 is infeasible"}
    assert status["solves"][0]["state"] == "rejected"


def test_written_then_two_failures_point_at_the_run_that_wrote_the_files(tmp_path, recorders):
    folder = tmp_path / "base"
    first = _start(recorders, folder)
    _succeed(first)
    first_finished = read_status_file(folder)["finished_at"]
    for _ in range(2):
        _fail(_start(recorders, folder))
        outputs = read_status_file(folder)["outputs"]
        assert outputs["parquet_state"] == "previous_run"
        assert outputs["previous"] == {"run_id": first.run_id, "finished_at": first_finished}


def test_partial_stays_partial_after_a_failure(tmp_path, recorders):
    folder = tmp_path / "base"
    rec = _start(recorders, folder)
    rec.outputs_cleared()  # write_outputs emptied the folder, then the run died
    rec.finish("failed", exit_code=1, error={"type": "OSError", "message": "disk full"})
    _fail(_start(recorders, folder))
    assert read_status_file(folder)["outputs"]["parquet_state"] == "partial"


def test_legacy_parquet_folder_becomes_previous_run_of_unknown_run(tmp_path, recorders):
    folder = tmp_path / "base"
    folder.mkdir()
    older, newer = folder / "a.parquet", folder / "b.parquet"
    older.write_bytes(b"x")
    newer.write_bytes(b"y")
    stamp_old = datetime(2026, 11, 1, 8, 0, 0).timestamp()
    stamp_new = datetime(2026, 11, 2, 9, 30, 0).timestamp()
    os.utime(older, (stamp_old, stamp_old))
    os.utime(newer, (stamp_new, stamp_new))
    _start(recorders, folder)
    outputs = read_status_file(folder)["outputs"]
    assert outputs["parquet_state"] == "previous_run"
    expected = datetime.fromtimestamp(stamp_new).astimezone().isoformat(timespec="seconds")
    assert outputs["previous"] == {"run_id": None, "finished_at": expected}


def test_a_foreign_run_id_stops_this_recorder(tmp_path, recorders):
    folder = tmp_path / "base"
    rec = _start(recorders, folder)
    taken_over = {"run_id": "another-run", "state": "running"}
    write_status_atomic(folder, taken_over)
    rec.solve_started("y2030", "y2030", 0, None)
    rec.finish("succeeded", exit_code=0, error=None)
    assert read_status_file(folder) == taken_over


def test_replace_retried_on_permission_error(tmp_path, monkeypatch):
    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError("held open by a reader")
        return real_replace(src, dst)

    monkeypatch.setattr(_owner.os, "replace", flaky_replace)
    write_status_atomic(tmp_path, {"run_id": "r1", "state": "running"})
    assert calls["n"] == 3
    assert read_status_file(tmp_path) == {"run_id": "r1", "state": "running"}
    assert list(tmp_path.glob("*.tmp")) == []


def _entry_for(verdict: str) -> dict:
    """A solve entry whose verdict is *verdict*."""
    return {
        "error": {"state": "error"},
        "infeasible": {"state": "rejected", "acceptance": "rejected", "model_status": "kInfeasible"},
        "unbounded": {"state": "rejected", "acceptance": "rejected", "model_status": "kUnbounded"},
        "limit": {"state": "rejected", "acceptance": "rejected", "model_status": "kTimeLimit"},
        "not_certified": {"state": "rejected", "acceptance": "rejected", "model_status": "kUnknown"},
        "no_incumbent": {"state": "rejected", "acceptance": "no_incumbent"},
        "not_converged": {"state": "accepted", "acceptance": "not_converged"},
        "near_optimal": {"state": "accepted", "acceptance": "near_optimal"},
        "optimal": {"state": "accepted", "acceptance": "optimal"},
    }[verdict]


def test_verdict_is_the_worst_solve_in_the_documented_order():
    assert VERDICT_ORDER == ("error", "infeasible", "unbounded", "limit", "not_certified", "no_incumbent",
                             "not_converged", "near_optimal", "optimal")
    for i, worse in enumerate(VERDICT_ORDER):
        assert worst_verdict([_entry_for(worse)]) == worse
        for better in VERDICT_ORDER[i + 1:]:
            assert worst_verdict([_entry_for(better), _entry_for(worse)]) == worse
    assert worst_verdict([{"state": "running"}]) is None
    assert worst_verdict([]) is None
