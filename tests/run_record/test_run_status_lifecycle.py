"""``_run_status.json`` through a run: start, solves, finish; the ``previous`` carry rules; foreign runs; retries."""
from __future__ import annotations

import ast
import errno
import logging
import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

import pytest

from flextool.run_options import RunOptions
from flextool.run_record import RunRecorder, active, read_status_file, results_usable, set_active, worst_verdict
from flextool.run_record import _owner, _status
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


# ---------------------------------------------------------------------------------------------------------
# Locks: the warning handler never waits for the recorder, and the recorder never logs while it holds its lock


def _bare_recorder(folder: Path) -> RunRecorder:
    """A recorder writing into *folder* without a heartbeat thread (deterministic: nothing writes on its own)."""
    folder.mkdir(parents=True, exist_ok=True)
    rec = RunRecorder()
    rec.folder = folder
    rec._status = {"schema": "flextool.run_status", "schema_version": 1, "run_id": rec.run_id,
                   "state": "running", "updated_at": None, "outputs": {"parquet_state": "none"},
                   "solves": [], "solves_omitted": 0, "warnings": [], "warnings_omitted": 0}
    write_status_atomic(folder, rec._status)
    return rec


def test_warning_handler_takes_no_recorder_lock(tmp_path):
    rec = _bare_recorder(tmp_path / "base")
    logger = logging.getLogger("flextool.test.warning_handler_lock")
    logger.propagate = False
    logger.addHandler(rec.warning_handler())
    held, release = threading.Event(), threading.Event()

    def hold():
        with rec._lock:
            held.set()
            release.wait(30)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert held.wait(10)
    worker = threading.Thread(target=lambda: logger.warning("from a worker thread"), daemon=True)
    worker.start()
    worker.join(10)
    returned = not worker.is_alive()
    release.set()
    holder.join(10)
    worker.join(10)
    logger.removeHandler(rec.warning_handler())
    assert returned, "the warning handler waited for the recorder lock"
    rec.finish("succeeded", exit_code=0, error=None)
    warnings = read_status_file(tmp_path / "base")["warnings"]
    assert [w["message"] for w in warnings] == ["from a worker thread"]


def test_warning_already_in_its_solve_entry_is_not_repeated(tmp_path):
    """Item 246 (review R4): a warning marked ``IN_SOLVE_RECORD`` (a solve kept at a limit, whose message is the
    solve's entry) is not added to ``warnings`` a second time."""
    from flextool.run_record import IN_SOLVE_RECORD

    rec = _bare_recorder(tmp_path / "base")
    logger = logging.getLogger("flextool.test.in_solve_record")
    logger.propagate = False
    logger.addHandler(rec.warning_handler())
    try:
        logger.warning("kept at a limit", extra={IN_SOLVE_RECORD: True})
        logger.warning("another warning")
    finally:
        logger.removeHandler(rec.warning_handler())
    rec.finish("succeeded", exit_code=0, error=None)
    assert [w["message"] for w in read_status_file(tmp_path / "base")["warnings"]] == ["another warning"]


def test_no_log_call_while_recorder_lock_held(tmp_path, monkeypatch):
    failing, taken_over = _bare_recorder(tmp_path / "a"), _bare_recorder(tmp_path / "b")
    rec = failing
    violations: list[str] = []
    seen: list[str] = []

    class Probe(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())
            result = {}

            def try_lock():
                got = rec._lock.acquire(timeout=0)
                result["free"] = got
                if got:
                    rec._lock.release()

            t = threading.Thread(target=try_lock)
            t.start()
            t.join(10)
            if not result.get("free"):
                violations.append(record.getMessage())

    probe = Probe(level=logging.INFO)
    status_logger = logging.getLogger("flextool.run_record._status")
    status_logger.addHandler(probe)
    try:
        def failing_write(*args, **kwargs):
            raise OSError(28, "No space left on device")

        with monkeypatch.context() as m:
            m.setattr(_status, "write_status_atomic", failing_write)
            failing.solve_started("y2030", "y2030", 0, None)
        rec = taken_over
        write_status_atomic(tmp_path / "b", {"run_id": "f" * 32, "state": "running"})
        taken_over.finish("succeeded", exit_code=0, error=None)
    finally:
        status_logger.removeHandler(probe)
    assert any("Cannot write the run record" in s for s in seen)
    assert any("Another run took over" in s for s in seen)
    assert violations == []


def test_logger_calls_only_after_the_lock_is_released():
    """In ``RunRecorder`` the module logger is used only by ``_flush_deferred`` (called after a ``with`` block)."""
    tree = ast.parse(Path(_status.__file__).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RunRecorder")
    users = set()
    for fn in (n for n in cls.body if isinstance(n, ast.FunctionDef)):
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and node.id == "_logger":
                users.add(fn.name)
    assert users == {"_flush_deferred"}


_DEADLOCK_SCRIPT = r'''
import logging, os, sys, threading
from pathlib import Path
from flextool.run_record import _status
from flextool.run_record._owner import write_status_atomic

folder = Path(sys.argv[1])
folder.mkdir(parents=True, exist_ok=True)
rec = _status.RunRecorder()
rec.folder = folder
rec._status = {"run_id": rec.run_id, "state": "running", "outputs": {}, "solves": [], "solves_omitted": 0,
               "warnings": [], "warnings_omitted": 0}
write_status_atomic(folder, rec._status)
root = logging.getLogger()
root.addHandler(rec.warning_handler())
b_in_emit = threading.Event()

class Message:
    def __str__(self):
        b_in_emit.set()      # thread B is inside the warning handler (its lock taken, if it has one)
        return "warning from thread B"

def failing_write(*args, **kwargs):
    b_in_emit.wait(30)       # thread A holds the recorder lock until B is inside the handler
    raise OSError(28, "No space left on device")

_status.write_status_atomic = failing_write

a = threading.Thread(target=lambda: rec.solve_started("y2030", "y2030", 0, None), daemon=True)
b = threading.Thread(target=lambda: logging.getLogger("other").warning(Message()), daemon=True)
a.start()
b.start()
a.join(30)
b.join(30)
print("ok" if not a.is_alive() and not b.is_alive() else "stuck", flush=True)
'''


def test_deadlock_scenario_exits(tmp_path):
    script = tmp_path / "deadlock.py"
    script.write_text(_DEADLOCK_SCRIPT, encoding="utf-8")
    try:
        proc = subprocess.run([sys.executable, str(script), str(tmp_path / "base")], capture_output=True,
                              text=True, timeout=60)
    except subprocess.TimeoutExpired:
        pytest.fail("the engine process hung (lock-order deadlock)")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


class _SignalAfterAcquire:
    """A recorder lock whose first ``__enter__`` acquires and then raises, as a signal handler would."""

    def __init__(self, inner):
        self._inner = inner
        self.armed = True

    def __enter__(self):
        self._inner.acquire()
        if self.armed:
            self.armed = False
            raise KeyboardInterrupt("signal right after the acquire")
        return self

    def __exit__(self, *exc):
        self._inner.release()
        return False

    def acquire(self, *args, **kwargs):
        return self._inner.acquire(*args, **kwargs)

    def release(self):
        self._inner.release()

    def _is_owned(self):
        return self._inner._is_owned()


def test_finish_completes_after_a_signal_left_the_lock_held(tmp_path):
    """SIGTERM -> ``RunTerminated`` can land right after an acquire; ``finish`` on the same thread must not hang."""
    rec = _bare_recorder(tmp_path / "base")
    rec._lock = _SignalAfterAcquire(rec._lock)
    done = threading.Event()

    def run():
        try:
            rec.note_solver("highs")
        except KeyboardInterrupt:
            pass
        rec.finish("interrupted", exit_code=143, error={"type": "RunTerminated", "message": "SIGTERM"})
        done.set()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    assert done.wait(10), "finish hung on the recorder lock"
    assert read_status_file(tmp_path / "base")["state"] == "interrupted"


# ---------------------------------------------------------------------------------------------------------
# Write failures are transient; the run-id guard reads only the head of the file


class _Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _failing_replace(monkeypatch, n: int, exc: OSError) -> dict:
    """``os.replace`` (as the status writer calls it) raises *exc* on its next *n* calls."""
    real = os.replace
    calls = {"n": 0, "failed": 0}

    def replace(src, dst):
        calls["n"] += 1
        if calls["failed"] < n:
            calls["failed"] += 1
            raise exc
        return real(src, dst)

    monkeypatch.setattr(_owner.os, "replace", replace)
    return calls


def test_transient_permission_error_then_success(tmp_path, recorders, monkeypatch):
    folder = tmp_path / "base"
    rec = _start(recorders, folder)
    set_active(rec)
    try:
        # One non-final write gives up after 5 replace attempts 50 ms apart (a Windows reader holds the file).
        _failing_replace(monkeypatch, 5, PermissionError(13, "held open by a reader"))
        rec.solve_started("y2030", "y2030", 0, None)
        assert active() is rec
        assert read_status_file(folder)["solves"] == []
        rec.solve_classified(model_status="kOptimal", acceptance="optimal", message="", objective=1.0,
                             warm_used=False)
        rec.solve_finished(0)
        rec.outputs_cleared()
        assert active() is rec
        active().outputs_written()
        rec.finish("succeeded", exit_code=0, error=None)
    finally:
        set_active(None)
    done = read_status_file(folder)
    assert (done["state"], done["outputs"]["parquet_state"]) == ("succeeded", "written")
    assert [s["name"] for s in done["solves"]] == ["y2030"]
    assert results_usable(done) is True


def test_oserror_backoff_schedule(tmp_path, monkeypatch, caplog):
    rec = _bare_recorder(tmp_path / "base")  # no heartbeat thread: only the calls below write
    clock = _Clock(100.0)
    monkeypatch.setattr(_status, "_monotonic", clock)
    calls = _failing_replace(monkeypatch, 3, OSError(errno.ENOSPC, "No space left on device"))
    caplog.set_level(logging.INFO, logger="flextool.run_record._status")
    attempts: list[float] = []

    def write_at(t: float) -> None:
        clock.now = t
        before = calls["n"]
        rec.note_solver(f"solver-{t}")  # a change, so there is something to write
        rec._update(force=True)
        if calls["n"] > before:
            attempts.append(t)

    for t in (100.0, 100.5, 100.99, 101.0, 102.9, 103.0, 106.9, 107.0, 107.5):
        write_at(t)
    # Waits of 1, 2 and 4 s after the 1st, 2nd and 3rd failure; the 4th attempt succeeds.
    assert attempts == [100.0, 101.0, 103.0, 107.0, 107.5]
    assert read_status_file(tmp_path / "base")["solver"] == "solver-107.5"
    records = [r for r in caplog.records if r.name == "flextool.run_record._status"]
    assert [(r.levelno, "retrying" in r.getMessage()) for r in records if r.levelno == logging.WARNING] == [
        (logging.WARNING, True)]
    infos = [r.getMessage() for r in records if r.levelno == logging.INFO]
    assert len(infos) == 1 and "after 3 failed attempts" in infos[0]


def test_backoff_wait_caps_at_30_seconds(tmp_path, monkeypatch):
    rec = _bare_recorder(tmp_path / "base")
    clock = _Clock(0.0)
    monkeypatch.setattr(_status, "_monotonic", clock)
    _failing_replace(monkeypatch, 100, OSError(errno.ENOSPC, "No space left on device"))
    waits = []
    for _ in range(8):
        rec._update(force=True)
        waits.append(rec._retry_at - clock.now)
        clock.now = rec._retry_at
    assert waits == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0, 30.0]


def test_final_write_ignores_backoff(tmp_path, monkeypatch):
    rec = _bare_recorder(tmp_path / "base")
    _failing_replace(monkeypatch, 1, OSError(errno.ENOSPC, "No space left on device"))
    rec.solve_started("y2030", "y2030", 0, None)
    rec._retry_at = _status._monotonic() + 3600.0
    rec.finish("succeeded", exit_code=0, error=None)
    assert read_status_file(tmp_path / "base")["state"] == "succeeded"


def test_failed_final_write_is_retried_by_close(tmp_path, monkeypatch):
    rec = _bare_recorder(tmp_path / "base")
    real = _status.write_status_atomic
    failed = {"n": 0}

    def write(folder, data, *, final=False):
        if final and not failed["n"]:
            failed["n"] += 1
            raise PermissionError(13, "held open by a reader")  # the 2 s of replaces and the in-place open failed
        return real(folder, data, final=final)

    monkeypatch.setattr(_status, "write_status_atomic", write)
    rec.finish("succeeded", exit_code=0, error=None)
    assert read_status_file(tmp_path / "base")["state"] == "running"
    rec.close()
    assert read_status_file(tmp_path / "base")["state"] == "succeeded"


@pytest.mark.parametrize("bad", [float("nan"), object()], ids=["nan", "not_json"])
def test_unserialisable_status_stops_the_recorder_but_not_the_run(tmp_path, bad):
    rec = _bare_recorder(tmp_path / "base")
    set_active(rec)
    try:
        rec._status["input"] = {"db_version": bad}  # a value that bypasses _finite / _jsonable
        rec.solve_started("y2030", "y2030", 0, None)
        assert rec._disabled
        assert active().is_recording is False and active().stopped is True
        rec.finish("succeeded", exit_code=0, error=None)  # does not raise
    finally:
        set_active(None)
    assert read_status_file(tmp_path / "base")["state"] == "running"


def test_guard_skips_full_read_when_the_head_shows_our_run_id(tmp_path, recorders, monkeypatch):
    rec = _start(recorders, tmp_path / "base")
    reads = {"n": 0}
    real = _status.read_status_file

    def counting(*args, **kwargs):
        reads["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(_status, "read_status_file", counting)
    for i in range(50):
        rec.solve_started(f"roll_{i}", "base", i, None)
    assert reads["n"] == 0
    assert len(read_status_file(tmp_path / "base")["solves"]) == 50


def test_guard_detects_a_takeover_of_identical_size_and_mtime(tmp_path, recorders):
    folder = tmp_path / "base"
    rec = _start(recorders, folder)
    path = folder / "_run_status.json"
    st = path.stat()
    foreign = read_status_file(folder)
    foreign["run_id"] = "0" * 32
    write_status_atomic(folder, foreign)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert path.stat().st_size == st.st_size and path.stat().st_mtime_ns == st.st_mtime_ns
    rec.solve_started("y2030", "y2030", 0, None)
    rec.finish("succeeded", exit_code=0, error=None)
    assert read_status_file(folder) == foreign


def test_takeover_leaves_the_other_runs_outputs_alone(tmp_path, recorders):
    from flextool.process_outputs.write_outputs import _note_parquet_written

    folder = tmp_path / "base"
    rec = _start(recorders, folder)
    set_active(rec)
    try:
        other = {"schema": "flextool.run_status", "schema_version": 1, "run_id": "1" * 32, "state": "running",
                 "outputs": {"parquet_state": "partial"}}
        write_status_atomic(folder, other)
        rec.solve_started("y2030", "y2030", 0, None)  # sees the takeover and stops
        _note_parquet_written(str(folder))
    finally:
        set_active(None)
    assert read_status_file(folder) == other


def test_rewriting_an_earlier_runs_outputs_marks_them_partial_until_written(tmp_path):
    """``flextool-write-outputs`` (no active recorder) on a folder of a finished run: the record says ``partial``
    from before the files are removed until they are written again, then ``written`` with a new ``written_at``,
    under the same run id (a comparison built meanwhile cannot take it for that run's results)."""
    from flextool.process_outputs.write_outputs import _note_parquet

    folder = tmp_path / "base"
    folder.mkdir()
    done = {"schema": "flextool.run_status", "schema_version": 1, "run_id": "1" * 32, "state": "succeeded",
            "outputs": {"parquet_state": "written", "written_at": "2026-11-01T10:00:00+02:00", "by": "run",
                        "previous": None, "reason": None}}
    write_status_atomic(folder, done)
    assert not active().is_recording
    _note_parquet(str(folder), written=False)
    status = read_status_file(folder)
    assert status["outputs"]["parquet_state"] == "partial" and not results_usable(status)
    assert status["run_id"] == "1" * 32
    _note_parquet(str(folder), written=True)
    status = read_status_file(folder)
    assert (status["outputs"]["parquet_state"], status["outputs"]["by"]) == ("written", "write_outputs")
    assert status["outputs"]["written_at"] != "2026-11-01T10:00:00+02:00" and results_usable(status)


def test_a_settled_read_waits_out_a_status_that_cannot_be_read_now(tmp_path, monkeypatch):
    answers = [{"state": "unreadable"}, {"state": "unreadable"}, {"state": "succeeded", "run_id": "r"}]
    monkeypatch.setattr(_owner, "read_status_file", lambda folder, strict=False: answers.pop(0))
    monkeypatch.setattr(_owner.time, "sleep", lambda s: None)
    assert _owner.read_status_settled(tmp_path) == {"state": "succeeded", "run_id": "r"}
    # Still unreadable when the time is up: that answer stands.
    monkeypatch.setattr(_owner, "read_status_file", lambda folder, strict=False: {"state": "unreadable"})
    assert _owner.read_status_settled(tmp_path, seconds=0.0) == {"state": "unreadable"}


# ---------------------------------------------------------------------------
# A run that ended before it started its record: the launcher's minimal record
# ---------------------------------------------------------------------------


def _expected(folder: Path) -> "_owner.ExpectedRun":
    return _owner.ExpectedRun(folder=folder, pid=999_999_999, launched_at=_owner.now_iso(), argv=["engine"],
                              input={"db_url": "sqlite:///in.sqlite", "scenario": "base"},
                              output={"subdir": folder.name})


def test_a_minimal_record_gives_way_to_a_run_that_starts_while_it_is_written(tmp_path, monkeypatch):
    old = {"run_id": "a" * 32, "state": "succeeded", "started_at": "2026-11-01T10:00:00+02:00",
           "finished_at": "2026-11-01T10:05:00+02:00", "outputs": {"parquet_state": "written"}}
    quiet, raced = tmp_path / "quiet", tmp_path / "raced"
    for folder in (quiet, raced):
        folder.mkdir()
        write_status_atomic(folder, old)
        (folder / RUN_LOG_FILE).write_text("# FlexTool run old\n", encoding="utf-8")

    # Without a race: written, the earlier run's parquet carried, then the log.
    assert _owner.record_run_without_marker(_expected(quiet), 1, log_lines=["early error"]) is True
    record = read_status_file(quiet)
    assert record["state"] == "failed" and record["outputs"]["previous"]["run_id"] == "a" * 32
    assert (quiet / RUN_LOG_FILE).read_text(encoding="utf-8").splitlines()[1:] == [
        "early error", "# gui: the run ended before it started its run record (exit code 1): failed"]

    # An engine starts in the folder after the launcher read the old record and before it replaces it: the new
    # run's record and log stay.
    real_carried = _owner.carried_outputs

    def engine_starts_meanwhile(folder, previous):
        outputs = real_carried(folder, previous)
        write_status_atomic(folder, {**old, "run_id": "b" * 32, "state": "running", "started_at": _owner.now_iso()})
        (folder / RUN_LOG_FILE).write_text("# FlexTool run new\n", encoding="utf-8")
        return outputs

    monkeypatch.setattr(_owner, "carried_outputs", engine_starts_meanwhile)
    assert _owner.record_run_without_marker(_expected(raced), 1, log_lines=["early error"]) is False
    assert read_status_file(raced)["run_id"] == "b" * 32
    assert (raced / RUN_LOG_FILE).read_text(encoding="utf-8") == "# FlexTool run new\n"
    assert not [p for p in raced.iterdir() if p.name.endswith(".tmp")]
