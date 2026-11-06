"""The ``flextool-run`` supervisor owns its child's output pipe and finishes a run record the child left unfinished.

The child is ``_stub_engine.py``: it writes a ``running`` status and the log header, prints the marker like the
engine with a pipe owner, and then crashes, kills itself or exits (see its docstring).  What reaches ``_run.log``
through the pipe -- a line written straight to fd 1, faulthandler's crash dump -- is what only the pipe owner
sees (``log_capture: "parent"``).
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import flextool_run_supervisor as supervisor
from flextool.run_record import RUN_LOG_FILE, classify_exit, read_status_file

REPO = Path(__file__).resolve().parents[2]
STUB = Path(__file__).with_name("_stub_engine.py")
posix_only = pytest.mark.skipif(os.name == "nt", reason="POSIX signals")


def _stub_cmd() -> list[str]:
    return [sys.executable, str(STUB)]


def _log(folder: Path) -> list[str]:
    return (folder / RUN_LOG_FILE).read_text(encoding="utf-8").splitlines()


@posix_only
def test_native_output_and_a_crash_dump_reach_the_log_and_the_run_is_crashed(tmp_path, capsys):
    folder = tmp_path / "out"
    status = supervisor.run_supervised([str(folder), "segv"], child_cmd=_stub_cmd())

    assert status == -signal.SIGSEGV
    lines = _log(folder)
    record = read_status_file(folder)
    assert lines[0] == f"# FlexTool run {record['run_id']} started {record['started_at']}"
    assert "python line ä" in lines
    assert "native line" in lines  # os.write(1, ...): not through sys.stdout
    assert "Fatal Python error: Segmentation fault" in lines  # faulthandler writes to fd 2
    assert not any(line.startswith("##flextool-run-record##") for line in lines)
    assert record["state"] == "crashed"
    assert record["exit_code"] == -signal.SIGSEGV
    assert record["error"]["type"] == "Crashed"
    assert [s["state"] for s in record["solves"]] == ["error"]
    # The console got the child's lines, but not the marker.
    echoed = capsys.readouterr().out
    assert "native line" in echoed and "##flextool-run-record##" not in echoed


def test_exit_0_with_a_running_record_is_succeeded_but_incomplete(tmp_path):
    folder = tmp_path / "out"
    assert supervisor.run_supervised([str(folder), "exit0"], child_cmd=_stub_cmd()) == 0
    record = read_status_file(folder)
    assert record["state"] == "succeeded"
    assert record["error"]["type"] == "RecordIncomplete"
    assert record["exit_code"] == 0


@posix_only
def test_a_child_killed_by_sigkill_is_killed_and_the_supervisor_dies_by_sigkill(tmp_path):
    folder = tmp_path / "out"
    code = (
        "import sys, flextool_run_supervisor as s; "
        f"s.main(sys.argv[1:], child_cmd=[sys.executable, {str(STUB)!r}])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(folder), "sigkill"], cwd=str(REPO), capture_output=True, timeout=120,
    )
    assert result.returncode == -signal.SIGKILL, result.stderr
    record = read_status_file(folder)
    assert record["state"] == "killed"
    assert "SIGKILL" in record["error"]["message"]
    assert "about to be killed" in _log(folder)


@posix_only
def test_a_second_ctrl_c_kills_the_child_and_the_run_is_interrupted(tmp_path):
    folder = tmp_path / "out"
    code = (
        "import sys, flextool_run_supervisor as s; "
        f"s.main(sys.argv[1:], child_cmd=[sys.executable, {str(STUB)!r}])"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code, str(folder), "stubborn"], cwd=str(REPO),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    try:
        assert proc.stdout.readline().strip() == b"sleeping"  # the child ignores SIGINT from here on
        proc.send_signal(signal.SIGINT)
        time.sleep(0.5)  # two signals before the handler runs would count as one
        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=60) == -signal.SIGINT
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close()
    record = read_status_file(folder)
    assert record["state"] == "interrupted"
    assert record["exit_code"] == -signal.SIGKILL
    assert record["error"]["message"].startswith("second SIGINT")


def test_the_launcher_is_added_only_when_the_command_line_names_none():
    assert supervisor.with_launcher(["db", "--threads", "2"]) == ["db", "--threads", "2", "--run-launcher",
                                                                  "flextool-run"]
    assert supervisor.with_launcher(["db", "--run-launcher", "calibrator"]) == ["db", "--run-launcher", "calibrator"]
    assert supervisor.with_launcher(["db", "--run-launcher=netload"]) == ["db", "--run-launcher=netload"]


def test_the_child_environment_adds_the_pipe_variables_and_leaves_os_environ_alone(monkeypatch):
    monkeypatch.delenv("FLEXTOOL_RUN_LOG_OWNER", raising=False)
    env = supervisor.child_environment({"FLEXTOOL_THREADS": "3"})
    assert env == {
        "FLEXTOOL_THREADS": "3", "FLEXTOOL_RUN_LOG_OWNER": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
    }
    supervisor.child_environment()
    assert "FLEXTOOL_RUN_LOG_OWNER" not in os.environ


@pytest.mark.parametrize(
    ("returncode", "plat", "state", "reason_part"),
    [
        (-9, "linux", "killed", "SIGKILL"),
        (-15, "linux", "interrupted", "SIGTERM"),
        (-1, "linux", "interrupted", "SIGHUP"),
        (-2, "linux", "interrupted", "SIGINT"),
        (-11, "linux", "crashed", "SIGSEGV"),
        (-4, "darwin", "crashed", "SIGILL"),
        (-10, "linux", "killed", "SIGUSR1"),
        (0, "linux", "succeeded", "without finishing"),
        (1, "linux", "failed", "exit code 1"),
        (0xC000013A, "win32", "interrupted", "Ctrl-C"),
        (0xC000013A - 2**32, "win32", "interrupted", "Ctrl-C"),
        (0xC0000005, "win32", "crashed", "access violation"),
        (3, "win32", "failed", "exit code 3"),
        (None, "linux", "crashed", "no exit code"),
    ],
)
def test_classify_exit(returncode, plat, state, reason_part):
    got_state, reason = classify_exit(returncode, plat)
    assert got_state == state
    assert reason_part in reason


def test_the_supervisor_does_not_import_flextool_or_polars():
    code = (
        "import json, sys, flextool_run_supervisor as s\n"
        "s.load_owner(); s.load_env_check()\n"
        "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in ('flextool', 'polars'))))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=str(REPO), capture_output=True, text=True, timeout=120, check=True,
    )
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []
