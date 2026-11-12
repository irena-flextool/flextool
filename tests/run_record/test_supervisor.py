"""The ``flextool-run`` supervisor owns its child's output pipe and finishes a run record the child left unfinished.

The child is ``_stub_engine.py``: it writes a ``running`` status and the log header, prints the marker like the
engine with a pipe owner, and then crashes, kills itself or exits (see its docstring).  What reaches ``_run.log``
through the pipe -- a line written straight to fd 1, faulthandler's crash dump -- is what only the pipe owner
sees (``log_capture: "parent"``).
"""
from __future__ import annotations

import enum
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
        "PYTHONSAFEPATH": "1",
    }
    supervisor.child_environment()
    assert "FLEXTOOL_RUN_LOG_OWNER" not in os.environ


def test_the_engine_does_not_get_the_stop_file_variable():
    env = supervisor.child_environment({"FLEXTOOL_RUN_STOP_FILE": "/somewhere/stop", "HOME": "/h"})
    assert "FLEXTOOL_RUN_STOP_FILE" not in env and env["HOME"] == "/h"


_SUPERVISOR_CODE = (
    "import sys, flextool_run_supervisor as s; "
    f"s.main(sys.argv[1:], child_cmd=[sys.executable, {str(STUB)!r}])"
)


def test_a_stop_file_kills_the_engine_and_records_the_run_as_killed(tmp_path):
    folder = tmp_path / "out"
    stop_file = tmp_path / "stop"
    env = {**os.environ, "FLEXTOOL_RUN_STOP_FILE": str(stop_file)}
    proc = subprocess.Popen(
        [sys.executable, "-c", _SUPERVISOR_CODE, str(folder), "sleep"], cwd=str(REPO), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    try:
        assert proc.stdout.readline().strip() == b"sleeping"
        stop_file.write_text("stopped from the FlexTool window", encoding="utf-8")
        assert proc.wait(timeout=60) == 1
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close()
    record = read_status_file(folder)
    assert record["state"] == "killed"
    assert record["error"] == {"type": "Killed", "message": "stopped from the FlexTool window"}
    assert [s["state"] for s in record["solves"]] == ["error"]


def test_a_stop_file_present_at_start_runs_nothing(tmp_path, monkeypatch):
    folder = tmp_path / "out"
    stop_file = tmp_path / "stop"
    stop_file.write_text("FlexTool was closed", encoding="utf-8")
    monkeypatch.setenv("FLEXTOOL_RUN_STOP_FILE", str(stop_file))

    assert supervisor.run_supervised([str(folder), "exit0"], child_cmd=_stub_cmd()) == 1
    assert not folder.exists()


@pytest.mark.parametrize(
    ("returncode", "plat", "state", "reason_part"),
    [
        # A signal is named, and so classified, by this computer's signal table (Windows has neither SIGKILL,
        # SIGHUP nor SIGUSR1, and numbers 1 CTRL_BREAK_EVENT; macOS numbers 10 SIGBUS and SIGUSR1 30): CI of PR #362.
        pytest.param(-9, "linux", "killed", "SIGKILL", marks=posix_only),
        (-15, "linux", "interrupted", "SIGTERM"),
        pytest.param(-1, "linux", "interrupted", "SIGHUP", marks=posix_only),
        (-2, "linux", "interrupted", "SIGINT"),
        (-11, "linux", "crashed", "SIGSEGV"),
        (-4, "darwin", "crashed", "SIGILL"),
        pytest.param(-getattr(signal, "SIGUSR1", 10), sys.platform, "killed", "SIGUSR1", marks=posix_only),
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


# macOS numbering (signal.h of Darwin) and Linux numbering (x86-64 / arm64) of the signals the classification uses.
_MACOS_SIGNALS = enum.IntEnum("Signals", {
    "SIGHUP": 1, "SIGINT": 2, "SIGILL": 4, "SIGABRT": 6, "SIGEMT": 7, "SIGFPE": 8, "SIGKILL": 9, "SIGBUS": 10,
    "SIGSEGV": 11, "SIGTERM": 15, "SIGUSR1": 30,
})
_LINUX_SIGNALS = enum.IntEnum("Signals", {
    "SIGHUP": 1, "SIGINT": 2, "SIGILL": 4, "SIGABRT": 6, "SIGBUS": 7, "SIGFPE": 8, "SIGKILL": 9, "SIGUSR1": 10,
    "SIGSEGV": 11, "SIGTERM": 15,
})


@pytest.mark.parametrize(
    ("table", "returncode", "state", "reason"),
    [
        (_MACOS_SIGNALS, -10, "crashed", "SIGBUS (bus error)"),
        (_MACOS_SIGNALS, -7, "killed", "terminated by SIGEMT"),
        (_MACOS_SIGNALS, -30, "killed", "terminated by SIGUSR1"),
        (_LINUX_SIGNALS, -7, "crashed", "SIGBUS (bus error)"),
        (_LINUX_SIGNALS, -10, "killed", "terminated by SIGUSR1"),
        (_LINUX_SIGNALS, -30, "killed", "terminated by signal 30"),
    ],
)
def test_classify_exit_names_signals_by_this_computers_numbering(monkeypatch, table, returncode, state, reason):
    """Item 215: SIGBUS is 10 on macOS (7 on Linux), so a macOS SIGBUS crash is ``crashed``, not ``killed``."""
    monkeypatch.setattr(signal, "Signals", table)
    assert classify_exit(returncode, "darwin" if table is _MACOS_SIGNALS else "linux") == (state, reason)
    assert supervisor._signal_name(-returncode) == reason.split(" (")[0].removeprefix("terminated by ")


def test_the_supervisor_does_not_import_flextool_or_polars():
    code = (
        "import json, sys, flextool_run_supervisor as s\n"
        "owner = s.load_owner(); env_check = s.load_env_check()\n"
        "assert owner.classify_exit(-11, 'linux')[0] == 'crashed' and owner._load_env_check() is env_check\n"
        "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in ('flextool', 'polars'))))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=str(REPO), capture_output=True, text=True, timeout=120, check=True,
    )
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []


# ---------------------------------------------------------------------------
# A stop request and the child's own end
# ---------------------------------------------------------------------------


class _FakeChild:
    """A child process that has ended with *returncode* (``None``: still running)."""

    pid = -1  # never signalled: the tests use the Windows stopper, which kills through ``kill``

    def __init__(self, returncode: int | None) -> None:
        self.returncode = returncode
        self.killed = False
        self.signals: list[int] = []

    def poll(self) -> int | None:
        return self.returncode

    def kill(self) -> None:
        self.killed = True

    def send_signal(self, signum: int) -> None:
        self.signals.append(signum)


@pytest.mark.parametrize("windows", [False, True])
def test_a_stop_after_the_child_ended_keeps_the_childs_exit_code(windows):
    stopper = supervisor._Stopper(windows=windows)
    child = _FakeChild(returncode=0)
    stopper.attach(child)
    stopper.handle(signal.SIGINT)
    stopper.handle(signal.SIGINT)  # the second Ctrl-C comes after the run had ended 0
    stopper.stop("stopped by the GUI")
    assert stopper.outcome is None and not stopper.stopped_by_launcher and not child.killed
    assert stopper.exit_status(0) == 0


def test_a_second_ctrl_c_before_the_child_exists_kills_it_when_it_starts():
    stopper = supervisor._Stopper(windows=True)
    stopper.handle(signal.SIGINT)
    stopper.handle(signal.SIGINT)
    child = _FakeChild(returncode=None)
    stopper.attach(child)
    assert child.killed
    assert stopper.outcome == ("interrupted", "second SIGINT: the run was killed")
    assert stopper.exit_status(1) == -1073741510  # STATUS_CONTROL_C_EXIT 0xC000013A as a signed 32-bit number


def test_the_supervisor_takes_the_signal_names_and_exit_codes_from_the_owner():
    owner = supervisor.load_owner()
    assert owner.WINDOWS_CTRL_C_EXIT_SIGNED == 0xC000013A - 2**32 == -1073741510
    assert supervisor._signal_name(signal.SIGTERM) == owner.signal_name(signal.SIGTERM) == "SIGTERM"
    assert owner.signal_name(None) == "signal None"


def test_inside_the_package_the_owner_classifies_exits_with_the_packages_env_check():
    import importlib

    from flextool.run_record import _owner

    assert _owner._load_env_check() is importlib.import_module("flextool.env_check")
