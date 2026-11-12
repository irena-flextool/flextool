"""The log of a commercial solver: shown as the solver writes it, once, and stopped with the solver (item 238).

The solver programs are POSIX stub scripts (no solver or licence needed), as in
``test_subprocess_solver_cli.py``.
"""
from __future__ import annotations

import io
import logging
import stat
import sys
import textwrap
import time
from pathlib import Path

import psutil
import pytest

from flextool.engine.solve import _solver_dispatch as dispatch
from flextool.engine.solve import _subprocess_solve as sps
from flextool.engine.input._solve_config import SolverConfig
from flextool.engine.solve._solver_dispatch import FlexToolUserError
from flextool.run_options import RunOptions
from tests.engine.test_subprocess_gurobi_name_recovery import _cost_lp

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="solver stubs are POSIX shebang scripts")

# A Gurobi JSON result of the 3-column ``_cost_lp``: status 2 (optimal), objective 7.5.
_GUROBI_OK = '{"SolutionInfo": {"Status": 2, "ObjVal": 7.5, "SolCount": 1}, "Vars": [{"VarName": "C0000001", "X": 10}]}'


def _stub(tmp_path: Path, body: str, name: str = "solver_stub") -> Path:
    path = tmp_path / name
    path.write_text("#!" + sys.executable + "\n" + textwrap.dedent(body))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _gurobi_stub(tmp_path: Path, before: str = "", exit_code: int = 0, result: str | None = _GUROBI_OK) -> Path:
    """A ``gurobi_cl`` stand-in: runs *before* (Python), writes *result* to its ``ResultFile=``, exits."""
    head = 'import sys\nresult = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("ResultFile=")][0]\n'
    tail = "" if result is None else f"with open(result, 'w') as fh:\n    fh.write({result!r})\n"
    return _stub(tmp_path, head + textwrap.dedent(before) + "\n" + tail + f"sys.exit({exit_code})\n")


def _solve(tmp_path: Path, monkeypatch, stub: Path, *, options=None, run_options=None, work=True):
    monkeypatch.setattr(sps, "_find_solver_binary", lambda name: stub)
    return sps._solve_commercial_subprocess(
        _cost_lp(["A", "B", "C"]), "gurobi", options, solve_name="y2030 roll", logger=logging.getLogger("t"),
        work_folder=tmp_path / "work" if work else None, run_options=run_options or RunOptions.from_env({}),
    )


class _SentinelSink(io.StringIO):
    """A ``sys.stdout`` that creates *sentinel* as soon as a line holding *marker* is written to it."""

    def __init__(self, sentinel: Path, marker: str) -> None:
        super().__init__()
        self._sentinel, self._marker = sentinel, marker

    def write(self, text: str) -> int:
        if self._marker in text:
            self._sentinel.touch()
        return super().write(text)


def test_log_reaches_the_console_while_the_solver_runs(tmp_path: Path, monkeypatch) -> None:
    """The stub prints LIVE-1 and then waits for the sentinel the console creates on seeing it; it fails (exit 3)
    if LIVE-1 only arrives after it exits, as with a captured log."""
    sentinel = tmp_path / "seen"
    stub = _gurobi_stub(tmp_path, before=f"""\
        import os, time
        print("LIVE-1", flush=True)
        for _ in range(100):
            if os.path.exists({str(sentinel)!r}):
                break
            time.sleep(0.1)
        else:
            sys.exit(3)
        """)
    sink = _SentinelSink(sentinel, "LIVE-1")
    monkeypatch.setattr(sys, "stdout", sink)
    sol = _solve(tmp_path, monkeypatch, stub)
    assert sol.shim_status == "OPTIMAL"
    assert sentinel.exists()


def test_log_is_printed_once_after_the_marker_and_saved(tmp_path: Path, monkeypatch, capsys) -> None:
    stub = _gurobi_stub(tmp_path, before='print("UNIQUE-LOG-LINE"); sys.stderr.write("prompt> ")')
    sol = _solve(tmp_path, monkeypatch, stub)
    out = capsys.readouterr().out
    assert sol.shim_status == "OPTIMAL" and sol.obj == pytest.approx(7.5)
    assert out.count("UNIQUE-LOG-LINE") == 1
    assert out.index(f"Running Gurobi ({stub})") < out.index("UNIQUE-LOG-LINE")
    # A last line without a newline (a console prompt) ends with one, so the next output starts its own line.
    assert "prompt> \n" in out
    log = tmp_path / "work" / "solve_data" / "subprocess" / "y2030_roll.log"
    assert log.read_text(encoding="utf-8") == "UNIQUE-LOG-LINE\nprompt> \n"


def test_failed_solve_shows_its_log_once(tmp_path: Path, monkeypatch, capsys, caplog) -> None:
    """Through ``run_one_solve``: the log line is on the console once and in no message."""
    stub = _gurobi_stub(tmp_path, before='print("UNIQUE-FAIL-LINE")', exit_code=7, result=None)
    monkeypatch.setattr(sps, "_find_solver_binary", lambda name: stub)
    with caplog.at_level(logging.DEBUG), pytest.raises(FlexToolUserError) as err:
        dispatch.run_one_solve(
            _cost_lp(["A", "B", "C"]), SolverConfig(name="gurobi"), logger=logging.getLogger("t"),
            solve_name="y2030", work_folder=tmp_path / "work", run_options=RunOptions.from_env({}),
            commercial_options=[],
        )
    text = capsys.readouterr().out + caplog.text + str(err.value)
    assert text.count("UNIQUE-FAIL-LINE") == 1
    assert "returncode=7" in str(err.value)
    assert "Its log is above and in" in str(err.value)
    # run_one_solve no longer logs the error itself (cmd_run_flextool logs every user error once).
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def _gone(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def _wait_gone(pid: int, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if _gone(pid):
            return True
        time.sleep(0.05)
    return _gone(pid)


def test_watchdog_stops_solver_and_its_child_and_keeps_the_log(tmp_path: Path, monkeypatch, capsys) -> None:
    """TimeLimit 0.5 s with no grace gives a 1 s watchdog: the solver and the process it started are stopped,
    and the log so far is kept."""
    monkeypatch.setattr(dispatch, "_WATCHDOG_GRACE_S", 0.0)
    pids = tmp_path / "pids"
    stub = _gurobi_stub(tmp_path, before=f"""\
        import os, subprocess, time
        print("PARTIAL-LOG", flush=True)
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        with open({str(pids)!r}, "w") as fh:
            fh.write(f"{{os.getpid()}} {{child.pid}}")
        time.sleep(60)
        """)
    t0 = time.monotonic()
    with pytest.raises(RuntimeError) as err:
        _solve(tmp_path, monkeypatch, stub, options={"TimeLimit": 0.5})
    assert time.monotonic() - t0 < 15.0
    msg = str(err.value)
    assert "stopped it after 1 s (2 x the limit + 0 s)" in msg
    assert "PARTIAL-LOG" in capsys.readouterr().out
    solver_pid, child_pid = (int(x) for x in pids.read_text().split())
    assert _wait_gone(solver_pid) and _wait_gone(child_pid)


def test_orphan_holding_the_log_open_is_stopped(tmp_path: Path, monkeypatch, caplog) -> None:
    """The solver starts a sleeper that inherits its output and exits 0: the solve returns after the bounded
    wait, warns once, parses the status and stops the sleeper."""
    monkeypatch.setattr(sps, "_READER_JOIN_S", 1.0)
    pid_file = tmp_path / "sleeper"
    stub = _gurobi_stub(tmp_path, before=f"""\
        import subprocess
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        with open({str(pid_file)!r}, "w") as fh:
            fh.write(str(child.pid))
        print("ORPHAN-TEST", flush=True)
        """)
    t0 = time.monotonic()
    with caplog.at_level(logging.WARNING):
        sol = _solve(tmp_path, monkeypatch, stub)
    assert time.monotonic() - t0 < 15.0
    assert sol.shim_status == "OPTIMAL"
    assert len([r for r in caplog.records if "stayed open" in r.getMessage()]) == 1
    assert _wait_gone(int(pid_file.read_text()))


def test_silent_hides_the_log_but_reads_and_saves_it(tmp_path: Path, monkeypatch, capsys, caplog) -> None:
    stub = _gurobi_stub(tmp_path, before='print("UNIQUE-LOG-LINE")')
    opts = RunOptions.from_env({"FLEXTOOL_SOLVER_LOG_LEVEL": "silent"})
    sol = _solve(tmp_path, monkeypatch, stub, run_options=opts)
    out = capsys.readouterr().out
    assert sol.shim_status == "OPTIMAL"
    assert "UNIQUE-LOG-LINE" not in out and "Running Gurobi" not in out
    log = tmp_path / "work" / "solve_data" / "subprocess" / "y2030_roll.log"
    assert "UNIQUE-LOG-LINE" in log.read_text(encoding="utf-8")

    # silent is honoured on a commercial solve, so it is not a HiGHS-only option; verbose still is.
    dispatch.reset_run_warnings()
    with caplog.at_level(logging.WARNING):
        dispatch.warn_highs_only_options("gurobi", opts, None)
        assert not caplog.records
        dispatch.warn_highs_only_options("gurobi", RunOptions.from_env({"FLEXTOOL_SOLVER_LOG_LEVEL": "verbose"}),
                                         None)
    assert [r.getMessage() for r in caplog.records if "FLEXTOOL_SOLVER_LOG_LEVEL=verbose" in r.getMessage()]


def test_silent_failure_message_carries_the_end_of_the_log(tmp_path: Path, monkeypatch, capsys) -> None:
    stub = _gurobi_stub(tmp_path, before='print("\\n".join(f"line {i}" for i in range(30)))', exit_code=2,
                        result=None)
    opts = RunOptions.from_env({"FLEXTOOL_SOLVER_LOG_LEVEL": "silent"})
    with pytest.raises(RuntimeError) as err:
        _solve(tmp_path, monkeypatch, stub, run_options=opts)
    msg = str(err.value)
    assert "line 29" in msg and "line 10" in msg and "line 9\n" not in msg  # the last 20 lines
    assert "y2030_roll.log" in msg
    assert "line 0" not in capsys.readouterr().out


def test_undecodable_bytes_do_not_stop_the_solve(tmp_path: Path, monkeypatch, capsys) -> None:
    stub = _gurobi_stub(tmp_path, before='sys.stdout.buffer.write(b"\\xff\\xfe ok\\n"); sys.stdout.flush()')
    sol = _solve(tmp_path, monkeypatch, stub)
    assert sol.shim_status == "OPTIMAL"
    assert " ok" in capsys.readouterr().out


def test_no_work_folder_means_no_log_file(tmp_path: Path, monkeypatch, capsys) -> None:
    stub = _gurobi_stub(tmp_path, before='print("TEMP-LOG")')
    sol = _solve(tmp_path, monkeypatch, stub, work=False)
    assert sol.shim_status == "OPTIMAL"
    assert "TEMP-LOG" in capsys.readouterr().out
    assert not (tmp_path / "work").exists()
