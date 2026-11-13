"""Item 240 B2: a solver wrapper set on this computer starts in front of a commercial solver's program.

The wrapper comes from ``FLEXTOOL_SOLVER_WRAPPER_<SOLVER>`` (a JSON list, or one program path) or from ``wrappers``
in this computer's ``solver_settings.json``, never from a database, a project or a flag.  Its first item is the
full path of a program; scripts that Windows runs through a command interpreter and the interpreters themselves
are refused.  The solver's own command (absolute program path, then its arguments) follows the wrapper's items;
stdin, the live log and the stop of the process tree work through it.

The solver programs are POSIX stub scripts, as in ``test_commercial_solver_log.py``; the run through the real
solver packages is in ``tests/engine/commercial_shim``.  Expected values follow from those rules by hand.
"""
from __future__ import annotations

import json
import os
import stat
import sys
import time
from pathlib import Path

import pytest

from flextool import _local_solver_settings as settings
from flextool._local_solver_settings import LocalSolverSettingsError, check_wrapper, solver_wrapper
from flextool.engine.solve import _solver_dispatch as dispatch
from flextool.engine.solve._solver_dispatch import FlexToolUserError
from tests.engine.test_commercial_solver_log import _gurobi_stub, _solve, _stub, _wait_gone

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="the stubs are POSIX shebang scripts")

WRAPPER_ENV = ("FLEXTOOL_SOLVER_WRAPPER_GUROBI", "FLEXTOOL_SOLVER_WRAPPER_CPLEX", "FLEXTOOL_SOLVER_WRAPPER_XPRESS",
               "FLEXTOOL_SOLVER_WRAPPER_COPT")


@pytest.fixture(autouse=True)
def user_home(tmp_path, monkeypatch) -> Path:
    home = tmp_path / "user_home"
    monkeypatch.setenv("FLEXTOOL_USER_HOME", str(home))
    for name in WRAPPER_ENV:
        monkeypatch.delenv(name, raising=False)
    return home


def _program(folder: Path, name: str, executable: bool = True) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text("#!/bin/sh\nexec \"$@\"\n")
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _recording_wrapper(tmp_path: Path) -> tuple[Path, Path]:
    """A wrapper that writes its argv (JSON) to a file, takes its own ``--queue NAME`` option, then runs the rest
    as the command, as the docs say: a Python script that ``subprocess.run`` its remaining arguments and returns the
    exit code (stdin passes through)."""
    record = tmp_path / "wrapper_argv.json"
    wrapper = _stub(tmp_path, f"""\
        import json, os, subprocess, sys
        with open({str(record)!r}, "w") as fh:
            json.dump({{"argv": sys.argv, "pid": os.getpid()}}, fh)
        command = sys.argv[1:]
        if command[:1] == ["--queue"]:
            command = command[2:]
        sys.exit(subprocess.run(command).returncode)
        """, name="lic_hold")
    return wrapper, record


# ---------------------------------------------------------------------------
# The rules of check_wrapper
# ---------------------------------------------------------------------------


@posix_only
@pytest.mark.parametrize(("make", "windows", "problem"), [
    (lambda d: ["relative/lic_hold"], False, "is not a full path"),
    (lambda d: [str(d / "missing")], False, "does not exist"),
    (lambda d: [str(_program(d, "x").parent)], False, "is not a file"),
    (lambda d: [str(_program(d, "plain", executable=False))], False, "is not executable"),
    (lambda d: [str(_program(d, "hold.bat"))], False, "a script that Windows runs through a command interpreter"),
    (lambda d: [str(_program(d, "HOLD.CMD"))], True, "a script that Windows runs through a command interpreter"),
    (lambda d: [str(_program(d, "hold.ps1"))], True, "a script that Windows runs"),
    (lambda d: [str(_program(d, "hold.vbs"))], True, "a script that Windows runs"),
    (lambda d: [str(_program(d, "hold.js"))], False, "a script that Windows runs"),
    (lambda d: [str(_program(d, "cmd.exe"))], True, "is a command interpreter"),
    (lambda d: [str(_program(d, "PowerShell.exe")), "-NoProfile", "-File", "x.ps1"], True, "is a command interpreter"),
    (lambda d: [str(_program(d, "pwsh"))], False, "is a command interpreter"),
    (lambda d: [str(_program(d, "wscript.exe"))], True, "is a command interpreter"),
    (lambda d: [str(_program(d, "cscript.exe"))], True, "is a command interpreter"),
    (lambda d: [str(_program(d, "hold"))], True, "on Windows the wrapper must be a program (.exe)"),
    (lambda d: [str(_program(d, "hold"))] + ["a"] * 32, False, "a list of 1 to 32 texts"),
    (lambda d: [], False, "a list of 1 to 32 texts"),
    (lambda d: str(_program(d, "hold")), False, "a list of 1 to 32 texts"),
    (lambda d: [str(_program(d, "hold")), "a\x00b"], False, "NUL"),
])
def test_refused_wrappers(tmp_path, make, windows, problem) -> None:
    with pytest.raises(LocalSolverSettingsError) as err:
        check_wrapper(make(tmp_path / "bin"), source="SRC", windows=windows)
    assert str(err.value).startswith("The solver wrapper from SRC") and problem in str(err.value)


@posix_only
@pytest.mark.parametrize("windows", [False, True])
def test_a_link_to_a_script_is_refused_by_its_target(tmp_path, windows) -> None:
    """A wrapper named like a program (``hold.exe``) that is a symbolic link to a ``.bat``: refused by the
    resolved name."""
    target = _program(tmp_path / "bin", "real.bat")
    link = tmp_path / "bin" / "hold.exe"
    link.symlink_to(target)
    with pytest.raises(LocalSolverSettingsError, match=r"real\.bat is a script"):
        check_wrapper([str(link)], source="SRC", windows=windows)


@posix_only
def test_a_program_with_arguments_passes(tmp_path) -> None:
    hold = _program(tmp_path / "bin", "hold")
    assert check_wrapper([str(hold), "--queue", "opt", ""], source="SRC", windows=False) == (
        str(hold), "--queue", "opt", "")
    exe = _program(tmp_path / "bin", "hold.exe")
    assert check_wrapper([str(exe)], source="SRC", windows=True) == (str(exe),)


# ---------------------------------------------------------------------------
# Where the wrapper is read from
# ---------------------------------------------------------------------------


@posix_only
def test_the_environment_wins_over_the_settings_file(tmp_path, monkeypatch) -> None:
    from_file = _program(tmp_path / "file bin", "hold_file")
    from_env = _program(tmp_path / "env bin", "hold env")  # a space: one program path, never split
    settings.set_solver_wrapper("cplex", [str(from_file), "--queue", "opt"])
    got = solver_wrapper("cplex")
    assert (got.argv, got.source) == ((str(from_file), "--queue", "opt"), str(settings.settings_path()))
    assert solver_wrapper("gurobi") is None
    monkeypatch.setenv("FLEXTOOL_SOLVER_WRAPPER_CPLEX", json.dumps([str(from_env), "-n", "10"]))
    got = solver_wrapper("cplex")
    assert (got.argv, got.source) == ((str(from_env), "-n", "10"), "environment variable FLEXTOOL_SOLVER_WRAPPER_CPLEX")
    monkeypatch.setenv("FLEXTOOL_SOLVER_WRAPPER_CPLEX", f" {from_env} ")
    assert solver_wrapper("cplex").argv == (str(from_env),)
    for off in ("[]", "", "  "):  # set but empty: no wrapper, also over the file
        monkeypatch.setenv("FLEXTOOL_SOLVER_WRAPPER_CPLEX", off)
        assert solver_wrapper("cplex") is None
    monkeypatch.setenv("FLEXTOOL_SOLVER_WRAPPER_CPLEX", "[not json")
    with pytest.raises(LocalSolverSettingsError, match="FLEXTOOL_SOLVER_WRAPPER_CPLEX is not a valid JSON list"):
        solver_wrapper("cplex")


@posix_only
def test_a_wrapper_in_the_project_or_current_folder_is_never_read(tmp_path, monkeypatch) -> None:
    """A settings file that came with a workspace (in the current folder, its ``solver_config/``, a ``config/``
    folder) does not set a wrapper: only this computer's user config folder does."""
    hold = _program(tmp_path / "bin", "hold")
    planted = json.dumps({"format": 1, "wrappers": {"gurobi": [str(hold)]}})
    ws = tmp_path / "ws"
    for place in (ws, ws / "solver_config", ws / "config"):
        place.mkdir(parents=True, exist_ok=True)
        (place / "solver_settings.json").write_text(planted)
    monkeypatch.chdir(ws)
    assert solver_wrapper("gurobi") is None


@posix_only
def test_the_solver_program_is_found_only_in_absolute_path_folders(tmp_path, monkeypatch) -> None:
    """A relative ``PATH`` entry (``.``, ``bin``) depends on the current folder, a workspace that may have come
    from someone else: skipped.  The program found is an absolute path."""
    from flextool.engine.solve._subprocess_solve import _find_solver_binary

    ws = tmp_path / "ws"
    planted = _program(ws / "bin", "gurobi_cl")
    _program(ws, "gurobi_cl")
    real = _program(tmp_path / "opt" / "gurobi", "gurobi_cl")
    monkeypatch.chdir(ws)
    monkeypatch.setenv("PATH", os.pathsep.join([".", "bin", "", str(tmp_path / "opt" / "gurobi")]))
    found = _find_solver_binary("gurobi")
    assert found == real and found.is_absolute() and found != planted


# ---------------------------------------------------------------------------
# A solve through the wrapper
# ---------------------------------------------------------------------------


class _Recorder:
    """A stand-in for the run recorder: keeps ``note_solver_wrapper`` calls, ignores the rest."""

    def __init__(self) -> None:
        self.wrappers: dict[str, dict] = {}

    def note_solver_wrapper(self, solver, record) -> None:
        self.wrappers.setdefault(solver, record)

    def __getattr__(self, name):
        return lambda *a, **k: None


@posix_only
def test_a_solve_runs_through_the_wrapper(tmp_path, monkeypatch, capsys) -> None:
    """The wrapper gets its own items, then the solver's absolute path and the solver's arguments; the solve's
    result is read as without a wrapper; the marker line names the wrapper and where it was set, and the run record
    keeps it."""
    from flextool import run_record

    wrapper, record = _recording_wrapper(tmp_path)
    stub = _gurobi_stub(tmp_path, before='print("GUROBI-LOG-LINE", flush=True)')
    monkeypatch.setenv("FLEXTOOL_SOLVER_WRAPPER_GUROBI", json.dumps([str(wrapper), "--queue", "opt"]))
    recorder = _Recorder()
    monkeypatch.setattr(run_record, "active", lambda: recorder)
    sol = _solve(tmp_path, monkeypatch, stub, options={"Threads": 2})
    assert sol.obj == 7.5
    argv = json.loads(record.read_text())["argv"]
    assert argv[:4] == [str(wrapper), "--queue", "opt", str(stub)]
    assert argv[4] == "Threads=2" and argv[5].startswith("ResultFile=") and argv[6].endswith(".mps")
    out = capsys.readouterr().out
    marker = (f"Running Gurobi ({stub}) through the solver wrapper "
              f"{json.dumps([str(wrapper), '--queue', 'opt'])} (from environment variable "
              "FLEXTOOL_SOLVER_WRAPPER_GUROBI)")
    assert marker in out.splitlines()
    assert "GUROBI-LOG-LINE" in out.split(marker, 1)[1]
    assert recorder.wrappers == {"gurobi": {"argv": [str(wrapper), "--queue", "opt"],
                                            "source": "environment variable FLEXTOOL_SOLVER_WRAPPER_GUROBI"}}


@posix_only
def test_the_watchdog_stops_the_wrapper_the_solver_and_its_child(tmp_path, monkeypatch, capsys) -> None:
    """A wrapper that keeps running while the solver runs under it: the 1 s watchdog stops the wrapper, the solver
    and the process the solver started, keeps the log, and the message names the wrapper."""
    monkeypatch.setattr(dispatch, "_WATCHDOG_GRACE_S", 0.0)
    wrapper, record = _recording_wrapper(tmp_path)
    pids = tmp_path / "pids"
    stub = _gurobi_stub(tmp_path, before=f"""\
        import os, subprocess, time
        print("PARTIAL-LOG", flush=True)
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        with open({str(pids)!r}, "w") as fh:
            fh.write(f"{{os.getpid()}} {{child.pid}}")
        time.sleep(60)
        """)
    monkeypatch.setenv("FLEXTOOL_SOLVER_WRAPPER_GUROBI", str(wrapper))
    t0 = time.monotonic()
    with pytest.raises(RuntimeError) as err:
        _solve(tmp_path, monkeypatch, stub, options={"TimeLimit": 0.5})
    assert time.monotonic() - t0 < 15.0
    assert f"(started through the solver wrapper {wrapper})" in str(err.value)
    assert "PARTIAL-LOG" in capsys.readouterr().out
    wrapper_pid = json.loads(record.read_text())["pid"]
    solver_pid, child_pid = (int(x) for x in pids.read_text().split())
    assert _wait_gone(wrapper_pid) and _wait_gone(solver_pid) and _wait_gone(child_pid)


@posix_only
def test_an_unusable_wrapper_stops_the_solve_before_the_model_is_written(tmp_path, monkeypatch) -> None:
    stub = _gurobi_stub(tmp_path)
    monkeypatch.setenv("FLEXTOOL_SOLVER_WRAPPER_GUROBI", "relative/lic_hold")
    with pytest.raises(FlexToolUserError, match="FLEXTOOL_SOLVER_WRAPPER_GUROBI: 'relative/lic_hold' is not a full"):
        _solve(tmp_path, monkeypatch, stub)
    assert not list((tmp_path / "work").rglob("*.mps"))


def test_export_keeps_the_wrapper_program_and_counts_its_arguments(tmp_path) -> None:
    from flextool.project_package import scrub_run_status

    status = {"state": "finished", "solver_wrappers": {
        "cplex": {"argv": ["/opt/lic/hold", "--token", "s3cret", "--queue"], "source": "environment variable X"}}}
    scrubbed = scrub_run_status(status, tmp_path)
    entry = scrubbed["solver_wrappers"]["cplex"]
    assert entry["arguments_left_out"] == 3 and len(entry["argv"]) == 1
    assert "s3cret" not in json.dumps(scrubbed)
