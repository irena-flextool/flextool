"""The run's thread budget: read from the command line before polars is imported, and applied to polars and HiGHS.

The pure parts (which command lines carry ``--threads``, the polars formula, the environment restore, the HiGHS
keys) are checked in this process.  The pre-import application can only be seen in a fresh interpreter, because
polars fixes its pool size when it is first imported, so those cases run ``python -c`` in separate processes.
"""
from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from flextool import _thread_budget as tb
from flextool.run_options import OPTIONS, RunOptions


# ---------------------------------------------------------------------------
# Which command lines carry --threads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "orig_argv", "expected"),
    [
        (["flextool-run", "db.sqlite", "--threads", "4"], [], 4),
        (["/venv/bin/flextool-run", "--threads=3", "db.sqlite"], [], 3),
        (["C:\\venv\\Scripts\\flextool-run.exe", "--threads", "2"], [], 2),
        # ``python -m``: sys.argv[0] is "-m" while the package __init__ runs; the module is in sys.orig_argv.
        (["-m", "db.sqlite", "--threads", "5"], ["python", "-m", "flextool.cli.cmd_run_flextool", "db.sqlite",
                                                 "--threads", "5"], 5),
        (["-m", "--threads", "2"], ["python", "-u", "-m", "flextool.cli.cmd_solve_mps", "--threads", "2"], 2),
        (["-m", "--threads", "6"], ["python", "-m", "flextool.calibrate", "--threads", "6"], 6),
        # Spine Toolbox runs the file as a script (subprocess or its Basic Console).
        (["/repo/flextool/cli/cmd_run_flextool.py", "sqlite:///x", "--threads", "7"], [], 7),
        # Not a FlexTool command: another program's --threads is never taken.
        (["pytest", "--threads", "4"], [], None),
        (["-m", "--threads", "4"], ["python", "-m", "pytest", "--threads", "4"], None),
        (["-c", "--threads", "4"], ["python", "-c", "import flextool", "--threads", "4"], None),
        # An abbreviation is not matched, and an invalid value counts as not given (the real parser reports it).
        (["flextool-run", "--thr", "4"], [], None),
        (["flextool-run", "--threads", "0"], [], None),
        (["flextool-run", "--threads", "many"], [], None),
        (["flextool-run", "--threads"], [], None),
        (["flextool-run", "db.sqlite"], [], None),
        # The assistant accepts the flag too.
        (["/venv/bin/flextool-assistant", "--mcp", "--threads", "2"], [], 2),
        (["-m", "--threads", "3"], ["python", "-m", "flextool_assistant_entry", "--threads", "3"], 3),
    ],
)
def test_cli_threads_from_argv(argv, orig_argv, expected):
    assert tb.cli_threads_from_argv(argv, orig_argv) == expected


@pytest.mark.parametrize(
    ("argv", "orig_argv", "expected"),
    [
        (["/venv/bin/flextool-assistant", "--mcp"], [], 1),
        (["C:\\venv\\Scripts\\flextool-assistant.exe", "--call", "list_runs"], [], 1),
        (["/repo/flextool_assistant_entry.py", "--mcp"], ["python", "-m", "flextool_assistant_entry", "--mcp"], 1),
        (["flextool-run", "db.sqlite"], [], None),
        (["pytest"], [], None),
    ],
)
def test_the_assistant_is_single_threaded_by_default(argv, orig_argv, expected):
    """P1-O10: the assistant never competes with a running model; ``--threads N`` raises its budget."""
    assert tb.command_default_threads(argv, orig_argv) == expected


# ---------------------------------------------------------------------------
# Budget and polars formula
# ---------------------------------------------------------------------------


def test_budget_precedence_flag_env_auto(monkeypatch):
    monkeypatch.setattr(tb, "available_cores", lambda: 12)
    assert tb.resolve_budget(3, {"FLEXTOOL_THREADS": "5"}) == (3, "cli")
    assert tb.resolve_budget(None, {"FLEXTOOL_THREADS": "5"}) == (5, "env")
    assert tb.resolve_budget(None, {}) == (12, "auto")
    # A malformed environment value falls through to the available cores.
    assert tb.resolve_budget(None, {"FLEXTOOL_THREADS": "0"}) == (12, "auto")


@pytest.mark.parametrize(
    ("budget", "environ", "expected"),
    [
        (32, {}, (8, "cap")),            # the cap
        (8, {}, (8, "budget")),
        (3, {}, (3, "budget")),          # the budget below the cap
        (3, {"POLARS_MAX_THREADS": "2"}, (2, "env")),      # the user's value below the budget
        (3, {"POLARS_MAX_THREADS": "6"}, (3, "budget")),   # ... but never above it
        (32, {"POLARS_MAX_THREADS": "16"}, (16, "env")),   # the user's value replaces the cap
        (32, {"POLARS_MAX_THREADS": "x"}, (8, "cap")),     # an unusable value is ignored
    ],
)
def test_polars_threads(budget, environ, expected):
    assert tb.polars_threads(budget, environ) == expected


def test_restore_env_puts_the_variable_back_exactly():
    env = {"POLARS_MAX_THREADS": "3"}
    tb.restore_env(None, env)
    assert "POLARS_MAX_THREADS" not in env
    env = {"POLARS_MAX_THREADS": "3"}
    tb.restore_env("2", env)
    assert env == {"POLARS_MAX_THREADS": "2"}
    env = {}
    tb.restore_env(None, env)
    assert env == {}


# ---------------------------------------------------------------------------
# HiGHS keys from the budget
# ---------------------------------------------------------------------------


def _opts(**flags) -> RunOptions:
    ns = {spec.dest: None for spec in OPTIONS.values() if spec.flag}
    ns.update(flags)
    return RunOptions.from_cli(SimpleNamespace(**ns), environ={})


@pytest.mark.parametrize(
    ("flags", "expected", "warns"),
    [
        ({}, {}, False),                                                  # no caller: as before
        ({"highs_threads": 1}, {}, False),
        ({"threads": 4}, {"threads": 4}, False),                          # budget: threads, parallel stays off
        ({"threads": 4, "highs_threads": 2}, {"threads": 2, "parallel": "on"}, False),
        ({"threads": 4, "highs_threads": 8}, {"threads": 4, "parallel": "on"}, True),   # clamped to the budget
        ({"threads": 1, "highs_threads": 8}, {"threads": 1}, True),
        ({"threads": 4, "highs_threads": 1}, {"threads": 1}, False),     # an explicit HiGHS 1 is kept
        ({"highs_threads": 3}, {"threads": 3, "parallel": "on"}, False),  # opt-in parallel simplex, unchanged
    ],
)
def test_highs_thread_options(flags, expected, warns):
    got, warning = tb.highs_thread_options(_opts(**flags))
    assert got == expected
    assert (warning is not None) == warns


# ---------------------------------------------------------------------------
# Pre-import application, each case in a fresh interpreter
# ---------------------------------------------------------------------------

_PROBE = (
    "import sys; sys.argv = {argv!r}; import flextool, polars, os; "
    "print(polars.thread_pool_size(), os.environ.get('POLARS_MAX_THREADS'))"
)


def _probe(argv: list[str], **env: str) -> tuple[int, str]:
    environ = {k: v for k, v in os.environ.items() if k not in ("POLARS_MAX_THREADS", "FLEXTOOL_THREADS")}
    environ.update(env)
    out = subprocess.run([sys.executable, "-c", _PROBE.format(argv=argv)], env=environ, capture_output=True,
                         text=True, timeout=180, check=True)
    size, value = out.stdout.strip().splitlines()[-1].split()
    return int(size), value


def test_threads_flag_sizes_polars_and_leaves_no_environment_behind():
    assert _probe(["flextool-run", "db.sqlite", "--threads", "3"]) == (3, "None")


def test_user_polars_max_threads_below_the_budget_wins_and_is_kept():
    assert _probe(["flextool-run", "--threads", "3"], POLARS_MAX_THREADS="2") == (2, "2")


def test_env_budget_is_capped_for_polars():
    # An explicit budget is taken as given (not reduced to the machine's cores), so FLEXTOOL_THREADS=16 gives
    # polars the cap on any runner.
    assert _probe(["flextool-run"], FLEXTOOL_THREADS="16") == (min(16, tb.THREAD_CAP_POLARS), "None")


def test_assistant_runs_polars_on_one_thread_unless_its_flag_says_otherwise():
    # FLEXTOOL_THREADS is the budget of a run; it does not raise the assistant's.
    assert _probe(["flextool-assistant", "--mcp"], FLEXTOOL_THREADS="4") == (1, "None")
    assert _probe(["flextool-assistant", "--mcp", "--threads", "2"]) == (2, "None")


def test_non_flextool_command_line_is_ignored():
    assert _probe(["pytest", "--threads", "3"], FLEXTOOL_THREADS="2") == (2, "None")


def test_polars_loaded_first_keeps_its_pool_and_the_environment():
    """polars imported before flextool (a Toolbox console, a notebook): its pool cannot change, nothing is
    written, and the decision says so (the run header prints it)."""
    code = (
        "import sys, os; import polars; before = polars.thread_pool_size(); "
        "sys.argv = ['flextool-run', '--threads', '3']; import flextool; "
        "from flextool._thread_budget import current; c = current(); "
        "print(before == polars.thread_pool_size(), os.environ.get('POLARS_MAX_THREADS'), c.polars_source, c.budget)"
    )
    environ = {k: v for k, v in os.environ.items() if k not in ("POLARS_MAX_THREADS", "FLEXTOOL_THREADS")}
    out = subprocess.run([sys.executable, "-c", code], env=environ, capture_output=True, text=True, timeout=180,
                         check=True)
    assert out.stdout.strip().splitlines()[-1].split() == ["True", "None", "already_loaded", "3"]


# ---------------------------------------------------------------------------
# A Spine Toolbox run: 1 thread for polars and every solver unless --threads / FLEXTOOL_THREADS
# ---------------------------------------------------------------------------

_TOOLBOX_FILE = "/proj/flextool/cli/cmd_run_flextool.py"


@pytest.mark.parametrize(
    ("argv", "orig_argv", "expected"),
    [
        # Toolbox subprocess: ``python cmd_run_flextool.py <args>`` in the specification's folder.
        (["cmd_run_flextool.py", "sqlite:///in.sqlite"], ["python", "cmd_run_flextool.py", "sqlite:///in.sqlite"],
         True),
        # Basic Console: ``python -i -u -c <repl>``, then ``sys.argv = [<full path>, ...]`` and the file exec-ed.
        ([_TOOLBOX_FILE, "sqlite:///in.sqlite"], ["python", "-i", "-u", "-c", "import spine_repl"], True),
        (["C:\\proj\\flextool\\cli\\CMD_RUN_FLEXTOOL.py", "x"], ["python.exe", "-i", "-u", "-c", "..."], True),
        # Jupyter console: ``%run`` in a kernel started with ``-m ipykernel_launcher``.
        ([_TOOLBOX_FILE, "x"], ["python", "-m", "ipykernel_launcher", "-f", "k.json"], True),
        # The FlexTool window, flextool-run and the workflow start the engine as ``-m``: sys.argv[0] is the
        # same file once the module runs, and still not a Toolbox run.
        ([_TOOLBOX_FILE, "x"], ["python", "-P", "-m", "flextool.cli.cmd_run_flextool", "x"], False),
        (["-m", "x"], ["python", "-m", "flextool.cli.cmd_run_flextool", "x"], False),
        (["flextool-run", "x"], [], False),
        (["/repo/run_flextool.py", "x"], ["python", "/repo/run_flextool.py", "x"], False),
        (["pytest"], ["python", "-m", "pytest"], False),
        ([], [], False),
    ],
)
def test_is_toolbox_run(argv, orig_argv, expected):
    assert tb.is_toolbox_run(argv, orig_argv) is expected


def test_toolbox_budget_precedence(monkeypatch):
    monkeypatch.setattr(tb, "available_cores", lambda: 12)
    assert tb.resolve_budget(None, {}, toolbox=True) == (1, "toolbox")
    assert tb.resolve_budget(None, {"FLEXTOOL_THREADS": "5"}, toolbox=True) == (5, "env")
    assert tb.resolve_budget(3, {"FLEXTOOL_THREADS": "5"}, toolbox=True) == (3, "cli")


@pytest.fixture
def toolbox_argv(monkeypatch):
    monkeypatch.setattr(sys, "argv", [_TOOLBOX_FILE, "sqlite:///in.sqlite"])
    monkeypatch.setattr(sys, "orig_argv", ["python", "-i", "-u", "-c", "import spine_repl"], raising=False)


def test_toolbox_run_gives_every_solver_one_thread(toolbox_argv, monkeypatch, capsys):
    from flextool.engine.solve import _benders
    from flextool.engine.solve import _solver_dispatch as sd

    monkeypatch.setattr(tb, "available_cores", lambda: 12)
    opts = _opts()
    assert tb.solver_budget(opts) == (1, "toolbox")
    assert tb.run_budget(opts) == (1, "toolbox")
    assert tb.highs_thread_options(opts) == ({"threads": 1}, None)
    got, warning = tb.highs_thread_options(_opts(highs_threads=4))
    assert got == {"threads": 1} and "Spine Toolbox" in warning
    assert tb.header_text(opts).startswith("1 (Spine Toolbox run): ")
    assert _benders._resolve_benders_workers(4, None, opts) == 1

    (override,) = sd._operator_native_overrides("gurobi", opts)
    assert (override.name, override.value, override.layer) == ("Threads", "1", sd.TOOLBOX_THREADS_LAYER)
    monkeypatch.setattr(sd, "_warned_thread_overrides", set())
    sd._note_thread_override("gurobi", sd.CommercialOption("Threads", "0", "gurobi.opt"), override)
    line = capsys.readouterr().out
    assert "the run's thread budget (Spine Toolbox run's default), instead of Threads 0 (all cores)" in line
    assert "--threads N to the FlexTool tool's arguments in Spine Toolbox" in line


def test_toolbox_run_with_threads_or_env_takes_them(toolbox_argv):
    from flextool.engine.solve import _solver_dispatch as sd

    opts = _opts(threads=3)
    assert tb.solver_budget(opts) == (3, "cli")
    assert tb.highs_thread_options(opts) == ({"threads": 3}, None)
    (override,) = sd._operator_native_overrides("gurobi", opts)
    assert (override.value, override.layer) == ("3", "flag --threads")
    env_opts = RunOptions.from_cli(SimpleNamespace(**{s.dest: None for s in OPTIONS.values() if s.flag}),
                                   environ={"FLEXTOOL_THREADS": "2"})
    assert tb.solver_budget(env_opts) == (2, "env")


def test_terminal_run_keeps_the_solvers_thread_settings(monkeypatch):
    from flextool.engine.solve import _solver_dispatch as sd

    monkeypatch.setattr(sys, "argv", ["flextool-run", "x"])
    assert tb.solver_budget(_opts()) is None
    assert sd._operator_native_overrides("gurobi", _opts()) == []


def test_toolbox_run_sizes_polars_to_one_thread():
    # The Basic Console's way: ``python -c``, sys.argv set to the file, then flextool imported.
    assert _probe([_TOOLBOX_FILE, "sqlite:///in.sqlite"]) == (1, "None")
    assert _probe([_TOOLBOX_FILE, "sqlite:///in.sqlite"], FLEXTOOL_THREADS="3") == (3, "None")
    assert _probe([_TOOLBOX_FILE, "sqlite:///in.sqlite", "--threads", "2"]) == (2, "None")
