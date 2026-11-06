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
    ],
)
def test_cli_threads_from_argv(argv, orig_argv, expected):
    assert tb.cli_threads_from_argv(argv, orig_argv) == expected


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
