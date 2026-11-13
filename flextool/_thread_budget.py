"""The thread budget of one FlexTool run, applied to polars before polars is imported.

One number *B* limits a run: FlexTool's own data processing (polars) and the solver (HiGHS, the commercial
solvers' thread count, Benders region solves).  It comes from ``--threads N`` > ``FLEXTOOL_THREADS`` > the
available cores; a Spine Toolbox run (:func:`is_toolbox_run`) takes 1 instead of the cores, because Toolbox runs
its tools in parallel and nothing shares the cores between them.  The assistant
(``flextool-assistant``) is the exception: it reads files while a model may be running and must not compete
with it, so its budget is 1 unless its own ``--threads N`` raises it (``FLEXTOOL_THREADS``, the budget of a run,
does not apply to it).

polars reads ``POLARS_MAX_THREADS`` once, when it is imported, and ``import flextool`` imports polars.  So the
package ``__init__`` calls :func:`apply_before_polars` before its first polars import: when the process is a
FlexTool command that accepts ``--threads``, the flag is read from the command line here (the real parser still
declares and checks it).  polars then gets ``min(B, POLARS_MAX_THREADS if the user set it, else 8)`` threads.
After polars is imported, :func:`restore_env` puts ``POLARS_MAX_THREADS`` back exactly as it was, so the
environment does not carry the value to child processes; a child gets ``--threads`` on its command line.

This module imports only the standard library and :mod:`flextool.run_options` (also stdlib-only).
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Mapping, Sequence

from flextool.run_options import OPTIONS, resolve

if TYPE_CHECKING:
    from flextool.run_options import RunOptions

#: polars gains nothing above 4-8 threads on FlexTool's workload and loses a little on large models; a user who
#: wants more sets ``POLARS_MAX_THREADS`` (still bounded by the budget).
THREAD_CAP_POLARS = 8

_POLARS_ENV = "POLARS_MAX_THREADS"

#: The assistant's command (``python -m`` module and script basenames): one thread unless its ``--threads`` says
#: otherwise (:func:`command_default_threads`).
_SINGLE_THREAD_MODULES = frozenset({"flextool_assistant_entry"})
_SINGLE_THREAD_SCRIPTS = frozenset({
    "flextool-assistant", "flextool-assistant.exe", "flextool-assistant-script.py", "flextool_assistant_entry.py",
})

#: ``python -m <module>`` entry points that accept ``--threads``.
_CLI_MODULES = frozenset({
    "flextool.cli.cmd_run_flextool",
    "flextool.cli.cmd_solve_mps",
    "flextool.cli.cmd_execute_flextool_workflow",
    "flextool.calibrate",
}) | _SINGLE_THREAD_MODULES

#: Script basenames (console scripts on every platform, and the files Spine Toolbox runs) that accept ``--threads``.
_CLI_SCRIPTS = frozenset({
    "flextool-run", "flextool-run.exe", "flextool-run-script.py",
    "flextool-execute-workflow", "flextool-execute-workflow.exe", "flextool-execute-workflow-script.py",
    "cmd_run_flextool.py", "cmd_solve_mps.py", "cmd_execute_flextool_workflow.py",
}) | _SINGLE_THREAD_SCRIPTS

#: The file Spine Toolbox runs for the FlexTool tool (``.spinetoolbox/specifications/Tool/flextool3.json``): as a
#: subprocess (``python cmd_run_flextool.py ...``), in its Basic Console (``sys.argv`` set, the file ``exec``-ed) or
#: in a Jupyter console (``%run``).  Every other FlexTool launcher starts the engine as ``-m
#: flextool.cli.cmd_run_flextool`` (the FlexTool window, ``flextool-run``, the workflow runner, the calibrator).
_TOOLBOX_SCRIPTS = frozenset({"cmd_run_flextool.py"})

#: The budget of a Spine Toolbox run that gives neither ``--threads`` nor ``FLEXTOOL_THREADS``.
TOOLBOX_DEFAULT_THREADS = 1

BudgetSource = Literal["cli", "env", "toolbox", "auto"]
PolarsSource = Literal["budget", "cap", "env", "already_loaded"]


@dataclass(frozen=True)
class ThreadBudget:
    """What :func:`apply_before_polars` decided for this process.

    *budget* is B; *source* where it came from (``toolbox`` = a Spine Toolbox run's default of 1, ``auto`` = the
    available cores).  *polars* is the polars thread
    count: the pool size it was given, or for ``already_loaded`` the size of the pool polars already had.
    *polars_source* says what limited it: ``budget`` (B), ``cap`` (:data:`THREAD_CAP_POLARS`), ``env`` (the
    user's ``POLARS_MAX_THREADS``) or ``already_loaded`` (polars was imported before flextool; its pool cannot
    change).
    """

    budget: int
    source: BudgetSource
    polars: int | None
    polars_source: PolarsSource

    @property
    def caller_set(self) -> bool:
        """Whether a flag or ``FLEXTOOL_THREADS`` set the budget (else it is the available cores)."""
        return self.source in ("cli", "env")


_applied: ThreadBudget | None = None
# The command line the decision was made for.  A later run in the same interpreter (a second run in a Spine
# Toolbox console) has another command line, and polars' pool is then the earlier run's.
_applied_argv: tuple[str, ...] | None = None


def available_cores() -> int:
    """Cores this process may run on: its CPU affinity (taskset, cgroup cpusets), else the machine's count."""
    count = getattr(os, "process_cpu_count", None)
    if count is not None:
        n = count()
        if n:
            return n
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _is_command(argv: Sequence[str], orig_argv: Sequence[str], modules: frozenset[str],
                scripts: frozenset[str]) -> bool:
    # Under ``python -m pkg.mod`` the package ``__init__`` runs while ``sys.argv[0]`` is still ``"-m"``; the
    # module name is only in ``sys.orig_argv``.
    for i, arg in enumerate(orig_argv[:-1]):
        if arg == "-m":
            return orig_argv[i + 1] in modules
    if not argv:
        return False
    # Either separator: a Windows path is split the same on every platform.  Windows names are case-insensitive.
    return argv[0].replace("\\", "/").rsplit("/", 1)[-1].lower() in scripts


def _is_flextool_cli(argv: Sequence[str], orig_argv: Sequence[str]) -> bool:
    return _is_command(argv, orig_argv, _CLI_MODULES, _CLI_SCRIPTS)


def is_toolbox_run(argv: Sequence[str] | None = None, orig_argv: Sequence[str] | None = None) -> bool:
    """Whether this process is a FlexTool run that Spine Toolbox started (*argv* / *orig_argv* default to ``sys``'s).

    Spine Toolbox sets no environment variable of its own, so the run is known by its command line: the file
    ``cmd_run_flextool.py`` run as a script (:data:`_TOOLBOX_SCRIPTS`).  Under ``python -m
    flextool.cli.cmd_run_flextool`` ``sys.argv[0]`` is that file's path too once the module runs, so a ``-m`` of a
    FlexTool command rules it out.  Read at the time of the call: a Basic Console sets ``sys.argv`` for each run.
    """
    argv = sys.argv if argv is None else argv
    orig_argv = getattr(sys, "orig_argv", []) if orig_argv is None else orig_argv
    for i, arg in enumerate(orig_argv[:-1]):
        if arg == "-m":
            if orig_argv[i + 1] in _CLI_MODULES:
                return False
            break
    if not argv:
        return False
    return argv[0].replace("\\", "/").rsplit("/", 1)[-1].lower() in _TOOLBOX_SCRIPTS


def command_default_threads(argv: Sequence[str], orig_argv: Sequence[str]) -> int | None:
    """The budget of a command line that gives no ``--threads``, when the command has its own: 1 for the
    assistant (it must not compete with a running model), else ``None`` (``FLEXTOOL_THREADS``, then the cores).

    The default counts as the command's own flag (source ``cli``), so ``FLEXTOOL_THREADS`` does not raise it.
    """
    return 1 if _is_command(argv, orig_argv, _SINGLE_THREAD_MODULES, _SINGLE_THREAD_SCRIPTS) else None


def cli_threads_from_argv(argv: Sequence[str], orig_argv: Sequence[str]) -> int | None:
    """The ``--threads N`` of a FlexTool command line, or ``None``.

    Read only when the process is a FlexTool command that accepts the flag (see :data:`_CLI_MODULES`,
    :data:`_CLI_SCRIPTS`), so another program's ``--threads`` is never taken.  Abbreviations are not matched;
    a value that is not a whole number of at least 1 counts as not given (the real parser reports it).
    """
    if not _is_flextool_cli(argv, orig_argv):
        return None
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--threads", type=str, default=None)
    try:
        known, _ = parser.parse_known_args(list(argv[1:]))
    except SystemExit:  # ``--threads`` without a value: the real parser reports it
        return None
    if known.threads is None:
        return None
    try:
        return OPTIONS["threads"].parse(known.threads)
    except ValueError:
        return None


def resolve_budget(cli: int | None, environ: Mapping[str, str], *, toolbox: bool = False) -> tuple[int, BudgetSource]:
    """B and its source: the flag, else ``FLEXTOOL_THREADS``, else :data:`TOOLBOX_DEFAULT_THREADS` for a Spine
    Toolbox run (*toolbox*), else :func:`available_cores`."""
    r = resolve(OPTIONS["threads"], cli=cli, environ=environ)
    if r.source in ("cli", "env") and r.value is not None:
        return int(r.value), r.source
    if toolbox:
        return TOOLBOX_DEFAULT_THREADS, "toolbox"
    return available_cores(), "auto"


def polars_threads(budget: int, environ: Mapping[str, str]) -> tuple[int, PolarsSource]:
    """polars' thread count for budget *budget*: ``min(B, POLARS_MAX_THREADS if set and valid, else the cap)``."""
    raw = environ.get(_POLARS_ENV)
    user: int | None = None
    if raw is not None and raw.strip():
        try:
            user = OPTIONS["threads"].parse(raw)
        except ValueError:
            user = None
    if user is not None:
        return (user, "env") if user <= budget else (budget, "budget")
    return (THREAD_CAP_POLARS, "cap") if THREAD_CAP_POLARS < budget else (budget, "budget")


def apply_before_polars(
    argv: Sequence[str] | None = None,
    orig_argv: Sequence[str] | None = None,
    environ: "os._Environ[str] | dict[str, str] | None" = None,
) -> str | None:
    """Resolve the budget and, when polars is not imported yet, set ``POLARS_MAX_THREADS`` for its import.

    Returns the previous ``POLARS_MAX_THREADS`` (``None`` when it was absent), to pass to :func:`restore_env`
    after polars is imported.  Records the decision for :func:`current`.
    """
    global _applied, _applied_argv
    env = os.environ if environ is None else environ
    argv = sys.argv if argv is None else argv
    _applied_argv = tuple(argv)
    orig_argv = getattr(sys, "orig_argv", []) if orig_argv is None else orig_argv
    previous = env.get(_POLARS_ENV)
    cli = cli_threads_from_argv(argv, orig_argv)
    if cli is None:
        cli = command_default_threads(argv, orig_argv)
    budget, source = resolve_budget(cli, env, toolbox=is_toolbox_run(argv, orig_argv))
    if "polars" in sys.modules:
        _applied = ThreadBudget(budget, source, _loaded_pool_size(), "already_loaded")
        return previous
    count, polars_source = polars_threads(budget, env)
    env[_POLARS_ENV] = str(count)
    _applied = ThreadBudget(budget, source, count, polars_source)
    return previous


def restore_env(previous: str | None, environ: "os._Environ[str] | dict[str, str] | None" = None) -> None:
    """Put ``POLARS_MAX_THREADS`` back as it was before :func:`apply_before_polars` (delete it when absent)."""
    env = os.environ if environ is None else environ
    if previous is None:
        env.pop(_POLARS_ENV, None)
    else:
        env[_POLARS_ENV] = previous


def _loaded_pool_size() -> int | None:
    polars = sys.modules.get("polars")
    size = getattr(polars, "thread_pool_size", None)
    try:
        return int(size()) if size is not None else None
    except Exception:  # a partly imported polars
        return None


def current() -> ThreadBudget | None:
    """The decision of :func:`apply_before_polars` in this process (``None`` before ``import flextool``)."""
    return _applied


def _same_command_line() -> bool:
    """Whether :func:`apply_before_polars` ran for this command line (not for an earlier run in this interpreter).

    Only the arguments are compared: under ``python -m`` the first entry is ``"-m"`` while the package
    ``__init__`` runs and the module's file afterwards.
    """
    return _applied_argv is not None and _applied_argv[1:] == tuple(sys.argv[1:])


def polars_pool() -> tuple[int | None, PolarsSource]:
    """polars' thread count in this process and what limited it, for this command line.

    The decision :func:`apply_before_polars` made for this command line; for a later run in the same interpreter
    (another command line) or before any decision, the pool polars already has, as ``already_loaded``.
    """
    applied = current()
    if applied is None or not _same_command_line():
        return _loaded_pool_size(), "already_loaded"
    return applied.polars, applied.polars_source


def highs_thread_options(run_options: "RunOptions") -> tuple[dict[str, object], str | None]:
    """The HiGHS ``threads`` / ``parallel`` options of a run, and a warning to show once (or ``None``).

    * ``--highs-threads h`` with h > 1: ``threads=h, parallel=on`` (HiGHS' parallel simplex, an explicit
      opt-in); with a caller-set budget B below h, h is lowered to B with a warning.
    * a solver budget B (:func:`solver_budget`: ``--threads`` / ``FLEXTOOL_THREADS``, or a Spine Toolbox run's
      default of 1) and no ``--highs-threads``: ``threads=B``; ``parallel`` stays as the lower layers set it
      (FlexTool pins ``off``).  An explicit ``--highs-threads 1`` with a budget gives ``threads=1``.
    * otherwise nothing: HiGHS' own default and ``highs.opt`` apply (the run is as before the budget existed).

    Only flag- and environment-sourced values count; registry defaults add nothing.
    """
    r = run_options.get("highs_threads")
    highs = r.value if r.source in ("cli", "env") else None
    solver = solver_budget(run_options)
    budget = solver[0] if solver is not None else None
    warning = None
    if highs is not None and int(highs) > 1:
        h = int(highs)
        if solver is not None and h > solver[0]:
            if solver[1] == "toolbox":
                warning = (f"--highs-threads {h} is more than a Spine Toolbox run's default of {solver[0]} thread "
                           f"(give --threads for more); HiGHS uses {solver[0]} thread")
            else:
                warning = (f"--highs-threads {h} is more than the run's --threads {solver[0]}; "
                           f"HiGHS uses {solver[0]} threads")
            h = solver[0]
        if h > 1:
            return {"threads": h, "parallel": "on"}, warning
        return {"threads": 1}, warning
    if budget is not None:
        return {"threads": 1 if highs is not None else int(budget)}, None
    return {}, None


def solver_budget(run_options: "RunOptions | None") -> tuple[int, BudgetSource] | None:
    """The thread budget the solvers take, or ``None`` (they keep their own thread settings).

    ``--threads`` / ``FLEXTOOL_THREADS`` (*run_options*; ``None`` reads the environment), else a Spine Toolbox
    run's :data:`TOOLBOX_DEFAULT_THREADS` (``toolbox``): every solver of a Toolbox run (HiGHS, the commercial
    solvers' thread count, Benders region solves at once) gets 1 thread, so the runs Toolbox executes in parallel
    do not each take every core.  The available cores (no budget given) leave the solvers as they are.
    """
    if run_options is None:
        from flextool.run_options import RunOptions

        run_options = RunOptions.from_env()
    r = run_options.get("threads")
    if r.source in ("cli", "env") and r.value is not None:
        return int(r.value), r.source
    if is_toolbox_run():
        return TOOLBOX_DEFAULT_THREADS, "toolbox"
    return None


def run_budget(run_options: "RunOptions") -> tuple[int, BudgetSource]:
    """B of a run whose options are *run_options*: :func:`solver_budget`, else the cores."""
    solver = solver_budget(run_options)
    return solver if solver is not None else (available_cores(), "auto")


def peek_mismatch(run_options: "RunOptions") -> str | None:
    """A warning when the parsed ``--threads`` differs from what :func:`apply_before_polars` read.

    Only an abbreviated flag (``--thr 4``) can cause it: the early read matches the full name only, so polars
    kept the pool it was given.  A later run in the same interpreter (another command line) is not checked:
    its polars pool was fixed by the first one, which the run header reports.
    """
    applied = current()
    if applied is None or applied.polars_source == "already_loaded" or not _same_command_line():
        return None
    budget, source = run_budget(run_options)
    if (budget, source) == (applied.budget, applied.source):
        return None
    return (f"spell out --threads (FlexTool reads the full flag name before polars starts); polars kept "
            f"{applied.polars} threads, the solver uses {budget}")


def header_text(run_options: "RunOptions", environ: Mapping[str, str] | None = None) -> str:
    """The run header's ``Threads`` value, e.g. ``4 (flag): polars 4, HiGHS 4``.

    The polars count is the pool polars has.  When it differs from what this run's budget asks for (polars was
    imported before flextool, or by an earlier run in this interpreter), the header says the budget applies to
    the solver only.
    """
    budget, source = run_budget(run_options)
    origin = {"cli": "flag", "env": "env FLEXTOOL_THREADS", "toolbox": "Spine Toolbox run",
              "auto": "available cores"}[source]
    pool = _loaded_pool_size()
    wanted, _ = polars_threads(budget, os.environ if environ is None else environ)
    if pool is None:
        polars = "polars ?"
    elif pool == wanted:
        polars = f"polars {pool}"
    else:
        polars = f"polars {pool} (polars was already loaded; --threads applies to the solver only)"
    highs_threads = highs_thread_options(run_options)[0].get("threads")
    highs = f"HiGHS {highs_threads}" if highs_threads is not None else "HiGHS default"
    return f"{budget} ({origin}): {polars}, {highs}"


__all__ = [
    "THREAD_CAP_POLARS",
    "ThreadBudget",
    "apply_before_polars",
    "available_cores",
    "cli_threads_from_argv",
    "current",
    "header_text",
    "highs_thread_options",
    "is_toolbox_run",
    "peek_mismatch",
    "polars_pool",
    "polars_threads",
    "resolve_budget",
    "restore_env",
    "run_budget",
    "solver_budget",
]
