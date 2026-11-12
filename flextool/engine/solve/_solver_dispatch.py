"""Multi-solver dispatch helpers (Phases 2 + 3 of the FlexTool multi-solver port).

This module owns:

* :func:`build_solver_options` — translate :class:`SolverConfig` into
  the option dict polar-high's ``solve()`` consumes (Phase 2).
* :func:`run_one_solve` — dispatch a single :class:`polar_high.Problem`
  to either ``Problem.solve()`` (in-process HiGHS, default — preserves
  streaming + ``Solution.highs``) or
  :func:`flextool.engine.solve._subprocess_solve.solve_via_subprocess`
  (every cold path — HiGHS via ``cmd_solve_mps`` and commercial solvers
  via their CLI binaries; both return real ``polar_high.Solution``
  objects with the HiGHS instance read back from the MPS).
* :class:`FlexToolUserError` — surface user-facing errors from the
  commercial path with actionable hints.

See ``specs/flextool-multi-solver-handoff.md`` Step 3 for the design
rationale and the canonical _PARAM_MAP table.
"""
from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, NamedTuple

from flextool._solver_option_rules import _PATH_OPTIONS as _PATH_OPTIONS
from flextool._solver_option_rules import _SERVER_OPTION_PREFIXES as _SERVER_OPTION_PREFIXES
from flextool._solver_option_rules import _SERVER_OPTIONS as _SERVER_OPTIONS
from flextool._solver_option_rules import _is_number
from flextool._solver_option_rules import _option_key as _option_key
from flextool._solver_option_rules import commercial_option_key as commercial_option_key
from flextool._solver_option_rules import cplex_set_word_problem
from flextool._solver_option_rules import highs_opt_lines as _highs_opt_lines
from flextool._solver_option_rules import is_path_option as is_path_option
from flextool._solver_option_rules import is_plain_file_name
from flextool._solver_option_rules import is_server_option as is_server_option
from flextool.engine.input._solve_config import SolverConfig

if TYPE_CHECKING:
    from pathlib import Path

    from polar_high import Problem

    from flextool._solver_config_source import SolverOptionFiles
    from flextool.run_options import RunOptions


# ---------------------------------------------------------------------------
# Effective HiGHS-options resolver (Batch C.1)
# ---------------------------------------------------------------------------


def _parse_highs_opt_file(path: Path | None) -> dict[str, str]:
    """Parse a HiGHS-style ``key=value`` options file into a flat dict.

    HiGHS' ``highs.opt`` syntax is one ``key = value`` line per option
    with optional surrounding whitespace; lines starting with ``#`` and
    blank lines are comments and skipped.  Unknown / malformed lines
    are also skipped (HiGHS itself tolerates them) so the floor never
    fails the engine; the user sees the misparse in HiGHS' own warning
    output when it loads the file.

    Returns an empty dict when *path* is None or does not exist — that
    is the steady state for in-process engine runs (the file is read
    by HiGHS on the CLI path only).  The resolver still calls this
    helper so a future change to wire the file through
    ``set_solver_options`` is a one-call edit.
    """
    if path is None or not path.is_file():
        return {}
    return _parse_highs_opt_text(path.read_text(encoding="utf-8"))


def _parse_highs_opt_text(text: str) -> dict[str, str]:
    """The ``key=value`` options of a ``highs.opt`` text (rules of :func:`_parse_highs_opt_file`)."""
    return {key: val for _, key, val in _highs_opt_lines(text)}


# ---------------------------------------------------------------------------
# Solver option names and values (item 236 A1)
# ---------------------------------------------------------------------------
#
# Every option that reaches a solver (an option file, the solve's
# ``solver_mip_gap`` / ``solver_arguments``, the run's flags and environment
# variables) is checked here before the solve starts.  CPLEX, Xpress and COPT
# get their options as lines of a command script, Gurobi as ``Name=value``
# arguments and HiGHS (``--save-memory``) as lines of an option file, so a
# line break, separator or space in a name or value could add commands or
# address another parameter.  Such an option is refused, never cleaned up.

#: An option name: a letter, then letters, digits, ``_`` and ``.``.
_OPTION_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.]*")
#: CPLEX also takes its interactive spelling: such words separated by single spaces (``mip tolerances mipgap``).
_CPLEX_OPTION_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.]*(?: [A-Za-z][A-Za-z0-9_.]*)*")
#: An option value: a number (``-1``, ``1e-4``, ``inf``) or one word; ``/``, ``\`` and ``:`` let a path through.
_OPTION_VALUE = re.compile(r"[A-Za-z0-9_.+\-/\\:]*")
_OPTION_NAME_MAX = 64
_OPTION_VALUE_MAX = 255


def _describe_character(ch: str) -> str:
    """Plain words for a refused character."""
    if ch in "\n\r\x0b\x0c\x1c\x1d\x1e\x85  ":
        return "a line break"
    if ch.isspace():
        return "a space"
    if not ch.isprintable():
        return f"a control character ({ch!r})"
    if ch in "'\"`":
        return "a quote"
    return f"the character {ch!r}"


def _short(text: str, limit: int = 80) -> str:
    """``repr`` of *text*, cut to *limit* characters (a message names hidden characters but stays one line)."""
    shown = repr(text)
    return shown if len(shown) <= limit else shown[: limit - 3] + "..."


def validate_solver_option(solver: str, name: object, value: object, where: str) -> None:
    """Refuse a solver option whose *name* or *value* could change the solver's command script or arguments.

    *where* names the source in the message: the option file and its line, ``solver_arguments of solve X``,
    ``solver_mip_gap of solve X`` or the flag / environment variable.  Raises :class:`FlexToolUserError`.
    Booleans and ``None`` pass; other numbers are checked as their text.
    """
    if value is None or isinstance(value, bool):
        value_text = ""
    else:
        value_text = str(value)
    name_text = str(name)
    name_rule = _CPLEX_OPTION_NAME if solver == "cplex" else _OPTION_NAME
    problem: str | None = None
    if not name_text:
        problem = "the name is empty"
    elif len(name_text) > _OPTION_NAME_MAX:
        problem = f"the name is longer than {_OPTION_NAME_MAX} characters"
    elif name_rule.fullmatch(name_text) is None:
        bad = next(
            (ch for ch in name_text if not re.fullmatch(r"[A-Za-z0-9_.]", ch) and not (solver == "cplex" and ch == " ")),
            None,
        )
        if bad is not None:
            problem = f"the name contains {_describe_character(bad)}"
        elif not name_text[0].isalpha():
            problem = "the name does not start with a letter"
        else:
            problem = "the name's words are not separated by single spaces, each starting with a letter"
    elif (set_word := cplex_set_word_problem(solver, name_text)) is not None:
        problem = f"the name {set_word}"
    elif len(value_text) > _OPTION_VALUE_MAX:
        problem = f"the value is longer than {_OPTION_VALUE_MAX} characters"
    elif _OPTION_VALUE.fullmatch(value_text) is None:
        bad = next(ch for ch in value_text if not re.fullmatch(r"[A-Za-z0-9_.+\-/\\:]", ch))
        problem = f"the value contains {_describe_character(bad)}"
    elif value_text.endswith("\\"):
        # A console that reads Tcl (Xpress) joins a line ending in a backslash with the next command.
        problem = "the value ends with a backslash"
    if problem is None:
        return
    name_rule_text = (
        "letters, digits, '_' and '.' (CPLEX: such words separated by single spaces)"
        if solver == "cplex" else "letters, digits, '_' and '.', starting with a letter"
    )
    raise FlexToolUserError(
        f"Refusing {solver} option {_short(name_text)} (value {_short(value_text)}) from {where}: {problem}. "
        f"A solver option name is {name_rule_text}; a value is a number or one word of letters, digits and "
        "'_ . + - / \\ :' (no spaces, quotes, ';', '#', '=', line breaks or a backslash at the end). "
        "Correct the option there and run again."
    )


# ---------------------------------------------------------------------------
# Options from the input database (item 236, owner decision 2026-11-12, option b)
# ---------------------------------------------------------------------------
#
# Trust follows the source.  The run's own machine is trusted: the
# ``solver_config/<solver>.opt`` files, the flags and the environment
# variables keep full paths and server settings.  The input database is not:
# a database can come from someone else, so a solve's ``solver_arguments``
# (and ``solver_mip_gap``) may not point a solver at a file outside the solve's
# work folder or at another machine.  An option that names a file, folder or
# path takes only a plain file name there, which lands in the solve's work
# folder (:func:`solver_workspace`); a Gurobi server / cloud / licence option is
# refused.  No option of the five solvers starts another program: FlexTool
# passes options only as ``set`` / ``setControl`` lines or ``Name=value``
# arguments, so the server options are the only other way out.  Both kinds are
# refused, never cleaned up; the general check (:func:`validate_solver_option`)
# applies to every source.  The option tables and the name rules are in
# :mod:`flextool._solver_option_rules` (no engine import: project import screens
# a shipped option file with them).


def solver_workspace(work_folder: "Path | str | None") -> Path | None:
    """The solve's work folder for solver files: ``<work_folder>/solve_data/subprocess`` (where a commercial
    solver and the ``--save-memory`` HiGHS run), or ``None`` without a work folder."""
    return Path(work_folder) / "solve_data" / "subprocess" if work_folder is not None else None


def check_database_solver_option(solver: str, name: object, value: object, where: str) -> None:
    """Refuse an option from the input database (*where*: ``solver_arguments of solve X``) that names a path or
    another machine.

    A path option (:func:`is_path_option`) takes a number, a boolean or a plain file name: ASCII, no ``/``,
    ``\\``, ``:`` or ``..``, not starting with ``.``.  A server option (:func:`is_server_option`) is refused.  Run
    :func:`validate_solver_option` as well; this check adds the database's limits only.  Raises
    :class:`FlexToolUserError` naming the option, its value and the source.
    """
    value_text = "" if value is None or isinstance(value, bool) else str(value)
    name_text = str(name)
    instead = (
        f"set it in the run's solver option file instead (solver_config/{solver}.opt, or that file in a folder "
        "given with --solver-config-dir): FlexTool trusts the files and the command line of the computer it runs "
        "on, not the input database."
    )
    if is_server_option(solver, name_text):
        raise FlexToolUserError(
            f"Refusing {solver} option {_short(name_text)} (value {_short(value_text)}) from {where}: it comes "
            "from the input database, and an option that connects the solver to another computer (a server, "
            f"cloud or licence service) or carries its credentials is not taken from there. Remove it from the "
            f"database and {instead}"
        )
    if not value_text or not is_path_option(solver, name_text) or _is_number(value_text):
        return
    if not is_plain_file_name(value_text):
        raise FlexToolUserError(
            f"Refusing {solver} option {_short(name_text)} (value {_short(value_text)}) from {where}: it comes "
            "from the input database, where an option that names a file or folder takes only a plain file name "
            "(ASCII letters, digits and '_ . + -', starting with a letter, digit or '_'; no '/', '\\', ':' or "
            "'..'), and FlexTool puts that file in the solve's work folder (solve_data/subprocess). For a file "
            f"elsewhere, remove the option from the database and {instead}"
        )


#: Solution-file options that older ``highs.opt.template`` versions shipped
#: (a GLPK-era leftover: ``write_solution_style=2`` is the glpsol format).
#: Every project created before the template was cleaned has a copy, so every
#: solve wrote an unused ``flextool.sol`` and HiGHS warned "The cost row for
#: the Glpsol solution file is reported last".  Nothing reads that file, so
#: the ``highs.opt`` layer drops these keys.  A solution file can still be
#: requested deliberately through the solve's ``solver_arguments`` (a plain
#: file name, written in the solve's work folder).
_HIGHS_OPT_IGNORED_KEYS: dict[str, str] = {
    "write_solution_to_file": "true",
    "solution_file": "flextool.sol",
    "write_solution_style": "2",
}
_warned_ignored_highs_opt_keys = False


def _drop_solution_file_options(
    options: dict[str, str], path: Path | None
) -> dict[str, str]:
    """Remove the solution-file keys from a parsed ``highs.opt`` dict.

    The exact old-template lines are dropped silently: FlexTool put them
    there, not the user, and nothing needs to be done about them.  Any other
    value for these keys was set by hand, so it is dropped with a one-time
    warning that says where to request a solution file instead.
    """
    global _warned_ignored_highs_opt_keys
    ignored = {k: options.pop(k) for k in list(options) if k in _HIGHS_OPT_IGNORED_KEYS}
    customised = {
        k: v for k, v in ignored.items()
        if v.strip().lower() != _HIGHS_OPT_IGNORED_KEYS[k]
    }
    if customised and not _warned_ignored_highs_opt_keys:
        _warned_ignored_highs_opt_keys = True
        logging.getLogger(__name__).warning(
            "Ignoring solution-file options in %s (%s): FlexTool reads its "
            "results from the solver directly and does not use a solution "
            "file. To have HiGHS write one anyway, set these options in the "
            "solve's solver_arguments (solution_file takes a plain file name "
            "there, written in the solve's work folder).",
            path,
            ", ".join(f"{k}={v}" for k, v in customised.items()),
        )
    return options


def _resolve_effective_highs_options(
    *,
    solver_arguments_map: Mapping[str, Any] | None,
    highs_opt_path: Path | None,
    cli_overrides: Mapping[str, Any] | None,
    baseline: Mapping[str, Any] | None = None,
    solver_config: SolverConfig | None = None,
    highs_opt_text: str | None = None,
    solve_name: str | None = None,
    workspace: Path | None = None,
) -> dict[str, Any]:
    """Resolve the effective HiGHS solver options for one solve.

    This is the one option set every HiGHS solve receives: warm, cold,
    ``--save-memory`` / subprocess and the Benders master and regions.

    Precedence (lowest → highest):

    1. ``baseline`` — engine-pinned defaults from
       :func:`flextool.engine.solve._orchestration._baseline_highs_options`
       (Curtis-Reid simplex scale + the four determinism keys from
       :data:`flextool.engine.solve._determinism.DETERMINISM_OPTIONS`).
       Each higher layer may overwrite these — operator intent wins.
    2. ``highs.opt`` — floor parsed from ``solver_config/highs.opt``
       via :func:`_parse_highs_opt_file`.  Project-level defaults the
       user has committed to disk.  Solution-file keys are dropped from
       this layer (:func:`_drop_solution_file_options`).
    3. ``solver_config`` — the convenience knobs of the active solve
       (``solve.solver_mip_gap`` → ``mip_rel_gap``; ``time_limit`` /
       ``threads`` when set), translated through :data:`_PARAM_MAP`.
    4. ``solver_arguments`` — the 1d-map authored on the active
       solve's ``solver_arguments`` parameter (Batch C.1+).  Raw
       options win over the convenience knobs.  From the input database:
       a file option takes a plain file name, placed in *workspace*, and
       no path (:func:`check_database_solver_option`).
    5. ``cli_overrides`` — keys injected by CLI flags
       (``--highs-threads``, ``--solver-time-limit``, …) via
       :func:`flextool.engine.solve._orchestration._operator_highs_overrides`.
       Highest precedence; the operator's command-line intent is
       authoritative.

    Empty / ``None`` layers are skipped cleanly.

    Parameters
    ----------
    solver_arguments_map
        The 1d-map dict authored on the solve's ``solver_arguments``
        parameter.  ``None`` and ``{}`` are equivalent.
    highs_opt_path
        Path to ``solver_config/highs.opt`` (or any equivalent).
        ``None`` skips this layer.
    highs_opt_text
        The ``highs.opt`` text the run read at its start
        (:class:`flextool._solver_config_source.SolverOptionFiles`); when
        given it is the layer and *highs_opt_path* only names it in messages.
    cli_overrides
        Dict of HiGHS option-keys → values to apply at the top of the
        precedence chain.  ``None`` and ``{}`` are equivalent.
    baseline
        Optional engine-pinned floor below all other layers.  When
        ``None`` an empty dict is used (callers that want the
        determinism + scale floor pass it explicitly).
    solver_config
        The active solve's :class:`SolverConfig`; only its ``time_limit``
        / ``mip_gap`` / ``threads`` knobs are read (``None`` values and a
        ``None`` config skip the layer).
    solve_name
        The solve, named in the message refusing an option
        (:func:`validate_solver_option`; every layer but *baseline* is checked).
    workspace
        The solve's work folder for solver files (:func:`solver_workspace`).
        ``solver_arguments`` and ``solver_mip_gap`` come from the input
        database, so they are also checked by
        :func:`check_database_solver_option`, and a plain file name they give
        a HiGHS file option (:data:`_PATH_OPTIONS`) becomes a path in
        *workspace* (created when needed; ``None`` leaves the name as it is,
        a file in the current folder).

    Returns
    -------
    dict[str, Any]
        The final option dict ready to forward to
        ``polar_high.Problem.set_solver_options``.
    """
    options: dict[str, Any] = dict(baseline) if baseline else {}
    if highs_opt_text is None and highs_opt_path is not None and highs_opt_path.is_file():
        highs_opt_text = highs_opt_path.read_text(encoding="utf-8")
    of_solve = f" of solve {solve_name}" if solve_name else ""
    file_options: dict[str, str] = {}
    if highs_opt_text is not None:
        file_label = str(highs_opt_path) if highs_opt_path is not None else "highs.opt"
        for lineno, key, val in _highs_opt_lines(highs_opt_text):
            if key not in _HIGHS_OPT_IGNORED_KEYS:  # dropped below, never passed on
                validate_solver_option("highs", key, val, f"{file_label}, line {lineno}")
        file_options = _parse_highs_opt_text(highs_opt_text)
    options.update(_drop_solution_file_options(file_options, highs_opt_path))
    if solver_config is not None:
        knobs = _PARAM_MAP["highs"]
        for knob in ("time_limit", "mip_gap", "threads"):
            value = getattr(solver_config, knob)
            if value is not None:
                validate_solver_option("highs", knobs[knob], value, f"solver_mip_gap{of_solve}")
                check_database_solver_option("highs", knobs[knob], value, f"solver_mip_gap{of_solve}")
                options[knobs[knob]] = value
    if solver_arguments_map:
        for key, value in solver_arguments_map.items():
            validate_solver_option("highs", key, value, f"solver_arguments{of_solve}")
            check_database_solver_option("highs", key, value, f"solver_arguments{of_solve}")
            if (
                workspace is not None and _option_key("highs", key) in _PATH_OPTIONS["highs"]
                and isinstance(value, str) and value
            ):
                workspace.mkdir(parents=True, exist_ok=True)
                value = str(workspace / value)
            options[str(key)] = value
    if cli_overrides:
        for key, value in cli_overrides.items():
            validate_solver_option("highs", key, value, "the run's flags or environment variables")
            options[str(key)] = value
    return options


# ---------------------------------------------------------------------------
# User-facing error type
# ---------------------------------------------------------------------------


class FlexToolUserError(Exception):
    """Raised for user-actionable misconfiguration of the multi-solver
    dispatch.

    Carries a message intended for direct surfacing to the user — installer
    hint, license hint, or "switch the solver" hint.  ``__cause__`` carries
    the underlying polar-high exception for debugging.
    """


# Per-solver native parameter names for the three "convenience" knobs
# normalised by FlexTool.  Anything outside these three goes through
# untranslated via ``SolverConfig.options``.  The one table for the
# database knob (``solve.solver_mip_gap``) and the operator layer
# (``--solver-time-limit`` / ``--solver-mip-gap`` and their environment
# twins).  The names are the ones each CLI takes: CPLEX's interactive
# ``set`` spelling (``mip tolerances mipgap``, not the Python API's
# dotted path), Xpress controls in upper case as in its templates.
_PARAM_MAP: dict[str, dict[str, str]] = {
    "highs":  {"time_limit": "time_limit", "mip_gap": "mip_rel_gap",          "threads": "threads"},
    "gurobi": {"time_limit": "TimeLimit",  "mip_gap": "MIPGap",                "threads": "Threads"},
    "cplex":  {"time_limit": "timelimit",  "mip_gap": "mip tolerances mipgap", "threads": "threads"},
    "xpress": {"time_limit": "MAXTIME",    "mip_gap": "MIPRELSTOP",            "threads": "THREADS"},
    "copt":   {"time_limit": "TimeLimit",  "mip_gap": "RelGap",                "threads": "Threads"},
}


def build_solver_options(solver_config: SolverConfig) -> dict[str, Any]:
    """Translate a :class:`SolverConfig` into the raw options dict that
    polar-high's ``solve()`` consumes.

    The database layers only (no option file, no operator flags, exact key
    match).  FlexTool's own commercial solves use
    :func:`resolve_commercial_options`, the full stack.

    1. The three convenience knobs (``time_limit`` / ``mip_gap`` /
       ``threads``), when set on *solver_config*, are translated to the
       chosen solver's native parameter name via :data:`_PARAM_MAP`.
       ``None`` values are skipped (no override).
    2. The raw ``solver_config.options`` dict is merged on top of the
       translated knobs — **raw options win** on key collisions.  The
       user knows what they're doing; if they hand-write
       ``solver_arguments = {"TimeLimit": 30}`` and also set
       ``solver_time_limit = 60``, the raw value (30) wins.

    Parameters
    ----------
    solver_config
        Solve-level config built by
        :meth:`flextool.engine.input._solve_config.SolveConfig.load_from_db`.

    Returns
    -------
    dict[str, Any]
        Option dict ready to ``**unpack`` into
        :func:`polar_high.solvers.solve`.  Empty dict when no
        convenience knobs are set and ``options`` is empty.

    Raises
    ------
    ValueError
        If ``solver_config.name`` is not in :data:`_PARAM_MAP` AND at
        least one convenience knob is set.  Raw-options-only with an
        unknown solver passes through silently so users can plug a
        future solver via ``solver_arguments`` before
        :data:`_PARAM_MAP` is updated.
    """
    has_convenience_knob = (
        solver_config.time_limit is not None
        or solver_config.mip_gap is not None
        or solver_config.threads is not None
    )
    mapping = _PARAM_MAP.get(solver_config.name)
    if mapping is None and has_convenience_knob:
        available = ", ".join(sorted(_PARAM_MAP.keys()))
        raise ValueError(
            f"unknown solver {solver_config.name!r}, expected one of: "
            f"{available}"
        )

    opts: dict[str, Any] = {}
    if mapping is not None:
        if solver_config.time_limit is not None:
            opts[mapping["time_limit"]] = solver_config.time_limit
        if solver_config.mip_gap is not None:
            opts[mapping["mip_gap"]] = solver_config.mip_gap
        if solver_config.threads is not None:
            opts[mapping["threads"]] = solver_config.threads

    # Raw options take precedence — see docstring.
    opts.update(solver_config.options)
    return opts


# ---------------------------------------------------------------------------
# Commercial option stack (item 230)
# ---------------------------------------------------------------------------
#
# One stack for every solver, lowest -> highest:
#   1. ``<solver>.opt`` (the run's solver-config folder);
#   2. the solve's ``solver_mip_gap`` (translated through :data:`_PARAM_MAP`);
#   3. the solve's ``solver_arguments`` (raw native names);
#   4. the operator layer: ``--solver-time-limit`` / ``--solver-mip-gap``, else
#      ``FLEXTOOL_SOLVER_TIME_LIMIT`` / ``FLEXTOOL_SOLVER_MIP_GAP``, and the thread
#      budget (``--threads`` / ``FLEXTOOL_THREADS``; item 232): it replaces the
#      solver's thread count, also a value meaning all cores, with one printed
#      line per run (:func:`_note_thread_override`).
# Names are compared as the solver reads them (:func:`commercial_option_key`:
# without case; CPLEX ``.`` read as a space; Gurobi without ``_``; Xpress
# without an ``XPRS_`` prefix), so ``MIPGap`` in the file and ``mip_gap`` in
# solver_arguments are one parameter: one line, at the position of its first appearance, with the
# highest layer's value and spelling (CPLEX: always the interactive spelling,
# since the dotted one is not a ``set`` command).

#: FlexTool's own result-file parameter, compared as :func:`commercial_option_key`: FlexTool adds it to the
#: solver's command (Gurobi ``ResultFile=`` in ``_gurobi_script``) and reads the solution from that file, so
#: it is dropped from every layer with one warning.  The other solvers get their result file through a
#: ``write`` command of their script, not an option.
_FLEXTOOL_OWN_OPTIONS: dict[str, frozenset[str]] = {
    "gurobi": frozenset({"resultfile"}),
}

#: Flags / environment variables that only HiGHS reads.  Given on a run whose
#: solve goes to a commercial solver, they are named once in a warning.
_HIGHS_ONLY_OPTIONS = (
    "highs_threads", "presolve", "user_bound_scale",
    "highs_time_limit", "highs_mip_gap",
)

#: Commercial watchdog: the solver enforces its own time limit; FlexTool's
#: subprocess timeout only guards a hung process.  Reading a large MPS before
#: the limit starts and writing the result file after it can take minutes,
#: hence the generous grace (``2 x limit + 600 s``).
_WATCHDOG_FACTOR = 2.0
_WATCHDOG_GRACE_S = 600.0

#: Warnings given once per run (cleared by :func:`reset_run_warnings` at the start of each run).
_warned_dropped_options: set[tuple[str, str, str]] = set()
_warned_highs_only: set[tuple[str, ...]] = set()
_warned_precommands: set[str] = set()
_warned_thread_overrides: set[tuple[str, str, str]] = set()


def reset_run_warnings() -> None:
    """Forget the once-per-run option warnings; ``run_chain_from_db`` calls it at the start of every run, so a
    second run in the same process (GUI, tests) warns again."""
    _warned_dropped_options.clear()
    _warned_highs_only.clear()
    _warned_precommands.clear()
    _warned_thread_overrides.clear()


def warn_solver_precommands(precommands: Mapping[str, Any] | None, logger: logging.Logger | None = None) -> None:
    """One warning per run for each solve whose ``solver_precommand`` has a value: it is not run (item 236 A3).

    *precommands* is ``SolverSettings.precommand`` (solve name -> the database value, as read; rolls and
    duplicated solves do not copy it).  A command stored in a database is never run: a database from someone
    else could otherwise run any program on this computer.  Nothing here starts a process.
    """
    for solve, value in (precommands or {}).items():
        text = "" if value is None else str(value).strip()
        if not text or solve in _warned_precommands:
            continue
        _warned_precommands.add(solve)
        (logger or logging.getLogger(__name__)).warning(
            "solver_precommand of solve %s (%s) is not run: FlexTool does not run commands stored in a database, "
            "for safety (a database from someone else could otherwise run any program on this computer). To hold "
            "a floating licence, use your solver's own licence server setup; a solver wrapper set on this "
            "computer is planned. See docs/solvers/index.md.",
            solve, _short(text),
        )


class CommercialOption(NamedTuple):
    """One parameter passed to a commercial solver: native name, value text and the layer that set it."""

    name: str
    value: str
    layer: str


def _native_value(solver: str, knob: str, value: object) -> str:
    """Text of a convenience-knob value for *solver*: Xpress ``MAXTIME`` is an
    integer control, so a time limit is rounded up to whole seconds."""
    from flextool.engine.solve._subprocess_solve import _format_native_opt_value

    if solver == "xpress" and knob == "time_limit":
        return str(int(math.ceil(float(value))))
    return _format_native_opt_value(value)


def _operator_native_overrides(
    solver: str, run_options: "RunOptions | None",
) -> list[CommercialOption]:
    """The operator layer of a commercial solve: the time limit and the MIP gap
    given by flag or by ``FLEXTOOL_SOLVER_*``, and the run's thread budget, under
    *solver*'s native names.

    threads: the run's budget (``--threads`` / ``FLEXTOOL_THREADS``) replaces the
    solver's thread count; without them the option file and ``solver_arguments``
    keep it (as for HiGHS).  The HiGHS-only ``FLEXTOOL_HIGHS_TIME_LIMIT`` /
    ``FLEXTOOL_HIGHS_MIP_GAP`` / ``--highs-threads`` do not apply.  A registry
    default (and an unset budget, which is the available cores) adds nothing.
    """
    from flextool.run_options import OPTIONS, RunOptions

    opts = run_options if run_options is not None else RunOptions.from_env()
    names = _PARAM_MAP[solver]
    out: list[CommercialOption] = []
    for knob, key in (
        ("time_limit", "solver_time_limit"), ("mip_gap", "solver_mip_gap"), ("threads", "threads"),
    ):
        resolved = opts.resolved.get(key)
        if resolved is None or resolved.source not in ("cli", "env") or resolved.value is None:
            continue
        spec = OPTIONS[key]
        layer = f"flag {spec.flag}" if resolved.source == "cli" else f"environment variable {spec.env}"
        out.append(CommercialOption(names[knob], _native_value(solver, knob, resolved.value), layer))
    return out


def resolve_commercial_options(
    solver: str,
    *,
    opt_file: Path | str | None,
    solver_config: SolverConfig | None,
    run_options: "RunOptions | None",
    solve_name: str | None = None,
    logger: logging.Logger | None = None,
    opt_text: str | None = None,
) -> list[CommercialOption]:
    """The parameters a commercial solve passes to *solver*, in final order.

    Layers (lowest -> highest): *opt_file* < the solve's ``solver_mip_gap`` <
    its ``solver_arguments`` < the operator layer
    (:func:`_operator_native_overrides`).  See the block comment above for the
    merge rule.  *opt_text* is the option file's text as the run read it at its
    start; when given, *opt_file* only names the file in messages (else the file
    is read).  Trust follows the source: the solve's ``solver_mip_gap`` and
    ``solver_arguments`` come from the input database and are also checked by
    :func:`check_database_solver_option` (a path option takes only a plain file
    name, which lands in the solve's work folder; no server options); the option
    file and the flags may hold both.  Every layer passes
    :func:`validate_solver_option`.  FlexTool's own result-file parameter
    (:data:`_FLEXTOOL_OWN_OPTIONS`) is dropped from every layer with one warning
    per run for each name and kind of layer.
    """
    from flextool.engine.solve._subprocess_solve import (
        _format_native_opt_value,
        _native_opt_lines,
        _parse_native_opt_text,
        _read_native_opt_text,
    )

    of_solve = f" of solve {solve_name}" if solve_name else ""
    own = _FLEXTOOL_OWN_OPTIONS.get(solver, frozenset())
    entries: list[CommercialOption] = []
    if opt_text is None and opt_file is not None:
        opt_text = _read_native_opt_text(Path(opt_file))
    file_options: dict[str, str] = {}
    file_label = str(opt_file) if opt_file is not None else f"{solver}.opt"
    if opt_text is not None:
        for lineno, name, value in _native_opt_lines(opt_text):
            if commercial_option_key(solver, name) not in own:  # dropped below, never passed on
                validate_solver_option(solver, name, value, f"{file_label}, line {lineno}")
        file_options = _parse_native_opt_text(opt_text)
    entries += [CommercialOption(k, v, file_label) for k, v in file_options.items()]
    n_file_entries = len(entries)
    if solver_config is not None:
        names = _PARAM_MAP[solver]
        entries += [
            CommercialOption(
                names[knob], _native_value(solver, knob, getattr(solver_config, knob)), f"solver_mip_gap{of_solve}"
            )
            for knob in ("time_limit", "mip_gap", "threads")
            if getattr(solver_config, knob) is not None
        ]
        entries += [
            CommercialOption(str(k), _format_native_opt_value(v), f"solver_arguments{of_solve}")
            for k, v in solver_config.options.items()
            if v is not None
        ]
    # The input database's layers: a path option takes a plain file name (the solver runs in the solve's work
    # folder, so the file lands there) and a server option is refused, before any option is dropped or merged.
    for option in entries[n_file_entries:]:
        validate_solver_option(solver, option.name, option.value, option.layer)
        check_database_solver_option(solver, option.name, option.value, option.layer)
    n_checked = len(entries)
    thread_key = commercial_option_key(solver, _PARAM_MAP[solver]["threads"])
    replaced_threads = None  # the lower layers' winning thread count, which the run's budget replaces
    for option in entries:
        if commercial_option_key(solver, option.name) == thread_key:
            replaced_threads = option
    operator = _operator_native_overrides(solver, run_options)
    entries += operator
    budget = next((o for o in operator if commercial_option_key(solver, o.name) == thread_key), None)
    if budget is not None and replaced_threads is not None:
        _note_thread_override(solver, replaced_threads, budget)

    merged: dict[str, CommercialOption] = {}
    for position, option in enumerate(entries):
        key = commercial_option_key(solver, option.name)
        if key in own:
            # One warning per run for each name and kind of layer (the option file, a solve's solver_mip_gap /
            # solver_arguments, the flags), not one per solve or roll: it names the first layer that set it.
            warn_key = (solver, key, option.layer.split(" of solve ", 1)[0])
            if warn_key not in _warned_dropped_options:
                _warned_dropped_options.add(warn_key)
                (logger or logging.getLogger(__name__)).warning(
                    "Ignoring %s option %s %s in %s: FlexTool names the solver's result file itself and reads "
                    "the solution from it.",
                    solver, option.name, _short(option.value), option.layer,
                )
            continue
        if position >= n_checked:  # the option file and the database layers were checked above
            validate_solver_option(solver, option.name, option.value, option.layer)
        merged[key] = option._replace(name=key) if solver == "cplex" else option
    return list(merged.values())


def _thread_number(value: str) -> float | None:
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _note_thread_override(solver: str, old: CommercialOption, new: CommercialOption) -> None:
    """One printed line per run when the run's thread budget *new* replaces a different thread count *old*
    of the option file or the input database (item 232).

    Printed, not logged: it sits just before the solve's printed ``Solver args`` line, where ``logger.info``
    does not reach the console.  Once per solver, kind of layer and old value, so rolls and repeated solves
    print it once.  The advice to raise ``--threads`` (``FLEXTOOL_THREADS`` when the budget came from it) is
    added only when the budget lowers the thread count (an "all cores" value: Gurobi / CPLEX ``0``, Xpress / COPT ``<= 0``, or a larger number).
    """
    old_number, new_number = _thread_number(old.value), _thread_number(new.value)
    if old_number is not None and old_number == new_number:
        return
    warn_key = (solver, old.layer.split(" of solve ", 1)[0], old.value)
    if warn_key in _warned_thread_overrides:
        return
    _warned_thread_overrides.add(warn_key)
    all_cores = old_number is not None and (old_number == 0 or (solver in ("xpress", "copt") and old_number < 0))
    lowers = all_cores or (old_number is not None and new_number is not None and old_number > new_number)
    if not lowers:
        tail = ""
    elif new.layer.startswith("environment variable "):
        env_name = new.layer.removeprefix("environment variable ")
        tail = f" For more solver threads raise {env_name} (or give --threads)."
    else:
        tail = " For more solver threads raise --threads (Threads per run in the FlexTool window)."
    print(
        f"{solver} uses {new.name} {new.value}, the run's thread budget ({new.layer}), instead of {old.name} "
        f"{old.value}{' (all cores)' if all_cores else ''} in {old.layer}.{tail}",
        flush=True,
    )


def commercial_opt_file_layer(
    solver: str,
    *,
    solver_option_files: "SolverOptionFiles | None",
    solver_config_dir: "Path | None" = None,
    run_options: "RunOptions | None" = None,
) -> tuple[str | None, str | None]:
    """The option-file layer of a commercial solve: ``(text, where it came from)``, or ``(None, None)``.

    The run's files read at its start win (*solver_option_files*); else ``<solver_config_dir>/<solver>.opt``;
    else the folder :func:`flextool._solver_config_source.resolve_solver_config_source` picks without a
    database (``--solver-config-dir`` / ``FLEXTOOL_SOLVER_CONFIG_DIR``, the current folder, the packaged
    default), for a caller outside ``run_chain_from_db``.
    """
    from flextool._solver_config_source import read_solver_option_files, resolve_solver_config_source

    files = solver_option_files
    if files is None:
        if solver_config_dir is not None:
            path = Path(solver_config_dir) / f"{solver}.opt"
            try:
                return path.read_text(encoding="utf-8"), str(path)
            except OSError:
                return None, None
        source = resolve_solver_config_source(run_options=run_options)
        files = read_solver_option_files(source, (solver,))
    text = files.text(solver)
    return (text, files.origin(solver)) if text is not None else (None, None)


def commercial_time_limit(solver: str, options: "list[CommercialOption] | Mapping[str, Any] | None") -> float | None:
    """The time limit (seconds) among the resolved *options* of *solver*, else ``None``.

    Read from the native time-limit name only (Xpress also ``TIMELIMIT``).  A
    value that is not a positive finite number (Gurobi's ``inf``) gives ``None``.
    Xpress ``MAXTIME``: a positive value stops an LP at the limit, and a MIP only
    once it has a solution (FlexTool's Xpress path solves LPs only); a negative
    value is a hard limit of its absolute value (xpress 46.01: ``MAXTIME -1`` and
    ``1`` both stop at 1 s on a MIP with incumbents, only ``-1`` without one), so
    its absolute value is the limit.
    """
    if not options or solver not in _PARAM_MAP:
        return None
    wanted = {commercial_option_key(solver, _PARAM_MAP[solver]["time_limit"])}
    if solver == "xpress":
        wanted.add("timelimit")
    items = (
        [(o.name, o.value) for o in options] if isinstance(options, list) else list(options.items())
    )
    limit: float | None = None
    for name, value in items:
        if commercial_option_key(solver, name) not in wanted:
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if solver == "xpress" and commercial_option_key(solver, name) == "maxtime":
            v = abs(v)
        limit = v if math.isfinite(v) and 0.0 < v < 1e15 else None
    return limit


def commercial_watchdog_seconds(time_limit: float | None) -> float | None:
    """The subprocess timeout of a commercial solve with *time_limit*: ``2 x limit + 600 s``; no limit, no
    watchdog."""
    if time_limit is None:
        return None
    return _WATCHDOG_FACTOR * float(time_limit) + _WATCHDOG_GRACE_S


def warn_highs_only_options(solver: str, run_options: "RunOptions | None", logger: logging.Logger | None) -> None:
    """One warning per run (per distinct set) naming the HiGHS-only flags / variables given while a solve runs
    on the commercial *solver*."""
    from flextool.run_options import OPTIONS, RunOptions

    opts = run_options if run_options is not None else RunOptions.from_env()
    given = []
    for key in _HIGHS_ONLY_OPTIONS:
        resolved = opts.resolved.get(key)
        if resolved is None or resolved.source not in ("cli", "env"):
            continue
        spec = OPTIONS[key]
        given.append(spec.flag if resolved.source == "cli" and spec.flag else spec.env)
    # ``silent`` hides a commercial solver's log too and ``normal`` is the default; only ``verbose`` (HiGHS'
    # developer log level) has no commercial meaning.
    level = opts.resolved.get("solver_log_level")
    if level is not None and level.source in ("cli", "env") and level.value == "verbose":
        spec = OPTIONS["solver_log_level"]
        name = spec.flag if level.source == "cli" and spec.flag else spec.env
        given.append(f"{name} verbose" if level.source == "cli" else f"{name}=verbose")
    if not given:
        return
    warn_key = (solver, *given)
    if warn_key in _warned_highs_only:
        return
    _warned_highs_only.add(warn_key)
    (logger or logging.getLogger(__name__)).warning(
        "%s only applies to HiGHS; this solve runs on %s and does not use it. Set %s's own parameter in the "
        "solve's solver_arguments or in %s.opt instead.",
        ", ".join(given), solver, solver, solver,
    )


def problem_is_mip(vars_map: Mapping[str, Any] | None) -> bool:
    """Whether a problem (its ``name -> Var`` map) has an integer column."""
    for var in (vars_map or {}).values():
        if getattr(var, "integer", False):
            frame = getattr(var, "frame", None)
            if frame is None or getattr(frame, "height", 1) > 0:
                return True
    return False


# ---------------------------------------------------------------------------
# Single-solve dispatch (Phase 3)
# ---------------------------------------------------------------------------


def run_one_solve(
    problem: "Problem",
    solver_config: SolverConfig,
    logger: logging.Logger | None = None,
    *,
    save_memory: bool = False,
    solve_name: str | None = None,
    work_folder: "Path | None" = None,
    run_options: "RunOptions | None" = None,
    solver_config_dir: "Path | None" = None,
    commercial_options: "list[CommercialOption] | None" = None,
    solver_option_files: "SolverOptionFiles | None" = None,
):
    """Dispatch *problem* to the chosen solver.

    The in-process HiGHS path is a direct call to
    ``problem.solve(keep_solver=True)``, which preserves streaming and the
    live ``Solution.highs`` (consumed by the output writer adapter).  Both
    HiGHS paths (in-process and ``save_memory`` subprocess) solve with the
    options already stored on *problem* (``set_solver_options`` with the
    resolved set, then autoscale Layer 3); *solver_config* selects the
    solver only.

    The commercial path (gurobi / cplex / xpress / copt) routes through
    :func:`flextool.engine.solve._subprocess_solve.solve_via_subprocess`,
    which spawns the solver's CLI binary against an MPS written by
    ``Problem.write_mps``, streams its log and reads the result file back
    into a duck-typed ``_SolHighsShim`` handle.  Downstream consumers
    (``flex_data.py``, ``_co2_accumulators.py``,
    ``process_outputs/read_parameters.py``) see a uniform
    :class:`polar_high.Solution` shape regardless of which solver ran.

    Parameters
    ----------
    problem
        The :class:`polar_high.Problem` to solve.
    solver_config
        Resolved per-solve configuration (defaults to HiGHS/direct when
        no ``solver_*`` parameter is authored on the solve).
    logger
        Optional logger (subprocess progress, option warnings).  Errors are
        raised as :class:`FlexToolUserError` and logged once by the caller
        (``cmd_run_flextool``), not here.
    save_memory
        When True on the HiGHS path, divert to
        :func:`flextool.engine.solve._subprocess_solve.solve_via_subprocess`:
        polar-high writes the LP to a temp MPS file via
        ``Problem.write_mps(release=True)`` (a direct polars→MPS writer
        that never builds a ``highspy.Highs`` instance, peaking at
        ~2-3 GB on the largest LPs vs ~45 GB for HiGHS' own
        ``writeModel``), then a ``flextool.cli.cmd_solve_mps``
        subprocess solves the MPS in a clean address space.  The
        parent reads the solution back via a read-only ``highspy.Highs``
        and wraps it as a ``polar_high.Solution`` identical in shape
        to the in-process return.  Trades file I/O for HiGHS' active-
        solve memory living outside the parent process.  Also disables
        warm-LP reuse for the cascade — the Problem is in ``_released``
        state after the write.  Silently ignored on the commercial path.
    solve_name
        Used by the subprocess path to name MPS/options/solution files.
        Defaults to ``"solve"`` when omitted.
    work_folder
        When provided, the subprocess path keeps its MPS/options/sol
        files under ``<work_folder>/solve_data/subprocess/`` for post-
        mortem inspection.  ``None`` uses a self-cleaning tempdir.
        Ignored on the in-process path.
    run_options
        The run's operator options, handed to the subprocess path (warm-start
        basis cache).  ``None`` reads the environment.
    solver_config_dir
        A folder whose ``<solver>.opt`` the commercial path reads, for a caller
        without *solver_option_files*.
    commercial_options
        The commercial parameters already resolved by the caller
        (:func:`resolve_commercial_options`, as the orchestrator does for its
        ``Solver args`` line); ``None`` resolves them here.
    solver_option_files
        The run's option files, read once at run start (``run_chain_from_db``);
        win over *solver_config_dir*.  With neither, the folder is resolved like
        a run without a database (:func:`commercial_opt_file_layer`).

    Returns
    -------
    polar_high.Solution
        The native polar-high Solution.  On the subprocess paths the
        contained ``highs`` is a :class:`_SolHighsShim` holding the parsed
        primal / duals and the real column / row names (no
        ``highspy.Highs`` is built in the parent).

    Raises
    ------
    FlexToolUserError
        If the requested solver's Python wrapper is not installed, the
        license check fails, or the solver returns a model-level error.
    """
    if solver_config.name == "highs":
        # HiGHS: the Problem already holds the one resolved option set
        # (engine baseline < highs.opt < solve.solver_mip_gap <
        # solver_arguments < CLI flags, ``_resolve_effective_highs_options``)
        # plus the autoscale Layer-3 keys merged on top, so both branches
        # below use ``problem._solver_options`` and never re-derive a
        # partial set from ``solver_config``.
        if save_memory:
            # Subprocess path: write MPS via Problem.write_mps directly
            # from polars frames, spawn flextool.cli.cmd_solve_mps, read
            # solution back.  The options reach the subprocess through the
            # .opt file written by solve_via_subprocess (write_mps itself
            # runs no solver and takes no options).
            from flextool.engine.solve._subprocess_solve import (
                solve_via_subprocess,
            )
            effective_opts = dict(getattr(problem, "_solver_options", None) or {})
            return solve_via_subprocess(
                problem,
                "highs",
                effective_opts,
                solve_name=solve_name or "solve",
                logger=logger,
                work_folder=work_folder,
                run_options=run_options,
            )
        # Names reach HiGHS only when a HiGHS option writes a file with them.
        return problem.solve(keep_solver=True, highs_names="auto")

    # Commercial path.  Always subprocess: write MPS via the cheap
    # ``Problem.write_mps`` (polars→MPS, no LpView), spawn the solver's
    # CLI binary, parse the result file.  The in-process commercial-solver
    # dispatch was retired alongside the in-process cold-HiGHS path —
    # the goal is a single hard memory bound (~2-3 GB for write_mps on
    # the largest LPs) for every cold solve regardless of solver.
    #
    # The options are the one stack of :func:`resolve_commercial_options`
    # (``<solver>.opt`` < solver_mip_gap < solver_arguments < flags /
    # FLEXTOOL_SOLVER_*); the resolved time limit also sets the watchdog.
    from flextool.engine.solve._subprocess_solve import (
        _BINARY_NAMES,
        solve_via_subprocess,
    )
    if solver_config.name not in _BINARY_NAMES and solver_config.name != "highs":
        # Unknown solver name — fail clean before we try anything else.
        from polar_high.solvers import available_solvers
        msg = (
            f"Unknown solver {solver_config.name!r}.  Supported solvers: "
            f"highs, gurobi, cplex, xpress, copt.  Installed wrappers: "
            f"{available_solvers}."
        )
        raise FlexToolUserError(msg)
    if solver_config.name == "xpress" and problem_is_mip(getattr(problem, "_vars", None)):
        # The Xpress console script runs ``lpoptimize``, which on a MIP solves
        # only the LP relaxation: refuse instead of returning a wrong answer.
        msg = (
            f"Solve {solve_name or 'solve'!r} has integer variables, but FlexTool's Xpress driver solves only "
            "linear problems (it would return the LP relaxation, not an integer solution). Use another solver "
            "for this solve, or remove its integer variables."
        )
        raise FlexToolUserError(msg)
    if commercial_options is None:
        opt_text, opt_label = commercial_opt_file_layer(
            solver_config.name,
            solver_option_files=solver_option_files,
            solver_config_dir=solver_config_dir,
            run_options=run_options,
        )
        commercial_options = resolve_commercial_options(
            solver_config.name,
            opt_file=opt_label,
            opt_text=opt_text,
            solver_config=solver_config,
            run_options=run_options,
            solve_name=solve_name,
            logger=logger,
        )
    warn_highs_only_options(solver_config.name, run_options, logger)
    options = {o.name: o.value for o in commercial_options}
    option_layers = {o.name: o.layer for o in commercial_options}
    try:
        return solve_via_subprocess(
            problem,
            solver_config.name,
            options,
            solve_name=solve_name or "solve",
            logger=logger,
            work_folder=work_folder,
            run_options=run_options,
            option_layers=option_layers,
        )
    except RuntimeError as e:
        msg = str(e)
        if msg.startswith("LICENSE: "):
            user_msg = (
                f"Solver {solver_config.name!r} subprocess failed a "
                f"license check.  Details: {msg[len('LICENSE: '):]}.  "
                f"See docs/solvers/{solver_config.name}.md#licensing "
                f"for help."
            )
            raise FlexToolUserError(user_msg) from e
        if "was not found on $PATH" in msg:
            user_msg = (
                f"Solver {solver_config.name!r} CLI binary is not "
                f"installed on this system.  {msg}  See "
                f"docs/solvers/{solver_config.name}.md for installation "
                f"instructions."
            )
            raise FlexToolUserError(user_msg) from e
        raise FlexToolUserError(
            f"Solver {solver_config.name!r} failed for solve {solve_name or 'solve'!r}: {msg}"
        ) from e


_LICENSE_PROBE_CACHE: dict[str, str] | None = None


def probe_solver_licenses() -> dict[str, str]:
    """Return ``{solver_name: status}`` for every solver in
    ``polar_high.solvers.available_solvers``.

    Status values:

    - ``"installed"`` — wrapper installed, license check passed (with
      either a free test license or a commercial one), trivial solve
      completed.  Neutral wording — does not imply the user holds a
      full commercial entitlement; a free trial license counts too.
    - ``"no-license"`` — wrapper installed but the solver refused on
      license grounds (commercial trial expired, no licence file, etc.).
    - ``"not-installed"`` — Python wrapper isn't on this system.
    - ``"probe-failed"`` — any other exception during the probe; the
      solver may or may not be functional on a real problem.

    The probe runs a 1-variable, 0-constraint LP through
    ``polar_high.solvers.solve(...)`` per solver.  Solver chatter on
    stdout is suppressed.  Result cached at module level so repeat
    cascade runs in the same Python process don't re-probe.

    Used by ``_orchestration.run_chain_from_db`` to print one INFO line
    per cascade, giving users a quick "is gurobi actually working on
    this machine" hint.
    """
    global _LICENSE_PROBE_CACHE
    if _LICENSE_PROBE_CACHE is not None:
        return _LICENSE_PROBE_CACHE
    import io
    import os
    from contextlib import redirect_stdout, redirect_stderr
    # Tests set FLEXTOOL_SKIP_SOLVER_PROBE=1 in conftest to avoid the
    # FICO Xpress Community LicenseWarning (and other startup chatter)
    # that the probe triggers via xpress.problem().  Skip silently and
    # cache an empty dict so callers' .items() iteration is a no-op.
    if os.environ.get("FLEXTOOL_SKIP_SOLVER_PROBE"):
        _LICENSE_PROBE_CACHE = {}
        return _LICENSE_PROBE_CACHE
    import polars as pl
    from polar_high import Problem
    from polar_high.solvers import (
        LicenseError,
        SolverError,
        SolverNotAvailableError,
        available_solvers,
    )
    from polar_high.solvers import solve as polar_solve

    statuses: dict[str, str] = {}
    # HiGHS / Xpress write probe chatter via C-level handles that
    # ``redirect_stdout`` alone can't catch.  Redirect the underlying
    # file descriptors for the duration of each probe so the startup
    # log stays clean.
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    try:
        for solver_name in available_solvers:
            try:
                p = Problem()
                df = pl.DataFrame({"i": [0]})
                v = p.add_var(
                    "x", dims=("i",), index=df, lower=0.0, upper=10.0,
                )
                p.set_objective(v.to_expr())
                os.dup2(devnull_fd, 1)
                os.dup2(devnull_fd, 2)
                # Some solver packages (gurobipy) log through Python
                # ``logging``, whose handlers hold their own stream objects:
                # mute logging too, or the chatter reaches the run log copy
                # (``flextool.run_record``) although the console is silent.
                _logging_disabled = logging.root.manager.disable
                logging.disable(logging.CRITICAL)
                try:
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        polar_solve(p, solver_name=solver_name)
                finally:
                    logging.disable(_logging_disabled)
                # The probe only confirms that a license file (free
                # test license or commercial) is *installed* — not that
                # the user holds a full commercial entitlement.  Use
                # the neutral "installed" wording to avoid overstating.
                statuses[solver_name] = "installed"
            except LicenseError:
                statuses[solver_name] = "no-license"
            except SolverNotAvailableError:
                statuses[solver_name] = "not-installed"
            except SolverError:
                statuses[solver_name] = "solver-error"
            except Exception:  # noqa: BLE001 — probe should never crash startup
                statuses[solver_name] = "probe-failed"
            finally:
                os.dup2(saved_stdout_fd, 1)
                os.dup2(saved_stderr_fd, 2)
    finally:
        os.close(devnull_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
    _LICENSE_PROBE_CACHE = statuses
    return statuses


__all__ = [
    "_PARAM_MAP",
    "CommercialOption",
    "FlexToolUserError",
    "commercial_option_key",
    "commercial_time_limit",
    "commercial_watchdog_seconds",
    "problem_is_mip",
    "resolve_commercial_options",
    "_parse_highs_opt_file",
    "_resolve_effective_highs_options",
    "build_solver_options",
    "check_database_solver_option",
    "is_path_option",
    "is_server_option",
    "probe_solver_licenses",
    "solver_workspace",
    "run_one_solve",
]
