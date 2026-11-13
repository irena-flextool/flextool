"""Command lines and output checks of the commercial-solver CLI driver.

No solver binary or licence is needed: the argv builders are pure functions and the licence check reads text.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from flextool.engine.solve._subprocess_solve import _gurobi_script, _looks_like_license_error


def test_gurobi_parameters_are_name_value_arguments(tmp_path: Path) -> None:
    """``gurobi_cl`` takes parameters only as ``Name=value`` arguments before the model file; it rejects
    ``ReadParams=<file>`` ("Unknown parameter: 'ReadParams' (ignored)") and then solves at its defaults."""
    params = (("Threads", "0"), ("MIPGap", "0.01"), ("TimeLimit", "3600"))
    binary, mps, sol = Path("/opt/gurobi/bin/gurobi_cl"), tmp_path / "solve.mps", tmp_path / "solve.sol"

    argv, stdin = _gurobi_script(binary, mps, sol, params)

    assert argv == [str(binary), "Threads=0", "MIPGap=0.01", "TimeLimit=3600", f"ResultFile={sol}", str(mps)]
    assert stdin is None


def test_gurobi_without_parameters(tmp_path: Path) -> None:
    """No parameters (no baseline and no solver_options): only the result file and the model."""
    binary, mps, sol = Path("gurobi_cl"), tmp_path / "m.mps", tmp_path / "m.sol"

    argv, _ = _gurobi_script(binary, mps, sol, ())

    assert argv == ["gurobi_cl", f"ResultFile={sol}", str(mps)]


# A normal Gurobi 13 log of an infeasible model: the banner names the licence file and the licence type.
_GUROBI_INFEASIBLE_LOG = """\
Set parameter LogToConsole to value 1
Using license file /Library/gurobi/gurobi.lic
Set parameter ResultFile to value "solve.sol"
Gurobi Optimizer version 13.0.2 build v13.0.2rc0 (mac64[arm] - Darwin 24.6.0 24G90)
Academic license 2655372 - for non-commercial use only - registered to someone@example.org
Restricted license - for non-production use only - expires 2027-11-24
Optimize a model with 1520 rows, 1104 columns and 4300 nonzeros
Presolve time: 0.01s

Model is infeasible
Best objective -, best bound -, gap -
"""


def test_licence_banner_of_an_infeasible_solve_is_not_a_licence_error() -> None:
    assert not _looks_like_license_error(_GUROBI_INFEASIBLE_LOG, "")
    assert not _looks_like_license_error("", _GUROBI_INFEASIBLE_LOG)


@pytest.mark.parametrize("text", [
    "ERROR 10009: No Gurobi license found (user someone, host build01, hostid 1a2b3c4d, cores 8)",
    "Error 10009: License expired 2026-10-31",
    "Error 10009: Request denied: license not valid for this host",
    "Error 10010: HostID mismatch (licensed to 1a2b3c4d, hostid is 5e6f7a8b)",
    "Error 10030: Model too large for size-limited license; visit https://gurobi.com/unrestricted",
    "Error 10022: Unable to contact token server 'tokens.example.org'",
    "Failed to obtain license from the Web License Service",
    "CPLEX Error  32201: ilm: CPLEX: no license found: no such file or directory.",
    "CPLEX Error  1016: Promotional version. Problem size limits exceeded.",
    "?2: Xpress licensing error 8: license file not found",
    "Could not find license file /opt/copt/license.dat",
    "COPT license error: licence has expired",
])
def test_licence_failure_is_a_licence_error(text: str) -> None:
    assert _looks_like_license_error("", text)
    assert _looks_like_license_error(_GUROBI_INFEASIBLE_LOG + text + "\n", "")


# ---------------------------------------------------------------------------
# One option stack for every solver (item 230): option file < solver_mip_gap < solver_arguments < flag / env.
# Expected values are worked out by hand from the layer rules (specs/gurobi_options_fix_design.md §3).
# ---------------------------------------------------------------------------

from flextool.cli.cmd_run_flextool import build_parser  # noqa: E402
from flextool.engine.input._solve_config import SolverConfig  # noqa: E402
from flextool.engine.solve._solver_dispatch import (  # noqa: E402
    _operator_native_overrides,
    commercial_time_limit,
    commercial_watchdog_seconds,
    reset_run_warnings,
    resolve_commercial_options,
)
from flextool.engine.solve._subprocess_solve import _SCRIPTS, commercial_params  # noqa: E402
from flextool.run_options import RunOptions  # noqa: E402

_FLAGS = ["--solver-mip-gap", "0.001", "--solver-time-limit", "1800"]


def _options(flags: list[str], environ: dict | None = None) -> RunOptions:
    return RunOptions.from_cli(build_parser().parse_args(["sqlite:///x.sqlite", *flags]), environ=environ or {})


def _passed(solver: str, tmp_path: Path, opt_text: str, cfg: SolverConfig, flags: list[str]):
    """The argv parameters (Gurobi) or the stdin script (the others) the CLI would get."""
    opt_file = tmp_path / f"{solver}.opt"
    opt_file.write_text(opt_text)
    resolved = resolve_commercial_options(solver, opt_file=opt_file, solver_config=cfg, run_options=_options(flags))
    merged = commercial_params({o.name: o.value for o in resolved})
    argv, stdin = _SCRIPTS[solver](Path("bin"), tmp_path / "m.mps", tmp_path / "m.sol", merged)
    return argv[1:-2], stdin


def test_gurobi_stack_flags_win_and_one_gap_entry(tmp_path: Path) -> None:
    cfg = SolverConfig(name="gurobi", mip_gap=0.01, options={"Presolve": 2, "mipgap": 0.005})
    params, _ = _passed("gurobi", tmp_path, "Threads 0\nMIPGap 0.002\n", cfg, _FLAGS)
    assert params == ["Threads=0", "MIPGap=0.001", "Presolve=2", "TimeLimit=1800"]


def test_gurobi_stack_without_flags_solver_arguments_win(tmp_path: Path) -> None:
    """solver_arguments beat the solver_mip_gap knob and the file; the winning spelling is kept, at the file's
    position."""
    cfg = SolverConfig(name="gurobi", mip_gap=0.01, options={"Presolve": 2, "mipgap": 0.005})
    params, _ = _passed("gurobi", tmp_path, "Threads 0\nMIPGap 0.002\n", cfg, [])
    assert params == ["Threads=0", "mipgap=0.005", "Presolve=2"]


@pytest.mark.parametrize("solver, opt_text, gap_line, time_line", [
    ("cplex", "threads 1\nmip tolerances mipgap 0.002\n", "set mip tolerances mipgap 0.001", "set timelimit 1800"),
    ("xpress", "THREADS 1\nMIPRELSTOP 0.002\n", "setControl MIPRELSTOP 0.001", "setControl MAXTIME 1800"),
    ("copt", "Threads 1\nRelGap 0.002\n", "set RelGap 0.001", "set TimeLimit 1800"),
])
def test_flags_reach_cplex_xpress_copt_under_native_names(solver, opt_text, gap_line, time_line, tmp_path) -> None:
    cfg = SolverConfig(name=solver, mip_gap=0.01)
    _, stdin = _passed(solver, tmp_path, opt_text, cfg, _FLAGS)
    lines = stdin.splitlines()
    assert gap_line in lines and time_line in lines
    gap_word = {"cplex": "mipgap", "xpress": "MIPRELSTOP", "copt": "RelGap"}[solver]
    assert sum(gap_word.lower() in line.lower() for line in lines) == 1, stdin
    assert "mip.tolerances.mipgap" not in stdin


def test_cplex_dotted_name_lands_on_the_interactive_spelling(tmp_path: Path) -> None:
    cfg = SolverConfig(name="cplex", options={"mip.tolerances.mipgap": 0.004})
    _, stdin = _passed("cplex", tmp_path, "threads 1\nmip tolerances mipgap 0.002\n", cfg, [])
    assert [line for line in stdin.splitlines() if "mipgap" in line] == ["set mip tolerances mipgap 0.004"]


def test_highs_environment_names_never_reach_a_commercial_solver() -> None:
    """D5: FLEXTOOL_HIGHS_* stay HiGHS-only; FLEXTOOL_SOLVER_* are the flags' twins for every solver."""
    from flextool.engine.solve._orchestration import _operator_highs_overrides

    highs_only = RunOptions.from_env({"FLEXTOOL_HIGHS_TIME_LIMIT": "120", "FLEXTOOL_HIGHS_MIP_GAP": "0.3"})
    assert _operator_native_overrides("gurobi", highs_only) == []
    assert _operator_highs_overrides(highs_only) == {"time_limit": 120.0, "mip_rel_gap": 0.3}
    both = RunOptions.from_env({"FLEXTOOL_HIGHS_TIME_LIMIT": "120", "FLEXTOOL_SOLVER_TIME_LIMIT": "90"})
    assert [(o.name, o.value) for o in _operator_native_overrides("gurobi", both)] == [("TimeLimit", "90")]
    assert _operator_highs_overrides(both) == {"time_limit": 90.0}


@pytest.mark.parametrize("solver, opts, expected", [
    ("gurobi", {"TimeLimit": "1800.0"}, 4200.0),
    ("gurobi", {"timelimit": "60"}, 720.0),
    ("cplex", {"timelimit": "300"}, 1200.0),
    ("xpress", {"MAXTIME": "60"}, 720.0),
    ("xpress", {"MAXTIME": "-60"}, 720.0),      # a negative MAXTIME is a hard limit of 60 s (review F2)
    ("xpress", {"TIMELIMIT": "30.5"}, 661.0),
    ("gurobi", {"Threads": "0"}, None),
    ("gurobi", {"TimeLimit": "inf"}, None),
])
def test_watchdog_is_twice_the_native_limit_plus_ten_minutes(solver, opts, expected) -> None:
    assert commercial_watchdog_seconds(commercial_time_limit(solver, opts)) == expected


def test_result_file_in_solver_arguments_is_dropped_with_a_warning(tmp_path: Path, caplog) -> None:
    cfg = SolverConfig(name="gurobi", options={"ResultFile": "mine.sol", "Presolve": 2})
    with caplog.at_level("WARNING"):
        params, _ = _passed("gurobi", tmp_path, "Threads 0\n", cfg, [])
    assert params == ["Threads=0", "Presolve=2"]
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1 and "ResultFile" in warnings[0] and "solver_arguments" in warnings[0]


def test_result_file_warning_is_once_per_run_not_per_solve(tmp_path: Path, caplog) -> None:
    """Review L7: the same ``ResultFile`` in the ``solver_arguments`` of two solves (or two rolls) warns once in a
    run, naming the first; the next run warns again."""
    reset_run_warnings()
    cfg = SolverConfig(name="gurobi", options={"ResultFile": "mine.sol"})
    with caplog.at_level("WARNING"):
        for solve in ("dispatch_roll_1", "dispatch_roll_2", "invest"):
            resolve_commercial_options("gurobi", opt_file=None, solver_config=cfg, run_options=_options([]),
                                       solve_name=solve)
        first_run = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        reset_run_warnings()
        resolve_commercial_options("gurobi", opt_file=None, solver_config=cfg, run_options=_options([]),
                                   solve_name="invest")
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(first_run) == 1 and "solver_arguments of solve dispatch_roll_1" in first_run[0]
    assert len(warnings) == 2 and "solver_arguments of solve invest" in warnings[1]


# ---------------------------------------------------------------------------
# The run's thread budget replaces the commercial thread count (item 232). Oracle: critique_232 (a)-(d).
# ---------------------------------------------------------------------------

def _thread_lines(solver: str, stdin: str) -> list[str]:
    word = {"cplex": "threads", "xpress": "THREADS", "copt": "Threads"}[solver]
    return [line for line in stdin.splitlines() if line.split()[-2:-1] == [word]]


def test_threads_flag_replaces_the_all_cores_gurobi_thread_count(tmp_path: Path, capsys) -> None:
    """(a) gurobi.opt ``Threads 0`` + ``--threads 4``: ``Threads=4`` at the file line's position, one line
    printed with the all-cores advice."""
    reset_run_warnings()
    params, _ = _passed("gurobi", tmp_path, "Threads 0\nMIPGap 0.002\n", SolverConfig(name="gurobi"),
                        ["--threads", "4"])
    assert params == ["Threads=4", "MIPGap=0.002"]
    out = capsys.readouterr().out.splitlines()
    assert out == [
        f"gurobi uses Threads 4, the run's thread budget (flag --threads), instead of Threads 0 (all cores) in "
        f"{tmp_path / 'gurobi.opt'}. For more solver threads raise --threads (Threads per run in the FlexTool "
        "window)."
    ]


@pytest.mark.parametrize("solver, opt_text, expected", [
    ("cplex", "threads 0\n", "set threads 2"),
    ("xpress", "THREADS -1\n", "setControl THREADS 2"),
    ("copt", "Threads -1\n", "set Threads 2"),
])
def test_threads_flag_replaces_the_cplex_xpress_copt_thread_count(solver, opt_text, expected, tmp_path,
                                                                   capsys) -> None:
    reset_run_warnings()
    _, stdin = _passed(solver, tmp_path, opt_text, SolverConfig(name=solver), ["--threads", "2"])
    assert _thread_lines(solver, stdin) == [expected]
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1 and "2, the run's thread budget" in out[0] and "(all cores)" in out[0], out
    assert out[0].endswith("(Threads per run in the FlexTool window).")


def test_thread_budget_from_the_environment_replaces_xpress_all_cores(tmp_path: Path, capsys) -> None:
    """(b) xpress.opt ``THREADS -1``, ``FLEXTOOL_THREADS=2``, no flag: ``setControl THREADS 2`` at the file
    line's position; the line names the environment variable."""
    reset_run_warnings()
    opt_file = tmp_path / "xpress.opt"
    opt_file.write_text("THREADS -1\nMIPRELSTOP 0.002\n")
    resolved = resolve_commercial_options("xpress", opt_file=opt_file, solver_config=SolverConfig(name="xpress"),
                                          run_options=_options([], {"FLEXTOOL_THREADS": "2"}))
    assert [(o.name, o.value) for o in resolved] == [("THREADS", "2"), ("MIPRELSTOP", "0.002")]
    out = capsys.readouterr().out.splitlines()
    assert out == [
        "xpress uses THREADS 2, the run's thread budget (environment variable FLEXTOOL_THREADS), instead of "
        f"THREADS -1 (all cores) in {opt_file}. For more solver threads raise FLEXTOOL_THREADS (or give "
        "--threads)."
    ]


def test_thread_budget_is_appended_when_no_layer_sets_threads(tmp_path: Path, capsys) -> None:
    """(c) cplex, no thread line anywhere, ``--threads 3``: ``set threads 3`` after the file entries; nothing
    printed."""
    reset_run_warnings()
    _, stdin = _passed("cplex", tmp_path, "mip tolerances mipgap 0.002\n", SolverConfig(name="cplex"),
                       ["--threads", "3"])
    assert [line for line in stdin.splitlines() if line.startswith("set ")][-2:] == [
        "set mip tolerances mipgap 0.002", "set threads 3"]
    assert capsys.readouterr().out == ""


def test_thread_budget_beats_solver_arguments_alias(tmp_path: Path, capsys) -> None:
    """A ``solver_arguments`` alias (Gurobi ``threads``) is the same parameter: one entry with the operator's
    spelling; the line names the database layer and the environment variable, with the advice (8 -> 3 lowers
    the thread count)."""
    reset_run_warnings()
    cfg = SolverConfig(name="gurobi", options={"threads": 8})
    resolved = resolve_commercial_options("gurobi", opt_file=None, solver_config=cfg,
                                          run_options=_options([], {"FLEXTOOL_THREADS": "3"}), solve_name="invest")
    assert [(o.name, o.value) for o in resolved] == [("Threads", "3")]
    out = capsys.readouterr().out.splitlines()
    assert out == [
        "gurobi uses Threads 3, the run's thread budget (environment variable FLEXTOOL_THREADS), instead of "
        "threads 8 in solver_arguments of solve invest. For more solver threads raise FLEXTOOL_THREADS (or give "
        "--threads)."
    ]


def test_thread_override_line_names_the_option_file_without_its_path(capsys) -> None:
    """A direct caller giving the option text without its file: the line names ``gurobi.opt``, never ``None``."""
    reset_run_warnings()
    resolved = resolve_commercial_options("gurobi", opt_text="Threads 0\n", opt_file=None, solver_config=None,
                                          run_options=_options(["--threads", "2"]))
    assert [(o.name, o.value, o.layer) for o in resolved] == [("Threads", "2", "flag --threads")]
    assert capsys.readouterr().out.splitlines() == [
        "gurobi uses Threads 2, the run's thread budget (flag --threads), instead of Threads 0 (all cores) in "
        "gurobi.opt. For more solver threads raise --threads (Threads per run in the FlexTool window)."]


def test_option_file_read_without_its_text_drops_a_utf8_bom(tmp_path: Path) -> None:
    """A caller giving only the file (no text): the file is read as UTF-8 and a leading BOM is not part of the
    first option's name, whatever the computer's locale encoding."""
    opt_file = tmp_path / "gurobi.opt"
    opt_file.write_bytes("\ufeffMIPGap 0.002\n".encode("utf-8"))
    resolved = resolve_commercial_options("gurobi", opt_file=opt_file, solver_config=None, run_options=_options([]))
    assert [(o.name, o.value) for o in resolved] == [("MIPGap", "0.002")]


def test_no_budget_keeps_the_option_file_thread_count(tmp_path: Path, capsys) -> None:
    """(d) gurobi.opt ``Threads 8``, no flag, no FLEXTOOL_THREADS: ``Threads=8`` as in 4.3; nothing printed."""
    reset_run_warnings()
    params, _ = _passed("gurobi", tmp_path, "Threads 8\n", SolverConfig(name="gurobi"), [])
    assert params == ["Threads=8"]
    assert capsys.readouterr().out == ""


def test_thread_override_line_is_once_per_run_and_without_advice_when_raising(tmp_path: Path, capsys) -> None:
    """Equal values print nothing; a raise (``Threads 1`` -> 4) prints without the advice; a lower budget
    prints once over two solves of a run and again after the next run's reset."""
    reset_run_warnings()
    opt_file = tmp_path / "gurobi.opt"

    def resolve(text: str, threads: str, solve: str) -> list[tuple[str, str]]:
        opt_file.write_text(text)
        resolved = resolve_commercial_options("gurobi", opt_file=opt_file, solver_config=None,
                                              run_options=_options(["--threads", threads]), solve_name=solve)
        return [(o.name, o.value) for o in resolved]

    assert resolve("Threads 1\n", "1", "a") == [("Threads", "1")]
    assert resolve("Threads 1.0\n", "1", "a") == [("Threads", "1")]
    assert capsys.readouterr().out == ""
    assert resolve("Threads 1\n", "4", "a") == [("Threads", "4")]
    assert capsys.readouterr().out.splitlines() == [
        f"gurobi uses Threads 4, the run's thread budget (flag --threads), instead of Threads 1 in {opt_file}."]
    assert resolve("Threads 4\n", "1", "dispatch") == [("Threads", "1")]
    assert resolve("Threads 4\n", "1", "invest") == [("Threads", "1")]
    assert len(capsys.readouterr().out.splitlines()) == 1
    reset_run_warnings()
    resolve("Threads 4\n", "1", "invest")
    assert len(capsys.readouterr().out.splitlines()) == 1


# ---------------------------------------------------------------------------
# The native names are real parameter names (checked against the installed wheels; skipped where absent)
# ---------------------------------------------------------------------------

_VALUES = {"time_limit": 60, "mip_gap": 0.01, "threads": 1}


def _set_through_wheel(solver: str, name: str, value) -> None:
    """Set parameter *name* (as FlexTool passes it to the CLI) through the solver's Python wheel; an unknown
    name raises."""
    if solver == "gurobi":
        import gurobipy as gp

        with gp.Env(params={"OutputFlag": 0}) as env, gp.Model(env=env) as m:
            m.setParam(name, value)
    elif solver == "cplex":
        import cplex

        c = cplex.Cplex()  # kept alive: the parameter tree holds only a weak reference to it
        node = c.parameters
        for part in name.split():  # the interactive ``set`` path: "mip tolerances mipgap"
            node = getattr(node, part)
        node.set(value)
    elif solver == "xpress":
        import xpress as xp

        p = xp.problem()
        p.setControl(name, value)
        assert p.getControl(name) == value
    else:
        import coptpy as cp

        cp.Envr().createModel().setParam(name, value)


@pytest.mark.parametrize("solver", ["gurobi", "cplex", "xpress", "copt"])
def test_native_names_are_known_to_the_solver(solver: str) -> None:
    import importlib.util

    from flextool.engine.solve._solver_dispatch import _PARAM_MAP

    module = {"gurobi": "gurobipy", "cplex": "cplex", "xpress": "xpress", "copt": "coptpy"}[solver]
    if importlib.util.find_spec(module) is None:
        pytest.skip(f"{module} not installed")
    for knob, name in _PARAM_MAP[solver].items():
        try:
            _set_through_wheel(solver, name, _VALUES[knob])
        except Exception as exc:  # noqa: BLE001 - a licence problem skips; a name problem fails below
            if any(h in str(exc).lower() for h in ("licen", "size-limited")):
                pytest.skip(f"{solver}: no usable licence ({exc})")
            raise AssertionError(f"{solver} does not know {name!r} ({knob}): {exc}") from exc


def test_whole_number_database_values_are_written_as_integers(tmp_path: Path) -> None:
    """The database stores every number as a float: ``solver_arguments`` ``Threads 4`` arrives as "4.0" and must
    reach the solver as ``4`` (the Xpress library refuses ``1.0`` for an integer control); a fraction keeps its
    digits, and a huge whole number keeps the float form."""
    cfg = SolverConfig(name="gurobi", options={"Threads": "4.0", "Method": 2.0, "MIPGap": "0.0005",
                                               "Cutoff": 1e20, "Seed": "-3.00"})
    params, _ = _passed("gurobi", tmp_path, "", cfg, [])
    assert params == ["Threads=4", "Method=2", "MIPGap=0.0005", "Cutoff=1e+20", "Seed=-3"]


# COPT 8.0.4's summary of an LP stopped by its time limit (coptpy log; copt_cmd prints the library's log).
_COPT_LP_TIMEOUT_LOG = """\
Method   Iteration           Objective  Primal.NInf   Dual.NInf        Time
Dual             0    1.0943977500e+03           12           0       0.00s

Solving unfinished
Status: Timeout  Objective: 1.0943977500e+03  Iterations: 0  Time: 0.00s
"""


def test_copt_lp_time_limit_is_read_with_and_without_a_solution_file(tmp_path: Path) -> None:
    """An LP stopped by its COPT time limit says ``Status: Timeout`` (not the MIP's ``stopped (time limit
    reached)``): ``TIME_LIMIT`` whether COPT wrote a solution file or not, so the acceptance check rejects the
    LP iterate, and a run without a file is told it hit the limit."""
    from flextool.engine.solve._subprocess_solve import _parse_copt_sol

    missing = _parse_copt_sol(tmp_path / "none.sol", _COPT_LP_TIMEOUT_LOG)
    assert (missing.status, missing.primal, missing.objective) == ("TIME_LIMIT", None, None)
    sol = tmp_path / "lp.sol"
    sol.write_text("# Objective value 1094.39775\nC0000001  1\nC0000002  0\n")
    written = _parse_copt_sol(sol, _COPT_LP_TIMEOUT_LOG)
    assert written.status == "TIME_LIMIT" and written.primal == {"C0000001": 1.0, "C0000002": 0.0}
    assert _parse_copt_sol(sol, "Status: Optimal  Objective: 1.0943977500e+03\n").status == "OPTIMAL"


#: The ends of COPT logs (coptpy 8.0.4, the library ``copt_cmd`` runs): the market split MIP stopped by
#: ``NodeLimit 5`` and by ``TimeLimit 0.5``, a small MIP solved, an LP stopped by ``BarIterLimit 1``.
_COPT_MIP_NODE_LIMIT_LOG = """\
Best solution   : 1.000000000
Best bound      : 0.000000000
Best gap        : 100.0000%
Solve time      : 0.21
Solve node      : 5
MIP status      : stopped (node limit reached)
Solution status : integer feasible
"""
_COPT_MIP_TIME_LIMIT_LOG = _COPT_MIP_NODE_LIMIT_LOG.replace("node limit", "time limit")
_COPT_MIP_SOLVED_LOG = "MIP status      : solved\nSolution status : integer optimal (relative gap limit 0.0001)\n"
_COPT_LP_ITER_LIMIT_LOG = "Solving unfinished\nStatus: IterLimit  Objective: 0.0000000000e+00  Iterations: 1(0)\n"


@pytest.mark.parametrize("log, status, gap", [
    (_COPT_MIP_NODE_LIMIT_LOG, "NODE_LIMIT", 1.0),
    (_COPT_MIP_NODE_LIMIT_LOG.replace("(node limit reached)", "(interrupted by user)"), "INTERRUPTED", None),
    (_COPT_MIP_NODE_LIMIT_LOG.replace("(node limit reached)", "(something new)"), "STOPPED", None),
    (_COPT_MIP_TIME_LIMIT_LOG, "TIME_LIMIT", 1.0),
    (_COPT_MIP_SOLVED_LOG, "OPTIMAL", None),
    (_COPT_LP_ITER_LIMIT_LOG, "ITERATION_LIMIT", None),   # the LP log has no gap line
    (_COPT_LP_ITER_LIMIT_LOG.replace("IterLimit", "Numerical"), "NUMERICAL", None),
    ("", "NO_STATUS", None),   # Logging 0: the log says nothing
])
def test_copt_status_with_a_solution_comes_from_its_log(log, status, gap, tmp_path: Path) -> None:
    """Review L2-2: only ``MIP status : solved`` / ``Status: Optimal`` is optimal; every other limit or an
    interrupt is named, a limit's gap is read from the log (a MIP at a search limit keeps its solution, item 246),
    and values with no status in the log are ``NO_STATUS``."""
    from flextool.engine.solve._subprocess_solve import _parse_copt_sol

    sol = tmp_path / "mip.sol"
    sol.write_text("# Objective value 1\nC0000001  1\nC0000002  0\n")
    parsed = _parse_copt_sol(sol, log)
    assert (parsed.status, parsed.primal, parsed.mip_gap) == (status, {"C0000001": 1.0, "C0000002": 0.0}, gap)


def test_copt_mip_at_a_node_limit_keeps_its_solution_naming_it(tmp_path: Path) -> None:
    """Item 246: a COPT MIP stopped by its node limit keeps its solution with a warning naming the limit, COPT's
    NodeLimit and the log's gap; an LP at its iteration limit stops the run (COPT's option is not verified, so
    the message names the solver's own parameter)."""
    from types import SimpleNamespace

    import polars as pl

    from flextool.engine.solve._acceptance import classify_acceptance
    from flextool.engine.solve._subprocess_solve import _parse_copt_sol

    sol_file = tmp_path / "mip.sol"
    sol_file.write_text("# Objective value 1\nC0000001  1\n")
    parsed = _parse_copt_sol(sol_file, _COPT_MIP_NODE_LIMIT_LOG)
    assert (parsed.status, parsed.mip_gap) == ("NODE_LIMIT", 1.0)
    mip_vars = {"u": SimpleNamespace(integer=True, frame=pl.DataFrame({"col_id": [0]}))}
    sol = SimpleNamespace(optimal=False, shim_status=parsed.status, shim_mip_gap=parsed.mip_gap, _vars=mip_vars,
                          shim_solver="copt", shim_primal_feasible=True)
    acc = classify_acceptance(sol, ranges_post=None, solve_name="unit")
    assert acc.accepted and acc.limit_stopped and acc.model_status == "kSolutionLimit", acc
    assert "stopped at its node limit" in acc.message and "(COPT NodeLimit, in the solver's option file" in acc.message
    assert "(relative MIP gap 100.0000%)" in acc.message

    lp = _parse_copt_sol(sol_file, _COPT_LP_ITER_LIMIT_LOG)
    lp_sol = SimpleNamespace(optimal=False, shim_status=lp.status, shim_mip_gap=lp.mip_gap, _vars={},
                             shim_solver="copt", shim_primal_feasible=True)
    rejected = classify_acceptance(lp_sol, ranges_post=None, solve_name="unit")
    assert not rejected.accepted and rejected.model_status == "kIterationLimit"
    assert "it is a linear problem" in rejected.message
    assert "(the solver's own iteration limit parameter)" in rejected.message


_RECORDING_SOLVER = """\
#!{py}
import json, os, sys
with open({record!r}, "w") as fh:
    json.dump({{"argv": sys.argv[1:], "cwd": os.getcwd(), "stdin": sys.stdin.read()}}, fh)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="the solver stub is a POSIX shebang script")
@pytest.mark.parametrize("solver", ["gurobi", "cplex", "xpress", "copt"])
def test_a_work_folder_with_a_space_reaches_the_solver_as_bare_file_names(solver: str, tmp_path: Path,
                                                                          monkeypatch) -> None:
    """The solver program runs in ``<work>/solve_data/subprocess`` and gets bare file names, so a work folder
    named ``my runs`` needs no quoting in any console script (nor a Windows path's backslashes any escaping).
    The console script is saved there as ``<solve>.script``."""
    import json
    import stat
    import sys

    from flextool.engine.solve import _subprocess_solve as sps
    from tests.engine.test_subprocess_gurobi_name_recovery import _cost_lp

    record = tmp_path / "record.json"
    stub = tmp_path / "solver_stub"
    stub.write_text(_RECORDING_SOLVER.format(py=sys.executable, record=str(record)))
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(sps, "_find_solver_binary", lambda name: stub)
    pb = _cost_lp(["coal fired", "gas"])
    work = tmp_path / "my runs"
    with pytest.raises(RuntimeError, match="no solution file"):
        sps._solve_commercial_subprocess(pb, solver, {"Threads" if solver != "xpress" else "THREADS": 1},
                                         solve_name="y2020 dispatch", logger=None, work_folder=work)
    got = json.loads(record.read_text())
    out_dir = work / "solve_data" / "subprocess"
    assert Path(got["cwd"]) == out_dir
    assert (out_dir / "y2020_dispatch.mps").is_file()
    result = "y2020_dispatch.json" if solver == "gurobi" else "y2020_dispatch.sol"
    if solver == "gurobi":
        assert got["argv"] == ["Threads=1", f"ResultFile={result}", "y2020_dispatch.mps"]
    else:
        read, write = {"cplex": ("read", f"write {result} sol"), "xpress": ("readprob", f"writeslxsol {result}"),
                       "copt": ("read", f"write {result}")}[solver]
        lines = got["stdin"].splitlines()
        assert f"{read} y2020_dispatch.mps" in lines and write in lines, lines
        # Review F7: the script is kept next to the MPS for a replay by hand.
        assert (out_dir / "y2020_dispatch.script").read_text() == got["stdin"]
    assert "my runs" not in json.dumps(got["argv"]) + got["stdin"]


# ---------------------------------------------------------------------------
# Item 247: a stopped child and an earlier run's result files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("returncode, expected", [
    (-9, "was stopped by SIGKILL (often the out-of-memory killer)"),
    (-15, "was stopped by SIGTERM"),
    (0xC0000409, "failed with exit code 0xC0000409"),
    (0xC0000017, "failed with exit code 0xC0000017 (out of memory)"),
    (7, "failed with exit code 7"),
])
def test_a_child_exit_is_described_by_its_signal_or_code(returncode: int, expected: str) -> None:
    """POSIX gives a signal as a negative exit code; Windows status codes read in hex."""
    from flextool.engine.solve._subprocess_solve import _describe_child_exit

    if returncode < 0 and sys.platform == "win32":
        pytest.skip("POSIX signals")
    assert _describe_child_exit(returncode) == expected


def _solved_once(work: Path):
    """Solve the tiny cost LP through the real HiGHS child in *work*: its ``s.sol`` stays there (objective 20)."""
    from flextool.engine.solve._subprocess_solve import solve_via_subprocess
    from tests.engine.test_subprocess_gurobi_name_recovery import _cost_lp

    sol = solve_via_subprocess(_cost_lp(["a", "b", "c"]), "highs", None, solve_name="s", work_folder=work)
    assert sol.optimal and sol.obj == pytest.approx(20.0)
    stale = work / "solve_data" / "subprocess" / "s.sol"
    assert stale.is_file()
    return stale


def _fake_child(monkeypatch, returncode: int) -> list[list[str]]:
    """Replace the HiGHS child with one that writes nothing and ends with *returncode*."""
    import subprocess

    from flextool.engine.solve import _subprocess_solve as sps

    calls: list[list[str]] = []

    def run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, returncode)

    monkeypatch.setattr(sps.subprocess, "run", run)
    return calls


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_a_highs_child_stopped_by_a_signal_is_named_and_the_old_solution_is_not_read(tmp_path: Path,
                                                                                      monkeypatch) -> None:
    """Item 247: a ``--save-memory`` child killed by SIGKILL (exit -9) stops the run with an error naming the
    signal and the preserved folder; an earlier run's ``.sol`` of the same solve is neither read nor kept."""
    from flextool.engine.solve._subprocess_solve import solve_via_subprocess
    from tests.engine.test_subprocess_gurobi_name_recovery import _cost_lp

    work = tmp_path / "work"
    stale = _solved_once(work)
    calls = _fake_child(monkeypatch, -9)
    with pytest.raises(RuntimeError) as err:
        solve_via_subprocess(_cost_lp(["a", "b"], [5.0, 6.0]), "highs", None, solve_name="s", work_folder=work)
    assert len(calls) == 1
    assert "SIGKILL" in str(err.value) and "out-of-memory" in str(err.value)
    assert str(stale.parent) in str(err.value)
    assert not stale.exists()


def test_a_highs_child_ending_with_exit_1_and_no_solution_is_an_error(tmp_path: Path, monkeypatch) -> None:
    """Review M1-1: exit 1 is also an uncaught exception in the child and, on Windows, ``taskkill /F``.  Without
    a solution file it stops the run with a clear error, never a raw FileNotFoundError and never the earlier
    run's ``.sol`` (objective 20) as this solve's answer."""
    from flextool.engine.solve._subprocess_solve import solve_via_subprocess
    from tests.engine.test_subprocess_gurobi_name_recovery import _cost_lp

    work = tmp_path / "work"
    stale = _solved_once(work)
    _fake_child(monkeypatch, 1)
    with pytest.raises(RuntimeError, match="ended without writing a solution") as err:
        solve_via_subprocess(_cost_lp(["a", "b"], [5.0, 6.0]), "highs", None, solve_name="s", work_folder=work)
    assert not isinstance(err.value, FileNotFoundError)
    assert "exit code 1" in str(err.value)
    assert not stale.exists()


def test_a_highs_solve_name_is_one_file_token_inside_the_work_folder(tmp_path: Path, monkeypatch) -> None:
    """Review N1: the ``--save-memory`` HiGHS path cleans a solve name into one file-name token, as the commercial
    path does, so ``\\``, ``/`` or ``..`` cannot put (or delete) its files outside ``solve_data/subprocess``."""
    from flextool.engine.solve._subprocess_solve import solve_via_subprocess
    from tests.engine.test_subprocess_gurobi_name_recovery import _cost_lp

    work = tmp_path / "work"
    calls = _fake_child(monkeypatch, 1)
    with pytest.raises(RuntimeError, match="ended without writing a solution"):
        solve_via_subprocess(_cost_lp(["a", "b"]), "highs", None, solve_name="..\\up/y 1", work_folder=work)
    sub = work / "solve_data" / "subprocess"
    mps = Path(calls[0][calls[0].index("--mps") + 1])
    assert mps == sub / ".._up_y_1.mps"
    assert mps.is_file()


def test_a_highs_whole_number_option_from_the_database_runs_in_the_save_memory_child(tmp_path: Path) -> None:
    """An integer option and a switch from ``solver_arguments`` arrive as float text (``"100000.0"``, ``"1.0"``);
    the resolved options pass the ``--save-memory`` child's HiGHS (before, it refused them and exited with an
    error)."""
    from flextool.engine.solve._solver_dispatch import _resolve_effective_highs_options
    from flextool.engine.solve._subprocess_solve import solve_via_subprocess
    from tests.engine.test_subprocess_gurobi_name_recovery import _cost_lp

    options = _resolve_effective_highs_options(
        solver_arguments_map={"simplex_iteration_limit": "100000.0", "mip_detect_symmetry": "1.0"},
        highs_opt_path=None, cli_overrides=None, solve_name="s",
    )
    sol = solve_via_subprocess(_cost_lp(["a", "b"]), "highs", options, solve_name="s", work_folder=tmp_path / "work")
    assert sol.optimal
    opt_lines = (tmp_path / "work" / "solve_data" / "subprocess" / "s.opt").read_text().splitlines()
    assert "simplex_iteration_limit=100000" in opt_lines and "mip_detect_symmetry=true" in opt_lines


_SILENT_SOLVER = """\
#!{py}
import sys
sys.stdin.read()
"""

#: An earlier run's OPTIMAL result of solve ``probe``, as each solver writes it (objective 123).
_STALE_RESULTS = {
    "gurobi": {"probe.json": '{"SolutionInfo": {"Status": 2, "ObjVal": 123.0, "SolCount": 1}, '
                             '"Vars": [{"VarName": "C0000001", "X": 0.7}, {"VarName": "C0000002", "X": 0.3}]}'},
    "xpress": {"probe.sol.slx": "NAME probe\n C C0000001 0.7\n C C0000002 0.3\nENDATA\n",
               "probe.sol.prt": "Optimal solution found\nObjective function value is 123\n"},
}


@pytest.mark.skipif(sys.platform == "win32", reason="the solver stub is a POSIX shebang script")
@pytest.mark.parametrize("solver", sorted(_STALE_RESULTS))
def test_a_solver_that_exits_0_without_a_result_never_gets_an_earlier_runs_file(solver: str, tmp_path: Path,
                                                                                monkeypatch) -> None:
    """A solver program (or a wrapper) that exits 0 and writes nothing, with an OPTIMAL result of an earlier run
    of the same solve in the work folder: the run stops with the no-solution error.  (Before the fix: OPTIMAL,
    objective 123, the earlier run's values.)"""
    import stat

    from flextool.engine.solve import _subprocess_solve as sps
    from tests.engine.test_subprocess_gurobi_name_recovery import _cost_lp

    stub = tmp_path / "solver_stub"
    stub.write_text(_SILENT_SOLVER.format(py=sys.executable))
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(sps, "_find_solver_binary", lambda name: stub)
    work = tmp_path / "work"
    out_dir = work / "solve_data" / "subprocess"
    out_dir.mkdir(parents=True)
    for name, text in _STALE_RESULTS[solver].items():
        (out_dir / name).write_text(text)
    with pytest.raises(RuntimeError, match="produced no solution file we could parse"):
        sps._solve_commercial_subprocess(_cost_lp(["a", "b"]), solver, None, solve_name="probe", logger=None,
                                         work_folder=work)
    assert not any((out_dir / name).exists() for name in _STALE_RESULTS[solver])


# ---------------------------------------------------------------------------
# Xpress MIPs: ``mipoptimize``, and the status from the log (item 230 row 20)
# ---------------------------------------------------------------------------


def test_xpress_script_solves_a_mip_with_mipoptimize_and_an_lp_with_lpoptimize(tmp_path: Path) -> None:
    """``lpoptimize`` on a MIP solves only its LP relaxation: a MIP gets ``mipoptimize``, an LP keeps the
    script it had."""
    from flextool.engine.solve._subprocess_solve import _xpress_script

    args = (Path("optimizer"), Path("s.mps"), Path("s.sol"), (("THREADS", "1"),))
    _, lp = _xpress_script(*args)
    _, mip = _xpress_script(*args, is_mip=True)
    assert lp == "setControl THREADS 1\nreadprob s.mps\nlpoptimize\nwriteslxsol s.sol\nwriteprtsol s.sol\nquit\n"
    assert mip == lp.replace("lpoptimize", "mipoptimize")


@pytest.mark.parametrize("controls, appended", [
    ((("MIPLOG", "0"),), ["setControl MIPLOG 1"]),
    ((("OUTPUTLOG", "0"),), ["setControl OUTPUTLOG 1"]),
    ((("outputlog", "3"), ("MIPLOG", "0")), ["setControl OUTPUTLOG 1", "setControl MIPLOG 1"]),
    ((("MIPLOG", "0"), ("MIPLOG", "3")), []),   # the last value wins in Xpress, and it logs
    ((("MIPLOG", "-5"), ("OUTPUTLOG", "1")), []),
    ((), []),   # Xpress' defaults log
])
def test_xpress_mip_script_sets_back_the_log_flextool_reads(controls, appended) -> None:
    """R2: on a MIP, a user's ``OUTPUTLOG`` other than 1 or ``MIPLOG 0`` would hide the lines FlexTool reads how
    the search ended from; ``setControl <NAME> 1`` follows the user's controls.  An LP's script is unchanged."""
    from flextool.engine.solve._subprocess_solve import _xpress_script

    args = (Path("optimizer"), Path("s.mps"), Path("s.sol"), (("THREADS", "1"), *controls))
    _, mip = _xpress_script(*args, is_mip=True)
    _, lp = _xpress_script(*args)
    user = ["setControl THREADS 1", *(f"setControl {n} {v}" for n, v in controls)]
    assert mip.splitlines()[: len(user) + len(appended) + 1] == [*user, *appended, "readprob s.mps"]
    assert lp.splitlines()[: len(user) + 1] == [*user, "readprob s.mps"]


@pytest.mark.parametrize("is_mip, flags, cfg_limit, opt_text, expected", [
    (True, ["--solver-time-limit", "59.5"], None, "", "-60"),   # rounded up, then negated: never -59
    (False, ["--solver-time-limit", "59.5"], None, "", "60"),
    (True, [], 30.2, "", "-31"),   # the solve's own time limit too
    (True, [], None, "MAXTIME 100\n", "100"),   # written by the user: passed as written
    (True, [], None, "MAXTIME -100\n", "-100"),
    (True, ["--solver-time-limit", "45"], None, "MAXTIME 100\n", "-45"),
])
def test_xpress_mip_time_limit_is_a_hard_maxtime(is_mip, flags, cfg_limit, opt_text, expected, tmp_path) -> None:
    """R3: FlexTool's own time limit (the flag, the solve's ``time_limit``) on an Xpress MIP is ``MAXTIME
    -<limit>``, a hard limit (a positive MAXTIME stops a MIP only once it has a solution); the watchdog takes the
    absolute value."""
    opt_file = tmp_path / "xpress.opt"
    opt_file.write_text(opt_text)
    resolved = resolve_commercial_options("xpress", opt_file=opt_file, run_options=_options(flags), is_mip=is_mip,
                                          solver_config=SolverConfig(name="xpress", time_limit=cfg_limit))
    assert [o.value for o in resolved if o.name == "MAXTIME"] == [expected]
    assert commercial_time_limit("xpress", resolved) == abs(float(expected))


#: The ends of Xpress MIP logs (the ``xpress`` 9.8.1 package's library log, which the console prints): the
#: market split MIP stopped by ``MAXTIME 1`` / ``MAXNODE 5`` / ``MAXMIPSOL 1``, stopped by ``MAXNODE 1`` with
#: heuristics and cuts off (no integer solution), a small MIP solved, an infeasible MIP.
_XPRESS_TIME_LIMIT_LOG = """\
Optimal solution found
STOPPING - MAXTIME limit reached (TIMELIMIT=1.00  time=1.00).
 *** Search unfinished ***    Time:     1.00 Nodes:       4693
Final MIP objective                   : 2.300000000000000e+01
Final MIP bound                       : 0.000000000000000e-01
  Solution time / primaldual integral :      1.00s/ 100.000000%
  Number of solutions found / nodes   :        11 /      4693
"""
_XPRESS_NODE_LIMIT_LOG = """\
STOPPING - MAXNODE limit reached (MAXNODE=5  nodes=5).
 *** Search unfinished ***    Time:     0.08 Nodes:          5
Final MIP objective                   : 3.999999999999999e+01
Final MIP bound                       : 1.000000000000000e+01
  Number of solutions found / nodes   :         7 /         5
"""
_XPRESS_SOLUTION_LIMIT_LOG = """\
STOPPING - MAXMIPSOL limit reached (MAXMIPSOL=1  mipsols=1).
 *** Maximum integer solutions *** Time:     0.01 Nodes:          0
 *** Search unfinished ***    Time:     0.01 Nodes:          0
Final MIP objective                   : 1.120000000000000e+03
Final MIP bound                       : 0.000000000000000e-01
  Number of solutions found / nodes   :         1 /         0
"""
_XPRESS_NODE_LIMIT_NO_SOLUTION_LOG = """\
STOPPING - MAXNODE limit reached (MAXNODE=1  nodes=1).
 *** Maximum nodes ***
 *** Search unfinished ***    Time:     0.01 Nodes:          1
  Number of solutions found / nodes   :         0 /         1
"""
_XPRESS_SOLVED_LOG = """\
Optimal solution found
 *** Search completed ***
Final MIP objective                   : -1.150000000000000e+01
Final MIP bound                       : -1.150000000000000e+01
  Number of solutions found / nodes   :         1 /         1
"""
_XPRESS_INFEASIBLE_LOG = """\
The problem is infeasible due to row R1
 *** Search completed ***
Problem is integer infeasible
  Number of solutions found / nodes   :         0 /         0
"""
#: An unbounded MIP (xpress 9.8.1 library, mipstatus 7): the search "completes", heuristics may count
#: solutions and print a final objective (review R1), or not.
_XPRESS_UNBOUNDED_LOG = """\
 *** Search completed ***
Problem is unbounded
Final MIP objective                   : -1.000000000000000e+06
Final MIP bound                       : -1.000000000000000e+06
  Number of solutions found / nodes   :         2 /         0
"""
_XPRESS_UNBOUNDED_NO_SOLUTION_LOG = """\
The problem is unbounded due to column C1
 *** Search completed ***
Problem is unbounded
  Number of solutions found / nodes   :         0 /         0
"""
#: An SLX file and the print solution as Xpress writes them; after a search stopped by a limit the print
#: solution still says ``Optimal solution found``.
_XPRESS_SLX = "NAME          noname\n  C C0000001                   1\n  C C0000002                0.75\nENDATA\n"
_XPRESS_PRT = "Solution Statistics\nMinimization performed\nOptimal solution found after 0 iterations\n" \
              "Objective function value is    23.000000\n"


@pytest.mark.parametrize("log, status, objective, gap", [
    (_XPRESS_TIME_LIMIT_LOG, "TIME_LIMIT", 23.0, 1.0),
    (_XPRESS_NODE_LIMIT_LOG, "NODE_LIMIT", 39.99999999999999, 0.75),
    (_XPRESS_SOLUTION_LIMIT_LOG, "SOLUTION_LIMIT", 1120.0, 1.0),
    (_XPRESS_SOLVED_LOG, "OPTIMAL", -11.5, None),
    (" *** Search unfinished ***\n  Number of solutions found / nodes   :         3 /        40\n", "UNFINISHED",
     23.0, None),   # a stop FlexTool does not know: rejected by the acceptance step, never OPTIMAL
])
def test_xpress_mip_status_comes_from_the_log_not_the_print_solution(log, status, objective, gap,
                                                                     tmp_path: Path) -> None:
    """A MIP stopped by MAXTIME / MAXNODE / MAXMIPSOL is that limit (a search limit: kept with a warning, item
    246), with the log's full-precision objective and the gap ``|obj - bound| / max(|obj|, |bound|)``, although
    the print solution says ``Optimal solution found``.  Only ``*** Search completed ***`` is OPTIMAL."""
    from flextool.engine.solve._subprocess_solve import _parse_xpress_sol

    (tmp_path / "s.sol.slx").write_text(_XPRESS_SLX)
    (tmp_path / "s.sol.prt").write_text(_XPRESS_PRT)
    parsed = _parse_xpress_sol(tmp_path / "s.sol", log, is_mip=True)
    assert (parsed.status, parsed.primal) == (status, {"C0000001": 1.0, "C0000002": 0.75})
    assert parsed.objective == pytest.approx(objective, rel=1e-15)
    assert parsed.mip_gap == (None if gap is None else pytest.approx(gap, rel=1e-12))


@pytest.mark.parametrize("log, status", [
    (_XPRESS_INFEASIBLE_LOG, "INFEASIBLE"),
    (_XPRESS_UNBOUNDED_LOG, "UNBOUNDED"),   # R1: not OPTIMAL from "Search completed", the heuristic point unused
    (_XPRESS_UNBOUNDED_NO_SOLUTION_LOG, "UNBOUNDED"),
    (_XPRESS_NODE_LIMIT_NO_SOLUTION_LOG, "NODE_LIMIT"),
    (_XPRESS_NODE_LIMIT_NO_SOLUTION_LOG.replace("MAXNODE limit", "MAXTIME limit"), "TIME_LIMIT"),
    (" *** Search completed ***\n", "NO_INTEGER_SOLUTION"),   # no solution count, no final objective
    ("", "NO_INTEGER_SOLUTION"),   # a log that says nothing proves no integer solution
])
def test_xpress_mip_without_an_integer_solution_never_reads_the_lp_relaxation(log, status, tmp_path: Path) -> None:
    """A search stopped before an integer solution leaves the LP relaxation in the SLX file (and the print
    solution says ``Optimal solution found``): there is no solution, so the run stops with the log's reason."""
    from flextool.engine.solve._subprocess_solve import _NO_SOLUTION_REASONS, _parse_xpress_sol

    (tmp_path / "s.sol.slx").write_text(_XPRESS_SLX)
    (tmp_path / "s.sol.prt").write_text(_XPRESS_PRT)
    parsed = _parse_xpress_sol(tmp_path / "s.sol", log, is_mip=True)
    assert (parsed.status, parsed.primal, parsed.objective) == (status, None, None)
    assert status in _NO_SOLUTION_REASONS


def test_xpress_lp_parse_is_unchanged_by_the_mip_reading(tmp_path: Path) -> None:
    """An LP keeps the print-solution reading: a MIP log line in its output changes nothing."""
    from flextool.engine.solve._subprocess_solve import _parse_xpress_sol

    (tmp_path / "s.sol.slx").write_text(_XPRESS_SLX)
    (tmp_path / "s.sol.prt").write_text(_XPRESS_PRT)
    parsed = _parse_xpress_sol(tmp_path / "s.sol", _XPRESS_NODE_LIMIT_NO_SOLUTION_LOG.replace("MAXNODE", "X"))
    assert (parsed.status, parsed.objective) == ("OPTIMAL", 23.0)


#: A stand-in ``optimizer`` that records its stdin and, like Xpress after a MIP search stopped before any
#: integer solution, writes the LP relaxation and a print solution saying ``Optimal solution found``.
_XPRESS_NO_INCUMBENT_STUB = """\
#!{py}
import sys
from pathlib import Path
script = sys.stdin.read()
Path("stdin.txt").write_text(script)
sol = next(line.split()[1] for line in script.splitlines() if line.startswith("writeslxsol"))
Path(sol).write_text({slx!r})
Path(sol + ".prt").write_text({prt!r})
print({log!r})
"""


@pytest.mark.skipif(sys.platform == "win32", reason="the solver stub is a POSIX shebang script")
def test_an_xpress_mip_runs_mipoptimize_and_stops_without_an_integer_solution(tmp_path: Path, monkeypatch) -> None:
    """A MIP on Xpress is no longer refused: the script holds ``mipoptimize``; a search stopped by its node
    limit before an integer solution stops the run with that reason, never with the LP relaxation's values."""
    import stat

    from flextool.engine.solve import _subprocess_solve as sps
    from tests.engine.test_subprocess_gurobi_name_recovery import _mip_lp

    stub = tmp_path / "optimizer"
    stub.write_text(_XPRESS_NO_INCUMBENT_STUB.format(py=sys.executable, slx=_XPRESS_SLX, prt=_XPRESS_PRT,
                                                     log=_XPRESS_NODE_LIMIT_NO_SOLUTION_LOG))
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(sps, "_find_solver_binary", lambda name: stub)
    work = tmp_path / "work"
    with pytest.raises(RuntimeError, match="stopped at its node limit before finding a solution"):
        sps._solve_commercial_subprocess(_mip_lp(["a", "b"]), "xpress", None, solve_name="probe", logger=None,
                                         work_folder=work)
    script = (work / "solve_data" / "subprocess" / "stdin.txt").read_text().splitlines()
    assert "mipoptimize" in script and "lpoptimize" not in script
