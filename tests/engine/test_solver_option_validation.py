"""Solver option names and values that could change a solver's command script are refused (item 236 A1, A4).

CPLEX / Xpress / COPT read their options as lines of a stdin command script, Gurobi as ``Name=value`` arguments and
HiGHS (``--save-memory``) as lines of an option file.  No solver binary is needed: the option stack and the script
builders are pure functions.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from flextool.engine.input._solve_config import SolverConfig
from flextool.engine.solve._solver_dispatch import (
    FlexToolUserError,
    _resolve_effective_highs_options,
    resolve_commercial_options,
    solver_workspace,
)
from flextool.engine.solve import _subprocess_solve
from flextool.engine.solve._subprocess_solve import _SCRIPTS, _join_script, _write_commercial_opt_file
from flextool.run_options import RunOptions

_NO_FLAGS = RunOptions.from_env({})
_COMMERCIAL = ["gurobi", "cplex", "xpress", "copt"]

#: (name, value) pairs that must be refused, with the words the message uses for the bad character.
_BAD = [
    ("Threads", "1\nquit", "line break"),
    ("Threads", "1\rquit", "line break"),
    ("Threads", "1 quit", "line break"),
    ("Threads\nquit", "1", "line break"),
    ("Threads", "1;quit", "';'"),
    ("Threads", "1'", "quote"),
    ("Threads", '"1"', "quote"),
    ("Threads", "1#x", "'#'"),
    ("Threads", "a=b", "'='"),
    ("Threads", "1 2", "space"),
    ("Threads", "1\x1b", "control character"),
    ("Thr;eads", "1", "';'"),
    ("Thr=eads", "1", "'='"),
    ("Threads", "1\u2028quit", "line break"),
    ("Threads", "\uff11", "the character"),
    ("Threads", "1$x", "'$'"),
    ("Threads", "[exec x]", "'['"),
    ("Threads", "1\\", "ends with a backslash"),
]


def _resolve(solver: str, *, opt_text: str | None = None, options: dict | None = None):
    return resolve_commercial_options(
        solver, opt_file="cfg/x.opt" if opt_text is not None else None, opt_text=opt_text,
        solver_config=SolverConfig(name=solver, options=options or {}), run_options=_NO_FLAGS, solve_name="invest",
    )


@pytest.mark.parametrize("solver", _COMMERCIAL)
@pytest.mark.parametrize("name, value, words", _BAD)
def test_bad_solver_arguments_are_refused_naming_the_solve(solver, name, value, words) -> None:
    with pytest.raises(FlexToolUserError) as err:
        _resolve(solver, options={name: value})
    message = str(err.value)
    assert "solver_arguments of solve invest" in message and words in message and "Thr" in message


@pytest.mark.parametrize("solver", _COMMERCIAL)
@pytest.mark.parametrize("line, words", [
    ("Threads 1;quit", "';'"),
    ("Threads '1'", "quote"),
    ("Thr\x1beads 1", "control character"),
])
def test_bad_option_file_lines_are_refused_naming_file_and_line(solver, line, words) -> None:
    text = f"# options\n\nRelGap 0.01\n{line}\n"
    with pytest.raises(FlexToolUserError) as err:
        _resolve(solver, opt_text=text)
    assert "cfg/x.opt, line 4" in str(err.value) and words in str(err.value)


@pytest.mark.parametrize("solver", _COMMERCIAL)
def test_option_file_value_ending_in_a_backslash_is_refused(solver) -> None:
    """A Tcl console (Xpress) joins a line ending in a backslash with the next command (``readprob``)."""
    with pytest.raises(FlexToolUserError, match="ends with a backslash") as err:
        _resolve(solver, opt_text="Threads 1\nTUNEROUTPUTPATH C:\\out\\\n")
    assert "cfg/x.opt, line 2" in str(err.value)
    assert [o.value for o in _resolve(solver, options={"A": "C:\\out\\x"})] == ["C:\\out\\x"]


def test_space_in_a_name_is_refused_except_cplex_single_spaces() -> None:
    with pytest.raises(FlexToolUserError, match="space"):
        _resolve("gurobi", options={"MIP Gap": "0.01"})
    with pytest.raises(FlexToolUserError, match="single spaces"):
        _resolve("cplex", options={"mip  tolerances mipgap": "0.01"})
    with pytest.raises(FlexToolUserError, match="start with a letter"):
        _resolve("xpress", options={"1THREADS": "1"})
    assert [(o.name, o.value) for o in _resolve("cplex", options={"mip tolerances mipgap": "0.01"})] == [
        ("mip tolerances mipgap", "0.01")
    ]


@pytest.mark.parametrize("solver", _COMMERCIAL)
def test_legitimate_values_pass(solver: str, tmp_path: Path) -> None:
    """Negative numbers, scientific notation, infinity and dotted names reach the solver unchanged; each script
    has one line per parameter plus the fixed commands."""
    values = {"A": "-1", "B": "1e-4", "C": "inf", "D": "-inf", "E": "1e+20", "F.g": "on", "G": True}
    options = _resolve(solver, options=values)
    assert [o.value for o in options] == ["-1", "1e-4", "inf", "-inf", "1e+20", "on", "1"]
    merged = _write_commercial_opt_file({o.name: o.value for o in options}, tmp_path / "s.opt")
    argv, stdin = _SCRIPTS[solver](Path("bin"), Path("m.mps"), Path("m.sol"), merged)
    fixed = {"gurobi": 3, "cplex": 4, "xpress": 5, "copt": 4}[solver]
    assert len(argv if stdin is None else stdin.splitlines()) == len(values) + fixed


@pytest.mark.parametrize("args, where", [
    ({"presolve": "off\nsolution_file=x"}, "solver_arguments of solve invest"),
    ({"mip_rel_gap": "0.01;x"}, "solver_arguments of solve invest"),
    ({"mip rel gap": "0.01"}, "solver_arguments of solve invest"),
])
def test_highs_solver_arguments_are_checked(args, where) -> None:
    """HiGHS gets the same check: its ``--save-memory`` option file is ``name=value`` lines."""
    with pytest.raises(FlexToolUserError) as err:
        _resolve_effective_highs_options(
            solver_arguments_map=args, highs_opt_path=None, cli_overrides=None, solve_name="invest"
        )
    assert where in str(err.value)


def test_highs_option_file_line_is_named_and_paths_pass_from_solver_config_only(tmp_path: Path) -> None:
    """A full path is taken from ``solver_config/highs.opt`` (trusted) and refused from ``solver_arguments`` (the
    input database); a plain file name from the database is placed in the solve's work folder."""
    with pytest.raises(FlexToolUserError, match=r"cfg/highs\.opt, line 2"):
        _resolve_effective_highs_options(
            solver_arguments_map=None, highs_opt_path=Path("cfg/highs.opt"), cli_overrides=None,
            highs_opt_text="threads=1\npresolve=off # note\n",
        )
    with pytest.raises(FlexToolUserError) as err:
        _resolve_effective_highs_options(
            solver_arguments_map={"solution_file": "/tmp/run/a.sol"}, highs_opt_path=None, cli_overrides=None,
            solve_name="invest",
        )
    assert "'solution_file'" in str(err.value) and "'/tmp/run/a.sol'" in str(err.value)
    assert "solver_arguments of solve invest" in str(err.value) and "input database" in str(err.value)
    assert "solver_config/highs.opt" in str(err.value)
    workspace = tmp_path / "work" / "solve_data" / "subprocess"
    options = _resolve_effective_highs_options(
        solver_arguments_map={"solution_file": "a.sol", "time_limit": "inf", "threads": "-1",
                              "write_solution_to_file": "true"},
        highs_opt_path=Path("cfg/highs.opt"), highs_opt_text="write_model_file = /tmp/run/a.mps\n",
        cli_overrides={"output_flag": False, "mip_rel_gap": 1e-4}, solve_name="invest",
        solver_config=SolverConfig(mip_gap=0.002), workspace=workspace,
    )
    assert options == {"write_model_file": "/tmp/run/a.mps", "mip_rel_gap": 1e-4,
                       "solution_file": str(workspace / "a.sol"), "time_limit": "inf", "threads": "-1",
                       "write_solution_to_file": "true", "output_flag": False}
    assert workspace.is_dir()
    assert solver_workspace(tmp_path / "work") == workspace  # the folder a commercial solver runs in


# ---------------------------------------------------------------------------
# Trust by source (owner decision 2026-11-12): the input database takes no paths and no server options
# ---------------------------------------------------------------------------

#: One file / folder option per solver, as the database would spell it.
_PATH_OPTION = {
    "highs": "log_file", "gurobi": "InputFile", "cplex": "workdir", "xpress": "TUNEROUTPUTPATH", "copt": "LogFile",
}
#: (A name that is not ASCII is already refused from every source by the general check.)
_PATH_VALUES = ["/tmp/x.log", "sub/x.log", "C:x.log", "..", "a..b", ".hidden", "-x", "x\\y", "C:\\x"]


def _resolve_db(solver: str, options: dict, *, opt_text: str | None = None, workspace: Path | None = None):
    """Resolve *options* as the solve's ``solver_arguments`` (the input database) of solve ``invest``."""
    if solver == "highs":
        return _resolve_effective_highs_options(
            solver_arguments_map=options, highs_opt_path=Path("cfg/highs.opt") if opt_text is not None else None,
            highs_opt_text=opt_text, cli_overrides=None, solve_name="invest", workspace=workspace,
        )
    return {o.name: o.value for o in _resolve(solver, opt_text=opt_text, options=options)}


@pytest.mark.parametrize("solver", ["highs", *_COMMERCIAL])
@pytest.mark.parametrize("value", _PATH_VALUES)
def test_database_path_option_is_refused(solver: str, value: str) -> None:
    name = _PATH_OPTION[solver]
    with pytest.raises(FlexToolUserError) as err:
        _resolve_db(solver, {name: value})
    message = str(err.value)
    assert f"{solver} option {name!r}" in message and "solver_arguments of solve invest" in message
    assert "input database" in message and f"solver_config/{solver}.opt" in message


@pytest.mark.parametrize("solver, name", [
    ("highs", "write_basis_file"), ("highs", "mip_improving_solution_file"), ("gurobi", "ResultFile"),
    ("gurobi", "NodefileDir"), ("cplex", "logfile"), ("cplex", "output.intsolfileprefix"), ("cplex", "write"),
    ("xpress", "TUNERMETHODFILE"), ("copt", "LogFile"),
])
def test_every_listed_path_option_is_refused_with_a_path_from_the_database(solver: str, name: str) -> None:
    with pytest.raises(FlexToolUserError, match="plain file name"):
        _resolve_db(solver, {name: "/tmp/x"})


@pytest.mark.parametrize("solver", ["highs", *_COMMERCIAL])
def test_database_plain_file_name_is_accepted_into_the_workspace(solver: str, tmp_path: Path) -> None:
    """HiGHS gets the name as a path in the solve's work folder (the in-process solve runs in the current
    folder); a commercial solver gets the bare name and runs in that folder (``test_subprocess_solver_cli``
    pins the folder it runs in)."""
    name = _PATH_OPTION[solver]
    workspace = tmp_path / "solve_data" / "subprocess"
    options = _resolve_db(solver, {name: "run_1.log"}, workspace=workspace)
    expected = str(workspace / "run_1.log") if solver == "highs" else "run_1.log"
    assert list(options.values()) == [expected]
    # A number or a switch in an option whose name says file / dir (Gurobi NodefileStart, BranchDir) passes.
    numeric = {"highs": "write_solution_to_file", "gurobi": "NodefileStart", "cplex": "mip strategy file",
               "xpress": "TUNERMETHODFILE", "copt": "LogFile"}[solver]
    assert list(_resolve_db(solver, {numeric: "0.5"}).values()) == ["0.5"]


_GUROBI_SERVER = ["ComputeServer", "CSManager", "CSAPIAccessID", "CSAPISecret", "CSGroup", "CSPriority", "CSRouter",
                  "CSTLSInsecure", "CSQueueTimeout", "CSAppName", "CSAuthToken", "CSBatchMode", "CSClientLog",
                  "CSIdleTimeout", "ServerPassword", "ServerTimeout", "TokenServer", "TSPort", "CloudAccessID",
                  "CloudSecretKey", "CloudPool", "CloudHost", "WLSAccessID", "WLSSecret", "WLSToken", "LicenseID",
                  "UserName", "WorkerPool", "WorkerPassword"]


@pytest.mark.parametrize("name", _GUROBI_SERVER)
def test_database_gurobi_server_option_is_refused(name: str) -> None:
    with pytest.raises(FlexToolUserError) as err:
        _resolve_db("gurobi", {name: "server1"})
    message = str(err.value)
    assert f"gurobi option {name!r} (value 'server1')" in message and "another computer" in message
    assert "input database" in message and "solver_config/gurobi.opt" in message


@pytest.mark.parametrize("solver", ["highs", *_COMMERCIAL])
def test_trusted_option_file_keeps_paths_and_server_options(solver: str) -> None:
    """``solver_config/<solver>.opt`` is the user's own machine: a full path (and, for Gurobi, a server) passes."""
    name = _PATH_OPTION[solver]
    if solver == "highs":
        options = _resolve_db(solver, {}, opt_text=f"{name} = /srv/runs/x.log\n")
        assert options == {name: "/srv/runs/x.log"}
        return
    server = "ComputeServer server1.example:61000\nCSAPISecret abc-123\n" if solver == "gurobi" else ""
    options = _resolve_db(solver, {}, opt_text=f"{name} C:\\runs\\x.log\n{server}")
    assert options[name.lower() if solver == "cplex" else name] == "C:\\runs\\x.log"
    if solver == "gurobi":
        assert options["ComputeServer"] == "server1.example:61000" and options["CSAPISecret"] == "abc-123"


@pytest.mark.parametrize("solver", ["highs", *_COMMERCIAL])
@pytest.mark.parametrize("value, words", [("/tmp/x.log quit", "space"), ("/tmp/x;quit", "';'"),
                                          ("C:\\runs\\", "ends with a backslash")])
def test_trusted_option_file_still_refuses_injection(solver: str, value: str, words: str) -> None:
    name = _PATH_OPTION[solver]
    line = f"{name} = {value}" if solver == "highs" else f"{name} {value}"
    with pytest.raises(FlexToolUserError, match=words) as err:
        _resolve_db(solver, {}, opt_text=f"{line}\n")
    assert "cfg/" in str(err.value) and "line 1" in str(err.value)


def test_trusted_flags_are_not_checked_as_the_database(monkeypatch) -> None:
    """The flags / environment variables are trusted (no database limits) but still pass the general check."""
    options = _resolve_effective_highs_options(
        solver_arguments_map=None, highs_opt_path=None, cli_overrides={"log_file": "/srv/runs/x.log"},
    )
    assert options == {"log_file": "/srv/runs/x.log"}
    with pytest.raises(FlexToolUserError, match="flags or environment variables"):
        _resolve_effective_highs_options(
            solver_arguments_map=None, highs_opt_path=None, cli_overrides={"log_file": "/x\nquit"},
        )


def test_script_builders_refuse_a_line_break_that_got_through(monkeypatch) -> None:
    """A4: the last check before the solver starts, independent of the option check."""
    with pytest.raises(FlexToolUserError, match="line break"):
        _join_script("cplex", ["set threads 1\nquit"], ["read m.mps", "optimize", "quit"])
    with pytest.raises(FlexToolUserError, match="backslash"):
        _join_script("xpress", ["setControl THREADS 1\\"], ["readprob m.mps", "lpoptimize", "quit"])
    monkeypatch.setattr(_subprocess_solve, "_parse_native_opt_file", lambda path: {"Threads": "1\nX=2"})
    for solver in _COMMERCIAL:
        with pytest.raises(FlexToolUserError, match="line break"):
            _SCRIPTS[solver](Path("bin"), Path("m.mps"), Path("m.sol"), Path("s.opt"))


@pytest.mark.parametrize("solver, expected", [
    ("cplex", "set threads 1\nset mip tolerances mipgap 0.002\nread m.mps\noptimize\nwrite m.sol sol\nquit\n"),
    ("xpress", "setControl THREADS 1\nsetControl MIPRELSTOP 0.002\nsetControl OUTPUTLOG 1\nreadprob m.mps\n"
               "lpoptimize\nwriteslxsol m.sol\nwriteprtsol m.sol\nquit\n"),
    ("copt", "set Threads 1\nset RelGap 0.002\nset Logging 1\nread m.mps\noptimize\nwrite m.sol\nquit\n"),
    ("gurobi", ["bin", "Threads=1", "MIPGap=0.002", "LogToConsole=1", "ResultFile=m.sol", "m.mps"]),
])
def test_shipped_default_options_give_the_same_script(solver, expected, tmp_path: Path) -> None:
    """A5 golden: the shipped ``<solver>.opt.template`` gives exactly the script / argv of 4.3."""
    import flextool

    template = Path(flextool.__file__).parent / "solver_config" / f"{solver}.opt.template"
    options = _resolve(solver, opt_text=template.read_text())
    merged = _write_commercial_opt_file({o.name: o.value for o in options}, tmp_path / "s.opt")
    argv, stdin = _SCRIPTS[solver](Path("bin"), Path("m.mps"), Path("m.sol"), merged)
    assert (argv if solver == "gurobi" else stdin) == expected
