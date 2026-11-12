"""Solver options: what each source may set (item 236 A1-A4).

CPLEX / Xpress / COPT read their options as lines of a stdin command script, Gurobi as ``Name=value`` arguments and
HiGHS (``--save-memory``) as lines of an option file, so an option that could add commands (a line break, a
separator, a quote, a trailing backslash, a non-ASCII character) is refused from every source.  Trust then follows
the source (owner decision 2026-11-12, option b): an option from the input database (``solver_arguments``,
``solver_mip_gap``) that names a file or folder takes only a plain file name, placed in the solve's work folder, and
a Gurobi server / cloud / licence option is refused there; the option files, flags and environment variables keep
full paths and server options.  Option names are matched as the solver reads them (Gurobi: case and ``_``; Xpress:
an ``XPRS_`` prefix; CPLEX: dotted spelling and abbreviations), pinned against the installed solver packages
(skipped where a package is absent).  ``solver_precommand`` is never run.

No solver binary is needed: the option stack and the script builders are pure functions.  The end to end runs
through each commercial program are ``commercial_shim/test_item230_commercial_shims.py``; the HiGHS end-to-end runs
here cover in-process and ``--save-memory``.
"""
from __future__ import annotations

import base64
import json
import logging
import sys
from pathlib import Path

import pytest

from flextool.engine.input._solve_config import SolverConfig
from flextool.engine.solve import _solver_dispatch
from flextool.engine.solve._solver_dispatch import (
    _PATH_OPTIONS,
    FlexToolUserError,
    _option_key,
    _resolve_effective_highs_options,
    commercial_option_key,
    is_path_option,
    reset_run_warnings,
    resolve_commercial_options,
    solver_workspace,
    warn_solver_precommands,
)
from flextool.engine.solve._subprocess_solve import _SCRIPTS, _join_script, commercial_params
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


#: A parameter name of each commercial solver, as the database or an option file spells it.
_GAP_NAME = {"gurobi": "MIPGap", "cplex": "mip tolerances mipgap", "xpress": "MIPRELSTOP", "copt": "RelGap"}


@pytest.mark.parametrize("solver", _COMMERCIAL)
def test_empty_value_is_refused_from_the_database(solver) -> None:
    """G2: an empty value shifts a CPLEX script line (CPLEX reads the next line as the value) and is dropped by
    Gurobi and Xpress; it is refused, naming the solve."""
    with pytest.raises(FlexToolUserError, match="the value is empty or missing") as err:
        _resolve(solver, options={"Threads": "1", _GAP_NAME[solver]: ""})
    assert "solver_arguments of solve invest" in str(err.value)
    # A missing value (None) is still skipped, a switch is still 1 / 0.
    assert [(o.name, o.value) for o in _resolve(solver, options={"Threads": None, "Presolve": False})] == [
        ("presolve" if solver == "cplex" else "Presolve", "0")]


def test_highs_empty_value_is_refused_in_process_as_in_the_save_memory_child() -> None:
    """Item 240 B4: HiGHS' option-file reader refused ``solution_file=`` (the --save-memory child stopped), while
    the in-process solve took it; an empty HiGHS value is now refused from every source, before any solve."""
    for where, kwargs in (("solver_arguments of solve invest", {"solver_arguments_map": {"solution_file": ""}}),
                          ("flags or environment variables", {"cli_overrides": {"log_file": ""}})):
        with pytest.raises(FlexToolUserError, match="the value is empty or missing") as err:
            _resolve_effective_highs_options(
                highs_opt_path=None, solve_name="invest", **{"solver_arguments_map": None, "cli_overrides": None,
                                                             **kwargs})
        assert where in str(err.value)
    with pytest.raises(FlexToolUserError, match="cfg/highs.opt, line 1"):
        _resolve_db("highs", {}, opt_text="presolve=\n")
    # The old templates' solution-file lines are still dropped, not refused.
    assert _resolve_db("highs", {}, opt_text="solution_file=\nthreads=1\n") == {"threads": "1"}


@pytest.mark.parametrize("solver", _COMMERCIAL)
def test_option_file_line_without_a_value_is_refused_naming_the_line(solver) -> None:
    """G2: a line of one word was skipped without a word; a CPLEX line of several words without its value read as
    a shorter parameter name with the last word as its value.  Both are refused, naming the file and line."""
    for text in ("Threads 1\nThreads\n", f"Threads 1\n{_GAP_NAME[solver]}\n"):
        with pytest.raises(FlexToolUserError, match="value is missing|value is empty") as err:
            _resolve(solver, opt_text=text)
        assert "cfg/x.opt, line 2" in str(err.value)


def test_cplex_value_that_is_a_name_word_is_refused_from_every_source() -> None:
    with pytest.raises(FlexToolUserError, match="'mip tolerances mipgap' reads as a parameter name"):
        _resolve("cplex", options={"mip tolerances": "mipgap"})
    with pytest.raises(FlexToolUserError, match="value is missing"):
        _resolve("cplex", opt_text="mip tol mipgap\n")
    # Text parameters (also abbreviated) take a word; a switch word, a number and a path pass everywhere.
    options = _resolve("cplex", options={
        "workdir": "scratch", "out intsol": "incumbent", "read fileencoding": "UTF-8", "cpumask": "ffff",
        "emphasis numerical": "yes", "timelimit": "1e+75", "mip tolerances mipgap": "0.01",
    })
    assert [o.value for o in options] == ["scratch", "incumbent", "UTF-8", "ffff", "yes", "1e+75", "0.01"]


def test_cplex_text_parameters_and_switch_words_match_its_parameters() -> None:
    """Pin the CPLEX value rule to ``Cplex().parameters``: its text parameters are the listed ones (plus the
    Interactive Optimizer's ``logfile``), no parameter word is a switch word, and the name of every parameter of
    several words, cut before its last word, refuses that word as its value."""
    cplex = pytest.importorskip("cplex")
    from cplex._internal._parameter_classes import Parameter, ParameterGroup

    leaves: dict[str, object] = {}

    def walk(group, path: list[str]) -> None:
        for attr in (a for a in dir(group) if not a.startswith("_")):
            child = getattr(group, attr, None)
            if isinstance(child, Parameter):
                leaves[" ".join([*path, attr])] = child.default()
            elif isinstance(child, ParameterGroup):
                walk(child, [*path, attr])

    model = cplex.Cplex()  # the parameter objects hold only a weak reference to it
    walk(model.parameters, [])
    assert len(leaves) > 150
    assert {n for n, d in leaves.items() if isinstance(d, str)} | {"logfile"} == _solver_dispatch._CPLEX_TEXT_PARAMETERS
    assert not {w for n in leaves for w in n.split(" ")} & _solver_dispatch._CPLEX_SWITCH_WORDS
    for name in leaves:
        *head, last = name.split(" ")
        if head and " ".join(head) not in _solver_dispatch._CPLEX_TEXT_PARAMETERS:
            assert _solver_dispatch._cplex_value_is_a_name_word(" ".join(head), last), name
        assert not _solver_dispatch._cplex_value_is_a_name_word(name, "2"), name


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
    merged = commercial_params({o.name: o.value for o in options})
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


@pytest.mark.parametrize("solver", ["highs", *_COMMERCIAL])
@pytest.mark.parametrize("file_name", ["gurobi.env", "Gurobi.ENV", "gurobi.env.", "gurobi.lic", "license.dat",
                                       "LICENSE.KEY", "client.ini", "xpauth.xpr", "nethasp.ini"])
def test_database_file_name_a_solver_reads_by_itself_is_refused(solver: str, file_name: str, tmp_path) -> None:
    """Every solver runs in the same work folder, so a file the database names there must not be one a solver
    reads from its folder at start (Gurobi's gurobi.env, a licence file, COPT / Xpress client settings)."""
    with pytest.raises(FlexToolUserError, match="reads by itself") as err:
        _resolve_db(solver, {_PATH_OPTION[solver]: file_name}, workspace=tmp_path)
    assert "solver_arguments of solve invest" in str(err.value) and f"solver_config/{solver}.opt" in str(err.value)
    # Another plain name is taken; the option file is trusted with any name.
    assert _resolve_db(solver, {_PATH_OPTION[solver]: "gurobi_env.log"}, workspace=tmp_path)
    line = f"{_PATH_OPTION[solver]} = {file_name}" if solver == "highs" else f"{_PATH_OPTION[solver]} {file_name}"
    assert list(_resolve_db(solver, {}, opt_text=line + "\n").values()) == [file_name]


@pytest.mark.parametrize("name", _GUROBI_SERVER)
def test_database_gurobi_server_option_is_refused(name: str) -> None:
    with pytest.raises(FlexToolUserError) as err:
        _resolve_db("gurobi", {name: "server1"})
    message = str(err.value)
    assert f"gurobi option {name!r} (value 'server1')" in message and "another computer" in message
    assert "input database" in message and "solver_config/gurobi.opt" in message


@pytest.mark.parametrize("name, value", [
    ("COMPUTE", "1"), ("XPRS_COMPUTE", "1"), ("compute", "1"), ("COMPUTEEXECSERVICE", "svc1"),
    ("XPRS_COMPUTEEXECSERVICE", "svc1"), ("ComputeJobPriority", "5"), ("COMPUTELOG", "2"),
])
def test_database_xpress_remote_solving_control_is_refused(name: str, value: str) -> None:
    """G1: Xpress solves remotely through its Insight Compute interface when ``COMPUTE`` is on; none of the
    ``COMPUTE...`` controls is taken from the input database, in any spelling the library reads."""
    with pytest.raises(FlexToolUserError) as err:
        _resolve_db("xpress", {name: value})
    message = str(err.value)
    assert f"xpress option {name!r} (value {value!r})" in message and "another computer" in message
    assert "solver_arguments of solve invest" in message and "solver_config/xpress.opt" in message


def test_xpress_remote_solving_controls_stay_allowed_in_the_option_file() -> None:
    options = _resolve_db("xpress", {}, opt_text="COMPUTE 1\nCOMPUTEEXECSERVICE svc1\n")
    assert options == {"COMPUTE": "1", "COMPUTEEXECSERVICE": "svc1"}


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
    for solver in _COMMERCIAL:
        with pytest.raises(FlexToolUserError, match="not one name and one value"):
            _SCRIPTS[solver](Path("bin"), Path("m.mps"), Path("m.sol"), (("Threads", "1\nX=2"),))
        with pytest.raises(FlexToolUserError, match="not one name and one value"):
            _SCRIPTS[solver](Path("bin"), Path("m.mps"), Path("m.sol"), (("Threads", ""),))


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
    merged = commercial_params({o.name: o.value for o in options})
    argv, stdin = _SCRIPTS[solver](Path("bin"), Path("m.mps"), Path("m.sol"), merged)
    assert (argv if solver == "gurobi" else stdin) == expected


# ---------------------------------------------------------------------------
# Trusted paths reach the solver; FlexTool's own result file stays in the command
# ---------------------------------------------------------------------------
#
# (Replaces item 236 A2's "path options are dropped from every layer" cases: under the trust split a path from the
# option file is passed on and one from the database is refused, so "dropped with a warning" no longer holds.)

FLEXTOOL_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = FLEXTOOL_ROOT / "tests"

#: FlexTool's own result file in each commercial solver's command (``<sol>`` is the result path).
_OWN_RESULT = {
    "gurobi": "ResultFile=<sol>", "cplex": "write <sol> sol", "xpress": "writeslxsol <sol>", "copt": "write <sol>",
}
_THREADS = {"highs": "threads", "gurobi": "Threads", "cplex": "threads", "xpress": "THREADS", "copt": "Threads"}


@pytest.fixture
def _fresh_run_warnings():
    reset_run_warnings()
    yield
    reset_run_warnings()


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


@pytest.mark.parametrize("solver", _COMMERCIAL)
def test_trusted_path_reaches_the_solver_and_own_result_file_is_kept(solver, tmp_path) -> None:
    target = tmp_path / "elsewhere" / "written_by_the_solver"
    name = _PATH_OPTION[solver]
    resolved = resolve_commercial_options(
        solver, opt_file="cfg/x.opt", opt_text=f"{_THREADS[solver]} 2\n{name} {target}\n",
        solver_config=SolverConfig(name=solver), run_options=_NO_FLAGS, solve_name="invest",
    )
    assert [o.value for o in resolved] == ["2", str(target)]
    merged = commercial_params({o.name: o.value for o in resolved})
    sol = tmp_path / "m.sol"
    argv, stdin = _SCRIPTS[solver](Path("bin"), tmp_path / "m.mps", sol, merged)
    sent = " ".join(argv) + (stdin or "")
    assert str(target) in sent, sent
    assert _OWN_RESULT[solver].replace("<sol>", str(sol)) in (argv if solver == "gurobi" else stdin.splitlines())


@pytest.mark.parametrize("layer", ["option file", "solver_arguments"])
def test_gurobi_result_file_is_flextools_own(layer, tmp_path, caplog, _fresh_run_warnings) -> None:
    """``ResultFile`` names the file FlexTool reads the solution from, so it is dropped from every layer (one
    warning per run, naming the layer) and FlexTool's own stays the only one in the command."""
    opt_text = "Threads 2\n" + ("Result_File mine.sol\n" if layer == "option file" else "")
    options = {"ResultFile": "mine.sol"} if layer == "solver_arguments" else {}
    with caplog.at_level(logging.WARNING):
        for solve in ("invest", "dispatch"):  # two solves of one run: one warning
            resolved = resolve_commercial_options(
                "gurobi", opt_file="cfg/x.opt", opt_text=opt_text,
                solver_config=SolverConfig(name="gurobi", options=options), run_options=_NO_FLAGS, solve_name=solve,
            )
    assert [(o.name, o.value) for o in resolved] == [("Threads", "2")]
    merged = commercial_params({o.name: o.value for o in resolved})
    argv, _ = _SCRIPTS["gurobi"](Path("bin"), tmp_path / "m.mps", tmp_path / "m.json", merged)
    assert [a for a in argv if a.casefold().replace("_", "").startswith("resultfile=")] == [
        f"ResultFile={tmp_path / 'm.json'}"]
    warnings = _warnings(caplog)
    where = "cfg/x.opt" if layer == "option file" else "solver_arguments of solve invest"
    assert len(warnings) == 1 and "Ignoring gurobi option" in warnings[0] and where in warnings[0], warnings
    with pytest.raises(FlexToolUserError, match="plain file name"):  # a path from the database is refused first
        _resolve("gurobi", options={"ResultFile": "/tmp/x.sol"})


def test_copt_parameters_pass(tmp_path) -> None:
    resolved = resolve_commercial_options(
        "copt", opt_file="cfg/copt.opt", opt_text="Threads 2\nRelGap 0.01\n",
        solver_config=SolverConfig(name="copt", options={"TimeLimit": 60}), run_options=_NO_FLAGS,
    )
    assert [(o.name, o.value) for o in resolved] == [("Threads", "2"), ("RelGap", "0.01"), ("TimeLimit", "60")]


def test_cplex_abbreviations_of_a_path_parameter_are_path_options() -> None:
    """The CPLEX Interactive Optimizer takes any unique abbreviation of each word of a ``set`` command."""
    for name in ("logfile", "log", "workd", "WorkDir", "output intsolfileprefix", "output.intsol", "o i",
                 "read", "write problem"):
        assert is_path_option("cplex", name), name
    for name in ("threads", "timelimit", "mip tolerances mipgap", "output clonelog", "workmem", "lpmethod"):
        assert not is_path_option("cplex", name), name
    # No abbreviations for the others.
    assert not is_path_option("gurobi", "Log") and is_path_option("gurobi", "logfile")
    with pytest.raises(FlexToolUserError, match="plain file name"):
        _resolve("cplex", options={"log": "/tmp/cplex.log"})
    assert [o.value for o in _resolve("cplex", options={"log": "cplex.log"})] == ["cplex.log"]


# ---------------------------------------------------------------------------
# Names matched as the solver reads them (security_a2a3_review F1 / F2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["Log_File", "LOG_FILE", "Sol_Files", "Input_File", "Nodefile_Dir", "result_file_"])
def test_gurobi_path_option_with_underscores_is_a_path_option(name, tmp_path) -> None:
    """The Gurobi library ignores case and ``_`` after the first character (``Log_File`` is ``LogFile``), so the
    database check does too; the option file (trusted) still passes the spelling on."""
    assert is_path_option("gurobi", name)
    with pytest.raises(FlexToolUserError, match="plain file name"):
        _resolve("gurobi", options={name: str(tmp_path / "elsewhere")})
    if commercial_option_key("gurobi", name) != "resultfile":
        resolved = _resolve("gurobi", opt_text=f"{name} {tmp_path / 'elsewhere'}\n")
        assert [(o.name, o.value) for o in resolved] == [(name, str(tmp_path / "elsewhere"))]


@pytest.mark.parametrize("name", ["Compute_Server", "CS_Manager", "WLS_Access_ID", "cloud_access_id"])
def test_gurobi_server_option_with_underscores_is_refused_from_the_database(name) -> None:
    with pytest.raises(FlexToolUserError, match="another computer"):
        _resolve("gurobi", options={name: "server1"})


def test_gurobi_spellings_of_one_parameter_merge_as_one(tmp_path) -> None:
    """``MIPGap`` (option file), ``Mip_Gap`` (solver_arguments) and the gap flag are one Gurobi parameter: one
    argument, with the highest layer's value; ``Time_Limit`` is read as the time limit."""
    from flextool.cli.cmd_run_flextool import build_parser
    from flextool.engine.solve._solver_dispatch import commercial_time_limit

    config = SolverConfig(name="gurobi", options={"Mip_Gap": 0.1, "time_limit": 60})
    resolved = resolve_commercial_options(
        "gurobi", opt_file="cfg/gurobi.opt", opt_text="MIPGap 0.3\nTIME_LIMIT 900\nThreads 1\n",
        solver_config=config, run_options=_NO_FLAGS,
    )
    assert [(o.name, o.value) for o in resolved] == [("Mip_Gap", "0.1"), ("time_limit", "60"), ("Threads", "1")]
    assert commercial_time_limit("gurobi", resolved) == 60
    flagged = resolve_commercial_options(
        "gurobi", opt_file="cfg/gurobi.opt", opt_text="MIPGap 0.3\n", solver_config=config,
        run_options=RunOptions.from_cli(build_parser().parse_args(["x.sqlite", "--solver-mip-gap", "0.05"]),
                                        environ={}),
    )
    merged = commercial_params({o.name: o.value for o in flagged})
    argv, _ = _SCRIPTS["gurobi"](Path("bin"), tmp_path / "m.mps", tmp_path / "m.json", merged)
    gaps = [a for a in argv if a.split("=", 1)[0].replace("_", "").casefold() == "mipgap"]
    assert gaps == ["MIPGap=0.05"], argv


@pytest.mark.parametrize("name", ["XPRS_TUNEROUTPUTPATH", "xprs_tuneroutputpath", "Xprs_TunerMethodFile"])
def test_xpress_path_control_with_prefix_is_a_path_option(name, tmp_path) -> None:
    """The Xpress library's control lookup ignores case and one ``XPRS_`` prefix, so the database check does too."""
    from flextool.engine.solve._solver_dispatch import commercial_time_limit

    assert is_path_option("xpress", name)
    with pytest.raises(FlexToolUserError, match="plain file name"):
        _resolve("xpress", options={name: str(tmp_path / "elsewhere")})
    resolved = _resolve("xpress", opt_text=f"THREADS 2\n{name} {tmp_path / 'elsewhere'}\n",
                        options={"XPRS_MAXTIME": 60})
    assert [(o.name, o.value) for o in resolved] == [("THREADS", "2"), (name, str(tmp_path / "elsewhere")),
                                                     ("XPRS_MAXTIME", "60")]
    assert commercial_time_limit("xpress", resolved) == 60


# ---------------------------------------------------------------------------
# The lists against the installed solver packages
# ---------------------------------------------------------------------------


def test_highs_path_options_are_its_file_options() -> None:
    highspy = pytest.importorskip("highspy")
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    for name in _PATH_OPTIONS["highs"]:
        assert h.getOptionType(name) == (highspy.HighsStatus.kOk, highspy.HighsOptionType.kString), name


def test_gurobi_path_options_are_its_parameters() -> None:
    pytest.importorskip("gurobipy")
    from gurobipy._paramdetails import param_details

    by_key = {n.casefold(): d for n, d in param_details.items()}
    for key in _PATH_OPTIONS["gurobi"]:
        assert key in by_key, key
        assert by_key[key]["values"]["type"] == ("int" if key == "record" else "string"), key


def test_gurobi_server_options_are_its_parameters() -> None:
    """Every listed server option that this gurobipy knows is a string or number parameter (the list also holds
    names of other Gurobi versions, so a missing one is not an error)."""
    pytest.importorskip("gurobipy")
    from gurobipy._paramdetails import param_details

    by_key = {commercial_option_key("gurobi", n): n for n in param_details}
    known = [k for k in _solver_dispatch._SERVER_OPTIONS["gurobi"] if k in by_key]
    assert {"computeserver", "csmanager", "cloudaccessid", "wlsaccessid", "tokenserver"} <= set(known)
    # Every Gurobi parameter whose name starts with a server prefix is refused from the database.
    for key, name in by_key.items():
        if key.startswith(_solver_dispatch._SERVER_OPTION_PREFIXES["gurobi"]):
            assert _solver_dispatch.is_server_option("gurobi", name), name


def test_xpress_server_options_are_its_remote_solving_controls() -> None:
    """The listed Xpress server options are controls of this xpress, and every control with the ``compute`` prefix
    is refused from the database."""
    import warnings

    warnings.simplefilter("ignore")
    xp = pytest.importorskip("xpress")
    controls = xp.problem().getControl()
    assert set(_solver_dispatch._SERVER_OPTIONS["xpress"]) <= set(controls)
    compute = {k for k in controls if k.startswith("compute")}
    assert compute == {"compute", "computeexecservice", "computejobpriority", "computelog"}
    assert all(_solver_dispatch.is_server_option("xpress", k.upper()) for k in compute)
    assert not _solver_dispatch.is_server_option("xpress", "MAXTIME")


def test_listed_names_are_comparison_keys() -> None:
    """The lists hold the comparison-key forms, so a lookup by key finds them."""
    for solver, names in _PATH_OPTIONS.items():
        assert all(_option_key(solver, n) == n for n in names), solver
    for solver, names in _solver_dispatch._SERVER_OPTIONS.items():
        assert all(_option_key(solver, n) == n for n in names), solver


def test_gurobi_library_reads_names_as_the_key_does() -> None:
    """Pin :func:`commercial_option_key` for Gurobi to the library's own lookup (``GRBgetparamtype``, the C
    function behind ``GRBsetparam`` and ``gurobi_cl``'s ``Name=value``): case and ``_`` after the first character
    are ignored; a leading ``_``, and any other character, is not."""
    import ctypes

    gp = pytest.importorskip("gurobipy")
    libs = sorted((Path(gp.__file__).parent / ".libs").glob("libgurobi*.so")) or sorted(
        Path(gp.__file__).parent.glob("libgurobi*.so"))
    if not libs:
        pytest.skip("gurobipy ships no libgurobi shared library on this platform")
    lib = ctypes.CDLL(str(libs[0]))
    lib.GRBemptyenv.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.GRBgetparamtype.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.GRBfreeenv.argtypes = [ctypes.c_void_p]
    env = ctypes.c_void_p()
    assert lib.GRBemptyenv(ctypes.byref(env)) == 0
    try:
        def known(name: str) -> bool:
            return lib.GRBgetparamtype(env, name.encode("latin-1")) > 0

        for name in ("Log_File", "LOG_FILE", "Nodefile_Dir", "Sol_Files", "Result_File", "LogFile_", "M_I_P_Gap",
                     "Compute_Server"):
            assert known(name), name
            assert commercial_option_key("gurobi", name) in {"logfile", "nodefiledir", "solfiles", "resultfile",
                                                             "mipgap", "computeserver"}, name
        for name in ("_LogFile", "Log-File", "Log.File", "Log File", "GRB_LogFile"):
            assert not known(name), name
            assert commercial_option_key("gurobi", name) != "logfile", name
    finally:
        lib.GRBfreeenv(env)


def test_xpress_library_reads_names_as_the_key_does() -> None:
    """Pin :func:`commercial_option_key` for Xpress to the library's control lookup (``getcontrolinfo``): case
    and one ``XPRS_`` prefix are ignored; a second prefix or ``XPRS`` without ``_`` is not."""
    import warnings

    warnings.simplefilter("ignore")
    xp = pytest.importorskip("xpress")
    prob = xp.problem()
    lookup = getattr(prob, "getControlInfo", None) or prob.getcontrolinfo

    def control_id(name: str):
        try:
            return lookup(name)[0]
        except Exception:  # the package's own error type for an unknown name
            return None

    for key in _PATH_OPTIONS["xpress"]:
        for name in (key.upper(), f"XPRS_{key.upper()}", f"xprs_{key}", f"Xprs_{key.title()}"):
            assert control_id(name) == control_id(key.upper()) is not None, name
            assert commercial_option_key("xpress", name) == key, name
        for name in (f"XPRS_XPRS_{key.upper()}", f"XPRS{key.upper()}"):
            assert control_id(name) is None, name
            assert commercial_option_key("xpress", name) != key, name


def test_cplex_path_options_are_its_parameters() -> None:
    cplex = pytest.importorskip("cplex")
    c = cplex.Cplex()
    for stream in (c.set_log_stream, c.set_results_stream, c.set_warning_stream):
        stream(None)
    assert isinstance(c.parameters.workdir.default(), str)
    assert isinstance(c.parameters.output.intsolfileprefix.default(), str)
    # ``logfile`` is an Interactive Optimizer parameter, not in the Python API.
    assert _PATH_OPTIONS["cplex"] == {"workdir", "output intsolfileprefix", "logfile"}


def test_xpress_path_options_are_its_controls() -> None:
    import warnings

    warnings.simplefilter("ignore")
    xp = pytest.importorskip("xpress")
    controls = xp.problem().getControl()
    for key in _PATH_OPTIONS["xpress"]:
        assert isinstance(controls[key], str), key


def test_copt_has_no_string_parameter() -> None:
    """COPT's library parameters are all numbers (its ``LogFile`` is listed for the console only)."""
    cp = pytest.importorskip("coptpy")
    try:
        model = cp.Envr().createModel("probe")
    except Exception as exc:  # licence errors come as the package's own exception type
        pytest.skip(f"coptpy cannot create a model here: {exc}")
    model.setParam("Logging", 0)
    for name in (n for n in dir(cp.COPT.Param) if not n.startswith("_")):
        try:
            info = model.getParamInfo(getattr(cp.COPT.Param, name))
        except AttributeError:  # multi-objective parameters are not model parameters
            continue
        assert all(isinstance(v, (int, float)) for v in info[1:]), (name, info)


# ---------------------------------------------------------------------------
# solver_precommand: a warning once per solve per run, never run
# ---------------------------------------------------------------------------


def test_precommand_warns_once_per_solve_per_run(caplog, _fresh_run_warnings) -> None:
    precommands = {"invest": "lic_hold --server x", "dispatch": "nice -n 19", "empty": "  ", "unset": None}
    with caplog.at_level(logging.WARNING):
        warn_solver_precommands(precommands)
        warn_solver_precommands(precommands)  # a second call of the same run (another runner, a roll)
        first_run = _warnings(caplog)
        reset_run_warnings()
        warn_solver_precommands({"invest": "lic_hold --server x"})
    assert len(first_run) == 2, first_run
    assert "solve invest" in first_run[0] and "lic_hold --server x" in first_run[0]
    assert "solve dispatch" in first_run[1] and "is not run" in first_run[1]
    assert "licence server" in first_run[0] and "for safety" in first_run[0]
    assert len(_warnings(caplog)) == 3


# ---------------------------------------------------------------------------
# HiGHS end to end: in-process and --save-memory
# ---------------------------------------------------------------------------

_E2E_SOLVE = "y2020_2day_dispatch"


def _b64(x, kind: str) -> list:
    return [base64.b64encode(json.dumps(x).encode()).decode(), kind]


@pytest.fixture(scope="module")
def paths_db(tmp_path_factory) -> tuple[str, Path]:
    """``tests.json`` with two scenarios on the ``coal`` LP, both with a ``solver_precommand`` that would create
    a marker file: ``item236_paths`` (HiGHS file options with full paths in ``solver_arguments``) and
    ``item236_names`` (the same options with plain file names): (url, folder)."""
    if str(TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(TESTS_DIR))
    from db_utils import json_to_db  # noqa: E402

    from flextool.update_flextool.db_migration import migrate_database

    folder = tmp_path_factory.mktemp("item236_paths")
    data = json.loads((TESTS_DIR / "fixtures" / "tests.json").read_text())
    precommand = f"{sys.executable} -c \"open(r'{folder / 'precommand_ran'}', 'w').close()\""
    for alt, log_file, model_file in (
        ("item236_paths", str(folder / "log_file_target.log"), str(folder / "model_target.mps")),
        ("item236_names", "named.log", "named.mps"),
    ):
        args = {"type": "map", "index_type": "str", "data": [
            ["log_file", log_file], ["write_model_to_file", "true"], ["write_model_file", model_file],
        ]}
        data["alternatives"].append([alt, ""])
        data["parameter_values"] += [
            ["solve", _E2E_SOLVE, "solver_arguments", _b64(args, "map"), alt],
            ["solve", _E2E_SOLVE, "solver_precommand", _b64(precommand, "str"), alt],
        ]
        data["scenarios"].append([alt, False, ""])
        chain = ["init", "west", "coal", alt]
        for a, before in zip(chain, chain[1:] + [None]):
            data["scenario_alternatives"].append([alt, a, before])
    json_path = folder / "paths.json"
    json_path.write_text(json.dumps(data))
    url = json_to_db(json_path, folder / "paths.sqlite")
    migrate_database(url)
    return url, folder


def _e2e_run_options(url: str, save_memory: bool):
    from flextool.cli.cmd_run_flextool import build_parser

    return RunOptions.from_cli(build_parser().parse_args([url, *(["--save-memory"] if save_memory else [])]),
                               environ={})


@pytest.mark.solver
@pytest.mark.parametrize("save_memory", [False, True], ids=["in_process", "save_memory"])
def test_highs_database_paths_are_refused_and_no_precommand_runs(
        save_memory, paths_db, tmp_path, monkeypatch, _fresh_run_warnings) -> None:
    from flextool.engine import run_chain_from_db

    url, folder = paths_db
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FlexToolUserError) as err:
        run_chain_from_db(url, "item236_paths", work_folder=tmp_path / "work",
                          run_options=_e2e_run_options(url, save_memory))
    assert "input database" in str(err.value) and f"solver_arguments of solve {_E2E_SOLVE}" in str(err.value)
    for target in ("log_file_target.log", "model_target.mps", "precommand_ran"):
        assert not (folder / target).exists(), target


@pytest.mark.solver
@pytest.mark.parametrize("save_memory", [False, True], ids=["in_process", "save_memory"])
def test_highs_database_file_names_land_in_the_work_folder(
        save_memory, paths_db, tmp_path, monkeypatch, caplog, _fresh_run_warnings) -> None:
    from flextool.engine import run_chain_from_db

    url, folder = paths_db
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.WARNING):
        steps = run_chain_from_db(url, "item236_names", work_folder=tmp_path / "work",
                                  run_options=_e2e_run_options(url, save_memory))
    assert steps[_E2E_SOLVE].optimal
    workspace = tmp_path / "work" / "solve_data" / "subprocess"
    assert (workspace / "named.log").is_file() and (workspace / "named.mps").is_file()
    assert not (tmp_path / "named.log").exists() and not (tmp_path / "named.mps").exists()
    assert not (folder / "precommand_ran").exists()
    precommand = [w for w in _warnings(caplog) if w.startswith("solver_precommand")]
    assert len(precommand) == 1 and f"solve {_E2E_SOLVE}" in precommand[0], precommand
    assert _solver_dispatch._warned_precommands == {_E2E_SOLVE}
