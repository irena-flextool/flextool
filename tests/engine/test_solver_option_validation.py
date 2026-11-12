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


def test_highs_option_file_line_is_named_and_file_options_are_dropped() -> None:
    with pytest.raises(FlexToolUserError, match=r"cfg/highs\.opt, line 2"):
        _resolve_effective_highs_options(
            solver_arguments_map=None, highs_opt_path=Path("cfg/highs.opt"), cli_overrides=None,
            highs_opt_text="threads=1\npresolve=off # note\n",
        )
    options = _resolve_effective_highs_options(
        solver_arguments_map={"solution_file": "/tmp/run/a.sol", "time_limit": "inf", "threads": "-1"},
        highs_opt_path=None, cli_overrides={"output_flag": False, "mip_rel_gap": 1e-4}, solve_name="invest",
        solver_config=SolverConfig(mip_gap=0.002),
    )
    # Item 236 A2: ``solution_file`` makes HiGHS write at a path, so it is dropped, not checked.
    assert options == {"mip_rel_gap": 1e-4, "time_limit": "inf", "threads": "-1", "output_flag": False}


def test_script_builders_refuse_a_line_break_that_got_through(monkeypatch) -> None:
    """A4: the last check before the solver starts, independent of the option check."""
    with pytest.raises(FlexToolUserError, match="line break"):
        _join_script("cplex", ["set threads 1\nquit"], ["read m.mps", "optimize", "quit"])
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
