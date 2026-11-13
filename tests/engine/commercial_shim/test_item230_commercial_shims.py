"""Item 230 (one option stack, the time-limit rule, the option-file folder) checked against the real solver
packages: gurobipy, cplex, xpress and coptpy behind shims of their command-line programs.

FlexTool's commercial path runs ``gurobi_cl`` / ``cplex`` / ``optimizer`` / ``copt_cmd``.  Those programs
are not installed here, so ``shims/`` stands in for each one: it takes the arguments and the stdin script
exactly as FlexTool writes them, solves with the solver's Python package, writes the result file with the
solver's own writer and prints the console lines FlexTool reads.  Each shim records the parameters **as the
solver reports them after the solve**, so the assertions below check what the solver got, not what FlexTool
meant to send.  The shims' docstrings list what they do not reproduce (the real programs' argument parsing,
their own console text, their exit codes).

Two entry points, both FlexTool's real solve path with the shim found on ``PATH``:

* ``run_chain_from_db`` on a small FlexTool database (``make_check_db.py``: the ``coal`` and
  ``coal_min_load_MIP_wind`` test scenarios per solver), for the LP, the MIP, the option stack and the limits that stop a run;
* ``run_one_solve`` + ``classify_acceptance`` (the cascade's own accept/reject step) on a hand-built market
  split MIP, the one model here hard enough to reach a 1 s time limit within the size-limited licences.

Expected values are set by hand from the layer rules (option file < ``solver_mip_gap`` <
``solver_arguments`` < flag), never read from FlexTool's code.

Run: ``pytest -m commercial_shim tests/engine/commercial_shim -n 0``.
"""
from __future__ import annotations

import functools
import importlib.util
import json
import logging
import os
import random
import stat
import sys
from pathlib import Path

import polars as pl
import pytest
from polar_high import Param, Problem, Sum

SHIMS = Path(__file__).resolve().parent / "shims"
SOLVE = "y2020_2day_dispatch"
SOLVERS = ("gurobi", "cplex", "xpress", "copt")
BINARY = {"gurobi": "gurobi_cl", "cplex": "cplex", "xpress": "optimizer", "copt": "copt_cmd"}
SHIM = {"gurobi": "gurobi_cl_shim.py", "cplex": "cplex_shim.py", "xpress": "optimizer_shim.py",
        "copt": "copt_cmd_shim.py"}
#: The native names of the MIP gap, the time limit and the thread count, from each solver's manual.  The shim
#: records CPLEX parameters by their interactive path (``mip tolerances mipgap``).
GAP = {"gurobi": "MIPGap", "cplex": "mip tolerances mipgap", "xpress": "MIPRELSTOP", "copt": "RelGap"}
TIME = {"gurobi": "TimeLimit", "cplex": "timelimit", "xpress": "MAXTIME", "copt": "TimeLimit"}
THREADS = {"gurobi": "Threads", "cplex": "threads", "xpress": "THREADS", "copt": "Threads"}


# ---------------------------------------------------------------------------
# Solver availability, shims on PATH
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _unavailable(solver: str) -> str | None:
    """Why *solver*'s package cannot solve a one-variable LP here (missing package or licence), else ``None``."""
    try:
        if solver == "gurobi":
            import gurobipy as gp

            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.start()
            m = gp.Model(env=env)
            m.addVar(ub=1.0, obj=-1.0)
            m.optimize()
            ok = m.Status == 2
        elif solver == "cplex":
            import cplex

            c = cplex.Cplex()
            for stream in (c.set_log_stream, c.set_results_stream, c.set_warning_stream):
                stream(None)
            c.variables.add(obj=[-1.0], ub=[1.0])
            c.solve()
            ok = c.solution.is_primal_feasible()
        elif solver == "xpress":
            import warnings

            warnings.simplefilter("ignore")
            import xpress as xp

            p = xp.problem()
            p.setControl("OUTPUTLOG", 0)
            x = p.addVariable(ub=1.0)
            p.setObjective(-1.0 * x)
            p.lpOptimize()
            ok = int(p.attributes.lpstatus) == 1
        else:
            import coptpy as cp

            m = cp.Envr().createModel("probe")
            m.setParam("Logging", 0)
            m.addVar(ub=1.0, obj=-1.0)
            m.solve()
            ok = m.status == 1
    except ImportError as exc:
        return f"{solver}: Python package not installed ({exc})"
    except Exception as exc:  # licence errors come as each package's own exception type
        return f"{solver}: package cannot solve here, licence? ({type(exc).__name__}: {exc})"
    return None if ok else f"{solver}: package did not solve a one-variable LP"


@pytest.fixture(params=SOLVERS)
def solver(request) -> str:
    reason = _unavailable(request.param)
    if reason:
        pytest.skip(reason)
    return request.param


@pytest.fixture
def shim(solver: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put a ``<binary>`` launcher for *solver*'s shim first on ``PATH``; return the shim's record file."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    launcher = bin_dir / BINARY[solver]
    launcher.write_text(
        f"#!{sys.executable}\n"
        "import runpy, sys\n"
        f"sys.argv[0] = {str(SHIMS / SHIM[solver])!r}\n"
        f"runpy.run_path({str(SHIMS / SHIM[solver])!r}, run_name='__main__')\n"
    )
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    record = tmp_path / "shim_record.jsonl"
    monkeypatch.setenv("FLEXTOOL_SHIM_RECORD", str(record))
    for name in ("FLEXTOOL_SOLVER_CONFIG_DIR", "FLEXTOOL_SOLVER_TIME_LIMIT", "FLEXTOOL_SOLVER_MIP_GAP",
                 "FLEXTOOL_HIGHS_TIME_LIMIT", "FLEXTOOL_HIGHS_MIP_GAP"):
        monkeypatch.delenv(name, raising=False)
    from flextool.engine.solve._subprocess_solve import _find_solver_binary

    assert _find_solver_binary(solver) == launcher
    return record


def _calls(record: Path) -> list[dict]:
    """The shim's record, one dict per call of the solver program."""
    if not record.is_file():
        return []
    return [json.loads(line) for line in record.read_text().splitlines()]


def _cfg(tmp_path: Path, solver: str, text: str) -> Path:
    """A solver-config folder holding *text* as ``<solver>.opt``."""
    cfg = tmp_path / "cfg"
    cfg.mkdir(exist_ok=True)
    (cfg / f"{solver}.opt").write_text(text)
    return cfg


def _run_options(db: str, *flags: str):
    from flextool.cli.cmd_run_flextool import build_parser
    from flextool.run_options import RunOptions

    return RunOptions.from_cli(build_parser().parse_args([db, *flags]), environ={})


# ---------------------------------------------------------------------------
# The FlexTool database: LP and MIP scenarios per solver
# ---------------------------------------------------------------------------


def _check_db_module():
    """``make_check_db.py`` next to this file, loaded by path."""
    name = "_item230_make_check_db"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / "make_check_db.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


@pytest.fixture(scope="module")
def db(tmp_path_factory) -> str:
    """The checks' database (``make_check_db.py``): per solver ``lp_<s>``, ``mip_<s>``, ``stack_<s>``,
    ``smg_<s>``; plus ``unknown_gurobi`` and ``iterlimit_xpress``."""
    return _check_db_module().build(tmp_path_factory.mktemp("commercial_shim_db"))


@pytest.fixture(scope="module")
def highs_objective(db, tmp_path_factory) -> dict[str, float]:
    """The objective of the LP and the MIP scenario solved by HiGHS in process (the reference)."""
    from flextool.engine import run_chain_from_db

    out = {}
    for key, scenario in (("lp", "coal"), ("mip", "coal_min_load_MIP_wind")):
        steps = run_chain_from_db(input_db_url=db, scenario_name=scenario,
                                  work_folder=tmp_path_factory.mktemp(f"highs_{key}"))
        out[key] = steps[SOLVE].obj
    return out


def _run(db: str, scenario: str, work: Path, cfg: Path, *flags: str):
    from flextool.engine import run_chain_from_db

    return run_chain_from_db(input_db_url=db, scenario_name=scenario, work_folder=work,
                             solver_config_dir=cfg, run_options=_run_options(db, *flags))


# ---------------------------------------------------------------------------
# LP and MIP through the whole run
# ---------------------------------------------------------------------------


#: A line of each solver library's own log (its banner), to see that the run shows the solver's log.
BANNER = {"gurobi": "Gurobi Optimizer version", "cplex": "Version identifier", "xpress": "FICO Xpress",
          "copt": "Cardinal Optimizer"}


def test_tiny_lp(solver, shim, db, highs_objective, tmp_path, capsys) -> None:
    """The ``coal`` LP through the solver's program: FlexTool's objective equals HiGHS', the parameters of
    the option file reached the solver, and the run shows the solver's own log after the ``Solver args``
    line."""
    cfg = _cfg(tmp_path, solver, f"{THREADS[solver]} 2\n")
    steps = _run(db, f"lp_{solver}", tmp_path / "w", cfg)
    assert steps[SOLVE].optimal
    assert steps[SOLVE].obj == pytest.approx(highs_objective["lp"], rel=1e-6)
    calls = _calls(shim)
    assert calls and all(c["params"] == {THREADS[solver]: 2} and not c["unknown"] for c in calls), calls
    out = capsys.readouterr().out
    args_line = f"Solver args ({solver}): {THREADS[solver]}=2"
    assert args_line in out and BANNER[solver] in out.split(args_line, 1)[1], out[-3000:]


def test_work_folder_with_a_space(solver, shim, db, highs_objective, tmp_path) -> None:
    """The ``coal`` LP with the work folder ``w with space``: the solver program gets bare file names in its own
    folder, so the run solves (a shim refuses a path with a space, as the real consoles' quoting is unknown)."""
    cfg = _cfg(tmp_path, solver, f"{THREADS[solver]} 2\n")
    work = tmp_path / "w with space"
    steps = _run(db, f"lp_{solver}", work, cfg)
    assert steps[SOLVE].optimal
    assert steps[SOLVE].obj == pytest.approx(highs_objective["lp"], rel=1e-6)
    calls = _calls(shim)
    assert calls and all("w with space" not in c.get("stdin", "") + json.dumps(c.get("argv", [])) for c in calls), calls


def test_tiny_mip_with_the_gap_flag(solver, shim, db, highs_objective, tmp_path, capsys) -> None:
    """``--solver-mip-gap 0.0123`` on the ``coal_min_load_MIP_wind`` MIP: the solver reports its gap parameter
    as 0.0123 (read back from the package), and the ``Solver args`` line says the same.  Xpress: FlexTool
    refuses the MIP before the solver program runs."""
    from flextool.engine.solve._solver_dispatch import FlexToolUserError

    cfg = _cfg(tmp_path, solver, f"{GAP[solver]} 0.5\n")
    if solver == "xpress":
        with pytest.raises(FlexToolUserError, match="integer variables.*solves only linear problems"):
            _run(db, "mip_xpress", tmp_path / "w", cfg, "--solver-mip-gap", "0.0123")
        assert _calls(shim) == []
        return
    steps = _run(db, f"mip_{solver}", tmp_path / "w", cfg, "--solver-mip-gap", "0.0123")
    assert steps[SOLVE].optimal
    # Within the requested gap of HiGHS' (near-)optimum.
    assert steps[SOLVE].obj == pytest.approx(highs_objective["mip"], rel=0.0123 + 1e-3)
    calls = _calls(shim)
    assert calls and all(c["is_mip"] and c["params"] == {GAP[solver]: 0.0123} for c in calls), calls
    printed = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("Solver args")]
    assert printed and all(ln == f"Solver args ({solver}): {GAP[solver]}=0.0123" for ln in printed), printed


# ---------------------------------------------------------------------------
# The option stack: file < solver_mip_gap < solver_arguments < flag, as the solver got it
# ---------------------------------------------------------------------------

#: The option file of the stack tests: a gap, a time limit and a thread count.
def _stack_file(solver: str) -> str:
    return f"{GAP[solver]} 0.3\n{TIME[solver]} 900\n{THREADS[solver]} 1\n"


@pytest.mark.parametrize(("scenario", "flags", "gap", "time_limit"), [
    # solver_arguments beat solver_mip_gap and the file; the time-limit flag beats solver_arguments.
    ("stack", ("--solver-time-limit", "700"), 0.1, 700),
    # The gap flag beats solver_arguments; the arguments' time limit beats the file's.
    ("stack", ("--solver-mip-gap", "0.05"), 0.05, 800),
    # solver_mip_gap beats the file; the file's time limit stays.
    ("smg", (), 0.2, 900),
])
def test_option_stack_as_the_solver_got_it(solver, shim, db, tmp_path, scenario, flags, gap, time_limit) -> None:
    cfg = _cfg(tmp_path, solver, _stack_file(solver))
    steps = _run(db, f"{scenario}_{solver}", tmp_path / "w", cfg, *flags)
    assert steps[SOLVE].optimal
    calls = _calls(shim)
    assert calls, "the solver program was not called"
    for call in calls:
        assert not call["unknown"], call
        got = call["params"]
        assert got[GAP[solver]] == pytest.approx(gap), got
        assert got[TIME[solver]] == pytest.approx(time_limit), got
        assert got[THREADS[solver]] == 1, got
        assert len(got) == 3, got


def test_a_solve_split_by_period_keeps_its_solver_and_settings(solver, shim, db, tmp_path) -> None:
    """``split_<s>``: ``realized_periods`` is a 2-D map, so FlexTool splits ``y2020_2day_dispatch`` into one solve
    per period.  Every one of them runs on the solver of the database solve, with its ``solver_mip_gap`` (0.2), its
    ``solver_arguments`` (time limit 800) and the option file (1 thread).  Before the fix the split solves lost the
    solver configuration and ran on HiGHS.  (The solver-wrapper check of feat/solver-security is left out here:
    the wrapper is not on this branch.)"""
    split_solves = _check_db_module().SPLIT_SOLVES
    cfg = _cfg(tmp_path, solver, f"{THREADS[solver]} 1\n")
    steps = _run(db, f"split_{solver}", tmp_path / "w", cfg)
    assert list(steps) == list(split_solves)
    assert all(steps[s].optimal for s in split_solves)
    calls = _calls(shim)
    assert len(calls) == len(split_solves), calls
    for call in calls:
        assert not call["unknown"], call
        assert call["params"] == pytest.approx({GAP[solver]: 0.2, TIME[solver]: 800, THREADS[solver]: 1}), call


@pytest.mark.parametrize("solver", ["gurobi", "cplex", "copt"], indirect=True)
def test_the_realization_solve_flextool_adds_runs_on_the_parents_solver(solver, shim, tmp_path) -> None:
    """``separate_dispatch_noinvest`` (``tests/fixtures/separate_dispatch_noinvest.json``) with ``stoch_rd`` on
    the solver and ``solver_mip_gap`` 0.2: the realization solve ``stoch_rd__realized_dispatch`` FlexTool adds
    also runs on that solver with that gap (before the fix it ran on HiGHS).  Its objective is the hand-computed
    4 380 000 of ``test_separate_dispatch_noinvest_capacity.py``.  Xpress is left out: the model has integers."""
    from flextool.engine import run_chain_from_db
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    check_db = _check_db_module()
    data = json.loads((check_db.REPO / "tests" / "fixtures" / "separate_dispatch_noinvest.json").read_text())
    data = check_db._lp_snapshot()._overlay(data, name="on_solver", base="separate_dispatch_noinvest", values=[
        ["solve", "stoch_rd", "solver", solver], ["solve", "stoch_rd", "solver_mip_gap", 0.2]])
    json_path = tmp_path / "rd.json"
    json_path.write_text(json.dumps(data))
    url = json_to_db(json_path, tmp_path / "rd.sqlite")
    migrate_database(url)
    steps = run_chain_from_db(input_db_url=url, scenario_name="on_solver", work_folder=tmp_path / "w",
                              solver_config_dir=_cfg(tmp_path, solver, f"{THREADS[solver]} 1\n"),
                              run_options=_run_options(url))
    assert list(steps) == ["stoch_rd", "stoch_rd__realized_dispatch"]
    assert steps["stoch_rd__realized_dispatch"].obj == pytest.approx(4_380_000.0, rel=1e-6)
    calls = _calls(shim)
    assert len(calls) == 2, calls
    assert all(c["params"] == pytest.approx({GAP[solver]: 0.2, THREADS[solver]: 1}) for c in calls), calls


@pytest.mark.parametrize("solver", ["gurobi"], indirect=True)
def test_unknown_gurobi_parameter_is_named_with_its_layer(solver, shim, db, tmp_path, caplog) -> None:
    """Gurobi does not know ``NoSuchParameter`` (in ``solver_arguments``): it solves without it, and FlexTool
    warns naming the parameter and the layer.  Gurobi only: FlexTool reads no such message from the others."""
    cfg = _cfg(tmp_path, "gurobi", "Threads 2\n")
    with caplog.at_level(logging.WARNING):
        steps = _run(db, "unknown_gurobi", tmp_path / "w", cfg)
    assert steps[SOLVE].optimal
    calls = _calls(shim)
    assert calls and all(c["unknown"] == ["NoSuchParameter"] for c in calls), calls
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("'NoSuchParameter'" in w and f"solver_arguments of solve {SOLVE}" in w for w in warnings), warnings


#: A parameter that makes the solver read or write at a path, for the option file (each solver's parameter
#: reference; COPT's library has none).  ``workd`` is CPLEX's abbreviation of ``workdir``; ``Nodefile_Dir`` is
#: Gurobi's ``NodefileDir`` (Gurobi ignores ``_`` in names).
FILE_PATH_OPTION = {"gurobi": "Nodefile_Dir", "cplex": "workd", "xpress": "TUNERMETHODFILE"}


def test_path_options_follow_the_source_and_precommand_is_not_used(
        solver, shim, db, highs_objective, tmp_path, caplog) -> None:
    """Item 236, the whole run through the solver's program.  A path parameter in ``solver_arguments`` (the input
    database, ``paths_<s>``) is refused before the solver starts, naming the parameter and the solve, and nothing
    is written at its path; the solve's ``solver_precommand`` is never run (its marker file is never created; COPT,
    with no database path, solves and warns once).  The same kind of parameter with a full path in the option file
    (trusted) reaches the solver, and FlexTool's own result file is still used (the LP's objective is HiGHS')."""
    from flextool.engine.solve._solver_dispatch import FlexToolUserError

    check_db = _check_db_module()
    folder = Path(db.removeprefix("sqlite:///")).parent
    marker = check_db.precommand_marker(folder, solver)
    cfg = _cfg(tmp_path, solver, f"{THREADS[solver]} 2\n")
    with caplog.at_level(logging.WARNING):
        if solver in check_db.ARG_PATH:
            with pytest.raises(FlexToolUserError) as err:
                _run(db, f"paths_{solver}", tmp_path / "w", cfg)
            message = str(err.value)
            assert repr(check_db.ARG_PATH[solver]) in message and f"solver_arguments of solve {SOLVE}" in message
            assert "input database" in message
            assert not _calls(shim), "the solver program was called"
        else:
            steps = _run(db, f"paths_{solver}", tmp_path / "w", cfg)
            assert steps[SOLVE].optimal
            precommand = [r.getMessage() for r in caplog.records
                          if r.levelno == logging.WARNING and r.getMessage().startswith("solver_precommand")]
            assert len(precommand) == 1 and f"solve {SOLVE}" in precommand[0] and "is not run" in precommand[0]
    assert not check_db.path_option_target(folder, solver).exists()
    assert not marker.exists(), "solver_precommand was run"
    if solver not in FILE_PATH_OPTION:
        return
    trusted = tmp_path / "solver_files"
    trusted.mkdir()
    cfg = _cfg(tmp_path, solver, f"{THREADS[solver]} 2\n{FILE_PATH_OPTION[solver]} {trusted}\n")
    steps = _run(db, f"lp_{solver}", tmp_path / "w2", cfg)
    assert steps[SOLVE].optimal
    assert steps[SOLVE].obj == pytest.approx(highs_objective["lp"], rel=1e-6)
    calls = _calls(shim)
    assert calls and all(c["params"][THREADS[solver]] == 2 for c in calls), calls
    if solver == "gurobi":
        assert all(c["result_file"].endswith(f"{SOLVE}.json") for c in calls), calls
    sent = json.dumps([c.get("argv", []) for c in calls]) + "".join(c.get("stdin", "") for c in calls)
    assert str(trusted) in sent, sent


# ---------------------------------------------------------------------------
# Limits: an LP that hits one stops the run; a MIP at its time limit keeps its solution (D2)
# ---------------------------------------------------------------------------


def test_lp_at_a_limit_stops_the_run(solver, shim, db, tmp_path, caplog) -> None:
    """Gurobi / CPLEX / COPT: ``--solver-time-limit 1e-6`` on the LP, which the solver obeys at once, before it
    has a solution.  Xpress takes whole seconds (FlexTool rounds the limit up to 1 s, more than this LP needs),
    so its run gets an LP iteration limit of 1 instead (``solver_arguments``, presolve off), which leaves an
    unfinished iterate in the solution file.  Every run must stop, saying why, and never use the iterate.

    * Gurobi (JSON status) and COPT (console ``Status: Timeout``): "stopped at its time limit before finding a
      solution".
    * CPLEX: it writes no solution file without a solution; its console says ``time limit exceeded``, so the
      run stops with the same "stopped at its time limit before finding a solution".
    * Xpress: the print solution says ``Problem is unfinished``; the acceptance check rejects it."""
    from flextool.engine.input._solve_state import FlexToolSolveError
    from flextool.engine.solve._solver_dispatch import FlexToolUserError

    cfg = _cfg(tmp_path, solver, f"{THREADS[solver]} 1\n")
    if solver == "xpress":
        scenario, flags = "iterlimit_xpress", ()
    else:
        scenario, flags = f"lp_{solver}", ("--solver-time-limit", "1e-6")
    expected_error = FlexToolSolveError if solver == "xpress" else FlexToolUserError
    with caplog.at_level(logging.INFO), pytest.raises(expected_error) as excinfo:
        _run(db, scenario, tmp_path / "w", cfg, *flags)
    if solver in ("gurobi", "copt", "cplex"):
        assert "stopped at its time limit before finding a solution" in str(excinfo.value), excinfo.value
    else:
        errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
        assert any(f"non-optimal solve for {SOLVE}" in e and "(solver status UNFINISHED)" in e for e in errors), errors
    (call,) = _calls(shim)
    if solver == "xpress":
        assert call["params"] == {"THREADS": 1, "LPITERLIMIT": 1, "PRESOLVE": 0} and call["status"] != 1, call
    else:
        assert call["params"] == {THREADS[solver]: 1, TIME[solver]: pytest.approx(1e-6)}, call


def _market_split(m: int = 6, seed: int = 1) -> Problem:
    """Cornuejols-Dawande market split as a minimisation: ``min sum(s+ + s-)`` s.t.
    ``sum_j a_ij x_j + s+_i - s-_i = floor(sum_j a_ij / 2)``, ``x`` binary, ``a_ij`` in 0..99, ``n = 10(m-1)``.

    An incumbent is trivial (any ``x`` with slacks) and the LP bound is 0, but closing the gap needs an
    enumeration: with m = 6 (50 binaries, 6 rows) no solver here finishes in a second, while the model is far
    inside every size-limited licence."""
    n = 10 * (m - 1)
    rng = random.Random(seed)
    a = [[rng.randint(0, 99) for _ in range(n)] for _ in range(m)]
    rows, cols = [f"i{i}" for i in range(m)], [f"j{j}" for j in range(n)]
    pb = Problem()
    x = pb.add_var("x", ("j",), pl.DataFrame({"j": cols}), lower=0.0, upper=1.0, integer=True)
    over = pl.DataFrame({"i": rows})
    sp = pb.add_var("sp", ("i",), over)
    sm = pb.add_var("sm", ("i",), over)
    coef = pl.DataFrame({"i": [r for r in rows for _ in cols], "j": cols * m,
                         "value": [float(v) for row in a for v in row]})
    rhs = pl.DataFrame({"i": rows, "value": [float(sum(row) // 2) for row in a]})
    pb.add_cstr("split", over=over, sense="==",
                lhs_terms={"ax": Sum(x * Param(("i", "j"), coef, name="a"), over="j"), "sp": sp, "sm": -1.0 * sm},
                rhs_terms={"d": Param(("i",), rhs, name="d")})
    pb.set_objective(Sum(sp, over="i") + Sum(sm, over="i"))
    return pb


def test_mip_at_its_time_limit_keeps_the_solution(solver, shim, tmp_path) -> None:
    """``--solver-time-limit 1`` on the market split MIP: the solver stops at 1 s with an incumbent, FlexTool
    reads ``TIME_LIMIT`` and the solver's gap, and the cascade's acceptance step keeps the solution with a
    warning (D2).  Xpress: FlexTool refuses the MIP before the solver program runs."""
    from flextool.engine.input._solve_config import SolverConfig
    from flextool.engine.solve._acceptance import classify_acceptance
    from flextool.engine.solve._solver_dispatch import FlexToolUserError, run_one_solve

    cfg = _cfg(tmp_path, solver, f"{THREADS[solver]} 2\n")
    opts = _run_options("unused.sqlite", "--solver-time-limit", "1")
    kwargs = dict(solve_name="market_split", work_folder=tmp_path / "w", run_options=opts, solver_config_dir=cfg)
    if solver == "xpress":
        with pytest.raises(FlexToolUserError, match="integer variables"):
            run_one_solve(_market_split(), SolverConfig(name=solver), **kwargs)
        assert _calls(shim) == []
        return
    sol = run_one_solve(_market_split(), SolverConfig(name=solver), **kwargs)
    (call,) = _calls(shim)
    assert call["params"] == {THREADS[solver]: 2, TIME[solver]: 1.0}, call
    assert call["is_mip"] and call["mip_gap"] is not None and call["mip_gap"] > 0.0, (
        f"{solver} closed the market split gap within 1 s; the model is no longer hard enough: {call}"
    )
    assert sol.shim_status == "TIME_LIMIT"
    assert sol.shim_mip_gap == pytest.approx(call["mip_gap"], abs=1e-4)
    acc = classify_acceptance(sol, ranges_post=None, solve_name="market_split")
    assert acc.accepted and acc.time_limited and acc.model_status == "kTimeLimit", acc
    assert f"(relative MIP gap {call['mip_gap']:.4%})" in acc.message, acc.message


#: The native name of the MIP node limit (CPLEX by its interactive path).
NODES = {"gurobi": "NodeLimit", "cplex": "mip limits nodes", "copt": "NodeLimit"}


def test_mip_at_a_node_limit_is_rejected_by_name(solver, shim, tmp_path) -> None:
    """Review F1 / D2: a node limit of 5 on the market split MIP stops the solver with an incumbent (found at
    the root) but no proof; only a time limit keeps such a solution, so the acceptance step rejects it, naming
    ``NODE_LIMIT`` (Gurobi from its JSON status, CPLEX from the file's status code, COPT from its log).
    Xpress: no MIP."""
    from flextool.engine.input._solve_config import SolverConfig
    from flextool.engine.solve._acceptance import classify_acceptance
    from flextool.engine.solve._solver_dispatch import run_one_solve

    if solver == "xpress":
        pytest.skip("FlexTool's Xpress driver solves only LPs")
    cfg = _cfg(tmp_path, solver, f"{THREADS[solver]} 1\n{NODES[solver]} 5\n")
    sol = run_one_solve(_market_split(), SolverConfig(name=solver), solve_name="market_split",
                        work_folder=tmp_path / "w", run_options=_run_options("unused.sqlite"), solver_config_dir=cfg)
    (call,) = _calls(shim)
    assert call["params"] == {THREADS[solver]: 1, NODES[solver]: 5}, call
    assert call["is_mip"] and call["mip_gap"] is not None and call["mip_gap"] > 0.0, call
    assert (sol.shim_status, sol.optimal) == ("NODE_LIMIT", False)
    acc = classify_acceptance(sol, ranges_post=None, solve_name="market_split")
    assert not acc.accepted and "(solver status NODE_LIMIT)" in acc.message, acc


# ---------------------------------------------------------------------------
# The option-file folder: a run from another folder
# ---------------------------------------------------------------------------


def test_run_from_another_folder_uses_the_named_folder(solver, shim, db, tmp_path, monkeypatch, caplog) -> None:
    """A run started from a folder that holds a stale ``solver_config/`` but given ``--solver-config-dir``:
    the solver gets the named folder's parameters, and the log says which folder and why."""
    from flextool.engine import run_chain_from_db

    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "solver_config").mkdir(parents=True)
    (elsewhere / "solver_config" / f"{solver}.opt").write_text(f"{THREADS[solver]} 7\n")
    monkeypatch.chdir(elsewhere)
    cfg = _cfg(tmp_path, solver, f"{THREADS[solver]} 3\n")
    with caplog.at_level(logging.INFO):
        steps = run_chain_from_db(input_db_url=db, scenario_name=f"lp_{solver}", work_folder=tmp_path / "w",
                                  run_options=_run_options(db, "--solver-config-dir", str(cfg)))
    assert steps[SOLVE].optimal
    calls = _calls(shim)
    assert calls and all(c["params"] == {THREADS[solver]: 3} for c in calls), calls
    assert f"Solver config: {cfg} (--solver-config-dir)" in [r.getMessage() for r in caplog.records]
