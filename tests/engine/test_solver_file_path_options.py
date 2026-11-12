"""Solver options that read or write files at a path are not passed on; ``solver_precommand`` is never run (item 236
A2, A3).

A database or an option file from someone else must not make a solver write over, or read, files at a path of its
author's choosing, nor run a program.  Path options are dropped from every layer with one warning per run naming the
option and where it was set; FlexTool's own result file is added after the filter and still used.  The lists of path
options are pinned against the installed solver packages (skipped where a package is absent).

The commercial checks need no solver binary (the option stack and the script builders are pure functions); the end
to end run through each commercial program is ``commercial_shim/test_item230_commercial_shims.py``
(``test_path_options_and_precommand_are_not_used``).  The HiGHS end-to-end runs here cover in-process and
``--save-memory``.
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
    _FILE_PATH_OPTIONS,
    _resolve_effective_highs_options,
    commercial_option_key,
    is_file_path_option,
    reset_run_warnings,
    resolve_commercial_options,
    warn_solver_precommands,
)
from flextool.engine.solve._subprocess_solve import _SCRIPTS, _write_commercial_opt_file
from flextool.run_options import RunOptions

FLEXTOOL_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = FLEXTOOL_ROOT / "tests"
_NO_FLAGS = RunOptions.from_env({})

#: One path option per solver, as a user would spell it (each solver's parameter reference).  COPT has none.
PATH_OPTION = {
    "highs": "log_file", "gurobi": "LogFile", "cplex": "workdir", "xpress": "TUNEROUTPUTPATH",
}
#: FlexTool's own result file in each commercial solver's command (``<sol>`` is the result path).
OWN_RESULT = {
    "gurobi": "ResultFile=<sol>", "cplex": "write <sol> sol", "xpress": "writeslxsol <sol>", "copt": "write <sol>",
}
#: A plain parameter of each solver that must still pass.
THREADS = {"highs": "threads", "gurobi": "Threads", "cplex": "threads", "xpress": "THREADS", "copt": "Threads"}


@pytest.fixture(autouse=True)
def _fresh_run_warnings():
    reset_run_warnings()
    yield
    reset_run_warnings()


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


# ---------------------------------------------------------------------------
# Each layer of each solver: dropped with one warning, FlexTool's result file still in the command
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solver", ["gurobi", "cplex", "xpress"])
@pytest.mark.parametrize("layer", ["option file", "solver_arguments"])
def test_commercial_path_option_is_dropped_and_own_result_file_kept(solver, layer, tmp_path, caplog) -> None:
    target = tmp_path / "elsewhere" / "written_by_the_solver"
    name = PATH_OPTION[solver]
    opt_text = f"{THREADS[solver]} 2\n"
    options: dict = {}
    if layer == "option file":
        opt_text += f"{name} {target}\n"
    else:
        options[name] = str(target)
    with caplog.at_level(logging.WARNING):
        for solve in ("invest", "dispatch"):  # two solves of one run: one warning
            resolved = resolve_commercial_options(
                solver, opt_file="cfg/x.opt", opt_text=opt_text,
                solver_config=SolverConfig(name=solver, options=options), run_options=_NO_FLAGS, solve_name=solve,
            )
    assert [(o.name, o.value) for o in resolved] == [(THREADS[solver], "2")]
    merged = _write_commercial_opt_file({o.name: o.value for o in resolved}, tmp_path / "solve.opt")
    sol = tmp_path / "m.sol"
    argv, stdin = _SCRIPTS[solver](Path("bin"), tmp_path / "m.mps", sol, merged)
    sent = merged.read_text() + " ".join(argv) + (stdin or "")
    assert str(target) not in sent and name.casefold() not in sent.casefold(), sent
    assert OWN_RESULT[solver].replace("<sol>", str(sol)) in (argv if solver == "gurobi" else stdin.splitlines())
    warnings = _warnings(caplog)
    where = "cfg/x.opt" if layer == "option file" else "solver_arguments of solve invest"
    assert len(warnings) == 1, warnings
    assert f"Ignoring {solver} option {name}" in warnings[0] and where in warnings[0], warnings


def test_copt_has_no_path_option_and_passes_its_parameters(tmp_path) -> None:
    resolved = resolve_commercial_options(
        "copt", opt_file="cfg/copt.opt", opt_text="Threads 2\nRelGap 0.01\n",
        solver_config=SolverConfig(name="copt", options={"TimeLimit": 60}), run_options=_NO_FLAGS,
    )
    assert [(o.name, o.value) for o in resolved] == [("Threads", "2"), ("RelGap", "0.01"), ("TimeLimit", "60")]
    assert _FILE_PATH_OPTIONS["copt"] == frozenset()


@pytest.mark.parametrize("layer", ["highs.opt", "solver_arguments"])
def test_highs_path_option_is_dropped_from_each_layer(layer, tmp_path, caplog) -> None:
    """The one HiGHS option set (in-process, ``--save-memory``, Benders) has no path option, with one warning."""
    target = tmp_path / "elsewhere.log"
    opt = tmp_path / "highs.opt"
    opt.write_text("threads=2\n" + (f"log_file={target}\n" if layer == "highs.opt" else ""))
    args = {"write_model_file": str(target), "presolve": "off"} if layer == "solver_arguments" else None
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            options = _resolve_effective_highs_options(
                solver_arguments_map=args, highs_opt_path=opt, cli_overrides=None, solve_name="dispatch",
            )
    assert options == ({"threads": "2", "presolve": "off"} if args else {"threads": "2"})
    warnings = _warnings(caplog)
    dropped = "write_model_file" if args else "log_file"
    where = "solver_arguments of solve dispatch" if args else str(opt)
    assert len(warnings) == 1 and f"Ignoring highs option {dropped}" in warnings[0] and where in warnings[0], warnings


def test_cplex_abbreviations_of_a_path_parameter_are_dropped() -> None:
    """The CPLEX Interactive Optimizer takes any unique abbreviation of each word of a ``set`` command."""
    for name in ("logfile", "log", "workd", "WorkDir", "output intsolfileprefix", "output.intsol", "o i"):
        assert is_file_path_option("cplex", name), name
    for name in ("threads", "timelimit", "mip tolerances mipgap", "output clonelog", "workmem", "lpmethod"):
        assert not is_file_path_option("cplex", name), name
    # No abbreviations for the others.
    assert not is_file_path_option("gurobi", "Log") and is_file_path_option("gurobi", "logfile")


# ---------------------------------------------------------------------------
# Names matched as the solver reads them (security_a2a3_review F1 / F2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["Log_File", "LOG_FILE", "Sol_Files", "Input_File", "Nodefile_Dir", "result_file_"])
def test_gurobi_path_option_with_underscores_is_dropped(name, tmp_path, caplog) -> None:
    """The Gurobi library ignores case and ``_`` after the first character (``Log_File`` is ``LogFile``), so the
    filter does too: from the option file and from ``solver_arguments``, never in ``gurobi_cl``'s arguments."""
    target = tmp_path / "elsewhere" / "written_by_gurobi"
    with caplog.at_level(logging.WARNING):
        resolved = resolve_commercial_options(
            "gurobi", opt_file="cfg/gurobi.opt", opt_text=f"Threads 2\n{name} {target}\n",
            solver_config=SolverConfig(name="gurobi", options={name: str(target)}), run_options=_NO_FLAGS,
            solve_name="dispatch",
        )
    assert [(o.name, o.value) for o in resolved] == [("Threads", "2")]
    merged = _write_commercial_opt_file({o.name: o.value for o in resolved}, tmp_path / "solve.opt")
    argv, _ = _SCRIPTS["gurobi"](Path("bin"), tmp_path / "m.mps", tmp_path / "m.json", merged)
    assert str(target) not in " ".join(argv), argv
    warnings = _warnings(caplog)
    assert len(warnings) == 2 and all(f"Ignoring gurobi option {name}" in w for w in warnings), warnings


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
    merged = _write_commercial_opt_file({o.name: o.value for o in flagged}, tmp_path / "solve.opt")
    argv, _ = _SCRIPTS["gurobi"](Path("bin"), tmp_path / "m.mps", tmp_path / "m.json", merged)
    gaps = [a for a in argv if a.split("=", 1)[0].replace("_", "").casefold() == "mipgap"]
    assert gaps == ["MIPGap=0.05"], argv


@pytest.mark.parametrize("name", ["XPRS_TUNEROUTPUTPATH", "xprs_tuneroutputpath", "Xprs_TunerMethodFile"])
def test_xpress_path_control_with_prefix_is_dropped(name, tmp_path, caplog) -> None:
    """The Xpress library's control lookup ignores case and one ``XPRS_`` prefix, so the filter does too."""
    target = tmp_path / "elsewhere"
    with caplog.at_level(logging.WARNING):
        resolved = resolve_commercial_options(
            "xpress", opt_file="cfg/xpress.opt", opt_text=f"THREADS 2\n{name} {target}\n",
            solver_config=SolverConfig(name="xpress", options={name: str(target), "XPRS_MAXTIME": 60}),
            run_options=_NO_FLAGS,
        )
    assert [(o.name, o.value) for o in resolved] == [("THREADS", "2"), ("XPRS_MAXTIME", "60")]
    assert len(_warnings(caplog)) == 2
    from flextool.engine.solve._solver_dispatch import commercial_time_limit

    assert commercial_time_limit("xpress", resolved) == 60


# ---------------------------------------------------------------------------
# The lists against the installed solver packages
# ---------------------------------------------------------------------------


def test_highs_path_options_are_its_file_options() -> None:
    highspy = pytest.importorskip("highspy")
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    for name in _FILE_PATH_OPTIONS["highs"]:
        assert h.getOptionType(name) == (highspy.HighsStatus.kOk, highspy.HighsOptionType.kString), name


def test_gurobi_path_options_are_its_parameters() -> None:
    pytest.importorskip("gurobipy")
    from gurobipy._paramdetails import param_details

    by_key = {n.casefold(): d for n, d in param_details.items()}
    for key in _FILE_PATH_OPTIONS["gurobi"]:
        assert key in by_key, key
        assert by_key[key]["values"]["type"] == ("int" if key == "record" else "string"), key


def test_listed_names_are_comparison_keys() -> None:
    """The lists hold :func:`commercial_option_key` forms, so a lookup by key finds them."""
    for solver, names in _FILE_PATH_OPTIONS.items():
        assert all(commercial_option_key(solver, n) == n for n in names), solver


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

        for name in ("Log_File", "LOG_FILE", "Nodefile_Dir", "Sol_Files", "Result_File", "LogFile_", "M_I_P_Gap"):
            assert known(name), name
            assert commercial_option_key("gurobi", name) in {"logfile", "nodefiledir", "solfiles", "resultfile",
                                                             "mipgap"}, name
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

    for key in _FILE_PATH_OPTIONS["xpress"]:
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
    assert _FILE_PATH_OPTIONS["cplex"] == {"workdir", "output intsolfileprefix", "logfile"}


def test_xpress_path_options_are_its_controls() -> None:
    import warnings

    warnings.simplefilter("ignore")
    xp = pytest.importorskip("xpress")
    controls = xp.problem().getControl()
    for key in _FILE_PATH_OPTIONS["xpress"]:
        assert isinstance(controls[key], str), key


def test_copt_has_no_string_parameter() -> None:
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


def test_precommand_warns_once_per_solve_per_run(caplog) -> None:
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

SOLVE = "y2020_2day_dispatch"
SCENARIO = "item236_paths"


def _b64(x, kind: str) -> list:
    return [base64.b64encode(json.dumps(x).encode()).decode(), kind]


@pytest.fixture(scope="module")
def paths_db(tmp_path_factory) -> tuple[str, Path]:
    """``tests.json`` with a scenario (the ``coal`` LP) whose solve has HiGHS path options in its
    ``solver_arguments`` and a ``solver_precommand`` that would create a marker file: (url, folder)."""
    if str(TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(TESTS_DIR))
    from db_utils import json_to_db  # noqa: E402

    from flextool.update_flextool.db_migration import migrate_database

    folder = tmp_path_factory.mktemp("item236_paths")
    data = json.loads((TESTS_DIR / "fixtures" / "tests.json").read_text())
    args = {"type": "map", "index_type": "str", "data": [
        ["log_file", str(folder / "log_file_target.log")],
        ["write_model_to_file", "true"],
        ["write_model_file", str(folder / "model_target.mps")],
    ]}
    precommand = f"{sys.executable} -c \"open(r'{folder / 'precommand_ran'}', 'w').close()\""
    data["alternatives"].append(["item236", ""])
    data["parameter_values"] += [
        ["solve", SOLVE, "solver_arguments", _b64(args, "map"), "item236"],
        ["solve", SOLVE, "solver_precommand", _b64(precommand, "str"), "item236"],
    ]
    data["scenarios"].append([SCENARIO, False, ""])
    chain = ["init", "west", "coal", "item236"]
    for alt, before in zip(chain, chain[1:] + [None]):
        data["scenario_alternatives"].append([SCENARIO, alt, before])
    json_path = folder / "paths.json"
    json_path.write_text(json.dumps(data))
    url = json_to_db(json_path, folder / "paths.sqlite")
    migrate_database(url)
    return url, folder


@pytest.mark.solver
@pytest.mark.parametrize("save_memory", [False, True], ids=["in_process", "save_memory"])
def test_highs_run_writes_nothing_at_the_paths_and_runs_no_precommand(
        save_memory, paths_db, tmp_path, monkeypatch, caplog) -> None:
    from flextool.cli.cmd_run_flextool import build_parser
    from flextool.engine import run_chain_from_db

    url, folder = paths_db
    monkeypatch.chdir(tmp_path)
    flags = ["--save-memory"] if save_memory else []
    run_options = RunOptions.from_cli(build_parser().parse_args([url, *flags]), environ={})
    with caplog.at_level(logging.WARNING):
        steps = run_chain_from_db(url, SCENARIO, work_folder=tmp_path / "work", run_options=run_options)
    assert steps[SOLVE].optimal
    for target in ("log_file_target.log", "model_target.mps", "precommand_ran"):
        assert not (folder / target).exists(), target
    warnings = _warnings(caplog)
    dropped = [w for w in warnings if w.startswith("Ignoring highs option")]
    assert len(dropped) == 2 and all(f"solver_arguments of solve {SOLVE}" in w for w in dropped), warnings
    assert {w.split()[3] for w in dropped} == {"log_file", "write_model_file"}
    precommand = [w for w in warnings if w.startswith("solver_precommand")]
    assert len(precommand) == 1 and f"solve {SOLVE}" in precommand[0], warnings
    assert _solver_dispatch._warned_precommands == {SOLVE}
