"""One layered HiGHS option set reaches every solve path (#22a, #22b, #39).

Precedence, lowest → highest: engine baseline (determinism pins + scale) < ``solver_config/highs.opt`` <
``solve.solver_mip_gap`` (→ ``mip_rel_gap``) < ``solve.solver_arguments`` < CLI flags; autoscale Layer 3 merges on
top of the built Problem.  The tests check what HiGHS actually received: the live HiGHS instance on the warm path,
the ``.opt`` file the ``cmd_solve_mps`` subprocess reads on the cold paths, and the Problem options at
``WarmProblem`` construction for the Benders master and regions.  Databases are built from JSON fixtures plus an
override alternative (never a checked-in ``.sqlite``).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
from polar_high import Problem

from flex_toy_existing_fixed_cost import data as toy_data

from flextool.engine.input._solve_config import SolverConfig
from flextool.engine.solve import _benders
from flextool.engine.solve._solver_dispatch import _resolve_effective_highs_options

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures"
HIGHS_OPT = "primal_feasibility_tolerance=1e-8\nsolver=simplex\nthreads=1\n"
SOLVE = "y2020_2day_dispatch"
# key → expected value on every single-solve path: highs.opt, the solve's mip gap and arguments, the CLI time
# limit, and the engine baseline.
EXPECTED = {
    "mip_rel_gap": 0.05,
    "presolve": "off",
    "primal_feasibility_tolerance": 1e-8,
    "solver": "simplex",
    "time_limit": 1000.0,
    "random_seed": 0,
    "simplex_scale_strategy": 2,
}


def _lp_snapshot():
    """``tools/lp_snapshot.py`` (fixture overlay helpers), loaded by path."""
    name = "_lp_snapshot_for_tests"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPO / "tools" / "lp_snapshot.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def _norm(value):
    """Option values as text (``.opt`` file) or typed (HiGHS) → comparable."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    try:
        return float(text)
    except ValueError:
        return text


# ---------------------------------------------------------------------------
# Resolver layer order (unit)
# ---------------------------------------------------------------------------


def test_resolver_layer_order(tmp_path) -> None:
    opt = tmp_path / "highs.opt"
    opt.write_text("mip_rel_gap=0.002\nthreads=1\n")
    base = {"solver": "simplex"}
    full = _resolve_effective_highs_options(
        solver_arguments_map={"mip_rel_gap": 0.01}, highs_opt_path=opt, cli_overrides={"threads": 2},
        baseline=base, solver_config=SolverConfig(mip_gap=0.05))
    assert full["mip_rel_gap"] == 0.01 and full["threads"] == 2 and full["solver"] == "simplex"
    no_args = _resolve_effective_highs_options(
        solver_arguments_map=None, highs_opt_path=opt, cli_overrides=None, baseline=base,
        solver_config=SolverConfig(mip_gap=0.05))
    assert no_args["mip_rel_gap"] == 0.05 and no_args["threads"] == "1"
    no_knob = _resolve_effective_highs_options(
        solver_arguments_map=None, highs_opt_path=opt, cli_overrides=None, baseline=base,
        solver_config=SolverConfig())
    assert no_knob["mip_rel_gap"] == "0.002"
    knobs = _resolve_effective_highs_options(
        solver_arguments_map=None, highs_opt_path=None, cli_overrides=None,
        solver_config=SolverConfig(time_limit=30.0, threads=3))
    assert knobs == {"time_limit": 30.0, "threads": 3}


def test_build_flextool_does_not_set_solver_options() -> None:
    from flextool.engine import build_flextool

    pb = Problem()
    build_flextool(pb, toy_data())
    assert pb._solver_options is None


# ---------------------------------------------------------------------------
# Warm, cold (warm=False) and save-memory solves receive the same set
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def options_db(tmp_path_factory) -> tuple[str, Path]:
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    data = json.loads((FIXTURES / "tests.json").read_text())
    data = _lp_snapshot()._overlay(
        data, name="custom", base=["init", "west", "coal"],
        values=[["solve", SOLVE, "solver_arguments",
                 {"index_type": "str", "index_name": "x", "data": [["presolve", "off"]]}],
                ["solve", SOLVE, "solver_mip_gap", 0.05]])
    work = tmp_path_factory.mktemp("solver_options_db")
    (work / "f.json").write_text(json.dumps(data))
    url = json_to_db(work / "f.json", work / "f.sqlite")
    migrate_database(url)
    cfg = work / "solver_config"
    cfg.mkdir()
    (cfg / "highs.opt").write_text(HIGHS_OPT)
    return url, cfg


def _run(url: str, cfg: Path, work: Path, *, warm: bool) -> dict:
    from flextool.engine import run_chain_from_db

    work.mkdir(parents=True, exist_ok=True)
    return run_chain_from_db(input_db_url=url, scenario_name="custom", work_folder=work,
                             solver_config_dir=cfg, warm=warm, keep_solutions=True)


def _opt_file_options(work: Path) -> dict:
    files = sorted((work / "solve_data" / "subprocess").glob("*.opt"))
    assert files, f"no subprocess .opt file under {work}"
    out: dict = {}
    for line in files[-1].read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def _highs_options(highs, keys) -> dict:
    out = {}
    for k in keys:
        got = highs.getOptionValue(k)
        out[k] = got[1] if isinstance(got, tuple) else got
    return out


def _assert_expected(received: dict, path: str) -> None:
    for k, v in EXPECTED.items():
        assert k in received, f"{path}: HiGHS did not receive {k}"
        assert _norm(received[k]) == _norm(v), f"{path}: {k} = {received[k]!r}, expected {v!r}"


def test_warm_and_cold_receive_the_same_options(options_db, tmp_path, monkeypatch) -> None:
    url, cfg = options_db
    monkeypatch.setenv("FLEXTOOL_HIGHS_TIME_LIMIT", "1000")
    monkeypatch.delenv("FLEXTOOL_SAVE_MEMORY", raising=False)
    steps = _run(url, cfg, tmp_path / "warm", warm=True)
    sol = list(steps.values())[-1].solution
    assert sol.highs is not None
    warm = _highs_options(sol.highs, EXPECTED)
    _assert_expected(warm, "warm")

    _run(url, cfg, tmp_path / "cold", warm=False)
    cold = _opt_file_options(tmp_path / "cold")
    _assert_expected(cold, "cold")
    assert {k: _norm(warm[k]) for k in EXPECTED} == {k: _norm(cold[k]) for k in EXPECTED}


def test_save_memory_receives_the_same_options(options_db, tmp_path, monkeypatch) -> None:
    url, cfg = options_db
    monkeypatch.setenv("FLEXTOOL_HIGHS_TIME_LIMIT", "1000")
    monkeypatch.setenv("FLEXTOOL_SAVE_MEMORY", "1")
    _run(url, cfg, tmp_path / "save_memory", warm=True)
    _assert_expected(_opt_file_options(tmp_path / "save_memory"), "save_memory")


def test_save_memory_child_gets_the_thread_budget(options_db, tmp_path, monkeypatch) -> None:
    """The ``--save-memory`` child solve gets the run's budget as ``--threads`` (it sizes the child's polars pool
    before polars is imported) and HiGHS ``threads`` in its options file."""
    import subprocess

    url, cfg = options_db
    monkeypatch.setenv("FLEXTOOL_SAVE_MEMORY", "1")
    monkeypatch.setenv("FLEXTOOL_THREADS", "2")
    real_run = subprocess.run
    commands: list[list[str]] = []

    def recording_run(cmd, *args, **kwargs):
        commands.append([str(c) for c in cmd])
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", recording_run)
    _run(url, cfg, tmp_path / "save_memory", warm=True)
    children = [c for c in commands if "flextool.cli.cmd_solve_mps" in c]
    assert children, "no save-memory child solve ran"
    for cmd in children:
        i = cmd.index("--threads")
        assert cmd[i + 1] == "2", cmd
    assert _norm(_opt_file_options(tmp_path / "save_memory")["threads"]) == 2.0


def test_solver_precommand_is_never_run(tmp_path, monkeypatch) -> None:
    """``solve.solver_precommand`` is kept only so that older databases read: a solve with one set starts no
    process (the command would come from the input database, so it must never reach a shell)."""
    import subprocess

    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    marker = tmp_path / "precommand_ran"
    data = json.loads((FIXTURES / "tests.json").read_text())
    data = _lp_snapshot()._overlay(
        data, name="custom", base=["init", "west", "coal"],
        values=[["solve", SOLVE, "solver_precommand", f"touch {marker}"]])
    (tmp_path / "f.json").write_text(json.dumps(data))
    url = json_to_db(tmp_path / "f.json", tmp_path / "f.sqlite")
    migrate_database(url)

    started: list[str] = []
    real_popen = subprocess.Popen

    class RecordingPopen(real_popen):
        def __init__(self, args, *a, **kw):
            started.append(str(args))
            super().__init__(args, *a, **kw)

    def recording_system(command):
        started.append(str(command))
        return 0

    monkeypatch.setattr(subprocess, "Popen", RecordingPopen)
    monkeypatch.setattr(os, "system", recording_system)
    monkeypatch.delenv("FLEXTOOL_SAVE_MEMORY", raising=False)
    steps = _run(url, None, tmp_path / "work", warm=True)
    assert steps, "the solve did not run"
    assert started == [], f"a solve with solver_precommand set started processes: {started}"
    assert not marker.exists()


# ---------------------------------------------------------------------------
# Benders master and regions (#39)
# ---------------------------------------------------------------------------


class _Recorder:
    """Stand-in for ``_benders.WarmProblem`` that records the Problem options at construction."""

    def __init__(self, real, stop_after: int | None = None):
        self.real, self.stop_after, self.seen = real, stop_after, []

    def __call__(self, problem):
        self.seen.append(dict(problem._solver_options or {}))
        if self.stop_after is not None and len(self.seen) >= self.stop_after:
            raise _Stop
        return self.real(problem)


class _Stop(Exception):
    pass


@pytest.fixture(scope="module")
def benders_db(tmp_path_factory) -> tuple[str, Path]:
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    data = json.loads((FIXTURES / "lh2_three_region_trade_invest.json").read_text())
    data = _lp_snapshot()._overlay(
        data, name="custom", base="lh2_three_region_trade_invest",
        values=[["solve", "lh2_trade_invest", "solver_arguments",
                 {"index_type": "str", "index_name": "x",
                  "data": [["primal_feasibility_tolerance", 1e-9]]}]])
    work = tmp_path_factory.mktemp("benders_options_db")
    (work / "f.json").write_text(json.dumps(data))
    url = json_to_db(work / "f.json", work / "f.sqlite")
    migrate_database(url)
    cfg = work / "solver_config"
    cfg.mkdir()
    (cfg / "highs.opt").write_text("dual_feasibility_tolerance=1e-8\n")
    return url, cfg


def test_benders_problems_receive_user_options(benders_db, tmp_path, monkeypatch) -> None:
    from flextool.engine import run_chain_from_db

    url, cfg = benders_db
    rec = _Recorder(_benders.WarmProblem)
    monkeypatch.setattr(_benders, "WarmProblem", rec)
    steps = run_chain_from_db(input_db_url=url, scenario_name="custom", work_folder=tmp_path / "w",
                              solver_config_dir=cfg, keep_solutions=True)
    assert steps
    assert len(rec.seen) >= 4, rec.seen       # three regions + the master
    for opts in rec.seen:
        assert _norm(opts.get("dual_feasibility_tolerance")) == 1e-8, opts
        assert _norm(opts.get("primal_feasibility_tolerance")) == 1e-9, opts
    regions = rec.seen[:3]
    assert all(o.get("run_crossover") == "on" for o in regions), regions


@pytest.fixture(scope="module")
def ti_data(scenario_run):
    from tests._support.load import load_scenario

    return load_scenario(scenario_run("lh2_three_region_trade_invest", db_fixture="lh2_trade_invest"))


def test_benders_regions_stay_single_threaded_with_parallel_workers(ti_data, monkeypatch) -> None:
    """With the region pass on several workers the user threads / parallel options reach the master only."""
    monkeypatch.delenv("FLEXTOOL_BENDERS_WORKERS", raising=False)
    rec = _Recorder(_benders.WarmProblem, stop_after=1)
    monkeypatch.setattr(_benders, "WarmProblem", rec)
    user = {"threads": 2, "parallel": "on", "primal_feasibility_tolerance": 1e-9}
    with pytest.raises(_Stop):
        _benders.solve_benders(ti_data, ["region_A", "region_B", "region_C"], workers=3, solver_options=user)
    region = rec.seen[0]
    assert "threads" not in region and "parallel" not in region, region
    assert region["primal_feasibility_tolerance"] == 1e-9 and region["run_crossover"] == "on"

    rec_seq = _Recorder(rec.real, stop_after=1)
    monkeypatch.setattr(_benders, "WarmProblem", rec_seq)
    with pytest.raises(_Stop):
        _benders.solve_benders(ti_data, ["region_A", "region_B", "region_C"], workers=1, solver_options=user)
    assert rec_seq.seen[0]["threads"] == 2 and rec_seq.seen[0]["parallel"] == "on"


def test_benders_master_receives_user_options(monkeypatch) -> None:
    built: dict = {}

    def fake_warm_problem(m):
        built["m"] = m
        raise _Stop

    monkeypatch.setattr(_benders._region_filter, "master_network_data", lambda data, *a, **k: data)
    monkeypatch.setattr(_benders, "WarmProblem", fake_warm_problem)
    master = object.__new__(_benders._BendersMaster)
    master.regions = ["r"]
    master._region_membership = None
    master._master_hosted_nodes = None
    master._obj_scale = 1.0
    master._solver_options = {"threads": 2, "parallel": "on", "mip_rel_gap": 0.05}
    with pytest.raises(_Stop):
        master._build_flextool_master(toy_data())
    assert built["m"]._solver_options == {"threads": 2, "parallel": "on", "mip_rel_gap": 0.05}


# ---------------------------------------------------------------------------
# Benders with a thread count and the parallel region pass (c22 review 1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("how", ["cli", "highs_opt"])
def test_benders_converges_with_threads_and_parallel_regions(tmp_path, monkeypatch, how) -> None:
    """``--highs-threads 2`` (FLEXTOOL_HIGHS_THREADS) or ``threads=2`` in highs.opt, default region workers: the
    master (threads 2) and the single-threaded parallel region pass share HiGHS' global scheduler.  Before, the
    master's warm re-solve after the region pass was rejected by HiGHS and Benders crashed."""
    from flextool.engine import run_chain_from_db
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    monkeypatch.delenv("FLEXTOOL_BENDERS_WORKERS", raising=False)
    monkeypatch.delenv("FLEXTOOL_THREADS", raising=False)  # no budget: the cores, so threads=2 is kept
    cfg = tmp_path / "solver_config"
    cfg.mkdir()
    if how == "cli":
        monkeypatch.setenv("FLEXTOOL_HIGHS_THREADS", "2")
        (cfg / "highs.opt").write_text("parallel=off\n")
    else:
        (cfg / "highs.opt").write_text("threads=2\nparallel=on\n")
    url = json_to_db(FIXTURES / "lh2_three_region_trade_invest.json", tmp_path / "f.sqlite")
    migrate_database(url)
    seen: list[dict] = []
    real = _benders.WarmProblem

    def record(problem):
        seen.append(dict(problem._solver_options or {}))
        return real(problem)

    monkeypatch.setattr(_benders, "WarmProblem", record)
    steps = run_chain_from_db(input_db_url=url, scenario_name="lh2_three_region_trade_invest",
                              work_folder=tmp_path / "w", solver_config_dir=cfg, keep_solutions=True)
    step = next(reversed(list(steps.values())))
    assert step.is_benders and step.optimal is True and step.obj is not None
    if _benders._WARM_REPINS_SCHEDULER:
        assert any(_norm(o.get("threads")) == 2.0 for o in seen), "the master did not get the thread count"


# ---------------------------------------------------------------------------
# The engine readers take the run's options; the environment is only their fallback
# ---------------------------------------------------------------------------

_READER_ENV = {
    "FLEXTOOL_HIGHS_THREADS": "4", "FLEXTOOL_HIGHS_PRESOLVE": "off", "FLEXTOOL_HIGHS_TIME_LIMIT": "7",
    "FLEXTOOL_HIGHS_MIP_GAP": "0.3", "FLEXTOOL_SOLVER_LOG_LEVEL": "silent", "FLEXTOOL_WARM_START": "1",
    "FLEXTOOL_SAVE_MEMORY": "1", "FLEXTOOL_SCALING": "off", "FLEXTOOL_USER_BOUND_SCALE": "-4",
}


def _env_options(environ: dict):
    from flextool.run_options import RunOptions

    return RunOptions.from_env(environ)


def test_no_flag_and_no_env_leave_highs_serial() -> None:
    """Unset ``--highs-threads`` and ``FLEXTOOL_HIGHS_THREADS``: no operator HiGHS key at all, so the engine's
    ``parallel=off`` pin stands and no ``threads`` reaches HiGHS — what the flag's former default 1 gave."""
    from flextool.engine.solve._orchestration import _baseline_highs_options, _operator_highs_overrides

    assert _operator_highs_overrides(_env_options({})) == {}
    effective = _resolve_effective_highs_options(
        solver_arguments_map={}, highs_opt_path=None, cli_overrides=_operator_highs_overrides(_env_options({})),
        baseline=_baseline_highs_options(), solver_config=SolverConfig())
    assert effective["parallel"] == "off" and "threads" not in effective
    one = _env_options({"FLEXTOOL_HIGHS_THREADS": "1"})
    assert _operator_highs_overrides(one) == _operator_highs_overrides(_env_options({}))


def test_thread_budget_reaches_highs_with_parallel_off() -> None:
    """``--threads B`` (FLEXTOOL_THREADS) alone gives HiGHS ``threads=B`` and leaves the engine's ``parallel=off``
    pin standing; ``--highs-threads`` keeps its parallel opt-in, at most B."""
    from flextool.engine.solve._orchestration import _baseline_highs_options, _operator_highs_overrides

    budget = _env_options({"FLEXTOOL_THREADS": "4"})
    assert _operator_highs_overrides(budget) == {"threads": 4}
    effective = _resolve_effective_highs_options(
        solver_arguments_map={}, highs_opt_path=None, cli_overrides=_operator_highs_overrides(budget),
        baseline=_baseline_highs_options(), solver_config=SolverConfig())
    assert effective["threads"] == 4 and effective["parallel"] == "off"
    assert _operator_highs_overrides(_env_options({"FLEXTOOL_THREADS": "4", "FLEXTOOL_HIGHS_THREADS": "2"})) == {
        "threads": 2, "parallel": "on"}
    assert _operator_highs_overrides(_env_options({"FLEXTOOL_THREADS": "4", "FLEXTOOL_HIGHS_THREADS": "8"})) == {
        "threads": 4, "parallel": "on"}


def test_operator_highs_layer_comes_from_run_options(monkeypatch) -> None:
    from flextool.engine.solve._orchestration import _operator_highs_overrides

    monkeypatch.delenv("FLEXTOOL_THREADS", raising=False)  # the suite's default budget is not in _READER_ENV
    for name, value in _READER_ENV.items():
        monkeypatch.setenv(name, value)
    assert _operator_highs_overrides(_env_options({})) == {}
    from_env = _operator_highs_overrides(None)
    assert from_env == _operator_highs_overrides(_env_options(_READER_ENV))
    assert from_env == {"threads": 4, "parallel": "on", "presolve": "off", "time_limit": 7.0,
                        "mip_rel_gap": 0.3, "output_flag": False}


def test_warm_start_and_basis_cache_come_from_run_options(tmp_path, monkeypatch) -> None:
    from flextool.engine.solve._orchestration import _basis_cache_active, _basis_cache_dir
    from flextool.engine.solve._subprocess_solve import resolve_basis_cache_dir

    monkeypatch.setenv("FLEXTOOL_WARM_START", "1")
    monkeypatch.setenv("FLEXTOOL_BASIS_CACHE_DIR", str(tmp_path / "env_cache"))
    assert _basis_cache_active("none", "highs", None) is True
    assert _basis_cache_active("none", "highs", _env_options({})) is False
    chosen = _env_options({"FLEXTOOL_WARM_START": "1", "FLEXTOOL_BASIS_CACHE_DIR": str(tmp_path / "chosen")})
    assert _basis_cache_active("none", "highs", chosen) is True
    assert _basis_cache_dir(tmp_path / "work", chosen) == tmp_path / "chosen"
    assert resolve_basis_cache_dir(tmp_path / "work", _env_options({})) == tmp_path / "work" / "basis_cache"
    assert resolve_basis_cache_dir(tmp_path / "work") == tmp_path / "env_cache"


def test_scaling_config_comes_from_run_options(monkeypatch) -> None:
    from polar_high.autoscale import ScalingMode

    from flextool.engine.autoscale import resolve_scaling_config

    monkeypatch.setenv("FLEXTOOL_SCALING", "off")
    monkeypatch.setenv("FLEXTOOL_USER_BOUND_SCALE", "-4")
    assert resolve_scaling_config(None).mode is ScalingMode.OFF
    assert resolve_scaling_config(None).user_bound_scale == -4
    clean = resolve_scaling_config(None, run_options=_env_options({}))
    assert clean.mode is ScalingMode.FULL and clean.user_bound_scale is None
    assert resolve_scaling_config(None, run_options=_env_options({}), db_scaling="basic").mode is ScalingMode.BASIC
    env_wins = resolve_scaling_config(None, run_options=_env_options({"FLEXTOOL_SCALING": "solver_only"}),
                                      db_scaling="basic")
    assert env_wins.mode is ScalingMode.SOLVER_ONLY


# ---------------------------------------------------------------------------
# Two runs in one process: each seeds and reads its own solver-config folder
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gurobi_db(tmp_path_factory) -> str:
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    data = json.loads((FIXTURES / "tests.json").read_text())
    data = _lp_snapshot()._overlay(data, name="custom", base=["init", "west", "coal"],
                                   values=[["solve", SOLVE, "solver", "gurobi"]])
    work = tmp_path_factory.mktemp("gurobi_db")
    (work / "f.json").write_text(json.dumps(data))
    url = json_to_db(work / "f.json", work / "f.sqlite")
    migrate_database(url)
    return url


def test_second_run_in_a_process_uses_its_own_solver_config(gurobi_db, tmp_path, monkeypatch) -> None:
    """Run 1 takes an explicit folder; run 2 (same process, no folder given) must take
    ``<cwd>/solver_config`` (its database folder has none) and its commercial-solver ``.opt`` file must come from
    that, not from run 1's.  The commercial binary is replaced by HiGHS in process; the ``.opt`` file is built by the real code."""
    from flextool.engine import run_chain_from_db
    from flextool.engine.solve import _subprocess_solve

    monkeypatch.delenv("FLEXTOOL_SOLVER_CONFIG_DIR", raising=False)
    first = tmp_path / "first_config"
    first.mkdir()
    (first / "gurobi.opt").write_text("Threads 11\n")
    cwd = tmp_path / "project"
    (cwd / "solver_config").mkdir(parents=True)
    (cwd / "solver_config" / "gurobi.opt").write_text("Threads 22\n")
    monkeypatch.chdir(cwd)
    built: list[str] = []

    def commercial(problem, solver_name, options, *, solve_name, logger, work_folder, option_layers=None,
                   run_options=None):
        opt = _subprocess_solve._write_commercial_opt_file(options, Path(work_folder) / f"{solve_name}.gurobi.opt")
        built.append(opt.read_text() if opt is not None else "")
        return problem.solve(keep_solver=True)

    monkeypatch.setattr(_subprocess_solve, "_solve_commercial_subprocess", commercial)
    run_chain_from_db(input_db_url=gurobi_db, scenario_name="custom", work_folder=tmp_path / "w1",
                      solver_config_dir=first)
    assert "FLEXTOOL_SOLVER_CONFIG_DIR" not in os.environ
    run_chain_from_db(input_db_url=gurobi_db, scenario_name="custom", work_folder=tmp_path / "w2")
    assert (tmp_path / "w2" / "solve_data" / "solver_config" / "gurobi.opt").read_text() == "Threads 22\n"
    assert len(built) == 2 and "11" in built[0] and "22" in built[1], built


def test_cli_gap_reaches_the_commercial_solver_and_its_args_line(gurobi_db, tmp_path, monkeypatch, capsys) -> None:
    """``--solver-mip-gap`` on a Gurobi solve: the passed set holds ``MIPGap 0.001`` (over the file's 0.002),
    and the echoed line names the solver and the native name, not HiGHS' ``mip_rel_gap``."""
    from flextool.cli.cmd_run_flextool import build_parser
    from flextool.engine import run_chain_from_db
    from flextool.engine.solve import _subprocess_solve
    from flextool.run_options import RunOptions

    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "gurobi.opt").write_text("Threads 0\nMIPGap 0.002\n")
    passed: list[dict] = []

    def commercial(problem, solver_name, options, *, solve_name, logger, work_folder, option_layers=None,
                   run_options=None):
        passed.append(dict(options))
        return problem.solve(keep_solver=True)

    monkeypatch.setattr(_subprocess_solve, "_solve_commercial_subprocess", commercial)
    run_options = RunOptions.from_cli(
        build_parser().parse_args([gurobi_db, "--solver-mip-gap", "0.001"]), environ={})
    run_chain_from_db(input_db_url=gurobi_db, scenario_name="custom", work_folder=tmp_path / "w",
                      solver_config_dir=cfg, run_options=run_options)
    assert passed and all(p == {"Threads": "0", "MIPGap": "0.001"} for p in passed), passed
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("Solver args")]
    assert lines and all(line == "Solver args (gurobi): Threads=0, MIPGap=0.001" for line in lines), lines


def test_option_files_are_read_once_per_run(gurobi_db, tmp_path, monkeypatch) -> None:
    """The run reads each source ``.opt`` once at its start and never reads its snapshot back: the ``Solver
    args`` line and the solve itself take the options from the texts read then.  An edit after the start (here
    in the commercial solve) leaves the passed set unchanged."""
    from flextool.engine import run_chain_from_db
    from flextool.engine.solve import _subprocess_solve

    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "gurobi.opt").write_text("Threads 3\n")
    (cfg / "highs.opt").write_text("threads=1\n")
    reads: list[Path] = []
    real_read_text = Path.read_text

    def counting_read_text(self, *args, **kwargs):
        if self.suffix == ".opt":
            reads.append(Path(self))
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read_text)
    passed: list[dict] = []

    def commercial(problem, solver_name, options, *, solve_name, logger, work_folder, option_layers=None,
                   run_options=None):
        (cfg / "gurobi.opt").write_text("Threads 9\n")
        passed.append(dict(options))
        return problem.solve(keep_solver=True)

    monkeypatch.setattr(_subprocess_solve, "_solve_commercial_subprocess", commercial)
    run_chain_from_db(input_db_url=gurobi_db, scenario_name="custom", work_folder=tmp_path / "w",
                      solver_config_dir=cfg)
    assert passed and all(p == {"Threads": "3"} for p in passed), passed
    assert reads.count(cfg / "gurobi.opt") == 1 and reads.count(cfg / "highs.opt") == 1, reads
    snapshot = tmp_path / "w" / "solve_data" / "solver_config"
    assert not [r for r in reads if r.parent == snapshot], reads


@pytest.mark.skipif(sys.platform == "win32", reason="the solver stub is a POSIX shebang script")
def test_cli_flags_reach_a_fake_gurobi_cl_through_the_whole_run(gurobi_db, tmp_path, monkeypatch, caplog) -> None:
    """Review L6, end to end through ``run_chain_from_db`` and the real subprocess path: a fake ``gurobi_cl``
    (it solves the MPS with highspy and writes a Gurobi JSON result) records its argv.

    ``--solver-config-dir cfg`` holds ``Threads 5`` / ``MIPGap 0.002``; ``--solver-mip-gap 0.001`` replaces the
    file's gap in place and ``--solver-time-limit 60`` is added after it, so every call passes
    ``Threads=5 MIPGap=0.001 TimeLimit=60`` before ``ResultFile=`` and the model.  The log names the folder
    and why, and the run's snapshot holds the file as read."""
    import logging

    from flextool.cli.cmd_run_flextool import build_parser
    from flextool.engine import run_chain_from_db
    from flextool.engine.solve import _subprocess_solve
    from flextool.run_options import RunOptions
    from tests.engine.test_subprocess_gurobi_name_recovery import _argv_of_calls, _write_fake_gurobi

    monkeypatch.delenv("FLEXTOOL_SOLVER_CONFIG_DIR", raising=False)
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "gurobi.opt").write_text("Threads 5\nMIPGap 0.002\n")
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    stub = _write_fake_gurobi(stub_dir, [], solve_mps=True)
    monkeypatch.setattr(_subprocess_solve, "_find_solver_binary", lambda name: stub)
    run_options = RunOptions.from_cli(build_parser().parse_args([
        gurobi_db, "--solver-config-dir", str(cfg), "--solver-mip-gap", "0.001", "--solver-time-limit", "60",
    ]), environ={})
    work = tmp_path / "w"
    with caplog.at_level(logging.INFO):
        run_chain_from_db(input_db_url=gurobi_db, scenario_name="custom", work_folder=work,
                          run_options=run_options)
    calls = _argv_of_calls(stub_dir)
    assert calls
    for argv in calls:
        assert argv[:3] == ["Threads=5", "MIPGap=0.001", "TimeLimit=60"], argv
        assert argv[3].startswith("ResultFile=") and len(argv) == 5, argv
    assert f"Solver config: {cfg} (--solver-config-dir)" in [r.getMessage() for r in caplog.records]
    assert (work / "solve_data" / "solver_config" / "gurobi.opt").read_text() == "Threads 5\nMIPGap 0.002\n"


@pytest.mark.parametrize("next_to_db", [False, True])
def test_the_cli_resolves_the_option_folder_from_its_own_arguments(gurobi_db, tmp_path, monkeypatch, capsys,
                                                                   next_to_db) -> None:
    """Review L2-4: ``cmd_run_flextool.main`` resolves the folder itself (``--project-folder-file``, the input
    database) and hands it to the engine.  Started from an empty folder with ``--project-folder-file
    tb/templates/project_folder.txt``: the engine gets ``tb/solver_config`` (Spine Toolbox project folder); with a
    ``solver_config/`` next to the database, that one (it comes first).  The header names the same folder."""
    import shutil
    import sys

    import flextool.engine
    from flextool._solver_config_source import REASON_DB_FOLDER, REASON_TOOLBOX, UNTRUSTED_SHORT
    from flextool.cli import cmd_run_flextool

    monkeypatch.delenv("FLEXTOOL_SOLVER_CONFIG_DIR", raising=False)
    tb = tmp_path / "tb"
    (tb / "templates").mkdir(parents=True)
    (tb / "templates" / "project_folder.txt").write_text("")
    (tb / "solver_config").mkdir()
    (tb / "solver_config" / "gurobi.opt").write_text("Threads 4\n")
    models = tmp_path / "models"
    models.mkdir()
    db = models / "f.sqlite"
    shutil.copyfile(gurobi_db.removeprefix("sqlite:///"), db)
    if next_to_db:
        (models / "solver_config").mkdir()
        (models / "solver_config" / "gurobi.opt").write_text("Threads 6\n")
        expected = (models / "solver_config", REASON_DB_FOLDER)
    else:
        expected = (tb / "solver_config", REASON_TOOLBOX)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    got: dict = {}

    def fake_run_chain_from_db(*args, **kwargs):
        got.update(kwargs)
        raise RuntimeError("stopped by the test after the folder was resolved")

    monkeypatch.setattr(flextool.engine, "run_chain_from_db", fake_run_chain_from_db)
    monkeypatch.setattr(sys, "argv", [
        "cmd_run_flextool", str(db), "--scenario-name", "custom", "--work-folder", str(tmp_path / "w"),
        "--output-location", str(tmp_path / "out"), "--project-folder-file", str(tb / "templates" / "project_folder.txt"),
    ])
    with pytest.raises(SystemExit) as excinfo:
        cmd_run_flextool.main()
    assert excinfo.value.code == 1
    source = got["solver_config_source"]
    assert (source.dir, source.reason) == expected
    # Item 240 B3: neither folder is in a workspace this computer set up, so the header says so.
    assert source.trusted is False
    header = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("Solver config:")]
    assert header == [f"Solver config: {expected[0]} ({expected[1]}; {UNTRUSTED_SHORT})"], header
