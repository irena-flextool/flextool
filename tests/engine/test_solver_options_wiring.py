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
