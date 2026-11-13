"""Item 246b: objective-valued solver options are written in unscaled currency.

An objective target / bound / cutoff or an absolute gap (HiGHS ``objective_bound``, ``objective_target``,
``mip_abs_gap``; Gurobi ``BestObjStop``, ``MIPGapAbs``; ...) is written in the currency of the model's cost data.
The solver compares it with the objective it works on: the cost × ``scale_the_objective`` (s, default 1e-6) and,
for HiGHS, × ``2**user_objective_scale`` (N).  The objective constant (the existing fixed cost, O) is part of the
objective and is scaled with it (item 245).  So HiGHS must get ``2**N · (s·V − O)`` for a level V and
``2**N · s·G`` for a gap G.

The tests check what the solver received: the live HiGHS handle (in-process), the ``.opt`` file of the
``--save-memory`` child, the ``Solver args`` line, and the Benders refusal.  Expected values are worked out by hand
from those formulas, never read from the code.  Databases are built from JSON fixtures (never a checked-in
``.sqlite``).  The commercial solvers are checked in ``commercial_shim/``.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from polar_high import Problem

from flex_toy_existing_fixed_cost import data as toy_data

from flextool.engine import build_flextool
from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.solve import _benders, _objective_options
from flextool.engine.solve._orchestration import _format_solver_args_line, _highs_echo_translations

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures"
SOLVE = "y2020_2day_dispatch"

needs_mapping = pytest.mark.skipif(
    not _objective_options.supported(),
    reason="the installed polar-high has no objective-option mapping (needs 3.14.0)",
)


def _lp_snapshot():
    """``tools/lp_snapshot.py`` (fixture overlay helpers), loaded by path."""
    name = "_lp_snapshot_for_tests"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPO / "tools" / "lp_snapshot.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


# ---------------------------------------------------------------------------
# Item 245 and the objective scale on the Problem (toy: dispatch 1 000 €, existing fixed cost 5 000 €)
# ---------------------------------------------------------------------------


def _toy(scale: float = 1e-6) -> Problem:
    pb = Problem()
    build_flextool(pb, toy_data(), include_existing_fixed_cost=True, scale_the_objective=scale)
    return pb


def test_existing_fixed_cost_constant_is_scaled_with_the_objective() -> None:
    """The fixture's objective is 1 000 € of fuel plus a 5 000 € constant: with s = 1e-6 the solved objective is
    6 000 × 1e-6 = 0.006 and the constant 5 000 × 1e-6 = 0.005 (before item 245 the constant stayed 5 000)."""
    pb = _toy()
    assert pb._obj_offset == pytest.approx(0.005, rel=1e-12)
    pb.set_solver_options({"output_flag": False})
    sol = pb.solve()
    assert sol.obj == pytest.approx(0.006, rel=1e-9)


@needs_mapping
def test_build_declares_the_objective_scale() -> None:
    assert _toy(1e-6).objective_scale == 1e-6
    assert _toy(1.0).objective_scale == 1.0


@needs_mapping
def test_solver_args_line_shows_the_solver_value_and_the_written_value() -> None:
    """s = 1e-6, O = 0.005 (5 000 € × s), N = 3 (factor 8), from the options applied to the Problem:
    objective_bound 6 500 € → 8 · (6 500e-6 − 0.005) = 0.012; mip_abs_gap 5e5 € → 8 · 0.5 = 4;
    the relative gap is not an objective value and stays."""
    pb = _toy()
    pb.set_solver_options({"user_objective_scale": 3, "output_flag": False})
    echoed = {"objective_bound": 6500.0, "mip_rel_gap": 0.01, "mip_abs_gap": 5e5}
    translated = _highs_echo_translations(pb, echoed, "s")
    assert translated == {
        "objective_bound": "0.012 (6500 in unscaled currency)",
        "mip_abs_gap": "4 (500000 in unscaled currency)",
    }
    assert _format_solver_args_line(echoed, translated=translated) == (
        "Solver args: objective_bound=0.012 (6500 in unscaled currency), mip_rel_gap=0.01, "
        "mip_abs_gap=4 (500000 in unscaled currency)"
    )


def test_an_older_polar_high_refuses_objective_options_and_runs_without_them(monkeypatch) -> None:
    """Without polar-high's mapping a currency value would reach the solver unchanged: refused, naming the
    polar-high version.  Every other option passes as it is."""
    monkeypatch.setattr(_objective_options, "_api", lambda: None)
    pb = Problem()
    plain = {"mip_rel_gap": 0.01, "time_limit": 60.0}
    assert _objective_options.translate_for_solver(pb, "highs", "mps", plain) == (plain, [])
    for solver, name in (("highs", "mip_abs_gap"), ("gurobi", "Best_Obj_Stop"), ("cplex", "mip.tolerances.absmipgap"),
                         ("xpress", "xprs_MIPABSCUTOFF"), ("copt", "absgap")):
        with pytest.raises(FlexToolConfigError, match=rf"{name}.*polar-high 3\.14\.0"):
            _objective_options.translate_for_solver(pb, solver, "mps", {name: 1.0, "time_limit": 60.0},
                                                    solve_name="s")


# ---------------------------------------------------------------------------
# A whole run: what HiGHS received, in-process and with --save-memory
# ---------------------------------------------------------------------------


def _db(tmp_path_factory, name: str, arguments: list[list]) -> str:
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    data = json.loads((FIXTURES / "tests.json").read_text())
    data = _lp_snapshot()._overlay(
        data, name="custom", base=["init", "west", "coal"],
        values=[["solve", SOLVE, "solver_arguments", {"index_type": "str", "index_name": "x", "data": arguments}]])
    work = tmp_path_factory.mktemp(name)
    (work / "f.json").write_text(json.dumps(data))
    url = json_to_db(work / "f.json", work / "f.sqlite")
    migrate_database(url)
    return url


#: An objective bound far above the optimum (inactive) and an absolute gap, in €.
ARGS = [["objective_bound", 1e15], ["mip_abs_gap", 5e5]]


@pytest.fixture(scope="module")
def options_db(tmp_path_factory) -> str:
    return _db(tmp_path_factory, "objective_options_db", ARGS)


#: ``highs.opt`` of each case: none, or the HiGHS objective factor 2**3 set by hand (Layer 3 respects it).
HIGHS_OPT = {1.0: "", 8.0: "user_objective_scale=3\n"}


def _run(url: str, work: Path, highs_opt: str = ""):
    from flextool.engine import run_chain_from_db

    cfg = work.parent / f"{work.name}_solver_config"
    cfg.mkdir(parents=True)
    (cfg / "highs.opt").write_text(highs_opt)
    return run_chain_from_db(input_db_url=url, scenario_name="custom", work_folder=work, keep_solutions=True,
                             solver_config_dir=cfg)


def _handle_options(highs, keys) -> dict:
    out = {}
    for k in keys:
        got = highs.getOptionValue(k)
        out[k] = got[1] if isinstance(got, tuple) else got
    return out


def _opt_file(work: Path) -> dict[str, str]:
    files = sorted((work / "solve_data" / "subprocess").glob("*.opt"))
    assert files, f"no subprocess .opt file under {work}"
    return dict(line.split("=", 1) for line in files[-1].read_text().splitlines() if "=" in line)


def _args_line(out: str) -> str:
    lines = [ln for ln in out.splitlines() if ln.startswith("Solver args")]
    assert lines, out[-2000:]
    return lines[-1]


@needs_mapping
@pytest.mark.parametrize("factor", [1.0, 8.0])
def test_in_process_highs_gets_the_scaled_values(options_db, factor, tmp_path, monkeypatch, capsys) -> None:
    """Autoscale off, s = 1e-6, O = 0: objective_bound 1e15 € → 1e9 · factor, mip_abs_gap 5e5 € → 0.5 · factor,
    factor = 2**N with N = 3 from highs.opt."""
    monkeypatch.setenv("FLEXTOOL_SCALING", "off")
    monkeypatch.delenv("FLEXTOOL_SAVE_MEMORY", raising=False)
    steps = _run(options_db, tmp_path / "w", HIGHS_OPT[factor])
    assert steps[SOLVE].optimal
    got = _handle_options(steps[SOLVE].solution.highs, ("objective_bound", "mip_abs_gap"))
    assert got["objective_bound"] == pytest.approx(1e9 * factor, rel=1e-12)
    assert got["mip_abs_gap"] == pytest.approx(0.5 * factor, rel=1e-12)
    line = _args_line(capsys.readouterr().out)
    assert f"objective_bound={1e9 * factor:.12g} (1e+15 in unscaled currency)" in line, line
    assert f"mip_abs_gap={0.5 * factor:.12g} (500000 in unscaled currency)" in line, line


@needs_mapping
def test_in_process_highs_follows_the_autoscale_objective_factor(options_db, tmp_path, monkeypatch) -> None:
    """Autoscale on: whatever N Layer 3 picks, HiGHS gets 1e15 × 1e-6 × 2**N."""
    monkeypatch.setenv("FLEXTOOL_SCALING", "full")
    monkeypatch.delenv("FLEXTOOL_SAVE_MEMORY", raising=False)
    steps = _run(options_db, tmp_path / "w")
    got = _handle_options(steps[SOLVE].solution.highs, ("objective_bound", "mip_abs_gap", "user_objective_scale"))
    factor = 2.0 ** int(got["user_objective_scale"])
    assert got["objective_bound"] == pytest.approx(1e9 * factor, rel=1e-12)
    assert got["mip_abs_gap"] == pytest.approx(0.5 * factor, rel=1e-12)


@needs_mapping
@pytest.mark.parametrize(("factor", "bound", "gap"), [(1.0, "1000000000.0", "0.5"), (8.0, "8000000000.0", "4.0")])
def test_save_memory_child_gets_the_scaled_values(options_db, factor, bound, gap, tmp_path, monkeypatch) -> None:
    """The child's option file holds the values the solver compares (as in-process); the child applies it as is."""
    monkeypatch.setenv("FLEXTOOL_SCALING", "off")
    monkeypatch.setenv("FLEXTOOL_SAVE_MEMORY", "1")
    steps = _run(options_db, tmp_path / "w", HIGHS_OPT[factor])
    assert steps[SOLVE].optimal
    opts = _opt_file(tmp_path / "w")
    assert (opts["objective_bound"], opts["mip_abs_gap"]) == (bound, gap)


# ---------------------------------------------------------------------------
# Benders: a whole-model target or bound is refused; absolute gaps are scaled for each Problem
# ---------------------------------------------------------------------------


@needs_mapping
def test_benders_refuses_an_objective_bound_naming_option_and_layer(tmp_path) -> None:
    from flextool.engine import run_chain_from_db
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    data = json.loads((FIXTURES / "lh2_three_region_trade_invest.json").read_text())
    data = _lp_snapshot()._overlay(
        data, name="custom", base="lh2_three_region_trade_invest",
        values=[["solve", "lh2_trade_invest", "solver_arguments",
                 {"index_type": "str", "index_name": "x",
                  "data": [["objective_bound", 4.5e9], ["mip_abs_gap", 1e5]]}]])
    (tmp_path / "f.json").write_text(json.dumps(data))
    url = json_to_db(tmp_path / "f.json", tmp_path / "f.sqlite")
    migrate_database(url)
    with pytest.raises(FlexToolConfigError) as err:
        run_chain_from_db(input_db_url=url, scenario_name="custom", work_folder=tmp_path / "w")
    msg = str(err.value)
    assert "Solve 'lh2_trade_invest' uses decomposition=benders: objective_bound=4500000000.0 " in msg, msg
    assert "(solver_arguments of solve lh2_trade_invest)" in msg and "decomposition=none" in msg, msg
    assert "mip_abs_gap=" not in msg, msg


@needs_mapping
def test_solve_benders_refuses_a_target_before_any_build(monkeypatch) -> None:
    def no_build(*a, **k):
        raise AssertionError("built a Problem before refusing the option")

    monkeypatch.setattr(_benders, "WarmProblem", no_build)
    with pytest.raises(FlexToolConfigError, match=r"objective_target=-1000000000.0 is a whole-model objective target"):
        _benders.solve_benders(toy_data(), ["a", "b"], solver_options={"objective_target": -1e9})


class _Stop(Exception):
    pass


@pytest.fixture(scope="module")
def ti_data(scenario_run):
    from tests._support.load import load_scenario

    return load_scenario(scenario_run("lh2_three_region_trade_invest", db_fixture="lh2_trade_invest"))


@needs_mapping
@pytest.mark.parametrize("master", ["flextool", "hand"])
def test_benders_master_and_regions_scale_an_absolute_gap(ti_data, master, monkeypatch) -> None:
    """Every Benders Problem (three regions, then the master) is in the scaled space s = 1e-3, so
    mip_abs_gap 1e5 € reaches each as 1e5 × 1e-3 = 100 times its own HiGHS factor."""
    seen: list[tuple[float, float, float]] = []

    def record(problem):
        opts = problem._solver_options or {}
        n = int(opts.get("user_objective_scale", 0))
        q = problem.objective_transform("highs", "direct", opts).diff(1e5)
        seen.append((problem.objective_scale, q, 100.0 * 2.0 ** n))
        if len(seen) == 4:
            raise _Stop
        return real(problem)

    real = _benders.WarmProblem
    monkeypatch.setattr(_benders, "WarmProblem", record)
    with pytest.raises(_Stop):
        _benders.solve_benders(ti_data, ["region_A", "region_B", "region_C"], master=master, workers=1,
                               scale_the_objective=1e-3, solver_options={"mip_abs_gap": 1e5})
    assert len(seen) == 4
    for scale, q, expected in seen:
        assert scale == 1e-3
        assert q == pytest.approx(expected, rel=1e-12)
