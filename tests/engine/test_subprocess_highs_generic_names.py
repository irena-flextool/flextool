"""The HiGHS subprocess solve (``--save-memory`` / warm start off) maps the
solution back by generic name (items 224 and 225).

The MPS is written with generic names (``C0000001`` for col_id 0,
``R0000002`` for constraint row 0); every value read back from the child's
``.sol`` goes to the col_id / row its name encodes, and the real names come
from the Problem.  Pinned here:

* a variable used in no constraint and not in the objective (an "empty
  column") in the middle of the LP gets its own value, and the variables
  after it keep theirs (item 224: the column used to move to the end of the
  column list, shifting every later value);
* an entity name with a space (``Battery Farm``) solves as in-process
  (item 225: free-format MPS split ``x[Battery Farm]`` into two tokens);
* a ``.sol`` whose names or counts do not line up with the LP raises;
* the reserve / capacity-margin scenario where item 224 was seen: the same
  values per variable as the in-process solve, with an empty variable family
  put in front of every other column.

Expected values are worked out by hand in each test's docstring.
"""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import polar_high.engine as phe
from polar_high import Param, Problem, Sum

from flextool.engine.solve import _subprocess_solve as sps
from flextool.engine.solve._subprocess_solve import (
    SubprocessSolutionMismatch,
    _parse_highs_sol,
    _solve_highs_subprocess,
)


def _subprocess_solve(problem: Problem, work: Path | None = None):
    return _solve_highs_subprocess(problem, {}, solve_name="s", logger=None, work_folder=work)


def _values(sol, name: str) -> list[float]:
    return sol.value(name)["value"].to_list()


# ---------------------------------------------------------------------------
# An empty column in the middle of the LP (item 224)
# ---------------------------------------------------------------------------


def _empty_middle_problem(e_lower: float, e_upper: float) -> Problem:
    """Columns in order a, e, b over i = {0}; ``e`` is in no row and not in the objective.

    min a + 2 b   s.t.   c: a + b >= 3,   0 <= a <= 1,   b >= 0,   e_lower <= e <= e_upper.
    """
    p = Problem()
    i = pl.DataFrame({"i": [0]})
    a = p.add_var("a", "i", i, lower=0.0, upper=1.0)
    p.add_var("e", "i", i, lower=e_lower, upper=e_upper)
    b = p.add_var("b", "i", i, lower=0.0)
    p.add_cstr("c", over=i, sense=">=", lhs_terms={"a": a, "b": b},
               rhs_terms={"r": Param(("i",), pl.DataFrame({"i": [0], "value": [3.0]}))})
    p.set_objective(a.to_expr() + 2.0 * b.to_expr(), sense="min")
    return p


@pytest.mark.parametrize(("e_lower", "e_upper", "e_value"), [
    (5.0, 5.0, 5.0),    # fixed: written only in BOUNDS before 3.12.4, HiGHS moved it to the end
    (0.0, math.inf, 0.0),   # default bounds: not written at all before 3.12.4, HiGHS dropped it
])
def test_empty_middle_column_gets_its_own_value(e_lower, e_upper, e_value) -> None:
    """a is the cheaper way to cover c but stops at 1, so a = 1 and b = 3 - 1 = 2;
    objective 1 + 2*2 = 5; the dual of c is the cost of the marginal unit, from b: 2.
    e sits at its only possible / cheapest value (5 fixed, or 0 at its lower bound)."""
    sol = _subprocess_solve(_empty_middle_problem(e_lower, e_upper))
    assert sol.optimal
    assert sol.col_names == ["a[0]", "e[0]", "b[0]"]
    assert sol.row_names == ["c[0]"]
    assert _values(sol, "a") == pytest.approx([1.0])
    assert _values(sol, "e") == pytest.approx([e_value])
    assert _values(sol, "b") == pytest.approx([2.0])
    assert sol.obj == pytest.approx(5.0)
    assert sol.row_dual.tolist() == pytest.approx([2.0])

    ref = _empty_middle_problem(e_lower, e_upper).solve()
    for name in ("a", "e", "b"):
        assert _values(sol, name) == pytest.approx(_values(ref, name)), name
    assert sol.row_dual.tolist() == pytest.approx(np.asarray(ref.row_dual).tolist())


# ---------------------------------------------------------------------------
# An entity name with a space (item 225)
# ---------------------------------------------------------------------------


def _spaced_name_problem() -> Problem:
    """x[u] for u in {"Battery Farm", "Grid"}, 0 <= x <= 10.

    min 1 x[Battery Farm] + 3 x[Grid]   s.t.   need[u]: x[u] >= 6 (Battery Farm), 2 (Grid).
    """
    u = pl.DataFrame({"u": ["Battery Farm", "Grid"]})
    p = Problem()
    x = p.add_var("x", "u", u, lower=0.0, upper=10.0)
    p.add_cstr("need", over=u, sense=">=", lhs_terms={"x": x},
               rhs_terms={"r": Param(("u",), u.with_columns(value=pl.Series([6.0, 2.0])))})
    p.set_objective(Param(("u",), u.with_columns(value=pl.Series([1.0, 3.0]))) * x, sense="min")
    return p


def test_entity_name_with_a_space_solves_as_in_process() -> None:
    """Each x sits at its need: x[Battery Farm] = 6, x[Grid] = 2; objective 1*6 + 3*2 = 12;
    the duals of need are the unit costs, 1 and 3.  (Read with real names, x[Battery Farm]
    became a column "x[Battery" and the model changed.)"""
    sol = _subprocess_solve(_spaced_name_problem())
    assert sol.optimal
    assert sol.col_names == ["x[Battery Farm]", "x[Grid]"]
    assert sol.row_names == ["need[Battery Farm]", "need[Grid]"]
    assert sol.value("x").sort("u").rows() == [("Battery Farm", pytest.approx(6.0)), ("Grid", pytest.approx(2.0))]
    assert sol.obj == pytest.approx(12.0)
    assert sol.row_dual.tolist() == pytest.approx([1.0, 3.0])

    ref = _spaced_name_problem().solve()
    assert _values(sol, "x") == pytest.approx(_values(ref, "x"))
    assert sol.obj == pytest.approx(ref.obj)


# ---------------------------------------------------------------------------
# A .sol that does not line up with the LP raises
# ---------------------------------------------------------------------------


def _child_sol(tmp_path: Path) -> Path:
    """Solve the fixed-e empty-middle LP (3 columns, 1 row) in the real child; return its .sol."""
    mps, sol = tmp_path / "m.mps", tmp_path / "m.sol"
    _empty_middle_problem(5.0, 5.0).write_mps(str(mps), emit_names=False, release=True)
    cp = subprocess.run([sys.executable, "-m", "flextool.cli.cmd_solve_mps",
                         "--mps", str(mps), "--solution", str(sol)], capture_output=True)
    assert cp.returncode == 0, cp.stderr
    return sol


def test_untouched_sol_parses_by_position(tmp_path: Path) -> None:
    col_value, col_dual, row_dual = _parse_highs_sol(_child_sol(tmp_path), 3, 1, require_primal=True)
    assert col_value.tolist() == pytest.approx([1.0, 5.0, 2.0])
    assert row_dual.tolist() == pytest.approx([2.0])
    # Reduced costs: a is at its upper bound with cost 1 - 2 (the dual of c) = -1; e has 0; b is basic.
    assert col_dual.tolist() == pytest.approx([-1.0, 0.0, 0.0])


def _tamper_missing_column(lines: list[str]) -> list[str]:
    i = lines.index("C0000002 5")      # the empty column, as a reader that drops it would leave it
    return lines[:i] + lines[i + 1:]


def _tamper_missing_column_and_count(lines: list[str]) -> list[str]:
    out = _tamper_missing_column(lines)
    return ["# Columns 2" if ln == "# Columns 3" else ln for ln in out]


def _tamper_renamed_column(lines: list[str]) -> list[str]:
    return ["C0000009 5" if ln == "C0000002 5" else ln for ln in lines]


def _tamper_moved_column(lines: list[str]) -> list[str]:
    """The empty column moved to the end of the primal columns (what HiGHS did with a BOUNDS-only column)."""
    i = lines.index("C0000002 5")
    moved = lines[:i] + lines[i + 1:]
    j = moved.index("C0000003 2")
    return moved[:j + 1] + ["C0000002 5"] + moved[j + 1:]


def _tamper_real_names(lines: list[str]) -> list[str]:
    return [ln.replace("C0000001", "a[0]") for ln in lines]


def _tamper_truncated(lines: list[str]) -> list[str]:
    """Cut after the primal columns: no primal rows, no duals."""
    j = lines.index("C0000003 2")
    return lines[:j + 1]


def _tamper_no_primal(lines: list[str]) -> list[str]:
    """Only the status lines (the file a crashed or interrupted write leaves)."""
    return lines[:2]


@pytest.mark.parametrize("tamper", [
    _tamper_missing_column, _tamper_missing_column_and_count, _tamper_renamed_column,
    _tamper_moved_column, _tamper_real_names, _tamper_truncated, _tamper_no_primal,
])
def test_tampered_sol_raises(tmp_path: Path, tamper) -> None:
    sol_path = _child_sol(tmp_path)
    lines = sol_path.read_text().splitlines()
    assert "C0000002 5" in lines and "C0000003 2" in lines, lines
    sol_path.write_text("\n".join(tamper(lines)) + "\n")
    with pytest.raises(SubprocessSolutionMismatch):
        _parse_highs_sol(sol_path, 3, 1, require_primal=True)


def test_tampered_sol_stops_the_subprocess_solve(tmp_path: Path, monkeypatch) -> None:
    """End to end: a child .sol missing a column makes ``_solve_highs_subprocess`` raise
    instead of returning shifted values."""
    real_run = subprocess.run

    def run_then_drop_a_column(cmd, *args, **kwargs):
        cp = real_run(cmd, *args, **kwargs)
        sol_path = Path(cmd[cmd.index("--solution") + 1])
        lines = sol_path.read_text().splitlines()
        sol_path.write_text("\n".join(_tamper_missing_column_and_count(lines)) + "\n")
        return cp

    monkeypatch.setattr(sps.subprocess, "run", run_then_drop_a_column)
    with pytest.raises(SubprocessSolutionMismatch, match="primal col section"):
        _subprocess_solve(_empty_middle_problem(5.0, 5.0), tmp_path / "work")


# ---------------------------------------------------------------------------
# The scenario where item 224 was seen
# ---------------------------------------------------------------------------

_SCENARIO = "network_coal_wind_reserve_co2_capacity_margin"


@pytest.fixture
def empty_family_first(monkeypatch):
    """Every Problem gets a fixed variable family ``v_test_empty`` (3 columns at 7, in no row and
    not in the objective) before its first own variable, so every model column comes after an
    empty one.  One member has a space in its name."""
    from flextool.engine.autoscale import _layer2_types as layer2
    from flextool.engine.autoscale._quantity_types import QuantityType

    monkeypatch.setitem(layer2.VARIABLE_FAMILIES, "v_test_empty", layer2.VarFamily(QuantityType.POWER))
    orig = phe.Problem.add_var

    def add_var(self, name, *args, **kwargs):
        if not getattr(self, "_test_empty_added", False):
            self._test_empty_added = True
            orig(self, "v_test_empty", ("x",), pl.DataFrame({"x": ["a b", "c", "d"]}), lower=7.0, upper=7.0)
        return orig(self, name, *args, **kwargs)

    monkeypatch.setattr(phe.Problem, "add_var", add_var)


def _run_scenario(url: str, cfg: Path, work: Path, *, warm: bool, save_memory: bool, monkeypatch):
    from flextool.engine import run_chain_from_db

    if save_memory:
        monkeypatch.setenv("FLEXTOOL_SAVE_MEMORY", "1")
    else:
        monkeypatch.delenv("FLEXTOOL_SAVE_MEMORY", raising=False)
    work.mkdir(parents=True)
    steps = run_chain_from_db(input_db_url=url, scenario_name=_SCENARIO, work_folder=work,
                              solver_config_dir=cfg, warm=warm, keep_solutions=True)
    sol = list(steps.values())[-1].solution
    assert sol is not None and sol.optimal
    return sol


@pytest.mark.parametrize(("warm", "save_memory"), [(True, True), (False, False)], ids=["save_memory", "warm_off"])
def test_reserve_capacity_margin_scenario_matches_in_process(
        test_db_url, test_solver_config_dir, tmp_path, monkeypatch, empty_family_first, warm, save_memory) -> None:
    """The capacity-margin shortfall of p2020 is 122.4522 MW (the golden
    ``slack__capacity_margin__d.csv`` and item 224); the scenario has no reserve shortfall.  The
    subprocess solve must give that, 7 for every ``v_test_empty`` column, and the in-process value
    of every variable (it used to report the 122.45 as a 1224.5 MW reserve shortfall)."""
    ref = _run_scenario(test_db_url, test_solver_config_dir, tmp_path / "in_process",
                        warm=True, save_memory=False, monkeypatch=monkeypatch)
    sol = _run_scenario(test_db_url, test_solver_config_dir, tmp_path / "subprocess",
                        warm=warm, save_memory=save_memory, monkeypatch=monkeypatch)
    assert isinstance(sol.highs, sps._SolHighsShim), "the solve did not go through the subprocess path"
    assert not isinstance(ref.highs, sps._SolHighsShim)

    for s in (ref, sol):
        assert _values(s, "v_test_empty") == pytest.approx([7.0, 7.0, 7.0])
        assert s.value("vq_capacity_margin").rows() == [("capacity_margin", "p2020", pytest.approx(122.4522, abs=1e-4))]
        assert s.value("vq_reserve")["value"].abs().max() == pytest.approx(0.0, abs=1e-9)

    assert set(sol._vars) == set(ref._vars)
    for name in ref._vars:
        a, b = ref.value(name), sol.value(name)
        dims = [c for c in a.columns if c != "value"]
        joined = a.join(b, on=dims, how="full", coalesce=True, nulls_equal=True)
        assert joined.height == a.height == b.height, name
        np.testing.assert_allclose(joined["value_right"].to_numpy(), joined["value"].to_numpy(),
                                   rtol=1e-7, atol=1e-6, err_msg=name)
    assert sol.obj == pytest.approx(ref.obj, rel=1e-9)



# ---------------------------------------------------------------------------
# A solve stopped by its time limit (item 230 review L2)
# ---------------------------------------------------------------------------


def _small_knapsack(integer: bool) -> Problem:
    """min -5 x - 4 y   s.t.   6 x + 4 y <= 24,   x + 2 y <= 6,   0 <= x, y <= 10 (integer when *integer*)."""
    p = Problem()
    idx = pl.DataFrame({"item": ["x", "y"]})
    v = p.add_var("v", ("item",), idx, lower=0.0, upper=10.0, integer=integer)
    one = pl.DataFrame({"r": [0]})
    for name, coef, rhs in (("c1", (6.0, 4.0), 24.0), ("c2", (1.0, 2.0), 6.0)):
        weights = Param(("item",), pl.DataFrame({"item": ["x", "y"], "value": list(coef)}), name=f"w_{name}")
        p.add_cstr(name, over=one, sense="<=", lhs_terms={"a": Sum(v * weights, over=("item",))},
                   rhs_terms={"d": Param(("r",), pl.DataFrame({"r": [0], "value": [rhs]}))})
    cost = Param(("item",), pl.DataFrame({"item": ["x", "y"], "value": [-5.0, -4.0]}), name="cost")
    p.set_objective(Sum(v * cost, over=("item",)), sense="min")
    return p


def test_save_memory_time_limit_status_mip_kept_lp_rejected(tmp_path: Path, monkeypatch) -> None:
    """``--save-memory``: a HiGHS solve stopped by its time limit is read as ``TIME_LIMIT`` from the solution
    file (``Model status / Time limit reached`` with a ``Feasible`` primal), then judged by the D2 rule.

    MIP: the child is replaced by one that loads FlexTool's MPS in highspy, gives HiGHS the incumbent
    x = y = 1 (feasible: 6 + 4 = 10 <= 24, 1 + 2 = 3 <= 6) and a time limit of 0 s, so HiGHS stops at once and
    writes that incumbent: values [1, 1], objective -5 - 4 = -9.  It is kept with a time-limit warning.
    LP: the real child with ``time_limit = 0`` stops at its start point x = y = 0 (objective 0), feasible,
    status ``TIME_LIMIT``; an LP at its time limit is rejected.
    """
    import highspy

    from flextool.engine.solve._acceptance import classify_acceptance

    def fake_child(cmd, *args, **kwargs):
        mps, sol_path = cmd[cmd.index("--mps") + 1], cmd[cmd.index("--solution") + 1]
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.readModel(mps)
        h.setOptionValue("time_limit", 0.0)
        incumbent = highspy.HighsSolution()
        incumbent.col_value = [1.0, 1.0]
        h.setSolution(incumbent)
        h.run()
        assert h.getModelStatus() == highspy.HighsModelStatus.kTimeLimit
        h.writeSolution(sol_path, 0)
        return subprocess.CompletedProcess(cmd, 1)   # the child exits 1 for any non-optimal status

    real_run = sps.subprocess.run
    monkeypatch.setattr(sps.subprocess, "run", fake_child)
    mip = _solve_highs_subprocess(_small_knapsack(True), {}, solve_name="s", logger=None,
                                  work_folder=tmp_path / "mip")
    monkeypatch.setattr(sps.subprocess, "run", real_run)
    assert (mip.optimal, mip.shim_status) == (False, "TIME_LIMIT")
    np.testing.assert_allclose(mip.col_value, [1.0, 1.0])
    assert mip.obj == pytest.approx(-9.0)
    kept = classify_acceptance(mip, ranges_post=None, solve_name="s")
    assert kept.accepted and kept.limit_stopped and kept.near_optimal

    lp = _solve_highs_subprocess(_small_knapsack(False), {"time_limit": 0.0}, solve_name="s", logger=None,
                                 work_folder=tmp_path / "lp")
    assert (lp.optimal, lp.shim_status) == (False, "TIME_LIMIT")
    np.testing.assert_allclose(lp.col_value, [0.0, 0.0])
    rejected = classify_acceptance(lp, ranges_post=None, solve_name="s")
    assert not rejected.accepted and "linear problem" in rejected.message


def test_save_memory_time_limit_names_the_gap_the_child_wrote(tmp_path: Path, monkeypatch) -> None:
    """``--save-memory``: the HiGHS child writes ``info.mip_gap`` to the ``--mip-gap`` file and the parent
    names it in the time-limit warning.  The child is the time-limit stub of the test above; it writes the gap
    0.002286 (printed ``0.2286%``).  When the child writes no gap file the warning points to the solver log."""
    import highspy

    from flextool.engine.solve._acceptance import classify_acceptance

    def fake_child(gap_text: str | None):
        def run(cmd, *args, **kwargs):
            mps, sol_path = cmd[cmd.index("--mps") + 1], cmd[cmd.index("--solution") + 1]
            h = highspy.Highs()
            h.setOptionValue("output_flag", False)
            h.readModel(mps)
            h.setOptionValue("time_limit", 0.0)
            incumbent = highspy.HighsSolution()
            incumbent.col_value = [1.0, 1.0]
            h.setSolution(incumbent)
            h.run()
            h.writeSolution(sol_path, 0)
            if gap_text is not None:
                Path(cmd[cmd.index("--mip-gap") + 1]).write_text(gap_text)
            return subprocess.CompletedProcess(cmd, 1)
        return run

    monkeypatch.setattr(sps.subprocess, "run", fake_child("0.002286"))
    sol = _solve_highs_subprocess(_small_knapsack(True), {}, solve_name="s", logger=None,
                                  work_folder=tmp_path / "gap")
    assert sol.shim_status == "TIME_LIMIT" and sol.shim_mip_gap == 0.002286
    acc = classify_acceptance(sol, ranges_post=None, solve_name="s")
    assert acc.accepted and "relative MIP gap 0.2286%" in acc.message

    monkeypatch.setattr(sps.subprocess, "run", fake_child(None))
    sol = _solve_highs_subprocess(_small_knapsack(True), {}, solve_name="s", logger=None,
                                  work_folder=tmp_path / "nogap")
    assert sol.shim_mip_gap is None
    acc = classify_acceptance(sol, ranges_post=None, solve_name="s")
    assert acc.accepted and "(the remaining gap is in the solver log above)" in acc.message

    # A later solve in the same work folder whose child writes no gap: the earlier run's file is not named.
    assert (tmp_path / "gap" / "solve_data" / "subprocess" / "s.mipgap").exists()
    sol = _solve_highs_subprocess(_small_knapsack(True), {}, solve_name="s", logger=None,
                                  work_folder=tmp_path / "gap")
    assert sol.shim_mip_gap is None
    acc = classify_acceptance(sol, ranges_post=None, solve_name="s")
    assert acc.accepted and "(the remaining gap is in the solver log above)" in acc.message


@pytest.mark.parametrize(("integer", "gap"), [
    (True, 0.0),          # integer optimum x = 4, y = 0 (objective -20; 6*4 = 24 <= 24, 4 <= 6), proven: gap 0
    (False, math.inf),    # an LP has no MIP gap: HiGHS reports inf
])
def test_child_writes_the_mip_gap(tmp_path: Path, integer: bool, gap: float) -> None:
    """The real ``cmd_solve_mps`` child writes HiGHS's ``info.mip_gap`` to ``--mip-gap``."""
    mps, sol, gap_file = tmp_path / "m.mps", tmp_path / "m.sol", tmp_path / "m.mipgap"
    _small_knapsack(integer).write_mps(str(mps))
    cp = subprocess.run([sys.executable, "-m", "flextool.cli.cmd_solve_mps", "--mps", str(mps),
                         "--solution", str(sol), "--mip-gap", str(gap_file)], capture_output=True)
    assert cp.returncode == 0, cp.stderr
    assert float(gap_file.read_text()) == gap


def _knapsack60(*, integer: bool = True) -> Problem:
    """60 binaries, 5 rows (``numpy.default_rng(1)``, weights and values in [10, 100), each capacity a third of
    its row's weight): HiGHS needs a search tree to prove its optimum."""
    rng = np.random.default_rng(1)
    w, v = rng.integers(10, 100, (5, 60)), rng.integers(10, 100, 60)
    items = pl.DataFrame({"item": [f"i{j}" for j in range(60)]})
    p = Problem()
    x = p.add_var("x", ("item",), items, lower=0.0, upper=1.0, integer=integer)
    for i in range(5):
        weight = Param(("item",), items.with_columns(value=pl.Series(w[i].astype(float))), name=f"w{i}")
        p.add_cstr(f"c{i}", over=pl.DataFrame({"r": [0]}), sense="<=", lhs_terms={"a": Sum(x * weight, over=("item",))},
                   rhs_terms={"d": Param(("r",), pl.DataFrame({"r": [0], "value": [float(w[i].sum() / 3)]}))})
    value = Param(("item",), items.with_columns(value=pl.Series(-v.astype(float))), name="v")
    p.set_objective(Sum(x * value, over=("item",)), sense="min")
    return p


def test_save_memory_child_at_a_solution_limit_keeps_a_mip(tmp_path: Path) -> None:
    """Item 246, the real ``--save-memory`` child: ``mip_max_improving_sols = 1`` stops HiGHS at its first
    incumbent (``Solution limit reached`` + ``Feasible``), read as ``SOLUTION_LIMIT`` with the child's gap, and
    the MIP is kept with a warning naming HiGHS's options."""
    from flextool.engine.solve._acceptance import classify_acceptance

    sol = _solve_highs_subprocess(_knapsack60(), {"mip_max_improving_sols": 1, "threads": 1}, solve_name="s",
                                  logger=None, work_folder=tmp_path)
    assert (sol.optimal, sol.shim_status, sol.shim_primal_feasible, sol.shim_solver) == (
        False, "SOLUTION_LIMIT", True, "highs")
    assert sol.shim_mip_gap is not None and math.isfinite(sol.shim_mip_gap) and sol.shim_mip_gap > 0.0
    acc = classify_acceptance(sol, ranges_post=None, solve_name="s")
    assert acc.accepted and acc.limit_stopped and acc.model_status == "kSolutionLimit", acc
    assert "mip_max_improving_sols" in acc.message


def test_save_memory_child_at_a_node_limit_without_a_solution_is_rejected(tmp_path: Path) -> None:
    from flextool.engine.solve._acceptance import classify_acceptance

    sol = _solve_highs_subprocess(_knapsack60(), {"mip_max_nodes": 0, "threads": 1}, solve_name="s",
                                  logger=None, work_folder=tmp_path)
    assert (sol.shim_status, sol.shim_primal_feasible) == ("SOLUTION_LIMIT", False)
    acc = classify_acceptance(sol, ranges_post=None, solve_name="s")
    assert not acc.accepted and acc.model_status == "kSolutionLimit"
    assert "before finding a feasible solution" in acc.message


def test_save_memory_child_lp_at_its_iteration_limit_is_rejected(tmp_path: Path) -> None:
    """Primal simplex with ``simplex_iteration_limit = 0`` stops with a feasible start point (measured, HiGHS
    1.15): ``ITERATION_LIMIT``, and an LP at a limit stops the run."""
    from flextool.engine.solve._acceptance import classify_acceptance

    sol = _solve_highs_subprocess(_knapsack60(integer=False),
                                  {"simplex_iteration_limit": 0, "simplex_strategy": 4, "threads": 1},
                                  solve_name="s", logger=None, work_folder=tmp_path)
    assert sol.shim_status == "ITERATION_LIMIT"
    acc = classify_acceptance(sol, ranges_post=None, solve_name="s")
    assert not acc.accepted and acc.model_status == "kIterationLimit"
    assert "linear problem" in acc.message and "simplex_iteration_limit" in acc.message


@pytest.mark.parametrize(("line", "primal", "expected"), [
    ("Time limit reached", "Feasible", ("TIME_LIMIT", True)),
    ("Solution limit reached", "None", ("SOLUTION_LIMIT", False)),
    ("Iteration limit reached", "Feasible", ("ITERATION_LIMIT", True)),
    ("Target for objective reached", "Feasible", ("OBJECTIVE_TARGET", True)),
    ("Bound on objective reached", "Infeasible", ("OBJECTIVE_BOUND", False)),
    ("Memory limit reached", "Feasible", ("MEM_LIMIT", True)),
    ("Interrupted by user", "Feasible", ("OTHER", True)),
])
def test_highs_sol_status_lines(line: str, primal: str, expected, tmp_path: Path) -> None:
    sol = tmp_path / "s.sol"
    sol.write_text(f"Model status\n{line}\n\n# Primal solution values\n{primal}\n")
    assert sps._highs_sol_status(sol) == expected
