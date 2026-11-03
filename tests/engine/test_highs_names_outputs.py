"""The in-process solve keeps LP names out of HiGHS; outputs are unchanged.

The warm and in-process cold HiGHS paths build with polar-high
``highs_names="auto"``: the names reach HiGHS only when a HiGHS option
writes a file that carries them.  The output writer then reads the names
through ``_NamedHighsView`` (from ``Solution.col_names`` / ``row_names``).
These tests compare against the same solve with ``highs_names="always"``
(HiGHS holds the names, the writer reads them from HiGHS): the names the
writers see and every output parquet must be identical.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polar_high import Problem, WarmProblem

from flextool.engine.solve import _warm

TESTS_DIR = Path(__file__).resolve().parents[1]


def _force_highs_names(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """Make every in-process FlexTool HiGHS solve use ``highs_names=mode``."""
    real_solve = Problem.solve

    def warm(pb: Problem, **_kw: object) -> WarmProblem:
        return WarmProblem(pb, highs_names=mode)

    def solve(self: Problem, *args: object, **kw: object):
        kw["highs_names"] = mode
        return real_solve(self, *args, **kw)

    monkeypatch.setattr(_warm, "WarmProblem", warm)
    monkeypatch.setattr(Problem, "solve", solve)


def test_build_warm_problem_keeps_names_out_by_default(scenario_run) -> None:
    from flextool.engine.output._writer import _NamedHighsView, _names_highs
    from tests._support.load import load_scenario

    data = load_scenario(scenario_run("wind_battery_invest"))
    wp = _warm._build_warm_problem(data, solve_name="test")
    sol = wp.solve()
    assert sol.highs_names_passed is False
    assert sol.highs.allVariableNames() == []
    assert isinstance(_names_highs(sol), _NamedHighsView)


def test_view_names_equal_highs_names_with_invest_rename(scenario_run, monkeypatch) -> None:
    """``allVariableNames()`` / ``getLp().row_names_`` through the view equal
    what HiGHS returns when it holds the names, before and after the
    ``v_invest_*`` rename; no row name is empty."""
    from flextool.engine.output._writer import _names_highs, _rename_invest_columns
    from tests._support.load import load_scenario

    data = load_scenario(scenario_run("wind_battery_invest"))
    sols = {}
    for mode in ("always", "auto"):
        with monkeypatch.context() as m:
            _force_highs_names(m, mode)
            sols[mode] = _warm._build_warm_problem(data, solve_name="test").solve()
    ref, got = sols["always"], sols["auto"]
    assert ref.highs_names_passed and not got.highs_names_passed
    ref_h, view = _names_highs(ref), _names_highs(got)
    assert ref_h is ref.highs
    assert np.array_equal(ref.col_value, got.col_value)

    assert view.allVariableNames() == ref_h.allVariableNames()
    rows = list(view.getLp().row_names_)
    assert rows == list(ref_h.getLp().row_names_)
    assert all(rows)

    _rename_invest_columns(ref, ref_h)
    _rename_invest_columns(got, view)
    names = view.allVariableNames()
    assert any(n.startswith("v_invest[") for n in names)
    assert names == ref_h.allVariableNames()
    # The view also forwards the solver's own API.
    assert np.array_equal(view.getSolution().col_value, ref_h.getSolution().col_value)


def test_view_unnamed_rows_match_highs() -> None:
    """Unnamed rows (``None`` / ``""``) read as HiGHS holds them once the
    names are passed (``row_<i>``); a row rename reaches the cached list."""
    import highspy
    from polar_high.engine import Solution, _pass_warm_names

    from flextool.engine.output._writer import _names_highs

    col_names = ["x[a]", "y[b]"]
    row_names = [None, "c1[a]", ""]
    h = highspy.Highs()
    h.silent()
    for _ in col_names:
        h.addVar(0.0, 1.0)
    for _ in row_names:
        h.addRow(0.0, 1.0, 2, [0, 1], [1.0, 1.0])
    h.run()
    named = highspy.Highs()
    named.silent()
    named.passModel(h.getLp())
    _pass_warm_names(named, col_names, row_names)

    sol = Solution(
        optimal=True, obj=0.0, col_value=np.zeros(2), row_dual=np.zeros(3),
        col_names=list(col_names), row_names=list(row_names), vars={},
        highs=h, highs_names_passed=False,
    )
    view = _names_highs(sol)
    assert list(view.getLp().row_names_) == list(named.getLp().row_names_)
    assert view.allVariableNames() == named.allVariableNames()
    view.passRowName(2, "c2[b]")
    assert view.getLp().row_names_[2] == "c2[b]"
    assert sol.row_names[2] == "c2[b]"
    assert h.getLp().row_names_ == []


def _space_name_db(tmp_path_factory) -> tuple[str, str]:
    sys.path.insert(0, str(TESTS_DIR / "engine"))
    try:
        from test_no_solution_file import SCENARIO, _build_db
    finally:
        sys.path.pop(0)
    return _build_db(tmp_path_factory, "space_names", []), SCENARIO


def _outputs(work: Path) -> dict[str, pl.DataFrame]:
    files = sorted((work / "output_raw").rglob("*.parquet"))
    assert files
    return {str(f.relative_to(work)): pl.read_parquet(f) for f in files}


def _run_twins(url: str, scenario: str, tmp_path: Path, monkeypatch, solver_config_dir) -> None:
    from flextool.engine import run_chain_from_db

    out = {}
    for mode in ("always", "auto"):
        with monkeypatch.context() as m:
            _force_highs_names(m, mode)
            work = tmp_path / mode
            run_chain_from_db(url, scenario, work_folder=work, solver_config_dir=solver_config_dir)
            out[mode] = _outputs(work)
    assert out["auto"].keys() == out["always"].keys()
    for name, ref in out["always"].items():
        got = out["auto"][name]
        # Row order of some set outputs (e.g. edd_invest) varies between
        # runs of the same mode, so compare the rows, not their order
        # (``__set_marker__`` is the row position).
        got, ref = (f.drop("__set_marker__", strict=False) for f in (got, ref))
        assert got.sort(got.columns).equals(ref.sort(ref.columns)), name


@pytest.mark.solver
@pytest.mark.parametrize("hoist", ["on", "off"])
def test_outputs_equal_with_names_kept_out_space_names(
        hoist, tmp_path, tmp_path_factory, monkeypatch, test_solver_config_dir) -> None:
    if hoist == "off":
        monkeypatch.setenv("FLEXTOOL_DISABLE_OUTPUT_HOIST", "1")
    url, scenario = _space_name_db(tmp_path_factory)
    _run_twins(url, scenario, tmp_path, monkeypatch, test_solver_config_dir)


@pytest.mark.solver
@pytest.mark.parametrize("scenario", [
    "wind_battery_invest_lifetime_renew_4solve",
    pytest.param("fullYear_roll", marks=pytest.mark.slow),  # rolling window, ~110 s for the twin
])
def test_outputs_equal_with_names_kept_out_multi_solve(
        scenario, tmp_path, monkeypatch, test_db_url, test_solver_config_dir) -> None:
    _run_twins(test_db_url, scenario, tmp_path, monkeypatch, test_solver_config_dir)
