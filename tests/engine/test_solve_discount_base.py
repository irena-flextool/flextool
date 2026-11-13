"""The discount base of a solve's own factors (``_npv`` "The discount base of a solve's own factors", item 242).

Every solve discounts its investments and its operations from the start of the run.  The base reaches the model
along three paths: ``p_inflation_op`` of a real solve and the investment walks (``_inflation_factors_lf`` with
``year_rows``) and the per-solve seeds of a synthetic sub-solve (``emit/_period_calc``).  These tests pin the one
formula of the three on a solve over p2 only, after p1 (5 years each), inflation 0.05, offsets inv 0 / ops 0.5.
Hand calculation: F_inv(p2) = sum_{y=5..9} 1.05**-y = 3.56187117, F_ops(p2) = sum_{y=5..9} 1.05**-(y + 0.5)
= 3.47603034.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine.preprocess._flex_data_provider import FlexDataProvider

F_INV_P2 = sum(1.05 ** -y for y in range(5, 10))
F_OPS_P2 = sum(1.05 ** -(y + 0.5) for y in range(5, 10))


def _utf8(columns: dict[str, list]) -> pl.DataFrame:
    return pl.DataFrame({k: [str(v) for v in vs] for k, vs in columns.items()},
                        schema={k: pl.Utf8 for k in columns})


def _later_solve_provider(years: list[tuple[str, float]]) -> FlexDataProvider:
    """The ``solve_data`` frames of a solve over p2 whose history holds p1 (5 years): ``years`` its
    ``p_years_represented`` rows of p2 (year label, width), counted from its own start."""
    prov = FlexDataProvider()
    prov.put("solve_data/period_with_history", _utf8({"period": ["p1", "p2"], "param": [0.0, 5.0]}))
    prov.put("solve_data/p_years_represented", _utf8({
        "period": ["p2"] * len(years), "years_from_solve": [y for y, _ in years],
        "p_years_from_solve": [y for y, _ in years], "p_years_represented": [w for _, w in years]}))
    for name in ("periodAll_set", "period_set", "period_in_use_set"):
        prov.put(f"solve_data/{name}", _utf8({"period": ["p2"]}))
    prov.put("solve_data/period__branch", _utf8({"period": ["p2"], "branch": ["p2"]}))
    for name in ("steps_in_use", "steps_complete_solve"):
        prov.put(f"solve_data/{name}", _utf8({"period": ["p2"], "step": ["t03"], "duration": [2.0]}))
    prov.put("solve_data/solve_current", _utf8({"solve": ["later"]}))
    prov.put("input/p_inflation_rate", _utf8({"model": ["m"], "value": [0.05]}))
    return prov


class _Source:
    """The model's inflation scalars: rate 0.05, the default offsets."""

    def parameter(self, cls: str, name: str) -> pl.DataFrame:
        if (cls, name) == ("model", "inflation_rate"):
            return pl.DataFrame({"value": [0.05]})
        raise KeyError(name)

    def parameter_default(self, cls: str, name: str):
        raise KeyError(name)


def _seed(prov: FlexDataProvider, name: str) -> float:
    df = prov.get(f"solve_data/{name}")
    return float(df.filter(pl.col(df.columns[0]) == "p2")[df.columns[1]][0])


def test_the_three_paths_give_the_run_start_factors() -> None:
    """A solve over p2 after p1: the real-solve factors (``_inflation_factors_lf`` with ``solve_year_rows``) and
    the sub-solve seeds of ``_period_calc`` are both the hand values, discounted from the start of the run."""
    from flextool.engine.preprocess.emit._period_calc import emit_period_calculated_params
    from flextool.engine.preprocess.params._npv import _inflation_factors_lf, solve_year_rows

    prov = _later_solve_provider([(str(y), 1.0) for y in range(5)])
    rows = solve_year_rows(Path("."), prov)
    assert rows is not None
    assert rows.select("d", "base").rows() == [("p2", float(y)) for y in range(5, 10)]
    factors = _inflation_factors_lf(_Source(), "later", ["p2"], year_rows=rows).collect()
    assert factors["inv_factor"][0] == pytest.approx(F_INV_P2, rel=1e-12)
    assert factors["ops_factor"][0] == pytest.approx(F_OPS_P2, rel=1e-12)
    emit_period_calculated_params(Path("input"), Path("solve_data"), provider=prov)
    assert _seed(prov, "p_inflation_factor_investment_yearly") == pytest.approx(F_INV_P2, rel=1e-12)
    assert _seed(prov, "p_inflation_factor_operations_yearly") == pytest.approx(F_OPS_P2, rel=1e-12)


def test_a_solve_that_starts_the_run_keeps_its_own_count() -> None:
    """Whole-year widths from the run start: the solve's own count already gives every year its base, so no
    year rows (the factors stay byte-identical); a fractional year before a period (item 259) or a later start
    give rows."""
    from flextool.engine.preprocess.params._npv import year_rows_from_history

    whole = {"p1": [(str(y), 1.0) for y in range(5)], "p2": [(str(y), 1.0) for y in range(5, 10)]}
    assert year_rows_from_history({"p1": 0.0, "p2": 5.0}, whole) is None
    fractional = {"p1": [("0", 1.0), ("1", 1.0), ("2", 0.5)], "p2": [("2.5", 1.0), ("3.5", 1.0)]}
    rows = year_rows_from_history({"p1": 0.0, "p2": 2.5}, fractional)
    assert rows is not None and rows.filter(pl.col("d") == "p2")["base"].to_list() == [2.5, 3.5]
    assert year_rows_from_history({"p2": 5.0}, {"p2": whole["p1"]}) is not None
    # The database map counts investments from 5 too (a whole map over p1, p2), the solve's own rows from 0.
    assert year_rows_from_history({"p2": 0.0}, {"p2": whole["p1"]}, also_counted=whole) is not None


def test_a_period_without_history_is_an_error() -> None:
    from flextool.engine.preprocess.params._npv import year_rows_from_history

    with pytest.raises(ValueError, match="Period 'p3' of solve 's'"):
        year_rows_from_history({"p2": 5.0}, {"p3": [("0", 1.0)]}, solve="s")


def test_notice_names_the_solve_once_and_only_with_inflation(capsys) -> None:
    """The notice fires while the marker is present and inflation discounts the costs; the rolls of a rolling
    solve are announced once, by the rolling solve's name."""
    from flextool.engine import _behaviour_changes as bc
    from flextool.engine.preprocess.params._npv import notice_discount_from_run_start

    bc.start_run({"chained_solve_discount_from_run_start": '{"since": 79}'})
    try:
        capsys.readouterr()
        notice_discount_from_run_start("later", 0.0)
        assert capsys.readouterr().out == ""
        for roll in ("roll_p2_roll_0", "roll_p2_roll_1", "later"):
            notice_discount_from_run_start(roll, 0.05)
        lines = capsys.readouterr().out.splitlines()
        assert len(lines) == 2 and "Affected solves: roll_p2." in lines[0] and "Affected solves: later." in lines[1]
    finally:
        bc.start_run({})
