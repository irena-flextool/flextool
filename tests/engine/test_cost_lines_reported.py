"""The reported investment-type costs by event (cost lines), hand-derived, and their total against the objective.

Model: ``tests.json``, the single solve over p2020 … p2035 (5-year periods, run years 0, 5, 10, 15, horizon end 20),
social_discount_rate i = 5 %, discount_offset_investment 1 (each year's payment at its end): v = 1/1.05, the
investment factor of a period F(d) = Σ_{k in d} v^(k+1), S[a, b) = Σ_{k=a}^{b−1} v^(k+1), D(t) = v^t.  coal_plant
without existing capacity (unless stated), fixed_cost 0, invest_cost 1000 CUR/kW (C = 1e6 CUR/MW), net_salvage_value
200 CUR/kW (S = 2e5), lifetime_technical L = payback 10, discount_rate r; 100 MW built at p2020 (pinned), af(r, n) =
r / (1 − (1+r)^−n), A = C·af(r, L), a_salv = S·(1+r)^−L·af(r, L).  Per the objective (invest_cost_p,
divest_built_cost_p), the oracle in each docstring:

* H1 (reinvest_choice, r = i): *investment* at p2020 = 100·A·S[0, 10) = 100 M (the loan is worth C at r = i);
  *salvage end of life* at p2030 (its end of life) = −100·a_salv·S[0, 10) = −100·S·v^10 = −12.278265 M.
* H2 (the 100 MW retired at p2025): *retirement* at p2025 = 100·(A·S[0, 5) + D(5)·A·ā(r, 5) − A·S[0, 10)) (the
  payments to the start of p2025 and the balance then, less the capital charged for the plant's years in service),
  0 at r = i and −3.932374 M at r = 8 % (investment 115.076621 M); *salvage economic* at p2025 = −100·S·D(5) =
  −15.670523 M; the end-of-life credit is given back at p2030 (+100·a_salv·S[0, 10)), so *salvage end of life* is 0.
* H3 (reinvest_automatic, r = i): the plant of p2020 serves [0, 10), its replacement (at the same values) [10, 20):
  *investment* 100 M at p2020, *replacement* 100·A·S[10, 20) = 61.391325 M at p2030, *salvage replaced*
  −12.278265 M at p2030 and −100·a_salv·S[10, 20) = −7.537790 M at the last period p2035 (its life ends at the
  horizon end).
* A plant past the horizon (reinvest_choice, L = 15, built at p2030): in service [10, 20) inside the horizon (its
  end of life 25 is past it), *investment* 100·A·S[10, 20) = 45.670825 M at p2030, *salvage end of life* the
  horizon share of its credit, −100·a_salv·S[10, 20) = −4.393690 M, at the last period p2035 (its full salvage
  value would be −100·S·v^25 = −5.906055 M).
* H4: 100 MW existing of age 5 (lifetime 10: end of life at year 5, the start of p2025) that may retire (pinned to
  retire nothing): its salvage credit a_x = S·(1+r)^−5·af(r, 5) over its service years [0, 5), *salvage
  pre-existing* −100·a_x·S[0, 5) = −100·S·v^5 = −15.670523 M at p2025 (r = i), an objective constant.

The cost table shows the four salvage causes as one *salvage* line (their sum per period); ``summary_solve.csv``
keeps them apart (each cause's total over the periods).  The single solves' "Total cost counted in the objective" is
the solver objective (the constants of existing capacity excluded).
"""
from __future__ import annotations

import pandas as pd
import pytest

from flextool.engine.preprocess.params._cost_lines import SALVAGE, SALVAGE_CAUSES
from tests.engine._unit_db import PLANT, b64
from tests.engine.test_chain_costs import MEGA, _no_battery_invest, _period_map, _write_csv
from tests.engine.test_user_constraint_capacity_terms import SINGLE

PERIODS = ("p2020", "p2025", "p2030", "p2035")
RATE = 0.05
V = 1.0 / 1.05
C, S, MW = 1e6, 2e5, 100.0
_TABLES: dict[str, tuple[pd.DataFrame, dict[str, float]]] = {}


def af(r: float, n: float) -> float:
    return r / (1.0 - (1.0 + r) ** -n)


def span(a: int, b: int) -> float:
    return sum(V ** (k + 1) for k in range(a, b))


def abar(r: float, x: float) -> float:
    return (1.0 - (1.0 + r) ** -x) / r


def _pins(at: dict[str, float]) -> list:
    return _period_map({d: at.get(d, 0.0) for d in PERIODS})


def _values(*, r: float = RATE, method: str = "reinvest_choice", life: float = 10.0, built: str = "p2020",
            retired: dict[str, float] | None = None, existing: float = 0.0, extra: list | None = None) -> list:
    values = [["model", "flexTool", "social_discount_rate", b64(RATE, "float")],
              ["model", "flexTool", "discount_offset_investment", b64(1.0, "float")],
              ["unit", PLANT, "existing", b64(existing, "float")],
              ["unit", PLANT, "fixed_cost", b64(0.0, "float")],
              ["unit", PLANT, "invest_cost", b64(1000.0, "float")],
              ["unit", PLANT, "net_salvage_value", b64(200.0, "float")],
              ["unit", PLANT, "lifetime_technical", b64(life, "float")],
              ["unit", PLANT, "discount_rate", b64(r, "float")],
              ["unit", PLANT, "lifetime_method", b64(method, "str")],
              ["unit", PLANT, "invest_method", b64("invest_retire_period", "str")],
              ["unit", PLANT, "invest_min_period", _pins({built: MW} if built else {})],
              ["unit", PLANT, "invest_max_period", _pins({built: MW} if built else {})],
              ["unit", PLANT, "retire_min_period", _pins(retired or {})],
              ["unit", PLANT, "retire_max_period", _pins(retired or {})]]
    return values + (extra or [])


def _summary(out) -> dict[str, float]:
    import csv
    rows: dict[str, float] = {}
    with open(out / "summary_solve.csv") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                try:
                    rows[row[0]] = float(row[1])
                except ValueError:
                    continue
    return rows


def _lines(tmp_path_factory, case: str, values: list) -> tuple[pd.DataFrame, dict[str, float]]:
    """The coal_plant rows of ``cost_unit_discounted__d`` (indexed by period) and the summary rows."""
    if case not in _TABLES:
        out = _write_csv(tmp_path_factory, f"lines_{case}", values, _no_battery_invest(SINGLE))
        table = pd.read_csv(out / "cost_unit_discounted__d.csv")
        _TABLES[case] = table[table["unit"] == PLANT].set_index("period"), _summary(out)
    return _TABLES[case]


def _assert_lines(table: pd.DataFrame, summary: dict[str, float], want: dict[tuple[str, str], float],
                  lines: tuple[str, ...]) -> None:
    """``want`` ((period, line by cause) → M CUR, 0 elsewhere) in the cost table, the salvage causes summed in its
    salvage line, and each line's total over the periods in the summary."""
    def shown(line: str) -> str:
        return SALVAGE if line in SALVAGE_CAUSES else line
    in_table: dict[tuple[str, str], float] = {}
    for (d, line), amount in want.items():
        in_table[(d, shown(line))] = in_table.get((d, shown(line)), 0.0) + amount
    for d in PERIODS:
        for line in dict.fromkeys(shown(li) for li in lines):
            got = table.loc[d, line] * MEGA if d in table.index else 0.0
            assert got == pytest.approx(in_table.get((d, line), 0.0) * MEGA, rel=1e-6, abs=1.0), (d, line)
    for line in lines:
        total = sum(amount for (_, li), amount in want.items() if li == line)
        assert summary[f"{line[0].upper()}{line[1:]} (M CUR)"] == pytest.approx(total, rel=1e-6, abs=1e-6), line


def _assert_objective(summary: dict[str, float]) -> None:
    (objective,) = [v for k, v in summary.items() if k.startswith("y2020")]
    assert summary["Total cost counted in the objective (M CUR)"] == pytest.approx(objective, rel=1e-9)


INVEST_LINES = ("investment", "replacement", "retirement", "salvage end of life", "salvage replaced",
                "salvage economic")


@pytest.mark.solver
def test_h1_investment_and_its_salvage_at_the_end_of_life(tmp_path_factory) -> None:
    table, summary = _lines(tmp_path_factory, "h1", _values())
    _assert_lines(table, summary, {("p2020", "investment"): 100.0, ("p2030", "salvage end of life"): -12.278265},
                  INVEST_LINES)
    _assert_objective(summary)


@pytest.mark.solver
@pytest.mark.parametrize("r, investment, retirement", [(0.05, 100.0, 0.0), (0.08, 115.076621, -3.932374)])
def test_h2_retirement_settles_the_loan_and_receives_the_salvage(tmp_path_factory, r, investment, retirement) -> None:
    a = C * af(r, 10.0)
    assert MW * (a * span(0, 5) + V ** 5 * a * abar(r, 5.0) - a * span(0, 10)) / MEGA == pytest.approx(
        retirement, abs=1e-6)
    assert MW * a * span(0, 10) / MEGA == pytest.approx(investment, abs=1e-6)
    table, summary = _lines(tmp_path_factory, f"h2_{r}", _values(r=r, retired={"p2025": MW}))
    _assert_lines(table, summary, {("p2020", "investment"): investment, ("p2025", "retirement"): retirement,
                          ("p2025", "salvage economic"): -15.670523}, INVEST_LINES)
    _assert_objective(summary)


@pytest.mark.solver
def test_h3_replacement_of_a_reinvest_automatic_plant(tmp_path_factory) -> None:
    table, summary = _lines(tmp_path_factory, "h3", _values(method="reinvest_automatic"))
    _assert_lines(table, summary, {("p2020", "investment"): 100.0, ("p2030", "replacement"): 61.391325,
                          ("p2030", "salvage replaced"): -12.278265, ("p2035", "salvage replaced"): -7.537790},
                  INVEST_LINES)
    _assert_objective(summary)


@pytest.mark.solver
def test_retiring_a_reinvest_automatic_build_takes_its_replacement_back_where_it_was_reported(
        tmp_path_factory) -> None:
    """The reinvest_automatic plant of p2020 (r = i, L = 10) retired at p2025, before its replacement comes online
    at p2030.  *retirement* at p2025 holds only the plant's own settlement, 100·(A·S[0, 5) + D(5)·A·ā(r, 5) −
    A·S[0, 10)) = 0 at r = i.  The replacement is never built: its capital, 100·A·S[10, 20) = 61.391325 M, is taken
    back under *replacement* at p2030, where the investment lines put it, so *replacement* nets to 0 there.  The
    salvage credits of both plants (−12.278265 M at p2030, −7.537790 M at p2035) are given back in *salvage replaced*
    (0), and the salvage value is received at p2025, −100·S·D(5) = −15.670523 M."""
    a = C * af(RATE, 10.0)
    assert MW * (a * span(0, 5) + V ** 5 * a * abar(RATE, 5.0) - a * span(0, 10)) == pytest.approx(0.0, abs=1e-3)
    assert MW * a * span(10, 20) / MEGA == pytest.approx(61.391325, abs=1e-6)
    table, summary = _lines(tmp_path_factory, "auto_retired",
                            _values(method="reinvest_automatic", retired={"p2025": MW}))
    _assert_lines(table, summary, {("p2020", "investment"): 100.0, ("p2025", "salvage economic"): -15.670523},
                  INVEST_LINES)
    _assert_objective(summary)


@pytest.mark.solver
def test_retiring_existing_reinvest_automatic_capacity_takes_its_replacement_back_where_it_was_reported(
        tmp_path_factory) -> None:
    """100 MW existing reinvest_automatic capacity (age 0, L = 10: replaced at p2030, at the same values) retired at
    p2025.  The replacement is an objective constant, *investment pre-existing* 100·A·S[10, 20) = 61.391325 M at
    p2030; the retirement takes it back under *replacement* in the same period, −61.391325 M, so the two net to 0
    there, and *retirement* holds only the settlement of the plant in service, which has no contract (0).  The
    salvage credits of the plant and its replacement, *salvage pre-existing* −100·S·v^10 = −12.278265 M at p2030 and
    −100·a_salv·S[10, 20) = −7.537790 M at p2035, are given back in *salvage replaced*, and the salvage value is
    received at p2025, −100·S·D(5) = −15.670523 M."""
    values = _values(method="reinvest_automatic", built="", existing=MW, retired={"p2025": MW},
                     extra=[["unit", PLANT, "age_at_model_start", b64(0.0, "float")]])
    table, summary = _lines(tmp_path_factory, "auto_existing_retired", values)
    _assert_lines(table, summary, {("p2025", "salvage economic"): -15.670523,
                                   ("p2030", "investment pre-existing"): 61.391325,
                                   ("p2030", "replacement"): -61.391325,
                                   ("p2030", "salvage pre-existing"): -12.278265,
                                   ("p2030", "salvage replaced"): 12.278265,
                                   ("p2035", "salvage pre-existing"): -7.537790,
                                   ("p2035", "salvage replaced"): 7.537790},
                  (*INVEST_LINES, "investment pre-existing", "salvage pre-existing"))


@pytest.mark.solver
def test_a_plant_past_the_horizon_reports_the_horizon_share_of_its_salvage(tmp_path_factory) -> None:
    a, a_salv = C * af(RATE, 15.0), S * 1.05 ** -15 * af(RATE, 15.0)
    assert MW * a * span(10, 20) / MEGA == pytest.approx(45.670825, abs=1e-6)
    assert -MW * a_salv * span(10, 20) / MEGA == pytest.approx(-4.393690, abs=1e-6)
    table, summary = _lines(tmp_path_factory, "past_h", _values(life=15.0, built="p2030"))
    _assert_lines(table, summary, {("p2030", "investment"): 45.670825, ("p2035", "salvage end of life"): -4.393690},
                  INVEST_LINES)
    _assert_objective(summary)


@pytest.mark.solver
def test_h4_salvage_credit_of_existing_capacity(tmp_path_factory) -> None:
    a_x = S * 1.05 ** -5 * af(RATE, 5.0)
    assert -MW * a_x * span(0, 5) / MEGA == pytest.approx(-15.670523, abs=1e-6)
    values = _values(built="", existing=MW, extra=[["unit", PLANT, "age_at_model_start", b64(5.0, "float")]])
    table, summary = _lines(tmp_path_factory, "h4", values)
    _assert_lines(table, summary, {("p2025", "salvage pre-existing"): -15.670523},
                  (*INVEST_LINES, "salvage pre-existing"))
    _assert_objective(summary)


@pytest.mark.solver
def test_fixed_costs_of_a_retired_build_in_each_period_add_up_to_the_objective(tmp_path_factory) -> None:
    """H2 (the 100 MW built at p2020 retired at p2025) with fixed_cost 10 CUR/kW a year (1e4 CUR/MW) and the default
    discount_offset_operations 0.5: S_op[a, b) = Σ_{k=a}^{b−1} 1.05^−(k+0.5), S_op[0, 5) = 4.436393, S_op[5, 10) =
    3.476030.  The plant would serve [0, 10); its fixed cost is reported in each period it is paid, from the
    capacity of its cohort: *fixed cost invested* 100·1e4·S_op[0, 5) = 4.436393 M at p2020 and 100·1e4·S_op[5, 10)
    = 3.476030 M at p2025 (the retired MW still count there, until the end of life at p2030), and the retirement
    saves the same p2025 amount, *fixed cost reduction of divestments* −3.476030 M at p2025; nothing from p2030.
    The objective then holds the fixed cost of p2020 only, and the reported total is the solver objective."""
    def s_op(a: int, b: int) -> float:
        return sum(1.05 ** -(k + 0.5) for k in range(a, b))

    assert (s_op(0, 5), s_op(5, 10)) == (pytest.approx(4.436393, abs=1e-6), pytest.approx(3.476030, abs=1e-6))
    values = [v for v in _values(retired={"p2025": MW}) if v[2] != "fixed_cost"] + [
        ["unit", PLANT, "fixed_cost", b64(10.0, "float")]]
    table, summary = _lines(tmp_path_factory, "h2_fixed", values)
    invested = {"p2020": 4.436393, "p2025": 3.476030}
    divested = {"p2025": -3.476030}
    for d in PERIODS:
        got_inv = table.loc[d, "fixed cost invested"] if d in table.index else 0.0
        got_div = table.loc[d, "fixed cost reduction of divestments"] if d in table.index else 0.0
        assert got_inv == pytest.approx(invested.get(d, 0.0), abs=1e-6), d
        assert got_div == pytest.approx(divested.get(d, 0.0), abs=1e-6), d
    assert summary["Fixed costs for invested entities (M CUR)"] == pytest.approx(4.436393 + 3.476030, abs=1e-6)
    assert summary["Fixed cost removal due to divested entities (M CUR)"] == pytest.approx(-3.476030, abs=1e-6)
    _assert_objective(summary)


@pytest.mark.solver
def test_benders_prices_the_cost_lines_of_the_monolith(tmp_path_factory) -> None:
    """``lh2_three_region_invest`` (a Benders investment solve over three regions, then a dispatch solve): wind_A
    (reinvest_choice, retirement allowed) builds 100 MW and retires 40 MW of them in y2030 (the fixture of
    ``test_retire_built_cohorts``).  The Benders solve's cost lines (per MW of each investment and retirement) are
    those of the same solve without decomposition, they add up to its coefficients, and the decisions they price
    are the same; so the costs the lines give are the monolith's.  (A Benders solve writes no result files of its
    own, so its costs are not in the result tables: see the handoff of this work.)"""
    import polars as pl

    from tests.engine._unit_db import RunCache, utf8
    from tests.engine.test_retire_built_cohorts import LH2_JSON, WIND
    values = [["unit", WIND, "existing", b64(0.0, "float")],
              ["unit", WIND, "lifetime_method", b64("reinvest_choice", "str")],
              ["unit", WIND, "invest_method", b64("invest_retire_period", "str")],
              ["unit", WIND, "net_salvage_value", b64(100_000.0, "float")],
              ["unit", WIND, "invest_max_period", _period_map({"y2030": 100.0})],
              ["unit", WIND, "retire_max_period", _period_map({"y2030": 40.0})]]
    runs = RunCache()
    steps = {}
    for kind, extra in (("benders", []), ("monolith", [["solve", "lh2_invest", "decomposition", b64("none", "str")]])):
        got, _, _ = runs.get(tmp_path_factory, alt=f"lines_lh2_{kind}", values=values + extra,
                             base_chain=["lh2_three_region", "lh2_three_region_invest"], fixture_json=LH2_JSON)
        steps[kind] = next(s for name, s in got.items() if "lh2_invest" in name)
    assert steps["benders"].is_benders and not steps["monolith"].is_benders
    for field, keys in (("edl_invest_lines_run", ("e", "d", "line", "d_report")),
                        ("cdl_divest_built_lines", ("e", "cohort", "d", "line", "d_report"))):
        frames = [getattr(steps[k].flex_data, field).filter(pl.col("e") == WIND).sort(*keys)
                  for k in ("benders", "monolith")]
        assert frames[0].height > 0, field
        assert frames[0].select(keys).equals(frames[1].select(keys)), field
        assert frames[0]["value"].to_list() == pytest.approx(frames[1]["value"].to_list(), rel=1e-12), field
    built = steps["benders"].flex_data
    lines = built.cdl_divest_built_lines.group_by("e", "cohort", "d").agg(pl.col("value").sum())
    coef = utf8(built.cd_divest_built_cost.frame).join(lines, on=["e", "cohort", "d"], how="left")
    assert coef["value"].to_list() == pytest.approx(coef["value_right"].fill_null(0.0).to_list(), rel=1e-12)
    retired = [utf8(steps[k].handoff.realized_retire).filter(pl.col("entity") == WIND)["value"].to_list()
               for k in ("benders", "monolith")]
    assert retired[0] == pytest.approx(retired[1], rel=1e-9)
    assert retired[0] == pytest.approx([40.0], rel=1e-9)
    runs.clear()


@pytest.mark.solver
def test_the_pre_existing_lines_are_the_objective_constants(tmp_path_factory) -> None:
    """H4's existing capacity built with include_existing_fixed_cost (which no run sets): the objective moves by
    exactly the reported pre-existing lines (the salvage credit −15.670523 M; no fixed cost, no replacement), and
    the decisions do not."""
    import polars as pl
    from polar_high import Problem

    from flextool.engine import build_flextool
    from tests.engine._unit_db import RunCache, utf8

    values = _values(built="", existing=MW, extra=[["unit", PLANT, "age_at_model_start", b64(5.0, "float")]])
    table, _ = _lines(tmp_path_factory, "h4", values)
    constants = sum(table[c].sum() for c in ("fixed cost pre-existing", "investment pre-existing",
                                             "salvage pre-existing") if c in table.columns) * MEGA
    assert constants == pytest.approx(-15.670523 * MEGA, rel=1e-6)
    runs = RunCache()
    steps, _, _ = runs.get(tmp_path_factory, alt="lines_h4_constants", values=values,
                           base_chain=_no_battery_invest(SINGLE))
    fd = list(steps.values())[-1].flex_data

    def solve(include: bool):
        pb = Problem()
        build_flextool(pb, fd, include_existing_fixed_cost=include)
        sol = pb.solve()
        assert sol.optimal
        return sol.obj, utf8(sol.value("v_divest_p")).filter(pl.col("p") == PLANT)["value"].to_list()

    (with_obj, with_retired), (without_obj, without_retired) = solve(True), solve(False)
    assert with_retired == pytest.approx(without_retired, abs=1e-9)
    assert with_obj - without_obj == pytest.approx(constants, rel=1e-6)
    runs.clear()
