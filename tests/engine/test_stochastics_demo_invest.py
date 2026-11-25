"""The investment example of ``docs/how_to/stochastic_examples.md``: ``invest_solve_branching_method``
``operations_only`` against ``investments_after_reveal_period`` on one small model, and an investment solve followed
by a dispatch solve.

Fixture ``tests/fixtures/stochastics.json``, the ``invest_*`` scenarios (``build_stochastics_demo.py``, the model in
its docstring): periods ``y2030`` (t0001-t0004) and ``y2035`` (t0005-t0008), ``years_represented`` 5 each, no
discounting.  Demand of ``town``: 100 MW in y2030; in y2035 160 MW (forecast ``high``) or 120 MW (``low``), weight
0.5 each, the forecasts made at t0001.  ``new_plant`` is the only supply: ``invest_cost`` 500 EUR/kW in y2030,
1500 EUR/kW in y2035, ``discount_rate`` 0, ``lifetime_technical`` 10 years, no operating cost; unserved energy
3000 EUR/MWh.

Hand calculation (``docs/reference/units.md``: with ``discount_rate`` 0 a plant pays ``invest_cost`` x 1000 /
``lifetime_technical`` EUR per MW and year, for each modelled year it serves):

* a MW built in y2030 serves 2030-2039, both periods: K1 = 500 x 1000 / 10 x 10 years = 500 000 EUR;
* a MW built in y2035 serves the 5 modelled years of y2035: K2 = 1500 x 1000 / 10 x 5 = 750 000 EUR;
* a MW short in y2035 for its 4 h costs 3000 x 4 x 8760 / 4 x 5 = 131.4 M EUR, so demand is always met.

``operations_only``: one y2035 investment x2 for both forecasts, x1 + x2 >= 160 and x1 >= 100; K1 < K2, so
x1 = 160, x2 = 0; objective 160 x K1 = 80.0 M EUR.  ``investments_after_reveal_period``: x1 shared, the y2035
investment per forecast, x2_high = (160 - x1)+, x2_low = (120 - x1)+; cost K1 x1 + 0.5 K2 (160 - x1)+ + 0.5 K2
(120 - x1)+ has slope 500 000 - 750 000 < 0 below 120 and 500 000 - 375 000 > 0 above, so x1 = 120, x2_high = 40,
x2_low = 0; objective 120 x K1 + 0.5 x 40 x K2 = 60.0 + 15.0 = 75.0 M EUR.  Both solves realize y2030 only: the
results hold the y2030 investment, its yearly cost (x1 x 100 000 EUR: 16 / 12 M EUR) and that cost over the 5 years
of y2030 (80 / 60 M EUR), and the y2030 dispatch of the realization (100 MW each hour).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from tests.engine import _solve_copy_models as scm

pytestmark = pytest.mark.solver

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "stochastics.json"
SCENARIOS = ("invest_operations_only", "invest_after_reveal_period", "invest_then_dispatch")
SOLVES = {
    "invest_operations_only": ["invest_plan", "invest_plan__realized_dispatch"],
    "invest_after_reveal_period": ["invest_plan", "invest_plan__realized_dispatch"],
    "invest_then_dispatch": ["invest_first", "dispatch_2030", "dispatch_2030__realized_dispatch"],
}
K1, K2 = 500 * 1000 / 10 * 10, 1500 * 1000 / 10 * 5
#: The y2030 investment x1 of each scenario and the realization solve that dispatches y2030.
X1 = {"invest_operations_only": 160.0, "invest_after_reveal_period": 120.0, "invest_then_dispatch": 120.0}
REALIZATION = {"invest_operations_only": "invest_plan__realized_dispatch",
               "invest_after_reveal_period": "invest_plan__realized_dispatch",
               "invest_then_dispatch": "dispatch_2030__realized_dispatch"}
Y2030 = ("t0001", "t0002", "t0003", "t0004")


#: The solve that realizes the y2030 investment: the ``solve`` column of the capacity tables (docs/results.md).
INVEST_SOLVE = {"invest_operations_only": "invest_plan", "invest_after_reveal_period": "invest_plan",
                "invest_then_dispatch": "invest_first"}


def _run(fixture: Path, folder: Path, scenarios) -> dict:
    """``{scenario: (steps, coded warnings, output CSV folder)}`` of *scenarios* of the database of *fixture*."""
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
    from flextool.process_outputs.write_outputs import write_outputs
    from tests._support.coded_warnings import capture_coded

    url = scm.build_db(fixture, folder)
    out = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")
        for scenario in scenarios:
            work = folder / scenario
            work.mkdir()
            mp.chdir(work)
            with capture_coded() as got:
                _built, steps = scm.run_capturing(url, scenario, work, csv_dump=True)
            last = next(reversed(steps.values()))
            write_outputs(
                scenario_name=scenario, output_location=str(work), subdir=scenario,
                output_config_path=str(scm.REPO / "flextool" / "schemas" / "default_plots.yaml"),
                write_methods=["csv"], fallback_output_location=str(work), raw_output_dir=str(work / "w" / "output_raw"),
                solution=last.solution, solve_name=last.solve_name,
                solve_steps=[(s.solve_name, s.flex_data, s.effective_solution) for s in steps.values()],
                flex_data_provider=FlexDataProvider())
            out[scenario] = (steps, got, work / "output_csv" / scenario)
    return out


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """``{scenario: (steps, coded warnings, output CSV folder)}``."""
    return _run(FIXTURE, tmp_path_factory.mktemp("stochastics_demo_invest"), SCENARIOS)


def _invest(step) -> dict[str, float]:
    frame = step.solution.value("v_invest_p").filter(pl.col("p").cast(pl.Utf8) == "new_plant")
    return {str(d): v for d, v in frame.select("d", "value").rows()}


def _csv(folder: Path, name: str) -> pl.DataFrame:
    return pl.read_csv(folder / name, infer_schema_length=0)


def _capacity_rows(capacity: pl.DataFrame) -> list[tuple]:
    """``(unit, solve, period, existing, invested, in_construction, divested, total)`` of each row of a
    ``unit_capacity__d`` table; an empty cell is None."""
    def number(value):
        return None if value is None else float(value)

    return [(r["unit"], r["solve"], r["period"], *(number(r[c]) for c in
                                                    ("existing", "invested", "in_construction", "divested", "total")))
            for r in capacity.iter_rows(named=True)]


def test_solves(runs):
    """An investment solve with a ``realized: yes`` branch gets its realization dispatch; in
    ``invest_then_dispatch`` the investment solve has no realized branch and the dispatch solve
    (``separate_dispatch``) has its own realization solve.  Every solve is optimal."""
    for scenario in SCENARIOS:
        steps, _got, _csv_dir = runs[scenario]
        assert list(steps) == SOLVES[scenario], scenario
        assert all(s.optimal for s in steps.values()), scenario


def test_operations_only_builds_one_fleet_for_both_futures(runs):
    steps, _got, _csv_dir = runs["invest_operations_only"]
    plan = steps["invest_plan"]
    assert _invest(plan) == pytest.approx({"y2030": 160.0, "y2035": 0.0}, abs=1e-6)
    assert plan.solution.obj == pytest.approx(160 * K1, rel=1e-9)            # 80.0 M EUR


@pytest.mark.parametrize("scenario,solve", [("invest_after_reveal_period", "invest_plan"),
                                            ("invest_then_dispatch", "invest_first")])
def test_after_reveal_period_invests_now_and_adapts_later(runs, scenario, solve):
    """The y2030 investment is one value on both forecasts' copies of y2030; the y2035 investment follows each
    forecast."""
    steps, _got, _csv_dir = runs[scenario]
    plan = steps[solve]
    assert _invest(plan) == pytest.approx({"y2030_high": 120.0, "y2030_low": 120.0,
                                           "y2035_high": 40.0, "y2035_low": 0.0}, abs=1e-6)
    assert plan.solution.obj == pytest.approx(120 * K1 + 0.5 * 40 * K2, rel=1e-9)   # 75.0 M EUR


def test_adapting_later_is_worth_five_million(runs):
    """The objective of operations_only less that of investments_after_reveal_period: 80.0 - 75.0 M EUR."""
    ops = runs["invest_operations_only"][0]["invest_plan"].solution.obj
    adapt = runs["invest_after_reveal_period"][0]["invest_plan"].solution.obj
    assert ops - adapt == pytest.approx(5.0e6, rel=1e-9)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_reported_results_are_the_y2030_investment_and_dispatch(runs, scenario):
    """The y2030 investment x1 is reported (cohort table, and the capacity table's invested and total, under the
    investment solve: under investments_after_reveal_period the decision is on the forecasts' copies of y2030,
    tied to one value, and reported under the period label), y2035 is not (look-ahead); its yearly
    cost is x1 x 100 000 EUR and its cost over the 5 years of y2030 x1 x 500 000 EUR (M EUR in the tables); the
    realization dispatches 100 MW in each y2030 hour with no unserved energy, and nothing in y2035."""
    _steps, _got, csv = runs[scenario]
    x1 = X1[scenario]
    cohorts = _csv(csv, "unit_cohort_capacity__d.csv")
    assert [(r["kind"], r["period_online"], float(r["capacity"])) for r in cohorts.iter_rows(named=True)] \
        == [("invested", "y2030", x1)]
    capacity = _csv(csv, "unit_capacity__d.csv")
    assert _capacity_rows(capacity) == [("new_plant", INVEST_SOLVE[scenario], "y2030", 0.0, x1, None, None, x1)]
    assert set(cohorts["solve"]) == {INVEST_SOLVE[scenario]}
    annual = _csv(csv, "annualized_costs__d.csv")
    assert [(r["period"], float(r["investment"])) for r in annual.iter_rows(named=True)] == [("y2030", x1 * 0.1)]
    discounted = {r["category"]: float(r["0"]) for r in _csv(csv, "costs_discounted.csv").iter_rows(named=True)}
    assert discounted["investment"] == pytest.approx(x1 * 0.5)
    flows = _csv(csv, "unit__outputNode__dt.csv").slice(1)          # row 1 names the node
    assert [(r["solve"], r["period"], r["time"], float(r["new_plant"])) for r in flows.iter_rows(named=True)] \
        == [(REALIZATION[scenario], "y2030", t, 100.0) for t in Y2030]
    slack = _csv(csv, "slack__upward__node_state__dt.csv")
    assert [float(v) for v in slack["town"]] == [0.0] * len(Y2030)


def test_dispatch_solve_starts_from_the_realized_investment(runs):
    """In invest_then_dispatch the dispatch solve invests nothing and dispatches y2030 on the 120 MW the investment
    solve realized; its objective is 0 (no operating cost, no unserved energy).  The capacity table names the
    solve that realized the investment, invest_first, and reports its 120 MW as invested."""
    steps, _got, csv = runs["invest_then_dispatch"]
    for name in ("dispatch_2030", "dispatch_2030__realized_dispatch"):
        assert steps[name].solution.obj == pytest.approx(0.0, abs=1e-6), name
    capacity = _csv(csv, "unit_capacity__d.csv")
    assert [(r["solve"], float(r["existing"]), float(r["invested"]), float(r["total"]))
            for r in capacity.iter_rows(named=True)] == [("invest_first", 0.0, 120.0, 120.0)]


def test_notices(runs):
    """No W-0640 (no ``separate_dispatch`` on an investment solve), no W-0641 (each dispatch runs on realized
    investments), no dispatch-before-investment error.  invest_then_dispatch: W-0588, the realization of
    ``dispatch_2030`` keeps nothing of its stochastic solve (no ``_fix_in_uc`` unit, no investment).  W-0553 (warm
    reuse off) comes from the test's ``warm=False``."""
    for scenario in SCENARIOS:
        _steps, got, _csv_dir = runs[scenario]
        expected = {"W-0553"} | ({"W-0588"} if scenario == "invest_then_dispatch" else set())
        assert set(got.codes()) == expected, (scenario, got.codes())


# --- a retirement in the reveal period --------------------------------------------------------------------------

RETIRE_SCENARIOS = ("invest_operations_only", "invest_after_reveal_period")


@pytest.fixture(scope="module")
def retire_runs(tmp_path_factory):
    """The investment scenarios with one more unit, ``old_plant`` -> town: 50 MW existing, retired at the start of
    y2030 (``retire_period``, ``retire_min_period`` = ``retire_max_period`` = 50 MW in y2030, 0 in y2035), no costs.
    It adds no capacity to any period, so the investments and objectives are those of the module docstring."""
    import json

    from tests.engine._unit_db import b64
    from tests.engine.test_chain_costs import _period_map

    folder = tmp_path_factory.mktemp("stochastics_demo_retire")
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    data["entities"] += [["unit", "old_plant", None], ["unit__outputNode", ["old_plant", "town"], None]]
    data["entity_alternatives"].append(["unit", ["old_plant"], "invest_demo", True])
    retire = _period_map({"y2030": 50.0, "y2035": 0.0})
    for par, value in (("existing", b64(50.0, "float")), ("efficiency", b64(1.0, "float")),
                       ("virtual_unitsize", b64(1.0, "float")), ("invest_method", b64("retire_period", "str")),
                       ("retire_min_period", retire), ("retire_max_period", retire)):
        data["parameter_values"].append(["unit", "old_plant", par, value, "invest_demo"])
    fixture = folder / "stochastics_retire.json"
    fixture.write_text(json.dumps(data), encoding="utf-8")
    return _run(fixture, folder, RETIRE_SCENARIOS)


@pytest.mark.parametrize("scenario", RETIRE_SCENARIOS)
def test_reveal_period_retirement_is_reported(retire_runs, scenario):
    """The 50 MW retired in y2030 are reported as divested in y2030 under the investment solve, with nothing in
    service after (total 0); under investments_after_reveal_period the retirement, like the investment, is on the
    forecasts' copies of y2030 and is reported under the period label.  new_plant is unchanged."""
    steps, _got, csv = retire_runs[scenario]
    x1 = X1[scenario]
    assert steps["invest_plan"].solution.obj == pytest.approx(
        x1 * K1 + (0.5 * 40 * K2 if scenario == "invest_after_reveal_period" else 0.0), rel=1e-9)
    rows = {r[0]: r for r in _capacity_rows(_csv(csv, "unit_capacity__d.csv"))}
    assert rows["new_plant"] == ("new_plant", "invest_plan", "y2030", 0.0, x1, None, None, x1)
    unit, solve, period, _existing, invested, in_construction, divested, total = rows["old_plant"]
    assert (unit, solve, period, invested, in_construction, divested, total) \
        == ("old_plant", "invest_plan", "y2030", None, None, 50.0, 0.0)
