"""Construction lead time in the capacity results: ``invested`` is the capacity that comes online in a period,
``in_construction`` the capacity decided but not yet operating; combined solves count every realized investment
once.

Hand values from ``tests/fixtures/tests.json`` (see ``test_construction_lead_time_limits``): coal_plant existing
500 MW, reinvest_automatic, periods p2020 … p2035 five years apart, the model horizon ends at 2040;
``invest_min_period = invest_max_period`` pins the capacity coming online in each period.  The results are read
from ``unit_capacity__d.csv`` written by the production output path (per-solve realized slices unioned at output
time).
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from tests.engine._unit_db import PLANT, b64, make_db
from tests.engine.test_construction_lead_time_limits import ENTITIES, PERIODS, _values
from tests.engine.test_user_constraint_capacity_terms import CHAIN, SINGLE

pytestmark = pytest.mark.solver

CAPACITY_COLUMNS = ["existing", "invested", "in_construction", "divested", "total"]


def _outputs(tmp_path_factory, url: str, scenario: str) -> dict[str, pd.DataFrame]:
    """Run ``scenario`` and write its CSV results; ``{csv stem: frame}`` of the capacity and cost results."""
    from flextool.engine import run_chain_from_db
    from flextool.process_outputs.write_outputs import write_outputs

    work = tmp_path_factory.mktemp(f"{scenario}_w")
    steps = run_chain_from_db(url, scenario_name=scenario, work_folder=work, csv_dump=True)
    assert steps and all(s.optimal for s in steps.values())
    last = next(reversed(list(steps.values())))
    write_outputs(scenario_name=scenario, output_location=str(work), subdir=scenario, write_methods=["csv"],
                  flex_data=last.flex_data, solution=last.solution, solve_name=last.solve_name,
                  flex_data_provider=getattr(last, "flex_data_provider", None),
                  raw_output_dir=str(work / "output_raw"))
    out = work / "output_csv" / scenario
    return {stem: pd.read_csv(out / f"{stem}.csv") for stem in ("unit_capacity__d", "cost_unit_discounted__d")}


def _plant(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cap = frames["unit_capacity__d"]
    assert list(cap.columns[-5:]) == CAPACITY_COLUMNS
    return cap[cap["unit"] == PLANT].set_index("period")[CAPACITY_COLUMNS].reindex(PERIODS)


def _column(plant: pd.DataFrame, name: str) -> dict[str, float]:
    """A capacity column per period; an empty cell (no decision) reads as 0."""
    return plant[name].fillna(0.0).to_dict()


def _run(tmp_path_factory, alt: str, lead: float, online: dict[str, float], chain: list,
         extra_values: list | None = None, extra_entities: list | None = None) -> dict[str, pd.DataFrame]:
    url, scen = make_db(tmp_path_factory, alt=alt, values=_values(lead, online) + (extra_values or []),
                        base_chain=chain, entities=ENTITIES + (extra_entities or []))
    return _outputs(tmp_path_factory, url, scen)


FIVE_YEAR = {"p2025": 100.0, "p2030": 50.0}


@pytest.fixture(scope="module")
def five_year(tmp_path_factory) -> dict[str, dict[str, pd.DataFrame]]:
    return {kind: _run(tmp_path_factory, f"lt5_out_{kind}", 5.0, FIVE_YEAR, chain)
            for kind, chain in (("single", SINGLE), ("chain", CHAIN))}


@pytest.mark.parametrize("kind", ["single", "chain"])
def test_five_year_lead_capacity_results(five_year, kind) -> None:
    """L = 5: decisions p2020: 100 and p2025: 50 come online one period later.  invested {p2025: 100,
    p2030: 50}, in construction {p2020: 100, p2025: 50}, capacity in service 500 + {0, 100, 150, 150}; in a
    chain each realized decision is counted once, in the period it comes online."""
    plant = _plant(five_year[kind])
    assert _column(plant, "invested") == pytest.approx(
        {"p2020": 0.0, "p2025": 100.0, "p2030": 50.0, "p2035": 0.0}, abs=1e-6)
    assert _column(plant, "in_construction") == pytest.approx(
        {"p2020": 100.0, "p2025": 50.0, "p2030": 0.0, "p2035": 0.0}, abs=1e-6)
    assert _column(plant, "total") == pytest.approx(
        {"p2020": 500.0, "p2025": 600.0, "p2030": 650.0, "p2035": 650.0}, abs=1e-6)
    assert _column(plant, "existing") == pytest.approx(dict.fromkeys(PERIODS, 500.0), abs=1e-6)


def _coal_costs(frames: dict[str, pd.DataFrame]) -> dict[str, float]:
    """coal_plant discounted investment and fixed costs of new capacity, summed over the run (M CUR)."""
    costs = frames["cost_unit_discounted__d"]
    coal = costs[costs["unit"] == PLANT].drop_duplicates()
    return {c: float(coal[c].sum()) for c in ("investment", "fixed cost invested")}


def test_five_year_lead_costs_paid_while_operating(tmp_path_factory, five_year) -> None:
    """The results charge the annuity and the fixed cost of new capacity only while it operates, at the
    invest_cost of the period it comes online: the five-year case has the same capacity in service in the same
    periods as the same capacity built without a lead time (decided in p2025 and p2030), so the discounted
    investment and fixed costs of new capacity in the results are the same."""
    no_lead = _run(tmp_path_factory, "lt0_costs", 0.0, FIVE_YEAR, SINGLE)
    assert _column(_plant(no_lead), "invested") == _column(_plant(five_year["single"]), "invested")
    lead = _coal_costs(five_year["single"])
    assert lead["fixed cost invested"] > 0.0
    assert lead == pytest.approx(_coal_costs(no_lead), rel=1e-9)


def test_ten_year_lead_capacity_results(tmp_path_factory) -> None:
    """L = 10 (single solve): decisions p2020: 100 and p2025: 50 come online two periods later.  invested
    {p2030: 100, p2035: 50}, in construction {p2020: 100, p2025: 150, p2030: 50}, capacity in service
    500 + {0, 0, 100, 150}."""
    plant = _plant(_run(tmp_path_factory, "lt10_out", 10.0, {"p2030": 100.0, "p2035": 50.0}, SINGLE))
    assert _column(plant, "invested") == pytest.approx(
        {"p2020": 0.0, "p2025": 0.0, "p2030": 100.0, "p2035": 50.0}, abs=1e-6)
    assert _column(plant, "in_construction") == pytest.approx(
        {"p2020": 100.0, "p2025": 150.0, "p2030": 50.0, "p2035": 0.0}, abs=1e-6)
    assert _column(plant, "total") == pytest.approx(
        {"p2020": 500.0, "p2025": 500.0, "p2030": 600.0, "p2035": 650.0}, abs=1e-6)


def _array(values: list[str]) -> list:
    return b64({"value_type": "str", "data": values, "index_name": "period"}, "array")


def _period_map(values: dict[str, object]) -> list:
    return b64({"index_type": "str", "index_name": "period", "data": [[k, v] for k, v in values.items()]}, "map")


def test_nested_invest_then_dispatch_counts_once(tmp_path_factory) -> None:
    """An investment solve realizes the decisions of p2020 … p2035 and a dispatch solve runs on the capacity
    it hands over.  The combined results count each investment once, in the period it comes online
    (L = 5 as in the five-year case): invested {p2025: 100, p2030: 50}, in construction {p2020: 100,
    p2025: 50}, capacity in service 500 + {0, 100, 150, 150}."""
    timesets = dict.fromkeys(PERIODS, "5weeks")
    years = dict.fromkeys(PERIODS, 5.0)
    solves = [
        ["model", "flexTool", "solves",
         b64({"value_type": "str", "data": ["lt_invest", "lt_dispatch"], "index_name": "sequence_index"}, "array")],
        ["solve", "lt_invest", "solve_mode", b64("single_solve", "str")],
        ["solve", "lt_invest", "period_timeset", _period_map(timesets)],
        ["solve", "lt_invest", "years_represented", _period_map(years)],
        ["solve", "lt_invest", "invest_periods", _array(PERIODS)],
        ["solve", "lt_invest", "realized_invest_periods", _array(PERIODS)],
        ["solve", "lt_dispatch", "solve_mode", b64("single_solve", "str")],
        ["solve", "lt_dispatch", "period_timeset", _period_map(timesets)],
        ["solve", "lt_dispatch", "years_represented", _period_map(years)],
        ["solve", "lt_dispatch", "realized_periods", _array(PERIODS)],
    ]
    frames = _run(tmp_path_factory, "lt5_nested", 5.0, FIVE_YEAR, CHAIN, extra_values=solves,
                  extra_entities=[["solve", "lt_invest", None], ["solve", "lt_dispatch", None]])
    plant = _plant(frames)
    assert _column(plant, "invested") == pytest.approx(
        {"p2020": 0.0, "p2025": 100.0, "p2030": 50.0, "p2035": 0.0}, abs=1e-6)
    assert _column(plant, "in_construction") == pytest.approx(
        {"p2020": 100.0, "p2025": 50.0, "p2030": 0.0, "p2035": 0.0}, abs=1e-6)
    assert _column(plant, "total") == pytest.approx(
        {"p2020": 500.0, "p2025": 600.0, "p2030": 650.0, "p2035": 650.0}, abs=1e-6)


def test_no_lead_time_in_construction_column_is_empty(tmp_path_factory) -> None:
    """Without a construction lead time the in_construction column is written and empty, and invested is the
    capacity of the decision of the period (L = 0: online pins equal the decisions)."""
    plant = _plant(_run(tmp_path_factory, "lt0_out", 0.0, FIVE_YEAR, SINGLE))
    assert plant["in_construction"].isna().all()
    assert _column(plant, "invested") == pytest.approx(
        {"p2020": 0.0, "p2025": 100.0, "p2030": 50.0, "p2035": 0.0}, abs=1e-6)


def test_results_schema_lists_in_construction() -> None:
    """The results-schema capacity descriptions name in_construction and the coming-online meaning."""
    from flextool._resources import package_data_path

    schema = json.loads(package_data_path("schemas/spinedb_results_schema.json").read_text())
    texts = [d[4] for d in schema["parameter_definitions"] if d[1] == "capacity"]
    assert texts and all("'in_construction' - Capacity decided but not yet operating in the period." in t
                         and "'invested' - Capacity that comes online in the period." in t for t in texts)
