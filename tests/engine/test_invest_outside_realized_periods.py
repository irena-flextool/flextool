"""A solve whose realized-invest periods are not all realized(-dispatch) periods reports every investment.

``realized_invest_periods = [p2035, p2040]`` with ``realized_periods = [p2035]``: the ``v_invest`` / ``v_divest``
row axis was the realized-dispatch axis only, so the p2040 investment had no row -- no invested capacity in
``unit_capacity__d`` and the p2040 investment cost missing from the cost tables (the LP was right).

Witness: the deterministic ``init`` + ``base`` alternatives of ``stoch_two_period_midhorizon_nreal0`` (elec demand
120 MW every step; free 100 MW wind at 0.6 in p2035 and 0 in p2040; investable ``base``, invest_max_total 150,
discount_rate 0.05, lifetime 1, years_represented 1 per period), with ``invest_cost`` changed to {p2035: 3,
p2040: 1} CUR/kW and ``realized_periods`` to [p2035].
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

SCENARIO = "deterministic_partial_realized"
SOLVE = "stoch_2p_mid"
# Hand values: p2035 residual 120 - 60 (wind) = 60 MW, p2040 residual 120 MW.  Invest 60 at p2035 and 60 more at
# p2040 (the p2035 capacity carries over).  Annuity at r = 0.05, n = 1: 0.05 / (1 - 1.05**-1) = 1.05.  The p2040
# annualized unit investment: 60 MW * 3000 CUR/MW * 1.05 + 60 MW * 1000 CUR/MW * 1.05 = 189 000 + 63 000 CUR
# = 0.252 M CUR.
EXPECTED_INVEST = {"p2035": 60.0, "p2040": 60.0}
EXPECTED_P2040_ANNUALIZED_INVEST_MCUR = 0.252


def _add_partial_scenario(db_url: str) -> None:
    from spinedb_api import Array, Map, to_database

    from flextool.update_flextool.db_handle import open_database_mapping

    def put(db, cls: str, entity: str, param: str, value) -> None:
        raw, value_type = to_database(value)
        db.add_parameter_value_item(entity_class_name=cls, entity_byname=(entity,), parameter_definition_name=param,
                                    alternative_name="partial", value=raw, type=value_type)

    with open_database_mapping(db_url) as db:
        db.add_alternative_item(name="partial")
        db.add_scenario_item(name=SCENARIO)
        for rank, alt in enumerate(("init", "base", "partial"), start=1):
            db.add_scenario_alternative_item(scenario_name=SCENARIO, alternative_name=alt, rank=rank)
        put(db, "unit", "base", "invest_cost", Map(["p2035", "p2040"], [3.0, 1.0]))
        put(db, "solve", SOLVE, "realized_periods", Array(["p2035"]))
        db.commit_session("test: realized_periods a strict subset of realized_invest_periods")


@pytest.mark.solver
def test_invest_period_outside_realized_periods_has_its_rows(tmp_path, monkeypatch):
    from flextool.engine import run_chain_from_db
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
    from flextool.process_outputs.write_outputs import write_outputs
    from flextool.update_flextool.db_migration import migrate_database
    from db_utils import json_to_db

    repo = Path(__file__).resolve().parents[2]
    db_url = json_to_db(repo / "tests" / "fixtures" / "stoch_two_period_midhorizon_nreal0.json",
                        tmp_path / "db.sqlite")
    migrate_database(db_url)
    _add_partial_scenario(db_url)

    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")
    monkeypatch.chdir(tmp_path)
    wf = tmp_path / "w"
    steps = run_chain_from_db(db_url, SCENARIO, work_folder=wf, csv_dump=True, keep_solutions=True)

    v_invest = pl.read_parquet(wf / "output_raw" / f"v_invest__{SOLVE}.parquet")
    assert v_invest["period"].to_list() == list(EXPECTED_INVEST), v_invest
    assert v_invest["base"].to_list() == pytest.approx(list(EXPECTED_INVEST.values()))

    last = next(reversed(steps.values()))
    last.flex_data_provider.snapshot_processed_inputs(wf)
    write_outputs(
        scenario_name=SCENARIO, output_location=str(wf), subdir="out",
        output_config_path=str(repo / "flextool" / "schemas" / "default_plots.yaml"),
        write_methods=["csv"], fallback_output_location=str(wf), raw_output_dir=str(wf / "output_raw"),
        solution=last.solution, solve_name=last.solve_name,
        solve_steps=[(s.solve_name, s.flex_data, s.effective_solution) for s in steps.values()],
        flex_data_provider=FlexDataProvider(),
    )
    csv_dir = wf / "output_csv" / "out"
    capacity = pl.read_csv(csv_dir / "unit_capacity__d.csv").filter(pl.col("unit") == "base").sort("period")
    assert capacity["period"].to_list() == ["p2035", "p2040"], capacity
    assert capacity["invested"].to_list() == pytest.approx([60.0, 60.0])
    costs = pl.read_csv(csv_dir / "annualized_costs__d.csv").filter(pl.col("period") == "p2040")
    assert costs["unit investment & retirement"].to_list() == pytest.approx([EXPECTED_P2040_ANNUALIZED_INVEST_MCUR])
