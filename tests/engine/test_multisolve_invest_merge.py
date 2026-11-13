"""A later solve's zero invest row does not overwrite the investment of the solve that realizes it (item 264).

Two solves in a chain:

* ``s1`` (``stoch_2p_mid``): periods p2035 / p2040, ``realized_periods = [p2035]``, ``realized_invest_periods =
  [p2035, p2040]`` -- it realizes the p2040 investment without dispatching p2040.
* ``s2``: periods p2040 / p2045, ``realized_periods = [p2040, p2045]``, ``invest_periods = realized_invest_periods
  = [p2045]`` -- it dispatches p2040 and realizes investment elsewhere (p2045).

``s2`` writes a ``v_invest`` row for every period on its dispatch axis, so a 0 row for p2040; the merge of the
per-solve slices must keep ``s1``'s p2040 value, not the later solve's 0.

Data: the deterministic ``init`` + ``base`` alternatives of ``stoch_two_period_midhorizon_nreal0`` (elec demand
120 MW every step; free 100 MW wind at 0.6 on ts1 = t0001-t0003 and 0 on ts2 = t0004-t0006; investable ``base``,
invest_max_total 150, discount_rate 0.05, other_operational_cost 1 CUR/MWh), with ``invest_cost`` {p2035: 3,
p2040: 1, p2045: 1} CUR/kW and ``lifetime`` 30 years (all capacity alive over the run).  p2045 uses ts1.

Hand values: p2035 residual 120 - 60 = 60 MW -> invest 60; p2040 residual 120 MW -> 60 more; p2045 residual 60 MW
< 120 MW existing -> invest 0.  Annuity at r = 0.05, n = 30: 0.05 / (1 - 1.05**-30) = 0.0650514.  The p2040
annualized unit investment (both investments operate in p2040): (60 MW * 3000 CUR/MW + 60 MW * 1000 CUR/MW) *
0.0650514 = 240 000 * 0.0650514 = 15 612.3 CUR = 0.0156123 M CUR.

A second chain (item 266) checks the investment duals the same way: ``s1`` does not invest in p2040, so the p2040
marginal (``unit_invest_marginal__d``) is the non-zero reduced cost of ``s1``, not ``s2``'s 0 row.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

SCENARIO = "two_solves_invest_ahead"
S1 = "stoch_2p_mid"
S2 = "s2_dispatch_p2040"
EXPECTED_INVESTED = {"p2035": 60.0, "p2040": 60.0, "p2045": 0.0}
_ANNUITY_30Y = 0.05 / (1.0 - 1.05 ** -30)
EXPECTED_P2040_ANNUALIZED_INVEST_MCUR = (60.0 * 3000.0 + 60.0 * 1000.0) * _ANNUITY_30Y / 1e6
SCENARIO_NO_P2040_INVEST = "two_solves_no_p2040_invest"


def _add_two_solve_scenario(db_url: str, *, no_p2040_invest: bool = False) -> None:
    from spinedb_api import Array, Map, to_database

    from flextool.update_flextool.db_handle import open_database_mapping

    scenario = SCENARIO_NO_P2040_INVEST if no_p2040_invest else SCENARIO
    alternative = "two_solves_no_p2040_invest" if no_p2040_invest else "two_solves"

    def put(db, cls: str, entity: str, param: str, value) -> None:
        raw, value_type = to_database(value)
        db.add_parameter_value_item(entity_class_name=cls, entity_byname=(entity,), parameter_definition_name=param,
                                    alternative_name=alternative, value=raw, type=value_type)

    with open_database_mapping(db_url) as db:
        db.add_alternative_item(name=alternative)
        db.add_scenario_item(name=scenario)
        for rank, alt in enumerate(("init", "base", alternative), start=1):
            db.add_scenario_alternative_item(scenario_name=scenario, alternative_name=alt, rank=rank)
        db.add_entity_item(entity_class_name="solve", name=S2)
        put(db, "model", "flextool", "solves", Array([S1, S2]))
        put(db, "unit", "base", "invest_cost", Map(["p2035", "p2040", "p2045"], [3.0, 1.0, 1.0]))
        put(db, "unit", "base", "lifetime", 30.0)
        put(db, "solve", S1, "realized_periods", Array(["p2035"]))
        # The no-p2040-invest chain: every period on ts1 (residual 60 MW) and 70 MW of existing base capacity.
        p2040_timeset = "ts1" if no_p2040_invest else "ts2"
        if no_p2040_invest:
            put(db, "unit", "base", "existing", 70.0)
            put(db, "solve", S1, "period_timeset", Map(["p2035", "p2040"], ["ts1", "ts1"]))
        put(db, "solve", S2, "period_timeset", Map(["p2040", "p2045"], [p2040_timeset, "ts1"]))
        put(db, "solve", S2, "realized_periods", Array(["p2040", "p2045"]))
        put(db, "solve", S2, "invest_periods", Array(["p2045"]))
        put(db, "solve", S2, "realized_invest_periods", Array(["p2045"]))
        put(db, "solve", S2, "solve_mode", "single_solve")
        put(db, "solve", S2, "years_represented", Map(["p2040", "p2045"], [1.0, 1.0]))
        db.commit_session("test: a later solve dispatches a period an earlier solve realized investment in")


def _run_chain_and_write_outputs(tmp_path, monkeypatch, *, no_p2040_invest: bool = False) -> tuple[Path, Path]:
    """Solve the two-solve chain and write its csv outputs; returns the work folder and the csv folder."""
    from flextool.engine import run_chain_from_db
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
    from flextool.process_outputs.write_outputs import write_outputs
    from flextool.update_flextool.db_migration import migrate_database
    from db_utils import json_to_db

    repo = Path(__file__).resolve().parents[2]
    db_url = json_to_db(repo / "tests" / "fixtures" / "stoch_two_period_midhorizon_nreal0.json",
                        tmp_path / "db.sqlite")
    migrate_database(db_url)
    _add_two_solve_scenario(db_url, no_p2040_invest=no_p2040_invest)
    scenario = SCENARIO_NO_P2040_INVEST if no_p2040_invest else SCENARIO

    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")
    monkeypatch.chdir(tmp_path)
    wf = tmp_path / "w"
    steps = run_chain_from_db(db_url, scenario, work_folder=wf, csv_dump=True, keep_solutions=True)
    assert [s.solve_name for s in steps.values()] == [S1, S2]

    last = next(reversed(steps.values()))
    last.flex_data_provider.snapshot_processed_inputs(wf)
    write_outputs(
        scenario_name=scenario, output_location=str(wf), subdir="out",
        output_config_path=str(repo / "flextool" / "schemas" / "default_plots.yaml"),
        write_methods=["csv"], fallback_output_location=str(wf), raw_output_dir=str(wf / "output_raw"),
        solution=last.solution, solve_name=last.solve_name,
        solve_steps=[(s.solve_name, s.flex_data, s.effective_solution) for s in steps.values()],
        flex_data_provider=FlexDataProvider(),
    )
    return wf, wf / "output_csv" / "out"


@pytest.mark.solver
def test_later_solve_does_not_overwrite_realized_investment(tmp_path, monkeypatch):
    wf, csv_dir = _run_chain_and_write_outputs(tmp_path, monkeypatch)

    # The per-solve slices: s1 realizes p2035 and p2040; s2 carries a 0 row at its dispatch period p2040.
    v1 = pl.read_parquet(wf / "output_raw" / f"v_invest__{S1}.parquet")
    assert dict(zip(v1["period"], v1["base"])) == pytest.approx({"p2035": 60.0, "p2040": 60.0})

    capacity = pl.read_csv(csv_dir / "unit_capacity__d.csv").filter(pl.col("unit") == "base").sort("period")
    assert dict(zip(capacity["period"], capacity["invested"])) == pytest.approx(EXPECTED_INVESTED), capacity
    # The investment table is labelled with the solve that realizes each investment.
    costs = pl.read_csv(csv_dir / "annualized_costs__d.csv").filter(pl.col("period") == "p2040")
    assert costs["unit investment & retirement"].to_list() == pytest.approx(
        [EXPECTED_P2040_ANNUALIZED_INVEST_MCUR], rel=1e-5)


@pytest.mark.solver
def test_invest_marginal_of_a_period_comes_from_the_realizing_solve(tmp_path, monkeypatch):
    """The p2040 investment marginal is ``s1``'s reduced cost, not ``s2``'s 0 row (item 266).

    Data as the module chain, but base has 70 MW existing and every period uses ts1 (``s1``: p2035 / p2040,
    ``s2``: p2040 / p2045), so the residual demand is 120 - 60 = 60 MW < 70 MW everywhere: nothing is invested,
    no capacity constraint binds and extra capacity has no dispatch value.  ``s2`` has no p2040 invest column, its
    p2040 row is a 0.

    Hand marginal = -(objective cost of 1 MW invested in d) / 1000 [CUR/kW] (value 0).  Undiscounted (factor 1,
    years_represented 1), 1 MW invested in d costs invest_cost[d] * 1000 CUR/MW * a (a = 0.0650514, r = 0.05,
    n = 30) in each period of the investing solve it is alive in:

    * p2035 (``s1``, alive in p2035 and p2040): -(3 * a * 2) = -0.3903084 CUR/kW
    * p2040 (``s1``, alive in p2040):           -(1 * a * 1) = -0.0650514 CUR/kW
    * p2045 (``s2``, alive in p2045):           -(1 * a * 1) = -0.0650514 CUR/kW
    """
    wf, csv_dir = _run_chain_and_write_outputs(tmp_path, monkeypatch, no_p2040_invest=True)

    # s1 reads its invest duals on the investment axis: a p2040 row although it dispatches p2035 only.
    d1 = pl.read_parquet(wf / "output_raw" / f"v_dual_invest_unit__{S1}.parquet")
    assert dict(zip(d1["period"], d1["base"])) == pytest.approx(
        {"p2035": 6000.0 * _ANNUITY_30Y, "p2040": 1000.0 * _ANNUITY_30Y}, rel=1e-5)

    capacity = pl.read_csv(csv_dir / "unit_capacity__d.csv").filter(pl.col("unit") == "base").sort("period")
    assert dict(zip(capacity["period"], capacity["invested"])) == pytest.approx(
        {"p2035": 0.0, "p2040": 0.0, "p2045": 0.0}), capacity
    marginal = pl.read_csv(csv_dir / "unit_invest_marginal__d.csv").sort("period")
    expected = {"p2035": -3.0 * _ANNUITY_30Y * 2, "p2040": -1.0 * _ANNUITY_30Y, "p2045": -1.0 * _ANNUITY_30Y}
    assert dict(zip(marginal["period"], marginal["base"])) == pytest.approx(expected, rel=1e-5), marginal
