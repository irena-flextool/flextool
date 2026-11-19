"""Capacity by cohort and retirements by cause (``output._cohort_capacity.cohort_capacity``,
``out_capacity.retirement_causes``).

Hand values (the pure function, five-year periods p2020 … p2035 at run years 0, 5, 10, 15): unit u has 100 MW
existing (end of life at year 10, the start of p2030); 50 MW are built at p2025 (lifetime 10: end of life at
p2035) and 20 MW of them retired at p2030::

    existing:p2020   100 in p2020, p2025
    invest:p2025      50 in p2025, 50 − 20 = 30 in p2030
    end_of_life       p2030: 100 (the existing capacity leaves), p2035: 30 (the rest of the investment)
    economic          p2030: 20

Relational, on solved runs written to CSV: the cohorts of an entity sum to the capacity table's ``total`` in every
period, and from one period to the next ``total[d] − total[d⁻] = invested[d] − end_of_life[d] − economic[d] +
rise[d]`` (``rise`` a rise of a period-varying ``existing``; a life extension keeps its parent's capacity, so it
adds as much as it removes), with ``economic`` the capacity table's ``divested``.  Cases: a single solve, a chain
of solves, an investment solve with a nested dispatch solve, an investment solve with a rolling dispatch solve, a
stochastic solve, reinvest_automatic (replacement plants), a life extension and a period-map ``existing`` that rises
and falls.
"""
from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from tests.engine._unit_db import PLANT, b64
from tests.engine.test_chain_costs import (
    ENTITIES,
    PERIODS,
    STOCH_RD_CHAIN,
    STOCH_RD_JSON,
    STOCH_RD_VALUES,
    _no_battery_invest,
    _period_map,
    _solve,
    _solves,
    _write_csv,
)
from tests.engine.test_user_constraint_capacity_terms import CHAIN, SINGLE

YEARS = {"p2020": 0.0, "p2025": 5.0, "p2030": 10.0, "p2035": 15.0}
#: The CSV files hold eight significant digits.
CSV = 1e-6


def _cohort(rows: list[tuple]) -> pl.DataFrame:
    keys = ("cohort", "e", "kind", "root", "d_on", "d_eol", "y_on", "y_eol")
    return pl.DataFrame([dict(zip(keys, r)) for r in rows],
                        schema={k: (pl.Float64 if k.startswith("y_") else pl.Utf8) for k in keys})


def test_cohort_capacity_and_causes_by_hand() -> None:
    from flextool.engine.output._cohort_capacity import cohort_capacity
    from flextool.process_outputs.out_capacity import retirement_causes

    cohort = _cohort([("existing:p2020", "u", "existing", "existing:p2020", "p2020", "p2030", 0.0, 10.0),
                      ("invest:p2025", "u", "invest", "invest:p2025", "p2025", "p2035", 5.0, 15.0)])
    alive = pl.DataFrame({"e": ["u"] * 4, "cohort": ["existing:p2020"] * 2 + ["invest:p2025"] * 2,
                          "kind": ["existing"] * 2 + ["invest"] * 2, "d": ["p2020", "p2025", "p2025", "p2030"]})
    pre = pl.DataFrame({"e": ["u", "u"], "d": ["p2020", "p2025"], "mw": [100.0, 100.0]})
    sizes = pl.DataFrame({"e": ["u"], "cohort": ["invest:p2025"], "mw": [50.0]})
    retired = pl.DataFrame({"e": ["u"], "cohort": ["invest:p2025"], "d_divest": ["p2030"], "d": ["p2030"],
                            "mw": [20.0]})
    out = cohort_capacity(cohort, alive, None, None, pre, sizes, retired, YEARS, PERIODS)
    got = {(c, d): v for c, d, v in out.select("cohort", "d", "capacity").iter_rows()}
    assert got == {("existing:p2020", "p2020"): 100.0, ("existing:p2020", "p2025"): 100.0,
                   ("invest:p2025", "p2025"): 50.0, ("invest:p2025", "p2030"): 30.0}
    kinds = dict(out.select("cohort", "kind").unique().iter_rows())
    assert kinds == {"existing:p2020": "existing", "invest:p2025": "invested"}
    causes = {d: (eol, econ) for _e, d, eol, econ, _r, _x in retirement_causes(out, PERIODS).iter_rows()}
    assert causes == {"p2020": (0.0, 0.0), "p2025": (0.0, 0.0), "p2030": (100.0, 20.0), "p2035": (30.0, 0.0)}


def test_carried_cohorts_and_their_retirements_count_in_a_later_solve() -> None:
    """A later solve (p2030, p2035) gets invest:p2025 (50 MW) from the handoff with 20 MW retired at p2025: 30 MW in
    p2030, none in p2035 (its end of life); its own decision of p2030 (40 MW) is a cohort of its own."""
    from flextool.engine.output._cohort_capacity import cohort_capacity

    cohort = _cohort([("invest:p2030", "u", "invest", "invest:p2030", "p2030", None, 10.0, float("inf"))])
    alive = pl.DataFrame({"e": ["u", "u"], "cohort": ["invest:p2030"] * 2, "kind": ["invest"] * 2,
                          "d": ["p2030", "p2035"]})
    carried = pl.DataFrame({"entity": ["u"], "cohort": ["invest:p2025"], "kind": ["invest"], "root": ["invest:p2025"],
                            "period_online": ["p2025"], "period_eol": ["p2035"], "y_on": [5.0], "y_eol": [15.0],
                            "size": [50.0]})
    carried_retire = pl.DataFrame({"entity": ["u"], "cohort": ["invest:p2025"], "period": ["p2025"], "value": [20.0]})
    empty_pre = pl.DataFrame(schema={"e": pl.Utf8, "d": pl.Utf8, "mw": pl.Float64})
    sizes = pl.DataFrame({"e": ["u"], "cohort": ["invest:p2030"], "mw": [40.0]})
    retired = pl.DataFrame(schema={"e": pl.Utf8, "cohort": pl.Utf8, "d_divest": pl.Utf8, "d": pl.Utf8,
                                   "mw": pl.Float64})
    out = cohort_capacity(cohort, alive, carried, carried_retire, empty_pre, sizes, retired, YEARS,
                          ["p2030", "p2035"])
    got = {(c, d): v for c, d, v in out.select("cohort", "d", "capacity").iter_rows()}
    assert got == {("invest:p2025", "p2030"): 30.0, ("invest:p2030", "p2030"): 40.0, ("invest:p2030", "p2035"): 40.0}


# ---------------------------------------------------------------------------
# Solved runs
# ---------------------------------------------------------------------------

def _plant(method: str, salvage: float = 50.0, **extra) -> list:
    values = [["model", "flexTool", "social_discount_rate", b64(0.05, "float")],
              ["unit", PLANT, "invest_method", b64(extra.pop("invest_method", "invest_retire_no_limit"), "str")],
              ["unit", PLANT, "lifetime_method", b64(method, "str")],
              ["unit", PLANT, "lifetime_technical", b64(10.0, "float")],
              ["unit", PLANT, "invest_cost", b64(400.0, "float")],
              ["unit", PLANT, "net_salvage_value", b64(salvage, "float")]]
    return values + [["unit", PLANT, k, v] for k, v in extra.items()]


EXISTING_MAP = {"p2020": 500.0, "p2025": 600.0, "p2030": 600.0, "p2035": 600.0}
NESTED = (_solves(["cc_invest", "cc_dispatch"])
          + _solve("cc_invest", PERIODS, invest_periods=PERIODS, realized_invest_periods=PERIODS)
          + _solve("cc_dispatch", PERIODS, realized_periods=PERIODS))
ROLLING = ([["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", b64(48.0, "float")],
            ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", b64(36.0, "float")]]
           + _solves(["cc_invest", "dispatch_fullYear_roll"])
           + _solve("cc_invest", PERIODS, invest_periods=PERIODS, realized_invest_periods=PERIODS))
SOLVE_ENTITIES = ENTITIES + [["solve", "cc_invest", None], ["solve", "cc_dispatch", None]]
CASES = {
    # case: (values, chain, entities, fixture json, entity, rise)
    "single": (_plant("reinvest_choice"), _no_battery_invest(SINGLE), None, None, PLANT, {}),
    "chain": (_plant("reinvest_choice"), _no_battery_invest(CHAIN), None, None, PLANT, {}),
    "nested": (_plant("reinvest_choice") + NESTED, _no_battery_invest(CHAIN), SOLVE_ENTITIES, None, PLANT, {}),
    "rolling": (_plant("reinvest_choice") + ROLLING,
                ["init", "west", "coal", "coal_invest", "wind", "fullYear", "5weeks", "dispatch_fullYear_roll"],
                SOLVE_ENTITIES, None, PLANT, {}),
    "automatic": (_plant("reinvest_automatic"), _no_battery_invest(SINGLE), None, None, PLANT, {}),
    "extension": (_plant("reinvest_choice", invest_method="not_allowed",
                         life_extension_cost=b64(100.0, "float"), life_extension_years=b64(10.0, "float")),
                  _no_battery_invest(SINGLE), None, None, PLANT, {}),
    "extension_chain": (_plant("reinvest_choice", invest_method="not_allowed",
                               life_extension_cost=b64(100.0, "float"), life_extension_years=b64(10.0, "float")),
                        _no_battery_invest(CHAIN), None, None, PLANT, {}),
    "existing_map": (_plant("reinvest_choice", existing=_period_map(EXISTING_MAP)), _no_battery_invest(SINGLE),
                     None, None, PLANT, {"p2025": 100.0}),
    # A stochastic solve and its realization solve (the cohort tables report the realized periods, as the
    # capacity tables).
    "stochastic": (STOCH_RD_VALUES + [["unit", "peaker", "lifetime_method", b64("reinvest_choice", "str")],
                                      ["unit", "peaker", "lifetime_technical", b64(10.0, "float")]],
                   STOCH_RD_CHAIN, None, STOCH_RD_JSON, "peaker", {}),
}
_TABLES: dict[str, dict[str, pd.DataFrame]] = {}


def _tables(tmp_path_factory, case: str) -> dict[str, pd.DataFrame]:
    if case not in _TABLES:
        values, chain, entities, fixture, _entity, _rise = CASES[case]
        out = _write_csv(tmp_path_factory, f"cohort_{case}", values, chain, entities=entities, fixture_json=fixture)
        _TABLES[case] = {stem: pd.read_csv(out / f"{stem}.csv")
                         for stem in ("unit_capacity__d", "unit_cohort_capacity__d", "unit_retirement__d")}
    return _TABLES[case]


@pytest.mark.solver
@pytest.mark.parametrize("case", sorted(CASES))
def test_cohorts_add_up_to_the_capacity_and_the_causes_explain_its_change(tmp_path_factory, case) -> None:
    entity, rise = CASES[case][4], CASES[case][5]
    tables = _tables(tmp_path_factory, case)
    capacity = tables["unit_capacity__d"].set_index(["unit", "period"]).fillna(0.0)
    cohorts = tables["unit_cohort_capacity__d"]
    causes = tables["unit_retirement__d"].set_index(["unit", "period"])
    assert (cohorts["unit"] == entity).any(), case
    by_entity = cohorts.groupby(["unit", "period"])["capacity"].sum()
    for key, total in capacity["total"].items():
        assert by_entity.get(key, 0.0) == pytest.approx(total, rel=CSV, abs=CSV), (case, key)
    periods = sorted(capacity.loc[entity].index)
    for before, d in zip(periods, periods[1:]):
        row, gone = capacity.loc[(entity, d)], causes.loc[(entity, d)]
        assert gone["economic"] == pytest.approx(row["divested"], rel=CSV, abs=CSV), (case, d)
        change = row["total"] - capacity.loc[(entity, before), "total"]
        want = row["invested"] - gone["end_of_life"] - gone["economic"] + rise.get(d, 0.0)
        assert change == pytest.approx(want, rel=CSV, abs=CSV), (case, d)


@pytest.mark.solver
def test_case_shapes(tmp_path_factory) -> None:
    """The cases hold what they claim: replacement plants under reinvest_automatic (the existing plant's chain
    replaced at p2030 with its capacity kept), an extension cohort and its cause, and the existing map's fall at
    the end of its life."""
    for case in ("nested", "rolling", "chain"):
        rows = _tables(tmp_path_factory, case)["unit_cohort_capacity__d"]
        assert (rows["kind"] == "invested").any(), case
    auto = _tables(tmp_path_factory, "automatic")
    rows = auto["unit_cohort_capacity__d"]
    replaced = rows[(rows["unit"] == PLANT) & (rows["kind"] == "replacement") & (rows["period"] == "p2030")]
    assert "gen:existing:p2020:2" in set(replaced["cohort"])
    causes = auto["unit_retirement__d"].set_index(["unit", "period"])
    assert causes.loc[(PLANT, "p2030"), "replaced"] == pytest.approx(replaced["capacity"].sum(), rel=1e-12)
    assert causes.loc[(PLANT, "p2030"), "end_of_life"] == pytest.approx(0.0, abs=1e-9)
    ext = _tables(tmp_path_factory, "extension")
    rows = ext["unit_cohort_capacity__d"]
    kept = rows[(rows["unit"] == PLANT) & (rows["kind"] == "extension")]
    assert set(kept["period"]) == {"p2030", "p2035"}
    assert kept["capacity"].to_numpy() == pytest.approx([500.0, 500.0], abs=1e-6)
    causes = ext["unit_retirement__d"].set_index(["unit", "period"])
    assert causes.loc[(PLANT, "p2030"), "extended"] == pytest.approx(500.0, abs=1e-6)
    assert causes.loc[(PLANT, "p2030"), "end_of_life"] == pytest.approx(0.0, abs=1e-6)
    # In a chain the extension decided by the solve of p2030 is carried to the solve of p2035 as an extension
    # cohort, never as existing capacity: the capacity table's existing is 0 there (the existing cohort ended at
    # p2030).
    chained = _tables(tmp_path_factory, "extension_chain")
    capacity = chained["unit_capacity__d"].set_index(["unit", "period"])
    assert capacity.loc[(PLANT, "p2035"), "existing"] == pytest.approx(0.0, abs=1e-9)
    assert capacity.loc[(PLANT, "p2035"), "total"] == pytest.approx(500.0, rel=CSV)
    rows = chained["unit_cohort_capacity__d"]
    assert rows[(rows["unit"] == PLANT) & (rows["period"] == "p2035")]["kind"].tolist() == ["extension"]
    mapped = _tables(tmp_path_factory, "existing_map")
    existing = mapped["unit_cohort_capacity__d"]
    existing = existing[(existing["unit"] == PLANT) & (existing["kind"] == "existing")].set_index("period")
    assert existing.loc["p2025", "capacity"] == pytest.approx(600.0 - mapped["unit_capacity__d"].set_index(
        ["unit", "period"]).loc[(PLANT, "p2020"), "divested"], rel=CSV)


@pytest.mark.solver
def test_existing_capacity_at_its_end_of_life(tmp_path_factory) -> None:
    """500 MW existing of age 5 with a lifetime of 10 years that may retire (it saves its fixed cost): in service
    in p2020 only (its end of life at year 5, the start of p2025), 500 MW less the part retired at p2020; what is
    left leaves at p2025 by end of life."""
    values = _plant("reinvest_choice", salvage=0.0, invest_method="retire_no_limit",
                    age_at_model_start=b64(5.0, "float"))
    out = _write_csv(tmp_path_factory, "cohort_age5", values, _no_battery_invest(SINGLE))
    rows = pd.read_csv(out / "unit_cohort_capacity__d.csv")
    rows = rows[rows["unit"] == PLANT]
    assert rows[["cohort", "kind", "period_online", "period_end_of_life", "period"]].values.tolist() == [
        ["existing:p2020", "existing", "p2020", "p2025", "p2020"]]
    retired = pd.read_csv(out / "unit_capacity__d.csv").set_index(["unit", "period"]).loc[(PLANT, "p2020"),
                                                                                          "divested"]
    assert rows["capacity"].to_numpy() == pytest.approx([500.0 - retired], rel=CSV)
    causes = pd.read_csv(out / "unit_retirement__d.csv").set_index(["unit", "period"])
    assert causes.loc[(PLANT, "p2025"), "end_of_life"] == pytest.approx(500.0 - retired, rel=CSV)
