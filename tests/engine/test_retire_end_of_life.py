"""Existing capacity reaches its end of life inside the horizon; retiring part of it before then.

Fixture ``tests.json`` (built from JSON) with the alternatives of ``test_chain_costs``: coal_plant has 500 MW
existing (one unit of 500 MW), fixed_cost 100 CUR/kW/yr, discount_rate 0.05; four periods p2020 … p2035 of 5
years each, so the run years are Y = 0, 5, 10, 15 and the horizon ends at 20.  SINGLE solves all four periods at
once, CHAIN (``invest_solveSequence_5weeks``) solves y2020 … y2035, each its own period and the next one.

End of life (``reinvest_choice`` / ``no_investment``): the run year of the run's first period plus ``lifetime``,
one value per entity, in every solve of the run.  Lifetime 15: EOL = 0 + 15 = 15, so the existing 500 MW operate
in p2020, p2025, p2030 (Y < 15) and are gone in p2035 (Y = 15 ≥ 15).
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._unit_db import FLEXTOOL_ROOT, b64, make_db, run, utf8, without_realization
from tests.engine.test_chain_costs import (
    F_OPS,
    FIXED,
    MEGA,
    PERIODS,
    _assert_same_results,
    _no_battery_invest,
    _results,
)
from tests.engine.test_user_constraint_capacity_terms import CHAIN, PLANT, SINGLE

FAN = FLEXTOOL_ROOT / "tests" / "fixtures" / "stoch_two_period_invest.json"


def _ed(param, entity: str = PLANT) -> dict[str, float]:
    frame = utf8(param.frame).filter(pl.col("e") == entity)
    return dict(zip(frame["d"].to_list(), frame["value"].to_list()))


def _lifetime(method: str, lifetime: float, entity: str = PLANT) -> list:
    return [["unit", entity, "lifetime", b64(lifetime, "float")],
            ["unit", entity, "lifetime_method", b64(method, "str")]]


NO_INVEST = [["unit", PLANT, "invest_method", b64("not_allowed", "str")]]


# ---------------------------------------------------------------------------
# End of life counted from the start of the run
# ---------------------------------------------------------------------------


@pytest.mark.solver
def test_existing_capacity_ends_in_the_same_period_in_a_chain(tmp_path_factory) -> None:
    """reinvest_choice, lifetime 15: the pre-existing capacity is 500 MW in p2020 … p2030 and 0 in p2035 in the
    single solve and in every solve of the chain (before: a later solve of the chain counted the lifetime from
    its own first period, so y2030 and y2035 kept 500 MW in p2035)."""
    values = _lifetime("reinvest_choice", 15.0) + NO_INVEST
    want = {"p2020": 500.0, "p2025": 500.0, "p2030": 500.0, "p2035": 0.0}
    for kind, chain in (("single", SINGLE), ("chain", CHAIN)):
        url, scen = make_db(tmp_path_factory, alt=f"eol_{kind}", values=values, base_chain=_no_battery_invest(chain))
        steps = run(tmp_path_factory, url, scen)
        for name, step in steps.items():
            got = _ed(step.flex_data.p_entity_pre_existing)
            assert got == {d: want[d] for d in got}, (kind, name)
            capacity = _ed(step.flex_data.p_entity_all_existing)
            assert capacity == {d: want[d] for d in capacity}, (kind, name)


@pytest.mark.solver
def test_chain_reports_fixed_cost_of_existing_capacity_until_its_end_of_life(tmp_path_factory) -> None:
    """The reported fixed cost of the existing 500 MW: 500 · 1e5 · F_ops(d) in p2020 … p2030, nothing in p2035;
    the chain reports what the single solve reports (before: the chain charged it in p2035 too)."""
    values = _lifetime("reinvest_choice", 15.0) + NO_INVEST
    single = _results(tmp_path_factory, "eol_fc_single", values, _no_battery_invest(SINGLE))
    chain = _results(tmp_path_factory, "eol_fc_chain", values, _no_battery_invest(CHAIN))
    _assert_same_results(chain, single)
    discounted = chain["cost_unit_discounted__d"]
    for d in PERIODS:
        rows = discounted[(discounted["period"] == d) & (discounted["unit"] == PLANT)]
        fixed = rows["fixed cost pre-existing"].sum() if len(rows) else 0.0
        want = 0.0 if d == "p2035" else 500.0 * FIXED * F_OPS[d]
        assert fixed * MEGA == pytest.approx(want, rel=1e-7, abs=1e-6), d


@pytest.mark.solver
@pytest.mark.parametrize("chain", [["init", "base", "stoch"], ["init", "base", "stoch", "recourse"]],
                         ids=["none", "recourse"])
def test_fan_in_the_first_period_counts_the_end_of_life_once(tmp_path_factory, chain) -> None:
    """``stoch_two_period_invest``: the fan (high, low) opens in the run's first period p2035 (year 0); p2040
    starts at year 1.  wind (existing 100 MW) with reinvest_choice and lifetime 1 ends at year 0 + 1 = 1, so it
    has 100 MW in p2035 and its branches and nothing in p2040 and its branches, as in the deterministic twin
    (before: the end of life was summed over the two branch periods of p2035, year 2, and wind ran in p2040)."""
    values = _lifetime("reinvest_choice", 1.0, "wind")
    url, scen = make_db(tmp_path_factory, alt=f"eol_fan_{len(chain)}", values=values, base_chain=chain,
                        fixture_json=FAN)
    step = list(without_realization(run(tmp_path_factory, url, scen)).values())[-1]
    pre = _ed(step.flex_data.p_entity_pre_existing, "wind")
    capacity = _ed(step.flex_data.p_entity_all_existing, "wind")
    for d in ("p2035_high", "p2035_low", "p2040_high", "p2040_low"):
        want = 100.0 if d.startswith("p2035") else 0.0
        assert pre[d] == want, d
        assert capacity[d] == want, d
    flow = utf8(step.solution.value("v_flow")).filter(pl.col("p") == "wind")
    assert flow.filter(pl.col("d").str.starts_with("p2040"))["value"].abs().max() == pytest.approx(0.0, abs=1e-9)
    assert flow.filter(pl.col("d").str.starts_with("p2035"))["value"].max() > 0.0
