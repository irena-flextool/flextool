"""Model-built ``reinvest_automatic`` capacity can be retired; a retirement before the end of life also avoids every
replacement still to come (``_retire_settlement.chain_retirement_costs``), on hand-derived cases.

Model: ``tests.json``, the single solve over p2020 … p2035 (5-year periods, run years 0, 5, 10, 15, horizon end
20).  coal_plant: lifetime_method reinvest_automatic, lifetime_technical L = 10, invest_method
invest_retire_no_limit, invest_cost 400 (C = 400 000 CUR/MW), net_salvage_value 200 in every period (S = 200 000),
discount_rate r = social_discount_rate i = 5 %, discount_offset_investment 1, fixed_cost 0.  A MW built in p2020 is
held by the plant of p2020 over [0, 10) and its replacement over [10, 20); v = 1/1.05::

    A = C·r/(1 − (1+r)^−10),  a_salv = S·(1+r)^−10·r/(1 − (1+r)^−10),  a_b = A − a_salv
    S[a, b) = Σ_{y=a}^{b−1} v^(y+1);  the investment coefficient of the p2020 decision a_b·S[0, 20) = 447 404.75

At r = i a plant's contract is worth C when it comes online and its salvage S at its end of life, so a MW built at
0 and retired at τ costs, all told (investment coefficient + retirement coefficient):

* τ = 5 (the first plant at m = 5; its replacement never bought): C − S·v^5 = 243 294.77;
* τ = 10 (the replacement's own start: never bought; the first plant ran its life and paid its salvage):
  C − S·v^10 = 277 217.35;
* τ = 15 (the replacement at m = 5): (C − S·v^10) + (C − S·v^5)·v^10 = 426 579.23.

Retiring and rebuilding an identical plant at τ (the twin decision of period τ has the lump a_b·S[τ, 20) of its own
chain) changes the cost by (C − S)·(v^m − v^L)/(1 − v^L)·v^(y_on) ≥ 0 when C ≥ S: +87 862.59 at τ = 5 (m = 5),
0 at τ = 10 (a seam: the same plant at the same price), +53 940.01 at τ = 15 (m = 5, y_on = 10).
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._unit_db import PLANT, RunCache, b64, utf8
from tests.engine.test_chain_costs import _no_battery_invest
from tests.engine.test_user_constraint_capacity_terms import SINGLE

R = 0.05
V = 1.0 / 1.05
C, S = 400_000.0, 200_000.0
_RUNS = RunCache()


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()


def ann(r: float, n: float) -> float:
    return r / (1.0 - (1.0 + r) ** -n)


def span(a: int, b: int) -> float:
    return sum(V ** (y + 1) for y in range(a, b))


VALUES = [["model", "flexTool", "social_discount_rate", b64(R, "float")],
          ["model", "flexTool", "discount_offset_investment", b64(1.0, "float")],
          ["unit", PLANT, "invest_cost", b64(400.0, "float")],
          ["unit", PLANT, "net_salvage_value", b64(200.0, "float")],
          ["unit", PLANT, "discount_rate", b64(R, "float")],
          ["unit", PLANT, "fixed_cost", b64(0.0, "float")],
          ["unit", PLANT, "invest_method", b64("invest_retire_no_limit", "str")],
          ["unit", PLANT, "lifetime_method", b64("reinvest_automatic", "str")],
          ["unit", PLANT, "lifetime_technical", b64(10.0, "float")]]


def _step(tmp_path_factory):
    steps, _, _ = _RUNS.get(tmp_path_factory, alt="ra_built", values=VALUES, base_chain=_no_battery_invest(SINGLE))
    return list(steps.values())[-1]


def _by_period(param, cohort: str | None = None) -> dict[str, float]:
    frame = utf8(param.frame).filter(pl.col("e") == PLANT)
    if cohort is not None:
        frame = frame.filter(pl.col("cohort") == cohort)
    return dict(zip(frame["d"].to_list(), frame["value"].to_list()))


@pytest.mark.solver
def test_retiring_before_the_end_of_life_avoids_the_replacement(tmp_path_factory) -> None:
    fd = _step(tmp_path_factory).flex_data
    a_b = C * ann(R, 10.0) - S * 1.05 ** -10 * ann(R, 10.0)
    lump = _by_period(fd.ed_entity_annual_discounted)["p2020"]
    assert lump == pytest.approx(a_b * span(0, 20), rel=1e-12)
    assert lump == pytest.approx(447_404.75, abs=0.01)
    retire = _by_period(fd.cd_divest_built_cost, "invest:p2020")
    fixed = _by_period(fd.cd_divest_built_fixed, "invest:p2020")
    assert all(v == 0.0 for v in fixed.values())
    want = {"p2025": C - S * V ** 5, "p2030": C - S * V ** 10, "p2035": (C - S * V ** 10) + (C - S * V ** 5) * V ** 10}
    assert want == pytest.approx({"p2025": 243_294.77, "p2030": 277_217.35, "p2035": 426_579.23}, abs=0.01)
    for d, total in want.items():
        assert lump + retire[d] == pytest.approx(total, rel=1e-12), d


@pytest.mark.solver
def test_retiring_and_rebuilding_an_identical_plant(tmp_path_factory) -> None:
    fd = _step(tmp_path_factory).flex_data
    lumps = _by_period(fd.ed_entity_annual_discounted)
    retire = _by_period(fd.cd_divest_built_cost, "invest:p2020")
    change = {d: retire[d] + lumps[d] for d in ("p2025", "p2030", "p2035")}
    base = (C - S) * (V ** 5 - V ** 10) / (1.0 - V ** 10)
    assert base == pytest.approx(87_862.59, abs=0.01)
    assert change["p2025"] == pytest.approx(base, rel=1e-9)
    assert change["p2030"] == pytest.approx(0.0, abs=1e-6)
    assert change["p2035"] == pytest.approx(base * V ** 10, rel=1e-9)
    assert base * V ** 10 == pytest.approx(53_940.01, abs=0.01)


@pytest.mark.solver
def test_carried_chain_is_retired_against_the_later_solves_horizon(tmp_path_factory) -> None:
    """The chain of solves y2020 … y2035 (each its own period and the next): 100 MW built in p2020 by y2020 (exactly
    100 MW there, nothing elsewhere), carried with its generations to y2025, which holds p2025 and p2030 (its
    horizon ends at 15).  A MW retired there in p2025 pays the first plant's contract as it ran (the payments of
    [0, 5) and, at the start of p2025, the rest of its loan B(5) = A·Σ_{k=1..5} (1+r)^−k, less S) and gives back
    the level price of its years in service that the solve sees: the first plant's [0, 10) (history included) and
    the replacement's [10, 15), as one solve ending at 15 would::

        coefficient = A·S[0, 5) + v^5·(B(5) − S) − a_b·S[0, 15)"""
    from tests.engine.test_chain_costs import _period_map
    from tests.engine.test_user_constraint_capacity_terms import CHAIN
    only_p2020 = _period_map({"p2020": 100.0, "p2025": 0.0, "p2030": 0.0, "p2035": 0.0})
    values = [v for v in VALUES if v[2] != "invest_method"] + [
        ["unit", PLANT, "invest_method", b64("invest_retire_period", "str")],
        ["unit", PLANT, "invest_max_period", only_p2020], ["unit", PLANT, "invest_min_period", only_p2020],
        ["unit", PLANT, "retire_max_period", b64(1000.0, "float")]]
    steps, _, _ = _RUNS.get(tmp_path_factory, alt="ra_chain", values=values, base_chain=_no_battery_invest(CHAIN))
    later = steps["y2025_5week"]
    carried = later.handoff.realized_cohort if later.handoff is not None else None
    fd = later.flex_data
    a = C * ann(R, 10.0)
    a_b = a - S * 1.05 ** -10 * ann(R, 10.0)
    balance = a * sum(1.05 ** -k for k in range(1, 6))
    want = a * span(0, 5) + V ** 5 * (balance - S) - a_b * span(0, 15)
    retire = _by_period(fd.cd_divest_built_cost, "invest:p2020")
    assert retire["p2025"] == pytest.approx(want, rel=1e-12)
    assert carried is not None
    gens = carried.filter(pl.col("kind") == "generation")["cohort"].to_list()
    assert {"gen:invest:p2020:1", "gen:invest:p2020:2"} <= set(gens)


@pytest.mark.solver
@pytest.mark.parametrize("method", ["reinvest_automatic", "reinvest_choice"])
def test_shared_chain_retired_in_every_branch_pays_its_first_plant_once(tmp_path_factory, method) -> None:
    """The hedge fixture with the fan opening in p2040 (one-year periods, run years 0 and 1, branch weights summing
    to 1); base: lifetime_technical 1, invest_cost 1 in p2035 and 3 in p2040, net_salvage_value 0.5, discount_rate =
    social_discount_rate = 5 %, offset 1; 50 MW built in the shared p2035.  Retired in each branch of p2040, the
    first plant has run its life (its salvage is in its price) and the replacement of each branch is never bought:
    the probability-weighted retirements and the investment add up to the first plant alone, C − S·D(1) = 1000 −
    500 / 1.05 = 523.81, the same as the reinvest_choice twin whose plant simply ends there (it has no retirement
    option in p2040: its cohort is out of service)."""
    from tests.engine.test_chain_costs import _period_map
    from tests.engine.test_retire_built_cohorts import HEDGE_JSON
    invest = _period_map({"p2035": 50.0, "p2040": 0.0})
    values = [["model", "flextool", "social_discount_rate", b64(0.05, "float")],
              ["model", "flextool", "discount_offset_investment", b64(1.0, "float")],
              ["unit", "base", "invest_method", b64("invest_retire_period", "str")],
              ["unit", "base", "lifetime_method", b64(method, "str")],
              ["unit", "base", "lifetime_technical", b64(1.0, "float")],
              ["unit", "base", "invest_cost", _period_map({"p2035": 1.0, "p2040": 3.0})],
              ["unit", "base", "discount_rate", b64(0.05, "float")],
              ["unit", "base", "net_salvage_value", b64(0.5, "float")],
              ["unit", "base", "invest_min_period", invest], ["unit", "base", "invest_max_period", invest],
              ["unit", "base", "retire_min_period", _period_map({"p2035": 0.0, "p2040": 0.0})],
              ["unit", "base", "retire_max_period", _period_map({"p2035": 50.0, "p2040": 50.0})]]
    steps, _, _ = _RUNS.get(tmp_path_factory, alt=f"ra_fan_{method}", values=values,
                            base_chain=["init", "base", "stoch", "rp"], fixture_json=HEDGE_JSON)
    fd = [s for name, s in steps.items() if "realized" not in name][-1].flex_data
    lump = dict(zip(*[utf8(fd.ed_entity_annual_discounted.frame).filter(pl.col("e") == "base")[c].to_list()
                      for c in ("d", "value")]))["p2035"]
    frame = utf8(fd.cd_divest_built_cost.frame).filter((pl.col("e") == "base") & (pl.col("cohort") == "invest:p2035"))
    cost = dict(zip(frame["d"].to_list(), frame["value"].to_list()))
    weights = dict(zip(*[utf8(fd.pd_branch_weight.frame)[c].to_list() for c in ("d", "value")]))
    branches = sorted(d for d in cost if d != "p2035")
    if method == "reinvest_automatic":
        assert len(branches) == 2 and sum(weights[d] for d in branches) == pytest.approx(1.0, rel=1e-12)
    else:
        assert branches == []
    total = lump + sum(weights[d] * cost[d] for d in branches)
    assert total == pytest.approx(1000.0 - 500.0 / 1.05, rel=1e-12)
    assert 1000.0 - 500.0 / 1.05 == pytest.approx(523.81, abs=0.01)
