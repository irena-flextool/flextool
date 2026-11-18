"""The investment cost of a ``reinvest_automatic`` plant: its chain of replacements, each at the values of the period
it comes online (``_npv.chain_lumps``), on hand-derived cases.

Model: ``tests.json``, the single solve over p2020 … p2035 (5-year periods, run years 0, 5, 10, 15, horizon end
20, end_of_life_method closest_seam).  coal_plant: lifetime_method reinvest_automatic, discount_rate r = 5 %,
social_discount_rate i = 5 %, discount_offset_investment 1 (each year's payment at its end), fixed_cost 0.
Per MW, with C = 1000·invest_cost of the period a generation comes online (D50)::

    A(C, n) = C·r / (1 − (1+r)^−n)       (offset 1: the timing factor (1+r)^(off−1) is 1)
    S[a, b) = Σ_{y=a}^{b−1} (1+i)^−(y+1)  (each service year's payment, discounted from the run start)

The objective coefficient of a decision (``ed_entity_annual_discounted``) charges each generation's level price for
its service years inside the horizon::

    value = Σ_g (a_cap[g] − a_salv[g])·S[y_on[g], min(y_eol[g], 20))

Before, a reinvest_automatic decision paid the annuity of its first plant for every period from its start
(A(C_1, n)·S[y_on, 20)), whatever its lifetime, salvage value or later invest_cost.

* Constant values, payback_time = lifetime_technical, no salvage: every generation has the price A and the
  generations cover [y_on, 20) without gap or overlap, so the chain charges A·S[y_on, 20) — the endless annuity —
  for any lifetime and any rounding of the ends (L 10: [0,10) [10,20); L 12: 12 → 10, then [10, 20); L 7: 7 → 5,
  [5,10) …; L 2: the one-period rule, [0,5) [5,10) …), and from p2025: A·S[5, 20).
* invest_cost 400 (p2020, p2025) / 300 (p2030, p2035), L 10: built in p2020, A(400 000)·S[0,10) +
  A(300 000)·S[10,20); built in p2025, A(400 000)·S[5,15) + A(300 000)·S[15,20).
* net_salvage_value 200 (every period), L 10: each plant's price is lowered by its salvage credit
  a_salv = 200 000·(1+r)^−10·r/(1 − (1+r)^−10), so the value is (A − a_salv)·S[0, 20).
* payback_time 20, lifetime_technical 40, r = 3 %: the plant's price a_cap = A(C, 20)·G(20)/G(40),
  G(x) = Σ_{k<x} (1+i)^−k (the payback spread evenly over the lifetime); one generation to the horizon end.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._unit_db import FLEXTOOL_ROOT, PLANT, RunCache, b64, utf8
from tests.engine.test_chain_costs import _no_battery_invest
from tests.engine.test_user_constraint_capacity_terms import SINGLE

R = 0.05
I_SOCIAL = 0.05
H = 20
_RUNS = RunCache()


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()


def ann(r: float, n: float) -> float:
    return r / (1.0 - (1.0 + r) ** -n)


def span(a: int, b: int, off: float = 1.0) -> float:
    """S[a, b): the service years a … b−1, each payment off years into its year."""
    return sum((1.0 + I_SOCIAL) ** -(y + off) for y in range(a, min(b, H)))


def _map(values: dict[str, float]) -> list:
    return b64({"index_type": "str", "index_name": "period", "data": [[k, v] for k, v in values.items()]}, "map")


def _values(life, *, cost=400.0, extra: list | None = None, off: float = 1.0, rate: float = R) -> list:
    return [["model", "flexTool", "social_discount_rate", b64(I_SOCIAL, "float")],
            ["model", "flexTool", "discount_offset_investment", b64(off, "float")],
            ["unit", PLANT, "invest_cost", cost if isinstance(cost, list) else b64(cost, "float")],
            ["unit", PLANT, "discount_rate", b64(rate, "float")],
            ["unit", PLANT, "fixed_cost", b64(0.0, "float")],
            ["unit", PLANT, "invest_method", b64("invest_no_limit", "str")],
            ["unit", PLANT, "lifetime_method", b64("reinvest_automatic", "str")],
            ["unit", PLANT, "lifetime_technical", b64(life, "float")], *(extra or [])]


def _lumps(tmp_path_factory, alt: str, values: list) -> dict[str, float]:
    steps, _, _ = _RUNS.get(tmp_path_factory, alt=alt, values=values, base_chain=_no_battery_invest(SINGLE))
    fd = list(steps.values())[-1].flex_data
    frame = utf8(fd.ed_entity_annual_discounted.frame).filter(pl.col("e") == PLANT)
    return dict(zip(frame["d"].to_list(), frame["value"].to_list()))


@pytest.mark.solver
@pytest.mark.parametrize("life", [10.0, 12.0, 7.0, 2.0])
def test_constant_chain_charges_the_endless_annuity(tmp_path_factory, life) -> None:
    a = 400_000.0 * ann(R, life)
    got = _lumps(tmp_path_factory, f"rp_const_{int(life)}", _values(life))
    assert got["p2020"] == pytest.approx(a * span(0, 20), rel=1e-12)
    assert got["p2025"] == pytest.approx(a * span(5, 20), rel=1e-12)


@pytest.mark.solver
def test_constant_chain_at_offset_zero(tmp_path_factory) -> None:
    """Offset 0: every payment at the start of its year, the annuity divided by 1 + r; the chain is still the
    endless annuity."""
    a = 400_000.0 * ann(R, 10.0) / (1.0 + R)
    got = _lumps(tmp_path_factory, "rp_const_off0", _values(10.0, off=0.0))
    assert got["p2020"] == pytest.approx(a * span(0, 20, off=0.0), rel=1e-12)


@pytest.mark.solver
def test_replacement_pays_the_invest_cost_of_its_online_period(tmp_path_factory) -> None:
    cost = _map({"p2020": 400.0, "p2025": 400.0, "p2030": 300.0, "p2035": 300.0})
    a400, a300 = 400_000.0 * ann(R, 10.0), 300_000.0 * ann(R, 10.0)
    got = _lumps(tmp_path_factory, "rp_falling", _values(10.0, cost=cost))
    assert got["p2020"] == pytest.approx(a400 * span(0, 10) + a300 * span(10, 20), rel=1e-12)   # 584 173.98
    assert got["p2025"] == pytest.approx(a400 * span(5, 15) + a300 * span(15, 20), rel=1e-12)   # 394 320.48


@pytest.mark.solver
def test_salvage_value_lowers_each_plants_price(tmp_path_factory) -> None:
    a = 400_000.0 * ann(R, 10.0)
    a_salv = 200_000.0 * (1.0 + R) ** -10 * ann(R, 10.0)
    got = _lumps(tmp_path_factory, "rp_nsv", _values(10.0, extra=[
        ["unit", PLANT, "net_salvage_value", b64(200.0, "float")]]))
    assert got["p2020"] == pytest.approx((a - a_salv) * span(0, 20), rel=1e-12)   # 447 404.75


@pytest.mark.solver
def test_payback_shorter_than_the_lifetime_is_spread_over_the_lifetime(tmp_path_factory) -> None:
    def g(x: int) -> float:
        return sum((1.0 + I_SOCIAL) ** -k for k in range(x))
    a_cap = 400_000.0 * ann(0.03, 20.0) * g(20) / g(40)
    got = _lumps(tmp_path_factory, "rp_payback", _values(40.0, rate=0.03, extra=[
        ["unit", PLANT, "payback_time", b64(20.0, "float")]]))
    assert got["p2020"] == pytest.approx(a_cap * span(0, 20), rel=1e-12)   # 243 347.43


STOCH_JSON = FLEXTOOL_ROOT / "tests" / "fixtures" / "stoch_two_period_invest.json"


@pytest.mark.solver
def test_replacement_inside_a_fan_pays_the_invest_cost_of_its_branch(tmp_path_factory) -> None:
    """``stoch_two_period_invest.json`` ``recourse``: one-year periods p2035, p2040 (run years 0, 1), forecast
    branches high / low from p2035, social_discount_rate 0 (every period's factor 1), offset 0, discount_rate 5 %.
    peaker: reinvest_automatic, lifetime_technical 1 (each period its own plant), invest_cost 1 in p2035 and 3 in
    p2040, the branch high's own 5 in p2040_high.  At offset 0 a one-year plant costs A = C·ann(5 %, 1)/1.05 = C,
    so a decision in a branch's p2035 pays its first plant at 1000 and its replacement at the value of its own
    branch's p2040: high 1000 + 5000 = 6000, low 1000 + 3000 = 4000 (before: 1000 a year for both years, 2000)."""
    cost = b64({"index_type": "str", "index_name": "period",
                "data": [["p2035", 1.0], ["p2040", 3.0], ["p2040_high", 5.0]]}, "map")
    steps, _, _ = _RUNS.get(tmp_path_factory, alt="rp_fan", fixture_json=STOCH_JSON,
                            values=[["unit", "peaker", "lifetime_method", b64("reinvest_automatic", "str")],
                                    ["unit", "peaker", "invest_cost", cost]],
                            base_chain=["init", "base", "stoch", "recourse"])
    fd = steps["stoch_2p_inv"].flex_data
    frame = utf8(fd.ed_entity_annual_discounted.frame).filter(pl.col("e") == "peaker")
    got = dict(zip(frame["d"].to_list(), frame["value"].to_list()))
    assert got["p2035_high"] == pytest.approx(6000.0, rel=1e-12)
    assert got["p2035_low"] == pytest.approx(4000.0, rel=1e-12)
    assert got["p2040_high"] == pytest.approx(5000.0, rel=1e-12)
    assert got["p2040_low"] == pytest.approx(3000.0, rel=1e-12)


def test_warning_names_an_entity_whose_plant_earns_money(capsys) -> None:
    """A generation priced below 0 a year (its salvage credit above its capital price: a replacement whose period
    has no invest_cost but a net_salvage_value) is named once per run; a priced one is not."""
    from flextool.engine import _behaviour_changes
    from flextool.engine.preprocess.params._cohorts import _cohort_frame
    from flextool.engine.preprocess.params._npv import warn_salvage_above_cost

    def row(e, cohort, a_cap, a_salv):
        return (cohort, e, "generation", "reinvest_automatic", "closest_seam", "p2020", "p2030", 10.0, 0.0, 10.0,
                20.0, None, 20.0, None, a_cap, a_cap, 10.0, 0.05, a_salv, 200_000.0, "invest:p2020", None)

    _behaviour_changes.start_run(None)
    table = _cohort_frame([row("free", "gen:invest:p2020:2", 0.0, 15_900.91),
                           row("priced", "gen:invest:p2020:2", 51_801.83, 15_900.91)])
    warn_salvage_above_cost(table)
    warn_salvage_above_cost(table)
    out = capsys.readouterr().out
    assert out.count("Warning: free:") == 1 and "priced" not in out


@pytest.mark.solver
def test_replacement_is_reported_when_it_comes_online(tmp_path_factory) -> None:
    """100 MW built in p2020 (invest_period, exactly 100 MW there and nothing elsewhere), invest_cost 400 in p2020 /
    p2025 and 300 in p2030 / p2035, L 10: the plant of p2020 serves [0, 10), its replacement (online in p2030 at
    the p2030 invest_cost) [10, 20).  The reported costs follow each plant:

    * the price per year in service (``edd_annuity``): A(400 000) in p2020 and p2025, A(300 000) in p2030 and p2035;
    * the investment lump split by the period each plant comes online (``edd_entity_annual_discounted_run``; the
      solve is the run): A(400 000)·S[0,10) at p2020, A(300 000)·S[10,20) at p2030;
    * cost_unit_discounted__d investment (M CUR): 100 · those in p2020 and p2030, 0 in p2025 and p2035;
      cost_unit_annualized__d investment: 100 · the yearly price of each period."""
    from tests.engine.test_chain_costs import MEGA, _period_map, _results, _row
    cost = _map({"p2020": 400.0, "p2025": 400.0, "p2030": 300.0, "p2035": 300.0})
    only_p2020 = _period_map({"p2020": 100.0, "p2025": 0.0, "p2030": 0.0, "p2035": 0.0})
    values = [v for v in _values(10.0, cost=cost) if v[2] != "invest_method"] + [
                                         ["unit", PLANT, "invest_method", b64("invest_period", "str")],
                                         ["unit", PLANT, "invest_max_period", only_p2020],
                                         ["unit", PLANT, "invest_min_period", only_p2020]]
    a400, a300 = 400_000.0 * ann(R, 10.0), 300_000.0 * ann(R, 10.0)
    yearly = {"p2020": a400, "p2025": a400, "p2030": a300, "p2035": a300}
    lumps = {"p2020": a400 * span(0, 10), "p2025": 0.0, "p2030": a300 * span(10, 20), "p2035": 0.0}

    steps, _, _ = _RUNS.get(tmp_path_factory, alt="rp_reported", values=values,
                            base_chain=_no_battery_invest(SINGLE))
    fd = list(steps.values())[-1].flex_data
    rows = utf8(fd.edd_annuity.frame).filter((pl.col("e") == PLANT) & (pl.col("d_invest") == "p2020"))
    assert dict(zip(rows["d"].to_list(), rows["value"].to_list())) == pytest.approx(yearly, rel=1e-12)
    rows = utf8(fd.edd_entity_annual_discounted_run.frame).filter(
        (pl.col("e") == PLANT) & (pl.col("d_invest") == "p2020"))
    assert dict(zip(rows["d"].to_list(), rows["value"].to_list())) == pytest.approx(
        {d: v for d, v in lumps.items() if v}, rel=1e-12)

    results = _results(tmp_path_factory, "rp_reported_csv", values, _no_battery_invest(SINGLE))
    for d in ("p2020", "p2025", "p2030", "p2035"):
        discounted = _row(results["cost_unit_discounted__d"], d)["investment"] * MEGA
        assert discounted == pytest.approx(100.0 * lumps[d], rel=1e-7, abs=1e-6), d
        annualized = _row(results["cost_unit_annualized__d"], d)["investment"] * MEGA
        assert annualized == pytest.approx(100.0 * yearly[d], rel=1e-7), d
