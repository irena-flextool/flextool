"""Life extensions carried across the solves of a chain (``_cohort_handoff.realized_cohorts`` / ``existing_capacity``,
``_extend.extension_frames``), hand-derived and relational.

Model: ``tests.json`` as ``test_life_extension`` (coal_plant 500 MW existing, reinvest_choice, lifetime 10: the
existing cohort ends at the start of p2030; invest_cost 400, discount_rate = social_discount_rate = 5 %, offset 1,
fixed_cost 0; life_extension_cost 200, life_extension_years 10), solved as the chain ``invest_solveSequence_5weeks``
(y2020 … y2035: each solve realizes its own period and looks one period ahead) and as the single solve.

* The solve y2030 holds p2030 and p2035: its extension of the existing cohort at p2030 costs what the single solve's
  does, 1000·200·v^10 = 122 782.65 per MW (the same periods in service), although the existing cohort's last
  period in service, p2025, is in its history: the extension is bounded by the 500 MW in service there (the run's
  existing capacity, read at p2025).
* It realizes the extension: the solve y2035 starts from it (500 MW more in service in p2035, a cohort of its own,
  kind extend), offers no extension of the existing cohort again, and counts it as capacity only: the investment
  already made (``p_entity_invested``, the prior investment of invest_max_total) and the previously invested
  capacity are the realized investments without it.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._unit_db import PLANT, RunCache, utf8
from tests.engine.test_chain_costs import _no_battery_invest
from tests.engine.test_life_extension import MW, V, _values
from tests.engine.test_user_constraint_capacity_terms import CHAIN, SINGLE

_RUNS = RunCache()
X = "extend:p2030:existing:p2020"


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()


def _steps(tmp_path_factory, alt: str, chain: list[str]) -> dict:
    steps, _, _ = _RUNS.get(tmp_path_factory, alt=alt, values=_values(cost=200.0, method="invest_total"),
                            base_chain=_no_battery_invest(chain))
    return steps


def _ed(param, keys=("cohort", "d")) -> dict:
    frame = utf8(param.frame).filter(pl.col("e") == PLANT)
    return {tuple(row[:-1]): row[-1] for row in frame.select(*keys, "value").iter_rows()}


@pytest.mark.solver
def test_extension_in_a_later_solve_costs_what_the_single_solve_charges(tmp_path_factory) -> None:
    single = list(_steps(tmp_path_factory, "lec_single", SINGLE).values())[-1].flex_data
    chain = _steps(tmp_path_factory, "lec_chain", CHAIN)["y2030_5week"].flex_data
    want = 200_000.0 * V ** 10
    assert want == pytest.approx(122_782.65, abs=0.01)
    assert _ed(chain.cd_extend_cost)[("existing:p2020", "p2030")] == pytest.approx(want, rel=1e-12)
    assert _ed(single.cd_extend_cost)[("existing:p2020", "p2030")] == pytest.approx(want, rel=1e-12)
    parents = (chain.cohort_extend.with_columns(pl.col(c).cast(pl.Utf8) for c in ("e", "d", "d_last"))
               .filter((pl.col("e") == PLANT) & (pl.col("cohort") == "existing:p2020")))
    assert parents.select("d", "d_last", "size").rows() == [("p2030", None, MW)]


@pytest.mark.solver
def test_a_realized_extension_is_carried_as_capacity_not_investment(tmp_path_factory) -> None:
    steps = _steps(tmp_path_factory, "lec_chain", CHAIN)
    decided = steps["y2030_5week"]
    extended = {c: v for (_p, c, d, v) in utf8(decided.solution.value("v_extend_p")).rows()
                if _p == PLANT and d == "p2030"}[("existing:p2020")] * MW
    assert extended == pytest.approx(MW, abs=1e-6)
    carried = decided.handoff.realized_cohort.filter(pl.col("entity") == PLANT)
    row = carried.filter(pl.col("cohort") == X).select("kind", "parent", "size", "y_on", "y_eol").row(0)
    assert row == ("extend", "existing:p2020", pytest.approx(extended), 10.0, 20.0)
    later = steps["y2035_5week"].flex_data
    # Not offered again (and the existing cohort is past its end of life).
    options = [] if later.cd_extend is None else utf8(later.cd_extend).filter(pl.col("e") == PLANT)["cohort"]
    assert "existing:p2020" not in list(options)
    # Capacity: the carried investments in service in p2035 and the extension.
    invested = {c: s for c, k, s, y_on, y_eol in carried.select("cohort", "kind", "size", "y_on", "y_eol").rows()
                if k == "invest" and y_on <= 15.0 < y_eol}
    assert _ed(later.p_entity_all_existing, ("d",))[("p2035",)] == pytest.approx(sum(invested.values()) + extended,
                                                                                rel=1e-12)
    assert _ed(later.p_entity_previously_invested_capacity, ("d",))[("p2035",)] == pytest.approx(
        sum(invested.values()), rel=1e-12)
    # The prior investment of invest_max_total: every realized investment, never the extension.
    every = sum(s for k, s in carried.select("kind", "size").rows() if k == "invest")
    prior = dict(utf8(later.p_entity_invested.frame).select("e", "value").rows())[PLANT]
    assert prior == pytest.approx(every, rel=1e-12)
    assert every < MW


def test_a_carried_extension_is_capacity_and_its_retirement_comes_off_it() -> None:
    """``existing_capacity`` of a later solve (p2035, run year 15) handed a 500 MW extension of the existing cohort
    (online at 10, end of life 20) and 200 MW of it retired in p2035, and an investment of 100 MW (online at 5, end
    of life 15, out of service in p2035): the capacity in service is 500 − 200 = 300 MW, the previously invested
    capacity and the investment already made count the investment only (0 in service, 100 in all), and the
    extension's retirement is no retirement of the existing cohort."""
    from flextool.engine.preprocess.params._cohort_handoff import existing_capacity, realized_cohort_schema
    from flextool.engine.preprocess.params._cohorts import _utf8_cohort_frame
    existing = _utf8_cohort_frame([("existing:p2020", PLANT, "existing", "reinvest_choice", "closest_seam", None,
                                    "p2020", 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, "p2030", *((None,) * 6),
                                    "existing:p2020", None)])
    schema = realized_cohort_schema()
    rows = [{"entity": PLANT, "cohort": "invest:p2025", "kind": "invest", "method": "reinvest_choice", "y_on": 5.0,
             "y_eol": 15.0, "size": 100.0, "root": "invest:p2025"},
            {"entity": PLANT, "cohort": X, "kind": "extend", "method": "reinvest_choice", "y_on": 10.0,
             "y_eol": 20.0, "size": 500.0, "root": X, "parent": "existing:p2020"}]
    carried = pl.DataFrame([[r.get(c) for c in schema] for r in rows], schema=schema, orient="row")
    retire = pl.DataFrame([(PLANT, X, "p2035", 200.0)], schema={"entity": pl.Utf8, "cohort": pl.Utf8,
                                                                "period": pl.Utf8, "value": pl.Float64},
                          orient="row")
    cap = existing_capacity([PLANT], ["p2035"], {"p2035": 15.0}, existing, {(PLANT, "p2035"): 0.0}, carried,
                            retire, [PLANT])
    assert cap.all[(PLANT, "p2035")] == pytest.approx(300.0)
    assert cap.carried[(PLANT, "p2035")] == pytest.approx(0.0)
    assert cap.invested[PLANT] == pytest.approx(100.0)
    assert cap.retired.get((PLANT, "p2035"), 0.0) == 0.0


def test_a_cohort_whose_extension_was_realized_is_not_extended_again() -> None:
    """A carried investment cohort reaching its end of life is a parent (its size less its realized retirements);
    once an earlier solve realized its extension it is not."""
    from flextool.engine.preprocess.params._cohort_handoff import realized_cohort_schema
    from flextool.engine.preprocess.params._extend import _parent_rows
    schema = realized_cohort_schema()

    def frame(rows: list[dict]) -> pl.DataFrame:
        return pl.DataFrame([[r.get(c) for c in schema] for r in rows], schema=schema, orient="row")

    cohort = {"entity": PLANT, "cohort": "invest:p2025", "kind": "invest", "method": "reinvest_choice",
              "eol_method": "closest_seam", "y_on": 5.0, "y_life": 10.0, "y_eol": 15.0, "size": 100.0}
    own = pl.DataFrame(schema={"e": pl.Utf8, "cohort": pl.Utf8, "kind": pl.Utf8, "method": pl.Utf8,
                               "eol_method": pl.Utf8, "d_build": pl.Utf8, "y_on": pl.Float64, "y_life": pl.Float64,
                               "y_eol": pl.Float64})
    retire = pl.DataFrame([(PLANT, "invest:p2025", "p2030", 30.0)],
                          schema={"entity": pl.Utf8, "cohort": pl.Utf8, "period": pl.Utf8, "value": pl.Float64},
                          orient="row")
    parents = _parent_rows(own, frame([cohort]), retire, {PLANT})
    assert parents.select("cohort", "kind", "remaining").rows() == [("invest:p2025", "carried", 70.0)]
    extension = {"entity": PLANT, "cohort": "extend:p2035:invest:p2025", "kind": "extend", "parent": "invest:p2025",
                 "y_on": 15.0, "y_life": 5.0, "y_eol": 20.0, "size": 70.0}
    assert _parent_rows(own, frame([cohort, extension]), retire, {PLANT}).height == 0


#: ``stoch_two_period_hedge.json`` (p2035 at year 0, p2040 at year 1, one year each; social_discount_rate 5 %):
#: ``base`` has 50 MW existing with lifetime 1 (it ends at p2040), reinvest_choice, no investment, discount_rate 5 %,
#: life_extension_cost 0.5 CUR/kW (E = 500 CUR/MW) for 1 year.  The fan opens in p2035 (``ws``) and both periods are
#: shared investment periods: the extension in each branch's copy of p2040 is one decision.
#: The fan opens in p2040 (``rp``); under investments_after_reveal_period the reveal period's decisions are shared.
HEDGE_CHAIN = ["init", "base", "stoch", "rp"]


@pytest.mark.solver
def test_extension_inside_the_shared_window_is_the_same_in_every_branch(tmp_path_factory) -> None:
    """Each branch's copy of p2040 has the extension of the existing cohort, priced E·ann(1)·D(1)·(1 + r)^0 =
    500·1.05·1.05^-2 = 476.19 per MW, and the non-anticipativity row ties the two."""
    from tests.engine._unit_db import b64, build, cstr_over
    from tests.engine.test_retire_built_cohorts import HEDGE_JSON
    values = [["model", "flextool", "social_discount_rate", b64(0.05, "float")],
              ["model", "flextool", "discount_offset_investment", b64(1.0, "float")],
              ["unit", "base", "invest_method", b64("not_allowed", "str")],
              ["unit", "base", "lifetime_method", b64("reinvest_choice", "str")],
              ["unit", "base", "lifetime_technical", b64(1.0, "float")],
              ["unit", "base", "existing", b64(50.0, "float")],
              ["unit", "base", "discount_rate", b64(0.05, "float")],
              ["unit", "base", "life_extension_cost", b64(0.5, "float")],
              ["unit", "base", "life_extension_years", b64(1.0, "float")],
              ["solve", "stoch_2p_hedge", "invest_solve_branching_method",
               b64("investments_after_reveal_period", "str")]]
    steps, _, _ = _RUNS.get(tmp_path_factory, alt="lec_fan", values=values, base_chain=HEDGE_CHAIN,
                            fixture_json=HEDGE_JSON)
    step = [s for name, s in steps.items() if "realized" not in name][-1]
    cost = {d: v for e, c, d, v in utf8(step.flex_data.cd_extend_cost.frame).rows() if e == "base"}
    assert sorted(cost) == ["p2040_high", "p2040_low"]
    want = 500.0 * 1.05 * 1.05 ** -2
    assert want == pytest.approx(476.19, abs=0.01)
    assert all(v == pytest.approx(want, rel=1e-12) for v in cost.values())
    rows = cstr_over(build(step.flex_data), "non_anticipativity_extend_p")
    assert rows is not None and rows.select("p", "cohort", "cohort_b").rows() == [
        ("base", "existing:p2035", "existing:p2035")]
    extended = {d: v for p, c, d, v in utf8(step.solution.value("v_extend_p")).rows() if p == "base"}
    assert extended["p2040_high"] == pytest.approx(extended["p2040_low"], abs=1e-9)


def _lh2_extension(tmp_path_factory, alt: str, extra: list) -> dict:
    """``lh2_three_region`` with two 5-year periods (y2030, y2035) in the Benders invest solve and the dispatch
    solve: wind_A's 800 MW existing (reinvest_choice, lifetime 5) end at y2035 and can be extended 5 years for 10
    CUR/kW."""
    from tests.engine._unit_db import b64
    from tests.engine.test_retire_built_cohorts import LH2_JSON
    periods = ["y2030", "y2035"]

    def arr(x):
        return b64({"type": "array", "value_type": "str", "data": x, "index_name": "period"}, "array")

    def mp(d):
        return b64({"type": "map", "index_type": "str", "index_name": "period", "rank": 1,
                    "data": [[k, v] for k, v in d.items()]}, "map")

    values = [["solve", s, "period_timeset", mp(dict.fromkeys(periods, "week168"))]
              for s in ("lh2_invest", "lh2_dispatch")]
    values += [["solve", s, "years_represented", mp(dict.fromkeys(periods, 5.0))]
               for s in ("lh2_invest", "lh2_dispatch")]
    values += [["solve", "lh2_invest", "invest_periods", arr(periods)],
               ["solve", "lh2_invest", "realized_invest_periods", arr(periods)],
               ["solve", "lh2_invest", "realized_periods", arr(periods)],
               ["solve", "lh2_dispatch", "realized_periods", arr(periods)],
               ["unit", "wind_A", "lifetime_method", b64("reinvest_choice", "str")],
               ["unit", "wind_A", "lifetime_technical", b64(5.0, "float")],
               ["unit", "wind_A", "life_extension_cost", b64(10.0, "float")],
               ["unit", "wind_A", "life_extension_years", b64(5.0, "float")]] + extra
    steps, _, _ = _RUNS.get(tmp_path_factory, alt=alt, values=values,
                            base_chain=["lh2_three_region", "lh2_three_region_invest"], fixture_json=LH2_JSON)
    return steps


@pytest.mark.solver
def test_extensions_under_benders_carry_as_without_decomposition(tmp_path_factory) -> None:
    """The Benders invest solve extends wind_A at y2035 and hands the extensions to the dispatch solve, as the same
    invest solve without decomposition: the same objective, the same carried extension cohorts and the same
    capacity in service."""
    from tests.engine._unit_db import b64

    def summary(steps: dict) -> tuple:
        invest = next(s for name, s in steps.items() if "lh2_invest" in name)
        dispatch = next(s for name, s in steps.items() if "lh2_dispatch" in name)
        carried = {c: s for c, k, s in invest.handoff.realized_cohort.filter(pl.col("entity") == "wind_A")
                   .select("cohort", "kind", "size").rows() if k == "extend"}
        capacity = dict(utf8(dispatch.flex_data.p_entity_all_existing.frame).filter(pl.col("e") == "wind_A")
                        .select("d", "value").rows())
        return invest.is_benders, invest.obj, carried, capacity

    benders = summary(_lh2_extension(tmp_path_factory, "lec_benders", []))
    monolith = summary(_lh2_extension(tmp_path_factory, "lec_monolith",
                                      [["solve", "lh2_invest", "decomposition", b64("none", "str")]]))
    assert benders[0] and not monolith[0]
    assert benders[1] == pytest.approx(monolith[1], rel=1e-7)
    assert set(benders[2]) == set(monolith[2]) == {"extend:y2035:existing:y2030", "extend:y2035:invest:y2030"}
    assert benders[2]["extend:y2035:existing:y2030"] == pytest.approx(800.0, rel=1e-7)
    for c in benders[2]:
        assert benders[2][c] == pytest.approx(monolith[2][c], rel=1e-7), c
    assert benders[3] == pytest.approx(monolith[3], rel=1e-7)
