"""The age of existing capacity (``age_at_model_start``, or ``model.start_year − build_year``) in a solved model.

Built from ``tests/fixtures/tests.json`` (alternatives init, west, wind, coal; coal_plant 500 MW existing) on the
periods p2020, p2025, p2030 representing 5, 5 and 10 years (run years 0, 5, 10; seams {0, 5, 10, 20}; the helpers
of ``test_end_of_life_method``), and from ``stoch_two_period.json`` (thermal 200 MW existing; p2035 and p2040
given 5 and 10 years: seams {0, 5, 15}).  The existing cohort starts to operate at ``y_on = 0 − age`` and ends its
life at ``Y = y_on + lifetime_technical``, rounded by ``end_of_life_method``; it is in service while
``y[d] < y_eol``.  Expected values, by hand:

* coal_plant reinvest_choice, lifetime_technical 12, age 3: Y = −3 + 12 = 9 → closest / next 10 (500, 500, 0),
  previous 5 (500, 0, 0);
* build_year 2017 with model.start_year 2020 is age 2020 − 2017 = 3: the same LP as age 3; age 0 and build_year
  = start_year are the same LP as no age (next_seam: Y = 12 → 20, 500, 500, 500);
* previous_seam, Y = 9 → 5: the existing capacity is 500, 0, 0 in a single solve, in each solve of the chain
  (p2020 | p2025, p2030), in every roll;
* thermal, lifetime 12, age 3 under closest_seam on the seams {0, 5, 15}: Y = 9, 9 − 5 = 4 < 15 − 9 = 6 → 5:
  200 MW in p2035 (y 0) and 0 in p2040 (y 5), its branch periods alike (without the age Y = 12 → 15: 200 in both);
* no_reinvestment, lifetime_technical 8, age 3: investment closes at the end of the first life counted from the
  model start without the age (y ≥ 0 + 8): open in p2020, p2025, closed in p2030, the same as with no age and
  under every end_of_life_method; the existing capacity itself ends at −3 + 8 = 5 (next_seam): 500, 0, 0;
* age 13, lifetime 12: Y = −1, before the model start → the existing capacity is 0 in every period, a warning names
  coal_plant once over a two-solve chain; age 0, lifetime 3 under previous_seam: Y = 3 rounds to the model start 0,
  the same warning.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine.preprocess.params._run_calendar import RUN_EXISTING_COHORT_KEY
from tests.engine._unit_db import RunCache, b64
from tests.engine.test_construction_lead_time_calendar import _same_lp
from tests.engine.test_end_of_life_method import (
    CHAIN,
    PERIODS,
    PLANT,
    SOLVE,
    _map,
    capacity,
    coal,
    model_solves,
    solve_values,
)

pytestmark = pytest.mark.solver

STOCH_JSON = Path(__file__).resolve().parents[1] / "fixtures" / "stoch_two_period.json"
THERMAL = "thermal"

_RUNS = RunCache()


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()


def age(value: float, entity: str = PLANT) -> list:
    return ["unit", entity, "age_at_model_start", b64(value, "float")]


def built(year: float, start: float) -> list:
    return [["unit", PLANT, "build_year", b64(year, "float")], ["model", "flexTool", "start_year", b64(start, "float")]]


def steps_of(tmp_path_factory, alt: str, values: list, solves: list[str] = (SOLVE,)) -> dict:
    steps, _, _ = _RUNS.get(tmp_path_factory, alt=alt, values=[model_solves(list(solves))] + values,
                            base_chain=CHAIN, entities=[["solve", s, None] for s in solves])
    return steps


def only(steps: dict):
    (step,) = steps.values()
    return step


@pytest.mark.parametrize("method,want", [
    ("closest_seam", [500.0, 500.0, 0.0]),
    ("next_seam", [500.0, 500.0, 0.0]),
    ("previous_seam", [500.0, 0.0, 0.0]),
])
def test_aged_existing_capacity_by_end_of_life_method(tmp_path_factory, method, want) -> None:
    step = only(steps_of(tmp_path_factory, f"age3_{method}", solve_values() + coal(12.0, method) + [age(3.0)]))
    assert capacity(step) == dict(zip(PERIODS, want))
    run_cohort = step.flex_data_provider.get(RUN_EXISTING_COHORT_KEY)
    row = run_cohort.filter((pl.col("kind") == "existing") & (pl.col("e").cast(pl.Utf8) == PLANT))
    assert row.select("age", "y_on", "y_end", "y_first_life_end").rows() == [(3.0, -3.0, 9.0, 12.0)]


def test_build_year_is_the_age_at_the_model_start(tmp_path_factory) -> None:
    by_age = steps_of(tmp_path_factory, "age3_prev", solve_values() + coal(12.0, "previous_seam") + [age(3.0)])
    by_year = steps_of(tmp_path_factory, "built2017", solve_values() + coal(12.0, "previous_seam")
                       + built(2017.0, 2020.0))
    _same_lp(only(by_age), only(by_year))
    assert capacity(only(by_year)) == dict(zip(PERIODS, [500.0, 0.0, 0.0]))


@pytest.mark.parametrize("variant", ["age0", "built_at_start"])
def test_no_age_is_new_at_the_model_start(tmp_path_factory, variant) -> None:
    unset = only(steps_of(tmp_path_factory, "age_unset", solve_values() + coal(12.0, "next_seam")))
    extra = [age(0.0)] if variant == "age0" else built(2020.0, 2020.0)
    aged = only(steps_of(tmp_path_factory, f"age_{variant}", solve_values() + coal(12.0, "next_seam") + extra))
    _same_lp(unset, aged)
    assert capacity(aged) == dict(zip(PERIODS, [500.0, 500.0, 500.0]))


# ---------------------------------------------------------------------------
# The aged existing cohort ends in the same period in every solve of a run


AGED_PREVIOUS = coal(12.0, "previous_seam") + [age(3.0)]
WANT = dict(zip(PERIODS, [500.0, 0.0, 0.0]))


def _assert_every_solve(steps: dict) -> None:
    held = set()
    for name, step in steps.items():
        got = capacity(step)
        assert got, name
        for d, v in got.items():
            assert v == WANT[d], (name, d, v)
            held.add(d)
    assert held == set(PERIODS)


def test_chain_ends_the_aged_cohort_in_the_same_period(tmp_path_factory) -> None:
    values = (solve_values("life_a", ["p2020"], [5.0]) + solve_values("life_b", ["p2025", "p2030"], [5.0, 10.0])
              + AGED_PREVIOUS)
    steps = steps_of(tmp_path_factory, "age_chain", values, solves=["life_a", "life_b"])
    assert list(steps) == ["life_a", "life_b"]
    _assert_every_solve(steps)


def test_rolling_ends_the_aged_cohort_in_the_same_period(tmp_path_factory) -> None:
    values = solve_values("life_roll", mode="rolling_window") + [
        ["solve", "life_roll", "rolling_solve_horizon", b64(36.0, "float")],
        ["solve", "life_roll", "rolling_solve_jump", b64(24.0, "float")]] + AGED_PREVIOUS
    steps = steps_of(tmp_path_factory, "age_roll", values, solves=["life_roll"])
    assert len(steps) > len(PERIODS)
    _assert_every_solve(steps)


def test_stochastic_fan_ends_the_aged_cohort_by_the_period_year(tmp_path_factory) -> None:
    values = [["solve", "stoch_2p", "years_represented", _map({"p2035": 5.0, "p2040": 10.0})],
              ["unit", THERMAL, "lifetime_method", b64("reinvest_choice", "str")],
              ["unit", THERMAL, "lifetime_technical", b64(12.0, "float")],
              ["unit", THERMAL, "end_of_life_method", b64("closest_seam", "str")],
              age(3.0, THERMAL)]
    steps, _, _ = _RUNS.get(tmp_path_factory, alt="age_stoch", values=values, base_chain=["init", "base", "stoch"],
                            fixture_json=STOCH_JSON)
    for name, step in steps.items():
        pyd = step.flex_data_provider.get("solve_data/p_years_d")
        years = {str(d): float(y) for d, y in pyd.select(pyd.columns[0], pyd.columns[-1]).iter_rows()}
        got = capacity(step, THERMAL)
        assert len(got) > 2, (name, got)
        for d, v in got.items():
            assert v == (200.0 if years[d] < 5.0 else 0.0), (name, d, years[d], v)


# ---------------------------------------------------------------------------
# no_reinvestment: the investment window is the first life from the model start, without the age


@pytest.mark.parametrize("variant", ["age3_next", "age3_previous", "age3_closest", "age0"])
def test_no_reinvestment_window_does_not_age(tmp_path_factory, variant) -> None:
    method = {"age3_next": "next_seam", "age3_previous": "previous_seam", "age3_closest": "closest_seam"}.get(variant)
    values = solve_values(invest=PERIODS) + [
        ["unit", PLANT, "lifetime_method", b64("no_reinvestment", "str")],
        ["unit", PLANT, "lifetime_technical", b64(8.0, "float")],
        ["unit", PLANT, "payback_time", b64(8.0, "float")],
        ["unit", PLANT, "invest_method", b64("invest_period", "str")],
        ["unit", PLANT, "invest_cost", b64(1000.0, "float")],
        ["unit", PLANT, "invest_max_period", _map(dict.fromkeys(PERIODS, 100.0))]]
    if method is not None:
        values.append(["unit", PLANT, "end_of_life_method", b64(method, "str")])
    if variant != "age0":
        values.append(age(3.0))
    step = only(steps_of(tmp_path_factory, f"noreinv_{variant}", values))
    offered = {str(d) for e, d in step.flex_data.ed_invest_set.select("e", "d").iter_rows() if str(e) == PLANT}
    assert offered == {"p2020", "p2025"}
    if variant == "age3_next":
        assert capacity(step) == dict(zip(PERIODS, [500.0, 0.0, 0.0]))


# ---------------------------------------------------------------------------
# Existing capacity past its end of life at the model start


@pytest.mark.parametrize("alt,values,where", [
    ("age13", coal(12.0) + [age(13.0)],
     "coal_plant: the end of life (lifetime_technical 12 years and age_at_model_start 13 years) is at year -1 of "
     "the run, before the model start"),
    ("life3_prev", coal(3.0, "previous_seam"),
     "coal_plant: the end of life (lifetime_technical 3 years) is at year 3 of the run, which end_of_life_method "
     "previous_seam rounds to the model start"),
])
def test_existing_capacity_past_its_end_of_life_warns_once(tmp_path_factory, capsys, alt, values, where) -> None:
    capsys.readouterr()
    chain = solve_values("life_a", ["p2020"], [5.0]) + solve_values("life_b", ["p2025", "p2030"], [5.0, 10.0])
    steps = steps_of(tmp_path_factory, alt, chain + values, solves=["life_a", "life_b"])
    printed = capsys.readouterr().out
    assert printed.count("Warning: unit " + where) == 1, printed
    assert "its existing capacity is 0 in every period. Check the age or lifetime_technical." in printed
    for step in steps.values():
        assert set(capacity(step).values()) == {0.0}


def test_build_year_after_the_model_start_stops_the_run(tmp_path_factory) -> None:
    from flextool.engine.input._solve_state import FlexToolConfigError
    with pytest.raises(FlexToolConfigError) as err:
        steps_of(tmp_path_factory, "built_late", solve_values() + coal(12.0) + built(2021.0, 2020.0))
    assert ("unit 'coal_plant' build_year 2021 is after model.start_year 2020: capacity built after the model start "
            "is an investment") in str(err.value)


def test_a_map_age_is_refused(tmp_path_factory) -> None:
    """``age_at_model_start`` is a constant: a period map is refused by the parameter-shape check."""
    from flextool.engine import SpineDbReader
    from flextool.engine.input._param_shapes import resolve_param_shape
    from flextool.engine.input._solve_state import FlexToolConfigError
    from tests.engine._unit_db import make_db
    url, scen = make_db(tmp_path_factory, alt="age_map", base_chain=CHAIN,
                        values=[["unit", PLANT, "age_at_model_start", _map({"p2020": 3.0})]])
    reader = SpineDbReader(url, scen)
    with pytest.raises(FlexToolConfigError):
        resolve_param_shape(reader, "unit", "age_at_model_start")
