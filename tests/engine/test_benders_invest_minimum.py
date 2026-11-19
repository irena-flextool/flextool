"""Item 286: an entity investment or retirement floor under Benders decomposition.

Fixture ``lh2_three_region.json``, chain ``lh2_three_region_invest``: ``lh2_invest`` (one period y2030,
decomposition benders over the three ``benders_regional`` groups region_A / region_B / region_C), then the
monolithic ``lh2_dispatch`` of y2030.  wind_A lives in region_A (its node elec_A).  Each region subproblem
holds only its own entities' capacity columns, so the floor of wind_A belongs to region_A alone: in region_B and
region_C it has nothing to bound (before the fix it was emitted there as ``0 >= floor``, infeasible).

Oracles are relational: the Benders solve meets the floor and its objective equals the same solve without
decomposition (``decomposition none``) within the Benders convergence gap.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._unit_db import FLEXTOOL_ROOT, RunCache, b64, make_db, utf8
from tests.engine.test_chain_costs import _period_map

LH2_JSON = FLEXTOOL_ROOT / "tests" / "fixtures" / "lh2_three_region.json"
BASE_CHAIN = ["lh2_three_region", "lh2_three_region_invest"]
WIND = "wind_A"
MONOLITH = [["solve", "lh2_invest", "decomposition", b64("none", "str")]]

_RUNS = RunCache()


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()


def _invest_step(tmp_path_factory, alt: str, values: list):
    steps, _, _ = _RUNS.get(tmp_path_factory, alt=alt, values=values, base_chain=BASE_CHAIN, fixture_json=LH2_JSON)
    return next(s for name, s in steps.items() if "lh2_invest" in name)


def _twin(tmp_path_factory, alt: str, values: list):
    """``(benders, monolith)`` invest steps of the same data."""
    benders = _invest_step(tmp_path_factory, alt, values)
    monolith = _invest_step(tmp_path_factory, f"{alt}_mono", values + MONOLITH)
    assert benders.is_benders and not monolith.is_benders
    return benders, monolith


def _assert_same_objective(benders, monolith) -> None:
    """The Benders objective (its upper bound, a feasible plan) is not below the monolithic optimum and is within
    the convergence tolerance of it."""
    mono = monolith.obj
    assert benders.obj >= mono * (1.0 - 1e-9)
    assert (benders.obj - mono) <= benders.benders_tol * abs(mono)


def _invested(step, entity: str) -> float:
    cohorts = step.handoff.realized_cohort
    rows = cohorts.filter((pl.col("entity").cast(pl.Utf8) == entity) & (pl.col("kind").cast(pl.Utf8) == "invest"))
    return float(rows["size"].sum())


def _existing(steps: dict, entity: str) -> dict[str, float]:
    dispatch = next(s for name, s in steps.items() if "lh2_dispatch" in name)
    capacity = utf8(dispatch.flex_data.p_entity_all_existing.frame).filter(pl.col("e") == entity)
    return dict(zip(capacity["d"], capacity["value"]))


#: wind_A with no existing capacity and an invest_cost of 1e7 CUR/kW: one MW costs about 7e8 CUR a year (5 % over 25
#: years), far above what it can earn (at most every hour at the 8000 CUR/MWh shortage penalty times its ~0.4
#: capacity factor, about 3e7 CUR a year), so without a floor it builds nothing and with invest_min_period 100 MW in
#: y2030 it builds exactly the 100 MW.
INVEST_FLOOR = [["unit", WIND, "existing", b64(0.0, "float")],
                ["unit", WIND, "invest_method", b64("invest_period", "str")],
                ["unit", WIND, "invest_cost", b64(1.0e7, "float")],
                ["unit", WIND, "invest_max_period", _period_map({"y2030": 1000.0})],
                ["unit", WIND, "invest_min_period", _period_map({"y2030": 100.0})]]


@pytest.mark.solver
def test_an_invest_minimum_of_one_region_leaves_the_other_regions_feasible(tmp_path_factory) -> None:
    """region_B and region_C solve (no floor row of wind_A there); region_A builds the 100 MW floor of wind_A, and
    the objective equals the solve without decomposition within the Benders tolerance."""
    benders, monolith = _twin(tmp_path_factory, "r286_invest", INVEST_FLOOR)
    assert _invested(monolith, WIND) == pytest.approx(100.0, rel=1e-6)
    assert _invested(benders, WIND) == pytest.approx(100.0, rel=1e-6)
    _assert_same_objective(benders, monolith)


#: wind_A keeps its 800 MW of existing capacity and may retire up to all of it in y2030 (retire_period), at least
#: 100 MW (retire_min_period).  A retirement saves nothing here (no fixed cost, no salvage) and loses energy worth up
#: to the shortage penalty, so the solve retires exactly the 100 MW and the dispatch solve starts from 700 MW.
RETIRE_FLOOR = [["unit", WIND, "invest_method", b64("retire_period", "str")],
                ["unit", WIND, "retire_max_period", _period_map({"y2030": 800.0})],
                ["unit", WIND, "retire_min_period", _period_map({"y2030": 100.0})]]


@pytest.mark.solver
def test_a_retire_minimum_of_one_region_leaves_the_other_regions_feasible(tmp_path_factory) -> None:
    """As the investment floor: the Benders solve retires the 100 MW of wind_A (700 MW left for the dispatch solve,
    as without decomposition) and its objective equals the monolithic one within the tolerance."""
    steps, _, _ = _RUNS.get(tmp_path_factory, alt="r286_retire", values=RETIRE_FLOOR, base_chain=BASE_CHAIN,
                            fixture_json=LH2_JSON)
    mono_steps, _, _ = _RUNS.get(tmp_path_factory, alt="r286_retire_mono", values=RETIRE_FLOOR + MONOLITH,
                                 base_chain=BASE_CHAIN, fixture_json=LH2_JSON)
    benders = next(s for name, s in steps.items() if "lh2_invest" in name)
    monolith = next(s for name, s in mono_steps.items() if "lh2_invest" in name)
    assert benders.is_benders and not monolith.is_benders
    assert _existing(mono_steps, WIND) == pytest.approx({"y2030": 700.0}, rel=1e-9)
    assert _existing(steps, WIND) == pytest.approx({"y2030": 700.0}, rel=1e-9)
    _assert_same_objective(benders, monolith)


@pytest.mark.solver
def test_a_group_invest_minimum_straddling_regions_is_refused(tmp_path_factory) -> None:
    """A group invest_min_period over wind_A (region_A) and wind_B (region_B) cannot be split between the region
    subproblems (each would have to build the whole minimum from its own members): split refuses it."""
    values = [["group", "wind_AB", "invest_method", b64("invest_period", "str")],
              ["group", "wind_AB", "invest_min_period", _period_map({"y2030": 100.0})],
              ["unit", "wind_B", "invest_method", b64("invest_period", "str")],
              ["unit", WIND, "invest_method", b64("invest_period", "str")]]
    entities = [["group", "wind_AB", None], ["group__unit", ["wind_AB", WIND], None],
                ["group__unit", ["wind_AB", "wind_B"], None]]
    url, scen = make_db(tmp_path_factory, alt="r286_group", values=values, base_chain=BASE_CHAIN,
                        entities=entities, fixture_json=LH2_JSON)
    from flextool.engine import run_chain_from_db

    with pytest.raises(RuntimeError, match=r"investment limit group 'wind_AB' has entities in regions "
                                           r"region_A, region_B"):
        run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp("r286_group_w"))
