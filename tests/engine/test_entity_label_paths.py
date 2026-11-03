"""Every entity's map is checked against the shapes its parameter takes; unnamed maps of different depths are
read per depth.

* A value whose depth the parameter does not take, or whose level names (``period`` / ``time``) are on the
  wrong level, raises ``FlexToolConfigError`` naming the entity, the parameter and the depth.  Before, such a
  value was dropped without a message (``commodity.price`` as a 2-level map made coal free).
* When the entities of one parameter have unnamed maps of different depths, each depth is read on its own with
  the usual rules.  Before, the reader put every row into the columns of the deepest map, so a one-level time
  map next to a period x time map lost its values.  A note says once per run how such maps were read.

Each read case is compared with its twin whose levels are named.  Fixtures: ``tests.json`` + an override
alternative, built from JSON (production reader, no axis Enums).
"""
from __future__ import annotations

import pytest

from flextool.engine.input._param_shapes import _SHAPE_LABELS, Shape, shape_rank
from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.spinedb_backend._axis_enums import FlexDataIntegrityError
from tests._support.fields import field_diff
from tests.engine._unit_db import CHP, PLANT, PLANT_CHAIN, TIMESTEPS, b64, make_db, run, solve
from tests.engine.test_ladder_by_shape import TESTS_JSON, _lps, _tier

CHAIN = ["init", "west", "coal_chp", "heat", "coal"]


def map1(label: str, values: dict) -> list:
    return b64({"index_type": "str", "index_name": label, "data": [[k, v] for k, v in values.items()]}, "map")


def map2(outer: str, inner: str, values: dict) -> list:
    return b64({"index_type": "str", "index_name": outer, "data": [
        [k, {"type": "map", "index_type": "str", "index_name": inner, "data": [[t, v] for t, v in sub.items()]}]
        for k, sub in values.items()]}, "map")


def test_shape_rank_is_the_number_of_index_levels() -> None:
    assert {s: shape_rank(s) for s in Shape} == {s: len(_SHAPE_LABELS[s]) for s in Shape}
    assert (shape_rank(Shape.SCALAR), shape_rank(Shape.MAP_TIME), shape_rank(Shape.MAP_PERIOD_TIME),
            shape_rank(Shape.MAP_PERIOD_TIER_FACET)) == (0, 1, 2, 3)


def test_two_level_commodity_price_is_an_error(tmp_path_factory) -> None:
    """``commodity.price`` takes a constant or a 1-level map; an unnamed 2-level map is an error naming coal.
    Before: dropped without a message (objective 9.30e8, coal free)."""
    url, scen = make_db(tmp_path_factory, alt="lp_price_d2", base_chain=PLANT_CHAIN,
                        values=[["commodity", "coal", "price", map2("x", "x", {"p2020": {"t0001": 99.0}})]])
    with pytest.raises(FlexToolConfigError) as err:
        run(tmp_path_factory, url, scen)
    text = str(err.value)
    assert "commodity 'coal', parameter 'price': the value has 2 index levels" in text
    assert "a constant, or a map over periods or over time (1 level)" in text


def test_two_level_period_parameter_is_an_error(tmp_path_factory) -> None:
    """``unit.invest_max_period`` (constant or period map) as an unnamed 2-level map is an error naming the
    unit.  Before: dropped without a message."""
    url, scen = make_db(tmp_path_factory, alt="lp_imp_d2", base_chain=PLANT_CHAIN,
                        values=[["unit", PLANT, "invest_max_period", map2("x", "x", {"p2020": {"a": 1.0}})]])
    with pytest.raises(FlexToolConfigError, match=r"unit 'coal_plant', parameter 'invest_max_period': the value "
                                                  r"has 2 index levels"):
        run(tmp_path_factory, url, scen)


def test_period_name_on_the_tier_level_of_a_ladder_is_an_error(tmp_path_factory) -> None:
    """A 2-level annual ladder whose tier level is named ``period`` is an error naming coal.  Before, the
    reader trimmed its rows as unknown periods."""
    import json

    from flextool.engine import run_chain_from_db
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    ladder = {"index_type": "str", "rank": 2, "index_name": "period",
              "data": [["1", _tier(20.0, 100000.0)], ["2", _tier(30.0, float("inf"))]]}
    data = json.loads(TESTS_JSON.read_text())
    data["parameter_values"] = [
        v for v in data["parameter_values"]
        if not (v[0] == "commodity" and v[1] == "coal" and v[2] == "price_ladder_annual")]
    data = _lps()._overlay(data, name="lp_ladder_period", base="coal_ladder_annual",
                           values=[["commodity", "coal", "price_ladder_annual", ladder]])
    work = tmp_path_factory.mktemp("lp_ladder_period")
    (work / "f.json").write_text(json.dumps(data))
    url = json_to_db(work / "f.json", work / "f.sqlite")
    migrate_database(url)
    with pytest.raises(FlexToolConfigError, match=r"commodity 'coal', parameter 'price_ladder_annual': level 1 "
                                                  r"of the value is named 'period'"):
        run_chain_from_db(url, "lp_ladder_period", work_folder=work / "w", warm=False)


_PT = {"p2020": {t: (0.3 if i % 2 else 0.6) for i, t in enumerate(TIMESTEPS)}}


def _availability_case(tmp_path_factory, alt: str, plant_labels: tuple[str, str], chp_value: list):
    return solve(tmp_path_factory, alt=alt, base_chain=CHAIN,
                 values=[["unit", PLANT, "availability", map2(*plant_labels, _PT)],
                         ["unit", CHP, "availability", chp_value]])


def test_unnamed_time_map_next_to_period_time_map(tmp_path_factory, capsys) -> None:
    """coal_plant: unnamed period x time availability; coal_chp: unnamed 1-level map over the time steps.
    ≡ the twin with named levels (objective, ``p_process_availability``).  Before: objective 8.68e8 (coal_chp's
    time steps read as periods) instead of 1.73e9.  The note naming both units is printed once."""
    chp = {t: 0.5 for t in TIMESTEPS}
    capsys.readouterr()
    case = _availability_case(tmp_path_factory, "lp_mix_t", ("x", "x"), map1("x", chp))
    notes = [ln for ln in capsys.readouterr().out.splitlines()
             if ln.startswith("Note: unit 'availability' has maps without index names")]
    twin = _availability_case(tmp_path_factory, "lp_mix_t_twin", ("period", "time"), map1("time", chp))
    assert case.solution.obj == pytest.approx(twin.solution.obj, rel=1e-9)
    assert field_diff(case.flex_data.p_process_availability, twin.flex_data.p_process_availability) is None
    assert len(notes) == 1
    assert "'coal_chp' over time" in notes[0] and "'coal_plant' over periods and time" in notes[0]


def test_unnamed_period_map_next_to_period_time_map(tmp_path_factory) -> None:
    """GUARD (read correctly before too): the same with coal_chp as an unnamed 1-level period map."""
    case = _availability_case(tmp_path_factory, "lp_mix_p", ("x", "x"), map1("x", {"p2020": 0.5}))
    twin = _availability_case(tmp_path_factory, "lp_mix_p_twin", ("period", "time"),
                              map1("period", {"p2020": 0.5}))
    assert case.solution.obj == pytest.approx(twin.solution.obj, rel=1e-9)
    assert field_diff(case.flex_data.p_process_availability, twin.flex_data.p_process_availability) is None


def test_one_level_map_keyed_by_periods_and_time_steps_is_an_error(tmp_path_factory) -> None:
    """An unnamed 1-level map whose keys are both a period and time steps, next to a 2-level map, cannot be
    placed: FlexDataIntegrityError naming coal_chp."""
    both = {"p2020": 0.5, **{t: 0.5 for t in TIMESTEPS}}
    url, scen = make_db(tmp_path_factory, alt="lp_mix_both", base_chain=CHAIN,
                        values=[["unit", PLANT, "availability", map2("x", "x", _PT)],
                                ["unit", CHP, "availability", map1("x", both)]])
    with pytest.raises(FlexDataIntegrityError, match="coal_chp"):
        run(tmp_path_factory, url, scen)



@pytest.mark.parametrize("named", [False, True], ids=["unnamed", "named"])
def test_upgrade_marks_unnamed_mixed_depths_and_runs_name_the_time_maps(tmp_path_factory, capsys, named) -> None:
    """The database upgrade marks a database with unnamed maps of one and of two levels in one parameter (not
    one whose levels are named); with the marker, a run prints the notice naming coal_chp, whose one-level map
    is now read over time."""
    from spinedb_api import DatabaseMapping

    from flextool.update_flextool import db_migration as m

    labels = ("period", "time") if named else ("x", "x")
    chp = map1("time" if named else "x", {t: 0.5 for t in TIMESTEPS})
    alt = f"lp_mark_{'named' if named else 'unnamed'}"
    url, scen = make_db(tmp_path_factory, alt=alt, base_chain=CHAIN,
                        values=[["unit", PLANT, "availability", map2(*labels, _PT)],
                                ["unit", CHP, "availability", chp]])
    with DatabaseMapping(url) as db:
        m._v77_unnamed_mixed_depth_marker(db)
        names = {md["name"] for md in db.get_items("metadata")}
        if not named:
            db.commit_session("marker")
    assert (m.BEHAVIOUR_CHANGE_MARKER_PREFIX + "unnamed_maps_read_per_depth" in names) is (not named)
    capsys.readouterr()
    run(tmp_path_factory, url, scen)
    notices = [ln for ln in capsys.readouterr().out.splitlines() if "are now read per depth" in ln]
    assert len(notices) == (0 if named else 1)
    if not named:
        assert "Affected unit availability values: coal_chp." in notices[0]
