"""User-constraint ``constant`` as a period map (database version 77).

A period map that gives every period of the solve the same value as a
constant builds the same user-constraint rows; another constraint of the same
model keeps its constant (here the spread default 0).  A map that misses a
period of the solve is an error naming the constraint.  Fixture: ``tests.json``
(init / west / coal: ``coal_plant`` into ``west``, one period ``p2020``) plus two
``less_than`` constraints on the plant's output flow.  Built from JSON.
"""
from __future__ import annotations

import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from tests.engine._lp_rows import row_bounds, row_terms
from tests.engine._unit_db import FLEXTOOL_ROOT, PLANT, PLANT_CHAIN, b64, build, make_db, run

ENTS = [["constraint", "c_cap", None], ["constraint", "c_other", None]]


def _values(constant) -> list:
    return [["constraint", "c_cap", "sense", b64("less_than", "str")],
            ["constraint", "c_other", "sense", b64("less_than", "str")],
            ["constraint", "c_cap", "constant", constant],
            ["unit__outputNode", [PLANT, "west"], "constraint_flow_coeff",
             b64({"index_type": "str", "index_name": "constraint", "data": [["c_cap", 1.0], ["c_other", 1.0]]},
                 "map")]]


def _rows(tmp_path_factory, alt: str, constant) -> dict[str, dict]:
    url, scen = make_db(tmp_path_factory, alt=alt, values=_values(constant), base_chain=PLANT_CHAIN,
                        entities=ENTS)
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    pb = build(step.flex_data)
    return {"terms": row_terms(pb, "process_constraint_less_than"),
            "bounds": row_bounds(pb, "process_constraint_less_than")}


def test_period_map_equals_constant(tmp_path_factory) -> None:
    scalar = _rows(tmp_path_factory, "uc_const_scalar", b64(300.0, "float"))
    mapped = _rows(tmp_path_factory, "uc_const_map",
                   b64({"index_type": "str", "index_name": "period", "data": [["p2020", 300.0]]}, "map"))
    rows = scalar["bounds"]
    assert any(r.startswith("process_constraint_less_than[c_cap,") for r in rows)
    assert any(r.startswith("process_constraint_less_than[c_other,") for r in rows)
    assert {r: b[1] for r, b in rows.items() if "c_cap," in r} == {r: 300.0 for r in rows if "c_cap," in r}
    assert mapped == scalar


def test_period_map_missing_a_period_is_an_error(tmp_path_factory) -> None:
    with pytest.raises(FlexToolConfigError, match="c_cap.*p2020"):
        _rows(tmp_path_factory, "uc_const_gap",
              b64({"index_type": "str", "index_name": "period", "data": [["p2030", 300.0]]}, "map"))


ROLL_CHAIN = ["init", "west", "battery", "wind", "fullYear", "dispatch_fullYear_roll"]
ROLL_PERIODS = ["p2020", "p2025", "p2030", "p2035"]
STOCH_JSON = FLEXTOOL_ROOT / "tests" / "fixtures" / "stochastics.json"


def _period_map(values: dict[str, float]) -> list:
    return b64({"index_type": "str", "index_name": "period", "data": [[k, v] for k, v in values.items()]}, "map")


def _rows_per_step(tmp_path_factory, alt: str, constant, *, chain, arc, fixture_json=None) -> list[dict]:
    values = [["constraint", "c_cap", "sense", b64("less_than", "str")],
              ["constraint", "c_other", "sense", b64("less_than", "str")],
              ["constraint", "c_cap", "constant", constant],
              ["unit__outputNode", arc, "constraint_flow_coeff",
               b64({"index_type": "str", "index_name": "constraint", "data": [["c_cap", 1.0], ["c_other", 1.0]]},
                   "map")]]
    kwargs = {} if fixture_json is None else {"fixture_json": fixture_json}
    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=chain, entities=ENTS, **kwargs)
    out = []
    for step in run(tmp_path_factory, url, scen).values():
        pb = build(step.flex_data)
        out.append({"terms": row_terms(pb, "process_constraint_less_than"),
                    "bounds": row_bounds(pb, "process_constraint_less_than")})
    return out


@pytest.mark.parametrize(("case", "chain", "arc", "periods", "fixture"), [
    ("rolling", ROLL_CHAIN, ["wind_plant", "west"], ROLL_PERIODS, None),
    ("stochastic", ["base", "init", "system", "2day"], ["coal_plant", "demand_node"], ["period1"], STOCH_JSON),
    ("rolling_stochastic", ["init", "base", "system", "1week_rolling"], ["coal_plant", "demand_node"],
     ["period1"], STOCH_JSON),
])
def test_period_map_equals_constant_in_every_solve(tmp_path_factory, case, chain, arc, periods, fixture) -> None:
    """One constraint with a period map and one with the spread default 0 in the same model: the map that
    repeats the constant gives every roll / branch the same rows as the constant."""
    scalar = _rows_per_step(tmp_path_factory, f"uc_c_{case}_s", b64(300.0, "float"), chain=chain, arc=arc,
                            fixture_json=fixture)
    mapped = _rows_per_step(tmp_path_factory, f"uc_c_{case}_m", _period_map({p: 300.0 for p in periods}),
                            chain=chain, arc=arc, fixture_json=fixture)
    assert len(scalar) == len(mapped) and scalar
    for s, m in zip(scalar, mapped):
        assert any("[c_cap," in r for r in s["bounds"]) and any("[c_other," in r for r in s["bounds"])
        assert {r: b[1] for r, b in s["bounds"].items() if "[c_other," in r} == {
            r: 0.0 for r in s["bounds"] if "[c_other," in r}
        assert m == s


def test_period_map_gives_each_period_its_value_in_rolling(tmp_path_factory) -> None:
    """Every roll's c_cap row of period d has the map's value for d (300, 310, 320, 330)."""
    values = dict(zip(ROLL_PERIODS, (300.0, 310.0, 320.0, 330.0)))
    steps = _rows_per_step(tmp_path_factory, "uc_c_roll_values", _period_map(values), chain=ROLL_CHAIN,
                           arc=["wind_plant", "west"])
    seen = set()
    for rows in steps:
        for r, (_, upper) in rows["bounds"].items():
            if "[c_cap," in r:
                d = r[r.index("[") + 1:-1].split(",")[1]
                seen.add(d)
                assert upper == values[d], r
    assert seen == set(ROLL_PERIODS)
