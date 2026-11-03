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
from tests.engine._unit_db import PLANT, PLANT_CHAIN, b64, build, make_db, run

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
