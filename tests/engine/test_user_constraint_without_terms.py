"""A user constraint whose terms all vanish in a solve.

``wind_plant`` has no ``invest_method`` in these scenarios, so a user constraint whose only term is its
``constraint_invested_capacity_coeff`` has no variable in the solve: each of its rows would read
``0 sense constant``.

* When that cannot hold (0 >= 600, 0 = 3, 0 <= -1, or one period of a period-map constant at 5 with >=), the
  build stops with a configuration error naming the constraint, the solve and the period.
* When it always holds (0 >= 0, 0 <= 5), the constraint gets no row and one warning (W-0004) per run names it.
* The outcome does not depend on other constraints: ``c_other`` (``coal_plant`` invested coefficient, a variable
  in every period) is added beside it in the parametrised twin.

Built from ``tests/fixtures/tests.json`` (never a checked-in ``.sqlite``).
"""
from __future__ import annotations

import logging

import polars as pl
import pytest

from flextool import run_warnings

from flextool.engine.input._solve_state import FlexToolConfigError
from tests.engine._unit_db import b64, build, make_db, run

SINGLE = ["init", "west", "wind", "coal", "coal_invest", "battery", "battery_invest", "5weeks",
          "invest_solveSequence_5weeks", "invest_allYears_5weeks"]
CHAIN = ["init", "west", "wind", "coal", "coal_invest", "battery", "battery_invest", "5weeks",
         "invest_solveSequence_5weeks"]
SINGLE_SOLVE = "y2020_2035_5week"
FAMILIES = [f"process_constraint_{s}{p}" for s in ("equal", "less_than", "greater_than") for p in ("", "_period")]


def _coef(cns: list[str]) -> list:
    return b64({"index_type": "str", "index_name": "constraint", "data": [[cn, 1.0] for cn in cns]}, "map")


def _values(sense: str, constant, *, other: bool) -> tuple[list, list]:
    entities = [["constraint", "c_chk", None]]
    values = [["constraint", "c_chk", "sense", b64(sense, "str")],
              ["constraint", "c_chk", "constant", constant],
              ["unit", "wind_plant", "constraint_invested_capacity_coeff", _coef(["c_chk"])]]
    if other:
        entities.append(["constraint", "c_other", None])
        values += [["constraint", "c_other", "sense", b64("greater_than", "str")],
                   ["constraint", "c_other", "constant", b64(0.0, "float")],
                   ["unit", "coal_plant", "constraint_invested_capacity_coeff", _coef(["c_other"])]]
    return entities, values


def _rows_of(pb, cn: str) -> int:
    total = 0
    for family in FAMILIES:
        for rec in pb.cstrs_named(family):
            if rec.name == family and rec.over is not None:
                total += rec.over.filter(pl.col("cn").cast(pl.Utf8) == cn).height
    return total


def _period_map(values: dict[str, float]) -> list:
    return b64({"index_type": "str", "index_name": "period", "data": [[k, v] for k, v in values.items()]}, "map")


@pytest.mark.parametrize("other", [False, True], ids=["alone", "beside_other_constraint"])
@pytest.mark.parametrize("sense, constant, shown", [
    ("greater_than", 600.0, "0 >= 600"),
    ("equal", 3.0, "0 = 3"),
    ("less_than", -1.0, "0 <= -1"),
])
def test_vanished_constraint_that_cannot_hold_is_an_error(tmp_path_factory, other, sense, constant, shown):
    entities, values = _values(sense, b64(constant, "float"), other=other)
    url, scen = make_db(tmp_path_factory, alt=f"uc_gone_{sense}", values=values, base_chain=SINGLE,
                        entities=entities)
    with pytest.raises(FlexToolConfigError) as err:
        run(tmp_path_factory, url, scen)
    text = str(err.value)
    assert "c_chk" in text and f"'{SINGLE_SOLVE}'" in text and shown in text, text
    assert "c_other" not in text, text


def test_vanished_constraint_with_a_period_map_constant_names_the_failing_period(tmp_path_factory):
    constant = _period_map({"p2020": 0.0, "p2025": 0.0, "p2030": 5.0, "p2035": 0.0})
    entities, values = _values("greater_than", constant, other=True)
    url, scen = make_db(tmp_path_factory, alt="uc_gone_period_map", values=values, base_chain=SINGLE,
                        entities=entities)
    with pytest.raises(FlexToolConfigError) as err:
        run(tmp_path_factory, url, scen)
    text = str(err.value)
    assert "c_chk (period p2030: 0 >= 5)" in text, text
    assert "p2020" not in text and "p2025" not in text and "p2035" not in text, text


@pytest.mark.parametrize("other", [False, True], ids=["alone", "beside_other_constraint"])
@pytest.mark.parametrize("sense, constant", [("greater_than", 0.0), ("less_than", 5.0)])
def test_vanished_constraint_that_always_holds_warns_once_and_has_no_row(tmp_path_factory, caplog, other, sense,
                                                                         constant):
    entities, values = _values(sense, b64(constant, "float"), other=other)
    url, scen = make_db(tmp_path_factory, alt=f"uc_idle_{sense}", values=values, base_chain=CHAIN,
                        entities=entities)
    caplog.set_level(logging.DEBUG, logger=run_warnings.LOGGER_NAME)
    steps = run(tmp_path_factory, url, scen)
    records = [r for r in caplog.records if r.name == run_warnings.LOGGER_NAME]
    lines = [r.getMessage() for r in records if r.levelno == logging.WARNING and "W-0004" in r.getMessage()]
    assert len(steps) > 1
    # One coded line in the run (each constraint once per run, however many solves leave it without variables).
    assert len(lines) == 1, lines
    assert lines[0].startswith("Warning: user constraint has no variable in solve '"), lines
    assert lines[0].endswith("[W-0004]"), lines
    # The fixture's other constraints are idle too: the line names the first, the debug log every one.
    debug = [r.getMessage() for r in records if r.levelno == logging.DEBUG]
    assert any(m.startswith("W-0004 all cases") and "c_chk" in m.split(": ", 1)[1].split("; ") for m in debug), debug
    assert sum(m.startswith("W-0004 c_chk: period ") for m in debug) == 1, debug
    for step in steps.values():
        pb = build(step.flex_data)
        assert _rows_of(pb, "c_chk") == 0
        if other:
            assert _rows_of(pb, "c_other") > 0
