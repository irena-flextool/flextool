"""Cyclic storage binding methods in the rolls of a rolling solve take no roll carry (#119).

how_to/solve_sequences.md (rolling notes): *bind_within_timeblock* "will bind the first step of the roll (whatever it is) to the
end of the timeblock", the same for *bind_within_period*, and *bind_within_solve* "binds the start state to the end
state of each roll".  So a cyclic node cycles within each roll: its first-step row holds the wrap
(v_state[wrap] − v_state[first])·su and nothing else.  Before the fix the roll carry (−v_state[first]·su +
roll_continue_state) was added on top: −2·su on v_state[first] and the carry in the bounds.  A bind_forward_only
node keeps the carry.  The battery's self-discharge is set to 0 so the coefficients are exactly ±su.

Database: ``tests/fixtures/tests.json`` + an override alternative (never a checked-in sqlite).
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._lp_rows import row_bounds, row_terms
from tests.engine._unit_db import b64, build, make_db, run, time_map, utf8

CHAIN = ["init", "west", "wind", "coal", "battery", "fullYear", "dispatch_fullYear_roll"]
SU = 50.0          # battery existing 50 MWh, no virtual_unitsize → state_unitsize = existing


def _rolls(tmp_path_factory, alt: str, method: str) -> list:
    values = [["node", "battery", "storage_binding_method", b64(method, "str")],
              ["node", "battery", "self_discharge_loss", b64(0.0, "float")],
              # 40 MWh into the battery at the last realized step of rolls 0 and 1: the 10 MW inverter cannot take
              # it out within the step, so the state handed to rolls 1 and 2 is ≥ 30 MWh.
              ["node", "battery", "inflow", time_map({"t0004": 40.0, "t0008": 40.0}, 0.0)],
              ["solve", "dispatch_fullYear_roll", "rolling_duration", b64(12.0, "float")]]
    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=CHAIN)
    steps = list(run(tmp_path_factory, url, scen).values())
    assert len(steps) >= 3, [s.solve_name for s in steps]
    return steps


def _first_row(step):
    fd = step.flex_data
    d0, t0 = utf8(fd.dt).sort("d", "t").row(0)
    name = f"nodeBalance_eq[battery,{d0},{t0}]"
    pb = build(fd)
    terms = row_terms(pb, "nodeBalance_eq", "battery,")[name]
    bounds = row_bounds(pb, "nodeBalance_eq", "battery,")[name]
    state = {c: v for c, v in terms.items() if c.startswith("v_state[")}
    return fd, (d0, t0), state, bounds


def _carry(fd) -> float:
    frame = utf8(fd.p_roll_continue_state.frame).filter(pl.col("n") == "battery")
    assert frame.height == 1
    return float(frame["value"][0])


def _carried(fd) -> None:
    assert _carry(fd) > 25.0, "the carry must be non-zero for the test to be meaningful"


@pytest.mark.parametrize("method", ["bind_within_timeblock", "bind_within_period", "bind_within_solve"])
def test_cyclic_roll_first_step_has_no_carry(tmp_path_factory, method):
    steps = _rolls(tmp_path_factory, f"c119{method[5:12]}", method)
    for step in steps[1:3]:
        fd, (d0, t0), state, bounds = _first_row(step)
        assert fd.p_nested_solve_first is not None and not bool(fd.p_nested_solve_first), "not a continuation roll"
        _carried(fd)
        first = f"v_state[battery,{d0},{t0}]"
        assert state.pop(first) == pytest.approx(-SU, rel=1e-12)
        # the wrap: the roll's own last step (of the timeblock / period / solve)
        assert len(state) == 1, state
        (wrap, coef), = state.items()
        assert coef == pytest.approx(SU, rel=1e-12)
        d1, t1 = utf8(fd.dt).sort("d", "t").row(-1)
        assert wrap == f"v_state[battery,{d1},{t1}]"   # the roll holds one period of one timeblock
        # no carry constant: the battery has no inflow at the roll's first step, so the row is = 0
        assert bounds == (pytest.approx(0.0, abs=1e-12), pytest.approx(0.0, abs=1e-12))


def test_forward_only_roll_keeps_carry(tmp_path_factory):
    steps = _rolls(tmp_path_factory, "c119fo", "bind_forward_only")
    for step in steps[1:3]:
        fd, (d0, t0), state, bounds = _first_row(step)
        _carried(fd)
        carry = _carry(fd)
        # the first step of the roll has no state change of its own: only the carried start
        assert state == {f"v_state[battery,{d0},{t0}]": pytest.approx(-SU, rel=1e-12)}
        assert bounds[0] == pytest.approx(-carry, rel=1e-9) and bounds[1] == pytest.approx(-carry, rel=1e-9)
