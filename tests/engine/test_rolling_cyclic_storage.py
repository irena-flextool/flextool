"""Cyclic storage binding methods in the rolls of a rolling solve (#119).

*bind_within_solve* continues each roll from the carried level, like a bind_forward_only node, and ends the horizon
at the instance's start level in the closing rolls (tests/engine/test_rolling_storage_anchor.py has the rest).
*bind_within_timeblock* and *bind_within_period* do the same per timeblock / period where a roll starts inside one
(tests/engine/test_rolling_storage_scopes.py).  Before #119 the roll carry was added on top of the wrap (−2·su on
v_state[first]); after it, until the per-scope carry, each roll cycled on its own part.
The battery's self-discharge is set to 0 so the coefficients are exactly ±su.

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


@pytest.mark.parametrize("method", ["bind_within_timeblock", "bind_within_period"])
def test_cyclic_roll_continues_its_timeblock_or_period(tmp_path_factory, method):
    """Rolls 1 and 2 start inside the timeblock / period roll 0 began (a roll boundary inside the cycle scope): they
    continue it from the carry, as bind_within_solve does, with no wrap of their first step, and end it at the level
    roll 0 started it from (its fix_start pin, 0.1 × 50 = 5 MWh) where they reach the end of the 12-h solve.  Before
    (#119), each roll cycled on its own part: its first-step row held the wrap and no carry."""
    from tests.engine._lp_rows import row_bounds as bounds_of

    steps = _rolls(tmp_path_factory, f"c119{method[5:12]}", method)
    for step in steps[1:3]:
        fd, (d0, t0), state, bounds = _first_row(step)
        assert fd.p_nested_solve_first is not None and not bool(fd.p_nested_solve_first), "not a continuation roll"
        _carried(fd)
        carry = _carry(fd)
        assert state == {f"v_state[battery,{d0},{t0}]": pytest.approx(-SU, rel=1e-12)}
        assert bounds[0] == pytest.approx(-carry, rel=1e-9) and bounds[1] == pytest.approx(-carry, rel=1e-9)
        close = bounds_of(build(fd), "storage_state_roll_close", "battery,")
        d1, t1 = utf8(fd.dt).sort("d", "t").row(-1)
        assert close == {f"storage_state_roll_close[battery,{d1},{t1}]": (pytest.approx(5.0), pytest.approx(5.0))}


def test_forward_only_roll_keeps_carry(tmp_path_factory):
    steps = _rolls(tmp_path_factory, "c119fo", "bind_forward_only")
    for step in steps[1:3]:
        fd, (d0, t0), state, bounds = _first_row(step)
        _carried(fd)
        carry = _carry(fd)
        # the first step of the roll has no state change of its own: only the carried start
        assert state == {f"v_state[battery,{d0},{t0}]": pytest.approx(-SU, rel=1e-12)}
        assert bounds[0] == pytest.approx(-carry, rel=1e-9) and bounds[1] == pytest.approx(-carry, rel=1e-9)


def test_within_solve_roll_continues_from_carry(tmp_path_factory):
    """bind_within_solve: rolls 1 and 2 (both reach t0012, the end of the 12-h solve) start from the carry, as a
    bind_forward_only node, and end at the level roll 0 started from: its fix_start pin, 0.1 × 50 = 5 MWh."""
    from tests.engine._lp_rows import row_bounds as bounds_of

    steps = _rolls(tmp_path_factory, "c119ws", "bind_within_solve")
    for step in steps[1:3]:
        fd, (d0, t0), state, bounds = _first_row(step)
        _carried(fd)
        carry = _carry(fd)
        assert state == {f"v_state[battery,{d0},{t0}]": pytest.approx(-SU, rel=1e-12)}
        assert bounds[0] == pytest.approx(-carry, rel=1e-9) and bounds[1] == pytest.approx(-carry, rel=1e-9)
        close = bounds_of(build(fd), "storage_state_roll_close", "battery,")
        d1, t1 = utf8(fd.dt).sort("d", "t").row(-1)
        assert close == {f"storage_state_roll_close[battery,{d1},{t1}]": (pytest.approx(5.0), pytest.approx(5.0))}
