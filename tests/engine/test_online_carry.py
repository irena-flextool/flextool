"""Pure-function tests for the online-commitment roll carry.

The step(s) of a continuation solve that take the previous solve's realized commitment (``p_roll_continue_online``)
as their online predecessor are the ramp carry's seam roots (``dt_online_carry`` = the steps of ``dt_ramp_carry``),
so the commitment stops at a gap, a new year and a representative-period start, as ramp limits and the minimum up /
down look-back do.  ``check_online_ramp_carry_consistency`` raises instead of letting the online carry and the
ramp carry / ramp link silently disagree.
"""
from __future__ import annotations

import pytest
import polars as pl

from flextool.engine.preprocess.params._ramp_links import check_online_ramp_carry_consistency, ramp_roll_carry


def _dt(rows: list[tuple[str, str]]) -> pl.DataFrame:
    return pl.DataFrame({"d": [r[0] for r in rows], "t": [r[1] for r in rows]})


def _carried_step(d: str, t: str) -> pl.DataFrame:
    return pl.DataFrame({"period": [d], "step": [t], "duration": [1.0]})


def _link(rows: list[tuple[str, str, str, str, int]]) -> pl.DataFrame:
    """``rows``: (period, time, previous_period, previous_within_solve, jump)."""
    return pl.DataFrame({
        "period": [r[0] for r in rows], "time": [r[1] for r in rows],
        "previous_period": [r[2] for r in rows], "previous_within_solve": [r[3] for r in rows],
        "jump": [str(r[4]) for r in rows]})


# ---------------------------------------------------------------------------
# The online carry steps are the ramp carry steps (``ramp_roll_carry``); the loader-level seam cases (adjacent,
# gap, new year, representative-period start, a carry without its step) are in
# ``tests/engine/loaders/test_a04_a05_online_ramp.py``.


def test_every_forecast_branch_starting_at_the_carried_step_takes_the_carry() -> None:
    """A fan seam: the previous solve realized (p1, t0004); this solve's period p1 is a period label of forecast
    branches b1 and b2 that both start at t0005, and the complete solve's predecessor of p1's t0005 is the carried
    step (jump 1).  Both branches continue the carried commitment; period p2, whose first step follows another
    step, does not."""
    dt = _dt([("b1", "t0005"), ("b1", "t0006"), ("b2", "t0005"), ("b2", "t0006"), ("p2", "t0007")])
    carried = _carried_step("p1", "t0004")
    link = _link([("p1", "t0005", "p1", "t0004", 1), ("p2", "t0007", "p1", "t0006", 1)])
    pb = pl.DataFrame({"period": ["p1", "p1"], "branch": ["b1", "b2"]})
    rows, _ = ramp_roll_carry(dt, carried, link, pb)
    assert sorted(rows.select("d", "t").rows()) == [("b1", "t0005"), ("b2", "t0005")]


# ---------------------------------------------------------------------------
# check_online_ramp_carry_consistency


def test_missing_online_carry_raises_instead_of_defaulting_to_zero() -> None:
    """An online process with a ramp-carry flow at a dt_ramp_carry step but no matching
    p_roll_continue_online entry raises, instead of the ramp split (part C) silently reading a false
    online-before-history of 0 there."""
    dt_ramp_carry = pl.DataFrame({"d": ["p1"], "t": ["t0005"]})
    p_roll_continue_flow_p = pl.DataFrame({"p": ["u"]})
    process_online = pl.DataFrame({"p": ["u"]})
    with pytest.raises(ValueError, match="no online carry"):
        check_online_ramp_carry_consistency(None, None, dt_ramp_carry, p_roll_continue_flow_p,
                                            process_online, None)


def test_disjoint_rules_raise_on_overlap() -> None:
    """A step cannot be both in dt_online_carry and a dt_ramp_link target: the two online-predecessor rules are
    defined on disjoint domains."""
    dt_online_carry = pl.DataFrame({"d": ["p1"], "t": ["t0005"]})
    dt_ramp_link = pl.DataFrame({"d": ["p1"], "t": ["t0005"], "d_previous": ["p0"], "t_previous": ["t0010"]})
    with pytest.raises(ValueError, match="disjoint"):
        check_online_ramp_carry_consistency(dt_online_carry, None, None, None, None, dt_ramp_link)


def test_consistent_carry_does_not_raise() -> None:
    """The happy path: matching online + ramp carry at the same step, no link overlap."""
    dt_ramp_carry = pl.DataFrame({"d": ["p1"], "t": ["t0005"]})
    dt_online_carry = pl.DataFrame({"d": ["p1"], "t": ["t0005"]})
    p_roll_continue_flow_p = pl.DataFrame({"p": ["u"]})
    p_roll_continue_online_p = pl.DataFrame({"p": ["u"]})
    process_online = pl.DataFrame({"p": ["u"]})
    dt_ramp_link = pl.DataFrame({"d": ["p2"], "t": ["t0001"], "d_previous": ["p1"], "t_previous": ["t0100"]})
    check_online_ramp_carry_consistency(dt_online_carry, p_roll_continue_online_p, dt_ramp_carry,
                                        p_roll_continue_flow_p, process_online, dt_ramp_link)
