"""Pure-function tests for the online-commitment roll carry (part A / N1, c14c).

``online_roll_carry_steps`` picks the step(s) of this roll that take the previous roll's carried
``v_online`` (``p_roll_continue_online``) as their online predecessor: the roll's real first step
(whatever the complete solve's predecessor's ``jump`` -- a new year still continues the commitment, unlike
the ramp carry), plus forecast branches starting with their anchor; when no period's complete-solve
predecessor matches the carried step, the roll's own first step in TIMELINE order (``dt``'s row order, not a
``(d, t)`` name sort) -- the bug this fixes (N1).  ``check_online_ramp_carry_consistency`` raises instead of
letting the two online-predecessor rules silently disagree (critique item 2).
"""
from __future__ import annotations

import pytest
import polars as pl

from flextool.engine.preprocess.params._ramp_links import (
    check_online_ramp_carry_consistency, online_roll_carry_steps,
)


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
# A3: online_roll_carry_steps


def test_adjacent_carried_step_takes_the_carry() -> None:
    """A3.1: the complete solve's predecessor of p1's first step (t0005) is the carried step (p1, t0004),
    jump == 1 (adjacent): p1's first step takes the carry."""
    dt = _dt([("p1", "t0005"), ("p1", "t0006")])
    carried = _carried_step("p1", "t0004")
    link = _link([("p1", "t0005", "p1", "t0004", 1)])
    rows = online_roll_carry_steps(dt, carried, link)
    assert rows.select("d", "t").rows() == [("p1", "t0005")]


def test_non_adjacent_carried_step_still_takes_the_carry() -> None:
    """A3.2: unlike the ramp carry, the online carry does not require jump == 1 -- a new year still
    continues the commitment (the unit stays committed across it)."""
    dt = _dt([("p2", "t0001")])
    carried = _carried_step("p1", "t0072")
    link = _link([("p2", "t0001", "p1", "t0072", 5)])   # jump 5: not adjacent, still matches
    rows = online_roll_carry_steps(dt, carried, link)
    assert rows.select("d", "t").rows() == [("p2", "t0001")]


def test_forecast_branch_starting_with_its_anchor_is_included() -> None:
    """A3.3: a forecast branch (b1) that starts at the same step as its anchor period (p1) takes the carry
    too; period p2 (not an anchor-sharing branch, and not matched by the carried step) does not."""
    dt = _dt([("p1", "t0005"), ("p1", "t0006"), ("b1", "t0005"), ("b1", "t0006"), ("p2", "t0001")])
    carried = _carried_step("p1", "t0004")
    link = _link([("p1", "t0005", "p1", "t0004", 1), ("p2", "t0001", "p1", "t0006", 1)])
    pb = pl.DataFrame({"period": ["p1", "p1"], "branch": ["p1", "b1"]})
    rows = online_roll_carry_steps(dt, carried, link, pb)
    assert sorted(rows.select("d", "t").rows()) == [("b1", "t0005"), ("p1", "t0005")]


def test_no_match_falls_back_to_timeline_order_not_name_sort() -> None:
    """A3.4 (N1, the bug this fixes): no period's complete-solve predecessor is the carried step (e.g. a
    nested child solve whose complete solve does not contain it) -- the carry lands on the roll's own first
    step in TIMELINE order (``dt``'s row order), not the ``(d, t)``-name-sorted first row.  ``dt`` here lists
    ``a2025`` before ``p2020`` is name-wise later but ``p2020`` is actually first in solve order; a (d, t)
    name sort would wrongly pick ``a2025`` (today's bug)."""
    dt = _dt([("p2020", "t0011"), ("a2025", "t0025"), ("p2020", "t0012")])
    carried = _carried_step("x", "t9999")               # matches nothing
    link = _link([("p2020", "t0011", "y", "t0001", 1)])  # predecessor is not the carried step
    rows = online_roll_carry_steps(dt, carried, link)
    assert rows.select("d", "t").rows() == [("p2020", "t0011")]


def test_no_carry_without_a_carried_step() -> None:
    """A3.5: first roll / non-rolling solve (no carried_step, or no dt) -> None."""
    dt = _dt([("p1", "t0001")])
    assert online_roll_carry_steps(dt, None, _link([("p1", "t0001", "p0", "t0010", 1)])) is None
    assert online_roll_carry_steps(None, _carried_step("p0", "t0010"), _link([])) is None


def test_ambiguous_carried_step_raises() -> None:
    """``carried_step`` must identify exactly one step of the previous roll."""
    dt = _dt([("p1", "t0001")])
    carried = pl.DataFrame({"period": ["p1", "p2"], "step": ["t0004", "t0004"], "duration": [1.0, 1.0]})
    with pytest.raises(ValueError, match="one step"):
        online_roll_carry_steps(dt, carried, _link([("p1", "t0001", "p1", "t0004", 1)]))


# ---------------------------------------------------------------------------
# critique item 2: check_online_ramp_carry_consistency


def test_missing_online_carry_raises_instead_of_defaulting_to_zero() -> None:
    """Item 2.1: an online process with a ramp-carry flow at a dt_ramp_carry step but no matching
    p_roll_continue_online entry raises, instead of the ramp split (part C) silently reading a false
    online-before-history of 0 there."""
    dt_ramp_carry = pl.DataFrame({"d": ["p1"], "t": ["t0005"]})
    p_roll_continue_flow_p = pl.DataFrame({"p": ["u"]})
    process_online = pl.DataFrame({"p": ["u"]})
    with pytest.raises(ValueError, match="no online carry"):
        check_online_ramp_carry_consistency(None, None, dt_ramp_carry, p_roll_continue_flow_p,
                                            process_online, None)


def test_missing_online_carry_step_raises() -> None:
    """Item 2.1 (step mismatch): the process has an online carry (p_roll_continue_online) but not at the
    ramp-carry step itself."""
    dt_ramp_carry = pl.DataFrame({"d": ["p1"], "t": ["t0005"]})
    dt_online_carry = pl.DataFrame({"d": ["p2"], "t": ["t0001"]})   # a different step
    p_roll_continue_flow_p = pl.DataFrame({"p": ["u"]})
    p_roll_continue_online_p = pl.DataFrame({"p": ["u"]})
    process_online = pl.DataFrame({"p": ["u"]})
    with pytest.raises(ValueError, match="no matching online-carry step"):
        check_online_ramp_carry_consistency(dt_online_carry, p_roll_continue_online_p, dt_ramp_carry,
                                            p_roll_continue_flow_p, process_online, None)


def test_disjoint_rules_raise_on_overlap() -> None:
    """Item 2.2: a step cannot be in both dt_online_carry (rule 2) and a dt_ramp_link target (rule 1) --
    the two online-predecessor rules are defined on disjoint domains."""
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
