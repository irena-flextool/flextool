"""Which periods a solve instance sees for the ``co2_max_total`` check (D54, ``engine/solve/_co2_total_scope``).

Hand-built solve trees, no solve: the rolls of one rolling solve see its periods together; nested lower solves are an
instance per parent roll; a realized-dispatch sibling sees its parent's periods; a period without years is not a
model period.
"""
from __future__ import annotations

import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.solve._co2_total_scope import (
    check_co2_total_sees_all_periods,
    solve_period_views,
    weightless_periods,
)


def test_rolls_see_their_solve_and_nested_children_their_own_window():
    """``roll`` (two rolls: p1, then p1 + p2) sees {p1, p2}.  ``disp`` runs nested under ``inv_r0`` (p1) and
    ``inv_r1`` (p2): two instances, {p1} and {p2}, which lack different periods, so the error lists them per
    parent.  No total cap: no check."""
    solves = ["roll_0", "roll_1", "disp_a", "disp_b"]
    complete = {"roll_0": "roll", "roll_1": "roll", "disp_a": "disp", "disp_b": "disp"}
    parent = {"roll_0": None, "roll_1": None, "disp_a": "inv_r0", "disp_b": "inv_r1"}
    periods = {"roll_0": ["p1"], "roll_1": ["p1", "p2"], "disp_a": ["p1"], "disp_b": ["p2"]}
    views = solve_period_views(solves, complete, parent, periods, {})
    assert views == {("roll", None): {"p1", "p2"}, ("disp", "inv_r0"): {"p1"}, ("disp", "inv_r1"): {"p2"}}
    with pytest.raises(FlexToolConfigError, match=r"solve 'disp' does not have, under 'inv_r0', period\(s\) 'p2'; "
                                                  r"under 'inv_r1', period\(s\) 'p1': "):
        check_co2_total_sees_all_periods(views, ["co2g"])
    check_co2_total_sees_all_periods(views, [])


def test_one_list_when_every_instance_lacks_the_same_periods():
    """Two solves without p3, the first of them nested under two parents that both lack it: one list, and one more
    solve (singular)."""
    views = {("a", "r0"): {"p1", "p2"}, ("a", "r1"): {"p1", "p2"}, ("b", None): {"p2"}, ("c", None): {"p1", "p2", "p3"}}
    with pytest.raises(FlexToolConfigError, match=r"solve 'a' \(and 1 more solve\) does not have period\(s\) 'p3': "):
        check_co2_total_sees_all_periods(views, ["co2g"])


def test_a_realization_sibling_sees_its_parent_and_weightless_periods_do_not_count():
    """``st__rd`` holds only the jump of p1 but sees its parent ``st`` (p0, p1).  ``late`` holds p1 alone, and p0
    has years_represented 0 in every solve that lists it, so it is not a model period and the check passes; with
    years 1 in one solve it counts and ``late`` lacks it."""
    solves = ["st", "st__rd", "late"]
    views = solve_period_views(solves, {s: s for s in solves}, {},
                               {"st": ["p0", "p1"], "st__rd": ["p1"], "late": ["p1"]}, {"st__rd": "st"})
    assert views[("st__rd", None)] == {"p0", "p1"}
    years = {"st": [("p0", 0.0), ("p1", 1.0)], "late": [("p1", 1.0)]}
    assert weightless_periods(years) == {"p0"}
    check_co2_total_sees_all_periods(views, ["co2g"], weightless_periods(years))
    years["other"] = [("p0", 1.0)]
    with pytest.raises(FlexToolConfigError, match=r"solve 'late' does not have period\(s\) 'p0'"):
        check_co2_total_sees_all_periods(views, ["co2g"], weightless_periods(years))
