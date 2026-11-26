"""Which periods a solve instance sees for the ``co2_max_total`` check (D54, ``engine/solve/_co2_total_scope``).

Hand-built solve trees, no solve: the rolls of one rolling solve see its periods together; nested lower solves are an
instance per parent roll; a realized-dispatch sibling sees its parent's periods; a forecast branch period counts as
its period label.
"""
from __future__ import annotations

import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.solve._co2_total_scope import check_co2_total_sees_all_periods, solve_period_views


def test_rolls_see_their_solve_and_nested_children_their_own_window():
    """``roll`` (two rolls: p1, then p1 + p2) sees {p1, p2}.  ``disp`` runs nested under ``inv_r0`` (p1) and
    ``inv_r1`` (p2): two instances, {p1} and {p2}, so the model {p1, p2} errors on ``disp``, not on ``roll``."""
    solves = ["roll_0", "roll_1", "disp_a", "disp_b"]
    complete = {"roll_0": "roll", "roll_1": "roll", "disp_a": "disp", "disp_b": "disp"}
    parent = {"roll_0": None, "roll_1": None, "disp_a": "inv_r0", "disp_b": "inv_r1"}
    periods = {"roll_0": ["p1"], "roll_1": ["p1", "p2"], "disp_a": ["p1"], "disp_b": ["p2"]}
    views = solve_period_views(solves, complete, parent, periods, {}, {})
    assert views == {("roll", None): {"p1", "p2"}, ("disp", "inv_r0"): {"p1"}, ("disp", "inv_r1"): {"p2"}}
    with pytest.raises(FlexToolConfigError, match=r"solve 'disp' does not have period\(s\) 'p2'"):
        check_co2_total_sees_all_periods(views, ["co2g"])
    check_co2_total_sees_all_periods(views, [])  # no total cap: no check


def test_branch_periods_and_a_realization_sibling_pass():
    """``st`` holds p1 and the forecast branch p1_A (its label p1), its sibling ``st__rd`` only the jump of p1, and
    ``late`` holds p1 too: every instance sees {p1}, the model's one period, so the check passes."""
    solves = ["st", "st__rd", "late"]
    complete = {s: s for s in solves}
    periods = {"st": ["p1", "p1_A"], "st__rd": ["p1"], "late": ["p1"]}
    views = solve_period_views(solves, complete, {}, periods, {"st": [("p1", "p1"), ("p1", "p1_A")]},
                               {"st__rd": "st"})
    assert all(seen == {"p1"} for seen in views.values())
    check_co2_total_sees_all_periods(views, ["co2g"])
