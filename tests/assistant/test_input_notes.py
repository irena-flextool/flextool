"""The notes on input parameters state their conclusion for the entity (assistant tools batch B1, T4).

Small models read a general rule plus "Here invest_method is invest_total" backwards (failure analysis
2026-11-12 §3.3: invest_max_period 250 taken as a binding cap under invest_total; "penalty is not used" under the
default regular).  Each expected text is written from the user documentation (reference/units.md invest_method,
reference/nodes.md inflow_method and penalty_method), not from the code.
"""
from __future__ import annotations

import pytest

from flextool.assistant.inputs._notes import LIMIT_USERS, methods, notes_for
from flextool.assistant.inputs._schema import shipped_schema

PERIOD = "invest_period, invest_period_total, invest_retire_period, invest_retire_period_total"
TOTAL = "invest_total, invest_period_total, invest_retire_total, invest_retire_period_total"


def _notes(cls: str, param: str, value) -> list[str]:
    return notes_for(cls, param, value_of=lambda _control: value)


@pytest.mark.parametrize(("method", "expected"), [
    (None, "Here invest_method is not_allowed (default), so the unit cannot invest and invest_max_period is not "
           "used."),
    ("invest_total", f"Here invest_method is invest_total, so invest_max_period is not used (only {PERIOD} use "
                     "it)."),
    ("invest_period_total", "Here invest_method is invest_period_total, so invest_max_period is used."),
    ("invest_no_limit", "Here invest_method is invest_no_limit, so investment has no limits and invest_max_period "
                        "is not used."),
    ("retire_period", "Here invest_method is retire_period, so it allows retirement only and invest_max_period is "
                      "not used."),
    ("cumulative_limits", "Here invest_method is cumulative_limits, so the cumulative capacity limits apply "
                          f"instead and invest_max_period is not used (only {PERIOD} use it)."),
])
def test_a_period_limit_under_each_kind_of_invest_method(method, expected) -> None:
    assert _notes("unit", "invest_max_period", method) == [expected]


def test_a_total_limit_a_cost_and_a_group() -> None:
    assert _notes("connection", "invest_max_total", "invest_period") == [
        f"Here invest_method is invest_period, so invest_max_total is not used (only {TOTAL} use it)."]
    # invest_cost is used by every method that invests, cumulative_limits too; the ×1000 note comes first.
    assert _notes("unit", "invest_cost", "cumulative_limits")[1] == (
        "Here invest_method is cumulative_limits, so the unit can invest and invest_cost is used.")
    assert _notes("group", "invest_max_total", None) == [
        "Here invest_method is not_active (default), so the group sets no investment limits and invest_max_total "
        "is not used."]
    assert _notes("group", "invest_min_period", "retire_total") == [
        "Here invest_method is retire_total, so the group limits retirement only and invest_min_period is not "
        "used."]


def test_without_the_entity_the_rule_is_given() -> None:
    assert notes_for("unit", "invest_max_period") == [
        f"Used only when the unit's invest_method is one of {PERIOD} (default not_allowed)."]
    assert notes_for("group", "invest_max_total") == [
        f"Used only when the group's invest_method is one of {TOTAL} (default not_active: the group sets no "
        "limits)."]
    # The penalty rule names neither "not used" nor "infeasible": those are the conclusion under off only.
    rule = notes_for("node", "penalty_up")[0]
    assert "not used" not in rule and "infeasible" not in rule
    # A value without a conclusion (an older database's spelling) gets the rule and the value.
    assert _notes("unit", "invest_max_period", "invest_sometimes") == [
        f"Used only when the unit's invest_method is one of {PERIOD} (default not_allowed). Here invest_method is "
        "invest_sometimes."]


def test_penalties_used_under_regular_and_not_under_off() -> None:
    assert _notes("node", "penalty_up", None) == [
        "Here penalty_method is regular (default), so penalty_up is used: it is the cost per MWh of the node's "
        "energy not served."]
    assert _notes("node", "penalty_down", "regular") == [
        "Here penalty_method is regular, so penalty_down is used: it is the cost per MWh of the node's excess "
        "energy."]
    assert _notes("node", "penalty_up", "off") == [
        "Here penalty_method is off, so penalty_up is not used: the node balance has no slack, and an imbalance "
        "makes the solve infeasible."]


@pytest.mark.parametrize(("method", "conclusion"), [
    (None, "Here inflow_method is use_original (default), so FlexTool uses the stored inflow as it is."),
    ("no_inflow", "Here inflow_method is no_inflow, so FlexTool ignores the stored inflow: the node has no "
                  "inflow."),
    ("scale_to_annual_flow", "Here inflow_method is scale_to_annual_flow, so FlexTool multiplies the stored inflow "
                             "by one factor so that its annual energy equals annual_flow: the stored values give "
                             "the shape."),
    ("scale_in_proportion", "Here inflow_method is scale_in_proportion, so FlexTool multiplies the stored inflow "
                            "by annual_flow divided by the stored inflow's annualized sum: the stored values give "
                            "the shape."),
    ("scale_to_annual_and_peak_flow", "Here inflow_method is scale_to_annual_and_peak_flow, so FlexTool transforms "
                                      "the stored inflow so that its annual energy equals annual_flow and its "
                                      "highest step reaches peak_inflow: the stored values give the shape."),
])
def test_inflow_under_each_inflow_method(method, conclusion) -> None:
    assert _notes("node", "inflow", method) == [conclusion]


def test_every_inflow_method_has_its_conclusion() -> None:
    assert len(shipped_schema().value_lists["inflow_methods"]) == 5
    for method in shipped_schema().value_lists["inflow_methods"]:
        assert "Used only" not in _notes("node", "inflow", method)[0]


@pytest.mark.parametrize(("size", "expected"), [
    (None, "Here virtual_unitsize is not set, so existing is the capacity itself (MW)."),
    (0.0, "Here virtual_unitsize is 0 (0 or below counts as not set), so existing is the capacity itself (MW)."),
    (-5.0, "Here virtual_unitsize is -5 (0 or below counts as not set), so existing is the capacity itself (MW)."),
    (250.0, "Here virtual_unitsize is 250, so existing is a number of units: the capacity is existing × 250 MW."),
])
def test_existing_with_and_without_a_virtual_unitsize(size, expected) -> None:
    assert _notes("unit", "existing", size) == [expected]
    if size is None:
        assert _notes("node", "existing", size) == [
            "Here virtual_unitsize is not set, so existing is the capacity itself (MWh)."]


def test_the_method_sets_cover_the_schema_lists() -> None:
    """The engine's sets name exactly the schema's values; each limit parameter exists for units."""
    m = methods()
    lists = shipped_schema().value_lists
    assert m.known == set(lists["invest_methods"]) | set(lists["group_invest_methods"])
    assert m.retire_only == {"retire_no_limit", "retire_period", "retire_total", "retire_period_total"}
    assert m.cumulative == {"cumulative_limits"}
    assert ", ".join(sorted(m.invest_period)) and m.invest_period == set(PERIOD.split(", "))
    assert m.invest_total == set(TOTAL.split(", "))
    assert all(("unit", p) in shipped_schema().parameters for p in LIMIT_USERS)
