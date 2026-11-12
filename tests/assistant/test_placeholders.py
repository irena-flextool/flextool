"""Placeholder arguments (assistant tools batch B1, T1): the "no value" spellings a model writes for an optional
argument it does not mean to set are left out, per tool, before the caps are clamped and the schema is checked.

Evidence (helpfulness analysis 2026-11-12): ``list_entities(member: "", name_contains: "")`` listed no units (#29);
``state: ""`` was refused (#21, #42).  What must stay: ``group_by: []`` (one total), every boolean
(``results_usable: false`` is a filter), ``min_value: 0`` (a threshold), required arguments.
"""
from __future__ import annotations

import json

import pytest

from flextool.assistant.llm_adapter import ToolCall
from flextool.assistant.placeholders import EMPTIABLE, Kind, unset_placeholders
from flextool.assistant.registry import all_tools, call_tool
from flextool.assistant.session import call_signature


def _unset(tool: str, **args) -> dict:
    return unset_placeholders(tool, args, all_tools()[tool].input_schema)


def test_every_entry_is_an_optional_non_boolean_argument_of_its_tool() -> None:
    tools = all_tools()
    for name, rules in EMPTIABLE.items():
        assert name in tools, name
        schema = tools[name].input_schema
        for key, kind in rules.items():
            spec = schema["properties"].get(key)
            assert spec is not None, (name, key)
            assert key not in schema.get("required", ()), (name, key)
            assert spec.get("type") != "boolean", (name, key)
            assert key != "group_by", name
            assert key not in tools[name].caps, (name, key)             # the caps have their own rule
            if kind is Kind.ENUM:
                assert "enum" in spec, (name, key)
            if kind is Kind.OBJECT:
                assert spec.get("required"), (name, key)
    assert EMPTIABLE["get_slack_summary"] == {"min_value": Kind.VALUE}


@pytest.mark.parametrize("tool,given,left", [
    # Text filters: null, empty, spaces and exactly "*" mean "any".
    ("list_entities", {"entity_class": "unit", "member": "", "name_contains": "  ", "scenario": "*"},
     {"entity_class": "unit"}),
    ("list_entities", {"entity_class": "unit", "name_contains": "coal*", "scenario": "coal"},
     {"entity_class": "unit", "name_contains": "coal*", "scenario": "coal"}),
    # A required argument is never touched, even empty.
    ("list_entities", {"entity_class": ""}, {"entity_class": ""}),
    ("get_entity", {"name": [], "scenario": None}, {"name": []}),
    # Enums: also "none" / "null" in any case; a real value stays.
    ("list_runs", {"state": "", "detail": "None", "results_usable": False, "all": False},
     {"results_usable": False, "all": False}),
    ("list_runs", {"state": "succeeded", "detail": "scan"}, {"state": "succeeded", "detail": "scan"}),
    ("search_docs", {"query": "x", "kind": "null"}, {"query": "x"}),
    # Lists: [] and "" unset; their items stay patterns.
    ("list_runs", {"folders": []}, {}),
    ("list_runs", {"folders": ["*"], "offset": None}, {"folders": ["*"]}),
    ("get_parameter", {"entity_class": "unit", "parameter": "existing", "parameters": [], "entity": "",
                       "entities": None}, {"entity_class": "unit", "parameter": "existing"}),
    # Query objects: an empty map or one with only empty entries; an object with an empty required key whole.
    ("query_result", {"run": "r", "table": "t", "filters": {"node": [], "unit": None}, "sort": {"by": ""},
                      "time_range": {"from": "t01", "to": ""}, "agg": "none", "top_n": 0},
     {"run": "r", "table": "t"}),
    ("query_result", {"run": "r", "table": "t", "filters": {"node": ["west"], "unit": []},
                      "sort": {"by": "value", "descending": False}, "time_range": {"from": "t01", "to": "t02"},
                      "agg": "sum", "top_n": 3},
     {"run": "r", "table": "t", "filters": {"node": ["west"]}, "sort": {"by": "value", "descending": False},
      "time_range": {"from": "t01", "to": "t02"}, "agg": "sum", "top_n": 3}),
    # group_by [] is a request (one total), never a placeholder; nor is by_difference false.
    ("query_result", {"run": "r", "table": "t", "group_by": []}, {"run": "r", "table": "t", "group_by": []}),
    ("compare_runs", {"table": "t", "runs": ["a", "b"], "group_by": [], "by_difference": False, "top_n": -2},
     {"table": "t", "runs": ["a", "b"], "group_by": [], "by_difference": False}),
    # min_value 0 is a threshold; only null and "" unset it.
    ("get_slack_summary", {"run": "r", "min_value": 0}, {"run": "r", "min_value": 0}),
    ("get_slack_summary", {"run": "r", "min_value": None}, {"run": "r"}),
    # The caps are not this rule's (CapArg.resolve gives their default for 0 or less).
    ("get_entity", {"name": ["x"], "max_rows": 0}, {"name": ["x"], "max_rows": 0}),
])
def test_placeholders_are_left_out(tool: str, given: dict, left: dict) -> None:
    assert _unset(tool, **given) == left


def test_a_tool_without_entries_is_unchanged() -> None:
    assert unset_placeholders("find_result_tables", {"query": "", "max_results": 0}, {}) == {"query": "",
                                                                                             "max_results": 0}
    assert unset_placeholders("no_such_tool", {"a": ""}, {}) == {"a": ""}


def test_placeholders_list_as_no_filter(ctx) -> None:
    """#29: member and name_contains empty listed no unit; now the call lists as one without them.  In scenario
    coal only coal_plant is active (its entity_alternative is in alternative coal, tests.json)."""
    plain = call_tool(ctx, "list_entities", {"entity_class": "unit", "scenario": "coal"})
    filled = call_tool(ctx, "list_entities", {"entity_class": "unit", "scenario": "coal", "member": "",
                                              "name_contains": "", "source": None, "max_results": 0})
    assert plain.ok and filled.ok, (plain.message, filled.message)
    assert filled.text == plain.text
    assert json.loads(filled.text)["entities"] == [{"entity": ["coal_plant"], "description": None}]


def _sig(name: str, **args) -> str:
    return call_signature(ToolCall("x", name, args, json.dumps(args)))


def test_the_repeat_check_reads_placeholders_as_call_tool_does() -> None:
    assert _sig("list_entities", entity_class="unit", member="", scenario="*") == _sig("list_entities",
                                                                                      entity_class="unit")
    assert _sig("query_result", run="r", table="t", filters={}, top_n=0) == _sig("query_result", run="r", table="t")
    # One total (group_by []) is another request than one summary per series (no group_by).
    assert _sig("query_result", run="r", table="t", group_by=[]) != _sig("query_result", run="r", table="t")
    assert _sig("list_runs", results_usable=False) != _sig("list_runs")
    assert _sig("get_slack_summary", run="r", min_value=0) != _sig("get_slack_summary", run="r")
