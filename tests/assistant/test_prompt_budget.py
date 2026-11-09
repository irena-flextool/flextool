"""The level texts and tool sets (P3 §8 test 1; §2).

* the instructions and tool schemas of each level and front end fit the level's prefix budget (digit-aware
  estimate, P3 §4.1);
* basic sends exactly the eight tools of P3-O15 (with list_sources, 2026-11-08), standard and advanced all
  eighteen;
* no layer sent at a level — the chat's five layers for each front end, and the connection's instructions —
  names a tool outside that level's set (critique M1);
* the level texts are the owner-approved ones (P3-O9, O15's tool sentence), byte-stable across builds;
* the connection's level prompts carry the same texts.

The plain-English rules (no abbreviations, no promises, no links) are P1 test 6
(``tests/assistant/llm/test_prompt_and_wording.py``), which covers every layer at every level.
"""
from __future__ import annotations

import hashlib
import re

import pytest

from flextool.assistant import prompt, registry
from flextool.assistant.budget import LEVEL_BUDGETS, approx_tokens, schema_tokens
from flextool.assistant.session import tool_specs

BASIC_TOOLS = ("find_result_tables", "get_entity", "get_parameter", "list_runs", "list_sources", "query_result",
               "read_doc", "search_docs")
FRONT_ENDS = ("terminal", "gui", "agent")
LEVELS = ("small", "medium", "large")

#: The basic level's tool sentence (P3-O15, with list_sources since 2026-11-08), word for word.
BASIC_TOOL_SENTENCE = ("- Use only these tools: search_docs, read_doc, get_entity, get_parameter, list_sources, "
                       "list_runs, find_result_tables and query_result.")
#: sha256 of each level text as approved (P3 §2.1, word for word; any change goes back to the owner, P3-O9).  The
#: basic text has said "at most three steps" since owner decision D1 (2026-11-09).
ADDENDA_SHA256 = {
    "small": "7972aedf2e4023a3b59be1b0b84db067369d26c237651b906de83d2bc9b89ec0",
    "medium": "e1b0aebb67e1413f3b3ee5cc7a6e3ec4fe5b5ec5d051888181cc8c19b372b9c3",
    "large": "2700bf6339a535c19f473ea18459c73887b7bfcfb397d1fd8dd09d8b76771e86",
}


#: The prefix budgets (C2-OD2, Cycle 2 C2-D1: basic 4 920, standard 9 000, advanced 9 100).
PREFIX_BUDGETS = {"small": 4_920, "medium": 9_000, "large": 9_100}
#: The prefix estimate (layers and tool schemas, digit-aware) of every level and front end.  A change of a level
#: text or a tool description moves one of them on purpose: update the pin, and the budget table in
#: ``specs/ai_assistant_autonomous_decisions.md`` (Cycle 2 C2-D-prefix), in the same commit.
PREFIX_PINS = {
    ("small", "gui"): 4_872, ("small", "terminal"): 4_852, ("small", "agent"): 4_792,
    ("medium", "gui"): 8_912, ("medium", "terminal"): 8_892, ("medium", "agent"): 8_832,
    ("large", "gui"): 8_998, ("large", "terminal"): 8_978, ("large", "agent"): 8_918,
}


def _prefix(front_end: str, level: str) -> int:
    layers = prompt.system_layers(front_end, level)
    return sum(approx_tokens(layer, "text") for layer in layers) + schema_tokens(tool_specs(level))


def _named_tools(text: str) -> set[str]:
    names = registry.all_tools()
    return {name for name in names if re.search(rf"(?<![\w]){re.escape(name)}(?![\w])", text)}


@pytest.mark.parametrize("front_end", FRONT_ENDS)
@pytest.mark.parametrize("level", LEVELS)
def test_the_prefix_fits_the_level_budget(front_end, level):
    prefix = _prefix(front_end, level)
    assert prefix <= LEVEL_BUDGETS[level].prefix_tokens, f"{level} {front_end}: {prefix} tokens"


@pytest.mark.parametrize("front_end", FRONT_ENDS)
@pytest.mark.parametrize("level", LEVELS)
def test_the_prefix_estimate_is_pinned(front_end, level):
    """Every level × front end (the agent's connection lists the same forms, ``test_mcp_levels``) equals its pin,
    and the pin is within the budget.  Before Cycle 2: no pins, budgets 4 850 / 8 700 / 9 000."""
    assert LEVEL_BUDGETS[level].prefix_tokens == PREFIX_BUDGETS[level]
    prefix = _prefix(front_end, level)
    assert prefix == PREFIX_PINS[(level, front_end)], (
        f"{level} {front_end}: {prefix} tokens, pinned {PREFIX_PINS[(level, front_end)]}: update the pin and the "
        "budget table in the decisions file")
    assert PREFIX_PINS[(level, front_end)] <= PREFIX_BUDGETS[level]


def test_basic_is_the_eight_tools():
    assert tuple(t.name for t in registry.tools_for_tier("small")) == BASIC_TOOLS
    assert tuple(t.name for t in tool_specs("basic")) == BASIC_TOOLS
    assert len(registry.all_tools()) == 18
    for level in ("medium", "large"):
        assert {t.name for t in registry.tools_for_tier(level)} == set(registry.all_tools())
    assert not registry.all_tools()["get_gui_context"].small_tier


@pytest.mark.parametrize("front_end", FRONT_ENDS)
@pytest.mark.parametrize("level", LEVELS)
def test_no_layer_names_a_tool_outside_the_level(front_end, level):
    allowed = {t.name for t in registry.tools_for_tier(level)}
    for i, layer in enumerate(prompt.system_layers(front_end, level), start=1):
        outside = _named_tools(layer) - allowed
        assert not outside, f"layer {i} at {level} names {sorted(outside)}"


def test_the_connection_instructions_name_only_basic_tools():
    """The connection's instructions are the same for every level (``--tier basic`` included)."""
    assert _named_tools(prompt.instructions("project")) <= set(BASIC_TOOLS)


def test_the_basic_text_lists_its_tools():
    assert BASIC_TOOL_SENTENCE in prompt.ADDENDA["small"]
    assert _named_tools(prompt.ADDENDA["small"]) == set(BASIC_TOOLS)
    assert prompt.LAYER_ORIENTATION.endswith("Start from an entity with get_entity.")


def test_the_level_texts_are_byte_stable():
    from importlib import reload

    before = dict(prompt.ADDENDA)
    reload(prompt)
    assert prompt.ADDENDA == before
    assert {level: hashlib.sha256(prompt.ADDENDA[level].encode("utf-8")).hexdigest()
            for level in LEVELS} == ADDENDA_SHA256
    for level in LEVELS:
        assert prompt.system_layers("gui", level)[3] == prompt.ADDENDA[level]
    assert prompt.ADDENDA["small"].startswith("Your level: basic.")
    assert prompt.ADDENDA["medium"].startswith("Your level: standard.")
    assert prompt.ADDENDA["large"].startswith("Your level: advanced.")


@pytest.mark.parametrize("level", LEVELS)
def test_the_connection_prompts_embed_the_level_texts(level):
    text = prompt.level_prompt(level)
    assert text == "Use these instructions for the rest of this conversation about FlexTool.\n\n" + \
        prompt.ADDENDA[level]
    asked = prompt.level_prompt(level, "How large is h2_tank?")
    assert asked.endswith("\n\nQuestion: How large is h2_tank?")
    assert prompt.level_prompt(level, "x" * 3_000).endswith("x" * prompt.QUESTION_MAX)
    assert len(prompt.level_prompt(level, "x" * 3_000)) == len(text) + len("\n\nQuestion: ") + prompt.QUESTION_MAX


def test_the_basic_planning_sentence_counts_steps():
    """Owner decision D1 (2026-11-09, review G11, C5-10): the basic text plans "at most three steps", the basic
    budget's number of steps, not three tool calls.  Each option is one constant away and keeps the prefix within
    the basic budget."""
    assert prompt.BASIC_PLAN_OPTION == "b"
    basic = prompt.ADDENDA["small"]
    assert ("- Use at most three steps for each question, so plan them: first find the right name (search_docs, "
            "find_result_tables or list_runs), then fetch the value.\n") in basic
    assert "tool calls" not in basic and LEVEL_BUDGETS["small"].max_rounds == 3
    assert not any(word in prompt.BASIC_PLAN_SENTENCES["c"] for word in ("three", "3"))
    for option, sentence in prompt.BASIC_PLAN_SENTENCES.items():
        text = basic.replace(prompt.BASIC_PLAN_SENTENCES["b"], sentence)
        assert text.count(sentence) == 1, option
        layers = [text if layer == basic else layer for layer in prompt.system_layers("gui", "small")]
        prefix = sum(approx_tokens(layer, "text") for layer in layers) + schema_tokens(tool_specs("small"))
        assert prefix <= LEVEL_BUDGETS["small"].prefix_tokens, option
