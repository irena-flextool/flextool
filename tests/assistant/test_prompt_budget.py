"""The level texts and tool sets (P3 §8 test 1; §2).

* the instructions and tool schemas of each level and front end fit the level's prefix budget (digit-aware
  estimate, P3 §4.1), the room the chat keeps for them;
* basic sends exactly the eight tools of P3-O15 (with list_sources, 2026-11-08), standard and advanced all
  nineteen;
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
#: basic text has said "at most three steps" since owner decision D1 (2026-11-09); every level's text ends with
#: "How to answer" since 2026-11-11 (basic: its lookup bullets).  The owner approved the 2026-11-12 texts (plain words
#: with the FlexTool term, "More detail:", the alternative, rounding, no guessing, inactive entity first; basic: the
#: inactive-entity sentence, no guessing, shorter "Do not:" bullets).
ADDENDA_SHA256 = {
    "small": "d54ffa8327ef1df765d21571fe2cc56c1e1d48085cbd6ea1aaaf95c381e8351c",
    "medium": "41f0f85cc42a56e11aeeb04a34a23668def897d7939c2b7491dea1d82da30685",
    "large": "c211a6938561730aad9eddac0261187bcb40c38e1b5834b2cc7458a94c6b7402",
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
    """The level's prefix budget is the room the chat keeps for the instructions and tool schemas when it fits
    results into the window (``budget.fit_caps`` / ``fitting_calls``), so it must cover the real prefix.  It bites
    at basic (a 16 384-token window); exact per-front-end pins were dropped 2026-11-11."""
    prefix = _prefix(front_end, level)
    assert prefix <= LEVEL_BUDGETS[level].prefix_tokens, f"{level} {front_end}: {prefix} tokens"


def test_basic_is_the_eight_tools():
    assert tuple(t.name for t in registry.tools_for_tier("small")) == BASIC_TOOLS
    assert tuple(t.name for t in tool_specs("basic")) == BASIC_TOOLS
    assert len(registry.all_tools()) == 19
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
