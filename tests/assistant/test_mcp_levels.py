"""The levels for AI agents (P3 §2.3, test 5): the installed ``flextool-assistant --mcp`` in a subprocess.

* Without ``--tier``: ``prompts/list`` gives basic, standard and advanced; ``prompts/get basic`` is one user
  message with the basic level text; ``tools/list`` gives the 18 tools.
* ``--tier basic``: ``tools/list`` gives the eight basic tools and only the basic prompt; any other tool is an
  error result that says how to get it; a basic tool answers.
* ``--tier standard`` keeps every tool and prompt.
* ``--result-cap`` follows ``--tier`` unless it is given (basic small, standard medium, advanced large).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from flextool.assistant import prompt
from tests.assistant.test_server_stdio import _MCP_MAJOR, _Connection, _script, _text

pytestmark = pytest.mark.skipif(_MCP_MAJOR < 2, reason="needs mcp 2.x (reinstall FlexTool: pip install -e .)")

#: The basic level's tools (P3 §2.2, P3-O15), written from the design, not from the registry.
BASIC_TOOLS = ["find_result_tables", "get_entity", "get_parameter", "list_runs", "list_sources", "query_result",
               "read_doc", "search_docs"]
REFUSED = ('The tool "compare_runs" is not part of the basic level. Restart the FlexTool connection without --tier '
           "basic to use it.")


def _server(project: Path, *extra: str) -> _Connection:
    conn = _Connection([str(_script()), "--mcp", "--project-path", str(project), "--scenario", "coal", *extra])
    conn.initialize()
    return conn


def _finish(conn: _Connection) -> None:
    code = conn.close()
    assert code == 0, conn.stderr_text()
    conn.assert_only_protocol_on_stdout()


def test_without_a_level_the_agent_gets_three_prompts_and_every_tool(assistant_project: Path) -> None:
    conn = _server(assistant_project)
    try:
        prompts = conn.send("prompts/list", {})["prompts"]
        assert [p["name"] for p in prompts] == ["basic", "standard", "advanced"]
        assert [p["description"] for p in prompts] == [
            "Work at the basic level: look up and explain values with a few tool calls.",
            "Work at the standard level: interpret results and compare scenarios.",
            "Work at the advanced level: trace slack and penalties, read equations, explain investment "
            "differences.",
        ]
        got = conn.send("prompts/get", {"name": "basic", "arguments": {"question": "What is the demand of west?"}})
        (message,) = got["messages"]
        assert message["role"] == "user"
        text = message["content"]["text"]
        assert text.startswith("Use these instructions for the rest of this conversation about FlexTool.\n\n")
        assert prompt.ADDENDA["small"] in text
        assert text.endswith("\n\nQuestion: What is the demand of west?")
        tools = conn.send("tools/list", {})["tools"]
        assert len(tools) == 18
        # Without a level the agent may pick any prompt: every tool in its full form (the superset).
        assert _listed(tools) == _full_forms()
    finally:
        _finish(conn)


def _listed(tools: list[dict]) -> dict[str, tuple[str, dict]]:
    return {t["name"]: (t["description"], t["inputSchema"]) for t in tools}


def _full_forms() -> dict[str, tuple[str, dict]]:
    from flextool.assistant import registry

    return {t.name: (t.description, t.input_schema) for t in registry.all_tools().values()}


def test_tier_basic_lists_exactly_what_the_basic_chat_sends(assistant_project: Path) -> None:
    """Critique m2: the prefix pins measure ``tool_specs`` of each level; the connection lists its tools itself.
    With ``--tier basic`` its tools/list (name, description, schema) is the basic chat's (the basic forms), with
    ``--tier standard`` the full forms.  Before: the connection had no basic forms."""
    from flextool.assistant.session import tool_specs

    conn = _server(assistant_project, "--tier", "basic")
    try:
        basic = _listed(conn.send("tools/list", {})["tools"])
    finally:
        _finish(conn)
    assert basic == {t.name: (t.description, t.input_schema) for t in tool_specs("small")}
    assert basic["list_sources"] != _full_forms()["list_sources"]
    conn = _server(assistant_project, "--tier", "standard")
    try:
        assert _listed(conn.send("tools/list", {})["tools"]) == _full_forms()
    finally:
        _finish(conn)


def test_tier_basic_serves_the_eight_basic_tools_and_refuses_the_rest(assistant_project: Path) -> None:
    conn = _server(assistant_project, "--tier", "basic")
    try:
        tools = conn.send("tools/list", {})["tools"]
        assert sorted(t["name"] for t in tools) == BASIC_TOOLS
        refused = conn.send("tools/call", {"name": "compare_runs",
                                           "arguments": {"runs": ["coal_1", "base_1"], "table": "node_d_ee"}})
        assert refused["isError"] is True and _text(refused) == REFUSED
        answered = conn.send("tools/call", {"name": "get_entity", "arguments": {"name": ["west"]}})
        assert answered["isError"] is False
        assert [m["entity"] for m in json.loads(_text(answered))["matches"]] == [["west"]]
        assert [p["name"] for p in conn.send("prompts/list", {})["prompts"]] == ["basic"]
    finally:
        _finish(conn)


def test_tier_standard_keeps_every_tool_and_prompt(assistant_project: Path) -> None:
    conn = _server(assistant_project, "--tier", "standard")
    try:
        assert len(conn.send("tools/list", {})["tools"]) == 18
        assert [p["name"] for p in conn.send("prompts/list", {})["prompts"]] == ["basic", "standard", "advanced"]
    finally:
        _finish(conn)


@pytest.mark.parametrize(("extra", "expected"), [
    ((), "medium"),
    (("--tier", "basic"), "small"),
    (("--tier", "standard"), "medium"),
    (("--tier", "advanced"), "large"),
    (("--tier", "basic", "--result-cap", "large"), "large"),
])
def test_the_level_sets_the_result_cap_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                               capsys: pytest.CaptureFixture, extra: tuple[str, ...],
                                               expected: str) -> None:
    from flextool.assistant import cli, context, registry, store

    seen: dict[str, object] = {}

    def make_context(project, **kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(context, "make_context", make_context)
    monkeypatch.setattr(store, "purge_at_start", lambda: None)
    monkeypatch.setattr(registry, "call_tool", lambda ctx, name, args: registry.ToolOutcome(True, {}, None, "{}"))
    assert cli.main(["--call", "list_runs", "--project-path", str(tmp_path), *extra]) == 0
    assert seen["result_cap"] == expected
    assert capsys.readouterr().out == "{}\n"
