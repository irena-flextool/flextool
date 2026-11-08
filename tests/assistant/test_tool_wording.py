"""The words the tools use: every description ends with the data sentence, and no text the user or the AI service
sees uses an abbreviation (LLM, ZDR, DB, LP), the word MCP (except the flag --mcp and the package mcp) or a
schema version number.  The same holds for the connection's instructions, the level texts and the command's own
messages."""
from __future__ import annotations

import re

import pytest

from flextool.assistant import cli, mcp_server, prompt, unit_wording
from flextool.assistant.errors import MESSAGES
from flextool.assistant.inputs import _values
from flextool.assistant.inputs._notes import traps
from flextool.assistant.registry import DATA_SENTENCE, all_tools
from flextool.update_flextool import FLEXTOOL_DB_VERSION

_FORBIDDEN = re.compile(r"\b(LLM|ZDR|DB|LP)s?\b")
_MCP = re.compile(r"MCP")
_VERSION = re.compile(rf"\bv?{FLEXTOOL_DB_VERSION}\b|\bv\d{{2}}\b")


def _texts() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for tool in all_tools().values():
        out.append((f"{tool.name}.title", tool.title))
        out.append((f"{tool.name}.description", tool.description))
    out += [(f"MESSAGES[{k}]", v) for k, v in MESSAGES.items()]
    out += [(f"session note {k}", v) for k, v in unit_wording.SESSION_NOTES.items()]
    out += [(f"trap {k}", t.text) for k, ts in traps().items() for t in ts]
    out.append(("RUN_TIME_NOTE", _values.RUN_TIME_NOTE))
    # What the connection server and the command say.
    out.append(("instructions", prompt.instructions("project")))
    out += [(f"ADDENDA[{k}]", v) for k, v in prompt.ADDENDA.items()]
    out += [(f"PROMPT_DESCRIPTIONS[{k}]", v) for k, v in prompt.PROMPT_DESCRIPTIONS.items()]
    out += [("PROMPT_LEAD", prompt.PROMPT_LEAD), ("DEGRADED_INSTRUCTIONS", mcp_server.DEGRADED_INSTRUCTIONS),
            ("DEGRADED_DESCRIPTION", mcp_server.DEGRADED_DESCRIPTION), ("cli.DATA_NOTICE", cli.DATA_NOTICE),
            ("cli.EXAMPLES", cli.EXAMPLES), ("cli.MCP_MISSING", cli.MCP_MISSING)]
    return out


def test_every_description_ends_with_the_data_sentence() -> None:
    for tool in all_tools().values():
        assert tool.description.endswith(" " + DATA_SENTENCE), tool.name
        assert "minimum" not in str(tool.input_schema) and "maximum" not in str(tool.input_schema), tool.name


@pytest.mark.parametrize("where,text", _texts(), ids=[w for w, _ in _texts()])
def test_plain_words(where: str, text: str) -> None:
    assert not _FORBIDDEN.search(text), f"{where}: {text}"
    assert not _MCP.search(text), f"{where}: {text}"
    assert not _VERSION.search(text), f"{where}: {text}"


def test_tool_names_and_small_tier() -> None:
    """The tools of this step and the basic set among them (P3 §2.2: search_docs, read_doc, get_entity,
    get_parameter, list_sources, list_runs, find_result_tables, query_result)."""
    tools = all_tools()
    small = {name for name, tool in tools.items() if tool.small_tier}
    basic = {"search_docs", "read_doc", "get_entity", "get_parameter", "list_sources", "list_runs",
             "find_result_tables", "query_result"}
    assert small == basic & set(tools)
    assert not tools["get_gui_context"].small_tier
