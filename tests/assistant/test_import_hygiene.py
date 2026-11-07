"""``--call`` and ``--mcp`` load no AI service library (P1 §15 test 13).

The connection for agents and the one-tool command start fast and never touch a key: neither ``anthropic``,
``openai`` nor ``keyring`` is imported, and neither is the chat in the terminal.  Each check runs the command's
entry in a fresh process (the test process has imported them all).  The store part of this check is in
``test_store_and_ledger.py``.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

LIBRARIES = ("anthropic", "openai", "keyring", "flextool.assistant.chat_cli")
CODE = """
import json, sys
import flextool_assistant_entry
try:
    code = flextool_assistant_entry.main({argv!r})
except SystemExit as exc:
    code = exc.code
sys.stderr.write("\\nIMPORTED " + json.dumps(sorted(m for m in {libraries!r} if m in sys.modules)) + "\\n")
sys.stderr.write("EXIT " + str(code) + "\\n")
"""


def _imported(argv: list[str]) -> tuple[list[str], str]:
    result = subprocess.run([sys.executable, "-c", CODE.format(argv=argv, libraries=LIBRARIES)],
                            capture_output=True, text=True, encoding="utf-8", timeout=300,
                            stdin=subprocess.DEVNULL)
    lines = result.stderr.splitlines()
    found = [line for line in lines if line.startswith("IMPORTED ")]
    assert found, result.stderr
    exit_line = next(line for line in lines if line.startswith("EXIT "))
    return json.loads(found[-1].removeprefix("IMPORTED ")), exit_line


def test_call_imports_no_ai_library(tmp_path: Path) -> None:
    imported, exit_line = _imported(["--project-path", str(tmp_path), "--call", "search_docs",
                                     '{"query": "penalty"}'])
    assert exit_line == "EXIT 0"
    assert imported == []


def test_the_agent_connection_imports_no_ai_library(tmp_path: Path) -> None:
    pytest.importorskip("mcp.server.lowlevel")
    imported, exit_line = _imported(["--mcp", "--project-path", str(tmp_path)])
    assert exit_line == "EXIT 0"                    # stdin is at its end at once: the connection closes
    assert imported == []
