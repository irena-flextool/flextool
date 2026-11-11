"""Reasoning written into the answer text (``<thought>…</thought>``, ``<think>…</think>``) is not shown, not sent
back and not stored (:mod:`flextool.assistant.thoughts`).

The session runs against :class:`FakeBackend`, which streams a reply's text in pieces of ``piece`` characters: with
``piece=1`` every tag is split at every character, inside the tag name too.  The filter itself is checked with every
two- and three-piece split of the same texts.  The rule for mentions: an opening tag counts only at the start of a
line outside a fenced code block.
"""
from __future__ import annotations

import itertools
import json
import logging
import threading
from pathlib import Path

import pytest

from flextool.assistant.context import make_context
from flextool.assistant.llm_adapter import TextDelta, ToolCallStarted
from flextool.assistant.session import Session, TurnCaps
from flextool.assistant.store import open_store
from flextool.assistant.thoughts import ThoughtFilter, strip_thoughts
from flextool.assistant.usage import Ledger
from tests.assistant.llm.fake_backend import FakeBackend
from tests.assistant.llm.test_session_loop import PRESET, USAGE, Tools, call, calls, say

WHOLE = "<thought>The user is asking about units.</thought>\nThe model has 3 units."
THINK = "<think>\nLet me look.\n</think>\n\nThe model has 3 units."
MID = "First part.\n<thought>check the second</thought>\nSecond part."
UNCLOSED = "The answer is 4.\n<thought>Now I should also"
MENTIONS = ("I thought about it: a `<thought>` tag in the text stays.\nNo <think> here either.\n"
            "```\n<thought>code, kept</thought>\n```\nEnd.")

#: (answer text, visible text): hand-derived from the rule.
CASES = [
    (WHOLE, "The model has 3 units."),
    (THINK, "The model has 3 units."),
    (MID, "First part.\nSecond part."),
    (UNCLOSED, "The answer is 4.\n"),
    (MENTIONS, MENTIONS),
]


@pytest.fixture
def tools(monkeypatch: pytest.MonkeyPatch) -> Tools:
    fake = Tools()
    fake.install(monkeypatch)
    return fake


@pytest.fixture
def ask(tmp_path: Path, tools: Tools, fake_backends, user_config_in_tmp: Path):
    """``ask(script, piece)`` runs one question; returns (result, shown pieces, events, session, backend)."""
    def run(script: list, piece: int):
        ctx = make_context(tmp_path / "project", input_source="base.sqlite", front_end="terminal")
        backend = FakeBackend(script, piece=piece)
        fake_backends.append(backend)
        store = open_store()
        conv = store.new_conversation(ctx, backend.origin, "medium")
        session = Session(ctx, backend, PRESET, conversation=conv, caps=TurnCaps(), tier="medium", store=store,
                          ledger=Ledger(user_config_in_tmp), facts=None, sleep=lambda _s: False)
        events: list = []
        result = session.run_turn("question", events.append, threading.Event())
        shown = [e.text for e in events if isinstance(e, TextDelta)]
        return result, shown, events, session, backend
    return run


def _filtered(pieces: list[str]) -> tuple[list[str], str]:
    f = ThoughtFilter()
    out = [f.feed(p) for p in pieces] + [f.finish()]
    return out, "".join(out)


@pytest.mark.parametrize(("text", "visible"), CASES)
def test_every_two_and_three_piece_split_gives_the_same_visible_text(text, visible):
    assert strip_thoughts(text) == visible
    for cuts in itertools.chain(itertools.combinations(range(1, len(text)), 1),
                                itertools.combinations(range(1, len(text)), 2)):
        bounds = [0, *cuts, len(text)]
        pieces = [text[a:b] for a, b in itertools.pairwise(bounds)]
        out, joined = _filtered(pieces)
        assert joined == visible, pieces
        # What is shown is always a start of the visible text: no half tag is shown and taken back.
        assert all(visible.startswith("".join(out[:i + 1])) for i in range(len(out))), pieces


@pytest.mark.parametrize("piece", [1, 2, 3, 7])
@pytest.mark.parametrize(("text", "visible"), CASES)
def test_the_session_shows_sends_and_stores_only_the_visible_text(ask, text, visible, piece):
    result, shown, _events, session, backend = ask([say(text), say("Next.")], piece)
    assert result.ok and result.text == visible
    assert "".join(shown) == visible
    if visible != MENTIONS:
        assert not any("<th" in s for s in shown)              # not even a half tag reaches the screen
    assert session.messages[-1].text == visible and session.messages[-1].provider_extra == {}
    stored = [m.get("text", "") for m in json.loads(_stored_text(session))["messages"] if m["role"] == "assistant"]
    assert stored == [visible]
    # The next request carries the visible text only.
    session.run_turn("again", lambda _e: None, threading.Event())
    sent = [m.text for m in backend.requests[-1].messages if m.role == "assistant"]
    assert sent == [visible]
    assert result.usage[0].output == USAGE.output                # the provider's count stays


def _stored_text(session: Session) -> str:
    store, conv = session.store, session.conversation
    return (store.project_dir(conv.project_key) / f"{conv.id}.json").read_text()


def test_an_unclosed_block_is_noted_in_the_log_only(ask, caplog):
    with caplog.at_level(logging.INFO, logger="flextool.assistant.thoughts"):
        result, shown, events, _session, _backend = ask([say(UNCLOSED)], 3)
    assert result.text == "The answer is 4.\n"
    assert any("unclosed reasoning block" in r.getMessage() for r in caplog.records)
    assert "Now I should" not in caplog.text                       # the reasoning itself is not logged
    assert not any("unclosed" in str(e).lower() for e in events)   # nor said in the chat


def test_a_thought_only_answer_with_a_tool_call_keeps_the_call_with_empty_text(ask, tools):
    reply = calls(call(text="a"), text="<thought>I need the echo tool.</thought>")
    result, shown, events, session, backend = ask([reply, say("Done.")], 4)
    assert result.ok and result.text == "Done."
    assert tools.ran == [("echo", {"text": "a"})]
    assert "".join(shown) == "Done."
    assert any(isinstance(e, ToolCallStarted) for e in events)
    first = next(m for m in session.messages if m.role == "assistant")
    assert first.text == "" and len(first.tool_calls) == 1
    assert [m.text for m in backend.requests[1].messages if m.role == "assistant"] == [""]


def test_an_error_keeps_only_the_visible_part(ask):
    from flextool.assistant.llm_errors import cancelled

    err = cancelled(streamed=True, partial_text="<thought>half way</thought>\nPartly ans")
    result, _shown, _events, session, _backend = ask([err], 5)
    assert not result.ok
    kept = [m.text for m in session.messages if m.role == "assistant"]
    assert kept == ["Partly ans"]


def test_a_message_without_a_block_is_unchanged():
    text = "Thought: the word alone is fine.\n  indented <thought> mid-line\n"
    assert strip_thoughts(text) == text
