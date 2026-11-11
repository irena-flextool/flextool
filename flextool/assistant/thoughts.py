"""Reasoning a model writes into its answer text, removed from what is shown and kept.

Some models put their reasoning in the message content instead of a reasoning field of their own: Gemma models
behind Google's OpenAI-compatible endpoint write ``<thought>…</thought>``, several local models ``<think>…</think>``.
That text is not part of the answer: it is not shown, not sent back in the history, not stored in the conversation
and not seen by the evaluation's graders.  It is dropped, not kept as provider data; the tokens stay counted, since
the provider counted them.

The rule that keeps a mention of the tag from being taken for a block: an opening tag counts only at the start of a
line (spaces and tabs before it allowed) outside a fenced code block (a line starting with ```` ``` ```` or
``~~~``).  ``<thought>`` inside an inline code span, or anywhere else within a line, stays in the text.  A closing
tag ends its block wherever it is; the whitespace after it is dropped with the block.  A block still open at the end
of the answer is dropped and noted in the log.

:class:`ThoughtFilter` works on streamed pieces: a piece that might be the start of a tag is held back until the
next piece settles it, so no half tag is ever shown.  :func:`stream_visible` wraps one backend request.  An error
after only reasoning was streamed counts as one before any part of the answer was shown, so it can be retried.
"""
from __future__ import annotations

import dataclasses
import logging
import threading
from typing import TYPE_CHECKING

from flextool.assistant.llm_adapter import TextDelta, ToolCallStarted
from flextool.assistant.llm_errors import MESSAGES, LLMError

if TYPE_CHECKING:
    from flextool.assistant.llm_adapter import Backend, OnEvent, Reply, Request

logger = logging.getLogger(__name__)

#: The opening tags and the closing tag of each.
TAGS = {"<thought>": "</thought>", "<think>": "</think>"}
_FENCES = ("```", "~~~")


class ThoughtFilter:
    """Feed the answer's text piece by piece (:meth:`feed`), then :meth:`finish`; each call returns the visible
    text that can be shown now."""

    def __init__(self) -> None:
        self._pending = ""          # text not yet settled
        self._close: str | None = None   # the closing tag of the open block
        self._line_start = True
        self._skip_space = False    # dropping the whitespace after a block
        self._in_fence = False
        self.unclosed = False       # an open block was dropped at the end

    def feed(self, piece: str) -> str:
        self._pending += piece
        return self._run(final=False)

    def finish(self) -> str:
        out = self._run(final=True)
        if self._close is not None:
            self.unclosed = True
            self._close = None
        self._pending = ""
        return out

    def _run(self, *, final: bool) -> str:
        out: list[str] = []
        while self._pending:
            if self._close is not None:
                at = self._pending.find(self._close)
                if at < 0:
                    # Only the end that may be the start of the closing tag is kept.
                    self._pending = "" if final else self._pending[-(len(self._close) - 1):]
                    break
                self._pending = self._pending[at + len(self._close):]
                self._close = None
                self._skip_space = True
                continue
            if self._skip_space:
                rest = self._pending.lstrip()
                self._pending = rest
                if not rest:
                    break
                self._skip_space = False
                self._line_start = True     # the block began a line; what follows begins it now
                continue
            if self._line_start:
                body = self._pending.lstrip(" \t")
                if not body or (not final and _may_become(body)):
                    if final:
                        out.append(self._pending)
                        self._pending = ""
                    break
                if body.startswith(_FENCES):
                    self._in_fence = not self._in_fence
                elif not self._in_fence:
                    tag = next((t for t in TAGS if body.startswith(t)), None)
                    if tag is not None:
                        self._close = TAGS[tag]
                        self._pending = body[len(tag):]
                        continue
                self._line_start = False
            end = self._pending.find("\n")
            if end < 0:
                out.append(self._pending)
                self._pending = ""
                break
            out.append(self._pending[:end + 1])
            self._pending = self._pending[end + 1:]
            self._line_start = True
        return "".join(out)


def _may_become(body: str) -> bool:
    """Whether *body* (the start of a line, without its indent) is too short to say whether it opens a block or a
    fence."""
    return any(len(body) < len(t) and t.startswith(body) for t in (*TAGS, *_FENCES))


def strip_thoughts(text: str) -> str:
    """*text* without its reasoning blocks (an open block at the end is dropped and logged)."""
    f = ThoughtFilter()
    out = f.feed(text) + f.finish()
    if f.unclosed:
        logger.info("An unclosed reasoning block at the end of the answer was dropped")
    return out


def stream_visible(backend: Backend, req: Request, on_event: OnEvent, cancel: threading.Event) -> Reply:
    """``backend.stream`` with the reasoning blocks removed from the text deltas, the reply's text and the text an
    error keeps.  Other events and the usage pass unchanged.  An error after a stream whose text was all reasoning
    and that started no tool call is marked as one before any part of the answer was shown (:func:`_not_shown`):
    the backend counted that text as streamed, but the user saw none of it."""
    f = ThoughtFilter()
    shown = False

    def visible(event: object) -> None:
        nonlocal shown
        if isinstance(event, TextDelta):
            text = f.feed(event.text)
            if text:
                shown = True
                on_event(TextDelta(text))
            return
        if isinstance(event, ToolCallStarted):
            shown = True
        on_event(event)  # type: ignore[arg-type]

    try:
        reply = backend.stream(req, visible, cancel)
    except LLMError as err:
        err.partial_text = strip_thoughts(err.partial_text)
        if err.streamed and not shown and not err.partial_text.strip():
            _not_shown(err)
        raise
    tail = f.finish()
    if tail:
        on_event(TextDelta(tail))
    if reply.message is None:
        return reply
    return dataclasses.replace(reply, message=dataclasses.replace(reply.message,
                                                                  text=strip_thoughts(reply.message.text)))


def _not_shown(err: LLMError) -> None:
    """Mark *err* as an error before any part of the answer was shown; an overload then says so in the words of
    one (the request may be retried, calibration 2026-11-10)."""
    err.streamed = False
    if err.message_key == "busy_streamed":
        key = "busy_local" if err.local else "busy"
        err.user_message = MESSAGES[key].format(**err.fields)
        err.message_key = key
        err.args = (err.user_message,)
