"""A scripted :class:`~flextool.assistant.llm_adapter.Backend` for the session and front-end tests.

Each ``stream`` call takes the next item of the script: a :class:`Reply` (its text is streamed as
:class:`TextDelta` pieces and its calls announced with :class:`ToolCallStarted`), an :class:`LLMError` (raised), a
:class:`Block` (waits until released, cancelled or aborted, then goes on with its ``then`` item) or a callable
``(request) -> item``.  Every request is recorded in ``requests`` and checked with
:func:`~tests.assistant.llm.history_check.validate`.

A request the script has no item for, or one whose history is not valid, is a fault of the test or of the session:
it raises ``AssertionError`` and is also recorded in :data:`FAULTS`, which the assistant tests' autouse fixture
checks at teardown — the session turns any exception into a failed answer (review G11, C5-3), so the raise alone
would pass unseen.
"""
from __future__ import annotations

import dataclasses
import threading
from dataclasses import dataclass, field

from flextool.assistant.llm_adapter import (
    ModelEntry,
    OnEvent,
    Origin,
    Reply,
    Request,
    TextDelta,
    ToolCallStarted,
)
from flextool.assistant.llm_errors import LLMError, cancelled
from tests.assistant.llm import history_check

#: The faults of the fake backends of this test (see the module docstring).
FAULTS: list[str] = []


def _fault(text: str) -> AssertionError:
    FAULTS.append(text)
    return AssertionError(text)


@dataclass
class Block:
    """Wait before answering, until :meth:`release`, the cancel event or :meth:`FakeBackend.abort`."""

    then: object = None
    release: threading.Event = field(default_factory=threading.Event)
    started: threading.Event = field(default_factory=threading.Event)


class FakeBackend:
    def __init__(self, script: list[object], *, origin: Origin = ("fake", "http://127.0.0.1:9/v1", "fake-model"),
                 models: list[ModelEntry] | None = None, piece: int = 7, validate: bool = True) -> None:
        self.script = list(script)
        self.validate = validate
        self.origin = origin
        self.models = list(models or [])
        self.piece = piece
        self.requests: list[Request] = []
        self.aborted = threading.Event()
        self.closed = False
        self._blocks: list[Block] = []

    def stream(self, req: Request, on_event: OnEvent, cancel: threading.Event) -> Reply:
        self.requests.append(req)
        self.aborted.clear()
        if self.validate:
            try:
                history_check.validate(req)
            except AssertionError as exc:
                raise _fault(f"request {len(self.requests)} is not a valid history: {exc}") from None
        if not self.script:
            raise _fault("the fake backend's script is used up")
        item = self.script.pop(0)
        while True:
            if callable(item) and not isinstance(item, (Reply, LLMError, Block)):
                item = item(req)
                continue
            if isinstance(item, Block):
                self._blocks.append(item)
                item.started.set()
                while not item.release.wait(0.01):
                    if cancel.is_set() or self.aborted.is_set():
                        raise cancelled()
                item = item.then
                continue
            break
        if isinstance(item, LLMError):
            raise item
        if not isinstance(item, Reply):
            raise TypeError(f"not a script item: {item!r}")
        if item.message is None:
            return item
        message = dataclasses.replace(item.message, origin=self.origin)
        shown = ""
        for start in range(0, len(message.text), self.piece):
            if cancel.is_set() or self.aborted.is_set():
                raise cancelled(streamed=bool(shown), partial_text=shown)
            piece = message.text[start:start + self.piece]
            shown += piece
            on_event(TextDelta(piece))
        for index, call in enumerate(message.tool_calls):
            on_event(ToolCallStarted(index, call.name))
        return dataclasses.replace(item, message=message)

    def abort(self) -> None:
        self.aborted.set()

    def list_models(self) -> list[ModelEntry]:
        return list(self.models)

    def close(self) -> None:
        self.closed = True

    def release_all(self) -> None:
        for block in self._blocks + [b for b in self.script if isinstance(b, Block)]:
            block.release.set()
        self.aborted.set()
