"""The worker thread of one chat tab (P2-D2, §5): everything that may wait — settings, key, data notice, probe,
model requests, tool calls, the conversation store — runs here, never on the Tk thread.

The tab talks to its :class:`AssistantWorker` only through jobs (:meth:`AssistantWorker.submit`) and reads what
happens from :attr:`AssistantWorker.events`, a :class:`queue.SimpleQueue` of ``(job_id | None, event)`` that the
chat window drains with ``after()`` polling.  The worker never touches a Tk object.

A question (:class:`TurnJob`) goes through these steps; each one first checks the job's cancel event, and a
stopped job ends with ``TurnRefused(job_id, "Stopped.")``:

1. the conversation: a new chat gets its number (``store.allocate_chat_number``) and an empty conversation
   (:class:`ChatReady`);
2. the settings, the provider, the address, the model and the key (:class:`Prepared` with the route the model
   name gives) — a missing piece is a :class:`SetupProblem`;
3. the data notice of a remote route (:class:`ConsentNeeded`): the job is *parked* while the inbox goes on, until
   :class:`ConsentAnswer` (a Stop answers ``False``);
4. the probe (an Ollama model may turn out to be forwarded to the cloud: a new :class:`Prepared` and notice);
5. the backend behind the process-wide :class:`~flextool.gui.assistant.gate.ProviderGate` and the session
   (``Session.switch`` when a chat goes on with other settings);
6. the window's snapshot of this question as the tools' defaults, :class:`TurnStarted`, ``Session.run_turn`` (its
   events go straight to the queue), :class:`TurnFinished`.

Steps 2–5 run again only after :class:`SettingsChanged` and only when what the settings give changed, so a change
during an answer applies to the chat's next question.  A chat a model has answered in keeps that provider, address
and model (and its level, unless the settings change the level of that same model); a new chat, or one without an
answer yet, takes the settings' ones.  The level is resolved again from the chat's model at each prepare, before
the probe from the facts last found for it and then from the probe's; the conversation stores only the user's
level.  :class:`Idle` follows every question (also a refused or
failed one); ``busy`` is set from :meth:`AssistantWorker.submit` until then.

:meth:`AssistantWorker.shutdown` stops the answer and ends the thread without waiting: the history is repaired,
saved and released, the backend closed, :attr:`AssistantWorker.exited` set.  Until then the worker is listed by
:func:`finishing_worker`, so a reopen of the same chat can wait for it.
"""
from __future__ import annotations

import dataclasses
import hashlib
import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from flextool.assistant.context import ContextSnapshot
from flextool.gui.assistant import strings
from flextool.gui.assistant.gate import GatedBackend, ProviderGate
from flextool.gui.assistant.gate import GateWait as GateWait  # re-exported: the tab shows the gate's waits

if TYPE_CHECKING:
    from flextool.assistant.endpoint import ConsentKey
    from flextool.assistant.llm_adapter import Backend, Connection, Message, Route
    from flextool.assistant.budget import Budget
    from flextool.assistant.probes import ModelFacts
    from flextool.assistant.providers import Preset
    from flextool.assistant.session import Session, TurnResult
    from flextool.assistant.settings import Settings
    from flextool.assistant.store import Conversation, Store

logger = logging.getLogger(__name__)


# --- jobs (Tk thread → worker) ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TurnJob:
    job_id: int
    text: str
    snapshot: ContextSnapshot | None
    cancel: threading.Event = field(default_factory=threading.Event, compare=False)


@dataclass(frozen=True)
class ConsentAnswer:
    job_id: int
    accepted: bool


@dataclass(frozen=True)
class OpenConversation:
    conversation_id: str


@dataclass(frozen=True)
class DeleteConversation:
    """Delete this tab's own conversation."""


@dataclass(frozen=True)
class SettingsChanged:
    pass


@dataclass(frozen=True)
class Shutdown:
    pass


# --- events (worker → Tk thread), besides every P1 event of the session --------------------------------------------
@dataclass(frozen=True)
class Prepared:
    route: Route
    indicator: str
    provider: str
    model: str
    level: str                      # basic / standard / advanced
    level_source: str               # "chosen" or "from the model name"
    key_source: str
    relaxed: tuple[str, ...]


@dataclass(frozen=True)
class ConsentNeeded:
    job_id: int
    consent_key: ConsentKey
    provider_label: str
    notice: str
    indicator: str


@dataclass(frozen=True)
class ChatReady:
    """A new chat has its conversation and number; *saved* is False when chats are not saved (expiry 0)."""

    conversation_id: str
    number: int | None
    saved: bool


@dataclass(frozen=True)
class TurnStarted:
    job_id: int


@dataclass(frozen=True)
class TurnFinished:
    job_id: int
    result: TurnResult
    totals: dict[str, Any]
    summary: str | None = None      # the "New chat with summary" seed when the conversation is too long


@dataclass(frozen=True)
class TurnRefused:
    job_id: int
    reason: str


@dataclass(frozen=True)
class ConversationLoaded:
    conversation_id: str
    number: int | None
    title: str | None
    messages: tuple[Message, ...]
    notice: str | None
    totals: dict[str, Any]


@dataclass(frozen=True)
class OpenRefused:
    """A saved chat could not be opened: *reason* is ``other_process``, ``this_process``, ``missing`` or
    ``damaged``."""

    conversation_id: str
    reason: str
    message: str


@dataclass(frozen=True)
class Deleted:
    conversation_id: str
    outcome: str                    # "deleted" | "locked_elsewhere" | "missing"


@dataclass(frozen=True)
class SetupProblem:
    message: str
    open_settings: bool


@dataclass(frozen=True)
class WorkerFault:
    message: str


@dataclass(frozen=True)
class Exited:
    pass


@dataclass(frozen=True)
class Idle:
    pass


BackendFactory = Callable[["Preset", "Connection"], "Backend"]
Probe = Callable[..., "ModelFacts"]

#: Error codes whose fix is in the settings ([Open settings] next to the message).
SETTINGS_ERRORS = frozenset({"auth", "no_key", "model", "permission", "quota", "credit", "routing", "tools",
                             "small_context"})

# --- workers that were shut down and are still finishing (a reopen of their chat waits for them) -------------------
_FINISHING: dict[str, AssistantWorker] = {}
_FINISHING_GUARD = threading.Lock()


def finishing_worker(conversation_id: str) -> AssistantWorker | None:
    """The worker of a closed chat that still holds *conversation_id*, if any."""
    with _FINISHING_GUARD:
        worker = _FINISHING.get(conversation_id)
    if worker is not None and worker.exited.is_set():
        return None
    return worker


def finishing_workers() -> list[AssistantWorker]:
    with _FINISHING_GUARD:
        return [w for w in _FINISHING.values() if not w.exited.is_set()]


def repair_history(messages: list[Message]) -> bool:
    """Make *messages* a history every provider accepts: each tool call answered once, right after its call and
    in call order (a missing result becomes "Stopped by the user."), no result without its call.  Changes the
    list in place; returns whether anything changed."""
    from flextool.assistant.llm_adapter import Message
    from flextool.assistant.session import SESSION_MESSAGES, normalise_tool_name

    out: list[Message] = []
    changed = False
    i = 0
    while i < len(messages):
        m = messages[i]
        if m.role == "tool":
            changed = True                  # a result without its call
            i += 1
            continue
        out.append(m)
        i += 1
        if m.role != "assistant" or not m.tool_calls:
            continue
        results: dict[str, Message] = {}
        start = i
        while i < len(messages) and messages[i].role == "tool":
            results.setdefault(messages[i].tool_call_id or "", messages[i])
            i += 1
        expected = [c.id for c in m.tool_calls]
        found = [r.tool_call_id for r in messages[start:i]]
        if found != expected:
            changed = True
        for call in m.tool_calls:
            result = results.get(call.id)
            if result is None:
                result = Message("tool", SESSION_MESSAGES["stopped"], tool_call_id=call.id,
                                 tool_name=normalise_tool_name(call.name), is_error=True, turn=m.turn)
            out.append(result)
    if changed:
        messages[:] = out
    return changed


@dataclass
class _Link:
    """The prepared connection of the chat."""

    fingerprint: tuple
    preset: Preset
    base_url: str
    conn: Connection
    facts: ModelFacts
    backend: GatedBackend
    level: str
    budget: Budget


@dataclass
class _Setup:
    settings: Settings
    preset: Preset
    base_url: str
    conn: Connection
    level: str
    level_source: str
    fingerprint: tuple
    override: str | None = None         # the user's level for this chat (basic / standard / advanced), or None


class _Refused(Exception):
    """The question is not sent; the events are already posted."""


class AssistantWorker:
    """The worker of one chat tab.  *tool_lock* and *gate* are the process-wide ones; *backend_factory*, *probe*
    and *config_root* are the test seams."""

    def __init__(self, project_path: Path, project_name: str, *, gate: ProviderGate,
                 tool_lock: threading.Lock | None = None, config_root: Path | None = None,
                 backend_factory: BackendFactory | None = None, probe: Probe | None = None) -> None:
        from flextool.assistant.context import TOOL_LOCK

        self.project_path = Path(project_path)
        self.project_name = project_name
        self.gate = gate
        self.tool_lock = tool_lock if tool_lock is not None else TOOL_LOCK
        self.config_root = config_root
        self._backend_factory = backend_factory or _default_backend_factory
        self._probe = probe
        self.events: queue.SimpleQueue = queue.SimpleQueue()
        self.exited = threading.Event()
        self.busy = threading.Event()
        self.conversation_id: str | None = None
        self._inbox: queue.SimpleQueue = queue.SimpleQueue()
        self._session: Session | None = None
        self._session_lock = threading.Lock()
        self._conversation: Conversation | None = None
        self._store: Store | None = None
        self._ctx = None
        self._link: _Link | None = None
        self._dirty = False
        self._parked: tuple[TurnJob, Any] | None = None
        self._current_job: TurnJob | None = None
        self._thread: threading.Thread | None = None
        self._shutting_down = False

    # --- the Tk side (any thread) ------------------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="flextool-assistant-chat", daemon=True)
            self._thread.start()

    def submit(self, job: object) -> None:
        if isinstance(job, TurnJob):
            self.busy.set()
        self._inbox.put(job)

    def stop_turn(self, job: TurnJob | None = None) -> None:
        """Stop *job* (default: the running one): its cancel event is set, a parked notice is answered no, and
        the response being read is closed on a short daemon thread, so the caller never waits."""
        job = job or self._current_job
        if job is None:
            return
        job.cancel.set()
        self._inbox.put(ConsentAnswer(job.job_id, False))
        with self._session_lock:
            session = self._session
        if session is not None:
            threading.Thread(target=_abort_quietly, args=(session,), name="flextool-assistant-abort",
                             daemon=True).start()

    def shutdown(self) -> None:
        """Stop the answer and end the worker; never waits."""
        self._shutting_down = True
        if self.conversation_id is not None and not self.exited.is_set():
            with _FINISHING_GUARD:
                _FINISHING[self.conversation_id] = self
        self.stop_turn()
        self._inbox.put(Shutdown())

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # --- the worker thread ----------------------------------------------------------------------------------
    def _emit(self, job_id: int | None, event: object) -> None:
        self.events.put((job_id, event))

    def _run(self) -> None:
        try:
            while True:
                item = self._inbox.get()
                if isinstance(item, Shutdown):
                    break
                try:
                    self._handle(item)
                except Exception as exc:  # noqa: BLE001 — every fault is shown and the history repaired
                    self._fault(exc, item)
        finally:
            self._finish()

    def _handle(self, item: object) -> None:
        if isinstance(item, TurnJob):
            if self._parked is not None:
                # A question while one waits for the notice cannot come from the tab (Send is unavailable).
                self._emit(item.job_id, TurnRefused(item.job_id, strings.NOT_SENT))
                return
            self._current_job = item
            self._advance(item, self._turn_flow(item), None)
        elif isinstance(item, ConsentAnswer):
            if self._parked is not None and self._parked[0].job_id == item.job_id:
                job, flow = self._parked
                self._parked = None
                self._advance(job, flow, item.accepted)
        elif isinstance(item, OpenConversation):
            self._open(item.conversation_id)
        elif isinstance(item, DeleteConversation):
            self._delete()
        elif isinstance(item, SettingsChanged):
            self._dirty = True

    def _advance(self, job: TurnJob, flow: Any, value: object) -> None:
        """Run the question's steps until they end or park for the data notice."""
        try:
            request = flow.send(value)
        except StopIteration:
            self._current_job = None
            self._idle()
            return
        except BaseException:
            self._current_job = None
            raise
        self._parked = (job, flow)
        self._emit(job.job_id, request)

    def _idle(self) -> None:
        self.busy.clear()
        self._emit(None, Idle())

    def _fault(self, exc: BaseException, item: object) -> None:
        logger.error("The AI assistant's chat worker failed", exc_info=exc)
        job_id = item.job_id if isinstance(item, (TurnJob, ConsentAnswer)) else None
        from flextool.assistant.llm_errors import internal_fault_message

        self._emit(job_id, WorkerFault(internal_fault_message(type(exc).__name__)))
        if self._parked is not None:
            self._parked[1].close()
            self._parked = None
        self._current_job = None
        conv = self._conversation
        if conv is not None and self._store is not None:
            try:
                repair_history(conv.messages)
                self._store.save(conv)
            except Exception:  # noqa: BLE001 — the fault is already shown
                logger.exception("The chat could not be saved after the fault")
        if isinstance(item, (TurnJob, ConsentAnswer)) or self.busy.is_set():
            self._idle()

    def _finish(self) -> None:
        try:
            if self._parked is not None:
                job, flow = self._parked
                self._parked = None
                flow.close()
                self._emit(job.job_id, TurnRefused(job.job_id, strings.STOPPED))
            conv = self._conversation
            if conv is not None and self._store is not None:
                if repair_history(conv.messages):
                    self._store.save(conv)
                self._store.close(conv)
            link = self._link
            if link is not None:
                link.backend.close()
        except Exception:  # noqa: BLE001 — the thread ends anyway; the checkpoint on disk stays valid
            if not self._shutting_down:
                logger.exception("The chat worker could not finish cleanly")
        finally:
            self.busy.clear()
            self.exited.set()
            if self.conversation_id is not None:
                with _FINISHING_GUARD:
                    if _FINISHING.get(self.conversation_id) is self:
                        del _FINISHING[self.conversation_id]
            self._emit(None, Exited())

    # --- the conversation -------------------------------------------------------------------------------------
    def _store_open(self) -> Store:
        if self._store is None:
            from flextool.assistant.store import open_store

            self._store = open_store(self.config_root)
        return self._store

    def _context(self):
        if self._ctx is None:
            from flextool.assistant.context import make_context

            ctx = make_context(self.project_path, front_end="gui")
            ctx.project_name = self.project_name
            ctx.lock = self.tool_lock
            self._ctx = ctx
        return self._ctx

    def _ensure_conversation(self) -> Conversation:
        if self._conversation is None:
            store = self._store_open()
            number = store.allocate_chat_number(self.project_path)
            conv = store.new_conversation(self._context(), None, None, number=number)
            self._conversation = conv
            self.conversation_id = conv.id
            self._emit(None, ChatReady(conv.id, number, store.expiry_days() > 0))
        return self._conversation

    def _open(self, conversation_id: str) -> None:
        from flextool.assistant.store import ConversationDamaged, ConversationLocked, ConversationNotFound

        if self._conversation is not None:
            self._emit(None, OpenRefused(conversation_id, "this_process", strings.STILL_FINISHING))
            return
        store = self._store_open()
        try:
            conv = store.open(self.project_path, conversation_id, if_locked="raise")
        except ConversationLocked as exc:
            message = strings.STILL_FINISHING if exc.owner == "this_process" else strings.OPEN_ELSEWHERE
            self._emit(None, OpenRefused(conversation_id, str(exc.owner), message))
            return
        except ConversationNotFound:
            self._emit(None, OpenRefused(conversation_id, "missing", strings.OPEN_MISSING))
            return
        except ConversationDamaged as exc:
            self._emit(None, OpenRefused(conversation_id, "damaged", str(exc)))
            return
        conv.project_name = self.project_name
        if conv.number is None:
            conv.number = store.allocate_chat_number(self.project_path)
        self._conversation = conv
        self.conversation_id = conv.id
        shown = self._bound_event()
        if shown is not None:
            self._emit(None, shown)
        self._emit(None, ConversationLoaded(conv.id, conv.number, conv.title, tuple(conv.messages), conv.notice,
                                            dict(conv.usage)))

    def _delete(self) -> None:
        conv = self._conversation
        if conv is None:
            return
        outcome = self._store_open().delete_conversation(self.project_path, conv.id)
        self._emit(None, Deleted(conv.id, outcome))

    # --- a question -------------------------------------------------------------------------------------------
    def _turn_flow(self, job: TurnJob):
        """The steps of one question (a generator: it yields a :class:`ConsentNeeded` and gets the answer)."""
        jid = job.job_id
        try:
            conv = self._ensure_conversation()
            self._check(job)
            link = self._link
            if link is None or self._dirty:
                setup = self._setup(job)
                if link is None or setup.fingerprint != link.fingerprint:
                    link = yield from self._prepare(job, setup)
                self._dirty = False
            self._check(job)
        except _Refused:
            return
        session = self._session
        assert session is not None
        snapshot = job.snapshot
        ctx = self._context()
        ctx.gui_snapshot = snapshot
        ctx.default_source = snapshot.input_source if snapshot is not None else None
        ctx.default_scenario = snapshot.scenario if snapshot is not None else None
        self._emit(jid, TurnStarted(jid))
        result = session.run_turn(job.text, lambda ev: self._emit(jid, ev), job.cancel)
        summary = None
        if result.error is not None and result.error.code == "context":
            from flextool.assistant.session import carry_summary

            summary = carry_summary(conv.messages)
        self._emit(jid, TurnFinished(jid, result, dict(conv.usage), summary))

    def _check(self, job: TurnJob) -> None:
        if job.cancel.is_set():
            self._emit(job.job_id, TurnRefused(job.job_id, strings.STOPPED))
            raise _Refused

    def _problem(self, job: TurnJob, message: str, open_settings: bool = True) -> _Refused:
        self._emit(job.job_id, SetupProblem(message, open_settings))
        self._emit(job.job_id, TurnRefused(job.job_id, strings.NOT_SENT))
        return _Refused()

    def _setup(self, job: TurnJob) -> _Setup:
        """Step 2: what the settings give, without asking any server.  A chat a model has answered in is held with
        that provider, address and model (:func:`~flextool.assistant.store.bound_origin`), whatever the settings
        name now; a chat without an answer takes the settings' ones.  The level is the user's level for the chat
        (the settings' one when they name the chat's model) and else automatic, resolved again for the model."""
        from flextool.assistant import endpoint, keys
        from flextool.assistant.backends._common import error_context
        from flextool.assistant.llm_adapter import Connection
        from flextool.assistant.llm_errors import NoKey, make_error, message_for
        from flextool.assistant.net import BaseUrlRefused, check_base_url, proxy_for
        from flextool.assistant.providers import PRESETS, model_id
        from flextool.assistant.settings import load_settings

        settings = load_settings(self.config_root)
        self.gate.set_parallel(settings.parallel_requests)
        provider = settings.provider
        bound = self._bound()
        if bound is not None:
            preset_id, raw_url, model = bound
            override = self._conversation.level if self._conversation is not None else None
        else:
            preset_id, raw_url, model = provider.preset or "", "", provider.model or ""
            override = provider.tier
        if not preset_id:
            raise self._problem(job, strings.CHOOSE_PROVIDER)
        preset = PRESETS.get(preset_id)
        if preset is None:
            raise self._problem(job, strings.UNKNOWN_PROVIDER.format(preset=preset_id))
        if bound is None:
            raw_url = provider.base_url or preset.base_url or ""
        if not raw_url:
            raise self._problem(job, strings.NEEDS_ADDRESS.format(provider=preset.label))
        try:
            base_url = check_base_url(raw_url)
        except BaseUrlRefused as exc:
            raise self._problem(job, str(exc)) from None
        if not model:
            raise self._problem(job, strings.CHOOSE_MODEL.format(provider=preset.label))
        if bound is not None and _settings_name(provider, preset, model, base_url):
            override = provider.tier            # the settings name this chat's model: their level holds
        from flextool.assistant.budget import free_tier_on

        if free_tier_on(preset.id, model, settings.free_tier):
            self.gate.set_parallel(1)           # the free plan sends one request at a time (P3 §4.6)
        self._check(job)
        key, source, note = keys.resolve_key(preset, base_url, config_root=self.config_root)
        self._check(job)
        route = endpoint.route_for(preset, base_url, model=model)
        conn = Connection(base_url=base_url, model=model_id(preset.id, model), api_key=key or "",
                          key_source=source.describe(preset) if key else "", knobs=dict(provider.knobs),
                          route=route)
        if preset.key_required and not key:
            err = make_error(NoKey, "no_key", error_context(preset, conn))
            message = message_for(err, "gui")
            raise self._problem(job, f"{message} {note}" if note else message)
        level, level_source = _level_before_probe(preset, base_url, model, override, route)
        fingerprint = (preset.id, base_url, model, override, tuple(sorted(
            (k, repr(v)) for k, v in provider.knobs.items())), conn.key_source,
            hashlib.sha256((key or "").encode("utf-8")).hexdigest()[:16], proxy_for(base_url),
            settings.parallel_requests, settings.free_tier, settings.compaction_every, settings.keep_recent_turns)
        return _Setup(settings, preset, base_url, conn, level, level_source, fingerprint, override)

    def _bound(self) -> tuple[str, str, str] | None:
        from flextool.assistant.store import bound_origin

        conv = self._conversation
        return bound_origin(conv) if conv is not None else None

    def _bound_event(self) -> Prepared | None:
        """What the tab shows for a reopened chat before its next question: its own provider, model, level and
        route (no key, no probe; the question prepares again).  ``None`` for a chat without an answer."""
        from flextool.assistant import endpoint
        from flextool.assistant.net import check_base_url
        from flextool.assistant.providers import PRESETS, relaxed_knobs
        from flextool.assistant.settings import load_settings

        bound = self._bound()
        if bound is None:
            return None
        preset_id, raw_url, model = bound
        preset = PRESETS.get(preset_id)
        if preset is None or not model:
            return None
        try:
            base_url = check_base_url(raw_url)
            settings = load_settings(self.config_root)
        except Exception:  # noqa: BLE001 — the question says what is wrong; the tab shows the settings then
            logger.debug("The chat's own model could not be described", exc_info=True)
            return None
        override = self._conversation.level if self._conversation is not None else None
        if _settings_name(settings.provider, preset, model, base_url):
            override = settings.provider.tier
        route = endpoint.route_for(preset, base_url, model=model)
        level, level_source = _level_before_probe(preset, base_url, model, override, route)
        from flextool.assistant import prompt

        names = {internal: name for name, internal in prompt.LEVELS.items()}
        return Prepared(route=route, indicator=endpoint.indicator_text(route), provider=route.label, model=model,
                        level=names.get(level, level), level_source=level_source, key_source="",
                        relaxed=tuple(relaxed_knobs(preset, settings.provider.knobs)))

    def _prepared_event(self, setup: _Setup, route: Route) -> Prepared:
        from flextool.assistant import endpoint, prompt
        from flextool.assistant.providers import relaxed_knobs

        names = {internal: name for name, internal in prompt.LEVELS.items()}
        return Prepared(route=route, indicator=endpoint.indicator_text(route), provider=route.label,
                        model=setup.conn.model, level=names.get(setup.level, setup.level),
                        level_source=setup.level_source, key_source=setup.conn.key_source,
                        relaxed=tuple(relaxed_knobs(setup.preset, setup.conn.knobs)))

    def _consent(self, job: TurnJob, setup: _Setup, route: Route):
        """Step 3: the data notice of *route*, when it needs one; parks the job until the answer."""
        from flextool.assistant import _files, endpoint

        key = endpoint.consent_key(setup.preset.id, setup.base_url, route)
        if not endpoint.needs_consent(key, self.config_root):
            return
        notice = endpoint.consent_notice(route, required_headers=setup.preset.required_headers)
        accepted = yield ConsentNeeded(job.job_id, key, route.label, notice, endpoint.indicator_text(route))
        if not accepted or job.cancel.is_set():
            reason = strings.STOPPED if job.cancel.is_set() else strings.NOTHING_SENT
            self._emit(job.job_id, TurnRefused(job.job_id, reason))
            raise _Refused
        try:
            endpoint.record_consent(key, self.config_root)
        except (_files.NewerFormat, _files.LockTimeout, OSError) as exc:
            from flextool.assistant.llm_adapter import Notice

            self._emit(job.job_id, Notice(strings.CONSENT_NOT_KEPT.format(reason=exc)))

    def _prepare(self, job: TurnJob, setup: _Setup):
        """Steps 3–5; returns the new link."""
        from flextool.assistant import endpoint, probes, prompt
        from flextool.assistant.llm_errors import LLMError, message_for
        from flextool.assistant.session import tool_specs
        from flextool.assistant.usage import ledger_key

        conn = setup.conn
        self._emit(job.job_id, self._prepared_event(setup, conn.route))
        yield from self._consent(job, setup, conn.route)
        self._check(job)
        needed = probes.needed_tokens(prompt.system_layers("gui", "small"), tool_specs("small"))
        probe = self._probe or probes.probe
        try:
            facts = probe(setup.preset, conn, job.cancel, needed=needed,
                          on_event=lambda ev: self._emit(job.job_id, ev))
        except LLMError as err:
            if err.code == "cancelled" or job.cancel.is_set():
                self._emit(job.job_id, TurnRefused(job.job_id, strings.STOPPED))
                raise _Refused from None
            raise self._problem(job, message_for(err, "gui"), err.code in SETTINGS_ERRORS) from None
        self._check(job)
        from flextool.assistant import tiers
        from flextool.assistant.llm_adapter import Notice

        for note in facts.notes:
            self._emit(job.job_id, Notice(note))
        route = endpoint.route_for(setup.preset, setup.base_url, facts=facts)
        try:
            tier = tiers.resolve_tier(setup.preset, conn.model, facts, setup.override, route)
        except tiers.WindowTooSmall as exc:
            raise self._problem(job, message_for(probes.window_error(setup.preset, conn, exc), "gui")) from None
        level_changed = tier.chosen != setup.level
        setup.level = tier.chosen
        setup.level_source = _level_source(tier, setup.override)
        for line in (tier.notice, tier.window_line()):
            if line:
                self._emit(job.job_id, Notice(line))
        if route != conn.route:
            conn = dataclasses.replace(conn, route=route)
            self._emit(job.job_id, self._prepared_event(setup, route))
            yield from self._consent(job, setup, route)
            self._check(job)
        elif level_changed:
            self._emit(job.job_id, self._prepared_event(setup, route))
        try:
            backend = self._backend_factory(setup.preset, conn)
        except LLMError as err:
            raise self._problem(job, message_for(err, "gui"), err.code in SETTINGS_ERRORS) from None
        gated = GatedBackend(backend, self.gate, ledger_key(setup.preset.id, setup.base_url,
                                                             forwarded_to=route.forwarded_to), label=route.label)
        from flextool.assistant import budget

        free = budget.free_tier_on(setup.preset.id, conn.model, setup.settings.free_tier)
        b = budget.budget_for(tier, free_tier=free, compaction_every=setup.settings.compaction_every,
                              keep_recent_turns=setup.settings.keep_recent_turns)
        link = _Link(setup.fingerprint, setup.preset, setup.base_url, conn, facts, gated, setup.level, b)
        self._attach(link, setup)
        return link

    def _attach(self, link: _Link, setup: _Setup) -> None:
        """Step 5: a new session, or the existing one going on with *link*."""
        from flextool.assistant.session import Session, TurnCaps
        from flextool.assistant.settings import mark_free_tier_suggested
        from flextool.assistant.usage import Ledger

        ctx = self._context()
        b = link.budget
        ctx.caps = b.result_caps
        self._ensure_conversation().level = setup.override     # the user's level, never the resolved one
        caps = TurnCaps.from_budget(b)
        old = self._link
        if self._session is None:
            store = self._store_open()
            config_root = self.config_root
            session = Session(ctx, link.backend, link.preset, conversation=self._ensure_conversation(), caps=caps,
                              tier=link.level, store=None if store.expiry_days() == 0 else store,
                              ledger=Ledger(self.config_root), facts=link.facts, route=link.conn.route, budget=b,
                              free_tier_hint=lambda key: mark_free_tier_suggested(key, config_root))
            with self._session_lock:
                self._session = session
        else:
            self._session.switch(link.backend, link.preset, tier=link.level, facts=link.facts,
                                 route=link.conn.route, caps=caps, budget=b)
        self._link = link
        if old is not None and old.backend is not link.backend:
            old.backend.close()

    # --- for the tests and the window -------------------------------------------------------------------------
    @property
    def session(self) -> Session | None:
        with self._session_lock:
            return self._session


def _settings_name(provider, preset: Preset, model: str, base_url: str) -> bool:
    """Whether the saved *provider* settings name *preset*, *base_url* and *model* (a chat's own)."""
    from flextool.assistant.net import BaseUrlRefused, check_base_url
    from flextool.assistant.providers import model_id

    if provider.preset != preset.id or not provider.model or model_id(preset.id, provider.model) != model:
        return False
    try:
        return check_base_url(provider.base_url or preset.base_url or "") == base_url
    except BaseUrlRefused:
        return False


def _level_source(tier, stored: str | None) -> str:
    """How the window names where the level comes from: "chosen" when the stored level is in use."""
    from flextool.assistant import tiers

    if stored and tiers.internal_level(stored) == tier.chosen:
        return "chosen"
    return "from the model name"


def _level_before_probe(preset: Preset, base_url: str, model: str, stored: str | None,
                        route: Route) -> tuple[str, str]:
    """The level before the probe (P3 §3.3): from the facts FlexTool last found for the model, else the name; a
    stored level above the allowed ones is lowered (the probe's facts decide again)."""
    from flextool.assistant import probes, tiers

    facts = probes.cached_facts(preset.id, base_url, model)
    try:
        tier = tiers.resolve_tier(preset, model, facts, stored, route)
    except tiers.WindowTooSmall:
        tier = tiers.resolve_tier(preset, model, None, stored, route)
    return tier.chosen, _level_source(tier, stored)


def _abort_quietly(session: Session) -> None:
    try:
        session.abort()
    except Exception:  # noqa: BLE001 — closing a response another thread reads may fail; the cancel event holds
        logger.debug("abort of the running answer failed", exc_info=True)


def _default_backend_factory(preset: Preset, conn: Connection) -> Backend:
    from flextool.assistant.llm_adapter import make_backend

    return make_backend(preset, conn)
