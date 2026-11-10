"""A turn checkpointed before each request and resumed with the same requests (calibration fixes 2026-11-10,
item 216: the calibration's ``run --resume`` continues a cell from the request that failed).

``Session(on_checkpoint=…)`` hands over a snapshot (plain JSON data) just before each request is sent;
``restore`` + ``resume_turn`` in a new session send that request again and run the turn on.  The tests:

* T0: a snapshot through JSON and back restores the messages (provider data, kinds, compaction), the turn's state
  and the same request.
* T1 (equivalence): for every request *k* of scripted conversations, run A retries *k* once inside the retries; run B
  stops at *k* (its retries used up) and a new session resumes B's last checkpoint.  B's requests (the stopped
  attempt once) equal A's as Gemini request bodies, byte for byte, and so do the turn results, the notices, the
  stored conversation and the redaction.
* Mutation check (critique M3): each snapshot field, reset to its default on restore, changes what at least one
  scripted case sends or gives (or the restore refuses), except four fields shown not to matter at a checkpoint.
* Key pins (critique M2): the snapshot holds every :class:`_TurnState` field and every session attribute that is
  conversation state; a field added later without a ``SNAPSHOT_FORMAT`` bump fails here.  ``restore`` is strict: a
  key missing or unknown is another format, never a default.
"""
from __future__ import annotations

import copy
import dataclasses
import itertools
import json
import math
import random
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from flextool.assistant import budget as budget_module
from flextool.assistant import llm_errors as E
from flextool.assistant import registry
from flextool.assistant import session as S
from flextool.assistant.context import make_context
from flextool.assistant.llm_adapter import Continuing, Message, Notice, Reply, ToolCall, Usage, make_backend
from flextool.assistant.llm_retry import RetryPolicy
from flextool.assistant.pacing import Pacer
from flextool.assistant.providers import PRESETS, schemas_sent
from flextool.assistant.registry import make_tool
from flextool.assistant.store import Conversation, message_to_json
from flextool.assistant.unit_wording import SESSION_NOTES, UNITS_CONVENTION, add_session_notes
from tests.assistant.llm import fake_backend
from tests.assistant.llm.fake_backend import FakeBackend
from tests.assistant.llm.wire import base_url, connection

PRESET = PRESETS["gemini"]
#: The fake backend's model: the same as the body renderer's, so the signatures are replayed in the bodies.
ORIGIN = ("gemini", base_url("gemini"), "gemma-test")
#: Outside the project: a tool registers it with the redaction (as project.sources does for an external source).
EXTERNAL = Path("/elsewhere/case data/outside.sqlite")
BIG = 1_400                         # characters of a large tool result


@pytest.fixture(scope="module")
def wire():
    """Gemini's Chat Completions backend, only to render the request bodies (it sends nothing)."""
    backend = make_backend(PRESET, connection("gemini", model="gemma-test"))
    yield backend
    backend.close()


@pytest.fixture(autouse=True)
def tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """``echo`` answers ``{"items": [text + size × "x"]}``; ``units`` adds the units note (in full until a request
    carried it, then by key); ``source`` registers :data:`EXTERNAL` with the redaction and names it."""
    def echo(_ctx, args: dict) -> dict:
        return {"items": [args.get("text", "") + "x" * int(args.get("size", 0))]}

    def units(ctx, args: dict) -> dict:
        return add_session_notes({"items": [args.get("text", "")]}, [UNITS_CONVENTION], ctx.notes_sent,
                                 ctx.notes_in_flight)

    def source(ctx, args: dict) -> dict:
        ctx.redactor.add_known(EXTERNAL)
        return {"items": [args.get("text", "")], "path": str(EXTERNAL)}

    props = {"text": {"type": "string", "description": "a text"}, "size": {"type": "integer", "description": "n"}}
    monkeypatch.setattr(registry, "_tools", {
        name: make_tool(name=name, title=name, description=f"The {name} tool.", properties=props, handler=fn,
                        area="orientation", small_tier=True)
        for name, fn in (("echo", echo), ("units", units), ("source", source))})


# --- scripts ------------------------------------------------------------------------------------------------------
_ids = itertools.count()


def call(text: str, name: str = "echo", size: int = 0) -> ToolCall:
    """A call with a Gemini thought signature (``extra_content``), replayed to the same model."""
    n = next(_ids)
    args = {"text": text, "size": size} if size else {"text": text}
    return ToolCall(f"call-{n:05d}", name, args, json.dumps(args),
                    provider_extra={"extra_content": {"google": {"thought_signature": f"sig-{n}"}}})


def calls(*tool_calls: ToolCall, text: str = "") -> Reply:
    return Reply(Message("assistant", text, tool_calls=list(tool_calls)), "tools", None, Usage(100, 0, 0, 10))


def say(text: str) -> Reply:
    return Reply(Message("assistant", text), "end", None, Usage(100, 0, 0, 10))


def reporting(reply: Reply, ratio: float) -> Callable:
    """*reply*, its reported prompt *ratio* × the session's own estimate of the request (a tokenizer that counts
    more than the estimate: the correction rises)."""
    def item(req):
        est = sum(budget_module.request_parts(req, ORIGIN, schemas=schemas_sent(PRESET, req.tool_choice)))
        return dataclasses.replace(reply, usage=Usage(math.ceil(ratio * est), 0, 0, 10))
    return item


def error(cls: type[E.LLMError], key: str, **kw) -> Callable:
    """A fresh error each time (the retries and the session change an error's message)."""
    def item(_req):
        return E.make_error(cls, key, E.ErrorContext(provider="Google Gemini"), **kw)

    item.retried = issubclass(cls, (E.RateLimited, E.ServerBusy))
    return item


def stops(case: Case) -> list[int]:
    """The requests a run is interrupted at: all but those answered by an error the retries retry (a resumed
    request starts a fresh retry budget, while A's retry of *k* has used one of the request's retries)."""
    return [k for k, item in enumerate(case.script) if not getattr(item, "retried", False)]


def busy() -> Callable:
    return error(E.ServerBusy, "busy", retry_after=0.0)



@dataclasses.dataclass
class Case:
    name: str
    tier: str
    questions: list[str]
    script: list
    caps: S.TurnCaps
    budget: object = None
    hint: bool = False
    preset: object = PRESET


def basic_case() -> Case:
    """Basic, three questions.  Turn 1: a units note in full, then by key; a part that reaches its call limit, the
    note (keeps result 2), part 2 repeats an earlier call → the tool-less limit request, whose reply calls a tool
    again (answered "not run").  Turn 2: the external source learned; two continuations, the second note offered
    the result carried into part 2 and the calls run there (part 2 ends at its step limit).  Turn 3: a plain
    answer."""
    caps = dataclasses.replace(S.TurnCaps.for_tier("basic"), max_calls_per_turn=3)
    a1 = call("a", "units")
    turn1 = [calls(a1, call("b"), text="Looking at a and b."), calls(call("c", "units"), call("d")),
             say("Found a, b and c.\nNext: e.\nKeep: 2"),
             calls(call("e")), calls(call("a", "units")),                  # part 2: a repeated call
             calls(call("f"))]                                             # the limit request's reply: calls only
    turn2 = [calls(call("s", "source"), call("g")), calls(call("h"), call("i")),     # i over the call limit
             say("Found s and h.\nNext: more.\nKeep: 1"),
             calls(call("j")), calls(call("k")), calls(call("l")),                  # part 2 at its step limit
             say("Still looking.\nKeep: 1, 3"), say("The second answer.")]
    turn3 = [say("The third answer.")]
    return Case("basic", "basic", ["What is a?", "And g?", "Thanks?"], turn1 + turn2 + turn3, caps)


def recovery_case() -> Case:
    """Advanced, four questions: an overflow recovery that works (turn 2); a rejected signature that comes again
    after the history was sent without signatures (turn 3 stops); an overflow that comes again after the
    compaction (turn 4: still too long)."""
    caps = S.TurnCaps.for_tier("advanced")
    script = [calls(call("a")), say("First."),
              error(E.ContextTooLong, "context"), calls(call("b")), say("Second."),
              error(E.HistoryRejected, "history"), error(E.HistoryRejected, "history"),
              error(E.ContextTooLong, "context"), error(E.ContextTooLong, "context")]
    return Case("recovery", "advanced", ["One?", "Two?", "Three?", "Four?"], script, caps)


def free_case() -> Case:
    """Standard on the free plan, three questions: the free-plan notice once, at the first."""
    b = budget_module.resolve_budget("medium", 131_072, "openrouter", local=False, free_tier=True)
    script = [calls(call("a")), say("First."), say("Second."), say("Third.")]
    return Case("free", "standard", ["One?", "Two?", "Three?"], script, S.TurnCaps.from_budget(b), budget=b)


def hint_case() -> Case:
    """Standard, three questions: turn 1 ends on a rate limit with a named wait (the free-plan hint, asked once
    per session), turn 3 again (no second hint)."""
    limited = error(E.RateLimited, "rpm", retry_after=0.0)
    script = [calls(call("a")), limited, limited, calls(call("b")), say("Second."), limited, limited]
    return Case("hint", "standard", ["One?", "Two?", "Three?"], script, S.TurnCaps.for_tier("standard"),
                hint=True)


def fit_case(window: int) -> Case:
    """Standard with a small window, a provider that replays no data (Mistral: this turn's results are shortened
    oldest first) and a tokenizer that counts more than the estimate (twice, then 1.1 times): the pre-send fit
    shortens results more than once in the turn (one notice)."""
    b = budget_module.resolve_budget("medium", 131_072, "openrouter", local=False)
    b = dataclasses.replace(b, window=window, max_output_tokens=500, fit=True)
    caps = dataclasses.replace(S.TurnCaps.from_budget(b), max_rounds=8, max_output_tokens=500)
    ratios = [2.0, 2.0, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]
    replies = [calls(call(f"r{i}", size=BIG)) for i in range(6)] + [say("Done.")]
    script = [reporting(r, q) for r, q in zip(replies, ratios)]
    return Case("fit", "standard", ["Big?"], script, caps, budget=b, preset=PRESETS["mistral"])


# --- runs ---------------------------------------------------------------------------------------------------------
class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> bool:
        self.t += seconds
        return False


def build(case: Case, script: list, project: Path, *, on_checkpoint=None, validate: bool = True):
    ctx = make_context(project, input_source="base.sqlite", front_end="terminal")
    backend = FakeBackend(script, origin=ORIGIN, validate=validate)
    conv = Conversation(id="cell", project_key="t", project_name=ctx.project_name, project_path=ctx.project_path,
                        created="2026-11-10T00:00:00Z", updated="2026-11-10T00:00:00Z", origin=ORIGIN,
                        level=case.tier)
    clock = Clock()
    session = S.Session(ctx, backend, case.preset, conversation=conv, caps=case.caps, tier=case.tier, store=None,
                        ledger=None, facts=None, budget=case.budget, sleep=clock.sleep, clock=clock.now,
                        rng=random.Random(0), pacer=Pacer(clock=clock.now),
                        retry_policy=RetryPolicy(max_retries=1, free_plan_rate_retries=1, jitter=0.0),
                        free_tier_hint=(lambda _key: True) if case.hint else None, on_checkpoint=on_checkpoint)
    return session, backend


@dataclasses.dataclass
class Outcome:
    requests: list
    results: list
    events: list
    session: S.Session


def run_a(case: Case, k: int, project: Path) -> Outcome:
    """The uninterrupted run: request *k* fails once and is retried inside the retries."""
    session, backend = build(case, case.script[:k] + [busy()] + case.script[k:], project)
    events: list = []
    results = [session.run_turn(q, events.append, threading.Event()) for q in case.questions]
    return Outcome(list(backend.requests), results, events, session)


def run_b(case: Case, k: int, project: Path, mutate: Callable[[dict], None] | None = None) -> Outcome:
    """The interrupted run: request *k* fails twice (the retries used up), the turn stops; a new session restores
    the last checkpoint (through JSON) and resumes.  *mutate* changes the snapshot first."""
    snaps: list[tuple[dict, int]] = []
    events: list = []
    first, backend = build(case, case.script[:k] + [busy(), busy()], project,
                           on_checkpoint=lambda snap: snaps.append((snap, len(events))))
    results = []
    for number, question in enumerate(case.questions):
        result = first.run_turn(question, events.append, threading.Event())
        if result.error is not None and result.error.code == "busy":
            break
        results.append(result)
    else:
        raise AssertionError(f"{case.name}: request {k} did not stop")
    snap, mark = snaps[-1]
    snap = json.loads(json.dumps(snap))
    if mutate is not None:
        mutate(snap)
    fallback = [say(f"fallback {n}") for n in range(20)] if mutate is not None else []
    second, backend2 = build(case, case.script[k:] + fallback, project, validate=mutate is None)
    later: list = []
    second.restore(snap)
    results.append(second.resume_turn(later.append, threading.Event()))
    results += [second.run_turn(q, later.append, threading.Event()) for q in case.questions[number + 1:]]
    # The stopped attempt is sent twice in B (the retry, then the resume), once failed and once retried in A.
    return Outcome(list(backend.requests)[:-1] + list(backend2.requests), results, events[:mark] + later, second)


def result_view(r: S.TurnResult) -> tuple:
    return (r.text, r.ok, r.error.code if r.error else None, r.error.user_message if r.error else None, r.rounds,
            [dataclasses.asdict(u) for u in r.usage], [dataclasses.asdict(q) for q in r.requests], r.calls,
            r.continuations, r.repeated)


def view(o: Outcome, wire) -> dict:
    return {"bodies": [json.dumps(wire.request_body(r), sort_keys=True, ensure_ascii=False) for r in o.requests],
            "results": [result_view(r) for r in o.results],
            "notices": [repr(e) for e in o.events if isinstance(e, (Notice, Continuing))],
            "messages": [message_to_json(m) for m in o.session.messages],
            "usage": dict(o.session.conversation.usage),
            "redacted": o.session.ctx.redact(f"see {EXTERNAL.parent}")}


@pytest.fixture(scope="module")
def cases(wire) -> list[Case]:
    assert wire.origin == ORIGIN
    return [basic_case(), recovery_case(), free_case(), hint_case(), fit_case(FIT_WINDOW)]


#: The fit case's window: the instructions, tools, question and three large results fit at the first
#: tokenizer ratio's correction only with this turn's results shortened (checked in the equivalence test).
FIT_WINDOW = 8_000


# --- T0 -----------------------------------------------------------------------------------------------------------
def test_a_snapshot_through_json_restores_the_messages_state_and_request(cases, tmp_path):
    """T0: the checkpoint before the basic case's fifth request (part 2, after a note and a continuation, with
    thought signatures in the history) restores, through JSON, the same messages, turn state and request."""
    case = cases[0]
    snaps: list = []
    session, _backend = build(case, case.script[:4] + [busy(), busy()], tmp_path / "p",
                              on_checkpoint=snaps.append)
    stopped = session.run_turn(case.questions[0], lambda _e: None, threading.Event())
    assert stopped.error.code == "busy"
    snap = snaps[-1]
    assert snap["state"]["part"] == 2 and snap["pending"]["site"] == "turn"
    restored, _ = build(case, [], tmp_path / "p")
    restored.restore(json.loads(json.dumps(snap)))
    want = [message_to_json(m) for m in Conversation_messages(snap)]
    assert [message_to_json(m) for m in restored.messages] == want
    kinds = [m["kind"] for m in want if "kind" in m]
    assert kinds == ["note_request", "note", "continuation"]
    assert any("provider_extra" in c for m in want for c in m.get("tool_calls", []))
    state, _usage, pending = restored._resumed
    assert S._state_json(state) == snap["state"]
    assert pending == S._Pending("turn", "auto", False, False)
    assert restored.request_sha256(restored._request("auto")) == snap["pending"]["request_sha256"]


def Conversation_messages(snap: dict) -> list[Message]:
    from flextool.assistant.store import message_from_json

    return [message_from_json(m) for m in snap["conversation"]["messages"]]


# --- T1 -----------------------------------------------------------------------------------------------------------
def test_a_resumed_turn_sends_the_same_requests_as_one_retried(cases, wire, tmp_path):
    """T1: for every request k of each case, the interrupted and resumed run sends A's requests byte for byte
    (Gemini bodies, signatures replayed) and gives A's turn results, notices, conversation and redaction."""
    project = tmp_path / "p"
    for case in cases:
        for k in stops(case):
            a = view(run_a(case, k, project), wire)
            b = view(run_b(case, k, project), wire)
            assert b == a, f"{case.name}: interrupted at request {k}"
    # What the cases cover (so a script change cannot quietly drop a path).
    basic = run_a(cases[0], 0, project)
    assert [r.continuations for r in basic.results] == [1, 2, 0] and basic.results[0].repeated
    units = [m.text for m in basic.session.messages if m.tool_name == "units" and not m.is_error]
    note = json.dumps(SESSION_NOTES[UNITS_CONVENTION], ensure_ascii=False)[1:-1]
    assert len(units) == 2 and note in units[0] and f'"{UNITS_CONVENTION}"' in units[1]     # in full, then by key
    assert basic.session.ctx.redact(f"see {EXTERNAL.parent}") == "see <source folder>"
    recovery = run_a(cases[1], 0, project)
    assert [r.error.code if r.error else None for r in recovery.results] == [None, None, "history", "context"]
    free = run_a(cases[2], 0, project)
    assert len([e for e in free.events if isinstance(e, Notice) and e.text.startswith("Free plan:")]) == 1
    hint = run_a(cases[3], 0, project)
    assert [r.error.code if r.error else None for r in hint.results] == ["rpm", None, "rpm"]
    assert len([e for e in hint.events if isinstance(e, Notice) and e.text == S.SESSION_MESSAGES["free_plan_hint"]]) == 1
    fit = run_a(cases[4], 0, project)
    fit_notices = [e for e in fit.events if isinstance(e, Notice) and e.text == S.SESSION_MESSAGES["fit"]]
    assert len(fit_notices) == 1 and sum(m.compacted for m in fit.session.messages) >= 2


# --- the mutation check (critique M3) -----------------------------------------------------------------------------
def _defaults() -> dict[str, object]:
    from flextool.assistant.store import _empty_usage

    out: dict[str, object] = {"format": 0, "turn_no": 0, "last_block": None, "free_notice_shown": False,
                              "hint_asked": False, "correction": {"factor": 1.0, "recent": []}, "notes_sent": [],
                              "known_paths": [], "turn_usage": []}
    out.update({f"conversation.{k}": v for k, v in {"id": "", "origin": None, "system_hash": "", "tools_hash": "",
                                                     "usage": _empty_usage(), "messages": []}.items()})
    out.update({f"state.{k}": v for k, v in S._state_json(S._TurnState()).items()})
    out.update({f"pending.{k}": v for k, v in {"site": "turn", "tool_choice": "auto", "overflow_done": False,
                                                "history_done": False, "request_sha256": "",
                                                "body_sha256": "", "est_input": 0}.items()})
    return out


#: Fields a resumed turn never reads at a checkpoint (each with why); the check asserts they indeed change nothing.
UNOBSERVABLE = {
    # Assigned from the answered request's messages right after its reply, before any tool runs (Session._ask).
    "notes_sent",
    # Read only while the tool-less request is not reached; a checkpoint after the part's limit was hit is the
    # note request (which resets it) or the tool-less request (state.limit_reached is then True for good).
    "state.limit_hit",
    # The conversation's own id: kept for the record, never sent.
    "conversation.id",
    # The whole body's hash with its stream fields, and the request's corrected estimate: for the record (the
    # resume points, the stops), not compared on restore.
    "pending.body_sha256",
    "pending.est_input",
}


def test_every_snapshot_field_is_needed(cases, wire, tmp_path):
    """M3: a snapshot field reset to its default on restore changes the requests, the turn results, the notices,
    the conversation or the redaction of at least one case and request (or the restore refuses it)."""
    project = tmp_path / "p"
    baseline = {(c.name, k): view(run_a(c, k, project), wire) for c in cases for k in stops(c)}
    faults = len(fake_backend.FAULTS)
    defaults = _defaults()
    assert set(defaults) == {*(k for k in S.SNAPSHOT_KEYS if k not in ("conversation", "state", "pending")),
                             *(f"conversation.{k}" for k in S.SNAPSHOT_CONVERSATION_KEYS),
                             *(f"state.{f.name}" for f in dataclasses.fields(S._TurnState)),
                             *(f"pending.{k}" for k in S.SNAPSHOT_PENDING_KEYS)}

    def changed(path: str) -> bool:
        def mutate(snap: dict) -> None:
            node, key = snap, path
            if "." in path:
                head, key = path.split(".")
                node = snap[head]
            node[key] = copy.deepcopy(defaults[path])

        for case in cases:
            for k in stops(case):
                try:
                    got = view(run_b(case, k, project, mutate), wire)
                except S.CheckpointMismatch:
                    return True
                except Exception:   # noqa: BLE001 — a mutated state that breaks the session is a difference too
                    return True
                if got != baseline[(case.name, k)]:
                    return True
        return False

    try:
        found = {path: changed(path) for path in sorted(defaults)}
    finally:
        del fake_backend.FAULTS[faults:]      # the mutated runs' fallback answers are not the test's faults
    assert ", ".join(sorted(p for p, hit in found.items() if not hit and p not in UNOBSERVABLE)) == ""
    assert sorted(p for p in UNOBSERVABLE if found[p]) == []


# --- the key pins (critique M2) -----------------------------------------------------------------------------------
#: The session attributes that are conversation state (in the snapshot), configuration (checked by the hashes and
#: the runner's setting key, or rebuilt from the setting) and the running turn's or the process's (not state).
STATE_ATTRS = {"conversation", "_turn_no", "_last_block", "_free_notice_shown", "_hint_asked", "correction"}
CONFIG_ATTRS = {"ctx", "caps", "store", "ledger", "_compaction_override", "free_tier_hint", "retry_policy", "_sleep",
                "_clock", "_rng", "_pacer", "on_attempt_error", "on_checkpoint", "budget", "compaction_every",
                "keep_recent_turns", "backend", "preset", "tier", "history", "history_turns", "facts", "route",
                "system", "tools", "tool_names", "system_hash", "tools_hash", "prefix_key", "error_ctx", "ledger_key",
                "_paced_by_backend", "pacer", "pace_key"}
RUNTIME_ATTRS = {"_turn_lock", "_cancel", "_live", "_unsaved", "_read_only_notice", "_resumed"}


def test_the_snapshot_keys_are_pinned(cases, tmp_path):
    """M2: a snapshot has exactly the pinned keys, the turn state every :class:`_TurnState` field; every session
    attribute is classified; the data classes the snapshot stores have the pinned fields.  A change here needs a
    ``SNAPSHOT_FORMAT`` bump (and the calibration's ``PARTIAL_FORMAT``)."""
    case = cases[0]
    snaps: list = []
    session, _ = build(case, case.script[:3] + [busy(), busy()], tmp_path / "p", on_checkpoint=snaps.append)
    session.run_turn(case.questions[0], lambda _e: None, threading.Event())
    snap = snaps[-1]
    assert tuple(snap) == S.SNAPSHOT_KEYS and snap["format"] == S.SNAPSHOT_FORMAT == 1
    assert tuple(snap["conversation"]) == S.SNAPSHOT_CONVERSATION_KEYS
    assert tuple(snap["pending"]) == S.SNAPSHOT_PENDING_KEYS
    assert tuple(snap["correction"]) == S.SNAPSHOT_CORRECTION_KEYS
    assert tuple(snap["state"]) == tuple(f.name for f in dataclasses.fields(S._TurnState)) == (
        "calls", "total_calls", "limit_hit", "fit_notice", "requests", "rounds", "part_rounds", "part", "carried",
        "ran", "outcomes", "seen", "tried", "repeated", "limit_reached", "final_text", "note_candidates")
    assert set(vars(session)) == STATE_ATTRS | CONFIG_ATTRS | RUNTIME_ATTRS
    fields = {cls.__name__: tuple(f.name for f in dataclasses.fields(cls))
              for cls in (Message, ToolCall, Usage, S.RequestTokens, S._Carried, budget_module.TokenCorrection)}
    assert fields == {
        "Message": ("role", "text", "tool_calls", "tool_call_id", "tool_name", "is_error", "provider_extra",
                    "origin", "turn", "compacted", "kind"),
        "ToolCall": ("id", "name", "arguments", "raw_arguments", "provider_extra"),
        "Usage": ("input_uncached", "cache_read", "cache_write", "output", "cost", "reported"),
        "RequestTokens": ("text", "tools", "reported"),
        "_Carried": ("label", "text", "ok", "outcome"),
        "TokenCorrection": ("factor", "recent")}


@pytest.mark.parametrize("change", ["missing", "unknown", "state_missing", "message_key", "format", "type"])
def test_restore_is_strict(cases, tmp_path, change):
    """A snapshot of another format is never restored with defaults: a key missing or unknown (also in the turn
    state and a message), another format number or a value of another type is "a checkpoint of another
    format"."""
    case = cases[0]
    snaps: list = []
    session, _ = build(case, case.script[:3] + [busy(), busy()], tmp_path / "p", on_checkpoint=snaps.append)
    session.run_turn(case.questions[0], lambda _e: None, threading.Event())
    snap = json.loads(json.dumps(snaps[-1]))
    if change == "missing":
        del snap["hint_asked"]
    elif change == "unknown":
        snap["new_field"] = 1
    elif change == "state_missing":
        del snap["state"]["seen"]
    elif change == "message_key":
        snap["conversation"]["messages"][0]["mood"] = "calm"
    elif change == "format":
        snap["format"] = 2
    else:
        snap["state"]["rounds"] = "2"
    restored, _ = build(case, [], tmp_path / "p")
    with pytest.raises(S.CheckpointMismatch) as refused:
        restored.restore(snap)
    assert refused.value.reason == "format" and str(refused.value) == "a checkpoint of another format"


def test_restore_refuses_other_instructions_and_a_changed_request(cases, tmp_path):
    """Other instructions or tools, or a history that no longer gives the pending request, are refused (each with
    its reason)."""
    case = cases[0]
    snaps: list = []
    session, _ = build(case, case.script[:3] + [busy(), busy()], tmp_path / "p", on_checkpoint=snaps.append)
    session.run_turn(case.questions[0], lambda _e: None, threading.Event())
    for change, reason in (("tools", "tools"), ("message", "request")):
        snap = json.loads(json.dumps(snaps[-1]))
        if change == "tools":
            snap["conversation"]["system_hash"] = "0" * 16
        else:
            snap["conversation"]["messages"][-1]["text"] += " (tampered)"
        restored, _ = build(case, [], tmp_path / "p")
        with pytest.raises(S.CheckpointMismatch) as refused:
            restored.restore(snap)
        assert refused.value.reason == reason


def test_the_request_hash_is_the_body_without_its_stream_fields(cases, tmp_path):
    """m5: with a backend that has a request body (the Chat Completions one), the checked hash is the body's
    without ``stream`` / ``stream_options``: a resume with ``--no-stream`` restores, while ``body_sha256`` (the
    record's) differs."""
    case = cases[0]

    class Bodied(FakeBackend):
        def __init__(self, script, *, stream: bool) -> None:
            super().__init__(script, origin=ORIGIN)
            self.wire = make_backend(PRESET, dataclasses.replace(connection("gemini", model="gemma-test"),
                                                                 stream=stream))

        def request_body(self, req):
            return self.wire.request_body(req)

        def close(self) -> None:
            self.wire.close()

    def session_with(backend, on_checkpoint=None) -> S.Session:
        ctx = make_context(tmp_path / "p", input_source="base.sqlite", front_end="terminal")
        conv = Conversation(id="cell", project_key="t", project_name=ctx.project_name,
                            project_path=ctx.project_path, created="x", updated="x", origin=ORIGIN, level="basic")
        return S.Session(ctx, backend, PRESET, conversation=conv, caps=case.caps, tier="basic", store=None,
                         ledger=None, facts=None, retry_policy=RetryPolicy(max_retries=0),
                         on_checkpoint=on_checkpoint)

    snaps: list = []
    streamed = Bodied(case.script[:2] + [busy()], stream=True)
    session_with(streamed, snaps.append).run_turn("What is a?", lambda _e: None, threading.Event())
    whole = Bodied(case.script[2:], stream=False)
    resumed = session_with(whole)
    resumed.restore(json.loads(json.dumps(snaps[-1])))
    req = resumed._request(snaps[-1]["pending"]["tool_choice"])
    assert resumed.request_sha256(req) == snaps[-1]["pending"]["request_sha256"]
    assert whole.request_body(req)["stream"] is False and streamed.request_body(req)["stream"] is True
    assert resumed.request_sha256(req) == S.Session.request_sha256(session_with(streamed), req)
    streamed.close()
    whole.close()
