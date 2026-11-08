"""Calls per question, the level's tools, the pre-send fit, the token correction, the truncation check and the
free plan in the agent loop (P3 §8 test 4; §4.0, §4.2, §4.4, §4.6).

The session runs against :class:`FakeBackend`.  Tool results here are rows of 10-significant-digit floats (as
query_result sends them), so their size in tokens is what the fit has to handle.  Every request the backend gets
is checked against the window it must fit, with the session's own estimate and correction.
"""
from __future__ import annotations

import dataclasses
import json
import random
import threading
from pathlib import Path

import pytest

from flextool.assistant import budget, registry
from flextool.assistant.budget import request_tokens, resolve_budget
from flextool.assistant.context import make_context
from flextool.assistant.llm_adapter import Message, Notice, Reply, ToolCall, Usage
from flextool.assistant.llm_errors import ErrorContext, RateLimited, make_error
from flextool.assistant.probes import ModelFacts
from flextool.assistant.providers import PRESETS
from flextool.assistant.registry import make_tool
from flextool.assistant.session import SESSION_MESSAGES, Session, TurnCaps
from flextool.assistant.pacing import Pacer
from flextool.assistant.settings import load_settings, mark_free_tier_suggested
from flextool.assistant.store import open_store
from flextool.assistant.usage import Ledger
from tests.assistant.llm import history_check
from tests.assistant.llm.fake_backend import FakeBackend

LIMIT = SESSION_MESSAGES["limit"]
SMALL_USAGE = Usage(100, 0, 0, 10)
_ids = iter(f"cap{n:05d}" for n in range(100_000))


def call(name: str = "rows", **args) -> ToolCall:
    raw = json.dumps(args)
    return ToolCall(next(_ids), name, args, raw)


def calls(*tool_calls: ToolCall, usage: Usage = SMALL_USAGE) -> Reply:
    return Reply(Message("assistant", "", tool_calls=list(tool_calls)), "tools", None, usage)


def say(text: str, usage: Usage = SMALL_USAGE) -> Reply:
    return Reply(Message("assistant", text), "end", None, usage)


class Tools:
    """``rows`` (basic): float rows, as many as fit the caps; ``deep`` (not basic)."""

    def __init__(self) -> None:
        self.ran: list[str] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def rows(_ctx, args: dict):
            self.ran.append("rows")
            rng = random.Random(int(args.get("seed", 0)))
            data = [[f"t{i:04d}", float(f"{rng.uniform(-900, 900):.10g}")] for i in range(1, 2_001)]
            return registry.Fit({"rows": data, "total_rows": len(data)}, rows_key="rows")

        def deep(_ctx, _args: dict) -> dict:
            self.ran.append("deep")
            return {"ok": True}

        props = {"seed": {"type": "integer"}}
        monkeypatch.setattr(registry, "_tools", {
            "rows": make_tool(name="rows", title="rows", description="Rows of floats.", properties=props,
                              handler=rows, area="results", small_tier=True),
            "deep": make_tool(name="deep", title="deep", description="A deeper tool.", properties={},
                              handler=deep, area="diagnostics", small_tier=False),
        })


@pytest.fixture
def tools(monkeypatch: pytest.MonkeyPatch) -> Tools:
    fake = Tools()
    fake.install(monkeypatch)
    return fake


def session_for(tmp_path: Path, script: list, *, tier: str = "small", preset_id: str = "ollama",
                b: budget.Budget | None = None, facts: ModelFacts | None = None, every: int | None = None,
                hint=None, caps: TurnCaps | None = None) -> tuple[Session, FakeBackend, list]:
    ctx = make_context(tmp_path / "project", front_end="terminal")
    clock = FakeClock()
    backend = FakeBackend(script)
    if b is not None:
        ctx.caps = b.result_caps
    conv = open_store().new_conversation(ctx, backend.origin, tier)
    session = Session(ctx, backend, PRESETS[preset_id], conversation=conv,
                      caps=caps or (TurnCaps.from_budget(b) if b is not None else TurnCaps.for_tier(tier)),
                      tier=tier, store=None, ledger=Ledger(tmp_path / "config"), facts=facts, budget=b,
                      compaction_every=every, free_tier_hint=hint, sleep=clock.sleep, clock=clock.now,
                      pacer=Pacer(clock=clock.now))
    return session, backend, []


class FakeClock:
    """A clock that only the waits move: retries and pacing take no real time."""

    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> bool:
        self.t += seconds
        return False


def ask(session: Session, events: list, text: str = "question"):
    return session.run_turn(text, events.append, threading.Event())


def check_fit(session: Session, backend: FakeBackend, window: int) -> None:
    """Every request sent fits 95 % of *window* with its answer, by the session's own estimate."""
    for req in backend.requests:
        history_check.validate(req)
        assert request_tokens(req, session.correction) + req.max_output_tokens <= 0.95 * window


# --- calls per question and the level's tools -------------------------------------------------------------------
def test_the_fourth_call_gets_the_limit_and_then_no_tools(tmp_path, tools):
    """Basic: three calls a question (P3 §4.1).  Two rounds of two calls: the fourth is refused, and in the last
    part (no continuation left, P3 §4.7) the next request is the tool-less one."""
    caps = dataclasses.replace(TurnCaps.for_tier("small"), max_continuations=0)
    session, backend, events = session_for(tmp_path, [
        calls(call(seed=1), call(seed=2)), calls(call(seed=3), call(seed=4)), say("What I found.")], caps=caps)
    result = ask(session, events)
    assert result.ok and result.text == "What I found." and result.calls == 3
    assert tools.ran == ["rows"] * 3
    third = backend.requests[2]
    results = [m for m in third.messages if m.role == "tool"]
    assert [m.is_error for m in results] == [False, False, False, True]
    assert results[3].text == "Not run: the limit of 3 tool calls for this question is reached."
    assert third.tool_choice == "none" and third.messages[-1].text == LIMIT
    assert [r.tool_choice for r in backend.requests[:2]] == ["auto", "auto"]


def test_a_tool_of_another_level_is_refused_before_the_registry(tmp_path, monkeypatch):
    """get_slack_summary at basic: the real tool set, and the registry is never asked."""
    asked: list[str] = []
    real = registry.call_tool
    monkeypatch.setattr(registry, "call_tool", lambda ctx, name, args: asked.append(name) or real(ctx, name, args))
    session, backend, events = session_for(tmp_path, [
        calls(call("get_slack_summary", run="coal_1")), say("I cannot trace that at this level.")])
    assert ask(session, events).ok
    result = [m for m in backend.requests[1].messages if m.role == "tool"][0]
    assert result.is_error and result.text == (
        "The tool get_slack_summary is not available at this assistant level. Available tools: find_result_tables, "
        "get_entity, get_parameter, list_runs, list_sources, query_result, read_doc, search_docs.")
    assert asked == []
    assert [t.name for t in backend.requests[0].tools] == [
        "find_result_tables", "get_entity", "get_parameter", "list_runs", "list_sources", "query_result", "read_doc",
        "search_docs"]


# --- the pre-send fit -------------------------------------------------------------------------------------------
def test_the_fit_at_16k_shortens_earlier_results_and_never_exceeds_the_window(tmp_path, tools):
    """Advanced (the full history, P3 §4.7) at a 16 384-token loaded context, with the turn-boundary compaction
    switched off (every 100 questions): the fit itself shortens earlier turns' results, once with a notice, and no
    request exceeds 95 % of the window.  (Basic sends no earlier results at all.)"""
    window = 16_384
    b = resolve_budget("large", window, "loaded", local=True)
    script = []
    for turn in range(4):
        script += [calls(call(seed=10 * turn + 1), call(seed=10 * turn + 2)), calls(call(seed=10 * turn + 3)),
                   say(f"Answer {turn}.")]
    session, backend, events = session_for(tmp_path, script, tier="large", b=b,
                                           facts=ModelFacts(loaded_context=window), every=100)
    for turn in range(4):
        assert ask(session, events, f"question {turn}").ok
    check_fit(session, backend, window)
    notices = [e.text for e in events if isinstance(e, Notice)]
    assert SESSION_MESSAGES["fit"] in notices
    assert any(m.compacted for m in session.messages if m.role == "tool")


def test_a_question_too_long_for_the_window_is_never_sent(tmp_path, tools):
    window = 16_384
    b = resolve_budget("small", window, "loaded", local=True)
    session, backend, events = session_for(tmp_path, [say("never")], b=b, facts=ModelFacts(loaded_context=window))
    result = ask(session, events, "x" * 60_000)
    assert not result.ok and result.error is not None and result.error.code == "context"
    assert backend.requests == []


def _many_results_script() -> list:
    return [calls(*(call(seed=s) for s in range(8))), say("With what I have.")]


def test_a_provider_without_replayed_data_has_this_turns_results_shortened(tmp_path, tools):
    """Ollama (no replayed data): eight results of one step do not fit 16 384 tokens with standard's output limit;
    this turn's results are shortened oldest first, the newest kept, and the request is sent."""
    window = 16_384
    b = resolve_budget("medium", window, "loaded", local=True)
    session, backend, events = session_for(tmp_path, _many_results_script(), tier="medium", b=b,
                                           facts=ModelFacts(loaded_context=window))
    result = ask(session, events)
    assert result.ok and len(backend.requests) == 2
    check_fit(session, backend, window)
    second = [m for m in backend.requests[1].messages if m.role == "tool"]
    assert second[-1].compacted is False and second[0].compacted is True
    flags = [m.compacted for m in second]
    assert flags == sorted(flags, reverse=True)          # oldest first
    assert backend.requests[1].tool_choice == "auto"


def test_a_provider_with_replayed_data_keeps_this_turn_and_answers_with_what_it_has(tmp_path, tools):
    """Gemini-like (thought signatures replayed): this turn is never edited; the request becomes the tool-less one
    (step 3) and, still too long, is not sent (step 4)."""
    window = 16_384
    b = resolve_budget("medium", window, "model", local=False)
    session, backend, events = session_for(tmp_path, _many_results_script(), tier="medium", preset_id="gemini",
                                           b=b, facts=ModelFacts(context_window=window))
    result = ask(session, events)
    assert not result.ok and result.error.code == "context"
    assert len(backend.requests) == 1                    # nothing sent after the results
    assert not any(m.compacted for m in session.messages)
    # The tool-less request was not sent either, so the history ends with this turn's eight results.
    assert [m.role for m in session.messages[-8:]] == ["tool"] * 8


def test_the_correction_from_reported_counts_tightens_the_fit(tmp_path, tools):
    """A provider reporting three times the estimated prompt: the same question's step with three results is
    shortened, where without the report it is sent as it is."""
    window = 16_384

    def run(reported: int | None) -> tuple[Session, FakeBackend, list]:
        b = resolve_budget("small", window, "loaded", local=True)
        usage = Usage(reported, 0, 0, 10) if reported is not None else SMALL_USAGE
        script = [calls(call(seed=1), call(seed=2), call(seed=3), usage=usage), say("Answer.")]
        # The loaded context is larger than the budget's window, so the reports are not a truncation.
        session, backend, events = session_for(tmp_path, script, b=b, facts=ModelFacts(loaded_context=4 * window))
        assert ask(session, events).ok
        return session, backend, events

    plain, plain_backend, plain_events = run(None)
    assert plain.correction.factor == 1.0
    assert SESSION_MESSAGES["fit"] not in [e.text for e in plain_events if isinstance(e, Notice)]
    assert not any(m.compacted for m in plain_backend.requests[1].messages)
    estimate = sum(budget.request_parts(plain_backend.requests[0]))
    tight, tight_backend, tight_events = run(3 * estimate)
    assert tight.correction.factor == pytest.approx(3.0)
    assert SESSION_MESSAGES["fit"] in [e.text for e in tight_events if isinstance(e, Notice)]
    assert any(m.compacted for m in tight_backend.requests[1].messages)
    check_fit(tight, tight_backend, window)



def test_a_prompt_that_fills_the_loaded_context_is_an_overflow(tmp_path, tools):
    """Ollama cut the prompt silently: a reported prompt of 99 % of the loaded context is not an answer."""
    loaded = 16_384
    b = resolve_budget("small", loaded, "loaded", local=True)
    session, backend, events = session_for(tmp_path, [
        say("An answer from a cut prompt.", Usage(16_250, 0, 0, 10)), say("The real answer.")],
        b=b, facts=ModelFacts(loaded_context=loaded))
    result = ask(session, events)
    assert result.ok and result.text == "The real answer."
    assert len(backend.requests) == 2
    assert any(isinstance(e, Notice) and "too long" in e.text for e in events)
    assert "An answer from a cut prompt." not in [m.text for m in session.messages]
    # 16 219 (below 99 % = 16 220.16) is an answer.
    session, backend, events = session_for(tmp_path, [say("Kept.", Usage(16_219, 0, 0, 10))], b=b,
                                           facts=ModelFacts(loaded_context=loaded))
    assert ask(session, events).text == "Kept." and len(backend.requests) == 1


# --- the free plan ----------------------------------------------------------------------------------------------
def _rate_limited(retry_after: float | None = None) -> RateLimited:
    return make_error(RateLimited, "rpm", ErrorContext(preset_id="gemini", provider="Google Gemini", model="m"),
                      retry_after=retry_after)


def test_a_rate_limit_suggests_the_free_plan_once_per_route(tmp_path, tools):
    """The hint comes only when a limit with a named wait (requests per minute or day) ends the question; a limit
    the retries get past, or one without a named wait (speed, which the pacing handles), shows none."""
    root = tmp_path / "config"
    hint = lambda key: mark_free_tier_suggested(key, root)       # noqa: E731
    b = resolve_budget("medium", 1_000_000, "model", local=False)
    session, backend, events = session_for(tmp_path, [_rate_limited(0.01), say("ok")], tier="medium",
                                           preset_id="gemini", b=b, hint=hint)
    assert ask(session, events).ok
    assert not [e for e in events if isinstance(e, Notice) and e.text == SESSION_MESSAGES["free_plan_hint"]]
    for expected in (1, 0):
        session, backend, events = session_for(tmp_path, [_rate_limited(0.01) for _ in range(4)], tier="medium",
                                               preset_id="gemini", b=b, hint=hint)
        assert not ask(session, events).ok
        shown = [e for e in events if isinstance(e, Notice) and e.text == SESSION_MESSAGES["free_plan_hint"]]
        assert len(shown) == expected
    assert load_settings(root).free_tier_suggested == [session.ledger_key]
    # With the free plan on there is no hint, and the first question says what the plan does.
    free = resolve_budget("medium", 1_000_000, "model", local=False, free_tier=True)
    session, backend, events = session_for(tmp_path, [_rate_limited(), say("ok"), say("again")], tier="medium",
                                           preset_id="openrouter", b=free, hint=lambda key: True)
    assert ask(session, events).ok and ask(session, events).ok
    texts = [e.text for e in events if isinstance(e, Notice)]
    assert SESSION_MESSAGES["free_plan_hint"] not in texts
    assert texts.count("Free plan: the assistant uses at most 4 steps for each question to stay within the "
                       "provider's request limits.") == 1
