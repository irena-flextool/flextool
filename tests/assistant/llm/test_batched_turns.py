"""Asking together (batching 2026-11-08): a basic question answered with one get_parameter call of lists, the
budget's steps and calls, and batched results at a 16 384-token window.

The session runs against :class:`FakeBackend` with the real tools on the assistant test project (built from
``tests/fixtures/tests.json``).  Fixture facts (tests.json, scenario ``all``; a later alternative wins): unit
coal_plant existing 2000 (retire over coal), invest_cost 400 (coal_invest over retire), availability unset (the
default 1.0); unit coal_chp existing 1000 (coal_chp over coal), invest_cost unset (no default), availability unset
(1.0).
"""
from __future__ import annotations

import json
import threading

from flextool.assistant.budget import request_tokens, resolve_budget
from flextool.assistant.context import make_context
from flextool.assistant.llm_adapter import Message, Reply, ToolCall, Usage
from flextool.assistant.probes import ModelFacts
from flextool.assistant.providers import PRESETS
from flextool.assistant.session import SESSION_MESSAGES, Session, TurnCaps
from flextool.assistant.store import open_store
from flextool.assistant.usage import Ledger
from tests.assistant.llm import history_check
from tests.assistant.llm.fake_backend import FakeBackend

USAGE = Usage(100, 0, 0, 10)
_ids = iter(f"batch{n:05d}" for n in range(100_000))
QUESTION = "What are the existing capacity, investment cost and availability of coal_plant and coal_chp in all?"
LOOKUP = {"entity_class": "unit", "parameters": ["existing", "invest_cost", "availability"],
          "entities": [["coal_plant"], ["coal_chp"]], "scenario": "all"}


def call(name: str, args: dict) -> ToolCall:
    return ToolCall(next(_ids), name, args, json.dumps(args))


def calls(*tool_calls: ToolCall) -> Reply:
    return Reply(Message("assistant", "", tool_calls=list(tool_calls)), "tools", None, USAGE)


def say(text: str) -> Reply:
    return Reply(Message("assistant", text), "end", None, USAGE)


def session_for(project, script: list, *, caps: TurnCaps, budget=None, facts=None) -> tuple[Session, FakeBackend]:
    ctx = make_context(project, front_end="terminal")
    if budget is not None:
        ctx.caps = budget.result_caps
    backend = FakeBackend(script)
    conv = open_store().new_conversation(ctx, backend.origin, "small")
    session = Session(ctx, backend, PRESETS["ollama"], conversation=conv, caps=caps, tier="small", store=None,
                      ledger=Ledger(project.parent / "ledger"), facts=facts, budget=budget, sleep=lambda _s: False)
    return session, backend


def ask(session: Session, text: str = QUESTION):
    return session.run_turn(text, lambda _e: None, threading.Event())


def test_basic_answers_three_parameters_of_two_units_in_one_step(assistant_project):
    """One get_parameter call with lists answers the six values; the second request is the answer."""
    session, backend = session_for(assistant_project, [calls(call("get_parameter", LOOKUP)), say("Answer.")],
                                   caps=TurnCaps.for_tier("basic"))
    result = ask(session)
    assert result.ok and result.rounds == 2 and result.calls == 1
    assert len(backend.requests) == 2
    tool_results = [m for m in backend.requests[1].messages if m.role == "tool"]
    assert len(tool_results) == 1 and not tool_results[0].is_error
    table = json.loads(tool_results[0].text)
    values = {(r[0][0], r[1]): (r[2], r[4], r[5]) for r in table["rows"]}
    assert values == {
        ("coal_plant", "existing"): (2000.0, "retire", False),
        ("coal_plant", "invest_cost"): (400.0, "coal_invest", False),
        ("coal_plant", "availability"): (1.0, None, True),
        ("coal_chp", "existing"): (1000.0, "coal_chp", False),
        ("coal_chp", "invest_cost"): (None, None, False),
        ("coal_chp", "availability"): (1.0, None, True)}
    for req in backend.requests:
        history_check.validate(req)


def test_basic_stops_after_three_steps(assistant_project):
    """One call a step: the fourth request is the tool-less one (three steps, no continuation left)."""
    lookups = [dict(LOOKUP, parameters=[p]) for p in ("existing", "invest_cost", "availability")]
    caps = TurnCaps(max_rounds=3, max_calls_per_round=3, max_calls_per_turn=6, max_continuations=0)
    session, backend = session_for(assistant_project, [calls(call("get_parameter", a)) for a in lookups]
                                   + [say("With what I have.")], caps=caps)
    result = ask(session)
    assert result.ok and result.calls == 3 and len(backend.requests) == 4
    assert [r.tool_choice for r in backend.requests] == ["auto", "auto", "auto", "none"]
    assert backend.requests[-1].messages[-1].text == SESSION_MESSAGES["limit"]


def test_basic_runs_at_most_six_calls_a_part(assistant_project):
    """Three steps of three calls: six run, the last step's three are refused by the part's limit."""
    one = [call("get_parameter", dict(LOOKUP, parameters=[p])) for p in ("existing", "invest_cost", "availability")]
    caps = TurnCaps(max_rounds=3, max_calls_per_round=3, max_calls_per_turn=6, max_continuations=0)
    script = [calls(*one), calls(*[call(c.name, c.arguments) for c in one]),
              calls(*[call(c.name, c.arguments) for c in one]), say("With what I have.")]
    session, backend = session_for(assistant_project, script, caps=caps)
    result = ask(session)
    assert result.ok and result.calls == 6
    results = [m for m in backend.requests[-1].messages if m.role == "tool"]
    assert [m.is_error for m in results] == [False] * 6 + [True] * 3
    assert results[-1].text == "Not run: the limit of 6 tool calls for this question is reached."


def test_batched_results_fit_a_16k_window(assistant_project):
    """Basic at a 16 384-token loaded context: three batched calls in one step (the part's three calls at this
    window), every request within 95 % of the window by the session's own estimate, nothing shortened."""
    window = 16_384
    b = resolve_budget("small", window, "loaded", local=True)
    assert (b.level.max_calls_per_turn, b.result_caps.max_chars) == (3, 2_000)
    lookups = [LOOKUP, dict(LOOKUP, scenario="coal"), dict(LOOKUP, scenario="coal_chp")]
    session, backend = session_for(assistant_project,
                                   [calls(*(call("get_parameter", a) for a in lookups)), say("Answer.")],
                                   caps=TurnCaps.from_budget(b), budget=b, facts=ModelFacts(loaded_context=window))
    result = ask(session)
    assert result.ok and result.calls == 3
    second = backend.requests[1]
    results = [m for m in second.messages if m.role == "tool"]
    assert len(results) == 3 and not any(m.is_error or m.compacted for m in results)
    assert all(len(m.text) <= 2_000 for m in results)
    for req in backend.requests:
        history_check.validate(req)
        assert request_tokens(req, session.correction) + req.max_output_tokens <= 0.95 * window
