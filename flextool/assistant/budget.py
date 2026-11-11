"""Token budgets of the chat (P3 §4): how large a request may get, how many steps and tool calls a question may
use, how much one tool result may hold, and how often earlier results are shortened — by level and by the model's
context window.

**Estimates** (:func:`approx_tokens`): digits and non-ASCII characters count one token each, the rest one token per
2.2 characters in JSON and per 3.2 in text, plus 4.  Tokenizers that give every digit its own token (Qwen,
Mistral, Gemma) make result JSON with floats about one token per 1.4 characters; the digit-aware form stays above
them without a tokenizer (checked against Qwen2.5 and o200k counts of a committed result fixture).  Provider data a
request replays to the model that produced it (thinking blocks, reasoning details, thought signatures) counts as
JSON, as the backend sends it; Ollama's reasoning text counts as text, and only on the messages after the last user
message, where the backend replays it (an estimate that is too high is never corrected down).  A per-session :class:`TokenCorrection` raises the estimates when a provider reports
more prompt tokens than estimated: by the largest ratio of the last :data:`CORRECTION_WINDOW` requests (never below
1.0).

**Budgets** (:data:`LEVEL_BUDGETS`, the free plan's in :data:`FREE_TIER_BUDGETS`) are safety caps, not usage
limits; :func:`resolve_budget` combines a level with the window (P3 §3.3):

| window | result caps | shortened every / keep |
|---|---|---|
| 12 288 – 32 767 | :func:`fit_caps`, at most the small preset | 1 / 0 |
| 32 768 – 65 535 | small | 4 / 2 |
| 65 536 – 199 999 | the level's preset, at most medium | 6 / 2 |
| 200 000 or more | the level's preset | 6 / 2 |

In the chat the result caps are both the defaults and the most a call may ask for.  Before every request whose
window is known (or the assumed local one) the session checks :func:`fits`: the request and the answer must stay
within 95 % of the window.  Standard library only (and :mod:`flextool.assistant.caps`, the key name of
:mod:`flextool.assistant.llm_adapter`).

**History and continuations** (P3 §4.7): a level's ``history`` says what each request carries of the earlier
questions — ``"answers"``: each earlier question with its final answer as plain text (at most ``history_turns`` of
them; ``None`` = all), ``"full"``: the whole conversation (with the compaction above).  A question whose step or tool
call limit is reached goes on with a fresh context at most ``max_continuations`` times.

| | basic | standard | advanced |
|---|---|---|---|
| steps / calls a part / calls a step | 3 / 6 / 3 | 8 / 16 / 8 | 12 / 30 / 8 |
| the same with the free plan | 3 / 6 / 3 | 4 / 16 / 8 | 6 / 30 / 8 |
| history | the last 2 answers | every answer | full |
| continuations (free plan) | 2 (1) | 1 (0) | 0 (0) |

Every part of a question has the level's steps and calls again.  One request carries at most a part's calls'
results (the last step's request, or the note request of a continuation), so :func:`fit_caps` plans for that many
(``planned_calls``: 6 / 6 / 8).  Below 32 768 tokens a part's calls are also limited to the results at the cap that
fit 95 % of the window (:func:`fitting_calls`; never fewer than a step's calls): at 16 384 tokens basic has three
calls a part with 2 000-character results, from about 22 000 tokens six.
"""
from __future__ import annotations

import dataclasses
import json
import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from flextool.assistant import caps as caps_module
from flextool.assistant.llm_adapter import ANTHROPIC_BLOCKS_KEY, REASONING_TEXT_KEY
from flextool.assistant.tiers import ASSUMED_LOCAL_WINDOW, ASSUMED_REMOTE_WINDOW, MIN_WINDOW

if TYPE_CHECKING:
    from flextool.assistant.llm_adapter import Message, Origin, Request, ToolSpec, Usage
    from flextool.assistant.tiers import TierChoice

#: Characters per token of the characters that are neither digits nor non-ASCII.  P3 §4.0 started JSON at 2.5;
#: Qwen2.5 gives result rows' punctuation and names 2.23 characters a token, so 2.2 keeps the estimate above it.
JSON_CHARS_PER_TOKEN = 2.2
TEXT_CHARS_PER_TOKEN = 3.2
#: Tokens added to every estimate (message framing).
ESTIMATE_OVERHEAD = 4
#: Share of the window a request and its answer may use (P3 §4.4).
FIT_SHARE = 0.95
#: The margin and the text history that :func:`fit_caps` keeps free.
FIT_MARGIN_SHARE = 0.10
HISTORY_TOKENS_MAX = 4_000
HISTORY_SHARE = 0.15
#: Characters per token of result JSON at its densest, as the estimate counts it (P3 §4.3).
RESULT_CHARS_PER_TOKEN = 1.5
MIN_RESULT_CHARS = 2_000
#: The token correction is the largest reported / estimated ratio of this many latest requests (review G11, C5-6):
#: a request that carried much replayed data no longer raises the estimates for the rest of the session.
CORRECTION_WINDOW = 4


# --- estimates ------------------------------------------------------------------------------------------------
def approx_tokens(text: str, kind: Literal["text", "json"] = "text") -> int:
    """About how many tokens *text* takes (P3 §4.0): digits and non-ASCII characters one token each, the rest
    / 2.2 (``json``) or / 3.2 (``text``), plus 4."""
    dense = sum(1 for ch in text if ch.isdigit() or ord(ch) > 127)
    rest = len(text) - dense
    per = JSON_CHARS_PER_TOKEN if kind == "json" else TEXT_CHARS_PER_TOKEN
    return dense + math.ceil(round(rest / per, 6)) + ESTIMATE_OVERHEAD      # round: 22 / 2.2 is 10, not 10.000…02


def reported_prompt_tokens(usage: Usage | None) -> int | None:
    """The prompt tokens a provider reported for one request (uncached + cache read + cache write), ``None`` when
    it reported none."""
    if usage is None or not usage.reported:
        return None
    parts = [n for n in (usage.input_uncached, usage.cache_read, usage.cache_write) if n is not None]
    return sum(parts) if parts else None


@dataclass
class TokenCorrection:
    """One per session: the largest reported / estimated prompt tokens of the last :data:`CORRECTION_WINDOW`
    requests that reported a count, never below 1.0.  A resumed conversation starts again at 1.0."""

    factor: float = 1.0
    recent: deque[float] = field(default_factory=lambda: deque(maxlen=CORRECTION_WINDOW))

    def update(self, reported_prompt: int | None, estimated: int) -> None:
        if reported_prompt is None or estimated <= 0:
            return
        self.recent.append(reported_prompt / estimated)
        self.factor = max(1.0, *self.recent)


def _extra_tokens(extra: dict) -> int:
    return approx_tokens(json.dumps(extra, ensure_ascii=False), "json") if extra else 0


def _message_parts(m: Message, origin: Origin | None = None, *, after_last_user: bool = False) -> tuple[int, int]:
    """(text tokens, JSON tokens) of one message.  The first message of a fresh context carries tool results, so
    it counts as JSON (the larger estimate).  For the model that produced it (*origin*) an assistant message's
    provider data counts as the backend sends it: Anthropic's content blocks instead of the text and the calls;
    other data (reasoning details, the calls' thought signatures) on top of them; the reasoning text (Ollama) as
    text, and only *after_last_user* (the backend replays it only there)."""
    if m.role == "tool" or m.kind == "continuation":
        return 0, approx_tokens(m.text, "json")
    same = origin is not None and m.origin is not None and tuple(m.origin) == tuple(origin)
    if same and m.provider_extra.get(ANTHROPIC_BLOCKS_KEY):
        return 0, approx_tokens(json.dumps(m.provider_extra[ANTHROPIC_BLOCKS_KEY], ensure_ascii=False), "json")
    text = approx_tokens(m.text, "text") if m.text else 0
    calls = sum(approx_tokens(f"{c.name}{c.raw_arguments}", "json") for c in m.tool_calls)
    if same:
        extra = {k: v for k, v in m.provider_extra.items() if k != REASONING_TEXT_KEY}
        calls += _extra_tokens(extra) + sum(_extra_tokens(c.provider_extra) for c in m.tool_calls)
        reasoning = m.provider_extra.get(REASONING_TEXT_KEY)
        if after_last_user and isinstance(reasoning, str) and reasoning:
            text += approx_tokens(reasoning, "text")
    return text, calls


def schema_tokens(tools: Iterable[ToolSpec]) -> int:
    """The estimate of the tool schemas a request carries (JSON)."""
    payload = json.dumps([{"name": t.name, "description": t.description, "parameters": t.input_schema}
                          for t in tools], ensure_ascii=False)
    return approx_tokens(payload, "json")


def request_parts(req: Request, origin: Origin | None = None, *, schemas: bool = True) -> tuple[int, int]:
    """(text part, tool part) of a request's estimate, uncorrected: the system layers and the user and assistant
    text are text; the tool schemas, the tool calls, the tool results and the provider data replayed to *origin*
    (the backend's model) are JSON (P3 §4.0).  ``schemas=False``: the request goes without its tool schemas (a
    tool-less request the backend sends without tools, :func:`~flextool.assistant.providers.schemas_sent`)."""
    text = sum(approx_tokens(layer, "text") for layer in req.system)
    tools = schema_tokens(req.tools) if schemas else 0
    last_user = max((i for i, m in enumerate(req.messages) if m.role == "user"), default=-1)
    for i, m in enumerate(req.messages):
        t, j = _message_parts(m, origin, after_last_user=i > last_user)
        text += t
        tools += j
    return text, tools


def request_tokens(req: Request, corr: TokenCorrection | None = None, origin: Origin | None = None, *,
                   schemas: bool = True) -> int:
    """The estimate of the prompt tokens of *req* (to *origin*), multiplied by the session's correction
    (``schemas``: as :func:`request_parts`)."""
    text, tools = request_parts(req, origin, schemas=schemas)
    factor = corr.factor if corr is not None else 1.0
    return math.ceil((text + tools) * factor)


def fits(prompt_tokens: int, max_output_tokens: int, window: int) -> bool:
    """Whether a request of *prompt_tokens* and an answer of *max_output_tokens* stay within 95 % of *window*."""
    return prompt_tokens + max_output_tokens <= FIT_SHARE * window


def needed_tokens(system: tuple[str, ...], tools: tuple[ToolSpec, ...], max_output_tokens: int) -> int:
    """The context the instructions, the tools and one answer need, and never less than the smallest window the
    assistant works with (:data:`~flextool.assistant.tiers.MIN_WINDOW`)."""
    prefix = sum(approx_tokens(layer, "text") for layer in system) + schema_tokens(tools)
    return max(MIN_WINDOW, math.ceil((prefix + max_output_tokens) / FIT_SHARE))


# --- budgets --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LevelBudget:
    prefix_tokens: int          # layers 1–5 and the tool schemas, estimated
    max_rounds: int
    max_calls_per_turn: int
    max_calls_per_round: int
    max_output_tokens: int      # per request, clamped to a quarter of a known window
    planned_calls: int          # the tool results one request may carry: the calls of a part (fit_caps)
    min_window: int
    max_continuations: int = 0  # fresh-context continuations of one question (P3 §4.7)
    history: str = "full"       # "answers" or "full": what a request carries of the earlier questions
    history_turns: int | None = None    # "answers": how many earlier questions (None: all)


#: Starting values that the calibration informs (P3 §4.1, §7.6).  The prefix budgets are above the estimate of
#: the shipped instructions and tool schemas at the worst front end (the GUI; ``tests/assistant/test_prompt_budget``
#: checks each level × front end fits).  P3 §4.1 drafted 3 500 / 7 500 / 8 500 before the schemas were final; basic rose
#: to 4 700 with list_sources, to 4 850 with get_parameter's lists, standard to 8 700; Cycle 2: basic
#: 4 920 (C2-D1: a 16 384-token window fits a part's three 2 000-character float-heavy results, 4 088 tokens, up to
#: a prefix of 4 923; C2-OD2's 4 950 assumed 1.5 characters a token, which such results undercut),
#: standard 9 000, advanced 9 100 (list_runs' scan and overview, list_sources' full text at standard and advanced);
#: 2026-11-11 "How to answer" in the level texts: basic 5 000 (three float-heavy 2 000-character results, 4 088
#: tokens, now overflow a 16 384-token window by about 50 tokens; the pre-send fit handles it), standard 9 350,
#: advanced 9 450.
#: The history and the continuations are the owner's of 2026-11-08 (P3 §4.7).  Budgets count steps and calls:
#: basic 3 steps, 6 calls a part, 3 calls a step (batching 2026-11-08; was 3 calls a part).
LEVEL_BUDGETS: dict[str, LevelBudget] = {
    "small": LevelBudget(5_000, 3, 6, 3, 4_096, 6, 12_288, max_continuations=2, history="answers", history_turns=2),
    "medium": LevelBudget(9_350, 8, 16, 8, 8_000, 6, 32_768, max_continuations=1, history="answers"),
    "large": LevelBudget(9_450, 12, 30, 8, 16_000, 8, 65_536, max_continuations=0, history="full"),
}
#: The free plan: fewer requests per question (P3 §4.6, §4.7).  Its steps are what cost requests, so it keeps
#: fewer steps (3 / 4 / 6) and the level's calls a step; its calls a part are the level's, at most steps × calls a
#: step.
FREE_TIER_ROUNDS: dict[str, tuple[int, int]] = {"small": (3, 1), "medium": (4, 0), "large": (6, 0)}
FREE_TIER_BUDGETS: dict[str, LevelBudget] = {
    level: dataclasses.replace(b, max_rounds=rounds,
                               max_calls_per_turn=min(b.max_calls_per_turn, rounds * b.max_calls_per_round),
                               max_continuations=continuations)
    for level, b in LEVEL_BUDGETS.items() for rounds, continuations in (FREE_TIER_ROUNDS[level],)
}
#: The result caps preset of each level when the window allows it.
LEVEL_CAPS: dict[str, str] = {"small": "small", "medium": "medium", "large": "large"}
#: The free plan's result caps are at most this preset.
FREE_TIER_CAPS_MAX = "medium"

#: (lowest window, compaction every, keep recent turns, result caps rule): "fit" = :func:`fit_caps` (at most
#: small), a preset name = at most that preset, ``None`` = the level's preset.
WINDOW_CLASSES: tuple[tuple[int, int, int, str | None], ...] = (
    (200_000, 6, 2, None),
    (65_536, 6, 2, "medium"),
    (32_768, 4, 2, "small"),
    (MIN_WINDOW, 1, 0, "fit"),
)

#: The OpenRouter model ids whose free plan is on by itself end in this (P3-O2).
FREE_SUFFIX = ":free"


def history_policy(tier: str) -> tuple[str, int | None]:
    """(history, history turns) of a level (internal name): what a request carries of the earlier questions."""
    b = LEVEL_BUDGETS[tier]
    return b.history, b.history_turns


def window_class(window: int) -> tuple[int, int, str | None]:
    """(compaction every, keep recent turns, result caps rule) of a window of *window* tokens."""
    for lowest, every, keep, rule in WINDOW_CLASSES:
        if window >= lowest:
            return every, keep, rule
    raise ValueError(f"a window of {window} tokens is below the assistant's minimum of {MIN_WINDOW}")


def _smaller(a: caps_module.Caps, b: caps_module.Caps) -> caps_module.Caps:
    """The smaller of each cap of *a* and *b* (every field but ``ceilings``: a new cap field must be added here,
    or it comes back as its default on every budget path, Cycle 2 critique M3)."""
    return caps_module.Caps(min(a.max_rows, b.max_rows), min(a.max_chars, b.max_chars),
                            min(a.max_results, b.max_results), min(a.max_items, b.max_items),
                            max_runs=min(a.max_runs, b.max_runs))


def fit_caps(window: int, level: LevelBudget, max_output_tokens: int, keep_recent_turns: int) -> caps_module.Caps:
    """The result caps that let a question's planned results fit a small window (P3 §4.3)."""
    small = caps_module.preset("small")
    available = (window - level.prefix_tokens - max_output_tokens - FIT_MARGIN_SHARE * window
                 - min(HISTORY_TOKENS_MAX, HISTORY_SHARE * window))
    per_call = available / (keep_recent_turns + 1) / level.planned_calls
    max_chars = int(round(min(max(per_call * RESULT_CHARS_PER_TOKEN, MIN_RESULT_CHARS), small.max_chars), 6))
    share = max_chars / small.max_chars
    return caps_module.Caps(max_rows=max(10, round(small.max_rows * share)), max_chars=max_chars,
                            max_results=max(3, round(small.max_results * share)),
                            max_items=max(3, round(small.max_items * share)), max_runs=small.max_runs)


def fitting_calls(window: int, level: LevelBudget, max_output_tokens: int, keep_recent_turns: int,
                  max_chars: int) -> int:
    """The tool calls of one part whose results, each at *max_chars* and as dense as result JSON gets
    (:data:`RESULT_CHARS_PER_TOKEN`), fit 95 % of *window* with the prefix, the answer and the text history: at least
    a step's calls, at most the level's calls a part (batching 2026-11-08)."""
    room = (FIT_SHARE * window - level.prefix_tokens - max_output_tokens
            - min(HISTORY_TOKENS_MAX, HISTORY_SHARE * window))
    n = math.floor(round(room / (keep_recent_turns + 1) / (max_chars / RESULT_CHARS_PER_TOKEN), 6))
    return max(level.max_calls_per_round, min(level.max_calls_per_turn, n))


def free_tier_on(preset_id: str, model: str, setting: bool | None) -> bool:
    """Whether the free plan is on: the setting when the user set it, else on only for an OpenRouter model whose
    id ends ``:free`` (P3-O2)."""
    if setting is not None:
        return setting
    return preset_id == "openrouter" and model.strip().lower().endswith(FREE_SUFFIX)


@dataclass(frozen=True)
class Budget:
    """The budget of one session: a level's numbers fitted to the model's window."""

    tier: str
    level: LevelBudget              # the free plan's numbers when it is on; calls a part fitted to a small window
    free_tier: bool
    window: int                     # the window the caps and the fit use (assumed when nothing reports one)
    window_source: str
    fit: bool                       # the pre-send fit runs (a known window, or the assumed local one)
    local: bool
    max_output_tokens: int
    compaction_every: int
    keep_recent_turns: int
    result_caps: caps_module.Caps   # defaults and, in ``ceilings``, the most a call may ask for

    @property
    def max_rounds(self) -> int:
        return self.level.max_rounds


def resolve_budget(tier: str, window: int | None, window_source: str | None, *, local: bool,
                   free_tier: bool = False, compaction_every: int | None = None,
                   keep_recent_turns: int | None = None, result_cap: str | None = None) -> Budget:
    """The budget of *tier* (internal name) for a model whose window is *window* (``None``: not known, the assumed
    one for where it runs).  *compaction_every* / *keep_recent_turns* override the window's rhythm (settings);
    *result_cap* (``--result-cap``) sets the default result caps under the window's ceiling."""
    level = (FREE_TIER_BUDGETS if free_tier else LEVEL_BUDGETS)[tier]
    known = window is not None and window_source not in (None, "assumed")
    if window is None:
        window = ASSUMED_LOCAL_WINDOW if local else ASSUMED_REMOTE_WINDOW
        window_source = "assumed"
    every, keep, rule = window_class(window)
    if compaction_every is not None:
        every = compaction_every
    if keep_recent_turns is not None:
        keep = keep_recent_turns
    max_output = min(level.max_output_tokens, window // 4) if known else level.max_output_tokens
    level_caps = caps_module.preset(LEVEL_CAPS[tier])
    if rule == "fit":
        ceiling = _smaller(fit_caps(window, level, max_output, keep), caps_module.preset("small"))
    elif rule is None:
        ceiling = level_caps
    else:
        ceiling = _smaller(level_caps, caps_module.preset(rule))
    if free_tier:
        ceiling = _smaller(ceiling, caps_module.preset(FREE_TIER_CAPS_MAX))
    if rule == "fit":
        calls = fitting_calls(window, level, max_output, keep, ceiling.max_chars)
        level = dataclasses.replace(level, max_calls_per_turn=calls)
    defaults = _smaller(caps_module.preset(result_cap), ceiling) if result_cap else ceiling
    result_caps = dataclasses.replace(defaults, ceilings=ceiling)
    return Budget(tier=tier, level=level, free_tier=free_tier, window=window, window_source=window_source or "assumed",
                  fit=known or local, local=local, max_output_tokens=max_output, compaction_every=max(1, every),
                  keep_recent_turns=max(0, keep), result_caps=result_caps)


def budget_for(choice: TierChoice, *, free_tier: bool = False, compaction_every: int | None = None,
               keep_recent_turns: int | None = None, result_cap: str | None = None) -> Budget:
    """:func:`resolve_budget` for the level and window of a :class:`~flextool.assistant.tiers.TierChoice`."""
    return resolve_budget(choice.chosen, choice.window, choice.window_source, local=choice.local,
                          free_tier=free_tier, compaction_every=compaction_every,
                          keep_recent_turns=keep_recent_turns, result_cap=result_cap)
