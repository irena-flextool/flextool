"""What can go wrong when FlexTool talks to an AI service, and what the user is told (P1 §4).

Every backend turns the provider library's exception into one :class:`LLMError` subclass with
:func:`classify_openai` or :func:`classify_anthropic`.  The classifier reads the error type in the body first (walked
recursively: a Gemini body may be a list), then the HTTP status, then the headers; an error inside a stream (no
status) is classified from its body alone.  :data:`MESSAGES` holds every text the user sees, in plain English with
no abbreviations; ``{provider}`` is the label of the route (for example "Ollama cloud").  The messages name the
terminal's commands; :func:`message_for` gives the FlexTool window's wording (:data:`GUI_MESSAGES`) where it
differs.
"""
from __future__ import annotations

import email.utils
import math
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from flextool.assistant.llm_adapter import Usage

MESSAGES: dict[str, str] = {
    "rpm_retry": ("{provider} is limiting how often requests can be sent. Trying again in {seconds} seconds "
                  "(attempt {n} of {max})."),
    "rpm": "{provider} is still limiting requests. Wait a minute and ask again.",
    "tpm_input": ("{provider} still refuses this request: it limits the input tokens it accepts per minute, and one "
                  "request this long may be over that limit by itself. Wait a minute and ask again; if it is "
                  "refused again, start a new conversation, or switch to a model or plan with a higher limit."),
    "rpm_paced": ("{provider} limits how many requests per second a free plan may send. FlexTool now waits between "
                  "requests; ask again."),
    "pace_wait": "{provider} limits how fast requests may be sent. Sending the next one in {seconds} seconds.",
    "inactive": ("Your {provider} account allows 0 requests per minute: its API access is not active yet. "
                 "Activate a plan in your {provider} account, then ask again."),
    "rpm_long": ("{provider} asks to wait about {minutes} minutes before the next request. Ask again later, or "
                 "switch provider in the settings."),
    "busy_retry": ("{provider} is overloaded at the moment. Trying again in {seconds} seconds (attempt {n} of "
                   "{max})."),
    "busy": "{provider} is still overloaded. Ask again in a moment.",
    "busy_local_retry": ("The local model server ({provider}) is busy, likely loading a model. Trying again in "
                         "{seconds} seconds (attempt {n} of {max})."),
    "busy_local": ("The local model server ({provider}) is still busy, likely loading a model. Ask again in a "
                   "minute."),
    "busy_streamed": "{provider} stopped the answer because it is overloaded. Ask again in a moment.",
    "ended_early_retry": ("{provider} ended the answer before any of it came. Trying again in {seconds} seconds "
                          "(attempt {n} of {max})."),
    "ended_early": "{provider} ended the answer before any of it came, several times. Ask again in a moment.",
    "tool_call_unreadable": ("{provider} could not read a tool call in the model's answer. Ask again, or try another "
                             "model or level."),
    "tool_call_unreadable_retry": ("{provider} could not read a tool call in the model's answer. Asking again "
                                   "(attempt {n} of {max})."),
    "tool_call_unreadable_final": ("{provider} could not read a tool call in the model's answer, {count} times in a "
                                   "row. Ask again, or try another model (/model) or a lower level (--tier)."),
    "quota": ("Your {provider} quota is used up for now. It resets on the provider's schedule, which FlexTool "
              "cannot see. You can switch to another provider or model."),
    "quota_daily": "{provider}: the daily quota of this model is used up; try tomorrow or another model.",
    "credit": ("Your {provider} account has no credit left. Add credit in your provider account, or switch "
               "provider."),
    "auth": ("{provider} did not accept the key (from {key_source}). Check the key, or store a new one with "
             "flextool-assistant --set-key."),
    "permission": "{provider} accepted the key but does not allow it to use {model}.",
    "context_retry": ("The conversation is too long for {model}. FlexTool removed older tool results and tried "
                      "again."),
    "context": ("The conversation is still too long for {model}. Start a new one with /new (/new carry keeps a "
                "short summary of this one)."),
    "tpm_too_large": ("This request is larger than your {provider} account may send in one minute (its answer: "
                      "{text}). A shorter conversation fits: start a new one, or switch to a model or plan with a "
                      "higher limit."),
    "history": ("The provider did not accept earlier reasoning data, so FlexTool sent the conversation without "
                "it."),
    "routing": ("No provider for this model matches your privacy setting (data collection: deny). Change it in "
                "settings or pick another model."),
    "routing_tools": "Some models also have no provider that supports tool calls.",
    "routing_zdr": "The setting to use only providers that keep no copy also applies.",
    "tools": ("The model {model} does not support tool calls, so the assistant cannot read your case data. Pick a "
              "model that supports tools."),
    "tools_ollama": "The model's page in the Ollama library lists 'tools' when it does.",
    "model": ("{provider} did not find the model {model} (its answer: {text}). See the available ones with "
              "flextool-assistant --list-models."),
    "connection_local": "Nothing answers at {host}:{port}. Start {server_name} and try again.",
    "connection_remote": "Could not reach {host}. Check the internet connection or proxy settings.",
    "connection_retry": "Could not reach {host}. Trying again in {seconds} seconds.",
    "timeout": "{provider} did not answer within {seconds} seconds.",
    "connect_timeout": "Could not connect to {host} within {seconds} seconds.",
    "refused": "{provider} declined to answer this request{reason_suffix}.",
    "length": "The answer was cut off at the output limit ({tokens} tokens).",
    "length_no_finish": "The connection ended before the answer was complete.",
    "no_key": ("No key found for {provider}. Set the environment variable {env_name}, or store the key with "
               "flextool-assistant --set-key."),
    "other": "{provider} returned an error (status {status}): {text}",
    "other_no_status": "{provider} returned an error: {text}",
    "redirect": ("The endpoint redirected to {host}. FlexTool does not follow redirects, so nothing was sent "
                 "there."),
    "cancelled": "Stopped.",
    "internal": "The assistant stopped because of an internal fault ({type}). Please report it.",
    "newer_format": "The assistant stopped: {text}",
    "unknown_stop": "The provider ended the answer for a reason FlexTool does not know ({reason}).",
    "model_list": "The model list could not be fetched: {reason}. Type the model name instead.",
    "small_context_ollama": ("This model runs with a {n}-token context, too small for the assistant's instructions "
                             "and tools (about {needed}). Set OLLAMA_CONTEXT_LENGTH=16384 (or num_ctx in a "
                             "Modelfile) and restart Ollama."),
    "small_context_lmstudio": ("LM Studio loaded this model with a {n}-token context. Load it again with a larger "
                               "context length (about {needed} or more) in LM Studio's model settings."),
    "small_context": ("This model has a {n}-token context, too small for the assistant's instructions and tools "
                      "(about {needed}). Choose a model with a larger context."),
    "probe_unreadable": ("{provider} at {host}:{port} gave an answer FlexTool could not read ({what}), so nothing "
                         "was sent to the model."),
}

#: The FlexTool window's wording of the messages that name terminal commands (P2 §9.5), by message key.
GUI_MESSAGES: dict[str, str] = {
    "auth": ("{provider} did not accept the key (from {key_source}). Check or replace the key in the assistant "
             "settings."),
    "no_key": ("No key found for {provider}. Paste a key in the assistant settings, or set the environment "
               "variable {env_name}."),
    "model": ("{provider} did not find the model {model} (its answer: {text}). A listed model can still be closed "
              "to your account: pick another one from the list in the assistant settings."),
    "context": ("The conversation is still too long for {model}. Start a new chat; \u201cNew chat with summary\u201d "
                "keeps a short summary of this one."),
    "internal": ("The assistant stopped because of an internal fault ({type}). Please report it; the details are "
                 "in the FlexTool log."),
    "tool_call_unreadable_final": ("{provider} could not read a tool call in the model's answer, {count} times in a "
                                   "row. Ask again, or pick another model or level in the assistant settings."),
}

#: Refusal categories in plain English; any other category is "the provider's usage rules".
REFUSAL_CATEGORIES: dict[str, str] = {
    "cyber": "computer security",
    "bio": "biology or chemistry safety",
    "general_harms": "possible harm",
}
REFUSAL_OTHER = "the provider's usage rules"

#: Longest provider text quoted in an error message.
MAX_PROVIDER_TEXT = 500
#: What a message quoting the provider says when the provider gave no text.
NO_PROVIDER_TEXT = "no reason given"
#: Longest provider message kept in :class:`ProviderSaid` (the calibration's spent.json and console).
MAX_PROVIDER_SAID = 300


@dataclass(frozen=True)
class QuotaViolation:
    """One quota a Google rate-limit answer names (``google.rpc.QuotaFailure.Violation``, else a ``* Quota exceeded
    for metric: …, limit: …, model: …`` line of its message): the *metric*, the quota id (*quota_id*, which names
    the period, such as ``…PerMinute…`` or ``…PerDay…``), the enforced *limit* and the *model*; ``""`` for each
    the answer does not name."""

    metric: str = ""
    quota_id: str = ""
    limit: str = ""
    model: str = ""

    @property
    def daily(self) -> bool:
        """Whether the quota counts per day (it resets the next day, so waiting a minute does not help)."""
        return any("perday" in v.lower() or "daily" in v.lower() for v in (self.metric, self.quota_id))

    def text(self) -> str:
        """``input tokens per minute (limit 15000, model gemma-4-31b-it)``."""
        details = ", ".join(p for p in (f"limit {self.limit}" if self.limit else "",
                                        f"model {self.model}" if self.model else "") if p)
        return f"{_quota_name(self)} ({details})" if details else _quota_name(self)

    def as_json(self) -> dict[str, str]:
        return {"metric": self.metric, "quota_id": self.quota_id, "limit": self.limit, "model": self.model}


def _quota_name(q: QuotaViolation) -> str:
    """What a quota counts, in words: ``input tokens per minute``; the metric's own name when it is not known."""
    metric = q.metric.rsplit("/", 1)[-1].lower()
    ident = q.quota_id.lower()
    both = f"{metric} {ident}"
    if "input_token" in both or "inputtoken" in both:
        what = "input tokens"
    elif "output_token" in both or "outputtoken" in both:
        what = "output tokens"
    elif "token" in both:
        what = "tokens"
    elif "request" in both:
        what = "requests"
    else:
        return q.metric.rsplit("/", 1)[-1] or q.quota_id or "a quota"
    per = next((words for key, words in (("perminute", "per minute"), ("perday", "per day"),
                                          ("perhour", "per hour")) if key in both), "")
    return f"{what} {per}" if per else what


@dataclass(frozen=True)
class ProviderSaid:
    """What the provider's own error answer said, beside FlexTool's message: the HTTP *status* (for an error inside
    a stream, the numeric code its body names, else ``None``), the provider's error *code* (Google's status name
    such as ``RESOURCE_EXHAUSTED``, OpenAI's or Anthropic's error type and code; ``""`` when none) and its
    *message*, redacted (no key, no local path, no link) and at most :data:`MAX_PROVIDER_SAID` characters (a
    Google quota answer's message leads with the quotas it names, which are never cut).  For a Google quota
    answer, the *quota* it names and the *retry_delay* (seconds) it suggests."""

    status: int | None
    code: str
    message: str
    quota: tuple[QuotaViolation, ...] = ()
    retry_delay: float | None = None

    def text(self) -> str:
        """``429 RESOURCE_EXHAUSTED — Quota exceeded for metric …`` (the parts it has)."""
        head = " ".join(p for p in (str(self.status) if self.status is not None else "", self.code) if p)
        if head and self.message:
            return f"{head} — {self.message}"
        return head or self.message or NO_PROVIDER_TEXT

    def as_json(self) -> dict[str, object]:
        return {"status": self.status, "code": self.code, "message": self.message,
                "quota": [q.as_json() for q in self.quota], "retry_delay": self.retry_delay}


@dataclass
class ErrorContext:
    """What the messages of one connection name: the route label, the model, where the key came from, the
    endpoint, and the server to start when a local endpoint does not answer."""

    preset_id: str = ""
    provider: str = "The AI service"
    model: str = ""
    key_source: str = ""
    host: str = ""
    port: int | None = None
    local: bool = False
    server_name: str | None = None
    key_env: str | None = None
    zdr: bool = False
    read_timeout: float = 300.0
    secret: str = field(default="", repr=False)

    def fields(self) -> dict[str, object]:
        return {"provider": self.provider, "model": self.model, "key_source": self.key_source,
                "host": self.host, "port": self.port if self.port is not None else "",
                "server_name": self.server_name or "the model server", "env_name": self.key_env or "",
                "seconds": _seconds(self.read_timeout)}


def render(key: str, ctx: ErrorContext | None = None, **extra: object) -> str:
    """The text of :data:`MESSAGES` ``[key]`` with the fields of *ctx* and *extra*."""
    values = (ctx or ErrorContext()).fields()
    values.update(extra)
    return MESSAGES[key].format(**values)


def _seconds(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


class LLMError(Exception):
    """A model request that did not give an answer, with the message for the user.

    *streamed* is True when part of the answer was already shown (nothing is retried then; *partial_text* keeps
    what was shown).  *partial_usage* is the usage of a request that reached the model but failed or was stopped.
    """

    code = "other"

    def __init__(self, user_message: str, *, provider: str = "", retry_after: float | None = None,
                 partial_usage: Usage | None = None, streamed: bool = False, status: int | None = None,
                 local: bool = False, partial_text: str = "", fields: dict[str, object] | None = None) -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.provider = provider
        self.retry_after = retry_after
        self.partial_usage = partial_usage
        self.streamed = streamed
        self.status = status
        self.local = local
        self.partial_text = partial_text
        self.fields = dict(fields or {})
        self.message_key: str | None = None     # the MESSAGES key the user message was made from
        #: The request-limit headers of the answer (:data:`flextool.assistant.pacing.RATE_HEADERS`), for a rate
        #: limit; the pacing reads the wait they name (review G9, C4-m1).
        self.rate_headers: dict[str, str] | None = None
        #: The pacing epoch the request started in, set by a backend that paces itself (the window's gate).
        self.pace_epoch: int | None = None
        #: The provider's own error answer (:class:`ProviderSaid`), for an error classified from one
        #: (:func:`classify_status`); ``None`` for one FlexTool found itself (no connection, a stream without an end).
        self.provider_said: ProviderSaid | None = None

    @property
    def params(self) -> dict[str, str]:
        """The values the message names (``provider``, ``model``, ``key_source``, ``env_name``, …) as text."""
        return {key: str(value) for key, value in self.fields.items()}

    def __str__(self) -> str:
        return self.user_message


class RateLimited(LLMError):
    code = "rpm"


class InputTokensPerMinute(RateLimited):
    """A rate limit whose quota counts the input tokens per minute (Google's ``…InputTokensPer…Minute`` quota, or its
    input-token quota line without a period): retried like any rate limit, but a request whose input alone is over
    the limit is refused every time, so the calibration stops resuming it (calibration fixes 2026-11-10)."""

    code = "tpm_input"


class AccessInactive(LLMError):
    """The provider reports a request limit of 0: the account's API access is not active (review G9, OR-2).  Not
    retried, and the pacing does not change."""

    code = "inactive"


class ServerBusy(LLMError):
    code = "busy"


class UnreadableToolCall(LLMError):
    """The service could not read a tool call in the model's answer (Ollama: ``500 error parsing tool call``, the
    model wrote its analysis into the call).  Not an overload: asked again at once, a few times, by its own counter
    (:mod:`flextool.assistant.llm_retry`); the request-limit gate and the overload backoff leave it alone."""

    code = "tool_call_unreadable"


class QuotaExhausted(LLMError):
    code = "quota"


class NoCredit(LLMError):
    code = "credit"


class AuthFailed(LLMError):
    code = "auth"


class NotPermitted(LLMError):
    code = "permission"


class ContextTooLong(LLMError):
    code = "context"


class HistoryRejected(LLMError):
    code = "history"


class NoMatchingEndpoint(LLMError):
    code = "routing"


class ToolsUnsupported(LLMError):
    code = "tools"


class ModelNotFound(LLMError):
    code = "model"


class Unreachable(LLMError):
    code = "connection"


class TimedOut(LLMError):
    code = "timeout"


class ContextTooSmall(LLMError):
    """A model on this computer runs with a context too small for the instructions and tools: no request is sent
    (the server would silently drop the start of the prompt)."""

    code = "small_context"


class Refused(LLMError):
    code = "refused"


class Truncated(LLMError):
    code = "length"


class NoKey(LLMError):
    code = "no_key"


class Cancelled(LLMError):
    code = "cancelled"


class ProviderError(LLMError):
    code = "other"


class InternalFault(LLMError):
    """A turn that ended on an error of FlexTool itself, not of the model request (review G11, C5-3)."""

    code = "internal"


ERROR_CLASSES: tuple[type[LLMError], ...] = (
    RateLimited, InputTokensPerMinute, AccessInactive, ServerBusy, UnreadableToolCall, QuotaExhausted, NoCredit,
    AuthFailed, NotPermitted, ContextTooLong, HistoryRejected,
    NoMatchingEndpoint, ToolsUnsupported, ModelNotFound, Unreachable, TimedOut, ContextTooSmall, Refused, Truncated,
    NoKey, Cancelled, ProviderError, InternalFault)


def make_error(cls: type[LLMError], key: str, ctx: ErrorContext, **kwargs: Any) -> LLMError:
    """An error of *cls* with the message *key* for *ctx*; *kwargs* go to :class:`LLMError`."""
    extra = kwargs.pop("message_fields", {})
    # The provider's text, for the messages that quote it, when the caller has none.
    fields = {"text": NO_PROVIDER_TEXT, **ctx.fields(), **extra}
    err = cls(MESSAGES[key].format(**fields), provider=ctx.provider, local=ctx.local, fields=fields, **kwargs)
    err.message_key = key
    return err


def message_for(err: LLMError, front_end: str) -> str:
    """The message of *err* for *front_end*: the FlexTool window's wording (``"gui"``) where it differs, else the
    terminal's (:attr:`LLMError.user_message`)."""
    if front_end == "gui" and err.message_key in GUI_MESSAGES and err.user_message == MESSAGES[err.message_key] \
            .format(**err.fields):
        return GUI_MESSAGES[err.message_key].format(**err.fields)
    return err.user_message


def provider_said_line(err: LLMError) -> str | None:
    """``Google Gemini said: 429 RESOURCE_EXHAUSTED — …``, the provider's own words after FlexTool's message;
    ``None`` when the error came from no provider answer."""
    if err.provider_said is None:
        return None
    return f"{err.provider or 'The AI service'} said: {err.provider_said.text()}"


def internal_fault_message(type_name: str) -> str:
    """The FlexTool window's message for an unexpected fault of the assistant itself."""
    return GUI_MESSAGES["internal"].format(type=type_name)


def cancelled(ctx: ErrorContext | None = None, **kwargs: Any) -> Cancelled:
    err = make_error(Cancelled, "cancelled", ctx or ErrorContext(), **kwargs)
    assert isinstance(err, Cancelled)
    return err


def refusal_reason(category: str | None, explanation: str | None) -> str | None:
    """The reason shown for a refusal: the provider's explanation, else the category in plain English (the raw
    category code is never shown)."""
    if explanation and explanation.strip():
        return explanation.strip()
    if category is None:
        return None
    return REFUSAL_CATEGORIES.get(category, REFUSAL_OTHER)


def refusal_message(ctx: ErrorContext, reason: str | None) -> str:
    suffix = f" (reason: {reason.rstrip('.')})" if reason else ""
    return render("refused", ctx, reason_suffix=suffix)


def routing_message(ctx: ErrorContext) -> str:
    parts = [MESSAGES["routing"], MESSAGES["routing_tools"]]
    if ctx.zdr:
        parts.append(MESSAGES["routing_zdr"])
    return " ".join(parts)


def tools_message(ctx: ErrorContext) -> str:
    text = render("tools", ctx)
    if ctx.preset_id == "ollama":
        text += " " + MESSAGES["tools_ollama"]
    return text


# --- reading an error body ----------------------------------------------------------------------------------------
@dataclass
class _Body:
    types: set[str] = field(default_factory=set)
    codes: set[str] = field(default_factory=set)
    statuses: set[str] = field(default_factory=set)
    reasons: set[str] = field(default_factory=set)
    quota_ids: list[str] = field(default_factory=list)
    retry_delays: list[str] = field(default_factory=list)
    #: Google's ``QuotaFailure.violations`` entries (``quotaMetric``, ``quotaId``, ``quotaDimensions``, …).
    violations: list[dict] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    strings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(self.strings).lower()


def _walk(obj: object) -> Iterator[tuple[str | None, object]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield str(key), value
            yield from _walk(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield None, value
            yield from _walk(value)


def _read_body(body: object) -> _Body:
    out = _Body()
    if isinstance(body, str):
        out.strings.append(body)
        out.messages.append(body)
        return out
    for key, value in _walk(body):
        if isinstance(value, str):
            out.strings.append(value)
        if key is None:
            continue
        if key == "type" and isinstance(value, str):
            out.types.add(value)
        elif key in ("code", "error_code") and isinstance(value, (str, int)) and not isinstance(value, bool):
            out.codes.add(str(value))
        elif key == "status" and isinstance(value, str):
            out.statuses.add(value)
        elif key == "reason" and isinstance(value, str):
            out.reasons.add(value)
        elif key == "quotaId" and isinstance(value, str):
            out.quota_ids.append(value)
        elif key == "retryDelay" and isinstance(value, str):
            out.retry_delays.append(value)
        elif key == "violations" and isinstance(value, list):
            out.violations.extend(v for v in value if isinstance(v, dict))
        elif key in ("message", "raw") and isinstance(value, str):
            out.messages.append(value)
    if not out.retry_delays and "RESOURCE_EXHAUSTED" in out.statuses:
        # Google's message names the wait too ("Please retry in 34.36s."): the fallback without RetryInfo.
        for message in out.messages:
            match = _RETRY_IN.search(message)
            if match:
                out.retry_delays.append(match.group(1))
                break
    return out


#: Google's message line per quota ("* Quota exceeded for metric: X, limit: N, model: M") and its suggested wait.
_QUOTA_LINE = re.compile(r"Quota exceeded for metric:\s*(?P<metric>[^,\s]+)(?:,\s*limit:\s*(?P<limit>[^,\s]+))?"
                         r"(?:,\s*model:\s*(?P<model>[^,\s]+))?")
_RETRY_IN = re.compile(r"\bretry in\s+(\d+(?:\.\d+)?)\s*s\b", re.IGNORECASE)


def _quota_of(info: _Body) -> tuple[QuotaViolation, ...]:
    """The quotas a Google answer names: its ``QuotaFailure`` violations, each completed from the message line of
    its metric; without violations, the message lines alone."""
    lines = [m.groupdict() for message in info.messages for m in _QUOTA_LINE.finditer(message)]
    by_metric = {line["metric"]: line for line in lines}
    out: list[QuotaViolation] = []
    for v in info.violations:
        metric = str(v.get("quotaMetric") or "")
        quota_id = str(v.get("quotaId") or "")
        if not metric and not quota_id:
            continue
        dims = v.get("quotaDimensions")
        line = by_metric.get(metric, {})
        value = v.get("quotaValue")
        limit = str(value) if isinstance(value, (str, int)) and not isinstance(value, bool) else ""
        model = str(dims.get("model") or "") if isinstance(dims, dict) else ""
        out.append(QuotaViolation(metric, quota_id, limit or (line.get("limit") or "").rstrip("."),
                                  model or (line.get("model") or "").rstrip(".")))
    if not out:
        out = [QuotaViolation(line["metric"], "", (line["limit"] or "").rstrip("."), (line["model"] or "").rstrip("."))
               for line in lines]
    return tuple(dict.fromkeys(out))


def _input_per_minute(q: QuotaViolation) -> bool:
    """Whether *q* counts input tokens per minute: its metric or quota id names input tokens and its quota id names
    no other period (a quota line of the message alone names none; Google's free tier counts input tokens per
    minute, CAL-FINDING 2026-11-10).  A daily quota is classified before."""
    both = f"{q.metric} {q.quota_id}".lower()
    if "input_token" not in both and "inputtoken" not in both:
        return False
    ident = q.quota_id.lower()
    return not ident or "perminute" in ident


#: Error types (Anthropic) and Google status names, with the HTTP status they stand for.
_TYPE_STATUS = {"authentication_error": 401, "permission_error": 403, "billing_error": 402,
                "rate_limit_error": 429, "overloaded_error": 529, "api_error": 500, "not_found_error": 404,
                "request_too_large": 413, "invalid_request_error": 400, "timeout_error": 504}
_GOOGLE_STATUS = {"UNAUTHENTICATED": 401, "PERMISSION_DENIED": 403, "RESOURCE_EXHAUSTED": 429, "UNAVAILABLE": 503,
                  "INTERNAL": 500, "DEADLINE_EXCEEDED": 504, "NOT_FOUND": 404, "INVALID_ARGUMENT": 400}
_BUSY_STATUS = frozenset({500, 502, 503, 504, 529})
_CONTEXT_PHRASES = ("context_length_exceeded", "prompt is too long", "maximum context length", "context window",
                    "too many tokens", "input token count", "maximum number of tokens")
_TOOL_PHRASES = ("does not support tools", "tool choice requires")
#: A tool call the service could not read in the model's answer (Ollama's 500 "error parsing tool call: raw='…'",
#: also inside a stream as an ``api_error``): the model's fault in this one answer, not an overload.
_UNREADABLE_CALL_PHRASES = ("error parsing tool call",)
#: OpenAI's 429 for one request above the per-minute token limit ("Request too large for … on tokens per min
#: (TPM): Limit …, Requested …. The input or output tokens must be reduced …"): a retry cannot succeed.
_TOO_LARGE_PHRASES = ("request too large", "must be reduced")


def _body_status(body: _Body) -> int | None:
    for code in body.codes:
        if code.isdigit() and 300 <= int(code) <= 599:
            return int(code)
    for name in body.statuses:
        if name in _GOOGLE_STATUS:
            return _GOOGLE_STATUS[name]
    for name in body.types:
        if name in _TYPE_STATUS:
            return _TYPE_STATUS[name]
    return None


def _parse_delay(text: str) -> float | None:
    text = text.strip().removesuffix("s")
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) and value >= 0 else None


def retry_after_from(headers: Any, body: _Body | None = None, *, now: float | None = None) -> float | None:
    """Seconds to wait: ``retry-after-ms``/1000, else ``retry-after`` (seconds or an HTTP date), else Google's
    ``RetryInfo.retryDelay``, else ``None``."""
    if headers is not None:
        ms = headers.get("retry-after-ms")
        if ms is not None:
            value = _parse_delay(str(ms))
            if value is not None:
                return value / 1000.0
        after = headers.get("retry-after")
        if after is not None:
            value = _parse_delay(str(after))
            if value is not None:
                return value
            try:
                when = email.utils.parsedate_to_datetime(str(after))
            except (TypeError, ValueError):
                when = None
            if when is not None:
                current = time.time() if now is None else now
                return max(0.0, when.timestamp() - current)
    if body is not None:
        for delay in body.retry_delays:
            value = _parse_delay(delay)
            if value is not None:
                return value
    return None


#: Mistral's request-limit headers per minute (seen 2026-11-08, OR-2).
LIMIT_MINUTE = "x-ratelimit-limit-req-minute"
REMAINING_MINUTE = "x-ratelimit-remaining-req-minute"
#: Google's quota text for a quota of 0 ("…, limit: 0, model: …"); "limit: 0.5" is not one.
_LIMIT_ZERO = re.compile(r"\blimit:\s*0(?![\d.])")


def _whole_number(value: object) -> int | None:
    try:
        number = float(str(value))
    except ValueError:
        return None
    return int(number) if math.isfinite(number) and number.is_integer() else None


def _limit_zero(headers: Any, info: _Body) -> bool:
    """Whether a rate-limit answer reports a limit of 0: Mistral's per-minute limit and remaining both 0, or a
    Google ``RESOURCE_EXHAUSTED`` whose quota text says ``limit: 0`` (a quota of 0 never resets)."""
    if headers is not None:
        limit, remaining = headers.get(LIMIT_MINUTE), headers.get(REMAINING_MINUTE)
        if limit is not None and remaining is not None and _whole_number(limit) == 0 \
                and _whole_number(remaining) == 0:
            return True
    return "RESOURCE_EXHAUSTED" in info.statuses and _LIMIT_ZERO.search(info.text) is not None


def _provider_text(ctx: ErrorContext, body: _Body, fallback: str) -> str:
    return _clean(ctx, next((m for m in body.messages if m.strip()), "") or fallback, MAX_PROVIDER_TEXT)


#: A link in a provider's text (FlexTool shows no links).
_URL = re.compile(r"(?:https?://|www\.)\S+?(?=[.,;:!?)]*(?:\s|$))", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")


def _sentences(text: str) -> list[str]:
    """*text* in sentences (a new line also ends one), each without its links: a sentence that leads to a link
    ("For more information …, head to: <link>.") is left out; when every sentence has one, only the links are."""
    marked = _URL.sub("\0", text)
    parts = [p.strip() for p in _SENTENCE_END.split(marked) if p.strip()]
    kept = [p for p in parts if "\0" not in p]
    if not kept:
        kept = [" ".join(p.replace("\0", " ").split()) for p in parts]
    return [p for p in kept if p and p not in (".", ":")]


def _clean(ctx: ErrorContext, text: str, limit: int, *, lead: str = "") -> str:
    """*text* redacted (no key, no local path) and without links, at most *limit* characters.  *lead* (a Google
    answer's quotas) comes first and is never cut: the sentences after it are left out, last first, to fit."""
    from flextool.assistant.redact import Redactor

    if ctx.secret:
        text = text.replace(ctx.secret, "<key>")
    sentences = _sentences(text)
    if lead:
        # The quota lines and the suggested wait are in the lead already.
        sentences = [s for s in sentences if not _QUOTA_LINE.search(s) and not _RETRY_IN.search(s)]
        while sentences and len(" ".join([lead, *sentences])) > limit:
            sentences.pop()
        return Redactor(None).redact(" ".join([lead, *sentences]))
    text = Redactor(None).redact(" ".join(" ".join(sentences).split()))
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _quota_lead(quota: tuple[QuotaViolation, ...], delay: float | None) -> str:
    """``quota: input tokens per minute (limit 15000, model gemma-4-31b-it); Google suggests waiting 34 s.``"""
    if not quota:
        return ""
    parts = ["quota: " + "; ".join(q.text() for q in quota)]
    if delay is not None:
        parts.append(f"Google suggests waiting {math.ceil(delay)} s")
    return "; ".join(parts) + "."


def _said(status: int | None, body: object, info: _Body, ctx: ErrorContext, fallback_text: str) -> ProviderSaid:
    """The :class:`ProviderSaid` of an error answer: its status, its error code and its first message (the library's
    text only when there is no body: an answer that was no JSON)."""
    if status is None:
        status = next((int(c) for c in sorted(info.codes) if c.isdigit() and 300 <= int(c) <= 599), None)
    # Google's status name, else the error code, else the error type (Anthropic's envelope type "error" says
    # nothing).
    names = (sorted(info.statuses) or sorted(c for c in info.codes if not c.isdigit())
             or sorted(t for t in info.types if t != "error"))
    message = next((m for m in info.messages if m.strip()), "") or (fallback_text if body is None else "")
    quota = _quota_of(info)
    delay = retry_after_from(None, info) if quota else None
    return ProviderSaid(status, ", ".join(names),
                        _clean(ctx, message, MAX_PROVIDER_SAID, lead=_quota_lead(quota, delay)), quota, delay)


def classify_status(status: int | None, body: object, headers: Any, ctx: ErrorContext, *, fallback_text: str = "",
                    now: float | None = None, **kwargs: Any) -> LLMError:
    """The error for an answer with HTTP *status* (``None`` for an error inside a stream), *body* (decoded JSON or
    text) and *headers*; *kwargs* go to :class:`LLMError` (``partial_usage``, ``streamed``, ``partial_text``).
    Every error it returns carries what the provider said (:attr:`LLMError.provider_said`)."""
    err = _classify_status(status, body, headers, ctx, fallback_text=fallback_text, now=now, **kwargs)
    err.provider_said = _said(status, body, _read_body(body), ctx, fallback_text)
    return err


def _classify_status(status: int | None, body: object, headers: Any, ctx: ErrorContext, *, fallback_text: str,
                     now: float | None, **kwargs: Any) -> LLMError:
    info = _read_body(body)
    effective = status if status is not None else _body_status(info)
    text = info.text or fallback_text.lower()
    types = info.types
    codes = info.codes
    err_kwargs = dict(kwargs, status=effective)

    def make(cls: type[LLMError], key: str, **extra: Any) -> LLMError:
        return make_error(cls, key, ctx, **err_kwargs, **extra)

    if effective is not None and 300 <= effective < 400:
        location = headers.get("location", "") if headers is not None else ""
        from urllib.parse import urlsplit

        host = urlsplit(location).hostname or location or ctx.host
        return make(ProviderError, "redirect", message_fields={"host": host})
    if (effective == 401 or "authentication_error" in types or "API_KEY_INVALID" in info.reasons
            or "UNAUTHENTICATED" in info.statuses):
        return make(AuthFailed, "auth")
    if effective == 403 or "permission_error" in types or "PERMISSION_DENIED" in info.statuses:
        return make(NotPermitted, "permission")
    if effective == 402 or "billing_error" in types or "insufficient credits" in text:
        return make(NoCredit, "credit")
    rate_limited = effective == 429 or "rate_limit_error" in types or "RESOURCE_EXHAUSTED" in info.statuses
    # In this order (review G9): too large for the minute, a limit of 0, no credit, a used-up quota, a rate limit.
    if rate_limited and any(p in text for p in _TOO_LARGE_PHRASES):
        return make(ContextTooLong, "tpm_too_large",
                    message_fields={"text": _provider_text(ctx, info, fallback_text).rstrip(".")})
    if rate_limited and _limit_zero(headers, info):
        return make(AccessInactive, "inactive")
    if rate_limited and ("insufficient_quota" in types or "insufficient_quota" in codes):
        # OpenAI: no credit left or the monthly spend cap reached; neither resets by itself.
        return make(NoCredit, "credit")
    if rate_limited and any(q.daily for q in _quota_of(info)):
        # A Google quota per day: waiting minutes does not help, so it is not waited for.
        return make(QuotaExhausted, "quota_daily")
    if rate_limited and (any("perday" in q.lower() or "daily" in q.lower() for q in info.quota_ids)
                         or "free-models-per-day" in text):
        return make(QuotaExhausted, "quota")
    if rate_limited:
        from flextool.assistant.pacing import rate_headers   # pacing imports this module

        input_tokens = any(_input_per_minute(q) for q in _quota_of(info))
        err = make(InputTokensPerMinute if input_tokens else RateLimited, "tpm_input" if input_tokens else "rpm",
                   retry_after=retry_after_from(headers, info, now=now))
        err.rate_headers = rate_headers(headers) if headers is not None else None
        return err
    if effective in (500, None) and any(p in text for p in _UNREADABLE_CALL_PHRASES):
        return make(UnreadableToolCall, "tool_call_unreadable")
    if "overloaded_error" in types or (effective in _BUSY_STATUS):
        streamed = bool(kwargs.get("streamed"))
        return make(ServerBusy, "busy_streamed" if streamed else "busy_local" if ctx.local else "busy",
                    retry_after=retry_after_from(headers, info, now=now))
    phrase_status = effective in (400, 413, None)
    if effective in (400, None) and (("signature" in text and "thinking" in text) or "thought_signature" in text):
        return make(HistoryRejected, "history")
    if ctx.preset_id == "openrouter" and effective == 404 and "endpoint" in text:
        err = make(NoMatchingEndpoint, "routing")
        err.user_message = routing_message(ctx)
        err.args = (err.user_message,)
        return err
    if effective == 404 or "not_found_error" in types or "NOT_FOUND" in info.statuses:
        # The provider's own words: a model in its list can still be retired or closed to new accounts.
        return make(ModelNotFound, "model", message_fields={"text": _provider_text(ctx, info, fallback_text).rstrip(".")})
    if phrase_status and (any(p in text for p in _CONTEXT_PHRASES) or ("exceeds" in text and "context" in text)):
        return make(ContextTooLong, "context")
    if effective in (400, None) and (any(p in text for p in _TOOL_PHRASES)
                                     or ("tool" in text and "not supported" in text)):
        err = make(ToolsUnsupported, "tools")
        err.user_message = tools_message(ctx)
        err.args = (err.user_message,)
        return err
    provider_text = _provider_text(ctx, info, fallback_text)
    if effective is None:
        return make(ProviderError, "other_no_status", message_fields={"text": provider_text})
    return make(ProviderError, "other", message_fields={"status": effective, "text": provider_text})


def _connection_error(ctx: ErrorContext, *, timeout: bool, connect: bool = False, **kwargs: Any) -> LLMError:
    if timeout and connect:
        from flextool.assistant.net import CONNECT_TIMEOUT

        return make_error(TimedOut, "connect_timeout", ctx, message_fields={"seconds": _seconds(CONNECT_TIMEOUT)},
                          **kwargs)
    if timeout:
        return make_error(TimedOut, "timeout", ctx, **kwargs)
    return make_error(Unreachable, "connection_local" if ctx.local else "connection_remote", ctx, **kwargs)


def _connect_timeout(exc: BaseException) -> bool:
    """Whether a library's timeout error came from the connect timeout (the libraries raise it ``from`` the
    ``httpx2`` exception), not the read timeout (review G9, C4-m3)."""
    import httpx2

    return isinstance(exc.__cause__, httpx2.ConnectTimeout)


def _other(exc: BaseException, ctx: ErrorContext, **kwargs: Any) -> LLMError:
    text = _provider_text(ctx, _Body(), f"{type(exc).__name__}: {exc}")
    return make_error(ProviderError, "other_no_status", ctx, message_fields={"text": text}, **kwargs)


def classify_openai(exc: BaseException, ctx: ErrorContext, *, now: float | None = None, **kwargs: Any) -> LLMError:
    """The :class:`LLMError` for an exception of the ``openai`` library (or any other exception)."""
    if isinstance(exc, LLMError):
        return exc
    import openai

    if isinstance(exc, openai.APITimeoutError):
        return _connection_error(ctx, timeout=True, connect=_connect_timeout(exc), **kwargs)
    if isinstance(exc, openai.APIConnectionError):
        return _connection_error(ctx, timeout=False, **kwargs)
    if isinstance(exc, openai.APIStatusError):
        return classify_status(exc.status_code, exc.body, exc.response.headers, ctx, fallback_text=exc.message,
                               now=now, **kwargs)
    if isinstance(exc, openai.APIError):          # an error inside the stream: no status
        return classify_status(None, exc.body, None, ctx, fallback_text=exc.message, now=now, **kwargs)
    return _other(exc, ctx, **kwargs)


def classify_anthropic(exc: BaseException, ctx: ErrorContext, *, now: float | None = None,
                       **kwargs: Any) -> LLMError:
    """The :class:`LLMError` for an exception of the ``anthropic`` library (or any other exception).  An error
    inside the stream arrives as a status error with the stream's status 200 and is classified from its body."""
    if isinstance(exc, LLMError):
        return exc
    import anthropic

    if isinstance(exc, anthropic.APITimeoutError):
        return _connection_error(ctx, timeout=True, connect=_connect_timeout(exc), **kwargs)
    if isinstance(exc, anthropic.APIConnectionError):
        return _connection_error(ctx, timeout=False, **kwargs)
    if isinstance(exc, anthropic.APIStatusError):
        status = exc.status_code if exc.status_code != 200 else None
        return classify_status(status, exc.body, exc.response.headers if status else None, ctx,
                               fallback_text=exc.message, now=now, **kwargs)
    if isinstance(exc, anthropic.APIError):
        return classify_status(None, exc.body, None, ctx, fallback_text=exc.message, now=now, **kwargs)
    return _other(exc, ctx, **kwargs)
