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
    "busy_streamed": "{provider} stopped the answer because it is overloaded. Ask again in a moment.",
    "quota": ("Your {provider} quota is used up for now. It resets on the provider's schedule, which FlexTool "
              "cannot see. You can switch to another provider or model."),
    "credit": ("Your {provider} account has no credit left. Add credit in your provider account, or switch "
               "provider."),
    "auth": ("{provider} did not accept the key (from {key_source}). Check the key, or store a new one with "
             "flextool-assistant --set-key."),
    "permission": "{provider} accepted the key but does not allow it to use {model}.",
    "context_retry": ("The conversation is too long for {model}. FlexTool removed older tool results and tried "
                      "again."),
    "context": ("The conversation is still too long for {model}. Start a new one with /new (/new carry keeps a "
                "short summary of this one)."),
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

    @property
    def params(self) -> dict[str, str]:
        """The values the message names (``provider``, ``model``, ``key_source``, ``env_name``, …) as text."""
        return {key: str(value) for key, value in self.fields.items()}

    def __str__(self) -> str:
        return self.user_message


class RateLimited(LLMError):
    code = "rpm"


class AccessInactive(LLMError):
    """The provider reports a request limit of 0: the account's API access is not active (review G9, OR-2).  Not
    retried, and the pacing does not change."""

    code = "inactive"


class ServerBusy(LLMError):
    code = "busy"


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


ERROR_CLASSES: tuple[type[LLMError], ...] = (
    RateLimited, AccessInactive, ServerBusy, QuotaExhausted, NoCredit, AuthFailed, NotPermitted, ContextTooLong, HistoryRejected,
    NoMatchingEndpoint, ToolsUnsupported, ModelNotFound, Unreachable, TimedOut, ContextTooSmall, Refused, Truncated,
    NoKey, Cancelled, ProviderError)


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
        elif key in ("message", "raw") and isinstance(value, str):
            out.messages.append(value)
    return out


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
    from flextool.assistant.redact import Redactor

    text = next((m for m in body.messages if m.strip()), "") or fallback
    if ctx.secret:
        text = text.replace(ctx.secret, "<key>")
    text = Redactor(None).redact(" ".join(text.split()))
    return text if len(text) <= MAX_PROVIDER_TEXT else text[:MAX_PROVIDER_TEXT - 1] + "…"


def classify_status(status: int | None, body: object, headers: Any, ctx: ErrorContext, *, fallback_text: str = "",
                    now: float | None = None, **kwargs: Any) -> LLMError:
    """The error for an answer with HTTP *status* (``None`` for an error inside a stream), *body* (decoded JSON or
    text) and *headers*; *kwargs* go to :class:`LLMError` (``partial_usage``, ``streamed``, ``partial_text``)."""
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
    if rate_limited and _limit_zero(headers, info):
        return make(AccessInactive, "inactive")
    if rate_limited and ("insufficient_quota" in types or "insufficient_quota" in codes
                         or any("perday" in q.lower() or "daily" in q.lower() for q in info.quota_ids)
                         or "free-models-per-day" in text):
        return make(QuotaExhausted, "quota")
    if rate_limited:
        from flextool.assistant.pacing import rate_headers   # pacing imports this module

        err = make(RateLimited, "rpm", retry_after=retry_after_from(headers, info, now=now))
        err.rate_headers = rate_headers(headers) if headers is not None else None
        return err
    if "overloaded_error" in types or (effective in _BUSY_STATUS):
        streamed = bool(kwargs.get("streamed"))
        return make(ServerBusy, "busy_streamed" if streamed else "busy",
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


def _connection_error(ctx: ErrorContext, *, timeout: bool, **kwargs: Any) -> LLMError:
    if timeout:
        return make_error(TimedOut, "timeout", ctx, **kwargs)
    return make_error(Unreachable, "connection_local" if ctx.local else "connection_remote", ctx, **kwargs)


def _other(exc: BaseException, ctx: ErrorContext, **kwargs: Any) -> LLMError:
    text = _provider_text(ctx, _Body(), f"{type(exc).__name__}: {exc}")
    return make_error(ProviderError, "other_no_status", ctx, message_fields={"text": text}, **kwargs)


def classify_openai(exc: BaseException, ctx: ErrorContext, *, now: float | None = None, **kwargs: Any) -> LLMError:
    """The :class:`LLMError` for an exception of the ``openai`` library (or any other exception)."""
    if isinstance(exc, LLMError):
        return exc
    import openai

    if isinstance(exc, openai.APITimeoutError):
        return _connection_error(ctx, timeout=True, **kwargs)
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
        return _connection_error(ctx, timeout=True, **kwargs)
    if isinstance(exc, anthropic.APIConnectionError):
        return _connection_error(ctx, timeout=False, **kwargs)
    if isinstance(exc, anthropic.APIStatusError):
        status = exc.status_code if exc.status_code != 200 else None
        return classify_status(status, exc.body, exc.response.headers if status else None, ctx,
                               fallback_text=exc.message, now=now, **kwargs)
    if isinstance(exc, anthropic.APIError):
        return classify_status(None, exc.body, None, ctx, fallback_text=exc.message, now=now, **kwargs)
    return _other(exc, ctx, **kwargs)
