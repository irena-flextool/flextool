"""Every provider error becomes the right error class and message (P1 §4).

The exceptions are the real ones of the provider libraries: each case sends a request through the library with a
mock transport that answers with the synthetic status, headers and body, and classifies what the library raises.
"""
from __future__ import annotations

import email.utils
import json
import time

import httpx2
import pytest

from flextool.assistant import llm_errors as E
from flextool.assistant.llm_errors import (
    ErrorContext,
    classify_anthropic,
    classify_openai,
)

SECRET = "sk-test-0123456789abcdefghijklmnop"


def _ctx(preset_id: str = "openai", **kw) -> ErrorContext:
    defaults = {"preset_id": preset_id, "provider": "Example AI", "model": "example-model",
                "key_source": "environment variable EXAMPLE_KEY", "host": "api.example.test", "port": 443,
                "local": False, "read_timeout": 300.0, "secret": SECRET}
    defaults.update(kw)
    return ErrorContext(**defaults)


def _handler(status: int, body: object = None, headers: dict | None = None, *, sse: str | None = None):
    def handle(request: httpx2.Request) -> httpx2.Response:
        if sse is not None:
            return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=sse.encode())
        content = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        return httpx2.Response(status, headers={"content-type": "application/json", **(headers or {})},
                               content=content)
    return handle


def openai_exception(handle) -> Exception:
    import openai

    client = openai.OpenAI(api_key=SECRET, base_url="https://api.example.test/v1", max_retries=0,
                           http_client=httpx2.Client(transport=httpx2.MockTransport(handle)))
    try:
        stream = client.chat.completions.create(model="example-model", stream=True,
                                                messages=[{"role": "user", "content": "hello"}])
        for _ in stream:
            pass
    except Exception as exc:  # noqa: BLE001 — the exception is the test's subject
        return exc
    raise AssertionError("the request did not fail")


def anthropic_exception(handle) -> Exception:
    import anthropic

    client = anthropic.Anthropic(api_key=SECRET, base_url="https://api.example.test", max_retries=0,
                                 open_telemetry=False,
                                 http_client=httpx2.Client(transport=httpx2.MockTransport(handle)))
    try:
        with client.messages.stream(model="example-model", max_tokens=100,
                                    messages=[{"role": "user", "content": "hello"}]) as stream:
            for _ in stream:
                pass
    except Exception as exc:  # noqa: BLE001
        return exc
    raise AssertionError("the request did not fail")


def _gemini_429(quota_id: str, delay: str) -> list:
    return [{"error": {"code": 429, "message": "You exceeded your current quota.", "status": "RESOURCE_EXHAUSTED",
                       "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                                    "violations": [{"quotaMetric": "generate_content_free_tier_requests",
                                                    "quotaId": quota_id}]},
                                   {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}]}}]


NOW = 1_800_000_000.0
HTTP_DATE = email.utils.formatdate(NOW + 30, usegmt=True)

# (case, library, preset, status, headers, body, class, retry_after)
CASES = [
    # Updated (review G9, C4-m8): no credit or the monthly spend cap, which does not reset by itself.
    ("insufficient_quota", "openai", "openai", 429, {},
     {"error": {"message": "You exceeded your current quota.", "type": "insufficient_quota",
                "code": "insufficient_quota"}}, E.NoCredit, None),
    ("gemini list body per day", "openai", "gemini", 429, {},
     _gemini_429("GenerateRequestsPerDayPerProjectPerModel-FreeTier", "41s"), E.QuotaExhausted, None),
    ("gemini retryDelay", "openai", "gemini", 429, {},
     _gemini_429("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "41s"), E.RateLimited, 41.0),
    ("retry-after-ms", "openai", "openai", 429, {"retry-after-ms": "1500", "retry-after": "9"},
     {"error": {"message": "Rate limit reached.", "type": "requests", "code": "rate_limit_exceeded"}},
     E.RateLimited, 1.5),
    ("retry-after seconds", "openai", "mistral", 429, {"retry-after": "7"},
     {"message": "Requests rate limit exceeded"}, E.RateLimited, 7.0),
    ("retry-after HTTP date", "anthropic", "anthropic", 429, {"retry-after": HTTP_DATE},
     {"type": "error", "error": {"type": "rate_limit_error", "message": "Rate limited."}}, E.RateLimited, 30.0),
    ("rate limit without a wait", "openai", "openai", 429, {},
     {"error": {"message": "Slow down.", "type": "requests"}}, E.RateLimited, None),
    ("openrouter no endpoints", "openai", "openrouter", 404, {},
     {"error": {"message": "No endpoints found matching your data policy (Free model publication).",
                "code": 404}}, E.NoMatchingEndpoint, None),
    ("openrouter no credit", "openai", "openrouter", 402, {},
     {"error": {"message": "Insufficient credits. Add more using https://example.test/credits", "code": 402}},
     E.NoCredit, None),
    ("openrouter free models per day", "openai", "openrouter", 429, {},
     {"error": {"message": "Rate limit exceeded: free-models-per-day. Add credits to unlock more.", "code": 429,
                "metadata": {"headers": {"X-RateLimit-Limit": "50"}}}}, E.QuotaExhausted, None),
    ("anthropic 529", "anthropic", "anthropic", 529, {},
     {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}, E.ServerBusy, None),
    ("anthropic billing_error", "anthropic", "anthropic", 400, {},
     {"type": "error", "error": {"type": "billing_error", "message": "Your credit balance is too low."}},
     E.NoCredit, None),
    ("anthropic prompt too long", "anthropic", "anthropic", 400, {},
     {"type": "error", "error": {"type": "invalid_request_error",
                                 "message": "prompt is too long: 210000 tokens > 200000 maximum"}},
     E.ContextTooLong, None),
    ("gemini input token count", "openai", "gemini", 400, {},
     [{"error": {"code": 400, "status": "INVALID_ARGUMENT",
                 "message": "The input token count (1200000) exceeds the maximum number of tokens allowed "
                            "(1048576)."}}], E.ContextTooLong, None),
    ("openai context_length_exceeded", "openai", "openai", 400, {},
     {"error": {"message": "This model's maximum context length is 128000 tokens.", "type": "invalid_request_error",
                "code": "context_length_exceeded"}}, E.ContextTooLong, None),
    ("anthropic signature rejected", "anthropic", "anthropic", 400, {},
     {"type": "error", "error": {"type": "invalid_request_error",
                                 "message": "messages.1.content.0: Invalid `signature` in `thinking` block"}},
     E.HistoryRejected, None),
    ("gemini thought_signature", "openai", "gemini", 400, {},
     [{"error": {"code": 400, "status": "INVALID_ARGUMENT",
                 "message": "Function call is missing a thought_signature in functionCall parts."}}],
     E.HistoryRejected, None),
    ("ollama tools", "openai", "ollama", 400, {},
     {"error": {"message": "registry.ollama.ai/library/example:2b does not support tools",
                "type": "invalid_request_error", "param": None, "code": None}}, E.ToolsUnsupported, None),
    ("vllm tool choice", "openai", "openai_compatible", 400, {},
     {"object": "error", "type": "BadRequestError", "code": 400,
      "message": "\"auto\" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set"},
     E.ToolsUnsupported, None),
    ("401", "openai", "openai", 401, {},
     {"error": {"message": "Incorrect API key provided.", "type": "invalid_request_error",
                "code": "invalid_api_key"}}, E.AuthFailed, None),
    ("gemini API_KEY_INVALID", "openai", "gemini", 400, {},
     [{"error": {"code": 400, "message": "API key not valid. Please pass a valid API key.",
                 "status": "INVALID_ARGUMENT",
                 "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo",
                              "reason": "API_KEY_INVALID"}]}}], E.AuthFailed, None),
    ("403", "anthropic", "anthropic", 403, {},
     {"type": "error", "error": {"type": "permission_error", "message": "Not allowed."}}, E.NotPermitted, None),
    ("404 model", "openai", "openai", 404, {},
     {"error": {"message": "The model does not exist.", "type": "invalid_request_error",
                "code": "model_not_found"}}, E.ModelNotFound, None),
    ("500", "openai", "mistral", 500, {}, {"message": "Internal error"}, E.ServerBusy, None),
    ("503 with retry-after", "openai", "gemini", 503, {"retry-after": "3"},
     [{"error": {"code": 503, "message": "The model is overloaded.", "status": "UNAVAILABLE"}}],
     E.ServerBusy, 3.0),
    ("307 redirect", "openai", "openai_compatible", 307, {"location": "https://elsewhere.example.test/v1/chat"},
     {"message": "moved"}, E.ProviderError, None),
    ("422 other", "openai", "mistral", 422, {},
     {"object": "error", "message": f"Extra inputs are not permitted (key {SECRET})", "type": "invalid_request_error"},
     E.ProviderError, None),
]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_status_errors(case):
    _, library, preset_id, status, headers, body, cls, retry_after = case
    ctx = _ctx(preset_id)
    if library == "openai":
        exc = openai_exception(_handler(status, body, headers))
        err = classify_openai(exc, ctx, now=NOW)
    else:
        exc = anthropic_exception(_handler(status, body, headers))
        err = classify_anthropic(exc, ctx, now=NOW)
    assert type(err) is cls
    assert err.retry_after == pytest.approx(retry_after) if retry_after is not None else err.retry_after is None
    assert err.user_message and SECRET not in err.user_message
    assert err.provider == "Example AI"


#: What each case's provider said (calibration fixes 2026-11-10): the HTTP status, Google's status name else the
#: error code else the error type, and the first message, redacted.
SAID = {
    "insufficient_quota": (429, "insufficient_quota", "You exceeded your current quota."),
    "gemini list body per day": (429, "RESOURCE_EXHAUSTED", "You exceeded your current quota."),
    "gemini retryDelay": (429, "RESOURCE_EXHAUSTED", "You exceeded your current quota."),
    "retry-after-ms": (429, "rate_limit_exceeded", "Rate limit reached."),
    "retry-after seconds": (429, "", "Requests rate limit exceeded"),
    "retry-after HTTP date": (429, "rate_limit_error", "Rate limited."),
    "rate limit without a wait": (429, "requests", "Slow down."),
    "openrouter no endpoints": (404, "", "No endpoints found matching your data policy (Free model publication)."),
    "openrouter no credit": (402, "", "Insufficient credits. Add more using https://example.test/credits"),
    "openrouter free models per day": (429, "", "Rate limit exceeded: free-models-per-day. Add credits to unlock "
                                                "more."),
    "anthropic 529": (529, "overloaded_error", "Overloaded"),
    "anthropic billing_error": (400, "billing_error", "Your credit balance is too low."),
    "anthropic prompt too long": (400, "invalid_request_error", "prompt is too long: 210000 tokens > 200000 maximum"),
    "gemini input token count": (400, "INVALID_ARGUMENT", "The input token count (1200000) exceeds the maximum "
                                                          "number of tokens allowed (1048576)."),
    "openai context_length_exceeded": (400, "context_length_exceeded",
                                       "This model's maximum context length is 128000 tokens."),
    "anthropic signature rejected": (400, "invalid_request_error",
                                     "messages.1.content.0: Invalid `signature` in `thinking` block"),
    "gemini thought_signature": (400, "INVALID_ARGUMENT",
                                 "Function call is missing a thought_signature in functionCall parts."),
    "ollama tools": (400, "invalid_request_error", "registry.ollama.ai/library/example:2b does not support tools"),
    "vllm tool choice": (400, "BadRequestError", "\"auto\" tool choice requires --enable-auto-tool-choice and "
                                                 "--tool-call-parser to be set"),
    "401": (401, "invalid_api_key", "Incorrect API key provided."),
    "gemini API_KEY_INVALID": (400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key."),
    "403": (403, "permission_error", "Not allowed."),
    "404 model": (404, "model_not_found", "The model does not exist."),
    "500": (500, "", "Internal error"),
    "503 with retry-after": (503, "UNAVAILABLE", "The model is overloaded."),
    "307 redirect": (307, "", "moved"),
    "422 other": (422, "invalid_request_error", "Extra inputs are not permitted (key <key>)"),
}


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_every_class_keeps_what_the_provider_said(case):
    """Calibration fixes 2026-11-10: whatever class an error answer becomes, it keeps the provider's own status,
    code and message (redacted, no key), for the calibration's spent.json and console."""
    name, library, preset_id, status, headers, body, _cls, _ = case
    ctx = _ctx(preset_id)
    if library == "openai":
        err = classify_openai(openai_exception(_handler(status, body, headers)), ctx, now=NOW)
    else:
        err = classify_anthropic(anthropic_exception(_handler(status, body, headers)), ctx, now=NOW)
    assert err.provider_said == E.ProviderSaid(*SAID[name])
    assert E.provider_said_line(err) == f"Example AI said: {err.provider_said.text()}"


def test_what_the_provider_said_inside_a_stream_capped_and_none():
    """An error inside a stream has no HTTP status: the code its body names stands in.  A long message is cut at
    300 characters.  An error of no provider answer (a refused connection) has nothing the provider said."""
    long = "Quota exceeded for metric: input_token_count. " * 10
    err = E.classify_status(None, [{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": long}}], None,
                            _ctx("gemini"))
    assert isinstance(err, E.RateLimited)
    said = err.provider_said
    assert (said.status, said.code) == (429, "RESOURCE_EXHAUSTED")
    assert len(said.message) == E.MAX_PROVIDER_SAID and said.message == long[:299] + "…"
    assert said.text() == f"429 RESOURCE_EXHAUSTED — {said.message}"

    def refuse(request):
        raise httpx2.ConnectError("refused", request=request)

    err = classify_openai(openai_exception(refuse), _ctx())
    assert err.provider_said is None and E.provider_said_line(err) is None


def test_messages_name_the_route_model_and_key_source():
    ctx = _ctx("openrouter", zdr=True)
    auth = classify_openai(openai_exception(_handler(401, {"error": {"message": "No auth"}})), ctx)
    assert auth.user_message == ("Example AI did not accept the key (from environment variable EXAMPLE_KEY). Check "
                                 "the key, or store a new one with flextool-assistant --set-key.")
    routing = classify_openai(openai_exception(_handler(404, {"error": {"message": "No endpoints found.",
                                                                         "code": 404}})), ctx)
    assert routing.user_message == (
        "No provider for this model matches your privacy setting (data collection: deny). Change it in settings or "
        "pick another model. Some models also have no provider that supports tool calls. The setting to use only "
        "providers that keep no copy also applies.")
    tools = classify_openai(openai_exception(_handler(400, {"error": {"message": "x does not support tools"}})),
                            _ctx("ollama", model="example:2b"))
    assert tools.user_message == (
        "The model example:2b does not support tool calls, so the assistant cannot read your case data. Pick a model "
        "that supports tools. The model's page in the Ollama library lists 'tools' when it does.")
    model = classify_anthropic(anthropic_exception(_handler(
        404, {"type": "error", "error": {"type": "not_found_error", "message": "model: x"}})), ctx)
    assert model.user_message == ("Example AI did not find the model example-model (its answer: model: x). See "
                                  "the available ones with flextool-assistant --list-models.")


def test_redirect_names_the_target_and_other_errors_quote_the_provider():
    redirect = classify_openai(openai_exception(_handler(307, {"m": 1}, {"location": "https://elsewhere.example.test/x"})),
                               _ctx("openai_compatible"))
    assert redirect.user_message == ("The endpoint redirected to elsewhere.example.test. FlexTool does not follow "
                                     "redirects, so nothing was sent there.")
    other = classify_openai(openai_exception(_handler(
        422, {"object": "error", "message": f"Extra inputs are not permitted ({SECRET})"})), _ctx("mistral"))
    assert other.user_message == ("Example AI returned an error (status 422): Extra inputs are not permitted "
                                  "(<key>)")
    assert other.status == 422
    long_text = classify_openai(openai_exception(_handler(418, {"message": "x" * 2000})), _ctx())
    assert len(long_text.user_message) < 600


def test_errors_inside_the_stream_are_classified_from_the_body():
    openrouter_chunks = (
        'data: {"id":"gen-1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,'
        '"delta":{"role":"assistant","content":"Hel"},"finish_reason":null}]}\n\n'
        'data: {"id":"gen-1","object":"chat.completion.chunk","created":1,"model":"m","error":{"code":502,'
        '"message":"Provider returned error"},"choices":[{"index":0,"delta":{"content":""},'
        '"finish_reason":"error"}]}\n\n')
    import openai

    exc = openai_exception(_handler(200, sse=openrouter_chunks))
    assert isinstance(exc, openai.APIError) and not isinstance(exc, openai.APIStatusError)
    err = classify_openai(exc, _ctx("openrouter"), streamed=True, partial_text="Hel")
    assert isinstance(err, E.ServerBusy) and err.streamed and err.partial_text == "Hel"
    assert err.user_message == "Example AI stopped the answer because it is overloaded. Ask again in a moment."

    unknown = classify_openai(openai_exception(_handler(200, sse='data: {"error":{"message":"odd failure"}}\n\n')),
                              _ctx("openrouter"))
    assert isinstance(unknown, E.ProviderError)
    assert unknown.user_message == "Example AI returned an error: odd failure"

    anthropic_events = (
        'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1","type":"message",'
        '"role":"assistant","model":"m","content":[],"stop_reason":null,"stop_sequence":null,'
        '"usage":{"input_tokens":10,"output_tokens":1}}}\n\n'
        'event: error\ndata: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}\n\n')
    exc = anthropic_exception(_handler(200, sse=anthropic_events))
    err = classify_anthropic(exc, _ctx("anthropic"))
    assert isinstance(err, E.ServerBusy) and err.status == 529


@pytest.mark.parametrize("library", ["openai", "anthropic"])
def test_connection_errors_and_timeouts(library):
    def refuse(request):
        raise httpx2.ConnectError("refused", request=request)

    def slow(request):
        raise httpx2.ReadTimeout("slow", request=request)

    exception = openai_exception if library == "openai" else anthropic_exception
    classify = classify_openai if library == "openai" else classify_anthropic
    local = _ctx("ollama", host="localhost", port=11434, local=True, server_name="Ollama", read_timeout=900.0)
    remote = _ctx("gemini", host="generativelanguage.example.test")
    err = classify(exception(refuse), local)
    assert isinstance(err, E.Unreachable) and err.local
    assert err.user_message == "Nothing answers at localhost:11434. Start Ollama and try again."
    err = classify(exception(refuse), remote)
    assert isinstance(err, E.Unreachable) and not err.local
    assert err.user_message == ("Could not reach generativelanguage.example.test. Check the internet connection or "
                                "proxy settings.")
    err = classify(exception(slow), local)
    assert isinstance(err, E.TimedOut)
    assert err.user_message == "Example AI did not answer within 900 seconds."


def test_an_error_that_is_not_the_librarys_is_a_provider_error():
    err = classify_openai(ValueError(f"bad json {SECRET}"), _ctx())
    assert isinstance(err, E.ProviderError)
    assert SECRET not in err.user_message and "ValueError" in err.user_message
    already = E.Cancelled("Stopped.")
    assert classify_anthropic(already, _ctx()) is already


def test_retry_after_parsing():
    assert E.retry_after_from({"retry-after": "  12 "}) == 12.0
    assert E.retry_after_from({"retry-after": "nonsense"}) is None
    assert E.retry_after_from({"retry-after": email.utils.formatdate(NOW - 60, usegmt=True)}, now=NOW) == 0.0
    assert E.retry_after_from({}, E._read_body({"retryDelay": "2.5s"})) == 2.5
    assert E.retry_after_from(None) is None
    assert abs(E.retry_after_from({"retry-after": email.utils.formatdate(time.time() + 100, usegmt=True)})
               - 100) < 3


def test_refusal_reasons_are_plain_english():
    ctx = _ctx()
    assert E.refusal_reason("cyber", None) == "computer security"
    assert E.refusal_reason("bio", "") == "biology or chemistry safety"
    assert E.refusal_reason("general_harms", None) == "possible harm"
    assert E.refusal_reason("frontier_llm", None) == "the provider's usage rules"
    assert E.refusal_reason("cyber", "The request asks for malware.") == "The request asks for malware."
    assert E.refusal_reason(None, None) is None
    assert E.refusal_message(ctx, "computer security") == ("Example AI declined to answer this request (reason: "
                                                           "computer security).")
    assert E.refusal_message(ctx, None) == "Example AI declined to answer this request."


def test_every_error_class_has_a_message():
    codes = {cls.code for cls in E.ERROR_CLASSES}
    for code in codes:
        assert any(key == code or key.startswith(code + "_") for key in E.MESSAGES), code


# --- a reported limit of 0 (review G9, OR-2) ---------------------------------------------------------------------
OR2_BODY = {"message": "Rate limit exceeded", "type": "rate_limited", "param": None, "code": "1300"}


def _gemini_limit(quota_id: str, limit: str) -> list:
    body = _gemini_429(quota_id, "41s")
    body[0]["error"]["message"] = (
        "You exceeded your current quota, please check your plan and billing details.\n* Quota exceeded for "
        f"metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: {limit}, model: "
        "gemini-2.5-pro")
    return body


def test_mistral_limit_zero_is_access_inactive():
    headers = {"x-ratelimit-limit-req-minute": "0", "x-ratelimit-remaining-req-minute": "0"}
    err = classify_openai(openai_exception(_handler(429, OR2_BODY, headers)), _ctx("mistral", provider="Mistral"))
    assert type(err) is E.AccessInactive and err.code == "inactive"
    assert err.user_message == ("Your Mistral account allows 0 requests per minute: its API access is not active "
                                "yet. Activate a plan in your Mistral account, then ask again.")


@pytest.mark.parametrize("headers", [
    {"x-ratelimit-limit-req-minute": "750", "x-ratelimit-remaining-req-minute": "0"},   # paid, at its limit
    {"x-ratelimit-limit-req-minute": "0", "x-ratelimit-remaining-req-minute": "3"},     # not both 0
    {"x-ratelimit-limit-req-minute": "0, 0", "x-ratelimit-remaining-req-minute": "0"},  # unreadable
    {}], ids=["paid-750-remaining-0", "remaining-3", "duplicated", "no-headers"])
def test_a_limit_above_zero_or_unread_is_a_rate_limit(headers):
    """Guard: only a reported limit of 0 with none remaining is an inactive account."""
    err = classify_openai(openai_exception(_handler(429, OR2_BODY, headers)), _ctx("mistral", provider="Mistral"))
    assert type(err) is E.RateLimited
    assert err.rate_headers == {k: v for k, v in headers.items()}


@pytest.mark.parametrize("quota_id", ["GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
                                      "GenerateRequestsPerDayPerProjectPerModel-FreeTier"])
def test_gemini_limit_zero_is_access_inactive(quota_id):
    """A Google quota of 0 per minute or per day never resets (critique G9 item 6; before: a rate limit or a
    quota "that resets on the provider's schedule")."""
    err = classify_openai(openai_exception(_handler(429, _gemini_limit(quota_id, "0"))),
                          _ctx("gemini", provider="Google Gemini"))
    assert type(err) is E.AccessInactive
    assert err.user_message.startswith("Your Google Gemini account allows 0 requests per minute")


def test_gemini_limit_above_zero_is_unchanged():
    """Guard: "limit: 0.5" and "limit: 15" are not a limit of 0."""
    minute = classify_openai(openai_exception(_handler(429, _gemini_limit(
        "GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "15"))), _ctx("gemini"))
    day = classify_openai(openai_exception(_handler(429, _gemini_limit(
        "GenerateRequestsPerDayPerProjectPerModel-FreeTier", "0.5"))), _ctx("gemini"))
    assert (type(minute), minute.retry_after, type(day)) == (E.RateLimited, 41.0, E.QuotaExhausted)


def test_a_stream_error_without_headers_is_classified():
    """Guard: a stream error has no headers (``None``)."""
    err = E.classify_status(None, {"error": {"type": "rate_limit_error", "message": "slow"}}, None, _ctx())
    assert type(err) is E.RateLimited and err.rate_headers is None


# --- OpenAI's two 429 kinds and the connect timeout (review G9: C4-m8, C4-m3) ------------------------------------
TOO_LARGE = ("Request too large for gpt-4o in organization org-example on tokens per min (TPM): Limit 30000, "
             "Requested 45000. The input or output tokens must be reduced in order to run successfully.")


def test_openai_request_too_large_is_context():
    """A request above the per-minute token limit cannot succeed on a retry: the session's trim, with its own
    message quoting the provider (before: a retried rate limit)."""
    body = {"error": {"message": TOO_LARGE, "type": "tokens", "param": None, "code": "rate_limit_exceeded"}}
    err = classify_openai(openai_exception(_handler(429, body)), _ctx(provider="OpenAI"))
    assert type(err) is E.ContextTooLong and err.message_key == "tpm_too_large"
    assert err.user_message == (
        "This request is larger than your OpenAI account may send in one minute (its answer: " + TOO_LARGE[:-1]
        + "). A shorter conversation fits: start a new one, or switch to a model or plan with a higher limit.")


def test_insufficient_quota_is_no_credit():
    body = {"error": {"message": "You exceeded your current quota, please check your plan and billing details.",
                      "type": "insufficient_quota", "param": None, "code": "insufficient_quota"}}
    err = classify_openai(openai_exception(_handler(429, body)), _ctx(provider="OpenAI"))
    assert type(err) is E.NoCredit
    assert err.user_message == ("Your OpenAI account has no credit left. Add credit in your provider account, or "
                                "switch provider.")


@pytest.mark.parametrize("library", ["openai", "anthropic"])
def test_connect_timeout_names_the_connect_time(library):
    """A connect timeout (10 s) is not reported as the 300 s read timeout (before: "did not answer within 300
    seconds")."""
    def unreachable(request):
        raise httpx2.ConnectTimeout("timed out", request=request)

    exception = openai_exception if library == "openai" else anthropic_exception
    classify = classify_openai if library == "openai" else classify_anthropic
    err = classify(exception(unreachable), _ctx(host="llm.example.test"))
    assert isinstance(err, E.TimedOut)
    assert err.user_message == "Could not connect to llm.example.test within 10 seconds."
