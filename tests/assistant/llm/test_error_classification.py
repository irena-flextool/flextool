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
                "local": False, "read_timeout": 120.0, "secret": SECRET}
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
    ("insufficient_quota", "openai", "openai", 429, {},
     {"error": {"message": "You exceeded your current quota.", "type": "insufficient_quota",
                "code": "insufficient_quota"}}, E.QuotaExhausted, None),
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
    assert model.user_message == ("Example AI has no model named example-model. See the available ones with "
                                  "flextool-assistant --list-models.")


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
