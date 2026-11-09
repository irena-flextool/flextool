"""The model probes and the route they give (P1 §15 test 9).

A ``http.server`` on ``127.0.0.1`` plays Ollama (``/api/show``, ``/api/generate``, ``/api/ps``), LM Studio (v1 and
v0 model endpoints) and OpenRouter's model list.  Checked: a small loaded context stops the chat before any model
request (the OLLAMA_CONTEXT_LENGTH message); no ``tools`` capability stops it; ``remote_host`` and ``:cloud`` /
``-cloud`` names give the Ollama cloud route and skip the load; an unreadable ``/api/show`` sends nothing more;
nothing but the model and ``keep_alive`` is sent to load a model; a probe of this computer ignores the proxy
variables, and every probe takes the route of the requests (C4-m5: the same proxy decision, NO_PROXY and the system
exceptions included, and the remote timeout for a remote server).  A second raw server plays the proxy; a remote
host is either the loopback server with ``net.endpoint_kind`` patched to "remote", or ``example.org`` resolved to
the loopback server (so nothing leaves this computer even when the route were wrong).
"""
from __future__ import annotations

import socket
import threading
import time
import urllib.request
from collections.abc import Iterator

import pytest

from flextool.assistant import endpoint, net, probes
from flextool.assistant.llm_adapter import Connection, ToolSpec, Waiting
from flextool.assistant.llm_errors import (
    Cancelled,
    ContextTooSmall,
    ProviderError,
    ServerBusy,
    ToolsUnsupported,
    LLMError,
    Unreachable,
)
from flextool.assistant.llm_adapter import Message, make_backend
from flextool.assistant.providers import PRESETS
from tests.assistant.llm.local_server import Answer, FakeServer, RawServer
from tests.assistant.llm.wire import connection, request

@pytest.fixture
def server() -> Iterator[FakeServer]:
    fake = FakeServer()
    yield fake
    fake.close()


def _conn(preset_id: str, base_url: str, model: str) -> tuple[object, Connection]:
    preset = PRESETS[preset_id]
    return preset, Connection(base_url=base_url, model=model, api_key="", key_source="none", knobs={},
                              route=endpoint.route_for(preset, base_url, model=model))


def _ollama(server: FakeServer, model: str = "qwen3:14b", *, capabilities: list[str] | None = None,
            loaded_context: int = 32768, size: int = 9_000_000_000, size_vram: int = 9_000_000_000,
            show_extra: dict | None = None) -> tuple[object, Connection]:
    show = {"capabilities": ["completion", "tools"] if capabilities is None else capabilities,
            "model_info": {"general.architecture": "qwen3", "qwen3.context_length": 40960}, **(show_extra or {})}
    server.routes[("POST", "/api/show")] = (200, show)
    server.routes[("POST", "/api/generate")] = (200, {"model": model, "done": True, "done_reason": "load"})
    server.routes[("GET", "/api/ps")] = (200, {"models": [
        {"name": "other:7b", "model": "other:7b", "size": 1, "size_vram": 1, "context_length": 2048},
        {"name": model, "model": model, "size": size, "size_vram": size_vram, "context_length": loaded_context}]})
    return _conn("ollama", f"http://127.0.0.1:{server.port}/v1", model)


def test_needed_tokens():
    # Never below the smallest window the assistant works with (P3 §3.3).
    assert probes.needed_tokens(("x" * 400,), ()) == 12_288
    # 40 000 letters of text: ceil(40 000 / 3.2) + 4 = 12 504; the empty tool list "[]" is ceil(2 / 2.5) + 4 = 5;
    # with the basic answer (4 096) that is 16 605 tokens, which must stay within 95 % of the context.
    assert probes.needed_tokens(("x" * 40_000,), ()) == 17_479          # ceil(16 605 / 0.95)
    tool = ToolSpec("t", "d", {"type": "object"})
    assert probes.needed_tokens(("x" * 40_000,), (tool,)) > probes.needed_tokens(("x" * 40_000,), ())


def test_ollama_small_loaded_context_sends_nothing(server):
    preset, conn = _ollama(server, loaded_context=4096)
    with pytest.raises(ContextTooSmall) as info:
        probes.probe(preset, conn, threading.Event(), needed=9000)
    assert info.value.user_message == (
        "This model runs with a 4096-token context, too small for the assistant's instructions and tools (about "
        "9000). Set OLLAMA_CONTEXT_LENGTH=16384 (or num_ctx in a Modelfile) and restart Ollama.")
    assert server.paths == ["/api/show", "/api/generate", "/api/ps"]
    # Loading sends the model and how long to keep it, nothing else: no thread or graphics-card limits.
    assert server.requests[1][2] == {"model": "qwen3:14b", "keep_alive": "10m"}
    assert server.requests[0][2] == {"model": "qwen3:14b"}


def test_ollama_local_model_facts_and_route(server):
    preset, conn = _ollama(server, loaded_context=16384, size=10, size_vram=6)
    facts = probes.probe(preset, conn, threading.Event(), needed=9000)
    assert (facts.context_window, facts.loaded_context, facts.context, facts.tools, facts.remote_host) == (
        40960, 16384, 16384, True, None)
    assert facts.partly_on_cpu and facts.notes == (
        "This model doesn't fit fully in graphics memory and will be slower.",)
    # The settings show the hint from the last probe (they never probe themselves, P1-O10).
    assert probes.recent_facts("ollama", conn.base_url + "/", conn.model) == facts
    assert probes.recent_facts("ollama", conn.base_url, "another-model") is None
    route = endpoint.route_for(preset, conn.base_url, facts=facts)
    assert (route.kind, route.label, route.forwarded_to, route.proxy_host) == ("loopback", "Ollama", None, None)
    assert endpoint.indicator_text(route) == (
        f"Endpoint on this computer (127.0.0.1:{server.port}). FlexTool sends data only there; if that server "
        "forwards requests, data leaves this computer.")


def test_ollama_model_without_tools(server):
    preset, conn = _ollama(server, capabilities=["completion"])
    with pytest.raises(ToolsUnsupported) as info:
        probes.probe(preset, conn, threading.Event(), needed=9000)
    assert "The model's page in the Ollama library lists 'tools' when it does." in info.value.user_message
    assert server.paths == ["/api/show"]


@pytest.mark.parametrize("remote_host,host,label,indicator", [
    ("https://ollama.com:443", "ollama.com", "Ollama cloud",
     "Data goes to: Ollama cloud (ollama.com), forwarded by Ollama on this computer."),
    ("https://gpu.example.org", "gpu.example.org", "Ollama, forwarded to gpu.example.org",
     "Data goes to: gpu.example.org, forwarded by Ollama on this computer."),
])
def test_ollama_remote_host_is_a_remote_route(server, remote_host, host, label, indicator):
    preset, conn = _ollama(server, "gpt-oss:120b", show_extra={"remote_host": remote_host,
                                                               "remote_model": "gpt-oss:120b"})
    facts = probes.probe(preset, conn, threading.Event(), needed=10**9)      # no context check for a cloud model
    assert facts.remote_host == host and facts.loaded_context is None and facts.context_window == 40960
    assert server.paths == ["/api/show"]                                     # not loaded here
    route = endpoint.route_for(preset, conn.base_url, facts=facts)
    assert (route.kind, route.label, route.forwarded_to) == ("remote", label, host)
    assert endpoint.indicator_text(route) == indicator


@pytest.mark.parametrize("model", ["gpt-oss:120b-cloud", "deepseek-v3.1:671b-cloud", "kimi:cloud"])
def test_ollama_cloud_by_name(server, model):
    preset, conn = _ollama(server, model)
    facts = probes.probe(preset, conn, threading.Event(), needed=10**9)
    assert facts.remote_host == "ollama.com"
    assert server.paths == ["/api/show"]
    assert endpoint.route_for(preset, conn.base_url, facts=facts).kind == "remote"
    # Without facts (the window opening), the name alone already gives the remote route.
    assert endpoint.route_for(preset, conn.base_url, model=model).label == "Ollama cloud"


@pytest.mark.parametrize("answer,error", [
    ((500, {"error": "internal"}), ServerBusy),
    ((200, "this is not JSON"), ProviderError),
    ((200, ["a", "list"]), ProviderError),
])
def test_unreadable_show_sends_nothing(server, answer, error):
    preset, conn = _ollama(server, "gpt-oss:120b-cloud")
    server.routes[("POST", "/api/show")] = answer
    with pytest.raises(error):
        probes.probe(preset, conn, threading.Event(), needed=9000)
    assert server.paths == ["/api/show"]


def test_no_server_is_unreachable():
    preset, conn = _conn("ollama", "http://127.0.0.1:9/v1", "kimi:cloud")
    with pytest.raises(Unreachable) as info:
        probes.probe(preset, conn, threading.Event())
    assert info.value.user_message == "Nothing answers at 127.0.0.1:9. Start Ollama and try again."


def test_probes_ignore_the_proxy_variables(server, monkeypatch):
    for name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    preset, conn = _ollama(server)
    assert probes.probe(preset, conn, threading.Event(), needed=9000).loaded_context == 32768


def test_cancel_while_the_model_loads(server, monkeypatch):
    monkeypatch.setattr(probes, "WAIT_INTERVAL", 0.05)
    preset, conn = _ollama(server)

    def slow_load(_body: object) -> Answer:
        server.release.wait(30)
        return 200, {"done": True}

    server.routes[("POST", "/api/generate")] = slow_load
    cancel = threading.Event()
    events: list[object] = []
    timer = threading.Timer(0.4, cancel.set)
    timer.start()
    started = time.monotonic()
    with pytest.raises(Cancelled):
        probes.probe(preset, conn, cancel, needed=9000, on_event=events.append)
    assert time.monotonic() - started < 2.0
    assert any(isinstance(e, Waiting) for e in events)
    assert "/api/ps" not in server.paths


def _lmstudio(server: FakeServer, *, loaded_context: int | None, capabilities: list[str],
              v1: bool = True) -> tuple[object, Connection]:
    model = "qwen/qwen3-8b"
    if v1:
        server.routes[("GET", "/api/v1/models")] = (200, {"data": [
            {"id": model, "context_length": 32768, "loaded": loaded_context is not None}]})
    details = {"id": model, "max_context_length": 32768, "capabilities": capabilities,
               "state": "loaded" if loaded_context is not None else "not-loaded"}
    if loaded_context is not None:
        details["loaded_context_length"] = loaded_context
    server.routes[("GET", f"/api/v0/models/{model}")] = (200, details)
    return _conn("lmstudio", f"http://127.0.0.1:{server.port}/v1", model)


def test_lmstudio_small_loaded_context(server):
    preset, conn = _lmstudio(server, loaded_context=4096, capabilities=["tool_use"])
    with pytest.raises(ContextTooSmall) as info:
        probes.probe(preset, conn, threading.Event(), needed=9000)
    assert info.value.user_message == (
        "LM Studio loaded this model with a 4096-token context. Load it again with a larger context length (about "
        "9000 or more) in LM Studio's model settings.")
    assert server.paths == ["/api/v1/models", "/api/v0/models/qwen/qwen3-8b"]


def test_lmstudio_without_tool_use_is_a_note_only(server):
    preset, conn = _lmstudio(server, loaded_context=32768, capabilities=[], v1=False)
    facts = probes.probe(preset, conn, threading.Event(), needed=9000)
    assert (facts.context_window, facts.loaded_context, facts.tools) == (32768, 32768, False)
    assert len(facts.notes) == 1 and "qwen/qwen3-8b" in facts.notes[0]


def test_openrouter_model_list_cached_for_a_day(server):
    server.routes[("GET", "/api/v1/models")] = (200, {"data": [
        {"id": "vendor/tools-model", "context_length": 131072, "supported_parameters": ["tools", "tool_choice"]},
        {"id": "vendor/plain-model", "context_length": 8192, "supported_parameters": ["temperature"]}]})
    base = f"http://127.0.0.1:{server.port}/api/v1"
    preset, conn = _conn("openrouter", base, "vendor/tools-model")
    facts = probes.probe(preset, conn, threading.Event(), clock=lambda: 1000.0)
    assert (facts.context_window, facts.tools) == (131072, True)
    _, plain = _conn("openrouter", base, "vendor/plain-model")
    with pytest.raises(ToolsUnsupported):
        probes.probe(preset, plain, threading.Event(), clock=lambda: 1000.0 + 3600)
    assert server.paths == ["/api/v1/models"]                 # the second probe read the cache
    probes.probe(preset, conn, threading.Event(), clock=lambda: 1000.0 + 25 * 3600)
    assert server.paths == ["/api/v1/models"] * 2             # a day later the list is fetched again


def test_anthropic_input_limit():
    class Limits:
        closed = False

        def context_window(self) -> int:
            return 200_000

    preset, conn = _conn("anthropic", "https://api.anthropic.com", "claude-test")
    assert probes.probe(preset, conn, threading.Event(), backend=Limits()).context_window == 200_000
    assert probes.probe(PRESETS["mistral"], conn, threading.Event()) == probes.ModelFacts()


@pytest.mark.parametrize("server_where,where", [
    ("http://127.0.0.1:11434/v1", "on this computer"),
    ("http://192.168.1.20:11434/v1", "at 192.168.1.20:11434"),
])
def test_the_forwarding_server_is_named_where_it_runs(server_where, where):
    """C4-n3: an Ollama elsewhere on the network is not "on this computer"."""
    ollama = PRESETS["ollama"]
    cloud = endpoint.route_for(ollama, server_where, model="kimi:cloud")
    assert endpoint.indicator_text(cloud) == f"Data goes to: Ollama cloud (ollama.com), forwarded by Ollama {where}."
    other = endpoint.route_for(ollama, server_where, facts=probes.ModelFacts(remote_host="gpu.example.org"))
    assert endpoint.indicator_text(other) == f"Data goes to: gpu.example.org, forwarded by Ollama {where}."


# --- the probe takes the route of the requests (C4-m5) -------------------------------------------------------------
PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")


@pytest.fixture
def proxy() -> Iterator[RawServer]:
    """A proxy that records the request line (the absolute address) and answers 502."""
    fake = RawServer(status=502, body=b'{"error": "from the proxy"}')
    yield fake
    fake.close()


def _proxy_variables(monkeypatch: pytest.MonkeyPatch, proxy_url: str | None, no_proxy: str | None = None) -> None:
    for name in PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    if proxy_url is not None:
        for name in ("HTTP_PROXY", "http_proxy"):
            monkeypatch.setenv(name, proxy_url)
    if no_proxy is not None:
        for name in ("NO_PROXY", "no_proxy"):
            monkeypatch.setenv(name, no_proxy)


def _remote_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(net, "endpoint_kind", lambda url: "remote")


def _probe_target(server: FakeServer, preset_id: str) -> tuple[object, Connection, str]:
    """A probe of *preset_id* at *server*, and the request line it starts with."""
    if preset_id == "ollama":
        preset, conn = _ollama(server)
        return preset, conn, f"POST http://127.0.0.1:{server.port}/api/show HTTP/1.1"
    preset, conn = _lmstudio(server, loaded_context=32768, capabilities=["tool_use"])
    return preset, conn, f"GET http://127.0.0.1:{server.port}/api/v1/models HTTP/1.1"


@pytest.mark.parametrize("preset_id", ["ollama", "lmstudio"])
def test_remote_probe_goes_through_the_proxy(preset_id, server, proxy, monkeypatch):
    _remote_loopback(monkeypatch)
    _proxy_variables(monkeypatch, proxy.url)
    # urllib's own bypass rules play no part: the route was decided by net.proxy_for.
    monkeypatch.setattr(urllib.request, "proxy_bypass", lambda host: True)
    preset, conn, first = _probe_target(server, preset_id)
    with pytest.raises(LLMError):
        probes.probe(preset, conn, threading.Event(), needed=9000)
    assert proxy.lines[0] == first
    assert server.paths == []                                   # nothing went around the proxy


@pytest.mark.parametrize("preset_id", ["ollama", "lmstudio"])
def test_remote_probe_uses_the_remote_timeout(preset_id, server, monkeypatch):
    _remote_loopback(monkeypatch)
    _proxy_variables(monkeypatch, None)
    timeouts: list[float] = []
    fetch = probes._Asker._fetch

    def spy(self, method, path, payload, timeout):
        timeouts.append(timeout)
        return fetch(self, method, path, payload, timeout)

    monkeypatch.setattr(probes._Asker, "_fetch", spy)
    preset, conn, _first = _probe_target(server, preset_id)
    probes.probe(preset, conn, threading.Event(), needed=9000)
    assert timeouts[0] == 10.0                                  # REMOTE_TIMEOUT; this computer gets 2 s
    if preset_id == "ollama":
        assert timeouts[1] == 900.0                             # loading a model keeps its own timeout


@pytest.fixture
def example_org(server: FakeServer, monkeypatch: pytest.MonkeyPatch) -> str:
    """``http://example.org:<server port>/v1``, a remote address whose name resolves to the loopback server."""
    real = socket.getaddrinfo

    def getaddrinfo(host, port, *args, **kwargs):
        return real("127.0.0.1" if host == "example.org" else host, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    server.routes[("GET", "/v1/models")] = (200, {"data": [{"id": "example-model", "context_length": 32768}]})
    server.routes[("POST", "/v1/chat/completions")] = (500, {"error": "the test only checks the route"})
    return f"http://example.org:{server.port}/v1"


def _probe_and_request(url: str) -> None:
    """Probe the custom endpoint at *url*, then send it a chat request through the real client."""
    preset = PRESETS["openai_compatible"]
    conn = connection("openai_compatible", url=url)
    probes.probe(preset, conn, threading.Event())               # a list it cannot read gives no facts
    backend = make_backend(preset, conn)
    try:
        with pytest.raises(LLMError):
            backend.stream(request([Message("user", "hello")]), lambda e: None, threading.Event())
    finally:
        backend.close()


def test_no_proxy_rules_give_the_probe_and_the_request_one_route(example_org, server, proxy, monkeypatch):
    port = server.port
    # NO_PROXY=.example.org exempts the subdomains, not example.org itself (curl's rule, as httpx2 reads it;
    # urllib's own bypass would exempt it): both go through the proxy.
    _proxy_variables(monkeypatch, proxy.url, ".example.org")
    _probe_and_request(example_org)
    assert proxy.lines == [f"GET http://example.org:{port}/v1/models HTTP/1.1",
                           f"POST http://example.org:{port}/v1/chat/completions HTTP/1.1"]
    assert server.paths == []
    # NO_PROXY=example.org exempts it: both go directly.
    _proxy_variables(monkeypatch, proxy.url, "example.org")
    _probe_and_request(example_org)
    assert server.paths == ["/v1/models", "/v1/chat/completions"]
    assert len(proxy.lines) == 2


def test_system_proxy_exceptions_give_the_probe_and_the_request_one_route(example_org, server, proxy,
                                                                          monkeypatch):
    """No proxy variable: the system settings name the proxy, and their exception list (Windows ProxyOverride,
    macOS exceptions, read by urllib.request.proxy_bypass) exempts the host for both."""
    import httpx2._utils

    _proxy_variables(monkeypatch, None)
    monkeypatch.setattr(httpx2._utils, "getproxies", lambda: {"http": proxy.url})
    monkeypatch.setattr(urllib.request, "proxy_bypass", lambda host: host == f"example.org:{server.port}")
    assert net.proxy_for(example_org) is None
    _probe_and_request(example_org)
    assert server.paths == ["/v1/models", "/v1/chat/completions"]
    assert proxy.lines == []
    # A host the exceptions do not list goes through the system proxy.
    monkeypatch.setattr(urllib.request, "proxy_bypass", lambda host: False)
    assert net.proxy_for(example_org) == proxy.url
