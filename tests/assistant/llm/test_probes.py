"""The model probes and the route they give (P1 §15 test 9).

A ``http.server`` on ``127.0.0.1`` plays Ollama (``/api/show``, ``/api/generate``, ``/api/ps``), LM Studio (v1 and
v0 model endpoints) and OpenRouter's model list.  Checked: a small loaded context stops the chat before any model
request (the OLLAMA_CONTEXT_LENGTH message); no ``tools`` capability stops it; ``remote_host`` and ``:cloud`` /
``-cloud`` names give the Ollama cloud route and skip the load; an unreadable ``/api/show`` sends nothing more;
nothing but the model and ``keep_alive`` is sent to load a model; the probes ignore the proxy variables.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import pytest

from flextool.assistant import endpoint, probes
from flextool.assistant.llm_adapter import Connection, ToolSpec, Waiting
from flextool.assistant.llm_errors import (
    Cancelled,
    ContextTooSmall,
    ProviderError,
    ServerBusy,
    ToolsUnsupported,
    Unreachable,
)
from flextool.assistant.providers import PRESETS
from tests.assistant.llm.local_server import Answer, FakeServer

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
    # '{"system": ["' (13) + 400 + '"], "tools": []}' (16) = 429 characters; 429 // 4 + 4000.
    assert probes.needed_tokens(("x" * 400,), ()) == 4107
    tool = ToolSpec("t", "d", {"type": "object"})
    assert probes.needed_tokens((), (tool,)) > probes.needed_tokens((), ())


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
