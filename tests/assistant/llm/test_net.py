"""FlexTool's own HTTP client (P1 §3.0): the environment variables the provider libraries read add nothing to a
request, proxies apply to remote endpoints only, redirects are not followed, odd base URLs are refused."""
from __future__ import annotations

import logging
import threading

import httpx2
import pytest

from flextool.assistant import llm_errors as E
from flextool.assistant import net
from flextool.assistant.llm_adapter import Message, make_backend
from flextool.assistant.providers import PRESETS
from tests.assistant.llm.wire import (
    KEY,
    Recorder,
    connection,
    json_response,
    openai_sse,
    payload,
    request,
    sse_response,
)

SDK_VARIABLES = {
    "OPENAI_CUSTOM_HEADERS": "Authorization: Bearer from-the-environment\nX-Custom-Header: leaked",
    "OPENAI_ORG_ID": "org-from-the-environment",
    "OPENAI_PROJECT_ID": "proj-from-the-environment",
    "OPENAI_API_KEY": "sk-from-the-environment",
    "OPENAI_BASE_URL": "https://elsewhere.example.test/v1",
    "ANTHROPIC_CUSTOM_HEADERS": "Authorization: Bearer from-the-environment\nX-Custom-Header: leaked",
    "ANTHROPIC_BASE_URL": "https://elsewhere.example.test",
    "ANTHROPIC_API_KEY": "sk-ant-from-the-environment",
    "ANTHROPIC_AUTH_TOKEN": "token-from-the-environment",
}
PROXY = "http://proxy.example.test:3128"


@pytest.fixture
def sdk_environment(monkeypatch):
    for name, value in SDK_VARIABLES.items():
        monkeypatch.setenv(name, value)


@pytest.fixture
def proxy_environment(monkeypatch):
    for name in ("NO_PROXY", "no_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.setenv(name, PROXY)


def _answer() -> httpx2.Response:
    return sse_response(openai_sse(payload("openai", "text_usage")["chunks"]))


def _send(preset_id: str, key: str | None = KEY):
    recorder = Recorder([_answer()])
    backend = make_backend(PRESETS[preset_id], connection(preset_id, key=key), transport=recorder.transport)
    backend.stream(request([Message("user", "hello")]), lambda e: None, threading.Event())
    return recorder.last


@pytest.mark.parametrize("preset_id", sorted(p.id for p in PRESETS.values() if p.backend == "openai"))
def test_only_allow_listed_headers_and_our_key_and_address(preset_id, sdk_environment):
    sent = _send(preset_id)
    assert set(sent.headers) <= net.ALLOWED_HEADERS | net.BACKEND_HEADERS["openai"]
    assert sent.headers["authorization"] == f"Bearer {KEY}"
    assert sent.headers["user-agent"] == "FlexTool-assistant"
    assert not any(h.startswith("x-") for h in sent.headers)
    base = connection(preset_id).base_url.rstrip("/")
    assert sent.url == f"{base}/chat/completions"
    assert "leaked" not in str(sent.headers.values())


def test_an_endpoint_without_a_key_never_gets_the_environment_key(sdk_environment):
    sent = _send("ollama", key=None)
    assert sent.headers["authorization"] == "Bearer not-needed"


def test_the_key_header_must_carry_the_resolved_key():
    hook = net.header_hook("the-key", net.ALLOWED_HEADERS)
    good = httpx2.Request("POST", "https://api.example.test/v1", headers={"Authorization": "Bearer the-key",
                                                                         "X-Stainless-OS": "Linux"})
    hook(good)
    assert "x-stainless-os" not in good.headers and good.headers["user-agent"] == "FlexTool-assistant"
    bad = httpx2.Request("POST", "https://api.example.test/v1", headers={"Authorization": "Bearer other"})
    with pytest.raises(RuntimeError) as info:
        hook(bad)
    assert "the-key" not in str(info.value)
    with pytest.raises(ValueError):
        net.build_http_client(connection("ollama"), backend="openai", api_key="")


@pytest.mark.parametrize("url,kind", [
    ("http://127.0.0.1:11434/v1", "loopback"),
    ("http://localhost:1234/v1", "loopback"),
    ("http://LOCALHOST:1234/v1", "loopback"),
    ("http://[::1]:8000/v1", "loopback"),
    ("http://[::ffff:127.0.0.1]:8000/v1", "loopback"),
    ("http://0.0.0.0:8000/v1", "loopback"),
    ("http://169.254.10.20:8000/v1", "local_network"),
    ("http://192.168.1.5:8000/v1", "local_network"),
    ("http://10.0.0.7/v1", "local_network"),
    ("http://172.20.1.1/v1", "local_network"),
    ("http://[fd00::12]:8000/v1", "local_network"),
    ("http://[fe80::1]:8000/v1", "local_network"),
    ("http://mybox.lan:8000/v1", "remote"),
    ("http://localhost.example.test/v1", "remote"),
    ("https://8.8.8.8/v1", "remote"),
    ("https://api.example.test/v1", "remote"),
])
def test_endpoint_kinds_use_the_literal_host(url, kind):
    assert net.endpoint_kind(url) == kind


def test_a_host_name_that_resolves_locally_is_still_remote(monkeypatch):
    import socket

    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(socket.AF_INET, 1, 6, "", ("127.0.0.1", 80))])
    assert net.endpoint_kind("http://my-workstation:8000/v1") == "remote"


def test_remote_endpoints_use_the_proxy_httpx_would_use(proxy_environment):
    url = "https://api.example.test/v1"
    assert net.proxy_for(url) == PROXY
    session = net.build_http_client(connection("openai_compatible", url=url), backend="openai", api_key=KEY)
    try:
        ours = session.client._transport_for_url(httpx2.URL(url))._pool
        reference = httpx2.Client(trust_env=True)._transport_for_url(httpx2.URL(url))._pool
        assert type(ours).__name__ == "HTTPProxy"
        assert ours._proxy_url == reference._proxy_url
        assert session.proxy == PROXY
    finally:
        session.close()
    route = net.route_for(url, "Custom endpoint")
    assert (route.kind, route.host, route.proxy_host) == ("remote", "api.example.test", "proxy.example.test")


@pytest.mark.parametrize("url", ["http://localhost:11434/v1", "http://127.0.0.1:1234/v1",
                                 "http://192.168.1.5:8000/v1"])
def test_local_endpoints_never_use_a_proxy(url, proxy_environment):
    assert net.proxy_for(url) is None
    session = net.build_http_client(connection("ollama", url=url), backend="openai", api_key=net.NO_KEY)
    try:
        pool = session.client._transport_for_url(httpx2.URL(url))._pool
        assert type(pool).__name__ == "ConnectionPool"
        assert session.client.trust_env is False
    finally:
        session.close()
    assert net.route_for(url, "Ollama").proxy_host is None


def test_no_proxy_rules_are_honoured(proxy_environment, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "api.example.test")
    monkeypatch.setenv("no_proxy", "api.example.test")
    assert net.proxy_for("https://api.example.test/v1") is None
    assert net.proxy_for("https://other.example.test/v1") == PROXY


def test_a_forwarded_route_is_remote():
    route = net.route_for("http://localhost:11434/v1", "Ollama cloud", forwarded_to="ollama.example.test")
    assert (route.kind, route.host, route.forwarded_to, route.proxy_host) == (
        "remote", "localhost:11434", "ollama.example.test", None)


def test_redirects_are_not_followed():
    recorder = Recorder([httpx2.Response(307, headers={"location": "https://elsewhere.example.test/v1/chat"}),
                         _answer()])
    backend = make_backend(PRESETS["openai_compatible"], connection("openai_compatible"),
                           transport=recorder.transport)
    with pytest.raises(E.ProviderError) as info:
        backend.stream(request([Message("user", "hello")]), lambda e: None, threading.Event())
    assert len(recorder.requests) == 1
    assert info.value.user_message == ("The endpoint redirected to elsewhere.example.test. FlexTool does not follow "
                                       "redirects, so nothing was sent there.")


@pytest.mark.parametrize("url,message", [
    ("https://user:secret@llm.example.test/v1", net.BASE_URL_CREDENTIALS),
    ("https://token@llm.example.test/v1", net.BASE_URL_CREDENTIALS),
    ("https://llm.example.test/v1?key=abc", net.BASE_URL_CREDENTIALS),
    ("ftp://llm.example.test/v1", net.BASE_URL_SCHEME),
    ("llm.example.test/v1", net.BASE_URL_SCHEME),
])
def test_odd_base_urls_are_refused(url, message):
    with pytest.raises(net.BaseUrlRefused) as info:
        net.check_base_url(url)
    assert str(info.value) == message
    assert net.check_base_url(" http://localhost:8000/v1 ") == "http://localhost:8000/v1"


def test_library_loggers_are_capped():
    for name in net_loggers():
        logging.getLogger(name).setLevel(logging.DEBUG)
    recorder = Recorder([json_response(200, {"data": []})])
    make_backend(PRESETS["openai"], connection("openai"), transport=recorder.transport)
    assert all(logging.getLogger(n).level == logging.WARNING for n in net_loggers())
    make_backend(PRESETS["openai"], connection("openai"), transport=recorder.transport, verbose=True)
    assert all(logging.getLogger(n).level == logging.INFO for n in net_loggers())


def net_loggers():
    from flextool.assistant.llm_adapter import SDK_LOGGERS

    return SDK_LOGGERS
