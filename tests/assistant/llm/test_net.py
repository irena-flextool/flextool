"""FlexTool's own HTTP client (P1 §3.0): the environment variables the provider libraries read add nothing to a
request and change none of its headers, proxies apply to remote endpoints only, redirects are not followed, odd base
URLs are refused, and a remote plain ``http://`` address is warned about (or refused, :data:`net.REMOTE_HTTP`)."""
from __future__ import annotations

import json
import logging
import threading

import httpx2
import pytest

from flextool.assistant import endpoint, net
from flextool.assistant import llm_errors as E
from flextool.assistant.llm_adapter import Message, make_backend
from flextool.assistant.providers import PRESETS
from tests.assistant.llm.wire import (
    KEY,
    Recorder,
    anthropic_sse,
    connection,
    json_response,
    openai_sse,
    payload,
    request,
    sse_response,
)
from tests.assistant.llm.local_server import RawServer

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


def _answer(backend: str = "openai") -> httpx2.Response:
    if backend == "anthropic":
        return sse_response(anthropic_sse(payload("anthropic", "text_end")["events"]))
    return sse_response(openai_sse(payload("openai", "text_usage")["chunks"]))


def _send(preset_id: str, key: str | None = KEY):
    recorder = Recorder([_answer(PRESETS[preset_id].backend)])
    backend = make_backend(PRESETS[preset_id], connection(preset_id, key=key), transport=recorder.transport)
    backend.stream(request([Message("user", "hello")]), lambda e: None, threading.Event())
    return recorder.last


@pytest.mark.parametrize("preset_id", sorted(PRESETS))
def test_only_allow_listed_headers_and_our_key_and_address(preset_id, sdk_environment):
    backend = PRESETS[preset_id].backend
    sent = _send(preset_id)
    assert set(sent.headers) <= (net.ALLOWED_HEADERS | set(net.FIXED_HEADERS) | net.COMPUTED_HEADERS
                                 | net.BACKEND_HEADERS[backend])
    assert sent.headers["authorization"] == f"Bearer {KEY}"
    assert sent.headers["user-agent"] == "FlexTool-assistant"
    assert not any(h.startswith("x-") for h in sent.headers)
    base = connection(preset_id).base_url.rstrip("/")
    path = "/v1/messages" if backend == "anthropic" else "/chat/completions"
    assert sent.url == f"{base}{path}"
    if backend == "anthropic":
        assert sent.headers["anthropic-version"]
    assert "leaked" not in str(sent.headers.values())


#: Custom-header lines that try to change every header FlexTool sets itself (C4-m2): the address, the framing,
#: the types, the connection and Anthropic's API version.  The second set has a chunked transfer encoding and no
#: length, which made ``httpx2`` leave the length out while the hook dropped the encoding (no framing at all).
OVERRIDES = {
    "all": "\n".join(["Host: evil.example", "Accept-Encoding: identity-evil", "Content-Length: 1",
                      "Connection: close", "Accept: text/evil", "Content-Type: text/evil",
                      "anthropic-version: 1999-01-01", "Transfer-Encoding: chunked"]),
    "chunked": "Transfer-Encoding: chunked",
}


@pytest.mark.parametrize("variant", sorted(OVERRIDES))
@pytest.mark.parametrize("preset_id", sorted(PRESETS))
def test_custom_header_variables_cannot_set_computed_headers(preset_id, variant, monkeypatch):
    """The real library, the real client and a server on this computer: what arrives is what FlexTool chose."""
    backend = PRESETS[preset_id].backend
    for name in ("OPENAI_CUSTOM_HEADERS", "ANTHROPIC_CUSTOM_HEADERS"):
        monkeypatch.setenv(name, OVERRIDES[variant])
    answer = (anthropic_sse(payload("anthropic", "text_end")["events"]) if backend == "anthropic"
              else openai_sse(payload("openai", "text_usage")["chunks"]))
    server = RawServer(body=answer, content_type="text/event-stream")
    try:
        url = server.url if backend == "anthropic" else server.url + "/v1"
        made = make_backend(PRESETS[preset_id], connection(preset_id, url=url))
        try:
            made.stream(request([Message("user", "hello")]), lambda e: None, threading.Event())
        finally:
            made.close()
    finally:
        server.close()
    (line, headers, body), = server.requests
    names = [name for name, _value in headers]
    assert len(names) == len(set(names))                    # no header twice
    sent = dict(headers)
    assert sent["host"] == f"127.0.0.1:{server.port}"
    assert sent["accept-encoding"] == "gzip, deflate"
    assert sent["connection"] == "keep-alive"
    assert sent["accept"] == "application/json"
    assert sent["content-type"] == "application/json"
    assert "transfer-encoding" not in sent
    assert sent["content-length"] == str(len(body))
    assert json.loads(body)["model"] == "example-model"     # the whole body arrived, framed by its length
    if backend == "anthropic":
        from anthropic._constants import ANTHROPIC_API_VERSION

        assert sent["anthropic-version"] == ANTHROPIC_API_VERSION
    else:
        assert "anthropic-version" not in sent


def test_a_streamed_body_is_sent_chunked():
    hook = net.header_hook("the-key", net.ALLOWED_HEADERS)
    streamed = httpx2.Request("POST", "https://api.example.test:8443/v1", content=iter([b"{}"]),
                              headers={"Authorization": "Bearer the-key", "Content-Length": "1"})
    hook(streamed)
    assert (streamed.headers["host"], streamed.headers["transfer-encoding"]) == ("api.example.test:8443", "chunked")
    assert "content-length" not in streamed.headers
    plain = httpx2.Request("GET", "https://api.example.test/v1/models",
                           headers={"Authorization": "Bearer the-key", "Content-Type": "text/evil"})
    hook(plain)
    assert plain.headers["host"] == "api.example.test"      # the default port is not written
    assert "content-type" not in plain.headers and "content-length" not in plain.headers


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


# --- a remote plain http:// address (C4-m6, owner decision D2 pending: "warn" is the default) -------------------
#: The notice hash before the plain-http sentence existed: the acknowledgements users already hold carry it.
NOTICE_HASH_BEFORE = "7dedb880c96c"
PLAIN = "http://gpu.example.org:11434/v1"


@pytest.fixture
def no_proxy_variables(monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)


def test_remote_http_is_warned_about_by_default():
    assert net.REMOTE_HTTP == "warn"


@pytest.mark.parametrize("url,plain", [
    (PLAIN, True),
    ("HTTP://GPU.Example.org/v1", True),
    ("https://gpu.example.org:11434/v1", False),
    ("http://192.168.1.5:11434/v1", False),
    ("http://localhost:11434/v1", False),
])
def test_warn_mode_says_the_connection_is_not_encrypted(url, plain, monkeypatch, no_proxy_variables):
    monkeypatch.setattr(net, "REMOTE_HTTP", "warn")
    assert net.plain_http_remote(url) is plain
    assert net.check_base_url(url) == url                   # used, with the warning
    route = net.route_for(url, "Custom endpoint")
    assert route.encrypted is not plain
    assert endpoint.indicator_text(route).endswith(endpoint.PLAIN_HTTP) is plain
    assert endpoint.consent_notice(route).endswith(endpoint.PLAIN_HTTP) is plain


def test_the_warning_texts(monkeypatch, no_proxy_variables):
    monkeypatch.setattr(net, "REMOTE_HTTP", "warn")
    route = net.route_for(PLAIN, "Custom endpoint")
    assert endpoint.indicator_text(route) == (
        "Data goes to: Custom endpoint (gpu.example.org:11434). The connection is not encrypted: the key and the "
        "data can be read on the way.")
    assert endpoint.consent_notice(route).endswith(
        "or LM Studio), which requires considerable local compute. The connection is not encrypted: the key and the "
        "data can be read on the way.")
    # A local server forwarding a cloud model is encrypted on the part FlexTool chooses: no warning.
    forwarded = net.route_for("http://localhost:11434/v1", "Ollama cloud", forwarded_to="ollama.com")
    assert forwarded.encrypted and not endpoint.indicator_text(forwarded).endswith(endpoint.PLAIN_HTTP)


def test_only_plain_http_routes_are_asked_again(monkeypatch, no_proxy_variables, user_config_in_tmp):
    import dataclasses

    monkeypatch.setattr(net, "REMOTE_HTTP", "warn")
    secure_url = "https://gpu.example.org:11434/v1"
    secure = endpoint.consent_key("openai_compatible", secure_url, net.route_for(secure_url, "Custom endpoint"))
    plain = endpoint.consent_key("openai_compatible", PLAIN, net.route_for(PLAIN, "Custom endpoint"))
    assert endpoint.NOTICE_HASH == NOTICE_HASH_BEFORE        # the https notice did not change
    # Acceptances made before the sentence existed: https keeps its acceptance, plain http is asked once more.
    endpoint.record_consent(dataclasses.replace(secure, notice=NOTICE_HASH_BEFORE), user_config_in_tmp)
    endpoint.record_consent(dataclasses.replace(plain, notice=NOTICE_HASH_BEFORE), user_config_in_tmp)
    assert endpoint.needs_consent(secure, user_config_in_tmp) is False
    assert endpoint.needs_consent(plain, user_config_in_tmp) is True
    endpoint.record_consent(plain, user_config_in_tmp)
    assert endpoint.needs_consent(plain, user_config_in_tmp) is False


def test_switching_scheme_asks_again(no_proxy_variables, user_config_in_tmp):
    """A pin (it held before C4-m6): http and https are two endpoints of the acknowledgement."""
    https_url = "https://gpu.example.org:11434/v1"
    https = endpoint.consent_key("openai_compatible", https_url, net.route_for(https_url, "Custom endpoint"))
    http = endpoint.consent_key("openai_compatible", PLAIN, net.route_for(PLAIN, "Custom endpoint"))
    endpoint.record_consent(https, user_config_in_tmp)
    assert endpoint.needs_consent(http, user_config_in_tmp) is True
    endpoint.record_consent(http, user_config_in_tmp)
    (user_config_in_tmp / endpoint.ACKS_FILE).unlink()
    endpoint.record_consent(http, user_config_in_tmp)
    assert endpoint.needs_consent(https, user_config_in_tmp) is True


def test_refuse_mode_refuses_remote_http_only(monkeypatch, no_proxy_variables):
    monkeypatch.setattr(net, "REMOTE_HTTP", "refuse")
    with pytest.raises(net.BaseUrlRefused) as info:
        net.check_base_url(PLAIN)
    assert str(info.value) == ("FlexTool sends a key and case data to an address outside this computer and your "
                               "local network only over https://. Use https://gpu.example.org.")
    for url in ("http://localhost:11434/v1", "http://192.168.1.5:11434/v1", "https://gpu.example.org:11434/v1"):
        assert net.check_base_url(url) == url
    with pytest.raises(net.BaseUrlRefused):
        net.build_http_client(connection("openai_compatible", url=PLAIN), backend="openai", api_key=KEY)


# --- the sockets a Stop shuts (C4-M1, C4-m7) ----------------------------------------------------------------------
class _FakeStream:
    def __init__(self) -> None:
        self.shut = 0

    def shutdown(self) -> None:
        self.shut += 1


def test_a_stream_added_after_a_stop_is_shut_at_once():
    """A connection whose connect was under way when Stop came is shut as it opens; after the next request
    begins, a new connection is left alone."""
    registry = net._SocketRegistry()
    registry.shutdown_all()                         # a Stop, with no connection open yet
    late = _FakeStream()
    registry.add(late)
    assert late.shut == 1
    registry.begin()
    fresh = _FakeStream()
    registry.add(fresh)
    assert fresh.shut == 0


def test_close_shuts_the_socket_down_first_and_closes_once():
    """Closing a tracked stream shuts its socket down before closing it, so a close from another thread wakes a
    blocked read: the peer sees the end although another descriptor still holds the socket open.  A second close
    and a later Stop do nothing, and nothing raises."""
    import socket

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    registry = net._SocketRegistry()
    stream = net._tracked_backend(registry).connect_tcp("127.0.0.1", server.getsockname()[1], timeout=5)
    peer, _ = server.accept()
    held = stream.get_extra_info("socket").dup()    # keeps the connection open past the close
    try:
        assert len(registry._streams) == 1
        stream.close()
        peer.settimeout(2)
        assert peer.recv(10) == b""                 # the shutdown's end of stream, not a timeout
        stream.close()
        registry.shutdown_all()
        assert len(registry._streams) == 0
        assert stream.get_extra_info("socket").fileno() == -1
    finally:
        held.close()
        peer.close()
        server.close()
