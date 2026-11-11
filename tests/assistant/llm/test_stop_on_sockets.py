"""Stop against real loopback sockets (P1 §3.0 Cancel, C4-M1/m7): an answer that went quiet mid-stream, a TLS
handshake that hangs (direct and through a CONNECT proxy), a request after a Stop, and no socket or descriptor
left behind by a handshake, whichever way it ends.

Each test waits until the worker thread is blocked where the test means it to be (its innermost Python frame in
the socket read or the TLS handshake) before it presses Stop, so the result does not depend on timing."""
from __future__ import annotations

import contextlib
import ipaddress
import json
import os
import socket
import ssl
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from flextool.assistant import llm_errors as E
from flextool.assistant.llm_adapter import Message, TextDelta, make_backend
from flextool.assistant.providers import PRESETS
from tests.assistant.llm.wire import KEY, anthropic_sse, connection, openai_sse, payload, request

STOP_WITHIN = 1.5           # seconds; before the fixes the worker stays blocked until the server lets go (10 s)
HOLD = 10.0                 # how long a server keeps a connection silent


def ask() -> list[Message]:
    return [Message("user", "How many units has west?")]


# --- helpers ------------------------------------------------------------------------------------------------------
def _run(call: Callable[[], object]) -> tuple[threading.Thread, list]:
    outcome: list = []

    def worker() -> None:
        try:
            outcome.append(call())
        except Exception as exc:  # noqa: BLE001 — the outcome is the test's subject
            outcome.append(exc)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread, outcome


def _blocked_in(thread: threading.Thread, function: str, file_part: str, timeout: float = 5.0) -> bool:
    """Wait until *thread*'s innermost Python frame is *function* in a file whose path contains *file_part*."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame = sys._current_frames().get(thread.ident)
        if frame is not None and frame.f_code.co_name == function and file_part in frame.f_code.co_filename:
            return True
        time.sleep(0.005)
    return False


def _open_fds() -> int:
    return len(os.listdir("/proc/self/fd"))


def _settled_fds(expected: int, timeout: float = 3.0) -> int:
    """The open descriptors once the servers' threads have closed their ends (they do so asynchronously)."""
    deadline = time.monotonic() + timeout
    count = _open_fds()
    while count != expected and time.monotonic() < deadline:
        time.sleep(0.02)
        count = _open_fds()
    return count


needs_proc_fd = pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="counts descriptors in /proc")


def _read_request(conn: socket.socket) -> bytes:
    data = b""
    while b"\r\n\r\n" not in data:
        part = conn.recv(65536)
        if not part:
            return data
        data += part
    head, _, body = data.partition(b"\r\n\r\n")
    length = 0
    for line in head.split(b"\r\n")[1:]:
        name, _, value = line.partition(b":")
        if name.strip().lower() == b"content-length":
            length = int(value.strip())
    while len(body) < length:
        part = conn.recv(65536)
        if not part:
            break
        body += part
    return head


class _Server:
    """A loopback server for one connection: *handle(conn)* runs on it; :meth:`close` lets a silent one go."""

    def __init__(self, handle: Callable[[socket.socket, _Server], None]) -> None:
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen()
        self.port = self.sock.getsockname()[1]
        self.accepted = threading.Event()
        self.ready = threading.Event()          # the server has sent what it sends before it goes silent
        self.stop = threading.Event()
        self.lines: list[bytes] = []

        def serve() -> None:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            self.accepted.set()
            try:
                handle(conn, self)
            except OSError:
                pass
            finally:
                conn.close()

        self.thread = threading.Thread(target=serve, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.stop.set()
        # A shutdown wakes an accept that is still waiting (a close from this thread would not).
        with contextlib.suppress(OSError):
            self.sock.shutdown(socket.SHUT_RDWR)
        self.sock.close()
        self.thread.join(5)


def _silent(conn: socket.socket, server: _Server) -> None:
    server.stop.wait(HOLD)


def _chunk(data: bytes) -> bytes:
    return f"{len(data):x}\r\n".encode() + data + b"\r\n"


def _one_chunk_then_quiet(preset: str) -> Callable[[socket.socket, _Server], None]:
    """Headers and the first part of an answer ("pa"), then silence."""
    if preset == "anthropic":
        events = payload("anthropic", "text_end")["events"][:2]
        events.append({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "pa"}})
        first = anthropic_sse(events)
    else:
        first = openai_sse([{"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
                             "choices": [{"index": 0, "delta": {"role": "assistant", "content": "pa"},
                                          "finish_reason": None}]}], done=False)

    def handle(conn: socket.socket, server: _Server) -> None:
        _read_request(conn)
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
                     + _chunk(first))
        server.ready.set()
        server.stop.wait(HOLD)

    return handle


def _connect_proxy(answer: bool) -> Callable[[socket.socket, _Server], None]:
    """A CONNECT proxy stub: it reads the CONNECT request and, when *answer*, says the tunnel is up; either way
    nothing comes through after that (the TLS handshake through the tunnel hangs)."""

    def handle(conn: socket.socket, server: _Server) -> None:
        server.lines.append(_read_request(conn).split(b"\r\n")[0])
        if answer:
            conn.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
        server.ready.set()
        server.stop.wait(HOLD)

    return handle


@pytest.fixture
def no_proxy_variables(monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)


@pytest.fixture(scope="module")
def self_signed(tmp_path_factory) -> tuple[Path, Path]:
    """A certificate for 127.0.0.1, signed by itself, and its key."""
    pytest.importorskip("cryptography")
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=1))
            .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
                           critical=False)
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))
    folder = tmp_path_factory.mktemp("tls")
    cert_file, key_file = folder / "cert.pem", folder / "key.pem"
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
    return cert_file, key_file


def _tls_answer(cert_file: Path, key_file: Path) -> Callable[[socket.socket, _Server], None]:
    """A TLS server with the self-signed certificate that answers with a whole text answer."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_file, key_file)
    body = openai_sse(payload("openai", "text_usage")["chunks"])

    def handle(conn: socket.socket, server: _Server) -> None:
        with context.wrap_socket(conn, server_side=True) as tls:
            _read_request(tls)
            tls.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: "
                        + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body)
            server.ready.set()

    return handle


# --- C4-M1: an answer that went quiet -----------------------------------------------------------------------------
@pytest.mark.parametrize("how", ["abort", "response_close"])
@pytest.mark.parametrize("preset", ["ollama", "anthropic"])
def test_stop_wakes_a_stream_that_went_quiet_mid_answer(preset, how, no_proxy_variables):
    """The model sent "pa" and went quiet (a thinking pause, a slow local model).  Stop ends the request at once
    and keeps the partial text.  ``response_close`` closes only the response, as the backend's response close
    does: every close shuts the socket down first, so that alone wakes the reader."""
    server = _Server(_one_chunk_then_quiet(preset))
    url = f"http://127.0.0.1:{server.port}" + ("/v1" if preset == "ollama" else "")
    backend = make_backend(PRESETS[preset], connection(preset, url=url, key=None if preset == "ollama" else KEY))
    first = threading.Event()
    thread, outcome = _run(lambda: backend.stream(
        request(ask()), lambda e: first.set() if isinstance(e, TextDelta) else None, threading.Event()))
    try:
        assert server.ready.wait(5) and first.wait(5)
        assert _blocked_in(thread, "read", "httpcore2"), "the worker should wait for the next part"
        stopped = time.monotonic()
        if how == "abort":
            backend.abort()
        else:
            backend._inflight.abort()
        thread.join(5)
        assert not thread.is_alive() and time.monotonic() - stopped < STOP_WITHIN
        err = outcome[0]
        assert isinstance(err, E.Cancelled), err
        assert err.partial_text == "pa" and err.streamed is True
    finally:
        server.close()
        thread.join(5)
        backend.close()


@pytest.mark.parametrize("stage", ["first_byte", "mid_answer"])
@pytest.mark.parametrize("preset", ["ollama", "anthropic"])
def test_stop_wakes_a_read_where_a_shutdown_does_not(preset, stage, monkeypatch, no_proxy_variables):
    """Where shutting a socket down does not wake a read blocked on it (macOS while an answer is quiet, Windows
    before the first byte: CI of PR #362), the read still ends within a slice of ``net.READ_SLICE``.  Here no
    shutdown reaches the socket at all, so only the read's own check can end it."""
    monkeypatch.setattr(socket.socket, "shutdown", lambda self, how: None)
    server = _Server(_silent if stage == "first_byte" else _one_chunk_then_quiet(preset))
    url = f"http://127.0.0.1:{server.port}" + ("/v1" if preset == "ollama" else "")
    backend = make_backend(PRESETS[preset], connection(preset, url=url, key=None if preset == "ollama" else KEY))
    first = threading.Event()
    thread, outcome = _run(lambda: backend.stream(
        request(ask()), lambda e: first.set() if isinstance(e, TextDelta) else None, threading.Event()))
    try:
        assert server.accepted.wait(5)
        if stage == "mid_answer":
            assert server.ready.wait(5) and first.wait(5)
        assert _blocked_in(thread, "read", "httpcore2"), "the worker should wait for the server"
        stopped = time.monotonic()
        backend.abort()
        thread.join(5)
        assert not thread.is_alive() and time.monotonic() - stopped < STOP_WITHIN
        err = outcome[0]
        assert isinstance(err, E.Cancelled), err
        assert err.partial_text == ("pa" if stage == "mid_answer" else "")
    finally:
        server.close()
        thread.join(5)
        backend.close()


# --- C4-m7: a Stop during the TLS handshake -----------------------------------------------------------------------
@needs_proc_fd
@pytest.mark.parametrize("route", ["direct", "proxy_tunnel", "proxy_connect"])
def test_stop_during_the_tls_handshake(route, monkeypatch, no_proxy_variables):
    """Stop while the TLS handshake waits for a server that never answers it — directly, or through a CONNECT
    proxy whose tunnel is up (``proxy_connect``: the proxy never answers the CONNECT itself).  The request ends at
    once as a Stop, and no socket stays registered or open."""
    baseline = _open_fds()
    if route == "direct":
        server = _Server(_silent)
        url = f"https://127.0.0.1:{server.port}/v1"
        preset, key = "ollama", None
    else:
        server = _Server(_connect_proxy(answer=route == "proxy_tunnel"))
        for name in ("HTTPS_PROXY", "https_proxy"):
            monkeypatch.setenv(name, f"http://127.0.0.1:{server.port}")
        url, preset, key = "https://llm.example.test/v1", "openai", KEY
    backend = make_backend(PRESETS[preset], connection(preset, url=url, key=key))
    thread, outcome = _run(lambda: backend.stream(request(ask()), lambda e: None, threading.Event()))
    try:
        assert server.accepted.wait(5)
        if route == "proxy_connect":
            assert server.ready.wait(5) and _blocked_in(thread, "read", "httpcore2")
        else:
            assert _blocked_in(thread, "do_handshake", "ssl"), "the worker should be in the TLS handshake"
        stopped = time.monotonic()
        backend.abort()
        thread.join(5)
        assert not thread.is_alive() and time.monotonic() - stopped < STOP_WITHIN
        assert isinstance(outcome[0], E.Cancelled), outcome[0]
        assert len(backend._http._registry._streams) == 0
        if route != "direct":
            assert server.lines == [b"CONNECT llm.example.test:443 HTTP/1.1"]
    finally:
        server.close()
        thread.join(5)
        backend.close()
    assert _settled_fds(baseline) == baseline


@needs_proc_fd
def test_a_failed_tls_handshake_leaves_no_socket_behind(self_signed, no_proxy_variables):
    """A server whose certificate is not trusted: the request fails (not as a Stop), and the handshake leaves no
    registered socket and no open descriptor (the duplicate taken for a Stop is closed)."""
    baseline = _open_fds()
    server = _Server(_tls_answer(*self_signed))
    backend = make_backend(PRESETS["ollama"], connection("ollama", url=f"https://127.0.0.1:{server.port}/v1",
                                                         key=None))
    try:
        with pytest.raises(E.LLMError) as info:
            backend.stream(request(ask()), lambda e: None, threading.Event())
        assert not isinstance(info.value, E.Cancelled)
        assert len(backend._http._registry._streams) == 0
    finally:
        server.close()
        backend.close()
    del info
    assert _settled_fds(baseline) == baseline


@needs_proc_fd
def test_a_tls_answer_still_arrives_whole(self_signed, monkeypatch, no_proxy_variables):
    """The handshake guard ends with the handshake and never shuts the connection: a trusted TLS server's answer
    arrives whole, and only the answer's own connection stays registered until the client closes."""
    import httpcore2

    cert_file, _ = self_signed
    monkeypatch.setattr(httpcore2, "default_ssl_context", lambda **_: ssl.create_default_context(cafile=cert_file))
    baseline = _open_fds()
    server = _Server(_tls_answer(*self_signed))
    backend = make_backend(PRESETS["ollama"], connection("ollama", url=f"https://127.0.0.1:{server.port}/v1",
                                                         key=None))
    try:
        reply = backend.stream(request(ask()), lambda e: None, threading.Event())
        assert reply.stop == "end"
        assert reply.message.text == "".join(
            c["choices"][0]["delta"].get("content") or "" for c in payload("openai", "text_usage")["chunks"]
            if c.get("choices"))
    finally:
        server.close()
        backend.close()
    assert len(backend._http._registry._streams) == 0
    assert _settled_fds(baseline) == baseline


# --- the Stop flag ------------------------------------------------------------------------------------------------
def test_a_request_after_a_stop_is_not_shut(no_proxy_variables):
    """A Stop shuts the connections a request opens until the next request starts: the model list asked after a
    Stop on the same backend gets its answer."""
    models = json.dumps({"object": "list", "data": [{"id": "qwen3:8b", "object": "model"}]}).encode()

    def handle(conn: socket.socket, server: _Server) -> None:
        _read_request(conn)
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                     + str(len(models)).encode() + b"\r\nConnection: close\r\n\r\n" + models)

    server = _Server(handle)
    backend = make_backend(PRESETS["ollama"], connection("ollama", url=f"http://127.0.0.1:{server.port}/v1",
                                                         key=None))
    try:
        backend.abort()                     # a Stop with nothing running
        assert [m.id for m in backend.list_models()] == ["qwen3:8b"]
    finally:
        server.close()
        backend.close()


def test_a_stop_before_the_request_starts_sends_nothing(no_proxy_variables):
    """The cancel event was set before ``stream`` began (Stop pressed as the turn started): the request is not
    sent, and the Stop is not lost to the flags the request resets."""
    server = _Server(_silent)
    backend = make_backend(PRESETS["ollama"], connection("ollama", url=f"http://127.0.0.1:{server.port}/v1",
                                                         key=None))
    cancel = threading.Event()
    cancel.set()
    try:
        with pytest.raises(E.Cancelled):
            backend.stream(request(ask()), lambda e: None, cancel)
        assert not server.accepted.wait(0.3)
    finally:
        server.close()
        backend.close()
