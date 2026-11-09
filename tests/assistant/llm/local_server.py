"""A ``http.server`` on ``127.0.0.1`` that plays a model server (Ollama, LM Studio, a model list) for the probe
and chat tests, and a raw one that records the requests as they came over the wire (headers, proxies)."""
from __future__ import annotations

import json
import socketserver
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

Answer = tuple[int, object]


class FakeServer:
    """Answers ``(method, path)`` from *routes*: a ``(status, body)`` pair (a dict body is sent as JSON, a string as
    it is) or a callable ``(request body) -> (status, body)``.  Every request is recorded."""

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], Answer | Callable[[object], Answer]] = {}
        self.requests: list[tuple[str, str, object, dict]] = []
        self.release = threading.Event()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def _answer(self, method: str) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                body = json.loads(raw) if raw else None
                fake.requests.append((method, self.path, body, dict(self.headers)))
                route = fake.routes.get((method, self.path), (404, {"error": "not found"}))
                status, out = route(body) if callable(route) else route
                data = (json.dumps(out) if isinstance(out, (dict, list)) else str(out)).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                self._answer("GET")

            def do_POST(self) -> None:
                self._answer("POST")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def paths(self) -> list[str]:
        return [path for _method, path, _body, _headers in self.requests]

    def close(self) -> None:
        self.release.set()
        self.server.shutdown()
        self.server.server_close()


class RawServer:
    """A server on ``127.0.0.1`` that records each request as it came over the wire — the request line, every
    header line (repeated ones too) and the body read by the request's own framing (``Content-Length`` or chunked;
    neither: no body) — and answers every request with *status*, *content_type* and *body*, then closes the
    connection.  Plays a model server whose headers are checked, or a proxy (the request line then carries the
    absolute address)."""

    def __init__(self, status: int = 200, body: bytes = b"{}", content_type: str = "application/json") -> None:
        self.status, self.body, self.content_type = status, body, content_type
        self.requests: list[tuple[str, list[tuple[str, str]], bytes]] = []
        raw = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                line = self.rfile.readline().decode("latin-1").rstrip("\r\n")
                headers: list[tuple[str, str]] = []
                while True:
                    header = self.rfile.readline().decode("latin-1").rstrip("\r\n")
                    if not header:
                        break
                    name, _, value = header.partition(":")
                    headers.append((name.strip().lower(), value.strip()))
                found = dict(headers)
                if "content-length" in found:
                    data = self.rfile.read(int(found["content-length"]))
                elif found.get("transfer-encoding", "").lower() == "chunked":
                    data = b""
                    while (size := int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)) > 0:
                        data += self.rfile.read(size)
                        self.rfile.readline()
                    self.rfile.readline()
                else:
                    data = b""
                raw.requests.append((line, headers, data))
                self.wfile.write(f"HTTP/1.1 {raw.status} X\r\nContent-Type: {raw.content_type}\r\nContent-Length: "
                                 f"{len(raw.body)}\r\nConnection: close\r\n\r\n".encode("latin-1") + raw.body)

        self.server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def lines(self) -> list[str]:
        return [line for line, _headers, _body in self.requests]

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
