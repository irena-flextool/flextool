"""A ``http.server`` on ``127.0.0.1`` that plays a model server (Ollama, LM Studio, a model list) for the probe
and chat tests."""
from __future__ import annotations

import json
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
