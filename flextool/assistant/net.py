"""The HTTP client FlexTool gives every AI service library (P1 §3.0).

FlexTool, not the library, decides what a request carries and where it goes:

* **Headers** are rebuilt from an allow-list by a request hook, after the library and its environment variables
  have added theirs: the library's identification headers (version, operating system, processor, Python),
  organisation and project ids, custom headers from the environment and beta switches are dropped; the
  ``user-agent`` is ``FlexTool-assistant``; the key header must carry the resolved key, else the request is
  stopped (a programming error, never silent).
* **Proxies**: an endpoint on this computer or the local network never goes through a proxy; a remote endpoint
  uses the proxy :func:`proxy_for` names (the system and environment settings, read the way ``httpx2`` reads them),
  so the proxy FlexTool shows is the one used.
* **Redirects** are never followed.
* **Response headers**: only the request-limit headers of :data:`~flextool.assistant.pacing.RATE_HEADERS` are
  read, by a response hook, for the pacing (P1 §5.1); they are kept as counts and times, never logged.
* **Cancel**: :meth:`HttpSession.abort_connections` shuts the open sockets down, which wakes a read that is
  waiting for the first byte (closing a socket from another thread does not).
"""
from __future__ import annotations

import contextlib
import ipaddress
import socket
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

from flextool.assistant.llm_adapter import Connection, EndpointKind, Route

if TYPE_CHECKING:
    import httpx2

USER_AGENT = "FlexTool-assistant"
#: Sent as the key when an endpoint needs none, so a library never falls back to a key from the environment.
NO_KEY = "not-needed"
#: Headers every request may carry.
ALLOWED_HEADERS = frozenset({"host", "accept", "accept-encoding", "content-type", "content-length", "connection",
                             "user-agent", "authorization"})
#: Headers a backend adds to :data:`ALLOWED_HEADERS`.
BACKEND_HEADERS = {"openai": frozenset(), "anthropic": frozenset({"anthropic-version"})}
#: The header that carries the key, and its value for a key.
KEY_HEADER = "authorization"

CONNECT_TIMEOUT = 10.0
#: Longest silence between two parts of an answer: a model on this computer may load or read the prompt for
#: minutes before it sends anything.
READ_TIMEOUT_REMOTE = 120.0
READ_TIMEOUT_LOCAL = 900.0

BASE_URL_CREDENTIALS = "Put the key in the key setting, not in the address; FlexTool stores and shows the address."
BASE_URL_SCHEME = "The address must start with http:// or https://, for example http://localhost:8000/v1."


class BaseUrlRefused(ValueError):
    """A base URL FlexTool does not use, with the message for the user."""


def key_header_value(api_key: str) -> str:
    return f"Bearer {api_key}"


# --- where an address points --------------------------------------------------------------------------------------
def _host(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def endpoint_kind(url: str) -> EndpointKind:
    """``loopback`` (this computer), ``local_network`` or ``remote``, from the literal host of *url* — no name is
    looked up, so any host name other than ``localhost`` counts as remote."""
    host = _host(url)
    if host == "localhost" or host.endswith(".localhost"):
        return "loopback"
    try:
        address = ipaddress.ip_address(host.split("%")[0])
    except ValueError:
        return "remote"
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    if address.is_loopback or address.is_unspecified:
        return "loopback"
    if address.is_private or address.is_link_local:
        return "local_network"
    return "remote"


def is_local(url: str) -> bool:
    return endpoint_kind(url) != "remote"


def read_timeout_for(url: str) -> float:
    return READ_TIMEOUT_LOCAL if is_local(url) else READ_TIMEOUT_REMOTE


def proxy_for(url: str) -> str | None:
    """The proxy a request to *url* goes through: ``None`` for an endpoint on this computer or the local network,
    else what the environment and system settings give (``urllib.request.getproxies()`` with the ``NO_PROXY``
    rules, matched exactly as ``httpx2`` matches them)."""
    if is_local(url):
        return None
    import httpx2
    from httpx2._utils import URLPattern, get_environment_proxies

    target = httpx2.URL(url)
    patterns = sorted((URLPattern(key), value) for key, value in get_environment_proxies().items())
    for pattern, proxy in patterns:
        if pattern.matches(target):
            return proxy
    return None


def check_base_url(url: str) -> str:
    """*url* if FlexTool uses it as a base URL; raises :class:`BaseUrlRefused` for an address with a user name,
    password or query (a key belongs in the key setting), or one that is not http(s)."""
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise BaseUrlRefused(BASE_URL_SCHEME)
    if parts.username is not None or parts.password is not None or parts.query or "@" in parts.netloc:
        raise BaseUrlRefused(BASE_URL_CREDENTIALS)
    return url.strip()


def normalise_endpoint(url: str) -> str:
    """*url* in one spelling for keys and acknowledgements: scheme and host in lower case, the port only when it is
    not the scheme's default, no trailing slash (``HTTP://LocalHost:11434/v1/`` -> ``http://localhost:11434/v1``)."""
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    if ":" in host:
        host = f"[{host}]"
    port = parts.port
    default = {"http": 80, "https": 443}.get(scheme)
    netloc = host if port is None or port == default else f"{host}:{port}"
    return f"{scheme}://{netloc}{parts.path.rstrip('/')}"


def _host_port(url: str) -> str:
    parts = urlsplit(url)
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    return f"{host}:{parts.port}" if parts.port else host


def route_for(base_url: str, label: str, *, forwarded_to: str | None = None) -> Route:
    """The route of a connection to *base_url*.  *forwarded_to* is the host a local server forwards the model's
    requests to (an Ollama cloud model): the route is then remote."""
    kind = endpoint_kind(base_url)
    proxy = proxy_for(base_url)
    proxy_host = urlsplit(proxy).hostname if proxy else None
    if forwarded_to:
        kind = "remote"
    return Route(kind=kind, host=_host_port(base_url), proxy_host=proxy_host, forwarded_to=forwarded_to,
                 label=label)


# --- sockets that can be shut down from another thread ------------------------------------------------------------
class _SocketRegistry:
    def __init__(self) -> None:
        self._streams: set[Any] = set()
        self._lock = threading.Lock()

    def add(self, stream: Any) -> None:
        with self._lock:
            self._streams.add(stream)

    def discard(self, stream: Any) -> None:
        with self._lock:
            self._streams.discard(stream)

    def shutdown_all(self) -> None:
        with self._lock:
            streams = list(self._streams)
        for stream in streams:
            # A stream that is gone, or a socket already shut, has nothing left to shut.
            with contextlib.suppress(Exception):
                sock = stream.get_extra_info("socket")
                if sock is not None:
                    sock.shutdown(socket.SHUT_RDWR)


def _tracked_backend(registry: _SocketRegistry):
    import httpcore2

    class TrackedStream(httpcore2.NetworkStream):
        def __init__(self, inner: httpcore2.NetworkStream) -> None:
            self._inner = inner
            registry.add(self)

        def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
            return self._inner.read(max_bytes, timeout)

        def write(self, buffer: bytes, timeout: float | None = None) -> None:
            self._inner.write(buffer, timeout)

        def close(self) -> None:
            registry.discard(self)
            self._inner.close()

        def start_tls(self, ssl_context, server_hostname=None, timeout=None):
            registry.discard(self)
            return TrackedStream(self._inner.start_tls(ssl_context, server_hostname, timeout))

        def get_extra_info(self, info: str) -> Any:
            return self._inner.get_extra_info(info)

    class TrackedBackend(httpcore2.NetworkBackend):
        def __init__(self) -> None:
            self._inner = httpcore2.SyncBackend()

        def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
            return TrackedStream(self._inner.connect_tcp(host, port, timeout=timeout, local_address=local_address,
                                                         socket_options=socket_options))

        def connect_unix_socket(self, path, timeout=None, socket_options=None):
            return TrackedStream(self._inner.connect_unix_socket(path, timeout=timeout,
                                                                 socket_options=socket_options))

        def sleep(self, seconds: float) -> None:
            self._inner.sleep(seconds)

    return TrackedBackend()


@dataclass
class HttpSession:
    """The ``httpx2`` client of one backend, and the way to shut its open connections."""

    client: httpx2.Client
    proxy: str | None
    _registry: _SocketRegistry = field(default_factory=_SocketRegistry, repr=False)
    _rate: dict[str, str] | None = field(default=None, repr=False)
    _rate_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def keep_rate_headers(self, response: Any) -> None:
        """The response hook: keep the request-limit headers of the latest response."""
        from flextool.assistant.pacing import rate_headers

        found = rate_headers(response.headers)
        with self._rate_lock:
            self._rate = found

    def take_rate_headers(self) -> dict[str, str] | None:
        """The request-limit headers of the latest response (``None`` when none arrived since the last take)."""
        with self._rate_lock:
            found, self._rate = self._rate, None
        return found

    def abort_connections(self) -> None:
        """Shut every open connection down at once; a request waiting on one fails immediately."""
        self._registry.shutdown_all()

    def close(self) -> None:
        self.client.close()


def header_hook(api_key: str, allowed: Iterable[str]) -> Callable[[Any], None]:
    """The request hook that keeps only the *allowed* headers and checks the key header."""
    keep = frozenset(h.lower() for h in allowed)
    expected = key_header_value(api_key)

    def hook(request: Any) -> None:
        for name in list(request.headers.keys()):
            if name.lower() not in keep:
                del request.headers[name]
        request.headers["user-agent"] = USER_AGENT
        if request.headers.get(KEY_HEADER) != expected:
            raise RuntimeError("the request's key header does not carry the key FlexTool resolved")

    return hook


def build_http_client(conn: Connection, *, backend: Literal["openai", "anthropic"], api_key: str,
                      read_timeout: float | None = None, required_headers: frozenset[str] = frozenset(),
                      transport: Any = None) -> HttpSession:
    """The client a backend of *conn* passes to its library.  *api_key* is the key sent (never empty);
    *transport* replaces the network (tests only)."""
    import httpx2

    if not api_key:
        raise ValueError("a request always carries a key; use NO_KEY for an endpoint without one")
    local = is_local(conn.base_url)
    proxy = None if local else proxy_for(conn.base_url)
    registry = _SocketRegistry()
    if transport is None:
        transport = httpx2.HTTPTransport(trust_env=not local, proxy=proxy)
        pool = getattr(transport, "_pool", None)
        if pool is None or not hasattr(pool, "_network_backend"):
            raise RuntimeError("this httpx2 version has another transport layout; the cancel hook cannot be set")
        pool._network_backend = _tracked_backend(registry)
    timeout = httpx2.Timeout(read_timeout if read_timeout is not None else read_timeout_for(conn.base_url),
                             connect=CONNECT_TIMEOUT)
    allowed = ALLOWED_HEADERS | BACKEND_HEADERS[backend] | frozenset(h.lower() for h in required_headers)
    session = HttpSession(client=None, proxy=proxy, _registry=registry)  # type: ignore[arg-type]
    session.client = httpx2.Client(transport=transport, trust_env=not local, follow_redirects=False, timeout=timeout,
                                   event_hooks={"request": [header_hook(api_key, allowed)],
                                                "response": [session.keep_rate_headers]})
    return session
