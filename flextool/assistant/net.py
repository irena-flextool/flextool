"""The HTTP client FlexTool gives every AI service library (P1 §3.0).

FlexTool, not the library, decides what a request carries and where it goes:

* **Headers** are rebuilt by a request hook, after the library and its environment variables
  (``OPENAI_CUSTOM_HEADERS``, ``ANTHROPIC_CUSTOM_HEADERS``) have added theirs: the library's identification
  headers (version, operating system, processor, Python), organisation and project ids, custom headers and beta
  switches are dropped.  Only the key header (and a header a preset requires) keeps the library's value, and the
  key header must carry the resolved key, else the request is stopped (a programming error, never silent).  Every
  other header FlexTool sets itself, so no environment variable can change it: the fixed ones
  (:data:`FIXED_HEADERS`, ``user-agent`` ``FlexTool-assistant``, Anthropic's ``anthropic-version`` of the
  installed library) and the ones computed from the request (:data:`COMPUTED_HEADERS`: the ``host`` of the URL
  and the body's ``content-type`` and length).
* **Proxies**: an endpoint on this computer or the local network never goes through a proxy; a remote endpoint
  uses the proxy :func:`proxy_for` names (the environment settings, read the way ``httpx2`` reads them with their
  ``NO_PROXY`` rules, else the system settings with their exceptions).  The requests and the probes
  (:mod:`flextool.assistant.probes`) both go exactly where :func:`proxy_for` says, so the proxy FlexTool shows is
  the one used.
* **Plain http**: a remote ``http://`` address sends the key and the data unencrypted (:func:`plain_http_remote`);
  :data:`REMOTE_HTTP` says whether FlexTool warns about it (the indicator and the data notice say so) or refuses
  the address.
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
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

from flextool.assistant.llm_adapter import Connection, EndpointKind, Route

if TYPE_CHECKING:
    import httpx2

USER_AGENT = "FlexTool-assistant"
#: Sent as the key when an endpoint needs none, so a library never falls back to a key from the environment.
NO_KEY = "not-needed"
#: The header that carries the key, and its value for a key.
KEY_HEADER = "authorization"
#: Headers that keep the value the library gives them: only the key header, which the hook checks.
ALLOWED_HEADERS = frozenset({KEY_HEADER})
#: Headers FlexTool sets to these values on every request, whatever the library or the environment put there.
#: Both libraries ask for JSON (a streamed answer included), and ``httpx2`` decodes gzip and deflate.
FIXED_HEADERS: Mapping[str, str] = {"accept": "application/json", "accept-encoding": "gzip, deflate",
                                    "connection": "keep-alive", "user-agent": USER_AGENT}
#: Headers computed from the request: the address, and for a body its type and framing.
COMPUTED_HEADERS = frozenset({"host", "content-type", "content-length", "transfer-encoding"})
#: The type of every request body (both libraries send JSON).
BODY_TYPE = "application/json"
#: Headers a backend sets to a fixed value of its own (:func:`backend_headers`).
BACKEND_HEADERS = {"openai": frozenset(), "anthropic": frozenset({"anthropic-version"})}

CONNECT_TIMEOUT = 10.0
#: Longest silence between two parts of an answer: a model on this computer may load or read the prompt for
#: minutes before it sends anything, and a large remote model may think for minutes before its first byte (one
#: took 88 s to the first streamed byte and over 150 s for a plain request).
READ_TIMEOUT_REMOTE = 300.0
READ_TIMEOUT_LOCAL = 900.0

BASE_URL_CREDENTIALS = "Put the key in the key setting, not in the address; FlexTool stores and shows the address."
BASE_URL_SCHEME = "The address must start with http:// or https://, for example http://localhost:8000/v1."
BASE_URL_PLAIN_REMOTE = ("FlexTool sends a key and case data to an address outside this computer and your local "
                         "network only over https://. Use https://{host}.")

#: What FlexTool does with a plain ``http://`` address outside this computer and the local network: ``"warn"``
#: (the indicator and the data notice say the connection is not encrypted) or ``"refuse"`` (the address is not
#: used).  Owner decision D2 (pending; ``"warn"`` is the recommended option).
REMOTE_HTTP: Literal["warn", "refuse"] = "warn"


class BaseUrlRefused(ValueError):
    """A base URL FlexTool does not use, with the message for the user."""


def key_header_value(api_key: str) -> str:
    return f"Bearer {api_key}"


def backend_headers(backend: Literal["openai", "anthropic"]) -> dict[str, str]:
    """The fixed headers of *backend*: for Anthropic the API version the installed library is written for (the
    value the library itself sends, which ``ANTHROPIC_CUSTOM_HEADERS`` could otherwise replace)."""
    if backend == "anthropic":
        from anthropic._constants import ANTHROPIC_API_VERSION

        return {"anthropic-version": ANTHROPIC_API_VERSION}
    return {}


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
    """The proxy a request to *url* goes through — the one route decision of the requests and the probes:
    ``None`` for an endpoint on this computer or the local network, else what the environment and system settings
    give (``urllib.request.getproxies()`` with the ``NO_PROXY`` rules, matched exactly as ``httpx2`` matches
    them), and ``None`` for a host the system settings list as an exception (:func:`_system_exception`)."""
    if is_local(url):
        return None
    import httpx2
    from httpx2._utils import URLPattern, get_environment_proxies

    target = httpx2.URL(url)
    patterns = sorted((URLPattern(key), value) for key, value in get_environment_proxies().items())
    for pattern, proxy in patterns:
        if pattern.matches(target):
            return None if proxy is None or _system_exception(url) else proxy
    return None


def _system_exception(url: str) -> bool:
    """Whether the system proxy settings list the host of *url* as an exception (Windows: the registry's
    ``ProxyOverride``; macOS: the exceptions of the network settings).  Only when no proxy environment variable is
    set: the proxies then come from the system settings, whose exceptions ``httpx2`` does not read; with proxy
    variables, their ``NO_PROXY`` rules (matched in :func:`proxy_for`) are the exceptions."""
    import urllib.request

    if urllib.request.getproxies_environment():
        return False
    return bool(urllib.request.proxy_bypass(urlsplit(url).netloc))


def plain_http_remote(url: str) -> bool:
    """Whether *url* is a plain ``http://`` address outside this computer and the local network: the key and the
    data then cross networks unencrypted.  This computer and the local network are a lower-risk class and never
    count."""
    return urlsplit(url.strip()).scheme.lower() == "http" and endpoint_kind(url.strip()) == "remote"


def _check_plain_http(url: str) -> None:
    """Raise :class:`BaseUrlRefused` for a remote ``http://`` address when :data:`REMOTE_HTTP` refuses those."""
    if REMOTE_HTTP == "refuse" and plain_http_remote(url):
        host = _host(url.strip())
        raise BaseUrlRefused(BASE_URL_PLAIN_REMOTE.format(host=f"[{host}]" if ":" in host else host))


def check_base_url(url: str) -> str:
    """*url* if FlexTool uses it as a base URL; raises :class:`BaseUrlRefused` for an address with a user name,
    password or query (a key belongs in the key setting), one that is not http(s), and — when :data:`REMOTE_HTTP`
    is ``"refuse"`` — a plain ``http://`` address outside this computer and the local network."""
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise BaseUrlRefused(BASE_URL_SCHEME)
    if parts.username is not None or parts.password is not None or parts.query or "@" in parts.netloc:
        raise BaseUrlRefused(BASE_URL_CREDENTIALS)
    _check_plain_http(url)
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
    requests to (an Ollama cloud model): the route is then remote, and *server_kind* keeps where that server runs.
    The route is not *encrypted* when *base_url* is a remote ``http://`` address (:func:`plain_http_remote`)."""
    server_kind = endpoint_kind(base_url)
    proxy = proxy_for(base_url)
    proxy_host = urlsplit(proxy).hostname if proxy else None
    kind: EndpointKind = "remote" if forwarded_to else server_kind
    return Route(kind=kind, host=_host_port(base_url), proxy_host=proxy_host, forwarded_to=forwarded_to,
                 label=label, server_kind=server_kind, encrypted=not plain_http_remote(base_url))


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


def _set_body_headers(request: Any) -> None:
    """Set the ``content-type`` and the framing of *request* from its body: ``content-length`` for a body in
    memory (and for an empty POST, PUT or PATCH, as ``httpx2`` sends it), ``transfer-encoding: chunked`` for a
    streamed one; a request without a body gets neither."""
    import httpx2

    for name in ("content-type", "content-length", "transfer-encoding"):
        request.headers.pop(name, None)
    try:
        body = request.content
    except httpx2.RequestNotRead:
        request.headers["content-type"] = BODY_TYPE
        request.headers["transfer-encoding"] = "chunked"
        return
    if body:
        request.headers["content-type"] = BODY_TYPE
    if body or request.method in ("POST", "PUT", "PATCH"):
        request.headers["content-length"] = str(len(body))


def header_hook(api_key: str, allowed: Iterable[str], fixed: Mapping[str, str] | None = None) -> Callable[[Any], None]:
    """The request hook: keeps only the *allowed* headers with the library's value, sets :data:`FIXED_HEADERS`
    and the backend's *fixed* ones, computes :data:`COMPUTED_HEADERS` from the request, and checks the key
    header."""
    keep = frozenset(h.lower() for h in allowed)
    values = {**FIXED_HEADERS, **(fixed or {})}
    expected = key_header_value(api_key)

    def hook(request: Any) -> None:
        for name in list(request.headers.keys()):
            if name.lower() not in keep:
                del request.headers[name]
        for name, value in values.items():
            request.headers[name] = value
        request.headers["host"] = request.url.netloc.decode("ascii")
        _set_body_headers(request)
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
    _check_plain_http(conn.base_url)
    local = is_local(conn.base_url)
    # The one route decision (the probes use it too).  The transport is passed in, so ``httpx2`` reads no proxy
    # of its own from the environment: the request goes exactly where proxy_for says.
    proxy = proxy_for(conn.base_url)
    registry = _SocketRegistry()
    if transport is None:
        transport = httpx2.HTTPTransport(trust_env=not local, proxy=proxy)
        pool = getattr(transport, "_pool", None)
        if pool is None or not hasattr(pool, "_network_backend"):
            raise RuntimeError("this httpx2 version has another transport layout; the cancel hook cannot be set")
        pool._network_backend = _tracked_backend(registry)
    timeout = httpx2.Timeout(read_timeout if read_timeout is not None else read_timeout_for(conn.base_url),
                             connect=CONNECT_TIMEOUT)
    allowed = ALLOWED_HEADERS | frozenset(h.lower() for h in required_headers)
    session = HttpSession(client=None, proxy=proxy, _registry=registry)  # type: ignore[arg-type]
    session.client = httpx2.Client(transport=transport, trust_env=not local, follow_redirects=False, timeout=timeout,
                                   event_hooks={"request": [header_hook(api_key, allowed,
                                                                        backend_headers(backend))],
                                                "response": [session.keep_rate_headers]})
    return session
