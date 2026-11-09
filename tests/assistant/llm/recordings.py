"""Recorded replies of real AI services (P1 §15a): the scrub, the scrub check and the replay.

``tools/record_llm_replies.py`` records a fixed two-question chat on the synthetic test project against a real
service and writes ``recorded/<preset>/<case>.json``; :func:`scrub` runs before anything is written and
:func:`problems` must then find nothing.  ``test_recorded_replies.py`` runs :func:`problems` and :func:`replay` over
every committed file.

A recording::

    {"format": 1, "preset", "model", "base_url", "tier", "max_continuations" (absent: 0, recorded before
     continuations existed), "knobs", "recorded" (UTC date), "flextool_version",
     "facts": {"context_window", "loaded_context", "tools", "remote_host"},
     "questions": [{"question", "final_text", "ok", "rounds", "tools": [names of the tools called]}],
     "probes": [{"method", "path", "request", "status", "body"}],          # the model server's own answers
     "exchanges": [{"phase": "probe" | "chat", "request": {"method", "path", "body"}, "status",
                    "headers": {allow-listed response headers}, "sse_lines": [...] | "body": ...}]}

The scrub drops every request header (none is recorded), replaces message, call, response and request ids by
placeholders of the same format (``chatcmpl-0000…1``), organisation and project ids, run ids, e-mail addresses and
the key, and runs the FlexTool redaction (project folder, home, user and host names, other absolute paths) over every
text, the top-level fields included.  Addresses of the local network are replaced too (:func:`placeholder_base_url`
and :class:`Scrubber`): a ``base_url`` other than the preset's own or one on this computer keeps its scheme, port
and route kind with a placeholder host (``192.0.2.1`` on the local network, else ``endpoint.example.test``) and the
preset's path; a ``remote_host`` outside the known services becomes ``forwarded.example.test``; in every text an IP
address outside loopback and the documentation ranges becomes a numbered documentation address (``192.0.2.N``,
``2001:db8::N``), and a host name of the local network (``*.lan``, ``*.local``, …, or one label after ``://``)
becomes ``host-N.example.test``.  Public host names in texts stay (a model may cite documentation).  Signatures
and thinking text stay: they are needed for the replay, and the project is synthetic.

The check (:func:`problems`) uses its own, stricter patterns, so that what the scrub misses is caught rather than
passed: an IP address outside loopback and the documentation ranges, a host of the local network, a host field
(``base_url``, ``remote_host``) outside the known services, and the original private host names given to it.
"""
from __future__ import annotations

import dataclasses
import functools
import ipaddress
import json
import re
import string
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

RECORDED = Path(__file__).resolve().parent / "recorded"
FORMAT = 1

#: Response headers kept in a recording; everything else (request ids, cookies, organisation) is dropped.
RESPONSE_HEADERS = frozenset({"content-type", "retry-after", "retry-after-ms"})
RESPONSE_HEADER_PREFIXES = ("x-ratelimit-", "anthropic-ratelimit-")
#: Keys whose values are signatures or encrypted reasoning: kept as they are and not checked for long runs.
SIGNATURE_KEYS = frozenset({"signature", "thought_signature", "encrypted_content", "data"})
#: Keys whose text value is an id.
ID_KEYS = frozenset({"id", "tool_call_id", "tool_use_id", "request_id", "request-id", "response_id"})
#: Ids found anywhere in a text, by their prefix.
ID_PREFIXES = ("chatcmpl-", "msg_", "toolu_", "gen-", "req_", "resp_", "call_", "org-", "proj_", "srvtoolu_")
_PREFIXED_ID = re.compile(r"(?<![A-Za-z0-9_-])(?:" + "|".join(re.escape(p) for p in ID_PREFIXES)
                          + r")[A-Za-z0-9_-]{4,}")
_RUN_ID = re.compile(r"(?<![0-9A-Za-z])[0-9a-f]{32}(?![0-9A-Za-z])")
#: A placeholder of a custom endpoint's address (its host may be private).
CUSTOM_BASE_URL = "https://endpoint.example.test/v1"
#: Placeholder hosts: a remote endpoint other than the preset's own, a host a model is forwarded to, and the
#: numbered host names of the local network found in texts.
REMOTE_PLACEHOLDER = "endpoint.example.test"
FORWARDED_PLACEHOLDER = "forwarded.example.test"
#: Keys whose text value is a host or an address.
HOST_KEYS = frozenset({"base_url", "remote_host"})
#: Name suffixes of the local network.
LOCAL_SUFFIXES = ("home.arpa", "localdomain", "intranet", "internal", "local", "home", "corp", "lan")
_DOC_NETWORKS = tuple(ipaddress.ip_network(n) for n in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24",
                                                        "2001:db8::/32"))
#: Keys of a streamed answer whose pieces make one text (the check reads them joined too).
STREAM_KEYS = frozenset({"content", "text", "arguments", "partial_json", "thinking", "reasoning",
                         "reasoning_content"})

# --- the check ----------------------------------------------------------------------------------------------------
_KEY_PATTERNS = (re.compile(r"sk-[A-Za-z0-9_-]{8,}"), re.compile(r"sk-ant-"), re.compile(r"sk-or-"),
                 re.compile(r"AIza[0-9A-Za-z_-]{10,}"))
_LONG_RUN = re.compile(r"[A-Za-z0-9]{32,}")
_ABSOLUTE_PATH = (re.compile(r"(?<![\w<~.])/(?:home|Users|tmp|var|private|root|mnt|opt|srv|media)/"),
                  re.compile(r"(?<![\w])[A-Za-z]:\\"))
_ADDRESS = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PROJECT_LINE = re.compile(r"^Project: (.*)$", re.MULTILINE)
_FORBIDDEN_WORDS = ("x-stainless", "openai-organization", "openai-project")


# --- hosts and addresses ------------------------------------------------------------------------------------------
@functools.cache
def _known_hosts() -> frozenset[str]:
    """The hosts of the presets' own addresses and of Ollama cloud."""
    from flextool.assistant.endpoint import OLLAMA_CLOUD_HOST
    from flextool.assistant.providers import PRESETS

    hosts = {(urlsplit(p.base_url).hostname or "").lower() for p in PRESETS.values() if p.base_url}
    return frozenset(hosts | {OLLAMA_CLOUD_HOST})


def _ip(text: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(text.strip("[]").split("%")[0])
    except ValueError:
        return None


def allowed_ip(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Loopback, unspecified and the documentation ranges (RFC 5737, RFC 3849)."""
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_loopback or address.is_unspecified or any(
        address.version == n.version and address in n for n in _DOC_NETWORKS)


def allowed_host(host: str) -> bool:
    """A host that may be in a recording: a preset's own, Ollama cloud (and its subdomains), ``localhost``, an
    ``example.test`` placeholder or an allowed IP address."""
    host = host.strip().strip("[]").lower().rstrip(".")
    from flextool.assistant.endpoint import OLLAMA_CLOUD_HOST

    if host in _known_hosts() or host.endswith("." + OLLAMA_CLOUD_HOST):
        return True
    if host == "localhost" or host.endswith(".localhost") or host == "example.test" \
            or host.endswith(".example.test"):
        return True
    address = _ip(host)
    return address is not None and allowed_ip(address)


def host_of(value: str) -> str:
    """The host of an address or a bare host (as ``probes.remote_host_of`` reads one)."""
    value = value.strip()
    try:
        return (urlsplit(value if "//" in value else "//" + value).hostname or value).lower()
    except ValueError:
        return value.lower()


def _with_host(url: str, host: str, *, path: str | None = None) -> str:
    """*url* with its host (and, when given, its path) replaced; scheme and port kept, any user name dropped."""
    parts = urlsplit(url if "//" in url else "//" + url)
    netloc = f"[{host}]" if ":" in host else host
    if parts.port is not None:
        netloc += f":{parts.port}"
    out = urlunsplit((parts.scheme, netloc, parts.path if path is None else path, parts.query, parts.fragment))
    return out if "//" in url else out[2:]


def placeholder_base_url(preset_id: str, url: str) -> str:
    """The address written into a recording for *url* of *preset_id*: a custom endpoint's placeholder
    (:data:`CUSTOM_BASE_URL`); the preset's own address or one on this computer as it is; else the same scheme
    and port with the preset's path and a placeholder host of the same route kind (``192.0.2.1`` /
    ``2001:db8::1`` on the local network, :data:`REMOTE_PLACEHOLDER` elsewhere)."""
    from flextool.assistant import net
    from flextool.assistant.providers import PRESETS

    if preset_id == "openai_compatible":
        return CUSTOM_BASE_URL
    preset = PRESETS.get(preset_id)
    default = preset.base_url if preset is not None else None
    if default and net.normalise_endpoint(url) == net.normalise_endpoint(default):
        return url
    kind = net.endpoint_kind(url)
    if kind == "loopback":
        return url
    path = urlsplit(default).path if default else "/v1"
    if kind == "local_network":
        address = _ip(urlsplit(url).hostname or "")
        return _with_host(url, "2001:db8::1" if address is not None and address.version == 6 else "192.0.2.1",
                          path=path)
    return _with_host(url, REMOTE_PLACEHOLDER, path=path)


def _hex_groups(token: str) -> list[str]:
    return [g for g in token.split(":") if g]


def _scrub_ipv6(token: str) -> bool:
    """Whether the scrub reads *token* as an IPv6 address: it parses, has two hex groups or more, and three
    colons or more or a group of three characters or more — so the Python slices ``x[1::2]`` and ``x[::2]``
    and the scope ``a::b`` stay."""
    groups = _hex_groups(token)
    return (_ip(token) is not None and len(groups) >= 2
            and (token.count(":") >= 3 or max(len(g) for g in groups) >= 3))


_SUFFIXES = "|".join(re.escape(s) for s in LOCAL_SUFFIXES)
# The scrub's patterns.
_SCRUB_IPV4 = re.compile(r"(?<![\w.])\d{1,3}(?:\.\d{1,3}){3}(?!\w|\.\d)")
_SCRUB_IPV6 = re.compile(r"(?<![\w:.])[0-9A-Fa-f]{0,4}(?::[0-9A-Fa-f]{0,4}){2,7}(?![\w:])")
_SCRUB_URL_HOST = re.compile(r"(?i)(?<=://)((?:[^\s/@\[\]]+@)?)([a-z0-9-]+(?:\.[a-z0-9-]+)*)")
_SCRUB_LOCAL_NAME = re.compile(r"(?i)(?<![\w.-])(?:[a-z0-9-]+\.)+(?:" + _SUFFIXES + r")(?![\w-]|\.[\w-])")
# The check's patterns: independent of the scrub's and looser in what they match, so a scrub miss is caught.
_CHECK_IPV4 = re.compile(r"(?<!\d)(?=(\d{1,3}(?:\.\d{1,3}){3}))")
_CHECK_IPV6 = re.compile(r"[0-9A-Fa-f:]*:[0-9A-Fa-f:]*:[0-9A-Fa-f:]*")
_CHECK_LOCAL_NAME = re.compile(r"(?i)[a-z0-9-]\.(?:" + _SUFFIXES + r"|arpa)(?![a-z0-9])")
_CHECK_URL_LABEL = re.compile(r"(?i)://(?:[^\s/@]*@)?([a-z0-9_-]+)(?![a-z0-9_.-])")


def _word(host: str) -> re.Pattern:
    """*host* as a whole name: not inside a longer word or name."""
    return re.compile(r"(?i)(?<![\w.-])" + re.escape(host) + r"(?![\w-]|\.[\w-])")


def _host_problems(key: str | None, text: str, private: tuple[str, ...]) -> list[str]:
    found = []
    if key in HOST_KEYS and not allowed_host(host_of(text)):
        found.append(f"a host under {key!r} that is not a known service")
    for match in _CHECK_IPV4.finditer(text):
        address = _ip(match.group(1))
        if address is not None and not allowed_ip(address):
            found.append(f"an IP address under {key!r}")
    for token in _CHECK_IPV6.findall(text):
        # Looser than the scrub's rule (a group of two characters is enough, no word boundary), so whatever the
        # scrub would replace is flagged here; one-character groups (the slice x[1::2]) are not addresses.
        address, groups = _ip(token), _hex_groups(token)
        if address is not None and len(groups) >= 2 and (token.count(":") >= 3 or max(map(len, groups)) >= 2) \
                and not allowed_ip(address):
            found.append(f"an IPv6 address under {key!r}")
    if _CHECK_LOCAL_NAME.search(text):
        found.append(f"a host name of the local network under {key!r}")
    for label in _CHECK_URL_LABEL.findall(text):
        if not allowed_host(label):
            found.append(f"a one-label host after :// under {key!r}")
    lower = text.lower()
    for host in private:
        if ("." in host and host in lower) or ("." not in host and _word(host).search(text)):
            found.append(f"the private host name {host!r} under {key!r}")
    return found


def sse_data(line: str) -> Any:
    """The JSON of an SSE ``data:`` line, else ``None``."""
    if not line.startswith("data:"):
        return None
    text = line[5:].strip()
    if not text or text == "[DONE]":
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def _strings(value: Any, key: str | None = None) -> Iterator[tuple[str | None, str]]:
    """Every text in *value* with the key it is under."""
    if isinstance(value, str):
        yield key, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _strings(v, str(k))
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v, key)


def _texts(record: dict[str, Any]) -> Iterator[tuple[str | None, str]]:
    """Every text of a recording, SSE lines read as their JSON."""
    for exchange in record.get("exchanges") or []:
        lines = exchange.get("sse_lines")
        rest = {k: v for k, v in exchange.items() if k != "sse_lines"}
        yield from _strings(rest)
        joined: dict[str, list[str]] = {}
        for line in lines or []:
            data = sse_data(line)
            if data is None:
                yield None, line
            else:
                for key, text in _strings(data):
                    yield key, text
                    if key in STREAM_KEYS:
                        joined.setdefault(key, []).append(text)
        # A streamed text read whole as well: an address split over two pieces is in neither piece.
        for key, pieces in joined.items():
            if len(pieces) > 1:
                yield key, "".join(pieces)
    yield from _strings({k: v for k, v in record.items() if k != "exchanges"})


def problems(record: dict[str, Any], *, project_name: str, private_hosts: Iterable[str] = ()) -> list[str]:
    """What must not be in a recording: a key, a long run of letters and digits outside a signature, an
    absolute path, an e-mail address, a request header, the provider libraries' identification headers, another
    project name than *project_name*, an IP address outside loopback and the documentation ranges, a host name of
    the local network, a host field outside the known services, and any of *private_hosts* (the hosts the
    recording was made with: a dotted one anywhere, a one-label one as a whole word)."""
    private = tuple(h.lower() for h in private_hosts if h and not allowed_host(h))
    found: list[str] = []
    whole = json.dumps(record, ensure_ascii=False).lower()
    found += [f"the text {word!r} is in the recording" for word in _FORBIDDEN_WORDS if word in whole]
    for i, exchange in enumerate(record.get("exchanges") or []):
        if "headers" in (exchange.get("request") or {}):
            found.append(f"exchange {i} keeps its request headers")
    for key, text in _texts(record):
        for pattern in _KEY_PATTERNS:
            if pattern.search(text):
                found.append(f"a key-like text under {key!r}: {pattern.pattern}")
        if key not in SIGNATURE_KEYS:
            run = _LONG_RUN.search(text)
            if run:
                found.append(f"a run of {len(run.group())} letters and digits under {key!r}")
        for pattern in _ABSOLUTE_PATH:
            if pattern.search(text):
                found.append(f"an absolute path under {key!r}")
        address = _ADDRESS.search(text)
        if address and key not in SIGNATURE_KEYS:
            found.append(f"an e-mail address under {key!r}")
        found += [f"the project name {name!r}, not {project_name!r}" for name in _PROJECT_LINE.findall(text)
                  if name.strip() != project_name]
        found += _host_problems(key, text, private)
    return sorted(set(found))


# --- the scrub ----------------------------------------------------------------------------------------------------
_ALPHABET = string.digits + string.ascii_letters


def _counter(n: int, width: int) -> str:
    out = ""
    while n:
        n, digit = divmod(n, len(_ALPHABET))
        out = _ALPHABET[digit] + out
    return out.rjust(width, "0")[-width:] if width else ""


class Scrubber:
    """One recording's replacements: each id gets one placeholder of its format, the same everywhere."""

    def __init__(self, *, secrets: tuple[str, ...], redact: Callable[[str], str],
                 hosts: Mapping[str, str] | None = None) -> None:
        """*hosts*: the private hosts the recording was made with and their placeholders (the endpoint's host, a
        forwarded host); they are numbered first, so a numbered placeholder never repeats one of them."""
        self.secrets = tuple(s for s in secrets if s)
        self.redact = redact
        self.ids: dict[str, str] = {}
        self.runs: dict[str, str] = {}
        self.names: dict[str, str] = {}             # host name -> placeholder
        self.v4: dict[str, str] = {}                # IPv4 address -> placeholder
        self.v6: dict[str, str] = {}
        self.numbered = 0                           # host-N.example.test given so far
        for host, placeholder in (hosts or {}).items():
            address = _ip(host)
            if address is None:
                self.names[host.lower()] = placeholder
            else:
                (self.v4 if address.version == 4 else self.v6)[str(address)] = placeholder.strip("[]")
        self._known = [(h, _word(h) if "." in h else re.compile(
            r"(?i)(?<![\w.-])" + re.escape(h) + r"(?=:\d)")) for h in sorted(self.names, key=len, reverse=True)]

    # --- hosts and addresses
    def name_placeholder(self, host: str) -> str:
        host = host.lower()
        if host not in self.names:
            self.numbered += 1
            self.names[host] = f"host-{self.numbered}.example.test"
        return self.names[host]

    def ip_placeholder(self, address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str:
        table = self.v4 if address.version == 4 else self.v6
        original = str(address)
        if original not in table:
            n = len(table) + 1
            if address.version == 6:
                table[original] = f"2001:db8::{n:x}"
            else:
                block, host = divmod(n - 1, 254)
                if block >= 3:
                    raise ValueError("more than 762 IPv4 addresses to replace in one recording")
                table[original] = ("192.0.2.", "198.51.100.", "203.0.113.")[block] + str(host + 1)
        return table[original]

    def hosts(self, text: str) -> str:
        """*text* with the private hosts and addresses replaced (see the module text)."""
        for host, pattern in self._known:
            text = pattern.sub(self.names[host], text)

        def url_host(match: re.Match) -> str:
            host = match.group(2)
            local = "." not in host or host.lower().endswith(tuple("." + s for s in LOCAL_SUFFIXES))
            if _ip(host) is not None or allowed_host(host) or not local:
                return match.group(0)
            return self.name_placeholder(host)        # any user name before the host is dropped

        def ipv4(match: re.Match) -> str:
            address = _ip(match.group())
            return match.group() if address is None or allowed_ip(address) else self.ip_placeholder(address)

        def ipv6(match: re.Match) -> str:
            token = match.group()
            if not _scrub_ipv6(token):
                return token
            address = _ip(token)
            return token if address is None or allowed_ip(address) else self.ip_placeholder(address)

        text = _SCRUB_URL_HOST.sub(url_host, text)
        text = _SCRUB_LOCAL_NAME.sub(lambda m: self.name_placeholder(m.group()), text)
        text = _SCRUB_IPV4.sub(ipv4, text)
        return _SCRUB_IPV6.sub(ipv6, text)

    def host_field(self, value: str, key: str) -> str:
        """A ``base_url`` or ``remote_host`` value: as it is for a known service, else with a placeholder host."""
        host = host_of(value)
        if allowed_host(host):
            return value
        if key == "remote_host":
            placeholder = FORWARDED_PLACEHOLDER
        else:
            address = _ip(host)
            placeholder = self.names.get(host, REMOTE_PLACEHOLDER) if address is None \
                else self.ip_placeholder(address)
        return _with_host(value, placeholder) if "//" in value or ":" in value.strip("[]") else placeholder

    def placeholder(self, original: str) -> str:
        if original not in self.ids:
            prefix = next((p for p in ID_PREFIXES if original.startswith(p)), "")
            body = original[len(prefix):]
            slots = [i for i, c in enumerate(body) if c.isalnum()]
            fill = _counter(len(self.ids) + 1, len(slots))
            chars = list(body)
            for slot, c in zip(slots, fill):
                chars[slot] = c
            self.ids[original] = prefix + "".join(chars)
        return self.ids[original]

    def collect(self, value: Any, key: str | None = None) -> None:
        """Note every id in *value* (first pass, so a call id and its result's id get one placeholder)."""
        for k, text in _strings(value, key):
            if k in ID_KEYS and text:
                self.placeholder(text)
            for found in _PREFIXED_ID.findall(text):
                self.placeholder(found)

    def _ids(self, text: str) -> str:
        for secret in self.secrets:
            text = text.replace(secret, "<key>")
        for original in sorted(self.ids, key=len, reverse=True):
            if original in text:
                text = text.replace(original, self.ids[original])
        return text

    def url_path(self, path: str) -> str:
        """A request's URL path: the key, ids and private hosts replaced (the path redaction would cut it)."""
        return self.hosts(self._ids(path))

    def text(self, text: str) -> str:
        text = self._ids(text)

        def run(match: re.Match) -> str:
            return self.runs.setdefault(match.group(), f"<run id {len(self.runs) + 1}>")

        return self.redact(self.hosts(_ADDRESS.sub("<address>", _RUN_ID.sub(run, text))))

    def value(self, value: Any, key: str | None = None) -> Any:
        if isinstance(value, str):
            if key in SIGNATURE_KEYS:
                return value
            return self.host_field(value, key) if key in HOST_KEYS else self.text(value)
        if isinstance(value, dict):
            return {k: self.value(v, str(k)) for k, v in value.items()}
        if isinstance(value, list):
            return [self.value(v, key) for v in value]
        return value

    def sse_line(self, line: str) -> str:
        data = sse_data(line)
        if data is None:
            return self.text(line)
        clean = self.value(data)
        if clean == data:
            return line
        return "data: " + json.dumps(clean, ensure_ascii=False, separators=(",", ":"))


def scrub(record: dict[str, Any], *, secrets: tuple[str, ...], redact: Callable[[str], str]) -> dict[str, Any]:
    """*record* with every id, key, run id, path, name and private address replaced (see the module text)."""
    base_url = str(record.get("base_url") or "")
    placeholder = placeholder_base_url(str(record.get("preset") or ""), base_url) if base_url else base_url
    hosts: dict[str, str] = {}
    base_host = host_of(base_url) if base_url else ""
    if base_host and not allowed_host(base_host):
        hosts[base_host] = host_of(placeholder)
    remote = (record.get("facts") or {}).get("remote_host")
    if isinstance(remote, str) and remote.strip() and not allowed_host(host_of(remote)):
        hosts.setdefault(host_of(remote), FORWARDED_PLACEHOLDER)
    scrubber = Scrubber(secrets=secrets, redact=redact, hosts=hosts)
    old_path = urlsplit(base_url).path.rstrip("/") if base_url else ""
    new_path = urlsplit(placeholder).path.rstrip("/") if placeholder else ""
    # Ids are collected from the chat only: a model list or a model server's answer names models under "id".
    for exchange in record.get("exchanges") or []:
        if exchange.get("phase", "chat") != "chat":
            continue
        scrubber.collect({k: v for k, v in exchange.items() if k != "sse_lines"})
        for line in exchange.get("sse_lines") or []:
            data = sse_data(line)
            if data is not None:
                scrubber.collect(data)
    out = {k: scrubber.value(v, k) for k, v in record.items() if k not in ("exchanges", "probes", "questions")}
    out["base_url"] = placeholder
    out["questions"] = scrubber.value(record.get("questions") or [])
    out["probes"] = scrubber.value(record.get("probes") or [])
    exchanges = []
    for exchange in record.get("exchanges") or []:
        clean = {k: scrubber.value(v, k) for k, v in exchange.items() if k not in ("sse_lines", "request")}
        request = exchange.get("request") or {}
        path = request.get("path")
        if isinstance(path, str):
            if old_path != new_path and (path == old_path or path.startswith(old_path + "/")):
                path = new_path + path[len(old_path):]
            path = scrubber.url_path(path)
        clean["request"] = {"method": request.get("method"), "path": path,
                            "body": scrubber.value(request.get("body"))}
        if "sse_lines" in exchange:
            clean["sse_lines"] = [scrubber.sse_line(line) for line in exchange["sse_lines"]]
        exchanges.append(clean)
    out["exchanges"] = exchanges
    return out


def keep_header(name: str) -> bool:
    lower = name.lower()
    return lower in RESPONSE_HEADERS or lower.startswith(RESPONSE_HEADER_PREFIXES)


# --- the replay ---------------------------------------------------------------------------------------------------
def recordings() -> list[Path]:
    """The committed recordings, sorted."""
    return sorted(RECORDED.glob("*/*.json")) if RECORDED.is_dir() else []


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def response_of(exchange: dict[str, Any]):
    """The ``httpx2`` response of a recorded exchange."""
    import httpx2

    headers = dict(exchange.get("headers") or {})
    if "sse_lines" in exchange:
        content = "\n".join(exchange["sse_lines"]).encode("utf-8")
        headers.setdefault("content-type", "text/event-stream")
    else:
        body = exchange.get("body")
        content = (body if isinstance(body, str) else json.dumps(body)).encode("utf-8")
        headers.setdefault("content-type", "application/json")
    return httpx2.Response(exchange["status"], headers=headers, content=content)


class ReplayTransport:
    """Answers each chat request with the next recorded chat exchange."""

    def __init__(self, record: dict[str, Any]) -> None:
        self.exchanges = [e for e in record.get("exchanges") or [] if e.get("phase", "chat") == "chat"]
        self.served = 0
        self.paths: list[str] = []

    def __call__(self, request):
        if self.served >= len(self.exchanges):
            raise AssertionError(f"the replay asked for more than the {len(self.exchanges)} recorded exchanges")
        exchange = self.exchanges[self.served]
        self.served += 1
        self.paths.append(request.url.path)
        return response_of(exchange)


def replay(record: dict[str, Any], project: Path, config_root: Path) -> tuple[list[str], ReplayTransport]:
    """Run the recorded questions through the real session and backend, the network replaced by the recorded
    answers; returns the final text of each question."""
    import httpx2

    from flextool.assistant.context import make_context
    from flextool.assistant.endpoint import route_for
    from flextool.assistant.llm_adapter import Connection, make_backend
    from flextool.assistant.probes import ModelFacts
    from flextool.assistant.providers import PRESETS
    from flextool.assistant.session import Session, TurnCaps, result_caps_for
    from flextool.assistant.store import open_store

    preset = PRESETS[record["preset"]]
    facts = ModelFacts(**{k: v for k, v in (record.get("facts") or {}).items()
                          if k in ("context_window", "loaded_context", "tools", "remote_host")})
    route = route_for(preset, record["base_url"], facts=facts)
    conn = Connection(base_url=record["base_url"], model=record["model"],
                      api_key="replay-key" if preset.key_required else "", key_source="the replay",
                      knobs=dict(record.get("knobs") or {}), route=route)
    transport = ReplayTransport(record)
    backend = make_backend(preset, conn, transport=httpx2.MockTransport(transport))
    ctx = make_context(project, front_end="terminal")
    level = record["tier"]
    ctx.caps = result_caps_for(level, facts)
    store = open_store(config_root)
    conversation = store.new_conversation(ctx, backend.origin, level)
    # The recording's own continuations: a recording made before they existed asked for an answer at the limit.
    caps = dataclasses.replace(TurnCaps.for_tier(level), max_continuations=int(record.get("max_continuations", 0)))
    session = Session(ctx, backend, preset, conversation=conversation, caps=caps, tier=level,
                      store=store, ledger=None, facts=facts, route=route, sleep=lambda _seconds: False)
    texts = []
    try:
        for question in record.get("questions") or []:
            result = session.run_turn(question["question"], lambda _event: None, threading.Event())
            texts.append(result.text)
    finally:
        store.close(conversation)
        backend.close()
    return texts, transport
