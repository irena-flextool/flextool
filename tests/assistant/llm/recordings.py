"""Recorded replies of real AI services (P1 §15a): the scrub, the scrub check and the replay.

``tools/record_llm_replies.py`` records a fixed two-question chat on the synthetic test project against a real
service and writes ``recorded/<preset>/<case>.json``; :func:`scrub` runs before anything is written and
:func:`problems` must then find nothing.  ``test_recorded_replies.py`` runs :func:`problems` and :func:`replay` over
every committed file.

A recording::

    {"format": 1, "preset", "model", "base_url", "tier", "knobs", "recorded" (UTC date), "flextool_version",
     "facts": {"context_window", "loaded_context", "tools", "remote_host"},
     "questions": [{"question", "final_text", "ok", "rounds", "tools": [names of the tools called]}],
     "probes": [{"method", "path", "request", "status", "body"}],          # the model server's own answers
     "exchanges": [{"phase": "probe" | "chat", "request": {"method", "path", "body"}, "status",
                    "headers": {allow-listed response headers}, "sse_lines": [...] | "body": ...}]}

The scrub drops every request header (none is recorded), replaces message, call, response and request ids by
placeholders of the same format (``chatcmpl-0000…1``), organisation and project ids, run ids, e-mail addresses and
the key, and runs the FlexTool redaction (project folder, home, user and host names, other absolute paths) over every text.  Signatures
and thinking text stay: they are needed for the replay, and the project is synthetic.
"""
from __future__ import annotations

import json
import re
import string
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

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

# --- the check ----------------------------------------------------------------------------------------------------
_KEY_PATTERNS = (re.compile(r"sk-[A-Za-z0-9_-]{8,}"), re.compile(r"sk-ant-"), re.compile(r"sk-or-"),
                 re.compile(r"AIza[0-9A-Za-z_-]{10,}"))
_LONG_RUN = re.compile(r"[A-Za-z0-9]{32,}")
_ABSOLUTE_PATH = (re.compile(r"(?<![\w<~.])/(?:home|Users|tmp|var|private|root|mnt|opt|srv|media)/"),
                  re.compile(r"(?<![\w])[A-Za-z]:\\"))
_ADDRESS = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PROJECT_LINE = re.compile(r"^Project: (.*)$", re.MULTILINE)
_FORBIDDEN_WORDS = ("x-stainless", "openai-organization", "openai-project")


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
        for line in lines or []:
            data = sse_data(line)
            if data is None:
                yield None, line
            else:
                yield from _strings(data)
    yield from _strings({k: v for k, v in record.items() if k != "exchanges"})


def problems(record: dict[str, Any], *, project_name: str) -> list[str]:
    """What must not be in a recording: a key, a long run of letters and digits outside a signature, an
    absolute path, an e-mail address, a request header, the provider libraries' identification headers, another
    project name than *project_name*."""
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

    def __init__(self, *, secrets: tuple[str, ...], redact: Callable[[str], str]) -> None:
        self.secrets = tuple(s for s in secrets if s)
        self.redact = redact
        self.ids: dict[str, str] = {}
        self.runs: dict[str, str] = {}

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

    def text(self, text: str) -> str:
        for secret in self.secrets:
            text = text.replace(secret, "<key>")
        for original in sorted(self.ids, key=len, reverse=True):
            if original in text:
                text = text.replace(original, self.ids[original])

        def run(match: re.Match) -> str:
            return self.runs.setdefault(match.group(), f"<run id {len(self.runs) + 1}>")

        return self.redact(_ADDRESS.sub("<address>", _RUN_ID.sub(run, text)))

    def value(self, value: Any, key: str | None = None) -> Any:
        if isinstance(value, str):
            return value if key in SIGNATURE_KEYS else self.text(value)
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
    """*record* with every id, key, run id, path and name replaced (see the module text)."""
    scrubber = Scrubber(secrets=secrets, redact=redact)
    # Ids are collected from the chat only: a model list or a model server's answer names models under "id".
    for exchange in record.get("exchanges") or []:
        if exchange.get("phase", "chat") != "chat":
            continue
        scrubber.collect({k: v for k, v in exchange.items() if k != "sse_lines"})
        for line in exchange.get("sse_lines") or []:
            data = sse_data(line)
            if data is not None:
                scrubber.collect(data)
    out = {k: v for k, v in record.items() if k not in ("exchanges", "probes", "questions")}
    out["questions"] = scrubber.value(record.get("questions") or [])
    out["probes"] = scrubber.value(record.get("probes") or [])
    exchanges = []
    for exchange in record.get("exchanges") or []:
        clean = {k: scrubber.value(v, k) for k, v in exchange.items() if k not in ("sse_lines", "request")}
        request = exchange.get("request") or {}
        clean["request"] = {"method": request.get("method"), "path": request.get("path"),
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
    session = Session(ctx, backend, preset, conversation=conversation, caps=TurnCaps.for_tier(level), tier=level,
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
