"""Token counts: the usage ledger (P1 §10.3) and the usage lines the front ends show.

``<config>/assistant/usage.json`` counts, per UTC month, per provider and model::

    {"format": 1, "months": {"2026-11": {"<provider key>": {"label": "OpenRouter", "models": {"<model>": {
        "requests", "input_uncached", "cache_read", "cache_write", "output", "not_reported",
        "rejected_attempts", "cost"}}}}}}

Only what the provider reported is counted, never an estimate (D6): ``requests`` are the requests that reached
the model, including stopped and failed ones (with whatever part of the usage was reported); a request whose input
or output count is missing adds one to ``not_reported``.  ``rejected_attempts`` are the rate-limit and busy answers
before any output.  ``cost`` is summed only from values the provider reported (OpenRouter's credits) and stays
``null`` otherwise.  No conversation content is written here; the ledger survives the conversation expiry,
including expiry 0.  Every update is a read-modify-replace under the file's lock; a failing update is a warning on
stderr, never a failed answer.
"""
from __future__ import annotations

import sys
import time
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flextool.assistant import _files
from flextool.assistant.llm_adapter import Usage

FILE_NAME = "usage.json"
TOKEN_FIELDS = ("input_uncached", "cache_read", "cache_write", "output")
COUNT_FIELDS = ("requests", *TOKEN_FIELDS, "not_reported", "rejected_attempts")


def empty_counts() -> dict[str, Any]:
    counts: dict[str, Any] = {name: 0 for name in COUNT_FIELDS}
    counts["cost"] = None
    return counts


def is_partial(usage: Usage) -> bool:
    """Whether *usage* lacks the input or the output count (a stopped or failed request, or none reported)."""
    return not usage.reported or usage.input_uncached is None or usage.output is None


def add_usage(counts: dict[str, Any], usage: Usage | None) -> None:
    """Count one request that reached the model into *counts*; ``None`` is a request with nothing reported."""
    for name in COUNT_FIELDS:
        counts.setdefault(name, 0)
    counts.setdefault("cost", None)
    counts["requests"] += 1
    if usage is None or not usage.reported:
        counts["not_reported"] += 1
        return
    for name in TOKEN_FIELDS:
        value = getattr(usage, name)
        if isinstance(value, int):
            counts[name] += value
    if is_partial(usage):
        counts["not_reported"] += 1
    if usage.cost is not None:
        counts["cost"] = (counts["cost"] or 0.0) + float(usage.cost)


def ledger_key(preset_id: str, base_url: str | None = None, *, forwarded_to: str | None = None) -> str:
    """The ledger's provider key: the preset id; ``openai_compatible:<host>`` for a custom endpoint;
    ``ollama_cloud`` for a model a local Ollama forwards to Ollama cloud."""
    if forwarded_to:
        return "ollama_cloud" if preset_id == "ollama" else f"{preset_id}:{forwarded_to}"
    if preset_id == "openai_compatible":
        from urllib.parse import urlsplit

        host = (urlsplit(base_url or "").hostname or "").lower()
        port = urlsplit(base_url or "").port
        return f"openai_compatible:{host}:{port}" if port else f"openai_compatible:{host}"
    return preset_id


def _month(now: datetime | None) -> str:
    return (now or _files.utc_now()).astimezone(timezone.utc).strftime("%Y-%m")


class Ledger:
    """The usage ledger in *config_root* (default: the user config folder)."""

    def __init__(self, config_root: Path | None = None) -> None:
        self.path = (config_root or _files.config_root()) / FILE_NAME

    def read(self) -> dict[str, Any]:
        """The ledger's data (an empty ledger when there is none or it cannot be read)."""
        try:
            data = _files.read_json(self.path)
        except (OSError, _files.UnreadableFile):
            data = None
        if not data or not isinstance(data.get("months"), dict):
            return {"format": _files.FORMAT, "months": {}}
        return data

    def _update(self, provider_key: str, label: str, model: str, now: datetime | None, change) -> bool:
        try:
            with _files.locked(self.path):
                try:
                    data = _files.read_json(self.path) or {"format": _files.FORMAT, "months": {}}
                except _files.UnreadableFile as exc:
                    aside = self.path.with_name(f"{self.path.name}.damaged-{int(time.time())}")
                    self.path.replace(aside)
                    print(f"Warning: {exc} It was kept as {aside.name} and a new usage record started.",
                          file=sys.stderr)
                    data = {"format": _files.FORMAT, "months": {}}
                if _files.is_newer(data):
                    raise _files.NewerFormat(self.path, data.get("format"))
                months = data.setdefault("months", {})
                provider = months.setdefault(_month(now), {}).setdefault(provider_key, {"label": label, "models": {}})
                provider["label"] = label
                counts = provider.setdefault("models", {}).setdefault(model or "(no model)", empty_counts())
                change(counts)
                _files.write_json_atomic(self.path, data)
            return True
        except (OSError, _files.LockTimeout, _files.NewerFormat) as exc:
            print(f"Warning: the usage record could not be updated: {exc}", file=sys.stderr)
            return False

    def record(self, provider_key: str, label: str, model: str, usage: Usage | None, *,
               now: datetime | None = None) -> bool:
        """Count one request that reached the model (also a stopped or failed one, with its partial usage)."""
        return self._update(provider_key, label, model, now, lambda counts: add_usage(counts, usage))

    def record_rejected(self, provider_key: str, label: str, model: str, attempts: int = 1, *,
                        now: datetime | None = None) -> bool:
        """Count rate-limit or busy answers that came before any output."""
        def change(counts: dict[str, Any]) -> None:
            counts["rejected_attempts"] = counts.get("rejected_attempts", 0) + attempts
        return self._update(provider_key, label, model, now, change)


# --- what the front ends show ----------------------------------------------------------------------------------
def number(value: int) -> str:
    """*value* with a space between thousands: ``3 210``."""
    return f"{value:,}".replace(",", " ")


def short_number(value: int) -> str:
    """*value* to the nearest thousand with "k" (``26k``), the exact number below 1 000 and millions with one
    decimal from 1 000 000 on (``1.2M``); halves round up."""
    value = int(value)
    if value < 0:
        return "-" + short_number(-value)
    if value < 1000:
        return str(value)
    thousands = (value + 500) // 1000
    if thousands < 1000:
        return f"{thousands}k"
    tenths = (value + 50_000) // 100_000
    return f"{tenths // 10}M" if tenths % 10 == 0 else f"{tenths // 10}.{tenths % 10}M"


def _cost(value: float) -> str:
    return f"{value:.6g}"


def _input(counts: Mapping[str, Any]) -> int:
    return sum(int(counts.get(name) or 0) for name in ("input_uncached", "cache_read", "cache_write"))


def _turn_counts(turn: Iterable[Usage | None]) -> dict[str, Any]:
    counts = empty_counts()
    for usage in turn:
        add_usage(counts, usage)
    return counts


def _unreported(counts: Mapping[str, Any]) -> bool:
    nothing = _input(counts) == 0 and counts["output"] == 0
    return counts["requests"] == 0 or (counts["not_reported"] == counts["requests"] and nothing)


def format_turn_line(turn: Iterable[Usage | None], conversation: Mapping[str, Any], provider: str) -> str:
    """The terminal's line after an answer, rounded (:func:`short_number`): ``tokens 3k in (3k from cache) · 412
    out · conversation 10k in · 1k out``, or ``tokens not reported by {provider}`` when no request of the turn
    reported any."""
    counts = _turn_counts(turn)
    if _unreported(counts):
        return f"tokens not reported by {provider}"
    line = f"tokens {short_number(_input(counts))} in"
    if counts["cache_read"]:
        line += f" ({short_number(counts['cache_read'])} from cache)"
    line += f" · {short_number(counts['output'])} out"
    if counts["not_reported"]:
        line += f" · {counts['not_reported']} of {counts['requests']} requests not fully reported"
    line += (f" · conversation {short_number(_input(conversation))} in · "
             f"{short_number(int(conversation.get('output') or 0))} out")
    if counts["cost"] is not None:
        line += f" · cost {_cost(counts['cost'])} (reported by {provider}, in credits)"
    return line


def format_token_line(turn: Iterable[Usage | None] | None, conversation: Mapping[str, Any], provider: str) -> str:
    """The chat window's token line, rounded: ``This question: 26k in, 8k out · Whole chat: 250k in, 5k out``
    ("in" counts every input token, read from the cache or written to it too).  Without a *turn* (a reopened
    chat) only the whole chat; ``""`` when nothing was used."""
    parts: list[str] = []
    if turn is not None:
        counts = _turn_counts(turn)
        if _unreported(counts):
            parts.append(f"This question: tokens not reported by {provider}")
        else:
            parts.append(f"This question: {short_number(_input(counts))} in, {short_number(counts['output'])} out")
            if counts["cost"] is not None:
                parts[-1] += f", cost {_cost(counts['cost'])} credits"
    chat_in, chat_out = _input(conversation), int(conversation.get("output") or 0)
    if chat_in or chat_out:
        parts.append(f"Whole chat: {short_number(chat_in)} in, {short_number(chat_out)} out")
    return " · ".join(parts)


def format_totals(data: Mapping[str, Any]) -> str:
    """The ledger as text, newest month first (``flextool-assistant --usage``, ``/usage``)."""
    months = data.get("months") or {}
    if not months:
        return "No usage recorded yet."
    lines: list[str] = []
    for month in sorted(months, reverse=True):
        lines.append(month)
        for key in sorted(months[month]):
            provider = months[month][key]
            label = provider.get("label") or key
            lines.append(f"  {label}")
            for model in sorted(provider.get("models") or {}):
                c = provider["models"][model]
                parts = [f"{number(int(c.get('requests') or 0))} requests",
                         f"{number(_input(c))} in"
                         + (f" ({number(int(c['cache_read']))} from cache)" if c.get("cache_read") else ""),
                         f"{number(int(c.get('output') or 0))} out"]
                if c.get("not_reported"):
                    parts.append(f"{number(int(c['not_reported']))} not fully reported")
                if c.get("rejected_attempts"):
                    parts.append(f"{number(int(c['rejected_attempts']))} rate-limited attempts")
                if c.get("cost") is not None:
                    parts.append(f"cost {_cost(float(c['cost']))} (reported by {label}, in credits)")
                lines.append(f"    {model}: " + " · ".join(parts))
    lines.append("Tokens as reported by each provider; FlexTool does not estimate what was not reported.")
    return "\n".join(lines)
