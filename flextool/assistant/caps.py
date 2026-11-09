"""Result caps: how much one tool result may hold, and fitting a payload into them.

The caps are per-call arguments (``max_rows``, ``max_chars``, ``max_results``, ``max_items``); a ``--result-cap``
preset gives their defaults and :data:`CEILINGS` the most any call may ask for.  In the chat the caps also carry
``ceilings`` — the most a call may ask for with this model's context (P3 §4.3) — and a call is clamped to the
smaller of the two.  ``call_tool`` clamps the
arguments before it validates them, so the tool schemas carry no ranges (the tool descriptions state them).

:func:`fit_json` turns a payload into the compact JSON text that is sent: every string redacted
(:func:`flextool.assistant.redact.redact`) and cleaned of invisible characters (:func:`clean`) — once: a
:class:`Rendered` string is already both — then whole trailing rows of the payload's row list dropped until the
text fits ``max_chars``, with a ``truncated`` note saying how many are shown.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable


@dataclass(frozen=True)
class Caps:
    max_rows: int        # table rows in one result
    max_chars: int       # characters of one result's JSON text
    max_results: int     # search hits
    max_items: int       # entries of a listing (entities, runs, scenarios)
    ceilings: Caps | None = None    # the most a call may ask for with this model's context (chat); None: CEILINGS


PRESETS: dict[str, Caps] = {
    "small": Caps(50, 6_000, 5, 100),
    "medium": Caps(200, 20_000, 8, 300),
    "large": Caps(1_000, 60_000, 10, 1_000),
}
DEFAULT_PRESET = "medium"
CEILINGS = Caps(5_000, 200_000, 20, 2_000)

#: Lengths at which user-written strings are cut (P0 §6).
NAME_MAX = 200
DESCRIPTION_MAX = 1_000
LOG_LINE_MAX = 500


def preset(name: str) -> Caps:
    """The caps of a ``--result-cap`` preset."""
    if name not in PRESETS:
        raise ValueError(f"the result cap is one of {', '.join(PRESETS)}")
    return PRESETS[name]


@dataclass(frozen=True)
class CapArg:
    """How ``call_tool`` fills and clamps one caps argument of a tool.

    *field* names the :class:`Caps` field that gives the default; *default* overrides it (``None`` = the preset's
    value, capped at *maximum*); *maximum* is the tool's own upper limit, never above the ceiling of *field*.
    """

    field: str
    maximum: int | None = None
    default: int | None = None

    def limit(self, caps: Caps | None = None) -> int:
        """The most a call may ask for: the global ceiling, the tool's maximum and the context's ceiling."""
        ceiling = getattr(CEILINGS, self.field)
        if caps is not None and caps.ceilings is not None:
            ceiling = min(ceiling, getattr(caps.ceilings, self.field))
        return ceiling if self.maximum is None else min(self.maximum, ceiling)

    def resolve(self, value: Any, caps: Caps) -> int:
        """*value* clamped to 1…the limit; the default when it is missing, not a number or NaN (``+inf`` is the
        limit, ``-inf`` 1: Python's JSON reads ``Infinity`` and ``NaN``)."""
        top = self.limit(caps)
        if (value is None or isinstance(value, bool) or not isinstance(value, (int, float))
                or (isinstance(value, float) and math.isnan(value))):
            base = self.default if self.default is not None else getattr(caps, self.field)
            return max(1, min(int(base), top))
        if isinstance(value, float) and math.isinf(value):
            return top if value > 0 else 1
        return max(1, min(int(value), top))


def clamp_caps(args: dict, cap_args: dict[str, CapArg], caps: Caps) -> dict:
    """*args* with every caps argument of the tool set: the given value clamped to its range, or the default."""
    out = dict(args)
    for name, rule in cap_args.items():
        out[name] = rule.resolve(args.get(name), caps)
    return out


def cap_text(text: Any, limit: int) -> Any:
    """A user string cut to *limit* characters (with an ellipsis); other values unchanged."""
    if isinstance(text, str) and len(text) > limit:
        return text[: max(limit - 1, 0)] + "…"
    return text


class Rendered(str):
    """A text that is already cleaned (:func:`clean`) and redacted: :func:`fit_json` and ``call_tool`` send it as
    it is.  Each text is redacted exactly once: a second pass over text that holds JSON-quoted names could read an
    escaped backslash as part of a path and take the quote's escape with it (review G7, B1)."""


#: Variation selectors (U+FE00–FE0F, U+E0100–E01EF): invisible, used to hide bytes in a text.
_VARIATION_SELECTORS = re.compile("[\ufe00-\ufe0f\U000e0100-\U000e01ef]")


@lru_cache(maxsize=4096)
def _invisible(ch: str) -> bool:
    return unicodedata.category(ch) in ("Cc", "Cf", "Co", "Cn", "Cs") or bool(_VARIATION_SELECTORS.match(ch))


def clean(text: str) -> str:
    """*text* without the characters a reader cannot see: control characters (but newline and tab), format
    characters (bidi overrides, zero-width characters, the Unicode tag characters that can spell hidden
    instructions), private-use and unassigned code points and variation selectors.  Every string that reaches the
    model passes it."""
    if text.isprintable() and not _VARIATION_SELECTORS.search(text):
        return text
    return "".join(ch for ch in text if ch in "\n\t" or not _invisible(ch))


def sent_text(text: str, redact: Callable[[str], str]) -> Rendered:
    """*text* cleaned and redacted, once: a :class:`Rendered` text passes unchanged."""
    if isinstance(text, Rendered):
        return text
    return Rendered(redact(clean(text)))


def dumps(payload: Any) -> str:
    """Compact JSON, non-ASCII kept."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


def _map_strings(value: Any, fn: Callable[[str], str]) -> Any:
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, dict):
        return {(fn(k) if isinstance(k, str) else k): _map_strings(v, fn) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_map_strings(v, fn) for v in value]
    return value


def fit_json(payload: dict, max_chars: int, redact: Callable[[str], str], *, rows_key: str | None = None,
             what: str = "rows", hint: str = "add filters, group_by or top_n") -> str:
    """The JSON text of *payload*, redacted, within *max_chars* characters.

    *rows_key* names the payload's list that may be shortened (default: its longest list).  Whole trailing rows
    are dropped and ``truncated`` says ``showing {k} of {n} {what} — {hint}``.  When the payload still does not fit
    with no rows, its strings are cut, and as a last resort only the ``truncated`` note is sent.
    """
    data = _map_strings(payload, lambda s: sent_text(s, redact))
    text = dumps(data)
    if len(text) <= max_chars:
        return text
    key = rows_key
    if key is None:
        lists = [(len(v), k) for k, v in data.items() if isinstance(v, list)]
        key = max(lists)[1] if lists else None
    if key is not None and isinstance(data.get(key), list) and data[key]:
        rows = data[key]
        n = len(rows)
        lo, hi = 0, n            # largest k with a fitting text, by bisection
        best: str | None = None
        while lo <= hi:
            k = (lo + hi) // 2
            trial = dict(data)
            trial[key] = rows[:k]
            trial["truncated"] = f"showing {k} of {n} {what} — {hint}"
            t = dumps(trial)
            if len(t) <= max_chars:
                best, lo = t, k + 1
            else:
                hi = k - 1
        if best is not None:
            return best
        data = dict(data)
        data[key] = []
        data["truncated"] = f"showing 0 of {n} {what} — {hint}"
    shrunk = _map_strings(data, lambda s: cap_text(s, 200))
    text = dumps(shrunk)
    if len(text) <= max_chars:
        return text
    return dumps({"truncated": f"The result is larger than {max_chars} characters — {hint}."})

