"""Excel sheet names: at most 31 characters, distinct without regard to case, the time-resolution suffix kept.

Excel cuts a longer sheet name and compares names case-insensitively.  Cutting the end of a name loses what tells a
per-step table from its per-period one (``…__dt`` / ``…__d``), so two tables can land on one name or on a name that
says the wrong resolution.  :func:`excel_sheet_name` shortens the stem instead and keeps the suffix.
"""
from __future__ import annotations

import re

#: Excel's limit on the length of a sheet name.
MAX_SHEET_NAME = 31
#: The time-resolution suffix starts at the last ``__d`` / ``__dt`` token that ends the name or is followed by
#: ``_<word>`` (``__d``, ``__dt``, ``__d_average``, ``__dt_…``) and runs to the end.
_SUFFIX_START = re.compile(r"__dt?(?=_[^_]|$)")


def _split(name: str) -> tuple[str, str]:
    """``(stem, suffix)`` of ``name`` (the suffix from the last ``__d…`` token, empty without one)."""
    hits = list(_SUFFIX_START.finditer(name))
    if not hits:
        return name, ""
    start = hits[-1].start()
    return name[:start], name[start:]


def excel_sheet_name(name: str, used: set[str]) -> str:
    """A sheet name for the table ``name``: ``name`` itself when it fits in :data:`MAX_SHEET_NAME` characters and no
    name of ``used`` is the same without regard to case; else the stem shortened in front of the time-resolution
    suffix, with ``_<n>`` (n = 1, 2, …) before the suffix when that is taken.  The result is added to ``used``
    (lower-case)."""
    stem, suffix = _split(name)
    if len(suffix) > MAX_SHEET_NAME - 1:
        stem, suffix = name, ""
    candidate = name if len(name) <= MAX_SHEET_NAME else stem[:MAX_SHEET_NAME - len(suffix)] + suffix
    n = 0
    while candidate.lower() in used:
        n += 1
        tag = f"_{n}"
        candidate = stem[:MAX_SHEET_NAME - len(suffix) - len(tag)] + tag + suffix
    used.add(candidate.lower())
    return candidate
