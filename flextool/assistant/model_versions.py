"""The latest version of each model family in a service's model list (owner feedback 2026-11-08).

A service lists every version it still answers for, also ones closed to new accounts (Gemini listed
``gemini-2.5-flash`` after it stopped serving it to new users).  The settings dialog and ``--list-models`` show only
the newest version of each family by default; "Show all models" / ``--all`` show the whole list.  Presets name no
models (P1-O1): the families come from the ids the service returns, never from a list of model names.

:func:`model_key` splits an id into its family and version:

* the ``models/`` prefix is dropped; a vendor path (``google/`` on OpenRouter) and a ``:suffix`` (``:free``)
  belong to the family;
* the first run of short numbers is the version (``2.5``, ``3``, ``3-1``, ``4-5``, ``v3.1``);
* numbers after it — dates (``20251001``, ``2025-08-07``, ``09-2025``), snapshots (``001``, ``2411``) — are the
  snapshot, a lower-priority part of the version;
* ``preview`` / ``exp`` / ``experimental`` / ``alpha`` / ``beta`` mark a preview, ``latest`` a moving alias;
* every other word stays in the family, in order (``gemini-flash``, ``gemini-flash-lite``, ``claude-haiku``,
  ``gpt-mini``, ``mistral-large``, ``llama-70b-instruct``);
* an id without a version number is never compared with numbered ones of the same words: it is a family of its
  own, together with its dated snapshots and its ``latest`` alias (``gpt-4o`` and ``gpt-4o-2024-08-06``).

:func:`latest_ids` keeps, in each family: the newest stable release (an undated alias before the dated snapshots
of its version); the newest preview when it is of a newer version than that release (or there is none); and every
``latest`` alias without a version number (one with a version, ``claude-3-5-sonnet-latest``, is that
version's undated name).  A ``latest`` alias stands for the dated snapshots of a family that has no version number
(``mistral-large-latest`` for ``mistral-large-2411``), so those are left out beside it; beside numbered versions
(``gemini-flash-latest``) it is shown as well, since it may point to a preview.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

_VERSION = re.compile(r"v?\d{1,2}(?:\.\d+)*", re.IGNORECASE)
_NUMBER = re.compile(r"v?\d+(?:\.\d+)*", re.IGNORECASE)
_YEAR = re.compile(r"(?:19|20)\d\d")
_PREVIEW = frozenset({"preview", "exp", "experimental", "alpha", "beta"})
_ALIAS = frozenset({"latest"})


@dataclass(frozen=True)
class ModelKey:
    family: str
    version: tuple[int, ...]        # () when the id has no version number
    snapshot: tuple[int, ...]       # () for an undated id
    preview: bool
    alias: bool                     # a moving "latest" name


def _ints(token: str) -> tuple[int, ...]:
    return tuple(int(p) for p in token.lower().lstrip("v").split(".") if p)


def _snapshot(tokens: list[str]) -> tuple[int, ...]:
    """The snapshot numbers, a date as one number YYYYMMDD (YYYY-MM-DD, YYYY-MM, MM-YYYY), else in order."""
    out: list[int] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if _YEAR.fullmatch(tok) and i + 1 < len(tokens) and len(tokens[i + 1]) == 2:
            day = tokens[i + 2] if i + 2 < len(tokens) and len(tokens[i + 2]) == 2 else "0"
            out.append(int(tok) * 10000 + int(tokens[i + 1]) * 100 + int(day))
            i += 3 if day != "0" else 2
        elif len(tok) == 2 and i + 1 < len(tokens) and _YEAR.fullmatch(tokens[i + 1]):
            out.append(int(tokens[i + 1]) * 10000 + int(tok) * 100)
            i += 2
        else:
            out.extend(_ints(tok))
            i += 1
    return tuple(out)


def model_key(model: str) -> ModelKey:
    """The family and version of the model id *model* (see the module text)."""
    name = model.strip().removeprefix("models/")
    vendor, _slash, base = name.rpartition("/")
    base, colon, suffix = base.partition(":")
    words: list[str] = []
    version: tuple[int, ...] = ()
    snapshot: list[str] = []
    preview = alias = False
    state = "before"                # before the version, in it, after it
    for tok in (t for t in re.split(r"[-_]", base) if t):
        low = tok.lower()
        if low in _ALIAS:
            alias = True
            continue
        if low in _PREVIEW or not _NUMBER.fullmatch(tok):
            preview = preview or low in _PREVIEW
            if low not in _PREVIEW:
                words.append(low)
            if state == "in":
                state = "after"         # a word ends the version
            continue
        if state != "after" and not snapshot and _VERSION.fullmatch(tok) and not _YEAR.fullmatch(tok):
            version += _ints(tok)
            state = "in"
        else:
            snapshot.append(low)
            state = "after"
    family = "-".join(words) or base.lower()
    if vendor:
        family = f"{vendor.lower()}/{family}"
    if colon:
        family = f"{family}:{suffix.lower()}"
    return ModelKey(family, version, _snapshot(snapshot), preview, alias)


def _rank(key: ModelKey) -> tuple:
    # An undated alias of a version before its dated snapshots; then the newest snapshot.
    return (key.version, (1,) if not key.snapshot else (0, *key.snapshot))


def latest_ids(ids: Iterable[str]) -> list[str]:
    """The ids of *ids* that are the latest version of their family, in the order given."""
    ids = list(ids)
    families: dict[tuple[str, bool], list[tuple[str, ModelKey]]] = {}
    for model in ids:
        key = model_key(model)
        # An id without a version number is never compared with numbered ones (``assistant-model`` is not an
        # older ``assistant-model-2``); it is the undated alias of its dated snapshots (``gpt-4o``).
        families.setdefault((key.family, bool(key.version)), []).append((model, key))
    keep: set[str] = set()
    for (_family, numbered), members in families.items():
        # A "latest" without a version number stands for the snapshots beside it; one with a version number
        # (``claude-3-5-sonnet-latest``) is the undated name of that version.
        aliases = [] if numbered else [m for m, k in members if k.alias]
        keep.update(aliases)
        rest = [] if aliases else members
        stable = [(m, k) for m, k in rest if not k.preview]
        previews = [(m, k) for m, k in rest if k.preview]
        best = max(stable, key=lambda mk: _rank(mk[1]), default=None)
        if best is not None:
            keep.add(best[0])
        newest = max(previews, key=lambda mk: _rank(mk[1]), default=None)
        if newest is not None and (best is None or newest[1].version > best[1].version):
            keep.add(newest[0])
    return [m for m in ids if m in keep]
