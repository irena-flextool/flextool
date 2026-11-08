"""The newest stable version of each model family in a service's model list (owner decisions 2026-11-08).

A service lists every version it still answers for, also ones closed to new accounts (Gemini listed
``gemini-2.5-flash`` after it stopped serving it to new users), and many pre-releases.  The settings dialog and
``--list-models`` show by default only stable releases, and of those only the newest version of each family; "Show
all models" / ``--all`` show the whole list.  Presets name no models (P1-O1): the families come from the ids the
service returns, never from a list of model names.

:func:`model_key` splits an id into its family and version:

* the ``models/`` prefix is dropped; a vendor path (``google/`` on OpenRouter) and a ``:suffix`` (``:free``)
  belong to the family;
* the first run of short numbers is the version (``2.5``, ``3``, ``3-1``, ``4-5``, ``v3.1``);
* numbers after it — dates (``20251001``, ``2025-08-07``, ``09-2025``), snapshots (``001``, ``2411``) — are the
  snapshot, a lower-priority part of the version;
* a pre-release word marks a pre-release: ``preview`` (Gemini ``-preview-05-06``, OpenAI ``-preview``), ``exp`` /
  ``experimental`` (Gemini ``-exp-1206``, OpenRouter ``-exp:free``), ``alpha``, ``beta``, ``rc``, ``pre``,
  ``dev``, ``nightly``, ``canary``, also with a number (``rc1``, ``beta2``);
* ``latest`` marks a moving alias (``gemini-flash-latest``, ``mistral-large-latest``);
* every other word stays in the family, in order (``gemini-flash``, ``gemini-flash-lite``, ``claude-haiku``,
  ``gpt-mini``, ``mistral-large``, ``llama-70b-instruct``);
* an id without a version number is never compared with numbered ones of the same words: it is a family of its
  own, together with its dated snapshots (``gpt-4o`` and ``gpt-4o-2024-08-06``).

:func:`latest_ids` leaves out every pre-release and every ``latest`` alias (its name does not say whether it points
to a stable model), and keeps in each family the newest stable release: an undated name before the dated snapshots
of its version, else the newest snapshot (``mistral-large-2411`` over ``-2407``).  A family with only pre-releases
is not shown.  :func:`hidden_kind` says why an id is not shown.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

_VERSION = re.compile(r"v?\d{1,2}(?:\.\d+)*", re.IGNORECASE)
_NUMBER = re.compile(r"v?\d+(?:\.\d+)*", re.IGNORECASE)
_YEAR = re.compile(r"(?:19|20)\d\d")
_PREVIEW = re.compile(r"(?:preview|exp|experimental|alpha|beta|rc|pre|dev|nightly|canary)\d*")
_ALIAS = frozenset({"latest"})


@dataclass(frozen=True)
class ModelKey:
    family: str
    version: tuple[int, ...]        # () when the id has no version number
    snapshot: tuple[int, ...]       # () for an undated id
    preview: bool                   # a pre-release
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
        marker = bool(_PREVIEW.fullmatch(low))
        if marker or not _NUMBER.fullmatch(tok):
            preview = preview or marker
            if not marker:
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
    """The ids of *ids* that are the newest stable release of their family, in the order given."""
    ids = list(ids)
    families: dict[tuple[str, bool], list[tuple[str, ModelKey]]] = {}
    for model in ids:
        key = model_key(model)
        if key.preview or key.alias:
            continue
        # An id without a version number is never compared with numbered ones (``assistant-model`` is not an
        # older ``assistant-model-2``); it is the undated name of its dated snapshots (``gpt-4o``).
        families.setdefault((key.family, bool(key.version)), []).append((model, key))
    keep = {max(members, key=lambda mk: _rank(mk[1]))[0] for members in families.values()}
    return [m for m in ids if m in keep]


def hidden_kind(model: str) -> str:
    """Why *model*, when :func:`latest_ids` leaves it out, is not shown: ``"pre-release"``, ``"alias"`` (a moving
    ``latest`` name) or ``"older"`` (a stable release with a newer one in its family)."""
    key = model_key(model)
    return "pre-release" if key.preview else "alias" if key.alias else "older"
