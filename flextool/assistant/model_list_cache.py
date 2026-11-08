"""The last model list fetched in the assistant settings, per preset and address (owner feedback 2026-11-08).

The settings dialog shows the saved list at once, with the date it was fetched, instead of an empty list until
[Refresh list] is pressed again.  The file is ``<user cache>/flextool/assistant/model_lists.json`` (UTF-8, written
atomically under a lock), keyed by ``<preset>|<address>`` with the address in :func:`net.normalise_endpoint`'s
spelling.  It holds the model ids and the fetch time only: no key, no header, nothing else of the request (an
address with a user name, password or query is refused before any fetch, :func:`net.check_base_url`).

Reading is one small local file; fetching the list stays on the dialog's background thread, which also writes the
file.  A file that cannot be read or written only loses the saved lists: they are fetched again.

The file also keeps the models a test (:mod:`flextool.assistant.model_check`) found **not available to your
account** (``"unavailable"``: by list key, model id → the time of the test).  A marked model is left out of the
default list and marked under "Show all models".  The mark goes when the list is fetched again (:func:`remember`),
when the key of the provider and address changes (:func:`clear_marks`), when a test of the model succeeds, and
after :data:`MARK_MAX_AGE`.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from flextool.assistant import _files, net

logger = logging.getLogger(__name__)

FILE = "model_lists.json"
FORMAT = 1
#: How long a "not available to your account" mark holds.
MARK_MAX_AGE = 7 * 24 * 3600.0


@dataclass(frozen=True)
class SavedList:
    ids: tuple[str, ...]
    fetched: float               # seconds since the epoch


def _root() -> Path:
    from flextool.assistant.probes import cache_root

    return cache_root()


def key(preset_id: str, base_url: str) -> str:
    return f"{preset_id}|{net.normalise_endpoint(base_url)}"


def load_all(root: Path | None = None) -> dict[str, SavedList]:
    """Every saved list, by :func:`key`; ``{}`` when there is none or the file cannot be read."""
    path = (root or _root()) / FILE
    try:
        data = _files.read_json(path, check_format=False) or {}
    except (OSError, _files.UnreadableFile):
        logger.info("The saved model lists could not be read; they are fetched again", exc_info=True)
        return {}
    lists = data.get("lists")
    out: dict[str, SavedList] = {}
    if not isinstance(lists, dict):
        return out
    for name, entry in lists.items():
        if not isinstance(entry, dict):
            continue
        ids, fetched = entry.get("ids"), entry.get("fetched")
        if (isinstance(ids, list) and all(isinstance(i, str) for i in ids)
                and isinstance(fetched, (int, float)) and not isinstance(fetched, bool)):
            out[str(name)] = SavedList(tuple(ids), float(fetched))
    return out


def lookup(lists: dict[str, SavedList], preset_id: str, base_url: str | None) -> SavedList | None:
    return lists.get(key(preset_id, base_url)) if base_url else None


def _edit(root: Path | None, change: Callable[[dict], None]) -> None:
    """Read the file, *change* its data in place and write it back, under the lock; a damaged file is written over.
    Raises what the lock or the write raises."""
    path = (root or _root()) / FILE
    with _files.locked(path):
        try:
            data = _files.read_json(path, check_format=False) or {}
            damaged = False
        except _files.UnreadableFile:
            data, damaged = {}, True
        if not isinstance(data.get("lists"), dict):
            data["lists"] = {}
        if not isinstance(data.get("unavailable"), dict):
            data["unavailable"] = {}
        change(data)
        new = {"format": FORMAT, "lists": data["lists"], "unavailable": data["unavailable"]}
        if damaged:                     # nothing of it can be kept: written over (no format to respect)
            _files.write_text_atomic(path, _files.dumps(new))
        else:
            _files.write_json_atomic(path, new)


def remember(preset_id: str, base_url: str, ids: list[str], *, now: float | None = None,
             root: Path | None = None) -> SavedList:
    """Save *ids* as the list of *preset_id* at *base_url* (call off the Tk thread); the models marked not available
    for it are marked no more.  Returns what was saved, also when the file could not be written."""
    saved = SavedList(tuple(ids), time.time() if now is None else now)
    name = key(preset_id, base_url)

    def change(data: dict) -> None:
        data["lists"][name] = {"fetched": saved.fetched, "ids": list(saved.ids)}
        data["unavailable"].pop(name, None)

    try:
        _edit(root, change)
    except (OSError, _files.LockTimeout, _files.NewerFormat):
        logger.info("The model list could not be saved; it is fetched again next time", exc_info=True)
    return saved


# --- models not available to the account ----------------------------------------------------------------------
Marks = dict[str, dict[str, float]]


def load_marks(root: Path | None = None) -> Marks:
    """Every "not available to your account" mark, by :func:`key` then model id (the time of the test)."""
    path = (root or _root()) / FILE
    try:
        data = _files.read_json(path, check_format=False) or {}
    except (OSError, _files.UnreadableFile):
        return {}
    marks = data.get("unavailable")
    out: Marks = {}
    if not isinstance(marks, dict):
        return out
    for name, models in marks.items():
        if isinstance(models, dict):
            out[str(name)] = {str(m): float(t) for m, t in models.items()
                              if isinstance(t, (int, float)) and not isinstance(t, bool)}
    return out


def unavailable(marks: Marks, preset_id: str, base_url: str | None, *, now: float | None = None) -> set[str]:
    """The models of *preset_id* at *base_url* marked not available within :data:`MARK_MAX_AGE`."""
    if not base_url:
        return set()
    now = time.time() if now is None else now
    return {m for m, t in marks.get(key(preset_id, base_url), {}).items() if 0 <= now - t <= MARK_MAX_AGE}


def mark_unavailable(preset_id: str, base_url: str, model: str, *, now: float | None = None,
                     root: Path | None = None) -> None:
    """Mark *model* not available to the account (call off the Tk thread)."""
    when = time.time() if now is None else now
    name = key(preset_id, base_url)

    def change(data: dict) -> None:
        models = data["unavailable"].get(name)
        models = models if isinstance(models, dict) else {}
        models[model] = when
        data["unavailable"][name] = models

    try:
        _edit(root, change)
    except (OSError, _files.LockTimeout, _files.NewerFormat):
        logger.info("The model could not be marked as not available", exc_info=True)


def clear_marks(preset_id: str, base_url: str, model: str | None = None, *, root: Path | None = None) -> None:
    """Remove the mark of *model*, or every mark of the provider and address (*model* ``None``: the key changed).
    Call off the Tk thread; nothing is written when there is nothing to remove."""
    name = key(preset_id, base_url)
    marked = load_marks(root).get(name, {})
    if not marked or (model is not None and model not in marked):
        return

    def change(data: dict) -> None:
        models = data["unavailable"].get(name)
        if model is None or not isinstance(models, dict):
            data["unavailable"].pop(name, None)
            return
        models.pop(model, None)
        if not models:
            data["unavailable"].pop(name, None)

    try:
        _edit(root, change)
    except (OSError, _files.LockTimeout, _files.NewerFormat):
        logger.info("The mark of the model could not be removed", exc_info=True)
