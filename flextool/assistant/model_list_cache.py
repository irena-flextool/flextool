"""The last model list fetched in the assistant settings, per preset and address (owner feedback 2026-11-08).

The settings dialog shows the saved list at once, with the date it was fetched, instead of an empty list until
[Refresh list] is pressed again.  The file is ``<user cache>/flextool/assistant/model_lists.json`` (UTF-8, written
atomically under a lock), keyed by ``<preset>|<address>`` with the address in :func:`net.normalise_endpoint`'s
spelling.  It holds the model ids and the fetch time only: no key, no header, nothing else of the request (an
address with a user name, password or query is refused before any fetch, :func:`net.check_base_url`).

Reading is one small local file; fetching the list stays on the dialog's background thread, which also writes the
file.  A file that cannot be read or written only loses the saved lists: they are fetched again.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from flextool.assistant import _files, net

logger = logging.getLogger(__name__)

FILE = "model_lists.json"
FORMAT = 1


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


def remember(preset_id: str, base_url: str, ids: list[str], *, now: float | None = None,
             root: Path | None = None) -> SavedList:
    """Save *ids* as the list of *preset_id* at *base_url* (call off the Tk thread).  Returns what was saved,
    also when the file could not be written."""
    saved = SavedList(tuple(ids), time.time() if now is None else now)
    path = (root or _root()) / FILE
    try:
        with _files.locked(path):
            try:
                data = _files.read_json(path, check_format=False) or {}
                damaged = False
            except _files.UnreadableFile:
                data, damaged = {}, True
            lists = data.get("lists") if isinstance(data.get("lists"), dict) else {}
            lists[key(preset_id, base_url)] = {"fetched": saved.fetched, "ids": list(saved.ids)}
            new = {"format": FORMAT, "lists": lists}
            if damaged:                     # nothing of it can be kept: written over (no format to respect)
                _files.write_text_atomic(path, _files.dumps(new))
            else:
                _files.write_json_atomic(path, new)
    except (OSError, _files.LockTimeout, _files.NewerFormat):
        logger.info("The model list could not be saved; it is fetched again next time", exc_info=True)
    return saved
