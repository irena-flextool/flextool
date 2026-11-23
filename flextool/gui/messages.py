"""The GUI message list's Tk-free core: message kinds, messages, the per-user store and the :class:`MessageCenter`.

Design: ``specs/gui_message_list_design.md`` §3 (package M1).  No Tk here: the window (``dialogs/
message_list_window.py``) and ``main_window.py`` drive a :class:`MessageCenter`; tests drive it headless.

Messages are computed, not stored
---------------------------------
A *source* is a callable ``source(store) -> Iterable[GuiMessage]`` (:data:`MessageSource`).  It builds each
message, texts included, from current facts every time :meth:`MessageCenter.recompute` runs: a state message
(W-0490, a file FlexTool cannot read) exists while its condition holds; an *event* message (a database was
upgraded) exists while the store has its ``pending`` entry (:meth:`MessageStore.pending`), which the source turns
into a message whose ``instance`` is the entry's ``run``.  Sources of database messages return messages only for
the input sources of the open project.  A source that raises is logged and skipped.

The store (``user_config_dir()/gui_messages.json``, one per user, never inside a database)
------------------------------------------------------------------------------------------------
``{"format": 1, "subjects": {<subject key>: {"label", "updated", "pending": {<spec id>: {"run", "posted",
"facts"}}, "dismissed": {"<id>|<instance>": date}, "shown": {"<id>|<instance>": n}}, "user": {..., "offered":
{"version", "checked"}}}}``.

- Subject keys: :func:`database_key` / :func:`project_key` (``"db:"`` / ``"project:"`` + ``normcase(abspath)``
  of the path the caller gives, never resolved here: on Windows that turns a mapped drive into its UNC path while the drive is online and leaves
  ``Z:\\...`` while it is offline, which would give one file two keys).  The same file reached through a drive
  letter and through its UNC path still has two keys: FlexTool keys the path its input source list holds (for a
  database outside the project folder that path is already resolved when it is added, so it is consistent).  On
  macOS ``normcase`` does nothing, so ``Model.sqlite`` and ``model.sqlite`` get two keys on its
  case-insensitive file system; that errs toward showing a message again and is accepted.
- Pruning (at each project open): a database or project subject goes when it was last written (``updated``) more
  than :data:`PRUNE_DAYS` days ago, or, only for subjects inside the opened project's folder, when its parent folder
  exists and the file (folder) does not.  Paths elsewhere are never probed (an offline network drive can block a
  file-system call for tens of seconds on Windows), so their entries go by age only.
- Writes are read-modify-write (merged with the file, so two FlexTool windows keep each other's entries), through
  a temporary file and ``os.replace`` (retried once after a ``PermissionError``, which Windows raises while another
  window reads the file).  A failed write keeps the change in memory and replays it at the next write; a change
  that raises (it cannot apply to the file's content) is logged and dropped.  There is no lock: two windows writing
  at the same moment can lose one dismissal or ``shown`` count (accepted).  A corrupt file counts as empty (one log
  warning); entries of the wrong type inside a known-format file are dropped at read (the rest is kept); a file of
  an unknown format is read as empty and never overwritten.
  Nothing here raises into the GUI except :meth:`MessageCenter.dismiss` of a non-dismissable message.

Auto-show (``recompute(trigger)``; design §3.4, decisions 18 and 20)
---------------------------------------------------------------------
- ``"project_open"`` / ``"refresh"``: open at the first active message that is not seen and not capped.
- ``"other"``: open only at an attention message that was not active at the previous recompute and is not seen.
- Capped: a message with ``attention=False`` (dismissable or not) whose ``shown`` count reached
  :data:`AUTO_SHOW_LIMIT`; it stays in the list, the count and the indicator.  Attention messages are never capped
  and have no store entry at all.  ``shown`` counts the project opens whose auto-shown window listed the message
  (:meth:`MessageCenter.window_closed` with ``opened_by="project_open"``).
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

from flextool.run_warnings.registry import DOCS_BASE_URL

_logger = logging.getLogger(__name__)

Scope = Literal["database", "project", "user"]
Trigger = Literal["project_open", "refresh", "other"]
STORE_NAME = "gui_messages.json"
STORE_FORMAT = 1
USER_KEY = "user"
AUTO_SHOW_LIMIT = 2
PRUNE_DAYS = 180
_SCOPE_RANK = {"database": 1, "project": 2, "user": 3}


@dataclass(frozen=True)
class MessageSpec:
    """One kind of message (:data:`MESSAGE_SPECS`)."""
    id: str
    scope: Scope
    dismissable: bool
    #: True: listed first, accent on the indicator, never capped; False: auto-show capped, dismissable or not.
    attention: bool
    #: ``"W-0490"`` (a coded warning's section) or ``"page.md#anchor"`` (a page of ``docs/``); the window opens it
    #: with ``message_box.docs_url``.
    docs: str


@dataclass(frozen=True)
class MessageAction:
    """One extra button of a message; the window calls ``run`` and stays open.  A text that ``run`` returns
    ("Copied") shows on the button for 2 s; *tooltip* is the button's hover text ("" for none)."""
    label: str
    run: Callable[[], str | None]
    tooltip: str = ""


@dataclass(frozen=True)
class GuiMessage:
    """One occurrence of a message, built by a source with its texts."""
    spec: MessageSpec
    subject_key: str
    subject_label: str
    title: str
    paragraphs: tuple[str, ...]
    #: What makes this occurrence new: the upgrade run, the offered version, a hash of a finding set.
    instance: str = ""
    action: MessageAction | None = None

    @property
    def key(self) -> str:
        return f"{self.spec.id}|{self.subject_key}|{self.instance}"

    @property
    def store_key(self) -> str:
        """The key under the subject in the store's ``dismissed`` / ``shown``."""
        return f"{self.spec.id}|{self.instance}"


#: The kinds of message, by id.  A later package may add one with :func:`register_spec`.
MESSAGE_SPECS: dict[str, MessageSpec] = {}


def register_spec(spec: MessageSpec) -> MessageSpec:
    if MESSAGE_SPECS.get(spec.id, spec) != spec:
        raise ValueError(f"message spec {spec.id!r} is already registered differently")
    MESSAGE_SPECS[spec.id] = spec
    return spec


_GUI_MIGRATION = "flextool_gui_interface.md#migration-and-frozen-databases"
for _spec in (
    MessageSpec("value_list_check", "database", dismissable=False, attention=True, docs="W-0490"),
    MessageSpec("list_defaults_lost", "database", dismissable=False, attention=False, docs="W-0491"),
    MessageSpec("database_unreadable", "database", dismissable=False, attention=True, docs=_GUI_MIGRATION),
    MessageSpec("database_upgraded", "database", dismissable=True, attention=False, docs=_GUI_MIGRATION),
    MessageSpec("solver_defaults_updated", "project", dismissable=True, attention=False,
                docs="flextool_gui_interface.md#solver-options-on-this-computer"),
    MessageSpec("update_available", "user", dismissable=True, attention=False,
                docs="install_gui.md#updating-flextool"),
):
    register_spec(_spec)


def docs_page_url(ref: str) -> str:
    """``"page.md#anchor"`` → ``<DOCS_BASE_URL>page/#anchor``; ``"index.md"`` → the site root."""
    page, _, anchor = ref.partition("#")
    page = page.removesuffix(".md")
    path = "" if page == "index" else page.removesuffix("/index") + "/"
    return DOCS_BASE_URL + path + (f"#{anchor}" if anchor else "")


def _path_key(prefix: str, path: str | os.PathLike, pathmod: Any) -> str:
    return prefix + pathmod.normcase(pathmod.abspath(os.fspath(path)))


def database_key(path: str | os.PathLike, *, pathmod: Any = os.path) -> str:
    """The store key of a database: ``"db:" + normcase(abspath(path))``, never resolved (module docstring)."""
    return _path_key("db:", path, pathmod)


def project_key(path: str | os.PathLike, *, pathmod: Any = os.path) -> str:
    """The store key of a project folder (as :func:`database_key`)."""
    return _path_key("project:", path, pathmod)


class MessageStore:
    """Pending events, dismissals, ``shown`` counts and the offered version, kept in one JSON file."""

    def __init__(self, path: Path, *, today: Callable[[], date] = date.today, pathmod: Any = os.path) -> None:
        self.path = Path(path)
        self._today = today
        self._pathmod = pathmod
        self._writable = True
        self._warned = False
        self._unsaved: list[Callable[[dict], None]] = []
        data = self._read()
        self._data: dict = data if data is not None else {}

    # -- reading -------------------------------------------------------------------------------------------------
    def _warn(self, text: str, *args: object) -> None:
        if not self._warned:
            self._warned = True
            _logger.warning(text, *args)

    def _read(self) -> dict | None:
        """The file's ``subjects``: empty when missing or corrupt; ``None`` when it cannot be read now (an
        ``OSError``) or has an unknown format (which also stops every write)."""
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except OSError as exc:
            self._warn("Message list: cannot read %s (%s).", self.path, exc)
            return None
        except ValueError as exc:
            self._warn("Message list: %s is damaged (%s); starting from an empty list.", self.path, exc)
            return {}
        if not isinstance(raw, dict) or raw.get("format") != STORE_FORMAT or not isinstance(raw.get("subjects"), dict):
            self._warn("Message list: %s has an unknown format; it is left as it is.", self.path)
            self._writable = False
            return None
        return {k: _clean_subject(v) for k, v in raw["subjects"].items() if isinstance(k, str) and isinstance(v, dict)}

    def _section(self, subject: str, name: str) -> dict:
        value = self._data.get(subject, {}).get(name, {})
        return value if isinstance(value, dict) else {}

    def pending(self, spec_id: str) -> dict[str, dict]:
        """``{subject key: entry}`` of the pending events of *spec_id* (an entry has ``run``, ``posted``, ``facts``
        and the subject's ``label``)."""
        out = {}
        for subject, entry in self._data.items():
            event = self._section(subject, "pending").get(spec_id)
            if isinstance(event, dict):
                out[subject] = {**event, "label": entry.get("label", "")}
        return out

    def is_dismissed(self, msg: GuiMessage) -> bool:
        """Whether *msg* was dismissed (always ``False`` for a non-dismissable kind, whatever the file holds)."""
        return msg.spec.dismissable and msg.store_key in self._section(msg.subject_key, "dismissed")

    def shown(self, msg: GuiMessage) -> int:
        value = self._section(msg.subject_key, "shown").get(msg.store_key, 0)
        return value if isinstance(value, int) else 0

    def offered_version(self) -> str | None:
        """The FlexTool version the last conclusive update check offered, if any."""
        version = self._section(USER_KEY, "offered").get("version")
        return version if isinstance(version, str) else None

    # -- writing -------------------------------------------------------------------------------------------------
    def _mutate(self, change: Callable[[dict], None]) -> None:
        """Apply *change* to the file's current content (merged) and write it; on failure keep it in memory and
        replay it at the next write.  When the file cannot be read now, only the in-memory state changes."""
        fresh = self._read() if self._writable else None
        if fresh is None:
            if _apply(change, self._data) and self._writable:
                self._unsaved.append(change)    # replayed over the file at the next successful read
            return
        self._unsaved = [c for c in (*self._unsaved, change) if _apply(c, fresh)]
        self._data = fresh
        if self._write(fresh):
            self._unsaved.clear()

    def _write(self, subjects: dict) -> bool:
        text = json.dumps({"format": STORE_FORMAT, "subjects": subjects}, indent=1, sort_keys=True)
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(text, encoding="utf-8")
            try:
                os.replace(tmp, self.path)
            except PermissionError:
                time.sleep(0.05)
                os.replace(tmp, self.path)
            return True
        except OSError as exc:
            self._warn("Message list: cannot save %s (%s); dismissals hold for this session only.", self.path, exc)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            return False

    def _subject(self, data: dict, subject: str, label: str | None = None) -> dict:
        entry = data.get(subject)
        if not isinstance(entry, dict):
            entry = data[subject] = {}
        for name in _SECTIONS:
            if not isinstance(entry.get(name, {}), dict):
                entry[name] = {}
        if label is not None:
            entry["label"] = label
        entry["updated"] = self._today().isoformat()
        return entry

    def post(self, spec_id: str, subject: str, label: str, *, run: str = "", facts: dict | None = None) -> None:
        """Record the event *spec_id* for *subject*, replacing an older one (and its ``shown`` count)."""
        def change(data: dict) -> None:
            entry = self._subject(data, subject, label)
            old = entry.setdefault("pending", {}).get(spec_id)
            if isinstance(old, dict):
                entry.get("shown", {}).pop(f"{spec_id}|{old.get('run', '')}", None)
            entry["pending"][spec_id] = {"run": run, "posted": self._today().isoformat(), "facts": facts or {}}
        self._mutate(change)

    def resolve(self, msg: GuiMessage) -> None:
        """Dismiss *msg*: remove its pending event (when it is that one) or record the dismissal; drop ``shown``."""
        def change(data: dict) -> None:
            entry = self._subject(data, msg.subject_key)
            pending = entry.get("pending", {})
            event = pending.get(msg.spec.id)
            if isinstance(event, dict):
                if event.get("run", "") == msg.instance:
                    del pending[msg.spec.id]
            else:
                entry.setdefault("dismissed", {})[msg.store_key] = self._today().isoformat()
            entry.get("shown", {}).pop(msg.store_key, None)
        self._mutate(change)

    def count_shown(self, msgs: Iterable[GuiMessage]) -> None:
        msgs = list(msgs)
        if not msgs:
            return

        def change(data: dict) -> None:
            for msg in msgs:
                shown = self._subject(data, msg.subject_key, msg.subject_label).setdefault("shown", {})
                old = shown.get(msg.store_key, 0)
                shown[msg.store_key] = (old if isinstance(old, int) else 0) + 1
        self._mutate(change)

    def set_offered_version(self, version: str | None) -> None:
        """Store the offered version (``None``: a conclusive check found nothing newer)."""
        def change(data: dict) -> None:
            entry = self._subject(data, USER_KEY, "")
            if version is None:
                entry.pop("offered", None)
            else:
                entry["offered"] = {"version": version, "checked": self._today().isoformat()}
        self._mutate(change)

    def rename_prefix(self, old: str | os.PathLike, new: str | os.PathLike) -> None:
        """Move every subject under the folder *old* (a renamed project) to *new*."""
        pm = self._pathmod
        old_path = pm.normcase(pm.abspath(os.fspath(old)))
        new_abs = pm.abspath(os.fspath(new))
        new_path = pm.normcase(new_abs)
        new_label = pm.basename(new_abs.rstrip(pm.sep))       # the name as typed: normcase lowercases on Windows

        def change(data: dict) -> None:
            for subject in list(data):
                prefix, sep, path = subject.partition(":")
                if not sep or prefix not in ("db", "project"):
                    continue
                if path == old_path or path.startswith(old_path.rstrip(pm.sep) + pm.sep):
                    entry = data.pop(subject)
                    if prefix == "project" and path == old_path:
                        entry["label"] = new_label
                    data[f"{prefix}:{new_path}{path[len(old_path):]}"] = entry
        self._mutate(change)

    def prune(self, within: str | os.PathLike | None = None) -> None:
        """Drop the subjects not written for :data:`PRUNE_DAYS` days, and the subjects of deleted files inside the
        folder *within* (the opened project; ``None``: none is probed) (module docstring)."""
        pm = self._pathmod
        oldest = (self._today() - timedelta(days=PRUNE_DAYS)).isoformat()
        root = pm.normcase(pm.abspath(os.fspath(within))).rstrip(pm.sep) if within is not None else None

        def gone(subject: str, entry: dict) -> bool:
            prefix, sep, path = subject.partition(":")
            if not sep or prefix not in ("db", "project"):
                return False
            if str(entry.get("updated", "9999")) < oldest:
                return True
            if root is None or not (path == root or path.startswith(root + pm.sep)):
                return False
            parent = pm.dirname(path.rstrip(pm.sep))
            try:
                return pm.isdir(parent) and not pm.exists(path)
            except OSError:
                return False

        stale = [s for s, e in self._data.items() if gone(s, e)]
        if stale:
            self._mutate(lambda data: [data.pop(s, None) for s in stale])


_SECTIONS = ("pending", "dismissed", "shown", "offered")


def _clean_subject(entry: dict) -> dict:
    """*entry* with its sections of the wrong type dropped (a hand edit, a sync conflict): a section must be a dict,
    a pending event a dict, a ``shown`` count an int."""
    clean = {k: v for k, v in entry.items() if k not in _SECTIONS or isinstance(v, dict)}
    if "pending" in clean:
        clean["pending"] = {k: v for k, v in clean["pending"].items() if isinstance(v, dict)}
    if "shown" in clean:
        clean["shown"] = {k: v for k, v in clean["shown"].items() if isinstance(v, int) and not isinstance(v, bool)}
    return clean


def _apply(change: Callable[[dict], None], data: dict) -> bool:
    """Apply *change* to *data*; a change that raises is logged and dropped (``False``), never raised to the GUI."""
    try:
        change(data)
    except Exception:
        _logger.exception("Message list: a change to the stored messages failed and is dropped")
        return False
    return True


#: A source of messages (module docstring).
MessageSource = Callable[[MessageStore], Iterable[GuiMessage]]


class MessageCenter:
    """The message list of one open project (a new one at each project open / GUI start, so nothing is seen).

    *rank* gives a subject key's order within its scope (the input-source number of a database); messages of equal
    rank keep their source order.  The window lists :attr:`messages` and calls :meth:`window_closed`.
    """

    def __init__(self, store_path: Path, sources: Iterable[MessageSource], *,
                 rank: Callable[[str], int] | None = None, today: Callable[[], date] = date.today,
                 pathmod: Any = os.path, project_dir: str | os.PathLike | None = None) -> None:
        self.store = MessageStore(store_path, today=today, pathmod=pathmod)
        self.store.prune(within=project_dir)
        self.sources = list(sources)
        self._rank = rank or (lambda _key: 0)
        self.seen: set[str] = set()
        self.messages: tuple[GuiMessage, ...] = ()
        self._previous: set[str] = set()
        self._counted: set[str] = set()

    def _collect(self) -> list[GuiMessage]:
        found: dict[str, GuiMessage] = {}
        for source in self.sources:
            try:
                for msg in source(self.store):
                    if msg.key not in found and not self.store.is_dismissed(msg):
                        found[msg.key] = msg
            except Exception:       # a broken source must never break the GUI
                _logger.exception("Message list: source %r failed", source)
        return sorted(found.values(), key=lambda m: (0 if m.spec.attention else _SCOPE_RANK[m.spec.scope],
                                                     self._rank(m.subject_key)))

    def is_capped(self, msg: GuiMessage) -> bool:
        return not msg.spec.attention and self.store.shown(msg) >= AUTO_SHOW_LIMIT

    def recompute(self, trigger: Trigger = "other") -> str | None:
        """Rebuild :attr:`messages`; return the key the window should open at, or ``None`` (only the indicator
        changes).  An open window updates in place whatever this returns."""
        msgs = self._collect()
        previous, self._previous = self._previous, {m.key for m in msgs}
        self.messages = tuple(msgs)
        for msg in msgs:
            if msg.key in self.seen:
                continue
            if trigger in ("project_open", "refresh"):
                if not self.is_capped(msg):
                    return msg.key
            elif msg.spec.attention and msg.key not in previous:
                return msg.key
        return None

    def indicator(self) -> tuple[int, bool]:
        """``(count, accent)``: accent while an attention message is active or a message is unseen."""
        accent = any(m.spec.attention or m.key not in self.seen for m in self.messages)
        return len(self.messages), accent

    def window_closed(self, listed: Iterable[str], *, opened_by: Trigger | None = None) -> None:
        """The window listed the messages *listed* (keys) and closed, however it was closed: they are seen; when the
        project-open auto-show opened it, each non-attention one counts one more ``shown`` (once per center)."""
        listed = set(listed)
        self.seen |= listed
        if opened_by == "project_open":
            fresh = [m for m in self.messages
                     if m.key in listed and not m.spec.attention and m.key not in self._counted]
            self._counted |= {m.key for m in fresh}
            self.store.count_shown(fresh)

    def dismiss(self, msg: GuiMessage) -> None:
        """Dismiss *msg* (raises ``ValueError`` for a non-dismissable message: a programming error)."""
        if not msg.spec.dismissable:
            raise ValueError(f"message {msg.spec.id!r} cannot be dismissed")
        self.store.resolve(msg)
        self.messages = tuple(m for m in self.messages if m.key != msg.key)
        self._previous.discard(msg.key)

    def post_event(self, spec_id: str, subject_key: str, label: str, *, run: str = "",
                   facts: dict | None = None) -> None:
        """Record an event (the caller then calls :meth:`recompute`, once after several posts)."""
        if spec_id not in MESSAGE_SPECS or not MESSAGE_SPECS[spec_id].dismissable:
            raise ValueError(f"{spec_id!r} is not a dismissable message kind")
        self.store.post(spec_id, subject_key, label, run=run, facts=facts)

    def rename_prefix(self, old: str | os.PathLike, new: str | os.PathLike) -> None:
        self.store.rename_prefix(old, new)
