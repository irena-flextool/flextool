"""Saved conversations, kept per project and deleted after the expiry (P1 §10.2, D3, D14).

Layout under ``<config>/assistant/``::

    projects.json                               project key -> {path, name, last_used, next_chat_number, open_chats}
    conversations/<project key>/<id>.json       one conversation, rewritten after each round
    conversations/<project key>/<id>.json.lock  the process that has the conversation open
    conversations/<project key>/<id>.title.json the name the user gave the chat (written at once, on its own)

An id is the UTC creation time ``YYYYMMDDTHHMMSSZ`` + ``-`` + 6 random hex digits.  A conversation open in a
session holds its lock file until :meth:`Store.close` (or the process ends: a lock of a gone process is stale), so
two windows or terminals never write one conversation.  In this process every open conversation is also in a
registry with a ``save_lock`` and a ``deleted`` mark: :meth:`Store.delete_conversation` marks it under that lock
before it removes the files, so a save that finishes later writes nothing.  Every change of ``projects.json``
runs under its lock (``_files.locked``).

:meth:`Store.purge_expired` runs at the start of every chat, of the agent connection and of each tool call from the
command line, and when FlexTool closes: it deletes the conversations not changed within the expiry (age from the
id's time and the file's modification time; no file is parsed), their title files, title files without a
conversation, temp files of writers that are gone and stale lock files; then it removes empty project folders and
the rows of projects without conversations that were not used within the expiry.  Conversations open in a running
session are never touched.  Expiry 0: nothing is written and everything not open is deleted; the usage ledger is
kept (D3).

Imports only the standard library, :mod:`._files`, :mod:`.settings` and the neutral message model.
"""
from __future__ import annotations

import contextlib
import os
import re
import secrets
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from flextool.assistant import _files, settings
from flextool.assistant.llm_adapter import Message, Origin, ToolCall

CONVERSATIONS = "conversations"
PROJECTS_FILE = "projects.json"
LOCKED_NEW = "This conversation is open in another FlexTool window or terminal; starting a new one."
FIRST_QUESTION_CHARS = 80
TITLE_CHARS = 60
#: A cleanup (purge, delete, rename) holds a conversation's lock for moments; an opener waits this long for it.
CLEANUP_WAIT = 2.0

_ID = re.compile(r"^(\d{8}T\d{6}Z)-[0-9a-f]{6}$")
_ID_TIME = "%Y%m%dT%H%M%SZ"
_CONTEXT_BLOCK = re.compile(r"^\s*\[FlexTool context[^\]]*\].*?\[end of context\]\s*", re.DOTALL)

LockedBy = Literal["this_process", "other_process"]


class StoreError(Exception):
    """A conversation that cannot be opened, with the message for the user."""


class ConversationNotFound(StoreError):
    def __init__(self, conversation_id: str | None) -> None:
        super().__init__("This project has no saved conversations." if conversation_id is None
                         else f"No saved conversation {conversation_id} in this project.")
        self.conversation_id = conversation_id


class ConversationLocked(StoreError):
    """The conversation is open elsewhere: in this process (another chat tab, or a closed chat still finishing)
    or in another process."""

    def __init__(self, owner: LockedBy) -> None:
        super().__init__("This conversation is open in another FlexTool window or terminal.")
        self.owner = owner


class ConversationDamaged(StoreError):
    pass


# --- the in-memory conversation -----------------------------------------------------------------------------------
def new_id(now: datetime | None = None) -> str:
    return (now or _files.utc_now()).astimezone(timezone.utc).strftime(_ID_TIME) + "-" + secrets.token_hex(3)


def id_time(conversation_id: str) -> datetime | None:
    match = _ID.match(conversation_id)
    if match is None:
        return None
    return datetime.strptime(match.group(1), _ID_TIME).replace(tzinfo=timezone.utc)


def is_id(text: str) -> bool:
    return isinstance(text, str) and _ID.match(text) is not None


def _check_id(conversation_id: str) -> str:
    if not is_id(conversation_id):
        raise ConversationNotFound(conversation_id)
    return conversation_id


def _empty_usage() -> dict[str, Any]:
    from flextool.assistant.usage import empty_counts

    counts = empty_counts()
    counts.pop("rejected_attempts", None)
    return counts


@dataclass(eq=False)
class Conversation:
    """One conversation.  ``save_lock`` serialises :meth:`Store.save` with :meth:`Store.delete_conversation`
    and :meth:`Store.close`; ``deleted`` and ``closed`` make every later save do nothing."""

    id: str
    project_key: str
    project_name: str
    project_path: Path
    created: str
    updated: str
    origin: Origin | None
    tier: str | None
    system_hash: str = ""
    tools_hash: str = ""
    messages: list[Message] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=_empty_usage)
    number: int | None = None
    title: str | None = None
    read_only: bool = False             # written by a newer FlexTool: never saved
    notice: str | None = None           # said to the user when it was opened (a new one instead of a locked one)
    deleted: bool = False
    closed: bool = False
    save_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _lock: _files.PidLock | None = field(default=None, repr=False)
    _title_pending: bool = field(default=False, repr=False)

    def to_json(self) -> dict[str, Any]:
        return {
            "format": _files.FORMAT, "id": self.id, "number": self.number,
            "project": {"key": self.project_key, "name": self.project_name},
            "created": self.created, "updated": self.updated,
            "origin": list(self.origin) if self.origin else None, "tier": self.tier,
            "system_hash": self.system_hash, "tools_hash": self.tools_hash,
            "messages": [message_to_json(m) for m in self.messages],
            "usage": dict(self.usage),
        }


def _origin(value: object) -> Origin | None:
    if isinstance(value, (list, tuple)) and len(value) == 3 and all(isinstance(v, str) for v in value):
        return (value[0], value[1], value[2])
    return None


def message_to_json(m: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"role": m.role, "text": m.text}
    if m.tool_calls:
        out["tool_calls"] = [{"id": c.id, "name": c.name, "arguments": c.arguments, "raw_arguments": c.raw_arguments,
                              **({"provider_extra": c.provider_extra} if c.provider_extra else {})}
                             for c in m.tool_calls]
    if m.tool_call_id is not None:
        out["tool_call_id"] = m.tool_call_id
    if m.tool_name is not None:
        out["tool_name"] = m.tool_name
    if m.is_error:
        out["is_error"] = True
    if m.provider_extra:
        out["provider_extra"] = m.provider_extra
    if m.origin is not None:
        out["origin"] = list(m.origin)
    out["turn"] = m.turn
    if m.compacted:
        out["compacted"] = True
    if m.kind:
        out["kind"] = m.kind
    return out


def message_from_json(d: dict[str, Any]) -> Message:
    role = d.get("role")
    if role not in ("user", "assistant", "tool"):
        raise ValueError(f"a message has the role {role!r}")
    calls = [ToolCall(id=str(c["id"]), name=str(c["name"]),
                      arguments=c.get("arguments") if isinstance(c.get("arguments"), dict) else None,
                      raw_arguments=str(c.get("raw_arguments", "")), provider_extra=dict(c.get("provider_extra") or {}))
             for c in d.get("tool_calls") or []]
    return Message(role=role, text=str(d.get("text", "")), tool_calls=calls, tool_call_id=d.get("tool_call_id"),
                   tool_name=d.get("tool_name"), is_error=bool(d.get("is_error", False)),
                   provider_extra=dict(d.get("provider_extra") or {}), origin=_origin(d.get("origin")),
                   turn=int(d.get("turn", 0)), compacted=bool(d.get("compacted", False)),
                   kind=str(d.get("kind") or ""))


def first_question(messages: list[Message] | list[dict[str, Any]]) -> str:
    """The first user question without the context block, on one line, at most 80 characters."""
    for m in messages:
        role, text = (m.role, m.text) if isinstance(m, Message) else (m.get("role"), m.get("text", ""))
        if role == "user":
            text = " ".join(_CONTEXT_BLOCK.sub("", str(text), count=1).split())
            return text if len(text) <= FIRST_QUESTION_CHARS else text[:FIRST_QUESTION_CHARS - 1].rstrip() + "…"
    return ""


@dataclass(frozen=True)
class ConversationInfo:
    id: str
    number: int | None
    title: str              # the name the user gave, else the first question
    named: bool             # the title is a name the user gave
    first_question: str
    created: str
    updated: str
    origin: Origin | None
    locked: LockedBy | None


# --- the registry of conversations open in this process -----------------------------------------------------------
_OPEN: dict[tuple[str, str, str], Conversation] = {}
_OPEN_GUARD = threading.RLock()
# Chat numbers counted in memory when nothing is saved (expiry 0), per (root, project key).
_MEMORY_NUMBERS: dict[tuple[str, str], int] = {}


def _warn(text: str) -> None:
    print(f"Warning: {text}", file=sys.stderr)


class Store:
    """The conversations under *root* (``<config>/assistant``)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._root_key = os.path.normcase(str(self.root.absolute()))

    # --- paths ------------------------------------------------------------------------------------------------
    @property
    def projects_path(self) -> Path:
        return self.root / PROJECTS_FILE

    @property
    def conversations_dir(self) -> Path:
        return self.root / CONVERSATIONS

    def project_dir(self, key: str) -> Path:
        return self.conversations_dir / key

    def _paths(self, key: str, conversation_id: str) -> tuple[Path, Path, Path]:
        folder = self.project_dir(key)
        return (folder / f"{conversation_id}.json", folder / f"{conversation_id}.json.lock",
                folder / f"{conversation_id}.title.json")

    def _lock(self, key: str, conversation_id: str, purpose: str = "owner") -> _files.PidLock:
        return _files.PidLock(self._paths(key, conversation_id)[1], purpose=purpose)

    def _registered(self, key: str, conversation_id: str) -> Conversation | None:
        with _OPEN_GUARD:
            return _OPEN.get((self._root_key, key, conversation_id))

    def _register(self, conv: Conversation) -> None:
        with _OPEN_GUARD:
            _OPEN[(self._root_key, conv.project_key, conv.id)] = conv

    def _unregister(self, conv: Conversation) -> None:
        with _OPEN_GUARD:
            if _OPEN.get((self._root_key, conv.project_key, conv.id)) is conv:
                del _OPEN[(self._root_key, conv.project_key, conv.id)]

    def expiry_days(self) -> int:
        return settings.expiry_days(self.root)

    # --- projects.json -----------------------------------------------------------------------------------------
    def _read_projects(self) -> dict[str, Any]:
        try:
            data = _files.read_json(self.projects_path)
        except (OSError, _files.UnreadableFile):
            return {}
        rows = (data or {}).get("projects")
        return rows if isinstance(rows, dict) else {}

    def _change_projects(self, change) -> Any:
        """Run ``change(rows)`` on the rows of ``projects.json`` under its lock and write them when it returns
        something other than ``None``-with-no-change; returns what *change* returned."""
        path = self.projects_path
        with _files.locked(path):
            try:
                data = _files.read_json(path) or {"format": _files.FORMAT, "projects": {}}
            except _files.UnreadableFile as exc:
                aside = path.with_name(f"{path.name}.damaged-{int(time.time())}")
                path.replace(aside)
                _warn(f"{exc} It was kept as {aside.name}.")
                data = {"format": _files.FORMAT, "projects": {}}
            newer = _files.is_newer(data)
            rows = data.get("projects")
            if not isinstance(rows, dict):
                rows = data["projects"] = {}
            before = _files.dumps(data)
            result = change(rows)
            if _files.dumps(data) != before:
                if newer:
                    raise _files.NewerFormat(path, data.get("format"))
                _files.write_json_atomic(path, data)
            return result

    def _touch_project(self, path: Path, name: str, key: str) -> None:
        """Record the project's row (path, name, last use); a busy or newer file only gets a warning."""
        def change(rows: dict[str, Any]) -> None:
            row = rows.setdefault(key, {})
            row.update({"path": str(path), "name": name, "last_used": _files.iso(_files.utc_now())})
        try:
            self._change_projects(change)
        except (OSError, _files.LockTimeout, _files.NewerFormat) as exc:
            _warn(f"the list of projects with saved conversations could not be updated: {exc}")

    @staticmethod
    def _project(project: Any) -> tuple[Path, str, str]:
        """(resolved path, display name, key) of a project given as a path or a context with ``project_path``."""
        raw = getattr(project, "project_path", project)
        path = Path(raw).resolve()
        name = getattr(project, "project_name", None) or path.name
        return path, name, _files.project_key(path)

    # --- conversations ------------------------------------------------------------------------------------------
    def new_conversation(self, project: Any, origin: Origin | None, tier: str | None, *, system_hash: str = "",
                         tools_hash: str = "", number: int | None = None, notice: str | None = None
                         ) -> Conversation:
        """A new, empty conversation of *project* (a path or an ``AssistantContext``), open in this process.
        Nothing is written until the first :meth:`save`, except the lock (and the project's row)."""
        path, name, key = self._project(project)
        now = _files.iso(_files.utc_now())
        saving = self.expiry_days() > 0
        while True:
            conversation_id = new_id()
            lock = self._lock(key, conversation_id) if saving else None
            if self._registered(key, conversation_id) is None and not self._paths(key, conversation_id)[0].exists():
                if lock is None or lock.try_acquire():
                    break
        conv = Conversation(id=conversation_id, project_key=key, project_name=name, project_path=path, created=now,
                            updated=now, origin=origin, tier=tier, system_hash=system_hash, tools_hash=tools_hash,
                            number=number, notice=notice, _lock=lock)
        self._register(conv)
        if saving:
            # After the lock file exists (the folder is not empty), so a purge in between keeps the row.
            self._touch_project(path, name, key)
        return conv

    def _conversation_files(self, key: str) -> list[tuple[str, Path]]:
        folder = self.project_dir(key)
        try:
            names = os.listdir(folder)
        except (FileNotFoundError, NotADirectoryError):
            return []
        out = []
        for entry in names:
            if entry.endswith(".json") and not entry.endswith(".title.json") and is_id(entry[:-5]):
                out.append((entry[:-5], folder / entry))
        return out

    @staticmethod
    def _last_change(conversation_id: str, path: Path) -> float:
        times = []
        created = id_time(conversation_id)
        if created is not None:
            times.append(created.timestamp())
        with contextlib.suppress(OSError):
            times.append(path.stat().st_mtime)
        return max(times) if times else 0.0

    def _newest_id(self, key: str) -> str | None:
        files = self._conversation_files(key)
        if not files:
            return None
        return max(files, key=lambda item: (self._last_change(*item), item[0]))[0]

    def _locked_by(self, key: str, conversation_id: str) -> LockedBy | None:
        conv = self._registered(key, conversation_id)
        if conv is not None and not conv.closed:
            return "this_process"
        lock = self._lock(key, conversation_id)
        owner, raw = _files.read_owner(lock.path)
        if raw is None or lock.is_stale(owner, raw):
            return None
        if owner is not None and owner.this_process:
            return "this_process"
        return "other_process"

    def _acquire_for_open(self, key: str, conversation_id: str) -> _files.PidLock | LockedBy:
        """The conversation's lock for this process, or who holds it.  A cleanup's lock is waited for."""
        deadline = time.monotonic() + CLEANUP_WAIT
        while True:
            lock = self._lock(key, conversation_id)
            if lock.try_acquire():
                return lock
            owner = lock.owner()
            if owner is not None and owner.purpose != "owner" and time.monotonic() < deadline:
                time.sleep(0.02)
                continue
            if owner is not None and owner.this_process:
                return "this_process"
            if owner is None and lock.path.exists() and time.monotonic() < deadline:
                time.sleep(0.02)        # being written this moment
                continue
            if owner is None and not lock.path.exists() and time.monotonic() < deadline:
                continue                # released in between: try again
            return "other_process"

    def open(self, project_path: Any, conversation_id: str | None = None, *,
             if_locked: Literal["new", "raise"] = "new") -> Conversation:
        """The saved conversation *conversation_id* of the project (the newest when ``None``), open in this
        process.  Open elsewhere: with ``if_locked="new"`` a new conversation with :data:`LOCKED_NEW` as its
        ``notice``; with ``"raise"`` :class:`ConversationLocked`.  Raises :class:`ConversationNotFound` when there
        is no such conversation (or none at all), and :class:`ConversationDamaged` for a file it cannot read."""
        path, name, key = self._project(project_path)
        if conversation_id is None:
            conversation_id = self._newest_id(key)
            if conversation_id is None:
                raise ConversationNotFound(None)
        _check_id(conversation_id)
        file, _lock_path, title_file = self._paths(key, conversation_id)
        if not file.exists():
            raise ConversationNotFound(conversation_id)
        with _OPEN_GUARD:
            registered = self._registered(key, conversation_id)
            held: _files.PidLock | LockedBy = "this_process" if registered is not None else \
                self._acquire_for_open(key, conversation_id)
            if not isinstance(held, _files.PidLock):
                if not file.exists():
                    raise ConversationNotFound(conversation_id)
                if if_locked == "raise":
                    raise ConversationLocked(held)
                data = self._peek(file)
                return self.new_conversation(project_path, _origin((data or {}).get("origin")),
                                             (data or {}).get("tier"), notice=LOCKED_NEW)
            try:
                data = _files.read_json(file)
            except (OSError, _files.UnreadableFile) as exc:
                held.release()
                raise ConversationDamaged(f"The saved conversation {conversation_id} could not be read: {exc}") \
                    from None
            if data is None:
                held.release()
                raise ConversationNotFound(conversation_id)
            try:
                conv = self._from_json(data, path, name, key, conversation_id)
            except (KeyError, TypeError, ValueError) as exc:
                held.release()
                raise ConversationDamaged(f"The saved conversation {conversation_id} could not be read: {exc}") \
                    from None
            conv.title = self._read_title(title_file)
            if conv.read_only:
                held.release()
                conv.notice = str(_files.NewerFormat(file, data.get("format")))
                return conv
            conv._lock = held
            self._register(conv)
        self._touch_project(path, name, key)
        return conv

    @staticmethod
    def _peek(file: Path) -> dict[str, Any] | None:
        try:
            return _files.read_json(file)
        except (OSError, _files.UnreadableFile):
            return None

    @staticmethod
    def _read_title(title_file: Path) -> str | None:
        try:
            data = _files.read_json(title_file)
        except (OSError, _files.UnreadableFile):
            return None
        title = (data or {}).get("title")
        return title if isinstance(title, str) and title else None

    @staticmethod
    def _from_json(data: dict[str, Any], path: Path, name: str, key: str, conversation_id: str) -> Conversation:
        usage = _empty_usage()
        if isinstance(data.get("usage"), dict):
            usage.update(data["usage"])
        number = data.get("number")
        return Conversation(
            id=conversation_id, project_key=key, project_name=name, project_path=path,
            created=str(data.get("created", "")), updated=str(data.get("updated", "")),
            origin=_origin(data.get("origin")), tier=data.get("tier"),
            system_hash=str(data.get("system_hash", "")), tools_hash=str(data.get("tools_hash", "")),
            messages=[message_from_json(m) for m in data.get("messages") or []], usage=usage,
            number=number if isinstance(number, int) and not isinstance(number, bool) else None,
            read_only=_files.is_newer(data))

    def save(self, conv: Conversation) -> bool:
        """Write *conv* (temp file + replace); ``True`` when written.  Nothing is written for a deleted, closed or
        read-only conversation, at expiry 0, or when another process holds the conversation's lock."""
        with conv.save_lock:
            if conv.deleted or conv.closed or conv.read_only:
                return False
            if self.expiry_days() == 0:
                return False
            if conv._lock is None or not conv._lock.held():
                lock = conv._lock or self._lock(conv.project_key, conv.id)
                if not lock.try_acquire():
                    _warn(f"the conversation {conv.id} is open in another FlexTool window or terminal, so it was "
                          "not saved here.")
                    return False
                first_lock = conv._lock is None
                conv._lock = lock
                if first_lock:
                    self._touch_project(conv.project_path, conv.project_name, conv.project_key)
            file, _lock, title_file = self._paths(conv.project_key, conv.id)
            conv.updated = _files.iso(_files.utc_now())
            _files.write_json_atomic(file, conv.to_json())
            if conv._title_pending:
                self._write_title(title_file, conv.title)
                conv._title_pending = False
            return True

    def close(self, conv: Conversation) -> None:
        """Release *conv*: its lock file is removed and later saves do nothing."""
        with conv.save_lock:
            conv.closed = True
            if conv._lock is not None:
                conv._lock.release()
            self._unregister(conv)

    def list_conversations(self, project_path: Any) -> list[ConversationInfo]:
        """The saved conversations of the project, the most recently changed first."""
        _path, _name, key = self._project(project_path)
        out = []
        for conversation_id, file in self._conversation_files(key):
            data = self._peek(file)
            if data is None:
                continue
            messages = [m for m in data.get("messages") or [] if isinstance(m, dict)]
            question = first_question(messages)
            title = self._read_title(self._paths(key, conversation_id)[2])
            number = data.get("number")
            out.append((self._last_change(conversation_id, file), ConversationInfo(
                id=conversation_id,
                number=number if isinstance(number, int) and not isinstance(number, bool) else None,
                title=title or question, named=title is not None, first_question=question,
                created=str(data.get("created", "")), updated=str(data.get("updated", "")),
                origin=_origin(data.get("origin")), locked=self._locked_by(key, conversation_id))))
        out.sort(key=lambda item: (item[0], item[1].id), reverse=True)
        return [info for _when, info in out]

    # --- titles, numbers, open chats --------------------------------------------------------------------------
    def _write_title(self, title_file: Path, title: str | None) -> None:
        if title:
            _files.write_json_atomic(title_file, {"format": _files.FORMAT, "title": title})
        else:
            with contextlib.suppress(FileNotFoundError):
                title_file.unlink()

    def set_title(self, project_path: Any, conversation_id: str, title: str) -> bool:
        """Name the chat *conversation_id* (an empty name: back to its number).  The name is written at once to
        its own small file, never waiting for a running answer; a chat not saved yet writes it with its first save.
        Returns whether the name is on disk (``False`` at expiry 0, for a deleted or unknown chat)."""
        _path, _name, key = self._project(project_path)
        _check_id(conversation_id)
        text = " ".join(str(title).split())[:TITLE_CHARS] or None
        file, _lock, title_file = self._paths(key, conversation_id)
        conv = self._registered(key, conversation_id)
        if conv is not None:
            if conv.deleted:
                return False
            conv.title = text
        if self.expiry_days() == 0:
            return False
        if not file.exists():
            if conv is not None:
                conv._title_pending = True
            return False
        self._write_title(title_file, text)
        return True

    def allocate_chat_number(self, project_path: Any) -> int:
        """The next chat number of the project: increasing, never given twice.  Kept in the project's row; at
        expiry 0 counted in memory from the stored value."""
        path, name, key = self._project(project_path)
        if self.expiry_days() == 0:
            row = self._read_projects().get(key) or {}
            stored = row.get("next_chat_number") if isinstance(row.get("next_chat_number"), int) else 1
            with _OPEN_GUARD:
                number = max(stored, _MEMORY_NUMBERS.get((self._root_key, key), 1))
                _MEMORY_NUMBERS[(self._root_key, key)] = number + 1
            return number

        def change(rows: dict[str, Any]) -> int:
            row = rows.setdefault(key, {})
            current = row.get("next_chat_number")
            number = current if isinstance(current, int) and not isinstance(current, bool) and current > 0 else 1
            with _OPEN_GUARD:
                number = max(number, _MEMORY_NUMBERS.pop((self._root_key, key), 1))
            row.update({"path": str(path), "name": name, "last_used": _files.iso(_files.utc_now()),
                        "next_chat_number": number + 1})
            return number
        return self._change_projects(change)

    def open_chats(self, project_path: Any) -> dict[str, Any]:
        """The chats the window had open for the project: ``{"ids": [...], "selected": id | None}``."""
        _path, _name, key = self._project(project_path)
        stored = (self._read_projects().get(key) or {}).get("open_chats") or {}
        ids = [i for i in stored.get("ids") or [] if is_id(i)] if isinstance(stored, dict) else []
        selected = stored.get("selected") if isinstance(stored, dict) else None
        return {"ids": ids, "selected": selected if selected in ids else None}

    def set_open_chats(self, project_path: Any, ids: list[str], selected: str | None) -> None:
        """Keep the open chats of the project (nothing at expiry 0)."""
        if self.expiry_days() == 0:
            return
        path, name, key = self._project(project_path)
        clean = [i for i in dict.fromkeys(ids) if is_id(i)]

        def change(rows: dict[str, Any]) -> None:
            row = rows.setdefault(key, {})
            row.update({"path": str(path), "name": name, "last_used": _files.iso(_files.utc_now()),
                        "open_chats": {"ids": clean, "selected": selected if selected in clean else None}})
        self._change_projects(change)

    # --- delete, forget, rename -------------------------------------------------------------------------------
    def _remove_files(self, key: str, conversation_id: str) -> bool:
        file, _lock, title_file = self._paths(key, conversation_id)
        removed = False
        for target in (file, title_file):
            try:
                target.unlink()
                removed = True
            except FileNotFoundError:
                pass
        return removed

    def delete_conversation(self, project_path: Any, conversation_id: str
                            ) -> Literal["deleted", "locked_elsewhere", "missing"]:
        """Delete a saved conversation, its lock and its title.  One open in this process is first marked
        deleted under its save lock (a later save writes nothing); it is never waited for beyond a save that is
        being written.  One open in another process is left alone."""
        _path, _name, key = self._project(project_path)
        if not is_id(conversation_id):
            return "missing"
        conv = self._registered(key, conversation_id)
        if conv is not None:
            with conv.save_lock:
                conv.deleted = True
                self._remove_files(key, conversation_id)
                if conv._lock is not None:
                    conv._lock.release()
            return "deleted"
        lock = self._lock(key, conversation_id, purpose="delete")
        if not lock.try_acquire():
            return "locked_elsewhere"
        try:
            removed = self._remove_files(key, conversation_id)
        finally:
            lock.release()
        return "deleted" if removed else "missing"

    def forget_project(self, project_path: Any) -> int:
        """Delete every saved conversation of a deleted project that no other process has open, and its row
        when nothing is left; returns how many conversations were deleted."""
        _path, _name, key = self._project(project_path)

        def change(rows: dict[str, Any]) -> int:
            count = 0
            ids = {cid for cid, _file in self._conversation_files(key)}
            ids |= self._title_ids(key)
            for conversation_id in sorted(ids):
                if self.delete_conversation(project_path, conversation_id) == "deleted":
                    count += 1
            self._clean_folder(key, time.time() + 1e9)
            if not self.project_dir(key).exists():
                rows.pop(key, None)
            return count
        return self._change_projects(change)

    def _title_ids(self, key: str) -> set[str]:
        try:
            names = os.listdir(self.project_dir(key))
        except (FileNotFoundError, NotADirectoryError):
            return set()
        return {n[:-11] for n in names if n.endswith(".title.json") and is_id(n[:-11])}

    def rename_project(self, old: Any, new: Any) -> int:
        """Move the saved conversations of a renamed project to its new key (the files no session has open; one
        open elsewhere stays under the old key and expires there); returns how many were moved."""
        _old_path, _old_name, old_key = self._project(old)
        new_path, new_name, new_key = self._project(new)

        def change(rows: dict[str, Any]) -> int:
            moved = 0
            if old_key != new_key:
                for conversation_id, file in self._conversation_files(old_key):
                    if self._registered(old_key, conversation_id) is not None:
                        continue
                    lock = self._lock(old_key, conversation_id, purpose="rename")
                    if not lock.try_acquire():
                        continue
                    try:
                        if self._move(file, conversation_id, old_key, new_key, new_name):
                            moved += 1
                    finally:
                        lock.release()
                self._clean_folder(old_key, 0.0)
            old_row = rows.get(old_key) or {}
            row = rows.setdefault(new_key, {})
            numbers = [n for n in (old_row.get("next_chat_number"), row.get("next_chat_number"))
                       if isinstance(n, int) and not isinstance(n, bool)]
            if "open_chats" in old_row and "open_chats" not in row:
                row["open_chats"] = old_row["open_chats"]
            row.update({"path": str(new_path), "name": new_name, "last_used": _files.iso(_files.utc_now())})
            if numbers:
                row["next_chat_number"] = max(numbers)
            if old_key != new_key and not self.project_dir(old_key).exists():
                rows.pop(old_key, None)
            return moved
        return self._change_projects(change)

    def _move(self, file: Path, conversation_id: str, old_key: str, new_key: str, new_name: str) -> bool:
        new_file, _new_lock, new_title = self._paths(new_key, conversation_id)
        if new_file.exists():
            return False
        data = self._peek(file)
        if data is None or _files.is_newer(data):
            return False
        data["project"] = {"key": new_key, "name": new_name}
        _files.write_json_atomic(new_file, data)
        title_file = self._paths(old_key, conversation_id)[2]
        title = self._read_title(title_file)
        if title:
            self._write_title(new_title, title)
        self._remove_files(old_key, conversation_id)
        return True

    # --- purge ------------------------------------------------------------------------------------------------
    def _clean_folder(self, key: str, cutoff: float) -> int:
        """Delete the conversations of one project folder last changed before *cutoff* (seconds since the epoch)
        that no session has open, orphan titles, dead writers' temp files, stale locks and left-over break
        markers; remove the folder when it is empty.  Returns how many conversations were deleted."""
        folder = self.project_dir(key)
        try:
            names = os.listdir(folder)
        except (FileNotFoundError, NotADirectoryError):
            return 0
        deleted = 0
        conversation_ids = {n[:-5] for n in names if n.endswith(".json") and is_id(n[:-5])}
        lock_ids = {n[:-10] for n in names if n.endswith(".json.lock") and is_id(n[:-10])}
        title_ids = {n[:-11] for n in names if n.endswith(".title.json") and is_id(n[:-11])}
        for conversation_id in sorted(conversation_ids | lock_ids | title_ids):
            if self._registered(key, conversation_id) is not None:
                continue
            lock = self._lock(key, conversation_id, purpose="purge")
            if not lock.try_acquire():
                continue                    # open in a running session (or another cleanup)
            try:
                file, _lock_path, title_file = self._paths(key, conversation_id)
                if conversation_id in conversation_ids and file.exists():
                    if self._last_change(conversation_id, file) < cutoff:
                        if self._remove_files(key, conversation_id):
                            deleted += 1
                else:
                    with contextlib.suppress(FileNotFoundError):
                        title_file.unlink()     # a title without its conversation
            finally:
                lock.release()
        for entry in names:
            target = folder / entry
            pid = _files.temp_writer_pid(entry)
            if pid is not None and not _files.pid_alive(pid):
                with contextlib.suppress(FileNotFoundError):
                    target.unlink()
            elif entry.endswith(".break"):
                with contextlib.suppress(OSError):
                    if time.time() - target.stat().st_mtime > 5.0:
                        target.unlink()
        with contextlib.suppress(OSError):
            folder.rmdir()                      # only when empty
        return deleted

    def purge_expired(self, now: datetime | None = None) -> int:
        """Delete what is older than the expiry in the settings (P1 §10.2); returns how many conversations were
        deleted.  Never raises for a file another process holds; a busy ``projects.json`` leaves the rows for the
        next purge."""
        days = self.expiry_days()
        moment = (now or _files.utc_now()).astimezone(timezone.utc)
        cutoff = (moment - timedelta(days=days)).timestamp() if days > 0 else float("inf")
        deleted = 0
        try:
            keys = [entry for entry in os.listdir(self.conversations_dir)
                    if (self.conversations_dir / entry).is_dir()]
        except (FileNotFoundError, NotADirectoryError):
            keys = []
        for key in sorted(keys):
            deleted += self._clean_folder(key, cutoff)
        self._purge_temp(self.root)

        def change(rows: dict[str, Any]) -> None:
            for key in list(rows):
                if self.project_dir(key).exists():
                    continue
                row = rows[key] if isinstance(rows[key], dict) else {}
                last = _parse_iso(row.get("last_used"))
                if days == 0 or last is None or last.timestamp() < cutoff:
                    del rows[key]
        if self.projects_path.exists():
            try:
                self._change_projects(change)
            except (OSError, _files.LockTimeout, _files.NewerFormat) as exc:
                _warn(f"the list of projects with saved conversations was not cleaned up: {exc}")
        return deleted

    @staticmethod
    def _purge_temp(folder: Path) -> None:
        """Delete the temp files of writers that are gone in *folder* (the settings, ledger and project list)."""
        try:
            names = os.listdir(folder)
        except (FileNotFoundError, NotADirectoryError):
            return
        for entry in names:
            pid = _files.temp_writer_pid(entry)
            if pid is not None and not _files.pid_alive(pid):
                with contextlib.suppress(FileNotFoundError, IsADirectoryError):
                    (folder / entry).unlink()


def _parse_iso(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def open_store(config_root: Path | None = None) -> Store:
    """The conversation store in *config_root* (default: ``<user config folder>/flextool/assistant``)."""
    return Store(config_root or _files.config_root())


def purge_at_start(config_root: Path | None = None) -> None:
    """The purge at the start of a chat, the agent connection and a command-line tool call: never fails the
    caller (a problem is a warning on stderr)."""
    try:
        open_store(config_root).purge_expired()
    except Exception as exc:        # noqa: BLE001  (the purge must never stop the tools from starting)
        _warn(f"old conversations could not be deleted: {exc}")
