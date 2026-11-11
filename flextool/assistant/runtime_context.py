"""The runtime context file: what the FlexTool window shows, for an AI agent connected to the same project (P2 §6.4).

While a project is open in the FlexTool window (also without a chat window, P2-O1) its process keeps one file::

    <config>/assistant/runtime/<project key>.<pid>.json
    {"format": 1, "pid": 4242, "written_at": "2026-11-07T10:00:00Z", "changed_at": "2026-11-07T09:58:12Z",
     "active_at": "2026-11-07T09:59:40Z", "flextool_version": "4.0.0", "project": {"key": "…", "name": "…"},
     "context": snapshot_payload(…)}

It holds names and selections only — no paths, no data values, nothing from the chats — and reaches a provider
only through an agent the user runs.  :class:`RuntimeContextWriter` writes it from a daemon thread (UTF-8, LF,
temp file + replace): the newest published payload wins (``changed_at``: when the selections last changed), and
:meth:`RuntimeContextWriter.stop` makes the thread delete the file as its last action.  The window's 1 s poll
calls :meth:`RuntimeContextWriter.tick` on the Tk thread, saying whether FlexTool has the keyboard focus
(``active_at``: when it last had it — the window in use).  The heartbeat rewrites ``written_at`` every
``heartbeat_s`` seconds only when the poll ticked since the last one (review G18, C3-m5): a Tk thread that hangs
stops the ticks, so its file goes stale.  Tk's modal dialogs keep running ``after`` timers, so the ticks go on
while one is open.

:func:`read_runtime_context` looks at the files of the project whose process is alive.  Of those written within
:data:`STALE_AFTER` seconds it takes the window in use: the latest ``active_at``, then ``changed_at``, then
``written_at`` (a field a file lacks — one of an older FlexTool — counts as earliest).  When there is no fresh
file but a live process's file is stale, the window is open but not answering (busy, or hung): ``"running"``
with no context.  Several FlexTool windows on one project each keep their own file.  :func:`cleanup` (at GUI
start) deletes the files of processes that are gone.

Standard library, :mod:`._files` (``psutil`` when installed, for the process check) and :mod:`.redact`.
"""
from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from flextool.assistant import _files
from flextool.assistant.context import ContextSnapshot, snapshot_payload

RUNTIME_DIR = "runtime"
#: A file not rewritten within this many seconds belongs to a window that is gone, hangs or is busy for longer than
#: that (its Tk thread has not ticked): its process being alive tells the last two from the first.
STALE_AFTER = 120.0
HEARTBEAT = 30.0
GuiRunning = Literal["running", "not running"]


def runtime_dir(config_root: Path | None = None) -> Path:
    return (config_root or _files.config_root()) / RUNTIME_DIR


def runtime_path(project_path: Path | str, *, config_root: Path | None = None, pid: int | None = None) -> Path:
    """The runtime file of the project for process *pid* (this process by default)."""
    key = _files.project_key(project_path)
    return runtime_dir(config_root) / f"{key}.{os.getpid() if pid is None else pid}.json"


def _flextool_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("flextool")
    except PackageNotFoundError:
        return "unknown"


def _file_pid(name: str) -> tuple[str, int] | None:
    """(project key, pid) of a runtime file name, ``None`` for any other name."""
    if not name.endswith(".json"):
        return None
    parts = name[:-5].split(".")
    if len(parts) != 2 or not parts[1].isdigit():
        return None
    return parts[0], int(parts[1])


class RuntimeContextWriter:
    """Keeps the runtime file of one open project of this process.

    :meth:`start` writes the file at once (the payload of a snapshot with no selections until the first
    :meth:`publish`); :meth:`publish` hands the thread a new payload (any thread; the latest wins); :meth:`stop`
    asks the thread to finish its write, delete the file and end, and waits for it at most *wait* seconds.
    Payloads are redacted for the project (paths, user and host names) before they are written.
    """

    def __init__(self, project_path: Path | str, project_name: str, *, config_root: Path | None = None,
                 heartbeat_s: float = HEARTBEAT, pid: int | None = None,
                 clock: Callable[[], datetime] = _files.utc_now) -> None:
        from flextool.assistant.redact import Redactor

        self.project_path = Path(project_path)
        self.project_name = project_name
        self.pid = os.getpid() if pid is None else pid
        self.path = runtime_path(self.project_path, config_root=config_root, pid=self.pid)
        self.heartbeat_s = heartbeat_s
        self._key = _files.project_key(self.project_path)
        self._redactor = Redactor(self.project_path)
        self._version = _flextool_version()
        self._guard = threading.Lock()
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._pending: dict[str, Any] | None = None
        self._current: dict[str, Any] = snapshot_payload(ContextSnapshot(project=project_name, front_end="gui"),
                                                         gui_running="running", redact=self._redactor.redact)
        self._thread: threading.Thread | None = None
        self._clock = clock
        #: Set by every :meth:`tick` (the window's poll ran): the next heartbeat may write.
        self._alive = threading.Event()
        self._changed_at: datetime | None = None
        self._active_at: datetime | None = None
        self._was_active = False
        self._focus_gained = False
        #: How many times the file was written, and how many heartbeats came (for the tests).
        self.writes = 0
        self.heartbeats = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._pending = self._current
        self._alive.set()
        self._thread = threading.Thread(target=self._run, name="flextool-assistant-runtime-context", daemon=True)
        self._thread.start()
        self._wake.set()

    def publish_snapshot(self, snapshot: ContextSnapshot | None) -> None:
        """Publish the payload of *snapshot* (redacted for this project)."""
        self.publish(snapshot_payload(snapshot, gui_running="running", redact=self._redactor.redact))

    def publish(self, payload: dict[str, Any]) -> None:
        """Have the thread write *payload* (a :func:`~flextool.assistant.context.snapshot_payload`)."""
        with self._guard:
            self._pending = payload
        self._wake.set()

    def tick(self, active: bool = False) -> None:
        """The window's poll ran (any thread; no I/O): the next heartbeat writes the file.  *active*: FlexTool has
        the keyboard focus now; the moment it gains it is written at once (the window in use)."""
        wake = False
        with self._guard:
            if active:
                self._active_at = self._clock()
                if not self._was_active:
                    self._focus_gained = wake = True
            self._was_active = active
        self._alive.set()
        if wake:
            self._wake.set()

    def stop(self, wait: float = 2.0) -> bool:
        """End the writer: the thread finishes a write under way, deletes the file and ends.  Waits for it at most
        *wait* seconds; ``True`` when it has ended (the file is gone)."""
        self._stopping.set()
        self._wake.set()
        thread = self._thread
        if thread is None:
            return True
        thread.join(wait)
        return not thread.is_alive()

    # --- the thread ---------------------------------------------------------------------------------------------
    def _record(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._guard:
            changed, active = self._changed_at, self._active_at
        return {"format": _files.FORMAT, "pid": self.pid, "written_at": _files.iso(self._clock()),
                "changed_at": _files.iso(changed) if changed is not None else None,
                "active_at": _files.iso(active) if active is not None else None,
                "flextool_version": self._version, "project": {"key": self._key, "name": self.project_name},
                "context": payload}

    def _write(self, payload: dict[str, Any]) -> None:
        try:
            _files.write_text_atomic(self.path, _files.dumps(self._record(payload)))
        except OSError:
            pass                    # the next publish or heartbeat writes it again
        self.writes += 1

    def _run(self) -> None:
        try:
            while not self._stopping.is_set():
                woken = self._wake.wait(self.heartbeat_s)
                self._wake.clear()
                with self._guard:
                    pending, self._pending = self._pending, None
                    focus, self._focus_gained = self._focus_gained, False
                    if pending is not None:
                        self._changed_at = self._clock()       # the publisher hands over only changes
                if pending is not None:
                    self._current = pending
                if self._stopping.is_set():
                    break
                beat = False
                if not woken:
                    self.heartbeats += 1
                    beat = self._alive.is_set()     # only while the window's poll ticks
                if pending is not None or focus or beat:
                    self._alive.clear()
                    self._write(self._current)
        finally:
            with contextlib.suppress(OSError):
                self.path.unlink()


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def read_runtime_context(project_path: Path | str, *, config_root: Path | None = None,
                         now: datetime | None = None) -> tuple[dict[str, Any] | None, GuiRunning]:
    """The context the FlexTool window in use publishes for the project, and whether a window is running.

    Of the runtime files of the project whose process is alive and that were written at most :data:`STALE_AFTER`
    seconds ago, the one with the latest ``active_at``, then ``changed_at``, then ``written_at``; others are
    ignored.  The context comes with those times and ``responding: true``.  When only stale files of live processes
    are left, the window is open but its Tk thread has not answered for that long (a long operation, or a hang):
    ``(None, "running")``."""
    key = _files.project_key(project_path)
    folder = runtime_dir(config_root)
    moment = (now or _files.utc_now()).astimezone(timezone.utc)
    try:
        names = os.listdir(folder)
    except (FileNotFoundError, NotADirectoryError):
        return None, "not running"
    best: tuple[tuple[datetime, datetime, datetime], dict[str, Any]] | None = None
    stale_alive = False
    for name in names:
        parsed = _file_pid(name)
        if parsed is None or parsed[0] != key or not _files.pid_alive(parsed[1]):
            continue
        try:
            data = _files.read_json(folder / name)
        except (OSError, _files.UnreadableFile):
            continue
        if not data or _files.is_newer(data) or not isinstance(data.get("context"), dict):
            continue
        written = _parse_time(data.get("written_at"))
        if written is None:
            continue
        if (moment - written).total_seconds() > STALE_AFTER:
            stale_alive = True
            continue
        order = (_parse_time(data.get("active_at")) or _EARLIEST, _parse_time(data.get("changed_at")) or _EARLIEST,
                 written)
        if best is None or order > best[0]:
            best = (order, data)
    if best is None:
        return None, "running" if stale_alive else "not running"
    data = best[1]
    context = dict(data["context"])
    context["gui"] = "running"
    context["responding"] = True
    context["written_at"] = data["written_at"]
    for field in ("changed_at", "active_at"):
        if isinstance(data.get(field), str):
            context[field] = data[field]
    return context, "running"


_EARLIEST = datetime.min.replace(tzinfo=timezone.utc)


def cleanup(config_root: Path | None = None) -> int:
    """Delete the runtime files (and temp files) of processes that are gone; returns how many files."""
    folder = runtime_dir(config_root)
    try:
        names = os.listdir(folder)
    except (FileNotFoundError, NotADirectoryError):
        return 0
    removed = 0
    for name in names:
        parsed = _file_pid(name)
        if parsed is not None:
            if _files.pid_alive(parsed[1]):
                continue
        elif not _files.temp_orphan(folder / name):
            continue
        with contextlib.suppress(FileNotFoundError):
            (folder / name).unlink()
            removed += 1
    return removed
