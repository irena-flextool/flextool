"""The assistant's own files in the user config folder (P1 §10): where they are, how they are written and locked.

Every file is UTF-8 JSON with LF line ends and a ``"format"`` number.  A file is never written in place: the text
goes to a temp file ``<name>.<pid>-<start ms>.<random>.tmp`` next to it, which then replaces the file
(``os.replace``), so a reader or a killed writer never sees half a file.  A file of a newer format (written by a newer
FlexTool) is read but never overwritten (:class:`NewerFormat`).  A file found damaged twice in a row, with the same
bytes, is moved aside as ``<name>.damaged-<unix seconds>-<hex>`` (:func:`quarantine`).

Two kinds of lock file guard what several processes (the terminal chat, the FlexTool window, the agent
connection) and threads share:

* :func:`locked` — a short lock around one read-modify-write (``<file>.lock``, waited for at most 2 seconds);
  a lock older than 30 seconds, or whose process is gone, is stale and taken over;
* :class:`PidLock` — a long-held lock that says which process owns something (an open conversation), stale only
  when its process is gone.

A lock file holds the owner's process id, its start time and a random token; the token makes a release remove
only its own lock, and breaking a stale lock is itself serialised (``<lock>.break``), so two processes that both
find the same stale lock never both take it; a release removes its lock the same way.  On Windows a file another
process reads cannot be deleted for that moment: such a call is retried for a bounded time (at most
:data:`LOCK_TIMEOUT` plus 0.63 s in one operation); a lock that still cannot be removed is marked ``released``
through the handle its token was checked on (every process then breaks it at once), and when even that write fails a
background thread tries again for 30 seconds.  All times are UTC (P1-D11).

Only the standard library is imported at module level (``platformdirs`` and ``psutil`` inside functions), so the
agent connection's start-up purge stays cheap.
"""
from __future__ import annotations

import atexit
import contextlib
import hashlib
import json
import os
import re
import secrets
import socket
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypeVar

FORMAT = 1
LOCK_TIMEOUT = 2.0          # seconds a writer waits for a short lock
LOCK_STALE_AFTER = 30.0     # seconds after which a short lock counts as left behind
_POLL = 0.01
_BREAK_STALE_AFTER = 5.0    # a break marker older than this was left by a killed breaker
_WINDOWS = sys.platform == "win32"
# On Windows a delete, replace or create fails while another process has the file open (a sharing violation, or a
# delete still pending): retried after each of these waits (0.63 s in all), then the PermissionError is raised.
_SHARING_WAITS = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32)
_SHARING_TOTAL = sum(_SHARING_WAITS)
_sleep = time.sleep           # tests replace both
_clock = time.monotonic
_RETRY_EVERY = 0.05         # a lock that could not be released is tried again this often ...
_RETRY_FOR = 30.0           # ... for this long (then it is stale for every other process anyway)
_STABLE_WAIT = 0.05         # a damaged file is looked at again after this wait before it is set aside
TEMP_ORPHAN_AGE = 60.0      # a temp file whose writer's start time differs is removed only when older than this
DAMAGED = re.compile(r"^(?P<base>.+)\.damaged-(?P<time>\d+)(?:-[0-9a-f]+)?$")

T = TypeVar("T")


class NewerFormat(Exception):
    """A file written by a newer FlexTool: read, never overwritten."""

    def __init__(self, path: Path, found: object) -> None:
        super().__init__(f"{path.name} was written by a newer FlexTool (format {found}); this FlexTool reads it "
                         "but does not change it.")
        self.path = path
        self.found = found


class UnreadableFile(Exception):
    """A file that is not the JSON object FlexTool wrote."""

    def __init__(self, path: Path, reason: str) -> None:
        super().__init__(f"{path.name} could not be read: {reason}")
        self.path = path


class TransientRead(OSError):
    """A file that read as damaged once but not twice in a row with the same bytes: left as it is, not written."""

    def __init__(self, path: Path) -> None:
        super().__init__(f"{path.name} could not be read just now (it was changing); it was left as it is.")
        self.path = path


class LockTimeout(TimeoutError):
    """Another FlexTool process held a lock for longer than a writer waits."""

    def __init__(self, path: Path) -> None:
        super().__init__(f"{path.name} is being changed by another FlexTool window or terminal; try again.")
        self.path = path


# --- where -------------------------------------------------------------------------------------------------------
def config_root() -> Path:
    """``<user config folder>/flextool/assistant``: the folder of every assistant file."""
    from flextool.user_dirs import user_config_dir

    return user_config_dir() / "assistant"


def project_key(path: Path | str) -> str:
    """The key of a project folder: ``sha256(normcase(resolved path))[:16]``."""
    text = os.path.normcase(str(Path(path).resolve()))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(moment: datetime) -> str:
    """*moment* as UTC ISO text with a ``Z``, to the second."""
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- writing ----------------------------------------------------------------------------------------------------
def _retry_sharing(op: Callable[[], T], *, missing_ok: bool = False, deadline: float | None = None) -> T | None:
    """Run *op*; on Windows a ``PermissionError`` is retried after each wait of :data:`_SHARING_WAITS` that ends
    before *deadline* (``time.monotonic``), then raised.  ``FileNotFoundError``: ``None`` with *missing_ok*, else
    raised.  Not on Windows *op* runs once: a POSIX ``PermissionError`` is a permission problem, not a sharing one."""
    waits = iter(_SHARING_WAITS if _WINDOWS else ())
    while True:
        try:
            return op()
        except FileNotFoundError:
            if missing_ok:
                return None
            raise
        except PermissionError:
            wait = next(waits, None)
            if wait is None or (deadline is not None and _clock() + wait > deadline):
                raise
            _sleep(wait)


def unlink_retrying(path: Path, *, deadline: float | None = None) -> bool:
    """Delete *path*: ``True`` when removed, ``False`` when it was already gone; the last ``PermissionError`` of
    the retries (:func:`_retry_sharing`) is raised."""
    def unlink() -> bool:
        os.unlink(path)
        return True
    return bool(_retry_sharing(unlink, missing_ok=True, deadline=deadline))


def temp_name(path: Path) -> Path:
    """The temp file a writer of *path* uses: ``<name>.<pid>-<start ms>.<random>.tmp`` in the same folder (``<pid>``
    alone when the process's start time cannot be read)."""
    start = _own_start()
    writer = f"{os.getpid()}-{int(start * 1000)}" if start is not None else str(os.getpid())
    return path.with_name(f"{path.name}.{writer}.{secrets.token_hex(4)}.tmp")


def temp_writer(name: str) -> tuple[int, float | None] | None:
    """The writer's process id and start time (seconds, ``None`` when the name has none) in a temp file name, or
    ``None`` for a name that is not one.  Names of both forms (with and without the start time) are read."""
    parts = name.rsplit(".", 3)
    if len(parts) != 4 or parts[3] != "tmp":
        return None
    pid, dash, start = parts[1].partition("-")
    if not pid.isdigit() or (dash and not start.isdigit()):
        return None
    return int(pid), (int(start) / 1000.0 if dash else None)


def temp_orphan(path: Path) -> bool:
    """Whether *path* is a temp file whose writer is gone: its process does not run, or (a reused process id) runs
    with another start time and the file is older than :data:`TEMP_ORPHAN_AGE` seconds."""
    writer = temp_writer(path.name)
    if writer is None:
        return False
    pid, started = writer
    if not pid_alive(pid):
        return True
    if started is None or pid_alive(pid, started):
        return False
    age = _age(path)
    return age is not None and age > TEMP_ORPHAN_AGE


def _create_temp(path: Path, data: bytes, private: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = temp_name(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(temp, flags, 0o600 if private else 0o666)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            temp.unlink()
        raise
    return temp


def _replace(temp: Path, path: Path) -> None:
    try:
        _retry_sharing(lambda: os.replace(temp, path))
    except PermissionError:
        with contextlib.suppress(OSError):
            os.unlink(temp)
        raise


def write_text_atomic(path: Path, text: str, *, private: bool = False) -> None:
    """Write *text* (UTF-8, LF) to *path* through a temp file and ``os.replace``.

    The folder is created right before the temp file; when a purge removed it in between, the write is tried once
    more.  *private* creates the temp file readable by the owner only (0600 on POSIX), before any byte is written.
    """
    data = text.replace("\r\n", "\n").encode("utf-8")
    try:
        temp = _create_temp(path, data, private)
    except FileNotFoundError:
        temp = _create_temp(path, data, private)
    try:
        _replace(temp, path)
    except FileNotFoundError:
        # The folder went away between the temp file and the replace (a purge); write it all again.
        temp = _create_temp(path, data, private)
        _replace(temp, path)


def dumps(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=1) + "\n"


def write_json_atomic(path: Path, data: dict[str, Any], *, private: bool = False,
                      format_checked: bool = False) -> None:
    """Write *data* as JSON; refuses (:class:`NewerFormat`) to replace a file of a newer format, and
    (:class:`UnreadableFile`) one that is not a JSON object.  *format_checked*: the caller has just read the file
    under the lock that guards it and checked :func:`is_newer` (or set an unreadable one aside), so it is not
    parsed again here."""
    if not format_checked:
        current = read_json(path, missing_ok=True, check_format=False)
        if current is not None:
            found = current.get("format")
            if isinstance(found, int) and found > FORMAT:
                raise NewerFormat(path, found)
    write_text_atomic(path, dumps(data), private=private)


def read_json(path: Path, *, missing_ok: bool = True, check_format: bool = True) -> dict[str, Any] | None:
    """The JSON object in *path*; ``None`` when the file does not exist (and *missing_ok*).

    Raises :class:`UnreadableFile` for anything but a JSON object.  With *check_format* a newer format is
    returned as it is; :func:`is_newer` tells the caller to treat it read-only.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        if missing_ok:
            return None
        raise
    except UnicodeDecodeError as exc:
        raise UnreadableFile(path, f"not UTF-8 text ({exc.reason})") from None
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise UnreadableFile(path, f"not valid JSON ({exc})") from None
    if not isinstance(data, dict):
        raise UnreadableFile(path, "not a JSON object")
    return data


def is_newer(data: dict[str, Any] | None) -> bool:
    """Whether *data* was written in a newer format than this FlexTool writes."""
    found = None if data is None else data.get("format")
    return isinstance(found, int) and found > FORMAT


def _snapshot(path: Path) -> tuple[bytes, int, int] | None:
    try:
        before = os.stat(path)
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    return data, before.st_mtime_ns, before.st_size


def quarantine(path: Path, still_damaged: Callable[[bytes], bool]) -> Path:
    """Move a damaged *path* aside as ``<name>.damaged-<unix seconds>-<4 hex>`` and return the new path.

    Damage is decided on stable evidence only: the file is read twice, :data:`_STABLE_WAIT` apart, and set aside
    only when both reads give the same bytes with the same modification time and size, and *still_damaged* holds
    for them.  Otherwise :class:`TransientRead` (an ``OSError``: the caller writes nothing this time).  A move that
    fails raises its ``OSError`` and leaves the file in place; the caller never writes over it then."""
    first = _snapshot(path)
    _sleep(_STABLE_WAIT)
    second = _snapshot(path)
    if first is None or first != second or not still_damaged(second[0]):
        raise TransientRead(path)
    aside = path.with_name(f"{path.name}.damaged-{int(time.time())}-{secrets.token_hex(2)}")
    _retry_sharing(lambda: os.replace(path, aside))
    return aside


def not_a_json_object(raw: bytes) -> bool:
    """Whether *raw* is not UTF-8 JSON text of an object (what :func:`read_json` refuses)."""
    try:
        return not isinstance(json.loads(raw.decode("utf-8")), dict)
    except ValueError:
        return True


# --- processes ---------------------------------------------------------------------------------------------------
_HOST = socket.gethostname()


def _start_time(pid: int) -> float | None:
    """The start time of process *pid* (seconds since the epoch), or ``None`` when it cannot be read."""
    try:
        import psutil
    except ImportError:
        return None
    try:
        return float(psutil.Process(pid).create_time())
    except Exception:       # noqa: BLE001  (gone, access denied, zombie: all mean "cannot say")
        return None


_OWN_START: dict[int, float | None] = {}     # keyed by process id: a forked child has its own


def _own_start() -> float | None:
    pid = os.getpid()
    if pid not in _OWN_START:
        _OWN_START[pid] = _start_time(pid)
    return _OWN_START[pid]


def pid_alive(pid: int, started: float | None = None) -> bool:
    """Whether process *pid* runs; with *started*, also whether it is the same process (a reused process id after a
    restart has another start time)."""
    if pid <= 0:
        return False
    if pid == os.getpid():
        own = _own_start()
        return started is None or own is None or abs(started - own) < 1.0
    try:
        import psutil
    except ImportError:
        return _pid_alive_stdlib(pid)
    if not psutil.pid_exists(pid):
        return False
    if started is None:
        return True
    now_started = _start_time(pid)
    return now_started is None or abs(now_started - started) < 1.0


def _pid_alive_stdlib(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(0x1000, False, pid)       # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5                 # access denied: it exists
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259                            # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# --- lock files --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LockOwner:
    pid: int
    host: str
    started: float | None
    token: str
    purpose: str = "owner"      # "owner", or a short cleanup ("purge", "delete", "rename", "forget")
    released: bool = False      # let go, but the file could not be deleted yet (Windows): stale for everyone

    @property
    def this_process(self) -> bool:
        return self.host == _HOST and self.pid == os.getpid()

    def alive(self) -> bool:
        """Whether the owner still runs.  A lock of another computer (a shared home folder) counts as alive."""
        if self.host != _HOST:
            return True
        return pid_alive(self.pid, self.started)


def _parse_owner(raw: str) -> LockOwner | None:
    try:
        data = json.loads(raw)
        return LockOwner(int(data["pid"]), str(data["host"]),
                         float(data["started"]) if data.get("started") is not None else None, str(data["token"]),
                         str(data.get("purpose") or "owner"), data.get("released") is True)
    except (ValueError, KeyError, TypeError, AttributeError):
        # Being written right now (created but not yet filled), or damaged: the caller judges it by its age.
        return None


def _read_lock_text(lock_path: Path) -> str:
    return lock_path.read_text(encoding="utf-8", errors="replace")


def read_owner(lock_path: Path, *, retry: bool = False, deadline: float | None = None
               ) -> tuple[LockOwner | None, str | None]:
    """The owner of *lock_path* and the file's raw text; ``(None, None)`` when there is no lock file and
    ``(None, "")`` when it cannot be read.  *retry*: a Windows sharing violation is retried
    (:func:`_retry_sharing`) before it counts as unreadable (not inside a polling loop, which retries anyway)."""
    try:
        raw = _retry_sharing(lambda: _read_lock_text(lock_path), deadline=deadline) if retry \
            else _read_lock_text(lock_path)
    except FileNotFoundError:
        return None, None
    except OSError:
        return None, ""
    assert raw is not None
    return _parse_owner(raw), raw


def _age(path: Path) -> float | None:
    try:
        return time.time() - path.stat().st_mtime
    except OSError:
        return None


RemoveResult = Literal["removed", "changed", "busy"]


def _remove_if_same(lock_path: Path, seen: str, deadline: float | None = None) -> RemoveResult:
    """Remove *lock_path* if it still holds exactly *seen*, serialised with every other breaker (and releaser) of
    this lock through ``<lock>.break``: ``"removed"`` (also when it was gone), ``"changed"`` (it holds something
    else now: left alone), ``"busy"`` (another breaker is at it, or Windows would not let go of the file)."""
    marker = lock_path.with_name(lock_path.name + ".break")
    try:
        fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    except FileExistsError:
        age = _age(marker)
        if age is not None and age > _BREAK_STALE_AFTER:
            with contextlib.suppress(OSError):
                unlink_retrying(marker, deadline=deadline)
        return "busy"
    except FileNotFoundError:
        return "removed"        # the folder is gone, and the lock with it
    except PermissionError:
        if not _WINDOWS:
            raise
        return "busy"           # a marker being deleted this moment
    os.close(fd)
    try:
        try:
            current = _retry_sharing(lambda: _read_lock_text(lock_path), deadline=deadline)
        except FileNotFoundError:
            return "removed"
        except PermissionError:
            return "busy"
        if current != seen:
            return "changed"
        try:
            unlink_retrying(lock_path, deadline=deadline)
        except PermissionError:
            return "busy"
        return "removed"
    finally:
        with contextlib.suppress(OSError):      # a marker left behind is taken over after 5 s
            unlink_retrying(marker, deadline=deadline)


def _break_stale(lock_path: Path, seen: str, deadline: float | None = None) -> None:
    """Remove the stale *lock_path* if it still holds exactly *seen* (:func:`_remove_if_same`)."""
    _remove_if_same(lock_path, seen, deadline)


def _read_fd(fd: int) -> str:
    os.lseek(fd, 0, os.SEEK_SET)
    chunks = []
    while True:
        chunk = os.read(fd, 65536)
        if not chunk:
            return b"".join(chunks).decode("utf-8", errors="replace")
        chunks.append(chunk)


def _mark_released(lock_path: Path, token: str, deadline: float | None) -> bool:
    """Mark the lock of *token* released in place, so every process breaks it at once; ``True`` when *lock_path*
    is no live lock of *token* any more (marked, gone, or another owner's, which is left untouched), ``False``
    when it could not even be opened or written.

    The token is checked and the mark written through one handle, so both act on the same file: a lock that was
    broken and taken by another owner in between is a different file, never written over."""
    flags = os.O_RDWR | getattr(os, "O_BINARY", 0)
    try:
        fd = _retry_sharing(lambda: os.open(lock_path, flags), missing_ok=True, deadline=deadline)
    except OSError:
        return False
    if fd is None:
        return True
    try:
        raw = _read_fd(fd)
        owner = _parse_owner(raw)
        if owner is None or owner.token != token:
            return True             # another owner's lock (or one being written this moment): not ours
        if owner.released:
            return True
        data = json.loads(raw)
        data["released"] = True
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, json.dumps(data).encode("utf-8"))
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


def _release_owned(lock_path: Path, token: str, deadline: float | None) -> bool:
    """Remove the lock of *token* (only while the file still holds that token), else mark it released;
    ``True`` when nothing more is to be done, ``False`` when it should be tried again."""
    owner, raw = read_owner(lock_path, retry=True, deadline=deadline)
    if raw is None:
        return True                 # gone
    if owner is not None:
        if owner.token != token:
            return True             # broken as stale and taken by another owner since: not ours
        if _remove_if_same(lock_path, raw, deadline) != "busy":
            return True
    return _mark_released(lock_path, token, deadline)


# Locks this process let go of but could neither remove nor mark released: (normcase path, token) -> (path, the
# monotonic time to give up).  A daemon thread retries them; a last try at exit.
_PENDING: dict[tuple[str, str], tuple[Path, float]] = {}
_PENDING_GUARD = threading.Lock()
_PENDING_THREAD: threading.Thread | None = None


def _queue_release(lock_path: Path, token: str) -> None:
    with _PENDING_GUARD:
        _PENDING[(os.path.normcase(str(lock_path)), token)] = (lock_path, _clock() + _RETRY_FOR)
    _start_retry_thread()


def retry_pending_releases() -> int:
    """One more try (no waiting) at each lock this process could not release; returns how many are left."""
    with _PENDING_GUARD:
        items = list(_PENDING.items())
    for key, (path, give_up) in items:
        try:
            done = _release_owned(path, key[1], deadline=_clock())
        except OSError:
            done = False
        if done or _clock() > give_up:
            if not done:
                print(f"Warning: the lock file {path.name} could not be removed; other FlexTool windows treat it as "
                      "left behind once it is old.", file=sys.stderr)
            with _PENDING_GUARD:
                _PENDING.pop(key, None)
    with _PENDING_GUARD:
        return len(_PENDING)


def _retry_loop() -> None:
    global _PENDING_THREAD
    wait = threading.Event()
    while True:
        wait.wait(_RETRY_EVERY)
        retry_pending_releases()
        with _PENDING_GUARD:
            if not _PENDING:
                _PENDING_THREAD = None
                return


def _start_retry_thread() -> None:
    global _PENDING_THREAD
    with _PENDING_GUARD:
        if _PENDING_THREAD is not None:
            return
        _PENDING_THREAD = threading.Thread(target=_retry_loop, name="flextool-lock-release", daemon=True)
        _PENDING_THREAD.start()


atexit.register(retry_pending_releases)


class PidLock:
    """A lock file that names the process holding it.

    :meth:`try_acquire` takes the lock if it is free or stale; *stale_after* (seconds) makes an old lock stale
    even when its process runs (``None``: only a gone process makes it stale).  *purpose* tells a process that
    finds the lock taken whether an owner holds it or a cleanup that lets go within moments.  *extra* adds fields
    to the lock file (a conversation's chat number, read by :meth:`Store._seed_number`).
    """

    def __init__(self, lock_path: Path, *, stale_after: float | None = None, purpose: str = "owner",
                 extra: Mapping[str, Any] | None = None) -> None:
        self.path = lock_path
        self.stale_after = stale_after
        self.purpose = purpose
        self.extra = dict(extra or {})
        self.token: str | None = None
        self._deadline: float | None = None     # set by acquire: bounds the Windows retries of each try

    def _content(self) -> str:
        assert self.token is not None
        return json.dumps({**self.extra, "pid": os.getpid(), "host": _HOST, "started": _own_start(),
                           "token": self.token, "purpose": self.purpose})

    def owner(self) -> LockOwner | None:
        return read_owner(self.path)[0]

    def is_stale(self, owner: LockOwner | None, raw: str | None) -> bool:
        if raw is None:
            return False
        age = _age(self.path)
        if owner is None:
            # A lock file without a readable owner is being written (milliseconds) or was left half-written.
            return age is not None and age > 1.0
        if owner.released:
            return True
        if self.stale_after is not None and age is not None and age > self.stale_after:
            return True
        return not owner.alive()

    def try_acquire(self) -> bool:
        """Take the lock: ``True`` when this process now holds it.  On Windows a lock file that is being deleted
        (or read) this moment counts as taken; the caller's poll tries again."""
        deadline = self._deadline if self._deadline is not None else _clock() + _SHARING_TOTAL
        for _ in range(2):
            self.token = secrets.token_hex(8)
            content = self._content().encode("utf-8")
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
            except FileExistsError:
                owner, raw = read_owner(self.path)
                if raw is not None and self.is_stale(owner, raw):
                    _break_stale(self.path, raw, deadline)
                    continue
                self.token = None
                return False
            except FileNotFoundError:
                continue        # the folder was removed between mkdir and open; once more
            except PermissionError:
                self.token = None
                if not _WINDOWS:
                    raise
                return False
            try:
                os.write(fd, content)
            finally:
                os.close(fd)
            return True
        self.token = None
        return False

    def acquire(self, timeout: float = LOCK_TIMEOUT) -> None:
        deadline = _clock() + timeout
        self._deadline = deadline
        try:
            while not self.try_acquire():
                if _clock() >= deadline:
                    raise LockTimeout(self.path)
                _sleep(_POLL)
        finally:
            self._deadline = None

    def held(self) -> bool:
        """Whether this object still holds its lock (the file holds its token)."""
        if self.token is None:
            return False
        owner = read_owner(self.path, retry=True)[0]
        return owner is not None and owner.token == self.token and not owner.released

    def release(self) -> None:
        """Remove the lock file if it is still this object's lock; never raises.  When Windows does not let go of
        the file it is marked released in place, and when even that fails it is tried again in the background."""
        token, self.token = self.token, None
        if token is None:
            return
        try:
            if not _release_owned(self.path, token, _clock() + _SHARING_TOTAL):
                _queue_release(self.path, token)
        except OSError as exc:
            print(f"Warning: the lock file {self.path.name} could not be removed: {exc}", file=sys.stderr)


# One threading lock per locked path in this process: threads of one process queue here first, so they do not
# spin on each other's lock file.
_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _thread_lock(path: Path) -> threading.Lock:
    key = os.path.normcase(str(path))
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.Lock())


@contextlib.contextmanager
def locked(path: Path, *, timeout: float = LOCK_TIMEOUT) -> Iterator[None]:
    """Hold ``<path>.lock`` for one read-modify-write of *path*; raises :class:`LockTimeout` after *timeout*."""
    thread_lock = _thread_lock(path)
    if not thread_lock.acquire(timeout=timeout):
        raise LockTimeout(path)
    try:
        lock = PidLock(path.with_name(path.name + ".lock"), stale_after=LOCK_STALE_AFTER)
        lock.acquire(timeout)
        try:
            yield
        finally:
            lock.release()
    finally:
        thread_lock.release()
