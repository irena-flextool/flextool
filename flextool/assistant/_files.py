"""The assistant's own files in the user config folder (P1 §10): where they are, how they are written and locked.

Every file is UTF-8 JSON with LF line ends and a ``"format"`` number.  A file is never written in place: the text
goes to a temp file ``<name>.<pid>.<random>.tmp`` next to it, which then replaces the file (``os.replace``), so a
reader or a killed writer never sees half a file.  A file of a newer format (written by a newer FlexTool) is read
but never overwritten (:class:`NewerFormat`).

Two kinds of lock file guard what several processes (the terminal chat, the FlexTool window, the agent
connection) and threads share:

* :func:`locked` — a short lock around one read-modify-write (``<file>.lock``, waited for at most 2 seconds);
  a lock older than 30 seconds, or whose process is gone, is stale and taken over;
* :class:`PidLock` — a long-held lock that says which process owns something (an open conversation), stale only
  when its process is gone.

A lock file holds the owner's process id, its start time and a random token; the token makes a release remove
only its own lock, and breaking a stale lock is itself serialised (``<lock>.break``), so two processes that both
find the same stale lock never both take it.  All times are UTC (P1-D11).

Only the standard library is imported at module level (``platformdirs`` and ``psutil`` inside functions), so the
agent connection's start-up purge stays cheap.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import secrets
import socket
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FORMAT = 1
LOCK_TIMEOUT = 2.0          # seconds a writer waits for a short lock
LOCK_STALE_AFTER = 30.0     # seconds after which a short lock counts as left behind
_POLL = 0.01
_BREAK_STALE_AFTER = 5.0    # a break marker older than this was left by a killed breaker
_WINDOWS_RETRIES = 5        # os.replace on Windows fails while another process reads the target
_WINDOWS_RETRY_WAIT = 0.05


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
def temp_name(path: Path) -> Path:
    """The temp file a writer of *path* uses: ``<name>.<pid>.<random>.tmp`` in the same folder."""
    return path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")


def temp_writer_pid(name: str) -> int | None:
    """The writer's process id in a temp file name, or ``None`` for a name that is not one."""
    parts = name.rsplit(".", 3)
    if len(parts) != 4 or parts[3] != "tmp" or not parts[1].isdigit():
        return None
    return int(parts[1])


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
    for attempt in range(_WINDOWS_RETRIES + 1):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if sys.platform != "win32" or attempt == _WINDOWS_RETRIES:
                with contextlib.suppress(OSError):
                    temp.unlink()
                raise
            time.sleep(_WINDOWS_RETRY_WAIT)


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


def write_json_atomic(path: Path, data: dict[str, Any], *, private: bool = False) -> None:
    """Write *data* as JSON; refuses (:class:`NewerFormat`) to replace a file of a newer format."""
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

    @property
    def this_process(self) -> bool:
        return self.host == _HOST and self.pid == os.getpid()

    def alive(self) -> bool:
        """Whether the owner still runs.  A lock of another computer (a shared home folder) counts as alive."""
        if self.host != _HOST:
            return True
        return pid_alive(self.pid, self.started)


def read_owner(lock_path: Path) -> tuple[LockOwner | None, str | None]:
    """The owner of *lock_path* and the file's raw text; ``(None, None)`` when there is no lock file."""
    try:
        raw = lock_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, None
    except OSError:
        return None, ""
    try:
        data = json.loads(raw)
        owner = LockOwner(int(data["pid"]), str(data["host"]),
                          float(data["started"]) if data.get("started") is not None else None, str(data["token"]),
                          str(data.get("purpose") or "owner"))
    except (ValueError, KeyError, TypeError):
        # Being written right now (created but not yet filled), or damaged: the caller judges it by its age.
        return None, raw
    return owner, raw


def _age(path: Path) -> float | None:
    try:
        return time.time() - path.stat().st_mtime
    except FileNotFoundError:
        return None


def _break_stale(lock_path: Path, seen: str) -> None:
    """Remove *lock_path* if it still holds exactly *seen*, serialised with every other breaker of this lock."""
    marker = lock_path.with_name(lock_path.name + ".break")
    try:
        fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    except FileExistsError:
        age = _age(marker)
        if age is not None and age > _BREAK_STALE_AFTER:
            with contextlib.suppress(FileNotFoundError):
                marker.unlink()
        return
    except FileNotFoundError:
        return
    os.close(fd)
    try:
        try:
            current = lock_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        if current == seen:
            with contextlib.suppress(FileNotFoundError):
                lock_path.unlink()
    finally:
        with contextlib.suppress(FileNotFoundError):
            marker.unlink()


class PidLock:
    """A lock file that names the process holding it.

    :meth:`try_acquire` takes the lock if it is free or stale; *stale_after* (seconds) makes an old lock stale
    even when its process runs (``None``: only a gone process makes it stale).  *purpose* tells a process that
    finds the lock taken whether an owner holds it or a cleanup that lets go within moments.
    """

    def __init__(self, lock_path: Path, *, stale_after: float | None = None, purpose: str = "owner") -> None:
        self.path = lock_path
        self.stale_after = stale_after
        self.purpose = purpose
        self.token: str | None = None

    def _content(self) -> str:
        assert self.token is not None
        return json.dumps({"pid": os.getpid(), "host": _HOST, "started": _own_start(), "token": self.token,
                           "purpose": self.purpose})

    def owner(self) -> LockOwner | None:
        return read_owner(self.path)[0]

    def is_stale(self, owner: LockOwner | None, raw: str | None) -> bool:
        if raw is None:
            return False
        age = _age(self.path)
        if owner is None:
            # A lock file without a readable owner is being written (milliseconds) or was left half-written.
            return age is not None and age > 1.0
        if self.stale_after is not None and age is not None and age > self.stale_after:
            return True
        return not owner.alive()

    def try_acquire(self) -> bool:
        """Take the lock: ``True`` when this process now holds it."""
        for _ in range(2):
            self.token = secrets.token_hex(8)
            content = self._content().encode("utf-8")
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
            except FileExistsError:
                owner, raw = read_owner(self.path)
                if raw is not None and self.is_stale(owner, raw):
                    _break_stale(self.path, raw)
                    continue
                self.token = None
                return False
            except FileNotFoundError:
                continue        # the folder was removed between mkdir and open; once more
            try:
                os.write(fd, content)
            finally:
                os.close(fd)
            return True
        self.token = None
        return False

    def acquire(self, timeout: float = LOCK_TIMEOUT) -> None:
        deadline = time.monotonic() + timeout
        while not self.try_acquire():
            if time.monotonic() >= deadline:
                raise LockTimeout(self.path)
            time.sleep(_POLL)

    def held(self) -> bool:
        """Whether this object still holds its lock (the file holds its token)."""
        if self.token is None:
            return False
        owner = self.owner()
        return owner is not None and owner.token == self.token

    def release(self) -> None:
        """Remove the lock file if it is still this object's lock."""
        if self.token is None:
            return
        try:
            if self.held():
                self.path.unlink()
        except FileNotFoundError:
            pass
        finally:
            self.token = None


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
