"""An exclusive, cross-process file lock (``fcntl.flock`` on POSIX, ``msvcrt.locking`` on Windows).

The lock is taken on a sidecar file (for example ``<db>.lock``), never on a SQLite file itself: on Windows SQLite
keeps its own byte-range locks at fixed offsets of the database file (the pending byte at 1 GiB), and
``msvcrt.locking`` locks are mandatory, so a lock on the database file could block SQLite's own reads and writes.

``flock`` and ``msvcrt.locking`` locks belong to the open descriptor or handle, so a crash or a kill releases them.
On POSIX a lock is waited for in the kernel; on Windows it is polled, and *timeout* bounds the wait there.
"""
from __future__ import annotations

import errno
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

#: The ``msvcrt.locking`` errors that mean "another handle holds the lock"; any other error is not retried.
#: ``errno.EDEADLOCK`` exists on Windows and Linux only (macOS has just ``EDEADLK``), so it is looked up softly.
_WINDOWS_HELD = {errno.EACCES, getattr(errno, "EDEADLOCK", errno.EDEADLK)}
#: The poll interval of the Windows wait, in seconds.
_POLL_S = 0.05


class FileLockError(OSError):
    """A file lock could not be taken: the file system does not support it, or the wait timed out."""

    def __str__(self) -> str:
        return str(self.strerror)


class FileLockTimeout(FileLockError, TimeoutError):
    """Another process held the lock for longer than the timeout (Windows)."""


def _name(lock_fh) -> str:
    return str(getattr(lock_fh, "name", "the lock file"))


def _unsupported(lock_fh, exc: OSError) -> FileLockError:
    return FileLockError(
        exc.errno,
        f"Cannot lock {_name(lock_fh)}: the file system does not support file locks ({exc.strerror or exc}). "
        f"Copy the database to a local folder and use the copy.",
    )


def try_lock_exclusive(lock_fh) -> bool:
    """Take an exclusive lock on an open file handle without waiting; return whether it was taken.

    Raises :class:`FileLockError` when the file system does not support the lock.
    """
    try:
        import fcntl
    except ImportError:
        import msvcrt

        # ``msvcrt.locking`` locks ``nbytes`` from the current file position: lock a single byte at offset 0.
        lock_fh.seek(0)
        try:
            msvcrt.locking(lock_fh.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if exc.errno in _WINDOWS_HELD:
                return False
            raise _unsupported(lock_fh, exc) from exc
        return True
    try:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    except OSError as exc:
        if exc.errno in (errno.EAGAIN, errno.EACCES):
            return False
        raise _unsupported(lock_fh, exc) from exc
    return True


def lock_exclusive(lock_fh, *, timeout: float | None = None) -> None:
    """Take an exclusive lock on an open file handle, waiting for it.

    On POSIX the wait is in the kernel and unbounded.  On Windows the lock is polled (``LK_LOCK`` gives up after
    about 10 s), for at most *timeout* seconds when it is given; then :class:`FileLockTimeout` names the lock file.
    Raises :class:`FileLockError` when the file system does not support the lock.
    """
    try:
        import fcntl
    except ImportError:
        deadline = None if timeout is None else time.monotonic() + timeout
        while not try_lock_exclusive(lock_fh):
            if deadline is not None and time.monotonic() >= deadline:
                raise FileLockTimeout(
                    errno.ETIMEDOUT,
                    f"Another process held the lock {_name(lock_fh)} for more than {timeout:g} s. If no other "
                    f"FlexTool is running on this database, delete the lock file and try again.",
                ) from None
            time.sleep(_POLL_S)
        return
    try:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
    except OSError as exc:
        raise _unsupported(lock_fh, exc) from exc


def unlock(lock_fh) -> None:
    """Release a lock taken by :func:`lock_exclusive` or :func:`try_lock_exclusive` (portable)."""
    try:
        import fcntl
    except ImportError:
        import msvcrt

        lock_fh.seek(0)
        try:
            msvcrt.locking(lock_fh.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
    else:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


@contextmanager
def exclusive_file_lock(
    lock_path: str | os.PathLike,
    *,
    timeout: float | None = None,
    on_wait: Callable[[], None] | None = None,
) -> Iterator[None]:
    """Hold an exclusive lock on *lock_path* (created if missing, left in place) for the ``with`` block.

    When the lock is not free at once, *on_wait* is called once before waiting.  *timeout* bounds the wait on
    Windows (see :func:`lock_exclusive`).
    """
    with open(lock_path, "a") as lock_fh:
        if not try_lock_exclusive(lock_fh):
            if on_wait is not None:
                on_wait()
            lock_exclusive(lock_fh, timeout=timeout)
        try:
            yield
        finally:
            unlock(lock_fh)
