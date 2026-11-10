"""A blocking, exclusive, cross-process file lock (``fcntl.flock`` on POSIX, ``msvcrt.locking`` on Windows).

The lock is taken on a sidecar file (for example ``<db>.lock``), never on a SQLite file itself: SQLite holds its own
byte-range locks on the database, and a Windows ``msvcrt.locking`` on it would clash with them.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager


def lock_exclusive(lock_fh) -> None:
    """Acquire a blocking exclusive lock on an open file handle.

    Portable across POSIX (``fcntl.flock``) and Windows (``msvcrt.locking``);
    ``fcntl`` does not exist on Windows, so dispatch on what imports."""
    try:
        import fcntl
    except ImportError:
        import msvcrt
        import time

        # ``msvcrt.locking`` locks ``nbytes`` from the current file position;
        # lock a single byte at offset 0.  Use the non-blocking variant in a
        # retry loop so we block indefinitely rather than giving up after the
        # ~10s LK_LOCK ceiling.
        lock_fh.seek(0)
        while True:
            try:
                msvcrt.locking(lock_fh.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                time.sleep(0.05)
    else:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)


def unlock(lock_fh) -> None:
    """Release a lock taken by :func:`lock_exclusive` (portable)."""
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
def exclusive_file_lock(lock_path: str | os.PathLike) -> Iterator[None]:
    """Hold an exclusive lock on *lock_path* (created if missing, left in place) for the ``with`` block."""
    with open(lock_path, "a") as lock_fh:
        lock_exclusive(lock_fh)
        try:
            yield
        finally:
            unlock(lock_fh)
