"""Migrate an input database before a model run, safely when several runs start on the same file.

``flextool-run --migrate`` (and ``execute_flextool_workflow.py``) call :func:`migrate_before_run` before the engine
starts.  Runs are often launched in parallel on one database, so every in-place migration of a local sqlite file
(:func:`~flextool.update_flextool.db_migration.migrate_database`, and through it ``flextool-migrate-database``, the
Toolbox **Migrate database** tool and the FlexTool window's refresh) holds :func:`migration_lock`, an exclusive lock
on the sidecar file ``<db>.migrate.lock``, and reads the version again once it holds the lock: the first migrator
migrates, the others wait for it and then find the database current.  A database that reads as current is never
opened for writing and never waits for the lock; one that cannot be read (another migration is committing) is
decided under the lock.

The frozen guard of :func:`~flextool.update_flextool.db_migration.migrate_database` applies: a frozen database older
than this FlexTool raises :class:`~flextool.update_flextool.frozen.FrozenDatabaseError`, and the run stops.
"""
from __future__ import annotations

import errno
import importlib.util
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from flextool._file_lock import FileLockError, exclusive_file_lock
from flextool.sqlite_url import sqlite_url_string

#: The suffix of the lock file next to the database.
LOCK_SUFFIX = ".migrate.lock"
#: The environment variable that sets how long (seconds) a migration waits for the lock on Windows.
TIMEOUT_ENV = "FLEXTOOL_MIGRATION_LOCK_TIMEOUT"
#: The default of :data:`TIMEOUT_ENV`.
DEFAULT_TIMEOUT_S = 600.0

_held = threading.local()
#: Whether the lock is polled (Windows, no ``fcntl``); only then is :data:`TIMEOUT_ENV` read.
_POLLED = importlib.util.find_spec("fcntl") is None


def migration_lock_path(path: Path) -> Path:
    """The sidecar lock file of the database at *path*."""
    return path.with_name(path.name + LOCK_SUFFIX)


def lock_timeout() -> float:
    """The wait for the migration lock on Windows, seconds (:data:`TIMEOUT_ENV`, default 10 minutes).

    Read only where the lock is polled (Windows): on POSIX the kernel waits and the variable is not read, so a bad
    value there does not stop a migration.
    """
    text = os.environ.get(TIMEOUT_ENV, "").strip()
    if not text:
        return DEFAULT_TIMEOUT_S
    try:
        value = float(text)
    except ValueError:
        value = 0.0
    if not value > 0:
        raise FileLockError(errno.EINVAL, f"{TIMEOUT_ENV} must be a positive number of seconds, not {text!r}")
    return value


@contextmanager
def migration_lock(path: Path) -> Iterator[None]:
    """Hold the migration lock of the local sqlite database *path* (re-entrant within a thread).

    Prints one ``Waiting for another migration of <db> ...`` line when the lock is not free at once.  Raises
    :class:`~flextool._file_lock.FileLockError` when the lock cannot be taken (no file-lock support, or the Windows
    wait timed out).
    """
    key = str(path.resolve())
    held: set[str] = _held.__dict__.setdefault("paths", set())
    if key in held:
        yield
        return

    def _wait() -> None:
        print(f"  Waiting for another migration of {key} ...", flush=True)

    timeout = lock_timeout() if _POLLED else None
    with exclusive_file_lock(migration_lock_path(Path(key)), timeout=timeout, on_wait=_wait):
        held.add(key)
        try:
            yield
        finally:
            held.discard(key)


def migrate_before_run(database: str) -> int | None:
    """Migrate the local sqlite database *database* (URL or path) if it is not current; return the version it had.

    A database at the current data version whose SpineDB schema is an older revision is upgraded too.  Returns
    ``None`` when nothing was migrated: the database is current (or newer, which the engine reports), another
    run migrated it while this one waited for the lock, or *database* is not a local sqlite file (the engine reports
    a missing file or another URL).  Raises :class:`~flextool.update_flextool.frozen.FrozenDatabaseError` for a frozen
    database that would have to be migrated, and :class:`~flextool._file_lock.FileLockError` when the lock cannot be
    taken.
    """
    from flextool.update_flextool.frozen import FrozenCheckError, frozen_guard, sqlite_path_from

    path = sqlite_path_from(database)
    if path is None or not path.is_file():
        return None
    path = path.resolve()
    needed, _version = _needed(path)
    if needed is None:
        return None
    # Refuse a frozen database before waiting for the lock (migrate_database checks again).  A marker that cannot be
    # read now (another migration is committing) is checked again under the lock.
    try:
        frozen_guard(path)
    except FrozenCheckError:
        pass
    with migration_lock(path):
        frozen_guard(path)
        return _migrate_locked(path)


def _needed(path: Path) -> tuple[str | None, int | None]:
    """What *path* needs, read read-only, and its data version.

    ``"data"`` (an older data version), ``"schema"`` (a current data version with an older SpineDB schema),
    ``"unknown"`` (the version could not be read: decide under the lock) or ``None`` (current or newer: the engine
    reports a newer one).
    """
    from flextool.update_flextool import FLEXTOOL_DB_VERSION
    from flextool.update_flextool.frozen import read_db_version_ro, schema_outdated_ro

    version = read_db_version_ro(path)
    if version is None:
        return "unknown", None
    if version > FLEXTOOL_DB_VERSION:
        return None, version
    if version < FLEXTOOL_DB_VERSION:
        return "data", version
    return ("schema" if schema_outdated_ro(path) else None), version


def _migrate_locked(path: Path) -> int | None:
    """The migration itself, run while holding the lock: the version is read again, as another run may have done it."""
    from flextool.update_flextool import FLEXTOOL_DB_VERSION
    from flextool.update_flextool.db_migration import migrate_database

    needed, version = _needed(path)
    if needed is None:
        if version == FLEXTOOL_DB_VERSION:
            print(f"  The input database was migrated by another run: {path}", flush=True)
        return None
    if needed == "unknown":
        # Not a FlexTool database (or unreadable with the lock held): the engine reports it.
        print(f"  Could not read the FlexTool version of the input database; not migrated: {path}", flush=True)
        return None
    if needed == "schema":
        print(f"  Upgrading the SpineDB schema of the input database: {path}", flush=True)
    else:
        print(f"  Migrating the input database from version {version} to {FLEXTOOL_DB_VERSION}: {path}", flush=True)
    migrate_database(sqlite_url_string(path))
    return version
