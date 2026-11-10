"""Migrate an input database before a model run, safely when several runs start on the same file.

``flextool-run --migrate`` (and ``execute_flextool_workflow.py``) call :func:`migrate_before_run` before the engine
starts.  Runs are often launched in parallel on one database, so the migration takes an exclusive lock on the sidecar
file ``<db>.migrate.lock`` and reads the version again once it holds the lock: the first run migrates, the others wait
for it and then find the database current.  The data version and the SpineDB schema revision are read read-only, so a
database that is already current is never opened for writing (and never waits for the lock).

The frozen guard of :func:`~flextool.update_flextool.db_migration.migrate_database` applies: a frozen database older
than this FlexTool raises :class:`~flextool.update_flextool.frozen.FrozenDatabaseError`, and the run stops.
"""
from __future__ import annotations

from pathlib import Path

from flextool._file_lock import exclusive_file_lock

#: The suffix of the lock file next to the database.
LOCK_SUFFIX = ".migrate.lock"


def migration_lock_path(path: Path) -> Path:
    """The sidecar lock file of the database at *path*."""
    return path.with_name(path.name + LOCK_SUFFIX)


def migrate_before_run(database: str) -> int | None:
    """Migrate the local sqlite database *database* (URL or path) if it is not current; return the version it had.

    A database at the current data version whose SpineDB schema is an older revision is upgraded too.  Returns
    ``None`` when nothing was migrated: the database is current (or newer, which the engine reports), another
    run migrated it while this one waited for the lock, or *database* is not a local sqlite file (the engine reports
    a missing file or another URL).  Raises :class:`~flextool.update_flextool.frozen.FrozenDatabaseError` for a frozen
    database that would have to be migrated.
    """
    from flextool.update_flextool.frozen import frozen_guard, sqlite_path_from

    path = sqlite_path_from(database)
    if path is None or not path.is_file():
        return None
    path = path.resolve()
    if _needed(path) is None:
        return None
    # Refuse a frozen database before waiting for the lock (migrate_database checks again).
    frozen_guard(path)
    with exclusive_file_lock(migration_lock_path(path)):
        return _migrate_locked(path)


def _needed(path: Path) -> str | None:
    """What *path* needs, read read-only: ``"data"`` (an older data version), ``"schema"`` (a current data version
    with an older SpineDB schema) or ``None`` (current, newer, or not a FlexTool database: the engine reports those).
    """
    from flextool.update_flextool import FLEXTOOL_DB_VERSION
    from flextool.update_flextool.frozen import read_db_version_ro, schema_outdated_ro

    version = read_db_version_ro(path)
    if version is None or version > FLEXTOOL_DB_VERSION:
        return None
    if version < FLEXTOOL_DB_VERSION:
        return "data"
    return "schema" if schema_outdated_ro(path) else None


def _migrate_locked(path: Path) -> int | None:
    """The migration itself, run while holding the lock: the version is read again, as another run may have done it."""
    from flextool.update_flextool import FLEXTOOL_DB_VERSION
    from flextool.update_flextool.db_migration import migrate_database
    from flextool.update_flextool.frozen import read_db_version_ro

    needed = _needed(path)
    if needed is None:
        print(f"  The input database was migrated by another run: {path}", flush=True)
        return None
    version = read_db_version_ro(path)
    if needed == "schema":
        print(f"  Upgrading the SpineDB schema of the input database: {path}", flush=True)
    else:
        print(f"  Migrating the input database from version {version} to {FLEXTOOL_DB_VERSION}: {path}", flush=True)
    migrate_database(f"sqlite:///{path}")
    return version
