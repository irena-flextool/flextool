"""Database version checking and upgrade utilities for FlexTool GUI.

Checks both the SpineDB API schema version and the FlexTool data version,
upgrading automatically when needed.  All errors are caught and returned
as human-readable messages so the GUI never crashes.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Callable

from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool.db_migration import MigrationCancelled
from flextool.sqlite_url import sqlite_url_string

logger = logging.getLogger(__name__)

ISSUE_TRACKER_URL = "https://github.com/irena-flextool/flextool/issues"


def _backup_path(db_path: Path) -> Path:
    """Return the sidecar path used to hold the pre-migration backup."""
    return db_path.with_name(db_path.name + ".premigration.bak")


def _restore_from_backup(db_path: Path, backup: Path) -> bool:
    """Restore *db_path* from *backup*, returning ``True`` on success.

    Removes any stale SQLite ``-wal`` / ``-shm`` sidecars first so the
    restored main file is not shadowed by a write-ahead log left behind
    by the aborted migration.
    """
    try:
        for suffix in ("-wal", "-shm"):
            side = db_path.with_name(db_path.name + suffix)
            if side.exists():
                side.unlink()
        shutil.copy2(backup, db_path)
        return True
    except Exception:
        logger.error(
            "Failed to restore %s from backup %s", db_path, backup, exc_info=True
        )
        return False


def _read_flextool_version(db_url: str) -> int | None:
    """Read the current FlexTool data version from a database.

    Reads with sqlite3 in read-only mode, so probing a file never upgrades its
    SpineDB schema or writes to it (an external or frozen source is only
    changed after the user agrees).  Returns the integer version, or ``None``
    if it cannot be determined.
    """
    from flextool.update_flextool.frozen import read_db_version_ro, sqlite_path_from

    path = sqlite_path_from(db_url)
    if path is None:
        return None
    return read_db_version_ro(path)


def get_target_flextool_version() -> int:
    """Return the FlexTool DB version this build migrates to."""
    return int(FLEXTOOL_DB_VERSION)


def needs_flextool_migration(db_path: Path) -> bool | None:
    """Return True if this file's FlexTool data version is below the target,
    or it is at the target but its SpineDB schema is an older revision.

    Both are read read-only.  Returns ``None`` if the version cannot be
    determined (file unreadable, not a FlexTool DB, etc.).
    """
    from flextool.update_flextool.frozen import schema_outdated_ro

    db_url = sqlite_url_string(db_path)
    current = _read_flextool_version(db_url)
    if current is None:
        return None
    if current < FLEXTOOL_DB_VERSION:
        return True
    return current == FLEXTOOL_DB_VERSION and schema_outdated_ro(db_path)


def check_and_upgrade_database(
    db_path: Path,
    *,
    progress_callback: Callable[[int, int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    allow_frozen: bool = False,
) -> tuple[bool, bool, list[str]]:
    """Check and upgrade a FlexTool database if needed.

    Performs two levels of upgrade:

    1. **SpineDB API schema upgrade** -- handled automatically by opening
       the database with ``DatabaseMapping(url, upgrade=True)``.
    2. **FlexTool data version upgrade** -- delegates to
       :func:`~flextool.update_flextool.db_migration.migrate_database`.

    Both steps mutate the file in place (migration commits per step, while
    the version stamp is written only at the very end), so before the first
    mutating operation a byte-for-byte backup copy is taken.  On success the
    backup is deleted; if the migration is cancelled or fails, the backup is
    restored so the database is always left either fully upgraded or exactly
    as it was found.  The backup is taken lazily — a database that needs no
    upgrade is never copied.

    Args:
        db_path: Path to the ``.sqlite`` file.
        progress_callback: Optional callable forwarded to
            :func:`migrate_database`.  Invoked before each migration
            step with ``(current_version, target_version, next_version)``.
        cancel_check: Optional callable forwarded to
            :func:`migrate_database`.  When it returns ``True``, the
            migration stops cleanly between steps, the database is restored
            to its original state, and this function returns instead of
            raising.
        allow_frozen: Migrate a frozen database in place (the user agreed);
            its marker is kept with the new version.  Without it, a frozen
            database is never opened for writing: one that needs an upgrade
            is reported as failed, one that is current is left as it is.

    Returns:
        A ``(was_upgraded, failed, messages)`` tuple where *was_upgraded* is
        ``True`` if any upgrade was performed and persisted, *failed* is
        ``True`` only if a migration error left work that had to be rolled
        back (the caller should not proceed to use the database), and
        *messages* is a list of human-readable descriptions of what happened.

    This function never raises -- all exceptions are caught and reported
    as messages.
    """
    messages: list[str] = []
    was_upgraded = False
    failed = False

    try:
        from flextool.update_flextool.db_handle import open_database_mapping  # noqa: F401  (availability check)
    except ImportError as exc:
        messages.append(f"Cannot check database version (spinedb_api not available): {exc}")
        return was_upgraded, failed, messages

    # The frozen check reads the file read-only, before anything below can
    # open it with ``upgrade=True``.  A marker that cannot be read now (another
    # migration is committing) is checked again under the migration lock.
    from flextool._file_lock import FileLockError
    from flextool.update_flextool.frozen import FrozenCheckError, FrozenDatabaseError, frozen_guard, is_current_ro
    from flextool.update_flextool.migrate_before_run import migration_lock

    def _guard() -> bool:
        """Whether the migration may go on; adds the frozen refusal to *messages*."""
        nonlocal failed
        try:
            frozen_guard(db_path, allow_frozen=allow_frozen)
        except FrozenDatabaseError as exc:
            if not exc.up_to_date:
                messages.append(str(exc))
                failed = True
            return False
        return True

    try:
        frozen_guard(db_path, allow_frozen=allow_frozen)
    except FrozenCheckError:
        pass
    except FrozenDatabaseError:
        _guard()
        return was_upgraded, failed, messages
    # A current database is never opened for writing and never waits for the lock.
    if is_current_ro(db_path):
        return was_upgraded, failed, messages

    # Every in-place migration of the file holds ``<db>.migrate.lock``
    # (shared with flextool-run --migrate and flextool-migrate-database); the
    # state is read again under it, as another process may have migrated it.
    try:
        with migration_lock(Path(db_path)):
            if not _guard():
                return was_upgraded, failed, messages
            return _upgrade_locked(db_path, messages, progress_callback, cancel_check, allow_frozen)
    except FileLockError as exc:
        messages.append(f"{Path(db_path).name}: not migrated: {exc}")
        return was_upgraded, True, messages


def _upgrade_locked(
    db_path: Path,
    messages: list[str],
    progress_callback: Callable[[int, int, int], None] | None,
    cancel_check: Callable[[], bool] | None,
    allow_frozen: bool,
) -> tuple[bool, bool, list[str]]:
    """The upgrade of :func:`check_and_upgrade_database`, run while holding the migration lock."""
    from flextool.update_flextool.db_handle import open_database_mapping

    was_upgraded = False
    failed = False
    db_url = sqlite_url_string(db_path)

    # Probe the schema read-only first so a database that is already current
    # is never copied.  An out-of-date schema makes this raise.
    try:
        with open_database_mapping(db_url, create=False, upgrade=False):
            pass
        schema_current = True
    except Exception:
        schema_current = False

    backup: Path | None = None

    def _ensure_backup() -> bool:
        """Take the pre-migration backup once; return ``True`` if available."""
        nonlocal backup
        if backup is not None:
            return True
        candidate = _backup_path(db_path)
        try:
            shutil.copy2(db_path, candidate)
            backup = candidate
            logger.info("Pre-migration backup written: %s", candidate)
            return True
        except Exception:
            logger.error("Could not create migration backup for %s", db_path, exc_info=True)
            return False

    try:
        # ── Step 1: SpineDB API schema upgrade ─────────────────────
        if not schema_current:
            if not _ensure_backup():
                messages.append(
                    f"{db_path.name}: could not create a safety backup before "
                    f"upgrading; the database was left untouched."
                )
                return was_upgraded, True, messages
            with open_database_mapping(db_url, create=False, upgrade=True):
                pass
            messages.append(f"{db_path.name}: SpineDB schema upgraded to latest version.")
            was_upgraded = True
            logger.info("SpineDB schema upgraded for %s", db_path)

        # ── Step 2: FlexTool data version upgrade ──────────────────
        version_before = _read_flextool_version(db_url)
        if version_before is not None and version_before < FLEXTOOL_DB_VERSION:
            if not _ensure_backup():
                messages.append(
                    f"{db_path.name}: could not create a safety backup before "
                    f"migrating; the database was left untouched."
                )
                return was_upgraded, True, messages

            from flextool.update_flextool.db_migration import migrate_database

            migrate_database(
                str(db_path),
                progress_callback=progress_callback,
                cancel_check=cancel_check,
                allow_frozen=allow_frozen,
            )

            version_after = _read_flextool_version(db_url)
            if version_after is not None and version_after > version_before:
                messages.append(
                    f"{db_path.name}: FlexTool data upgraded from version "
                    f"{version_before} to {version_after}."
                )
                was_upgraded = True
                logger.info(
                    "FlexTool data upgraded %s: v%s -> v%s",
                    db_path,
                    version_before,
                    version_after,
                )

    except MigrationCancelled:
        was_upgraded = False
        if backup is not None and _restore_from_backup(db_path, backup):
            messages.append(
                f"{db_path.name}: migration cancelled — the database was "
                f"restored to its original state. Re-run to migrate it."
            )
        else:
            failed = True
            messages.append(
                f"{db_path.name}: migration cancelled, but the database could "
                f"NOT be restored automatically. Do not use it; restore it "
                f"from your own backup and report this at {ISSUE_TRACKER_URL}."
            )
        logger.info("FlexTool migration cancelled for %s", db_path)

    except Exception as exc:
        import traceback as _tb
        tb_text = _tb.format_exc()
        was_upgraded = False
        failed = True
        if backup is not None and _restore_from_backup(db_path, backup):
            messages.append(
                f"{db_path.name}: migration FAILED — the database was restored "
                f"to its original state, so it is safe to keep using the "
                f"unmigrated copy. Please report this at {ISSUE_TRACKER_URL}: "
                f"copy the traceback below and, if possible, share the database "
                f"file.\n\nError: {exc}\n\nTraceback:\n{tb_text}"
            )
        else:
            messages.append(
                f"{db_path.name}: migration FAILED and the database could NOT be "
                f"restored automatically. Do not use it; restore it from your "
                f"own backup. Please report this at {ISSUE_TRACKER_URL}: copy the "
                f"traceback below and, if possible, share the database file."
                f"\n\nError: {exc}\n\nTraceback:\n{tb_text}"
            )
        logger.warning("FlexTool migration failed for %s: %s", db_path, exc, exc_info=True)

    finally:
        if backup is not None:
            try:
                backup.unlink(missing_ok=True)
            except Exception:
                logger.debug("Could not remove migration backup %s", backup, exc_info=True)

    return was_upgraded, failed, messages
