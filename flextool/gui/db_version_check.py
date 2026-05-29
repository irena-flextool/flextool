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

    Returns the integer version, or ``None`` if it cannot be determined.
    """
    try:
        from spinedb_api import DatabaseMapping, from_database

        with DatabaseMapping(db_url, create=False, upgrade=True) as db:
            sq = db.object_parameter_definition_sq
            settings_param = (
                db.query(sq)
                .filter(sq.c.object_class_name == "model")
                .filter(sq.c.parameter_name == "version")
                .one_or_none()
            )
            if settings_param is None:
                return 0
            return int(
                from_database(settings_param.default_value, settings_param.default_type)
            )
    except Exception:
        logger.debug("Could not read FlexTool version from %s", db_url, exc_info=True)
        return None


def get_target_flextool_version() -> int:
    """Return the FlexTool DB version this build migrates to."""
    return int(FLEXTOOL_DB_VERSION)


def needs_flextool_migration(db_path: Path) -> bool | None:
    """Return True if this file's FlexTool data version is below the target.

    Returns ``None`` if the version cannot be determined (file unreadable,
    not a FlexTool DB, etc.).
    """
    db_url = f"sqlite:///{db_path}"
    current = _read_flextool_version(db_url)
    if current is None:
        return None
    return current < FLEXTOOL_DB_VERSION


def check_and_upgrade_database(
    db_path: Path,
    *,
    progress_callback: Callable[[int, int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
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
        from spinedb_api import DatabaseMapping
    except ImportError as exc:
        messages.append(f"Cannot check database version (spinedb_api not available): {exc}")
        return was_upgraded, failed, messages

    db_url = f"sqlite:///{db_path}"

    # Probe the schema read-only first so a database that is already current
    # is never copied.  An out-of-date schema makes this raise.
    try:
        with DatabaseMapping(db_url, create=False, upgrade=False):
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
            with DatabaseMapping(db_url, create=False, upgrade=True):
                pass
            messages.append(f"{db_path.name}: SpineDB schema upgraded to latest version.")
            was_upgraded = True
            logger.info("SpineDB schema upgraded for %s", db_path)

        # ── Step 2: FlexTool data version upgrade ──────────────────
        # Schema is current now, so reading the version does not mutate.
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
