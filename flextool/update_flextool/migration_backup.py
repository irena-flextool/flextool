"""The pre-migration backup of a database file: a migration either completes or leaves the file as it was found.

A migration commits step by step and stamps the data version only at the end, so a failure part way would leave a
half-migrated file.  :func:`restored_on_failure` copies the file aside before the first write and copies it back on
any exception (a cancellation too).  :func:`~flextool.update_flextool.db_migration.migrate_database` and the
window's upgrade (:mod:`flextool.gui.db_version_check`) both use it.
"""
from __future__ import annotations

import contextlib
import logging
import shutil
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)


def backup_path(db_path: Path) -> Path:
    """The sidecar path that holds the pre-migration backup of *db_path*."""
    return db_path.with_name(db_path.name + ".premigration.bak")


def restore_from_backup(db_path: Path, backup: Path) -> bool:
    """Restore *db_path* from *backup*; ``True`` on success.

    Removes any stale SQLite ``-wal`` / ``-shm`` sidecars first so the restored main file is not shadowed by a
    write-ahead log left behind by the aborted migration.
    """
    try:
        for suffix in ("-wal", "-shm"):
            side = db_path.with_name(db_path.name + suffix)
            if side.exists():
                side.unlink()
        shutil.copy2(backup, db_path)
        return True
    except Exception:
        logger.error("Failed to restore %s from backup %s", db_path, backup, exc_info=True)
        return False


def remove_backup(backup: Path) -> None:
    """Delete *backup* (a failure to delete is only logged)."""
    try:
        backup.unlink(missing_ok=True)
    except Exception:
        logger.debug("Could not remove migration backup %s", backup, exc_info=True)


class MigrationRestoreError(RuntimeError):
    """A migration failed and its database could not be restored from the pre-migration backup."""


@contextlib.contextmanager
def restored_on_failure(db_path: Path) -> Iterator[None]:
    """Copy *db_path* to :func:`backup_path` before the body runs; on any exception from the body copy it back and
    re-raise (the database is as it was found), else keep the migrated file.  The copy is deleted either way, except
    when the restore fails: then it is kept and :class:`MigrationRestoreError` (naming it) is raised from the
    original error.  A failure to make the copy raises before the body runs, so the file is never touched without
    one."""
    backup = backup_path(db_path)
    try:
        shutil.copy2(db_path, backup)
    except BaseException:
        remove_backup(backup)
        raise
    logger.info("Pre-migration backup written: %s", backup)
    try:
        yield
    except BaseException as exc:
        if not restore_from_backup(db_path, backup):
            raise MigrationRestoreError(
                f"The migration of {db_path} failed and the database could not be restored; its original is kept "
                f"at {backup}") from exc
        print(f"{db_path}: the migration did not complete; the database was restored to its state before it",
              flush=True)
        remove_backup(backup)
        raise
    remove_backup(backup)
