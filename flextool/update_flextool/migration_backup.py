"""The pre-migration backup of a database file: a migration either completes or leaves the file as it was found.

A migration commits step by step and stamps the data version only at the end, so a failure part way would leave a
half-migrated file.  :func:`restored_on_failure` copies the file to ``<db>.premigration.bak`` (beside it, as large as
it) before the first write and, on any exception (a cancellation or Ctrl+C too), moves that copy back over the file
in one atomic rename.  :func:`~flextool.update_flextool.db_migration.migrate_database` and the window's upgrade
(:mod:`flextool.gui.db_version_check`) both use it.

Whenever the copy cannot be put back (a failure or an interruption during the restore), it is kept and its path is
printed with how to recover the original by hand.  A copy left from such a migration is never overwritten: a new
migration of the file refuses to start until it is dealt with.
"""
from __future__ import annotations

import contextlib
import errno
import logging
import os
import shutil
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

#: Windows refuses a rename over a file another program holds open; only there is the restore a copy instead.
_WINDOWS = os.name == "nt"


class MigrationBackupError(RuntimeError):
    """The pre-migration copy could not be made, so the database was not migrated (and is unchanged)."""


class MigrationRestoreError(RuntimeError):
    """A migration failed and its database could not be restored from the pre-migration copy (which is kept)."""


def backup_path(db_path: Path) -> Path:
    """The sidecar path that holds the pre-migration copy of *db_path*."""
    return db_path.with_name(db_path.name + ".premigration.bak")


def recovery_text(db_path: Path, backup: Path) -> str:
    """How to recover the original of *db_path* from its kept pre-migration copy *backup*."""
    return (f"The database {db_path} may be half migrated. Its original is kept at {backup}. To recover it, close "
            f"every program that uses the database, delete {db_path.name} and rename {backup.name} to "
            f"{db_path.name} (both in {db_path.parent}).")


def _made_on(backup: Path) -> str:
    """When *backup* was written (its creation time; its modification time is the original's, kept by the copy)."""
    try:
        st = backup.stat()
    except OSError:
        return "at an unknown time"
    made = getattr(st, "st_birthtime", None) or st.st_ctime
    return datetime.fromtimestamp(made).strftime("on %Y-%m-%d %H:%M")


def kept_copy_error(db_path: Path, backup: Path) -> MigrationBackupError:
    """The refusal to migrate *db_path* while the pre-migration copy *backup* of an earlier migration is there."""
    return MigrationBackupError(
        f"{db_path} was not migrated: the pre-migration copy {backup.name}, made {_made_on(backup)} by an earlier "
        f"migration, is still there (in {db_path.parent}). If that migration did not complete (it was stopped or "
        f"its restore failed), the copy is the original and the database may be half migrated: close every program "
        f"that uses the database, delete {db_path.name} and rename {backup.name} to {db_path.name}. If that "
        f"migration completed and the database has been used since, the copy is out of date: delete "
        f"{backup.name}. Then migrate again.")


def refuse_kept_copy(db_path: Path) -> None:
    """Raise :func:`kept_copy_error` when a pre-migration copy of *db_path* is still there: it may be the only intact
    original, so nothing may migrate (or overwrite it) until the user has put it back or deleted it."""
    backup = backup_path(db_path)
    if backup.exists():
        raise kept_copy_error(db_path, backup)


def make_backup(db_path: Path) -> Path:
    """Copy *db_path* to :func:`backup_path` and return that path.

    Raises :class:`MigrationBackupError` (the file is unchanged, no partial copy is left) when a copy from an earlier
    migration is still there (:func:`refuse_kept_copy`), when the disk has no room for the copy, or when it cannot be
    written.
    """
    backup = backup_path(db_path)
    refuse_kept_copy(db_path)
    size = db_path.stat().st_size
    try:
        free = shutil.disk_usage(db_path.parent).free
    except OSError:
        free = None
    if free is not None and free < size:
        raise MigrationBackupError(
            f"{db_path} was not migrated: there is no room for its pre-migration copy ({size / 1e6:.1f} MB needed "
            f"beside it, {free / 1e6:.1f} MB free in {db_path.parent}). Free some space and migrate again.")
    try:
        shutil.copy2(db_path, backup)
    except BaseException as exc:
        with contextlib.suppress(OSError):
            backup.unlink(missing_ok=True)
        if isinstance(exc, OSError):
            if exc.errno in (errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)):
                reason = f"there is no room for its pre-migration copy ({size / 1e6:.1f} MB needed beside it)"
            else:
                reason = f"its pre-migration copy {backup} could not be written ({exc.strerror or exc})"
            raise MigrationBackupError(f"{db_path} was not migrated: {reason}. The database is unchanged.") from exc
        raise
    logger.info("Pre-migration backup written: %s", backup)
    return backup


def restore_from_backup(db_path: Path, backup: Path) -> bool:
    """Put the pre-migration copy *backup* back as *db_path*; ``True`` on success (the copy is then gone).

    The copy replaces the file in one rename (``os.replace``: atomic, needs no space), so the file is either the
    migrated one or the original, never a mix.  Only where Windows refuses the rename because another program holds
    the file is the copy written over it instead (said on the console); the copy is deleted only after that write
    has completed.  Stale SQLite ``-wal`` / ``-shm`` sidecars are removed first so the restored file is not shadowed
    by a write-ahead log of the aborted migration.  A failure is logged and gives ``False`` (the copy is kept); an
    interruption propagates (the copy is kept unless the rename completed).
    """
    try:
        for suffix in ("-wal", "-shm"):
            side = db_path.with_name(db_path.name + suffix)
            if side.exists():
                side.unlink()
        try:
            os.replace(backup, db_path)
        except PermissionError:
            if not _WINDOWS:
                raise
            print(f"{db_path} is open in another program, so it is restored by copying {backup} over it (not "
                  "atomic: if this is interrupted, recover the original from the copy).", flush=True)
            shutil.copy2(backup, db_path)
            remove_backup(backup)
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


def _report_kept_backup(db_path: Path, backup: Path) -> None:
    """Print where the original is, when the pre-migration copy is still there."""
    if backup.exists():
        print(recovery_text(db_path, backup), flush=True)


@contextlib.contextmanager
def restored_on_failure(db_path: Path) -> Iterator[None]:
    """Run the body with a pre-migration copy of *db_path* (:func:`make_backup`): on any exception from the body, put
    the copy back (:func:`restore_from_backup`) and re-raise, so the database is as it was found; else delete the copy.

    When the copy cannot be put back, it is kept, :func:`recovery_text` is printed and :class:`MigrationRestoreError`
    is raised from the original error.  An interruption during the restore also prints it (when the copy is still
    there) and propagates.
    """
    backup = make_backup(db_path)
    try:
        yield
    except BaseException as exc:
        try:
            restored = restore_from_backup(db_path, backup)
            if restored:
                print(f"{db_path}: the migration did not complete; the database was restored to its state before "
                      "it", flush=True)
        except BaseException:
            _report_kept_backup(db_path, backup)
            raise
        if not restored:
            _report_kept_backup(db_path, backup)
            raise MigrationRestoreError(
                f"The migration of {db_path} failed and the database could not be restored. "
                f"{recovery_text(db_path, backup)}") from exc
        raise
    remove_backup(backup)
