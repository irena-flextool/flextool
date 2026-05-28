"""Database version checking and upgrade utilities for FlexTool GUI.

Checks both the SpineDB API schema version and the FlexTool data version,
upgrading automatically when needed.  All errors are caught and returned
as human-readable messages so the GUI never crashes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool.db_migration import MigrationCancelled

logger = logging.getLogger(__name__)


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
) -> tuple[bool, list[str]]:
    """Check and upgrade a FlexTool database if needed.

    Performs two levels of upgrade:

    1. **SpineDB API schema upgrade** -- handled automatically by opening
       the database with ``DatabaseMapping(url, upgrade=True)``.
    2. **FlexTool data version upgrade** -- delegates to
       :func:`~flextool.update_flextool.db_migration.migrate_database`.

    Args:
        db_path: Path to the ``.sqlite`` file.
        progress_callback: Optional callable forwarded to
            :func:`migrate_database`.  Invoked before each migration
            step with ``(current_version, target_version, next_version)``.
        cancel_check: Optional callable forwarded to
            :func:`migrate_database`.  When it returns ``True``, the
            migration stops cleanly between steps and this function
            returns a "cancelled" message instead of raising.

    Returns:
        A ``(was_upgraded, messages)`` tuple where *was_upgraded* is ``True``
        if any upgrades were performed and *messages* is a list of
        human-readable descriptions of what happened.

    This function never raises -- all exceptions are caught and reported
    as messages.
    """
    messages: list[str] = []
    was_upgraded = False

    try:
        from spinedb_api import DatabaseMapping
    except ImportError as exc:
        messages.append(f"Cannot check database version (spinedb_api not available): {exc}")
        return was_upgraded, messages

    db_url = f"sqlite:///{db_path}"

    # ── Step 1: SpineDB API schema upgrade ─────────────────────────
    try:
        try:
            with DatabaseMapping(db_url, create=False, upgrade=False):
                pass  # Schema is already current
        except Exception:
            # Schema needs upgrading -- reopen with upgrade=True
            try:
                with DatabaseMapping(db_url, create=False, upgrade=True):
                    pass
                messages.append(f"{db_path.name}: SpineDB schema upgraded to latest version.")
                was_upgraded = True
                logger.info("SpineDB schema upgraded for %s", db_path)
            except Exception as exc:
                messages.append(f"{db_path.name}: SpineDB schema upgrade failed: {exc}")
                return was_upgraded, messages
    except Exception as exc:
        messages.append(f"{db_path.name}: database check failed: {exc}")
        return was_upgraded, messages

    # ── Step 2: FlexTool data version upgrade ──────────────────────
    try:
        version_before = _read_flextool_version(db_url)

        from flextool.update_flextool.db_migration import migrate_database

        migrate_database(
            str(db_path),
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )

        version_after = _read_flextool_version(db_url)

        if (
            version_before is not None
            and version_after is not None
            and version_after > version_before
        ):
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
    except MigrationCancelled as exc:
        version_before_safe = version_before if version_before is not None else 0
        if exc.last_completed_version > version_before_safe:
            was_upgraded = True
        messages.append(
            f"{db_path.name}: FlexTool data migration cancelled by user at version "
            f"{exc.last_completed_version}. Re-run is safe: completed steps are idempotent."
        )
        logger.info(
            "FlexTool migration cancelled for %s at version %s",
            db_path,
            exc.last_completed_version,
        )
    except Exception as exc:
        import traceback as _tb
        tb_text = _tb.format_exc()
        messages.append(
            f"{db_path.name}: FlexTool data migration FAILED — migration was "
            f"cancelled. The database may be partially upgraded; do not modify "
            f"it before re-running the migration (re-run is safe: completed "
            f"steps are idempotent). To report this bug, please attach the "
            f"traceback below.\n\n"
            f"Error: {exc}\n\n"
            f"Traceback:\n{tb_text}"
        )
        logger.warning("FlexTool migration failed for %s: %s", db_path, exc, exc_info=True)

    return was_upgraded, messages
