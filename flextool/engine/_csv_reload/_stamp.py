"""Reader side of the dump format stamp: refuse incompatible dumps, warn on every reload.

The writer side (``_dump_meta.json``, its fields and merge rule) is :mod:`flextool.engine.preprocess._dump_stamp`.

:data:`SUPPORTED_DUMP_FORMAT` is the dump layout this package reads.  :data:`VERIFIED_FLEXTOOL` /
:data:`VERIFIED_SCHEMA` record the engine the reload was last checked against; they are shown in the warning and
changed only by whoever re-syncs this package with the engine, never by normal engine work.
"""
from __future__ import annotations

import warnings
from pathlib import Path

from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.preprocess._dump_stamp import STAMP_FILE, _engine_identity, read_dump_stamp

SUPPORTED_DUMP_FORMAT: int = 1
VERIFIED_FLEXTOOL: str = "4.1.1"
VERIFIED_SCHEMA: int = 76

# Fields a stamp must carry for the compatibility check.
_REQUIRED_FIELDS: tuple[str, ...] = ("dump_format", "schema_version")


class CsvReloadWarning(UserWarning):
    """Emitted on every CSV reload: the reload is unsupported and may be stale."""


def check_dump(dump_dir: Path | str, force: bool) -> dict | None:
    """Check ``dump_dir``'s stamp against this reader and the running engine; return the stamp.

    Raises ``FileNotFoundError`` when ``dump_dir`` is not a directory (also with ``force``).  Without ``force``,
    raises :class:`FlexToolConfigError` when the stamp is absent, unreadable or lacks ``dump_format`` /
    ``schema_version``, when its dump format is not :data:`SUPPORTED_DUMP_FORMAT`, or when its schema version is not
    the running engine's.  A different FlexTool version or git commit is not refused (the version does not change
    per commit in a development checkout, and a git difference alone does not prove staleness); the warning shows
    both.  With ``force``, returns whatever stamp could be read (``None`` when there is none).
    """
    dump_dir = Path(dump_dir)
    if not dump_dir.is_dir():
        raise FileNotFoundError(f"CSV dump folder {dump_dir} does not exist or is not a directory")
    stamp = read_dump_stamp(dump_dir)
    if force:
        return stamp
    if stamp is None or any(k not in stamp for k in _REQUIRED_FIELDS):
        raise FlexToolConfigError(
            f"{dump_dir} has no readable {STAMP_FILE}, so it is not a dump written by FlexData.dump_csvs or "
            f"FlexDataProvider.snapshot_processed_inputs of this FlexTool. CSV reload is unsupported; pass "
            f"force=True to load it anyway."
        )
    found_format = stamp["dump_format"]
    if found_format != SUPPORTED_DUMP_FORMAT:
        raise FlexToolConfigError(
            f"{dump_dir} is dump format {found_format}; this CSV reload reads dump format {SUPPORTED_DUMP_FORMAT}. "
            f"CSV reload is unsupported and not kept in sync; pass force=True to load it anyway."
        )
    engine_schema = _engine_identity()["schema_version"]
    found_schema = stamp["schema_version"]
    if found_schema != engine_schema:
        raise FlexToolConfigError(
            f"{dump_dir} was written with database schema {found_schema}; this engine uses schema {engine_schema}. "
            f"CSV reload is unsupported and not kept in sync; pass force=True to load it anyway."
        )
    return stamp


def _render_git(value: object) -> str:
    return "unknown" if value is None else str(value)


def reload_warning_text(dump_dir: Path | str, stamp: dict | None) -> str:
    """The :class:`CsvReloadWarning` message for a reload of ``dump_dir`` with ``stamp`` (``None``: no stamp)."""
    engine = _engine_identity()
    if stamp is None:
        dump_desc = f"no {STAMP_FILE}"
    else:
        dump_desc = (
            f"dump format {stamp.get('dump_format', 'unknown')}, flextool {stamp.get('flextool', 'unknown')} "
            f"(git {_render_git(stamp.get('git'))}), schema {stamp.get('schema_version', 'unknown')}"
        )
    text = (
        f"CSV reload is unsupported: flextool.engine._csv_reload is not kept in sync with the engine, so a model "
        f"reloaded from a CSV dump may be stale and can differ from the solved model (it has no solve-to-solve "
        f"carry). Dump {dump_dir}: {dump_desc}. This engine: dump format {SUPPORTED_DUMP_FORMAT}, flextool "
        f"{engine['flextool']} (git {_render_git(engine['git'])}), schema {engine['schema_version']}. Reload last "
        f"verified with flextool {VERIFIED_FLEXTOOL}, schema {VERIFIED_SCHEMA}. Files in the folder are not "
        f"checked against the stamp, so CSVs left in a reused folder by an earlier run are not detected."
    )
    previous = stamp.get("previous") if stamp is not None else None
    if isinstance(previous, dict):
        text += (
            f" The folder also holds files of an earlier dump (flextool {previous.get('flextool', 'unknown')}, "
            f"schema {previous.get('schema_version', 'unknown')}); stale CSVs may be read."
        )
    return text


def warn_reload(dump_dir: Path | str, stamp: dict | None, *, stacklevel: int) -> None:
    """Emit the :class:`CsvReloadWarning` for ``dump_dir``.

    ``stacklevel`` counts from the caller of this function (1 = that caller), so the public entry points pass it
    through and the warning points at the user's line, not at this package.
    """
    warnings.warn(reload_warning_text(dump_dir, stamp), CsvReloadWarning, stacklevel=stacklevel + 1)


__all__ = [
    "SUPPORTED_DUMP_FORMAT",
    "VERIFIED_FLEXTOOL",
    "VERIFIED_SCHEMA",
    "CsvReloadWarning",
    "check_dump",
    "reload_warning_text",
    "warn_reload",
]
