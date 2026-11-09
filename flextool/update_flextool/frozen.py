"""Frozen FlexTool input databases: a marker that stops a database from being migrated in place.

A released model (a reference case, a study's final input) is *frozen* with ``flextool-freeze``: a row in the
SpineDB ``metadata`` table, ``name = "flextool_frozen"``, ``value = "<label>|<data version>|<ISO date>"``.  The
marker travels with the file and is not tied to an alternative, so the engine never sees it.

Reading the marker and the data version uses plain :mod:`sqlite3` in read-only mode, so a check never upgrades the
SpineDB schema of the file, never writes and never takes a write lock.  :func:`frozen_guard` is the first statement of
:func:`~flextool.update_flextool.db_migration.migrate_database` and of
:func:`~flextool.gui.db_version_check.check_and_upgrade_database`: a frozen database older than this FlexTool is
migrated only when the caller passes ``allow_frozen=True``, and a frozen database that is already current is left
unchanged.  :func:`migrate_copy` migrates a copy and leaves the original byte-identical.

The SHA-256 of a released file belongs in the project manifest, not here: writing the marker changes the hash.
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable
from urllib.parse import quote

logger = logging.getLogger(__name__)

FROZEN_KEY = "flextool_frozen"
MIGRATED_FROM_KEY = "flextool_migrated_from"
#: The metadata value column holds 255 characters; the migrated-from value adds a version and a SHA-256 to the label.
MAX_LABEL_LENGTH = 170
_SEPARATOR = "|"


@dataclass(frozen=True)
class FrozenInfo:
    """The frozen marker of a database."""

    label: str
    version: int | None
    date: str

    def describe(self) -> str:
        """``"<label>", frozen at version <v> on <date>`` for messages."""
        version = f"version {self.version}" if self.version is not None else "an unknown version"
        return f"'{self.label}', frozen at {version} on {self.date}"

    def to_value(self) -> str:
        """The metadata value of this marker."""
        version = "" if self.version is None else str(self.version)
        return _SEPARATOR.join((self.label, version, self.date))


@dataclass(frozen=True)
class MigratedFrom:
    """The ``flextool_migrated_from`` marker of a migrated copy of a frozen database."""

    label: str
    version: int | None
    sha256: str

    def to_value(self) -> str:
        """The metadata value of this marker."""
        version = "" if self.version is None else str(self.version)
        return _SEPARATOR.join((self.label, version, "sha256:" + self.sha256))


class FrozenDatabaseError(RuntimeError):
    """A frozen database would be migrated without ``allow_frozen=True``."""

    def __init__(self, path: Path, info: FrozenInfo, db_version: int | None, target_version: int) -> None:
        self.path = path
        self.info = info
        self.db_version = db_version
        self.target_version = target_version
        found = f"version {db_version}" if db_version is not None else "an unreadable version"
        super().__init__(
            f"{path.name} is frozen ({info.describe()}). It is at {found}; this FlexTool needs version "
            f"{target_version}. The frozen file is not migrated in place. Migrate a copy: "
            f"flextool-migrate-database {path} --copy-to NEW.sqlite (or, to change the frozen file itself, "
            f"--allow-migrate-frozen)."
        )


class FreezeError(ValueError):
    """``freeze`` refused: a bad label, or a database that is not at the current version."""


def _target_version() -> int:
    from flextool.update_flextool import FLEXTOOL_DB_VERSION

    return int(FLEXTOOL_DB_VERSION)


def sqlite_path_from(database: str | Path) -> Path | None:
    """The file of a plain path or a ``sqlite:///`` URL (its query string dropped); ``None`` for other URLs."""
    if isinstance(database, Path):
        return database
    text = str(database)
    if text.startswith("sqlite:///"):
        return Path(text[len("sqlite:///"):].split("?", 1)[0])
    if "://" in text:
        return None
    return Path(text)


def _connect_ro(path: Path) -> sqlite3.Connection:
    """A read-only connection: never creates the file, never writes, never upgrades."""
    uri = "file:" + quote(Path(path).resolve().as_posix()) + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _metadata_rows(path: Path, name: str) -> list[tuple[int, str]]:
    """``(id, value)`` of the metadata rows called *name*, oldest first; ``[]`` without a ``metadata`` table."""
    if not Path(path).is_file():
        return []
    try:
        con = _connect_ro(path)
    except sqlite3.Error:
        return []
    try:
        rows = con.execute("SELECT id, value FROM metadata WHERE name = ? ORDER BY id", (name,)).fetchall()
    except sqlite3.Error:
        return []
    finally:
        con.close()
    return [(int(row_id), str(value)) for row_id, value in rows]


def _int_or_none(text: str) -> int | None:
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _newest(path: Path, name: str) -> str | None:
    rows = _metadata_rows(path, name)
    if not rows:
        return None
    if len(rows) > 1:
        logger.warning("%s has %d '%s' metadata rows; using the newest.", Path(path).name, len(rows), name)
    return rows[-1][1]


def parse_frozen(value: str) -> FrozenInfo:
    """A :class:`FrozenInfo` from a metadata value (a label that holds ``|`` keeps the last two fields apart)."""
    parts = value.rsplit(_SEPARATOR, 2)
    if len(parts) != 3:
        return FrozenInfo(label=value, version=None, date="")
    label, version, when = parts
    return FrozenInfo(label=label, version=_int_or_none(version), date=when)


def read_frozen(path: str | Path) -> FrozenInfo | None:
    """The frozen marker of the database at *path*, or ``None`` (no marker, no ``metadata`` table, no file)."""
    value = _newest(Path(path), FROZEN_KEY)
    return None if value is None else parse_frozen(value)


def read_migrated_from(path: str | Path) -> MigratedFrom | None:
    """The ``flextool_migrated_from`` marker of a migrated copy, or ``None``."""
    value = _newest(Path(path), MIGRATED_FROM_KEY)
    if value is None:
        return None
    parts = value.rsplit(_SEPARATOR, 2)
    if len(parts) != 3 or not parts[2].startswith("sha256:"):
        return None
    return MigratedFrom(label=parts[0], version=_int_or_none(parts[1]), sha256=parts[2][len("sha256:"):])


def read_db_version_ro(path: str | Path) -> int | None:
    """The FlexTool data version (``model.version`` default) read with sqlite3 read-only.

    Returns ``0`` for a Spine database without the parameter (as the migration assumes), and ``None`` when the file
    is missing or is not a Spine database.  Reading never upgrades or writes the file.
    """
    path = Path(path)
    if not path.is_file():
        return None
    queries = (
        "SELECT pd.default_value, pd.default_type FROM parameter_definition pd "
        "JOIN entity_class ec ON pd.entity_class_id = ec.id WHERE ec.name = 'model' AND pd.name = 'version'",
        # SpineDB schemas from before the entity refactor.
        "SELECT pd.default_value, pd.default_type FROM parameter_definition pd "
        "JOIN object_class oc ON pd.object_class_id = oc.id WHERE oc.name = 'model' AND pd.name = 'version'",
    )
    try:
        con = _connect_ro(path)
    except sqlite3.Error:
        return None
    try:
        row = None
        for query in queries:
            try:
                row = con.execute(query).fetchone()
            except sqlite3.Error:
                continue
            break
        else:
            return None
    finally:
        con.close()
    if row is None:
        return 0
    value, value_type = row
    try:
        from spinedb_api import from_database

        if isinstance(value, str):
            value = value.encode("utf-8")
        return int(float(from_database(value, value_type)))
    except Exception:
        logger.debug("Could not decode the FlexTool version of %s", path, exc_info=True)
        return None


def sha256_of(path: str | Path) -> str:
    """The SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_guard(database: str | Path, *, allow_frozen: bool = False) -> FrozenInfo | None:
    """The check that runs before a migration opens the file.

    Returns ``None`` for a database that is not frozen (or not a local sqlite file), and the marker of a frozen one
    that *allow_frozen* lets the caller migrate.  Raises :class:`FrozenDatabaseError` for a frozen database without
    *allow_frozen*; its ``db_version == target_version`` tells the caller that the file is current and only has to be
    left unchanged.
    """
    path = sqlite_path_from(database)
    if path is None or not path.is_file():
        return None
    info = read_frozen(path)
    if info is None:
        return None
    if allow_frozen:
        return info
    raise FrozenDatabaseError(path, info, read_db_version_ro(path), _target_version())


def _validate_label(label: str) -> str:
    label = (label or "").strip()
    if not label:
        raise FreezeError("A frozen database needs a label, e.g. \"UAE reference 1.0\".")
    if _SEPARATOR in label:
        raise FreezeError(f"The label cannot contain '{_SEPARATOR}'.")
    if len(label) > MAX_LABEL_LENGTH:
        raise FreezeError(f"The label is {len(label)} characters long; at most {MAX_LABEL_LENGTH} fit.")
    return label


def _replace_rows(db, name: str, value: str | None) -> None:
    """Remove every metadata row called *name* and, when *value* is given, add one with it (a DatabaseMapping)."""
    ids = [item["id"] for item in db.get_items("metadata") if item["name"] == name]
    if ids:
        db.remove_items("metadata", *ids)
    if value is not None:
        db.add_item("metadata", name=name, value=value)


def set_frozen_version(db, version: int) -> None:
    """Rewrite the version part of the frozen marker in an open DatabaseMapping (an allowed in-place migration).

    The label and the date are kept; the caller commits.
    """
    rows = sorted((item for item in db.get_items("metadata") if item["name"] == FROZEN_KEY), key=lambda i: i["id"])
    if not rows:
        return
    info = parse_frozen(rows[-1]["value"])
    _replace_rows(db, FROZEN_KEY, FrozenInfo(label=info.label, version=int(version), date=info.date).to_value())


def _open(path: Path):
    from flextool.update_flextool.db_handle import open_database_mapping

    return open_database_mapping(f"sqlite:///{Path(path).resolve().as_posix()}", create=False, upgrade=False)


def freeze(path: str | Path, label: str, *, today: str | None = None) -> FrozenInfo:
    """Freeze the database at *path* with *label*; only a database at the current version can be frozen."""
    path = Path(path)
    label = _validate_label(label)
    if not path.is_file():
        raise FreezeError(f"No database file at {path}.")
    version = read_db_version_ro(path)
    target = _target_version()
    if version != target:
        found = f"version {version}" if version is not None else "an unreadable version"
        raise FreezeError(
            f"{path.name} is at {found}; only a database at the current version ({target}) can be frozen. "
            f"Migrate it first: flextool-migrate-database {path}"
        )
    info = FrozenInfo(label=label, version=version, date=today or date.today().isoformat())
    with _open(path) as db:
        _replace_rows(db, FROZEN_KEY, info.to_value())
        db.commit_session(f"freeze: {label}")
    return info


def unfreeze(path: str | Path) -> FrozenInfo | None:
    """Remove the frozen marker; returns the marker removed, or ``None`` when the database was not frozen."""
    path = Path(path)
    info = read_frozen(path)
    if info is None:
        return None
    with _open(path) as db:
        _replace_rows(db, FROZEN_KEY, None)
        db.commit_session(f"unfreeze: {info.label}")
    return info


def migrate_copy(
    source: str | Path,
    destination: str | Path,
    *,
    progress_callback: Callable[[int, int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> MigratedFrom:
    """Copy a frozen database to *destination* and migrate the copy; the original stays byte-identical.

    The copy is not frozen: its frozen marker is replaced by ``flextool_migrated_from`` =
    ``<label>|<old version>|sha256:<hash of the original>``.  A copy left by a failed or cancelled migration is
    removed.  Refuses an existing *destination*.
    """
    from flextool.update_flextool.db_migration import migrate_database

    source = Path(source)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"{destination} exists already; give a new file for the migrated copy.")
    if source.resolve() == destination.resolve():
        raise FileExistsError("The copy must be a different file from the original.")
    info = read_frozen(source)
    old_version = read_db_version_ro(source)
    marker = MigratedFrom(
        label=info.label if info is not None else source.stem, version=old_version, sha256=sha256_of(source)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    try:
        migrate_database(
            "sqlite:///" + destination.resolve().as_posix(), allow_frozen=True, progress_callback=progress_callback, cancel_check=cancel_check
        )
        with _open(destination) as db:
            _replace_rows(db, FROZEN_KEY, None)
            _replace_rows(db, MIGRATED_FROM_KEY, marker.to_value())
            db.commit_session(f"migrated copy of frozen database '{marker.label}'")
    except BaseException:
        for suffix in ("", "-journal", "-wal", "-shm"):
            Path(str(destination) + suffix).unlink(missing_ok=True)
        raise
    return marker

