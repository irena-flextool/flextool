"""Canonical FlexTool databases stored as JSON in the package.

Each entry in :data:`CANONICAL_DATABASES` pairs a JSON resource — the
source of truth, shipped inside ``flextool/textual_templates/canonical_databases/``
in the wheel — with the on-disk ``.sqlite`` path it materialises into
relative to the user's current working directory.

The user-facing workflow is one of:

``python -m flextool.update_flextool.canonical_databases materialize``
    Recreate any missing ``.sqlite`` files from the bundled JSON sources.
    Idempotent for files that already exist.  Called automatically by
    :func:`flextool.update_flextool.update_flextool` and by
    ``tests/conftest.py`` so users and CI always have ``.sqlite`` files
    to point Spine Toolbox at.

``python -m flextool.update_flextool.canonical_databases migrate-all``
    Round-trip every canonical JSON through
    ``initialize_database -> migrate_database -> export_database``.
    Run from a source checkout (editable install) whenever
    ``FLEXTOOL_DB_VERSION`` is bumped; commit the resulting JSON diffs
    alongside the migration step.  See ``CONTRIBUTING.md``.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from flextool._resources import package_data_path
from flextool.update_flextool.db_migration import migrate_database
from flextool.update_flextool.export_database import export_database
from flextool.update_flextool.initialize_database import initialize_database


# ``json_name`` is resolved against the bundled
# ``flextool/textual_templates/canonical_databases/`` directory via
# ``importlib.resources``.  ``sqlite_path`` is interpreted relative to
# the user's current working directory — Spine Toolbox project files and
# scenario references still see the same layout they always have.
@dataclass(frozen=True)
class CanonicalDatabase:
    json_name: str
    sqlite_path: str


CANONICAL_DATABASES: tuple[CanonicalDatabase, ...] = (
    CanonicalDatabase("templates_examples.json", "templates/examples.sqlite"),
    CanonicalDatabase("templates_time_settings_only.json", "templates/time_settings_only.sqlite"),
    CanonicalDatabase("howto_aggregate_output.json", "how to example databases/aggregate_output.sqlite"),
    CanonicalDatabase("howto_connections.json", "how to example databases/connections.sqlite"),
    CanonicalDatabase("howto_demand.json", "how to example databases/Demand.sqlite"),
    CanonicalDatabase("howto_hydro_reservoir.json", "how to example databases/hydro_reservoir.sqlite"),
    CanonicalDatabase("howto_hydro_reservoir_with_pump.json", "how to example databases/hydro_reservoir_with_pump.sqlite"),
    CanonicalDatabase("howto_non_sync_and_curtailment.json", "how to example databases/non_sync_and_curtailment.sqlite"),
    CanonicalDatabase("howto_ramp_and_start_up.json", "how to example databases/ramp_and_start_up.sqlite"),
    CanonicalDatabase("howto_stochastics.json", "how to example databases/stochastics.sqlite"),
)


def _json_source(db: CanonicalDatabase) -> Path:
    return package_data_path(f"textual_templates/canonical_databases/{db.json_name}")


def materialize(overwrite: bool = False) -> None:
    """Create every canonical SQLite from its JSON source.

    With ``overwrite=False`` (the default), existing ``.sqlite`` files
    in the working tree are left untouched — preserves user edits.
    Pass ``overwrite=True`` to force a refresh (used by the migrate-all
    command and by callers that need a guaranteed-current SQLite).

    Destination paths are interpreted relative to the current working
    directory; the canonical JSON sources are read from the installed
    ``flextool`` package via :mod:`importlib.resources`.
    """
    for db in CANONICAL_DATABASES:
        json_abs = _json_source(db)
        sqlite_abs = Path.cwd() / db.sqlite_path
        if not json_abs.is_file():
            print(f"!! canonical JSON missing in package: {db.json_name}")
            continue
        if sqlite_abs.exists():
            if not overwrite:
                continue
            sqlite_abs.unlink()
        sqlite_abs.parent.mkdir(parents=True, exist_ok=True)
        initialize_database(str(json_abs), str(sqlite_abs))


def migrate_all() -> None:
    """Round-trip every canonical JSON through the current schema.

    For each entry:

    1. Materialise the bundled JSON into a throwaway ``.sqlite``.
    2. Run :func:`migrate_database` against it — walks
       ``model.version`` up to :data:`FLEXTOOL_DB_VERSION`.
    3. Re-export the migrated ``.sqlite`` back over the bundled JSON
       file in place.

    Requires an editable install: the JSON files are written back to the
    on-disk package source, which only exists for ``pip install -e .``
    checkouts.  Wheel installs (``site-packages/``) are read-only and
    will fail here — that's intentional; only the maintainer running
    a schema bump should be regenerating canonical JSONs.
    """
    for db in CANONICAL_DATABASES:
        json_abs = _json_source(db)
        if not json_abs.is_file():
            print(f"!! canonical JSON missing in package: {db.json_name}")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            staging = os.path.join(tmp, "staging.sqlite")
            initialize_database(str(json_abs), staging)
            migrate_database(staging)
            export_database(staging, str(json_abs))
        print(f"migrated canonical JSON: {db.json_name}")


def _seed_from_cwd_sqlites() -> None:
    """One-time: populate JSONs from the SQLites currently in the tree.

    Reads each ``CanonicalDatabase.sqlite_path`` (CWD-relative) and
    exports it back over the package-bundled JSON source.  Only useful
    in a source checkout where the JSON files live on disk and can be
    written.
    """
    for db in CANONICAL_DATABASES:
        sqlite_abs = Path.cwd() / db.sqlite_path
        json_abs = _json_source(db)
        if not sqlite_abs.exists():
            print(f"!! cannot seed {db.json_name}: source SQLite missing at {db.sqlite_path}")
            continue
        json_abs.parent.mkdir(parents=True, exist_ok=True)
        export_database(str(sqlite_abs), str(json_abs))
        print(f"seeded {db.json_name} from {db.sqlite_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("materialize", help="Create missing .sqlite files from JSON")
    sub.add_parser(
        "materialize-force", help="Recreate every canonical .sqlite from JSON"
    )
    sub.add_parser(
        "migrate-all", help="Round-trip every JSON through the current schema"
    )
    sub.add_parser(
        "seed",
        help="One-time: export every CWD .sqlite to its canonical JSON",
    )
    args = parser.parse_args(argv)

    if args.cmd == "materialize":
        materialize(overwrite=False)
    elif args.cmd == "materialize-force":
        materialize(overwrite=True)
    elif args.cmd == "migrate-all":
        migrate_all()
    elif args.cmd == "seed":
        _seed_from_cwd_sqlites()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
