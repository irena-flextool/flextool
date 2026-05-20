"""Canonical FlexTool databases stored as JSON in git.

Each entry in :data:`CANONICAL_DATABASES` is a JSON file under
``version/canonical_databases/`` paired with the on-disk ``.sqlite``
path it materialises into.  The JSON form is the source of truth: it
diffs cleanly in pull requests, compresses well in git packs, and is
trivially mergeable when two branches touch the same example data.

The user-facing workflow is one of:

``python -m flextool.update_flextool.canonical_databases materialize``
    Recreate any missing ``.sqlite`` files from JSON.  Idempotent for
    files that already exist.  Called automatically by
    :func:`flextool.update_flextool.update_flextool` on every git pull
    so that users always have ``.sqlite`` files to point Spine Toolbox
    at.

``python -m flextool.update_flextool.canonical_databases migrate-all``
    Round-trip every canonical JSON through
    ``initialize_database -> migrate_database -> export_database``.
    Run this once whenever ``FLEXTOOL_DB_VERSION`` is bumped (i.e. a
    step is added to :mod:`flextool.update_flextool.db_migration`);
    commit the resulting JSON diffs alongside the migration step.  See
    ``CONTRIBUTING.md``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from dataclasses import dataclass

from flextool.update_flextool.export_database import export_database
from flextool.update_flextool.initialize_database import initialize_database
from flextool.update_flextool.db_migration import migrate_database


# ``json_path`` is repo-relative.  ``sqlite_path`` is also repo-relative
# and is the on-disk location users expect to find — keeping the same
# layout as before the JSON-canonical migration means Spine Toolbox
# project files and example references don't need to change.
@dataclass(frozen=True)
class CanonicalDatabase:
    json_path: str
    sqlite_path: str


CANONICAL_DATABASES: tuple[CanonicalDatabase, ...] = (
    CanonicalDatabase(
        json_path="version/canonical_databases/templates_examples.json",
        sqlite_path="templates/examples.sqlite",
    ),
    CanonicalDatabase(
        json_path="version/canonical_databases/templates_time_settings_only.json",
        sqlite_path="templates/time_settings_only.sqlite",
    ),
    CanonicalDatabase(
        json_path="version/canonical_databases/howto_aggregate_output.json",
        sqlite_path="how to example databases/aggregate_output.sqlite",
    ),
    CanonicalDatabase(
        json_path="version/canonical_databases/howto_connections.json",
        sqlite_path="how to example databases/connections.sqlite",
    ),
    CanonicalDatabase(
        json_path="version/canonical_databases/howto_demand.json",
        sqlite_path="how to example databases/Demand.sqlite",
    ),
    CanonicalDatabase(
        json_path="version/canonical_databases/howto_hydro_reservoir.json",
        sqlite_path="how to example databases/hydro_reservoir.sqlite",
    ),
    CanonicalDatabase(
        json_path="version/canonical_databases/howto_hydro_reservoir_with_pump.json",
        sqlite_path="how to example databases/hydro_reservoir_with_pump.sqlite",
    ),
    CanonicalDatabase(
        json_path="version/canonical_databases/howto_non_sync_and_curtailment.json",
        sqlite_path="how to example databases/non_sync_and_curtailment.sqlite",
    ),
    CanonicalDatabase(
        json_path="version/canonical_databases/howto_ramp_and_start_up.json",
        sqlite_path="how to example databases/ramp_and_start_up.sqlite",
    ),
    CanonicalDatabase(
        json_path="version/canonical_databases/howto_stochastics.json",
        sqlite_path="how to example databases/stochastics.sqlite",
    ),
)


def _repo_root() -> str:
    # canonical_databases.py -> update_flextool/ -> flextool/ -> repo root.
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )


def materialize(overwrite: bool = False) -> None:
    """Create every canonical SQLite from its JSON source.

    With ``overwrite=False`` (the default), existing ``.sqlite`` files
    are left untouched — this preserves any user edits in the working
    tree, mirroring the prior ``update_flextool`` behaviour for
    ``input_data.sqlite``.  Pass ``overwrite=True`` to force a refresh
    (used by templates that must always track the canonical form, and
    by the migrate-all command).
    """
    root = _repo_root()
    for db in CANONICAL_DATABASES:
        json_abs = os.path.join(root, db.json_path)
        sqlite_abs = os.path.join(root, db.sqlite_path)
        if not os.path.exists(json_abs):
            print(f"!! canonical JSON missing: {db.json_path}")
            continue
        if os.path.exists(sqlite_abs):
            if not overwrite:
                continue
            os.remove(sqlite_abs)
        os.makedirs(os.path.dirname(sqlite_abs) or ".", exist_ok=True)
        initialize_database(json_abs, sqlite_abs)


def migrate_all() -> None:
    """Round-trip every canonical JSON through the current schema.

    For each entry:

    1. Materialise the JSON into a throwaway ``.sqlite`` (at whatever
       schema version the JSON was authored for).
    2. Run :func:`migrate_database` against it — that walks
       ``model.version`` up to :data:`FLEXTOOL_DB_VERSION`.
    3. Re-export the migrated ``.sqlite`` back over the canonical JSON
       file in place.

    Run after bumping ``FLEXTOOL_DB_VERSION`` to keep the canonical
    JSONs current.  The on-disk ``.sqlite`` files (if any) are left
    alone — call :func:`materialize` with ``overwrite=True`` if you
    also want to refresh them.
    """
    root = _repo_root()
    for db in CANONICAL_DATABASES:
        json_abs = os.path.join(root, db.json_path)
        if not os.path.exists(json_abs):
            print(f"!! canonical JSON missing: {db.json_path}")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            staging = os.path.join(tmp, "staging.sqlite")
            initialize_database(json_abs, staging)
            migrate_database(staging)
            export_database(staging, json_abs)
        print(f"migrated canonical JSON: {db.json_path}")


def _seed_from_existing_sqlites() -> None:
    """One-time: populate JSONs from the SQLites currently in the tree.

    Used by ``--seed`` when the JSONs do not yet exist.  Idempotent —
    overwrites any existing JSON, but the round-trip is bit-stable so
    re-running on a clean tree is a no-op.
    """
    root = _repo_root()
    for db in CANONICAL_DATABASES:
        sqlite_abs = os.path.join(root, db.sqlite_path)
        json_abs = os.path.join(root, db.json_path)
        if not os.path.exists(sqlite_abs):
            print(f"!! cannot seed {db.json_path}: source SQLite missing at {db.sqlite_path}")
            continue
        os.makedirs(os.path.dirname(json_abs) or ".", exist_ok=True)
        export_database(sqlite_abs, json_abs)
        print(f"seeded {db.json_path} from {db.sqlite_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("materialize", help="Create missing .sqlite files from JSON")
    p_mat_force = sub.add_parser(
        "materialize-force", help="Recreate every canonical .sqlite from JSON"
    )
    sub.add_parser(
        "migrate-all", help="Round-trip every JSON through the current schema"
    )
    sub.add_parser(
        "seed",
        help="One-time: export every tracked .sqlite to its canonical JSON",
    )
    args = parser.parse_args(argv)

    if args.cmd == "materialize":
        materialize(overwrite=False)
    elif args.cmd == "materialize-force":
        materialize(overwrite=True)
    elif args.cmd == "migrate-all":
        migrate_all()
    elif args.cmd == "seed":
        _seed_from_existing_sqlites()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
