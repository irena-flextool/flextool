"""Bring every ``tests/fixtures/*.json`` to the current schema.

Mirrors :mod:`flextool.update_flextool.canonical_databases` for the test
fixture corpus.  The format used here differs from canonical_databases
(base64-packed values vs. UTF-8 decoded), so this module uses
``tests.db_utils.json_to_db`` / ``tests.db_utils.db_to_json`` rather than
the canonical ``initialize_database`` / ``export_database`` pair.

Each fixture is migrated (JSON -> SQLite -> migrate -> JSON, dropping the
behaviour-change markers the migration writes) and then its schema
sections are replaced by the schema's
(:func:`flextool.update_flextool.fixture_schema.sync_schema_sections`).
A fixture without a ``model.version`` definition skips the migration and
is only synced.  Every ``tests/fixtures/*.json`` that has
``parameter_definitions`` is covered.

Run after a schema change alongside ``sync_master_json_template`` and
``canonical_databases migrate-all``.  See CONTRIBUTING.md.

Usage:
    python -m flextool.update_flextool.test_fixtures migrate-all
    python -m flextool.update_flextool.test_fixtures verify

The ``verify`` subcommand is intended for CI: it performs the same
round-trip in a temp file and exits non-zero (printing a short diff
summary) when a committed fixture is out of sync with the current
schema.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from flextool.update_flextool.db_migration import migrate_repository_database
from flextool.update_flextool.fixture_schema import (
    dump_fixture_json,
    fixture_version,
    section_diffs,
    sync_schema_sections,
)


def _repo_root() -> Path:
    # .../flextool-engine/flextool/update_flextool/test_fixtures.py
    return Path(__file__).resolve().parent.parent.parent


def _fixtures_dir() -> Path:
    """Locate ``tests/fixtures/`` from a source checkout.

    Walks up from this file (``flextool/update_flextool/test_fixtures.py``)
    to the repo root and joins ``tests/fixtures``.  Fails loudly if the
    directory is missing — this module is only meaningful in an editable
    install where the fixtures are present on disk.
    """
    fixtures = _repo_root() / "tests" / "fixtures"
    if not fixtures.is_dir():
        raise RuntimeError(
            f"tests/fixtures/ not found at {fixtures}. "
            "This migration must be run from a source checkout."
        )
    return fixtures


def fixture_paths(fixtures_dir: Path | None = None) -> list[Path]:
    """Every ``*.json`` in the fixtures directory that has ``parameter_definitions``, sorted by name."""
    fixtures = fixtures_dir if fixtures_dir is not None else _fixtures_dir()
    out = []
    for path in sorted(fixtures.glob("*.json")):
        with open(path) as f:
            if "parameter_definitions" in json.load(f):
                out.append(path)
    return out


def _round_trip(json_path: Path, out_path: Path) -> None:
    """JSON -> SQLite -> migrate (markers dropped) -> JSON, in the packed format."""
    # Imported lazily: ``tests`` exists only in a source checkout, whose
    # root a fixture builder run as a script does not have on sys.path.
    if str(_repo_root()) not in sys.path:
        sys.path.insert(0, str(_repo_root()))
    from tests.db_utils import db_to_json, json_to_db

    with tempfile.TemporaryDirectory() as tmp:
        sqlite_path = Path(tmp) / "staging.sqlite"
        url = json_to_db(json_path, sqlite_path)
        migrate_repository_database(url)
        db_to_json(sqlite_path, out_path)


def migrate_fixture(json_path: Path, out_path: Path | None = None) -> None:
    """Migrate one fixture and replace its schema sections with the schema's.

    Writes to ``out_path`` (default: over ``json_path``).  A fixture
    without a ``model.version`` definition is only synced.  Raises
    :class:`~flextool.update_flextool.fixture_schema.FixtureSchemaError`
    when the fixture's data does not fit the schema.
    """
    out_path = out_path if out_path is not None else json_path
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    with tempfile.TemporaryDirectory() as tmp:
        source = json_path
        if fixture_version(data, "packed") is None:
            # Written against the current schema: the sync adds the version
            # definition, so the round trip below does not migrate.
            source = Path(tmp) / "synced.json"
            dump_fixture_json(sync_schema_sections(data, "packed"), source)
        staging = Path(tmp) / json_path.name
        _round_trip(source, staging)
        with open(staging) as f:
            data = json.load(f)
    dump_fixture_json(sync_schema_sections(data, "packed"), out_path)


def write_built_fixture(spec: dict, out_path: Path) -> None:
    """Write a fixture a builder script authored, then migrate and sync it like ``migrate-all``.

    The committed file then has the layout ``verify`` reproduces.
    """
    dump_fixture_json(spec, out_path)
    migrate_fixture(out_path)


def migrate_all(fixtures_dir: Path | None = None) -> None:
    """Migrate and sync every fixture in place.

    Diffs may be large when a fixture's schema sections catch up with the
    schema; that is expected.
    """
    for src in fixture_paths(fixtures_dir):
        migrate_fixture(src)
        print(f"migrated test fixture: {src.name}")


def _summarise_diff(committed: Path, regenerated: Path) -> str:
    """Return a short human-readable summary of which top-level sections
    differ between two fixture JSONs, with the schema-section differences.

    Used by ``verify`` to surface drift without flooding the CI log.
    """
    with open(committed) as f:
        a = json.load(f)
    with open(regenerated) as f:
        b = json.load(f)
    keys = sorted(set(a) | set(b))
    lines = []
    for key in keys:
        av = a.get(key)
        bv = b.get(key)
        if av == bv:
            continue
        la = len(av) if isinstance(av, list) else "?"
        lb = len(bv) if isinstance(bv, list) else "?"
        lines.append(f"  {key}: committed={la} regenerated={lb}")
    if "parameter_definitions" in a:
        diffs = section_diffs(a, "packed")
        lines += [f"    {line}" for line in diffs[:20]]
        if len(diffs) > 20:
            lines.append(f"    ... {len(diffs) - 20} more schema-section differences")
    return "\n".join(lines) if lines else "  (no top-level section differs — check value-level)"


def _compare(src: Path, regenerated: Path) -> bool:
    with open(src) as f:
        committed_text = f.read()
    with open(regenerated) as f:
        regenerated_text = f.read()
    if committed_text == regenerated_text:
        print(f"ok: {src.name}")
        return True
    print(f"DRIFT: {src.name}")
    print(_summarise_diff(src, regenerated))
    return False


def verify_all(fixtures_dir: Path | None = None) -> int:
    """Verify that every fixture is byte-stable through migrate + schema sync.

    Returns 0 when every file is stable, or the number of drifted files.
    A fixture whose data does not fit the schema counts as drifted.
    """
    mismatches = 0
    for src in fixture_paths(fixtures_dir):
        with tempfile.TemporaryDirectory() as tmp:
            regenerated = Path(tmp) / src.name
            try:
                migrate_fixture(src, regenerated)
            except ValueError as exc:
                mismatches += 1
                print(f"DRIFT: {src.name}\n  {exc}")
                continue
            if not _compare(src, regenerated):
                mismatches += 1
    if mismatches:
        print(
            f"\n{mismatches} test fixture(s) out of sync with current schema. "
            "Run `python -m flextool.update_flextool.test_fixtures migrate-all`."
        )
    return mismatches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser(
        "migrate-all",
        help="Migrate every tests/fixtures/*.json and sync its schema sections with the schema",
    )
    sub.add_parser(
        "verify",
        help="Check every test fixture is byte-stable through the current schema",
    )
    args = parser.parse_args(argv)

    if args.cmd == "migrate-all":
        migrate_all()
        return 0
    elif args.cmd == "verify":
        return verify_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
