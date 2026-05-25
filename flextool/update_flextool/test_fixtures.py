"""Round-trip every tests/fixtures/*.json through the current schema migration.

Mirrors :mod:`flextool.update_flextool.canonical_databases` for the test
fixture corpus.  The format used here differs from canonical_databases
(base64-packed values vs. UTF-8 decoded), so this module uses
``tests.db_utils.json_to_db`` / ``tests.db_utils.db_to_json`` rather than
the canonical ``initialize_database`` / ``export_database`` pair.

Run after a schema bump alongside ``sync_master_json_template`` and
``canonical_databases migrate-all``.  See CONTRIBUTING.md.

Usage:
    python -m flextool.update_flextool.test_fixtures migrate-all
    python -m flextool.update_flextool.test_fixtures verify

The ``verify`` subcommand is intended for CI: it performs the same
round-trip in a temp file and exits non-zero (printing a short diff
summary) when any committed fixture is out of sync with the current
schema.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from flextool.update_flextool.db_migration import migrate_database


# Resolved at runtime against the repository checkout — these fixtures live
# under ``tests/fixtures/`` (outside the installed wheel), so we walk up
# from this module to find the source tree.
TEST_FIXTURES: tuple[str, ...] = (
    "tests.json",
    "stochastics.json",
    "lh2_three_region.json",
    "h2_trade_parity.json",
    "multi_ts_branch1.json",
    "stochastics_pbt_inflow.json",
    "branch2_parent_period.json",
)


def _fixtures_dir() -> Path:
    """Locate ``tests/fixtures/`` from a source checkout.

    Walks up from this file (``flextool/update_flextool/test_fixtures.py``)
    to the repo root and joins ``tests/fixtures``.  Fails loudly if the
    directory is missing — this module is only meaningful in an editable
    install where the fixtures are present on disk.
    """
    here = Path(__file__).resolve()
    # .../flextool-engine/flextool/update_flextool/test_fixtures.py
    repo_root = here.parent.parent.parent
    fixtures = repo_root / "tests" / "fixtures"
    if not fixtures.is_dir():
        raise RuntimeError(
            f"tests/fixtures/ not found at {fixtures}. "
            "This migration must be run from a source checkout."
        )
    return fixtures


def _migrate_one(json_path: Path, out_path: Path) -> None:
    """Round-trip a single fixture: JSON -> SQLite -> migrate -> JSON.

    Uses ``tests.db_utils.json_to_db`` and ``tests.db_utils.db_to_json``
    so the base64 value-packing convention used by the test fixtures is
    preserved across the round-trip.
    """
    # Imported lazily so this module can be loaded outside a checkout
    # (e.g. the canonical_databases command in a wheel install) without
    # tripping over the test-package layering inversion.
    from tests.db_utils import db_to_json, json_to_db

    with tempfile.TemporaryDirectory() as tmp:
        sqlite_path = Path(tmp) / "staging.sqlite"
        url = json_to_db(json_path, sqlite_path)
        migrate_database(url)
        db_to_json(sqlite_path, out_path)


def migrate_all() -> None:
    """Migrate every ``tests/fixtures/*.json`` file in place.

    Writes the result back over the source JSON.  Diffs may be large
    when a fixture's ``parameter_definitions`` are catching up to a new
    schema; that is expected.
    """
    fixtures = _fixtures_dir()
    for name in TEST_FIXTURES:
        src = fixtures / name
        if not src.is_file():
            print(f"!! test fixture missing: {src}")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            staging_json = Path(tmp) / name
            _migrate_one(src, staging_json)
            shutil.copyfile(staging_json, src)
        print(f"migrated test fixture: {name}")


def _summarise_diff(committed: Path, regenerated: Path) -> str:
    """Return a short human-readable summary of which top-level sections
    differ between two fixture JSONs.

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
    return "\n".join(lines) if lines else "  (no top-level section differs — check value-level)"


def verify_all() -> int:
    """Verify that every committed fixture matches its migrated form.

    Returns 0 when every file is byte-stable through the round-trip, or
    a positive count of mismatched files otherwise.  Intended for CI use
    via ``python -m flextool.update_flextool.test_fixtures verify``.
    """
    fixtures = _fixtures_dir()
    mismatches = 0
    for name in TEST_FIXTURES:
        src = fixtures / name
        if not src.is_file():
            print(f"!! test fixture missing: {src}")
            mismatches += 1
            continue
        with tempfile.TemporaryDirectory() as tmp:
            regenerated = Path(tmp) / name
            _migrate_one(src, regenerated)
            with open(src) as f:
                committed_text = f.read()
            with open(regenerated) as f:
                regenerated_text = f.read()
            if committed_text == regenerated_text:
                print(f"ok: {name}")
            else:
                mismatches += 1
                print(f"DRIFT: {name}")
                print(_summarise_diff(src, regenerated))
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
        help="Round-trip every tests/fixtures/*.json through the current schema",
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
