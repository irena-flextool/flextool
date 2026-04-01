"""Regenerate flextool_template_master.json from itself + migrations.

Workflow:
  1. Create a temporary DB from the current flextool_template_master.json
  2. Run db_migration.py on it (applies any new migration steps)
  3. Export the migrated DB back to flextool_template_master.json

This ensures the master template always reflects the latest schema
after db_migration.py has been updated.

Usage:
    python -m flextool.update_flextool.sync_master_json_template [--verify]

    --verify  Don't overwrite; just check if the template is up to date.
              Exits with code 1 if it needs regeneration (for CI).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from spinedb_api import DatabaseMapping, export_data, import_data

from flextool.update_flextool.db_migration import migrate_database

VERSION_DIR = Path(__file__).resolve().parent.parent.parent / "version"
MASTER_TEMPLATE = VERSION_DIR / "flextool_template_master.json"


def _create_db_from_template(template_path: Path, db_path: Path) -> None:
    """Create a fresh SQLite DB from a JSON template."""
    with open(template_path) as f:
        template = json.load(f)

    url = f"sqlite:///{db_path}"
    with DatabaseMapping(url, create=True) as db:
        for alt in db.find_alternatives():
            try:
                db.remove_alternative(name=alt["name"])
            except Exception:
                pass
        import_data(db, **template)
        db.commit_session("Initialized from master template")


def _export_db_to_json(db_path: Path) -> dict:
    """Export a Spine DB to a JSON-serializable dict."""
    url = f"sqlite:///{db_path}"
    with DatabaseMapping(url) as db:
        data = export_data(db)
    # export_data returns tuples; convert to lists for JSON
    return {k: [list(item) for item in v] for k, v in data.items()}


def sync_master_template(*, verify_only: bool = False) -> bool:
    """Regenerate (or verify) the master JSON template.

    Returns True if the template is up to date, False if it needed updating.
    """
    if not MASTER_TEMPLATE.exists():
        print(f"ERROR: Master template not found: {MASTER_TEMPLATE}", file=sys.stderr)
        sys.exit(2)

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "template.sqlite"

        # Step 1: Create DB from current master template
        _create_db_from_template(MASTER_TEMPLATE, db_path)

        # Step 2: Run all migrations
        migrate_database(str(db_path))

        # Step 3: Export migrated DB
        new_data = _export_db_to_json(db_path)

    # Step 4: Compare or overwrite
    new_json = json.dumps(new_data, indent=2, ensure_ascii=False) + "\n"

    with open(MASTER_TEMPLATE) as f:
        old_json = f.read()

    if new_json == old_json:
        print("Master template is up to date.")
        return True

    if verify_only:
        # Show what changed
        old_data = json.loads(old_json)
        for key in sorted(set(list(old_data.keys()) + list(new_data.keys()))):
            old_count = len(old_data.get(key, []))
            new_count = len(new_data.get(key, []))
            if old_count != new_count:
                print(f"  {key}: {old_count} → {new_count}")
        print("Master template is OUT OF DATE. Run without --verify to regenerate.")
        return False

    with open(MASTER_TEMPLATE, "w") as f:
        f.write(new_json)
    print(f"Regenerated {MASTER_TEMPLATE}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate flextool_template_master.json after migration changes."
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check if template is up to date without overwriting (for CI).",
    )
    args = parser.parse_args()

    up_to_date = sync_master_template(verify_only=args.verify)
    if not up_to_date:
        sys.exit(1)


if __name__ == "__main__":
    main()
