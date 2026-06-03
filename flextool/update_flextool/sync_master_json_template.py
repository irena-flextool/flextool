"""Regenerate schemas/spinedb_schema.json from itself + migrations.

Workflow:
  1. Create a temporary DB from the current schemas/spinedb_schema.json
  2. Run db_migration.py on it (applies any new migration steps)
  3. Export the migrated DB back to schemas/spinedb_schema.json

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

from flextool._resources import package_data_path

SCHEMAS_DIR = package_data_path("schemas")
SPINEDB_SCHEMA = SCHEMAS_DIR / "spinedb_schema.json"
RESULTS_SCHEMA = SCHEMAS_DIR / "spinedb_results_schema.json"

# Legacy FlexTool repo holding the authoritative results template SpineDB.
# The engine repo has no checked-in copy of this template, so the regenerator
# reads it from the legacy path. Override via the ``template_path`` argument.
DEFAULT_RESULTS_TEMPLATE = Path(
    "/home/jkiviluo/sources/flextool/templates/results_template.sqlite"
)

# Duplicate reserve entity classes carried by the legacy results template.
# They have identical dimensions to the canonical ``unit__reserve__upDown__node``
# class, carry no parameter_definitions, and nothing references them. They are
# trimmed from the exported results schema; the canonical
# ``unit__reserve__upDown__node`` and ``connection__reserve__upDown__node``
# classes are kept.
_RESULTS_TRIM_CLASSES = frozenset(
    {
        "unit__reserve__upDown__node.1",
        "unit__reserve__upDown__node.2",
        "unit__reserve__upDown__node.3",
    }
)


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
    if not SPINEDB_SCHEMA.exists():
        print(f"ERROR: Master template not found: {SPINEDB_SCHEMA}", file=sys.stderr)
        sys.exit(2)

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "template.sqlite"

        # Step 1: Create DB from current master template
        _create_db_from_template(SPINEDB_SCHEMA, db_path)

        # Step 2: Run all migrations
        migrate_database(str(db_path))

        # Step 3: Export migrated DB
        new_data = _export_db_to_json(db_path)

    # Step 4: Compare or overwrite
    new_json = json.dumps(new_data, indent=2, ensure_ascii=False) + "\n"

    with open(SPINEDB_SCHEMA) as f:
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
                print(f"  {key}: {old_count} -> {new_count}")
        print("Master template is OUT OF DATE. Run without --verify to regenerate.")
        return False

    with open(SPINEDB_SCHEMA, "w") as f:
        f.write(new_json)
    print(f"Regenerated {SPINEDB_SCHEMA}")
    return True


def regenerate_results_schema_json(
    *,
    template_path: Path = DEFAULT_RESULTS_TEMPLATE,
    output_path: Path = RESULTS_SCHEMA,
    verify_only: bool = False,
) -> bool:
    """Regenerate (or verify) ``schemas/spinedb_results_schema.json``.

    Exports the legacy results-template SpineDB via ``export_data`` into the
    same top-level shape as the input schema (``entity_classes``,
    ``parameter_definitions``, ``alternatives``), trims the duplicate reserve
    classes (``unit__reserve__upDown__node.1/.2/.3``), and writes the result as
    JSON. Mirrors :func:`_export_db_to_json` for serialization consistency.

    Returns True if ``output_path`` is up to date, False if it needed updating.
    """
    template_path = Path(template_path)
    if not template_path.exists():
        print(
            f"ERROR: Results template not found: {template_path}", file=sys.stderr
        )
        sys.exit(2)

    url = f"sqlite:///{template_path}"
    with DatabaseMapping(url) as db:
        data = export_data(db)

    # Trim duplicate reserve classes; nothing references them and they carry
    # no parameter_definitions, so the param count is unaffected.
    entity_classes = [
        ec for ec in data.get("entity_classes", []) if ec[0] not in _RESULTS_TRIM_CLASSES
    ]
    data["entity_classes"] = entity_classes

    # export_data returns tuples; convert to lists for JSON (mirrors
    # _export_db_to_json).
    new_data = {k: [list(item) for item in v] for k, v in data.items()}
    new_json = json.dumps(new_data, indent=2, ensure_ascii=False) + "\n"

    if output_path.exists():
        with open(output_path) as f:
            old_json = f.read()
        if new_json == old_json:
            print("Results schema is up to date.")
            return True
        if verify_only:
            old_data = json.loads(old_json)
            for key in sorted(set(list(old_data.keys()) + list(new_data.keys()))):
                old_count = len(old_data.get(key, []))
                this_count = len(new_data.get(key, []))
                if old_count != this_count:
                    print(f"  {key}: {old_count} -> {this_count}")
            print(
                "Results schema is OUT OF DATE. Run without --verify to regenerate."
            )
            return False
    elif verify_only:
        print(f"Results schema is MISSING: {output_path}")
        return False

    with open(output_path, "w") as f:
        f.write(new_json)
    print(f"Regenerated {output_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate schemas/spinedb_schema.json after migration changes."
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check if template is up to date without overwriting (for CI).",
    )
    parser.add_argument(
        "--results",
        action="store_true",
        help="Regenerate schemas/spinedb_results_schema.json instead of the input schema.",
    )
    args = parser.parse_args()

    if args.results:
        up_to_date = regenerate_results_schema_json(verify_only=args.verify)
    else:
        up_to_date = sync_master_template(verify_only=args.verify)
    if not up_to_date:
        sys.exit(1)


if __name__ == "__main__":
    main()
