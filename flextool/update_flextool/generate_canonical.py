"""Generate flextool/schemas/canonical_databases/*.json from tests/fixtures/*.json.

Reads recipes from ``flextool/schemas/canonical_databases/_recipes.yaml``.

Format bridge
-------------
``tests/fixtures/*.json`` uses **base64-packed** values (via
:mod:`tests.db_utils`).  ``flextool/schemas/canonical_databases/*.json``
uses **UTF-8 decoded** values (via
:mod:`flextool.update_flextool.export_database`).  Rather than translating
between formats directly, each converter is used in its native direction:

1. Read the source JSON via :func:`tests.db_utils.json_to_db` into a temp
   SQLite (base64 -> bytes).
2. Run :func:`flextool.update_flextool.db_migration.migrate_database` —
   idempotent after Stage 2a of the test_fixtures migration.
3. Filter: delete excluded scenarios, alternatives, and entities listed
   in the recipe.  ``spinedb_api`` cascade-removes scenario_alternatives,
   entity_alternatives, parameter_values, and multi-dim child entities
   automatically.
4. Export the SQLite via :func:`flextool.update_flextool.export_database`
   (bytes -> UTF-8) over the canonical JSON.

CLI
---
``python -m flextool.update_flextool.generate_canonical``
    Generate every recipe.

``python -m flextool.update_flextool.generate_canonical templates_examples``
    Generate just one recipe.

``python -m flextool.update_flextool.generate_canonical --verify``
    Regenerate to a temp dir and diff against the committed JSONs; exit
    non-zero with a short summary on drift.  Intended for CI.

Out of scope (still hand-maintained, follow-up work)
----------------------------------------------------
``howto_*.json`` (eight files) and ``templates_time_settings_only.json``
are not yet covered by a recipe.  Adding them requires per-database
investigation (each carves a different slice of the test universe); they
will pick up recipes one at a time in follow-up commits.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from flextool.update_flextool.db_migration import migrate_database
from flextool.update_flextool.export_database import export_database


# ``Path(__file__).resolve().parents[1]`` -> ``flextool/`` package root
# inside the source checkout.  Both the recipe file and the destination
# JSON tree live under ``flextool/schemas/canonical_databases/``.
RECIPES_FILE = (
    Path(__file__).resolve().parents[1]
    / "schemas"
    / "canonical_databases"
    / "_recipes.yaml"
)
OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "schemas" / "canonical_databases"
)
# Source paths in the recipe (``tests/fixtures/...``) are resolved against
# the repository root — one level above ``flextool/``.
SOURCE_ROOT = Path(__file__).resolve().parents[2]


def _load_recipes() -> dict[str, dict[str, Any]]:
    """Parse ``_recipes.yaml`` and return its top-level dict.

    Fails loudly if the file is missing — this module is a source-checkout
    maintenance tool, never invoked from a wheel install.
    """
    if not RECIPES_FILE.is_file():
        raise RuntimeError(f"Recipes file missing: {RECIPES_FILE}")
    with open(RECIPES_FILE) as f:
        return yaml.safe_load(f) or {}


def _filter_db(
    sqlite_url: str,
    exclude_scenarios: list[str],
    exclude_alternatives: list[str],
    exclude_entities: list[list[str]],
) -> None:
    """Apply the recipe's exclusion lists to a staging SQLite.

    Order matters: scenarios first (cascades scenario_alternatives),
    then alternatives (cascades entity_alternatives + parameter_values
    bound to those alternatives), then entities (cascades multi-dim
    child entities and any remaining parameter_values).
    """
    # spinedb_api is the canonical handle for filtering Spine DBs — it
    # owns the cascade semantics so we don't have to track FK ordering
    # by hand.
    from spinedb_api import DatabaseMapping

    if not (exclude_scenarios or exclude_alternatives or exclude_entities):
        # Recipe wants the source verbatim — skip the DatabaseMapping
        # open/commit cycle entirely (spinedb_api raises NothingToCommit
        # if commit_session is called with no pending changes).
        return
    with DatabaseMapping(sqlite_url, create=False, upgrade=False) as db:
        for name in exclude_scenarios:
            db.remove_scenario(name=name)
        for name in exclude_alternatives:
            db.remove_alternative(name=name)
        for cls_name, ent_name in exclude_entities:
            db.remove_entity(entity_class_name=cls_name, name=ent_name)
        db.commit_session("Filtered for canonical export")


def generate_one(recipe_name: str, output_path: Path | None = None) -> Path:
    """Generate a single recipe's output JSON.

    Args:
        recipe_name: Top-level key in ``_recipes.yaml``.
        output_path: Override the default ``OUTPUT_DIR / <recipe>.json``
            destination.  Used by :func:`verify_all` to stage into a
            temp directory.

    Returns the path written.
    """
    recipes = _load_recipes()
    if recipe_name not in recipes:
        raise KeyError(
            f"Unknown recipe {recipe_name!r}. "
            f"Known: {sorted(recipes)}"
        )
    recipe = recipes[recipe_name]
    source_rel = recipe["source"]
    source_abs = (SOURCE_ROOT / source_rel).resolve()
    if not source_abs.is_file():
        raise FileNotFoundError(f"Recipe source missing: {source_abs}")

    exclude_scenarios = recipe.get("exclude_scenarios") or []
    exclude_alternatives = recipe.get("exclude_alternatives") or []
    exclude_entities = recipe.get("exclude_entities") or []

    dest = output_path if output_path is not None else (
        OUTPUT_DIR / f"{recipe_name}.json"
    )

    # The base64 round-trip lives in ``tests.db_utils`` — imported lazily
    # so the module can still be referenced in environments where the
    # ``tests`` package isn't on ``sys.path`` (this matters only for
    # accidental wheel-install invocation; generate_canonical itself is
    # source-checkout only).
    from tests.db_utils import json_to_db

    with tempfile.TemporaryDirectory() as tmp:
        staging_sqlite = Path(tmp) / f"{recipe_name}.sqlite"
        url = json_to_db(source_abs, staging_sqlite)
        migrate_database(url)
        _filter_db(url, exclude_scenarios, exclude_alternatives, exclude_entities)
        export_database(str(staging_sqlite), str(dest))

    return dest


def generate_all() -> list[Path]:
    """Generate every recipe defined in ``_recipes.yaml``.

    Returns the list of destination paths written, in YAML declaration
    order.
    """
    recipes = _load_recipes()
    written: list[Path] = []
    for name in recipes:
        dest = generate_one(name)
        print(f"generated {dest.relative_to(SOURCE_ROOT)}")
        written.append(dest)
    return written


def _summarise_diff(committed: Path, regenerated: Path) -> str:
    """Short top-level-section summary for ``verify``.

    Lists every key whose list length differs (or that's missing on one
    side).  Value-level drift inside a same-length list is reported as
    ``length match, value-level drift``.
    """
    with open(committed) as f:
        a = json.load(f)
    with open(regenerated) as f:
        b = json.load(f)
    keys = sorted(set(a) | set(b))
    lines: list[str] = []
    for key in keys:
        av = a.get(key)
        bv = b.get(key)
        if av == bv:
            continue
        la = len(av) if isinstance(av, list) else "?"
        lb = len(bv) if isinstance(bv, list) else "?"
        if la == lb:
            lines.append(f"  {key}: length match ({la}) — value-level drift")
        else:
            lines.append(f"  {key}: committed={la} regenerated={lb}")
    return "\n".join(lines) if lines else "  (top-level sections match — check whitespace)"


def verify_all() -> int:
    """Regenerate every recipe to a temp dir and diff against committed.

    Returns the count of mismatched recipes (0 on success).  Intended for
    CI: surfaces drift between ``tests/fixtures/*.json`` and the
    committed ``canonical_databases/*.json``.
    """
    recipes = _load_recipes()
    mismatches = 0
    for name in recipes:
        committed = OUTPUT_DIR / f"{name}.json"
        if not committed.is_file():
            print(f"MISSING: {committed}")
            mismatches += 1
            continue
        with tempfile.TemporaryDirectory() as tmp:
            regenerated = Path(tmp) / f"{name}.json"
            generate_one(name, output_path=regenerated)
            with open(committed) as f:
                a = f.read()
            with open(regenerated) as f:
                b = f.read()
            if a == b:
                print(f"ok: {name}.json")
            else:
                mismatches += 1
                print(f"DRIFT: {name}.json")
                print(_summarise_diff(committed, regenerated))
    if mismatches:
        print(
            f"\n{mismatches} canonical database(s) out of sync. "
            "Run `python -m flextool.update_flextool.generate_canonical`."
        )
    return mismatches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "recipe",
        nargs="?",
        default=None,
        help="Recipe name to generate (default: generate all).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="CI mode: regenerate to a temp dir and diff against committed.",
    )
    args = parser.parse_args(argv)

    if args.verify:
        if args.recipe is not None:
            parser.error("--verify takes no positional argument")
        return verify_all()
    if args.recipe is None:
        generate_all()
    else:
        dest = generate_one(args.recipe)
        print(f"generated {dest.relative_to(SOURCE_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
