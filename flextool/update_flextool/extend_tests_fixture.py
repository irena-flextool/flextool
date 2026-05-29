"""Apply a YAML delta to a ``tests/fixtures/*.json`` test fixture.

Stage 2c addendum to the Spine-DB fixture maintenance toolkit.  Sits
alongside :mod:`flextool.update_flextool.test_fixtures` (round-trip
migrator) and :mod:`flextool.update_flextool.generate_canonical`
(subset-and-export filter).  Provides a third, agent-friendly path:
small, declarative additions described in YAML rather than driven
through the SpineDB editor GUI.

Scope
-----
Append-only.  Adds new entities, alternatives, parameter values, and
scenarios.  Editing existing entries is intentionally out of scope —
that workflow remains the SpineDB editor or a Python migration in
``db_migration.py``.  Complex parameter value types (time-series,
maps, arrays) are also out of scope; the YAML accepts scalar values
only (int / float / str / bool / None).  Anything more structured
should be edited via the SpineDB editor.

YAML schema
-----------
::

    target: tests/fixtures/tests.json   # repo-relative path

    new_entities:
      - {class: node, name: my_node, description: "(Optional)"}
      - {class: connection__node__node, entities: [my_conn, src, dst]}

    new_alternatives:
      - {name: my_feature_init, description: "..."}

    new_parameter_values:
      - {class: node, entity: [my_node],
         alternative: my_feature_init,
         parameter: inflow, value: 42.0}

    new_scenarios:
      - {name: my_feature_demo,
         alternatives: [Base, my_feature_init]}

    regenerate_canonical: true   # optional

Workflow
--------
1. Load source ``tests/fixtures/*.json`` into a temp SQLite via
   :func:`tests.db_utils.json_to_db`.
2. Validate the delta against ``flextool/schemas/spinedb_schema.json``
   (entity classes exist, parameters exist) AND against the source
   contents (append-only — names must not collide with existing
   entities / alternatives / scenarios / (entity, parameter,
   alternative) value tuples).
3. Apply the delta via :mod:`spinedb_api`.
4. Export the SQLite back over the source JSON via
   :func:`tests.db_utils.db_to_json`.
5. Optionally regenerate any canonical_databases recipe whose
   ``source`` field references the modified fixture.

CLI
---
::

    python -m flextool.update_flextool.extend_tests_fixture <yaml>
    python -m flextool.update_flextool.extend_tests_fixture <yaml> \\
        --target tests/fixtures/tests.json
    python -m flextool.update_flextool.extend_tests_fixture --validate <yaml>
"""

from __future__ import annotations

import argparse
import difflib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml


# Repo-root resolution mirrors generate_canonical.py / test_fixtures.py —
# this module is a source-checkout maintenance tool, never invoked from
# an installed wheel.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "schemas" / "spinedb_schema.json"
)


# ----------------------------------------------------------------------------
# Schema indexing
# ----------------------------------------------------------------------------


def _load_schema() -> dict[str, Any]:
    """Return the parsed ``spinedb_schema.json``.

    Loaded lazily by :func:`apply_delta` / :func:`validate_delta` rather
    than at import time so the module can be imported in environments
    where the source checkout layout differs (e.g. for type-checking).
    """
    if not _SCHEMA_PATH.is_file():
        raise RuntimeError(f"Schema file missing: {_SCHEMA_PATH}")
    with open(_SCHEMA_PATH) as f:
        return json.load(f)


def _index_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Build the lookup tables needed by :func:`validate_delta`.

    Returns a dict with:

    * ``entity_classes``  : ``{name: dimension_name_list}`` — empty
      tuple for 0-dim ("primary") classes, populated for relationship
      classes.
    * ``parameters``      : ``{(class_name, param_name): None}`` — set
      of valid (class, parameter) pairs.

    Indexed once per :func:`apply_delta` call.  The schema lists are
    short enough (~30 entity_classes, ~230 parameter_definitions) that
    a single linear pass is fine.
    """
    entity_classes: dict[str, tuple[str, ...]] = {}
    for row in schema.get("entity_classes", []):
        # row layout: [name, dimension_name_list, description, ...]
        name = row[0]
        dims = tuple(row[1]) if row[1] else ()
        entity_classes[name] = dims

    parameters: dict[tuple[str, str], None] = {}
    for row in schema.get("parameter_definitions", []):
        # row layout: [entity_class_name, name, default_value, ...]
        parameters[(row[0], row[1])] = None

    return {"entity_classes": entity_classes, "parameters": parameters}


# ----------------------------------------------------------------------------
# Source-state indexing
# ----------------------------------------------------------------------------


def _index_source(source_json: Path) -> dict[str, Any]:
    """Build the lookup tables for append-only enforcement.

    Reads the source JSON directly (no SQLite round-trip needed for
    name lookups).  Returns:

    * ``entities``      : ``set[(class, name)]`` for 0-dim entities;
      ``set[(class, tuple(element_names))]`` for multi-dim entities.
      Both forms are keyed by class, so a name collision in a *different*
      class is correctly allowed.
    * ``alternatives`` : ``set[name]``
    * ``scenarios``    : ``set[name]``
    * ``parameter_values`` : ``set[(class, entity_name, parameter,
      alternative)]`` — the uniqueness key for SpineDB parameter values.

    Used only for validation diagnostics; the actual append happens
    against a fresh SQLite via spinedb_api which enforces the same
    uniqueness constraints at the DB level.
    """
    with open(source_json) as f:
        data = json.load(f)

    entities: set[tuple[str, Any]] = set()
    for row in data.get("entities", []):
        # row layout: [class_name, name_or_element_list, description]
        cls_name = row[0]
        ident = row[1]
        if isinstance(ident, list):
            entities.add((cls_name, tuple(ident)))
        else:
            entities.add((cls_name, ident))

    alternatives = {row[0] for row in data.get("alternatives", [])}
    scenarios = {row[0] for row in data.get("scenarios", [])}

    parameter_values: set[tuple[str, str, str, str]] = set()
    for row in data.get("parameter_values", []):
        # row layout: [class, entity_name, parameter, value, alternative].
        # entity_name is a list for multi-dim entities; flatten it via
        # _multi_dim_name so the lookup key matches the form we build
        # for delta rows (``_multi_dim_name(ent_elements)``).
        ent_name = (
            _multi_dim_name(row[1]) if isinstance(row[1], list) else row[1]
        )
        parameter_values.add((row[0], ent_name, row[2], row[4]))

    return {
        "entities": entities,
        "alternatives": alternatives,
        "scenarios": scenarios,
        "parameter_values": parameter_values,
    }


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------


_SCALAR_TYPES: tuple[type, ...] = (int, float, str, bool, type(None))


def _suggest(name: str, candidates: list[str], cutoff: float = 0.6) -> str:
    """Format a ``did you mean`` clause for an unknown identifier.

    Returns an empty string when no close match exists, otherwise
    " (did you mean 'X'?)" with the single best suggestion.  Uses
    :func:`difflib.get_close_matches` so the threshold matches what
    Python's own ``KeyError`` style suggestions use.
    """
    hits = difflib.get_close_matches(name, candidates, n=1, cutoff=cutoff)
    return f" (did you mean {hits[0]!r}?)" if hits else ""


def _multi_dim_name(elements: list[str]) -> str:
    """Construct SpineDB's canonical multi-dim entity name.

    Spine uses double-underscore concatenation of element names.  We
    surface this in error messages so users can grep their fixture for
    pre-existing collisions.
    """
    return "__".join(elements)


def validate_delta(delta: dict, schema: dict, source_index: dict | None = None) -> list[str]:
    """Return validation errors for a parsed YAML delta.

    Empty list means the delta is safe to apply.  Each string is a
    standalone, readable error message — callers print them one per
    line.

    Two layers of checks:

    * Schema-level (always) — entity_classes exist; parameter names
      exist for the named entity class; values are scalar.
    * Source-level (when ``source_index`` is provided) — append-only:
      no name collisions with existing entities, alternatives,
      scenarios, or (class, entity, parameter, alternative) value
      tuples.

    The two layers are split because the YAML can be schema-valid but
    source-invalid; callers (e.g. the ``--validate`` CLI mode) may want
    schema-only validation when the target fixture isn't accessible.
    """
    errors: list[str] = []
    idx = _index_schema(schema)
    entity_classes: dict[str, tuple[str, ...]] = idx["entity_classes"]
    parameters: dict[tuple[str, str], None] = idx["parameters"]

    # Pre-resolved name lists for suggestion strings.  Build once so
    # the per-row hot loop doesn't re-iterate the schema.
    class_names = list(entity_classes)

    # Track names declared earlier in *this* delta so the second of two
    # identically-named entries in the same YAML is rejected even when
    # the source lookup passes (otherwise a typo would slip through and
    # land as a NothingToCommit at apply time).
    declared_entities: set[tuple[str, Any]] = set()
    declared_alternatives: set[str] = set()
    declared_scenarios: set[str] = set()
    declared_parameter_values: set[tuple[str, str, str, str]] = set()

    src_entities = source_index["entities"] if source_index else set()
    src_alternatives = source_index["alternatives"] if source_index else set()
    src_scenarios = source_index["scenarios"] if source_index else set()
    src_param_values = source_index["parameter_values"] if source_index else set()

    # --- new_entities ----------------------------------------------------
    for i, row in enumerate(delta.get("new_entities") or []):
        prefix = f"new_entities[{i}]"
        if not isinstance(row, dict):
            errors.append(f"{prefix}: expected a mapping, got {type(row).__name__}")
            continue
        cls = row.get("class")
        if not cls:
            errors.append(f"{prefix}: missing required field 'class'")
            continue
        if cls not in entity_classes:
            errors.append(
                f"{prefix}: entity_class {cls!r} not in schema"
                + _suggest(cls, class_names)
            )
            continue
        dims = entity_classes[cls]
        if dims:
            # multi-dim: 'entities' is required, 'name' is forbidden
            if "name" in row:
                errors.append(
                    f"{prefix}: multi-dim class {cls!r} takes 'entities: [...]', not 'name'"
                )
                continue
            elems = row.get("entities")
            if not isinstance(elems, list) or not all(isinstance(e, str) for e in elems):
                errors.append(
                    f"{prefix}: multi-dim class {cls!r} requires 'entities: [...]' as a list of strings"
                )
                continue
            if len(elems) != len(dims):
                errors.append(
                    f"{prefix}: class {cls!r} has {len(dims)} dimensions {list(dims)}, "
                    f"got {len(elems)} element(s) {elems!r}"
                )
                continue
            key = (cls, tuple(elems))
            display = _multi_dim_name(elems)
        else:
            if "entities" in row:
                errors.append(
                    f"{prefix}: 0-dim class {cls!r} takes 'name', not 'entities'"
                )
                continue
            name = row.get("name")
            if not isinstance(name, str):
                errors.append(f"{prefix}: 0-dim class {cls!r} requires a string 'name'")
                continue
            key = (cls, name)
            display = name
        if key in src_entities:
            errors.append(
                f"{prefix}: entity {display!r} of class {cls!r} already exists in target "
                "(append-only — edits go through the SpineDB editor)"
            )
            continue
        if key in declared_entities:
            errors.append(
                f"{prefix}: entity {display!r} of class {cls!r} declared more than once in this delta"
            )
            continue
        declared_entities.add(key)

    # --- new_alternatives ------------------------------------------------
    for i, row in enumerate(delta.get("new_alternatives") or []):
        prefix = f"new_alternatives[{i}]"
        if not isinstance(row, dict):
            errors.append(f"{prefix}: expected a mapping, got {type(row).__name__}")
            continue
        name = row.get("name")
        if not isinstance(name, str):
            errors.append(f"{prefix}: missing or non-string 'name'")
            continue
        if name in src_alternatives:
            errors.append(
                f"{prefix}: alternative {name!r} already exists in target "
                "(append-only)"
            )
            continue
        if name in declared_alternatives:
            errors.append(f"{prefix}: alternative {name!r} declared more than once in this delta")
            continue
        declared_alternatives.add(name)

    # --- new_parameter_values --------------------------------------------
    # The set of entity *names* available after this delta is applied =
    # source entities ∪ entities declared earlier in this YAML.  For
    # 0-dim classes we compare by name; for multi-dim, by element tuple
    # (the SpineDB convention).
    available_entities: set[tuple[str, Any]] = set(src_entities) | declared_entities
    available_alternatives: set[str] = set(src_alternatives) | declared_alternatives

    for i, row in enumerate(delta.get("new_parameter_values") or []):
        prefix = f"new_parameter_values[{i}]"
        if not isinstance(row, dict):
            errors.append(f"{prefix}: expected a mapping, got {type(row).__name__}")
            continue
        cls = row.get("class")
        if not cls or cls not in entity_classes:
            errors.append(
                f"{prefix}: entity_class {cls!r} not in schema"
                + _suggest(cls or "", class_names)
            )
            continue
        param = row.get("parameter")
        if not isinstance(param, str):
            errors.append(f"{prefix}: missing or non-string 'parameter'")
            continue
        if (cls, param) not in parameters:
            class_params = [p for c, p in parameters if c == cls]
            errors.append(
                f"{prefix}: parameter {param!r} not in schema for class {cls!r}"
                + _suggest(param, class_params)
            )
            continue
        # 'entity' is always a list of element names — for 0-dim, a
        # single-element list.  This keeps the YAML uniform whether the
        # class is 0-dim or multi-dim.
        ent = row.get("entity")
        if not isinstance(ent, list) or not all(isinstance(e, str) for e in ent):
            errors.append(f"{prefix}: 'entity' must be a list of strings")
            continue
        dims = entity_classes[cls]
        if dims and len(ent) != len(dims):
            errors.append(
                f"{prefix}: class {cls!r} has {len(dims)} dimensions, got {len(ent)} element(s)"
            )
            continue
        if not dims and len(ent) != 1:
            errors.append(
                f"{prefix}: 0-dim class {cls!r} requires exactly one element in 'entity'"
            )
            continue
        entity_key: tuple[str, Any] = (cls, tuple(ent) if dims else ent[0])
        if entity_key not in available_entities:
            display = _multi_dim_name(ent) if dims else ent[0]
            errors.append(
                f"{prefix}: entity {display!r} of class {cls!r} not found in target "
                "(and not declared earlier in this delta)"
            )
            continue
        # Default alternative to "Base" — that's the SpineDB convention
        # and the name used by the FlexTool test fixtures.
        alt = row.get("alternative", "Base")
        if not isinstance(alt, str):
            errors.append(f"{prefix}: 'alternative' must be a string")
            continue
        if alt not in available_alternatives:
            errors.append(
                f"{prefix}: alternative {alt!r} not found in target "
                "(and not declared earlier in this delta)"
                + _suggest(alt, sorted(available_alternatives))
            )
            continue
        value = row.get("value", ...)
        if value is ...:
            errors.append(f"{prefix}: missing required field 'value'")
            continue
        if isinstance(value, (dict, list)):
            errors.append(
                f"{prefix}: structured 'value' ({type(value).__name__}) is out of scope. "
                "This script supports scalar values only (int, float, str, bool, None). "
                "For time-series, maps, or arrays use the SpineDB editor."
            )
            continue
        if not isinstance(value, _SCALAR_TYPES):
            errors.append(
                f"{prefix}: 'value' must be int/float/str/bool/None, got {type(value).__name__}"
            )
            continue
        # SpineDB's parameter_value key.  Use the element-list-joined
        # name for multi-dim entities (the form db_to_json round-trips).
        entity_name = _multi_dim_name(ent) if dims else ent[0]
        pv_key = (cls, entity_name, param, alt)
        if pv_key in src_param_values:
            errors.append(
                f"{prefix}: parameter_value ({cls}, {entity_name}, {param}, {alt}) "
                "already exists in target (append-only — edits go through the SpineDB editor)"
            )
            continue
        if pv_key in declared_parameter_values:
            errors.append(
                f"{prefix}: parameter_value ({cls}, {entity_name}, {param}, {alt}) "
                "declared more than once in this delta"
            )
            continue
        declared_parameter_values.add(pv_key)

    # --- new_scenarios ---------------------------------------------------
    for i, row in enumerate(delta.get("new_scenarios") or []):
        prefix = f"new_scenarios[{i}]"
        if not isinstance(row, dict):
            errors.append(f"{prefix}: expected a mapping, got {type(row).__name__}")
            continue
        name = row.get("name")
        if not isinstance(name, str):
            errors.append(f"{prefix}: missing or non-string 'name'")
            continue
        if name in src_scenarios:
            errors.append(
                f"{prefix}: scenario {name!r} already exists in target (append-only)"
            )
            continue
        if name in declared_scenarios:
            errors.append(f"{prefix}: scenario {name!r} declared more than once in this delta")
            continue
        alts = row.get("alternatives")
        if not isinstance(alts, list) or not alts or not all(isinstance(a, str) for a in alts):
            errors.append(f"{prefix}: 'alternatives' must be a non-empty list of strings")
            continue
        for a in alts:
            if a not in available_alternatives:
                errors.append(
                    f"{prefix}: alternative {a!r} not found in target "
                    "(and not declared earlier in this delta)"
                    + _suggest(a, sorted(available_alternatives))
                )
                break
        else:
            declared_scenarios.add(name)
            continue
        # Reached only when the for/else 'break' fired; do nothing more.

    return errors


# ----------------------------------------------------------------------------
# Application
# ----------------------------------------------------------------------------


def _apply_delta_to_db(sqlite_url: str, delta: dict, entity_classes: dict[str, tuple[str, ...]]) -> None:
    """Apply the delta to an open SpineDB.

    Splits apart from :func:`apply_delta` so the spinedb_api dependency
    stays out of the validation hot path.  All four sections share one
    ``DatabaseMapping`` context and one ``commit_session`` so a partial
    failure rolls back cleanly.
    """
    # Lazy import so module import doesn't pay the spinedb_api startup
    # cost when only validation is requested.
    from spinedb_api import DatabaseMapping, to_database

    with DatabaseMapping(sqlite_url, create=False, upgrade=False) as db:
        # --- entities ----------------------------------------------------
        for row in delta.get("new_entities") or []:
            cls = row["class"]
            dims = entity_classes[cls]
            if dims:
                db.add_entity(
                    entity_class_name=cls,
                    element_name_list=tuple(row["entities"]),
                    description=row.get("description"),
                )
            else:
                db.add_entity(
                    entity_class_name=cls,
                    name=row["name"],
                    description=row.get("description"),
                )
        # --- alternatives ------------------------------------------------
        for row in delta.get("new_alternatives") or []:
            db.add_alternative(
                name=row["name"],
                description=row.get("description", ""),
            )
        # --- parameter values --------------------------------------------
        for row in delta.get("new_parameter_values") or []:
            cls = row["class"]
            dims = entity_classes[cls]
            ent_elements = row["entity"]
            value, type_ = to_database(row["value"])
            db.add_parameter_value(
                entity_class_name=cls,
                # SpineDB needs the *byname* tuple — for 0-dim that's a
                # one-element tuple, for multi-dim the element list.
                entity_byname=tuple(ent_elements),
                parameter_definition_name=row["parameter"],
                alternative_name=row.get("alternative", "Base"),
                value=value,
                type=type_,
            )
        # --- scenarios + ranked scenario_alternatives --------------------
        for row in delta.get("new_scenarios") or []:
            name = row["name"]
            db.add_scenario(name=name, description=row.get("description", ""))
            # SpineDB stores scenario_alternatives as a linked list
            # serialised by rank.  We assign rank 1..N in declaration
            # order — this matches what generate_canonical / the
            # SpineDB editor write.
            for rank, alt in enumerate(row["alternatives"], start=1):
                db.add_scenario_alternative(
                    scenario_name=name,
                    alternative_name=alt,
                    rank=rank,
                )
        db.commit_session("extend_tests_fixture YAML delta")


# ----------------------------------------------------------------------------
# Recipe regeneration
# ----------------------------------------------------------------------------


def _regenerate_canonical_for(target_rel: str) -> list[str]:
    """Regenerate every canonical recipe whose ``source`` matches the target.

    ``target_rel`` is the repo-relative path of the modified fixture
    (e.g. ``tests/fixtures/tests.json``).  Returns the list of recipe
    names that were regenerated, in declaration order.

    No-op when no recipe references the fixture — many test fixtures
    have no downstream canonical output (they're test-only).
    """
    # Lazy import to keep the validation path cheap and to avoid the
    # circular structure if generate_canonical were ever to import this
    # module (it currently doesn't, but the lazy guard is cheap).
    from flextool.update_flextool.generate_canonical import (
        _load_recipes,
        generate_one,
    )

    recipes = _load_recipes()
    regenerated: list[str] = []
    for name, recipe in recipes.items():
        if recipe.get("source") == target_rel:
            generate_one(name)
            regenerated.append(name)
    return regenerated


# ----------------------------------------------------------------------------
# Top-level API
# ----------------------------------------------------------------------------


def _resolve_target(delta: dict, target_path: Path | None) -> Path:
    """Resolve and validate the target fixture path.

    Honours, in order: explicit ``target_path`` argument, then the
    ``target`` field from the YAML.  The result is resolved against
    the repo root when relative.  Fails loudly if neither is supplied
    or the file is missing.
    """
    if target_path is not None:
        candidate = target_path
    else:
        target_rel = delta.get("target")
        if not target_rel:
            raise ValueError(
                "YAML delta is missing the 'target' field and no --target was given"
            )
        candidate = Path(target_rel)
    if not candidate.is_absolute():
        candidate = (_REPO_ROOT / candidate).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Target fixture not found: {candidate}")
    return candidate


def apply_delta(
    yaml_path: Path,
    target_path: Path | None = None,
    regenerate_canonical: bool | None = None,
) -> int:
    """Apply a YAML delta to a ``tests/fixtures/*.json``.

    Args:
        yaml_path: Path to the YAML delta file.
        target_path: If provided, overrides the YAML's ``target`` field.
        regenerate_canonical: If not None, overrides the YAML's
            ``regenerate_canonical`` field.

    Returns 0 on success, non-zero on validation failure (with
    per-error diagnostics printed to stderr).  All-or-nothing: the
    SQLite is committed atomically, then re-exported over the source
    JSON only if validation and apply both succeed.
    """
    if not yaml_path.is_file():
        print(f"YAML delta file not found: {yaml_path}", file=sys.stderr)
        return 2
    with open(yaml_path) as f:
        delta = yaml.safe_load(f) or {}

    target = _resolve_target(delta, target_path)

    schema = _load_schema()
    source_index = _index_source(target)
    errors = validate_delta(delta, schema, source_index=source_index)
    if errors:
        print(f"Validation failed for {yaml_path} (target: {target}):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    # Imported lazily — mirrors the pattern in generate_canonical.py.
    from tests.db_utils import db_to_json, json_to_db

    entity_classes = _index_schema(schema)["entity_classes"]

    with tempfile.TemporaryDirectory() as tmp:
        staging_sqlite = Path(tmp) / "staging.sqlite"
        url = json_to_db(target, staging_sqlite)
        _apply_delta_to_db(url, delta, entity_classes)
        # Export over a staging path first, then move atomically over
        # the source — protects against half-written JSON if db_to_json
        # raises mid-write.
        staging_json = Path(tmp) / "out.json"
        db_to_json(staging_sqlite, staging_json)
        shutil.copyfile(staging_json, target)

    regen = regenerate_canonical if regenerate_canonical is not None else bool(
        delta.get("regenerate_canonical")
    )
    if regen:
        # Recipes are keyed by repo-relative paths.  Recompute the
        # relative form (the resolved Path may differ in case / symlink
        # form on weird platforms) before lookup.
        try:
            target_rel = str(target.relative_to(_REPO_ROOT))
        except ValueError:
            print(
                f"Cannot regenerate canonical: target {target} is outside repo root {_REPO_ROOT}",
                file=sys.stderr,
            )
            return 3
        regenerated = _regenerate_canonical_for(target_rel)
        if regenerated:
            print(f"Regenerated canonical: {regenerated}")
        else:
            print(
                f"regenerate_canonical=true, but no recipe references {target_rel} — "
                "nothing to regenerate"
            )

    print(f"Applied delta {yaml_path.name} -> {target}")
    return 0


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("yaml", type=Path, help="Path to the YAML delta file.")
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="Override the YAML's 'target' field (repo-relative or absolute).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation only; do not write to the target fixture.",
    )
    parser.add_argument(
        "--regenerate-canonical",
        dest="regenerate_canonical",
        action="store_true",
        default=None,
        help="Force regenerate-canonical (override YAML's regenerate_canonical field).",
    )
    args = parser.parse_args(argv)

    if args.validate:
        if not args.yaml.is_file():
            print(f"YAML delta file not found: {args.yaml}", file=sys.stderr)
            return 2
        with open(args.yaml) as f:
            delta = yaml.safe_load(f) or {}
        schema = _load_schema()
        source_index: dict | None = None
        try:
            target = _resolve_target(delta, args.target)
            source_index = _index_source(target)
        except (ValueError, FileNotFoundError) as e:
            # Source not resolvable -> schema-only validation.  Useful
            # for editing a YAML without the repo checkout's target on
            # disk (rare but supported).
            print(f"NOTE: skipping source-level checks ({e})", file=sys.stderr)
        errors = validate_delta(delta, schema, source_index=source_index)
        if errors:
            print(f"Validation failed for {args.yaml}:", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            return 1
        print(f"OK: {args.yaml} validates against current schema")
        return 0

    return apply_delta(
        yaml_path=args.yaml,
        target_path=args.target,
        regenerate_canonical=args.regenerate_canonical,
    )


if __name__ == "__main__":
    raise SystemExit(main())
