"""One-off: correct the v58 description fields + flowGroup icon in place.

The v58 migration (``_migrate_v58_carve_flowgroup_out_of_group`` in
``flextool/update_flextool/db_migration.py``) was committed with
placeholder/wrong values for three parameter descriptions and with no
``display_icon`` on the ``flowGroup`` entity class.  The migration block
itself has now been corrected, but every committed artifact that is
*already at schema v58* keeps the old values, because the idempotent
``migrate-all`` is a no-op on a v58 DB and never rewrites descriptions.

This script patches those artifacts directly, via the DatabaseMapping
API, so they match the corrected migration:

* ``flowGroup.flow_aggregator``      -> new description (see FLOW_AGGREGATOR_DESC)
* ``group.print_dispatch``           -> new description
* ``group.print_indicators``         -> new description
* ``flowGroup`` entity-class          -> ``display_icon`` = 143488754774337
  (group's icon code in the low 24 bits + colour #828094 in the high
  24 bits, per Spine's ``display_icon = icon_code + (color_code << 24)``).

Loader distinction (must match the round-trip each file family uses so the
output stays byte-compatible with CI verify):

* ``tests/fixtures/*.json``           — base64 round-trip
  (``tests.db_utils.json_to_db`` / ``db_to_json``).
* canonical ``*.json``                — UTF-8 round-trip
  (``initialize_database`` / ``export_database``).
* ``flextool/schemas/spinedb_schema.json`` — ``import_data`` / ``export_data``
  round-trip (the same path ``sync_master_json_template`` uses).

The two GENERATED canonical files (``templates_examples.json`` from
``tests.json``; ``howto_stochastics.json`` from ``stochastics.json`` — see
``canonical_databases/_recipes.yaml``) are NOT patched here.  Patch their
SOURCE fixtures (handled by this script) and then run
``python -m flextool.update_flextool.generate_canonical`` to regenerate
them, so ``generate_canonical --verify`` stays green.

Defs/classes absent from a given DB are skipped (only ``null_override_*``
fixtures lack them; everything migrated through v58 has them).

Run::

    ~/venv-spi/bin/python tests/fixtures/_fix_v58_descriptions.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json  # noqa: E402

from spinedb_api import DatabaseMapping, export_data, import_data  # noqa: E402

from flextool.update_flextool.export_database import export_database  # noqa: E402
from flextool.update_flextool.initialize_database import initialize_database  # noqa: E402
from tests.db_utils import db_to_json, json_to_db  # noqa: E402

# --- Corrected values (kept byte-for-byte identical to db_migration.py) ----

FLOW_AGGREGATOR_DESC = (
    "Choice of flow-aggregation method for this flowGroup: 'none' (used for flow limits only, "
    "no aggregated output), 'dispatch_plots_only' (used as an aggregated flowGroup within "
    "'print_dispatch' group of nodes - do not include one flow in multiple flow aggregators to "
    "avoid double counting), 'standalone_aggregator_only' (just make time series available for "
    "manual output processing / simple plots), 'both'."
)
PRINT_DISPATCH_DESC = (
    "Combines unit/connection flows, flowGroup aggregated flows, and other balance elements "
    "(slack, demand, internal_losses, etc.) into a set of timeseries and a preconfigured plot per group."
)
PRINT_INDICATORS_DESC = (
    "Output indicators for the set of nodes in the group (Loss of Load, VRE generation, etc.)."
)
FLOWGROUP_DISPLAY_ICON = 143488754774337

# Parameter-definition descriptions to set: (entity_class, name, description).
PDEF_DESCS = (
    ("flowGroup", "flow_aggregator", FLOW_AGGREGATOR_DESC),
    ("group", "print_dispatch", PRINT_DISPATCH_DESC),
    ("group", "print_indicators", PRINT_INDICATORS_DESC),
)

# tests/fixtures sources of GENERATED canonical files — regenerated, not patched.
GENERATED_FROM = {
    "tests.json": "templates_examples",
    "stochastics.json": "howto_stochastics",
}


def _patch_db(url: str) -> None:
    """Apply the description + icon corrections to an open DB, then commit.

    Idempotent and tolerant of absent defs/classes (skips them).
    """
    changed = False
    with DatabaseMapping(url) as db:
        # flowGroup entity-class display_icon.
        ecs = db.find_entity_classes(name="flowGroup")
        if ecs:
            if ecs[0]["display_icon"] != FLOWGROUP_DISPLAY_ICON:
                # update_entity_class returns a single PublicItem (not a tuple),
                # raising on failure; verified explicitly via the re-read below.
                db.update_entity_class(
                    name="flowGroup", display_icon=FLOWGROUP_DISPLAY_ICON
                )
                changed = True
            after = db.find_entity_classes(name="flowGroup")[0]["display_icon"]
            if after != FLOWGROUP_DISPLAY_ICON:
                raise RuntimeError(
                    f"flowGroup display_icon expected {FLOWGROUP_DISPLAY_ICON}, got {after}"
                )

        # Parameter-definition descriptions.
        for cls, name, desc in PDEF_DESCS:
            pdefs = db.find_parameter_definitions(entity_class_name=cls, name=name)
            if not pdefs:
                continue
            if pdefs[0]["description"] != desc:
                _, err = db.update_item(
                    "parameter_definition", id=pdefs[0]["id"], description=desc
                )
                if err:
                    raise RuntimeError(f"{cls}.{name} description update failed: {err}")
                changed = True
            after = db.find_parameter_definitions(entity_class_name=cls, name=name)[0]
            if after["description"] != desc:
                raise RuntimeError(
                    f"{cls}.{name} description mismatch after update:\n"
                    f"  expected: {desc!r}\n  got:      {after['description']!r}"
                )

        if changed:
            db.commit_session("v58: correct flow_aggregator / print_* descriptions + flowGroup icon")


def _has_target_defs(json_path: Path) -> bool:
    """True iff the JSON carries any of the v58 defs this patch touches.

    Guards against round-tripping hand-authored fixtures that are NOT
    managed by the round-trip path (e.g. ``null_override_clears_map.json``):
    re-emitting those reformats / reorders them, producing a spurious
    structural diff.  Only DBs migrated through v58 carry these defs.
    """
    with open(json_path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return False
    pdefs = data.get("parameter_definitions", [])
    if any(p[0] == "flowGroup" and p[1] == "flow_aggregator" for p in pdefs):
        return True
    if any(p[0] == "group" and p[1] in ("print_dispatch", "print_indicators") for p in pdefs):
        return True
    return any(ec[0] == "flowGroup" for ec in data.get("entity_classes", []))


def _fix_fixture(json_path: Path) -> None:
    """tests/fixtures/*.json — base64 round-trip."""
    if not _has_target_defs(json_path):
        print(f"skip (no v58 defs)  {json_path.relative_to(REPO_ROOT)}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        sqlite_path = Path(tmp) / "staging.sqlite"
        url = json_to_db(json_path, sqlite_path)
        _patch_db(url)
        db_to_json(sqlite_path, json_path)
    print(f"rewrote (fixture)   {json_path.relative_to(REPO_ROOT)}")


def _fix_canonical(json_path: Path) -> None:
    """canonical *.json — UTF-8 initialize_database / export_database round-trip."""
    if not _has_target_defs(json_path):
        print(f"skip (no v58 defs)  {json_path.relative_to(REPO_ROOT)}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        sqlite_path = Path(tmp) / "staging.sqlite"
        initialize_database(str(json_path), str(sqlite_path))
        _patch_db(f"sqlite:///{sqlite_path}")
        export_database(str(sqlite_path), str(json_path))
    print(f"rewrote (canonical) {json_path.relative_to(REPO_ROOT)}")


def _fix_schema(json_path: Path) -> None:
    """spinedb_schema.json — import_data / export_data round-trip (sync's path)."""
    with open(json_path) as f:
        template = json.load(f)
    with tempfile.TemporaryDirectory() as tmp:
        sqlite_path = Path(tmp) / "schema.sqlite"
        url = f"sqlite:///{sqlite_path}"
        with DatabaseMapping(url, create=True) as db:
            for alt in db.find_alternatives():
                try:
                    db.remove_alternative(name=alt["name"])
                except Exception:
                    pass
            import_data(db, **template)
            db.commit_session("Initialized from master template")
        _patch_db(url)
        with DatabaseMapping(url) as db:
            data = export_data(db)
        new_data = {k: [list(item) for item in v] for k, v in data.items()}
    new_json = json.dumps(new_data, indent=2, ensure_ascii=False) + "\n"
    with open(json_path, "w") as f:
        f.write(new_json)
    print(f"rewrote (schema)    {json_path.relative_to(REPO_ROOT)}")


def main() -> None:
    fixtures_dir = REPO_ROOT / "tests" / "fixtures"
    canonical_dir = REPO_ROOT / "flextool" / "schemas" / "canonical_databases"
    schema = REPO_ROOT / "flextool" / "schemas" / "spinedb_schema.json"

    # 1. Master schema.
    _fix_schema(schema)

    # 2. tests/fixtures/*.json (base64) — including the generated-canonical sources.
    for json_path in sorted(fixtures_dir.glob("*.json")):
        _fix_fixture(json_path)

    # 3. Canonical *.json — but skip the GENERATED ones (regenerated below).
    generated_outputs = set(GENERATED_FROM.values())
    for json_path in sorted(canonical_dir.glob("*.json")):
        if json_path.stem in generated_outputs:
            continue
        _fix_canonical(json_path)

    # 4. Regenerate the generated canonical files from their patched sources.
    from flextool.update_flextool.generate_canonical import generate_all

    print("regenerating generated canonical files ...")
    generate_all()


if __name__ == "__main__":
    main()
