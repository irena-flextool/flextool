"""Round-trip test for the committed FlexTool results schema JSON.

The committed ``spinedb_results_schema.json`` is the single source of truth for
the results-DB scaffold (CLAUDE.md invariant 3: never read a checked-in
``.sqlite`` for data). This test loads the JSON, imports it into a fresh tmp
sqlite, reopens, and asserts the trimmed class set and parameter counts.
"""
from __future__ import annotations

import json

from spinedb_api import DatabaseMapping, import_data

from flextool._resources import package_data_path

RESULTS_SCHEMA = package_data_path("schemas") / "spinedb_results_schema.json"

# Post-trim ground truth (measured from the legacy results template after the
# .1/.2/.3 reserve-class trim).
EXPECTED_ENTITY_CLASSES = 12
EXPECTED_PARAMETER_DEFINITIONS = 58  # 52 base + 6 per-entity cost break-down
#   (unit/connection/node × cost_annualized/cost_discounted)

TRIMMED_CLASSES = {
    "unit__reserve__upDown__node.1",
    "unit__reserve__upDown__node.2",
    "unit__reserve__upDown__node.3",
}
CANONICAL_RESERVE_CLASSES = {
    "unit__reserve__upDown__node",
    "connection__reserve__upDown__node",
}


def test_results_schema_roundtrip(tmp_path):
    with open(RESULTS_SCHEMA) as f:
        schema = json.load(f)

    # JSON shape mirrors the input schema export.
    assert "entity_classes" in schema
    assert "parameter_definitions" in schema
    assert "alternatives" in schema

    db_url = f"sqlite:///{tmp_path / 'results.sqlite'}"
    with DatabaseMapping(db_url, create=True) as db:
        count, errors = import_data(db, **schema)
        assert errors == [], f"import_data reported errors: {errors}"
        assert count > 0
        db.commit_session("Imported results schema")

    with DatabaseMapping(db_url) as db:
        class_names = {ec["name"] for ec in db.find_entity_classes()}
        param_defs = list(db.find_parameter_definitions())

    assert len(class_names) == EXPECTED_ENTITY_CLASSES
    assert len(param_defs) == EXPECTED_PARAMETER_DEFINITIONS

    # Duplicate reserve classes trimmed; canonical ones present.
    assert not (class_names & TRIMMED_CLASSES)
    assert CANONICAL_RESERVE_CLASSES <= class_names
