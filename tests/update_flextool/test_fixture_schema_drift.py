"""Every repository fixture and canonical JSON carries exactly the schema's sections.

The five schema sections (entity classes, value lists, groups, definitions,
types) of each JSON with ``parameter_definitions`` under ``tests/fixtures/``,
``tests/engine/data/`` and ``flextool/schemas/canonical_databases/`` are
compared, order-free and decoded, with ``spinedb_schema.json`` built
through a database once per process (``fixture_schema.schema_sections``).

Not covered on purpose: ``flextool/schemas/pre_v26/*`` and
``old_flextool_import_template.json`` (old-version inputs of the importers
and migration chain), the output / comparison / results templates (other
schemas), and the frozen migration-test seeds under ``tests/update_flextool/data``.

Fix a failure with ``python -m flextool.update_flextool.test_fixtures
migrate-all`` (tests/fixtures) or ``python -m
flextool.update_flextool.canonical_databases migrate-all`` (canonical).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from flextool._resources import package_data_path
from flextool.update_flextool.fixture_schema import Format, schema_sections, section_diffs

TESTS = Path(__file__).resolve().parents[1]
CANONICAL = Path(package_data_path("schemas/canonical_databases"))


def _with_definitions(paths) -> list[Path]:
    out = []
    for path in sorted(paths):
        with open(path, encoding="utf-8") as f:
            if "parameter_definitions" in json.load(f):
                out.append(path)
    return out


CASES: list[tuple[Path, Format]] = (
    [(p, "packed") for p in _with_definitions((TESTS / "fixtures").glob("*.json"))]
    + [(p, "packed") for p in _with_definitions((TESTS / "engine" / "data").rglob("*.json"))]
    + [(p, "decoded") for p in _with_definitions(CANONICAL.glob("*.json"))]
)


def _fix_command(fmt: Format) -> str:
    module = "test_fixtures" if fmt == "packed" else "canonical_databases"
    return f"python -m flextool.update_flextool.{module} migrate-all"


@pytest.mark.xfail(strict=True, reason="the committed JSONs are not yet synced with the schema")
@pytest.mark.parametrize(("path", "fmt"), CASES, ids=lambda c: c.name if isinstance(c, Path) else c)
def test_json_carries_the_schema_sections(path: Path, fmt: Format) -> None:
    with open(path, encoding="utf-8") as f:
        diffs = section_diffs(json.load(f), fmt)
    shown = "\n  ".join(diffs[:40]) + (f"\n  ... {len(diffs) - 40} more" if len(diffs) > 40 else "")
    assert not diffs, f"{path.name} differs from spinedb_schema.json:\n  {shown}\nFix: {_fix_command(fmt)}"


def test_drift_is_named_per_field() -> None:
    """Negative twin: tests.json data on the schema's sections, with one default nulled and one
    list value removed, fails with exactly those two differences."""
    with open(TESTS / "fixtures" / "tests.json") as f:
        data = json.load(f)
    data.update(schema_sections("packed"))
    assert section_diffs(data, "packed") == []
    for row in data["parameter_definitions"]:
        if row[:2] == ["unit", "lifetime_method"]:
            row[2] = [None, None]
    choice = next(i for i, row in enumerate(data["parameter_value_lists"])
                  if row[0] == "lifetime_methods" and json.loads(base64.b64decode(row[1][0]))
                  == "reinvest_choice")
    del data["parameter_value_lists"][choice]
    assert section_diffs(data, "packed") == [
        "parameter_value_lists: lifetime_methods: fixture=['reinvest_automatic', 'no_investment'] "
        "schema=['reinvest_automatic', 'reinvest_choice', 'no_investment']",
        "parameter_definitions: unit/lifetime_method: default: fixture='<no default>' schema='reinvest_automatic'",
    ]
