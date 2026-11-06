"""``test_fixtures`` / ``canonical_databases``: migrate, then sync the schema sections.

Runs on copies in a temporary directory; no committed file is written.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from flextool._resources import package_data_path
from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool.canonical_databases import migrate_canonical
from flextool.update_flextool.fixture_schema import fixture_version, normalise_data, section_diffs
from flextool.update_flextool.test_fixtures import fixture_paths, migrate_fixture, verify_all

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SEED = Path(__file__).resolve().parent / "data" / "v75_seed_stoch_two_period.json"
MARKER_PREFIX = "flextool_behaviour_change:"


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_every_fixture_with_definitions_is_covered() -> None:
    expected = sorted(p for p in FIXTURES.glob("*.json") if "parameter_definitions" in _load(p))
    assert fixture_paths() == expected
    assert {"null_override_clears_map.json", "stoch_two_period.json", "w4c_divergent_dispatch.json"} <= {
        p.name for p in expected}


def test_migration_test_seed_is_below_the_current_version() -> None:
    """The frozen seed of the version-77 migration tests is not brought to the current version."""
    assert fixture_version(_load(SEED), "packed") < 77
    assert SEED.parent != FIXTURES


def test_older_fixture_is_migrated_synced_and_carries_no_markers(tmp_path: Path) -> None:
    """A fixture below 77 goes through the shared marker filter: the v77 markers are dropped."""
    src = tmp_path / "fixtures" / "seed.json"
    src.parent.mkdir()
    shutil.copyfile(SEED, src)
    migrate_fixture(src)
    out = _load(src)
    assert fixture_version(out, "packed") == FLEXTOOL_DB_VERSION
    assert section_diffs(out, "packed") == []
    assert not [row for row in out.get("metadata", []) if str(row[0]).startswith(MARKER_PREFIX)]
    assert verify_all(src.parent) == 0, "a migrated fixture is stable through verify"


def test_verify_reports_a_drifted_fixture(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    shutil.copyfile(SEED, fixtures / "a.json")
    migrate_fixture(fixtures / "a.json")
    drifted = _load(fixtures / "a.json")
    for row in drifted["parameter_definitions"]:
        if row[:2] == ["unit", "lifetime_method"]:
            row[2] = [None, None]
    with open(fixtures / "b.json", "w") as f:
        json.dump(drifted, f, indent=2)
    assert verify_all(fixtures) == 1


def test_version_less_fixture_is_only_synced(tmp_path: Path) -> None:
    """Sync-only path: no migration runs; the result gains the version and is stable."""
    data = _load(FIXTURES / "null_override_clears_map.json")
    data["parameter_definitions"] = [r for r in data["parameter_definitions"] if r[:2] != ["model", "version"]]
    src = tmp_path / "null_override.json"
    with open(src, "w") as f:
        json.dump(data, f, indent=2)
    migrate_fixture(src)
    out = _load(src)
    assert fixture_version(out, "packed") == FLEXTOOL_DB_VERSION
    assert section_diffs(out, "packed") == []
    before, after = normalise_data(data, "packed"), normalise_data(out, "packed")
    for key in ("parameter_values", "scenarios", "scenario_alternatives"):
        assert after[key] == before[key], key
    # The hand-written entity rows omit the description an export writes.
    assert {row[:2] for row in after["entities"]} == {row[:2] for row in before["entities"]}
    first = src.read_text()
    migrate_fixture(src)
    assert src.read_text() == first


def test_canonical_database_is_migrated_and_synced(tmp_path: Path) -> None:
    src = tmp_path / "howto_connections.json"
    shutil.copyfile(package_data_path("schemas/canonical_databases/howto_connections.json"), src)
    original = _load(src)
    migrate_canonical(src)
    out = _load(src)
    assert section_diffs(out, "decoded") == []
    assert normalise_data(out, "decoded")["parameter_values"] == normalise_data(original, "decoded")[
        "parameter_values"]
    first = src.read_text()
    migrate_canonical(src)
    assert src.read_text() == first
