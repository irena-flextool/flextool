"""``social_discount_rate``, ``discount_offset_investment`` and ``discount_offset_operations`` end to end (data
version 82).

``tests.json`` scenario ``multi_year`` (a chain of solves that invests) with an override alternative that sets the
three away from their defaults: rate 0.03, offsets 1 and 0.25 (the rate differs from the investors' discount_rate
0.05, at which the offset of the investment payments would cancel out).  The same database written with the
version-81 names (``inflation_rate``, ``inflation_offset_investment``, ``inflation_offset_operations``) and stamped
81, migrated, has the schema and the values of the database written with the new names, and builds the same LP in
every solve.  Each setting alone changes the LP, so the equality is not that of settings the engine ignores.

Fixtures are built from JSON (``tests.json``).
"""
from __future__ import annotations

import json

import pytest

from flextool.update_flextool.db_migration.v82 import V82_PARAMETER_RENAMES
from tests.engine._unit_db import FLEXTOOL_ROOT, add_override_alternative, b64
from tests.engine.test_lifetime_shapes import _lp
from tests.engine.test_v81_names_end_to_end import _schema_and_values

TESTS_JSON = FLEXTOOL_ROOT / "tests" / "fixtures" / "tests.json"
SCENARIO = "multi_year"
SETTINGS = {"social_discount_rate": 0.03, "discount_offset_investment": 1.0, "discount_offset_operations": 0.25}
NEW_TO_OLD = {new: old for old, new in V82_PARAMETER_RENAMES.items()}


def _chain(data: dict, scenario: str) -> list[str]:
    nxt = {a: b for s, a, b in data["scenario_alternatives"] if s == scenario}
    alt = next(a for a in nxt if a not in set(nxt.values()))
    chain = []
    while alt is not None:
        chain.append(alt)
        alt = nxt.get(alt)
    return chain


def _with_settings(settings: dict[str, float], alt: str) -> tuple[dict, str]:
    """``tests.json`` with the override alternative *alt* setting *settings* on ``multi_year``'s chain."""
    data = json.loads(TESTS_JSON.read_text(encoding="utf-8"))
    values = [["model", "flexTool", name, b64(v, "float")] for name, v in settings.items()]
    return data, add_override_alternative(data, alt, values, _chain(data, SCENARIO))


def _as_version_81(data: dict) -> dict:
    """``data`` written with the version-81 names (definitions, types and values) and stamped 81."""
    old = json.loads(json.dumps(data))
    old["parameter_definitions"] = [[r[0], NEW_TO_OLD.get(r[1], r[1]) if r[0] == "model" else r[1], *r[2:]]
                                    for r in old["parameter_definitions"]]
    for r in old["parameter_definitions"]:
        if r[:2] == ["model", "version"]:
            r[2] = b64(81, "float")
    old["parameter_types"] = [[r[0], NEW_TO_OLD.get(r[1], r[1]) if r[0] == "model" else r[1], *r[2:]]
                              for r in old["parameter_types"]]
    old["parameter_values"] = [[r[0], r[1], NEW_TO_OLD.get(r[2], r[2]) if r[0] == "model" else r[2], *r[3:]]
                               for r in old["parameter_values"]]
    return old


def _db(tmp_path_factory, data: dict, tag: str) -> str:
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    work = tmp_path_factory.mktemp(tag)
    (work / "f.json").write_text(json.dumps(data), encoding="utf-8")
    url = json_to_db(work / "f.json", work / "f.sqlite")
    migrate_database(url)
    return url


def _lps(tmp_path_factory, url: str, scenario: str, tag: str) -> dict:
    from flextool.engine import run_chain_from_db

    steps = run_chain_from_db(url, scenario, work_folder=tmp_path_factory.mktemp(f"{tag}_w"), keep_solutions=True)
    assert steps
    return {name: _lp(step.flex_data) for name, step in steps.items()}


@pytest.fixture(scope="module")
def direct_lps(tmp_path_factory) -> dict:
    data, scenario = _with_settings(SETTINGS, "discount_settings")
    return _lps(tmp_path_factory, _db(tmp_path_factory, data, "v82_direct"), scenario, "v82_direct")


@pytest.mark.solver
def test_version_81_names_migrate_to_the_same_model(tmp_path_factory, direct_lps) -> None:
    """The settings written under the version-81 names, migrated: the same definitions, list members and values
    (each stored value under its new name) as the database written with the new names, and the same LP."""
    data, scenario = _with_settings(SETTINGS, "discount_settings")
    old = _as_version_81(data)
    assert {r[2] for r in old["parameter_values"] if r[0] == "model"} >= set(V82_PARAMETER_RENAMES)
    old_url = _db(tmp_path_factory, old, "v82_old")
    new_url = _db(tmp_path_factory, data, "v82_new")
    migrated, direct = _schema_and_values(old_url), _schema_and_values(new_url)
    assert not {k for k in migrated["defs"] if k[0] == "model" and k[1] in V82_PARAMETER_RENAMES}
    assert migrated == direct
    assert _lps(tmp_path_factory, old_url, scenario, "v82_old") == direct_lps


@pytest.mark.solver
@pytest.mark.parametrize("name", sorted(SETTINGS))
def test_each_setting_changes_the_model(tmp_path_factory, direct_lps, name) -> None:
    """Without one of the three settings (its default instead) the LP differs: the migrated equality above covers
    a setting the engine reads."""
    data, scenario = _with_settings({k: v for k, v in SETTINGS.items() if k != name}, f"without_{name}")
    assert _lps(tmp_path_factory, _db(tmp_path_factory, data, f"v82_{name}"), scenario, f"v82_{name}") != direct_lps
