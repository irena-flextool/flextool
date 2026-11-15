"""Database version 80: the description of each period-map limit says that ``inf`` (``-inf`` for a lower limit) is
no limit in that period.

A v79 database (built from a v75 seed JSON and migrated up to 79, never a checked-in ``.sqlite``) is migrated to 80:
each limit's description gets its sentence once (a rerun adds nothing), the new text is the shipped schema's, and
no parameter value changes.
"""
from __future__ import annotations

import json
from pathlib import Path

from spinedb_api import DatabaseMapping

from flextool._resources import package_data_path
from flextool.update_flextool import db_migration as m
from flextool.update_flextool.db_migration import migrate_database
from tests.db_utils import json_to_db

SEED = Path(__file__).resolve().parent / "data" / "v75_seed_stoch_two_period.json"


def _descriptions(url: str) -> dict:
    with DatabaseMapping(url) as db:
        return {(p["entity_class_name"], p["name"]): p["description"] or ""
                for p in db.get_items("parameter_definition")}


def _values(url: str) -> list:
    with DatabaseMapping(url) as db:
        return sorted((v["entity_class_name"], v["entity_byname"], v["parameter_definition_name"],
                       v["alternative_name"], bytes(v["value"])) for v in db.get_items("parameter_value"))


def test_v80_appends_the_inf_sentence_once(tmp_path_factory) -> None:
    url = json_to_db(SEED, tmp_path_factory.mktemp("v80") / "seed.sqlite")
    migrate_database(url, up_to=79)
    before, values_before = _descriptions(url), _values(url)
    migrate_database(url, up_to=80)
    after = _descriptions(url)
    with open(package_data_path("schemas/spinedb_schema.json"), encoding="utf-8") as f:
        shipped = {(r[0], r[1]): r[4] for r in json.load(f)["parameter_definitions"]}
    suffixes = m._v80_suffixes()
    assert ("group", "co2_max_period") in suffixes and ("constraint", "constant") in suffixes
    for key, suffix in suffixes.items():
        if key not in before:
            continue
        assert after[key] == before[key].rstrip() + suffix
        assert after[key] == shipped[key]
    assert {k: v for k, v in after.items() if k not in suffixes} == {
        k: v for k, v in before.items() if k not in suffixes}
    assert _values(url) == values_before
    with DatabaseMapping(url) as db:
        m._migrate_v80_limit_inf_descriptions(db)
    assert _descriptions(url) == after
