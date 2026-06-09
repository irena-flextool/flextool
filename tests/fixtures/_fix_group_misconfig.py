"""One-off: remove misconfigured group output flags from tests.json.

Removes four data items from ``tests/fixtures/tests.json`` (schema v57):

1. group entity ``anti_energy_group`` (cascades its two parameter_values
   ``flow_aggregator`` + ``output_nodeGroup_dispatch`` in alt
   ``aggregate_outputs``; it has no relationship memberships).
2. ``group.to_west_node.output_nodeGroup_dispatch`` in alt ``aggregate_outputs``.
3. ``group.electricity.flow_aggregator`` in alt ``init``.
4. ``group.east_group.flow_aggregator`` in alt ``aggregate_outputs``.

These are output-shaping flags only (no LP effect); the sole golden that
changes is
``tests/expected/aggregate_outputs_network_coal_wind_chp/group_flows__dt.csv``.

Round-trips through the same json_to_db / db_to_json path used by
``flextool.update_flextool.test_fixtures`` so the output stays byte-stable
under ``test_fixtures verify``.

Run::

    ~/venv-spi/bin/python tests/fixtures/_fix_group_misconfig.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spinedb_api import DatabaseMapping  # noqa: E402

from tests.db_utils import db_to_json, json_to_db  # noqa: E402

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "tests.json"

# (entity_class, entity_name, parameter, alternative) for explicit PV removals.
PV_REMOVALS = [
    ("group", "to_west_node", "output_nodeGroup_dispatch", "aggregate_outputs"),
    ("group", "electricity",  "flow_aggregator",           "init"),
    ("group", "east_group",   "flow_aggregator",           "aggregate_outputs"),
]


def _apply(url: str) -> None:
    # NOTE: no create=True — json_to_db already created + committed the DB.
    with DatabaseMapping(url) as db:
        # 1. Entity removal. Cascades the entity's own parameter_values
        #    (anti_energy_group/flow_aggregator + /output_nodeGroup_dispatch).
        ents = db.find_entities(entity_class_name="group", name="anti_energy_group")
        if not ents:
            raise RuntimeError("anti_energy_group not found — fixture changed?")
        for e in ents:
            items, errors = db.remove_items("entity", e["id"], strict=True)
            if errors:
                raise RuntimeError(f"entity removal errors: {errors}")

        # 2-4. Explicit single parameter_value removals.
        for cls, name, param, alt in PV_REMOVALS:
            pvs = db.find_parameter_values(
                entity_class_name=cls,
                entity_byname=(name,),
                parameter_definition_name=param,
                alternative_name=alt,
            )
            if len(pvs) != 1:
                raise RuntimeError(
                    f"expected 1 PV for {name}/{param}/{alt}, found {len(pvs)}"
                )
            _, errors = db.remove_items("parameter_value", pvs[0]["id"], strict=True)
            if errors:
                raise RuntimeError(f"PV removal errors ({name}/{param}): {errors}")

        db.commit_session("Remove misconfigured group output flags")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        sqlite_path = Path(tmp) / "staging.sqlite"
        url = json_to_db(FIXTURE, sqlite_path)   # JSON -> DB (commits import)
        _apply(url)                              # 4 removals + commit
        db_to_json(sqlite_path, FIXTURE)         # DB -> JSON (base64-packed)
    print(f"rewrote {FIXTURE}")


if __name__ == "__main__":
    main()
