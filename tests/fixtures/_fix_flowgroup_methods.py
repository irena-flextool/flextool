"""One-off: correct ``flowGroup.flow_aggregator`` roles in tests.json (v58).

Background (spec ``nodegroup_flowgroup_db_split.md`` §3.4/§3.5): the generic
v58 bool->enum migration maps the two old booleans
``(flow_aggregator, output_flowGroup_indicators)`` per-alternative, which for
tests.json leaves the coal arcs sitting in TWO dispatch-bound aggregators
(``to_west_node`` and ``coal_electricity`` both ``dispatch_plots_only``).  That
double-counts the coal arc in the nodeGroup dispatch BALANCE and fires the C2
overlap warning (``validate_group_output_memberships``).

Generic-migration result (the before-state this script corrects):

* ``coal_electricity``        -> ``dispatch_plots_only``  @ alt ``aggregate_outputs``
* ``connections_with_east``   -> ``both``                 @ alt ``aggregate_outputs``
* ``to_west_node``            -> ``dispatch_plots_only``  @ alt ``aggregate_outputs``
                              -> ``standalone_aggregator_only`` @ alt ``init``

Intended roles (§3.5) so the coal arcs sit in exactly ONE dispatch-bound
aggregator (no double-count, no C2 warning):

* ``to_west_node``      -> ``standalone_aggregator_only`` (single value;
  remove the spurious per-alternative ``dispatch_plots_only`` override in
  ``aggregate_outputs`` so the ``init`` ``standalone_aggregator_only`` value
  governs all scenarios -- ``to_west_node`` is NOT dispatch-bound anywhere).
* ``coal_electricity``  -> ``both``  (was ``dispatch_plots_only``).
* ``connections_with_east`` -> ``both``  (unchanged).

tests.json is already at schema v58, so NO migration runs here -- this is a
pure data correction.  Round-trips through the same json_to_db / db_to_json
path used by ``flextool.update_flextool.test_fixtures`` so the output stays
byte-stable under ``test_fixtures verify``.

Run::

    ~/venv-spi/bin/python tests/fixtures/_fix_flowgroup_methods.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spinedb_api import DatabaseMapping, to_database  # noqa: E402

from tests.db_utils import db_to_json, json_to_db  # noqa: E402

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "tests.json"

# Single parameter_value to REMOVE: the spurious per-alternative dispatch
# override on ``to_west_node`` (collapses to the ``init`` standalone value).
PV_REMOVAL = ("flowGroup", "to_west_node", "flow_aggregator", "aggregate_outputs")

# Single parameter_value to OVERWRITE in place: coal_electricity -> both.
PV_UPDATE = (
    ("flowGroup", "coal_electricity", "flow_aggregator", "aggregate_outputs", "both"),
)


def _apply(url: str) -> None:
    # NOTE: no create=True — json_to_db already created + committed the DB.
    with DatabaseMapping(url) as db:
        # 1. Remove the spurious dispatch override on to_west_node.
        cls, name, param, alt = PV_REMOVAL
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
        if pvs[0]["parsed_value"] != "dispatch_plots_only":
            raise RuntimeError(
                f"{name}/{param}/{alt} expected dispatch_plots_only, "
                f"found {pvs[0]['parsed_value']!r}"
            )
        _, errors = db.remove_items("parameter_value", pvs[0]["id"], strict=True)
        if errors:
            raise RuntimeError(f"PV removal errors ({name}/{param}): {errors}")

        # Sanity: the surviving to_west_node value must be standalone_only.
        survivors = db.find_parameter_values(
            entity_class_name=cls,
            entity_byname=(name,),
            parameter_definition_name=param,
        )
        if len(survivors) != 1 or survivors[0]["parsed_value"] != (
            "standalone_aggregator_only"
        ):
            vals = [(s["alternative_name"], s["parsed_value"]) for s in survivors]
            raise RuntimeError(
                f"after removal {name} should be a single "
                f"standalone_aggregator_only value, found {vals}"
            )

        # 2. Overwrite coal_electricity -> both.
        for cls, name, param, alt, newval in PV_UPDATE:
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
            if pvs[0]["parsed_value"] != "dispatch_plots_only":
                raise RuntimeError(
                    f"{name}/{param}/{alt} expected dispatch_plots_only, "
                    f"found {pvs[0]['parsed_value']!r}"
                )
            value, vtype = to_database(newval)
            db.add_update_item(
                "parameter_value",
                entity_class_name=cls,
                entity_byname=(name,),
                parameter_definition_name=param,
                alternative_name=alt,
                value=value,
                type=vtype,
            )
            # Confirm the write landed with the intended enum value.
            after = db.find_parameter_values(
                entity_class_name=cls,
                entity_byname=(name,),
                parameter_definition_name=param,
                alternative_name=alt,
            )
            if len(after) != 1 or after[0]["parsed_value"] != newval:
                raise RuntimeError(
                    f"PV update failed for {name}/{param}/{alt}: "
                    f"{[(a['alternative_name'], a['parsed_value']) for a in after]}"
                )

        db.commit_session("Correct flowGroup flow_aggregator roles (§3.5)")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        sqlite_path = Path(tmp) / "staging.sqlite"
        url = json_to_db(FIXTURE, sqlite_path)   # JSON -> DB (commits import)
        _apply(url)                              # correction + commit
        db_to_json(sqlite_path, FIXTURE)         # DB -> JSON (base64-packed)
    print(f"rewrote {FIXTURE}")


if __name__ == "__main__":
    main()
