"""Part 6 of v82: the behaviour-change markers of the storage carry in rolling and nested solves.

``rolling_cyclic_storage_anchored`` names the storage nodes with ``bind_within_solve`` /
``bind_within_solve_blended_weights`` when some solve is a ``rolling_window`` solve; ``nested_storage_start_borrowed``
names the storage nodes with a ``storage_nested_fix_method`` when some solve contains solves.  On a hand-built
database:

* ``cyc``: storage, bind_within_solve, fix_quantity → every marker;
* ``blend``: storage, bind_within_solve_blended_weights → the rolling marker;
* ``wp``: storage, bind_within_period, fix_usage → every marker;
* ``blocks``: storage, bind_intraperiod_blocks → the rolling marker;
* ``fwd``: storage, bind_forward_only, fix_price → the borrowed-start marker only;
* ``bal``: a balance node with bind_within_solve → none.

``nested_storage_fix_cumulative`` names the storage nodes with ``fix_usage`` or ``fix_quantity`` when some solve
contains solves.
"""
from __future__ import annotations

import json

from spinedb_api import Array, DatabaseMapping, to_database

from flextool.update_flextool import db_migration as m
from flextool.update_flextool.db_migration import v82


def _db(*, rolling: bool = True, nested: bool = True) -> DatabaseMapping:
    db = DatabaseMapping("sqlite://", create=True)
    for cls in ("node", "solve"):
        db.add_item("entity_class", name=cls)
    db.add_item("alternative", name="base")
    for name in ("node_type", "storage_binding_method", "storage_nested_fix_method"):
        db.add_item("parameter_definition", entity_class_name="node", name=name)
    for name in ("solve_mode", "contains_solves"):
        db.add_item("parameter_definition", entity_class_name="solve", name=name)
    values = {
        ("node", "cyc"): {"node_type": "storage", "storage_binding_method": "bind_within_solve",
                          "storage_nested_fix_method": "fix_quantity"},
        ("node", "blend"): {"node_type": "storage", "storage_binding_method": "bind_within_solve_blended_weights"},
        ("node", "wp"): {"node_type": "storage", "storage_binding_method": "bind_within_period",
                         "storage_nested_fix_method": "fix_usage"},
        ("node", "blocks"): {"node_type": "storage", "storage_binding_method": "bind_intraperiod_blocks"},
        ("node", "fwd"): {"node_type": "storage", "storage_binding_method": "bind_forward_only",
                          "storage_nested_fix_method": "fix_price"},
        ("node", "bal"): {"node_type": "balance", "storage_binding_method": "bind_within_solve"},
        ("solve", "up"): {"solve_mode": "single_solve"} | ({"contains_solves": Array(["st"])} if nested else {}),
        ("solve", "st"): {"solve_mode": "rolling_window" if rolling else "single_solve"},
    }
    for (cls, entity), params in values.items():
        db.add_item("entity", entity_class_name=cls, name=entity)
        for name, value in params.items():
            raw, typ = to_database(value)
            db.add_item("parameter_value", entity_class_name=cls, entity_byname=(entity,),
                        parameter_definition_name=name, alternative_name="base", value=raw, type=typ)
    return db


def _marker(db, change: str) -> dict | None:
    for md in db.get_items("metadata"):
        if md["name"] == m.BEHAVIOUR_CHANGE_MARKER_PREFIX + change:
            return json.loads(md["value"])
    return None


def test_markers_name_the_rolling_and_nested_storage_nodes() -> None:
    with _db() as db:
        v82._v82_rolling_storage_markers(db)
        assert _marker(db, "rolling_cyclic_storage_anchored") == {"since": 82,
                                                                  "nodes": ["blend", "blocks", "cyc", "wp"]}
        assert _marker(db, "nested_storage_start_borrowed") == {"since": 82, "nodes": ["cyc", "fwd", "wp"]}
        assert _marker(db, "nested_storage_fix_cumulative") == {"since": 82, "nodes": ["cyc", "wp"]}
        # Idempotent: a rerun writes the same markers.
        v82._v82_rolling_storage_markers(db)
        assert _marker(db, "rolling_cyclic_storage_anchored") == {"since": 82,
                                                                  "nodes": ["blend", "blocks", "cyc", "wp"]}


def test_no_marker_without_rolling_or_nested_solves() -> None:
    with _db(rolling=False, nested=False) as db:
        v82._v82_rolling_storage_markers(db)
        assert _marker(db, "rolling_cyclic_storage_anchored") is None
        assert _marker(db, "nested_storage_start_borrowed") is None
        assert _marker(db, "nested_storage_fix_cumulative") is None


def test_reference_start_marker_names_reference_nodes_without_fix_start() -> None:
    """Part 7 (Phase 5): ``storage_reference_start`` names the storage nodes with use_reference_value and no
    fix_start (fix_end turns the reference off): they now start from the reference value instead of a free
    level."""
    db = DatabaseMapping("sqlite://", create=True)
    db.add_item("entity_class", name="node")
    db.add_item("alternative", name="base")
    for name in ("node_type", "storage_solve_horizon_method", "storage_start_end_method"):
        db.add_item("parameter_definition", entity_class_name="node", name=name)
    values = {"ref": {"node_type": "storage", "storage_solve_horizon_method": "use_reference_value"},
              "ref_end": {"node_type": "storage", "storage_solve_horizon_method": "use_reference_value",
                          "storage_start_end_method": "fix_end"},
              "ref_fixed": {"node_type": "storage", "storage_solve_horizon_method": "use_reference_value",
                            "storage_start_end_method": "fix_start"},
              "free": {"node_type": "storage", "storage_solve_horizon_method": "free"},
              "bal": {"node_type": "balance", "storage_solve_horizon_method": "use_reference_value"}}
    with db:
        for entity, params in values.items():
            db.add_item("entity", entity_class_name="node", name=entity)
            for name, value in params.items():
                raw, typ = to_database(value)
                db.add_item("parameter_value", entity_class_name="node", entity_byname=(entity,),
                            parameter_definition_name=name, alternative_name="base", value=raw, type=typ)
        v82._v82_storage_accounting_markers(db)
        assert _marker(db, "storage_reference_start") == {"since": 82, "nodes": ["ref"]}
