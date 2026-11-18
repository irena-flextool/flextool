"""Part 7 of v82: the behaviour-change markers of the storage start and end accounting.

``rp_storage_chain_pinned`` names the storage nodes with a ``*_blended_weights`` storage_binding_method when some
timeset has representative_period_weights; ``storage_end_matches_start`` names the storage nodes that chain forward
(bind_forward_only, bind_forward_only_blended_weights, or no storage_binding_method: the default).  On a hand-built
database:

* ``ws_b``: storage, bind_within_solve_blended_weights → the rp marker;
* ``fo_b``: storage, bind_forward_only_blended_weights → both markers;
* ``fwd``: storage, bind_forward_only → the end marker;
* ``dflt``: storage without a storage_binding_method → the end marker;
* ``cyc``: storage, bind_within_solve → neither;
* ``bal``: a balance node with bind_forward_only_blended_weights → neither.
"""
from __future__ import annotations

import json

from spinedb_api import DatabaseMapping, Map, to_database

from flextool.update_flextool import db_migration as m


def _db(*, rp: bool = True) -> DatabaseMapping:
    db = DatabaseMapping("sqlite://", create=True)
    for cls in ("node", "timeset"):
        db.add_item("entity_class", name=cls)
    db.add_item("alternative", name="base")
    for name in ("node_type", "storage_binding_method"):
        db.add_item("parameter_definition", entity_class_name="node", name=name)
    db.add_item("parameter_definition", entity_class_name="timeset", name="representative_period_weights")
    values = {
        ("node", "ws_b"): {"node_type": "storage", "storage_binding_method": "bind_within_solve_blended_weights"},
        ("node", "fo_b"): {"node_type": "storage", "storage_binding_method": "bind_forward_only_blended_weights"},
        ("node", "fwd"): {"node_type": "storage", "storage_binding_method": "bind_forward_only"},
        ("node", "dflt"): {"node_type": "storage"},
        ("node", "cyc"): {"node_type": "storage", "storage_binding_method": "bind_within_solve"},
        ("node", "bal"): {"node_type": "balance", "storage_binding_method": "bind_forward_only_blended_weights"},
        ("timeset", "ts"): ({"representative_period_weights": Map(["t1"], [Map(["t1"], [1.0])])} if rp else {}),
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


def test_markers_name_the_blended_and_forward_storage_nodes() -> None:
    with _db() as db:
        m._v82_storage_accounting_markers(db)
        assert _marker(db, "rp_storage_chain_pinned") == {"since": 82, "nodes": ["fo_b", "ws_b"]}
        assert _marker(db, "storage_end_matches_start") == {"since": 82, "nodes": ["dflt", "fo_b", "fwd"]}
        # Idempotent: a rerun writes the same markers.
        m._v82_storage_accounting_markers(db)
        assert _marker(db, "rp_storage_chain_pinned") == {"since": 82, "nodes": ["fo_b", "ws_b"]}


def test_no_rp_marker_without_representative_periods() -> None:
    with _db(rp=False) as db:
        m._v82_storage_accounting_markers(db)
        assert _marker(db, "rp_storage_chain_pinned") is None
        assert _marker(db, "storage_end_matches_start") == {"since": 82, "nodes": ["dflt", "fo_b", "fwd"]}
