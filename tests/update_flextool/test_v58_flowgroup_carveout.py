"""Unit test for the v58 migration carving ``flowGroup`` out of ``group``.

Builds a v57 database from the hand-maintained schema JSON (NEVER a
checked-in ``.sqlite`` — see CLAUDE.md invariant #3 / architecture
"JSON-fixture single source of truth"), seeds four ``group`` entities that
exercise every branch of the per-entity migration rule (spec
``nodegroup_flowgroup_db_split.md`` §4.1), migrates to v58 and asserts the
structural carve-out, value/EA preservation, bool->enum mapping, renames and
the §8/§9 metadata heal.
"""

from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping, from_database, to_database

from flextool.update_flextool import initialize_database, migrate_database

SCHEMA = Path(__file__).resolve().parents[2] / "flextool" / "schemas" / "spinedb_schema.json"


def _db_version(url):
    """Read the ``model.version`` parameter-DEFINITION default (the value
    ``migrate_database`` itself keys off — proposal Finding 1)."""
    with DatabaseMapping(url) as db:
        sq = db.object_parameter_definition_sq
        row = (
            db.query(sq)
            .filter(sq.c.object_class_name == "model")
            .filter(sq.c.parameter_name == "version")
            .one_or_none()
        )
        assert row is not None, "model.version parameter definition missing"
        return int(from_database(row.default_value, row.default_type))


def _vals(db, entity_class_name, entity_byname, parameter_definition_name, **extra):
    return [
        from_database(v["value"], v["type"])
        for v in db.find_parameter_values(
            entity_class_name=entity_class_name,
            entity_byname=entity_byname,
            parameter_definition_name=parameter_definition_name,
            **extra,
        )
    ]


@pytest.fixture
def migrated_url(tmp_path_factory):
    """A v57 schema DB seeded with the four branch-exercising groups, then
    migrated to v58.  Returns the sqlite URL."""
    db_path = tmp_path_factory.mktemp("v58") / "v58_carveout.sqlite"
    initialize_database(str(SCHEMA), str(db_path))
    url = "sqlite:///" + str(db_path)

    assert _db_version(url) == 57, "fresh schema DB must read as v57 before migration"

    yes_v, yes_t = to_database("yes")
    with DatabaseMapping(url) as db:
        # member entities for the 3-dim / node memberships
        for n in ("u1", "u2"):
            db.add_entity(entity_class_name="unit", name=n)
        for n in ("n1", "n2", "n3"):
            db.add_entity(entity_class_name="node", name=n)
        db.add_entity(entity_class_name="connection", name="c1")

        # pure_flow: 3-dim flow membership + flow_aggregator=yes (Base) +
        # a max_instant_flow value (Base) + an entity_alternative.
        db.add_entity(entity_class_name="group", name="pure_flow")
        db.add_entity(entity_class_name="group__unit__node", entity_byname=("pure_flow", "u1", "n1"))
        db.add_parameter_value(
            entity_class_name="group", entity_byname=("pure_flow",),
            parameter_definition_name="flow_aggregator", alternative_name="base",
            value=yes_v, type=yes_t,
        )
        mif_v, mif_t = to_database(100.0)
        db.add_parameter_value(
            entity_class_name="group", entity_byname=("pure_flow",),
            parameter_definition_name="max_instant_flow", alternative_name="base",
            value=mif_v, type=mif_t,
        )
        db.add_entity_alternative(
            entity_class_name="group", entity_byname=("pure_flow",),
            alternative_name="base", active=True,
        )

        # pure_node: group__node membership + output_nodeGroup_dispatch=yes.
        db.add_entity(entity_class_name="group", name="pure_node")
        db.add_entity(entity_class_name="group__node", entity_byname=("pure_node", "n1"))
        db.add_parameter_value(
            entity_class_name="group", entity_byname=("pure_node",),
            parameter_definition_name="output_nodeGroup_dispatch", alternative_name="base",
            value=yes_v, type=yes_t,
        )

        # dual: group__connection__node membership + group__node membership +
        # output_flowGroup_indicators=yes + max_cumulative_flow value + a
        # NON-flow param (inertia_limit).
        db.add_entity(entity_class_name="group", name="dual")
        db.add_entity(entity_class_name="group__connection__node", entity_byname=("dual", "c1", "n2"))
        db.add_entity(entity_class_name="group__node", entity_byname=("dual", "n3"))
        db.add_parameter_value(
            entity_class_name="group", entity_byname=("dual",),
            parameter_definition_name="output_flowGroup_indicators", alternative_name="base",
            value=yes_v, type=yes_t,
        )
        mcf_v, mcf_t = to_database(50.0)
        db.add_parameter_value(
            entity_class_name="group", entity_byname=("dual",),
            parameter_definition_name="max_cumulative_flow", alternative_name="base",
            value=mcf_v, type=mcf_t,
        )
        il_v, il_t = to_database(5.0)
        db.add_parameter_value(
            entity_class_name="group", entity_byname=("dual",),
            parameter_definition_name="inertia_limit", alternative_name="base",
            value=il_v, type=il_t,
        )

        # vacuous: group__node membership + a stray flow_aggregator=yes value
        # (no 3-dim flow membership).
        db.add_entity(entity_class_name="group", name="vacuous")
        db.add_entity(entity_class_name="group__node", entity_byname=("vacuous", "n1"))
        db.add_parameter_value(
            entity_class_name="group", entity_byname=("vacuous",),
            parameter_definition_name="flow_aggregator", alternative_name="base",
            value=yes_v, type=yes_t,
        )
        db.commit_session("seed v58 test groups")

    migrate_database(url, up_to=58)
    assert _db_version(url) == 58, "migration must bump version to 58"
    return url


def test_v58_carveout(migrated_url):
    url = migrated_url
    with DatabaseMapping(url) as db:
        # --- structural classes ---
        for cls in ("flowGroup", "flowGroup__unit__node", "flowGroup__connection__node"):
            assert db.find_entity_classes(name=cls), f"{cls} class missing"
        for cls in ("group__unit__node", "group__connection__node"):
            assert not db.find_entity_classes(name=cls), f"old {cls} class still present"

        # --- old flow defs gone from group (all six) ---
        for name in (
            "max_instant_flow", "min_instant_flow", "max_cumulative_flow",
            "min_cumulative_flow", "flow_aggregator", "output_flowGroup_indicators",
        ):
            assert not db.find_parameter_definitions(entity_class_name="group", name=name), (
                f"group.{name} should have been dropped"
            )

        # --- renamed output flags ---
        assert db.find_parameter_definitions(entity_class_name="group", name="print_dispatch")
        assert db.find_parameter_definitions(entity_class_name="group", name="print_indicators")
        assert not db.find_parameter_definitions(entity_class_name="group", name="output_nodeGroup_dispatch")
        assert not db.find_parameter_definitions(entity_class_name="group", name="output_nodeGroup_indicators")

        # --- flow_aggregator def on flowGroup: bound value-list + default none ---
        fa_defs = db.find_parameter_definitions(entity_class_name="flowGroup", name="flow_aggregator")
        assert fa_defs, "flowGroup.flow_aggregator definition missing"
        fa_def = fa_defs[0]
        assert fa_def["parameter_value_list_name"] == "flow_aggregator_methods"
        assert from_database(fa_def["default_value"], fa_def["default_type"]) == "none"
        assert db.find_parameter_value_lists(name="flow_aggregator_methods"), "value-list missing"

        # --- pure_flow: re-homed to flowGroup, group entity gone ---
        assert db.find_entities(entity_class_name="flowGroup", name="pure_flow")
        assert not db.find_entities(entity_class_name="group", name="pure_flow"), (
            "pure_flow group entity should be removed (flow-only)"
        )
        assert [e["entity_byname"] for e in db.find_entities(entity_class_name="flowGroup__unit__node")] == [
            ("pure_flow", "u1", "n1")
        ]
        assert _vals(db, "flowGroup", ("pure_flow",), "max_instant_flow") == [100.0]
        assert _vals(db, "flowGroup", ("pure_flow",), "flow_aggregator") == ["dispatch_plots_only"]
        assert [
            ea["alternative_name"]
            for ea in db.find_entity_alternatives(entity_class_name="flowGroup", entity_byname=("pure_flow",))
        ] == ["base"], "entity_alternative must be copied to flowGroup"

        # --- pure_node: stays group, value under print_dispatch ---
        assert db.find_entities(entity_class_name="group", name="pure_node")
        assert not db.find_entities(entity_class_name="flowGroup", name="pure_node")
        assert _vals(db, "group", ("pure_node",), "print_dispatch") == ["yes"]

        # --- dual: SPLIT ---
        assert db.find_entities(entity_class_name="group", name="dual"), "dual group entity must be kept"
        assert db.find_entities(entity_class_name="flowGroup", name="dual"), "dual flowGroup entity must exist"
        # group keeps node membership + inertia_limit; lost connection membership
        group_node_bynames = {e["entity_byname"] for e in db.find_entities(entity_class_name="group__node")}
        assert ("dual", "n3") in group_node_bynames, "dual must keep its group__node membership"
        assert _vals(db, "group", ("dual",), "inertia_limit") == [5.0]
        # flowGroup gets connection membership + max_cumulative_flow + enum
        assert [e["entity_byname"] for e in db.find_entities(entity_class_name="flowGroup__connection__node")] == [
            ("dual", "c1", "n2")
        ]
        assert _vals(db, "flowGroup", ("dual",), "max_cumulative_flow") == [50.0]
        assert _vals(db, "flowGroup", ("dual",), "flow_aggregator") == ["standalone_aggregator_only"]

        # --- vacuous: stays group, flow_aggregator value dropped (def gone) ---
        assert db.find_entities(entity_class_name="group", name="vacuous")
        assert not db.find_entities(entity_class_name="flowGroup", name="vacuous")
        # the def is gone (already asserted above); the entity remains harmlessly.

        # --- §8 metadata heal ---
        cmc = db.find_parameter_definitions(entity_class_name="group", name="cumulative_max_capacity")
        assert cmc, "group.cumulative_max_capacity missing"
        assert cmc[0]["parameter_type_list"], "cumulative_max_capacity should have a parameter_type_list"

        # --- §9 node.invest_forced description ---
        inv = db.find_parameter_definitions(entity_class_name="node", name="invest_forced")
        assert inv, "node.invest_forced missing"
        assert inv[0]["description"] == (
            "[MWh] Forces the model to invest exactly this amount of new storage capacity, "
            "overriding the investment optimisation (equivalent to setting invest_min and "
            "invest_max equal). Constant or period."
        )


def test_v58_rerun_is_noop(migrated_url):
    """A second migrate to v58 on an already-v58 DB must be a clean no-op."""
    url = migrated_url
    assert _db_version(url) == 58
    migrate_database(url, up_to=58)
    assert _db_version(url) == 58

    with DatabaseMapping(url) as db:
        # No duplication of entities / classes from the re-run.
        assert sorted(e["name"] for e in db.find_entities(entity_class_name="flowGroup")) == ["dual", "pure_flow"]
        assert sorted(e["name"] for e in db.find_entities(entity_class_name="group")) == ["dual", "pure_node", "vacuous"]
        assert len(db.find_entities(entity_class_name="flowGroup__unit__node")) == 1
        assert len(db.find_entities(entity_class_name="flowGroup__connection__node")) == 1
        # value-list created exactly once
        assert len(db.find_parameter_value_lists(name="flow_aggregator_methods")) == 1
