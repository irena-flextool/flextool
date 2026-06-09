"""Unit test for the v58 migration carving ``flowGroup`` out of ``group``.

A 57->58 migration test must construct its pre-migration (v57) state
INDEPENDENT of the current schema JSON (which now ships v58 after the C1
``sync_master_json_template`` regen) and INDEPENDENT of ``tests/fixtures/*.json``
(still v57 today, but they will be re-migrated to v58 in a later phase).  So
neither the schema JSON nor the test fixtures can be the v57 source.

Instead this test builds the v57 base by EXPLICITLY constructing — via the
SpineDB ``DatabaseMapping`` API on a fresh ``create=True`` DB — exactly the
old-vocab structures the v58 migration reads/moves (the ``group`` flow defs +
3-dim flow classes, the output flags, the §8/§9 heal targets, the ``model.version``
definition stamped to 57.0), then seeds four ``group`` entities that exercise
every branch of the per-entity migration rule (spec
``nodegroup_flowgroup_db_split.md`` §4.1), migrates to v58 and asserts the
structural carve-out, value/EA preservation, bool->enum mapping, renames and
the §8/§9 metadata heal.

This keeps the test robust to the committed schema's version (it never reads
``flextool/schemas/spinedb_schema.json``) and to the fixtures' version.
"""

import pytest
from spinedb_api import DatabaseMapping, from_database, to_database

from flextool.update_flextool import migrate_database


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


def _build_v57_base(url):
    """Construct exactly the old-vocab (v57) structures the v58 migration
    reads/moves, on a fresh ``create=True`` DB, and stamp ``model.version=57``.

    This is the SINGLE SOURCE OF the test's v57 state — deliberately NOT the
    schema JSON (now v58) nor the test fixtures (still v57 but soon re-migrated).
    """
    no_v, no_t = to_database("no")
    fiftyseven_v, fiftyseven_t = to_database(57.0)
    with DatabaseMapping(url, create=True) as db:
        # the seed (and the migration's EA copy) keys off a lowercase ``base``
        # alternative; a fresh create=True DB ships only the default ``Base``.
        db.add_alternative(name="base")

        # --- entity classes the migration touches ---
        db.add_entity_class(name="model")
        db.add_entity_class(name="node")
        db.add_entity_class(name="unit")
        db.add_entity_class(name="connection")
        db.add_entity_class(name="group")
        db.add_entity_class(name="group__node", dimension_name_list=("group", "node"))
        db.add_entity_class(name="group__unit__node", dimension_name_list=("group", "unit", "node"))
        db.add_entity_class(name="group__connection__node", dimension_name_list=("group", "connection", "node"))

        # --- parameter_groups the migration tags flowGroup defs with
        #     (Step 2: parameter_group_name="flow_limit" / "output").  These
        #     are name-keyed (not entity-class-scoped) and assumed pre-existing.
        db.add_parameter_group(name="flow_limit", color="fccde5", priority=55)
        db.add_parameter_group(name="output", color="ffffb3", priority=2)

        # --- value list yes_no (members "yes","no") ---
        db.add_parameter_value_list(name="yes_no")
        for idx, m in enumerate(("yes", "no")):
            mv, mt = to_database(m)
            db.add_list_value(parameter_value_list_name="yes_no", index=idx, value=mv, type=mt)

        # --- model.version definition (default 57.0): this is what
        #     migrate_database keys off via object_parameter_definition_sq.
        db.add_parameter_definition(
            entity_class_name="model", name="version",
            default_value=fiftyseven_v, default_type=fiftyseven_t,
            description="Contains database version information.",
        )

        # --- group flow defs the migration reads + drops (Step 4) ---
        for name in ("max_instant_flow", "min_instant_flow", "max_cumulative_flow", "min_cumulative_flow"):
            db.add_parameter_definition(
                entity_class_name="group", name=name,
                parameter_type_list=("float", "1d_map"),
                description=f"[MW] {name} for the aggregated flow of all group members.",
            )
        # bool flow flags bound to yes_no (re-mapped to flow_aggregator enum)
        db.add_parameter_definition(
            entity_class_name="group", name="flow_aggregator",
            default_value=no_v, default_type=no_t,
            parameter_value_list_name="yes_no",
            description="Aggregates the flows of the members for dispatch plots.",
        )
        db.add_parameter_definition(
            entity_class_name="group", name="output_flowGroup_indicators",
            default_value=no_v, default_type=no_t,
            parameter_value_list_name="yes_no",
            description="Standalone aggregated-flow indicators for this flow group.",
        )

        # --- output flags the migration renames (Step 5) ---
        db.add_parameter_definition(
            entity_class_name="group", name="output_nodeGroup_dispatch",
            default_value=no_v, default_type=no_t,
            parameter_value_list_name="yes_no",
            description="Creates the timewise dispatch output for this node group.",
        )
        db.add_parameter_definition(
            entity_class_name="group", name="output_nodeGroup_indicators",
            default_value=no_v, default_type=no_t,
            parameter_value_list_name="yes_no",
            description="Creates the indicator output for this node group.",
        )

        # --- a non-flow param (dual-role branch): inertia_limit ---
        db.add_parameter_definition(
            entity_class_name="group", name="inertia_limit",
            description="[MWs] Minimum inertia for the group.",
        )

        # --- §8 heal target: cumulative_max_capacity WITHOUT a type_list
        #     (the migration adds ("float","1d_map")).
        db.add_parameter_definition(
            entity_class_name="group", name="cumulative_max_capacity",
            description="[MWh] Maximum cumulative capacity.",
        )

        # --- §9 heal target: node.invest_forced WITHOUT a description
        #     (the migration adds the canonical description).
        db.add_parameter_definition(
            entity_class_name="node", name="invest_forced",
            parameter_type_list=("float", "1d_map"),
        )

        db.commit_session("build v57 base for v58 carve-out test")


@pytest.fixture
def migrated_url(tmp_path_factory):
    """A v57 DB (explicitly constructed, schema-version-independent) seeded with
    the four branch-exercising groups, then migrated to v58.  Returns the URL."""
    db_path = tmp_path_factory.mktemp("v58") / "v58_carveout.sqlite"
    url = "sqlite:///" + str(db_path)
    _build_v57_base(url)

    assert _db_version(url) == 57, "explicitly built base DB must read as v57 before migration"

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
