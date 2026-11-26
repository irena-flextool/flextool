"""Data versions 26-43."""
import logging
import math

from spinedb_api import from_database, SpineDBAPIError, to_database

from flextool.update_flextool.db_migration.common import (
    _already_applied,
    _commit_step,
    _later_definition,
    _remove_items_bounded,
    _rename_or_drop_parameter_definition,
    _reused_later,
    _upsert_definition,
    add_value_list_manual,
    remove_parameters_manual,
)


def _migrate_v26(db) -> None:
    db.add_update_item("parameter_definition", entity_class_name="connection", name="reactance",
        parameter_type_list=("float",),
        description="[p.u.] Per-unit reactance of the transmission line. Used for DC power flow when the connection is within a nodeGroup that has transfer_method set to dc_power_flow_with_angles. The susceptance used in the flow equation is base_MVA / reactance.")
    db.add_update_item("parameter_definition", entity_class_name="group", name="transfer_method",
        parameter_type_list=("str",),
        description="Override transfer_method for all connections within this nodeGroup. Options: use_connection_transfer_methods (default, no override), no_losses_no_variable_cost, regular, exact, variable_cost_only, dc_power_flow_with_angles. When set to dc_power_flow_with_angles, connections between member nodes use B-theta DC power flow (requires reactance parameter on connections).")
    db.add_update_item("parameter_definition", entity_class_name="group", name="base_MVA",
        parameter_type_list=("float",),
        description="[MVA] Base power for the per-unit system used in DC power flow. Default 100. susceptance = base_MVA / reactance_pu.")
    db.add_update_item("parameter_definition", entity_class_name="group", name="candidate_precapacity_to_avoid_big_m",
        parameter_type_list=("float",),
        description="[MW] Small pre-existing capacity assigned to investment candidate connections in DC power flow groups that have zero existing capacity. This avoids the need for big-M / MIP constraints by ensuring the angle constraint is always active. Default 0.1 MW. The value should be small enough to not affect results but large enough for numerical stability.")
    db.add_update_item("parameter_definition", entity_class_name="group", name="reference_node",
        parameter_type_list=("str",),
        description="Name of the reference bus node (angle fixed to zero) for DC power flow. Optional — if not specified, automatically selected as the node with the largest existing capacity in each connected component of the DC power flow network.")
    _commit_step(db,"Added DC power flow parameters")


def _migrate_v27(db) -> None:
    add_value_list_manual(db, [["minimum_time_methods", "none"]])
    db.update_item("parameter_definition", entity_class_name="unit", name="minimum_time_method",
        description="Choice between minimum up- and downtimes (none, min_downtime, min_uptime, both). Setting this to anything other than 'none' will activate online variables (at least linear) for the unit. Default: none (no minimum time constraints).")
    db.update_item("parameter_definition", entity_class_name="unit", name="min_uptime",
        description="[hours] Minimum time the unit must stay online after starting up. Requires minimum_time_method set to 'min_uptime' or 'both'. Constant.")
    db.update_item("parameter_definition", entity_class_name="unit", name="min_downtime",
        description="[hours] Minimum time the unit must stay offline after shutting down. Requires minimum_time_method set to 'min_downtime' or 'both'. Constant.")
    # Fix penalty parameter descriptions (#308, #300)
    db.update_item("parameter_definition", entity_class_name="node", name="penalty_up",
        description="[CUR/MWh] Penalty cost for decreasing consumption in the node (energy not served). Constant, Period or Time.")
    db.update_item("parameter_definition", entity_class_name="node", name="penalty_down",
        description="[CUR/MWh] Penalty cost for increasing consumption in the node (excess energy). Constant, Period or Time.")
    db.update_item("parameter_definition", entity_class_name="group", name="penalty_capacity_margin",
        description="[CUR/kW] Penalty for violating the capacity margin constraint. Uses operational discounting (not annualized over lifetime like investment costs), so the value is not directly comparable to annualized investment costs. Constant or period.")
    db.update_item("parameter_definition", entity_class_name="group", name="penalty_inertia",
        description="[CUR/MWs] Penalty for violating the inertia constraint. Cost scales with the duration of the violation. Constant or period.")
    _commit_step(db,"Added minimum time method support and fixed penalty descriptions")


def _migrate_v28(db) -> None:
    # Rename entity-level interest_rate -> discount_rate on unit,
    # connection, node; and model-level discount_rate -> inflation_rate
    # plus the two discount_offset_* -> inflation_offset_* renames.
    #
    # These renames must be collision-safe.  Up to FlexTool 4.6.0
    # ``migrate_database`` wrote ``model.version`` only once, at the
    # very end of the whole loop, so a migration interrupted after a later
    # step already created the *target* definition (the v28 rename
    # itself, or the v48 ``add_update_item`` default-setter) but
    # before the final version write leaves the DB with the target
    # name present while ``model.version`` still reads its old,
    # pre-28 value.  A subsequent re-application of the v25 template
    # (re-init / GUI "apply template" / re-import) restores the
    # *source* name, so on the next migration this step sees BOTH.
    # A bare ``update_parameter_definition(name=target)`` would then
    # hit spinedb_api's unique-name constraint and abort the whole
    # migration ("there's already a parameter_definition ...").
    # When the target already exists we drop the stale source
    # definition instead of renaming onto it.
    entity_rate_desc = ("[e.g. 0.05 equals 5%] Discount rate for investments (WACC). "
        "Reflects the financing cost and risk premium for this technology. When the "
        "model inflation_rate > 0, this should be a nominal rate. When inflation_rate "
        "= 0, this should be a real rate. Used to annualize investment costs over the "
        "lifetime. Constant or period.")
    _rename_or_drop_parameter_definition(db, "unit", "interest_rate", "discount_rate", entity_rate_desc)
    _rename_or_drop_parameter_definition(db, "connection", "interest_rate", "discount_rate", entity_rate_desc)
    _rename_or_drop_parameter_definition(db, "node", "interest_rate", "discount_rate", entity_rate_desc)
    # Rename model-level discount_rate -> inflation_rate
    _rename_or_drop_parameter_definition(db, "model", "discount_rate", "inflation_rate",
        "[e.g. 0.02 for 2%] Model-wide inflation rate applied to all future costs. When "
        "inputs are in real (constant-price) terms, set to 0. When inputs are in nominal "
        "terms, set to expected inflation. Default: 0 (real inputs).")
    # Rename model-level offset parameters (not the v82 ones of the same names)
    for kind, desc in (
            ("investment", "[years] Offset for when investment costs occur within a year. Default 0 "
                           "(beginning of year)."),
            ("operations", "[years] Offset for when operational costs occur within a year. Default 0.5 "
                           "(middle of year).")):
        if not _reused_later(db, 28, "model", f"discount_offset_{kind}"):
            _rename_or_drop_parameter_definition(db, "model", f"discount_offset_{kind}",
                                                 f"inflation_offset_{kind}", desc)
    _commit_step(db,"Renamed economic parameters: interest_rate->discount_rate, discount_rate->inflation_rate")


def _migrate_v29(db) -> None:
    add_value_list_manual(db, [
        ["transfer_methods_group", "use_connection_transfer_methods"],
        ["transfer_methods_group", "no_losses_no_variable_cost"],
        ["transfer_methods_group", "regular"],
        ["transfer_methods_group", "exact"],
        ["transfer_methods_group", "variable_cost_only"],
        ["transfer_methods_group", "dc_power_flow_with_angles"],
    ])
    default_val, default_type = to_database("use_connection_transfer_methods")
    db.add_update_item("parameter_definition", entity_class_name="group", name="transfer_method",
        default_value=default_val, default_type=default_type,
        parameter_value_list_name="transfer_methods_group",
        description="Override transfer_method for all connections within this nodeGroup. Options: use_connection_transfer_methods (default, no override), no_losses_no_variable_cost, regular, exact, variable_cost_only, dc_power_flow_with_angles. When set to dc_power_flow_with_angles, connections between member nodes use B-theta DC power flow (requires reactance parameter on connections).")
    _commit_step(db,"Added transfer_methods_group parameter_value_list for group transfer_method")


def _migrate_v32(db) -> None:
    # Rename constraint_capacity_coefficient -> constraint_invested_capacity_coefficient
    # on unit, connection, node. The old expression in flextool.mod was buggy
    # (summed v_invest[e, d] once per active d_invest in edd_invest, giving
    # #active-investments * v_invest[e, d]); it is being fixed to emit exactly
    # v_invest[e, d] — current-period new build. In single-period models the
    # old and new outputs coincide; in multi-period models the old output was
    # incorrect.
    invested_desc = (
        "A map of coefficients (index: constraint name, value: coefficient) "
        "that places v_invest[e, d] — new-build capacity decided in the "
        "current period — on the left side of the user-defined constraint. "
        "Not multiplied by unitsize."
    )
    parameter_definitions = db.mapped_table("parameter_definition")
    for cls in ("unit", "connection", "node"):
        # Already renamed (the v32 name or its v56 one): a migration that failed at a later step (v35,
        # FlexTool 4.0 to 4.5) left v32 applied under the old stamp.  Neither name still stops at db.item.
        later = _later_definition(db, 31, cls, "constraint_capacity_coefficient")
        if later is not None:
            _already_applied(32, f"{cls}.constraint_capacity_coefficient is {later[1]}")
            continue
        param = db.item(parameter_definitions,
                        entity_class_name=cls,
                        name="constraint_capacity_coefficient")
        if param:
            db.update_parameter_definition(
                id=param["id"],
                name="constraint_invested_capacity_coefficient",
                description=invested_desc)
    # Add constraint_cumulative_pre_built_capacity_coefficient — cumulative
    # new-build capacity from all periods strictly before d, ignoring
    # retirements (learning-effect semantics).
    prebuilt_desc = (
        "A map of coefficients (index: constraint name, value: coefficient) "
        "that places the cumulative pre-built capacity at period d — data "
        "baseline plus every v_invest made in periods strictly BEFORE d, "
        "retirements ignored — on the left side of the user-defined "
        "constraint. Enables learning-effect and period-over-period growth "
        "limits. Not multiplied by unitsize."
    )
    for cls in ("unit", "connection", "node"):
        _upsert_definition(db, 32, cls, "constraint_cumulative_pre_built_capacity_coefficient",
                           description=prebuilt_desc)
    _commit_step(db,
        "Renamed constraint_capacity_coefficient -> "
        "constraint_invested_capacity_coefficient; added "
        "constraint_cumulative_pre_built_capacity_coefficient")


def _migrate_v33(db) -> None:
    db.add_update_item("parameter_definition",
        entity_class_name="timeset",
        name="timeset_weights",
        description=(
            "Per-timestep weight map (index: timestep name, value: "
            "float) applied to cost and slack terms in the objective. "
            "Use for non-RP models where timesteps represent unequal "
            "fractions of the year (e.g. seasonal yearsplit on a "
            "coarse timeslice structure). Weights are normalized per "
            "period to sum to 1 and then scaled by the number of "
            "active timesteps so that uniform input reproduces the "
            "default (weight = 1 per step). Must not be combined "
            "with representative_period_weights on the same timeset."))
    _commit_step(db,"Added timeset.timeset_weights parameter")


def _migrate_v35(db) -> None:
    # Split the old `coefficient` parameter on unit__inputNode /
    # unit__outputNode into three separate parameters:
    #   - flow_coefficient: energy-unit conversion for the flow
    #     in node-balance and conversion_indirect equations
    #     (renamed from 'coefficient', same semantics).
    #   - max_capacity_coefficient: scales the per-edge upper cap
    #     (maxFlow / maxToSource / maxFromSource / ramp-up).
    #     Default 1.0.
    #   - min_capacity_coefficient: scales the per-edge lower cap
    #     (minFlow_minload / minFromSource_minload / min-load
    #     terms / ramp-down). Default 1.0.
    # Use case that forced the split: extraction CHP with a
    # heat output whose flow_coefficient < 1 (small balance
    # scaler) but whose max capacity is still full nameplate.
    flow_desc = (
        "[factor] Energy-unit conversion factor for this flow "
        "in the node balance and conversion_indirect equations. "
        "Value of 0 removes the edge from capacity / ramp / min-"
        "load constraints entirely (hydro-pass-through pattern)."
    )
    maxcap_desc = (
        "[factor, default 1.0] Fraction of the unit's capacity "
        "available to this edge's upper cap (maxToSink / "
        "maxFromSource / ramp). For extraction CHP set to 1.0 "
        "on each output so each can reach full capacity when "
        "the other drops."
    )
    mincap_desc = (
        "[factor, default 1.0] Fraction of the unit's capacity "
        "imposed as a lower cap on this edge when online "
        "(combined multiplicatively with the unit-level "
        "min_load). Set to 0 to remove the lower cap on this "
        "edge (e.g. heat output of an extraction CHP that may "
        "drop to zero in pure-condensing mode)."
    )
    default_one_val, default_one_type = to_database(1.0)
    for cls in ("unit__inputNode", "unit__outputNode"):
        param = db.get_item("parameter_definition", entity_class_name=cls, name="coefficient")
        later = _later_definition(db, 34, cls, "coefficient")
        if later is not None:
            _already_applied(35, f"{cls}.coefficient is renamed to {later[1]}")
        elif param:
            _v35_drop_v23_flow_coefficient(db, cls)
            db.update_parameter_definition(
                id=param["id"],
                name="flow_coefficient",
                description=flow_desc)
        for name, desc in (("max_capacity_coefficient", maxcap_desc),
                           ("min_capacity_coefficient", mincap_desc)):
            _upsert_definition(db, 35, cls, name, default_value=default_one_val,
                               default_type=default_one_type, parameter_type_list=("float",),
                               description=desc)
    _commit_step(db,
        "Renamed coefficient -> flow_coefficient; added "
        "max_capacity_coefficient and min_capacity_coefficient "
        "on unit__inputNode and unit__outputNode")


def _migrate_v36(db) -> None:
    # Backfill v35: preserve the OLD coefficient behaviour for
    # existing databases where coefficient was set to a non-
    # default value. The old formulas were:
    #   sink (unit__outputNode): v_flow ≤ online × coef
    #     → new:  max_capacity_coefficient = coef
    #   source (unit__inputNode): v_flow × coef ≤ online
    #                           ⇔ v_flow ≤ online / coef
    #     → new:  max_capacity_coefficient = 1 / coef
    # For min-load the coefficient scaled both sides the same
    # way, so min_capacity_coefficient = coef on both classes
    # preserves behaviour.
    # Only entities whose flow_coefficient was *explicitly set*
    # are affected; those relying on the default 1.0 already
    # get max/min = 1.0 from the defaults introduced in v35.
    db.mapped_table("parameter_value")
    for cls in ("unit__outputNode", "unit__inputNode") if not _v38_ran(db, 36) else ():
        caps = _float_values(db, cls, ("max_capacity_coefficient", "min_capacity_coefficient"))
        existing = list(db.find_parameter_values(
            entity_class_name=cls,
            parameter_definition_name="flow_coefficient"))
        for pv in existing:
            try:
                coef = float(pv["parsed_value"])
            except (TypeError, ValueError):
                continue
            # Skip default-valued rows — backfill is a no-op.
            if coef == 1.0:
                continue
            if cls == "unit__outputNode" and coef != 0.0 and _v37_flipped(caps, pv, 1.0 / coef):
                _already_applied(36, f"{cls} {tuple(pv['entity_byname'])} flow_coefficient is flipped")
                continue
            if cls == "unit__outputNode":
                max_cap = coef
            else:
                max_cap = (1.0 / coef) if coef != 0 else 0.0
            min_cap = coef
            for pname, pval in (("max_capacity_coefficient", max_cap),
                                ("min_capacity_coefficient", min_cap)):
                value, vtype = to_database(pval)
                db.add_update_item(
                    "parameter_value",
                    entity_class_name=cls,
                    entity_byname=pv["entity_byname"],
                    parameter_definition_name=pname,
                    alternative_name=pv["alternative_name"],
                    value=value, type=vtype)
    try:
        _commit_step(db,
            "Backfilled max_capacity_coefficient and "
            "min_capacity_coefficient from flow_coefficient for "
            "entities where flow_coefficient != 1.0")
    except SpineDBAPIError:
        pass


def _migrate_v37(db) -> None:
    # v35/v36 introduced flow_coefficient with the *old* sink
    # semantics (v_flow divided by flow_coefficient in the
    # balance — asymmetric with the source side, which
    # multiplies). v37 flips the sink side to multiplication in
    # flextool.mod so both sides mean "fuel-equivalent energy
    # per unit of flow". To preserve numerical results for
    # existing databases, invert every explicit non-zero
    # flow_coefficient value on unit__outputNode: replace x
    # with 1/x. Defaults (1.0) are left alone; 0 (hydro-
    # pass-through marker) is left alone.
    db.mapped_table("parameter_value")
    caps = _float_values(db, "unit__outputNode", ("max_capacity_coefficient", "min_capacity_coefficient"))
    for pv in list(db.find_parameter_values(
            entity_class_name="unit__outputNode",
            parameter_definition_name="flow_coefficient")) if not _v38_ran(db, 37) else ():
        try:
            val = float(pv["parsed_value"])
        except (TypeError, ValueError):
            continue
        if val == 0.0 or val == 1.0:
            continue
        inv = 1.0 / val
        if _v37_flipped(caps, pv, inv):
            _already_applied(37, f"unit__outputNode {tuple(pv['entity_byname'])} flow_coefficient is flipped")
            continue
        new_val, new_type = to_database(inv)
        db.update_item(
            "parameter_value",
            id=pv["id"],
            value=new_val, type=new_type)
    try:
        _commit_step(db,
            "Flipped unit__outputNode.flow_coefficient values "
            "to 1/x to match the new multiplicative semantics "
            "on the sink side of the balance")
    except SpineDBAPIError:
        pass


def _migrate_v38(db) -> None:
    # Consolidate has_balance / has_storage / node_type (which
    # previously carried the single value 'balance_within_period')
    # into one node_type parameter with four allowed values:
    # 'commodity', 'balance', 'storage', 'balance_within_period'.
    # Hard cut — the two yes/no flags are dropped.  See
    # the node-type consolidation plan.

    # 1. Extend the node_type value list with the three new entries.
    #    'balance_within_period' was already added in v22.
    add_value_list_manual(db, [
        ["node_type", "commodity"],
        ["node_type", "balance"],
        ["node_type", "storage"],
    ])

    # 2. Re-declare node_type with default='balance' and an updated
    #    description.  'balance' is the most common user intent;
    #    nodes that previously relied on the 'no balance' default
    #    will get 'commodity' written explicitly in step 3 so their
    #    semantics are preserved.
    default_val, default_type = to_database("balance")
    db.add_update_item(
        "parameter_definition",
        entity_class_name="node", name="node_type",
        default_value=default_val, default_type=default_type,
        parameter_value_list_name="node_type",
        description=(
            "Role of this node in the LP.  "
            "'commodity' = price-exposed source/sink with no "
            "balance constraint (e.g. fuel imports, no storage); "
            "'balance' = energy balance maintained every timestep "
            "(default); 'storage' = balance plus a state variable "
            "(battery, reservoir); 'balance_within_period' = "
            "balance aggregated over the whole period (e.g. an "
            "annual gas budget)."
        ),
    )

    # 3. Derive node_type from the old flags for every (node, alt)
    #    pair that has any relevant data, and explicitly write
    #    'commodity' for node entities that had no flag anywhere
    #    so their prior 'no balance' semantics are preserved.
    def _is_yes(pv):
        val = pv.get("parsed_value")
        return val is not None and str(val).lower() == "yes"

    has_balance_by_key = {
        (pv["entity_byname"], pv["alternative_name"]): _is_yes(pv)
        for pv in db.find_parameter_values(
            entity_class_name="node",
            parameter_definition_name="has_balance",
        )
    }
    has_storage_by_key = {
        (pv["entity_byname"], pv["alternative_name"]): _is_yes(pv)
        for pv in db.find_parameter_values(
            entity_class_name="node",
            parameter_definition_name="has_storage",
        )
    }
    existing_node_type_keys = {
        (pv["entity_byname"], pv["alternative_name"])
        for pv in db.find_parameter_values(
            entity_class_name="node",
            parameter_definition_name="node_type",
        )
    }

    all_keys_with_flags = (
        set(has_balance_by_key.keys())
        | set(has_storage_by_key.keys())
        | existing_node_type_keys
    )

    for key in all_keys_with_flags:
        if key in existing_node_type_keys:
            # Explicit node_type already present — the only
            # pre-v38 value was 'balance_within_period' and it
            # wins over any has_balance / has_storage flag on
            # the same (node, alt) pair.
            continue
        hb = has_balance_by_key.get(key, False)
        hs = has_storage_by_key.get(key, False)
        if hs:
            if not hb:
                # Pre-v38 this combination was rejected at solve
                # time, but the data is recoverable: storage
                # implies balance, so migrate as
                # node_type='storage' and warn so the user can
                # verify intent.
                logging.warning(
                    "Node '%s' (alternative '%s'): "
                    "has_storage=yes with has_balance!=yes — "
                    "this combination was rejected at solve time "
                    "prior to v38.  Migrating as "
                    "node_type='storage' (storage implies "
                    "balance).  Verify that the resulting node "
                    "balance is the intended behaviour.",
                    key[0][0], key[1],
                )
            new_type = "storage"
        elif hb:
            new_type = "balance"
        else:
            new_type = "commodity"
        value, vtype = to_database(new_type)
        db.add_update_item(
            "parameter_value",
            entity_class_name="node",
            entity_byname=key[0],
            parameter_definition_name="node_type",
            alternative_name=key[1],
            value=value, type=vtype,
        )

    # Preserve the pre-v38 'no balance' default for nodes that
    # have no explicit node_type.  The new schema default is
    # 'balance', so without this step a node that was
    # 'no balance' in v25 (no has_balance flag) would silently
    # gain a per-timestep balance constraint.
    #
    # Strategy — a single dedicated FLOOR alternative, never a
    # blanket 'Base' rewrite.  Create one alternative
    # (``default_node_type_from_migration``), add it to every
    # scenario at a rank STRICTLY BELOW all existing
    # alternatives, and write node_type='commodity' for every
    # node into it.  Because the floor sits at the lowest rank
    # in EVERY scenario, any real 'balance'/'storage' (written
    # above from the flags) or pre-existing
    # 'balance_within_period' always outranks and overrides it.
    # This reproduces the old 'absent flag ⇒ commodity' default
    # exactly, without the old 'Base' hazard — 'Base' floats to
    # the TOP in some scenarios and there overrode the physics;
    # a bottom-pinned dedicated alternative cannot, in any
    # scenario or across shared scenarios.  (Note: this does not
    # repair databases already migrated by the previous, buggy
    # v38 step — those need the stray node_type values cleaned
    # up by hand.)
    floor_alt = "default_node_type_from_migration"
    # The floor goes with the flags' removal (one commit): without both flags it is there, and the
    # nodes and scenarios made since then must not get it.
    write_floor = (any(db.get_item("parameter_definition", entity_class_name="node", name=name)
                       for name in ("has_balance", "has_storage"))
                   or not _already_applied(38, "node.has_balance and has_storage are removed"))
    if write_floor:
        db.add_update_item(
            "alternative",
            name=floor_alt,
            description=(
                "Auto-created by the v38 migration: carries the "
                "'commodity' fallback node_type (pre-v38 'no "
                "balance' default).  Pinned to the lowest rank of "
                "every scenario so any explicit 'balance'/'storage'/"
                "'balance_within_period' overrides it."
            ),
        )

    # Add the floor alternative to every scenario at a rank
    # strictly below the scenario's current minimum.
    scen_name_by_id = {
        s["id"]: s["name"] for s in db.find_scenarios()
    }
    min_rank_by_scenario: dict = {}
    for sa in db.find_scenario_alternatives():
        sid = sa["scenario_id"]
        r = sa["rank"]
        if sid not in min_rank_by_scenario or r < min_rank_by_scenario[sid]:
            min_rank_by_scenario[sid] = r
    for sid, min_rank in min_rank_by_scenario.items() if write_floor else ():
        db.add_update_item(
            "scenario_alternative",
            scenario_name=scen_name_by_id[sid],
            alternative_name=floor_alt,
            rank=min_rank - 1,
        )

    # Write the 'commodity' floor for every node.  Nodes that
    # got 'balance'/'storage' from the flags (or carry
    # 'balance_within_period') keep those in higher-ranked
    # alternatives and are unaffected here.
    commodity_val, commodity_type = to_database("commodity")
    for ent in db.find_entities(entity_class_name="node") if write_floor else ():
        db.add_update_item(
            "parameter_value",
            entity_class_name="node",
            entity_byname=ent["entity_byname"],
            parameter_definition_name="node_type",
            alternative_name=floor_alt,
            value=commodity_val, type=commodity_type,
        )

    # 4. Remove the old has_balance / has_storage parameter
    #    definitions and their single-entry value lists.
    remove_parameters_manual(db, [
        ["node", "has_balance"],
        ["node", "has_storage"],
    ])
    for vl_name in ("has_balance", "has_storage"):
        vl = db.get_item("parameter_value_list", name=vl_name)
        if vl:
            db.remove_items("parameter_value_list", vl["id"])
        else:
            _already_applied(38, f"value list {vl_name} is removed")

    try:
        _commit_step(db,
            "v38: consolidated has_balance, has_storage and "
            "node_type ('balance_within_period') into a single "
            "node_type parameter with four values"
        )
    except SpineDBAPIError:
        pass


def _migrate_v39(db) -> None:
    # Replace the hard-coded 1e6 upper bound on "unconstrained"
    # variables (invest_no_limit + zero-coefficient flow bounds)
    # with a model-level parameter.  Default 1,000,000 matches
    # the previous behaviour.  Needed by the commodity price
    # ladder feature to bound infinite-capacity tiers without
    # baking another literal into the mod.
    default_val, default_type = to_database(1000000.0)
    db.add_update_item(
        "parameter_definition",
        entity_class_name="model",
        name="max_flow_for_unconstrained_variables",
        default_value=default_val, default_type=default_type,
        parameter_type_list=("float",),
        description=(
            "[MW] Upper bound assigned to LP variables that "
            "have no other cap (invest_no_limit capacity; "
            "flows through edges whose max_capacity_coefficient "
            "is zero; infinite-capacity commodity tiers).  "
            "Keep large enough not to constrain the physical "
            "solution but small enough to avoid numerical "
            "issues (default 1,000,000)."
        ),
    )
    try:
        _commit_step(db,
            "v39: added model.max_flow_for_unconstrained_variables "
            "(replaces hard-coded 1e6 in flextool.mod)"
        )
    except SpineDBAPIError:
        pass


def _migrate_v40(db) -> None:
    # Commodity price ladder — foundation commit.  Adds the three
    # new commodity-entity parameters and the price_method value
    # list.  No LP behaviour yet — v_trade / ladder constraints /
    # objective terms land in a later commit.
    add_value_list_manual(db, [
        ["price_method", "price"],
        ["price_method", "price_ladder_annual"],
        ["price_method", "price_ladder_cumulative"],
    ])
    default_val, default_type = to_database("price")
    db.add_update_item(
        "parameter_definition",
        entity_class_name="commodity",
        name="price_method",
        default_value=default_val, default_type=default_type,
        parameter_value_list_name="price_method",
        parameter_type_list=("str",),
        description=(
            "How the commodity's price enters the LP.  "
            "'price' = scalar or time-series price x flow "
            "(current behaviour); 'price_ladder_annual' = "
            "stepped supply curve with a per-year quantity cap "
            "per tier; 'price_ladder_cumulative' = stepped "
            "supply curve with a total-model quantity cap per "
            "tier (handoff-carried across rolling solves)."
        ),
    )
    default_val, default_type = to_database(1.0)
    db.add_update_item(
        "parameter_definition",
        entity_class_name="commodity",
        name="unitsize",
        default_value=default_val, default_type=default_type,
        parameter_type_list=("float",),
        description=(
            "Numeric scaling for the v_trade variable column "
            "(analogous to virtual_unitsize on process/node "
            "entities).  The variable is in user-MWh divided "
            "by this value.  Pick so the largest tier sits at "
            "O(10) in the scaled LP."
        ),
    )
    db.add_update_item(
        "parameter_definition",
        entity_class_name="commodity",
        name="price_ladder_cumulative",
        parameter_type_list=("2d_map",),
        description=(
            "Stepped supply curve for "
            "price_method='price_ladder_cumulative'.  "
            "2d map with rows 'tier,price,quantity' — one row "
            "per tier, giving the tier's unit price and its "
            "cumulative quantity cap.  1-based integer tier "
            "index.  quantity=inf marks an unbounded tail tier.  "
            "Period-agnostic — the cap is a single total across "
            "the full model horizon."
        ),
    )
    db.add_update_item(
        "parameter_definition",
        entity_class_name="commodity",
        name="price_ladder_annual",
        parameter_type_list=("2d_map", "3d_map"),
        description=(
            "Stepped supply curve for "
            "price_method='price_ladder_annual'.  Two forms "
            "accepted: 2d map with rows 'tier,price,quantity' "
            "applies the same per-year limit every period; 3d "
            "map with rows 'period,tier,price,quantity' varies "
            "the limit per period.  1-based integer tier.  "
            "quantity=inf marks an unbounded tail tier."
        ),
    )
    # Correct a stale parameter_type_list on
    # reserve__upDown__unit__node.increase_reserve_ratio: v23's
    # inline type list had ("str",), but the parameter is a
    # ratio and the sibling reserve__upDown__connection__node
    # already carries ("float",).  Patch it here in v40 (rather
    # than editing the historical v23 data) so DBs coming
    # through v23 -> v40 converge on the correct type.
    db.add_update_item(
        "parameter_definition",
        entity_class_name="reserve__upDown__unit__node",
        name="increase_reserve_ratio",
        parameter_type_list=("float",),
    )
    try:
        _commit_step(db,
            "v40: added commodity.price_method, commodity.unitsize, "
            "commodity.price_ladder_cumulative and "
            "commodity.price_ladder_annual (no LP behaviour yet); "
            "corrected reserve__upDown__unit__node."
            "increase_reserve_ratio parameter_type_list str -> float"
        )
    except SpineDBAPIError:
        pass


def _migrate_v41(db) -> None:
    # Fix the "storate_state_end" typo on node.  The correct
    # parameter "storage_state_end" already exists (added in
    # v22/v23) but a typo'd sibling has lived alongside it
    # since v23.  Some user DBs may have accumulated values on
    # the typo'd name via imported templates.  Migrate those
    # values to the correct name non-destructively, then drop
    # the typo'd definition.
    typo_name = "storate_state_end"
    good_name = "storage_state_end"

    # 1. Ensure the correct parameter exists.  In practice it
    #    will already have been added by v22/v23, but defend
    #    against edge cases where it was removed or never
    #    materialised.
    existing_good = list(db.find_parameter_definitions(
        entity_class_name="node", name=good_name,
    ))
    if not existing_good:
        default_val, default_type = to_database(0.0)
        db.add_update_item(
            "parameter_definition",
            entity_class_name="node", name=good_name,
            default_value=default_val, default_type=default_type,
            parameter_type_list=("float",),
            description=(
                "[0-1] Relative state of storage at the end of "
                "the last model solve (overrides "
                "'storage_state_end_reference'). Constant."
            ),
        )

    # 2. Copy any parameter values on the typo'd name over to
    #    the good name if the target (node, alternative) slot
    #    is not already populated.
    good_value_keys = {
        (pv["entity_byname"], pv["alternative_name"])
        for pv in db.find_parameter_values(
            entity_class_name="node",
            parameter_definition_name=good_name,
        )
    }
    for pv in list(db.find_parameter_values(
            entity_class_name="node",
            parameter_definition_name=typo_name)):
        key = (pv["entity_byname"], pv["alternative_name"])
        if key in good_value_keys:
            continue  # non-destructive: keep existing good value
        db.add_update_item(
            "parameter_value",
            entity_class_name="node",
            entity_byname=pv["entity_byname"],
            parameter_definition_name=good_name,
            alternative_name=pv["alternative_name"],
            value=pv["value"], type=pv["type"],
        )
        good_value_keys.add(key)

    # 3. Delete the typo'd parameter definition (cascades to
    #    its parameter_value rows).
    typo_defs = list(db.find_parameter_definitions(
        entity_class_name="node", name=typo_name,
    ))
    if typo_defs:
        _remove_items_bounded(
            db, "parameter_definition", [typo_defs[0]["id"]],
            "v41: removed node.storate_state_end",
        )

    try:
        _commit_step(db,
            "v41: fix storate_state_end typo -> storage_state_end"
        )
    except SpineDBAPIError:
        pass


def _migrate_v42(db) -> None:
    # Rename + split of the three group-level output-control
    # parameters:
    #   output_node_flows     -> output_nodeGroup_dispatch
    #   output_aggregate_flows -> flow_aggregator
    #   output_results        -> output_nodeGroup_indicators
    #                            + output_flowGroup_indicators
    #                            (split based on group memberships:
    #                             group__node -> indicators_node,
    #                             group__unit__node or
    #                             group__connection__node ->
    #                             indicators_flow, both -> both).
    # The new parameters use the yes_no value list and default to
    # unset (equivalent to "no").
    add_value_list_manual(db, [
        ["yes_no", "yes"], ["yes_no", "no"]
    ])

    # 1. Add the four new parameter definitions.  Descriptions
    #    are minimal here; Agent 3 will enrich them.  Not one that v58
    #    renamed already.
    _upsert_definition(
        db, 42, "group",
        "output_nodeGroup_dispatch",
        parameter_value_list_name="yes_no",
        parameter_type_list=("str",),
        description=(
            "Creates the timewise flow output for this node "
            "group (node-group dispatch table). Renamed from "
            "output_node_flows."
        ),
    )
    _upsert_definition(
        db, 42, "group",
        "flow_aggregator",
        parameter_value_list_name="yes_no",
        parameter_type_list=("str",),
        description=(
            "Used with group_unit_node or group_connection_node "
            "to combine the flows when producing the dispatch "
            "output of a node group. Renamed from "
            "output_aggregate_flows."
        ),
    )
    _upsert_definition(
        db, 42, "group",
        "output_nodeGroup_indicators",
        parameter_value_list_name="yes_no",
        parameter_type_list=("str",),
        description=(
            "Flag to output node-group indicator results for "
            "groups whose members are nodes (group__node)."
        ),
    )
    _upsert_definition(
        db, 42, "group",
        "output_flowGroup_indicators",
        parameter_value_list_name="yes_no",
        parameter_type_list=("str",),
        description=(
            "Flag to output flow-group indicator results for "
            "groups whose members are flows "
            "(group__unit__node or group__connection__node)."
        ),
    )

    # 2. Copy values from old parameters to their direct renames.
    #    output_node_flows     -> output_nodeGroup_dispatch
    #    output_aggregate_flows -> flow_aggregator
    rename_map = {
        "output_node_flows": "output_nodeGroup_dispatch",
        "output_aggregate_flows": "flow_aggregator",
    }
    for old_name, new_name in rename_map.items():
        existing_new_keys = {
            (pv["entity_byname"], pv["alternative_name"])
            for pv in db.find_parameter_values(
                entity_class_name="group",
                parameter_definition_name=new_name,
            )
        }
        for pv in list(db.find_parameter_values(
                entity_class_name="group",
                parameter_definition_name=old_name)):
            key = (pv["entity_byname"], pv["alternative_name"])
            if key in existing_new_keys:
                continue
            db.add_update_item(
                "parameter_value",
                entity_class_name="group",
                entity_byname=pv["entity_byname"],
                parameter_definition_name=new_name,
                alternative_name=pv["alternative_name"],
                value=pv["value"], type=pv["type"],
            )
            existing_new_keys.add(key)

    # 3. Split output_results into the two new indicator
    #    parameters based on the group's memberships.
    #    - group__node members -> output_nodeGroup_indicators
    #    - group__unit__node or group__connection__node members
    #      -> output_flowGroup_indicators
    #    - both kinds present -> both parameters written
    #    - neither -> drop silently
    groups_with_node_members: set[tuple] = set()
    for ent in db.find_entities(entity_class_name="group__node"):
        # entity_byname is (group, node)
        byname = ent["entity_byname"]
        if byname:
            groups_with_node_members.add((byname[0],))
    groups_with_flow_members: set[tuple] = set()
    for cls in ("group__unit__node", "group__connection__node"):
        for ent in db.find_entities(entity_class_name=cls):
            byname = ent["entity_byname"]
            if byname:
                groups_with_flow_members.add((byname[0],))

    existing_node_ind_keys = {
        (pv["entity_byname"], pv["alternative_name"])
        for pv in db.find_parameter_values(
            entity_class_name="group",
            parameter_definition_name="output_nodeGroup_indicators",
        )
    }
    existing_flow_ind_keys = {
        (pv["entity_byname"], pv["alternative_name"])
        for pv in db.find_parameter_values(
            entity_class_name="group",
            parameter_definition_name="output_flowGroup_indicators",
        )
    }
    for pv in list(db.find_parameter_values(
            entity_class_name="group",
            parameter_definition_name="output_results")):
        byname = pv["entity_byname"]
        alt = pv["alternative_name"]
        has_nodes = byname in groups_with_node_members
        has_flows = byname in groups_with_flow_members
        if has_nodes and (byname, alt) not in existing_node_ind_keys:
            db.add_update_item(
                "parameter_value",
                entity_class_name="group",
                entity_byname=byname,
                parameter_definition_name="output_nodeGroup_indicators",
                alternative_name=alt,
                value=pv["value"], type=pv["type"],
            )
            existing_node_ind_keys.add((byname, alt))
        if has_flows and (byname, alt) not in existing_flow_ind_keys:
            db.add_update_item(
                "parameter_value",
                entity_class_name="group",
                entity_byname=byname,
                parameter_definition_name="output_flowGroup_indicators",
                alternative_name=alt,
                value=pv["value"], type=pv["type"],
            )
            existing_flow_ind_keys.add((byname, alt))
        # neither => drop silently (no-op)

    # 4. Remove old parameter definitions (cascades values).
    remove_parameters_manual(db, [
        ["group", "output_node_flows"],
        ["group", "output_aggregate_flows"],
        ["group", "output_results"],
    ])

    # 5. Remove old value lists now that nothing references them.
    # Use find_parameter_value_lists (plural) so a DB rebuilt from
    # a JSON fixture that pre-dates these legacy lists (e.g.
    # tests.json was exported after v38, never carrying the v8-era
    # output_node_flows list) doesn't blow up on a strict ``db.item``
    # lookup that raises when the row is absent.
    for vl_name in ("output_node_flows", "output_results"):
        vls = list(db.find_parameter_value_lists(name=vl_name))
        if vls:
            try:
                db.remove_items("parameter_value_list", vls[0]["id"])
            except SpineDBAPIError:
                pass

    try:
        _commit_step(db,
            "v42: renamed output_node_flows -> "
            "output_nodeGroup_dispatch, output_aggregate_flows "
            "-> flow_aggregator; split output_results into "
            "output_nodeGroup_indicators + "
            "output_flowGroup_indicators based on group "
            "memberships"
        )
    except SpineDBAPIError:
        pass


def _migrate_v43(db) -> None:
    # Parameter-group metadata foothold.  Create an "Outputs"
    # parameter_group and tag the four group-level output
    # parameters with it.  Spine's parameter_definition table
    # carries an optional parameter_group_name slot (the 6th
    # slot in the export 6-tuple) that FlexTool has never
    # populated.  This migration is a deliberately narrow
    # foothold: future parameter additions should categorise
    # themselves using the same mechanism (see
    # docs/reference/parameter_groups.md, "Parameter groups (metadata)").
    # v44 renames the group "output": with that one and without "Outputs", v43 is applied.
    if not db.get_item("parameter_group", name="Outputs") and db.get_item("parameter_group", name="output"):
        _already_applied(43, 'parameter group "Outputs" is "output"')
    else:
        db.add_update_item(
            "parameter_group",
            name="Outputs",
            color="a6cee3",  # light blue, 6-hex-digit, no '#'
            priority=10,
        )
        for param_name in (
            "output_nodeGroup_dispatch",
            "output_nodeGroup_indicators",
            "output_flowGroup_indicators",
            "flow_aggregator",
        ):
            _upsert_definition(db, 43, "group", param_name, parameter_group_name="Outputs")
    try:
        _commit_step(db,
            "v43: added 'Outputs' parameter_group and tagged "
            "the four group-level output parameters "
            "(output_nodeGroup_dispatch, "
            "output_nodeGroup_indicators, "
            "output_flowGroup_indicators, flow_aggregator) "
            "with it"
        )
    except SpineDBAPIError:
        pass


def _float_values(db, entity_class_name: str, names) -> dict:
    """``{(parameter, entity_byname, alternative): float}`` of the scalar *names* values of *entity_class_name*."""
    out = {}
    for name in names:
        for pv in db.find_parameter_values(entity_class_name=entity_class_name, parameter_definition_name=name):
            try:
                out[(name, tuple(pv["entity_byname"]), pv["alternative_name"])] = float(pv["parsed_value"])
            except (TypeError, ValueError):
                continue
    return out


def _v37_flipped(caps: dict, pv, inv: float) -> bool:
    """v36 wrote ``max_capacity_coefficient = min_capacity_coefficient = flow_coefficient`` on the unit__outputNode
    row *pv* (*caps*: :func:`_float_values`), v37 then flipped it: max = min = 1/flow (*inv*) means flipped.  (On
    unflipped content max = min = flow, so that holds only for flow = 1, which both steps skip.)

    Known limit (review m10): on genuine v35 / v36 content a user value max = min = 1/flow in the row's alternative
    reads as flipped, and v36 / v37 then keep it unflipped.  No release shipped DB version 35 or 36 (v3.23.0 is 31,
    v3.24.0 is 37), so only dev databases can meet it."""
    key = (tuple(pv["entity_byname"]), pv["alternative_name"])
    return all(math.isclose(caps.get((name, *key), math.nan), inv, rel_tol=1e-12)
               for name in ("max_capacity_coefficient", "min_capacity_coefficient"))


def _v38_ran(db, version: int) -> bool:
    """v38 removed ``node.has_balance`` and its value list, which its clean run reads: both absent means v38 ran, and
    v36 / v37 (step *version*) must not rescale or flip ``flow_coefficient`` again.  It (and v38's floor) keys on the
    definitions, not on content: a pre-38 database whose user deleted both definitions skips the rescaling and the
    floor (review m11; very unlikely)."""
    if (db.get_item("parameter_value_list", name="has_balance")
            or db.get_item("parameter_definition", entity_class_name="node", name="has_balance")):
        return False
    return _already_applied(version, "v38 removed node.has_balance")


def _v35_drop_v23_flow_coefficient(db, cls: str) -> None:
    """Remove the ``flow_coefficient`` definition that the v23 step of FlexTool 4.5.0 and older created on *cls*.

    That step typed every name of its list and so created ``flow_coefficient`` (type ``float``, no description, no
    default, no values) beside ``coefficient``; v35 renames ``coefficient`` to ``flow_coefficient`` and collided with
    it.  The empty definition is removed so the rename keeps the ``coefficient`` values.  A ``flow_coefficient`` that
    holds values (or a default) is not that definition: the step stops with a message instead of choosing between them.
    """
    stray = db.get_item("parameter_definition", entity_class_name=cls, name="flow_coefficient")
    if not stray:
        return
    values = db.find_parameter_values(entity_class_name=cls, parameter_definition_name="flow_coefficient")
    if values or from_database(stray["default_value"], stray["default_type"]) is not None:
        raise SpineDBAPIError(
            f"v35: {cls} has both coefficient and flow_coefficient, and flow_coefficient has values or a default; "
            f"coefficient is renamed to flow_coefficient in this version.  Move the values you want to keep to "
            f"coefficient and remove the flow_coefficient definition, then migrate again.")
    db.remove_parameter_definition(id=stray["id"])
