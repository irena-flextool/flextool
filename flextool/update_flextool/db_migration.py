import json
import os
import argparse
from spinedb_api import import_data, DatabaseMapping, from_database, SpineDBAPIError, to_database
from spinedb_api.exception import NothingToCommit
import logging

from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool._resources import package_data_path


def _pre_v26_template(name: str) -> str:
    """Return the absolute filesystem path of a bundled pre-v26 template JSON."""
    return str(package_data_path(f"schemas/pre_v26/{name}"))


def _commit_step(db, message):
    """Commit a migration step, tolerating the no-op case.

    spinedb_api raises ``NothingToCommit`` when ``add_update_item`` /
    ``update_item`` calls produced zero net changes — typically because
    the requested rows already exist with identical fields (a step that
    has been hand-applied or carried in by an earlier partial migration
    that never bumped ``model.version``).  Treating that signal as
    fatal aborts the rest of the migration mid-chain and the version
    bump never persists, leaving the DB stuck at the pre-step version.
    Log and continue so subsequent steps still run.
    """
    try:
        db.commit_session(message)
    except NothingToCommit:
        logging.info("Migration step idempotent (no changes): %s", message)


def migrate_database(database_path, up_to: int | None = None):
    """Migrate a FlexTool database to a target schema version.

    Args:
        database_path: Path or URL to the SQLite database.
        up_to: Target version to migrate to.  When ``None`` (the default),
            migrates all the way to :data:`FLEXTOOL_DB_VERSION`.
    """

    if database_path.startswith('sqlite://') or database_path.startswith('http://'):
        mapping_name = database_path
    elif os.path.exists(database_path) and database_path.endswith(".sqlite"):
        mapping_name = 'sqlite:///' + database_path
    else:
        logging.critical("No sqlite file at " + database_path)
        exit(-1)

    with DatabaseMapping(mapping_name, create = False, upgrade = True) as db:
        sq= db.object_parameter_definition_sq
        settings_parameter = db.query(sq).filter(sq.c.object_class_name == "model").filter(sq.c.parameter_name == "version").one_or_none()
        if settings_parameter is None:
            #if no version assume version 0
            print("No version found. Assuming version 0, if older, migration might not work")
            version = 0
        else:
            version = from_database(settings_parameter.default_value, settings_parameter.default_type)

        next_version = int(version) + 1
        new_version = up_to if up_to is not None else FLEXTOOL_DB_VERSION

        while next_version <= new_version:
            if next_version == 0:
                add_version(db)
            elif next_version == 1:
                add_new_parameters(db, _pre_v26_template('flextool_template_v2.json'))
            elif next_version == 2:
                add_new_parameters(db, _pre_v26_template('flextool_template_rolling_window.json'))
            elif next_version == 3:
                add_new_parameters(db, _pre_v26_template('flextool_template_lifetime_method.json'))
            elif next_version == 4:
                add_new_parameters(db, _pre_v26_template('flextool_template_drop_down.json'))
            elif next_version == 5:
                add_new_parameters(db, _pre_v26_template('flextool_template_optional_outputs.json'))
            elif next_version == 6:
                add_new_parameters(db, _pre_v26_template('flextool_template_default_value.json'))
            elif next_version == 7:
                add_new_parameters(db, _pre_v26_template('flextool_template_rolling_start_remove.json'))
            elif next_version == 8:
                add_new_parameters(db, _pre_v26_template('flextool_template_output_node_flows.json'))
            elif next_version == 9:
                add_new_parameters(db, _pre_v26_template('flextool_template_constant_default.json'))
            elif next_version == 10:
                add_new_parameters(db, _pre_v26_template('flextool_template_storage_binding_defaults.json'))
            elif next_version == 11:
                change_optional_output_type(db, _pre_v26_template('flextool_template_default_optional_output.json'))
            elif next_version == 12:
                new_parameters = [["group", "output_aggregate_flows", None, "output_node_flows", "Used with group_unit_node or group_connection_node to combine the flows when producing the output_node_flows of a node group."],
                                  ["group", "output_node_flows", None, "output_node_flows" ,"Creates the timewise flow output for this node group (group_flow_t)"]]
                add_parameters_manual(db,new_parameters)
                remove_parameters_manual(db,[["solve","rolling_start_time"]])
            elif next_version == 13:
                new_value_list = [["load_share_type","equal"],["load_share_type","inflow_weighted"],["load_share_type","no"]]
                new_parameters = [["group", "share_loss_of_load", "no", "load_share_type", "Force the upward slack of the nodes in this group to be equal or inflow (demand) weighted"]]
                #value list needs to be added first
                add_value_list_manual(db,new_value_list)
                add_parameters_manual(db,new_parameters)
                remove_parameters_manual(db,[["node","storate_state_end"]])
            elif next_version == 14:
                new_parameters = [["model","exclude_entity_outputs", "yes", "optional_output", "Excludes results on node, unit and connection level, but preserves group level results"]]
                remove_parameters_manual(db, [["model","results"]])
                add_parameters_manual(db, new_parameters)
            elif next_version == 15:
                remove_parameters_manual(db,[["unit", "invest_forced"], ["unit", "retire_forced"], ["connection", "invest_forced"], ["connection", "retire_forced"],
                                             ["node", "invest_forced"], ["node", "retire_forced"]])
                add_parameters_manual(db, [["group", "co2_max_period", "no_method", "co2_methods", "[tCO2] Annualized maximum limit for emitted CO2 in each period."]])
            elif next_version == 16:
                add_parameters_manual(db, [["group", "co2_max_period", None, None, "[tCO2] Annualized maximum limit for emitted CO2 in each period."]])
            elif next_version == 17:
                add_value_list_manual(db,[["yes_no", "yes"], ["yes_no", "no"]])
                new_parameters = [["solve", "stochastic_branches", None, None, "[4d-Map], Sets branches included in the solve. [Period, branch, start_time (time_step), realized (yes/no), weight (number)]. Only one of the branches should be realized for each start_time"],
                                  ["group", "include_stochastics", "no", "yes_no", "Includes the stochastic branches to be used for the nodes/units/connections in this group"],
                                  ["model", "output_horizon", "no", "yes_no", "Outputs the flows in the horizons. Used for testing the model."]]
                add_parameters_manual(db,new_parameters)
            elif next_version == 18:
                new_parameters = [["group", "penalty_inertia", 5000, None, "[CUR/MWs] Penalty for violating the inertia constraint. Constant or period."],
                                  ["group", "penalty_capacity_margin", 5000, None, "[CUR/MWh] Penalty for violating the capacity margin constraint. Constant or period."],
                                  ["group", "penalty_non_synchronous", 5000, None, "[CUR/MWh] Penalty for violating the non synchronous constraint. Constant or period."]]
                new_relationships = [["reserve__upDown__group", "penalty_reserve", 5000, None, "[CUR/MW] Penalty for violating a reserve constraint. Constant."]]
                add_parameters_manual(db,new_parameters)
                add_relationships_manual(db,new_relationships)
            elif next_version == 19:
                remove_parameters_manual(db, [["constraint", "is_active"], ["reserve__upDown__unit__node", "is_active"]])
            elif next_version == 20:
                remove_parameters_manual(db, [["connection", "is_active"], ["node", "is_active"], ["unit", "is_active"], ["reserve__upDown__connection", "is_active"]])
            elif next_version == 21:
                new_value_list = [["storage_nested_fix_method","fix_nothing"],["storage_nested_fix_method","fix_quantity"],["storage_nested_fix_method","fix_price"], ["storage_nested_fix_method","fix_usage"]]
                add_value_list_manual(db,new_value_list)
            elif next_version == 22:
                db.add_update_item("parameter_value_list", name = "node_type")
                add_value_list_manual(db,[["node_type","balance_within_period"],["invest_methods","cumulative_limits"]])
                db.add_update_item("parameter_definition", entity_class_name= "node", name= "node_type", parameter_value_list_name = "node_type", description = "Selection for the node to have period balance, instead of time step balance.")
                db.add_update_item("parameter_definition", entity_class_name= "node", name= "cumulative_max_capacity", description = "[MWh] Maximum cumulative capacity (considers existing, invested and retired capacity). Constant or period.")
                db.add_update_item("parameter_definition", entity_class_name= "node", name= "cumulative_min_capacity", description = "[MWh] Minimum cumulative capacity (considers existing, invested and retired capacity). Constant or period.")
                db.add_update_item("parameter_definition", entity_class_name= "connection", name= "cumulative_max_capacity", description = "[MW] Maximum cumulative capacity (considers existing, invested and retired capacity). Constant or period.")
                db.add_update_item("parameter_definition", entity_class_name= "connection", name= "cumulative_min_capacity", description = "[MW] Minimum cumulative capacity (considers existing, invested and retired capacity). Constant or period.")
                db.add_update_item("parameter_definition", entity_class_name= "unit", name= "cumulative_max_capacity", description = "[MW] Maximum cumulative capacity (considers existing, invested and retired capacity). Constant or period.")
                db.add_update_item("parameter_definition", entity_class_name= "unit", name= "cumulative_min_capacity", description = "[MW] Minimum cumulative capacity (considers existing, invested and retired capacity). Constant or period.")
                db.update_item("parameter_definition", entity_class_name= "node", name= "existing", description = "[MWh] Existing storage capacity. Constant or Period")
                db.update_item("parameter_definition", entity_class_name= "connection", name= "existing", description = "[MW] Existing capacity. Constant or Period")
                db.update_item("parameter_definition", entity_class_name= "unit", name= "existing", description = "[MW] Existing capacity. Constant or Period")
                db.update_item("parameter_definition", entity_class_name= "node", name= "penalty_up", description = "[CUR/MW] Penalty cost for decreasing consumption in the node. Constant, Period or Time.")
                db.update_item("parameter_definition", entity_class_name= "node", name= "penalty_down", description = "[CUR/MW] Penalty cost for increasing consumption in the node. Constant, Period or Time.")
                db.update_item("parameter_definition", entity_class_name= "unit__outputNode", name= "other_operational_cost", description = "[CUR/MWh] Other operational variable cost for energy flows. Constant, Period or Time.")
                db.update_item("parameter_definition", entity_class_name= "unit__inputNode", name= "other_operational_cost", description = "[CUR/MWh] Other operational variable cost for energy flows. Constant, Period or Time.")
                db.update_item("parameter_definition", entity_class_name= "connection", name= "other_operational_cost", description = "[CUR/MWh] Other operational variable cost for trasferring over the connection. Constant, Period or time.")
                db.update_item("parameter_definition", entity_class_name= "solve", name= "solve_mode", description = "A single_solve or rolling_window for a set of rolling optimisation windows solved in a sequence.")
                _commit_step(db,"Added cumulative investments")
            elif next_version == 23:
                db.add_update_item("parameter_definition", entity_class_name= "commodity", name= "price", description = "[CUR/MWh or other unit] Price of the commodity. Constant, period or time.")
                db.add_update_item("parameter_definition", entity_class_name= "group", name= "co2_price", description = "[CUR/ton] CO2 price for a group of nodes. Constant, period or time.")
                update_parameter_types_v23(db)
            elif next_version == 24:
                db.add_update_item("parameter_definition", entity_class_name= "connection", name= "delay", parameter_type_list = ("float","1d_map"), description = "[hours] A time delay between the input node and the output node - works only with one-way connections (or units). Either a constant indicating the time difference in hours or a map of time differences (index: time difference in hours, value: weight). Each weight indicates its share of the original flow and the weights should sum to 1. Requires that the time resolutions in the model are always integer multiples of these time differences.")
                db.add_update_item("parameter_definition", entity_class_name= "unit", name= "delay", parameter_type_list = ("float","1d_map"), description = "[hours] A time delay between the input nodes and the output nodes. Either a constant indicating the time difference in hours or a map of time differences (index: time difference in hours, value: weight). Each weight indicates its share of the original flow and the weights should sum to 1. Requires that the time resolutions in the model are always integer multiples of these time differences.")
            elif next_version == 25:
                update_timestructure(db)
                db.add_update_item("parameter_definition", entity_class_name= "model", name= "periods_available", parameter_type_list = ("array",), description = "(Optional) Array of periods available for the model. Periods that are in the data, but are not in period_timeset.")
                db.add_update_item("parameter_definition", entity_class_name= "solve", name= "contains_solves", parameter_type_list = ("str","array"), description = "Array of solves - used for nested solve sequencesArray of solves - used for nested solve sequences")
            elif next_version == 26:
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
            elif next_version == 27:
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
            elif next_version == 28:
                parameter_definitions = db.mapped_table("parameter_definition")
                # Rename entity-level interest_rate -> discount_rate on unit, connection, node
                param = db.item(parameter_definitions, entity_class_name="unit", name="interest_rate")
                if param:
                    db.update_parameter_definition(id=param["id"], name="discount_rate",
                        description="[e.g. 0.05 equals 5%] Discount rate for investments (WACC). Reflects the financing cost and risk premium for this technology. When the model inflation_rate > 0, this should be a nominal rate. When inflation_rate = 0, this should be a real rate. Used to annualize investment costs over the lifetime. Constant or period.")
                param = db.item(parameter_definitions, entity_class_name="connection", name="interest_rate")
                if param:
                    db.update_parameter_definition(id=param["id"], name="discount_rate",
                        description="[e.g. 0.05 equals 5%] Discount rate for investments (WACC). Reflects the financing cost and risk premium for this technology. When the model inflation_rate > 0, this should be a nominal rate. When inflation_rate = 0, this should be a real rate. Used to annualize investment costs over the lifetime. Constant or period.")
                param = db.item(parameter_definitions, entity_class_name="node", name="interest_rate")
                if param:
                    db.update_parameter_definition(id=param["id"], name="discount_rate",
                        description="[e.g. 0.05 equals 5%] Discount rate for investments (WACC). Reflects the financing cost and risk premium for this technology. When the model inflation_rate > 0, this should be a nominal rate. When inflation_rate = 0, this should be a real rate. Used to annualize investment costs over the lifetime. Constant or period.")
                # Rename model-level discount_rate -> inflation_rate
                param = db.item(parameter_definitions, entity_class_name="model", name="discount_rate")
                if param:
                    db.update_parameter_definition(id=param["id"], name="inflation_rate",
                        description="[e.g. 0.02 for 2%] Model-wide inflation rate applied to all future costs. When inputs are in real (constant-price) terms, set to 0. When inputs are in nominal terms, set to expected inflation. Default: 0 (real inputs).")
                # Rename model-level offset parameters
                param = db.item(parameter_definitions, entity_class_name="model", name="discount_offset_investment")
                if param:
                    db.update_parameter_definition(id=param["id"], name="inflation_offset_investment",
                        description="[years] Offset for when investment costs occur within a year. Default 0 (beginning of year).")
                param = db.item(parameter_definitions, entity_class_name="model", name="discount_offset_operations")
                if param:
                    db.update_parameter_definition(id=param["id"], name="inflation_offset_operations",
                        description="[years] Offset for when operational costs occur within a year. Default 0.5 (middle of year).")
                _commit_step(db,"Renamed economic parameters: interest_rate->discount_rate, discount_rate->inflation_rate")
            elif next_version == 29:
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
            elif next_version == 30:
                add_value_list_manual(db, [
                    ["storage_binding_methods", "bind_using_blended_weights"],
                ])
            elif next_version == 31:
                add_value_list_manual(db, [
                    ["storage_binding_methods", "bind_intraperiod_blocks"],
                ])
            elif next_version == 32:
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
                    db.add_update_item("parameter_definition",
                        entity_class_name=cls,
                        name="constraint_cumulative_pre_built_capacity_coefficient",
                        description=prebuilt_desc)
                _commit_step(db,
                    "Renamed constraint_capacity_coefficient → "
                    "constraint_invested_capacity_coefficient; added "
                    "constraint_cumulative_pre_built_capacity_coefficient")
            elif next_version == 33:
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
            elif next_version == 34:
                # New lifetime_method 'no_investment': asset retires after
                # lifetime (like reinvest_choice) but no further v_invest is
                # allowed once the first-period lifetime window has elapsed —
                # a one-shot investment, no rebuild. Motivating case: life-
                # extension refurbishments that cannot be physically repeated.
                add_value_list_manual(db, [
                    ["lifetime_methods", "no_investment"],
                ])
            elif next_version == 35:
                # Split the old `coefficient` parameter on unit__inputNode /
                # unit__outputNode into three separate parameters:
                #   - flow_coefficient: energy-unit conversion for the flow
                #     in node-balance and conversion_indirect equations
                #     (renamed from 'coefficient', same semantics).
                #   - max_capacity_coefficient: scales the per-edge upper cap
                #     (maxToSink / maxToSource / maxFromSource / ramp-up).
                #     Default 1.0.
                #   - min_capacity_coefficient: scales the per-edge lower cap
                #     (minToSink_minload / minFromSource_minload / min-load
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
                parameter_definitions = db.mapped_table("parameter_definition")
                default_one_val, default_one_type = to_database(1.0)
                for cls in ("unit__inputNode", "unit__outputNode"):
                    param = db.item(parameter_definitions,
                                    entity_class_name=cls, name="coefficient")
                    if param:
                        db.update_parameter_definition(
                            id=param["id"],
                            name="flow_coefficient",
                            description=flow_desc)
                    db.add_update_item("parameter_definition",
                        entity_class_name=cls,
                        name="max_capacity_coefficient",
                        default_value=default_one_val,
                        default_type=default_one_type,
                        description=maxcap_desc)
                    db.add_update_item("parameter_definition",
                        entity_class_name=cls,
                        name="min_capacity_coefficient",
                        default_value=default_one_val,
                        default_type=default_one_type,
                        description=mincap_desc)
                _commit_step(db,
                    "Renamed coefficient → flow_coefficient; added "
                    "max_capacity_coefficient and min_capacity_coefficient "
                    "on unit__inputNode and unit__outputNode")
            elif next_version == 36:
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
                parameter_values = db.mapped_table("parameter_value")
                for cls in ("unit__outputNode", "unit__inputNode"):
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
                        "entities where flow_coefficient ≠ 1.0")
                except SpineDBAPIError:
                    pass
            elif next_version == 37:
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
                parameter_values = db.mapped_table("parameter_value")
                for pv in list(db.find_parameter_values(
                        entity_class_name="unit__outputNode",
                        parameter_definition_name="flow_coefficient")):
                    try:
                        val = float(pv["parsed_value"])
                    except (TypeError, ValueError):
                        continue
                    if val == 0.0 or val == 1.0:
                        continue
                    inv = 1.0 / val
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
            elif next_version == 38:
                # Consolidate has_balance / has_storage / node_type (which
                # previously carried the single value 'balance_within_period')
                # into one node_type parameter with four allowed values:
                # 'commodity', 'balance', 'storage', 'balance_within_period'.
                # Hard cut — the two yes/no flags are dropped.  See
                # rivendell/PLAN_node_type_consolidation.md.

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
                # don't have an explicit node_type in a given alternative.
                # The new schema default is 'balance', so without this step
                # those nodes would silently gain a balance constraint.
                # Strategy: for every (node, alternative) pair where the
                # node's entity_alternative makes it active AND no
                # node_type value yet exists for that pair, write
                # node_type='commodity'.  Pairs where has_balance=yes or
                # has_storage=yes were set (and therefore got balance /
                # storage written above) are excluded via the running
                # ``written_keys`` set.
                written_keys = set(all_keys_with_flags)  # everything migrated above
                written_keys |= existing_node_type_keys  # pre-existing bwp entries
                commodity_val, commodity_type = to_database("commodity")
                # Collect (node, alt) activations from entity_alternative rows.
                node_active_by_alt: dict[tuple, set[str]] = {}
                for ea in db.find_entity_alternatives(entity_class_name="node"):
                    if not ea.get("active", True):
                        continue
                    node_active_by_alt.setdefault(
                        ea["entity_byname"], set()
                    ).add(ea["alternative_name"])
                # Also cover Base for every node, as the catch-all default.
                # (Spine scenarios don't all include Base, but many do, and
                # even when they don't, the entity_alternative loop above
                # covers the alts they DO include.)
                all_node_entities = [
                    ent["entity_byname"]
                    for ent in db.find_entities(entity_class_name="node")
                ]
                for byname in all_node_entities:
                    alts = node_active_by_alt.get(byname, set()) | {"Base"}
                    for alt in alts:
                        if (byname, alt) in written_keys:
                            continue
                        db.add_update_item(
                            "parameter_value",
                            entity_class_name="node",
                            entity_byname=byname,
                            parameter_definition_name="node_type",
                            alternative_name=alt,
                            value=commodity_val, type=commodity_type,
                        )
                        written_keys.add((byname, alt))

                # 4. Remove the old has_balance / has_storage parameter
                #    definitions and their single-entry value lists.
                remove_parameters_manual(db, [
                    ["node", "has_balance"],
                    ["node", "has_storage"],
                ])
                for vl_name in ("has_balance", "has_storage"):
                    vl = db.item(
                        db.mapped_table("parameter_value_list"), name=vl_name,
                    )
                    if vl:
                        db.remove_items("parameter_value_list", vl["id"])

                try:
                    _commit_step(db,
                        "v38: consolidated has_balance, has_storage and "
                        "node_type ('balance_within_period') into a single "
                        "node_type parameter with four values"
                    )
                except SpineDBAPIError:
                    pass
            elif next_version == 39:
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
            elif next_version == 40:
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
            elif next_version == 41:
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
                    db.remove_items(
                        "parameter_definition", typo_defs[0]["id"]
                    )

                try:
                    _commit_step(db,
                        "v41: fix storate_state_end typo -> storage_state_end"
                    )
                except SpineDBAPIError:
                    pass
            elif next_version == 42:
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
                #    are minimal here; Agent 3 will enrich them.
                db.add_update_item(
                    "parameter_definition",
                    entity_class_name="group",
                    name="output_nodeGroup_dispatch",
                    parameter_value_list_name="yes_no",
                    parameter_type_list=("str",),
                    description=(
                        "Creates the timewise flow output for this node "
                        "group (node-group dispatch table). Renamed from "
                        "output_node_flows."
                    ),
                )
                db.add_update_item(
                    "parameter_definition",
                    entity_class_name="group",
                    name="flow_aggregator",
                    parameter_value_list_name="yes_no",
                    parameter_type_list=("str",),
                    description=(
                        "Used with group_unit_node or group_connection_node "
                        "to combine the flows when producing the dispatch "
                        "output of a node group. Renamed from "
                        "output_aggregate_flows."
                    ),
                )
                db.add_update_item(
                    "parameter_definition",
                    entity_class_name="group",
                    name="output_nodeGroup_indicators",
                    parameter_value_list_name="yes_no",
                    parameter_type_list=("str",),
                    description=(
                        "Flag to output node-group indicator results for "
                        "groups whose members are nodes (group__node)."
                    ),
                )
                db.add_update_item(
                    "parameter_definition",
                    entity_class_name="group",
                    name="output_flowGroup_indicators",
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
            elif next_version == 43:
                # Parameter-group metadata foothold.  Create an "Outputs"
                # parameter_group and tag the four group-level output
                # parameters with it.  Spine's parameter_definition table
                # carries an optional parameter_group_name slot (the 6th
                # slot in the export 6-tuple) that FlexTool has never
                # populated.  This migration is a deliberately narrow
                # foothold: future parameter additions should categorise
                # themselves using the same mechanism (see
                # docs/reference.md, "Parameter groups (metadata)").
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
                    db.add_update_item(
                        "parameter_definition",
                        entity_class_name="group",
                        name=param_name,
                        parameter_group_name="Outputs",
                    )
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
            elif next_version == 44:
                _migrate_v44_parameter_groups(db)
            elif next_version == 45:
                _migrate_v45_parameter_group_colors(db)
            elif next_version == 46:
                _migrate_v46_use_row_scaling(db)
            elif next_version == 47:
                # Register "unidirectional" as a first-class connection
                # transfer_method.  It maps to method_1way_1var_off — a
                # single non-negative flow variable, no reverse-direction
                # variable.  This covers the common "one-way transmission"
                # use case that previously forced users to work around
                # with a unit instead of a connection.
                #
                # Schema touch points:
                #
                # 1. transfer_methods value list (attached to
                #    connection.transfer_method) — the per-connection
                #    dropdown must offer "unidirectional".
                # 2. transfer_methods_group value list (attached to
                #    group.transfer_method) — the group-level override
                #    must also accept "unidirectional".
                # 3. Description refreshes on both parameter definitions
                #    so the dropdown tooltip advertises the new option.
                add_value_list_manual(db, [
                    ["transfer_methods", "unidirectional"],
                    ["transfer_methods_group", "unidirectional"],
                ])
                db.add_update_item(
                    "parameter_definition",
                    entity_class_name="connection",
                    name="transfer_method",
                    description=(
                        "Choice of transfer method. Options: regular (default), "
                        "no_losses_no_variable_cost, variable_cost_only, exact, "
                        "unidirectional. 'unidirectional' restricts flow to "
                        "source → sink only (single non-negative variable); "
                        "'regular'/'exact'/'variable_cost_only' are bidirectional "
                        "two-variable variants; 'no_losses_no_variable_cost' is a "
                        "single signed variable with no losses and no VOM."
                    ),
                )
                db.add_update_item(
                    "parameter_definition",
                    entity_class_name="group",
                    name="transfer_method",
                    description=(
                        "Override transfer_method for all connections within "
                        "this nodeGroup. Options: use_connection_transfer_methods "
                        "(default, no override), no_losses_no_variable_cost, "
                        "regular, exact, variable_cost_only, unidirectional, "
                        "dc_power_flow_with_angles. Setting 'unidirectional' "
                        "gates every member connection to one-way flow (source "
                        "→ sink only). Setting 'dc_power_flow_with_angles' "
                        "uses B-theta DC power flow (requires reactance on "
                        "connections)."
                    ),
                )
                try:
                    _commit_step(db,
                        "Added 'unidirectional' to transfer_methods and "
                        "transfer_methods_group value lists"
                    )
                except SpineDBAPIError:
                    pass
            elif next_version == 48:
                # Restore the 0.05 default on entity-level discount_rate.
                # Migration v28 renamed interest_rate → discount_rate but
                # did not preserve the default, so DBs without an explicit
                # discount_rate value fell through to 0 and produced a
                # 0 / 0 divide inside the annuity factor
                # r / (1 − 1/(1+r)^n) in flextool.mod.
                default_val, default_type = to_database(0.05)
                for cls in ("connection", "node", "unit"):
                    db.add_update_item(
                        "parameter_definition",
                        entity_class_name=cls,
                        name="discount_rate",
                        default_value=default_val,
                        default_type=default_type,
                    )
                try:
                    _commit_step(db,
                        "v48: set default_value=0.05 on "
                        "connection/node/unit discount_rate "
                        "(fixes 0/0 in annuity factor when unset)"
                    )
                except SpineDBAPIError:
                    pass
            elif next_version == 49:
                # Enum defaults that were never set on their parameter
                # definitions.  Each is a method/sense whose absence puts
                # the entity into a silently-wrong state: a constraint
                # with no sense is never enforced, a node with no
                # inflow_method defaults back to use_original inside the
                # model anyway, and a reserve without reserve_method has
                # all its method-based branches turned off.  All values
                # are already in the corresponding parameter_value_list.
                for (cls, param, value) in (
                    ("node", "inflow_method", "use_original"),
                    ("constraint", "sense", "equal"),
                    ("reserve__upDown__group", "reserve_method", "no_reserve"),
                ):
                    dv, dt = to_database(value)
                    db.add_update_item(
                        "parameter_definition",
                        entity_class_name=cls,
                        name=param,
                        default_value=dv,
                        default_type=dt,
                    )
                try:
                    _commit_step(db,
                        "v49: restore advisory enum defaults — "
                        "node.inflow_method=use_original; "
                        "constraint.sense=equal; "
                        "reserve__upDown__group.reserve_method=no_reserve"
                    )
                except SpineDBAPIError:
                    pass
            elif next_version == 50:
                # Move new_stepduration from entity_class `timeset` to
                # entity_class `solve`.
                #
                # Pre-v50 semantics: new_stepduration was declared per
                # timeset, but FlexTool rejected any config where two
                # timesets used by the same solve carried different
                # new_stepduration values (see the "More than one
                # timeline in the solve or the same timeline with
                # different step durations in different timesets"
                # check in timeline_config.py).  So the parameter was
                # effectively solve-scoped already — this migration
                # makes the scope explicit.
                _migrate_v50_new_stepduration_to_solve(db)
            elif next_version == 51:
                # Group-level temporal resolution + decomposition schema
                # (Agent 1.1).  Adds two new parameters on entity_class
                # `group`:
                #
                # * ``new_stepduration`` — float, default None.  When set,
                #   members of this group (nodes / units / connections)
                #   operate at this stepduration, overriding the
                #   solve-level new_stepduration for those entities.
                #   Enables e.g. hourly electricity nodes alongside daily
                #   H2 nodes in the same solve.
                # * ``decomposition_method`` — enum string, default
                #   "none".  Reserved for Agent 3.2 (Lagrangian region
                #   decomposition); only the schema lands here.
                #
                # Also adds the ``decomposition_methods`` value list.
                _migrate_v51_group_block_resolution(db)
            elif next_version == 52:
                # Multi-solver dispatch (Phase 1 of the polar-high
                # multi-solver handoff).  Adds seven new solver-selection
                # parameters on the ``solve`` entity so a user can choose
                # a solver per-solve (HiGHS stays default; Gurobi /
                # CPLEX / Xpress / COPT become opt-in via polar-high).
                # No LP / engine behaviour changes here — schema only.
                _migrate_v52_solver_selection(db)
            elif next_version == 53:
                # Storage-binding single-valued migration, Phase 1.
                # Wires the existing ``storage_binding_methods``
                # parameter_value_list to the ``node.storage_binding_method``
                # parameter_definition so Spine UI enforces the
                # enumeration and rejects free-form strings at edit
                # time.  The value-list members themselves were added in
                # v30 / v31 / ``update_timestructure``; this step only
                # closes the schema-level wiring.
                _migrate_v53_storage_binding_value_list(db)
            elif next_version == 54:
                # Storage-binding single-valued migration, Phase 2.
                # Ports existing array-valued
                # ``node.storage_binding_method`` rows onto the new
                # single-string contract.  For each array, the highest
                # priority element present is picked (see
                # ``_STORAGE_BINDING_PRIORITY`` inside the step).  Rows
                # that are already scalar strings are left untouched.
                # Arrays whose contents are entirely outside the known
                # method set raise ``SpineDBAPIError`` naming the entity
                # — surface, don't guess.
                _migrate_v54_storage_binding_arrays_to_scalar(db)
            elif next_version == 55:
                # Storage-binding restructure Phase A — rename the
                # three legacy method names that v53/v54 left in the
                # value_list to their clean-set counterparts, drop the
                # legacy members, and add the four new members the
                # restructure introduces (two of which gain their
                # constraint implementations only in later phases).
                # Touches both parameter_value rows and the
                # ``storage_binding_methods`` parameter_value_list.
                _migrate_v55_storage_binding_rename_and_extend(db)
            elif next_version == 56:
                # Drop ``model.debug`` from the schema.  The parameter
                # was a leftover from the legacy flextoolrunner / GAMS
                # path that emitted ``input/debug.csv``; no
                # engine_polars module reads it.  Debug control now
                # lives purely on the CLI (``--debug={off,basic,full}``)
                # and in the GUI's ProjectSettings.debug_level — see
                # ``flextool/cli/cmd_run_flextool.py`` and
                # ``flextool/gui/data_models.py``.  Removes the
                # parameter_definition AND any parameter_value rows
                # carrying it.
                _migrate_v56_remove_model_debug(db)
                # Backfill the missing ``description`` field on
                # ``group.cumulative_max_capacity`` and
                # ``group.cumulative_min_capacity``.  Both
                # parameter_definitions exist in the schema since v22
                # but only ``node``/``connection``/``unit`` got their
                # descriptions populated in the v22 migration block —
                # the ``group`` rows were left with NULL/empty text.
                # The schema-template JSON already carries the canonical
                # phrasing; this brings legacy databases in line.
                _migrate_v56_add_group_cumulative_capacity_descriptions(db)
                # Clear ``default_value`` on five parameter_definition
                # rows whose schema-declared default disagrees with how
                # the engine actually consumes the parameter.  The
                # rationale per row lives in
                # ``_audit_reports/v56_default_audit.md`` and in the
                # helper's docstring.  Only ``high``-confidence rows are
                # patched here; ``medium`` / ``surface`` rows are left
                # for user review.
                _migrate_v56_fix_wrong_defaults(db)
                # Shorten the ``_coefficient`` suffix on the four user-
                # constraint coefficient parameters to ``_coeff`` across
                # every entity class that declares them
                # (connection / connection__node / node / unit /
                # unit__inputNode / unit__outputNode).  Pure name change
                # — descriptions, default values, value-list bindings
                # and engine semantics are untouched.  The schema
                # template JSON is updated in the same commit so a
                # fresh v55 init lands on the shortened names; the
                # engine, input_derivation, autoscale and export
                # modules are renamed in lock-step.
                _migrate_v56_rename_constraint_coefficient_to_coeff(db)
                # Rename ``flow_coefficient`` → ``conversion_flow_coeff``
                # on ``unit__inputNode`` / ``unit__outputNode``.  Same
                # suffix shape as the four constraint-coefficient
                # renames immediately above (``_coeff``), with the
                # ``conversion_`` prefix making the parameter's role
                # explicit: it scales the conversion of input → output
                # energy in unit dispatch (the node-balance and
                # ``conversion_indirect`` equations).  Pure name
                # change — description, default value (1.0), value-
                # list bindings and engine semantics are untouched.
                # The schema template JSON, engine_polars frame attrs,
                # CSV filename suffixes, input_derivation cl_pars,
                # autoscale quantity types, pandas accessors and the
                # docs are renamed in lock-step.
                _migrate_v56_rename_flow_coefficient_to_conversion_flow_coeff(db)
                # Rename ``max_capacity_coefficient`` →
                # ``capacity_max_coeff`` on ``unit__inputNode`` /
                # ``unit__outputNode``.  Same ``_coeff`` suffix as the
                # earlier batches; the noun ``capacity`` now leads and
                # the qualifier ``max`` follows so the parameter sorts
                # alphabetically with the other capacity-related rows
                # (``capacity``, ``capacity_existing``, ``capacity_max``
                # invest cap, …).  Pure name change — description,
                # default value (1.0), parameter_value_list and engine
                # semantics are untouched.  Engine_polars derived
                # params, autoscale quantity types, input_derivation
                # cl_pars, CSV filename suffixes
                # (``p_process_source_max_capacity_coefficient.csv`` →
                # ``p_process_source_capacity_max_coeff.csv`` and sink),
                # and docs are renamed in lock-step.
                _migrate_v56_rename_max_capacity_coefficient_to_capacity_max_coeff(db)
                # Rename ``min_capacity_coefficient`` →
                # ``capacity_min_coeff`` on ``unit__inputNode`` /
                # ``unit__outputNode``.  Mirror of the immediately
                # preceding ``capacity_max_coeff`` rename: same
                # ``_coeff`` suffix, same noun-leads-qualifier-follows
                # ordering so the parameter sorts alphabetically with
                # the other capacity-related rows
                # (``capacity_max_coeff``, ``capacity_min_coeff``, …).
                # Pure name change — description, default value (1.0),
                # parameter_value_list and engine semantics are
                # untouched.  Autoscale quantity types,
                # input_derivation cl_pars, CSV filename suffixes
                # (``p_process_source_min_capacity_coefficient.csv`` →
                # ``p_process_source_capacity_min_coeff.csv`` and sink),
                # and docs are renamed in lock-step.
                _migrate_v56_rename_min_capacity_coefficient_to_capacity_min_coeff(db)
                # Drop ``model.exclude_entity_outputs``.  The parameter
                # was a gate on the three per-period capacity dumps
                # (``unit_capacity.csv``, ``connection_capacity.csv``,
                # ``node_capacity.csv``) emitted by
                # :mod:`flextool.process_outputs.handoff_writers`.  The
                # schema's ``"yes"`` default made "exclude" silently the
                # default for every database — which inverts the
                # parameter name's apparent intent ("exclude" reads as
                # opt-in, behaved as opt-out).  Per-entity capacity rows
                # now always emit; aggregated/group outputs continue to
                # be controlled by the three ``group.output_*`` set
                # selectors (unaffected).  The gate site in
                # ``handoff_writers``, the cl_pars emitter in
                # ``input_derivation/_specs.py``, the SET_LIKE_NAMES
                # entry in ``spinedb_backend/_backend.py``, the autoscale
                # quantity-type row, the ``export_settings.yaml`` params
                # listing and the v44 parameter_group membership map are
                # all stripped in the same commit.
                _migrate_v56_remove_exclude_entity_outputs(db)
                # Drop ``model.output_node_balance_t``.  Dead toggle:
                # nothing in ``engine_polars`` reads its row from the
                # ``optional_outputs.csv`` set, and the ``enable_set``
                # check in :mod:`flextool.engine_polars._emit_per_solve`
                # only looks for ``output_horizon``.  Schema row, the
                # SET_LIKE_NAMES bookkeeping entry, the autoscale
                # quantity-type table row and the export_settings.yaml
                # params list entry are stripped in the same commit.
                _migrate_v56_remove_output_node_balance_t(db)
                # Drop ``model.output_ramp_envelope``.  Dead toggle:
                # the flag is plumbed into the ``optional_outputs.csv``
                # multi-emitter but nothing on the engine side reads
                # its row from ``enable_optional_outputs`` (only
                # ``output_horizon`` is consulted).  Schema row,
                # input_derivation cl_pars entry, SET_LIKE_NAMES entry,
                # autoscale quantity-type row and export_settings.yaml
                # params list entry are stripped in the same commit.
                _migrate_v56_remove_output_ramp_envelope(db)
                # Drop ``model.output_unit__node_flow_t``.  Dead toggle:
                # the flag was plumbed into the ``optional_outputs.csv``
                # multi-emitter but nothing on the engine side reads
                # its row from ``enable_optional_outputs`` (only
                # ``output_horizon`` is consulted).  Schema row,
                # input_derivation cl_pars entry, SET_LIKE_NAMES entry,
                # autoscale quantity-type row, export_settings.yaml
                # params list and the legacy regen_lh2_three_region.py
                # ``yes`` override are stripped in the same commit.
                _migrate_v56_remove_output_unit__node_flow_t(db)
                # Drop ``model.output_unit__node_ramp_t``.  Dead toggle:
                # plumbed into ``optional_outputs.csv`` but nothing on
                # the engine side reads its row.  Schema row,
                # input_derivation cl_pars entry, SET_LIKE_NAMES entry,
                # autoscale quantity-type row and export_settings.yaml
                # params list entry are stripped in the same commit.
                _migrate_v56_remove_output_unit__node_ramp_t(db)
                # Drop ``model.output_connection__node__node_flow_t``.
                # Dead toggle: plumbed into ``optional_outputs.csv`` but
                # nothing on the engine side reads its row.  Schema row,
                # input_derivation cl_pars entry, SET_LIKE_NAMES entry,
                # autoscale quantity-type row, export_settings.yaml
                # params list entry and the legacy
                # regen_lh2_three_region.py ``yes`` override are
                # stripped in the same commit.
                _migrate_v56_remove_output_connection__node__node_flow_t(db)
                # Drop ``model.output_connection_flow_separate``.  Last
                # of the Batch-B dead-toggle removals; same shape as the
                # other six.  After this commit the
                # ``optional_outputs.csv`` cl_pars emitter holds only
                # ``output_horizon`` — the one flag actually consumed
                # by ``_emit_per_solve``.
                _migrate_v56_remove_output_connection_flow_separate(db)
            else:
                print("Version invalid")
            next_version += 1

        if version < new_version:
            version_up = [["model", "version", new_version, None, "Contains database version information."]]
            (num,log) = import_data(db, object_parameters = version_up)
            print(database_path+ " updated to version "+ str(new_version))
            _commit_step(db,"Updated Flextool data structure to version " + str(new_version))
        else:
            print(database_path+ " already up-to-date at version "+ str(version))

def add_version(db):
    # this function adds the version information to the database if there is none

    version_up = [["model", "version", 1, None, "Contains database version information."]]
    (num,log) = import_data(db, object_parameters = version_up)
    _commit_step(db,"Added version parameter")

    return 0

def remove_parameters_manual(db,obj_param_names):
    sq_def = db.object_parameter_definition_sq
    id_list = []
    for name_list in obj_param_names:
        object_name = name_list[0]
        parameter_name = name_list[1]
        param = db.query(sq_def).filter(sq_def.c.object_class_name == object_name).filter(sq_def.c.parameter_name == parameter_name).one_or_none()
        if param != None:
            id_list.append(param.id)

    try:
        db.remove_items('parameter_definition', *id_list)
        _commit_step(db,"Removed parameters")
    except SpineDBAPIError:
        print("This removal has been done before, continuing")
    return 0

def add_parameters_manual(db,new_parameters):
    (num,log) = import_data(db, object_parameters = new_parameters)
    try:
        _commit_step(db,"Added new parameters")
    except SpineDBAPIError:
        print("These parameters have been added before, continuing") 
    return 0

def add_relationships_manual(db, new_relationships):
    (num,log) = import_data(db, relationship_parameters = new_relationships)
    try:
        _commit_step(db,"Added new parameters")
    except SpineDBAPIError:
        print("These parameters have been added before, continuing") 
    return 0

def add_value_list_manual(db, new_value_lists):
    (num,log) = import_data(db,parameter_value_lists = new_value_lists)
    try:
        _commit_step(db,"Added new parameter value lists")
    except SpineDBAPIError:
        print("These value lists have been added before, continuing")
    return 0

def add_new_parameters(db, filepath):

    #get template JSON. This can be the master or old template if conflicting migrations in between
    with open(filepath) as json_file:
        template = json.load(json_file)

    #Parameter values need to be added first! Otherwise the new value_list_name cannot be connected!
    #Add parameter_value_lists. Note that object_parameter import and value_list import work differently. Former replaces all, latter adds what is missing.
    (num,log) = import_data(db, parameter_value_lists = template["parameter_value_lists"])

    #With objective parameters, no duplicates are created. These will replace the old ones or create new. There will always be imports.
    (num,log) = import_data(db, object_parameters = template["object_parameters"])

    try:
        _commit_step(db,"Added new parameters")
    except SpineDBAPIError:
        print("These parameters have been added before, continuing") 
    return 0

def change_optional_output_type(db, filepath):

    sq= db.entity_parameter_value_sq
    sq_def = db.object_parameter_definition_sq
    enable_parameter_query = db.query(sq).filter(sq.c.object_class_name == "model").filter(sq.c.parameter_name == "enable_optional_outputs")
    disable_parameter_query = db.query(sq).filter(sq.c.object_class_name == "model").filter(sq.c.parameter_name == "disable_optional_outputs")
    enable_parameter_definition =  db.query(sq_def).filter(sq_def.c.object_class_name == "model").filter(sq_def.c.parameter_name == "enable_optional_outputs").one_or_none()
    disable_parameter_definition =  db.query(sq_def).filter(sq_def.c.object_class_name == "model").filter(sq_def.c.parameter_name == "disable_optional_outputs").one_or_none()

    paramset_enable = []
    paramset_disable = []
    for param in enable_parameter_query:
        enable_optional_outputs = from_database(param.value, param.type).values
        meta = [param.entity_class_name, param.entity_name, param.alternative_name]
        paramset_enable.append((meta,enable_optional_outputs))

    for param in disable_parameter_query:
        disable_optional_outputs = from_database(param.value, param.type).values
        meta = [param.entity_class_name, param.entity_name, param.alternative_name]
        paramset_disable.append((meta,disable_optional_outputs))

    add_new_parameters(db, filepath)

    for param in paramset_enable:
        for output_name in param[1]:
            if output_name == 'ramp_envelope':
                parameter_name = 'output_ramp_envelope'
            elif output_name == 'unit__node_ramp_t':
                parameter_name = 'output_unit__node_ramp_t'
            elif output_name == 'connection_flow_separate' or output_name == 'connection_flow_one_direction':
                parameter_name = 'output_connection_flow_separate'
            else:
                parameter_name = 'invalid'
            if parameter_name != 'invalid':
                new_output = [(param[0][0], param[0][1], parameter_name, "yes", param[0][2])]
                (num,log) = import_data(db, object_parameter_values = new_output)
    for param in paramset_disable:
        for output_name in param[1]:
            if output_name == 'unit_flow_t' or output_name == 'unit__node_flow_t':
                parameter_name = 'output_unit__node_flow_t'
            elif output_name == 'connection_flow_t' or output_name == 'connection__node__node_flow_t':
                parameter_name = 'output_connection__node__node_flow_t'
            else:
                parameter_name = 'invalid'
            if parameter_name != 'invalid':
                new_output = [(param[0][0], param[0][1], parameter_name, "no", param[0][2])]
                (num,log) = import_data(db, object_parameter_values = new_output)
    
    if enable_parameter_definition != None:
        db.remove_items('parameter_definition', *[enable_parameter_definition.id,disable_parameter_definition.id])
    try:
        _commit_step(db,"Changed optional outputs")
    except SpineDBAPIError:
        print("This change has been done before, continuing") 
    return 0

def update_timestructure(db):
   
    timeblocks__timelines = db.find_entities(entity_class_name="timeblockSet__timeline")
    block_durations = db.find_parameter_values(entity_class_name="timeblockSet", parameter_definition_name="block_duration")
    timeblock_entity_class_item = db.item(db.mapped_table("entity_class"), name="timeblockSet")
    db.update_entity_class(id = timeblock_entity_class_item["id"], name='timeset')
    db.add_parameter_definition(entity_class_name= "timeset", name= "timeline", parameter_type_list = ("str",), description = "The name of the timeline that the timeset uses. (String)")
    for block_duration in block_durations:
        timeline_found = False
        for timeblocks__timeline in timeblocks__timelines:
            if timeblocks__timeline["entity_byname"][0] == block_duration["entity_byname"][0]:
                if timeline_found:
                    print(f'More than one timeline connected to the timeblockSet {timeblocks__timeline["entity_byname"][0]}. Converting only one to timeset - timeline')
                else: 
                    value_x, type_ = to_database(timeblocks__timeline["entity_byname"][1])
                    param_table = db.mapped_table("parameter_value")
                    db.add(
                        param_table, 
                        entity_class_name="timeset", 
                        parameter_definition_name="timeline",
                        entity_byname=(timeblocks__timeline["entity_byname"][0],),
                        alternative_name=block_duration["alternative_name"],
                        value=value_x,
                        type=type_,
                    )
                    timeline_found = True
    
    t__t_entity_class_item = db.item(db.mapped_table("entity_class"), name="timeblockSet__timeline")
    db.remove_entity_class(id = t__t_entity_class_item["id"])
    timeline_duration_in_years = db.item(db.mapped_table("parameter_definition"), entity_class_name="timeline", name = "timeline_duration_in_years")
    db.remove_parameter_definition(id = timeline_duration_in_years["id"])
    #rename params or their description if timeblockSet is mentioned
    parameter_definitions = db.mapped_table("parameter_definition")
    param = db.item(parameter_definitions, entity_class_name= "solve", name = "period_timeblockSet")
    db.update_parameter_definition(id = param["id"], name = "period_timeset", description = "Map of periods with associated timesets that will be included in the solve. Index: period name, value: timeset name.")
    param = db.item(parameter_definitions, entity_class_name= "timeset", name = "block_duration")
    db.update_parameter_definition(id = param["id"], name = "timeset_duration", description = "Index: name of the the timestep that starts the timeset, value: duration of the block in timesteps")
    param = db.item(parameter_definitions, entity_class_name= "timeset", name = "new_stepduration")
    db.update_parameter_definition(id = param["id"], description = "Hours. Creates a new `timeline` from the old for this `timeset` with this timestep duration. The new timeline will sum or average the other timeseries data like `profile` and `inflow` for the new timesteps.")
    param = db.item(parameter_definitions, entity_class_name= "node", name = "storage_binding_method")
    db.update_parameter_definition(id = param["id"], description = "Choice how the storage state will be maintained over discontinuos timelines. The default value 'bind_forward_only' will bind forward over any holes in the used timeline, but will not bind end to the start. Meanwhile 'bind_between_timesets' will bind the storage end state at the end of the timeset to the beginning of the timeset. 'bind_within_period', 'bind_within_solve' and bind_within_model' will act similarly but over increasingly longer time span. Separate parameters (e.g. 'storage_state_start') can force bindings. By default, storage start state is bound to 0.")
    p_value, p_type = to_database("bind_within_timeblock") 
    param_list_value = db.item(db.mapped_table("list_value"), parameter_value_list_name = "storage_binding_methods", value = p_value, type = p_type)
    p_value, p_type = to_database("bind_within_timeset") 
    db.update_list_value(id = param_list_value["id"], value = p_value, type = p_type)

def update_parameter_types_v23(db):
    type_list = get_parameter_type_list_v23()
    for i in type_list:
        db.add_update_item("parameter_definition", entity_class_name = i[0], name = i[1], parameter_type_list = i[2])

def get_parameter_type_list_v23():
    types = [["commodity", "co2_content", ("float",)],
             ["commodity", "price", ("float","1d_map")],
             ["connection", "availability", ("float","1d_map")],
             ["connection", "constraint_capacity_coefficient", ("1d_map",)],
             ["connection", "cumulative_max_capacity", ("float","1d_map")],
             ["connection", "cumulative_min_capacity", ("float","1d_map")],
             ["connection", "efficiency", ("float","1d_map")],
             ["connection", "existing", ("float","1d_map")],
             ["connection", "fixed_cost", ("float","1d_map")],
             ["connection", "interest_rate", ("float","1d_map")],
             ["connection", "invest_cost", ("float","1d_map")],
             ["connection", "invest_max_period", ("1d_map",)],
             ["connection", "invest_max_total", ("float",)],
             ["connection", "invest_method", ("str",)],
             ["connection", "invest_min_period", ("1d_map",)],
             ["connection", "invest_min_total", ("float",)],
             ["connection", "is_DC", ("str",)],
             ["connection", "lifetime", ("float","1d_map")],
             ["connection", "lifetime_method", ("str",)],
             ["connection", "other_operational_cost", ("float","1d_map")],
             ["connection", "retire_max_period", ("1d_map",)],
             ["connection", "retire_max_total", ("float",)],
             ["connection", "retire_min_period", ("1d_map",)],
             ["connection", "retire_min_total", ("float",)],
             ["connection", "salvage_value", ("float","1d_map")],
             ["connection", "startup_cost", ("float",)],
             ["connection", "startup_method", ("str",)],
             ["connection", "transfer_method", ("str",)],
             ["connection", "virtual_unitsize", ("float",)],
             ["constraint", "constant", ("float",)],
             ["constraint", "sense", ("str",)],
             ["group", "capacity_margin", ("float","1d_map")],
             ["group", "co2_max_period", ("1d_map",)],
             ["group", "co2_max_total", ("float",)],
             ["group", "co2_method", ("str",)],
             ["group", "co2_price", ("float","1d_map")],
             ["group", "has_capacity_margin", ("str",)],
             ["group", "has_inertia", ("str",)],
             ["group", "has_non_synchronous", ("str",)],
             ["group", "include_stochastics", ("str",)],
             ["group", "inertia_limit", ("float","1d_map")],
             ["group", "invest_max_period", ("1d_map",)],
             ["group", "invest_max_total", ("float",)],
             ["group", "invest_method", ("str",)],
             ["group", "invest_min_period", ("1d_map",)],
             ["group", "invest_min_total", ("float",)],
             ["group", "max_cumulative_flow", ("float","1d_map")],
             ["group", "max_instant_flow", ("float","1d_map")],
             ["group", "min_cumulative_flow", ("float","1d_map")],
             ["group", "min_instant_flow", ("float","1d_map")],
             ["group", "non_synchronous_limit", ("float","1d_map")],
             ["group", "flow_aggregator",  ("str",)],
             ["group", "output_nodeGroup_dispatch", ("str",)],
             ["group", "output_nodeGroup_indicators", ("str",)],
             ["group", "output_flowGroup_indicators", ("str",)],
             ["group", "penalty_capacity_margin", ("float","1d_map")],
             ["group", "penalty_inertia", ("float","1d_map")],
             ["group", "penalty_non_synchronous", ("float","1d_map")],
             ["group", "share_loss_of_load", ("str",)],
             ["model", "debug", ("str",)],
             ["model", "discount_offset_investment", ("float",)],
             ["model", "discount_offset_operations", ("float",)],
             ["model", "discount_rate", ("float",)],
             ["model", "exclude_entity_outputs", ("str",)],
             ["model", "output_connection__node__node_flow_t", ("str",)],
             ["model", "output_connection_flow_separate", ("str",)],
             ["model", "output_horizon", ("str",)],
             ["model", "output_node_balance_t", ("str",)],
             ["model", "output_ramp_envelope", ("str",)],
             ["model", "output_unit__node_flow_t", ("str",)],
             ["model", "output_unit__node_ramp_t", ("str",)],
             ["model", "solves", ("array",)],
             ["model", "version", ("float",)],
             ["node", "annual_flow", ("float","1d_map")],
             ["node", "availability", ("float","1d_map","3d_map")],
             ["node", "constraint_capacity_coefficient", ("1d_map",)],
             ["node", "constraint_state_coefficient", ("1d_map",)],
             ["node", "cumulative_max_capacity", ("float","1d_map")],
             ["node", "cumulative_min_capacity", ("float","1d_map")],
             ["node", "existing", ("float","1d_map")],
             ["node", "fixed_cost", ("float","1d_map")],
             ["node", "has_balance", ("str",)],
             ["node", "has_storage", ("str",)],
             ["node", "inflow", ("float","1d_map","3d_map")],
             ["node", "inflow_method", ("str",)],
             ["node", "interest_rate", ("float","1d_map")],
             ["node", "invest_cost", ("float","1d_map")],
             ["node", "invest_forced", ("float","1d_map")],
             ["node", "invest_max_period", ("1d_map",)],
             ["node", "invest_max_total", ("float",)],
             ["node", "invest_method", ("str",)],
             ["node", "invest_min_period", ("1d_map",)],
             ["node", "invest_min_total", ("float",)],
             ["node", "lifetime", ("float","1d_map")],
             ["node", "lifetime_method", ("str",)],
             ["node", "node_type", ("str",)],
             ["node", "peak_inflow", ("float","1d_map")],
             ["node", "penalty_down", ("float","1d_map")],
             ["node", "penalty_up", ("float","1d_map")],
             ["node", "retire_max_period", ("1d_map",)],
             ["node", "retire_max_total", ("float",)],
             ["node", "retire_min_period", ("1d_map",)],
             ["node", "retire_min_total", ("float",)],
             ["node", "salvage_value",  ("float","1d_map")],
             ["node", "self_discharge_loss", ("float","1d_map")],
             ["node", "storage_binding_method", ("str",)],
             ["node", "storage_nested_fix_method", ("str",)],
             ["node", "storage_solve_horizon_method", ("str",)],
             ["node", "storage_start_end_method", ("str",)],
             ["node", "storage_state_end", ("float",)],
             ["node", "storage_state_reference_price", ("float","1d_map")],
             ["node", "storage_state_reference_value", ("float","1d_map")],
             ["node", "storage_state_start", ("float",)],
             ["node", "storate_state_end", ("float",)],
             ["node", "virtual_unitsize", ("float",)],
             ["profile", "profile", ("1d_map","3d_map")],
             ["solve", "contains_solves", ("array",)],
             ["solve", "fix_storage_periods", ("array","2d_map")],
             ["solve", "highs_method", ("str",)],
             ["solve", "highs_parallel", ("str",)],
             ["solve", "highs_presolve", ("str",)],
             ["solve", "invest_periods", ("array","2d_map")],
             ["solve", "period_timeblockSet", ("1d_map",)],
             ["solve", "realized_invest_periods", ("array","2d_map")],
             ["solve", "realized_periods", ("array","2d_map")],
             ["solve", "rolling_duration", ("float",)],
             ["solve", "rolling_solve_horizon", ("float",)],
             ["solve", "rolling_solve_jump", ("float",)],
             ["solve", "solve_mode", ("str",)],
             ["solve", "solver", ("str",)],
             ["solve", "solver_arguments", ("array",)],
             ["solve", "solver_precommand", ("str",)],
             ["solve", "stochastic_branches", ("4d_map",)],
             ["solve", "years_represented", ("1d_map",)],
             ["timeblockSet", "block_duration", ("1d_map",)],
             ["timeblockSet", "new_stepduration", ("float",)],
             ["timeline", "timeline_duration_in_years", ("float",)],
             ["timeline", "timestep_duration", ("1d_map",)],
             ["unit", "availability", ("float","1d_map","3d_map")],
             ["unit", "constraint_capacity_coefficient", ("1d_map",)],
             ["unit", "conversion_method", ("str",)],
             ["unit", "cumulative_max_capacity", ("float","1d_map")],
             ["unit", "cumulative_min_capacity", ("float","1d_map")],
             ["unit", "efficiency", ("float","1d_map","3d_map")],
             ["unit", "efficiency_at_min_load", ("float",)],
             ["unit", "existing", ("float","1d_map")],
             ["unit", "fixed_cost", ("float","1d_map")],
             ["unit", "interest_rate", ("float","1d_map")],
             ["unit", "invest_cost", ("float","1d_map")],
             ["unit", "invest_max_period", ("1d_map",)],
             ["unit", "invest_max_total", ("float",)],
             ["unit", "invest_method", ("str",)],
             ["unit", "invest_min_period", ("1d_map",)],
             ["unit", "invest_min_total", ("float",)],
             ["unit", "lifetime", ("float","1d_map")],
             ["unit", "lifetime_method", ("str",)],
             ["unit", "min_downtime", ("float",)],
             ["unit", "min_load", ("float","1d_map","3d_map")],
             ["unit", "min_uptime", ("float",)],
             ["unit", "minimum_time_method", ("str",)],
             ["unit", "retire_max_period", ("1d_map",)],
             ["unit", "retire_max_total", ("float",)],
             ["unit", "retire_min_period", ("1d_map",)],
             ["unit", "retire_min_total", ("float",)],
             ["unit", "salvage_value", ("float","1d_map")],
             ["unit", "startup_cost", ("float",)],
             ["unit", "startup_method", ("str",)],
             ["unit", "virtual_unitsize", ("float",)],
             ["connection__node", "constraint_flow_coefficient", ("1d_map",)],
             ["connection__profile", "profile_method", ("str",)],
             ["node__profile", "profile_method", ("str",)],
             ["unit__inputNode", "flow_coefficient", ("float",)],
             ["unit__inputNode", "max_capacity_coefficient", ("float",)],
             ["unit__inputNode", "min_capacity_coefficient", ("float",)],
             ["unit__inputNode", "constraint_flow_coefficient", ("1d_map",)],
             ["unit__inputNode", "inertia_constant", ("float",)],
             ["unit__inputNode", "is_non_synchronous", ("str",)],
             ["unit__inputNode", "other_operational_cost", ("float","1d_map","3d_map")],
             ["unit__inputNode", "ramp_cost", ("float",)],
             ["unit__inputNode", "ramp_method", ("str",)],
             ["unit__inputNode", "ramp_speed_down", ("float",)],
             ["unit__inputNode", "ramp_speed_up", ("float",)],
             ["unit__outputNode", "flow_coefficient", ("float",)],
             ["unit__outputNode", "max_capacity_coefficient", ("float",)],
             ["unit__outputNode", "min_capacity_coefficient", ("float",)],
             ["unit__outputNode", "constraint_flow_coefficient", ("1d_map",)],
             ["unit__outputNode", "inertia_constant", ("float",)],
             ["unit__outputNode", "is_non_synchronous", ("str",)],
             ["unit__outputNode", "other_operational_cost", ("float","1d_map","3d_map")],
             ["unit__outputNode", "ramp_cost", ("float",)],
             ["unit__outputNode", "ramp_method", ("str",)],
             ["unit__outputNode", "ramp_speed_down", ("float",)],
             ["unit__outputNode", "ramp_speed_up", ("float",)],
             ["reserve__upDown__group", "increase_reserve_ratio", ("float",)],
             ["reserve__upDown__group", "penalty_reserve", ("float",)],
             ["reserve__upDown__group", "reservation",  ("float","1d_map","3d_map")],
             ["reserve__upDown__group", "reserve_method", ("str",)],
             ["unit__node__profile", "profile_method", ("str",)],
             ["reserve__upDown__connection__node", "increase_reserve_ratio", ("float",)],
             ["reserve__upDown__connection__node", "large_failure_ratio", ("float",)],
             ["reserve__upDown__connection__node", "max_share", ("float",)],
             ["reserve__upDown__connection__node", "reliability", ("float",)],
             ["reserve__upDown__unit__node", "increase_reserve_ratio", ("str",)],
             ["reserve__upDown__unit__node", "large_failure_ratio", ("float",)],
             ["reserve__upDown__unit__node", "max_share", ("float",)],
             ["reserve__upDown__unit__node", "reliability", ("float",)]
             ]
    
    return types


# ---------------------------------------------------------------------------
# v44: full parameter_group metadata across every parameter_definition.
# ---------------------------------------------------------------------------

# Group definitions: (name, color_6hex, priority).  See
# rivendell/PROPOSAL_parameter_groups.md for the rationale behind the
# tiered priority scheme (asset physics → decision overlays → model plumbing).
_V44_PARAMETER_GROUPS: tuple[tuple[str, str, int], ...] = (
    ("basics",         "b3cde3", 10),
    ("investment",     "fdbf6f", 20),
    ("retirement",     "ffb870", 25),
    ("storage",        "cab2d6", 30),
    ("tech_advanced",  "b2df8a", 35),
    ("reserve",        "fb9a99", 40),
    ("emission",       "ccebc5", 45),
    ("network",        "80b1d3", 50),
    ("flow_limit",     "fccde5", 55),
    ("constraint",     "bc80bd", 70),
    ("model",          "d9d9d9", 80),
    ("solve_basics",   "bebada", 85),
    ("solve_advanced", "9f94c6", 87),
    ("timeline",       "ffed6f", 90),
    # "output" is kept with its existing colour (was the v43 "Outputs"
    # foothold); only its casing + priority change.
    ("output",         "a6cee3", 95),
)


def _v44_build_parameter_group_map() -> dict[tuple[str, str], str]:
    """Return a {(entity_class, parameter_name): group_name} dict.

    This is the data-driven membership table — one entry per
    parameter_definition row in the v43 master template.  See the
    proposal document for the rationale behind each assignment.
    """
    m: dict[tuple[str, str], str] = {}

    # --- basics ---------------------------------------------------------
    basics_map: dict[str, tuple[str, ...]] = {
        "commodity": (
            "price", "unitsize", "price_method",
            "price_ladder_annual", "price_ladder_cumulative",
        ),
        "connection": (
            "availability", "efficiency", "existing", "virtual_unitsize",
            "other_operational_cost", "transfer_method",
        ),
        "connection__profile": ("profile_method",),
        "node": (
            "availability", "existing", "virtual_unitsize", "annual_flow",
            "peak_inflow", "inflow", "inflow_method", "node_type",
            "penalty_up", "penalty_down",
        ),
        "node__profile": ("profile_method",),
        "profile": ("profile",),
        "unit": (
            "availability", "efficiency", "efficiency_at_min_load",
            "existing", "virtual_unitsize", "conversion_method", "min_load",
        ),
        "unit__inputNode": (
            "flow_coefficient", "max_capacity_coefficient",
            "min_capacity_coefficient", "other_operational_cost",
        ),
        "unit__outputNode": (
            "flow_coefficient", "max_capacity_coefficient",
            "min_capacity_coefficient", "other_operational_cost",
        ),
        "unit__node__profile": ("profile_method",),
    }
    for ec, params in basics_map.items():
        for p in params:
            m[(ec, p)] = "basics"

    # --- investment ----------------------------------------------------
    invest_shared = (
        "invest_cost", "invest_method", "invest_max_period",
        "invest_max_total", "invest_min_period", "invest_min_total",
        "cumulative_max_capacity", "cumulative_min_capacity",
        "lifetime", "lifetime_method", "discount_rate", "fixed_cost",
    )
    for ec in ("connection", "node", "unit"):
        for p in invest_shared:
            m[(ec, p)] = "investment"
    m[("node", "invest_forced")] = "investment"
    for p in (
        "invest_method", "invest_max_period", "invest_max_total",
        "invest_min_period", "invest_min_total",
        "cumulative_max_capacity", "cumulative_min_capacity",
        "capacity_margin", "has_capacity_margin",
        "penalty_capacity_margin",
    ):
        m[("group", p)] = "investment"

    # --- retirement ----------------------------------------------------
    for ec in ("connection", "node", "unit"):
        for p in (
            "retire_max_period", "retire_max_total",
            "retire_min_period", "retire_min_total", "salvage_value",
        ):
            m[(ec, p)] = "retirement"

    # --- storage -------------------------------------------------------
    for p in (
        "self_discharge_loss", "storage_binding_method",
        "storage_nested_fix_method", "storage_solve_horizon_method",
        "storage_start_end_method", "storage_state_start",
        "storage_state_end", "storage_state_reference_price",
        "storage_state_reference_value",
    ):
        m[("node", p)] = "storage"

    # --- tech_advanced -------------------------------------------------
    for ec in ("connection", "unit"):
        for p in ("startup_cost", "startup_method"):
            m[(ec, p)] = "tech_advanced"
    for p in ("min_uptime", "min_downtime", "minimum_time_method"):
        m[("unit", p)] = "tech_advanced"
    for ec in ("connection", "unit"):
        m[(ec, "delay")] = "tech_advanced"
    for ec in ("unit__inputNode", "unit__outputNode"):
        for p in (
            "ramp_cost", "ramp_method",
            "ramp_speed_up", "ramp_speed_down",
        ):
            m[(ec, p)] = "tech_advanced"

    # --- reserve -------------------------------------------------------
    for p in (
        "reservation", "reserve_method",
        "penalty_reserve", "increase_reserve_ratio",
    ):
        m[("reserve__upDown__group", p)] = "reserve"
    for ec in (
        "reserve__upDown__connection__node",
        "reserve__upDown__unit__node",
    ):
        for p in (
            "increase_reserve_ratio", "large_failure_ratio",
            "max_share", "reliability",
        ):
            m[(ec, p)] = "reserve"

    # --- emission ------------------------------------------------------
    m[("commodity", "co2_content")] = "emission"
    for p in ("co2_method", "co2_max_period", "co2_max_total", "co2_price"):
        m[("group", p)] = "emission"

    # --- network -------------------------------------------------------
    for p in ("is_DC", "reactance"):
        m[("connection", p)] = "network"
    for p in (
        "base_MVA", "reference_node",
        "candidate_precapacity_to_avoid_big_m", "transfer_method",
        "has_inertia", "inertia_limit", "penalty_inertia",
        "has_non_synchronous", "non_synchronous_limit",
        "penalty_non_synchronous",
    ):
        m[("group", p)] = "network"
    for ec in ("unit__inputNode", "unit__outputNode"):
        for p in ("is_non_synchronous", "inertia_constant"):
            m[(ec, p)] = "network"

    # --- flow_limit ----------------------------------------------------
    for p in (
        "max_cumulative_flow", "min_cumulative_flow",
        "max_instant_flow", "min_instant_flow", "share_loss_of_load",
    ):
        m[("group", p)] = "flow_limit"

    # --- constraint ----------------------------------------------------
    for p in ("constant", "sense"):
        m[("constraint", p)] = "constraint"
    for ec in ("connection__node", "unit__inputNode", "unit__outputNode"):
        m[(ec, "constraint_flow_coefficient")] = "constraint"
    for ec in ("connection", "node", "unit"):
        for p in (
            "constraint_invested_capacity_coefficient",
            "constraint_cumulative_pre_built_capacity_coefficient",
        ):
            m[(ec, p)] = "constraint"
    m[("node", "constraint_state_coefficient")] = "constraint"

    # --- model ---------------------------------------------------------
    for p in (
        "version", "solves", "periods_available", "inflation_rate",
        "inflation_offset_operations", "inflation_offset_investment",
        "max_flow_for_unconstrained_variables",
    ):
        m[("model", p)] = "model"
    m[("group", "include_stochastics")] = "solve_advanced"

    # --- solve_basics --------------------------------------------------
    for p in (
        "solver", "solve_mode", "period_timeset", "realized_periods",
        "invest_periods", "years_represented",
    ):
        m[("solve", p)] = "solve_basics"

    # --- solve_advanced -----------------------------------------------
    # timeline_hole_multiplier is created + tagged in v44 Step 4 (it
    # didn't exist in the schema before that step).  Listing it here too
    # would be a no-op since Step 3's loop pre-dates its existence.
    for p in (
        "solver_arguments", "solver_precommand", "highs_presolve",
        "highs_method", "highs_parallel", "rolling_duration",
        "rolling_solve_horizon", "rolling_solve_jump",
        "realized_invest_periods", "fix_storage_periods",
        "stochastic_branches", "contains_solves",
    ):
        m[("solve", p)] = "solve_advanced"

    # --- timeline ------------------------------------------------------
    m[("timeline", "timestep_duration")] = "timeline"
    for p in (
        "timeline", "timeset_duration", "new_stepduration",
    ):
        m[("timeset", p)] = "timeline"
    m[("timeset", "timeset_weights")] = "solve_advanced"

    # --- output --------------------------------------------------------
    for p in (
        "output_nodeGroup_dispatch", "output_nodeGroup_indicators",
        "output_flowGroup_indicators", "flow_aggregator",
    ):
        m[("group", p)] = "output"
    for p in (
        "debug", "exclude_entity_outputs", "output_horizon",
        "output_node_balance_t", "output_ramp_envelope",
        "output_unit__node_flow_t", "output_unit__node_ramp_t",
        "output_connection_flow_separate",
        "output_connection__node__node_flow_t",
    ):
        m[("model", p)] = "output"

    return m


def _migrate_v44_parameter_groups(db) -> None:
    """Apply full parameter_group metadata.

    Behaviour:
      1. Rename the existing ``"Outputs"`` parameter_group to ``"output"``
         (casing fix), keeping its colour ``a6cee3`` and bumping its
         priority to 95.  Rename cascades to every parameter_definition
         that references it (Spine tracks the link by id, not by name).
      2. Add the other 14 parameter_groups from
         :data:`_V44_PARAMETER_GROUPS`.
      3. Assign every parameter_definition to its group from the
         data-driven membership map built by
         :func:`_v44_build_parameter_group_map`.
      4. Define ``solve.timeline_hole_multiplier`` (float, default 1.0)
         and assign it to ``solve_advanced``.  The parameter has been
         consumed end-to-end by the writer chain
         (``input_writer._WRITE_ENTITY_PARAMETER_SPECS`` →
         ``solve_writers.write_hole_multiplier`` → ``solve_data/
         solve_hole_multiplier.csv`` → mod's ``p_hole_multiplier``)
         since well before v44, but was never declared in the schema —
         flextool relied on a Python-convention default of 1.0.  This
         step closes the gap so the schema is the single source of
         truth (matters for downstream consumers like the native engine
         that read defaults from ``parameter_definition.default_value``).
    """
    # Step 1: rename the existing Outputs group (or create it if it is
    # somehow missing) and update its priority.  Doing an id-keyed update
    # preserves the link to the four output-parameter_definitions that
    # v43 already tagged.  Use find_parameter_groups (plural) so a
    # re-iterated migration on a DB that already has "output" (post-v44
    # state, version rolled back) doesn't blow up on the lookup.
    outputs_groups = list(db.find_parameter_groups(name="Outputs"))
    outputs_group = outputs_groups[0] if outputs_groups else None
    if outputs_group is not None:
        db.update_item(
            "parameter_group",
            id=outputs_group["id"],
            name="output",
            color="a6cee3",
            priority=95,
        )
    else:
        db.add_update_item(
            "parameter_group",
            name="output",
            color="a6cee3",
            priority=95,
        )

    # Step 2: add the remaining 14 groups.  add_update_item is idempotent
    # on name, so re-running is safe.
    for name, color, priority in _V44_PARAMETER_GROUPS:
        if name == "output":
            continue  # handled in step 1
        db.add_update_item(
            "parameter_group",
            name=name,
            color=color,
            priority=priority,
        )

    # Step 3: assign every parameter_definition to its group.
    group_map = _v44_build_parameter_group_map()
    for (entity_class_name, param_name), group_name in group_map.items():
        db.add_update_item(
            "parameter_definition",
            entity_class_name=entity_class_name,
            name=param_name,
            parameter_group_name=group_name,
        )

    # Step 4: declare solve.timeline_hole_multiplier (float, default 1.0)
    # and tag it to solve_advanced.  add_update_item is idempotent on
    # (entity_class_name, name) so re-running on a DB that already has
    # the parameter (e.g. one carried in by hand) is a no-op except for
    # back-filling missing fields (default, type, group).
    thm_default_val, thm_default_type = to_database(1.0)
    db.add_update_item(
        "parameter_definition",
        entity_class_name="solve",
        name="timeline_hole_multiplier",
        parameter_type_list=("float",),
        default_value=thm_default_val,
        default_type=thm_default_type,
        parameter_group_name="solve_advanced",
        description=(
            "[unitless] Multiplier applied to the inverse-step-duration "
            "term in nodeBalance_eq and storage-binding constraints "
            "across timeline gaps (holes).  Tunes how strongly state "
            "differences are penalised across discontinuities in the "
            "timeline.  Default 1.0 mirrors the .mod default and matches "
            "pre-v44 Python-convention behaviour."
        ),
    )

    try:
        _commit_step(db,
            "v44: renamed 'Outputs' parameter_group to 'output'; added 14 "
            "new parameter_groups (basics, investment, retirement, storage, "
            "tech_advanced, reserve, emission, network, flow_limit, "
            "constraint, model, solve_basics, solve_advanced, timeline); "
            "assigned every parameter_definition to its group per "
            "rivendell/PROPOSAL_parameter_groups.md; "
            "declared solve.timeline_hole_multiplier (float, default 1.0) "
            "in solve_advanced — closes the schema gap for a parameter "
            "the writer chain has consumed for some time via a "
            "Python-convention default."
        )
    except SpineDBAPIError:
        pass


# ---------------------------------------------------------------------------
# v45: recolour parameter_groups for light/dark-theme compatibility.
# ---------------------------------------------------------------------------
#
# The v44 palette used ColorBrewer pastels (luminance ~0.7-0.85) which wash
# out on a light IDE background and read as low-contrast on a dark one.
# v45 replaces them with mid-tone colours (relative luminance ~0.25-0.55)
# in three emotional registers:
#   * Calm cool tones for the 10 groups users normally reach for.
#   * Mild warm (soft salmon / gold) for specialised overlays that engage
#     optional physics or silently conflict with dispatch — network,
#     flow_limit.
#   * Stronger warm (amber / brick / rose) for groups where misuse can
#     break a solve — tech_advanced, solve_advanced, constraint.
#
# Names + priorities are unchanged; only the colour field is updated.
_V45_GROUP_COLORS: tuple[tuple[str, str], ...] = (
    # calm tier
    ("basics",         "6fa8c7"),  # sky blue — foundational
    ("investment",     "7fb095"),  # sage green
    ("retirement",     "a3b08c"),  # muted olive
    ("storage",        "a598c7"),  # lavender
    ("reserve",        "7aaeb0"),  # teal
    ("emission",       "8cbb8a"),  # leaf green
    ("model",          "a3a3a3"),  # neutral gray
    ("solve_basics",   "8c94b0"),  # slate
    ("timeline",       "c2b870"),  # muted gold
    ("output",         "9ac2d1"),  # cyan
    # mild warn tier
    ("network",        "d9a8a0"),  # soft salmon
    ("flow_limit",     "d9bf96"),  # soft amber
    # strong warn tier
    ("tech_advanced",  "d9925c"),  # amber-orange
    ("solve_advanced", "b56f6f"),  # brick red
    ("constraint",     "a36784"),  # dusky rose
)


def _migrate_v45_parameter_group_colors(db) -> None:
    """Update the colour on every parameter_group to the v45 palette.

    Does not touch names or priorities — both are stable from v44.  Does
    not touch parameter_definition group assignments — those already point
    by id and follow the renames/updates transparently.
    """
    for name, color in _V45_GROUP_COLORS:
        item = db.item(db.mapped_table("parameter_group"), name=name)
        if item is None:
            # Shouldn't happen if v44 ran, but don't crash if a group is
            # missing — just skip.
            continue
        db.update_item(
            "parameter_group",
            id=item["id"],
            color=color,
        )
    try:
        _commit_step(db,
            "v45: updated parameter_group colours to a mid-tone palette "
            "(readable on both light and dark IDE themes); calm cool tones "
            "for the common groups, warm tones for advanced / risk-prone "
            "groups"
        )
    except SpineDBAPIError:
        pass


class FlexToolMigrationError(RuntimeError):
    """Raised when a database migration hits an unresolvable data
    inconsistency and cannot continue safely."""


def _migrate_v50_new_stepduration_to_solve(db) -> None:
    """Move ``new_stepduration`` from ``timeset`` to ``solve``.

    Rationale
    ---------
    Pre-v50 the parameter lived on ``timeset``, but FlexTool already
    rejected any configuration in which two timesets used by the same
    solve carried different ``new_stepduration`` values (see
    ``timeline_config.create_averaged_timeseries`` — "More than one
    timeline in the solve or the same timeline with different step
    durations in different timesets").  The parameter was therefore
    solve-scoped in practice; this migration makes that explicit.

    Steps
    -----
    1. Capture the description + default from the existing
       ``timeset.new_stepduration`` parameter_definition so the new one
       carries identical metadata.
    2. Add ``solve.new_stepduration`` with the captured metadata, under
       the ``timeline`` parameter_group if it exists.
    3. For every existing ``timeset.new_stepduration`` parameter_value,
       find the solves that use that timeset via their
       ``solve.period_timeset`` map and copy the value onto each such
       solve under the same alternative.  If two timesets used by the
       same solve carry different values under the same alternative,
       raise :class:`FlexToolMigrationError` — the pre-v50 runtime
       would have errored on that config too.
    4. Remove ``timeset.new_stepduration`` (cascades to its values).
    """

    parameter_definitions = db.mapped_table("parameter_definition")

    # --- Step 1: capture existing definition metadata ---------------
    # ``db.item`` raises ``SpineDBAPIError`` when the row is absent
    # (spinedb_api ≥ 0.34); the fallback branch below was written for
    # the older silently-None return, so guard with try/except.
    try:
        timeset_def = db.item(
            parameter_definitions,
            entity_class_name="timeset",
            name="new_stepduration",
        )
    except SpineDBAPIError:
        timeset_def = None
    if timeset_def is None:
        # Nothing to migrate — schema was already missing the old
        # definition (hand-edited DB).  Create the solve-level
        # definition with sensible defaults and return.
        default_val, default_type = to_database(None)
        description = (
            "Hours. Creates a new `timeline` from the old for this "
            "`solve` with this timestep duration. The new timeline "
            "will sum or average the other timeseries data like "
            "`profile` and `inflow` for the new timesteps."
        )
    else:
        default_val = timeset_def["default_value"]
        default_type = timeset_def["default_type"]
        description = (
            timeset_def.get("description")
            or "Hours. Creates a new `timeline` from the old for this "
            "`solve` with this timestep duration. The new timeline "
            "will sum or average the other timeseries data like "
            "`profile` and `inflow` for the new timesteps."
        )
        # Refresh the wording to point at "solve" now that we're moving.
        description = description.replace("for this `timeset`", "for this `solve`")

    # --- Step 2: create solve.new_stepduration definition -----------
    db.add_update_item(
        "parameter_definition",
        entity_class_name="solve",
        name="new_stepduration",
        default_value=default_val,
        default_type=default_type,
        parameter_type_list=("float",),
        description=description,
    )
    # Attach the timeline parameter_group when it exists (v44+).
    if db.item(db.mapped_table("parameter_group"), name="timeline") is not None:
        db.add_update_item(
            "parameter_definition",
            entity_class_name="solve",
            name="new_stepduration",
            parameter_group_name="timeline",
        )

    # --- Step 3: propagate values from timesets to their solves -----

    # Build timeset -> list[(solve, alternative)] from every
    # solve.period_timeset map.  A period_timeset map has index =
    # period and value = timeset_name; we only care about the set of
    # timesets each solve refers to per alternative.
    solves_by_timeset_alt: dict[tuple[str, str], set[str]] = {}
    for pv in db.find_parameter_values(
        entity_class_name="solve", parameter_definition_name="period_timeset",
    ):
        solve_name = pv["entity_byname"][0]
        alt = pv["alternative_name"]
        parsed = from_database(pv["value"], pv["type"])
        # parsed is a spinedb_api.Map: indexes=periods, values=timesets.
        try:
            timesets_in_map = list(parsed.values)
        except AttributeError:
            # Unexpected scalar — skip (no timeset reference).
            continue
        for ts in timesets_in_map:
            ts_name = str(ts)
            solves_by_timeset_alt.setdefault((ts_name, alt), set()).add(solve_name)

    # Also index period_timeset values by solve so we can propagate
    # across alternatives when a timeset value's alternative doesn't
    # match any period_timeset alternative on the same solve.
    timesets_of_solve_any_alt: dict[str, set[str]] = {}
    for (ts_name, alt), solves in solves_by_timeset_alt.items():
        for solve_name in solves:
            timesets_of_solve_any_alt.setdefault(solve_name, set()).add(ts_name)

    # Walk every existing timeset.new_stepduration value and derive
    # the corresponding solve.new_stepduration values.
    # Track assignments so we can detect conflicts.
    written: dict[tuple[str, str], tuple[bytes, str]] = {}

    for pv in list(db.find_parameter_values(
        entity_class_name="timeset",
        parameter_definition_name="new_stepduration",
    )):
        timeset_name = pv["entity_byname"][0]
        ts_alt = pv["alternative_name"]
        value = pv["value"]
        vtype = pv["type"]

        # Which solves reference this timeset?  Match by alternative
        # first; fall back to "any alternative" when the timeset value
        # is defined in an alt where the solve didn't also define its
        # period_timeset.  (Common when both live under the scenario's
        # timeline alternative only.)
        candidate_solves = solves_by_timeset_alt.get((timeset_name, ts_alt), set())
        if not candidate_solves:
            candidate_solves = {
                s for s, ts_set in timesets_of_solve_any_alt.items()
                if timeset_name in ts_set
            }

        for solve_name in candidate_solves:
            key = (solve_name, ts_alt)
            if key in written:
                prev_value, prev_type = written[key]
                if (prev_value, prev_type) != (value, vtype):
                    raise FlexToolMigrationError(
                        f"solve '{solve_name}' (alternative '{ts_alt}') "
                        f"has timesets with conflicting "
                        f"new_stepduration values.  The pre-v50 runtime "
                        f"rejected this configuration at solve time; "
                        f"the v50 migration cannot reconcile it "
                        f"automatically.  Set a single consistent value "
                        f"on all timesets used by this solve before "
                        f"re-running the migration."
                    )
                continue
            db.add_update_item(
                "parameter_value",
                entity_class_name="solve",
                entity_byname=(solve_name,),
                parameter_definition_name="new_stepduration",
                alternative_name=ts_alt,
                value=value,
                type=vtype,
            )
            written[key] = (value, vtype)

    # --- Step 4: drop timeset.new_stepduration ---------------------
    if timeset_def is not None:
        db.remove_parameter_definition(id=timeset_def["id"])

    try:
        _commit_step(db,
            "v50: move new_stepduration from timeset to solve; "
            "propagate values via solve.period_timeset; drop "
            "timeset.new_stepduration (parameter was already "
            "effectively solve-scoped, see timeline_config.py)."
        )
    except SpineDBAPIError:
        pass


def _migrate_v51_group_block_resolution(db) -> None:
    """Add group-level ``new_stepduration`` and ``decomposition_method``.

    Rationale
    ---------
    FlexTool's ``group`` abstraction already carries a variety of
    cross-entity concerns (CO2 caps, reserves, inertia, transfer method
    overrides).  Agent 1.1 introduces two new group-level parameters
    that are the entry point for the flex-temporal / decomposition
    refactor:

    * ``new_stepduration`` — hours; when set, members of the group
      (nodes / units / connections) dispatch at this stepduration,
      overriding the solve-level ``new_stepduration`` (v50) for those
      entities.  This makes mixed-resolution models possible (e.g.
      hourly power + daily hydrogen in the same solve).
    * ``decomposition_method`` — enum string on the
      ``decomposition_methods`` value list (``none``,
      ``lagrangian_region``).  Default ``none`` preserves existing
      behaviour; ``lagrangian_region`` is reserved for Agent 3.2 and
      has no LP behaviour attached yet.

    Only schema + defaults land in this migration — block derivation
    and overlap-set generation are Python-side work.
    """

    # --- Value list for decomposition_method ----------------------------
    add_value_list_manual(db, [
        ["decomposition_methods", "none"],
        ["decomposition_methods", "lagrangian_region"],
    ])

    # --- group.new_stepduration -----------------------------------------
    default_val, default_type = to_database(None)
    db.add_update_item(
        "parameter_definition",
        entity_class_name="group",
        name="new_stepduration",
        default_value=default_val,
        default_type=default_type,
        parameter_type_list=("float",),
        description=(
            "Hours. Members of this group operate at this step "
            "duration. Overrides the solve-level new_stepduration "
            "for these entities. Used for multi-resolution models "
            "where some nodes (e.g. fuel markets) need coarser "
            "dispatch resolution than others (e.g. power systems)."
        ),
    )
    # Attach the solve_advanced parameter_group when it exists (v44+).
    if db.item(db.mapped_table("parameter_group"), name="solve_advanced") is not None:
        db.add_update_item(
            "parameter_definition",
            entity_class_name="group",
            name="new_stepduration",
            parameter_group_name="solve_advanced",
        )

    # --- group.decomposition_method -------------------------------------
    default_val, default_type = to_database("none")
    db.add_update_item(
        "parameter_definition",
        entity_class_name="group",
        name="decomposition_method",
        default_value=default_val,
        default_type=default_type,
        parameter_value_list_name="decomposition_methods",
        parameter_type_list=("str",),
        description=(
            "Decomposition strategy to apply to this group. "
            "Currently supported: 'none' (no decomposition — "
            "default), 'lagrangian_region' (group is solved as an "
            "independent region with shared-commodity coupling)."
        ),
    )
    # Attach decomposition_method to the solve_advanced parameter_group
    # — the flag is experimental and nests under the same heading as
    # ``solve.use_row_scaling`` and similar opt-in features.
    if db.item(db.mapped_table("parameter_group"), name="solve_advanced") is not None:
        db.add_update_item(
            "parameter_definition",
            entity_class_name="group",
            name="decomposition_method",
            parameter_group_name="solve_advanced",
        )

    try:
        _commit_step(db,
            "v51: added group.new_stepduration and "
            "group.decomposition_method (Agent 1.1 flex-temporal + "
            "decomposition foundation); no LP behaviour yet."
        )
    except SpineDBAPIError:
        pass


def _migrate_v52_solver_selection(db) -> None:
    """Add per-solve solver-selection parameters (Phase 1 multi-solver).

    Rationale
    ---------
    FlexTool historically had a single ``solve.solver`` parameter bound
    to a small value list (``glpsol`` / ``highs`` / ``cplex``) and ran
    HiGHS via highspy in-process.  Phase 1 of the polar-high multi-
    solver handoff (see ``specs/flextool-multi-solver-handoff.md``)
    moves solver dispatch behind polar-high and exposes per-solve user
    controls.  This migration lands the schema:

    * New value lists ``solvers``, ``solver_io_apis``, ``solver_log_levels``.
    * ``solver`` re-bound to the new ``solvers`` value list (HiGHS stays
      default; commercial solvers Gurobi / CPLEX / Xpress / COPT are
      opt-in).
    * Six new parameters: ``solver_io_api``, ``solver_options`` (free-
      form map forwarded to the solver), ``solver_time_limit``,
      ``solver_mip_gap``, ``solver_threads``, ``solver_log_level``.

    All seven parameters attach to the ``solve_advanced`` parameter
    group (consistent with ``solver_precommand`` / ``solver_arguments``
    in v44).  No LP / writer behaviour changes here — engine wiring
    arrives in later phases.
    """

    # --- Migrate legacy 'glpsol' values to 'highs' --------------------
    # GLPK/glpsol was retired in Δ.22 (binary deleted, model file
    # deleted).  Any pre-existing ``solve.solver == "glpsol"`` value
    # would point at a non-functional solver and would also fail the
    # new ``solvers`` value-list check below (glpsol is no longer a
    # member).  Rewrite in place to 'highs' while the parameter is
    # still bound to the legacy ``solver`` value list (which contains
    # 'highs' as well as 'glpsol').
    highs_value, highs_type = to_database("highs")
    for pv in db.find_parameter_values(
        entity_class_name="solve",
        parameter_definition_name="solver",
    ):
        if pv["parsed_value"] == "glpsol":
            db.add_update_item(
                "parameter_value",
                entity_class_name="solve",
                entity_byname=pv["entity_byname"],
                parameter_definition_name="solver",
                alternative_name=pv["alternative_name"],
                value=highs_value,
                type=highs_type,
            )

    # --- Transform legacy ``solver`` value list into ``solvers`` ------
    # Pre-v52 ``solve.solver`` was bound to a list named ``solver``
    # with members [glpsol, highs, cplex].  v52 expands this to
    # [highs, gurobi, cplex, xpress, copt] and renames to ``solvers``.
    #
    # Update the existing list in place (rather than drop + recreate)
    # so the parameter_definition's foreign-key reference stays valid
    # and dependent parameter_value rows keep their list_value_ref
    # entries through the migration.  Spine cascades the drop to
    # dependent rows, which made the drop-and-readd path unreliable.
    pvl_table = db.mapped_table("parameter_value_list")
    legacy_solver_list = db.item(pvl_table, name="solver")
    if legacy_solver_list is not None:
        legacy_id = legacy_solver_list["id"]
        # Add missing members: gurobi, xpress, copt.
        for member in ("gurobi", "xpress", "copt"):
            v_bytes, v_type = to_database(member)
            db.add_update_item(
                "list_value",
                parameter_value_list_name="solver",  # current (pre-rename) name
                value=v_bytes,
                type=v_type,
            )
        # Remove the glpsol member.  No parameter_value rows reference
        # it after the glpsol→highs rewrite above, so the cascade-delete
        # is a no-op here.  ``db.item`` raises when no match exists,
        # so iterate via ``find_list_values`` and match by value.
        glpsol_value_bytes, _ = to_database("glpsol")
        for lv in db.find_list_values(
            parameter_value_list_name="solver",
        ):
            if lv["value"] == glpsol_value_bytes:
                db.remove_item("list_value", lv["id"])
                break
        # Rename the list to ``solvers`` (plural).  The
        # parameter_definition's FK is by id; parameter_value rows'
        # list_value_ref entries are unaffected.
        db.update_item(
            "parameter_value_list", id=legacy_id, name="solvers",
        )

    # --- Value lists ---------------------------------------------------
    # ``solvers`` either already exists from the transform above (when
    # the legacy ``solver`` list was renamed) OR needs to be created
    # from scratch (fresh DB / no pre-v52 history).  ``add_value_list_manual``
    # is idempotent on member names so we can call it either way.
    add_value_list_manual(db, [
        ["solvers", "highs"],
        ["solvers", "gurobi"],
        ["solvers", "cplex"],
        ["solvers", "xpress"],
        ["solvers", "copt"],
        ["solver_io_apis", "direct"],
        ["solver_io_apis", "mps"],
        ["solver_io_apis", "lp"],
        ["solver_log_levels", "silent"],
        ["solver_log_levels", "normal"],
        ["solver_log_levels", "verbose"],
    ])

    has_solve_advanced = (
        db.item(db.mapped_table("parameter_group"), name="solve_advanced")
        is not None
    )

    # --- solve.solver (rebind to new ``solvers`` value list) -----------
    # Existing parameter from v1; only the value list and description
    # change.  ``add_update_item`` keyed on (entity_class_name, name)
    # is idempotent and preserves any user-set defaults via
    # parameter_value rows (we only touch the definition's default).
    default_val, default_type = to_database("highs")
    db.add_update_item(
        "parameter_definition",
        entity_class_name="solve",
        name="solver",
        default_value=default_val,
        default_type=default_type,
        parameter_value_list_name="solvers",
        parameter_type_list=("str",),
        description=(
            "Solver to use for this solve. One of polar-high's "
            "available_solvers ('highs', 'gurobi', 'cplex', 'xpress', "
            "'copt'). HiGHS is the bundled default; commercial solvers "
            "require a separate install and license."
        ),
    )
    if has_solve_advanced:
        db.add_update_item(
            "parameter_definition",
            entity_class_name="solve",
            name="solver",
            parameter_group_name="solve_advanced",
        )

    # --- solve.solver_io_api ------------------------------------------
    default_val, default_type = to_database("direct")
    db.add_update_item(
        "parameter_definition",
        entity_class_name="solve",
        name="solver_io_api",
        default_value=default_val,
        default_type=default_type,
        parameter_value_list_name="solver_io_apis",
        parameter_type_list=("str",),
        description=(
            "How the model is handed to the solver: 'direct' (in-process "
            "API, fastest), 'mps' or 'lp' (file fallback for solvers or "
            "environments without a direct binding)."
        ),
    )
    if has_solve_advanced:
        db.add_update_item(
            "parameter_definition",
            entity_class_name="solve",
            name="solver_io_api",
            parameter_group_name="solve_advanced",
        )

    # --- solve.solver_options -----------------------------------------
    # Free-form map of key->value options passed through to the solver.
    # Default is left unset (None); user must populate the map
    # explicitly.  This mirrors the pre-existing convention for
    # solver_arguments (array, ``parameter_type_list=("array",)``):
    # awkward defaults are avoided by leaving the slot empty.
    default_val, default_type = to_database(None)
    db.add_update_item(
        "parameter_definition",
        entity_class_name="solve",
        name="solver_options",
        default_value=default_val,
        default_type=default_type,
        # Spine encodes Map rank inline: "1d_map" means Map rank 1
        # (single index level → scalar value).  ``solver_options`` is
        # a flat key→value mapping (option_name → string/float), so
        # rank 1 is correct.  Bare "map" is invalid (Map requires
        # rank ≥ 1).
        parameter_type_list=("1d_map",),
        description=(
            "Map of solver-specific option name -> value, forwarded "
            "raw to the chosen solver. Use the convenience parameters "
            "(solver_time_limit, solver_mip_gap, solver_threads) for "
            "the common cases; this map is for anything else."
        ),
    )
    if has_solve_advanced:
        db.add_update_item(
            "parameter_definition",
            entity_class_name="solve",
            name="solver_options",
            parameter_group_name="solve_advanced",
        )

    # --- solve.solver_time_limit --------------------------------------
    default_val, default_type = to_database(None)
    db.add_update_item(
        "parameter_definition",
        entity_class_name="solve",
        name="solver_time_limit",
        default_value=default_val,
        default_type=default_type,
        parameter_type_list=("float",),
        description=(
            "Wall-clock time limit for the solver in seconds. "
            "Normalised across solvers by polar-high. None = no limit."
        ),
    )
    if has_solve_advanced:
        db.add_update_item(
            "parameter_definition",
            entity_class_name="solve",
            name="solver_time_limit",
            parameter_group_name="solve_advanced",
        )

    # --- solve.solver_mip_gap -----------------------------------------
    default_val, default_type = to_database(None)
    db.add_update_item(
        "parameter_definition",
        entity_class_name="solve",
        name="solver_mip_gap",
        default_value=default_val,
        default_type=default_type,
        parameter_type_list=("float",),
        description=(
            "Relative MIP optimality gap at which the solver may stop "
            "early. Normalised across solvers by polar-high. None = "
            "solver default."
        ),
    )
    if has_solve_advanced:
        db.add_update_item(
            "parameter_definition",
            entity_class_name="solve",
            name="solver_mip_gap",
            parameter_group_name="solve_advanced",
        )

    # --- solve.solver_threads -----------------------------------------
    default_val, default_type = to_database(None)
    db.add_update_item(
        "parameter_definition",
        entity_class_name="solve",
        name="solver_threads",
        default_value=default_val,
        default_type=default_type,
        # Spine has no integer value type; everything numeric is
        # stored as float.  Users entering ``4`` get a float on
        # disk; the runtime coerces back to int via ``_opt_int``
        # in _solve_config.py.
        parameter_type_list=("float",),
        description=(
            "Maximum number of solver worker threads. Normalised "
            "across solvers by polar-high. None = solver default "
            "(usually all available cores)."
        ),
    )
    if has_solve_advanced:
        db.add_update_item(
            "parameter_definition",
            entity_class_name="solve",
            name="solver_threads",
            parameter_group_name="solve_advanced",
        )

    # --- solve.solver_log_level ---------------------------------------
    default_val, default_type = to_database("normal")
    db.add_update_item(
        "parameter_definition",
        entity_class_name="solve",
        name="solver_log_level",
        default_value=default_val,
        default_type=default_type,
        parameter_value_list_name="solver_log_levels",
        parameter_type_list=("str",),
        description=(
            "Verbosity of the solver log: 'silent' (suppress), "
            "'normal' (default summary output), 'verbose' (detailed "
            "per-iteration output)."
        ),
    )
    if has_solve_advanced:
        db.add_update_item(
            "parameter_definition",
            entity_class_name="solve",
            name="solver_log_level",
            parameter_group_name="solve_advanced",
        )


    try:
        _commit_step(db,
            "v52: added solver-selection parameters on solve "
            "(solver rebound to 'solvers' value list; new "
            "solver_io_api, solver_options, solver_time_limit, "
            "solver_mip_gap, solver_threads, solver_log_level); "
            "migrated legacy 'glpsol' values to 'highs'; "
            "removed legacy 'solver' value list."
        )
    except SpineDBAPIError:
        pass


def _migrate_v46_use_row_scaling(db) -> None:
    """Add the ``use_row_scaling`` parameter on the ``solve`` entity (Agent 5).

    Default "no" preserves pre-Agent-5 behaviour exactly; setting it to
    "yes" on a solve makes the AMPL model derive
    ``node_capacity_for_scaling`` / ``group_capacity_for_scaling`` from
    connected-unit ``unitsize`` (rounded to the nearest power of 10) so
    matrix coefficients stay in a narrower band.
    """
    add_parameters_manual(db, [[
        "solve",
        "use_row_scaling",
        "no",
        "yes_no",
        "Enable automatic row scaling (experimental): derive "
        "node_capacity_for_scaling / group_capacity_for_scaling from "
        "connected-unit unitsizes (rounded to nearest power of 10) so "
        "HiGHS sees matrix coefficients on a narrower range.  Default "
        "'no' preserves pre-scaling behaviour exactly.  See "
        "flextool/SLACK_CONVENTION.md.",
    ]])
    try:
        db.add_update_item(
            "parameter_definition",
            entity_class_name="solve",
            name="use_row_scaling",
            parameter_group_name="solve_advanced",
        )
        _commit_step(db,
            "v46: added solve.use_row_scaling parameter (Agent 5 LP-scaling): "
            "per-solve opt-in for automatic row scaling; default 'no' "
            "preserves pre-scaling behaviour."
        )
    except SpineDBAPIError:
        pass


def _migrate_v53_storage_binding_value_list(db) -> None:
    """Wire ``storage_binding_methods`` value list to
    ``node.storage_binding_method`` parameter_definition.

    Rationale
    ---------
    The ``storage_binding_methods`` parameter_value_list was first
    created in v30 (``bind_using_blended_weights``) and extended in
    v31 (``bind_intraperiod_blocks``); ``update_timestructure`` (called
    from the v22 step) further mutates one of its members from
    ``bind_within_timeblock`` to ``bind_within_timeset``.  However, the
    ``node.storage_binding_method`` parameter_definition was never
    bound to this list — its ``parameter_value_list_id`` stayed NULL.
    The practical effect was that Spine UI cheerfully accepted
    arbitrary strings (and even ``array``-typed values, see the
    H2_trade.sqlite case in
    ``_audit_reports/storage_binding_method_callsites.md`` §9) and the
    backend silently flattened the array, fueling the 2026-04
    additive-flag bug now being reverted.

    Phase 1 of the single-valued migration closes the schema gap.
    Phase 2 ports existing array-valued data onto the new contract.
    The ingestion guard in ``flextool/spinedb_backend/_backend.py``
    (see ``parameter_values``) is the runtime safety net that fires
    if a v52-or-older DB containing array values is opened directly,
    without running this migration first.
    """
    pvl_table = db.mapped_table("parameter_value_list")
    sbm_list = db.item(pvl_table, name="storage_binding_methods")
    if sbm_list is None:
        # Pre-v30 DB that somehow skipped the value-list creation —
        # surface the issue rather than silently no-op.  v30 / v31 are
        # idempotent ``add_value_list_manual`` calls so any reasonable
        # upgrade path will have populated this list before reaching
        # v53.  If it is genuinely missing we cannot wire anything; the
        # assertion below will trip and force the operator to look.
        raise SpineDBAPIError(
            "v53 migration: parameter_value_list "
            "'storage_binding_methods' not found.  Re-run the "
            "migration starting from a version <=31 so the list is "
            "created, then retry."
        )

    parameter_definitions = db.mapped_table("parameter_definition")
    param = db.item(
        parameter_definitions,
        entity_class_name="node",
        name="storage_binding_method",
    )
    if param is None:
        raise SpineDBAPIError(
            "v53 migration: parameter_definition "
            "'node.storage_binding_method' not found.  The v1 schema "
            "is expected to define it; cannot wire value_list."
        )

    db.update_parameter_definition(
        id=param["id"],
        parameter_value_list_name="storage_binding_methods",
    )

    # In-migration assertion: the wiring actually took effect.  We
    # re-read the parameter_definition row and compare its
    # parameter_value_list_id (or _name) against the list we wired.
    parameter_definitions = db.mapped_table("parameter_definition")
    param_after = db.item(
        parameter_definitions,
        entity_class_name="node",
        name="storage_binding_method",
    )
    wired_id = param_after.get("parameter_value_list_id")
    wired_name = param_after.get("parameter_value_list_name")
    if wired_id != sbm_list["id"] and wired_name != "storage_binding_methods":
        raise SpineDBAPIError(
            "v53 migration: post-write verification failed — "
            "node.storage_binding_method.parameter_value_list_id is "
            f"{wired_id!r} / name {wired_name!r}, expected list id "
            f"{sbm_list['id']!r} (name 'storage_binding_methods')."
        )

    _commit_step(
        db,
        "v53: wired storage_binding_methods value_list to "
        "node.storage_binding_method parameter_definition (Phase 1 of "
        "the single-valued storage_binding_method migration).",
    )


#: Priority order (highest first) used by the v54 migration to collapse
#: array-valued ``node.storage_binding_method`` rows down to a single
#: string.  Mirrors the audit's H2_trade.sqlite recommendation
#: (see ``_audit_reports/storage_binding_method_callsites.md`` §9):
#: when multiple methods coexist in a single array, the RP-aware
#: ``bind_using_blended_weights`` wins because it carries the most
#: state-tracking machinery; ``bind_forward_only`` loses because it is
#: the silent default and should only surface when nothing else asked
#: for richer semantics.
_STORAGE_BINDING_PRIORITY: tuple[str, ...] = (
    "bind_using_blended_weights",
    "bind_intraperiod_blocks",
    "bind_within_solve",
    "bind_within_period",
    "bind_within_timeset",
    "bind_forward_only",
)


def _migrate_v54_storage_binding_arrays_to_scalar(db) -> None:
    """Rewrite every array-valued ``node.storage_binding_method`` row
    as a single string per :data:`_STORAGE_BINDING_PRIORITY`.

    Background
    ----------
    The 2026-04 list-valued design (now being reverted) silently
    flattened array-typed ``storage_binding_method`` values into one
    row per array element, which downstream additive logic in
    ``calc_storage_vre.py`` turned into double-counted state-change
    residuals.  v52 ingestion guard (Phase 1) now rejects arrays at
    load time; this v54 step is the data-side counterpart that ports
    pre-existing databases (e.g. H2_trade.sqlite, with 15 array-valued
    entries — see ``_audit_reports/storage_binding_method_callsites.md``
    §9) onto the new single-string contract.

    Behaviour
    ---------
    For each ``parameter_value`` row of
    ``node.storage_binding_method``:

    - If ``type != "array"`` (scalar string or other): leave untouched.
    - If ``type == "array"``: pick the highest-priority element from
      :data:`_STORAGE_BINDING_PRIORITY` that appears in the array's
      values, then overwrite the row in-place with that single string.
      Preserves ``entity_id`` / ``alternative_id`` / ``entity_byname``;
      only ``value`` and ``type`` change.
    - If the array contains *only* strings that are not in the priority
      list (i.e. nothing matched), raise ``SpineDBAPIError`` naming
      the entity and the unknown contents.  We refuse to guess.

    Post-migration assertion: every remaining row for
    ``node.storage_binding_method`` is verified to have ``type == "str"``.
    """
    priority_set = set(_STORAGE_BINDING_PRIORITY)

    updated: list[tuple[tuple[str, ...], str, list, str]] = []
    for pv in list(db.find_parameter_values(
        entity_class_name="node",
        parameter_definition_name="storage_binding_method",
    )):
        if pv["type"] != "array":
            continue

        entity_byname = pv["entity_byname"]
        alt_name = pv["alternative_name"]
        parsed = pv["parsed_value"]
        try:
            members = list(parsed.values)
        except AttributeError as exc:
            raise SpineDBAPIError(
                "v54 migration: node.storage_binding_method row for "
                f"entity {entity_byname!r} alternative {alt_name!r} "
                f"has type='array' but parsed_value {parsed!r} does "
                "not expose a .values list — cannot port to scalar."
            ) from exc

        picked: str | None = None
        for candidate in _STORAGE_BINDING_PRIORITY:
            if candidate in members:
                picked = candidate
                break
        if picked is None:
            raise SpineDBAPIError(
                "v54 migration: node.storage_binding_method array for "
                f"entity {entity_byname!r} (alternative {alt_name!r}) "
                f"contains only unknown methods {members!r}.  Expected "
                "at least one of "
                f"{list(_STORAGE_BINDING_PRIORITY)!r}.  Refusing to "
                "guess; fix the source data and retry."
            )

        new_value, new_type = to_database(picked)
        db.update_parameter_value(
            id=pv["id"],
            value=new_value,
            type=new_type,
        )
        updated.append((entity_byname, alt_name, members, picked))

    # Post-write verification: every remaining row must be scalar str.
    for pv in db.find_parameter_values(
        entity_class_name="node",
        parameter_definition_name="storage_binding_method",
    ):
        if pv["type"] != "str":
            raise SpineDBAPIError(
                "v54 migration: post-write verification failed — "
                "node.storage_binding_method row for entity "
                f"{pv['entity_byname']!r} alternative "
                f"{pv['alternative_name']!r} still has type "
                f"{pv['type']!r} (expected 'str').  Migration is "
                "incomplete; aborting before commit."
            )

    if updated:
        summary_lines = [
            f"  {ent} ({alt}): {arr} -> {pick}"
            for ent, alt, arr, pick in updated
        ]
        logging.info(
            "v54 migration ported %d array-valued "
            "node.storage_binding_method rows to scalar strings:\n%s",
            len(updated),
            "\n".join(summary_lines),
        )

    _commit_step(
        db,
        "v54: ported array-valued node.storage_binding_method "
        "parameter_value rows to scalar strings (Phase 2 of the "
        "single-valued storage_binding_method migration).  "
        f"{len(updated)} row(s) rewritten.",
    )


#: Map of legacy ``node.storage_binding_method`` scalar values to the
#: clean-seven-method names introduced by the storage-binding
#: restructure (Phase A).  Used by v55 to rewrite parameter_value rows
#: in-place and to determine which value_list members to drop.
#:
#: - ``bind_within_timeset`` becomes ``bind_within_timeblock``: a timeset
#:   can contain several timeblocks; the binding always operated per
#:   block, not per set — the old name was misleading.
#: - ``bind_using_blended_weights`` becomes
#:   ``bind_within_solve_blended_weights``: makes the cycle-closure
#:   scope explicit (a solve), in line with the two new variants the
#:   restructure adds (per-period and forward-only blended weights).
#: - ``bind_within_model`` becomes ``bind_within_solve``: ``model`` was
#:   already removed from the v53/v54 value_list, but any stray
#:   parameter_value rows still using it migrate to the largest in-scope
#:   binding window we offer.
_STORAGE_BINDING_RENAMES_V55: dict[str, str] = {
    "bind_within_timeset": "bind_within_timeblock",
    "bind_using_blended_weights": "bind_within_solve_blended_weights",
    "bind_within_model": "bind_within_solve",
}

#: Members of the ``storage_binding_methods`` value_list to drop in v55
#: (the three "rename-from" names).  Removed via ``find_list_values`` +
#: ``remove_item`` because Spine's value_list operations don't
#: re-validate dependent parameter_value rows; the data-rewrite in step
#: (a) of the v55 helper has already cleared those references.
_STORAGE_BINDING_DROPPED_V55: tuple[str, ...] = (
    "bind_within_timeset",
    "bind_using_blended_weights",
    "bind_within_model",
)

#: Members to add (idempotently) to the ``storage_binding_methods``
#: value_list in v55.  Two of these (``bind_within_period_blended_weights``
#: and ``bind_forward_only_blended_weights``) gain their constraint
#: implementations only in later phases of the restructure — Phase A
#: only declares them in the value_list so the wired Spine UI can offer
#: them and so subsequent phases have a stable enum to key off.
_STORAGE_BINDING_ADDED_V55: tuple[str, ...] = (
    "bind_within_timeblock",
    "bind_within_solve_blended_weights",
    "bind_within_period_blended_weights",
    "bind_forward_only_blended_weights",
)

#: Expected exact membership of ``storage_binding_methods`` after the
#: v55 step completes.  Used by the post-migration verification block
#: to fail loudly if drops or adds didn't take effect.
_STORAGE_BINDING_EXPECTED_V55: frozenset[str] = frozenset({
    "bind_within_period",
    "bind_within_solve",
    "bind_within_timeblock",
    "bind_forward_only",
    "bind_within_solve_blended_weights",
    "bind_within_period_blended_weights",
    "bind_forward_only_blended_weights",
    "bind_intraperiod_blocks",
})

# Canonical post-v55 description for ``node.storage_binding_method``.
# Mirrors ``flextool/schemas/spinedb_schema.json``'s parameter_definition
# entry verbatim so that DBs migrated through v55 carry the same text in
# their own ``parameter_definition`` row as freshly-seeded DBs.  When the
# schema description text changes (Phase F-style scrub or later), update
# this constant in lockstep so existing-DB migration stays aligned with
# new-DB seeding.
_STORAGE_BINDING_METHOD_DESCRIPTION_V55: str = (
    "Choice how the storage state will be maintained over discontinuous "
    "timelines. Seven cycle-scope methods (the state-continuity family): "
    "'bind_within_timeblock' cycles state within each timeblock (cycle "
    "closes at block boundaries); 'bind_within_period' cycles within each "
    "FlexTool period and chains blocks inside the period; "
    "'bind_within_solve' cycles across the whole solve horizon; "
    "'bind_forward_only' (default) chains state forward across the solve "
    "with no end-to-start closure; 'bind_within_solve_blended_weights' is "
    "the representative-period variant of bind_within_solve (RP weighting "
    "+ solve-level cycle closure); 'bind_within_period_blended_weights' "
    "is the RP variant of bind_within_period (per-period RP weighting, "
    "each period closes independently); "
    "'bind_forward_only_blended_weights' is the RP variant of "
    "bind_forward_only (RP weighting, no cycle closure). One additional "
    "value, 'bind_intraperiod_blocks', is structurally an aggregation "
    "method rather than a cycle-scope method: state is held constant "
    "within each block and the block-total flow is balanced at the "
    "boundary. Silent-degrade behaviour: any '*_blended_weights' method "
    "on a node in a solve whose active timeset has no "
    "representative_period_weights is automatically downgraded to the "
    "corresponding non-RP variant for that solve, so the same storage "
    "entity can drive an RP investment solve and a chronological "
    "dispatch solve back-to-back. Separate parameters (e.g. "
    "'storage_state_start') can force additional bindings. By default, "
    "storage start state is bound to 0."
)


def _migrate_v55_storage_binding_rename_and_extend(db) -> None:
    """Rename legacy ``storage_binding_method`` scalar values to their
    clean-seven-method names and refresh the
    ``storage_binding_methods`` value_list.

    Background
    ----------
    Phase A of the storage-binding-methods restructure.  v53/v54 left
    the value_list carrying three names that the new design replaces:

    - ``bind_within_timeset``         → ``bind_within_timeblock``
    - ``bind_using_blended_weights``  → ``bind_within_solve_blended_weights``
    - ``bind_within_model``           → ``bind_within_solve`` (legacy
      ``bind_within_model`` was already dropped from the value_list in
      ``update_timestructure``-era history; this step catches any
      stray parameter_value rows still carrying the string)

    Phase A also seeds the value_list with the two upcoming
    blended-weights variants (``bind_within_period_blended_weights``
    and ``bind_forward_only_blended_weights``) whose constraint
    implementations land in Phases D and E.  Adding them now (with an
    empty implementation) lets the wired Spine UI surface the
    enumeration without further schema churn later.

    Behaviour
    ---------
    Step (a) — value_list extension.  Add the four new members
    (idempotent via ``add_value_list_manual``) BEFORE rewriting any
    parameter_value rows.  Order matters: v53 wired the value_list to
    ``node.storage_binding_method``, so Spine validates every
    ``update_parameter_value`` against current list membership.
    Writing ``bind_within_solve_blended_weights`` to a row before that
    name exists in the list would raise.

    Step (b) — data rewrite.  For each ``parameter_value`` row of
    ``node.storage_binding_method`` whose scalar string value is one of
    :data:`_STORAGE_BINDING_RENAMES_V55`, rewrite the row in-place with
    the renamed string (same entity, alternative, type).  Other values
    pass through untouched.  Array-typed rows are not expected at this
    point (v54 collapsed them to scalars and v52's ingestion guard
    rejects new arrays); the helper does not iterate them.

    Step (c) — value_list cleanup.  Drop the three rename-from members
    from the list.  Safe now that step (b) has cleared every
    parameter_value row referencing them.

    Step (d) — refresh the parameter_definition description.  The
    schema-template text in ``flextool/schemas/spinedb_schema.json``
    was rewritten in Phase F to enumerate the seven cycle-scope
    methods + ``bind_intraperiod_blocks`` (aggregation) + the
    silent-degrade behaviour.  Fresh DBs seeded from the schema get
    the new text automatically; existing DBs being migrated up to
    v55 must have their own ``parameter_definition`` row rewritten
    so the in-DB help text matches.  Mirrors
    :data:`_STORAGE_BINDING_METHOD_DESCRIPTION_V55` verbatim.

    Step (e) — in-migration verification.  Re-query
    ``node.storage_binding_method`` rows and assert no legacy string
    remains; re-query the value_list and assert exact expected
    membership.  ``SpineDBAPIError`` is raised with the offending
    entries if either check fails — surface, don't guess.
    """
    # ---- Step (a): extend the storage_binding_methods value_list ----
    # Add the four new members BEFORE the data rewrite so the v53
    # wiring on node.storage_binding_method accepts the renamed names.
    pvl_table = db.mapped_table("parameter_value_list")
    sbm_list = db.item(pvl_table, name="storage_binding_methods")
    if sbm_list is None:
        raise SpineDBAPIError(
            "v55 migration: parameter_value_list "
            "'storage_binding_methods' not found.  v30/v31/v53 are "
            "expected to have populated and wired it before this step; "
            "cannot extend a list that does not exist."
        )

    add_value_list_manual(
        db,
        [["storage_binding_methods", name]
         for name in _STORAGE_BINDING_ADDED_V55],
    )

    # ---- Step (b): rename parameter_value rows ----------------------
    renamed: list[tuple[tuple[str, ...], str, str, str]] = []
    for pv in list(db.find_parameter_values(
        entity_class_name="node",
        parameter_definition_name="storage_binding_method",
    )):
        if pv["type"] != "str":
            # v54 left every row scalar-str; anything else is a
            # surprise we refuse to mutate silently.
            raise SpineDBAPIError(
                "v55 migration: node.storage_binding_method row for "
                f"entity {pv['entity_byname']!r} alternative "
                f"{pv['alternative_name']!r} has unexpected type "
                f"{pv['type']!r} (expected 'str' post-v54).  Re-run v54 "
                "first, or fix the source row, before retrying."
            )
        old_value = pv["parsed_value"]
        new_value = _STORAGE_BINDING_RENAMES_V55.get(old_value)
        if new_value is None:
            # Pass-through value (already in the clean set).
            continue
        new_value_bytes, new_value_type = to_database(new_value)
        db.update_parameter_value(
            id=pv["id"],
            value=new_value_bytes,
            type=new_value_type,
        )
        renamed.append((
            pv["entity_byname"], pv["alternative_name"], old_value, new_value,
        ))

    # ---- Step (b) verification: no legacy names remain --------------
    legacy = set(_STORAGE_BINDING_RENAMES_V55)
    offenders_a: list[tuple[tuple[str, ...], str, str]] = []
    for pv in db.find_parameter_values(
        entity_class_name="node",
        parameter_definition_name="storage_binding_method",
    ):
        if pv["type"] == "str" and pv["parsed_value"] in legacy:
            offenders_a.append((
                pv["entity_byname"], pv["alternative_name"], pv["parsed_value"],
            ))
    if offenders_a:
        raise SpineDBAPIError(
            "v55 migration: post-rewrite verification failed — "
            "node.storage_binding_method rows still carry legacy "
            f"names: {offenders_a!r}.  Expected zero rows in "
            f"{sorted(legacy)!r}."
        )

    # ---- Step (c): drop the three rename-from value_list members ----
    # ``find_list_values`` walks the list; we match by encoded value
    # bytes (same pattern as the v52 ``glpsol`` drop a few hundred
    # lines above).
    dropped_lvs: list[str] = []
    for legacy_name in _STORAGE_BINDING_DROPPED_V55:
        legacy_bytes, _ = to_database(legacy_name)
        for lv in list(db.find_list_values(
            parameter_value_list_name="storage_binding_methods",
        )):
            if lv["value"] == legacy_bytes:
                db.remove_item("list_value", lv["id"])
                dropped_lvs.append(legacy_name)
                break

    # ---- Step (d): refresh parameter_definition description ---------
    # Mirror the Phase F rewrite of the schema-template description so
    # existing-DB migration emits the same in-DB help text as fresh-DB
    # seeding from spinedb_schema.json.
    parameter_definitions = db.mapped_table("parameter_definition")
    sbm_def = db.item(
        parameter_definitions,
        entity_class_name="node",
        name="storage_binding_method",
    )
    if sbm_def is None:
        raise SpineDBAPIError(
            "v55 migration: parameter_definition "
            "'node.storage_binding_method' not found.  v1 schema is "
            "expected to define it; cannot refresh description."
        )
    db.update_parameter_definition(
        id=sbm_def["id"],
        description=_STORAGE_BINDING_METHOD_DESCRIPTION_V55,
    )
    sbm_def_after = db.item(
        parameter_definitions,
        entity_class_name="node",
        name="storage_binding_method",
    )
    if sbm_def_after["description"] != _STORAGE_BINDING_METHOD_DESCRIPTION_V55:
        raise SpineDBAPIError(
            "v55 migration: parameter_definition description refresh "
            "did not take effect for node.storage_binding_method."
        )

    # ---- Step (e) verification: exact value_list membership ---------
    members_after = {
        from_database(lv["value"], lv["type"])
        for lv in db.find_list_values(
            parameter_value_list_name="storage_binding_methods",
        )
    }
    if members_after != _STORAGE_BINDING_EXPECTED_V55:
        missing = sorted(_STORAGE_BINDING_EXPECTED_V55 - members_after)
        extra = sorted(members_after - _STORAGE_BINDING_EXPECTED_V55)
        raise SpineDBAPIError(
            "v55 migration: post-refresh value_list membership "
            f"mismatch — missing {missing!r}, extra {extra!r}.  "
            f"Expected exactly {sorted(_STORAGE_BINDING_EXPECTED_V55)!r}; "
            f"got {sorted(members_after)!r}."
        )

    if renamed:
        summary_lines = [
            f"  {ent} ({alt}): {old} -> {new}"
            for ent, alt, old, new in renamed
        ]
        logging.info(
            "v55 migration renamed %d node.storage_binding_method "
            "rows:\n%s",
            len(renamed),
            "\n".join(summary_lines),
        )
    if dropped_lvs:
        logging.info(
            "v55 migration dropped storage_binding_methods members: %s",
            sorted(dropped_lvs),
        )

    _commit_step(
        db,
        "v55: renamed legacy node.storage_binding_method scalar values "
        "to the clean-seven-method set (bind_within_timeset -> "
        "bind_within_timeblock; bind_using_blended_weights -> "
        "bind_within_solve_blended_weights; bind_within_model -> "
        "bind_within_solve) and refreshed the storage_binding_methods "
        "value_list (dropped the three rename-from members, added "
        "bind_within_timeblock, bind_within_solve_blended_weights, "
        "bind_within_period_blended_weights, "
        "bind_forward_only_blended_weights).  Also refreshed the "
        "node.storage_binding_method parameter_definition description "
        "to match the post-Phase-F schema-template text.  Phase A of "
        "the storage-binding restructure.  "
        f"{len(renamed)} row(s) renamed.",
    )


def _migrate_v56_remove_model_debug(db) -> None:
    """Drop the ``model.debug`` parameter from the schema.

    The parameter dates back to the legacy flextoolrunner / GAMS path
    that emitted ``input/debug.csv`` for the .mod file to consume.  In
    the engine_polars rewrite (FlexTool v4) nothing reads it: the
    cl_pars emitter in :mod:`flextool.input_derivation._specs` still
    produced the CSV, but no downstream module touched the file.

    Debug-level control is now purely a runtime concern, exposed
    through:

    * ``flextool/cli/cmd_run_flextool.py`` — the tri-valued
      ``--debug={off,basic,full}`` flag, default ``off``, bare
      ``--debug`` → ``basic``.
    * ``flextool/gui/data_models.py`` — ``ProjectSettings.debug_level``
      persisted in ``settings.yaml`` and surfaced as a dropdown in the
      main window.

    Removing the DB parameter eliminates a silently-broken contract
    (any value users set was discarded) and reduces schema noise.

    Side effects: every ``parameter_value`` row referencing
    ``model.debug`` is dropped alongside the ``parameter_definition``
    when ``remove_parameters_manual`` invokes ``db.remove_items``
    (cascading delete is handled by spinedb_api).
    """
    remove_parameters_manual(db, [["model", "debug"]])


def _migrate_v56_add_group_cumulative_capacity_descriptions(db) -> None:
    """Populate the missing description text on the two
    ``group.cumulative_*_capacity`` parameter_definitions.

    Both parameters have existed in the schema since the v22 migration
    that introduced cumulative-capacity bounds across
    ``node``/``connection``/``unit``/``group``.  That migration block
    seeded ``description`` for the first three entity classes but not
    for ``group``, so any database initialised before this helper ran
    carries NULL/empty description text on those two rows.  The
    canonical phrasing already lives in
    ``flextool/schemas/spinedb_schema.json`` (and is mirrored in the
    quantity-type comments under
    ``flextool/engine_polars/autoscale/_quantity_types.py``); this
    helper brings legacy databases in line.
    """
    db.update_item(
        "parameter_definition",
        entity_class_name="group",
        name="cumulative_max_capacity",
        description=(
            "[MW or MWh] Maximum cumulative capacity for a group of "
            "entities (considers existing, invested and retired "
            "capacity). Constant or period."
        ),
    )
    db.update_item(
        "parameter_definition",
        entity_class_name="group",
        name="cumulative_min_capacity",
        description=(
            "[MW or MWh] Minimum cumulative capacity for a group of "
            "entities (considers existing, invested and retired "
            "capacity). Constant or period."
        ),
    )
    _commit_step(
        db,
        "Populated description text on group.cumulative_max_capacity "
        "and group.cumulative_min_capacity (left blank by the v22 "
        "migration that introduced them).",
    )


def _migrate_v56_fix_wrong_defaults(db) -> None:
    """Fix ``default_value`` on seven ``parameter_definition`` rows
    whose schema-declared default disagrees with how the engine actually
    consumes the parameter.

    Audit: ``_audit_reports/v56_default_audit.md``.  Five of the rows
    (the ``rows_to_clear`` tuple below) were classified ``high`` and are
    cleared to ``(None, None)`` because the engine reads them via
    ``parameter_explicit`` (schema default silently dropped) or because
    the current default is a corrupt artefact.  The remaining two rows
    were originally classified ``medium`` and have since been resolved
    by user approval (see audit "high-confidence (resolved)" section):

    * ``model.inflation_offset_investment`` — current default ``1.0``,
      patched to ``0.0`` to match the engine fallback in
      :func:`flextool.engine_polars._derived_npv._inflation_scalars`
      (line 355) and the symmetric fallback in ``_emit_period_calc.py``
      (line 295).

    * ``commodity.unitsize`` — current default ``1.0``, patched to
      ``None``.  The engine reads via
      :func:`flextool.engine_polars._direct_params.p_commodity_unitsize_from_source`
      (``_entity_scalar_explicit``) so the schema default is dropped;
      the price-ladder consumer in
      :func:`flextool.engine_polars._commodity_ladder._commodity_unitsize_param`
      substitutes ``1.0`` internally when the explicit Param is absent.
      The description is rewritten to name the gating feature
      (``commodity.price_method = price_ladder_*``) and explain the
      absent → identity behaviour.

    * ``reserve__upDown__connection__node.large_failure_ratio`` and
      ``reserve__upDown__unit__node.large_failure_ratio`` — currently
      carry an empty-string default (``""``).  The N-1 reserve consumer
      in :func:`flextool.engine_polars._emit_reserve._compute_reserve_filters`
      gates each ``(p, r, ud, n)`` on
      ``p_prn.get((..., "large_failure_ratio"), 0.0) > 0`` — any
      non-zero enables the constraint, so the contract is "absent /
      0 / null = disabled, explicit positive value = enabled".  The
      empty string is a corrupt artefact for a ``float``-typed
      parameter; the sister rows ``increase_reserve_ratio`` on the
      same two classes already carry ``null, null``.

    * ``reserve__upDown__group.penalty_reserve`` — currently ``5000.0``.
      :func:`flextool.engine_polars._direct_params.p_reserve_upDown_group_penalty_reserve_from_source`
      reads via ``parameter_explicit`` and drops the broadcast default
      on the floor (the function's own docstring claims "None default
      — explicit rows only").  Soft-reserve violations enter the
      objective as ``vq_reserve * reservation * penalty * op_factor``
      (``_reserve.py``); the schema default of 5000 misleads users
      into thinking a soft-slack term is enabled by default when in
      fact no explicit row → no penalty term emitted.

    * ``reserve__upDown__connection__node.max_share`` — currently
      ``0.0``.  The consumer
      :func:`flextool.engine_polars._direct_params._process_reserve_node_param`
      also uses ``parameter_explicit``; the sister row on
      ``reserve__upDown__unit__node`` already has ``null, null``.

    * ``node.storage_state_start`` — currently ``0.0``.
      :func:`flextool.engine_polars._direct_params.p_state_start_from_source`
      reads explicit rows only; the docstring already claims
      "Default ``None`` (schema).".  If the schema default were
      actually honoured every storage node would be force-pinned to
      state 0 at the start of each rolling solve under the (also
      default) ``fix_start`` binding — a silent LP perturbation.

    The matching schema-template rows in
    ``flextool/schemas/spinedb_schema.json`` are updated in the same
    commit so a fresh v55 init lands on the corrected contract.
    """
    rows_to_clear: tuple[tuple[str, str], ...] = (
        ("reserve__upDown__connection__node", "large_failure_ratio"),
        ("reserve__upDown__unit__node",       "large_failure_ratio"),
        ("reserve__upDown__group",            "penalty_reserve"),
        ("reserve__upDown__connection__node", "max_share"),
        ("node",                              "storage_state_start"),
    )
    for entity_class_name, name in rows_to_clear:
        db.update_item(
            "parameter_definition",
            entity_class_name=entity_class_name,
            name=name,
            default_value=None,
            default_type=None,
        )

    # model.inflation_offset_investment — engine fallback is 0.0, not
    # the schema's 1.0.  Use to_database() per CONTRIBUTING.md to
    # encode the float default safely.
    inflation_default_value, inflation_default_type = to_database(0.0)
    db.update_item(
        "parameter_definition",
        entity_class_name="model",
        name="inflation_offset_investment",
        default_value=inflation_default_value,
        default_type=inflation_default_type,
    )

    # commodity.unitsize — clear the silently-dropped 1.0 default and
    # rewrite the description to name the price-ladder gate
    # (commodity.price_method = price_ladder_annual /
    # price_ladder_cumulative) and the absent → identity semantics.
    commodity_unitsize_description = (
        "Per-commodity scaling coefficient applied to the v_trade tier "
        "variable when the commodity uses the price-ladder feature "
        "(gated by commodity.price_method = price_ladder_annual or "
        "price_ladder_cumulative).  When set, v_trade is expressed in "
        "user-MWh divided by this value; pick the unitsize so the "
        "largest tier quantity sits at O(10) in the scaled LP.  The "
        "coefficient multiplies v_trade in the commodity_ladder_balance "
        "LHS, the per-tier cap LHS, and the per-tier objective term.  "
        "When absent (the default), the price-ladder consumer "
        "substitutes 1.0 internally so v_trade is in user-MWh "
        "(identity scaling).  Ignored entirely when "
        "commodity.price_method is not a price_ladder_* value."
    )
    db.update_item(
        "parameter_definition",
        entity_class_name="commodity",
        name="unitsize",
        default_value=None,
        default_type=None,
        description=commodity_unitsize_description,
    )

    _commit_step(
        db,
        "v56 wrong-default cleanup: cleared default_value/default_type on "
        "reserve__upDown__{connection,unit}__node.large_failure_ratio, "
        "reserve__upDown__group.penalty_reserve, "
        "reserve__upDown__connection__node.max_share, "
        "node.storage_state_start, and commodity.unitsize (the latter "
        "also gets a rewritten description naming the price-ladder "
        "gate); set model.inflation_offset_investment default to 0.0 "
        "to match the engine fallback.  See "
        "_audit_reports/v56_default_audit.md.",
    )


def _migrate_v56_rename_constraint_coefficient_to_coeff(db) -> None:
    """Rename the four user-constraint ``*_coefficient`` parameters to
    ``*_coeff`` on every entity class that declares them.

    The four parameters and their per-class footprint match the
    schema-template snapshot under
    ``flextool/schemas/spinedb_schema.json``:

    * ``constraint_flow_coefficient`` →
      ``constraint_flow_coeff`` on
      ``connection__node`` / ``unit__inputNode`` / ``unit__outputNode``.
    * ``constraint_cumulative_pre_built_capacity_coefficient`` →
      ``constraint_cumulative_pre_built_capacity_coeff`` on
      ``connection`` / ``node`` / ``unit``.
    * ``constraint_invested_capacity_coefficient`` →
      ``constraint_invested_capacity_coeff`` on
      ``connection`` / ``node`` / ``unit``.
    * ``constraint_state_coefficient`` →
      ``constraint_state_coeff`` on ``node``.

    Pure name change: every other column on the
    ``parameter_definition`` row (description, default value,
    parameter_value_list, parameter_group, valid types) is preserved
    by passing ``description`` through unchanged.  Existing
    ``parameter_value`` rows that reference the old name follow the
    rename automatically because spinedb_api tracks the link by id,
    not by name.

    The engine_polars frame attributes, autoscale quantity-type
    table, input_derivation cl_pars specs, export_to_tabular
    settings, and the docs are renamed in the same commit so the
    pipeline stays internally consistent.
    """
    renames: tuple[tuple[str, str, str], ...] = (
        ("connection",       "constraint_cumulative_pre_built_capacity_coefficient",
                             "constraint_cumulative_pre_built_capacity_coeff"),
        ("connection",       "constraint_invested_capacity_coefficient",
                             "constraint_invested_capacity_coeff"),
        ("connection__node", "constraint_flow_coefficient",
                             "constraint_flow_coeff"),
        ("node",             "constraint_cumulative_pre_built_capacity_coefficient",
                             "constraint_cumulative_pre_built_capacity_coeff"),
        ("node",             "constraint_invested_capacity_coefficient",
                             "constraint_invested_capacity_coeff"),
        ("node",             "constraint_state_coefficient",
                             "constraint_state_coeff"),
        ("unit",             "constraint_cumulative_pre_built_capacity_coefficient",
                             "constraint_cumulative_pre_built_capacity_coeff"),
        ("unit",             "constraint_invested_capacity_coefficient",
                             "constraint_invested_capacity_coeff"),
        ("unit__inputNode",  "constraint_flow_coefficient",
                             "constraint_flow_coeff"),
        ("unit__outputNode", "constraint_flow_coefficient",
                             "constraint_flow_coeff"),
    )
    parameter_definitions = db.mapped_table("parameter_definition")
    for cls, old_name, new_name in renames:
        # ``db.item()`` raises ``SpineDBAPIError`` (not None) when the
        # row doesn't exist; that's the steady-state once the schema
        # template JSON has been re-synced to the renamed names and a
        # fresh DB is bootstrapped from it.  Treat "row already renamed"
        # as idempotent — the helper must be safe to re-run.
        try:
            param = db.item(parameter_definitions,
                            entity_class_name=cls, name=old_name)
        except SpineDBAPIError:
            param = None
        if param:
            db.update_parameter_definition(
                id=param["id"],
                name=new_name,
                description=param.get("description"),
            )
    _commit_step(
        db,
        "v56 rename constraint_*_coefficient → constraint_*_coeff: "
        "connection.constraint_cumulative_pre_built_capacity_coefficient → "
        "constraint_cumulative_pre_built_capacity_coeff; "
        "connection.constraint_invested_capacity_coefficient → "
        "constraint_invested_capacity_coeff; "
        "connection__node.constraint_flow_coefficient → "
        "constraint_flow_coeff; "
        "node.constraint_cumulative_pre_built_capacity_coefficient → "
        "constraint_cumulative_pre_built_capacity_coeff; "
        "node.constraint_invested_capacity_coefficient → "
        "constraint_invested_capacity_coeff; "
        "node.constraint_state_coefficient → "
        "constraint_state_coeff; "
        "unit.constraint_cumulative_pre_built_capacity_coefficient → "
        "constraint_cumulative_pre_built_capacity_coeff; "
        "unit.constraint_invested_capacity_coefficient → "
        "constraint_invested_capacity_coeff; "
        "unit__inputNode.constraint_flow_coefficient → "
        "constraint_flow_coeff; "
        "unit__outputNode.constraint_flow_coefficient → "
        "constraint_flow_coeff.",
    )


def _migrate_v56_rename_flow_coefficient_to_conversion_flow_coeff(db) -> None:
    """Rename ``flow_coefficient`` to ``conversion_flow_coeff`` on
    every entity class that declares it.

    Footprint matches the schema-template snapshot under
    ``flextool/schemas/spinedb_schema.json`` — ``unit__inputNode`` and
    ``unit__outputNode``.  The new name shares the ``_coeff`` suffix
    with the four user-constraint coefficients renamed by
    :func:`_migrate_v56_rename_constraint_coefficient_to_coeff` and
    keeps the ``conversion_`` prefix that signals the parameter's
    role: it scales the conversion of input → output energy in the
    unit dispatch / node-balance / ``conversion_indirect`` equations
    (see flextool.mod:2557-2580 and the engine_polars dispatch path
    in ``model.py`` §F.4).

    Pure name change: every other column on the
    ``parameter_definition`` row (description, default value of 1.0,
    parameter_value_list, parameter_group ``basics``, valid types) is
    preserved.  Existing ``parameter_value`` rows that reference the
    old name follow the rename automatically because spinedb_api
    tracks the link by id, not by name.

    The engine_polars frame attributes (``p_process_source_flow_coef``
    → ``p_process_source_conversion_flow_coeff`` and the sink
    counterpart), autoscale quantity-type table, input_derivation
    cl_pars specs, the CSV filename suffixes
    (``p_process_source_flow_coefficient.csv`` →
    ``p_process_source_conversion_flow_coeff.csv`` and sink), the
    pandas accessor names in ``process_outputs/read_parameters.py``,
    and the docs are renamed in the same commit so the pipeline stays
    internally consistent.
    """
    renames: tuple[tuple[str, str, str], ...] = (
        ("unit__inputNode",  "flow_coefficient", "conversion_flow_coeff"),
        ("unit__outputNode", "flow_coefficient", "conversion_flow_coeff"),
    )
    parameter_definitions = db.mapped_table("parameter_definition")
    for cls, old_name, new_name in renames:
        # ``db.item()`` raises ``SpineDBAPIError`` (not None) when the
        # row doesn't exist; that's the steady-state once the schema
        # template JSON has been re-synced to the renamed names and a
        # fresh DB is bootstrapped from it.  Treat "row already renamed"
        # as idempotent — the helper must be safe to re-run.
        try:
            param = db.item(parameter_definitions,
                            entity_class_name=cls, name=old_name)
        except SpineDBAPIError:
            param = None
        if param:
            db.update_parameter_definition(
                id=param["id"],
                name=new_name,
                description=param.get("description"),
            )
    _commit_step(
        db,
        "v56 rename flow_coefficient → conversion_flow_coeff: "
        "unit__inputNode.flow_coefficient → conversion_flow_coeff; "
        "unit__outputNode.flow_coefficient → conversion_flow_coeff.",
    )


def _migrate_v56_rename_max_capacity_coefficient_to_capacity_max_coeff(db) -> None:
    """Rename ``max_capacity_coefficient`` to ``capacity_max_coeff`` on
    every entity class that declares it.

    Footprint matches the schema-template snapshot under
    ``flextool/schemas/spinedb_schema.json`` — ``unit__inputNode`` and
    ``unit__outputNode``.  The shortened ``_coeff`` suffix aligns with
    the v56 convention introduced by
    :func:`_migrate_v56_rename_constraint_coefficient_to_coeff` and
    :func:`_migrate_v56_rename_flow_coefficient_to_conversion_flow_coeff`.
    Reordering puts the noun ``capacity`` first and the qualifier
    ``max`` second, which groups the parameter alphabetically with the
    other capacity-related parameters on these classes.

    Pure name change: every other column on the
    ``parameter_definition`` row (description, default value of 1.0,
    parameter_value_list, parameter_group ``basics``, valid types) is
    preserved.  Existing ``parameter_value`` rows that reference the
    old name follow the rename automatically because spinedb_api
    tracks the link by id, not by name.

    The engine_polars derived-param helpers
    (``_arc_max_capacity_coef_lf`` / ``_process_source_sink_coeff_zero_lf``),
    autoscale quantity-type table, input_derivation cl_pars specs,
    the CSV filename suffixes
    (``p_process_source_max_capacity_coefficient.csv`` →
    ``p_process_source_capacity_max_coeff.csv`` and sink), and the
    docs are renamed in the same commit so the pipeline stays
    internally consistent.
    """
    renames: tuple[tuple[str, str, str], ...] = (
        ("unit__inputNode",  "max_capacity_coefficient", "capacity_max_coeff"),
        ("unit__outputNode", "max_capacity_coefficient", "capacity_max_coeff"),
    )
    parameter_definitions = db.mapped_table("parameter_definition")
    for cls, old_name, new_name in renames:
        # ``db.item()`` raises ``SpineDBAPIError`` (not None) when the
        # row doesn't exist; that's the steady-state once the schema
        # template JSON has been re-synced to the renamed names and a
        # fresh DB is bootstrapped from it.  Treat "row already renamed"
        # as idempotent — the helper must be safe to re-run.
        try:
            param = db.item(parameter_definitions,
                            entity_class_name=cls, name=old_name)
        except SpineDBAPIError:
            param = None
        if param:
            db.update_parameter_definition(
                id=param["id"],
                name=new_name,
                description=param.get("description"),
            )
    _commit_step(
        db,
        "v56 rename max_capacity_coefficient → capacity_max_coeff: "
        "unit__inputNode.max_capacity_coefficient → capacity_max_coeff; "
        "unit__outputNode.max_capacity_coefficient → capacity_max_coeff.",
    )


def _migrate_v56_rename_min_capacity_coefficient_to_capacity_min_coeff(db) -> None:
    """Rename ``min_capacity_coefficient`` to ``capacity_min_coeff`` on
    every entity class that declares it.

    Footprint matches the schema-template snapshot under
    ``flextool/schemas/spinedb_schema.json`` — ``unit__inputNode`` and
    ``unit__outputNode``.  The shortened ``_coeff`` suffix aligns with
    the v56 convention introduced by
    :func:`_migrate_v56_rename_constraint_coefficient_to_coeff` and
    :func:`_migrate_v56_rename_flow_coefficient_to_conversion_flow_coeff`.
    Reordering puts the noun ``capacity`` first and the qualifier
    ``min`` second, which groups the parameter alphabetically with the
    other capacity-related parameters on these classes — directly
    after ``capacity_max_coeff`` (renamed by the sibling helper
    :func:`_migrate_v56_rename_max_capacity_coefficient_to_capacity_max_coeff`).

    Pure name change: every other column on the
    ``parameter_definition`` row (description, default value of 1.0,
    parameter_value_list, parameter_group ``basics``, valid types) is
    preserved.  Existing ``parameter_value`` rows that reference the
    old name follow the rename automatically because spinedb_api
    tracks the link by id, not by name.

    The autoscale quantity-type table, input_derivation cl_pars
    specs, the CSV filename suffixes
    (``p_process_source_min_capacity_coefficient.csv`` →
    ``p_process_source_capacity_min_coeff.csv`` and sink), and the
    docs are renamed in the same commit so the pipeline stays
    internally consistent.
    """
    renames: tuple[tuple[str, str, str], ...] = (
        ("unit__inputNode",  "min_capacity_coefficient", "capacity_min_coeff"),
        ("unit__outputNode", "min_capacity_coefficient", "capacity_min_coeff"),
    )
    parameter_definitions = db.mapped_table("parameter_definition")
    for cls, old_name, new_name in renames:
        # ``db.item()`` raises ``SpineDBAPIError`` (not None) when the
        # row doesn't exist; that's the steady-state once the schema
        # template JSON has been re-synced to the renamed names and a
        # fresh DB is bootstrapped from it.  Treat "row already renamed"
        # as idempotent — the helper must be safe to re-run.
        try:
            param = db.item(parameter_definitions,
                            entity_class_name=cls, name=old_name)
        except SpineDBAPIError:
            param = None
        if param:
            db.update_parameter_definition(
                id=param["id"],
                name=new_name,
                description=param.get("description"),
            )
    _commit_step(
        db,
        "v56 rename min_capacity_coefficient → capacity_min_coeff: "
        "unit__inputNode.min_capacity_coefficient → capacity_min_coeff; "
        "unit__outputNode.min_capacity_coefficient → capacity_min_coeff.",
    )


def _migrate_v56_remove_exclude_entity_outputs(db) -> None:
    """Drop the ``model.exclude_entity_outputs`` parameter from the schema.

    The parameter was the single gate behind
    :func:`flextool.process_outputs.handoff_writers._exclude_entity_outputs_active`
    which short-circuited the three per-period capacity dumps
    (``unit_capacity.csv``, ``connection_capacity.csv``,
    ``node_capacity.csv``) whenever its value resolved to ``"yes"``.  The
    schema's default of ``"yes"`` made "exclude" the silent default for
    every database that did not override it explicitly, which inverted
    the intent of the parameter name ("exclude" reads as an opt-in but
    behaved as an opt-out).

    The user-facing semantic is now simply "always emit per-entity
    capacity rows".  Aggregated/group-level outputs continue to be
    controlled by the three ``group.output_*`` set selectors
    (``output_nodeGroup_dispatch``, ``output_nodeGroup_indicators``,
    ``output_flowGroup_indicators``) — those are unaffected.

    The gate site in :mod:`flextool.process_outputs.handoff_writers` is
    deleted in the same commit, along with the cl_pars emitter in
    :mod:`flextool.input_derivation._specs` (which produced
    ``input/exclude_entity_outputs.csv``, the only file the gate read),
    and the bookkeeping rows in
    :data:`flextool.spinedb_backend._backend.SET_LIKE_NAMES`,
    :mod:`flextool.engine_polars.autoscale._quantity_types`,
    :mod:`flextool.export_to_tabular.export_settings` and the v44
    parameter_group membership map above.

    Side effects: every ``parameter_value`` row referencing
    ``model.exclude_entity_outputs`` is dropped alongside the
    ``parameter_definition`` when ``remove_parameters_manual`` invokes
    ``db.remove_items`` (cascading delete is handled by spinedb_api).
    """
    remove_parameters_manual(db, [["model", "exclude_entity_outputs"]])


def _migrate_v56_remove_output_node_balance_t(db) -> None:
    """Drop the ``model.output_node_balance_t`` parameter from the schema.

    Dead toggle: no module in :mod:`flextool.engine_polars` reads the
    ``optional_outputs.csv`` row for this flag and no per-flag branch
    exists in the per-solve emitter
    (:mod:`flextool.engine_polars._emit_per_solve`).  Only
    ``output_horizon`` is checked from the ``enable_optional_outputs``
    set.  The remaining cl_pars entry in
    :mod:`flextool.input_derivation._specs` (the ``optional_outputs.csv``
    multi-param emitter) already does NOT include
    ``output_node_balance_t`` — this helper just clears the schema row,
    the SET_LIKE_NAMES bookkeeping entry, the autoscale quantity-type
    table row and the ``export_settings.yaml`` params list entry.

    Side effects: every ``parameter_value`` row referencing
    ``model.output_node_balance_t`` is dropped alongside the
    ``parameter_definition`` when ``remove_parameters_manual`` invokes
    ``db.remove_items`` (cascading delete is handled by spinedb_api).
    """
    remove_parameters_manual(db, [["model", "output_node_balance_t"]])


def _migrate_v56_remove_output_ramp_envelope(db) -> None:
    """Drop the ``model.output_ramp_envelope`` parameter from the schema.

    Dead toggle: the flag IS plumbed into the multi-param
    ``optional_outputs.csv`` emitter in
    :mod:`flextool.input_derivation._specs`, but nothing on the engine
    side reads its row from the resulting ``enable_optional_outputs``
    set — only ``output_horizon`` is checked in
    :mod:`flextool.engine_polars._emit_per_solve`.  Any value users set
    has been silently dropped.

    This helper removes the schema row; sibling edits in the same
    commit strip it from the input_derivation cl_pars, the
    SET_LIKE_NAMES table, the autoscale quantity-type table and the
    export_settings.yaml params list.

    Side effects: every ``parameter_value`` row referencing
    ``model.output_ramp_envelope`` is dropped alongside the
    ``parameter_definition`` when ``remove_parameters_manual`` invokes
    ``db.remove_items``.
    """
    remove_parameters_manual(db, [["model", "output_ramp_envelope"]])


def _migrate_v56_remove_output_unit__node_flow_t(db) -> None:
    """Drop the ``model.output_unit__node_flow_t`` parameter from the schema.

    Dead toggle: the flag IS plumbed into the multi-param
    ``optional_outputs.csv`` emitter, but nothing on the engine side
    reads its row from the resulting ``enable_optional_outputs`` set —
    only ``output_horizon`` is checked.  The
    ``unit__inputNode__dt`` / ``unit__outputNode__dt`` golden CSVs
    that tests rely on are produced by the always-on
    write-handoff path, not by this gate.

    This helper removes the schema row; sibling edits in the same
    commit strip it from the input_derivation cl_pars, the
    SET_LIKE_NAMES table, the autoscale quantity-type table and the
    export_settings.yaml params list.  The legacy
    ``tests/fixtures/regen_lh2_three_region.py`` generator no longer
    appends a ``yes`` override for the parameter.

    Side effects: every ``parameter_value`` row referencing
    ``model.output_unit__node_flow_t`` is dropped alongside the
    ``parameter_definition``.
    """
    remove_parameters_manual(db, [["model", "output_unit__node_flow_t"]])


def _migrate_v56_remove_output_unit__node_ramp_t(db) -> None:
    """Drop the ``model.output_unit__node_ramp_t`` parameter from the schema.

    Dead toggle: the flag was plumbed into the multi-param
    ``optional_outputs.csv`` emitter but nothing on the engine side
    reads its row from ``enable_optional_outputs`` — only
    ``output_horizon`` is consulted.  No per-flag emission branch
    exists; any value users set was silently dropped.

    Sibling edits in the same commit strip it from the input_derivation
    cl_pars, the SET_LIKE_NAMES table, the autoscale quantity-type
    table and the export_settings.yaml params list.

    Side effects: every ``parameter_value`` row referencing
    ``model.output_unit__node_ramp_t`` is dropped alongside the
    ``parameter_definition``.
    """
    remove_parameters_manual(db, [["model", "output_unit__node_ramp_t"]])


def _migrate_v56_remove_output_connection__node__node_flow_t(db) -> None:
    """Drop the ``model.output_connection__node__node_flow_t`` parameter
    from the schema.

    Dead toggle: the flag was plumbed into the multi-param
    ``optional_outputs.csv`` emitter but nothing on the engine side
    reads its row from ``enable_optional_outputs`` — only
    ``output_horizon`` is consulted.  Any value users set was silently
    dropped.

    Sibling edits in the same commit strip it from the input_derivation
    cl_pars, the SET_LIKE_NAMES table, the autoscale quantity-type
    table, the export_settings.yaml params list and the legacy
    ``tests/fixtures/regen_lh2_three_region.py`` ``yes`` override.

    Side effects: every ``parameter_value`` row referencing
    ``model.output_connection__node__node_flow_t`` is dropped alongside
    the ``parameter_definition``.
    """
    remove_parameters_manual(db, [["model", "output_connection__node__node_flow_t"]])


def _migrate_v56_remove_output_connection_flow_separate(db) -> None:
    """Drop the ``model.output_connection_flow_separate`` parameter
    from the schema.

    Dead toggle and the last of the Batch-B output flag removals: the
    parameter was plumbed into the multi-param ``optional_outputs.csv``
    emitter but nothing on the engine side reads its row from
    ``enable_optional_outputs`` — only ``output_horizon`` is consulted.
    Any value users set was silently dropped.

    Sibling edits in the same commit strip it from the input_derivation
    cl_pars (which leaves only ``output_horizon`` in the
    ``optional_outputs.csv`` emitter — the parameter that IS actually
    consumed), the SET_LIKE_NAMES table, the autoscale quantity-type
    table and the export_settings.yaml params list.

    Side effects: every ``parameter_value`` row referencing
    ``model.output_connection_flow_separate`` is dropped alongside the
    ``parameter_definition``.
    """
    remove_parameters_manual(db, [["model", "output_connection_flow_separate"]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename',help= "The filepath of the database to be migrated")
    args = parser.parse_args()
    migrate_database(args.filename)