import json
import os
import argparse
from spinedb_api import import_data, DatabaseMapping, from_database, SpineDBAPIError, to_database
import logging

from flextool.update_flextool import FLEXTOOL_DB_VERSION

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
                add_new_parameters(db, './version/flextool_template_v2.json')
            elif next_version == 2:
                add_new_parameters(db, './version/flextool_template_rolling_window.json')
            elif next_version == 3:
                add_new_parameters(db, './version/flextool_template_lifetime_method.json')
            elif next_version == 4:
                add_new_parameters(db, './version/flextool_template_drop_down.json')
            elif next_version == 5:
                add_new_parameters(db, './version/flextool_template_optional_outputs.json')
            elif next_version == 6:
                add_new_parameters(db, './version/flextool_template_default_value.json')
            elif next_version == 7:
                add_new_parameters(db, './version/flextool_template_rolling_start_remove.json')
            elif next_version == 8:
                add_new_parameters(db, './version/flextool_template_output_node_flows.json')
            elif next_version == 9:
                add_new_parameters(db, './version/flextool_template_constant_default.json')
            elif next_version == 10:
                add_new_parameters(db, './version/flextool_template_storage_binding_defaults.json')
            elif next_version == 11:
                change_optional_output_type(db,'./version/flextool_template_default_optional_output.json')
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
                db.commit_session("Added cumulative investments")
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
                db.commit_session("Added DC power flow parameters")
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
                db.commit_session("Added minimum time method support and fixed penalty descriptions")
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
                db.commit_session("Renamed economic parameters: interest_rate->discount_rate, discount_rate->inflation_rate")
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
                db.commit_session("Added transfer_methods_group parameter_value_list for group transfer_method")
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
                db.commit_session(
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
                db.commit_session("Added timeset.timeset_weights parameter")
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
                db.commit_session(
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
                    db.commit_session(
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
                    db.commit_session(
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
                    if hs and not hb:
                        raise SpineDBAPIError(
                            f"Node '{key[0][0]}' (alternative '{key[1]}') has "
                            f"has_storage=yes but has_balance is not 'yes'.  "
                            f"This combination was rejected at solve time "
                            f"prior to v38 and cannot be migrated "
                            f"automatically.  Set has_balance=yes (or remove "
                            f"has_storage) before running the migration."
                        )
                    if hb and hs:
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
                    db.commit_session(
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
                    db.commit_session(
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
                    parameter_type_list=("1d_map",),
                    description=(
                        "Stepped supply curve for "
                        "price_method='price_ladder_cumulative'.  "
                        "Structure: Map(tier -> {price, quantity}).  "
                        "1-based integer tier index.  quantity=inf marks an "
                        "unbounded tail tier.  Period-agnostic — the cap is "
                        "a single total across the full model horizon."
                    ),
                )
                db.add_update_item(
                    "parameter_definition",
                    entity_class_name="commodity",
                    name="price_ladder_annual",
                    parameter_type_list=("1d_map", "2d_map"),
                    description=(
                        "Stepped supply curve for "
                        "price_method='price_ladder_annual'.  Two forms "
                        "accepted: 1d Map(tier -> {price, quantity}) applies "
                        "the same limit every period; 2d "
                        "Map(tier -> Map(period -> {price, quantity})) "
                        "varies per period.  1-based integer tier.  "
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
                    db.commit_session(
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
                    db.commit_session(
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
                for vl_name in ("output_node_flows", "output_results"):
                    vl = db.item(
                        db.mapped_table("parameter_value_list"), name=vl_name,
                    )
                    if vl:
                        try:
                            db.remove_items("parameter_value_list", vl["id"])
                        except SpineDBAPIError:
                            pass

                try:
                    db.commit_session(
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
                    db.commit_session(
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
                # Two touch points in the schema:
                #
                # 1. transfer_methods_group value list.  The group-level
                #    override must accept "unidirectional" so users can
                #    promote a subnet to unidirectional via the group
                #    override as well.
                # 2. Description refresh on group.transfer_method so the
                #    dropdown tooltip advertises the new option.
                #
                # connection.transfer_method itself carries only
                # parameter_type_list=("str",) today — no value list
                # constraint — so "unidirectional" is already accepted at
                # the connection level without a further schema edit.
                add_value_list_manual(db, [
                    ["transfer_methods_group", "unidirectional"],
                ])
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
                    db.commit_session(
                        "Added 'unidirectional' to transfer_methods_group"
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
                    db.commit_session(
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
                    db.commit_session(
                        "v49: restore advisory enum defaults — "
                        "node.inflow_method=use_original; "
                        "constraint.sense=equal; "
                        "reserve__upDown__group.reserve_method=no_reserve"
                    )
                except SpineDBAPIError:
                    pass
            else:
                print("Version invalid")
            next_version += 1

        if version < new_version:
            version_up = [["model", "version", new_version, None, "Contains database version information."]]
            (num,log) = import_data(db, object_parameters = version_up)
            print(database_path+ " updated to version "+ str(new_version))
            db.commit_session("Updated Flextool data structure to version " + str(new_version))
        else:
            print(database_path+ " already up-to-date at version "+ str(version))

def add_version(db):
    # this function adds the version information to the database if there is none

    version_up = [["model", "version", 1, None, "Contains database version information."]]
    (num,log) = import_data(db, object_parameters = version_up)
    db.commit_session("Added version parameter")

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
        db.commit_session("Removed parameters")
    except SpineDBAPIError:
        print("This removal has been done before, continuing")
    return 0

def add_parameters_manual(db,new_parameters):
    (num,log) = import_data(db, object_parameters = new_parameters)
    try:
        db.commit_session("Added new parameters")
    except SpineDBAPIError:
        print("These parameters have been added before, continuing") 
    return 0

def add_relationships_manual(db, new_relationships):
    (num,log) = import_data(db, relationship_parameters = new_relationships)
    try:
        db.commit_session("Added new parameters")
    except SpineDBAPIError:
        print("These parameters have been added before, continuing") 
    return 0

def add_value_list_manual(db, new_value_lists):
    (num,log) = import_data(db,parameter_value_lists = new_value_lists)
    try:
        db.commit_session("Added new parameter value lists")
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
        db.commit_session("Added new parameters")
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
        db.commit_session("Changed optional outputs")
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
    m[("group", "include_stochastics")] = "model"

    # --- solve_basics --------------------------------------------------
    for p in (
        "solver", "solve_mode", "period_timeset", "realized_periods",
        "invest_periods", "years_represented",
    ):
        m[("solve", p)] = "solve_basics"

    # --- solve_advanced -----------------------------------------------
    # timeline_hole_multiplier belongs here "if present" (proposal).  The
    # migration below handles that conditionally, so we don't pre-declare
    # it in this map.
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
        "timeset_weights",
    ):
        m[("timeset", p)] = "timeline"

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
      4. If ``solve.timeline_hole_multiplier`` exists in this database,
         assign it to ``solve_advanced`` (proposal says "if present").
         Otherwise skip silently.
    """
    # Step 1: rename the existing Outputs group (or create it if it is
    # somehow missing) and update its priority.  Doing an id-keyed update
    # preserves the link to the four output-parameter_definitions that
    # v43 already tagged.
    outputs_group = db.item(
        db.mapped_table("parameter_group"), name="Outputs",
    )
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

    # Step 4: conditionally tag timeline_hole_multiplier if present.
    thm = list(db.find_parameter_definitions(
        entity_class_name="solve", name="timeline_hole_multiplier",
    ))
    if thm:
        db.add_update_item(
            "parameter_definition",
            entity_class_name="solve",
            name="timeline_hole_multiplier",
            parameter_group_name="solve_advanced",
        )

    try:
        db.commit_session(
            "v44: renamed 'Outputs' parameter_group to 'output'; added 14 "
            "new parameter_groups (basics, investment, retirement, storage, "
            "tech_advanced, reserve, emission, network, flow_limit, "
            "constraint, model, solve_basics, solve_advanced, timeline); "
            "assigned every parameter_definition to its group per "
            "rivendell/PROPOSAL_parameter_groups.md"
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
        db.commit_session(
            "v45: updated parameter_group colours to a mid-tone palette "
            "(readable on both light and dark IDE themes); calm cool tones "
            "for the common groups, warm tones for advanced / risk-prone "
            "groups"
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
        db.commit_session(
            "v46: added solve.use_row_scaling parameter (Agent 5 LP-scaling): "
            "per-solve opt-in for automatic row scaling; default 'no' "
            "preserves pre-scaling behaviour."
        )
    except SpineDBAPIError:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename',help= "The filepath of the database to be migrated")
    args = parser.parse_args()
    migrate_database(args.filename)