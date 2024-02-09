import json
import argparse
import os
import sys
from spinedb_api import import_data, DatabaseMapping, from_database

def migrate_database(database_path):

    if not os.path.exists(database_path) or not database_path.endswith(".sqlite"):
        print("No sqlite file at " + database_path)
        exit(-1)

    db = DatabaseMapping('sqlite:///' + database_path, create = False)
    sq= db.object_parameter_definition_sq
    settings_parameter = db.query(sq).filter(sq.c.object_class_name == "model").filter(sq.c.parameter_name == "version").one_or_none()
    if settings_parameter is None:
        #if no version assume version 0
        print("No version found. Assuming version 0, if older, migration might not work")
        version = 0
    else:
        version = from_database(settings_parameter.default_value, settings_parameter.default_type)

    next_version = int(version) + 1
    new_version = 17

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

    db.remove_items(**{'parameter_definition': id_list})
    db.commit_session("Removed parameters")
    return 0

def add_parameters_manual(db,new_parameters):
    (num,log) = import_data(db, object_parameters = new_parameters)
    db.commit_session("Added new parameters")
    return 0

def add_value_list_manual(db, new_value_lists):
    (num,log) = import_data(db,parameter_value_lists = new_value_lists)
    db.commit_session("Added new parameter value lists")
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

    db.commit_session("Added output node flows")

    return 0

def change_optional_output_type(db, filepath):

    sq= db.entity_parameter_value_sq
    sq_def = db.object_parameter_definition_sq
    enable_parameter_query = db.query(sq).filter(sq.c.object_class_name == "model").filter(sq.c.parameter_name == "enable_optional_outputs").all()
    disable_parameter_query = db.query(sq).filter(sq.c.object_class_name == "model").filter(sq.c.parameter_name == "disable_optional_outputs").all()
    enable_parameter_definition =  db.query(sq_def).filter(sq_def.c.object_class_name == "model").filter(sq_def.c.parameter_name == "enable_optional_outputs").one_or_none()
    disable_parameter_definition =  db.query(sq_def).filter(sq_def.c.object_class_name == "model").filter(sq_def.c.parameter_name == "disable_optional_outputs").one_or_none()

    paramset_enable = []
    paramset_disable = []
    for param in enable_parameter_query:
        enable_optional_outputs = from_database(param._asdict()['value'].decode(), "array").values
        meta = [param.entity_class_name, param.entity_name, param.alternative_name]
        paramset_enable.append((meta,enable_optional_outputs))

    for param in disable_parameter_query:
        disable_optional_outputs = from_database(param._asdict()['value'].decode(), "array").values
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
    
    db.remove_items(**{'parameter_definition': [enable_parameter_definition.id,disable_parameter_definition.id]})
    db.commit_session("Changed optional outputs")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help= "Filepath, absolute or relative to flextool folder")
    args = parser.parse_args()
    migrate_database(args.filepath)