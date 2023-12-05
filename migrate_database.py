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
    new_version = 11

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
            change_optional_output_type(db,'./version/flextool_template_default_optional_output.json' )
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

def add_new_parameters(db, filepath):

    #get template JSON. This can be the master or old template if conflicting migrations in between
    with open(filepath) as json_file:
        template = json.load(json_file)

    #Parameter values need to be added first! Otherwise the new value_list_name cannot be connected!
    #Add parameter_value_lists. Note that object_parameter import and value_list import work differently. Former replaces all, latter adds what is missing.
    (num,log) = import_data(db, parameter_value_lists = template["parameter_value_lists"])

    #With objective parameters, no duplicates are created. These will replace the old ones or create new. There will always be imports.
    (num,log) = import_data(db, object_parameters = template["object_parameters"])
    print(filepath)
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

    
    db.remove_items(**{'parameter_definition': [enable_parameter_definition.id]})
    db.commit_session("Changed optional outputs")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help= "Filepath, absolute or relative to flextool folder")
    args = parser.parse_args()
    migrate_database(args.filepath)