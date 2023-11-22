import json
import argparse
import os
import sys
from spinedb_api import import_data, DatabaseMapping, from_database



def migrate_database(database_path):

    update_functions=[
                     add_version,
                     add_lifetime_method,
                     add_rolling_window,
                     add_drop_down,
                     add_optional_outputs,
                     add_default_value,
                     add_rolling_start_remove]

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

    version_updated_flag = False
    for index, func in enumerate(update_functions):
        if index >= version:
            completed = func(db)
            if completed != 0:
                print(str(database_path) + " migration failed in the jump to version " + str(version + 1))
                exit(-1)
            version += 1
            version_updated_flag = True

    if version_updated_flag:
        version_up = [["model", "version", version, None, "Contains database version information."]]
        (num,log) = import_data(db, object_parameters = version_up)
        print(database_path+ " updated to version "+ str(version))
        db.commit_session("Updated Flextool data structure to version " + str(version))
    else:
        print(database_path+ " already up-to-date at version "+ str(version))

def add_rolling_window(db):
    #get template JSON. This can be the master or old template if conflicting migrations in between
    with open ('./version/flextool_template_rolling_window.json') as json_file:
        template = json.load(json_file)

    #With objective parameters, no duplicates are created. These will replace the old ones or create new. There will always be imports.
    (num,log) = import_data(db, object_parameters = template["object_parameters"])

    #Add parameter_value_lists. Note that object_parameter import and value_list import work differently. Former replaces all, latter adds what is missing.
    (num,log) = import_data(db, parameter_value_lists = template["parameter_value_lists"])
    db.commit_session("Added rolling_window object parameters and parameter value lists")
    return 0 

def add_drop_down(db):

    #get template JSON. This can be the master or old template if conflicting migrations in between
    with open ('./version/flextool_template_drop_down.json') as json_file:
        template = json.load(json_file)

    #With objective parameters, no duplicates are created. These will replace the old ones or create new. There will always be imports.
    (num,log) = import_data(db, object_parameters = template["object_parameters"])

    #With objective parameters, no duplicates are created. These will replace the old ones or create new. There will always be imports.
    (num,log) = import_data(db, relationship_parameters = template["relationship_parameters"])

    #Add parameter_value_lists. Note that object_parameter import and value_list import work differently. Former replaces all, latter adds what is missing.
    (num,log) = import_data(db, parameter_value_lists = template["parameter_value_lists"])
    db.commit_session("Updated relationship_parameters, object parameters and parameter value lists")
    return 0 

def add_version(db):
    # this function adds the version information to the database if there is none

    version_up = [["model", "version", 1, None, "Contains database version information."]]
    (num,log) = import_data(db, object_parameters = version_up)
    db.commit_session("Added version parameter")

    return 0

def add_lifetime_method(db):
    
    #get template JSON. This can be the master or old template if conflicting migrations in between
    with open ('./version/flextool_template_v2.json') as json_file:
        template = json.load(json_file)

    #With objective parameters, no duplicates are created. These will replace the old ones or create new. There will always be imports.
    (num,log) = import_data(db, object_parameters = template["object_parameters"])

    #Add parameter_value_lists. Note that object_parameter import and value_list import work differently. Former replaces all, latter adds what is missing.
    (num,log) = import_data(db, parameter_value_lists = template["parameter_value_lists"])
    db.commit_session("Added lifetime_method object parameters and parameter value lists")

    return 0

def add_optional_outputs(db):

    #get template JSON. This can be the master or old template if conflicting migrations in between
    with open ('./version/flextool_template_optional_outputs.json') as json_file:
        template = json.load(json_file)

    #With objective parameters, no duplicates are created. These will replace the old ones or create new. There will always be imports.
    (num,log) = import_data(db, object_parameters = template["object_parameters"])

    db.commit_session("Added optional outputs object parameters and parameter value lists")

    return 0

def add_default_value(db):

    #get template JSON. This can be the master or old template if conflicting migrations in between
    with open ('./version/flextool_template_default_value.json') as json_file:
        template = json.load(json_file)

    #With objective parameters, no duplicates are created. These will replace the old ones or create new. There will always be imports.
    (num,log) = import_data(db, object_parameters = template["object_parameters"])

    db.commit_session("Added optional outputs object parameters and parameter value lists")

    return 0

def add_rolling_start_remove(db):
    
    #get template JSON. This can be the master or old template if conflicting migrations in between
    with open ('./version/flextool_template_rolling_start_remove.json') as json_file:
        template = json.load(json_file)

    #With objective parameters, no duplicates are created. These will replace the old ones or create new. There will always be imports.
    (num,log) = import_data(db, object_parameters = template["object_parameters"])

    #Add parameter_value_lists. Note that object_parameter import and value_list import work differently. Former replaces all, latter adds what is missing.
    (num,log) = import_data(db, parameter_value_lists = template["parameter_value_lists"])
    db.commit_session("Added lifetime_method object parameters and parameter value lists")

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help= "Filepath, absolute or relative to flextool folder")
    args = parser.parse_args()
    migrate_database(args.filepath)