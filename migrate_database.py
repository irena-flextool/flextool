import json
import argparse
import os
import sys
from spinedb_api import import_data, DatabaseMapping, from_database



def migrate_database(database_path):

    new_version = 9.0

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
    if version < new_version:
        add_new_parameters(db)
        version_updated_flag = True

    if version_updated_flag:
        version_up = [["model", "version", new_version, None, "Contains database version information."]]
        (num,log) = import_data(db, object_parameters = version_up)
        print(database_path+ " updated to version "+ str(new_version))
        db.commit_session("Updated Flextool data structure to version " + str(new_version))
    else:
        print(database_path+ " already up-to-date at version "+ str(version))

def add_new_parameters(db):

    #get template JSON. This can be the master or old template if conflicting migrations in between
    with open ('./version/flextool_template_master.json') as json_file:
        template = json.load(json_file)

    #Parameter values need to be added first! Otherwise the new value_list_name cannot be connected!
    #Add parameter_value_lists. Note that object_parameter import and value_list import work differently. Former replaces all, latter adds what is missing.
    (num,log) = import_data(db, parameter_value_lists = template["parameter_value_lists"])

    #With objective parameters, no duplicates are created. These will replace the old ones or create new. There will always be imports.
    (num,log) = import_data(db, object_parameters = template["object_parameters"])

    db.commit_session("Added output node flows")

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help= "Filepath, absolute or relative to flextool folder")
    args = parser.parse_args()
    migrate_database(args.filepath)