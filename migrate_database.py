import json
import argparse
import os
import sys
from spinedb_api import import_data, DatabaseMapping, from_database



def migrate_database(database_path):

    update_functions=[add_version]

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

    for index, func in enumerate(update_functions):
        if index >= version:
            completed = func(db)
            if completed != 0:
                print(str(database_path) + " migration failed in the jump to version " + str(version + 1))
                exit(-1)
            version += 1

    version_up = [["model", "version", version, None, "Contains database version information."]]
    (num,log) = import_data(db, object_parameters = version_up)
    print(str(num)+" imports made to " + database_path)
    
    db.commit_session("Updated Flextool data structure to version " + str(version))

def add_version(db):
    # this function adds the version information to the database if there is none

    version_up = [["model", "version", 1, None, "Contains database version information."]]
    (num,log) = import_data(db, object_parameters = version_up)
    print(str(num)+" imports made")
    db.commit_session("Added version parameter")

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help= "Filepath, absolute or relative to flextool folder")
    args = parser.parse_args()
    migrate_database(args.filepath)