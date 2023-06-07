import json
import subprocess
import sys
from spinedb_api import import_data, DatabaseMapping, export_object_parameters



def migrate_database(database_path):

    update_functions=[add_version]

    db = DatabaseMapping('sqlite:///' + database_path)
    objects = export_object_parameters(db)
    settings = next((x for x in objects if x[0]=="model" and x[1]=="version"), None)
    if settings == None:
        #if no version assume version 0
        print("No version found. Assuming version 0, if older, migration might not work")
        version = 0
    else:
        version = settings[2]

    for index, func in enumerate(update_functions):
        if index >= version:
            completed = func(db)
            if completed != 0:
                return completed
            version += 1

    version_up = [["model", "version", version, None, "Contains database version information."]]
    (num,log) = import_data(db, object_parameters = version_up)
    print(str(num)+" imports made to " + database_path)
    
    db.commit_session("Updated Flextool data structure to version " + str(version))
   
    return 0

def add_version(db):
    # this function adds the version information to the databases if there is none

    #get template JSON, reserve the option to use older templates if more than one conflicting version jump required
    #with open ('./version/flextool_template_master.json') as json_file:
    #    template = json.load(json_file) 

    version_up = [["model", "version", 1, None, "Contains database version information."]]
    (num,log) = import_data(db, object_parameters = version_up)
    print(str(num)+" imports made")
    db.commit_session("Added version parameter")

    return 0

if __name__ == '__main__':
    migrate_database(sys.argv[1])