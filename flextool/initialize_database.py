import json
import argparse
from spinedb_api import import_data, DatabaseMapping
from flextool.migrate_database import migrate_database


def initialize_database(database_name="new_database.sqlite"):

    if not database_name.endswith(".sqlite"):
        print("Give a name with .sqlite file extension")
        exit(-1)

    #get template JSON. This should be kept up to date
    with open ('./version/flextool_template_master.json') as json_file:
        template = json.load(json_file)


    with DatabaseMapping('sqlite:///' + database_name, create = True) as new_db:
        (num,log) = import_data(new_db,**template)
        print(str(num)+" imports made")
        print("Initialized")
        new_db.commit_session("Initialized")

    migrate_database(database_name)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename',help= "The name of the new database")
    args = parser.parse_args()
    initialize_database(args.filename)