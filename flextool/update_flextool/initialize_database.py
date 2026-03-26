import json
import argparse
from spinedb_api import import_data, DatabaseMapping


def initialize_database(json_template, database_name="new_database.sqlite"):

    if not database_name.endswith(".sqlite"):
        print("Give a name with .sqlite file extension")
        exit(-1)

    #get template JSON. This should be kept up to date
    with open (json_template) as json_file:
        template = json.load(json_file)


    with DatabaseMapping('sqlite:///' + database_name, create = True) as new_db:
        # Remove any default alternatives created by spinedb_api (e.g. 'Base')
        # so they don't conflict with the template's own alternatives.
        for alt in new_db.find_alternatives():
            try:
                new_db.remove_alternative(name=alt['name'])
            except Exception:
                pass
        (num,log) = import_data(new_db,**template)
        print(str(num)+" imports made")
        print("Initialized " + database_name)
        new_db.commit_session("Initialized")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json_filepath',help= "The filepath of the source JSON")
    parser.add_argument('db_filepath',help= "The filepath of the new database")
    args = parser.parse_args()
    initialize_database(args.json_filepath, args.db_filepath)