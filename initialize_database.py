import json
import sys
from spinedb_api import import_data, DatabaseMapping, create_new_spine_database

def initialize_database(database_name="new_database.sqlite"):

    #get template JSON
    with open ('./version/flextool_template_master.json') as json_file:
        template = json.load(json_file)


    new_engine = create_new_spine_database('sqlite:///' +database_name)
    new_db = DatabaseMapping('sqlite:///' + database_name)
    (num,log) = import_data(new_db,**template)
    print(str(num)+"imports made")
    new_db.commit_session("Initialized")

    return 0



if __name__ == '__main__':
    initialize_database(sys.argv[1])