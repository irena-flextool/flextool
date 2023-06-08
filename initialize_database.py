import json
import sys
from spinedb_api import import_data, DatabaseMapping

def initialize_database(database_name="new_database.sqlite"):

    if not database_name.endswith(".sqlite"):
        print("Give a name with .sqlite file extension")
        exit(-1)

    #get template JSON. This should be kept up to date
    with open ('./version/flextool_template_master.json') as json_file:
        template = json.load(json_file)


    new_db = DatabaseMapping('sqlite:///' + database_name, create = True)
    (num,log) = import_data(new_db,**template)
    print(str(num)+" imports made")
    print("Initialized")
    new_db.commit_session("Initialized")




if __name__ == '__main__':
    initialize_database(sys.argv[1])