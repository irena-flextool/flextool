import json
import os
import subprocess
import shutil
from spinedb_api import import_data, DatabaseMapping
from migrate_database import migrate_database


def update_flextool():

    shutil.copy('./.spinetoolbox/project.json','./.spinetoolbox/project_temp.json')
    completed = subprocess.run(["git","restore","."])
    completed = subprocess.run(["git","pull"])
    shutil.copy('./.spinetoolbox/project_temp.json','./.spinetoolbox/project.json')
    os.remove('./.spinetoolbox/project_temp.json')

    if completed.returncode != 0:
        print("Failed to get the new version")
        exit(-1)

    if not os.path.exists('Input_data.sqlite'):
        shutil.copy('input_data_template.sqlite','Input_data.sqlite')
    if not os.path.exists('Results.sqlite'):
        shutil.copy('Results_template.sqlite','Results.sqlite')

    db_to_update = []
    db_to_update.append('init.sqlite')
    db_to_update.append('input_data_template.sqlite')
    
    #add the database used in the input_data tool
    with open ('./.spinetoolbox/project.json') as json_file:
        specifications = json.load(json_file) 
    path = specifications['items']['Input_data']['url']['database']['path']
    db_to_update.append(path)

    #add the databases in the example folder
    dir_list = os.listdir('./how to example databases')
    for i in dir_list:
        if i.endswith(".sqlite"):
            db_to_update.append("how to example databases/"+ i)

    #migrate the databases to new version
    for i in db_to_update:
        migrate_database(i)

    #update result parameter definitions
    db = DatabaseMapping('sqlite:///' + 'Results.sqlite', create = False)
    #get template JSON. This can be the master or old template if conflicting migrations in between
    with open ('./version/flextool_template_results_master.json') as json_file:
        template = json.load(json_file)
    #these update the old descriptions, but wont remove them or change names (the new name is created, but old stays)
    (num,log) = import_data(db, object_parameters = template["object_parameters"])
    (num,log) = import_data(db, relationship_parameters = template["relationship_parameters"])
    db.commit_session("Updated relationship_parameters, object parameters to the Results.sqlite")
    

if __name__ == '__main__':
    update_flextool()