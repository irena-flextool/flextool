import json
import os
import subprocess

from migrate_database import migrate_database


def update_flextool():

    #completed = subprocess.run(["git","restore", "."])
    completed = subprocess.run(["git","pull"])

    if completed != 0:
        print("Failed to get the new version")
        exit(-1)

    db_to_update = []
    db_to_update.append('/init.sqlite')
    db_to_update.append('/input_data_template.sqlite')
    
    #add the database used in the input_data tool
    with open ('./.spinetoolbox/project.json') as json_file:
        specifications = json.load(json_file) 
    path = specifications['items']['Input_data']['url']['database']['path']
    db_to_update.append(path)

    #add the databases in the example folder
    dir_list = os.listdir('./how to example databases')
    for i in dir_list:
        if ".sqlite" in i:
            db_to_update.append("./how to example databases/"+ i)

    #migrate the databases to new version
    for i in db_to_update:
        completed = migrate_database(i)
        if completed != 0:
            print("database update failed")
            exit(-1)
    return 0


if __name__ == '__main__':
    update_flextool()