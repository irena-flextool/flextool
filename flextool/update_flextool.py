import argparse
import json
import os
import subprocess
import shutil
try: 
    from spinedb_api import import_data, DatabaseMapping, SpineDBAPIError
except ModuleNotFoundError:
    exit("Cannot find the required Spine-Toolbox module. Check that the environment is activated and the toolbox is installed")
from flextool.migrate_database import migrate_database
from flextool.initialize_database import initialize_database


def update_flextool(skip_git):

    shutil.copy("./.spinetoolbox/project.json", "./.spinetoolbox/project_temp.json")
    if not skip_git:
        completed = subprocess.run(["git", "restore", "."])
        if completed.returncode != 0:
            print("Failed to restore version controlled files.")
            exit(-1)

        completed = subprocess.run(["git", "pull"])
        if completed.returncode != 0:
            print("Failed to get the new version.")
            exit(-1)

    migrate_project("./.spinetoolbox/project_temp.json","./.spinetoolbox/project.json")
    os.remove("./.spinetoolbox/project_temp.json")

    if not os.path.exists("input_data.sqlite"):
        initialize_database("input_data.sqlite")
    if not os.path.exists("templates/input_data_template.sqlite"):
        initialize_database("templates/input_data_template.sqlite")
    if not os.path.exists("example_input.xlsx"):
        shutil.copy("./templates/example_input_template.xlsx", "./example_input.xlsx")

    db_to_update = []
    db_to_update.append("templates/examples.sqlite")
    db_to_update.append("input_data.sqlite")
    db_to_update.append("templates/input_data_template.sqlite")
    db_to_update.append("templates/time_settings_only.sqlite")

    # add the database used in the input_data tool
    with open("./.spinetoolbox/project.json") as json_file:
        specifications = json.load(json_file)
    path = specifications["items"]["Input_data"]["url"]["database"]["path"]
    db_to_update.append(path)

    # add the databases in the example folder
    dir_list = os.listdir("./how to example databases")
    for i in dir_list:
        if i.endswith(".sqlite"):
            db_to_update.append("how to example databases/" + i)

    # migrate the databases to new version
    for i in db_to_update:
        migrate_database(i)

    result_template_path = './version/flextool_template_results_master.json'
    #replace the template sqlite
    if os.path.exists('templates/results_template.sqlite'):
        os.remove('templates/results_template.sqlite')
    initialize_result_database('templates/results_template.sqlite', result_template_path)

    if not os.path.exists("results.sqlite"):
        shutil.copy("templates/results_template.sqlite", "results.sqlite")
    #update result parameter definitions    
    db = DatabaseMapping('sqlite:///' + 'results.sqlite', create = False, upgrade = True)
    #get template JSON. This can be the master or old template if conflicting migrations in between
    with open (result_template_path) as json_file:
        template = json.load(json_file)
    #these update the old descriptions, but wont remove them or change names (the new name is created, but old stays)
    (num,log) = import_data(db, object_parameters = template["object_parameters"])
    (num,log) = import_data(db, relationship_parameters = template["relationship_parameters"])

    try:
        db.commit_session("Updated relationship_parameters, object parameters to the Results.sqlite")
    except SpineDBAPIError:
        print("These parameters have been added before, continuing") 
    return 0

def initialize_result_database(filename,json_path):

    db = DatabaseMapping('sqlite:///' + filename, create = True)

    with open (json_path) as json_file:
        template = json.load(json_file)

    (num,log) = import_data(db,**template)
    print("Result database initialized")
    db.commit_session("Result database initialized")

def migrate_project(old_path, new_path):
    #purpose of this is to update some of the items that users should not need to modify
    #done simply by copying from the git project.json
    #should be replaced if major changes to project.json
    
    #items that are copied
    items = [
        "Replace with examples",
        "FlexTool",
        "Export_to_CSV",
        "Import_results",
        "Plot_results",
        "Plot_settings",
    ]

    with open(old_path) as old_json:
        old_dict = json.load(old_json)
    with open(new_path) as new_json:
        new_dict = json.load(new_json)
    
    for item in items:
        if item in old_dict["items"].keys():
            for param in old_dict["items"][item].keys():
                if param != "x" and param != "y":
                    old_dict["items"][item][param] = new_dict["items"][item][param]

    if ("Open_summary" not in old_dict["items"].keys()) and ("Open_summary" in new_dict["items"].keys()):
        old_dict["items"]["Open_summary"] = new_dict["items"]["Open_summary"]
        old_dict["project"]["connections"].append({"name": "from FlexTool3 to Open_summary", "from": ["FlexTool3","bottom"],"to": ["Open_summary","right"]})
        old_dict["project"]["specifications"]["Tool"].append({"type": "path","relative": True,"path": ".spinetoolbox/specifications/Tool/open_summary.json"})

    
    with open("./.spinetoolbox/project_temp2.json", "w") as outfile: 
        json.dump(old_dict, outfile, indent=4)

    shutil.copy("./.spinetoolbox/project_temp2.json", new_path)
    os.remove("./.spinetoolbox/project_temp2.json")


def main():
    update_flextool(False)


if __name__ == '__main__':
    main()
