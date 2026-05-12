import json
import os
import subprocess
import shutil
try: 
    from spinedb_api import import_data, DatabaseMapping
except ModuleNotFoundError:
    exit("Cannot find the required Spine-Toolbox module. Check that the environment is activated and the toolbox is installed")
from spinedb_api.exception import NothingToCommit
from flextool.update_flextool.db_migration import migrate_database
from flextool.update_flextool.initialize_database import initialize_database


def _reinstall_if_needed():
    """Re-install flextool and refresh its declared dependencies.

    Runs ``pip install --upgrade [-e] <repo_root>`` after every
    ``git pull``.  Three reasons:

    1. **Editable installs need this too.**  ``pip install -e .`` picks up
       source-tree changes immediately but does NOT pull in new
       dependencies that appeared in ``pyproject.toml``.  Without this
       reinstall, an editable user pulling a commit that adds (say)
       ``polar-high`` to ``[project.dependencies]`` would see
       ``ModuleNotFoundError`` on the next solver invocation.
    2. **``--upgrade``** re-resolves the install target and bumps any
       direct dependency whose pyproject pin moved past the currently
       installed version (e.g. ``highspy>=1.14`` floor bumps land
       automatically).  Replaces the previous bespoke
       ``pip install --upgrade highspy>=1.14`` workaround.
    3. **NO ``--upgrade-strategy=eager``** — pip's default
       ``only-if-needed`` semantics keep transitive dependencies stable
       unless flextool's own pins force a move.  This is deliberate:
       eager upgrades churn the user's whole environment (numpy,
       pandas, SQLAlchemy, etc.) and break other editable installs
       (spinedb-api, toolbox extras) that pin to specific versions.

    The install target is resolved from ``__file__`` rather than ``.``
    so the re-install always targets THIS repo, regardless of the
    caller's CWD (Spine Toolbox typically invokes from the project
    directory, not the flextool repo root).
    """
    import sys
    import os

    # flextool/update_flextool/self_update.py  →  flextool/  →  repo root.
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    )))

    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", "flextool"],
        capture_output=True, text=True,
    )
    is_editable = "Editable project location" in result.stdout

    print(
        f"Refreshing flextool install and dependencies "
        f"({'editable' if is_editable else 'non-editable'}, target={repo_root})..."
    )
    target = ["-e", repo_root] if is_editable else [repo_root]
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", *target],
    )
    if completed.returncode != 0:
        flag = "-e " if is_editable else ""
        print(
            f"Warning: pip install failed.  Run "
            f"'pip install --upgrade {flag}{repo_root}' manually."
        )


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

    # Re-install the package so that code changes from git pull take effect.
    # Skip if the package is installed in editable mode (changes are live already).
    _reinstall_if_needed()

    migrate_project("./.spinetoolbox/project_temp.json","./.spinetoolbox/project.json")
    os.remove("./.spinetoolbox/project_temp.json")

    # Create input databases if they do not exist.
    if not os.path.exists("input_data.sqlite"):
        initialize_database("./version/flextool_template_master.json", "input_data.sqlite")
    if not os.path.exists("templates/input_data_template.sqlite"):
        initialize_database("./version/flextool_template_master.json", "templates/input_data_template.sqlite")

    # Copy excel example from templates. It has no migration --> do it manually
    if not os.path.exists("example_input.xlsx"):
        shutil.copy("./templates/example_input_template.xlsx", "./example_input.xlsx")

    # Create user copies of the auxiliary databases
    if not os.path.exists("output_settings.sqlite"):
        initialize_database("./version/output_settings_template.json", "output_settings.sqlite")
    if not os.path.exists("output_info.sqlite"):
        initialize_database("./version/output_info_template.json", "output_info.sqlite")
    if not os.path.exists("comparison_settings.sqlite"):
        initialize_database("./version/comparison_settings_template.json", "comparison_settings.sqlite")

    # Keep templates up-to-date
    if os.path.exists("templates/output_settings.sqlite"):
        os.remove("templates/output_settings.sqlite")
    initialize_database("./version/output_settings_template.json", "templates/output_settings.sqlite")
    if os.path.exists("templates/output_info.sqlite"):
        os.remove("templates/output_info.sqlite")
    initialize_database("./version/output_info_template.json", "templates/output_info.sqlite")
    if os.path.exists("templates/comparison_settings.sqlite"):
        os.remove("templates/comparison_settings.sqlite")
    initialize_database("./version/comparison_settings_template.json", "templates/comparison_settings.sqlite")

    db_to_update = []
    db_to_update.append("templates/examples.sqlite")
    db_to_update.append("input_data.sqlite")
    db_to_update.append("templates/input_data_template.sqlite")
    db_to_update.append("templates/time_settings_only.sqlite")

    # add the database used in the input_data tool
    with open("./.spinetoolbox/project.json") as json_file:
        specifications = json.load(json_file)
    path = specifications["items"]["Input data"]["url"]["database"]["path"]
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
    #get template JSON. This can be the master or old template if conflicting migrations in between
    with open (result_template_path) as json_file:
        template = json.load(json_file)
    with DatabaseMapping('sqlite:///' + 'results.sqlite', create = False, upgrade = True) as db:
        #these update the old descriptions, but wont remove them or change names (the new name is created, but old stays)
        (num,log) = import_data(db, object_parameters = template["object_parameters"])
        (num,log) = import_data(db, relationship_parameters = template["relationship_parameters"])
        try:
            db.commit_session("Updated relationship_parameters, object parameters to the Results.sqlite")
        except NothingToCommit:
            print("These parameters have been added before, continuing")
        return 0

def initialize_result_database(filename,json_path):

    with open (json_path) as json_file:
        template = json.load(json_file)
    with DatabaseMapping('sqlite:///' + filename, create = True) as db:
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

    if ("Migrate database version" not in old_dict["items"].keys()) and ("Migrate database version" in new_dict["items"].keys()):
        old_dict["items"]["Migrate database version"] = new_dict["items"]["Migrate database version"]
        old_dict["project"]["connections"].append({"name": "from Migrate database version to Input_data", 
                                                   "from": ["Migrate database version","right"],
                                                   "to": ["Input_data","bottom"]})
        old_dict["project"]["specifications"]["Tool"].append({  "type": "path",
                                                                "relative": True,
                                                                "path": ".spinetoolbox/specifications/Tool/migrate_database.json"})

    
    with open("./.spinetoolbox/project_temp2.json", "w") as outfile: 
        json.dump(old_dict, outfile, indent=4)

    shutil.copy("./.spinetoolbox/project_temp2.json", new_path)
    os.remove("./.spinetoolbox/project_temp2.json")


def main():
    update_flextool(False)


if __name__ == '__main__':
    main()
