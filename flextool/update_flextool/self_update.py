import json
import os
import subprocess
import shutil
try: 
    from spinedb_api import import_data, DatabaseMapping
except ModuleNotFoundError:
    exit("Cannot find the required Spine-Toolbox module. Check that the environment is activated and the toolbox is installed")
from spinedb_api.exception import NothingToCommit
from flextool._resources import package_data_path
from flextool.update_flextool.db_migration import migrate_database
from flextool.update_flextool.initialize_database import initialize_database
from flextool.update_flextool import canonical_databases


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

    # Locate bundled JSON templates from the installed flextool package.
    master_json = str(package_data_path("schemas/spinedb_schema.json"))
    output_settings_json = str(package_data_path("schemas/output_settings_template.json"))
    output_info_json = str(package_data_path("schemas/output_info_template.json"))
    comparison_settings_json = str(package_data_path("schemas/comparison_settings_template.json"))

    # Create input databases if they do not exist.
    os.makedirs("templates", exist_ok=True)
    if not os.path.exists("input_data.sqlite"):
        initialize_database(master_json, "input_data.sqlite")
    if not os.path.exists("templates/input_data_template.sqlite"):
        initialize_database(master_json, "templates/input_data_template.sqlite")

    # Create user copies of the auxiliary databases
    if not os.path.exists("output_settings.sqlite"):
        initialize_database(output_settings_json, "output_settings.sqlite")
    if not os.path.exists("output_info.sqlite"):
        initialize_database(output_info_json, "output_info.sqlite")
    if not os.path.exists("comparison_settings.sqlite"):
        initialize_database(comparison_settings_json, "comparison_settings.sqlite")

    # Keep CWD-resident template SQLites up-to-date (Spine Toolbox refs).
    for sqlite_rel, json_src in (
        ("templates/output_settings.sqlite", output_settings_json),
        ("templates/output_info.sqlite", output_info_json),
        ("templates/comparison_settings.sqlite", comparison_settings_json),
    ):
        if os.path.exists(sqlite_rel):
            os.remove(sqlite_rel)
        initialize_database(json_src, sqlite_rel)

    # Materialize canonical example/template SQLites from their JSON
    # sources (``flextool/schemas/canonical_databases/*.json``).
    # Skips files that already exist so user edits in the working tree
    # survive.  The canonical JSONs are kept current by
    # ``python -m flextool.update_flextool.canonical_databases migrate-all``
    # which is part of the schema-migration workflow — see
    # CONTRIBUTING.md.
    canonical_databases.materialize(overwrite=False)

    # Generate XLSX templates from the canonical SQLites:
    #   - example_input_template.xlsx  → derived from examples.sqlite
    #   - empty_input_template.xlsx    → derived from input_data_template.sqlite
    # Regenerating each ``update_flextool()`` keeps them in sync with the
    # current schema without us having to track the binary in git.
    from flextool.export_to_tabular import export_to_excel
    _xlsx_pairs = (
        ("templates/examples.sqlite", "templates/example_input_template.xlsx"),
        ("templates/input_data_template.sqlite", "templates/empty_input_template.xlsx"),
    )
    for sqlite_rel, xlsx_rel in _xlsx_pairs:
        if not os.path.exists(sqlite_rel):
            continue
        try:
            export_to_excel(f"sqlite:///{sqlite_rel}", xlsx_rel)
        except Exception as exc:
            print(f"Warning: failed to generate {xlsx_rel}: {exc}")

    # Copy the user-facing example_input.xlsx on first run.  Done after
    # the template regeneration so users get the freshly-built XLSX.
    if not os.path.exists("example_input.xlsx") and os.path.exists("templates/example_input_template.xlsx"):
        shutil.copy("templates/example_input_template.xlsx", "example_input.xlsx")

    # Seed solver_config/highs.opt from the bundled template on first run.
    # The runtime file is user-editable; the template ships in the package.
    os.makedirs("solver_config", exist_ok=True)
    if not os.path.exists("solver_config/highs.opt"):
        shutil.copy(
            str(package_data_path("solver_config/highs.opt.template")),
            "solver_config/highs.opt",
        )

    # Migrate user-owned SQLites (canonical ones are already at current
    # version because they were just materialized from the canonical
    # JSONs, which migrate-all keeps at FLEXTOOL_DB_VERSION).
    db_to_update = [
        "input_data.sqlite",
        "templates/input_data_template.sqlite",
    ]

    # add the database used in the input_data tool
    with open("./.spinetoolbox/project.json") as json_file:
        specifications = json.load(json_file)
    path = specifications["items"]["Input data"]["url"]["database"]["path"]
    db_to_update.append(path)

    # migrate the databases to new version
    for i in db_to_update:
        migrate_database(i)

    result_template_path = str(package_data_path("schemas/pre_v26/flextool_template_results_master.json"))
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
