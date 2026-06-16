import json
import logging
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


def _ensure_compatible_solver_stack():
    """Verify the native solver stack (polars, highspy/HiGHS) runs on this
    machine and, if a component crashes natively, re-install the compatible
    build automatically.

    Two known cases (see ``flextool.env_check``): the default ``polars``
    wheel on a CPU lacking the SIMD baseline it was built for (cure:
    ``polars-lts-cpu``), and ``highspy==1.14.0`` import-crashing on older
    Windows (cure: ``highspy==1.13.1``).  Both surface as a native crash —
    a Windows access violation (exit ``3221225477``) or a POSIX
    ``SIGILL``/``SIGSEGV``.  We probe in a child process so the crash is
    observable and pinpoint which component died, then apply its remedy.

    Run unattended here (the user already opted into installs by choosing
    "Update FlexTool"), but loudly: the banner names what is happening and
    how to stop it, and everything prints to stdout so it lands in the
    update log window for copy-paste.
    """
    import sys
    from flextool import env_check

    print("\nChecking that the solver libraries run on this computer...")
    probe = env_check.probe_solver_stack()
    print(env_check.diagnostics_report(probe))
    if not probe.is_native_fault:
        # OK, or an ordinary error (e.g. a package missing) that a build
        # swap would not fix — leave it for the normal install machinery.
        return

    steps = env_check.remediation_steps(probe.failed_component, sys.executable)
    if steps is None:
        # A native crash with no package-level remedy (e.g. polar_high).
        print("\n" + env_check.UNFIXABLE_HELP)
        return

    print("\n" + env_check.remediation_banner(probe.failed_component))
    for step in steps:
        print("\n$ " + " ".join(step))
        completed = subprocess.run(step)
        if completed.returncode != 0:
            print(
                f"Warning: '{' '.join(step)}' exited {completed.returncode}. "
                f"Run it manually to finish the fix."
            )
            return

    reprobe = env_check.probe_solver_stack()
    print("\n" + reprobe.summary())
    if not reprobe.ok:
        print("\n" + env_check.UNFIXABLE_HELP)


def _sync_settings_param_defs(json_template, sqlite_path):
    """Back-fill parameter DEFINITIONS from a settings template into an
    EXISTING user settings DB.

    The user-editable settings DBs (``output_settings.sqlite`` etc.) are only
    created-if-absent, so a DB made by an older FlexTool version never gains
    options added later (e.g. ``output-spinedb``).  This imports just the
    schema — entity classes + parameter definitions/types — so new options
    appear in the editor, WITHOUT importing ``parameter_values`` (the user's
    existing choices are never overwritten).  Idempotent; non-fatal on error.
    """
    from flextool.update_flextool.export_database import keep_serialized_unparse
    try:
        with open(json_template) as json_file:
            template = json.load(json_file)
        payload = {
            key: template[key]
            for key in ("entity_classes", "parameter_definitions", "parameter_types")
            if key in template
        }
        if not payload:
            return
        with DatabaseMapping('sqlite:///' + sqlite_path, create=False) as db:
            (num, _log) = import_data(db, unparse_value=keep_serialized_unparse, **payload)
            if num:
                db.commit_session(
                    "flextool-update: back-fill settings parameter definitions"
                )
    except NothingToCommit:
        pass
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not sync parameter definitions into "
              f"{sqlite_path}: {exc}")


_PROJECT_FOLDER_TEMPLATE = (
    "# FlexTool output project folder (user-local; not tracked by git).\n"
    "# Write ONE path below — absolute, or relative to the FlexTool root —\n"
    "# to direct this workflow's outputs (output_parquet/, results.sqlite,\n"
    "# plots, and the per-project plot_settings.yaml) into that project folder.\n"
    "# Leave blank to use the FlexTool root.\n"
    "# Use projects sub-folders to allow access by FlexTool GUI as well.\n"
    "# Using the project specific input_sources folder for the input data db\n"
    "# makes it visible in the GUI too.\n"
    "# e.g.:  projects/Rivendell\n"
)


def ensure_runtime_files(verbose: bool = False) -> None:
    """Idempotently materialize the per-installation runtime files that
    FlexTool needs but does not track in git.

    Every step is **create-if-missing**: an existing user-edited file is
    never overwritten, so this is safe to call on every GUI launch *and*
    as the seeding step of :func:`update_flextool`.  Each group is wrapped
    so that one failure (e.g. a locked DB) logs a warning and the rest
    still run — a seeding hiccup must never block GUI startup.

    Paths are interpreted relative to the current working directory (the
    FlexTool workspace root), matching :func:`update_flextool` and the
    GUI's ``get_projects_dir`` convention.

    Covers: the root input/settings SQLites and their ``templates/``
    copies, the canonical example DBs + XLSX templates, the user-facing
    ``example_input.xlsx``, the ``solver_config/<solver>.opt`` files,
    ``templates/project_folder.txt``, and ``results.sqlite``.

    It deliberately does NOT perform the update-only operations (git
    restore/pull, reinstall, the unconditional template refresh that
    keeps artifacts schema-current after a pull, parameter-definition
    back-fill into existing DBs, or DB-version migration) — those remain
    in :func:`update_flextool`.
    """
    master_json = str(package_data_path("schemas/spinedb_schema.json"))
    output_settings_json = str(package_data_path("schemas/output_settings_template.json"))
    output_info_json = str(package_data_path("schemas/output_info_template.json"))
    comparison_settings_json = str(package_data_path("schemas/comparison_settings_template.json"))

    os.makedirs("templates", exist_ok=True)
    os.makedirs("solver_config", exist_ok=True)

    # 1. Root input/settings DBs + their templates/ copies, from the
    #    bundled package JSON.  Created once; user edits then survive.
    try:
        for sqlite_rel, json_src in (
            ("input_data.sqlite", master_json),
            ("templates/input_data_template.sqlite", master_json),
            ("output_settings.sqlite", output_settings_json),
            ("output_info.sqlite", output_info_json),
            ("comparison_settings.sqlite", comparison_settings_json),
            ("templates/output_settings.sqlite", output_settings_json),
            ("templates/output_info.sqlite", output_info_json),
            ("templates/comparison_settings.sqlite", comparison_settings_json),
        ):
            if not os.path.exists(sqlite_rel):
                if verbose:
                    print(f"Seeding {sqlite_rel}")
                initialize_database(json_src, sqlite_rel)
    except Exception as exc:
        logging.warning("ensure_runtime_files: input/settings DB seeding failed: %s", exc)

    # 2. Canonical example DBs, the XLSX templates derived from them, and
    #    the user-facing example_input.xlsx — all create-if-missing.
    try:
        canonical_databases.materialize(overwrite=False)
        from flextool.export_to_tabular import export_to_excel
        for sqlite_rel, xlsx_rel in (
            ("templates/examples.sqlite", "templates/example_input_template.xlsx"),
            ("templates/input_data_template.sqlite", "templates/empty_input_template.xlsx"),
        ):
            if os.path.exists(sqlite_rel) and not os.path.exists(xlsx_rel):
                export_to_excel(f"sqlite:///{sqlite_rel}", xlsx_rel)
        if not os.path.exists("example_input.xlsx") and os.path.exists("templates/example_input_template.xlsx"):
            shutil.copy("templates/example_input_template.xlsx", "example_input.xlsx")
    except Exception as exc:
        logging.warning("ensure_runtime_files: example/template seeding failed: %s", exc)

    # 3. Live solver options — HiGHS + the four commercial solvers — each
    #    seeded from its bundled <solver>.opt.template if missing.  The
    #    runtime files are gitignored and user-editable (delete one to
    #    reset it to the bundled default).
    try:
        for _solver in ("highs", "gurobi", "cplex", "xpress", "copt"):
            _runtime = f"solver_config/{_solver}.opt"
            if not os.path.exists(_runtime):
                shutil.copy(
                    str(package_data_path(f"solver_config/{_solver}.opt.template")),
                    _runtime,
                )
    except Exception as exc:
        logging.warning("ensure_runtime_files: solver_config seeding failed: %s", exc)

    # 4. templates/project_folder.txt — a commented guide; its CONTENTS are
    #    load-bearing for Spine Toolbox (the run Tool reads the first
    #    non-comment line as the output project folder).  Gitignored.
    try:
        if not os.path.exists("templates/project_folder.txt"):
            with open("templates/project_folder.txt", "w", encoding="utf-8") as fh:
                fh.write(_PROJECT_FOLDER_TEMPLATE)
    except Exception as exc:
        logging.warning("ensure_runtime_files: project_folder.txt seeding failed: %s", exc)

    # 5. results.sqlite (and its template), both create-if-missing.
    try:
        result_template_path = str(package_data_path("schemas/pre_v26/flextool_template_results_master.json"))
        if not os.path.exists("templates/results_template.sqlite"):
            initialize_result_database("templates/results_template.sqlite", result_template_path)
        if not os.path.exists("results.sqlite") and os.path.exists("templates/results_template.sqlite"):
            shutil.copy("templates/results_template.sqlite", "results.sqlite")
    except Exception as exc:
        logging.warning("ensure_runtime_files: results DB seeding failed: %s", exc)


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

    # After (re)install, make sure the native solver stack actually runs on
    # this machine; auto-fix the known incompatible-build cases (polars CPU
    # baseline, highspy 1.14.0 on older Windows) if a component crashes.
    _ensure_compatible_solver_stack()

    migrate_project("./.spinetoolbox/project_temp.json","./.spinetoolbox/project.json")
    os.remove("./.spinetoolbox/project_temp.json")

    # Idempotently materialize all per-installation runtime files
    # (root/settings DBs + templates, canonical examples + XLSX,
    # example_input.xlsx, solver_config/*.opt, project_folder.txt,
    # results.sqlite).  Shared with the GUI startup path, so "an update
    # brought a new file to copy" is handled by the same create-if-missing
    # logic in both places — no duplication.
    ensure_runtime_files()

    # ----- Update-only refresh of EXISTING artifacts (after a git pull) ---
    # These differ from the create-if-missing seeding above: they bring
    # already-present files up to the current schema, so they must NOT run
    # on every GUI launch (that is why they stay here, not in
    # ensure_runtime_files).
    output_settings_json = str(package_data_path("schemas/output_settings_template.json"))
    output_info_json = str(package_data_path("schemas/output_info_template.json"))
    comparison_settings_json = str(package_data_path("schemas/comparison_settings_template.json"))

    # Back-fill parameter definitions added in newer versions into EXISTING
    # user settings DBs.  Options added later (e.g. output-spinedb) must be
    # added to DBs made by older versions.  Only schema is synced; user
    # parameter values are untouched.
    for sqlite_rel, json_src in (
        ("output_settings.sqlite", output_settings_json),
        ("output_info.sqlite", output_info_json),
        ("comparison_settings.sqlite", comparison_settings_json),
    ):
        if os.path.exists(sqlite_rel):
            _sync_settings_param_defs(json_src, sqlite_rel)

    # Keep CWD-resident template SQLites up-to-date (Spine Toolbox refs):
    # unconditional regen so a git pull's schema change is reflected.
    for sqlite_rel, json_src in (
        ("templates/output_settings.sqlite", output_settings_json),
        ("templates/output_info.sqlite", output_info_json),
        ("templates/comparison_settings.sqlite", comparison_settings_json),
    ):
        if os.path.exists(sqlite_rel):
            os.remove(sqlite_rel)
        initialize_database(json_src, sqlite_rel)

    # Refresh XLSX templates from the canonical SQLites so they track the
    # current schema (unconditional regen; the binaries are not in git).
    from flextool.export_to_tabular import export_to_excel
    for sqlite_rel, xlsx_rel in (
        ("templates/examples.sqlite", "templates/example_input_template.xlsx"),
        ("templates/input_data_template.sqlite", "templates/empty_input_template.xlsx"),
    ):
        if not os.path.exists(sqlite_rel):
            continue
        try:
            export_to_excel(f"sqlite:///{sqlite_rel}", xlsx_rel)
        except Exception as exc:
            print(f"Warning: failed to generate {xlsx_rel}: {exc}")

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

    # Refresh the results template (unconditional) and ensure results.sqlite
    # exists (ensure_runtime_files seeds it; this is a belt-and-braces copy
    # for the case it was removed since).
    result_template_path = str(package_data_path("schemas/pre_v26/flextool_template_results_master.json"))
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
