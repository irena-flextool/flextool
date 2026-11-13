import json
import logging
import os
import subprocess
import shutil
from pathlib import Path
try: 
    from spinedb_api import import_data
    from flextool.update_flextool.db_handle import open_database_mapping
except ModuleNotFoundError:
    exit("Cannot find the required Spine-Toolbox module. Check that the environment is activated and the toolbox is installed")
from spinedb_api.exception import NothingToCommit
from flextool._child_python import pip_env, python_module_argv
from flextool._resources import package_data_path
from flextool.update_flextool.db_migration import migrate_database
from flextool.update_flextool.initialize_database import initialize_database
from flextool.update_flextool import canonical_databases
from flextool.sqlite_url import sqlite_url_string


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
    import os

    # flextool/update_flextool/self_update.py  →  flextool/  →  repo root.
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    )))

    result = subprocess.run(
        python_module_argv("pip", "show", "flextool"),
        capture_output=True, text=True, env=pip_env(),
    )
    is_editable = "Editable project location" in result.stdout

    print(
        f"Refreshing flextool install and dependencies "
        f"({'editable' if is_editable else 'non-editable'}, target={repo_root})..."
    )
    target = ["-e", repo_root] if is_editable else [repo_root]
    completed = subprocess.run(
        # -P only, no PYTHONSAFEPATH (pip_env): pip runs the setup scripts of the packages it builds, which may need
        # their own folder on sys.path.
        python_module_argv("pip", "install", "--upgrade", *target),
        env=pip_env(),
    )
    if completed.returncode != 0:
        flag = "-e " if is_editable else ""
        print(
            f"Warning: pip install failed.  Run "
            f"'pip install --upgrade {flag}{repo_root}' manually."
        )
    if is_editable:
        _keep_bundle_relocatable()


def _keep_bundle_relocatable():
    """In the Windows Toolbox bundle, point the editable install to the clone by a relative path again.

    ``pip install -e`` writes the clone's absolute path; the bundle must keep
    working after it is moved.  Does nothing outside the bundle.
    """
    from flextool.update_flextool.bundle_install import relativise_bundle_install

    try:
        relative = relativise_bundle_install()
    except (OSError, RuntimeError) as exc:
        print(f"Warning: could not make the FlexTool editable install relative: {exc}")
        return
    if relative is not None:
        print(f"FlexTool editable install made relative: {relative}")


def _run_fix_steps(steps) -> bool:
    """Run pip *steps* in order, echoing each; stop at the first failure."""
    for step in steps:
        print("\n$ " + " ".join(step))
        completed = subprocess.run(step, env=pip_env())
        if completed.returncode != 0:
            print(
                f"Warning: '{' '.join(step)}' exited {completed.returncode}. "
                f"Run it manually to finish the fix."
            )
            return False
    return True


def _ensure_compatible_solver_stack():
    """Verify the native solver stack (polars, highspy/HiGHS) runs on this
    machine and, if a component crashes natively, re-install the compatible
    build automatically.

    Two known cases (see ``flextool.env_check``): the default ``polars``
    runtime on a CPU without AVX2/FMA/BMI2/MOVBE (cure: add the compatible
    runtime, ``polars[rtcompat]``), and an old ``highspy`` (1.14.0)
    import-crashing on older Windows (cure: highspy 1.15 or later, which
    ships its own C++ runtime).  Both surface as a native
    crash — a Windows access violation (exit ``3221225477``) or a POSIX
    ``SIGILL``/``SIGSEGV``.  The CPU flags are read first, so the compatible
    polars runtime goes in before polars ever crashes; then the imports are
    probed in a child process so a crash is observable and the component
    that died is pinpointed and its remedy applied.

    Run unattended here (the user already opted into installs by choosing
    "Update FlexTool"), but loudly: the banner names what is happening and
    how to stop it, and everything prints to stdout so it lands in the
    update log window for copy-paste.
    """
    import sys
    from flextool import env_check

    print("\nChecking that the solver libraries run on this computer...")
    verdict = env_check.cpu_verdict()
    if verdict.status == env_check.CPU_UNSUPPORTED:
        print("\n" + env_check.unsupported_cpu_help(verdict))
        return
    if env_check.proactive_polars_fix_needed():
        print("\n" + env_check.proactive_banner(verdict))
        if not _run_fix_steps(env_check.polars_compat_steps(sys.executable)):
            return

    probe = env_check.probe_solver_stack()
    print(env_check.diagnostics_report(probe))
    if not probe.is_native_fault:
        # OK, or an ordinary error (e.g. a package missing) that a build
        # swap would not fix — leave it for the normal install machinery.
        return

    steps = env_check.remediation_steps(probe.failed_component, sys.executable)
    if steps is None:
        # A native crash with no package-level remedy (e.g. polar_high, or a
        # compatible polars runtime that is already in place).
        print("\n" + env_check.unfixable_help())
        return

    print("\n" + env_check.remediation_banner(probe.failed_component))
    if not _run_fix_steps(steps):
        return

    reprobe = env_check.probe_solver_stack()
    print("\n" + reprobe.summary())
    if not reprobe.ok:
        print("\n" + env_check.unfixable_help())


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
        with open(json_template, encoding="utf-8") as json_file:
            template = json.load(json_file)
        payload = {
            key: template[key]
            for key in ("entity_classes", "parameter_definitions", "parameter_types")
            if key in template
        }
        if not payload:
            return
        with open_database_mapping(sqlite_url_string(sqlite_path), create=False) as db:
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
    "# e.g.:  projects/MyProject\n"
)


def _refresh_solver_config(folder: Path, *, verbose: bool = False, warn: bool = False) -> None:
    """Seed each ``<folder>/<solver>.opt`` from its packaged template when missing, and replace one that is an
    unedited copy of an EARLIER FlexTool template with the current one (item 233).

    An edited file (any byte differs, CRLF read as LF) and a copy of the current template are left alone, so in
    the steady state nothing is written.  The template's bytes are copied without its file permissions (a
    read-only installed template must not give a read-only option file), to ``<solver>.opt.tmp-<pid>``, given the
    replaced file's permissions (a file the user made private stays private) and then moved onto the file.  A
    failure keeps the file: logged at INFO, or as a warning when *warn* (``flextool-update``; at every FlexTool
    window start a locked workspace would warn every time).

    A symbolic link is never seeded through or replaced, whatever it points to: it is the user's arrangement (an
    option file shared by a team or kept elsewhere), its target may be outside the workspace or used by others,
    and replacing the link would cut it off.  Replacing an unedited earlier default changes comments only, so
    keeping the link costs nothing.
    """
    from flextool._solver_config_source import SOLVER_NAMES, _normalised_sha256, is_unedited_template

    say = logging.warning if warn else logging.info
    for solver in SOLVER_NAMES:
        runtime = folder / f"{solver}.opt"
        if runtime.is_symlink():
            continue
        try:
            template = Path(package_data_path(f"solver_config/{solver}.opt.template"))
            if not runtime.exists():
                if verbose:
                    print(f"Seeding {runtime}")
                shutil.copyfile(template, runtime)
                continue
            data = runtime.read_bytes()
            current = template.read_bytes()
        except OSError as exc:
            say("Could not seed or read %s (%s); it is left as it is.", runtime, exc)
            continue
        if not is_unedited_template(solver, data) or _normalised_sha256(data) == _normalised_sha256(current):
            continue
        tmp = folder / f"{solver}.opt.tmp-{os.getpid()}"
        try:
            shutil.copyfile(template, tmp)
            shutil.copymode(runtime, tmp)
            os.replace(tmp, runtime)
        except OSError as exc:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            say("Could not update %s to FlexTool's current default (%s); the file is kept.", runtime, exc)
            continue
        msg = f"Updated {runtime.as_posix()} to FlexTool's current default (it was an unedited earlier default)."
        logging.info(msg)
        if verbose:
            print(msg)


def _list_workspace(solver_config_is_own: bool, *, warn: bool = False) -> None:
    """List the current folder as a workspace this computer set up when its ``solver_config/`` was missing or held
    nothing but FlexTool's own default files before this call (item 240 B3).  Otherwise, unless it is listed
    already, say once why it is not (a warning for ``flextool-update``, an INFO line at the window's start)."""
    from flextool._local_solver_settings import (
        LocalSolverSettingsError, add_trusted_workspace, is_trusted_workspace, settings_path,
    )

    workspace = Path.cwd()
    try:
        if solver_config_is_own:
            if add_trusted_workspace(workspace):
                logging.info("Listed %s as a workspace this computer set up (in %s).", workspace, settings_path())
            return
        if is_trusted_workspace(workspace):
            return
    except LocalSolverSettingsError as exc:
        logging.warning("%s The workspace %s is not listed as set up by this computer.", exc, workspace)
        return
    (logging.warning if warn else logging.info)(
        "%s holds solver option files that FlexTool did not write, so this workspace is not marked as set up by "
        "this computer: runs that find these files on their own give them the input database's rules (no file "
        "paths outside the work folder, no server settings, only listed parameter names). If you made them, or "
        "have read them and trust them, tick \"Trust this workspace's solver option files\" in the FlexTool "
        "window (Solver options, On this computer), run flextool-trust-workspace %s, or pass the folder with "
        "--solver-config-dir (or FLEXTOOL_SOLVER_CONFIG_DIR).",
        workspace / "solver_config", workspace,
    )


def ensure_runtime_files(verbose: bool = False, *, warn: bool = False) -> None:
    """Idempotently materialize the per-installation runtime files that
    FlexTool needs but does not track in git.

    Every step is **create-if-missing**, except that a
    ``solver_config/<solver>.opt`` that is an unedited copy of an earlier
    FlexTool default is replaced by the current one (item 233,
    :func:`_refresh_solver_config`): an existing user-edited file is never
    overwritten, so this is safe to call on every GUI launch *and* as the
    seeding step of :func:`update_flextool` (which passes *warn*, so a
    file it cannot replace is a warning there and an INFO line at the
    GUI start).  Each group is wrapped
    so that one failure (e.g. a locked DB) logs a warning and the rest
    still run — a seeding hiccup must never block GUI startup.

    Paths are interpreted relative to the current working directory (the
    FlexTool workspace root), matching :func:`update_flextool` and the
    GUI's ``get_projects_dir`` convention.

    Covers: the root input/settings SQLites and their ``templates/``
    copies, the canonical example DBs + XLSX templates, the user-facing
    ``example_input.xlsx``, the ``solver_config/<solver>.opt`` files,
    ``templates/project_folder.txt``, and ``results.sqlite``.

    It also lists the workspace as one this computer set up (item 240 B3, :func:`_list_workspace`), but only when
    ``solver_config/`` was missing or held nothing but FlexTool's own default files: a received workspace with
    edited option files is not listed by being opened.

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
    # Item 240 B3: before anything is seeded, whether solver_config/ is missing or holds nothing but FlexTool's
    # own default files.  Only then is this workspace listed as set up by this computer (step 3b).
    from flextool._solver_config_source import holds_only_flextool_defaults
    solver_config_is_own = holds_only_flextool_defaults(Path("solver_config"))
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
                export_to_excel(sqlite_url_string(sqlite_rel), xlsx_rel)
        if not os.path.exists("example_input.xlsx") and os.path.exists("templates/example_input_template.xlsx"):
            shutil.copy("templates/example_input_template.xlsx", "example_input.xlsx")
    except Exception as exc:
        logging.warning("ensure_runtime_files: example/template seeding failed: %s", exc)

    # 3. Live solver options — HiGHS + the four commercial solvers — each
    #    seeded from its bundled <solver>.opt.template if missing, and an
    #    unedited copy of an earlier template replaced by the current one.
    #    The runtime files are gitignored and user-editable (delete one to
    #    reset it to the bundled default).
    try:
        _refresh_solver_config(Path("solver_config"), verbose=verbose, warn=warn)
    except Exception as exc:
        logging.warning("ensure_runtime_files: solver_config seeding failed: %s", exc)

    # 3b. Item 240 B3: list this workspace as one this computer set up (in the user config folder, outside the
    #     workspace), so the option files of its solver_config/ are trusted when a run finds them on its own.
    try:
        _list_workspace(solver_config_is_own, warn=warn)
    except Exception as exc:
        logging.warning("ensure_runtime_files: listing the workspace failed: %s", exc)

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
        result_template_path = str(package_data_path("schemas/spinedb_results_schema.json"))
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
    # baseline, a highspy older than 1.15 on older Windows) if a component crashes.
    _ensure_compatible_solver_stack()

    migrate_project("./.spinetoolbox/project_temp.json","./.spinetoolbox/project.json")
    os.remove("./.spinetoolbox/project_temp.json")

    # Idempotently materialize all per-installation runtime files
    # (root/settings DBs + templates, canonical examples + XLSX,
    # example_input.xlsx, solver_config/*.opt, project_folder.txt,
    # results.sqlite).  Shared with the GUI startup path, so "an update
    # brought a new file to copy" is handled by the same create-if-missing
    # logic in both places — no duplication.  ``warn``: a solver_config/*.opt
    # that cannot be brought to the current default is a warning here.
    ensure_runtime_files(warn=True)

    # ----- Update-only refresh of EXISTING artifacts (after a git pull) ---
    # These differ from the seeding above: they bring already-present files
    # up to the current schema, so they must NOT run on every GUI launch
    # (that is why they stay here, not in ensure_runtime_files).  The one
    # refresh ensure_runtime_files does run, of unedited solver_config/*.opt
    # copies, writes nothing once the files are current.
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
            export_to_excel(sqlite_url_string(sqlite_rel), xlsx_rel)
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

    # migrate the databases to new version; a frozen database older than
    # this FlexTool is skipped and listed (migrate a copy of it with
    # ``flextool-migrate-database DB --copy-to NEW.sqlite``).
    from flextool.update_flextool.frozen import FrozenDatabaseError
    skipped_frozen: list[str] = []
    for i in db_to_update:
        try:
            migrate_database(i)
        except FrozenDatabaseError as exc:
            skipped_frozen.append(str(exc))
    if skipped_frozen:
        print("These frozen databases were not migrated:")
        for line in skipped_frozen:
            print("  " + line)

    # Refresh the results template (unconditional) and ensure results.sqlite
    # exists (ensure_runtime_files seeds it; this is a belt-and-braces copy
    # for the case it was removed since).
    result_template_path = str(package_data_path("schemas/spinedb_results_schema.json"))
    if os.path.exists('templates/results_template.sqlite'):
        os.remove('templates/results_template.sqlite')
    initialize_result_database('templates/results_template.sqlite', result_template_path)

    if not os.path.exists("results.sqlite"):
        shutil.copy("templates/results_template.sqlite", "results.sqlite")
    #update result parameter definitions
    #get template JSON. This can be the master or old template if conflicting migrations in between
    with open (result_template_path) as json_file:
        template = json.load(json_file)
    with open_database_mapping(sqlite_url_string('results.sqlite'), create = False, upgrade = True) as db:
        #these update the old descriptions, but wont remove them or change names (the new name is created, but old stays)
        (num,log) = import_data(db, parameter_definitions = template["parameter_definitions"])
        try:
            db.commit_session("Updated relationship_parameters, object parameters to the Results.sqlite")
        except NothingToCommit:
            print("These parameters have been added before, continuing")
        return 0

def initialize_result_database(filename,json_path):

    with open (json_path) as json_file:
        template = json.load(json_file)
    with open_database_mapping(sqlite_url_string(filename), create = True) as db:
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

    with open(old_path, encoding="utf-8") as old_json:
        old_dict = json.load(old_json)
    with open(new_path, encoding="utf-8") as new_json:
        new_dict = json.load(new_json)
    
    for item in items:
        if item in old_dict["items"].keys():
            for param in old_dict["items"][item].keys():
                if param != "x" and param != "y":
                    old_dict["items"][item][param] = new_dict["items"][item][param]

    # The Open_summary item of old projects ran open_summary.py, which no longer exists, and its specification file
    # is gone: drop the item, its connections and the specification entry.
    old_dict["items"].pop("Open_summary", None)
    project = old_dict.get("project", {})
    project["connections"] = [c for c in project.get("connections", [])
                              if "Open_summary" not in (c.get("from", [None])[0], c.get("to", [None])[0])]
    tool_specs = project.get("specifications", {}).get("Tool")
    if tool_specs is not None:
        project["specifications"]["Tool"] = [
            spec for spec in tool_specs if not str(spec.get("path", "")).endswith("specifications/Tool/open_summary.json")
        ]

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
