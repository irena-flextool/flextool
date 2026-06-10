"""Open the project's results database(s) in the standalone Spine DB editor.

This is the Toolbox-interop launcher Tool. It resolves the per-project
``results.sqlite`` at RUNTIME and launches the Spine DB editor on it,
DETACHED, so the Tool item completes immediately.

Two resolution modes (``--results-db-url`` overrides both):

* **From Output info (preferred):** ``--output-locations-db-url`` points at
  the Output-info DB — the SAME source Re-create results and Scenario
  comparison read.  Each scenario's ``scenario/output_location`` records where
  its outputs (and ``results.sqlite``) were written.  We collect every DISTINCT
  location and open every ``<location>/results.sqlite`` that exists — as tabs
  in a single Spine DB editor (its ``main`` accepts multiple positional URLs).
  Normally all scenarios share one project folder, so this is one DB.
* **From the project-folder redirect (fallback):** ``--project-folder-file``
  resolves ``<project>/results.sqlite`` via ``resolve_output_path`` (the 5-tier
  rule shared with the run Tool), for running this Tool without an Output-info
  link.

Launch mechanism (the only supported standalone entry point — the
``spine-db-editor`` console script, ``spinetoolbox.spine_db_editor.main:main``):
its ``main()`` parses positional ``url`` args, so we invoke

    <python> -m spinetoolbox.spine_db_editor.main sqlite:///<path> [sqlite:///<path> ...]

with the SAME interpreter that runs this script (``sys.executable``).
"""
import argparse
import subprocess
import sys
from pathlib import Path

from flextool.cli.cmd_run_flextool import resolve_output_path


def resolve_results_db_path(project_folder_file, output_location=None):
    """Resolve the filesystem path of the project's ``results.sqlite``.

    Reuses ``resolve_output_path`` (the 5-tier rule shared with the run
    Tool) to find the project/output root, then appends ``results.sqlite``.
    """
    output_root = resolve_output_path(
        input_db_url=None,
        flextool_location=None,
        output_location=output_location,
        cwd=Path.cwd(),
        project_folder_file=project_folder_file,
    )
    return Path(output_root) / "results.sqlite"


def output_locations_from_db(output_info_db_url):
    """Return the DISTINCT per-scenario ``output_location`` paths recorded in
    the Output-info DB, in first-seen order.

    Mirrors ``cmd_write_outputs``: scenarios come from the alternative filter
    applied to the DB (or, unfiltered, from every ``scenario`` entity); each
    scenario's ``scenario/output_location`` value names where its outputs were
    written.  The DB URL may be a ``sqlite:///`` path or a Spine DB-server
    ``http://`` URL (both open via ``DatabaseMapping``).
    """
    from spinedb_api import DatabaseMapping

    locations = []
    with DatabaseMapping(output_info_db_url) as db:
        filter_configs = db.get_filter_configs()
        scenario_names = []
        if filter_configs and filter_configs[0].get("alternatives"):
            scenario_names = list(filter_configs[0]["alternatives"])
        if not scenario_names:
            scenario_names = [
                e["name"]
                for e in db.get_entity_items(entity_class_name="scenario")
            ]
        for scenario_name in scenario_names:
            pv = db.get_parameter_value_item(
                entity_class_name="scenario",
                entity_byname=(scenario_name,),
                parameter_definition_name="output_location",
                alternative_name=scenario_name,
            )
            if not pv:
                continue
            loc = pv["parsed_value"]
            if loc and loc not in locations:
                locations.append(loc)
    return locations


def _to_sqlite_url(db_path):
    """Build a ``sqlite:///`` URL for an absolute/relative filesystem path."""
    return f"sqlite:///{Path(db_path).as_posix()}"


def _build_launch_argv(db_urls):
    """Build the argv that opens the standalone Spine DB editor.

    ``db_urls`` may be a single URL string or a list of URLs (each becomes a
    tab).  The supported standalone entry point is
    ``spinetoolbox.spine_db_editor.main:main``, whose ``main()`` reads
    positional ``url`` args.
    """
    if isinstance(db_urls, str):
        db_urls = [db_urls]
    return [
        sys.executable,
        "-m",
        "spinetoolbox.spine_db_editor.main",
        *db_urls,
    ]


def launch_db_editor(db_urls, _popen=subprocess.Popen):
    """Launch the Spine DB editor on ``db_urls`` (str or list), DETACHED.

    ``_popen`` is injectable for testing so the launch argv can be asserted
    without spawning a real GUI process.  Returns the spawned handle.
    """
    argv = _build_launch_argv(db_urls)
    # start_new_session detaches the child into its own process group so it
    # outlives this Tool process.  No ``.wait()`` — the Tool returns at once.
    return _popen(argv, start_new_session=True)


def _open_from_output_info(output_info_db_url):
    """Open every existing ``<output_location>/results.sqlite`` found via the
    Output-info DB, as tabs in one editor.  Returns a process exit code."""
    locations = output_locations_from_db(output_info_db_url)
    if not locations:
        print(
            "No scenario output locations found in the Output-info DB "
            f"({output_info_db_url}); run the workflow first.",
            file=sys.stderr,
        )
        return 1
    urls, missing = [], []
    for loc in locations:
        path = Path(loc) / "results.sqlite"
        if path.exists():
            urls.append(_to_sqlite_url(path))
        else:
            missing.append(path)
    if missing:
        print(
            "results.sqlite missing at: "
            + ", ".join(str(p) for p in missing)
            + " (those scenarios were not run with output-spinedb).",
            file=sys.stderr,
        )
    if not urls:
        print(
            "No results.sqlite found at any scenario output location; run "
            "the workflow with output-spinedb enabled first.",
            file=sys.stderr,
        )
        return 1
    print(f"Opening Spine DB editor on {len(urls)} results database(s): "
          + ", ".join(urls))
    launch_db_editor(urls)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Open the project's results.sqlite in the standalone Spine DB "
            "editor.  Preferred: resolve locations from the Output-info DB "
            "(--output-locations-db-url), the same source Re-create results "
            "uses, opening every scenario's results.sqlite as tabs.  "
            "Fallback: resolve <project>/results.sqlite from "
            "--project-folder-file."
        )
    )
    parser.add_argument(
        "--output-locations-db-url",
        metavar="URL",
        default=None,
        help=(
            "Output-info DB URL (sqlite:/// or http://). Opens every distinct "
            "scenario <output_location>/results.sqlite that exists, as tabs."
        ),
    )
    parser.add_argument(
        "--project-folder-file",
        metavar="PATH",
        default=None,
        help=(
            "Fallback: path to the user-local file whose CONTENTS name the "
            "project folder (templates/project_folder.txt).  results.sqlite is "
            "resolved at <project>/results.sqlite."
        ),
    )
    parser.add_argument(
        "--output-location",
        metavar="PATH",
        default=None,
        help="Explicit project/output root override (used by the fallback).",
    )
    parser.add_argument(
        "--results-db-url",
        metavar="URL_OR_PATH",
        default=None,
        help=(
            "Open this DB directly instead of resolving from Output info or "
            "the project folder.  Accepts a sqlite:/// URL or a bare path."
        ),
    )
    args = parser.parse_args(argv)

    # 1. Explicit --results-db-url short-circuits everything.
    if args.results_db_url:
        value = args.results_db_url
        if "://" in value:
            db_url = value
            db_path = (
                Path(value.replace("sqlite:///", "", 1))
                if value.startswith("sqlite:")
                else None
            )
        else:
            db_path = Path(value)
            db_url = _to_sqlite_url(db_path)
        if db_path is not None and not db_path.exists():
            print(
                f"results.sqlite not found at {db_path}; run the workflow with "
                f"output-spinedb enabled first (then re-run this Tool).",
                file=sys.stderr,
            )
            return 1
        print(f"Opening Spine DB editor on {db_url}")
        launch_db_editor(db_url)
        return 0

    # 2. Preferred: resolve locations from the Output-info DB (open all).
    if args.output_locations_db_url:
        return _open_from_output_info(args.output_locations_db_url)

    # 3. Fallback: the project-folder redirect → <project>/results.sqlite.
    db_path = resolve_results_db_path(
        args.project_folder_file, output_location=args.output_location
    )
    if not db_path.exists():
        print(
            f"results.sqlite not found at {db_path}; run the workflow with "
            f"output-spinedb enabled first (then re-run this Tool).",
            file=sys.stderr,
        )
        return 1
    db_url = _to_sqlite_url(db_path)
    print(f"Opening Spine DB editor on {db_url}")
    launch_db_editor(db_url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
