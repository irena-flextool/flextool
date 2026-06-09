"""Open the project's results database in the standalone Spine DB editor.

This is the Toolbox-interop launcher Tool: it resolves the per-project
``results.sqlite`` at RUNTIME (a committed Data Store URL cannot follow
the user-local ``templates/project_folder.txt`` redirect that the FlexTool
run Tool reads), then launches the Spine DB editor on it, DETACHED, so the
Tool item completes immediately.

Project-folder resolution is NOT duplicated here: it reuses
``resolve_output_path`` / ``_read_project_folder_file`` from
``cmd_run_flextool`` so this launcher and the run Tool always agree on
where the project (and therefore ``results.sqlite``) lives.

Launch mechanism (the only supported standalone entry point — see the
``spine-db-editor`` console_scripts entry in Spine Toolbox's
``pyproject.toml``, ``spinetoolbox.spine_db_editor.main:main``): the
editor's ``main()`` parses positional ``url`` args, so we invoke

    <python> -m spinetoolbox.spine_db_editor.main sqlite:///<path>

with the SAME interpreter that runs this script (``sys.executable``),
guaranteeing the editor resolves against the active environment.
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

    Args:
        project_folder_file: path to the user-local file whose contents
            name the project folder (``templates/project_folder.txt``).
        output_location: optional explicit override (tier 1).

    Returns:
        pathlib.Path: ``<project>/results.sqlite``.
    """
    output_root = resolve_output_path(
        input_db_url=None,
        flextool_location=None,
        output_location=output_location,
        cwd=Path.cwd(),
        project_folder_file=project_folder_file,
    )
    return Path(output_root) / "results.sqlite"


def _to_sqlite_url(db_path):
    """Build a ``sqlite:///`` URL for an absolute/relative filesystem path."""
    # ``Path.as_posix`` keeps forward slashes on every platform, which is
    # what the SQLAlchemy ``sqlite:///`` form expects.
    return f"sqlite:///{Path(db_path).as_posix()}"


def _build_launch_argv(db_url):
    """Build the argv that opens the standalone Spine DB editor on ``db_url``.

    The supported standalone entry point is
    ``spinetoolbox.spine_db_editor.main:main`` (the ``spine-db-editor``
    console script).  Its ``main()`` reads positional ``url`` args, so we
    run the module with ``-m`` using the current interpreter.
    """
    return [
        sys.executable,
        "-m",
        "spinetoolbox.spine_db_editor.main",
        db_url,
    ]


def launch_db_editor(db_url, _popen=subprocess.Popen):
    """Launch the Spine DB editor on ``db_url``, DETACHED (non-blocking).

    ``_popen`` is injectable for testing so the launch argv can be
    asserted without spawning a real GUI process.

    Returns the spawned process handle (or whatever ``_popen`` returns).
    """
    argv = _build_launch_argv(db_url)
    # start_new_session detaches the child into its own process group so it
    # outlives this Tool process; stdin/out/err are left at their defaults
    # (the GUI manages its own).  No ``.wait()`` — the Tool returns at once.
    return _popen(argv, start_new_session=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Open the project's results.sqlite in the standalone Spine DB "
            "editor.  Resolves the project folder from "
            "--project-folder-file (the same user-local redirect the "
            "FlexTool run Tool uses), so the editor follows wherever the "
            "user pointed their outputs."
        )
    )
    parser.add_argument(
        "--project-folder-file",
        metavar="PATH",
        default=None,
        help=(
            "Path to the user-local file whose CONTENTS name the project "
            "folder (templates/project_folder.txt).  results.sqlite is "
            "resolved at <project>/results.sqlite."
        ),
    )
    parser.add_argument(
        "--output-location",
        metavar="PATH",
        default=None,
        help="Explicit project/output root override (highest precedence).",
    )
    parser.add_argument(
        "--results-db-url",
        metavar="URL_OR_PATH",
        default=None,
        help=(
            "Open this DB directly instead of resolving "
            "<project>/results.sqlite.  Accepts a sqlite:/// URL or a bare "
            "filesystem path."
        ),
    )
    args = parser.parse_args(argv)

    # Explicit --results-db-url short-circuits resolution.  A bare path is
    # normalised to a sqlite:/// URL; a value already carrying a scheme is
    # used verbatim (and its on-disk existence is only checked for sqlite).
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
    else:
        db_path = resolve_results_db_path(
            args.project_folder_file, output_location=args.output_location
        )
        db_url = _to_sqlite_url(db_path)

    # Missing-file branch: report clearly and exit WITHOUT launching.  Only
    # checked when we have a local filesystem path (sqlite / resolved).
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


if __name__ == "__main__":
    sys.exit(main())
