"""``flextool-project``: export a FlexTool project to one zip file, import one, or check one.

``flextool-project export <project> [-o FILE] [--results none|all|RUN,RUN] [--no-external]`` writes the inputs,
the portable settings, the plot settings, the workspace's solver option files and the chosen result runs, with a
manifest of versions and checksums.  ``flextool-project import FILE [--name NAME]`` unpacks it into ``projects/``
(exit code 3 when the name is taken).  ``flextool-project verify FILE`` checks the checksums and versions without
unpacking anything.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flextool import project_package as pp

EXIT_OK = 0
EXIT_PROBLEM = 1
EXIT_REFUSED = 2
EXIT_NAME_CLASH = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flextool-project",
        description="Export a FlexTool project to one .flextool.zip file, import one, or check one.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="Write a project package")
    export.add_argument("project", help="The project folder, or the name of a project in ./projects")
    export.add_argument("-o", "--output", help="The package file (default: ./<Name>_<YYYYMMDD>.flextool.zip)")
    export.add_argument("--results", default="none",
                        help="The result runs to include: none (default), all, or a comma-separated list of run "
                             "folders (e.g. base_1,coal_1)")
    export.add_argument("--no-external", action="store_true",
                        help="Do not copy the external input files into the package")
    export.add_argument("--solver-config-dir",
                        help="The folder of the solver .opt files (default: $FLEXTOOL_SOLVER_CONFIG_DIR, else "
                             "./solver_config)")

    imp = sub.add_parser("import", help="Unpack a project package into the projects folder")
    imp.add_argument("file", help="The .flextool.zip file")
    imp.add_argument("--name", help="The name of the new project (default: the name in the package)")
    imp.add_argument("--projects-dir", default="projects", help="The projects folder (default: ./projects)")

    verify = sub.add_parser("verify", help="Check a package's checksums and versions without unpacking it")
    verify.add_argument("file", help="The .flextool.zip file")
    return parser


def _project_dir(project: str) -> Path:
    path = Path(project)
    if path.is_dir():
        return path.resolve()
    return (Path.cwd() / "projects" / project).resolve()


def _mb(n: int) -> str:
    return f"{n / 1e6:.1f} MB"


def _export(args: argparse.Namespace) -> int:
    project_dir = _project_dir(args.project)
    if not project_dir.is_dir():
        print(f"No project folder at {project_dir}", file=sys.stderr)
        return EXIT_PROBLEM
    choice = args.results.strip()
    if choice == "none":
        runs: list[str] = []
    elif choice == "all":
        runs = [r.name for r in pp.list_runs(project_dir)]
    else:
        runs = [r.strip() for r in choice.split(",") if r.strip()]
    output = Path(args.output) if args.output else Path.cwd() / pp.default_package_name(project_dir.name)
    try:
        manifest = pp.export_project(
            project_dir, output, runs, include_external=not args.no_external,
            solver_config_dir=Path(args.solver_config_dir) if args.solver_config_dir else None,
        )
    except pp.PackageError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_PROBLEM
    size = sum(f["bytes"] for f in manifest.files.values())
    print(f"Wrote {output} ({len(manifest.files)} files, {_mb(size)} uncompressed).")
    print("Result runs: " + (", ".join(manifest.results) if manifest.results else "none"))
    if manifest.solver_config:
        print("Solver options: " + ", ".join(manifest.solver_config))
    for old, new in manifest.bundled_external.items():
        print(f"External input {old} copied in as input_sources/{new} (same source number).")
    for name in manifest.external_not_included:
        print(f"External input {name} is not in the package.")
    for run in manifest.stale_runs:
        print(f"Warning: the input of run {run} changed after the run; its results do not match the shipped "
              "input.")
    return EXIT_OK


def _import(args: argparse.Namespace) -> int:
    try:
        report = pp.import_project(args.file, Path(args.projects_dir), args.name)
    except pp.NameClashError as exc:
        print(f"{exc} Use --name {exc.suggestion}.", file=sys.stderr)
        return EXIT_NAME_CLASH
    except pp.PackageError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_REFUSED
    for line in pp.describe_import(report):
        print(line)
    return EXIT_OK


def _verify(args: argparse.Namespace) -> int:
    report = pp.verify_package(args.file)
    manifest = report.manifest
    if manifest is not None:
        print(f"Project {manifest.project}, created {manifest.created} with FlexTool {manifest.flextool} "
              f"(polar-high {manifest.polar_high}, highspy {manifest.highspy}, polars {manifest.polars}, "
              f"spinedb_api {manifest.spinedb_api}, Python {manifest.python}, {manifest.platform}).")
        for path, version in sorted(manifest.db_versions.items()):
            frozen = f", frozen ({manifest.frozen[path]})" if path in manifest.frozen else ""
            print(f"  {path}: data version {version}{frozen}")
        print("Result runs: " + (", ".join(manifest.results) if manifest.results else "none"))
        if manifest.solver_config:
            print("Solver options: " + ", ".join(manifest.solver_config)
                  + f" (unpacked to {pp.IMPORTED_SOLVER_CONFIG_DIR}/; copy them to the workspace's solver_config "
                    f"folder or set FLEXTOOL_SOLVER_CONFIG_DIR to use them)")
        if manifest.flextool:
            print(f"To run with the same FlexTool: pip install flextool=={manifest.flextool}")
    for warning in report.warnings:
        print(f"Warning: {warning}")
    for problem in report.problems:
        print(f"Problem: {problem}", file=sys.stderr)
    if report.ok:
        print(f"OK: {report.checked_files} files match their checksums. {pp.CHECKSUM_NOTE}")
        return EXIT_OK
    return EXIT_PROBLEM


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "export":
        return _export(args)
    if args.command == "import":
        return _import(args)
    return _verify(args)


if __name__ == "__main__":
    sys.exit(main())
