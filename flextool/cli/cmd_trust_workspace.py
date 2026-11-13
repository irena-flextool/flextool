"""``flextool-trust-workspace <folder>``: trust a workspace's or a GUI project's solver option files on this computer.

The terminal twin of the FlexTool window's "Trust this workspace's / this project's solver option files" (item 240
B3, review b2b3 L6), for users who never open the window.  It shows the option files of ``<folder>/solver_config``
that differ from FlexTool's defaults, with their lines, and asks before it lists the folder in this computer's
``solver_settings.json`` (:mod:`flextool._local_solver_settings`), bound to the folder as it is now (its identity
and its files).  ``--remove`` takes the folder off the list; ``--list`` shows the list.

Why a command and not a run flag (design D4): a run flag could be carried by a Spine Toolbox project's tool
command line, so a received project could trust itself as it runs.  This is a separate command that a person types
and answers; it writes only this computer's settings file.  Something that can run it unasked (a Toolbox project,
a launcher script) can already run any program as the user, so the command opens no new way in.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_ERROR = 2


def _option_files(folder: Path) -> list[Path]:
    """The ``*.opt`` files of ``<folder>/solver_config`` that are not an unedited FlexTool default."""
    from flextool._solver_config_source import is_unedited_template

    config = folder / "solver_config"
    out = []
    try:
        entries = sorted(config.iterdir()) if config.is_dir() else []
    except OSError:
        return out
    for entry in entries:
        if not entry.name.casefold().endswith(".opt"):
            continue
        try:
            if is_unedited_template(entry.name[:-4], entry.read_bytes()):
                continue
        except OSError:
            pass
        out.append(entry)
    return out


def _show(path: Path) -> None:
    from flextool._local_solver_settings import is_link_like

    link = " (a link: its target can change later)" if is_link_like(path) else ""
    print(f"--- {path}{link}")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"    cannot read it: {exc}")
        return
    for number, line in enumerate(text.splitlines(), start=1):
        print(f"{number:4}  {line}")


def main(argv: list[str] | None = None) -> int:
    from flextool import _local_solver_settings as local

    parser = argparse.ArgumentParser(
        prog="flextool-trust-workspace",
        description=("Trust the solver option files of a FlexTool workspace or GUI project on this computer: their "
                     "file paths and server settings then reach the solver in runs that find the folder on their "
                     "own. Shows the files and asks first."),
    )
    parser.add_argument("folder", nargs="?", help="the workspace or project folder (the one holding solver_config/)")
    parser.add_argument("--remove", action="store_true", help="take the folder off this computer's list")
    parser.add_argument("--list", action="store_true", help="show the folders this computer trusts")
    args = parser.parse_args(argv)
    if args.list:
        try:
            for key in local.trusted_workspaces():
                print(key)
        except local.LocalSolverSettingsError as exc:
            print(exc, file=sys.stderr)
            return EXIT_ERROR
        return EXIT_OK
    if not args.folder:
        parser.error("give the workspace or project folder")
    folder = Path(args.folder).expanduser().absolute()
    if folder.name == "solver_config":
        folder = folder.parent
        print(f"Trust is given to the folder that holds solver_config/: {folder}")
    if not folder.is_dir():
        print(f"{folder} is not a folder.", file=sys.stderr)
        return EXIT_ERROR
    try:
        if args.remove:
            local.remove_trusted_workspace(folder)
            print(f"{folder} is no longer trusted on this computer.")
            return EXIT_OK
        status = local.trust_status(folder)
    except local.LocalSolverSettingsError as exc:
        print(exc, file=sys.stderr)
        return EXIT_ERROR
    files = _option_files(folder)
    if status.trusted and not status.changed_files(folder):
        print(f"{folder} is trusted on this computer already.")
        return EXIT_OK
    if status.problem:
        print(f"Not trusted now: {status.problem}.")
    for _name, why in status.changed_files(folder) if status.trusted else ():
        print(f"Not covered by the trust now: {why}.")
    if files:
        print(f"The option files of {folder / 'solver_config'} that differ from FlexTool's defaults:")
        for path in files:
            _show(path)
        print("Once trusted, their file paths, server and licence settings and unlisted parameter names reach the "
              "solver. Trust them only if you made them or have read them. The trust is for this folder as it is "
              "now: a folder put at its path later is not trusted, and a later change to a file is trusted only "
              "when your account made it and no other account can change the file.")
    else:
        print(f"{folder / 'solver_config'} holds no option file that differs from FlexTool's defaults; option "
              "files you put there later are trusted.")
    try:
        answer = input(f"Trust the solver option files of {folder} on this computer? [y/N] ")
    except EOFError:
        answer = ""
    if answer.strip().casefold() not in ("y", "yes"):
        print("Nothing changed.")
        return EXIT_REFUSED
    try:
        local.update_settings_file(lambda data: local.apply_settings(data, workspace=folder, trust_workspace=True))
    except local.LocalSolverSettingsError as exc:
        print(exc, file=sys.stderr)
        return EXIT_ERROR
    print(f"Trusted {folder} (in {local.settings_path()}).")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
