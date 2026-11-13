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

What the person reads is what gets trusted (item 240 final review M2, L5, L6):

- every line, file name and folder it shows goes through :func:`flextool._safe_text.printable`, so a control or
  bidirectional character in a received file (an escape sequence that moves the cursor up and erases a line) is
  shown as ``\\x1b`` / ``\\u202e`` and cannot hide a line; such a line is marked;
- the files are read once (:func:`flextool._local_solver_settings.show_folder`): the lines shown and the hashes the
  trust binds come from the same bytes, and the trust is written only when the folder still holds them at the
  answer (else nothing is trusted and the command says which files changed);
- it asks only when its input is a terminal: piped input (``echo y | ...``) and a program that starts it without
  one are refused.  There is no ``--yes``: a switch that answers for the person would be exactly the run flag D4
  keeps out of command lines that a received Spine Toolbox project carries, and a script that wants to trust a
  folder unasked can already run code as the user without it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flextool._safe_text import printable

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_ERROR = 2


def _p(value: object) -> str:
    """*value* as text that cannot act on the terminal (:func:`printable`)."""
    return printable(str(value))


def _stdin_is_terminal() -> bool:
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (AttributeError, ValueError, OSError):
        return False


def _option_files(folder: Path, contents: dict[str, bytes]) -> list[Path]:
    """The ``*.opt`` files of ``<folder>/solver_config`` that are not an unedited FlexTool default (judged on
    *contents*, the bytes shown, for the files a trust binds)."""
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
        data = contents.get(entry.name)
        if data is not None and is_unedited_template(entry.name[:-4], data):
            continue
        out.append(entry)
    return out


def _show_line(number: int, raw: bytes) -> None:
    try:
        text, note = raw.decode("utf-8"), ""
    except UnicodeDecodeError:
        text, note = raw.decode("utf-8", errors="backslashreplace"), "  (holds bytes that are not UTF-8 text)"
    shown = printable(text)
    if shown != text and not note:
        note = "  (holds control characters, shown as \\x.. / \\u....)"
    print(f"{number:4}  {shown}{note}")


def _show(path: Path, data: bytes | None) -> None:
    """Show the option file *path* with its lines: *data* is its content as the trust binds it; ``None`` for a
    link (never trusted by its content) or a file that could not be read."""
    from flextool._local_solver_settings import is_link_like

    link = is_link_like(path)
    print(f"--- {_p(path)}" + (" (a link: its target can change later, so it is never trusted by its content)"
                              if link else ""))
    if data is None:
        try:
            data = path.read_bytes()
        except OSError as exc:
            print(f"    cannot read it: {_p(exc)}")
            return
    lines = data.replace(b"\r\n", b"\n").split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    for number, raw in enumerate(lines, start=1):
        _show_line(number, raw)


def main(argv: list[str] | None = None) -> int:
    from flextool import _local_solver_settings as local

    parser = argparse.ArgumentParser(
        prog="flextool-trust-workspace",
        description=("Trust the solver option files of a FlexTool workspace or GUI project on this computer: their "
                     "file paths and server settings then reach the solver in runs that find the folder on their "
                     "own. Shows the files and asks first; run it in a terminal (piped input is refused)."),
    )
    parser.add_argument("folder", nargs="?", help="the workspace or project folder (the one holding solver_config/)")
    parser.add_argument("--remove", action="store_true", help="take the folder off this computer's list")
    parser.add_argument("--list", action="store_true", help="show the folders this computer trusts")
    args = parser.parse_args(argv)
    if args.list:
        try:
            for key in local.trusted_workspaces():
                print(_p(key))
        except local.LocalSolverSettingsError as exc:
            print(_p(exc), file=sys.stderr)
            return EXIT_ERROR
        return EXIT_OK
    if not args.folder:
        parser.error("give the workspace or project folder")
    folder = Path(args.folder).expanduser().absolute()
    if folder.name == "solver_config":
        folder = folder.parent
        print(f"Trust is given to the folder that holds solver_config/: {_p(folder)}")
    if args.remove:
        # Also for a folder that is gone: its stale entry can be taken off (item 240 final review L6).
        try:
            local.remove_trusted_workspace(folder)
        except local.LocalSolverSettingsError as exc:
            print(_p(exc), file=sys.stderr)
            return EXIT_ERROR
        print(f"{_p(folder)} is no longer trusted on this computer.")
        return EXIT_OK
    if not folder.is_dir():
        print(f"{_p(folder)} is not a folder.", file=sys.stderr)
        return EXIT_ERROR
    try:
        status = local.trust_status(folder)
    except local.LocalSolverSettingsError as exc:
        print(_p(exc), file=sys.stderr)
        return EXIT_ERROR
    if status.trusted and not status.changed_files(folder):
        print(f"{_p(folder)} is trusted on this computer already.")
        return EXIT_OK
    if not _stdin_is_terminal():
        print(f"Nothing was trusted: flextool-trust-workspace asks you to read the option files of {_p(folder)} and "
              "answer, and its input is not a terminal (piped, or started by a program). Run it in a terminal.",
              file=sys.stderr)
        return EXIT_REFUSED
    shown = local.show_folder(folder)
    files = _option_files(folder, dict(shown.contents))
    if status.problem:
        print(f"Not trusted now: {_p(status.problem)}.")
    for _name, why in status.changed_files(folder) if status.trusted else ():
        print(f"Not covered by the trust now: {_p(why)}.")
    if files:
        print(f"The option files of {_p(folder / 'solver_config')} that differ from FlexTool's defaults:")
        for path in files:
            _show(path, shown.contents.get(path.name))
        print("Once trusted, their file paths, server and licence settings and unlisted parameter names reach the "
              "solver. Trust them only if you made them or have read them. The trust is for this folder as it is "
              "now: a folder put at its path later is not trusted, and a later change to a file is trusted only "
              "when your account made it and no other account can change the file.")
    else:
        print(f"{_p(folder / 'solver_config')} holds no option file that differs from FlexTool's defaults; "
              "option files you put there later are trusted when your account writes them and no other account "
              "can change them.")
    try:
        answer = input(f"Trust the solver option files of {_p(folder)} on this computer? [y/N] ")
    except EOFError:
        answer = ""
    if answer.strip().casefold() not in ("y", "yes"):
        print("Nothing changed.")
        return EXIT_REFUSED
    try:
        local.update_settings_file(lambda data: local.apply_settings(
            data, workspace=folder, trust_workspace=True, workspace_shown=shown))
    except local.ShownFilesChanged as exc:
        print(_p(exc), file=sys.stderr)
        return EXIT_REFUSED
    except local.LocalSolverSettingsError as exc:
        print(_p(exc), file=sys.stderr)
        return EXIT_ERROR
    print(f"Trusted {_p(folder)} (in {_p(local.settings_path())}).")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
