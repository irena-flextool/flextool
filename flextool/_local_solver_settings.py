"""Solver settings of this computer, never of a project or database (item 240 B1, B3).

``<user config folder>/solver_settings.json`` (:func:`flextool.user_dirs.user_config_dir`, ``"format": 1``) holds:

- ``allow_unlisted_parameters`` (``true`` / ``false``): whether solver parameter names that are not in FlexTool's
  list of the solver's parameters (:mod:`flextool.engine.solve._solver_catalogue`) are taken from input databases.
  Off by default.  ``FLEXTOOL_ALLOW_UNLISTED_SOLVER_PARAMETERS`` = ``1`` / ``0`` (also ``true`` / ``false``,
  ``yes`` / ``no``) wins over the file.
- ``trusted_workspaces`` (a list of folders): the workspaces this computer set up (B3).  A ``solver_config/``
  folder that a run finds on its own (next to the input database, in the workspace of a project database, in the
  Spine Toolbox project folder, in the current folder, or the FlexTool window's workspace) has its option files
  trusted (file paths, server settings) only when the folder it is in is listed.  ``ensure_runtime_files``
  (``flextool-update``, the FlexTool window's start) lists its workspace only when it creates ``solver_config/``
  there or finds nothing in it but FlexTool's own default files; the FlexTool window's "Trust this workspace"
  setting lists it on the user's word.  The list is outside every workspace, so a file that came with a
  workspace cannot put it there.

Not in the database, a project folder, ``settings.yaml`` or ``projects.yaml``: those travel with a project or a
workspace.  No command-line flag either: a Spine Toolbox project carries tool command lines and can come from
someone else.  The value is read when it is needed (an unlisted name from a database), so a damaged file stops
only a run that depends on it, with a message naming it.

Only the standard library and :mod:`flextool.user_dirs` are imported.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

ALLOW_UNLISTED_ENV = "FLEXTOOL_ALLOW_UNLISTED_SOLVER_PARAMETERS"
SETTINGS_FILE_NAME = "solver_settings.json"
FORMAT = 1
TRUSTED_WORKSPACES_KEY = "trusted_workspaces"
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


class LocalSolverSettingsError(ValueError):
    """The environment variable or the settings file holds a value FlexTool cannot read."""


@dataclass(frozen=True)
class AllowUnlisted:
    """Whether unlisted solver parameter names pass from databases, and where that was set."""

    value: bool
    source: str  # ``environment variable FLEXTOOL_...``, the settings file's path, or ``default``

    def record(self) -> dict[str, object]:
        return {"value": self.value, "source": self.source}


def settings_path() -> Path:
    """This computer's solver settings file (it need not exist)."""
    from flextool.user_dirs import user_config_dir

    return user_config_dir() / SETTINGS_FILE_NAME


def read_settings_file(path: Path | None = None) -> dict:
    """The settings file's content (``{}`` when there is none); raises :class:`LocalSolverSettingsError` when it
    is not a FlexTool solver settings file of a format this FlexTool reads."""
    path = settings_path() if path is None else path
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise LocalSolverSettingsError(f"FlexTool cannot read its solver settings file {path} ({exc}).") from exc
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise LocalSolverSettingsError(
            f"FlexTool's solver settings file {path} is not valid JSON ({exc}). Correct or delete it.") from exc
    if not isinstance(data, dict):
        raise LocalSolverSettingsError(f"FlexTool's solver settings file {path} does not hold a JSON object.")
    fmt = data.get("format", FORMAT)
    if not isinstance(fmt, int) or isinstance(fmt, bool) or fmt > FORMAT:
        raise LocalSolverSettingsError(
            f"FlexTool's solver settings file {path} has format {fmt!r}, written by a newer FlexTool; this "
            f"FlexTool reads format {FORMAT}. Update FlexTool, or delete the file.")
    return data


def allow_unlisted_parameters(environ: Mapping[str, str] | None = None) -> AllowUnlisted:
    """The allow switch for unlisted solver parameter names from databases (environment first, then the
    settings file, else off)."""
    env = os.environ if environ is None else environ
    raw = env.get(ALLOW_UNLISTED_ENV, "").strip()
    if raw:
        word = raw.casefold()
        if word not in _TRUE | _FALSE:
            raise LocalSolverSettingsError(
                f"{ALLOW_UNLISTED_ENV}={raw!r} is not 1 or 0. Set it to 1 to allow solver parameter names that "
                "are not in FlexTool's list from input databases, or to 0 (or unset it) to refuse them.")
        return AllowUnlisted(word in _TRUE, f"environment variable {ALLOW_UNLISTED_ENV}")
    path = settings_path()
    data = read_settings_file(path)
    if "allow_unlisted_parameters" not in data:
        return AllowUnlisted(False, "default")
    value = data["allow_unlisted_parameters"]
    if not isinstance(value, bool):
        raise LocalSolverSettingsError(
            f"FlexTool's solver settings file {path}: allow_unlisted_parameters is {value!r}; write true or false.")
    return AllowUnlisted(value, str(path))


def allow_unlisted_record(environ: Mapping[str, str] | None = None) -> dict[str, object]:
    """The run record's ``allow_unlisted_parameters``: ``{"value": bool, "source": ...}``; an unreadable setting
    gives ``{"value": None, "source": <the message>}`` (the run stops only if it needs the setting)."""
    try:
        return allow_unlisted_parameters(environ).record()
    except LocalSolverSettingsError as exc:
        return {"value": None, "source": str(exc)}


def update_settings_file(change: Callable[[dict], None], path: Path | None = None) -> None:
    """Read the settings file, let *change* edit its content in place, and write it back (under its lock, through a
    temporary file).  A damaged file, or one of a newer format, is not replaced: :class:`LocalSolverSettingsError`."""
    from flextool.assistant._files import LockTimeout, locked, write_text_atomic

    path = settings_path() if path is None else path
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with locked(path):
            data = dict(read_settings_file(path))
            change(data)
            data["format"] = FORMAT
            write_text_atomic(path, json.dumps(data, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    except LockTimeout as exc:
        raise LocalSolverSettingsError(f"FlexTool's solver settings file {path}: {exc}") from exc
    except OSError as exc:
        raise LocalSolverSettingsError(f"FlexTool cannot write its solver settings file {path} ({exc}).") from exc


# --- workspaces this computer set up (B3) --------------------------------------------------------------------


def workspace_key(folder: str | os.PathLike) -> str:
    """How a workspace is listed: its absolute path with symbolic links resolved, in the case the platform
    compares (``os.path.normcase``: lower case on Windows)."""
    try:
        resolved = Path(folder).resolve()
    except (OSError, RuntimeError):
        resolved = Path(os.path.abspath(folder))
    return os.path.normcase(str(resolved))


def trusted_workspaces(path: Path | None = None) -> tuple[str, ...]:
    """The workspaces this computer set up, as listed; raises :class:`LocalSolverSettingsError` for a damaged file
    or a value that is not a list of folder names."""
    path = settings_path() if path is None else path
    value = read_settings_file(path).get(TRUSTED_WORKSPACES_KEY, [])
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise LocalSolverSettingsError(
            f"FlexTool's solver settings file {path}: {TRUSTED_WORKSPACES_KEY} is not a list of folder names.")
    return tuple(value)


def is_trusted_workspace(folder: str | os.PathLike, path: Path | None = None) -> bool:
    """Whether *folder* is a workspace this computer set up: listed by :func:`workspace_key`, or the same folder on
    disk as a listed one (macOS keeps a path's case as typed although its disks compare without case).  Raises
    :class:`LocalSolverSettingsError` when the settings file cannot be read."""
    listed = trusted_workspaces(path)
    if not listed:
        return False
    key = workspace_key(folder)
    if key in listed:
        return True
    for entry in listed:
        try:
            if os.path.samefile(entry, folder):
                return True
        except OSError:
            continue
    return False


def add_trusted_workspace(folder: str | os.PathLike, path: Path | None = None) -> bool:
    """List *folder* as a workspace this computer set up; ``False`` when it already was."""
    key = workspace_key(folder)
    added = False

    def change(data: dict) -> None:
        nonlocal added
        current = data.get(TRUSTED_WORKSPACES_KEY, [])
        current = [item for item in current if isinstance(item, str) and item] if isinstance(current, list) else []
        if key not in current:
            current.append(key)
            added = True
        data[TRUSTED_WORKSPACES_KEY] = current

    update_settings_file(change, path)
    return added


def remove_trusted_workspace(folder: str | os.PathLike, path: Path | None = None) -> None:
    """Take *folder* off the list of workspaces this computer set up."""
    key = workspace_key(folder)

    def change(data: dict) -> None:
        current = data.get(TRUSTED_WORKSPACES_KEY, [])
        current = current if isinstance(current, list) else []
        kept = []
        for item in current:
            if not isinstance(item, str) or item == key:
                continue
            try:
                if os.path.samefile(item, folder):
                    continue
            except OSError:
                pass
            kept.append(item)
        data[TRUSTED_WORKSPACES_KEY] = kept

    update_settings_file(change, path)
