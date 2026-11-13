"""Solver settings of this computer, never of a project or database (item 240 B1, B2, B3).

``<user config folder>/solver_settings.json`` (:func:`flextool.user_dirs.user_config_dir`, ``"format": 1``) holds:

- ``allow_unlisted_parameters`` (``true`` / ``false``): whether solver parameter names that are not in FlexTool's
  list of the solver's parameters (:mod:`flextool.engine.solve._solver_catalogue`) are taken from input databases.
  Off by default.  ``FLEXTOOL_ALLOW_UNLISTED_SOLVER_PARAMETERS`` = ``1`` / ``0`` (also ``true`` / ``false``,
  ``yes`` / ``no``) wins over the file.
- ``trusted_workspaces`` (a list of folders): the workspaces and GUI projects this computer set up (B3).  A
  ``solver_config/`` folder that a run finds on its own (next to the input database, in the project or workspace
  of a project database, in the Spine Toolbox project folder, in the current folder, or the FlexTool window's
  project and workspace) has its option files trusted (file paths, server settings) only when the folder it is
  in is listed.  ``ensure_runtime_files`` (``flextool-update``, the FlexTool window's start) lists its workspace
  only when it creates ``solver_config/`` there or finds nothing in it but FlexTool's own default files; the
  FlexTool window lists a project when it creates it or opens it in that state with no received option file
  waiting for review (:func:`flextool._solver_config_source.note_project_set_up_here`), and takes a project off
  the list when it is deleted or a package is imported at its path.  The window's "Trust this workspace's / this
  project's solver option files" settings list them on the user's word.  The list is outside every workspace,
  so a file that came with a workspace or a project cannot put it there.
- ``wrappers`` (``{"cplex": ["/usr/local/bin/lic_hold", "--queue", "opt"], ...}``): a program that starts in
  front of a commercial solver's program (B2), for example to hold a floating licence or to lower the solver's
  priority.  ``FLEXTOOL_SOLVER_WRAPPER_<SOLVER>`` wins over the file: a JSON list, or one program path (never
  split); empty or ``[]`` means no wrapper.  :func:`check_wrapper` holds the rules.

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
WRAPPERS_KEY = "wrappers"
#: The solvers whose program can run through a wrapper (HiGHS runs inside FlexTool).
WRAPPER_SOLVERS: tuple[str, ...] = ("gurobi", "cplex", "xpress", "copt")
#: The environment variable of each solver's wrapper.
WRAPPER_ENV: dict[str, str] = {
    "gurobi": "FLEXTOOL_SOLVER_WRAPPER_GUROBI",
    "cplex": "FLEXTOOL_SOLVER_WRAPPER_CPLEX",
    "xpress": "FLEXTOOL_SOLVER_WRAPPER_XPRESS",
    "copt": "FLEXTOOL_SOLVER_WRAPPER_COPT",
}
MAX_WRAPPER_ITEMS = 32
MAX_WRAPPER_ITEM_CHARS = 4096
#: Files Windows runs through a command interpreter (cmd.exe, PowerShell, Windows Script Host), which would read
#: the solver's path and arguments that FlexTool appends as a command line of its own.
_SCRIPT_SUFFIXES = frozenset({".bat", ".cmd", ".ps1", ".psm1", ".vbs", ".vbe", ".js", ".jse", ".wsf", ".wsh"})
#: Command interpreters: given FlexTool's arguments, ``cmd /c``, ``powershell -Command``, ``pwsh -c`` and
#: ``wscript`` / ``cscript`` read them as a command line or a script, not as the program to run.
_COMMAND_INTERPRETERS = frozenset({"cmd", "powershell", "powershell_ise", "pwsh", "wscript", "cscript"})
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


def _without_workspace(data: dict, folder: str | os.PathLike) -> list[str]:
    key = workspace_key(folder)
    current = data.get(TRUSTED_WORKSPACES_KEY, [])
    kept = []
    for item in current if isinstance(current, list) else []:
        if not isinstance(item, str) or not item or item == key:
            continue
        try:
            if os.path.samefile(item, folder):
                continue
        except OSError:
            pass
        kept.append(item)
    return kept


def remove_trusted_workspace(folder: str | os.PathLike, path: Path | None = None) -> None:
    """Take *folder* off the list of workspaces this computer set up."""

    def change(data: dict) -> None:
        data[TRUSTED_WORKSPACES_KEY] = _without_workspace(data, folder)

    update_settings_file(change, path)


def apply_settings(
    data: dict,
    *,
    wrappers: Mapping[str, list[str] | tuple[str, ...]] | None = None,
    allow_unlisted: bool | None = None,
    workspace: str | os.PathLike | None = None,
    trust_workspace: bool | None = None,
    project: str | os.PathLike | None = None,
    trust_project: bool | None = None,
) -> None:
    """Set several settings in *data* (the file's content, for :func:`update_settings_file`) at once: each wrapper
    of *wrappers* (an empty list removes it; check them with :func:`check_wrapper` first), the unlisted-names
    switch, and whether *workspace* and the GUI *project* are on the list (``None``: unchanged)."""
    if wrappers:
        current = data.get(WRAPPERS_KEY)
        current = dict(current) if isinstance(current, dict) else {}
        for solver, argv in wrappers.items():
            if argv:
                current[solver] = list(argv)
            else:
                current.pop(solver, None)
        data[WRAPPERS_KEY] = current
    if allow_unlisted is not None:
        data["allow_unlisted_parameters"] = bool(allow_unlisted)
    for folder, trust in ((workspace, trust_workspace), (project, trust_project)):
        if folder is not None and trust is not None:
            kept = _without_workspace(data, folder)
            data[TRUSTED_WORKSPACES_KEY] = [*kept, workspace_key(folder)] if trust else kept


# --- solver wrapper (B2) -----------------------------------------------------------------------------------------


def wrapper_env_name(solver: str) -> str:
    """``FLEXTOOL_SOLVER_WRAPPER_<SOLVER>``."""
    return WRAPPER_ENV[solver]


@dataclass(frozen=True)
class SolverWrapper:
    """The wrapper of one solver: the program and its arguments, and where it was set."""

    solver: str
    argv: tuple[str, ...]
    source: str  # ``environment variable FLEXTOOL_SOLVER_WRAPPER_X`` or the settings file's path

    @property
    def program(self) -> str:
        return self.argv[0]

    def shown(self) -> str:
        """The argument list as JSON, as the log line shows it."""
        return json.dumps(list(self.argv), ensure_ascii=False)

    def record(self) -> dict[str, object]:
        return {"argv": list(self.argv), "source": self.source}


def check_wrapper(argv: object, *, source: str, windows: bool | None = None) -> tuple[str, ...]:
    """The wrapper *argv* (from *source*) as a tuple, or :class:`LocalSolverSettingsError` saying what is wrong.

    A list of 1 to :data:`MAX_WRAPPER_ITEMS` texts of at most :data:`MAX_WRAPPER_ITEM_CHARS` characters without
    NUL.  The first is the program: a full path (FlexTool never searches ``PATH`` for it) of an existing file,
    symbolic links followed; on Windows (*windows*, default: this computer) a program (``.exe``), elsewhere an
    executable file.  Refused whatever the system, by the given and by the resolved name: a script that Windows
    runs through a command interpreter (``.bat``, ``.cmd``, ``.ps1``, ``.vbs``, ``.js``, ...) and a command
    interpreter itself (``cmd``, ``powershell``, ``pwsh``, ``wscript``, ``cscript``), because they read the
    arguments FlexTool appends (the solver's path and arguments) as a command line.  The wrapper runs without a
    shell: the arguments reach it as they are.
    """
    windows = os.name == "nt" if windows is None else windows
    head = f"The solver wrapper from {source}"
    if (
        not isinstance(argv, (list, tuple)) or not argv or len(argv) > MAX_WRAPPER_ITEMS
        or not all(isinstance(item, str) for item in argv)
    ):
        raise LocalSolverSettingsError(
            f"{head} must be a list of 1 to {MAX_WRAPPER_ITEMS} texts: the wrapper program's full path, then its "
            f"arguments, for example [\"/usr/bin/nice\", \"-n\", \"10\"].")
    for item in argv:
        if "\x00" in item or len(item) > MAX_WRAPPER_ITEM_CHARS:
            raise LocalSolverSettingsError(
                f"{head}: an item holds a NUL character or is longer than {MAX_WRAPPER_ITEM_CHARS} characters.")
    program = argv[0]
    path = Path(program)
    if not program.strip() or not path.is_absolute():
        raise LocalSolverSettingsError(
            f"{head}: {program!r} is not a full path. Give the wrapper program's full path (FlexTool does not look "
            "for it on PATH or in the current folder).")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        raise LocalSolverSettingsError(f"{head}: the program {program} does not exist.") from None
    if not resolved.is_file():
        raise LocalSolverSettingsError(f"{head}: {program} is not a file.")
    for name in dict.fromkeys((path.name, resolved.name)):
        stem, suffix = os.path.splitext(name.casefold())
        if suffix in _SCRIPT_SUFFIXES:
            raise LocalSolverSettingsError(
                f"{head}: {name} is a script that Windows runs through a command interpreter, which would read the "
                "solver's path and arguments that FlexTool adds as a command line. Use a program instead: on "
                "Windows python.exe (full path) with your Python script as its first argument.")
        if stem in _COMMAND_INTERPRETERS:
            raise LocalSolverSettingsError(
                f"{head}: {name} is a command interpreter, which would read the solver's path and arguments that "
                "FlexTool adds as a command line or a script. Use a program that starts its arguments as a "
                "command, for example python (full path) with your Python script as its first argument.")
    if windows:
        if path.suffix.casefold() != ".exe" or resolved.suffix.casefold() != ".exe":
            raise LocalSolverSettingsError(f"{head}: on Windows the wrapper must be a program (.exe): {program}.")
    elif not os.access(resolved, os.X_OK):
        raise LocalSolverSettingsError(f"{head}: {program} is not executable (chmod +x).")
    return tuple(argv)


def solver_wrapper(solver: str, environ: Mapping[str, str] | None = None) -> SolverWrapper | None:
    """The wrapper of *solver* on this computer, checked (:func:`check_wrapper`), or ``None``.

    ``FLEXTOOL_SOLVER_WRAPPER_<SOLVER>`` first: a JSON list (starting with ``[``), else one program path, never
    split; set but empty, or ``[]``, means no wrapper (also over the file).  Else ``wrappers`` in the settings
    file.  Never read from a database, a project, a workspace or a flag.  Raises
    :class:`LocalSolverSettingsError` for a value that cannot be used, naming where it was set."""
    env = os.environ if environ is None else environ
    name = wrapper_env_name(solver)
    raw = env.get(name)
    if raw is not None:
        text = raw.strip()
        source = f"environment variable {name}"
        if not text:
            return None
        if text.startswith("["):
            try:
                value = json.loads(text)
            except ValueError as exc:
                raise LocalSolverSettingsError(
                    f"{name} is not a valid JSON list ({exc}). Write it as [\"/full/path/program\", \"argument\", "
                    "...], or give only the program's full path.") from exc
        else:
            value = [text]
    else:
        path = settings_path()
        source = str(path)
        wrappers = read_settings_file(path).get(WRAPPERS_KEY, {})
        if not isinstance(wrappers, dict):
            raise LocalSolverSettingsError(
                f"FlexTool's solver settings file {path}: {WRAPPERS_KEY} is not an object of solver: list.")
        value = wrappers.get(solver)
        if value is None:
            return None
    if value == []:
        return None
    return SolverWrapper(solver, check_wrapper(value, source=source), source)


def set_solver_wrapper(solver: str, argv: list[str] | tuple[str, ...] | None, path: Path | None = None) -> None:
    """Write the wrapper of *solver* to the settings file (``None`` or empty: remove it), after
    :func:`check_wrapper`."""
    if solver not in WRAPPER_SOLVERS:
        raise ValueError(f"no solver wrapper for {solver!r}")
    path = settings_path() if path is None else path
    value = list(check_wrapper(list(argv), source=str(path))) if argv else None

    def change(data: dict) -> None:
        wrappers = data.get(WRAPPERS_KEY)
        wrappers = dict(wrappers) if isinstance(wrappers, dict) else {}
        if value is None:
            wrappers.pop(solver, None)
        else:
            wrappers[solver] = value
        data[WRAPPERS_KEY] = wrappers

    update_settings_file(change, path)


def set_allow_unlisted_parameters(value: bool, path: Path | None = None) -> None:
    """Write ``allow_unlisted_parameters`` to the settings file."""

    def change(data: dict) -> None:
        data["allow_unlisted_parameters"] = bool(value)

    update_settings_file(change, path)
