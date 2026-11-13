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
  project's solver option files" settings and ``flextool-trust-workspace <folder>`` list them on the user's word.
  The list is outside every workspace, so a file that came with a workspace or a project cannot put it there.
  Each entry is bound to the folder trusted (device, inode and birth time; on Windows volume serial, file ID and
  creation time) and to the hashes of its option files then: a folder put at the same path later is not trusted,
  and a file changed since is trusted only when this account wrote it and no other account can change it
  (:class:`FolderTrust`, review b2b3 M1).  Entries must be full paths (L1).
- ``wrappers`` (``{"cplex": ["/usr/local/bin/lic_hold", "--queue", "opt"], ...}``): a program that starts in
  front of a commercial solver's program (B2), for example to hold a floating licence or to lower the solver's
  priority.  ``FLEXTOOL_SOLVER_WRAPPER_<SOLVER>`` wins over the file: a JSON list, or one program path (never
  split); empty or ``[]`` means no wrapper.  :func:`check_wrapper` holds the rules.

Not in the database, a project folder, ``settings.yaml`` or ``projects.yaml``: those travel with a project or a
workspace.  No command-line flag either: a Spine Toolbox project carries tool command lines and can come from
someone else (``flextool-trust-workspace`` is a separate command a person types and answers, see
:mod:`flextool.cli.cmd_trust_workspace`).  The value is read when it is needed (an unlisted name from a database), so a damaged file stops
only a run that depends on it, with a message naming it.

Only the standard library and :mod:`flextool.user_dirs` are imported.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

_logger = logging.getLogger(__name__)

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
#: Windows programs that read their arguments as a command line, a script or a command to look up (review b2b3
#: L4): ``wsl`` and ``bash`` (the WSL launcher, Git Bash) join them into a shell command; ``mshta``,
#: ``rundll32``, ``regsvr32``, ``forfiles``, ``conhost`` and ``msiexec`` run what they name.  Refused on Windows
#: only: on Linux and macOS ``/bin/bash wrap.sh`` passes FlexTool's arguments as ``"$@"``.
_WINDOWS_COMMAND_RUNNERS = frozenset({
    "wsl", "wslhost", "bash", "sh", "zsh", "mshta", "rundll32", "regsvr32", "forfiles", "conhost", "msiexec",
})
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
            write_text_atomic(path, json.dumps(data, indent=1, ensure_ascii=False, sort_keys=True) + "\n",
                              private=True)
    except LockTimeout as exc:
        raise LocalSolverSettingsError(f"FlexTool's solver settings file {path}: {exc}") from exc
    except OSError as exc:
        raise LocalSolverSettingsError(f"FlexTool cannot write its solver settings file {path} ({exc}).") from exc


# --- workspaces this computer set up (B3) --------------------------------------------------------------------
#
# An entry is ``{"path": <workspace_key>, "id": [st_dev, st_ino] | null, "files": {"<solver>.opt": sha256}}``:
# the folder's path, the identity of the folder that was trusted at that path (review b2b3 M1: a folder put at
# the same path later, such as a received copy unpacked where the trusted one was deleted, is a different folder,
# so it is not trusted), and the SHA-256 (CRLF read as LF) of each option file in its ``solver_config/`` when it
# was trusted.  An option file that differs from its hash (or is new) is trusted only when this account wrote it
# and only this account can change it (:func:`only_you_can_change`); otherwise it asks to be trusted again.  A
# plain text entry is what development builds of 4.4 wrote: it is bound to the folder now at that path, with its
# current files, the first time FlexTool reads it (:func:`trust_status`), which trusts no more than the text
# entry did.

#: Why a changed option file is taken as the user's own edit: see :func:`only_you_can_change`.
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


def workspace_key(folder: str | os.PathLike) -> str:
    """How a workspace is listed: its absolute path with symbolic links resolved, in the case the platform
    compares (``os.path.normcase``: lower case on Windows)."""
    try:
        resolved = Path(folder).resolve()
    except (OSError, RuntimeError):
        resolved = Path(os.path.abspath(folder))
    return os.path.normcase(str(resolved))


Identity = tuple[int, int, "int | None"]


def folder_identity(folder: str | os.PathLike) -> Identity | None:
    """``(st_dev, st_ino, birth time in ns)`` of *folder* (links followed): the device and the inode on Linux and
    macOS, the volume serial number and the file ID on Windows (NTFS and ReFS give one).  The birth time is there
    because a file system can give a deleted folder's inode number to the next folder made (Linux ext4 does so at
    once), so device and inode alone do not tell a folder deleted and made again at the same path; ``None`` when
    the system does not give it.  ``None`` as a whole when the folder cannot be read or its file system gives no
    file ID (``st_ino`` 0: FAT, some network drives)."""
    try:
        st = os.stat(folder)
    except OSError:
        return None
    if not st.st_ino:
        return None
    return (int(st.st_dev), int(st.st_ino), _birth_ns(folder, st))


def _birth_ns(folder: str | os.PathLike, st: os.stat_result) -> int | None:
    """When *folder* was made, in ns: ``st_birthtime`` (Windows, macOS), else Linux ``statx``; ``None`` when not
    given."""
    ns = getattr(st, "st_birthtime_ns", None)
    if ns is not None:
        return int(ns)
    seconds = getattr(st, "st_birthtime", None)
    if seconds is not None:
        return int(seconds * 1_000_000_000)
    if not sys.platform.startswith("linux"):
        return None
    try:
        import ctypes

        statx = ctypes.CDLL(None, use_errno=True).statx
        statx.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_uint, ctypes.c_void_p]
        statx.restype = ctypes.c_int
        buffer = ctypes.create_string_buffer(256)  # struct statx is 256 bytes
        at_fdcwd, statx_btime = -100, 0x800
        if statx(at_fdcwd, os.fsencode(os.fspath(folder)), 0, statx_btime, buffer) != 0:
            return None
        raw = buffer.raw
        if not int.from_bytes(raw[0:4], sys.byteorder) & statx_btime:  # stx_mask: the file system gave no btime
            return None
        # stx_btime at offset 80: tv_sec (s64), tv_nsec (u32).
        seconds = int.from_bytes(raw[80:88], sys.byteorder, signed=True)
        return seconds * 1_000_000_000 + int.from_bytes(raw[88:92], sys.byteorder)
    except (AttributeError, OSError, ValueError):
        return None


def same_identity(stored: Identity | None, current: Identity | None) -> bool:
    """Whether the folder of identity *current* is the one trusted with *stored* (a birth time counts when both
    have one)."""
    if stored is None or current is None:
        return stored is None
    if stored[:2] != current[:2]:
        return False
    return stored[2] is None or current[2] is None or stored[2] == current[2]


def is_link_like(path: str | os.PathLike) -> bool:
    """Whether *path* is a symbolic link, a Windows junction or another Windows reparse point (review b2b3 L5): an
    arrangement whose target can change later, never taken as FlexTool's own folder or file."""
    p = Path(path)
    try:
        if p.is_symlink():
            return True
        is_junction = getattr(p, "is_junction", None)
        if is_junction is not None and is_junction():
            return True
        attributes = getattr(os.lstat(p), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(attributes & _FILE_ATTRIBUTE_REPARSE_POINT)


def _sha256(data: bytes) -> str:
    """The hash :mod:`flextool._solver_config_source` uses for option files (CRLF read as LF)."""
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def option_file_hashes(folder: str | os.PathLike) -> dict[str, str]:
    """``{"<solver>.opt": sha256}`` of the option files in ``<folder>/solver_config`` (plain files only; a link is
    left out, so it is never trusted by its hash)."""
    config = Path(folder) / "solver_config"
    out: dict[str, str] = {}
    try:
        entries = sorted(config.iterdir()) if config.is_dir() and not is_link_like(config) else []
        for entry in entries:
            if not entry.name.casefold().endswith(".opt") or is_link_like(entry) or not entry.is_file():
                continue
            out[entry.name] = _sha256(entry.read_bytes())
    except OSError:
        pass
    return out


def only_you_can_change(st: os.stat_result | None, *, windows: bool | None = None) -> str | None:
    """``None`` when the option file of ``os.lstat`` result *st* was written by this account and no other account
    can change it, else why not (review b2b3 M1: then a change since the folder was trusted is the user's own edit,
    trusted without asking again).

    Linux and macOS: a plain file (no link) owned by this account, not writable by others, and not writable by its
    group unless the group is this account's own private group (the group named after the account, with no other
    members: the default on many Linux systems, whose ``umask`` 002 makes new files group-writable).

    Windows: a plain file (no link or reparse point).  Windows gives a new file the account that writes it as its
    owner and, in a user's own folders, an access list that lets no other ordinary account write; reading the
    owner and the access list needs the Win32 security interface, which FlexTool does not use.  The folder's
    identity (volume serial number and file ID) is still checked, so a replaced folder is not trusted; a folder
    whose file system gives no file ID gets no such pass (:class:`FolderTrust`)."""
    if st is None:
        return "FlexTool cannot read it"
    windows = os.name == "nt" if windows is None else windows
    if stat.S_ISLNK(st.st_mode) or getattr(st, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT:
        return "it is a link, whose target can change"
    if not stat.S_ISREG(st.st_mode):
        return "it is not a plain file"
    if windows:
        return None
    uid = os.geteuid()
    if st.st_uid != uid:
        return "another account owns it"
    if st.st_mode & stat.S_IWOTH:
        return "every account can write it"
    if st.st_mode & stat.S_IWGRP and not _own_private_group(st.st_gid, uid):
        return "its group can write it"
    return None


def _own_private_group(gid: int, uid: int) -> bool:
    try:
        import grp
        import pwd

        group = grp.getgrgid(gid)
        return not group.gr_mem and group.gr_name == pwd.getpwuid(uid).pw_name
    except (ImportError, KeyError, OSError):
        return False


@dataclass(frozen=True)
class TrustedFolder:
    """One entry of ``trusted_workspaces``: the folder's key (:func:`workspace_key`), its identity when it was
    trusted (``None``: the file system gave none), and its option files' hashes then (``None``: a text entry of a
    4.4 development build, not yet bound)."""

    path: str
    identity: Identity | None
    files: Mapping[str, str] | None

    def record(self) -> dict[str, object]:
        return {"path": self.path, "id": list(self.identity) if self.identity else None,
                "files": dict(sorted((self.files or {}).items()))}


def _bound(folder: str | os.PathLike) -> TrustedFolder:
    """The entry that trusts *folder* as it is now."""
    return TrustedFolder(workspace_key(folder), folder_identity(folder), option_file_hashes(folder))


def _parse_entry(item: object, path: Path) -> TrustedFolder:
    bad = LocalSolverSettingsError(
        f"FlexTool's solver settings file {path}: {TRUSTED_WORKSPACES_KEY} holds {item!r}, which is not a folder "
        "this computer trusted. Write full folder paths, or delete the entry and trust the folder again (Solver "
        "options -> On this computer, or flextool-trust-workspace <folder>).")
    if isinstance(item, str):
        raw, identity, files = item, None, None
    elif isinstance(item, dict) and isinstance(item.get("path"), str):
        raw = item["path"]
        ident = item.get("id")
        if ident is None:
            identity = None
        elif (isinstance(ident, list) and len(ident) == 3
              and all(isinstance(n, int) and not isinstance(n, bool) for n in ident[:2])
              and (ident[2] is None or (isinstance(ident[2], int) and not isinstance(ident[2], bool)))):
            identity = (ident[0], ident[1], ident[2])
        else:
            raise bad
        files = item.get("files", {})
        if not isinstance(files, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in files.items()):
            raise bad
    else:
        raise bad
    if not raw or not os.path.isabs(raw):
        # "." would match every folder a run starts from; "~/x" never matches (review b2b3 L1).
        raise LocalSolverSettingsError(
            f"FlexTool's solver settings file {path}: {TRUSTED_WORKSPACES_KEY} holds {raw!r}, which is not a full "
            "folder path. Write full folder paths (no '.', '~' or relative folders).")
    return TrustedFolder(workspace_key(raw), identity, files)


def trusted_entries(path: Path | None = None) -> tuple[TrustedFolder, ...]:
    """The folders this computer trusted; raises :class:`LocalSolverSettingsError` for a damaged file, a value that
    is not a list of entries, or an entry that is not a full folder path."""
    path = settings_path() if path is None else path
    value = read_settings_file(path).get(TRUSTED_WORKSPACES_KEY, [])
    if not isinstance(value, list):
        raise LocalSolverSettingsError(
            f"FlexTool's solver settings file {path}: {TRUSTED_WORKSPACES_KEY} is not a list of folders.")
    return tuple(_parse_entry(item, path) for item in value)


def trusted_workspaces(path: Path | None = None) -> tuple[str, ...]:
    """The keys (:func:`workspace_key`) of the folders this computer trusted (see :func:`trusted_entries`)."""
    return tuple(entry.path for entry in trusted_entries(path))


def _matches(entry_path: str, folder: str | os.PathLike, key: str) -> bool:
    if entry_path == key:
        return True
    try:
        # macOS keeps a path's case as typed although its disks compare without case.
        return os.path.samefile(entry_path, folder)
    except OSError:
        return False


def _entry_path(item: object) -> str | None:
    if isinstance(item, str):
        return item
    if isinstance(item, dict) and isinstance(item.get("path"), str):
        return item["path"]
    return None


def _check_settings_outside(folder: str | os.PathLike, path: Path) -> None:
    """Refuse a settings file inside *folder* (review b2b3 L2): a folder that came from someone else could then
    carry its own list."""
    try:
        inside = Path(path).resolve().is_relative_to(Path(folder).resolve())
    except (OSError, RuntimeError):
        inside = False
    if inside:
        raise LocalSolverSettingsError(
            f"FlexTool's solver settings file {path} lies inside {folder}, so a copy of that folder would carry "
            f"its own settings; FlexTool does not use it there. Set {USER_HOME_HINT} to a folder outside your "
            "workspaces.")


USER_HOME_HINT = "FLEXTOOL_USER_HOME (or XDG_CONFIG_HOME)"


@dataclass(frozen=True)
class FolderTrust:
    """Whether a folder (a workspace or a GUI project) is one this computer trusted: *entry* is its entry
    (``None``: not listed); *problem* why a listed path is not trusted (the folder there was replaced)."""

    entry: TrustedFolder | None
    problem: str | None = None

    @property
    def trusted(self) -> bool:
        return self.entry is not None and self.problem is None

    def file_problem(self, name: str, digest: str, st: os.stat_result | None) -> str | None:
        """``None`` when the option file *name* (``"gurobi.opt"``) of the trusted folder, with hash *digest* and
        ``os.lstat`` result *st*, is trusted: as it was when the folder was trusted, or changed by this account
        (:func:`only_you_can_change`) in a folder whose identity was checked.  Else why not, for the message."""
        if not self.trusted or self.entry is None:
            return self.problem or "the folder is not one this computer trusted"
        if (self.entry.files or {}).get(name) == digest:
            return None
        what = "changed" if name in (self.entry.files or {}) else "was added"
        if self.entry.identity is None:
            return (f"{name} {what} after this computer trusted the folder, and the folder's drive gives it no "
                    "identity, so FlexTool cannot tell that it is still your folder")
        why = only_you_can_change(st)
        if why is None:
            return None
        return f"{name} {what} after this computer trusted the folder, and FlexTool cannot tell that you did it ({why})"

    def changed_files(self, folder: str | os.PathLike) -> list[tuple[str, str]]:
        """``(name, why)`` of the option files in ``<folder>/solver_config`` that this trust does not cover
        (:meth:`file_problem`)."""
        config = Path(folder) / "solver_config"
        out: list[tuple[str, str]] = []
        try:
            entries = sorted(config.iterdir()) if config.is_dir() else []
        except OSError:
            return out
        for entry in entries:
            if not entry.name.casefold().endswith(".opt"):
                continue
            try:
                st = os.lstat(entry)
                digest = _sha256(entry.read_bytes())
            except OSError:
                out.append((entry.name, "FlexTool cannot read it"))
                continue
            problem = self.file_problem(entry.name, digest, st)
            if problem is not None:
                out.append((entry.name, problem))
        return out


def trust_status(folder: str | os.PathLike, path: Path | None = None) -> FolderTrust:
    """Whether *folder* is a folder this computer trusted, and the entry that says so.  A listed path whose folder
    is not the one trusted there (another identity: deleted and made again, unpacked or copied there) is not
    trusted (review b2b3 M1).  A text entry of a 4.4 development build is bound to the folder now there.  Raises
    :class:`LocalSolverSettingsError` when the settings file cannot be read, or lies inside *folder*."""
    if path is None:
        from flextool.user_dirs import config_dir_overridden

        path = settings_path()
        if config_dir_overridden():
            # A folder named by FLEXTOOL_USER_HOME or XDG_CONFIG_HOME inside the current folder is ignored
            # already (flextool.user_dirs); this covers the folder judged here (review b2b3 L2).
            _check_settings_outside(folder, path)
    key = workspace_key(folder)
    entry = next((e for e in trusted_entries(path) if _matches(e.path, folder, key)), None)
    if entry is None:
        return FolderTrust(None)
    if entry.files is None:
        bound = _bound(folder)
        try:
            update_settings_file(lambda data: _put_entry(data, folder, bound), path)
            _logger.info("Bound the trusted folder %s to the folder now at that path (%s).", folder, path)
        except LocalSolverSettingsError as exc:
            _logger.warning("%s", exc)
        return FolderTrust(bound)
    if entry.identity is not None and not same_identity(entry.identity, folder_identity(folder)):
        return FolderTrust(entry, (
            f"the folder at {folder} is not the one this computer trusted: it was replaced since (deleted and "
            "made again, or unpacked or copied there)"))
    return FolderTrust(entry)


def is_trusted_workspace(folder: str | os.PathLike, path: Path | None = None) -> bool:
    """Whether *folder* is a folder this computer trusted, and still the same folder (:func:`trust_status`).
    Raises :class:`LocalSolverSettingsError` when the settings file cannot be read."""
    return trust_status(folder, path).trusted


def _without_folder(items: list, folder: str | os.PathLike) -> list:
    """*items* (the list as stored) without the entries of *folder*; every other item is kept as it is, also one
    FlexTool cannot read (review b2b3 L7: changing one folder never drops the others)."""
    key = workspace_key(folder)
    kept = []
    for item in items:
        entry_path = _entry_path(item)
        if entry_path and os.path.isabs(entry_path) and (
                workspace_key(entry_path) == key or _matches(entry_path, folder, key)):
            continue
        kept.append(item)
    return kept


def _stored_list(data: dict) -> list:
    current = data.get(TRUSTED_WORKSPACES_KEY, [])
    if not isinstance(current, list):
        raise LocalSolverSettingsError(
            f"FlexTool's solver settings file: {TRUSTED_WORKSPACES_KEY} is not a list of folders; correct or delete "
            "it first.")
    return current


def _put_entry(data: dict, folder: str | os.PathLike, entry: TrustedFolder | None) -> None:
    kept = _without_folder(_stored_list(data), folder)
    data[TRUSTED_WORKSPACES_KEY] = [*kept, entry.record()] if entry is not None else kept


def add_trusted_workspace(folder: str | os.PathLike, path: Path | None = None) -> bool:
    """List *folder*, as it is now, as a folder this computer set up; ``False`` when it already was (the same
    folder, bound to its files)."""
    added = False

    def change(data: dict) -> None:
        nonlocal added
        key = workspace_key(folder)
        for item in _stored_list(data):
            entry_path = _entry_path(item)
            if not (entry_path and os.path.isabs(entry_path) and _matches(workspace_key(entry_path), folder, key)):
                continue
            try:
                entry = _parse_entry(item, path or settings_path())
            except LocalSolverSettingsError:
                break
            if entry.files is not None and same_identity(entry.identity, folder_identity(folder)):
                return
            break
        _put_entry(data, folder, _bound(folder))
        added = True

    update_settings_file(change, path)
    return added


def remove_trusted_workspace(folder: str | os.PathLike, path: Path | None = None) -> None:
    """Take *folder* off the list of folders this computer trusted."""
    update_settings_file(lambda data: _put_entry(data, folder, None), path)


def move_trusted_workspace(old: str | os.PathLike, new: str | os.PathLike, path: Path | None = None) -> None:
    """Carry the trust of the folder renamed from *old* to *new* (the same folder: its identity and its files'
    hashes are kept, so a rename never trusts a change the old entry did not cover).  Nothing when *old* was not
    listed, or the folder at *new* is not the one trusted at *old*."""

    def change(data: dict) -> None:
        key = workspace_key(old)
        for item in _stored_list(data):
            entry_path = _entry_path(item)
            if entry_path and os.path.isabs(entry_path) and workspace_key(entry_path) == key:
                entry = _parse_entry(item, path or settings_path())
                break
        else:
            return
        _put_entry(data, old, None)
        if entry.files is None or not same_identity(entry.identity, folder_identity(new)):
            return
        _put_entry(data, new, TrustedFolder(workspace_key(new), entry.identity, entry.files))

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
            # Ticked: trusted as it is now (the user has read its files); unticked: off the list.  Every other
            # entry is kept as stored (review b2b3 L7).
            _put_entry(data, folder, _bound(folder) if trust else None)


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
        """How the run log shows the wrapper: the program and the number of its arguments, never the arguments
        (review b2b3 L3, a change of D7: the AI assistant can send the run log to its provider, and an argument
        can be a licence token).  The full list is in the run record only (:meth:`record`), which stays on this
        computer and is scrubbed to the same form when a project is exported."""
        count = len(self.argv) - 1
        return f"{self.program} with {count} argument{'' if count == 1 else 's'}"

    def record(self) -> dict[str, object]:
        return {"argv": list(self.argv), "source": self.source}


def check_wrapper(argv: object, *, source: str, windows: bool | None = None) -> tuple[str, ...]:
    """The wrapper *argv* (from *source*) as a tuple, or :class:`LocalSolverSettingsError` saying what is wrong.

    A list of 1 to :data:`MAX_WRAPPER_ITEMS` texts of at most :data:`MAX_WRAPPER_ITEM_CHARS` characters without
    NUL.  The first is the program: a full path (FlexTool never searches ``PATH`` for it) of an existing file,
    symbolic links followed; on Windows (*windows*, default: this computer) a program (``.exe``), elsewhere an
    executable file.  Refused whatever the system, by the given and by the resolved name: a script that Windows
    runs through a command interpreter (``.bat``, ``.cmd``, ``.ps1``, ``.vbs``, ``.js``, ...) and a command
    interpreter itself (``cmd``, ``powershell``, ``pwsh``, ``wscript``, ``cscript``; on Windows also ``wsl``, ``bash`` and the
    other :data:`_WINDOWS_COMMAND_RUNNERS`), because they read the
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
        if windows and stem in _WINDOWS_COMMAND_RUNNERS:
            raise LocalSolverSettingsError(
                f"{head}: {name} reads its arguments as a command line or runs what they name, so it would read the "
                "solver's path and arguments that FlexTool adds as a command of its own. Use a program (.exe) that "
                "starts its arguments as a command, for example python.exe (full path) with your Python script.")
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
