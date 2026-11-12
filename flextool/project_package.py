"""Project packages: export a FlexTool project to one zip file, check one, and import one (no Tk).

A package is ``<Name>_<YYYYMMDD>.flextool.zip``.  Every project file is under ``<Name>/``; ``manifest.json`` at the
root lists the FlexTool, solver and library versions, the data version and frozen marker of each input database,
the result runs it holds, the settings it left out, and the SHA-256, size and modification time (``mtime_ns``) of
every file.

What a package holds:

* always: ``input_sources/`` (with the external input files copied in, under the same source numbers, unless the
  caller says not to), ``settings.yaml`` without the machine-specific settings (:data:`DROPPED_SETTINGS`),
  ``plot_settings.yaml``, and the solver option files the project's runs use (``solver_config/``: per solver
  the first ``<solver>.opt`` of the project's ``solver_config`` folder, then the workspace's, leaving out unedited
  FlexTool templates, which equal the defaults).  Import unpacks them into ``solver_config/`` of the new project,
  where its runs use them ahead of the workspace's files, file by file; a file with an option that names a path
  or another computer is kept as ``<solver>.opt.review`` (never read by a run) until the user checks it, and the
  workspace's own files never change;
* the result runs the caller picks (``output_parquet/<run>/``, with ``output_parquet/_axis_bounds.json`` cut to
  those runs).  By default each run record (``_run_status.json``) has this computer's folder paths, user name and
  host name replaced by placeholders and the run log (``_run.log``) is left out (``scrub_run_records=False`` ships
  them as they are).  A run that is still being written is refused;
* never: what FlexTool makes again from the results or uses as scratch (:data:`REGENERATED`), such as
  ``output_csv/``, ``output_plots/`` and ``results.sqlite``, nor SQLite ``-journal`` / ``-wal`` / ``-shm`` files
  and Office lock files.

A ``.sqlite`` input is copied byte for byte (so a released database keeps its SHA-256) unless a ``-journal`` or
``-wal`` file next to it says a writer may be at work; then a consistent copy is taken with the :mod:`sqlite3`
backup API from a read-only connection.  A file that changes while it is copied is refused.  Every member name is
checked against the rules of Windows (reserved names such as ``CON``, the characters ``<>:"|?*``, a trailing dot
or space, names that differ only in case) when writing and when reading a package.

Import checks every checksum before it writes anything, refuses member names that leave the project folder,
unpacks into ``projects/.<name>.partial/``, gives each file its recorded modification time back (so a run's record
still matches its input file) and only then renames the folder into place.  The checksums detect a damaged file,
not a deliberate change: the manifest is not signed.

Reading the databases uses plain :mod:`sqlite3` in read-only mode (:mod:`flextool.update_flextool.frozen`), so an
export never migrates or writes an input file; a frozen database travels as it is and stays frozen.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import shutil
import socket
import sqlite3
import sys
import tempfile
import time
import zipfile
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Callable

import yaml

logger = logging.getLogger(__name__)

FORMAT = "flextool.project"
FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
PACKAGE_SUFFIX = ".flextool.zip"

SETTINGS_FILE = "settings.yaml"
PLOT_SETTINGS_FILE = "plot_settings.yaml"
INPUT_DIR = "input_sources"
RESULTS_DIR = "output_parquet"
SOLVER_CONFIG_DIR = "solver_config"
#: Where FlexTool 4.3 import put the shipped solver option files inside the project (not read by a run); the
#: FlexTool window renames it to ``solver_config/`` (:func:`adopt_imported_solver_config`).
IMPORTED_SOLVER_CONFIG_DIR = "imported_solver_config"
#: The suffix of a shipped option file kept back for the user to check (:func:`settle_shipped_solver_config`).
HELD_SUFFIX = ".review"

#: ``settings.yaml`` keys that belong to one machine: left out of a package.
DROPPED_SETTINGS = ("execution_limits", "max_workers", "scenario_resource_history", "external_refs")
#: Project folders and files never packaged: FlexTool makes them again from the results, or they are scratch.
REGENERATED = (
    "output_plots", "output_csv", "output_excel", "output_plot_comparisons", "output_parquet_comparison",
    "results.sqlite", "results.sqlite.lock", "work", "converted", "intermediate",
)
#: The folders of a new project (``flextool.gui.project_utils.PROJECT_SUBDIRS``); import makes the missing ones.
_PROJECT_SUBDIRS = (
    "input_sources", "converted", "intermediate", "work", "output_plots", "output_parquet", "output_csv",
    "output_excel", "output_plot_comparisons",
)
_CHUNK = 1 << 20
#: The run record and the run log of a result run (``flextool.run_record``).
RUN_STATUS_NAME = "_run_status.json"
RUN_LOG_NAME = "_run.log"
#: The shared y-axis ranges of the result viewer, one entry per run (``flextool.plot_outputs.shared_manifest``).
AXIS_BOUNDS_NAME = "_axis_bounds.json"
#: Run states that mean the run may still be writing its folder (``flextool.run_record.effective_state``).
ACTIVE_RUN_STATES = ("running", "unknown")
#: The largest modification time a package may record: the end of year 9999, in ns.
_MAX_MTIME_NS = 253_402_300_799 * 10**9
#: Free space kept on the disk on top of the unpacked size of a package.
_DISK_MARGIN_BYTES = 64 * 10**6
_WINDOWS_RESERVED = frozenset({"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                               *(f"LPT{i}" for i in range(1, 10))})
_WINDOWS_BAD_CHARS = frozenset('<>:"|?*')


class PackageError(RuntimeError):
    """A package cannot be written, read or imported."""


class PackageExistsError(PackageError):
    """The package file exists already and replacing it was not allowed."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"{path} exists already.")


class NameClashError(PackageError):
    """A project with the import name exists already; ``suggestion`` is the first free ``<Name>_<n>``."""

    def __init__(self, name: str, suggestion: str) -> None:
        self.name = name
        self.suggestion = suggestion
        super().__init__(f"A project named '{name}' exists already. Import it under another name, e.g. "
                         f"'{suggestion}'.")


@dataclass
class RunInfo:
    """One result run of a project: its folder under ``output_parquet``, its size, and whether it is stale."""

    name: str
    bytes: int
    #: The input file of the run changed after the run read it (``None``: cannot be told).
    stale: bool | None = None
    #: The state of the run record (``flextool.run_record.effective_state``; ``None``: no record).
    state: str | None = None

    @property
    def active(self) -> bool:
        """The run may still be writing its folder: it cannot be packaged now."""
        return self.state in ACTIVE_RUN_STATES


@dataclass
class Manifest:
    """``manifest.json`` of a package."""

    project: str
    created: str
    flextool: str | None = None
    polar_high: str | None = None
    highspy: str | None = None
    polars: str | None = None
    spinedb_api: str | None = None
    python: str = ""
    platform: str = ""
    db_versions: dict[str, int | None] = field(default_factory=dict)
    frozen: dict[str, str] = field(default_factory=dict)
    results: list[str] = field(default_factory=list)
    stale_runs: list[str] = field(default_factory=list)
    dropped_settings: list[str] = field(default_factory=list)
    #: External input files copied into ``input_sources/``: source name -> packaged name.
    bundled_external: dict[str, str] = field(default_factory=dict)
    #: External input files not in the package (left out, or missing at export).
    external_not_included: list[str] = field(default_factory=list)
    solver_config: list[str] = field(default_factory=list)
    #: ``"<Name>/<path>"`` -> ``{"sha256", "bytes", "mtime_ns"}``.
    files: dict[str, dict] = field(default_factory=dict)
    format: str = FORMAT
    format_version: int = FORMAT_VERSION
    #: Whether the run records were stripped of this computer's paths, user and host (and the logs left out).
    run_records_scrubbed: bool = True
    #: Notes of the export (not written to the package): files open in Excel, unreadable data versions.
    export_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "format": self.format, "format_version": self.format_version, "created": self.created,
            "project": self.project, "flextool": self.flextool, "polar_high": self.polar_high,
            "highspy": self.highspy, "polars": self.polars, "spinedb_api": self.spinedb_api,
            "python": self.python, "platform": self.platform, "db_versions": self.db_versions,
            "frozen": self.frozen, "results": self.results, "stale_runs": self.stale_runs,
            "dropped_settings": self.dropped_settings, "bundled_external": self.bundled_external,
            "external_not_included": self.external_not_included, "solver_config": self.solver_config,
            "run_records_scrubbed": self.run_records_scrubbed, "files": self.files,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Manifest:
        if not isinstance(data, dict) or data.get("format") != FORMAT:
            raise PackageError("This file is not a FlexTool project package (its manifest.json is missing or "
                               "of another kind).")
        version = data.get("format_version")
        if not isinstance(version, int) or version > FORMAT_VERSION:
            raise PackageError(f"The package is of format version {version}; this FlexTool reads version "
                               f"{FORMAT_VERSION} and older. Update FlexTool.")
        project = data.get("project")
        files = data.get("files")
        if not isinstance(project, str) or not project or not isinstance(files, dict):
            raise PackageError("The manifest.json of the package is incomplete (no project name or file list).")
        return cls(
            project=project, created=str(data.get("created") or ""),
            flextool=data.get("flextool"), polar_high=data.get("polar_high"), highspy=data.get("highspy"),
            polars=data.get("polars"), spinedb_api=data.get("spinedb_api"),
            python=str(data.get("python") or ""), platform=str(data.get("platform") or ""),
            db_versions=dict(data.get("db_versions") or {}), frozen=dict(data.get("frozen") or {}),
            results=list(data.get("results") or []), stale_runs=list(data.get("stale_runs") or []),
            dropped_settings=list(data.get("dropped_settings") or []),
            bundled_external=dict(data.get("bundled_external") or {}),
            external_not_included=list(data.get("external_not_included") or []),
            solver_config=list(data.get("solver_config") or []), files=files, format_version=version,
            run_records_scrubbed=bool(data.get("run_records_scrubbed", False)),
        )


@dataclass
class ImportReport:
    """What :func:`import_project` did."""

    name: str
    project_path: Path
    manifest: Manifest
    warnings: list[str] = field(default_factory=list)
    #: ``(file name, "same" | "different" | "not in workspace")`` per shipped ``.opt`` file, compared with the file
    #: of that solver the workspace's runs use.
    solver_config: list[tuple[str, str]] = field(default_factory=list)
    #: Where the shipped solver options are and how they are used ("" when they all equal the workspace's).
    solver_config_hint: str = ""
    #: A short unified diff (workspace -> shipped) per ``.opt`` file that differs.
    solver_config_diffs: dict[str, list[str]] = field(default_factory=dict)
    #: The problems of each shipped file kept as ``<solver>.opt.review`` (not used until the user renames it).
    solver_config_held: dict[str, list[str]] = field(default_factory=dict)
    #: Shipped files that were unedited FlexTool templates: not unpacked (they equal the defaults).
    solver_config_defaults: list[str] = field(default_factory=list)


@dataclass
class VerifyReport:
    """What :func:`verify_package` found."""

    manifest: Manifest | None
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_files: int = 0

    @property
    def ok(self) -> bool:
        return self.manifest is not None and not self.problems


#: What :func:`verify_package`'s report and the CLI say about the checksums.
CHECKSUM_NOTE = ("The checksums detect files damaged on the way, not deliberate changes: the manifest is not "
                 "signed.")


# ---------------------------------------------------------------------------------------------------------------
# Helpers


def project_solver_config_dirs(project_dir: str | Path, solver_config_dir: object = None) -> list[Path]:
    """The ladder of solver option folders a GUI run of *project_dir* uses: ``<project>/solver_config`` (when
    there), then ``$FLEXTOOL_SOLVER_CONFIG_DIR``'s folders, else ``<workspace>/solver_config`` (the current folder,
    where FlexTool and the GUI run).  The GUI passes the same folders to its runs as ``--solver-config-dir``, so an
    exported project ships the files its runs use.  *solver_config_dir* (one folder or a list, highest first)
    replaces the ladder, as the run flag does."""
    from flextool._solver_config_source import gui_solver_config_dirs, split_solver_config_dirs

    given = split_solver_config_dirs(solver_config_dir)  # type: ignore[arg-type]
    return given if given else gui_solver_config_dirs(project_dir)


def effective_solver_config_files(folders: list[Path]) -> dict[str, Path]:
    """``{"<solver>.opt": file}`` of the option files a run with the ladder *folders* reads, per solver the first
    folder's file (see :mod:`flextool._solver_config_source`), leaving out an unedited FlexTool template (it
    equals the default a run without the file uses).  Only the five solver files: no other ``.opt`` is read."""
    from flextool._solver_config_source import (
        REASON_ARGUMENT,
        SOLVER_NAMES,
        SolverConfigSource,
        is_unedited_template,
    )

    if not folders:
        return {}
    source = SolverConfigSource(folders[0], REASON_ARGUMENT, lower=tuple((f, REASON_ARGUMENT) for f in folders[1:]))
    out: dict[str, Path] = {}
    for solver in SOLVER_NAMES:
        found = source.file_for(solver)
        if found is None:
            continue
        try:
            if is_unedited_template(solver, found[0].read_bytes()):
                continue
        except OSError:
            continue
        out[f"{solver}.opt"] = found[0]
    return out


def settle_shipped_solver_config(folder: Path, *, drop_templates: bool) -> tuple[dict[str, list[str]], list[str]]:
    """Screen the solver option files of a shipped project's ``solver_config/`` *folder* (another person's files).

    A ``<solver>.opt`` with an option that names a file or folder outside the solve's work folder, or that
    connects the solver to another computer (:func:`flextool._solver_option_rules.screen_solver_option_file`), is
    renamed to ``<solver>.opt.review`` (its modification time kept): no run reads it until the user has checked
    it and renamed it back.  With *drop_templates* an unedited FlexTool template is removed (it equals the default;
    runs never take one from a project folder).  Returns ``({file name: problems} of the held files, [file names
    of the removed templates]})``.
    """
    from flextool._solver_config_source import SOLVER_NAMES, is_unedited_template
    from flextool._solver_option_rules import screen_solver_option_file

    held: dict[str, list[str]] = {}
    dropped: list[str] = []
    for solver in SOLVER_NAMES:
        path = folder / f"{solver}.opt"
        if not path.is_file():
            continue
        data = path.read_bytes()
        if is_unedited_template(solver, data):
            if drop_templates:
                path.unlink()
                dropped.append(path.name)
            continue
        problems = screen_solver_option_file(solver, data.decode("utf-8", errors="replace"))
        if problems:
            _rename_with_retry(path, path.with_name(path.name + HELD_SUFFIX))
            held[path.name] = problems
    return held, dropped


def held_file_line(name: str, problems: list[str]) -> str:
    """What a held shipped option file waits for (the import report and the first-open log line)."""
    return (f"Solver options {name}: kept as {SOLVER_CONFIG_DIR}/{name}{HELD_SUFFIX} and not used, because "
            f"{'; '.join(problems)}. Check it, then rename it to {name} to use it.")


def default_package_name(project: str, today: datetime | None = None) -> str:
    """``<Name>_<YYYYMMDD>.flextool.zip``."""
    return f"{project}_{(today or datetime.now()).strftime('%Y%m%d')}{PACKAGE_SUFFIX}"


def _versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {}
    for key, dist in (("flextool", "flextool"), ("polar_high", "polar-high"), ("highspy", "highspy"),
                      ("polars", "polars"), ("spinedb_api", "spinedb_api")):
        try:
            out[key] = version(dist)
        except PackageNotFoundError:
            out[key] = None
    return out


def installed_flextool_version() -> str | None:
    return _versions()["flextool"]


def _version_key(text: str) -> tuple:
    try:
        from packaging.version import Version

        return (0, Version(text))
    except Exception:
        return (1, tuple(int(part) for part in re.findall(r"\d+", text)))


def is_newer_version(theirs: str | None, ours: str | None) -> bool:
    """Whether version *theirs* is newer than *ours* (``False`` when either is unknown)."""
    if not theirs or not ours:
        return False
    a, b = _version_key(theirs), _version_key(ours)
    if a[0] != b[0]:
        return False
    return a[1] > b[1]


def _target_db_version() -> int:
    from flextool.update_flextool import FLEXTOOL_DB_VERSION

    return int(FLEXTOOL_DB_VERSION)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_lock_or_journal(name: str) -> bool:
    """Office and SQLite side files, lock files and run-record temp files: never packaged."""
    return (name.startswith(("~$", ".~lock.", RUN_STATUS_NAME + "."))
            or name.endswith(("-journal", "-wal", "-shm", ".lock")))


def component_problem(part: str) -> str | None:
    """Why *part* cannot be a file or folder name on Windows (``None`` when it can)."""
    if any(c in _WINDOWS_BAD_CHARS or ord(c) < 32 for c in part):
        return 'holds a character Windows does not allow in file names (< > : " | ? * or a control character)'
    if part.endswith((".", " ")):
        return "ends with a dot or a space, which Windows drops"
    if part.split(".", 1)[0].rstrip(" ").upper() in _WINDOWS_RESERVED:
        return "is a name Windows reserves for devices (CON, PRN, AUX, NUL, COM1-9, LPT1-9)"
    return None


def _portability_problem(rel: str) -> str | None:
    """Why the relative POSIX path *rel* cannot be written on Windows (``None`` when it can)."""
    for part in rel.split("/"):
        reason = component_problem(part)
        if reason is not None:
            return f"'{rel}': '{part}' {reason}"
    return None


def _case_clashes(rels: list[str]) -> list[str]:
    """Problems of relative POSIX paths that differ only in case, or of a file that is also a folder."""
    problems = []
    seen: dict[str, str] = {}
    folders: dict[str, str] = {}
    for rel in rels:
        parts = rel.split("/")
        for i in range(1, len(parts)):
            folders.setdefault("/".join(parts[:i]).casefold(), "/".join(parts[:i]))
    for rel in rels:
        key = rel.casefold()
        if key in seen:
            problems.append(f"'{seen[key]}' and '{rel}' differ only in case: Windows and macOS keep only one.")
        else:
            seen[key] = rel
        if key in folders:
            problems.append(f"'{rel}' is both a file and the folder of '{folders[key]}/...'.")
    return problems


def _files_under(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and not _is_lock_or_journal(p.name))


def _folder_bytes(folder: Path) -> int:
    total = 0
    for path in _files_under(folder):
        try:
            total += path.stat().st_size
        except OSError:
            pass
    return total


def _load_settings(project_dir: Path) -> dict:
    path = project_dir / SETTINGS_FILE
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise PackageError(f"Unable to read {path}: {exc}") from exc
    return data if isinstance(data, dict) else {}


def _source_file_of_run(project_dir: Path, run: str, settings: dict) -> Path | None:
    """The input file in the project (as it would be shipped) that the run *run* was made from."""
    from flextool.gui.scenario_key import LEGACY_SOURCE_NUMBER, resolve_source_number

    owners = {str(k): int(v) for k, v in (settings.get("bare_output_owners") or {}).items()}
    number, _scenario = resolve_source_number(run, owners)
    if number == LEGACY_SOURCE_NUMBER:
        return None
    record = (settings.get("source_registry") or {}).get(str(number)) or {}
    rel = record.get("path") if isinstance(record, dict) else None
    if not rel:
        names = {v: k for k, v in (settings.get("input_source_numbers") or {}).items()}
        name = names.get(number)
        if name is None:
            return None
        external = (settings.get("external_refs") or {}).get(name)
        rel = external or f"{INPUT_DIR}/{name}"
    return (project_dir / rel).resolve()


def _run_is_stale(project_dir: Path, run: str, settings: dict) -> bool | None:
    """Whether the input file of *run* changed after the run read it (``None`` when it cannot be told)."""
    from flextool.run_record import parse_iso

    status_path = project_dir / RESULTS_DIR / run / "_run_status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    recorded = status.get("input") if isinstance(status, dict) else None
    if not isinstance(recorded, dict):
        return None
    source = _source_file_of_run(project_dir, run, settings)
    if source is None or not source.is_file():
        return None
    then = recorded.get("source_file_mtime") if recorded.get("source_file") else recorded.get("db_mtime")
    stamp = parse_iso(then)
    if stamp is None:
        return None
    return int(source.stat().st_mtime) > int(stamp.timestamp())


def _run_state(folder: Path) -> str | None:
    """The state of the run record in *folder*: a ``running`` record is checked against its process and
    heartbeat (``running``, ``unknown`` or ``abandoned``)."""
    from flextool.run_record import effective_state, read_status_file

    return effective_state(read_status_file(folder))


def list_runs(project_dir: str | Path) -> list[RunInfo]:
    """The result runs of a project: the folders of ``output_parquet`` (not ``_*``), with size, staleness and the
    state of their run record."""
    project_dir = Path(project_dir)
    base = project_dir / RESULTS_DIR
    if not base.is_dir():
        return []
    settings = _load_settings(project_dir)
    runs = []
    for entry in sorted(base.iterdir(), key=lambda p: p.name):
        if not entry.is_dir() or entry.name.startswith(("_", ".")):
            continue
        runs.append(RunInfo(entry.name, _folder_bytes(entry), _run_is_stale(project_dir, entry.name, settings),
                            _run_state(entry)))
    return runs


def active_run_message(names: list[str]) -> str:
    """What to say about ticked runs that are still being written."""
    return (f"These runs are still being written (or their process ended less than 10 minutes ago without "
            f"finishing the run record): {', '.join(names)}. Wait for them to finish, or leave them out.")


def axis_bounds_for_runs(project_dir: str | Path, runs: list[str]) -> bytes | None:
    """``output_parquet/_axis_bounds.json`` cut to *runs*, as the bytes to package (``None``: nothing to ship)."""
    from flextool.plot_outputs.shared_manifest import filter_manifest_to_scenarios, load_axis_bounds_manifest

    if not runs or not (Path(project_dir) / RESULTS_DIR / AXIS_BOUNDS_NAME).is_file():
        return None
    loaded = load_axis_bounds_manifest(Path(project_dir))
    kept = filter_manifest_to_scenarios(loaded, runs) if isinstance(loaded, dict) else {}
    if not kept:
        return None
    return json.dumps(kept, indent=2, sort_keys=True).encode("utf-8")


def _path_replacements(project_dir: Path) -> list[tuple[str, str]]:
    """``(text, placeholder)`` pairs for :func:`scrub_run_status`, longest first."""
    pairs: list[tuple[str, str]] = []
    for text in {str(project_dir), project_dir.as_posix()}:
        pairs.append((text, "<project>"))
    try:
        home = Path.home()
    except (KeyError, RuntimeError):
        home = None
    if home is not None and len(str(home)) > 1:
        for text in {str(home), home.as_posix()}:
            pairs.append((text, "~"))
    return sorted(pairs, key=lambda pair: -len(pair[0]))


#: An absolute folder part of a path (POSIX or Windows) that is not the project or the home folder.  A path may
#: follow ``:`` or ``;`` (the second folder of a ``FLEXTOOL_SOLVER_CONFIG_DIR`` value); a URL's ``://`` is not one.
_ABS_DIR = re.compile(r"(?<![\w./~>\\-])(?:(?:/[^\s/\\\"'<>|:*?]+)+/|[A-Za-z]:[\\/](?:[^\s\\/\"'<>|:*?]+[\\/])*)")
#: Keys of a run record that name this computer or its user.
_PERSONAL_KEYS = frozenset({"host", "hostname", "user", "username"})


def scrub_run_status(status: object, project_dir: Path) -> object:
    """A copy of the run record *status* without this computer's paths, user name and host name.

    The project folder becomes ``<project>``, the home folder ``~``, other absolute folders ``<path>/`` (the file
    name stays), the host and user name ``<host>`` / ``<user>``, and ``host`` / ``user`` fields ``None``.  The times
    and the verdict, which the GUI and the assistant read, are kept.
    """
    pairs = _path_replacements(project_dir)
    words = []
    try:
        host = socket.gethostname()
    except OSError:
        host = ""
    if len(host) >= 3:
        words.append((re.compile(rf"\b{re.escape(host)}\b", re.IGNORECASE), "<host>"))
    try:
        import getpass

        user = getpass.getuser()
    except Exception:  # noqa: BLE001 — no user name (a container): nothing to remove
        user = ""
    if len(user) >= 3:
        words.append((re.compile(rf"\b{re.escape(user)}\b"), "<user>"))

    def text(value: str) -> str:
        for old, new in pairs:
            value = value.replace(old, new)
        value = _ABS_DIR.sub(lambda m: "<path>" + m.group(0)[-1], value)
        for pattern, new in words:
            value = pattern.sub(new, value)
        return value

    def walk(value: object) -> object:
        if isinstance(value, dict):
            return {k: (None if str(k).lower() in _PERSONAL_KEYS else walk(v)) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, str):
            return text(value)
        return value

    return walk(status)


def pip_line(manifest: Manifest) -> str | None:
    """``pip install flextool==X polar-high==Y highspy==Z polars==W`` for the versions the package was made
    with (``None`` when the FlexTool version is unknown)."""
    if not manifest.flextool:
        return None
    pins = [f"{dist}=={ver}" for dist, ver in (("flextool", manifest.flextool), ("polar-high", manifest.polar_high),
                                              ("highspy", manifest.highspy), ("polars", manifest.polars)) if ver]
    return "pip install " + " ".join(pins)


def always_included(project_dir: str | Path, solver_config_dir: object = None) -> dict:
    """What every package of *project_dir* holds: ``{"inputs": [(name, bytes)], "external": [(name, path,
    exists)], "settings": bool, "plot_settings": bool, "solver_config_files": {name: path},
    "solver_config_dirs": [folders]}`` (the option files of :func:`effective_solver_config_files` for the ladder
    of :func:`project_solver_config_dirs`)."""
    project_dir = Path(project_dir)
    settings = _load_settings(project_dir)
    inputs = [(p.relative_to(project_dir / INPUT_DIR).as_posix(), p.stat().st_size)
              for p in _files_under(project_dir / INPUT_DIR)]
    external = []
    for name, rel in (settings.get("external_refs") or {}).items():
        path = (project_dir / rel).resolve()
        external.append((name, path, path.is_file()))
    folders = project_solver_config_dirs(project_dir, solver_config_dir)
    return {
        "inputs": inputs, "external": external, "settings": (project_dir / SETTINGS_FILE).is_file(),
        "plot_settings": (project_dir / PLOT_SETTINGS_FILE).is_file(),
        "solver_config_files": effective_solver_config_files(folders), "solver_config_dirs": folders,
    }


def _bundled_name(name: str, taken: set[str]) -> str:
    """*name*, or ``<stem>_external<n><suffix>`` when a file of that name is in ``input_sources`` already."""
    if name.lower() not in taken:
        return name
    stem, suffix = Path(name).stem, Path(name).suffix
    n = 1
    while True:
        candidate = f"{stem}_external{'' if n == 1 else n}{suffix}"
        if candidate.lower() not in taken:
            return candidate
        n += 1


def _portable_settings(settings: dict, bundled: dict[str, str], external: dict[str, str],
                       left_out: list[str] | None = None) -> dict:
    """*settings* without the machine-specific keys; bundled external sources point at ``input_sources/``,
    under their own source numbers.  The registry path of an external source left out of the package is blanked
    (it named a place on the sender's computer); its name and number stay, so its runs keep their source."""
    out = {k: v for k, v in settings.items() if k not in DROPPED_SETTINGS}
    left_rels = {external[name] for name in (left_out or []) if name in external}
    if not bundled and not left_rels:
        return out
    numbers = dict(out.get("input_source_numbers") or {})
    registry = {k: dict(v) if isinstance(v, dict) else v for k, v in (out.get("source_registry") or {}).items()}
    for record in registry.values():
        if isinstance(record, dict) and record.get("path") in left_rels:
            record["path"] = ""
    checked = list(out.get("checked_input_sources") or [])
    for old, new in bundled.items():
        rel = external[old]
        if old in numbers and new != old:
            numbers[new] = numbers.pop(old)
        number = numbers.get(new)
        new_rel = f"{INPUT_DIR}/{new}"
        for key, record in registry.items():
            if isinstance(record, dict) and (record.get("path") == rel or (
                    number is not None and key == str(number))):
                record["path"] = new_rel
                record["name"] = new
        if number is not None and str(number) not in registry:
            registry[str(number)] = {"name": new, "path": new_rel}
        checked = [new if c == old else c for c in checked]
    if "input_source_numbers" in out or numbers:
        out["input_source_numbers"] = numbers
    if "source_registry" in out or registry:
        out["source_registry"] = registry
    if "checked_input_sources" in out:
        out["checked_input_sources"] = checked
    return out


def _db_facts(path: Path) -> tuple[int | None, str | None]:
    """The data version and the frozen marker value of a sqlite file, read-only."""
    from flextool.update_flextool import frozen as fz

    version = fz.read_db_version_ro(path)
    try:
        info = fz.read_frozen(path)
    except fz.FrozenCheckError as exc:
        raise PackageError(str(exc)) from exc
    return version, (info.to_value() if info is not None else None)


# ---------------------------------------------------------------------------------------------------------------
# Export


@dataclass
class _Entry:
    arcname: str          # under <Name>/
    source: Path | None   # a file to copy, or None for *data*
    data: bytes | None = None
    #: The modification time to record (default: the source's, or now for *data*).
    mtime_ns: int | None = None
    #: Check that *source* does not change while it is copied (``False`` for a snapshot taken by FlexTool).
    watch: bool = True


def _sqlite_has_writer_files(path: Path) -> bool:
    return any(path.with_name(path.name + suffix).exists() for suffix in ("-journal", "-wal"))


def _sqlite_snapshot(path: Path, dest: Path) -> None:
    """A consistent copy of the database *path* in *dest*, read through a read-only connection."""
    uri = path.resolve().as_uri() + "?mode=ro"
    try:
        with closing(sqlite3.connect(uri, uri=True, timeout=30)) as src, closing(sqlite3.connect(dest)) as dst:
            src.backup(dst)
    except sqlite3.Error as exc:
        raise PackageError(f"Unable to take a consistent copy of {path} ({exc}). It may be in the middle of a "
                           "change: close the program that writes to it (the database editor, an import) and "
                           "export again.") from exc


def _zip_time(mtime_ns: int) -> tuple[int, int, int, int, int, int]:
    stamp = min(max(mtime_ns / 1e9, 315_576_000.0), 4_102_444_800.0)   # zip dates: 1980 to 2107
    return time.localtime(stamp)[:6]


def export_project(
    project_dir: str | Path,
    zip_path: str | Path,
    runs: list[str] | None = None,
    include_external: bool = True,
    *,
    solver_config_dir: object = None,
    progress: Callable[[int, int], None] | None = None,
    overwrite: bool = False,
    scrub_run_records: bool = True,
) -> Manifest:
    """Write the package of *project_dir* to *zip_path*; returns its manifest.

    *runs* are the result runs (folder names under ``output_parquet``) to include: ``None`` or ``[]`` includes
    none; a run that is still being written is refused.  *include_external* copies the external input files into
    ``input_sources/`` of the package (same source numbers).  *solver_config_dir* (one folder or a list,
    highest first) defaults to the project's ladder, :func:`project_solver_config_dirs`.  *progress(done_bytes, total_bytes)* is called while the files are written.
    The zip is written next to *zip_path* and renamed into place when complete; an existing *zip_path* raises
    :class:`PackageExistsError` unless *overwrite*.  *scrub_run_records* (the default) replaces this computer's
    paths, user name and host name in each run's ``_run_status.json`` and leaves ``_run.log`` out.  Every failure
    is a :class:`PackageError`.
    """
    zip_path = Path(zip_path)
    try:
        return _export(Path(project_dir), zip_path, runs, include_external, solver_config_dir, progress,
                       overwrite, scrub_run_records)
    except PackageError:
        raise
    except OSError as exc:
        raise PackageError(f"Unable to write the package {zip_path}: {exc}") from exc


def _export(
    project_dir: Path,
    zip_path: Path,
    runs: list[str] | None,
    include_external: bool,
    solver_config_dir: object,
    progress: Callable[[int, int], None] | None,
    overwrite: bool,
    scrub_run_records: bool,
) -> Manifest:
    project_dir = project_dir.resolve()
    if not project_dir.is_dir():
        raise PackageError(f"No project folder at {project_dir}")
    name = project_dir.name
    if zip_path.resolve().is_relative_to(project_dir):
        raise PackageError("Write the package outside the project folder.")
    if zip_path.exists() and not overwrite:
        raise PackageExistsError(zip_path)
    settings = _load_settings(project_dir)
    available = {r.name: r for r in list_runs(project_dir)}
    runs = list(dict.fromkeys(runs or []))
    unknown = [r for r in runs if r not in available]
    if unknown:
        raise PackageError(f"No result run named {', '.join(unknown)} in {project_dir / RESULTS_DIR}")
    active = [r for r in runs if available[r].active]
    if active:
        raise PackageError(active_run_message(active))

    warnings: list[str] = []
    entries: list[_Entry] = []
    inputs = _files_under(project_dir / INPUT_DIR)
    taken = {p.relative_to(project_dir / INPUT_DIR).as_posix().lower() for p in inputs}
    for path in inputs:
        entries.append(_Entry(f"{INPUT_DIR}/{path.relative_to(project_dir / INPUT_DIR).as_posix()}", path))
        if path.with_name("~$" + path.name).exists():
            warnings.append(f"{path.name} is open in Excel: the package holds its last saved version.")

    external = {str(k): str(v) for k, v in (settings.get("external_refs") or {}).items()}
    bundled: dict[str, str] = {}
    not_included: list[str] = []
    for ext_name, rel in external.items():
        src = (project_dir / rel).resolve()
        if not include_external or not src.is_file():
            not_included.append(ext_name)
            continue
        packaged = _bundled_name(ext_name, taken)
        taken.add(packaged.lower())
        bundled[ext_name] = packaged
        entries.append(_Entry(f"{INPUT_DIR}/{packaged}", src))
        if src.with_name("~$" + src.name).exists():
            warnings.append(f"{src.name} is open in Excel: the package holds its last saved version.")

    if settings or (project_dir / SETTINGS_FILE).is_file():
        portable = _portable_settings(settings, bundled, external, not_included)
        text = yaml.dump(portable, default_flow_style=False, sort_keys=False, allow_unicode=True)
        entries.append(_Entry(SETTINGS_FILE, None, text.encode("utf-8")))
    if (project_dir / PLOT_SETTINGS_FILE).is_file():
        entries.append(_Entry(PLOT_SETTINGS_FILE, project_dir / PLOT_SETTINGS_FILE))

    opts = effective_solver_config_files(project_solver_config_dirs(project_dir, solver_config_dir))
    for opt_name, path in opts.items():
        entries.append(_Entry(f"{SOLVER_CONFIG_DIR}/{opt_name}", path))

    if runs:
        base = project_dir / RESULTS_DIR
        bounds = axis_bounds_for_runs(project_dir, runs)
        if bounds is not None:
            entries.append(_Entry(f"{RESULTS_DIR}/{AXIS_BOUNDS_NAME}", None, bounds,
                                  (base / AXIS_BOUNDS_NAME).stat().st_mtime_ns))
        for run in runs:
            for path in _files_under(base / run):
                arcname = f"{RESULTS_DIR}/{path.relative_to(base).as_posix()}"
                if scrub_run_records and path.parent == base / run and path.name == RUN_LOG_NAME:
                    continue
                if scrub_run_records and path.parent == base / run and path.name == RUN_STATUS_NAME:
                    try:
                        status = json.loads(path.read_text(encoding="utf-8"))
                    except ValueError as exc:
                        raise PackageError(f"The run record {path} cannot be read: {exc}") from exc
                    data = json.dumps(scrub_run_status(status, project_dir), indent=2, ensure_ascii=False)
                    entries.append(_Entry(arcname, None, data.encode("utf-8"), path.stat().st_mtime_ns))
                    continue
                entries.append(_Entry(arcname, path))

    rels = [e.arcname for e in entries]
    problems = [p for p in (_portability_problem(rel) for rel in rels) if p] + _case_clashes(rels)
    if problems:
        raise PackageError("These files cannot be packaged, because a package must unpack on Windows, macOS "
                           "and Linux; rename them: " + " ".join(problems))

    versions = _versions()
    manifest = Manifest(
        project=name, created=datetime.now().astimezone().isoformat(timespec="seconds"),
        python=platform.python_version(), platform=sys.platform, results=runs,
        stale_runs=[r for r in runs if available[r].stale], dropped_settings=list(DROPPED_SETTINGS),
        bundled_external=bundled, external_not_included=not_included, solver_config=list(opts),
        run_records_scrubbed=scrub_run_records, export_warnings=warnings, **versions,
    )

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    partial = zip_path.with_name(zip_path.name + ".partial")
    scratch = Path(tempfile.mkdtemp(prefix=".flextool-export-", dir=zip_path.parent))
    try:
        for i, entry in enumerate(entries):
            if entry.source is None or not entry.arcname.lower().endswith(".sqlite"):
                continue
            if _sqlite_has_writer_files(entry.source):
                snapshot = scratch / f"{i}.sqlite"
                mtime_ns = entry.source.stat().st_mtime_ns
                _sqlite_snapshot(entry.source, snapshot)
                entry.source, entry.mtime_ns, entry.watch = snapshot, mtime_ns, False
            version, frozen_value = _db_facts(entry.source)
            manifest.db_versions[entry.arcname] = version
            if version is None:
                warnings.append(f"The data version of {entry.arcname} could not be read (locked, or not a "
                                "FlexTool database): the receiver is not warned if it is newer than theirs.")
            if frozen_value is not None:
                manifest.frozen[entry.arcname] = frozen_value

        total = sum(e.source.stat().st_size if e.source is not None else len(e.data or b"") for e in entries)
        done = 0
        with zipfile.ZipFile(partial, "w", allowZip64=True) as zf:
            for entry in entries:
                arcname = f"{name}/{entry.arcname}"
                before = entry.source.stat() if entry.source is not None else None
                if entry.mtime_ns is not None:
                    mtime_ns = entry.mtime_ns
                elif before is not None:
                    mtime_ns = before.st_mtime_ns
                else:
                    mtime_ns = time.time_ns()
                info = zipfile.ZipInfo(arcname, _zip_time(mtime_ns))
                info.external_attr = 0o644 << 16
                info.compress_type = (zipfile.ZIP_STORED if arcname.lower().endswith(".parquet")
                                      else zipfile.ZIP_DEFLATED)
                digest = hashlib.sha256()
                size = 0
                with zf.open(info, "w", force_zip64=True) as dst:
                    if entry.source is not None:
                        with open(entry.source, "rb") as src:
                            for chunk in iter(lambda: src.read(_CHUNK), b""):
                                digest.update(chunk)
                                dst.write(chunk)
                                size += len(chunk)
                                done += len(chunk)
                                if progress is not None:
                                    progress(done, total)
                    else:
                        data = entry.data or b""
                        digest.update(data)
                        dst.write(data)
                        size = len(data)
                        done += size
                if entry.source is not None and entry.watch and before is not None:
                    after = entry.source.stat()
                    if after.st_mtime_ns != before.st_mtime_ns or after.st_size != size:
                        raise PackageError(f"{entry.source} changed while it was being exported. Wait until "
                                           "the program writing it is done and export again.")
                manifest.files[arcname] = {"sha256": digest.hexdigest(), "bytes": size, "mtime_ns": mtime_ns}
            zf.writestr(MANIFEST_NAME, json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False),
                        compress_type=zipfile.ZIP_DEFLATED)
        os.replace(partial, zip_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if progress is not None:
        progress(total, total)
    return manifest


# ---------------------------------------------------------------------------------------------------------------
# Reading and checking a package


def _unfinished_message(zip_path: str | Path) -> str | None:
    if str(zip_path).endswith(".partial"):
        return f"{zip_path} is an unfinished package (its export was stopped); export it again."
    return None


def read_manifest(zip_path: str | Path) -> Manifest:
    """The manifest of a package, without unpacking anything."""
    if (message := _unfinished_message(zip_path)) is not None:
        raise PackageError(message)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            return _manifest_of(zf)
    except (OSError, zipfile.BadZipFile) as exc:
        raise PackageError(f"Unable to read {zip_path}: {exc}") from exc


def _manifest_of(zf: zipfile.ZipFile) -> Manifest:
    try:
        raw = zf.read(MANIFEST_NAME)
    except KeyError:
        raise PackageError("This file is not a FlexTool project package (it has no manifest.json).") from None
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise PackageError(f"The manifest.json of the package cannot be read: {exc}") from exc
    return Manifest.from_dict(data)


def _unsafe(name: str) -> bool:
    """A member name that is absolute or leaves its folder (zip-slip)."""
    if not name or name.startswith(("/", "\\")) or "\\" in name or re.match(r"^[A-Za-z]:", name):
        return True
    return any(part in ("..", "") for part in PurePosixPath(name).parts) or name != PurePosixPath(name).as_posix()


def _dest_rel(rel: str) -> str:
    """Where the member *rel* (relative to ``<Name>/``) is unpacked in the project: at the same place (the
    shipped solver options too: ``solver_config/``)."""
    return rel


def _facts_problem(arcname: str, facts: object) -> str | None:
    """Why the manifest entry *facts* of *arcname* is not a valid ``{"sha256", "bytes", "mtime_ns"}``."""
    if not isinstance(facts, dict) or not isinstance(facts.get("sha256"), str):
        return f"Manifest entry '{arcname}' has no checksum."
    size = facts.get("bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        return f"Manifest entry '{arcname}' has no valid size."
    mtime_ns = facts.get("mtime_ns")
    if mtime_ns is not None and (isinstance(mtime_ns, bool) or not isinstance(mtime_ns, int)
                                 or not 0 <= mtime_ns <= _MAX_MTIME_NS):
        return f"Manifest entry '{arcname}' has a modification time that is not a time ({mtime_ns!r})."
    return None


def _check_members(zf: zipfile.ZipFile, manifest: Manifest) -> list[str]:
    """Problems with the member names and the manifest entries: unsafe or non-portable names, names that differ
    only in case, members not in the manifest, manifest files not there, sizes and times that are not valid."""
    problems = []
    prefix = manifest.project + "/"
    if _unsafe(manifest.project) or "/" in manifest.project:
        problems.append(f"The project name '{manifest.project}' in the manifest is not a folder name.")
    elif (reason := component_problem(manifest.project)) is not None:
        problems.append(f"The project name '{manifest.project}' in the manifest {reason}.")
    names = set()
    for info in zf.infolist():
        if info.filename == MANIFEST_NAME or info.is_dir():
            continue
        names.add(info.filename)
        if _unsafe(info.filename) or not info.filename.startswith(prefix):
            problems.append(f"Member '{info.filename}' would be written outside the project folder.")
        elif info.filename not in manifest.files:
            problems.append(f"Member '{info.filename}' is not listed in the manifest.")
    dests = []
    for listed, facts in manifest.files.items():
        if _unsafe(listed) or not listed.startswith(prefix):
            problems.append(f"Manifest entry '{listed}' would be written outside the project folder.")
            continue
        if listed not in names:
            problems.append(f"File '{listed}' of the manifest is not in the package.")
        reason = _portability_problem(listed[len(prefix):])
        if reason is not None:
            problems.append(f"Member {reason}.")
        reason = _facts_problem(listed, facts)
        if reason is not None:
            problems.append(reason)
        dests.append(_dest_rel(listed[len(prefix):]))
    problems.extend(_case_clashes(dests))
    return problems


def _check_hashes(zf: zipfile.ZipFile, manifest: Manifest) -> tuple[list[str], int]:
    problems = []
    checked = 0
    for arcname, facts in manifest.files.items():
        try:
            info = zf.getinfo(arcname)
        except KeyError:
            continue  # reported by _check_members
        digest = hashlib.sha256()
        size = 0
        try:
            with zf.open(info) as src:
                for chunk in iter(lambda: src.read(_CHUNK), b""):
                    digest.update(chunk)
                    size += len(chunk)
        except (zipfile.BadZipFile, OSError) as exc:
            problems.append(f"{arcname}: cannot be read ({exc}).")
            continue
        checked += 1
        if digest.hexdigest() != facts.get("sha256") or size != facts.get("bytes"):
            problems.append(f"{arcname}: the checksum does not match the manifest (the file is damaged).")
    return problems, checked


def version_messages(manifest: Manifest) -> tuple[list[str], list[str]]:
    """``(blocking problems, warnings)`` of the manifest's versions against this FlexTool."""
    problems, warnings = [], []
    target = _target_db_version()
    for path, version in sorted(manifest.db_versions.items()):
        if isinstance(version, int) and version > target:
            problems.append(f"{path} is at FlexTool data version {version}; this FlexTool reads up to version "
                            f"{target}. It cannot be opened: upgrade FlexTool.")
        elif isinstance(version, int) and version < target:
            frozen = " It is frozen: FlexTool asks before migrating it." if path in manifest.frozen else ""
            warnings.append(f"{path} is at data version {version}; it is migrated to version {target} when the "
                            f"project is opened.{frozen}")
    ours = installed_flextool_version()
    if is_newer_version(manifest.flextool, ours):
        warnings.append(f"The package was made with FlexTool {manifest.flextool}, newer than this FlexTool "
                        f"({ours}): results and inputs may use features this version lacks.")
    installed = _versions()
    for key, label in (("polar_high", "polar-high"), ("highspy", "highspy")):
        theirs, mine = getattr(manifest, key), installed.get(key)
        if theirs and mine and theirs != mine:
            warnings.append(f"The package was made with {label} {theirs}; this computer has {mine}. A re-run "
                            "can differ slightly.")
    if manifest.stale_runs:
        warnings.append("The input of these runs changed after the run: " + ", ".join(manifest.stale_runs))
    for name in manifest.external_not_included:
        warnings.append(f"Input {name} was not in the package (it was outside the project folder and left out, "
                        "or missing): its scenarios cannot be run again until you add that file.")
    return problems, warnings


def verify_package(zip_path: str | Path) -> VerifyReport:
    """Check a package without unpacking it: member names, every checksum, and the versions."""
    if (message := _unfinished_message(zip_path)) is not None:
        return VerifyReport(None, [message])
    try:
        zf = zipfile.ZipFile(zip_path)
    except (OSError, zipfile.BadZipFile) as exc:
        return VerifyReport(None, [f"Unable to read {zip_path}: {exc}"])
    with zf:
        try:
            manifest = _manifest_of(zf)
        except PackageError as exc:
            return VerifyReport(None, [str(exc)])
        problems = _check_members(zf, manifest)
        hash_problems, checked = _check_hashes(zf, manifest)
        version_problems, warnings = version_messages(manifest)
    return VerifyReport(manifest, problems + hash_problems + version_problems, warnings, checked)


# ---------------------------------------------------------------------------------------------------------------
# Import


def name_taken(projects_dir: str | Path, name: str) -> bool:
    """Whether *projects_dir* holds a project (or any entry) called *name*, compared case-insensitively."""
    projects_dir = Path(projects_dir)
    if not projects_dir.is_dir():
        return False
    lowered = name.lower()
    return any(entry.name.lower() == lowered for entry in projects_dir.iterdir())


def suggest_name(projects_dir: str | Path, name: str) -> str:
    """*name* when it is free, else the first free ``<name>_<n>`` (n = 2, 3, ...)."""
    if not name_taken(projects_dir, name):
        return name
    n = 2
    while name_taken(projects_dir, f"{name}_{n}"):
        n += 1
    return f"{name}_{n}"


def invalid_name_reason(name: str) -> str | None:
    """Why *name* cannot be a project folder name (``None`` when it can)."""
    if not name.strip():
        return "Enter a project name."
    if "/" in name or "\\" in name or name in (".", "..") or name.startswith("."):
        return "A project name cannot hold '/' or '\\' or start with '.'."
    if name != name.strip():
        return "A project name cannot start or end with a space."
    reason = component_problem(name)
    if reason is not None:
        return f"A project name cannot be used on Windows when it {reason}."
    return None


def solver_config_activation_hint(project_path: Path, workspace_dirs: list[Path]) -> str:
    """Where the shipped solver options of an imported project are and how its runs use them."""
    workspace = ", ".join(str(folder) for folder in workspace_dirs)
    return (f"The shipped solver options are in {project_path / SOLVER_CONFIG_DIR}. Runs of this project use "
            f"them ahead of the workspace's ({workspace}), file by file; delete a file there to use the "
            "workspace's.")


def adopt_imported_solver_config(project_path: str | Path) -> list[str]:
    """Rename a FlexTool 4.3 import's ``imported_solver_config/`` to ``solver_config/`` (its runs then use the
    shipped files), screened as at import (:func:`settle_shipped_solver_config`; unedited templates stay, runs
    skip them).  Only when ``solver_config/`` does not exist.  Returns the lines to log: what was done, one per
    held file; when both folders exist, one line saying ``imported_solver_config/`` is not read; nothing when
    there is no ``imported_solver_config/``.  A rename that fails gives one line and leaves the folder as it is.
    """
    project_path = Path(project_path)
    imported = project_path / IMPORTED_SOLVER_CONFIG_DIR
    own = project_path / SOLVER_CONFIG_DIR
    name = project_path.name
    if not imported.is_dir():
        return []
    if own.exists():
        return [f"Project {name}: {IMPORTED_SOLVER_CONFIG_DIR}/ is not read; this project's runs use "
                f"{SOLVER_CONFIG_DIR}/ (move any file you need from it and delete {IMPORTED_SOLVER_CONFIG_DIR}/)."]
    try:
        _rename_with_retry(imported, own)
    except OSError as exc:
        return [f"Project {name}: cannot rename {IMPORTED_SOLVER_CONFIG_DIR}/ to {SOLVER_CONFIG_DIR}/ ({exc}); "
                "this project's runs keep using the workspace's solver options."]
    try:
        held, _dropped = settle_shipped_solver_config(own, drop_templates=False)
    except OSError as exc:
        held = {}
        lines = [f"Project {name}: cannot check the solver option files in {own} ({exc})."]
    else:
        lines = []
    from flextool._solver_config_source import SOLVER_NAMES

    active = [f"{s}.opt" for s in SOLVER_NAMES if (own / f"{s}.opt").is_file()]
    lines.insert(0, f"Project {name}: renamed {IMPORTED_SOLVER_CONFIG_DIR}/ to {SOLVER_CONFIG_DIR}/. This "
                    f"project's runs now use its solver option files ({', '.join(active) or 'none'}) ahead of the "
                    "workspace's.")
    lines.extend(f"Project {name}: " + held_file_line(file_name, problems) for file_name, problems in held.items())
    return lines


def import_project(
    zip_path: str | Path,
    projects_dir: str | Path,
    name: str | None = None,
    *,
    solver_config_dir: object = None,
) -> ImportReport:
    """Unpack the package *zip_path* as project *name* (default: the manifest's name) in *projects_dir*.

    Raises :class:`NameClashError` when the name is taken (case-insensitively) and :class:`PackageError` when a
    member name is unsafe, a checksum does not match or a database is newer than this FlexTool; nothing is written
    then.  A disk without room for the unpacked files, a folder that cannot be written and the like are a
    :class:`PackageError` too; the half-written folder is removed.  The shipped solver option files go to
    ``solver_config/`` of the project, screened (:func:`settle_shipped_solver_config`).  *solver_config_dir* (the
    workspace's folders, highest first; default ``$FLEXTOOL_SOLVER_CONFIG_DIR``, else ``./solver_config``) is
    only compared with them, never written.
    """
    projects_dir = Path(projects_dir)
    try:
        return _import(zip_path, projects_dir, name, solver_config_dir)
    except PackageError:
        raise
    except (OSError, OverflowError, ValueError) as exc:
        raise PackageError(f"Unable to import {zip_path} into {projects_dir}: {exc}") from exc


def _free_bytes(folder: Path) -> int | None:
    probe = folder
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return None


def _rename_with_retry(src: Path, dst: Path) -> None:
    """``src.rename(dst)``, tried again a few times: on Windows an antivirus scan can hold a new file briefly."""
    for attempt in range(4):
        try:
            src.rename(dst)
            return
        except PermissionError:
            if attempt == 3:
                raise
            time.sleep(0.5)


def _import(
    zip_path: str | Path, projects_dir: Path, name: str | None, solver_config_dir: object,
) -> ImportReport:
    if (message := _unfinished_message(zip_path)) is not None:
        raise PackageError(message)
    try:
        zf = zipfile.ZipFile(zip_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise PackageError(f"Unable to read {zip_path}: {exc}") from exc
    with zf:
        manifest = _manifest_of(zf)
        target_name = manifest.project if name is None else name
        reason = invalid_name_reason(target_name)
        if reason is not None:
            raise PackageError(reason)
        if name_taken(projects_dir, target_name):
            raise NameClashError(target_name, suggest_name(projects_dir, target_name))
        problems = _check_members(zf, manifest)
        if problems:
            raise PackageError("The package is refused: " + " ".join(problems))
        version_problems, warnings = version_messages(manifest)
        if version_problems:
            raise PackageError(" ".join(version_problems))
        needed = sum(facts["bytes"] for facts in manifest.files.values())
        free = _free_bytes(projects_dir)
        if free is not None and needed + _DISK_MARGIN_BYTES > free:
            raise PackageError(f"The project needs {needed / 1e6:.1f} MB unpacked, but the disk of {projects_dir} "
                               f"has {free / 1e6:.1f} MB free. Free some space and import again.")
        hash_problems, _checked = _check_hashes(zf, manifest)
        if hash_problems:
            raise PackageError("The package is damaged; nothing was imported. " + " ".join(hash_problems))

        projects_dir.mkdir(parents=True, exist_ok=True)
        final = projects_dir / target_name
        partial = projects_dir / f".{target_name}.partial"
        if partial.exists():
            shutil.rmtree(partial)
        prefix = manifest.project + "/"
        try:
            partial.mkdir()
            for arcname, facts in manifest.files.items():
                dest = partial / _dest_rel(arcname[len(prefix):])
                if not dest.resolve().is_relative_to(partial.resolve()):
                    raise PackageError(f"Member '{arcname}' would be written outside the project folder.")
                dest.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with zf.open(arcname) as src, open(dest, "wb") as dst:
                    for chunk in iter(lambda: src.read(_CHUNK), b""):
                        digest.update(chunk)
                        dst.write(chunk)
                if digest.hexdigest() != facts.get("sha256"):
                    raise PackageError(f"{arcname}: the checksum changed while unpacking; nothing was imported.")
                mtime_ns = facts.get("mtime_ns")
                if isinstance(mtime_ns, int):
                    os.utime(dest, ns=(mtime_ns, mtime_ns))
            for sub in _PROJECT_SUBDIRS:
                (partial / sub).mkdir(exist_ok=True)
            held, dropped = settle_shipped_solver_config(partial / SOLVER_CONFIG_DIR, drop_templates=True)
            if name_taken(projects_dir, target_name):
                raise NameClashError(target_name, suggest_name(projects_dir, target_name))
            _rename_with_retry(partial, final)
        except BaseException:
            shutil.rmtree(partial, ignore_errors=True)
            raise

    report = ImportReport(target_name, final, manifest, warnings, solver_config_held=held,
                          solver_config_defaults=dropped)
    from flextool._solver_config_source import (
        SOLVER_NAMES,
        split_solver_config_dirs,
        workspace_solver_config_dirs,
    )

    own = final / SOLVER_CONFIG_DIR
    shipped = [(f"{s}.opt", own / f"{s}.opt") for s in SOLVER_NAMES if (own / f"{s}.opt").is_file()]
    shipped += [(file_name, own / (file_name + HELD_SUFFIX)) for file_name in held]
    if shipped:
        workspace = split_solver_config_dirs(solver_config_dir) or workspace_solver_config_dirs()  # type: ignore[arg-type]
        theirs = effective_solver_config_files(workspace)
        for file_name, path in sorted(shipped):
            mine = theirs.get(file_name)
            if mine is None:
                state = "not in workspace"
            elif _hash_file(mine) == _hash_file(path):
                state = "same"
            else:
                state = "different"
                report.solver_config_diffs[file_name] = _short_diff(mine, path)
            report.solver_config.append((file_name, state))
        if held or any(state != "same" for _name, state in report.solver_config):
            report.solver_config_hint = solver_config_activation_hint(final, workspace)
    return report


def _short_diff(mine: Path, shipped: Path, limit: int = 40) -> list[str]:
    """The unified diff of two small text files (at most *limit* lines)."""
    import difflib

    def lines(path: Path) -> list[str]:
        try:
            return path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []

    diff = list(difflib.unified_diff(lines(mine), lines(shipped), "workspace", "shipped", lineterm="", n=0))
    return diff[:limit] + (["..."] if len(diff) > limit else [])


def describe_import(report: ImportReport) -> list[str]:
    """Lines that tell the user what an import did (for the CLI and the GUI)."""
    lines = [f"Imported project '{report.name}' to {report.project_path}."]
    if report.manifest.results:
        lines.append("Result runs: " + ", ".join(report.manifest.results))
    lines.extend(report.warnings)
    for name, state in report.solver_config:
        if state == "same":
            lines.append(f"Solver options {name}: the same as this workspace's.")
        elif state == "different":
            diff = report.solver_config_diffs.get(name) or []
            lines.append(f"Solver options {name}: different from this workspace's."
                         + ("\n" + "\n".join(diff) if diff else ""))
        else:
            lines.append(f"Solver options {name}: this workspace has none.")
    for name, problems in report.solver_config_held.items():
        lines.append(held_file_line(name, problems))
    for name in report.solver_config_defaults:
        lines.append(f"Solver options {name}: FlexTool's default options, not unpacked (runs use the workspace's "
                     "file or FlexTool's default).")
    if report.solver_config_hint:
        lines.append(report.solver_config_hint)
    return lines
