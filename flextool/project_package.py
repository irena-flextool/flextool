"""Project packages: export a FlexTool project to one zip file, check one, and import one (no Tk).

A package is ``<Name>_<YYYYMMDD>.flextool.zip``.  Every project file is under ``<Name>/``; ``manifest.json`` at the
root lists the FlexTool, solver and library versions, the data version and frozen marker of each input database,
the result runs it holds, the settings it left out, and the SHA-256, size and modification time (``mtime_ns``) of
every file.

What a package holds:

* always: ``input_sources/`` (with the external input files copied in, under the same source numbers, unless the
  caller says not to), ``settings.yaml`` without the machine-specific settings (:data:`DROPPED_SETTINGS`),
  ``plot_settings.yaml``, and the ``.opt`` files of the workspace's ``solver_config`` folder (as
  ``solver_config/``; import puts them in ``imported_solver_config/`` of the project and never changes the
  workspace's own files);
* the result runs the caller picks (``output_parquet/<run>/``, with ``output_parquet/_*.json`` when there is one);
* never: what FlexTool makes again from the results or uses as scratch (:data:`REGENERATED`), such as
  ``output_csv/``, ``output_plots/`` and ``results.sqlite``.

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
import sys
import time
import zipfile
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
#: Where import puts the shipped solver option files inside the project.
IMPORTED_SOLVER_CONFIG_DIR = "imported_solver_config"

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


class PackageError(RuntimeError):
    """A package cannot be written, read or imported."""


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

    def to_dict(self) -> dict:
        return {
            "format": self.format, "format_version": self.format_version, "created": self.created,
            "project": self.project, "flextool": self.flextool, "polar_high": self.polar_high,
            "highspy": self.highspy, "polars": self.polars, "spinedb_api": self.spinedb_api,
            "python": self.python, "platform": self.platform, "db_versions": self.db_versions,
            "frozen": self.frozen, "results": self.results, "stale_runs": self.stale_runs,
            "dropped_settings": self.dropped_settings, "bundled_external": self.bundled_external,
            "external_not_included": self.external_not_included, "solver_config": self.solver_config,
            "files": self.files,
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
        )


@dataclass
class ImportReport:
    """What :func:`import_project` did."""

    name: str
    project_path: Path
    manifest: Manifest
    warnings: list[str] = field(default_factory=list)
    #: ``(file name, "same" | "different" | "not in workspace")`` per shipped ``.opt`` file.
    solver_config: list[tuple[str, str]] = field(default_factory=list)
    #: How to make the shipped solver options active ("" when the package has none).
    solver_config_hint: str = ""


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


def default_solver_config_dir() -> Path:
    """The folder of the solver option files a run uses: ``$FLEXTOOL_SOLVER_CONFIG_DIR``, else
    ``<workspace>/solver_config`` (the current folder, where FlexTool and the GUI run)."""
    env = os.environ.get("FLEXTOOL_SOLVER_CONFIG_DIR")
    return Path(env) if env else Path.cwd() / SOLVER_CONFIG_DIR


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
    return name.startswith("~$") or name.startswith(".~lock.") or name.endswith(("-journal", ".lock"))


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


def list_runs(project_dir: str | Path) -> list[RunInfo]:
    """The result runs of a project: the folders of ``output_parquet`` (not ``_*``), with size and staleness."""
    project_dir = Path(project_dir)
    base = project_dir / RESULTS_DIR
    if not base.is_dir():
        return []
    settings = _load_settings(project_dir)
    runs = []
    for entry in sorted(base.iterdir(), key=lambda p: p.name):
        if not entry.is_dir() or entry.name.startswith(("_", ".")):
            continue
        runs.append(RunInfo(entry.name, _folder_bytes(entry), _run_is_stale(project_dir, entry.name, settings)))
    return runs


def always_included(project_dir: str | Path, solver_config_dir: Path | None = None) -> dict:
    """What every package of *project_dir* holds: ``{"inputs": [(name, bytes)], "external": [(name, path,
    exists)], "settings": bool, "plot_settings": bool, "solver_config": [names], "solver_config_dir": Path}``."""
    project_dir = Path(project_dir)
    settings = _load_settings(project_dir)
    inputs = [(p.relative_to(project_dir / INPUT_DIR).as_posix(), p.stat().st_size)
              for p in _files_under(project_dir / INPUT_DIR)]
    external = []
    for name, rel in (settings.get("external_refs") or {}).items():
        path = (project_dir / rel).resolve()
        external.append((name, path, path.is_file()))
    config_dir = solver_config_dir or default_solver_config_dir()
    opts = sorted(p.name for p in config_dir.glob("*.opt") if p.is_file()) if config_dir.is_dir() else []
    return {
        "inputs": inputs, "external": external, "settings": (project_dir / SETTINGS_FILE).is_file(),
        "plot_settings": (project_dir / PLOT_SETTINGS_FILE).is_file(), "solver_config": opts,
        "solver_config_dir": config_dir,
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


def _portable_settings(settings: dict, bundled: dict[str, str], external: dict[str, str]) -> dict:
    """*settings* without the machine-specific keys; bundled external sources point at ``input_sources/``,
    under their own source numbers."""
    out = {k: v for k, v in settings.items() if k not in DROPPED_SETTINGS}
    if not bundled:
        return out
    numbers = dict(out.get("input_source_numbers") or {})
    registry = {k: dict(v) if isinstance(v, dict) else v for k, v in (out.get("source_registry") or {}).items()}
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
    mtime_ns: int | None = None


def export_project(
    project_dir: str | Path,
    zip_path: str | Path,
    runs: list[str] | None = None,
    include_external: bool = True,
    *,
    solver_config_dir: str | Path | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> Manifest:
    """Write the package of *project_dir* to *zip_path*; returns its manifest.

    *runs* are the result runs (folder names under ``output_parquet``) to include: ``None`` or ``[]`` includes
    none.  *include_external* copies the external input files into ``input_sources/`` of the package (same source
    numbers).  *solver_config_dir* defaults to :func:`default_solver_config_dir`.  *progress(done_bytes,
    total_bytes)* is called while the files are written.  The zip is written next to *zip_path* and renamed into
    place when complete.
    """
    project_dir = Path(project_dir).resolve()
    if not project_dir.is_dir():
        raise PackageError(f"No project folder at {project_dir}")
    name = project_dir.name
    zip_path = Path(zip_path)
    if zip_path.resolve().is_relative_to(project_dir):
        raise PackageError("Write the package outside the project folder.")
    settings = _load_settings(project_dir)
    available = {r.name: r for r in list_runs(project_dir)}
    runs = list(dict.fromkeys(runs or []))
    unknown = [r for r in runs if r not in available]
    if unknown:
        raise PackageError(f"No result run named {', '.join(unknown)} in {project_dir / RESULTS_DIR}")

    entries: list[_Entry] = []
    inputs = _files_under(project_dir / INPUT_DIR)
    taken = {p.relative_to(project_dir / INPUT_DIR).as_posix().lower() for p in inputs}
    for path in inputs:
        entries.append(_Entry(f"{INPUT_DIR}/{path.relative_to(project_dir / INPUT_DIR).as_posix()}", path))

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

    if settings or (project_dir / SETTINGS_FILE).is_file():
        portable = _portable_settings(settings, bundled, external)
        text = yaml.dump(portable, default_flow_style=False, sort_keys=False, allow_unicode=True)
        entries.append(_Entry(SETTINGS_FILE, None, text.encode("utf-8")))
    if (project_dir / PLOT_SETTINGS_FILE).is_file():
        entries.append(_Entry(PLOT_SETTINGS_FILE, project_dir / PLOT_SETTINGS_FILE))

    config_dir = Path(solver_config_dir) if solver_config_dir is not None else default_solver_config_dir()
    opts = sorted(p for p in config_dir.glob("*.opt") if p.is_file()) if config_dir.is_dir() else []
    for path in opts:
        entries.append(_Entry(f"{SOLVER_CONFIG_DIR}/{path.name}", path))

    if runs:
        base = project_dir / RESULTS_DIR
        for path in sorted(base.glob("_*")):
            if path.is_file():
                entries.append(_Entry(f"{RESULTS_DIR}/{path.name}", path))
        for run in runs:
            for path in _files_under(base / run):
                entries.append(_Entry(f"{RESULTS_DIR}/{path.relative_to(base).as_posix()}", path))

    versions = _versions()
    manifest = Manifest(
        project=name, created=datetime.now().astimezone().isoformat(timespec="seconds"),
        python=platform.python_version(), platform=sys.platform, results=runs,
        stale_runs=[r for r in runs if available[r].stale], dropped_settings=list(DROPPED_SETTINGS),
        bundled_external=bundled, external_not_included=not_included, solver_config=[p.name for p in opts],
        **versions,
    )
    for entry in entries:
        if entry.source is not None and entry.arcname.lower().endswith(".sqlite"):
            version, frozen_value = _db_facts(entry.source)
            manifest.db_versions[entry.arcname] = version
            if frozen_value is not None:
                manifest.frozen[entry.arcname] = frozen_value

    total = sum(e.source.stat().st_size if e.source is not None else len(e.data or b"") for e in entries)
    done = 0
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    partial = zip_path.with_name(zip_path.name + ".partial")
    try:
        with zipfile.ZipFile(partial, "w", allowZip64=True) as zf:
            for entry in entries:
                arcname = f"{name}/{entry.arcname}"
                compress = zipfile.ZIP_STORED if arcname.lower().endswith(".parquet") else zipfile.ZIP_DEFLATED
                if entry.source is not None:
                    stat = entry.source.stat()
                    mtime_ns = stat.st_mtime_ns
                    info = zipfile.ZipInfo.from_file(entry.source, arcname, strict_timestamps=False)
                else:
                    mtime_ns = time.time_ns()
                    info = zipfile.ZipInfo(arcname, datetime.now().timetuple()[:6])
                    info.external_attr = 0o644 << 16
                info.compress_type = compress
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
                manifest.files[arcname] = {"sha256": digest.hexdigest(), "bytes": size, "mtime_ns": mtime_ns}
            zf.writestr(MANIFEST_NAME, json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False),
                        compress_type=zipfile.ZIP_DEFLATED)
        os.replace(partial, zip_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    if progress is not None:
        progress(total, total)
    return manifest


# ---------------------------------------------------------------------------------------------------------------
# Reading and checking a package


def read_manifest(zip_path: str | Path) -> Manifest:
    """The manifest of a package, without unpacking anything."""
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


def _check_members(zf: zipfile.ZipFile, manifest: Manifest) -> list[str]:
    """Problems with the member names: unsafe names, members not in the manifest, manifest files not there."""
    problems = []
    prefix = manifest.project + "/"
    if _unsafe(manifest.project) or "/" in manifest.project:
        problems.append(f"The project name '{manifest.project}' in the manifest is not a folder name.")
    names = set()
    for info in zf.infolist():
        if info.filename == MANIFEST_NAME or info.is_dir():
            continue
        names.add(info.filename)
        if _unsafe(info.filename) or not info.filename.startswith(prefix):
            problems.append(f"Member '{info.filename}' would be written outside the project folder.")
        elif info.filename not in manifest.files:
            problems.append(f"Member '{info.filename}' is not listed in the manifest.")
    for listed in manifest.files:
        if _unsafe(listed) or not listed.startswith(prefix):
            problems.append(f"Manifest entry '{listed}' would be written outside the project folder.")
        elif listed not in names:
            problems.append(f"File '{listed}' of the manifest is not in the package.")
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
    if manifest.stale_runs:
        warnings.append("The input of these runs changed after the run: " + ", ".join(manifest.stale_runs))
    return problems, warnings


def verify_package(zip_path: str | Path) -> VerifyReport:
    """Check a package without unpacking it: member names, every checksum, and the versions."""
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
    return None


def solver_config_activation_hint(project_path: Path, workspace_dir: Path) -> str:
    """How to make the shipped solver options of an imported project the ones a run uses."""
    imported = project_path / IMPORTED_SOLVER_CONFIG_DIR
    return (f"The solver options shipped with the project are in {imported}. FlexTool runs use the .opt files "
            f"of {workspace_dir}. To run with the shipped options, copy {IMPORTED_SOLVER_CONFIG_DIR}/*.opt to "
            f"{workspace_dir}, or set FLEXTOOL_SOLVER_CONFIG_DIR={imported} before starting FlexTool.")


def import_project(
    zip_path: str | Path,
    projects_dir: str | Path,
    name: str | None = None,
    *,
    solver_config_dir: str | Path | None = None,
) -> ImportReport:
    """Unpack the package *zip_path* as project *name* (default: the manifest's name) in *projects_dir*.

    Raises :class:`NameClashError` when the name is taken (case-insensitively) and :class:`PackageError` when a
    member name is unsafe, a checksum does not match or a database is newer than this FlexTool; nothing is written
    then.  *solver_config_dir* (default :func:`default_solver_config_dir`) is only compared with the shipped
    ``.opt`` files, never written.
    """
    projects_dir = Path(projects_dir)
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
                rel = arcname[len(prefix):]
                if rel.startswith(SOLVER_CONFIG_DIR + "/"):
                    rel = IMPORTED_SOLVER_CONFIG_DIR + rel[len(SOLVER_CONFIG_DIR):]
                dest = partial / rel
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
            if name_taken(projects_dir, target_name):
                raise NameClashError(target_name, suggest_name(projects_dir, target_name))
            partial.rename(final)
        except BaseException:
            shutil.rmtree(partial, ignore_errors=True)
            raise

    report = ImportReport(target_name, final, manifest, warnings)
    shipped = sorted((final / IMPORTED_SOLVER_CONFIG_DIR).glob("*.opt"))
    if shipped:
        workspace = Path(solver_config_dir) if solver_config_dir is not None else default_solver_config_dir()
        for path in shipped:
            mine = workspace / path.name
            if not mine.is_file():
                state = "not in workspace"
            elif _hash_file(mine) == _hash_file(path):
                state = "same"
            else:
                state = "different"
            report.solver_config.append((path.name, state))
        if any(state != "same" for _name, state in report.solver_config):
            report.solver_config_hint = solver_config_activation_hint(final, workspace)
    return report


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
            lines.append(f"Solver options {name}: different from this workspace's.")
        else:
            lines.append(f"Solver options {name}: this workspace has none.")
    if report.solver_config_hint:
        lines.append(report.solver_config_hint)
    return lines
