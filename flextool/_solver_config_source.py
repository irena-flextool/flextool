"""Where a run takes its solver option files from, and the run's snapshot of them.

A run reads ``highs.opt``, ``gurobi.opt``, ``cplex.opt``, ``xpress.opt`` and ``copt.opt`` from one
``solver_config`` folder.  :func:`resolve_solver_config_source` picks that folder, first that applies:

1. an explicit folder: ``run_chain_from_db(solver_config_dir=...)``, else ``--solver-config-dir``;
2. ``$FLEXTOOL_SOLVER_CONFIG_DIR`` (the flag's environment twin);
3. the input database's layout (a file database only, a Spine Toolbox ``?spinedbfilter=`` query stripped):
   ``solver_config/`` next to the database, then, for a database in a GUI project
   (``<workspace>/projects/<P>/input_sources/`` or ``.../intermediate/``), ``<workspace>/solver_config/``.  The
   check is bounded to these folders; it never climbs further up;
4. the Spine Toolbox project folder: ``--project-folder-file F`` gives ``F/../../solver_config``;
5. ``solver_config/`` in the current folder;
6. none: FlexTool's packaged defaults (``flextool/solver_config/<solver>.opt.template``), with one warning.

Tiers 3-5 count only a folder that holds at least one ``<solver>.opt``.  An explicit or environment folder is
taken as given; a solver whose file is missing there uses its packaged default.

:func:`snapshot_solver_option_files` reads the chosen files once at run start, writes the copy
``<work_folder>/solve_data/solver_config/`` as the record of what the run used, and returns the texts every
solve of the run reads from (an edit during the run does not reach later solves).

This module imports nothing from the engine, so the CLI, the GUI, project packaging and the calibrator can use it.
(It is not ``flextool/solver_config.py``: that name is the package-data folder of the templates.)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from flextool.run_options import RunOptions

_logger = logging.getLogger(__name__)

#: The solvers whose ``<solver>.opt`` FlexTool manages (also the seed list of ``ensure_runtime_files``).
SOLVER_NAMES: tuple[str, ...] = ("highs", "gurobi", "cplex", "xpress", "copt")

SOLVER_CONFIG_FOLDER = "solver_config"

#: The snapshot folder below the work folder.
SNAPSHOT_RELATIVE = Path("solve_data") / SOLVER_CONFIG_FOLDER

REASON_ARGUMENT = "solver_config_dir argument"
REASON_FLAG = "--solver-config-dir"
REASON_ENV = "FLEXTOOL_SOLVER_CONFIG_DIR"
REASON_DB_FOLDER = "next to the input database"
REASON_DB_WORKSPACE = "workspace of the input database"
REASON_TOOLBOX = "Spine Toolbox project folder"
REASON_CWD = "current folder"
REASON_DEFAULTS = "FlexTool defaults"

#: The GUI project folders an input database can sit in: ``<workspace>/projects/<P>/<folder>/<db>``.
_PROJECT_DB_FOLDERS = ("input_sources", "intermediate")


@dataclass(frozen=True)
class SolverConfigSource:
    """The folder a run takes its solver option files from (``None``: the packaged defaults) and why."""

    dir: Path | None
    reason: str
    tried: tuple[Path, ...] = ()

    def describe(self) -> str:
        """``<dir> (<reason>)`` for the run header and log; the folders tried when none was found."""
        if self.dir is not None:
            return f"{self.dir} ({self.reason})"
        if self.tried:
            return f"{REASON_DEFAULTS} (no solver_config folder in: {', '.join(str(p) for p in self.tried)})"
        return REASON_DEFAULTS


def input_db_filesystem_path(input_db_url: str | os.PathLike | None) -> Path | None:
    """The file of a SQLite input database URL or bare path, else ``None`` (a server database, empty value).

    Strips the ``sqlite:///`` scheme and an appended Spine filter query (``?spinedbfilter=...``, which Spine
    Toolbox adds).  Pure string -> path; does not touch the filesystem.
    """
    if not input_db_url:
        return None
    text = str(input_db_url)
    if "://" in text and not text.startswith("sqlite:"):
        return None
    no_scheme = text.replace("sqlite:///", "", 1)
    no_query = no_scheme.split("?", 1)[0]
    if not no_query:
        return None
    return Path(no_query)


def _has_option_file(folder: Path) -> bool:
    try:
        return any((folder / f"{name}.opt").is_file() for name in SOLVER_NAMES)
    except OSError:
        return False


def _database_candidates(input_db_url: str | os.PathLike | None) -> list[tuple[Path, str]]:
    """The bounded database-layout folders of tier 3, in order."""
    db_path = input_db_filesystem_path(input_db_url)
    if db_path is None:
        return []
    try:
        db_path = db_path.resolve()
    except OSError:
        pass
    db_folder = db_path.parent
    out = [(db_folder / SOLVER_CONFIG_FOLDER, REASON_DB_FOLDER)]
    project = db_folder.parent
    if db_folder.name in _PROJECT_DB_FOLDERS and project.parent.name == "projects":
        out.append((project.parent.parent / SOLVER_CONFIG_FOLDER, REASON_DB_WORKSPACE))
    return out


def resolve_solver_config_source(
    *,
    explicit: str | os.PathLike | None = None,
    run_options: "RunOptions | None" = None,
    environ: Mapping[str, str] | None = None,
    input_db_url: str | os.PathLike | None = None,
    project_folder_file: str | os.PathLike | None = None,
    cwd: str | os.PathLike | None = None,
) -> SolverConfigSource:
    """Pick the run's ``solver_config`` folder (see the module docstring for the tiers).

    *run_options* carries ``--solver-config-dir`` and its environment twin; ``None`` resolves them from
    *environ* (default ``os.environ``).  *cwd* defaults to the current folder.
    """
    if explicit is not None and str(explicit).strip():
        return SolverConfigSource(Path(explicit), REASON_ARGUMENT)
    if run_options is None:
        from flextool.run_options import RunOptions

        run_options = RunOptions.from_env(environ)
    chosen = run_options.resolved.get("solver_config_dir")
    if chosen is not None and chosen.value and chosen.source in ("cli", "env"):
        return SolverConfigSource(Path(chosen.value), REASON_FLAG if chosen.source == "cli" else REASON_ENV)

    candidates = _database_candidates(input_db_url)
    if project_folder_file:
        try:
            anchor = Path(project_folder_file).resolve().parent.parent
        except OSError:
            anchor = Path(project_folder_file).parent.parent
        candidates.append((anchor / SOLVER_CONFIG_FOLDER, REASON_TOOLBOX))
    candidates.append((Path(cwd if cwd is not None else Path.cwd()) / SOLVER_CONFIG_FOLDER, REASON_CWD))
    tried: list[Path] = []
    for folder, reason in candidates:
        if folder in tried:
            continue
        if _has_option_file(folder):
            return SolverConfigSource(folder, reason)
        tried.append(folder)
    return SolverConfigSource(None, REASON_DEFAULTS, tuple(tried))


def workspace_solver_config_dir(environ: Mapping[str, str] | None = None, cwd: str | os.PathLike | None = None) -> Path:
    """The workspace's solver option folder, for launchers that know the workspace (GUI, project packaging).

    ``$FLEXTOOL_SOLVER_CONFIG_DIR`` when set, else ``<cwd>/solver_config`` (the GUI's current folder is the
    workspace).  The GUI passes it to every run as ``--solver-config-dir``, so its runs never depend on where
    the input database is.
    """
    env = (os.environ if environ is None else environ).get("FLEXTOOL_SOLVER_CONFIG_DIR", "").strip()
    if env:
        return Path(env)
    return Path(cwd if cwd is not None else Path.cwd()) / SOLVER_CONFIG_FOLDER


def _template_path(solver: str) -> Path | None:
    from flextool._resources import package_data_path

    try:
        path = Path(package_data_path(f"{SOLVER_CONFIG_FOLDER}/{solver}.opt.template"))
    except Exception:  # noqa: BLE001 - a broken install must not stop the run
        return None
    return path if path.is_file() else None


@dataclass(frozen=True)
class SolverOptionFiles:
    """The option files of one run, read once at its start.

    *texts* maps a solver to its file's text; *origins* to where it came from (the file, or the packaged
    template), used in messages.  *snapshot_dir* is the record written under the work folder (``None`` when it
    could not be written).
    """

    source: SolverConfigSource
    snapshot_dir: Path | None = None
    texts: Mapping[str, str] = field(default_factory=dict)
    origins: Mapping[str, str] = field(default_factory=dict)

    def text(self, solver: str) -> str | None:
        return self.texts.get(solver)

    def origin(self, solver: str) -> str:
        return self.origins.get(solver, f"{solver}.opt")


def read_solver_option_files(source: SolverConfigSource, names: tuple[str, ...] = SOLVER_NAMES) -> SolverOptionFiles:
    """Read the option file of each solver in *names* from *source* (else its packaged template), once."""
    texts: dict[str, str] = {}
    origins: dict[str, str] = {}
    for solver in names:
        candidates: list[tuple[Path, str]] = []
        if source.dir is not None:
            own = source.dir / f"{solver}.opt"
            candidates.append((own, str(own)))
        template = _template_path(solver)
        if template is not None:
            candidates.append((template, f"FlexTool default ({solver}.opt.template)"))
        for path, origin in candidates:
            try:
                texts[solver] = path.read_text(encoding="utf-8")
            except OSError:
                continue
            origins[solver] = origin
            break
    return SolverOptionFiles(source=source, texts=texts, origins=origins)


def snapshot_solver_option_files(
    source: SolverConfigSource,
    work_folder: str | os.PathLike,
    logger: logging.Logger | None = None,
    names: tuple[str, ...] = SOLVER_NAMES,
) -> SolverOptionFiles:
    """Read the run's option files once and write them to ``<work_folder>/solve_data/solver_config/``.

    Every run writes the snapshot afresh (an older one is replaced).  Logs the source (``Solver config:``), one
    warning when no folder was found, one line for the solvers that use the packaged default in a chosen
    folder, and one warning when a ``<work_folder>/solver_config/`` from an earlier version differs from what
    the run uses (it is no longer read).  A snapshot that cannot be written gives a warning; the run goes on with
    the texts read.
    """
    log = logger or _logger
    files = read_solver_option_files(source, names)
    log.info("Solver config: %s", source.describe())
    if source.dir is None:
        log.warning(
            "No solver_config folder found (tried %s); every solver uses FlexTool's default options. Use "
            "--solver-config-dir (or FLEXTOOL_SOLVER_CONFIG_DIR) to choose the folder.",
            ", ".join(str(p) for p in source.tried) or "none",
        )
    elif not source.dir.is_dir():
        log.warning(
            "Solver config folder %s (%s) does not exist; every solver uses FlexTool's default options.",
            source.dir, source.reason,
        )
    else:
        defaults = [
            f"{s}.opt" for s in names if s in files.origins and files.origins[s] != str(source.dir / f"{s}.opt")
        ]
        if defaults:
            log.info("Solver config: FlexTool defaults for %s (not in %s)", ", ".join(defaults), source.dir)

    work = Path(work_folder)
    _warn_stale_work_folder_copy(work, source, files, log)
    snapshot = work / SNAPSHOT_RELATIVE
    try:
        snapshot.mkdir(parents=True, exist_ok=True)
        for solver in names:
            target = snapshot / f"{solver}.opt"
            if solver in files.texts:
                target.write_text(files.texts[solver], encoding="utf-8")
            elif target.exists():
                target.unlink()
    except OSError as exc:
        log.warning("Cannot write the solver option snapshot %s (%s); the run continues.", snapshot, exc)
        return SolverOptionFiles(source=source, snapshot_dir=None, texts=files.texts, origins=files.origins)
    return SolverOptionFiles(source=source, snapshot_dir=snapshot, texts=files.texts, origins=files.origins)


def _warn_stale_work_folder_copy(
    work_folder: Path, source: SolverConfigSource, files: SolverOptionFiles, log: logging.Logger,
) -> None:
    """One warning when ``<work_folder>/solver_config/`` (the copy of FlexTool 4.3 and older) differs from the
    files the run uses.  Not when it is the source itself (a run from the workspace)."""
    old = work_folder / SOLVER_CONFIG_FOLDER
    if not old.is_dir():
        return
    try:
        if source.dir is not None and old.resolve() == source.dir.resolve():
            return
    except OSError:
        return
    differing = []
    for solver in SOLVER_NAMES:
        path = old / f"{solver}.opt"
        if not path.is_file():
            continue
        try:
            if path.read_text(encoding="utf-8") != files.texts.get(solver):
                differing.append(path.name)
        except OSError:
            continue
    if differing:
        log.warning(
            "Ignoring %s (%s): runs no longer read solver option files from the work folder; this run uses %s. "
            "To use that folder, pass --solver-config-dir %s (or set FLEXTOOL_SOLVER_CONFIG_DIR).",
            old, ", ".join(differing), source.describe(), old,
        )


__all__ = [
    "SNAPSHOT_RELATIVE",
    "SOLVER_NAMES",
    "SolverConfigSource",
    "SolverOptionFiles",
    "input_db_filesystem_path",
    "read_solver_option_files",
    "resolve_solver_config_source",
    "snapshot_solver_option_files",
    "workspace_solver_config_dir",
]
