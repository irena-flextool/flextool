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

Tiers 3-5 count only a folder that holds at least one ``<solver>.opt``.  They skip ``<current folder>/solver_config``
when every ``.opt`` in it is an unedited FlexTool template (this or an earlier shipped version) and the current
folder is not a FlexTool workspace: that is the copy FlexTool 4.3 and older left in the folder a run started from,
not a choice of options.  The run then takes the next tier and warns, naming the skipped folder.  A FlexTool
workspace (the folder ``ensure_runtime_files`` seeds: it holds ``templates/project_folder.txt``, written in the
same step as ``solver_config/``, by 4.3 and 4.4 alike) is never skipped: its ``solver_config`` is the folder the
FlexTool window passes to every run, its unedited files are the seeded defaults (the same option values as the
packaged templates), and a later edit there is meant for runs from the workspace.  An explicit or environment
folder is taken as given; a solver whose file is missing there uses its packaged default.

:func:`snapshot_solver_option_files` reads the chosen files once at run start, writes the copy
``<work_folder>/solve_data/solver_config/`` as the record of what the run used, and returns the texts every
solve of the run reads from (an edit during the run does not reach later solves).

This module imports nothing from the engine, so the CLI, the GUI, project packaging and the calibrator can use it.
(It is not ``flextool/solver_config.py``: that name is the package-data folder of the templates.)
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

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

#: The file ``ensure_runtime_files`` seeds in a FlexTool workspace (with ``solver_config/``), relative to it.
_WORKSPACE_MARKER = Path("templates") / "project_folder.txt"

#: sha256 (of the text with CRLF read as LF) of every shipped version of ``flextool/solver_config/<solver>.opt.template``
#: up to 4.4.0, from the git history (``git log --all -- 'flextool*/solver_config/<solver>.opt.template'``, and the
#: older ``bin/highs.opt.template``).  A template edit adds its hash here: the test
#: ``test_shipped_template_hashes_are_pinned`` fails until the installed template's hash is listed (so the version
#: a later release replaces is always recognised) and checks the list against a copy pinned in the test.
#: A copy saved by an editor that adds a UTF-8 byte-order mark (or drops the final newline) does not match: the
#: folder is then used, which errs on the safe side (its values are the defaults' anyway).
SHIPPED_TEMPLATE_SHA256: Mapping[str, frozenset[str]] = {
    "highs": frozenset({
        "2151672a9aad61f534bdda16033e1848a678ab293896c5993a28dcf9c2956bed",
        "ae7ba9b2e0194773b2a02db935c8663724187744e010497d59153e37fefecea6",
        "cf28ea5eccb931bcdf41c98e0df62f280d3e2a6991f981b62b8221b8c8a5f3a1",
        "e52d4fb08ffba66357b164cb55ba97e9175f3674742b7ab5039f62f94f18cb03",
    }),
    "gurobi": frozenset({
        "04353fb0e11360d9bc4adeaeb30318f082e1379d8ba44b4f44f2a953c9a90dcb",
        "245ffff3fcf99e0d79eb03fd8a95b73c9433c59f92f08d08f5f27f8ab723cc6a",
        "7911b914b4c8f1dc4eb335be410251111c12d762dcecb18d4ac19cb2a912c690",
        "6f9b0366041a5d8310477f6275e6e9d2abac17e73c8bbdde6f0d9c70692c8c0e",
        "9d6548f8e598f688106e8dfe452fceb3dd11d73d15df4dffeb9e0ccdcee28c09",
    }),
    "cplex": frozenset({
        "4154578f6822e2dab71db1a797a9bb87e99ae6445124949b142603303e336999",
        "59629b91d5aadc35ea3f17a6ae62c5c67fc5641a1022d2a3381bf78550c8205e",
        "a90f70f4ea8c7daf201f0678584ff7b2d7225236972f43a0aa114e3a95d1ec33",
        "deea8744ce329573e540129d74cbf85cc86d7a09569d309a3badf16c2334d436",
    }),
    "xpress": frozenset({
        "049e374d011e7b592581c14e47cfe809941fd69f2e17ec438ce28a4b3c8ce228",
        "c12c35af12b782fa811c3ad8f2f71362c4c1776ec857f8d82b2cb2af5d2a3413",
        "da4d04664f020fc5b3b0501f3a7e63fdeb6a3e32bc4f6cfc270667de366827c2",
        "fa3929c9f1428595390d6673559a13c2c5dc33ccb1e4456950331a1e1de31a6e",
    }),
    "copt": frozenset({
        "38ce93bb6bde7fdcf707324df888533f0a375c9fce24bbf9e1554e28e3ce527f",
        "666de1c91d1f64a6bad0006c2bb4b3535d84589f010d8b27a744a9f0be3f3d16",
        "bc876c81c0b5fc89c26aef2fff370e37bb78d9e2edd2c0f50afa2f773a52e276",
        "d3f27615d136a9185ba2cbf097a60112d6418db27d89b1701e894818d0ad2bbc",
    }),
}


@dataclass(frozen=True)
class SolverConfigSource:
    """The folder a run takes its solver option files from (``None``: the packaged defaults) and why."""

    dir: Path | None
    reason: str
    tried: tuple[Path, ...] = ()
    #: Folders passed over because they hold only unedited FlexTool templates (see :func:`is_template_copy`).
    skipped: tuple[Path, ...] = ()

    def describe(self) -> str:
        """``<dir> (<reason>)`` for the run header and log; the folders tried when none was found."""
        skipped = (
            f"; skipped {', '.join(str(p) for p in self.skipped)}: unedited FlexTool templates" if self.skipped else ""
        )
        if self.dir is not None:
            return f"{self.dir} ({self.reason}{skipped})"
        if self.tried:
            return f"{REASON_DEFAULTS} (no solver_config folder in: {', '.join(str(p) for p in self.tried)}{skipped})"
        if skipped:
            return f"{REASON_DEFAULTS} ({skipped[2:]})"
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


def _normalised_sha256(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def _template_hashes(solver: str) -> frozenset[str]:
    """The shipped versions of *solver*'s template, plus the installed one."""
    known = SHIPPED_TEMPLATE_SHA256.get(solver, frozenset())
    current = _template_path(solver)
    if current is None:
        return known
    try:
        return known | {_normalised_sha256(current.read_bytes())}
    except OSError:
        return known


def is_unedited_template(solver: str, data: bytes) -> bool:
    """Whether *data* (a ``<solver>.opt``'s bytes) is byte for byte, CRLF read as LF, a shipped FlexTool template
    of *solver*: an earlier one or the installed one."""
    return _normalised_sha256(data) in _template_hashes(solver)


def is_template_copy(folder: Path) -> bool:
    """Whether *folder* holds at least one ``<solver>.opt`` and each one is byte for byte (CRLF read as LF) a
    shipped FlexTool template: the copy an earlier FlexTool left, not files someone edited."""
    found = False
    for solver in SOLVER_NAMES:
        path = folder / f"{solver}.opt"
        try:
            if not path.is_file():
                continue
            data = path.read_bytes()
        except OSError:
            return False
        if not is_unedited_template(solver, data):
            return False
        found = True
    return found


def is_flextool_workspace(folder: Path) -> bool:
    """Whether *folder* is a FlexTool workspace: it holds ``templates/project_folder.txt``, which
    ``ensure_runtime_files`` (FlexTool window start, ``flextool-update``) seeds together with ``solver_config/``."""
    try:
        return (folder / _WORKSPACE_MARKER).is_file()
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
    cwd_folder = Path(cwd if cwd is not None else Path.cwd()) / SOLVER_CONFIG_FOLDER
    candidates.append((cwd_folder, REASON_CWD))
    tried: list[Path] = []
    skipped: list[Path] = []
    seen: set[Path] = set()  # resolved, so a symlinked current folder is looked at once
    cwd_key = _folder_key(cwd_folder)
    for folder, reason in candidates:
        key = _folder_key(folder)
        if key in seen:
            continue
        seen.add(key)
        if _has_option_file(folder):
            if key == cwd_key and not is_flextool_workspace(cwd_folder.parent) and is_template_copy(folder):
                # FlexTool 4.3 and older copied the templates into the folder a run started from.
                skipped.append(folder)
                continue
            return SolverConfigSource(folder, reason, skipped=tuple(skipped))
        tried.append(folder)
    return SolverConfigSource(None, REASON_DEFAULTS, tuple(tried), tuple(skipped))


def _folder_key(folder: Path) -> Path:
    try:
        return folder.resolve()
    except OSError:
        return folder


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

    def record(self) -> dict[str, Any]:
        """The run record's ``solver_config`` block: the folder (``None``: FlexTool's defaults) and why, the
        folders skipped as unedited template copies, the snapshot folder, and per option file its source (the
        file, or ``"FlexTool default"``), the SHA-256 of the text the run used (CRLF read as LF, the hash of
        :data:`SHIPPED_TEMPLATE_SHA256`) and whether that is an unedited FlexTool template, and whether this
        computer allows solver parameter names that are not in FlexTool's lists from databases, and where that was
        set (:func:`flextool._local_solver_settings.allow_unlisted_record`).  Paths are absolute, so the record
        means the same read from elsewhere."""
        from flextool._local_solver_settings import allow_unlisted_record

        files: dict[str, dict[str, Any]] = {}
        for solver in SOLVER_NAMES:
            text = self.texts.get(solver)
            if text is None:
                continue
            digest = _normalised_sha256(text.encode("utf-8"))
            origin = self.origins.get(solver, "")
            files[f"{solver}.opt"] = {
                "source": "FlexTool default" if origin.startswith("FlexTool default") else _absolute(origin),
                "sha256": digest,
                "default": digest in _template_hashes(solver),
            }
        return {
            "dir": _absolute(self.source.dir) if self.source.dir is not None else None,
            "reason": self.source.reason,
            "skipped": [_absolute(p) for p in self.source.skipped],
            "snapshot_dir": _absolute(self.snapshot_dir) if self.snapshot_dir is not None else None,
            "files": files,
            "allow_unlisted_parameters": allow_unlisted_record(),
        }


def _absolute(path: str | os.PathLike) -> str:
    """*path* made absolute (as a text), or as given if that fails."""
    try:
        return str(Path(path).resolve())
    except (OSError, RuntimeError):
        return str(path)


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
    warning when no folder was found (or one per folder skipped as an unedited template copy, which also names the
    folder used), one line for the solvers that use the packaged default in a chosen
    folder, and one warning when a ``<work_folder>/solver_config/`` from an earlier version differs from what
    the run uses (it is no longer read).  A snapshot that cannot be written gives a warning; the run goes on with
    the texts read.
    """
    log = logger or _logger
    files = read_solver_option_files(source, names)
    log.info("Solver config: %s", source.describe())
    for folder in source.skipped:
        log.warning(
            "Not using %s: it holds only FlexTool's default option files (probably copied there by FlexTool 4.3 "
            "or older), so this run uses %s. To use your own option files, pass --solver-config-dir <the folder "
            "with your files> (or set FLEXTOOL_SOLVER_CONFIG_DIR). Delete %s to stop this warning.",
            folder,
            "FlexTool's default options" if source.dir is None else f"the files in {source.dir} ({source.reason})",
            folder,
        )
    if source.dir is None:
        if not source.skipped:  # else the skipped-folder warning above says the defaults are used
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
        if any(old.resolve() == p.resolve() for p in source.skipped):
            return  # already named by the skipped-folder warning
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
    "is_flextool_workspace",
    "is_template_copy",
    "read_solver_option_files",
    "resolve_solver_config_source",
    "snapshot_solver_option_files",
    "workspace_solver_config_dir",
]
