"""Where a run takes its solver option files from, and the run's snapshot of them.

A run reads ``highs.opt``, ``gurobi.opt``, ``cplex.opt``, ``xpress.opt`` and ``copt.opt`` FILE BY FILE from an
ordered **ladder** of ``solver_config`` folders: each solver's file comes from the first folder of the ladder that
has ``<solver>.opt``; a solver that no folder has uses FlexTool's packaged default
(``flextool/solver_config/<solver>.opt.template``).  :func:`resolve_solver_config_source` builds the ladder from
the first tier that applies:

1. folders given explicitly, and only these (a closed ladder: nothing from the database's tree or the current
   folder is added): ``run_chain_from_db(solver_config_dir=...)`` (one folder, or a list), else the folders a
   launcher found, ``--workspace-solver-config-dir`` (repeatable, highest first), above the folders the user named,
   ``--solver-config-dir`` (repeat the flag for several folders, highest first), else
   ``$FLEXTOOL_SOLVER_CONFIG_DIR`` (several folders separated by :data:`os.pathsep`: ``:`` on Linux and macOS,
   ``;`` on Windows).  The FlexTool window passes ``<project>/solver_config`` (when the project has that folder)
   and the workspace's ``solver_config`` as found folders (or, when ``$FLEXTOOL_SOLVER_CONFIG_DIR`` is set, the
   project's as found and the variable's folders as named), so a GUI run takes each file from the project, else
   the workspace, else FlexTool's default;
2. the input database's layout (a file database only, a Spine Toolbox ``?spinedbfilter=`` query stripped), bounded
   to these folders (it never climbs further up):

   a. ``solver_config/`` next to the database, alone;
   b. for a database in a GUI project (``<workspace>/projects/<P>/input_sources/`` or ``.../intermediate/``): the
      ladder ``<workspace>/projects/<P>/solver_config``, ``<workspace>/solver_config`` (the same answer as the
      FlexTool window's run of the project);

3. the Spine Toolbox project folder: ``--project-folder-file F`` gives ``F/../../solver_config``;
4. ``solver_config/`` in the current folder;
5. none: FlexTool's packaged defaults for every solver, with one warning.

Tiers 2-4 count only when one of their folders holds a ``<solver>.opt`` it would give.  A GUI project's folder
(``<workspace>/projects/<P>/solver_config``, in any tier) never gives an unedited FlexTool template (this or an
earlier shipped version, :func:`is_unedited_template`): such a file holds the options FlexTool shipped then, not a
choice made for the project, and taking it would hide the workspace's file of that solver (4.4 development
builds shipped every file of the workspace, templates too, in a project package).  The tiers skip ``<current folder>/solver_config`` when every ``.opt`` in it is an unedited template and
the current folder is not a FlexTool workspace: that is the copy FlexTool 4.3 and older left in the folder a run
started from, not a choice of options.  The run then takes the next tier and warns, naming the skipped folder.  A
FlexTool workspace (the folder ``ensure_runtime_files`` seeds: it holds ``templates/project_folder.txt``, written in
the same step as ``solver_config/``, by 4.3 and 4.4 alike) is never skipped: its ``solver_config`` is the folder the
FlexTool window passes to every run, its unedited files are the seeded defaults (the same option values as the
packaged templates), and a later edit there is meant for runs from the workspace.

:func:`snapshot_solver_option_files` reads the chosen files once at run start, writes the copy
``<work_folder>/solve_data/solver_config/`` as the record of what the run used, and returns the texts every
solve of the run reads from (an edit during the run does not reach later solves).

**Trust** (item 240 B3), judged per folder of the ladder, so file by file.  The files of a folder the user named
(``solver_config_dir=``, ``--solver-config-dir``, ``$FLEXTOOL_SOLVER_CONFIG_DIR``) and the packaged defaults are
trusted: file paths and server settings in them reach the solver.  A folder found on its own (by a launcher with
``--workspace-solver-config-dir``, or in tiers 2-4) is trusted only when the folder it is in (a workspace, or a GUI
project for ``<workspace>/projects/<P>/solver_config``) is one this computer set up (``trusted_workspaces`` in
:mod:`flextool._local_solver_settings`).  A file that is byte for byte a shipped FlexTool template is FlexTool's
own wherever it lies.  Every other file's lines get the input database's rules (``check_database_solver_option``),
with :meth:`SolverOptionFiles.untrusted_note` saying why and how to trust its folder; so a trusted workspace's file
that a project's untrusted file overrides does not lend the project's file its trust.
``templates/project_folder.txt`` is no trust anchor: it is plain content and can come with a received workspace.

This module imports nothing from the engine, so the CLI, the GUI, project packaging and the calibrator can use it.
(It is not ``flextool/solver_config.py``: that name is the package-data folder of the templates.)
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

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
REASON_WORKSPACE = "workspace of the FlexTool window"
REASON_PROJECT = "project of the FlexTool window"
REASON_DB_FOLDER = "next to the input database"
REASON_DB_PROJECT = "project of the input database"
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
        "8a6870c450976813fbef15099368709edfba7eb30e7c8da29a961bca9d57ae5e",
        "9d6548f8e598f688106e8dfe452fceb3dd11d73d15df4dffeb9e0ccdcee28c09",
    }),
    "cplex": frozenset({
        "4154578f6822e2dab71db1a797a9bb87e99ae6445124949b142603303e336999",
        "59629b91d5aadc35ea3f17a6ae62c5c67fc5641a1022d2a3381bf78550c8205e",
        "a90f70f4ea8c7daf201f0678584ff7b2d7225236972f43a0aa114e3a95d1ec33",
        "deea8744ce329573e540129d74cbf85cc86d7a09569d309a3badf16c2334d436",
        "fc228498eaca1e262817f531736c261d5f93cf6d076a74ace0059c4df805de47",
    }),
    "xpress": frozenset({
        "049e374d011e7b592581c14e47cfe809941fd69f2e17ec438ce28a4b3c8ce228",
        "c12c35af12b782fa811c3ad8f2f71362c4c1776ec857f8d82b2cb2af5d2a3413",
        "da4d04664f020fc5b3b0501f3a7e63fdeb6a3e32bc4f6cfc270667de366827c2",
        "f2eb353b652e6ad66ba38cca065a8427cb71f11fde4d03ef19cfa5c895776c7d",
        "fa3929c9f1428595390d6673559a13c2c5dc33ccb1e4456950331a1e1de31a6e",
    }),
    "copt": frozenset({
        "38ce93bb6bde7fdcf707324df888533f0a375c9fce24bbf9e1554e28e3ce527f",
        "666de1c91d1f64a6bad0006c2bb4b3535d84589f010d8b27a744a9f0be3f3d16",
        "bc876c81c0b5fc89c26aef2fff370e37bb78d9e2edd2c0f50afa2f773a52e276",
        "4527f2fe84191f8e7ae54b55c987521c43e3653214de549de2b25c0b62496cb5",
        "d3f27615d136a9185ba2cbf097a60112d6418db27d89b1701e894818d0ad2bbc",
    }),
}


@dataclass(frozen=True)
class SolverConfigSource:
    """The ladder of folders a run takes its solver option files from, and why.

    *dir* / *reason* are the top folder (``None``: FlexTool's packaged defaults) and why it was chosen; *lower*
    the rest of the ladder, highest first, each with its reason.  A solver's file comes from the first folder of
    :attr:`layers` that has it (:meth:`file_for`), else the packaged default.  *tried* are the folders looked at
    when none was found; *skipped* the folders passed over because they hold only unedited FlexTool templates
    (see :func:`is_template_copy`).  *trusted* / *trust_note* say whether the top folder's files may name paths
    and servers (item 240 B3; ``False``: the input database's rules) and why not; *lower_notes* the same for each
    folder of *lower* (``None``: trusted).
    """

    dir: Path | None
    reason: str
    tried: tuple[Path, ...] = ()
    skipped: tuple[Path, ...] = ()
    lower: tuple[tuple[Path, str], ...] = ()
    #: Whether the top folder's files may name paths and servers (item 240 B3); ``False``: the input database's
    #: rules.
    trusted: bool = True
    #: Why the top folder is not trusted and how to trust it (``None`` when trusted), for the messages.
    trust_note: str | None = None
    #: Per folder of *lower*, why it is not trusted (``None``: trusted); missing entries are trusted.
    lower_notes: tuple[str | None, ...] = ()
    #: Per folder of :attr:`layers`, whether its trust comes from this computer's list (``True``: each of its files
    #: is checked against the list's entry, review b2b3 M1) rather than from the user naming it; missing: named.
    listed: tuple[bool, ...] = ()

    def layer_listed(self, index: int) -> bool:
        """Whether the folder at *index* of :attr:`layers` is trusted through this computer's list."""
        return index < len(self.listed) and self.listed[index]

    def layer_note(self, index: int) -> str | None:
        """Why the folder at *index* of :attr:`layers` is not trusted (``None``: trusted)."""
        if index == 0:
            return None if self.trusted else (self.trust_note or untrusted_note(self.dir or Path("."), self.reason))
        notes = self.lower_notes
        return notes[index - 1] if index - 1 < len(notes) else None

    @property
    def layers(self) -> tuple[tuple[Path, str], ...]:
        """``(folder, reason)`` of every folder of the ladder, highest first (empty: the packaged defaults)."""
        if self.dir is None:
            return ()
        return ((self.dir, self.reason), *self.lower)

    def file_for(self, solver: str) -> tuple[Path, str] | None:
        """``(file, reason)`` of the ``<solver>.opt`` this ladder gives *solver* now, ``None``: the packaged
        default."""
        index, _passed = _ladder_file(self.layers, solver)
        if index is None:
            return None
        folder, reason = self.layers[index]
        return folder / f"{solver}.opt", reason

    def describe(self) -> str:
        """The ladder for the run header and log.

        One folder: ``<dir> (<reason>)``.  Several folders (or a GUI project's folder that holds an unedited
        template): per folder the files it gives, in ladder order, then the solvers that use FlexTool's defaults:
        ``<P>/solver_config (--solver-config-dir): gurobi.opt; <ws>/solver_config (--solver-config-dir): highs.opt;
        FlexTool defaults: cplex.opt, xpress.opt, copt.opt``.  A folder that gives no file is left out; one that
        does not exist is listed as ``<dir> (<reason>): does not exist``.  The folders tried when none was found.
        """
        skipped = (
            f"; skipped {', '.join(str(p) for p in self.skipped)}: unedited FlexTool templates" if self.skipped else ""
        )
        if self.dir is None:
            if self.tried:
                return (f"{REASON_DEFAULTS} (no solver_config folder in: {', '.join(str(p) for p in self.tried)}"
                        f"{skipped})")
            if skipped:
                return f"{REASON_DEFAULTS} ({skipped[2:]})"
            return REASON_DEFAULTS
        given: dict[str, int | None] = {}
        skipped_files: list[Path] = []
        for solver in SOLVER_NAMES:
            given[solver], passed = _ladder_file(self.layers, solver)
            skipped_files.extend(passed)
        if not self.lower and not skipped_files:
            untrusted = "" if self.trusted else f"; {_untrusted_short(self.dir)}"
            return f"{self.dir} ({self.reason}{untrusted}{skipped})"
        parts: list[str] = []
        for index, (folder, reason) in enumerate(self.layers):
            names = [f"{s}.opt" for s in SOLVER_NAMES if given[s] == index]
            untrusted = "" if self.layer_note(index) is None else f"; {_untrusted_short(folder)}"
            if names:
                parts.append(f"{folder} ({reason}{untrusted}): {', '.join(names)}")
            elif not _is_dir(folder):
                parts.append(f"{folder} ({reason}): does not exist")
        defaults = [f"{s}.opt" for s in SOLVER_NAMES if given[s] is None]
        if defaults:
            parts.append(f"{REASON_DEFAULTS}: {', '.join(defaults)}")
        if skipped_files:
            parts.append(f"skipped {', '.join(str(p) for p in skipped_files)}: unedited FlexTool templates")
        return "; ".join(parts) + skipped


def _is_dir(folder: Path) -> bool:
    try:
        return folder.is_dir()
    except OSError:
        return False


#: How this system compares file names: case-folded on Windows (``Projects\P`` is ``projects\P`` there), exact
#: elsewhere.  A module name so that a test can stand in for Windows.
_normcase = os.path.normcase


def _same_name(name: str, expected: str) -> bool:
    """Whether the folder name *name* is *expected* on this system (case-insensitively on Windows)."""
    return _normcase(name) == _normcase(expected)


def is_project_solver_config(folder: Path) -> bool:
    """Whether *folder* is a GUI project's ``solver_config`` (``<workspace>/projects/<P>/solver_config``)."""
    return _same_name(folder.name, SOLVER_CONFIG_FOLDER) and _same_name(folder.parent.parent.name, "projects")


def _layer_gives(folder: Path, solver: str) -> tuple[bool, bool]:
    """``(gives, skipped)``: whether *folder* gives *solver*'s file (it has ``<solver>.opt``), and whether it has
    one but does not give it (a GUI project's folder, the file an unedited FlexTool template).  Only a project's
    file is read here."""
    path = folder / f"{solver}.opt"
    try:
        if not path.is_file():
            return False, False
        if not is_project_solver_config(folder):
            return True, False
        data = path.read_bytes()
    except OSError:
        return False, False
    if is_unedited_template(solver, data):
        return False, True
    return True, False


def _ladder_file(layers: tuple[tuple[Path, str], ...], solver: str) -> tuple[int | None, list[Path]]:
    """``(index of the layer that gives *solver*'s file or None, files passed over as unedited templates)``."""
    passed: list[Path] = []
    for index, (folder, _reason) in enumerate(layers):
        gives, skipped = _layer_gives(folder, solver)
        if skipped:
            passed.append(folder / f"{solver}.opt")
        if gives:
            return index, passed
    return None, passed


def split_solver_config_dirs(value: str | os.PathLike | Sequence[str | os.PathLike] | None) -> list[Path]:
    """The folders of a ``solver_config_dir`` value, highest first.

    A text or path is ONE folder (also when it holds :data:`os.pathsep`: a folder name may); a sequence (a
    repeated ``--solver-config-dir``, the split ``$FLEXTOOL_SOLVER_CONFIG_DIR``, a Python list) is the folders in
    order.  Empty entries are dropped; a folder given twice (the same after resolving) is kept once, at its first
    place.
    """
    if value is None:
        return []
    items = [value] if isinstance(value, (str, os.PathLike)) else list(value)
    out: list[Path] = []
    seen: set[Path] = set()
    for item in items:
        if item is None or not str(item).strip():
            continue
        folder = Path(str(item).strip()) if isinstance(item, str) else Path(item)
        key = _folder_key(folder)
        if key in seen:
            continue
        seen.add(key)
        out.append(folder)
    return out


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


def holds_only_flextool_defaults(folder: Path) -> bool:
    """Whether *folder* (a workspace's or a project's ``solver_config``) is missing, or holds no option file but
    unedited FlexTool templates (this or an earlier shipped version): the state in which FlexTool lists the folder
    it is in as set up by this computer (item 240 B3).  A Windows junction or other reparse point counts as a
    symbolic link (review b2b3 L5).

    Every file whose name ends in ``.opt`` (any case) counts; other files are ignored, since no run reads them.  A
    symbolic link (the folder or a file in it) is the user's own arrangement, whose target may change later: it is
    never taken as FlexTool's own."""
    from flextool._local_solver_settings import is_link_like

    try:
        if is_link_like(folder):
            return False
        if not folder.exists():
            return True
        if not folder.is_dir():
            return False
        for entry in folder.iterdir():
            if not entry.name.casefold().endswith(".opt"):
                continue
            solver = entry.name[:-4]
            if solver not in SOLVER_NAMES or is_link_like(entry) or not entry.is_file():
                return False
            if not is_unedited_template(solver, entry.read_bytes()):
                return False
    except OSError:
        return False
    return True


#: How :meth:`SolverConfigSource.describe` marks a folder that is not trusted (a GUI project's: :data:`UNTRUSTED_SHORT_PROJECT`).
UNTRUSTED_SHORT = "not a workspace this computer set up: its files get the input database's rules"
UNTRUSTED_SHORT_PROJECT = "not a project this computer set up: its files get the input database's rules"
#: The folder in which FlexTool 4.4 development builds unpacked a package's solver options (never read by a run).
IMPORTED_SOLVER_CONFIG_FOLDER = "imported_solver_config"
#: The suffix of a shipped option file that project import or the first open holds back (never read by a run).
HELD_SUFFIX = ".review"


def _untrusted_short(folder: Path) -> str:
    return UNTRUSTED_SHORT_PROJECT if is_project_solver_config(folder) else UNTRUSTED_SHORT


#: The terminal command that trusts a folder's option files (review b2b3 L6).
TRUST_COMMAND = "flextool-trust-workspace"


def untrusted_note(folder: Path, reason: str, problem: str | None = None, *, changed: str | None = None) -> str:
    """The message part that says why the option files of *folder* (found as *reason*) get the input database's
    rules, and how to trust them (item 240 B3).  A GUI project's folder (``<workspace>/projects/<P>/solver_config``)
    is trusted through its project, any other through the workspace (the folder it is in).  *changed*: the folder
    is trusted, but this file changed since in a way FlexTool cannot take as the user's own edit (review b2b3 M1)."""
    owner = "project" if is_project_solver_config(folder) else "workspace"
    rules = (": a file option takes only a plain file name, placed in the solve's work folder; no server or licence "
             "settings; only parameter names in FlexTool's list.")
    if changed is not None:
        head = (f"An option file in {folder} ({reason}) gets the input database's rules, because {changed}{rules} "
                "If you made the change, or have read the file and trust it, trust the folder again: ")
    else:
        damaged = f" ({problem})" if problem else ""
        head = (f"The option files in {folder} ({reason}) get the input database's rules, because FlexTool found "
                f"that folder on its own and it is not in a {owner} this computer set up{damaged}{rules} If you "
                "made these files, or have read them and trust them, ")
    command = f"run {TRUST_COMMAND} {folder.parent} in a terminal"
    if owner == "project":
        return head + (
            f"tick \"Trust this project's solver option files\" in the FlexTool window (Solver options, On this "
            f"computer, with the project open), {command}, or pass the folder with --solver-config-dir (or set "
            "FLEXTOOL_SOLVER_CONFIG_DIR). The FlexTool window marks a project as set up by this computer only when "
            "it creates the project, or opens it while its solver_config/ is missing or holds nothing but "
            "FlexTool's own default files and no received option file waits for review, so option files that "
            "came with a project (a project package, a copied folder) are never trusted on their own. The trust "
            "is for that folder: a folder put at its path later is not trusted, and a later change to a file is "
            "trusted only when your account made it and no other account can change the file."
        )
    return head + (
        f"pass the folder with --solver-config-dir (or set FLEXTOOL_SOLVER_CONFIG_DIR), {command}, or tick "
        "\"Trust this workspace's solver option files\" in the FlexTool window (Solver options, On this computer). "
        "flextool-update and the FlexTool window mark a workspace as set up by this computer only when they "
        "create its solver_config/ or find nothing in it but FlexTool's own default files, so a workspace "
        "received with edited option files is never marked on its own. The trust is for that folder: a folder "
        "put at its path later is not trusted, and a later change to a file is trusted only when your account "
        "made it and no other account can change the file."
    )


def awaits_review(project: Path) -> bool:
    """Whether the GUI project *project* holds received solver option files that are not in use yet: an
    ``imported_solver_config/`` (a 4.4 development build's import) or a held ``<solver>.opt.review``."""
    try:
        if (project / IMPORTED_SOLVER_CONFIG_FOLDER).exists():
            return True
        own = project / SOLVER_CONFIG_FOLDER
        return own.is_dir() and any(entry.name.casefold().endswith(".opt" + HELD_SUFFIX) for entry in own.iterdir())
    except OSError:
        return True


def note_project_set_up_here(project: str | os.PathLike) -> bool:
    """List the GUI project *project* as set up by this computer (``trusted_workspaces``, item 240 B3) when its
    ``solver_config/`` is missing or holds nothing but FlexTool's own default files
    (:func:`holds_only_flextool_defaults`) and no received option file waits for review (:func:`awaits_review`):
    then any option file that appears there later was put there on this computer.  The FlexTool window calls it
    when it creates or opens a project.  Returns whether the project was added; a settings file that cannot be
    read or written adds nothing (logged)."""
    from flextool._local_solver_settings import LocalSolverSettingsError, add_trusted_workspace

    folder = Path(project)
    if not folder.is_dir() or awaits_review(folder):
        return False
    if not holds_only_flextool_defaults(folder / SOLVER_CONFIG_FOLDER):
        return False
    try:
        return add_trusted_workspace(folder)
    except LocalSolverSettingsError as exc:
        _logger.warning("Could not note project %s as set up by this computer: %s", folder, exc)
        return False


def forget_trusted_folder(folder: str | os.PathLike) -> None:
    """Take *folder* (a project or workspace) off this computer's trusted list, when it is on it: before received
    files arrive at that path (a project import, the first open's rename of ``imported_solver_config/``) or after
    the folder is deleted, so a later folder at the same path does not inherit the trust.  A settings file that
    cannot be read or written is logged."""
    from flextool._local_solver_settings import LocalSolverSettingsError, remove_trusted_workspace, trust_status

    try:
        if trust_status(folder).entry is not None:  # listed, also when the folder there is not the one trusted
            remove_trusted_workspace(folder)
    except LocalSolverSettingsError as exc:
        _logger.warning("Could not take %s off this computer's trusted folders: %s", folder, exc)


def _folder_trust(folder: Path, reason: str) -> tuple[bool, str | None]:
    """``(trusted, note)`` of a folder found on its own: trusted when the folder it is in (a workspace, or a GUI
    project for a project's ``solver_config``) is listed."""
    from flextool._local_solver_settings import LocalSolverSettingsError, trust_status

    try:
        status = trust_status(folder.parent)
    except LocalSolverSettingsError as exc:
        return False, untrusted_note(folder, reason, str(exc))
    if status.trusted:
        return True, None
    return False, untrusted_note(folder, reason, status.problem)


def _found_ladder(
    layers: Sequence[tuple[Path, str, bool]], *, tried: tuple[Path, ...] = (), skipped: tuple[Path, ...] = (),
) -> "SolverConfigSource":
    """The ladder of *layers* ``(folder, reason, named)``, highest first: a folder the user named is trusted, one
    found on its own is judged by :func:`_folder_trust`."""
    # Absolute, so that what a folder is never depends on the current folder or on the text of its name (review
    # b2b3 M3).
    layers = [(Path(os.path.abspath(folder)), reason, named) for folder, reason, named in layers]
    notes: list[str | None] = []
    for folder, reason, named in layers:
        notes.append(None if named else _folder_trust(folder, reason)[1])
    (top, top_reason, _named), *rest = layers
    return SolverConfigSource(
        top, top_reason, tried, skipped, lower=tuple((folder, reason) for folder, reason, _n in rest),
        trusted=notes[0] is None, trust_note=notes[0], lower_notes=tuple(notes[1:]),
        listed=tuple(not named for _folder, _reason, named in layers),
    )


def is_flextool_workspace(folder: Path) -> bool:
    """Whether *folder* is a FlexTool workspace: it holds ``templates/project_folder.txt``, which
    ``ensure_runtime_files`` (FlexTool window start, ``flextool-update``) seeds together with ``solver_config/``."""
    try:
        return (folder / _WORKSPACE_MARKER).is_file()
    except OSError:
        return False


def _database_candidates(input_db_url: str | os.PathLike | None) -> list[list[tuple[Path, str]]]:
    """The bounded database-layout ladders of tier 2, in order: the folder next to the database, then (a GUI
    project's database) the project's and the workspace's folder."""
    db_path = input_db_filesystem_path(input_db_url)
    if db_path is None:
        return []
    try:
        db_path = db_path.resolve()
    except OSError:
        pass
    db_folder = db_path.parent
    out = [[(db_folder / SOLVER_CONFIG_FOLDER, REASON_DB_FOLDER)]]
    project = db_folder.parent
    if any(_same_name(db_folder.name, n) for n in _PROJECT_DB_FOLDERS) and _same_name(project.parent.name, "projects"):
        out.append([
            (project / SOLVER_CONFIG_FOLDER, REASON_DB_PROJECT),
            (project.parent.parent / SOLVER_CONFIG_FOLDER, REASON_DB_WORKSPACE),
        ])
    return out


def _closed_ladder(folders: list[Path], reason: str) -> SolverConfigSource:
    return SolverConfigSource(folders[0], reason, lower=tuple((folder, reason) for folder in folders[1:]))


def resolve_solver_config_source(
    *,
    explicit: str | os.PathLike | Sequence[str | os.PathLike] | None = None,
    run_options: "RunOptions | None" = None,
    environ: Mapping[str, str] | None = None,
    input_db_url: str | os.PathLike | None = None,
    project_folder_file: str | os.PathLike | None = None,
    cwd: str | os.PathLike | None = None,
) -> SolverConfigSource:
    """Build the run's ladder of ``solver_config`` folders (see the module docstring for the tiers).

    *explicit* is one folder or a list of them, highest first.  *run_options* carries ``--solver-config-dir``
    and its environment twin; ``None`` resolves them from *environ* (default ``os.environ``).  *cwd* defaults to
    the current folder.
    """
    folders = split_solver_config_dirs(explicit)
    if folders:
        return _closed_ladder(folders, REASON_ARGUMENT)
    if run_options is None:
        from flextool.run_options import RunOptions

        run_options = RunOptions.from_env(environ)
    layers: list[tuple[Path, str, bool]] = []
    found = run_options.resolved.get("workspace_solver_config_dir")
    if found is not None and found.value and found.source == "cli":
        # Folders a launcher (the FlexTool window) found: taken as given, each trusted only if listed.
        layers += [(folder, REASON_PROJECT if is_project_solver_config(folder) else REASON_WORKSPACE, False)
                   for folder in split_solver_config_dirs(found.value)]
    chosen = run_options.resolved.get("solver_config_dir")
    if chosen is not None and chosen.value and chosen.source in ("cli", "env"):
        reason = REASON_FLAG if chosen.source == "cli" else REASON_ENV
        layers += [(folder, reason, True) for folder in split_solver_config_dirs(chosen.value)]
    if layers:
        kept: list[tuple[Path, str, bool]] = []
        seen_keys: set[Path] = set()
        for folder, reason, named in layers:
            key = _folder_key(folder)
            if key not in seen_keys:  # a folder both found and named keeps its first (higher) place
                seen_keys.add(key)
                kept.append((folder, reason, named))
        return _found_ladder(kept)

    groups = _database_candidates(input_db_url)
    if project_folder_file:
        try:
            anchor = Path(project_folder_file).resolve().parent.parent
        except OSError:
            anchor = Path(project_folder_file).parent.parent
        groups.append([(anchor / SOLVER_CONFIG_FOLDER, REASON_TOOLBOX)])
    cwd_folder = Path(cwd if cwd is not None else Path.cwd()) / SOLVER_CONFIG_FOLDER
    groups.append([(cwd_folder, REASON_CWD)])
    tried: list[Path] = []
    skipped: list[Path] = []
    seen: set[Path] = set()  # resolved, so a symlinked current folder is looked at once
    cwd_key = _folder_key(cwd_folder)
    for group in groups:
        ladder: list[tuple[Path, str]] = []
        gives_a_file = False
        for folder, reason in group:
            key = _folder_key(folder)
            if key in seen:
                continue
            seen.add(key)
            if not _has_option_file(folder):
                tried.append(folder)
                continue
            if key == cwd_key and not is_flextool_workspace(cwd_folder.parent) and is_template_copy(folder):
                # FlexTool 4.3 and older copied the templates into the folder a run started from.
                skipped.append(folder)
                continue
            usable = any(_layer_gives(folder, solver)[0] for solver in SOLVER_NAMES)
            if not usable and len(group) > 1:
                # A GUI project's folder of unedited templates only: it gives nothing in a ladder.
                skipped.append(folder)
                continue
            ladder.append((folder, reason))
            gives_a_file = gives_a_file or usable
        if ladder and gives_a_file:
            return _found_ladder([(folder, reason, False) for folder, reason in ladder], skipped=tuple(skipped))
        tried.extend(folder for folder, _reason in ladder)
    return SolverConfigSource(None, REASON_DEFAULTS, tuple(tried), tuple(skipped))


def _folder_key(folder: Path) -> Path:
    try:
        return folder.resolve()
    except OSError:
        return folder


def workspace_solver_config_dirs(
    environ: Mapping[str, str] | None = None, cwd: str | os.PathLike | None = None,
) -> list[Path]:
    """The workspace's solver option folders, for launchers that know the workspace (GUI, project packaging).

    The folders of ``$FLEXTOOL_SOLVER_CONFIG_DIR`` when set (separated by :data:`os.pathsep`), else
    ``<cwd>/solver_config`` (the GUI's current folder is the workspace).
    """
    from flextool.run_options import split_path_list

    env = (os.environ if environ is None else environ).get("FLEXTOOL_SOLVER_CONFIG_DIR", "")
    folders = split_solver_config_dirs(split_path_list(env))
    if folders:
        return folders
    return [Path(cwd if cwd is not None else Path.cwd()) / SOLVER_CONFIG_FOLDER]


def gui_solver_config_dirs(
    project_path: str | os.PathLike | None,
    environ: Mapping[str, str] | None = None,
    cwd: str | os.PathLike | None = None,
) -> list[Path]:
    """The ladder a FlexTool window run of the project at *project_path* uses: ``<project>/solver_config`` when
    that folder exists, then :func:`workspace_solver_config_dirs` (``$FLEXTOOL_SOLVER_CONFIG_DIR`` replaces only
    the workspace's folder; the project's stays on top).  The GUI passes it to every run as
    flags (:func:`gui_solver_config_flags`), so its runs never depend on where the input
    database is.  A project without the folder is left out of the ladder, so it gives no "does not exist"
    warning.  The window passes it split by trust (:func:`gui_solver_config_flags`)."""
    folders: list[Path] = []
    if project_path is not None:
        own = Path(project_path) / SOLVER_CONFIG_FOLDER
        if _is_dir(own):
            folders.append(own)
    return split_solver_config_dirs([*folders, *workspace_solver_config_dirs(environ, cwd)])


def gui_solver_config_ladder(
    project_path: str | os.PathLike | None,
    environ: Mapping[str, str] | None = None,
    cwd: str | os.PathLike | None = None,
) -> tuple[list[Path], list[Path]]:
    """The ladder of :func:`gui_solver_config_dirs` split by trust (item 240 B3): ``(found, named)``.  *found* are
    the folders the FlexTool window found on its own (the project's ``solver_config`` when it exists, and the
    workspace's when ``$FLEXTOOL_SOLVER_CONFIG_DIR`` is not set), passed as ``--workspace-solver-config-dir``
    and trusted only when this computer set up their project or workspace; *named* are the folders of
    ``$FLEXTOOL_SOLVER_CONFIG_DIR`` (the user's choice, trusted), passed as ``--solver-config-dir``.  The child
    puts *found* above *named*, so the order of :func:`gui_solver_config_dirs` holds."""
    from flextool.run_options import split_path_list

    found: list[Path] = []
    if project_path is not None:
        own = Path(project_path) / SOLVER_CONFIG_FOLDER
        if _is_dir(own):
            found.append(own)
    env = (os.environ if environ is None else environ).get("FLEXTOOL_SOLVER_CONFIG_DIR", "")
    named = split_solver_config_dirs(split_path_list(env))
    if not named:
        found.append(Path(cwd if cwd is not None else Path.cwd()) / SOLVER_CONFIG_FOLDER)
    return split_solver_config_dirs(found), named


def gui_solver_config_flags(
    project_path: str | os.PathLike | None,
    environ: Mapping[str, str] | None = None,
    cwd: str | os.PathLike | None = None,
) -> list[str]:
    """The command-line flags of :func:`gui_solver_config_ladder`: ``--workspace-solver-config-dir <folder>``
    per found folder, then ``--solver-config-dir <folder>`` per named one, each folder its own argument."""
    found, named = gui_solver_config_ladder(project_path, environ, cwd)
    out: list[str] = []
    for folder in found:
        out += ["--workspace-solver-config-dir", str(folder)]
    return out + solver_config_dir_flags(named)


def solver_config_dir_flags(folders: Sequence[str | os.PathLike]) -> list[str]:
    """``--solver-config-dir <folder>`` once per folder, highest first (each folder its own argument, so no
    folder name is ever split)."""
    out: list[str] = []
    for folder in split_solver_config_dirs(folders):
        out += ["--solver-config-dir", str(folder)]
    return out


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
    could not be written).  *defaults* are the solvers that use FlexTool's packaged default; *file_notes* why a
    file of a folder trusted through this computer's list is not covered by that trust (it changed since in a way
    FlexTool cannot take as the user's own edit, review b2b3 M1).
    """

    source: SolverConfigSource
    snapshot_dir: Path | None = None
    texts: Mapping[str, str] = field(default_factory=dict)
    origins: Mapping[str, str] = field(default_factory=dict)
    defaults: frozenset[str] = frozenset()
    file_notes: Mapping[str, str] = field(default_factory=dict)

    def text(self, solver: str) -> str | None:
        return self.texts.get(solver)

    def layer_index(self, solver: str) -> int | None:
        """The index in the ladder of the folder *solver*'s file came from; ``None``: FlexTool's packaged default
        (matched by the exact file path, never by the text of the origin)."""
        origin = self.origins.get(solver)
        if origin is None:
            return None
        return next(
            (i for i, (folder, _r) in enumerate(self.source.layers) if origin == str(folder / f"{solver}.opt")), None,
        )

    def untrusted_note(self, solver: str) -> str | None:
        """``None`` when *solver*'s file is trusted, else why its lines get the input database's rules and how to
        trust its folder (item 240 B3).

        FlexTool's own file is told by its content only: the text is byte for byte (CRLF read as LF) a shipped
        template, this version's or an earlier one (review b2b3 M3: never by the text of its origin or the name of
        a folder).  Any other file is judged by the folder of the ladder it came from: trusted when the user named
        that folder, or when this computer's list trusts the folder and covers the file (it is as it was then, or
        the user's own later edit).  A trusted workspace's file overridden by an untrusted project's file leaves
        the project's file untrusted.  A file whose folder is not on the ladder is not trusted."""
        text = self.texts.get(solver)
        if text is None:
            return None
        if is_unedited_template(solver, text.encode("utf-8")):
            return None
        index = self.layer_index(solver)
        if index is None:
            origin = self.origins.get(solver, f"{solver}.opt")
            return (f"FlexTool cannot tell which folder {origin} came from, so its lines get the input database's "
                    "rules. Pass its folder with --solver-config-dir to trust it.")
        note = self.source.layer_note(index)
        if note is not None:
            return note
        return self.file_notes.get(solver)

    def origin(self, solver: str) -> str:
        return self.origins.get(solver, f"{solver}.opt")

    def record(self) -> dict[str, Any]:
        """The run record's ``solver_config`` block: the top folder (``None``: FlexTool's defaults) and why, every
        folder of the ladder with its reason (``layers``), the folders skipped as unedited template copies, the
        snapshot folder, and per option file its source (the file, or ``"FlexTool default"``), the reason of the
        folder it came from (``"FlexTool default"`` for the packaged one), whether it was trusted (item 240 B3),
        the SHA-256 of the text the run used (CRLF read as LF, the hash of :data:`SHIPPED_TEMPLATE_SHA256`) and
        whether that is an unedited FlexTool template; whether the top folder and each folder of the ladder are
        trusted; and whether this computer allows solver parameter names that are not in FlexTool's lists from
        databases, and where that was set (:func:`flextool._local_solver_settings.allow_unlisted_record`).  Paths
        are absolute, so the record means the same read from elsewhere."""
        from flextool._local_solver_settings import allow_unlisted_record

        files: dict[str, dict[str, Any]] = {}
        layers = self.source.layers
        for solver in SOLVER_NAMES:
            text = self.texts.get(solver)
            if text is None:
                continue
            digest = _normalised_sha256(text.encode("utf-8"))
            index = self.layer_index(solver)
            default = solver in self.defaults
            files[f"{solver}.opt"] = {
                "source": "FlexTool default" if default or index is None else _absolute(self.origins[solver]),
                "reason": "FlexTool default" if default or index is None else layers[index][1],
                "trusted": self.untrusted_note(solver) is None,
                "sha256": digest,
                "default": digest in _template_hashes(solver),
            }
        return {
            "dir": _absolute(self.source.dir) if self.source.dir is not None else None,
            "reason": self.source.reason,
            "trusted": self.source.trusted,
            "layers": [
                {"dir": _absolute(folder), "reason": reason, "trusted": self.source.layer_note(i) is None}
                for i, (folder, reason) in enumerate(layers)
            ],
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
    """Read the option file of each solver in *names* from the first folder of *source*'s ladder that gives it
    (else its packaged template), once.  A file of a folder trusted through this computer's list is checked
    against the list's entry with the bytes read and the file's ``lstat`` (review b2b3 M1)."""
    from flextool._local_solver_settings import LocalSolverSettingsError, trust_status

    texts: dict[str, str] = {}
    origins: dict[str, str] = {}
    defaults: set[str] = set()
    file_notes: dict[str, str] = {}
    statuses: dict[Path, object] = {}
    for solver in names:
        for index, (folder, reason) in enumerate(source.layers):
            if not _layer_gives(folder, solver)[0]:
                continue
            path = folder / f"{solver}.opt"
            try:
                st = os.lstat(path)
                data = path.read_bytes()
                # As read_text: universal newlines.
                texts[solver] = data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
            except UnicodeDecodeError as exc:
                _logger.warning("Not reading %s: it is not UTF-8 text (%s).", path, exc)
                continue
            except OSError:
                continue
            origins[solver] = str(path)
            if source.layer_listed(index) and source.layer_note(index) is None:
                if folder not in statuses:
                    try:
                        statuses[folder] = trust_status(folder.parent)
                    except LocalSolverSettingsError as exc:
                        statuses[folder] = str(exc)
                status = statuses[folder]
                problem = status if isinstance(status, str) else status.file_problem(
                    path.name, _normalised_sha256(data), st)
                if problem is not None and not is_unedited_template(solver, data):
                    file_notes[solver] = untrusted_note(folder, reason, changed=problem)
            break
        else:
            template = _template_path(solver)
            if template is None:
                continue
            try:
                texts[solver] = template.read_text(encoding="utf-8")
            except OSError:
                continue
            origins[solver] = f"FlexTool default ({solver}.opt.template)"
            defaults.add(solver)
    return SolverOptionFiles(source=source, texts=texts, origins=origins, defaults=frozenset(defaults),
                             file_notes=file_notes)


def snapshot_solver_option_files(
    source: SolverConfigSource,
    work_folder: str | os.PathLike,
    logger: logging.Logger | None = None,
    names: tuple[str, ...] = SOLVER_NAMES,
) -> SolverOptionFiles:
    """Read the run's option files once and write them to ``<work_folder>/solve_data/solver_config/``.

    Every run writes the snapshot afresh (an older one is replaced).  Logs the ladder (``Solver config:``), one
    warning when no folder was found (or one per folder skipped as an unedited template copy, which also names the
    folder used), one warning per folder of the ladder that does not exist, for a one-folder ladder one line for
    the solvers that use the packaged default, and one warning when a ``<work_folder>/solver_config/`` from an
    earlier version differs from what the run uses (it is no longer read).  A snapshot that cannot be written
    gives a warning; the run goes on with the texts read.
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
    elif source.lower:
        for folder, reason in source.layers:
            if not _is_dir(folder):
                log.warning(
                    "Solver config folder %s (%s) does not exist; its solvers use the next folder or FlexTool's "
                    "default options.",
                    folder, reason,
                )
    elif not source.dir.is_dir():
        log.warning(
            "Solver config folder %s (%s) does not exist; every solver uses FlexTool's default options.",
            source.dir, source.reason,
        )
    else:
        passed = [p for s in names for p in _ladder_file(source.layers, s)[1]]
        defaults = [
            f"{s}.opt" for s in names if s in files.origins and files.origins[s] != str(source.dir / f"{s}.opt")
        ]
        if defaults and not passed:  # with a skipped template the header names the defaults already
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
        return replace(files, snapshot_dir=None)
    return replace(files, snapshot_dir=snapshot)


def _warn_stale_work_folder_copy(
    work_folder: Path, source: SolverConfigSource, files: SolverOptionFiles, log: logging.Logger,
) -> None:
    """One warning when ``<work_folder>/solver_config/`` (the copy of FlexTool 4.3 and older) differs from the
    files the run uses.  Not when it is a folder of the ladder itself (a run from the workspace)."""
    old = work_folder / SOLVER_CONFIG_FOLDER
    if not old.is_dir():
        return
    try:
        if any(old.resolve() == folder.resolve() for folder, _reason in source.layers):
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
    "TRUST_COMMAND",
    "gui_solver_config_dirs",
    "gui_solver_config_flags",
    "gui_solver_config_ladder",
    "awaits_review",
    "forget_trusted_folder",
    "holds_only_flextool_defaults",
    "note_project_set_up_here",
    "input_db_filesystem_path",
    "is_flextool_workspace",
    "is_project_solver_config",
    "is_template_copy",
    "is_unedited_template",
    "read_solver_option_files",
    "resolve_solver_config_source",
    "snapshot_solver_option_files",
    "solver_config_dir_flags",
    "split_solver_config_dirs",
    "untrusted_note",
    "workspace_solver_config_dirs",
]
