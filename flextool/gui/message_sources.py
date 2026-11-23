"""The message list's real sources (design ``specs/gui_message_list_design.md`` §3.1; package M3a).

Each source is a ``source(store) -> list[GuiMessage]`` (:data:`flextool.gui.messages.MessageSource`) that builds
its messages, texts included, from current facts at every recompute.  No Tk here.

- ``value_list_check`` (W-0490, package M3b): one message per input-source database that
  ``possible_value_list_overwrite_ro`` flags (:meth:`~flextool.gui.input_sources.InputSourceManager.value_list_findings`),
  with a "Copy repair command" button.  Not dismissable: a valid repair record (the repair tool) ends the finding,
  and with it the message.  TODO(v5.0.0): remove with W-0490.
- ``list_defaults_lost`` (W-0491, package P7b): one message per input-source database whose shipped list defaults
  an upgrade before 4.6.3 left empty
  (:meth:`~flextool.gui.input_sources.InputSourceManager.list_defaults_findings`), with the same button.  Not
  dismissable and not attention (a note: it opens the list by itself at two project opens per set of empty
  defaults); only the repair tool's part B record ends it.  TODO(v5.0.0): remove with W-0491.
- ``database_upgraded``: one message per database with a pending upgrade event (posted by :func:`post_upgrades`
  after a migration run), for the databases that are input sources of the open project.
- ``database_unreadable``: one message per :class:`~flextool.gui.input_sources.PlanningNote` of the latest
  migration plan (a file whose data version or frozen marker cannot be read).
- ``solver_defaults_updated``: the open project's pending event (posted when its old solver settings were
  updated); its text is the shared behaviour-change text (:func:`~flextool.gui.settings_io.solver_defaults_message`).
- ``update_available``: the stored offered FlexTool version, for a PyPI install with the start-up update check on.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from flextool.cli.cmd_repair_value_list import repair_command
from flextool.gui.messages import (
    MESSAGE_SPECS,
    USER_KEY,
    GuiMessage,
    MessageAction,
    MessageCenter,
    MessageSource,
    MessageStore,
    database_key,
    project_key,
)

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from flextool.gui.input_sources import PlanningNote
    from flextool.update_flextool.list_default_restore import LostDefaults
    from flextool.update_flextool.value_list_history import Finding

_VALUE_LIST = MESSAGE_SPECS["value_list_check"]
_LIST_DEFAULTS = MESSAGE_SPECS["list_defaults_lost"]
_UPGRADED = MESSAGE_SPECS["database_upgraded"]
_UNREADABLE = MESSAGE_SPECS["database_unreadable"]
_SOLVER = MESSAGE_SPECS["solver_defaults_updated"]
_UPDATE = MESSAGE_SPECS["update_available"]

_OLDER_FLEXTOOL = "An older FlexTool may not be able to run it any more."
_GONE_WHEN_READABLE = ("Close other programs that use it and press Refresh; this message goes away once FlexTool "
                       "can read the file.")


@dataclass(frozen=True)
class UpgradeResult:
    """What the migration of one file did (``main_window.upgrade_one``).

    *before* / *after*: the data version read before and after (``None`` when it could not be read); *copy_of*:
    the frozen file's name when *path* is its upgraded copy; *messages*: those of ``check_and_upgrade_database``.
    """
    name: str
    path: Path
    before: int | None
    after: int | None
    copy_of: str | None
    upgraded: bool
    failed: bool
    messages: list[str] = field(default_factory=list)


def post_upgrades(center: MessageCenter | None, results: Iterable[UpgradeResult], run: str) -> None:
    """Post a ``database_upgraded`` event for every file that was upgraded and did not fail (*run*: the start of
    the migration run, which makes each event a new message).  Nothing without a center."""
    if center is None:
        return
    for r in results:
        if not r.upgraded or r.failed:
            continue
        facts: dict[str, Any] = {"from": r.before, "to": r.after, "copy_of": r.copy_of}
        if r.before is not None and r.before == r.after:
            facts["schema"] = True
        center.post_event("database_upgraded", database_key(r.path), Path(r.path).name, run=run, facts=facts)


def failure_lines(results: Iterable[UpgradeResult], *, cancelled: bool) -> list[str]:
    """The lines of the "upgrade failed" box: the first paragraph of each message of a failed file, and of a file
    that neither upgraded nor failed but has messages (such as "spinedb_api not available"), unless the user
    cancelled the run (then the Execution window has them).  Tracebacks stay in the Execution window
    (:func:`_box_line`)."""
    lines: list[str] = []
    for r in results:
        if r.failed or (not r.upgraded and r.messages and not cancelled):
            lines.extend(_box_line(msg) for msg in r.messages if msg.strip())
    return lines


def _box_line(message: str) -> str:
    """A migration message without its traceback (``check_and_upgrade_database`` puts ``Error: …`` and the
    traceback after blank lines); the traceback is in the Execution window, so "below" points there."""
    head, *rest = (part.strip() for part in message.split("\n\n"))
    head = head.replace("the traceback below", "the traceback in the Execution window")
    error = next((part for part in rest if part.startswith("Error:")), None)
    return f"{head} {error}" if error else head


# -- texts -----------------------------------------------------------------------------------------------------------
def upgraded_text(facts: dict) -> tuple[str, tuple[str, ...]]:
    """Title and paragraphs of a ``database_upgraded`` message, from the facts of its event."""
    before, after, copy_of = facts.get("from"), facts.get("to"), facts.get("copy_of")
    if before is not None and after is not None and before != after:
        versions = f"from data version {before} to {after}"
    else:
        versions = f"to data version {after}" if after is not None else "to the current data version"
    if copy_of:
        return "An upgraded copy was added", (
            f"This is an upgraded copy of the frozen database {copy_of} ({versions}), added as a new input "
            "source. The frozen file is unchanged.",)
    if facts.get("schema"):
        return "This database was upgraded", (
            f"FlexTool updated the database format of this file so that this FlexTool can run it. {_OLDER_FLEXTOOL}",)
    return "This database was upgraded", (
        f"FlexTool upgraded it {versions} so that this FlexTool can run it. {_OLDER_FLEXTOOL}",)


def unreadable_text(note: PlanningNote) -> tuple[str, tuple[str, ...]]:
    """Title and paragraphs of a ``database_unreadable`` message (inventory L2b)."""
    if note.kind == "frozen_unreadable":
        return "FlexTool cannot tell whether this file is frozen", (
            "FlexTool could not read the file's frozen marker, so it did not check whether the file needs an "
            "upgrade. Another program may be using the file, or it may be damaged.",
            _GONE_WHEN_READABLE,
            f"Details: {note.text}")
    return "FlexTool cannot read this database file", (
        "FlexTool could not read which data version this file has, so it did not upgrade it. The file may be open "
        "in another program, damaged, or not a FlexTool database.",
        _GONE_WHEN_READABLE)


REPAIR_LABEL = "Copy repair command"
REPAIR_COPIED = "Copied"


def repair_shell(os_name: str = os.name) -> str:
    """The shell the copied repair command is written for: PowerShell on Windows (the default shell of Windows
    Terminal; a cmd-quoted command fails there when the Python path has a space), a POSIX shell elsewhere."""
    return "powershell" if os_name == "nt" else "posix"


def repair_tooltip(os_name: str = os.name) -> str:
    """Where to paste the copied repair command (:func:`repair_shell`)."""
    return "Paste into PowerShell" if os_name == "nt" else "Paste into a terminal"


LIST_DEFAULTS_TITLE = "Shipped defaults lost in an old upgrade"
_STORAGE_DEFAULT = "node.storage_start_end_method"


def list_defaults_text(names: Iterable[str]) -> tuple[str, tuple[str, ...]]:
    """Title and paragraph of a ``list_defaults_lost`` message (W-0491) for the empty bug-reachable *names*: the
    storage sentence only when ``node.storage_start_end_method`` is among them (the one default that can change
    results)."""
    first = "An earlier upgrade left some default values of this database empty."
    stays = "This message stays until the repair is recorded."
    if _STORAGE_DEFAULT in set(names):
        return LIST_DEFAULTS_TITLE, (
            f"{first} Results can differ only in storage start and end handling (storage_start_end_method). The "
            "repair tool makes a repaired copy and asks about those storage nodes: use \"Copy repair command\" "
            f"and paste it into a terminal. {stays}",)
    return LIST_DEFAULTS_TITLE, (
        f"{first} Results do not change. The repair tool makes a repaired copy with the shipped defaults: use "
        f"\"Copy repair command\" and paste it into a terminal. {stays}",)


SOLVER_DEFAULTS_TITLE = "This project's solver settings were updated"


def update_text(offered: str, installed: str) -> tuple[str, tuple[str, ...]]:
    """Title and paragraphs of the ``update_available`` message (inventory L4)."""
    return "A newer FlexTool is available", (
        f"FlexTool {offered} is available; you have {installed}. Click Update FlexTool… at the top right to "
        "install it.",)


#: The W-0490 message when its text cannot be built for a database (the finding stands; the docs explain it).
W0490_FALLBACK = ("Please check this database", (
    "A bug in a library that FlexTool uses may have changed this database when it was upgraded. Open the "
    "documentation to see what to check.",
    "FlexTool's repair tool writes a repaired copy and leaves this file as it is: the Copy repair command button "
    "copies the command to paste into a terminal.",
    "This message stays until a repaired file is in its place."))


#: The W-0490 message when the database's upgrade history cannot be read (no finding can be ruled out).
W0490_UNREADABLE = ("Please check this database", (
    "FlexTool could not read this database's upgrade history, so it cannot tell whether a bug in a library that "
    "FlexTool uses changed it when it was upgraded. Open the documentation to see what to check.",
    "FlexTool's repair tool checks the file and writes a repaired copy if one is needed: the Copy repair command "
    "button copies the command to paste into a terminal.",
    "FlexTool reads the file again at the next refresh."))


# -- sources ---------------------------------------------------------------------------------------------------------
def copy_repair_action(path: Path, clipboard: Callable[[str], None]) -> MessageAction:
    """The "Copy repair command" button: puts ``repair_command(path, shell=repair_shell())`` (the repair tool's one
    command builder) on the clipboard; the button then reads "Copied" for a moment.  The GUI never runs the tool."""
    def run() -> str:
        clipboard(repair_command(path, shell=repair_shell()))
        return REPAIR_COPIED
    return MessageAction(REPAIR_LABEL, run, tooltip=repair_tooltip())


def value_list_source(findings: Callable[[], Iterable[tuple[Path, list[Finding] | None]]],
                      clipboard: Callable[[str], None],
                      unreadable: Callable[[], Iterable[Path]] = tuple) -> MessageSource:
    """``value_list_check`` (W-0490) messages, one per database in *findings()* (``value_list_findings``: the
    flagged input sources of the open project; findings ``None``: its history could not be read, the generic
    :data:`W0490_UNREADABLE` text).  The generic text is left out for a database in *unreadable()* (the files with
    a ``database_unreadable`` message, such as "FlexTool cannot read this database file": that one is not
    dismissable either and stays until FlexTool can read the file, so nothing is silenced).  TODO(v5.0.0): remove with W-0490."""
    def source(_store: MessageStore) -> list[GuiMessage]:
        from flextool.update_flextool.value_list_history import gui_message

        out = []
        found_list = list(findings())
        cannot_read = ({database_key(p) for p in unreadable()}
                       if any(found is None for _p, found in found_list) else set())
        for path, found in found_list:
            if found is None and database_key(path) in cannot_read:
                continue
            try:        # one database's text failing must not hide W-0490 for the others, nor for itself
                title, paragraphs = W0490_UNREADABLE if found is None else gui_message(found)
            except Exception:
                _logger.exception("Message list: the W-0490 text of %s failed", path)
                title, paragraphs = W0490_FALLBACK
            out.append(GuiMessage(_VALUE_LIST, database_key(path), Path(path).name, title, paragraphs,
                                  action=copy_repair_action(path, clipboard)))
        return out
    return source


def list_defaults_source(findings: Callable[[], Iterable[tuple[Path, LostDefaults]]],
                         clipboard: Callable[[str], None]) -> MessageSource:
    """``list_defaults_lost`` (W-0491) messages, one per database in *findings()* (``list_defaults_findings``);
    the instance is the hash of the database's full set of empty defaults, so a changed set is a new message.
    TODO(v5.0.0): remove with W-0491."""
    def source(_store: MessageStore) -> list[GuiMessage]:
        out = []
        for path, lost in findings():
            try:        # one database failing must not hide the others' messages
                title, paragraphs = list_defaults_text(lost.names)
                out.append(GuiMessage(_LIST_DEFAULTS, database_key(path), Path(path).name, title, paragraphs,
                                      instance=lost.empty_hash, action=copy_repair_action(path, clipboard)))
            except Exception:
                _logger.exception("Message list: the W-0491 message of %s failed", path)
        return out
    return source


def upgraded_source(database_keys: Callable[[], Iterable[str]]) -> MessageSource:
    """``database_upgraded`` messages of the pending events of the databases in *database_keys()* (the input
    sources of the open project)."""
    def source(store: MessageStore) -> list[GuiMessage]:
        keys = set(database_keys())
        out = []
        for subject, entry in store.pending("database_upgraded").items():
            if subject not in keys:
                continue
            title, paragraphs = upgraded_text(entry.get("facts") or {})
            out.append(GuiMessage(_UPGRADED, subject, entry.get("label", ""), title, paragraphs,
                                  instance=str(entry.get("run", ""))))
        return out
    return source


def unreadable_source(notes: Callable[[], Iterable[PlanningNote]]) -> MessageSource:
    """``database_unreadable`` messages, one per note of the latest migration plan (*notes()*)."""
    def source(_store: MessageStore) -> list[GuiMessage]:
        out = []
        for note in notes():
            title, paragraphs = unreadable_text(note)
            out.append(GuiMessage(_UNREADABLE, database_key(note.path), note.name, title, paragraphs))
        return out
    return source


def solver_defaults_source(project_dir: Path, project_name: str) -> MessageSource:
    """The ``solver_defaults_updated`` message of the open project, while its event is pending."""
    subject = project_key(project_dir)

    def source(store: MessageStore) -> list[GuiMessage]:
        entry = store.pending("solver_defaults_updated").get(subject)
        if entry is None:
            return []
        from flextool.gui.settings_io import solver_defaults_message

        return [GuiMessage(_SOLVER, subject, f"Project {project_name}", SOLVER_DEFAULTS_TITLE,
                           solver_defaults_message(project_name), instance=str(entry.get("run", "")))]
    return source


def update_source(*, enabled: Callable[[], bool], is_git: Callable[[], bool], installed: Callable[[], str],
                  is_newer: Callable[[str], bool]) -> MessageSource:
    """The ``update_available`` message: the stored offered version while it is newer than the installed one, the
    install is not a git checkout and the start-up update check is on (*enabled()*)."""
    def source(store: MessageStore) -> list[GuiMessage]:
        offered = store.offered_version()
        if offered is None or not enabled() or is_git() or not is_newer(offered):
            return []
        title, paragraphs = update_text(offered, installed())
        return [GuiMessage(_UPDATE, USER_KEY, "", title, paragraphs, instance=offered)]
    return source


def update_check_enabled(global_settings: object) -> bool:
    """Whether the start-up update check runs: the preference is on and ``FLEXTOOL_NO_UPDATE_CHECK`` is unset."""
    return (not os.environ.get("FLEXTOOL_NO_UPDATE_CHECK")
            and bool(getattr(global_settings, "check_updates_on_startup", True)))


def project_sources(main: Any, project_dir: Path, project_name: str) -> list[MessageSource]:
    """The sources of the project *project_name* open in *main* (the main window; its input source manager and
    global settings are read at each recompute)."""
    from flextool.update_flextool import install_info

    git = install_info.is_git_install()

    def mgr() -> Any:
        return getattr(main, "input_source_mgr", None)

    def database_keys() -> list[str]:
        m = mgr()
        return [database_key(p) for p in m._registered_sqlite_paths()] if m is not None else []

    def notes() -> list[PlanningNote]:
        m = mgr()
        return list(getattr(m, "planning_notes", [])) if m is not None else []

    def unreadable_paths() -> list[Path]:
        return [note.path for note in notes()]

    def value_list_findings() -> list[tuple[Path, list[Finding] | None]]:
        m = mgr()
        return m.value_list_findings() if m is not None else []

    def list_defaults_findings() -> list[tuple[Path, LostDefaults]]:
        m = mgr()
        return m.list_defaults_findings() if m is not None else []

    def clipboard(text: str) -> None:
        main.clipboard_clear()
        main.clipboard_append(text)

    return [
        value_list_source(value_list_findings, clipboard, unreadable_paths),
        list_defaults_source(list_defaults_findings, clipboard),
        unreadable_source(notes),
        upgraded_source(database_keys),
        solver_defaults_source(project_dir, project_name),
        update_source(enabled=lambda: update_check_enabled(getattr(main, "global_settings", None)),
                      is_git=lambda: git, installed=install_info.flextool_version,
                      is_newer=install_info.is_newer_than_installed),
    ]
