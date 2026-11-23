"""Consent dialogs for upgrading (migrating) externally-referenced and frozen FlexTool databases, on the common
FlexTool message box."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from typing import Callable, Literal

from flextool.gui.dialogs import message_box

_DOCS_MIGRATION = "flextool_gui_interface.md#migration-and-frozen-databases"


def ask_external_migration_consent(
    parent: tk.Misc,
    external_files: list[tuple[str, Path, int, int]],
) -> Literal["in_place", "copy_to_project", "cancel"]:
    """Ask the user how to migrate external (non-project) input source files.

    ``external_files`` is a list of ``(source_name, abs_path,
    current_version, target_version)`` tuples; all entries share the
    same target version.

    Returns:
        ``"in_place"`` to migrate the external files where they live,
        ``"copy_to_project"`` to copy them into the project folder and
        migrate the copies, or ``"cancel"`` to abort (also Escape and the
        window button).
    """
    target_version = external_files[0][3]
    lines = []
    for source_name, abs_path, current_version, _t in external_files:
        change = (
            f"version {current_version}, database format update"
            if current_version == target_version
            else f"version {current_version} to {target_version}"
        )
        lines.append(f"{source_name}: {abs_path} ({change})")
    answer = message_box.ask(
        parent, "Upgrade files outside the project?",
        f"These files are outside the project folder, and this FlexTool needs data version {target_version}:",
        message_box.name_list(lines),
        "Upgrading changes the files. Upgrade them where they are, or copy them into the project and upgrade the "
        "copies.",
        buttons=(("Upgrade in place", "in_place"), ("Copy into project and upgrade", "copy_to_project"),
                 ("Cancel", None)),
        default="copy_to_project", docs=_DOCS_MIGRATION,
    )
    return answer if answer in ("in_place", "copy_to_project") else "cancel"


FrozenChoice = Literal["copy", "in_place", "cancel"]


def ask_frozen_migration_choice(
    parent: tk.Misc,
    source_name: str,
    path: Path,
    label: str,
    current_version: int | None,
    target_version: int,
    copy_name: str,
    *,
    confirm_in_place: Callable[[tk.Misc], bool] | None = None,
) -> FrozenChoice:
    """Ask how to migrate a frozen input source that is older than this FlexTool.

    Returns ``"copy"`` (default: migrate a copy, *copy_name* in
    ``input_sources/``; the frozen file stays unchanged), ``"in_place"`` (after
    a second confirm, *confirm_in_place*; the frozen file changes and keeps its
    marker with the new version) or ``"cancel"`` (skip the source this
    session; also Escape and the window button).  Declining the second confirm
    asks the first question again.
    """
    if confirm_in_place is None:
        confirm_in_place = _confirm_in_place
    found = f"version {current_version}" if current_version is not None else "unreadable version"
    while True:
        answer = message_box.ask(
            parent, "This frozen database needs an upgrade",
            f"{source_name} is frozen ('{label}', {found}); this FlexTool needs version {target_version}.",
            str(path),
            f"Upgrade a copy (recommended): FlexTool adds the copy {copy_name} as a new input source, so its "
            "results go to new folders. The frozen file stays as it is.",
            f"Upgrade in place changes the frozen file itself; Cancel leaves it as it is, and {source_name} "
            "cannot run in this session.",
            buttons=(("Upgrade a copy", "copy"), ("Upgrade in place", "in_place"), ("Cancel", None)),
            default="copy", docs=_DOCS_MIGRATION,
        )
        if answer == "in_place" and not confirm_in_place(parent):
            continue
        return answer if answer in ("copy", "in_place") else "cancel"


def _confirm_in_place(parent: tk.Misc) -> bool:
    """The second confirm of "Upgrade in place"; focus starts on Cancel."""
    return message_box.ask(
        parent, "Change the frozen file?",
        "The frozen file itself is upgraded. It stays frozen, with the new version.",
        buttons=(("Upgrade it", "upgrade"), ("Cancel", None)), kind="danger",
    ) == "upgrade"
