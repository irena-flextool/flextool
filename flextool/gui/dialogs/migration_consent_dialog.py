"""Consent dialogs for migrating externally-referenced and frozen FlexTool databases."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Callable, Literal


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
        migrate the copies, or ``"cancel"`` to abort.
    """
    target_version = external_files[0][3]

    dlg = tk.Toplevel(parent)
    dlg.title("External database migration")
    dlg.transient(parent)
    dlg.grab_set()
    dlg.resizable(False, False)

    result: Literal["in_place", "copy_to_project", "cancel"] = "cancel"

    msg = (
        "The following input source file(s) are stored OUTSIDE the "
        f"project folder and need to be migrated to FlexTool DB "
        f"version {target_version}:"
    )
    lbl = ttk.Label(dlg, text=msg, wraplength=480, justify="left")
    lbl.pack(padx=16, pady=(16, 8), anchor="w")

    list_frame = ttk.Frame(dlg, borderwidth=1, relief="sunken")
    list_frame.pack(padx=16, pady=(0, 8), fill="x")

    height = min(6, len(external_files))
    listbox = tk.Listbox(list_frame, height=height, activestyle="none")
    h_scroll = ttk.Scrollbar(
        list_frame, orient="horizontal", command=listbox.xview,
    )
    listbox.configure(xscrollcommand=h_scroll.set)

    for source_name, abs_path, current_version, _t in external_files:
        change = (
            f"v{current_version}, SpineDB schema upgrade"
            if current_version == target_version
            else f"v{current_version} → v{target_version}"
        )
        listbox.insert("end", f"{source_name}  —  {abs_path}  ({change})")

    listbox.pack(side="top", fill="x")
    h_scroll.pack(side="bottom", fill="x")

    note = ttk.Label(
        dlg,
        text="Migration modifies the file in place. Choose how to proceed:",
        wraplength=480,
        justify="left",
    )
    note.pack(padx=16, pady=(0, 8), anchor="w")

    btn_frame = ttk.Frame(dlg)
    btn_frame.pack(padx=16, pady=(4, 16))

    def _choose(choice: Literal["in_place", "copy_to_project", "cancel"]) -> None:
        nonlocal result
        result = choice
        dlg.destroy()

    ttk.Button(
        btn_frame, text="Migrate in place",
        command=lambda: _choose("in_place"),
    ).pack(side="left", padx=4)
    ttk.Button(
        btn_frame, text="Copy all to project and migrate",
        command=lambda: _choose("copy_to_project"),
    ).pack(side="left", padx=4)
    ttk.Button(
        btn_frame, text="Cancel",
        command=lambda: _choose("cancel"),
    ).pack(side="left", padx=4)

    dlg.protocol("WM_DELETE_WINDOW", lambda: _choose("cancel"))
    dlg.bind("<Escape>", lambda _e: _choose("cancel"))

    # Center on parent
    dlg.update_idletasks()
    pw = parent.winfo_width()
    ph = parent.winfo_height()
    px = parent.winfo_rootx()
    py = parent.winfo_rooty()
    dw = dlg.winfo_width()
    dh = dlg.winfo_height()
    dlg.geometry(f"+{px + (pw - dw) // 2}+{py + (ph - dh) // 2}")

    parent.wait_window(dlg)
    return result


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
    session).
    """
    if confirm_in_place is None:
        confirm_in_place = _confirm_in_place

    dlg = tk.Toplevel(parent)
    dlg.title("Frozen database")
    dlg.transient(parent)
    dlg.grab_set()
    dlg.resizable(False, False)

    result: FrozenChoice = "cancel"
    found = f"v{current_version}" if current_version is not None else "an unreadable version"
    msg = (
        f"{source_name} is frozen ('{label}', {found}). This FlexTool needs "
        f"v{target_version}.\n\n{path}\n\n"
        f"Migrate a copy: the copy {copy_name} is added to the input sources "
        f"and migrated; it is a new input source with its own number, so its "
        f"results go to new result folders. The frozen file is not changed.\n\n"
        f"Migrate in place: the frozen file itself is migrated.\n\n"
        f"Cancel: {source_name} is not migrated and cannot be run in this "
        f"session."
    )
    ttk.Label(dlg, text=msg, wraplength=520, justify="left").pack(
        padx=16, pady=(16, 8), anchor="w"
    )

    btn_frame = ttk.Frame(dlg)
    btn_frame.pack(padx=16, pady=(4, 16))

    def _choose(choice: FrozenChoice) -> None:
        nonlocal result
        if choice == "in_place" and not confirm_in_place(dlg):
            return
        result = choice
        dlg.destroy()

    copy_btn = ttk.Button(
        btn_frame, name="copy", text="Migrate a copy",
        command=lambda: _choose("copy"),
    )
    copy_btn.pack(side="left", padx=4)
    ttk.Button(
        btn_frame, name="in_place", text="Migrate in place",
        command=lambda: _choose("in_place"),
    ).pack(side="left", padx=4)
    ttk.Button(
        btn_frame, name="cancel", text="Cancel",
        command=lambda: _choose("cancel"),
    ).pack(side="left", padx=4)

    dlg.protocol("WM_DELETE_WINDOW", lambda: _choose("cancel"))
    dlg.bind("<Escape>", lambda _e: _choose("cancel"))
    dlg.bind("<Return>", lambda _e: _choose("copy"))
    copy_btn.focus_set()

    dlg.update_idletasks()
    pw = parent.winfo_width()
    ph = parent.winfo_height()
    px = parent.winfo_rootx()
    py = parent.winfo_rooty()
    dw = dlg.winfo_width()
    dh = dlg.winfo_height()
    dlg.geometry(f"+{px + (pw - dw) // 2}+{py + (ph - dh) // 2}")

    parent.wait_window(dlg)
    return result


def _confirm_in_place(parent: tk.Misc) -> bool:
    """The second confirm of "Migrate in place"."""
    from tkinter import messagebox

    return messagebox.askokcancel(
        "Migrate the frozen file",
        "The frozen file will change. Its frozen marker is kept with the new "
        "version. Continue?",
        parent=parent,
        icon="warning",
    )
