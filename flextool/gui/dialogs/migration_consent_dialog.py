"""Consent dialog for migrating externally-referenced FlexTool databases."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Literal


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
        listbox.insert(
            "end",
            f"{source_name}  —  {abs_path}  "
            f"(v{current_version} → v{target_version})",
        )

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
