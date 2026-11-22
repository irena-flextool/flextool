"""The "Please check this database" box (W-0490): a short text, "Open checklist" and "Close".

TODO(v5.0.0): remove with W-0490 (``flextool/update_flextool/value_list_history.py``).
"""
from __future__ import annotations

import tkinter as tk
import webbrowser
from tkinter import ttk
from typing import Callable

TITLE = "Please check this database"


def show_database_check(
    parent: tk.Misc,
    text: str,
    url: str,
    *,
    open_url: Callable[[str], object] = webbrowser.open,
) -> None:
    """Show *text* with "Open checklist" (opens *url* with *open_url*, the box stays open) and "Close"."""
    dlg = tk.Toplevel(parent)
    dlg.title(TITLE)
    dlg.transient(parent)
    dlg.grab_set()
    dlg.resizable(False, False)

    ttk.Label(dlg, name="text", text=text, wraplength=480, justify="left").pack(
        padx=16, pady=(16, 8), anchor="w"
    )

    btn_frame = ttk.Frame(dlg, name="buttons")
    btn_frame.pack(padx=16, pady=(4, 16))
    ttk.Button(btn_frame, name="open", text="Open checklist", command=lambda: open_url(url)).pack(
        side="left", padx=4
    )
    close_btn = ttk.Button(btn_frame, name="close", text="Close", command=dlg.destroy)
    close_btn.pack(side="left", padx=4)

    dlg.protocol("WM_DELETE_WINDOW", dlg.destroy)
    dlg.bind("<Escape>", lambda _e: dlg.destroy())
    close_btn.focus_set()

    dlg.update_idletasks()
    pw = parent.winfo_width()
    ph = parent.winfo_height()
    px = parent.winfo_rootx()
    py = parent.winfo_rooty()
    dw = dlg.winfo_width()
    dh = dlg.winfo_height()
    dlg.geometry(f"+{px + (pw - dw) // 2}+{py + (ph - dh) // 2}")

    parent.wait_window(dlg)
