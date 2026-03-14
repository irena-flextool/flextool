from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox

from flextool.gui.project_utils import (
    create_project,
    get_projects_dir,
    list_projects,
    rename_project,
)


class ProjectDialog(tk.Toplevel):
    """Modal dialog for creating, selecting, and renaming projects.

    After the dialog closes, ``self.result`` holds the selected project
    name or ``None`` if the user cancelled.
    """

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.title("Project menu")
        self.result: str | None = None

        # ── Modal behaviour ─────────────────────────────────────────
        self.transient(parent)
        self.grab_set()

        # ── Dialog size ─────────────────────────────────────────────
        self.geometry("420x400")
        self.resizable(False, False)

        self._rename_entry: tk.Entry | None = None

        self._build_widgets()
        self._populate_projects()

        # Close via window-manager "X"
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        # Centre on parent
        self.update_idletasks()
        px = parent.winfo_rootx()
        py = parent.winfo_rooty()
        pw = parent.winfo_width()
        ph = parent.winfo_height()
        w = self.winfo_width()
        h = self.winfo_height()
        x = px + (pw - w) // 2
        y = py + (ph - h) // 2
        self.geometry(f"+{x}+{y}")

        # Block until closed
        parent.wait_window(self)

    # ── Widget construction ─────────────────────────────────────────

    def _build_widgets(self) -> None:
        pad = dict(padx=10, pady=5)

        # -- New project section --
        new_frame = ttk.LabelFrame(self, text="New project", padding=8)
        new_frame.pack(fill="x", **pad)

        entry_row = ttk.Frame(new_frame)
        entry_row.pack(fill="x")

        ttk.Label(entry_row, text="Name:").pack(side="left")
        self._new_name_var = tk.StringVar()
        self._new_name_entry = ttk.Entry(entry_row, textvariable=self._new_name_var, width=30)
        self._new_name_entry.pack(side="left", padx=(5, 0), fill="x", expand=True)

        self._create_btn = ttk.Button(new_frame, text="Create", command=self._on_create)
        self._create_btn.pack(anchor="w", pady=(5, 0))

        # Allow Enter in the entry to create
        self._new_name_entry.bind("<Return>", lambda _e: self._on_create())

        # -- Existing projects section --
        existing_frame = ttk.LabelFrame(self, text="Existing projects", padding=8)
        existing_frame.pack(fill="both", expand=True, **pad)

        list_frame = ttk.Frame(existing_frame)
        list_frame.pack(fill="both", expand=True)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        self._project_listbox = tk.Listbox(list_frame, selectmode="browse", activestyle="dotbox")
        self._project_listbox.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self._project_listbox.yview)
        self._project_listbox.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky="ns")

        # Double-click opens
        self._project_listbox.bind("<Double-Button-1>", lambda _e: self._on_open())
        # F2 triggers inline rename
        self._project_listbox.bind("<F2>", lambda _e: self._start_inline_rename())

        # -- Bottom buttons --
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", **pad)

        self._open_btn = ttk.Button(btn_frame, text="Open project", command=self._on_open)
        self._open_btn.pack(side="left")

        self._cancel_btn = ttk.Button(btn_frame, text="Cancel", command=self._on_cancel)
        self._cancel_btn.pack(side="right")

    # ── Helpers ──────────────────────────────────────────────────────

    def _populate_projects(self) -> None:
        """Refresh the listbox with the current project list."""
        self._project_listbox.delete(0, "end")
        for name in list_projects():
            self._project_listbox.insert("end", name)

    def _selected_project(self) -> str | None:
        """Return the currently selected project name, or None."""
        sel = self._project_listbox.curselection()
        if not sel:
            return None
        return self._project_listbox.get(sel[0])

    # ── Event handlers ──────────────────────────────────────────────

    def _on_create(self) -> None:
        name = self._new_name_var.get().strip()
        if not name:
            messagebox.showwarning("Invalid name", "Please enter a project name.", parent=self)
            return
        # Disallow names with path separators or other problematic chars
        if "/" in name or "\\" in name:
            messagebox.showwarning(
                "Invalid name",
                "Project name must not contain '/' or '\\'.",
                parent=self,
            )
            return
        # Check for duplicate before creating
        if (get_projects_dir() / name).exists():
            messagebox.showwarning(
                "Already exists",
                f"A project named '{name}' already exists.",
                parent=self,
            )
            return
        try:
            create_project(name)
        except OSError as exc:
            messagebox.showerror("Error", str(exc), parent=self)
            return

        self._populate_projects()
        # Select the newly created project in the listbox
        items = self._project_listbox.get(0, "end")
        try:
            idx = list(items).index(name)
            self._project_listbox.selection_clear(0, "end")
            self._project_listbox.selection_set(idx)
            self._project_listbox.see(idx)
        except ValueError:
            pass
        self._new_name_var.set("")

    def _on_open(self) -> None:
        name = self._selected_project()
        if name is None:
            messagebox.showinfo("No selection", "Please select a project first.", parent=self)
            return
        self.result = name
        self.grab_release()
        self.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self.grab_release()
        self.destroy()

    # ── Inline rename ───────────────────────────────────────────────

    def _start_inline_rename(self) -> None:
        """Overlay an Entry widget on the selected listbox item for renaming."""
        sel = self._project_listbox.curselection()
        if not sel:
            return
        index = sel[0]
        old_name = self._project_listbox.get(index)

        # Cancel any previous rename entry
        self._cancel_inline_rename()

        # Position the entry over the listbox item
        bbox = self._project_listbox.bbox(index)
        if bbox is None:
            return
        x, y, width, height = bbox

        entry = tk.Entry(self._project_listbox, width=0)
        entry.place(x=x, y=y, width=max(width, self._project_listbox.winfo_width()), height=height)
        entry.insert(0, old_name)
        entry.select_range(0, "end")
        entry.focus_set()

        self._rename_entry = entry
        self._rename_index = index
        self._rename_old_name = old_name

        entry.bind("<Return>", lambda _e: self._finish_inline_rename())
        entry.bind("<Escape>", lambda _e: self._cancel_inline_rename())
        entry.bind("<FocusOut>", lambda _e: self._cancel_inline_rename())

    def _finish_inline_rename(self) -> None:
        if self._rename_entry is None:
            return
        new_name = self._rename_entry.get().strip()
        old_name = self._rename_old_name
        self._rename_entry.destroy()
        self._rename_entry = None

        if not new_name or new_name == old_name:
            return

        try:
            rename_project(old_name, new_name)
        except FileExistsError:
            messagebox.showwarning(
                "Already exists",
                f"A project named '{new_name}' already exists.",
                parent=self,
            )
            return
        except FileNotFoundError:
            messagebox.showerror(
                "Not found",
                f"Project '{old_name}' no longer exists.",
                parent=self,
            )
            return
        except OSError as exc:
            messagebox.showerror("Error", str(exc), parent=self)
            return

        self._populate_projects()
        # Re-select the renamed project
        items = self._project_listbox.get(0, "end")
        try:
            idx = list(items).index(new_name)
            self._project_listbox.selection_clear(0, "end")
            self._project_listbox.selection_set(idx)
            self._project_listbox.see(idx)
        except ValueError:
            pass

    def _cancel_inline_rename(self) -> None:
        if self._rename_entry is not None:
            self._rename_entry.destroy()
            self._rename_entry = None
