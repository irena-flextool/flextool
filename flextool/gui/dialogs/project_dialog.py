from __future__ import annotations

import logging
import shutil
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, messagebox

from flextool.gui.project_utils import (
    create_project,
    get_projects_dir,
    list_projects,
    rename_project,
)

logger = logging.getLogger(__name__)


class ProjectDialog(tk.Toplevel):
    """Modal dialog for creating, selecting, renaming, and deleting projects.

    After the dialog closes:
      * ``self.result`` holds the selected project name (Open clicked) or
        ``None`` if the user cancelled.
      * ``self.deleted_names`` lists projects that were deleted while the
        dialog was open.  The caller can use it to check whether the
        currently-open project was removed.
    """

    def __init__(self, parent: tk.Misc, current_project: str | None = None) -> None:
        super().__init__(parent)
        self.title("Project menu")
        self.result: str | None = None
        self.deleted_names: list[str] = []
        self._current_project = current_project

        # ── Modal behaviour ─────────────────────────────────────────
        self.transient(parent)
        self.grab_set()

        # ── Font metrics for DPI-aware sizing ──────────────────────
        from flextool.gui.ui_metrics import get_metrics
        _metrics = get_metrics(self)
        cw: int = _metrics.cw
        lh: int = _metrics.lh

        # ── Dialog size ─────────────────────────────────────────────
        self.geometry(f"{cw * 70}x{lh * 30}")
        self.minsize(cw * 50, lh * 20)

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
        new_frame = ttk.LabelFrame(self, text="Create an empty project", padding=8)
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

        # Enable/disable Ok when listbox selection changes
        self._project_listbox.bind("<<ListboxSelect>>", lambda _e: self._update_ok_state())

        # -- Bottom buttons --
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", **pad)

        self._open_btn = ttk.Button(btn_frame, text="Open the selected project", command=self._on_open, state="disabled")
        self._open_btn.pack(side="left")

        self._delete_btn = ttk.Button(
            btn_frame,
            text="Delete",
            command=self._on_delete_clicked,
            state="disabled",
        )
        self._delete_btn.pack(side="left", padx=(10, 0))

        self._cancel_btn = ttk.Button(btn_frame, text="Cancel", command=self._on_cancel)
        self._cancel_btn.pack(side="right")

    # ── Helpers ──────────────────────────────────────────────────────

    def _update_ok_state(self) -> None:
        """Enable Open/Delete buttons when a project is selected."""
        if self._project_listbox.curselection():
            self._open_btn.configure(state="normal")
            self._delete_btn.configure(state="normal")
        else:
            self._open_btn.configure(state="disabled")
            self._delete_btn.configure(state="disabled")

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
        self._update_ok_state()

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

    # ── Delete project ──────────────────────────────────────────────

    def _on_delete_clicked(self) -> None:
        """Confirm and recursively delete the selected project directory."""
        name = self._selected_project()
        if name is None:
            return
        if not self._confirm_delete(name):
            return

        project_path = get_projects_dir() / name
        try:
            shutil.rmtree(project_path)
        except OSError as exc:
            logger.warning("Failed to delete project %r", name, exc_info=True)
            messagebox.showerror(
                "Delete failed",
                f"Could not delete project '{name}':\n\n{exc}",
                parent=self,
            )
            return

        self.deleted_names.append(name)
        self._populate_projects()
        self._update_ok_state()

    def _confirm_delete(self, name: str) -> bool:
        """Show a custom confirmation dialog; return True if user confirms.

        The confirmation uses a capital-letters warning line and a
        regular-text explanation that the deletion is recursive
        (including any imported input databases).  Default focus is
        Cancel so an accidental Enter doesn't trigger deletion.
        """
        dlg = tk.Toplevel(self)
        dlg.title("Delete project")
        dlg.transient(self)
        dlg.grab_set()
        dlg.resizable(False, False)

        # Sizing — match the parent dialog's DPI-aware metric helper.
        from flextool.gui.ui_metrics import get_metrics
        _m = get_metrics(dlg)
        cw, lh = _m.cw, _m.lh
        wrap_px = cw * 60

        confirmed = {"value": False}

        body = ttk.Frame(dlg, padding=(cw * 2, lh))
        body.pack(fill="both", expand=True)

        warning_font = tkfont.Font(font="TkDefaultFont")
        warning_font.configure(weight="bold", size=warning_font.cget("size") + 2)

        ttk.Label(
            body,
            text="ALL PROJECT DATA WILL BE PERMANENTLY DELETED",
            font=warning_font,
            foreground="#b00020",
            wraplength=wrap_px,
            justify="left",
        ).pack(anchor="w", pady=(0, lh))

        ttk.Label(
            body,
            text=(
                "This includes input databases (Excel, SQLite) that were "
                "copied or imported into the project folder, as well as "
                "all output results, plots, settings, and any other files "
                "under the project directory."
            ),
            wraplength=wrap_px,
            justify="left",
        ).pack(anchor="w")

        ttk.Label(
            body,
            text=f"Project: {name}",
            wraplength=wrap_px,
            justify="left",
        ).pack(anchor="w", pady=(lh, 0))

        # -- Buttons --
        btn_row = ttk.Frame(dlg, padding=(cw * 2, 0, cw * 2, lh))
        btn_row.pack(fill="x")

        def _do_delete() -> None:
            confirmed["value"] = True
            dlg.grab_release()
            dlg.destroy()

        def _do_cancel() -> None:
            confirmed["value"] = False
            dlg.grab_release()
            dlg.destroy()

        # Try to use a danger style if the theme defines one; fall back
        # silently to the default button style.
        delete_btn = ttk.Button(btn_row, text="Delete", command=_do_delete)
        try:
            ttk.Style().configure("Danger.TButton", foreground="#b00020")
            delete_btn.configure(style="Danger.TButton")
        except tk.TclError:
            pass
        delete_btn.pack(side="left")

        cancel_btn = ttk.Button(btn_row, text="Cancel", command=_do_cancel)
        cancel_btn.pack(side="right")

        dlg.protocol("WM_DELETE_WINDOW", _do_cancel)
        dlg.bind("<Escape>", lambda _e: _do_cancel())
        # Default focus = Cancel (defensive — Enter should not delete).
        cancel_btn.focus_set()

        # Centre on parent dialog.
        dlg.update_idletasks()
        px = self.winfo_rootx()
        py = self.winfo_rooty()
        pw = self.winfo_width()
        ph = self.winfo_height()
        w = dlg.winfo_width()
        h = dlg.winfo_height()
        x = px + (pw - w) // 2
        y = py + (ph - h) // 2
        dlg.geometry(f"+{x}+{y}")

        self.wait_window(dlg)
        return confirmed["value"]

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
