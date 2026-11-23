from __future__ import annotations

import logging
import shutil
import tkinter as tk
from collections.abc import Callable
from tkinter import ttk

from flextool.gui.assistant import strings as assistant_strings
from flextool.gui.dialogs import message_box
from flextool.gui.project_utils import (
    create_project,
    get_projects_dir,
    list_projects,
    notify_assistant_project_deleted,
    rename_project,
)

logger = logging.getLogger(__name__)

_DOCS_SHARING = "flextool_gui_interface.md#sharing-a-project"


class ProjectDialog(tk.Toplevel):
    """Modal dialog for creating, selecting, renaming, and deleting projects.

    After the dialog closes:
      * ``self.result`` holds the selected project name (Open clicked) or
        ``None`` if the user cancelled.
      * ``self.deleted_names`` lists projects that were deleted while the
        dialog was open.  The caller can use it to check whether the
        currently-open project was removed.
      * ``self.renamed_current`` is the new name of the currently-open
        project when it was renamed here (``None`` otherwise).

    *before_project_change(name)* is asked before a project is renamed or
    deleted (the AI assistant closes its chat window of that project and lets
    its workers finish); ``False`` refuses the change.

    **Export…** writes the selected project to a ``.flextool.zip`` package
    (refused while *jobs_running(name)* is true); **Import…** unpacks one as a
    new project, which is then opened (``self.result``).
    """

    def __init__(
        self,
        parent: tk.Misc,
        current_project: str | None = None,
        before_project_change: Callable[[str], bool] | None = None,
        jobs_running: Callable[[str], bool] | None = None,
    ) -> None:
        super().__init__(parent)
        self.title("Project menu")
        self.result: str | None = None
        self.deleted_names: list[str] = []
        self.renamed_current: str | None = None
        self._current_project = current_project
        self._before_project_change = before_project_change
        self._jobs_running = jobs_running

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

        self._export_btn = ttk.Button(
            btn_frame, text="Export…", command=self._on_export_clicked, state="disabled",
        )
        self._export_btn.pack(side="left", padx=(10, 0))
        self._import_btn = ttk.Button(btn_frame, text="Import…", command=self._on_import_clicked)
        self._import_btn.pack(side="left", padx=(10, 0))

        self._cancel_btn = ttk.Button(btn_frame, text="Cancel", command=self._on_cancel)
        self._cancel_btn.pack(side="right")

    # ── Helpers ──────────────────────────────────────────────────────

    def _update_ok_state(self) -> None:
        """Enable Open/Delete buttons when a project is selected."""
        state = "normal" if self._project_listbox.curselection() else "disabled"
        for button in (self._open_btn, self._delete_btn, self._export_btn):
            button.configure(state=state)

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

    def _project_change_allowed(self, name: str) -> bool:
        """Ask the AI assistant whether *name* may be renamed or deleted now; a refusal is said."""
        if self._before_project_change is None or self._before_project_change(name):
            return True
        # The assistant's own text (design decision 9): only the title is FlexTool's.
        message_box.tell(self, "Wait for the AI assistant", assistant_strings.PROJECT_STILL_FINISHING)
        return False

    # ── Event handlers ──────────────────────────────────────────────

    def _on_create(self) -> None:
        name = self._new_name_var.get().strip()
        if not name:
            message_box.tell(self, "Type a project name", "Type a name, then press Create.", kind="warning")
            return
        # Disallow names with path separators or other problematic chars
        if "/" in name or "\\" in name:
            message_box.tell(self, "Remove / and \\ from the name", "A project name cannot contain '/' or '\\'.",
                             kind="warning")
            return
        # Check for duplicate before creating
        if (get_projects_dir() / name).exists():
            message_box.tell(self, "Choose another project name", f"A project named '{name}' already exists.",
                             kind="warning")
            return
        try:
            create_project(name)
        except OSError as exc:
            logger.warning("Creating project %r failed", name, exc_info=True)
            message_box.tell(self, "The project was not created", str(exc), kind="error")
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
            message_box.tell(self, "Select a project first", "Click a project in the list, then press Open the selected project.")
            return
        self.result = name
        self.grab_release()
        self.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self.grab_release()
        self.destroy()

    # ── Export / import ─────────────────────────────────────────────

    def _on_export_clicked(self) -> None:
        """Export the selected project to a project package."""
        from flextool.gui.dialogs.project_package_dialog import show_export_dialog

        name = self._selected_project()
        if name is None:
            return
        jobs = None
        if self._jobs_running is not None:
            def jobs() -> bool:
                return self._jobs_running(name)
        show_export_dialog(self, get_projects_dir() / name, jobs_running=jobs)

    def _on_import_clicked(self) -> None:
        """Import a project package as a new project and open it."""
        from flextool import project_package
        from flextool.gui.dialogs.project_package_dialog import ask_import

        report = ask_import(self, get_projects_dir())
        if report is None:
            self._populate_projects()
            return
        review = bool(report.solver_config_held) or any(state == "different" for _n, state in report.solver_config)
        message_box.tell(
            self, "Project imported: review its solver options" if review else "Project imported",
            *project_package.describe_import(report), docs=_DOCS_SHARING,
        )
        self.result = report.name
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
        if not self._project_change_allowed(name):
            return

        project_path = get_projects_dir() / name
        resolved = project_path.resolve()
        try:
            shutil.rmtree(project_path)
            # A project made or imported later at this path must not inherit this one's trust (item 240 B3).
            from flextool._solver_config_source import forget_trusted_folder
            forget_trusted_folder(resolved)
        except OSError as exc:
            logger.warning("Failed to delete project %r", name, exc_info=True)
            message_box.tell(self, "The project was not deleted", str(exc),
                             "Close any program that uses its files and try again.", kind="error")
            return
        notify_assistant_project_deleted(resolved)

        self.deleted_names.append(name)
        self._populate_projects()
        self._update_ok_state()

    def _confirm_delete(self, name: str) -> bool:
        """Ask before the project folder is deleted with everything in it; focus starts on Cancel."""
        answer = message_box.ask(
            self, "Delete this project?",
            f"Project '{name}' is deleted with everything in its folder: the input sources copied into it, the "
            "results, plots and settings. External input sources stay where they are.",
            assistant_strings.PROJECT_DELETE_NOTE,
            "This cannot be undone.",
            buttons=(("Delete", "delete"), ("Cancel", None)), kind="danger",
        )
        return answer == "delete"

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
        if not self._project_change_allowed(old_name):
            return

        try:
            rename_project(old_name, new_name)
        except FileExistsError:
            message_box.tell(self, "Choose another project name", f"A project named '{new_name}' already exists.",
                             kind="warning")
            return
        except FileNotFoundError:
            message_box.tell(self, "This project no longer exists",
                             f"'{old_name}' was moved or deleted outside FlexTool.", kind="error")
            return
        except OSError as exc:
            logger.warning("Renaming project %r failed", old_name, exc_info=True)
            message_box.tell(self, "The project was not renamed", str(exc), kind="error")
            return

        if old_name == self._current_project:
            self._current_project = new_name
            self.renamed_current = new_name
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
