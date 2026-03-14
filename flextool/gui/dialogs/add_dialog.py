from __future__ import annotations

import logging
import shutil
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from flextool.gui.project_utils import get_projects_dir

logger = logging.getLogger(__name__)


class AddDialog(tk.Toplevel):
    """Modal dialog for adding input source files to a project.

    After the dialog closes, ``self.result`` is ``True`` if any files were
    added (so the caller knows to refresh the input sources list).
    """

    def __init__(self, parent: tk.Misc, project_path: Path) -> None:
        super().__init__(parent)
        self.title("Add input sources")
        self.result: bool = False
        self._project_path = project_path
        self._input_dir = project_path / "input_sources"
        self._input_dir.mkdir(parents=True, exist_ok=True)

        # Root directories for templates
        self._flextool_root = get_projects_dir().parent

        # ── Modal behaviour ─────────────────────────────────────────
        self.transient(parent)
        self.grab_set()

        # ── Font metrics for DPI-aware sizing ──────────────────────
        default_font = tkfont.nametofont("TkDefaultFont")
        self._cw: int = default_font.measure("0")
        lh: int = default_font.metrics("linespace")

        # ── Dialog size ─────────────────────────────────────────────
        self.geometry(f"{self._cw * 55}x{lh * 20}")
        self.resizable(False, False)

        self._build_widgets()

        # Close via window-manager "X"
        self.protocol("WM_DELETE_WINDOW", self._on_back)

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

        # ── Copy to project section ─────────────────────────────────
        copy_frame = ttk.LabelFrame(self, text="Copy to project", padding=8)
        copy_frame.pack(fill="x", **pad)

        self._selected_files_var = tk.StringVar(value="No files selected")
        choose_btn = ttk.Button(
            copy_frame, text="Choose file dialog...", command=self._on_choose_files
        )
        choose_btn.pack(fill="x", pady=(0, 4))

        self._files_label = ttk.Label(
            copy_frame,
            textvariable=self._selected_files_var,
            wraplength=self._cw * 50,
            justify="left",
        )
        self._files_label.pack(fill="x", pady=(0, 4))

        self._copy_btn = ttk.Button(
            copy_frame, text="Copy", command=self._on_copy
        )
        self._copy_btn.pack(anchor="e")

        self._selected_file_paths: list[str] = []

        # ── Add FlexTool input Excel section ────────────────────────
        xlsx_frame = ttk.LabelFrame(
            self, text="Add FlexTool input Excel", padding=8
        )
        xlsx_frame.pack(fill="x", **pad)

        xlsx_row = ttk.Frame(xlsx_frame)
        xlsx_row.pack(fill="x")

        ttk.Label(xlsx_row, text="Name:").pack(side="left")
        self._xlsx_name_var = tk.StringVar(value="input")
        xlsx_entry = ttk.Entry(
            xlsx_row, textvariable=self._xlsx_name_var, width=25
        )
        xlsx_entry.pack(side="left", padx=(5, 5))
        ttk.Label(xlsx_row, text=".xlsx").pack(side="left")

        xlsx_add_btn = ttk.Button(
            xlsx_row, text="Add", command=self._on_add_xlsx
        )
        xlsx_add_btn.pack(side="right")

        # ── Add FlexTool input database section ─────────────────────
        sqlite_frame = ttk.LabelFrame(
            self, text="Add FlexTool input database", padding=8
        )
        sqlite_frame.pack(fill="x", **pad)

        sqlite_row = ttk.Frame(sqlite_frame)
        sqlite_row.pack(fill="x")

        ttk.Label(sqlite_row, text="Name:").pack(side="left")
        self._sqlite_name_var = tk.StringVar(value="input")
        sqlite_entry = ttk.Entry(
            sqlite_row, textvariable=self._sqlite_name_var, width=25
        )
        sqlite_entry.pack(side="left", padx=(5, 5))
        ttk.Label(sqlite_row, text=".sqlite").pack(side="left")

        sqlite_add_btn = ttk.Button(
            sqlite_row, text="Add", command=self._on_add_sqlite
        )
        sqlite_add_btn.pack(side="right")

        # ── Close button (bottom right) ──────────────────────────────
        close_frame = ttk.Frame(self)
        close_frame.pack(fill="x", padx=10, pady=(10, 10))

        close_btn = ttk.Button(
            close_frame, text="Close", command=self._on_back
        )
        close_btn.pack(side="right")

    # ── Event handlers ──────────────────────────────────────────────

    def _on_choose_files(self) -> None:
        """Open a file chooser for xlsx/sqlite files."""
        initial_dir = self._flextool_root
        if not initial_dir.is_dir():
            initial_dir = Path.home()

        filepaths = filedialog.askopenfilenames(
            parent=self,
            title="Select input source files",
            initialdir=str(initial_dir),
            filetypes=[
                ("FlexTool inputs", "*.xlsx *.ods *.sqlite"),
                ("Excel files", "*.xlsx *.ods"),
                ("SQLite databases", "*.sqlite"),
                ("All files", "*.*"),
            ],
        )
        if filepaths:
            self._selected_file_paths = list(filepaths)
            names = [Path(fp).name for fp in filepaths]
            self._selected_files_var.set(", ".join(names))
        else:
            self._selected_file_paths = []
            self._selected_files_var.set("No files selected")

    def _on_copy(self) -> None:
        """Copy selected files to the input_sources directory."""
        if not self._selected_file_paths:
            messagebox.showinfo(
                "No files",
                "Please choose files first.",
                parent=self,
            )
            return

        errors: list[str] = []
        copied = 0
        for fp_str in self._selected_file_paths:
            fp = Path(fp_str)
            dest = self._input_dir / fp.name
            if dest.exists():
                overwrite = messagebox.askyesno(
                    "File exists",
                    f"'{fp.name}' already exists in input_sources.\n"
                    "Do you want to overwrite it?",
                    parent=self,
                )
                if not overwrite:
                    continue
            try:
                shutil.copy2(str(fp), str(dest))
                copied += 1
            except OSError as exc:
                errors.append(f"{fp.name}: {exc}")
                logger.error("Failed to copy %s: %s", fp, exc)

        if errors:
            messagebox.showerror(
                "Copy errors",
                "Some files could not be copied:\n" + "\n".join(errors),
                parent=self,
            )

        if copied > 0:
            self.result = True
            messagebox.showinfo(
                "Done",
                f"Copied {copied} file(s) to input_sources.",
                parent=self,
            )

        # Reset selection
        self._selected_file_paths = []
        self._selected_files_var.set("No files selected")

    def _on_add_xlsx(self) -> None:
        """Copy the example xlsx template to input_sources."""
        name = self._xlsx_name_var.get().strip()
        if not name:
            messagebox.showwarning(
                "Invalid name", "Please enter a file name.", parent=self
            )
            return

        filename = f"{name}.xlsx"
        dest = self._input_dir / filename
        if dest.exists():
            messagebox.showwarning(
                "Already exists",
                f"'{filename}' already exists in input_sources.",
                parent=self,
            )
            return

        template = self._flextool_root / "templates" / "example_input_template.xlsx"
        if not template.exists():
            messagebox.showerror(
                "Template missing",
                f"Cannot find template:\n{template}",
                parent=self,
            )
            return

        try:
            shutil.copy2(str(template), str(dest))
        except OSError as exc:
            messagebox.showerror("Error", str(exc), parent=self)
            return

        self.result = True
        messagebox.showinfo(
            "Done",
            f"Created '{filename}' in input_sources.",
            parent=self,
        )

    def _on_add_sqlite(self) -> None:
        """Create a new Spine database from the master template."""
        name = self._sqlite_name_var.get().strip()
        if not name:
            messagebox.showwarning(
                "Invalid name", "Please enter a file name.", parent=self
            )
            return

        filename = f"{name}.sqlite"
        dest = self._input_dir / filename
        if dest.exists():
            messagebox.showwarning(
                "Already exists",
                f"'{filename}' already exists in input_sources.",
                parent=self,
            )
            return

        json_template = (
            self._flextool_root / "version" / "flextool_template_master.json"
        )
        if not json_template.exists():
            messagebox.showerror(
                "Template missing",
                f"Cannot find template:\n{json_template}",
                parent=self,
            )
            return

        try:
            from flextool.update_flextool.initialize_database import (
                initialize_database,
            )

            initialize_database(str(json_template), str(dest))
        except ImportError:
            messagebox.showerror(
                "Missing dependency",
                "spinedb_api is required to create SQLite databases.",
                parent=self,
            )
            return
        except Exception as exc:
            messagebox.showerror("Error", str(exc), parent=self)
            return

        self.result = True
        messagebox.showinfo(
            "Done",
            f"Created '{filename}' in input_sources.",
            parent=self,
        )

    def _on_back(self) -> None:
        """Close the dialog."""
        self.grab_release()
        self.destroy()
