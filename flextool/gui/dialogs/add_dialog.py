from __future__ import annotations

import logging
import shutil
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import messagebox, ttk

from flextool.gui.dialogs.file_picker import FilePickerDialog
from flextool.gui.project_utils import get_projects_dir

logger = logging.getLogger(__name__)

# Default name for the alternative created during old FlexTool import
_OLD_FLEX_ALTERNATIVE = "base"


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
        self.geometry(f"{self._cw * 55}x{lh * 25}")
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

        choose_copy_btn = ttk.Button(
            copy_frame,
            text="Choose and copy files...",
            command=self._on_choose_and_copy,
        )
        choose_copy_btn.pack(fill="x")

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

        # ── Convert from old FlexTool section ─────────────────────────
        old_frame = ttk.LabelFrame(
            self, text="Convert from FlexTool 2.0 input file", padding=8
        )
        old_frame.pack(fill="x", **pad)

        convert_btn = ttk.Button(
            old_frame,
            text="Choose file and convert...",
            command=self._on_convert_old_flextool,
        )
        convert_btn.pack(fill="x")

        # ── Close button (very bottom) ───────────────────────────────
        close_frame = ttk.Frame(self)
        close_frame.pack(fill="x", side="bottom", padx=10, pady=(15, 10))

        close_btn = ttk.Button(
            close_frame, text="Close", command=self._on_back
        )
        close_btn.pack(side="right")

    # ── Event handlers ──────────────────────────────────────────────

    def _on_choose_and_copy(self) -> None:
        """Open a file chooser and immediately copy selected files."""
        initial_dir = self._flextool_root
        if not initial_dir.is_dir():
            initial_dir = Path.home()

        # Determine dialog size from the main window
        try:
            root = self.winfo_toplevel()
            main_window_width = root.winfo_width()
            screen_height = root.winfo_screenheight()
        except Exception:
            main_window_width = 700
            screen_height = 800

        picker = FilePickerDialog(
            self,
            title="Select input source files",
            initialdir=str(initial_dir),
            filetypes=[
                ("FlexTool inputs", "*.xlsx *.ods *.sqlite"),
                ("Excel files", "*.xlsx *.ods"),
                ("SQLite databases", "*.sqlite"),
                ("All files", "*"),
            ],
            multiple=True,
            width=main_window_width,
            height=int(screen_height * 0.75),
        )
        filepaths = picker.result
        if not filepaths:
            return

        # Copy files immediately
        errors: list[str] = []
        copied_names: list[str] = []
        for fp in filepaths:
            fp = Path(fp)
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
                copied_names.append(fp.name)
            except OSError as exc:
                errors.append(f"{fp.name}: {exc}")
                logger.error("Failed to copy %s: %s", fp, exc)

        if errors:
            messagebox.showerror(
                "Copy errors",
                "Some files could not be copied:\n" + "\n".join(errors),
                parent=self,
            )

        # Check and upgrade any copied sqlite databases
        upgrade_messages: list[str] = []
        for fp in filepaths:
            fp = Path(fp)
            if fp.suffix.lower() == ".sqlite":
                dest = self._input_dir / fp.name
                if dest.exists():
                    try:
                        from flextool.gui.db_version_check import (
                            check_and_upgrade_database,
                        )

                        _upgraded, msgs = check_and_upgrade_database(dest)
                        upgrade_messages.extend(msgs)
                    except Exception as exc:
                        upgrade_messages.append(
                            f"{fp.name}: version check error: {exc}"
                        )
                        logger.warning(
                            "Version check failed for %s: %s", dest, exc, exc_info=True
                        )

        if copied_names:
            self.result = True
            file_list = "\n".join(f"  - {name}" for name in copied_names)
            done_text = (
                f"Done — copied {len(copied_names)} file(s):\n{file_list}"
            )
            if upgrade_messages:
                done_text += "\n\n" + "\n".join(upgrade_messages)
            messagebox.showinfo(
                "Done",
                done_text,
                parent=self,
            )

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

    def _on_convert_old_flextool(self) -> None:
        """Open a file chooser for an old FlexTool .xlsm, convert to Spine DB."""
        initial_dir = Path.home()

        # Determine dialog size from the main window (same as choose-and-copy)
        try:
            root = self.winfo_toplevel()
            main_window_width = root.winfo_width()
            screen_height = root.winfo_screenheight()
        except Exception:
            main_window_width = 700
            screen_height = 800

        picker = FilePickerDialog(
            self,
            title="Select old FlexTool input file",
            initialdir=str(initial_dir),
            filetypes=[
                ("Old FlexTool Excel", "*.xlsm *.xlsx"),
                ("All files", "*"),
            ],
            multiple=False,
            width=main_window_width,
            height=int(screen_height * 0.75),
        )
        filepath = picker.result
        if not filepath:
            return

        filepath = Path(filepath)
        dest_name = filepath.stem + ".sqlite"
        dest = self._input_dir / dest_name

        if dest.exists():
            overwrite = messagebox.askyesno(
                "File exists",
                f"'{dest_name}' already exists in input_sources.\n"
                "Do you want to overwrite it?",
                parent=self,
            )
            if not overwrite:
                return
            # Remove the existing file so initialize_database can create fresh
            try:
                dest.unlink()
            except OSError as exc:
                messagebox.showerror(
                    "Error",
                    f"Could not remove existing file:\n{exc}",
                    parent=self,
                )
                return

        # Locate the FlexTool template for creating a new database
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

        # Show busy cursor during conversion
        self.config(cursor="watch")
        self.update()

        try:
            # Step 1: Create empty Spine DB from template
            from flextool.update_flextool.initialize_database import (
                initialize_database,
            )
            initialize_database(str(json_template), str(dest))

            # Step 2: Read old FlexTool data
            from flextool.process_inputs.read_old_flextool import (
                read_old_flextool,
            )
            data = read_old_flextool(str(filepath))

            # Step 3: Write to the new database
            from flextool.process_inputs.write_old_flextool_to_db import (
                write_old_flextool_to_db,
            )
            db_url = f"sqlite:///{dest}"
            write_old_flextool_to_db(
                data, db_url, alternative_name=_OLD_FLEX_ALTERNATIVE
            )

        except ImportError as exc:
            messagebox.showerror(
                "Missing dependency",
                f"A required package is not installed:\n{exc}",
                parent=self,
            )
            # Clean up partial database
            if dest.exists():
                dest.unlink(missing_ok=True)
            return
        except Exception as exc:
            logger.error("Old FlexTool conversion failed: %s", exc, exc_info=True)
            messagebox.showerror(
                "Conversion failed",
                f"Failed to convert '{filepath.name}':\n{exc}",
                parent=self,
            )
            # Clean up partial database
            if dest.exists():
                dest.unlink(missing_ok=True)
            return
        finally:
            self.config(cursor="")

        self.result = True
        messagebox.showinfo(
            "Done",
            f"Converted '{filepath.name}' → '{dest_name}' in input_sources.",
            parent=self,
        )

    def _on_back(self) -> None:
        """Close the dialog."""
        self.grab_release()
        self.destroy()
