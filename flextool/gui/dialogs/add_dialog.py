from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import messagebox, ttk

from flextool.gui.dialogs.file_picker import FilePickerDialog
from flextool.gui.project_utils import get_projects_dir

logger = logging.getLogger(__name__)

# Default name for the alternative created during old FlexTool import
_OLD_FLEX_ALTERNATIVE = "base"


def _ask_migration_choice(
    parent: tk.Misc,
    filename: str,
    version_str: str,
    current_version: int,
) -> str:
    """Ask how to handle an older Excel file that needs migration.

    Returns:
        ``"sqlite"`` to convert to Spine DB, ``"excel"`` to migrate and
        keep as Excel, or ``"cancel"`` to do nothing.
    """
    dlg = tk.Toplevel(parent)
    dlg.title("Version migration needed")
    dlg.transient(parent)
    dlg.grab_set()
    dlg.resizable(False, False)

    result = "cancel"

    msg = (
        f"'{filename}' is version {version_str} "
        f"(current is {current_version}) and needs to be migrated.\n\n"
        f"Choose how to handle the update:"
    )
    lbl = ttk.Label(dlg, text=msg, wraplength=420, justify="left")
    lbl.pack(padx=16, pady=(16, 8))

    btn_frame = ttk.Frame(dlg)
    btn_frame.pack(padx=16, pady=(4, 16))

    def _choose(choice: str) -> None:
        nonlocal result
        result = choice
        dlg.destroy()

    ttk.Button(
        btn_frame, text="Convert to Spine DB",
        command=lambda: _choose("sqlite"),
    ).pack(side="left", padx=4)
    ttk.Button(
        btn_frame, text="Update Excel",
        command=lambda: _choose("excel"),
    ).pack(side="left", padx=4)
    ttk.Button(
        btn_frame, text="Cancel",
        command=lambda: _choose("cancel"),
    ).pack(side="left", padx=4)

    dlg.protocol("WM_DELETE_WINDOW", lambda: _choose("cancel"))

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


class AddDialog(tk.Toplevel):
    """Modal dialog for adding input source files to a project.

    After the dialog closes, ``self.result`` is ``True`` if any files were
    added (so the caller knows to refresh the input sources list).
    """

    def __init__(
        self, parent: tk.Misc, project_path: Path,
        execution_mgr=None,
    ) -> None:
        super().__init__(parent)
        self.title("Add input sources")
        self.result: bool = False
        self.old_convert_started: bool = False  # True if old FlexTool conversion was started
        self.files_to_convert: list[str] = []  # Excel files to convert to sqlite
        self.files_to_update_xlsx: list[str] = []  # Excel files to round-trip (migrate and keep as xlsx)
        self._project_path = project_path
        self._input_dir = project_path / "input_sources"
        self._input_dir.mkdir(parents=True, exist_ok=True)
        self._execution_mgr = execution_mgr

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

        self._import_sens_btn = ttk.Button(
            old_frame,
            text="Import sensitivities from master file...",
            command=self._on_import_sensitivities,
            state="disabled",
        )
        self._import_sens_btn.pack(fill="x", pady=(5, 0))

        # Stored after successful 2.0 base conversion
        self._last_base_xlsm: Path | None = None
        self._last_target_sqlite: Path | None = None

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
                ("FlexTool inputs", "*.xlsx *.xlsm *.ods *.sqlite"),
                ("Excel files", "*.xlsx *.xlsm *.ods"),
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

        # Validate and copy files
        from flextool.process_inputs import (
            detect_excel_format, ExcelFormat, CURRENT_FLEXTOOL_DB_VERSION,
        )

        errors: list[str] = []
        copied_names: list[str] = []
        for fp in filepaths:
            fp = Path(fp)

            # Validate Excel files before copying
            needs_migration = False
            migration_choice: str | None = None
            if fp.suffix.lower() in (".xlsx", ".xlsm", ".ods"):
                info = detect_excel_format(fp)

                if info.format == ExcelFormat.OLD_V2:
                    messagebox.showwarning(
                        "FlexTool 2.0 file",
                        f"'{fp.name}' is a FlexTool 2.0 file.\n\n"
                        "Use 'Convert from FlexTool 2.0 input file' instead.",
                        parent=self,
                    )
                    continue

                # Check whether the file needs migration
                needs_migration = (
                    info.format == ExcelFormat.SPECIFICATION
                    or (
                        info.format == ExcelFormat.SELF_DESCRIBING
                        and info.version is not None
                        and info.version < CURRENT_FLEXTOOL_DB_VERSION
                    )
                )

                if info.format == ExcelFormat.UNKNOWN:
                    messagebox.showwarning(
                        "Unrecognised format",
                        f"'{fp.name}' does not appear to be a valid FlexTool "
                        f"input file (no scenario sheet found).",
                        parent=self,
                    )
                    continue

                # For older files, ask how to handle migration before copying
                if needs_migration:
                    version_str = str(info.version) if info.version is not None else "unknown"
                    migration_choice = _ask_migration_choice(
                        self,
                        fp.name,
                        version_str,
                        CURRENT_FLEXTOOL_DB_VERSION,
                    )
                    if migration_choice == "cancel":
                        continue

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
                continue

            if migration_choice == "sqlite":
                self.files_to_convert.append(fp.name)
            elif migration_choice == "excel":
                self.files_to_update_xlsx.append(fp.name)

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

        # Files undergoing conversion are excluded from the "Done"
        # notification — they will appear after conversion finishes.
        migrating = set(self.files_to_convert) | set(self.files_to_update_xlsx)
        ready_names = [n for n in copied_names if n not in migrating]

        if copied_names:
            self.result = True
        if ready_names:
            file_list = "\n".join(f"  - {name}" for name in ready_names)
            done_text = (
                f"Done — copied {len(ready_names)} file(s):\n{file_list}"
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
        """Open a file chooser for an old FlexTool .xlsm, convert to Spine DB.

        The conversion runs as an auxiliary job in the ExecutionManager.
        The dialog closes immediately and progress is shown in the
        execution window.
        """
        initial_dir = self._flextool_root

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
            try:
                dest.unlink()
            except OSError as exc:
                messagebox.showerror(
                    "Error",
                    f"Could not remove existing file:\n{exc}",
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

        # Create empty Spine DB from template
        try:
            from flextool.update_flextool.initialize_database import (
                initialize_database,
            )
            initialize_database(str(json_template), str(dest))
        except Exception as exc:
            messagebox.showerror(
                "Initialisation failed",
                f"Could not create target database:\n{exc}",
                parent=self,
            )
            if dest.exists():
                dest.unlink(missing_ok=True)
            return

        # Build conversion command
        db_url = f"sqlite:///{dest}"
        cmd = [
            sys.executable, "-m",
            "flextool.cli.cmd_read_old_flextool",
            str(filepath),
            db_url,
        ]

        if self._execution_mgr is not None:
            # Run as auxiliary job — close dialog, show in execution window
            from flextool.gui.execution_manager import JobType

            job = self._execution_mgr.add_auxiliary_job(
                JobType.OLD_CONVERT,
                f"Convert FlexTool 2.0: {filepath.name}",
                f"old_convert:{filepath.name}",
            )
            self._execution_mgr.append_stdout(
                job.job_id, f"Converting '{filepath.name}' \u2192 '{dest_name}'"
            )
            self._execution_mgr.append_stdout(job.job_id, " ".join(cmd))
            self._execution_mgr.append_stdout(job.job_id, "")

            mgr = self._execution_mgr  # capture for thread

            def _worker() -> None:
                import os as _os
                success = False
                try:
                    env = {**_os.environ, "PYTHONUNBUFFERED": "1"}
                    proc = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1,
                        cwd=str(self._flextool_root), env=env,
                    )
                    with mgr._lock:
                        job.process = proc

                    for line in proc.stdout:  # type: ignore[union-attr]
                        mgr.append_stdout(job.job_id, line.rstrip("\n"))

                    proc.wait()
                    success = proc.returncode == 0
                    if not success:
                        mgr.append_stdout(
                            job.job_id,
                            f"\nConversion failed (exit code {proc.returncode}).",
                        )
                        if dest.exists():
                            dest.unlink(missing_ok=True)
                except Exception as exc:
                    logger.error("Old FlexTool conversion failed: %s", exc, exc_info=True)
                    mgr.append_stdout(job.job_id, f"\nError: {exc}")
                    if dest.exists():
                        dest.unlink(missing_ok=True)

                mgr.finish_job(job.job_id, success)
                if success:
                    try:
                        main_window.after(0, main_window._refresh_input_sources)
                    except Exception:
                        pass

            main_window = self.master
            threading.Thread(target=_worker, daemon=True).start()

            self.result = True
            self.old_convert_started = True
            # Store paths so sensitivity import can find them
            self._last_base_xlsm = filepath
            self._last_target_sqlite = dest
            self._import_sens_btn.configure(state="normal")
        else:
            # Fallback: no execution manager — run synchronously with busy cursor
            self.config(cursor="watch")
            self.update()
            try:
                from flextool.process_inputs.read_old_flextool import read_old_flextool
                from flextool.process_inputs.write_old_flextool_to_db import write_old_flextool_to_db

                data = read_old_flextool(str(filepath))
                write_old_flextool_to_db(data, db_url, alternative_name=_OLD_FLEX_ALTERNATIVE)
            except Exception as exc:
                logger.error("Old FlexTool conversion failed: %s", exc, exc_info=True)
                messagebox.showerror("Conversion failed", str(exc), parent=self)
                if dest.exists():
                    dest.unlink(missing_ok=True)
                return
            finally:
                self.config(cursor="")

            self.result = True
            self._last_base_xlsm = filepath
            self._last_target_sqlite = dest
            self._import_sens_btn.configure(state="normal")
            messagebox.showinfo(
                "Done",
                f"Converted '{filepath.name}' \u2192 '{dest_name}' in input_sources.\n\n"
                "You can now import sensitivities from a master file, "
                "or close this dialog.",
                parent=self,
            )

    def _on_import_sensitivities(self) -> None:
        """Import sensitivities from a FlexTool 2.0 master file into the
        database that was just created by the base conversion."""
        if self._last_base_xlsm is None or self._last_target_sqlite is None:
            return

        try:
            root = self.winfo_toplevel()
            main_window_width = root.winfo_width()
            screen_height = root.winfo_screenheight()
        except Exception:
            main_window_width = 700
            screen_height = 800

        # Let user pick the master file (start in same directory as base)
        picker = FilePickerDialog(
            self,
            title="Select FlexTool 2.0 master file (with sensitivity definitions)",
            initialdir=str(self._last_base_xlsm.parent),
            filetypes=[
                ("Old FlexTool Excel", "*.xlsm *.xlsx"),
                ("All files", "*"),
            ],
            multiple=False,
            width=main_window_width,
            height=int(screen_height * 0.75),
        )
        master_path = picker.result
        if not master_path:
            return

        target_db_url = f"sqlite:///{self._last_target_sqlite}"
        cmd = [
            sys.executable, "-m",
            "flextool.cli.cmd_import_sensitivities",
            str(master_path),
            str(self._last_base_xlsm),
            target_db_url,
        ]

        if self._execution_mgr is not None:
            from flextool.gui.execution_manager import JobType

            job = self._execution_mgr.add_auxiliary_job(
                JobType.CONVERSION,
                f"Import sensitivities → '{self._last_target_sqlite.name}'",
                f"import_sensitivities:{self._last_target_sqlite.name}",
            )
            self._execution_mgr.append_stdout(
                job.job_id,
                f"Importing sensitivities from '{Path(master_path).name}'\n",
            )
            self._execution_mgr.append_stdout(job.job_id, " ".join(cmd))
            self._execution_mgr.append_stdout(job.job_id, "")

            mgr = self._execution_mgr

            def _worker() -> None:
                import os as _os
                success = False
                try:
                    env = {**_os.environ, "PYTHONUNBUFFERED": "1"}
                    proc = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1,
                        cwd=str(self._flextool_root), env=env,
                    )
                    with mgr._lock:
                        job.process = proc

                    for line in proc.stdout:  # type: ignore[union-attr]
                        mgr.append_stdout(job.job_id, line.rstrip("\n"))

                    proc.wait()
                    success = proc.returncode == 0
                    if success:
                        mgr.append_stdout(job.job_id, "\nSensitivity import succeeded.")
                    else:
                        mgr.append_stdout(
                            job.job_id,
                            f"\nSensitivity import failed (exit code {proc.returncode}).",
                        )
                except Exception as exc:
                    logger.error("Sensitivity import failed: %s", exc, exc_info=True)
                    mgr.append_stdout(job.job_id, f"\nError: {exc}")

                mgr.finish_job(job.job_id, success)
                # Refresh the main window so new scenarios appear checked
                if success:
                    try:
                        main_window.after(0, main_window._refresh_and_check_new_scenarios)
                    except Exception:
                        pass  # Main window may be gone

            main_window = self.master  # capture before thread starts
            threading.Thread(target=_worker, daemon=True).start()
            self._import_sens_btn.configure(state="disabled")
            self.result = True
        else:
            # Fallback: run synchronously
            self.config(cursor="watch")
            self.update()
            try:
                from flextool.process_inputs.read_old_flextool import (
                    read_old_flextool, read_old_flextool_sensitivities,
                )
                from flextool.process_inputs.write_old_flextool_to_db import (
                    write_sensitivities_to_db,
                )

                data = read_old_flextool(str(self._last_base_xlsm))
                sensitivities = read_old_flextool_sensitivities(str(master_path))
                write_sensitivities_to_db(
                    sensitivities, data, target_db_url,
                    base_alternative=_OLD_FLEX_ALTERNATIVE,
                )
            except Exception as exc:
                logger.error("Sensitivity import failed: %s", exc, exc_info=True)
                messagebox.showerror("Import failed", str(exc), parent=self)
                return
            finally:
                self.config(cursor="")

            self._import_sens_btn.configure(state="disabled")
            self.result = True
            messagebox.showinfo(
                "Done",
                f"Sensitivities imported into '{self._last_target_sqlite.name}'.",
                parent=self,
            )

    def _on_back(self) -> None:
        """Close the dialog."""
        self.grab_release()
        self.destroy()
