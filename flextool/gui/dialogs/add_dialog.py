from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from flextool._child_python import SAFE_PATH_FLAG, child_env
from flextool._resources import package_data_path
from flextool.gui.dialogs import message_box
from flextool.gui.dialogs.file_picker import FilePickerDialog
from flextool.gui.project_utils import get_projects_dir
from flextool.sqlite_url import sqlite_url_string

logger = logging.getLogger(__name__)

# Default name for the alternative created during old FlexTool import
_OLD_FLEX_ALTERNATIVE = "base"

_DOCS_CONVERT_FT2 = "flextool_gui_interface.md#convert-a-flextool-20-model"
_DOCS_SOURCES = "flextool_gui_interface.md#input-sources"
_DOCS_EXCEL = "excel_interface.md"
_DOCS_TROUBLESHOOTING = "install_gui.md#troubleshooting"


def _older_version_text(filename: str, version: int | None, current_version: int) -> str:
    """"'F' is version V; this FlexTool uses version C." (an unknown version: "an older input layout")."""
    old = f"is version {version}" if version is not None else "uses an older input layout"
    return f"'{filename}' {old}; this FlexTool uses version {current_version}."


def _tell_ft2_file(parent: tk.Misc, filename: str) -> None:
    """#65: a FlexTool 2.0 workbook chosen for copy or reference."""
    message_box.tell(
        parent, "Use Convert for FlexTool 2.0 files",
        f"'{filename}' is a FlexTool 2.0 file. Add it with 'Convert from FlexTool 2.0 input file' instead.",
        kind="warning", docs=_DOCS_CONVERT_FT2,
    )


def _tell_unknown_excel(parent: tk.Misc, filename: str) -> None:
    """#66: an Excel file that detect_excel_format cannot place (it does not open, or has no FlexTool version)."""
    message_box.tell(
        parent, "Choose a FlexTool input file",
        f"'{filename}' does not look like a FlexTool input file: it does not open, or it has no FlexTool version "
        "information. Check that you chose the right file.",
        kind="warning", docs=_DOCS_SOURCES,
    )


def _ask_replace(parent: tk.Misc, filename: str, when: str = "Replacing it deletes the old one.") -> bool:
    """#67: True when the user agrees to replace *filename* in input_sources."""
    return message_box.ask(
        parent, "Replace the existing file?",
        f"'{filename}' is already in this project's input sources. {when}",
        buttons=(("Replace", "replace"), ("Cancel", None)), kind="danger",
    ) == "replace"


def _tell_added(parent: tk.Misc, names: list[str], one: str, several: str) -> None:
    """#69: the files just added; *one* takes the name, *several* comes before the list."""
    if len(names) == 1:
        message_box.tell(parent, "File added", one.format(names[0]))
    else:
        message_box.tell(parent, "Files added", several, message_box.name_list(names))


def _check_new_name(parent: tk.Misc, name: str, dest: Path) -> bool:
    """#72 / #73: True when *name* is given and *dest* is free."""
    if not name:
        message_box.tell(parent, "Type a file name", "Type a name for the new file, then try again.", kind="warning")
        return False
    if dest.exists():
        message_box.tell(
            parent, "Choose another file name", f"'{dest.name}' is already in this project's input sources.",
            kind="warning",
        )
        return False
    return True


def _tell_template_missing(parent: tk.Misc, path: Path) -> None:
    """#74 (as #41)."""
    message_box.tell(
        parent, "FlexTool's template file is missing",
        f"FlexTool could not find {path}. The installation may be incomplete: update or reinstall FlexTool.",
        kind="error", docs=_DOCS_TROUBLESHOOTING,
    )


def _tell_create_failed(parent: tk.Misc, exc: Exception) -> None:
    """#75 / #76: the new empty input file was not created (the error is logged with its traceback)."""
    if isinstance(exc, ImportError):
        message_box.tell(
            parent, "A package FlexTool needs is missing", str(exc),
            "The installation may be incomplete: update or reinstall FlexTool.",
            kind="error", docs=_DOCS_TROUBLESHOOTING,
        )
        return
    logger.warning("Could not create the input file: %s", exc, exc_info=True)
    message_box.tell(
        parent, "The file was not created", str(exc), "The full error is in the FlexTool log.", kind="error",
    )


def ask_old_flextool_master(
    parent: tk.Misc, initial_dir: Path, width: int, screen_height: int,
) -> Path | None | bool:
    """Ask for the optional FlexTool 2.0 master workbook of a conversion.

    Returns the chosen path, ``None`` for no sensitivities, or ``False`` when
    the user cancelled (the conversion should not start).
    """
    while True:
        answer = message_box.ask(
            parent, "Import the sensitivities too?",
            "FlexTool 2.0 keeps the sensitivities in the master workbook, the one with the 'Sensitivity "
            "definitions' sheet. FlexTool can import them as alternatives and scenarios; you choose the master "
            "workbook next.",
            buttons=(("Import them", "import"), ("Skip them", "skip"), ("Cancel", None)),
            docs=_DOCS_CONVERT_FT2,
        )
        if answer is None:
            return False
        if answer == "skip":
            return None
        picker = FilePickerDialog(
            parent,
            title="Select FlexTool 2.0 master workbook (with sensitivity definitions)",
            initialdir=str(initial_dir),
            filetypes=[
                ("Old FlexTool Excel", "*.xlsm *.xlsx"),
                ("All files", "*"),
            ],
            multiple=False,
            width=width,
            height=int(screen_height * 0.75),
        )
        if picker.result:
            return Path(picker.result)
        # No file chosen: ask again (Import / Skip / Cancel), so a cancelled
        # picker never aborts the conversion without a word.


def _ask_migration_choice(
    parent: tk.Misc,
    filename: str,
    version: int | None,
    current_version: int,
) -> str:
    """#64: ask how to handle an older Excel file that is being copied into the project.

    Returns:
        ``"sqlite"`` to convert to Spine DB, ``"excel"`` to migrate and
        keep as Excel, or ``"cancel"`` to do nothing (Cancel, Escape and the window button).
    """
    answer = message_box.ask(
        parent, "Upgrade this Excel file",
        _older_version_text(filename, version, current_version),
        "Convert it to a database (recommended: the database editor can open it), or update the Excel file. "
        "This changes the copy in the project; the file you chose stays as it is.",
        buttons=(("Convert to database", "sqlite"), ("Update Excel", "excel"), ("Cancel", None)),
        docs=_DOCS_EXCEL,
    )
    return answer if answer is not None else "cancel"


class AddDialog(tk.Toplevel):
    """Modal dialog for adding input source files to a project.

    After the dialog closes, ``self.result`` is ``True`` if any files were
    added (so the caller knows to refresh the input sources list).
    """

    def __init__(
        self, parent: tk.Misc, project_path: Path,
        execution_mgr=None, input_source_mgr=None,
        convert_callback=None, update_xlsx_callback=None,
    ) -> None:
        super().__init__(parent)
        self.title("Add input sources")
        self.result: bool = False
        self.old_convert_started: bool = False  # True if old FlexTool conversion was started
        self.files_to_convert: list[str] = []  # Excel files converted to sqlite (for the "Done" notice)
        self.files_to_update_xlsx: list[str] = []  # Excel files round-tripped in place (for the "Done" notice)
        self._project_path = project_path
        self._input_dir = project_path / "input_sources"
        self._input_dir.mkdir(parents=True, exist_ok=True)
        self._execution_mgr = execution_mgr
        self._input_source_mgr = input_source_mgr
        # Callbacks that start a migration job immediately (streamed to the
        # Execution window) as soon as the user picks how to handle a file,
        # instead of deferring until the dialog is closed.
        self._convert_callback = convert_callback
        self._update_xlsx_callback = update_xlsx_callback

        # Root directories for templates
        self._flextool_root = get_projects_dir().parent

        # ── Modal behaviour ─────────────────────────────────────────
        self.transient(parent)
        self.grab_set()

        # No fixed size — the dialog takes only the space its widgets need.
        # The old fixed cw*77 x lh*45 box left most of it empty and grew huge
        # once cw/lh scaled up under DPI. resizable(False) keeps the fit.
        self.resizable(False, False)

        self._build_widgets()

        # Close via window-manager "X"
        self.protocol("WM_DELETE_WINDOW", self._on_back)

        # Size to natural content (clamped to the screen) and centre on parent.
        self.update_idletasks()
        w = min(self.winfo_reqwidth(), self.winfo_screenwidth())
        h = min(self.winfo_reqheight(), self.winfo_screenheight())
        px = parent.winfo_rootx()
        py = parent.winfo_rooty()
        pw = parent.winfo_width()
        ph = parent.winfo_height()
        x = px + (pw - w) // 2
        y = py + (ph - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

        # Block until closed
        parent.wait_window(self)

    # ── Widget construction ─────────────────────────────────────────

    def _build_widgets(self) -> None:
        pad = dict(padx=10, pady=5)
        from flextool.gui.ui_metrics import get_metrics
        lh = get_metrics(self).lh
        section_pad = dict(padx=10, pady=(lh, 5))

        # Use the named-font string so a live UI font size change reaches
        # these LabelFrame titles.
        style = ttk.Style()
        style.configure("AddDialog.TLabelframe.Label", font="TkDefaultFont")

        # ── Copy to project section ─────────────────────────────────
        copy_frame = ttk.LabelFrame(
            self, text="Copy an existing FlexTool 3.x input file to the project", padding=8, style="AddDialog.TLabelframe",
        )
        copy_frame.pack(fill="x", **pad)

        choose_copy_btn = ttk.Button(
            copy_frame,
            text="Choose and copy files...",
            command=self._on_choose_and_copy,
        )
        choose_copy_btn.pack(fill="x")

        # ── Add external reference section ──────────────────────────
        ext_frame = ttk.LabelFrame(
            self,
            text="Add outside input file to the project (reference only, not copied)",
            padding=8, style="AddDialog.TLabelframe",
        )
        ext_frame.pack(fill="x", **pad)

        choose_ext_btn = ttk.Button(
            ext_frame,
            text="Choose external files...",
            command=self._on_choose_external,
        )
        choose_ext_btn.pack(fill="x")

        # ── Add empty FlexTool input Excel section ──────────────────
        xlsx_frame = ttk.LabelFrame(
            self, text="Add empty FlexTool input Excel", padding=8,
            style="AddDialog.TLabelframe",
        )
        xlsx_frame.pack(fill="x", **section_pad)

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

        # ── Add empty FlexTool input database section ───────────────
        sqlite_frame = ttk.LabelFrame(
            self, text="Add empty FlexTool input database", padding=8,
            style="AddDialog.TLabelframe",
        )
        sqlite_frame.pack(fill="x", **section_pad)

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
            self, text="Convert from FlexTool 2.0 input file", padding=8,
            style="AddDialog.TLabelframe",
        )
        old_frame.pack(fill="x", **section_pad)

        convert_row = ttk.Frame(old_frame)
        convert_row.pack(fill="x")
        convert_btn = ttk.Button(
            convert_row,
            text="Convert input data file (and its sensitivities)...",
            command=self._on_convert_old_flextool,
        )
        convert_btn.pack(side="left")

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
                    _tell_ft2_file(self, fp.name)
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
                    _tell_unknown_excel(self, fp.name)
                    continue

                # For older files, ask how to handle migration before copying
                if needs_migration:
                    migration_choice = _ask_migration_choice(
                        self,
                        fp.name,
                        info.version,
                        CURRENT_FLEXTOOL_DB_VERSION,
                    )
                    if migration_choice == "cancel":
                        continue

            dest = self._input_dir / fp.name
            if dest.exists() and not _ask_replace(self, fp.name):
                continue
            try:
                shutil.copy2(str(fp), str(dest))
                copied_names.append(fp.name)
            except OSError as exc:
                errors.append(f"{fp.name}: {exc}")
                logger.error("Failed to copy %s: %s", fp, exc)
                continue

            # Start the migration job immediately so the user does not have to
            # close the dialog first; progress streams to the Execution window.
            if migration_choice == "sqlite":
                self.files_to_convert.append(fp.name)
                self._run_migration_callback(self._convert_callback, fp.name)
            elif migration_choice == "excel":
                self.files_to_update_xlsx.append(fp.name)
                self._run_migration_callback(self._update_xlsx_callback, fp.name)

        if errors:
            message_box.tell(
                self, "Some files were not added",
                "FlexTool could not copy these files into the project:", message_box.name_list(errors),
                "Check that they exist and are not open in another program.",
                kind="error",
            )

        # When no migration callbacks are wired (e.g. in tests), the queued
        # files are migrated by the main window's input-source refresh right
        # after this dialog closes instead.

        # Files undergoing conversion are excluded from the "Done"
        # notification — they will appear after conversion finishes.
        migrating = set(self.files_to_convert) | set(self.files_to_update_xlsx)
        ready_names = [n for n in copied_names if n not in migrating]

        if copied_names:
            self.result = True
        if ready_names:
            _tell_added(
                self, ready_names, "'{}' is copied into the project's input sources.",
                "These files are copied into the project's input sources:",
            )

    def _run_migration_callback(self, callback, name: str) -> None:
        """Invoke a migration-start callback with the modal grab released.

        The callback (``_convert_xlsx_to_sqlite`` / ``_update_xlsx_version``)
        starts a background job and may raise its own message boxes or the
        Execution window.  Releasing this dialog's grab first keeps those
        windows reachable; the callback returns promptly (it only kicks off a
        worker thread), after which the dialog re-grabs and stays modal.
        """
        if callback is None:
            return
        had_grab = self.grab_current() is self
        if had_grab:
            self.grab_release()
        try:
            callback(name)
        finally:
            if had_grab and self.winfo_exists():
                self.grab_set()

    def _on_choose_external(self) -> None:
        """Register external files as input source references (no copy)."""
        if self._input_source_mgr is None:
            message_box.tell(
                self, "Open a project first",
                "Choose a project in the Project list at the top, or create one with Projects….",
            )
            return

        initial_dir = self._flextool_root
        if not initial_dir.is_dir():
            initial_dir = Path.home()

        try:
            root = self.winfo_toplevel()
            main_window_width = root.winfo_width()
            screen_height = root.winfo_screenheight()
        except Exception:
            main_window_width = 700
            screen_height = 800

        picker = FilePickerDialog(
            self,
            title="Select external input source files",
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

        from flextool.process_inputs import (
            detect_excel_format, ExcelFormat, CURRENT_FLEXTOOL_DB_VERSION,
        )

        errors: list[str] = []
        added_names: list[str] = []
        for fp in filepaths:
            fp = Path(fp)

            if fp.suffix.lower() in (".xlsx", ".xlsm", ".ods"):
                info = detect_excel_format(fp)
                if info.format == ExcelFormat.OLD_V2:
                    _tell_ft2_file(self, fp.name)
                    continue
                if info.format == ExcelFormat.UNKNOWN:
                    _tell_unknown_excel(self, fp.name)
                    continue
                needs_migration = (
                    info.format == ExcelFormat.SPECIFICATION
                    or (
                        info.format == ExcelFormat.SELF_DESCRIBING
                        and info.version is not None
                        and info.version < CURRENT_FLEXTOOL_DB_VERSION
                    )
                )
                # #71: only an external Excel file is never updated; an external .sqlite gets the
                # migration consent dialog at the next refresh.
                if needs_migration and message_box.ask(
                    self, "Add an outdated Excel file?",
                    _older_version_text(fp.name, info.version, CURRENT_FLEXTOOL_DB_VERSION)
                    + " FlexTool does not update an Excel file outside the project, so its scenarios may not run.",
                    "To have it updated, add a copy to the project instead.",
                    buttons=(("Add anyway", "add"), ("Cancel", None)), kind="danger", docs=_DOCS_EXCEL,
                ) != "add":
                    continue

            try:
                name, _rel = self._input_source_mgr.add_external_ref(fp)
                added_names.append(name)
            except ValueError as exc:
                errors.append(str(exc))
            except Exception as exc:
                errors.append(f"{fp.name}: {exc}")
                logger.error("Failed to add external ref %s: %s", fp, exc)

        if errors:
            message_box.tell(
                self, "Some files were not added",
                "FlexTool could not add these files:", message_box.name_list(errors), kind="error",
            )

        if added_names:
            self.result = True
            _tell_added(
                self, added_names,
                "'{}' is now in the input sources as an external reference; the file stays where it is.",
                "These files are now in the input sources as external references; they stay where they are:",
            )

    def _on_add_xlsx(self) -> None:
        """Create an empty FlexTool Excel from the JSON master template."""
        name = self._xlsx_name_var.get().strip()
        filename = f"{name}.xlsx"
        dest = self._input_dir / filename
        if not _check_new_name(self, name, dest):
            return

        json_template = (
            package_data_path("schemas/spinedb_schema.json")
        )
        if not json_template.exists():
            _tell_template_missing(self, json_template)
            return

        import tempfile

        tmp_sqlite = None
        try:
            from flextool.update_flextool.initialize_database import (
                initialize_database,
            )
            from flextool.export_to_tabular.export_to_excel import export_to_excel
            from flextool.export_to_tabular.db_reader import read_database
            from flextool.export_to_tabular.sheet_config import load_settings
            from flextool.gui.dialogs.group_picker import GroupPickerDialog

            # Create a temporary sqlite from the JSON template
            with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
                tmp_sqlite = Path(tmp.name)
            initialize_database(str(json_template), str(tmp_sqlite))

            # Read parameter groups + required list, then ask the user
            db_url = sqlite_url_string(tmp_sqlite)
            dc = read_database(db_url)
            settings = load_settings()
            groups = [
                {"name": name, **meta}
                for name, meta in dc.parameter_groups.items()
            ]
            required = settings.get("required_groups", []) or []

            picker = GroupPickerDialog(self, groups, required)
            if picker.result is None:
                # Cancelled — abort silently, no output created
                return

            include_groups = (
                None if picker.result == "all" else list(picker.result)
            )
            export_to_excel(db_url, str(dest), include_groups=include_groups)
        except ImportError as exc:
            _tell_create_failed(self, exc)
            return
        except Exception as exc:
            _tell_create_failed(self, exc)
            if dest.exists():
                dest.unlink(missing_ok=True)
            return
        finally:
            if tmp_sqlite is not None and tmp_sqlite.exists():
                tmp_sqlite.unlink(missing_ok=True)

        self.result = True
        message_box.tell(self, "File created", f"'{filename}' is now in the input sources.")

    def _on_add_sqlite(self) -> None:
        """Create a new Spine database from the master template."""
        name = self._sqlite_name_var.get().strip()
        filename = f"{name}.sqlite"
        dest = self._input_dir / filename
        if not _check_new_name(self, name, dest):
            return

        json_template = (
            package_data_path("schemas/spinedb_schema.json")
        )
        if not json_template.exists():
            _tell_template_missing(self, json_template)
            return

        try:
            from flextool.update_flextool.initialize_database import (
                initialize_database,
            )

            initialize_database(str(json_template), str(dest))
        except Exception as exc:  # ImportError included: _tell_create_failed tells it apart
            _tell_create_failed(self, exc)
            return

        self.result = True
        message_box.tell(self, "File created", f"'{filename}' is now in the input sources.")

    def _on_convert_old_flextool(self) -> None:
        """Convert a FlexTool 2.0 input data file (and optionally the
        sensitivities of its master workbook) to a Spine DB.

        The whole import runs in ``cmd_read_old_flextool``: a new database at
        the frozen import schema, the base data, the sensitivities (when a
        master workbook is given), migration to the current version and the
        end report of what could not be translated.  The sensitivities must
        be written before the migration, so the master is asked for here.

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
            title="Select FlexTool 2.0 input data file",
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

        master_path = ask_old_flextool_master(
            self, filepath.parent, main_window_width, screen_height,
        )
        if master_path is False:
            return

        dest_name = filepath.stem + ".sqlite"
        dest = self._input_dir / dest_name

        if dest.exists() and not _ask_replace(
            self, dest_name, "The converted database replaces it when the conversion has succeeded.",
        ):
            return

        # Import into a working file OUTSIDE input_sources/ (the only folder
        # the input-source scan reads) and move it in only when the import,
        # which includes the migration, has succeeded.  A refresh during the
        # import (another job finishing, the Refresh button) can then never
        # find the database at the import version and migrate it under the
        # importer.  Same filesystem (both under the project): the move is
        # an atomic rename, which also replaces an existing destination.
        work_dir = self._project_path / "intermediate"
        work_dir.mkdir(parents=True, exist_ok=True)
        work = work_dir / f"{filepath.stem}.ft2import.sqlite"
        work.unlink(missing_ok=True)  # leftover of a crashed import
        old_sources: set[str] = set()
        if self._input_source_mgr is not None:
            old_sources = {s.name for s in self._input_source_mgr._sources}

        db_url = sqlite_url_string(work)
        cmd = [
            sys.executable, SAFE_PATH_FLAG, "-m",
            "flextool.cli.cmd_read_old_flextool",
            str(filepath),
            db_url,
        ]
        if master_path is not None:
            cmd += ["--master", str(master_path)]

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

            main_window = self.master
            mgr = self._execution_mgr  # capture for threads

            # Make the output visible even if the execution window was closed.
            main_window._open_or_raise_execution_window()
            if main_window.execution_window is not None:
                main_window.execution_window.select_job(job.job_id)

            def _worker() -> None:
                rc = 1
                try:
                    env = child_env(PYTHONUNBUFFERED="1")
                    proc = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1, cwd=str(self._flextool_root),
                        env=env,
                    )
                    with mgr._lock:
                        job.process = proc
                    for line in proc.stdout:  # type: ignore[union-attr]
                        mgr.append_stdout(job.job_id, line.rstrip("\n"))
                    proc.wait()
                    rc = proc.returncode
                    if rc != 0:
                        mgr.append_stdout(
                            job.job_id,
                            f"\nConversion failed (exit code {rc}).",
                        )
                    else:
                        os.replace(work, dest)
                        mgr.append_stdout(
                            job.job_id, f"Input source '{dest_name}' is ready.")
                except Exception as exc:
                    rc = 1
                    logger.error("Old FlexTool conversion failed: %s", exc, exc_info=True)
                    mgr.append_stdout(job.job_id, f"\nError: {exc}")
                work.unlink(missing_ok=True)

                mgr.finish_job(job.job_id, rc == 0)
                if rc == 0:
                    def _show_new_source() -> None:
                        # Tick the new source and its scenarios (the dialog
                        # closed before the database existed).
                        main_window._refresh_input_sources()
                        main_window._autocheck_new_sources(old_sources)
                    try:
                        main_window.after(0, _show_new_source)
                    except Exception:
                        pass  # Main window may be gone

            threading.Thread(target=_worker, daemon=True).start()

            self.result = True
            self.old_convert_started = True
        else:
            # Fallback: no execution manager — run synchronously with busy cursor
            self.config(cursor="watch")
            self.update()
            try:
                from flextool.process_inputs.write_old_flextool_to_db import (
                    import_old_flextool_xlsm,
                )

                report = import_old_flextool_xlsm(
                    str(filepath), db_url,
                    alternative_name=_OLD_FLEX_ALTERNATIVE,
                    master=str(master_path) if master_path is not None else None,
                )
                os.replace(work, dest)
            except Exception as exc:
                logger.error("Old FlexTool conversion failed: %s", exc, exc_info=True)
                message_box.tell(
                    self, "The conversion did not work", str(exc), "The full error is in the FlexTool log.",
                    kind="error", docs=_DOCS_CONVERT_FT2,
                )
                return
            finally:
                work.unlink(missing_ok=True)
                self.config(cursor="")

            logger.info("%s", report.format())
            self.result = True
            failed = (f" {len(report.failed)} values could not be written."
                      if report.failed else "")
            message_box.tell(
                self, "Converted: check the import report",
                f"'{filepath.name}' is now '{dest_name}' in the input sources. {len(report.unmapped)} FlexTool 2 "
                f"inputs were not carried over or work differently in FlexTool 4.{failed}",
                "The import report in the FlexTool log (flextool_gui.log in the projects folder) says what to do "
                "for each.",
                docs=_DOCS_CONVERT_FT2,
            )

    def _on_back(self) -> None:
        """Close the dialog."""
        self.grab_release()
        self.destroy()
