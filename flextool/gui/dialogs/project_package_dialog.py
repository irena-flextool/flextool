"""The project export and import dialogs (Manage projects → **Export…** / **Import…**).

Export lists what every package holds (greyed), a checkbox tree of the project's result runs with their sizes, the
external input files option and the package file; the total is the uncompressed size (parquet barely compresses).
A ticked run whose input changed after the run is warned about; a run still being written (by this window, Spine
Toolbox or the terminal, as its run record says) is marked and refused.  By default the run records lose this
computer's paths, user and host name and the run logs stay home.  Export is refused while the project has running
jobs, asks before replacing an existing package file, and writes the zip on a worker thread.

Import reads the package's manifest without unpacking it, fills the project name (``<Name>_2`` when the name is
taken), shows the version warnings and refuses a database newer than this FlexTool.  The work is done by
:mod:`flextool.project_package`.
"""
from __future__ import annotations

import logging
import threading
import tkinter as tk
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from flextool import project_package as pp
from flextool.gui.check_tree import CheckTreeController
from flextool.gui.data_models import TIMESTAMP_FORMAT

logger = logging.getLogger(__name__)

WARNING_FG = "#e0a800"
GREY_FG = "#808080"
ERROR_FG = "#d03030"
_CHECKED, _UNCHECKED = "■", "□"
_POLL_MS = 100


def format_size(n: int) -> str:
    """``12.3 MB`` style sizes (decimal units)."""
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}"
    return f"{n} B"


def _centre(win: tk.Toplevel, parent: tk.Misc) -> None:
    win.update_idletasks()
    x = parent.winfo_rootx() + (parent.winfo_width() - win.winfo_width()) // 2
    y = parent.winfo_rooty() + (parent.winfo_height() - win.winfo_height()) // 2
    win.geometry(f"+{max(x, 0)}+{max(y, 0)}")


class _Worker:
    """Runs *fn* on a thread; the dialog polls :attr:`done` from the Tk thread."""

    def __init__(self, fn: Callable[[], object]) -> None:
        self.done = False
        self.result: object = None
        self.error: BaseException | None = None
        self.progress = (0, 0)

        def run() -> None:
            try:
                self.result = fn()
            except BaseException as exc:   # reported on the Tk thread
                self.error = exc
            finally:
                self.done = True

        self.thread = threading.Thread(target=run, name="flextool-project-package", daemon=True)
        self.thread.start()


class ExportDialog(tk.Toplevel):
    """Export the project at *project_dir* to a package.

    *jobs_running()* tells whether the project has running or pending jobs (Export is refused then).  After a
    successful export ``self.manifest`` is the package's manifest and ``self.zip_path`` its file.
    """

    def __init__(
        self,
        parent: tk.Misc,
        project_dir: Path,
        *,
        jobs_running: Callable[[], bool] | None = None,
        solver_config_dir: Path | list[Path] | None = None,
        target_dir: Path | None = None,
    ) -> None:
        super().__init__(parent)
        self._parent = parent
        self._project_dir = Path(project_dir)
        self._jobs_running = jobs_running
        self._solver_config_dir = solver_config_dir
        self.manifest: pp.Manifest | None = None
        self.zip_path: Path | None = None
        self._worker: _Worker | None = None
        self.title(f"Export project {self._project_dir.name}")
        self.transient(parent)

        from flextool.gui.ui_metrics import get_metrics
        metrics = get_metrics(self)
        self._cw, self._lh = metrics.cw, metrics.lh
        self.minsize(self._cw * 70, self._lh * 26)

        self._fixed = pp.always_included(self._project_dir, solver_config_dir)
        self._runs = {run.name: run for run in pp.list_runs(self._project_dir)}
        target = (target_dir or Path.cwd()) / pp.default_package_name(self._project_dir.name)
        self._target_var = tk.StringVar(value=str(target))
        #: The package file the user agreed to replace (in the save dialog of Browse…, or when asked).
        self._replace_ok: str | None = None
        self._external_var = tk.BooleanVar(value=True)
        self._scrub_var = tk.BooleanVar(value=True)
        self._build()
        self._update_total()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Escape>", lambda _e: self._on_close())
        _centre(self, parent)
        self.grab_set()

    # ── Widgets ─────────────────────────────────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        pad = dict(padx=10, pady=4)
        wrap = self._cw * 80
        fixed = ttk.LabelFrame(self, text="Always in the package", padding=6)
        fixed.pack(fill="x", **pad)
        inputs = self._fixed["inputs"]
        input_bytes = sum(size for _name, size in inputs)
        names = ", ".join(name for name, _size in inputs) or "none"
        lines = [
            f"Input sources ({format_size(input_bytes)}): {names}",
            "settings.yaml, without the settings of this machine (execution limits, parallel executions, "
            "learned memory budgets)",
        ]
        if self._fixed["plot_settings"]:
            lines.append("plot_settings.yaml")
        opts = self._fixed["solver_config_files"]
        if opts:
            lines.append("Solver options (the files this project's runs use): "
                         + ", ".join(f"{name} from {path.parent}" for name, path in opts.items()))
        else:
            lines.append("Solver options: none (this project's runs use FlexTool's defaults; looked in "
                         + ", ".join(str(folder) for folder in self._fixed["solver_config_dirs"]) + ")")
        withheld = [pp.withheld_file_line(name, path, problems)
                    for name, (path, problems) in self._fixed["solver_config_withheld"].items()]
        lines.append("With each result run: its run record (_run_status.json), which can show this computer's "
                     "folder paths, user name and host name, and its run log (_run.log)")
        for line in lines:
            ttk.Label(fixed, text=line, foreground=GREY_FG, wraplength=wrap, justify="left").pack(anchor="w")
        for line in withheld:
            ttk.Label(fixed, text=line, foreground=WARNING_FG, wraplength=wrap, justify="left").pack(anchor="w")
        ttk.Checkbutton(
            fixed, text="Remove this computer's folder paths, user name and host name from the run records, and "
                        "leave the run logs out", variable=self._scrub_var).pack(anchor="w", pady=(4, 0))
        external = self._fixed["external"]
        if external:
            present = [name for name, _path, exists in external if exists]
            self._external_check = ttk.Checkbutton(
                fixed, text=f"Copy external input files into the package ({', '.join(present) or 'none found'})",
                variable=self._external_var, command=self._update_total)
            self._external_check.pack(anchor="w", pady=(4, 0))
            for name, path, exists in external:
                if not exists:
                    ttk.Label(fixed, text=f"{name} is missing ({path}); it cannot be copied.",
                              foreground=WARNING_FG, wraplength=wrap, justify="left").pack(anchor="w")

        runs = ttk.LabelFrame(self, text="Result runs", padding=6)
        runs.pack(fill="both", expand=True, **pad)
        buttons = ttk.Frame(runs)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="None", command=lambda: self._set_all(False)).pack(side="left")
        ttk.Button(buttons, text="All", command=lambda: self._set_all(True)).pack(side="left", padx=(6, 0))
        frame = ttk.Frame(runs)
        frame.pack(fill="both", expand=True, pady=(4, 0))
        self.tree = ttk.Treeview(frame, columns=("check", "run", "size", "note"), show="headings",
                                 height=8, selectmode="extended")
        for col, text, width in (("check", "", 3), ("run", "Run", 30), ("size", "Size", 10),
                                 ("note", "Note", 30)):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=self._cw * width, stretch=col in ("run", "note"),
                             anchor="center" if col == "check" else "w")
        self.tree.tag_configure("stale", foreground=WARNING_FG)
        self.tree.tag_configure("active", foreground=ERROR_FG)
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        for run in self._runs.values():
            if run.active:
                note, tags = "running: cannot be exported yet", ("active",)
            elif run.stale:
                note, tags = "input changed after the run", ("stale",)
            else:
                note, tags = "", ()
            self.tree.insert("", "end", iid=run.name, values=(_UNCHECKED, run.name, format_size(run.bytes), note),
                             tags=tags)
        if not self._runs:
            ttk.Label(runs, text="This project has no results.", foreground=GREY_FG).pack(anchor="w")
        self.checks = CheckTreeController(self.tree, checked_glyph=_CHECKED, unchecked_glyph=_UNCHECKED,
                                          on_toggle=lambda _iids: self._update_total())

        self.total_label = ttk.Label(self, text="")
        self.total_label.pack(anchor="w", **pad)
        self.stale_label = ttk.Label(self, text="", foreground=WARNING_FG, wraplength=wrap, justify="left")
        self.stale_label.pack(anchor="w", padx=10)

        target = ttk.Frame(self)
        target.pack(fill="x", **pad)
        ttk.Label(target, text="Package file:").pack(side="left")
        ttk.Entry(target, textvariable=self._target_var).pack(side="left", fill="x", expand=True, padx=(5, 5))
        ttk.Button(target, text="Browse…", command=self._browse).pack(side="left")

        self.progress = ttk.Progressbar(self, mode="determinate", maximum=1000)
        self.progress.pack(fill="x", **pad)
        self.status_label = ttk.Label(self, text="", wraplength=wrap, justify="left")
        self.status_label.pack(anchor="w", padx=10)

        bottom = ttk.Frame(self)
        bottom.pack(fill="x", **pad)
        self.export_button = ttk.Button(bottom, text="Export", command=self._on_export)
        self.export_button.pack(side="left")
        self.close_button = ttk.Button(bottom, text="Cancel", command=self._on_close)
        self.close_button.pack(side="right")

    # ── State ───────────────────────────────────────────────────────────────────────────────────────────────

    def selected_runs(self) -> list[str]:
        return [name for name in self._runs if self.checks.is_checked(name)]

    def _set_all(self, value: bool) -> None:
        for name in self._runs:
            self.checks.set_checked(name, value, notify=False)
        self._update_total()

    def total_bytes(self) -> int:
        """The uncompressed size of the package with the current choices."""
        total = sum(size for _name, size in self._fixed["inputs"])
        if self._external_var.get():
            for _name, path, exists in self._fixed["external"]:
                if exists:
                    total += path.stat().st_size
        for name in ("settings.yaml", "plot_settings.yaml"):
            path = self._project_dir / name
            if path.is_file():
                total += path.stat().st_size
        total += sum(path.stat().st_size for path in self._fixed["solver_config_files"].values())
        runs = self.selected_runs()
        total += sum(self._runs[name].bytes for name in runs)
        bounds = pp.axis_bounds_for_runs(self._project_dir, runs)
        if bounds is not None:
            total += len(bounds)
        return total

    def _update_total(self) -> None:
        runs = self.selected_runs()
        self.total_label.configure(
            text=f"Total: {format_size(self.total_bytes())} uncompressed, {len(runs)} of {len(self._runs)} "
                 f"result runs")
        stale = [name for name in runs if self._runs[name].stale]
        self.stale_label.configure(
            text=("The input of these runs changed after the run, so their results do not match the shipped "
                  f"input: {', '.join(stale)}") if stale else "")

    def _browse(self) -> None:
        current = Path(self._target_var.get())
        chosen = filedialog.asksaveasfilename(
            parent=self, title="Save the project package", initialdir=str(current.parent),
            initialfile=current.name, defaultextension=".zip",
            filetypes=[("FlexTool project package", "*.zip"), ("All files", "*")])
        if chosen:
            self._target_var.set(chosen)
            self._replace_ok = chosen          # the save dialog asked about replacing it

    def _set_status(self, text: str, colour: str = "") -> None:
        self.status_label.configure(text=text, foreground=colour)

    # ── Actions ─────────────────────────────────────────────────────────────────────────────────────────────

    def _on_export(self) -> None:
        if self._worker is not None:
            return
        if self._jobs_running is not None and self._jobs_running():
            self._set_status("The project has running or pending jobs. Wait for the runs to finish (or stop "
                             "them) and export then.", ERROR_FG)
            return
        target = self._target_var.get().strip()
        if not target:
            self._set_status("Choose the package file.", ERROR_FG)
            return
        zip_path = Path(target)
        runs = self.selected_runs()
        active = [name for name in runs if self._runs[name].active]
        if active:
            self._set_status(pp.active_run_message(active), ERROR_FG)
            return
        overwrite = False
        if zip_path.exists():
            if self._replace_ok != target and not messagebox.askyesno(
                    "Replace the package?", f"{zip_path} exists already. Replace it?", parent=self):
                return
            overwrite = True
        include_external = self._external_var.get()
        scrub = self._scrub_var.get()
        self.export_button.configure(state="disabled")
        self._set_status("Exporting…")

        def progress(done: int, total: int) -> None:
            worker.progress = (done, total)

        worker = _Worker(lambda: pp.export_project(
            self._project_dir, zip_path, runs, include_external,
            solver_config_dir=self._solver_config_dir, progress=progress, overwrite=overwrite,
            scrub_run_records=scrub))
        self._worker = worker
        self.zip_path = zip_path
        self.after(_POLL_MS, self._poll)

    def _poll(self) -> None:
        worker = self._worker
        if worker is None:
            return
        done, total = worker.progress
        if total:
            self.progress.configure(value=1000 * done / total)
        if not worker.done:
            self.after(_POLL_MS, self._poll)
            return
        self._worker = None
        if worker.error is not None:
            logger.warning("Project export failed", exc_info=worker.error)
            self.export_button.configure(state="normal")
            self.progress.configure(value=0)
            self._set_status(f"Export failed: {worker.error}", ERROR_FG)
            return
        self.manifest = worker.result  # type: ignore[assignment]
        self.progress.configure(value=1000)
        lines = [f"Exported to {self.zip_path}."]
        if self.manifest.stale_runs:
            lines.append("The input of these runs changed after the run: " + ", ".join(self.manifest.stale_runs))
        if self.manifest.external_not_included:
            lines.append("Not in the package: " + ", ".join(self.manifest.external_not_included))
        lines.extend(self.manifest.export_warnings)
        self._set_status("\n".join(lines))
        self.close_button.configure(text="Close")

    def _on_close(self) -> None:
        if self._worker is not None:
            self._set_status("The export is still being written; close when it is done.", ERROR_FG)
            return
        self.grab_release()
        self.destroy()


class ImportDialog(tk.Toplevel):
    """Import a project package into *projects_dir*.  After a successful import ``self.result`` is the new
    project's name and ``self.report`` the :class:`~flextool.project_package.ImportReport`."""

    def __init__(
        self,
        parent: tk.Misc,
        projects_dir: Path,
        *,
        zip_path: Path | None = None,
        solver_config_dir: Path | list[Path] | None = None,
    ) -> None:
        super().__init__(parent)
        self._projects_dir = Path(projects_dir)
        self._solver_config_dir = solver_config_dir
        self.result: str | None = None
        self.report: pp.ImportReport | None = None
        self.manifest: pp.Manifest | None = None
        self._blocking: list[str] = []
        self._worker: _Worker | None = None
        self.title("Import project")
        self.transient(parent)

        from flextool.gui.ui_metrics import get_metrics
        metrics = get_metrics(self)
        self._cw, self._lh = metrics.cw, metrics.lh
        self.minsize(self._cw * 70, self._lh * 18)
        self._file_var = tk.StringVar()
        self._name_var = tk.StringVar()
        self._build()
        self._name_var.trace_add("write", lambda *_a: self._update_state())
        if zip_path is not None:
            self.load(Path(zip_path))
        self._update_state()
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.bind("<Escape>", lambda _e: self._on_cancel())
        _centre(self, parent)
        self.grab_set()

    def _build(self) -> None:
        pad = dict(padx=10, pady=4)
        wrap = self._cw * 80
        row = ttk.Frame(self)
        row.pack(fill="x", **pad)
        ttk.Label(row, text="Package file:").pack(side="left")
        entry = ttk.Entry(row, textvariable=self._file_var)
        entry.pack(side="left", fill="x", expand=True, padx=(5, 5))
        entry.bind("<Return>", lambda _e: self.load(Path(self._file_var.get().strip())))
        ttk.Button(row, text="Browse…", command=self._browse).pack(side="left")

        self.summary_label = ttk.Label(self, text="Choose a .flextool.zip package.", wraplength=wrap,
                                       justify="left")
        self.summary_label.pack(anchor="w", **pad)
        self.warning_label = ttk.Label(self, text="", foreground=WARNING_FG, wraplength=wrap, justify="left")
        self.warning_label.pack(anchor="w", padx=10)

        name_row = ttk.Frame(self)
        name_row.pack(fill="x", **pad)
        ttk.Label(name_row, text="Project name:").pack(side="left")
        self.name_entry = ttk.Entry(name_row, textvariable=self._name_var, width=40)
        self.name_entry.pack(side="left", fill="x", expand=True, padx=(5, 0))
        self.name_note = ttk.Label(self, text="", foreground=ERROR_FG, wraplength=wrap, justify="left")
        self.name_note.pack(anchor="w", padx=10)

        self.progress = ttk.Progressbar(self, mode="indeterminate")
        self.progress.pack(fill="x", **pad)
        bottom = ttk.Frame(self)
        bottom.pack(fill="x", **pad)
        self.import_button = ttk.Button(bottom, text="Import", command=self._on_import, state="disabled")
        self.import_button.pack(side="left")
        ttk.Button(bottom, text="Cancel", command=self._on_cancel).pack(side="right")
        self.status_label = ttk.Label(self, text="", foreground=ERROR_FG, wraplength=wrap, justify="left")
        self.status_label.pack(anchor="w", padx=10)

    def _browse(self) -> None:
        from flextool.gui.dialogs.file_picker import FilePickerDialog

        picker = FilePickerDialog(self, title="Choose a FlexTool project package", initialdir=str(Path.cwd()),
                                  filetypes=[("FlexTool project package", "*.zip"), ("All files", "*")])
        self.grab_set()
        if picker.result:
            self.load(Path(picker.result))

    def load(self, zip_path: Path) -> None:
        """Read the manifest of *zip_path* (nothing is unpacked) and fill the summary, warnings and name."""
        self._file_var.set(str(zip_path))
        self.manifest = None
        self._blocking = []
        try:
            manifest = pp.read_manifest(zip_path)
        except pp.PackageError as exc:
            self.summary_label.configure(text=str(exc))
            self.warning_label.configure(text="")
            self._update_state()
            return
        self.manifest = manifest
        created = manifest.created
        try:
            created = datetime.fromisoformat(manifest.created).strftime(TIMESTAMP_FORMAT)
        except ValueError:
            pass
        lines = [f"Project {manifest.project}, made {created} with FlexTool {manifest.flextool or 'unknown'}."]
        for path, version in sorted(manifest.db_versions.items()):
            frozen = " (frozen)" if path in manifest.frozen else ""
            lines.append(f"  {path}: data version {version}{frozen}")
        lines.append("Result runs: " + (", ".join(manifest.results) if manifest.results else "none"))
        if manifest.solver_config:
            lines.append("Solver options: " + ", ".join(manifest.solver_config)
                         + f" (unpacked to {pp.SOLVER_CONFIG_DIR}/ of the project; its runs use them ahead of "
                         "the workspace's, with the input database's rules until you trust the project's solver "
                         "option files)")
        self.summary_label.configure(text="\n".join(lines))
        self._blocking, warnings = pp.version_messages(manifest)
        self.warning_label.configure(text="\n".join(self._blocking + warnings),
                                     foreground=ERROR_FG if self._blocking else WARNING_FG)
        self._name_var.set(pp.suggest_name(self._projects_dir, manifest.project))
        self._update_state()

    def _update_state(self) -> None:
        name = self._name_var.get()
        note = ""
        ok = self.manifest is not None and not self._blocking and self._worker is None
        if self.manifest is not None:
            reason = pp.invalid_name_reason(name)
            if reason is not None:
                note, ok = reason, False
            elif pp.name_taken(self._projects_dir, name):
                note, ok = (f"A project named '{name}' exists already, e.g. use "
                            f"'{pp.suggest_name(self._projects_dir, name)}'."), False
        self.name_note.configure(text=note)
        self.import_button.configure(state="normal" if ok else "disabled")

    def _on_import(self) -> None:
        if self.manifest is None or self._worker is not None:
            return
        zip_path = Path(self._file_var.get().strip())
        name = self._name_var.get()
        self._worker = _Worker(lambda: pp.import_project(
            zip_path, self._projects_dir, name, solver_config_dir=self._solver_config_dir))
        self._update_state()
        self.progress.start(50)
        self.after(_POLL_MS, self._poll)

    def _poll(self) -> None:
        worker = self._worker
        if worker is None:
            return
        if not worker.done:
            self.after(_POLL_MS, self._poll)
            return
        self._worker = None
        self.progress.stop()
        if worker.error is not None:
            logger.warning("Project import failed", exc_info=worker.error)
            self._update_state()        # a name taken meanwhile is said there, and Import stays disabled
            if not isinstance(worker.error, pp.NameClashError):
                self.name_note.configure(text=f"Import failed: {worker.error}")
                self.import_button.configure(state="disabled")
            return
        self.report = worker.result  # type: ignore[assignment]
        self.result = self.report.name
        self.grab_release()
        self.destroy()

    def _on_cancel(self) -> None:
        if self._worker is not None:
            self.status_label.configure(text="The project is still being imported; close when it is done.")
            return
        self.result = None
        self.grab_release()
        self.destroy()


def _regrab(parent: tk.Misc) -> None:
    try:
        if isinstance(parent, tk.Toplevel) and parent.winfo_exists():
            parent.grab_set()
    except tk.TclError:
        pass


def show_export_dialog(parent: tk.Misc, project_dir: Path, *, jobs_running: Callable[[], bool] | None = None
                       ) -> pp.Manifest | None:
    """Open the export dialog and wait; returns the manifest of the written package, or ``None``."""
    dlg = ExportDialog(parent, project_dir, jobs_running=jobs_running)
    parent.wait_window(dlg)
    _regrab(parent)
    return dlg.manifest


def ask_import(parent: tk.Misc, projects_dir: Path) -> pp.ImportReport | None:
    """Open the import dialog and wait; returns the import report, or ``None`` when nothing was imported."""
    dlg = ImportDialog(parent, projects_dir)
    parent.wait_window(dlg)
    _regrab(parent)
    return dlg.report
