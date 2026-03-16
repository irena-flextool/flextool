from __future__ import annotations

import logging
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import messagebox, ttk

from flextool.gui.config_parser import parse_plot_configs
from flextool.gui.dialogs.file_picker import FilePickerDialog
from flextool.gui.data_models import PlotSettings, ProjectSettings
from flextool.gui.project_utils import get_projects_dir
from flextool.gui.settings_io import save_project_settings

logger = logging.getLogger(__name__)


def _flextool_root() -> Path:
    """Return the repository root (three levels up from gui/)."""
    return get_projects_dir().parent


def _relative_config_path(abs_path: Path) -> str:
    """Return the path relative to the FlexTool root when possible."""
    root = _flextool_root()
    try:
        return str(abs_path.relative_to(root))
    except ValueError:
        return str(abs_path)


class _PlotSection:
    """Widget group for one plot-settings section (single or comparison).

    Manages a ``ttk.LabelFrame`` containing start-time / duration entries,
    a config-file selector, and a list of config checkboxes.
    """

    def __init__(
        self,
        parent: tk.Misc,
        label: str,
        settings: PlotSettings,
        default_config_file: str,
    ) -> None:
        self._settings = settings
        self._default_config_file = default_config_file

        self.frame = ttk.LabelFrame(parent, text=label, padding=10)

        # ── Start time ─────────────────────────────────────────────
        row = 0
        ttk.Label(self.frame, text="Time series plot start time (as integer timestep):").grid(
            row=row, column=0, sticky="w", pady=(0, 4),
        )
        self._start_var = tk.StringVar(value=str(settings.start_time))
        start_entry = ttk.Entry(self.frame, textvariable=self._start_var, width=10)
        start_entry.grid(row=row, column=1, sticky="w", padx=(10, 0), pady=(0, 4))
        _register_int_validation(start_entry)

        # ── Duration ───────────────────────────────────────────────
        row = 1
        ttk.Label(self.frame, text="Time series plot duration (as number of timesteps):").grid(
            row=row, column=0, sticky="w", pady=(0, 4),
        )
        self._duration_var = tk.StringVar(value=str(settings.duration))
        dur_entry = ttk.Entry(self.frame, textvariable=self._duration_var, width=10)
        dur_entry.grid(row=row, column=1, sticky="w", padx=(10, 0), pady=(0, 4))
        _register_int_validation(dur_entry)

        # ── Config file selector ───────────────────────────────────
        row = 2
        config_file = settings.config_file or default_config_file
        self._config_path = _resolve_config_path(config_file)

        config_row = ttk.Frame(self.frame)
        config_row.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(4, 4))
        config_row.columnconfigure(1, weight=1)

        ttk.Label(config_row, text="Plot config file:").grid(row=0, column=0, sticky="w")
        self._config_label_var = tk.StringVar(
            value=_relative_config_path(self._config_path)
        )
        ttk.Label(config_row, textvariable=self._config_label_var).grid(
            row=0, column=1, sticky="w", padx=(5, 5),
        )
        ttk.Button(config_row, text="Change", command=self._on_change_config).grid(
            row=0, column=2, sticky="e",
        )

        # ── Active configs list (Treeview with checkboxes) ─────────
        row = 3
        ttk.Label(self.frame, text="Active configs:").grid(
            row=row, column=0, sticky="w", pady=(4, 2),
        )

        row = 4
        self._lh = tkfont.nametofont("TkDefaultFont").metrics("linespace")
        cw = tkfont.nametofont("TkDefaultFont").measure("0")

        config_frame = ttk.Frame(self.frame)
        config_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        config_frame.columnconfigure(0, weight=1)
        config_frame.rowconfigure(0, weight=1)

        self._config_tree = ttk.Treeview(
            config_frame,
            columns=("check", "name"),
            show="headings",
            selectmode="extended",
            height=6,
        )
        self._config_tree.heading("check", text="\u25bd")
        self._config_tree.heading("name", text="Config")
        self._config_tree.column("check", width=cw * 3, minwidth=cw * 3, stretch=False)
        self._config_tree.column("name", width=cw * 20, minwidth=cw * 10, stretch=True)
        self._config_tree.grid(row=0, column=0, sticky="nsew")

        config_scroll = ttk.Scrollbar(config_frame, orient="vertical", command=self._config_tree.yview)
        config_scroll.grid(row=0, column=1, sticky="ns")
        self._config_tree.configure(yscrollcommand=config_scroll.set)

        self._config_tree.bind("<Button-1>", self._on_config_click)
        self._config_tree.bind("<space>", self._on_config_space)

        self._populate_configs()

    # ── Public helpers ────────────────────────────────────────────

    # Unicode checkbox characters (same as main window)
    _CHECK_ON = "\u25a3"   # ▣
    _CHECK_OFF = "\u25a1"  # □

    def collect(self) -> PlotSettings:
        """Return a ``PlotSettings`` built from the current widget state."""
        start = _parse_int(self._start_var.get(), self._settings.start_time)
        duration = _parse_int(self._duration_var.get(), self._settings.duration)
        config_file = _relative_config_path(self._config_path)
        active: list[str] = []
        for item in self._config_tree.get_children():
            values = self._config_tree.item(item, "values")
            if values and values[0] == self._CHECK_ON:
                active.append(values[1])
        return PlotSettings(
            start_time=start,
            duration=duration,
            config_file=config_file,
            active_configs=active,
        )

    # ── Internal ──────────────────────────────────────────────────

    def _populate_configs(self) -> None:
        """Populate the config Treeview from the YAML file."""
        for item in self._config_tree.get_children():
            self._config_tree.delete(item)

        config_names = parse_plot_configs(self._config_path)
        if not config_names:
            return

        saved_active = self._settings.active_configs
        saved_config_file = self._settings.config_file or self._default_config_file
        current_rel = _relative_config_path(self._config_path)
        is_same_file = (current_rel == saved_config_file)

        for name in config_names:
            if is_same_file and saved_active:
                check = self._CHECK_ON if name in saved_active else self._CHECK_OFF
            else:
                check = self._CHECK_ON
            self._config_tree.insert("", "end", values=(check, name))

    def _on_config_click(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """Toggle checkbox on click in the check column."""
        region = self._config_tree.identify_region(event.x, event.y)
        if region != "cell":
            return
        column = self._config_tree.identify_column(event.x)
        if column == "#1":  # check column
            item = self._config_tree.identify_row(event.y)
            if item:
                current = self._config_tree.set(item, "check")
                new_val = self._CHECK_OFF if current == self._CHECK_ON else self._CHECK_ON
                self._config_tree.set(item, "check", new_val)

    def _on_config_space(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        """Toggle checkboxes for selected items on space."""
        for item in self._config_tree.selection():
            current = self._config_tree.set(item, "check")
            new_val = self._CHECK_OFF if current == self._CHECK_ON else self._CHECK_ON
            self._config_tree.set(item, "check", new_val)

    def _on_change_config(self) -> None:
        """Open a file chooser to select a different YAML config file."""
        root = _flextool_root()
        initial_dir = self._config_path.parent
        if not initial_dir.is_dir():
            initial_dir = root

        # Determine dialog size from the top-level window
        try:
            toplevel = self.frame.winfo_toplevel()
            main_window_width = toplevel.winfo_width()
            screen_height = toplevel.winfo_screenheight()
        except Exception:
            main_window_width = 700
            screen_height = 800

        picker = FilePickerDialog(
            self.frame,
            title="Select plot config file",
            initialdir=str(initial_dir),
            filetypes=[
                ("YAML files", "*.yaml *.yml"),
                ("All files", "*"),
            ],
            multiple=False,
            width=main_window_width,
            height=int(screen_height * 0.75),
        )
        filepath = picker.result
        if not filepath:
            return

        new_path = Path(filepath) if not isinstance(filepath, Path) else filepath
        if not new_path.is_file():
            messagebox.showerror(
                "Invalid file",
                f"Cannot read:\n{new_path}",
                parent=self.frame,
            )
            return

        self._config_path = new_path
        self._config_label_var.set(_relative_config_path(new_path))
        self._populate_configs()


class PlotDialog(tk.Toplevel):
    """Modal dialog for configuring plot settings.

    Provides two sections -- one for single-scenario plots and one for
    scenario-comparison plots -- each with start-time, duration,
    config-file selection and active-config checkboxes.
    """

    def __init__(
        self,
        parent: tk.Tk,
        project_path: Path,
        settings: ProjectSettings,
    ) -> None:
        super().__init__(parent)
        self.title("Plot settings")
        self._project_path = project_path
        self._settings = settings

        # ── Modal behaviour ──────────────────────────────────────
        self.transient(parent)
        self.grab_set()

        # ── Font metrics for DPI-aware sizing ───────────────────
        default_font = tkfont.nametofont("TkDefaultFont")
        cw: int = default_font.measure("0")
        lh: int = default_font.metrics("linespace")

        # ── Dialog size ──────────────────────────────────────────
        self.geometry(f"{cw * 60}x{lh * 42}")
        self.resizable(True, True)
        self.minsize(cw * 60, lh * 30)

        self._build_widgets()

        # Close via window-manager "X"
        self.protocol("WM_DELETE_WINDOW", self._on_ok)

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

    # ── Widget construction ──────────────────────────────────────

    def _build_widgets(self) -> None:
        pad = dict(padx=10, pady=5)

        # ── Single scenario section ──────────────────────────────
        self._single_section = _PlotSection(
            self,
            label="Single scenario settings:",
            settings=self._settings.single_plot_settings,
            default_config_file="templates/default_plots.yaml",
        )
        self._single_section.frame.pack(fill="x", **pad)

        # ── Comparison section ───────────────────────────────────
        self._comp_section = _PlotSection(
            self,
            label="Scenario comparison settings:",
            settings=self._settings.comparison_plot_settings,
            default_config_file="templates/default_comparison_plots.yaml",
        )
        self._comp_section.frame.pack(fill="x", **pad)

        # ── OK button ────────────────────────────────────────────
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=10, pady=(5, 10))

        ttk.Button(btn_frame, text="OK", width=10, command=self._on_ok).pack(
            side="right",
        )

    # ── Actions ──────────────────────────────────────────────────

    def _on_ok(self) -> None:
        """Save settings and close the dialog."""
        self._settings.single_plot_settings = self._single_section.collect()
        self._settings.comparison_plot_settings = self._comp_section.collect()

        try:
            save_project_settings(self._project_path, self._settings)
        except OSError as exc:
            logger.error("Failed to save plot settings: %s", exc)
            messagebox.showerror(
                "Save error",
                f"Could not save settings:\n{exc}",
                parent=self,
            )

        self.grab_release()
        self.destroy()


# ── Helpers ──────────────────────────────────────────────────────────


def _resolve_config_path(config_file: str) -> Path:
    """Resolve a config file string to an absolute ``Path``.

    If the string looks like a relative path (e.g.
    ``templates/default_plots.yaml``), it is resolved relative to the
    FlexTool root.  Absolute paths are returned as-is.
    """
    p = Path(config_file)
    if p.is_absolute():
        return p
    return _flextool_root() / p


def _parse_int(text: str, fallback: int) -> int:
    """Parse *text* as a non-negative integer, returning *fallback* on failure."""
    try:
        value = int(text)
        return max(0, value)
    except (ValueError, TypeError):
        return fallback


def _register_int_validation(entry: ttk.Entry) -> None:
    """Register a validation callback that only allows digits."""
    vcmd = (entry.winfo_toplevel().register(_validate_int), "%P")
    entry.configure(validate="key", validatecommand=vcmd)


def _validate_int(new_value: str) -> bool:
    """Return ``True`` if *new_value* is empty or consists entirely of digits."""
    if new_value == "":
        return True
    return new_value.isdigit()
