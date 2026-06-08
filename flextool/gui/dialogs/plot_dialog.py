from __future__ import annotations

import logging
import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import messagebox, ttk

from flextool.gui.check_tree import CheckTreeController
from flextool.gui.config_parser import parse_plot_configs
from flextool.gui.dialogs.file_picker import FilePickerDialog
from flextool.gui.data_models import PlotSettings, ProjectSettings
from flextool.gui.settings_io import save_project_settings

logger = logging.getLogger(__name__)


def _flextool_root() -> Path:
    """Return the user's working directory (where project files live).

    Historically this was the repo root (three levels up from ``gui/``).
    In a wheel install that path is inside ``site-packages/`` and not
    user-relevant; CWD is the workspace.
    """
    return Path.cwd()


def _relative_config_path(abs_path: Path) -> str:
    """Return the path relative to CWD when possible."""
    try:
        return str(abs_path.relative_to(Path.cwd()))
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
        show_dispatch: bool = False,
        project_path: Path | None = None,
    ) -> None:
        self._settings = settings
        self._default_config_file = default_config_file
        self._project_path = project_path

        self.frame = ttk.LabelFrame(parent, text=label, padding=10)
        self.frame.columnconfigure(1, weight=1)

        # ── Start time ─────────────────────────────────────────────
        row = 0
        ttk.Label(self.frame, text="Start timestep (integer):").grid(
            row=row, column=0, sticky="w", pady=(0, 4),
        )
        self._start_var = tk.StringVar(value=str(settings.start_time))
        start_entry = ttk.Entry(self.frame, textvariable=self._start_var, width=10)
        start_entry.grid(row=row, column=1, sticky="w", padx=(10, 0), pady=(0, 4))
        _register_int_validation(start_entry)

        # ── Duration ───────────────────────────────────────────────
        row = 1
        ttk.Label(self.frame, text="Duration (number of timesteps):").grid(
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

        # ── Dispatch checkbox (comparison section only) ───────────
        self._dispatch_var: tk.BooleanVar | None = None
        if show_dispatch:
            row = 3
            dispatch_row = ttk.Frame(self.frame)
            dispatch_row.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(4, 2))

            self._dispatch_var = tk.BooleanVar(value=settings.dispatch_plots)
            ttk.Checkbutton(
                dispatch_row, text="Dispatch plots", variable=self._dispatch_var,
            ).pack(side="left")

        # ── "Just one file per plot" checkbox ─────────────────────
        row = 4 if show_dispatch else 3
        self._only_first_file_var = tk.BooleanVar(value=settings.only_first_file)
        ttk.Checkbutton(
            self.frame, text="Just one file per plot", variable=self._only_first_file_var,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(4, 2))

        # ── Active configs list (Treeview with checkboxes) ─────────
        row += 1
        ttk.Label(self.frame, text="Active configs:").grid(
            row=row, column=0, sticky="w", pady=(4, 2),
        )

        row += 1
        self.frame.rowconfigure(row, weight=1)
        from flextool.gui.ui_metrics import get_metrics
        _metrics = get_metrics(self)
        self._lh = _metrics.lh
        cw = _metrics.cw

        config_frame = ttk.Frame(self.frame)
        config_frame.grid(row=row, column=0, columnspan=2, sticky="nsew", pady=(0, 4))
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
        self._config_tree.column("check", width=int(cw * 3.45), minwidth=int(cw * 3.45), stretch=False)
        self._config_tree.column("name", width=cw * 20, minwidth=cw * 10, stretch=True)
        self._config_tree.grid(row=0, column=0, sticky="nsew")

        config_scroll = ttk.Scrollbar(config_frame, orient="vertical", command=self._config_tree.yview)
        config_scroll.grid(row=0, column=1, sticky="ns")
        self._config_tree.configure(yscrollcommand=config_scroll.set)

        self._config_check_ctrl = CheckTreeController(
            self._config_tree,
            check_column="check",
            checked_glyph=self._CHECK_ON,
            unchecked_glyph=self._CHECK_OFF,
        )

        self._populate_configs()

    # ── Public helpers ────────────────────────────────────────────

    # Unicode checkbox characters (same as main window)
    _CHECK_ON = "\u25a0"   # ■
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
        dispatch = self._dispatch_var.get() if self._dispatch_var is not None else self._settings.dispatch_plots
        return PlotSettings(
            start_time=start,
            duration=duration,
            config_file=config_file,
            active_configs=active,
            dispatch_plots=dispatch,
            only_first_file=self._only_first_file_var.get(),
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

    A "Colors, order..." button (top row) opens the shared per-project
    ``plot_settings.yaml`` editor, which applies to ALL plots.  Below it
    a single section holds the plot-generation settings (start-time,
    duration, config-file selection, dispatch toggle and active-config
    checkboxes).  Single-scenario and comparison plots used to take
    separate config files; they now share ``default_plots.yaml``, so one
    section drives both — its values are written to both the single and
    comparison ``PlotSettings`` (still consumed by separate output
    commands) on OK.
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
        from flextool.gui.ui_metrics import get_metrics
        _metrics = get_metrics(self)
        cw: int = _metrics.cw
        lh: int = _metrics.lh

        # ── Dialog size ──────────────────────────────────────────
        # One section now (was two side-by-side), so roughly half-width.
        self.geometry(f"{cw * 104}x{lh * 42}")
        self.resizable(True, True)
        self.minsize(cw * 76, lh * 28)

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
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)  # the settings section expands

        # ── "Colors, order..." on its own row at the top ─────────
        # ``plot_settings.yaml`` (colors / order / sign) applies to ALL
        # plots — single, comparison and dispatch — so its editor sits in
        # its own row above the (shared) plot-generation settings rather
        # than tucked into a corner.
        top_frame = ttk.Frame(self)
        top_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        ttk.Button(
            top_frame, text="Colors, order...", command=self._on_change_colors,
        ).pack(side="left")

        # ── Plot-generation settings (single + comparison share one) ─
        # Single-scenario and comparison plots used to take separate
        # config files; they now share ``default_plots.yaml``, so one
        # section drives both.  ``dispatch_plots`` (comparison-only) is
        # carried from the comparison settings into this merged section.
        section_settings = replace(
            self._settings.single_plot_settings,
            dispatch_plots=self._settings.comparison_plot_settings.dispatch_plots,
        )
        self._section = _PlotSection(
            self,
            label="Plot settings:",
            settings=section_settings,
            default_config_file="templates/default_plots.yaml",
            show_dispatch=True,
            project_path=self._project_path,
        )
        self._section.frame.grid(
            row=1, column=0, sticky="nsew", padx=10, pady=(0, 5),
        )

        # ── Button row ───────────────────────────────────────────
        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(5, 10))

        ttk.Button(btn_frame, text="OK", width=10, command=self._on_ok).pack(
            side="right",
        )

    # ── Actions ──────────────────────────────────────────────────

    def _on_change_colors(self) -> None:
        """Open the shared per-project plot-settings (colors/order) editor.

        Always edits the PROJECT's ``plot_settings.yaml`` (seeding it from
        the bundled default if absent), never the bundled package file.  No
        re-render is needed here — this dialog has no live preview; the
        edited file is used on the next plot generation.
        """
        if self._project_path is None:
            messagebox.showinfo(
                "No project",
                "No project is loaded.",
                parent=self,
            )
            return

        from flextool.gui.dialogs.plot_settings_picker import PlotSettingsPicker
        from flextool.gui.project_utils import seed_plot_settings

        # No live preview in the batch dialog → no on_apply callback.
        project_file = seed_plot_settings(self._project_path)
        PlotSettingsPicker(self, project_file)

    def _on_ok(self) -> None:
        """Save settings and close the dialog."""
        # One section drives both single-scenario and comparison plots;
        # write its values into both PlotSettings (still consumed by
        # separate output commands).  ``variant_durations`` is not edited
        # here, so carry each object's existing values forward.
        collected = self._section.collect()
        self._settings.single_plot_settings = replace(
            collected,
            variant_durations=self._settings.single_plot_settings.variant_durations,
        )
        self._settings.comparison_plot_settings = replace(
            collected,
            variant_durations=self._settings.comparison_plot_settings.variant_durations,
        )

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

    Absolute paths are returned as-is.  A bare name like
    ``default_plots.yaml`` or the legacy
    ``templates/default_plots.yaml`` is resolved against the bundled
    ``schemas/`` package data.  Any other relative path is
    resolved against the user's CWD.
    """
    p = Path(config_file)
    if p.is_absolute():
        return p
    if p.name in {"default_plots.yaml", "default_plot_settings.yaml"}:
        from flextool._resources import package_data_path
        return package_data_path(f"schemas/{p.name}")
    return Path.cwd() / p


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
