from __future__ import annotations

import logging
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import messagebox, ttk

import yaml

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

        # ── Dispatch checkbox + edit button (comparison section only) ─
        self._dispatch_var: tk.BooleanVar | None = None
        if show_dispatch:
            row = 3
            dispatch_row = ttk.Frame(self.frame)
            dispatch_row.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(4, 2))

            self._dispatch_var = tk.BooleanVar(value=settings.dispatch_plots)
            ttk.Checkbutton(
                dispatch_row, text="Dispatch plots", variable=self._dispatch_var,
            ).pack(side="left")

            ttk.Button(
                dispatch_row, text="Edit dispatch plot config",
                command=self._on_edit_dispatch_config,
            ).pack(side="left", padx=(15, 0))

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
        self._lh = tkfont.nametofont("TkDefaultFont").metrics("linespace")
        cw = tkfont.nametofont("TkDefaultFont").measure("0")

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

    def _on_edit_dispatch_config(self) -> None:
        """Open the dispatch plot config.yaml in a text editor dialog."""
        if self._project_path is None:
            messagebox.showinfo(
                "No project",
                "No project is loaded.",
                parent=self.frame,
            )
            return
        config_path = self._project_path / "output_plot_comparisons" / "config.yaml"
        if not config_path.exists():
            messagebox.showinfo(
                "No config file",
                "Dispatch plot config.yaml does not exist yet.\n\n"
                "It will be created automatically when dispatch plots\n"
                "are generated for the first time.",
                parent=self.frame,
            )
            return
        DispatchConfigEditor(self.frame, config_path)

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
        self.geometry(f"{cw * 110}x{lh * 30}")
        self.resizable(True, True)
        self.minsize(cw * 80, lh * 20)

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
        # Use grid so the two sections sit side-by-side and expand
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # ── Single scenario section (left) ───────────────────────
        self._single_section = _PlotSection(
            self,
            label="Single scenario settings:",
            settings=self._settings.single_plot_settings,
            default_config_file="templates/default_plots.yaml",
        )
        self._single_section.frame.grid(
            row=0, column=0, sticky="nsew", padx=(10, 5), pady=(10, 5),
        )

        # ── Comparison section (right) ───────────────────────────
        self._comp_section = _PlotSection(
            self,
            label="Scenario comparison settings:",
            settings=self._settings.comparison_plot_settings,
            default_config_file="templates/default_comparison_plots.yaml",
            show_dispatch=True,
            project_path=self._project_path,
        )
        self._comp_section.frame.grid(
            row=0, column=1, sticky="nsew", padx=(5, 10), pady=(10, 5),
        )

        # ── OK button ────────────────────────────────────────────
        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(5, 10))

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


class DispatchConfigEditor(tk.Toplevel):
    """Modal text editor for the dispatch plot config.yaml.

    Shows an instruction area above the editable text, validates YAML on
    save, and refuses to save invalid syntax.
    """

    def __init__(self, parent: tk.Misc, config_path: Path) -> None:
        super().__init__(parent)
        self.title(f"Edit dispatch plot config — {config_path.name}")
        self._config_path = config_path

        self.transient(parent)
        self.grab_set()

        # ── Sizing ────────────────────────────────────────────────
        default_font = tkfont.nametofont("TkDefaultFont")
        cw = default_font.measure("0")
        lh = default_font.metrics("linespace")
        mono_font = tkfont.nametofont("TkFixedFont")

        self.geometry(f"{cw * 90}x{lh * 40}")
        self.resizable(True, True)
        self.minsize(cw * 60, lh * 20)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # ── Instructions ──────────────────────────────────────────
        info_frame = ttk.LabelFrame(self, text="Instructions", padding=6)
        info_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))

        info_text = (
            "Define colors for entities and groups — they will persist over "
            "scenarios as best they can.\n"
            "The order defines the stacking order in dispatch plots "
            "(first item is on top).\n"
            "Use named colors from: "
            "https://matplotlib.org/stable/gallery/color/named_colors.html\n"
            "Deleting this file resets all colors."
        )
        ttk.Label(info_frame, text=info_text, wraplength=cw * 80).pack(
            fill="x",
        )

        # ── Text editor ──────────────────────────────────────────
        edit_frame = ttk.Frame(self)
        edit_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=4)
        edit_frame.columnconfigure(0, weight=1)
        edit_frame.rowconfigure(0, weight=1)

        self._text = tk.Text(edit_frame, wrap="none", font=mono_font, undo=True)
        self._text.grid(row=0, column=0, sticky="nsew")

        vscroll = ttk.Scrollbar(edit_frame, orient="vertical", command=self._text.yview)
        vscroll.grid(row=0, column=1, sticky="ns")
        self._text.configure(yscrollcommand=vscroll.set)

        hscroll = ttk.Scrollbar(edit_frame, orient="horizontal", command=self._text.xview)
        hscroll.grid(row=1, column=0, sticky="ew")
        self._text.configure(xscrollcommand=hscroll.set)

        # Load file content
        try:
            content = config_path.read_text(encoding="utf-8")
        except OSError as exc:
            content = f"# Error reading file: {exc}"
        self._text.insert("1.0", content)

        # ── Buttons ───────────────────────────────────────────────
        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(4, 10))

        ttk.Button(btn_frame, text="Cancel", command=self._on_cancel).pack(
            side="right", padx=(5, 0),
        )
        ttk.Button(btn_frame, text="Save and close", command=self._on_save).pack(
            side="right",
        )

        # ── Keyboard shortcuts ────────────────────────────────────
        self.bind("<Escape>", lambda e: self._on_cancel())

        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        # Centre on parent
        self.update_idletasks()
        try:
            px, py = parent.winfo_rootx(), parent.winfo_rooty()
            pw, ph = parent.winfo_width(), parent.winfo_height()
        except Exception:
            px, py, pw, ph = 100, 100, 800, 600
        w, h = self.winfo_width(), self.winfo_height()
        self.geometry(f"+{px + (pw - w) // 2}+{py + (ph - h) // 2}")

        parent.wait_window(self)

    def _on_save(self) -> None:
        """Validate YAML and save if valid."""
        content = self._text.get("1.0", "end-1c")

        # Validate YAML syntax
        try:
            yaml.safe_load(content)
        except yaml.YAMLError as exc:
            messagebox.showerror(
                "Invalid YAML",
                f"The file contains YAML syntax errors and cannot be saved:\n\n{exc}",
                parent=self,
            )
            return

        try:
            self._config_path.write_text(content, encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Save error", f"Could not write file:\n{exc}", parent=self)
            return

        self.grab_release()
        self.destroy()

    def _on_cancel(self) -> None:
        """Close without saving."""
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
