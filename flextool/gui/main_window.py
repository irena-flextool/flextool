from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import ttk, messagebox, simpledialog

from flextool.gui.project_utils import (
    create_project,
    get_projects_dir,
    list_projects,
    rename_project,
)
from flextool.gui.settings_io import (
    load_global_settings,
    load_project_settings,
    save_global_settings,
    save_project_settings,
)
from flextool.gui.data_models import GlobalSettings, ProjectSettings, ScenarioInfo
from flextool.gui.input_sources import InputSourceManager
from flextool.gui.scenario_lists import AvailableScenarioManager, ExecutedScenarioManager
from flextool.gui.execution_manager import ExecutionJob, ExecutionManager, JobStatus
from flextool.gui.execution_window import ExecutionWindow
from flextool.gui.output_actions import OutputActionManager
from flextool.gui.result_viewer import ResultViewer
from flextool.gui.dialogs.plot_dialog import PlotDialog
from flextool.gui.error_handling import safe_callback
from flextool.gui.platform_utils import (
    open_file_in_default_app,
    open_folder,
    open_spine_db_editor,
)
from flextool.gui.db_editor_integration import DbEditorManager


logger = logging.getLogger(__name__)


def _unlink_sqlite(db_path: Path) -> None:
    """Delete an SQLite file and its WAL/SHM journals, retrying on Windows lock errors.

    On Windows, SQLAlchemy's connection pool may briefly hold file handles
    after a ``DatabaseMapping`` context manager exits (the engine is not
    disposed in ``__exit__``).  A short retry loop lets the GC and OS
    release the handles before we give up.
    """
    import gc
    import time

    for path in (db_path, db_path.with_suffix(".sqlite-wal"), db_path.with_suffix(".sqlite-shm")):
        if not path.exists():
            continue
        for attempt in range(5):
            try:
                path.unlink()
                break
            except PermissionError:
                if attempt == 4:
                    raise
                gc.collect()
                time.sleep(0.5)


# Unicode checkbox characters for Treeview checkbox simulation
# Using geometric shapes (U+25A1 / U+25A3) which render noticeably larger
# than ballot box characters at the same font size.
CHECK_ON = "\u25a3"   # ▣
CHECK_OFF = "\u25a1"  # □
STATUS_OK = "\u2713"  # ✓
STATUS_ERR = "\u2717"  # ✗
STATUS_EDITING = "\u23f3"  # ⏳

# Animated spinner frames for output action progress indication
_SPINNER_FRAMES = ["\u29d6", "\u29d7"]  # ⧖ ⧗ (hourglass variants)


class MainWindow(tk.Tk):
    """Main application window for FlexTool GUI.

    All widgets are created and placed via the grid geometry manager.
    """

    def __init__(self, initial_theme: str = "dark") -> None:
        super().__init__()

        # ── DPI scaling — must come before any widget/font access ─
        from flextool.gui.platform_utils import (
            apply_dpi_scaling, normalize_default_font_size, scale_theme_fonts,
        )

        dpi_factor = apply_dpi_scaling(self)

        # ── Apply sv_ttk theme before any widgets are created ─────
        import sv_ttk

        if initial_theme == "light":
            sv_ttk.set_theme("light")
        else:
            sv_ttk.set_theme("dark")  # "dark" and "os" both default to dark

        # Rescale sv_ttk's hardcoded pixel-size fonts for high-DPI displays
        scale_theme_fonts(self, dpi_factor)

        # Force a consistent font size across all platforms
        normalize_default_font_size(self, size=10)

        self.title("FlexTool")

        # ── Window icon (works on Windows, macOS, Linux) ──────────
        icon_path = Path(__file__).resolve().parent.parent.parent / "docs" / "irena_flextool_favicon.png"
        if icon_path.exists():
            try:
                self._icon_image = tk.PhotoImage(file=str(icon_path))
                self.iconphoto(True, self._icon_image)
            except Exception:
                pass  # Non-fatal: skip if image can't be loaded

        # ── Font metrics for DPI-aware sizing ─────────────────────
        default_font = tkfont.nametofont("TkDefaultFont")
        self._char_width: int = default_font.measure("0")
        self._line_height: int = default_font.metrics("linespace")
        bold_font = default_font.copy()
        bold_font.configure(weight="bold")
        self._bold_font = bold_font

        # ── Treeview row height and selection visibility ──────────
        style = ttk.Style()
        row_height = self._line_height
        style.configure("Treeview", rowheight=row_height)

        # Make selected rows clearly visible in both dark and light themes
        style.map(
            "Treeview",
            background=[("selected", "#3874c8")],
            foreground=[("selected", "#ffffff")],
        )

        # ── Custom button styles for visual highlighting ──────────
        # Note: Accent.TButton is built into sv_ttk and reliably renders
        # as a visually prominent button.  The old Green.TButton approach
        # did not work because sv_ttk ignores ttk background overrides.
        style.configure("Grey.TButton", foreground="#888888")
        style.map(
            "Grey.TButton",
            foreground=[("active", "#888888"), ("disabled", "#888888")],
        )

        # ── State ──────────────────────────────────────────────────
        self.current_project: str | None = None
        self.global_settings = GlobalSettings()
        self.project_settings = ProjectSettings()
        self.input_source_mgr: InputSourceManager | None = None
        self.avail_scenario_mgr: AvailableScenarioManager | None = None
        self.exec_scenario_mgr: ExecutedScenarioManager | None = None
        self.execution_mgr: ExecutionManager | None = None
        self.execution_window: ExecutionWindow | None = None
        self._result_viewer: ResultViewer | None = None
        self.output_action_mgr: OutputActionManager | None = None
        self._output_action_failed: set[str] = set()
        self._pending_execution_scenarios: list[ScenarioInfo] = []
        self._lock_check_timer_id: str | None = None
        self.db_editor_mgr = DbEditorManager()

        # Pre-conversion state for xlsx→sqlite before execution
        self._xlsx_converting_sources: set[str] = set()
        self._xlsx_pending_scenarios: list[ScenarioInfo] = []
        self._xlsx_conversion_queue: list[tuple[str, Path]] = []

        # Sort mode for each treeview: "alpha" (by name) or "number" (by # column)
        self._input_sort_mode: str = "alpha"
        self._available_sort_mode: str = "alpha"
        self._executed_sort_mode: str = "alpha"

        # ── Font-metric locals used throughout widget construction ──
        cw = self._char_width
        lh = self._line_height

        # Allow the window content to stretch
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # ── Outer frame ──────────────────────────────────────────────
        outer = ttk.Frame(self, padding=10)
        outer.grid(row=0, column=0, sticky="nsew")

        # We'll use a high-level grid inside `outer`.
        # Columns: 0-3 left area, 4 center area, 5-7 right area
        # Let columns with treeviews expand.
        outer.columnconfigure(0, weight=1)   # input sources / available scenarios
        outer.columnconfigure(1, weight=0)   # buttons column
        outer.columnconfigure(2, weight=0)   # auto-generate / plot & exec menus
        outer.columnconfigure(3, weight=0)   # auto-generate continued
        outer.columnconfigure(4, weight=0)   # output actions frame (right-aligned)
        outer.columnconfigure(5, weight=1)   # executed scenarios (lower section)

        # ── Row 0: Project selector ──────────────────────────────────
        row = 0
        ttk.Label(outer, text="Project:").grid(row=row, column=0, sticky="w", padx=(0, 5))

        self.project_combo = ttk.Combobox(outer, state="readonly", width=30)
        self.project_combo.grid(row=row, column=0, sticky="w", padx=(60, 5))
        self.project_combo.bind("<<ComboboxSelected>>", self._on_project_combo_selected)
        self.project_combo.bind("<F2>", self._on_combo_rename)
        self.project_combo.bind("<Double-Button-1>", self._on_combo_rename)

        self.project_menu_btn = ttk.Button(
            outer, text="Project menu", command=self._on_project_menu_btn
        )
        self.project_menu_btn.grid(row=row, column=1, sticky="w", padx=5)

        # ── Theme radio buttons (far right of row 0) ─────────────
        theme_frame = ttk.Frame(outer)
        theme_frame.grid(row=row, column=3, columnspan=3, sticky="e", padx=(20, 0))

        self._theme_var = tk.StringVar(value=initial_theme)
        for text, value in [("OS theme", "os"), ("Dark", "dark"), ("Light", "light")]:
            ttk.Radiobutton(
                theme_frame, text=text, variable=self._theme_var,
                value=value, command=self._on_theme_change,
            ).pack(side="left", padx=3)

        # ── Row 1: Section headers ───────────────────────────────────
        row = 1
        ttk.Label(outer, text="Input sources", font=self._bold_font).grid(
            row=row, column=0, sticky="sw", pady=(10, 2)
        )
        ttk.Label(outer, text="Auto-generate", font=self._bold_font).grid(
            row=row, column=2, columnspan=2, sticky="sw", padx=(20, 0), pady=(10, 2)
        )

        # ── Rows 2-8: Input sources tree + buttons + auto-gen + output status ──
        # --- Input sources Treeview (rows 2-8, col 0) ---
        input_frame = ttk.Frame(outer)
        input_frame.grid(row=2, column=0, rowspan=7, sticky="nsew", padx=(0, 5))
        input_frame.columnconfigure(0, weight=1)
        input_frame.rowconfigure(0, weight=1)

        self.input_sources_tree = ttk.Treeview(
            input_frame,
            columns=("check", "name", "number", "status"),
            show="headings",
            selectmode="extended",
            height=8,
        )
        self.input_sources_tree.heading("check", text="▽")
        self.input_sources_tree.heading(
            "name", text="Name \u25b2",
            command=self._sort_input_by_name,
        )
        self.input_sources_tree.heading(
            "number", text="#",
            command=self._sort_input_by_number,
        )
        self.input_sources_tree.heading("status", text="")
        self.input_sources_tree.column("check", width=cw * 2, minwidth=cw * 2, stretch=False)
        self.input_sources_tree.column("name", width=cw * 25, minwidth=cw * 12)
        self.input_sources_tree.column("number", width=cw * 2, minwidth=cw * 2, stretch=False)
        self.input_sources_tree.column("status", width=cw * 3, minwidth=cw * 3, stretch=False)
        self.input_sources_tree.grid(row=0, column=0, sticky="nsew")

        input_scroll = ttk.Scrollbar(input_frame, orient="vertical", command=self.input_sources_tree.yview)
        input_scroll.grid(row=0, column=1, sticky="ns")
        self._setup_autohide_scrollbar(self.input_sources_tree, input_scroll)

        # Bind click for checkbox toggling (filtering) and selection change (button state)
        self.input_sources_tree.bind("<Button-1>", self._on_input_source_click)
        self.input_sources_tree.bind("<Double-1>", self._on_input_source_dblclick)
        self.input_sources_tree.bind("<B1-Motion>", self._on_tree_drag_select)
        self.input_sources_tree.bind("<<TreeviewSelect>>", lambda _e: self._update_input_button_states())
        self.input_sources_tree.bind("<Button-3>", self._on_input_source_right_click)

        # --- Input source buttons (col 1, rows 2-8) ---
        btn_col = 1
        self.add_source_btn = ttk.Button(
            outer, text="Add", width=8, command=self._on_add_source
        )
        self.add_source_btn.grid(row=2, column=btn_col, sticky="nw", padx=5, pady=2)

        self.edit_source_btn = ttk.Button(
            outer, text="Edit", width=8, command=self._on_edit_source, state="disabled"
        )
        self.edit_source_btn.grid(row=4, column=btn_col, sticky="nw", padx=5, pady=2)

        self.convert_source_btn = ttk.Button(
            outer, text="Convert", width=8, command=self._on_convert_source, state="disabled"
        )
        self.convert_source_btn.grid(row=5, column=btn_col, sticky="nw", padx=5, pady=2)

        self.delete_source_btn = ttk.Button(
            outer, text="Delete", width=8, command=self._on_delete_source, state="disabled"
        )
        self.delete_source_btn.grid(row=6, column=btn_col, sticky="nw", padx=5, pady=2)

        self.refresh_btn = ttk.Button(
            outer, text="Refresh", width=8, command=self._on_refresh_sources
        )
        self.refresh_btn.grid(row=8, column=btn_col, sticky="nw", padx=5, pady=2)

        # --- Auto-generate checkboxes (col 2-3, rows 2-6) ---
        self.auto_scen_plots_var = tk.BooleanVar(value=True)
        self.auto_scen_excels_var = tk.BooleanVar(value=False)
        self.auto_scen_csvs_var = tk.BooleanVar(value=True)
        self.auto_comp_plots_var = tk.BooleanVar(value=True)
        self.auto_comp_excel_var = tk.BooleanVar(value=False)

        auto_frame = ttk.Frame(outer)
        auto_frame.grid(row=2, column=2, rowspan=5, columnspan=2, sticky="nw", padx=(20, 10))

        self.auto_scen_plots_cb = ttk.Checkbutton(
            auto_frame, text="Scen. plots", variable=self.auto_scen_plots_var
        )
        self.auto_scen_plots_cb.grid(row=0, column=0, sticky="w", pady=2)

        self.auto_scen_excels_cb = ttk.Checkbutton(
            auto_frame, text="Scen. Excels", variable=self.auto_scen_excels_var
        )
        self.auto_scen_excels_cb.grid(row=1, column=0, sticky="w", pady=2)

        self.auto_scen_csvs_cb = ttk.Checkbutton(
            auto_frame, text="Scen. csvs", variable=self.auto_scen_csvs_var
        )
        self.auto_scen_csvs_cb.grid(row=2, column=0, sticky="w", pady=2)

        self.auto_comp_plots_cb = ttk.Checkbutton(
            auto_frame, text="Comp. plots", variable=self.auto_comp_plots_var
        )
        self.auto_comp_plots_cb.grid(row=3, column=0, sticky="w", pady=2)

        self.auto_comp_excel_cb = ttk.Checkbutton(
            auto_frame, text="Comp. Excel", variable=self.auto_comp_excel_var
        )
        self.auto_comp_excel_cb.grid(row=4, column=0, sticky="w", pady=2)

        # Trace auto-generate vars to save settings on toggle
        self.auto_scen_plots_var.trace_add("write", self._on_auto_gen_toggled)
        self.auto_scen_excels_var.trace_add("write", self._on_auto_gen_toggled)
        self.auto_scen_csvs_var.trace_add("write", self._on_auto_gen_toggled)
        self.auto_comp_plots_var.trace_add("write", self._on_auto_gen_toggled)
        self.auto_comp_excel_var.trace_add("write", self._on_auto_gen_toggled)

        # --- Png settings and Execution jobs buttons (col 2-3, rows 7-8) ---
        self.plot_menu_btn = ttk.Button(
            outer, text="Png settings", width=14,
            command=self._on_plot_menu,
        )
        self.plot_menu_btn.grid(row=7, column=2, columnspan=2, sticky="nw", padx=(20, 10), pady=2)

        self.execution_menu_btn = ttk.Button(
            outer, text="Execution jobs", width=14,
            command=self._on_execution_menu,
        )
        self.execution_menu_btn.grid(row=8, column=2, columnspan=2, sticky="nw", padx=(20, 10), pady=2)

        # --- Output actions LabelFrame (col 5, rows 2-8, right-aligned, above executed scenarios) ---
        # Use tk.LabelFrame (not ttk) so that background color changes apply
        # uniformly to the entire frame interior, not just the label row.
        # Pull the background color from the ttk theme so it matches dark/light mode.
        theme_bg = style.lookup("TFrame", "background") or self.cget("background")
        theme_fg = style.lookup("TLabel", "foreground") or "white"
        self.output_frame = tk.LabelFrame(
            outer, text="Output actions", padx=5, pady=5,
            bg=theme_bg, fg=theme_fg,
        )
        self.output_frame.grid(
            row=2, column=5, rowspan=7, sticky="se", padx=(10, 0), pady=2,
        )
        # Store default bg so we can revert the green tint later
        self._output_frame_default_bg = theme_bg


        output_info: list[tuple[str, str, str | None]] = [
            ("Re-plot scenarios", "scen_plots", "Show"),
            ("Scenarios to Excel", "scen_excel", "Show"),
            ("Scenarios to csvs", "scen_csvs", "Show"),
            ("Comparison pngs", "comp_plots", "Show"),
            ("Comparison to Excel", "comp_excel", "Open"),
        ]

        self.output_status_labels: dict[str, ttk.Button] = {}
        self.output_action_btns: dict[str, ttk.Button] = {}
        self._output_spinners: dict[str, ttk.Label] = {}
        self._spinner_timer_ids: dict[str, str] = {}

        # Map keys to their generation handler method names
        _gen_commands: dict[str, str] = {
            "scen_plots": "_on_gen_scen_plots",
            "scen_excel": "_on_gen_scen_excel",
            "scen_csvs": "_on_gen_scen_csvs",
            "comp_plots": "_on_gen_comp_plots",
            "comp_excel": "_on_gen_comp_excel",
        }
        _show_commands: dict[str, str] = {
            "scen_plots": "_on_show_scen_plots",
            "scen_excel": "_on_show_scen_excel",
            "scen_csvs": "_on_show_scen_csvs",
            "comp_plots": "_on_show_comp_plots",
            "comp_excel": "_on_show_comp_excel",
        }

        for i, (label_text, key, action_text) in enumerate(output_info):
            status_btn = ttk.Button(
                self.output_frame, text=label_text, width=20,
                command=getattr(self, _gen_commands[key]),
            )
            status_btn.grid(row=i, column=0, sticky="w", padx=(0, 2), pady=2)
            self.output_status_labels[key] = status_btn

            spinner_label = ttk.Label(self.output_frame, text="  ", width=2, anchor="center")
            spinner_label.grid(row=i, column=1, padx=2, pady=2)
            self._output_spinners[key] = spinner_label

            if action_text is not None:
                action_btn = ttk.Button(
                    self.output_frame, text=action_text, width=5,
                    command=getattr(self, _show_commands[key]),
                )
                action_btn.grid(row=i, column=2, sticky="w", padx=(2, 0), pady=2)
                self.output_action_btns[key] = action_btn

        # Results viewer button (opens the ResultViewer window)
        next_row = len(output_info)
        self.view_results_btn = ttk.Button(
            self.output_frame, text="Results viewer", width=20,
            command=self._on_view_results,
        )
        self.view_results_btn.grid(
            row=next_row, column=0, columnspan=3, sticky="w", padx=(0, 2), pady=(18, 2),
        )

        # ── Separator ────────────────────────────────────────────────
        sep = ttk.Separator(outer, orient="horizontal")
        sep.grid(row=9, column=0, columnspan=6, sticky="ew", pady=10)

        # ── Row 10: Scenario section headers ─────────────────────────
        row = 10
        ttk.Label(outer, text="Available scenarios", font=self._bold_font).grid(
            row=row, column=0, columnspan=2, sticky="sw", pady=(0, 2)
        )
        ttk.Label(outer, text="Executed scenarios", font=self._bold_font).grid(
            row=row, column=2, columnspan=4, sticky="sw", padx=(20, 0), pady=(0, 2)
        )

        # ── Row 11: Available scenarios Treeview ─────────────────────
        row = 11
        # Make the scenario rows expand vertically
        outer.rowconfigure(row, weight=1)

        avail_frame = ttk.Frame(outer)
        avail_frame.grid(row=row, column=0, columnspan=2, sticky="nsew", padx=(0, 5))
        avail_frame.columnconfigure(0, weight=1)
        avail_frame.rowconfigure(0, weight=1)

        self.available_tree = ttk.Treeview(
            avail_frame,
            columns=("check", "source_num", "scenario_name"),
            show="headings",
            selectmode="extended",
            height=8,
        )
        self.available_tree.heading("check", text="▽")
        self.available_tree.heading(
            "source_num", text="#",
            command=self._sort_available_by_number,
        )
        self.available_tree.heading(
            "scenario_name", text="Scenario \u25b2",
            command=self._sort_available_by_name,
        )
        self.available_tree.column("check", width=cw * 3, minwidth=cw * 3, stretch=False)
        self.available_tree.column("source_num", width=cw * 3, minwidth=cw * 3, stretch=False)
        self.available_tree.column("scenario_name", width=cw * 25, minwidth=cw * 12, stretch=True)
        self.available_tree.grid(row=0, column=0, sticky="nsew")

        avail_scroll = ttk.Scrollbar(avail_frame, orient="vertical", command=self.available_tree.yview)
        avail_scroll.grid(row=0, column=1, sticky="ns")
        self._setup_autohide_scrollbar(self.available_tree, avail_scroll)

        self.available_tree.bind("<Button-1>", self._on_available_click)
        self.available_tree.bind("<B1-Motion>", self._on_tree_drag_select)
        self.available_tree.bind("<space>", self._on_available_space)
        self.available_tree.bind("<Button-3>", self._on_available_right_click)

        # ── Row 11: Executed scenarios Treeview ──────────────────────
        exec_frame = ttk.Frame(outer)
        exec_frame.grid(row=row, column=2, columnspan=4, sticky="nsew", padx=(20, 0))
        exec_frame.columnconfigure(0, weight=1)
        exec_frame.rowconfigure(0, weight=1)

        self.executed_tree = ttk.Treeview(
            exec_frame,
            columns=("check", "source_num", "scenario_name", "view", "timestamp"),
            show="headings",
            selectmode="extended",
            height=8,
        )
        self.executed_tree.heading("check", text="▽")
        self.executed_tree.heading(
            "source_num", text="#",
            command=self._sort_executed_by_number,
        )
        self.executed_tree.heading(
            "scenario_name", text="Scenario \u25b2",
            command=self._sort_executed_by_name,
        )
        self.executed_tree.heading("view", text="")
        self.executed_tree.heading(
            "timestamp", text="Timestamp",
            command=self._sort_executed_by_timestamp,
        )
        self.executed_tree.column("check", width=cw * 3, minwidth=cw * 3, stretch=False)
        self.executed_tree.column("source_num", width=cw * 3, minwidth=cw * 3, stretch=False)
        self.executed_tree.column("scenario_name", width=cw * 25, minwidth=cw * 12, stretch=True)
        self.executed_tree.column("view", width=cw * 7, minwidth=cw * 6, stretch=False, anchor="center")
        self.executed_tree.column("timestamp", width=cw * 16, minwidth=cw * 16, stretch=False)
        self.executed_tree.grid(row=0, column=0, sticky="nsew")

        # No row-level tag for View — Treeview tags color the entire row.
        # The ▶ View text is visually distinct on its own.

        exec_scroll = ttk.Scrollbar(exec_frame, orient="vertical", command=self.executed_tree.yview)
        exec_scroll.grid(row=0, column=1, sticky="ns")
        self._setup_autohide_scrollbar(self.executed_tree, exec_scroll)

        self.executed_tree.bind("<Button-1>", self._on_executed_click)
        self.executed_tree.bind("<B1-Motion>", self._on_tree_drag_select)
        self.executed_tree.bind("<space>", self._on_executed_space)
        self.executed_tree.bind("<<TreeviewSelect>>", self._on_executed_selection_changed)
        self.executed_tree.bind("<Button-3>", self._on_executed_right_click)

        # ── Row 12: Bottom action buttons ────────────────────────────
        row = 12
        bottom_left = ttk.Frame(outer)
        bottom_left.grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 0))

        self.select_all_btn = ttk.Button(
            bottom_left, text="Select all [A]",
            command=self._on_select_all,
        )
        self.select_all_btn.grid(row=0, column=0, padx=(0, 10))

        self.check_btn = ttk.Button(
            bottom_left, text="Check/uncheck\nselected [Space]",
            command=self._on_check_selected,
        )
        self.check_btn.grid(row=0, column=1, padx=(0, 10))

        self.add_to_execution_btn = ttk.Button(
            bottom_left, text="Add checked scenarios to\nthe execution list [F9]",
            command=self._on_add_to_execution,
        )
        self.add_to_execution_btn.grid(row=0, column=2, padx=(0, 10))

        bottom_right = ttk.Frame(outer)
        bottom_right.grid(row=row, column=2, columnspan=5, sticky="e", pady=(8, 0))

        self.check_executed_btn = ttk.Button(
            bottom_right, text="Check/uncheck\nall [E]",
            command=self._on_check_executed,
        )
        self.check_executed_btn.grid(row=0, column=0, padx=(0, 10))

        self.delete_results_btn = ttk.Button(
            bottom_right, text="Delete selected\nresults irrevocably",
            command=self._on_delete_results,
        )
        self.delete_results_btn.grid(row=0, column=1)

        # ── Keyboard shortcuts ──────────────────────────────────────
        self.bind_all("<Alt-Key-c>", lambda e: self._on_check_selected())
        self.bind_all("<F9>", lambda e: self._on_add_to_execution())
        self.bind_all("<Control-Key-a>", self._on_ctrl_a)
        self.bind_all("<Control-Key-A>", self._on_ctrl_a)
        # Plain 'a' also selects all (only in Treeviews, not text entries)
        self.bind_all("<Key-a>", self._on_key_a)
        self.bind_all("<Key-A>", self._on_key_a)
        # 'e' toggles checkboxes on selected executed scenarios
        self.bind_all("<Key-e>", self._on_key_e)
        self.bind_all("<Key-E>", self._on_key_e)

        # ── Window close handler ─────────────────────────────────────
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── Window sizing: compute from actual widget layout ─────────
        self.update_idletasks()
        nat_width = self.winfo_reqwidth()
        nat_height = self.winfo_reqheight()
        screen_h = self.winfo_screenheight()
        # Use natural width; for height, fill the screen minus taskbar space
        win_height = max(nat_height, screen_h - lh * 4)
        self.geometry(f"{nat_width}x{win_height}+0+0")
        self.minsize(nat_width, nat_height)

        # ── Startup logic ────────────────────────────────────────────
        self._startup()

    # ── Auto-hide scrollbar helper ──────────────────────────────────

    @staticmethod
    def _setup_autohide_scrollbar(
        tree: ttk.Treeview,
        scrollbar: ttk.Scrollbar,
    ) -> None:
        """Configure *scrollbar* to appear only when *tree* content overflows.

        The scrollbar must already be placed via ``grid()``.  Its grid
        configuration is captured once so that ``grid_remove()`` /
        ``grid(**info)`` can toggle visibility without losing placement.
        """
        grid_info: dict = scrollbar.grid_info()

        def _on_scroll_set(first: str, last: str) -> None:
            scrollbar.set(first, last)
            if float(first) <= 0.0 and float(last) >= 1.0:
                scrollbar.grid_remove()
            else:
                scrollbar.grid(**grid_info)

        tree.configure(yscrollcommand=_on_scroll_set)
        # Hide immediately if nothing to scroll yet
        scrollbar.grid_remove()

    # ── Startup ──────────────────────────────────────────────────────

    def _startup(self) -> None:
        """Initialise project state on application start."""
        projects_dir = get_projects_dir()
        projects_dir.mkdir(parents=True, exist_ok=True)

        self.global_settings = load_global_settings(projects_dir)
        self._refresh_project_combo()

        recent = self.global_settings.recent_project
        if recent and (projects_dir / recent).is_dir():
            self._switch_project(recent)
        else:
            # No valid recent project -- highlight Project menu and show dialog
            self.project_menu_btn.configure(style="Accent.TButton")
            self.after(100, self._show_project_dialog_if_needed)

    def _show_project_dialog_if_needed(self) -> None:
        """Open the ProjectDialog if no project is currently loaded."""
        if self.current_project is not None:
            return
        self._open_project_dialog()

    # ── Project combo events ─────────────────────────────────────────

    def _refresh_project_combo(self) -> None:
        """Repopulate the project dropdown with current project list."""
        projects = list_projects()
        self.project_combo["values"] = projects
        if self.current_project and self.current_project in projects:
            self.project_combo.set(self.current_project)
        elif not self.current_project:
            self.project_combo.set("")

    def _on_project_combo_selected(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        selected = self.project_combo.get()
        if selected and selected != self.current_project:
            self._switch_project(selected)

    def _on_combo_rename(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        """Trigger rename of the current project via a simple dialog."""
        if not self.current_project:
            messagebox.showinfo("No project", "No project is currently loaded.")
            return

        new_name = simpledialog.askstring(
            "Rename project",
            f"Rename '{self.current_project}' to:",
            initialvalue=self.current_project,
            parent=self,
        )
        if not new_name or new_name.strip() == self.current_project:
            return
        new_name = new_name.strip()

        try:
            rename_project(self.current_project, new_name)
        except FileExistsError:
            messagebox.showwarning(
                "Already exists",
                f"A project named '{new_name}' already exists.",
            )
            return
        except FileNotFoundError:
            messagebox.showerror(
                "Not found",
                f"Project '{self.current_project}' no longer exists.",
            )
            return
        except OSError as exc:
            messagebox.showerror("Error", str(exc))
            return

        # Update global settings if the renamed project was the recent one
        if self.global_settings.recent_project == self.current_project:
            self.global_settings.recent_project = new_name
            save_global_settings(get_projects_dir(), self.global_settings)

        self._switch_project(new_name)

    # ── Project menu button ──────────────────────────────────────────

    def _on_project_menu_btn(self) -> None:
        self._open_project_dialog()

    def _open_project_dialog(self) -> None:
        """Open the ProjectDialog and handle its result."""
        # Import here to avoid circular imports at module level
        from flextool.gui.dialogs.project_dialog import ProjectDialog

        dlg = ProjectDialog(self)
        if dlg.result:
            self._switch_project(dlg.result)

    # ── Theme toggle ──────────────────────────────────────────────

    def _on_theme_change(self) -> None:
        """Handle theme radio button change: save setting and inform user."""
        new_theme = self._theme_var.get()
        self.global_settings.theme = new_theme
        save_global_settings(get_projects_dir(), self.global_settings)
        messagebox.showinfo("Theme", "Restart to update theme.")

    # ── Project switching ────────────────────────────────────────────

    def _switch_project(self, name: str) -> None:
        """Switch to the project with the given *name*."""
        # Close execution window and manager from the previous project
        if self.execution_mgr is not None and self.execution_mgr.has_pending_or_running():
            result = messagebox.askyesno(
                "Jobs running",
                "There are running or pending execution jobs for the current project.\n"
                "Kill all jobs and switch project?",
                parent=self,
            )
            if not result:
                # Reset the combo back – it already shows the new name
                self.project_combo.set(self.current_project or "")
                return
            self.execution_mgr.kill_all()

        if self.execution_window is not None and self.execution_window.winfo_exists():
            self.execution_window.destroy()
        self.execution_window = None
        if self._result_viewer is not None and self._result_viewer.winfo_exists():
            self._result_viewer.destroy()
        self._result_viewer = None
        self.execution_mgr = None
        self.output_action_mgr = None

        self.current_project = name

        # Remove green highlight from Project menu button
        self.project_menu_btn.configure(style="TButton")

        # Update combo
        self._refresh_project_combo()
        self.project_combo.set(name)

        # Update window title
        self.title(f"FlexTool \u2014 {name}")

        # Save as recent
        self.global_settings.recent_project = name
        save_global_settings(get_projects_dir(), self.global_settings)

        # Load project settings
        projects_dir = get_projects_dir()
        project_path = projects_dir / name
        self.project_settings = load_project_settings(project_path)

        # Create input source manager and scenario managers
        self.input_source_mgr = InputSourceManager(project_path, self.project_settings)
        self.avail_scenario_mgr = AvailableScenarioManager(self.project_settings)
        self.exec_scenario_mgr = ExecutedScenarioManager(project_path)
        self.output_action_mgr = OutputActionManager(
            project_path=project_path,
            settings=self.project_settings,
            execution_mgr=self.execution_mgr,
            on_complete=self._on_output_action_complete,
        )

        # Sync auto-generate checkboxes with loaded settings
        self._load_auto_gen_vars()

        self._clear_all_lists()
        self._refresh_input_sources()
        self._refresh_executed_scenarios()

        # Start periodic lock file checking
        self._start_lock_check_timer()

    def _clear_all_lists(self) -> None:
        """Clear all treeview widgets."""
        for item in self.input_sources_tree.get_children():
            self.input_sources_tree.delete(item)
        for item in self.available_tree.get_children():
            self.available_tree.delete(item)
        for item in self.executed_tree.get_children():
            self.executed_tree.delete(item)

    # ── Window close ─────────────────────────────────────────────────

    def _on_close(self) -> None:
        """Save state and close the application."""
        # Warn if executions are still active
        if self.execution_mgr is not None and self.execution_mgr.has_pending_or_running():
            result = messagebox.askyesno(
                "Jobs running",
                "There are running or pending execution jobs.\n"
                "Kill all jobs and close?",
                parent=self,
            )
            if not result:
                return

        # Kill all running subprocesses (cleanup is idempotent)
        try:
            if self.execution_mgr is not None:
                self.execution_mgr.cleanup()
        except Exception:
            logger.exception("Error killing execution jobs during close")

        # Cancel periodic lock check timer
        try:
            if self._lock_check_timer_id is not None:
                self.after_cancel(self._lock_check_timer_id)
                self._lock_check_timer_id = None
        except Exception:
            pass

        # Clear xlsx pre-conversion state
        self._xlsx_converting_sources.clear()
        self._xlsx_conversion_queue.clear()
        self._xlsx_pending_scenarios.clear()

        # Close execution window if open
        try:
            if self.execution_window is not None and self.execution_window.winfo_exists():
                self.execution_window.destroy()
        except Exception:
            pass
        self.execution_window = None
        self.execution_mgr = None

        # Close result viewer if open
        try:
            if self._result_viewer is not None and self._result_viewer.winfo_exists():
                self._result_viewer.destroy()
        except Exception:
            pass
        self._result_viewer = None

        # Save all current settings
        if self.current_project:
            try:
                self.global_settings.recent_project = self.current_project
                save_global_settings(get_projects_dir(), self.global_settings)

                # Persist auto-generate settings
                self.project_settings.auto_generate_scen_plots = self.auto_scen_plots_var.get()
                self.project_settings.auto_generate_scen_excels = self.auto_scen_excels_var.get()
                self.project_settings.auto_generate_scen_csvs = self.auto_scen_csvs_var.get()
                self.project_settings.auto_generate_comp_plots = self.auto_comp_plots_var.get()
                self.project_settings.auto_generate_comp_excel = self.auto_comp_excel_var.get()

                # Persist scenario order
                if self.avail_scenario_mgr:
                    self.project_settings.scenario_order = (
                        self.avail_scenario_mgr.get_order()
                    )

                # Persist checkbox states
                self._collect_checked_input_sources()
                self._collect_checked_available_scenarios()
                self._collect_checked_executed_scenarios()

                project_path = get_projects_dir() / self.current_project
                save_project_settings(project_path, self.project_settings)
            except Exception:
                logger.exception("Error saving settings during close")

        try:
            self.destroy()
        except Exception:
            pass

    # ── Periodic lock file checking ───────────────────────────────────

    def _start_lock_check_timer(self) -> None:
        """Start (or restart) the periodic lock file check timer."""
        if self._lock_check_timer_id is not None:
            self.after_cancel(self._lock_check_timer_id)
        self._lock_check_timer_id = self.after(5000, self._check_lock_files)

    def _check_lock_files(self) -> None:
        """Periodically check lock file status for existing input sources.

        Updates the status indicator column in the input sources Treeview
        without doing a full directory re-scan.
        """
        if self.input_source_mgr is None:
            self._lock_check_timer_id = None
            return

        changed = False
        for item in self.input_sources_tree.get_children():
            values = self.input_sources_tree.item(item, "values")
            if not values:
                continue
            source_name = values[1]
            old_status_char = values[3]
            filepath = self.input_source_mgr.input_dir / source_name

            if not filepath.exists():
                continue

            is_locked = self.input_source_mgr._check_lock(filepath)
            # For sqlite files, also check if a tracked editor process is running
            if (
                not is_locked
                and source_name.lower().endswith(".sqlite")
                and self.db_editor_mgr.is_editor_running(source_name)
            ):
                is_locked = True
            # Also treat sources being pre-converted as locked
            if not is_locked and source_name in self._xlsx_converting_sources:
                is_locked = True
            if is_locked and old_status_char != STATUS_EDITING:
                self.input_sources_tree.set(item, "status", STATUS_EDITING)
                changed = True
            elif not is_locked and old_status_char == STATUS_EDITING:
                # Lock was released -- mark as OK (will be verified on next full refresh)
                self.input_sources_tree.set(item, "status", STATUS_OK)
                changed = True

        if changed:
            self._update_input_button_states()
            # Update editing source background in available scenarios
            self._update_available_scenario_tags()

        # Re-schedule
        self._lock_check_timer_id = self.after(5000, self._check_lock_files)

    # ── Button enable/disable during operations ───────────────────────

    def _set_buttons_enabled(self, enabled: bool) -> None:
        """Enable or disable major action buttons.

        This is a safeguard against clicking buttons while the GUI is
        processing an operation.
        """
        state = "normal" if enabled else "disabled"
        buttons = [
            self.add_source_btn,
            self.edit_source_btn,
            self.convert_source_btn,
            self.delete_source_btn,
            self.refresh_btn,
            self.add_to_execution_btn,
            self.delete_results_btn,
            self.plot_menu_btn,
            self.execution_menu_btn,
            self.view_results_btn,
        ]
        for btn in buttons:
            try:
                btn.configure(state=state)
            except Exception:
                pass
        # Output status buttons
        for btn in self.output_status_labels.values():
            try:
                btn.configure(state=state)
            except Exception:
                pass
        for btn in self.output_action_btns.values():
            try:
                btn.configure(state=state)
            except Exception:
                pass

    # ── Drag-to-select for Treeviews ────────────────────────────────

    def _on_tree_drag_select(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """Extend selection to the item under the cursor during B1 drag."""
        tree = event.widget
        if not isinstance(tree, ttk.Treeview):
            return
        item = tree.identify_row(event.y)
        if item:
            tree.selection_add(item)

    # ── Treeview checkbox toggle handlers ────────────────────────────
    # These detect a click on the "check" column and toggle the character.

    def _toggle_check(self, tree: ttk.Treeview, item: str, col: str) -> None:
        """Toggle a checkbox character in *tree* for *item* in *col*."""
        current = tree.set(item, col)
        new_value = CHECK_OFF if current == CHECK_ON else CHECK_ON
        tree.set(item, col, new_value)

    # ── Column sort handlers ──────────────────────────────────────

    def _update_sort_headings(
        self,
        tree: ttk.Treeview,
        active_col: str,
        col_labels: dict[str, str],
    ) -> None:
        """Update heading text to show ▲ indicator on the active sort column."""
        for col, label in col_labels.items():
            if col == active_col:
                tree.heading(col, text=f"{label} \u25b2")
            else:
                tree.heading(col, text=label)

    _INPUT_COL_LABELS: dict[str, str] = {"name": "Name", "number": "#"}
    _AVAIL_COL_LABELS: dict[str, str] = {"scenario_name": "Scenario", "source_num": "#"}
    _EXEC_COL_LABELS: dict[str, str] = {
        "scenario_name": "Scenario", "source_num": "#", "timestamp": "Timestamp",
    }

    def _sort_input_by_name(self) -> None:
        """Sort input sources tree alphabetically by name."""
        self._input_sort_mode = "alpha"
        self._update_sort_headings(self.input_sources_tree, "name", self._INPUT_COL_LABELS)
        self._sort_tree_items(self.input_sources_tree, col_index=1, numeric=False)

    def _sort_input_by_number(self) -> None:
        """Sort input sources tree by source number."""
        self._input_sort_mode = "number"
        self._update_sort_headings(self.input_sources_tree, "number", self._INPUT_COL_LABELS)
        self._sort_tree_items(self.input_sources_tree, col_index=2, numeric=True)

    def _sort_available_by_name(self) -> None:
        """Sort available scenarios tree alphabetically by scenario name."""
        self._available_sort_mode = "alpha"
        self._update_sort_headings(self.available_tree, "scenario_name", self._AVAIL_COL_LABELS)
        self._sort_tree_items(self.available_tree, col_index=2, numeric=False)

    def _sort_available_by_number(self) -> None:
        """Sort available scenarios tree by source number."""
        self._available_sort_mode = "number"
        self._update_sort_headings(self.available_tree, "source_num", self._AVAIL_COL_LABELS)
        self._sort_tree_items(self.available_tree, col_index=1, numeric=True)

    def _sort_executed_by_name(self) -> None:
        """Sort executed scenarios tree alphabetically by scenario name."""
        self._executed_sort_mode = "alpha"
        self._update_sort_headings(self.executed_tree, "scenario_name", self._EXEC_COL_LABELS)
        self._sort_tree_items(self.executed_tree, col_index=2, numeric=False)

    def _sort_executed_by_number(self) -> None:
        """Sort executed scenarios tree by source number."""
        self._executed_sort_mode = "number"
        self._update_sort_headings(self.executed_tree, "source_num", self._EXEC_COL_LABELS)
        self._sort_tree_items(self.executed_tree, col_index=1, numeric=True, secondary_col=4)

    def _sort_executed_by_timestamp(self) -> None:
        """Sort executed scenarios tree by timestamp."""
        self._executed_sort_mode = "timestamp"
        self._update_sort_headings(self.executed_tree, "timestamp", self._EXEC_COL_LABELS)
        self._sort_tree_items(self.executed_tree, col_index=4, numeric=False)

    def _sort_tree_items(
        self,
        tree: ttk.Treeview,
        col_index: int,
        numeric: bool,
        secondary_col: int | None = None,
    ) -> None:
        """Re-sort existing treeview items in place by the given column."""
        items = [(tree.item(iid, "values"), iid) for iid in tree.get_children()]
        if numeric:
            def sort_key(pair: tuple) -> tuple:
                vals = pair[0]
                try:
                    primary = int(vals[col_index])
                except (ValueError, IndexError):
                    primary = 0
                secondary = ""
                if secondary_col is not None:
                    try:
                        secondary = str(vals[secondary_col])
                    except IndexError:
                        pass
                return (primary, secondary)
        else:
            def sort_key(pair: tuple) -> tuple:  # type: ignore[no-redef]
                vals = pair[0]
                try:
                    return (str(vals[col_index]).lower(),)
                except IndexError:
                    return ("",)

        items.sort(key=sort_key)
        for idx, (_vals, iid) in enumerate(items):
            tree.move(iid, "", idx)

    def _on_input_source_click(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        tree = self.input_sources_tree
        region = tree.identify_region(event.x, event.y)
        if region != "cell":
            return
        column = tree.identify_column(event.x)
        if column == "#1":  # "check" column
            item = tree.identify_row(event.y)
            if item:
                self._toggle_check(tree, item, "check")
                self._update_available_scenarios()
                self._save_checked_input_sources()

    def _on_input_source_dblclick(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """Double-click on an input source row opens it for editing."""
        item = self.input_sources_tree.identify_row(event.y)
        if item:
            # Select the item so _on_edit_source sees it as selected
            self.input_sources_tree.selection_set(item)
            self._on_edit_source()

    def _on_available_click(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        tree = self.available_tree
        region = tree.identify_region(event.x, event.y)
        if region != "cell":
            return
        column = tree.identify_column(event.x)
        if column == "#1":  # "check" column
            item = tree.identify_row(event.y)
            if item:
                self._toggle_check(tree, item, "check")
                self._update_add_to_execution_style()
                self._save_checked_available_scenarios()

    def _on_executed_click(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        tree = self.executed_tree
        region = tree.identify_region(event.x, event.y)
        if region != "cell":
            return
        column = tree.identify_column(event.x)
        if column == "#1":  # "check" column
            item = tree.identify_row(event.y)
            if item:
                self._toggle_check(tree, item, "check")
                self._update_output_status()
                self._save_checked_executed_scenarios()
        elif column == "#4":  # "view" column
            item = tree.identify_row(event.y)
            if item:
                values = tree.item(item, "values")
                if values and values[3]:
                    scenario_name = values[2]
                    self._view_scenario_plots(scenario_name)

    # ── Right-click context menus ──────────────────────────────────

    def _on_input_source_right_click(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """Show context menu for input sources tree."""
        tree = self.input_sources_tree
        item = tree.identify_row(event.y)
        if item:
            # Select the right-clicked row if not already selected
            if item not in tree.selection():
                tree.selection_set(item)
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Edit", command=self._on_edit_source)
        menu.add_command(label="Convert", command=self._on_convert_source)
        menu.add_separator()
        menu.add_command(label="Delete", command=self._on_delete_source)
        menu.add_separator()
        menu.add_command(label="Refresh", command=self._on_refresh_sources)
        menu.tk_popup(event.x_root, event.y_root)

    def _on_available_right_click(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """Show context menu for available scenarios tree."""
        tree = self.available_tree
        item = tree.identify_row(event.y)
        if item:
            if item not in tree.selection():
                tree.selection_set(item)
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Check/uncheck selected", command=self._on_check_selected)
        menu.add_separator()
        menu.add_command(
            label="Add selected to execution jobs",
            command=self._on_add_selected_to_execution,
        )
        menu.tk_popup(event.x_root, event.y_root)

    def _on_executed_right_click(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """Show context menu for executed scenarios tree."""
        tree = self.executed_tree
        item = tree.identify_row(event.y)
        if item:
            if item not in tree.selection():
                tree.selection_set(item)

        menu = tk.Menu(self, tearoff=0)
        menu.add_command(
            label="Check/uncheck selected",
            command=self._on_executed_space_from_menu,
        )
        menu.add_separator()
        # View — only if a single scenario is right-clicked and has plots
        if item:
            values = tree.item(item, "values")
            if values and values[3]:  # has view text
                scenario_name = values[2]
                menu.add_command(
                    label="View results",
                    command=lambda s=scenario_name: self._view_scenario_plots(s),
                )
                menu.add_separator()
        menu.add_command(
            label="Delete irrevocably",
            command=self._on_delete_results,
        )
        menu.tk_popup(event.x_root, event.y_root)

    def _on_executed_space_from_menu(self) -> None:
        """Toggle checkboxes for selected items in executed_tree (from context menu)."""
        for item in self.executed_tree.selection():
            self._toggle_check(self.executed_tree, item, "check")
        self._update_output_status()
        self._save_checked_executed_scenarios()

    # ── Input source management ──────────────────────────────────────

    def _on_add_source(self) -> None:
        """Open the Add dialog and refresh sources if files were added."""
        if not self.current_project:
            messagebox.showinfo("No project", "No project is currently loaded.")
            return

        # Remember existing sources so we can detect what's new
        old_sources: set[str] = set()
        if self.input_source_mgr:
            old_sources = {s.name for s in self.input_source_mgr._sources}

        from flextool.gui.dialogs.add_dialog import AddDialog

        project_path = get_projects_dir() / self.current_project
        self._ensure_execution_mgr()
        dlg = AddDialog(self, project_path, execution_mgr=self.execution_mgr)
        if dlg.result:
            self._refresh_input_sources()
            self._autocheck_new_sources(old_sources)
        if dlg.old_convert_started:
            self._open_or_raise_execution_window()
        # Handle files the user chose to migrate.  The conversion
        # callbacks refresh the input sources again when finished.
        for name in dlg.files_to_convert:
            self._convert_xlsx_to_sqlite(name, confirm=False)
        for name in dlg.files_to_update_xlsx:
            self._update_xlsx_version(name)

    def _autocheck_new_sources(self, old_sources: set[str]) -> None:
        """Check (tick) newly added input sources and their available scenarios."""
        # 1. Check the new input sources in the input_sources_tree
        new_source_numbers: set[int] = set()
        for item in self.input_sources_tree.get_children():
            values = self.input_sources_tree.item(item, "values")
            if values and values[1] not in old_sources:
                self.input_sources_tree.set(item, "check", CHECK_ON)
                try:
                    new_source_numbers.add(int(values[2]))
                except (ValueError, IndexError):
                    pass

        if not new_source_numbers:
            return

        # 2. Save input source check state so _update_available_scenarios sees them
        self._save_checked_input_sources()

        # 3. Rebuild available scenarios with the new sources included
        self._update_available_scenarios()

        # 4. Check all scenarios that belong to the new sources
        for item in self.available_tree.get_children():
            values = self.available_tree.item(item, "values")
            if values:
                try:
                    src_num = int(values[1])  # source_number column
                except (ValueError, IndexError):
                    continue
                if src_num in new_source_numbers:
                    self.available_tree.set(item, "check", CHECK_ON)

        # 5. Persist the updated check states
        self._save_checked_available_scenarios()

    def _refresh_and_check_new_scenarios(self) -> None:
        """Refresh input sources and check any newly appeared scenarios.

        Called after sensitivity import completes — the source already exists
        but new scenarios have been added to it.
        """
        # Remember which scenarios are currently shown
        old_scenarios: set[str] = set()
        for item in self.available_tree.get_children():
            values = self.available_tree.item(item, "values")
            if values:
                old_scenarios.add(str(values[0]))  # scenario name

        self._refresh_input_sources()

        # Check any scenarios that weren't there before
        for item in self.available_tree.get_children():
            values = self.available_tree.item(item, "values")
            if values and str(values[0]) not in old_scenarios:
                self.available_tree.set(item, "check", CHECK_ON)

        self._save_checked_available_scenarios()
        self._update_add_to_execution_style()
        self._update_add_to_execution_style()

    def _on_refresh_sources(self) -> None:
        """Refresh input sources by re-scanning the directory."""
        if not self.input_source_mgr:
            return
        self._refresh_input_sources()

    def _refresh_input_sources(self) -> None:
        """Re-scan input sources and repopulate the treeview."""
        if not self.input_source_mgr:
            return

        # Check and upgrade sqlite databases before reading scenarios
        upgrade_messages = self.input_source_mgr.check_db_versions()
        if upgrade_messages:
            messagebox.showinfo(
                "Database upgrades",
                "\n".join(upgrade_messages),
                parent=self,
            )

        sources = self.input_source_mgr.refresh()

        # Clear input sources tree
        for item in self.input_sources_tree.get_children():
            self.input_sources_tree.delete(item)

        # Configure tag for error rows
        self.input_sources_tree.tag_configure("error", background="#ffcccc")

        # Populate input sources tree
        saved_checked = set(self.project_settings.checked_input_sources)
        has_saved_state = len(saved_checked) > 0
        for source in sources:
            if source.status == "ok":
                status_char = STATUS_OK
            elif source.status == "editing":
                status_char = STATUS_EDITING
            else:
                status_char = STATUS_ERR

            # Restore checkbox: if we have saved state, only check those in saved list;
            # otherwise default to CHECK_ON (first load / no saved state)
            if has_saved_state:
                check_char = CHECK_ON if source.name in saved_checked else CHECK_OFF
            else:
                check_char = CHECK_ON

            tags = ("error",) if source.status == "error" else ()
            self.input_sources_tree.insert(
                "",
                "end",
                values=(check_char, source.name, source.number, status_char),
                tags=tags,
            )

        # Apply current sort mode
        if self._input_sort_mode == "alpha":
            self._sort_tree_items(self.input_sources_tree, col_index=1, numeric=False)
        else:
            self._sort_tree_items(self.input_sources_tree, col_index=2, numeric=True)

        # Update Add button appearance based on whether there are sources
        self._update_add_button_style(len(sources) == 0)

        # Update available scenarios
        self._update_available_scenarios()

        # Update Edit / Convert / Delete button states
        self._update_input_button_states()

    def _update_add_button_style(self, no_sources: bool) -> None:
        """Highlight the Add button when there are no input sources.

        Uses sv_ttk's built-in Accent.TButton style which reliably renders
        as a visually prominent button (Green.TButton background is ignored
        by the Sun Valley theme engine).
        """
        if no_sources:
            self.add_source_btn.configure(style="Accent.TButton")
        else:
            self.add_source_btn.configure(style="TButton")

    def _update_add_to_execution_style(self) -> None:
        """Highlight 'Add selected to execution list' green when scenarios are checked."""
        has_checked = False
        for item in self.available_tree.get_children():
            values = self.available_tree.item(item, "values")
            if values and values[0] == CHECK_ON:
                has_checked = True
                break
        if has_checked:
            self.add_to_execution_btn.configure(style="Accent.TButton")
        else:
            self.add_to_execution_btn.configure(style="TButton")

    def _update_execution_menu_style(self) -> None:
        """Highlight 'Execution jobs' when there are jobs and the window is not open."""
        has_jobs = False
        if self.execution_mgr is not None:
            has_jobs = len(self.execution_mgr.get_jobs()) > 0
        window_open = (
            self.execution_window is not None
            and self.execution_window.winfo_exists()
        )
        if has_jobs and not window_open:
            self.execution_menu_btn.configure(style="Accent.TButton")
        else:
            self.execution_menu_btn.configure(style="TButton")

    def _update_output_frame_style(self) -> None:
        """Tint the output actions LabelFrame when executed scenarios are checked."""
        has_checked = False
        for item in self.executed_tree.get_children():
            values = self.executed_tree.item(item, "values")
            if values and values[0] == CHECK_ON:
                has_checked = True
                break
        if has_checked:
            self.output_frame.configure(bg="#2E7D32")
        else:
            self.output_frame.configure(bg=self._output_frame_default_bg)

    def _refresh_and_autocheck_scenario(
        self, scenario_name: str, finish_timestamp: str = "",
    ) -> None:
        """Refresh executed scenarios and auto-check the newly completed one."""
        self._refresh_executed_scenarios()
        # Find and check the newly completed scenario, updating timestamp
        for item in self.executed_tree.get_children():
            values = self.executed_tree.item(item, "values")
            if values and values[2] == scenario_name:
                self.executed_tree.set(item, "check", CHECK_ON)
                if finish_timestamp:
                    self.executed_tree.set(item, "timestamp", finish_timestamp)
                break
        self._update_output_status()

    def _get_selected_source_names(self) -> list[str]:
        """Return the names of input sources whose checkboxes are checked."""
        selected: list[str] = []
        for item in self.input_sources_tree.get_children():
            values = self.input_sources_tree.item(item, "values")
            if values and values[0] == CHECK_ON:
                selected.append(values[1])  # name column
        return selected

    def _update_available_scenarios(self) -> None:
        """Repopulate the available scenarios treeview based on selected input sources."""
        # Clear available scenarios tree
        for item in self.available_tree.get_children():
            self.available_tree.delete(item)

        if not self.input_source_mgr:
            self._update_add_to_execution_style()
            return

        selected_sources = self._get_selected_source_names()
        # If nothing is selected, show all scenarios
        if not selected_sources:
            scenarios = self.input_source_mgr.get_all_scenarios()
        else:
            scenarios = self.input_source_mgr.get_all_scenarios(selected_sources)

        # Apply persistent ordering via AvailableScenarioManager
        if self.avail_scenario_mgr:
            scenarios = self.avail_scenario_mgr.update_scenarios(scenarios)

        # Build a set of source numbers whose input source has editing status
        editing_source_numbers: set[int] = set()
        for item in self.input_sources_tree.get_children():
            values = self.input_sources_tree.item(item, "values")
            if values and values[3] == STATUS_EDITING:
                try:
                    editing_source_numbers.add(int(values[2]))
                except (ValueError, IndexError):
                    pass

        # Configure tag for editing-source scenarios (reddish background)
        self.available_tree.tag_configure("editing_source", background="#662222")

        saved_checked_avail = set(self.project_settings.checked_available_scenarios)
        for scenario in scenarios:
            key = f"{scenario.source_number}|{scenario.name}"
            check_char = CHECK_ON if key in saved_checked_avail else CHECK_OFF
            tags: tuple[str, ...] = ()
            if scenario.source_number in editing_source_numbers:
                tags = ("editing_source",)
            self.available_tree.insert(
                "",
                "end",
                values=(check_char, scenario.source_number, scenario.name),
                tags=tags,
            )

        # Apply current sort mode
        if self._available_sort_mode == "alpha":
            self._sort_tree_items(self.available_tree, col_index=2, numeric=False)
        else:
            self._sort_tree_items(self.available_tree, col_index=1, numeric=True)

        self._update_add_to_execution_style()

    def _update_available_scenario_tags(self) -> None:
        """Update editing-source tags on existing available scenario rows without full repopulation."""
        editing_source_numbers: set[int] = set()
        for item in self.input_sources_tree.get_children():
            values = self.input_sources_tree.item(item, "values")
            if values and values[3] == STATUS_EDITING:
                try:
                    editing_source_numbers.add(int(values[2]))
                except (ValueError, IndexError):
                    pass

        self.available_tree.tag_configure("editing_source", background="#662222")

        for item in self.available_tree.get_children():
            values = self.available_tree.item(item, "values")
            if not values:
                continue
            try:
                source_num = int(values[1])
            except (ValueError, IndexError):
                continue
            if source_num in editing_source_numbers:
                self.available_tree.item(item, tags=("editing_source",))
            else:
                self.available_tree.item(item, tags=())

    # ── Input source button state management ────────────────────────

    def _get_checked_sources(self) -> list[tuple[str, str]]:
        """Return (name, status_char) for each checked input source row."""
        checked: list[tuple[str, str]] = []
        for item in self.input_sources_tree.get_children():
            values = self.input_sources_tree.item(item, "values")
            if values and values[0] == CHECK_ON:
                # values: (check, name, number, status)
                checked.append((values[1], values[3]))
        return checked

    def _get_selected_sources(self) -> list[tuple[str, str]]:
        """Return (name, status_char) for each highlighted (selected) input source row."""
        selected: list[tuple[str, str]] = []
        for item in self.input_sources_tree.selection():
            values = self.input_sources_tree.item(item, "values")
            if values:
                # values: (check, name, number, status)
                selected.append((values[1], values[3]))
        return selected

    def _update_input_button_states(self) -> None:
        """Enable or disable Edit, Convert, Delete based on Treeview selection (not checkboxes)."""
        selected = self._get_selected_sources()

        # ── Edit: exactly one selected, not in editing state ──
        if len(selected) == 1:
            _name, status = selected[0]
            if status == STATUS_EDITING:
                self.edit_source_btn.configure(state="disabled")
            else:
                self.edit_source_btn.configure(state="normal")
        else:
            self.edit_source_btn.configure(state="disabled")

        # ── Convert: exactly one selected, xlsx or sqlite, status OK ──
        if len(selected) == 1:
            name, status = selected[0]
            is_convertible = name.lower().endswith((".xlsx", ".sqlite"))
            if is_convertible and status == STATUS_OK:
                self.convert_source_btn.configure(state="normal")
            else:
                self.convert_source_btn.configure(state="disabled")
        else:
            self.convert_source_btn.configure(state="disabled")

        # ── Delete: at least one selected ──
        if selected:
            self.delete_source_btn.configure(state="normal")
        else:
            self.delete_source_btn.configure(state="disabled")

    # ── Edit button handler ─────────────────────────────────────────

    @safe_callback
    def _on_edit_source(self) -> None:
        """Open the selected (highlighted) input source for editing."""
        if not self.input_source_mgr or not self.current_project:
            return

        selected = self._get_selected_sources()
        if len(selected) != 1:
            return

        source_name, _status = selected[0]
        project_path = get_projects_dir() / self.current_project
        filepath = project_path / "input_sources" / source_name

        if not filepath.exists():
            messagebox.showerror("File not found", f"Cannot find:\n{filepath}")
            return

        ext = filepath.suffix.lower()
        if ext in (".xlsx", ".ods"):
            try:
                open_file_in_default_app(filepath)
                self.input_source_mgr.mark_as_editing(source_name)
            except OSError as exc:
                messagebox.showerror("Error", f"Could not open file:\n{exc}")
                return
        elif ext == ".sqlite":
            db_url = f"sqlite:///{filepath}"
            proc = self.db_editor_mgr.open_database(db_url, source_name)
            if proc is None:
                messagebox.showinfo(
                    "spine-db-editor not found",
                    "The spine-db-editor command was not found on your system.\n\n"
                    "To edit .sqlite input sources, install Spine Toolbox:\n\n"
                    '  pip install ".[toolbox]"\n\n'
                    "(run from the flextool directory)",
                )
                return
        else:
            messagebox.showinfo("Unsupported", f"Cannot edit files of type '{ext}'.")
            return

        # Refresh to show editing status
        self._refresh_input_sources()

    # ── Convert button handler ──────────────────────────────────────

    @safe_callback
    def _on_convert_source(self) -> None:
        """Convert the selected input source between xlsx and sqlite formats."""
        if not self.input_source_mgr or not self.current_project:
            return

        selected = self._get_selected_sources()
        if len(selected) != 1:
            return

        source_name, _status = selected[0]
        ext = Path(source_name).suffix.lower()

        if ext == ".xlsx":
            self._convert_xlsx_to_sqlite(source_name)
        elif ext == ".sqlite":
            self._convert_sqlite_to_xlsx(source_name)

    # ── Conversion: xlsx → sqlite ────────────────────────────────

    def _convert_xlsx_to_sqlite(self, source_name: str, confirm: bool = True) -> None:
        """Convert an xlsx input source to sqlite format via subprocess.

        Args:
            source_name: Name of the xlsx file in ``input_sources/``.
            confirm: If True (default), ask the user to confirm before
                converting.  Set to False when the caller has already
                obtained confirmation (e.g. from the add-dialog).
        """
        project_path = get_projects_dir() / self.current_project
        input_dir = project_path / "input_sources"
        xlsx_path = input_dir / source_name

        if not xlsx_path.exists():
            messagebox.showerror("File not found", f"Cannot find:\n{xlsx_path}")
            return

        stem = Path(source_name).stem
        target_sqlite = input_dir / f"{stem}.sqlite"

        # Check if target already exists in input_sources/
        if not self._resolve_file_conflict(target_sqlite):
            return

        if confirm:
            answer = messagebox.askokcancel(
                "Convert to database",
                f"Convert '{source_name}' to a database input source?\n\n"
                f"The xlsx will be moved to the 'converted' folder for safekeeping.",
            )
            if not answer:
                return

        target_db_url = f"sqlite:///{target_sqlite}"

        # Detect Excel format / version and initialize DB with the right template
        from flextool.process_inputs import (
            detect_excel_format, ExcelFormat, CURRENT_FLEXTOOL_DB_VERSION,
        )
        from flextool.update_flextool.initialize_database import initialize_database
        from flextool.update_flextool.db_migration import migrate_database
        info = detect_excel_format(xlsx_path)

        flextool_root = Path(__file__).resolve().parent.parent.parent
        needs_migration = False

        if info.format == ExcelFormat.SELF_DESCRIBING and (
            info.version is None or info.version >= CURRENT_FLEXTOOL_DB_VERSION
        ):
            # Current version — import directly against the current schema
            template = flextool_root / "version" / "flextool_template_master.json"
        else:
            # Older Excel (SPECIFICATION or older SELF_DESCRIBING):
            # init from v25 base, migrate to the Excel's version, import, then
            # migrate the rest of the way to current after import.
            template = flextool_root / "version" / "flextool_template_v25.json"
            needs_migration = True

        if not template.exists():
            messagebox.showerror("Template missing", f"Cannot find template:\n{template}")
            return
        initialize_database(str(template), str(target_sqlite))

        if needs_migration and info.version is not None and info.version > 25:
            # Bring the empty schema up to the Excel's version so parameter
            # names match during import.  The final migration to current
            # happens after import via _run_conversion_subprocess.
            migrate_database(str(target_sqlite), up_to=info.version)

        # Build the appropriate subprocess command
        if info.format == ExcelFormat.SELF_DESCRIBING:
            cmd = [
                sys.executable, "-m",
                "flextool.cli.cmd_read_self_describing_tabular_input",
                str(xlsx_path),
                target_db_url,
                "--keep-entities",
            ]
        elif info.format == ExcelFormat.SPECIFICATION:
            cmd = [
                sys.executable, "-m",
                "flextool.cli.cmd_read_tabular_input",
                target_db_url,
                "--tabular-file-path",
                str(xlsx_path),
                "--migration-follows",
            ]
        elif info.format == ExcelFormat.OLD_V2:
            cmd = [
                sys.executable, "-m",
                "flextool.cli.cmd_read_old_flextool",
                str(xlsx_path),
                target_db_url,
            ]
        else:
            messagebox.showerror(
                "Unknown format",
                f"Cannot determine the Excel format of '{source_name}'.\n\n"
                "Expected either a self-describing FlexTool Excel, a FlexTool 3.x "
                "specification-based Excel, or an old FlexTool 2.x .xlsm file.",
            )
            return

        # Build version note for the log window
        version_note: str | None = None
        if info.version is not None and info.version < CURRENT_FLEXTOOL_DB_VERSION:
            version_note = (
                f"Note: Excel is version {info.version}, "
                f"current FlexTool version is {CURRENT_FLEXTOOL_DB_VERSION}. "
                f"The database has been migrated to version {CURRENT_FLEXTOOL_DB_VERSION}. "
                f"You can convert back to xlsx to get an updated Excel."
            )

        self._run_conversion_subprocess(
            cmd, source_name, xlsx_path, target_sqlite,
            f"Convert: {source_name} \u2192 sqlite",
            migrate_db_path=str(target_sqlite) if needs_migration else None,
            version_note=version_note,
        )

    # ── Update xlsx version via round-trip ───────���──────────────

    def _update_xlsx_version(self, source_name: str) -> None:
        """Migrate an older xlsx to the current version in-place.

        Round-trips through a temporary SQLite database:
        old xlsx → temp sqlite (import + migrate) → updated xlsx.
        """
        from flextool.gui.execution_manager import JobType
        from flextool.process_inputs import (
            detect_excel_format, ExcelFormat, CURRENT_FLEXTOOL_DB_VERSION,
        )
        from flextool.update_flextool.initialize_database import initialize_database
        from flextool.update_flextool.db_migration import migrate_database

        project_path = get_projects_dir() / self.current_project
        input_dir = project_path / "input_sources"
        xlsx_path = input_dir / source_name

        if not xlsx_path.exists():
            messagebox.showerror("File not found", f"Cannot find:\n{xlsx_path}")
            return

        info = detect_excel_format(xlsx_path)
        flextool_root = Path(__file__).resolve().parent.parent.parent

        # Create a temporary directory for the intermediate sqlite
        import tempfile
        tmp_dir = Path(tempfile.mkdtemp(prefix="flextool_update_"))
        tmp_sqlite = tmp_dir / "temp_import.sqlite"
        tmp_db_url = f"sqlite:///{tmp_sqlite}"

        # Choose template and build import command (same logic as _convert)
        if info.format == ExcelFormat.SELF_DESCRIBING and (
            info.version is None or info.version >= CURRENT_FLEXTOOL_DB_VERSION
        ):
            # Already current — nothing to do
            try:
                tmp_dir.rmdir()
            except OSError:
                pass
            return
        elif info.format == ExcelFormat.SELF_DESCRIBING:
            template = flextool_root / "version" / "flextool_template_v25.json"
            import_cmd = [
                sys.executable, "-m",
                "flextool.cli.cmd_read_self_describing_tabular_input",
                str(xlsx_path), tmp_db_url, "--keep-entities",
            ]
        else:
            # SPECIFICATION format
            template = flextool_root / "version" / "flextool_template_v25.json"
            import_cmd = [
                sys.executable, "-m",
                "flextool.cli.cmd_read_tabular_input",
                tmp_db_url, "--tabular-file-path", str(xlsx_path),
                "--migration-follows",
            ]

        export_cmd = [
            sys.executable, "-m",
            "flextool.cli.cmd_export_to_tabular",
            tmp_db_url,
            str(xlsx_path),
        ]

        # Initialize the temp database
        initialize_database(str(template), str(tmp_sqlite))

        # Pre-migrate to Excel's version for self-describing
        if (
            info.format == ExcelFormat.SELF_DESCRIBING
            and info.version is not None
            and 25 < info.version < CURRENT_FLEXTOOL_DB_VERSION
        ):
            migrate_database(str(tmp_sqlite), up_to=info.version)

        # Set up execution job
        self._ensure_execution_mgr()
        if self.execution_mgr is None:
            return

        job = self.execution_mgr.add_auxiliary_job(
            JobType.CONVERSION,
            f"Update: {source_name} \u2192 version {CURRENT_FLEXTOOL_DB_VERSION}",
            f"format_convert:{source_name}",
        )
        mgr = self.execution_mgr
        mgr.append_stdout(job.job_id, f"Updating {source_name} to version {CURRENT_FLEXTOOL_DB_VERSION}\n")
        mgr.append_stdout(job.job_id, "Step 1: Import into temporary database")
        mgr.append_stdout(job.job_id, " ".join(import_cmd))
        mgr.append_stdout(job.job_id, "")

        self._open_or_raise_execution_window()
        if self.execution_window is not None:
            self.execution_window.select_job(job.job_id)

        def _worker() -> None:
            import shutil as _shutil
            success = False
            try:
                env = {**os.environ, "PYTHONUNBUFFERED": "1"}

                # Step 1: import old Excel into temp sqlite
                proc = subprocess.Popen(
                    import_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, cwd=str(flextool_root), env=env,
                )
                with mgr._lock:
                    job.process = proc
                for line in proc.stdout:  # type: ignore[union-attr]
                    mgr.append_stdout(job.job_id, line.rstrip("\n"))
                proc.wait()

                if proc.returncode != 0:
                    mgr.append_stdout(job.job_id, f"\nImport failed (exit code {proc.returncode}).")
                    return

                # Step 2: migrate temp sqlite to current version
                mgr.append_stdout(job.job_id, "\nStep 2: Migrating database to current version...")
                try:
                    from flextool.update_flextool.db_migration import migrate_database as _migrate
                    _migrate(str(tmp_sqlite))
                    mgr.append_stdout(job.job_id, "Database migration completed.")
                except Exception as mig_exc:
                    mgr.append_stdout(job.job_id, f"Database migration failed: {mig_exc}")
                    return

                # Step 3: export back to Excel
                mgr.append_stdout(job.job_id, "\nStep 3: Exporting updated database to Excel...")
                mgr.append_stdout(job.job_id, " ".join(export_cmd))
                proc2 = subprocess.Popen(
                    export_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, cwd=str(flextool_root), env=env,
                )
                with mgr._lock:
                    job.process = proc2
                for line in proc2.stdout:  # type: ignore[union-attr]
                    mgr.append_stdout(job.job_id, line.rstrip("\n"))
                proc2.wait()

                if proc2.returncode != 0:
                    mgr.append_stdout(job.job_id, f"\nExport failed (exit code {proc2.returncode}).")
                    return

                success = True
                mgr.append_stdout(
                    job.job_id,
                    f"\nSuccessfully updated {source_name} to version {CURRENT_FLEXTOOL_DB_VERSION}.",
                )
            except Exception as exc:
                logger.error("Version update failed: %s", exc, exc_info=True)
                mgr.append_stdout(job.job_id, f"\nError: {exc}")
            finally:
                # Clean up temp directory
                try:
                    _shutil.rmtree(tmp_dir, ignore_errors=True)
                except Exception:
                    pass

            mgr.finish_job(job.job_id, success)
            self.after(0, self._refresh_input_sources)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

    # ── Conversion: sqlite → xlsx ────────────────────────────────

    def _convert_sqlite_to_xlsx(self, source_name: str) -> None:
        """Convert a sqlite input source to xlsx format via subprocess."""
        project_path = get_projects_dir() / self.current_project
        input_dir = project_path / "input_sources"
        sqlite_path = input_dir / source_name

        if not sqlite_path.exists():
            messagebox.showerror("File not found", f"Cannot find:\n{sqlite_path}")
            return

        stem = Path(source_name).stem
        target_xlsx = input_dir / f"{stem}.xlsx"

        if not self._resolve_file_conflict(target_xlsx):
            return

        answer = messagebox.askokcancel(
            "Convert to Excel",
            f"Convert '{source_name}' to Excel format?\n\n"
            f"The sqlite will be moved to the 'converted' folder for safekeeping.",
        )
        if not answer:
            return

        db_url = f"sqlite:///{sqlite_path}"
        cmd = [
            sys.executable, "-m",
            "flextool.cli.cmd_export_to_tabular",
            db_url,
            str(target_xlsx),
        ]

        self._run_conversion_subprocess(
            cmd, source_name, sqlite_path, target_xlsx,
            f"Convert: {source_name} \u2192 xlsx",
        )

    # ── Conversion subprocess runner ─────────────────────────────

    def _run_conversion_subprocess(
        self,
        cmd: list[str],
        source_name: str,
        source_path: Path,
        target_path: Path,
        description: str,
        migrate_db_path: str | None = None,
        version_note: str | None = None,
    ) -> None:
        """Run a conversion command as an auxiliary job in the execution window.

        After the subprocess completes, optionally runs database migration,
        then moves the source to converted/ and refreshes the input sources.
        """
        from flextool.gui.execution_manager import JobType

        self._ensure_execution_mgr()
        if self.execution_mgr is None:
            return

        job = self.execution_mgr.add_auxiliary_job(
            JobType.CONVERSION,
            description,
            f"format_convert:{source_name}",
        )

        flextool_root = get_projects_dir().parent
        cmd_str = " ".join(cmd)
        self.execution_mgr.append_stdout(job.job_id, f"Converting {description}\n")
        self.execution_mgr.append_stdout(job.job_id, cmd_str)
        self.execution_mgr.append_stdout(job.job_id, "")

        self._open_or_raise_execution_window()
        if self.execution_window is not None:
            self.execution_window.select_job(job.job_id)

        mgr = self.execution_mgr  # capture for thread

        def _worker() -> None:
            success = False
            try:
                env = {**os.environ, "PYTHONUNBUFFERED": "1"}
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, cwd=str(flextool_root), env=env,
                )
                with mgr._lock:
                    job.process = proc

                for line in proc.stdout:  # type: ignore[union-attr]
                    mgr.append_stdout(job.job_id, line.rstrip("\n"))

                proc.wait()
                success = proc.returncode == 0

                if success and migrate_db_path:
                    mgr.append_stdout(job.job_id, "\nMigrating database to current version...")
                    try:
                        from flextool.update_flextool.db_migration import migrate_database
                        migrate_database(migrate_db_path)
                        mgr.append_stdout(job.job_id, "Database migration completed.")
                    except Exception as mig_exc:
                        mgr.append_stdout(job.job_id, f"Database migration failed: {mig_exc}")
                        success = False

                if success:
                    mgr.append_stdout(job.job_id, "\nConversion succeeded.")
                    if version_note:
                        mgr.append_stdout(job.job_id, f"\n{version_note}")
                else:
                    mgr.append_stdout(
                        job.job_id, f"\nConversion failed (exit code {proc.returncode})."
                    )
            except Exception as exc:
                logger.error("Conversion subprocess failed: %s", exc, exc_info=True)
                mgr.append_stdout(job.job_id, f"\nError: {exc}")

            mgr.finish_job(job.job_id, success)
            self.after(0, self._conversion_finished,
                       success, source_name, source_path, target_path)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

    def _conversion_finished(
        self,
        success: bool,
        source_name: str,
        source_path: Path,
        target_path: Path,
    ) -> None:
        """Handle post-conversion tasks on the main thread."""
        if not success:
            if target_path.exists():
                try:
                    target_path.unlink()
                except OSError:
                    pass
            return

        project_path = get_projects_dir() / self.current_project
        converted_dir = project_path / "converted"
        moved = self._move_to_converted(source_path, converted_dir)

        if moved and source_name in self.project_settings.input_source_numbers:
            del self.project_settings.input_source_numbers[source_name]
            save_project_settings(project_path, self.project_settings)

        self._refresh_input_sources()

    # ── xlsx pre-conversion pipeline (before execution) ──────────

    def _start_xlsx_preconversion(self, scenarios: list[ScenarioInfo]) -> None:
        """Convert unique xlsx sources to intermediate sqlite, then dispatch jobs.

        Each source gets its own auxiliary job entry in the execution window.
        """
        project_path = get_projects_dir() / self.current_project
        intermediate_dir = project_path / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)

        # Always reconvert — the user may have edited the xlsx since last run.
        seen: set[str] = set()
        queue: list[tuple[str, Path]] = []
        for s in scenarios:
            if s.source_name not in seen:
                seen.add(s.source_name)
                self.execution_mgr._converted_xlsx.discard(s.source_name)
                xlsx_path = project_path / "input_sources" / s.source_name
                queue.append((s.source_name, xlsx_path))

        self._xlsx_pending_scenarios = list(scenarios)

        if not queue:
            self._xlsx_preconversion_done(success=True)
            return

        self._xlsx_conversion_queue = queue
        self._xlsx_converting_sources = {name for name, _ in queue}
        self._update_available_scenario_tags()

        # Show progress in the execution window
        self._open_or_raise_execution_window()
        self._xlsx_convert_next()

    def _xlsx_convert_next(self) -> None:
        """Convert the next xlsx source in the queue."""
        from flextool.gui.execution_manager import JobType

        if not self._xlsx_conversion_queue:
            self._xlsx_preconversion_done(success=True)
            return

        source_name, xlsx_path = self._xlsx_conversion_queue.pop(0)
        project_path = get_projects_dir() / self.current_project
        stem = Path(source_name).stem
        db_path = project_path / "intermediate" / f"{stem}.sqlite"
        target_db_url = f"sqlite:///{db_path}"

        if db_path.exists():
            _unlink_sqlite(db_path)

        # Detect format and build command
        from flextool.process_inputs import (
            detect_excel_format, ExcelFormat, CURRENT_FLEXTOOL_DB_VERSION,
        )
        from flextool.update_flextool.initialize_database import initialize_database
        from flextool.update_flextool.db_migration import migrate_database
        info = detect_excel_format(xlsx_path)
        flextool_root = Path(__file__).resolve().parent.parent.parent
        migrate_db_path: str | None = None

        if info.format == ExcelFormat.SELF_DESCRIBING and (
            info.version is None or info.version >= CURRENT_FLEXTOOL_DB_VERSION
        ):
            # Current version — import directly against the current schema
            template = flextool_root / "version" / "flextool_template_master.json"
            cmd = [
                sys.executable, "-m",
                "flextool.cli.cmd_read_self_describing_tabular_input",
                str(xlsx_path), target_db_url, "--keep-entities",
            ]
        elif info.format == ExcelFormat.SELF_DESCRIBING:
            # Older self-describing: init from v25 base, pre-migrate to
            # the Excel's version, import, then migrate to current.
            template = flextool_root / "version" / "flextool_template_v25.json"
            cmd = [
                sys.executable, "-m",
                "flextool.cli.cmd_read_self_describing_tabular_input",
                str(xlsx_path), target_db_url, "--keep-entities",
            ]
            migrate_db_path = str(db_path)
        else:
            template = flextool_root / "version" / "flextool_template_v25.json"
            cmd = [
                sys.executable, "-m",
                "flextool.cli.cmd_read_tabular_input",
                target_db_url, "--tabular-file-path", str(xlsx_path),
                "--migration-follows",
            ]
            migrate_db_path = str(db_path)

        # Initialize database from template
        initialize_database(str(template), str(db_path))

        # For older self-describing Excel, bring the empty schema up to the
        # Excel's version so parameter names match during import.
        if (
            info.format == ExcelFormat.SELF_DESCRIBING
            and info.version is not None
            and 25 < info.version < CURRENT_FLEXTOOL_DB_VERSION
        ):
            migrate_database(str(db_path), up_to=info.version)

        # Create auxiliary job in the execution window
        job = self.execution_mgr.add_auxiliary_job(
            JobType.CONVERSION,
            f"Convert: {source_name}",
            f"conversion:{source_name}",
            insert_before_source=source_name,
        )

        self.execution_mgr.append_stdout(job.job_id, f"Converting {source_name} ...")
        self.execution_mgr.append_stdout(job.job_id, " ".join(cmd))
        self.execution_mgr.append_stdout(job.job_id, "")

        self._open_or_raise_execution_window()
        if self.execution_window is not None:
            self.execution_window.select_job(job.job_id)

        def _worker() -> None:
            success = False
            try:
                env = {**os.environ, "PYTHONUNBUFFERED": "1"}
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, cwd=str(flextool_root), env=env,
                )
                with self.execution_mgr._lock:
                    job.process = proc

                for line in proc.stdout:  # type: ignore[union-attr]
                    self.execution_mgr.append_stdout(job.job_id, line.rstrip("\n"))

                proc.wait()
                success = proc.returncode == 0

                if success and migrate_db_path:
                    self.execution_mgr.append_stdout(job.job_id, "\nMigrating database...")
                    try:
                        from flextool.update_flextool.db_migration import migrate_database
                        migrate_database(migrate_db_path)
                        self.execution_mgr.append_stdout(job.job_id, "Migration completed.")
                    except Exception as exc:
                        self.execution_mgr.append_stdout(job.job_id, f"Migration failed: {exc}")
                        success = False

                status = "succeeded" if success else f"failed (exit code {proc.returncode})"
                self.execution_mgr.append_stdout(job.job_id, f"\n{source_name}: {status}")
            except Exception as exc:
                self.execution_mgr.append_stdout(job.job_id, f"\nError: {exc}")

            self.execution_mgr.finish_job(job.job_id, success)
            self.after(0, self._xlsx_one_source_finished, source_name, success)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

    def _xlsx_one_source_finished(self, source_name: str, success: bool) -> None:
        """Called on main thread when one xlsx source conversion finishes."""
        if success:
            self.execution_mgr._converted_xlsx.add(source_name)
            self._xlsx_converting_sources.discard(source_name)
            self._xlsx_convert_next()
        else:
            self._xlsx_preconversion_done(success=False)

    def _xlsx_preconversion_done(self, success: bool) -> None:
        """Called when all xlsx conversions are done (or one failed)."""
        self._xlsx_converting_sources.clear()
        self._xlsx_conversion_queue.clear()
        self._update_available_scenario_tags()

        scenarios = self._xlsx_pending_scenarios
        self._xlsx_pending_scenarios = []

        if success:
            if scenarios:
                added = self.execution_mgr.add_jobs(scenarios)
                self.execution_mgr.start()
                self._update_execution_menu_style()
                self._open_or_raise_execution_window()
                if added and self.execution_window is not None:
                    self.execution_window.select_job(added[-1].job_id)
        else:
            messagebox.showerror(
                "Conversion failed",
                "xlsx \u2192 sqlite conversion failed.\n\n"
                "Check the conversion entry in the execution window.\n"
                "Scenarios will not be executed.",
                parent=self,
            )

    # ── File conflict resolution helpers ─────────────────────────

    def _ask_file_conflict(self, filepath: Path) -> str:
        """Ask user how to handle an existing file.

        Returns: "overwrite", "rename", or "cancel".
        """
        result = messagebox.askyesnocancel(
            "File already exists",
            f"'{filepath.name}' already exists in:\n"
            f"  {filepath.parent}\n\n"
            f"Yes = Overwrite existing file\n"
            f"No = Rename existing to .backup (with content hash)\n"
            f"Cancel = Abort",
        )
        if result is True:
            return "overwrite"
        elif result is False:
            return "rename"
        return "cancel"

    @staticmethod
    def _backup_with_hash(filepath: Path) -> Path:
        """Rename a file to include a content hash and .backup suffix.

        Example: foo.xlsx → foo.a1b2c3d4e5f6.backup.xlsx
        """
        content_hash = hashlib.sha256(filepath.read_bytes()).hexdigest()[:12]
        backup_name = f"{filepath.stem}.{content_hash}.backup{filepath.suffix}"
        backup_path = filepath.parent / backup_name
        filepath.rename(backup_path)
        return backup_path

    def _resolve_file_conflict(self, target_path: Path) -> bool:
        """Check if target exists and resolve the conflict.

        Returns True if resolved (file removed/renamed or didn't exist),
        False if user cancelled.
        """
        if not target_path.exists():
            return True
        action = self._ask_file_conflict(target_path)
        if action == "cancel":
            return False
        if action == "rename":
            self._backup_with_hash(target_path)
        elif action == "overwrite":
            target_path.unlink()
        return True

    def _move_to_converted(self, source_path: Path, converted_dir: Path) -> bool:
        """Move source file to converted/ folder, handling conflicts.

        Returns True if moved successfully, False otherwise.
        """
        converted_dir.mkdir(parents=True, exist_ok=True)
        dest = converted_dir / source_path.name

        if dest.exists():
            action = self._ask_file_conflict(dest)
            if action == "cancel":
                return False
            if action == "rename":
                self._backup_with_hash(dest)
            elif action == "overwrite":
                dest.unlink()

        try:
            shutil.move(str(source_path), str(dest))
            return True
        except OSError as exc:
            messagebox.showwarning(
                "Move failed",
                f"Conversion succeeded but the file could not be moved "
                f"to the 'converted' folder:\n{exc}",
            )
            return False

    # ── Delete button handler ───────────────────────────────────────

    @safe_callback
    def _on_delete_source(self) -> None:
        """Delete the selected (highlighted) input source file(s)."""
        if not self.input_source_mgr or not self.current_project:
            return

        selected = self._get_selected_sources()
        if not selected:
            return

        project_path = get_projects_dir() / self.current_project
        names = [name for name, _ in selected]
        names_str = "\n  ".join(names)

        answer = messagebox.askyesno(
            "Delete input source",
            f"Are you really sure you want to delete the input source?\n\n"
            f"  {names_str}\n\n"
            f"It will not be possible to retrieve. Another option is to move "
            f"it to another folder from the current location at "
            f"'projects/{self.current_project}/input_sources' manually.",
            icon="warning",
        )
        if not answer:
            return

        input_dir = project_path / "input_sources"
        for source_name in names:
            filepath = input_dir / source_name
            try:
                if filepath.exists():
                    filepath.unlink()
            except OSError as exc:
                messagebox.showerror(
                    "Delete failed",
                    f"Could not delete '{source_name}':\n{exc}",
                )

            # Remove from input_source_numbers
            if source_name in self.project_settings.input_source_numbers:
                del self.project_settings.input_source_numbers[source_name]

        save_project_settings(project_path, self.project_settings)

        # Remove related entries from the execution list
        if self.execution_mgr is not None:
            for source_name in names:
                self.execution_mgr.remove_jobs_for_source(source_name)
            # Refresh the execution window if open
            if self.execution_window is not None and self.execution_window.winfo_exists():
                self.execution_window.schedule_refresh()

        self._refresh_input_sources()

    # ── Ctrl-A select all ────────────────────────────────────────

    def _on_ctrl_a(self, event: tk.Event) -> str | None:  # type: ignore[type-arg]
        """Select all items in the focused Treeview, if any."""
        return self._select_all_in_focused_tree(event)

    def _on_key_a(self, event: tk.Event) -> str | None:  # type: ignore[type-arg]
        """Select all items in the focused Treeview on plain 'a' press.

        Only fires when focus is on a Treeview (not text entries).
        """
        widget = event.widget
        # Only handle 'a' when a Treeview has focus (skip entries, text widgets)
        w = widget
        while w is not None:
            if isinstance(w, ttk.Treeview):
                return self._select_all_in_focused_tree(event)
            if isinstance(w, (tk.Entry, ttk.Entry, tk.Text)):
                return None
            w = getattr(w, "master", None)
        return None

    def _on_select_all(self) -> None:
        """Select all items in whichever treeview last had focus."""
        # Try each tree — select all in the one that has focus, or the first non-empty one
        for tree in (self.input_sources_tree, self.available_tree, self.executed_tree):
            if str(tree.focus_get()) == str(tree) or tree.focus_get() is tree:
                children = tree.get_children()
                if children:
                    tree.selection_set(children)
                return
        # Fallback: select all in available_tree
        children = self.available_tree.get_children()
        if children:
            self.available_tree.selection_set(children)

    def _select_all_in_focused_tree(self, event: tk.Event) -> str | None:  # type: ignore[type-arg]
        """Select all items in the focused Treeview, if any."""
        widget = event.widget
        # Walk up to find the Treeview that contains the focused widget
        while widget is not None:
            if isinstance(widget, ttk.Treeview):
                children = widget.get_children()
                if children:
                    widget.selection_set(children)
                return "break"
            widget = getattr(widget, "master", None)
        return None

    # ── Space key handlers for checkbox toggling ──────────────────

    def _on_check_selected(self) -> None:
        """Toggle checkboxes for all selected (highlighted) items in available_tree."""
        for item in self.available_tree.selection():
            self._toggle_check(self.available_tree, item, "check")
        self._update_add_to_execution_style()
        self._save_checked_available_scenarios()

    def _on_available_space(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        """Toggle checkboxes for all selected (highlighted) items in available_tree."""
        for item in self.available_tree.selection():
            self._toggle_check(self.available_tree, item, "check")
        self._update_add_to_execution_style()
        self._save_checked_available_scenarios()

    def _on_executed_space(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        """Toggle checkboxes for all selected (highlighted) items in executed_tree."""
        for item in self.executed_tree.selection():
            self._toggle_check(self.executed_tree, item, "check")
        self._update_output_status()
        self._save_checked_executed_scenarios()

    def _on_check_executed(self) -> None:
        """Check all executed scenarios if any are unchecked, otherwise uncheck all."""
        children = self.executed_tree.get_children()
        all_checked = all(
            self.executed_tree.item(item, "values")[0] == CHECK_ON
            for item in children
            if self.executed_tree.item(item, "values")
        )
        new_state = CHECK_OFF if all_checked else CHECK_ON
        for item in children:
            self.executed_tree.set(item, "check", new_state)
        self._update_output_status()
        self._save_checked_executed_scenarios()

    def _on_key_e(self, event: tk.Event) -> str | None:  # type: ignore[type-arg]
        """Toggle checkboxes on selected executed scenarios on 'e' press."""
        widget = event.widget
        w = widget
        while w is not None:
            if isinstance(w, (tk.Entry, ttk.Entry, tk.Text)):
                return None
            w = getattr(w, "master", None)
        self._on_check_executed()
        return "break"

    # ── Executed scenarios management ────────────────────────────

    def _refresh_executed_scenarios(self) -> None:
        """Scan for executed scenario results and repopulate the executed_tree.

        Preserves existing checkbox states so that auto-checked scenarios
        (from ``_refresh_and_autocheck_scenario``) are not unchecked.
        """
        # Remember which scenarios are currently checked
        previously_checked: set[str] = set()
        for item in self.executed_tree.get_children():
            values = self.executed_tree.item(item, "values")
            if values and values[0] == CHECK_ON:
                previously_checked.add(values[2])  # scenario_name

        # Clear executed scenarios tree
        for item in self.executed_tree.get_children():
            self.executed_tree.delete(item)

        if not self.exec_scenario_mgr:
            return

        executed = self.exec_scenario_mgr.scan_executed()

        # Try to map source numbers from current input sources
        source_number_map: dict[str, int] = {}
        if self.input_source_mgr:
            for scenario_info in self.input_source_mgr.get_all_scenarios():
                source_number_map[scenario_info.name] = scenario_info.source_number

        # On first load (no items were in tree), restore from saved settings
        if not previously_checked:
            previously_checked = set(self.project_settings.checked_executed_scenarios)

        for info in executed:
            src_num = source_number_map.get(info.name, info.source_number)
            check_char = CHECK_ON if info.name in previously_checked else CHECK_OFF
            # Check if plots exist for this scenario
            plot_dir = self.exec_scenario_mgr.project_path / "output_plots" / info.name
            has_plots = plot_dir.is_dir() and any(plot_dir.iterdir())
            view_text = "\u25b6 View" if has_plots else ""
            self.executed_tree.insert(
                "",
                "end",
                values=(check_char, src_num, info.name, view_text, info.timestamp),
            )

        # Apply current sort mode
        if self._executed_sort_mode == "alpha":
            self._sort_tree_items(self.executed_tree, col_index=2, numeric=False)
        elif self._executed_sort_mode == "number":
            self._sort_tree_items(
                self.executed_tree, col_index=1, numeric=True, secondary_col=4,
            )
        else:  # timestamp
            self._sort_tree_items(self.executed_tree, col_index=4, numeric=False)

        self._update_output_status()

    def _on_executed_selection_changed(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        """Update output status indicators when executed_tree selection changes."""
        self._update_output_status()

    def _repopulate_available_tree(self, select_indices: list[int] | None = None) -> None:
        """Repopulate available_tree from the manager's scenario list and re-select items."""
        if not self.avail_scenario_mgr:
            return

        # Remember which scenario keys had checkboxes checked (by source_num|name)
        checked_keys: set[str] = set()
        children = self.available_tree.get_children()
        for item in children:
            values = self.available_tree.item(item, "values")
            if values and values[0] == CHECK_ON:
                key = f"{values[1]}|{values[2]}"
                checked_keys.add(key)

        # Clear and repopulate
        for item in self.available_tree.get_children():
            self.available_tree.delete(item)

        # Build editing source number set for red background tagging
        editing_source_numbers: set[int] = set()
        for item in self.input_sources_tree.get_children():
            values = self.input_sources_tree.item(item, "values")
            if values and values[3] == STATUS_EDITING:
                try:
                    editing_source_numbers.add(int(values[2]))
                except (ValueError, IndexError):
                    pass
        self.available_tree.tag_configure("editing_source", background="#662222")

        scenarios = self.avail_scenario_mgr.scenarios
        for scenario in scenarios:
            key = f"{scenario.source_number}|{scenario.name}"
            check_char = CHECK_ON if key in checked_keys else CHECK_OFF
            tags: tuple[str, ...] = ()
            if scenario.source_number in editing_source_numbers:
                tags = ("editing_source",)
            self.available_tree.insert(
                "",
                "end",
                values=(check_char, scenario.source_number, scenario.name),
                tags=tags,
            )

        # Apply current sort mode
        if self._available_sort_mode == "alpha":
            self._sort_tree_items(self.available_tree, col_index=2, numeric=False)
        else:
            self._sort_tree_items(self.available_tree, col_index=1, numeric=True)

        # Re-select items at new positions
        if select_indices:
            new_children = self.available_tree.get_children()
            for idx in select_indices:
                if 0 <= idx < len(new_children):
                    self.available_tree.selection_add(new_children[idx])

        self._update_add_to_execution_style()

    def _save_scenario_order(self) -> None:
        """Persist the current scenario order to project settings."""
        if not self.avail_scenario_mgr or not self.current_project:
            return
        self.project_settings.scenario_order = self.avail_scenario_mgr.get_order()
        project_path = get_projects_dir() / self.current_project
        save_project_settings(project_path, self.project_settings)

    # ── Add to execution list handler ────────────────────────────

    def _on_add_selected_to_execution(self) -> None:
        """Add selected (highlighted) scenarios from available_tree to execution."""
        if not self.avail_scenario_mgr or not self.current_project:
            return
        scenario_by_key: dict[str, ScenarioInfo] = {
            f"{s.source_number}|{s.name}": s
            for s in self.avail_scenario_mgr.scenarios
        }
        selected: list[ScenarioInfo] = []
        for item in self.available_tree.selection():
            values = self.available_tree.item(item, "values")
            if values:
                key = f"{values[1]}|{values[2]}"
                if key in scenario_by_key:
                    selected.append(scenario_by_key[key])
        if selected:
            self._add_scenarios_to_execution(selected)

    @safe_callback
    def _on_add_to_execution(self) -> None:
        """Add checked scenarios from available_tree to the execution queue."""
        if not self.avail_scenario_mgr or not self.current_project:
            return
        checked = self.avail_scenario_mgr.get_checked_scenarios(self.available_tree)
        if checked:
            self._add_scenarios_to_execution(checked)

    def _add_scenarios_to_execution(self, scenarios: list[ScenarioInfo]) -> None:
        """Shared logic for adding scenarios to the execution queue."""
        if not self.avail_scenario_mgr or not self.current_project:
            return

        if self._xlsx_converting_sources:
            messagebox.showwarning(
                "Conversion in progress",
                "An xlsx conversion is already running.\n"
                "Wait for it to finish or abort it first.",
                parent=self,
            )
            return

        checked = scenarios
        if not checked:
            logger.info("No scenarios checked for execution")
            return

        # Check if any checked scenarios come from editing sources (Change 5)
        editing_source_numbers: set[int] = set()
        for item in self.input_sources_tree.get_children():
            values = self.input_sources_tree.item(item, "values")
            if not values:
                continue
            source_name = values[1]
            # Existing check: treeview status column shows editing
            if values[3] == STATUS_EDITING:
                try:
                    editing_source_numbers.add(int(values[2]))
                except (ValueError, IndexError):
                    pass
                continue
            # Enhanced check: sqlite sources with a running editor process
            if source_name.lower().endswith(".sqlite") and self.input_source_mgr:
                filepath = self.input_source_mgr.input_dir / source_name
                if self.db_editor_mgr.has_uncommitted_changes(filepath):
                    try:
                        editing_source_numbers.add(int(values[2]))
                    except (ValueError, IndexError):
                        pass

        editing_scenarios = [
            s for s in checked if s.source_number in editing_source_numbers
        ]
        if editing_scenarios:
            names_str = ", ".join(s.name for s in editing_scenarios)
            result = messagebox.askyesno(
                "Unsaved changes",
                f"Some selected scenarios come from input sources that "
                f"may have unsaved changes (DB editor still open):\n\n"
                f"  {names_str}\n\n"
                f"Continue anyway?",
                parent=self,
            )
            if not result:
                return

        # Filter out scenarios already in the execution queue (pending or running)
        self._ensure_execution_mgr()
        assert self.execution_mgr is not None

        already_queued: set[str] = set()
        for job in self.execution_mgr.get_jobs():
            if job.status in (JobStatus.PENDING, JobStatus.RUNNING):
                already_queued.add(job.scenario_name)

        duplicates = [s for s in checked if s.name in already_queued]
        new_scenarios = [s for s in checked if s.name not in already_queued]

        if duplicates and not new_scenarios:
            dup_names = ", ".join(s.name for s in duplicates)
            messagebox.showwarning(
                "Already in execution list",
                f"All selected scenarios are already pending or running:\n\n"
                f"  {dup_names}\n\n"
                f"They will not be added again.",
                parent=self,
            )
            return

        if duplicates:
            dup_names = ", ".join(s.name for s in duplicates)
            messagebox.showinfo(
                "Some already queued",
                f"These scenarios are already pending or running and "
                f"will be skipped:\n\n  {dup_names}",
                parent=self,
            )

        if not new_scenarios:
            return

        self._pending_execution_scenarios = new_scenarios
        names = [s.name for s in new_scenarios]
        logger.info("Scenarios queued for execution: %s", names)

        # Partition into xlsx and sqlite scenarios
        xlsx_scenarios = [s for s in new_scenarios
                          if s.source_name.lower().endswith((".xlsx", ".xls", ".ods"))]
        sqlite_scenarios = [s for s in new_scenarios if s not in xlsx_scenarios]

        # Dispatch sqlite scenarios immediately
        if sqlite_scenarios:
            added = self.execution_mgr.add_jobs(sqlite_scenarios)
            self.execution_mgr.start()
            self._update_execution_menu_style()
            self._open_or_raise_execution_window()
            if added and self.execution_window is not None:
                self.execution_window.select_job(added[-1].job_id)

        # Handle xlsx scenarios through pre-conversion
        if xlsx_scenarios:
            self._start_xlsx_preconversion(xlsx_scenarios)

    # ── Execution menu handler ───────────────────────────────────

    def _on_plot_menu(self) -> None:
        """Open the PlotDialog to configure plot settings."""
        if not self.current_project:
            messagebox.showinfo(
                "No project",
                "Please select or create a project first.",
            )
            return
        project_path = get_projects_dir() / self.current_project
        PlotDialog(self, project_path, self.project_settings)

    def _on_execution_menu(self) -> None:
        """Open the ExecutionWindow (or raise it if already open)."""
        self._ensure_execution_mgr()
        self._open_or_raise_execution_window()

    def _on_view_results(self) -> None:
        """Open the ResultViewer, or update its scenarios if already open."""
        if not self.current_project:
            messagebox.showinfo(
                "No project",
                "Please select or create a project first.",
            )
            return
        self._open_or_raise_result_viewer()

    def _open_or_raise_result_viewer(self) -> None:
        """Open a new ResultViewer or raise an existing one.

        If the viewer is already open, refreshes its scenario data
        from the current checked executed scenarios.
        """
        if (
            self._result_viewer is not None
            and self._result_viewer.winfo_exists()
        ):
            # Update the viewer's scenario data before raising
            self._result_viewer._on_update()
            self._result_viewer.deiconify()
            self._result_viewer.attributes("-topmost", True)
            self._result_viewer.attributes("-topmost", False)
            self._result_viewer.focus_force()
            self._update_view_results_btn()
            return

        project_path = get_projects_dir() / self.current_project
        self._result_viewer = ResultViewer(
            master=self,
            project_path=project_path,
            settings=self.project_settings,
            scenario_db_map=self._get_scenario_db_map(),
        )
        self._update_view_results_btn()
        # When the viewer closes, revert button text
        self._result_viewer.bind("<Destroy>", lambda e: self._update_view_results_btn())

    def _update_view_results_btn(self) -> None:
        """Update the Results viewer button text and style based on viewer and scenario state."""
        viewer_open = (
            self._result_viewer is not None
            and self._result_viewer.winfo_exists()
        )
        if viewer_open:
            self.view_results_btn.configure(text="Update view scenarios")
        else:
            self.view_results_btn.configure(text="Results viewer")

        # Blue accent when there are checked executed scenarios (results to show)
        has_checked = False
        for item in self.executed_tree.get_children():
            values = self.executed_tree.item(item, "values")
            if values and values[0] == CHECK_ON:
                has_checked = True
                break
        if has_checked:
            self.view_results_btn.configure(style="Accent.TButton")
        else:
            self.view_results_btn.configure(style="TButton")

    def _get_scenario_db_map(self) -> dict[str, Path]:
        """Build a mapping of scenario names to database paths.

        For each input source:
        - .sqlite files are used directly from input_sources/
        - .xlsx files use the converted .sqlite from intermediate/
        """
        db_map: dict[str, Path] = {}
        if self.input_source_mgr is None:
            return db_map

        project_path = get_projects_dir() / self.current_project
        for source in self.input_source_mgr._sources:
            if source.file_type == "sqlite":
                db_path = project_path / "input_sources" / source.name
            else:
                # xlsx/ods -> look for converted sqlite in intermediate/
                stem = Path(source.name).stem
                db_path = project_path / "intermediate" / f"{stem}.sqlite"

            if db_path.is_file():
                for scenario in source.scenarios:
                    db_map[scenario] = db_path

        return db_map

    # ── Execution helpers ────────────────────────────────────────

    def _ensure_execution_mgr(self) -> None:
        """Create the ExecutionManager if it does not exist yet."""
        if self.execution_mgr is not None:
            return
        if not self.current_project:
            return

        project_path = get_projects_dir() / self.current_project
        self.execution_mgr = ExecutionManager(
            project_path=project_path,
            settings=self.project_settings,
            on_status_change=self._on_job_status_change,
            on_all_finished=self._on_all_jobs_finished,
        )

    def _open_or_raise_execution_window(self) -> None:
        """Open a new ExecutionWindow or raise an existing one."""
        if (
            self.execution_window is not None
            and self.execution_window.winfo_exists()
        ):
            self.execution_window.deiconify()
            self.execution_window.attributes("-topmost", True)
            self.execution_window.attributes("-topmost", False)
            self.execution_window.focus_force()
            self._update_execution_menu_style()
            return

        if self.execution_mgr is None:
            return

        self.execution_window = ExecutionWindow(self, self.execution_mgr)
        self._update_execution_menu_style()
        # When the window closes, revert button accent
        self.execution_window.bind(
            "<Destroy>", lambda e: self._update_execution_menu_style(),
        )

    def _on_job_status_change(self, job: ExecutionJob) -> None:
        """Callback from ExecutionManager when a job's status changes.

        This is called from worker threads, so we schedule GUI updates
        via ``self.after()``.
        """
        if job.status == JobStatus.SUCCESS:
            self.after(
                0, self._refresh_and_autocheck_scenario,
                job.scenario_name, job.finish_timestamp,
            )
        elif job.status in (JobStatus.FAILED, JobStatus.KILLED):
            self.after(0, self._refresh_executed_scenarios)

        # Update execution menu button highlight (Change 3)
        self.after(0, self._update_execution_menu_style)

        # Notify the execution window (if open)
        if (
            self.execution_window is not None
            and self.execution_window.winfo_exists()
        ):
            self.execution_window.schedule_refresh()

    def _on_all_jobs_finished(self) -> None:
        """Callback when all execution jobs have completed.

        Called from the scheduler thread -- schedule GUI updates safely.
        """
        self.after(0, self._refresh_executed_scenarios)
        self.after(0, self._update_output_status)
        self.after(0, self._update_execution_menu_style)

    # ── Delete results handler ───────────────────────────────────

    @safe_callback
    def _on_delete_results(self) -> None:
        """Delete output files for selected (highlighted) scenarios in executed_tree."""
        if not self.exec_scenario_mgr or not self.current_project:
            return

        # Gather selected scenario names from the executed tree
        selected_names: list[str] = []
        for item in self.executed_tree.selection():
            values = self.executed_tree.item(item, "values")
            if values:
                selected_names.append(values[2])  # scenario_name column

        if not selected_names:
            return

        names_str = "\n  ".join(selected_names)
        answer = messagebox.askyesno(
            "Delete results",
            f"Are you sure you want to permanently delete results for "
            f"the selected scenarios?\n\n  {names_str}\n\n"
            f"This will remove all output files "
            f"(parquet, plots, Excel, CSV).",
            icon="warning",
        )
        if not answer:
            return

        self.exec_scenario_mgr.delete_results(selected_names)
        self._refresh_executed_scenarios()

    # ── Output status indicator updates ──────────────────────────

    def _update_output_status(self) -> None:
        """Update the output status labels based on checked executed scenarios."""
        if not self.exec_scenario_mgr:
            self._reset_output_status()
            self._update_output_frame_style()
            return

        # Gather checked scenario names from the executed tree
        checked_names: list[str] = []
        for item in self.executed_tree.get_children():
            values = self.executed_tree.item(item, "values")
            if values and values[0] == CHECK_ON:
                checked_names.append(values[2])  # scenario_name column

        if not checked_names:
            self._reset_output_status()
            self._update_output_frame_style()
            self._update_view_results_btn()
            return

        # Check per-scenario outputs
        outputs = self.exec_scenario_mgr.check_outputs(checked_names)

        # Aggregate: all checked scenarios have the output?
        all_have_plots = all(outputs[n]["has_plots"] for n in checked_names)
        all_have_excel = all(outputs[n]["has_excel"] for n in checked_names)
        all_have_csvs = all(outputs[n]["has_csvs"] for n in checked_names)

        # Check comparison outputs
        comp = self.exec_scenario_mgr.check_comparison_outputs(checked_names)

        # For comparison outputs, also verify that the currently checked
        # scenarios match the scenarios that were used to generate them.
        checked_set = set(checked_names)
        comp_plots_match = comp["has_comp_plots"] and (
            checked_set == set(self.project_settings.comp_plots_scenarios)
        )
        comp_excel_match = comp["has_comp_excel"] and (
            checked_set == set(self.project_settings.comp_excel_scenarios)
        )

        status_map = {
            "scen_plots": ("Re-plot scenarios", all_have_plots),
            "scen_excel": ("Scenarios to Excel", all_have_excel),
            "scen_csvs": ("Scenarios to csvs", all_have_csvs),
            "comp_plots": ("Comparison pngs", comp_plots_match),
            "comp_excel": ("Comparison to Excel", comp_excel_match),
        }

        for key, (full_name, has_output) in status_map.items():
            self.output_status_labels[key].configure(text=full_name)
            if has_output:
                self._output_spinners[key].configure(text="\u2713")
                if key in self.output_action_btns:
                    self.output_action_btns[key].configure(style="TButton")
            elif key in self._output_action_failed:
                self._output_spinners[key].configure(text="\u2298")
                if key in self.output_action_btns:
                    self.output_action_btns[key].configure(style="Grey.TButton")
            else:
                self._output_spinners[key].configure(text="  ")
                if key in self.output_action_btns:
                    self.output_action_btns[key].configure(style="Grey.TButton")

        self._update_output_frame_style()
        self._update_view_results_btn()

    def _reset_output_status(self) -> None:
        """Reset all output status labels to the default (no output) state."""
        default_names = {
            "scen_plots": "Re-plot scenarios",
            "scen_excel": "Scenarios to Excel",
            "scen_csvs": "Scenarios to csvs",
            "comp_plots": "Comparison pngs",
            "comp_excel": "Comparison to Excel",
        }
        for key, full_name in default_names.items():
            self.output_status_labels[key].configure(text=full_name)
            self._output_spinners[key].configure(text="  ")
            if key in self.output_action_btns:
                self.output_action_btns[key].configure(style="Grey.TButton")

    # ── Auto-generate checkbox management ─────────────────────────

    def _load_auto_gen_vars(self) -> None:
        """Set auto-generate BooleanVars from the loaded project settings."""
        s = self.project_settings
        self.auto_scen_plots_var.set(s.auto_generate_scen_plots)
        self.auto_scen_excels_var.set(s.auto_generate_scen_excels)
        self.auto_scen_csvs_var.set(s.auto_generate_scen_csvs)
        self.auto_comp_plots_var.set(s.auto_generate_comp_plots)
        self.auto_comp_excel_var.set(s.auto_generate_comp_excel)

    def _on_auto_gen_toggled(self, *_args: object) -> None:
        """Save auto-generate settings when any checkbox is toggled."""
        self.project_settings.auto_generate_scen_plots = self.auto_scen_plots_var.get()
        self.project_settings.auto_generate_scen_excels = self.auto_scen_excels_var.get()
        self.project_settings.auto_generate_scen_csvs = self.auto_scen_csvs_var.get()
        self.project_settings.auto_generate_comp_plots = self.auto_comp_plots_var.get()
        self.project_settings.auto_generate_comp_excel = self.auto_comp_excel_var.get()

        if self.current_project:
            project_path = get_projects_dir() / self.current_project
            save_project_settings(project_path, self.project_settings)

    # ── Output generation button handlers ─────────────────────────

    def _save_current_settings(self) -> None:
        """Persist the current project settings to disk."""
        if self.current_project:
            project_path = get_projects_dir() / self.current_project
            save_project_settings(project_path, self.project_settings)

    def _get_checked_executed_names(self) -> list[str]:
        """Return scenario names that are checked in the executed_tree."""
        checked: list[str] = []
        for item in self.executed_tree.get_children():
            values = self.executed_tree.item(item, "values")
            if values and values[0] == CHECK_ON:
                checked.append(values[2])  # scenario_name column
        return checked

    def _ensure_output_action_mgr(self) -> OutputActionManager | None:
        """Return the OutputActionManager, creating it if needed."""
        if self.output_action_mgr is not None:
            # Ensure execution_mgr reference is up-to-date (it may have
            # been None when the OutputActionManager was first created
            # during project load, before any scenarios were queued).
            if self.output_action_mgr._execution_mgr is None:
                self._ensure_execution_mgr()
                self.output_action_mgr._execution_mgr = self.execution_mgr
            return self.output_action_mgr
        if not self.current_project:
            return None
        project_path = get_projects_dir() / self.current_project
        self._ensure_execution_mgr()
        self.output_action_mgr = OutputActionManager(
            project_path=project_path,
            settings=self.project_settings,
            execution_mgr=self.execution_mgr,
            on_complete=self._on_output_action_complete,
        )
        return self.output_action_mgr

    def _start_output_action(self, key: str) -> list[str] | None:
        """Common setup for output action buttons.

        Returns the checked scenario names, or None if there are none or
        the action manager is not available.
        """
        names = self._get_checked_executed_names()
        if not names:
            messagebox.showinfo("No selection", "Please select executed scenarios first.")
            return None
        mgr = self._ensure_output_action_mgr()
        if mgr is None:
            return None
        self._output_action_failed.discard(key)
        self.output_status_labels[key].configure(state="disabled")
        self._start_spinner(key)
        # Show progress in the execution window
        self._open_or_raise_execution_window()
        return names

    @safe_callback
    def _on_gen_scen_plots(self) -> None:
        """Generate plots for checked executed scenarios."""
        names = self._start_output_action("scen_plots")
        if names and self.output_action_mgr:
            self.output_action_mgr.run_scenario_plots(names)

    @safe_callback
    def _on_gen_scen_excel(self) -> None:
        """Generate Excel files for checked executed scenarios."""
        names = self._start_output_action("scen_excel")
        if names and self.output_action_mgr:
            self.output_action_mgr.run_scenario_excel(names)

    @safe_callback
    def _on_gen_scen_csvs(self) -> None:
        """Generate CSV files for checked executed scenarios."""
        names = self._start_output_action("scen_csvs")
        if names and self.output_action_mgr:
            self.output_action_mgr.run_scenario_csvs(names)

    @safe_callback
    def _on_gen_comp_plots(self) -> None:
        """Generate comparison plots for checked executed scenarios."""
        names = self._start_output_action("comp_plots")
        if names and self.output_action_mgr:
            self.project_settings.comp_plots_scenarios = list(names)
            self._save_current_settings()
            self.output_action_mgr.run_comparison_plots(names)

    @safe_callback
    def _on_gen_comp_excel(self) -> None:
        """Generate comparison Excel for checked executed scenarios."""
        names = self._start_output_action("comp_excel")
        if names and self.output_action_mgr:
            self.project_settings.comp_excel_scenarios = list(names)
            self._save_current_settings()
            self.output_action_mgr.run_comparison_excel(names)

    # ── Spinner animation for output actions ─────────────────────

    def _start_spinner(self, key: str) -> None:
        """Start an animated hourglass spinner next to the output action button."""
        label = self._output_spinners.get(key)
        if label is None:
            return
        self._animate_spinner(key, 0)

    def _stop_spinner(self, key: str) -> None:
        """Stop the animated hourglass spinner."""
        timer_id = self._spinner_timer_ids.pop(key, None)
        if timer_id is not None:
            try:
                self.after_cancel(timer_id)
            except Exception:
                pass
        label = self._output_spinners.get(key)
        if label is not None:
            label.configure(text="  ")

    def _animate_spinner(self, key: str, frame_idx: int) -> None:
        """Update the spinner label to the next animation frame."""
        label = self._output_spinners.get(key)
        if label is None or not self.winfo_exists():
            return
        char = _SPINNER_FRAMES[frame_idx % len(_SPINNER_FRAMES)]
        label.configure(text=char)
        next_idx = (frame_idx + 1) % len(_SPINNER_FRAMES)
        self._spinner_timer_ids[key] = self.after(
            500, self._animate_spinner, key, next_idx
        )

    def _on_output_action_complete(self, action_name: str, success: bool) -> None:
        """Callback from OutputActionManager when an action finishes.

        Called from a worker thread -- schedule GUI updates via ``self.after()``.
        """
        self.after(0, self._handle_output_action_done, action_name, success)

    def _handle_output_action_done(self, action_name: str, success: bool) -> None:
        """Re-enable the action button and refresh output status (runs on main thread)."""
        self._stop_spinner(action_name)
        if not success:
            self._output_action_failed.add(action_name)
        if action_name in self._output_spinners:
            if success:
                self._output_spinners[action_name].configure(text="\u2713")
            else:
                self._output_spinners[action_name].configure(text="\u2298")
        if action_name in self.output_status_labels:
            self.output_status_labels[action_name].configure(state="normal")
        self._update_output_status()
        if not success:
            logger.warning("Output action '%s' finished with errors", action_name)

    # ── Show / Open button handlers ───────────────────────────────

    @safe_callback
    def _on_show_scen_plots(self) -> None:
        """Open the output_plots folder."""
        if not self.current_project:
            return
        project_path = get_projects_dir() / self.current_project
        plots_dir = project_path / "output_plots"
        if plots_dir.is_dir():
            try:
                open_folder(plots_dir)
            except OSError as exc:
                logger.warning("Could not open plots folder: %s", exc)

    @safe_callback
    def _on_show_scen_excel(self) -> None:
        """Open the output_excel folder."""
        if not self.current_project:
            return
        project_path = get_projects_dir() / self.current_project
        excel_dir = project_path / "output_excel"
        if excel_dir.is_dir():
            try:
                open_folder(excel_dir)
            except OSError as exc:
                logger.warning("Could not open Excel folder: %s", exc)

    @safe_callback
    def _on_show_scen_csvs(self) -> None:
        """Open the CSV folder for the checked executed scenarios."""
        if not self.current_project:
            return
        names = self._get_checked_executed_names()
        if not names:
            return
        project_path = get_projects_dir() / self.current_project
        if len(names) == 1:
            csv_dir = project_path / "output_csv" / names[0]
        else:
            csv_dir = project_path / "output_csv"
        if csv_dir.is_dir():
            try:
                open_folder(csv_dir)
            except OSError as exc:
                logger.warning("Could not open CSV folder: %s", exc)

    @safe_callback
    def _on_show_comp_plots(self) -> None:
        """Open the output_plot_comparisons folder."""
        if not self.current_project:
            return
        project_path = get_projects_dir() / self.current_project
        comp_dir = project_path / "output_plot_comparisons"
        if comp_dir.is_dir():
            try:
                open_folder(comp_dir)
            except OSError as exc:
                logger.warning("Could not open comparison plots folder: %s", exc)

    @safe_callback
    def _on_show_comp_excel(self) -> None:
        """Open the first .xlsx from output_plot_comparisons/."""
        if not self.current_project:
            return
        mgr = self._ensure_output_action_mgr()
        if mgr is None:
            return
        xlsx = mgr.find_comparison_excel()
        if xlsx is not None:
            try:
                open_file_in_default_app(xlsx)
            except OSError as exc:
                logger.warning("Could not open comparison Excel: %s", exc)

    # ── View scenario plots (from executed_tree view column) ──────

    def _view_scenario_plots(self, scenario_name: str) -> None:
        """Open the ResultViewer for the given scenario."""
        if not self.current_project:
            return
        # Auto-check this scenario in the executed tree so the viewer picks it up
        for item in self.executed_tree.get_children():
            values = self.executed_tree.item(item, "values")
            if values and values[2] == scenario_name:
                self.executed_tree.set(item, "check", CHECK_ON)
                break
        self._save_checked_executed_scenarios()
        self._open_or_raise_result_viewer()

    # ── Checkbox state persistence helpers ────────────────────────

    def _collect_checked_input_sources(self) -> None:
        """Read checked source names from the tree into project settings."""
        checked: list[str] = []
        for item in self.input_sources_tree.get_children():
            values = self.input_sources_tree.item(item, "values")
            if values and values[0] == CHECK_ON:
                checked.append(values[1])  # name column
        self.project_settings.checked_input_sources = checked

    def _save_checked_input_sources(self) -> None:
        """Persist checked input source names to settings."""
        self._collect_checked_input_sources()
        if self.current_project:
            project_path = get_projects_dir() / self.current_project
            save_project_settings(project_path, self.project_settings)

    def _collect_checked_available_scenarios(self) -> None:
        """Read checked available scenario keys from the tree into project settings."""
        checked: list[str] = []
        for item in self.available_tree.get_children():
            values = self.available_tree.item(item, "values")
            if values and values[0] == CHECK_ON:
                key = f"{values[1]}|{values[2]}"
                checked.append(key)
        self.project_settings.checked_available_scenarios = checked

    def _save_checked_available_scenarios(self) -> None:
        """Persist checked available scenario keys to settings."""
        self._collect_checked_available_scenarios()
        if self.current_project:
            project_path = get_projects_dir() / self.current_project
            save_project_settings(project_path, self.project_settings)

    def _collect_checked_executed_scenarios(self) -> None:
        """Read checked executed scenario names from the tree into project settings."""
        checked: list[str] = []
        for item in self.executed_tree.get_children():
            values = self.executed_tree.item(item, "values")
            if values and values[0] == CHECK_ON:
                checked.append(values[2])  # scenario_name column
        self.project_settings.checked_executed_scenarios = checked

    def _save_checked_executed_scenarios(self) -> None:
        """Persist checked executed scenario names to settings."""
        self._collect_checked_executed_scenarios()
        if self.current_project:
            project_path = get_projects_dir() / self.current_project
            save_project_settings(project_path, self.project_settings)
