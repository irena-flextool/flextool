from __future__ import annotations

import hashlib
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox, simpledialog

from flextool._resources import package_data_path
from flextool.gui.project_utils import (
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
from flextool.gui.check_tree import CheckTreeController
from flextool.gui.data_models import GlobalSettings, ProjectSettings, ScenarioInfo
from flextool.gui.input_sources import InputSourceManager
from flextool.gui.scenario_lists import (
    AvailableScenarioManager,
    ExecutedScenarioManager,
    prune_dangling_scenario_state,
)
from flextool.gui.execution_manager import ExecutionJob, ExecutionManager, JobStatus
from flextool.gui.execution_window import ExecutionWindow
from flextool.gui.output_actions import OutputActionManager
from flextool.gui.result_viewer import ResultViewer
from flextool.gui.db_version_check import check_and_upgrade_database
from flextool.gui.dialogs.migration_consent_dialog import ask_external_migration_consent
from flextool.gui.dialogs.migration_progress_dialog import MigrationProgressDialog
from flextool.gui.dialogs.plot_dialog import PlotDialog
from flextool.gui.error_handling import safe_callback
from flextool.gui.platform_utils import (
    open_file_in_default_app,
    open_folder,
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


# Unicode checkbox characters for Treeview checkbox simulation.
# Checked = filled square U+25A0, unchecked = empty square U+25A1. Solid-vs-empty
# is the highest-contrast pair and both render large; the previous checked glyph
# U+25A3 differed from U+25A1 only by a small inner mark, so it read as "small /
# hard to tell if checked", most acutely on Windows.
CHECK_ON = "\u25a0"   # ■
CHECK_OFF = "\u25a1"  # □
STATUS_OK = "\u2713"      # ✓
STATUS_ERR = "\u2717"     # ✗
STATUS_EMPTY = "\u25cb"   # (no scenarios; Geometric Shapes circle — was U+2300 diameter sign)
STATUS_EDITING = "\u25b6" # open in editor (Geometric Shapes; was U+23F3)
STATUS_RETIRED = "\u2298" # ghost row: input file gone, only results remain

# Input-sources tree iid prefixes
_EXT_IID_PREFIX = "ext:"      # external reference
_GHOST_IID_PREFIX = "ghost:"  # retired source (file gone, results survive)


def _source_name_from_iid(iid: str) -> str:
    """Strip the ``ext:`` prefix used for external-reference tree iids."""
    if iid.startswith(_EXT_IID_PREFIX):
        return iid[len(_EXT_IID_PREFIX):]
    return iid


def _is_ghost_iid(iid: str) -> bool:
    """True for a retired "ghost" input-source row (no live file)."""
    return iid.startswith(_GHOST_IID_PREFIX)

# Animated spinner frames for output action progress indication
_SPINNER_FRAMES = ["\u25d0", "\u25d3", "\u25d1", "\u25d2"]  # rotating circle (Geometric Shapes; render on Windows Tk, unlike the old U+29D6/7 hourglasses)

# Glyphs for the File-outputs "Gen." button. The button is clickable to
# (re-)generate the output; the recycle ring (U+21BB) is the persistent
# affordance that signals "press me", and the trailing mark reports state.
_GEN_RING = "\u21bb"                  # recycle ring -- the clickable affordance
# The status slot is ALWAYS filled (ring + mark) so the ring and the mark keep a
# fixed horizontal position in every row; a "no data yet" row shows an en-dash
# placeholder rather than a blank, which would otherwise re-centre the ring.
_GEN_PENDING = f"{_GEN_RING} \u2013"  # not produced yet (ring + en-dash placeholder)
_GEN_EXISTS = f"{_GEN_RING} \u2713"   # produced (ring + check)
_GEN_FAILED = f"{_GEN_RING} \u2717"   # last run failed (ring + cross)


class MainWindow(tk.Tk):
    """Main application window for FlexTool GUI.

    All widgets are created and placed via the grid geometry manager.
    """

    def __init__(self, initial_theme: str = "dark") -> None:
        # Mark the process DPI-aware BEFORE the Tk root window is created.
        # Windows ignores the request once an HWND exists; setting it here
        # ensures winfo_screenwidth()/natural-size queries report the true,
        # un-virtualized screen so windows are sized right from the start.
        from flextool.gui.platform_utils import set_process_dpi_awareness
        set_process_dpi_awareness()

        super().__init__()

        # Marshal worker-thread → main-thread GUI work through a queue pumped
        # by the Tk event loop (which also runs during modal ``wait_window``
        # loops). tkinter is not thread-safe: even ``self.after()`` registers a
        # Tcl command and raises "main thread is not in main loop" when called
        # off the main thread on some platforms (notably macOS), so worker
        # threads must never touch Tk directly — they call ``post_to_main``.
        self._main_thread_queue: queue.Queue = queue.Queue()
        self.after(50, self._pump_main_thread_queue)

        # Last known "is a newer version available?" result (None = unknown).
        self._update_available: bool | None = None

        # ── DPI scaling — must come before any widget/font access ─
        from flextool.gui.platform_utils import (
            apply_dpi_scaling, scale_theme_fonts,
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

        # Load global settings early so the user's saved font size drives
        # setup_fonts on the very first call — avoids a 10pt → saved-size
        # flash that would otherwise be visible at every startup. The
        # YAML loader is pure file IO (no Tk dependency), so it's safe to
        # call here before any widgets exist.
        self.global_settings = load_global_settings(get_projects_dir())

        # Configure role-aware named fonts (body/heading/tooltip/code).
        # code_font_size_pt = 0 means "auto" → derive as body + 2 so logs
        # render at a comfortably readable size next to body text.
        from flextool.gui.ui_metrics import setup_fonts
        _body_pt = self.global_settings.font_size_pt or 10
        _code_pt = self.global_settings.code_font_size_pt or (_body_pt + 2)
        setup_fonts(self, body_pt=_body_pt, code_pt=_code_pt)

        self.title("FlexTool")

        # ── Window icon (works on Windows, macOS, Linux) ──────────
        # docs/ is shipped in the source repo but not in the PyPI wheel.
        # In editable installs the icon is found; in wheel installs it
        # silently doesn't load (the iconphoto call is non-fatal).
        icon_path = Path(__file__).resolve().parent.parent.parent / "docs" / "irena_flextool_favicon.png"
        if icon_path.exists():
            try:
                self._icon_image = tk.PhotoImage(file=str(icon_path))
                self.iconphoto(True, self._icon_image)
            except Exception:
                pass  # Non-fatal: skip if image can't be loaded

        # ── Font metrics for DPI-aware sizing ─────────────────────
        from flextool.gui.ui_metrics import get_metrics
        _metrics = get_metrics(self)
        self._char_width: int = _metrics.cw
        self._line_height: int = _metrics.lh
        # Use the named heading font for bold labels so live size changes
        # via _set_font_size reach widgets that already exist. Passing the
        # string name (rather than a Font object copy) keeps tkinter
        # bound to the named font.
        self._bold_font = "TkHeadingFont"

        # ── Treeview row height and selection visibility ──────────
        # Add ~25% vertical padding so rows don't touch the row above —
        # otherwise large fonts (high-DPI) clip and small fonts make
        # the trees look cramped.  Min of 24px keeps trees readable on
        # setups where DPI auto-detection underreports or where the
        # user has chosen a small font size.
        style = ttk.Style()
        row_height = _metrics.row_height
        style.configure("Treeview", rowheight=row_height)

        # Make LabelFrame titles track the body font live (named-font
        # string reference, not a snapshot Font object).
        style.configure("TLabelframe.Label", font="TkDefaultFont")

        # Selected rows stay blue regardless of which tree currently
        # holds focus. The "selected !focus" mapping is listed first so
        # it shadows sv_ttk's dark-grey unfocused-selection default.
        style.map(
            "Treeview",
            background=[
                ("selected !focus", "#3874c8"),
                ("selected", "#3874c8"),
            ],
            foreground=[
                ("selected !focus", "#ffffff"),
                ("selected", "#ffffff"),
            ],
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

        # Compact buttons for the File-outputs table — small vertical
        # padding so the rows sit tightly together. (horizontal, vertical)
        style.configure("Output.TButton", padding=(5, 0))
        style.configure("Output.Grey.TButton", foreground="#888888", padding=(5, 0))
        style.map(
            "Output.Grey.TButton",
            foreground=[("active", "#888888"), ("disabled", "#888888")],
        )
        # The Gen. button text is LEFT-anchored so the ↻ ring sits at a fixed x
        # in every row.  Combined with the always-filled status slot
        # (↻ –/✓/✗) the trailing mark also stays put; centred text would shift
        # the ring whenever a row's mark is absent or a different width.
        style.configure("Gen.TButton", padding=(5, 0), anchor="w")

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

        # Tooltip for input source status column
        self._input_status_tip: tk.Toplevel | None = None

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
        # Top-section column layout:
        #   col 0  input sources tree           (stretch)
        #   col 1  input source buttons         (narrow)
        #   col 2  side menu: Debug / themes / Png / Exec / Results
        #   col 3  spacer
        #   col 4  spacer (weight=1, eats slack so col 5 is pushed right)
        #   col 5  File outputs box             (fixed width, sticky="ne")
        # The lower section reuses the same columns: executed_tree
        # spans cols 2-5 with sticky="nsew" and stretches via col 4's
        # weight; available_tree at cols 0-1 stretches via col 0.
        outer.columnconfigure(0, weight=1)   # input / available scenarios
        outer.columnconfigure(1, weight=0)   # source buttons
        outer.columnconfigure(2, weight=0)   # side menu
        outer.columnconfigure(3, weight=0)   # spacer
        outer.columnconfigure(4, weight=1)   # spacer (push col 5 right)
        outer.columnconfigure(5, weight=0)   # File outputs (right-aligned)

        # ── Row 0: Project selector ──────────────────────────────────
        row = 0
        ttk.Label(outer, text="Project:").grid(row=row, column=0, sticky="w", padx=(0, 5))

        self.project_combo = ttk.Combobox(outer, state="readonly", width=30)
        self.project_combo.grid(row=row, column=0, sticky="w", padx=(60, 5))
        self.project_combo.bind("<<ComboboxSelected>>", self._on_project_combo_selected)
        self.project_combo.bind("<F2>", self._on_combo_rename)
        self.project_combo.bind("<Double-Button-1>", self._on_combo_rename)

        self.project_menu_btn = ttk.Button(
            outer, text="Projects…", command=self._open_project_dialog
        )
        self.project_menu_btn.grid(row=row, column=1, sticky="w", padx=5)

        # Top-right corner of row 0: UI settings (font size cascade + reset
        # window layout) with Update FlexTool to its right.
        top_right = ttk.Frame(outer)
        top_right.grid(row=row, column=5, sticky="e", padx=5)

        self.ui_settings_btn = ttk.Button(
            top_right, text="UI settings…", command=self._on_ui_settings_btn
        )
        self.ui_settings_btn.pack(side="left", padx=(0, 5))

        self.update_btn = ttk.Button(
            top_right, text="Update FlexTool…", command=self._on_update_flextool
        )
        self.update_btn.pack(side="left")

        # Debug / themes / Png settings / Execution jobs / Results viewer
        # used to occupy row 0 and a separate bottom-of-section row; both
        # now live in the side menu column to the left of the Outputs
        # table (built further below). _theme_var is created here so the
        # Project popup can refer to it, but the radio buttons themselves
        # are placed inside the side menu.
        self._theme_var = tk.StringVar(value=initial_theme)
        # Tri-state debug verbosity: "off" | "basic" | "full".  Legacy
        # boolean settings.yaml entries are normalised to one of these
        # by ``settings_io.load_project_settings``.
        self.debug_var = tk.StringVar(value="off")
        self.debug_var.trace_add("write", self._on_auto_gen_toggled)
        self.save_memory_var = tk.BooleanVar(value=False)
        self.save_memory_var.trace_add("write", self._on_auto_gen_toggled)

        # ── Solver options vars ────────────────────────────────────
        # Persisted into ProjectSettings; ExecutionManager._build_run_command
        # appends the matching CLI flag only when the value differs from
        # the default below.  See ProjectSettings docstrings for the
        # allowed value sets.  The five vars below back the Solver
        # options dialog (launched from the "Solver options…" button in
        # the side menu, above the Debug radio group).
        self.solver_log_level_var = tk.StringVar(value="normal")
        self.solver_time_limit_var = tk.IntVar(value=0)
        self.matrix_file_format_var = tk.StringVar(value="mps")
        self.scaling_var = tk.StringVar(value="full")
        self.presolve_var = tk.StringVar(value="choose")

        for _v in (
            self.solver_log_level_var, self.solver_time_limit_var,
            self.matrix_file_format_var, self.scaling_var,
            self.presolve_var,
        ):
            _v.trace_add("write", self._on_auto_gen_toggled)

        # ── Row 1: Section headers ───────────────────────────────────
        from flextool.gui.hover_tooltip import attach_tooltip

        row = 1
        input_lbl = ttk.Label(outer, text="Input sources", font=self._bold_font)
        input_lbl.grid(row=row, column=0, sticky="sw", pady=(10, 2))
        attach_tooltip(input_lbl, (
            "Input files (.xlsx/.sqlite) for FlexTool scenarios.\n"
            "\n"
            "  \u2022 Double-click to edit a source\n"
            "  \u2022 Right-click for actions (Edit, Convert, Delete)\n"
            "  \u2022 Click column headers to sort"
        ))

        # "File outputs" header sits above the bordered LabelFrame
        # below — placed in col 5 with sticky="sw" and the same left
        # padding as the box so its left edge lines up with the box.
        outputs_lbl = ttk.Label(outer, text="File outputs", font=self._bold_font)
        outputs_lbl.grid(
            row=row, column=5, sticky="sw", padx=(20, 0), pady=(4, 0)
        )
        attach_tooltip(outputs_lbl, (
            "Per-checked-executed-scenario output artefacts on disk.\n"
            "\n"
            "  • Auto-gen: produce this output automatically after\n"
            "    every scenario run.\n"
            "  • Gen.: click the ↻ to (re-)generate. ↻ = not produced\n"
            "    yet, ↻ ✓ = exists, ↻ ✗ = last run failed.\n"
            "  • Show: open the folder (or the file, for Comparison\n"
            "    SpineDB / Excel) in the system file manager.\n"
            "\n"
            "Full results are always stored as parquet files; these\n"
            "exports are derived artefacts. The result viewer does\n"
            "not depend on any of them — it reads the parquets\n"
            "directly."
        ))

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
        self.input_sources_tree.column("check", width=int(cw * 4.6), minwidth=int(cw * 4.6), stretch=False)
        self.input_sources_tree.column("name", width=cw * 25, minwidth=cw * 12)
        self.input_sources_tree.column("number", width=cw * 4, minwidth=cw * 4, stretch=False)
        self.input_sources_tree.column("status", width=cw * 3, minwidth=cw * 3, stretch=False)
        self.input_sources_tree.grid(row=0, column=0, sticky="nsew")

        input_scroll = ttk.Scrollbar(input_frame, orient="vertical", command=self.input_sources_tree.yview)
        input_scroll.grid(row=0, column=1, sticky="ns")
        self._setup_autohide_scrollbar(self.input_sources_tree, input_scroll)

        # Checkbox click + space handled by shared CheckTreeController.
        self._input_sources_check_ctrl = CheckTreeController(
            self.input_sources_tree,
            check_column="check",
            checked_glyph=CHECK_ON,
            unchecked_glyph=CHECK_OFF,
            on_toggle=self._on_input_sources_toggled,
            can_check=lambda iid: not _is_ghost_iid(iid),
        )
        self.input_sources_tree.bind("<Double-1>", self._on_input_source_dblclick)
        self.input_sources_tree.bind("<B1-Motion>", self._on_tree_drag_select)
        self.input_sources_tree.bind("<<TreeviewSelect>>", lambda _e: self._update_input_button_states())
        self.input_sources_tree.bind("<Button-3>", self._on_input_source_right_click)
        self.input_sources_tree.bind("<Motion>", self._on_input_source_motion)
        self.input_sources_tree.bind("<Leave>", lambda e: self._hide_input_status_tip())
        self.input_sources_tree.bind("<Shift-Up>", self._on_shift_arrow_up)
        self.input_sources_tree.bind("<Shift-Down>", self._on_shift_arrow_down)
        self.input_sources_tree.bind("<FocusIn>", self._on_tree_focus_in)

        # --- Input source buttons (col 1, rows 2-8) ---
        btn_col = 1
        self.add_source_btn = ttk.Button(
            outer, text="Add…", width=8, command=self._on_add_source
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

        # --- Unified Outputs table (col 2-4, rows 2-7) ----------------
        # Columns: Output name | Auto-gen | Status | Show
        # Auto-gen is the same boolean that previously lived in the
        # separate "Auto-generate" checkbox group; the status column
        # doubles as the manual generate trigger (click to regenerate).
        self.auto_scen_plots_var = tk.BooleanVar(value=True)
        self.auto_scen_excels_var = tk.BooleanVar(value=False)
        self.auto_scen_csvs_var = tk.BooleanVar(value=True)
        self.auto_comp_plots_var = tk.BooleanVar(value=True)
        self.auto_comp_excel_var = tk.BooleanVar(value=False)
        self.auto_comp_spinedb_var = tk.BooleanVar(value=False)

        # The legacy green-tint affordance is dropped (it didn't survive the
        # move to ttk and the per-row \u21bb \u2713/\u2717 state makes it redundant).
        # sv_ttk's Card.TFrame: a themed bordered panel WITHOUT the empty-title
        # top reservation that ttk.LabelFrame imposes (~21 px), which was the
        # stubborn gap between the "File outputs" header and the first row.
        self.output_frame = ttk.Frame(outer, style="Card.TFrame", padding=(8, 2))
        self.output_frame.grid(
            row=2, column=5, rowspan=7,
            sticky="ne", padx=(20, 0), pady=0,
        )

        # Column header row uses ttk.Label so it inherits the theme.
        ttk.Label(self.output_frame, text="Output", anchor="w").grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 2),
        )
        ttk.Label(self.output_frame, text="Auto-gen").grid(
            row=0, column=1, padx=(0, 8), pady=(0, 2),
        )
        # "Gen." (not "Status"): the cell is a button that (re-)generates
        # the output; the ↻ ring makes that evident. The action column header
        # is left blank — the Show/Open button labels are self-explanatory.
        ttk.Label(self.output_frame, text="Gen.").grid(
            row=0, column=2, padx=(0, 8), pady=(0, 2),
        )
        ttk.Label(self.output_frame, text="").grid(
            row=0, column=3, padx=(0, 0), pady=(0, 2),
        )

        # (display_name, key, auto_var, show_label) for each row.
        # ``show_label`` is "Show" for folder targets, "Open" for the
        # single-file Comparison Excel.
        output_info: list[tuple[str, str, tk.BooleanVar, str]] = [
            ("Scenario pngs",    "scen_plots", self.auto_scen_plots_var,  "Show"),
            ("Scenario Excels",  "scen_excel", self.auto_scen_excels_var, "Show"),
            ("Scenario csvs",    "scen_csvs",  self.auto_scen_csvs_var,   "Show"),
            ("Comparison pngs",   "comp_plots",   self.auto_comp_plots_var,   "Show"),
            ("Comparison SpineDB", "comp_spinedb", self.auto_comp_spinedb_var, "Open"),
            ("Comparison Excel",  "comp_excel",   self.auto_comp_excel_var,   "Open"),
        ]

        self.output_status_labels: dict[str, ttk.Button] = {}
        self.output_action_btns: dict[str, ttk.Button] = {}
        # _output_spinners is kept as an alias to output_status_labels
        # so the existing spinner animation code keeps working without
        # changes \u2014 the status cell now plays both roles.
        self._output_spinners: dict[str, ttk.Button | tk.Label] = {}
        self._spinner_timer_ids: dict[str, str] = {}

        # Display names used by status updaters; kept in sync with the
        # output_info table so renames only need to happen in one place.
        self._output_display_names: dict[str, str] = {
            key: name for name, key, _v, _s in output_info
        }

        _gen_commands: dict[str, str] = {
            "scen_plots": "_on_gen_scen_plots",
            "scen_excel": "_on_gen_scen_excel",
            "scen_csvs":  "_on_gen_scen_csvs",
            "comp_plots": "_on_gen_comp_plots",
            "comp_spinedb": "_on_gen_comp_spinedb",
            "comp_excel": "_on_gen_comp_excel",
        }
        _show_commands: dict[str, str] = {
            "scen_plots": "_on_show_scen_plots",
            "scen_excel": "_on_show_scen_excel",
            "scen_csvs":  "_on_show_scen_csvs",
            "comp_plots": "_on_show_comp_plots",
            "comp_spinedb": "_on_show_comp_spinedb",
            "comp_excel": "_on_show_comp_excel",
        }

        _row_tooltips: dict[str, str] = {
            "scen_plots": (
                "Per-scenario PNG plots written under\n"
                "<output>/<scenario>/plots/.\n"
                "Layout and time ranges follow Png settings;\n"
                "the result viewer is independent of these files."
            ),
            "scen_excel": (
                "Per-scenario Excel workbook with the main result\n"
                "tables (flows, capacities, costs, …) written under\n"
                "<output>/<scenario>/."
            ),
            "scen_csvs": (
                "Per-scenario CSV exports of the result tables\n"
                "under <output>/<scenario>/csv/.\n"
                "Handy for piping into other tools."
            ),
            "comp_plots": (
                "PNG plots overlaying the checked executed scenarios,\n"
                "written under <output>/comparison/plots/.\n"
                "Regenerated whenever the set of checked scenarios\n"
                "changes."
            ),
            "comp_spinedb": (
                "Single SpineDB (results.sqlite) in the project root,\n"
                "holding the processed results of all executed scenarios\n"
                "as separate alternatives.\n"
                "\n"
                "Produced only during the solve — tick Auto-gen and\n"
                "(re-)run the scenarios. It cannot be regenerated from\n"
                "the stored parquet files, so the Status button only\n"
                "reports whether the file exists."
            ),
            "comp_excel": (
                "Single Excel workbook comparing the checked executed\n"
                "scenarios side by side, written to\n"
                "<output>/comparison/."
            ),
        }

        for i, (display_name, key, auto_var, show_label) in enumerate(output_info):
            row_i = i + 1  # header is row 0
            name_lbl = ttk.Label(
                self.output_frame, text=display_name, anchor="w",
            )
            name_lbl.grid(row=row_i, column=0, sticky="w", padx=(0, 8), pady=0)
            attach_tooltip(name_lbl, _row_tooltips[key])

            cb = ttk.Checkbutton(self.output_frame, variable=auto_var)
            cb.grid(row=row_i, column=1, padx=(0, 8), pady=0)

            # "Gen." button: the recycle ring (↻) is the persistent
            # click-to-(re)generate affordance; a trailing ✓/✗ reports
            # state. Compact style keeps the row short.
            status_btn = ttk.Button(
                self.output_frame, text=_GEN_PENDING, width=4,
                style="Gen.TButton",
                command=getattr(self, _gen_commands[key]),
            )
            status_btn.grid(row=row_i, column=2, padx=(0, 8), pady=0)
            self.output_status_labels[key] = status_btn
            self._output_spinners[key] = status_btn  # alias

            action_btn = ttk.Button(
                self.output_frame, text=show_label, width=5,
                style="Output.TButton",
                command=getattr(self, _show_commands[key]),
            )
            action_btn.grid(row=row_i, column=3, sticky="w", padx=(0, 0), pady=0)
            self.output_action_btns[key] = action_btn

        # Trace auto-generate vars to save settings on toggle.
        self.auto_scen_plots_var.trace_add("write", self._on_auto_gen_toggled)
        self.auto_scen_excels_var.trace_add("write", self._on_auto_gen_toggled)
        self.auto_scen_csvs_var.trace_add("write", self._on_auto_gen_toggled)
        self.auto_comp_plots_var.trace_add("write", self._on_auto_gen_toggled)
        self.auto_comp_spinedb_var.trace_add("write", self._on_auto_gen_toggled)
        self.auto_comp_excel_var.trace_add("write", self._on_auto_gen_toggled)

        # --- Side menu column (col 2): two vertical groups -------------
        # Top:    Debug · themes · Png settings
        # Bottom: Execution jobs · Results viewer (anchored to the
        #         bottom of the section, touching the divider)
        # A weighted-1 spacer row in the middle absorbs the slack so the
        # two groups separate cleanly.
        from flextool.gui.hover_tooltip import attach_tooltip as _attach_tip
        side_menu = ttk.Frame(outer)
        side_menu.grid(
            row=2, column=2, rowspan=7, sticky="nsew", padx=(20, 0), pady=2,
        )
        side_menu.rowconfigure(4, weight=1)  # stretch spacer

        self.save_memory_cb = ttk.Checkbutton(
            side_menu, text="Save memory", variable=self.save_memory_var,
        )
        self.save_memory_cb.grid(row=0, column=0, sticky="w", pady=(0, 4))
        _attach_tip(self.save_memory_cb, (
            "Run scenarios with --save-memory.\n"
            "\n"
            "Builds the LP, writes it to a temp MPS file, drops\n"
            "everything Python-side, then spawns a separate HiGHS\n"
            "subprocess to actually solve. The parent (FlexTool +\n"
            "polars data) sits idle while the child does its active-\n"
            "solve work, so the two never compound in the same process.\n"
            "\n"
            "Solve takes more time (file operations) - depends on model.\n"
            "Disables warm-LP reuse across cascade iterations\n"
            "(each sub-solve rebuilds fully).\n"
            "Try when models run out of memory."
        ))

        # ── Solver options launcher (modal dialog) ─────────────────
        # Bundles the v56 CLI knobs that don't influence results — only
        # "ways to solve".  Clicking the button opens a ``Toplevel`` modal
        # populated with five controls (Log level, Time limit, Matrix
        # file format, Scaling, Presolve).  See ProjectSettings for the
        # per-control defaults and the dialog factory below for the
        # control layout / tooltips; ExecutionManager appends each flag
        # only when it differs from the default so the engine command
        # line stays clean on the common path.  Placed above the Debug
        # radio group so it sits with the other run-time knobs.
        self.solver_opts_btn = ttk.Button(
            side_menu, width=22,
            text="Solver options…",
            command=self._open_solver_options_dialog,
        )
        self.solver_opts_btn.grid(row=1, column=0, sticky="w", pady=(0, 4))
        _attach_tip(self.solver_opts_btn, (
            "Open the Solver options dialog.\n"
            "\n"
            "Controls how scenarios reach the solver (log verbosity,\n"
            "wall-clock limit, on-disk matrix format, FlexTool\n"
            "autoscaler strategy, HiGHS presolve).\n"
            "\n"
            "These are 'ways to solve' knobs — they do not change\n"
            "the optimisation results (usually).  HiGHS thread count is\n"
            "controlled separately by the Execution jobs window\n"
            "(execution_limits.max_cores_per_job)."
        ))

        # Debug: tri-state radio group (Off / Basic / Full).  The
        # values map 1:1 to the ``--debug={off,basic,full}`` CLI flag in
        # ``cmd_run_flextool``; ExecutionManager appends ``--csv-dump``
        # only when "Full" is selected so the heavy intermediate-CSV
        # I/O stays opt-in.
        debug_frame = ttk.Frame(side_menu)
        debug_frame.grid(row=2, column=0, sticky="w", pady=(0, 4))
        debug_label = ttk.Label(debug_frame, text="Debug:")
        debug_label.pack(side="left", padx=(0, 4))
        debug_radios: list[ttk.Radiobutton] = []
        for _text, _value in (("Off", "off"), ("Basic", "basic"), ("Full", "full")):
            rb = ttk.Radiobutton(
                debug_frame, text=_text, variable=self.debug_var,
                value=_value,
            )
            rb.pack(side="left", padx=(0, 4))
            debug_radios.append(rb)
        self.debug_cb = debug_frame  # retained name for external refs
        _attach_tip(debug_frame, (
            "Diagnostic verbosity for scenario execution.\n"
            "\n"
            "  • Off    — no extra flags.\n"
            "  • Basic  — --debug=basic: verbose memory checkpoints\n"
            "             and DEBUG-level engine logging. No tracemalloc;\n"
            "             negligible runtime overhead.\n"
            "  • Full   — --debug=full + --csv-dump: Basic plus\n"
            "             tracemalloc-backed memory diagnostics CSV\n"
            "             and retained intermediate input CSVs.\n"
            "             Tracemalloc instruments every Python\n"
            "             allocation and typically slows allocation-\n"
            "             heavy phases by 2-5×. Use only for\n"
            "             allocation-regression investigations."
        ))

        self.plot_menu_btn = ttk.Button(
            side_menu, text="Png settings…", width=22,
            command=self._on_plot_menu,
        )
        self.plot_menu_btn.grid(row=3, column=0, sticky="w", pady=(0, 2))

        # Row 5 is the stretch spacer; bottom group lives in rows 6-7.
        self.execution_menu_btn = ttk.Button(
            side_menu, text="Execution jobs…", width=22,
            command=self._on_execution_menu,
        )
        self.execution_menu_btn.grid(row=6, column=0, sticky="sw", pady=2)

        # Width 22 to fit the alternate label "Update view scenarios"
        # when the viewer is already open.
        self.view_results_btn = ttk.Button(
            side_menu, text="Results viewer…", width=22,
            command=self._on_view_results,
        )
        self.view_results_btn.grid(row=7, column=0, sticky="sw", pady=(2, 0))

        # ── Separator ────────────────────────────────────────────────
        sep = ttk.Separator(outer, orient="horizontal")
        sep.grid(row=9, column=0, columnspan=6, sticky="ew", pady=10)

        # ── Row 10: Scenario section headers ─────────────────────────
        row = 10
        avail_lbl = ttk.Label(outer, text="Available scenarios [V]", font=self._bold_font)
        avail_lbl.grid(row=row, column=0, columnspan=2, sticky="sw", pady=(0, 2))
        attach_tooltip(avail_lbl, (
            "Scenarios found in checked input sources.\n"
            "\n"
            "  \u2022 V \u2014 focus this list\n"
            "  \u2022 A \u2014 select all rows\n"
            "  \u2022 Space \u2014 check/uncheck selected\n"
            "  \u2022 Right-click for actions\n"
            "  \u2022 Click column headers to sort"
        ))

        exec_lbl = ttk.Label(outer, text="Executed scenarios [X]", font=self._bold_font)
        exec_lbl.grid(
            row=row, column=2, columnspan=4, sticky="sw", padx=(20, 0), pady=(0, 2)
        )
        attach_tooltip(exec_lbl, (
            "Scenarios with results in output_parquet/.\n"
            "\n"
            "  \u2022 X \u2014 focus this list\n"
            "  \u2022 E \u2014 check/uncheck all\n"
            "  \u2022 Space \u2014 check/uncheck selected\n"
            "  \u2022 Right-click for actions\n"
            "  \u2022 Click column headers to sort"
        ))

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
        self.available_tree.column("check", width=int(cw * 3.45), minwidth=int(cw * 3.45), stretch=False)
        self.available_tree.column("source_num", width=cw * 3, minwidth=cw * 3, stretch=False)
        self.available_tree.column("scenario_name", width=cw * 25, minwidth=cw * 12, stretch=True)
        self.available_tree.grid(row=0, column=0, sticky="nsew")

        avail_scroll = ttk.Scrollbar(avail_frame, orient="vertical", command=self.available_tree.yview)
        avail_scroll.grid(row=0, column=1, sticky="ns")
        self._setup_autohide_scrollbar(self.available_tree, avail_scroll)

        self._available_check_ctrl = CheckTreeController(
            self.available_tree,
            check_column="check",
            checked_glyph=CHECK_ON,
            unchecked_glyph=CHECK_OFF,
            on_toggle=self._on_available_toggled,
        )
        self.available_tree.bind("<B1-Motion>", self._on_tree_drag_select)
        self.available_tree.bind("<Button-3>", self._on_available_right_click)
        self.available_tree.bind("<Shift-Up>", self._on_shift_arrow_up)
        self.available_tree.bind("<Shift-Down>", self._on_shift_arrow_down)
        self.available_tree.bind("<FocusIn>", self._on_tree_focus_in)

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
        self.executed_tree.column("check", width=int(cw * 3.45), minwidth=int(cw * 3.45), stretch=False)
        self.executed_tree.column("source_num", width=cw * 3, minwidth=cw * 3, stretch=False)
        self.executed_tree.column("scenario_name", width=cw * 29, minwidth=cw * 12, stretch=True)
        self.executed_tree.column("view", width=cw * 3, minwidth=cw * 3, stretch=False, anchor="center")
        self.executed_tree.column("timestamp", width=int(cw * 24.7), minwidth=int(cw * 24.7), stretch=False)
        self.executed_tree.grid(row=0, column=0, sticky="nsew")

        # No row-level tag for View — Treeview tags color the entire row.
        # "orphan" tag: executed scenarios whose source number does not match
        # any current input source (parent source was deleted). Greyed so
        # the user can see them and decide whether to delete the outputs.
        self.executed_tree.tag_configure("orphan", foreground="#888888")

        exec_scroll = ttk.Scrollbar(exec_frame, orient="vertical", command=self.executed_tree.yview)
        exec_scroll.grid(row=0, column=1, sticky="ns")
        self._setup_autohide_scrollbar(self.executed_tree, exec_scroll)

        self.executed_tree.bind("<Button-1>", self._on_executed_click)
        self.executed_tree.bind("<B1-Motion>", self._on_tree_drag_select)
        # CheckTreeController bound after the legacy click handler so the
        # legacy handler's "view" column branch still runs first; the
        # controller's <Button-1> early-returns on non-check columns.
        self._executed_check_ctrl = CheckTreeController(
            self.executed_tree,
            check_column="check",
            checked_glyph=CHECK_ON,
            unchecked_glyph=CHECK_OFF,
            on_toggle=self._on_executed_toggled,
        )
        self.executed_tree.bind("<<TreeviewSelect>>", self._on_executed_selection_changed)
        self.executed_tree.bind("<Button-3>", self._on_executed_right_click)
        self.executed_tree.bind("<Shift-Up>", self._on_shift_arrow_up)
        self.executed_tree.bind("<Shift-Down>", self._on_shift_arrow_down)
        self.executed_tree.bind("<FocusIn>", self._on_tree_focus_in)

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
        # 'v' focuses available scenarios, 'x' focuses executed scenarios
        self.bind_all("<Key-v>", self._on_key_v)
        self.bind_all("<Key-x>", self._on_key_x)

        # ── Window close handler ─────────────────────────────────────
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # When the application is re-activated (Alt-Tab back, un-minimize,
        # etc.) the ttk selection draws dim until a widget is repainted.
        # Nudge each tree's selection to force it to re-render with the
        # focused (blue) colors immediately.
        self.bind("<FocusIn>", self._on_main_focus_in)

        # Track DPI changes when the window moves between monitors.
        # Tk does not re-flow widgets on scaling changes, so we limit
        # ourselves to refreshing font sizes — a partial fix that beats
        # leaving the user with frozen-tiny or frozen-huge text until they
        # restart FlexTool.
        from flextool.gui.ui_metrics import monitor_dpi
        self._last_dpi = monitor_dpi(self)
        self.bind("<Configure>", self._on_main_configure, add="+")

        # ── Window sizing: compute from actual widget layout ─────────
        self.update_idletasks()
        nat_width = self.winfo_reqwidth()
        nat_height = self.winfo_reqheight()
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        # Width: the natural layout width, but never wider than the screen.
        # Without this clamp a DPI-aware layout whose natural width exceeds a
        # small (e.g. laptop) display runs off-screen, and the weight=1 spacer
        # columns stretch the content apart instead of staying compact.
        # Height: fill the screen minus taskbar space so the tree lists show
        # as many rows as fit — extra height becomes more rows, not whitespace.
        win_width = min(nat_width, screen_w)
        win_height = max(nat_height, screen_h - lh * 4)
        self.geometry(f"{win_width}x{win_height}+0+0")
        # minsize must also stay within the screen, or the window can't be
        # shrunk to fit and the WM may re-inflate it past the display edge.
        self.minsize(min(nat_width, screen_w), min(nat_height, screen_h))

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

        # Check for a newer FlexTool version in the background; if one is
        # available, highlight the Update button (blue, like other call-outs).
        self.after(1500, self._check_update_async)

        # Once per environment, verify that the installed polars build runs
        # on this CPU; if it crashes natively, offer a one-click swap to
        # polars-lts-cpu (the whole reason a "wrong wheel for this machine"
        # install otherwise dies with an opaque 0xC0000005 mid-solve).
        self.after(2200, self._check_polars_async)

    def _check_update_async(self) -> None:
        """Probe for a newer version off the UI thread, then update the button.

        Skipped when the "Check for updates on startup" preference is off (set
        in the Update FlexTool dialog) or when ``FLEXTOOL_NO_UPDATE_CHECK`` is
        set — either way, no outbound request is made.
        """
        if os.environ.get("FLEXTOOL_NO_UPDATE_CHECK"):
            return
        if not self.global_settings.check_updates_on_startup:
            return

        def _worker() -> None:
            try:
                from flextool.update_flextool import install_info

                available = install_info.update_available()
            except Exception:
                logger.debug("Update check failed", exc_info=True)
                available = False
            self.post_to_main(self._apply_update_indicator, available)

        threading.Thread(target=_worker, daemon=True).start()

    def _apply_update_indicator(self, available: bool | None) -> None:
        """Blue-highlight the Update button when an update is available."""
        self._update_available = available
        if not hasattr(self, "update_btn") or not self.update_btn.winfo_exists():
            return
        self.update_btn.configure(
            style="Accent.TButton" if available else "TButton"
        )
        if available:
            from flextool.gui.hover_tooltip import attach_tooltip

            attach_tooltip(self.update_btn, "A newer FlexTool version is available.")

    # ── Native solver-stack compatibility self-check ──────────────
    def _check_polars_async(self, force: bool = False, from_crash: bool = False) -> None:
        """Probe — off the UI thread — whether the native solver stack
        (polars, highspy/HiGHS) runs on this machine.

        Runs once per environment: skipped when the cached fingerprint still
        matches (and not *force*d), or when ``FLEXTOOL_NO_ENV_CHECK`` is set.
        The probe runs the imports in a child process, so a native crash is
        observed via its exit code rather than taking down the GUI.

        *from_crash* means this was triggered by a scenario that just died
        natively; if the solver stack then probes clean, the crash was
        elsewhere (DB layer, mixed/conda env, network drive) and we say so.
        """
        if os.environ.get("FLEXTOOL_NO_ENV_CHECK"):
            return

        from flextool import env_check

        if not force and self.global_settings.polars_check_fingerprint == env_check.env_fingerprint():
            return

        def _worker() -> None:
            try:
                probe = env_check.probe_solver_stack()
            except Exception:
                logger.debug("solver-stack probe failed to run", exc_info=True)
                return
            self.post_to_main(self._apply_solver_probe, probe, from_crash)

        threading.Thread(target=_worker, daemon=True).start()

    def _apply_solver_probe(self, probe, from_crash: bool = False) -> None:
        """Handle a solver-stack probe result on the main thread."""
        from flextool import env_check

        if probe.ok:
            # Healthy: remember this environment so we don't probe again
            # until the interpreter or a solver build changes.
            self.global_settings.polars_check_fingerprint = env_check.env_fingerprint()
            save_global_settings(get_projects_dir(), self.global_settings)
            if from_crash:
                # A scenario crashed natively yet the solver libraries import
                # and run fine — so the crash was not a solver wheel. Point at
                # the real culprit class instead of leaving the user guessing.
                messagebox.showwarning(
                    "Scenario crashed — check your Python environment",
                    "The scenario stopped with a native crash, but FlexTool's "
                    "solver libraries (polars and HiGHS) import and run fine on "
                    "this computer.\n\n" + env_check.UNFIXABLE_HELP,
                )
            return

        if not probe.is_native_fault:
            # An ordinary error (e.g. a package not importable) that a build
            # swap would not fix — log it and stay out of the user's way.
            logger.warning("solver self-check: %s", probe.summary())
            return

        if not env_check.has_remediation(probe.failed_component):
            # A native crash with no package-level remedy (e.g. polar_high).
            messagebox.showerror("FlexTool cannot run on this computer yet", env_check.UNFIXABLE_HELP)
            return

        self._offer_solver_fix(probe)

    def _offer_solver_fix(self, probe) -> None:
        """Ask the user to let FlexTool re-install the compatible build."""
        from flextool import env_check

        comp = probe.failed_component
        what = {
            "polars": "the 'polars' data library",
            "highspy": "the HiGHS solver (highspy)",
        }.get(comp, "a solver library")
        proceed = messagebox.askyesno(
            "FlexTool cannot run on this computer yet",
            f"{what} crashed when FlexTool tested it on this computer — "
            "scenarios will not run until this is fixed.\n\n"
            "FlexTool can automatically re-install the compatible version "
            "for you. Progress is shown in the execution window.\n\n"
            "Re-install the compatible version now?",
            icon="warning",
        )
        if not proceed:
            return

        from flextool.gui.execution_manager import JobType

        steps = env_check.remediation_steps(comp, sys.executable)
        self._run_cli_job(
            steps,
            job_type=JobType.ENV_REPAIR,
            description=f"Fix {comp} compatibility",
            action_key="solver_compat_fix",
            intro=env_check.remediation_banner(comp),
            on_finish=self._on_solver_fix_finished,
        )

    def _on_solver_fix_finished(self, success: bool, _output: str) -> None:
        """After the fix job: re-probe and cache the verdict (main thread)."""
        from flextool import env_check

        reprobe = env_check.probe_solver_stack()
        if self.execution_mgr is not None:
            # Surface the re-check result in the same log the fix wrote to.
            for job in self.execution_mgr.get_jobs():
                if job.action_key == "solver_compat_fix":
                    self.execution_mgr.append_stdout(job.job_id, "\n" + reprobe.summary())
                    break

        if reprobe.ok:
            self.global_settings.polars_check_fingerprint = env_check.env_fingerprint()
            save_global_settings(get_projects_dir(), self.global_settings)
            messagebox.showinfo(
                "Compatibility fixed",
                "FlexTool re-installed the compatible solver libraries and "
                "verified they run on this computer. You can run scenarios now.",
            )
        else:
            messagebox.showerror("Still not working", env_check.UNFIXABLE_HELP)

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

    # ── UI settings button ───────────────────────────────────────────

    def _styled_popup_menu(self, parent: tk.Misc) -> tk.Menu:
        """Return a ``tk.Menu`` themed to match the current ttk theme.

        Native ``tk.Menu`` ignores ttk styling, so on the sv_ttk dark
        theme the radio bullet renders as a dark dot on a dark background
        and the default menu font does not track ``TkDefaultFont``.  We
        pull bg/fg from the ttk style and bind the body-font name so the
        popup matches the rest of the UI.
        """
        style = ttk.Style()
        bg = style.lookup("TFrame", "background") or self.cget("background")
        fg = style.lookup("TLabel", "foreground") or "#ffffff"
        return tk.Menu(
            parent,
            tearoff=0,
            background=bg,
            foreground=fg,
            activebackground=fg,
            activeforeground=bg,
            selectcolor=fg,
            font="TkDefaultFont",
            borderwidth=0,
        )

    def _install_menu_hover_dismiss(
        self,
        top_menu: tk.Menu,
        submenus: list[tk.Menu],
        delay_ms: int = 400,
    ) -> None:
        """Unpost ``top_menu`` shortly after the mouse leaves all menus.

        Native ``tk.Menu`` only closes on click; users expect popup
        menus to also dismiss on mouse-leave.  The delay keeps the menu
        open while the cursor briefly crosses the border into a cascade
        submenu (cascades fire ``<Leave>`` on the parent).
        """
        self._menu_dismiss_id: str | None = None

        def _cancel() -> None:
            if self._menu_dismiss_id is not None:
                try:
                    self.after_cancel(self._menu_dismiss_id)
                except tk.TclError:
                    pass
                self._menu_dismiss_id = None

        def _schedule(_event: object) -> None:
            _cancel()
            self._menu_dismiss_id = self.after(delay_ms, top_menu.unpost)

        for m in (top_menu, *submenus):
            m.bind("<Leave>", _schedule, add="+")
            m.bind("<Enter>", lambda _e: _cancel(), add="+")

    def _on_ui_settings_btn(self) -> None:
        """Show the UI-settings popup menu under the button.

        Holds two items: a cascading "UI font size" menu (presets plus
        Custom…) and "Reset window layout".  These were previously part
        of the Project popup menu but moved here so the Project button
        can be a direct action that opens Manage projects.
        """
        menu = self._styled_popup_menu(self)

        # UI font size cascade — radio for presets + Custom...
        size_menu = self._styled_popup_menu(menu)
        current = self.global_settings.font_size_pt
        self._font_size_var = getattr(self, "_font_size_var", None) or tk.IntVar(
            value=current
        )
        self._font_size_var.set(current)
        for sz in (9, 10, 11, 12, 14):
            size_menu.add_radiobutton(
                label=f"{sz} pt",
                variable=self._font_size_var,
                value=sz,
                command=lambda s=sz: self._set_font_size(s),
            )
        size_menu.add_separator()
        size_menu.add_command(label="Custom...", command=self._on_font_size_custom)
        menu.add_cascade(label="UI font size", menu=size_menu)

        theme_menu = self._styled_popup_menu(menu)
        for label, value in (("OS theme", "os"), ("Dark", "dark"), ("Light", "light")):
            theme_menu.add_radiobutton(
                label=label,
                variable=self._theme_var,
                value=value,
                command=self._on_theme_change,
            )
        menu.add_cascade(label="Theme", menu=theme_menu)

        menu.add_command(
            label="Reset window layout",
            command=self._on_reset_window_layout,
        )

        # Auto-dismiss when the mouse leaves all menus.  The delay keeps
        # the menu visible while the cursor briefly crosses the border
        # to a cascade submenu.
        self._install_menu_hover_dismiss(menu, [size_menu, theme_menu])

        # Post the menu just under the button, right-aligned so the
        # menu's right edge lines up with the button's right edge.
        btn = self.ui_settings_btn
        menu.update_idletasks()
        x = btn.winfo_rootx() + btn.winfo_width() - menu.winfo_reqwidth()
        y = btn.winfo_rooty() + btn.winfo_height()
        try:
            menu.tk_popup(x, y)
        finally:
            menu.grab_release()

    def _set_font_size(self, size_pt: int) -> None:
        """Apply *size_pt* as the UI body/menu/heading font size and persist."""
        size_pt = max(6, min(int(size_pt), 32))
        self.global_settings.font_size_pt = size_pt
        # Keep the radio variable in sync in case this was called via
        # _on_font_size_custom (the variable only auto-updates for the
        # radio-button command path).
        if getattr(self, "_font_size_var", None) is not None:
            try:
                self._font_size_var.set(size_pt)
            except tk.TclError:
                pass

        try:
            save_global_settings(get_projects_dir(), self.global_settings)
        except Exception:
            logger.warning("Failed to save font size", exc_info=True)

        from flextool.gui.ui_metrics import setup_fonts, get_metrics
        _code_pt = self.global_settings.code_font_size_pt or (size_pt + 2)
        setup_fonts(self, body_pt=size_pt, code_pt=_code_pt)
        # Treeview rowheight depends on line-height — recompute & re-apply.
        _m = get_metrics(self)
        ttk.Style().configure("Treeview", rowheight=_m.row_height)
        self._char_width = _m.cw
        self._line_height = _m.lh
        # _bold_font is the *string* "TkHeadingFont" — the named font has
        # already been reconfigured by setup_fonts above, so every widget
        # using font="TkHeadingFont" picks up the new size live. Nothing
        # to refresh here.

    def _on_font_size_custom(self) -> None:
        """Prompt for a custom font size in points."""
        current = self.global_settings.font_size_pt
        new = simpledialog.askinteger(
            "UI font size",
            "Body font size (points, 6–32):",
            parent=self,
            initialvalue=current,
            minvalue=6,
            maxvalue=32,
        )
        if new is not None:
            self._set_font_size(new)

    def _open_project_dialog(self) -> None:
        """Open the ProjectDialog and handle its result.

        If the currently-open project was deleted via the dialog, reset
        the main window to a no-project state (clear combo, drop child
        windows, repopulate combo) before honouring any new selection.
        """
        # Import here to avoid circular imports at module level
        from flextool.gui.dialogs.project_dialog import ProjectDialog

        dlg = ProjectDialog(self, current_project=self.current_project)

        # Handle deletion of the currently-open project before any
        # potential switch — _switch_project below would otherwise
        # overwrite current_project and mask the deletion.
        if self.current_project and self.current_project in dlg.deleted_names:
            self._reset_to_no_project()

        if dlg.result:
            self._switch_project(dlg.result)
        else:
            # No project opened — refresh the combo so any deletions
            # are reflected even if the user cancelled afterwards.
            self._refresh_project_combo()

    def _reset_to_no_project(self) -> None:
        """Tear down per-project state after the open project is deleted."""
        # Close child windows tied to the deleted project.
        if self.execution_mgr is not None:
            try:
                self.execution_mgr.kill_all()
            except Exception:
                logger.warning(
                    "Failed to kill execution jobs after project deletion",
                    exc_info=True,
                )
        if self.execution_window is not None and self.execution_window.winfo_exists():
            self.execution_window.destroy()
        self.execution_window = None
        if self._result_viewer is not None and self._result_viewer.winfo_exists():
            self._result_viewer.destroy()
        self._result_viewer = None
        self.execution_mgr = None
        self.output_action_mgr = None

        self.current_project = None
        # Clear "recent project" so a future launch doesn't try to
        # re-open the deleted one.
        if self.global_settings.recent_project:
            self.global_settings.recent_project = ""
            try:
                save_global_settings(get_projects_dir(), self.global_settings)
            except Exception:
                logger.warning(
                    "Failed to clear recent project after deletion",
                    exc_info=True,
                )

        # Reset UI: combo, title, highlight the Project button.
        self._refresh_project_combo()
        self.project_combo.set("")
        self.title("FlexTool")
        try:
            self.project_menu_btn.configure(style="Accent.TButton")
        except tk.TclError:
            pass

    def _on_reset_window_layout(self) -> None:
        """Clear all saved window/sash positions and apply defaults.

        Resets:
          * ResultViewer window_geometry, left_pane_width, scenario_pane_height
            and the recorded layout_cw.
          * ExecutionWindow exec_jobs_sash and exec_jobs_layout_cw.

        Any currently open ResultViewer / ExecutionWindow is closed; the
        next time it opens it uses computed defaults.
        """
        confirm = messagebox.askyesno(
            "Reset window layout",
            "Discard saved window sizes and sash positions for the result "
            "viewer and execution window?\n\nOpen result viewer and "
            "execution window will be closed; their defaults will apply "
            "the next time you open them.",
            parent=self,
        )
        if not confirm:
            return

        # Project-scoped settings (viewer) — only if a project is loaded
        if self.current_project:
            vs = self.project_settings.viewer_settings
            vs.window_geometry = ""
            vs.left_pane_width = 0
            vs.scenario_pane_height = 0
            vs.layout_cw = 0
            try:
                projects_dir = get_projects_dir()
                project_path = projects_dir / self.current_project
                save_project_settings(project_path, self.project_settings)
            except Exception:
                logger.warning(
                    "Failed to save project settings after layout reset",
                    exc_info=True,
                )

        # Global settings (execution window sash)
        self.global_settings.exec_jobs_sash = 0
        self.global_settings.exec_jobs_layout_cw = 0
        try:
            save_global_settings(get_projects_dir(), self.global_settings)
        except Exception:
            logger.warning(
                "Failed to save global settings after layout reset",
                exc_info=True,
            )

        # Close any open child windows so defaults take effect next open
        if self._result_viewer is not None and self._result_viewer.winfo_exists():
            self._result_viewer.destroy()
        self._result_viewer = None

        if self.execution_window is not None and self.execution_window.winfo_exists():
            self.execution_window.destroy()
        self.execution_window = None

        messagebox.showinfo(
            "Reset window layout",
            "Saved window layout cleared. Defaults will apply next time the "
            "result viewer or execution window opens.",
            parent=self,
        )

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
        self.exec_scenario_mgr = ExecutedScenarioManager(
            project_path, self.project_settings
        )
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

        # One-shot cleanup: drop scenario references from settings.yaml
        # that no longer correspond to either an available input scenario
        # or an on-disk output folder. This handles drift from prior
        # sessions where sources were removed without a follow-up sweep.
        if self._prune_dangling_scenario_state():
            self._save_current_settings()
            # Re-render lists to drop any pruned entries from view.
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
                self.project_settings.auto_generate_comp_spinedb = self.auto_comp_spinedb_var.get()
                self.project_settings.auto_generate_comp_excel = self.auto_comp_excel_var.get()
                self.project_settings.debug_level = self.debug_var.get()
                self.project_settings.save_memory = self.save_memory_var.get()

                # Persist solver options.  Reuse the trace handler so
                # the validate + commit logic stays in one place.
                self._on_auto_gen_toggled()

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

        def is_external(iid):
            return iid.startswith(_EXT_IID_PREFIX)

        sources_by_name = {s.name: s for s in self.input_source_mgr._sources}
        for item in self.input_sources_tree.get_children():
            values = self.input_sources_tree.item(item, "values")
            if not values:
                continue
            source_name = _source_name_from_iid(item)
            old_status_char = values[3]
            if is_external(item):
                source = next(
                    (s for s in self.input_source_mgr._sources
                     if s.name == source_name and s.external_rel_path),
                    None,
                )
            else:
                source = sources_by_name.get(source_name)
            if source is not None:
                filepath = self.input_source_mgr.resolve_path(source)
            else:
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

    def _on_input_sources_toggled(self, _changed: list[str]) -> None:
        """CheckTreeController callback for input_sources_tree."""
        self._update_available_scenarios()
        self._save_checked_input_sources()

    def _on_input_source_dblclick(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """Double-click on an input source row opens it for editing."""
        item = self.input_sources_tree.identify_row(event.y)
        if item and not _is_ghost_iid(item):
            # Select the item so _on_edit_source sees it as selected
            self.input_sources_tree.selection_set(item)
            self._on_edit_source()

    def _on_available_toggled(self, _changed: list[str]) -> None:
        """CheckTreeController callback for available_tree."""
        self._update_add_to_execution_style()
        self._save_checked_available_scenarios()

    def _on_executed_toggled(self, _changed: list[str]) -> None:
        """CheckTreeController callback for executed_tree."""
        self._update_output_status()
        self._save_checked_executed_scenarios()

    def _on_executed_click(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """Handle the "view" column on click; check column is owned by the controller."""
        tree = self.executed_tree
        region = tree.identify_region(event.x, event.y)
        if region != "cell":
            return
        column = tree.identify_column(event.x)
        if column == "#4":  # "view" column
            item = tree.identify_row(event.y)
            if item:
                values = tree.item(item, "values")
                if values and values[3]:
                    scenario_name = values[2]
                    try:
                        src_num = int(values[1])
                    except (ValueError, IndexError):
                        src_num = None
                    self._view_scenario_plots(scenario_name, src_num)

    # ── Right-click context menus ──────────────────────────────────

    def _on_input_source_right_click(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """Show context menu for input sources tree."""
        tree = self.input_sources_tree
        item = tree.identify_row(event.y)
        if item:
            # Select the right-clicked row if not already selected
            if item not in tree.selection():
                tree.selection_set(item)

        if item and _is_ghost_iid(item):
            self._show_ghost_source_menu(item, event.x_root, event.y_root)
            return

        menu = self._styled_popup_menu(self)
        menu.add_command(label="Edit", command=self._on_edit_source)
        menu.add_command(label="Convert", command=self._on_convert_source)
        menu.add_separator()
        menu.add_command(label="Delete", command=self._on_delete_source)
        menu.add_separator()
        menu.add_command(label="Refresh", command=self._on_refresh_sources)
        menu.tk_popup(event.x_root, event.y_root)

    def _show_ghost_source_menu(self, item: str, x_root: int, y_root: int) -> None:
        """Context menu for a retired ghost row (deleted file, results remain).

        Offers re-linking the orphaned result folders to a live source
        (a cascade of current sources) or deleting the stale results
        outright. Either action lets the next refresh drop the ghost.
        """
        try:
            number = int(item[len(_GHOST_IID_PREFIX):])
        except ValueError:
            return

        menu = self._styled_popup_menu(self)
        live = [
            s for s in (self.input_source_mgr._sources if self.input_source_mgr else [])
        ]
        if live:
            relink = self._styled_popup_menu(menu)
            for src in live:
                relink.add_command(
                    label=f"{src.name}  (source {src.number})",
                    command=lambda n=src.number: self._relink_ghost_results(number, n),
                )
            menu.add_cascade(label="Re-link these results to…", menu=relink)
        else:
            menu.add_command(
                label="Re-link these results to… (no live source)",
                state="disabled",
            )
        menu.add_separator()
        menu.add_command(
            label="Delete these stale results",
            command=lambda: self._delete_ghost_results(number),
        )
        menu.add_separator()
        menu.add_command(label="Refresh", command=self._on_refresh_sources)
        menu.tk_popup(x_root, y_root)

    def _delete_ghost_results(self, number: int) -> None:
        """Delete the orphaned result folders behind a retired ghost row."""
        if not self.input_source_mgr:
            return
        folders = self.input_source_mgr.result_folders_for_number(number)
        if not folders:
            self._refresh_input_sources()
            return
        names = [f.name for f, _ in folders]
        answer = messagebox.askyesno(
            "Delete stale results",
            f"Permanently delete {len(folders)} executed-scenario result "
            f"folder(s) left over from retired source {number} "
            "(not retrievable)?\n\n  "
            + "\n  ".join(names[:20])
            + ("\n  …" if len(names) > 20 else ""),
            icon="warning",
            parent=self,
        )
        if not answer:
            return
        self.input_source_mgr.delete_results(number)
        self._refresh_input_sources()
        self._refresh_executed_scenarios()

    def _relink_ghost_results(self, old_number: int, new_number: int) -> None:
        """Re-attribute a retired source's result folders to a live source."""
        if not self.input_source_mgr:
            return
        folders = self.input_source_mgr.result_folders_for_number(old_number)
        if not folders:
            self._refresh_input_sources()
            return
        answer = messagebox.askyesno(
            "Re-link results",
            f"Re-link {len(folders)} result folder(s) from retired source "
            f"{old_number} to source {new_number}?\n\n"
            "The folders are renamed to record the new source number.",
            parent=self,
        )
        if not answer:
            return
        _moved, conflicts = self.input_source_mgr.relink_results(
            old_number, new_number
        )
        self._refresh_input_sources()
        self._refresh_executed_scenarios()
        if conflicts:
            messagebox.showwarning(
                "Some results not re-linked",
                "These result folders already existed under source "
                f"{new_number} (or could not be moved) and were left in "
                "place:\n\n  " + "\n  ".join(conflicts),
                parent=self,
            )

    def _on_input_source_motion(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """Show a tooltip when hovering over an error/empty row."""
        tree = self.input_sources_tree
        item = tree.identify_row(event.y)
        if not item:
            self._hide_input_status_tip()
            return
        values = tree.item(item, "values")
        if not values:
            self._hide_input_status_tip()
            return

        status_char = values[3]
        source_name = _source_name_from_iid(item)
        is_external = item.startswith(_EXT_IID_PREFIX)
        tip_lines: list[str] = []
        if _is_ghost_iid(item):
            ghost = next(
                (s for s in (self.input_source_mgr._last_ghosts
                             if self.input_source_mgr else [])
                 if f"{_GHOST_IID_PREFIX}{s.number}" == item),
                None,
            )
            count = ghost.result_count if ghost else 0
            named = ghost.name if ghost and not ghost.name.startswith("(source ") else None
            origin = f"input file '{named}'" if named else "a deleted input file"
            tip_lines.append(
                f"Retired source {values[2]}: {origin} is gone, but "
                f"{count} executed scenario result(s) still use it.\n"
                "Kept so the results stay attributable; it disappears "
                "automatically once those results are removed.\n"
                "Right-click to re-link the results to a live source or "
                "delete them."
            )
        elif status_char == STATUS_EMPTY:
            tip_lines.append("No scenarios found in this file.")
        elif status_char == STATUS_ERR:
            tip_lines.append("Could not read scenarios (invalid or missing file).")
        elif status_char == STATUS_EDITING:
            tip_lines.append("File is currently open for editing.")
        if is_external:
            rel = self.project_settings.external_refs.get(source_name, "")
            tip_lines.append(
                f"External reference: {rel}\n"
                "Delete removes the reference only; the file stays in place."
            )
        if not tip_lines:
            self._hide_input_status_tip()
            return
        tip_text = "\n\n".join(tip_lines)

        # Show or reposition tooltip
        if self._input_status_tip is not None:
            try:
                self._input_status_tip.wm_geometry(
                    f"+{event.x_root + 15}+{event.y_root + 10}"
                )
                self._input_status_tip.children["!label"].configure(text=tip_text)
                return
            except (tk.TclError, KeyError):
                self._input_status_tip = None

        tw = tk.Toplevel(self)
        tw.wm_overrideredirect(True)
        tw.wm_attributes("-topmost", True)
        tw.wm_geometry(f"+{event.x_root + 15}+{event.y_root + 10}")
        lbl = tk.Label(
            tw, text=tip_text, justify="left",
            background="#333333", foreground="#ffffff",
            relief="solid", borderwidth=1, padx=8, pady=4,
        )
        lbl.pack()
        self._input_status_tip = tw

    def _hide_input_status_tip(self) -> None:
        """Destroy the input source status tooltip if visible."""
        if self._input_status_tip is not None:
            try:
                self._input_status_tip.destroy()
            except tk.TclError:
                pass
            self._input_status_tip = None

    def _on_available_right_click(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """Show context menu for available scenarios tree."""
        tree = self.available_tree
        item = tree.identify_row(event.y)
        if item:
            if item not in tree.selection():
                tree.selection_set(item)
        menu = self._styled_popup_menu(self)
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

        menu = self._styled_popup_menu(self)
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
                try:
                    src_num = int(values[1])
                except (ValueError, IndexError):
                    src_num = None
                menu.add_command(
                    label="View results",
                    command=lambda s=scenario_name, n=src_num: self._view_scenario_plots(s, n),
                )
                menu.add_separator()
        menu.add_command(
            label="Delete irrevocably",
            command=self._on_delete_results,
        )
        menu.tk_popup(event.x_root, event.y_root)

    def _on_executed_space_from_menu(self) -> None:
        """Toggle checkboxes for selected items in executed_tree (from context menu)."""
        self._executed_check_ctrl.toggle_selected()

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
        dlg = AddDialog(
            self, project_path,
            execution_mgr=self.execution_mgr,
            input_source_mgr=self.input_source_mgr,
        )
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
            if _is_ghost_iid(item):
                continue  # never tick a retired ghost row
            values = self.input_sources_tree.item(item, "values")
            if values and _source_name_from_iid(item) not in old_sources:
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

    def _run_db_migrations_with_ui(self) -> tuple[bool, list[str]]:
        """Plan, gather consent for, and run pending DB migrations with UI.

        Returns a ``(failed, messages)`` tuple.  *failed* is ``True`` if any
        database could not be migrated and had to be rolled back, in which
        case the caller should not proceed to use those sources.  Step-level
        progress is streamed to the Execution window; a modal dialog keeps
        the interface locked until the run finishes.
        """
        from flextool.gui.execution_manager import JobType

        assert self.input_source_mgr is not None
        mgr = self.input_source_mgr

        (
            internal_to_migrate,
            external_to_migrate,
            planning_messages,
        ) = mgr.plan_db_migrations()

        if not internal_to_migrate and not external_to_migrate:
            return False, planning_messages

        consent: str = "in_place"
        if external_to_migrate:
            consent = ask_external_migration_consent(self, external_to_migrate)

        final_list: list[tuple[str, Path]] = list(internal_to_migrate)
        if consent == "in_place":
            for name, abs_path, _curr, _tgt in external_to_migrate:
                final_list.append((name, abs_path))
        elif consent == "copy_to_project":
            settings_changed = False
            for name, _abs_path, _curr, _tgt in external_to_migrate:
                dst, error = mgr.copy_external_to_project(name)
                if error is None:
                    final_list.append((name, dst))
                    settings_changed = True
                else:
                    planning_messages.append(error)
            if settings_changed:
                save_project_settings(
                    get_projects_dir() / self.current_project,
                    self.project_settings,
                )
        # consent == "cancel" -> skip externals entirely

        if not final_list:
            return False, planning_messages

        total = len(final_list)

        # Stream step-level progress to the Execution window (like conversion),
        # set up before the modal gate grabs input.
        self._ensure_execution_mgr()
        exec_mgr = self.execution_mgr
        job_id: int | None = None
        if exec_mgr is not None:
            job = exec_mgr.add_auxiliary_job(
                JobType.MIGRATION,
                "Database migration",
                "db_migration",
            )
            job_id = job.job_id
            exec_mgr.append_stdout(
                job_id, f"Migrating {total} database file(s) to the current version.\n"
            )
            self._open_or_raise_execution_window()
            if self.execution_window is not None:
                self.execution_window.select_job(job_id)

        dialog = MigrationProgressDialog(
            self,
            title="Migrating databases",
            initial_status=f"Preparing to migrate {total} file(s)…",
        )

        def _emit(line: str) -> None:
            if exec_mgr is not None and job_id is not None:
                exec_mgr.append_stdout(job_id, line)

        worker_messages: list[str] = []
        outcome = {"failed": False}

        def _worker() -> None:
            try:
                for i, (name, path) in enumerate(final_list, start=1):
                    if dialog.cancel_requested:
                        break
                    dialog.update_status(f"Migrating {name} ({i}/{total})…")
                    _emit(f"[{i}/{total}] {name}: checking database version…")

                    def _progress_cb(
                        curr: int,
                        target: int,
                        nxt: int,
                        _name: str = name,
                        _i: int = i,
                    ) -> None:
                        dialog.update_status(
                            f"Migrating {_name} ({_i}/{total})… "
                            f"step v{curr} → v{nxt} (target v{target})"
                        )
                        _emit(
                            f"[{_i}/{total}] {_name}: applying step "
                            f"v{curr} → v{nxt} (target v{target})"
                        )

                    was_upgraded, failed, messages = check_and_upgrade_database(
                        path,
                        progress_callback=_progress_cb,
                        cancel_check=lambda: dialog.cancel_requested,
                    )
                    worker_messages.extend(messages)
                    for msg in messages:
                        _emit(msg)
                    if not messages and not was_upgraded:
                        _emit(f"[{i}/{total}] {name}: already up to date.")
                    if failed:
                        outcome["failed"] = True
            finally:
                if exec_mgr is not None and job_id is not None:
                    if outcome["failed"]:
                        exec_mgr.append_stdout(
                            job_id, "\nMigration finished with errors."
                        )
                    else:
                        exec_mgr.append_stdout(job_id, "\nMigration finished.")
                    exec_mgr.finish_job(job_id, not outcome["failed"])
                dialog.mark_finished()

        threading.Thread(target=_worker, daemon=True).start()
        self.wait_window(dialog)

        return outcome["failed"], planning_messages + worker_messages

    def _refresh_input_sources(self) -> None:
        """Re-scan input sources and repopulate the treeview."""
        if not self.input_source_mgr:
            return

        migration_failed, upgrade_messages = self._run_db_migrations_with_ui()
        if migration_failed:
            messagebox.showerror(
                "Database migration failed",
                "\n".join(upgrade_messages)
                + "\n\nThe full log is shown in the Execution window.",
                parent=self,
            )
        elif upgrade_messages:
            messagebox.showinfo(
                "Database upgrades",
                "\n".join(upgrade_messages),
                parent=self,
            )

        sources = self.input_source_mgr.refresh()

        # Clear input sources tree
        for item in self.input_sources_tree.get_children():
            self.input_sources_tree.delete(item)

        # Configure tags for problem rows
        self.input_sources_tree.tag_configure("error", background="#6b2020")
        self.input_sources_tree.tag_configure("empty", background="#5c4a00")
        # Retired "ghost" rows: input file gone, only results remain. Dimmed
        # foreground signals they are informational and not runnable.
        self.input_sources_tree.tag_configure("retired", foreground="#7a7a7a")
        # Brief flash used when an open is refused because the file is
        # already open elsewhere — overlays the row's normal tags.
        self.input_sources_tree.tag_configure("flash_open", background="#aa3333")

        # Populate input sources tree
        saved_checked = set(self.project_settings.checked_input_sources)
        has_saved_state = len(saved_checked) > 0
        for source in sources:
            if source.retired:
                # Ghost row: no checkbox, dimmed, distinct status glyph.
                self.input_sources_tree.insert(
                    "",
                    "end",
                    iid=f"{_GHOST_IID_PREFIX}{source.number}",
                    values=("", source.name, source.number, STATUS_RETIRED),
                    tags=("retired",),
                )
                continue

            if source.status == "ok":
                status_char = STATUS_OK
            elif source.status == "editing":
                status_char = STATUS_EDITING
            elif source.status == "empty":
                status_char = STATUS_EMPTY
            else:
                status_char = STATUS_ERR

            # Restore checkbox: if we have saved state, only check those in saved list;
            # otherwise default to CHECK_ON (first load / no saved state)
            if has_saved_state:
                check_char = CHECK_ON if source.name in saved_checked else CHECK_OFF
            else:
                check_char = CHECK_ON

            if source.status == "error":
                tags = ("error",)
            elif source.status == "empty":
                tags = ("empty",)
            elif source.external_rel_path:
                tags = ("external",)
            else:
                tags = ()
            if source.external_rel_path:
                display_name = f"{source.name}  \u2192  {source.external_rel_path}"
                iid = f"ext:{source.name}"
            else:
                display_name = source.name
                iid = source.name
            self.input_sources_tree.insert(
                "",
                "end",
                iid=iid,
                values=(check_char, display_name, source.number, status_char),
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
        """Update the Add/Create/Update button text + state + style.

        Three states, driven by checked rows in ``available_tree`` and
        whether on-disk results already exist for those scenarios under
        ``output_parquet/<resolved subdir>/``:

        - 0 checked scenarios → button is disabled, base label
          ("Add checked scenarios to the execution list [F9]"), plain
          style.
        - ≥1 checked, none with results on disk → enabled, label
          starts with "Create" (signals "new results"), Accent style.
        - ≥1 checked, at least one with results on disk → enabled,
          label starts with "Update" (signals "regen"), Accent style.

        Results existence is checked per scenario via
        :func:`resolve_subdir_for_read` against ``output_parquet/``.
        """
        from flextool.gui.scenario_key import resolve_subdir_for_read

        # Collect checked rows: each row is (source_number, scenario_name).
        checked_pairs: list[tuple[int, str]] = []
        for item in self.available_tree.get_children():
            values = self.available_tree.item(item, "values")
            if not values or values[0] != CHECK_ON:
                continue
            # available_tree columns: ("check", "source_num", "scenario_name")
            try:
                src_num = int(values[1])
            except (ValueError, IndexError):
                continue
            checked_pairs.append((src_num, values[2]))

        if not checked_pairs:
            self.add_to_execution_btn.configure(
                state="disabled",
                style="TButton",
                text="Add checked scenarios to\nthe execution list [F9]",
            )
            return

        # Determine whether any checked scenario already has results on
        # disk under output_parquet/<resolved subdir>/.
        any_results = False
        if self.current_project:
            parquet_root = (
                get_projects_dir() / self.current_project / "output_parquet"
            )
            bare_owners = self.project_settings.bare_output_owners
            for src_num, scen_name in checked_pairs:
                subdir = resolve_subdir_for_read(bare_owners, src_num, scen_name)
                scen_dir = parquet_root / subdir
                if scen_dir.is_dir():
                    try:
                        if any(scen_dir.iterdir()):
                            any_results = True
                            break
                    except OSError:
                        # Permission / I/O glitch — treat as "no results"
                        # rather than crashing the trace.
                        continue

        verb = "Update" if any_results else "Create"
        self.add_to_execution_btn.configure(
            state="normal",
            style="Accent.TButton",
            text=f"{verb} checked scenarios on\nthe execution list [F9]",
        )

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
        """No-op: the legacy tint of the Output actions LabelFrame is
        gone since the File outputs box uses ttk.LabelFrame and inherits
        the theme. Per-row ✓/✗ status icons make the "ready to act"
        affordance visible without an extra colour cue.
        """
        return

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
                selected.append(_source_name_from_iid(item))
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
                checked.append((_source_name_from_iid(item), values[3]))
        return checked

    def _get_selected_sources(self) -> list[tuple[str, str]]:
        """Return (name, status_char) for each highlighted (selected) input source row."""
        selected: list[tuple[str, str]] = []
        for item in self.input_sources_tree.selection():
            if _is_ghost_iid(item):
                continue  # ghost rows are informational, not actionable
            values = self.input_sources_tree.item(item, "values")
            if values:
                selected.append((_source_name_from_iid(item), values[3]))
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

        # ── Convert: exactly one selected, xlsx or sqlite, status OK, not external ──
        if len(selected) == 1:
            name, status = selected[0]
            is_convertible = name.lower().endswith((".xlsx", ".sqlite"))
            is_external = (
                self.input_source_mgr is not None
                and name in self.input_source_mgr.settings.external_refs
            )
            if is_convertible and status == STATUS_OK and not is_external:
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
        source = next(
            (s for s in self.input_source_mgr._sources if s.name == source_name),
            None,
        )
        if source is not None:
            filepath = self.input_source_mgr.resolve_path(source)
        else:
            project_path = get_projects_dir() / self.current_project
            filepath = project_path / "input_sources" / source_name

        if not filepath.exists():
            messagebox.showerror("File not found", f"Cannot find:\n{filepath}")
            return

        ext = filepath.suffix.lower()
        if ext in (".xlsx", ".ods"):
            if self.input_source_mgr._check_lock(filepath):
                self._flash_input_source_open(source_name)
                return
            try:
                open_file_in_default_app(filepath)
                self.input_source_mgr.mark_as_editing(source_name)
            except OSError as exc:
                messagebox.showerror("Error", f"Could not open file:\n{exc}")
                return
        elif ext == ".sqlite":
            if self.db_editor_mgr.is_editor_running(source_name):
                self._flash_input_source_open(source_name)
                return
            db_url = f"sqlite:///{filepath}"
            proc = self.db_editor_mgr.open_database(db_url, source_name)
            if proc is None:
                # spine-db-editor not found: point at 'Update FlexTool' and
                # offer the system default .sqlite app as a fallback.
                # askyesnocancel: Yes=Update, No=default app, Cancel=nothing.
                choice = messagebox.askyesnocancel(
                    "Spine DB Editor not available",
                    "The 'spine-db-editor' command was not found, so .sqlite "
                    "input sources cannot be opened in the Spine DB Editor.\n\n"
                    "It is part of Spine Toolbox. You can install it via "
                    "'Update FlexTool' (tick 'Install Spine Toolbox'), or "
                    'manually from the flextool directory:\n\n'
                    '  pip install -e ".[toolbox]"\n\n'
                    "Open 'Update FlexTool' now?\n\n"
                    "Choose 'No' to open the file with your system's default "
                    "application for .sqlite files instead.",
                    parent=self,
                )
                if choice is True:
                    self._on_update_flextool(preselect_toolbox=True)
                elif choice is False:
                    try:
                        open_file_in_default_app(filepath)
                    except OSError as exc:
                        messagebox.showerror(
                            "Error", f"Could not open file:\n{exc}"
                        )
                return
            # spine-db-editor was found and launched, but an incomplete Spine
            # Toolbox install can still crash it on startup. Check shortly after
            # launch and explain the failure instead of leaving the user with a
            # silent no-op (the crash otherwise only shows in the terminal).
            self.after(
                2000, self._check_db_editor_launch, proc, source_name
            )
        else:
            messagebox.showinfo("Unsupported", f"Cannot edit files of type '{ext}'.")
            return

        # Refresh to show editing status
        self._refresh_input_sources()

    def _on_update_flextool(self, preselect_toolbox: bool = False) -> None:
        """Open the Update dialog and, if confirmed, run the update as a job."""
        from flextool.gui.dialogs.update_dialog import UpdateDialog
        from flextool.gui.execution_manager import JobType
        from flextool.update_flextool import install_info

        default_toolbox = preselect_toolbox or install_info.toolbox_installed()
        dlg = UpdateDialog(
            self,
            install_description=install_info.describe_install(),
            is_git=install_info.is_git_install(),
            default_toolbox=default_toolbox,
            check_on_startup=self.global_settings.check_updates_on_startup,
            update_available=self._update_available,
            check_fn=install_info.update_available,
            post_to_main=self.post_to_main,
        )
        self.wait_window(dlg)

        # The dialog may have refreshed availability via its Check button.
        self._apply_update_indicator(dlg.update_available)

        # Persist the "check on startup" preference whether or not the user
        # proceeded with the update.
        if dlg.check_on_startup != self.global_settings.check_updates_on_startup:
            self.global_settings.check_updates_on_startup = dlg.check_on_startup
            save_global_settings(get_projects_dir(), self.global_settings)

        if not dlg.proceed:
            return

        steps, cwd = install_info.upgrade_steps(dlg.include_toolbox)
        self._run_cli_job(
            steps,
            job_type=JobType.UPDATE,
            description="Update FlexTool",
            action_key="update_flextool",
            cwd=cwd,
            intro="Updating FlexTool"
            + (" with Spine Toolbox" if dlg.include_toolbox else "")
            + "…\n",
            on_finish=self._update_finished,
        )

    def _update_finished(self, success: bool, output: str = "") -> None:
        """Report update result; prompt for restart only if code actually changed."""
        if not success:
            messagebox.showerror(
                "Update failed",
                "The update did not complete. See the Execution window log for "
                "the full output (use 'Copy log text' to share it).",
                parent=self,
            )
            return

        # Decide whether anything actually changed, so we don't tell the user to
        # restart after a no-op update.
        from flextool.update_flextool import install_info

        low = output.lower()
        if install_info.is_git_install():
            changed = "already up to date" not in low  # git pulled new commits
        else:
            changed = "successfully installed" in low  # pip upgraded the wheel

        if not changed:
            self._apply_update_indicator(False)
            messagebox.showinfo(
                "Already up to date",
                "FlexTool is already at the latest version — nothing to update.",
                parent=self,
            )
            return

        # Something was updated: clear the highlight and ask for a restart.
        self._apply_update_indicator(False)
        messagebox.showinfo(
            "Update complete — restart required",
            "FlexTool was updated successfully.\n\n"
            "Please close and restart FlexTool for the new version to take "
            "effect. The running application is still using the old code.",
            parent=self,
        )

    def _check_db_editor_launch(self, proc, source_name: str) -> None:
        """Surface a Spine DB Editor that crashed right after launch.

        Scheduled shortly after :meth:`_on_edit_source` launches the editor.
        If the process is still alive it opened fine and nothing happens; if
        it has already exited with an error, explain why and how to fix it.
        """
        if proc.poll() is None:
            return  # still running — launched successfully
        if proc.returncode == 0:
            return  # exited cleanly (e.g. user closed it already)

        details = self.db_editor_mgr.read_launch_output(proc).strip()

        # Put the full traceback in the Execution window, where there is room
        # to read it and a Copy button to share it.
        self._ensure_execution_mgr()
        if self.execution_mgr is not None:
            from flextool.gui.execution_manager import JobType

            job = self.execution_mgr.add_auxiliary_job(
                JobType.DB_EDITOR,
                f"Spine DB Editor: {source_name}",
                f"db_editor:{source_name}",
            )
            self.execution_mgr.append_stdout(
                job.job_id,
                f"The Spine DB Editor exited immediately (code "
                f"{proc.returncode}) when opening '{source_name}'.\n",
            )
            for line in (details or "(no output captured)").splitlines():
                self.execution_mgr.append_stdout(job.job_id, line)
            self.execution_mgr.finish_job(job.job_id, False)
            self._open_or_raise_execution_window()
            if self.execution_window is not None:
                self.execution_window.select_job(job.job_id)

        if messagebox.askyesno(
            "Could not open database",
            "The Spine DB Editor failed to start, so the database could not "
            "be opened. This usually means Spine Toolbox is missing or only "
            "partially installed.\n\n"
            "The full error is shown in the Execution window. You can "
            "(re)install Spine Toolbox via 'Update FlexTool' (tick 'Install "
            "Spine Toolbox'), or manually:\n\n"
            '  pip install -e ".[toolbox]"\n\n'
            "Open 'Update FlexTool' now?",
            parent=self,
        ):
            self._on_update_flextool(preselect_toolbox=True)

    def _flash_input_source_open(self, source_name: str) -> None:
        """Briefly flash a source row red to signal "already open elsewhere"."""
        candidates = (source_name, f"ext:{source_name}")
        iid = next(
            (c for c in candidates if self.input_sources_tree.exists(c)),
            None,
        )
        if iid is None:
            return
        original = tuple(self.input_sources_tree.item(iid, "tags") or ())
        self.input_sources_tree.item(iid, tags=("flash_open",) + original)
        try:
            self.bell()
        except tk.TclError:
            pass

        def _revert() -> None:
            if self.input_sources_tree.exists(iid):
                self.input_sources_tree.item(iid, tags=original)

        self.after(700, _revert)

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

        Path.cwd()  # subprocess cwd — user workspace, formerly the repo root
        needs_migration = False

        if info.format == ExcelFormat.SELF_DESCRIBING and (
            info.version is None or info.version >= CURRENT_FLEXTOOL_DB_VERSION
        ):
            # Current version — import directly against the current schema
            template = package_data_path("schemas/spinedb_schema.json")
        elif info.format == ExcelFormat.OLD_V2:
            # The old FlexTool 2.x importer targets a FROZEN schema version
            # (v56): it writes v56-era parameters that do not exist in the v25
            # base, so initialising from v25 would silently drop them (and
            # mis-derive every node as 'commodity').  Init from the frozen
            # import template; cmd_read_old_flextool migrates the result to the
            # current schema after writing.
            from flextool.process_inputs.write_old_flextool_to_db import (
                OLD_FLEXTOOL_IMPORT_SCHEMA,
            )
            template = package_data_path(OLD_FLEXTOOL_IMPORT_SCHEMA)
        else:
            # Older Excel (SPECIFICATION or older SELF_DESCRIBING):
            # init from v25 base, migrate to the Excel's version, import, then
            # migrate the rest of the way to current after import.
            template = package_data_path("schemas/pre_v26/flextool_template_v25.json")
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
        flextool_root = Path.cwd()  # subprocess cwd — user workspace, formerly the repo root

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
            template = package_data_path("schemas/pre_v26/flextool_template_v25.json")
            import_cmd = [
                sys.executable, "-m",
                "flextool.cli.cmd_read_self_describing_tabular_input",
                str(xlsx_path), tmp_db_url, "--keep-entities",
            ]
        else:
            # SPECIFICATION format
            template = package_data_path("schemas/pre_v26/flextool_template_v25.json")
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
            self.post_to_main(self._refresh_input_sources)

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

    def post_to_main(self, fn, *args) -> None:
        """Run ``fn(*args)`` on the Tk main thread.

        Thread-safe and does not touch Tk, so it is the safe way for worker
        threads to schedule GUI work (unlike ``self.after`` / any Tk call,
        which must only be used from the main thread).
        """
        self._main_thread_queue.put((fn, args))

    def _pump_main_thread_queue(self) -> None:
        """Drain queued worker→main callbacks on the main thread, then reschedule."""
        try:
            while True:
                fn, args = self._main_thread_queue.get_nowait()
                try:
                    fn(*args)
                except Exception:
                    logger.exception("main-thread task failed")
        except queue.Empty:
            pass
        self.after(50, self._pump_main_thread_queue)

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
            self.post_to_main(self._conversion_finished,
                              success, source_name, source_path, target_path)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

    def _run_cli_job(
        self,
        steps: list[list[str]],
        *,
        job_type,
        description: str,
        action_key: str,
        cwd: Path | None = None,
        intro: str | None = None,
        on_finish=None,
    ) -> None:
        """Run one or more subprocess steps as a single auxiliary Execution job.

        Steps run in order, with combined stdout/stderr streamed to the job
        log; the first non-zero exit aborts the rest. When the job ends,
        *on_finish* (if given) is invoked on the Tk main thread with the
        overall success flag. Generic counterpart to
        :meth:`_run_conversion_subprocess` for commands with no file-move
        bookkeeping (e.g. self-update, toolbox install).
        """
        self._ensure_execution_mgr()
        if self.execution_mgr is None:
            return

        job = self.execution_mgr.add_auxiliary_job(job_type, description, action_key)
        mgr = self.execution_mgr
        if intro:
            mgr.append_stdout(job.job_id, intro)

        self._open_or_raise_execution_window()
        if self.execution_window is not None:
            self.execution_window.select_job(job.job_id)

        cwd_str = str(cwd) if cwd is not None else None

        def _worker() -> None:
            success = True
            collected: list[str] = []
            try:
                env = {**os.environ, "PYTHONUNBUFFERED": "1"}
                for step in steps:
                    mgr.append_stdout(job.job_id, "$ " + " ".join(step))
                    proc = subprocess.Popen(
                        step, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1, cwd=cwd_str, env=env,
                    )
                    with mgr._lock:
                        job.process = proc
                    for line in proc.stdout:  # type: ignore[union-attr]
                        text = line.rstrip("\n")
                        collected.append(text)
                        mgr.append_stdout(job.job_id, text)
                    proc.wait()
                    if proc.returncode != 0:
                        success = False
                        mgr.append_stdout(
                            job.job_id,
                            f"\nStep failed (exit code {proc.returncode}); stopping.",
                        )
                        break
            except Exception as exc:
                logger.error("CLI job '%s' failed: %s", action_key, exc, exc_info=True)
                mgr.append_stdout(job.job_id, f"\nError: {exc}")
                success = False

            mgr.finish_job(job.job_id, success)
            if on_finish is not None:
                self.post_to_main(on_finish, success, "\n".join(collected))

        threading.Thread(target=_worker, daemon=True).start()

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
                xlsx_path = self.execution_mgr._resolve_source_path(s.source_name)
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
        flextool_root = Path.cwd()  # subprocess cwd — user workspace, formerly the repo root
        migrate_db_path: str | None = None

        if info.format == ExcelFormat.SELF_DESCRIBING and (
            info.version is None or info.version >= CURRENT_FLEXTOOL_DB_VERSION
        ):
            # Current version — import directly against the current schema
            template = package_data_path("schemas/spinedb_schema.json")
            cmd = [
                sys.executable, "-m",
                "flextool.cli.cmd_read_self_describing_tabular_input",
                str(xlsx_path), target_db_url, "--keep-entities",
            ]
        elif info.format == ExcelFormat.SELF_DESCRIBING:
            # Older self-describing: init from v25 base, pre-migrate to
            # the Excel's version, import, then migrate to current.
            template = package_data_path("schemas/pre_v26/flextool_template_v25.json")
            cmd = [
                sys.executable, "-m",
                "flextool.cli.cmd_read_self_describing_tabular_input",
                str(xlsx_path), target_db_url, "--keep-entities",
            ]
            migrate_db_path = str(db_path)
        else:
            template = package_data_path("schemas/pre_v26/flextool_template_v25.json")
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
            self.post_to_main(self._xlsx_one_source_finished, source_name, success)

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
        external_names = [
            n for n in names if n in self.project_settings.external_refs
        ]
        local_names = [n for n in names if n not in self.project_settings.external_refs]

        # Split confirmation messages: external refs are just unlinked from
        # the project, local files are deleted from disk.
        lines: list[str] = []
        if local_names:
            lines.append(
                "The following files will be DELETED from "
                f"'projects/{self.current_project}/input_sources' "
                "(not retrievable):\n  " + "\n  ".join(local_names)
            )
        if external_names:
            lines.append(
                "The following external references will be REMOVED "
                "(the original files stay where they are):\n  "
                + "\n  ".join(external_names)
            )
        answer = messagebox.askyesno(
            "Delete input source",
            "Are you really sure?\n\n" + "\n\n".join(lines),
            icon="warning",
        )
        if not answer:
            return

        input_dir = project_path / "input_sources"
        for source_name in local_names:
            filepath = input_dir / source_name
            try:
                if filepath.exists():
                    filepath.unlink()
            except OSError as exc:
                messagebox.showerror(
                    "Delete failed",
                    f"Could not delete '{source_name}':\n{exc}",
                )
            if source_name in self.project_settings.input_source_numbers:
                del self.project_settings.input_source_numbers[source_name]

        for source_name in external_names:
            self.input_source_mgr.remove_external_ref(source_name)
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

        # Source removal can leave scenarios in settings.yaml whose only
        # backing was the removed source's scenario list. Sweep them now
        # so they don't haunt the UI on the next session.
        if self._prune_dangling_scenario_state():
            self._save_current_settings()
            self._refresh_executed_scenarios()

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
        """Toggle checkboxes for all selected (highlighted) items in available_tree.

        Applies the shared selection-based rule via ``CheckTreeController``:
        all checked -> uncheck all; else -> check all.
        """
        self._available_check_ctrl.toggle_selected()

    def _on_check_executed(self) -> None:
        """Check/uncheck *all* executed scenarios via the shared rule.

        Selects every row first so the controller's selection-aware rule
        applies to the whole list (matching the button label "all").
        """
        children = self.executed_tree.get_children()
        if not children:
            return
        self.executed_tree.selection_set(children)
        self._executed_check_ctrl.toggle_selected()

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

    def _on_key_v(self, event: tk.Event) -> str | None:  # type: ignore[type-arg]
        """Focus the available scenarios tree on 'v' press."""
        w = event.widget
        while w is not None:
            if isinstance(w, (tk.Entry, ttk.Entry, tk.Text)):
                return None
            w = getattr(w, "master", None)
        self._focus_tree(self.available_tree)
        return "break"

    def _on_key_x(self, event: tk.Event) -> str | None:  # type: ignore[type-arg]
        """Focus the executed scenarios tree on 'x' press."""
        w = event.widget
        while w is not None:
            if isinstance(w, (tk.Entry, ttk.Entry, tk.Text)):
                return None
            w = getattr(w, "master", None)
        self._focus_tree(self.executed_tree)
        return "break"

    def _on_tree_focus_in(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """When a treeview gets focus, clear selection in the other trees.

        Only clears when another tree had an active selection, so that
        clicking buttons or other non-tree widgets doesn't destroy state.
        """
        focused = event.widget
        if focused not in (self.input_sources_tree, self.available_tree, self.executed_tree):
            return
        for tree in (self.input_sources_tree, self.available_tree, self.executed_tree):
            if tree is not focused and tree.selection():
                tree.selection_remove(*tree.selection())

    def _on_main_focus_in(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """Re-draw tree selections when the root window regains focus.

        ttk draws a Treeview's selection with dim colors while the root
        window is deactivated (Alt-Tab away, minimize). When the window
        re-activates, the rows don't automatically repaint until a widget
        is touched — opening the result viewer happens to do so, which is
        why the user reported blue coming back only when opening it.
        Nudging every tree's selection here forces an immediate repaint
        in the focused-style blue.
        """
        # Only respond to the root window's FocusIn, not bubbled events
        # from child widgets (which would cause redundant repaints).
        if event.widget is not self:
            return
        for tree in (self.input_sources_tree, self.available_tree, self.executed_tree):
            sel = tree.selection()
            if sel:
                # selection_set with the existing selection is a no-op
                # semantically but forces ttk to invalidate the row
                # display, which is the repaint we want.
                tree.selection_set(sel)

    def _on_main_configure(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        """Refresh font sizes if Tk's effective DPI changed.

        Called on every <Configure> (resize, move). Only acts when the
        measured DPI differs from the last-checked value by more than 10%,
        which roughly corresponds to crossing into a monitor with a
        different scaling factor.

        Note: this rescales fonts only. Already-placed widgets retain
        their pixel positions until the user resizes or re-opens them.
        """
        from flextool.gui.ui_metrics import monitor_dpi
        new_dpi = monitor_dpi(self)
        if self._last_dpi <= 0:
            self._last_dpi = new_dpi
            return
        ratio = new_dpi / self._last_dpi
        if 0.90 <= ratio <= 1.10:
            return
        self._last_dpi = new_dpi
        # Re-apply font sizes so points render at the new DPI.
        from flextool.gui.ui_metrics import setup_fonts, get_metrics
        from tkinter import ttk
        body_pt = self.global_settings.font_size_pt or 10
        code_pt = self.global_settings.code_font_size_pt or (body_pt + 2)
        setup_fonts(self, body_pt=body_pt, code_pt=code_pt)
        _m = get_metrics(self)
        ttk.Style().configure("Treeview", rowheight=_m.row_height)
        self._char_width = _m.cw
        self._line_height = _m.lh
        # _bold_font is the named-font string "TkHeadingFont" — already
        # reconfigured by setup_fonts above.

    def _focus_tree(self, tree: ttk.Treeview) -> None:
        """Give keyboard focus to a treeview, ensuring arrow keys work."""
        tree.focus_set()
        children = tree.get_children()
        if not children:
            return
        # Use existing selection, or default to first item
        sel = tree.selection()
        item = sel[0] if sel else children[0]
        tree.selection_set(item)
        tree.focus(item)  # set internal focus item for arrow navigation
        tree.see(item)

    def _on_shift_arrow_up(self, event: tk.Event) -> str:  # type: ignore[type-arg]
        """Extend selection upward with Shift+Up."""
        tree = event.widget
        if not isinstance(tree, ttk.Treeview):
            return "break"
        focused = tree.focus()
        if not focused:
            return "break"
        prev_item = tree.prev(focused)
        if not prev_item:
            return "break"
        tree.selection_add(prev_item)
        tree.focus(prev_item)
        tree.see(prev_item)
        return "break"

    def _on_shift_arrow_down(self, event: tk.Event) -> str:  # type: ignore[type-arg]
        """Extend selection downward with Shift+Down."""
        tree = event.widget
        if not isinstance(tree, ttk.Treeview):
            return "break"
        focused = tree.focus()
        if not focused:
            return "break"
        next_item = tree.next(focused)
        if not next_item:
            return "break"
        tree.selection_add(next_item)
        tree.focus(next_item)
        tree.see(next_item)
        return "break"

    # ── Executed scenarios management ────────────────────────────

    def _refresh_executed_scenarios(self) -> None:
        """Scan for executed scenario results and repopulate the executed_tree.

        Preserves existing checkbox states so that auto-checked scenarios
        (from ``_refresh_and_autocheck_scenario``) are not unchecked.
        """
        # Remember which (source_number, scenario_name) pairs are currently checked
        previously_checked: set[tuple[int, str]] = set()
        for item in self.executed_tree.get_children():
            values = self.executed_tree.item(item, "values")
            if values and values[0] == CHECK_ON:
                try:
                    previously_checked.add((int(values[1]), values[2]))
                except (ValueError, IndexError):
                    pass

        # Clear executed scenarios tree
        for item in self.executed_tree.get_children():
            self.executed_tree.delete(item)

        if not self.exec_scenario_mgr:
            return

        executed = self.exec_scenario_mgr.scan_executed()

        # On first load (no items were in tree), restore from saved settings
        if not previously_checked:
            from flextool.gui.scenario_key import parse_key
            for k in self.project_settings.checked_executed_scenarios:
                previously_checked.add(parse_key(k))

        current_source_numbers = set(
            self.project_settings.input_source_numbers.values()
        )
        for info in executed:
            key = (info.source_number, info.name)
            check_char = CHECK_ON if key in previously_checked else CHECK_OFF
            view_text = "\u25b6"
            tags = ("orphan",) if info.source_number not in current_source_numbers else ()
            self.executed_tree.insert(
                "",
                "end",
                values=(check_char, info.source_number, info.name, view_text, info.timestamp),
                tags=tags,
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
        sources_by_name = (
            {s.name: s for s in self.input_source_mgr._sources}
            if self.input_source_mgr else {}
        )
        for item in self.input_sources_tree.get_children():
            values = self.input_sources_tree.item(item, "values")
            if not values:
                continue
            source_name = _source_name_from_iid(item)
            # Existing check: treeview status column shows editing
            if values[3] == STATUS_EDITING:
                try:
                    editing_source_numbers.add(int(values[2]))
                except (ValueError, IndexError):
                    pass
                continue
            # Enhanced check: sqlite sources with a running editor process
            if source_name.lower().endswith(".sqlite") and self.input_source_mgr:
                source = sources_by_name.get(source_name)
                if source is not None:
                    filepath = self.input_source_mgr.resolve_path(source)
                else:
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

        Phase B contract: pressing the "Update view scenarios" / "Results
        viewer" button is the single source of truth for the viewer
        scenarios set.  We collect whatever is currently checked in the
        executed-scenarios tree, translate it to on-disk subdir names, and
        hand the resulting list to the viewer.  The viewer compares it
        against the scenarios recorded in
        ``output_parquet_comparison/_metadata.json`` and rebuilds the
        combined parquets only when the two sets differ — toggling
        scenarios *inside* the viewer never triggers a rebuild.
        """
        # Sync the persisted checked state with the live tree first so the
        # subdir derivation below sees the latest user edits.
        self._collect_checked_executed_scenarios()
        desired = self._main_window_checked_executed_subdirs()

        if (
            self._result_viewer is not None
            and self._result_viewer.winfo_exists()
        ):
            self._result_viewer._scenario_db_map = self._get_scenario_db_map()
            self._result_viewer.refresh_to_viewer_scenarios(desired)
            self._result_viewer.deiconify()
            self._result_viewer.lift()
            self._result_viewer.focus_force()
            self._update_view_results_btn()
            return

        project_path = get_projects_dir() / self.current_project
        self._result_viewer = ResultViewer(
            master=self,
            project_path=project_path,
            settings=self.project_settings,
            scenario_db_map=self._get_scenario_db_map(),
            desired_viewer_scenarios=desired,
        )
        self._update_view_results_btn()
        # When the viewer closes, revert button text
        self._result_viewer.bind("<Destroy>", lambda e: self._update_view_results_btn())

    def _main_window_checked_executed_subdirs(self) -> list[str]:
        """Return on-disk subdir names for currently-checked executed rows.

        Reads the executed scenarios tree, picks the rows whose check
        glyph is on, resolves each ``(source_number, scenario_name)`` pair
        through :func:`resolve_subdir_for_read`, and returns the list in
        tree order.  Used to derive the *desired viewer scenarios* set
        passed to :class:`ResultViewer` on cold-open or refresh.
        """
        from flextool.gui.scenario_key import resolve_subdir_for_read
        bare_owners = self.project_settings.bare_output_owners
        result: list[str] = []
        for item in self.executed_tree.get_children():
            values = self.executed_tree.item(item, "values")
            if not values or values[0] != CHECK_ON:
                continue
            try:
                src_num = int(values[1])
            except (ValueError, IndexError):
                continue
            scen_name = values[2]
            subdir = resolve_subdir_for_read(bare_owners, src_num, scen_name)
            result.append(subdir)
        return result

    def _update_view_results_btn(self) -> None:
        """Update the Results viewer button text and style based on viewer and scenario state."""
        viewer_open = (
            self._result_viewer is not None
            and self._result_viewer.winfo_exists()
        )
        if viewer_open:
            self.view_results_btn.configure(text="Update view scenarios")
        else:
            self.view_results_btn.configure(text="Results viewer…")

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
        """Build a mapping of scenario subdirs to database paths.

        Keys are the on-disk subdir form ``<source_number>_<scenario_name>``
        so the map does not alias same-named scenarios from different input
        sources. For each input source:

        - .sqlite files are used directly (possibly outside the project, for
          external references)
        - .xlsx files use the converted .sqlite from intermediate/
        """
        from flextool.gui.scenario_key import resolve_subdir_for_read
        db_map: dict[str, Path] = {}
        if self.input_source_mgr is None:
            return db_map

        project_path = get_projects_dir() / self.current_project
        bare_owners = self.project_settings.bare_output_owners
        for source in self.input_source_mgr._sources:
            if source.file_type == "sqlite":
                db_path = self.input_source_mgr.resolve_path(source)
            else:
                stem = Path(source.name).stem
                db_path = project_path / "intermediate" / f"{stem}.sqlite"

            if db_path.is_file():
                for scenario in source.scenarios:
                    subdir = resolve_subdir_for_read(
                        bare_owners, source.number, scenario
                    )
                    db_map[subdir] = db_path

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
            global_settings=self.global_settings,
        )
        # ExecutionManager seeds max_workers from ProjectSettings (with
        # the legacy GlobalSettings.max_workers as fallback) in its
        # constructor, so no extra apply step is needed here.

    def _open_or_raise_execution_window(self) -> None:
        """Open a new ExecutionWindow or raise an existing one."""
        if (
            self.execution_window is not None
            and self.execution_window.winfo_exists()
        ):
            self.execution_window.deiconify()
            self.execution_window.lift()
            self.execution_window.focus_force()
            self._update_execution_menu_style()
            return

        if self.execution_mgr is None:
            return

        self.execution_window = ExecutionWindow(
            self, self.execution_mgr, global_settings=self.global_settings,
        )
        self._update_execution_menu_style()
        # When the window closes, revert button accent
        self.execution_window.bind(
            "<Destroy>", lambda e: self._update_execution_menu_style(),
        )

    def _on_job_status_change(self, job: ExecutionJob) -> None:
        """Callback from ExecutionManager when a job's status changes.

        Invoked from worker threads, so the GUI work is marshalled onto the
        main thread (tkinter — including ``after`` — is not thread-safe).
        """
        self.post_to_main(self._apply_job_status_change, job)

    def _apply_job_status_change(self, job: ExecutionJob) -> None:
        """Apply a job status change; runs on the main thread."""
        if job.status == JobStatus.SUCCESS:
            self._refresh_and_autocheck_scenario(
                job.scenario_name, job.finish_timestamp,
            )
        elif job.status in (JobStatus.FAILED, JobStatus.KILLED):
            self._refresh_executed_scenarios()
            # A scenario that died from a native crash may be an incompatible
            # solver wheel — re-probe and offer the fix. force, because the
            # cached fingerprint may say "passed" if the environment changed.
            # from_crash so that if the solver stack probes clean, we tell the
            # user the crash was elsewhere (env / network drive), not silently.
            if getattr(job, "native_fault", False):
                self.global_settings.polars_check_fingerprint = ""
                self._check_polars_async(force=True, from_crash=True)

        # Update execution menu button highlight (Change 3)
        self._update_execution_menu_style()

        # Notify the execution window (if open)
        if (
            self.execution_window is not None
            and self.execution_window.winfo_exists()
        ):
            self.execution_window.schedule_refresh()

    def _on_all_jobs_finished(self) -> None:
        """Callback when all execution jobs have completed.

        Called from the scheduler thread -- marshal GUI updates to the main
        thread.
        """
        self.post_to_main(self._refresh_executed_scenarios)
        self.post_to_main(self._update_output_status)
        self.post_to_main(self._update_execution_menu_style)

    # ── Delete results handler ───────────────────────────────────

    @safe_callback
    def _on_delete_results(self) -> None:
        """Delete output files for selected (highlighted) scenarios in executed_tree."""
        if not self.exec_scenario_mgr or not self.current_project:
            return

        # Gather (source_number, scenario_name) pairs from the executed tree
        selected_ids: list[tuple[int, str]] = []
        for item in self.executed_tree.selection():
            values = self.executed_tree.item(item, "values")
            if not values:
                continue
            try:
                selected_ids.append((int(values[1]), values[2]))
            except (ValueError, IndexError):
                continue

        if not selected_ids:
            return

        names_str = "\n  ".join(name for _, name in selected_ids)
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

        self.exec_scenario_mgr.delete_results(selected_ids)
        # Ownership of bare names may have been released → persist.
        self._save_current_settings()
        # Sweep dangling scenario references from settings.yaml. The
        # delete may have removed the last on-disk backing for scenarios
        # whose source has also already been removed.
        if self._prune_dangling_scenario_state():
            self._save_current_settings()
        self._refresh_executed_scenarios()

    # ── Output status indicator updates ──────────────────────────

    def _update_output_status(self) -> None:
        """Update the output status labels based on checked executed scenarios."""
        if not self.exec_scenario_mgr:
            self._reset_output_status()
            self._update_output_frame_style()
            return

        # Gather checked (source_number, scenario_name) pairs from the executed tree
        checked_ids = self._get_checked_executed_ids()

        if not checked_ids:
            self._reset_output_status()
            self._update_output_frame_style()
            self._update_view_results_btn()
            return

        # Check per-scenario outputs (keyed by compound key)
        from flextool.gui.scenario_key import format_key
        outputs = self.exec_scenario_mgr.check_outputs(checked_ids)
        checked_keys = [format_key(sn, name) for sn, name in checked_ids]

        # Aggregate: all checked scenarios have the output?
        all_have_plots = all(outputs[k]["has_plots"] for k in checked_keys)
        all_have_excel = all(outputs[k]["has_excel"] for k in checked_keys)
        all_have_csvs = all(outputs[k]["has_csvs"] for k in checked_keys)

        # Check comparison outputs (comparison outputs are project-wide, not per-scenario)
        comp = self.exec_scenario_mgr.check_comparison_outputs([n for _, n in checked_ids])

        # For comparison outputs, verify that the currently checked compound
        # keys match the ones used to generate the last comparison.
        checked_set = set(checked_keys)
        comp_plots_match = comp["has_comp_plots"] and (
            checked_set == set(self.project_settings.comp_plots_scenarios)
        )
        comp_excel_match = comp["has_comp_excel"] and (
            checked_set == set(self.project_settings.comp_excel_scenarios)
        )

        status_map = {
            "scen_plots": all_have_plots,
            "scen_excel": all_have_excel,
            "scen_csvs":  all_have_csvs,
            "comp_plots": comp_plots_match,
            # The SpineDB is a project-wide accumulating database (one
            # alternative per executed scenario) with no per-set regen
            # event to match against, so its status is simple existence.
            "comp_spinedb": comp["has_comp_spinedb"],
            "comp_excel": comp_excel_match,
        }

        # Status text on the button doubles as the indicator: \u2713 when the
        # output exists, \u2298 on a recorded failure, blank otherwise. The
        # button itself is clickable to (re-)generate.
        for key, has_output in status_map.items():
            if has_output:
                self._output_spinners[key].configure(text=_GEN_EXISTS)
                if key in self.output_action_btns:
                    self.output_action_btns[key].configure(style="Output.TButton")
            elif key in self._output_action_failed:
                self._output_spinners[key].configure(text=_GEN_FAILED)
                if key in self.output_action_btns:
                    self.output_action_btns[key].configure(style="Output.Grey.TButton")
            else:
                self._output_spinners[key].configure(text=_GEN_PENDING)
                if key in self.output_action_btns:
                    self.output_action_btns[key].configure(style="Output.Grey.TButton")

        self._update_output_frame_style()
        self._update_view_results_btn()

    def _reset_output_status(self) -> None:
        """Reset all output status indicators to the default blank state."""
        for key in self._output_display_names:
            self._output_spinners[key].configure(text=_GEN_PENDING)
            if key in self.output_action_btns:
                self.output_action_btns[key].configure(style="Output.Grey.TButton")

    # ── Auto-generate checkbox management ─────────────────────────

    def _load_auto_gen_vars(self) -> None:
        """Set auto-generate BooleanVars from the loaded project settings."""
        s = self.project_settings
        self._suppress_auto_gen_save = True
        try:
            self.auto_scen_plots_var.set(s.auto_generate_scen_plots)
            self.auto_scen_excels_var.set(s.auto_generate_scen_excels)
            self.auto_scen_csvs_var.set(s.auto_generate_scen_csvs)
            self.auto_comp_plots_var.set(s.auto_generate_comp_plots)
            self.auto_comp_spinedb_var.set(s.auto_generate_comp_spinedb)
            self.auto_comp_excel_var.set(s.auto_generate_comp_excel)
            self.debug_var.set(s.debug_level)
            self.save_memory_var.set(s.save_memory)
            # Solver options.
            self.solver_log_level_var.set(s.solver_log_level)
            self.solver_time_limit_var.set(s.solver_time_limit)
            self.matrix_file_format_var.set(s.matrix_file_format)
            self.scaling_var.set(s.scaling)
            self.presolve_var.set(s.presolve)
        finally:
            self._suppress_auto_gen_save = False

    def _on_auto_gen_toggled(self, *_args: object) -> None:
        """Save auto-generate settings when any checkbox is toggled."""
        if getattr(self, "_suppress_auto_gen_save", False):
            return
        self.project_settings.auto_generate_scen_plots = self.auto_scen_plots_var.get()
        self.project_settings.auto_generate_scen_excels = self.auto_scen_excels_var.get()
        self.project_settings.auto_generate_scen_csvs = self.auto_scen_csvs_var.get()
        self.project_settings.auto_generate_comp_plots = self.auto_comp_plots_var.get()
        self.project_settings.auto_generate_comp_spinedb = self.auto_comp_spinedb_var.get()
        self.project_settings.auto_generate_comp_excel = self.auto_comp_excel_var.get()
        self.project_settings.debug_level = self.debug_var.get()
        self.project_settings.save_memory = self.save_memory_var.get()

        # Solver options.  Spinbox / IntVar may surface ValueError when
        # the user is mid-edit (empty text field); skip the write in
        # that transient state — the trace fires again on the next
        # keystroke.  These vars back the modal Solver options dialog
        # launched from the side menu; the trace also fires when the
        # dialog's OK handler copies dialog-local values back into
        # these vars.
        _sll = self.solver_log_level_var.get()
        if _sll in ("silent", "normal", "verbose"):
            self.project_settings.solver_log_level = _sll
        try:
            self.project_settings.solver_time_limit = max(0, int(self.solver_time_limit_var.get()))
        except (TypeError, ValueError, tk.TclError):
            pass
        _mff = self.matrix_file_format_var.get()
        if _mff in ("mps", "lp"):
            self.project_settings.matrix_file_format = _mff
        _scl = self.scaling_var.get()
        if _scl in ("off", "solver_only", "basic", "full"):
            self.project_settings.scaling = _scl
        _ps = self.presolve_var.get()
        if _ps in ("on", "off", "choose"):
            self.project_settings.presolve = _ps

        if self.current_project:
            project_path = get_projects_dir() / self.current_project
            save_project_settings(project_path, self.project_settings)

    def _open_solver_options_dialog(self) -> None:
        """Open the modal Solver options dialog.

        Builds a ``tk.Toplevel`` parented to the main window with five
        controls (Log level, Time limit, Matrix file format, Scaling,
        Presolve).  Each control is bound to a *dialog-local* Tk var
        seeded from the corresponding main-window var so Cancel can
        discard pending edits without touching the persisted settings.
        OK copies the dialog-local values back into the main-window
        vars, whose write-trace (``_on_auto_gen_toggled``) saves the
        project settings.
        """
        from flextool.gui.hover_tooltip import attach_tooltip as _attach_tip

        dlg = tk.Toplevel(self.root)
        dlg.title("Solver options")
        dlg.transient(self.root)
        dlg.resizable(False, False)

        # Seed dialog-local vars from the main-window vars so Cancel
        # discards cleanly.
        d_sll = tk.StringVar(value=self.solver_log_level_var.get())
        d_stl = tk.IntVar(value=self.solver_time_limit_var.get())
        d_mff = tk.StringVar(value=self.matrix_file_format_var.get())
        d_scl = tk.StringVar(value=self.scaling_var.get())
        d_ps = tk.StringVar(value=self.presolve_var.get())

        body = ttk.Frame(dlg, padding=12)
        body.grid(row=0, column=0, sticky="nsew")
        body.columnconfigure(1, weight=1)

        row = 0
        # Log level
        ttk.Label(body, text="Log level:").grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=4,
        )
        log_row = ttk.Frame(body)
        log_row.grid(row=row, column=1, sticky="w", pady=4)
        for _t, _v in (("Silent", "silent"), ("Normal", "normal"), ("Verbose", "verbose")):
            ttk.Radiobutton(
                log_row, text=_t, variable=d_sll, value=_v,
            ).pack(side="left", padx=(0, 6))
        _attach_tip(log_row, (
            "HiGHS log verbosity (--solver-log-level).\n"
            "  • silent   — output_flag=false (suppress HiGHS console).\n"
            "  • normal   — output_flag=true (default).\n"
            "  • verbose  — output_flag=true + log_dev_level=2 for\n"
            "              per-iteration solver telemetry."
        ))
        row += 1

        # Time limit
        ttk.Label(body, text="Time limit (s):").grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=4,
        )
        tl_spin = ttk.Spinbox(
            body, from_=0, to=10**9, width=12, textvariable=d_stl,
        )
        tl_spin.grid(row=row, column=1, sticky="w", pady=4)
        _attach_tip(tl_spin, (
            "HiGHS wall-clock time limit in whole seconds\n"
            "(--solver-time-limit SECONDS). 0 means no limit\n"
            "(the CLI's unset default). Routed through the\n"
            "effective-options resolver as a CLI override\n"
            "(highest precedence)."
        ))
        row += 1

        # Matrix file format
        ttk.Label(body, text="Matrix file format:").grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=4,
        )
        mff_row = ttk.Frame(body)
        mff_row.grid(row=row, column=1, sticky="w", pady=4)
        for _t, _v in (("MPS", "mps"), ("LP", "lp")):
            ttk.Radiobutton(
                mff_row, text=_t, variable=d_mff, value=_v,
            ).pack(side="left", padx=(0, 6))
        _attach_tip(mff_row, (
            "On-disk format when the solver is dispatched via a matrix\n"
            "file (--matrix-file-format). The in-process vs. file\n"
            "decision is implicit:\n"
            "  • HiGHS + no Save memory  → direct (in-process; this\n"
            "                                flag is ignored).\n"
            "  • HiGHS + Save memory     → file write (polar-high\n"
            "                                round-trips through MPS\n"
            "                                internally; this flag is\n"
            "                                ignored on that path too).\n"
            "  • Commercial solver       → file write using the chosen\n"
            "                                format here."
        ))
        row += 1

        # Scaling
        ttk.Label(body, text="Scaling:").grid(
            row=row, column=0, sticky="nw", padx=(0, 8), pady=4,
        )
        scl_row = ttk.Frame(body)
        scl_row.grid(row=row, column=1, sticky="w", pady=4)
        _scl_opts = (
            ("Off", "off"), ("Solver only", "solver_only"),
            ("Basic", "basic"), ("Full", "full"),
        )
        for _i, (_t, _v) in enumerate(_scl_opts):
            ttk.Radiobutton(
                scl_row, text=_t, variable=d_scl, value=_v,
            ).grid(row=_i // 2, column=_i % 2, sticky="w", padx=(0, 6))
        _attach_tip(scl_row, (
            "FlexTool autoscaler strategy (--scaling).\n"
            "  • off          Disable ALL scaling (incl. HiGHS internal\n"
            "                  matrix equilibration). Raw numerics.\n"
            "  • solver_only  Disable FlexTool autoscaler; HiGHS still\n"
            "                  scales the matrix internally.\n"
            "  • basic        Compute LP ranges (Layer 1) and\n"
            "                  recommend user_*_scale (Layer 3). No\n"
            "                  LP-array mutation; MPS exports unscaled.\n"
            "  • full         Full autoscaler: range detection,\n"
            "                  semantic per-type LP scaling (Layer 2),\n"
            "                  and HiGHS user_*_scale recommendation\n"
            "                  (Layer 3). Default."
        ))
        row += 1

        # Presolve
        ttk.Label(body, text="Presolve:").grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=4,
        )
        ps_row = ttk.Frame(body)
        ps_row.grid(row=row, column=1, sticky="w", pady=4)
        for _t, _v in (("On", "on"), ("Off", "off"), ("Choose", "choose")):
            ttk.Radiobutton(
                ps_row, text=_t, variable=d_ps, value=_v,
            ).pack(side="left", padx=(0, 6))
        _attach_tip(ps_row, (
            "HiGHS presolve override (--presolve).\n"
            "  • on / off — explicit override.\n"
            "  • choose   — leave the CLI flag unset; the engine keeps\n"
            "                its determinism-pinned 'on' default.\n"
            "Off disables presolve entirely (much slower but useful\n"
            "for memory or numerical diagnostics)."
        ))
        row += 1

        # OK / Cancel buttons
        btn_row = ttk.Frame(body)
        btn_row.grid(row=row, column=0, columnspan=2, sticky="e", pady=(12, 0))

        def _on_cancel() -> None:
            dlg.destroy()

        def _on_ok() -> None:
            # Validate each before commit; bad values stay on the
            # current main-window value silently (the dialog already
            # gates radio buttons to the allowed set).
            _v_sll = d_sll.get()
            if _v_sll in ("silent", "normal", "verbose"):
                self.solver_log_level_var.set(_v_sll)
            try:
                self.solver_time_limit_var.set(max(0, int(d_stl.get())))
            except (TypeError, ValueError, tk.TclError):
                pass
            _v_mff = d_mff.get()
            if _v_mff in ("mps", "lp"):
                self.matrix_file_format_var.set(_v_mff)
            _v_scl = d_scl.get()
            if _v_scl in ("off", "solver_only", "basic", "full"):
                self.scaling_var.set(_v_scl)
            _v_ps = d_ps.get()
            if _v_ps in ("on", "off", "choose"):
                self.presolve_var.set(_v_ps)
            dlg.destroy()

        ttk.Button(btn_row, text="Cancel", command=_on_cancel).pack(
            side="right", padx=(6, 0),
        )
        ttk.Button(btn_row, text="OK", command=_on_ok).pack(
            side="right",
        )

        dlg.bind("<Escape>", lambda _e: _on_cancel())
        dlg.protocol("WM_DELETE_WINDOW", _on_cancel)

        # Show, then make modal.  ``grab_set`` after geometry settles
        # avoids a Linux/X11 race where the grab fires before the
        # window is mapped.
        dlg.update_idletasks()
        # Centre on the main window.
        try:
            px = self.root.winfo_rootx()
            py = self.root.winfo_rooty()
            pw = self.root.winfo_width()
            ph = self.root.winfo_height()
            dw = dlg.winfo_reqwidth()
            dh = dlg.winfo_reqheight()
            dlg.geometry(f"+{px + (pw - dw) // 2}+{py + (ph - dh) // 2}")
        except tk.TclError:
            pass
        dlg.grab_set()
        dlg.focus_set()

    # ── Output generation button handlers ─────────────────────────

    def _save_current_settings(self) -> None:
        """Persist the current project settings to disk."""
        if self.current_project:
            project_path = get_projects_dir() / self.current_project
            save_project_settings(project_path, self.project_settings)

    def _prune_dangling_scenario_state(self) -> bool:
        """Drop scenario references from settings that no longer have a backing.

        A scenario stays in settings.yaml as long as it is EITHER an
        available scenario in some loaded input source OR has on-disk
        results under ``output_parquet/``. This method computes those two
        sets and delegates to :func:`prune_dangling_scenario_state`.

        Returns True if anything was pruned (the caller should then
        persist settings).
        """
        if not self.current_project:
            return False

        # available_keys: (source_number, scenario_name) for every
        # scenario surfaced by any loaded input source.
        available_keys: set[tuple[int, str]] = set()
        if self.input_source_mgr is not None:
            for source in self.input_source_mgr._sources:
                if source.status != "ok":
                    continue
                for scen_name in source.scenarios:
                    available_keys.add((source.number, scen_name))

        # executed_subdirs: direct children of output_parquet/, skipping
        # underscore-prefixed manifest entries.
        executed_subdirs: set[str] = set()
        project_path = get_projects_dir() / self.current_project
        parquet_root = project_path / "output_parquet"
        if parquet_root.is_dir():
            for entry in parquet_root.iterdir():
                if entry.is_dir() and not entry.name.startswith("_"):
                    executed_subdirs.add(entry.name)

        return prune_dangling_scenario_state(
            self.project_settings, available_keys, executed_subdirs
        )

    def _get_checked_executed_names(self) -> list[str]:
        """Return scenario names that are checked in the executed_tree."""
        checked: list[str] = []
        for item in self.executed_tree.get_children():
            values = self.executed_tree.item(item, "values")
            if values and values[0] == CHECK_ON:
                checked.append(values[2])  # scenario_name column
        return checked

    def _get_checked_executed_ids(self) -> list[tuple[int, str]]:
        """Return (source_number, scenario_name) pairs for each checked executed row."""
        checked: list[tuple[int, str]] = []
        for item in self.executed_tree.get_children():
            values = self.executed_tree.item(item, "values")
            if values and values[0] == CHECK_ON:
                try:
                    checked.append((int(values[1]), values[2]))
                except (ValueError, IndexError):
                    continue
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

    def _start_output_action(self, key: str) -> list[tuple[int, str]] | None:
        """Common setup for output action buttons.

        Returns the checked (source_number, scenario_name) pairs, or None
        if there are none or the action manager is not available.
        """
        ids = self._get_checked_executed_ids()
        if not ids:
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
        return ids

    @safe_callback
    def _on_gen_scen_plots(self) -> None:
        """Generate plots for checked executed scenarios."""
        ids = self._start_output_action("scen_plots")
        if ids and self.output_action_mgr:
            self.output_action_mgr.run_scenario_plots(ids)

    @safe_callback
    def _on_gen_scen_excel(self) -> None:
        """Generate Excel files for checked executed scenarios."""
        ids = self._start_output_action("scen_excel")
        if ids and self.output_action_mgr:
            self.output_action_mgr.run_scenario_excel(ids)

    @safe_callback
    def _on_gen_scen_csvs(self) -> None:
        """Generate CSV files for checked executed scenarios."""
        ids = self._start_output_action("scen_csvs")
        if ids and self.output_action_mgr:
            self.output_action_mgr.run_scenario_csvs(ids)

    @safe_callback
    def _on_gen_comp_plots(self) -> None:
        """Generate comparison plots for checked executed scenarios."""
        from flextool.gui.scenario_key import format_key
        ids = self._start_output_action("comp_plots")
        if ids and self.output_action_mgr:
            self.project_settings.comp_plots_scenarios = [
                format_key(sn, name) for sn, name in ids
            ]
            self._save_current_settings()
            self.output_action_mgr.run_comparison_plots(ids)

    @safe_callback
    def _on_gen_comp_excel(self) -> None:
        """Generate comparison Excel for checked executed scenarios."""
        from flextool.gui.scenario_key import format_key
        ids = self._start_output_action("comp_excel")
        if ids and self.output_action_mgr:
            self.project_settings.comp_excel_scenarios = [
                format_key(sn, name) for sn, name in ids
            ]
            self._save_current_settings()
            self.output_action_mgr.run_comparison_excel(ids)

    @safe_callback
    def _on_gen_comp_spinedb(self) -> None:
        """Explain that the SpineDB cannot be regenerated on demand.

        Unlike the other outputs, the SpineDB results database is built by
        the writer from the live solve namespaces (``s`` / ``par``); it is
        skipped on the parquet-replay path the other regen actions use.
        So there is no manual generate — the user enables Auto-gen and
        (re-)runs the scenarios.
        """
        messagebox.showinfo(
            "Comparison SpineDB",
            "The SpineDB results database is written during the solve and "
            "cannot be regenerated from the stored parquet files.\n\n"
            "Tick its Auto-gen box and (re-)run the scenarios to produce "
            "results.sqlite.",
        )

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
            label.configure(text=_GEN_PENDING)

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

        Called from a worker thread -- marshal GUI updates to the main thread.
        """
        self.post_to_main(self._handle_output_action_done, action_name, success)

    def _handle_output_action_done(self, action_name: str, success: bool) -> None:
        """Re-enable the action button and refresh output status (runs on main thread)."""
        self._stop_spinner(action_name)
        if not success:
            self._output_action_failed.add(action_name)
        if action_name in self._output_spinners:
            if success:
                self._output_spinners[action_name].configure(text=_GEN_EXISTS)
            else:
                self._output_spinners[action_name].configure(text=_GEN_FAILED)
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
        ids = self._get_checked_executed_ids()
        if not ids:
            return
        from flextool.gui.scenario_key import resolve_subdir_for_read
        project_path = get_projects_dir() / self.current_project
        if len(ids) == 1:
            subdir = resolve_subdir_for_read(
                self.project_settings.bare_output_owners, *ids[0]
            )
            csv_dir = project_path / "output_csv" / subdir
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

    @safe_callback
    def _on_show_comp_spinedb(self) -> None:
        """Open the project-wide results.sqlite SpineDB.

        Prefer the Spine DB Editor when ``spine-db-editor`` is available on
        PATH (i.e. Spine Toolbox is installed in the venv). If it is not,
        explain that Spine Toolbox can be added via 'Update FlexTool' and
        offer to open the file with the system's default .sqlite application
        instead.
        """
        if not self.current_project:
            return
        project_path = get_projects_dir() / self.current_project
        results_db = project_path / "results.sqlite"
        if not results_db.is_file():
            messagebox.showinfo(
                "Comparison SpineDB",
                "results.sqlite does not exist yet. Tick its Auto-gen box "
                "and (re-)run the scenarios to produce it.",
            )
            return

        source_name = "results.sqlite"
        if self.db_editor_mgr.is_editor_running(source_name):
            # Already open in a Spine DB Editor we launched — don't spawn a
            # duplicate window (there is no row button here to flash).
            messagebox.showinfo(
                "Comparison SpineDB",
                "results.sqlite is already open in the Spine DB Editor.",
            )
            return
        db_url = f"sqlite:///{results_db}"
        proc = self.db_editor_mgr.open_database(db_url, source_name)
        if proc is None:
            # spine-db-editor not found: inform the user, point at
            # 'Update FlexTool', and offer the system default .sqlite app
            # as a fallback. askyesnocancel: Yes=Update, No=default app,
            # Cancel=do nothing.
            choice = messagebox.askyesnocancel(
                "Spine DB Editor not available",
                "The 'spine-db-editor' command was not found, so the "
                "comparison results.sqlite cannot be opened in the Spine DB "
                "Editor.\n\n"
                "It is part of Spine Toolbox, which can be added via "
                "'Update FlexTool' (tick 'Install Spine Toolbox').\n\n"
                "Open 'Update FlexTool' now?\n\n"
                "Choose 'No' to open results.sqlite with your system's "
                "default application for .sqlite files instead.",
                parent=self,
            )
            if choice is True:
                self._on_update_flextool(preselect_toolbox=True)
            elif choice is False:
                try:
                    open_file_in_default_app(results_db)
                except OSError as exc:
                    logger.warning("Could not open results SpineDB: %s", exc)
            return
        # spine-db-editor was found and launched, but an incomplete Spine
        # Toolbox install can still crash it on startup. Check shortly after
        # launch and explain the failure instead of leaving a silent no-op.
        self.after(2000, self._check_db_editor_launch, proc, source_name)

    # ── View scenario plots (from executed_tree view column) ──────

    def _view_scenario_plots(self, scenario_name: str, source_number: int | None = None) -> None:
        """Open the ResultViewer in single mode focused on *scenario_name*.

        When *source_number* is given, matches on ``(source_number, scenario_name)``
        so same-named scenarios from different sources don't alias.

        If the scenario is already part of the viewer set (its row is
        checked in executed_tree), this is a pure single-view: the
        viewer opens / raises without triggering a comparison rebuild.
        If it isn't checked yet, we auto-check it and go through the
        rebuild path so the new scenario shows up everywhere.
        """
        if not self.current_project:
            return
        target_item: str | None = None
        for item in self.executed_tree.get_children():
            values = self.executed_tree.item(item, "values")
            if not values:
                continue
            if values[2] != scenario_name:
                continue
            if source_number is not None:
                try:
                    if int(values[1]) != source_number:
                        continue
                except (ValueError, IndexError):
                    continue
            target_item = item
            break

        was_already_checked = (
            target_item is not None
            and self.executed_tree.item(target_item, "values")[0] == CHECK_ON
        )
        if target_item is not None and not was_already_checked:
            self.executed_tree.set(target_item, "check", CHECK_ON)
            self._save_checked_executed_scenarios()

        if was_already_checked:
            self._open_result_viewer_no_rebuild()
        else:
            self._open_or_raise_result_viewer()

        if self._result_viewer is not None and self._result_viewer.winfo_exists():
            self._result_viewer.show_scenario_in_single_mode(scenario_name)

    def _open_result_viewer_no_rebuild(self) -> None:
        """Open or raise the ResultViewer without triggering a rebuild.

        Used by the View action: the user is asking for a single-mode
        view of one scenario, so we mustn't kick off the comparison
        combine just because metadata.json may be out of sync. The
        explicit "Update view scenarios" button is still the way to
        pull main-window check changes into the comparison set.
        """
        if (
            self._result_viewer is not None
            and self._result_viewer.winfo_exists()
        ):
            self._result_viewer.deiconify()
            self._result_viewer.lift()
            self._result_viewer.focus_force()
            self._update_view_results_btn()
            return

        project_path = get_projects_dir() / self.current_project
        self._result_viewer = ResultViewer(
            master=self,
            project_path=project_path,
            settings=self.project_settings,
            scenario_db_map=self._get_scenario_db_map(),
            desired_viewer_scenarios=None,
        )
        self._update_view_results_btn()
        self._result_viewer.bind("<Destroy>", lambda e: self._update_view_results_btn())

    # ── Checkbox state persistence helpers ────────────────────────

    def _collect_checked_input_sources(self) -> None:
        """Read checked source names from the tree into project settings."""
        checked: list[str] = []
        for item in self.input_sources_tree.get_children():
            values = self.input_sources_tree.item(item, "values")
            if values and values[0] == CHECK_ON:
                checked.append(_source_name_from_iid(item))
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
        """Read checked executed scenario keys from the tree into project settings."""
        from flextool.gui.scenario_key import format_key
        checked: list[str] = []
        for item in self.executed_tree.get_children():
            values = self.executed_tree.item(item, "values")
            if values and values[0] == CHECK_ON:
                try:
                    src_num = int(values[1])
                except (ValueError, IndexError):
                    continue
                checked.append(format_key(src_num, values[2]))
        self.project_settings.checked_executed_scenarios = checked

    def _save_checked_executed_scenarios(self) -> None:
        """Persist checked executed scenario names to settings."""
        self._collect_checked_executed_scenarios()
        if self.current_project:
            project_path = get_projects_dir() / self.current_project
            save_project_settings(project_path, self.project_settings)
