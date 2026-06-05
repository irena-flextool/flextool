"""ResultViewer — Toplevel window for browsing and displaying result plots."""

from __future__ import annotations

import logging
import threading
import time
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tkinter import ttk
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from flextool.lean_parquet import read_lean_parquet
from flextool.gui.check_tree import CheckTreeController
from flextool.gui.data_models import ProjectSettings
from flextool.gui.network_graph import build_network_figure
from flextool.gui.plot_canvas import PlotCanvas
from flextool.gui.plot_config_reader import (
    PlotConfigData, PlotEntry, PlotGroup, PlotVariant, parse_plot_config,
)
from flextool.gui.settings_io import save_project_settings
from flextool.plot_outputs.config import PlotConfig, PLOT_FIELD_NAMES, _is_single_config, flatten_new_format
from flextool.plot_outputs.orchestrator import prepare_plot_data
from flextool.plot_outputs.color_template import resolve_plot_settings_path
from flextool.scenario_comparison.data_models import DispatchMappings, TimeSeriesResults
from flextool.scenario_comparison.db_reader import (
    build_scenario_folders_from_dir, collect_parquet_files, combine_parquet_files,
)
from flextool.scenario_comparison.dispatch_data import prepare_dispatch_data
from flextool.scenario_comparison.dispatch_mappings import (
    load_dispatch_mappings,
    resolve_data_scenario_tag,
)
from flextool.scenario_comparison.dispatch_plots import _build_dispatch_figure

if TYPE_CHECKING:
    from flextool.gui.data_models import PlotSettings
    from flextool.plot_outputs.plan import PlotPlan

logger = logging.getLogger(__name__)


class ResultViewer(tk.Toplevel):
    """Non-modal window for browsing and displaying result plots.

    This window provides scenario selection, a plot tree parsed from
    the YAML config, a variant panel, and a placeholder for future
    matplotlib plot rendering.
    """

    def __init__(
        self,
        master: tk.Tk,
        project_path: Path,
        settings: ProjectSettings,
        scenario_db_map: dict[str, Path] | None = None,
        desired_viewer_scenarios: list[str] | None = None,
    ) -> None:
        super().__init__(master)
        self.title("Result Viewer")

        self._project_path = project_path
        self._settings = settings
        self._scenario_db_map = scenario_db_map or {}
        self._viewer_settings = settings.viewer_settings

        # Plot config data
        self._plot_groups: list[PlotGroup] = []
        # Map tree item iid -> PlotEntry for quick lookup
        self._tree_entry_map: dict[str, PlotEntry] = {}
        # All unique variant letters across all config entries (created once)
        self._all_variant_letters: list[str] = []
        # Tooltip toplevel
        self._tooltip: tk.Toplevel | None = None

        # Focus model:
        #   _focus_col: variant letter index the cursor is on (-1 = in tree, not canvas)
        #   _active_entry_iid: which entry is currently displayed (solid blue)
        #   _active_variant: which variant letter is displayed (solid blue)
        self._focus_col: int = -1  # -1 means focus is in the tree
        self._active_entry_iid: str = ""
        self._active_variant: str = self._viewer_settings.last_variant or ""

        # Mode variable: "single", "comparison", "network"
        self._mode = tk.StringVar(value=self._viewer_settings.last_mode or "single")

        # File navigation state
        self._file_index = 0
        self._file_count = 1

        # Per-variant slider state.
        # Starts are session-local (not persisted): {letter: start}.
        # Durations are persisted in settings.<plot>.variant_durations and
        # are looked up there directly — see ``_active_plot_settings``.
        self._variant_start_state: dict[str, int] = {}
        self._last_slider_variant: str | None = None

        # Template-supplied default durations (per variant letter).
        # Populated in ``_populate_plot_tree`` whenever the YAML config is
        # (re)loaded. Values are int or the sentinel string ``"all"``.
        self._template_default_durations: dict[str, int | str] = {}

        # Guard flag — when True, ``_on_time_range_changed``'s persistence
        # path is skipped. Set around programmatic ``_duration_var.set(...)``
        # calls inside ``_update_time_range`` so a data-driven clamp does
        # not overwrite the user's saved intent.
        self._suppress_duration_save: bool = False

        # Pending after() id for debounced duration save, or None if no
        # save is currently scheduled.
        self._duration_save_after_id: str | None = None

        # Caches for parquet pipeline.
        #
        # Phase E generalises ``_parquet_cache_key`` from a fixed
        # ``(scenario, result_key)`` tuple to a free-form tuple so
        # both single-scenario reads and the comparison-mode lazy
        # plan-parquet union can share the slot.  Concretely:
        #
        # * single mode key:    ``(scenario, result_key)``
        # * comparison-union:   ``("_comparison_union", result_key,
        #                         sub_config, viewer_scenarios_tuple)``
        #
        # ``_parquet_cache_df`` holds the unioned DataFrame so repeated
        # renders for the same plot don't re-read the per-scenario plan
        # parquets.  The key prefix disambiguates the two flavours so
        # cache lookups never alias.
        self._yaml_cache: dict[Path, dict] = {}
        self._break_times_cache: dict[str, set[str] | None] = {}
        self._parquet_cache_key: tuple = ("", "")
        self._parquet_cache_df: pd.DataFrame | None = None

        # Live plan cache: PlotPlan computed from full data, reused across
        # slider changes.  Cleared on entry/variant/scenario change.
        self._live_plan: 'PlotPlan | None' = None
        self._live_plan_key: tuple[str, str, str] = ("", "", "")

        # Cross-scenario axis-bounds manifest.  Loaded lazily on first
        # use and reloaded whenever the on-disk file's mtime advances
        # (so batch runs in the background pick up naturally on the next
        # render).  ``None`` means no manifest was found on disk;
        # callers fall back to the per-plan ranges in that case.
        self._axis_manifest: dict | None = None
        self._axis_manifest_mtime: float = 0.0

        # Guard against recursive replots from time range updates
        self._updating_time_range = False

        # Async figure building
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="plot")
        self._render_gen = 0  # incremented on each replot; stale results discarded
        # Pending "Rendering…" placeholder after() id. The placeholder is
        # shown only if a render takes longer than _PLACEHOLDER_DELAY_MS, so
        # quick switches flip straight between figures (better for spotting
        # plot-to-plot differences) instead of flashing text on every switch.
        self._placeholder_after_id: str | None = None
        self._figure_cache: dict[tuple, plt.Figure] = {}  # prefetched figures
        self._figure_cache_lock = threading.Lock()

        # Availability manifest for three-level variant display
        self._current_availability: set[tuple[str, str]] = set()

        # Comparison tree state — the on-disk set the last regen was
        # built for may differ from the viewer's current ticks until a
        # regen is triggered.
        self._comp_needs_regen: bool = False

        # Monotonic generation counter for comparison combine requests.
        # Each user-driven state change bumps this; background workers
        # capture the value at submit time, and the GUI completion
        # callback drops results whose captured gen is no longer the
        # latest. Prevents stale data from leaking onto the plot when
        # the user toggles checkboxes faster than combines complete.
        # Tk callbacks run single-threaded on the main thread, so plain
        # attribute access is safe here.
        self._comp_request_gen: int = 0

        # Viewer scenarios (NOT current scenarios) scheduled for the next
        # comparison combine, set by :meth:`refresh_to_viewer_scenarios`
        # so :meth:`_regenerate_comparison` knows which subdirs to fold
        # into the combined parquets.  ``None`` means "no scheduled
        # refresh — fall back to the legacy current-tree-based path".
        self._scheduled_viewer_scenarios: list[str] | None = None

        # Dispatch mode state
        self._dispatch_mappings: DispatchMappings | None = None
        self._dispatch_results: TimeSeriesResults | None = None
        self._dispatch_scenario: str = ""  # folder name dispatch data is loaded for
        self._dispatch_data_tag: str = ""  # in-data scenario tag (may differ from folder)
        self._dispatch_ylims: dict[str, tuple[float, float]] = {}  # accumulated per-nodeGroup
        self._dispatch_columns: dict[str, list[str]] = {}  # accumulated column order

        # ── Font metrics for DPI-aware sizing ────────────────────────
        from flextool.gui.ui_metrics import get_metrics
        _metrics = get_metrics(self)
        cw: int = _metrics.cw
        lh: int = _metrics.lh

        # ── Window sizing & positioning ──────────────────────────────
        self._line_height = lh
        self._char_width = cw
        self.minsize(cw * 80, lh * 30)

        master.update_idletasks()
        main_x = master.winfo_x()
        master.winfo_y()
        main_w = master.winfo_width()
        screen_w = master.winfo_screenwidth()
        screen_h = master.winfo_screenheight()

        taskbar_margin = lh * 4
        usable_h = screen_h - taskbar_margin

        if screen_w < 1920:
            self.geometry(f"{screen_w}x{usable_h}+0+0")
        else:
            viewer_x = main_x + main_w
            viewer_w = max(screen_w - viewer_x, cw * 80)
            self.geometry(f"{viewer_w}x{usable_h}+{viewer_x}+0")

        # Restore saved geometry, clamped to the current screen so a
        # value saved at a different resolution does not run off-screen.
        if self._viewer_settings.window_geometry:
            from flextool.gui.ui_metrics import clamp_geometry, rescale_geometry
            saved_geom = rescale_geometry(
                self._viewer_settings.window_geometry,
                self._viewer_settings.layout_cw,
                cw,
            )
            clamped = clamp_geometry(
                saved_geom,
                screen_w, screen_h,
                min_w=cw * 80, min_h=lh * 30,
            )
            if clamped is not None:
                try:
                    self.geometry(clamped)
                except tk.TclError:
                    pass

        # ── Build layout ─────────────────────────────────────────────
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._paned = ttk.PanedWindow(self, orient="horizontal")
        self._paned.grid(row=0, column=0, sticky="nsew")

        self._build_left_column()
        self._build_right_column()

        # ── Configure tree tags and selection highlight ──────────────
        self._plot_tree.tag_configure("disabled", foreground="grey")

        # Make the plot-tree selection stand out in both light and dark
        # themes. Scoped to PlotTree.Treeview so the main window's
        # "selected !focus" -> blue mapping on the global Treeview style
        # is not overwritten when this viewer opens.
        style = ttk.Style()
        style.map(
            "PlotTree.Treeview",
            background=[
                ("selected !focus", "#2074d5"),
                ("selected", "#2074d5"),
            ],
            foreground=[
                ("selected !focus", "#ffffff"),
                ("selected", "#ffffff"),
            ],
        )

        # Theme-aware colors for variant grid
        self._fg_color = style.lookup("TLabel", "foreground") or "black"
        self._bg_color = style.lookup("TLabel", "background") or "white"

        # ── Resolve config paths ─────────────────────────────────────
        self._single_config_path = self._resolve_config_path(
            self._settings.single_plot_settings.config_file,
            "templates/default_plots.yaml",
        )
        # Comparison mode now reads the same merged config as single mode;
        # entries with ``scenario_rule`` set carry the comparison-mode info.
        self._comparison_config_path = self._resolve_config_path(
            self._settings.comparison_plot_settings.config_file,
            "templates/default_plots.yaml",
        )

        # ── Initial population ───────────────────────────────────────
        self._populate_scenarios()
        self._on_mode_changed()

        # ── Tab focus cycling ────────────────────────────────────────
        self._scenario_listbox.bind("<Tab>", self._focus_plot_tree)
        self._plot_tree.bind("<Tab>", self._focus_variant_canvas)

        # ── Global key bindings ──────────────────────────────────────
        # Use bind_all so these work even when child widgets (Treeview,
        # Listbox) have focus — they consume Prior/Next for scrolling
        # before Toplevel bindings fire (especially on Windows).
        self.bind_all("<Prior>", self._on_prev_file_event)
        self.bind_all("<Next>", self._on_next_file_event)
        self.bind_all("<Left>", self._on_prev_file_event)
        self.bind_all("<Right>", self._on_next_file_event)

        # Panel focus shortcuts: s = Scenarios, p = Plots
        # Bind on specific widgets so Treeview type-ahead doesn't consume them
        for w in (self._plot_tree, self._variant_canvas, self._scenario_listbox):
            w.bind("<Key-s>", self._on_focus_scenarios_event)
            w.bind("<Key-p>", self._on_focus_plots_event)
        # Also bind at toplevel level for when buttons/controls have focus
        self.bind("<Key-s>", self._on_focus_scenarios_event)
        self.bind("<Key-p>", self._on_focus_plots_event)

        # ── Window close ─────────────────────────────────────────────
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── Restore saved sash positions after layout is ready ───────
        self.after(50, self._restore_sash_position)

        # ── Cold-open auto-refresh ───────────────────────────────────
        # When the main window passed an explicit "desired viewer
        # scenarios" set (Phase B contract), reconcile against
        # ``_metadata.json`` and rebuild the combined parquets only when
        # the two differ.  ``None`` skips the reconciliation and leaves
        # whatever's on disk untouched.
        if desired_viewer_scenarios is not None:
            self.after(0, lambda d=list(desired_viewer_scenarios):
                       self.refresh_to_viewer_scenarios(d))

    # ------------------------------------------------------------------
    # Layout builders
    # ------------------------------------------------------------------

    def _build_left_column(self) -> None:
        """Build the left column: scenario listbox + plot tree."""
        left = ttk.Frame(self._paned, padding=5)
        # Request a wider default left pane (40 chars)
        left.configure(width=self._char_width * 40)
        self._paned.add(left, weight=0)

        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)

        # Vertical PanedWindow so user can resize Scenarios vs Plots
        self._left_paned = ttk.PanedWindow(left, orient="vertical")
        self._left_paned.grid(row=0, column=0, sticky="nsew")

        # ── Scenario listbox ─────────────────────────────────────────
        from flextool.gui.hover_tooltip import attach_tooltip
        # Use the named-font string so size changes via _set_font_size
        # reach the label live.
        _lf_font = "TkDefaultFont"
        scen_label = ttk.Label(left, text=" Scenarios [S] ", font=_lf_font)
        scen_frame = ttk.LabelFrame(self._left_paned, labelwidget=scen_label, padding=5)
        attach_tooltip(scen_label, (
            "Scenarios available in the chosen results.\n"
            "\n"
            "  \u2022 S \u2014 focus this list\n"
            "  \u2022 \u2191\u2193 \u2014 navigate scenarios"
        ))
        self._left_paned.add(scen_frame, weight=0)
        scen_frame.columnconfigure(0, weight=1)
        scen_frame.rowconfigure(0, weight=1)

        self._scenario_listbox = tk.Listbox(
            scen_frame,
            selectmode="browse",
            height=8,
            exportselection=False,
        )
        self._scenario_listbox.grid(row=0, column=0, sticky="nsew")

        scen_scroll = ttk.Scrollbar(
            scen_frame, orient="vertical", command=self._scenario_listbox.yview
        )
        scen_scroll.grid(row=0, column=1, sticky="ns")
        self._scenario_listbox.configure(yscrollcommand=scen_scroll.set)

        self._scenario_listbox.bind("<<ListboxSelect>>", self._on_scenario_selected)

        # ── Comparison scenarios tree (hidden by default) ────────────
        # A ttk.Treeview with a check column + name column replaces the
        # former list of ttk.Checkbuttons so the user gets the familiar
        # multi-select pattern: click / Shift+Click / Ctrl+Click select
        # rows, Space smart-toggles check on the whole selection,
        # Ctrl-A selects every row.
        comp_outer = ttk.Frame(scen_frame)
        comp_outer.grid(row=0, column=0, columnspan=2, sticky="nsew")
        comp_outer.columnconfigure(0, weight=1)
        comp_outer.rowconfigure(0, weight=1)
        comp_outer.grid_remove()  # hidden initially
        self._comp_outer_frame = comp_outer

        self._comp_tree = ttk.Treeview(
            comp_outer,
            columns=("check", "name"),
            show="headings",
            selectmode="extended",
        )
        self._comp_tree.heading("check", text="")
        self._comp_tree.heading("name", text="Scenario")
        _cw = self._char_width
        self._comp_tree.column("check", width=int(_cw * 3.45), minwidth=int(_cw * 3.45), stretch=False)
        self._comp_tree.column("name", width=_cw * 20, minwidth=_cw * 12, stretch=True)
        self._comp_tree.grid(row=0, column=0, sticky="nsew")

        comp_scroll = ttk.Scrollbar(
            comp_outer, orient="vertical", command=self._comp_tree.yview,
        )
        comp_scroll.grid(row=0, column=1, sticky="ns")
        self._comp_tree.configure(yscrollcommand=comp_scroll.set)

        self._comp_tree.bind("<Control-a>", self._on_comp_tree_ctrl_a)
        self._comp_tree.bind("<Control-A>", self._on_comp_tree_ctrl_a)
        self._comp_tree.bind("<Alt-Up>", self._on_comp_tree_alt_up)
        self._comp_tree.bind("<Alt-Down>", self._on_comp_tree_alt_down)
        self._comp_check_ctrl = CheckTreeController(
            self._comp_tree,
            check_column="check",
            checked_glyph=self._COMP_CHECK_ON,
            unchecked_glyph=self._COMP_CHECK_OFF,
            on_toggle=self._on_comp_tree_toggled,
        )
        from flextool.gui.tree_reorder import DragReorderController
        self._comp_drag = DragReorderController(
            self._comp_tree,
            check_column="check",
            on_reorder=self._on_comp_tree_reordered,
        )
        attach_tooltip(
            self._comp_tree, "Drag to reorder, or Alt+Up/Down",
        )

        # ── Plot tree + variant canvas ───────────────────────────────
        plots_label = ttk.Label(left, text=" Plots [P] ", font=_lf_font)
        tree_frame = ttk.LabelFrame(self._left_paned, labelwidget=plots_label, padding=5)
        attach_tooltip(plots_label, (
            "Available plots (plots without data greyed out).\n"
            "\n"
            "  \u2022 P \u2014 focus this list\n"
            "  \u2022 \u2191\u2193\u2190\u2192 \u2014 navigate plots\n"
            "  \u2022 PgUp/PgDn \u2014 prev/next file"
        ))
        self._left_paned.add(tree_frame, weight=1)
        tree_frame.columnconfigure(0, weight=1)
        # column 1 = variant canvas (fixed width, set later)
        # column 2 = scrollbar (fixed width)
        tree_frame.rowconfigure(0, weight=1)

        # Use a per-widget style so the variant-canvas focus dance below
        # (which hides this tree's selection while the canvas has focus)
        # doesn't bleed into the global "Treeview" style and disturb the
        # main window's input_sources / available / executed / jobs
        # selection rendering.
        self._plot_tree = ttk.Treeview(
            tree_frame,
            show="tree",
            selectmode="browse",
            style="PlotTree.Treeview",
        )
        self._plot_tree.grid(row=0, column=0, sticky="nsew")

        # Variant canvas — sits to the right of the tree, shares the scrollbar
        self._variant_canvas = tk.Canvas(
            tree_frame, width=0, highlightthickness=0,
        )
        self._variant_canvas.grid(row=0, column=1, sticky="ns")
        self._variant_canvas.configure(takefocus=True)

        # Shared vertical scrollbar
        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical")
        tree_scroll.grid(row=0, column=2, sticky="ns")

        def _tree_yscroll(*args):
            tree_scroll.set(*args)
            self._schedule_variant_redraw()

        tree_scroll.configure(command=self._plot_tree.yview)
        self._plot_tree.configure(yscrollcommand=_tree_yscroll)

        # Pending redraw id (to coalesce rapid scroll events)
        self._variant_redraw_id: str | None = None

        self._plot_tree.bind("<<TreeviewSelect>>", self._on_tree_selected)
        self._plot_tree.bind("<Motion>", self._on_tree_motion)
        self._plot_tree.bind("<Leave>", self._hide_tooltip)
        self._plot_tree.bind("<Up>", self._on_tree_key_up)
        self._plot_tree.bind("<Down>", self._on_tree_key_down)
        self._plot_tree.bind("<<TreeviewOpen>>", lambda e: self._schedule_variant_redraw())
        self._plot_tree.bind("<<TreeviewClose>>", lambda e: self._schedule_variant_redraw())
        self._plot_tree.bind("<Configure>", lambda e: self._schedule_variant_redraw())

        # Canvas click and keyboard bindings
        self._variant_canvas.bind("<Button-1>", self._on_variant_canvas_click)
        self._variant_canvas.bind("<FocusIn>", self._on_canvas_focus_in)
        self._variant_canvas.bind("<FocusOut>", lambda e: self._redraw_variant_grid())
        self._variant_canvas.bind("<Left>", self._on_variant_left)
        self._variant_canvas.bind("<Right>", self._on_variant_right)
        self._variant_canvas.bind("<Up>", self._on_variant_key_up)
        self._variant_canvas.bind("<Down>", self._on_variant_key_down)
        self._variant_canvas.bind("<Tab>", self._focus_scenario_listbox)

        # Tree focus management
        self._plot_tree.bind("<FocusIn>", self._on_tree_focus_in)

        # Tree Right arrow → move focus to variant canvas
        self._plot_tree.bind("<Right>", self._on_tree_key_right)
        self._plot_tree.bind("<Left>", self._on_tree_key_left)

    def _build_right_column(self) -> None:
        """Build the right column: compact control bar + plot area."""
        right = ttk.Frame(self._paned, padding=5)
        self._paned.add(right, weight=1)

        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)  # plot area gets all extra space

        # ── Combined control frame ───────────────────────────────────
        self._control_frame = ttk.Frame(right, padding=(5, 2))
        self._control_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        self._control_frame.columnconfigure(2, weight=1)  # time frame fills remaining

        # Col 0: File navigation (Prev on top, Next on bottom)
        file_nav_frame = ttk.Frame(self._control_frame)
        file_nav_frame.grid(row=0, column=0, rowspan=2, sticky="ns", padx=(0, 5))

        self._prev_file_btn = ttk.Button(
            file_nav_frame, text="\u25c0 Prev", width=6,
            command=self._on_prev_file,
        )
        self._prev_file_btn.pack(side="top", pady=(0, 1))

        self._file_label = ttk.Label(file_nav_frame, text="File 1/1", anchor="center")
        self._file_label.pack(side="top", pady=1)

        self._next_file_btn = ttk.Button(
            file_nav_frame, text="Next \u25b6", width=6,
            command=self._on_next_file,
        )
        self._next_file_btn.pack(side="top", pady=(1, 0))

        self._update_file_nav()

        # Col 1: Mode radio buttons (stacked vertically)
        mode_frame = ttk.Frame(self._control_frame)
        mode_frame.grid(row=0, column=1, rowspan=2, sticky="ns", padx=(0, 10))

        for text, value in [("Single", "single"), ("Comparison", "comparison"), ("Dispatch", "dispatch"), ("Network", "network")]:
            rb = ttk.Radiobutton(
                mode_frame, text=text, variable=self._mode, value=value,
                command=self._on_mode_changed,
            )
            rb.pack(side="top", anchor="w")

        # Col 2: Start slider + Duration spinbox (stacked, slider fills width)
        time_frame = ttk.Frame(self._control_frame)
        time_frame.grid(row=0, column=2, rowspan=2, sticky="nsew", padx=(0, 10))
        time_frame.columnconfigure(1, weight=1)

        ttk.Label(time_frame, text="Start").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self._start_var = tk.IntVar(value=self._settings.single_plot_settings.start_time)
        self._start_scale = ttk.Scale(
            time_frame, from_=0, to=8760, orient="horizontal",
            variable=self._start_var,
        )
        self._start_scale.grid(row=0, column=1, sticky="ew")

        ttk.Label(time_frame, text="Duration").grid(row=1, column=0, sticky="w", padx=(0, 5))
        self._duration_steps = (1, 2, 3, 4, 6, 12, 24, 72, 168, 240, 336, 504, 672, 1344, 2688, 5376, 8760)
        self._duration_var = tk.IntVar(value=self._settings.single_plot_settings.duration or 168)
        self._duration_spin = ttk.Spinbox(
            time_frame, values=self._duration_steps,
            textvariable=self._duration_var, width=6,
        )
        self._duration_spin.grid(row=1, column=1, sticky="w")

        self._change_colors_btn = ttk.Button(
            time_frame, text="Plot settings", command=self._on_change_colors,
        )
        self._change_colors_btn.grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(4, 0),
        )

        # Bind changes to trigger replot
        self._start_var.trace_add("write", self._on_time_range_changed)
        self._duration_var.trace_add("write", self._on_time_range_changed)

        # ── Plot canvas ──────────────────────────────────────────────
        self._plot_canvas = PlotCanvas(right)
        self._plot_canvas._cache.max_gb = self._viewer_settings.cache_gb
        self._plot_canvas.grid(row=1, column=0, sticky="nsew")
        self._plot_canvas.show_message("Select a plot to display")

    # ------------------------------------------------------------------
    # Config path resolution
    # ------------------------------------------------------------------

    def _resolve_config_path(self, user_config: str, default_relative: str) -> Path:
        """Resolve a plot config path.

        If *user_config* is set and exists, use it.  Otherwise fall back
        to the bundled default that ships with the package (no longer a
        repo-root path — works in wheel installs).
        """
        if user_config:
            p = Path(user_config)
            if p.is_absolute() and p.is_file():
                return p
            # Try relative to project
            candidate = self._project_path / user_config
            if candidate.is_file():
                return candidate

        # ``default_relative`` historically pointed at the repo-root
        # ``templates/`` dir; the canonical YAML now lives inside the
        # package as ``schemas/`` and is fetched via
        # ``package_data_path`` so wheel installs find it too.
        from flextool._resources import package_data_path
        basename = Path(default_relative).name
        return package_data_path(f"schemas/{basename}")

    # ------------------------------------------------------------------
    # Scenario discovery
    # ------------------------------------------------------------------

    def _scan_scenarios(self) -> list[str]:
        """List scenario subdirectories in output_parquet/ that are checked.

        Subdirectories are either the bare scenario name (when this
        source owns it) or ``<scenario_name>_<source_number>`` (when
        another source owns the bare name). The checked set in project
        settings uses the compound key ``<source_number>|<scenario_name>``
        — we translate between the two forms here using the
        bare-ownership map so the viewer's internal identifier remains
        the on-disk subdir.

        Order is taken from ``settings.executed_scenario_order``: saved
        names that still exist on disk come first (in saved order),
        followed by any newly-seen names in alphabetical order. The
        settings list is rewritten to match (drops missing names, appends
        new ones) and persisted via the debounced save path.
        """
        from flextool.gui.scenario_key import resolve_source_number, format_key
        parquet_dir = self._project_path / "output_parquet"
        if not parquet_dir.is_dir():
            return []
        on_disk = sorted(
            d.name for d in parquet_dir.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        )
        checked_keys = set(self._settings.checked_executed_scenarios)
        if checked_keys:
            bare_owners = self._settings.bare_output_owners
            on_disk = [
                s for s in on_disk
                if format_key(*resolve_source_number(s, bare_owners)) in checked_keys
            ]

        saved_order = list(self._settings.executed_scenario_order)
        on_disk_set = set(on_disk)
        ordered: list[str] = [s for s in saved_order if s in on_disk_set]
        ordered_set = set(ordered)
        new_names = [s for s in on_disk if s not in ordered_set]
        resolved = ordered + new_names

        if saved_order != resolved:
            self._settings.executed_scenario_order = list(resolved)
            try:
                self._schedule_settings_save()
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Failed to schedule executed_scenario_order save",
                    exc_info=True,
                )

        return resolved

    def _populate_scenarios(self) -> None:
        """Populate the scenario listbox."""
        self._scenario_listbox.delete(0, "end")
        scenarios = self._scan_scenarios()
        for name in scenarios:
            self._scenario_listbox.insert("end", name)

        if not scenarios:
            return

        # Try to restore last selected scenario
        target = self._viewer_settings.last_scenario
        idx = 0
        if target and target in scenarios:
            idx = scenarios.index(target)

        self._scenario_listbox.selection_set(idx)
        self._scenario_listbox.see(idx)
        self._scenario_listbox.event_generate("<<ListboxSelect>>")

    def show_scenario_in_single_mode(self, scenario_name: str) -> None:
        """Switch to single mode and select *scenario_name* in the list.

        Used by the main window's "view" button so a click on one
        executed scenario lands the user on its single-mode view. The
        plot entry/variant restoration that follows is driven by the
        viewer's existing last-entry/last-variant tracking, which is
        intentionally not scenario-specific.
        """
        if self._mode.get() != "single":
            self._mode.set("single")
            self._on_mode_changed()
        try:
            entries = list(self._scenario_listbox.get(0, "end"))
        except tk.TclError:
            return
        if scenario_name not in entries:
            return
        idx = entries.index(scenario_name)
        self._scenario_listbox.selection_clear(0, "end")
        self._scenario_listbox.selection_set(idx)
        self._scenario_listbox.see(idx)
        self._scenario_listbox.event_generate("<<ListboxSelect>>")

    # ------------------------------------------------------------------
    # Plot tree population
    # ------------------------------------------------------------------

    def _get_config_path_for_mode(self) -> Path:
        """Return the config path for the current mode."""
        mode = self._mode.get()
        if mode == "comparison":
            return self._comparison_config_path
        return self._single_config_path

    def _populate_plot_tree(self) -> None:
        """Parse config YAML and build the plot tree."""
        # Clear existing tree
        for item in self._plot_tree.get_children():
            self._plot_tree.delete(item)
        self._tree_entry_map.clear()

        mode = self._mode.get()
        if mode == "network" or mode == "dispatch":
            return

        config_path = self._get_config_path_for_mode()
        parsed = parse_plot_config(config_path)
        if isinstance(parsed, PlotConfigData):
            self._plot_groups = parsed.groups
            self._template_default_durations = dict(parsed.default_durations)
        else:  # defensive: legacy list shape
            self._plot_groups = list(parsed)
            self._template_default_durations = {}

        for group in self._plot_groups:
            group_iid = f"group_{group.number}"
            group_text = f"{group.number} {group.name}"
            self._plot_tree.insert(
                "", "end", iid=group_iid, text=group_text, open=True,
            )

            for entry in group.entries:
                entry_iid = f"entry_{entry.number}"
                label = f"{entry.number} {entry.full_name}"
                self._plot_tree.insert(
                    group_iid, "end", iid=entry_iid, text=label,
                )
                self._tree_entry_map[entry_iid] = entry

        # Collect all unique variant letters and size the canvas
        self._collect_all_variant_letters()
        self._update_variant_canvas_width()

        # Grey out entries without matching parquet data
        self._update_tree_availability()

        # Try to restore last selected entry or select first available
        self._restore_or_select_first_entry()

    def _load_availability_from_dir(self, plan_dir: Path) -> set[tuple[str, str]]:
        """Load availability manifest from plan_dir/_availability.json."""
        import json

        avail_path = plan_dir / "_availability.json"
        if not avail_path.exists():
            return set()
        try:
            with open(avail_path) as f:
                data = json.load(f)
            return {(r, s) for r, s in data.get("available", [])}
        except (json.JSONDecodeError, OSError):
            return set()

    def _resolve_comparison_availability(self) -> set[tuple[str, str]]:
        """Return the (result_key, sub_config) availability for comparison mode.

        Resolution order (Phase C):

        1. If ``output_parquet_comparison/plot_plans/_availability.json``
           exists AND ``output_parquet_comparison/_metadata.json`` records
           the same viewer-scenarios set as
           :meth:`_get_comparison_viewer_scenarios`, use the combined
           availability file (authoritative for the actual combined plan
           parquets).
        2. Else compute the **union** of each viewer scenario's per-scenario
           ``output_parquet/<scenario>/plot_plans/_availability.json``.
           This makes grey-out state accurate immediately after a "Update
           view scenarios" press, before the slow combine has finished
           rewriting the comparison availability file.
        3. Else return an empty set (everything greys out, no plans known).

        A per-scenario file missing for one of the viewer scenarios is
        treated as "no contribution" — the union proceeds with whatever
        files are present.  This silently drops coverage for that scenario
        until its per-scenario run finishes; that's deliberate so the user
        sees the most accurate available picture without erroring out.
        """
        comparison_plan_dir = (
            self._project_path / "output_parquet_comparison" / "plot_plans"
        )
        comparison_avail_path = comparison_plan_dir / "_availability.json"

        viewer_scenarios = self._get_comparison_viewer_scenarios()
        viewer_set = {str(s) for s in viewer_scenarios}

        if comparison_avail_path.exists():
            metadata_scenarios = set(self._read_metadata_scenarios())
            if metadata_scenarios == viewer_set and viewer_set:
                return self._load_availability_from_dir(comparison_plan_dir)

        if not viewer_set:
            return set()

        union: set[tuple[str, str]] = set()
        for scenario in viewer_scenarios:
            plan_dir = (
                self._project_path / "output_parquet" / scenario / "plot_plans"
            )
            union |= self._load_availability_from_dir(plan_dir)
        return union

    def _update_tree_availability(self) -> None:
        """Grey out entries that have no matching parquet/plan data for selected scenario(s).

        Also stores the loaded availability set in ``self._current_availability``
        so that the variant grid can distinguish *defined-but-unavailable* from
        *available* variants.
        """
        mode = self._mode.get()

        if mode == "comparison":
            available_pairs = self._resolve_comparison_availability()
        else:
            scenarios = self._get_selected_scenarios()
            if not scenarios:
                # No scenario selected -- mark all as disabled
                self._current_availability = set()
                for iid in self._tree_entry_map:
                    self._plot_tree.item(iid, tags=("disabled",))
                return

            # Merge availability across all selected scenarios
            available_pairs: set[tuple[str, str]] = set()
            for scenario in scenarios:
                plan_dir = self._project_path / "output_parquet" / scenario / "plot_plans"
                available_pairs |= self._load_availability_from_dir(plan_dir)

        self._current_availability = available_pairs

        for iid, entry in self._tree_entry_map.items():
            # An entry is available if ANY of its variants has data
            has_any = any(
                (v.result_key, v.sub_config) in available_pairs
                or (v.result_key, "*") in available_pairs
                for v in entry.variants
            )
            if has_any:
                self._plot_tree.item(iid, tags=())
            else:
                self._plot_tree.item(iid, tags=("disabled",))

    def _is_entry_disabled(self, iid: str) -> bool:
        """Return True if the tree item has the 'disabled' tag."""
        return "disabled" in self._plot_tree.item(iid, "tags")

    def _restore_or_select_first_entry(self) -> None:
        """Select the last entry from settings, or the first available entry."""
        target_iid = f"entry_{self._viewer_settings.last_entry}" if self._viewer_settings.last_entry else ""

        if target_iid and self._plot_tree.exists(target_iid) and not self._is_entry_disabled(target_iid):
            self._plot_tree.selection_set(target_iid)
            self._plot_tree.see(target_iid)
            return

        # Select first non-disabled entry
        for group_iid in self._plot_tree.get_children():
            for entry_iid in self._plot_tree.get_children(group_iid):
                if not self._is_entry_disabled(entry_iid):
                    self._plot_tree.selection_set(entry_iid)
                    self._plot_tree.see(entry_iid)
                    return

    # ------------------------------------------------------------------
    # Scenario selection
    # ------------------------------------------------------------------

    def _get_selected_scenarios(self) -> list[str]:
        """Return the list of currently selected scenario names."""
        if self._mode.get() == "comparison":
            return self._get_comparison_scenarios()
        indices = self._scenario_listbox.curselection()
        return [self._scenario_listbox.get(i) for i in indices]

    def _on_scenario_selected(self, _event: tk.Event | None = None) -> None:
        """Handle scenario listbox selection change."""
        scenarios = self._get_selected_scenarios()
        if scenarios:
            self._viewer_settings.last_scenario = scenarios[0]

        mode = self._mode.get()
        if mode == "network":
            self._render_network()
            return
        if mode == "dispatch":
            # Re-populate dispatch tree (nodeGroups may differ per scenario)
            self._dispatch_scenario = ""  # force reload
            self._populate_dispatch_tree()
            # Trigger replot if something is selected
            selection = self._plot_tree.selection()
            if selection and selection[0].startswith("dispatch_"):
                node_group = selection[0][len("dispatch_"):]
                if scenarios:
                    self._display_dispatch(scenarios[0], node_group)
            return
        if mode == "comparison":
            # Scenario selection is informational in comparison mode
            self._trigger_replot()
            return

        # Single mode
        self._update_tree_availability()
        # Re-select entry if current one became disabled
        selection = self._plot_tree.selection()
        if selection and self._is_entry_disabled(selection[0]):
            self._restore_or_select_first_entry()
        else:
            self._trigger_replot()

    # ------------------------------------------------------------------
    # Tree selection
    # ------------------------------------------------------------------

    def _on_tree_selected(self, _event: tk.Event | None = None) -> None:
        """Handle plot tree selection change."""
        selection = self._plot_tree.selection()
        if not selection:
            return

        iid = selection[0]

        # Skip group headers — let the tree handle expand/collapse
        if iid.startswith("group_"):
            return

        # Handle dispatch mode
        if iid.startswith("dispatch_"):
            node_group = iid[len("dispatch_"):]
            scenarios = self._get_selected_scenarios()
            if scenarios:
                self._display_dispatch(scenarios[0], node_group)
            return

        # If disabled, find nearest non-disabled entry
        if self._is_entry_disabled(iid):
            self._restore_or_select_first_entry()
            return

        entry = self._tree_entry_map.get(iid)
        if entry is None:
            return

        # Update viewer settings
        self._viewer_settings.last_entry = entry.number

        # Update active entry iid
        self._active_entry_iid = iid

        # Reset file index and clear prefetched figures
        self._file_index = 0
        self._file_count = 1
        self._clear_figure_cache()

        self._update_file_nav()
        self._populate_variant_panel(entry)
        self._trigger_replot()

    # ------------------------------------------------------------------
    # Tree keyboard navigation
    # ------------------------------------------------------------------

    def _get_all_visible_items(self) -> list[str]:
        """Return flat list of all visible tree items (groups + entries)."""
        visible: list[str] = []
        for group_iid in self._plot_tree.get_children():
            visible.append(group_iid)
            if self._plot_tree.item(group_iid, "open"):
                for entry_iid in self._plot_tree.get_children(group_iid):
                    visible.append(entry_iid)
        return visible

    def _should_skip_item(self, iid: str) -> bool:
        """Return True if this item should be skipped during navigation.

        Open groups are skipped (their children are visible).
        Closed groups stop the cursor so the user can open them.
        Disabled (greyed-out) entries are always skipped.
        """
        if "disabled" in self._plot_tree.item(iid, "tags"):
            return True
        if not iid.startswith("group_"):
            return False
        return bool(self._plot_tree.item(iid, "open"))

    def _on_tree_key_right(self, event: tk.Event) -> str:
        """Handle Right arrow in tree.

        On a closed group: open it (default behavior).
        On an entry or open group: move focus to variant canvas.
        """
        selection = self._plot_tree.selection()
        if selection:
            iid = selection[0]
            if iid.startswith("group_") and not self._plot_tree.item(iid, "open"):
                # Let default handler open the branch
                self._plot_tree.item(iid, open=True)
                self._plot_tree.event_generate("<<TreeviewOpen>>")
                return "break"
        # Move focus to variant canvas
        self._variant_canvas.focus_set()
        return "break"

    def _on_tree_key_left(self, event: tk.Event) -> str:
        """Handle Left arrow in tree.

        On an open group: close it.
        On an entry: select the parent group if it's open, or do nothing.
        """
        selection = self._plot_tree.selection()
        if selection:
            iid = selection[0]
            if iid.startswith("group_") and self._plot_tree.item(iid, "open"):
                self._plot_tree.item(iid, open=False)
                self._plot_tree.event_generate("<<TreeviewClose>>")
                return "break"
            if iid.startswith("entry_"):
                # Move to parent group
                parent = self._plot_tree.parent(iid)
                if parent:
                    self._plot_tree.selection_set(parent)
                    self._plot_tree.see(parent)
                return "break"
        return "break"

    def _on_tree_key_up(self, event: tk.Event) -> str:
        """Move selection up in the tree.

        Skips open group headers (their children are navigable instead).
        Stops on closed group headers so user can open them.
        """
        visible = self._get_all_visible_items()
        selection = self._plot_tree.selection()
        if not selection:
            return "break"
        current = selection[0]
        if current not in visible:
            # Current item not visible (e.g., parent was closed) — select last visible
            if visible:
                self._plot_tree.selection_set(visible[-1])
                self._plot_tree.see(visible[-1])
            return "break"
        idx = visible.index(current)
        for new_idx in range(idx - 1, -1, -1):
            if not self._should_skip_item(visible[new_idx]):
                self._plot_tree.selection_set(visible[new_idx])
                self._plot_tree.see(visible[new_idx])
                self._plot_tree.event_generate("<<TreeviewSelect>>")
                return "break"
        return "break"

    def _on_tree_key_down(self, event: tk.Event) -> str:
        """Move selection down in the tree.

        Skips open group headers (their children are navigable instead).
        Stops on closed group headers so user can open them.
        """
        visible = self._get_all_visible_items()
        selection = self._plot_tree.selection()
        if not selection:
            return "break"
        current = selection[0]
        if current not in visible:
            # Current item not visible — select first visible
            if visible:
                self._plot_tree.selection_set(visible[0])
                self._plot_tree.see(visible[0])
            return "break"
        idx = visible.index(current)
        for new_idx in range(idx + 1, len(visible)):
            if not self._should_skip_item(visible[new_idx]):
                self._plot_tree.selection_set(visible[new_idx])
                self._plot_tree.see(visible[new_idx])
                self._plot_tree.event_generate("<<TreeviewSelect>>")
                return "break"
        return "break"

    # ------------------------------------------------------------------
    # Tree tooltip
    # ------------------------------------------------------------------

    def _on_tree_motion(self, event: tk.Event) -> None:
        """Show tooltip with full name when hovering over an entry."""
        item = self._plot_tree.identify_row(event.y)
        if not item or (not item.startswith("entry_") and not item.startswith("dispatch_")):
            self._hide_tooltip()
            return

        # Dispatch items use the nodeGroup name directly
        if item.startswith("dispatch_"):
            full_text = item[len("dispatch_"):]
        else:
            entry = self._tree_entry_map.get(item)
            if entry is None:
                self._hide_tooltip()
                return
            full_text = f"{entry.number} {entry.full_name}"
        if self._tooltip is not None:
            try:
                self._tooltip_label.configure(text=full_text)
                self._tooltip.geometry(
                    f"+{event.x_root + 15}+{event.y_root + 10}"
                )
                return
            except tk.TclError:
                self._tooltip = None

        self._tooltip = tk.Toplevel(self)
        self._tooltip.wm_overrideredirect(True)
        self._tooltip.wm_geometry(f"+{event.x_root + 15}+{event.y_root + 10}")

        # Use a plain tk.Label (not ttk) so we can set explicit colors
        # that work in both light and dark themes.
        self._tooltip_label = tk.Label(
            self._tooltip, text=full_text,
            background="#333333", foreground="#ffffff",
            relief="solid", borderwidth=1,
            padx=4, pady=2,
        )
        self._tooltip_label.pack()

    def _hide_tooltip(self, _event: tk.Event | None = None) -> None:
        """Destroy the tooltip if it exists."""
        if self._tooltip is not None:
            try:
                self._tooltip.destroy()
            except tk.TclError:
                pass
            self._tooltip = None

    # ------------------------------------------------------------------
    # Variant panel
    # ------------------------------------------------------------------

    def _collect_all_variant_letters(self) -> None:
        """Collect all unique variant letters, respecting order_of_variants from config."""
        # Gather all letters that actually appear in the config
        all_letters: set[str] = set()
        for group in self._plot_groups:
            for entry in group.entries:
                for v in entry.variants:
                    all_letters.add(v.letter)

        # Read order_of_variants from the YAML config
        config_path = self._get_config_path_for_mode()
        config_order: list[str] = []
        if config_path in self._yaml_cache:
            config_order = self._yaml_cache[config_path].get("order_of_variants", [])
        elif config_path.is_file():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict):
                    config_order = data.get("order_of_variants", [])
                    self._yaml_cache[config_path] = data
            except (yaml.YAMLError, OSError):
                pass

        # Use config order for known letters, then append any extras
        ordered: list[str] = []
        for letter in config_order:
            if letter in all_letters:
                ordered.append(letter)
                all_letters.discard(letter)
        # Append any remaining letters not in the config order
        for letter in sorted(all_letters):
            ordered.append(letter)

        self._all_variant_letters = ordered

    def _update_variant_canvas_width(self) -> None:
        """Set the variant canvas width based on the number of variant letters."""
        if not self._all_variant_letters:
            self._variant_canvas.configure(width=0)
            return
        box_w = self._char_width * 2
        n_letters = len(self._all_variant_letters)
        self._variant_canvas.configure(width=n_letters * box_w + 2)

    def _schedule_variant_redraw(self) -> None:
        """Schedule a variant grid redraw, coalescing rapid events."""
        if self._variant_redraw_id is not None:
            self.after_cancel(self._variant_redraw_id)
        self._variant_redraw_id = self.after(10, self._do_variant_redraw)

    def _do_variant_redraw(self) -> None:
        """Execute the deferred variant grid redraw."""
        self._variant_redraw_id = None
        self._redraw_variant_grid()

    def _redraw_variant_grid(self) -> None:
        """Redraw all variant boxes on the canvas to align with visible tree rows."""
        self._variant_canvas.delete("all")

        if not self._all_variant_letters:
            return

        box_w = self._char_width * 2

        # Draw the letters in the *same* font the themed tree rows use, not
        # the bare TkDefaultFont. Under sv_ttk a ttk.Treeview renders its
        # cells in SunValleyBodyFont (≈14pt) while TkDefaultFont is ≈10pt,
        # so a hard-coded TkDefaultFont made the variant letters look 2-3
        # sizes smaller than the adjacent tree text on every platform.
        letter_font = str(
            ttk.Style().lookup("PlotTree.Treeview", "font")
        ) or "TkDefaultFont"

        # Determine focus state
        try:
            canvas_has_focus = (self.focus_get() is self._variant_canvas)
        except KeyError:
            canvas_has_focus = False
        tree_selection = self._plot_tree.selection()
        selected_iid = tree_selection[0] if tree_selection else ""

        for iid, entry in self._tree_entry_map.items():
            try:
                bbox = self._plot_tree.bbox(iid)
            except (tk.TclError, ValueError):
                continue
            if not bbox:
                continue  # item not visible

            _, y, _, row_h = bbox
            # Build a lookup from letter -> PlotVariant for this entry
            variant_by_letter: dict[str, object] = {v.letter: v for v in entry.variants}

            for col_idx, letter in enumerate(self._all_variant_letters):
                x = col_idx * box_w

                # Find the PlotVariant for this letter (if defined for this entry)
                variant = variant_by_letter.get(letter)
                is_defined = variant is not None

                # Check if this specific variant has data (is available)
                is_available = False
                if is_defined:
                    is_available = (
                        (variant.result_key, variant.sub_config) in self._current_availability
                        or (variant.result_key, "*") in self._current_availability
                    )

                # Active plot indicator (solid blue) -- always visible for the displayed plot
                is_active = (
                    is_defined
                    and iid == self._active_entry_iid
                    and letter == self._active_variant
                )

                # Focus cursor (dashed rectangle) -- shown on ANY cell when canvas has focus
                is_focused = (
                    canvas_has_focus
                    and iid == selected_iid
                    and col_idx == self._focus_col
                )

                # Visual states for defined variants
                if is_active:
                    self._variant_canvas.create_rectangle(
                        x + 1, y + 1, x + box_w - 1, y + row_h - 1,
                        fill="#2074d5", outline="",
                    )
                    text_color = "#ffffff"
                elif not is_defined:
                    text_color = None  # no text for undefined
                elif not is_available:
                    text_color = "grey"
                else:
                    text_color = self._fg_color

                # Focus rectangle drawn on ANY cell (including blank/grey)
                if is_focused:
                    self._variant_canvas.create_rectangle(
                        x + 2, y + 2, x + box_w - 2, y + row_h - 2,
                        outline="#ff8800", width=2, dash=(4, 4),
                    )

                # Draw letter text (skip for undefined cells)
                if text_color is not None:
                    display = letter if letter else "?"
                    self._variant_canvas.create_text(
                        x + box_w // 2, y + row_h // 2,
                        text=display, fill=text_color, font=letter_font,
                    )

    def _on_variant_canvas_click(self, event: tk.Event) -> None:
        """Handle click on a variant box in the canvas."""
        self._variant_canvas.focus_set()
        if not self._all_variant_letters:
            return

        box_w = self._char_width * 2

        # Find which letter column was clicked
        col_idx = int(event.x / box_w)
        if col_idx < 0 or col_idx >= len(self._all_variant_letters):
            return
        self._focus_col = col_idx

        # Find which entry row was clicked (match y to tree bbox)
        for iid, entry in self._tree_entry_map.items():
            try:
                bbox = self._plot_tree.bbox(iid)
            except (tk.TclError, ValueError):
                continue
            if not bbox:
                continue
            _, y, _, row_h = bbox
            if y <= event.y < y + row_h:
                # Select this entry in tree
                self._plot_tree.selection_set(iid)
                self._plot_tree.see(iid)
                # Try to activate the focused cell
                self._try_activate_focused()
                self._redraw_variant_grid()
                return

    def _find_nearest_available(self, available: set[str]) -> str:
        """Find nearest available variant letter to the active one.

        Searches left first, then right in the _all_variant_letters list.
        """
        if not available:
            return ""
        if self._active_variant in available:
            return self._active_variant

        try:
            idx = self._all_variant_letters.index(self._active_variant)
        except ValueError:
            idx = 0

        # Search left then right
        for offset in range(1, len(self._all_variant_letters)):
            left_idx = idx - offset
            if left_idx >= 0 and self._all_variant_letters[left_idx] in available:
                return self._all_variant_letters[left_idx]
            right_idx = idx + offset
            if right_idx < len(self._all_variant_letters) and self._all_variant_letters[right_idx] in available:
                return self._all_variant_letters[right_idx]

        # Fallback
        return next(iter(available))

    def _populate_variant_panel(self, entry: PlotEntry) -> None:
        """Update variant state for the given entry and redraw the grid.

        If the active variant is not available for this entry, picks the
        nearest available variant.  The active variant persists across
        tree navigation when possible.
        """
        available_with_data: set[str] = set()
        for v in entry.variants:
            is_avail = (
                (v.result_key, v.sub_config) in self._current_availability
                or (v.result_key, "*") in self._current_availability
            )
            if is_avail:
                available_with_data.add(v.letter)

        if self._active_variant not in available_with_data and available_with_data:
            self._active_variant = self._find_nearest_available(available_with_data)

        self._redraw_variant_grid()

    def _on_variant_clicked(self, letter: str) -> None:
        """Handle variant selection (from canvas click or keyboard)."""
        selection = self._plot_tree.selection()
        if not selection:
            return
        iid = selection[0]
        if not iid.startswith("entry_"):
            return
        entry = self._tree_entry_map.get(iid)
        if not entry:
            return

        # Only act if the variant is defined and available for the selected entry
        variant = next((v for v in entry.variants if v.letter == letter), None)
        if variant is None:
            return  # not defined
        is_available = (
            (variant.result_key, variant.sub_config) in self._current_availability
            or (variant.result_key, "*") in self._current_availability
        )
        if not is_available:
            return  # defined but no data

        self._active_variant = letter
        self._active_entry_iid = iid
        self._viewer_settings.last_variant = letter
        self._redraw_variant_grid()
        self._file_index = 0
        self._clear_figure_cache()
        self._trigger_replot()

    def _on_variant_left(self, event: tk.Event) -> str:
        """Navigate focus cursor left in the variant grid.

        Moves to ANY cell position (including empty/undefined ones).
        At the leftmost position, returns focus to the tree.
        """
        if self._focus_col <= 0:
            # At leftmost -- return focus to tree
            self._plot_tree.focus_set()
            return "break"
        self._focus_col -= 1
        self._try_activate_focused()
        self._redraw_variant_grid()
        return "break"

    def _on_variant_right(self, event: tk.Event) -> str:
        """Navigate focus cursor right in the variant grid.

        Moves to ANY cell position (including empty/undefined ones).
        """
        if not self._all_variant_letters:
            return "break"
        if self._focus_col >= len(self._all_variant_letters) - 1:
            return "break"  # at rightmost
        self._focus_col += 1
        self._try_activate_focused()
        self._redraw_variant_grid()
        return "break"

    def _get_selected_entry_available_variants(self) -> set[str]:
        """Return variant letters that are both defined and have data for the selected entry."""
        selection = self._plot_tree.selection()
        if selection and selection[0].startswith("entry_"):
            entry = self._tree_entry_map.get(selection[0])
            if entry:
                result: set[str] = set()
                for v in entry.variants:
                    if (
                        (v.result_key, v.sub_config) in self._current_availability
                        or (v.result_key, "*") in self._current_availability
                    ):
                        result.add(v.letter)
                return result
        return set()

    def _on_variant_key_up(self, event: tk.Event) -> str:
        """Handle Up / Shift+Up in the variant panel.

        - Up: jump to prev row with any available data. Show focus variant
          if available, otherwise find nearest available and activate it.
        - Shift+Up: jump to prev row that has available data at the
          current focus column specifically.
        """
        shift_held = bool(event.state & 0x1)
        if shift_held:
            self._move_to_next_with_focus_col(-1)
        else:
            self._move_to_next_available_row(-1)
        return "break"

    def _on_variant_key_down(self, event: tk.Event) -> str:
        """Handle Down / Shift+Down in the variant panel.

        - Down: jump to next row with any available data. Show focus variant
          if available, otherwise find nearest available and activate it.
        - Shift+Down: jump to next row that has available data at the
          current focus column specifically.
        """
        shift_held = bool(event.state & 0x1)
        if shift_held:
            self._move_to_next_with_focus_col(1)
        else:
            self._move_to_next_available_row(1)
        return "break"

    def _get_visible_entries(self) -> list[str]:
        """Return flat list of entry iids in open branches."""
        visible: list[str] = []
        for group_iid in self._plot_tree.get_children():
            if self._plot_tree.item(group_iid, "open"):
                for entry_iid in self._plot_tree.get_children(group_iid):
                    visible.append(entry_iid)
        return visible

    def _move_tree_selection(self, direction: int) -> None:
        """Move tree selection by *direction* (-1 = up, +1 = down), skipping disabled entries."""
        selection = self._plot_tree.selection()
        if not selection:
            self._restore_or_select_first_entry()
            return

        current = selection[0]
        visible = self._get_visible_entries()

        if current not in visible:
            self._restore_or_select_first_entry()
            return

        idx = visible.index(current)
        step = 1 if direction > 0 else -1
        new_idx = idx + step

        while 0 <= new_idx < len(visible):
            if not self._is_entry_disabled(visible[new_idx]):
                self._plot_tree.selection_set(visible[new_idx])
                self._plot_tree.see(visible[new_idx])
                self._plot_tree.event_generate("<<TreeviewSelect>>")
                return
            new_idx += step

    def _move_to_next_with_focus_col(self, direction: int) -> None:
        """Jump to next/prev row that has available data at the focus column.

        The focus column stays fixed; only rows where that specific
        variant letter has data are considered.
        """
        if self._focus_col < 0 or self._focus_col >= len(self._all_variant_letters):
            return
        focus_letter = self._all_variant_letters[self._focus_col]

        visible = self._get_visible_entries()
        selection = self._plot_tree.selection()
        if not selection or selection[0] not in visible:
            return
        idx = visible.index(selection[0])
        new_idx = idx + direction

        while 0 <= new_idx < len(visible):
            iid = visible[new_idx]
            entry = self._tree_entry_map.get(iid)
            if entry:
                # Check if this entry has an available variant at focus_letter
                for v in entry.variants:
                    if v.letter == focus_letter and (
                        (v.result_key, v.sub_config) in self._current_availability
                        or (v.result_key, "*") in self._current_availability
                    ):
                        self._plot_tree.selection_set(iid)
                        self._plot_tree.see(iid)
                        self._active_variant = focus_letter
                        self._active_entry_iid = iid
                        self._viewer_settings.last_variant = focus_letter
                        self._viewer_settings.last_entry = entry.number
                        self._file_index = 0
                        self._clear_figure_cache()
                        self._trigger_replot()
                        self._redraw_variant_grid()
                        return
            new_idx += direction

    def _move_tree_to_next_with_active(self, direction: int) -> None:
        """Move tree selection to next/prev entry that has the active variant.

        Only stops at non-disabled entries whose variant set includes
        ``_active_variant``.
        """
        visible = self._get_visible_entries()
        selection = self._plot_tree.selection()
        if not selection or selection[0] not in visible:
            return

        idx = visible.index(selection[0])
        step = 1 if direction > 0 else -1
        new_idx = idx + step

        while 0 <= new_idx < len(visible):
            iid = visible[new_idx]
            if not self._is_entry_disabled(iid):
                entry = self._tree_entry_map.get(iid)
                if entry and any(
                    v.letter == self._active_variant
                    and (
                        (v.result_key, v.sub_config) in self._current_availability
                        or (v.result_key, "*") in self._current_availability
                    )
                    for v in entry.variants
                ):
                    self._plot_tree.selection_set(iid)
                    self._plot_tree.see(iid)
                    self._plot_tree.event_generate("<<TreeviewSelect>>")
                    return
            new_idx += step

    def _get_current_variant_index(self) -> int:
        """Return index of the active variant in _all_variant_letters, or 0."""
        if self._active_variant in self._all_variant_letters:
            return self._all_variant_letters.index(self._active_variant)
        return 0

    def _show_variant_panel(self) -> None:
        """Show the variant canvas."""
        self._update_variant_canvas_width()
        self._redraw_variant_grid()

    def _hide_variant_panel(self) -> None:
        """Hide the variant canvas."""
        self._variant_canvas.configure(width=0)
        self._variant_canvas.delete("all")

    # ------------------------------------------------------------------
    # Focus management
    # ------------------------------------------------------------------

    def _on_canvas_focus_in(self, event: tk.Event) -> None:
        """Handle variant canvas receiving focus."""
        self._focus_col = max(0, self._focus_col)  # ensure valid
        # Hide the plot-tree selection while focus is in the variant
        # canvas. Scoped to PlotTree.Treeview so other trees keep their
        # blue selection.
        style = ttk.Style()
        style.map(
            "PlotTree.Treeview",
            background=[
                ("selected !focus", self._bg_color),
                ("selected", self._bg_color),
            ],
            foreground=[
                ("selected !focus", self._fg_color),
                ("selected", self._fg_color),
            ],
        )
        self._redraw_variant_grid()

    def _on_tree_focus_in(self, event: tk.Event) -> None:
        """Handle plot tree receiving focus."""
        self._focus_col = -1  # focus is in tree, not canvas
        # Restore the plot-tree selection highlight (still scoped to
        # PlotTree.Treeview).
        style = ttk.Style()
        style.map(
            "PlotTree.Treeview",
            background=[
                ("selected !focus", "#2074d5"),
                ("selected", "#2074d5"),
            ],
            foreground=[
                ("selected !focus", "#ffffff"),
                ("selected", "#ffffff"),
            ],
        )
        self._redraw_variant_grid()

    def _try_activate_focused(self) -> None:
        """If the focused cell has an available variant, make it the active plot."""
        if self._focus_col < 0 or self._focus_col >= len(self._all_variant_letters):
            return
        letter = self._all_variant_letters[self._focus_col]

        selection = self._plot_tree.selection()
        if not selection:
            return
        iid = selection[0]
        entry = self._tree_entry_map.get(iid)
        if not entry:
            return

        # Check if this variant is defined AND available
        variant = next((v for v in entry.variants if v.letter == letter), None)
        if variant is None:
            return  # undefined -- do nothing
        is_available = (
            (variant.result_key, variant.sub_config) in self._current_availability
            or (variant.result_key, "*") in self._current_availability
        )
        if not is_available:
            return  # no data -- do nothing

        # Activate this plot
        self._active_entry_iid = iid
        self._active_variant = letter
        self._viewer_settings.last_variant = letter
        self._viewer_settings.last_entry = entry.number
        self._file_index = 0
        self._clear_figure_cache()
        self._trigger_replot()

    def _move_tree_selection_from_canvas(
        self, direction: int, skip_disabled: bool = True,
    ) -> None:
        """Move tree selection up/down while keeping focus in canvas.

        When *skip_disabled* is True, disabled (greyed-out) entries are
        skipped.  When False, every visible entry is a valid stop
        (used by Shift+arrow to allow landing on blank/grey rows).
        """
        visible = self._get_visible_entries()
        selection = self._plot_tree.selection()
        if not selection or selection[0] not in visible:
            if visible:
                self._plot_tree.selection_set(visible[0])
            return
        idx = visible.index(selection[0])
        new_idx = idx + direction
        while 0 <= new_idx < len(visible):
            if not skip_disabled or not self._is_entry_disabled(visible[new_idx]):
                self._plot_tree.selection_set(visible[new_idx])
                self._plot_tree.see(visible[new_idx])
                return
            new_idx += direction

    def _move_to_next_available_row(self, direction: int) -> None:
        """Move to the next row that has any available data, then activate.

        If the focus variant (_focus_col) is available on the new row,
        activate it.  Otherwise find the nearest available variant on
        that row and activate that instead.  The dashed rectangle stays
        at _focus_col regardless.
        """
        visible = self._get_visible_entries()
        selection = self._plot_tree.selection()
        if not selection or selection[0] not in visible:
            return
        idx = visible.index(selection[0])
        new_idx = idx + direction

        while 0 <= new_idx < len(visible):
            iid = visible[new_idx]
            entry = self._tree_entry_map.get(iid)
            if entry:
                # Check if this row has any available variant
                available_letters = self._entry_available_letters(entry)
                if available_letters:
                    self._plot_tree.selection_set(iid)
                    self._plot_tree.see(iid)
                    # Try focus_col variant first
                    focus_letter = (
                        self._all_variant_letters[self._focus_col]
                        if 0 <= self._focus_col < len(self._all_variant_letters)
                        else ""
                    )
                    if focus_letter in available_letters:
                        self._active_variant = focus_letter
                    else:
                        self._active_variant = self._find_nearest_available(
                            available_letters
                        )
                    self._active_entry_iid = iid
                    self._viewer_settings.last_variant = self._active_variant
                    self._viewer_settings.last_entry = entry.number
                    self._file_index = 0
                    self._clear_figure_cache()
                    self._trigger_replot()
                    self._redraw_variant_grid()
                    return
            new_idx += direction

    def _entry_available_letters(self, entry: PlotEntry) -> set[str]:
        """Return the set of variant letters with data for an entry."""
        result: set[str] = set()
        for v in entry.variants:
            if (
                (v.result_key, v.sub_config) in self._current_availability
                or (v.result_key, "*") in self._current_availability
            ):
                result.add(v.letter)
        return result

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def _on_mode_changed(self) -> None:
        """Handle mode radio button change."""
        mode = self._mode.get()
        self._viewer_settings.last_mode = mode

        if mode == "comparison":
            # Show comparison checkboxes, hide scenario listbox.
            # Phase B: do NOT auto-rebuild based on the live tick state —
            # rebuilding the combined parquets is exclusively driven by
            # ``refresh_to_viewer_scenarios`` (i.e. the main window's
            # "Update view scenarios" button or cold-open reconciliation).
            # Mode switches inside the viewer are pure view changes.
            self._scenario_listbox.grid_remove()
            self._comp_outer_frame.grid()
            self._populate_comparison_checkboxes()
            self._populate_plot_tree()
            self._show_variant_panel()
        else:
            # Show scenario listbox, hide comparison checkboxes
            self._comp_outer_frame.grid_remove()
            self._scenario_listbox.grid()

            if mode == "single":
                self._scenario_listbox.configure(selectmode="browse")
                self._populate_plot_tree()
                self._show_variant_panel()
            elif mode == "dispatch":
                self._scenario_listbox.configure(selectmode="browse")
                self._hide_variant_panel()
                # Make sure a scenario is selected so the dispatch tree
                # and the first figure can actually render. Coming back
                # from comparison mode, the listbox may retain no
                # selection (grid_remove doesn't clear it, but _scan
                # changes may have emptied it).
                if (
                    not self._scenario_listbox.curselection()
                    and self._scenario_listbox.size() > 0
                ):
                    self._scenario_listbox.selection_set(0)
                    self._scenario_listbox.see(0)
                # Populate tree with nodeGroups instead of plot entries
                self._populate_dispatch_tree()
                # _populate_dispatch_tree's selection_set fires
                # <<TreeviewSelect>> but the event arrives after the
                # tree has also processed a clear (empty → first-item)
                # transition, which can swallow it. Render explicitly.
                self._render_first_dispatch_figure()
            elif mode == "network":
                self._scenario_listbox.configure(selectmode="browse")
                # Clear plot tree
                for item in self._plot_tree.get_children():
                    self._plot_tree.delete(item)
                self._tree_entry_map.clear()
                self._hide_variant_panel()
                self._render_network()

    # ------------------------------------------------------------------
    # File navigation
    # ------------------------------------------------------------------

    def _update_file_nav(self) -> None:
        """Update file navigation label and button states."""
        self._file_label.configure(
            text=f"File {self._file_index + 1}/{self._file_count}"
        )
        state = "normal" if self._file_count > 1 else "disabled"
        self._prev_file_btn.configure(state=state)
        self._next_file_btn.configure(state=state)

    def _on_focus_scenarios_event(self, event: tk.Event) -> str | None:  # type: ignore[type-arg]
        """Focus the scenarios area on 's' press (only in this window).

        In comparison mode the scenario listbox is hidden, so focus the
        first checkbox in the comparison frame instead.
        """
        try:
            if event.widget.winfo_toplevel() is not self:
                return None
            if isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Text)):
                return None
        except (tk.TclError, AttributeError):
            return None
        if self._mode.get() == "comparison":
            self._comp_tree.focus_set()
            children = self._comp_tree.get_children()
            if children and not self._comp_tree.selection():
                # Select first row for keyboard UX if nothing selected.
                self._comp_tree.selection_set(children[0])
                self._comp_tree.focus(children[0])
        else:
            self._scenario_listbox.focus_set()
        return "break"

    def _on_focus_plots_event(self, event: tk.Event) -> str | None:  # type: ignore[type-arg]
        """Focus the plot tree on 'p' press (only in this window)."""
        try:
            if event.widget.winfo_toplevel() is not self:
                return None
            if isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Text)):
                return None
        except (tk.TclError, AttributeError):
            return None
        self._plot_tree.focus_set()
        return "break"

    def _on_prev_file_event(self, event: tk.Event) -> str | None:  # type: ignore[type-arg]
        """Handle prev file key event — only if from this window."""
        try:
            if event.widget.winfo_toplevel() is not self:
                return None
        except (tk.TclError, AttributeError):
            return None
        # For Left/Right arrows, let the plot tree and scenario listbox
        # handle them natively (expand/collapse, scrolling)
        keysym = getattr(event, "keysym", "")
        if keysym in ("Left", "Right"):
            w = event.widget
            if isinstance(w, (ttk.Treeview, tk.Listbox)):
                return None
        self._on_prev_file()
        return "break"

    def _on_next_file_event(self, event: tk.Event) -> str | None:  # type: ignore[type-arg]
        """Handle next file key event — only if from this window."""
        try:
            if event.widget.winfo_toplevel() is not self:
                return None
        except (tk.TclError, AttributeError):
            return None
        # For Left/Right arrows, let the plot tree and scenario listbox
        # handle them natively (expand/collapse, scrolling)
        keysym = getattr(event, "keysym", "")
        if keysym in ("Left", "Right"):
            w = event.widget
            if isinstance(w, (ttk.Treeview, tk.Listbox)):
                return None
        self._on_next_file()
        return "break"

    def _on_prev_file(self) -> None:
        """Navigate to previous file."""
        if self._file_index > 0:
            self._file_index -= 1
            self._update_file_nav()
            self._trigger_replot()

    def _on_next_file(self) -> None:
        """Navigate to next file."""
        if self._file_index < self._file_count - 1:
            self._file_index += 1
            self._update_file_nav()
            self._trigger_replot()

    # ------------------------------------------------------------------
    # Time range controls
    # ------------------------------------------------------------------

    def _active_plot_settings(self) -> 'PlotSettings':  # type: ignore[name-defined]
        """Return the PlotSettings for the active mode (single vs comparison).

        ``dispatch`` and ``network`` modes also use the single settings as
        a sane fallback — they do not surface their own duration controls.
        """
        if self._mode.get() == "comparison":
            return self._settings.comparison_plot_settings
        return self._settings.single_plot_settings

    def _resolve_initial_duration(self, letter: str, data_length: int) -> int:
        """Compute the duration to show the FIRST time *letter* is viewed.

        Resolution order (short-circuits on the first match):

        1. If ``letter`` is already in ``settings.variant_durations`` →
           saved user intent wins, even on a fresh GUI session.
        2. Else if ``letter`` is in ``self._template_default_durations``:
              - value ``"all"`` → resolve to ``data_length`` (or 168 if
                ``data_length <= 0``);
              - integer value → use that integer.
           The resolved int is then persisted into
           ``settings.variant_durations[letter]`` and a settings save is
           triggered so the user's first sighting becomes the new baseline.
        3. Else fall back to 168, except for ``'w'`` which preserves the
           legacy "full timeline" behaviour when no template default exists.
        """
        plot_settings = self._active_plot_settings()
        saved = plot_settings.variant_durations
        if letter in saved:
            try:
                return int(saved[letter])
            except (TypeError, ValueError):
                pass

        template = self._template_default_durations
        if letter in template:
            tmpl_val = template[letter]
            if isinstance(tmpl_val, str) and tmpl_val.strip().lower() == "all":
                resolved = data_length if data_length > 0 else 168
            else:
                try:
                    resolved = int(tmpl_val)
                except (TypeError, ValueError):
                    resolved = 168
            saved[letter] = int(resolved)
            self._schedule_settings_save()
            return int(resolved)

        if letter == 'w':
            return data_length if data_length > 0 else 168
        return 168

    def _update_time_range(self, data_length: int) -> None:
        """Update the Start slider and Duration spinbox for *data_length* rows.

        Saves the current start under the previous variant letter and
        restores the values for the current variant letter. Durations are
        resolved against persisted settings (and template defaults on the
        first sighting); see :meth:`_resolve_initial_duration`.

        The spinbox value MAY be clamped down to ``data_length`` for
        display, but the persisted user intent in
        ``settings.variant_durations[letter]`` is *not* overwritten by
        that clamp — the ``_suppress_duration_save`` guard ensures the
        trace callback ignores the programmatic write.
        """
        self._updating_time_range = True
        plot_settings = self._active_plot_settings()
        try:
            # ── Save outgoing variant's start state ──
            prev = getattr(self, '_last_slider_variant', None)
            current = self._active_variant or 'h'
            if prev is not None and prev != current:
                self._variant_start_state[prev] = self._start_var.get()
                # Outgoing duration is already in plot_settings — last
                # user-driven write came through the trace handler and
                # was already persisted (or scheduled to be).

            # ── Restore or initialise incoming variant's slider state ──
            if current != prev:
                # Determine duration: settings → template → fallback.
                if current in plot_settings.variant_durations:
                    saved_dur = int(plot_settings.variant_durations[current])
                else:
                    saved_dur = self._resolve_initial_duration(current, data_length)
                # Determine start: session-local dict, default 0.
                saved_start = self._variant_start_state.get(current, 0)

                self._suppress_duration_save = True
                try:
                    self._duration_var.set(int(saved_dur))
                finally:
                    self._suppress_duration_save = False
                self._start_var.set(saved_start)
            self._last_slider_variant = current

            # The user's chosen duration is preserved as-is regardless
            # of how short the current scenario's data is — the renderer
            # simply leaves empty space on the right when data ends
            # before the chosen duration. This keeps the time axis
            # stable when the user toggles between scenarios with
            # different model horizons.
            self._duration_var.get()

            # Spinbox values include the standard steps (always; we no
            # longer trim to <= data_length so the user can dial up to
            # whatever they want).
            self._duration_spin.configure(
                values=tuple(self._duration_steps), state="normal",
            )

            # Start slider ranges over the full data_length so the user
            # can scroll through the actual data window.
            max_start = max(0, data_length - 1)
            self._start_scale.configure(to=max(max_start, 1))
            if max_start <= 0:
                self._start_var.set(0)
                self._start_scale.configure(state="disabled")
            else:
                self._start_scale.configure(state="normal")
                if self._start_var.get() > max_start:
                    self._start_var.set(max_start)
        finally:
            self._updating_time_range = False

    def _on_change_colors(self) -> None:
        """Open the per-project colors editor and re-render on save.

        Always edits the PROJECT's ``plot_settings.yaml`` (seeding it from
        the bundled default if absent), never the bundled package file.
        On a successful save the color-template cache is cleared and the
        currently displayed plot is re-rendered with the new colors — the
        already-loaded parquet/dataframe is reused (no reload).
        """
        from flextool.gui.dialogs.plot_colors_editor import PlotColorsEditor
        from flextool.gui.project_utils import seed_plot_settings

        # Edit the project copy; seed from the bundled default if missing.
        project_file = seed_plot_settings(self._project_path)

        editor = PlotColorsEditor(self, project_file)
        if not editor.saved:
            return

        # Colors changed → the cached template and any colored figures are
        # stale.  Clear the template cache so the edited file is re-read,
        # invalidate the cached live plan (forces compute_live_plan, which
        # reads color_path, to rebuild shared_color_map) and any prefetched
        # figures, then re-render the current plot from the already-loaded
        # dataframe.
        from flextool.plot_outputs.color_template import _clear_cache
        _clear_cache()
        self._clear_figure_cache()
        self._trigger_replot()

    def _on_time_range_changed(self, *_args) -> None:
        """Handle Start or Duration change.

        Persists the *unclamped* user intent into
        ``settings.variant_durations`` and triggers a debounced settings
        save (500 ms). Programmatic clamps inside ``_update_time_range``
        bypass this path via ``_suppress_duration_save``.
        """
        if self._updating_time_range:
            return
        if not self._suppress_duration_save:
            letter = self._active_variant or self._last_slider_variant
            if letter:
                try:
                    intent = int(self._duration_var.get())
                except tk.TclError:
                    intent = None
                if intent is not None and intent > 0:
                    plot_settings = self._active_plot_settings()
                    plot_settings.variant_durations[letter] = intent
                    self._schedule_settings_save()
        self._file_index = 0
        self._clear_prefetched_figures()
        self._trigger_replot()

    def _schedule_settings_save(self, delay_ms: int = 500) -> None:
        """Debounced ``save_project_settings`` — coalesce keystroke bursts.

        Cancels any previously scheduled save and schedules a new one
        ``delay_ms`` from now. Saves run on the Tk main thread.
        """
        if self._duration_save_after_id is not None:
            try:
                self.after_cancel(self._duration_save_after_id)
            except (tk.TclError, ValueError):
                pass
            self._duration_save_after_id = None
        try:
            self._duration_save_after_id = self.after(
                delay_ms, self._flush_settings_save,
            )
        except tk.TclError:
            # Window may already be destroyed — fall back to immediate save.
            self._flush_settings_save()

    def _flush_settings_save(self) -> None:
        """Persist project settings now and clear the pending after() id."""
        self._duration_save_after_id = None
        try:
            save_project_settings(self._project_path, self._settings)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to persist variant durations", exc_info=True)

    # ------------------------------------------------------------------
    # Figure cache management
    # ------------------------------------------------------------------

    def _clear_figure_cache(self) -> None:
        """Discard all prefetched figures AND the live plan.

        Called on structural changes (entry/variant/scenario).  For slider
        changes use :meth:`_clear_prefetched_figures` instead so the live
        plan (which is slider-independent) is preserved.
        """
        with self._figure_cache_lock:
            self._figure_cache.clear()
        self._live_plan = None
        self._live_plan_key = ("", "", "")

    def _clear_prefetched_figures(self) -> None:
        """Discard prefetched figures only, keep the live plan."""
        with self._figure_cache_lock:
            self._figure_cache.clear()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def _on_update(self) -> None:
        """Backward-compatible alias for :meth:`refresh_to_viewer_scenarios`.

        The Phase B contract is that the main window's
        "Update view scenarios" button is the only path that rebuilds the
        combined comparison parquets, and the main window now calls
        :meth:`refresh_to_viewer_scenarios` directly with an explicit
        desired set.  This shim remains so any older call sites or
        diagnostic tools that still reference ``_on_update`` keep working
        — it derives ``desired`` from the persisted main-window-checked
        list.
        """
        from flextool.gui.scenario_key import (
            parse_key, resolve_subdir_for_read,
        )
        bare_owners = self._settings.bare_output_owners
        desired: list[str] = []
        for entry in self._settings.checked_executed_scenarios:
            try:
                src_num, scen_name = parse_key(entry)
            except ValueError:
                continue
            desired.append(
                resolve_subdir_for_read(bare_owners, src_num, scen_name)
            )
        self.refresh_to_viewer_scenarios(desired)

    def _read_metadata_scenarios(self) -> list[str]:
        """Return the scenario list recorded in ``_metadata.json``.

        Empty list when the file is missing, unreadable, or doesn't
        contain a ``scenarios`` array.  This is the canonical record of
        the *viewer scenarios* set (the scenarios materialised into the
        combined comparison parquets).
        """
        import json as _json
        meta_path = (
            self._project_path
            / "output_parquet_comparison"
            / "_metadata.json"
        )
        if not meta_path.is_file():
            return []
        try:
            with open(meta_path) as f:
                payload = _json.load(f)
        except (OSError, _json.JSONDecodeError):
            return []
        scenarios = payload.get("scenarios")
        if not isinstance(scenarios, list):
            return []
        return [str(s) for s in scenarios if isinstance(s, str)]

    def refresh_to_viewer_scenarios(self, desired: list[str]) -> None:
        """Reconcile the on-disk viewer-scenarios set with *desired*.

        Phase B entry point invoked by the main window's
        "Update view scenarios" button (and the cold-open path inside
        :meth:`__init__`).  Compares *desired* against the scenarios
        recorded in ``output_parquet_comparison/_metadata.json``:

        * **Differs:** schedule *desired* as the next combine target and
          kick off :meth:`_regenerate_comparison`.  When that finishes,
          ``_metadata.json`` is rewritten and the comparison tree
          repopulates so the viewer reflects the new viewer scenarios.
        * **Matches:** no rebuild needed; just refresh the plot
          tree + figure so any stale availability is recomputed.
        """
        self.config(cursor="watch")
        try:
            self.update_idletasks()
        except tk.TclError:
            pass

        # Clear caches that depend on scenario contents — both the
        # rebuild-and-no-rebuild paths benefit from a fresh slate.
        self._yaml_cache.clear()
        self._break_times_cache.clear()
        self._current_availability = set()
        self._clear_figure_cache()
        self._dispatch_scenario = ""
        self._dispatch_mappings = None
        self._dispatch_results = None
        self._dispatch_ylims.clear()
        self._dispatch_columns.clear()
        if hasattr(self, '_dispatch_metadata_cache'):
            del self._dispatch_metadata_cache
        self._plot_canvas._cache.clear()
        self._parquet_cache_key = ("", "")
        self._parquet_cache_df = None
        # Force the shared axis-bounds manifest to reload on the next
        # render — a batch run may have just rewritten it.
        self._axis_manifest = None
        self._axis_manifest_mtime = 0.0

        self._populate_scenarios()
        if self._mode.get() == "comparison":
            self._populate_comparison_checkboxes()

        current_meta = set(self._read_metadata_scenarios())
        desired_set = set(desired)
        if current_meta != desired_set:
            # Schedule the rebuild for *exactly* the desired set, then
            # kick off the combine.  ``_on_comparison_ready`` will
            # repopulate the plot tree once the parquets land.
            self._scheduled_viewer_scenarios = list(desired)
            self._settings.scenarios_changed = False
            self._comp_needs_regen = True
            self._regenerate_comparison(list(desired))
        else:
            # Sets match — no rebuild, just refresh availability + plot.
            self._comp_needs_regen = False
            self._on_mode_changed()

        try:
            self.config(cursor="")
        except tk.TclError:
            pass

    # ------------------------------------------------------------------
    # Tab focus cycling
    # ------------------------------------------------------------------

    def _focus_plot_tree(self, _event: tk.Event | None = None) -> str:
        """Move focus to the plot tree."""
        self._plot_tree.focus_set()
        # Ensure something is selected
        if not self._plot_tree.selection():
            self._restore_or_select_first_entry()
        return "break"

    def _focus_variant_canvas(self, _event: tk.Event | None = None) -> str:
        """Move focus to the variant canvas."""
        if self._all_variant_letters:
            self._variant_canvas.focus_set()
        return "break"

    def _focus_scenario_listbox(self, _event: tk.Event | None = None) -> str:
        """Move focus back to the scenario listbox (or comparison tree in comparison mode)."""
        if self._mode.get() == "comparison":
            self._comp_tree.focus_set()
            children = self._comp_tree.get_children()
            if children and not self._comp_tree.selection():
                self._comp_tree.selection_set(children[0])
                self._comp_tree.focus(children[0])
        else:
            self._scenario_listbox.focus_set()
        return "break"

    # ------------------------------------------------------------------
    # Plot display
    # ------------------------------------------------------------------

    def _get_active_variant(self, entry: PlotEntry) -> PlotVariant | None:
        """Return the PlotVariant matching the active variant letter."""
        for v in entry.variants:
            if v.letter == self._active_variant:
                return v
        # Fall back to first variant
        return entry.variants[0] if entry.variants else None

    def _load_single_plot_config(
        self, result_key: str, sub_config: str,
    ) -> PlotConfig | None:
        """Load PlotConfig from the **single-mode** YAML regardless of current mode.

        Used by the merged-config comparison path: when a single-mode
        config carries ``scenario_rule``, comparison view derives the
        comparison config from it via
        :func:`flextool.scenario_comparison.plan_union.derive_comparison_config`.
        """
        return self._load_plot_config_from(
            self._single_config_path, result_key, sub_config,
        )

    def _load_plot_config_from(
        self, config_path: Path, result_key: str, sub_config: str,
    ) -> PlotConfig | None:
        """Load a PlotConfig from a specific YAML file path."""
        if config_path not in self._yaml_cache:
            if not config_path.is_file():
                return None
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            except (yaml.YAMLError, OSError) as exc:
                logger.error("Failed to read plot config %s: %s", config_path, exc)
                return None
            if not isinstance(data, dict):
                return None
            self._yaml_cache[config_path] = data
        data = self._yaml_cache[config_path]
        plots = data.get("plots")
        if not isinstance(plots, dict):
            return None
        plots = flatten_new_format(plots)
        entry = plots.get(result_key)
        if not isinstance(entry, dict):
            return None
        if _is_single_config(entry):
            if sub_config != "default":
                return None
            raw = entry
        else:
            raw = entry.get(sub_config)
            if not isinstance(raw, dict):
                return None
        unknown_keys = [k for k in raw if k not in PLOT_FIELD_NAMES]
        if unknown_keys:
            logger.debug(
                "Plot config '%s': ignoring unknown setting(s): %s",
                result_key, ", ".join(repr(k) for k in unknown_keys),
            )
        filtered = {k: v for k, v in raw.items() if k in PLOT_FIELD_NAMES}
        if "axis_scale_min_max" in filtered and "axis_bounds" not in filtered:
            filtered["axis_bounds"] = filtered.pop("axis_scale_min_max")
        elif "axis_scale_min_max" in filtered:
            del filtered["axis_scale_min_max"]
        filtered.pop("variant", None)
        try:
            return PlotConfig(**filtered)
        except TypeError as exc:
            logger.error("Failed to create PlotConfig for '%s': %s", result_key, exc)
            return None

    def _load_plot_config(self, result_key: str, sub_config: str) -> PlotConfig | None:
        """Load PlotConfig for a result_key from the active YAML config file."""
        config_path = self._get_config_path_for_mode()

        # Use cached YAML if available
        if config_path not in self._yaml_cache:
            if not config_path.is_file():
                return None
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            except (yaml.YAMLError, OSError) as exc:
                logger.error("Failed to read plot config %s: %s", config_path, exc)
                return None
            if not isinstance(data, dict):
                return None
            self._yaml_cache[config_path] = data

        data = self._yaml_cache[config_path]
        plots = data.get("plots")
        if not isinstance(plots, dict):
            return None

        # Flatten new-format entries (entry-name grouping) to flat result_key mapping
        plots = flatten_new_format(plots)

        entry = plots.get(result_key)
        if not isinstance(entry, dict):
            return None

        # Determine which raw config dict to use
        if _is_single_config(entry):
            # Direct config: the entry dict IS the plot settings
            if sub_config != "default":
                return None
            raw = entry
        else:
            # Named-config dict: look up the sub_config key
            raw = entry.get(sub_config)
            if not isinstance(raw, dict):
                return None

        # Filter unknown keys and handle backward-compat alias (same as orchestrator)
        unknown_keys = [k for k in raw if k not in PLOT_FIELD_NAMES]
        if unknown_keys:
            logger.debug(
                "Plot config '%s': ignoring unknown setting(s): %s",
                result_key, ", ".join(repr(k) for k in unknown_keys),
            )
        filtered = {k: v for k, v in raw.items() if k in PLOT_FIELD_NAMES}
        if "axis_scale_min_max" in filtered and "axis_bounds" not in filtered:
            filtered["axis_bounds"] = filtered.pop("axis_scale_min_max")
        elif "axis_scale_min_max" in filtered:
            del filtered["axis_scale_min_max"]
        filtered.pop("variant", None)

        try:
            return PlotConfig(**filtered)
        except TypeError as exc:
            logger.error("Failed to create PlotConfig for '%s': %s", result_key, exc)
            return None

    def _get_axis_manifest(self) -> dict | None:
        """Return the cross-scenario axis-bounds manifest (mtime-cached).

        Returns ``None`` when no manifest is available; never raises.
        The on-disk manifest is ``stat``ed on every call: if its mtime is
        newer than the cached copy, the file is re-read.  This lets the
        viewer pick up manifest rewrites from post-run hooks without any
        explicit invalidation.
        """
        manifest_path = (
            self._project_path
            / "output_parquet"
            / "_axis_bounds.json"
        )
        try:
            mtime = manifest_path.stat().st_mtime if manifest_path.is_file() else 0.0
        except OSError:
            mtime = 0.0

        if mtime == 0.0:
            # File missing — drop any previous cached view.
            if self._axis_manifest is not None:
                self._axis_manifest = None
            self._axis_manifest_mtime = 0.0
            return None

        if mtime <= self._axis_manifest_mtime and self._axis_manifest is not None:
            return self._axis_manifest

        try:
            from flextool.plot_outputs.shared_manifest import (
                load_axis_bounds_manifest,
            )
            self._axis_manifest = load_axis_bounds_manifest(
                self._project_path
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed to load shared axis manifest", exc_info=True,
            )
            self._axis_manifest = None
        self._axis_manifest_mtime = mtime
        return self._axis_manifest

    def _get_axis_active_scenarios(self) -> set[str] | None:
        """Return the scenarios that should contribute to the shared y-axis.

        - In *single* mode the active set is the list of currently-checked
          executed scenarios (from :meth:`_scan_scenarios`).  That list
          reflects the user's own statement of which scenarios are
          relevant for comparison, so it's the natural filter for the
          shared axis — and it updates the moment a checkbox is toggled,
          without needing any file rewrite.
        - In *comparison* mode we return the **viewer scenarios** set —
          the scenarios locked in at the last "Update view scenarios"
          press (i.e. the union currently materialised into the combined
          parquets).  Using this set rather than the currently-ticked
          subset in ``_comp_tree`` is what *freezes* the y-axis as the
          user toggles individual scenarios on/off.  See
          :meth:`_get_comparison_viewer_scenarios`.
        - In *network* / *dispatch* modes the override is not applicable;
          the caller skips it entirely so the return value is irrelevant.

        Returns ``None`` in unsupported modes / on internal errors so the
        caller can early-out rather than forwarding ``None`` to
        :func:`apply_manifest_to_plan` (which would silently union over
        every scenario in the manifest).
        """
        mode = self._mode.get()
        if mode == "single":
            try:
                scenarios = self._scan_scenarios()
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Failed to collect active scenarios for axis manifest",
                    exc_info=True,
                )
                return None
            return set(scenarios)
        if mode == "comparison":
            try:
                return set(self._get_comparison_viewer_scenarios())
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Failed to collect viewer scenarios for axis manifest",
                    exc_info=True,
                )
                return None
        return None

    def _get_comparison_viewer_scenarios(self) -> list[str]:
        """Return the *viewer scenarios* set for comparison mode.

        These are the scenarios locked in at the last "Update view
        scenarios" press — i.e. the set whose results are currently
        materialised into ``output_parquet_comparison/*.parquet``.  The
        canonical record is the ``scenarios`` list in
        ``output_parquet_comparison/_metadata.json``; we fall back to
        ``settings.comp_viewer_scenarios`` (which tracks the live tree
        state) only when the metadata is missing or unreadable, since
        that's the closest available approximation on a cold open.

        This set is **not** the currently-ticked subset in ``_comp_tree``
        — that's the *current scenarios*, which the caller toggles freely
        without changing the axis scope.
        """
        import json as _json

        meta_path = (
            self._project_path
            / "output_parquet_comparison"
            / "_metadata.json"
        )
        if meta_path.is_file():
            try:
                with open(meta_path) as f:
                    payload = _json.load(f)
                scenarios = payload.get("scenarios")
                if isinstance(scenarios, list):
                    return [str(s) for s in scenarios if isinstance(s, str)]
            except (OSError, _json.JSONDecodeError):
                logger.warning(
                    "Could not read comparison metadata at %s",
                    meta_path, exc_info=True,
                )
        # Fall back to the persisted viewer-tree state.
        return list(self._settings.comp_viewer_scenarios)

    def _apply_axis_manifest(
        self, plan, result_key: str, sub_config: str,
    ) -> None:
        """Override a plan's subplot_y_ranges from the shared manifest.

        No-op when the manifest is missing or has no matching entry.
        Safe to call on every render — the override is a cheap dict
        lookup.  Mutates the plan in place before handing it to
        ``build_figure_from_plan`` so the call site doesn't need a new
        parameter.

        In *single* mode the override is filtered to the set of currently
        checked executed scenarios: as the user toggles checkboxes the
        next render naturally picks up a new union.

        In *comparison* mode the override is filtered to the **viewer
        scenarios** set (the scenarios locked in at the last "Update view
        scenarios" press — see :meth:`_get_comparison_viewer_scenarios`).
        That keeps the y-axis frozen while the user toggles individual
        scenarios on/off in the comparison tree: ticking changes
        visibility only, never axes.

        If :meth:`_get_axis_active_scenarios` returns ``None`` (network /
        dispatch mode, or an exception inside the helper) we skip the
        override entirely rather than forwarding ``None`` to
        :func:`apply_manifest_to_plan` — the latter treats ``None`` as
        "union over every scenario in the manifest", which would silently
        defeat the scoped filter.
        """
        if plan is None:
            return
        manifest = self._get_axis_manifest()
        if manifest is None:
            return
        active = self._get_axis_active_scenarios()
        if active is None:
            # Non-supported mode (or active-set lookup failed): the
            # caller shouldn't be using the shared-manifest override at
            # all.  See the docstring above for why we can't pass
            # ``None`` on through.
            return
        try:
            from flextool.plot_outputs.shared_manifest import (
                apply_manifest_to_plan,
            )
            apply_manifest_to_plan(
                plan, manifest, result_key, sub_config,
                scenarios=active,
            )
        except Exception:  # noqa: BLE001
            # Never let manifest application break the viewer — the
            # fallback is the plan's own per-scenario y-ranges.
            logger.warning(
                "Failed to apply axis manifest for %s/%s",
                result_key, sub_config, exc_info=True,
            )

    def _load_parquet(self, scenario: str, result_key: str) -> pd.DataFrame | None:
        """Load a parquet file for the given scenario and result_key.

        Per-scenario parquets have 'scenario' as the top column MultiIndex
        level (added at write time).  Strip it so the DataFrame matches the
        dimension rules in the plot config.  Caches the last loaded DataFrame.
        """
        cache_key = (scenario, result_key)
        if cache_key == self._parquet_cache_key and self._parquet_cache_df is not None:
            return self._parquet_cache_df
        path = self._project_path / "output_parquet" / scenario / f"{result_key}.parquet"
        if not path.exists():
            return None
        df = read_lean_parquet(path)
        if isinstance(df.columns, pd.MultiIndex) and 'scenario' in df.columns.names:
            df = df.droplevel('scenario', axis=1)
        self._parquet_cache_key = cache_key
        self._parquet_cache_df = df
        return df

    def _load_break_times(self, scenario: str) -> set[str] | None:
        """Load timeline break times from parquet, cached per scenario."""
        if scenario in self._break_times_cache:
            return self._break_times_cache[scenario]

        path = self._project_path / "output_parquet" / scenario / "timeline_breaks.parquet"
        if not path.exists():
            self._break_times_cache[scenario] = None
            return None

        try:
            df = read_lean_parquet(path)
            # Extract break time values as strings
            if df.empty:
                result: set[str] | None = None
            else:
                # The parquet has a column with break time values
                result = set(df.iloc[:, 0].astype(str))
            self._break_times_cache[scenario] = result
            return result
        except Exception:  # noqa: BLE001
            logger.warning("Failed to read timeline breaks for %s", scenario, exc_info=True)
            self._break_times_cache[scenario] = None
            return None

    def _make_figure_cache_key(
        self, scenario: str, result_key: str, sub_config: str,
        file_index: int, start: int, duration: int,
    ) -> tuple:
        return (scenario, result_key, sub_config, file_index, start, duration)

    def _display_from_parquet(self, scenario: str, entry: PlotEntry, variant: PlotVariant) -> None:
        """Load parquet, build PlotConfig, render Figure, display it.

        If a pre-computed PlotPlan exists, use it for instant rendering.
        If the figure is already cached (from prefetch), display instantly.
        Otherwise submit building to a background thread and display on
        completion.
        """
        # 1. Load parquet (cached) and config (cached) — synchronous, fast
        df = self._load_parquet(scenario, variant.result_key)
        if df is None:
            self._plot_canvas.show_message(f"No data: {variant.result_key}.parquet")
            return

        config = self._load_plot_config(variant.result_key, variant.sub_config)
        if config is None:
            self._plot_canvas.show_message(f"No config for {variant.result_key}")
            return

        break_times = self._load_break_times(scenario)
        plot_name = config.plot_name or variant.full_name

        # 1b. Try live plan (cached in memory) or disk plan for instant rendering.
        #     The plan caches dimension rules, layout, and colors; only the
        #     time slice and matplotlib rendering run on each slider change.
        try:
            from flextool.plot_outputs.plan import (
                load_plot_plan, compute_live_plan,
            )
            plan_key = (scenario, variant.result_key, variant.sub_config)
            if self._live_plan_key == plan_key and self._live_plan is not None:
                plan = self._live_plan
            else:
                # Try disk plan first, then compute on-the-fly
                plan_dir = self._project_path / "output_parquet" / scenario / "plot_plans"
                plan = load_plot_plan(plan_dir, variant.result_key, variant.sub_config)
                if plan is None:
                    plan = compute_live_plan(
                        df, config, plot_name, break_times,
                        color_path=resolve_plot_settings_path(self._project_path),
                    )
                self._live_plan = plan
                self._live_plan_key = plan_key

            if plan is not None:
                # Apply cross-scenario axis override every render (cheap
                # dict lookup).  This handles the case where the manifest
                # finished loading AFTER the plan was cached as well as
                # the normal freshly-loaded plan.
                self._apply_axis_manifest(
                    plan, variant.result_key, variant.sub_config,
                )
                # Use the plan's processed_df length for the slider range
                # (aggregated/weekly plots have a shorter processed_df)
                self._update_time_range(len(plan.processed_df))
                start = self._start_var.get()
                duration = self._duration_var.get()
                plot_rows = (start, start + duration)
                self._file_count = plan.total_file_count
                self._file_index = min(self._file_index, max(0, self._file_count - 1))
                self._update_file_nav()
                # build_figure_from_plan can be slow for plots with many
                # bars / ticks (matplotlib does the heavy lifting). Run it
                # on the background executor and display when ready so the
                # UI thread stays responsive.
                self._render_gen += 1
                gen = self._render_gen
                self._schedule_placeholder(gen, f"Rendering {variant.full_name}…")
                self._executor.submit(
                    self._build_figure_from_plan_async,
                    gen, plan, self._file_index, plot_rows, variant,
                )
                return
        except Exception:
            logger.warning("Plan path failed for %s", variant.result_key, exc_info=True)

        # Fallback: use raw data length for slider range
        self._update_time_range(len(df))
        start = self._start_var.get()
        duration = self._duration_var.get()

        # 2. Check prefetch cache for instant display
        cache_key = self._make_figure_cache_key(
            scenario, variant.result_key, variant.sub_config,
            self._file_index, start, duration,
        )
        with self._figure_cache_lock:
            cached_fig = self._figure_cache.pop(cache_key, None)

        if cached_fig is not None:
            self._cancel_placeholder()
            self._plot_canvas.display_figure(cached_fig)
            logger.info("Plot %s: CACHED [file %d]", variant.result_key, self._file_index)
            self._prefetch_adjacent(scenario, variant, df, config, plot_name, break_times, start, duration)
            return

        # 3. Invalidate stale in-flight builds
        self._render_gen += 1
        gen = self._render_gen
        self._schedule_placeholder(gen, f"Rendering {variant.full_name}…")

        # 4. Submit build to background thread
        self._executor.submit(
            self._build_figure_async, gen, df, config, plot_name,
            plot_rows, break_times, self._file_index,
            scenario, variant, start, duration,
        )

    def _build_figure_from_plan_async(
        self, generation: int, plan, file_index, plot_rows, variant,
    ) -> None:
        """Run plan-based figure building on the background executor.

        Mirrors :meth:`_build_figure_async` but for the precomputed-plan
        path. The plan-path is "instant" for most plots, but bar plots
        with hundreds of items per file (matplotlib's barh + per-tick
        formatter loop) can take tens of seconds — long enough for the
        UI to freeze if we run it on the Tk main thread.
        """
        from flextool.plot_outputs.plan import build_figure_from_plan
        t0 = time.perf_counter()
        try:
            fig = build_figure_from_plan(plan, file_index, plot_rows)
        except Exception as exc:
            self.after(0, self._on_figure_error, generation, str(exc), variant.result_key)
            return
        t1 = time.perf_counter()
        logger.info(
            "Plot %s: from plan build=%.0fms [file %d/%d, bg thread]",
            variant.result_key, (t1 - t0) * 1000,
            file_index, plan.total_file_count,
        )
        self.after(0, self._on_plan_figure_ready, generation, fig, variant)

    def _on_plan_figure_ready(self, generation: int, fig, variant) -> None:
        """Main-thread display callback for a plan-path build."""
        if generation != self._render_gen:
            return  # stale — user moved on
        self._cancel_placeholder()
        if fig is not None:
            self._plot_canvas.display_figure(fig)
        else:
            self._plot_canvas.show_message(f"No plottable data for {variant.full_name}")

    def _build_figure_async(
        self, generation: int, df, config, plot_name, plot_rows,
        break_times, file_index, scenario, variant, start, duration,
    ) -> None:
        """Run in background thread: build figure, then schedule display on main thread."""
        t0 = time.perf_counter()
        try:
            figures, total_count = prepare_plot_data(
                df, config, plot_name, plot_rows, break_times,
                only_file_index=file_index,
                color_path=resolve_plot_settings_path(self._project_path),
            )
        except Exception as exc:
            self.after(0, self._on_figure_error, generation, str(exc), variant.result_key)
            return
        t1 = time.perf_counter()

        fig = figures[0][1] if figures else None
        logger.info(
            "Plot %s: build=%.0fms [file %d/%d, bg thread]",
            variant.result_key, (t1 - t0) * 1000, file_index, total_count,
        )

        self.after(0, self._on_figure_ready, generation, fig, total_count,
                   scenario, variant, start, duration)

    def _on_figure_ready(
        self, generation: int, fig, total_count: int,
        scenario, variant, start, duration,
    ) -> None:
        """Main-thread callback: display figure if still current."""
        if generation != self._render_gen:
            return  # stale result — figure will be garbage-collected

        self._cancel_placeholder()
        self._file_count = max(total_count, 1)
        self._file_index = min(self._file_index, max(0, self._file_count - 1))
        self._update_file_nav()

        if fig is not None:
            t0 = time.perf_counter()
            self._plot_canvas.display_figure(fig)
            logger.info("Plot %s: display=%.0fms", variant.result_key, (time.perf_counter() - t0) * 1000)
            # Prefetch adjacent pages
            df = self._load_parquet(scenario, variant.result_key)
            config = self._load_plot_config(variant.result_key, variant.sub_config)
            if df is not None and config is not None:
                break_times = self._load_break_times(scenario)
                plot_name = config.plot_name or variant.full_name
                self._prefetch_adjacent(scenario, variant, df, config, plot_name, break_times, start, duration)
        else:
            self._plot_canvas.show_message(f"No plottable data for {variant.full_name}")

    def _on_figure_error(self, generation: int, error_msg: str, result_key: str) -> None:
        """Main-thread callback: show error if still current."""
        if generation != self._render_gen:
            return
        self._cancel_placeholder()
        logger.error("prepare_plot_data failed for '%s': %s", result_key, error_msg)
        self._plot_canvas.show_message(f"Plot error: {error_msg}")

    # ── Deferred "Rendering…" placeholder ────────────────────────────────
    # Showing the placeholder text immediately on every plot switch wipes the
    # current figure, which makes flipping back and forth to spot differences
    # jarring. Instead we keep the previous figure up and only fall back to
    # text if the new render hasn't arrived within this delay.
    _PLACEHOLDER_DELAY_MS = 300

    def _schedule_placeholder(self, generation: int, text: str) -> None:
        """Show *text* on the canvas, but only after _PLACEHOLDER_DELAY_MS.

        Cancels any previously scheduled placeholder first. The fire is gated
        on *generation* so a stale timer never overwrites a newer render.
        """
        self._cancel_placeholder()

        def _fire() -> None:
            self._placeholder_after_id = None
            if generation != self._render_gen:
                return
            try:
                self._plot_canvas.show_message(text)
            except tk.TclError:
                pass  # viewer closed mid-flight

        self._placeholder_after_id = self.after(self._PLACEHOLDER_DELAY_MS, _fire)

    def _cancel_placeholder(self) -> None:
        """Cancel a pending deferred placeholder, if any."""
        if self._placeholder_after_id is not None:
            try:
                self.after_cancel(self._placeholder_after_id)
            except (tk.TclError, ValueError):
                pass
            self._placeholder_after_id = None

    def _prefetch_adjacent(
        self, scenario, variant, df, config, plot_name, break_times, start, duration,
    ) -> None:
        """Submit background builds for adjacent file pages."""
        plot_rows = (start, start + duration)
        for adj_offset in (1, -1):
            adj_index = self._file_index + adj_offset
            if adj_index < 0 or adj_index >= self._file_count:
                continue
            cache_key = self._make_figure_cache_key(
                scenario, variant.result_key, variant.sub_config,
                adj_index, start, duration,
            )
            with self._figure_cache_lock:
                if cache_key in self._figure_cache:
                    continue
            self._executor.submit(
                self._prefetch_build, cache_key, df, config, plot_name,
                plot_rows, break_times, adj_index, variant.result_key,
            )

    def _prefetch_build(self, cache_key, df, config, plot_name, plot_rows, break_times, file_index, result_key) -> None:
        """Background thread: build and cache a figure for prefetch."""
        try:
            figures, _ = prepare_plot_data(
                df, config, plot_name, plot_rows, break_times,
                only_file_index=file_index,
            )
            if figures:
                with self._figure_cache_lock:
                    self._figure_cache[cache_key] = figures[0][1]
                logger.info("Prefetch %s: file %d ready", result_key, file_index)
        except Exception:
            pass  # prefetch failure is not critical

    def _trigger_replot(self) -> None:
        """Called when scenario, entry, or variant changes."""
        scenarios = self._get_selected_scenarios()
        if not scenarios:
            self._plot_canvas.show_message("No scenario selected")
            return

        mode = self._mode.get()

        # Dispatch mode is handled directly by _on_tree_selected
        if mode == "dispatch":
            selection = self._plot_tree.selection()
            if selection and selection[0].startswith("dispatch_"):
                node_group = selection[0][len("dispatch_"):]
                self._display_dispatch(scenarios[0], node_group)
            return

        selection = self._plot_tree.selection()
        if not selection or not selection[0].startswith("entry_"):
            self._plot_canvas.show_message("No plot selected")
            return

        entry = self._tree_entry_map.get(selection[0])
        if not entry:
            return

        variant = self._get_active_variant(entry)
        if not variant:
            self._plot_canvas.show_message("No variant selected")
            return

        if mode == "single":
            self._display_from_parquet(scenarios[0], entry, variant)
        elif mode == "comparison":
            self._display_comparison(entry, variant)

    # ------------------------------------------------------------------
    # Comparison mode
    # ------------------------------------------------------------------

    def _display_comparison_from_single(
        self,
        entry: PlotEntry,
        variant: PlotVariant,
        single_cfg: PlotConfig,
    ) -> None:
        """Render comparison via merged-config (prototype) path.

        The single-mode config carries ``scenario_rule`` (and optionally
        ``comparison_overrides``).  We:

        1. Read each viewer scenario's **raw** result parquet and union
           them with ``scenario`` as the outermost col-MultiIndex level.
        2. Derive a comparison-mode PlotConfig by prepending ``s`` to the
           column part of ``index_types`` and ``scenario_rule`` to the
           column part of ``rules``, then merging in
           ``comparison_overrides``.
        3. Run ``compute_live_plan`` against the unioned raw frame with
           the derived config and render.

        No per-scenario plan parquet is read on this path — the dim-rule
        pivot logic is applied directly to the raw union, which is the
        only shape the augmented config knows how to interpret.
        """
        from flextool.scenario_comparison.plan_union import (
            derive_comparison_config, union_raw_data,
        )
        from flextool.plot_outputs.plan import (
            build_figure_from_plan, compute_live_plan,
        )

        viewer_scenarios = self._get_comparison_viewer_scenarios()
        checked = set(self._get_comparison_scenarios())

        df = union_raw_data(
            self._project_path, list(viewer_scenarios), variant.result_key,
        )
        if df is None:
            self._plot_canvas.show_message(
                f"No raw parquet for {variant.result_key}\n"
                f"in viewer scenarios."
            )
            return

        # Filter to currently-ticked scenarios (visibility toggle).
        if (
            checked
            and isinstance(df.columns, pd.MultiIndex)
            and "scenario" in df.columns.names
        ):
            mask = df.columns.get_level_values("scenario").isin(checked)
            df = df.loc[:, mask]

        if df.empty:
            self._plot_canvas.show_message(f"Empty data for {variant.result_key}")
            return

        try:
            cmp_cfg = derive_comparison_config(single_cfg)
        except (ValueError, TypeError) as exc:
            logger.warning(
                "derive_comparison_config failed for %s: %s",
                variant.result_key, exc,
            )
            self._plot_canvas.show_message(
                f"Cannot derive comparison config for {variant.result_key}\n"
                f"({exc})"
            )
            return

        break_times = self._load_comparison_break_times(viewer_scenarios)
        plot_name = cmp_cfg.plot_name or variant.full_name

        plan_key = (
            "_comparison_merged", variant.result_key, variant.sub_config,
            tuple(sorted(checked)),
        )
        if self._live_plan_key == plan_key and self._live_plan is not None:
            plan = self._live_plan
        else:
            plan = compute_live_plan(
                df, cmp_cfg, plot_name, break_times,
                color_path=resolve_plot_settings_path(self._project_path),
            )
            self._live_plan = plan
            self._live_plan_key = plan_key

        if plan is None:
            self._plot_canvas.show_message(
                f"compute_live_plan returned None for {variant.result_key}"
            )
            return

        self._apply_axis_manifest(plan, variant.result_key, variant.sub_config)
        self._update_time_range(len(plan.processed_df))
        start = self._start_var.get()
        duration = self._duration_var.get()
        plot_rows = (start, start + duration)
        self._file_count = plan.total_file_count
        self._file_index = min(self._file_index, max(0, self._file_count - 1))
        self._update_file_nav()
        fig = build_figure_from_plan(plan, self._file_index, plot_rows)
        if fig is not None:
            self._plot_canvas.display_figure(fig)
            logger.info(
                "Comparison %s (merged-config): file %d/%d",
                variant.result_key, self._file_index, plan.total_file_count,
            )
        else:
            self._plot_canvas.show_message(
                f"build_figure_from_plan returned None for {variant.result_key}"
            )

    def _display_comparison(self, entry: PlotEntry, variant: PlotVariant) -> None:
        """Render comparison plot from the merged single-mode config.

        Comparison rendering is driven by the ``scenario_rule`` field on
        the single-mode config (see :mod:`flextool.scenario_comparison.plan_union`)
        or, alternatively, by ``map_dimensions_for_plots`` supplied in
        ``comparison_overrides`` (which makes ``scenario_rule`` irrelevant).
        Configs that carry neither have no comparison view and surface a
        clear message instead of trying to render something undefined.
        """
        from flextool.scenario_comparison.plan_union import has_comparison_view
        single_cfg = self._load_single_plot_config(
            variant.result_key, variant.sub_config,
        )
        if single_cfg is None:
            self._plot_canvas.show_message(f"No config for {variant.result_key}")
            return
        if not has_comparison_view(single_cfg):
            self._plot_canvas.show_message(
                f"No comparison rendering for {variant.result_key}\n"
                f"(no `scenario_rule` or `comparison_overrides."
                f"map_dimensions_for_plots` in default_plots.yaml)."
            )
            return
        self._display_comparison_from_single(entry, variant, single_cfg)

    def _load_comparison_break_times(
        self, viewer_scenarios: list[str] | None = None,
    ) -> set[str] | None:
        """Return the union of timeline-break times across viewer scenarios.

        Phase E: ``_regenerate_comparison`` no longer writes a combined
        ``output_parquet_comparison/timeline_breaks.parquet``, so we
        union the per-scenario ``output_parquet/<scen>/timeline_breaks
        .parquet`` files at view time instead.  Falls back to the
        legacy combined file when no viewer scenarios are passed and
        the combined file happens to exist (e.g. CLI run).

        Cached under the ``"_comparison"`` slot so repeat callers
        don't re-read the per-scenario files.
        """
        if "_comparison" in self._break_times_cache:
            return self._break_times_cache["_comparison"]

        scenarios = list(viewer_scenarios or [])
        if scenarios:
            union: set[str] = set()
            any_present = False
            for scen in scenarios:
                p = (
                    self._project_path
                    / "output_parquet" / scen / "timeline_breaks.parquet"
                )
                if not p.exists():
                    continue
                any_present = True
                try:
                    df = read_lean_parquet(p)
                    if not df.empty:
                        union.update(df.iloc[:, 0].astype(str))
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "Failed to read timeline breaks for %s", scen,
                        exc_info=True,
                    )
            result: set[str] | None = union if any_present else None
            self._break_times_cache["_comparison"] = result
            return result

        # Legacy fallback (no viewer scenarios passed and / or CLI-built
        # combined file is around): read the combined parquet.
        path = (
            self._project_path
            / "output_parquet_comparison"
            / "timeline_breaks.parquet"
        )
        if not path.exists():
            self._break_times_cache["_comparison"] = None
            return None
        try:
            df = read_lean_parquet(path)
            if df.empty:
                result = None
            else:
                result = set(df.iloc[:, 0].astype(str))
            self._break_times_cache["_comparison"] = result
            return result
        except Exception:  # noqa: BLE001
            self._break_times_cache["_comparison"] = None
            return None

    # ------------------------------------------------------------------
    # Comparison scenarios tree (viewer-mode check state)
    # ------------------------------------------------------------------

    # Unicode check glyphs — matches the main window's input/executed trees.
    _COMP_CHECK_ON = "\u25a0"   # ■
    _COMP_CHECK_OFF = "\u25a1"  # □

    def _get_comparison_scenarios(self) -> list[str]:
        """Return list of checked scenario names for comparison."""
        result: list[str] = []
        for iid in self._comp_tree.get_children():
            values = self._comp_tree.item(iid, "values")
            if values and values[0] == self._COMP_CHECK_ON:
                result.append(values[1])
        return result

    def _populate_comparison_checkboxes(self) -> None:
        """Populate the comparison scenarios tree with current check state.

        Restores check state in this order:

        1. ``settings.comp_viewer_scenarios`` — the user's last-known ticks
           inside the viewer (saved on every click / space toggle).
        2. Otherwise fall back to ``_metadata.json`` (the last-run
           comparison's scenarios) so a freshly-opened project still
           reflects what was actually generated.
        """
        # Clear existing rows
        for iid in self._comp_tree.get_children():
            self._comp_tree.delete(iid)

        scenarios = self._scan_scenarios()
        if not scenarios:
            return

        checked_from_settings = [
            s for s in self._settings.comp_viewer_scenarios if s in scenarios
        ]
        if checked_from_settings:
            checked_set: set[str] = set(checked_from_settings)
        else:
            import json
            meta_path = self._project_path / "output_parquet_comparison" / "_metadata.json"
            checked_set = set()
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        checked_set = set(json.load(f).get("scenarios", []))
                except (json.JSONDecodeError, OSError):
                    pass

        for name in scenarios:
            glyph = self._COMP_CHECK_ON if name in checked_set else self._COMP_CHECK_OFF
            self._comp_tree.insert(
                "", "end", iid=name, values=(glyph, name),
            )

    def _save_comparison_check_state(self) -> None:
        """Persist the current viewer-comparison check state to settings.

        The settings object is shared with the main window (passed into
        the viewer's constructor), but the main window only persists on
        its own lifecycle events — we save here so the state survives a
        viewer crash or mode switch.
        """
        self._settings.comp_viewer_scenarios = self._get_comparison_scenarios()
        try:
            from flextool.gui.settings_io import save_project_settings
            save_project_settings(self._project_path, self._settings)
        except Exception:
            logger.warning("Could not persist comp_viewer_scenarios", exc_info=True)

    def _set_comp_check(self, name: str, checked: bool) -> None:
        """Flip the check glyph on one row."""
        if not self._comp_tree.exists(name):
            return
        glyph = self._COMP_CHECK_ON if checked else self._COMP_CHECK_OFF
        self._comp_tree.set(name, "check", glyph)

    def _on_comp_tree_toggled(self, _changed: list[str]) -> None:
        """CheckTreeController callback: filter-only replot after a toggle.

        Phase B: persisting the tick set + replotting is delegated to
        :meth:`_on_comp_check_state_changed` (which saves once).  We no
        longer regenerate the combined parquets from this path — that's
        the main-window button's job.
        """
        self._on_comp_check_state_changed()

    def _on_comp_tree_ctrl_a(self, _event: tk.Event) -> str:
        """Ctrl-A selects every row (does not alter check state)."""
        children = self._comp_tree.get_children()
        if children:
            self._comp_tree.selection_set(children)
            self._comp_tree.focus(children[0])
        return "break"

    def _on_comp_tree_reordered(self, new_order: list[str]) -> None:
        """Persist a new tree order and refresh comparison figures."""
        self._settings.executed_scenario_order = list(new_order)
        self._schedule_settings_save()
        # Re-render so the figures reflect the new order. The comparison
        # combine itself doesn't need to re-run (data is unchanged); the
        # figure-side ordering is driven by ``_get_comparison_scenarios``.
        self._clear_figure_cache()
        self._trigger_replot()

    def _on_comp_tree_alt_up(self, _event: tk.Event) -> str:
        sel = list(self._comp_tree.selection())
        if not sel:
            return "break"
        children = list(self._comp_tree.get_children())
        try:
            indices = sorted(children.index(iid) for iid in sel)
        except ValueError:
            return "break"
        if indices[0] == 0:
            return "break"
        for idx in indices:
            self._comp_tree.move(children[idx], "", idx - 1)
        self._on_comp_tree_reordered(list(self._comp_tree.get_children()))
        return "break"

    def _on_comp_tree_alt_down(self, _event: tk.Event) -> str:
        sel = list(self._comp_tree.selection())
        if not sel:
            return "break"
        children = list(self._comp_tree.get_children())
        try:
            indices = sorted(
                (children.index(iid) for iid in sel), reverse=True,
            )
        except ValueError:
            return "break"
        if indices[0] >= len(children) - 1:
            return "break"
        for idx in indices:
            self._comp_tree.move(children[idx], "", idx + 1)
        self._on_comp_tree_reordered(list(self._comp_tree.get_children()))
        return "break"

    def _on_comp_check_state_changed(self) -> None:
        """Handle a user-driven comparison check-state change.

        Phase B contract: toggling a row in the viewer's Scenarios tree
        is purely a *current scenarios* filter — it never triggers a
        rebuild of the combined parquets and never widens or narrows the
        axis-bounds set (the latter is frozen by Phase A's manifest
        scoping).  The figure simply hides or reveals the toggled
        scenario's lines/bars/area and the persisted state is saved so
        the next session reopens with the same ticks.
        """
        # Persist the new tick set.  ``_save_comparison_check_state``
        # writes it to ``settings.comp_viewer_scenarios``, which is now
        # exclusively the *current scenarios* record (the viewer
        # scenarios live in ``_metadata.json``).
        self._save_comparison_check_state()
        # The cached plan is keyed on the checked tuple; clear so the
        # next replot rebuilds against the filtered DataFrame.
        self._clear_figure_cache()
        self._trigger_replot()

    def _regenerate_comparison(self, scenario_names: list[str]) -> None:
        """Update the comparison-mode metadata and refresh the viewer.

        Phase E (lazy plan-parquet union) collapses this from a multi-GB
        cross-scenario raw-data combine to a metadata-only operation:

        1. Write ``output_parquet_comparison/_metadata.json`` with the
           viewer scenarios.
        2. Drop in-memory caches that depend on the viewer-scenarios
           set (figure cache, axis manifest, dispatch metadata, the
           unioned plan-parquet cache).
        3. Hand off to :meth:`_on_comparison_ready` to repopulate the
           comparison tree and the plot tree.

        The unit-of-work moves entirely to render time: when the user
        opens a plot, :meth:`_display_comparison_from_single` reads each
        viewer scenario's raw parquet, derives a comparison config from
        the single-mode rules + ``scenario_rule``, and renders.

        ``_comp_request_gen`` is still bumped so any in-flight async
        work from earlier rebuilds doesn't clobber the current state
        on completion — kept for forward-compatibility if async work
        is reintroduced here.
        """
        # Bump the generation counter — current state-change supersedes
        # any earlier in-flight async render that hasn't completed yet.
        self._comp_request_gen += 1
        gen = self._comp_request_gen

        self._plot_canvas.show_message("Updating comparison data...")

        try:
            self._write_comparison_metadata(list(scenario_names))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to write comparison metadata: %s", exc, exc_info=True,
            )
            self._on_comparison_failed(gen, exc)
            return

        self._on_comparison_ready(gen)

    def _write_comparison_metadata(self, scenario_names: list[str]) -> None:
        """Atomically write ``output_parquet_comparison/_metadata.json``.

        The metadata file is the canonical record of the viewer-scenarios
        set; downstream callers (Phase A axis manifest filtering,
        Phase D dispatch metadata, and Phase E's union path) read it to
        decide which per-scenario subdirs to consult.
        """
        import json
        import tempfile

        comp_dir = self._project_path / "output_parquet_comparison"
        comp_dir.mkdir(parents=True, exist_ok=True)
        meta_path = comp_dir / "_metadata.json"
        payload = {"scenarios": list(scenario_names)}

        fd, tmp_name = tempfile.mkstemp(
            dir=str(comp_dir), prefix="_metadata_", suffix=".tmp",
        )
        tmp_path = Path(tmp_name)
        try:
            with open(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            tmp_path.replace(meta_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    def _on_comparison_ready(self, gen: int) -> None:
        """Called on main thread when comparison parquets are ready.

        Drops the result if a newer state-change has been queued since
        this combine was submitted; otherwise refreshes caches, refreshes
        the comparison tree (to reflect the new viewer scenarios), and
        triggers a replot.
        """
        if gen != self._comp_request_gen:
            # A newer request has superseded this one; ignore the stale
            # result so the plot keeps the latest selection in flight.
            return
        self._comp_needs_regen = False
        self._scheduled_viewer_scenarios = None
        self._break_times_cache.clear()
        self._clear_figure_cache()
        self._current_availability = set()
        # Phase E: drop the unioned plan-parquet cache; the viewer
        # scenarios set just changed so the cached union no longer
        # matches.
        self._parquet_cache_key = ("", "")
        self._parquet_cache_df = None
        # Phase A axis manifest is keyed on disk; ``_get_axis_manifest``
        # picks up file rewrites by mtime, but invalidating the cached
        # view here forces a fresh read on the next render.
        self._axis_manifest = None
        self._axis_manifest_mtime = 0.0
        if hasattr(self, '_dispatch_metadata_cache'):
            del self._dispatch_metadata_cache
        # The viewer scenarios set just changed on disk — re-render the
        # comparison tree so its rows match ``_metadata.json`` again.
        if self._mode.get() == "comparison":
            self._populate_comparison_checkboxes()
        self._populate_plot_tree()
        # Try to display current selection
        self._trigger_replot()

    def _on_comparison_failed(self, gen: int, exc: BaseException) -> None:
        """Called on main thread when a comparison combine raised.

        Suppresses the error message if a newer request has superseded
        this one — the in-flight render shouldn't be clobbered by a
        stale failure.
        """
        if gen != self._comp_request_gen:
            return
        self._scheduled_viewer_scenarios = None
        self._plot_canvas.show_message(f"Comparison update failed:\n{exc}")

    # ------------------------------------------------------------------
    # Dispatch mode
    # ------------------------------------------------------------------

    def _populate_dispatch_tree(self) -> None:
        """Populate the tree with nodeGroups from dispatch data."""
        # Clear existing tree and entry map
        for item in self._plot_tree.get_children():
            self._plot_tree.delete(item)
        self._tree_entry_map.clear()

        scenarios = self._get_selected_scenarios()
        if not scenarios:
            return

        scenario = scenarios[0]
        parquet_dir = self._project_path / "output_parquet" / scenario
        if not parquet_dir.is_dir():
            return

        # Load dispatch groups (groups flagged for dispatch table output).
        # The model's nodeGroupDispatch set is the source of truth; the
        # similarly-named nodeGroupIndicators is for summary indicators
        # and a project may flag either independently.
        dispatch_groups_path = parquet_dir / "nodeGroupDispatch.parquet"
        if not dispatch_groups_path.exists():
            return

        df = read_lean_parquet(dispatch_groups_path)
        if df.empty:
            return

        group_col = 'group' if 'group' in df.columns else 'nodeGroupDispatch'
        flagged_groups = set(df[group_col].unique())

        # Filter to groups that actually have node members (group_node.parquet)
        group_node_path = parquet_dir / "group_node.parquet"
        if group_node_path.exists():
            gn_df = read_lean_parquet(group_node_path)
            if not gn_df.empty and 'group' in gn_df.columns:
                groups_with_nodes = set(gn_df['group'].unique())
                flagged_groups &= groups_with_nodes

        node_groups = sorted(flagged_groups)

        # Insert as flat list items (no group hierarchy)
        for ng in node_groups:
            iid = f"dispatch_{ng}"
            self._plot_tree.insert("", "end", iid=iid, text=ng)

        # Select first item
        if node_groups:
            first_iid = f"dispatch_{node_groups[0]}"
            self._plot_tree.selection_set(first_iid)
            self._plot_tree.see(first_iid)

    def _load_dispatch_data(self, scenario: str) -> bool:
        """Load dispatch data for a scenario. Returns True if successful."""
        if self._dispatch_scenario == scenario and self._dispatch_mappings is not None:
            return True  # Already loaded

        parquet_dir = self._project_path / "output_parquet" / scenario
        if not parquet_dir.is_dir():
            return False

        # Load dispatch mappings
        raw_mappings = load_dispatch_mappings(parquet_dir)

        # Build DispatchMappings with scenario in index.  Always tag by the
        # folder identity, overwriting any model-scenario tag baked in at
        # write time, so the folder name is the single dispatch identity
        # (matches combine_parquet_files, which re-tags the results the same
        # way).  See specs/dispatch_scenario_identity_retag.md.
        mapping_fields: dict[str, pd.DataFrame | None] = {}
        for key, df in raw_mappings.items():
            if df is not None and not df.empty:
                df_copy = df.copy()
                df_copy['scenario'] = scenario
                df_copy = df_copy.set_index('scenario')
                mapping_fields[key] = df_copy
            else:
                mapping_fields[key] = df
        self._dispatch_mappings = DispatchMappings(**mapping_fields)
        # Data and mappings are now both tagged by the folder identity, so
        # the cached slice key is just the folder name.
        self._dispatch_data_tag = resolve_data_scenario_tag(
            self._dispatch_mappings, scenario,
        )

        # Load TimeSeriesResults
        scenario_folders = build_scenario_folders_from_dir(
            self._project_path / "output_parquet", [scenario]
        )
        files_by_name = collect_parquet_files(scenario_folders, output_subdir="")
        combined = combine_parquet_files(files_by_name, num_scenarios=1)
        self._dispatch_results = TimeSeriesResults.from_dict(combined)

        self._dispatch_scenario = scenario
        return True

    def _load_dispatch_metadata(self) -> dict | None:
        """Load cross-scenario dispatch metadata (ylims, columns) if available.

        Resolution order (Phase D):

        1. If ``output_parquet_comparison/_dispatch_metadata.json`` exists
           AND ``output_parquet_comparison/_metadata.json`` records the
           same viewer-scenarios set as
           :meth:`_get_comparison_viewer_scenarios`, use the combined
           file (authoritative for the locked-in viewer set).
        2. Else compute via
           :func:`union_dispatch_metadata` over the per-scenario
           ``output_parquet/<scenario>/_dispatch_metadata.json`` files.
           Missing per-scenario files contribute nothing (silently
           skipped, like Phase C's availability union).
        3. Else ``None``.
        """
        if hasattr(self, '_dispatch_metadata_cache'):
            return self._dispatch_metadata_cache

        import json
        from flextool.scenario_comparison.dispatch_plots import (
            union_dispatch_metadata,
        )

        viewer_scenarios = self._get_comparison_viewer_scenarios()
        viewer_set = {str(s) for s in viewer_scenarios}

        combined_path = (
            self._project_path
            / "output_parquet_comparison"
            / "_dispatch_metadata.json"
        )
        if combined_path.exists():
            metadata_scenarios = set(self._read_metadata_scenarios())
            if metadata_scenarios == viewer_set and viewer_set:
                try:
                    with open(combined_path, "r", encoding="utf-8") as f:
                        self._dispatch_metadata_cache = json.load(f)
                    return self._dispatch_metadata_cache
                except (json.JSONDecodeError, OSError):
                    pass  # fall through to union

        if viewer_scenarios:
            try:
                meta = union_dispatch_metadata(
                    self._project_path, list(viewer_scenarios),
                )
                if meta.get("nodeGroups"):
                    self._dispatch_metadata_cache = meta
                    return self._dispatch_metadata_cache
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Failed to union per-scenario dispatch metadata",
                    exc_info=True,
                )

        self._dispatch_metadata_cache = None
        return None

    def _render_first_dispatch_figure(self) -> None:
        """Render the first dispatch nodeGroup for the selected scenario.

        Called after ``_populate_dispatch_tree`` to guarantee a figure
        appears on mode entry, since the virtual ``<<TreeviewSelect>>``
        event from programmatic selection isn't always delivered before
        the layout settles.
        """
        scenarios = self._get_selected_scenarios()
        if not scenarios:
            self._plot_canvas.show_message("Select a scenario to view dispatch")
            return
        selection = self._plot_tree.selection()
        if not selection or not selection[0].startswith("dispatch_"):
            # No node groups in the tree — nothing to render
            self._plot_canvas.show_message(
                "No dispatch node groups available for this scenario"
            )
            return
        node_group = selection[0][len("dispatch_"):]
        self._display_dispatch(scenarios[0], node_group)

    def _display_dispatch(self, scenario: str, node_group: str) -> None:
        """Render and display a dispatch plot for a nodeGroup."""
        from flextool.scenario_comparison.dispatch_plots import _compute_ylim

        if not self._load_dispatch_data(scenario):
            self._plot_canvas.show_message(f"Could not load dispatch data for {scenario}")
            return

        results = self._dispatch_results
        mappings = self._dispatch_mappings

        # Prepare dispatch data — slice by the in-data scenario tag, which
        # may differ from the folder name (GUI run-index suffix).
        df_dispatch, inflow = prepare_dispatch_data(
            results, mappings, self._dispatch_data_tag, node_group,
        )

        if df_dispatch is None or df_dispatch.empty:
            self._plot_canvas.show_message(f"No dispatch data for {node_group}")
            return

        self._update_time_range(len(df_dispatch))

        # Get timeline from start/duration controls
        start = self._start_var.get()
        duration = self._duration_var.get()
        timeline = (start, start + duration)

        # Accumulate ylims and column order across scenarios for this nodeGroup
        ymin, ymax = _compute_ylim(df_dispatch, timeline, inflow)
        if node_group in self._dispatch_ylims:
            old_min, old_max = self._dispatch_ylims[node_group]
            self._dispatch_ylims[node_group] = (min(old_min, ymin), max(old_max, ymax))
            # Add new columns preserving existing order
            for col in df_dispatch.columns:
                if col not in self._dispatch_columns[node_group]:
                    self._dispatch_columns[node_group].append(col)
        else:
            self._dispatch_ylims[node_group] = (ymin, ymax)
            self._dispatch_columns[node_group] = list(df_dispatch.columns)

        # Apply accumulated ylim with margin
        acc_min, acc_max = self._dispatch_ylims[node_group]
        margin = (acc_max - acc_min) * 0.05
        ylim = (acc_min - margin, acc_max + margin)

        # Ensure consistent column order across scenarios
        for col in self._dispatch_columns[node_group]:
            if col not in df_dispatch.columns:
                df_dispatch[col] = 0.0
        df_dispatch = df_dispatch[[c for c in self._dispatch_columns[node_group] if c in df_dispatch.columns]]

        # Also apply pre-computed metadata from comparison pipeline if available
        dispatch_meta = self._load_dispatch_metadata()
        if dispatch_meta:
            ng_meta = dispatch_meta.get("nodeGroups", {}).get(node_group)
            if ng_meta and "ylim" in ng_meta:
                pre_min, pre_max = ng_meta["ylim"]
                ylim = (min(ylim[0], pre_min), max(ylim[1], pre_max))

        # Load break times
        break_times = self._load_break_times(scenario)

        # Build figure
        fig = _build_dispatch_figure(
            df_dispatch, inflow,
            title=f"{node_group} \u2014 {scenario}",
            timeline=timeline,
            ylim=ylim,
            break_times=break_times,
        )

        if fig is None:
            self._plot_canvas.show_message(f"No plottable data for {node_group}")
            return

        # No file navigation for dispatch (single figure per nodeGroup)
        self._file_count = 1
        self._file_index = 0
        self._update_file_nav()

        self._plot_canvas.display_figure(fig)

    # ------------------------------------------------------------------
    # Network rendering
    # ------------------------------------------------------------------

    def _render_network(self) -> None:
        """Render the network graph for the selected scenario."""
        scenarios = self._get_selected_scenarios()
        if not scenarios:
            self._plot_canvas.show_message("Select a scenario to display the network graph")
            return

        scenario = scenarios[0]
        db_path = self._scenario_db_map.get(scenario)
        if not db_path:
            self._plot_canvas.show_message(
                f"No database found for scenario '{scenario}'"
            )
            return

        db_url = f"sqlite:///{db_path}"
        fig = build_network_figure(db_url)
        if fig is None:
            self._plot_canvas.show_message(
                "No latitude/longitude data found for nodes in this database.\n"
                "Geographic coordinates (lat, lon) must be defined as node parameters."
            )
            return

        self._plot_canvas.display_figure(fig)

    # ------------------------------------------------------------------
    # Window close
    # ------------------------------------------------------------------

    def _restore_sash_position(self) -> None:
        """Restore saved sash positions after the window is laid out.

        Clamps each saved pixel value to the current pane so a position
        captured at a different DPI / screen size cannot collapse one
        side to zero.
        """
        from flextool.gui.ui_metrics import clamp_sash, rescale_pixels
        try:
            self.update_idletasks()
            paned_total = self._paned.winfo_width()
            target = clamp_sash(
                rescale_pixels(
                    self._viewer_settings.left_pane_width,
                    self._viewer_settings.layout_cw,
                    self._char_width,
                ),
                paned_total,
                min_px=self._char_width * 20,
            )
            if target > 0:
                self._paned.sashpos(0, target)
        except (tk.TclError, IndexError):
            pass
        try:
            left_paned_total = self._left_paned.winfo_height()
            target = clamp_sash(
                rescale_pixels(
                    self._viewer_settings.scenario_pane_height,
                    self._viewer_settings.layout_cw,
                    self._char_width,
                ),
                left_paned_total,
                min_px=self._line_height * 4,
            )
            if target > 0:
                self._left_paned.sashpos(0, target)
        except (tk.TclError, IndexError):
            pass

    def _on_close(self) -> None:
        """Handle window close — persist settings and clean up resources."""
        self._hide_tooltip()
        self._render_gen += 1  # invalidate in-flight builds
        self._executor.shutdown(wait=False)
        self._clear_figure_cache()

        # Cancel any pending debounced save — the ``save_project_settings``
        # call below already covers the latest variant_durations state.
        if self._duration_save_after_id is not None:
            try:
                self.after_cancel(self._duration_save_after_id)
            except (tk.TclError, ValueError):
                pass
            self._duration_save_after_id = None

        # Unbind global key bindings added with bind_all
        for seq in ("<Prior>", "<Next>", "<Left>", "<Right>"):
            self.unbind_all(seq)

        # Save window geometry and sash positions
        self._viewer_settings.layout_cw = self._char_width
        self._viewer_settings.window_geometry = self.geometry()
        try:
            self._viewer_settings.left_pane_width = self._paned.sashpos(0)
        except (tk.TclError, IndexError):
            pass
        try:
            self._viewer_settings.scenario_pane_height = self._left_paned.sashpos(0)
        except (tk.TclError, IndexError):
            pass

        # Save comparison scenarios if in comparison mode
        if self._mode.get() == "comparison":
            self._settings.comp_plots_scenarios = self._get_comparison_scenarios()

        # Persist all settings
        try:
            save_project_settings(self._project_path, self._settings)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to save viewer settings on close", exc_info=True)

        # Clean up matplotlib resources
        self._plot_canvas.cleanup()

        self.destroy()
