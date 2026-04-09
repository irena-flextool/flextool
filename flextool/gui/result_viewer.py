"""ResultViewer — Toplevel window for browsing and displaying result plots."""

from __future__ import annotations

import logging
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tkinter import ttk

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from flextool.lean_parquet import read_lean_parquet
from flextool.gui.data_models import ProjectSettings
from flextool.gui.network_graph import build_network_figure
from flextool.gui.plot_canvas import PlotCanvas
from flextool.gui.plot_config_reader import PlotEntry, PlotGroup, PlotVariant, parse_plot_config
from flextool.gui.project_utils import get_projects_dir
from flextool.gui.settings_io import save_project_settings
from flextool.plot_outputs.config import PlotConfig, PLOT_FIELD_NAMES, _is_single_config, flatten_new_format
from flextool.plot_outputs.orchestrator import prepare_plot_data
from flextool.scenario_comparison.data_models import DispatchMappings, TimeSeriesResults
from flextool.scenario_comparison.db_reader import (
    build_scenario_folders_from_dir, collect_parquet_files, combine_parquet_files,
    combine_scenario_parquets,
)
from flextool.scenario_comparison.dispatch_data import prepare_dispatch_data
from flextool.scenario_comparison.dispatch_mappings import load_dispatch_mappings
from flextool.scenario_comparison.dispatch_plots import _build_dispatch_figure

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

        # Caches for parquet pipeline
        self._yaml_cache: dict[Path, dict] = {}
        self._break_times_cache: dict[str, set[str] | None] = {}
        self._parquet_cache_key: tuple[str, str] = ("", "")
        self._parquet_cache_df: pd.DataFrame | None = None

        # Guard against recursive replots from time range updates
        self._updating_time_range = False

        # Async figure building
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="plot")
        self._render_gen = 0  # incremented on each replot; stale results discarded
        self._figure_cache: dict[tuple, plt.Figure] = {}  # prefetched figures
        self._figure_cache_lock = threading.Lock()

        # Availability manifest for three-level variant display
        self._current_availability: set[tuple[str, str]] = set()

        # Comparison checkbox state
        self._comp_check_vars: dict[str, tk.BooleanVar] = {}
        self._comp_needs_regen: bool = False

        # Dispatch mode state
        self._dispatch_mappings: DispatchMappings | None = None
        self._dispatch_results: TimeSeriesResults | None = None
        self._dispatch_scenario: str = ""  # scenario for which dispatch data is loaded
        self._dispatch_ylims: dict[str, tuple[float, float]] = {}  # accumulated per-nodeGroup
        self._dispatch_columns: dict[str, list[str]] = {}  # accumulated column order

        # ── Font metrics for DPI-aware sizing ────────────────────────
        default_font = tkfont.nametofont("TkDefaultFont")
        cw: int = default_font.measure("0")
        lh: int = default_font.metrics("linespace")

        # ── Window sizing & positioning ──────────────────────────────
        self._line_height = lh
        self._char_width = cw
        self.minsize(cw * 80, lh * 30)

        master.update_idletasks()
        main_x = master.winfo_x()
        main_y = master.winfo_y()
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

        # Restore saved geometry if available
        if self._viewer_settings.window_geometry:
            try:
                self.geometry(self._viewer_settings.window_geometry)
            except tk.TclError:
                pass  # saved geometry may not fit current screen

        # ── Build layout ─────────────────────────────────────────────
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._paned = ttk.PanedWindow(self, orient="horizontal")
        self._paned.grid(row=0, column=0, sticky="nsew")

        self._build_left_column()
        self._build_right_column()

        # ── Configure tree tags and selection highlight ──────────────
        self._plot_tree.tag_configure("disabled", foreground="grey")

        # Make the selected item stand out in both light and dark themes
        style = ttk.Style()
        style.map(
            "Treeview",
            background=[("selected", "#2074d5")],
            foreground=[("selected", "#ffffff")],
        )

        # Theme-aware colors for variant grid
        self._fg_color = style.lookup("TLabel", "foreground") or "black"
        self._bg_color = style.lookup("TLabel", "background") or "white"

        # ── Resolve config paths ─────────────────────────────────────
        self._single_config_path = self._resolve_config_path(
            self._settings.single_plot_settings.config_file,
            "templates/default_plots.yaml",
        )
        self._comparison_config_path = self._resolve_config_path(
            self._settings.comparison_plot_settings.config_file,
            "templates/default_comparison_plots.yaml",
        )

        # ── Initial population ───────────────────────────────────────
        self._populate_scenarios()
        self._on_mode_changed()

        # ── Tab focus cycling ────────────────────────────────────────
        self._scenario_listbox.bind("<Tab>", self._focus_plot_tree)
        self._plot_tree.bind("<Tab>", self._focus_variant_canvas)

        # ── Global key bindings ──────────────────────────────────────
        self.bind("<Prior>", lambda e: self._on_prev_file())
        self.bind("<Next>", lambda e: self._on_next_file())

        # ── Window close ─────────────────────────────────────────────
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Layout builders
    # ------------------------------------------------------------------

    def _build_left_column(self) -> None:
        """Build the left column: scenario listbox + plot tree."""
        left = ttk.Frame(self._paned, padding=5)
        self._paned.add(left, weight=0)

        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)

        # ── Scenario listbox ─────────────────────────────────────────
        scen_frame = ttk.LabelFrame(left, text="Scenarios", padding=5)
        scen_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
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

        # ── Comparison checkbox frame (hidden by default) ────────────
        comp_outer = ttk.Frame(scen_frame)
        comp_outer.grid(row=0, column=0, columnspan=2, sticky="nsew")
        comp_outer.columnconfigure(0, weight=1)
        comp_outer.rowconfigure(0, weight=1)
        comp_outer.grid_remove()  # hidden initially
        self._comp_outer_frame = comp_outer

        comp_canvas = tk.Canvas(comp_outer, highlightthickness=0)
        comp_canvas.grid(row=0, column=0, sticky="nsew")
        comp_scroll = ttk.Scrollbar(
            comp_outer, orient="vertical", command=comp_canvas.yview,
        )
        comp_scroll.grid(row=0, column=1, sticky="ns")
        comp_canvas.configure(yscrollcommand=comp_scroll.set)

        self._comp_check_frame = ttk.Frame(comp_canvas)
        self._comp_check_frame_id = comp_canvas.create_window(
            (0, 0), window=self._comp_check_frame, anchor="nw",
        )
        self._comp_canvas = comp_canvas

        def _on_comp_frame_configure(_event: tk.Event) -> None:
            comp_canvas.configure(scrollregion=comp_canvas.bbox("all"))

        def _on_comp_canvas_configure(event: tk.Event) -> None:
            comp_canvas.itemconfig(self._comp_check_frame_id, width=event.width)

        self._comp_check_frame.bind("<Configure>", _on_comp_frame_configure)
        comp_canvas.bind("<Configure>", _on_comp_canvas_configure)

        # ── Plot tree + variant canvas ───────────────────────────────
        tree_frame = ttk.LabelFrame(left, text="Plots", padding=5)
        tree_frame.grid(row=1, column=0, sticky="nsew")
        tree_frame.columnconfigure(0, weight=1)
        # column 1 = variant canvas (fixed width, set later)
        # column 2 = scrollbar (fixed width)
        tree_frame.rowconfigure(0, weight=1)

        self._plot_tree = ttk.Treeview(
            tree_frame,
            show="tree",
            selectmode="browse",
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
        self._duration_var = tk.IntVar(value=self._settings.single_plot_settings.duration or 168)
        self._duration_spin = ttk.Spinbox(
            time_frame, from_=1, to=8760, textvariable=self._duration_var, width=6,
        )
        self._duration_spin.grid(row=1, column=1, sticky="w")

        # Bind changes to trigger replot
        self._start_var.trace_add("write", self._on_time_range_changed)
        self._duration_var.trace_add("write", self._on_time_range_changed)

        # Col 3: Update button
        self._update_btn = ttk.Button(
            self._control_frame, text="Update", width=7,
            command=self._on_update,
        )
        self._update_btn.grid(row=0, column=3, rowspan=2, sticky="ns", padx=(10, 0))

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
        to *default_relative* resolved from the projects parent directory
        (same approach as OutputActionManager).
        """
        if user_config:
            p = Path(user_config)
            if p.is_absolute() and p.is_file():
                return p
            # Try relative to project
            candidate = self._project_path / user_config
            if candidate.is_file():
                return candidate

        # Fall back to templates/ relative to get_projects_dir().parent
        return get_projects_dir().parent / default_relative

    # ------------------------------------------------------------------
    # Scenario discovery
    # ------------------------------------------------------------------

    def _scan_scenarios(self) -> list[str]:
        """List scenario subdirectories in output_parquet/ that are checked in Executed scenarios."""
        parquet_dir = self._project_path / "output_parquet"
        if not parquet_dir.is_dir():
            return []
        available = sorted(d.name for d in parquet_dir.iterdir() if d.is_dir())
        # Filter to only checked scenarios
        checked = self._settings.checked_executed_scenarios
        if checked:
            available = [s for s in available if s in checked]
        return available

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
        self._plot_groups = parse_plot_config(config_path)

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
        """Load availability manifest from plan_dir/_availability.json.

        Falls back to checking parquet file existence (old behavior)
        when the manifest does not exist.
        """
        import json

        avail_path = plan_dir / "_availability.json"
        if avail_path.exists():
            try:
                with open(avail_path) as f:
                    data = json.load(f)
                return {(r, s) for r, s in data.get("available", [])}
            except (json.JSONDecodeError, OSError):
                pass

        # Fallback: infer availability from parquet file existence.
        # The plan_dir parent is the parquet output directory.
        parquet_dir = plan_dir.parent
        result: set[tuple[str, str]] = set()
        if parquet_dir.is_dir():
            for f in parquet_dir.iterdir():
                if f.suffix == ".parquet" and f.is_file():
                    result.add((f.stem, "default"))
                    # Wildcard entry so that any sub_config matches
                    result.add((f.stem, "*"))
        return result

    def _update_tree_availability(self) -> None:
        """Grey out entries that have no matching parquet/plan data for selected scenario(s).

        Also stores the loaded availability set in ``self._current_availability``
        so that the variant grid can distinguish *defined-but-unavailable* from
        *available* variants.
        """
        mode = self._mode.get()

        if mode == "comparison":
            plan_dir = self._project_path / "output_parquet_comparison" / "plot_plans"
            available_pairs = self._load_availability_from_dir(plan_dir)
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

    def _should_skip_group(self, iid: str) -> bool:
        """Return True if this group header should be skipped during navigation.

        Open groups are skipped (their children are visible).
        Closed groups stop the cursor so the user can open them.
        """
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
            if not self._should_skip_group(visible[new_idx]):
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
            if not self._should_skip_group(visible[new_idx]):
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
                        text=display, fill=text_color, font=("TkDefaultFont",),
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
        # Hide tree selection highlight
        style = ttk.Style()
        style.map(
            "Treeview",
            background=[("selected", self._bg_color)],
            foreground=[("selected", self._fg_color)],
        )
        self._redraw_variant_grid()

    def _on_tree_focus_in(self, event: tk.Event) -> None:
        """Handle plot tree receiving focus."""
        self._focus_col = -1  # focus is in tree, not canvas
        # Restore tree selection highlight
        style = ttk.Style()
        style.map(
            "Treeview",
            background=[("selected", "#2074d5")],
            foreground=[("selected", "#ffffff")],
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
            # Show comparison checkboxes, hide scenario listbox
            self._scenario_listbox.grid_remove()
            self._comp_outer_frame.grid()
            self._populate_comparison_checkboxes()
            self._check_comparison_freshness()
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
                # Populate tree with nodeGroups instead of plot entries
                self._populate_dispatch_tree()
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

    def _update_time_range(self, data_length: int) -> None:
        """Update the Start slider range based on data length."""
        self._updating_time_range = True
        try:
            duration = self._duration_var.get()
            max_start = max(0, data_length - duration)
            self._start_scale.configure(to=max_start)
            # Clamp current start value
            if self._start_var.get() > max_start:
                self._start_var.set(max_start)
            # Update duration max
            self._duration_spin.configure(to=data_length)
        finally:
            self._updating_time_range = False

    def _on_time_range_changed(self, *_args) -> None:
        """Handle Start or Duration change — trigger replot."""
        if self._updating_time_range:
            return
        self._file_index = 0
        self._clear_figure_cache()
        self._trigger_replot()

    # ------------------------------------------------------------------
    # Figure cache management
    # ------------------------------------------------------------------

    def _clear_figure_cache(self) -> None:
        """Discard all prefetched figures."""
        with self._figure_cache_lock:
            self._figure_cache.clear()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def _on_update(self) -> None:
        """Re-scan scenarios, regenerate comparison if needed, reload everything."""
        # Clear all caches first
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

        self._populate_scenarios()
        if self._mode.get() == "comparison":
            self._populate_comparison_checkboxes()

        # Always check if comparison parquets need regenerating:
        # scenarios may have been re-run or the checked list may have changed.
        self._ensure_comparison_fresh()

        self._comp_needs_regen = False
        self._on_mode_changed()

    def _ensure_comparison_fresh(self) -> None:
        """Regenerate comparison parquets if stale or missing.

        Stale means: scenarios_changed flag is set (execution finished),
        checked scenario list differs from ``_metadata.json``, or the
        combined parquets don't exist yet.
        """
        checked = self._get_comparison_scenarios()
        if not checked:
            checked = self._scan_scenarios()
            if not checked:
                return
            for name, var in self._comp_check_vars.items():
                var.set(name in checked)

        needs_regen = self._settings.scenarios_changed

        if not needs_regen:
            meta_path = self._project_path / "output_parquet_comparison" / "_metadata.json"
            if not meta_path.exists():
                needs_regen = True
            else:
                import json
                try:
                    with open(meta_path) as f:
                        existing = set(json.load(f).get("scenarios", []))
                    if set(checked) != existing:
                        needs_regen = True
                except (json.JSONDecodeError, OSError):
                    needs_regen = True

        if needs_regen:
            self._settings.scenarios_changed = False
            self._regenerate_comparison(checked)
        self._comp_needs_regen = False

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
        """Move focus back to the scenario listbox (or checkbox frame in comparison mode)."""
        if self._mode.get() == "comparison":
            children = self._comp_check_frame.winfo_children()
            if children:
                children[0].focus_set()
            else:
                self._comp_canvas.focus_set()
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

        self._update_time_range(len(df))
        break_times = self._load_break_times(scenario)
        plot_name = config.plot_name or variant.full_name
        start = self._start_var.get()
        duration = self._duration_var.get()
        plot_rows = (start, start + duration)

        # 1b. Try pre-computed PlotPlan for instant rendering
        try:
            from flextool.plot_outputs.plan import load_plot_plan, build_figure_from_plan
            plan_dir = self._project_path / "output_parquet" / scenario / "plot_plans"
            plan = load_plot_plan(plan_dir, variant.result_key, variant.sub_config)
            if plan is not None:
                self._file_count = plan.total_file_count
                self._file_index = min(self._file_index, max(0, self._file_count - 1))
                self._update_file_nav()
                fig = build_figure_from_plan(plan, self._file_index)
                if fig is not None:
                    self._plot_canvas.display_figure(fig)
                    logger.info(
                        "Plot %s: from plan [file %d/%d]",
                        variant.result_key, self._file_index, plan.total_file_count,
                    )
                    return
        except Exception:
            pass  # fall through to normal pipeline

        # 2. Check prefetch cache for instant display
        cache_key = self._make_figure_cache_key(
            scenario, variant.result_key, variant.sub_config,
            self._file_index, start, duration,
        )
        with self._figure_cache_lock:
            cached_fig = self._figure_cache.pop(cache_key, None)

        if cached_fig is not None:
            self._plot_canvas.display_figure(cached_fig)
            logger.info("Plot %s: CACHED [file %d]", variant.result_key, self._file_index)
            self._prefetch_adjacent(scenario, variant, df, config, plot_name, break_times, start, duration)
            return

        # 3. Invalidate stale in-flight builds
        self._render_gen += 1
        gen = self._render_gen

        # 4. Submit build to background thread
        self._executor.submit(
            self._build_figure_async, gen, df, config, plot_name,
            plot_rows, break_times, self._file_index,
            scenario, variant, start, duration,
        )

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
        logger.error("prepare_plot_data failed for '%s': %s", result_key, error_msg)
        self._plot_canvas.show_message(f"Plot error: {error_msg}")

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

    def _display_comparison(self, entry: PlotEntry, variant: PlotVariant) -> None:
        """Render comparison plot from pre-combined parquets."""
        comp_dir = self._project_path / "output_parquet_comparison"
        parquet_path = comp_dir / f"{variant.result_key}.parquet"

        if not parquet_path.exists():
            self._plot_canvas.show_message(
                f"No comparison data found for {variant.result_key}\n"
                f"Run 'Scenario comparison' first."
            )
            return

        # Try pre-computed PlotPlan first
        try:
            from flextool.plot_outputs.plan import load_plot_plan, build_figure_from_plan
            plan_dir = comp_dir / "plot_plans"
            plan = load_plot_plan(plan_dir, variant.result_key, variant.sub_config)
            if plan is not None:
                self._update_time_range(len(plan.processed_df))
                self._file_count = plan.total_file_count
                self._file_index = min(self._file_index, max(0, self._file_count - 1))
                self._update_file_nav()
                fig = build_figure_from_plan(plan, self._file_index)
                if fig is not None:
                    self._plot_canvas.display_figure(fig)
                    logger.info(
                        "Comparison %s: from plan [file %d/%d]",
                        variant.result_key, self._file_index, plan.total_file_count,
                    )
                    return
        except Exception:
            pass  # fall through to normal pipeline

        df = read_lean_parquet(parquet_path)
        if df.empty:
            self._plot_canvas.show_message(f"Empty data for {variant.result_key}")
            return

        # Filter to only checked scenarios
        checked = set(self._get_comparison_scenarios())
        if checked and isinstance(df.columns, pd.MultiIndex) and 'scenario' in df.columns.names:
            scenario_level = df.columns.get_level_values('scenario')
            mask = scenario_level.isin(checked)
            df = df.loc[:, mask]
            if df.empty:
                self._plot_canvas.show_message("No data for selected scenarios")
                return

        self._update_time_range(len(df))

        # Load config from comparison config
        config = self._load_plot_config(variant.result_key, variant.sub_config)
        if config is None:
            self._plot_canvas.show_message(f"No config for {variant.result_key}")
            return

        # Load break times from comparison dir
        break_times = self._load_comparison_break_times()

        plot_name = config.plot_name or variant.full_name
        start = self._start_var.get()
        duration = self._duration_var.get()
        plot_rows = (start, start + duration)

        figures, total_count = prepare_plot_data(
            df, config, plot_name, plot_rows, break_times,
            only_file_index=self._file_index,
        )

        self._file_count = max(total_count, 1)
        self._file_index = min(self._file_index, max(0, self._file_count - 1))
        self._update_file_nav()

        if figures:
            filename, fig = figures[0]
            self._plot_canvas.display_figure(fig)
        else:
            self._plot_canvas.show_message(f"No plottable data for {variant.full_name}")

    def _load_comparison_break_times(self) -> set[str] | None:
        """Load break times from comparison parquet directory."""
        if "_comparison" in self._break_times_cache:
            return self._break_times_cache["_comparison"]

        path = self._project_path / "output_parquet_comparison" / "timeline_breaks.parquet"
        if not path.exists():
            self._break_times_cache["_comparison"] = None
            return None

        try:
            df = read_lean_parquet(path)
            if df.empty:
                result: set[str] | None = None
            else:
                result = set(df.iloc[:, 0].astype(str))
            self._break_times_cache["_comparison"] = result
            return result
        except Exception:  # noqa: BLE001
            self._break_times_cache["_comparison"] = None
            return None

    def _get_comparison_scenarios(self) -> list[str]:
        """Return list of checked scenario names for comparison."""
        return [name for name, var in self._comp_check_vars.items() if var.get()]

    def _populate_comparison_checkboxes(self) -> None:
        """Build checkboxes for all available scenarios."""
        # Clear existing
        for widget in self._comp_check_frame.winfo_children():
            widget.destroy()
        self._comp_check_vars.clear()

        # Get all scenarios
        scenarios = self._scan_scenarios()

        # Load currently combined scenarios from _metadata.json
        import json
        meta_path = self._project_path / "output_parquet_comparison" / "_metadata.json"
        combined_scenarios: set[str] = set()
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    combined_scenarios = set(json.load(f).get("scenarios", []))
            except (json.JSONDecodeError, OSError):
                pass

        for name in scenarios:
            var = tk.BooleanVar(value=(name in combined_scenarios))
            self._comp_check_vars[name] = var
            cb = ttk.Checkbutton(
                self._comp_check_frame, text=name, variable=var,
                command=self._on_comp_checkbox_changed,
            )
            cb.pack(fill="x", anchor="w", padx=2, pady=1)

    def _on_comp_checkbox_changed(self) -> None:
        """Handle comparison checkbox change — filter and replot immediately."""
        import json

        # Check if we need scenarios not in the combined parquet
        checked = set(self._get_comparison_scenarios())
        meta_path = self._project_path / "output_parquet_comparison" / "_metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    combined = set(json.load(f).get("scenarios", []))
            except (json.JSONDecodeError, OSError):
                combined = set()
        else:
            combined = set()

        if checked - combined:
            # New scenarios not in combined — need regeneration
            self._comp_needs_regen = True
            self._regenerate_comparison(list(checked))
        else:
            # Subset of combined — just filter and replot
            self._comp_needs_regen = False
            self._clear_figure_cache()
            self._trigger_replot()

    def _check_comparison_freshness(self) -> None:
        """Check if comparison parquets match the checkbox selection, regenerate if not."""
        checked = self._get_comparison_scenarios()
        if not checked:
            return

        import json
        meta_path = self._project_path / "output_parquet_comparison" / "_metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    existing = set(json.load(f).get("scenarios", []))
                if set(checked) == existing:
                    return  # already up to date
            except (json.JSONDecodeError, OSError):
                pass

        # Need regeneration
        self._regenerate_comparison(checked)

    def _regenerate_comparison(self, scenario_names: list[str]) -> None:
        """Regenerate comparison parquets in a background thread."""
        self._plot_canvas.show_message("Updating comparison data...")

        def _do_combine() -> None:
            try:
                combine_scenario_parquets(self._project_path, scenario_names)
                self.after(0, self._on_comparison_ready)
            except Exception as exc:
                self.after(0, lambda: self._plot_canvas.show_message(
                    f"Comparison update failed:\n{exc}"
                ))

        self._executor.submit(_do_combine)

    def _on_comparison_ready(self) -> None:
        """Called on main thread when comparison parquets are ready."""
        self._comp_needs_regen = False
        self._break_times_cache.clear()
        self._clear_figure_cache()
        self._current_availability = set()
        if hasattr(self, '_dispatch_metadata_cache'):
            del self._dispatch_metadata_cache
        self._populate_plot_tree()
        # Try to display current selection
        self._trigger_replot()

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

        # Load dispatch groups (groups flagged for dispatch output)
        dispatch_groups_path = parquet_dir / "outputNodeGroup_does_specified_flows.parquet"
        if not dispatch_groups_path.exists():
            return

        df = read_lean_parquet(dispatch_groups_path)
        if df.empty:
            return

        flagged_groups = set(df['group'].unique())

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

        # Build DispatchMappings with scenario in index
        # For single scenario, add scenario column and set as index
        mapping_fields: dict[str, pd.DataFrame | None] = {}
        for key, df in raw_mappings.items():
            if df is not None and not df.empty:
                df_copy = df.copy()
                if 'scenario' not in df_copy.columns:
                    df_copy['scenario'] = scenario
                df_copy = df_copy.set_index('scenario')
                mapping_fields[key] = df_copy
            else:
                mapping_fields[key] = df
        self._dispatch_mappings = DispatchMappings(**mapping_fields)

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
        """Load cross-scenario dispatch metadata (ylims, columns) if available."""
        if hasattr(self, '_dispatch_metadata_cache'):
            return self._dispatch_metadata_cache
        import json
        meta_path = self._project_path / "output_parquet_comparison" / "_dispatch_metadata.json"
        if not meta_path.exists():
            self._dispatch_metadata_cache = None
            return None
        try:
            with open(meta_path, "r") as f:
                self._dispatch_metadata_cache = json.load(f)
            return self._dispatch_metadata_cache
        except (json.JSONDecodeError, OSError):
            self._dispatch_metadata_cache = None
            return None

    def _display_dispatch(self, scenario: str, node_group: str) -> None:
        """Render and display a dispatch plot for a nodeGroup."""
        from flextool.scenario_comparison.dispatch_plots import _compute_ylim

        if not self._load_dispatch_data(scenario):
            self._plot_canvas.show_message(f"Could not load dispatch data for {scenario}")
            return

        results = self._dispatch_results
        mappings = self._dispatch_mappings

        # Prepare dispatch data
        df_dispatch, inflow = prepare_dispatch_data(
            results, mappings, scenario, node_group,
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

    def _on_close(self) -> None:
        """Handle window close — persist settings and clean up resources."""
        self._hide_tooltip()
        self._render_gen += 1  # invalidate in-flight builds
        self._executor.shutdown(wait=False)
        self._clear_figure_cache()

        # Save window geometry
        self._viewer_settings.window_geometry = self.geometry()

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
