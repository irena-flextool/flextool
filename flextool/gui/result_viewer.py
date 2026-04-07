"""ResultViewer — Toplevel window for browsing and displaying result plots."""

from __future__ import annotations

import logging
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import ttk

import pandas as pd
import yaml

from flextool.gui.data_models import ProjectSettings
from flextool.gui.network_graph import build_network_figure
from flextool.gui.plot_canvas import PlotCanvas
from flextool.gui.plot_config_reader import PlotEntry, PlotGroup, PlotVariant, parse_plot_config
from flextool.gui.project_utils import get_projects_dir
from flextool.gui.settings_io import save_project_settings
from flextool.plot_outputs.config import PlotConfig, PLOT_FIELD_NAMES, _is_single_config
from flextool.plot_outputs.orchestrator import prepare_plot_data
from flextool.scenario_comparison.data_models import DispatchMappings, TimeSeriesResults
from flextool.scenario_comparison.db_reader import (
    build_scenario_folders_from_dir, collect_parquet_files, combine_parquet_files,
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
        # Dual variant state: desired (user's explicit choice) and shown (actually displayed)
        self._desired_variant: str = self._viewer_settings.last_variant or ""
        self._shown_variant: str = ""
        # All unique variant letters across all config entries (created once)
        self._all_variant_letters: list[str] = []
        # List of variant buttons currently displayed
        self._variant_buttons: list[ttk.Button] = []
        # Tooltip toplevel
        self._tooltip: tk.Toplevel | None = None

        # Mode variable: "single", "comparison", "network"
        self._mode = tk.StringVar(value=self._viewer_settings.last_mode or "single")

        # File navigation state
        self._file_index = 0
        self._file_count = 1

        # Caches for parquet pipeline
        self._yaml_cache: dict[Path, dict] = {}
        self._break_times_cache: dict[str, set[str] | None] = {}

        # Dispatch mode state
        self._dispatch_mappings: DispatchMappings | None = None
        self._dispatch_results: TimeSeriesResults | None = None
        self._dispatch_scenario: str = ""  # scenario for which dispatch data is loaded

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

        # ── Configure tree disabled tag ──────────────────────────────
        self._plot_tree.tag_configure("disabled", foreground="grey")

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
        self._plot_tree.bind("<Tab>", self._focus_variant_panel)
        # Variant panel Tab is handled dynamically when buttons are created

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

        # ── Plot tree ────────────────────────────────────────────────
        tree_frame = ttk.LabelFrame(left, text="Plots", padding=5)
        tree_frame.grid(row=1, column=0, sticky="nsew")
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)

        self._plot_tree = ttk.Treeview(
            tree_frame,
            show="tree",
            selectmode="browse",
        )
        self._plot_tree.grid(row=0, column=0, sticky="nsew")

        tree_scroll = ttk.Scrollbar(
            tree_frame, orient="vertical", command=self._plot_tree.yview
        )
        tree_scroll.grid(row=0, column=1, sticky="ns")
        self._plot_tree.configure(yscrollcommand=tree_scroll.set)

        self._plot_tree.bind("<<TreeviewSelect>>", self._on_tree_selected)
        self._plot_tree.bind("<Motion>", self._on_tree_motion)
        self._plot_tree.bind("<Leave>", self._hide_tooltip)

    def _build_right_column(self) -> None:
        """Build the right column: compact control bar + plot area."""
        right = ttk.Frame(self._paned, padding=5)
        self._paned.add(right, weight=1)

        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)  # plot area gets all extra space

        # ── Combined control frame ───────────────────────────────────
        self._control_frame = ttk.Frame(right, padding=(5, 2))
        self._control_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        self._control_frame.columnconfigure(3, weight=1)  # time frame fills remaining

        # Col 0: Variant buttons frame
        self._variant_frame = ttk.LabelFrame(self._control_frame, text="Variant", padding=(2, 1))
        self._variant_frame.grid(row=0, column=0, rowspan=2, sticky="ns", padx=(0, 5))
        # Placeholder label shown when no variants are available
        self._variant_placeholder = ttk.Label(
            self._variant_frame, text="...", foreground="grey",
        )
        self._variant_placeholder.pack(side="left")

        # Col 1: File navigation (Prev on top, Next on bottom)
        file_nav_frame = ttk.Frame(self._control_frame)
        file_nav_frame.grid(row=0, column=1, rowspan=2, sticky="ns", padx=(0, 5))

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

        # Col 2: Mode radio buttons (stacked vertically)
        mode_frame = ttk.Frame(self._control_frame)
        mode_frame.grid(row=0, column=2, rowspan=2, sticky="ns", padx=(0, 10))

        for text, value in [("Single", "single"), ("Comparison", "comparison"), ("Dispatch", "dispatch"), ("Network", "network")]:
            rb = ttk.Radiobutton(
                mode_frame, text=text, variable=self._mode, value=value,
                command=self._on_mode_changed,
            )
            rb.pack(side="top", anchor="w")

        # Col 3: Start slider + Duration spinbox (stacked, slider fills width)
        time_frame = ttk.Frame(self._control_frame)
        time_frame.grid(row=0, column=3, rowspan=2, sticky="nsew", padx=(0, 10))
        time_frame.columnconfigure(1, weight=1)

        ttk.Label(time_frame, text="Start").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self._start_var = tk.IntVar(value=self._settings.single_plot_settings.start_time)
        self._start_scale = ttk.Scale(
            time_frame, from_=0, to=8760, orient="horizontal",
            variable=self._start_var,
        )
        self._start_scale.grid(row=0, column=1, sticky="ew")
        # Disabled until plot pipeline integration (PNG mode has no time range)
        self._start_scale.configure(state="disabled")

        ttk.Label(time_frame, text="Duration").grid(row=1, column=0, sticky="w", padx=(0, 5))
        self._duration_var = tk.IntVar(value=self._settings.single_plot_settings.duration or 168)
        self._duration_spin = ttk.Spinbox(
            time_frame, from_=1, to=8760, textvariable=self._duration_var, width=6,
        )
        self._duration_spin.grid(row=1, column=1, sticky="w")
        # Disabled until plot pipeline integration
        self._duration_spin.configure(state="disabled")

        # Col 4: Refresh button
        self._refresh_btn = ttk.Button(
            self._control_frame, text="Refresh", width=7,
            command=self._on_refresh,
        )
        self._refresh_btn.grid(row=0, column=4, rowspan=2, sticky="ns", padx=(10, 0))

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
        """List scenario subdirectories in output_parquet/."""
        parquet_dir = self._project_path / "output_parquet"
        if not parquet_dir.is_dir():
            return []
        return sorted(d.name for d in parquet_dir.iterdir() if d.is_dir())

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

        # Collect all unique variant letters across all entries and create buttons once
        self._collect_all_variant_letters()
        self._create_variant_buttons()

        # Grey out entries without matching parquet data
        self._update_tree_availability()

        # Try to restore last selected entry or select first available
        self._restore_or_select_first_entry()

    def _update_tree_availability(self) -> None:
        """Grey out entries that have no matching parquet files for selected scenario(s)."""
        mode = self._mode.get()

        if mode == "comparison":
            # Check comparison parquet directory
            comp_dir = self._project_path / "output_parquet_comparison"
            available_keys: set[str] = set()
            if comp_dir.is_dir():
                for f in comp_dir.iterdir():
                    if f.suffix == ".parquet" and f.is_file():
                        available_keys.add(f.stem)
        else:
            scenarios = self._get_selected_scenarios()
            if not scenarios:
                # No scenario selected -- mark all as disabled
                for iid in self._tree_entry_map:
                    self._plot_tree.item(iid, tags=("disabled",))
                return

            # Collect available parquet file stems for selected scenarios
            available_keys = set()
            for scenario in scenarios:
                parquet_dir = self._project_path / "output_parquet" / scenario
                if parquet_dir.is_dir():
                    for f in parquet_dir.iterdir():
                        if f.suffix == ".parquet" and f.is_file():
                            available_keys.add(f.stem)

        for iid, entry in self._tree_entry_map.items():
            # An entry is available if ANY of its variants' result_keys
            # have a matching parquet file
            has_data = any(
                v.result_key in available_keys for v in entry.variants
            )
            if has_data:
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

        # Reset file index
        self._file_index = 0
        self._file_count = 1

        self._update_file_nav()
        self._populate_variant_panel(entry)
        self._trigger_replot()

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
        """Collect all unique variant letters across all entries in the config."""
        seen: set[str] = set()
        ordered: list[str] = []
        for group in self._plot_groups:
            for entry in group.entries:
                for v in entry.variants:
                    if v.letter not in seen:
                        seen.add(v.letter)
                        ordered.append(v.letter)
        self._all_variant_letters = ordered

    def _create_variant_buttons(self) -> None:
        """Create variant buttons once for all letters in the config."""
        # Destroy old buttons
        for btn in self._variant_buttons:
            btn.destroy()
        self._variant_buttons.clear()
        self._variant_placeholder.pack_forget()

        if not self._all_variant_letters:
            self._variant_placeholder.configure(text="No variants")
            self._variant_placeholder.pack(side="left")
            return

        for letter in self._all_variant_letters:
            display = letter or "?"
            btn = ttk.Button(
                self._variant_frame,
                text=display,
                width=3,
                command=lambda v=letter: self._on_variant_clicked(v),
            )
            btn._letter = letter  # type: ignore[attr-defined]  # store letter as attribute
            btn.pack(side="left", padx=2, pady=1)
            self._variant_buttons.append(btn)

            # Bind Left/Right for variant navigation, Up/Down for tree navigation
            btn.bind("<Left>", self._on_variant_left)
            btn.bind("<Right>", self._on_variant_right)
            btn.bind("<Up>", self._on_variant_key_up)
            btn.bind("<Down>", self._on_variant_key_down)
            btn.bind("<Tab>", self._focus_scenario_listbox)

    def _find_nearest_available(self, available: set[str]) -> str:
        """Find nearest available variant letter to the desired one.

        Searches left first, then right in the _all_variant_letters list.
        """
        if not available:
            return ""
        if self._desired_variant in available:
            return self._desired_variant

        try:
            idx = self._all_variant_letters.index(self._desired_variant)
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
        """Update variant button states for the given entry (don't recreate).

        Updates the *shown* variant to match desired when available,
        otherwise picks the nearest available.  The *desired* variant
        is never changed here — it persists across tree navigation.
        """
        available = {v.letter for v in entry.variants}

        for btn in self._variant_buttons:
            letter = btn._letter  # type: ignore[attr-defined]
            if letter in available:
                btn.configure(state="normal")
            else:
                btn.configure(state="disabled")

        # Shown variant: desired if available, else nearest available
        if self._desired_variant in available:
            self._shown_variant = self._desired_variant
        else:
            self._shown_variant = self._find_nearest_available(available)

        self._highlight_variants()

    def _highlight_variants(self) -> None:
        """Visually highlight the shown and desired variant buttons.

        - Shown variant: ``Accent.TButton`` (solid highlight).
        - Desired variant (when different from shown): ``Desired.TButton``
          (groove relief to indicate the user's persistent choice).
        - Both on same button: ``Accent.TButton`` (solid takes priority).
        - Neither: ``TButton`` (default).
        """
        # Ensure the Desired style exists (idempotent)
        style = ttk.Style()
        style.configure("Desired.TButton", relief="groove")

        for btn in self._variant_buttons:
            letter = btn._letter  # type: ignore[attr-defined]
            if letter == self._shown_variant:
                btn.configure(style="Accent.TButton")
            elif letter == self._desired_variant:
                btn.configure(style="Desired.TButton")
            else:
                btn.configure(style="TButton")

    def _on_variant_clicked(self, letter: str) -> None:
        """Handle variant button click."""
        # Only act if the button is for an available variant
        selection = self._plot_tree.selection()
        if selection and selection[0].startswith("entry_"):
            entry = self._tree_entry_map.get(selection[0])
            if entry:
                available = {v.letter for v in entry.variants}
                if letter not in available:
                    return

        self._desired_variant = letter
        self._shown_variant = letter  # clicking always sets both
        self._viewer_settings.last_variant = letter
        self._highlight_variants()
        self._file_index = 0
        self._trigger_replot()

    def _on_variant_left(self, event: tk.Event) -> str:
        """Navigate to previous enabled variant button.

        Changes desired variant to the previous available one; shown follows.
        """
        if not self._variant_buttons:
            return "break"
        current_idx = self._get_focused_variant_index()
        # Find previous enabled button
        for new_idx in range(current_idx - 1, -1, -1):
            btn = self._variant_buttons[new_idx]
            if str(btn.cget("state")) != "disabled":
                btn.focus_set()
                self._on_variant_clicked(btn._letter)  # type: ignore[attr-defined]
                break
        return "break"

    def _on_variant_right(self, event: tk.Event) -> str:
        """Navigate to next enabled variant button.

        Changes desired variant to the next available one; shown follows.
        """
        if not self._variant_buttons:
            return "break"
        current_idx = self._get_focused_variant_index()
        # Find next enabled button
        for new_idx in range(current_idx + 1, len(self._variant_buttons)):
            btn = self._variant_buttons[new_idx]
            if str(btn.cget("state")) != "disabled":
                btn.focus_set()
                self._on_variant_clicked(btn._letter)  # type: ignore[attr-defined]
                break
        return "break"

    def _on_variant_key_up(self, event: tk.Event) -> str:
        """Handle Up / Shift+Up in the variant panel.

        - Up: jump to prev visible entry that has the desired variant.
        - Shift+Up: move to prev visible entry regardless; keep desired,
          shown = nearest available.
        """
        shift_held = bool(event.state & 0x1)
        if shift_held:
            self._move_tree_selection(-1)
        else:
            self._move_tree_to_next_with_desired(-1)
        return "break"

    def _on_variant_key_down(self, event: tk.Event) -> str:
        """Handle Down / Shift+Down in the variant panel.

        - Down: jump to next visible entry that has the desired variant.
        - Shift+Down: move to next visible entry regardless; keep desired,
          shown = nearest available.
        """
        shift_held = bool(event.state & 0x1)
        if shift_held:
            self._move_tree_selection(1)
        else:
            self._move_tree_to_next_with_desired(1)
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

    def _move_tree_to_next_with_desired(self, direction: int) -> None:
        """Move tree selection to next/prev entry that has the desired variant.

        Only stops at non-disabled entries whose variant set includes
        ``_desired_variant``.
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
                if entry and self._desired_variant in {v.letter for v in entry.variants}:
                    self._plot_tree.selection_set(iid)
                    self._plot_tree.see(iid)
                    self._plot_tree.event_generate("<<TreeviewSelect>>")
                    return
            new_idx += step

    def _get_focused_variant_index(self) -> int:
        """Return index of the currently focused variant button, or 0."""
        focused = self.focus_get()
        for i, btn in enumerate(self._variant_buttons):
            if btn is focused:
                return i
        # Fall back to desired variant
        for i, btn in enumerate(self._variant_buttons):
            if btn._letter == self._desired_variant:  # type: ignore[attr-defined]
                return i
        return 0

    def _show_variant_panel(self) -> None:
        """Show the variant panel."""
        self._variant_frame.grid()

    def _hide_variant_panel(self) -> None:
        """Hide the variant panel and disable all buttons."""
        for btn in self._variant_buttons:
            btn.configure(state="disabled")
        self._shown_variant = ""

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def _on_mode_changed(self) -> None:
        """Handle mode radio button change."""
        mode = self._mode.get()
        self._viewer_settings.last_mode = mode

        if mode == "single":
            self._scenario_listbox.configure(selectmode="browse")
            self._populate_plot_tree()
            self._show_variant_panel()
        elif mode == "comparison":
            self._scenario_listbox.configure(selectmode="browse")
            self._populate_comparison_scenarios()
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
    # Refresh
    # ------------------------------------------------------------------

    def _on_refresh(self) -> None:
        """Re-scan scenarios and re-populate everything."""
        self._yaml_cache.clear()
        self._break_times_cache.clear()
        self._dispatch_scenario = ""
        self._dispatch_mappings = None
        self._dispatch_results = None
        self._populate_scenarios()
        self._on_mode_changed()

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

    def _focus_variant_panel(self, _event: tk.Event | None = None) -> str:
        """Move focus to the first variant button."""
        if self._variant_buttons:
            # Focus the active variant button
            for btn in self._variant_buttons:
                if btn.cget("text") == self._desired_variant:
                    btn.focus_set()
                    return "break"
            self._variant_buttons[0].focus_set()
        return "break"

    def _focus_scenario_listbox(self, _event: tk.Event | None = None) -> str:
        """Move focus back to the scenario listbox."""
        self._scenario_listbox.focus_set()
        return "break"

    # ------------------------------------------------------------------
    # Plot display
    # ------------------------------------------------------------------

    def _get_active_variant(self, entry: PlotEntry) -> PlotVariant | None:
        """Return the PlotVariant matching the shown variant letter."""
        for v in entry.variants:
            if v.letter == self._shown_variant:
                return v
        # Fall back to first variant
        return entry.variants[0] if entry.variants else None

    def _build_plot_name(self, entry: PlotEntry, variant: PlotVariant) -> str:
        """Reconstruct the plot_name used as the file basename.

        The plot pipeline saves files using the ``plot_name`` field from the
        YAML config, which follows the pattern::

            "{group}.{entry_sub}.{variant_letter} {human_name}"

        e.g. ``"0.0.t Loss of load (upward slack)"``.
        When the variant letter is empty the dot is omitted:
        ``"5.0 Emissions CO2 total"``.
        """
        if variant.letter:
            return f"{entry.number}.{variant.letter} {variant.full_name}"
        return f"{entry.number} {variant.full_name}"

    def _find_png_files(self, scenario: str, entry: PlotEntry, variant: PlotVariant) -> list[Path]:
        """Find PNG files for *variant* of *entry* in the given *scenario*.

        Checks for both single-file and split-file naming conventions.
        Returns sorted list of matching paths (may be empty).
        """
        mode = self._mode.get()
        if mode == "comparison":
            plot_dir = self._project_path / "output_plot_comparisons"
        else:
            plot_dir = self._project_path / "output_plots" / scenario

        if not plot_dir.is_dir():
            return []

        plot_name = self._build_plot_name(entry, variant)
        single = plot_dir / f"{plot_name}.png"
        if single.is_file():
            return [single]

        # Check for split files: {plot_name}_01.png, _02.png, ...
        split_files: list[Path] = []
        idx = 1
        while True:
            candidate = plot_dir / f"{plot_name}_{idx:02d}.png"
            if candidate.is_file():
                split_files.append(candidate)
                idx += 1
            else:
                break

        # Also check for file-member variants: {plot_name}_{member}.png
        # Scan directory for files starting with the plot_name prefix
        if not split_files:
            prefix = plot_name + "_"
            for p in sorted(plot_dir.iterdir()):
                if p.name.startswith(prefix) and p.suffix == ".png":
                    split_files.append(p)

        return split_files

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

        entry = plots.get(result_key)
        if not isinstance(entry, dict):
            return None

        # Determine which raw config dict to use
        if sub_config == "default" and _is_single_config(entry):
            raw = entry
        elif sub_config != "default" and not _is_single_config(entry):
            raw = entry.get(sub_config)
            if not isinstance(raw, dict):
                return None
        else:
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

        try:
            return PlotConfig(**filtered)
        except TypeError as exc:
            logger.error("Failed to create PlotConfig for '%s': %s", result_key, exc)
            return None

    def _load_parquet(self, scenario: str, result_key: str) -> pd.DataFrame | None:
        """Load a parquet file for the given scenario and result_key."""
        path = self._project_path / "output_parquet" / scenario / f"{result_key}.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def _load_break_times(self, scenario: str) -> set[str] | None:
        """Load timeline break times from parquet, cached per scenario."""
        if scenario in self._break_times_cache:
            return self._break_times_cache[scenario]

        path = self._project_path / "output_parquet" / scenario / "timeline_breaks.parquet"
        if not path.exists():
            self._break_times_cache[scenario] = None
            return None

        try:
            df = pd.read_parquet(path)
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

    def _display_from_parquet(self, scenario: str, entry: PlotEntry, variant: PlotVariant) -> None:
        """Load parquet, build PlotConfig, render Figure, display it."""
        import matplotlib.pyplot as plt

        # 1. Load parquet
        df = self._load_parquet(scenario, variant.result_key)
        if df is None:
            self._plot_canvas.show_message(f"No data: {variant.result_key}.parquet")
            return

        # 2. Load plot config
        config = self._load_plot_config(variant.result_key, variant.sub_config)
        if config is None:
            self._plot_canvas.show_message(f"No config for {variant.result_key}")
            return

        # 3. Load break times
        break_times = self._load_break_times(scenario)

        # 4. Build plot name (for figure title)
        plot_name = config.plot_name or variant.full_name

        # 5. Get plot_rows from start/duration controls
        start = self._start_var.get()
        duration = self._duration_var.get()
        plot_rows = (start, start + duration)

        # 6. Call prepare_plot_data
        figures = prepare_plot_data(df, config, plot_name, plot_rows, break_times)

        # 7. Update file navigation
        self._file_count = max(len(figures), 1)
        self._file_index = min(self._file_index, max(0, self._file_count - 1))
        self._update_file_nav()

        # 8. Display the figure at current file_index
        if figures:
            filename, fig = figures[self._file_index]
            self._plot_canvas.display_figure(fig)
            # Close figures we're not displaying to free memory
            for i, (_, f) in enumerate(figures):
                if i != self._file_index:
                    plt.close(f)
        else:
            self._plot_canvas.show_message(f"No plottable data for {variant.full_name}")

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
        import matplotlib.pyplot as plt

        comp_dir = self._project_path / "output_parquet_comparison"
        parquet_path = comp_dir / f"{variant.result_key}.parquet"

        if not parquet_path.exists():
            self._plot_canvas.show_message(
                f"No comparison data found for {variant.result_key}\n"
                f"Run 'Scenario comparison' first."
            )
            return

        df = pd.read_parquet(parquet_path)
        if df.empty:
            self._plot_canvas.show_message(f"Empty data for {variant.result_key}")
            return

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

        figures = prepare_plot_data(df, config, plot_name, plot_rows, break_times)

        self._file_count = max(len(figures), 1)
        self._file_index = min(self._file_index, max(0, self._file_count - 1))
        self._update_file_nav()

        if figures:
            filename, fig = figures[self._file_index]
            self._plot_canvas.display_figure(fig)
            for i, (_, f) in enumerate(figures):
                if i != self._file_index:
                    plt.close(f)
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
            df = pd.read_parquet(path)
            if df.empty:
                result: set[str] | None = None
            else:
                result = set(df.iloc[:, 0].astype(str))
            self._break_times_cache["_comparison"] = result
            return result
        except Exception:  # noqa: BLE001
            self._break_times_cache["_comparison"] = None
            return None

    def _populate_comparison_scenarios(self) -> None:
        """Show scenarios from the last comparison run (informational)."""
        import json
        meta_path = self._project_path / "output_parquet_comparison" / "_metadata.json"
        if not meta_path.exists():
            return
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            scenarios = meta.get("scenarios", [])
            self._scenario_listbox.delete(0, "end")
            for name in scenarios:
                self._scenario_listbox.insert("end", name)
            if scenarios:
                self._scenario_listbox.selection_set(0)
        except (json.JSONDecodeError, OSError):
            pass

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

        # Load dispatch groups (the authoritative list of plottable groups)
        dispatch_groups_path = parquet_dir / "outputNodeGroup_does_specified_flows.parquet"
        if not dispatch_groups_path.exists():
            return

        df = pd.read_parquet(dispatch_groups_path)
        if df.empty:
            return

        node_groups = sorted(df['group'].unique().tolist())

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
        files_by_name = collect_parquet_files(scenario_folders, parquet_subdir="")
        combined = combine_parquet_files(files_by_name, num_scenarios=1)
        self._dispatch_results = TimeSeriesResults.from_dict(combined)

        self._dispatch_scenario = scenario
        return True

    def _display_dispatch(self, scenario: str, node_group: str) -> None:
        """Render and display a dispatch plot for a nodeGroup."""
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

        # Get timeline from start/duration controls
        start = self._start_var.get()
        duration = self._duration_var.get()
        timeline = (start, start + duration)

        # Load break times
        break_times = self._load_break_times(scenario)

        # Build figure
        fig = _build_dispatch_figure(
            df_dispatch, inflow,
            title=f"{node_group} \u2014 {scenario}",
            timeline=timeline,
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

        # Save window geometry
        self._viewer_settings.window_geometry = self.geometry()

        # Save comparison scenarios if in comparison mode
        if self._mode.get() == "comparison":
            self._settings.comp_plots_scenarios = self._get_selected_scenarios()

        # Persist all settings
        try:
            save_project_settings(self._project_path, self._settings)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to save viewer settings on close", exc_info=True)

        # Clean up matplotlib resources
        self._plot_canvas.cleanup()

        self.destroy()
