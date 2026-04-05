"""ResultViewer — Toplevel window for browsing and displaying result plots."""

from __future__ import annotations

import logging
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import ttk

from flextool.gui.data_models import ProjectSettings
from flextool.gui.network_graph import build_network_figure
from flextool.gui.plot_canvas import PlotCanvas
from flextool.gui.plot_config_reader import PlotEntry, PlotGroup, PlotVariant, parse_plot_config
from flextool.gui.project_utils import get_projects_dir

logger = logging.getLogger(__name__)

# Maximum characters for tree entry labels before truncation
_TREE_LABEL_LIMIT = 25


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
        # Currently active variant letter
        self._active_variant: str = ""
        # List of variant buttons currently displayed
        self._variant_buttons: list[ttk.Button] = []
        # Tooltip toplevel
        self._tooltip: tk.Toplevel | None = None

        # Mode variable: "single", "comparison", "network"
        self._mode = tk.StringVar(value=self._viewer_settings.last_mode or "single")

        # File navigation state
        self._file_index = 0
        self._file_count = 1

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

        # Override Up/Down to skip disabled entries
        self._plot_tree.bind("<Up>", self._on_tree_key_up)
        self._plot_tree.bind("<Down>", self._on_tree_key_down)

    def _build_right_column(self) -> None:
        """Build the right column: nav bar, variant panel, plot placeholder."""
        right = ttk.Frame(self._paned, padding=5)
        self._paned.add(right, weight=1)

        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)  # plot area gets all extra space

        # ── Navigation bar ───────────────────────────────────────────
        nav_frame = ttk.LabelFrame(right, text="Navigation", padding=5)
        nav_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        nav_frame.columnconfigure(1, weight=1)

        # Row 0: File navigation
        self._prev_file_btn = ttk.Button(
            nav_frame, text="\u25c0 Prev", width=8,
            command=self._on_prev_file,
        )
        self._prev_file_btn.grid(row=0, column=0, padx=(0, 5))

        self._file_label = ttk.Label(nav_frame, text="File 1/1", anchor="center")
        self._file_label.grid(row=0, column=1, sticky="ew")

        self._next_file_btn = ttk.Button(
            nav_frame, text="Next \u25b6", width=8,
            command=self._on_next_file,
        )
        self._next_file_btn.grid(row=0, column=2, padx=(5, 0))

        self._update_file_nav()

        # Row 1: Start slider + Duration spinbox
        ttk.Label(nav_frame, text="Start:").grid(row=1, column=0, sticky="w", pady=(5, 0))

        self._start_var = tk.IntVar(value=self._settings.single_plot_settings.start_time)
        self._start_scale = ttk.Scale(
            nav_frame, from_=0, to=8760, orient="horizontal",
            variable=self._start_var,
        )
        self._start_scale.grid(row=1, column=1, sticky="ew", padx=5, pady=(5, 0))

        dur_frame = ttk.Frame(nav_frame)
        dur_frame.grid(row=1, column=2, sticky="e", pady=(5, 0))
        ttk.Label(dur_frame, text="Duration:").pack(side="left", padx=(0, 3))
        self._duration_var = tk.IntVar(value=self._settings.single_plot_settings.duration or 168)
        self._duration_spin = ttk.Spinbox(
            dur_frame, from_=1, to=8760, textvariable=self._duration_var, width=6,
        )
        self._duration_spin.pack(side="left")

        # Row 2: Mode radio buttons + Refresh
        mode_frame = ttk.Frame(nav_frame)
        mode_frame.grid(row=2, column=0, columnspan=2, sticky="w", pady=(5, 0))

        for text, value in [("Single", "single"), ("Comparison", "comparison"), ("Network", "network")]:
            rb = ttk.Radiobutton(
                mode_frame, text=text, variable=self._mode, value=value,
                command=self._on_mode_changed,
            )
            rb.pack(side="left", padx=(0, 10))

        self._refresh_btn = ttk.Button(
            nav_frame, text="Refresh", width=8,
            command=self._on_refresh,
        )
        self._refresh_btn.grid(row=2, column=2, sticky="e", pady=(5, 0))

        # ── Variant panel ────────────────────────────────────────────
        self._variant_frame = ttk.LabelFrame(right, text="Variant", padding=(5, 2))
        self._variant_frame.grid(row=1, column=0, sticky="ew", pady=(0, 5))
        # Placeholder label shown when no variants are available
        self._variant_placeholder = ttk.Label(
            self._variant_frame, text="Select a plot entry", foreground="grey",
        )
        self._variant_placeholder.pack(side="left")

        # ── Plot canvas ──────────────────────────────────────────────
        self._plot_canvas = PlotCanvas(right)
        self._plot_canvas.grid(row=2, column=0, sticky="nsew")
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
        if mode == "network":
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
                label = f"{entry.number} {entry.short_name}"
                self._plot_tree.insert(
                    group_iid, "end", iid=entry_iid, text=label,
                )
                self._tree_entry_map[entry_iid] = entry

        # Grey out entries without matching parquet data
        self._update_tree_availability()

        # Try to restore last selected entry or select first available
        self._restore_or_select_first_entry()

    def _update_tree_availability(self) -> None:
        """Grey out entries that have no matching parquet files for selected scenario(s)."""
        scenarios = self._get_selected_scenarios()
        if not scenarios:
            # No scenario selected -- mark all as disabled
            for iid in self._tree_entry_map:
                self._plot_tree.item(iid, tags=("disabled",))
            return

        # Collect available parquet file stems for selected scenarios
        available_keys: set[str] = set()
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

        if self._mode.get() == "network":
            self._render_network()
            return

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

        # If a group header is selected, do nothing (don't update variant panel)
        if iid.startswith("group_"):
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

        # Update variant panel
        self._populate_variant_panel(entry)

        # Display the plot
        self._trigger_replot()

    # ------------------------------------------------------------------
    # Tree keyboard navigation (skip disabled entries)
    # ------------------------------------------------------------------

    def _on_tree_key_up(self, event: tk.Event) -> str:
        """Move to previous non-disabled entry, skipping disabled ones."""
        selection = self._plot_tree.selection()
        if not selection:
            return "break"

        current = selection[0]
        prev_item = self._plot_tree.prev(current)

        # Walk up, skipping disabled entries
        while prev_item:
            if prev_item.startswith("entry_") and not self._is_entry_disabled(prev_item):
                self._plot_tree.selection_set(prev_item)
                self._plot_tree.see(prev_item)
                self._plot_tree.event_generate("<<TreeviewSelect>>")
                return "break"
            # If it's a group, try the last child of the previous group
            if prev_item.startswith("group_"):
                children = self._plot_tree.get_children(prev_item)
                for child in reversed(children):
                    if not self._is_entry_disabled(child):
                        self._plot_tree.selection_set(child)
                        self._plot_tree.see(child)
                        self._plot_tree.event_generate("<<TreeviewSelect>>")
                        return "break"
            prev_item = self._plot_tree.prev(prev_item)

        # If current is inside a group, try entries above in the same group
        parent = self._plot_tree.parent(current)
        if parent:
            siblings = list(self._plot_tree.get_children(parent))
            idx = siblings.index(current) if current in siblings else -1
            for i in range(idx - 1, -1, -1):
                if not self._is_entry_disabled(siblings[i]):
                    self._plot_tree.selection_set(siblings[i])
                    self._plot_tree.see(siblings[i])
                    self._plot_tree.event_generate("<<TreeviewSelect>>")
                    return "break"

            # Try previous groups
            group_prev = self._plot_tree.prev(parent)
            while group_prev:
                if group_prev.startswith("group_"):
                    children = self._plot_tree.get_children(group_prev)
                    for child in reversed(children):
                        if not self._is_entry_disabled(child):
                            self._plot_tree.selection_set(child)
                            self._plot_tree.see(child)
                            self._plot_tree.event_generate("<<TreeviewSelect>>")
                            return "break"
                group_prev = self._plot_tree.prev(group_prev)

        return "break"

    def _on_tree_key_down(self, event: tk.Event) -> str:
        """Move to next non-disabled entry, skipping disabled ones."""
        selection = self._plot_tree.selection()
        if not selection:
            # Select first available
            self._restore_or_select_first_entry()
            return "break"

        current = selection[0]

        # If current is inside a group, try entries below in the same group
        parent = self._plot_tree.parent(current)
        if parent:
            siblings = list(self._plot_tree.get_children(parent))
            idx = siblings.index(current) if current in siblings else -1
            for i in range(idx + 1, len(siblings)):
                if not self._is_entry_disabled(siblings[i]):
                    self._plot_tree.selection_set(siblings[i])
                    self._plot_tree.see(siblings[i])
                    self._plot_tree.event_generate("<<TreeviewSelect>>")
                    return "break"

        # Try next groups
        if parent:
            next_group = self._plot_tree.next(parent)
        else:
            next_group = self._plot_tree.next(current)

        while next_group:
            if next_group.startswith("group_"):
                children = self._plot_tree.get_children(next_group)
                for child in children:
                    if not self._is_entry_disabled(child):
                        self._plot_tree.selection_set(child)
                        self._plot_tree.see(child)
                        self._plot_tree.event_generate("<<TreeviewSelect>>")
                        return "break"
            elif next_group.startswith("entry_") and not self._is_entry_disabled(next_group):
                self._plot_tree.selection_set(next_group)
                self._plot_tree.see(next_group)
                self._plot_tree.event_generate("<<TreeviewSelect>>")
                return "break"
            next_group = self._plot_tree.next(next_group)

        return "break"

    # ------------------------------------------------------------------
    # Tree tooltip
    # ------------------------------------------------------------------

    def _on_tree_motion(self, event: tk.Event) -> None:
        """Show tooltip with full name when hovering over a truncated entry."""
        item = self._plot_tree.identify_row(event.y)
        if not item or not item.startswith("entry_"):
            self._hide_tooltip()
            return

        entry = self._tree_entry_map.get(item)
        if entry is None or entry.full_name == entry.short_name:
            self._hide_tooltip()
            return

        # Show tooltip
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

        self._tooltip_label = ttk.Label(
            self._tooltip, text=full_text,
            background="#ffffe0", relief="solid", borderwidth=1,
            padding=(4, 2),
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

    def _populate_variant_panel(self, entry: PlotEntry) -> None:
        """Repopulate the variant panel with buttons for the given entry."""
        # Clear existing buttons
        for btn in self._variant_buttons:
            btn.destroy()
        self._variant_buttons.clear()
        self._variant_placeholder.pack_forget()

        if not entry.variants:
            self._variant_placeholder.configure(text="No variants available")
            self._variant_placeholder.pack(side="left")
            self._active_variant = ""
            return

        # Determine which variant to select
        target_variant = self._viewer_settings.last_variant
        available_letters = [v.letter for v in entry.variants]
        if target_variant not in available_letters:
            target_variant = available_letters[0]

        self._active_variant = target_variant
        self._viewer_settings.last_variant = target_variant

        for variant in entry.variants:
            letter = variant.letter or "?"
            btn = ttk.Button(
                self._variant_frame,
                text=letter,
                width=3,
                command=lambda v=variant.letter: self._on_variant_clicked(v),
            )
            btn.pack(side="left", padx=2, pady=1)
            self._variant_buttons.append(btn)

        self._highlight_active_variant()

        # Bind Left/Right for variant navigation
        for btn in self._variant_buttons:
            btn.bind("<Left>", self._on_variant_left)
            btn.bind("<Right>", self._on_variant_right)
            btn.bind("<Tab>", self._focus_scenario_listbox)

    def _highlight_active_variant(self) -> None:
        """Visually highlight the active variant button."""
        for btn in self._variant_buttons:
            letter = btn.cget("text")
            if letter == self._active_variant or (not self._active_variant and letter == "?"):
                btn.configure(style="Accent.TButton")
            else:
                btn.configure(style="TButton")

    def _on_variant_clicked(self, letter: str) -> None:
        """Handle variant button click."""
        self._active_variant = letter
        self._viewer_settings.last_variant = letter
        self._highlight_active_variant()
        # Reset file navigation
        self._file_index = 0
        self._trigger_replot()

    def _on_variant_left(self, event: tk.Event) -> str:
        """Navigate to previous variant button."""
        if not self._variant_buttons:
            return "break"
        current_idx = self._get_focused_variant_index()
        if current_idx > 0:
            new_idx = current_idx - 1
            self._variant_buttons[new_idx].focus_set()
            letter = self._variant_buttons[new_idx].cget("text")
            self._on_variant_clicked(letter)
        return "break"

    def _on_variant_right(self, event: tk.Event) -> str:
        """Navigate to next variant button."""
        if not self._variant_buttons:
            return "break"
        current_idx = self._get_focused_variant_index()
        if current_idx < len(self._variant_buttons) - 1:
            new_idx = current_idx + 1
            self._variant_buttons[new_idx].focus_set()
            letter = self._variant_buttons[new_idx].cget("text")
            self._on_variant_clicked(letter)
        return "break"

    def _get_focused_variant_index(self) -> int:
        """Return index of the currently focused variant button, or 0."""
        focused = self.focus_get()
        for i, btn in enumerate(self._variant_buttons):
            if btn is focused:
                return i
        # Fall back to active variant
        for i, btn in enumerate(self._variant_buttons):
            if btn.cget("text") == self._active_variant:
                return i
        return 0

    def _show_variant_panel(self) -> None:
        """Show the variant panel."""
        self._variant_frame.grid()

    def _hide_variant_panel(self) -> None:
        """Hide the variant panel and clear its content."""
        for btn in self._variant_buttons:
            btn.destroy()
        self._variant_buttons.clear()
        self._active_variant = ""
        self._variant_placeholder.configure(text="")
        self._variant_placeholder.pack(side="left")

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
            self._scenario_listbox.configure(selectmode="extended")
            self._populate_plot_tree()
            self._show_variant_panel()
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
                if btn.cget("text") == self._active_variant:
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
        """Return the PlotVariant matching the active variant letter."""
        for v in entry.variants:
            if v.letter == self._active_variant:
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

    def _trigger_replot(self) -> None:
        """Called when scenario, entry, or variant changes."""
        scenarios = self._get_selected_scenarios()
        if not scenarios:
            self._plot_canvas.show_message("No scenario selected")
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

        # Find PNG files
        png_files = self._find_png_files(scenarios[0], entry, variant)
        self._file_count = max(len(png_files), 1)
        self._file_index = min(self._file_index, max(0, self._file_count - 1))
        self._update_file_nav()

        if png_files:
            self._plot_canvas.display_png(png_files[self._file_index])
        else:
            self._plot_canvas.show_message(
                f"No plot files found for\n{variant.full_name}"
            )

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
        """Handle window close — persist settings."""
        self._hide_tooltip()
        self.destroy()
