from __future__ import annotations

import tkinter as tk
from tkinter import ttk

# Unicode checkbox characters for Treeview checkbox simulation
CHECK_ON = "\u2611"   # ☑
CHECK_OFF = "\u2610"  # ☐
STATUS_OK = "\u2713"  # ✓
STATUS_ERR = "\u2717"  # ✗


class MainWindow(tk.Tk):
    """Main application window for FlexTool GUI.

    All widgets are created and placed via the grid geometry manager.
    No commands are wired -- buttons and controls are inert placeholders.
    """

    def __init__(self) -> None:
        super().__init__()
        self.title("FlexTool")

        # ── Window sizing ────────────────────────────────────────────
        min_width = 1100
        min_height = 700
        screen_h = self.winfo_screenheight()
        # Leave a small margin at top/bottom for taskbars
        win_height = max(min_height, screen_h - 80)
        self.geometry(f"{min_width}x{win_height}+0+0")
        self.minsize(min_width, min_height)

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
        outer.columnconfigure(2, weight=0)   # spacer / auto-generate
        outer.columnconfigure(3, weight=0)   # auto-generate / plot & exec menus
        outer.columnconfigure(4, weight=0)   # output status labels
        outer.columnconfigure(5, weight=0)   # Show/Open buttons
        outer.columnconfigure(6, weight=1)   # executed scenarios

        # ── Row 0: Project selector ──────────────────────────────────
        row = 0
        ttk.Label(outer, text="Project:").grid(row=row, column=0, sticky="w", padx=(0, 5))

        self.project_combo = ttk.Combobox(outer, state="readonly", width=30)
        self.project_combo.grid(row=row, column=0, sticky="w", padx=(60, 5))

        self.project_menu_btn = ttk.Button(outer, text="Project menu")
        self.project_menu_btn.grid(row=row, column=1, sticky="w", padx=5)

        # ── Row 1: Section headers ───────────────────────────────────
        row = 1
        ttk.Label(outer, text="Input sources", font=("", 10, "bold")).grid(
            row=row, column=0, sticky="sw", pady=(10, 2)
        )
        ttk.Label(outer, text="Auto-generate", font=("", 10, "bold")).grid(
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
            selectmode="browse",
            height=8,
        )
        self.input_sources_tree.heading("check", text="")
        self.input_sources_tree.heading("name", text="Name")
        self.input_sources_tree.heading("number", text="#")
        self.input_sources_tree.heading("status", text="")
        self.input_sources_tree.column("check", width=30, minwidth=30, stretch=False)
        self.input_sources_tree.column("name", width=180, minwidth=100)
        self.input_sources_tree.column("number", width=30, minwidth=30, stretch=False)
        self.input_sources_tree.column("status", width=30, minwidth=30, stretch=False)
        self.input_sources_tree.grid(row=0, column=0, sticky="nsew")

        input_scroll = ttk.Scrollbar(input_frame, orient="vertical", command=self.input_sources_tree.yview)
        self.input_sources_tree.configure(yscrollcommand=input_scroll.set)
        input_scroll.grid(row=0, column=1, sticky="ns")

        # Bind click for checkbox toggling
        self.input_sources_tree.bind("<Button-1>", self._on_input_source_click)

        # --- Input source buttons (col 1, rows 2-8) ---
        btn_col = 1
        self.add_source_btn = ttk.Button(outer, text="Add", width=8)
        self.add_source_btn.grid(row=2, column=btn_col, sticky="nw", padx=5, pady=2)

        self.edit_source_btn = ttk.Button(outer, text="Edit", width=8)
        self.edit_source_btn.grid(row=4, column=btn_col, sticky="nw", padx=5, pady=2)

        self.convert_source_btn = ttk.Button(outer, text="Convert", width=8)
        self.convert_source_btn.grid(row=5, column=btn_col, sticky="nw", padx=5, pady=2)

        self.delete_source_btn = ttk.Button(outer, text="Delete", width=8)
        self.delete_source_btn.grid(row=6, column=btn_col, sticky="nw", padx=5, pady=2)

        self.refresh_btn = ttk.Button(outer, text="Refresh", width=8)
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

        # --- Plot menu and Execution menu buttons (col 2-3, rows 7-8) ---
        self.plot_menu_btn = ttk.Button(outer, text="Plot menu", width=14)
        self.plot_menu_btn.grid(row=7, column=2, columnspan=2, sticky="nw", padx=(20, 10), pady=2)

        self.execution_menu_btn = ttk.Button(outer, text="Execution menu", width=14)
        self.execution_menu_btn.grid(row=8, column=2, columnspan=2, sticky="nw", padx=(20, 10), pady=2)

        # --- Output status labels + Show/Open buttons (cols 4-5, rows 2-6) ---
        output_info = [
            ("Scen. plots", "scen_plots"),
            ("Scen. Excel", "scen_excel"),
            ("Scen. csvs", "scen_csvs"),
            ("Comp. plots", "comp_plots"),
            ("Comp. Excel", "comp_excel"),
        ]
        # "Show" for plots/csvs, "Open" for Excel
        action_labels = ["Show", "Open", "Show", "Show", "Open"]

        self.output_status_labels: dict[str, ttk.Label] = {}
        self.output_action_btns: dict[str, ttk.Button] = {}

        for i, ((label_text, key), action_text) in enumerate(zip(output_info, action_labels)):
            status_label = ttk.Label(
                outer, text=f"{label_text} {STATUS_ERR}", width=16, anchor="w"
            )
            status_label.grid(row=2 + i, column=4, sticky="w", padx=(10, 5), pady=2)
            self.output_status_labels[key] = status_label

            action_btn = ttk.Button(outer, text=action_text, width=5)
            action_btn.grid(row=2 + i, column=5, sticky="w", padx=2, pady=2)
            self.output_action_btns[key] = action_btn

        # ── Separator ────────────────────────────────────────────────
        sep = ttk.Separator(outer, orient="horizontal")
        sep.grid(row=9, column=0, columnspan=7, sticky="ew", pady=10)

        # ── Row 10: Scenario section headers ─────────────────────────
        row = 10
        ttk.Label(outer, text="Available scenarios", font=("", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="sw", pady=(0, 2)
        )
        ttk.Label(outer, text="Executed scenarios", font=("", 10, "bold")).grid(
            row=row, column=2, columnspan=5, sticky="sw", padx=(20, 0), pady=(0, 2)
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
        self.available_tree.heading("check", text="")
        self.available_tree.heading("source_num", text="#")
        self.available_tree.heading("scenario_name", text="Scenario")
        self.available_tree.column("check", width=30, minwidth=30, stretch=False)
        self.available_tree.column("source_num", width=30, minwidth=30, stretch=False)
        self.available_tree.column("scenario_name", width=200, minwidth=100)
        self.available_tree.grid(row=0, column=0, sticky="nsew")

        avail_scroll = ttk.Scrollbar(avail_frame, orient="vertical", command=self.available_tree.yview)
        self.available_tree.configure(yscrollcommand=avail_scroll.set)
        avail_scroll.grid(row=0, column=1, sticky="ns")

        self.available_tree.bind("<Button-1>", self._on_available_click)

        # ── Row 11: Executed scenarios Treeview ──────────────────────
        exec_frame = ttk.Frame(outer)
        exec_frame.grid(row=row, column=2, columnspan=5, sticky="nsew", padx=(20, 0))
        exec_frame.columnconfigure(0, weight=1)
        exec_frame.rowconfigure(0, weight=1)

        self.executed_tree = ttk.Treeview(
            exec_frame,
            columns=("check", "source_num", "scenario_name", "timestamp"),
            show="headings",
            selectmode="extended",
            height=8,
        )
        self.executed_tree.heading("check", text="")
        self.executed_tree.heading("source_num", text="#")
        self.executed_tree.heading("scenario_name", text="Scenario")
        self.executed_tree.heading("timestamp", text="Timestamp")
        self.executed_tree.column("check", width=30, minwidth=30, stretch=False)
        self.executed_tree.column("source_num", width=30, minwidth=30, stretch=False)
        self.executed_tree.column("scenario_name", width=180, minwidth=100)
        self.executed_tree.column("timestamp", width=130, minwidth=100)
        self.executed_tree.grid(row=0, column=0, sticky="nsew")

        exec_scroll = ttk.Scrollbar(exec_frame, orient="vertical", command=self.executed_tree.yview)
        self.executed_tree.configure(yscrollcommand=exec_scroll.set)
        exec_scroll.grid(row=0, column=1, sticky="ns")

        self.executed_tree.bind("<Button-1>", self._on_executed_click)

        # ── Row 12: Bottom action buttons ────────────────────────────
        row = 12
        bottom_left = ttk.Frame(outer)
        bottom_left.grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 0))

        self.add_to_execution_btn = ttk.Button(
            bottom_left, text="Add selected to\nthe execution list"
        )
        self.add_to_execution_btn.grid(row=0, column=0, padx=(0, 10))

        move_frame = ttk.Frame(bottom_left)
        move_frame.grid(row=0, column=1, padx=10)

        self.move_up_btn = ttk.Button(move_frame, text="\u25b2", width=3)
        self.move_up_btn.grid(row=0, column=1, padx=2)

        self.move_down_btn = ttk.Button(move_frame, text="\u25bc", width=3)
        self.move_down_btn.grid(row=0, column=2, padx=2)

        self.move_label = ttk.Label(move_frame, text="Move\nselected")
        self.move_label.grid(row=0, column=0, padx=(0, 4))

        bottom_right = ttk.Frame(outer)
        bottom_right.grid(row=row, column=2, columnspan=5, sticky="e", pady=(8, 0))

        self.delete_results_btn = ttk.Button(
            bottom_right, text="Delete selected\nresults irrevocably"
        )
        self.delete_results_btn.grid(row=0, column=0)

    # ── Treeview checkbox toggle handlers ────────────────────────────
    # These detect a click on the "check" column and toggle the character.

    def _toggle_check(self, tree: ttk.Treeview, item: str, col: str) -> None:
        """Toggle a checkbox character in *tree* for *item* in *col*."""
        current = tree.set(item, col)
        new_value = CHECK_OFF if current == CHECK_ON else CHECK_ON
        tree.set(item, col, new_value)

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
