from __future__ import annotations

import logging
import shutil
import tkinter as tk
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
from flextool.gui.data_models import GlobalSettings, ProjectSettings
from flextool.gui.input_sources import InputSourceManager
from flextool.gui.platform_utils import (
    open_file_in_default_app,
    open_spine_db_editor,
)

logger = logging.getLogger(__name__)

# Unicode checkbox characters for Treeview checkbox simulation
CHECK_ON = "\u2611"   # ☑
CHECK_OFF = "\u2610"  # ☐
STATUS_OK = "\u2713"  # ✓
STATUS_ERR = "\u2717"  # ✗
STATUS_EDITING = "\u23f3"  # ⏳


class MainWindow(tk.Tk):
    """Main application window for FlexTool GUI.

    All widgets are created and placed via the grid geometry manager.
    """

    def __init__(self) -> None:
        super().__init__()
        self.title("FlexTool")

        # ── State ──────────────────────────────────────────────────
        self.current_project: str | None = None
        self.global_settings = GlobalSettings()
        self.project_settings = ProjectSettings()
        self.input_source_mgr: InputSourceManager | None = None

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
        self.project_combo.bind("<<ComboboxSelected>>", self._on_project_combo_selected)
        self.project_combo.bind("<F2>", self._on_combo_rename)
        self.project_combo.bind("<Double-Button-1>", self._on_combo_rename)

        self.project_menu_btn = ttk.Button(
            outer, text="Project menu", command=self._on_project_menu_btn
        )
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

        # ── Window close handler ─────────────────────────────────────
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── Startup logic ────────────────────────────────────────────
        self._startup()

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
            # No valid recent project -- show project dialog after mainloop starts
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

    # ── Project switching ────────────────────────────────────────────

    def _switch_project(self, name: str) -> None:
        """Switch to the project with the given *name*."""
        self.current_project = name

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

        # Create input source manager and populate treeviews
        self.input_source_mgr = InputSourceManager(project_path, self.project_settings)
        self._clear_all_lists()
        self._refresh_input_sources()

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
        if self.current_project:
            self.global_settings.recent_project = self.current_project
            save_global_settings(get_projects_dir(), self.global_settings)
        self.destroy()

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
                self._update_available_scenarios()
                self._update_input_button_states()

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

    # ── Input source management ──────────────────────────────────────

    def _on_add_source(self) -> None:
        """Open the Add dialog and refresh sources if files were added."""
        if not self.current_project:
            messagebox.showinfo("No project", "No project is currently loaded.")
            return

        from flextool.gui.dialogs.add_dialog import AddDialog

        project_path = get_projects_dir() / self.current_project
        dlg = AddDialog(self, project_path)
        if dlg.result:
            self._refresh_input_sources()

    def _on_refresh_sources(self) -> None:
        """Refresh input sources by re-scanning the directory."""
        if not self.input_source_mgr:
            return
        self._refresh_input_sources()

    def _refresh_input_sources(self) -> None:
        """Re-scan input sources and repopulate the treeview."""
        if not self.input_source_mgr:
            return

        sources = self.input_source_mgr.refresh()

        # Clear input sources tree
        for item in self.input_sources_tree.get_children():
            self.input_sources_tree.delete(item)

        # Configure tag for error rows
        self.input_sources_tree.tag_configure("error", background="#ffcccc")

        # Populate input sources tree
        for source in sources:
            if source.status == "ok":
                status_char = STATUS_OK
            elif source.status == "editing":
                status_char = STATUS_EDITING
            else:
                status_char = STATUS_ERR

            tags = ("error",) if source.status == "error" else ()
            self.input_sources_tree.insert(
                "",
                "end",
                values=(CHECK_ON, source.name, source.number, status_char),
                tags=tags,
            )

        # Update Add button appearance based on whether there are sources
        self._update_add_button_style(len(sources) == 0)

        # Update available scenarios
        self._update_available_scenarios()

        # Update Edit / Convert / Delete button states
        self._update_input_button_states()

    def _update_add_button_style(self, no_sources: bool) -> None:
        """Highlight the Add button in green when there are no input sources."""
        if no_sources:
            # Use a custom style with green background
            style = ttk.Style()
            style.configure("Green.TButton", background="#90ee90")
            self.add_source_btn.configure(style="Green.TButton")
        else:
            self.add_source_btn.configure(style="TButton")

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
            return

        selected_sources = self._get_selected_source_names()
        # If nothing is selected, show all scenarios
        if not selected_sources:
            scenarios = self.input_source_mgr.get_all_scenarios()
        else:
            scenarios = self.input_source_mgr.get_all_scenarios(selected_sources)

        for scenario in scenarios:
            self.available_tree.insert(
                "",
                "end",
                values=(CHECK_OFF, scenario.source_number, scenario.name),
            )

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

    def _update_input_button_states(self) -> None:
        """Enable or disable Edit, Convert, Delete based on current selection."""
        checked = self._get_checked_sources()

        # ── Edit: exactly one checked, not in editing state ──
        if len(checked) == 1:
            _name, status = checked[0]
            if status == STATUS_EDITING:
                self.edit_source_btn.configure(state="disabled")
            else:
                self.edit_source_btn.configure(state="normal")
        else:
            self.edit_source_btn.configure(state="disabled")

        # ── Convert: exactly one checked, xlsx, status OK ──
        if len(checked) == 1:
            name, status = checked[0]
            is_xlsx = name.lower().endswith(".xlsx")
            if is_xlsx and status == STATUS_OK:
                self.convert_source_btn.configure(state="normal")
            else:
                self.convert_source_btn.configure(state="disabled")
        else:
            self.convert_source_btn.configure(state="disabled")

        # ── Delete: at least one checked ──
        if checked:
            self.delete_source_btn.configure(state="normal")
        else:
            self.delete_source_btn.configure(state="disabled")

    # ── Edit button handler ─────────────────────────────────────────

    def _on_edit_source(self) -> None:
        """Open the selected input source for editing."""
        if not self.input_source_mgr or not self.current_project:
            return

        checked = self._get_checked_sources()
        if len(checked) != 1:
            return

        source_name, _status = checked[0]
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
            proc = open_spine_db_editor(db_url)
            if proc is None:
                messagebox.showinfo(
                    "spine-db-editor not found",
                    "The spine-db-editor command was not found on your system.\n\n"
                    "Install it with:  pip install spine-db-editor",
                )
                return
        else:
            messagebox.showinfo("Unsupported", f"Cannot edit files of type '{ext}'.")
            return

        # Refresh to show editing status
        self._refresh_input_sources()

    # ── Convert button handler ──────────────────────────────────────

    def _on_convert_source(self) -> None:
        """Convert the selected xlsx input source to a sqlite database."""
        if not self.input_source_mgr or not self.current_project:
            return

        checked = self._get_checked_sources()
        if len(checked) != 1:
            return

        source_name, _status = checked[0]
        if not source_name.lower().endswith(".xlsx"):
            return

        answer = messagebox.askokcancel(
            "Convert to database",
            "Are you sure you want to convert the selected xlsx input source "
            "to a database input source? xlsx will be copied to 'converted' "
            "folder for safekeeping.",
        )
        if not answer:
            return

        project_path = get_projects_dir() / self.current_project
        input_dir = project_path / "input_sources"
        xlsx_path = input_dir / source_name

        if not xlsx_path.exists():
            messagebox.showerror("File not found", f"Cannot find:\n{xlsx_path}")
            return

        # Determine target sqlite path
        stem = Path(source_name).stem
        target_sqlite = input_dir / f"{stem}.sqlite"
        target_db_url = f"sqlite:///{target_sqlite}"

        try:
            # Import conversion utilities
            from flextool.process_inputs.read_tabular_with_specification import (
                TabularReader,
            )
            from flextool.process_inputs.write_to_input_db import (
                write_to_flextool_input_db,
            )

            # Locate the specification JSON
            cli_dir = Path(__file__).resolve().parent.parent / "cli"
            json_path = str(cli_dir / ".." / "import_excel_input.json")

            tabular_reader = TabularReader(json_path)
            write_to_flextool_input_db(
                str(xlsx_path), tabular_reader, target_db_url, input_type="excel"
            )
        except Exception as exc:
            logger.error("Conversion failed: %s", exc, exc_info=True)
            messagebox.showerror(
                "Conversion failed",
                f"An error occurred during conversion:\n{exc}",
            )
            return

        # Move xlsx to converted/ folder
        converted_dir = project_path / "converted"
        converted_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(xlsx_path), str(converted_dir / source_name))
        except OSError as exc:
            messagebox.showwarning(
                "Move failed",
                f"Conversion succeeded but the xlsx could not be moved "
                f"to the 'converted' folder:\n{exc}",
            )

        # Remove old xlsx from input_source_numbers if present
        if source_name in self.project_settings.input_source_numbers:
            del self.project_settings.input_source_numbers[source_name]
            save_project_settings(project_path, self.project_settings)

        self._refresh_input_sources()
        messagebox.showinfo(
            "Conversion complete",
            f"'{source_name}' has been converted to '{stem}.sqlite'.\n"
            f"The xlsx has been moved to the 'converted' folder.",
        )

    # ── Delete button handler ───────────────────────────────────────

    def _on_delete_source(self) -> None:
        """Delete the selected input source file(s)."""
        if not self.input_source_mgr or not self.current_project:
            return

        checked = self._get_checked_sources()
        if not checked:
            return

        project_path = get_projects_dir() / self.current_project
        names = [name for name, _ in checked]
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
        self._refresh_input_sources()
