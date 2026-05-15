"""Non-modal Toplevel window for managing FlexTool scenario executions."""

from __future__ import annotations

import logging
import os
import tkinter as tk
from tkinter import ttk, messagebox

from flextool.gui.execution_manager import ExecutionJob, ExecutionManager, JobStatus, JobType
from flextool.gui.hover_tooltip import attach_tooltip

logger = logging.getLogger(__name__)

# Status icons for the scenarios list
_STATUS_ICONS: dict[JobStatus, str] = {
    JobStatus.SUCCESS: "\u2713",   # ✓
    JobStatus.FAILED: "\u2717",    # ✗
    JobStatus.RUNNING: "\u23f3",   # ⏳
    JobStatus.PENDING: "\u2610",   # ☐
    JobStatus.KILLED: "\u2717",    # ✗
}


def _format_status_text(status: dict) -> str:
    """Format the dispatcher status dict as a single-line label."""
    base = (
        f"{status['running']}/{status['max_threads']} threads | "
        f"{status['used_gb']:.1f}/{status['total_gb']:.1f} GB"
    )
    if status['pending'] > 0:
        if status['thread_limited']:
            base += " | Thread limited"
        elif status['memory_limited']:
            base += " | Memory limited"
    return base


class ExecutionWindow(tk.Toplevel):
    """Non-modal window for monitoring and controlling scenario executions.

    This window coexists with the main window -- it never calls
    ``grab_set()`` or ``wait_window()``.
    """

    def __init__(
        self,
        parent: tk.Tk,
        execution_mgr: ExecutionManager,
        global_settings=None,
    ) -> None:
        super().__init__(parent)
        self.title("Execution Jobs")

        self._mgr = execution_mgr
        self._global_settings = global_settings
        self._viewed_job_id: int | None = None
        # Track how many stdout lines we have already displayed per job
        # so that we only append new lines on each poll cycle.
        self._displayed_counts: dict[int, int] = {}
        # Guard flag to suppress <<TreeviewSelect>> events fired by
        # programmatic selection_set inside _refresh_job_list.
        self._refreshing_list: bool = False

        # ── Font metrics for DPI-aware sizing ────────────────────────
        from flextool.gui.ui_metrics import get_metrics
        _metrics = get_metrics(self)
        cw: int = _metrics.cw
        lh: int = _metrics.lh
        self._char_width = cw
        # Use the named-font string so live size changes reach tk.Text.
        self._mono_font = "TkFixedFont"

        # ── Window sizing & positioning ────────────────────────────────
        self._line_height = lh
        self.minsize(cw * 70, lh * 20)

        # Get main window position and dimensions
        # Note: parent is the MainWindow (tk.Tk instance)
        # We need to call update_idletasks() on parent to get accurate geometry
        parent.update_idletasks()
        main_x = parent.winfo_x()
        main_y = parent.winfo_y()
        main_w = parent.winfo_width()
        screen_w = parent.winfo_screenwidth()
        screen_h = parent.winfo_screenheight()

        # Account for taskbar (estimate)
        taskbar_margin = self._line_height * 4 if hasattr(self, '_line_height') else 80
        usable_h = screen_h - taskbar_margin

        if screen_w < 1920:
            # Small screen: full screen, overlap main window
            self.geometry(f"{screen_w}x{usable_h}+0+0")
        else:
            # Large screen: right of main window, touching but not overlapping
            exec_x = main_x + main_w
            exec_w = max(screen_w - exec_x, 400)  # minimum 400px wide
            self.geometry(f"{exec_w}x{usable_h}+{exec_x}+0")

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # ── Top row: max parallel executions ─────────────────────────
        top_frame = ttk.Frame(self, padding=(10, 5))
        top_frame.grid(row=0, column=0, sticky="e")

        ttk.Label(top_frame, text="Max. parallel executions:").pack(side="left", padx=(0, 5))

        cpu = os.cpu_count() or 2
        default_workers = max(1, cpu - 1)
        # Prefer the persisted value (projects.yaml) if one is stored
        if self._global_settings and self._global_settings.max_workers > 0:
            initial = max(1, min(cpu * 2, self._global_settings.max_workers))
            self._mgr.max_workers = initial
        else:
            initial = default_workers
        self._max_workers_var = tk.IntVar(value=initial)
        self._max_workers_spin = ttk.Spinbox(
            top_frame,
            from_=1,
            to=cpu * 2,
            textvariable=self._max_workers_var,
            width=4,
            command=self._on_max_workers_changed,
        )
        self._max_workers_spin.pack(side="left")
        # Sync on Enter key and focus-out (command only fires on arrow clicks)
        self._max_workers_spin.bind("<Return>", lambda e: self._on_max_workers_changed())
        self._max_workers_spin.bind("<FocusOut>", lambda e: self._on_max_workers_changed())
        # Also sync the spinbox to the current manager value
        self._max_workers_var.set(self._mgr.max_workers)

        # Cores per job
        ttk.Label(top_frame, text="  Cores per job:").pack(side="left", padx=(15, 5))

        limits = self._global_settings.execution_limits if self._global_settings else None
        initial_cores = limits.max_cores_per_job if (limits and limits.max_cores_per_job > 0) else 1
        self._cores_per_job_var = tk.IntVar(value=initial_cores)
        self._cores_per_job_spin = ttk.Spinbox(
            top_frame,
            from_=1,
            to=cpu * 2,
            textvariable=self._cores_per_job_var,
            width=4,
            command=self._on_cores_per_job_changed,
        )
        self._cores_per_job_spin.pack(side="left")
        self._cores_per_job_spin.bind("<Return>", lambda e: self._on_cores_per_job_changed())
        self._cores_per_job_spin.bind("<FocusOut>", lambda e: self._on_cores_per_job_changed())

        # Memory budget per job (soft target; used to pick a victim under global pressure)
        mem_label = ttk.Label(top_frame, text="  Memory budget/job (GB, 0=auto):")
        mem_label.pack(side="left", padx=(15, 5))

        initial_cap = limits.memory_cap_per_job_gb if limits else 0.0
        self._mem_cap_var = tk.DoubleVar(value=initial_cap)
        self._mem_cap_spin = ttk.Spinbox(
            top_frame,
            from_=0.0,
            to=4096.0,
            increment=1.0,
            textvariable=self._mem_cap_var,
            width=6,
            command=self._on_mem_cap_changed,
        )
        self._mem_cap_spin.pack(side="left")
        self._mem_cap_spin.bind("<Return>", lambda e: self._on_mem_cap_changed())
        self._mem_cap_spin.bind("<FocusOut>", lambda e: self._on_mem_cap_changed())

        _budget_tip = (
            "Default budget for each job. Soft target — jobs are allowed to\n"
            "exceed it when the system has memory to spare. Only enforced\n"
            "when free RAM drops below the reserve or swap grows past the\n"
            "allowance: the watchdog then kills whichever running job is\n"
            "most over its budget.\n"
            "\n"
            "0 = auto: after a scenario has been run successfully, its\n"
            "measured peak (×1.5 safety margin) becomes its budget for the\n"
            "next run, persisted in the project's settings.yaml. Scenarios\n"
            "that haven't been run yet fall back to:\n"
            "    (system memory − reserve) ÷ max parallel jobs."
        )
        attach_tooltip(mem_label, _budget_tip)
        attach_tooltip(self._mem_cap_spin, _budget_tip)

        # Min free RAM
        min_free_label = ttk.Label(top_frame, text="  Min free RAM (GB):")
        min_free_label.pack(side="left", padx=(15, 5))

        initial_reserve = limits.system_reserve_gb if limits else 4.0
        self._min_free_var = tk.DoubleVar(value=initial_reserve)
        self._min_free_spin = ttk.Spinbox(
            top_frame,
            from_=0.0,
            to=64.0,
            increment=1.0,
            textvariable=self._min_free_var,
            width=5,
            command=self._on_min_free_changed,
        )
        self._min_free_spin.pack(side="left")
        self._min_free_spin.bind("<Return>", lambda e: self._on_min_free_changed())
        self._min_free_spin.bind("<FocusOut>", lambda e: self._on_min_free_changed())

        _min_free_tip = (
            "Floor for system free memory. The watchdog only acts when\n"
            "BOTH this floor is breached AND the swap allowance is\n"
            "exhausted. As long as free RAM stays above this threshold,\n"
            "no job is killed.\n"
            "\n"
            "Together with the swap allowance, this keeps FlexTool from\n"
            "triggering a system-wide OOM that would take down other\n"
            "applications (browser, editor, etc.)."
        )
        attach_tooltip(min_free_label, _min_free_tip)
        attach_tooltip(self._min_free_spin, _min_free_tip)

        # Allow swap
        swap_label = ttk.Label(top_frame, text="  Allow swap (GB):")
        swap_label.pack(side="left", padx=(15, 5))

        initial_swap = limits.swap_allowance_gb if limits else 0.0
        self._swap_allow_var = tk.DoubleVar(value=initial_swap)
        self._swap_allow_spin = ttk.Spinbox(
            top_frame,
            from_=0.0,
            to=512.0,
            increment=1.0,
            textvariable=self._swap_allow_var,
            width=5,
            command=self._on_swap_allow_changed,
        )
        self._swap_allow_spin.pack(side="left")
        self._swap_allow_spin.bind("<Return>", lambda e: self._on_swap_allow_changed())
        self._swap_allow_spin.bind("<FocusOut>", lambda e: self._on_swap_allow_changed())

        _swap_tip = (
            "How much swap growth to tolerate while free RAM is below\n"
            "the reserve. Measured as growth since FlexTool started\n"
            "(pre-existing system swap is ignored).\n"
            "\n"
            "Plenty of free RAM ⇒ swap growth is harmless and never\n"
            "triggers a kill. Only when BOTH free RAM dips below the\n"
            "reserve AND swap growth has reached this allowance does\n"
            "the watchdog kill the most-over-budget running job.\n"
            "\n"
            "0 = no swap headroom: kill as soon as free RAM dips\n"
            "below the reserve. Allowing swap (>0) gives the system\n"
            "room to absorb pressure while large models complete, but\n"
            "they will run much slower because pages move between\n"
            "RAM and disk."
        )
        attach_tooltip(swap_label, _swap_tip)
        attach_tooltip(self._swap_allow_spin, _swap_tip)

        # ── Horizontal PanedWindow for Jobs / Progress ─────────────────
        self._paned = ttk.PanedWindow(self, orient="horizontal")
        self._paned.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        # Don't let the Jobs pane shrink below this — ttk.PanedWindow
        # with weight=0 on the left and weight=1 on the right will
        # squeeze the Jobs pane to near-zero when the window is first
        # resized narrower than the Progress pane's minimum and then
        # widened again.
        self._jobs_min_width: int = cw * 30
        # First-open default: 3× the minimum, so the Jobs columns fit
        # without horizontal scrolling. Overridden by the saved sash
        # (exec_jobs_sash) when there is one.
        self._jobs_default_width: int = cw * 90
        self._paned.bind("<Configure>", self._enforce_jobs_min_width)

        # ── Jobs list (left) ──────────────────────────────────────────
        left_frame = ttk.LabelFrame(self._paned, text="Jobs", padding=5)
        self._paned.add(left_frame, weight=0)
        left_frame.rowconfigure(0, weight=1)
        left_frame.columnconfigure(0, weight=1)

        self._job_tree = ttk.Treeview(
            left_frame,
            columns=("status", "source", "scenario", "peak", "timestamp"),
            show="headings",
            selectmode="extended",
        )
        self._job_tree.heading("status", text="")
        self._job_tree.heading("source", text="#")
        self._job_tree.heading("scenario", text="Scenario")
        self._job_tree.heading("peak", text="Peak GB")
        self._job_tree.heading("timestamp", text="Timestamp")

        self._job_tree.column("status", width=cw * 4, minwidth=cw * 4, stretch=False)
        self._job_tree.column("source", width=cw * 4, minwidth=cw * 3, stretch=False)
        self._job_tree.column("scenario", width=cw * 20, minwidth=cw * 10, stretch=True)
        self._job_tree.column("peak", width=int(cw * 8.8), minwidth=int(cw * 8.8), stretch=False)
        self._job_tree.column("timestamp", width=int(cw * 23.4), minwidth=int(cw * 23.4), stretch=False)

        self._job_tree.grid(row=0, column=0, sticky="nsew")

        job_scroll = ttk.Scrollbar(left_frame, orient="vertical", command=self._job_tree.yview)
        job_scroll.grid(row=0, column=1, sticky="ns")
        self._setup_autohide_scrollbar(self._job_tree, job_scroll)

        self._job_tree.bind("<<TreeviewSelect>>", self._on_job_selected)
        self._job_tree.bind("<B1-Motion>", self._on_drag_select)

        # ── Progress panel (right) ───────────────────────────────────
        right_frame = ttk.LabelFrame(self._paned, text="Progress", padding=5)
        self._paned.add(right_frame, weight=1)
        right_frame.rowconfigure(0, weight=1)
        right_frame.columnconfigure(0, weight=1)

        self._output_text = tk.Text(
            right_frame,
            wrap="none",
            font=self._mono_font,
        )
        self._output_text.grid(row=0, column=0, sticky="nsew")
        # Keep the widget editable (state='normal') so text selection and
        # Ctrl+C work reliably on all platforms, but block keyboard input.
        self._output_text.bind("<Key>", self._on_key_press)

        out_vscroll = ttk.Scrollbar(right_frame, orient="vertical", command=self._output_text.yview)
        self._output_text.configure(yscrollcommand=out_vscroll.set)
        out_vscroll.grid(row=0, column=1, sticky="ns")

        out_hscroll = ttk.Scrollbar(right_frame, orient="horizontal", command=self._output_text.xview)
        self._output_text.configure(xscrollcommand=out_hscroll.set)
        out_hscroll.grid(row=1, column=0, sticky="ew")

        # ── Buttons row ──────────────────────────────────────────────
        btn_frame = ttk.Frame(self, padding=(10, 5, 10, 10))
        btn_frame.grid(row=2, column=0, sticky="ew")

        col = 0
        self._pause_btn = ttk.Button(
            btn_frame, text="Pause executions", command=self._on_pause_toggle
        )
        self._pause_btn.grid(row=0, column=col, rowspan=2, padx=(0, 10), sticky="ns")

        col += 1
        move_frame = ttk.Frame(btn_frame)
        move_frame.grid(row=0, column=col, rowspan=2, padx=(0, 10))

        self._move_up_btn = ttk.Button(
            move_frame, text="\u25b2", width=3, command=self._on_move_up
        )
        self._move_up_btn.grid(row=0, column=0, padx=(0, 2))
        ttk.Label(move_frame, text="(PgUp)").grid(row=0, column=1, padx=(0, 4))

        self._move_down_btn = ttk.Button(
            move_frame, text="\u25bc", width=3, command=self._on_move_down
        )
        self._move_down_btn.grid(row=1, column=0, padx=(0, 2))
        ttk.Label(move_frame, text="(PgDn)").grid(row=1, column=1, padx=(0, 4))

        col += 1
        self._kill_btn = ttk.Button(
            btn_frame, text="Kill selected", command=self._on_kill_selected
        )
        self._kill_btn.grid(row=0, column=col, rowspan=2, padx=(0, 10), sticky="ns")

        col += 1
        self._remove_btn = ttk.Button(
            btn_frame, text="Remove selected", command=self._on_remove_selected
        )
        self._remove_btn.grid(row=0, column=col, rowspan=2, padx=(0, 10), sticky="ns")

        col += 1
        self._kill_all_btn = ttk.Button(btn_frame, text="Kill all", command=self._on_kill_all)
        self._kill_all_btn.grid(row=0, column=col, rowspan=2, padx=(0, 10), sticky="ns")

        col += 1
        self._status_label = ttk.Label(btn_frame, text="", anchor="center")
        self._status_label.grid(row=0, column=col, rowspan=2, padx=15, sticky="ew")
        btn_frame.columnconfigure(col, weight=1)  # status label stretches

        col += 1
        self._close_btn = ttk.Button(btn_frame, text="Close", command=self._on_close_attempt)
        self._close_btn.grid(row=0, column=col, rowspan=2, sticky="ns")

        # ── Keyboard shortcuts for move ──────────────────────────────
        self.bind("<Prior>", lambda e: self._on_move_up())
        self.bind("<Next>", lambda e: self._on_move_down())

        # ── Window close handler ─────────────────────────────────────
        self.protocol("WM_DELETE_WINDOW", self._on_close_attempt)

        # ── Restore saved sash position (or apply default) ───────────
        self.after(50, self._restore_sash)

        # ── Initial population and start polling ─────────────────────
        self._refresh_job_list()
        self._update_button_states()
        self._poll_updates()

    # ------------------------------------------------------------------
    # Auto-hide scrollbar helper
    # ------------------------------------------------------------------

    @staticmethod
    def _setup_autohide_scrollbar(
        tree: ttk.Treeview,
        scrollbar: ttk.Scrollbar,
    ) -> None:
        """Configure *scrollbar* to appear only when *tree* content overflows."""
        grid_info: dict = scrollbar.grid_info()

        def _on_scroll_set(first: str, last: str) -> None:
            scrollbar.set(first, last)
            if float(first) <= 0.0 and float(last) >= 1.0:
                scrollbar.grid_remove()
            else:
                scrollbar.grid(**grid_info)

        tree.configure(yscrollcommand=_on_scroll_set)
        scrollbar.grid_remove()

    # ------------------------------------------------------------------
    # Drag-to-select
    # ------------------------------------------------------------------

    def _on_drag_select(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """Extend selection to the item under the cursor during B1 drag."""
        item = self._job_tree.identify_row(event.y)
        if item:
            self._job_tree.selection_add(item)

    # ------------------------------------------------------------------
    # Max workers
    # ------------------------------------------------------------------

    def _on_max_workers_changed(self) -> None:
        """Sync the spinbox value to the ExecutionManager and persist it."""
        try:
            val = self._max_workers_var.get()
        except tk.TclError:
            return
        self._mgr.max_workers = val
        if self._global_settings is not None and val > 0:
            self._global_settings.max_workers = val
            try:
                from flextool.gui.project_utils import get_projects_dir
                from flextool.gui.settings_io import save_global_settings
                save_global_settings(get_projects_dir(), self._global_settings)
            except Exception:
                logger.warning("Could not persist max_workers to projects.yaml", exc_info=True)

    def _on_cores_per_job_changed(self) -> None:
        """Sync cores-per-job to global settings."""
        try:
            val = self._cores_per_job_var.get()
        except tk.TclError:
            return
        if val < 1:
            val = 1
            self._cores_per_job_var.set(val)
        if self._global_settings is None:
            return
        self._global_settings.execution_limits.max_cores_per_job = val
        try:
            from flextool.gui.project_utils import get_projects_dir
            from flextool.gui.settings_io import save_global_settings
            save_global_settings(get_projects_dir(), self._global_settings)
        except Exception:
            logger.warning("Could not persist max_cores_per_job to projects.yaml", exc_info=True)

    def _on_mem_cap_changed(self) -> None:
        """Sync memory budget/job to global settings."""
        try:
            val = float(self._mem_cap_var.get())
        except (tk.TclError, ValueError):
            return
        if val < 0:
            val = 0.0
            self._mem_cap_var.set(val)
        if self._global_settings is None:
            return
        self._global_settings.execution_limits.memory_cap_per_job_gb = val
        try:
            from flextool.gui.project_utils import get_projects_dir
            from flextool.gui.settings_io import save_global_settings
            save_global_settings(get_projects_dir(), self._global_settings)
        except Exception:
            logger.warning("Could not persist memory_cap_per_job_gb to projects.yaml", exc_info=True)

    def _on_min_free_changed(self) -> None:
        """Sync min-free-RAM threshold to global settings."""
        try:
            val = float(self._min_free_var.get())
        except (tk.TclError, ValueError):
            return
        if val < 0:
            val = 0.0
            self._min_free_var.set(val)
        if self._global_settings is None:
            return
        self._global_settings.execution_limits.system_reserve_gb = val
        try:
            from flextool.gui.project_utils import get_projects_dir
            from flextool.gui.settings_io import save_global_settings
            save_global_settings(get_projects_dir(), self._global_settings)
        except Exception:
            logger.warning("Could not persist system_reserve_gb to projects.yaml", exc_info=True)

    def _on_swap_allow_changed(self) -> None:
        """Sync swap-allowance threshold to global settings.

        Shows a warning dialog the first time the user raises the allowance
        above zero in this session.
        """
        try:
            val = float(self._swap_allow_var.get())
        except (tk.TclError, ValueError):
            return
        if val < 0:
            val = 0.0
            self._swap_allow_var.set(val)
        if self._global_settings is None:
            return
        self._global_settings.execution_limits.swap_allowance_gb = val
        try:
            from flextool.gui.project_utils import get_projects_dir
            from flextool.gui.settings_io import save_global_settings
            save_global_settings(get_projects_dir(), self._global_settings)
        except Exception:
            logger.warning("Could not persist swap_allowance_gb to projects.yaml", exc_info=True)

    # ------------------------------------------------------------------
    # Job list display
    # ------------------------------------------------------------------

    def _refresh_job_list(self) -> None:
        """Rebuild the job Treeview from the ExecutionManager's job list."""
        jobs = self._mgr.get_jobs()

        try:
            status = self._mgr.get_execution_status()
            self._status_label.config(text=_format_status_text(status))
        except (tk.TclError, AttributeError):
            pass  # window may be tearing down

        # Remember current selection so we can restore it
        prev_selection = set(self._job_tree.selection())

        # Configure tags for failed/killed rows (bright red)
        self._job_tree.tag_configure("failed", foreground="#ff3333")

        # Suppress <<TreeviewSelect>> events while we rebuild the tree
        self._refreshing_list = True
        try:
            # Clear tree
            for item in self._job_tree.get_children():
                self._job_tree.delete(item)

            # Repopulate
            restore_iids: list[str] = []
            for job in jobs:
                icon = _STATUS_ICONS.get(job.status, "?")
                ts = job.finish_timestamp if job.status in (JobStatus.SUCCESS, JobStatus.FAILED, JobStatus.KILLED) else ""
                iid = str(job.job_id)

                if job.job_type == JobType.SCENARIO:
                    source_col = str(job.source_number)
                    name_col = job.scenario_name
                else:
                    source_col = ""
                    name_col = job.display_name

                tags: tuple[str, ...] = ()
                if job.status in (JobStatus.FAILED, JobStatus.KILLED):
                    tags = ("failed",)

                peak_col = f"{job.peak_rss_mb / 1024:.1f}" if job.peak_rss_mb > 0 else ""
                if peak_col and getattr(job, 'killed_for_memory', False):
                    peak_col += " ⚠"

                self._job_tree.insert(
                    "",
                    "end",
                    iid=iid,
                    values=(icon, source_col, name_col, peak_col, ts),
                    tags=tags,
                )
                if iid in prev_selection:
                    restore_iids.append(iid)

            # Restore selection
            if restore_iids:
                self._job_tree.selection_set(restore_iids)
            else:
                if prev_selection:
                    pass
        finally:
            self._refreshing_list = False

    def _on_job_selected(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        """Handle selection change in the job tree.

        With multi-select enabled, we show the log for the first selected
        job in the progress panel.
        """
        # Ignore events triggered programmatically during list refresh
        if self._refreshing_list:
            return

        selection = self._job_tree.selection()
        if selection:
            try:
                first_id = int(selection[0])
            except (ValueError, IndexError):
                first_id = None
        else:
            first_id = None

        if first_id == self._viewed_job_id:
            return

        self._viewed_job_id = first_id

        # Switching to a different job: clear the text widget and reset its
        # displayed count so all lines are re-inserted from scratch.
        self._output_text.delete("1.0", "end")
        if first_id is not None:
            self._displayed_counts[first_id] = 0
        self._update_output_display()

    # ------------------------------------------------------------------
    # Output display
    # ------------------------------------------------------------------

    def _on_key_press(self, event: tk.Event) -> str | None:  # type: ignore[type-arg]
        """Block keyboard input into the output text widget while still
        allowing Ctrl+C (copy) and Ctrl+A (select all)."""
        if event.state & 0x4 and event.keysym in ("c", "C", "a", "A"):
            return None  # Allow the event to propagate
        return "break"  # Suppress all other key events

    def _update_output_display(self) -> None:
        """Append new stdout lines for the currently viewed job.

        Only lines that haven't been displayed yet are inserted.  If the
        user has scrolled up to read earlier output, auto-scroll is
        suppressed so the viewport stays put.
        """
        job_id = self._viewed_job_id
        if job_id is None:
            return

        lines = self._mgr.get_stdout(job_id)
        already_shown = self._displayed_counts.get(job_id, 0)
        new_count = len(lines)
        if new_count <= already_shown:
            return

        # Check whether the view is currently at the bottom *before*
        # inserting new text.
        at_bottom = self._output_text.yview()[1] >= 0.99

        # Append only the new lines
        new_lines = lines[already_shown:]
        for line in new_lines:
            self._output_text.insert("end", line + "\n")
        self._displayed_counts[job_id] = new_count

        # Auto-scroll only when the user is already at (or near) the bottom
        if at_bottom:
            self._output_text.see("end")

    # ------------------------------------------------------------------
    # Button states
    # ------------------------------------------------------------------

    def _get_selected_job_ids(self) -> list[int]:
        """Return job IDs for all selected items in the tree."""
        ids: list[int] = []
        for iid in self._job_tree.selection():
            try:
                ids.append(int(iid))
            except (ValueError, IndexError):
                pass
        return ids

    def _update_button_states(self) -> None:
        """Enable/disable buttons based on current execution state."""
        jobs = self._mgr.get_jobs()
        has_pending_or_running = any(
            j.status in (JobStatus.PENDING, JobStatus.RUNNING) for j in jobs
        )
        # Pause only applies to the scenario scheduler
        has_pending_scenarios = any(
            j.status == JobStatus.PENDING
            for j in jobs if j.job_type == JobType.SCENARIO
        )
        is_paused = self._mgr.is_paused

        selected_ids = set(self._get_selected_job_ids())
        job_by_id = {j.job_id: j for j in jobs}

        # Pause/Continue toggle: enabled when there are pending scenario jobs
        if has_pending_scenarios or is_paused:
            self._pause_btn.configure(state="normal")
            if is_paused:
                self._pause_btn.configure(text="Continue executions")
            else:
                self._pause_btn.configure(text="Pause executions")
        else:
            self._pause_btn.configure(state="disabled", text="Pause executions")

        # Kill: enabled if any selected job is running or pending
        can_kill = any(
            job_by_id.get(jid) is not None
            and job_by_id[jid].status in (JobStatus.RUNNING, JobStatus.PENDING)
            for jid in selected_ids
        )
        self._kill_btn.configure(state="normal" if can_kill else "disabled")

        # Remove: enabled if any selected job is finished or killed
        can_remove = any(
            job_by_id.get(jid) is not None
            and job_by_id[jid].status in (JobStatus.SUCCESS, JobStatus.FAILED, JobStatus.KILLED)
            for jid in selected_ids
        )
        self._remove_btn.configure(state="normal" if can_remove else "disabled")

        # Kill all: enabled if there are running or pending jobs
        self._kill_all_btn.configure(
            state="normal" if has_pending_or_running else "disabled"
        )

        # Close: only enabled when nothing is running or pending
        self._close_btn.configure(
            state="normal" if not has_pending_or_running else "disabled"
        )

    # ------------------------------------------------------------------
    # Periodic polling
    # ------------------------------------------------------------------

    def _poll_updates(self) -> None:
        """Periodically refresh the job list and output display."""
        if not self.winfo_exists():
            return

        # Auto-select newly created auxiliary jobs
        pending_id = self._mgr._pending_select_job_id
        if pending_id is not None:
            self._mgr._pending_select_job_id = None
            self._refresh_job_list()
            self.select_job(pending_id)
        else:
            self._refresh_job_list()

        self._update_output_display()
        self._update_button_states()

        self.after(500, self._poll_updates)

    # ------------------------------------------------------------------
    # Thread-safe refresh (called from ExecutionManager callbacks)
    # ------------------------------------------------------------------

    def schedule_refresh(self) -> None:
        """Schedule a job list refresh on the main thread.

        Safe to call from any thread.
        """
        if self.winfo_exists():
            self.after(0, self._refresh_job_list)

    def select_job(self, job_id: int) -> None:
        """Select and show the log for the job with *job_id*.

        If the tree hasn't been populated with this job yet, a refresh is
        triggered first.
        """
        iid = str(job_id)
        if not self._job_tree.exists(iid):
            self._refresh_job_list()
        if self._job_tree.exists(iid):
            self._job_tree.selection_set(iid)
            self._job_tree.see(iid)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_pause_toggle(self) -> None:
        """Toggle between pausing and continuing execution."""
        if self._mgr.is_paused:
            self._mgr.resume()
        else:
            self._mgr.pause()
        self.after(100, self._update_button_states)

    def _on_kill_selected(self) -> None:
        """Kill all selected running or pending jobs.

        Pending jobs that have not started execution are automatically
        removed from the list.
        """
        selected_ids = self._get_selected_job_ids()
        if not selected_ids:
            return

        jobs = self._mgr.get_jobs()
        job_by_id = {j.job_id: j for j in jobs}

        for jid in selected_ids:
            job = job_by_id.get(jid)
            if job is None:
                continue
            if job.status == JobStatus.RUNNING:
                self._mgr.kill_job(jid)
            elif job.status == JobStatus.PENDING:
                # Pending jobs that were never executed: remove immediately
                self._mgr.remove_job(jid)

        self._refresh_job_list()
        self._update_button_states()

    def _on_remove_selected(self) -> None:
        """Remove all selected finished or killed jobs from the list."""
        selected_ids = self._get_selected_job_ids()
        if not selected_ids:
            return

        for jid in selected_ids:
            self._mgr.remove_job(jid)

        self._refresh_job_list()
        self._update_button_states()

    def _on_kill_all(self) -> None:
        """Kill all running processes and cancel pending jobs.

        Pending jobs that have not been executed are auto-removed.
        """
        jobs = self._mgr.get_jobs()
        pending_ids = [j.job_id for j in jobs if j.status == JobStatus.PENDING]

        self._mgr.kill_all()

        # Remove pending jobs (they were never executed)
        for jid in pending_ids:
            self._mgr.remove_job(jid)

        self._refresh_job_list()
        self._update_button_states()

    def _on_move_up(self) -> None:
        """Move the selected pending job one position earlier in the queue."""
        selected_ids = self._get_selected_job_ids()
        if len(selected_ids) != 1:
            return
        self._mgr.move_pending_up(selected_ids[0])
        self._refresh_job_list()
        # Re-select the moved item
        iid = str(selected_ids[0])
        if self._job_tree.exists(iid):
            self._job_tree.selection_set(iid)

    def _on_move_down(self) -> None:
        """Move the selected pending job one position later in the queue."""
        selected_ids = self._get_selected_job_ids()
        if len(selected_ids) != 1:
            return
        self._mgr.move_pending_down(selected_ids[0])
        self._refresh_job_list()
        # Re-select the moved item
        iid = str(selected_ids[0])
        if self._job_tree.exists(iid):
            self._job_tree.selection_set(iid)

    def _restore_sash(self) -> None:
        """Restore the saved Jobs/Progress sash position, clamped to the
        current pane width so a value saved at a different DPI cannot
        collapse the Progress pane to zero. When no saved value exists,
        apply ``_jobs_default_width`` so the Jobs columns are visible
        on first open.
        """
        from flextool.gui.ui_metrics import clamp_sash, rescale_pixels
        try:
            self.update_idletasks()
            paned_total = self._paned.winfo_width()
            saved = 0
            saved_cw = 0
            if self._global_settings is not None:
                saved = self._global_settings.exec_jobs_sash
                saved_cw = self._global_settings.exec_jobs_layout_cw
            if saved > 0:
                target = clamp_sash(
                    rescale_pixels(saved, saved_cw, self._char_width),
                    paned_total,
                    min_px=self._jobs_min_width,
                )
            else:
                target = clamp_sash(
                    self._jobs_default_width,
                    paned_total,
                    min_px=self._jobs_min_width,
                )
            if target > 0:
                self._paned.sashpos(0, target)
        except (tk.TclError, IndexError):
            pass

    def _enforce_jobs_min_width(self, _event: tk.Event) -> None:
        """Keep the Jobs pane from collapsing during window resize.

        The Jobs pane has ``weight=0`` so that extra horizontal space goes
        to the Progress pane; but on Linux/ttk a narrow-then-wide window
        resize cycle can leave the sash near zero. This clamps the sash
        back to a readable minimum whenever the pane window is laid out.
        """
        try:
            if self._paned.sashpos(0) < self._jobs_min_width:
                # Only clamp if the overall pane is wide enough to honour it.
                if self._paned.winfo_width() > self._jobs_min_width + 50:
                    self._paned.sashpos(0, self._jobs_min_width)
        except (tk.TclError, IndexError):
            pass

    def _save_sash(self) -> None:
        """Save the current sash position to global settings."""
        if self._global_settings is None:
            return
        try:
            self._global_settings.exec_jobs_sash = self._paned.sashpos(0)
        except (tk.TclError, IndexError):
            pass
        self._global_settings.exec_jobs_layout_cw = self._char_width
        try:
            from flextool.gui.project_utils import get_projects_dir
            from flextool.gui.settings_io import save_global_settings
            save_global_settings(get_projects_dir(), self._global_settings)
        except Exception:
            pass

    def _on_close_attempt(self) -> None:
        """Handle the close button or window manager close request."""
        if self._mgr.has_pending_or_running():
            messagebox.showwarning(
                "Jobs still active",
                "There are running or pending jobs.\n"
                "Please use 'Kill all' or 'Wind down' first.",
                parent=self,
            )
            return
        self._save_sash()
        self.destroy()
