"""Non-modal Toplevel window for managing FlexTool scenario executions."""

from __future__ import annotations

import logging
import os
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, messagebox

from flextool.gui.execution_manager import ExecutionJob, ExecutionManager, JobStatus

logger = logging.getLogger(__name__)

# Status icons for the scenarios list
_STATUS_ICONS: dict[JobStatus, str] = {
    JobStatus.SUCCESS: "\u2713",   # ✓
    JobStatus.FAILED: "\u2717",    # ✗
    JobStatus.RUNNING: "\u23f3",   # ⏳
    JobStatus.PENDING: "\u2610",   # ☐
    JobStatus.KILLED: "\u2717",    # ✗
}


class ExecutionWindow(tk.Toplevel):
    """Non-modal window for monitoring and controlling scenario executions.

    This window coexists with the main window -- it never calls
    ``grab_set()`` or ``wait_window()``.
    """

    def __init__(
        self,
        parent: tk.Tk,
        execution_mgr: ExecutionManager,
    ) -> None:
        super().__init__(parent)
        self.title("Execution Menu")
        self.transient(parent)

        self._mgr = execution_mgr
        self._selected_job_id: int | None = None
        # Track how many stdout lines we have already displayed per job
        # so that we only append new lines on each poll cycle.
        self._displayed_counts: dict[int, int] = {}
        # Guard flag to suppress <<TreeviewSelect>> events fired by
        # programmatic selection_set inside _refresh_job_list.
        self._refreshing_list: bool = False

        # ── Font metrics for DPI-aware sizing ────────────────────────
        default_font = tkfont.nametofont("TkDefaultFont")
        cw: int = default_font.measure("0")
        lh: int = default_font.metrics("linespace")
        mono_font = tkfont.nametofont("TkFixedFont")
        self._mono_font = mono_font

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

        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        # ── Top row: max parallel executions ─────────────────────────
        top_frame = ttk.Frame(self, padding=(10, 5))
        top_frame.grid(row=0, column=0, columnspan=2, sticky="e")

        ttk.Label(top_frame, text="Max. parallel executions:").pack(side="left", padx=(0, 5))

        cpu = os.cpu_count() or 2
        default_workers = max(1, cpu - 1)
        self._max_workers_var = tk.IntVar(value=default_workers)
        self._max_workers_spin = ttk.Spinbox(
            top_frame,
            from_=1,
            to=cpu * 2,
            textvariable=self._max_workers_var,
            width=4,
            command=self._on_max_workers_changed,
        )
        self._max_workers_spin.pack(side="left")
        # Also sync the spinbox to the current manager value
        self._max_workers_var.set(self._mgr.max_workers)

        # ── Scenarios list (left) ────────────────────────────────────
        left_frame = ttk.LabelFrame(self, text="Scenarios", padding=5)
        left_frame.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=5)
        left_frame.rowconfigure(0, weight=1)
        left_frame.columnconfigure(0, weight=1)

        self._job_tree = ttk.Treeview(
            left_frame,
            columns=("status", "source", "scenario", "timestamp"),
            show="headings",
            selectmode="browse",
        )
        self._job_tree.heading("status", text="")
        self._job_tree.heading("source", text="#")
        self._job_tree.heading("scenario", text="Scenario")
        self._job_tree.heading("timestamp", text="Timestamp")

        self._job_tree.column("status", width=cw * 3, minwidth=cw * 3, stretch=False)
        self._job_tree.column("source", width=cw * 4, minwidth=cw * 3, stretch=False)
        self._job_tree.column("scenario", width=cw * 20, minwidth=cw * 10, stretch=True)
        self._job_tree.column("timestamp", width=cw * 16, minwidth=cw * 10)

        self._job_tree.grid(row=0, column=0, sticky="nsew")

        job_scroll = ttk.Scrollbar(left_frame, orient="vertical", command=self._job_tree.yview)
        self._job_tree.configure(yscrollcommand=job_scroll.set)
        job_scroll.grid(row=0, column=1, sticky="ns")

        self._job_tree.bind("<<TreeviewSelect>>", self._on_job_selected)

        # ── Progress panel (right) ───────────────────────────────────
        right_frame = ttk.LabelFrame(self, text="Progress", padding=5)
        right_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=5)
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
        btn_frame.grid(row=2, column=0, columnspan=2, sticky="ew")

        self._start_btn = ttk.Button(btn_frame, text="Start executions", command=self._on_start)
        self._start_btn.pack(side="left", padx=(0, 10))

        self._kill_remove_btn = ttk.Button(
            btn_frame, text="Kill / Remove selected", command=self._on_kill_remove
        )
        self._kill_remove_btn.pack(side="left", padx=(0, 10))

        self._wind_down_btn = ttk.Button(btn_frame, text="Wind down", command=self._on_wind_down)
        self._wind_down_btn.pack(side="left", padx=(0, 10))

        self._kill_all_btn = ttk.Button(btn_frame, text="Kill all", command=self._on_kill_all)
        self._kill_all_btn.pack(side="left", padx=(0, 10))

        self._close_btn = ttk.Button(btn_frame, text="Close", command=self._on_close_attempt)
        self._close_btn.pack(side="right")

        # ── Window close handler ─────────────────────────────────────
        self.protocol("WM_DELETE_WINDOW", self._on_close_attempt)

        # ── Initial population and start polling ─────────────────────
        self._refresh_job_list()
        self._update_button_states()
        self._poll_updates()

    # ------------------------------------------------------------------
    # Max workers
    # ------------------------------------------------------------------

    def _on_max_workers_changed(self) -> None:
        """Sync the spinbox value to the ExecutionManager."""
        try:
            val = self._max_workers_var.get()
        except tk.TclError:
            return
        self._mgr.max_workers = val

    # ------------------------------------------------------------------
    # Job list display
    # ------------------------------------------------------------------

    def _refresh_job_list(self) -> None:
        """Rebuild the scenarios Treeview from the ExecutionManager's job list."""
        jobs = self._mgr.get_jobs()

        # Remember current selection so we can restore it
        prev_selected_id = self._selected_job_id

        # Suppress <<TreeviewSelect>> events while we rebuild the tree
        self._refreshing_list = True
        try:
            # Clear tree
            for item in self._job_tree.get_children():
                self._job_tree.delete(item)

            # Repopulate
            select_iid: str | None = None
            for job in jobs:
                icon = _STATUS_ICONS.get(job.status, "?")
                ts = job.finish_timestamp if job.status in (JobStatus.SUCCESS, JobStatus.FAILED, JobStatus.KILLED) else ""
                iid = str(job.job_id)
                self._job_tree.insert(
                    "",
                    "end",
                    iid=iid,
                    values=(icon, job.source_number, job.scenario_name, ts),
                )
                if job.job_id == prev_selected_id:
                    select_iid = iid

            # Restore selection
            if select_iid and self._job_tree.exists(select_iid):
                self._job_tree.selection_set(select_iid)
            elif not select_iid:
                # Previously selected job was removed; clear output
                self._selected_job_id = None
                self._clear_output()
        finally:
            self._refreshing_list = False

    def _on_job_selected(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        """Handle selection change in the job tree."""
        # Ignore events triggered programmatically during list refresh
        if self._refreshing_list:
            return

        selection = self._job_tree.selection()
        if selection:
            try:
                new_id = int(selection[0])
            except (ValueError, IndexError):
                new_id = None
        else:
            new_id = None

        if new_id == self._selected_job_id:
            return

        self._selected_job_id = new_id

        # Switching to a different job: clear the text widget and reset its
        # displayed count so all lines are re-inserted from scratch.
        self._clear_output()
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

    def _clear_output(self) -> None:
        """Clear the output text widget and reset displayed count for the
        currently selected job."""
        self._output_text.delete("1.0", "end")
        if self._selected_job_id is not None:
            self._displayed_counts[self._selected_job_id] = 0

    def _update_output_display(self) -> None:
        """Append new stdout lines for the currently selected job.

        Only lines that haven't been displayed yet are inserted.  If the
        user has scrolled up to read earlier output, auto-scroll is
        suppressed so the viewport stays put.
        """
        job_id = self._selected_job_id
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

    def _update_button_states(self) -> None:
        """Enable/disable buttons based on current execution state."""
        jobs = self._mgr.get_jobs()
        has_pending = any(j.status == JobStatus.PENDING for j in jobs)
        has_running = any(j.status == JobStatus.RUNNING for j in jobs)
        has_pending_or_running = has_pending or has_running

        # Start: enabled only if there are pending jobs and no scheduler is running
        # (simplified: enabled when there are pending jobs)
        self._start_btn.configure(
            state="normal" if has_pending else "disabled"
        )

        # Kill/Remove: enabled if something is selected
        sel = self._job_tree.selection()
        self._kill_remove_btn.configure(
            state="normal" if sel else "disabled"
        )

        # Wind down: enabled if there are running jobs
        self._wind_down_btn.configure(
            state="normal" if has_running else "disabled"
        )

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

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_start(self) -> None:
        """Start executing pending jobs.

        After calling start(), we schedule the button-state update via
        ``after()`` instead of calling it synchronously.  This gives the
        scheduler thread time to start and release any initial lock
        acquisition, avoiding a potential deadlock where the main thread
        and the scheduler thread both contend for ``_lock``.
        """
        self._mgr.start()
        self.after(100, self._update_button_states)

    def _on_kill_remove(self) -> None:
        """Kill or remove the selected job."""
        if self._selected_job_id is None:
            return

        jobs = self._mgr.get_jobs()
        job = next((j for j in jobs if j.job_id == self._selected_job_id), None)
        if job is None:
            return

        if job.status == JobStatus.RUNNING:
            self._mgr.kill_job(job.job_id)
        elif job.status in (JobStatus.SUCCESS, JobStatus.FAILED, JobStatus.KILLED, JobStatus.PENDING):
            self._mgr.remove_job(job.job_id)

        self._refresh_job_list()
        self._update_button_states()

    def _on_wind_down(self) -> None:
        """Let running jobs finish but stop starting new ones."""
        self._mgr.wind_down()
        self._update_button_states()

    def _on_kill_all(self) -> None:
        """Kill all running processes and cancel pending jobs."""
        self._mgr.kill_all()
        self._refresh_job_list()
        self._update_button_states()

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
        self.destroy()
