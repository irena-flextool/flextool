"""Modal progress dialog for the automatic FlexTool DB migration."""
from __future__ import annotations

import queue
import tkinter as tk
from tkinter import ttk

# Mirrors flextool.gui.main_window._SPINNER_FRAMES; duplicated to keep this
# dialog free of dependencies on main_window.
_SPINNER_FRAMES = ["◐", "◓", "◑", "◒"]  # rotating circle (Geometric Shapes; render on Windows Tk, unlike the old U+29D6/7 hourglasses)


class MigrationProgressDialog(tk.Toplevel):
    """Modal progress dialog for the automatic DB migration.

    The caller runs the migration in a worker thread. This dialog drives
    the spinner from the Tk main thread via ``after`` and exposes:

      - ``update_status(text: str)`` — thread-safe; schedules a label
        update on the main thread.
      - ``mark_finished()`` — thread-safe; schedules dialog close.
      - ``cancel_requested`` — read-only ``bool`` property; True after
        the user clicked Cancel (worker polls this via ``cancel_check``).

    The dialog is shown by constructing it and then calling
    ``parent.wait_window(self)``.  It does not spawn the worker thread
    itself — the caller is responsible for that.
    """

    def __init__(
        self,
        parent: tk.Misc,
        title: str = "Migrating database…",
        initial_status: str = "Preparing…",
    ) -> None:
        super().__init__(parent)
        self.title(title)
        self.transient(parent)
        self.resizable(False, False)

        self._cancel_requested: bool = False
        self._finished: bool = False
        # The ``after`` ids must be read/written from the main thread only.
        self._spinner_after_id: str | None = None
        self._pump_after_id: str | None = None
        self._spinner_idx: int = 0
        # Worker threads enqueue updates here; a main-thread pump applies them.
        # tkinter is not thread-safe (even ``after`` fails off the main thread
        # on macOS), so the public methods must not touch Tk.
        self._ui_queue: queue.Queue = queue.Queue()

        self._status_var = tk.StringVar(value=initial_status)
        status_lbl = ttk.Label(
            self,
            textvariable=self._status_var,
            wraplength=440,
            justify="left",
        )
        status_lbl.pack(padx=16, pady=(16, 8), anchor="w", fill="x")

        self._spinner_var = tk.StringVar(value=_SPINNER_FRAMES[0])
        spinner_lbl = ttk.Label(
            self,
            textvariable=self._spinner_var,
            font=("TkDefaultFont", 16),
            justify="left",
        )
        spinner_lbl.pack(padx=16, pady=(0, 8), anchor="w")

        note_lbl = ttk.Label(
            self,
            text=(
                "The interface is locked until the migration finishes. "
                "Detailed progress is shown in the Execution window. "
                "Click Cancel to stop after the current step — a cancelled "
                "database is restored to its original state."
            ),
            wraplength=440,
            justify="left",
        )
        note_lbl.pack(padx=16, pady=(0, 8), anchor="w", fill="x")

        btn_frame = ttk.Frame(self)
        btn_frame.pack(padx=16, pady=(4, 16), fill="x")

        self._cancel_btn = ttk.Button(
            btn_frame, text="Cancel", command=self._on_cancel,
        )
        self._cancel_btn.pack(side="right")

        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.bind("<Escape>", lambda _e: self._on_cancel())

        # Center on parent
        self.update_idletasks()
        pw = parent.winfo_width()
        ph = parent.winfo_height()
        px = parent.winfo_rootx()
        py = parent.winfo_rooty()
        dw = self.winfo_width()
        dh = self.winfo_height()
        self.geometry(f"+{px + (pw - dw) // 2}+{py + (ph - dh) // 2}")

        # Kick off the spinner animation and the worker-update pump.
        self._spinner_after_id = self.after(200, self._tick_spinner)
        self._pump_after_id = self.after(50, self._pump_queue)

        # Grab input once the window is actually mapped. ``grab_set`` raises
        # "grab failed: window not viewable" if called before Tk has mapped
        # the Toplevel — which happens when this dialog is constructed from
        # inside another Tk callback (e.g. the project-combo selection).
        self._install_grab()

    def _install_grab(self) -> None:
        """Grab input once the window is viewable, retrying on the main thread."""
        if self._finished or not self.winfo_exists():
            return
        if self.winfo_viewable():
            try:
                self.grab_set()
            except tk.TclError:
                pass
            return
        self.after(50, self._install_grab)

    # ── Public API (thread-safe wrappers) ───────────────────────────

    @property
    def cancel_requested(self) -> bool:
        """Whether the user has requested cancellation."""
        return self._cancel_requested

    def update_status(self, text: str) -> None:
        """Thread-safe: enqueue a status-label update (no Tk call here)."""
        self._ui_queue.put(("status", text))

    def mark_finished(self) -> None:
        """Thread-safe: enqueue dialog close (no Tk call here)."""
        self._ui_queue.put(("finish", None))

    # ── Main-thread helpers ─────────────────────────────────────────

    def _pump_queue(self) -> None:
        """Apply queued worker updates on the main thread, then reschedule.

        Drains the whole queue each tick, applying only the most recent status
        (intermediate ones are stale) and closing if a finish was requested.
        """
        if self._finished:
            return
        latest_status: str | None = None
        finish = False
        try:
            while True:
                kind, payload = self._ui_queue.get_nowait()
                if kind == "status":
                    latest_status = payload
                elif kind == "finish":
                    finish = True
        except queue.Empty:
            pass

        if latest_status is not None:
            self._set_status(latest_status)
        if finish:
            self._finish()
            return
        self._pump_after_id = self.after(50, self._pump_queue)

    def _set_status(self, text: str) -> None:
        if self._finished:
            return
        try:
            self._status_var.set(text)
        except tk.TclError:
            pass

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        if self._spinner_after_id is not None:
            try:
                self.after_cancel(self._spinner_after_id)
            except tk.TclError:
                pass
            self._spinner_after_id = None
        if self._pump_after_id is not None:
            try:
                self.after_cancel(self._pump_after_id)
            except tk.TclError:
                pass
            self._pump_after_id = None
        try:
            self.grab_release()
        except tk.TclError:
            pass
        try:
            self.destroy()
        except tk.TclError:
            pass

    def _tick_spinner(self) -> None:
        if self._finished:
            return
        self._spinner_idx = (self._spinner_idx + 1) % len(_SPINNER_FRAMES)
        try:
            self._spinner_var.set(_SPINNER_FRAMES[self._spinner_idx])
        except tk.TclError:
            return
        self._spinner_after_id = self.after(200, self._tick_spinner)

    def _on_cancel(self) -> None:
        if self._cancel_requested:
            return
        self._cancel_requested = True
        try:
            self._cancel_btn.configure(
                state="disabled", text="Cancelling…",
            )
        except tk.TclError:
            pass

    def destroy(self) -> None:  # noqa: D401 - override for cleanup
        if self._spinner_after_id is not None:
            try:
                self.after_cancel(self._spinner_after_id)
            except tk.TclError:
                pass
            self._spinner_after_id = None
        if self._pump_after_id is not None:
            try:
                self.after_cancel(self._pump_after_id)
            except tk.TclError:
                pass
            self._pump_after_id = None
        super().destroy()
