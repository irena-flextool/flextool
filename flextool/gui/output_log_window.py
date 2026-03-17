"""Non-modal log window for output action subprocesses."""

from __future__ import annotations

import subprocess
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk


class OutputLogWindow(tk.Toplevel):
    """Non-modal window that displays the command and live stdout of a subprocess.

    The window stays open after the process finishes so the user can
    review the output.  Two buttons are provided:

    * **Close** -- closes the window without stopping the process.
    * **Stop and close** -- kills the process and closes the window.
    """

    def __init__(self, parent: tk.Misc, title: str) -> None:
        super().__init__(parent)
        self.title(title)
        self.transient(parent)

        self._process: subprocess.Popen | None = None

        # ── Font metrics ─────────────────────────────────────────
        mono_font = tkfont.nametofont("TkFixedFont")
        default_font = tkfont.nametofont("TkDefaultFont")
        cw = default_font.measure("0")
        lh = default_font.metrics("linespace")

        # ── Window sizing ────────────────────────────────────────
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        taskbar_margin = lh * 4
        win_h = screen_h - taskbar_margin
        win_w = 1800
        x = max(0, screen_w - win_w)
        self.geometry(f"{win_w}x{win_h}+{x}+0")
        self.minsize(cw * 50, lh * 10)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # ── Text widget ─────────────────────────────────────────
        text_frame = ttk.Frame(self)
        text_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 4))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)

        self._text = tk.Text(text_frame, wrap="word", font=mono_font)
        self._text.grid(row=0, column=0, sticky="nsew")
        self._text.bind("<Key>", self._on_key_press)

        vscroll = ttk.Scrollbar(text_frame, orient="vertical", command=self._text.yview)
        self._text.configure(yscrollcommand=vscroll.set)
        vscroll.grid(row=0, column=1, sticky="ns")

        # ── Buttons ──────────────────────────────────────────────
        btn_frame = ttk.Frame(self, padding=(8, 4, 8, 8))
        btn_frame.grid(row=1, column=0, sticky="ew")

        self._stop_close_btn = ttk.Button(
            btn_frame, text="Stop and close", command=self._on_stop_and_close,
        )
        self._stop_close_btn.pack(side="left", padx=(0, 10))

        self._close_btn = ttk.Button(
            btn_frame, text="Close", command=self._on_close,
        )
        self._close_btn.pack(side="right")

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Public API ───────────────────────────────────────────────

    def set_process(self, proc: subprocess.Popen) -> None:
        """Register the subprocess so Stop can kill it."""
        self._process = proc

    def append_line(self, line: str) -> None:
        """Append a line of text. Must be called from the main thread."""
        if not self.winfo_exists():
            return
        at_bottom = self._text.yview()[1] >= 0.99
        self._text.insert("end", line + "\n")
        if at_bottom:
            self._text.see("end")

    def mark_finished(self, success: bool) -> None:
        """Mark the process as finished — update button states."""
        if not self.winfo_exists():
            return
        self._process = None
        self._stop_close_btn.configure(state="disabled")

    # ── Event handlers ───────────────────────────────────────────

    def _on_key_press(self, event: tk.Event) -> str | None:  # type: ignore[type-arg]
        """Allow Ctrl+C/A but block other input."""
        if event.state & 0x4 and event.keysym in ("c", "C", "a", "A"):
            return None
        return "break"

    def _on_close(self) -> None:
        """Close the window without stopping the process."""
        self.destroy()

    def _on_stop_and_close(self) -> None:
        """Kill the process and close the window."""
        if self._process is not None:
            try:
                self._process.kill()
            except OSError:
                pass
            self._process = None
        self.destroy()
