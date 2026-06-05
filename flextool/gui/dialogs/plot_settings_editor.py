"""Modal YAML editor for a project's ``plot_settings.yaml`` file.

Lets the user edit this project's plot settings (colors, stacking order,
and sign coloring) directly as YAML.  Validates syntax on save and
refuses to persist invalid YAML.  This is a plain text editor (a richer
color picker is a later stage); it is *not* the dispatch plot config
editor.
"""

from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import yaml

logger = logging.getLogger(__name__)


class PlotSettingsEditor(tk.Toplevel):
    """Modal text editor for a project's ``plot_settings.yaml``.

    Shows an instruction area above the editable text, validates YAML on
    save, and refuses to save invalid syntax.  On a successful save the
    dialog closes; the caller is responsible for re-rendering with the new
    colors.  ``self.saved`` reports whether a save happened.
    """

    def __init__(self, parent: tk.Misc, settings_path: Path) -> None:
        super().__init__(parent)
        self.title("Plot settings")
        self._settings_path = Path(settings_path)
        self.saved = False

        self.transient(parent)
        self.grab_set()

        # ── Sizing ────────────────────────────────────────────────
        from flextool.gui.ui_metrics import get_metrics
        _metrics = get_metrics(self)
        cw = _metrics.cw
        lh = _metrics.lh
        # Use the named-font string so live size changes reach tk.Text.
        mono_font = "TkFixedFont"

        self.geometry(f"{cw * 108}x{lh * 40}")
        self.resizable(True, True)
        self.minsize(cw * 60, lh * 20)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # ── Instructions ──────────────────────────────────────────
        info_frame = ttk.LabelFrame(self, text="Instructions", padding=6)
        info_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))

        info_text = (
            "Edit this project's plot settings: the colors used when "
            "plotting its results, the stacking order (entry order, top "
            "to bottom), and an optional 'neg_color' for an entity's "
            "negative-side part.\n"
            "Entries under 'categories' map a result/parameter name to a "
            "color (exact match); entries under 'entities' map an entity "
            "name to a color (case-insensitive).\n"
            "Colors are '#RRGGBB' hex strings or [r, g, b] lists; an "
            "entity may instead be {color: ..., neg_color: ...}.\n"
            "These settings apply to this project only — deleting "
            "plot_settings.yaml falls back to a built-in starting point."
        )
        ttk.Label(info_frame, text=info_text, wraplength=cw * 80).pack(
            fill="x",
        )

        # ── Text editor ──────────────────────────────────────────
        edit_frame = ttk.Frame(self)
        edit_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=4)
        edit_frame.columnconfigure(0, weight=1)
        edit_frame.rowconfigure(0, weight=1)

        self._text = tk.Text(edit_frame, wrap="none", font=mono_font, undo=True)
        self._text.grid(row=0, column=0, sticky="nsew")

        vscroll = ttk.Scrollbar(edit_frame, orient="vertical", command=self._text.yview)
        vscroll.grid(row=0, column=1, sticky="ns")
        self._text.configure(yscrollcommand=vscroll.set)

        hscroll = ttk.Scrollbar(edit_frame, orient="horizontal", command=self._text.xview)
        hscroll.grid(row=1, column=0, sticky="ew")
        self._text.configure(xscrollcommand=hscroll.set)

        # Load file content
        try:
            content = self._settings_path.read_text(encoding="utf-8")
        except OSError as exc:
            content = f"# Error reading file: {exc}"
        self._text.insert("1.0", content)

        # ── Buttons ───────────────────────────────────────────────
        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(4, 10))

        ttk.Button(btn_frame, text="Cancel", command=self._on_cancel).pack(
            side="right", padx=(5, 0),
        )
        ttk.Button(btn_frame, text="Save and close", command=self._on_save).pack(
            side="right",
        )

        # ── Keyboard shortcuts ────────────────────────────────────
        self.bind("<Escape>", lambda e: self._on_cancel())

        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        # Centre on parent
        self.update_idletasks()
        try:
            px, py = parent.winfo_rootx(), parent.winfo_rooty()
            pw, ph = parent.winfo_width(), parent.winfo_height()
        except Exception:
            px, py, pw, ph = 100, 100, 800, 600
        w, h = self.winfo_width(), self.winfo_height()
        self.geometry(f"+{px + (pw - w) // 2}+{py + (ph - h) // 2}")

        parent.wait_window(self)

    def _on_save(self) -> None:
        """Validate YAML and save if valid."""
        content = self._text.get("1.0", "end-1c")

        # Validate YAML syntax
        try:
            yaml.safe_load(content)
        except yaml.YAMLError as exc:
            messagebox.showerror(
                "Invalid YAML",
                f"The file contains YAML syntax errors and cannot be saved:\n\n{exc}",
                parent=self,
            )
            return

        try:
            self._settings_path.write_text(content, encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Save error", f"Could not write file:\n{exc}", parent=self)
            return

        self.saved = True
        self.grab_release()
        self.destroy()

    def _on_cancel(self) -> None:
        """Close without saving."""
        self.grab_release()
        self.destroy()
