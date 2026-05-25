"""Modal dialog for picking parameter_group names to include in an export.

Used by :class:`AddDialog` when the user adds an empty FlexTool input
Excel: the dialog shows every parameter_group from the source DB as a
checkbox row, with the *required* groups highlighted at the top and a
hover tooltip explaining why they matter.

Theming
-------
The dialog reads the active ttk theme's background/foreground colours
(via :class:`ttk.Style`) and applies them to the plain ``tk`` widgets it
uses for the row list, so it stays readable under both light and dark
themes — the rest of the FlexTool GUI uses the same pattern.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Sequence

from flextool.gui.hover_tooltip import attach_tooltip


_REQUIRED_TOOLTIP = (
    "Required for a functioning FlexTool model.\n"
    "You can still uncheck this — a working model can be assembled by "
    "combining data from multiple input sources in Spine Toolbox."
)

# Highlight tints chosen to read well on top of light and dark theme
# surfaces.  Both keep enough luminance for the forced-black text on
# required rows to stay legible.
_HIGHLIGHT_LIGHT = "#fff4c2"
_HIGHLIGHT_DARK = "#e8d488"


class GroupPickerDialog(tk.Toplevel):
    """Modal picker for parameter_group selection.

    After the dialog closes, ``self.result`` is one of:

    * ``None`` — user cancelled, do not export.
    * ``"all"`` — every group was checked; caller should pass
      ``include_groups=None`` to ``export_to_excel`` (preserves the
      original unfiltered semantics, including any ungrouped params).
    * ``list[str]`` — the explicitly selected group names.
    """

    def __init__(
        self,
        parent: tk.Misc,
        groups: Sequence[dict],
        required: Sequence[str],
    ) -> None:
        super().__init__(parent)
        self.title("Choose parameter groups")
        self.transient(parent)
        self.grab_set()
        self.resizable(False, True)

        self.result: list[str] | str | None = None

        # Pull theme colours so plain tk widgets match dark / light mode.
        style = ttk.Style(self)
        self._theme_bg = (
            style.lookup("TFrame", "background")
            or self.cget("background")
        )
        self._theme_fg = style.lookup("TLabel", "foreground") or "black"
        self._highlight_bg = (
            _HIGHLIGHT_DARK
            if _is_dark_color(self, self._theme_bg)
            else _HIGHLIGHT_LIGHT
        )

        # Build display order: required first (in YAML order), then
        # remaining groups sorted by priority then name.
        by_name = {g["name"]: g for g in groups}
        required_present = [name for name in required if name in by_name]
        required_set = set(required_present)
        remaining = [g for g in groups if g["name"] not in required_set]
        remaining.sort(key=lambda g: (g.get("priority") or 999, g["name"]))
        ordered: list[dict] = [by_name[n] for n in required_present] + remaining

        self._vars: dict[str, tk.BooleanVar] = {
            g["name"]: tk.BooleanVar(value=True) for g in ordered
        }
        self._all_names: list[str] = [g["name"] for g in ordered]

        self._build_ui(ordered, required_set)

        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self._center_on_parent(parent)
        parent.wait_window(self)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(
        self, ordered: list[dict], required_set: set[str],
    ) -> None:
        # Outer container — ttk.Frame inherits theme automatically.
        outer = ttk.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)

        intro = ttk.Label(
            outer,
            text=(
                "Select which parameter groups to include in the new Excel "
                "file.\nHighlighted rows are required for a functioning "
                "FlexTool model."
            ),
            justify="left",
        )
        intro.pack(anchor="w", pady=(0, 8))

        # Bulk action row
        bulk = ttk.Frame(outer)
        bulk.pack(fill="x", pady=(0, 6))
        ttk.Button(
            bulk, text="Select all", command=self._on_select_all,
        ).pack(side="left", padx=(0, 4))
        ttk.Button(
            bulk, text="Required only",
            command=lambda: self._on_required_only(required_set),
        ).pack(side="left", padx=(0, 4))
        ttk.Button(
            bulk, text="Clear all", command=self._on_clear_all,
        ).pack(side="left")

        # Scrollable group list — uses tk.Frame/Canvas so we can colour
        # the highlight rows.  Background pulled from the ttk theme.
        list_frame = tk.Frame(
            outer, bg=self._theme_bg, highlightthickness=1,
            highlightbackground="#888888",
        )
        list_frame.pack(fill="both", expand=True)

        # Canvas size derived from font metrics so the dialog scales with
        # the user's font size instead of being pinned at 360×570 px.
        from flextool.gui.ui_metrics import get_metrics
        _m = get_metrics(self)
        canvas = tk.Canvas(
            list_frame, bg=self._theme_bg, highlightthickness=0,
            width=_m.cw * 36, height=_m.lh * 30,
        )
        canvas.pack(side="left", fill="both", expand=True)

        sb = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        sb.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=sb.set)

        rows_frame = tk.Frame(canvas, bg=self._theme_bg)
        canvas.create_window((0, 0), window=rows_frame, anchor="nw")

        def _on_rows_configure(_event: tk.Event) -> None:  # type: ignore[type-arg]
            canvas.configure(scrollregion=canvas.bbox("all"))

        rows_frame.bind("<Configure>", _on_rows_configure)

        # Mouse wheel scroll
        def _on_mousewheel(event: tk.Event) -> None:  # type: ignore[type-arg]
            canvas.yview_scroll(-int(event.delta / 120), "units")

        canvas.bind("<Enter>", lambda _e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda _e: canvas.unbind_all("<MouseWheel>"))

        # Render rows
        for g in ordered:
            self._make_row(rows_frame, g, g["name"] in required_set)

        # OK / Cancel
        btn_row = ttk.Frame(outer)
        btn_row.pack(fill="x", pady=(10, 0))
        ttk.Button(btn_row, text="Cancel", command=self._on_cancel).pack(side="right")
        ttk.Button(btn_row, text="OK", command=self._on_ok).pack(side="right", padx=(0, 6))

    def _make_row(
        self, parent: tk.Widget, group: dict, is_required: bool,
    ) -> None:
        bg = self._highlight_bg if is_required else self._theme_bg
        # On highlight rows we also override the foreground so text stays
        # readable on the yellow tint regardless of the parent theme.
        fg = "#000000" if is_required else self._theme_fg

        row = tk.Frame(parent, bg=bg)
        row.pack(fill="x")

        # Color swatch
        swatch_color = _normalise_color(group.get("color"))
        swatch_frame = tk.Frame(row, bg=bg, width=22, height=22)
        swatch_frame.pack_propagate(False)
        swatch_frame.pack(side="left", padx=(8, 6), pady=2)
        swatch = tk.Frame(
            swatch_frame, bg=swatch_color or "#cccccc",
            highlightthickness=1, highlightbackground="#888888",
        )
        swatch.place(x=4, y=4, width=14, height=14)

        cb = tk.Checkbutton(
            row,
            text=group["name"],
            variable=self._vars[group["name"]],
            bg=bg, fg=fg,
            activebackground=bg, activeforeground=fg,
            selectcolor=bg,           # checkbox indicator interior
            highlightthickness=0,
            anchor="w",
        )
        cb.pack(side="left", fill="x", expand=True, padx=(0, 8))

        if is_required:
            attach_tooltip(row, _REQUIRED_TOOLTIP)
            attach_tooltip(cb, _REQUIRED_TOOLTIP)
            attach_tooltip(swatch, _REQUIRED_TOOLTIP)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_select_all(self) -> None:
        for v in self._vars.values():
            v.set(True)

    def _on_clear_all(self) -> None:
        for v in self._vars.values():
            v.set(False)

    def _on_required_only(self, required_set: set[str]) -> None:
        for name, v in self._vars.items():
            v.set(name in required_set)

    def _on_ok(self) -> None:
        selected = [n for n in self._all_names if self._vars[n].get()]
        if not selected:
            messagebox.showwarning(
                "No groups selected",
                "Select at least one group, or press Cancel.",
                parent=self,
            )
            return
        if len(selected) == len(self._all_names):
            self.result = "all"
        else:
            self.result = selected
        self.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self.destroy()

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def _center_on_parent(self, parent: tk.Misc) -> None:
        self.update_idletasks()
        try:
            pw = parent.winfo_width()
            ph = parent.winfo_height()
            px = parent.winfo_rootx()
            py = parent.winfo_rooty()
        except tk.TclError:
            return
        dw = self.winfo_width()
        dh = self.winfo_height()
        self.geometry(f"+{px + (pw - dw) // 2}+{py + (ph - dh) // 2}")


def _normalise_color(value: str | None) -> str | None:
    """Coerce a hex colour string to '#rrggbb' form for Tk, else None."""
    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None
    if s.startswith("#"):
        return s
    if len(s) in (6, 8) and all(c in "0123456789abcdefABCDEF" for c in s):
        return f"#{s[:6]}"
    return None


def _is_dark_color(widget: tk.Widget, color: str) -> bool:
    """Return True when *color* is a dark surface (so we should use a
    dark-theme highlight tint).

    Uses Tk's built-in colour parser so it accepts hex, rgb tuples,
    and X11 / system colour names.  Falls back to "light" on parse
    failure.
    """
    try:
        r, g, b = widget.winfo_rgb(color)
    except tk.TclError:
        return False
    # winfo_rgb returns 16-bit channels (0..65535).
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return luminance < 32768
