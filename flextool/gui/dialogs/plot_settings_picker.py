"""Non-modal color/order picker for a project's ``plot_settings.yaml``.

This is the GUI editor that replaces the old plain-text
``PlotSettingsEditor``.  It treats the project ``plot_settings.yaml`` as a
plain STRUCTURED data file (pyyaml load -> dict -> dump with
``sort_keys=False``).  The window is **non-modal** so it can be used at the
same time as the result viewer: the **Apply** button writes the file and
calls an ``on_apply`` callback (the viewer re-renders = live preview) while
the window stays open.

The window loads the file, renders one ``ttk.Notebook`` tab per present
section, lists every entry in a ``ttk.Treeview`` with a composite
``[pos][neg]`` swatch image next to its name, and wires Apply / Save and
exit / Cancel to the file + the ``on_apply`` callback.  Editing
interactions: reorder (drag + Alt-arrow), per-row color dialog
(double-click), **Refresh from DB** (re-fetch entity names from the
project's input DB(s) and add new + prune stale), and multi-level
**Undo/Redo** over the in-memory working dict.
"""

from __future__ import annotations

import base64
import colorsys
import copy
import logging
import struct
import tkinter as tk
import zlib
from collections.abc import Callable
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
import yaml

from flextool.scenario_comparison.plot_settings_seed import dump_plot_settings

logger = logging.getLogger(__name__)

# Entity classes, in the order they should appear as tabs.
_ENTITY_CLASSES = ("nodeGroup", "flowGroup", "unit", "connection", "node")
# Category subsections, in tab order.
_CATEGORY_SECTIONS = ("costs", "node_flows", "nodegroup_flows", "dispatch")

# Swatch geometry (pixels).  Entity rows show two boxes (positive | gap |
# negative); categories / scenarios show one.  ``_SWATCH_GAP`` separates the
# two boxes; ``_SWATCH_PAD_L`` / ``_SWATCH_PAD_R`` inset the boxes from the
# (indicator-less) cell edge and from the row name.
_SWATCH_H = 14
_SWATCH_W = 14  # width of ONE box
_SWATCH_GAP = 4  # transparent gap between the positive and negative boxes
_SWATCH_PAD_L = 3  # small left inset
_SWATCH_PAD_R = 6  # gap after the boxes, before the row name
_SWATCH_BORDER = (0x80, 0x80, 0x80)  # gray 1px frame so swatches read on any bg


def _swatch_width(two_box: bool) -> int:
    """Total swatch image width for a two-box (entity) or single row."""
    boxes = _SWATCH_W * 2 + _SWATCH_GAP if two_box else _SWATCH_W
    return _SWATCH_PAD_L + boxes + _SWATCH_PAD_R


def _to_rgb255(value) -> tuple[int, int, int]:
    """Best-effort parse of a YAML color value to an ``(r, g, b)`` 0..255 tuple.

    Accepts ``#RRGGBB`` hex strings, ``[r, g, b]`` lists (0..1 floats or
    0..255 ints), and matplotlib named colors (e.g. ``crimson``, ``aqua``
    used by the dispatch defaults).  Falls back to a neutral mid-gray for
    anything unparseable so a row always shows a swatch.
    """
    # Hex / list path reuses the canonical plot-side parser (0..1 floats).
    from flextool.plot_outputs.color_template import _parse_color_value

    rgb = _parse_color_value(value)
    if rgb is None and isinstance(value, str):
        # Named colors (matplotlib) — only the dispatch defaults use these.
        try:
            import matplotlib.colors as mcolors

            rgb = mcolors.to_rgb(value.strip())
        except (ValueError, ImportError):
            rgb = None
    if rgb is None:
        return (0x99, 0x99, 0x99)
    return (
        max(0, min(255, round(rgb[0] * 255))),
        max(0, min(255, round(rgb[1] * 255))),
        max(0, min(255, round(rgb[2] * 255))),
    )


def _resolve_pos_neg(value) -> tuple[object, object | None]:
    """Split an entity entry into ``(pos_value, neg_value_or_None)``.

    A bare color -> ``(color, None)`` (pos == neg).  A mapping
    ``{color, neg_color}`` -> ``(color, neg_color)``; a mapping with only
    ``color`` -> ``(color, None)``.
    """
    if isinstance(value, dict):
        return value.get("color"), value.get("neg_color")
    return value, None


def _to_hex(value) -> str:
    """Normalize any YAML color value to a lowercase ``#RRGGBB`` hex string.

    Reuses the same best-effort parser as the swatches so a named color
    (e.g. ``lime``) or an ``[r, g, b]`` list becomes a concrete hex the
    color chooser can seed its initial swatch from.
    """
    return "#%02x%02x%02x" % _to_rgb255(value)


def _photo_from_rgb(arr: np.ndarray) -> tk.PhotoImage:
    """Build a Tk ``PhotoImage`` from an ``(H, W, 3)`` uint8 RGB array.

    Encodes a minimal truecolor PNG (numpy + stdlib ``zlib``/``struct``,
    no Pillow) and hands the base64 to Tk, which reads PNG natively from
    8.6 on.  ~2 ms for a 168x168 square, fast enough for live hue drags.
    """
    h, w, _ = arr.shape
    rows = np.empty((h, 1 + w * 3), "uint8")
    rows[:, 0] = 0  # PNG filter type 0 (none) per scanline
    rows[:, 1:] = np.ascontiguousarray(arr).reshape(h, w * 3)

    def _chunk(typ: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + typ + data
            + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(rows.tobytes(), 6))
        + _chunk(b"IEND", b"")
    )
    return tk.PhotoImage(data=base64.b64encode(png))


def _sv_square_photo(hue: float, size: int) -> tk.PhotoImage:
    """Saturation (x, 0→1) × Value (y, top 1 → bottom 0) square at *hue*."""
    ss = np.linspace(0.0, 1.0, size)[None, :].repeat(size, 0)
    vv = np.linspace(1.0, 0.0, size)[:, None].repeat(size, 1)
    i = int(hue * 6.0) % 6
    f = hue * 6.0 - int(hue * 6.0)
    p = vv * (1.0 - ss)
    q = vv * (1.0 - f * ss)
    t = vv * (1.0 - (1.0 - f) * ss)
    rgb = [
        (vv, t, p), (q, vv, p), (p, vv, t),
        (p, q, vv), (t, p, vv), (vv, p, q),
    ][i]
    arr = (np.stack(rgb, -1) * 255.0 + 0.5).astype("uint8")
    return _photo_from_rgb(arr)


def _hue_bar_photo(width: int, height: int) -> tk.PhotoImage:
    """Vertical full-saturation/value hue ramp (top 0 → bottom 1)."""
    hues = np.linspace(0.0, 1.0, height)
    rgb = np.array([colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in hues])
    arr = (np.repeat(rgb[:, None, :], width, 1) * 255.0 + 0.5).astype("uint8")
    return _photo_from_rgb(arr)


class _ColorChooser(ttk.Frame):
    """An embedded HSV color picker (no OS dialog, no Pillow).

    A saturation/value square + a vertical hue slider (both drag-pickable)
    with a live "new" preview and editable Hex / R,G,B / H,S,V fields.  The
    canonical state is ``(h, s, v)`` floats in ``0..1``; RGB/Hex are derived.

    Callbacks:

    * ``on_change(hex)`` — fired on EVERY change (user or programmatic);
      used by the dialog to mirror a linked negative and update previews.
    * ``on_user_edit()`` — fired only on USER-initiated changes (square /
      slider / entry commit); used by the dialog to break the link when the
      user edits the negative.
    """

    SQ = 168          # square edge (px)
    HUE_W = 18        # hue-bar width (px)

    def __init__(
        self,
        parent: tk.Misc,
        initial_hex: str,
        on_change: Callable[[str], None] | None = None,
        on_user_edit: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._on_change = on_change
        self._on_user_edit = on_user_edit
        self._sq_img: tk.PhotoImage | None = None
        self._hue_img: tk.PhotoImage | None = None

        r, g, b = (c / 255.0 for c in _to_rgb255(initial_hex))
        self._h, self._s, self._v = colorsys.rgb_to_hsv(r, g, b)

        # ── Square + hue canvases ─────────────────────────────────
        self._sq = tk.Canvas(
            self, width=self.SQ, height=self.SQ, highlightthickness=1,
            highlightbackground="#808080", cursor="crosshair",
        )
        self._sq.grid(row=0, column=0, rowspan=8, sticky="nw")
        self._hue = tk.Canvas(
            self, width=self.HUE_W, height=self.SQ, highlightthickness=1,
            highlightbackground="#808080", cursor="sb_v_double_arrow",
        )
        self._hue.grid(row=0, column=1, rowspan=8, sticky="nw", padx=(6, 10))

        self._sq.bind("<ButtonPress-1>", self._on_square_drag)
        self._sq.bind("<B1-Motion>", self._on_square_drag)
        self._hue.bind("<ButtonPress-1>", self._on_hue_drag)
        self._hue.bind("<B1-Motion>", self._on_hue_drag)

        # ── Fields column ─────────────────────────────────────────
        fields = ttk.Frame(self)
        fields.grid(row=0, column=2, sticky="nw")

        self._preview = tk.Label(
            fields, width=10, height=2, relief="solid", borderwidth=1,
        )
        self._preview.grid(row=0, column=0, columnspan=2, sticky="w",
                           pady=(0, 8))

        self._entries: dict[str, ttk.Entry] = {}

        def _row(rownum: int, label: str, key: str, commit) -> None:
            ttk.Label(fields, text=label).grid(
                row=rownum, column=0, sticky="e", padx=(0, 4), pady=1,
            )
            ent = ttk.Entry(fields, width=10)
            ent.grid(row=rownum, column=1, sticky="w", pady=1)
            ent.bind("<Return>", commit)
            ent.bind("<FocusOut>", commit)
            self._entries[key] = ent

        _row(1, "Hex", "hex", self._commit_hex)
        _row(2, "R", "r", self._commit_rgb)
        _row(3, "G", "g", self._commit_rgb)
        _row(4, "B", "b", self._commit_rgb)
        _row(5, "H°", "h", self._commit_hsv)
        _row(6, "S%", "s", self._commit_hsv)
        _row(7, "V%", "v", self._commit_hsv)

        self._render_hue_bar()
        self._render_square()
        self._sync_all()

    # ── Public API ────────────────────────────────────────────────
    def get_hex(self) -> str:
        r, g, b = colorsys.hsv_to_rgb(self._h, self._s, self._v)
        return "#%02x%02x%02x" % (
            round(r * 255), round(g * 255), round(b * 255),
        )

    def set_hex(self, hex_str: str, *, user: bool = False) -> None:
        """Set the color from a hex string (programmatic mirror by default)."""
        r, g, b = (c / 255.0 for c in _to_rgb255(hex_str))
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        self._apply_hsv(h, s, v, user=user, redraw_square=True)

    # ── Rendering ─────────────────────────────────────────────────
    def _render_square(self) -> None:
        self._sq_img = _sv_square_photo(self._h, self.SQ)
        self._sq.delete("img")
        self._sq.create_image(0, 0, anchor="nw", image=self._sq_img, tags="img")
        self._sq.tag_lower("img")
        self._draw_square_marker()

    def _render_hue_bar(self) -> None:
        self._hue_img = _hue_bar_photo(self.HUE_W, self.SQ)
        self._hue.delete("img")
        self._hue.create_image(0, 0, anchor="nw", image=self._hue_img, tags="img")
        self._hue.tag_lower("img")
        self._draw_hue_marker()

    def _draw_square_marker(self) -> None:
        x = self._s * (self.SQ - 1)
        y = (1.0 - self._v) * (self.SQ - 1)
        self._sq.delete("marker")
        for col, rad in (("#000000", 6), ("#ffffff", 5)):
            self._sq.create_oval(
                x - rad, y - rad, x + rad, y + rad,
                outline=col, width=1, tags="marker",
            )

    def _draw_hue_marker(self) -> None:
        y = self._h * (self.SQ - 1)
        self._hue.delete("marker")
        self._hue.create_rectangle(
            0, y - 2, self.HUE_W, y + 2,
            outline="#000000", width=1, tags="marker",
        )
        self._hue.create_rectangle(
            1, y - 1, self.HUE_W - 1, y + 1,
            outline="#ffffff", width=1, tags="marker",
        )

    def _update_preview(self) -> None:
        self._preview.configure(background=self.get_hex())

    # ── Entry sync (no recursion: set via delete/insert) ──────────
    def _set_entry(self, key: str, text: str) -> None:
        ent = self._entries[key]
        had = str(ent.cget("state"))
        ent.configure(state="normal")
        ent.delete(0, "end")
        ent.insert(0, text)
        ent.configure(state=had)

    def _sync_entries(self) -> None:
        r, g, b = colorsys.hsv_to_rgb(self._h, self._s, self._v)
        self._set_entry("hex", self.get_hex())
        self._set_entry("r", str(round(r * 255)))
        self._set_entry("g", str(round(g * 255)))
        self._set_entry("b", str(round(b * 255)))
        self._set_entry("h", str(round(self._h * 360)))
        self._set_entry("s", str(round(self._s * 100)))
        self._set_entry("v", str(round(self._v * 100)))

    def _sync_all(self) -> None:
        self._draw_square_marker()
        self._draw_hue_marker()
        self._update_preview()
        self._sync_entries()

    # ── Central update ────────────────────────────────────────────
    def _apply_hsv(
        self, h: float, s: float, v: float, *,
        user: bool, redraw_square: bool,
    ) -> None:
        self._h = min(1.0, max(0.0, h))
        self._s = min(1.0, max(0.0, s))
        self._v = min(1.0, max(0.0, v))
        if redraw_square:
            self._render_square()
        self._sync_all()
        if self._on_change is not None:
            self._on_change(self.get_hex())
        if user and self._on_user_edit is not None:
            self._on_user_edit()

    # ── Canvas interaction ────────────────────────────────────────
    def _on_square_drag(self, event: tk.Event) -> None:
        s = min(1.0, max(0.0, event.x / (self.SQ - 1)))
        v = 1.0 - min(1.0, max(0.0, event.y / (self.SQ - 1)))
        self._apply_hsv(self._h, s, v, user=True, redraw_square=False)

    def _on_hue_drag(self, event: tk.Event) -> None:
        h = min(1.0, max(0.0, event.y / (self.SQ - 1)))
        self._apply_hsv(h, self._s, self._v, user=True, redraw_square=True)

    # ── Entry commits ─────────────────────────────────────────────
    def _commit_hex(self, _event=None) -> None:
        raw = self._entries["hex"].get().strip()
        rgb = _parse_hex_or_none(raw)
        if rgb is None:
            self._sync_entries()  # revert invalid text
            return
        h, s, v = colorsys.rgb_to_hsv(*(c / 255.0 for c in rgb))
        self._apply_hsv(h, s, v, user=True, redraw_square=True)

    def _commit_rgb(self, _event=None) -> None:
        try:
            r = _clamp_int(self._entries["r"].get(), 0, 255)
            g = _clamp_int(self._entries["g"].get(), 0, 255)
            b = _clamp_int(self._entries["b"].get(), 0, 255)
        except ValueError:
            self._sync_entries()
            return
        h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        self._apply_hsv(h, s, v, user=True, redraw_square=True)

    def _commit_hsv(self, _event=None) -> None:
        try:
            h = _clamp_int(self._entries["h"].get(), 0, 360) / 360.0
            s = _clamp_int(self._entries["s"].get(), 0, 100) / 100.0
            v = _clamp_int(self._entries["v"].get(), 0, 100) / 100.0
        except ValueError:
            self._sync_entries()
            return
        self._apply_hsv(h, s, v, user=True, redraw_square=True)


def _parse_hex_or_none(text: str) -> tuple[int, int, int] | None:
    """Parse ``#rrggbb`` / ``rrggbb`` to an ``(r, g, b)`` tuple, else None."""
    s = text.strip().lstrip("#")
    if len(s) != 6:
        return None
    try:
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except ValueError:
        return None


def _clamp_int(text: str, lo: int, hi: int) -> int:
    """Parse *text* to an int clamped to ``[lo, hi]`` (raises on non-int)."""
    return min(hi, max(lo, int(float(text.strip()))))


class ColorPickerDialog(tk.Toplevel):
    """Modal embedded color editor for one row.

    Hosts an embedded :class:`_ColorChooser` (saturation/value square + hue
    slider + numeric fields).  With ``single=False`` (the default, for entity
    rows) it shows TWO choosers side by side — left **positive**, right
    **negative** — under a **"Link negative to positive"** checkbox:

    * **linked** (checked): the negative mirrors the positive (and its value
      is ignored on OK).  Default for a bare ``"#color"`` entry.  The panel
      stays interactive — see below.
    * **unlinked** (unchecked): positive and negative are independent.
      Default for a ``{color, neg_color}`` entry.

    The negative panel is never greyed out: while linked its value has no
    meaning, but the moment the user edits it that is taken as intent to
    separate, so it **auto-breaks the link** (unchecks the box).  Re-checking
    RE-LINKS (``neg := pos``).

    With ``single=True`` (categories / scenarios — one color, no negative)
    only the positive chooser is shown; the negative panel and the link
    checkbox are hidden.

    The result is read from :pyattr:`result` after ``wait_window``:
    ``(pos_hex, neg_hex_or_None)`` on OK (``neg_hex`` is ``None`` when single
    or linked, else the explicit negative), or ``None`` on Cancel.
    """

    def __init__(
        self,
        parent: tk.Misc,
        name: str,
        pos_hex: str,
        neg_hex: str,
        linked: bool,
        *,
        single: bool = False,
    ) -> None:
        super().__init__(parent)
        self.title(f"Color — {name}")
        self.transient(parent)
        self.resizable(False, False)

        self._single = single
        self._linked = tk.BooleanVar(value=True if single else linked)
        self.result: tuple[str, str | None] | None = None

        body = ttk.Frame(self, padding=12)
        body.grid(row=0, column=0, sticky="nsew")

        if not single:
            ttk.Checkbutton(
                body,
                text="Link negative to positive",
                variable=self._linked,
                command=self._on_link_toggle,
            ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        panels = ttk.Frame(body)
        panels.grid(row=1, column=0, columnspan=2, sticky="nw")

        pos_panel = ttk.Frame(panels)
        pos_panel.grid(row=0, column=0, sticky="nw")
        if not single:
            ttk.Label(pos_panel, text="Positive").grid(
                row=0, column=0, sticky="w", pady=(0, 2),
            )
        self._pos_chooser = _ColorChooser(
            pos_panel, _to_hex(pos_hex), on_change=self._on_pos_change,
        )
        self._pos_chooser.grid(row=1, column=0, sticky="nw")

        self._neg_chooser: _ColorChooser | None = None
        if not single:
            neg_panel = ttk.Frame(panels)
            neg_panel.grid(row=0, column=1, sticky="nw", padx=(16, 0))
            ttk.Label(neg_panel, text="Negative").grid(
                row=0, column=0, sticky="w", pady=(0, 2),
            )
            self._neg_chooser = _ColorChooser(
                neg_panel, _to_hex(neg_hex),
                on_user_edit=self._on_neg_user_edit,
            )
            self._neg_chooser.grid(row=1, column=0, sticky="nw")

        # OK / Cancel.
        btns = ttk.Frame(self, padding=(12, 0, 12, 12))
        btns.grid(row=1, column=0, sticky="ew")
        ttk.Button(btns, text="Cancel", command=self._on_cancel).pack(
            side="right", padx=(5, 0),
        )
        self._ok_button = ttk.Button(
            btns, text="OK", command=self._on_ok, default="active",
        )
        self._ok_button.pack(side="right")

        if not single:
            self._apply_link_state()

        self.bind("<Return>", lambda _e: self._on_ok())
        self.bind("<Escape>", lambda _e: self._on_cancel())
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self._grab_when_viewable()

    def _grab_when_viewable(self) -> None:
        """Acquire the modal grab as soon as the window is mapped.

        ``grab_set`` on a not-yet-mapped Toplevel raises ``"grab failed:
        window not viewable"`` under some window managers (the Toplevel is
        created but the WM hasn't mapped it yet).  Retry on the Tk event
        loop until it maps, then take keyboard focus on OK (the Enter
        target).  Guarded against a window destroyed before it ever mapped.
        """
        if not self.winfo_exists():
            return
        try:
            self.grab_set()
        except tk.TclError:
            self.after(20, self._grab_when_viewable)
            return
        try:
            self._ok_button.focus_set()
        except tk.TclError:
            pass

    # ── Link logic ────────────────────────────────────────────────
    def _on_pos_change(self, hex_str: str) -> None:
        """Mirror the positive onto a linked negative (programmatic)."""
        if (
            not self._single
            and self._linked.get()
            and self._neg_chooser is not None
        ):
            self._neg_chooser.set_hex(hex_str, user=False)

    def _on_neg_user_edit(self) -> None:
        """A user edit to the negative separates the two colors → unlink."""
        if self._linked.get():
            self._linked.set(False)
            self._apply_link_state()

    def _on_link_toggle(self) -> None:
        """Checkbox toggled: re-check mirrors neg:=pos; uncheck frees neg."""
        self._apply_link_state()

    def _apply_link_state(self) -> None:
        """When linked, mirror the negative onto the positive.

        The negative panel stays interactive even while linked: its value
        is simply ignored on OK (linked → neg is None), and the moment the
        user touches it the link auto-breaks (see ``_on_neg_user_edit``).
        """
        if self._neg_chooser is None:
            return
        if self._linked.get():
            self._neg_chooser.set_hex(
                self._pos_chooser.get_hex(), user=False,
            )

    # ── Result ────────────────────────────────────────────────────
    def _on_ok(self) -> None:
        pos = self._pos_chooser.get_hex()
        if self._single or self._linked.get() or self._neg_chooser is None:
            self.result = (pos, None)
        else:
            self.result = (pos, self._neg_chooser.get_hex())
        self.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self.destroy()


class PlotSettingsPicker(tk.Toplevel):
    """Non-modal color/order picker for a project ``plot_settings.yaml``.

    Parameters
    ----------
    parent:
        Owning widget; the window is ``transient`` to it but NOT modal
        (no ``grab_set``), so the result viewer stays usable.
    settings_path:
        Path to the project ``plot_settings.yaml`` (already seeded by the
        caller via ``seed_plot_settings``).
    on_apply:
        Optional zero-arg callback invoked after every successful write
        (Apply / Save and exit) and after a Cancel restore, so an opener
        with a live preview (the result viewer) re-renders with the
        current on-disk colors.  Pass ``None`` for no live preview (the
        PNG batch dialog).
    """

    def __init__(
        self,
        parent: tk.Misc,
        settings_path: Path,
        on_apply: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.title("Colors and order")
        self._settings_path = Path(settings_path)
        self._on_apply = on_apply

        # Non-modal: transient (stacks with the parent, no taskbar entry on
        # some WMs) but NO grab_set / wait_window — the viewer stays live.
        self.transient(parent)

        # Strong references to every PhotoImage so Tk does not GC them
        # (a GC'd image renders blank in the Treeview cell).
        self._swatches: list[tk.PhotoImage] = []

        # Per-tree reorder bookkeeping: maps a Treeview to the
        # ``(*keys,)`` path of its section in the working dict (e.g.
        # ``("entities", "unit")`` or ``("scenarios",)``) so a reorder can
        # rewrite exactly that section.  Drag state distinguishes a MOVE
        # drag (press began on an already-selected row → move the whole
        # selection) from a native draw-select drag (press on an unselected
        # row → let ttk extend the selection).  ``_drag_anchor`` is the
        # pressed row; ``_drag_moved`` records whether the drag actually
        # reordered (so a click-without-move can collapse the selection).
        self._tree_section: dict[ttk.Treeview, tuple[str, ...]] = {}
        self._drag_move: dict[ttk.Treeview, bool] = {}
        self._drag_anchor: dict[ttk.Treeview, str | None] = {}
        self._drag_moved: dict[ttk.Treeview, bool] = {}
        # Whether a tree's rows carry a positive AND negative color
        # (entities) or a single color (categories / scenarios).  Drives
        # the double-click edit dialog and the swatch rebuild.
        self._tree_composite: dict[ttk.Treeview, bool] = {}
        # Per-row PhotoImage references after an in-place swatch rebuild,
        # keyed by ``(tree, item_id)`` so the replacement survives GC
        # (the originals in ``self._swatches`` stay too).
        self._row_swatches: dict[tuple[ttk.Treeview, str], tk.PhotoImage] = {}

        # Working state = parsed dict; snapshot the original TEXT for Cancel.
        self._original_text = self._read_text()
        self._data = self._parse(self._original_text)

        # In-memory edit history (deep copies of ``self._data``).  Every
        # mutator calls ``_snapshot()`` (push pre-edit state, clear redo)
        # before it changes ``self._data``; undo/redo move states between
        # the two stacks and rebuild every tab from ``self._data``.  This
        # history is independent of the on-disk Cancel snapshot above.
        self._undo_stack: list[dict] = []
        self._redo_stack: list[dict] = []
        # Set by undo/redo button construction; refreshed by
        # ``_update_history_buttons`` to reflect stack emptiness.
        self._undo_button: ttk.Button | None = None
        self._redo_button: ttk.Button | None = None

        # ── Sizing ────────────────────────────────────────────────
        from flextool.gui.ui_metrics import get_metrics

        _metrics = get_metrics(self)
        cw = _metrics.cw
        lh = _metrics.lh
        self.geometry(f"{cw * 70}x{lh * 34}")
        self.resizable(True, True)
        self.minsize(cw * 40, lh * 16)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._tree_style = self._make_tree_style()

        # ── Notebook with one tab per present section ─────────────
        self._notebook = ttk.Notebook(self)
        self._notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 4))
        self._build_tabs()

        # ── Buttons ───────────────────────────────────────────────
        # Buttons are CREATED in tab-traversal order (Refresh → Undo → Redo
        # → Apply → Save and exit → Cancel) — Tk's focus traversal follows
        # the parent's child order.  ``grid`` then places them visually
        # (left cluster / flexible spacer / right cluster), decoupling the
        # on-screen layout from the traversal order.
        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(4, 10))
        btn_frame.columnconfigure(3, weight=1)  # spacer between clusters

        ttk.Button(
            btn_frame, text="Refresh from DB", command=self._on_refresh,
        ).grid(row=0, column=0)
        self._undo_button = ttk.Button(
            btn_frame, text="Undo", command=self._on_undo,
        )
        self._undo_button.grid(row=0, column=1, padx=(5, 0))
        self._redo_button = ttk.Button(
            btn_frame, text="Redo", command=self._on_redo,
        )
        self._redo_button.grid(row=0, column=2, padx=(5, 0))
        ttk.Button(
            btn_frame, text="Apply", command=self._on_apply_clicked,
        ).grid(row=0, column=4, padx=(5, 0))
        ttk.Button(
            btn_frame, text="Save and exit", command=self._on_save_exit,
        ).grid(row=0, column=5, padx=(5, 0))
        ttk.Button(
            btn_frame, text="Cancel", command=self._on_cancel,
        ).grid(row=0, column=6, padx=(5, 0))
        self._update_history_buttons()

        # ── Keyboard-shortcut hint strip (two rows) ───────────────
        ttk.Label(
            self,
            text=(
                "Enter: edit selected row    "
                "Alt+↑ / Alt+↓: move row    drag: reorder\n"
                "Ctrl+Enter: Apply    "
                "Ctrl+Z: undo    Ctrl+Y: redo    Esc: close"
            ),
            foreground="gray",
            anchor="w",
            justify="left",
        ).grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 8))

        self.bind("<Escape>", lambda _e: self._on_cancel())
        # Ctrl+Enter applies the whole tool (write + re-render the open
        # plot), distinct from a plain Enter (edit the selected row).
        self.bind("<Control-Return>", lambda _e: self._on_apply_clicked())
        self.bind("<Control-KP_Enter>", lambda _e: self._on_apply_clicked())
        self.bind("<Control-z>", lambda _e: self._on_undo())
        self.bind("<Control-y>", lambda _e: self._on_redo())
        self.bind("<Control-Shift-Z>", lambda _e: self._on_redo())
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

    # ── File I/O ──────────────────────────────────────────────────
    def _read_text(self) -> str:
        try:
            return self._settings_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Cannot read %s: %s", self._settings_path, exc)
            return ""

    @staticmethod
    def _parse(text: str) -> dict:
        try:
            parsed = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            logger.warning("Cannot parse plot_settings.yaml: %s", exc)
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _write(self) -> None:
        """Dump the working dict back to the project file."""
        self._settings_path.write_text(
            dump_plot_settings(self._data), encoding="utf-8",
        )

    # ── Swatches ──────────────────────────────────────────────────
    def _make_swatch(self, pos, neg=None, reserve_neg: bool = False) -> tk.PhotoImage:
        """Build a swatch ``PhotoImage`` for one row.

        Two fixed columns of boxes: the LEFT (positive) box is always drawn;
        the RIGHT (negative) box is drawn only when *neg* is set.  On entity
        rows pass ``reserve_neg=True`` so the image is always two boxes wide
        even when *neg* is ``None`` — the negative half is left transparent,
        which keeps every row's name text aligned and makes the negative box
        appear/disappear in a fixed column.  Categories / scenarios (no
        negative concept) pass ``reserve_neg=False`` for a single box.

        The returned image is appended to ``self._swatches`` so it survives
        garbage collection (a GC'd image renders blank in the cell).
        """
        pos_rgb = _to_rgb255(pos)
        has_neg = neg is not None
        two_box = has_neg or reserve_neg
        width = _swatch_width(two_box)
        img = tk.PhotoImage(width=width, height=_SWATCH_H)

        def _fill(x0: int, x1: int, rgb: tuple[int, int, int]) -> None:
            color = "#%02x%02x%02x" % rgb
            border = "#%02x%02x%02x" % _SWATCH_BORDER
            for x in range(x0, x1):
                for y in range(_SWATCH_H):
                    edge = (
                        x == x0 or x == x1 - 1
                        or y == 0 or y == _SWATCH_H - 1
                    )
                    img.put(border if edge else color, (x, y))

        _fill(_SWATCH_PAD_L, _SWATCH_PAD_L + _SWATCH_W, pos_rgb)
        if has_neg:
            nx = _SWATCH_PAD_L + _SWATCH_W + _SWATCH_GAP
            _fill(nx, nx + _SWATCH_W, _to_rgb255(neg))
        # else: a reserved-but-unset negative box stays transparent, so it is
        # invisible while the column (and name alignment) is kept.

        self._swatches.append(img)
        return img

    # ── Edit history (undo / redo) ────────────────────────────────
    def _snapshot(self) -> None:
        """Push the PRE-edit ``self._data`` onto the undo stack.

        Called by EVERY mutator (color write-back, reorder sync, refresh)
        before it changes ``self._data``, so the undo stack always holds the
        states to roll back to.  A fresh edit invalidates any redo history,
        so the redo stack is cleared here.  States are deep copies so later
        in-place edits to ``self._data`` cannot mutate them.
        """
        self._undo_stack.append(copy.deepcopy(self._data))
        self._redo_stack.clear()
        self._update_history_buttons()

    def _on_undo(self) -> None:
        """Roll back one edit: restore the last pre-edit ``self._data``."""
        if not self._undo_stack:
            return
        self._redo_stack.append(copy.deepcopy(self._data))
        self._data = self._undo_stack.pop()
        self._rebuild_all_tabs()
        self._update_history_buttons()

    def _on_redo(self) -> None:
        """Re-apply one undone edit."""
        if not self._redo_stack:
            return
        self._undo_stack.append(copy.deepcopy(self._data))
        self._data = self._redo_stack.pop()
        self._rebuild_all_tabs()
        self._update_history_buttons()

    def _update_history_buttons(self) -> None:
        """Enable/disable Undo/Redo to reflect each stack's emptiness."""
        if self._undo_button is not None:
            self._undo_button.configure(
                state="normal" if self._undo_stack else "disabled",
            )
        if self._redo_button is not None:
            self._redo_button.configure(
                state="normal" if self._redo_stack else "disabled",
            )

    def _rebuild_all_tabs(self) -> None:
        """Discard every tab/tree and rebuild them from ``self._data``.

        Used by undo/redo/refresh so the displayed trees + swatches exactly
        match the (possibly replaced) working dict — order, colors, and row
        presence.  All per-tree bookkeeping and swatch references are reset
        so stale rows/images cannot leak across a rebuild.
        """
        for tab_id in self._notebook.tabs():
            self._notebook.forget(tab_id)
            self.nametowidget(tab_id).destroy()
        self._tree_section.clear()
        self._drag_move.clear()
        self._drag_anchor.clear()
        self._drag_moved.clear()
        self._tree_composite.clear()
        self._row_swatches.clear()
        self._swatches.clear()
        self._build_tabs()

    # ── Tab construction ──────────────────────────────────────────
    def _make_tree_style(self) -> str:
        """Return a Treeview style whose item layout drops the disclosure
        indicator, so leaf rows start flush-left (no ~18px indent) — the
        swatch carries its own small inset instead.  Falls back to the
        default ``"Treeview"`` style if the layout override is rejected.
        """
        name = "PlotSettingsPicker.Treeview"
        style = ttk.Style(self)
        try:
            style.layout(name, style.layout("Treeview"))
            style.layout(
                f"{name}.Item",
                [("Treeitem.padding", {"sticky": "nswe", "children": [
                    ("Treeitem.image", {"side": "left", "sticky": ""}),
                    ("Treeitem.text", {"side": "left", "sticky": ""}),
                ]})],
            )
        except tk.TclError:
            return "Treeview"
        return name

    def _build_tabs(self) -> None:
        """Create the tabs.

        Entity-class tabs (nodeGroup / flowGroup / unit / connection / node)
        are ALWAYS shown — even when empty — so it is visible that a class
        has no entities (and "Refresh from DB" can populate it).  Category
        and scenario tabs are shown only when present.
        """
        entities = self._data.get("entities")
        entities = entities if isinstance(entities, dict) else {}
        for cls in _ENTITY_CLASSES:
            section = entities.get(cls)
            rows = list(section.items()) if isinstance(section, dict) else []
            self._add_tab(
                title=cls,
                rows=rows,
                composite=True,
                section_path=("entities", cls),
            )

        categories = self._data.get("categories")
        if isinstance(categories, dict):
            for name in _CATEGORY_SECTIONS:
                section = categories.get(name)
                if isinstance(section, dict) and section:
                    self._add_tab(
                        title=name,
                        rows=list(section.items()),
                        composite=False,
                        section_path=("categories", name),
                    )

        scenarios = self._data.get("scenarios")
        if isinstance(scenarios, dict) and scenarios:
            self._add_tab(
                title="scenarios",
                rows=list(scenarios.items()),
                composite=False,
                section_path=("scenarios",),
            )

    def _add_tab(
        self,
        title: str,
        rows: list[tuple[str, object]],
        composite: bool,
        section_path: tuple[str, ...],
    ) -> None:
        """Add a Notebook tab with a scrollable single-column Treeview."""
        frame = ttk.Frame(self._notebook)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        tree = ttk.Treeview(
            frame, show="tree", selectmode="extended", style=self._tree_style,
        )
        tree.grid(row=0, column=0, sticky="nsew")

        vscroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        vscroll.grid(row=0, column=1, sticky="ns")
        tree.configure(yscrollcommand=vscroll.set)

        hscroll = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        hscroll.grid(row=1, column=0, sticky="ew")
        tree.configure(xscrollcommand=hscroll.set)

        for name, value in rows:
            if composite:
                pos, neg = _resolve_pos_neg(value)
                image = self._make_swatch(pos, neg, reserve_neg=True)
            else:
                image = self._make_swatch(value, None)
            tree.insert("", "end", text=str(name), image=image)

        # Register for reordering and wire drag + keyboard moves.
        self._tree_section[tree] = section_path
        self._drag_move[tree] = False
        self._drag_anchor[tree] = None
        self._drag_moved[tree] = False
        self._tree_composite[tree] = composite
        tree.bind("<ButtonPress-1>", self._on_drag_start)
        tree.bind("<B1-Motion>", self._on_drag_motion)
        tree.bind("<ButtonRelease-1>", self._on_drag_end)
        tree.bind("<Alt-Up>", self._on_key_move_up)
        tree.bind("<Alt-Down>", self._on_key_move_down)
        tree.bind("<Double-Button-1>", self._on_row_double_click)
        tree.bind("<Return>", self._on_row_return)
        tree.bind("<FocusIn>", self._on_tree_focus_in)
        # Ctrl+Enter must Apply even with the tree focused: the plain
        # ``<Return>`` (no-modifier) binding above otherwise also matches a
        # Ctrl+Return and consumes it (returns "break"), so bind the more
        # specific accelerator on the tree too — it wins on this widget.
        tree.bind("<Control-Return>", self._on_apply_shortcut)
        tree.bind("<Control-KP_Enter>", self._on_apply_shortcut)

        self._notebook.add(frame, text=title)

    # ── Reordering (drag + keyboard) ──────────────────────────────
    def _selected_rows(self, tree: ttk.Treeview) -> list[str]:
        """Current selection in top-to-bottom row order (focus as fallback)."""
        sel = set(tree.selection())
        if not sel:
            f = tree.focus()
            if f:
                sel = {f}
        return [r for r in tree.get_children("") if r in sel]

    def _reorder_selection_block(
        self, tree: ttk.Treeview, insert_at: int,
    ) -> bool:
        """Place the selected rows as a contiguous block at *insert_at*.

        *insert_at* is an index into the rows that are NOT selected (so the
        block is dropped between them).  Preserves each row's value, keeps
        the selection, and returns True iff the order actually changed.
        """
        order = list(tree.get_children(""))
        selected = self._selected_rows(tree)
        if not selected:
            return False
        remaining = [r for r in order if r not in set(selected)]
        insert_at = max(0, min(insert_at, len(remaining)))
        new_order = remaining[:insert_at] + selected + remaining[insert_at:]
        if new_order == order:
            return False
        for i, row in enumerate(new_order):
            tree.move(row, "", i)
        tree.selection_set(selected)
        tree.focus(selected[0])
        return True

    def _on_drag_start(self, event: tk.Event) -> str | None:
        """Begin a MOVE drag only when the press is on a selected row.

        Pressing an already-selected row starts moving the whole selection
        (return ``"break"`` to keep the multi-selection intact).  Pressing
        an unselected row / empty space is left to ttk's default handler so
        a click selects and a drag DRAW-selects a range.
        """
        tree = event.widget
        if tree not in self._tree_section:
            return None
        row = tree.identify_row(event.y) or None
        self._drag_moved[tree] = False
        if row and row in set(tree.selection()):
            self._drag_move[tree] = True
            self._drag_anchor[tree] = row
            return "break"
        self._drag_move[tree] = False
        self._drag_anchor[tree] = None
        return None

    def _on_drag_motion(self, event: tk.Event) -> None:
        """Move the selection to follow the cursor (move drags only)."""
        tree = event.widget
        if not self._drag_move.get(tree):
            return  # native draw-select drag — do not interfere
        target = tree.identify_row(event.y)
        if not target:
            return
        order = list(tree.get_children(""))
        selected = set(self._selected_rows(tree))
        target_idx = order.index(target)
        # Drop the block where the cursor is: count non-selected rows above
        # the target row to get the insertion slot among them.
        insert_at = sum(
            1 for r in order[:target_idx] if r not in selected
        )
        if target not in selected:
            insert_at += 1  # land the block just past the hovered row
        if self._reorder_selection_block(tree, insert_at):
            self._drag_moved[tree] = True
            tree.see(target)

    def _on_drag_end(self, event: tk.Event) -> None:
        """Finish a drag: persist a move, or collapse a no-move click."""
        tree = event.widget
        was_move = self._drag_move.get(tree)
        self._drag_move[tree] = False
        if not was_move:
            return
        if self._drag_moved.get(tree):
            section_path = self._tree_section.get(tree)
            if section_path is not None:
                self._sync_section_order_from_tree(section_path, tree)
        else:
            # A plain click on a selected row (no drag) collapses the
            # multi-selection down to that one row, as a normal click would.
            anchor = self._drag_anchor.get(tree)
            if anchor and tree.exists(anchor):
                tree.selection_set(anchor)
                tree.focus(anchor)

    def _on_key_move_up(self, event: tk.Event) -> str:
        """Alt+Up: move the selected row(s) up one position."""
        return self._key_move(event.widget, -1)

    def _on_key_move_down(self, event: tk.Event) -> str:
        """Alt+Down: move the selected row(s) down one position."""
        return self._key_move(event.widget, +1)

    def _key_move(self, tree: ttk.Treeview, delta: int) -> str:
        """Shift the selected row(s) by ``delta`` as a group; persist order.

        Returns ``"break"`` so Tk's default Alt-arrow handling does not
        also fire.
        """
        if tree not in self._tree_section:
            return "break"
        order = list(tree.get_children(""))
        selected = self._selected_rows(tree)
        if not selected:
            return "break"
        sset = set(selected)
        first = order.index(selected[0])
        last = order.index(selected[-1])
        if delta < 0 and first == 0:
            return "break"
        if delta > 0 and last == len(order) - 1:
            return "break"
        # Unselected rows above the block; the block jumps one such row
        # up (insert_at - 1) or down (insert_at + 1) among the others.
        before = sum(1 for r in order[:first] if r not in sset)
        insert_at = before - 1 if delta < 0 else before + 1
        if self._reorder_selection_block(tree, insert_at):
            tree.see(selected[0] if delta < 0 else selected[-1])
            self._sync_section_order_from_tree(self._tree_section[tree], tree)
        return "break"

    def _sync_section_order_from_tree(
        self, section_path: tuple[str, ...], tree: ttk.Treeview,
    ) -> None:
        """Rebuild ``section_path`` in the working dict to match the tree.

        The new dict preserves each entry's VALUE (bare color or
        ``{color, neg_color}``) and only reorders the keys to match the
        tree's current top-to-bottom row order.  Centralised so 6.5's
        undo can reuse it.
        """
        # Resolve the existing section mapping (the value source of truth).
        section = self._data
        for key in section_path:
            if not isinstance(section, dict):
                return
            section = section.get(key)
        if not isinstance(section, dict):
            return

        # Tree row order, by name (column #0 text).
        ordered_names = [
            tree.item(iid, "text") for iid in tree.get_children("")
        ]
        # Rebuild preserving values; keep any keys not represented as rows
        # (defensive — should not happen) appended in their original order.
        rebuilt: dict[str, object] = {}
        for name in ordered_names:
            if name in section:
                rebuilt[name] = section[name]
        for name, value in section.items():
            if name not in rebuilt:
                rebuilt[name] = value

        # A reorder that did not actually change the key order (e.g. a press
        # + release with no motion) is a no-op — do not record an undo step.
        if list(rebuilt.keys()) == list(section.keys()):
            return

        # Record the pre-edit state, then write back into the parent
        # container so the change is in-place for the dict Apply/Save dump.
        self._snapshot()
        parent = self._data
        for key in section_path[:-1]:
            parent = parent[key]
        parent[section_path[-1]] = rebuilt

    # ── Color editing (double-click / Enter) ──────────────────────
    def _restore_tree_focus(self, tree: ttk.Treeview, item: str) -> None:
        """Return keyboard focus to *tree* and re-select/refocus *item*.

        Called after a color dialog closes so the focus lands back in the
        list (not lost to the window) and Enter can edit the next row.
        """
        tree.focus_set()
        if item and tree.exists(item):
            tree.selection_set(item)
            tree.focus(item)
            tree.see(item)

    def _on_apply_shortcut(self, _event: tk.Event | None = None) -> str:
        """Ctrl+Enter from a tree → Apply (write + re-render), consume key."""
        self._on_apply_clicked()
        return "break"

    def _on_tree_focus_in(self, event: tk.Event) -> None:
        """Activate a row when the tree gains focus (e.g. via Tab).

        Without an active row the arrow / Alt-arrow keys have nothing to act
        on.  Tk's item-focus persists across focus changes, so reuse the
        last-active row when it still exists, otherwise activate the first.
        """
        tree = event.widget
        if tree not in self._tree_section:
            return
        item = tree.focus()
        if not item or not tree.exists(item):
            children = tree.get_children("")
            if not children:
                return
            item = children[0]
        tree.focus(item)
        tree.selection_set(item)
        tree.see(item)

    def _on_row_return(self, event: tk.Event) -> str:
        """Enter on a row opens its color editor (keyboard parity with the
        double-click)."""
        tree = event.widget
        if tree not in self._tree_section:
            return ""
        item = tree.focus() or (
            tree.selection()[0] if tree.selection() else ""
        )
        if not item:
            return "break"
        self._edit_row_color(tree, item)
        return "break"

    def _on_row_double_click(self, event: tk.Event) -> str:
        """Open the color editor for the double-clicked row.

        Resolves the row under the cursor; a double-click on empty space is
        ignored.  Returns ``"break"`` so the click is consumed and does not
        feed the drag/select bindings from 6.3.
        """
        tree = event.widget
        if tree not in self._tree_section:
            return ""
        item = tree.identify_row(event.y)
        if not item:
            return "break"
        # A double-click must not leave a stale move-drag primed.
        self._drag_move[tree] = False
        self._edit_row_color(tree, item)
        return "break"

    def _section_dict(self, section_path: tuple[str, ...]) -> dict | None:
        """Resolve a section path to its mapping in the working dict."""
        section = self._data
        for key in section_path:
            if not isinstance(section, dict):
                return None
            section = section.get(key)
        return section if isinstance(section, dict) else None

    def _edit_row_color(self, tree: ttk.Treeview, item: str) -> None:
        """Open the appropriate color dialog and write the result back."""
        section_path = self._tree_section[tree]
        section = self._section_dict(section_path)
        if section is None:
            return
        name = tree.item(item, "text")
        if name not in section:
            return
        value = section[name]

        if self._tree_composite.get(tree, False):
            self._edit_entity_color(tree, item, section, name, value)
        else:
            self._edit_single_color(tree, item, section, name, value)

    def _edit_entity_color(
        self,
        tree: ttk.Treeview,
        item: str,
        section: dict,
        name: str,
        value,
    ) -> None:
        """Edit a pos/neg entity entry via the modal lock dialog."""
        pos_val, neg_val = _resolve_pos_neg(value)
        # A bare entry (no explicit negative) opens LINKED.
        linked = neg_val is None
        pos_hex = _to_hex(pos_val)
        neg_hex = _to_hex(neg_val if neg_val is not None else pos_val)

        dialog = ColorPickerDialog(self, name, pos_hex, neg_hex, linked)
        self.wait_window(dialog)
        self._restore_tree_focus(tree, item)
        if dialog.result is None:
            return  # Cancel: no change.

        new_pos, new_neg = dialog.result
        # Record the pre-edit state before mutating the working dict.
        self._snapshot()
        if new_neg is None:
            section[name] = new_pos
            self._rebuild_row_swatch(tree, item, new_pos, None)
        else:
            section[name] = {"color": new_pos, "neg_color": new_neg}
            self._rebuild_row_swatch(tree, item, new_pos, new_neg)

    def _edit_single_color(
        self,
        tree: ttk.Treeview,
        item: str,
        section: dict,
        name: str,
        value,
    ) -> None:
        """Edit a single-color (category / scenario) entry.

        Uses the same embedded picker as entities but in single mode (the
        negative chooser + link checkbox are hidden), so categories /
        scenarios get the same look without a negative concept.
        """
        cur = _to_hex(value)
        dialog = ColorPickerDialog(self, name, cur, cur, True, single=True)
        self.wait_window(dialog)
        self._restore_tree_focus(tree, item)
        if dialog.result is None:
            return  # Cancel: no change.
        new_color = dialog.result[0]
        # Record the pre-edit state before mutating the working dict.
        self._snapshot()
        section[name] = new_color
        self._rebuild_row_swatch(tree, item, new_color, None)

    def _rebuild_row_swatch(
        self,
        tree: ttk.Treeview,
        item: str,
        pos,
        neg,
    ) -> None:
        """Rebuild and re-attach a row's swatch image in place."""
        image = self._make_swatch(
            pos, neg, reserve_neg=self._tree_composite.get(tree, False),
        )
        # Keep a per-row reference so the replacement is not GC'd (the
        # superseded image stays referenced in ``self._swatches`` too, but
        # is no longer displayed).
        self._row_swatches[(tree, item)] = image
        tree.item(item, image=image)

    # ── Refresh from the input DB ─────────────────────────────────
    def _discover_input_dbs(self) -> list[str]:
        """Return ``sqlite:///`` URLs for the project's input/intermediate DBs.

        The project root is the settings file's parent directory.  Every
        ``*.sqlite`` under ``<project>/input_sources`` and
        ``<project>/intermediate`` is a candidate input DB; URLs are
        returned in a stable (sorted) order so the union is deterministic.
        """
        project_root = self._settings_path.parent
        urls: list[str] = []
        for sub in ("input_sources", "intermediate"):
            db_dir = project_root / sub
            if not db_dir.is_dir():
                continue
            for path in sorted(db_dir.glob("*.sqlite")):
                if path.is_file():
                    urls.append(f"sqlite:///{path}")
        return urls

    def _fetch_entity_union(self, db_urls: list[str]) -> dict[str, set[str]]:
        """Union per-class entity names across every input DB (one open each).

        Reuses
        :func:`flextool.scenario_comparison.input_entity_colors.fetch_entities_by_class`
        — one :class:`DatabaseMapping` open per DB — and unions the returned
        per-class name sets across DBs.
        """
        from flextool.scenario_comparison.input_entity_colors import (
            RELEVANT_ENTITY_CLASSES,
            fetch_entities_by_class,
        )

        union: dict[str, set[str]] = {
            cls: set() for cls in RELEVANT_ENTITY_CLASSES
        }
        for url in db_urls:
            per_db = fetch_entities_by_class(url)
            for cls, names in per_db.items():
                union[cls].update(names)
        return union

    def _on_refresh(self) -> None:
        """Re-fetch entities from the project DB(s): ADD new + PRUNE stale.

        Discovers the project's input DB(s), unions their per-class entity
        names, and updates ``self._data['entities']`` in place: discovered
        names not already present are appended with a default-palette color
        (:func:`assign_palette_colors`); existing entries no longer in the DB
        are removed; surviving entries keep their order and (edited) values.
        ``categories`` and ``scenarios`` are never touched.  Records one undo
        step and rebuilds the entity tabs.  Shows an info box and changes
        nothing when no input DB is found.
        """
        from flextool.scenario_comparison.config_builder import (
            assign_palette_colors,
        )
        from flextool.scenario_comparison.input_entity_colors import (
            RELEVANT_ENTITY_CLASSES,
        )

        db_urls = self._discover_input_dbs()
        if not db_urls:
            messagebox.showinfo(
                "Refresh from DB",
                "No input database found.",
                parent=self,
            )
            return

        union = self._fetch_entity_union(db_urls)

        # Build the new entities mapping per class: keep existing entries (in
        # order, with their values) that still exist in the DB, append newly
        # discovered names with a palette color.  Prune the rest.
        entities = self._data.get("entities")
        if not isinstance(entities, dict):
            entities = {}

        new_entities: dict[str, dict] = {}
        changed = False
        for cls in RELEVANT_ENTITY_CLASSES:
            discovered = union.get(cls, set())
            existing = entities.get(cls)
            existing = existing if isinstance(existing, dict) else {}

            rebuilt: dict[str, object] = {}
            # Preserve existing entries (order + value) still in the DB.
            for name, value in existing.items():
                if name in discovered:
                    rebuilt[name] = value
                else:
                    changed = True  # pruned a stale entry
            # Append newly discovered names with a palette color.
            new_names = sorted(n for n in discovered if n not in rebuilt)
            if new_names:
                changed = True
                for name, color in assign_palette_colors(new_names).items():
                    rebuilt[name] = color

            if rebuilt:
                new_entities[cls] = rebuilt
            elif cls in existing and existing:
                # The class lost all its entities; dropping it is a change.
                changed = True

        if not changed:
            return

        self._snapshot()
        if new_entities:
            self._data["entities"] = new_entities
        else:
            self._data.pop("entities", None)
        self._rebuild_all_tabs()

    # ── Buttons ───────────────────────────────────────────────────
    def _on_apply_clicked(self) -> None:
        """Write the working dict and re-render the preview; stay open."""
        self._write()
        if self._on_apply is not None:
            self._on_apply()

    def _on_save_exit(self) -> None:
        """Write, re-render the preview, and close."""
        self._write()
        if self._on_apply is not None:
            self._on_apply()
        self.destroy()

    def _on_cancel(self) -> None:
        """Restore the on-open file text, revert any preview, and close."""
        try:
            self._settings_path.write_text(
                self._original_text, encoding="utf-8",
            )
        except OSError as exc:
            logger.warning(
                "Cannot restore %s on cancel: %s", self._settings_path, exc,
            )
        if self._on_apply is not None:
            self._on_apply()
        self.destroy()
