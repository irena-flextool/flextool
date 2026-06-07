"""Non-modal color/order picker for a project's ``plot_settings.yaml``.

This is the GUI editor that replaces the old plain-text
``PlotSettingsEditor``.  It treats the project ``plot_settings.yaml`` as a
plain STRUCTURED data file (pyyaml load -> dict -> dump with
``sort_keys=False``).  The window is **non-modal** so it can be used at the
same time as the result viewer: the **Apply** button writes the file and
calls an ``on_apply`` callback (the viewer re-renders = live preview) while
the window stays open.

This sub-commit (Stage 6.2) is the WINDOW SKELETON only: load the file,
render one ``ttk.Notebook`` tab per present section, list every entry in a
``ttk.Treeview`` with a composite ``[pos][neg]`` swatch image next to its
name, and wire the three buttons (Apply / Save and exit / Cancel) to the
file + the ``on_apply`` callback.  EDITING interactions — reordering, the
color-picker dialog, refresh/undo — are LATER sub-commits and are
deliberately not implemented here.
"""

from __future__ import annotations

import logging
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import ttk

import yaml

from flextool.scenario_comparison.plot_settings_seed import dump_plot_settings

logger = logging.getLogger(__name__)

# Entity classes, in the order they should appear as tabs.
_ENTITY_CLASSES = ("group", "unit", "connection", "node")
# Category subsections, in tab order.
_CATEGORY_SECTIONS = ("costs", "node_flows", "nodegroup_flows", "dispatch")

# Swatch geometry (pixels).  Two side-by-side boxes for entities, one for
# categories / scenarios.
_SWATCH_H = 14
_SWATCH_W = 14  # width of ONE box; a composite (pos|neg) is twice this.
_SWATCH_BORDER = (0x80, 0x80, 0x80)  # gray 1px frame so swatches read on any bg


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

        # Working state = parsed dict; snapshot the original TEXT for Cancel.
        self._original_text = self._read_text()
        self._data = self._parse(self._original_text)

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

        # ── Notebook with one tab per present section ─────────────
        self._notebook = ttk.Notebook(self)
        self._notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 4))
        self._build_tabs()

        # ── Buttons ───────────────────────────────────────────────
        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(4, 10))

        ttk.Button(btn_frame, text="Cancel", command=self._on_cancel).pack(
            side="right", padx=(5, 0),
        )
        ttk.Button(
            btn_frame, text="Save and exit", command=self._on_save_exit,
        ).pack(side="right", padx=(5, 0))
        ttk.Button(btn_frame, text="Apply", command=self._on_apply_clicked).pack(
            side="right",
        )

        self.bind("<Escape>", lambda _e: self._on_cancel())
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
    def _make_swatch(self, pos, neg=None) -> tk.PhotoImage:
        """Build a composite swatch ``PhotoImage`` for one row.

        ``neg is None`` -> a SINGLE box (categories / scenarios, or an
        entity with no distinct negative color).  Otherwise a TWO-box
        ``[pos][neg]`` composite.  The returned image is also appended to
        ``self._swatches`` so it survives garbage collection.
        """
        pos_rgb = _to_rgb255(pos)
        composite = neg is not None
        width = _SWATCH_W * 2 if composite else _SWATCH_W
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

        _fill(0, _SWATCH_W, pos_rgb)
        if composite:
            _fill(_SWATCH_W, width, _to_rgb255(neg))

        self._swatches.append(img)
        return img

    # ── Tab construction ──────────────────────────────────────────
    def _build_tabs(self) -> None:
        """Create one tab per section present in the working dict."""
        entities = self._data.get("entities")
        if isinstance(entities, dict):
            for cls in _ENTITY_CLASSES:
                section = entities.get(cls)
                if isinstance(section, dict) and section:
                    self._add_tab(
                        title=cls,
                        rows=list(section.items()),
                        composite=True,
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
                    )

        scenarios = self._data.get("scenarios")
        if isinstance(scenarios, dict) and scenarios:
            self._add_tab(
                title="scenarios",
                rows=list(scenarios.items()),
                composite=False,
            )

    def _add_tab(
        self,
        title: str,
        rows: list[tuple[str, object]],
        composite: bool,
    ) -> None:
        """Add a Notebook tab with a scrollable single-column Treeview."""
        frame = ttk.Frame(self._notebook)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        tree = ttk.Treeview(frame, show="tree", selectmode="browse")
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
                image = self._make_swatch(pos, neg)
            else:
                image = self._make_swatch(value, None)
            tree.insert("", "end", text=str(name), image=image)

        self._notebook.add(frame, text=title)

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
