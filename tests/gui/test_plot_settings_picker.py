"""Tests for the non-modal color/order picker (Stage 6.2).

Covers:
* ``PlotSettingsPicker`` builds a Notebook with one tab per present
  section, each tab's Treeview populated with the right names, and a
  composite swatch ``PhotoImage`` kept alive (not GC'd) per row.
* ``_write`` round-trips the working dict to byte-valid YAML; Apply writes
  + invokes ``on_apply``; Save-and-exit writes + invokes + closes; Cancel
  restores the on-open file text + invokes ``on_apply``.
* ``ResultViewer._on_change_colors`` seeds a project ``plot_settings.yaml``
  when absent, never overwrites an existing one, edits only the project
  copy (never the bundled package file), and opens the picker non-modally
  with ``_apply_color_settings`` as the ``on_apply`` callback.
* ``ResultViewer._apply_color_settings`` clears the cache and re-renders /
  rebuilds the color map (the reusable recolor body).
* The PNG settings ``PlotDialog`` opens the picker with NO ``on_apply``.

All Tk widgets are constructed under a withdrawn root; run headless via
``xvfb-run -a``.
"""

from __future__ import annotations

import tkinter as tk
import types
from pathlib import Path
from tkinter import ttk

import pytest
import yaml

try:
    import matplotlib
    matplotlib.use("Agg")
except Exception:
    pass


@pytest.fixture()
def tk_root():
    """Create a withdrawn Tk root; skip if no display is available."""
    try:
        root = tk.Tk()
        root.withdraw()
        yield root
        root.destroy()
    except tk.TclError:
        pytest.skip("No display available")


def _bundled_default() -> Path:
    from flextool._resources import package_data_path
    return package_data_path("schemas/default_plot_settings.yaml")


# ---------------------------------------------------------------------------
#  PlotSettingsPicker — tabs, trees, swatches
# ---------------------------------------------------------------------------


_SAMPLE = {
    "scenarios": {"S1": "#1f77b4", "S2": "#ff7f0e"},
    "categories": {
        "costs": {"co2": "#4d4d4d"},
        "dispatch": {"Charge": "lime"},  # named color
    },
    "entities": {
        "unit": {
            "coal": "#212121",
            "chp": {"color": "#E64A19", "neg_color": "#9c3010"},
        },
        "node": {"n1": "#4FC3F7"},
    },
}


def _make_picker(tk_root, tmp_path, data=None, on_apply=None):
    from flextool.gui.dialogs.plot_settings_picker import PlotSettingsPicker

    f = tmp_path / "plot_settings.yaml"
    f.write_text(
        yaml.safe_dump(data if data is not None else _SAMPLE, sort_keys=False),
        encoding="utf-8",
    )
    picker = PlotSettingsPicker(tk_root, f, on_apply=on_apply)
    return picker, f


def _tab_titles(picker) -> list[str]:
    nb = picker._notebook
    return [nb.tab(tid, "text") for tid in nb.tabs()]


def _tree_in_tab(picker, index):
    nb = picker._notebook
    frame = nb.nametowidget(nb.tabs()[index])
    for child in frame.winfo_children():
        if isinstance(child, ttk.Treeview):
            return child
    raise AssertionError("no Treeview in tab")


def _row_names(tree) -> list[str]:
    return [tree.item(iid, "text") for iid in tree.get_children("")]


class TestPickerBuild:
    def test_tabs_only_for_present_sections(self, tk_root, tmp_path):
        picker, _ = _make_picker(tk_root, tmp_path)
        # entities: unit, node (group/connection absent) → categories: costs,
        # dispatch (node_flows/nodegroup_flows absent) → scenarios.
        assert _tab_titles(picker) == [
            "unit", "node", "costs", "dispatch", "scenarios",
        ]

    def test_empty_sections_skipped(self, tk_root, tmp_path):
        data = {
            "scenarios": {},
            "categories": {"costs": {}},
            "entities": {"unit": {}, "node": {"n1": "#abcdef"}},
        }
        picker, _ = _make_picker(tk_root, tmp_path, data=data)
        # Only the non-empty node entities tab survives.
        assert _tab_titles(picker) == ["node"]

    def test_rows_populated_with_names(self, tk_root, tmp_path):
        picker, _ = _make_picker(tk_root, tmp_path)
        titles = _tab_titles(picker)
        unit = _tree_in_tab(picker, titles.index("unit"))
        assert _row_names(unit) == ["coal", "chp"]
        scen = _tree_in_tab(picker, titles.index("scenarios"))
        assert _row_names(scen) == ["S1", "S2"]
        costs = _tree_in_tab(picker, titles.index("costs"))
        assert _row_names(costs) == ["co2"]

    def test_swatches_created_and_referenced(self, tk_root, tmp_path):
        picker, _ = _make_picker(tk_root, tmp_path)
        # One swatch per row: 2 unit + 1 node + 1 costs + 1 dispatch + 2 scen.
        assert len(picker._swatches) == 7
        # Every swatch is a live PhotoImage (not GC'd).
        for img in picker._swatches:
            assert isinstance(img, tk.PhotoImage)
            assert img.width() > 0 and img.height() > 0

    def test_entity_swatches_always_reserve_neg_column(self, tk_root, tmp_path):
        """Every entity row is two boxes wide so names align: the negative
        column is reserved even for a bare entity (negative box transparent),
        and an entity with neg_color fills it.  Category / scenario rows have
        no negative concept → single box."""
        from flextool.gui.dialogs.plot_settings_picker import _SWATCH_W

        picker, _ = _make_picker(tk_root, tmp_path)
        titles = _tab_titles(picker)
        unit = _tree_in_tab(picker, titles.index("unit"))
        coal_iid, chp_iid = unit.get_children("")
        costs = _tree_in_tab(picker, titles.index("costs"))
        co2_iid = costs.get_children("")[0]

        def _img_width(tree, iid) -> int:
            name = tree.item(iid, "image")
            if isinstance(name, (list, tuple)):
                name = name[0]
            return int(tk_root.tk.call(name, "cget", "-width"))

        # Bare entity reserves the neg column (two boxes wide).
        assert _img_width(unit, coal_iid) == _SWATCH_W * 2
        # neg_color entity also two boxes wide (neg box drawn).
        assert _img_width(unit, chp_iid) == _SWATCH_W * 2
        # Category rows have no negative → single box.
        assert _img_width(costs, co2_iid) == _SWATCH_W

    def test_non_modal_no_grab(self, tk_root, tmp_path):
        """The picker must not grab input (usable alongside the viewer)."""
        picker, _ = _make_picker(tk_root, tmp_path)
        # The picker must not be the current grab.
        assert picker.grab_current() in (None, "")


# ---------------------------------------------------------------------------
#  PlotSettingsPicker — Apply / Save / Cancel + on_apply wiring
# ---------------------------------------------------------------------------


class TestPickerButtons:
    def test_apply_writes_roundtrip_and_calls_on_apply(self, tk_root, tmp_path):
        calls = []
        picker, f = _make_picker(
            tk_root, tmp_path, on_apply=lambda: calls.append(1),
        )
        picker._on_apply_clicked()
        # on_apply fired; window stayed open.
        assert calls == [1]
        assert picker.winfo_exists()
        # File round-trips equal to the working dict.
        assert yaml.safe_load(f.read_text(encoding="utf-8")) == picker._data
        assert picker._data == _SAMPLE

    def test_save_and_exit_writes_and_closes(self, tk_root, tmp_path):
        calls = []
        picker, f = _make_picker(
            tk_root, tmp_path, on_apply=lambda: calls.append(1),
        )
        picker._on_save_exit()
        assert calls == [1]
        assert not picker.winfo_exists()
        assert yaml.safe_load(f.read_text(encoding="utf-8")) == _SAMPLE

    def test_cancel_restores_original_and_calls_on_apply(self, tk_root, tmp_path):
        calls = []
        picker, f = _make_picker(
            tk_root, tmp_path, on_apply=lambda: calls.append(1),
        )
        original = f.read_text(encoding="utf-8")
        # Simulate a prior Apply that changed the file on disk.
        f.write_text("scenarios:\n  X: '#000000'\n", encoding="utf-8")
        picker._on_cancel()
        # Original on-open text restored byte-for-byte; on_apply (revert) fired.
        assert f.read_text(encoding="utf-8") == original
        assert calls == [1]
        assert not picker.winfo_exists()

    def test_no_on_apply_is_fine(self, tk_root, tmp_path):
        """Picker opened with no callback (PNG dialog) just writes."""
        picker, f = _make_picker(tk_root, tmp_path, on_apply=None)
        picker._on_apply_clicked()  # must not raise
        assert yaml.safe_load(f.read_text(encoding="utf-8")) == _SAMPLE


# ---------------------------------------------------------------------------
#  PlotSettingsPicker — reordering (drag + keyboard) → persisted order
# ---------------------------------------------------------------------------


def _section(data: dict, path: tuple[str, ...]) -> dict:
    cur = data
    for key in path:
        cur = cur[key]
    return cur


class TestPickerReorder:
    def test_keyboard_alt_down_moves_row_and_syncs_dict(
        self, tk_root, tmp_path,
    ):
        picker, _ = _make_picker(tk_root, tmp_path)
        titles = _tab_titles(picker)
        unit = _tree_in_tab(picker, titles.index("unit"))
        assert _row_names(unit) == ["coal", "chp"]

        # Focus the top row and Alt-Down it.
        first = unit.get_children("")[0]
        unit.focus(first)
        unit.selection_set(first)
        picker._key_move(unit, +1)

        # Tree order flipped and the moved row stays selected/focused.
        assert _row_names(unit) == ["chp", "coal"]
        assert unit.focus() == first
        assert unit.selection() == (first,)

        # Working dict section reordered; values intact (chp keeps mapping).
        sect = _section(picker._data, ("entities", "unit"))
        assert list(sect.keys()) == ["chp", "coal"]
        assert sect["chp"] == {"color": "#E64A19", "neg_color": "#9c3010"}
        assert sect["coal"] == "#212121"

    def test_keyboard_alt_up_at_top_is_noop(self, tk_root, tmp_path):
        picker, _ = _make_picker(tk_root, tmp_path)
        titles = _tab_titles(picker)
        scen = _tree_in_tab(picker, titles.index("scenarios"))
        top = scen.get_children("")[0]
        scen.focus(top)
        picker._key_move(scen, -1)
        assert _row_names(scen) == ["S1", "S2"]
        assert list(_section(picker._data, ("scenarios",)).keys()) == ["S1", "S2"]

    def test_alt_arrow_bindings_registered(self, tk_root, tmp_path):
        """Each tree has <Alt-Up>/<Alt-Down> bound (event-level wiring).

        A real keystroke cannot be routed headlessly without a window
        manager (``focus_set`` cannot acquire input focus under bare
        Xvfb), so we assert the bindings exist on every tab's tree and
        that they dispatch our handlers.  The move+sync behaviour itself
        is exercised through the handlers below.
        """
        picker, _ = _make_picker(tk_root, tmp_path)
        for tree in picker._tree_section:
            assert tree.bind("<Alt-Up>")
            assert tree.bind("<Alt-Down>")
            assert tree.bind("<ButtonPress-1>")
            assert tree.bind("<B1-Motion>")
            assert tree.bind("<ButtonRelease-1>")

    def test_alt_down_event_invokes_handler(self, tk_root, tmp_path):
        """Synthesise the <Alt-Down> event object and feed it through the
        bound handler (the same callable Tk would invoke), proving the
        event path — not just an ad-hoc method call — reorders + persists.
        """
        picker, _ = _make_picker(tk_root, tmp_path)
        titles = _tab_titles(picker)
        scen = _tree_in_tab(picker, titles.index("scenarios"))
        first = scen.get_children("")[0]
        scen.focus(first)
        scen.selection_set(first)
        evt = types.SimpleNamespace(widget=scen)
        result = picker._on_key_move_down(evt)
        assert result == "break"  # default Alt-arrow handling suppressed
        assert _row_names(scen) == ["S2", "S1"]
        assert list(_section(picker._data, ("scenarios",)).keys()) == ["S2", "S1"]

    def test_drag_handlers_reorder_and_persist(
        self, tk_root, tmp_path, monkeypatch,
    ):
        """Driving the bound drag handlers reorders + persists order.

        Headless Treeview rows have no real geometry, so we map cursor-y
        to a row via a stubbed ``identify_row`` (the only Tk geometry call
        the handlers make); everything else is the real handler logic.
        """
        picker, _ = _make_picker(tk_root, tmp_path)
        titles = _tab_titles(picker)
        unit = _tree_in_tab(picker, titles.index("unit"))
        coal, chp = unit.get_children("")

        # y == 0 → coal (top), y == 1 → chp (bottom).
        monkeypatch.setattr(
            unit, "identify_row",
            lambda y: {0: coal, 1: chp}.get(y, ""),
        )

        def _ev(y):
            return types.SimpleNamespace(widget=unit, y=y)

        # Press on coal, drag down onto chp, release.
        picker._on_drag_start(_ev(0))
        assert picker._drag_item[unit] == coal
        picker._on_drag_motion(_ev(1))
        picker._on_drag_end(_ev(1))

        assert _row_names(unit) == ["chp", "coal"]
        assert picker._drag_item[unit] is None
        sect = _section(picker._data, ("entities", "unit"))
        assert list(sect.keys()) == ["chp", "coal"]
        assert sect["chp"] == {"color": "#E64A19", "neg_color": "#9c3010"}

    def test_drag_on_empty_space_is_noop(self, tk_root, tmp_path, monkeypatch):
        picker, _ = _make_picker(tk_root, tmp_path)
        titles = _tab_titles(picker)
        unit = _tree_in_tab(picker, titles.index("unit"))
        # identify_row off the rows returns "" → drag candidate None.
        monkeypatch.setattr(unit, "identify_row", lambda y: "")
        empty = types.SimpleNamespace(widget=unit, y=10_000)
        picker._on_drag_start(empty)
        assert picker._drag_item[unit] is None
        picker._on_drag_motion(empty)
        picker._on_drag_end(empty)
        assert _row_names(unit) == ["coal", "chp"]

    def test_reordered_order_is_written_to_file(self, tk_root, tmp_path):
        """After a reorder, Apply writes the file with the new key order
        and values intact (sort_keys=False preserves it)."""
        picker, f = _make_picker(tk_root, tmp_path)
        titles = _tab_titles(picker)
        unit = _tree_in_tab(picker, titles.index("unit"))
        first = unit.get_children("")[0]
        unit.focus(first)
        unit.selection_set(first)
        picker._key_move(unit, +1)  # coal → below chp

        picker._on_apply_clicked()

        loaded = yaml.safe_load(f.read_text(encoding="utf-8"))
        # File key order matches the tree's new top-to-bottom order.
        assert list(loaded["entities"]["unit"].keys()) == ["chp", "coal"]
        # Values intact through the round-trip.
        assert loaded["entities"]["unit"]["chp"] == {
            "color": "#E64A19", "neg_color": "#9c3010",
        }
        assert loaded["entities"]["unit"]["coal"] == "#212121"
        # Untouched sections unchanged.
        assert loaded["scenarios"] == {"S1": "#1f77b4", "S2": "#ff7f0e"}

    def test_sync_preserves_other_sections(self, tk_root, tmp_path):
        """Reordering one tab must not disturb other sections of the dict."""
        picker, _ = _make_picker(tk_root, tmp_path)
        titles = _tab_titles(picker)
        scen = _tree_in_tab(picker, titles.index("scenarios"))
        first = scen.get_children("")[0]
        scen.focus(first)
        picker._key_move(scen, +1)
        # entities/categories untouched and identical to the input.
        assert picker._data["entities"] == _SAMPLE["entities"]
        assert picker._data["categories"] == _SAMPLE["categories"]
        assert list(picker._data["scenarios"].keys()) == ["S2", "S1"]


# ---------------------------------------------------------------------------
#  ColorPickerDialog — pos/neg + lock semantics (Stage 6.4)
# ---------------------------------------------------------------------------


def _mock_askcolor(monkeypatch, hex_value):
    """Make ``tkinter.colorchooser.askcolor`` return a fixed hex (no UI)."""
    import tkinter.colorchooser as cc

    def _fake(*_a, **_k):
        if hex_value is None:
            return (None, None)  # Cancel.
        return ((0, 0, 0), hex_value)

    monkeypatch.setattr(cc, "askcolor", _fake)


class TestColorPickerDialog:
    def test_linked_pick_pos_mirrors_neg_returns_none_neg(
        self, tk_root, tmp_path, monkeypatch,
    ):
        from flextool.gui.dialogs.plot_settings_picker import ColorPickerDialog

        dlg = ColorPickerDialog(tk_root, "coal", "#212121", "#212121", True)
        _mock_askcolor(monkeypatch, "#00FF00")
        dlg._pick_pos()
        # While linked the negative mirrors the positive.
        assert dlg._pos == "#00ff00"
        assert dlg._neg == "#00ff00"
        assert dlg._linked.get() is True
        dlg._on_ok()
        # Linked → neg returned as None (bare entry).
        assert dlg.result == ("#00ff00", None)

    def test_pick_neg_unlinks_and_separates(
        self, tk_root, tmp_path, monkeypatch,
    ):
        from flextool.gui.dialogs.plot_settings_picker import ColorPickerDialog

        dlg = ColorPickerDialog(tk_root, "coal", "#212121", "#212121", True)
        assert dlg._linked.get() is True
        _mock_askcolor(monkeypatch, "#aabbcc")
        dlg._pick_neg()
        # Picking a negative deliberately unlinks and separates the colors.
        assert dlg._linked.get() is False
        assert dlg._pos == "#212121"
        assert dlg._neg == "#aabbcc"
        dlg._on_ok()
        assert dlg.result == ("#212121", "#aabbcc")

    def test_entry_with_neg_opens_unlinked(self, tk_root, tmp_path):
        from flextool.gui.dialogs.plot_settings_picker import ColorPickerDialog

        dlg = ColorPickerDialog(
            tk_root, "chp", "#E64A19", "#9c3010", False,
        )
        assert dlg._linked.get() is False
        # The negative "Pick…" button is enabled when unlinked.
        assert str(dlg._neg_button.cget("state")) == "normal"
        dlg._on_ok()
        assert dlg.result == ("#e64a19", "#9c3010")

    def test_relink_collapses_to_pos(self, tk_root, tmp_path):
        from flextool.gui.dialogs.plot_settings_picker import ColorPickerDialog

        dlg = ColorPickerDialog(
            tk_root, "chp", "#E64A19", "#9c3010", False,
        )
        # Re-check the link box → neg := pos, neg button disabled.
        dlg._linked.set(True)
        dlg._on_link_toggle()
        assert dlg._neg == "#e64a19"
        assert str(dlg._neg_button.cget("state")) == "disabled"
        dlg._on_ok()
        assert dlg.result == ("#e64a19", None)

    def test_cancel_returns_none(self, tk_root, tmp_path):
        from flextool.gui.dialogs.plot_settings_picker import ColorPickerDialog

        dlg = ColorPickerDialog(tk_root, "coal", "#212121", "#212121", True)
        dlg._on_cancel()
        assert dlg.result is None

    def test_neg_button_disabled_when_linked(self, tk_root, tmp_path):
        from flextool.gui.dialogs.plot_settings_picker import ColorPickerDialog

        dlg = ColorPickerDialog(tk_root, "coal", "#212121", "#212121", True)
        assert str(dlg._neg_button.cget("state")) == "disabled"


# ---------------------------------------------------------------------------
#  Picker double-click → edit → write-back + swatch rebuild (Stage 6.4)
# ---------------------------------------------------------------------------


def _patch_dialog(monkeypatch, result):
    """Replace ``ColorPickerDialog`` with a non-blocking fake.

    The real dialog blocks on ``wait_window``; the fake is a tiny Toplevel
    that destroys itself immediately and exposes a preset ``result`` so the
    picker's write-back path can be driven headlessly.
    """
    import flextool.gui.dialogs.plot_settings_picker as mod

    class _FakeDialog(tk.Toplevel):
        def __init__(self, parent, name, pos_hex, neg_hex, linked):
            super().__init__(parent)
            self.withdraw()
            self.result = result
            self.opened = (name, pos_hex, neg_hex, linked)
            captured["dialog"] = self
            self.after(0, self.destroy)

    captured: dict = {}
    monkeypatch.setattr(mod, "ColorPickerDialog", _FakeDialog)
    return captured


class TestPickerDoubleClickEdit:
    def test_entity_linked_pick_writes_bare_color(
        self, tk_root, tmp_path, monkeypatch,
    ):
        picker, _ = _make_picker(tk_root, tmp_path)
        titles = _tab_titles(picker)
        unit = _tree_in_tab(picker, titles.index("unit"))
        coal = unit.get_children("")[0]

        captured = _patch_dialog(monkeypatch, ("#00ff00", None))
        before = list(picker._swatches)
        picker._edit_row_color(unit, coal)

        # Bare "coal" opened LINKED; write-back is a bare color string.
        assert captured["dialog"].opened[3] is True
        sect = _section(picker._data, ("entities", "unit"))
        assert sect["coal"] == "#00ff00"
        # Order + other entries untouched.
        assert list(sect.keys()) == ["coal", "chp"]
        assert sect["chp"] == {"color": "#E64A19", "neg_color": "#9c3010"}
        # A new swatch image was created and attached to the row.
        assert len(picker._swatches) == len(before) + 1
        assert (unit, coal) in picker._row_swatches
        img = picker._row_swatches[(unit, coal)]
        assert isinstance(img, tk.PhotoImage)
        name = unit.item(coal, "image")
        if isinstance(name, (list, tuple)):
            name = name[0]
        assert str(name) == str(img)

    def test_entity_unlink_pick_writes_color_neg(
        self, tk_root, tmp_path, monkeypatch,
    ):
        from flextool.gui.dialogs.plot_settings_picker import _SWATCH_W

        picker, _ = _make_picker(tk_root, tmp_path)
        titles = _tab_titles(picker)
        unit = _tree_in_tab(picker, titles.index("unit"))
        coal = unit.get_children("")[0]

        _patch_dialog(monkeypatch, ("#111111", "#222222"))
        picker._edit_row_color(unit, coal)

        sect = _section(picker._data, ("entities", "unit"))
        assert sect["coal"] == {"color": "#111111", "neg_color": "#222222"}
        # Composite (two-box) swatch now attached.
        img = picker._row_swatches[(unit, coal)]
        assert img.width() == _SWATCH_W * 2

    def test_entity_with_neg_opens_unlinked(
        self, tk_root, tmp_path, monkeypatch,
    ):
        picker, _ = _make_picker(tk_root, tmp_path)
        titles = _tab_titles(picker)
        unit = _tree_in_tab(picker, titles.index("unit"))
        chp = unit.get_children("")[1]

        captured = _patch_dialog(monkeypatch, ("#E64A19", "#9c3010"))
        picker._edit_row_color(unit, chp)
        # {color, neg_color} entry opened UNLINKED with both hexes seeded.
        name, pos_hex, neg_hex, linked = captured["dialog"].opened
        assert name == "chp"
        assert linked is False
        assert pos_hex == "#e64a19"
        assert neg_hex == "#9c3010"

    def test_relink_collapses_to_bare(self, tk_root, tmp_path, monkeypatch):
        from flextool.gui.dialogs.plot_settings_picker import _SWATCH_W

        picker, _ = _make_picker(tk_root, tmp_path)
        titles = _tab_titles(picker)
        unit = _tree_in_tab(picker, titles.index("unit"))
        chp = unit.get_children("")[1]

        # Dialog returns linked result (neg None) for a previously-split row.
        _patch_dialog(monkeypatch, ("#abcdef", None))
        picker._edit_row_color(unit, chp)

        sect = _section(picker._data, ("entities", "unit"))
        assert sect["chp"] == "#abcdef"  # collapsed to bare
        img = picker._row_swatches[(unit, chp)]
        # Entity rows always reserve the (now transparent) negative column,
        # so the image stays two boxes wide even after collapsing to bare.
        assert img.width() == _SWATCH_W * 2

    def test_category_row_edits_bare_color(
        self, tk_root, tmp_path, monkeypatch,
    ):
        picker, _ = _make_picker(tk_root, tmp_path)
        titles = _tab_titles(picker)
        costs = _tree_in_tab(picker, titles.index("costs"))
        co2 = costs.get_children("")[0]

        _mock_askcolor(monkeypatch, "#FEDCBA")
        before = list(picker._swatches)
        picker._edit_row_color(costs, co2)

        sect = _section(picker._data, ("categories", "costs"))
        assert sect["co2"] == "#fedcba"
        assert len(picker._swatches) == len(before) + 1
        assert (costs, co2) in picker._row_swatches

    def test_scenario_row_edits_bare_color(
        self, tk_root, tmp_path, monkeypatch,
    ):
        picker, _ = _make_picker(tk_root, tmp_path)
        titles = _tab_titles(picker)
        scen = _tree_in_tab(picker, titles.index("scenarios"))
        s1 = scen.get_children("")[0]

        _mock_askcolor(monkeypatch, "#0a0b0c")
        picker._edit_row_color(scen, s1)
        sect = _section(picker._data, ("scenarios",))
        assert sect["S1"] == "#0a0b0c"
        assert list(sect.keys()) == ["S1", "S2"]  # order intact

    def test_cancel_makes_no_change(self, tk_root, tmp_path, monkeypatch):
        picker, _ = _make_picker(tk_root, tmp_path)
        titles = _tab_titles(picker)
        unit = _tree_in_tab(picker, titles.index("unit"))
        coal = unit.get_children("")[0]

        _patch_dialog(monkeypatch, None)  # Cancel.
        before = list(picker._swatches)
        picker._edit_row_color(unit, coal)
        sect = _section(picker._data, ("entities", "unit"))
        assert sect["coal"] == "#212121"  # unchanged
        # No new swatch, no row-swatch override.
        assert len(picker._swatches) == len(before)
        assert (unit, coal) not in picker._row_swatches

    def test_category_cancel_makes_no_change(
        self, tk_root, tmp_path, monkeypatch,
    ):
        picker, _ = _make_picker(tk_root, tmp_path)
        titles = _tab_titles(picker)
        costs = _tree_in_tab(picker, titles.index("costs"))
        co2 = costs.get_children("")[0]

        _mock_askcolor(monkeypatch, None)  # Cancel.
        picker._edit_row_color(costs, co2)
        sect = _section(picker._data, ("categories", "costs"))
        assert sect["co2"] == "#4d4d4d"  # unchanged

    def test_double_click_empty_space_is_noop(
        self, tk_root, tmp_path, monkeypatch,
    ):
        picker, _ = _make_picker(tk_root, tmp_path)
        titles = _tab_titles(picker)
        unit = _tree_in_tab(picker, titles.index("unit"))
        monkeypatch.setattr(unit, "identify_row", lambda y: "")

        called = []
        monkeypatch.setattr(
            picker, "_edit_row_color",
            lambda *a, **k: called.append(a),
        )
        evt = types.SimpleNamespace(widget=unit, y=10_000)
        result = picker._on_row_double_click(evt)
        assert result == "break"
        assert called == []  # no edit on empty space

    def test_double_click_on_row_opens_editor_and_clears_drag(
        self, tk_root, tmp_path, monkeypatch,
    ):
        picker, _ = _make_picker(tk_root, tmp_path)
        titles = _tab_titles(picker)
        unit = _tree_in_tab(picker, titles.index("unit"))
        coal = unit.get_children("")[0]
        monkeypatch.setattr(unit, "identify_row", lambda y: coal)
        # Prime a stale drag candidate as a prior ButtonPress would.
        picker._drag_item[unit] = coal

        called = []
        monkeypatch.setattr(
            picker, "_edit_row_color",
            lambda tree, item: called.append((tree, item)),
        )
        evt = types.SimpleNamespace(widget=unit, y=0)
        result = picker._on_row_double_click(evt)
        # Resolves the row, clears the drag candidate (no reorder), edits.
        assert result == "break"
        assert called == [(unit, coal)]
        assert picker._drag_item[unit] is None

    def test_double_click_binding_registered(self, tk_root, tmp_path):
        picker, _ = _make_picker(tk_root, tmp_path)
        for tree in picker._tree_section:
            assert tree.bind("<Double-Button-1>")


# ---------------------------------------------------------------------------
#  PlotDialog — shared "Colors, order..." button opens the picker
# ---------------------------------------------------------------------------


def _iter_buttons(widget):
    for child in widget.winfo_children():
        if isinstance(child, ttk.Button):
            yield child
        yield from _iter_buttons(child)


class TestPlotDialogColorsButton:
    def test_button_seeds_and_opens_picker_no_preview(
        self, tk_root, tmp_path, monkeypatch,
    ):
        """The dialog-level "Colors, order..." button seeds the project
        ``plot_settings.yaml`` and opens the picker on that project copy
        with NO ``on_apply`` (the batch dialog has no live preview)."""
        from flextool.gui.dialogs.plot_dialog import PlotDialog
        from flextool.gui.data_models import ProjectSettings

        project = tmp_path / "proj"
        project.mkdir()
        assert not (project / "plot_settings.yaml").exists()

        opened = {}

        class _FakePicker:
            def __init__(self, parent, path, on_apply=None):
                opened["parent"] = parent
                opened["path"] = Path(path)
                opened["on_apply"] = on_apply

        monkeypatch.setattr(
            "flextool.gui.dialogs.plot_settings_picker.PlotSettingsPicker",
            _FakePicker,
        )

        captured = {}

        def drive():
            dlg = captured["dialog"]
            buttons = [
                b for b in _iter_buttons(dlg)
                if str(b.cget("text")) == "Colors, order..."
            ]
            captured["button_count"] = len(buttons)
            if buttons:
                buttons[0].invoke()
            dlg._on_ok()

        class _Probe(PlotDialog):
            def __init__(self, parent, project_path, settings):
                captured["dialog"] = self
                parent.after(0, drive)
                super().__init__(parent, project_path, settings)

        _Probe(tk_root, project, ProjectSettings())

        assert captured["button_count"] == 1
        seeded = project / "plot_settings.yaml"
        assert seeded.is_file()
        assert seeded.read_bytes() == _bundled_default().read_bytes()
        assert opened["path"] == seeded
        # PNG batch dialog → no live preview.
        assert opened["on_apply"] is None

    def test_dispatch_config_editor_is_gone(self):
        """The old ``DispatchConfigEditor`` and its handler are removed."""
        from flextool.gui.dialogs import plot_dialog

        assert not hasattr(plot_dialog, "DispatchConfigEditor")
        assert not hasattr(plot_dialog._PlotSection, "_on_edit_dispatch_config")


# ---------------------------------------------------------------------------
#  Refresh from DB — re-fetch entities, ADD new + PRUNE stale (Stage 6.5)
# ---------------------------------------------------------------------------


def _mock_fetch(monkeypatch, per_class):
    """Patch ``fetch_entities_by_class`` to return a fixed per-class mapping
    (no real DB), and patch ``_discover_input_dbs`` to report one DB so the
    refresh runs the union/add/prune path headlessly."""
    import flextool.scenario_comparison.input_entity_colors as iec

    monkeypatch.setattr(
        iec, "fetch_entities_by_class", lambda _url: dict(per_class),
    )


class TestPickerRefresh:
    def test_refresh_adds_new_and_prunes_stale(
        self, tk_root, tmp_path, monkeypatch,
    ):
        from flextool.gui.dialogs.plot_settings_picker import PlotSettingsPicker

        picker, _ = _make_picker(tk_root, tmp_path)
        # Discovery returns one DB; the DB has unit {coal, gas} and node {n1}.
        # → unit: "chp" is stale (pruned), "gas" is new (added); node: n1 stays.
        monkeypatch.setattr(
            PlotSettingsPicker, "_discover_input_dbs",
            lambda self: ["sqlite:///fake.sqlite"],
        )
        _mock_fetch(monkeypatch, {"unit": ["coal", "gas"], "node": ["n1"]})

        picker._on_refresh()

        unit = _section(picker._data, ("entities", "unit"))
        # coal kept (with its edited value), chp pruned, gas added.
        assert "coal" in unit
        assert "chp" not in unit
        assert "gas" in unit
        assert unit["coal"] == "#212121"  # existing value preserved
        # New name got a palette hex color appended AFTER existing entries.
        assert list(unit.keys()) == ["coal", "gas"]
        assert isinstance(unit["gas"], str) and unit["gas"].startswith("#")
        # node unchanged.
        assert _section(picker._data, ("entities", "node")) == {"n1": "#4FC3F7"}
        # categories / scenarios untouched.
        assert picker._data["categories"] == _SAMPLE["categories"]
        assert picker._data["scenarios"] == _SAMPLE["scenarios"]

    def test_refresh_rebuilds_trees(self, tk_root, tmp_path, monkeypatch):
        from flextool.gui.dialogs.plot_settings_picker import PlotSettingsPicker

        picker, _ = _make_picker(tk_root, tmp_path)
        monkeypatch.setattr(
            PlotSettingsPicker, "_discover_input_dbs",
            lambda self: ["sqlite:///fake.sqlite"],
        )
        _mock_fetch(monkeypatch, {"unit": ["coal", "gas"], "node": ["n1"]})

        picker._on_refresh()

        titles = _tab_titles(picker)
        unit = _tree_in_tab(picker, titles.index("unit"))
        # Tree rows match the rebuilt dict: chp gone, gas present.
        assert _row_names(unit) == ["coal", "gas"]

    def test_refresh_records_one_undo_step(self, tk_root, tmp_path, monkeypatch):
        from flextool.gui.dialogs.plot_settings_picker import PlotSettingsPicker

        picker, _ = _make_picker(tk_root, tmp_path)
        monkeypatch.setattr(
            PlotSettingsPicker, "_discover_input_dbs",
            lambda self: ["sqlite:///fake.sqlite"],
        )
        _mock_fetch(monkeypatch, {"unit": ["coal", "gas"], "node": ["n1"]})

        before = {k: dict(v) if isinstance(v, dict) else v
                  for k, v in picker._data["entities"].items()}
        picker._on_refresh()
        assert len(picker._undo_stack) == 1
        # Undo restores the pre-refresh entities.
        picker._on_undo()
        assert picker._data["entities"]["unit"] == before["unit"]

    def test_refresh_no_db_shows_info_and_no_change(
        self, tk_root, tmp_path, monkeypatch,
    ):
        from flextool.gui.dialogs import plot_settings_picker as mod
        from flextool.gui.dialogs.plot_settings_picker import PlotSettingsPicker

        picker, _ = _make_picker(tk_root, tmp_path)
        monkeypatch.setattr(
            PlotSettingsPicker, "_discover_input_dbs", lambda self: [],
        )
        shown = []
        monkeypatch.setattr(
            mod.messagebox, "showinfo",
            lambda *a, **k: shown.append((a, k)),
        )

        before = picker._data
        picker._on_refresh()
        assert len(shown) == 1  # info box shown
        assert picker._data == _SAMPLE  # unchanged
        assert picker._data is before
        assert picker._undo_stack == []  # no edit recorded

    def test_refresh_idempotent_no_change_no_undo(
        self, tk_root, tmp_path, monkeypatch,
    ):
        """A refresh whose DB exactly matches the current entities records no
        undo step (nothing added or pruned)."""
        from flextool.gui.dialogs.plot_settings_picker import PlotSettingsPicker

        picker, _ = _make_picker(tk_root, tmp_path)
        monkeypatch.setattr(
            PlotSettingsPicker, "_discover_input_dbs",
            lambda self: ["sqlite:///fake.sqlite"],
        )
        # Exactly the current entities (unit: coal, chp; node: n1).
        _mock_fetch(
            monkeypatch, {"unit": ["coal", "chp"], "node": ["n1"]},
        )
        picker._on_refresh()
        assert picker._undo_stack == []
        assert list(_section(picker._data, ("entities", "unit")).keys()) == [
            "coal", "chp",
        ]

    def test_discover_input_dbs_scans_both_dirs(self, tk_root, tmp_path):
        """Discovery scans <project>/input_sources and <project>/intermediate
        for *.sqlite (project root = settings file's parent)."""
        project = tmp_path
        (project / "input_sources").mkdir()
        (project / "intermediate").mkdir()
        (project / "input_sources" / "a.sqlite").write_bytes(b"")
        (project / "intermediate" / "b.sqlite").write_bytes(b"")
        # A non-sqlite file is ignored.
        (project / "input_sources" / "notes.txt").write_text("x")

        picker, _ = _make_picker(tk_root, tmp_path)
        urls = picker._discover_input_dbs()
        assert urls == [
            f"sqlite:///{project / 'input_sources' / 'a.sqlite'}",
            f"sqlite:///{project / 'intermediate' / 'b.sqlite'}",
        ]


# ---------------------------------------------------------------------------
#  Undo / Redo over the working dict (Stage 6.5)
# ---------------------------------------------------------------------------


class TestPickerUndoRedo:
    def test_color_edit_undo_restores_and_tree_shows_old(
        self, tk_root, tmp_path, monkeypatch,
    ):
        picker, _ = _make_picker(tk_root, tmp_path)
        titles = _tab_titles(picker)
        unit = _tree_in_tab(picker, titles.index("unit"))
        coal = unit.get_children("")[0]

        _patch_dialog(monkeypatch, ("#00ff00", None))
        picker._edit_row_color(unit, coal)
        assert _section(picker._data, ("entities", "unit"))["coal"] == "#00ff00"
        assert len(picker._undo_stack) == 1

        picker._on_undo()
        # Working dict restored to the pre-edit value.
        assert _section(picker._data, ("entities", "unit"))["coal"] == "#212121"
        # Trees rebuilt to match: re-resolve the (new) tree + row.
        titles = _tab_titles(picker)
        unit2 = _tree_in_tab(picker, titles.index("unit"))
        assert _row_names(unit2) == ["coal", "chp"]
        assert picker._redo_stack and not picker._undo_stack

    def test_redo_reapplies(self, tk_root, tmp_path, monkeypatch):
        picker, _ = _make_picker(tk_root, tmp_path)
        titles = _tab_titles(picker)
        unit = _tree_in_tab(picker, titles.index("unit"))
        coal = unit.get_children("")[0]

        _patch_dialog(monkeypatch, ("#00ff00", None))
        picker._edit_row_color(unit, coal)
        picker._on_undo()
        assert _section(picker._data, ("entities", "unit"))["coal"] == "#212121"
        picker._on_redo()
        assert _section(picker._data, ("entities", "unit"))["coal"] == "#00ff00"

    def test_reorder_undo_restores_order(self, tk_root, tmp_path):
        picker, _ = _make_picker(tk_root, tmp_path)
        titles = _tab_titles(picker)
        unit = _tree_in_tab(picker, titles.index("unit"))
        first = unit.get_children("")[0]
        unit.focus(first)
        unit.selection_set(first)
        picker._key_move(unit, +1)
        assert list(_section(picker._data, ("entities", "unit")).keys()) == [
            "chp", "coal",
        ]
        picker._on_undo()
        assert list(_section(picker._data, ("entities", "unit")).keys()) == [
            "coal", "chp",
        ]
        titles = _tab_titles(picker)
        unit2 = _tree_in_tab(picker, titles.index("unit"))
        assert _row_names(unit2) == ["coal", "chp"]

    def test_multi_level_undo(self, tk_root, tmp_path, monkeypatch):
        picker, _ = _make_picker(tk_root, tmp_path)
        titles = _tab_titles(picker)
        unit = _tree_in_tab(picker, titles.index("unit"))
        coal = unit.get_children("")[0]

        # Edit 1: recolor coal.
        _patch_dialog(monkeypatch, ("#111111", None))
        picker._edit_row_color(unit, coal)
        # Edit 2: recolor coal again.
        _patch_dialog(monkeypatch, ("#222222", None))
        coal = _tree_in_tab(
            picker, _tab_titles(picker).index("unit"),
        ).get_children("")[0]
        picker._edit_row_color(
            _tree_in_tab(picker, _tab_titles(picker).index("unit")), coal,
        )
        assert _section(picker._data, ("entities", "unit"))["coal"] == "#222222"
        assert len(picker._undo_stack) == 2

        picker._on_undo()
        assert _section(picker._data, ("entities", "unit"))["coal"] == "#111111"
        picker._on_undo()
        assert _section(picker._data, ("entities", "unit"))["coal"] == "#212121"
        assert not picker._undo_stack

    def test_new_edit_clears_redo(self, tk_root, tmp_path, monkeypatch):
        picker, _ = _make_picker(tk_root, tmp_path)
        unit = _tree_in_tab(picker, _tab_titles(picker).index("unit"))
        coal = unit.get_children("")[0]

        _patch_dialog(monkeypatch, ("#111111", None))
        picker._edit_row_color(unit, coal)
        picker._on_undo()
        assert picker._redo_stack  # redo available

        # A fresh edit clears the redo stack.
        unit = _tree_in_tab(picker, _tab_titles(picker).index("unit"))
        coal = unit.get_children("")[0]
        _patch_dialog(monkeypatch, ("#333333", None))
        picker._edit_row_color(unit, coal)
        assert picker._redo_stack == []

    def test_buttons_disabled_at_stack_ends(self, tk_root, tmp_path, monkeypatch):
        picker, _ = _make_picker(tk_root, tmp_path)
        # Both stacks empty at open → both buttons disabled.
        assert str(picker._undo_button.cget("state")) == "disabled"
        assert str(picker._redo_button.cget("state")) == "disabled"

        unit = _tree_in_tab(picker, _tab_titles(picker).index("unit"))
        coal = unit.get_children("")[0]
        _patch_dialog(monkeypatch, ("#111111", None))
        picker._edit_row_color(unit, coal)
        # After an edit: undo enabled, redo still disabled.
        assert str(picker._undo_button.cget("state")) == "normal"
        assert str(picker._redo_button.cget("state")) == "disabled"

        picker._on_undo()
        # After undo: undo disabled (stack empty), redo enabled.
        assert str(picker._undo_button.cget("state")) == "disabled"
        assert str(picker._redo_button.cget("state")) == "normal"

    def test_undo_redo_key_bindings_registered(self, tk_root, tmp_path):
        picker, _ = _make_picker(tk_root, tmp_path)
        assert picker.bind("<Control-z>")
        assert picker.bind("<Control-y>")
        assert picker.bind("<Control-Shift-Z>")

    def test_undo_on_empty_stack_is_noop(self, tk_root, tmp_path):
        picker, _ = _make_picker(tk_root, tmp_path)
        before = picker._data
        picker._on_undo()  # no edits yet → no-op
        assert picker._data is before
        picker._on_redo()  # nothing to redo → no-op
        assert picker._data is before


# ---------------------------------------------------------------------------
#  _on_change_colors — seeding + opens picker non-modally with on_apply
# ---------------------------------------------------------------------------


def _make_stub_viewer(project_path: Path, live_plan=None):
    """A minimal stand-in carrying just what ``_on_change_colors`` and
    ``_apply_color_settings`` touch.  Binds the real unbound methods."""
    from flextool.gui.result_viewer import ResultViewer

    stub = types.SimpleNamespace()
    stub._project_path = project_path
    stub._live_plan = live_plan
    stub.calls = []
    stub._clear_figure_cache = lambda: stub.calls.append("clear_figure_cache")
    stub._clear_prefetched_figures = lambda: stub.calls.append(
        "clear_prefetched_figures"
    )
    stub._trigger_replot = lambda: stub.calls.append("trigger_replot")
    stub._on_change_colors = types.MethodType(
        ResultViewer._on_change_colors, stub,
    )
    stub._apply_color_settings = types.MethodType(
        ResultViewer._apply_color_settings, stub,
    )
    return stub


class TestOnChangeColorsSeeding:
    def test_seeds_project_file_and_opens_picker_with_callback(
        self, tk_root, tmp_path, monkeypatch,
    ):
        import flextool.gui.result_viewer as rv

        project = tmp_path / "proj"
        project.mkdir()
        assert not (project / "plot_settings.yaml").exists()

        opened = {}

        class _FakePicker:
            def __init__(self, parent, path, on_apply=None):
                opened["path"] = Path(path)
                opened["on_apply"] = on_apply

        monkeypatch.setattr(
            "flextool.gui.dialogs.plot_settings_picker.PlotSettingsPicker",
            _FakePicker,
        )

        stub = _make_stub_viewer(project)
        stub._on_change_colors()

        seeded = project / "plot_settings.yaml"
        assert seeded.is_file(), "project plot_settings.yaml must be seeded"
        assert seeded.read_bytes() == _bundled_default().read_bytes()
        assert opened["path"] == seeded
        assert opened["path"] != _bundled_default()
        # Picker gets the viewer's reusable recolor body as on_apply.
        assert opened["on_apply"] == stub._apply_color_settings
        # Opening alone does not recolor (that happens on Apply).
        assert stub.calls == []
        assert rv is not None

    def test_does_not_overwrite_existing_file(self, tk_root, tmp_path, monkeypatch):
        project = tmp_path / "proj"
        project.mkdir()
        existing = project / "plot_settings.yaml"
        custom = "categories:\n  costs:\n    mine: '#010203'\n"
        existing.write_text(custom, encoding="utf-8")

        class _FakePicker:
            def __init__(self, parent, path, on_apply=None):
                pass

        monkeypatch.setattr(
            "flextool.gui.dialogs.plot_settings_picker.PlotSettingsPicker",
            _FakePicker,
        )

        stub = _make_stub_viewer(project)
        stub._on_change_colors()

        assert existing.read_text(encoding="utf-8") == custom


class TestApplyColorSettings:
    def test_clears_cache_and_rerenders_when_no_live_plan(
        self, tk_root, tmp_path, monkeypatch,
    ):
        from flextool.plot_outputs import color_template

        project = tmp_path / "proj"
        project.mkdir()
        (project / "plot_settings.yaml").write_text(
            "entities:\n  node:\n    n1: '#abcdef'\n", encoding="utf-8",
        )

        cleared = []
        monkeypatch.setattr(
            color_template, "_clear_cache", lambda: cleared.append(True),
        )

        # No live plan cached → full clear + recompute fallback.
        stub = _make_stub_viewer(project, live_plan=None)
        stub._apply_color_settings()

        assert cleared == [True]
        assert stub.calls == ["clear_figure_cache", "trigger_replot"]

    def test_rebuilds_color_map_in_place(self, tk_root, tmp_path, monkeypatch):
        """A cached live plan with hints recolors IN PLACE: plan identity
        preserved, only prefetched figures dropped."""
        from flextool.plot_outputs import color_template
        from flextool.plot_outputs.plan import PlotPlan
        import pandas as pd

        project = tmp_path / "proj"
        project.mkdir()
        (project / "plot_settings.yaml").write_text(
            "entities:\n  unit:\n    coal: '#00ff00'\n", encoding="utf-8",
        )
        color_template._clear_cache()

        plan = PlotPlan(
            chart_type='stack',
            plot_name='p',
            total_file_count=1,
            processed_df=pd.DataFrame({'coal': [1.0]}),
            effective_plot_specs=[(None, ['coal'])],
            file_batches=[[0]],
            shared_color_map={'coal': (1.0, 0.0, 0.0)},  # old red
            color_entity_class='unit',
        )

        stub = _make_stub_viewer(project, live_plan=plan)
        stub._apply_color_settings()

        assert stub._live_plan is plan
        assert plan.shared_color_map == {'coal': (0.0, 1.0, 0.0)}
        assert stub.calls == ["clear_prefetched_figures", "trigger_replot"]


# ---------------------------------------------------------------------------
#  Minor UX fixes: Enter-to-apply + focus, neg round-trip, tab order
# ---------------------------------------------------------------------------


class TestColorPickerDialogKeys:
    def test_return_bound_and_ok_is_default(self, tk_root):
        from flextool.gui.dialogs.plot_settings_picker import ColorPickerDialog

        dlg = ColorPickerDialog(tk_root, "coal", "#212121", "#212121", True)
        # Enter is bound at the dialog level (→ OK / apply).
        assert dlg.bind("<Return>") != ""
        # OK is the default button (the Enter target visually).
        oks = [b for b in _iter_buttons(dlg) if str(b.cget("text")) == "OK"]
        assert oks and str(oks[0].cget("default")) == "active"
        dlg._on_cancel()


class TestPickerKeyboardEdit:
    def test_enter_on_row_opens_editor(self, tk_root, tmp_path, monkeypatch):
        picker, _ = _make_picker(tk_root, tmp_path)
        titles = _tab_titles(picker)
        unit = _tree_in_tab(picker, titles.index("unit"))
        coal = unit.get_children("")[0]
        unit.focus(coal)

        captured = _patch_dialog(monkeypatch, None)
        picker._on_row_return(types.SimpleNamespace(widget=unit))
        # The editor opened for the focused row.
        assert captured["dialog"].opened[0] == "coal"

    def test_focus_returns_to_tree_after_edit(
        self, tk_root, tmp_path, monkeypatch,
    ):
        picker, _ = _make_picker(tk_root, tmp_path)
        titles = _tab_titles(picker)
        unit = _tree_in_tab(picker, titles.index("unit"))
        coal = unit.get_children("")[0]

        _patch_dialog(monkeypatch, ("#00ff00", None))
        picker._edit_row_color(unit, coal)
        # Focus item + selection land back on the edited row.
        assert unit.focus() == coal
        assert coal in unit.selection()


class TestPickerNegRoundTrip:
    def test_set_neg_then_reopen_opens_unlinked_with_neg(
        self, tk_root, tmp_path, monkeypatch,
    ):
        picker, _ = _make_picker(tk_root, tmp_path)
        titles = _tab_titles(picker)
        unit = _tree_in_tab(picker, titles.index("unit"))
        coal = unit.get_children("")[0]

        # 1) Give bare 'coal' a distinct negative (unlinked) and save it.
        _patch_dialog(monkeypatch, ("#111111", "#222222"))
        picker._edit_row_color(unit, coal)
        sect = _section(picker._data, ("entities", "unit"))
        assert sect["coal"] == {"color": "#111111", "neg_color": "#222222"}

        # 2) Reopen: the negative is remembered → opens UNLINKED with it.
        captured = _patch_dialog(monkeypatch, None)
        picker._edit_row_color(unit, coal)
        name, pos_hex, neg_hex, linked = captured["dialog"].opened
        assert (name, pos_hex, neg_hex, linked) == (
            "coal", "#111111", "#222222", False,
        )


class TestPickerTabOrder:
    def test_button_traversal_order(self, tk_root, tmp_path):
        """Tk focus traversal follows child creation order; the buttons must
        be created Refresh → Undo → Redo → Apply → Save → Cancel."""
        picker, _ = _make_picker(tk_root, tmp_path)
        texts = [str(b.cget("text")) for b in _iter_buttons(picker)]
        assert texts == [
            "Refresh from DB", "Undo", "Redo",
            "Apply", "Save and exit", "Cancel",
        ]
