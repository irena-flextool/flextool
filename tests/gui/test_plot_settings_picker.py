"""Tests for the non-modal color/order picker (Stage 6.2).

Covers:
* ``PlotSettingsPicker`` builds a two-pane layout — a left category
  selector (fixed order, not reorderable) + a right entity list per
  category — each entity Treeview populated with the right names, and a
  composite swatch ``PhotoImage`` kept alive (not GC'd) per row.
* ``_write`` round-trips the working dict to byte-valid YAML; the debounced
  live flush writes + invokes ``on_apply`` (leaving the Cancel baseline
  untouched); Close writes + invokes + closes (keeps changes); Cancel
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
    # Category titles in display order — read from the left selector so this
    # asserts the on-screen list, not just the bookkeeping attribute.
    cat = picker._category_list
    titles = [cat.item(i, "text") for i in cat.get_children("")]
    assert titles == picker._category_order
    return titles


def _tree_in_tab(picker, index):
    return picker._category_trees[picker._category_order[index]]


def _row_names(tree) -> list[str]:
    return [tree.item(iid, "text") for iid in tree.get_children("")]


class TestPickerBuild:
    def test_tabs_entity_classes_always_shown_then_present_categories(
        self, tk_root, tmp_path,
    ):
        picker, _ = _make_picker(tk_root, tmp_path)
        # All entity-class tabs always appear (even empty nodeGroup/flowGroup/
        # connection), then present categories (costs, dispatch), then
        # scenarios.
        assert _tab_titles(picker) == [
            "nodeGroup", "flowGroup", "unit", "connection", "node",
            "costs", "dispatch", "scenarios",
        ]

    def test_entity_tabs_shown_even_when_empty(self, tk_root, tmp_path):
        data = {
            "scenarios": {},
            "categories": {"costs": {}},
            "entities": {"unit": {}, "node": {"n1": "#abcdef"}},
        }
        picker, _ = _make_picker(tk_root, tmp_path, data=data)
        # Every entity class shows (so absent classes are visible); the
        # scenarios tab is always shown and last; empty category sections are
        # still skipped.  (tmp_path has no output_parquet, so scenarios stays
        # empty.)
        assert _tab_titles(picker) == [
            "nodeGroup", "flowGroup", "unit", "connection", "node", "scenarios",
        ]

    def test_scenarios_populated_from_output_folders(self, tk_root, tmp_path):
        # Executed-scenario folders under output_parquet/ seed the scenarios
        # section on open (palette colors); _-prefixed manifest dirs skipped.
        pq = tmp_path / "output_parquet"
        (pq / "base_1").mkdir(parents=True)
        (pq / "high_2").mkdir()
        (pq / "_manifest").mkdir()
        picker, _ = _make_picker(tk_root, tmp_path, data={"entities": {"unit": {}}})
        assert "scenarios" in _tab_titles(picker)
        scen = picker._data.get("scenarios", {})
        assert set(scen) == {"base_1", "high_2"}
        assert all(isinstance(v, str) and v.startswith("#") for v in scen.values())

    def test_scenario_folder_merge_is_add_only(self, tk_root, tmp_path):
        pq = tmp_path / "output_parquet"
        (pq / "base_1").mkdir(parents=True)
        (pq / "new_1").mkdir()
        data = {"scenarios": {"base_1": "#123456"}}
        picker, _ = _make_picker(tk_root, tmp_path, data=data)
        scen = picker._data["scenarios"]
        assert scen["base_1"] == "#123456"  # existing color preserved
        assert "new_1" in scen and scen["new_1"].startswith("#")

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
        from flextool.gui.dialogs.plot_settings_picker import _swatch_width

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

        # Bare entity reserves the neg column (two-box width).
        assert _img_width(unit, coal_iid) == _swatch_width(True)
        # neg_color entity also two-box width (neg box drawn).
        assert _img_width(unit, chp_iid) == _swatch_width(True)
        # Category rows have no negative → single-box width.
        assert _img_width(costs, co2_iid) == _swatch_width(False)
        # Two-box rows are wider than single (reserved gap + neg box).
        assert _swatch_width(True) > _swatch_width(False)

    def test_rows_start_flush_left_no_indicator(self, tk_root, tmp_path):
        """The disclosure-indicator indent is removed (its ~18px is the
        empty space on the left), so the swatch starts flush-left and
        carries its own small inset instead."""
        picker, _ = _make_picker(tk_root, tmp_path)
        if picker._tree_style == "Treeview":
            pytest.skip("custom Treeview layout unsupported in this Tk")
        style = ttk.Style(picker)
        layout = repr(style.layout(f"{picker._tree_style}.Item")).lower()
        assert "indicator" not in layout  # the left-indent source is gone
        assert "image" in layout

    def test_non_modal_no_grab(self, tk_root, tmp_path):
        """The picker must not grab input (usable alongside the viewer)."""
        picker, _ = _make_picker(tk_root, tmp_path)
        # The picker must not be the current grab.
        assert picker.grab_current() in (None, "")


# ---------------------------------------------------------------------------
#  PlotSettingsPicker — two-pane category-list + entity-list layout
# ---------------------------------------------------------------------------


class TestPickerTwoPaneLayout:
    def test_category_list_is_single_select_selector(self, tk_root, tmp_path):
        """The left pane is a single-select ttk.Treeview listing the category
        titles in the fixed display order."""
        picker, _ = _make_picker(tk_root, tmp_path)
        cat = picker._category_list
        assert isinstance(cat, ttk.Treeview)
        assert str(cat.cget("selectmode")) == "browse"
        left_titles = [cat.item(i, "text") for i in cat.get_children("")]
        assert (
            left_titles
            == picker._category_order
            == [
                "nodeGroup", "flowGroup", "unit", "connection", "node",
                "costs", "dispatch", "scenarios",
            ]
        )

    def test_category_list_not_bound_to_reorder(self, tk_root, tmp_path):
        """The left selector is a pure selector: no drag / Alt-arrow / move
        reorder bindings are attached (only the entity trees carry those)."""
        picker, _ = _make_picker(tk_root, tmp_path)
        cat = picker._category_list
        assert not cat.bind("<Alt-Up>")
        assert not cat.bind("<Alt-Down>")
        assert not cat.bind("<ButtonPress-1>")
        assert not cat.bind("<B1-Motion>")
        # ... while every entity tree DOES carry the move bindings.
        for tree in picker._category_trees.values():
            assert tree.bind("<Alt-Up>")
            assert tree.bind("<ButtonPress-1>")

    def test_selecting_category_shows_its_entity_tree(self, tk_root, tmp_path):
        """Selecting a left category shows that category's entity tree and
        hides the previously shown one."""
        picker, _ = _make_picker(tk_root, tmp_path)
        # Default open shows the first category (nodeGroup).
        assert picker._current_category == "nodeGroup"
        assert picker._category_frames["nodeGroup"].winfo_manager() == "grid"

        # Select 'unit' via the left list → its entity tree becomes visible.
        picker._category_list.selection_set(picker._category_items["unit"])
        picker._on_category_select()
        assert picker._current_category == "unit"
        assert picker._category_frames["unit"].winfo_manager() == "grid"
        # The previously-shown category's tree is now hidden.
        assert picker._category_frames["nodeGroup"].winfo_manager() == ""

    def test_undo_preserves_selected_category(
        self, tk_root, tmp_path, monkeypatch,
    ):
        """A rebuild (undo) keeps the currently-selected category shown when
        it still exists."""
        picker, _ = _make_picker(tk_root, tmp_path)
        picker._category_list.selection_set(picker._category_items["unit"])
        picker._on_category_select()
        assert picker._current_category == "unit"

        unit = picker._category_trees["unit"]
        coal = unit.get_children("")[0]
        _patch_dialog(monkeypatch, ("#00ff00", None))
        picker._edit_row_color(unit, coal)
        picker._on_undo()
        # After the rebuild the selection is still 'unit' (not reset to first).
        assert picker._current_category == "unit"
        assert picker._category_frames["unit"].winfo_manager() == "grid"

    def test_default_height_is_about_80pct_screen(self, tk_root, tmp_path):
        """The dialog defaults to ~80% of the screen height.  Skipped
        gracefully if the WM ignores the requested geometry under Xvfb."""
        picker, _ = _make_picker(tk_root, tmp_path)
        picker.deiconify()
        picker.update()
        if not picker.winfo_ismapped():
            pytest.skip("WM did not map the window under Xvfb")
        expected = int(picker.winfo_screenheight() * 0.8)
        actual = picker.winfo_height()
        if abs(actual - expected) > 10:
            pytest.skip("WM ignored the requested geometry under Xvfb")
        assert abs(actual - expected) <= 10


# ---------------------------------------------------------------------------
#  PlotSettingsPicker — entity tabs auto-populate on OPEN (DB ∪ aggregates)
# ---------------------------------------------------------------------------


class TestPickerEntitySeeding:
    """On open the entity tabs seed additively from the UNION of the input
    DB(s) and the dispatch output aggregates — no manual Refresh needed."""

    def test_entities_populate_from_db_on_open(
        self, tk_root, tmp_path, monkeypatch,
    ):
        # DB FALLBACK path: tmp_path has no output_parquet, so the parquet-first
        # discovery finds nothing and the on-open merge falls back to the input
        # DB.  Seed the DB discovery BEFORE construction so the merge runs.
        from flextool.gui.dialogs.plot_settings_picker import PlotSettingsPicker

        monkeypatch.setattr(
            PlotSettingsPicker, "_discover_input_dbs",
            lambda self: ["sqlite:///fake.sqlite"],
        )
        _mock_fetch(monkeypatch, {"unit": ["coal", "gas"], "node": ["n1"]})

        picker, _ = _make_picker(
            tk_root, tmp_path, data={"entities": {"unit": {}}},
        )
        # DB entities appear on open (no Refresh click) with palette colors.
        unit = _section(picker._data, ("entities", "unit"))
        assert set(unit) == {"coal", "gas"}
        assert all(
            isinstance(v, str) and v.startswith("#") for v in unit.values()
        )
        # A class absent from the seed data is created from discovery too.
        assert set(_section(picker._data, ("entities", "node"))) == {"n1"}

    def test_dispatch_aggregate_populates_flowgroup_on_open(
        self, tk_root, tmp_path,
    ):
        # A processGroup aggregate that lives ONLY in solved output (never the
        # input DB) must become listable under flowGroup on open.
        import pandas as pd

        pq = tmp_path / "output_parquet" / "base_1"
        pq.mkdir(parents=True)
        pd.DataFrame(
            {"group": ["elec"], "group_aggregate": ["Fossil"]},
        ).to_parquet(
            pq / "nodeGroupDispatch__processGroup_Unit_to_group.parquet",
        )

        picker, _ = _make_picker(
            tk_root, tmp_path, data={"entities": {"flowGroup": {}}},
        )
        fg = _section(picker._data, ("entities", "flowGroup"))
        assert "Fossil" in fg
        assert isinstance(fg["Fossil"], str) and fg["Fossil"].startswith("#")

    def test_on_open_entity_merge_is_add_only(
        self, tk_root, tmp_path, monkeypatch,
    ):
        from flextool.gui.dialogs.plot_settings_picker import PlotSettingsPicker

        monkeypatch.setattr(
            PlotSettingsPicker, "_discover_input_dbs",
            lambda self: ["sqlite:///fake.sqlite"],
        )
        # DB knows coal + gas; 'chp' is a pre-existing entry absent from the DB.
        _mock_fetch(monkeypatch, {"unit": ["coal", "gas"]})
        data = {"entities": {"unit": {"coal": "#123456", "chp": "#654321"}}}

        picker, _ = _make_picker(tk_root, tmp_path, data=data)
        unit = _section(picker._data, ("entities", "unit"))
        # Existing colors + order preserved; chp NOT pruned (add-only); gas
        # appended after the existing entries.
        assert unit["coal"] == "#123456"
        assert unit["chp"] == "#654321"
        assert list(unit.keys())[:2] == ["coal", "chp"]
        assert "gas" in unit and unit["gas"].startswith("#")

    def test_on_open_merge_noop_without_sources(self, tk_root, tmp_path):
        # No input DB and no output_parquet → on-open entity merge is a no-op
        # (preserves the prior behaviour for projects with nothing to discover).
        data = {"entities": {"unit": {"coal": "#212121"}}}
        picker, _ = _make_picker(tk_root, tmp_path, data=data)
        assert picker._data["entities"] == {"unit": {"coal": "#212121"}}

    def test_new_flowgroup_entries_ordered_by_std_on_open(
        self, tk_root, tmp_path,
    ):
        """New flowGroup entries are appended in std-dev order (ascending,
        matching ``_order_dispatch_columns``' remaining buckets), independent
        of the discovery (alphabetical) order."""
        pq = tmp_path / "output_parquet" / "base_1"
        _write_dispatch_flowgroup_bundle(
            pq, "base_1",
            [
                ("Fossil", "u_fossil", [10.0, 0.0, 10.0, 0.0]),  # high std
                ("Wind", "u_wind", [1.0, 2.0, 1.0, 2.0]),        # low std
            ],
        )
        picker, _ = _make_picker(
            tk_root, tmp_path, data={"entities": {"flowGroup": {}}},
        )
        fg = _section(picker._data, ("entities", "flowGroup"))
        # Ascending std dev → Wind (low) before Fossil (high), NOT alphabetical
        # (which discovery would yield as Fossil, Wind).
        assert list(fg.keys()) == ["Wind", "Fossil"]
        assert all(
            isinstance(v, str) and v.startswith("#") for v in fg.values()
        )

    def test_on_open_merge_add_only_ignores_std_reorder(
        self, tk_root, tmp_path,
    ):
        """An existing user order + colors are preserved on open even when the
        parquet std-dev order would place the entries differently (add-only:
        the std order applies ONLY to genuinely new names)."""
        pq = tmp_path / "output_parquet" / "base_1"
        _write_dispatch_flowgroup_bundle(
            pq, "base_1",
            [
                ("Fossil", "u_fossil", [10.0, 0.0, 10.0, 0.0]),
                ("Wind", "u_wind", [1.0, 2.0, 1.0, 2.0]),
            ],
        )
        # Both already present, in alphabetical order (std order is Wind,
        # Fossil) with the user's own colors.
        data = {"entities": {
            "flowGroup": {"Fossil": "#aaaaaa", "Wind": "#bbbbbb"},
        }}
        picker, _ = _make_picker(tk_root, tmp_path, data=data)
        fg = _section(picker._data, ("entities", "flowGroup"))
        # Neither reordered nor recolored.
        assert list(fg.keys()) == ["Fossil", "Wind"]
        assert fg["Fossil"] == "#aaaaaa"
        assert fg["Wind"] == "#bbbbbb"


# ---------------------------------------------------------------------------
#  PlotSettingsPicker — flowGroup tab lists ONLY dispatch aggregators
# ---------------------------------------------------------------------------


def _mock_flow_aggregators(monkeypatch, names):
    """Patch ``fetch_flow_aggregator_flowgroups`` to a fixed set (no real DB)."""
    import flextool.scenario_comparison.input_entity_colors as iec

    monkeypatch.setattr(
        iec, "fetch_flow_aggregator_flowgroups", lambda _url: set(names),
    )


class TestFlowGroupDispatchFilter:
    """The flowGroup tab lists ONLY dispatch-aggregator flowGroups — the union
    of the input-DB ``flow_aggregator`` ∈ {dispatch_plots_only, both} and the
    parquet ``group_aggregate`` names.  Non-aggregators stay in ``self._data``
    (the file round-trips) but are not drawn."""

    def test_db_aggregator_listed_others_hidden_but_kept(
        self, tk_root, tmp_path, monkeypatch,
    ):
        from flextool.gui.dialogs.plot_settings_picker import PlotSettingsPicker

        monkeypatch.setattr(
            PlotSettingsPicker, "_discover_input_dbs",
            lambda self: ["sqlite:///fake.sqlite"],
        )
        # DB says only 'AggBoth' is a dispatch aggregator.
        _mock_flow_aggregators(monkeypatch, {"AggBoth"})
        data = {"entities": {
            "flowGroup": {"AggBoth": "#111111", "PlainNone": "#222222"},
            "unit": {"coal": "#333333"},
        }}
        picker, _ = _make_picker(tk_root, tmp_path, data=data)

        # Only the aggregator is drawn in the flowGroup tab…
        assert _row_names(picker._category_trees["flowGroup"]) == ["AggBoth"]
        # …but BOTH remain in the working dict so the file round-trips.
        fg = _section(picker._data, ("entities", "flowGroup"))
        assert set(fg) == {"AggBoth", "PlainNone"}
        # Other categories are never filtered.
        assert _row_names(picker._category_trees["unit"]) == ["coal"]

    def test_parquet_group_aggregate_listed_others_hidden(
        self, tk_root, tmp_path,
    ):
        # A processGroup aggregate present as a parquet ``group_aggregate`` is a
        # dispatch aggregator; a plain flowGroup in the file is not.
        pq = tmp_path / "output_parquet" / "base_1"
        _write_dispatch_flowgroup_bundle(
            pq, "base_1", [("Fossil", "u_fossil", [10.0, 0.0, 10.0, 0.0])],
        )
        data = {"entities": {"flowGroup": {
            "Fossil": "#aaaaaa", "PlainNone": "#bbbbbb",
        }}}
        picker, _ = _make_picker(tk_root, tmp_path, data=data)

        assert _row_names(picker._category_trees["flowGroup"]) == ["Fossil"]
        fg = _section(picker._data, ("entities", "flowGroup"))
        assert set(fg) == {"Fossil", "PlainNone"}
        assert "Fossil" in picker._dispatch_flowgroup_aggregators

    def test_empty_aggregator_set_shows_all_flowgroups(
        self, tk_root, tmp_path,
    ):
        # No parquet and no input DB → empty aggregator set → filter DISABLED
        # (never wrongly hide every flowGroup on a pre-solve project).
        data = {"entities": {"flowGroup": {
            "A": "#aaaaaa", "B": "#bbbbbb",
        }}}
        picker, _ = _make_picker(tk_root, tmp_path, data=data)
        assert picker._dispatch_flowgroup_aggregators == set()
        assert _row_names(picker._category_trees["flowGroup"]) == ["A", "B"]

    def test_rebuild_reapplies_filter(
        self, tk_root, tmp_path, monkeypatch,
    ):
        # Undo/redo rebuild via ``_build_panes`` → the same filter must reapply.
        from flextool.gui.dialogs.plot_settings_picker import PlotSettingsPicker

        monkeypatch.setattr(
            PlotSettingsPicker, "_discover_input_dbs",
            lambda self: ["sqlite:///fake.sqlite"],
        )
        _mock_flow_aggregators(monkeypatch, {"AggBoth"})
        data = {"entities": {"flowGroup": {
            "AggBoth": "#111111", "PlainNone": "#222222",
        }}}
        picker, _ = _make_picker(tk_root, tmp_path, data=data)
        picker._rebuild_panes()
        assert _row_names(picker._category_trees["flowGroup"]) == ["AggBoth"]


# ---------------------------------------------------------------------------
#  PlotSettingsPicker — live flush / Close / Cancel + on_apply wiring
# ---------------------------------------------------------------------------


class TestPickerButtons:
    def test_flush_writes_roundtrip_and_calls_on_apply(self, tk_root, tmp_path):
        calls = []
        picker, f = _make_picker(
            tk_root, tmp_path, on_apply=lambda: calls.append(1),
        )
        # The debounced live flush writes the file and re-renders; the window
        # stays open (it is not a close action).
        picker._flush_live_update()
        assert calls == [1]
        assert picker.winfo_exists()
        # File round-trips equal to the working dict.
        assert yaml.safe_load(f.read_text(encoding="utf-8")) == picker._data
        assert picker._data == _SAMPLE

    def test_flush_does_not_touch_cancel_baseline(self, tk_root, tmp_path):
        """A live flush must NOT move the on-open baseline: a later Cancel
        still reverts every change made since the dialog opened."""
        picker, f = _make_picker(tk_root, tmp_path, on_apply=lambda: None)
        original = f.read_text(encoding="utf-8")
        # Edit the working dict, then let a live flush write it to disk.
        picker._data["scenarios"]["S1"] = "#123456"
        picker._flush_live_update()
        assert (
            yaml.safe_load(f.read_text(encoding="utf-8"))["scenarios"]["S1"]
            == "#123456"
        )
        # Cancel reverts to the on-open text (baseline untouched by the flush).
        picker._on_cancel()
        assert f.read_text(encoding="utf-8") == original

    def test_close_writes_and_closes(self, tk_root, tmp_path):
        calls = []
        picker, f = _make_picker(
            tk_root, tmp_path, on_apply=lambda: calls.append(1),
        )
        picker._on_close()
        assert calls == [1]
        assert not picker.winfo_exists()
        assert yaml.safe_load(f.read_text(encoding="utf-8")) == _SAMPLE

    def test_cancel_restores_original_and_calls_on_apply(self, tk_root, tmp_path):
        calls = []
        picker, f = _make_picker(
            tk_root, tmp_path, on_apply=lambda: calls.append(1),
        )
        original = f.read_text(encoding="utf-8")
        # Simulate a prior live write that changed the file on disk.
        f.write_text("scenarios:\n  X: '#000000'\n", encoding="utf-8")
        picker._on_cancel()
        # Original on-open text restored byte-for-byte; on_apply (revert) fired.
        assert f.read_text(encoding="utf-8") == original
        assert calls == [1]
        assert not picker.winfo_exists()

    def test_no_on_apply_is_fine(self, tk_root, tmp_path):
        """Picker opened with no callback (PNG dialog) just writes on flush."""
        picker, f = _make_picker(tk_root, tmp_path, on_apply=None)
        picker._flush_live_update()  # must not raise
        assert yaml.safe_load(f.read_text(encoding="utf-8")) == _SAMPLE

    def test_close_keeps_edit(self, tk_root, tmp_path):
        """After an edit, Close writes the edited content and destroys; the
        file holds the edit (Close keeps changes)."""
        picker, f = _make_picker(tk_root, tmp_path, on_apply=lambda: None)
        picker._data["scenarios"]["S1"] = "#abcdef"
        picker._on_close()
        assert not picker.winfo_exists()
        assert (
            yaml.safe_load(f.read_text(encoding="utf-8"))["scenarios"]["S1"]
            == "#abcdef"
        )

    def test_cancel_reverts_edit_to_on_open_text(self, tk_root, tmp_path):
        """After an edit (and its live write), Cancel restores the on-open
        text byte-for-byte."""
        picker, f = _make_picker(tk_root, tmp_path, on_apply=lambda: None)
        original = f.read_text(encoding="utf-8")
        picker._data["scenarios"]["S1"] = "#abcdef"
        picker._flush_live_update()  # live write to disk
        picker._on_cancel()
        assert f.read_text(encoding="utf-8") == original

    def test_escape_and_window_x_route_to_keep_handler(
        self, tk_root, tmp_path, monkeypatch,
    ):
        """Escape and the window ``X`` (WM_DELETE_WINDOW) route to the KEEP
        handler (``_on_close``), NOT ``_on_cancel`` — live edits must not be
        silently reverted on close."""
        picker, _ = _make_picker(tk_root, tmp_path, on_apply=lambda: None)
        called = []
        # Both bindings dispatch through instance attribute lookup, so the
        # recorders are seen.
        picker._on_close = lambda: called.append("close")
        picker._on_cancel = lambda: called.append("cancel")

        # A key event is only delivered to a mapped, focused, NON-transient
        # window under bare Xvfb (a transient Toplevel never takes focus), so
        # clear transient + force focus purely to route the synthesised key.
        picker.wm_transient("")
        picker.deiconify()
        picker.update()
        picker.focus_force()
        picker.update()
        picker.event_generate("<Escape>", when="now")
        # The window ``X`` handler is invoked directly (no event needed).
        picker.tk.call(picker.protocol("WM_DELETE_WINDOW"))

        assert called == ["close", "close"]
        assert "cancel" not in called

    def test_apply_and_refresh_buttons_are_gone(self, tk_root, tmp_path):
        """No Apply button and no Refresh button in the dialog."""
        picker, _ = _make_picker(tk_root, tmp_path)
        texts = {str(b.cget("text")) for b in _iter_buttons(picker)}
        assert "Apply" not in texts
        assert "Refresh from DB" not in texts


class TestPickerLiveUpdate:
    """Every user mutation schedules a debounced live write + on_apply."""

    def test_reorder_schedules_live_update(self, tk_root, tmp_path):
        calls = []
        picker, f = _make_picker(
            tk_root, tmp_path, on_apply=lambda: calls.append(1),
        )
        # Opening alone does not schedule a write.
        assert picker._live_after_id is None

        titles = _tab_titles(picker)
        unit = _tree_in_tab(picker, titles.index("unit"))
        first = unit.get_children("")[0]
        unit.focus(first)
        unit.selection_set(first)
        picker._key_move(unit, +1)  # coal → below chp (a real reorder)

        # A live update is now pending (debounced); flushing it writes + fires.
        assert picker._live_after_id is not None
        picker._flush_live_update()
        assert picker._live_after_id is None
        assert calls == [1]
        loaded = yaml.safe_load(f.read_text(encoding="utf-8"))
        assert list(loaded["entities"]["unit"].keys()) == ["chp", "coal"]

    def test_color_edit_schedules_live_update(
        self, tk_root, tmp_path, monkeypatch,
    ):
        calls = []
        picker, f = _make_picker(
            tk_root, tmp_path, on_apply=lambda: calls.append(1),
        )
        titles = _tab_titles(picker)
        unit = _tree_in_tab(picker, titles.index("unit"))
        coal = unit.get_children("")[0]

        _patch_dialog(monkeypatch, ("#00ff00", None))
        picker._edit_row_color(unit, coal)

        assert picker._live_after_id is not None
        picker._flush_live_update()
        assert calls == [1]
        loaded = yaml.safe_load(f.read_text(encoding="utf-8"))
        assert loaded["entities"]["unit"]["coal"] == "#00ff00"

    def test_noop_reorder_does_not_schedule(self, tk_root, tmp_path):
        """An Alt-Up at the top (no actual reorder) schedules nothing."""
        picker, _ = _make_picker(tk_root, tmp_path)
        titles = _tab_titles(picker)
        scen = _tree_in_tab(picker, titles.index("scenarios"))
        top = scen.get_children("")[0]
        scen.focus(top)
        picker._key_move(scen, -1)  # already at top → no change
        assert picker._live_after_id is None

    def test_schedule_debounces_to_single_pending(self, tk_root, tmp_path):
        """A burst of mutations collapses into ONE pending after id."""
        picker, _ = _make_picker(tk_root, tmp_path, on_apply=lambda: None)
        picker._schedule_live_update()
        first = picker._live_after_id
        assert first is not None
        picker._schedule_live_update()
        second = picker._live_after_id
        assert second is not None
        # The prior pending call was cancelled and replaced (not stacked).
        assert second != first
        picker._flush_live_update()
        assert picker._live_after_id is None


# ---------------------------------------------------------------------------
#  PlotSettingsPicker — reordering (drag + keyboard) → persisted order
# ---------------------------------------------------------------------------


def _section(data: dict, path: tuple[str, ...]) -> dict:
    cur = data
    for key in path:
        cur = cur[key]
    return cur


def _write_dispatch_flowgroup_bundle(pdir, scenario, aggregates):
    """Write a minimal-but-real dispatch output bundle under *pdir*.

    Lets the picker's parquet-first discovery run end-to-end
    (``discover_dispatch_entities`` for the class names + ``prepare_dispatch_data``
    for the per-entity std devs) instead of monkeypatching the DB path.

    *aggregates* is a list of ``(aggregate_name, unit_name, values)`` — each an
    ``Unit_to_group`` processGroup aggregate whose single member unit produces
    *values* (a per-time flow series) into node ``n1`` of nodeGroup ``elec``.
    The aggregate's dispatch column std dev is therefore
    ``pd.Series(values).std()`` (all-positive columns keep their name).
    """
    import pandas as pd

    pdir.mkdir(parents=True, exist_ok=True)
    k = len(aggregates)
    # nodeGroup 'elec' flagged for dispatch, with member node 'n1'.
    pd.DataFrame({"group": ["elec"]}).to_parquet(
        pdir / "nodeGroupDispatch.parquet",
    )
    pd.DataFrame({"group": ["elec"], "node": ["n1"]}).to_parquet(
        pdir / "group_node.parquet",
    )
    # Unit_to_group aggregates + their single-unit membership.
    pd.DataFrame({
        "group": ["elec"] * k,
        "group_aggregate": [a for a, _u, _v in aggregates],
    }).to_parquet(
        pdir / "nodeGroupDispatch__processGroup_Unit_to_group.parquet",
    )
    pd.DataFrame({
        "group": ["elec"] * k,
        "group_aggregate": [a for a, _u, _v in aggregates],
        "process": [u for _a, u, _v in aggregates],
        "unit": [u for _a, u, _v in aggregates],
        "node": ["n1"] * k,
    }).to_parquet(
        pdir / "nodeGroupDispatch__processGroup__process__unit__to_node.parquet",
    )
    # unit_outputNode_dt_ee: (scenario, unit, node) columns over a time index.
    n = len(aggregates[0][2]) if aggregates else 0
    time_idx = pd.Index(range(n), name="time")
    flow = pd.DataFrame(
        {i: v for i, (_a, _u, v) in enumerate(aggregates)}, index=time_idx,
    )
    flow.columns = pd.MultiIndex.from_tuples(
        [(scenario, u, "n1") for _a, u, _v in aggregates],
        names=["scenario", "unit", "node"],
    )
    flow.to_parquet(pdir / "unit_outputNode_dt_ee.parquet")


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

        # A move drag starts on an ALREADY-SELECTED row.  Select coal, then
        # press on it, drag down onto chp, release.
        unit.selection_set(coal)
        assert picker._on_drag_start(_ev(0)) == "break"  # move mode engaged
        assert picker._drag_move[unit] is True
        picker._on_drag_motion(_ev(1))
        picker._on_drag_end(_ev(1))

        assert _row_names(unit) == ["chp", "coal"]
        assert picker._drag_move[unit] is False
        sect = _section(picker._data, ("entities", "unit"))
        assert list(sect.keys()) == ["chp", "coal"]
        assert sect["chp"] == {"color": "#E64A19", "neg_color": "#9c3010"}

    def test_drag_on_unselected_row_selects_and_moves(
        self, tk_root, tmp_path, monkeypatch,
    ):
        """Pressing an UNSELECTED row (no modifier) collapses the selection
        onto it and starts a MOVE drag, so a drag begun on a not-yet-selected
        item picks it up and reorders."""
        picker, _ = _make_picker(tk_root, tmp_path)
        titles = _tab_titles(picker)
        unit = _tree_in_tab(picker, titles.index("unit"))
        coal, chp = unit.get_children("")
        monkeypatch.setattr(
            unit, "identify_row", lambda y: {0: coal, 1: chp}.get(y, ""))

        def _ev(y):
            return types.SimpleNamespace(widget=unit, y=y)

        # Another row is selected; press on the UNSELECTED coal.
        unit.selection_set(chp)
        assert picker._on_drag_start(_ev(0)) == "break"
        # coal is now the (sole) selection and a move drag is engaged.
        assert picker._drag_move[unit] is True
        assert unit.selection() == (coal,)
        picker._on_drag_motion(_ev(1))
        picker._on_drag_end(_ev(1))
        assert _row_names(unit) == ["chp", "coal"]
        sect = _section(picker._data, ("entities", "unit"))
        assert list(sect.keys()) == ["chp", "coal"]

    def test_drag_start_with_modifier_defers_to_native(
        self, tk_root, tmp_path, monkeypatch,
    ):
        """A Shift/Ctrl press is left to ttk (extend/toggle multi-select),
        so no move drag is primed."""
        picker, _ = _make_picker(tk_root, tmp_path)
        titles = _tab_titles(picker)
        unit = _tree_in_tab(picker, titles.index("unit"))
        coal, chp = unit.get_children("")
        monkeypatch.setattr(
            unit, "identify_row", lambda y: {0: coal, 1: chp}.get(y, ""))
        # Control (0x0004) held while pressing coal → native selection.
        evt = types.SimpleNamespace(widget=unit, y=0, state=0x0004)
        assert picker._on_drag_start(evt) is None
        assert picker._drag_move[unit] is False

    def test_drag_to_upper_half_of_top_row_reaches_index_zero(
        self, tk_root, tmp_path, monkeypatch,
    ):
        """Dragging onto the UPPER half of the first row drops the block at
        index 0 — the top slot must be reachable (regression: a hard '+1'
        made it impossible to drag anything above the first row)."""
        picker, _ = _make_picker(tk_root, tmp_path)
        unit = _tree_in_tab(picker, _tab_titles(picker).index("unit"))
        coal, chp = unit.get_children("")
        # coal occupies y-rows 0..20 (midpoint 10); chp 20..40.
        monkeypatch.setattr(
            unit, "identify_row",
            lambda y: coal if y < 20 else chp,
        )
        monkeypatch.setattr(
            unit, "bbox",
            lambda item: (0, 0, 100, 20) if item == coal else (0, 20, 100, 20),
        )

        def _ev(y):
            return types.SimpleNamespace(widget=unit, y=y)

        # Grab the BOTTOM row and drag it onto the UPPER half of the top row.
        unit.selection_set(chp)
        picker._on_drag_start(_ev(30))          # press on chp
        picker._on_drag_motion(_ev(2))          # hover coal's upper half
        picker._on_drag_end(_ev(2))
        assert _row_names(unit) == ["chp", "coal"]
        assert list(_section(picker._data, ("entities", "unit")).keys()) == [
            "chp", "coal",
        ]

    def test_drag_to_lower_half_keeps_below(self, tk_root, tmp_path, monkeypatch):
        """Dragging onto the LOWER half of a row drops the block after it."""
        data = {"entities": {"unit": {
            "a": "#111111", "b": "#222222", "c": "#333333",
        }}}
        picker, _ = _make_picker(tk_root, tmp_path, data=data)
        unit = _tree_in_tab(picker, _tab_titles(picker).index("unit"))
        a, b, c = unit.get_children("")
        rows = {a: (0, 0, 100, 20), b: (0, 20, 100, 20), c: (0, 40, 100, 20)}
        monkeypatch.setattr(
            unit, "identify_row",
            lambda y: a if y < 20 else (b if y < 40 else c),
        )
        monkeypatch.setattr(unit, "bbox", lambda item: rows[item])

        def _ev(y):
            return types.SimpleNamespace(widget=unit, y=y)

        # Drag 'a' (top) down onto the LOWER half of 'b' → lands after b.
        unit.selection_set(a)
        picker._on_drag_start(_ev(5))
        picker._on_drag_motion(_ev(38))  # b's lower half
        picker._on_drag_end(_ev(38))
        assert _row_names(unit) == ["b", "a", "c"]

    def test_drag_start_takes_keyboard_focus(
        self, tk_root, tmp_path, monkeypatch,
    ):
        """Pressing a row must claim the tree's KEYBOARD focus so the
        Alt-Up/Alt-Down reorder bindings fire (they were dead because the
        press returned 'break' and suppressed ttk's default focus grab)."""
        picker, _ = _make_picker(tk_root, tmp_path)
        unit = _tree_in_tab(picker, _tab_titles(picker).index("unit"))
        coal = unit.get_children("")[0]
        monkeypatch.setattr(unit, "identify_row", lambda y: coal)
        focus_calls = []
        monkeypatch.setattr(unit, "focus_set", lambda: focus_calls.append(1))

        picker._on_drag_start(types.SimpleNamespace(widget=unit, y=0))
        assert focus_calls == [1]

    def test_extended_selectmode(self, tk_root, tmp_path):
        """Trees allow multi-selection (Shift/Ctrl + native draw-select)."""
        picker, _ = _make_picker(tk_root, tmp_path)
        for tree in picker._tree_section:
            assert str(tree.cget("selectmode")) == "extended"

    def test_alt_move_group_of_selected_rows(self, tk_root, tmp_path):
        """Alt+Down moves the WHOLE selection as a block."""
        data = {"entities": {"unit": {
            "a": "#111111", "b": "#222222", "c": "#333333", "d": "#444444",
        }}}
        picker, _ = _make_picker(tk_root, tmp_path, data=data)
        unit = _tree_in_tab(picker, _tab_titles(picker).index("unit"))
        a, b, c, d = unit.get_children("")
        unit.selection_set(a, b)  # select the top two
        picker._key_move(unit, +1)  # move the pair down one
        assert _row_names(unit) == ["c", "a", "b", "d"]
        assert set(unit.selection()) == {a, b}
        assert list(_section(picker._data, ("entities", "unit")).keys()) == [
            "c", "a", "b", "d",
        ]

    def test_reordered_order_is_written_to_file(self, tk_root, tmp_path):
        """After a reorder, the live flush writes the file with the new key
        order and values intact (sort_keys=False preserves it)."""
        picker, f = _make_picker(tk_root, tmp_path)
        titles = _tab_titles(picker)
        unit = _tree_in_tab(picker, titles.index("unit"))
        first = unit.get_children("")[0]
        unit.focus(first)
        unit.selection_set(first)
        picker._key_move(unit, +1)  # coal → below chp

        picker._flush_live_update()

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


class TestColorPickerDialog:
    """Embedded two-chooser dialog: link semantics + result contract."""

    def test_linked_default_mirrors_and_returns_none_neg(self, tk_root):
        from flextool.gui.dialogs.plot_settings_picker import ColorPickerDialog

        dlg = ColorPickerDialog(tk_root, "coal", "#212121", "#777777", True)
        # Linked: the negative mirrors the positive on open.
        assert dlg._linked.get() is True
        assert dlg._neg_chooser.get_hex() == dlg._pos_chooser.get_hex()
        dlg._on_ok()
        # Linked → neg returned as None (bare entry).
        assert dlg.result == ("#212121", None)

    def test_pos_change_mirrors_to_linked_neg(self, tk_root):
        from flextool.gui.dialogs.plot_settings_picker import ColorPickerDialog

        dlg = ColorPickerDialog(tk_root, "coal", "#212121", "#212121", True)
        # A positive change (e.g. via the square) mirrors onto the negative.
        dlg._pos_chooser.set_hex("#00ff00", user=True)
        assert dlg._neg_chooser.get_hex() == "#00ff00"
        assert dlg._linked.get() is True

    def test_editing_negative_breaks_link(self, tk_root):
        from flextool.gui.dialogs.plot_settings_picker import ColorPickerDialog

        dlg = ColorPickerDialog(tk_root, "coal", "#212121", "#212121", True)
        assert dlg._linked.get() is True
        # A USER edit of the negative separates the colors → auto-unlink.
        dlg._neg_chooser.set_hex("#aabbcc", user=True)
        assert dlg._linked.get() is False
        dlg._on_ok()
        assert dlg.result == ("#212121", "#aabbcc")

    def test_entry_with_neg_opens_unlinked(self, tk_root):
        from flextool.gui.dialogs.plot_settings_picker import ColorPickerDialog

        dlg = ColorPickerDialog(tk_root, "chp", "#E64A19", "#9c3010", False)
        assert dlg._linked.get() is False
        assert dlg._pos_chooser.get_hex() == "#e64a19"
        assert dlg._neg_chooser.get_hex() == "#9c3010"
        dlg._on_ok()
        assert dlg.result == ("#e64a19", "#9c3010")

    def test_relink_collapses_to_pos(self, tk_root):
        from flextool.gui.dialogs.plot_settings_picker import ColorPickerDialog

        dlg = ColorPickerDialog(tk_root, "chp", "#E64A19", "#9c3010", False)
        # Re-check the link box → neg := pos.
        dlg._linked.set(True)
        dlg._on_link_toggle()
        assert dlg._neg_chooser.get_hex() == "#e64a19"
        dlg._on_ok()
        assert dlg.result == ("#e64a19", None)

    def test_cancel_returns_none(self, tk_root):
        from flextool.gui.dialogs.plot_settings_picker import ColorPickerDialog

        dlg = ColorPickerDialog(tk_root, "coal", "#212121", "#212121", True)
        dlg._on_cancel()
        assert dlg.result is None

    def test_single_mode_hides_negative_and_returns_one_color(self, tk_root):
        from flextool.gui.dialogs.plot_settings_picker import ColorPickerDialog

        dlg = ColorPickerDialog(
            tk_root, "co2", "#4d4d4d", "#4d4d4d", True, single=True,
        )
        # No negative chooser in single mode.
        assert dlg._neg_chooser is None
        assert dlg._pos_chooser.get_hex() == "#4d4d4d"
        dlg._on_ok()
        assert dlg.result == ("#4d4d4d", None)

    def test_chooser_roundtrips_hex(self, tk_root):
        from flextool.gui.dialogs.plot_settings_picker import _ColorChooser

        ch = _ColorChooser(tk_root, "#3a7bd5")
        assert ch.get_hex() == "#3a7bd5"
        ch.set_hex("#ff8800")
        assert ch.get_hex() == "#ff8800"


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
        def __init__(self, parent, name, pos_hex, neg_hex, linked, *,
                     single=False):
            super().__init__(parent)
            self.withdraw()
            self.result = result
            self.opened = (name, pos_hex, neg_hex, linked)
            self.single = single
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
        from flextool.gui.dialogs.plot_settings_picker import _swatch_width

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
        assert img.width() == _swatch_width(True)

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
        from flextool.gui.dialogs.plot_settings_picker import _swatch_width

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
        # so the image stays two-box width even after collapsing to bare.
        assert img.width() == _swatch_width(True)

    def test_category_row_edits_bare_color(
        self, tk_root, tmp_path, monkeypatch,
    ):
        picker, _ = _make_picker(tk_root, tmp_path)
        titles = _tab_titles(picker)
        costs = _tree_in_tab(picker, titles.index("costs"))
        co2 = costs.get_children("")[0]

        _patch_dialog(monkeypatch, ("#fedcba", None))
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

        _patch_dialog(monkeypatch, ("#0a0b0c", None))
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

        _patch_dialog(monkeypatch, None)  # Cancel.
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
        # Prime a stale move-drag as a prior ButtonPress would.
        picker._drag_move[unit] = True

        called = []
        monkeypatch.setattr(
            picker, "_edit_row_color",
            lambda tree, item: called.append((tree, item)),
        )
        evt = types.SimpleNamespace(widget=unit, y=0)
        result = picker._on_row_double_click(evt)
        # Resolves the row, clears the move-drag (no reorder), edits.
        assert result == "break"
        assert called == [(unit, coal)]
        assert picker._drag_move[unit] is False

    def test_double_click_binding_registered(self, tk_root, tmp_path):
        picker, _ = _make_picker(tk_root, tmp_path)
        for tree in picker._tree_section:
            assert tree.bind("<Double-Button-1>")


# ---------------------------------------------------------------------------
#  PlotSettingsPicker — dispatch specials reorderable WITHIN each sign group
#  (the flowGroups divider is the boundary; positives above, negatives below)
# ---------------------------------------------------------------------------


# All six special columns in the default visual (top-to-bottom) order — the
# file's ``categories.dispatch`` key order IS the visual stack order the engine
# honors — plus a couple of ordinary categories/entities so the "other
# categories stay reorderable" checks have something to move.
_DISPATCH_ALL = {
    "scenarios": {"S1": "#1f77b4"},
    "categories": {
        "costs": {"co2": "#4d4d4d", "vom": "#222222"},
        "dispatch": {
            # positive group, visual top→bottom
            "LossOfLoad": "crimson",
            "Discharge": "aqua",
            "Import": "indigo",
            # negative group, visual top→bottom
            "internal_losses": "darkgray",
            "Export": "purple",
            "Charge": "lime",
        },
    },
    "entities": {"unit": {"coal": "#212121", "gas": "#333333"}},
}

_DISPATCH_ROWS = [
    "LossOfLoad", "Discharge", "Import",
    "flowGroups",
    "internal_losses", "Export", "Charge",
]


def _divider_iid(tree):
    """Return the synthetic divider row's item id (by its display text)."""
    for iid in tree.get_children(""):
        if tree.item(iid, "text") == "flowGroups":
            return iid
    return None


class TestDispatchSignGroupReorderDivider:
    """The dispatch category renders its specials in two sign groups separated
    by an immovable, non-editable ``flowGroups`` divider.  Specials are
    reorderable WITHIN each sign group (positives among themselves above the
    divider, negatives among themselves below), never across it; the divider
    itself never moves and is never written to ``self._data``.  Colors stay
    editable."""

    def test_rows_follow_file_order_within_each_group(
        self, tk_root, tmp_path,
    ):
        picker, _ = _make_picker(tk_root, tmp_path, data=_DISPATCH_ALL)
        dispatch = picker._category_trees["dispatch"]
        # Positives (file order), divider, negatives (file order).
        assert _row_names(dispatch) == _DISPATCH_ROWS

    def test_scrambled_file_order_is_preserved_within_groups(
        self, tk_root, tmp_path,
    ):
        """A non-default file key order is honored within each sign group (the
        picker no longer forces a canonical special order)."""
        data = {
            "categories": {"dispatch": {
                "Export": "purple",
                "Import": "indigo",
                "Charge": "lime",
                "Discharge": "aqua",
                "internal_losses": "darkgray",
                "LossOfLoad": "crimson",
            }},
            "entities": {"unit": {}},
        }
        picker, _ = _make_picker(tk_root, tmp_path, data=data)
        dispatch = picker._category_trees["dispatch"]
        # Positives in file order (Import, Discharge, LossOfLoad), divider, then
        # negatives in file order (Export, Charge, internal_losses).
        assert _row_names(dispatch) == [
            "Import", "Discharge", "LossOfLoad",
            "flowGroups",
            "Export", "Charge", "internal_losses",
        ]

    def test_divider_is_not_a_data_key(self, tk_root, tmp_path):
        picker, _ = _make_picker(tk_root, tmp_path, data=_DISPATCH_ALL)
        sect = _section(picker._data, ("categories", "dispatch"))
        assert "flowGroups" not in sect
        assert set(sect) == {
            "LossOfLoad", "Discharge", "Import",
            "internal_losses", "Export", "Charge",
        }

    def test_divider_is_tracked_and_tagged(self, tk_root, tmp_path):
        picker, _ = _make_picker(tk_root, tmp_path, data=_DISPATCH_ALL)
        dispatch = picker._category_trees["dispatch"]
        div = _divider_iid(dispatch)
        assert div is not None
        assert (dispatch, div) in picker._divider_items
        assert picker._is_divider(dispatch, div) is True
        from flextool.gui.dialogs.plot_settings_picker import _DIVIDER_TAG
        assert _DIVIDER_TAG in dispatch.item(div, "tags")

    def test_color_edit_on_divider_is_noop(
        self, tk_root, tmp_path, monkeypatch,
    ):
        picker, _ = _make_picker(tk_root, tmp_path, data=_DISPATCH_ALL)
        dispatch = picker._category_trees["dispatch"]
        div = _divider_iid(dispatch)

        captured = _patch_dialog(monkeypatch, ("#00ff00", None))
        before = dict(_section(picker._data, ("categories", "dispatch")))
        picker._edit_row_color(dispatch, div)

        assert "dialog" not in captured
        assert _section(picker._data, ("categories", "dispatch")) == before
        assert "flowGroups" not in _section(
            picker._data, ("categories", "dispatch"),
        )

    def test_real_special_row_is_color_editable(
        self, tk_root, tmp_path, monkeypatch,
    ):
        picker, f = _make_picker(
            tk_root, tmp_path, data=_DISPATCH_ALL, on_apply=lambda: None,
        )
        dispatch = picker._category_trees["dispatch"]
        lol = dispatch.get_children("")[0]
        assert dispatch.item(lol, "text") == "LossOfLoad"

        captured = _patch_dialog(monkeypatch, ("#00ff00", None))
        picker._edit_row_color(dispatch, lol)
        assert captured["dialog"].opened[0] == "LossOfLoad"
        sect = _section(picker._data, ("categories", "dispatch"))
        assert sect["LossOfLoad"] == "#00ff00"
        # A color edit does not reorder the rows.
        picker._flush_live_update()
        assert _row_names(dispatch) == _DISPATCH_ROWS

    # ── Reorder WITHIN a sign group (Alt-move) ─────────────────────
    def test_alt_move_positive_within_top_group_reorders_and_writes(
        self, tk_root, tmp_path,
    ):
        picker, _ = _make_picker(tk_root, tmp_path, data=_DISPATCH_ALL)
        dispatch = picker._category_trees["dispatch"]
        # Move Discharge (2nd positive) up above LossOfLoad.
        discharge = dispatch.get_children("")[1]
        assert dispatch.item(discharge, "text") == "Discharge"
        dispatch.focus(discharge)
        dispatch.selection_set(discharge)
        assert picker._key_move(dispatch, -1) == "break"

        assert _row_names(dispatch) == [
            "Discharge", "LossOfLoad", "Import",
            "flowGroups",
            "internal_losses", "Export", "Charge",
        ]
        # categories.dispatch rewritten in the new (divider-excluded) order,
        # positives first then negatives; live update scheduled.
        assert list(
            _section(picker._data, ("categories", "dispatch")).keys()
        ) == [
            "Discharge", "LossOfLoad", "Import",
            "internal_losses", "Export", "Charge",
        ]
        assert picker._live_after_id is not None
        assert "flowGroups" not in _section(
            picker._data, ("categories", "dispatch"),
        )

    def test_alt_move_negative_within_bottom_group_reorders_and_writes(
        self, tk_root, tmp_path,
    ):
        picker, _ = _make_picker(tk_root, tmp_path, data=_DISPATCH_ALL)
        dispatch = picker._category_trees["dispatch"]
        # Move Export (2nd negative) up above internal_losses.
        export = dispatch.get_children("")[5]
        assert dispatch.item(export, "text") == "Export"
        dispatch.focus(export)
        dispatch.selection_set(export)
        assert picker._key_move(dispatch, -1) == "break"

        assert _row_names(dispatch) == [
            "LossOfLoad", "Discharge", "Import",
            "flowGroups",
            "Export", "internal_losses", "Charge",
        ]
        assert list(
            _section(picker._data, ("categories", "dispatch")).keys()
        ) == [
            "LossOfLoad", "Discharge", "Import",
            "Export", "internal_losses", "Charge",
        ]

    def test_alt_move_positive_down_into_divider_is_clamped(
        self, tk_root, tmp_path,
    ):
        """A positive at the bottom of its group cannot cross the divider — the
        move is clamped (no crossing) and nothing changes; the divider stays."""
        picker, _ = _make_picker(tk_root, tmp_path, data=_DISPATCH_ALL)
        dispatch = picker._category_trees["dispatch"]
        imp = dispatch.get_children("")[2]  # Import — last positive
        assert dispatch.item(imp, "text") == "Import"
        dispatch.focus(imp)
        dispatch.selection_set(imp)
        assert picker._key_move(dispatch, +1) == "break"

        # Unchanged, divider still at index 3, no live update, no undo.
        assert _row_names(dispatch) == _DISPATCH_ROWS
        assert dispatch.get_children("")[3] == _divider_iid(dispatch)
        assert picker._live_after_id is None
        assert picker._undo_stack == []

    def test_alt_move_negative_up_into_divider_is_clamped(
        self, tk_root, tmp_path,
    ):
        picker, _ = _make_picker(tk_root, tmp_path, data=_DISPATCH_ALL)
        dispatch = picker._category_trees["dispatch"]
        il = dispatch.get_children("")[4]  # internal_losses — top negative
        assert dispatch.item(il, "text") == "internal_losses"
        dispatch.focus(il)
        dispatch.selection_set(il)
        assert picker._key_move(dispatch, -1) == "break"

        assert _row_names(dispatch) == _DISPATCH_ROWS
        assert dispatch.get_children("")[3] == _divider_iid(dispatch)
        assert picker._live_after_id is None

    def test_divider_itself_never_moves_via_alt(self, tk_root, tmp_path):
        """Focusing the divider and Alt-moving it is a no-op (it drops out of
        the movable selection); it never enters self._data."""
        picker, _ = _make_picker(tk_root, tmp_path, data=_DISPATCH_ALL)
        dispatch = picker._category_trees["dispatch"]
        div = _divider_iid(dispatch)
        dispatch.focus(div)
        dispatch.selection_set(div)
        assert picker._key_move(dispatch, -1) == "break"
        assert picker._key_move(dispatch, +1) == "break"
        assert _row_names(dispatch) == _DISPATCH_ROWS
        assert "flowGroups" not in _section(
            picker._data, ("categories", "dispatch"),
        )
        assert picker._undo_stack == []

    # ── Reorder WITHIN a sign group (drag) + clamp ─────────────────
    def test_drag_positive_within_group_reorders_and_persists(
        self, tk_root, tmp_path, monkeypatch,
    ):
        picker, _ = _make_picker(tk_root, tmp_path, data=_DISPATCH_ALL)
        dispatch = picker._category_trees["dispatch"]
        rows = dispatch.get_children("")
        lol, imp = rows[0], rows[2]  # LossOfLoad (top), Import (last positive)
        # Map cursor-y to rows: 0→LossOfLoad, 2→Import.
        monkeypatch.setattr(
            dispatch, "identify_row",
            lambda y: {0: lol, 2: imp}.get(y, ""),
        )

        def _ev(y):
            return types.SimpleNamespace(widget=dispatch, y=y)

        dispatch.selection_set(lol)
        assert picker._on_drag_start(_ev(0)) == "break"
        assert picker._drag_move[dispatch] is True
        picker._on_drag_motion(_ev(2))   # drag down onto Import
        picker._on_drag_end(_ev(2))

        # LossOfLoad moved to the bottom of the positive group (above divider).
        assert _row_names(dispatch) == [
            "Discharge", "Import", "LossOfLoad",
            "flowGroups",
            "internal_losses", "Export", "Charge",
        ]
        assert list(
            _section(picker._data, ("categories", "dispatch")).keys()
        ) == [
            "Discharge", "Import", "LossOfLoad",
            "internal_losses", "Export", "Charge",
        ]

    def test_drag_positive_onto_negative_is_clamped_to_boundary(
        self, tk_root, tmp_path, monkeypatch,
    ):
        """Dragging a positive down onto a negative row clamps it to just above
        the divider (no crossing); the divider stays put."""
        picker, _ = _make_picker(tk_root, tmp_path, data=_DISPATCH_ALL)
        dispatch = picker._category_trees["dispatch"]
        rows = dispatch.get_children("")
        imp, chg = rows[2], rows[6]  # Import (last positive), Charge (last neg)
        monkeypatch.setattr(
            dispatch, "identify_row",
            lambda y: {2: imp, 6: chg}.get(y, ""),
        )

        def _ev(y):
            return types.SimpleNamespace(widget=dispatch, y=y)

        dispatch.selection_set(imp)
        assert picker._on_drag_start(_ev(2)) == "break"
        picker._on_drag_motion(_ev(6))   # drag down onto Charge (a negative)
        picker._on_drag_end(_ev(6))

        # No crossing: rows unchanged, divider still at index 3.
        assert _row_names(dispatch) == _DISPATCH_ROWS
        assert dispatch.get_children("")[3] == _divider_iid(dispatch)

    def test_drag_start_on_divider_swallows_and_primes_no_move(
        self, tk_root, tmp_path, monkeypatch,
    ):
        picker, _ = _make_picker(tk_root, tmp_path, data=_DISPATCH_ALL)
        dispatch = picker._category_trees["dispatch"]
        div = _divider_iid(dispatch)
        monkeypatch.setattr(dispatch, "identify_row", lambda y: div)

        def _ev(y):
            return types.SimpleNamespace(widget=dispatch, y=y)

        before_rows = _row_names(dispatch)
        # Press on the divider is swallowed ("break"), no move primed, and it
        # is not selected.
        assert picker._on_drag_start(_ev(0)) == "break"
        assert picker._drag_move[dispatch] is False
        assert div not in dispatch.selection()
        picker._on_drag_end(_ev(0))
        assert _row_names(dispatch) == before_rows

    # ── Other categories / entity classes unaffected ──────────────
    def test_other_category_still_reorderable_and_no_divider(
        self, tk_root, tmp_path,
    ):
        picker, _ = _make_picker(tk_root, tmp_path, data=_DISPATCH_ALL)
        costs = picker._category_trees["costs"]
        assert costs not in picker._fixed_order_trees
        assert _divider_iid(costs) is None
        assert _row_names(costs) == ["co2", "vom"]

        top = costs.get_children("")[0]
        costs.focus(top)
        costs.selection_set(top)
        picker._key_move(costs, +1)  # co2 → below vom
        assert _row_names(costs) == ["vom", "co2"]
        assert list(_section(picker._data, ("categories", "costs")).keys()) == [
            "vom", "co2",
        ]

    def test_entity_class_still_reorderable(self, tk_root, tmp_path):
        picker, _ = _make_picker(tk_root, tmp_path, data=_DISPATCH_ALL)
        unit = picker._category_trees["unit"]
        assert unit not in picker._fixed_order_trees
        assert _divider_iid(unit) is None
        top = unit.get_children("")[0]
        unit.focus(top)
        unit.selection_set(top)
        picker._key_move(unit, +1)
        assert _row_names(unit) == ["gas", "coal"]

    def test_rebuild_reconstructs_order_and_single_divider(
        self, tk_root, tmp_path,
    ):
        picker, _ = _make_picker(tk_root, tmp_path, data=_DISPATCH_ALL)
        picker._rebuild_panes()
        dispatch = picker._category_trees["dispatch"]
        assert _row_names(dispatch) == _DISPATCH_ROWS
        dividers = [
            iid for iid in dispatch.get_children("")
            if dispatch.item(iid, "text") == "flowGroups"
        ]
        assert len(dividers) == 1
        assert (dispatch, dividers[0]) in picker._divider_items
        # Reorder still constrained after the rebuild (Import can't cross down).
        imp = dispatch.get_children("")[2]
        dispatch.focus(imp)
        dispatch.selection_set(imp)
        picker._key_move(dispatch, +1)
        assert _row_names(dispatch) == _DISPATCH_ROWS

    def test_partial_specials_only_present_rows_and_divider(
        self, tk_root, tmp_path,
    ):
        """Only specials present in the file are shown; the divider still sits
        between the positive and negative groups."""
        data = {
            "categories": {"dispatch": {
                "Charge": "lime", "Import": "indigo",
            }},
            "entities": {"unit": {}},
        }
        picker, _ = _make_picker(tk_root, tmp_path, data=data)
        dispatch = picker._category_trees["dispatch"]
        assert _row_names(dispatch) == ["Import", "flowGroups", "Charge"]


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
#  Entity discovery helpers (input-DB fallback, used by on-open seeding)
# ---------------------------------------------------------------------------


def _mock_fetch(monkeypatch, per_class):
    """Patch ``fetch_entities_by_class`` to return a fixed per-class mapping
    (no real DB) so the on-open DB-fallback merge runs headlessly."""
    import flextool.scenario_comparison.input_entity_colors as iec

    monkeypatch.setattr(
        iec, "fetch_entities_by_class", lambda _url: dict(per_class),
    )


class TestPickerEntityDiscovery:
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
    # Dispatch order/ylim caches that _apply_color_settings must invalidate.
    stub._dispatch_ylims = {}
    stub._dispatch_columns = {}
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

    def test_clears_dispatch_order_cache(self, tk_root, tmp_path):
        """Apply must invalidate the dispatch column/ylim caches so a
        dispatch re-render re-derives its stacking order from the edited
        template (else colors update but order stays frozen until reopen)."""
        project = tmp_path / "proj"
        project.mkdir()
        (project / "plot_settings.yaml").write_text(
            "entities:\n  node:\n    n1: '#abcdef'\n", encoding="utf-8",
        )

        stub = _make_stub_viewer(project, live_plan=None)
        # Prime the caches as a prior dispatch render would.
        stub._dispatch_columns = {"elec": ["a", "b"]}
        stub._dispatch_ylims = {"elec": (0.0, 1.0)}

        stub._apply_color_settings()

        # Both order-bearing caches are cleared in lockstep.
        assert stub._dispatch_columns == {}
        assert stub._dispatch_ylims == {}


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

    def test_grab_not_viewable_does_not_abort_construction(
        self, tk_root, monkeypatch,
    ):
        """A ``grab_set`` that fails because the window is not yet viewable
        must NOT propagate out of ``__init__`` — otherwise the whole color
        edit aborts and nothing is saved (the real-WM crash).  The grab is
        retried on the event loop instead."""
        from flextool.gui.dialogs.plot_settings_picker import ColorPickerDialog

        calls = {"n": 0}
        orig = tk.Toplevel.grab_set

        def flaky(self):
            calls["n"] += 1
            if calls["n"] == 1:
                raise tk.TclError("grab failed: window not viewable")
            return orig(self)

        monkeypatch.setattr(tk.Toplevel, "grab_set", flaky)
        # Must construct cleanly despite the first grab_set raising.
        dlg = ColorPickerDialog(tk_root, "coal", "#212121", "#212121", True)
        assert dlg.winfo_exists()
        assert calls["n"] >= 1
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
        be created Undo → Redo → Cancel → Close (no Apply, no Refresh)."""
        picker, _ = _make_picker(tk_root, tmp_path)
        texts = [str(b.cget("text")) for b in _iter_buttons(picker)]
        assert texts == ["Undo", "Redo", "Cancel", "Close"]


class TestPickerFocusAndHint:
    def test_focus_in_activates_first_row_when_none(
        self, tk_root, tmp_path,
    ):
        picker, _ = _make_picker(tk_root, tmp_path)
        titles = _tab_titles(picker)
        unit = _tree_in_tab(picker, titles.index("unit"))
        # Nothing active yet (never clicked / never tabbed in).
        assert unit.focus() == ""
        picker._on_tree_focus_in(types.SimpleNamespace(widget=unit))
        first = unit.get_children("")[0]
        assert unit.focus() == first
        assert first in unit.selection()

    def test_focus_in_keeps_last_active_row(self, tk_root, tmp_path):
        picker, _ = _make_picker(tk_root, tmp_path)
        titles = _tab_titles(picker)
        unit = _tree_in_tab(picker, titles.index("unit"))
        last = unit.get_children("")[1]  # second row was active last
        unit.focus(last)
        picker._on_tree_focus_in(types.SimpleNamespace(widget=unit))
        # The previously-active row is reused, not reset to the first.
        assert unit.focus() == last
        assert last in unit.selection()

    def test_hint_label_is_two_rows(self, tk_root, tmp_path):
        picker, _ = _make_picker(tk_root, tmp_path)
        labels = [
            w for w in picker.winfo_children() if isinstance(w, ttk.Label)
        ]
        assert labels
        assert "\n" in str(labels[-1].cget("text"))
        # The hint mentions live updates, not a Ctrl+Enter Apply.
        assert "apply live" in str(labels[-1].cget("text"))
        assert "Ctrl+Enter" not in str(labels[-1].cget("text"))
