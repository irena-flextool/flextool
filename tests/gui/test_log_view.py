"""The run-output log pane (flextool/gui/log_view.py): line classes, wrap mode, warning-code links, folded rows.

The pure helpers run without Tk; the widget tests need a display (they are marked ``gui``: run them with
``env -u DISPLAY xvfb-run -a python -m pytest tests/gui/test_log_view.py --run-gui``).
"""
from __future__ import annotations

import tkinter as tk
from collections.abc import Iterator
from types import SimpleNamespace

import pytest

from flextool.gui import log_view as lv
from flextool.gui.log_view import LogView

TIMING_HEADER = "                                      time |      RSS memory |   system memory |     system swap"
TIMING_ROW = "Matrix built by polar-high 10000s (+10000) |     544 MB (-5) | 16.9 GB (-0.01) |    120 MB (+10)"


# ── pure helpers ──────────────────────────────────────────────────────────


def test_timing_rows_and_header_are_recognised_by_the_new_format() -> None:
    assert lv.is_timing_row(TIMING_HEADER)
    assert lv.is_timing_row(TIMING_ROW)
    assert lv.is_timing_row("Run start                           0s (-) |   174 MB (+174) | 18.1 GB (+18.1) |"
                            "       1 MB (+1)")
    assert not lv.is_timing_row("Solver  |  1.2 GB  |  0:00:03")          # the old two-space format
    assert not lv.is_timing_row("node | value | 3 MB")


@pytest.mark.parametrize(("line", "tags"), [
    ("Error: the model has no solves", ["error"]),
    ("Traceback (most recent call last):", ["error"]),
    ("ValueError: bad value", ["error"]),
    ("Warning: flowGroup 'g' has no flows (3 cases, e.g. 'g1') [W-0123]", ["warn"]),
    ("Note: results are usable", ["note"]),
    ("WARNING: from a library", ["warn"]),
    ("the solve had an error in it", ["error"]),
    ("Warnings: 2 kinds, 5 cases", []),                    # the tally line is not itself a warning
    ("errors=0 warnings_seen", []),                         # no whole word
    ("ErrorHandler set up", []),
    ("plain line", []),
])
def test_lines_outside_the_solver_block(line: str, tags: list[str]) -> None:
    assert LogView.classify(SimpleNamespace(highs_block_active=False), line) == tags  # type: ignore[arg-type]


def test_warning_docs_url_uses_the_lower_case_code_as_anchor() -> None:
    assert lv.warning_docs_url("[W-0123]") == lv.WARNINGS_DOCS_URL + "#w-0123"
    assert lv.warning_docs_url("W-0007").endswith("/reference/warnings/#w-0007")
    assert lv.WARNING_CODE_RE.findall("Warning: x [W-0001] and [W-12]") == ["[W-0001]"]


def test_wrap_mode_wraps_only_when_100_columns_fit() -> None:
    assert lv.wrap_mode_for(800, 8) == "char"
    assert lv.wrap_mode_for(799, 8) == "none"
    assert lv.wrap_mode_for(10, 0) == "char"


def test_similar_rows_share_the_row_label() -> None:
    """The fold key is the first comma-separated field: the slack rows of a real summary (``Created, <node>,
    <period>, <value>``) differ from the second field on, and fold as one run per slack type."""
    assert lv.similar_row_key("Created , n1, p2030, 4") == "Created"
    assert lv.similar_row_key("Created, n2, p2035, 0.5") == "Created"
    assert lv.similar_row_key("a, b") is None and lv.similar_row_key(", b, c") is None
    assert lv.similar_row_key("no commas") is None


# ── the widget ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def tk_root() -> Iterator[tk.Tk]:
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("No display available")
    root.withdraw()
    yield root
    try:
        root.destroy()
    except tk.TclError:
        pass


@pytest.fixture
def view(tk_root: tk.Tk) -> Iterator[tuple[LogView, list[str]]]:
    opened: list[str] = []
    text = tk.Text(tk_root, font="TkFixedFont")
    view = LogView(text, open_url=opened.append)
    view.configure_tags()
    yield view, opened
    text.destroy()


def _displayed(txt: tk.Text) -> str:
    """The text as shown (elided rows left out)."""
    return str(txt.tk.call(txt._w, "get", "-displaychars", "1.0", "end-1c"))


SUMMARY = [
    "--- Solve summary ---",
    '"Solve","Objective","Total cost from solver, includes all penalty costs"',
    "Created, n1, p2030, 1",
    "Created, n2, p2030, 2",
    "Created, n2, p2035, 3",
    "Created, n3, p2030, 4",
    "Removed, n1, p2030, 12",
    "--- End of solve summary ---",
    "after",
]


@pytest.mark.gui
def test_summary_runs_fold_and_unfold_and_copy_keeps_the_original_lines(view) -> None:
    v, _ = view
    v.append(["before"])
    v.append(SUMMARY[:4])              # a run may be split over two polls
    v.append(SUMMARY[4:])
    txt = v.text
    toggles = txt.tag_ranges(LogView.FOLD_TOGGLE)
    assert len(toggles) == 2
    assert txt.get(*toggles) == "  ▸ 3 more similar rows (show)\n"
    shown = _displayed(txt)
    assert "Created, n1, p2030, 1\n  ▸ 3 more similar rows (show)\nRemoved, n1, p2030, 12" in shown
    assert "n2" not in shown
    # Copy yields the original lines: hidden rows included, toggle left out.
    assert v.selected_or_all_text() == "\n".join(["before", *SUMMARY]) + "\n"
    seq = int(next(n for n in txt.tag_names(toggles[0]) if n.startswith("foldtoggle"))[len("foldtoggle"):])
    v.toggle_fold(seq)
    shown = _displayed(txt)
    assert "▾ 3 more similar rows (hide)" in shown and "Created, n3, p2030, 4" in shown
    v.toggle_fold(seq)
    assert "n3" not in _displayed(txt)
    # Only the summary block folds.
    v.append([f"Created, x{i}, p1, 1" for i in range(5)])
    assert len(txt.tag_ranges(LogView.FOLD_TOGGLE)) == 2


@pytest.mark.gui
def test_selection_copy_skips_a_toggle_line(view, tk_root) -> None:
    v, _ = view
    v.append(SUMMARY)
    txt = v.text
    txt.tag_add("sel", "2.0", "end-1c")
    assert v.text_between("sel.first", "sel.last") == "\n".join(SUMMARY[1:]) + "\n"


@pytest.mark.gui
def test_warning_codes_are_links_and_lines_get_the_wrap_indent(view) -> None:
    v, opened = view
    v.append(["Warning: flowGroup has no flows (2 cases, e.g. 'g1') [W-0042]"])
    txt = v.text
    rng = txt.tag_ranges("wlink")
    assert txt.get(*rng) == "[W-0042]"
    assert "warn" in txt.tag_names("1.0") and "wrapindent" in txt.tag_names("1.0")
    txt.mark_set("current", rng[0])
    v._on_link_click()
    assert opened == [lv.WARNINGS_DOCS_URL + "#w-0042"]
    assert int(str(txt.tag_cget("wrapindent", "lmargin2"))) == 2 * v._char_px()


@pytest.mark.gui
def test_narrow_pane_scrolls_instead_of_wrapping(view) -> None:
    v, _ = view
    char = v._char_px()
    assert v.apply_wrap(120 * char) == "char"
    assert str(v.text.cget("wrap")) == "char"
    assert v.apply_wrap(60 * char) == "none"
    assert str(v.text.cget("wrap")) == "none"


@pytest.mark.gui
def test_reset_restarts_the_solver_block_and_the_summary(view) -> None:
    v, _ = view
    v.append(["Running HiGHS 1.9", "--- Solve summary ---", "Slack, a, 1"])
    v.reset()
    assert v.text.get("1.0", "end-1c") == "" and not v.highs_block_active
    v.append(["Slack, a, 1", "Slack, a, 2", "Slack, a, 3", "x"])
    assert not v.text.tag_ranges(LogView.FOLD_TOGGLE)
