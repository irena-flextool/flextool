"""The run-output log pane of the execution window: colouring, wrapping, warning-code links and folded rows.

The pure helpers (line patterns, :func:`warning_docs_url`, :func:`similar_row_key`, :func:`find_similar_runs`,
:func:`wrap_mode_for`) need no Tk; :class:`LogView` drives one ``tk.Text``.  The GUI does not import the engine.
"""
from __future__ import annotations

import re
import tkinter as tk
import webbrowser
from collections.abc import Callable, Hashable, Iterable, Sequence
from tkinter import font as tkfont

#: A line that opens a solver's log block, which the run output greys until the next phase row: HiGHS' banner and
#: the line FlexTool prints before a commercial solver's log (``Running <Solver> (<program>)``; the names are the
#: engine's ``_subprocess_solve._DISPLAY_NAMES``, pinned by a test).
SOLVER_BLOCK_STARTS = ("Running HiGHS", "Running Gurobi", "Running CPLEX", "Running Xpress", "Running COPT")

#: Base URL of the user documentation (``flextool/docs_build/parameters.py`` uses the same site).
DOCS_BASE_URL = "https://irena-flextool.github.io/flextool/"
#: The warnings reference page.  TODO(execution-log P2/P3): point this at the warning registry's
#: ``WARNINGS_DOCS_URL`` once the registry exists (kept as a literal here so the GUI does not import the engine).
WARNINGS_DOCS_URL = DOCS_BASE_URL + "reference/warnings/"

#: A warning code at the end of a FlexTool message, e.g. ``[W-0123]``.
WARNING_CODE_RE = re.compile(r"\[W-\d{4}\]")

#: A phase-progress data row: ``<label> <time>s (<±section>) | <memory cell> | ...`` (the time cell is seconds to
#: 3 significant figures, the memory cell ``<value> <kB|MB|GB|TB> (<±delta>)`` or ``-``).
_TIMING_ROW_RE = re.compile(r"\S \s*\d[\d.]*s \((?:[+-][\d.]+|-)\) \| +(?:\d[\d.]* [kMGT]B \([^)]*\)|-)(?: |$)")
#: The phase-progress header row.
_TIMING_HEADER_RE = re.compile(r"^\s+time \| +RSS memory \| ")

#: Inside a solver block only a line *starting* with a warning / error is flagged.
_SOLVER_FLAG_RE = re.compile(r"\s*(warning|error)\b", re.IGNORECASE)
#: Outside it a whole word is required ("errors" / "warnings" / "ValueError" do not count).
_WORD_RE = re.compile(r"\b(warning|error)\b", re.IGNORECASE)
#: A Python exception line of a traceback ("ValueError: ...", "spinedb_api.exception.NothingToCommit: ...").
_EXCEPTION_RE = re.compile(r"[A-Za-z_][\w.]*(?:Error|Exception)(?::|$)")

#: The solve-summary block, whose similar rows fold.
SUMMARY_START = "--- Solve summary ---"
SUMMARY_END = "--- End of solve summary ---"

#: A run of at least this many similar rows folds.
MIN_FOLD_RUN = 3
#: The log wraps at the window width, but never narrower than this many columns (narrower: horizontal scroll).
MIN_WRAP_COLUMNS = 100
#: Continuation lines of a wrapped log line are indented this many characters.
WRAP_INDENT_CHARS = 2


def warning_docs_url(code: str) -> str:
    """The docs URL explaining warning *code* (``"W-0123"`` or ``"[W-0123]"``): the page anchor is the lower-case code."""
    return f"{WARNINGS_DOCS_URL}#{code.strip('[]').lower()}"


def is_timing_row(line: str) -> bool:
    """Whether *line* is a phase-progress (timing / memory) row or its header."""
    return bool(_TIMING_HEADER_RE.match(line) or _TIMING_ROW_RE.search(line))


def similar_row_key(line: str) -> tuple[str, str] | None:
    """The first two comma-separated fields of *line* (stripped), or ``None`` when it has fewer than three fields."""
    fields = line.split(",")
    if len(fields) < 3:
        return None
    first, second = fields[0].strip(), fields[1].strip()
    if not first:
        return None
    return first, second


def find_similar_runs(lines: Sequence[str], key: Callable[[str], Hashable | None] = similar_row_key,
                      min_run: int = MIN_FOLD_RUN) -> list[tuple[int, int]]:
    """``(start, length)`` of every run of at least *min_run* consecutive *lines* sharing a non-``None`` *key*."""
    runs: list[tuple[int, int]] = []
    start, prev = 0, None
    for i, line in enumerate([*lines, None]):
        k = key(line) if line is not None else None
        if k is not None and k == prev:
            continue
        if prev is not None and i - start >= min_run:
            runs.append((start, i - start))
        start, prev = i, k
    return runs


def wrap_mode_for(width_px: int, char_px: int, min_columns: int = MIN_WRAP_COLUMNS) -> str:
    """``"char"`` (wrap at the window) when *width_px* holds at least *min_columns* characters, else ``"none"``."""
    if char_px <= 0:
        return "char"
    return "char" if width_px // char_px >= min_columns else "none"


def classify_message(line: str) -> list[str]:
    """The colour tag of a FlexTool line outside a solver block: ``error`` / ``warn`` / ``note`` or none."""
    stripped = line.lstrip()
    if stripped.startswith(("Error:", "Traceback (most recent call last)")):
        return ["error"]
    if stripped.startswith("Warning:"):
        return ["warn"]
    if stripped.startswith("Note:"):
        return ["note"]
    if _EXCEPTION_RE.match(stripped):
        return ["error"]
    words = {m.group(1).lower() for m in _WORD_RE.finditer(line)}
    if "error" in words:
        return ["error"]
    if "warning" in words:
        return ["warn"]
    return []


def _blend(base16: tuple[int, int, int], target_hex: str, frac: float) -> str:
    from flextool.gui.ui_metrics import blend_rgb
    return blend_rgb(base16, target_hex, frac)


class LogView:
    """Appends run-output lines to one ``tk.Text`` with colour tags, wrap indent, warning-code links and folded
    similar rows in the solve summary.

    The text keeps the original lines: wrapping is display-only and the fold toggle lines are left out of
    :meth:`text` (what Copy puts on the clipboard).
    """

    FOLD_TOGGLE = "fold_toggle"

    def __init__(self, text: tk.Text, open_url: Callable[[str], object] = webbrowser.open) -> None:
        self.text = text
        self._open_url = open_url
        self.highs_block_active = False
        self._summary_active = False
        self._run_key: Hashable | None = None
        self._run_mark: str | None = None
        self._run_count = 0
        self._fold_seq = 0
        text.configure(wrap="char")
        text.bind("<Configure>", self._on_configure, add="+")
        text.bind("<<Copy>>", self._on_copy_event)
        text.tag_bind("wlink", "<Button-1>", self._on_link_click)
        text.tag_bind(self.FOLD_TOGGLE, "<Button-1>", self._on_toggle_click)
        for tag in ("wlink", self.FOLD_TOGGLE):
            text.tag_bind(tag, "<Enter>", lambda _e: text.configure(cursor="hand2"))
            text.tag_bind(tag, "<Leave>", lambda _e: text.configure(cursor="xterm"))

    # ── tags / geometry ───────────────────────────────────────────────

    def _char_px(self) -> int:
        try:
            return max(1, tkfont.Font(font=self.text.cget("font")).measure("0"))
        except tk.TclError:
            return 8

    def configure_tags(self) -> None:
        """Configure the colour tags from the widget's live theme colours so they read in dark and light modes.

        Backgrounds are subtle tints of the real base background; foregrounds are pulled partway toward a saturated
        hue so they stand out on either base.
        """
        txt = self.text
        txt.tag_configure("wrapindent", lmargin2=WRAP_INDENT_CHARS * self._char_px())
        try:
            base_bg = txt.winfo_rgb(txt.cget("background"))
            base_fg = txt.winfo_rgb(txt.cget("foreground"))
        except tk.TclError:
            return  # colours unavailable (widget not realised) — stay plain
        txt.tag_configure("cmd", background=_blend(base_bg, "#ffd000", 0.16))
        txt.tag_configure("highs", background=_blend(base_bg, "#808080", 0.14))
        txt.tag_configure("solvestart", background=_blend(base_bg, "#33cc33", 0.18),
                          foreground=_blend(base_fg, "#4aa3ff", 0.55))
        txt.tag_configure("mem", foreground=_blend(base_fg, "#4aa3ff", 0.55))
        txt.tag_configure("note", foreground=_blend(base_fg, "#808080", 0.40))
        txt.tag_configure("warn", foreground=_blend(base_fg, "#ff9500", 0.70))
        txt.tag_configure("solverwarn", foreground=_blend(base_fg, "#ff9500", 0.60))
        txt.tag_configure("error", foreground=_blend(base_fg, "#ff3b30", 0.70))
        txt.tag_configure("wlink", foreground=_blend(base_fg, "#4aa3ff", 0.70), underline=True)
        txt.tag_configure(self.FOLD_TOGGLE, foreground=_blend(base_fg, "#808080", 0.40), underline=True)
        # Keep the selection highlight on top of every custom tag (they would otherwise paint over it).
        txt.tag_raise("sel")

    def _on_configure(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        self.apply_wrap(int(event.width))

    def apply_wrap(self, width_px: int) -> str:
        """Wrap at the visible width when it holds :data:`MIN_WRAP_COLUMNS` characters, else scroll horizontally."""
        txt = self.text
        try:
            inner = width_px - 2 * sum(int(str(txt.cget(opt))) for opt in
                                       ("borderwidth", "highlightthickness", "padx"))
        except (tk.TclError, ValueError):
            inner = width_px
        char_px = self._char_px()
        mode = wrap_mode_for(inner, char_px)
        if str(txt.cget("wrap")) != mode:
            txt.configure(wrap=mode)
        txt.tag_configure("wrapindent", lmargin2=WRAP_INDENT_CHARS * char_px)
        return mode

    # ── classification ────────────────────────────────────────────────

    def classify(self, element: str) -> list[str]:
        """The colour tags of one log *element* (one ``stdout_lines`` entry — usually one line; the command echo is
        one multi-line element).

        A small state machine tracks the solver-output block (:data:`SOLVER_BLOCK_STARTS`), which greys until the
        next phase-progress row; inside it only lines starting with WARNING / ERROR are flagged (amber / red).
        """
        if "\\\n" in element or element.startswith("systemd-run"):
            self.highs_block_active = False
            return ["cmd"]
        if is_timing_row(element):
            self.highs_block_active = False
            return ["mem"]
        if element.startswith("Solve start:"):
            self.highs_block_active = False
            return ["solvestart"]
        if element.startswith(SOLVER_BLOCK_STARTS):
            self.highs_block_active = True
        if not self.highs_block_active:
            return classify_message(element)
        # Whole block greys, blank lines included, so the solver dump reads as one contiguous panel.
        tags = ["highs"]
        m = _SOLVER_FLAG_RE.match(element)
        if m:
            tags.append("error" if m.group(1).lower() == "error" else "solverwarn")
        return tags

    # ── appending ─────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear the text and restart the state machines (the log is re-rendered from scratch)."""
        self.text.delete("1.0", "end")
        self.highs_block_active = False
        self._summary_active = False
        self._clear_run()

    def append(self, lines: Iterable[str]) -> None:
        """Append *lines* (raw ``stdout_lines`` entries) to the end of the text."""
        txt = self.text
        for line in lines:
            was_active = self.highs_block_active
            tags = self.classify(line)
            now_active = self.highs_block_active
            self._track_summary(line)
            if "highs" in tags and not line.strip():
                # In-block blank: grey it provisionally, but mark it pending — a blank that turns out to be
                # *trailing* (the separator before the timer table) reverts to regular once the block ends.
                txt.insert("end", line + "\n", ("highs", "pending_blank", "wrapindent"))
                continue
            self._insert(line, tags)
            if "highs" in tags:
                # Real dump content after the blanks → they were internal; keep them grey.
                txt.tag_remove("pending_blank", "1.0", "end")
            elif was_active and not now_active:
                # This row closed the solver block → any pending blanks were trailing; un-grey them.
                ranges = txt.tag_ranges("pending_blank")
                for k in range(0, len(ranges), 2):
                    txt.tag_remove("highs", ranges[k], ranges[k + 1])
                txt.tag_remove("pending_blank", "1.0", "end")

    def _insert(self, line: str, tags: Sequence[str]) -> None:
        txt = self.text
        start = txt.index("end-1c")
        txt.insert("end", line + "\n", (*tags, "wrapindent"))
        for m in WARNING_CODE_RE.finditer(line):
            txt.tag_add("wlink", f"{start}+{m.start()}c", f"{start}+{m.end()}c")

    # ── folding similar rows of the solve summary ─────────────────────

    def _clear_run(self) -> None:
        if self._run_mark is not None:
            try:
                self.text.mark_unset(self._run_mark)
            except tk.TclError:
                pass
        self._run_key, self._run_mark, self._run_count = None, None, 0

    def _track_summary(self, line: str) -> None:
        """Called before *line* is inserted: opens / closes the summary block and the current run of similar rows."""
        stripped = line.strip()
        if stripped == SUMMARY_START:
            self._close_run()
            self._summary_active = True
            return
        if not self._summary_active:
            return
        if stripped == SUMMARY_END:
            self._close_run()
            self._summary_active = False
            return
        key = similar_row_key(line)
        if key is not None and key == self._run_key:
            self._run_count += 1
            return
        self._close_run()
        if key is not None:
            self._fold_seq += 1
            self._run_key, self._run_count = key, 1
            self._run_mark = f"foldstart{self._fold_seq}"
            self.text.mark_set(self._run_mark, "end-1c")
            self.text.mark_gravity(self._run_mark, "left")

    def _close_run(self) -> None:
        """Fold the current run (all but its first row) when it has :data:`MIN_FOLD_RUN` rows or more."""
        if self._run_mark is not None and self._run_count >= MIN_FOLD_RUN:
            self._fold(self._run_mark, self._run_count, self._fold_seq)
        self._clear_run()

    def _fold(self, start_mark: str, count: int, seq: int) -> None:
        txt = self.text
        hidden_start = txt.index(f"{start_mark} lineend +1c")
        hidden_end = txt.index("end-1c")
        body = f"fold{seq}"
        txt.tag_configure(body, elide=True)
        txt.tag_add(body, hidden_start, hidden_end)
        txt.insert(hidden_start, self._toggle_text(count - 1, shown=False),
                   (self.FOLD_TOGGLE, f"foldtoggle{seq}", "wrapindent"))

    @staticmethod
    def _toggle_text(hidden: int, *, shown: bool) -> str:
        arrow, action = ("▾", "hide") if shown else ("▸", "show")
        return f"  {arrow} {hidden} more similar rows ({action})\n"

    def _on_toggle_click(self, _event: tk.Event | None = None) -> str:  # type: ignore[type-arg]
        names = self.text.tag_names("current")
        for name in names:
            if name.startswith("foldtoggle"):
                self.toggle_fold(int(name[len("foldtoggle"):]))
                break
        return "break"

    def toggle_fold(self, seq: int) -> None:
        """Show or hide the rows of fold *seq*."""
        txt = self.text
        body, toggle = f"fold{seq}", f"foldtoggle{seq}"
        ranges = txt.tag_ranges(toggle)
        if not ranges:
            return
        shown = bool(txt.tk.getboolean(txt.tag_cget(body, "elide") or 0))   # currently hidden → show it
        txt.tag_configure(body, elide=not shown)
        old = txt.get(ranges[0], ranges[1])
        hidden = int(re.search(r"(\d+) more", old).group(1))  # type: ignore[union-attr]
        start = txt.index(ranges[0])
        txt.delete(ranges[0], ranges[1])
        txt.insert(start, self._toggle_text(hidden, shown=shown), (self.FOLD_TOGGLE, toggle, "wrapindent"))

    # ── links and copy ────────────────────────────────────────────────

    def _on_link_click(self, _event: tk.Event | None = None) -> str:  # type: ignore[type-arg]
        rng = self.text.tag_prevrange("wlink", "current+1c")
        if rng:
            self._open_url(warning_docs_url(self.text.get(*rng)))
        return "break"

    def text_between(self, start: str, end: str) -> str:
        """The text from *start* to *end* without the fold toggle lines (hidden rows are included)."""
        txt = self.text
        start, end = txt.index(start), txt.index(end)
        parts: list[str] = []
        pos = start
        ranges = txt.tag_ranges(self.FOLD_TOGGLE)
        for k in range(0, len(ranges), 2):
            t0, t1 = txt.index(ranges[k]), txt.index(ranges[k + 1])
            if txt.compare(t1, "<=", pos) or txt.compare(t0, ">=", end):
                continue
            if txt.compare(t0, ">", pos):
                parts.append(txt.get(pos, t0))
            pos = t1 if txt.compare(t1, "<", end) else end
        if txt.compare(pos, "<", end):
            parts.append(txt.get(pos, end))
        return "".join(parts)

    def selected_or_all_text(self) -> str:
        """The selection, or the whole log when nothing is selected, as original lines."""
        try:
            return self.text_between("sel.first", "sel.last")
        except tk.TclError:
            return self.text_between("1.0", "end-1c")

    def _on_copy_event(self, _event: tk.Event | None = None) -> str:  # type: ignore[type-arg]
        """Ctrl+C / <<Copy>>: copy the selection as original lines (nothing selected: nothing copied)."""
        try:
            text = self.text_between("sel.first", "sel.last")
        except tk.TclError:
            return "break"
        if text:
            self.text.clipboard_clear()
            self.text.clipboard_append(text)
        return "break"
