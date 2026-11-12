"""The transcript of one chat (P2 §4.5): a read-only ``tk.Text`` that streams the answer and shows each tool call
as one line that opens to show what was sent.

Tk thread only.  The answer arrives in pieces (:meth:`Transcript.stream`): the raw text of the line still
arriving is shown as it comes, and each completed line is replaced by its Markdown rendering
(:class:`~flextool.gui.assistant.markdown_stream.MarkdownStream`), so a long answer costs one short delete and
insert per batch.  The view follows the end only while it is at the end.

The widget takes no typing: keys that change text are dropped (copy, select all and moving around stay), Tab and
Shift+Tab move the focus on, Escape stops the answer, Enter or space on a tool line opens or closes it, and a click
on a tool line does the same.  A question sent has two small buttons after it (item 220): ↻ asks it again and ✎
puts it in the input box to change (the window's ``on_question(action, n)`` with ``"again"`` or ``"edit"`` and
the question's number); R and E on a question line and its right-click menu do the same.  The buttons are not Tab
stops and not part of copied text; a question merged into the next one has them on the last line only.  It draws no insertion cursor (only the input box has one): while it has the focus,
the tool or context line the keyboard is on is shaded instead, so Enter and space still show what they open.  Its
tags take their colours from the widget's own theme colours.  On macOS Command+C copies and Command+A selects all.

A long chat stays cheap to lay out (review G19, C7-m10): every tool and context line shares a few tags (``tool`` or
``context`` with the click bindings once, ``tooldetail``, and one elided ``detail_closed`` on each closed detail), and
each line is found by one mark at its start (``tool_{n}``, ``context_{n}``; a question ``user_{n}``).  The
question buttons are embedded windows after ``user_{n}_end``, destroyed by :meth:`Transcript.clear`.

Fonts (owner decisions 2026-11-08, :mod:`flextool.gui.assistant.fonts`): the text is in the chat's base font
(FlexTool's body text font one point larger; a ``tk.Text`` would otherwise use the fixed-width font, and bold text
in the proportional font then looked smaller).  Bold, italic and the headings are its family, code, tables and the
details of a tool call the fixed-width family at its size.  :func:`sync_fonts` sets them from the body font again —
at every answer and on a theme change — so a changed font size reaches them (a font is configured only when a
value differs).
"""
from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import ttk
from typing import TYPE_CHECKING

from flextool.assistant.chat_cli import args_text
from flextool.gui.assistant import strings
from flextool.gui.assistant.fonts import CHAT_FONT, sync_fonts
from flextool.gui.assistant.markdown_stream import Line, MarkdownStream
from flextool.gui.assistant.markdown_stream import render as render_markdown
from flextool.gui.assistant.notices import is_aqua, question_symbols

if TYPE_CHECKING:
    from flextool.assistant.llm_adapter import Message

#: Keys that move or select without changing the text (``execution_window.py``'s read-only filter).
NAV_KEYSYMS = frozenset({"Left", "Right", "Up", "Down", "Home", "End", "Prior", "Next", "Shift_L", "Shift_R",
                         "Control_L", "Control_R", "Alt_L", "Alt_R", "Meta_L", "Meta_R", "Super_L", "Super_R"})
#: The tag shading the tool or context line the keyboard is on (the transcript draws no insertion cursor).
KEYBOARD_LINE = "keyboard_line"
#: The shared elided tag on the detail of a tool or context line while it is closed.
DETAIL_CLOSED = "detail_closed"
#: The prefixes of the marks at the start of a tool or context line (the keys of :meth:`Transcript.toggle`).
LINE_PREFIXES = ("tool_", "context_")
#: The style of a question's ↻ and ✎ buttons: a Toolbutton on the transcript's own background.
QUESTION_BUTTON_STYLE = "AssistantQuestion.Toolbutton"
#: What a question's buttons, keys and menu ask the window for.
AGAIN, EDIT = "again", "edit"
TOOL_OPEN = "▾"
TOOL_CLOSED = "▸"
DETAIL_INDENT = "    "


def _shortcut_modifiers(widget: tk.Misc) -> int:
    """The ``event.state`` bits of a key held with R or E that make it not the question's key (it is a shortcut or
    types another character).  Tk reports the modifiers differently per windowing system:

    * Control is 0x4 everywhere;
    * X11: Mod1 (0x8) is Alt; Mod2 (0x10) is usually Num Lock, so it is not counted;
    * aqua (macOS): Mod1 (0x8) is Command, Mod2 (0x10) is Option (Option+R types "®");
    * win32: 0x8 is Num Lock (not counted), Alt is 0x20000.
    """
    try:
        system = str(widget.tk.call("tk", "windowingsystem"))
    except tk.TclError:
        system = "x11"
    if system == "aqua":
        return 0x4 | 0x8 | 0x10
    if system == "win32":
        return 0x4 | 0x20000
    return 0x4 | 0x8


class Transcript(ttk.Frame):
    """The transcript widget.  *on_escape* stops the answer; *bind_commands(widget)* adds the window's tab
    commands to the text widget's own bindings; *on_question(action, n)* asks question *n* again (``"again"``) or
    puts it in the input box (``"edit"``) — without it a question gets no buttons; from the R and E keys it is
    called with ``key=True`` (R asks first, review decision D8)."""

    def __init__(self, parent: tk.Misc, *, on_escape: Callable[[], None] | None = None,
                 bind_commands: Callable[[tk.Misc], None] | None = None,
                 on_question: Callable[..., None] | None = None) -> None:
        super().__init__(parent)
        # No insertion cursor: zero width, and no blink timer (insertofftime 0).  The insert mark still moves
        # with the arrow keys, for selecting with Shift and for Enter / space on a tool line.
        self.text = tk.Text(self, wrap="word", height=10, undo=False, borderwidth=0, highlightthickness=1,
                            padx=6, pady=4, cursor="arrow", insertwidth=0, insertofftime=0,
                            insertborderwidth=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.text.yview, takefocus=0)
        self.text.configure(yscrollcommand=self.scrollbar.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self._on_escape = on_escape
        self._on_question = on_question
        self._turns: dict[int, int] = {}                    # a question line's id → the question's number
        self._buttons: dict[int, list[ttk.Button]] = {}     # the lines that have the ↻ ✎ buttons
        self._question_menu: tk.Menu | None = None
        self._md = MarkdownStream()
        self._held = ""                 # the answer's last line held back: it may be a leaked "Keep:" line
        self._answer_open = False
        self._in_turn = False
        self._tool_count = 0
        self._user_count = 0
        self._tool_headers: dict[int, str] = {}
        self._example = False
        self._aqua = is_aqua(self.text)
        self._shortcut_modifiers = _shortcut_modifiers(self.text)
        self._configure_tags()
        self._bind_lines()
        self._bind_keys(bind_commands)
        # A new theme: the colours and the fonts again (a changed body font size reaches the derived fonts).
        self.text.bind("<<ThemeChanged>>", lambda _e: self._configure_tags(), add="+")

    # --- appearance -----------------------------------------------------------------------------------------
    def _configure_tags(self) -> None:
        from flextool.gui.ui_metrics import blend_rgb, rgb_hex

        txt = self.text
        try:
            bg = txt.winfo_rgb(txt.cget("background"))
            fg = txt.winfo_rgb(txt.cget("foreground"))
        except tk.TclError:
            return
        fonts = sync_fonts(txt)
        txt.configure(font=CHAT_FONT)
        bold = fonts["bold"]
        dim = blend_rgb(fg, rgb_hex(bg), 0.45)
        txt.tag_configure("speaker", font=bold)
        txt.tag_configure("user", spacing1=4)
        txt.tag_configure("user_pending", foreground=dim, spacing1=4)
        txt.tag_configure("assistant", spacing1=2)
        txt.tag_configure("tool", foreground=blend_rgb(fg, "#4aa3ff", 0.45))
        txt.tag_configure("tooldetail", font=fonts["code"], lmargin1=24, lmargin2=24, foreground=dim)
        txt.tag_configure("notice", foreground=dim)
        txt.tag_configure("usage", foreground=dim)
        txt.tag_configure("error", foreground=blend_rgb(fg, "#ff3b30", 0.70))
        txt.tag_configure("example", foreground=dim)
        txt.tag_configure("context", foreground=dim)
        txt.tag_configure("md_bold", font=bold)
        txt.tag_configure("md_italic", font=fonts["italic"])
        txt.tag_configure("md_code", font=fonts["code"], background=blend_rgb(bg, "#808080", 0.14))
        txt.tag_configure("md_codeblock", font=fonts["code"], background=blend_rgb(bg, "#808080", 0.10),
                          lmargin1=12, lmargin2=12)
        txt.tag_configure("md_table", font=fonts["code"])
        txt.tag_configure("md_h1", font=fonts["h1"], spacing1=6)
        txt.tag_configure("md_h2", font=fonts["h2"], spacing1=4)
        txt.tag_configure("md_h3", font=bold, spacing1=2)
        txt.tag_configure("md_list", lmargin1=8, lmargin2=20)
        txt.tag_configure(KEYBOARD_LINE, background=blend_rgb(bg, "#4aa3ff", 0.16))
        txt.tag_configure(DETAIL_CLOSED, elide=True)
        txt.tag_raise("sel")
        try:
            ttk.Style(txt).configure(QUESTION_BUTTON_STYLE, background=rgb_hex(bg), padding=(2, 0))
        except tk.TclError:
            pass

    # --- keys -----------------------------------------------------------------------------------------------
    def _bind_keys(self, bind_commands: Callable[[tk.Misc], None] | None) -> None:
        txt = self.text
        txt.bind("<Key>", self._on_key)
        txt.bind("<Tab>", lambda _e: self._focus(next_=True))
        txt.bind("<Shift-Tab>", lambda _e: self._focus(next_=False))
        try:
            txt.bind("<ISO_Left_Tab>", lambda _e: self._focus(next_=False))
        except tk.TclError:
            pass
        txt.bind("<Escape>", self._escape)
        txt.bind("<Return>", self._toggle_at_insert)
        txt.bind("<KP_Enter>", self._toggle_at_insert)
        txt.bind("<space>", self._toggle_at_insert)
        txt.bind("<Control-a>", self._select_all)
        txt.bind("<Control-A>", self._select_all)
        if self._aqua:
            txt.bind("<Command-a>", self._select_all)
        # After the class bindings have moved the insert mark (KeyRelease, ButtonRelease) and on focus changes.
        for sequence in ("<KeyRelease>", "<ButtonRelease-1>", "<FocusIn>", "<FocusOut>"):
            txt.bind(sequence, lambda _e: self.mark_keyboard_line(), add="+")
        for virtual in ("<<Paste>>", "<<Cut>>", "<<Clear>>", "<<PasteSelection>>", "<<Undo>>", "<<Redo>>"):
            txt.bind(virtual, lambda _e: "break")
        if not self._aqua:
            txt.bind("<Button-2>", lambda _e: "break")          # middle click pastes on X11 and Windows
        for sequence in ("<Button-2>", "<Control-Button-1>") if self._aqua else ("<Button-3>",):
            txt.bind(sequence, self._post_question_menu)
        if bind_commands is not None:
            bind_commands(txt)

    def _on_key(self, event: tk.Event) -> str | None:  # type: ignore[type-arg]
        # Control, or on macOS Command (Mod1 there; on X11 Mod1 is Alt): copy and select all reach the class binding.
        modifiers = 0x4 | 0x8 if self._aqua else 0x4
        if event.state & modifiers and event.keysym in ("c", "C", "a", "A", "Insert"):
            return None
        if event.keysym in NAV_KEYSYMS:
            return None
        if event.keysym in ("r", "R", "e", "E") and not event.state & self._shortcut_modifiers:
            line = self.question_line_at("insert")
            if line is not None:
                self._question_action(AGAIN if event.keysym in ("r", "R") else EDIT, line, key=True)
        return "break"

    def _focus(self, *, next_: bool) -> str:
        target = self.text.tk_focusNext() if next_ else self.text.tk_focusPrev()
        if target is not None:
            target.focus_set()
        return "break"

    def _escape(self, _event: tk.Event) -> str:  # type: ignore[type-arg]
        if self._on_escape is not None:
            self._on_escape()
        return "break"

    def _select_all(self, _event: tk.Event) -> str:  # type: ignore[type-arg]
        self.text.tag_add("sel", "1.0", "end-1c")
        return "break"

    def _toggle_at_insert(self, _event: tk.Event) -> str:  # type: ignore[type-arg]
        key = self._line_tag_at_insert()
        if key is not None:
            self.toggle(key)
        return "break"

    def _line_at(self, index: str) -> str | None:
        """The key of the tool or context line holding *index* (the mark at its line start), or None."""
        txt = self.text
        try:
            if not (set(txt.tag_names(f"{index} linestart")) & {"tool", "context"}):
                return None
            for kind, name, _where in txt.dump(f"{index} linestart", f"{index} linestart + 1 chars", mark=True):
                if kind == "mark" and name.startswith(LINE_PREFIXES):
                    return name
        except tk.TclError:
            pass
        return None

    def _line_tag_at_insert(self) -> str | None:
        return self._line_at("insert")

    def _bind_lines(self) -> None:
        """The click bindings of every tool and context line, once on their shared tags."""
        txt = self.text
        txt.tag_bind("tool", "<Button-1>", self._click_line)
        txt.tag_bind("tool", "<Enter>", lambda _e: txt.configure(cursor="hand2"))
        txt.tag_bind("tool", "<Leave>", lambda _e: txt.configure(cursor="arrow"))
        txt.tag_bind("context", "<Button-1>", self._click_line)

    def _click_line(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        key = self._line_at(f"@{event.x},{event.y}")
        if key is not None:
            self.toggle(key)

    def mark_keyboard_line(self) -> None:
        """Shade the tool, context or question line holding the insert mark while the transcript has the focus (on a
        question line R and E work)."""
        txt = self.text
        try:
            txt.tag_remove(KEYBOARD_LINE, "1.0", "end")
            if txt.focus_get() is not txt:
                return
            if self._line_tag_at_insert() is not None:
                txt.tag_add(KEYBOARD_LINE, "insert linestart", "insert lineend")
                return
            line = self.question_line_at("insert")
            if line is not None:
                txt.tag_add(KEYBOARD_LINE, f"user_{line}", f"user_{line}_end lineend")
        except (tk.TclError, KeyError):
            pass

    # --- the question buttons, keys and menu (item 220) ------------------------------------------------------------
    def set_turn(self, line: int, turn: int) -> None:
        """Question line *line* is question number *turn*: it gets the ↻ ✎ buttons, which an earlier line of the same
        question loses (a question that got no answer goes on with the next one under its number)."""
        if turn < 1 or not self._has_mark(f"user_{line}"):
            return
        self._turns[line] = turn
        for other, number in list(self._turns.items()):
            if other != line and number == turn:
                self._remove_buttons(other)
        if self._on_question is not None and line not in self._buttons:
            self._add_buttons(line)

    def turn_of(self, line: int) -> int | None:
        return self._turns.get(line)

    def line_of(self, turn: int) -> int | None:
        """The line of question *turn* that has its buttons (its last one), or None."""
        return next((line for line in self._buttons if self._turns.get(line) == turn), None)

    def has_buttons(self, line: int) -> bool:
        return line in self._buttons

    def question_buttons(self, line: int) -> list[ttk.Button]:
        """The ↻ and ✎ buttons of *line* (none when it has none)."""
        return list(self._buttons.get(line, []))

    def turn_text(self, turn: int) -> str:
        """Question *turn* as shown: the text of its lines (two when a question that got no answer went on with the
        next one), joined as the chat keeps them; "" when the transcript has no line of it."""
        lines = sorted(line for line, number in self._turns.items() if number == turn)
        return "\n\n".join(text for text in (self.user_text(line).strip() for line in lines) if text)

    def later_turns(self, turn: int) -> int:
        """How many questions after question *turn* the transcript shows as sent."""
        return len({number for number in self._turns.values() if number > turn})

    def question_line_at(self, index: str) -> int | None:
        """The question line (one with buttons) holding *index*, or None."""
        txt = self.text
        try:
            where = txt.index(index)
            for line in self._buttons:
                if (txt.compare(f"user_{line}", "<=", where)
                        and txt.compare(where, "<=", f"user_{line}_end lineend")):
                    return line
        except tk.TclError:
            pass
        return None

    def _question_action(self, action: str, line: int, *, key: bool = False) -> None:
        turn = self._turns.get(line)
        if turn is not None and self._on_question is not None:
            if key:
                self._on_question(action, turn, key=True)
            else:
                self._on_question(action, turn)

    def _add_buttons(self, line: int) -> None:
        from flextool.gui.hover_tooltip import attach_tooltip

        again, edit = question_symbols(self.text)
        made = []
        for offset, (label, action, tip) in enumerate(((again, AGAIN, strings.ASK_AGAIN_TOOLTIP),
                                                       (edit, EDIT, strings.EDIT_TOOLTIP))):
            button = ttk.Button(self.text, text=label, style=QUESTION_BUTTON_STYLE, takefocus=0, cursor="hand2",
                                command=lambda a=action, n=line: self._question_action(a, n))
            self.text.window_create(f"user_{line}_end + {offset} chars", window=button, padx=2)
            attach_tooltip(button, tip)
            made.append(button)
        self._buttons[line] = made

    def _remove_buttons(self, line: int) -> None:
        for button in self._buttons.pop(line, []):
            try:
                self.text.delete(self.text.index(button))
            except tk.TclError:
                pass
            try:
                button.destroy()
            except tk.TclError:
                pass

    def _post_question_menu(self, event: tk.Event) -> str | None:  # type: ignore[type-arg]
        line = self.question_line_at(f"@{event.x},{event.y}")
        if line is None:
            return None
        menu = self._question_menu
        if menu is None:
            from flextool.gui.assistant.chat_window import styled_menu

            menu = self._question_menu = styled_menu(self)
        menu.delete(0, "end")
        menu.add_command(label=strings.ASK_AGAIN, accelerator=strings.ASK_AGAIN_KEY,
                         command=lambda: self._question_action(AGAIN, line))
        menu.add_command(label=strings.EDIT_QUESTION, accelerator=strings.EDIT_KEY,
                         command=lambda: self._question_action(EDIT, line))
        menu.tk_popup(event.x_root, event.y_root)
        return "break"

    # --- writing --------------------------------------------------------------------------------------------
    def _at_end(self) -> bool:
        try:
            return self.text.yview()[1] >= 0.999
        except tk.TclError:
            return True

    def _insert(self, text: str, tags: tuple[str, ...] = (), index: str = "end-1c") -> None:
        self.text.insert(index, text, tags)

    def _follow(self, was_at_end: bool) -> None:
        if was_at_end:
            self.text.see("end")

    def _ensure_newline(self) -> None:
        if self.text.index("end-1c") != "1.0" and self.text.get("end-2c", "end-1c") != "\n":
            self._insert("\n")

    def clear(self) -> None:
        for buttons in self._buttons.values():
            for button in buttons:
                button.destroy()
        self._buttons.clear()
        self._turns.clear()
        for child in list(self.text.children.values()):
            child.destroy()                 # an error's buttons (Tk destroys them with their text, Python not)
        self.text.delete("1.0", "end")
        stale = [m for m in self.text.mark_names() if str(m).startswith(LINE_PREFIXES + ("user_",))]
        if stale:
            self.text.mark_unset(*stale)
        self._tool_headers.clear()
        self._md = MarkdownStream()
        self._held = ""
        self._answer_open = False
        self._in_turn = False
        self._example = False

    def show_example(self, text: str) -> None:
        """The example of an empty "New chat"."""
        self.clear()
        self._insert(text + "\n", ("example",))
        self._example = True

    def clear_example(self) -> None:
        if self._example:
            self.clear()

    def append_user(self, text: str, *, pending: bool) -> int:
        """The user's question at the end; returns its id for :meth:`mark_user`."""
        self.clear_example()
        self._close_answer()
        self._ensure_newline()
        self._user_count += 1
        n = self._user_count
        start = self.text.index("end-1c")
        self._insert(strings.YOU + "  ", ("speaker",))
        self._insert(text, ("user_pending" if pending else "user",))
        self.text.mark_set(f"user_{n}_end", "end-1c")
        self.text.mark_gravity(f"user_{n}_end", "left")
        self._insert("\n")
        self._line_mark(f"user_{n}", start)
        self._in_turn = False
        self.text.see("end")
        return n

    def mark_user(self, line: int, state: str) -> None:
        """*state*: ``"sent"`` (no longer dim) or ``"not_sent"`` ("Not sent." after it)."""
        if not self._has_mark(f"user_{line}"):
            return
        start = self.text.index(f"user_{line} + {len(strings.YOU) + 2} chars")
        self.text.tag_remove("user_pending", start, f"user_{line}_end")
        self.text.tag_add("user", start, f"user_{line}_end")
        if state == "not_sent":
            self.text.insert(f"user_{line}_end", "  " + strings.NOT_SENT, ("notice",))

    def user_text(self, line: int) -> str:
        """The question of *line* as shown (without the speaker and any "Not sent.")."""
        if not self._has_mark(f"user_{line}"):
            return ""
        start = self.text.index(f"user_{line} + {len(strings.YOU) + 2} chars")
        return self.text.get(start, f"user_{line}_end")

    def _has_mark(self, name: str) -> bool:
        return name in self.text.mark_names()

    def _line_mark(self, name: str, index: str) -> None:
        """The mark *name* at the start of a line (right gravity: a detail inserted just before the line, at the end
        of the line above, moves it on; set again after every edit at the line's start)."""
        self.text.mark_set(name, index)
        self.text.mark_gravity(name, "right")

    def begin_answer(self) -> None:
        if self._answer_open:
            return
        self.clear_example()
        self._ensure_newline()
        if not self._in_turn:
            self._insert(strings.ASSISTANT + "  ", ("speaker",))
            self._in_turn = True
        self.text.mark_set("answer_end", "end-1c")
        self.text.mark_gravity("answer_end", "right")
        self.text.mark_set("pending_start", "answer_end")
        self.text.mark_gravity("pending_start", "left")
        self._md = MarkdownStream()
        self._answer_open = True
        try:
            sync_fonts(self.text)           # a font size changed since the last answer reaches bold, code, …
        except tk.TclError:
            pass

    def _insert_line(self, line: Line, index: str) -> None:
        for text, tags in line:
            self.text.insert(index, text, ("assistant",) + tuple(tags))

    def stream(self, delta: str) -> None:
        """Add a piece of the answer.  A last line that is or may become a ``Keep:`` line (the model's working notes
        leaking into its answer, :func:`~flextool.assistant.session.may_become_keep_line`) is held back until the
        answer goes on or ends (:meth:`end_answer`), so the live answer shows what the saved chat shows."""
        from flextool.assistant.session import may_become_keep_line

        if not delta:
            return
        at_end = self._at_end()
        self.begin_answer()
        text, self._held = self._held + delta, ""
        content = text.rstrip()
        start = content.rfind("\n") + 1
        if content and may_become_keep_line(content[start:]):
            text, self._held = text[:start], text[start:]
        if text:
            self._feed(text)
        self._follow(at_end)

    def _feed(self, delta: str) -> None:
        lines = self._md.feed(delta)
        if not lines:
            self.text.insert("answer_end", delta, ("assistant",))
        else:
            self.text.delete("pending_start", "answer_end")
            for line in lines:
                self._insert_line(line, "answer_end")
            self.text.mark_set("pending_start", "answer_end")
            if self._md.pending:
                self.text.insert("answer_end", self._md.pending, ("assistant",))

    def _close_answer(self, final: str | None = None, *, notice: bool = False) -> None:
        """Close the answer.  A held-back ``Keep:`` line is shown unless *final* (the turn's answer as the session
        stored it) leaves it out, or a *notice* closes the answer: the session's notices after an answer come
        before the turn's end, and a final answer that ends in a ``Keep:`` line follows a note request (the
        repeated-steps notice comes only after one), so the session leaves the line out."""
        from flextool.assistant.session import is_keep_line

        if not self._answer_open:
            return
        held, self._held = self._held, ""
        if held:
            dropped = is_keep_line(held) and (notice or (final is not None
                                                         and not final.rstrip().endswith(held.strip())))
            if not dropped:
                self._feed(held)
        self.text.delete("pending_start", "answer_end")
        self._insert_line(self._md.finish(), "answer_end")
        self._answer_open = False
        self._ensure_newline()

    def end_answer(self, final: str | None = None) -> None:
        """The answer has ended; *final*: the turn's answer as the session stored it (a leaked ``Keep:`` line left
        out), ``None`` when the answer was stopped."""
        at_end = self._at_end()
        self._close_answer(final)
        self._follow(at_end)

    def tool_running(self, name: str, args_preview: str) -> int:
        """A tool line for a call that is running; returns its id for :meth:`tool_done`."""
        at_end = self._at_end()
        self._close_answer()
        self._ensure_newline()
        self._tool_count += 1
        n = self._tool_count
        header = f"{name} {args_text(args_preview)}".rstrip()
        self._tool_headers[n] = header
        start = self.text.index("end-1c")
        self._insert(f"{TOOL_CLOSED} {header}…\n", ("tool",))
        self._line_mark(f"tool_{n}", start)
        self._follow(at_end)
        return n

    def tool_done(self, line: int, ok: bool, summary: str, seconds: float, arguments: str, result_preview: str,
                  provider: str) -> None:
        """The finished call: its outcome on the line, and the detail block (hidden until opened)."""
        from flextool.assistant.usage import number

        at_end = self._at_end()
        key = f"tool_{line}"
        header = self._tool_headers.pop(line, None)
        if not self._has_mark(key):
            return
        header = header or f"{args_text(arguments)}"
        outcome = summary if ok else f"{strings.TOOL_ERROR}: {summary}"
        start = self.text.index(key)
        self.text.delete(start, f"{start} lineend + 1 chars")
        self.text.insert(start, f"{TOOL_CLOSED} {header} → {outcome} ({seconds:.1f} s)\n", ("tool",))
        self._line_mark(key, start)
        end = self.text.index(f"{start} lineend + 1 chars")
        detail = (f"{DETAIL_INDENT}{strings.ARGUMENTS} {arguments}\n"
                  f"{DETAIL_INDENT}{strings.SENT_TO.format(provider=provider, n=number(4000))} {result_preview}\n")
        self.text.insert(end, detail, ("tooldetail", DETAIL_CLOSED))
        self._follow(at_end)

    def line_range(self, key: str) -> tuple[str, str] | None:
        """The text of the tool (or context) line *key*, its newline left out; None when there is no such line."""
        if not self._has_mark(key):
            return None
        start = self.text.index(key)
        return start, self.text.index(f"{start} lineend")

    def detail_range(self, key: str) -> tuple[str, str] | None:
        """The detail of the tool (or context) line *key* (the ``tooldetail`` run starting on the next line), or
        None while it has none (a tool still running)."""
        line = self.line_range(key)
        if line is None:
            return None
        start = self.text.index(f"{line[1]} + 1 chars")
        if "tooldetail" not in self.text.tag_names(start):
            return None
        found = self.text.tag_nextrange("tooldetail", start)
        if not found or self.text.compare(found[0], "!=", start):
            return None
        return str(found[0]), str(found[1])

    def toggle(self, key: str) -> None:
        """Open or close the detail of the tool (or context) line *key*."""
        detail = self.detail_range(key)
        if detail is None:
            return
        hidden = DETAIL_CLOSED in self.text.tag_names(detail[0])
        if hidden:
            self.text.tag_remove(DETAIL_CLOSED, *detail)
        else:
            self.text.tag_add(DETAIL_CLOSED, *detail)
        start = self.text.index(key)
        if self.text.get(start, f"{start} + 1 chars") in (TOOL_OPEN, TOOL_CLOSED):
            tags = ("tool",) if key.startswith("tool_") else ("context",)
            # The new arrow after the old one, then the old one out: the line's mark stays at the line start.
            self.text.insert(f"{start} + 1 chars", TOOL_OPEN if hidden else TOOL_CLOSED, tags)
            self.text.delete(start, f"{start} + 1 chars")

    def is_open(self, key: str) -> bool:
        """Whether the detail of the tool (or context) line *key* is shown."""
        detail = self.detail_range(key)
        return detail is not None and DETAIL_CLOSED not in self.text.tag_names(detail[0])

    def notice(self, text: str) -> None:
        at_end = self._at_end()
        self._close_answer(notice=True)
        self._ensure_newline()
        self._insert(text + "\n", ("notice",))
        self._follow(at_end)

    def error(self, text: str, actions: list[tuple[str, Callable[[], None]]]) -> list[ttk.Button]:
        """An error message, with a button for each of *actions* after it."""
        at_end = self._at_end()
        self._close_answer()
        self._ensure_newline()
        self._insert(text, ("error",))
        buttons = []
        for label, command in actions:
            button = ttk.Button(self.text, text=label, command=command)
            self._insert(" ")
            self.text.window_create("end-1c", window=button, padx=2, pady=2)
            buttons.append(button)
        self._insert("\n")
        self._follow(at_end)
        return buttons

    # --- a reopened chat ------------------------------------------------------------------------------------
    def render_history(self, messages: list[Message] | tuple[Message, ...], provider: str, *,
                       seed: str | None = None) -> None:
        """A saved conversation: questions, answers (Markdown), tool calls closed with their stored result, the
        context blocks hidden behind one line each.  *seed* (the summary the chat's first question started with)
        is not shown: the first question is shown as the user typed it."""
        from flextool.assistant import prompt
        from flextool.assistant.session import continuation_part, continuing_text

        self.clear()
        results = {m.tool_call_id: m for m in messages if m.role == "tool"}
        first = True
        for m in messages:
            if m.kind == "continuation":
                part = continuation_part(m)
                if part is not None:
                    self.notice(continuing_text(*part))
                continue
            if m.kind in ("note_request", "note"):
                continue                # the session's own exchange; the next part shows what came of it
            if m.role == "user":
                if m.kind == "limit":
                    continue                # the session's own instruction (review G11, C5-1)
                # The leading block is the question's context line; a block inside the text (a question that got no
                # answer went on with the next one) is left out of the question shown.
                rest = prompt.strip_context_block(m.text)
                block = m.text.lstrip()[:len(m.text.lstrip()) - len(rest)].strip() if rest != m.text else ""
                if block:
                    self._context_line(block)
                question = prompt.remove_context_blocks(m.text)
                if first and seed and question.startswith(f"{seed}\n\n"):
                    question = question[len(seed) + 2:]         # as session.question_text
                first = False
                line = self.append_user(question, pending=False)
                self.set_turn(line, m.turn)
            elif m.role == "assistant":
                if m.text.strip():
                    self._ensure_newline()
                    if not self._in_turn:
                        self._insert(strings.ASSISTANT + "  ", ("speaker",))
                        self._in_turn = True
                    for line in render_markdown(m.text):
                        self._insert_line(line, "end-1c")
                    self._ensure_newline()
                for call in m.tool_calls:
                    self._history_tool(call, results.get(call.id), provider)
        self.text.see("end")

    def _context_line(self, block: str) -> None:
        self._ensure_newline()
        self._tool_count += 1
        n = self._tool_count
        start = self.text.index("end-1c")
        self._insert(strings.CONTEXT_SENT + "\n", ("context",))
        self._line_mark(f"context_{n}", start)
        self._insert(block + "\n", ("tooldetail", DETAIL_CLOSED))

    def _history_tool(self, call, result: Message | None, provider: str) -> None:  # noqa: ANN001 — a ToolCall
        from flextool.assistant.usage import number

        self._ensure_newline()
        self._tool_count += 1
        n = self._tool_count
        header = f"{call.name} {args_text(call.raw_arguments)}".rstrip()
        if result is None:
            outcome = strings.TOOL_NOT_RUN
        elif result.is_error:
            outcome = strings.TOOL_ERROR
        else:
            outcome = strings.TOOL_DONE
        if result is not None and result.compacted:
            outcome += f" · {strings.COMPACTED}"
        start = self.text.index("end-1c")
        self._insert(f"{TOOL_CLOSED} {header} → {outcome}\n", ("tool",))
        self._line_mark(f"tool_{n}", start)
        preview = (result.text if result is not None else "")[:4000]
        detail = (f"{DETAIL_INDENT}{strings.ARGUMENTS} {call.raw_arguments}\n"
                  f"{DETAIL_INDENT}{strings.SENT_TO.format(provider=provider, n=number(4000))} {preview}\n")
        self._insert(detail, ("tooldetail", DETAIL_CLOSED))

    def contents(self) -> str:
        """The visible text (for the tests)."""
        return self.text.get("1.0", "end-1c")
