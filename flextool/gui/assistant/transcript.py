"""The transcript of one chat (P2 §4.5): a read-only ``tk.Text`` that streams the answer and shows each tool call
as one line that opens to show what was sent.

Tk thread only.  The answer arrives in pieces (:meth:`Transcript.stream`): the raw text of the line still
arriving is shown as it comes, and each completed line is replaced by its Markdown rendering
(:class:`~flextool.gui.assistant.markdown_stream.MarkdownStream`), so a long answer costs one short delete and
insert per batch.  The view follows the end only while it is at the end.

The widget takes no typing: keys that change text are dropped (copy, select all and moving around stay), Tab and
Shift+Tab move the focus on, Escape stops the answer, Enter or space on a tool line opens or closes it, and a click
on a tool line does the same.  It draws no insertion cursor (only the input box has one): while it has the focus,
the tool or context line the keyboard is on is shaded instead, so Enter and space still show what they open.  Its
tags take their colours from the widget's own theme colours.

Fonts (owner decisions 2026-11-08, :mod:`flextool.gui.assistant.fonts`): the text is in the chat's base font
(FlexTool's body text font one point larger; a ``tk.Text`` would otherwise use the fixed-width font, and bold text
in the proportional font then looked smaller).  Bold, italic and the headings are its family, code, tables and the
details of a tool call the fixed-width family at its size.  :func:`sync_fonts` sets them from the body font again —
at every answer and on a theme change — so a changed font size reaches them.
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

if TYPE_CHECKING:
    from flextool.assistant.llm_adapter import Message

#: Keys that move or select without changing the text (``execution_window.py``'s read-only filter).
NAV_KEYSYMS = frozenset({"Left", "Right", "Up", "Down", "Home", "End", "Prior", "Next", "Shift_L", "Shift_R",
                         "Control_L", "Control_R", "Alt_L", "Alt_R", "Meta_L", "Meta_R", "Super_L", "Super_R"})
#: The tag shading the tool or context line the keyboard is on (the transcript draws no insertion cursor).
KEYBOARD_LINE = "keyboard_line"
TOOL_OPEN = "▾"
TOOL_CLOSED = "▸"
DETAIL_INDENT = "    "


class Transcript(ttk.Frame):
    """The transcript widget.  *on_escape* stops the answer; *bind_commands(widget)* adds the window's tab
    commands to the text widget's own bindings."""

    def __init__(self, parent: tk.Misc, *, on_escape: Callable[[], None] | None = None,
                 bind_commands: Callable[[tk.Misc], None] | None = None) -> None:
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
        self._md = MarkdownStream()
        self._answer_open = False
        self._in_turn = False
        self._tool_count = 0
        self._user_count = 0
        self._tool_headers: dict[int, str] = {}
        self._example = False
        self._configure_tags()
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
        txt.tag_raise("sel")

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
        # After the class bindings have moved the insert mark (KeyRelease, ButtonRelease) and on focus changes.
        for sequence in ("<KeyRelease>", "<ButtonRelease-1>", "<FocusIn>", "<FocusOut>"):
            txt.bind(sequence, lambda _e: self.mark_keyboard_line(), add="+")
        for virtual in ("<<Paste>>", "<<Cut>>", "<<Clear>>", "<<PasteSelection>>", "<<Undo>>", "<<Redo>>"):
            txt.bind(virtual, lambda _e: "break")
        if txt.tk.call("tk", "windowingsystem") != "aqua":
            txt.bind("<Button-2>", lambda _e: "break")          # middle click pastes on X11 and Windows
        if bind_commands is not None:
            bind_commands(txt)

    def _on_key(self, event: tk.Event) -> str | None:  # type: ignore[type-arg]
        if event.state & 0x4 and event.keysym in ("c", "C", "a", "A", "Insert"):
            return None
        if event.keysym in NAV_KEYSYMS:
            return None
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
        tag = self._line_tag_at_insert()
        if tag is not None:
            self.toggle(tag)
        return "break"

    def _line_tag_at_insert(self) -> str | None:
        for tag in self.text.tag_names("insert"):
            if tag.startswith("tool_") or tag.startswith("context_"):
                return tag
        return None

    def mark_keyboard_line(self) -> None:
        """Shade the tool or context line holding the insert mark while the transcript has the focus."""
        txt = self.text
        try:
            txt.tag_remove(KEYBOARD_LINE, "1.0", "end")
            if txt.focus_get() is txt and self._line_tag_at_insert() is not None:
                txt.tag_add(KEYBOARD_LINE, "insert linestart", "insert lineend")
        except (tk.TclError, KeyError):
            pass

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
        self.text.delete("1.0", "end")
        self._md = MarkdownStream()
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
        tag = f"user_{n}"
        self._insert(strings.YOU + "  ", ("speaker", tag))
        self._insert(text, ("user_pending" if pending else "user", tag))
        self.text.mark_set(f"user_{n}_end", "end-1c")
        self.text.mark_gravity(f"user_{n}_end", "left")
        self._insert("\n", (tag,))
        self._in_turn = False
        self.text.see("end")
        return n

    def mark_user(self, line: int, state: str) -> None:
        """*state*: ``"sent"`` (no longer dim) or ``"not_sent"`` ("Not sent." after it)."""
        tag = f"user_{line}"
        ranges = self.text.tag_ranges(tag)
        if not ranges:
            return
        self.text.tag_remove("user_pending", ranges[0], ranges[-1])
        self.text.tag_add("user", ranges[0], ranges[-1])
        if state == "not_sent":
            self.text.insert(f"user_{line}_end", "  " + strings.NOT_SENT, ("notice", tag))

    def user_text(self, line: int) -> str:
        """The question of *line* as shown (without the speaker and any "Not sent.")."""
        ranges = self.text.tag_ranges(f"user_{line}")
        if not ranges:
            return ""
        start = self.text.index(f"{ranges[0]} + {len(strings.YOU) + 2} chars")
        return self.text.get(start, f"user_{line}_end")

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
        """Add a piece of the answer."""
        if not delta:
            return
        at_end = self._at_end()
        self.begin_answer()
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
        self._follow(at_end)

    def _close_answer(self) -> None:
        if not self._answer_open:
            return
        self.text.delete("pending_start", "answer_end")
        self._insert_line(self._md.finish(), "answer_end")
        self._answer_open = False
        self._ensure_newline()

    def end_answer(self) -> None:
        at_end = self._at_end()
        self._close_answer()
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
        self._insert(f"{TOOL_CLOSED} {header}…\n", ("tool", f"tool_{n}"))
        self._bind_tool(n)
        self._follow(at_end)
        return n

    def tool_done(self, line: int, ok: bool, summary: str, seconds: float, arguments: str, result_preview: str,
                  provider: str) -> None:
        """The finished call: its outcome on the line, and the detail block (hidden until opened)."""
        from flextool.assistant.usage import number

        at_end = self._at_end()
        tag = f"tool_{line}"
        ranges = self.text.tag_ranges(tag)
        if not ranges:
            return
        header = self._tool_headers.get(line) or f"{args_text(arguments)}"
        outcome = summary if ok else f"{strings.TOOL_ERROR}: {summary}"
        self.text.delete(ranges[0], ranges[-1])
        start = ranges[0]
        self.text.insert(start, f"{TOOL_CLOSED} {header} → {outcome} ({seconds:.1f} s)\n", ("tool", tag))
        end = self.text.index(f"{start} lineend + 1 chars")
        detail = (f"{DETAIL_INDENT}{strings.ARGUMENTS} {arguments}\n"
                  f"{DETAIL_INDENT}{strings.SENT_TO.format(provider=provider, n=number(4000))} {result_preview}\n")
        self.text.insert(end, detail, ("tooldetail", f"tooldetail_{line}"))
        self.text.tag_configure(f"tooldetail_{line}", elide=True)
        self._follow(at_end)

    def _bind_tool(self, n: int) -> None:
        tag = f"tool_{n}"
        self.text.tag_bind(tag, "<Button-1>", lambda _e, t=tag: self.toggle(t))
        self.text.tag_bind(tag, "<Enter>", lambda _e: self.text.configure(cursor="hand2"))
        self.text.tag_bind(tag, "<Leave>", lambda _e: self.text.configure(cursor="arrow"))

    def toggle(self, tag: str) -> None:
        """Open or close the detail of the tool (or context) line *tag*."""
        detail = self._detail_tag(tag)
        if not self.text.tag_ranges(detail):
            return
        hidden = self._elided(detail)
        self.text.tag_configure(detail, elide=not hidden)
        ranges = self.text.tag_ranges(tag)
        if ranges:
            first = self.text.get(ranges[0], f"{ranges[0]} + 1 chars")
            if first in (TOOL_OPEN, TOOL_CLOSED):
                tags = ("tool", tag) if tag.startswith("tool_") else ("context", tag)
                self.text.delete(ranges[0], f"{ranges[0]} + 1 chars")
                self.text.insert(ranges[0], TOOL_OPEN if hidden else TOOL_CLOSED, tags)

    def _elided(self, tag: str) -> bool:
        value = self.text.tag_cget(tag, "elide")
        try:
            return bool(self.text.tk.getboolean(value)) if value not in ("", None) else False
        except (tk.TclError, ValueError):
            return False

    @staticmethod
    def _detail_tag(tag: str) -> str:
        return tag.replace("tool_", "tooldetail_", 1) if tag.startswith("tool_") else \
            tag.replace("context_", "contextdetail_", 1)

    def is_open(self, tag: str) -> bool:
        """Whether the detail of the tool (or context) line *tag* is shown."""
        detail = self._detail_tag(tag)
        return bool(self.text.tag_ranges(detail)) and not self._elided(detail)

    def notice(self, text: str) -> None:
        at_end = self._at_end()
        self._close_answer()
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
    def render_history(self, messages: list[Message] | tuple[Message, ...], provider: str) -> None:
        """A saved conversation: questions, answers (Markdown), tool calls closed with their stored result, the
        context blocks hidden behind one line each."""
        from flextool.assistant import prompt
        from flextool.assistant.session import SESSION_MESSAGES, continuation_part, continuing_text

        self.clear()
        results = {m.tool_call_id: m for m in messages if m.role == "tool"}
        for m in messages:
            if m.kind == "continuation":
                part = continuation_part(m)
                if part is not None:
                    self.notice(continuing_text(*part))
                continue
            if m.kind in ("note_request", "note"):
                continue                # the session's own exchange; the next part shows what came of it
            if m.role == "user":
                if m.text.startswith(SESSION_MESSAGES["limit"]):
                    continue
                question = prompt.strip_context_block(m.text)
                block = m.text.lstrip()[:len(m.text.lstrip()) - len(question)].strip() if question != m.text else ""
                if block:
                    self._context_line(block)
                self.append_user(question, pending=False)
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
        tag = f"context_{n}"
        self._insert(strings.CONTEXT_SENT + "\n", ("context", tag))
        self._insert(block + "\n", ("tooldetail", f"contextdetail_{n}"))
        self.text.tag_configure(f"contextdetail_{n}", elide=True)
        self.text.tag_bind(tag, "<Button-1>", lambda _e, t=tag: self.toggle(t))

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
        self._tool_headers[n] = header
        self._insert(f"{TOOL_CLOSED} {header} → {outcome}\n", ("tool", f"tool_{n}"))
        self._bind_tool(n)
        preview = (result.text if result is not None else "")[:4000]
        detail = (f"{DETAIL_INDENT}{strings.ARGUMENTS} {call.raw_arguments}\n"
                  f"{DETAIL_INDENT}{strings.SENT_TO.format(provider=provider, n=number(4000))} {preview}\n")
        self._insert(detail, ("tooldetail", f"tooldetail_{n}"))
        self.text.tag_configure(f"tooldetail_{n}", elide=True)

    def contents(self) -> str:
        """The visible text (for the tests)."""
        return self.text.get("1.0", "end-1c")
