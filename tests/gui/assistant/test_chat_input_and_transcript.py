"""The input box and the transcript of a chat (P2 §13.2 test 7; P2-O5, P2-O9, critique M1, M8, M9).

Enter sends (also Shift+Enter), Ctrl+Enter makes a new line; what is typed while a question is prepared stays; the
question shows at once, "Not sent." after a declined notice with the text back in an empty box; the placeholder
is never part of the text; Tab walks the window in the designed order; Enter on a tool line opens it and the line
says what was sent; the main window's shortcuts never fire from the chat; the answer's Markdown is tagged; an
authentication error offers [Open settings].

Run headless:  xvfb-run -a python -m pytest tests/gui/assistant --run-gui
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from flextool.assistant.llm_errors import AuthFailed, ErrorContext, make_error
from flextool.assistant.probes import ModelFacts
from flextool.gui.assistant.notices import ConsentDialog
from flextool.gui.assistant.transcript import QUESTION_BUTTON_STYLE
from tests.assistant.gui_support.worker_rig import LOCAL, REMOTE, Tools, save_provider, say, tool_call
from tests.assistant.llm.fake_backend import Block
from tests.gui.assistant.tk_helpers import key, pump, widgets
from tests.gui.assistant.window_rig import Backends, Windows, ask, wait_idle

pytestmark = pytest.mark.gui

PLACEHOLDER = "Ask here — Enter: send, Ctrl+Enter: new line"


@pytest.fixture
def tools(monkeypatch: pytest.MonkeyPatch):
    fake = Tools()
    fake.install(monkeypatch)
    yield fake
    fake.release.set()


@pytest.fixture
def backends() -> Backends:
    return Backends()


@pytest.fixture
def windows(tk_root, tmp_path: Path, backends, tools):
    project = tmp_path / "project"
    project.mkdir()
    made = Windows(tk_root, project, backends)
    yield made
    tools.release.set()
    made.close_all()


def _text(tab) -> str:
    return tab.input.get("1.0", "end-1c")


# --- Enter, Ctrl+Enter, typing on -------------------------------------------------------------------------------
def test_enter_sends_ctrl_enter_makes_a_new_line_shift_enter_sends(windows, backends):
    save_provider(LOCAL)
    backends.add([say("First."), say("Second.")])
    window = windows.open()
    tab = window.current_tab()
    tab.input.insert("1.0", "line one")
    key(tab.input, "<Control-Return>")
    assert _text(tab) == "line one\n"
    assert not tab.busy and not tab.has_messages
    tab.input.insert("end", "line two")
    key(tab.input, "<Return>")
    assert _text(tab) == ""
    assert "You  line one\nline two" in tab.transcript.contents()
    wait_idle(window, tab)
    tab.input.insert("1.0", "again")
    key(tab.input, "<Shift-Return>")
    wait_idle(window, tab)
    assert "Second." in tab.transcript.contents()
    assert len(backends.made[0].requests) == 2


def test_typing_while_the_question_is_prepared_stays(windows, backends):
    save_provider(LOCAL)
    backends.add([say("Done.")])
    release = threading.Event()

    def slow_probe(_preset, _conn, cancel, **_kwargs) -> ModelFacts:
        release.wait(10)
        return ModelFacts()

    window = windows.open(probe=slow_probe)
    tab = ask(window, "first question")
    pump(window, lambda: tab.status_text == "Preparing…")
    assert "disabled" in tab.send_button.state()
    tab.input.insert("1.0", "the next question")
    key(tab.input, "<Return>")                                 # Send is unavailable: the text stays
    assert _text(tab) == "the next question"
    assert tab.status_text == "You can send when the current answer is done."
    release.set()
    wait_idle(window, tab)
    assert _text(tab) == "the next question"
    assert "disabled" not in tab.send_button.state()


def test_the_question_shows_at_once_and_not_sent_after_a_declined_notice(windows, backends, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    save_provider(REMOTE)
    backends.add([say("never")])
    window = windows.open()
    tab = ask(window, "Which units invest?")
    assert "You  Which units invest?" in tab.transcript.contents()          # before any event arrived
    ranges = tab.transcript.text.tag_ranges("user_pending")
    assert ranges and "Which units invest?" in tab.transcript.text.get(ranges[0], ranges[-1])
    pump(window, lambda: window.consents.current is not None, timeout=5)
    window.consents.current[2].cancel()
    wait_idle(window, tab)
    assert "Which units invest?  Not sent." in tab.transcript.contents()
    assert _text(tab) == "Which units invest?"
    assert backends.made == []


def test_the_placeholder(windows, backends):
    save_provider(LOCAL)
    backends.add([say("Done.")])
    window = windows.open()
    tab = window.current_tab()
    pump(window, timeout=0.05)
    assert tab.placeholder_shown
    assert str(tab._placeholder.cget("text")) == PLACEHOLDER
    from flextool.gui.assistant.fonts import CHAT_FONT          # the input box and its placeholder: the chat font
    assert str(tab.input.cget("font")) == CHAT_FONT == str(tab._placeholder.cget("font"))
    assert _text(tab) == ""
    key(tab.input, "<Key-x>")
    assert _text(tab) == "x"
    assert not tab.placeholder_shown
    tab.input.delete("1.0", "end")
    tab.refresh_placeholder()
    assert tab.placeholder_shown
    ask(window, "question", tab)
    assert _text(tab) == "" and not tab.placeholder_shown              # the chat has a message now
    wait_idle(window, tab)
    assert not tab.placeholder_shown
    assert PLACEHOLDER not in tab.transcript.contents()
    assert window.new_tab.placeholder_shown


# --- focus order -------------------------------------------------------------------------------------------------
def _walk(window, start, steps: int) -> list:
    seen = []
    widget = start
    for _ in range(steps):
        key(widget, "<Tab>")
        widget = window.focus_lastfor()
        seen.append(widget)
    return seen


def test_tab_walks_the_window_in_order(windows, backends):
    save_provider(LOCAL)
    block = Block(then=say("Busy answer."))
    backends.add([say("Done."), block])
    window = windows.open(open_settings=lambda _parent: None)
    tab = ask(window, "question")
    wait_idle(window, tab)
    tab.input.focus_force()
    walked = _walk(window, tab.input, 8)
    assert walked == [tab.close_button, tab.send_button, window.chats_button, window.settings_button,
                      window.how_button, window.notebook, tab.transcript.text, tab.input]
    key(tab.input, "<Shift-Tab>")
    assert window.focus_lastfor() is tab.transcript.text
    ask(window, "again", tab)                               # busy: Stop instead of Send
    assert block.started.wait(5)
    walked = _walk(window, tab.input, 3)
    assert walked == [tab.close_button, tab.stop_button, window.chats_button]
    block.release.set()
    wait_idle(window, tab)


# --- the transcript ----------------------------------------------------------------------------------------------
def test_enter_on_a_tool_line_opens_it_and_it_says_what_was_sent(windows, backends):
    save_provider(LOCAL)
    backends.add([tool_call("echo", "north"), say("The north node.")])
    window = windows.open()
    tab = ask(window, "question")
    wait_idle(window, tab)
    text = tab.transcript.text
    (start, end) = tab.transcript.line_range("tool_1")
    header = text.get(start, end)
    assert header.startswith("▸ echo text=\"north\" → 1 items (")
    assert not tab.transcript.is_open("tool_1")
    text.mark_set("insert", f"{start} + 3 chars")
    key(text, "<Return>")
    assert tab.transcript.is_open("tool_1")
    assert text.get(start, f"{start} + 1 chars") == "▾"
    detail = text.get(*tab.transcript.detail_range("tool_1"))
    assert 'Arguments: {"text": "north"}' in detail
    assert "Sent to Custom endpoint (first 4 000 characters): " in detail
    key(text, "<Return>")
    assert not tab.transcript.is_open("tool_1")
    before = tab.transcript.contents()
    key(text, "<Key-x>")                                       # read-only
    key(text, "<BackSpace>")
    assert tab.transcript.contents() == before
    assert text.get(*tab.transcript.line_range("tool_1")) == header


def test_the_transcript_draws_no_cursor_but_copies_and_shows_the_keyboard_line(windows, backends):
    """Only the input box shows an insertion cursor; the transcript stays selectable, copyable, reachable by Tab,
    and shades the tool line the keyboard is on instead."""
    save_provider(LOCAL)
    backends.add([tool_call("echo", "north"), say("The north node.")])
    window = windows.open()
    tab = ask(window, "question")
    wait_idle(window, tab)
    text = tab.transcript.text
    assert int(text.cget("insertwidth")) == 0 and int(text.cget("insertofftime")) == 0
    assert int(tab.input.cget("insertwidth")) > 0 and int(tab.input.cget("insertofftime")) > 0
    assert str(text.cget("takefocus")) != "0"
    # Mouse-style selection and Ctrl+C copy.
    start = text.search("The north node.", "1.0")
    text.tag_add("sel", start, f"{start} + 9 chars")
    text.focus_force()
    pump(window, lambda: window.focus_get() is text)
    window.clipboard_clear()
    key(text, "<Control-c>")
    assert window.clipboard_get() == "The north"
    # The keyboard on a tool line: the line is shaded while the transcript has the focus.
    (line_start, line_end) = tab.transcript.line_range("tool_1")
    text.mark_set("insert", f"{line_start} + 2 chars")
    key(text, "<KeyRelease-Right>")
    shaded = text.tag_ranges("keyboard_line")
    assert shaded and text.compare(shaded[0], "==", f"{line_start} linestart")
    text.mark_set("insert", start)
    key(text, "<KeyRelease-Down>")
    assert not text.tag_ranges("keyboard_line")


def test_the_streamed_markdown_is_tagged(windows, backends):
    save_provider(LOCAL)
    backends.add([say("# Head\nSome **bold** and `code`.\n- item")])
    window = windows.open()
    tab = ask(window, "question")
    wait_idle(window, tab)
    text = tab.transcript.text

    def tagged(tag: str) -> list[str]:
        ranges = text.tag_ranges(tag)
        return [text.get(ranges[i], ranges[i + 1]) for i in range(0, len(ranges), 2)]

    assert tagged("md_h1") == ["Head\n"]
    assert tagged("md_bold") == ["bold"]
    assert tagged("md_code") == ["code"]
    assert "".join(tagged("md_list")) == "• item"
    assert "**" not in tab.transcript.contents()


def test_an_auth_error_offers_the_settings_and_shortcuts_stay_out(windows, backends, tk_root):
    save_provider(LOCAL)
    err = make_error(AuthFailed, "auth", ErrorContext(provider="Custom endpoint", key_source="the key file"))
    backends.add([err])
    opened = []
    window = windows.open(open_settings=lambda parent: opened.append(parent))
    hits: list[str] = []
    for sequence in ("<Key-e>", "<F9>", "<Alt-Key-c>"):
        tk_root.bind_all(sequence, lambda _e, s=sequence: hits.append(s))
    try:
        tab = ask(window, "question")
        wait_idle(window, tab)
        assert ("Custom endpoint did not accept the key (from the key file). Check or replace the key in the "
                "assistant settings.") in tab.transcript.contents()
        (button,) = [w for w in widgets(tab.transcript.text) if str(w.cget("text")) == "Open settings"]
        for widget in (tab.input, window.notebook, button):
            for sequence in ("<Key-e>", "<F9>", "<Alt-Key-c>"):
                key(widget, sequence)
        pump(window, timeout=0.05)
        assert hits == []
        assert _text(tab) == "e"                              # typing itself works
        button.invoke()
        assert opened == [window]
        assert tab.tab_text().startswith("! ")
    finally:
        for sequence in ("<Key-e>", "<F9>", "<Alt-Key-c>"):
            tk_root.unbind_all(sequence)


def test_stop_writes_stopped_at_once_and_keeps_the_partial_answer(windows, backends):
    save_provider(LOCAL)
    block = Block(then=say("never shown"))
    backends.add([block, say("Next answer.")])
    window = windows.open()
    tab = ask(window, "question")
    assert block.started.wait(5)
    pump(window, lambda: tab.status_text == "Thinking…")
    key(tab.input, "<Escape>")
    assert tab.transcript.contents().rstrip().endswith("Stopped.")
    wait_idle(window, tab)
    ask(window, "next", tab)
    wait_idle(window, tab)
    assert "Next answer." in tab.transcript.contents()


def test_the_consent_dialog_is_the_windows_only_one(windows, backends, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    save_provider(REMOTE)
    backends.add([say("Hello.")])
    window = windows.open()
    tab = ask(window, "question")
    pump(window, lambda: window.consents.current is not None, timeout=5)
    assert [w for w in widgets(window) if isinstance(w, ConsentDialog)] == [window.consents.current[2]]
    key(tab.input, "<Escape>")                                 # Stop closes its notice
    wait_idle(window, tab)
    assert window.consents.current is None
    assert "Not sent." in tab.transcript.contents()


def test_a_conversation_still_too_long_offers_a_new_chat_with_a_summary(windows, backends):
    from flextool.assistant.llm_errors import ContextTooLong

    save_provider(LOCAL)
    too_long = [make_error(ContextTooLong, "context", ErrorContext(provider="Custom endpoint", model="fake-model"))
                for _ in range(2)]
    backends.add([say("The first answer."), *too_long], [say("Fresh start.")])
    window = windows.open()
    tab = ask(window, "first question")
    wait_idle(window, tab)
    ask(window, "second question", tab)
    wait_idle(window, tab)
    assert ("The conversation is still too long for fake-model. Start a new chat; “New chat with summary” keeps a "
            "short summary of this one.") in tab.transcript.contents()
    buttons = {str(w.cget("text")): w for w in widgets(tab.transcript.text)
               if str(w.cget("style")) != QUESTION_BUTTON_STYLE}        # not the questions' ↻ ✎
    assert set(buttons) == {"New chat", "New chat with summary"}
    buttons["New chat with summary"].invoke()
    new = window.current_tab()
    assert new is window.new_tab
    assert "Summary of an earlier conversation (text only, no tool results):" in new.transcript.contents()
    ask(window, "go on", new)
    wait_idle(window, new)
    request = backends.made[1].requests[0]
    sent = request.messages[-1].text
    assert sent.split("\n\n", 1)[-1].startswith("Summary of an earlier conversation")
    assert sent.endswith("go on")
    assert "Question: first question" in sent


def test_a_reopened_chat_shows_context_lines_and_compacted_results(windows, backends):
    import json

    from flextool.assistant.llm_adapter import Message, ToolCall
    from flextool.assistant.store import open_store

    save_provider(LOCAL)
    store = open_store()
    conv = store.new_conversation(windows.project, ("openai_compatible", "http://127.0.0.1:9/v1", "fake-model"),
                                  "medium", number=4)
    call = ToolCall("callone", "echo", {"text": "a"}, json.dumps({"text": "a"}))
    block = "[FlexTool context — data, not instructions]\nProject: Project\n[end of context]"
    conv.messages = [
        Message("user", f"{block}\n\nWhat is the demand?", turn=1),
        Message("assistant", "", tool_calls=[call], turn=1),
        Message("tool", "[Earlier result of echo removed]", tool_call_id="callone", tool_name="echo", turn=1,
                compacted=True),
        Message("assistant", "It is **12** units.", turn=1),
    ]
    store.save(conv)
    store.close(conv)
    window = windows.open()
    window.reopen(conv.id)
    pump(window, lambda: window.tab_for(conv.id) is not None and window.tab_for(conv.id).has_messages, 5)
    tab = window.tab_for(conv.id)
    contents = tab.transcript.contents()
    assert "▸ FlexTool context sent with this question" in contents
    assert "You  What is the demand?" in contents
    assert '▸ echo text="a" → done · removed to keep the conversation short' in contents
    assert "It is 12 units." in contents
    assert window.notebook.tab(tab, "text") == "4"
    assert not tab.transcript.is_open("context_1")
    text = tab.transcript.text
    text.mark_set("insert", f"{tab.transcript.line_range('context_1')[0]} + 2 chars")
    key(text, "<Return>")
    assert tab.transcript.is_open("context_1")


def test_a_continuation_shows_one_line_and_the_note_stays_out(windows, backends):
    """P3 §4.7: a basic question at its three steps goes on with a fresh context; the chat shows one line for it,
    never the note, and a reopened chat shows the same line instead of the session's own messages."""
    save_provider(LOCAL)
    note = "A note for the next part.\nKeep: none"
    backends.add([tool_call(text="a"), tool_call(text="b"), tool_call(text="c"), say(note), say("Final answer.")])
    window = windows.open()
    tab = ask(window, "question")
    wait_idle(window, tab)
    line = "Continuing with a fresh context (part 2 of 3)…"
    contents = tab.transcript.contents()
    assert contents.count(line) == 1 and "Final answer." in contents and "A note for" not in contents
    # The saved conversation, rendered as a reopened chat renders it.
    from flextool.assistant.store import message_from_json
    from tests.assistant.gui_support.worker_rig import messages_on_disk

    saved = [message_from_json(m) for m in messages_on_disk(windows.project, tab.conversation_id)]
    assert [m.kind for m in saved].count("continuation") == 1
    tab.transcript.render_history(saved, tab.provider)
    again = tab.transcript.contents()
    assert again.count(line) == 1 and "Final answer." in again and "You  question" in again
    assert "A note for" not in again and "This question continues" not in again


def test_a_keep_line_leaked_into_the_answer_is_left_out_live_and_reopened(windows, backends):
    """B1 T5: after a note request the model ends its final answer with its note's "Keep: 1" line.  The session
    leaves it out of the stored answer, the chat window holds it back while it streams: the live chat and the
    reopened one show the same answer, without it."""
    save_provider(LOCAL)
    backends.add([tool_call(text="a"), tool_call(text="b"), tool_call(text="c"), say("Found a.\nKeep: 1"),
                  say("Final answer.\n\nKeep: 1, 2\n")])
    window = windows.open()
    tab = ask(window, "question")
    wait_idle(window, tab)
    contents = tab.transcript.contents()
    assert contents.rstrip().endswith("Final answer.") and "Keep" not in contents
    from flextool.assistant.store import message_from_json
    from tests.assistant.gui_support.worker_rig import messages_on_disk

    saved = [message_from_json(m) for m in messages_on_disk(windows.project, tab.conversation_id)]
    assert saved[-1].text == "Final answer."
    tab.transcript.render_history(saved, tab.provider)
    again = tab.transcript.contents()
    assert again.rstrip().endswith("Final answer.") and "Keep" not in again


def test_a_line_that_only_starts_like_keep_is_shown(host):
    """The held-back line is shown once it is no Keep line, or when the stored answer kept it (no note was
    requested); a notice closing the answer leaves a Keep line out (the session's notices come before the end)."""
    from flextool.gui.assistant.transcript import Transcript

    transcript = Transcript(host)
    try:
        transcript.pack()
        for piece in ("Answer.\nKe", "ep it", " simple.\nK"):
            transcript.stream(piece)
        assert transcript.contents().endswith("Keep it simple.\n")       # the last "K" is held back
        transcript.stream("eep: 3")
        transcript.end_answer("Answer.\nKeep it simple.\nKeep: 3")       # no note: the session kept the line
        assert transcript.contents().rstrip().endswith("Keep it simple.\nKeep: 3")
        transcript.stream("Second.\nKeep: none")
        transcript.notice("A notice.")
        assert "Keep: none" not in transcript.contents() and transcript.contents().rstrip().endswith("A notice.")
        transcript.stream("Third.\nKeep: 2")
        transcript.end_answer()                                             # stopped: shown as stored
        assert transcript.contents().rstrip().endswith("Third.\nKeep: 2")
    finally:
        transcript.destroy()


# --- the fonts of the answer ------------------------------------------------------------------------------------
def test_bold_italic_and_code_have_the_size_of_the_body_text(host):
    """Owner decisions 2026-11-08: the chat text is one named base font, 11 pt at the GUI's default 10 pt; bold and
    italic are its family and size; inline code, code blocks, tables and tool details the fixed-width family at
    the same size; headings 14 / 13 / 11 bold (13 / 12 / 10 at body 10, in proportion); a changed body size
    reaches them all."""
    import tkinter.font as tkfont

    from flextool.gui.assistant.fonts import CHAT_FONT
    from flextool.gui.assistant.transcript import Transcript

    body = tkfont.nametofont("TkTextFont", root=host)
    fixed = tkfont.nametofont("TkFixedFont", root=host).actual()
    old_size = body.cget("size")
    body.configure(size=10)                     # the GUI's default body size
    transcript = Transcript(host)
    try:
        transcript.pack()
        text = transcript.text

        def actual(tag: str) -> dict:
            return tkfont.Font(root=host, font=text.tag_cget(tag, "font")).actual()

        assert str(text.cget("font")) == CHAT_FONT
        base = tkfont.Font(root=host, font=text.cget("font")).actual()
        assert base["family"] == body.actual("family") and base["size"] == 11
        bold, italic, code = actual("md_bold"), actual("md_italic"), actual("md_code")
        assert (bold["family"], bold["size"], bold["weight"], bold["slant"]) == (
            base["family"], base["size"], "bold", "roman")
        assert (italic["family"], italic["size"], italic["weight"], italic["slant"]) == (
            base["family"], base["size"], "normal", "italic")
        assert code["family"] == fixed["family"] and code["size"] == base["size"]
        assert actual("md_codeblock") == code and actual("md_table") == code and actual("tooldetail") == code
        assert actual("speaker") == bold
        assert [actual(h)["size"] for h in ("md_h1", "md_h2", "md_h3")] == [14, 13, 11]
        assert all(actual(h)["weight"] == "bold" for h in ("md_h1", "md_h2", "md_h3"))
        # The size of the body text changes (the Settings font size, another monitor): the next answer follows.
        body.configure(size=14)
        transcript.begin_answer()
        transcript.stream("the peak was **246 MW** in `node_a`\n")
        transcript.end_answer()
        assert tkfont.nametofont(CHAT_FONT, root=host).actual("size") == 15
        assert actual("md_bold")["size"] == 15 == actual("md_italic")["size"] == actual("md_code")["size"]
        assert [actual(h)["size"] for h in ("md_h1", "md_h2", "md_h3")] == [20, 17, 15]
    finally:
        body.configure(size=old_size)
        transcript.destroy()


# --- the token line (owner GUI feedback 2026-11-08 (2)) ------------------------------------------------------------
def test_the_token_line_stays_at_the_bottom_and_not_in_the_transcript(windows, backends):
    from flextool.assistant.llm_adapter import Message, Reply, Usage

    def reply(text: str) -> Reply:
        return Reply(Message("assistant", text), "end", None, Usage(2_000, 500, 100, 800))

    save_provider(LOCAL)
    backends.add([reply("First. " + "Line.\n" * 200), reply("Second.")])
    window = windows.open()
    tab = ask(window, "One")
    wait_idle(window, tab)
    assert str(tab.totals.cget("text")) == "This question: 3k in, 800 out · Whole chat: 3k in, 800 out"
    ask(window, "Two")
    wait_idle(window, tab)
    assert str(tab.totals.cget("text")) == "This question: 3k in, 800 out · Whole chat: 5k in, 2k out"
    assert "tokens" not in tab.transcript.contents() and "in, " not in tab.transcript.contents()
    # Below the buttons, inside the tab, whatever the transcript's scroll position.
    window.update()
    for where in (0.0, 1.0):
        tab.transcript.text.yview_moveto(where)
        window.update()
        assert tab.totals.winfo_ismapped()
        assert tab.totals.winfo_rooty() >= tab.send_button.winfo_rooty() + tab.send_button.winfo_height()
        assert tab.totals.winfo_rooty() + tab.totals.winfo_height() <= tab.winfo_rooty() + tab.winfo_height()


# --- a reopened chat: the session's instruction is found by its kind (review G11) ---------------------------------
def test_a_reopened_chat_hides_the_limit_instruction_by_kind_not_by_text(host):
    """The "tool limit reached" instruction (kind ``limit``) is not shown; a question of the user's own that quotes
    that instruction is shown (C5-1, critique MINOR 8)."""
    from flextool.assistant.llm_adapter import Message
    from flextool.assistant.session import SESSION_MESSAGES
    from flextool.gui.assistant.transcript import Transcript

    limit = SESSION_MESSAGES["limit"]
    quoted = f"{limit} What does this line mean?"
    messages = [Message("user", "first question", turn=1), Message("user", limit, turn=1, kind="limit"),
                Message("assistant", "What I have.", turn=1), Message("user", quoted, turn=2),
                Message("assistant", "It means the steps were used up.", turn=2)]
    transcript = Transcript(host)
    try:
        transcript.pack()
        transcript.render_history(messages, "Custom endpoint")
        shown = transcript.contents()
    finally:
        transcript.destroy()
    assert shown.count(limit) == 1 and "What does this line mean?" in shown
    assert "first question" in shown and "What I have." in shown


# --- review G19: macOS keys, fonts configured on a change only, shared tool-line tags -----------------------------
def test_command_c_copies_and_command_a_selects_all_on_macos(host, monkeypatch):
    """On aqua Command is Mod1 and ``<<Copy>>`` is ``<Command-c>`` (C7-m6); simulated with a patched windowing
    check and aqua's ``<<Copy>>`` sequence."""
    from flextool.gui.assistant import transcript as module

    monkeypatch.setattr(module, "is_aqua", lambda _widget: True, raising=False)
    host.event_add("<<Copy>>", "<Mod1-c>")
    transcript = module.Transcript(host)
    try:
        transcript.pack()
        transcript.notice("Ask about the north node.")
        text = transcript.text
        text.tag_add("sel", "1.0", "1.9")
        host.clipboard_clear()
        key(text, "<Mod1-c>")
        assert host.clipboard_get() == "Ask about"
        text.tag_remove("sel", "1.0", "end")
        key(text, "<Mod1-a>")
        assert text.tag_ranges("sel") and text.get("sel.first", "sel.last") == transcript.contents()
    finally:
        host.event_delete("<<Copy>>", "<Mod1-c>")
        transcript.destroy()


def test_sync_fonts_configures_a_font_only_when_a_value_changed(host, monkeypatch):
    """Every font configure re-lays out every text widget, so the per-answer sync sets nothing when nothing changed
    (C7-m8); a changed body size still reaches all six fonts, and only their size is set."""
    import tkinter.font as tkfont

    from flextool.gui.assistant.fonts import CHAT_FONT, FONTS, sync_fonts

    body = tkfont.nametofont("TkTextFont", root=host)
    old_size = body.cget("size")
    calls: list[tuple[str, dict]] = []
    original = tkfont.Font.configure

    def counting(self, **options):
        if options:
            calls.append((self.name, options))
        return original(self, **options)

    monkeypatch.setattr(tkfont.Font, "configure", counting)
    try:
        body.configure(size=10)
        sync_fonts(host)
        calls.clear()
        sync_fonts(host)
        assert calls == []
        body.configure(size=12)
        calls.clear()
        sync_fonts(host)
        assert sorted(name for name, _ in calls) == sorted([CHAT_FONT, *FONTS.values()])
        assert all(set(options) == {"size"} for _, options in calls)
        assert tkfont.nametofont(CHAT_FONT, root=host).actual("size") == 13      # body 12 + 1
    finally:
        body.configure(size=old_size)


def test_tool_lines_share_their_tags_and_still_open_one_by_one(host):
    """No tag, binding or elided tag per tool line or question (C7-m10): 20 tool lines started together and finished
    in order, one opened by click, one by toggle; the others stay closed and the question keeps its text."""
    from flextool.gui.assistant import strings
    from flextool.gui.assistant.transcript import Transcript

    transcript = Transcript(host)
    try:
        transcript.pack(fill="both", expand=True)
        host.update()
        text = transcript.text
        question = transcript.append_user("first question", pending=True)
        ids = [transcript.tool_running("echo", '{"text": "a"}') for _ in range(20)]
        for n in ids:
            transcript.tool_done(n, True, f"{n} items", 0.1, '{"text": "a"}', f"result {n}", "Custom endpoint")
        assert not [tag for tag in text.tag_names() if tag.rsplit("_", 1)[-1].isdigit()]
        bound = [tag for tag in text.tag_names() if text.tk.call(text._w, "tag", "bind", tag)]
        assert sorted(bound) == ["context", "tool"]
        assert transcript._tool_headers == {}
        for n in ids:
            assert text.get(*transcript.line_range(f"tool_{n}")) == f'▸ echo text="a" → {n} items (0.1 s)'
            assert f"result {n}" in text.get(*transcript.detail_range(f"tool_{n}"))
        transcript.toggle("tool_3")
        assert [n for n in ids if transcript.is_open(f"tool_{n}")] == [3]
        shown = str(text.tk.call(text._w, "get", "-displaychars", "1.0", "end-1c"))
        assert shown.count("result ") == 1 and "result 3\n" in shown
        assert text.get(*transcript.line_range("tool_3")).startswith("▾ echo")
        # A click on the 7th line opens it (one binding on the shared tag finds the line).
        start = transcript.line_range("tool_7")[0]
        text.see(start)
        host.update()
        x, y, _w, _h = text.bbox(f"{start} + 3 chars")
        text.event_generate("<Motion>", x=x + 1, y=y + 1)      # the text's "current" character follows the pointer
        text.event_generate("<Button-1>", x=x + 1, y=y + 1)
        assert [n for n in ids if transcript.is_open(f"tool_{n}")] == [3, 7]
        transcript.toggle("tool_3")
        assert [n for n in ids if transcript.is_open(f"tool_{n}")] == [7]
        transcript.mark_user(question, "not_sent")
        assert transcript.user_text(question) == "first question"
        assert f"You  first question  {strings.NOT_SENT}\n" in transcript.contents()
        assert not text.tag_ranges("user_pending")
        transcript.clear()
        assert not [m for m in text.mark_names() if m.startswith(("tool_", "context_", "user_"))]
        assert transcript.line_range("tool_1") is None and not transcript.is_open("tool_1")
    finally:
        transcript.destroy()


# --- asking a question again: the buttons, keys and menu of a question (item 220) ----------------------------------
def _question_transcript(host, actions: list, keys: list | None = None):
    """A transcript whose questions' actions go to *actions* as (action, n), and whether each came from a key
    (R / E: ``key=True``) to *keys*."""
    from flextool.gui.assistant.transcript import Transcript

    def on_question(action: str, n: int, *, key: bool = False) -> None:
        actions.append((action, n))
        if keys is not None:
            keys.append(key)

    transcript = Transcript(host, on_question=on_question)
    transcript.pack(fill="both", expand=True)
    host.update()
    return transcript


def test_a_question_has_its_buttons_once_sent_and_only_on_the_last_line_of_its_number(host):
    """↻ ✎ come with the worker's word that the question was sent, never on a "Not sent." line; a question that got
    no answer and the next one share a number, and only the last line keeps the buttons; they are no Tab stops, never
    part of copied text, and no embedded widget is left after clear()."""
    actions: list = []
    transcript = _question_transcript(host, actions)
    try:
        text = transcript.text
        first = transcript.append_user("first question", pending=True)
        assert not transcript.has_buttons(first)
        transcript.mark_user(first, "sent")
        transcript.set_turn(first, 1)
        again, edit = transcript.question_buttons(first)
        assert (str(again.cget("text")), str(edit.cget("text"))) == ("↻", "✎")
        assert {str(b.cget("style")) for b in (again, edit)} == {QUESTION_BUTTON_STYLE}
        assert {str(b.cget("takefocus")) for b in (again, edit)} == {"0"}
        refused = transcript.append_user("a question not sent", pending=True)
        transcript.mark_user(refused, "not_sent")
        assert not transcript.has_buttons(refused)
        stopped = transcript.append_user("stopped question", pending=True)
        transcript.mark_user(stopped, "sent")
        transcript.set_turn(stopped, 2)
        assert transcript.has_buttons(stopped)
        merged = transcript.append_user("the next question", pending=True)
        transcript.mark_user(merged, "sent")
        transcript.set_turn(merged, 2)
        assert not transcript.has_buttons(stopped) and transcript.has_buttons(merged)
        assert transcript.line_of(2) == merged
        assert transcript.turn_text(2) == "stopped question\n\nthe next question"
        assert (transcript.later_turns(1), transcript.later_turns(2)) == (1, 0)
        again.invoke()
        transcript.question_buttons(merged)[1].invoke()
        assert actions == [("again", 1), ("edit", 2)]
        assert transcript.user_text(first) == "first question"
        # Copied text has no buttons in it.
        text.tag_add("sel", "1.0", "end-1c")
        host.clipboard_clear()
        key(text, "<Control-c>")
        assert host.clipboard_get() == transcript.contents()
        assert "You  first question\n" in host.clipboard_get() and "↻" not in host.clipboard_get()
        # Tab goes past the buttons.
        key(text, "<Tab>")
        assert host.focus_get() not in [*transcript.question_buttons(first), *transcript.question_buttons(merged)]
        transcript.clear()
        assert text.winfo_children() == [] and not text.children
        assert transcript.line_of(1) is None and transcript.turn_text(1) == ""
    finally:
        transcript.destroy()


def test_r_and_e_on_a_question_line_and_its_right_click_menu(host, monkeypatch):
    import tkinter as tk

    actions: list = []
    keys: list = []
    transcript = _question_transcript(host, actions, keys)
    try:
        text = transcript.text
        transcript.notice("A notice line.")
        line = transcript.append_user("Which units invest?", pending=False)
        transcript.set_turn(line, 3)
        before = transcript.contents()
        text.mark_set("insert", f"user_{line} + 7 chars")
        key(text, "<Key-r>")
        key(text, "<Key-E>")
        text.mark_set("insert", "1.2")
        key(text, "<Key-r>")                                # a notice line: nothing
        assert actions == [("again", 3), ("edit", 3)] and keys == [True, True]
        assert transcript.contents() == before              # no key typed into the transcript
        # Held with Control, R and E are not the question's keys (which other bits count per windowing system:
        # transcript._shortcut_modifiers; Alt's bit is not set by a synthetic event on a bare X server).
        text.mark_set("insert", f"user_{line} + 7 chars")
        key(text, "<Control-Key-r>")
        key(text, "<Control-Key-e>")
        assert len(actions) == 2
        text.mark_set("insert", f"user_{line} + 7 chars")
        key(text, "<KeyRelease-Right>")
        shaded = text.tag_ranges("keyboard_line")
        assert shaded and text.compare(shaded[0], "==", f"user_{line}")
        posted: list[tk.Menu] = []
        monkeypatch.setattr(tk.Menu, "tk_popup", lambda self, *_a, **_k: posted.append(self))
        text.see(f"user_{line}")
        host.update()
        x, y, _w, _h = text.bbox(f"user_{line} + 7 chars")
        text.event_generate("<Button-3>", x=x + 1, y=y + 1)
        (menu,) = posted
        assert [menu.entrycget(i, "label") for i in range(menu.index("end") + 1)] == ["Ask again", "Edit"]
        menu.invoke(0)
        menu.invoke(1)
        assert actions[2:] == [("again", 3), ("edit", 3)] and keys[2:] == [False, False]   # the menu: no key
        x, y, _w, _h = text.bbox("1.2")
        text.event_generate("<Button-3>", x=x + 1, y=y + 1)
        assert len(posted) == 1                             # not a question line: no menu
        assert text.winfo_children() == transcript.question_buttons(line)    # the menu is not the text's
    finally:
        transcript.destroy()


def test_a_reopened_chat_shows_each_question_without_its_context_blocks(host):
    """The leading block is the question's context line; a block inside a question (one that got no answer went on
    with the next) is not shown; every question has its buttons, and drawing the chat again leaks no widget."""
    from flextool.assistant import prompt
    from flextool.assistant.context import ContextSnapshot
    from flextool.assistant.llm_adapter import Message
    from flextool.gui.assistant import strings

    block = prompt.context_block(ContextSnapshot("Project"))
    later = prompt.context_block(ContextSnapshot("Project", scenario="wet"))
    messages = [Message("user", f"{block}\n\nfirst question\n\n{later}\n\nsecond try", turn=1),
                Message("assistant", "One.", turn=1), Message("user", "next one", turn=2),
                Message("assistant", "Two.", turn=2)]
    actions: list = []
    transcript = _question_transcript(host, actions)
    try:
        transcript.render_history(messages, "Custom endpoint")
        contents = transcript.contents()
        assert contents.count(strings.CONTEXT_SENT) == 1 and "wet" not in contents
        assert "You  first question\n\nsecond try\n" in contents
        assert transcript.turn_text(1) == "first question\n\nsecond try" and transcript.turn_text(2) == "next one"
        assert transcript.line_of(1) is not None and transcript.line_of(2) is not None
        transcript.render_history(messages, "Custom endpoint")
        assert len(transcript.text.winfo_children()) == 4 and len(transcript.text.children) == 4
        transcript.clear()
        assert transcript.text.winfo_children() == []
    finally:
        transcript.destroy()


def test_the_buttons_say_ask_again_and_edit_without_the_symbol_font(host, monkeypatch):
    from flextool.gui.assistant import notices

    monkeypatch.setattr(notices.sys, "platform", "win32")
    families = ["Arial"]
    monkeypatch.setattr(notices.tkfont, "families", lambda *_a, **_k: tuple(families))
    transcript = _question_transcript(host, [])
    try:
        line = transcript.append_user("a question", pending=False)
        transcript.set_turn(line, 1)
        assert [str(b.cget("text")) for b in transcript.question_buttons(line)] == ["Ask again", "Edit"]
        families.append("Segoe UI Symbol")
        other = transcript.append_user("another", pending=False)
        transcript.set_turn(other, 2)
        assert [str(b.cget("text")) for b in transcript.question_buttons(other)] == ["↻", "✎"]
    finally:
        transcript.destroy()
