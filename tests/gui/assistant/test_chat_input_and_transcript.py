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
    (start, end) = text.tag_ranges("tool_1")[:2]
    header = text.get(start, end)
    assert header.startswith("▸ echo text=\"north\" → 1 items (")
    assert not tab.transcript.is_open("tool_1")
    text.mark_set("insert", f"{start} + 3 chars")
    key(text, "<Return>")
    assert tab.transcript.is_open("tool_1")
    assert text.get(start, f"{start} + 1 chars") == "▾"
    detail = text.get(*text.tag_ranges("tooldetail_1")[:2])
    assert 'Arguments: {"text": "north"}' in detail
    assert "Sent to Custom endpoint (first 4 000 characters): " in detail
    key(text, "<Return>")
    assert not tab.transcript.is_open("tool_1")
    before = tab.transcript.contents()
    key(text, "<Key-x>")                                       # read-only
    key(text, "<BackSpace>")
    assert tab.transcript.contents() == before
    assert text.get(*text.tag_ranges("tool_1")[:2]) == header


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
    (line_start, line_end) = text.tag_ranges("tool_1")[:2]
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
    buttons = {str(w.cget("text")): w for w in widgets(tab.transcript.text)}
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
    text.mark_set("insert", f"{text.tag_ranges('context_1')[0]} + 2 chars")
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
