"""The answer's Markdown while it streams (P2 §13.1 test 1).

The parser must give the same tagged lines whatever the pieces the provider streams: the answer fed whole and in
random pieces of 1–7 characters gives identical segments.  The expected segments are written out here from the
subset in P2 §4.5, not taken from the parser.
"""
from __future__ import annotations

import random

import pytest

from flextool.gui.assistant.markdown_stream import MarkdownStream, render

ANSWER = (
    "# Loss of load\n"
    "## In S2_Dry_1\n"
    "### Details\n"
    "The **upward slack** appears at node `north` in **p2030**.\n"
    "An *italic* word, *two `x` words*, but 2 * 3 * 4, a*b*c and *open stay literal.\n"
    "- first item with `code`\n"
    "* second **bold** item\n"
    "2. numbered item\n"
    "```\n"
    "x = **not bold** and `not code`\n"
    "  indented\n"
    "```\n"
    "| node | slack |\n"
    "|------|-------|\n"
    "An unclosed **bold and an unclosed `tick stay literal.\n"
    "A [link](https://example.org) is plain text.\n"
    "#### four hashes stay literal\n"
    "last line without end"
)

EXPECTED = [
    [("Loss of load\n", ("md_h1",))],
    [("In S2_Dry_1\n", ("md_h2",))],
    [("Details\n", ("md_h3",))],
    [("The ", ()), ("upward slack", ("md_bold",)), (" appears at node ", ()), ("north", ("md_code",)),
     (" in ", ()), ("p2030", ("md_bold",)), (".\n", ())],
    [("An ", ()), ("italic", ("md_italic",)), (" word, ", ()), ("two ", ("md_italic",)),
     ("x", ("md_italic", "md_code")), (" words", ("md_italic",)),
     (", but 2 * 3 * 4, a*b*c and *open stay literal.\n", ())],
    [("• ", ("md_list",)), ("first item with ", ("md_list",)), ("code", ("md_list", "md_code")),
     ("\n", ("md_list",))],
    [("• ", ("md_list",)), ("second ", ("md_list",)), ("bold", ("md_list", "md_bold")),
     (" item\n", ("md_list",))],
    [("2. ", ("md_list",)), ("numbered item\n", ("md_list",))],
    [],
    [("x = **not bold** and `not code`\n", ("md_codeblock",))],
    [("  indented\n", ("md_codeblock",))],
    [],
    [("| node | slack |\n", ("md_table",))],
    [("|------|-------|\n", ("md_table",))],
    [("An unclosed **bold and an unclosed `tick stay literal.\n", ())],
    [("A [link](https://example.org) is plain text.\n", ())],
    [("#### four hashes stay literal\n", ())],
    [("last line without end", ())],
]


def _streamed(text: str, seed: int) -> list:
    rng = random.Random(seed)
    stream = MarkdownStream()
    lines: list = []
    i = 0
    while i < len(text):
        n = rng.randint(1, 7)
        lines += stream.feed(text[i:i + n])
        i += n
    last = stream.finish()
    if last:
        lines.append(last)
    return lines


def test_the_whole_answer_gives_the_written_segments():
    assert render(ANSWER) == EXPECTED


@pytest.mark.parametrize("seed", range(20))
def test_random_pieces_give_the_same_segments(seed):
    assert _streamed(ANSWER, seed) == EXPECTED


def test_one_character_at_a_time():
    stream = MarkdownStream()
    lines = []
    for ch in ANSWER:
        lines += stream.feed(ch)
    lines.append(stream.finish())
    assert lines == EXPECTED


def test_no_link_tag_and_only_the_known_tags():
    tags = {tag for line in render(ANSWER) for _text, seg_tags in line for tag in seg_tags}
    assert tags == {"md_h1", "md_h2", "md_h3", "md_bold", "md_italic", "md_code", "md_list", "md_codeblock",
                    "md_table"}
    assert not any("link" in tag for tag in tags)


def test_a_fence_still_open_at_the_end_is_code_and_the_parser_starts_fresh():
    stream = MarkdownStream()
    assert stream.feed("```\nopen **x**\n") == [[], [("open **x**\n", ("md_codeblock",))]]
    assert stream.finish() == []
    assert stream.feed("**b**\n") == [[("b", ("md_bold",)), ("\n", ())]]


def test_the_partial_line_stays_pending_until_its_end():
    stream = MarkdownStream()
    assert stream.feed("**bo") == []
    assert stream.pending == "**bo"
    assert stream.feed("ld** x\n") == [[("bold", ("md_bold",)), (" x\n", ())]]
    assert stream.pending == ""


def test_crlf_line_ends_and_empty_markers():
    assert render("a\r\nb ** c `` d\n") == [[("a\n", ())], [("b ** c `` d\n", ())]]
