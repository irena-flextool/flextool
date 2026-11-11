"""The short Markdown of an answer as tagged text, line by line while it streams (P2 §4.5).

:class:`MarkdownStream` takes the answer in pieces of any size (:meth:`feed`) and gives back each line once it is
complete, as segments ``(text, tags)`` that the transcript inserts with those Tk tag names; :meth:`finish` gives
the last line, which has no line end.  The pieces never matter: the same answer fed whole or in pieces of one
character gives the same segments, because a line is parsed only when its end has arrived and the only state
carried between lines is whether a fenced code block is open.

The subset (what the system prompt tells the model, P2 §7):

* ``#``, ``##``, ``###`` headings (``md_h1`` … ``md_h3``; more ``#`` stay literal);
* list items (``-``, ``*``, ``+`` → "•", numbered items keep their number; ``md_list``);
* ``**bold**`` (``md_bold``), ``*italic*`` (``md_italic``) and ```` `code` ```` (``md_code``) inside a line; a
  marker without its closing partner on the same line stays literal.  An italic ``*`` opens only before a
  character that is not a space and not after a letter or digit, and closes only after one that is not a space and
  not before a letter or digit (``2 * 3 * 4`` and ``a*b*c`` stay literal); ``_`` never marks italic (FlexTool's
  names have underscores);
* fenced blocks between ```` ``` ```` lines (``md_codeblock``; the fence lines themselves are not shown; a block
  still open when the answer ends is shown as code);
* lines that start with ``|`` verbatim in the fixed font (``md_table``).

Links are not made clickable and no link tag exists: ``[text](address)`` is shown as written.  Standard library
only.
"""
from __future__ import annotations

import re

Segment = tuple[str, tuple[str, ...]]
Line = list[Segment]

_HEADING = re.compile(r"^(#{1,3})[ \t]+(.*?)[ \t#]*$")
_FENCE = re.compile(r"^[ ]{0,3}(```|~~~)")
_BULLET = re.compile(r"^([ \t]*)([-*+])[ \t]+(.*)$")
_NUMBERED = re.compile(r"^([ \t]*)(\d{1,9}[.)])[ \t]+(.*)$")


def _italic_end(text: str, start: int) -> int:
    """The index of the ``*`` that closes an italic span opened at *start* - 1, or -1 (see the module text)."""
    end = text.find("*", start)
    if end <= start or text.startswith("**", end):
        return -1
    inner = text[start:end]
    if inner[0].isspace() or inner[-1].isspace() or (end + 1 < len(text) and text[end + 1].isalnum()):
        return -1
    return end


def inline(text: str, base: tuple[str, ...] = ()) -> Line:
    """The segments of one line's text: ``**bold**``, ``*italic*`` and ```` `code` ```` spans, the rest plain
    (with *base*).  A marker whose partner is not on the line is literal text."""
    out: Line = []
    plain: list[str] = []

    def flush() -> None:
        if plain:
            out.append(("".join(plain), base))
            plain.clear()

    i = 0
    n = len(text)
    while i < n:
        if text[i] == "`":
            run = len(text[i:]) - len(text[i:].lstrip("`"))
            ticks = "`" * run
            end = text.find(ticks, i + run)
            if end != -1 and text[i + run:end].strip():
                flush()
                out.append((text[i + run:end], base + ("md_code",)))
                i = end + run
                continue
            plain.append(ticks)
            i += run
            continue
        if text.startswith("**", i):
            end = text.find("**", i + 2)
            if end != -1 and text[i + 2:end].strip():
                flush()
                out.extend(inline(text[i + 2:end], base + ("md_bold",)))
                i = end + 2
                continue
            plain.append("**")
            i += 2
            continue
        if text[i] == "*" and (i == 0 or not text[i - 1].isalnum()):
            end = _italic_end(text, i + 1)
            if end != -1:
                flush()
                out.extend(inline(text[i + 1:end], base + ("md_italic",)))
                i = end + 1
                continue
        plain.append(text[i])
        i += 1
    flush()
    return out


class MarkdownStream:
    """The incremental parser: :meth:`feed` pieces, get the completed lines; :meth:`finish` at the end."""

    def __init__(self) -> None:
        self._pending = ""
        self._in_fence = False
        self._fence = ""

    @property
    def pending(self) -> str:
        """The raw text of the line that is not complete yet."""
        return self._pending

    def feed(self, text: str) -> list[Line]:
        """Add *text*; return the lines it completed, each ending with ``"\\n"`` in its last segment (a fence line
        that is not shown gives an empty line list entry)."""
        self._pending += text
        lines: list[Line] = []
        while True:
            cut = self._pending.find("\n")
            if cut < 0:
                return lines
            raw = self._pending[:cut]
            self._pending = self._pending[cut + 1:]
            lines.append(self._line(raw.removesuffix("\r"), end="\n"))

    def finish(self) -> Line:
        """The last line (no line end); the parser is ready for a new answer afterwards."""
        raw, self._pending = self._pending, ""
        line = self._line(raw, end="") if raw else []
        self._in_fence = False
        self._fence = ""
        return line

    def _line(self, raw: str, *, end: str) -> Line:
        fence = _FENCE.match(raw)
        if self._in_fence:
            if fence and fence.group(1) == self._fence and not raw.strip().strip(self._fence[0]):
                self._in_fence = False
                return []
            return [(raw + end, ("md_codeblock",))]
        if fence:
            self._in_fence = True
            self._fence = fence.group(1)
            return []
        if raw.lstrip().startswith("|"):
            return [(raw + end, ("md_table",))]
        heading = _HEADING.match(raw)
        if heading:
            tag = f"md_h{len(heading.group(1))}"
            segments = inline(heading.group(2), (tag,))
            return _with_end(segments, end, (tag,))
        bullet = _BULLET.match(raw)
        if bullet:
            indent, _mark, rest = bullet.groups()
            segments = [(indent + "• ", ("md_list",))] + inline(rest, ("md_list",))
            return _with_end(segments, end, ("md_list",))
        numbered = _NUMBERED.match(raw)
        if numbered:
            indent, mark, rest = numbered.groups()
            segments = [(f"{indent}{mark} ", ("md_list",))] + inline(rest, ("md_list",))
            return _with_end(segments, end, ("md_list",))
        return _with_end(inline(raw), end, ())


def _with_end(segments: Line, end: str, tags: tuple[str, ...]) -> Line:
    """*segments* with the line end added to the last plain segment of the same *tags* (or as its own)."""
    if not end:
        return segments
    if segments and segments[-1][1] == tags:
        text, last_tags = segments[-1]
        return segments[:-1] + [(text + end, last_tags)]
    return segments + [(end, tags)]


def render(text: str) -> list[Line]:
    """All lines of a whole answer (a reopened chat's history)."""
    stream = MarkdownStream()
    lines = stream.feed(text)
    last = stream.finish()
    if last:
        lines.append(last)
    return lines
