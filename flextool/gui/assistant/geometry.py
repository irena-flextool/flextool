"""Where the chat window opens and how big it is (P2 §4.2, P2-O11/O12): the arithmetic only, without Tk.

The window is as tall as the work area of the main window's monitor (taskbars and docks excluded) and about 80
characters of the transcript font wide — or the work area's width when that is smaller.  The work area is the
*frame* rectangle the window manager allows; Tk's ``geometry()`` sets the *client* size and the frame position, so
the title bar (*decoration* ``top``) and the border are taken off the client size so that the whole frame fits.
The window goes to the side the Execution window does not cover, else to the right edge.  The measured values
come from :mod:`flextool.gui.assistant.chat_window`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

#: The text whose average character width stands for "a character" of the transcript font.
SAMPLE = "The quick brown fox jumps over the lazy dog, 0123456789."
#: Characters of the transcript font, and of the fixed font (tables and code), the window is wide.
CHARACTERS = 80
FIXED_CHARACTERS = 72

Rect = tuple[int, int, int, int]        # (x, y, width, height)


@dataclass(frozen=True)
class Decoration:
    """What the window manager adds around the client area: the title bar height (``top``) and the border width
    on the other sides."""

    top: int = 0
    border: int = 0


def average_char_px(sample_px: int, sample: str = SAMPLE) -> float:
    """The average character width from the measured width of *sample*."""
    return sample_px / max(1, len(sample))


def content_width(avg_char_px: float, fixed_char_px: float, chrome_px: int = 0) -> int:
    """The client width the text needs: 80 average characters or 72 fixed ones, whichever is wider, plus the
    scroll bar and paddings (*chrome_px*)."""
    return int(round(max(avg_char_px * CHARACTERS, fixed_char_px * FIXED_CHARACTERS))) + max(0, chrome_px)


def chat_window_geometry(work_area: Rect, decoration: Decoration, avg_char_px: float, fixed_char_px: float,
                         exec_rect: Rect | None = None, *, chrome_px: int = 0, min_size: tuple[int, int] = (0, 0)
                         ) -> str:
    """The ``WxH+X+Y`` for ``geometry()``: client size, frame position.

    *work_area* is the frame rectangle the window may cover; *exec_rect* the Execution window's frame when it is
    open.  The frame (client + decoration) never leaves the work area, also when *min_size* (the client minimum)
    is larger than it allows.
    """
    wx, wy, ww, wh = work_area
    top, border = max(0, decoration.top), max(0, decoration.border)
    max_client_w = max(1, ww - 2 * border)
    max_client_h = max(1, wh - top - border)
    width = min(max_client_w, max(min_size[0], content_width(avg_char_px, fixed_char_px, chrome_px)))
    height = max_client_h
    frame_w = width + 2 * border
    if exec_rect is not None and _covers_right(exec_rect, work_area):
        x = wx
    else:
        x = wx + ww - frame_w
    return f"{width}x{height}+{x}+{wy}"


def _covers_right(exec_rect: Rect, work_area: Rect) -> bool:
    """Whether the Execution window lies more on the right half of the work area than on the left half."""
    ex, _ey, ew, _eh = exec_rect
    wx, _wy, ww, _wh = work_area
    middle = wx + ww / 2
    right = max(0.0, min(ex + ew, wx + ww) - max(ex, middle))
    left = max(0.0, min(ex + ew, middle) - max(ex, wx))
    return right > left


def parse_geometry(text: str) -> Rect | None:
    """Tk's ``WxH+X+Y`` (``+-8`` for a negative offset) as ``(x, y, w, h)``; ``None`` for anything else, also for
    an offset counted from the right or bottom edge."""
    m = _GEOMETRY.match(text.strip())
    if not m:
        return None
    return int(m.group(3)), int(m.group(4)), int(m.group(1)), int(m.group(2))


_GEOMETRY = re.compile(r"^(\d+)x(\d+)\+(-?\d+)\+(-?\d+)$")
