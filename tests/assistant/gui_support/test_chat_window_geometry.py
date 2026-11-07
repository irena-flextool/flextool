"""Where the chat window opens (P2 §13.1 test 5a, P2-O12, critique 2 MJ5).

The frame — client area plus a 22 px title bar and a 4 px border — must fit the work area of the main window's
monitor; the width is 80 average characters or 72 fixed ones, whichever is wider, capped by the work area; the
window goes to the side the Execution window does not cover.  The expected numbers are worked out by hand here.
"""
from __future__ import annotations

from flextool.gui.assistant.geometry import (
    Decoration,
    average_char_px,
    chat_window_geometry,
    content_width,
    parse_geometry,
)

WORK = (1920, 0, 1920, 1040)          # a second monitor right of the first, with a 40 px taskbar below
DECO = Decoration(top=22, border=4)


def _frame(geometry: str, deco: Decoration = DECO) -> tuple[int, int, int, int]:
    x, y, w, h = parse_geometry(geometry)
    return x, y, w + 2 * deco.border, h + deco.top + deco.border


def test_the_frame_fills_the_work_area_height_and_stays_inside():
    geometry = chat_window_geometry(WORK, DECO, avg_char_px=7.0, fixed_char_px=8.0, chrome_px=40)
    # 80 * 7 = 560, 72 * 8 = 576 -> 576 + 40 = 616 client px; height 1040 - 22 - 4 = 1014.
    assert geometry == "616x1014+3216+0"           # x = 1920 + 1920 - (616 + 8)
    fx, fy, fw, fh = _frame(geometry)
    assert (fy, fh) == (0, 1040)
    assert fx >= WORK[0] and fx + fw == WORK[0] + WORK[2]


def test_the_average_characters_win_when_wider():
    assert content_width(avg_char_px=9.0, fixed_char_px=7.0) == 720       # 80 * 9 > 72 * 7 = 504
    assert content_width(avg_char_px=6.0, fixed_char_px=7.5, chrome_px=10) == 550    # 72 * 7.5 + 10


def test_the_width_is_capped_by_a_narrow_work_area():
    narrow = (0, 30, 500, 700)
    geometry = chat_window_geometry(narrow, DECO, avg_char_px=8.0, fixed_char_px=8.0, chrome_px=40)
    assert geometry == "492x674+0+30"              # 500 - 2*4 wide, 700 - 22 - 4 high
    fx, fy, fw, fh = _frame(geometry)
    assert (fx, fy, fw, fh) == narrow


def test_opposite_the_execution_window():
    exec_right = (3000, 0, 760, 1000)            # mostly on the right half of the work area
    geometry = chat_window_geometry(WORK, DECO, 7.0, 8.0, exec_right, chrome_px=40)
    assert geometry == "616x1014+1920+0"           # left edge of the work area
    exec_left = (1920, 0, 900, 1000)
    assert chat_window_geometry(WORK, DECO, 7.0, 8.0, exec_left, chrome_px=40) == "616x1014+3216+0"


def test_without_decoration_and_with_a_minimum():
    geometry = chat_window_geometry((0, 0, 1000, 800), Decoration(), 5.0, 5.0, min_size=(600, 100))
    assert geometry == "600x800+400+0"             # 80 * 5 = 400 < minimum 600


def test_average_from_the_sample_and_parsing():
    assert average_char_px(560, "x" * 80) == 7.0
    assert parse_geometry("616x1014+3216+-8") == (3216, -8, 616, 1014)
    assert parse_geometry("616x1014-0+0") is None
