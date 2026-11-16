"""Where a context menu opens next to the pointer (no display needed)."""
from flextool.gui._menus import menu_position


def test_opens_down_right_of_the_pointer():
    assert menu_position(100, 200, 60, 80, 1920, 1080) == (101, 201)


def test_near_the_right_edge_opens_to_the_left():
    # 1900 + 1 + 60 > 1920 -> left of the pointer: 1900 - 60 - 1
    assert menu_position(1900, 200, 60, 80, 1920, 1080) == (1839, 201)


def test_near_the_bottom_edge_opens_above():
    # 1050 + 1 + 80 > 1080 -> above the pointer: 1050 - 80 - 1
    assert menu_position(100, 1050, 60, 80, 1920, 1080) == (101, 969)


def test_in_the_bottom_right_corner_the_menu_is_never_under_the_pointer():
    left, top = menu_position(1900, 1050, 60, 80, 1920, 1080)
    assert (left, top) == (1839, 969)
    assert not (left <= 1900 < left + 60 and top <= 1050 < top + 80)


def test_a_menu_larger_than_the_space_on_either_side_stays_on_screen():
    assert menu_position(10, 10, 60, 80, 50, 50) == (0, 0)
