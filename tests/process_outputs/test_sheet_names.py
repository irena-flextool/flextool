"""Excel sheet names keep their time-resolution suffix (audit bug 4).

Excel cuts a sheet name at 31 characters and compares names without regard to case.  Before, the writer cut the
end: ``unit_curtailment__outputNode__dt`` became ``unit_curtailment__outputNode__d`` (the per-period name for the
per-step table) and the real ``__d`` table ``unit_curtailment__outputNode_1``.  The shipped long names have
hand-made sheet names (``SHEET_NAMES``); the automatic shortening is the fallback for user renames.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from flextool.process_outputs._sheet_names import MAX_SHEET_NAME, SHEET_NAMES, excel_sheet_name

DEFAULT_PLOTS = Path(__file__).resolve().parents[2] / "flextool" / "schemas" / "default_plots.yaml"

# User-renamed tables longer than 31 characters (not in SHEET_NAMES): the automatic fallback.
PAIRS = [
    ("my_unit_curtailment__outputNode__dt", "my_unit_curtailment__outputNode__d"),
    ("my_unit_outputNode_VRE_curtailment__dt", "my_unit_outputNode_VRE_curtailment__d"),
    ("my_slack__reserve__upDown__group__dt", "my_slack__reserve__upDown__group__d"),
    ("my_process__reserve__upDown__node__dt", "my_process__reserve__upDown__node__d_average"),
]


def _shipped_display_names() -> list[str]:
    """The display names of every output ``default_plots.yaml`` exports."""
    rename = yaml.safe_load(DEFAULT_PLOTS.read_text(encoding="utf-8"))["rename"]
    names = []
    for entry in rename.values():
        name, export = (entry[0], bool(entry[1])) if isinstance(entry, list) else (entry, True)
        if export:
            names.append(str(name))
    return names


def test_every_shipped_sheet_name_fits_and_is_unique() -> None:
    """Each exported table gets a sheet name of at most 31 characters that is its own hand-made or display name
    (never an automatic cut or counter), unique without regard to case."""
    used: set[str] = set()
    sheets = [excel_sheet_name(n, used) for n in _shipped_display_names()]
    assert all(len(s) <= MAX_SHEET_NAME for s in sheets), [s for s in sheets if len(s) > MAX_SHEET_NAME]
    assert len({s.lower() for s in sheets}) == len(sheets)
    names = _shipped_display_names()
    assert sheets == [SHEET_NAMES.get(n, n) for n in names]


def test_the_hand_made_names_keep_the_resolution_and_cover_every_long_name() -> None:
    shipped = set(_shipped_display_names())
    assert {n for n in shipped if len(n) > MAX_SHEET_NAME} <= set(SHEET_NAMES)
    assert set(SHEET_NAMES) <= shipped
    for long, short in SHEET_NAMES.items():
        assert len(short) <= MAX_SHEET_NAME
        assert short.endswith("__dt") == long.endswith("__dt")


@pytest.mark.parametrize("per_step, per_period", PAIRS)
def test_a_pair_gets_two_names_with_their_own_suffix(per_step, per_period) -> None:
    used: set[str] = set()
    a, b = excel_sheet_name(per_step, used), excel_sheet_name(per_period, used)
    assert a != b and len(a) <= MAX_SHEET_NAME and len(b) <= MAX_SHEET_NAME
    assert a.endswith("__dt")
    assert b.endswith(per_period[per_period.rindex("__d"):])


def test_a_short_name_is_kept() -> None:
    assert excel_sheet_name("reserve__group__provider__dt", set()) == "reserve__group__provider__dt"


def test_names_that_differ_only_in_case_are_told_apart() -> None:
    used: set[str] = set()
    assert excel_sheet_name("Costs__d", used) == "Costs__d"
    assert excel_sheet_name("costs__d", used) == "costs_1__d"


def test_the_counter_stays_within_the_limit() -> None:
    used: set[str] = set()
    names = [excel_sheet_name("unit_outputNode_VRE_curtailment_share__dt", used) for _ in range(120)]
    assert len(set(names)) == 120
    assert all(len(n) <= MAX_SHEET_NAME and n.endswith("__dt") for n in names)
