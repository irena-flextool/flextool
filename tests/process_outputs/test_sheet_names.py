"""Excel sheet names keep their time-resolution suffix (audit bug 4).

Excel cuts a sheet name at 31 characters and compares names without regard to case.  Before, the writer cut the
end: ``unit_curtailment__outputNode__dt`` became ``unit_curtailment__outputNode__d`` (the per-period name for the
per-step table) and the real ``__d`` table ``unit_curtailment__outputNode_1``.
"""
from __future__ import annotations

import pytest

from flextool.process_outputs._sheet_names import MAX_SHEET_NAME, excel_sheet_name

PAIRS = [
    ("unit_curtailment__outputNode__dt", "unit_curtailment__outputNode__d"),
    ("unit_outputNode_VRE_curtailment__dt", "unit_outputNode_VRE_curtailment__d"),
    ("slack__reserve__upDown__group__dt", "slack__reserve__upDown__group__d"),
    ("process__reserve__upDown__node__dt", "process__reserve__upDown__node__d_average"),
]


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
