"""Slice D — Phase D1 β narrowing: LineageFilterError.

Design: ``specs/sliceD_recourse_invest_design.md`` §6 / §18 D1.
The four blanket ``except Exception`` swallows in ``apply_derived_c`` /
``apply_synthetic_invest_sets`` must re-raise a lineage failure instead
of silently reverting to the unfiltered first-stage sets.  This pins:

* ``LineageFilterError`` is a ``ValueError`` subclass (keeps Slice B's
  ``pytest.raises(ValueError)`` pins green), and
* the two Slice B lineage guards raise the precise type.

The apply-site re-raise itself is exercised end-to-end once lineage is
active (D4); at D1 the raise path is unreachable (lineage is None), so
this file pins the type contract that makes the re-raise selective.
"""
from __future__ import annotations

import polars as pl
import pytest

from flextool.engine_polars._solve_state import LineageFilterError


def test_lineage_filter_error_is_valueerror():
    assert issubclass(LineageFilterError, ValueError)


def test_assert_lineage_castable_raises_lineage_filter_error():
    from flextool.engine_polars._derived_walks import _assert_lineage_castable

    enum_d = pl.Enum(["p2035", "p2040"])
    lineage = pl.DataFrame({
        "d": pl.Series("d", ["p2035", "p9999"], dtype=pl.Utf8),
        "d_other": pl.Series("d_other", ["p2035", "p2040"], dtype=pl.Utf8),
    })
    with pytest.raises(LineageFilterError, match="p9999"):
        _assert_lineage_castable(lineage, enum_d)


def test_assert_recourse_preconditions_raises_lineage_filter_error(
        monkeypatch):
    from flextool.engine_polars import _derived_branch

    monkeypatch.setattr(
        _derived_branch, "check_recourse_npv_preconditions",
        lambda *a, **k: ["synthetic violation for the test"],
    )
    with pytest.raises(LineageFilterError, match="preconditions violated"):
        _derived_branch.assert_recourse_npv_preconditions(None)
