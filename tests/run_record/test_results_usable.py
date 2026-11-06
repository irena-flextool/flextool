"""``results_usable`` / ``unusable_reason``: the one rule for whether a run folder's results may be offered (D5, O5).

Usable: a folder without a run record (legacy; the caller still checks for parquet files), or a run that
``succeeded`` and wrote its parquet.  Anything else is not, whatever an earlier run left in the folder.
"""
from __future__ import annotations

import pytest

from flextool.run_record import results_usable, unusable_reason


def _status(state: str, parquet_state: str, message: str | None = None) -> dict:
    return {
        "state": state,
        "error": {"type": "X", "message": message} if message else None,
        "outputs": {"parquet_state": parquet_state},
    }


def test_legacy_folder_without_record_is_usable():
    assert results_usable(None) is True
    assert unusable_reason(None) is None


def test_succeeded_run_that_wrote_parquet_is_usable():
    status = _status("succeeded", "written")
    assert results_usable(status) is True
    assert unusable_reason(status) is None


@pytest.mark.parametrize("parquet_state", ["not_requested", "skipped", "none"])
def test_succeeded_run_without_parquet_of_its_own_is_not_usable(parquet_state):
    status = _status("succeeded", parquet_state)
    assert results_usable(status) is False
    assert "no parquet" in unusable_reason(status)


@pytest.mark.parametrize("state, word", [
    ("failed", "failed"),
    ("running", "running"),
    ("killed", "killed"),
    ("crashed", "crashed"),
    ("interrupted", "interrupted"),
])
def test_unfinished_or_failed_run_is_not_usable_even_with_earlier_parquet(state, word):
    status = _status(state, "previous_run")
    assert results_usable(status) is False
    assert word in unusable_reason(status)


def test_failed_reason_carries_the_error_message():
    status = _status("failed", "previous_run", message="solve for y2030 is infeasible")
    assert unusable_reason(status) == "latest run failed: solve for y2030 is infeasible"
