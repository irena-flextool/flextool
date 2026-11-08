"""``results_usable`` / ``unusable_reason``: the one rule for whether a run folder's results may be offered (D5, O5).

Usable: a folder without a run record (legacy; the caller still checks for parquet files), or a run that
``succeeded`` and wrote its parquet.  Anything else is not, whatever an earlier run left in the folder.
"""
from __future__ import annotations

import os
import time

import pytest

from flextool.run_record import (
    RUN_STATUS_FILE,
    UNUSABLE_CODES,
    read_status_file,
    results_usable,
    unusable_code,
    unusable_reason,
)
from flextool.run_record import _owner


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


@pytest.mark.parametrize("state, parquet_state, effective, code", [
    ("succeeded", "written", None, None),
    ("running", "previous_run", "running", "running"),
    ("running", "partial", "unknown", "running"),
    ("running", "previous_run", "abandoned", "abandoned"),
    ("failed", "previous_run", None, "failed"),
    ("interrupted", "partial", None, "failed"),
    ("killed", "previous_run", None, "failed"),
    ("crashed", "none", None, "failed"),
    ("succeeded", "previous_run", None, "previous_run"),
    ("succeeded", "partial", None, "partial"),
    ("succeeded", "not_requested", None, "no_parquet"),
    ("succeeded", "skipped", None, "no_parquet"),
    ("succeeded", "none", None, "no_parquet"),
])
def test_unusable_code(state, parquet_state, effective, code):
    status = _status(state, parquet_state)
    assert unusable_code(status, effective) == code
    assert (code is None) == results_usable(status)


def test_legacy_folder_has_no_unusable_code():
    assert unusable_code(None) is None


# ---------------------------------------------------------------------------------------------------------
# A record that cannot be read, or has the wrong shape, is never usable (and never raises)


@pytest.mark.parametrize("status", [
    {"state": "succeeded", "outputs": [1]},
    {"state": 3, "outputs": {"parquet_state": "written"}},
    {"state": "failed", "error": "x"},
    {"state": "succeeded", "outputs": {"parquet_state": "written"}, "solves": [1]},
    {"state": "succeeded", "outputs": {"parquet_state": "written"}, "input": "x"},
    {"state": "succeeded", "outputs": {"parquet_state": "written"}, "versions": [1]},
    {"state": "succeeded", "outputs": {"parquet_state": "written"}, "warnings": ["w"]},
], ids=["outputs_list", "state_int", "error_str", "solves_items", "input_str", "versions_list", "warnings_items"])
def test_malformed_shapes_are_not_usable(status):
    assert results_usable(status) is False
    assert unusable_code(status) == "malformed"
    assert unusable_reason(status) == "the run record is damaged; see the run log"


def test_unreadable_record_is_not_usable_also_as_a_copy():
    import copy
    import json

    unreadable = {"state": "unreadable"}
    for s in (unreadable, dict(unreadable), copy.deepcopy(unreadable), json.loads(json.dumps(unreadable))):
        assert results_usable(s) is False
        assert unusable_code(s) == "unreadable"
        assert unusable_reason(s) == ("the run record could not be read (it may be being written); try again in "
                                      "a moment")
    assert {"unreadable", "malformed"} <= set(UNUSABLE_CODES)


def _old(path):
    then = time.time() - 60
    os.utime(path, (then, then))


def test_strict_read_classifies(tmp_path):
    folder = tmp_path / "base"
    folder.mkdir()
    path = folder / RUN_STATUS_FILE
    assert read_status_file(folder) is None and read_status_file(folder, strict=True) is None
    cases = [
        ('{"format": 1, "state": "failed", "outputs": {"parq', False, "unreadable"),   # being written
        ('{"format": 1, "state": "failed", "outputs": {"parq', True, "malformed"),     # cut short long ago
        ("[1]", False, "malformed"),
        ('{"state": "succeeded", "outputs": [1]}', False, "malformed"),
    ]
    for text, old, state in cases:
        path.write_text(text, encoding="utf-8")
        if old:
            _old(path)
        assert read_status_file(folder, strict=True) == {"state": state}, text
    path.write_bytes(b'{"state": "succ\xff\xfe"}')
    assert read_status_file(folder, strict=True) == {"state": "unreadable"}
    _old(path)
    assert read_status_file(folder, strict=True) == {"state": "malformed"}
    # The lenient read keeps its old answers (None for anything it cannot use as a dict).
    path.write_text("[1]", encoding="utf-8")
    assert read_status_file(folder) is None


def test_strict_read_of_a_locked_file_is_unreadable(tmp_path, monkeypatch):
    folder = tmp_path / "base"
    folder.mkdir()
    (folder / RUN_STATUS_FILE).write_text('{"state": "succeeded"}', encoding="utf-8")
    def locked(*args, **kwargs):
        raise PermissionError(13, "The process cannot access the file because it is being used")

    monkeypatch.setattr(_owner, "open", locked, raising=False)
    assert read_status_file(folder, strict=True) == {"state": "unreadable"}
    assert read_status_file(folder) is None
