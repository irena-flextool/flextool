"""``_run.log``: the tail cap, the marker and the pipe owner's choice of marker."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from flextool.run_record import RunLogOwner, RunLogWriter, format_marker, parse_marker, write_status_atomic
from flextool.run_record._owner import RUN_LOG_FILE, write_log_header

HEAD = 1000
TAIL = 2000
HEADER = "# FlexTool run r1 started 2026-11-06T10:00:00+02:00"
_OMISSION = re.compile(rb"\[\.\.\. (\d+) bytes of log omitted \.\.\.\]\n")


def test_compaction_keeps_head_and_tail_and_counts_every_dropped_byte(tmp_path):
    path = tmp_path / RUN_LOG_FILE
    write_log_header(path, "r1", "2026-11-06T10:00:00+02:00")
    original = (HEADER + "\n").encode("utf-8")
    writer = RunLogWriter(path, head=HEAD, tail=TAIL)
    omitted_values = set()
    # 400 numbered lines of 20 bytes: 8000 bytes, well past HEAD + 2*TAIL = 5000 twice over.
    for i in range(400):
        line = f"line {i:05d} ...ä...".ljust(18, ".")  # ``ä`` is 2 bytes in UTF-8
        writer.write_line(line)
        original += (line + "\n").encode("utf-8")
        if writer.omitted_bytes:
            omitted_values.add(writer.omitted_bytes)
    writer.close()
    assert len(omitted_values) >= 2, "expected at least two compactions"

    data = path.read_bytes()
    assert len(data) <= HEAD + 2 * TAIL
    assert data.splitlines()[0].decode("utf-8") == HEADER
    found = list(_OMISSION.finditer(data))
    assert len(found) == 1
    omission = found[0]
    head_part, tail_part = data[:omission.start()], data[omission.end():]
    # The head is a prefix of the full log cut at a newline, the tail a suffix starting at a line (at most TAIL
    # bytes at the last compaction, plus what was written after it).
    assert original.startswith(head_part) and head_part.endswith(b"\n") and len(head_part) <= HEAD
    assert original.endswith(tail_part) and len(tail_part) <= 2 * TAIL
    assert original[len(original) - len(tail_part) - 1:len(original) - len(tail_part)] == b"\n"
    dropped = len(original) - len(head_part) - len(tail_part)
    assert int(omission.group(1)) == dropped


def test_marker_round_trips_a_non_ascii_folder():
    folder = Path("/tmp") / "projektit ä" / "output_parquet" / "ö_1"
    marker = format_marker(folder, "abc123")
    assert marker.isascii()
    assert parse_marker(marker + "\n") == (folder, "abc123")
    assert parse_marker("Solve start: y2030, 1/1") is None


def _run_folder(tmp_path: Path, name: str, run_id: str) -> Path:
    folder = tmp_path / name
    folder.mkdir()
    write_status_atomic(folder, {"run_id": run_id, "state": "succeeded"})
    write_log_header(folder / RUN_LOG_FILE, run_id, "2026-11-06T10:00:00+02:00")
    return folder


def test_owner_takes_only_the_first_marker(tmp_path):
    first = _run_folder(tmp_path, "first", "r1")
    second = _run_folder(tmp_path, "second", "r2")
    owner = RunLogOwner(launcher="test", preamble=["$ flextool-run input.sqlite"])
    assert owner.feed("before the marker\n") is True
    assert owner.feed(format_marker(first, "r1") + "\n") is False
    assert owner.feed("after the marker\n") is True
    second_marker = format_marker(second, "r2")
    assert owner.feed(second_marker + "\n") is True
    assert owner.close(0) == first
    assert (first / RUN_LOG_FILE).read_text(encoding="utf-8").splitlines() == [
        "# FlexTool run r1 started 2026-11-06T10:00:00+02:00",
        "$ flextool-run input.sqlite",
        "before the marker",
        "after the marker",
        second_marker,
    ]
    assert (second / RUN_LOG_FILE).read_text(encoding="utf-8").splitlines() == [
        "# FlexTool run r2 started 2026-11-06T10:00:00+02:00",
    ]


# ---------------------------------------------------------------------------------------------------------
# Another run's writer: a takeover truncates the log; the old writer must stop instead of writing at its offset


def test_writer_stops_after_truncation(tmp_path):
    path = tmp_path / RUN_LOG_FILE
    write_log_header(path, "A", "2026-11-06T10:00:00+02:00")
    a = RunLogWriter(path, head=HEAD, tail=TAIL)
    for i in range(3):
        a.write_line(f"A line {i}")
    write_log_header(path, "B", "2026-11-06T10:05:00+02:00")  # run B takes the folder over
    b = RunLogWriter(path, head=HEAD, tail=TAIL)
    b.write_line("B line 0")
    b.write_line("B line 1")
    a.write_line("A late")
    b.close()
    a.close()
    data = path.read_bytes()
    assert b"\x00" not in data
    assert b"A late" not in data
    assert data.decode("utf-8").splitlines() == ["# FlexTool run B started 2026-11-06T10:05:00+02:00", "B line 0",
                                                 "B line 1"]
    assert a.taken_over and not b.taken_over


def test_writer_stops_when_the_other_run_grew_past_its_offset(tmp_path):
    path = tmp_path / RUN_LOG_FILE
    write_log_header(path, "A", "2026-11-06T10:00:00+02:00")
    a = RunLogWriter(path, head=HEAD, tail=TAIL)
    a.write_line("A line 0")
    write_log_header(path, "B", "2026-11-06T10:05:00+02:00")
    b = RunLogWriter(path, head=HEAD, tail=TAIL)
    for i in range(10):
        b.write_line(f"B line {i} with some more text")
    a.write_line("A late")
    a.write_omission(5)
    b.close()
    a.close()
    lines = path.read_bytes().decode("utf-8").splitlines()
    assert lines == ["# FlexTool run B started 2026-11-06T10:05:00+02:00"] + [
        f"B line {i} with some more text" for i in range(10)]
    assert a.taken_over


def test_own_compactions_never_read_as_a_takeover(tmp_path):
    path = tmp_path / RUN_LOG_FILE
    write_log_header(path, "A", "2026-11-06T10:00:00+02:00")
    writer = RunLogWriter(path, head=64, tail=64)
    for i in range(500):
        writer.write_line(f"line {i:04d}")
        if i % 50 == 0:
            writer.write_omission(3)
    assert not writer.taken_over and not writer.closed
    writer.close()
    assert path.read_bytes().decode("utf-8").splitlines()[-1] == "line 0499"


def test_a_failed_compaction_closes_without_reading_as_a_takeover(tmp_path):
    path = tmp_path / RUN_LOG_FILE
    write_log_header(path, "A", "2026-11-06T10:00:00+02:00")
    writer = RunLogWriter(path, head=64, tail=64)

    class FailingTruncate:
        def __init__(self, fh):
            self._fh = fh

        def truncate(self, *args):
            raise OSError(28, "No space left on device")

        def __getattr__(self, name):
            return getattr(self._fh, name)

    writer._fh = FailingTruncate(writer._fh)
    with pytest.raises(OSError):
        for i in range(100):
            writer.write_line(f"line {i:04d}")
    assert writer.closed and not writer.taken_over
