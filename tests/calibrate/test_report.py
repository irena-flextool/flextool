"""Unit tests for the pure calibration report renderer (C2).

Construct a synthetic :class:`CalibResult` directly — no solve — and assert
that :func:`write_report` produces the two CSVs with the pinned columns, the
right rows, and deterministic order, and that :func:`format_summary` mentions
the convergence status, the final adders, and the flagged node under the
resource-capped section.
"""

from __future__ import annotations

import csv
from pathlib import Path

from flextool.calibrate._loop import CalibResult, IterRecord
from flextool.calibrate._report import (
    LONG_COLUMNS,
    LONG_FILENAME,
    SUMMARY_COLUMNS,
    SUMMARY_FILENAME,
    format_summary,
    write_report,
)


def _synthetic_result() -> CalibResult:
    """A two-iteration run: 'west' keeps shedding and is flagged capped."""
    baseline = IterRecord(
        iteration=0,
        adders={},
        residual={"west": 100.0, "east": 10.0},
        curtailment={"west": 5.0, "east": 0.0},
        penalty_total=2.0,
        penalty_by_node={"west": 1.5, "east": 0.5},
    )
    step1 = IterRecord(
        iteration=1,
        adders={"west": 50.0, "east": 5.0},
        residual={"west": 80.0, "east": 0.0},
        curtailment={"west": 40.0, "east": 0.0},
        penalty_total=1.2,
        penalty_by_node={"west": 1.2, "east": 0.0},
    )
    return CalibResult(
        converged=False,
        iterations_run=2,
        final_adders={"west": 50.0, "east": 5.0},
        trajectory=[baseline, step1],
        guard_flagged_nodes=["west"],
    )


def test_write_report_long_csv(tmp_path: Path):
    result = _synthetic_result()
    long_path, summary_path = write_report(result, out_dir=tmp_path)

    assert long_path == tmp_path / LONG_FILENAME
    assert summary_path == tmp_path / SUMMARY_FILENAME

    with long_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))

    assert rows[0] == LONG_COLUMNS
    body = rows[1:]
    # 2 iterations x 2 nodes, ordered iterations ascending then node sorted.
    assert [(r[0], r[1]) for r in body] == [
        ("0", "east"),
        ("0", "west"),
        ("1", "east"),
        ("1", "west"),
    ]
    # Spot-check the iteration-0 west row content.
    west0 = body[1]
    assert west0[2] == "100.0"  # residual_mwh
    assert west0[3] == "0.0"  # adder_mwh (baseline snapshot)
    assert west0[4] == "5.0"  # curtailment_mwh
    assert west0[5] == "1.5"  # penalty_meur
    assert west0[6] == "True"  # flagged (west is in the final guard set)
    # east is never flagged.
    east0 = body[0]
    assert east0[6] == "False"


def test_write_report_summary_csv(tmp_path: Path):
    result = _synthetic_result()
    _, summary_path = write_report(result, out_dir=tmp_path)

    with summary_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))

    assert rows[0] == SUMMARY_COLUMNS
    body = rows[1:]
    assert [r[0] for r in body] == ["0", "1"]

    # iteration 0: total unserved 110, penalty 2.0, both nodes shed, 1 flagged.
    assert body[0][1] == "110.0"
    assert body[0][2] == "2.0"
    assert body[0][3] == "2"  # n_shedding
    assert body[0][4] == "1"  # n_flagged (run-level count)
    # body[*][5] is solve_seconds (0.0 for synthetic records).
    assert body[0][6] == "False"  # not converged
    # iteration 1: only west sheds now.
    assert body[1][1] == "80.0"
    assert body[1][3] == "1"
    assert body[1][6] == "False"


def test_write_report_creates_out_dir(tmp_path: Path):
    nested = tmp_path / "a" / "b" / "report"
    result = _synthetic_result()
    paths = write_report(result, out_dir=nested)
    assert all(p.exists() for p in paths)


def test_format_summary_contents():
    result = _synthetic_result()
    text = format_summary(result)

    assert "NOT CONVERGED" in text
    # Final adders appear, largest (west) listed.
    assert "west" in text
    assert "east" in text
    # Resource-capped section names the flagged node.
    assert "Resource-capped" in text
    lines = text.splitlines()
    capped_idx = next(i for i, ln in enumerate(lines) if "Resource-capped" in ln)
    capped_block = "\n".join(lines[capped_idx:])
    assert "west" in capped_block
    # Baseline->final unserved figures present.
    assert "110" in text and "80" in text


def test_format_summary_converged_and_no_flags():
    baseline = IterRecord(
        iteration=0,
        adders={},
        residual={"west": 100.0},
        curtailment={"west": 0.0},
        penalty_total=1.0,
        penalty_by_node={"west": 1.0},
    )
    final = IterRecord(
        iteration=1,
        adders={"west": 40.0},
        residual={"west": 0.0},
        curtailment={"west": 0.0},
        penalty_total=0.0,
        penalty_by_node={"west": 0.0},
    )
    result = CalibResult(
        converged=True,
        iterations_run=2,
        final_adders={"west": 40.0},
        trajectory=[baseline, final],
        guard_flagged_nodes=[],
        stop_reason="converged",
    )
    text = format_summary(result)
    assert "CONVERGED" in text and "NOT CONVERGED" not in text
    assert "MODULO" not in text
    # No flagged nodes -> the resource-capped section says none.
    lines = text.splitlines()
    capped_idx = next(i for i, ln in enumerate(lines) if "Resource-capped" in ln)
    assert lines[capped_idx + 1].strip() == "none"


def test_format_summary_stalled_converged_modulo():
    """A stalled run renders the 'converged modulo resource-capped' status."""
    baseline = IterRecord(
        iteration=0,
        adders={},
        residual={"west": 100.0},
        curtailment={"west": 0.0},
        penalty_total=1.0,
        penalty_by_node={"west": 1.0},
    )
    stalled = IterRecord(
        iteration=1,
        adders={"west": 40.0},
        residual={"west": 60.0},
        curtailment={"west": 30.0},
        penalty_total=0.8,
        penalty_by_node={"west": 0.8},
    )
    result = CalibResult(
        converged=False,
        iterations_run=2,
        final_adders={"west": 40.0},
        trajectory=[baseline, stalled],
        guard_flagged_nodes=["west"],
        stop_reason="stalled",
    )
    text = format_summary(result)
    # A stalled run is "converged modulo the resource-capped nodes" — it must
    # NOT read as a plain budget-exhausted failure, and it must NOT tell the
    # operator to just raise --iterations (the demand-margin lever is spent).
    assert "MODULO" in text
    assert "NOT CONVERGED" not in text
    assert "modulo" in text.lower()
    assert "exhausted" in text.lower()
    assert "raise --iterations" not in text
    # The residual is tied to the resource-capped, guard-flagged node.
    lines = text.splitlines()
    capped_idx = next(i for i, ln in enumerate(lines) if "Resource-capped" in ln)
    capped_block = "\n".join(lines[capped_idx:])
    assert "west" in capped_block
