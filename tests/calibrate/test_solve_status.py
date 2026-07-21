"""Tests for the calibrator's resilient solve-success detector.

Output files are synthesised with the *real* FlexTool parquet writer
(:func:`flextool.lean_parquet.write_lean_parquet`) at the exact on-disk
shape the detector reads (``output_parquet/<subdir>/<key>.parquet``), so
the tests exercise the same footer-metadata read path the calibrator will
hit — without paying for a real solve.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import pytest

from flextool.calibrate import (
    OutputCheck,
    SolveOutcome,
    assess_solve,
    default_required_outputs,
)
from flextool.lean_parquet import write_lean_parquet


# --- fixtures / helpers -----------------------------------------------------


def _slack_frame(n_rows: int = 3) -> pd.DataFrame:
    """A small frame shaped like ``node_slack_up_d_e`` (period × node)."""
    idx = pd.Index([f"y20{25 + i}" for i in range(n_rows)], name="period")
    return pd.DataFrame({"nodeA": [0.0] * n_rows, "nodeB": [1.0] * n_rows},
                        index=idx)


def _cost_frame(n_rows: int = 2) -> pd.DataFrame:
    """A small frame shaped like ``cost_node_discounted_d_ec``."""
    idx = pd.MultiIndex.from_tuples(
        [("y2025", "nodeA", "upward slack penalty")][:n_rows]
        + [("y2026", "nodeB", "operation")][: max(0, n_rows - 1)],
        names=["period", "node", "class"],
    )
    return pd.DataFrame({"value": [3.14, 2.71][:n_rows]}, index=idx)


def _write_outputs(parquet_dir: Path, *, slack_rows: int = 3,
                   cost_rows: int = 2) -> None:
    """Write both required parquets into *parquet_dir* at the real shape."""
    parquet_dir.mkdir(parents=True, exist_ok=True)
    write_lean_parquet(
        _slack_frame(slack_rows), parquet_dir / "node_slack_up_d_e.parquet"
    )
    write_lean_parquet(
        _cost_frame(cost_rows),
        parquet_dir / "cost_node_discounted_d_ec.parquet",
    )


@pytest.fixture
def parquet_dir(tmp_path: Path) -> Path:
    return tmp_path / "output_parquet" / "MyScenario"


# --- default / resolution ---------------------------------------------------


def test_default_required_outputs_resolve_via_registry():
    """The default is the single robust slack gate, resolved via the registry.

    ``cost_node_discounted_d_ec`` is intentionally NOT a default requirement
    (out_costs.py can legitimately skip it), so the always-present
    ``node_slack_up_d_e`` is the sole default success signal.
    """
    defaults = default_required_outputs()
    assert defaults == ("node_slack_up_d_e.parquet",)
    # And it is actually a registered output (guards a rename).
    from flextool.process_outputs._output_meta import OUTPUT_TRANSFORM

    assert "node_slack_up_d_e" in OUTPUT_TRANSFORM


# --- happy path -------------------------------------------------------------


def test_complete_outputs_exit_zero_succeeds(parquet_dir: Path):
    _write_outputs(parquet_dir)
    outcome = assess_solve(parquet_dir, exit_code=0)
    assert isinstance(outcome, SolveOutcome)
    assert outcome.succeeded is True
    assert outcome.outputs_complete is True
    assert outcome.exit_code == 0
    assert all(isinstance(c, OutputCheck) and c.ok for c in outcome.per_solve)
    assert "exit code 0" in outcome.reason


def test_complete_outputs_no_exit_code_succeeds(parquet_dir: Path):
    _write_outputs(parquet_dir)
    outcome = assess_solve(parquet_dir)
    assert outcome.succeeded is True
    assert outcome.exit_code is None


# --- the writer-KeyError case: nonzero exit, complete outputs ---------------


def test_complete_outputs_nonzero_exit_overrides_to_success(parquet_dir: Path):
    # The override is only safe when freshness is verified, so started_at
    # must be supplied (files are written 'now', after started).
    started = time.time() - 5.0
    _write_outputs(parquet_dir)
    outcome = assess_solve(parquet_dir, exit_code=1, started_at=started)
    assert outcome.succeeded is True
    assert outcome.outputs_complete is True
    assert outcome.exit_code == 1
    # Reason must call out the overridden exit, and name the writer-crash
    # rationale so an operator understands why a nonzero exit passed.
    assert "OVERRIDDEN" in outcome.reason
    assert "Shared-alternative write failed" in outcome.reason


def test_nonzero_exit_without_started_at_fails(parquet_dir: Path):
    """The closed stale-masking hole.

    Complete outputs + nonzero exit but NO started_at: freshness is
    unverifiable, so the override cannot be made safely — a genuinely
    failed solve could have left a prior run's outputs lingering.  Must
    FAIL rather than override to success.
    """
    _write_outputs(parquet_dir)
    outcome = assess_solve(parquet_dir, exit_code=1)  # no started_at
    assert outcome.succeeded is False
    # Outputs ARE complete; the refusal is about freshness, not completeness.
    assert outcome.outputs_complete is True
    assert outcome.exit_code == 1
    assert "freshness" in outcome.reason.lower()
    assert "started_at" in outcome.reason
    assert "OVERRIDDEN" not in outcome.reason


# --- genuine failures: missing / empty / corrupt outputs --------------------


def test_missing_required_output_fails(parquet_dir: Path):
    # Explicitly require both tables; delete one → completeness fails and
    # the specific missing file is named.
    _write_outputs(parquet_dir)
    (parquet_dir / "cost_node_discounted_d_ec.parquet").unlink()
    outcome = assess_solve(
        parquet_dir,
        exit_code=0,
        required_outputs=["node_slack_up_d_e", "cost_node_discounted_d_ec"],
    )
    assert outcome.succeeded is False
    assert outcome.outputs_complete is False
    missing = [c for c in outcome.per_solve if not c.ok]
    assert [c.filename for c in missing] == [
        "cost_node_discounted_d_ec.parquet"
    ]
    assert missing[0].present is False
    assert "cost_node_discounted_d_ec.parquet: missing" in outcome.reason


def test_empty_required_output_fails(parquet_dir: Path):
    # A present-but-zero-row parquet must NOT count as a produced result.
    _write_outputs(parquet_dir, slack_rows=0)
    outcome = assess_solve(parquet_dir, exit_code=0)
    assert outcome.succeeded is False
    assert outcome.outputs_complete is False
    empty = next(
        c for c in outcome.per_solve
        if c.filename == "node_slack_up_d_e.parquet"
    )
    assert empty.present is True
    assert empty.num_rows == 0
    assert empty.ok is False


def test_corrupt_required_output_fails(parquet_dir: Path):
    _write_outputs(parquet_dir)
    # Truncate one parquet so the footer is unreadable.
    (parquet_dir / "node_slack_up_d_e.parquet").write_bytes(b"not a parquet")
    outcome = assess_solve(parquet_dir, exit_code=0)
    assert outcome.succeeded is False
    corrupt = next(
        c for c in outcome.per_solve
        if c.filename == "node_slack_up_d_e.parquet"
    )
    assert corrupt.present is True
    assert corrupt.num_rows is None
    assert corrupt.ok is False


def test_missing_output_dir_fails(tmp_path: Path):
    outcome = assess_solve(tmp_path / "does_not_exist", exit_code=0)
    assert outcome.succeeded is False
    assert outcome.outputs_complete is False
    assert "does not exist" in outcome.reason


# --- stale-output guard (the detectable prior-run masking case) -------------


def test_stale_outputs_nonzero_exit_fails_with_started_at(parquet_dir: Path):
    """A genuine failure that left a *previous* run's files in place.

    write_outputs is skipped on a failed cascade and does NOT empty the
    parquet dir, so stale complete files remain.  With ``started_at`` set to
    'now' and the files back-dated, the detector must catch the staleness
    and fail rather than treating exit!=0 + complete as the writer-crash
    success.
    """
    _write_outputs(parquet_dir)
    old = time.time() - 3600.0
    for fn in default_required_outputs():
        os.utime(parquet_dir / fn, (old, old))
    started = time.time()
    outcome = assess_solve(
        parquet_dir, exit_code=1, started_at=started
    )
    assert outcome.succeeded is False
    assert outcome.outputs_complete is False
    stale = [c for c in outcome.per_solve if not c.ok]
    assert stale, "expected stale outputs to be flagged"
    assert all(c.present and c.num_rows and not c.fresh for c in stale)
    assert "STALE" in outcome.reason


def test_fresh_outputs_nonzero_exit_succeeds_with_started_at(parquet_dir: Path):
    """The true writer-crash case: fresh, complete outputs + nonzero exit."""
    started = time.time() - 5.0  # subprocess launched a moment ago
    _write_outputs(parquet_dir)  # written 'now', i.e. after started
    outcome = assess_solve(
        parquet_dir, exit_code=1, started_at=started
    )
    assert outcome.succeeded is True
    assert outcome.outputs_complete is True
    assert "OVERRIDDEN" in outcome.reason
    assert "verified fresh" in outcome.reason


# --- custom required_outputs (keys and filenames both accepted) -------------


def test_custom_required_outputs_as_keys_and_filenames(parquet_dir: Path):
    _write_outputs(parquet_dir)
    # Only require the slack file, expressed once as a key, once as a name.
    for spec in (["node_slack_up_d_e"], ["node_slack_up_d_e.parquet"]):
        outcome = assess_solve(parquet_dir, exit_code=0, required_outputs=spec)
        assert outcome.succeeded is True
        assert len(outcome.per_solve) == 1
        assert outcome.per_solve[0].filename == "node_slack_up_d_e.parquet"


def test_unknown_required_key_raises(parquet_dir: Path):
    _write_outputs(parquet_dir)
    with pytest.raises(KeyError):
        assess_solve(
            parquet_dir, exit_code=0, required_outputs=["not_a_real_output"]
        )
