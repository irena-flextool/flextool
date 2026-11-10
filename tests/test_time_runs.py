"""``flextool-time-runs``: phase split of ``timings.csv`` and one real timed run."""
from __future__ import annotations

import csv
import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from flextool.cli.cmd_time_runs import main, phase_seconds

_HEADER = ["phase", "subphase", "solve", "roll_index", "seconds", "started_at_iso", "cumulative_s"]


def _rows(*rows: tuple[str, str, float]) -> list[dict[str, str]]:
    # cumulative_s is deliberately nonsense: the split must not read it.
    return [dict(zip(_HEADER, (p, s, "", "", f"{sec}", "", "999"))) for p, s, sec in rows]


def test_phase_split_from_whitelisted_rows() -> None:
    """Hand-summed buckets; ``warm_used`` (a 1.0 flag) and ``handoff_part`` (inside ``handoff``) are not added."""
    rows = _rows(
        ("read_input", "", 0.5),
        ("preprocessing_global", "", 0.01),
        ("roll_setup", "", 0.02), ("roll_setup", "", 0.03),
        ("per_iter", "preprocessing", 0.1), ("per_iter", "preprocessing", 0.2),
        ("per_iter", "lp_build", 0.3), ("per_iter", "lp_build", 0.4),
        ("per_iter", "solve", 0.05), ("per_iter", "solve", 0.15),
        ("per_iter", "warm_used", 0.0), ("per_iter", "warm_used", 1.0),
        ("handoff_part", "write_outputs_for_solve", 0.25), ("handoff_part", "write_outputs_for_solve", 0.25),
        ("per_iter", "handoff", 0.3), ("per_iter", "handoff", 0.3),
        ("all_solves", "", 3.0),
        ("write_outputs", "Read configuration files", 0.2),
        ("write_outputs", "Computed plot plans", 0.4),
        ("write_outputs", "Wrote dispatch metadata", 0.1),
        ("write_outputs", "Plotted figures", 0.6),
        ("write_outputs", "total", 1.5),
        ("total", "", 4.6),
    )
    got = phase_seconds(rows, wall_s=6.0)
    expected = {"startup": 1.4, "read_input": 0.5, "preprocess": 1.06, "solve": 0.2, "outputs": 1.0,
                "viewer": 0.5, "plots": 0.6, "untimed": 0.74}
    assert got == pytest.approx(expected)


def test_phase_split_of_a_run_that_stopped() -> None:
    """No ``total`` row: start-up and untimed are unknown, the rows that are there still count."""
    got = phase_seconds(_rows(("read_input", "", 0.5), ("per_iter", "solve", 0.05)), wall_s=2.0)
    assert got["startup"] is None and got["untimed"] is None
    assert got["read_input"] == pytest.approx(0.5)
    assert got["solve"] == pytest.approx(0.05)
    assert got["outputs"] == 0.0
    assert phase_seconds([], wall_s=1.0)["read_input"] is None


def test_one_example_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``coal`` from the examples: one table row, phases from the run's timings.csv, over a tiny target."""
    monkeypatch.chdir(tmp_path)
    out_csv = tmp_path / "times.csv"
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(["--source", "templates_examples", "--scenario", "coal", "--target", "0.001",
                     "--strict", "--csv", str(out_csv)])
    text = buf.getvalue()
    assert code == 2, text  # ran fine, over the target, --strict
    assert text.startswith("Machine: ")
    assert any(line.startswith("! coal ") and line.rstrip().endswith("optimal") for line in text.splitlines())
    with open(out_csv, newline="") as f:
        (row,) = list(csv.DictReader(f))
    wall = float(row["wall_s"])
    assert row["scenario"] == "coal" and row["ok"] == "1" and row["over_target"] == "1"
    assert float(row["read_input_s"]) > 0
    assert float(row["preprocess_s"]) > 0 and float(row["solve_s"]) > 0
    assert float(row["plots_s"]) == 0.0  # plots off by default
    assert 0 <= float(row["untimed_s"]) < 0.2 * wall
    parts = sum(float(row[f"{k}_s"]) for k in
                ("startup", "read_input", "preprocess", "solve", "outputs", "viewer", "plots", "untimed"))
    assert parts == pytest.approx(wall, abs=0.01)
