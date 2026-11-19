"""The phase-progress (timing / memory) table rows: 3 significant figures, kB-TB units, fixed bar columns."""
from __future__ import annotations

import logging
import re

import pytest

from flextool.engine.solve._orchestration import _MemoryRecorder as M

GB = 1024.0


@pytest.mark.parametrize(("value", "text"), [
    (0.0, "0"), (0.0123, "0.01"), (0.123, "0.12"), (9.1, "9.1"), (16.94, "16.9"),
    (112.4, "112"), (544.0, "544"), (999.6, "1000"), (12345.0, "12300"), (10000.0, "10000"),
])
def test_three_significant_figures_in_at_most_four_digits_below_1000(value: float, text: str) -> None:
    assert M._fmt_sig3(value) == text


def test_memory_cells_keep_the_absolute_in_0_999_and_the_delta_in_its_unit() -> None:
    assert M._fmt_mem_cell(544, -5) == "     544 MB (-5)"
    assert M._fmt_mem_cell(16.9 * GB, -0.01 * GB) == " 16.9 GB (-0.01)"
    assert M._fmt_mem_cell(16.9 * GB, 1) == "    16.9 GB (+0)"          # negligible delta
    assert M._fmt_mem_cell(0.5, 0.5).strip() == "512 kB (+512)"
    assert M._fmt_mem_cell(999.7, 0).strip() == "0.98 GB (+0)"         # 1000 MB rounds into GB
    assert M._fmt_mem_cell(5 * GB * GB, 0).strip().startswith("5 TB")
    assert M._fmt_mem_cell(None, None) == " " * 15 + "-"
    assert all(len(M._fmt_mem_cell(mb, d)) == 16 for mb, d in [(544, -5), (16.9 * GB, -0.01 * GB), (0, 0)])


def test_time_cells_are_seconds_to_three_significant_figures() -> None:
    assert M._fmt_time_cell(0.123, None).strip() == "0.12s (-)"
    assert M._fmt_time_cell(9.1, 0.5).strip() == "9.1s (+0.5)"
    assert M._fmt_time_cell(112.4, 100.2).strip() == "112s (+100)"
    assert M._fmt_time_cell(10000, 10000) == "10000s (+10000)"


def test_the_target_row_and_fixed_bar_columns() -> None:
    row = M._fmt_row("Matrix built by polar-high", M._fmt_time_cell(10000, 10000),
                     [M._fmt_mem_cell(544, -5), M._fmt_mem_cell(16.9 * GB, -0.01 * GB), M._fmt_mem_cell(120, 10)])
    assert row == ("Matrix built by polar-high 10000s (+10000) |     544 MB (-5) | 16.9 GB (-0.01) "
                   "|    120 MB (+10)")
    short = M._fmt_row("Run start", M._fmt_time_cell(0.12, None),
                       [M._fmt_mem_cell(230, 230), M._fmt_mem_cell(8 * GB, 8 * GB), M._fmt_mem_cell(0, 0)])
    bars = [m.start() for m in re.finditer(r"\|", row)]
    assert [m.start() for m in re.finditer(r"\|", short)] == bars
    assert all(row[b - 1] == " " for b in bars)
    # Each cell between two bars is 17 characters wide.
    assert [b - a - 1 for a, b in zip(bars, bars[1:])] == [17, 17]


def test_a_long_label_stretches_the_lead_and_keeps_a_space_before_the_time() -> None:
    row = M._fmt_row("A label longer than the label column", M._fmt_time_cell(9.1, 0.5),
                     [M._fmt_mem_cell(230, 0)])
    assert row.startswith("A label longer than the label column 9.1s (+0.5) |")


def test_header_and_rows_printed_by_a_checkpoint(capsys: pytest.CaptureFixture[str]) -> None:
    rec = M(csv_path=None, enabled=False)
    log = logging.getLogger("test_phase_progress_rows")
    rec.checkpoint("run_start", log, user_label="Run start")
    rec.checkpoint("matrix", log, user_label="Matrix built by polar-high")
    header, first, second = capsys.readouterr().out.splitlines()
    assert re.fullmatch(r" +time \| +RSS memory \| +system memory \| +system swap", header)
    bars = [m.start() for m in re.finditer(r"\|", first)]
    assert [m.start() for m in re.finditer(r"\|", header)] == bars
    assert [m.start() for m in re.finditer(r"\|", second)] == bars
    assert re.match(r"Run start +\d[\d.]*s \(-\) \| ", first)
    assert re.match(r"Matrix built by polar-high +\d[\d.]*s \(\+[\d.]+\) \| ", second)
