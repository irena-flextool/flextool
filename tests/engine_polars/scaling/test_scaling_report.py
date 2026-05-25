"""Tests for engine_polars/scaling_report.py — write_scaling_report.

Runs the in-memory scaling reporter on work_all and work_network_all_tech
and checks:
  1. The output file is created and non-empty.
  2. All expected section headers are present in the file.
  3. Key diagnostic data from the ScaleTable appears in the report.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from flextool.engine_polars import load_flextool
from flextool.engine_polars.scaling import analyze_solve, clear_cache
from flextool.engine_polars.scaling_report import write_scaling_report

_SOLVE_ALL = "y2020_5week"
_SOLVE_NET = "y2020_2day_dispatch"

# Section headers that must appear in every scaling report (from scaling_report.py).
EXPECTED_SECTION_HEADERS = [
    "FlexTool scaling diagnostic report",
    "-- 2. Scaling decisions",
    "-- 3. Coefficient-family ranges",
    "-- 4. Bimodal coefficient distributions",
    "-- 5. Composite-scale mismatch",
    "-- 6. Near-duplicate parameter clusters",
    "-- 7. Escape-tier / slack activity",
    "-- 8. HiGHS matrix-range summary",
    "-- 8.5 Coefficient sources",
    "-- 9. Summary",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def flex_all(scenario_workdir):
    return load_flextool(scenario_workdir("all"))


@pytest.fixture(scope="module")
def flex_net(scenario_workdir):
    return load_flextool(scenario_workdir("network_all_tech"))


@pytest.fixture(autouse=True)
def _clear_scale_cache():
    clear_cache()
    yield
    clear_cache()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_report(solve_name, flex_data, tmp_path) -> tuple[Path, str]:
    """Run analyze_solve + write_scaling_report; return (path, text)."""
    table = analyze_solve(solve_name, flex_data, work_folder=tmp_path)
    solve_data_dir = tmp_path / "solve_data"
    out_path = write_scaling_report(
        scale_table=table,
        flex_data=flex_data,
        solve_data_dir=solve_data_dir,
        solve_name=solve_name,
        solution=None,         # no live HiGHS object in unit tests
        stdout_summary=False,  # suppress extra stdout noise in test output
    )
    return out_path, out_path.read_text()


# ---------------------------------------------------------------------------
# File-existence and non-empty tests
# ---------------------------------------------------------------------------


class TestReportFileOutput:
    def test_report_written_for_work_all(self, flex_all, tmp_path):
        out_path, _ = _run_report(_SOLVE_ALL, flex_all, tmp_path)
        assert out_path.exists(), f"report not written: {out_path}"

    def test_report_non_empty_for_work_all(self, flex_all, tmp_path):
        out_path, text = _run_report(_SOLVE_ALL, flex_all, tmp_path)
        assert len(text.strip()) > 0

    def test_report_written_for_work_net(self, flex_net, tmp_path):
        out_path, _ = _run_report(_SOLVE_NET, flex_net, tmp_path)
        assert out_path.exists()

    def test_report_non_empty_for_work_net(self, flex_net, tmp_path):
        out_path, text = _run_report(_SOLVE_NET, flex_net, tmp_path)
        assert len(text.strip()) > 0

    def test_report_filename_is_scaling_report_txt(self, flex_all, tmp_path):
        out_path, _ = _run_report(_SOLVE_ALL, flex_all, tmp_path)
        assert out_path.name == "scaling_report.txt"

    def test_report_in_solve_data_dir(self, flex_all, tmp_path):
        out_path, _ = _run_report(_SOLVE_ALL, flex_all, tmp_path)
        assert out_path.parent == tmp_path / "solve_data"


# ---------------------------------------------------------------------------
# Section-header presence tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("header", EXPECTED_SECTION_HEADERS)
def test_section_header_present_work_all(flex_all, tmp_path, header):
    _, text = _run_report(_SOLVE_ALL, flex_all, tmp_path)
    assert header in text, (
        f"Section header not found in report for work_all: {header!r}\n"
        f"Report excerpt:\n{text[:500]}"
    )


@pytest.mark.parametrize("header", EXPECTED_SECTION_HEADERS)
def test_section_header_present_work_net(flex_net, tmp_path, header):
    _, text = _run_report(_SOLVE_NET, flex_net, tmp_path)
    assert header in text, (
        f"Section header not found in report for work_network_all_tech: {header!r}"
    )


# ---------------------------------------------------------------------------
# Content sanity tests — key ScaleTable fields appear in the text
# ---------------------------------------------------------------------------


class TestReportContentWorkAll:
    def test_solve_name_in_report(self, flex_all, tmp_path):
        _, text = _run_report(_SOLVE_ALL, flex_all, tmp_path)
        assert _SOLVE_ALL in text

    def test_use_row_scaling_in_report(self, flex_all, tmp_path):
        table = analyze_solve(_SOLVE_ALL, flex_all, work_folder=tmp_path)
        clear_cache()
        _, text = _run_report(_SOLVE_ALL, flex_all, tmp_path)
        assert f"recommended={table.use_row_scaling}" in text

    def test_family_names_in_report(self, flex_all, tmp_path):
        _, text = _run_report(_SOLVE_ALL, flex_all, tmp_path)
        for family in ("entity_unitsize", "node_inflow", "vom_and_op_costs"):
            assert family in text, (
                f"Family name {family!r} not found in report text"
            )

    def test_near_duplicate_skipped_note(self, flex_all, tmp_path):
        """In the polars path, near-duplicate section always shows the
        'not available' placeholder rather than actual CSV scan results."""
        _, text = _run_report(_SOLVE_ALL, flex_all, tmp_path)
        assert "near-duplicate scan not available in the polars engine path" in text

    def test_summary_line_present(self, flex_all, tmp_path):
        """Report ends with one of three summary lines."""
        _, text = _run_report(_SOLVE_ALL, flex_all, tmp_path)
        assert any(
            phrase in text
            for phrase in (
                "Model well-scaled",
                "Model poorly scaled",
                "Model scaled acceptably",
            )
        ), "No summary line found in report"


class TestReportContentWorkNet:
    def test_solve_name_in_report(self, flex_net, tmp_path):
        _, text = _run_report(_SOLVE_NET, flex_net, tmp_path)
        assert _SOLVE_NET in text

    def test_family_names_in_report(self, flex_net, tmp_path):
        _, text = _run_report(_SOLVE_NET, flex_net, tmp_path)
        for family in ("entity_unitsize", "node_inflow", "vom_and_op_costs"):
            assert family in text

    def test_summary_line_present(self, flex_net, tmp_path):
        _, text = _run_report(_SOLVE_NET, flex_net, tmp_path)
        assert any(
            phrase in text
            for phrase in (
                "Model well-scaled",
                "Model poorly scaled",
                "Model scaled acceptably",
            )
        )

    def test_report_ends_with_newline(self, flex_net, tmp_path):
        _, text = _run_report(_SOLVE_NET, flex_net, tmp_path)
        assert text.endswith("\n"), "Report file should end with a newline"


# ---------------------------------------------------------------------------
# Returns-Path check
# ---------------------------------------------------------------------------


def test_coefficient_sources_section_present_work_all(flex_all, tmp_path):
    """Section 8.5 (coefficient-source diagnostic) must always appear.

    With ``solution=None`` (as in unit tests) the per-category lists are
    not populated, so the section emits the "unavailable" notice — which
    is still useful evidence that the section was wired in correctly.
    """
    _, text = _run_report(_SOLVE_ALL, flex_all, tmp_path)
    assert "-- 8.5 Coefficient sources" in text, (
        f"Section 8.5 header missing from report:\n{text[:1000]}"
    )
    # When no live solver instance is supplied, an "unavailable" line must
    # be emitted so users can tell the section was reached but skipped.
    assert "coefficient sources unavailable" in text, (
        "Expected 'coefficient sources unavailable' notice when "
        "solution=None"
    )


def test_write_scaling_report_returns_path_object(flex_all, tmp_path):
    table = analyze_solve(_SOLVE_ALL, flex_all, work_folder=tmp_path)
    solve_data_dir = tmp_path / "solve_data"
    result = write_scaling_report(
        scale_table=table,
        flex_data=flex_all,
        solve_data_dir=solve_data_dir,
        solve_name=_SOLVE_ALL,
        solution=None,
        stdout_summary=False,
    )
    assert isinstance(result, Path)
