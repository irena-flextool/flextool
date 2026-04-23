"""Unit tests for ``flextool.flextoolrunner.scaling_report`` (Agent 10).

Three focus areas:

1. Section rendering with fake :class:`ScaleTable` inputs — make sure
   every section appears and the locked recommendation text is
   reproduced verbatim when a composite-scale mismatch fires.
2. Bimodal detector on synthetic log10 distributions — tight clusters
   and uniform distributions must NOT be flagged; clean two-cluster
   inputs MUST be flagged.
3. Composite-scale-mismatch detector on a crafted
   ``process__source.csv`` / ``process__sink.csv`` / unitsize trio.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flextool.flextoolrunner.scaling import FamilyStats, ScaleTable
from flextool.flextoolrunner.scaling_report import (
    BIMODAL_GAP_DECADES,
    COMPOSITE_MISMATCH_LOG10_THRESHOLD,
    MISMATCH_RECOMMENDATION,
    BimodalSplit,
    MismatchPair,
    _collect_family_log10_values,
    _format_mismatch_recommendation,
    detect_bimodal,
    find_composite_mismatches,
    parse_highs_log,
    write_scaling_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _minimal_unitsize_csv(path: Path, sizes: dict[str, float]) -> None:
    header = "entity," + ",".join(sizes.keys())
    values = "value," + ",".join(str(v) for v in sizes.values())
    _write(path, header + "\n" + values + "\n")


def _empty_family_stats() -> FamilyStats:
    return FamilyStats(n_values=0, n_zero=0, n_nonzero=0)


def _fake_scale_table(solve_name: str = "test_solve") -> ScaleTable:
    return ScaleTable(
        solve_name=solve_name,
        use_row_scaling="no",
        scale_the_objective=1e-6,
        family_ranges={
            "entity_unitsize": FamilyStats(
                n_values=3, n_zero=0, n_nonzero=3,
                log10_min=0.0, log10_p10=0.2, log10_median=1.0,
                log10_p90=1.8, log10_max=2.0,
                abs_min=1.0, abs_median=10.0, abs_max=100.0,
            ),
            "node_inflow": _empty_family_stats(),
            "node_annual_flow": _empty_family_stats(),
            "vom_and_op_costs": _empty_family_stats(),
            "capex_invest": _empty_family_stats(),
            "node_penalty": _empty_family_stats(),
        },
        unitsize_spread_log10=2.0,
        rough_obj_estimate=1e6,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        source_dir="/tmp",
    )


# ---------------------------------------------------------------------------
# Bimodal detector
# ---------------------------------------------------------------------------


def test_bimodal_flat_distribution_not_bimodal() -> None:
    """A tight unimodal cluster must not be flagged."""
    log10_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    assert detect_bimodal(log10_vals) is None


def test_bimodal_uniform_wide_not_bimodal() -> None:
    """A uniformly-spread distribution with no clear gap is not bimodal.

    10 values evenly spaced over 5 decades — largest adjacent gap is
    0.55, well under the 2-decade threshold.
    """
    log10_vals = [i * 5 / 9 for i in range(10)]
    assert detect_bimodal(log10_vals) is None


def test_bimodal_clean_two_clusters() -> None:
    """Two tight clusters separated by 4 decades must be flagged."""
    lower = [-2.0, -2.05, -1.95, -2.1, -1.9]
    upper = [3.0, 3.1, 2.9, 3.2, 2.8]
    split = detect_bimodal(lower + upper)
    assert isinstance(split, BimodalSplit)
    assert split.gap_decades > BIMODAL_GAP_DECADES
    assert split.n_lower == 5
    assert split.n_upper == 5
    assert split.lower_share == 0.5
    assert split.upper_share == 0.5
    assert split.lower_center_log10 < 0
    assert split.upper_center_log10 > 0


def test_bimodal_too_small_minority_not_flagged() -> None:
    """A lone outlier (<10% of values) must not trigger bimodal flagging."""
    main = [0.0] * 20
    outlier = [5.0]  # 1/21 ~ 4.8% < 10%
    assert detect_bimodal(main + outlier) is None


def test_bimodal_minimum_size() -> None:
    """Detector needs at least 4 values to bother computing a share."""
    assert detect_bimodal([0.0, 5.0]) is None
    assert detect_bimodal([0.0, 0.1, 5.0]) is None


# ---------------------------------------------------------------------------
# Composite-scale-mismatch detector
# ---------------------------------------------------------------------------


def test_composite_mismatch_found(tmp_path: Path) -> None:
    """A crafted graph with a 1000x unitsize ratio must be flagged."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _minimal_unitsize_csv(
        input_dir / "p_entity_unitsize.csv",
        {
            "small_unit": 0.01,
            "big_unit": 100.0,
            "shared_node": 100.0,
        },
    )
    # Both units connect to the same node shared_node.
    _write(
        input_dir / "process__source.csv",
        "process,source\nsmall_unit,shared_node\nbig_unit,shared_node\n",
    )
    _write(input_dir / "process__sink.csv", "process,sink\n")
    mismatches = find_composite_mismatches(input_dir)
    assert len(mismatches) >= 1
    # Highest ratio involves small_unit on one side — either big_unit or
    # shared_node could be the other side (both at 100.0).
    top = mismatches[0]
    assert top.ratio >= 10 ** COMPOSITE_MISMATCH_LOG10_THRESHOLD
    assert top.node == "shared_node"
    assert top.small_entity == "small_unit"
    assert top.large_entity in {"big_unit", "shared_node"}
    # We should also find the small_unit vs big_unit pair specifically.
    has_small_big_pair = any(
        {m.small_entity, m.large_entity} == {"small_unit", "big_unit"}
        for m in mismatches
    )
    assert has_small_big_pair


def test_composite_mismatch_no_false_positive(tmp_path: Path) -> None:
    """Entities within an order of magnitude of each other must not be flagged."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _minimal_unitsize_csv(
        input_dir / "p_entity_unitsize.csv",
        {"unit_a": 100.0, "unit_b": 200.0, "node_x": 500.0},
    )
    _write(
        input_dir / "process__source.csv",
        "process,source\nunit_a,node_x\nunit_b,node_x\n",
    )
    _write(input_dir / "process__sink.csv", "process,sink\n")
    assert find_composite_mismatches(input_dir) == []


def test_composite_mismatch_disconnected_not_flagged(tmp_path: Path) -> None:
    """Entities that share NO node must not be flagged as mismatched."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _minimal_unitsize_csv(
        input_dir / "p_entity_unitsize.csv",
        {
            "tiny": 0.01,
            "huge": 10000.0,
            "node_a": 1.0,
            "node_b": 10000.0,
        },
    )
    # tiny -> node_a, huge -> node_b (different nodes; not directly connected)
    _write(
        input_dir / "process__source.csv",
        "process,source\ntiny,node_a\nhuge,node_b\n",
    )
    _write(input_dir / "process__sink.csv", "process,sink\n")
    mismatches = find_composite_mismatches(input_dir)
    # Only the mismatch within a single node's cloud counts; tiny and huge
    # should NEVER pair up because they don't share a node.
    cross_pair = [
        m for m in mismatches
        if {m.small_entity, m.large_entity} == {"tiny", "huge"}
    ]
    assert cross_pair == []


def test_composite_mismatch_empty_inputs(tmp_path: Path) -> None:
    """Missing CSVs must yield an empty mismatch list, not crash."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    assert find_composite_mismatches(input_dir) == []


# ---------------------------------------------------------------------------
# Recommendation text formatting
# ---------------------------------------------------------------------------


def test_recommendation_text_has_both_options() -> None:
    """The locked recommendation template must include aggregate+sequential."""
    assert "Aggregate the small-side units" in MISMATCH_RECOMMENDATION
    assert "sequential models" in MISMATCH_RECOMMENDATION
    assert "{node}" in MISMATCH_RECOMMENDATION
    assert "{small_entity}" in MISMATCH_RECOMMENDATION
    assert "{large_entity}" in MISMATCH_RECOMMENDATION


def test_recommendation_rendered_ascii_only() -> None:
    """Rendered recommendation must be ASCII (no unicode)."""
    m = MismatchPair(
        process="tiny_unit",
        node="west",
        role="source",
        process_unitsize=0.01,
        node_unitsize=1000.0,
        ratio=100000.0,
        small_entity="tiny_unit",
        small_size=0.01,
        large_entity="mega_unit",
        large_size=1000.0,
    )
    text = _format_mismatch_recommendation(m)
    text.encode("ascii")  # must not raise
    assert "west" in text
    assert "tiny_unit" in text
    assert "mega_unit" in text


# ---------------------------------------------------------------------------
# HiGHS log parsing
# ---------------------------------------------------------------------------


def test_parse_highs_log_ranges() -> None:
    fake_log = """Running HiGHS 1.14.0 (git hash: abc): Copyright blah
MIP flextool has 100 rows; 200 cols; 400 nonzeros; 0 integer variables
Coefficient ranges:
  Matrix  [3e-03, 2e+04]
  Cost    [9e-03, 2e+03]
  Bound   [1e+00, 1e+03]
  RHS     [3e-03, 1e+04]
"""
    parsed = parse_highs_log(fake_log)
    assert parsed["version"] == "1.14.0"
    assert parsed["matrix_range"] == (3e-3, 2e4)
    assert parsed["cost_range"] == (9e-3, 2e3)
    assert parsed["bound_range"] == (1.0, 1e3)
    assert parsed["rhs_range"] == (3e-3, 1e4)


def test_parse_highs_log_missing_fields() -> None:
    parsed = parse_highs_log("no coefficient ranges block here")
    assert parsed["version"] is None
    assert parsed["matrix_range"] is None
    assert parsed["cost_range"] is None


# ---------------------------------------------------------------------------
# Section rendering end-to-end
# ---------------------------------------------------------------------------


def test_write_scaling_report_end_to_end(tmp_path: Path, capsys) -> None:
    """Render a full report with minimum inputs; every section present."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _minimal_unitsize_csv(
        input_dir / "p_entity_unitsize.csv",
        {"a": 1.0, "b": 10.0, "c": 100.0},
    )
    _write(input_dir / "process__source.csv", "process,source\n")
    _write(input_dir / "process__sink.csv", "process,sink\n")
    solve_data = tmp_path / "solve_data"
    solve_data.mkdir()
    log_path = tmp_path / "HiGHS.log"
    log_path.write_text(
        "Running HiGHS 9.9.9 (test)\n"
        "Coefficient ranges:\n"
        "  Matrix  [1e-02, 1e+02]\n"
        "  Cost    [1e+00, 1e+03]\n"
        "  Bound   [1e+00, 1e+03]\n"
        "  RHS     [1e-02, 1e+02]\n"
    )
    table = _fake_scale_table("test_solve")
    out_path = write_scaling_report(
        scale_table=table,
        input_dir=input_dir,
        solve_data_dir=solve_data,
        solve_name="test_solve",
        highs_log_path=log_path,
        output_raw_dir=tmp_path / "output_raw",  # does not exist → no slack
        stdout_summary=True,
    )
    assert out_path.exists()
    text = out_path.read_text()
    # All nine sections present.
    assert "FlexTool scaling diagnostic report" in text
    assert "2. Scaling decisions" in text
    assert "3. Coefficient-family ranges" in text
    assert "4. Bimodal coefficient distributions" in text
    assert "5. Composite-scale mismatch" in text
    assert "6. Near-duplicate" in text
    assert "7. Escape-tier" in text
    assert "8. HiGHS matrix-range summary" in text
    assert "9. Summary" in text
    # HiGHS version echoed.
    assert "9.9.9" in text
    # Summary line classifies as well-scaled or acceptable.
    assert any(
        line in text
        for line in (
            "Model well-scaled: no significant diagnostics",
            "Model scaled acceptably",
            "Model poorly scaled",
        )
    )
    # Stdout echo happened.
    captured = capsys.readouterr().out
    assert "[scaling-report] solve='test_solve'" in captured


def test_write_scaling_report_mismatch_path(tmp_path: Path, capsys) -> None:
    """When a composite mismatch fires, the recommendation text must appear
    in the report AND stdout."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _minimal_unitsize_csv(
        input_dir / "p_entity_unitsize.csv",
        {"tiny": 0.01, "huge": 10000.0, "shared": 10000.0},
    )
    _write(
        input_dir / "process__source.csv",
        "process,source\ntiny,shared\nhuge,shared\n",
    )
    _write(input_dir / "process__sink.csv", "process,sink\n")
    solve_data = tmp_path / "solve_data"
    solve_data.mkdir()

    table = _fake_scale_table("composite_solve")
    out_path = write_scaling_report(
        scale_table=table,
        input_dir=input_dir,
        solve_data_dir=solve_data,
        solve_name="composite_solve",
        highs_log_path=None,
        output_raw_dir=tmp_path / "output_raw",
        stdout_summary=True,
    )
    text = out_path.read_text()
    assert "Composite-scale mismatch detected" in text
    assert "Aggregate the small-side units" in text
    assert "sequential models" in text
    assert "tiny" in text
    assert "huge" in text
    # stdout echoed full recommendation text when mismatch fired
    captured = capsys.readouterr().out
    assert "Composite-scale mismatch detected" in captured
    assert "model poorly scaled" in captured


def test_collect_family_log10_values(tmp_path: Path) -> None:
    """Sanity-check the helper that rescans for per-family log10 lists."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _minimal_unitsize_csv(
        input_dir / "p_entity_unitsize.csv",
        {"a": 10.0, "b": 1000.0},
    )
    table = _fake_scale_table()
    out = _collect_family_log10_values(table.family_ranges, input_dir)
    assert "entity_unitsize" in out
    # log10(10)=1, log10(1000)=3
    assert sorted(round(v, 6) for v in out["entity_unitsize"]) == [1.0, 3.0]
