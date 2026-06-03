"""Phase 1g — reporting self-test.

Pins the operator-facing console formatters (``format_console_summary``
and ``format_nonoptimal_hint``) plus the YAML audit's multi-layer shape.
Each test constructs the minimal Layer-1 / Layer-2 / Layer-3 data
structures by hand so the formatters can be exercised without spinning
up a real LP solve.

These tests are the contract between the orchestrator wire-in (which
threads pre/post ranges + Layer 2/3 plans into the formatters) and the
user-visible single line printed at the start of the autoscale-active
solve.  Edit either side, edit this file.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

from flextool.engine_polars.autoscale import (
    Layer2Plan,
    Layer3Plan,
    QuantityType,
    RangeReport,
    format_console_summary,
    format_nonoptimal_hint,
    write_report,
)
from flextool.engine_polars.autoscale._report import (
    render_layer2,
    render_layer3,
)


def _range_report(
    *,
    matrix: tuple[float, float] = (1.0, 1e7),
    cost: tuple[float, float] = (1.0, 1e11),
    bound: tuple[float, float] = (math.nan, math.nan),
    rhs: tuple[float, float] = (1.0, 1e10),
    trigger: bool = True,
) -> RangeReport:
    """Build a hand-rolled :class:`RangeReport`.

    The defaults reproduce the four-decade pattern called out in the
    Phase 1g spec example (Matrix=7.1d, Cost=11.6d, Bound=empty,
    RHS=10.6d-ish), so the tests can assert on the rendered substrings
    without re-deriving them.
    """
    los = [g[0] for g in (matrix, cost, bound, rhs) if not math.isnan(g[0])]
    his = [g[1] for g in (matrix, cost, bound, rhs) if not math.isnan(g[1])]
    cross = max(his) / min(los) if los and his else math.nan
    return RangeReport(
        matrix=matrix,
        cost=cost,
        bound=bound,
        rhs=rhs,
        cross_group_max_ratio=cross,
        trigger=trigger,
    )


def _layer2_plan(n_cols: int = 10, n_rows: int = 4) -> Layer2Plan:
    """Minimal Layer2Plan with three per-type exponents."""
    return Layer2Plan(
        col_factors=np.ones(n_cols, dtype=np.float64),
        row_factors=np.ones(n_rows, dtype=np.float64),
        type_exponents={
            QuantityType.ENERGY: -6,
            QuantityType.POWER: -4,
            QuantityType.DIMENSIONLESS: 7,
        },
        type_buckets_before={
            QuantityType.ENERGY: (1.0, 1e6),
            QuantityType.POWER: (1.0, 1e4),
            QuantityType.DIMENSIONLESS: (1e-7, 1.0),
        },
        type_buckets_after={
            QuantityType.ENERGY: (1.0, 1.0),
            QuantityType.POWER: (1.0, 1.0),
            QuantityType.DIMENSIONLESS: (1.0, 1.0),
        },
        skipped_rows=[],
        skipped_integer_cols=[],
    )


def _layer3_plan() -> Layer3Plan:
    return Layer3Plan(
        user_objective_scale=0,
        user_bound_scale=-9,
        simplex_scale_strategy=2,
        reasoning="auto-recommended from post-Layer-2 ranges",
    )


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def test_console_summary_format() -> None:
    """Trigger=True case must render pre/post ranges, L2 exponents, L3."""
    ranges_pre = _range_report(trigger=True)
    ranges_post = _range_report(
        matrix=(1.0, 1e7),
        cost=(1.0, 10 ** 4.7),
        bound=(math.nan, math.nan),
        rhs=(1.0, 1e4),
        trigger=False,
    )
    line = format_console_summary(
        ranges_pre=ranges_pre,
        ranges_post=ranges_post,
        layer2_plan=_layer2_plan(),
        layer3_plan=_layer3_plan(),
        threshold_decades=9.0,
    )
    # Pre-range section.
    assert "Autoscale by polar-high:" in line
    assert "ranges pre" in line
    assert "Matrix=7.0d" in line
    assert "Cost=11.0d" in line
    assert "Bound=empty" in line
    # Layer 2 exponents (per-type, signed).
    assert "energy:-6" in line
    assert "power:-4" in line
    assert "dimensionless:+7" in line
    # Layer 3 trio.
    assert "user_obj=0" in line
    assert "user_bnd=-9" in line
    assert "simplex=2" in line
    # Post-range section.
    assert "ranges post" in line
    assert "Cost=4.7d" in line


def test_console_summary_no_trigger() -> None:
    """Trigger=False case must yield the quieter 'within comfort zone' line."""
    ranges_pre = _range_report(
        matrix=(1.0, 1e3),
        cost=(1.0, 1e3),
        bound=(math.nan, math.nan),
        rhs=(1.0, 1e3),
        trigger=False,
    )
    line = format_console_summary(
        ranges_pre=ranges_pre,
        ranges_post=None,
        layer2_plan=None,
        layer3_plan=None,
        threshold_decades=9.0,
    )
    assert "within HiGHS comfort zone" in line
    assert "9" in line  # threshold surfaced
    assert "no scaling applied" in line


# ---------------------------------------------------------------------------
# Non-optimal hint
# ---------------------------------------------------------------------------


def test_nonoptimal_hint_triggered() -> None:
    """Layer-1 trigger=True must produce a multi-line hint with the three
    documented remediation paths.

    The decade-count strings are computed from the same ``hi/lo`` ratio
    the formatter uses, so we can assert on the exact substrings without
    re-deriving the log.
    """
    ranges_pre = _range_report(trigger=True)
    hint = format_nonoptimal_hint(ranges_pre)
    assert hint, "non-empty hint expected when trigger=True"
    # Key substrings the spec mandates.
    assert re.search(r"poorly-scaled", hint)
    assert re.search(r"RHS spans \d+\.\d decades", hint)
    assert re.search(r"Cost spans \d+\.\d decades", hint)
    assert "Check unit conventions" in hint
    assert "--highs-threads 1" in hint
    assert "--scaling=solver_only" in hint


def test_nonoptimal_hint_not_triggered_when_well_scaled() -> None:
    """trigger=False must produce no scaling hint — printing one on a
    well-conditioned LP would be misleading."""
    ranges_pre = _range_report(
        matrix=(1.0, 1e3),
        cost=(1.0, 1e3),
        bound=(math.nan, math.nan),
        rhs=(1.0, 1e3),
        trigger=False,
    )
    hint = format_nonoptimal_hint(ranges_pre)
    assert hint == ""


# ---------------------------------------------------------------------------
# YAML audit
# ---------------------------------------------------------------------------


def test_yaml_report_contains_all_layers(tmp_path: Path) -> None:
    """A report tree carrying Layer 1/2/3 must round-trip through the
    YAML emitter with all three sections present."""
    ranges = _range_report(trigger=True)
    layer2 = _layer2_plan()
    layer3 = _layer3_plan()

    tree: dict = {
        "layer1": ranges,
        "layer2": render_layer2(layer2),
        "layer3": render_layer3(layer3),
    }
    path = tmp_path / "autoscale_test.yaml"
    written = write_report(tree, path)
    assert written == path
    text = written.read_text(encoding="utf-8")

    # Each section heading must be present at indent-0.
    for key in ("layer1:", "layer2:", "layer3:"):
        assert re.search(rf"^{re.escape(key)}", text, re.MULTILINE), key

    # Layer 1 surface: the four range groups + trigger.
    assert "matrix:" in text
    assert "cost:" in text
    assert "bound:" in text
    assert "rhs:" in text
    assert "trigger: true" in text

    # Layer 2 surface: per-type exponents + skipped counts.
    assert "type_exponents:" in text
    assert "energy: -6" in text
    assert "power: -4" in text
    assert "dimensionless: 7" in text
    assert "skipped_rows_count: 0" in text

    # Layer 3 surface: the three HiGHS knobs + reasoning string.
    assert "user_objective_scale: 0" in text
    assert "user_bound_scale: -9" in text
    assert "simplex_scale_strategy: 2" in text
    assert "reasoning:" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
