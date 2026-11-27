"""Benders-mode half-flow uncap for the regional splitter.

In Benders decomposition the TRUE cross-region capacity limit
``f ≤ C·unitsize`` lives in the MASTER, so each region's cross-region
virtual half-flow must be effectively UNCAPPED — otherwise a greenfield
pipe (whose inherited ``existing`` is 0) pins its ``maxFlow`` row to zero
trade, the original false-convergence bug (``specs/benders_option_c.md``
Phase-2-revised Point 5).

These tests pin both halves of the contract on the
``lh2_three_region_trade_invest`` fixture (regions A/B/C, greenfield
``pipe_AB`` / ``pipe_BC``):

* WITHOUT ``benders_uncap_cross_region`` the half-flow's
  ``p_flow_upper_existing`` resolves to ~0 (today's behaviour — the bug);
* WITH the mode it resolves to ``inf`` (no limit: the region LP has no
  ``maxFlow`` row for it) so a positive flow pin is feasible (the master
  then owns the real cap).
"""
from __future__ import annotations

import polars as pl
import pytest

from flextool.engine.solve._region_filter import split
from flextool.engine.solve._region_filter._region_data import _BENDERS_UNCAP_SENTINEL
from tests._support.load import load_scenario


@pytest.fixture(scope="module")
def ti_run(scenario_run):
    return scenario_run(
        "lh2_three_region_trade_invest", db_fixture="lh2_trade_invest"
    )


@pytest.fixture(scope="module")
def ti_data(ti_run):
    return load_scenario(ti_run)


@pytest.fixture(scope="module")
def ti_splits_default(ti_data):
    return split(ti_data, regions=["region_A", "region_B", "region_C"])


@pytest.fixture(scope="module")
def ti_splits_benders(ti_data):
    return split(
        ti_data,
        regions=["region_A", "region_B", "region_C"],
        benders_uncap_cross_region=True,
    )


def _export_hf(splits, region_idx: int, original_p: str):
    s = splits[region_idx]
    return s, next(
        hf for hf in s.half_flows
        if hf.original_p == original_p and hf.side == "export"
    )


def _hf_existing_cap(data, virtual_p: str) -> list[float]:
    """Resolve the half-flow's ``p_flow_upper_existing`` value(s) — the
    ``maxFlow`` RHS factor that bounds the cross-region trade flow."""
    return (data.p_flow_upper_existing.frame
            .filter(pl.col("p") == virtual_p)
            .sort("d")["value"].to_list())


class TestDefaultPathInheritsZeroCap:
    """Greenfield cross-region half-flows inherit the original arc's
    ``existing``-cap (0) — the literal root cause the Benders mode fixes.
    This is today's byte-identical behaviour; it must NOT move."""

    def test_pipe_ab_export_cap_is_zero(self, ti_splits_default) -> None:
        sA, hf = _export_hf(ti_splits_default, 0, "pipe_AB")
        caps = _hf_existing_cap(sA.data, hf.virtual_p)
        assert caps, "expected a p_flow_upper_existing row for the half-flow"
        assert all(c == 0.0 for c in caps)

    def test_pipe_bc_export_cap_is_zero(self, ti_splits_default) -> None:
        # region_B exports toward region_C over pipe_BC.
        sB, hf = _export_hf(ti_splits_default, 1, "pipe_BC")
        caps = _hf_existing_cap(sB.data, hf.virtual_p)
        assert caps
        assert all(c == 0.0 for c in caps)


class TestBendersModeUncaps:
    """With the mode ON the half-flow's existing-cap is swapped for the
    large sentinel so any flow the master pins stays feasible (the
    ``maxFlow`` row is structurally slack)."""

    def test_pipe_ab_export_cap_is_sentinel(self, ti_splits_benders) -> None:
        bA, hf = _export_hf(ti_splits_benders, 0, "pipe_AB")
        caps = _hf_existing_cap(bA.data, hf.virtual_p)
        assert caps
        assert all(c == _BENDERS_UNCAP_SENTINEL for c in caps)

    def test_pipe_bc_export_cap_is_sentinel(self, ti_splits_benders) -> None:
        bB, hf = _export_hf(ti_splits_benders, 1, "pipe_BC")
        caps = _hf_existing_cap(bB.data, hf.virtual_p)
        assert caps
        assert all(c == _BENDERS_UNCAP_SENTINEL for c in caps)

    def test_half_flow_capacity_is_no_limit(self, ti_splits_benders) -> None:
        """The half-flow's existing capacity is ``inf`` (no limit), not a
        finite stand-in: the master owns the real cap."""
        import math

        assert math.isinf(_BENDERS_UNCAP_SENTINEL) and _BENDERS_UNCAP_SENTINEL > 0
        for region_idx, pipe in ((0, "pipe_AB"), (1, "pipe_BC")):
            s, hf = _export_hf(ti_splits_benders, region_idx, pipe)
            assert all(math.isinf(c) for c in _hf_existing_cap(s.data, hf.virtual_p))

    def test_region_lp_has_no_max_flow_row_for_half_flows(self, ti_splits_benders) -> None:
        """An inf capacity is no limit: the region LP has no ``maxFlow`` row
        for any half-flow (only for its own arcs), and its bound / RHS pool
        holds no 1e12-class stand-in."""
        import math

        from polar_high import Problem
        from polar_high.autoscale import ScalingConfig, detect_ranges

        from flextool.engine.model.core import build_flextool

        for s in ti_splits_benders:
            virtual = {hf.virtual_p for hf in s.half_flows}
            assert virtual
            pb = Problem()
            build_flextool(pb, s.data)
            for name, _proto, over in pb._cstrs:
                if name.startswith("maxFlow") and over is not None and "p" in over.columns:
                    ps = set(over["p"].cast(pl.Utf8).to_list())
                    assert not (ps & virtual), (s.region, name, sorted(ps & virtual))
            rep = detect_ranges(pb, ScalingConfig())
            assert rep.bound_rhs_meaningful is not None
            assert rep.bound_rhs_meaningful[1] < 1e12, (s.region, rep.bound_rhs_meaningful)
            for hi in (rep.rhs[1], rep.bound[1]):  # nan: no entry of that kind
                assert math.isnan(hi) or hi < 1e12, (s.region, rep.rhs, rep.bound)


def test_default_is_byte_identical_to_today(
    ti_splits_default, ti_data
) -> None:
    """Sanity guard: with the mode OFF the half-flow existing-cap equals
    the original arc's inherited value verbatim (no sentinel leak)."""
    sA, hf = _export_hf(ti_splits_default, 0, "pipe_AB")
    orig = (ti_data.p_flow_upper_existing.frame
            .filter((pl.col("p") == "pipe_AB")
                    & (pl.col("source") == hf.original_source)
                    & (pl.col("sink") == hf.original_sink))
            .sort("d")["value"].to_list())
    new = _hf_existing_cap(sA.data, hf.virtual_p)
    assert new == orig
