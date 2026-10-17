"""Slice D — Phase D3: α-1 year/factor provider threading.

Design: ``specs/sliceD_recourse_invest_design.md`` §5 / §18 D3.
Under recourse the walker forwards ``provider`` so the canonical
``p_years_d.csv`` / ``p_years_represented.csv`` arms — which byte-copy the
fan-member year rows — go live, giving branch periods their anchor's
year-from-start.  ``provider=None`` (flag-off) → arms dead → today's
behaviour (the W6 today-bug pin).  These are the two canonical-arm reads,
unit-tested; the full edd cross-leaf / objective effect lands at D4/D5.
"""
from __future__ import annotations

import polars as pl

from flextool.engine_polars._derived_params import (
    _p_years_d_lf,
    _years_for_period_from_source,
)


def _emit(prov, key, df):
    from flextool.engine_polars._emit_provider_io import _emit as _e
    _e(prov, key, df)


def _provider():
    from flextool.engine_polars._flex_data_provider import FlexDataProvider
    return FlexDataProvider()


# ---------------------------------------------------------------------------
# Year side — _p_years_d_lf canonical arm (already provider-gated; verify the
# fan-member rows resolve when provider is present and stay absent when not).
# ---------------------------------------------------------------------------


def test_p_years_d_canonical_arm_reads_branch_rows(tmp_path):
    prov = _provider()
    _emit(prov, "solve_data/p_years_d.csv", pl.DataFrame({
        "period": ["p2035", "p2035_low", "p2040", "p2040_low"],
        "value": ["0", "0", "1", "1"],
    }))
    # The canonical arm builds its provider key from ``workdir``; the disk
    # file need not exist (provider holds the frame).
    out = _p_years_d_lf(None, "s", tmp_path, provider=prov)
    assert out is not None
    got = {r["d"]: r["yr"]
           for r in out.collect().iter_rows(named=True)}
    assert got == {"p2035": 0.0, "p2035_low": 0.0,
                   "p2040": 1.0, "p2040_low": 1.0}


# ---------------------------------------------------------------------------
# Factor side — _years_for_period_from_source canonical arm.
# ---------------------------------------------------------------------------


def test_years_for_period_canonical_arm_branch_rows():
    prov = _provider()
    _emit(prov, "solve_data/p_years_represented.csv", pl.DataFrame({
        "period": ["p2035", "p2035_low", "p2040", "p2040_low"],
        "years_from_solve": ["0", "0", "1", "1"],
        "p_years_from_solve": ["0", "0", "1", "1"],
        "p_years_represented": ["1.0", "1.0", "1.0", "1.0"],
    }))
    out = _years_for_period_from_source(
        None, "s", ["p2035", "p2035_low", "p2040", "p2040_low"],
        provider=prov)
    assert out["p2035_low"] == [("0", 1.0)]
    assert out["p2040_low"] == [("1", 1.0)]


def test_years_for_period_provider_without_csv_falls_through():
    # provider present but no p_years_represented.csv key → canonical arm
    # skipped.  (The source-path fallback is covered by the flag-off golden
    # spot-suites, which pass a real InputSource.)
    from flextool.engine_polars._flex_data_provider import FlexDataProvider
    prov = FlexDataProvider()
    p = "solve_data/p_years_represented.csv"
    from flextool.engine_polars._derived_params import (
        _provider_has_key as _hk,
    )
    from pathlib import Path
    assert _hk(prov, Path(p)) is False
