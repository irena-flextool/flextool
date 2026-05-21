"""Tests for ``write_fix_storage_files_from_handoff`` — the wide-frame
fan-out that turns ``SolveHandoff.fix_storage`` back into the three
per-metric ``fix_storage_{quantity,price,usage}.csv`` files.

(The legacy ``capture_post_solve`` round-trip tests were retired
alongside the function in Phase 3 of
``specs/provider_consolidation.md``; the cascade builds
``SolveHandoff`` directly from the flexpy ``Solution`` via
``build_handoff_from_flexpy``.)
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from flextool.flextoolrunner.solve_handoff import (
    write_fix_storage_files_from_handoff,
)


# ---------------------------------------------------------------------------
# write_fix_storage_files_from_handoff — round-trip with NULL-aware fan-out
# ---------------------------------------------------------------------------


def test_solve_handoff_round_trip_fix_storage(tmp_path: Path) -> None:
    """Wide ``[node, period, time, quantity, price, usage]`` → three per-metric
    files, each containing ONLY rows where its metric is non-NULL.  Verifies
    per-metric independence (mixed NULLs across rows) plus the on-disk
    ``time → step`` rename in ``solve_handoff.py:301``."""
    sd = tmp_path / "solve_data"
    sd.mkdir()
    # Row 1: only quantity.  Row 2: only price.  Row 3: all three set.
    fix_storage = pl.DataFrame(
        {
            "node":     ["battery", "battery", "tank"],
            "period":   ["p2020",   "p2020",   "p2025"],
            "time":     ["t0001",   "t0024",   "t0001"],
            "quantity": [50.0,      None,      11.0],
            "price":    [None,      -1600.0,   22.0],
            "usage":    [None,      None,      33.0],
        },
        schema={
            "node":     pl.Utf8,
            "period":   pl.Utf8,
            "time":     pl.Utf8,
            "quantity": pl.Float64,
            "price":    pl.Float64,
            "usage":    pl.Float64,
        },
    )

    write_fix_storage_files_from_handoff(fix_storage, sd)

    # quantity: rows where ``quantity`` is non-null (rows 1 and 3).
    qdf = pl.read_csv(sd / "fix_storage_quantity.csv")
    assert qdf.columns == ["node", "period", "step", "p_fix_storage_quantity"]
    assert qdf.to_dicts() == [
        {"node": "battery", "period": "p2020", "step": "t0001",
         "p_fix_storage_quantity": 50.0},
        {"node": "tank",    "period": "p2025", "step": "t0001",
         "p_fix_storage_quantity": 11.0},
    ]

    # price: rows where ``price`` is non-null (rows 2 and 3).
    pdf = pl.read_csv(sd / "fix_storage_price.csv")
    assert pdf.columns == ["node", "period", "step", "p_fix_storage_price"]
    assert pdf.to_dicts() == [
        {"node": "battery", "period": "p2020", "step": "t0024",
         "p_fix_storage_price": -1600.0},
        {"node": "tank",    "period": "p2025", "step": "t0001",
         "p_fix_storage_price": 22.0},
    ]

    # usage: rows where ``usage`` is non-null (only row 3).
    udf = pl.read_csv(sd / "fix_storage_usage.csv")
    assert udf.columns == ["node", "period", "step", "p_fix_storage_usage"]
    assert udf.to_dicts() == [
        {"node": "tank", "period": "p2025", "step": "t0001",
         "p_fix_storage_usage": 33.0},
    ]
