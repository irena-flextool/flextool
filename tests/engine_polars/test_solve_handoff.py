"""Unit-level tests for the cascade's in-memory ``SolveHandoff`` consume helpers.

Phase 3 of ``specs/provider_consolidation.md`` deleted the disk-reading
``capture_post_solve`` constructor; the cascade builds ``SolveHandoff``
directly from the polar_high ``Solution`` via ``build_handoff_from_solution``.
End-to-end coverage lives in ``test_chain_handoff_writers.py``.

The unit-level test preserved here verifies a consume helper in
isolation:

* :func:`test_cumulative_loaders_consume_from_handoff` — the three
  cumulative-handoff prior loaders read from a populated
  ``SolveHandoff`` instead of disk when the optional ``prior_handoff``
  kwarg is supplied.

Phase 4.1i retired ``write_fix_storage_files_from_handoff``; the
corresponding disk-fan-out unit test was deleted alongside the helper.

Audit pass H retired the legacy disk-based cumulative-handoff writer
module; the three loader bodies the test below exercises are inlined
here as private helpers because they were the only surviving references
to that module.  Production reads handoff frames directly via
``read_handoff_frame(provider, K.HANDOFF_*)``; the loader shape (file
path + optional ``prior_handoff`` kwarg) is preserved here purely so the
test continues to assert the in-memory short-circuit semantics.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl

from flextool.engine_polars._solve_handoff import (
    SolveHandoff,
)


# ---------------------------------------------------------------------------
# Inlined prior-loader helpers (verbatim from the retired legacy
# disk-based cumulative-handoff writer; see audit pass H).  Each helper
# checks ``prior_handoff`` first (in-memory consume) and falls back to
# the CSV path (header-only seed → empty).
# ---------------------------------------------------------------------------


def _load_prior_cum_realized_mwh(
    path: Path,
    *,
    prior_handoff: "SolveHandoff | None" = None,
) -> dict[tuple[str, int, str], float]:
    """``{(commodity, tier, period): cum_mwh}`` from the previous roll's
    accumulator.  Handoff takes precedence over disk."""
    if (
        prior_handoff is not None
        and prior_handoff.cumulative_commodity is not None
    ):
        out: dict[tuple[str, int, str], float] = {}
        for r in prior_handoff.cumulative_commodity.iter_rows(named=True):
            try:
                tier = int(r["tier"])
                val = float(r["p_ladder_cum_realized_mwh"])
            except (ValueError, TypeError):
                continue
            out[(str(r["commodity"]), tier, str(r["period"]))] = val
        return out
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    needed = {"commodity", "tier", "period", "p_ladder_cum_realized_mwh"}
    if df.empty or not needed.issubset(df.columns):
        return {}
    out = {}
    for _, row in df.iterrows():
        try:
            tier = int(row["tier"])
            val = float(row["p_ladder_cum_realized_mwh"])
        except (ValueError, TypeError):
            continue
        out[(str(row["commodity"]), tier, str(row["period"]))] = val
    return out


def _load_prior_cum_sim_hours(
    path: Path,
    *,
    prior_handoff: "SolveHandoff | None" = None,
) -> dict[str, float]:
    """``{period: cum_hours}`` from the previous roll's accumulator.
    Handoff takes precedence over disk."""
    if (
        prior_handoff is not None
        and prior_handoff.cum_sim_hours is not None
    ):
        return {
            str(r["period"]): float(r["p_ladder_cum_sim_hours"])
            for r in prior_handoff.cum_sim_hours.iter_rows(named=True)
        }
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    needed = {"period", "p_ladder_cum_sim_hours"}
    if df.empty or not needed.issubset(df.columns):
        return {}
    return {
        str(r["period"]): float(r["p_ladder_cum_sim_hours"])
        for _, r in df.iterrows()
    }


def _load_prior_co2_cum_realized_tonnes(
    path: Path,
    *,
    prior_handoff: "SolveHandoff | None" = None,
) -> dict[tuple[str, str], float]:
    """``{(group, period): cum_tonnes}`` from the previous roll's CO2
    accumulator.  Handoff takes precedence over disk."""
    if (
        prior_handoff is not None
        and prior_handoff.cumulative_co2 is not None
    ):
        out: dict[tuple[str, str], float] = {}
        for r in prior_handoff.cumulative_co2.iter_rows(named=True):
            try:
                val = float(r["value"])
            except (ValueError, TypeError):
                continue
            out[(str(r["group"]), str(r["period"]))] = val
        return out
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    needed = {"group", "period", "p_co2_cum_realized_tonnes"}
    if df.empty or not needed.issubset(df.columns):
        return {}
    out = {}
    for _, row in df.iterrows():
        try:
            val = float(row["p_co2_cum_realized_tonnes"])
        except (ValueError, TypeError):
            continue
        out[(str(row["group"]), str(row["period"]))] = val
    return out


def test_cumulative_loaders_consume_from_handoff(tmp_path):
    """Unit-level: the three cumulative-handoff prior loaders read from
    a populated ``SolveHandoff`` instead of disk when the optional
    ``prior_handoff`` kwarg is supplied.  Verifies disk is bypassed by
    pointing the file path at a non-existent location — only the
    handoff path can produce non-empty output."""
    # Doesn't exist — file-only path would return {}.
    nope = tmp_path / "does_not_exist.csv"

    h = SolveHandoff(
        cumulative_co2=pl.DataFrame({
            "group":  ["g1", "g2"],
            "period": ["p2025", "p2025"],
            "value":  [12.5, 99.0],
        }),
        # Phase 4.1a — handoff carrier schemas match the canonical
        # ``solve_data/`` column names so the iteration-start translator
        # can route them through unchanged.
        cumulative_commodity=pl.DataFrame({
            "commodity":                 ["coal"],
            "tier":                      [1],
            "period":                    ["p2025"],
            "p_ladder_cum_realized_mwh": [42.0],
        }),
        cum_sim_hours=pl.DataFrame({
            "period":                 ["p2025", "p2030"],
            "p_ladder_cum_sim_hours": [8760.0, 4380.0],
        }),
    )

    co2 = _load_prior_co2_cum_realized_tonnes(nope, prior_handoff=h)
    assert co2 == {("g1", "p2025"): 12.5, ("g2", "p2025"): 99.0}

    mwh = _load_prior_cum_realized_mwh(nope, prior_handoff=h)
    assert mwh == {("coal", 1, "p2025"): 42.0}

    hrs = _load_prior_cum_sim_hours(nope, prior_handoff=h)
    assert hrs == {"p2025": 8760.0, "p2030": 4380.0}

    # Sanity: without the handoff, the same call returns empty (proving
    # the disk path is what the handoff is bypassing).
    assert _load_prior_co2_cum_realized_tonnes(nope) == {}
    assert _load_prior_cum_realized_mwh(nope) == {}
    assert _load_prior_cum_sim_hours(nope) == {}


# Phase 4.1i — ``test_write_fix_storage_files_from_handoff`` was deleted
# alongside the helper.  The wide → narrow CSV fan-out path has no
# consumers (per-metric ``handoff/*`` Provider keys cover all readers);
# end-to-end coverage of the new path lives in the rolling/chain
# handoff suites.
