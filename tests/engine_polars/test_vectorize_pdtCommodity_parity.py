"""Byte-parity gate for the vectorized ``pdtCommodity`` derive.

Tier A (design §1): the vectorized
``derive_pdtCommodity_vectorized`` must produce a frame BYTE-IDENTICAL to
the legacy per-cell-loop ``derive_pdtCommodity`` on BOTH fixtures:

* ``fullYear``                     — rolling, non-stochastic.
* ``2_day_stochastic_dispatch``    — stochastic.

``pdtCommodity`` is a pure inline 3-branch cascade
(``pt_commodity`` → ``pd_commodity`` → ``p_commodity`` → ``0.0``) — no
fold, so it is byte-exact on both fixtures regardless of stochasticity.

The Provider is reconstructed by globbing EVERY CSV in ``work/input``
and ``work/solve_data`` and dual-registering each under both the
parent-qualified key (``solve_data/<stem>``) and the bare ``<stem>``
key (design §6 / S6 — glob, do not under-register).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars._emit_period_params import (
    derive_pdtCommodity,
    derive_pdtCommodity_vectorized,
)


def _provider_from_workdir(workdir: Path):
    """Reconstruct a Provider from every CSV in input/ + solve_data/.

    Dual-registers each frame under ``"<parent>/<stem>"`` AND the bare
    ``"<stem>"`` key so both ``_provider_key``-qualified lookups and any
    bare lookup resolve.
    """
    from flextool.engine_polars._flex_data_provider import FlexDataProvider

    provider = FlexDataProvider()
    for parent in ("input", "solve_data"):
        d = workdir / parent
        if not d.is_dir():
            continue
        for csv_path in sorted(d.glob("*.csv")):
            try:
                df = pl.read_csv(csv_path)
            except Exception:
                continue
            stem = csv_path.stem
            provider.put(f"{parent}/{stem}", df)
            provider.put(stem, df)
    return provider


_CASES = [
    ("fullYear", "main"),
    ("2_day_stochastic_dispatch", "stochastic"),
]


@pytest.mark.parametrize("scenario,db_fixture", _CASES)
def test_vectorized_pdtCommodity_matches_legacy(
    scenario, db_fixture, scenario_workdir,
):
    work = scenario_workdir(scenario, db_fixture=db_fixture)
    p = _provider_from_workdir(work)
    inp = work / "input"
    sdd = work / "solve_data"

    df_legacy = derive_pdtCommodity(inp, sdd, provider=p)
    df_vec = derive_pdtCommodity_vectorized(inp, sdd, provider=p)

    # A fixture with an empty commodity domain produces 0 rows; parity at
    # 0 rows is still valid parity, so no height assertion is forced.

    # Tier A — strict byte-parity.
    assert df_vec.equals(df_legacy), (
        f"{scenario}: vectorized pdtCommodity != legacy (Tier-A byte "
        f"parity). legacy {df_legacy.shape}, vec {df_vec.shape}"
    )
