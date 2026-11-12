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

from flextool.engine.preprocess.emit._period_params import (
    derive_pdtCommodity,
    derive_pdtCommodity_vectorized,
)


def _provider_from_workdir(workdir: Path):
    """Reconstruct a Provider from every CSV in input/ + solve_data/.

    Dual-registers each frame under ``"<parent>/<stem>"`` AND the bare
    ``"<stem>"`` key so both ``_provider_key``-qualified lookups and any
    bare lookup resolve.
    """
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider

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


def test_vectorized_pdtCommodity_branch_period_inherits_anchor_pd(tmp_path):
    """A stochastic branch period without a ``pd_commodity`` row of its own
    takes its anchor period's (#55), in both twins; time-first still holds:

    * ``coal``: anchor ``pd`` 17 → ``d1_b`` gets 17, not the ``p`` scalar 16
      it got before (non-vacuous).
    * ``gas``: ``d1_b`` authors its own ``pd`` (8) → it wins.
    * ``oil``: ``pt`` at ``t2`` (40) beats the inherited ``pd`` (30) in
      ``d1_b`` as in ``d1``.
    """
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider

    provider = FlexDataProvider()

    def put(parent: str, stem: str, df: pl.DataFrame) -> None:
        provider.put(f"{parent}/{stem}", df)
        provider.put(stem, df)

    put("input", "commodity", pl.DataFrame({"commodity": ["coal", "gas", "oil"]}))
    put("solve_data", "period__branch", pl.DataFrame({
        "period": ["d1", "d1"], "branch": ["d1", "d1_b"]}))
    put("solve_data", "steps_in_use", pl.DataFrame({
        "period": ["d1", "d1", "d1_b", "d1_b"],
        "time": ["t1", "t2", "t1", "t2"]}))
    put("input", "pd_commodity", pl.DataFrame({
        "commodity": ["coal", "gas", "gas", "oil"], "param": ["price"] * 4,
        "period": ["d1", "d1", "d1_b", "d1"], "value": [17.0, 6.0, 8.0, 30.0]}))
    put("solve_data", "pt_commodity", pl.DataFrame({
        "commodity": ["oil"], "param": ["price"], "time": ["t2"], "value": [40.0]}))
    put("input", "p_commodity", pl.DataFrame({
        "commodity": ["coal"], "param": ["price"], "value": [16.0]}))

    inp = tmp_path / "input"
    sdd = tmp_path / "solve_data"
    df_legacy = derive_pdtCommodity(inp, sdd, provider=provider)
    df_vec = derive_pdtCommodity_vectorized(inp, sdd, provider=provider)

    leg = {(r[0], r[2], r[3]): r[4] for r in df_legacy.iter_rows()}
    assert leg[("coal", "d1_b", "t1")] == leg[("coal", "d1", "t1")] == repr(17.0), leg
    assert leg[("coal", "d1_b", "t1")] != repr(16.0), "branch period still on the p scalar"
    assert leg[("gas", "d1", "t1")] == repr(6.0), leg
    assert leg[("gas", "d1_b", "t1")] == repr(8.0), leg
    assert leg[("oil", "d1_b", "t1")] == repr(30.0), leg
    assert leg[("oil", "d1_b", "t2")] == leg[("oil", "d1", "t2")] == repr(40.0), leg
    assert df_vec.equals(df_legacy), (
        f"branch-period pd_commodity: vectorized != legacy.\n"
        f"legacy:\n{df_legacy}\nvec:\n{df_vec}")
