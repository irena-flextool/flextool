"""Byte-parity gate for the vectorized ``pdtGroup`` derive.

Tier A (design §1): the vectorized ``derive_pdtGroup_vectorized`` must
produce a frame BYTE-IDENTICAL to the legacy per-cell-loop
``derive_pdtGroup`` on BOTH fixtures:

* ``fullYear``                     — rolling, non-stochastic.
* ``2_day_stochastic_dispatch``    — stochastic.

``pdtGroup`` is a pure inline 4-branch cascade
(``pt_group`` → ``pd_group`` → ``p_group`` → ``0.0``) — no fold, so it is
byte-exact on both fixtures regardless of stochasticity.

The Provider is reconstructed by globbing EVERY CSV in ``work/input`` and
``work/solve_data`` and dual-registering each under both the
parent-qualified key (``solve_data/<stem>``) and the bare ``<stem>`` key
(design §6 / S6 — glob, do not under-register).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine.preprocess.emit._period_params import (
    derive_pdtGroup,
    derive_pdtGroup_vectorized,
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
def test_vectorized_pdtGroup_matches_legacy(
    scenario, db_fixture, scenario_workdir,
):
    work = scenario_workdir(scenario, db_fixture=db_fixture)
    p = _provider_from_workdir(work)
    inp = work / "input"
    sdd = work / "solve_data"

    df_legacy = derive_pdtGroup(inp, sdd, provider=p)
    df_vec = derive_pdtGroup_vectorized(inp, sdd, provider=p)

    # Tier A — strict byte-parity.
    assert df_vec.equals(df_legacy), (
        f"{scenario}: vectorized pdtGroup != legacy (Tier-A byte "
        f"parity). legacy {df_legacy.shape}, vec {df_vec.shape}"
    )


def _put_branch_common(put, entity_col: str, entities: list[str]) -> None:
    """Anchor ``d1`` with one branch period ``d1_b`` (period__branch
    (d1, d1), (d1, d1_b)); both in steps_in_use."""
    put("input", entity_col, pl.DataFrame({entity_col: entities}))
    put("solve_data", "period__branch", pl.DataFrame({
        "period": ["d1", "d1"], "branch": ["d1", "d1_b"]}))
    put("solve_data", "steps_in_use", pl.DataFrame({
        "period": ["d1", "d1", "d1_b", "d1_b"],
        "time": ["t1", "t2", "t1", "t2"]}))


def test_vectorized_pdtGroup_branch_period_inherits_anchor_pd(tmp_path):
    """A stochastic branch period without a ``pd_group`` row of its own takes
    its anchor period's (#55), in both twins; time-first still holds:

    * ``g1.co2_price``: anchor ``pd`` 50 → ``d1_b`` gets 50, not the ``p``
      scalar 10 it got before (non-vacuous).
    * ``g2.co2_price``: ``d1_b`` authors its own ``pd`` (70) → it wins.
    * ``g3.co2_price``: ``pt`` at ``t2`` (90) beats the inherited ``pd`` in
      ``d1_b`` as it beats the anchor's in ``d1``.
    """
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider

    provider = FlexDataProvider()

    def put(parent: str, stem: str, df: pl.DataFrame) -> None:
        provider.put(f"{parent}/{stem}", df)
        provider.put(stem, df)

    _put_branch_common(put, "group", ["g1", "g2", "g3"])
    put("input", "pd_group", pl.DataFrame({
        "group": ["g1", "g2", "g2", "g3"], "param": ["co2_price"] * 4,
        "period": ["d1", "d1", "d1_b", "d1"], "value": [50.0, 60.0, 70.0, 80.0]}))
    put("solve_data", "pt_group", pl.DataFrame({
        "group": ["g3"], "param": ["co2_price"], "time": ["t2"], "value": [90.0]}))
    put("input", "p_group", pl.DataFrame({
        "group": ["g1"], "param": ["co2_price"], "value": [10.0]}))

    inp = tmp_path / "input"
    sdd = tmp_path / "solve_data"
    df_legacy = derive_pdtGroup(inp, sdd, provider=provider)
    df_vec = derive_pdtGroup_vectorized(inp, sdd, provider=provider)

    leg = {(r[0], r[1], r[2], r[3]): r[4] for r in df_legacy.iter_rows()}
    c = "co2_price"
    assert leg[("g1", c, "d1_b", "t1")] == leg[("g1", c, "d1", "t1")] == repr(50.0), leg
    assert leg[("g1", c, "d1_b", "t1")] != repr(10.0), "branch period still on the p scalar"
    assert leg[("g2", c, "d1", "t1")] == repr(60.0), leg
    assert leg[("g2", c, "d1_b", "t1")] == repr(70.0), leg
    assert leg[("g3", c, "d1_b", "t1")] == repr(80.0), leg
    assert leg[("g3", c, "d1_b", "t2")] == leg[("g3", c, "d1", "t2")] == repr(90.0), leg
    assert df_vec.equals(df_legacy), (
        f"branch-period pd_group: vectorized != legacy.\n"
        f"legacy:\n{df_legacy}\nvec:\n{df_vec}")
