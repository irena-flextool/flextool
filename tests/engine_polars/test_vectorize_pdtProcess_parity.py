"""Byte-parity gate for the vectorized ``pdtProcess`` derive.

Tier A (design §1): the vectorized
``derive_pdtProcess_vectorized`` must produce a frame BYTE-IDENTICAL to
the legacy per-cell-loop ``derive_pdtProcess`` on BOTH fixtures:

* ``fullYear``                     — rolling, non-stochastic.
* ``2_day_stochastic_dispatch``    — stochastic (exercises the
  branch-1 stochastic fold + branch-2 parent-period fold, and must
  produce branch-period rows so the gate is non-vacuous).

The Provider is reconstructed by globbing EVERY CSV in ``work/input``
and ``work/solve_data`` and dual-registering each under both the
parent-qualified key (``solve_data/<stem>``) and the bare ``<stem>``
key (design §6 / S6 — the 7-item helper in ``test_pbt_node_inflow.py``
under-registers; glob instead).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars._emit_pdt_params import (
    derive_pdtProcess,
    derive_pdtProcess_vectorized,
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
    ("fullYear", "main", False),
    ("2_day_stochastic_dispatch", "stochastic", True),
]


@pytest.mark.parametrize("scenario,db_fixture,expect_branch", _CASES)
def test_vectorized_pdtProcess_matches_legacy(
    scenario, db_fixture, expect_branch, scenario_workdir,
):
    work = scenario_workdir(scenario, db_fixture=db_fixture)
    p = _provider_from_workdir(work)
    inp = work / "input"
    sdd = work / "solve_data"

    df_legacy = derive_pdtProcess(inp, sdd, provider=p)
    df_vec = derive_pdtProcess_vectorized(inp, sdd, provider=p)

    if expect_branch:
        # Non-vacuous: the stochastic fixture must produce rows AND
        # branch-period rows (period names carrying a ``_`` branch
        # suffix), so the fold branches are actually exercised.  The
        # non-stochastic ``fullYear`` fixture legitimately has an empty
        # ``process_TimeParam_in_use`` domain (no time-varying process
        # params) → 0 rows; parity at 0 rows is still valid parity, so
        # no height assertion there.
        assert df_legacy.height > 0, (
            f"{scenario}: legacy pdtProcess produced 0 rows — stochastic "
            f"fold gate is vacuous (check the reconstructed Provider)"
        )
        saw_branch = any(
            "_" in d for d in df_legacy["period"].unique().to_list()
        )
        assert saw_branch, (
            f"{scenario}: no branch-period rows — fold gate is vacuous"
        )

    # Tier A — strict byte-parity.
    assert df_vec.equals(df_legacy), (
        f"{scenario}: vectorized pdtProcess != legacy (Tier-A byte "
        f"parity). legacy {df_legacy.shape}, vec {df_vec.shape}"
    )


def test_vectorized_pdtProcess_exercises_pbt_fold(tmp_path):
    """Drive the actual branch-1/branch-2 fold join with a synthetic
    ``pbt_process`` and gate byte-parity vs the legacy scalar cascade.

    Both shipped fixtures (``fullYear`` and
    ``2_day_stochastic_dispatch``) have an EMPTY ``pbt_process.csv`` —
    their process-param values resolve through ``p``/def1, so the fold
    join (``build_fold_frame``) never fires on them.  This synthetic
    fixture has process-level pbt rows that exercise BOTH critique
    corrections from ``specs/vectorize_per_roll.md`` §4:

    * **Multi-parent multiplicity (S2):** ``solar`` has a single
      ``(b1, s1, t1)`` pbt value shared by two parents ``pA``/``pB`` of
      child period ``dchild`` → must be summed ONCE PER PARENT (= 20.0),
      not single-counted.
    * **Stoch fall-through (S2/§12.6):** ``wind`` is a stochastic entity
      but has no branch (``tb_for_d``) under ``dchild`` (branch-1 miss)
      → must FALL THROUGH to the branch-2 parent fold (= 2.0), not be
      dropped because it is stochastic.
    * **Stoch fold (branch 1):** ``wind`` under ``d1`` sums two branches
      ``b1``+``b2`` (= 3.0).

    The legacy ``derive_pdtProcess`` (scalar ``PdtLookup.get`` loop) is
    the oracle; ``derive_pdtProcess_vectorized`` must be byte-identical.
    """
    from flextool.engine_polars._flex_data_provider import FlexDataProvider

    provider = FlexDataProvider()

    def put(parent: str, stem: str, df: pl.DataFrame) -> None:
        provider.put(f"{parent}/{stem}", df)
        provider.put(stem, df)

    # pbt_process: (process, param, branch, time_start, time, value)
    put("input", "pbt_process", pl.DataFrame({
        "process": ["wind", "wind", "solar"],
        "param": ["cap", "cap", "cap"],
        "branch": ["b1", "b2", "b1"],
        "time_start": ["s1", "s1", "s1"],
        "time": ["t1", "t1", "t1"],
        "value": [1.0, 2.0, 10.0],
    }))
    # Empty pd/pt/p so the fold (and 0-default) are the only resolvers.
    put("input", "pd_process", pl.DataFrame(schema={
        "process": pl.Utf8, "param": pl.Utf8, "period": pl.Utf8,
        "value": pl.Float64}))
    put("input", "pt_process", pl.DataFrame(schema={
        "process": pl.Utf8, "param": pl.Utf8, "time": pl.Utf8,
        "value": pl.Float64}))
    put("input", "p_process", pl.DataFrame(schema={
        "process": pl.Utf8, "param": pl.Utf8, "value": pl.Float64}))
    # first_timesteps: (period, time_start)
    put("solve_data", "first_timesteps", pl.DataFrame({
        "period": ["d1", "dchild"], "time_start": ["s1", "s1"]}))
    # solve_branch__time_branch: (period, branch) — d1 has 2 branches;
    # parents pA/pB each have one branch b1; dchild has NONE (forces the
    # stoch fall-through for wind).
    put("solve_data", "solve_branch__time_branch", pl.DataFrame({
        "period": ["d1", "d1", "pA", "pB"],
        "branch": ["b1", "b2", "b1", "b1"]}))
    # period__branch: read with key_col=1 → pe_for_d[col1] = [col0].
    # dchild's parents are pA and pB.
    put("solve_data", "period__branch", pl.DataFrame({
        "parent": ["pA", "pB"], "child": ["dchild", "dchild"]}))
    put("input", "group__process", pl.DataFrame({
        "group": ["g"], "process": ["wind"]}))
    put("input", "groupIncludeStochastics", pl.DataFrame({"group": ["g"]}))
    # Domain (process, param) × dt (period, time) — entity-major.
    put("solve_data", "process_TimeParam_in_use", pl.DataFrame({
        "process": ["wind", "solar"], "param": ["cap", "cap"]}))
    put("solve_data", "steps_in_use", pl.DataFrame({
        "period": ["d1", "dchild"], "time": ["t1", "t1"]}))

    inp = tmp_path / "input"
    sdd = tmp_path / "solve_data"

    df_legacy = derive_pdtProcess(inp, sdd, provider=provider)
    df_vec = derive_pdtProcess_vectorized(inp, sdd, provider=provider)

    # Non-vacuous: the fold must actually have produced the corrected
    # values (so the join path, not just the 0-default, is exercised).
    legacy_map = {
        (r[0], r[2]): r[4] for r in df_legacy.iter_rows()
    }  # (process, period) -> value-string (time is constant "t1")
    assert legacy_map[("wind", "d1")] == repr(3.0), legacy_map
    assert legacy_map[("wind", "dchild")] == repr(2.0), legacy_map
    assert legacy_map[("solar", "dchild")] == repr(20.0), legacy_map
    assert legacy_map[("solar", "d1")] == repr(0.0), legacy_map

    assert df_vec.equals(df_legacy), (
        f"synthetic pbt fold: vectorized != legacy.\n"
        f"legacy:\n{df_legacy}\nvec:\n{df_vec}"
    )
