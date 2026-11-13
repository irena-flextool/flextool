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

from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.preprocess.emit._pdt_params import (
    derive_pdtProcess,
    derive_pdtProcess_vectorized,
)
from tests.engine._fan_fold_fixture import FAN_CASES


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
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider

    provider = FlexDataProvider()

    def put(parent: str, stem: str, df: pl.DataFrame) -> None:
        provider.put(f"{parent}/{stem}", df)
        provider.put(stem, df)

    # pbt_process: (process, param, branch, time_start, time, value)
    put("solve_data", "pbt_process", pl.DataFrame({
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
    put("solve_data", "pt_process", pl.DataFrame(schema={
        "process": pl.Utf8, "param": pl.Utf8, "time": pl.Utf8,
        "value": pl.Float64}))
    put("input", "p_process", pl.DataFrame(schema={
        "process": pl.Utf8, "param": pl.Utf8, "value": pl.Float64}))
    # first_timesteps: (period, time_start)
    put("solve_data", "first_timesteps", pl.DataFrame({
        "period": ["d1", "dchild"], "time_start": ["s1", "s1"]}))
    # Every period is a fan period whose fan start is its own first step (its tiers stay [own]); without
    # these rows each period would lie before the reveal, where only the solve's start is read (item 250).
    put("solve_data", "branch_time_start", pl.DataFrame({
        "period": ["d1", "dchild"], "step": ["s1", "s1"], "read_first": ["own", "own"]}))
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


def _branch_provider(pd_rows: dict, p_rows: dict, pbt_rows: list, pt_rows: "dict | None" = None,
                      pdt_rows: "dict | None" = None):
    """Synthetic stochastic provider: anchor ``d1`` with one branch period ``d1_b``
    (``period__branch`` (d1, d1), (d1, d1_b)); ``d1`` has time branch ``b1``."""
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider

    pt_rows = pt_rows or {}
    pdt_rows = pdt_rows or {}
    provider = FlexDataProvider()

    def put(parent: str, stem: str, df: pl.DataFrame) -> None:
        provider.put(f"{parent}/{stem}", df)
        provider.put(stem, df)

    put("solve_data", "pbt_process", pl.DataFrame(pbt_rows, schema={
        "process": pl.Utf8, "param": pl.Utf8, "branch": pl.Utf8,
        "time_start": pl.Utf8, "time": pl.Utf8, "value": pl.Float64},
        orient="row"))
    put("input", "pd_process", pl.DataFrame(
        [(*k, v) for k, v in pd_rows.items()], schema={
            "process": pl.Utf8, "param": pl.Utf8, "period": pl.Utf8,
            "value": pl.Float64}, orient="row"))
    put("solve_data", "pt_process", pl.DataFrame(
        [(*k, v) for k, v in pt_rows.items()], schema={
            "process": pl.Utf8, "param": pl.Utf8, "time": pl.Utf8,
            "value": pl.Float64}, orient="row"))
    put("input", "p_process", pl.DataFrame(
        [(*k, v) for k, v in p_rows.items()], schema={
            "process": pl.Utf8, "param": pl.Utf8, "value": pl.Float64},
        orient="row"))
    put("solve_data", "pdt_process", pl.DataFrame(
        [(*k, v) for k, v in pdt_rows.items()], schema={
            "process": pl.Utf8, "param": pl.Utf8, "period": pl.Utf8, "time": pl.Utf8,
            "value": pl.Float64}, orient="row"))
    put("solve_data", "first_timesteps", pl.DataFrame({
        "period": ["d1", "d1_b"], "time_start": ["s1", "s1"]}))
    # Every period is a fan period whose fan start is its own first step (its tiers stay [own]); without
    # these rows each period would lie before the reveal, where only the solve's start is read (item 250).
    put("solve_data", "branch_time_start", pl.DataFrame({
        "period": ["d1", "d1_b"], "step": ["s1", "s1"], "read_first": ["own", "own"]}))
    put("solve_data", "solve_branch__time_branch", pl.DataFrame({
        "period": ["d1"], "branch": ["b1"]}))
    put("solve_data", "period__branch", pl.DataFrame({
        "period": ["d1", "d1"], "branch": ["d1", "d1_b"]}))
    put("input", "group__process", pl.DataFrame(schema={
        "group": pl.Utf8, "process": pl.Utf8}))
    put("input", "groupIncludeStochastics", pl.DataFrame(schema={
        "group": pl.Utf8}))
    procs = sorted({k[0] for k in pd_rows} | {k[0] for k in p_rows}
                   | {r[0] for r in pbt_rows} | {k[0] for k in pt_rows} | {k[0] for k in pdt_rows})
    put("solve_data", "process_TimeParam_in_use", pl.DataFrame({
        "process": procs, "param": ["oc"] * len(procs)}))
    put("solve_data", "steps_in_use", pl.DataFrame({
        "period": ["d1", "d1_b"], "time": ["t1", "t1"]}))
    return provider


def test_vectorized_pdtProcess_branch_period_inherits_anchor_pd(tmp_path):
    """A stochastic branch period without a ``pd`` row of its own takes its
    anchor period's ``pd`` row (#55), in both twins, and the rest of the
    cascade order holds:

    * ``u``: ``pd`` only for the anchor ``d1`` → ``d1_b`` gets 3.0, not the
      ``p`` scalar 1.0 it got before (non-vacuous).
    * ``v``: ``d1_b`` authors its own ``pd`` row (9.0) → it wins.
    * ``w``: anchor ``pd`` 5.0 plus a ``pbt`` parent-fold hit for ``d1_b``
      (branch ``b1`` of the anchor) → the fold (7.0) still wins.
    * ``x``: a genuine period x time (``pdt``) row for the anchor on top of
      a ``pd`` row AND a ``pt`` row of the same param — the pdt tier
      outranks both on the anchor, and ``d1_b`` (no pdt row of its own)
      inherits the anchor's pdt row rather than falling through to ``pd``.
    """
    provider = _branch_provider(
        pd_rows={("u", "oc", "d1"): 3.0, ("v", "oc", "d1"): 4.0,
                 ("v", "oc", "d1_b"): 9.0, ("w", "oc", "d1"): 5.0,
                 ("x", "oc", "d1"): 10.0},
        p_rows={("u", "oc"): 1.0},
        pbt_rows=[("w", "oc", "b1", "s1", "t1", 7.0)],
        pt_rows={("x", "oc", "t1"): 20.0},
        pdt_rows={("x", "oc", "d1", "t1"): 30.0})
    inp = tmp_path / "input"
    sdd = tmp_path / "solve_data"
    df_legacy = derive_pdtProcess(inp, sdd, provider=provider)
    df_vec = derive_pdtProcess_vectorized(inp, sdd, provider=provider)

    leg = {(r[0], r[2]): r[4] for r in df_legacy.iter_rows()}
    assert leg[("u", "d1")] == leg[("u", "d1_b")] == repr(3.0), leg
    assert leg[("u", "d1_b")] != repr(1.0), "branch period still on the p scalar"
    assert leg[("v", "d1")] == repr(4.0), leg
    assert leg[("v", "d1_b")] == repr(9.0), leg
    assert leg[("w", "d1_b")] == repr(7.0), leg
    assert leg[("x", "d1")] == repr(30.0), "pdt must outrank pd and pt on the anchor"
    assert leg[("x", "d1_b")] == repr(30.0), "d1_b must inherit the anchor's pdt row, not fall through to pd"
    assert df_vec.equals(df_legacy), (
        f"branch-period pd: vectorized != legacy.\n"
        f"legacy:\n{df_legacy}\nvec:\n{df_vec}")


@pytest.mark.parametrize("case", FAN_CASES)
def test_vectorized_pdtProcess_fan_start_fallback(case, tmp_path):
    """Items 228 / 255: a non-empty ``fan_for_d`` (``branch_time_start``) -- each fold cell reads the forecast at
    the fan start (``read_first`` = ``fan``, as every solve writes it; a cell it does not cover that a later
    forecast does is an input error in both twins), or at its period's own first step first in a frame that says
    ``own`` -- byte-identical in both twins,
    values hand-derived in
    :mod:`tests.engine._fan_fold_fixture`."""
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
    from tests.engine._fan_fold_fixture import (
        STEPS, expected_fold, fan_case, pbt_frame, put_fan_scaffold, uncovered_error)

    provider = FlexDataProvider()

    def put(parent: str, stem: str, df: pl.DataFrame) -> None:
        provider.put(f"{parent}/{stem}", df)
        provider.put(stem, df)

    rolling, rows = fan_case(case)
    put_fan_scaffold(put, rolling=rolling)
    put("solve_data", "pbt_process", pbt_frame({"process": "entity", "param": "cap"}, rows=rows))
    put("input", "pd_process", pl.DataFrame(schema={
        "process": pl.Utf8, "param": pl.Utf8, "period": pl.Utf8, "value": pl.Float64}))
    put("solve_data", "pt_process", pl.DataFrame(schema={
        "process": pl.Utf8, "param": pl.Utf8, "time": pl.Utf8, "value": pl.Float64}))
    put("input", "p_process", pl.DataFrame(schema={
        "process": pl.Utf8, "param": pl.Utf8, "value": pl.Float64}))
    put("input", "group__process", pl.DataFrame({"group": ["g"], "process": ["X"]}))
    put("solve_data", "process_TimeParam_in_use", pl.DataFrame({
        "process": ["X", "Y"], "param": ["cap", "cap"]}))

    inp, sdd = tmp_path / "input", tmp_path / "solve_data"
    if case == "uncovered":
        for derive in (derive_pdtProcess, derive_pdtProcess_vectorized):
            with pytest.raises(FlexToolConfigError, match=uncovered_error("unit or connection 'Y'", "cap")):
                derive(inp, sdd, provider=provider)
        return
    df_legacy = derive_pdtProcess(inp, sdd, provider=provider)
    df_vec = derive_pdtProcess_vectorized(inp, sdd, provider=provider)
    leg = {(r[0], r[2], r[3]): r[4] for r in df_legacy.iter_rows()}
    for e in ("X", "Y"):
        for d, t in STEPS:
            assert leg[(e, d, t)] == repr(expected_fold(rolling).get((e, d, t), 0.0)), (e, d, t, leg)
    assert df_vec.equals(df_legacy), f"fan fallback: vectorized != legacy.\nlegacy:\n{df_legacy}\nvec:\n{df_vec}"
