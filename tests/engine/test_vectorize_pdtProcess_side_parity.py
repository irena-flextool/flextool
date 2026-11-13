"""Byte-parity gate for the vectorized ``pdtProcess_{source,sink}`` derives.

Tier A (design §1): the vectorized ``_derive_pdtProcess_side_vectorized``
must produce a frame BYTE-IDENTICAL to the legacy per-cell-loop
``_derive_pdtProcess_side`` on BOTH fixtures and for BOTH the source and
sink sides:

* ``fullYear``                     — rolling, non-stochastic.
* ``2_day_stochastic_dispatch``    — stochastic (exercises the
  branch-1 stochastic fold + branch-2 parent-period fold).

PerSide differs from ``pdtProcess`` in three ways the gate must honour
(design §4 / §5 PerSide rows + the M1 note):

* 3-col entity key ``(process, side, param)`` read via ``_read_triples``
  from ``process_<side>_sourceSinkTimeParam_in_use.csv``.
* 6-branch cascade: fold → ``pd`` → ``pt`` → ``p`` → ``0.0`` — NO def1,
  NO class-default.
* (M1) the stochastic-fold membership test keys on the ``process``
  column ALONE (``p ∈ _stoch_process``), not the full 3-col key, even
  though the pbt join key is ``(process, side, param, tb, ts, t)``.

The Provider is reconstructed by globbing EVERY CSV in ``work/input``
and ``work/solve_data`` and dual-registering each under both the
parent-qualified key (``solve_data/<stem>``) and the bare ``<stem>``
key (design §6 / S6 — glob, do not copy the under-registering 7-item
helper in ``test_pbt_node_inflow.py``).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.preprocess.emit._pdt_params import (
    _derive_pdtProcess_side,
    _derive_pdtProcess_side_vectorized,
    derive_pdtProcess_sink,
    derive_pdtProcess_source,
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


def _legacy_side(inp: Path, sdd: Path, *, side_col: str, provider):
    """Legacy per-cell-loop derive for the given side."""
    return _derive_pdtProcess_side(
        pbt_csv=inp / f"pbt_process_{side_col}.csv",
        pd_csv=inp / f"pd_process_{side_col}.csv",
        pt_csv=inp / f"pt_process_{side_col}.csv",
        p_csv=inp / f"p_process_{side_col}.csv",
        period_time_first_csv=sdd / "first_timesteps.csv",
        solve_branch_csv=sdd / "solve_branch__time_branch.csv",
        period_branch_csv=sdd / "period__branch.csv",
        group_process_csv=inp / "group__process.csv",
        group_stochastic_csv=inp / "groupIncludeStochastics.csv",
        domain_csv=sdd / f"process_{side_col}_sourceSinkTimeParam_in_use.csv",
        dt_csv=sdd / "steps_in_use.csv",
        side_col=side_col,
        provider=provider,
    )


def _vec_side(inp: Path, sdd: Path, *, side_col: str, provider):
    """Vectorized derive for the given side."""
    return _derive_pdtProcess_side_vectorized(
        pbt_csv=inp / f"pbt_process_{side_col}.csv",
        pd_csv=inp / f"pd_process_{side_col}.csv",
        pt_csv=inp / f"pt_process_{side_col}.csv",
        p_csv=inp / f"p_process_{side_col}.csv",
        period_time_first_csv=sdd / "first_timesteps.csv",
        solve_branch_csv=sdd / "solve_branch__time_branch.csv",
        period_branch_csv=sdd / "period__branch.csv",
        group_process_csv=inp / "group__process.csv",
        group_stochastic_csv=inp / "groupIncludeStochastics.csv",
        domain_csv=sdd / f"process_{side_col}_sourceSinkTimeParam_in_use.csv",
        dt_csv=sdd / "steps_in_use.csv",
        side_col=side_col,
        provider=provider,
    )


_CASES = [
    ("fullYear", "main"),
    ("2_day_stochastic_dispatch", "stochastic"),
]


@pytest.mark.parametrize("scenario,db_fixture", _CASES)
@pytest.mark.parametrize("side_col", ["source", "sink"])
def test_vectorized_pdtProcess_side_matches_legacy(
    scenario, db_fixture, side_col, scenario_workdir,
):
    work = scenario_workdir(scenario, db_fixture=db_fixture)
    p = _provider_from_workdir(work)
    inp = work / "input"
    sdd = work / "solve_data"

    df_legacy = _legacy_side(inp, sdd, side_col=side_col, provider=p)
    df_vec = _vec_side(inp, sdd, side_col=side_col, provider=p)

    # Tier A — strict byte-parity (empty-domain fixtures still assert
    # parity at 0 rows, which is valid parity).
    assert df_vec.equals(df_legacy), (
        f"{scenario}/{side_col}: vectorized pdtProcess_{side_col} != legacy "
        f"(Tier-A byte parity). legacy {df_legacy.shape}, vec {df_vec.shape}"
    )


@pytest.mark.parametrize("scenario,db_fixture", _CASES)
def test_public_wrappers_match_vectorized(
    scenario, db_fixture, scenario_workdir,
):
    """The public ``derive_pdtProcess_{source,sink}`` (legacy oracle) and
    the vectorized side derive agree end-to-end via the public entry."""
    work = scenario_workdir(scenario, db_fixture=db_fixture)
    p = _provider_from_workdir(work)
    inp = work / "input"
    sdd = work / "solve_data"

    src_legacy = derive_pdtProcess_source(inp, sdd, provider=p)
    src_vec = _vec_side(inp, sdd, side_col="source", provider=p)
    assert src_vec.equals(src_legacy)

    snk_legacy = derive_pdtProcess_sink(inp, sdd, provider=p)
    snk_vec = _vec_side(inp, sdd, side_col="sink", provider=p)
    assert snk_vec.equals(snk_legacy)


def _put_factory(provider):
    def put(parent: str, stem: str, df: pl.DataFrame) -> None:
        provider.put(f"{parent}/{stem}", df)
        provider.put(stem, df)
    return put


@pytest.mark.parametrize("side_col", ["source", "sink"])
def test_vectorized_pdtProcess_side_exercises_pbt_fold(side_col, tmp_path):
    """Drive the actual branch-1/branch-2 PerSide fold join with a
    synthetic ``pbt_process_<side>`` and gate byte-parity vs the legacy
    scalar cascade.

    Both shipped fixtures have an EMPTY ``pbt_process_source.csv`` /
    ``pbt_process_sink.csv`` (mirroring ``pdtProcess``'s empty
    ``pbt_process.csv``) — their per-arc values resolve through
    ``pd``/``pt``/``p``/0, so the fold join (``build_fold_frame``) never
    fires on real data.  This synthetic fixture has per-side pbt rows
    that exercise ALL THREE corrected fold paths from
    ``specs/vectorize_per_roll.md`` §4 PLUS the M1 process-only stoch
    filter:

    * **Stoch fold (branch 1):** ``wind`` under ``d1`` sums two branches
      ``b1``+``b2`` (= 3.0).
    * **Stoch fall-through (S2/§12.6):** ``wind`` is a stochastic process
      but has NO branch under ``dchild`` (branch-1 miss) → must FALL
      THROUGH to the branch-2 parent fold (= 2.0), not be dropped because
      it is stochastic.
    * **Multi-parent multiplicity (S2):** ``solar`` has a single
      ``(b1, s1, t1)`` pbt value shared by two parents ``pA``/``pB`` of
      child period ``dchild`` → summed ONCE PER PARENT (= 20.0).
    * **(M1) process-only stoch filter:** ``wind`` carries TWO distinct
      sides (``nA``/``nB``); membership keys on ``process == "wind"``
      alone, so BOTH side-rows take the stochastic branch.  A negative
      control ``solar`` (non-stochastic process) must NOT take branch 1
      even though its pbt rows share branches with ``wind``.

    The legacy ``_derive_pdtProcess_side`` (scalar ``PdtLookupPerSide.get``
    loop) is the oracle; ``_derive_pdtProcess_side_vectorized`` must be
    byte-identical.
    """
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider

    provider = FlexDataProvider()
    put = _put_factory(provider)

    # pbt_process_<side>: (process, side, param, branch, time_start, time,
    # value).  wind carries two sides nA/nB; solar carries nA.
    put("input", f"pbt_process_{side_col}", pl.DataFrame({
        "process": ["wind", "wind", "wind", "wind", "solar"],
        side_col:  ["nA", "nA", "nB", "nB", "nA"],
        "param":   ["cap", "cap", "cap", "cap", "cap"],
        "branch":  ["b1", "b2", "b1", "b2", "b1"],
        "time_start": ["s1", "s1", "s1", "s1", "s1"],
        "time":    ["t1", "t1", "t1", "t1", "t1"],
        "value":   [1.0, 2.0, 1.0, 2.0, 10.0],
    }))
    # Empty pd/pt/p so the fold (and 0-default) are the only resolvers.
    put("input", f"pd_process_{side_col}", pl.DataFrame(schema={
        "process": pl.Utf8, side_col: pl.Utf8, "param": pl.Utf8,
        "period": pl.Utf8, "value": pl.Float64}))
    put("input", f"pt_process_{side_col}", pl.DataFrame(schema={
        "process": pl.Utf8, side_col: pl.Utf8, "param": pl.Utf8,
        "time": pl.Utf8, "value": pl.Float64}))
    put("input", f"p_process_{side_col}", pl.DataFrame(schema={
        "process": pl.Utf8, side_col: pl.Utf8, "param": pl.Utf8,
        "value": pl.Float64}))
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
    # group__process flags wind (process-level) as stochastic; solar is
    # the non-stochastic negative control.
    put("input", "group__process", pl.DataFrame({
        "group": ["g"], "process": ["wind"]}))
    put("input", "groupIncludeStochastics", pl.DataFrame({"group": ["g"]}))
    # Domain (process, side, param) × dt (period, time) — entity-major.
    # wind appears on TWO sides to exercise the process-only filter.
    put("solve_data", f"process_{side_col}_sourceSinkTimeParam_in_use",
        pl.DataFrame({
            "process": ["wind", "wind", "solar"],
            side_col:  ["nA", "nB", "nA"],
            "param":   ["cap", "cap", "cap"]}))
    put("solve_data", "steps_in_use", pl.DataFrame({
        "period": ["d1", "dchild"], "time": ["t1", "t1"]}))

    inp = tmp_path / "input"
    sdd = tmp_path / "solve_data"

    df_legacy = _legacy_side(inp, sdd, side_col=side_col, provider=provider)
    df_vec = _vec_side(inp, sdd, side_col=side_col, provider=provider)

    # Non-vacuous: the fold must actually have produced the corrected
    # values (so the join path, not just the 0-default, is exercised).
    # (process, side, period) -> value-string (time is constant "t1").
    legacy_map = {(r[0], r[1], r[3]): r[5] for r in df_legacy.iter_rows()}
    # wind/d1: both sides take branch 1 (process-only filter) = 3.0.
    assert legacy_map[("wind", "nA", "d1")] == repr(3.0), legacy_map
    assert legacy_map[("wind", "nB", "d1")] == repr(3.0), legacy_map
    # wind/dchild: branch-1 miss → falls through to parent fold = 2.0.
    assert legacy_map[("wind", "nA", "dchild")] == repr(2.0), legacy_map
    assert legacy_map[("wind", "nB", "dchild")] == repr(2.0), legacy_map
    # solar (non-stoch process): branch 1 NOT taken; falls to parent
    # fold under dchild (multi-parent, once per parent) = 20.0.
    assert legacy_map[("solar", "nA", "dchild")] == repr(20.0), legacy_map
    # solar/d1: non-stoch, no parent for d1 → 0.0 default.
    assert legacy_map[("solar", "nA", "d1")] == repr(0.0), legacy_map

    assert df_vec.equals(df_legacy), (
        f"synthetic pbt fold ({side_col}): vectorized != legacy.\n"
        f"legacy:\n{df_legacy}\nvec:\n{df_vec}"
    )


@pytest.mark.parametrize("side_col", ["source", "sink"])
def test_vectorized_pdtProcess_side_branch_period_inherits_anchor_pd(
        side_col, tmp_path):
    """PerSide (4-tuple ``pd`` key, period at position 3): a stochastic
    branch period without a ``pd`` row of its own takes its anchor period's
    (#55), in both twins.

    * ``(u, nA)``: anchor ``pd`` 3.0, ``p`` scalar 1.0 → ``d1_b`` gets 3.0
      and no longer the 1.0 it got before (non-vacuous).
    * ``(u, nB)``: ``d1_b`` authors its own ``pd`` (9.0) → it wins; the
      other side of the same process is unaffected by it.
    * ``(w, nA)``: anchor ``pd`` 5.0 and a ``pbt`` parent-fold hit → the
      fold (7.0) still wins.
    """
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider

    provider = FlexDataProvider()
    put = _put_factory(provider)
    put("input", f"pbt_process_{side_col}", pl.DataFrame({
        "process": ["w"], side_col: ["nA"], "param": ["oc"],
        "branch": ["b1"], "time_start": ["s1"], "time": ["t1"],
        "value": [7.0]}))
    put("input", f"pd_process_{side_col}", pl.DataFrame({
        "process": ["u", "u", "u", "w"], side_col: ["nA", "nB", "nB", "nA"],
        "param": ["oc"] * 4, "period": ["d1", "d1", "d1_b", "d1"],
        "value": [3.0, 4.0, 9.0, 5.0]}))
    put("input", f"pt_process_{side_col}", pl.DataFrame(schema={
        "process": pl.Utf8, side_col: pl.Utf8, "param": pl.Utf8,
        "time": pl.Utf8, "value": pl.Float64}))
    put("input", f"p_process_{side_col}", pl.DataFrame({
        "process": ["u"], side_col: ["nA"], "param": ["oc"], "value": [1.0]}))
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
    put("solve_data", f"process_{side_col}_sourceSinkTimeParam_in_use",
        pl.DataFrame({"process": ["u", "u", "w"],
                      side_col: ["nA", "nB", "nA"], "param": ["oc"] * 3}))
    put("solve_data", "steps_in_use", pl.DataFrame({
        "period": ["d1", "d1_b"], "time": ["t1", "t1"]}))

    inp = tmp_path / "input"
    sdd = tmp_path / "solve_data"
    df_legacy = _legacy_side(inp, sdd, side_col=side_col, provider=provider)
    df_vec = _vec_side(inp, sdd, side_col=side_col, provider=provider)

    leg = {(r[0], r[1], r[3]): r[5] for r in df_legacy.iter_rows()}
    assert leg[("u", "nA", "d1_b")] == leg[("u", "nA", "d1")] == repr(3.0), leg
    assert leg[("u", "nA", "d1_b")] != repr(1.0), "branch period still on the p scalar"
    assert leg[("u", "nB", "d1")] == repr(4.0), leg
    assert leg[("u", "nB", "d1_b")] == repr(9.0), leg
    assert leg[("w", "nA", "d1_b")] == repr(7.0), leg
    assert df_vec.equals(df_legacy), (
        f"{side_col}: branch-period pd: vectorized != legacy.\n"
        f"legacy:\n{df_legacy}\nvec:\n{df_vec}")


@pytest.mark.parametrize("side_col", ["source", "sink"])
@pytest.mark.parametrize("case", ["single", "rolling_lookahead", "rolling_uncovered"])
def test_vectorized_pdtProcess_side_fan_start_fallback(side_col, case, tmp_path):
    """Item 228: a non-empty ``fan_for_d`` (``branch_time_start``) -- each fold cell reads the forecast at its
    period's own first step, else at the fan start (a roll's look-ahead period: the fan start only; a cell it
    does not cover that a later forecast does is an input error in both twins) -- byte-identical in both twins,
    values hand-derived in
    :mod:`tests.engine._fan_fold_fixture`."""
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
    from tests.engine._fan_fold_fixture import (
        STEPS, expected_fold, fan_case, pbt_frame, put_fan_scaffold, uncovered_error)

    provider = FlexDataProvider()
    put = _put_factory(provider)
    rolling, rows = fan_case(case)
    put_fan_scaffold(put, rolling=rolling)
    put("input", f"pbt_process_{side_col}", pbt_frame(
        {"process": "entity", side_col: "nA", "param": "cap"}, rows=rows))
    put("input", f"pd_process_{side_col}", pl.DataFrame(schema={
        "process": pl.Utf8, side_col: pl.Utf8, "param": pl.Utf8, "period": pl.Utf8, "value": pl.Float64}))
    put("input", f"pt_process_{side_col}", pl.DataFrame(schema={
        "process": pl.Utf8, side_col: pl.Utf8, "param": pl.Utf8, "time": pl.Utf8, "value": pl.Float64}))
    put("input", f"p_process_{side_col}", pl.DataFrame(schema={
        "process": pl.Utf8, side_col: pl.Utf8, "param": pl.Utf8, "value": pl.Float64}))
    put("input", "group__process", pl.DataFrame({"group": ["g"], "process": ["X"]}))
    put("solve_data", f"process_{side_col}_sourceSinkTimeParam_in_use", pl.DataFrame({
        "process": ["X", "Y"], side_col: ["nA", "nA"], "param": ["cap", "cap"]}))

    inp, sdd = tmp_path / "input", tmp_path / "solve_data"
    if case == "rolling_uncovered":
        for derive in (_legacy_side, _vec_side):
            with pytest.raises(FlexToolConfigError, match=uncovered_error("unit or connection 'Y' (nA)", "cap")):
                derive(inp, sdd, side_col=side_col, provider=provider)
        return
    df_legacy = _legacy_side(inp, sdd, side_col=side_col, provider=provider)
    df_vec = _vec_side(inp, sdd, side_col=side_col, provider=provider)
    leg = {(r[0], r[3], r[4]): r[5] for r in df_legacy.iter_rows()}
    for e in ("X", "Y"):
        for d, t in STEPS:
            assert leg[(e, d, t)] == repr(expected_fold(rolling).get((e, d, t), 0.0)), (e, d, t, leg)
    assert df_vec.equals(df_legacy), f"fan fallback: vectorized != legacy.\nlegacy:\n{df_legacy}\nvec:\n{df_vec}"
