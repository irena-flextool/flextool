"""Parity gate for the vectorized ``pdtNodeInflow`` derive.

``pdtNodeInflow`` is the single-entity-key (``node``, NO param axis)
3-branch family: (1) stochastic fold, (2) parent-period fold, (3) the
deterministic gated additive SUM of the four inflow-scaling methods
(``scale_to_annual_flow``, ``scale_in_proportion``,
``scale_to_annual_and_peak_flow``, ``use_original``).  The ``pbt_node_inflow``
key is a 4-tuple ``(node, tb, ts, t)`` (no param), so the fold runs on the
single ``["node"]`` entity-key column (identical skeleton to
``pdtProfile``); branch 3 is a varCost-style membership-gated sum with a
signed-zero normalization.

The fold (branches 1-2) DOES fire on the ``stochastics_pbt_inflow``
fixture, which authors an ``inflow`` 3d_map routing to
``pbt_node_inflow.csv``.  Branch 3 is genuinely exercised on the main
``fullYear`` fixture (inflow methods authored).

Tier policy (design §1): assert strict ``df_vec.equals(df_legacy)``
(Tier A) first.  A multi-term fold or a multi-term branch-3 sum's pairwise
reduction can drift from the legacy sequential ``total += v`` by last-ULP
(``|Δ| ≤ ~1e-12``); that legitimately demotes the offending cells to
**Tier B** (parse both ``value`` columns to float, key cols ``.equals``,
values ``rtol ≤ 1e-12``).  Any NON-ULP difference is a real bug — those
are NOT tolerated; the test fails hard.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars._emit_period_params import (
    derive_pdtNodeInflow,
    derive_pdtNodeInflow_vectorized,
)

_KEY_COLS = ["node", "period", "time"]


def _provider_from_workdir(workdir: Path):
    """Reconstruct a Provider from every CSV in input/ + solve_data/.

    Dual-registers each frame under ``"<parent>/<stem>"`` AND the bare
    ``"<stem>"`` key so both ``_provider_key``-qualified lookups and any
    bare lookup resolve (design §6 / S6 — glob, do not under-register).
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


def _assert_parity(df_legacy: pl.DataFrame, df_vec: pl.DataFrame,
                   label: str) -> str:
    """Tier-A strict ``.equals``; demote to Tier B only on pure ULP drift.

    Returns ``"A"`` if byte-parity held, ``"B"`` if a legitimate
    last-ULP float-only demotion was applied (max ``|Δ| ≤ ~1e-12``).
    Raises on any structural mismatch or a non-ULP value difference
    (a real bug must STOP the gate, not be loosened away).
    """
    assert df_vec.columns == df_legacy.columns, (
        f"{label}: column mismatch legacy={df_legacy.columns} "
        f"vec={df_vec.columns}")
    assert df_vec.shape == df_legacy.shape, (
        f"{label}: shape mismatch legacy={df_legacy.shape} "
        f"vec={df_vec.shape}")

    if df_vec.equals(df_legacy):
        return "A"

    # Key columns MUST match byte-for-byte (row order + identity); only
    # the value column may drift, and only by last-ULP float noise.
    assert df_vec.select(_KEY_COLS).equals(df_legacy.select(_KEY_COLS)), (
        f"{label}: KEY columns differ — this is a structural bug, NOT a "
        f"float-ULP demotion. STOP.")

    leg_v = df_legacy["value"].cast(pl.Float64)
    vec_v = df_vec["value"].cast(pl.Float64)
    diff = (leg_v - vec_v).abs()
    max_abs = diff.max() or 0.0
    if max_abs > 1e-12:
        bad = df_legacy.with_columns(
            df_vec["value"].alias("vec_value"),
            diff.alias("abs_diff"),
        ).filter(pl.col("abs_diff") > 1e-12)
        raise AssertionError(
            f"{label}: value drift {max_abs:.3e} exceeds Tier-B tolerance "
            f"1e-12 — NOT a last-ULP demotion, a real bug:\n{bad}")
    return "B"


# --- helpers to reconstruct the inputs from a Provider ---------------------

def _pbt_inflow_from_provider(p) -> dict:
    """Rebuild the ``pbt_node_inflow`` dict (verbatim legacy reader)."""
    from flextool.engine_polars._emit_period_params import _cell_str

    out: dict = {}
    df = p.get("input/pbt_node_inflow")
    if df is None:
        return out
    for row in df.iter_rows():
        if len(row) < 5:
            continue
        c0, c1, c2, c3 = (_cell_str(row[0]), _cell_str(row[1]),
                          _cell_str(row[2]), _cell_str(row[3]))
        if c0 and c1 and c2 and c3:
            try:
                out[(c0, c1, c2, c3)] = float(row[4])
            except (ValueError, TypeError):
                continue
    return out


def _inflow_method_pairs_from_provider(p) -> frozenset:
    """Rebuild the ``node__inflow_method`` pair set (verbatim legacy)."""
    from flextool.engine_polars._emit_period_params import _cell_str

    out: set = set()
    df = p.get("solve_data/node__inflow_method")
    if df is None:
        return frozenset(out)
    for row in df.iter_rows():
        if len(row) < 2:
            continue
        c0, c1 = _cell_str(row[0]), _cell_str(row[1])
        if c0 and c1:
            out.add((c0, c1))
    return frozenset(out)


# --- Part 1: fullYear (main) — branch 3 exercised, fold inert --------------

def test_vectorized_pdtNodeInflow_fullYear(scenario_workdir):
    work = scenario_workdir("fullYear", db_fixture="main")
    p = _provider_from_workdir(work)
    inp = work / "input"
    sdd = work / "solve_data"

    df_legacy = derive_pdtNodeInflow(inp, sdd, provider=p)
    df_vec = derive_pdtNodeInflow_vectorized(inp, sdd, provider=p)

    # fullYear authors no pbt_node_inflow 3d_map ⇒ the fold is inert here;
    # branch 3 (the inflow-scaling sum) is the genuine exercise.
    pbt = _pbt_inflow_from_provider(p)
    assert not pbt, (
        "fullYear unexpectedly authors pbt_node_inflow — the fold can "
        "fire; re-check the tier detection on this fixture")

    # Non-vacuity: a real inflow-scaling method is authored, the node
    # domain is non-empty, and at least one branch-3 value is non-zero.
    pairs = _inflow_method_pairs_from_provider(p)
    methods = {m for (_n, m) in pairs}
    assert methods & {
        "scale_to_annual_flow", "scale_in_proportion",
        "scale_to_annual_and_peak_flow", "use_original",
    }, (
        "fullYear authors no inflow-scaling method — branch 3 is vacuous; "
        f"got methods={methods}")
    assert df_legacy.height > 0, (
        "fullYear: legacy pdtNodeInflow produced 0 rows — gate vacuous")
    nonzero = df_legacy.filter(
        pl.col("value").cast(pl.Float64) != 0.0).height
    assert nonzero > 0, (
        "fullYear: NO non-zero branch-3 value emitted — branch 3 is "
        "vacuous on this fixture")

    tier = _assert_parity(df_legacy, df_vec, "fullYear")
    print(f"\n[pdtNodeInflow parity] fullYear: Tier {tier} "
          f"(branch-3 exercised, {nonzero} non-zero cells, fold inert)")


# --- Part 2: 2_day_stochastic_dispatch -------------------------------------

def test_vectorized_pdtNodeInflow_stochastic(scenario_workdir):
    work = scenario_workdir(
        "2_day_stochastic_dispatch", db_fixture="stochastic")
    p = _provider_from_workdir(work)
    inp = work / "input"
    sdd = work / "solve_data"

    df_legacy = derive_pdtNodeInflow(inp, sdd, provider=p)
    df_vec = derive_pdtNodeInflow_vectorized(inp, sdd, provider=p)

    assert df_legacy.height > 0, (
        "2_day_stochastic_dispatch: legacy pdtNodeInflow produced 0 rows "
        "— gate vacuous")

    tier = _assert_parity(
        df_legacy, df_vec, "2_day_stochastic_dispatch")
    print(f"\n[pdtNodeInflow parity] 2_day_stochastic_dispatch: "
          f"Tier {tier}")


# --- Part 3: stochastics_pbt_inflow — THE FOLD FIRES (pbt_node_inflow) ------

def test_vectorized_pdtNodeInflow_pbt_inflow_fixture(scenario_workdir):
    """``stochastics_pbt_inflow`` authors an ``inflow`` 3d_map that routes
    to ``pbt_node_inflow.csv`` → branches 1/2 (the fold) genuinely fire on
    real data."""
    work = scenario_workdir(
        "2_day_stochastic_dispatch", db_fixture="stochastics_pbt_inflow")
    p = _provider_from_workdir(work)
    inp = work / "input"
    sdd = work / "solve_data"

    df_legacy = derive_pdtNodeInflow(inp, sdd, provider=p)
    df_vec = derive_pdtNodeInflow_vectorized(inp, sdd, provider=p)

    assert df_legacy.height > 0, (
        "stochastics_pbt_inflow: legacy pdtNodeInflow produced 0 rows — "
        "gate vacuous")

    # Non-vacuous fold: pbt_node_inflow IS authored on this fixture; the
    # fold (branches 1/2) genuinely fires.
    pbt = _pbt_inflow_from_provider(p)
    assert pbt, (
        "stochastics_pbt_inflow: pbt_node_inflow is EMPTY — the fold is "
        "NOT exercised on this fixture; the gate would be vacuous.  A "
        "fixture change has removed the inflow 3d_map; restore fold "
        "coverage")

    tier = _assert_parity(df_legacy, df_vec, "stochastics_pbt_inflow")
    print(f"\n[pdtNodeInflow parity] stochastics_pbt_inflow: Tier {tier} "
          f"(fold fires, pbt_node_inflow rows={len(pbt)})")


# --- Part 4: synthetic coverage (every differentiating branch) -------------

def test_vectorized_pdtNodeInflow_synthetic_branches(tmp_path):
    """Drive the vectorized derive vs the legacy scalar cascade on a
    synthetic fixture exercising EVERY differentiating branch of
    ``pdtNodeInflow``:

    Branch 3 (deterministic gated sum):

    * **scale_to_annual_flow:** ``annN`` (balance, af≠0) → pfa*pti.
    * **scale_in_proportion:** ``propN`` (balance, af≠0) → pfp*pti.
    * **scale_to_annual_and_peak_flow:** ``peakN`` (balance, af≠0 AND
      pk≠0) → slope*pti − section (signed-zero edge: under a period whose
      ``slope*pti − section == 0`` the cell must render ``"0.0"``).
    * **use_original:** ``origN`` (balance) → pti.
    * **af gated OFF (≠0.0, NOT is_not_null):** ``zeroAfN`` authors
      ``pdNode_af == 0.0`` AND a ``scale_to_annual_flow`` method → the
      term gates OFF → 0.0 (a real 0.0 key must NOT pass).
    * **non-balance-union → 0.0 floor:** ``nonBalN`` has a method + pti but
      is NOT in the balance union → 0.0.
    * **no_inflow excluded:** ``noInfN`` has method ``no_inflow`` → it must
      be ABSENT from the output domain entirely.

    Branches 1/2 (the fold):

    * **multi-term stoch fold:** ``stochN`` (stochastic) under ``d1`` has a
      ``pbt_node_inflow`` row hit by TWO ``(tb, ts)`` combos → summed
      (multi-term ⇒ genuine tier probe).
    * **stoch fall-through (branch-1 miss → branch-2 hit):** ``stochN``
      under ``dchild`` has NO branch (branch-1 miss) but its ``b1`` value
      is reached via the parent ``pA`` → branch-2 fold.

    The legacy ``derive_pdtNodeInflow`` (scalar cascade) is the oracle;
    ``derive_pdtNodeInflow_vectorized`` must match (Tier A expected —
    small integer-valued sums; Tier B acceptable on genuine ULP drift).
    """
    from flextool.engine_polars._flex_data_provider import FlexDataProvider

    provider = FlexDataProvider()

    def put(parent: str, stem: str, df: pl.DataFrame) -> None:
        provider.put(f"{parent}/{stem}", df)
        provider.put(stem, df)

    # node domain (entity-major order preserved).  noInfN is no_inflow →
    # excluded; the rest are eligible.
    put("input", "node", pl.DataFrame({
        "node": ["annN", "propN", "peakN", "origN", "zeroAfN",
                 "nonBalN", "stochN", "noInfN"],
    }))
    # inflow methods (single-valued per node).
    put("solve_data", "node__inflow_method", pl.DataFrame({
        "node": ["annN", "propN", "peakN", "origN", "zeroAfN",
                 "nonBalN", "noInfN"],
        "method": ["scale_to_annual_flow", "scale_in_proportion",
                   "scale_to_annual_and_peak_flow", "use_original",
                   "scale_to_annual_flow", "use_original", "no_inflow"],
    }))
    # balance union: everyone EXCEPT nonBalN (and stochN, which folds).
    put("solve_data", "nodeBalance", pl.DataFrame({
        "node": ["annN", "propN", "peakN", "origN", "zeroAfN"],
    }))
    put("solve_data", "nodeBalancePeriod", pl.DataFrame({
        "node": [],
    }, schema={"node": pl.Utf8}))

    # ptNode_inflow: (node, time, value) → pti.
    put("solve_data", "ptNode_inflow", pl.DataFrame({
        "node": ["annN", "propN", "peakN", "origN", "zeroAfN", "nonBalN"],
        "time": ["t1", "t1", "t1", "t1", "t1", "t1"],
        "value": [2.0, 3.0, 5.0, 9.0, 4.0, 11.0],
    }))
    # pdNode: (node, param, period, value).  af for ann/prop/peak ≠0;
    # zeroAfN authors af == 0.0 (gate OFF); peakN also authors peak ≠0.
    put("solve_data", "pdNode", pl.DataFrame({
        "node": ["annN", "propN", "peakN", "peakN", "zeroAfN"],
        "param": ["annual_flow", "annual_flow", "annual_flow",
                  "peak_inflow", "annual_flow"],
        "period": ["d1", "d1", "d1", "d1", "d1"],
        "value": [100.0, 100.0, 100.0, 7.0, 0.0],
    }))
    # period_flow_annual_multiplier: (node, period, value) → pfa.
    put("solve_data", "period_flow_annual_multiplier", pl.DataFrame({
        "node": ["annN"], "period": ["d1"], "value": [1.5]}))
    # period_flow_proportional_multiplier: (node, period, value) → pfp.
    put("solve_data", "period_flow_proportional_multiplier", pl.DataFrame({
        "node": ["propN"], "period": ["d1"], "value": [2.0]}))
    # new_old_slope / new_old_section: (node, period, value).  peakN under
    # d1: slope*pti − section = 4*5 − 20 = 0.0 (signed-zero edge → "0.0").
    put("solve_data", "new_old_slope", pl.DataFrame({
        "node": ["peakN"], "period": ["d1"], "value": [4.0]}))
    put("solve_data", "new_old_section", pl.DataFrame({
        "node": ["peakN"], "period": ["d1"], "value": [20.0]}))

    # --- fold inputs (branches 1/2) for stochN ---------------------------
    # pbt_node_inflow: (node, branch, time_start, time, value).  stochN b1
    # under d1 hit by TWO (tb, ts) combos (b1×s1, b1×s2) → multi-term
    # stoch sum 5+5 = 10.  b1 also reached via parent pA under dchild.
    put("input", "pbt_node_inflow", pl.DataFrame({
        "node": ["stochN", "stochN"],
        "branch": ["b1", "b1"],
        "time_start": ["s1", "s2"],
        "time": ["t1", "t1"],
        "value": [5.0, 5.0],
    }))
    # first_timesteps: (period, time_start) → ts_for_d.  d1 has TWO starts
    # (s1, s2) → the stoch fold sums two terms; dchild has s1.
    put("solve_data", "first_timesteps", pl.DataFrame({
        "period": ["d1", "d1", "dchild"],
        "time_start": ["s1", "s2", "s1"]}))
    # solve_branch__time_branch: (period, branch) → tb_for_d.  d1 has b1;
    # dchild has NONE (forces stochN's stoch fall-through); parent pA has b1.
    put("solve_data", "solve_branch__time_branch", pl.DataFrame({
        "period": ["d1", "pA"],
        "branch": ["b1", "b1"]}))
    # period__branch: read key_col=1 → pe_for_d[dchild] = [pA].
    put("solve_data", "period__branch", pl.DataFrame({
        "parent": ["pA"], "child": ["dchild"]}))
    # Stochastic group holding stochN.
    put("input", "group__node", pl.DataFrame({
        "group": ["g_node"], "node": ["stochN"]}))
    put("input", "groupIncludeStochastics", pl.DataFrame({
        "group": ["g_node"]}))

    # steps_in_use: (period, time).  Two periods d1, dchild (single time
    # t1 each) so the fold + branch-3 cells coexist.
    put("solve_data", "steps_in_use", pl.DataFrame({
        "period": ["d1", "dchild"], "time": ["t1", "t1"]}))

    inp = tmp_path / "input"
    sdd = tmp_path / "solve_data"

    df_legacy = derive_pdtNodeInflow(inp, sdd, provider=provider)
    df_vec = derive_pdtNodeInflow_vectorized(inp, sdd, provider=provider)

    # Oracle sanity: every differentiating branch landed where expected.
    # (node, period) -> value-string (time is constant "t1").
    leg = {(r[0], r[1]): r[3] for r in df_legacy.iter_rows()}
    assert ("noInfN", "d1") not in leg, (
        "no_inflow node leaked into the domain")  # excluded
    assert leg[("annN", "d1")] == repr(1.5 * 2.0), leg     # pfa*pti
    assert leg[("propN", "d1")] == repr(2.0 * 3.0), leg    # pfp*pti
    # peakN: slope*pti − section = 4*5 − 20 = 0.0 (signed-zero → "0.0").
    assert leg[("peakN", "d1")] == repr(0.0), leg
    assert leg[("origN", "d1")] == repr(9.0), leg          # pti
    assert leg[("zeroAfN", "d1")] == repr(0.0), leg        # af==0 → gated
    assert leg[("nonBalN", "d1")] == repr(0.0), leg        # non-balance
    assert leg[("stochN", "d1")] == repr(10.0), leg        # stoch sum 5+5
    assert leg[("stochN", "dchild")] == repr(5.0), leg     # fall-thru b2

    tier = _assert_parity(df_legacy, df_vec, "synthetic")
    print(f"\n[pdtNodeInflow parity] synthetic branches: Tier {tier}")
    # Tier A expected for the integer-valued synthetic sums.
    if tier != "A":
        pytest.fail(
            "synthetic: unexpected Tier-B demotion on integer-valued "
            "synthetic sums — investigate")
