"""Parity gate for the vectorized ``pdtProfile`` derive.

``pdtProfile`` is the single-entity-key (``profile``, NO param axis)
inline 5-branch cascade: (1) stochastic fold, (2) parent-period fold,
(3) ``pt_profile[(profile, time)]``, (4) ``p_profile[profile]``, (5)
literal ``0.0``.  The ``pbt_profile`` key is a 4-tuple
``(profile, tb, ts, t)`` (no param), so the fold runs on the single
``["profile"]`` entity-key column.

UNLIKE every other fold-bearing family converted so far, the
``pbt_profile`` 3d_map IS authored on a real fixture: the
``2_day_stochastic_dispatch`` (``stochastic``) fixture populates
``pbt_profile.csv`` (a stochastic profile authored as a 3d_map), so the
fold GENUINELY FIRES on real data (critique D1).  The test asserts that
non-vacuously (``pbt_profile`` non-empty ≥1 row) and detects the tier
from the matched stoch cells' fold term-count (critique D2).

Tier policy (design §1): assert strict ``df_vec.equals(df_legacy)``
(Tier A) first.  A multi-term fold's ``group_by().sum()`` pairwise
reduction can drift from the legacy sequential ``total += v`` by last-ULP
(``|Δ| ≤ ~1e-12``); that legitimately demotes the offending cells to
**Tier B** (parse both ``value`` columns to float, key cols ``.equals``,
values ``rtol ≤ 1e-12``).  Any NON-ULP difference is a real bug — those
are NOT tolerated; the test fails hard.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from flextool.engine_polars._emit_period_params import (
    derive_pdtProfile,
    derive_pdtProfile_vectorized,
)

_KEY_COLS = ["profile", "period", "time"]


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


# --- helpers to reconstruct the fold inputs from a Provider ----------------

def _pbt_profile_from_provider(p) -> dict:
    """Rebuild the ``pbt_profile`` dict (verbatim legacy reader logic)."""
    from flextool.engine_polars._emit_period_params import _cell_str

    out: dict = {}
    df = p.get("input/pbt_profile")
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


def _fold_index_from_provider(p, stem: str, key_col: int) -> dict:
    """Rebuild a ``_read_pairs_to_dict`` index (verbatim legacy logic)."""
    from flextool.engine_polars._emit_period_params import _cell_str

    out: dict = {}
    df = p.get(f"solve_data/{stem}")
    if df is None:
        return out
    other_col = 1 - key_col
    for row in df.iter_rows():
        if len(row) < 2:
            continue
        c0, c1 = _cell_str(row[0]), _cell_str(row[1])
        if c0 and c1:
            cells = (c0, c1)
            out.setdefault(cells[key_col], []).append(cells[other_col])
    return out


# --- Part 1: fullYear (non-stochastic) — fold inert, strict Tier A ---------

def test_vectorized_pdtProfile_fullYear(scenario_workdir):
    work = scenario_workdir("fullYear", db_fixture="main")
    p = _provider_from_workdir(work)
    inp = work / "input"
    sdd = work / "solve_data"

    df_legacy = derive_pdtProfile(inp, sdd, provider=p)
    df_vec = derive_pdtProfile_vectorized(inp, sdd, provider=p)

    # fullYear authors no pbt_profile 3d_map ⇒ the fold is inert here.
    pbt = _pbt_profile_from_provider(p)
    assert not pbt, (
        "fullYear unexpectedly authors pbt_profile — the fold can fire; "
        "re-check the tier detection on this fixture")

    tier = _assert_parity(df_legacy, df_vec, "fullYear")
    assert tier == "A", (
        "fullYear: unexpected Tier-B demotion with no profile fold "
        "(empty pbt_profile) — investigate")


# --- Part 2: 2_day_stochastic_dispatch — THE FOLD FIRES HERE (critique D1) -

def test_vectorized_pdtProfile_stochastic(scenario_workdir):
    work = scenario_workdir(
        "2_day_stochastic_dispatch", db_fixture="stochastic")
    p = _provider_from_workdir(work)
    inp = work / "input"
    sdd = work / "solve_data"

    df_legacy = derive_pdtProfile(inp, sdd, provider=p)
    df_vec = derive_pdtProfile_vectorized(inp, sdd, provider=p)

    assert df_legacy.height > 0, (
        "2_day_stochastic_dispatch: legacy pdtProfile produced 0 rows — "
        "gate vacuous")

    # Non-vacuous fold: pbt_profile IS authored on this stochastic fixture
    # (critique D1) — the fold genuinely fires.  Assert it is non-empty.
    pbt = _pbt_profile_from_provider(p)
    assert pbt, (
        "2_day_stochastic_dispatch: pbt_profile is EMPTY — the fold is "
        "NOT exercised on this fixture; the gate would be vacuous.  A "
        "fixture change has removed the pbt_profile 3d_map; restore "
        "fold coverage")

    # Detect the tier (critique D2): over the matched stoch cells, the
    # max fold term-count = max(len(tb_for_d[d]) * len(ts_for_d[d])).
    ts_for_d = _fold_index_from_provider(p, "first_timesteps", 0)
    tb_for_d = _fold_index_from_provider(
        p, "solve_branch__time_branch", 0)
    periods = {r[0] for r in df_legacy.select("period").iter_rows()}
    max_terms = 0
    for d in periods:
        terms = len(tb_for_d.get(d, ())) * len(ts_for_d.get(d, ()))
        if terms > max_terms:
            max_terms = terms

    tier = _assert_parity(
        df_legacy, df_vec, "2_day_stochastic_dispatch")

    if max_terms <= 1:
        # Single-term fold ⇒ no pairwise-sum drift ⇒ Tier A must hold.
        assert tier == "A", (
            "2_day_stochastic_dispatch: Tier-B demotion on a single-term "
            f"fold (max_terms={max_terms}) — unexpected, investigate")
    # If max_terms >= 2 the fold is genuinely multi-term and a Tier-B
    # ULP demotion is legitimate; _assert_parity already enforced the
    # 1e-12 ceiling.
    print(f"\n[pdtProfile parity] 2_day_stochastic_dispatch: Tier {tier} "
          f"(fold fires, pbt_profile rows={len(pbt)}, "
          f"max fold term-count={max_terms})")


# --- Part 3: synthetic fold-coverage (the genuine multi-term gate) ---------

def test_vectorized_pdtProfile_synthetic_fold_and_cascade(tmp_path):
    """Drive the vectorized derive vs the legacy scalar cascade on a
    synthetic fixture exercising EVERY differentiating branch of
    ``pdtProfile`` plus the process+node stochastic UNION:

    * **Stoch branch-sum (multi-term):** ``solarA`` (stoch via a
      stochastic PROCESS) under ``d1`` sums two DISTINCT branches
      ``b1``+``b2`` of its ``pbt_profile`` → 1+2 = 3.0 (multi-term ⇒
      genuine tier check).
    * **Stoch fall-through + multi-parent (S2/§12.6):** ``solarA`` is
      stochastic but has no branch under ``dchild`` (branch-1 miss) →
      falls through to the branch-2 parent fold; its ``b1`` value is
      reached via BOTH parents ``pA``/``pB`` → summed once per parent
      → 1+1 = 2.0.
    * **Multi-parent multiplicity (non-stoch, S2):** ``windB`` (non-stoch)
      has a single ``(b1, s1, t1)`` pbt value shared by two parents
      ``pA``/``pB`` of ``dchild`` → summed ONCE PER PARENT = 20.0; under
      ``d1`` (no parent mapping) it folds to nothing → falls to 0.0.
    * **pt branch (3):** ``loadC`` has a ``pt_profile`` value → 7.0.
    * **p branch (4):** ``loadD`` has only a scalar ``p_profile`` → 0.5.
    * **literal 0.0 (5):** ``loadE`` resolves nowhere → 0.0.
    * **process+node UNION:** ``solarA`` reached via a stochastic
      PROCESS (``process__profile``), ``hydroN`` via a stochastic NODE
      (``node__profile``), ``coGenP`` via a stochastic process_node
      (``process__node__profile``); all three must be in ``stoch_profile``
      so their pbt rows fold in.  ``hydroN``/``coGenP`` also fall through
      to the multi-parent branch-2 fold under ``dchild`` (4→8, 6→12),
      exercising the stoch fall-through on the node/process_node UNION
      members too.

    The legacy ``derive_pdtProfile`` (scalar cascade) is the oracle;
    ``derive_pdtProfile_vectorized`` must match (Tier A expected — small
    integer-valued sums; Tier B acceptable on genuine ULP drift).
    """
    from flextool.engine_polars._flex_data_provider import FlexDataProvider

    provider = FlexDataProvider()

    def put(parent: str, stem: str, df: pl.DataFrame) -> None:
        provider.put(f"{parent}/{stem}", df)
        provider.put(stem, df)

    # pbt_profile: (profile, branch, time_start, time, value) — 4-tuple key
    # (profile, tb, ts, t).  solarA: TWO DISTINCT branches b1=1, b2=2
    # under d1 (stoch sum → 3); b1 reached via parents pA/pB under dchild
    # (fall-through, multi-parent → 1+1 = 2).  windB: b1=10 shared by
    # pA/pB.  hydroN / coGenP each one branch (stoch via node /
    # process_node UNION).
    put("input", "pbt_profile", pl.DataFrame({
        "profile": ["solarA", "solarA", "windB", "hydroN", "coGenP"],
        "branch": ["b1", "b2", "b1", "b1", "b1"],
        "time_start": ["s1", "s1", "s1", "s1", "s1"],
        "time": ["t1", "t1", "t1", "t1", "t1"],
        "value": [1.0, 2.0, 10.0, 4.0, 6.0],
    }))
    # pt_profile: (profile, time, value).
    put("solve_data", "pt_profile", pl.DataFrame({
        "profile": ["loadC"],
        "time": ["t1"],
        "value": [7.0],
    }))
    # p_profile: (profile, value).
    put("input", "p_profile", pl.DataFrame({
        "profile": ["loadD"],
        "value": [0.5],
    }))
    # profile.csv domain — entity-major order preserved.
    put("input", "profile", pl.DataFrame({
        "profile": ["solarA", "windB", "hydroN", "coGenP",
                    "loadC", "loadD", "loadE"],
    }))
    # first_timesteps: (period, time_start) → ts_for_d.
    put("solve_data", "first_timesteps", pl.DataFrame({
        "period": ["d1", "dchild"], "time_start": ["s1", "s1"]}))
    # solve_branch__time_branch: (period, branch) → tb_for_d. d1 has 2
    # branches; parents pA/pB each have b1; dchild has NONE (forces the
    # stoch fall-through for solarA).
    put("solve_data", "solve_branch__time_branch", pl.DataFrame({
        "period": ["d1", "d1", "pA", "pB"],
        "branch": ["b1", "b2", "b1", "b1"]}))
    # period__branch: read with key_col=1 → pe_for_d[child] = [parent].
    put("solve_data", "period__branch", pl.DataFrame({
        "parent": ["pA", "pB"], "child": ["dchild", "dchild"]}))
    # Stochastic groups: g_proc holds the stochastic process, g_node the
    # stochastic node.
    put("input", "group__process", pl.DataFrame({
        "group": ["g_proc"], "process": ["procX"]}))
    put("input", "group__node", pl.DataFrame({
        "group": ["g_node"], "node": ["nodeY"]}))
    put("input", "groupIncludeStochastics", pl.DataFrame({
        "group": ["g_proc", "g_node"]}))
    # process__profile__profile_method: procX (stoch) refs solarA → UNION.
    put("input", "process__profile__profile_method", pl.DataFrame({
        "process": ["procX"],
        "profile": ["solarA"],
        "profile_method": ["upper_limit"]}))
    # node__profile__profile_method: nodeY (stoch) refs hydroN → UNION.
    put("input", "node__profile__profile_method", pl.DataFrame({
        "node": ["nodeY"],
        "profile": ["hydroN"],
        "profile_method": ["upper_limit"]}))
    # process__node__profile__profile_method: procX (stoch) col0, profile
    # in col2 → coGenP → UNION.
    put("input", "process__node__profile__profile_method", pl.DataFrame({
        "process": ["procX"],
        "node": ["nodeZ"],
        "profile": ["coGenP"],
        "profile_method": ["upper_limit"]}))
    put("solve_data", "steps_in_use", pl.DataFrame({
        "period": ["d1", "dchild"], "time": ["t1", "t1"]}))

    inp = tmp_path / "input"
    sdd = tmp_path / "solve_data"

    df_legacy = derive_pdtProfile(inp, sdd, provider=provider)
    df_vec = derive_pdtProfile_vectorized(inp, sdd, provider=provider)

    # Oracle sanity: every differentiating branch landed where expected.
    # (profile, period) -> value-string (time is constant "t1").
    leg = {(r[0], r[1]): r[3] for r in df_legacy.iter_rows()}
    assert leg[("solarA", "d1")] == repr(3.0), leg       # stoch sum (1+2)
    assert leg[("solarA", "dchild")] == repr(2.0), leg   # fall-thru+2 par
    assert leg[("windB", "d1")] == repr(0.0), leg        # no parent → 0.0
    assert leg[("windB", "dchild")] == repr(20.0), leg   # multi-parent
    assert leg[("hydroN", "d1")] == repr(4.0), leg       # stoch via node
    assert leg[("hydroN", "dchild")] == repr(8.0), leg   # node fall-thru
    assert leg[("coGenP", "d1")] == repr(6.0), leg       # stoch via p_n_p
    assert leg[("coGenP", "dchild")] == repr(12.0), leg  # p_n_p fall-thru
    assert leg[("loadC", "d1")] == repr(7.0), leg        # pt branch
    assert leg[("loadD", "d1")] == repr(0.5), leg        # p branch
    assert leg[("loadE", "d1")] == repr(0.0), leg        # literal 0.0

    tier = _assert_parity(df_legacy, df_vec, "synthetic")
    print(f"\n[pdtProfile parity] synthetic fold+UNION: Tier {tier}")
