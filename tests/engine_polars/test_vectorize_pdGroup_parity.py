"""Parity gate for the vectorized ``pdGroup`` derive (period-only).

``pdGroup`` is a 5-branch period-only cascade with a branch-SUM fold
(``pd_group`` direct → branch-sum over ``branches_for_d[d]`` → ``p_group``
→ ``5000`` default set → ``0``).  Because the fold is a sum, multi-term
cells could drift at the last ULP (legacy sequential ``sum`` vs polars
``group_by().sum()``), so the test is **strict-first, Tier-B-fallback**:

* ``fullYear`` (non-stochastic): branch-sum is ≤1 term → strict
  ``.equals`` byte-parity (Tier A).
* ``2_day_stochastic_dispatch``: try strict ``.equals``; if it holds,
  assert it (Tier A); if it fails, the difference must be confined to the
  value column and pass ``rtol≤1e-12`` while the key columns ``.equals``
  (Tier B).

Real fixtures may keep the fold ≤1 term, so a **synthetic** multi-term
fold test (D4) drives the vectorized derive vs the legacy on a hand-built
Provider where a ``(g, param, d)`` has ≥2 branch periods each carrying a
distinct ``pd_group`` value — the genuine branch-sum gate.  A second
synthetic case proves the D3 fix: a DUPLICATED ``group`` entry must NOT
double-count the branch-sum (vec == legacy).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars._emit_period_params import (
    derive_pdGroup,
    derive_pdGroup_vectorized,
)
from flextool.engine_polars._flex_data_provider import FlexDataProvider


def _provider_from_workdir(workdir: Path):
    """Reconstruct a Provider from every CSV in input/ + solve_data/.

    Dual-registers each frame under ``"<parent>/<stem>"`` AND the bare
    ``"<stem>"`` key.
    """
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


def _assert_tier_a_or_b(df_vec, df_legacy, label):
    """Assert strict ``.equals`` (Tier A) or fall to Tier-B tolerance.

    Returns the tier string actually used ("A" or "B").
    """
    if df_vec.equals(df_legacy):
        return "A"
    # Tier B — keys must match exactly; values compared as floats.
    key_cols = ["group", "param", "period"]
    assert df_vec.shape == df_legacy.shape, (
        f"{label}: shape mismatch {df_vec.shape} != {df_legacy.shape}"
    )
    assert df_vec.select(key_cols).equals(df_legacy.select(key_cols)), (
        f"{label}: Tier-B key columns differ (not a pure value-drift "
        f"difference)"
    )
    v_vec = df_vec["value"].cast(pl.Float64).to_list()
    v_leg = df_legacy["value"].cast(pl.Float64).to_list()
    import math
    for a, b in zip(v_vec, v_leg):
        assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (
            f"{label}: Tier-B value drift {a} vs {b} exceeds rtol 1e-12"
        )
    return "B"


_CASES = [
    ("fullYear", "main"),
    ("2_day_stochastic_dispatch", "stochastic"),
]


@pytest.mark.parametrize("scenario,db_fixture", _CASES)
def test_vectorized_pdGroup_matches_legacy(
    scenario, db_fixture, scenario_workdir,
):
    work = scenario_workdir(scenario, db_fixture=db_fixture)
    p = _provider_from_workdir(work)
    inp = work / "input"
    sdd = work / "solve_data"

    df_legacy = derive_pdGroup(inp, sdd, provider=p)
    df_vec = derive_pdGroup_vectorized(inp, sdd, provider=p)

    if scenario == "fullYear":
        # Non-stochastic → branch-sum ≤1 term → byte-exact (Tier A).
        assert df_vec.equals(df_legacy), (
            f"{scenario}: vectorized pdGroup != legacy (Tier-A byte "
            f"parity). legacy {df_legacy.shape}, vec {df_vec.shape}"
        )
    else:
        tier = _assert_tier_a_or_b(df_vec, df_legacy, scenario)
        # Surface which tier the stochastic fixture landed in.
        print(f"\n[pdGroup parity] {scenario} landed in Tier {tier}")


# ---------------------------------------------------------------------------
# Synthetic multi-term branch-sum + D3 dup-safety gate (D4).
# ---------------------------------------------------------------------------

def _synthetic_provider(*, group_rows, pd_group_rows, p_group_rows,
                        branch_rows, period_rows):
    """Build a Provider for the period-only pdGroup derive.

    Keys match ``_provider_key`` for ``input_dir="input"`` /
    ``solve_data_dir="solve_data"``: readers fetch
    ``input/<stem>`` and ``solve_data/<stem>``.

    * group_rows:   list[str]            → input/group.csv (col "group")
    * pd_group_rows:list[(g,param,d,v)]  → input/pd_group.csv
    * p_group_rows: list[(g,param,v)]    → input/p_group.csv
    * branch_rows:  list[(branch_d, d)]  → solve_data/period__branch.csv
    * period_rows:  list[str]            → solve_data/period_in_use_set.csv
    """
    provider = FlexDataProvider()

    g_df = pl.DataFrame({"group": group_rows}, schema={"group": pl.Utf8})
    provider.put("input/group", g_df)

    pd_df = pl.DataFrame(
        {
            "group": [r[0] for r in pd_group_rows],
            "param": [r[1] for r in pd_group_rows],
            "period": [r[2] for r in pd_group_rows],
            "value": [r[3] for r in pd_group_rows],
        },
        schema={
            "group": pl.Utf8, "param": pl.Utf8,
            "period": pl.Utf8, "value": pl.Float64,
        },
    )
    provider.put("input/pd_group", pd_df)

    p_df = pl.DataFrame(
        {
            "group": [r[0] for r in p_group_rows],
            "param": [r[1] for r in p_group_rows],
            "value": [r[2] for r in p_group_rows],
        },
        schema={"group": pl.Utf8, "param": pl.Utf8, "value": pl.Float64},
    )
    provider.put("input/p_group", p_df)

    # period__branch.csv is (branch_period, period) — branch in col 0.
    pb_df = pl.DataFrame(
        {
            "branch_period": [r[0] for r in branch_rows],
            "period": [r[1] for r in branch_rows],
        },
        schema={"branch_period": pl.Utf8, "period": pl.Utf8},
    )
    provider.put("solve_data/period__branch", pb_df)

    piu_df = pl.DataFrame(
        {"period": period_rows}, schema={"period": pl.Utf8})
    provider.put("solve_data/period_in_use_set", piu_df)

    return provider


def test_synthetic_multiterm_branch_sum():
    """A (g, param, d) with ≥2 branch periods each carrying a distinct
    ``pd_group`` value must SUM both (multi-term fold) — vec vs legacy.

    Uses ``max_cumulative_flow`` (a ``_GROUP_PERIOD_PARAM`` member, not in
    the 5000-default set) so the fold path is the active branch.
    """
    param = "max_cumulative_flow"
    # Period "d1" has two branch periods "b1", "b2"; no direct
    # pd_group[(g,param,d1)], so the branch-sum fires with 2 terms.
    provider = _synthetic_provider(
        group_rows=["gA"],
        pd_group_rows=[
            ("gA", param, "b1", 1.25),
            ("gA", param, "b2", 2.5),
        ],
        p_group_rows=[],
        branch_rows=[("b1", "d1"), ("b2", "d1")],
        period_rows=["d1"],
    )
    inp = Path("input")
    sdd = Path("solve_data")

    df_legacy = derive_pdGroup(inp, sdd, provider=provider)
    df_vec = derive_pdGroup_vectorized(inp, sdd, provider=provider)

    # The (gA, max_cumulative_flow, d1) cell must be 1.25 + 2.5 = 3.75.
    cell = df_legacy.filter(
        (pl.col("group") == "gA")
        & (pl.col("param") == param)
        & (pl.col("period") == "d1")
    )
    assert cell.height == 1
    assert float(cell["value"][0]) == pytest.approx(3.75, rel=1e-12)

    tier = _assert_tier_a_or_b(df_vec, df_legacy, "synthetic-multiterm")
    print(f"\n[pdGroup synthetic multiterm] landed in Tier {tier}")


def test_synthetic_duplicate_group_no_double_count():
    """D3 fix: a DUPLICATED ``group`` entry must NOT double-count the
    branch-sum — vec must equal legacy.

    ``group.csv`` lists ``gA`` twice; the domain therefore has each
    ``(gA, param)`` twice, and each duplicate row must carry the SAME
    branch-sum (3.75), not double it.
    """
    param = "max_cumulative_flow"
    provider = _synthetic_provider(
        group_rows=["gA", "gA"],  # duplicated group
        pd_group_rows=[
            ("gA", param, "b1", 1.25),
            ("gA", param, "b2", 2.5),
        ],
        p_group_rows=[],
        branch_rows=[("b1", "d1"), ("b2", "d1")],
        period_rows=["d1"],
    )
    inp = Path("input")
    sdd = Path("solve_data")

    df_legacy = derive_pdGroup(inp, sdd, provider=provider)
    df_vec = derive_pdGroup_vectorized(inp, sdd, provider=provider)

    # Every (gA, param, d1) cell — there are two (one per dup group row) —
    # must be 3.75, NOT 7.5.
    cells = df_legacy.filter(
        (pl.col("group") == "gA")
        & (pl.col("param") == param)
        & (pl.col("period") == "d1")
    )
    assert cells.height == 2
    for v in cells["value"]:
        assert float(v) == pytest.approx(3.75, rel=1e-12)

    tier = _assert_tier_a_or_b(df_vec, df_legacy, "synthetic-dup-group")
    print(f"\n[pdGroup synthetic dup-group] landed in Tier {tier}")
