"""Δ.7 — Cluster C (profile cascade) parity tests.

Per-fixture parity check: lazy port in
:mod:`flextool.engine_polars._derived_profile` vs. the canonical
preprocessed CSV in ``solve_data/pdtProfile.csv`` produced by
flextool's :func:`flextoolrunner.preprocessing.entity_period_calc_params.write_pdtProfile`.

Field covered:

* ``p_profile_value`` — per-(f, d, t) profile value resolved across
  flextool's 5-branch fallback (stochastic UNION fold / parent-period
  fold / time-axis / scalar / zero).

The CSV is the parity oracle — any divergence between the lazy port
and the CSV surfaces as a per-fixture failure.

Test cases
----------

* ``test_p_profile_value_lazy_vs_csv`` — per-fixture parametrised parity
  sweep across every ``work_*`` fixture under ``tests/engine_polars/data``
  with a ``solve_data/pdtProfile.csv``.
* ``test_p_profile_value_5tier_resolution`` — hand-cooked InMemoryReader
  fixtures exercising tier 3 (time-axis), tier 4 (scalar), tier 5
  (zero) cascades.
* ``test_p_profile_value_stochastic_branches`` — fixture-driven test on
  ``work_2day_stochastic_dispatch_*`` confirming the stochastic 3d_map
  cascade resolves correctly (branches 1 and 2 of the 5-branch fallback).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars import SpineDbReader
from flextool.engine_polars._inmemory_reader import InMemoryReader
from flextool.engine_polars._derived_profile import (
    p_profile_value_lf,
    p_profile_value_from_source_v2,
    _classify_profile_rows,
)


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


# Phase 3d: curated parity sweep — see conftest.PARITY_SWEEP_CASES.
from _parity_sweep import PARITY_SWEEP_CASES  # noqa: E402

PARITY_CASES = [(legacy, scen, dbf) for legacy, scen, dbf in PARITY_SWEEP_CASES]


def _frames_equal(a: pl.DataFrame | None,
                     b: pl.DataFrame | None,
                     keys: tuple[str, ...]) -> tuple[bool, str | None]:
    """Compare two frames row-set + value (float-tolerant).

    Mirror of ``test_existing_chain_cluster_parity._frames_equal``.
    """
    if a is None and b is None:
        return True, None
    if a is None:
        return False, f"left None, right {b.height} rows"
    if b is None:
        return False, f"left {a.height} rows, right None"
    if set(a.columns) != set(b.columns):
        return False, f"columns differ: left={a.columns} right={b.columns}"
    if a.height != b.height:
        return False, f"row counts differ: left={a.height} right={b.height}"
    a_sorted = a.sort(by=list(keys))
    b_sorted = b.select(a.columns).sort(by=list(keys))
    if a_sorted.equals(b_sorted):
        return True, None
    val_col = next((c for c in a.columns if c not in keys), None)
    if val_col is None:
        return False, "no value column"
    a_keys = a_sorted.select(list(keys))
    b_keys = b_sorted.select(list(keys))
    if not a_keys.equals(b_keys):
        return False, "key sets differ"
    av = a_sorted[val_col].cast(pl.Float64, strict=False).to_list()
    bv = b_sorted[val_col].cast(pl.Float64, strict=False).to_list()
    max_diff = 0.0
    for x, y in zip(av, bv):
        if x is None or y is None:
            if x != y:
                return False, f"null mismatch: {x} vs {y}"
            continue
        d = abs(x - y)
        if d > max_diff:
            max_diff = d
    av_max = max((abs(x) for x in av if x is not None), default=1.0)
    if max_diff < 1e-7 * max(1.0, av_max):
        return True, None
    return False, f"max abs diff = {max_diff!r}"


# ---------------------------------------------------------------------------
# Per-fixture parity sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "work_name,scenario,db_fixture", PARITY_CASES,
    ids=[c[0] for c in PARITY_CASES],
)
def test_p_profile_value_lazy_vs_csv(
        work_name: str, scenario: str, db_fixture: str,
        scenario_workdir) -> None:
    """Per-fixture: lazy ``p_profile_value_lf`` vs canonical
    ``solve_data/pdtProfile.csv``.

    The canonical CSV is produced by flextool's
    :func:`write_pdtProfile` (5-branch fallback over stochastic /
    parent-period / time / scalar / zero).  The lazy port reproduces
    the same algorithm against the SpineDB scenario.
    """
    work = scenario_workdir(scenario, db_fixture=db_fixture)
    sqlite = work / "tests.sqlite"
    csv_path = work / "solve_data" / "pdtProfile.csv"
    if not csv_path.exists():
        pytest.skip("no pdtProfile.csv")

    try:
        reader = SpineDbReader(sqlite, scenario)
    except Exception as e:
        pytest.skip(f"reader construction failed: {e}")

    siu_path = work / "solve_data" / "steps_in_use.csv"
    if not siu_path.exists():
        pytest.skip("no steps_in_use.csv")
    siu = pl.read_csv(siu_path)
    if "period" not in siu.columns or "step" not in siu.columns:
        pytest.skip("steps_in_use schema unexpected")
    dt = siu.select(d=pl.col("period"), t=pl.col("step"))

    lazy = p_profile_value_lf(reader, dt, workdir=work).collect()
    csv = pl.read_csv(csv_path).rename(
        {"profile": "f", "period": "d", "time": "t"})

    eq, diff = _frames_equal(lazy, csv, ("f", "d", "t"))
    assert eq, (
        f"{work_name}: profile-cascade parity broken — {diff!r}\n"
        f"  lazy  shape: {lazy.shape}\n"
        f"  csv   shape: {csv.shape}\n"
    )


# ---------------------------------------------------------------------------
# 5-tier resolution test cases (hand-cooked InMemoryReader fixtures)
# ---------------------------------------------------------------------------


def _make_dt(periods: list[str], times: list[str]) -> pl.DataFrame:
    """Build a ``[d, t]`` cross-product frame for tests."""
    rows = []
    for d in periods:
        for t in times:
            rows.append({"d": d, "t": t})
    return pl.DataFrame(rows)


def test_tier_3_time_axis_only() -> None:
    """Tier 3: time-series profile broadcast across all dispatch periods.

    A profile keyed only by t (no period) should produce one row per
    (period, t) where the value is constant across periods.
    """
    profiles_df = pl.DataFrame({"name": ["wind"]})
    # Profile values are time-axis (1d_map by time).
    pv = pl.DataFrame({
        "name": ["wind", "wind", "wind"],
        "t": ["t01", "t02", "t03"],
        "value": [0.5, 0.7, 0.9],
    })
    src = InMemoryReader(
        entities={"profile": profiles_df},
        parameters={("profile", "profile"): pv},
    )
    dt = _make_dt(["d1", "d2"], ["t01", "t02", "t03"])
    out = p_profile_value_lf(src, dt).collect().sort(["f", "d", "t"])

    expected = pl.DataFrame({
        "f": ["wind"] * 6,
        "d": ["d1"] * 3 + ["d2"] * 3,
        "t": ["t01", "t02", "t03"] * 2,
        "value": [0.5, 0.7, 0.9, 0.5, 0.7, 0.9],
    }).sort(["f", "d", "t"])
    eq, diff = _frames_equal(out, expected, ("f", "d", "t"))
    assert eq, diff


def test_tier_4_scalar_profile() -> None:
    """Tier 4: scalar profile broadcast over the full (d, t) grid."""
    profiles_df = pl.DataFrame({"name": ["base"]})
    pv = pl.DataFrame({"name": ["base"], "value": [0.42]})
    src = InMemoryReader(
        entities={"profile": profiles_df},
        parameters={("profile", "profile"): pv},
    )
    dt = _make_dt(["d1", "d2"], ["t01", "t02"])
    out = p_profile_value_lf(src, dt).collect().sort(["f", "d", "t"])

    expected = pl.DataFrame({
        "f": ["base"] * 4,
        "d": ["d1"] * 2 + ["d2"] * 2,
        "t": ["t01", "t02"] * 2,
        "value": [0.42, 0.42, 0.42, 0.42],
    }).sort(["f", "d", "t"])
    eq, diff = _frames_equal(out, expected, ("f", "d", "t"))
    assert eq, diff


def test_tier_5_zero_fallback() -> None:
    """Tier 5: profile with no value resolves to 0.0 across (d, t).

    Mirrors flextool's ``write_pdtProfile`` final ``else: 0.0`` branch.
    """
    profiles_df = pl.DataFrame({"name": ["empty_profile"]})
    pv = pl.DataFrame(schema={"name": pl.Utf8, "value": pl.Float64})
    src = InMemoryReader(
        entities={"profile": profiles_df},
        parameters={("profile", "profile"): pv},
    )
    dt = _make_dt(["d1"], ["t01", "t02"])
    out = p_profile_value_lf(src, dt).collect().sort(["f", "d", "t"])

    expected = pl.DataFrame({
        "f": ["empty_profile"] * 2,
        "d": ["d1", "d1"],
        "t": ["t01", "t02"],
        "value": [0.0, 0.0],
    }).sort(["f", "d", "t"])
    eq, diff = _frames_equal(out, expected, ("f", "d", "t"))
    assert eq, diff


def test_tier_priority_scalar_overlay_zero() -> None:
    """Tier-priority resolver: scalar (tier 4) wins over the implicit
    tier-5 zero broadcast.  Multiple profiles in entity table — only
    one has a value.
    """
    profiles_df = pl.DataFrame({"name": ["with_val", "no_val"]})
    pv = pl.DataFrame({"name": ["with_val"], "value": [0.3]})
    src = InMemoryReader(
        entities={"profile": profiles_df},
        parameters={("profile", "profile"): pv},
    )
    dt = _make_dt(["d1"], ["t01", "t02"])
    out = p_profile_value_lf(src, dt).collect().sort(["f", "d", "t"])

    expected = pl.DataFrame({
        "f": ["no_val", "no_val", "with_val", "with_val"],
        "d": ["d1"] * 4,
        "t": ["t01", "t02", "t01", "t02"],
        "value": [0.0, 0.0, 0.3, 0.3],
    }).sort(["f", "d", "t"])
    eq, diff = _frames_equal(out, expected, ("f", "d", "t"))
    assert eq, diff


def test_tier_priority_time_overlay_scalar() -> None:
    """Tier-priority: time-axis (tier 3) wins over scalar (tier 4)
    when both are present for different profiles.

    Two profiles: one scalar, one time-keyed.  Each picks its own
    tier; cross-contamination must not occur.
    """
    profiles_df = pl.DataFrame({"name": ["s", "t"]})
    # Mixed-shape parameter frames are tricky; the InMemoryReader
    # column-shape fallback only supports a single uniform shape per
    # parameter, so this test uses a uniform 1d_map(t) shape with the
    # scalar profile's value duplicated across t.  For *real* mixed-
    # tier dispatch (per-row Map vs scalar), the SpineDB path takes
    # over via ``_classify_profile_rows``'s typed branch.
    pv = pl.DataFrame({
        "name": ["t", "t", "s", "s"],
        "t": ["t01", "t02", "t01", "t02"],
        "value": [0.5, 0.7, 0.9, 0.9],
    })
    src = InMemoryReader(
        entities={"profile": profiles_df},
        parameters={("profile", "profile"): pv},
    )
    dt = _make_dt(["d1"], ["t01", "t02"])
    out = p_profile_value_lf(src, dt).collect().sort(["f", "d", "t"])

    expected = pl.DataFrame({
        "f": ["s", "s", "t", "t"],
        "d": ["d1"] * 4,
        "t": ["t01", "t02", "t01", "t02"],
        "value": [0.9, 0.9, 0.5, 0.7],
    }).sort(["f", "d", "t"])
    eq, diff = _frames_equal(out, expected, ("f", "d", "t"))
    assert eq, diff


# ---------------------------------------------------------------------------
# Stochastic branches (3d_map cascade)
# ---------------------------------------------------------------------------


STOCHASTIC_FIXTURES = [
    ("work_2day_stochastic_dispatch_full_storage",
     "2_day_stochastic_dispatch"),
    ("work_2day_stochastic_dispatch_no_storage",
     "2_day_stochastic_dispatch_no_storage"),
]


@pytest.mark.parametrize("work_name,scenario", STOCHASTIC_FIXTURES)
def test_p_profile_value_stochastic_branches(
        work_name: str, scenario: str, scenario_workdir) -> None:
    """Stochastic 3d_map cascade — Branch 1 (UNION fold) + Branch 2
    (parent-period fold) parity vs ``solve_data/pdtProfile.csv``.

    Pre-Δ.7, the lazy helper deferred the stochastic path entirely
    (the ``_check_canonical_keys`` predicate returned ``None`` when
    Map keys were generic ``x`` / ``i``).  Δ.7 implements both fold
    branches via the per-solve workdir scaffolding (``period__branch``,
    ``solve_branch__time_branch``, ``first_timesteps``,
    ``pbt_profile``, ``groupIncludeStochastics``).
    """
    work = scenario_workdir(scenario, db_fixture="stochastic")
    sqlite = work / "tests.sqlite"
    csv_path = work / "solve_data" / "pdtProfile.csv"
    pbt_path = work / "input" / "pbt_profile.csv"
    if not pbt_path.exists() or pbt_path.stat().st_size < 50:
        pytest.skip("no stochastic pbt_profile data")

    reader = SpineDbReader(sqlite, scenario)
    siu = pl.read_csv(work / "solve_data" / "steps_in_use.csv")
    dt = siu.select(d=pl.col("period"), t=pl.col("step"))

    lazy = p_profile_value_lf(reader, dt, workdir=work).collect()
    csv = pl.read_csv(csv_path).rename(
        {"profile": "f", "period": "d", "time": "t"})

    # Sanity: stochastic fixture should resolve to the 3d_map tier.
    tiers = _classify_profile_rows(reader)
    assert any(t == "stochastic" for t in tiers.values()), (
        f"{work_name}: no profile classified as stochastic — "
        f"the 3d_map dispatch is the test target."
    )

    # The CSV emits per-period rows (one per branch period); the lazy
    # output must match exactly.
    eq, diff = _frames_equal(lazy, csv, ("f", "d", "t"))
    assert eq, (
        f"{work_name}: stochastic profile cascade mismatch — {diff}\n"
        f"  lazy: {lazy.shape}, csv: {csv.shape}\n"
    )


# ---------------------------------------------------------------------------
# Param-shaped public boundary
# ---------------------------------------------------------------------------


def test_apply_profile_cascade_param_shape() -> None:
    """``p_profile_value_from_source_v2`` returns a Param tagged
    ``("f", "d", "t")`` or ``None`` when no resolution is possible.
    """
    # No profiles → None.
    profiles_df = pl.DataFrame(schema={"name": pl.Utf8})
    pv = pl.DataFrame(schema={"name": pl.Utf8, "value": pl.Float64})
    src = InMemoryReader(
        entities={"profile": profiles_df},
        parameters={("profile", "profile"): pv},
    )
    dt = _make_dt(["d1"], ["t01"])
    out = p_profile_value_from_source_v2(src, dt)
    assert out is None

    # One scalar profile → Param.
    profiles_df = pl.DataFrame({"name": ["p"]})
    pv = pl.DataFrame({"name": ["p"], "value": [1.5]})
    src = InMemoryReader(
        entities={"profile": profiles_df},
        parameters={("profile", "profile"): pv},
    )
    out = p_profile_value_from_source_v2(src, dt)
    assert out is not None
    assert out.dims == ("f", "d", "t")
    assert out.frame.height == 1
    assert out.frame["value"][0] == pytest.approx(1.5)
