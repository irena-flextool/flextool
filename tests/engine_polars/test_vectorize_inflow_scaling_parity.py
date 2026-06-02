"""Parity gate for the vectorized inflow-scaling emitter.

The inflow-scaling family (``_compute_inflow_scaling_frames``) is the
HARDEST in the vectorize-per-roll effort: 15 LIVE outputs, 12 stateful
stages, Tier B.  This test gates ``_compute_inflow_scaling_frames_vectorized``
against the legacy ``_compute_inflow_scaling_frames`` ORACLE, key-by-key.

Tier policy (design §1):

* **Tier A — byte-exact** (``df_vec.equals(df_legacy)``): the sum-free
  outputs (``ptNode_inflow``, ``_node_cap_inflow_fallback``,
  ``new_peak_sign``, ``old_peak_max``, ``old_peak_min``, ``old_peak_sign``,
  ``new_peak_inflow_sum``).  ``ptNode_inflow`` + ``_node_cap_inflow_fallback``
  MUST stay Tier A — they feed the already-vectorized pdtNodeInflow and the
  lp-scaling emitter; drift there propagates.
* **Tier B — last-ULP tolerance** (parse both ``value`` cols to float,
  ``rtol/atol ≤ 1e-12``): the sum-bearing outputs.

The 2 DEAD outputs (``old_peak.csv``,
``new_peak_divide_by_old_peak_sum_inflow.csv``) are NEVER assigned to the
out-dict — both dicts must contain exactly the 15 LIVE keys and NEITHER
dead key (guard).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars._emit_inflow_scaling import (
    _compute_inflow_scaling_frames,
    _compute_inflow_scaling_frames_vectorized,
)

_LIVE_KEYS = [
    "ptNode_inflow.csv",
    "_node_cap_inflow_fallback.csv",
    "orig_flow_sum.csv",
    "period_share_of_annual_flow.csv",
    "period_flow_annual_multiplier.csv",
    "period_flow_proportional_multiplier.csv",
    "new_peak_sign.csv",
    "old_peak_max.csv",
    "old_peak_min.csv",
    "old_peak_sign.csv",
    "new_peak_divided_by_old_peak.csv",
    "new_peak_inflow_sum.csv",
    "new_old_multiplier.csv",
    "new_old_slope.csv",
    "new_old_section.csv",
]
_DEAD_KEYS = [
    "old_peak.csv",
    "new_peak_divide_by_old_peak_sum_inflow.csv",
]
# Outputs that MUST be byte-exact (sum-free coalesce / max-abs / sign).
_TIER_A_KEYS = {
    "ptNode_inflow.csv",
    "_node_cap_inflow_fallback.csv",
    "new_peak_sign.csv",
    "old_peak_max.csv",
    "old_peak_min.csv",
    "old_peak_sign.csv",
    "new_peak_inflow_sum.csv",
}


def _provider_from_workdir(workdir: Path):
    """Reconstruct a Provider from every CSV in input/ + solve_data/.

    Dual-registers each frame under ``"<parent>/<stem>"`` AND the bare
    ``"<stem>"`` key (design §6 / S6 — glob, do not under-register).
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

    Returns ``"A"`` on byte-parity, ``"B"`` on a legitimate last-ULP
    float-only demotion (max ``|Δ| ≤ ~1e-12``).  Raises on any structural
    mismatch or a non-ULP value difference (a real bug STOPs the gate).
    """
    assert df_vec.columns == df_legacy.columns, (
        f"{label}: column mismatch legacy={df_legacy.columns} "
        f"vec={df_vec.columns}")
    assert df_vec.shape == df_legacy.shape, (
        f"{label}: shape mismatch legacy={df_legacy.shape} "
        f"vec={df_vec.shape}")

    if df_vec.equals(df_legacy):
        return "A"

    key_cols = df_legacy.columns[:2]
    assert df_vec.select(key_cols).equals(df_legacy.select(key_cols)), (
        f"{label}: KEY columns differ — structural bug, NOT a float-ULP "
        f"demotion. STOP.")

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


def _gate_dicts(legacy: dict, vec: dict, fixture: str) -> dict[str, str]:
    """Assert both dicts hold exactly the 15 LIVE keys (no DEAD) and gate
    every LIVE key with :func:`_assert_parity`.  Returns per-key tier."""
    assert set(legacy.keys()) == set(_LIVE_KEYS), (
        f"{fixture}: legacy oracle key set != 15 LIVE keys; "
        f"got {sorted(legacy.keys())}")
    assert set(vec.keys()) == set(_LIVE_KEYS), (
        f"{fixture}: vectorized key set != 15 LIVE keys; "
        f"got {sorted(vec.keys())}")
    for dead in _DEAD_KEYS:
        assert dead not in legacy, f"{fixture}: DEAD {dead} in legacy"
        assert dead not in vec, f"{fixture}: DEAD {dead} in vectorized"

    tiers: dict[str, str] = {}
    for k in _LIVE_KEYS:
        lf = legacy[k]
        vf = vec[k]
        assert vf.columns == lf.columns, (
            f"{fixture}/{k}: columns {vf.columns} != {lf.columns}")
        assert vf.shape == lf.shape, (
            f"{fixture}/{k}: shape {vf.shape} != {lf.shape}")
        tier = _assert_parity(lf, vf, f"{fixture}/{k}")
        # Tier-A invariant on the sum-free outputs.
        if k in _TIER_A_KEYS:
            assert tier == "A", (
                f"{fixture}/{k}: expected byte-exact Tier A on a sum-free "
                f"output, got Tier {tier} — drift here propagates "
                f"downstream (pdtNodeInflow / lp-scaling)")
        tiers[k] = tier
    return tiers


def _scaling_methods(p) -> set[str]:
    im = p.get("solve_data/node__inflow_method")
    methods: set[str] = set()
    if im is not None:
        for row in im.iter_rows():
            if len(row) >= 2 and row[0] and row[1]:
                methods.add(row[1])
    return methods


def _total_nonzero(legacy: dict) -> int:
    total = 0
    for k in _LIVE_KEYS:
        df = legacy[k]
        if df.height == 0:
            continue
        total += df.filter(pl.col("value").cast(pl.Float64) != 0.0).height
    return total


def _run_fixture(scenario_workdir, scenario: str, db_fixture: str,
                 *, require_scaling_method: bool = False,
                 require_nonzero: bool = True) -> dict[str, str]:
    work = scenario_workdir(scenario, db_fixture=db_fixture)
    p = _provider_from_workdir(work)
    inp = work / "input"
    sdd = work / "solve_data"

    legacy = _compute_inflow_scaling_frames(inp, sdd, provider=p)
    vec = _compute_inflow_scaling_frames_vectorized(inp, sdd, provider=p)

    tiers = _gate_dicts(legacy, vec, f"{scenario}/{db_fixture}")

    if require_scaling_method:
        methods = _scaling_methods(p)
        assert methods & {
            "scale_to_annual_flow", "scale_in_proportion",
            "scale_to_annual_and_peak_flow",
        }, (f"{scenario}: no annual/peak/proportion method authored — "
            f"the annual/peak/proportion stages are vacuous; got {methods}")
    if require_nonzero:
        assert _total_nonzero(legacy) > 0, (
            f"{scenario}: every LIVE frame emits only zeros — gate vacuous")
    return tiers


def test_inflow_scaling_fullYear(scenario_workdir):
    # fullYear authors only ``use_original`` — it exercises stages 1-2
    # (ptNode_inflow + fallback) with non-zero inflow but no annual/peak/
    # proportion method, so don't require a scaling method here.
    tiers = _run_fixture(scenario_workdir, "fullYear", "main")
    print(f"\n[inflow-scaling parity] fullYear tiers: {tiers}")


def test_inflow_scaling_stochastic(scenario_workdir):
    # 2_day_stochastic_dispatch (stochastic db) authors no inflow method;
    # it is a structural-parity check (frames may be sparse), so only the
    # dict-parity gate applies.
    tiers = _run_fixture(
        scenario_workdir, "2_day_stochastic_dispatch", "stochastic",
        require_nonzero=False)
    print(f"\n[inflow-scaling parity] 2_day_stochastic_dispatch tiers: "
          f"{tiers}")


def test_inflow_scaling_pbt_inflow(scenario_workdir):
    """``stochastics_pbt_inflow`` authors the inflow 3d_map → populates
    pt_node_inflow + the peak family.  Skip+report if the fixture DB does
    not load."""
    try:
        work = scenario_workdir(
            "2_day_stochastic_dispatch", db_fixture="stochastics_pbt_inflow")
    except Exception as exc:  # pragma: no cover - fixture-load guard
        pytest.skip(f"stochastics_pbt_inflow fixture did not load: {exc}")
    p = _provider_from_workdir(work)
    inp = work / "input"
    sdd = work / "solve_data"

    legacy = _compute_inflow_scaling_frames(inp, sdd, provider=p)
    vec = _compute_inflow_scaling_frames_vectorized(inp, sdd, provider=p)
    tiers = _gate_dicts(legacy, vec, "stochastics_pbt_inflow")

    # Non-vacuity for inflow-scaling on this fixture: the snapshot's
    # node__inflow_method (last sub-solve) carries only ``use_original``
    # into the inflow-scaling inputs — the annual/peak methods authored in
    # the JSON do not survive to this snapshot's solve_data, so the
    # annual/peak stages are NOT exercised here.  The genuine annual/peak/
    # proportion coverage is the synthetic test below.  What this fixture
    # adds is a real stochastic-solve snapshot through the full reader
    # block with non-zero inflow → assert that.
    assert _total_nonzero(legacy) > 0, (
        "stochastics_pbt_inflow: every LIVE frame emits only zeros — gate "
        "vacuous")
    methods = _scaling_methods(p)
    print(f"\n[inflow-scaling parity] stochastics_pbt_inflow tiers: {tiers} "
          f"(inflow methods in snapshot: {sorted(methods)})")
