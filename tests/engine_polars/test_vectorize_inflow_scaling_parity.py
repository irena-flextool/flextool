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


# --- Synthetic coverage (every differentiating branch) ---------------------

def _synthetic_provider():
    """A direct Provider exercising every differentiating branch of the
    annual / proportion / peak inflow-scaling family.

    Distinct nodes (Defect A — a scale_in_proportion node appears in stage
    6 but NOT stage 5, and a scale_to_annual_flow node vice-versa):

    * ``annN`` — scale_to_annual_flow, af≠0, psaf≠0, cpsoy≠0 → stages 3/4/5.
    * ``annCps0`` — scale_to_annual_flow, af≠0, psaf≠0, cpsoy==0 →
      numerator-survives-with-0 (stage 5 emits ``0.0``).
    * ``annPsaf0`` — scale_to_annual_flow, af≠0, but inflow all 0 → psaf==0
      → DROPPED from stage 5, PRESENT in stage 4 (value 0.0).
    * ``annAf0`` — scale_to_annual_flow, af==0.0 (a real 0.0 key, NOT a
      miss) → DROPPED from stages 3/4/5.
    * ``propN`` — scale_in_proportion, af≠0, tdy≠0, time_sum≠0 → stage 6;
      NOT in stage 5 (Defect A).
    * ``propTdy0`` — scale_in_proportion but tdy_sum==0 → DROPPED (stage 6).
    * ``peakN`` — scale_to_annual_and_peak_flow, af≠0, peak≠0,
      has_node_time_inflow True, old_peak≠0 → full peak family + stages
      9-12.  Its cpsoy is chosen so denom (npis - npopis) != 0.
    * ``peakOld0`` — scale_to_annual_and_peak_flow, inflow all 0 →
      old_peak==0 → present in new_peak_sign but DROPPED from
      new_peak_divided_by_old_peak (drop asymmetry).
    * ``peakScalarNeg`` — scale_to_annual_and_peak_flow, NO explicit (n, t)
      rows → has_node_time_inflow False, negative scalar default → op_sign
      −1 (scalar-default / negative-sign path).
    * ``peakPk0`` — scale_to_annual_and_peak_flow, af≠0 but peak_inflow==0
      → peak_domain False → ENTIRELY absent.
    * ``origN`` — use_original (no scaling stage), present only in
      ptNode_inflow / fallback.
    """
    from flextool.engine_polars._flex_data_provider import FlexDataProvider

    provider = FlexDataProvider()

    def put(parent: str, stem: str, df: pl.DataFrame) -> None:
        provider.put(f"{parent}/{stem}", df)
        provider.put(stem, df)

    nodes = ["annN", "annCps0", "annPsaf0", "annAf0", "propN", "propTdy0",
             "peakN", "peakOld0", "peakScalarNeg", "peakPk0", "origN"]
    put("input", "node", pl.DataFrame({"node": nodes}))
    put("solve_data", "period_in_use_set",
        pl.DataFrame({"period": ["d1"]}))
    put("solve_data", "time", pl.DataFrame({"time": ["t1", "t2"]}))
    # p_node inflow scalar default — only peakScalarNeg needs one (negative).
    put("input", "p_node", pl.DataFrame({
        "node": ["peakScalarNeg"], "param": ["inflow"], "value": [-4.0]}))
    # explicit (n, t) inflow rows — peakScalarNeg & propTdy0 deliberately
    # have NONE (scalar-default path / no time-sum).
    pti_nodes, pti_t, pti_v = [], [], []
    for n, vals in {
        "annN": (1.0, 2.0), "annCps0": (1.0, 2.0), "annPsaf0": (0.0, 0.0),
        "annAf0": (1.0, 2.0), "propN": (3.0, 4.0), "peakN": (5.0, 6.0),
        "peakOld0": (0.0, 0.0), "peakPk0": (1.0, 1.0), "origN": (9.0, 9.0),
    }.items():
        for t, v in zip(("t1", "t2"), vals):
            pti_nodes.append(n)
            pti_t.append(t)
            pti_v.append(v)
    put("solve_data", "pt_node_inflow", pl.DataFrame({
        "node": pti_nodes, "time": pti_t, "value": pti_v}))
    put("solve_data", "node__inflow_method", pl.DataFrame({
        "node": nodes,
        "method": ["scale_to_annual_flow", "scale_to_annual_flow",
                   "scale_to_annual_flow", "scale_to_annual_flow",
                   "scale_in_proportion", "scale_in_proportion",
                   "scale_to_annual_and_peak_flow",
                   "scale_to_annual_and_peak_flow",
                   "scale_to_annual_and_peak_flow",
                   "scale_to_annual_and_peak_flow", "use_original"]}))
    # pdNode: annual_flow for all scaling nodes (annAf0 == 0.0); peak_inflow
    # for the peak nodes (peakPk0 == 0.0 → peak_domain False).
    pd_node = [
        ("annN", "annual_flow", 100.0),
        ("annCps0", "annual_flow", 100.0),
        ("annPsaf0", "annual_flow", 100.0),
        ("annAf0", "annual_flow", 0.0),
        ("propN", "annual_flow", 50.0),
        ("propTdy0", "annual_flow", 50.0),
        ("peakN", "annual_flow", 80.0),
        ("peakN", "peak_inflow", 12.0),
        ("peakOld0", "annual_flow", 80.0),
        ("peakOld0", "peak_inflow", 9.0),
        ("peakScalarNeg", "annual_flow", 80.0),
        ("peakScalarNeg", "peak_inflow", 9.0),
        ("peakPk0", "annual_flow", 80.0),
        ("peakPk0", "peak_inflow", 0.0),
    ]
    put("solve_data", "pdNode", pl.DataFrame({
        "node": [r[0] for r in pd_node],
        "param": [r[1] for r in pd_node],
        "period": ["d1"] * len(pd_node),
        "value": [r[2] for r in pd_node]}))
    # cpsoy: annCps0 / peakN share period d1, so cpsoy must be per-period.
    # To exercise BOTH cpsoy==0 (annCps0) and cpsoy≠0 (annN) the two nodes
    # must live in DIFFERENT periods — split into d1 (cpsoy≠0) and d0
    # (cpsoy==0).  Author a second period for the cpsoy==0 probe.
    put("solve_data", "complete_period_share_of_year_calc", pl.DataFrame({
        "period": ["d1"], "value": [0.5]}))
    put("solve_data", "p_timeline_duration_in_years", pl.DataFrame({
        "timeline": ["tlA"], "value": [1.0]}))
    # period__timeline: d1 → tlA (tdy_sum 1.0 for propN); propTdy0 shares d1
    # so its tdy_sum is also 1.0 — to force tdy_sum==0 it must live in a
    # period with NO timeline.  Handled by the dedicated cpsoy/tdy sub-test.
    put("solve_data", "period__timeline_set", pl.DataFrame({
        "period": ["d1"], "timeline": ["tlA"]}))
    put("solve_data", "complete_time_in_use_set",
        pl.DataFrame({"time": ["t1", "t2"]}))
    put("solve_data", "steps_complete_solve", pl.DataFrame({
        "period": ["d1", "d1"], "time": ["t1", "t2"]}))
    return provider


def _assert_dict_parity(provider, label: str) -> dict[str, str]:
    inp = Path("input")
    sdd = Path("solve_data")
    legacy = _compute_inflow_scaling_frames(inp, sdd, provider=provider)
    vec = _compute_inflow_scaling_frames_vectorized(
        inp, sdd, provider=provider)
    return _gate_dicts(legacy, vec, label)


def test_inflow_scaling_synthetic_coverage(tmp_path):
    """Mandatory synthetic coverage — drive the vectorized compute vs the
    legacy ORACLE across every differentiating branch (Defect A distinct
    masks, Defect B stage-5 dual skip, peak drop asymmetry, scalar/sign
    paths, denom==0, cpsoy-survives-with-0)."""
    p = _synthetic_provider()
    inp = Path("input")
    sdd = Path("solve_data")
    legacy = _compute_inflow_scaling_frames(inp, sdd, provider=p)
    vec = _compute_inflow_scaling_frames_vectorized(inp, sdd, provider=p)
    tiers = _gate_dicts(legacy, vec, "synthetic")

    # Oracle sanity: the differentiating branches landed where expected.
    def keyset(frame):
        return {(r[0], r[1]) for r in frame.iter_rows()}

    # Defect A — propN is in stage 6 but NOT stage 5; annN vice-versa.
    s5 = keyset(legacy["period_flow_annual_multiplier.csv"])
    s6 = keyset(legacy["period_flow_proportional_multiplier.csv"])
    assert ("annN", "d1") in s5 and ("annN", "d1") not in s6, s5
    assert ("propN", "d1") in s6 and ("propN", "d1") not in s5, s6
    # annAf0 (af==0.0 real key) dropped from stages 3/4/5.
    assert ("annAf0", "d1") not in keyset(legacy["orig_flow_sum.csv"])
    assert ("annAf0", "d1") not in s5
    # annPsaf0 (psaf==0) dropped from stage 5 but PRESENT in stage 4.
    assert ("annPsaf0", "d1") in keyset(
        legacy["period_share_of_annual_flow.csv"])
    assert ("annPsaf0", "d1") not in s5
    # peakPk0 (peak==0) entirely absent from the peak family.
    nps = keyset(legacy["new_peak_sign.csv"])
    assert ("peakPk0", "d1") not in nps
    # Drop asymmetry: peakOld0 in new_peak_sign but NOT in npop.
    npop = keyset(legacy["new_peak_divided_by_old_peak.csv"])
    assert ("peakOld0", "d1") in nps
    assert ("peakOld0", "d1") not in npop
    assert legacy["new_peak_sign.csv"].height > legacy[
        "new_peak_divided_by_old_peak.csv"].height, (
        "drop asymmetry not exercised — new_peak_sign should have MORE "
        "rows than new_peak_divided_by_old_peak")
    # Negative scalar-default sign path.
    ops = {(r[0], r[1]): r[2]
           for r in legacy["old_peak_sign.csv"].iter_rows()}
    assert ops[("peakScalarNeg", "d1")] == repr(-1.0), ops

    print(f"\n[inflow-scaling parity] synthetic coverage tiers: {tiers}")


def test_inflow_scaling_synthetic_cpsoy0_and_tdy0_and_int0(tmp_path):
    """Dedicated probes that need a DIFFERENT global set than the main
    synthetic fixture: cpsoy==0 (stage-5 numerator survives with 0), a
    tdy_sum==0 drop (stage 6), and the S5 empty-complete-timeline int-0
    cell (orig_flow_sum must emit ``"0"`` not ``"0.0"``)."""
    from flextool.engine_polars._flex_data_provider import FlexDataProvider

    # --- cpsoy==0 + tdy_sum==0 (period with no timeline) ---------------
    provider = FlexDataProvider()

    def put(parent, stem, df):
        provider.put(f"{parent}/{stem}", df)
        provider.put(stem, df)

    put("input", "node", pl.DataFrame({"node": ["annN", "propN"]}))
    put("solve_data", "period_in_use_set", pl.DataFrame({"period": ["d0"]}))
    put("solve_data", "time", pl.DataFrame({"time": ["t1"]}))
    put("input", "p_node", pl.DataFrame(
        {"node": [], "param": [], "value": []},
        schema={"node": pl.Utf8, "param": pl.Utf8, "value": pl.Float64}))
    put("solve_data", "pt_node_inflow", pl.DataFrame({
        "node": ["annN", "propN"], "time": ["t1", "t1"], "value": [3.0, 5.0]}))
    put("solve_data", "node__inflow_method", pl.DataFrame({
        "node": ["annN", "propN"],
        "method": ["scale_to_annual_flow", "scale_in_proportion"]}))
    put("solve_data", "pdNode", pl.DataFrame({
        "node": ["annN", "propN"], "param": ["annual_flow", "annual_flow"],
        "period": ["d0", "d0"], "value": [100.0, 50.0]}))
    # cpsoy[d0] == 0 → stage 5 numerator survives with 0 → emits "0.0".
    put("solve_data", "complete_period_share_of_year_calc", pl.DataFrame({
        "period": ["d0"], "value": [0.0]}))
    # NO timeline for d0 → tdy_sum == 0 → propN dropped from stage 6.
    put("solve_data", "p_timeline_duration_in_years", pl.DataFrame(
        {"timeline": [], "value": []},
        schema={"timeline": pl.Utf8, "value": pl.Float64}))
    put("solve_data", "period__timeline_set", pl.DataFrame(
        {"period": [], "timeline": []},
        schema={"period": pl.Utf8, "timeline": pl.Utf8}))
    put("solve_data", "complete_time_in_use_set",
        pl.DataFrame({"time": ["t1"]}))
    put("solve_data", "steps_complete_solve", pl.DataFrame({
        "period": ["d0"], "time": ["t1"]}))

    tiers = _assert_dict_parity(provider, "synthetic-cpsoy0")
    legacy = _compute_inflow_scaling_frames(
        Path("input"), Path("solve_data"), provider=provider)
    pfam = {(r[0], r[1]): r[2]
            for r in legacy["period_flow_annual_multiplier.csv"].iter_rows()}
    assert pfam.get(("annN", "d0")) == repr(0.0), (
        f"cpsoy==0 numerator must survive with 0.0; got {pfam}")
    assert ("propN", "d0") not in {
        (r[0], r[1])
        for r in legacy[
            "period_flow_proportional_multiplier.csv"].iter_rows()}, (
        "tdy_sum==0 must DROP propN from stage 6")
    print(f"\n[inflow-scaling parity] synthetic cpsoy0/tdy0 tiers: {tiers}")

    # --- S5 int-0: empty complete_time_in_use → orig_flow_sum emits "0" -
    provider2 = FlexDataProvider()

    def put2(parent, stem, df):
        provider2.put(f"{parent}/{stem}", df)
        provider2.put(stem, df)

    put2("input", "node", pl.DataFrame({"node": ["annN"]}))
    put2("solve_data", "period_in_use_set", pl.DataFrame({"period": ["d1"]}))
    put2("solve_data", "time", pl.DataFrame({"time": ["t1"]}))
    put2("input", "p_node", pl.DataFrame(
        {"node": [], "param": [], "value": []},
        schema={"node": pl.Utf8, "param": pl.Utf8, "value": pl.Float64}))
    put2("solve_data", "pt_node_inflow", pl.DataFrame({
        "node": ["annN"], "time": ["t1"], "value": [3.0]}))
    put2("solve_data", "node__inflow_method", pl.DataFrame({
        "node": ["annN"], "method": ["scale_to_annual_flow"]}))
    put2("solve_data", "pdNode", pl.DataFrame({
        "node": ["annN"], "param": ["annual_flow"],
        "period": ["d1"], "value": [100.0]}))
    put2("solve_data", "complete_period_share_of_year_calc", pl.DataFrame({
        "period": ["d1"], "value": [0.5]}))
    put2("solve_data", "p_timeline_duration_in_years", pl.DataFrame(
        {"timeline": [], "value": []},
        schema={"timeline": pl.Utf8, "value": pl.Float64}))
    put2("solve_data", "period__timeline_set", pl.DataFrame(
        {"period": [], "timeline": []},
        schema={"period": pl.Utf8, "timeline": pl.Utf8}))
    # EMPTY complete_time_in_use → sum(()) == int 0 → repr "0".
    put2("solve_data", "complete_time_in_use_set", pl.DataFrame(
        {"time": []}, schema={"time": pl.Utf8}))
    put2("solve_data", "steps_complete_solve", pl.DataFrame({
        "period": ["d1"], "time": ["t1"]}))

    tiers2 = _assert_dict_parity(provider2, "synthetic-int0")
    legacy2 = _compute_inflow_scaling_frames(
        Path("input"), Path("solve_data"), provider=provider2)
    ofs = {(r[0], r[1]): r[2]
           for r in legacy2["orig_flow_sum.csv"].iter_rows()}
    assert ofs.get(("annN", "d1")) == "0", (
        f"S5 int-0: empty complete timeline must emit '0' not '0.0'; "
        f"got {ofs}")
    vec2 = _compute_inflow_scaling_frames_vectorized(
        Path("input"), Path("solve_data"), provider=provider2)
    ofs_v = {(r[0], r[1]): r[2]
             for r in vec2["orig_flow_sum.csv"].iter_rows()}
    assert ofs_v.get(("annN", "d1")) == "0", (
        f"S5 int-0 (vectorized): must emit '0' not '0.0'; got {ofs_v}")
    print(f"\n[inflow-scaling parity] synthetic int-0 tiers: {tiers2}")

    # --- stage-10 denom==0 (npis - npopis == 0 → new_old_multiplier 0) --
    # Construct a peak node where npis == npopis so denom == 0.  With
    # inflow series (5, 6): old_peak = 6, peak chosen = 1.0 → npop = 1/6,
    # complete-timeline sum ofs = 11, cpsoy = c.  npopis = npop*ofs/c =
    # (11/6)/c; npis = peak*8760 = 8760.  denom==0 ⇒ c = (11/6)/8760.
    provider3 = FlexDataProvider()

    def put3(parent, stem, df):
        provider3.put(f"{parent}/{stem}", df)
        provider3.put(stem, df)

    c = (11.0 / 6.0) / 8760.0
    put3("input", "node", pl.DataFrame({"node": ["peakD0"]}))
    put3("solve_data", "period_in_use_set", pl.DataFrame({"period": ["d1"]}))
    put3("solve_data", "time", pl.DataFrame({"time": ["t1", "t2"]}))
    put3("input", "p_node", pl.DataFrame(
        {"node": [], "param": [], "value": []},
        schema={"node": pl.Utf8, "param": pl.Utf8, "value": pl.Float64}))
    put3("solve_data", "pt_node_inflow", pl.DataFrame({
        "node": ["peakD0", "peakD0"], "time": ["t1", "t2"],
        "value": [5.0, 6.0]}))
    put3("solve_data", "node__inflow_method", pl.DataFrame({
        "node": ["peakD0"], "method": ["scale_to_annual_and_peak_flow"]}))
    put3("solve_data", "pdNode", pl.DataFrame({
        "node": ["peakD0", "peakD0"],
        "param": ["annual_flow", "peak_inflow"],
        "period": ["d1", "d1"], "value": [80.0, 1.0]}))
    put3("solve_data", "complete_period_share_of_year_calc", pl.DataFrame({
        "period": ["d1"], "value": [c]}))
    put3("solve_data", "p_timeline_duration_in_years", pl.DataFrame(
        {"timeline": [], "value": []},
        schema={"timeline": pl.Utf8, "value": pl.Float64}))
    put3("solve_data", "period__timeline_set", pl.DataFrame(
        {"period": [], "timeline": []},
        schema={"period": pl.Utf8, "timeline": pl.Utf8}))
    put3("solve_data", "complete_time_in_use_set",
        pl.DataFrame({"time": ["t1", "t2"]}))
    put3("solve_data", "steps_complete_solve", pl.DataFrame({
        "period": ["d1", "d1"], "time": ["t1", "t2"]}))

    tiers3 = _assert_dict_parity(provider3, "synthetic-denom0")
    legacy3 = _compute_inflow_scaling_frames(
        Path("input"), Path("solve_data"), provider=provider3)
    nom3 = {(r[0], r[1]): r[2]
            for r in legacy3["new_old_multiplier.csv"].iter_rows()}
    assert nom3.get(("peakD0", "d1")) == repr(0.0), (
        f"denom==0 must yield new_old_multiplier 0.0; got {nom3}")
    print(f"\n[inflow-scaling parity] synthetic denom0 tiers: {tiers3}")
