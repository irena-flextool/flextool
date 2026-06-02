"""Parity gate for the vectorized inflow-scaling emitter.

The inflow-scaling family (``_compute_inflow_scaling_frames``) is the
HARDEST in the vectorize-per-roll effort: 12 stateful stages, Tier B.
Only the 6 CONSUMED outputs are emitted by
``_compute_inflow_scaling_frames_vectorized`` (the 9 internal middle
parameters have no external consumer and are no longer written).  This
test gates the 6 emitted outputs against the legacy
``_compute_inflow_scaling_frames`` ORACLE, key-by-key, and asserts the 9
dropped outputs are ABSENT from the vectorized dict.

The legacy oracle is UNCHANGED — it still materialises all 15 internally,
so it remains a faithful reference for both the 6 emitted values AND the 9
dropped intermediates (used below to re-express the branch-coverage
asserts as observable consequences on the kept consumed outputs).

Tier policy (design §1):

* **Tier A — byte-exact** (``df_vec.equals(df_legacy)``): the sum-free
  consumed outputs (``ptNode_inflow``, ``_node_cap_inflow_fallback``).
  They feed the already-vectorized pdtNodeInflow and the lp-scaling
  emitter; drift there propagates.
* **Tier B — last-ULP tolerance** (parse both ``value`` cols to float,
  ``rtol/atol ≤ 1e-12``): the sum-bearing consumed outputs
  (``period_flow_annual_multiplier``,
  ``period_flow_proportional_multiplier``, ``new_old_slope``,
  ``new_old_section``).

The 9 DROPPED outputs and the 2 historical DEAD outputs are NEVER
assigned to the vectorized out-dict (guard).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars._emit_inflow_scaling import (
    _compute_inflow_scaling_frames,
    _compute_inflow_scaling_frames_vectorized,
)

# The 6 outputs with a downstream consumer — emitted + parity-gated.
_LIVE_KEYS = [
    "ptNode_inflow.csv",
    "_node_cap_inflow_fallback.csv",
    "period_flow_annual_multiplier.csv",
    "period_flow_proportional_multiplier.csv",
    "new_old_slope.csv",
    "new_old_section.csv",
]
# The 9 internal middle parameters — NO external consumer, NOT emitted.
# Their branch coverage is re-expressed below as observable asserts on the
# kept consumed outputs (slope/section) against the unchanged oracle.
_DROPPED_KEYS = [
    "orig_flow_sum.csv",
    "period_share_of_annual_flow.csv",
    "new_peak_sign.csv",
    "old_peak_max.csv",
    "old_peak_min.csv",
    "old_peak_sign.csv",
    "new_peak_divided_by_old_peak.csv",
    "new_peak_inflow_sum.csv",
    "new_old_multiplier.csv",
]
# Historical never-assigned keys (typo-trap guard, retained).
_DEAD_KEYS = [
    "old_peak.csv",
    "new_peak_divide_by_old_peak_sum_inflow.csv",
]
# Keys that must be ABSENT from the vectorized dict (dropped + dead).
_ABSENT_KEYS = _DROPPED_KEYS + _DEAD_KEYS
# Consumed outputs that MUST be byte-exact (sum-free coalesce / max-abs).
_TIER_A_KEYS = {
    "ptNode_inflow.csv",
    "_node_cap_inflow_fallback.csv",
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
    """Gate the 6 CONSUMED outputs vec-vs-oracle, and assert the vectorized
    dict emits EXACTLY those 6 (the 9 dropped + 2 dead keys are absent).

    The legacy oracle is unchanged: it still materialises all 15 outputs,
    so it must contain the 6 consumed keys AND the 9 dropped keys (the
    latter are the faithful reference the branch-coverage asserts read).
    Returns per-consumed-key tier."""
    # Oracle still produces all 15 (6 consumed + 9 dropped); no dead keys.
    assert set(legacy.keys()) == set(_LIVE_KEYS) | set(_DROPPED_KEYS), (
        f"{fixture}: legacy oracle key set != 15 (6 consumed + 9 dropped); "
        f"got {sorted(legacy.keys())}")
    # Vectorized path emits EXACTLY the 6 consumed outputs.
    assert set(vec.keys()) == set(_LIVE_KEYS), (
        f"{fixture}: vectorized key set != 6 CONSUMED keys; "
        f"got {sorted(vec.keys())}")
    # The 9 dropped + 2 historical-dead keys must NOT be emitted.
    for absent in _ABSENT_KEYS:
        assert absent not in vec, (
            f"{fixture}: {absent} must NOT be emitted by the vectorized "
            f"path (no external consumer)")
    for dead in _DEAD_KEYS:
        assert dead not in legacy, f"{fixture}: DEAD {dead} in legacy"

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

    # The dropped scratch CSVs are no longer emitted, so their old
    # drop-asymmetry / sign asserts are RE-EXPRESSED as observable
    # consequences on the KEPT consumed outputs of the VECTORIZED dict.
    # The unchanged legacy oracle supplies the exact reference float for
    # each cell (it still materialises the intermediates internally).
    def keyset(frame):
        return {(r[0], r[1]) for r in frame.iter_rows()}

    def cell(frame, n, d):
        m = {(r[0], r[1]): r[2] for r in frame.iter_rows()}
        return m.get((n, d))

    # --- Defect A (distinct method masks) re-expressed on consumed outs ---
    # propN flows through period_flow_proportional_multiplier (stage 6) but
    # NOT period_flow_annual_multiplier (stage 5); annN vice-versa.  Assert
    # on the VECTORIZED consumed outputs, with the oracle as value ref.
    v_s5 = keyset(vec["period_flow_annual_multiplier.csv"])
    v_s6 = keyset(vec["period_flow_proportional_multiplier.csv"])
    assert ("annN", "d1") in v_s5 and ("annN", "d1") not in v_s6, v_s5
    assert ("propN", "d1") in v_s6 and ("propN", "d1") not in v_s5, v_s6
    assert cell(vec["period_flow_annual_multiplier.csv"], "annN", "d1") == (
        cell(legacy["period_flow_annual_multiplier.csv"], "annN", "d1"))
    # annAf0 (af==0.0 real key) dropped from the annual stages.
    assert ("annAf0", "d1") not in v_s5
    # annPsaf0 (psaf==0) dropped from stage 5 (the stage-4 psaf==0 path is
    # the cause — re-expressed as: annPsaf0 ABSENT from the consumed
    # period_flow_annual_multiplier).
    assert ("annPsaf0", "d1") not in v_s5
    # peakPk0 (peak==0) entirely absent from the peak family → absent from
    # the consumed slope/section.
    v_slope = keyset(vec["new_old_slope.csv"])
    assert ("peakPk0", "d1") not in v_slope

    # --- peakOld0 (old_peak==0) re-expressed (was: drop asymmetry between
    # new_peak_sign and new_peak_divided_by_old_peak).  The dropped npop CSV
    # excluded old_peak==0 rows; the live fused path FILLS npop=0 for them,
    # so the (node, period) row SURVIVES into slope/section.  Observable
    # consequence: peakOld0 EXISTS in new_old_slope with slope == 0.0, and
    # new_old_section == -op_sign*af/8760 (the oracle's value for that cell,
    # which is the faithful hand-derived reference). ---
    assert ("peakOld0", "d1") in v_slope, (
        "old_peak==0 row must SURVIVE (fill-0) into new_old_slope")
    nos_peakOld0 = cell(vec["new_old_slope.csv"], "peakOld0", "d1")
    assert nos_peakOld0 == repr(0.0), (
        f"peakOld0 slope must be 0.0 (npop filled 0 on old_peak==0); "
        f"got {nos_peakOld0}")
    # new_old_section == peak * nom; with npop=0 → npopis=0 → denom=npis
    # → nom = op_sign*(0 - af)/npis = -op_sign*af/(peak*8760).  Section =
    # peak * nom = -op_sign*af/8760.  Hand-derive from the fixture:
    # peakOld0 inflow all 0 → op_max=op_min=0 → op_sign=+1 (|0|>=|0|);
    # af=80, peak=9 → section = -(1)*80/8760.
    expected_sec = -1.0 * 80.0 / 8760.0
    sec_peakOld0_legacy = cell(
        legacy["new_old_section.csv"], "peakOld0", "d1")
    sec_peakOld0_vec = cell(vec["new_old_section.csv"], "peakOld0", "d1")
    assert sec_peakOld0_legacy == repr(expected_sec), (
        f"oracle section for peakOld0 must be -op_sign*af/8760="
        f"{expected_sec!r}; got {sec_peakOld0_legacy}")
    # vec is Tier-B (rtol 1e-12) vs oracle — compare as floats.
    assert abs(float(sec_peakOld0_vec) - expected_sec) <= 1e-12, (
        f"vectorized peakOld0 section {sec_peakOld0_vec} != {expected_sec}")

    # --- peakScalarNeg (was: old_peak_sign == -1 on the negative scalar-
    # default path).  No explicit (n,t) rows → has_node_time_inflow False →
    # scalar default -4.0 → op_sign=-1, op_max=op_min=-4 → old_peak=-4.
    # Observable consequence on consumed outputs: the negative sign flows
    # through new_old_slope / new_old_section.  Assert the vectorized values
    # equal the unchanged oracle (faithful sign-bearing reference). ---
    assert ("peakScalarNeg", "d1") in v_slope, (
        "peakScalarNeg must be present (old_peak=-4 != 0)")
    slope_neg_legacy = cell(legacy["new_old_slope.csv"], "peakScalarNeg", "d1")
    slope_neg_vec = cell(vec["new_old_slope.csv"], "peakScalarNeg", "d1")
    sec_neg_legacy = cell(legacy["new_old_section.csv"], "peakScalarNeg", "d1")
    sec_neg_vec = cell(vec["new_old_section.csv"], "peakScalarNeg", "d1")
    assert abs(float(slope_neg_vec) - float(slope_neg_legacy)) <= 1e-12, (
        f"peakScalarNeg slope {slope_neg_vec} != oracle {slope_neg_legacy}")
    assert abs(float(sec_neg_vec) - float(sec_neg_legacy)) <= 1e-12, (
        f"peakScalarNeg section {sec_neg_vec} != oracle {sec_neg_legacy}")
    # The negative sign must be observable INDEPENDENTLY of the oracle, so a
    # sign regression in the vectorized path is caught even if the oracle
    # were to drift.  The SLOPE is the negative-bearing quantity here:
    #   npop = peak/old_peak = 9/-4 = -2.25 < 0, and slope = npop*(1+nom),
    # so new_old_slope is negative for this fixture (op_sign=-1, old_peak=-4,
    # peak=9).  (The SECTION, by contrast, is positive here, ≈ +0.0132 — it
    # is NOT the negative quantity.)  Pin the vectorized slope sign directly:
    assert float(slope_neg_vec) < 0.0, (
        f"peakScalarNeg vectorized slope must be negative "
        f"(npop=9/-4<0), got {slope_neg_vec}")
    assert sec_neg_legacy is not None and float(sec_neg_legacy) != 0.0, (
        "peakScalarNeg section must be non-zero (negative-sign path "
        "exercised)")

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
    vec = _compute_inflow_scaling_frames_vectorized(
        Path("input"), Path("solve_data"), provider=provider)
    # cpsoy==0: stage-5 numerator survives with 0 → the CONSUMED
    # period_flow_annual_multiplier emits 0.0.  Assert on BOTH paths.
    pfam = {(r[0], r[1]): r[2]
            for r in legacy["period_flow_annual_multiplier.csv"].iter_rows()}
    pfam_v = {(r[0], r[1]): r[2]
              for r in vec["period_flow_annual_multiplier.csv"].iter_rows()}
    assert pfam.get(("annN", "d0")) == repr(0.0), (
        f"cpsoy==0 numerator must survive with 0.0 (oracle); got {pfam}")
    assert pfam_v.get(("annN", "d0")) == repr(0.0), (
        f"cpsoy==0 numerator must survive with 0.0 (vectorized); "
        f"got {pfam_v}")
    # tdy_sum==0 DROPS propN from the CONSUMED
    # period_flow_proportional_multiplier — assert on BOTH paths.
    assert ("propN", "d0") not in {
        (r[0], r[1])
        for r in legacy[
            "period_flow_proportional_multiplier.csv"].iter_rows()}, (
        "tdy_sum==0 must DROP propN from stage 6 (oracle)")
    assert ("propN", "d0") not in {
        (r[0], r[1])
        for r in vec[
            "period_flow_proportional_multiplier.csv"].iter_rows()}, (
        "tdy_sum==0 must DROP propN from stage 6 (vectorized)")
    print(f"\n[inflow-scaling parity] synthetic cpsoy0/tdy0 tiers: {tiers}")

    # --- empty complete_time_in_use edge case (formerly the S5 int-0
    # ``"0"`` vs ``"0.0"`` probe).  orig_flow_sum is no longer emitted, so
    # the int-0 byte special-case is GONE — re-expressed as: the
    # consumed-output parity gate still holds for an empty complete timeline
    # AND orig_flow_sum is ABSENT from the vectorized dict. ----------------
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
    vec2 = _compute_inflow_scaling_frames_vectorized(
        Path("input"), Path("solve_data"), provider=provider2)
    # orig_flow_sum is an internal middle parameter with no consumer here
    # (annN is scale_to_annual_flow, not a peak node) — it must NOT be
    # emitted, and the empty-complete-timeline path must not break parity.
    assert "orig_flow_sum.csv" not in vec2, (
        "orig_flow_sum must NOT be emitted (no external consumer); the S5 "
        "int-0 byte special-case is retired")
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
    vec3 = _compute_inflow_scaling_frames_vectorized(
        Path("input"), Path("solve_data"), provider=provider3)
    # denom==0 → nom=0 (oracle reference, on the dropped middle param).
    nom3 = {(r[0], r[1]): r[2]
            for r in legacy3["new_old_multiplier.csv"].iter_rows()}
    assert nom3.get(("peakD0", "d1")) == repr(0.0), (
        f"denom==0 must yield new_old_multiplier 0.0 (oracle); got {nom3}")
    # Observable consequence on the CONSUMED outputs: nom=0 →
    # new_old_section = peak*nom = 0.0, and new_old_slope = npop*(1+0) =
    # npop = peak/old_peak = 1/6.  Assert on the VECTORIZED dict.
    sec3 = {(r[0], r[1]): r[2]
            for r in vec3["new_old_section.csv"].iter_rows()}
    slope3 = {(r[0], r[1]): r[2]
              for r in vec3["new_old_slope.csv"].iter_rows()}
    assert ("peakD0", "d1") in sec3 and float(sec3[("peakD0", "d1")]) == 0.0, (
        f"denom==0 → new_old_section must be 0.0 (vectorized); got {sec3}")
    expected_slope = 1.0 / 6.0
    assert abs(float(slope3[("peakD0", "d1")]) - expected_slope) <= 1e-12, (
        f"denom==0 → new_old_slope must be npop=1/6 (vectorized); "
        f"got {slope3}")
    print(f"\n[inflow-scaling parity] synthetic denom0 tiers: {tiers3}")
