"""Parity gate for the vectorized lp-scaling emitter.

The lp-scaling family computes 9 stage frames, but only 3 are CONSUMED
and emitted: ``node_capacity_for_scaling``, ``group_capacity_for_scaling``,
``inv_group_cap``.  The other 6 (``_node_cap_unitsize_sum``,
``_node_cap_raw``, ``_node_cap_pow10``, ``inv_node_cap``,
``_group_cap_raw``, ``_group_cap_pow10``) are stage-to-stage
middle-products with no functional reader; the vectorized emitter no
longer writes them.

The legacy ``_compute_lp_scaling_frames`` ORACLE STILL computes all 9
frames in-memory — it is the source of truth both for the 3 emitted
survivors (parity-gated key-by-key) and for the per-stage diagnostics on
the 6 dropped frames (read from the oracle dict, never from the vec dict
which no longer holds them).  This preserves the Defect-X half-decade,
Defect-Y scaling-off-unconditional, self-loop double-count, fallback-
branch and int-"0" coverage without re-emitting dead CSVs.

Tier policy (critique Defect X):

* NO lp key is hard-asserted Tier A in :func:`_gate_dicts`.  On real
  fixtures the unitsize / group sums are exact-integer so no ULP drift
  occurs and the surviving LP-coefficient keys
  (node_capacity_for_scaling / group_capacity_for_scaling /
  inv_group_cap) land Tier A byte-exact in practice.  But a pathological
  sum landing exactly on a half-decade ``10^(k+0.5)`` could flip a
  ``_pow10_round_clamped`` decade bucket → factor-10 gap →
  ``_assert_parity`` RAISES (loud, correct — never masked).
  ``_TIER_A_KEYS`` is therefore EMPTY: ``_assert_parity`` reports the
  achieved tier per key; we assert only structural + ≤1e-12.

The 3 SURVIVING outputs: all 3-col all-Utf8, value = repr(v).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from flextool.engine_polars._emit_lp_scaling import (
    _compute_lp_scaling_frames,
    _compute_lp_scaling_frames_vectorized,
)

# The 3 CONSUMED outputs the vectorized emitter still produces.
_LIVE_KEYS = [
    "node_capacity_for_scaling.csv",
    "group_capacity_for_scaling.csv",
    "inv_group_cap.csv",
]
# The 6 dropped stage-to-stage middle-products: still in the legacy ORACLE
# dict (read for diagnostics), MUST be absent from the vectorized dict.
_DROPPED_KEYS = [
    "_node_cap_unitsize_sum.csv",
    "_node_cap_raw.csv",
    "_node_cap_pow10.csv",
    "inv_node_cap.csv",
    "_group_cap_raw.csv",
    "_group_cap_pow10.csv",
]
# The LP-coefficient keys expected byte-exact (Tier A) on real fixtures —
# exact-integer sums, so no ULP drift.  Reported, not hard-asserted.
_LP_COEFF_KEYS = {
    "node_capacity_for_scaling.csv",
    "group_capacity_for_scaling.csv",
    "inv_group_cap.csv",
}
# Tier-A hard-assert set is EMPTY (critique Defect X): let _assert_parity
# report the achieved tier; never force Tier A on any lp key.
_TIER_A_KEYS: set[str] = set()


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
    mismatch or a non-ULP value difference (a real bug — e.g. a flipped
    pow10 decade bucket → factor-10 gap — STOPs the gate).
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
            f"1e-12 — NOT a last-ULP demotion, a real bug (possibly a "
            f"flipped pow10 decade bucket):\n{bad}")
    return "B"


def _gate_dicts(legacy: dict, vec: dict, fixture: str) -> dict[str, str]:
    """Gate the 3 SURVIVING keys with :func:`_assert_parity`.

    The legacy ORACLE still computes all 9 frames (3 survivors + 6 dropped
    middle-products) — assert it holds exactly that 9-key set.  The
    vectorized dict must hold EXACTLY the 3 survivors and NONE of the 6
    dropped keys.  Returns per-key tier for the survivors."""
    assert set(legacy.keys()) == set(_LIVE_KEYS) | set(_DROPPED_KEYS), (
        f"{fixture}: legacy oracle key set != 9 (3 live + 6 dropped) keys; "
        f"got {sorted(legacy.keys())}")
    assert set(vec.keys()) == set(_LIVE_KEYS), (
        f"{fixture}: vectorized key set != 3 SURVIVING keys; "
        f"got {sorted(vec.keys())}")
    # The 6 dropped middle-products must be ABSENT from the vectorized dict
    # (reading any of them from vec would KeyError — they are oracle-only).
    for k in _DROPPED_KEYS:
        assert k not in vec, (
            f"{fixture}: dropped middle-product {k!r} unexpectedly present "
            f"in the vectorized dict")

    tiers: dict[str, str] = {}
    for k in _LIVE_KEYS:
        lf = legacy[k]
        vf = vec[k]
        assert vf.columns == lf.columns, (
            f"{fixture}/{k}: columns {vf.columns} != {lf.columns}")
        assert vf.shape == lf.shape, (
            f"{fixture}/{k}: shape {vf.shape} != {lf.shape}")
        tier = _assert_parity(lf, vf, f"{fixture}/{k}")
        if k in _TIER_A_KEYS:  # empty by design — never fires
            assert tier == "A", (
                f"{fixture}/{k}: expected byte-exact Tier A, got {tier}")
        tiers[k] = tier
    return tiers


def _total_nonzero(legacy: dict) -> int:
    total = 0
    for k in _LIVE_KEYS:
        df = legacy[k]
        if df.height == 0:
            continue
        total += df.filter(pl.col("value").cast(pl.Float64) != 0.0).height
    return total


def _assert_fallback_present(work: Path, p) -> None:
    """Fail loudly if the inflow fallback CSV is absent from the snapshot.

    Critique: a missing ``_node_cap_inflow_fallback.csv`` silently routes
    every node to the 1.0 raw path, and oracle parity STILL passes — which
    would mask the lp↔inflow coupling.  Require it present.
    """
    on_disk = (work / "solve_data" / "_node_cap_inflow_fallback.csv").exists()
    in_provider = (
        p.get("solve_data/_node_cap_inflow_fallback") is not None
        or p.get("_node_cap_inflow_fallback") is not None
    )
    assert on_disk or in_provider, (
        f"{work}: _node_cap_inflow_fallback.csv absent from the snapshot — "
        f"every node would silently route to the 1.0 raw path and oracle "
        f"parity would mask the lp↔inflow coupling")


def _run_fixture(scenario_workdir, scenario: str, db_fixture: str,
                 *, require_nonzero: bool = True) -> dict[str, str]:
    work = scenario_workdir(scenario, db_fixture=db_fixture)
    p = _provider_from_workdir(work)
    inp = work / "input"
    sdd = work / "solve_data"

    _assert_fallback_present(work, p)

    legacy = _compute_lp_scaling_frames(inp, sdd, provider=p)
    vec = _compute_lp_scaling_frames_vectorized(inp, sdd, provider=p)

    tiers = _gate_dicts(legacy, vec, f"{scenario}/{db_fixture}")

    # Non-vacuity guard: ≥1 group, ≥1 node, ≥1 non-zero emitted value.
    nodes = p.get("input/node")
    groups = p.get("input/group")
    assert nodes is not None and nodes.height >= 1, (
        f"{scenario}: no nodes — gate vacuous")
    assert groups is not None and groups.height >= 1, (
        f"{scenario}: no groups — gate vacuous")
    if require_nonzero:
        assert _total_nonzero(legacy) > 0, (
            f"{scenario}: every LIVE frame emits only zeros — gate vacuous")
    return tiers


def test_lp_scaling_fullYear(scenario_workdir):
    tiers = _run_fixture(scenario_workdir, "fullYear", "main")
    # On real fixtures the LP-coefficient keys land byte-exact (Tier A).
    for k in _LP_COEFF_KEYS:
        assert tiers[k] == "A", (
            f"fullYear/{k}: expected byte-exact Tier A on the LP-coefficient "
            f"key (exact-integer sums), got Tier {tiers[k]}")
    print(f"\n[lp-scaling parity] fullYear tiers: {tiers}")


def test_lp_scaling_stochastic(scenario_workdir):
    # 2_day_stochastic_dispatch (stochastic db): lp has no stochastic
    # branch, so this is a structural-parity check (frames may be sparse).
    tiers = _run_fixture(
        scenario_workdir, "2_day_stochastic_dispatch", "stochastic",
        require_nonzero=False)
    print(f"\n[lp-scaling parity] 2_day_stochastic_dispatch tiers: {tiers}")


# --- Synthetic coverage (every differentiating branch) ---------------------


def _put(provider, parent: str, stem: str, df: pl.DataFrame) -> None:
    provider.put(f"{parent}/{stem}", df)
    provider.put(stem, df)


def _assert_dict_parity(provider, label: str) -> dict[str, str]:
    inp = Path("input")
    sdd = Path("solve_data")
    legacy = _compute_lp_scaling_frames(inp, sdd, provider=provider)
    vec = _compute_lp_scaling_frames_vectorized(inp, sdd, provider=provider)
    return _gate_dicts(legacy, vec, label)


def test_lp_scaling_synthetic_node_chain(tmp_path):
    """Synthetic coverage for the node chain U / R / P10:

    * self-loop arc (source==sink) → unitsize DOUBLE-count (both ifs fire);
    * a node that is an end of ≥3 arcs → ≥3-term unitsize sum (Tier-B
      demotion probe);
    * node with usz==0 + fallback>0 → fallback branch of R;
    * node with usz==0 + fallback==0 → 1.0 branch of R;
    * node with usz==0 + fallback MISSING → 1.0 branch of R;
    * arc-less node → unitsize "0.0" (float, not int-0).
    """
    from flextool.engine_polars._flex_data_provider import FlexDataProvider

    provider = FlexDataProvider()

    # Nodes:
    #   selfLoop  — one arc source==sink==selfLoop, usz 7.0 → counted TWICE
    #               → unitsize 14.0.
    #   multiArc  — sink of 3 distinct arcs (usz 1.0, 2.0, 3.0) → 6.0 sum
    #               (≥3-term Tier-B probe).
    #   fbNode    — no arc (usz 0) but fallback>0 → R picks fallback.
    #   fb0Node   — no arc, fallback==0.0 → R picks 1.0.
    #   missNode  — no arc, fallback MISSING → R picks 1.0.
    #   arcless   — never an arc end at all → unitsize "0.0".
    nodes = ["selfLoop", "multiArc", "fbNode", "fb0Node", "missNode",
             "arcless"]
    _put(provider, "input", "node", pl.DataFrame({"node": nodes}))
    _put(provider, "input", "group",
         pl.DataFrame({"group": ["g1"]}))
    _put(provider, "input", "group__node",
         pl.DataFrame({"group": ["g1"], "node": ["multiArc"]}))
    _put(provider, "solve_data", "period_in_use_set",
         pl.DataFrame({"period": ["d1"]}))
    # scaling active.
    _put(provider, "solve_data", "solve_current",
         pl.DataFrame({"solve": ["s1"]}))
    _put(provider, "solve_data", "p_use_row_scaling",
         pl.DataFrame({"solve": ["s1"], "value": [1.0]}))
    # process_source_sink triples (process, source, sink).
    pss = [
        ("pSelf", "selfLoop", "selfLoop"),   # self-loop → double count
        ("pA", "x", "multiArc"),             # multiArc sink, usz 1.0
        ("pB", "y", "multiArc"),             # multiArc sink, usz 2.0
        ("pC", "z", "multiArc"),             # multiArc sink, usz 3.0
    ]
    _put(provider, "solve_data", "process_source_sink", pl.DataFrame({
        "process": [r[0] for r in pss],
        "source": [r[1] for r in pss],
        "sink": [r[2] for r in pss]}))
    _put(provider, "solve_data", "p_entity_unitsize", pl.DataFrame({
        "process": ["pSelf", "pA", "pB", "pC"],
        "value": [7.0, 1.0, 2.0, 3.0]}))
    # fallback: fbNode>0; fb0Node==0.0; missNode ABSENT.
    _put(provider, "solve_data", "_node_cap_inflow_fallback", pl.DataFrame({
        "node": ["fbNode", "fb0Node"],
        "period": ["d1", "d1"],
        "value": [55.0, 0.0]}))

    tiers = _assert_dict_parity(provider, "synthetic-node-chain")

    legacy = _compute_lp_scaling_frames(
        Path("input"), Path("solve_data"), provider=provider)
    usz = {(r[0], r[1]): r[2]
           for r in legacy["_node_cap_unitsize_sum.csv"].iter_rows()}
    # self-loop double count.
    assert usz[("selfLoop", "d1")] == repr(14.0), usz
    # ≥3-term sum.
    assert usz[("multiArc", "d1")] == repr(6.0), usz
    # arc-less node → "0.0" float (not int-0).
    assert usz[("arcless", "d1")] == "0.0", usz
    raw = {(r[0], r[1]): r[2]
           for r in legacy["_node_cap_raw.csv"].iter_rows()}
    assert raw[("fbNode", "d1")] == repr(55.0), raw     # fallback branch
    assert raw[("fb0Node", "d1")] == repr(1.0), raw     # fb==0 → 1.0
    assert raw[("missNode", "d1")] == repr(1.0), raw    # fb miss → 1.0
    assert raw[("selfLoop", "d1")] == repr(14.0), raw   # usz>0 branch
    print(f"\n[lp-scaling parity] synthetic node-chain tiers: {tiers}")


def test_lp_scaling_synthetic_scaling_inactive(tmp_path):
    """``scaling_active=False`` collapse: node_capacity_for_scaling and
    inv_node_cap collapse to 1.0 regardless of the pow10 value (the
    scaling_active Python branch).  The same provider as a control where
    scaling IS active proves the gate distinguishes the two."""
    from flextool.engine_polars._flex_data_provider import FlexDataProvider

    def build(active: bool):
        provider = FlexDataProvider()
        _put(provider, "input", "node", pl.DataFrame({"node": ["nA"]}))
        _put(provider, "input", "group", pl.DataFrame({"group": ["g1"]}))
        _put(provider, "input", "group__node",
             pl.DataFrame({"group": ["g1"], "node": ["nA"]}))
        _put(provider, "solve_data", "period_in_use_set",
             pl.DataFrame({"period": ["d1"]}))
        _put(provider, "solve_data", "solve_current",
             pl.DataFrame({"solve": ["s1"]}))
        # p_use_row_scaling sum < 0.5 → scaling_active False.
        _put(provider, "solve_data", "p_use_row_scaling",
             pl.DataFrame({"solve": ["s1"],
                           "value": [1.0 if active else 0.0]}))
        # nA: usz 500.0 → raw 500 → pow10 1000 (a value clearly != 1.0).
        _put(provider, "solve_data", "process_source_sink", pl.DataFrame({
            "process": ["p1"], "source": ["x"], "sink": ["nA"]}))
        _put(provider, "solve_data", "p_entity_unitsize", pl.DataFrame({
            "process": ["p1"], "value": [500.0]}))
        _put(provider, "solve_data", "_node_cap_inflow_fallback",
             pl.DataFrame({"node": [], "period": [], "value": []},
                          schema={"node": pl.Utf8, "period": pl.Utf8,
                                  "value": pl.Float64}))
        return provider

    # --- scaling INACTIVE: NCFS / INC collapse to 1.0 ------------------
    p_off = build(active=False)
    tiers_off = _assert_dict_parity(p_off, "synthetic-scaling-off")
    legacy_off = _compute_lp_scaling_frames(
        Path("input"), Path("solve_data"), provider=p_off)
    ncfs_off = {(r[0], r[1]): r[2]
                for r in legacy_off[
                    "node_capacity_for_scaling.csv"].iter_rows()}
    inc_off = {(r[0], r[1]): r[2]
               for r in legacy_off["inv_node_cap.csv"].iter_rows()}
    assert ncfs_off[("nA", "d1")] == repr(1.0), ncfs_off
    assert inc_off[("nA", "d1")] == repr(1.0), inc_off
    # pow10 is still the REAL value (1000) — the collapse is NCFS-only.
    p10_off = {(r[0], r[1]): r[2]
               for r in legacy_off["_node_cap_pow10.csv"].iter_rows()}
    assert p10_off[("nA", "d1")] == repr(1000.0), p10_off

    # --- control: scaling ACTIVE → NCFS == pow10 ----------------------
    p_on = build(active=True)
    _assert_dict_parity(p_on, "synthetic-scaling-on")
    legacy_on = _compute_lp_scaling_frames(
        Path("input"), Path("solve_data"), provider=p_on)
    ncfs_on = {(r[0], r[1]): r[2]
               for r in legacy_on[
                   "node_capacity_for_scaling.csv"].iter_rows()}
    assert ncfs_on[("nA", "d1")] == repr(1000.0), ncfs_on
    print(f"\n[lp-scaling parity] synthetic scaling-off tiers: {tiers_off}")


def test_lp_scaling_synthetic_group_chain(tmp_path):
    """Synthetic coverage for the group chain GR / GP10 / GCFS / IGC:

    * member-less group → GR emits the literal ``"0"`` (S5 int-0), on BOTH
      legacy and vectorized;
    * group with ≥3 member nodes → ≥3-term graw (Tier-B demotion probe);
    * Defect Y — with scaling_active=False, GR and GP10 still emit the REAL
      sum (UNCONDITIONAL), while GCFS / IGC collapse to 1.0;
    * a member node ABSENT from the node list → ncfs default 1.0 in the GR
      sum (the ``.get((n, d), 1.0)`` branch).
    """
    from flextool.engine_polars._flex_data_provider import FlexDataProvider

    def build(active: bool):
        provider = FlexDataProvider()
        # Member nodes with distinct unitsizes so ncfs (= pow10) differs:
        #   m1 usz 9.0   → raw 9   → pow10 10.0
        #   m2 usz 90.0  → raw 90  → pow10 100.0
        #   m3 usz 900.0 → raw 900 → pow10 1000.0
        # gMulti members {m1, m2, m3, mGhost}: graw = 10 + 100 + 1000 +
        #   1.0 (mGhost absent from node list → ncfs default 1.0) = 1111.0
        #   (≥3-term sum → Tier-B probe).
        # gEmpty: no group__node row → member-less → GR "0".
        nodes = ["m1", "m2", "m3"]
        _put(provider, "input", "node", pl.DataFrame({"node": nodes}))
        _put(provider, "input", "group",
             pl.DataFrame({"group": ["gMulti", "gEmpty"]}))
        # gMulti includes mGhost which is NOT in the node list.
        _put(provider, "input", "group__node", pl.DataFrame({
            "group": ["gMulti", "gMulti", "gMulti", "gMulti"],
            "node": ["m1", "m2", "m3", "mGhost"]}))
        _put(provider, "solve_data", "period_in_use_set",
             pl.DataFrame({"period": ["d1"]}))
        _put(provider, "solve_data", "solve_current",
             pl.DataFrame({"solve": ["s1"]}))
        _put(provider, "solve_data", "p_use_row_scaling",
             pl.DataFrame({"solve": ["s1"],
                           "value": [1.0 if active else 0.0]}))
        _put(provider, "solve_data", "process_source_sink", pl.DataFrame({
            "process": ["pm1", "pm2", "pm3"],
            "source": ["x", "y", "z"],
            "sink": ["m1", "m2", "m3"]}))
        _put(provider, "solve_data", "p_entity_unitsize", pl.DataFrame({
            "process": ["pm1", "pm2", "pm3"],
            "value": [9.0, 90.0, 900.0]}))
        _put(provider, "solve_data", "_node_cap_inflow_fallback",
             pl.DataFrame({"node": [], "period": [], "value": []},
                          schema={"node": pl.Utf8, "period": pl.Utf8,
                                  "value": pl.Float64}))
        return provider

    # --- scaling ACTIVE ----------------------------------------------
    p_on = build(active=True)
    tiers_on = _assert_dict_parity(p_on, "synthetic-group-on")
    # GR / GP10 are dropped middle-products → read from the ORACLE dict only
    # (the vec dict no longer holds them; _gate_dicts already proved the
    # surviving GCFS/IGC byte-parity).
    legacy = _compute_lp_scaling_frames(
        Path("input"), Path("solve_data"), provider=p_on)

    graw_l = {(r[0], r[1]): r[2]
              for r in legacy["_group_cap_raw.csv"].iter_rows()}
    # member-less group → literal "0" (S5 int-0).
    assert graw_l[("gEmpty", "d1")] == "0", graw_l
    # ≥3-term sum incl. the ghost-node default 1.0.
    assert graw_l[("gMulti", "d1")] == repr(1111.0), graw_l
    # GP10 of the member-less group: graw 0 → pow10 1.0.
    gp10_l = {(r[0], r[1]): r[2]
              for r in legacy["_group_cap_pow10.csv"].iter_rows()}
    assert gp10_l[("gEmpty", "d1")] == repr(1.0), gp10_l
    # GCFS active → equals GP10 (when scaling_active, the SURVIVING
    # group_capacity_for_scaling output reproduces the GP10 stage value).
    gcfs_l = {(r[0], r[1]): r[2]
              for r in legacy[
                  "group_capacity_for_scaling.csv"].iter_rows()}
    assert gcfs_l[("gMulti", "d1")] == gp10_l[("gMulti", "d1")], gcfs_l

    # --- Defect Y: scaling INACTIVE → GR/GP10 UNCONDITIONAL ----------
    p_off = build(active=False)
    tiers_off = _assert_dict_parity(p_off, "synthetic-group-off")
    legacy_off = _compute_lp_scaling_frames(
        Path("input"), Path("solve_data"), provider=p_off)
    graw_off = {(r[0], r[1]): r[2]
                for r in legacy_off["_group_cap_raw.csv"].iter_rows()}
    gp10_off = {(r[0], r[1]): r[2]
                for r in legacy_off["_group_cap_pow10.csv"].iter_rows()}
    gcfs_off = {(r[0], r[1]): r[2]
                for r in legacy_off[
                    "group_capacity_for_scaling.csv"].iter_rows()}
    igc_off = {(r[0], r[1]): r[2]
               for r in legacy_off["inv_group_cap.csv"].iter_rows()}
    # Defect Y: GR is computed UNCONDITIONALLY — it always sums the ncfs
    # values, it is NOT itself replaced by 1.0.  But ncfs ITSELF collapses
    # to 1.0 when scaling is inactive, so graw = sum of four 1.0s = 4.0
    # (NOT 1.0, and NOT 1111.0).  GP10(4.0) = 10^round(log10(4)) = 10.0.
    # The proof of "GR not GCFS-gated" is that GR/GP10 emit 4.0/10.0 while
    # GCFS/IGC are forced to 1.0 — three DISTINCT values.
    assert graw_off[("gMulti", "d1")] == repr(4.0), graw_off
    assert gp10_off[("gMulti", "d1")] == repr(10.0), gp10_off
    # GCFS / IGC collapse to 1.0 (the gated stage).
    assert gcfs_off[("gMulti", "d1")] == repr(1.0), gcfs_off
    assert igc_off[("gMulti", "d1")] == repr(1.0), igc_off

    print(f"\n[lp-scaling parity] synthetic group-chain tiers on={tiers_on} "
          f"off={tiers_off}")


def test_lp_scaling_synthetic_half_decade_probe(tmp_path):
    """Half-decade bucket-stability probe (critique requirement 6).

    Author a node ``raw`` and a group ``graw`` at / near the half-decade
    boundary ``10^(k+0.5) = sqrt(10)*10^k`` and assert the vectorized pow10
    bucket matches the legacy bucket on BOTH the node and the group chain —
    proving the shared scalar-UDF chain is bucket-stable (no ULP-driven
    decade flip).  ``round(log10(v))`` of a value just below the half-
    decade rounds DOWN (10^k); just above rounds UP (10^(k+1)).
    """
    import math

    from flextool.engine_polars._flex_data_provider import FlexDataProvider

    half = math.sqrt(10.0)  # 10^0.5 ≈ 3.1623 — the decade boundary.
    # Node raw just BELOW the boundary → pow10 should be 1.0 (10^0); just
    # ABOVE → 10.0 (10^1).  A group graw straddling 10^1.5 similarly.
    nbelow = half * (1.0 - 1e-9)
    nabove = half * (1.0 + 1e-9)

    provider = FlexDataProvider()
    nodes = ["below", "above"]
    _put(provider, "input", "node", pl.DataFrame({"node": nodes}))
    _put(provider, "input", "group",
         pl.DataFrame({"group": ["gBoundary"]}))
    # gBoundary sums below+above ncfs.  We don't assert the exact group
    # bucket value (it depends on the node pow10s) — we assert vec==legacy
    # byte-parity, which is the real bucket-stability claim.
    _put(provider, "input", "group__node", pl.DataFrame({
        "group": ["gBoundary", "gBoundary"], "node": ["below", "above"]}))
    _put(provider, "solve_data", "period_in_use_set",
         pl.DataFrame({"period": ["d1"]}))
    _put(provider, "solve_data", "solve_current",
         pl.DataFrame({"solve": ["s1"]}))
    _put(provider, "solve_data", "p_use_row_scaling",
         pl.DataFrame({"solve": ["s1"], "value": [1.0]}))
    _put(provider, "solve_data", "process_source_sink", pl.DataFrame({
        "process": ["pb", "pa"], "source": ["x", "y"],
        "sink": ["below", "above"]}))
    _put(provider, "solve_data", "p_entity_unitsize", pl.DataFrame({
        "process": ["pb", "pa"], "value": [nbelow, nabove]}))
    _put(provider, "solve_data", "_node_cap_inflow_fallback",
         pl.DataFrame({"node": [], "period": [], "value": []},
                      schema={"node": pl.Utf8, "period": pl.Utf8,
                              "value": pl.Float64}))

    tiers = _assert_dict_parity(provider, "synthetic-half-decade")
    legacy = _compute_lp_scaling_frames(
        Path("input"), Path("solve_data"), provider=provider)
    p10 = {(r[0], r[1]): r[2]
           for r in legacy["_node_cap_pow10.csv"].iter_rows()}
    # The boundary rounds: below → 10^0 = 1.0; above → 10^1 = 10.0.
    assert p10[("below", "d1")] == repr(1.0), p10
    assert p10[("above", "d1")] == repr(10.0), p10
    print(f"\n[lp-scaling parity] synthetic half-decade tiers: {tiers}")
