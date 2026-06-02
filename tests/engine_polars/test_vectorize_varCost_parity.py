"""Parity gate for the vectorized ``varCost`` pair + ``pssdt_varCost``
filters ×4.

The ``varCost`` pair (``pdtProcess__source__sink__dt_varCost`` basic +
``_alwaysProcess``) is NOT a coalesce cascade — it is a membership-GATED
SUM of three terms (design §5 / S7)::

    v = 0.0
    if (p, src) ∈ proc_src: v += pdt_src[(p, src, d, t)]
    if (p, snk) ∈ proc_snk: v += pdt_snk[(p, snk, d, t)]
    basic:  v += pdt[(p, d, t)]                            # unconditional
    always: if (p, snk) ∈ proc_snk or (p, snk) ∈ proc_src:
                v += pdt[(p, d, t)]

The four ``pssdt_varCost_*`` filters are KEY-only coordinate predicates
(value ≠ 0, membership-gated) — no value column.

Tier policy (design §1): each output is asserted strict
``df_vec.equals(df_legacy)`` (Tier A) first.  varCost is a PER-ROW sum
(not a group-by reduction) of at most three terms, with signed zero
normalized to match the legacy ``v = 0.0`` accumulator — so Tier A is
expected on real and synthetic data.  ``_assert_parity`` still demotes a
single offending cell to Tier B only on legitimate last-ULP drift
(``|Δ| ≤ ~1e-12``); any non-ULP difference is a real bug and the gate
fails hard.

The real fixtures (``fullYear`` / ``2_day_stochastic_dispatch``) author
NO ``other_operational_cost`` (all three pdt dicts are empty → varCost
all-zero, filters all-empty); the real-fixture gate is therefore VACUOUS
for coverage purposes and is asserted+reported as such.  The SYNTHETIC
tests below carry the genuine coverage: they author non-zero
``other_operational_cost`` so every term, gate, and predicate fires.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from flextool.engine_polars._emit_period_params import (
    _derive_pssdt_varCost_filters,
    _derive_pssdt_varCost_filters_vectorized,
    _derive_varCost_pair,
    _derive_varCost_pair_vectorized,
    _read_pdt_at_param,
)

_KEY_COLS = ["process", "source", "sink", "period", "time"]
_FILTER_NAMES = (
    "pssdt_varCost_noEff",
    "pssdt_varCost_eff_unit_source",
    "pssdt_varCost_eff_unit_sink",
    "pssdt_varCost_eff_connection",
)


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


def _ooc_dicts(p, sdd: Path) -> tuple[dict, dict, dict]:
    """The three ``other_operational_cost`` pdt dicts (verbatim readers).

    Used to detect whether a real fixture authors any
    ``other_operational_cost`` at all (an all-empty triple ⇒ the
    real-fixture varCost/filter gate is vacuous — the synthetic tests
    carry the real coverage).
    """
    pdt = _read_pdt_at_param(
        sdd / "pdtProcess.csv", param_col=1,
        param_value="other_operational_cost",
        key_cols=(0, 2, 3), val_col=4, provider=p)
    pdt_src = _read_pdt_at_param(
        sdd / "pdtProcess_source.csv", param_col=2,
        param_value="other_operational_cost",
        key_cols=(0, 1, 3, 4), val_col=5, provider=p)
    pdt_snk = _read_pdt_at_param(
        sdd / "pdtProcess_sink.csv", param_col=2,
        param_value="other_operational_cost",
        key_cols=(0, 1, 3, 4), val_col=5, provider=p)
    return pdt, pdt_src, pdt_snk


# --- Part 1: real fixtures (gate is likely VACUOUS — assert + report) ------

def _check_real_fixture(scenario_workdir, scenario: str, db_fixture: str):
    work = scenario_workdir(scenario, db_fixture=db_fixture)
    p = _provider_from_workdir(work)
    inp = work / "input"
    sdd = work / "solve_data"

    # varCost: compare BOTH outputs (basic AND always).
    leg_basic, leg_always = _derive_varCost_pair(inp, sdd, provider=p)
    vec_basic, vec_always = _derive_varCost_pair_vectorized(
        inp, sdd, provider=p)
    t_basic = _assert_parity(leg_basic, vec_basic, f"{scenario}:varCost.basic")
    t_always = _assert_parity(
        leg_always, vec_always, f"{scenario}:varCost.always")

    # filters: four KEY-only frames, Tier A strict `.equals`.
    leg_filters = _derive_pssdt_varCost_filters(inp, sdd, provider=p)
    vec_filters = _derive_pssdt_varCost_filters_vectorized(inp, sdd, provider=p)
    for name, lf, vf in zip(_FILTER_NAMES, leg_filters, vec_filters):
        assert vf.columns == lf.columns, (
            f"{scenario}:{name}: column mismatch {lf.columns} vs {vf.columns}")
        assert vf.equals(lf), (
            f"{scenario}:{name}: KEY-only filter parity failed\n"
            f"legacy=\n{lf}\nvec=\n{vf}")

    # Vacuity guard (do NOT silently pass a vacuous gate): report whether
    # the fixture authors any other_operational_cost.
    pdt, pdt_src, pdt_snk = _ooc_dicts(p, sdd)
    vacuous = not (pdt or pdt_src or pdt_snk)
    print(
        f"\n[varCost parity] {scenario}: varCost.basic=Tier {t_basic}, "
        f"varCost.always=Tier {t_always}, filters=Tier A; "
        f"other_operational_cost authored? "
        f"{'NO (gate VACUOUS — synthetic carries coverage)' if vacuous else 'YES'} "
        f"(pdt={len(pdt)}, pdt_src={len(pdt_src)}, pdt_snk={len(pdt_snk)})")
    return vacuous


def test_real_fixture_fullYear(scenario_workdir):
    _check_real_fixture(scenario_workdir, "fullYear", "main")


def test_real_fixture_stochastic(scenario_workdir):
    _check_real_fixture(
        scenario_workdir, "2_day_stochastic_dispatch", "stochastic")


# --- Part 2: synthetic varCost coverage (the genuine gate) -----------------

def test_synthetic_varCost_pair(tmp_path):
    """Drive the vectorized varCost producer vs the legacy scalar oracle on
    a synthetic fixture exercising EVERY term / gate path:

    * **src-only term:** ``(pA, s1, k1)`` — ``(pA, s1) ∈ proc_src`` with a
      ``pdt_src`` value; ``(pA, k1) ∉ proc_snk``; no ``pdt`` → v = src.
    * **snk-only term:** ``(pB, s2, k2)`` — ``(pB, k2) ∈ proc_snk`` with a
      ``pdt_snk`` value; ``(pB, s2) ∉ proc_src``; no ``pdt`` → v = snk.
    * **src + snk:** ``(pC, s3, k3)`` — both memberships + both values.
    * **src + snk + pdt (3-term):** ``(pD, s4, k4)`` — both memberships +
      both values + a ``pdt[(pD, d, t)]`` value (basic adds it).
    * **(p, src) ∉ proc_src ⇒ src gated OFF:** ``(pE, s5, k5)`` has a
      ``pdt_src`` value but NO ``proc_src`` arc → src term excluded.
    * **basic-vs-always pdt-gate DIVERGENCE:** ``(pF, s6, k6)`` with a
      ``pdt[(pF, d, t)]`` value where ``(pF, k6) ∉ proc_snk`` AND
      ``(pF, k6) ∉ proc_src`` → basic adds pdt, always gates it OFF → the
      two outputs differ for this cell.

    ``pss_always`` includes ``pF`` so the divergence row is in the always
    domain.  Oracle = legacy ``_derive_varCost_pair``; parity asserted on
    both basic and always (Tier A expected).
    """
    from flextool.engine_polars._flex_data_provider import FlexDataProvider

    provider = FlexDataProvider()

    def put(parent: str, stem: str, df: pl.DataFrame) -> None:
        provider.put(f"{parent}/{stem}", df)
        provider.put(stem, df)

    # process_source_sink (basic domain).
    put("solve_data", "process_source_sink", pl.DataFrame({
        "process": ["pA", "pB", "pC", "pD", "pE", "pF"],
        "source": ["s1", "s2", "s3", "s4", "s5", "s6"],
        "sink": ["k1", "k2", "k3", "k4", "k5", "k6"],
    }))
    # process_source_sink_alwaysProcess: include pF so the pdt-gate
    # divergence is in the always domain (and pD for a 3-term always cell).
    put("solve_data", "process_source_sink_alwaysProcess", pl.DataFrame({
        "process": ["pD", "pF"],
        "source": ["s4", "s6"],
        "sink": ["k4", "k6"],
    }))
    # process__source membership (proc_src): pA, pC, pD have src arcs; pE
    # does NOT (src term gated OFF), pF does NOT (so the always pdt gate's
    # proc_src branch is also OFF for k6 — k6 not a source either).
    put("input", "process__source", pl.DataFrame({
        "process": ["pA", "pC", "pD"],
        "source": ["s1", "s3", "s4"],
    }))
    # process__sink membership (proc_snk): pB, pC, pD have snk arcs; pF
    # does NOT (so the always pdt gate's proc_snk branch is OFF for k6).
    put("input", "process__sink", pl.DataFrame({
        "process": ["pB", "pC", "pD"],
        "sink": ["k2", "k3", "k4"],
    }))
    # pdtProcess_source (other_operational_cost): src term values.
    put("solve_data", "pdtProcess_source", pl.DataFrame({
        "process": ["pA", "pC", "pD", "pE"],
        "source": ["s1", "s3", "s4", "s5"],
        "param": ["other_operational_cost"] * 4,
        "period": ["d1"] * 4,
        "time": ["t1"] * 4,
        "value": [3.0, 5.0, 7.0, 9.0],
    }))
    # pdtProcess_sink (other_operational_cost): snk term values.
    put("solve_data", "pdtProcess_sink", pl.DataFrame({
        "process": ["pB", "pC", "pD"],
        "sink": ["k2", "k3", "k4"],
        "param": ["other_operational_cost"] * 3,
        "period": ["d1"] * 3,
        "time": ["t1"] * 3,
        "value": [11.0, 13.0, 17.0],
    }))
    # pdtProcess (other_operational_cost): process-level pdt term values.
    # pD (3-term) and pF (basic-vs-always divergence) carry a value.
    put("solve_data", "pdtProcess", pl.DataFrame({
        "process": ["pD", "pF"],
        "param": ["other_operational_cost"] * 2,
        "period": ["d1"] * 2,
        "time": ["t1"] * 2,
        "value": [19.0, 23.0],
    }))
    put("solve_data", "steps_in_use", pl.DataFrame({
        "period": ["d1"], "time": ["t1"]}))

    inp = tmp_path / "input"
    sdd = tmp_path / "solve_data"

    leg_basic, leg_always = _derive_varCost_pair(inp, sdd, provider=provider)
    vec_basic, vec_always = _derive_varCost_pair_vectorized(
        inp, sdd, provider=provider)

    # Oracle sanity: every term path landed where expected.
    lb = {(r[0], r[1], r[2]): r[5] for r in leg_basic.iter_rows()}
    assert lb[("pA", "s1", "k1")] == repr(3.0), lb        # src only
    assert lb[("pB", "s2", "k2")] == repr(11.0), lb       # snk only
    assert lb[("pC", "s3", "k3")] == repr(5.0 + 13.0), lb  # src + snk
    assert lb[("pD", "s4", "k4")] == repr(7.0 + 17.0 + 19.0), lb  # 3-term
    assert lb[("pE", "s5", "k5")] == repr(0.0), lb        # src gated OFF
    assert lb[("pF", "s6", "k6")] == repr(23.0), lb       # basic adds pdt

    la = {(r[0], r[1], r[2]): r[5] for r in leg_always.iter_rows()}
    # always: pD pdt gated ON ((pD, k4) ∈ proc_snk) → same 3-term.
    assert la[("pD", "s4", "k4")] == repr(7.0 + 17.0 + 19.0), la
    # always: pF pdt gated OFF ((pF, k6) ∉ proc_snk and ∉ proc_src) → only
    # src + snk, and pF has neither → 0.0.  This DIVERGES from basic (23.0).
    assert la[("pF", "s6", "k6")] == repr(0.0), la
    assert lb[("pF", "s6", "k6")] != la[("pF", "s6", "k6")], (
        "basic-vs-always pdt-gate divergence not exercised")

    t_basic = _assert_parity(leg_basic, vec_basic, "synthetic:varCost.basic")
    t_always = _assert_parity(
        leg_always, vec_always, "synthetic:varCost.always")
    assert t_basic == "A", "synthetic varCost.basic: unexpected Tier-B"
    assert t_always == "A", "synthetic varCost.always: unexpected Tier-B"
    print(f"\n[varCost parity] synthetic pair: basic=Tier {t_basic}, "
          f"always=Tier {t_always}")


# --- Part 3: synthetic filters coverage (the genuine gate) -----------------

def test_synthetic_pssdt_varCost_filters(tmp_path):
    """Drive the vectorized filters vs the legacy oracle on a synthetic
    fixture making all four filters non-empty and exercising each
    predicate, including value==0 exclusion and membership exclusion:

    * **noEff:** ``(pN, sN, kN)`` has ``varcost ≠ 0`` → kept; a sibling
      cell with ``varcost == 0`` → excluded.
    * **eff_unit_source:** ``(pS, sS, kS)`` ∈ proc_src + ``pdt_src ≠ 0`` →
      kept; ``(pS2, sS2, kS2)`` has ``pdt_src ≠ 0`` but ∉ proc_src →
      excluded (membership gate).
    * **eff_unit_sink:** ``(pK, sK, kK)`` ∈ proc_snk + ``pdt_snk ≠ 0`` →
      kept.
    * **eff_connection:** ``(pE, sE, kE)`` has ``pdt ≠ 0`` (no membership
      gate) → kept; a cell with ``pdt == 0`` → excluded.

    Oracle = legacy ``_derive_pssdt_varCost_filters``; Tier A ``.equals``.
    """
    from flextool.engine_polars._flex_data_provider import FlexDataProvider

    provider = FlexDataProvider()

    def put(parent: str, stem: str, df: pl.DataFrame) -> None:
        provider.put(f"{parent}/{stem}", df)
        provider.put(stem, df)

    # noEff domain: the varcost-predicate filter (filter 1).
    put("solve_data", "process_source_sink_noEff", pl.DataFrame({
        "process": ["pN", "pZ"],
        "source": ["sN", "sZ"],
        "sink": ["kN", "kZ"],
    }))
    # eff domain: filters 2/3/4 all read this domain.
    put("solve_data", "process_source_sink_eff", pl.DataFrame({
        "process": ["pS", "pS2", "pK", "pE", "pE0"],
        "source": ["sS", "sS2", "sK", "sE", "sE0"],
        "sink": ["kS", "kS2", "kK", "kE", "kE0"],
    }))
    # varCost values (filter 1 predicate): pN ≠ 0 kept, pZ == 0 excluded.
    put("solve_data", "pdtProcess__source__sink__dt_varCost", pl.DataFrame({
        "process": ["pN", "pZ"],
        "source": ["sN", "sZ"],
        "sink": ["kN", "kZ"],
        "period": ["d1", "d1"],
        "time": ["t1", "t1"],
        "value": [4.0, 0.0],
    }))
    # proc_src membership: pS in, pS2 NOT (excluded by membership gate).
    put("input", "process__source", pl.DataFrame({
        "process": ["pS"],
        "source": ["sS"],
    }))
    # proc_snk membership: pK in.
    put("input", "process__sink", pl.DataFrame({
        "process": ["pK"],
        "sink": ["kK"],
    }))
    # pdt_src (filter 2 value): pS ≠ 0 (kept), pS2 ≠ 0 (but ∉ proc_src).
    put("solve_data", "pdtProcess_source", pl.DataFrame({
        "process": ["pS", "pS2"],
        "source": ["sS", "sS2"],
        "param": ["other_operational_cost"] * 2,
        "period": ["d1"] * 2,
        "time": ["t1"] * 2,
        "value": [5.0, 6.0],
    }))
    # pdt_snk (filter 3 value): pK ≠ 0 (kept).
    put("solve_data", "pdtProcess_sink", pl.DataFrame({
        "process": ["pK"],
        "sink": ["kK"],
        "param": ["other_operational_cost"],
        "period": ["d1"],
        "time": ["t1"],
        "value": [8.0],
    }))
    # pdt (filter 4 value, no gate): pE ≠ 0 (kept), pE0 == 0 (excluded).
    put("solve_data", "pdtProcess", pl.DataFrame({
        "process": ["pE", "pE0"],
        "param": ["other_operational_cost"] * 2,
        "period": ["d1"] * 2,
        "time": ["t1"] * 2,
        "value": [9.0, 0.0],
    }))
    put("solve_data", "steps_in_use", pl.DataFrame({
        "period": ["d1"], "time": ["t1"]}))

    inp = tmp_path / "input"
    sdd = tmp_path / "solve_data"

    leg = _derive_pssdt_varCost_filters(inp, sdd, provider=provider)
    vec = _derive_pssdt_varCost_filters_vectorized(inp, sdd, provider=provider)

    # Oracle sanity: each filter is non-empty and exercises its predicate.
    no_eff, eff_src, eff_snk, eff_conn = leg
    assert no_eff.height == 1 and no_eff.row(0)[0] == "pN", no_eff
    assert eff_src.height == 1 and eff_src.row(0)[0] == "pS", eff_src
    assert eff_snk.height == 1 and eff_snk.row(0)[0] == "pK", eff_snk
    assert eff_conn.height == 1 and eff_conn.row(0)[0] == "pE", eff_conn

    for name, lf, vf in zip(_FILTER_NAMES, leg, vec):
        assert vf.columns == lf.columns, (
            f"synthetic:{name}: column mismatch {lf.columns} vs {vf.columns}")
        assert vf.equals(lf), (
            f"synthetic:{name}: KEY-only filter parity failed\n"
            f"legacy=\n{lf}\nvec=\n{vf}")
    print("\n[varCost parity] synthetic filters: Tier A (all four non-empty)")
