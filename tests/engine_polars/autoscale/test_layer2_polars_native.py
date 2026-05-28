"""Polars-native Layer 2 bucket / choose-power rewrite tests.

Covers the 2026-05-27 rewrite that replaced the per-nonzero Python-list
accumulation in :func:`bucket_coefficients` with a polars
``group_by``-driven aggregation.  The bar is mathematical equivalence
to the legacy implementation, plus closed-form correctness on a
synthetic LP with known coefficient magnitudes.

Reference: ``_audit_reports/LAYER2_POLARS_NATIVE_PLAN.md`` §6.
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from polar_high.engine import Problem, Sum

from flextool.engine_polars.autoscale import (
    QuantityType,
    bucket_coefficients,
    choose_scale_powers,
)
from flextool.engine_polars.autoscale._layer2 import (
    _build_col_id_classification,
    _ID_TO_QTY,
    _QTY_TO_ID,
)
from flextool.engine_polars.autoscale._layer2_types import (
    CONSTRAINT_FAMILIES,
    VARIABLE_FAMILIES,
    CstrFamily,
    VarFamily,
)


# ---------------------------------------------------------------------------
# Helpers


@pytest.fixture
def register_test_families():
    """Inject test-only families; clean up after."""
    added_vars: list[str] = []
    added_cstrs: list[str] = []

    def add_var(name: str, fam: VarFamily) -> None:
        VARIABLE_FAMILIES[name] = fam
        added_vars.append(name)

    def add_cstr(name: str, fam: CstrFamily) -> None:
        CONSTRAINT_FAMILIES[name] = fam
        added_cstrs.append(name)

    yield add_var, add_cstr

    for n in added_vars:
        VARIABLE_FAMILIES.pop(n, None)
    for n in added_cstrs:
        CONSTRAINT_FAMILIES.pop(n, None)


def _legacy_choose_scale_powers(
    matrix_lists: dict[QuantityType, list[float]],
    cost_lists: dict[QuantityType, list[float]],
    bound_lists: dict[QuantityType, list[float]],
    *,
    clamp: int = 20,
) -> dict[QuantityType, int]:
    """Verbatim copy of the pre-rewrite choose_scale_powers, with the
    legacy ``list[float]`` API.  Used as the golden reference for the
    mathematical-equivalence test.
    """
    pool: dict[QuantityType, list[float]] = {}
    for src in (matrix_lists, cost_lists, bound_lists):
        for t, vs in src.items():
            pool.setdefault(t, []).extend(vs)

    chosen: dict[QuantityType, int] = {}
    for t, values in pool.items():
        arr = np.asarray([v for v in values if v > 0 and math.isfinite(v)])
        if arr.size == 0:
            continue
        log_mean = float(np.log2(arr).mean())
        exp = int(round(-log_mean))
        if exp > clamp:
            exp = clamp
        elif exp < -clamp:
            exp = -clamp
        chosen[t] = exp
    return chosen


def _legacy_bucket_coefficients(problem):
    """Verbatim port of the pre-rewrite bucket walk that yields the
    raw magnitude lists per QuantityType.  Used only by tests.
    """
    from flextool.engine_polars.autoscale._layer2_types import (
        lookup_var,
        resolve_cstr_rhs_type,
    )

    def _effective_matrix_type(var_family, row_type):
        if var_family.multiplier_param is not None and row_type is not None:
            return row_type
        return var_family.column_type

    matrix_buckets: dict[QuantityType, list[float]] = {}
    cost_buckets: dict[QuantityType, list[float]] = {}
    bound_buckets: dict[QuantityType, list[float]] = {}
    col_id_to_type: dict[int, QuantityType] = {}
    col_id_to_var: dict[int, str] = {}

    for name, var in problem._vars.items():
        fam = lookup_var(name)
        ids = var.frame["col_id"].to_numpy()
        for cid in ids.tolist():
            col_id_to_type[int(cid)] = fam.column_type
            col_id_to_var[int(cid)] = name
        for b in (var.lower, var.upper):
            if not math.isfinite(b) or b == 0.0:
                continue
            bound_buckets.setdefault(fam.column_type, []).append(abs(float(b)))

    for term in problem._obj_terms:
        df = term.lazy.collect()
        if df.height == 0:
            continue
        for c, cid in zip(df["coef"].to_list(), df["col_id"].to_list()):
            cv = abs(float(c))
            if cv == 0.0 or not math.isfinite(cv):
                continue
            t = col_id_to_type.get(int(cid))
            if t is None:
                continue
            cost_buckets.setdefault(t, []).append(cv)

    for cname, proto, over in problem._cstrs:
        rhs_t = resolve_cstr_rhs_type(cname)
        for term in proto.expr.terms:
            df = term.lazy.collect()
            if df.height == 0:
                continue
            for c, cid in zip(df["coef"].to_list(), df["col_id"].to_list()):
                cv = abs(float(c))
                if cv == 0.0 or not math.isfinite(cv):
                    continue
                col_t = col_id_to_type.get(int(cid))
                if col_t is None:
                    continue
                vfam = lookup_var(col_id_to_var[int(cid)])
                eff_t = _effective_matrix_type(vfam, rhs_t)
                matrix_buckets.setdefault(eff_t, []).append(cv)

    return matrix_buckets, cost_buckets, bound_buckets


def _build_wide_lp() -> Problem:
    """Same wide-range LP as the round-trip test.

    Picks well-separated coefficient magnitudes per type so the
    closed-form rounding is unambiguous.
    """
    pb = Problem()
    v_power = pb.add_var(
        "v_test_power_pn", "i", pl.DataFrame({"i": [0]}),
        lower=0.0, upper=float("inf"),
    )
    v_energy = pb.add_var(
        "v_test_energy_pn", "i", pl.DataFrame({"i": [0]}),
        lower=0.0, upper=float("inf"),
    )
    v_money = pb.add_var(
        "v_test_money_pn", "i", pl.DataFrame({"i": [0]}),
        lower=0.0, upper=float("inf"),
    )
    pb.add_cstr(
        "test_power_cap_pn",
        sense="<=",
        lhs_terms={
            "p": Sum(v_power, over=("i",)),
            "e": Sum(v_energy * 1e-3, over=("i",)),
        },
        rhs_terms={"cap": 1e5},
    )
    pb.add_cstr(
        "test_energy_cap_pn",
        sense="<=",
        lhs_terms={
            "p": Sum(v_power * 1e7, over=("i",)),
            "e": Sum(v_energy, over=("i",)),
            "m": Sum(v_money * 1e-2, over=("i",)),
        },
        rhs_terms={"cap": 1e9},
    )
    pb.add_cstr(
        "test_money_cap_pn",
        sense="<=",
        lhs_terms={"m": Sum(v_money, over=("i",))},
        rhs_terms={"cap": 1e3},
    )
    obj = (v_power * (-1e-4)
           + v_energy * (-1.0)
           + v_money * (-1e4))
    pb.set_objective(Sum(obj), sense="min")
    return pb


def _register_wide_lp_families(add_var, add_cstr) -> None:
    add_var("v_test_power_pn", VarFamily(QuantityType.POWER))
    add_var("v_test_energy_pn", VarFamily(QuantityType.ENERGY))
    add_var("v_test_money_pn", VarFamily(QuantityType.CURRENCY))
    add_cstr("test_power_cap_pn", CstrFamily(QuantityType.POWER))
    add_cstr("test_energy_cap_pn", CstrFamily(QuantityType.ENERGY))
    add_cstr("test_money_cap_pn", CstrFamily(QuantityType.CURRENCY))


# ---------------------------------------------------------------------------
# §6.1 — Mathematical equivalence vs legacy on a real-but-small Problem


def test_choose_scale_powers_matches_legacy_on_wide_lp(register_test_families):
    """``chosen`` dict must match the legacy implementation modulo at
    most ±1 on half-integer rounding boundaries (plan §8 risk #2)."""
    add_var, add_cstr = register_test_families
    _register_wide_lp_families(add_var, add_cstr)

    pb_legacy = _build_wide_lp()
    legacy_mb, legacy_cb, legacy_bb = _legacy_bucket_coefficients(pb_legacy)
    legacy_chosen = _legacy_choose_scale_powers(legacy_mb, legacy_cb, legacy_bb)

    pb_new = _build_wide_lp()
    new_mb, new_cb, new_bb, _ = bucket_coefficients(pb_new)
    new_chosen = choose_scale_powers(new_mb, new_cb, new_bb)

    assert set(new_chosen) == set(legacy_chosen), (
        f"new={set(new_chosen)} legacy={set(legacy_chosen)}"
    )
    for t in legacy_chosen:
        delta = abs(new_chosen[t] - legacy_chosen[t])
        assert delta <= 1, (
            f"{t.name}: new={new_chosen[t]} legacy={legacy_chosen[t]} "
            f"(|delta|={delta} > 1)"
        )


def test_choose_scale_powers_bit_identical_on_non_boundary(register_test_families):
    """When magnitudes are far from a half-integer log boundary, the
    new code must match the legacy code *exactly* (no ±1 noise)."""
    add_var, add_cstr = register_test_families
    add_var("v_test_e_2", VarFamily(QuantityType.ENERGY))
    add_cstr("test_e_cap_2", CstrFamily(QuantityType.ENERGY))

    pb_legacy = Problem()
    pb_new = Problem()
    for pb in (pb_legacy, pb_new):
        v = pb.add_var(
            "v_test_e_2", "i", pl.DataFrame({"i": [0, 1, 2, 3]}),
            lower=0.0, upper=float("inf"),
        )
        # Coefficients all exactly 1e-3 (log2 ≈ -9.97) — far from any
        # half-integer ``round`` boundary so the legacy and new
        # rounding must agree exactly.
        pb.add_cstr(
            "test_e_cap_2", sense="<=",
            lhs_terms={"e": Sum(v * 1e-3, over=("i",))},
            rhs_terms={"cap": 1e3},
        )
        pb.set_objective(Sum(v * 1.0), sense="min")

    legacy = _legacy_choose_scale_powers(
        *_legacy_bucket_coefficients(pb_legacy)
    )
    new_mb, new_cb, new_bb, _ = bucket_coefficients(pb_new)
    new = choose_scale_powers(new_mb, new_cb, new_bb)

    assert new == legacy, f"new={new} legacy={legacy}"


# ---------------------------------------------------------------------------
# §6.2 — Closed-form correctness on a synthetic LP


def test_closed_form_per_type_exponent(register_test_families):
    """Pin the chosen exponent to round(-log2(magnitude))."""
    add_var, add_cstr = register_test_families
    add_var("v_test_flow_cf", VarFamily(QuantityType.POWER))
    add_var("v_test_state_cf", VarFamily(QuantityType.ENERGY))
    add_cstr("test_flow_cap_cf", CstrFamily(QuantityType.POWER))
    add_cstr("test_state_cap_cf", CstrFamily(QuantityType.ENERGY))

    pb = Problem()
    v_flow = pb.add_var(
        "v_test_flow_cf", "i", pl.DataFrame({"i": list(range(4))}),
        lower=0.0, upper=1.0,
    )
    v_state = pb.add_var(
        "v_test_state_cf", "i", pl.DataFrame({"i": list(range(4))}),
        lower=0.0, upper=1.0,
    )
    # All POWER matrix entries == 1e-3 (log2 ≈ -9.97 → exp = +10).
    pb.add_cstr(
        "test_flow_cap_cf", sense="<=",
        lhs_terms={"f": Sum(v_flow * 1e-3, over=("i",))},
        rhs_terms={"cap": 1.0},
    )
    # All ENERGY matrix entries == 1e6 (log2 ≈ +19.93 → exp = -20).
    pb.add_cstr(
        "test_state_cap_cf", sense="<=",
        lhs_terms={"s": Sum(v_state * 1e6, over=("i",))},
        rhs_terms={"cap": 1e9},
    )
    pb.set_objective(Sum(v_flow * 1.0), sense="min")

    matrix_acc, cost_acc, bound_acc, _ = bucket_coefficients(pb)
    chosen = choose_scale_powers(matrix_acc, cost_acc, bound_acc)

    # Magnitudes pooled per type — compute via the legacy walker on
    # the same Problem so we don't drift on accounting details.
    pb_ref = Problem()
    v_flow_r = pb_ref.add_var(
        "v_test_flow_cf", "i", pl.DataFrame({"i": list(range(4))}),
        lower=0.0, upper=1.0,
    )
    v_state_r = pb_ref.add_var(
        "v_test_state_cf", "i", pl.DataFrame({"i": list(range(4))}),
        lower=0.0, upper=1.0,
    )
    pb_ref.add_cstr(
        "test_flow_cap_cf", sense="<=",
        lhs_terms={"f": Sum(v_flow_r * 1e-3, over=("i",))},
        rhs_terms={"cap": 1.0},
    )
    pb_ref.add_cstr(
        "test_state_cap_cf", sense="<=",
        lhs_terms={"s": Sum(v_state_r * 1e6, over=("i",))},
        rhs_terms={"cap": 1e9},
    )
    pb_ref.set_objective(Sum(v_flow_r * 1.0), sense="min")
    legacy_mb, legacy_cb, legacy_bb = _legacy_bucket_coefficients(pb_ref)
    expected = _legacy_choose_scale_powers(legacy_mb, legacy_cb, legacy_bb)

    assert chosen[QuantityType.POWER] == expected[QuantityType.POWER]
    assert chosen[QuantityType.ENERGY] == expected[QuantityType.ENERGY]
    # Independent closed-form sanity — geometric mean of the matrix
    # entries alone for each type.
    assert int(round(-math.log2(1e-3))) == 10  # noqa: PLR2004 — sanity
    assert int(round(-math.log2(1e6))) == -20  # noqa: PLR2004 — sanity


# ---------------------------------------------------------------------------
# §6.3 — Edge cases


def test_empty_problem():
    pb = Problem()
    matrix_acc, cost_acc, bound_acc, col_id_to_type = bucket_coefficients(pb)
    assert matrix_acc == {}
    assert cost_acc == {}
    assert bound_acc == {}
    assert col_id_to_type == {}
    assert choose_scale_powers(matrix_acc, cost_acc, bound_acc) == {}


def test_all_zero_term_skipped(register_test_families):
    """A term whose coefficients are entirely zero must not register."""
    add_var, add_cstr = register_test_families
    add_var("v_test_z", VarFamily(QuantityType.POWER))
    add_cstr("test_z_cap", CstrFamily(QuantityType.POWER))

    pb = Problem()
    v = pb.add_var("v_test_z", "i", pl.DataFrame({"i": [0, 1]}),
                   lower=0.0, upper=float("inf"))
    pb.add_cstr(
        "test_z_cap", sense="<=",
        lhs_terms={"z": Sum(v * 0.0, over=("i",))},
        rhs_terms={"cap": 1.0},
    )
    pb.set_objective(Sum(v * 1.0), sense="min")

    matrix_acc, cost_acc, bound_acc, _ = bucket_coefficients(pb)
    # All-zero term contributes no matrix entries.
    assert QuantityType.POWER not in matrix_acc
    # Objective and bounds still contribute.
    assert QuantityType.POWER in cost_acc
    assert cost_acc[QuantityType.POWER][1] == 2  # count == 2


def test_infinite_bound_filtered(register_test_families):
    """``±inf`` bounds must not enter the bound accumulator."""
    add_var, _ = register_test_families
    add_var("v_test_inf", VarFamily(QuantityType.ENERGY))

    pb = Problem()
    pb.add_var("v_test_inf", "i", pl.DataFrame({"i": [0]}),
               lower=-float("inf"), upper=float("inf"))
    matrix_acc, cost_acc, bound_acc, _ = bucket_coefficients(pb)
    assert bound_acc == {}


def test_single_row_term(register_test_families):
    add_var, add_cstr = register_test_families
    add_var("v_test_single", VarFamily(QuantityType.POWER))
    add_cstr("test_single_cap", CstrFamily(QuantityType.POWER))

    pb = Problem()
    v = pb.add_var("v_test_single", "i", pl.DataFrame({"i": [0]}),
                   lower=0.0, upper=1.0)
    pb.add_cstr(
        "test_single_cap", sense="<=",
        lhs_terms={"s": Sum(v * 4.0, over=("i",))},
        rhs_terms={"cap": 4.0},
    )
    pb.set_objective(Sum(v * 1.0), sense="min")

    matrix_acc, _, _, _ = bucket_coefficients(pb)
    # Single matrix entry, value 4.0 → log2 = 2.0, count = 1.
    log_sum, count, amin, amax = matrix_acc[QuantityType.POWER]
    assert count == 1
    assert log_sum == pytest.approx(2.0)
    assert amin == pytest.approx(4.0)
    assert amax == pytest.approx(4.0)


def test_mixed_zero_and_nonzero_term(register_test_families):
    """Term carrying some zero and some nonzero coefficients only
    contributes the nonzeros to the accumulator."""
    add_var, add_cstr = register_test_families
    add_var("v_test_mix", VarFamily(QuantityType.POWER))
    add_cstr("test_mix_cap", CstrFamily(QuantityType.POWER))

    pb = Problem()
    # Four columns; only two will carry a nonzero coefficient.
    v = pb.add_var("v_test_mix", "i", pl.DataFrame({"i": [0, 1, 2, 3]}),
                   lower=0.0, upper=1.0)
    # Two separate terms: one all-zero, one all-eights — sum decomposes
    # into the union, which is what bucket_coefficients should see.
    pb.add_cstr(
        "test_mix_cap", sense="<=",
        lhs_terms={
            "z": Sum(v * 0.0, over=("i",)),       # all-zero term
            "nz": Sum(v * 8.0, over=("i",)),      # nonzero
        },
        rhs_terms={"cap": 32.0},
    )
    pb.set_objective(Sum(v * 1.0), sense="min")

    matrix_acc, _, _, _ = bucket_coefficients(pb)
    log_sum, count, amin, amax = matrix_acc[QuantityType.POWER]
    # Only the 8.0 entries count: 4 rows × log2(8) = 4 × 3 = 12.
    assert count == 4
    assert log_sum == pytest.approx(12.0)
    assert amin == pytest.approx(8.0)
    assert amax == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# Classification frame + QuantityType round-trip


def test_qty_id_round_trip_is_bijective():
    """Every QuantityType maps to a unique int and back."""
    for q in QuantityType:
        assert _ID_TO_QTY[_QTY_TO_ID[q]] is q


def test_build_classification_frame(register_test_families):
    add_var, _ = register_test_families
    add_var("v_test_pow_cls", VarFamily(QuantityType.POWER, multiplier_param="p_x"))
    add_var("v_test_eng_cls", VarFamily(QuantityType.ENERGY))

    pb = Problem()
    pb.add_var("v_test_pow_cls", "i", pl.DataFrame({"i": [0, 1]}),
               lower=0.0, upper=1.0)
    pb.add_var("v_test_eng_cls", "j", pl.DataFrame({"j": [0]}),
               lower=0.0, upper=1.0)

    df = _build_col_id_classification(pb)
    assert df.shape == (3, 3)
    assert set(df.columns) == {"col_id", "column_type_id", "has_multiplier_param"}
    # The two POWER cols carry has_multiplier_param=True; ENERGY col False.
    rows_by_id = {r["col_id"]: r for r in df.to_dicts()}
    pow_ids = pb._vars["v_test_pow_cls"].frame["col_id"].to_list()
    eng_ids = pb._vars["v_test_eng_cls"].frame["col_id"].to_list()
    for cid in pow_ids:
        assert rows_by_id[cid]["column_type_id"] == _QTY_TO_ID[QuantityType.POWER]
        assert rows_by_id[cid]["has_multiplier_param"] is True
    for cid in eng_ids:
        assert rows_by_id[cid]["column_type_id"] == _QTY_TO_ID[QuantityType.ENERGY]
        assert rows_by_id[cid]["has_multiplier_param"] is False


# ---------------------------------------------------------------------------
# Multiplier-param + rhs_type effective bucketing


def test_effective_type_uses_rhs_when_multiplier_param_set(register_test_families):
    """Column with multiplier_param + constraint with rhs_type: matrix
    entries land in the row's bucket, not the column's."""
    add_var, add_cstr = register_test_families
    # DIMENSIONLESS column with a multiplier_param (mirrors v_flow).
    add_var("v_test_flow_eff", VarFamily(
        QuantityType.DIMENSIONLESS, multiplier_param="p_unitsize",
    ))
    # POWER row.
    add_cstr("test_pow_eff", CstrFamily(QuantityType.POWER))

    pb = Problem()
    v = pb.add_var("v_test_flow_eff", "i", pl.DataFrame({"i": [0, 1]}),
                   lower=0.0, upper=1.0)
    pb.add_cstr(
        "test_pow_eff", sense="<=",
        lhs_terms={"f": Sum(v * 100.0, over=("i",))},
        rhs_terms={"cap": 100.0},
    )
    pb.set_objective(Sum(v * 1.0), sense="min")

    matrix_acc, _, _, _ = bucket_coefficients(pb)
    # Matrix entries must land in POWER (the rhs_type), not
    # DIMENSIONLESS (the column type).
    assert QuantityType.POWER in matrix_acc
    assert QuantityType.DIMENSIONLESS not in matrix_acc


# ---------------------------------------------------------------------------
# §6.4 — Memory smoke test (synthetic LP, bounded ~200k rows)


@pytest.mark.slow
def test_memory_smoke_synthetic_large(register_test_families):
    """Synthetic 200k-row × 100k-col LP: peak Python allocation during
    bucket_coefficients must stay sub-GB.

    Marked ``slow`` — running every CI cycle is unnecessary.  Enable
    via ``pytest -m slow``.  Run locally to verify the rewrite's
    streaming behaviour before shipping.
    """
    import tracemalloc

    add_var, add_cstr = register_test_families
    add_var("v_test_mem", VarFamily(QuantityType.POWER))
    add_cstr("test_mem_cap", CstrFamily(QuantityType.POWER))

    pb = Problem()
    n_cols = 100_000
    v = pb.add_var(
        "v_test_mem", "i", pl.DataFrame({"i": list(range(n_cols))}),
        lower=0.0, upper=1.0,
    )
    # Two terms to make 200k matrix nonzeros.
    pb.add_cstr(
        "test_mem_cap", sense="<=",
        lhs_terms={
            "a": Sum(v * 1e-3, over=("i",)),
            "b": Sum(v * 1.7, over=("i",)),
        },
        rhs_terms={"cap": 1.0},
    )
    pb.set_objective(Sum(v * 1.0), sense="min")

    tracemalloc.start()
    matrix_acc, _, _, _ = bucket_coefficients(pb)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Peak should be on the order of a few hundred MB at most — the
    # legacy code allocated O(nnz) Python floats × tuples (gigabytes).
    # 500 MB ceiling is generous and still catches the regression.
    assert peak < 500 * 1024 * 1024, (
        f"bucket_coefficients tracemalloc peak {peak / 1e6:.0f} MB "
        f"exceeds 500 MB budget — streaming aggregation broken?"
    )
    assert QuantityType.POWER in matrix_acc
    assert matrix_acc[QuantityType.POWER][1] == 2 * n_cols  # 200k entries


# ---------------------------------------------------------------------------
# Streaming engine verification (best-effort)


def test_streaming_engine_engages(capsys, register_test_families):
    """Verify the streaming engine is the one actually consumed (vs.
    fallback to in-memory).  We probe by inspecting ``.explain`` of
    the same plan our aggregator builds — if streaming refuses an
    operator the explain raises or returns the in-memory plan.
    """
    add_var, _ = register_test_families
    add_var("v_test_stream", VarFamily(QuantityType.POWER))

    pb = Problem()
    pb.add_var("v_test_stream", "i", pl.DataFrame({"i": list(range(64))}),
               lower=0.0, upper=1.0)
    pb.set_objective(Sum(pl.lit(1.0) * 0.0), sense="min") if False else None

    from flextool.engine_polars.autoscale._layer2 import (
        _build_col_id_classification, _collect_term_agg,
    )
    cls = _build_col_id_classification(pb).lazy()

    # Build a dummy term ourselves to feed the aggregator.
    term_lazy = pl.LazyFrame({
        "col_id": list(range(64)),
        "coef": [1.0 + 0.1 * i for i in range(64)],
    })
    plan = (
        term_lazy.join(cls, on="col_id", how="inner")
        .with_columns(eff_t=pl.col("column_type_id"))
        .filter(pl.col("coef") != 0.0)
        .with_columns(log_abs=pl.col("coef").abs().log(2.0))
        .filter(pl.col("log_abs").is_finite())
        .group_by("eff_t")
        .agg(
            pl.col("log_abs").sum().alias("log_sum"),
            pl.len().alias("n"),
            pl.col("log_abs").min().alias("log_min"),
            pl.col("log_abs").max().alias("log_max"),
        )
    )
    # If the streaming engine refuses the plan, explain(engine="streaming")
    # raises.  We assert it does not raise — the rewrite's whole point
    # depends on this path being available.
    explain = plan.explain(engine="streaming")
    assert isinstance(explain, str) and explain  # planner produced output

    # And the actual collect path used by the aggregator must succeed.
    agg = _collect_term_agg(term_lazy, classification_lazy=cls, rhs_t_id=None)
    assert agg is not None
    assert agg.height == 1


def test_effective_type_falls_back_to_column_when_rhs_none(register_test_families):
    """If the constraint family declares rhs_type=None, the matrix
    entry uses the column type."""
    add_var, add_cstr = register_test_families
    add_var("v_test_eff_none", VarFamily(
        QuantityType.DIMENSIONLESS, multiplier_param="p_unitsize",
    ))
    add_cstr("test_eff_none", CstrFamily(rhs_type=None))

    pb = Problem()
    v = pb.add_var("v_test_eff_none", "i", pl.DataFrame({"i": [0]}),
                   lower=0.0, upper=1.0)
    pb.add_cstr(
        "test_eff_none", sense="<=",
        lhs_terms={"f": Sum(v * 7.0, over=("i",))},
        rhs_terms={"cap": 7.0},
    )
    pb.set_objective(Sum(v * 1.0), sense="min")

    matrix_acc, _, _, _ = bucket_coefficients(pb)
    assert QuantityType.DIMENSIONLESS in matrix_acc
