"""Slice C — recourse validation guards.

Design: ``specs/sliceC_optin_guards_design.md`` §2/§4 (G1, G2a-c, G3).
The three guards keep the ``stochastic_invest_method`` opt-in inert until
Slice D:

* Guard 1 — flag hard-reject (``stochastic_invest_method != none``) at the
  top of ``run()`` + a fail-fast pass in ``_validate_model_solve``.
* Guard 2 — Benders x genuine stochastics, flag-independent
  (``_is_genuinely_stochastic`` = ``period_branch_full`` d!=b tokens
  intersected with ``period_in_use_set``).
* Guard 3 — handoff synthetic-name assertion in
  ``build_handoff_from_solution``.

G1/G2 are solver-free (hand-built frames / stub state).  G3 solves one
small fixture once and drives the real handoff builder.
"""
from __future__ import annotations

import dataclasses
import re
import shutil

import polars as pl
import pytest

from flextool.engine_polars._solve_config import SolveConfig
from flextool.engine_polars._solve_state import FlexToolConfigError
from flextool.engine_polars._stochastic_detect import (
    is_genuinely_stochastic,
    synthetic_branch_tokens,
)


# ---------------------------------------------------------------------------
# Frame builders — mirror the runtime shapes verified empirically on
# ``stoch_two_period`` (period_branch_full is Enum-typed with DISJOINT
# d/b category sets; period_in_use_set is col ``d``).
# ---------------------------------------------------------------------------


def _pbf(pairs, *, enum=False) -> pl.DataFrame:
    df = pl.DataFrame(
        {"d": [p[0] for p in pairs], "b": [p[1] for p in pairs]}
    )
    if enum:
        d_cats = sorted({p[0] for p in pairs})
        b_cats = sorted({p[1] for p in pairs if p[1] is not None})
        df = df.with_columns(
            pl.col("d").cast(pl.Enum(d_cats)),
            pl.col("b").cast(pl.Enum(b_cats)),
        )
    return df


def _piu(periods, *, enum=False) -> pl.DataFrame:
    df = pl.DataFrame({"d": list(periods)})
    if enum:
        df = df.with_columns(pl.col("d").cast(pl.Enum(sorted(set(periods)))))
    return df


# ===========================================================================
# Guard 1 — flag hard-reject
# ===========================================================================


def _solve_config_with_flag(model_solve, flag_map) -> SolveConfig:
    """A bare SolveConfig carrying only the two fields the guard reads —
    ``model_solve`` and ``stochastic_invest_method`` (+ its resolver)."""
    sc = object.__new__(SolveConfig)
    sc.model_solve = model_solve
    sc.stochastic_invest_method = flag_map
    return sc


def test_g1_resolver_normalisation():
    sc = _solve_config_with_flag(
        {"m": ["s"]},
        {"s": "recourse", "loud": "RECOURSE", "blank": "", "junk": "xyz"},
    )
    assert sc.stochastic_invest_method_for("s") == "recourse"
    assert sc.stochastic_invest_method_for("loud") == "recourse"  # case-fold
    assert sc.stochastic_invest_method_for("blank") == "none"
    assert sc.stochastic_invest_method_for("junk") == "none"
    assert sc.stochastic_invest_method_for("absent") == "none"  # default


def test_g1_validate_model_solve_accepts_recourse():
    """Slice D: Guard 1 is LIFTED for 'recourse' — the fail-fast pass no
    longer raises on a recourse solve and returns the solve list."""
    from flextool.engine_polars._orchestration import _validate_model_solve

    sc = _solve_config_with_flag({"m": ["s_ok", "s_rec"]},
                                  {"s_rec": "recourse"})
    state = dataclasses.make_dataclass("S", ["solve"])(solve=sc)
    assert _validate_model_solve(state) == ["s_ok", "s_rec"]


def test_g1_raw_value_typo_rejected():
    """Slice D §12/F4: the guard inspects the RAW authored value (the
    resolver would silently collapse a typo to 'none'), so an unrecognised
    authored value raises — while 'recourse' / blank / absent do not."""
    from flextool.engine_polars._orchestration import (
        _stochastic_invest_raw_guard,
        _validate_model_solve,
    )

    for bad in ("recuorse", "expected_value"):
        sc = _solve_config_with_flag({"m": ["s"]}, {"s": bad})
        with pytest.raises(FlexToolConfigError) as exc:
            _stochastic_invest_raw_guard(sc, "s")
        assert bad in str(exc.value)
        state = dataclasses.make_dataclass("S", ["solve"])(solve=sc)
        with pytest.raises(FlexToolConfigError):
            _validate_model_solve(state)

    # Accepted / tolerated raw values → no raise.
    for ok in ("recourse", "", "none", "RECOURSE"):
        sc = _solve_config_with_flag({"m": ["s"]}, {"s": ok})
        _stochastic_invest_raw_guard(sc, "s")  # no raise
    # Absent → no raise.
    _stochastic_invest_raw_guard(
        _solve_config_with_flag({"m": ["s"]}, {}), "s")


def test_g1_validate_model_solve_passes_flag_none():
    from flextool.engine_polars._orchestration import _validate_model_solve

    sc = _solve_config_with_flag({"m": ["s1", "s2"]}, {})  # all default none
    state = dataclasses.make_dataclass("S", ["solve"])(solve=sc)
    assert _validate_model_solve(state) == ["s1", "s2"]


def test_g1_rolling_suffix_strip_composition():
    """The ``run()``-site guard strips the ``_roll_N`` suffix before the
    resolver lookup, so a rolling child of a flag-on solve is rejected."""
    sc = _solve_config_with_flag({"m": ["my_solve"]},
                                  {"my_solve": "recourse"})
    base = re.sub(r"_roll_\d+$", "", "my_solve_roll_2")
    assert base == "my_solve"
    assert sc.stochastic_invest_method_for(base) == "recourse"


# ===========================================================================
# Guard 2 — Benders x genuine stochastics
# ===========================================================================


class _StubData:
    def __init__(self, pbf, piu):
        self.period_branch_full = pbf
        self.period_in_use_set = piu


def test_g2a_genuine_stochastic_detected():
    """The stoch_two_period shape: (p2035,p2035_low) is a synthetic branch
    with LP vars (p2035_low in PIU) — detector True."""
    pbf = _pbf([("p2035", "p2035"), ("p2035", "p2035_rlz"),
                ("p2035", "p2035_low")], enum=True)
    piu = _piu(["p2035", "p2035_low"], enum=True)
    assert synthetic_branch_tokens(pbf) == {"p2035_rlz", "p2035_low"}
    assert is_genuinely_stochastic(_StubData(pbf, piu)) is True


def test_g2a_solve_benders_raises_on_stochastic():
    """The defensive site at the top of ``solve_benders`` fires before any
    region work."""
    from flextool.engine_polars._benders import solve_benders

    pbf = _pbf([("p2035", "p2035"), ("p2035", "p2035_low")], enum=True)
    piu = _piu(["p2035", "p2035_low"], enum=True)
    with pytest.raises(FlexToolConfigError) as exc:
        solve_benders(_StubData(pbf, piu), ["region_a", "region_b"])
    assert "stochastic" in str(exc.value).lower()


def test_g2b_deterministic_not_detected():
    """Deterministic shape: b is NULL (never a synthetic token) — False,
    and the Enum/null comparison does not crash."""
    pbf = _pbf([("p2035", None), ("p2040", None)], enum=True)
    piu = _piu(["p2035", "p2040"], enum=True)
    assert synthetic_branch_tokens(pbf) == set()
    assert is_genuinely_stochastic(_StubData(pbf, piu)) is False


def test_g2b_self_rows_only_not_detected():
    pbf = _pbf([("p2035", "p2035"), ("p2040", "p2040")], enum=True)
    piu = _piu(["p2035", "p2040"], enum=True)
    assert synthetic_branch_tokens(pbf) == set()
    assert is_genuinely_stochastic(_StubData(pbf, piu)) is False


def test_g2c_rolling_realized_only_bookkeeping_not_detected():
    """The false-positive pin (plan §5.1): the rolling metadata row
    (period1, period1_realized) has d!=b but period1_realized is NOT in
    period_in_use — the PIU intersection excludes it, so the detector
    returns False.  The BROAD token set still contains it (Guard 3's
    concern), which the shared idiom exposes."""
    pbf = _pbf([("period1", "period1"), ("period1", "period1_realized")],
               enum=True)
    piu = _piu(["period1"], enum=True)   # period1_realized absent (metadata)
    # Broad set (Guard 3) includes the metadata token deliberately.
    assert synthetic_branch_tokens(pbf) == {"period1_realized"}
    # PIU-intersected detector (Guard 2) does NOT fire.
    assert is_genuinely_stochastic(_StubData(pbf, piu)) is False


def test_g2_none_frames_degrade_false():
    assert is_genuinely_stochastic(_StubData(None, None)) is False
    # PIU None -> degrade to today's behaviour (load-bearing only when PIU
    # is populated).
    pbf = _pbf([("p2035", "p2035_low")], enum=True)
    assert is_genuinely_stochastic(_StubData(pbf, None)) is False
    assert synthetic_branch_tokens(None) == set()


# ===========================================================================
# Guard 3 — handoff synthetic-name assertion (solver-based integration)
# ===========================================================================

SCENARIO = "coal_ladder_cumulative"
SOLVE_NAME = "y2020_2day_dispatch"
REALIZE_CSV = "solve_data/realized_invest_periods_of_current_solve.csv"


@pytest.fixture(scope="module")
def _solved(scenario_workdir):
    from polar_high import Problem
    from flextool.engine_polars import build_flextool, load_flextool
    from flextool.engine_polars.input import build_handoff_from_solution

    work = scenario_workdir(SCENARIO)
    data = load_flextool(work)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal
    return sol, work, data, build_handoff_from_solution


@pytest.mark.solver
def test_g3_clean_real_names_no_raise(_solved):
    """The happy path — a deterministic (real-named) flex_data commits no
    synthetic period, so the guard is inert."""
    sol, work, data, build_handoff = _solved
    handoff = build_handoff(sol, work, SOLVE_NAME, flex_data=data)
    assert handoff is not None


@pytest.mark.solver
def test_g3_config_level_synthetic_in_realize_invest_raises(
        _solved, tmp_path):
    """realized_invest_periods listing a synthetic branch token → config
    raise (case a)."""
    sol, work, data, build_handoff = _solved
    twork = tmp_path / "work_g3a"
    shutil.copytree(work, twork)
    csv = twork / REALIZE_CSV
    csv.parent.mkdir(parents=True, exist_ok=True)
    csv.write_text("period\np2020_low\n")
    # flex_data whose period_branch_full marks p2020_low as a synthetic
    # fan member of the real anchor p2020.
    bad = dataclasses.replace(
        data,
        period_branch_full=_pbf([("p2020", "p2020"),
                                 ("p2020", "p2020_low")], enum=True),
    )
    with pytest.raises(FlexToolConfigError) as exc:
        build_handoff(sol, twork, SOLVE_NAME, flex_data=bad)
    assert "p2020_low" in str(exc.value)


@pytest.mark.solver
def test_g3_none_frame_degrades_no_raise(_solved, tmp_path):
    """period_branch_full unavailable → the guard is a no-op even when the
    realize CSV carries a synthetic token (the documented None-degrade;
    Guard 2 covers that path's stochastic case)."""
    sol, work, data, build_handoff = _solved
    twork = tmp_path / "work_g3n"
    shutil.copytree(work, twork)
    csv = twork / REALIZE_CSV
    csv.parent.mkdir(parents=True, exist_ok=True)
    csv.write_text("period\np2020_low\n")
    degraded = dataclasses.replace(data, period_branch_full=None)
    # No raise — synthetic_branch_tokens(None) is empty, guard skipped.
    handoff = build_handoff(sol, twork, SOLVE_NAME, flex_data=degraded)
    assert handoff is not None
