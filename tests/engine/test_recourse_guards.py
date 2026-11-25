"""The guards around investments that branch per forecast (``invest_solve_branching_method``):

* Guard 1 — ``invest_solve_branching_method``: an unrecognised raw value is
  refused (``_invest_branching_raw_guard``, at the top of ``run()`` and in
  ``_validate_model_solve``), and so is the setting where it does not apply
  (``SolveConfig.check_invest_branching``: a rolling or dispatch solve, no
  ``stochastic_branches``, a realized period whose investments are per
  branch).
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

import polars as pl
import pytest

from flextool.engine.input._solve_config import SolveConfig
from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.preprocess._stochastic_detect import (
    is_genuinely_stochastic,
    synthetic_branch_tokens,
)
from tests._support.load import clone_provider, load_scenario


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
# Guard 1 — invest_solve_branching_method: the raw value and where it may be set
# ===========================================================================


def _solve_config_with_flag(model_solve, flag_map, **fields) -> SolveConfig:
    """A bare SolveConfig carrying ``model_solve`` and ``invest_solve_branching_method`` (+ the fields its
    resolver and :meth:`SolveConfig.check_invest_branching` read, empty unless given in *fields*)."""
    sc = object.__new__(SolveConfig)
    sc.model_solve = model_solve
    sc.invest_solve_branching_method = flag_map
    # ``_validate_model_solve`` also runs the dispatch_realization_method raw guard, which reads this dict; keep it
    # empty so only the branching guard is under test here.
    sc.dispatch_realization_method = {}
    for name in ("contains_solves", "map2d_duplicate_of", "deterministic_branches", "solve_modes",
                 "stochastic_branches", "timesets_used_by_solves", "invest_periods", "realized_invest_periods",
                 "realized_periods"):
        setattr(sc, name, fields.get(name, {}))
    return sc


def test_g1_resolver_normalisation():
    sc = _solve_config_with_flag(
        {"m": ["s"]},
        {"s": "investments_after_reveal_period", "loud": "INVESTMENTS_AT_REVEAL_PERIOD", "blank": "",
         "junk": "xyz"},
    )
    assert sc.invest_branching_method_for("s") == "investments_after_reveal_period"
    assert sc.invest_branching_method_for("loud") == "investments_at_reveal_period"  # case-fold
    assert sc.invest_branching_method_for("blank") == "operations_only"
    assert sc.invest_branching_method_for("junk") == "operations_only"
    assert sc.invest_branching_method_for("absent") == "operations_only"  # default
    assert sc.recourse_invest("s") and sc.recourse_invest("loud")
    assert not sc.recourse_invest("blank") and not sc.recourse_invest("absent")


def test_g1_validate_model_solve_accepts_the_members():
    from flextool.engine.solve._orchestration import _validate_model_solve

    sc = _solve_config_with_flag({"m": ["s_ok", "s_after", "s_at"]},
                                  {"s_after": "investments_after_reveal_period",
                                   "s_at": "investments_at_reveal_period"})
    state = dataclasses.make_dataclass("S", ["solve"])(solve=sc)
    assert _validate_model_solve(state) == ["s_ok", "s_after", "s_at"]


def test_g1_raw_value_typo_rejected():
    """The guard inspects the RAW authored value (the resolver reads a typo as operations_only), so an unrecognised
    authored value raises, naming the value, while the members, blank and absent do not.  The old values of
    ``stochastic_invest_method`` are unknown here: every database is migrated to v84 before it runs."""
    from flextool.engine.solve._orchestration import (
        _invest_branching_raw_guard,
        _validate_model_solve,
    )

    for bad in ("investments_after_reveal", "recourse", "none"):
        sc = _solve_config_with_flag({"m": ["s"]}, {"s": bad})
        with pytest.raises(FlexToolConfigError) as exc:
            _invest_branching_raw_guard(sc, "s")
        assert bad in str(exc.value)
        state = dataclasses.make_dataclass("S", ["solve"])(solve=sc)
        with pytest.raises(FlexToolConfigError):
            _validate_model_solve(state)

    for ok in ("operations_only", "", "investments_after_reveal_period", "INVESTMENTS_AT_REVEAL_PERIOD"):
        sc = _solve_config_with_flag({"m": ["s"]}, {"s": ok})
        _invest_branching_raw_guard(sc, "s")  # no raise
    _invest_branching_raw_guard(_solve_config_with_flag({"m": ["s"]}, {}), "s")


def test_g1_validate_model_solve_passes_flag_absent():
    from flextool.engine.solve._orchestration import _validate_model_solve

    sc = _solve_config_with_flag({"m": ["s1", "s2"]}, {})
    state = dataclasses.make_dataclass("S", ["solve"])(solve=sc)
    assert _validate_model_solve(state) == ["s1", "s2"]


# Solve ``v`` over p1, p2 (in that order), forecasts A / B from p2 (the reveal period), an investment solve unless
# a test says otherwise.  Rows of stochastic_branches: (period, branch, analysis_time, realized, weight).
_FAN = [("p2", "A", "t1", "no", 1.0), ("p2", "B", "t1", "no", 1.0)]
_ORDER = [("p1", "ts"), ("p2", "ts")]


def _branching(method, **fields) -> SolveConfig:
    base = {"solve_modes": {"v": "single_solve"}, "stochastic_branches": {"v": list(_FAN)},
            "timesets_used_by_solves": {"v": list(_ORDER)}, "invest_periods": {"v": [("v", "p1"), ("v", "p2")]}}
    base.update(fields)
    return _solve_config_with_flag({"m": ["v"]}, {"v": method}, **base)


def test_g1_rolling_solve_with_the_setting_is_refused():
    """T9: a rolling solve is never an investment solve (its rolls build no investment), so the setting is refused
    there, naming the solve and pointing to dispatch_realization_method."""
    sc = _branching("investments_after_reveal_period", solve_modes={"v": "rolling_window"})
    with pytest.raises(FlexToolConfigError) as exc:
        sc.check_invest_branching()
    text = str(exc.value)
    assert "Solve 'v' sets invest_solve_branching_method investments_after_reveal_period" in text
    assert "it is a rolling solve" in text and "dispatch_realization_method" in text


def test_g1_dispatch_solve_with_the_setting_is_refused():
    """T9: a single solve without invest_periods / realized_invest_periods is a dispatch solve: refused too."""
    sc = _branching("operations_only", invest_periods={})
    with pytest.raises(FlexToolConfigError) as exc:
        sc.check_invest_branching()
    assert "it has no invest_periods or realized_invest_periods" in str(exc.value)


def test_g1_setting_without_stochastic_branches_is_refused():
    """T10: an investment solve without stochastic_branches has nothing to branch: refused, naming the solve."""
    sc = _branching("investments_at_reveal_period", stochastic_branches={})
    with pytest.raises(FlexToolConfigError) as exc:
        sc.check_invest_branching()
    text = str(exc.value)
    assert "Solve 'v' sets invest_solve_branching_method investments_at_reveal_period" in text
    assert "has no stochastic_branches" in text


@pytest.mark.parametrize("method, realized, refused", [
    # after_reveal: the reveal period p2 is shared, so it can be realized; nothing after it here.
    ("investments_after_reveal_period", ["p1", "p2"], False),
    # at_reveal: p2 invests per branch -> realizing it is refused; p1 (before the reveal) is fine.
    ("investments_at_reveal_period", ["p1", "p2"], True),
    ("investments_at_reveal_period", ["p1"], False),
    # operations_only: every period shared, anything can be realized.
    ("operations_only", ["p1", "p2"], False),
])
def test_g1_realized_period_with_per_branch_investments_is_refused(method, realized, refused):
    """Refusal 3: a realized period whose investments are per branch cannot be realized (after the reveal under
    investments_after_reveal_period, from the reveal on under investments_at_reveal_period); the error names the
    period and the parameter."""
    sc = _branching(method, realized_invest_periods={"v": [("v", p) for p in realized]})
    if not refused:
        sc.check_invest_branching()
        return
    with pytest.raises(FlexToolConfigError) as exc:
        sc.check_invest_branching()
    text = str(exc.value)
    assert "realized_invest_periods include 'p2'" in text and "from the reveal period on ('p2')" in text


def test_g1_after_reveal_refuses_a_period_after_the_reveal():
    """Refusal 3 under investments_after_reveal_period with the reveal at p1: p2 invests per branch, so a realized
    p2 (operations here) is refused; p1 alone is not."""
    sc = _branching("investments_after_reveal_period",
                    stochastic_branches={"v": [("p1", "A", "t1", "no", 1.0), ("p1", "B", "t1", "no", 1.0)]},
                    realized_periods={"v": [("v", "p1"), ("v", "p2")]})
    with pytest.raises(FlexToolConfigError) as exc:
        sc.check_invest_branching()
    assert "realized_periods include 'p2'" in str(exc.value)
    assert "after the reveal period ('p1')" in str(exc.value)
    sc.realized_periods = {"v": [("v", "p1")]}
    sc.check_invest_branching()


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
    from flextool.engine.solve._benders import solve_benders

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
REALIZE_KEY = "solve_data/realized_invest_periods_of_current_solve"


@pytest.fixture(scope="module")
def _solved(scenario_run):
    """Solve the last sub-solve, loaded through the production provider
    path; the handoff builder then gets the arguments the cascade passes
    (the step's provider, the prior handoff, the load folder)."""
    from polar_high import Problem
    from flextool.engine import build_flextool

    run = scenario_run(SCENARIO)
    assert run.last.solve_name == SOLVE_NAME
    data = load_scenario(run)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal
    return sol, run, data


def _build_handoff(sol, run, flex_data, provider):
    from flextool.engine.preprocess.flex_data import build_handoff_from_solution

    return build_handoff_from_solution(
        sol, run.load_folder, SOLVE_NAME,
        prior_handoff=run.last.prior_handoff, flex_data=flex_data,
        provider=provider)


def _provider_realizing(run, period: str):
    """A clone of the step's provider whose realize-invest key lists only
    *period* (dtype kept from the cascade's frame)."""
    from flextool.engine.preprocess.flex_data import _read_realize_invest_periods

    provider = clone_provider(run.last.provider)
    orig = provider.get(REALIZE_KEY)
    assert orig is not None and orig.columns == ["period"]
    assert period not in _read_realize_invest_periods(
        run.load_folder / f"{REALIZE_KEY}.csv", provider=provider)
    provider.put(REALIZE_KEY, pl.DataFrame({"period": [period]}).cast(orig.schema))
    # The handoff builder reads the overlaid key (not a disk file).
    assert _read_realize_invest_periods(
        run.load_folder / f"{REALIZE_KEY}.csv", provider=provider) == {period}
    return provider


@pytest.mark.solver
def test_g3_clean_real_names_no_raise(_solved):
    """The happy path — a deterministic (real-named) flex_data commits no
    synthetic period, so the guard is inert."""
    sol, run, data = _solved
    handoff = _build_handoff(sol, run, data, clone_provider(run.last.provider))
    assert handoff is not None


@pytest.mark.solver
def test_g3_config_level_synthetic_in_realize_invest_raises(_solved):
    """realized_invest_periods listing a synthetic branch token → config
    raise (case a)."""
    sol, run, data = _solved
    provider = _provider_realizing(run, "p2020_low")
    # flex_data whose period_branch_full marks p2020_low as a synthetic
    # fan member of the real anchor p2020.
    bad = dataclasses.replace(
        data,
        period_branch_full=_pbf([("p2020", "p2020"),
                                 ("p2020", "p2020_low")], enum=True),
    )
    with pytest.raises(FlexToolConfigError) as exc:
        _build_handoff(sol, run, bad, provider)
    assert "p2020_low" in str(exc.value)


@pytest.mark.solver
def test_g3_none_frame_degrades_no_raise(_solved):
    """period_branch_full unavailable → the guard is a no-op even when the
    realize key carries a synthetic token (the documented None-degrade;
    Guard 2 covers that path's stochastic case)."""
    sol, run, data = _solved
    provider = _provider_realizing(run, "p2020_low")
    degraded = dataclasses.replace(data, period_branch_full=None)
    # No raise — synthetic_branch_tokens(None) is empty, guard skipped.
    handoff = _build_handoff(sol, run, degraded, provider)
    assert handoff is not None
