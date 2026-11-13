"""Δ.5 — Cluster A (annual integration / NPV) parity tests.

Per-fixture parity check that the new lazy-polars NPV helpers in
``flextool.engine.preprocess.params._npv`` produce frame-equal output
(modulo column ordering) vs the eager ``ed_entity_annual_family_from_source``
helper in ``flextool.engine.preprocess.params._derived`` — which itself is
parity-clean against flextool's CSV-side reference output (gated by
``test_db_direct_inflation_2pct``).

Two layers of test coverage:

1. **Per-fixture parity sweep** — every scenario in
   ``_parity_sweep.PARITY_SWEEP_CASES`` runs both helpers (lazy + eager)
   on its cascade's post-solve provider and compares the resulting
   Param frames row-for-row.  Any field that is
   ``None`` on either side is checked symmetrically.

2. **Spot tests** — hand-derived analytical oracles for the canonical
   inflation cascade (rate=2%, 4 periods × 5 years) on the
   ``work_inflation_check`` fixture.  This test mirrors
   ``test_db_direct_inflation_2pct.py`` but on the new lazy path.

The parity sweep is the gating test: any divergence between lazy and
eager outputs surfaces as a per-fixture failure with a diff frame.
"""
from __future__ import annotations

import math
import polars as pl
import pytest

from flextool.engine.preprocess.params import _npv as npv
from flextool.engine.preprocess.params._derived import (
    _read_active_solve,
    _solve_periods,
    _period_in_use_set,
    _periodAll_from_source,
    _read_period_with_history,
)
from tests._support.load import ScenarioRun, clone_provider, load_scenario


def _lazy_inputs(run: ScenarioRun):
    """``(reader, provider, folder)`` for calling the lazy NPV helpers on the run's last step.

    ``provider`` is a clone of the step's post-solve provider (the frames the cascade solved with);
    ``folder`` is the run's empty load folder, so no helper can fall back to disk.
    """
    return run.reader(), clone_provider(run.last.provider), run.load_folder


# Phase 3d: curated parity sweep — replaces the legacy disk-discovery
# function that walked ``data/work_*/``.  Each tuple drives
# ``scenario_run`` (db_fixture column) to run the scenario's cascade
# on demand under tmp.
from _parity_sweep import PARITY_SWEEP_CASES  # noqa: E402

PARITY_CASES = [(legacy, scen, dbf) for legacy, scen, dbf in PARITY_SWEEP_CASES]


# ---------------------------------------------------------------------------
# Per-fixture parity sweep
# ---------------------------------------------------------------------------


def _frames_equal(a: pl.DataFrame | None, b: pl.DataFrame | None,
                    keys: tuple[str, ...]) -> tuple[bool, str | None]:
    """Compare two frames for row-set equality on ``keys``+ value column.

    Returns ``(equal, diff_message)`` — ``diff_message`` is a short
    summary of the divergence when ``equal=False``.
    Both ``None`` is considered equal.
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
    # Float-tolerant comparison.
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
    if max_diff < 1e-7 * max(1.0, max(abs(x) for x in av if x is not None) or 1.0):
        return True, None
    return False, f"max abs diff = {max_diff!r}"


def _param_frame(p) -> pl.DataFrame | None:
    if p is None:
        return None
    return p.frame


@pytest.mark.parametrize(
    "work_name,scenario,db_fixture", PARITY_CASES,
    ids=[c[0] for c in PARITY_CASES],
)
def test_npv_lazy_vs_eager_parity(
        work_name: str, scenario: str, db_fixture: str,
        scenario_run) -> None:
    """Per-fixture parity: lazy NPV helpers vs eager
    ``ed_entity_annual_family_from_source``.

    Compares the 4 NPV-family fields plus ``p_inflation_op`` and
    ``p_ed_fixed_cost``.  Both paths read the same fixture; any
    divergence indicates a bug in the lazy port.
    """
    run = scenario_run(scenario, db_fixture=db_fixture)
    reader, provider, work = _lazy_inputs(run)
    # Build the FlexData via the production loader call — this populates the
    # eager NPV fields via the existing apply_derived_f path.
    data_eager = load_scenario(run)

    # Slice D (§11.4): the eager reference
    # ``ed_entity_annual_family_from_source`` embeds its own period walk
    # and is NOT lineage-aware, so under ``recourse`` its per-leaf annuity
    # windows diverge from the lazy lineage-filtered path by design.  The
    # byte-equality assumption is valid only while lineage is None — skip
    # the sweep on any flag-on fixture (guards against a recourse fixture
    # ever entering PARITY_CASES).
    if getattr(data_eager, "recourse_invest", False):
        pytest.skip("eager-vs-lazy NPV parity does not hold under "
                    "recourse (lineage-filtered windows; design §11.4)")

    # Re-read via the lazy entry points and compare per-field.
    active_solve = _read_active_solve(work, provider=provider)
    period_in_use = _period_in_use_set(reader, active_solve, work,
                                          provider=provider)
    period_universe = _periodAll_from_source(reader, active_solve,
                                                workdir=work,
                                                provider=provider)
    period_invest = _solve_periods(reader, active_solve, "invest_periods") or []
    period_with_history = (_read_period_with_history(work, provider=provider)
                              or list(period_in_use))
    # The solve's years from the start of the run, as apply_npv passes them (a later solve of a chain, item 242).
    year_rows = npv.solve_year_rows(work, provider, reader, active_solve, list(period_universe))
    # Compute lazy variants.
    lazy_infl = npv.p_inflation_op_from_source(
        reader, active_solve, period_in_use, period_universe, provider=provider, year_rows=year_rows)
    lazy_fc = npv.p_ed_fixed_cost_from_source(reader, period_with_history)
    lazy_ann = lazy_div = lazy_lfc = lazy_lfcd = None
    # Mirror apply_npv's no_invest short-circuit so the lazy path
    # matches the eager loader's gate.
    ed_invest = getattr(data_eager, "ed_invest_set", None)
    ed_divest = getattr(data_eager, "ed_divest_set", None)
    no_invest = (
        (ed_invest is None or ed_invest.height == 0)
        and (ed_divest is None or ed_divest.height == 0)
    )
    if active_solve is not None and not no_invest:
        lazy_ann = npv.ed_entity_annual_discounted_from_source(
            reader, active_solve,
            period_invest, period_in_use, period_universe, year_rows=year_rows)
        lazy_div = npv.ed_entity_annual_divest_discounted_from_source(
            reader, active_solve,
            period_invest, period_in_use, period_universe, year_rows=year_rows)
        lazy_lfc = npv.ed_lifetime_fixed_cost_from_source(
            reader, active_solve,
            period_with_history, period_in_use, period_universe, year_rows=year_rows)
        lazy_lfcd = npv.ed_lifetime_fixed_cost_divest_from_source(
            reader, active_solve,
            period_invest, period_in_use, period_universe, year_rows=year_rows)

    # Compare.
    failures: list[str] = []
    fields: list[tuple[str, object, object, tuple[str, ...]]] = [
        ("p_inflation_op", _param_frame(data_eager.p_inflation_op),
         _param_frame(lazy_infl), ("d",)),
        ("p_ed_fixed_cost", _param_frame(data_eager.p_ed_fixed_cost),
         _param_frame(lazy_fc), ("e", "d")),
        ("ed_entity_annual_discounted",
         _param_frame(data_eager.ed_entity_annual_discounted),
         _param_frame(lazy_ann), ("e", "d")),
        ("ed_entity_annual_divest_discounted",
         _param_frame(data_eager.ed_entity_annual_divest_discounted),
         _param_frame(lazy_div), ("e", "d")),
        ("ed_lifetime_fixed_cost",
         _param_frame(data_eager.ed_lifetime_fixed_cost),
         _param_frame(lazy_lfc), ("e", "d")),
        ("ed_lifetime_fixed_cost_divest",
         _param_frame(data_eager.ed_lifetime_fixed_cost_divest),
         _param_frame(lazy_lfcd), ("e", "d")),
    ]
    for name, eager_frame, lazy_frame, keys in fields:
        ok, msg = _frames_equal(eager_frame, lazy_frame, keys)
        if not ok:
            failures.append(f"{name}: {msg}\n  eager:\n{eager_frame}\n  lazy:\n{lazy_frame}")

    if failures:
        pytest.fail("\n\n".join(failures))


# ---------------------------------------------------------------------------
# Spot test — hand-derived inflation 2pct on work_inflation_check
# ---------------------------------------------------------------------------


# Phase 3d: ``work_inflation_check`` rebuilt from
# ``wind_battery_invest_lifetime_renew_inflation_2pct`` via
# ``scenario_run``.
SCENARIO_INFLATION = "wind_battery_invest_lifetime_renew_inflation_2pct"

R = 0.02
PERIODS_INFL = ["p2020", "p2025", "p2030", "p2035"]
PDY_INFL = {"p2020": 0.0, "p2025": 5.0, "p2030": 10.0, "p2035": 15.0}
OFFSET_OPS = 0.5
OFFSET_INV = 0.0


def _years_for(d: str) -> range:
    base = PDY_INFL[d]
    return range(int(base), int(base) + 5)


def _ops_factor(d: str) -> float:
    return sum((1.0 + R) ** -(y + OFFSET_OPS) for y in _years_for(d))


def _inv_factor(d: str) -> float:
    return sum((1.0 + R) ** -(y + OFFSET_INV) for y in _years_for(d))


def _annuity(invest_cost_eur_per_kw: float,
                   discount_rate: float,
                   lifetime_years: float) -> float:
    r, n = discount_rate, lifetime_years
    return invest_cost_eur_per_kw * 1000.0 * r / (1.0 - (1.0 / (1.0 + r)) ** n)


ANN_WIND = _annuity(1000.0, 0.04, 5.0)
ANN_BATT = _annuity(200.0, 0.05, 10.0)


def _annual_disc_reinvest_automatic(annuity: float, d: str) -> float:
    pdy_d = PDY_INFL[d]
    return annuity * sum(_inv_factor(d_all)
                              for d_all in PERIODS_INFL
                              if PDY_INFL[d_all] >= pdy_d)


def _val(param, *idx_keys: str) -> float:
    df = param.frame
    keys = [c for c in df.columns if c != "value"]
    assert len(idx_keys) == len(keys)
    expr = pl.col(keys[0]) == idx_keys[0]
    for c, v in zip(keys[1:], idx_keys[1:]):
        expr = expr & (pl.col(c) == v)
    sub = df.filter(expr)
    assert sub.height == 1
    return float(sub["value"][0])


def _approx(actual: float, expected: float, *, abs_tol: float = 1e-9) -> bool:
    """1e-9 absolute, or 1e-12 relative (floating-point rounding of a 1e6-scale annuity)."""
    return math.isclose(actual, expected, abs_tol=abs_tol, rel_tol=1e-12)


def test_npv_lazy_p_inflation_op_2pct(scenario_run):
    """``p_inflation_op[d]`` from the lazy helper must match the
    hand-derived 2 % oracle to 1e-9 absolute.
    """
    reader, provider, work = _lazy_inputs(scenario_run(SCENARIO_INFLATION))
    active_solve = _read_active_solve(work, provider=provider)
    period_in_use = _period_in_use_set(reader, active_solve, work,
                                          provider=provider)
    period_universe = _periodAll_from_source(
        reader, active_solve, workdir=work, provider=provider)
    p_infl = npv.p_inflation_op_from_source(
        reader, active_solve, period_in_use, period_universe)
    assert p_infl is not None
    for d in PERIODS_INFL:
        actual = _val(p_infl, d)
        expected = _ops_factor(d)
        assert _approx(actual, expected), (
            f"p_inflation_op[{d}]: actual={actual!r}, expected={expected!r}"
        )


def test_npv_lazy_ed_entity_annual_discounted_2pct(scenario_run):
    """Lazy ``ed_entity_annual_discounted`` for wind+battery (both
    ``reinvest_automatic`` lifetime_method) must match the hand-derived
    annuity × inv_factor sum to 1e-9 absolute (or 1e-12 relative).
    """
    reader, provider, work = _lazy_inputs(scenario_run(SCENARIO_INFLATION))
    active_solve = _read_active_solve(work, provider=provider)
    period_in_use = _period_in_use_set(reader, active_solve, work,
                                          provider=provider)
    period_universe = _periodAll_from_source(
        reader, active_solve, workdir=work, provider=provider)
    period_invest = _solve_periods(reader, active_solve, "invest_periods") or []
    ed = npv.ed_entity_annual_discounted_from_source(
        reader, active_solve,
        period_invest, period_in_use, period_universe)
    assert ed is not None
    for d in PERIODS_INFL:
        for ent, ann in (("wind_plant", ANN_WIND), ("battery", ANN_BATT)):
            actual = _val(ed, ent, d)
            expected = _annual_disc_reinvest_automatic(ann, d)
            assert _approx(actual, expected), (
                f"ed[{ent}, {d}]: actual={actual!r}, expected={expected!r}"
            )


# ---------------------------------------------------------------------------
# discount_rate 0 is 0 %, a missing rate is the 5 % default (both annuity paths)
# ---------------------------------------------------------------------------
#
# invest_cost 1000 CUR/kW, lifetime 20 a.  Hand values:
#   r = 0          → 1000·1000/20                 = 50 000 CUR/MW/a (straight line);
#   r missing/0.05 → 1e6·0.05/(1 − 1.05^−20)      = 80 242.58719 CUR/MW/a.

ANN_R0 = 50_000.0
ANN_R5 = 80_242.58719


class _Lookup:
    """Duck-typed ``PdLookup`` (``get`` / ``has``) over a ``{(e, param): value}`` dict."""

    def __init__(self, values: dict[tuple[str, str], float]) -> None:
        self._v = values

    def get(self, e: str, param: str, d: str) -> float:
        return self._v.get((e, param), 0.0)

    def has(self, e: str, param: str, d: str) -> bool:
        return (e, param) in self._v


def _lazy_annuity(disc: float | None, life: float | None, cost: float = 1000.0) -> float:
    from flextool.engine.input._inmemory_reader import InMemoryReader

    def frame(v):
        return pl.DataFrame({"name": ["u"], "value": [v]})

    params = {("unit", "invest_cost"): frame(cost),
              ("unit", "invest_method"): pl.DataFrame({"name": ["u"], "value": ["invest_no_limit"]})}
    if disc is not None:
        params[("unit", "discount_rate")] = frame(disc)
    if life is not None:
        params[("unit", "lifetime")] = frame(life)
    src = InMemoryReader({"unit": pl.DataFrame({"name": ["u"]})}, params,
                         defaults={("unit", "discount_rate"): 0.05})
    ed = pl.LazyFrame({"e": ["u"], "d": ["p2020"]})
    out = npv._per_method_annuity_lf(src, ed, "invest_cost", npv._INVEST_NOT_ALLOWED).collect()
    return float(out["ann"][0])


def _emit_annuity(disc: float | None, life: float | None, cost: float = 1000.0) -> float:
    from flextool.engine.preprocess.emit._entity_annual import _annuity

    values = {("u", "invest_cost"): cost}
    if disc is not None:
        values[("u", "discount_rate")] = disc
    if life is not None:
        values[("u", "lifetime")] = life
    return _annuity(_Lookup(values), "u", "p2020", "invest_cost")


@pytest.mark.parametrize("path", [_lazy_annuity, _emit_annuity], ids=["lazy", "emit"])
@pytest.mark.parametrize("disc, expected", [(0.0, ANN_R0), (None, ANN_R5), (0.05, ANN_R5)],
                         ids=["zero", "missing", "explicit_5pct"])
def test_annuity_discount_rate(path, disc, expected) -> None:
    assert math.isclose(path(disc, 20.0), expected, rel_tol=1e-9), (path.__name__, disc)


@pytest.mark.parametrize("path", [_lazy_annuity, _emit_annuity], ids=["lazy", "emit"])
def test_annuity_without_lifetime_is_an_error(path) -> None:
    from flextool.engine.input._solve_state import FlexToolConfigError

    with pytest.raises(FlexToolConfigError, match="lifetime is missing or not above 0 where invest_cost is not 0: "
                                                  "'u' in period p2020"):
        path(0.05, None)
    assert path(0.05, None, cost=0.0) == 0.0


def test_annuity_factor_small_rate_is_the_straight_line() -> None:
    from flextool.engine.preprocess._annuity import annuity_factor, annuity_factor_expr

    assert annuity_factor(0.0, 20.0) == 1 / 20
    assert math.isclose(annuity_factor(1e-12, 20.0), 1 / 20, rel_tol=1e-9)
    got = pl.select(annuity_factor_expr(pl.lit(1e-12), pl.lit(20.0))).item()
    assert math.isclose(got, 1 / 20, rel_tol=1e-9)
