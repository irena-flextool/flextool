"""Commodity price ladders in stochastic solves (#32, #41).

#32: a ladder purchase is an operating cost, so in a stochastic solve it carries the probability of its
period's branch (``pdt_branch_weight``, the weight in ``op_factor``), like every other operating cost.
Before the fix every branch counted in full.

#41: the forecast branch periods of a stochastic solve (``period1_upper``, …) are minted per solve, so no
ladder row can be authored for them.  They use the annual ladder of their anchor period (price and tier
quantity together); before the fix purchases in the branches had no price and no tier limit.

#59: a rolling chain's carried realized MWh per tier (``p_ladder_cum_realized_mwh``, populated from the
solve handoff by ``_extract_cumulative_commodity``) is only ever keyed under the REAL period name — the
realized branch keeps the real name, so ``v_trade`` there already is its purchase, but a forecast branch
period of the SAME year has no row of its own there either.  ``_eq_tier_cap_annual`` must give it the
anchor's carried value, the same way it already does for ``p_period_share`` / ``p_f_d_k`` (#89); a
dispatch-parent solve that has an auto-spawned realized-dispatch sibling (OC1 separate_dispatch) must
add none of its own degenerate contribution either — the sibling re-solves the same hours for real and
carries that forward instead.

Vehicle: ``2_day_stochastic_dispatch`` of ``tests/fixtures/stochastics.json`` (one real period
``period1`` = the period label, which has no dispatch, and four forecast branch periods; a forecasts-only fan), coal moved to a ladder.  A
ladder whose cheap tier is the flat price (16) and unlimited must cost exactly what the flat price
costs.  The rolling smoke test uses ``1_week_rolling_wind``.  Databases are built from the JSON
fixture (never a checked-in ``.sqlite``); the stochastic branch weights are rewritten in the
``stochastic_branches`` map for the skewed cases.
"""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import polars as pl
import pytest

from tests.engine._unit_db import b64, build, cstr_over, make_db, run, utf8

REPO_ROOT = Path(__file__).resolve().parents[2]
STOCHASTICS_JSON = REPO_ROOT / "tests" / "fixtures" / "stochastics.json"
CHAIN_2DAY = ["base", "init", "system", "2day"]
CHAIN_ROLL = ["init", "base", "system", "1week_rolling"]
BRANCHES = ("realized", "upper", "lower", "mid")
BRANCH_PERIODS = tuple(f"period1_{b}" for b in BRANCHES)
ALL_PERIODS = ("period1",) + BRANCH_PERIODS
FLAT_PRICE = 16.0
# Input weights 1/2/3/4: the four forecasts renormalise to 0.1/0.2/0.3/0.4 (the period label has no dispatch).
SKEWED = {"realized": 1.0, "upper": 2.0, "lower": 3.0, "mid": 4.0}
SKEWED_BW = {"period1_realized": 0.1, "period1_upper": 0.2, "period1_lower": 0.3, "period1_mid": 0.4}
WEIGHTS = {"uniform": None, "skewed": SKEWED}


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


# ---------------------------------------------------------------------------
# Database helpers


def _fixture_json(tmp_path_factory, weights: dict[str, float] | None) -> Path:
    """``stochastics.json``, with the ``2day_dispatch`` branch weights rewritten to ``weights``."""
    if weights is None:
        return STOCHASTICS_JSON
    data = json.loads(STOCHASTICS_JSON.read_text())
    rewritten = False
    for row in data["parameter_values"]:
        if row[0] == "solve" and row[1] == "2day_dispatch" and row[2] == "stochastic_branches":
            spine_map = json.loads(base64.b64decode(row[3][0]))
            for _period, period_map in spine_map["data"]:
                for branch, branch_map in period_map["data"]:
                    for _step, leaf in branch_map["data"]:
                        leaf["data"] = [[leaf["data"][0][0], weights[branch]]]
            row[3][0] = base64.b64encode(json.dumps(spine_map).encode()).decode("ascii")
            rewritten = True
    assert rewritten, "2day_dispatch stochastic_branches not found; the fixture layout changed"
    path = tmp_path_factory.mktemp("stoch_weights") / "stochastics.json"
    path.write_text(json.dumps(data))
    return path


def _tiers(price1: float, quantity1: float, price2: float = 30.0) -> dict:
    """tier → {price, quantity}: tier 1 at ``price1`` up to ``quantity1``, tier 2 at ``price2`` unlimited."""
    def tier(price, quantity):
        return {"type": "map", "index_type": "str", "rank": 1,
                "data": [["price", price], ["quantity", quantity]]}
    return {"type": "map", "index_type": "str", "rank": 2, "index_name": "tier",
            "data": [["1", tier(price1, quantity1)], ["2", tier(price2, math.inf)]]}


def _annual(form: str, tiers: dict) -> list:
    """coal on the annual ladder: ``2d`` (no period index), ``3d_label`` (keyed by ``period1`` only) or
    ``3d_all`` (keyed by ``period1`` and every branch period name)."""
    if form == "2d":
        value = {k: v for k, v in tiers.items() if k != "type"}
    else:
        periods = ("period1",) if form == "3d_label" else ALL_PERIODS
        value = {"index_type": "str", "rank": 3, "index_name": "period",
                 "data": [[p, tiers] for p in periods]}
    return [["commodity", "coal", "price_method", b64("price_ladder_annual", "str")],
            ["commodity", "coal", "price_ladder_annual", b64(value, "map")]]


def _cumulative(tiers: dict) -> list:
    value = {k: v for k, v in tiers.items() if k != "type"}
    return [["commodity", "coal", "price_method", b64("price_ladder_cumulative", "str")],
            ["commodity", "coal", "price_ladder_cumulative", b64(value, "map")]]


def _steps(tmp_path_factory, alt: str, values: list, weights=None, chain=CHAIN_2DAY) -> dict:
    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=chain,
                        fixture_json=_fixture_json(tmp_path_factory, weights))
    return run(tmp_path_factory, url, scen)


def _last(steps: dict):
    return list(steps.values())[-1]


def _obj(steps: dict) -> float:
    return _last(steps).solution.obj


def _v_trade(step) -> pl.DataFrame:
    return utf8(step.solution.value("v_trade"))


# ---------------------------------------------------------------------------
# Fixtures


@pytest.fixture(scope="module")
def flat(tmp_path_factory):
    """Coal at its flat price 16 (no ladder), per weight case."""
    return {name: _steps(tmp_path_factory, f"c5_flat_{name}", [], weights=w) for name, w in WEIGHTS.items()}


@pytest.fixture(scope="module")
def binding_q(flat):
    """A tier-1 quantity that binds: half of the largest branch coal purchase (coal runs in one branch only)."""
    step = _last(flat["uniform"])
    flows = utf8(step.solution.value("v_flow")).filter(
        (pl.col("p") == "coal_plant") & (pl.col("source") == "coal_market"))
    fd = step.flex_data
    w = utf8(fd.p_step_duration.frame).join(utf8(fd.p_timestep_weight.frame), on=["d", "t"], suffix="_w")
    size = float(utf8(fd.p_unitsize.frame).filter(pl.col("p") == "coal_plant")["value"][0])
    slope = utf8(fd.p_slope.frame).filter(pl.col("p") == "coal_plant").select("d", "t", slope="value")
    mwh = (flows.join(w, on=["d", "t"]).join(slope, on=["d", "t"])
           .group_by("d").agg((pl.col("value") * size * pl.col("slope") * pl.col("value_right")
                               * pl.col("value_w")).sum().alias("mwh")))
    branch_mwh = mwh.filter(pl.col("d").is_in(list(BRANCH_PERIODS)))["mwh"]
    assert branch_mwh.len() == len(BRANCH_PERIODS) and branch_mwh.max() > 0
    return 0.5 * float(branch_mwh.max())


@pytest.fixture(scope="module")
def period_share(flat) -> float:
    """``p_period_share[d]`` of this fixture — the same for every period (the period label and every forecast
    branch cover the same 2-day calendar window): 2/365 = 0.0054794520...

    ``_eq_tier_cap_annual`` treats the authored ladder quantity as a per-calendar-YEAR amount and
    scales it down to this period's own share of the year (#89) — so a quantity authored directly as
    ``binding_q`` no longer makes the tier's *realized* MWh cap equal ``binding_q``; it makes the
    *per-year* allowance equal ``binding_q``, and the period only gets ``binding_q · period_share`` of
    it.  ``binding_q / period_share`` is the per-year quantity that gets this fixture's 2-day period
    exactly ``binding_q`` MWh of headroom — see ``_tier_quantity`` below.
    """
    fd = _last(flat["uniform"]).flex_data
    values = utf8(fd.p_period_share.frame)["value"].unique().to_list()
    assert len(values) == 1, f"expected one p_period_share value for this fixture, got {values}"
    return float(values[0])


def _tier_quantity(binding_q: float, period_share: float) -> float:
    """The per-year tier-1 quantity to author so the period's realized cap equals ``binding_q`` MWh."""
    return binding_q / period_share


# ---------------------------------------------------------------------------
# #32 (+ #41 for the 2d form): a ladder at the flat price costs the flat price


@pytest.mark.parametrize("weights", list(WEIGHTS))
@pytest.mark.parametrize("form", ["3d_all", "2d"])
def test_ladder_at_flat_price_equals_flat_price_cost(tmp_path_factory, flat, form, weights) -> None:
    steps = _steps(tmp_path_factory, f"c5_flat_ladder_{form}_{weights}",
                   _annual(form, _tiers(FLAT_PRICE, math.inf)), weights=WEIGHTS[weights])
    assert _obj(steps) == pytest.approx(_obj(flat[weights]), rel=1e-9)


def test_cumulative_ladder_at_flat_price_equals_flat_price_cost(tmp_path_factory, flat) -> None:
    steps = _steps(tmp_path_factory, "c5_flat_cum", _cumulative(_tiers(FLAT_PRICE, math.inf)), weights=SKEWED)
    assert _obj(steps) == pytest.approx(_obj(flat["skewed"]), rel=1e-9)


def test_ladder_objective_coefficient_carries_branch_weight(tmp_path_factory) -> None:
    """The objective coefficient of v_trade[c,n,d,i] is cu · price_i · inflation_op / period_share · bw[d]."""
    prices = {"1": 10.0, "2": 30.0}
    step = _last(_steps(tmp_path_factory, "c5_coef", _annual("3d_all", _tiers(prices["1"], 1e6)),
                        weights=SKEWED))
    fd = step.flex_data
    annual = (utf8(fd.p_inflation_op.frame).join(utf8(fd.p_period_share.frame), on="d")
              .select("d", f=pl.col("value") / pl.col("value_right")))
    factor = dict(zip(annual["d"], annual["f"]))
    m = build(fd).canonicalise()
    coef = {nm: float(c) for nm, c in zip(m.col_names, m.col_obj) if nm.startswith("v_trade[")}
    seen = set()
    for nm, c in coef.items():
        _c, _n, d, i = nm[nm.index("[") + 1:-1].split(",")
        assert c == pytest.approx(prices[i] * factor[d] * SKEWED_BW[d], rel=1e-12, abs=0.0), nm
        seen.add(d)
    assert seen == set(BRANCH_PERIODS)


# ---------------------------------------------------------------------------
# #41: branch periods inherit the anchor period's annual ladder


def test_branch_periods_inherit_annual_ladder_price(tmp_path_factory, flat) -> None:
    steps = _steps(tmp_path_factory, "c5_label_price", _annual("3d_label", _tiers(FLAT_PRICE, math.inf)))
    assert _obj(steps) == pytest.approx(_obj(flat["uniform"]), rel=1e-9)


@pytest.fixture(scope="module")
def cap_reference(tmp_path_factory, binding_q, period_share):
    """The binding-cap ladder keyed by every period name (the authored-everywhere reference).

    The authored quantity is ``_tier_quantity(binding_q, period_share)`` (the per-year amount whose
    period_share-scaled realization is exactly ``binding_q`` — see the ``period_share`` fixture), so the
    branch that actually needs tier 1 binds there.
    """
    quantity = _tier_quantity(binding_q, period_share)
    return {name: _steps(tmp_path_factory, f"c5_cap_all_{name}", _annual("3d_all", _tiers(10.0, quantity)),
                         weights=w)
            for name, w in WEIGHTS.items()}


@pytest.mark.parametrize("weights", list(WEIGHTS))
@pytest.mark.parametrize("form", ["3d_label", "2d"])
def test_branch_periods_inherit_annual_tier_cap(tmp_path_factory, binding_q, period_share, cap_reference,
                                                form, weights) -> None:
    quantity = _tier_quantity(binding_q, period_share)
    steps = _steps(tmp_path_factory, f"c5_cap_{form}_{weights}", _annual(form, _tiers(10.0, quantity)),
                   weights=WEIGHTS[weights])
    ref = cap_reference[weights]
    assert _obj(steps) == pytest.approx(_obj(ref), rel=1e-9)
    for s in (steps, ref):
        step = _last(s)
        rows = cstr_over(build(step.flex_data), "ladder_tier_cap_annual_roll")
        assert rows is not None
        assert set(rows.filter(pl.col("i") == "1")["d"]) == set(BRANCH_PERIODS)
        tier1 = _v_trade(step).filter((pl.col("i") == "1") & pl.col("d").is_in(list(BRANCH_PERIODS)))
        assert (tier1["value"] - binding_q).abs().min() <= 1e-7 * binding_q, "the tier-1 cap does not bind"


# ---------------------------------------------------------------------------
# Rolling: per-roll branch read and weights, warm-start cascade with the ladder Params


def test_rolling_ladder_at_flat_price_equals_flat_price_cost(tmp_path_factory) -> None:
    ladder = _steps(tmp_path_factory, "c5_roll_ladder", _annual("2d", _tiers(FLAT_PRICE, math.inf)),
                    chain=CHAIN_ROLL)
    flat_roll = _steps(tmp_path_factory, "c5_roll_flat", [], chain=CHAIN_ROLL)
    assert len(ladder) == len(flat_roll) > 1
    assert sum(s.solution.obj for s in ladder.values()) == pytest.approx(
        sum(s.solution.obj for s in flat_roll.values()), rel=1e-9)


# ---------------------------------------------------------------------------
# #59: a branch period inherits the anchor's carried realized MWh in the next roll


def test_branch_periods_inherit_realized_mwh_carry(tmp_path_factory, binding_q, period_share) -> None:
    """``_eq_tier_cap_annual`` must give a branch period the anchor's carried ``p_ladder_cum_realized_mwh``
    (the running cross-roll "already bought" total), the same way it already gives it the anchor's
    ``p_period_share`` / ``p_f_d_k`` (#89) -- #59.

    The carrier is a solve-handoff carrier, not a DB-authored one, so a "next roll" is simulated by
    authoring it directly on an already-solved single solve's FlexData: ``build()`` builds the LP
    straight from the FlexData it is given, with no re-derivation (the same technique
    ``test_ladder_objective_coefficient_carries_branch_weight`` uses to inspect objective coefficients).

    TEST authors the carry at the anchor (``period1``) only -- the shape
    ``_extract_cumulative_commodity`` actually produces, since the realized branch keeps the real
    period name (#59).  REF authors the SAME carry directly at every period, branches included -- the
    shape a branch would need to see its remaining budget reduced the same way the anchor's is.  Before
    the fix, a branch's carry falls through to 0 (full, un-reduced cap) in TEST while REF's branch has
    its own authored (reduced) cap, so the two solves differ for the branch whose natural demand exceeds
    both cap levels (the max-demand branch the ``binding_q`` fixture is built around); after the fix
    they match.
    """
    from polar_high import Param

    quantity = _tier_quantity(binding_q, period_share)
    steps = _steps(tmp_path_factory, "c59_rlz_carry", _annual("3d_label", _tiers(10.0, quantity)))
    fd = _last(steps).flex_data
    realized = 0.3 * binding_q  # leaves 0.7 * binding_q of headroom -- still less than the max branch's
                                # natural (uncapped) demand, so the cap keeps binding.

    def _tier1_with_carry(carried_periods: tuple[str, ...]) -> dict[str, float]:
        frame = pl.DataFrame({
            "c": ["coal"] * len(carried_periods), "i": ["1"] * len(carried_periods),
            "d": list(carried_periods), "value": [realized] * len(carried_periods),
        })
        fd.p_ladder_cum_realized_mwh = Param(("c", "i", "d"), frame)
        sol = build(fd).solve()
        tier1 = utf8(sol.value("v_trade")).filter(
            (pl.col("i") == "1") & pl.col("d").is_in(list(BRANCH_PERIODS)))
        return dict(zip(tier1["d"].to_list(), tier1["value"].to_list()))

    test_mwh = _tier1_with_carry(("period1",))
    ref_mwh = _tier1_with_carry(ALL_PERIODS)

    assert set(test_mwh) == set(ref_mwh) == set(BRANCH_PERIODS)
    for b in BRANCH_PERIODS:
        assert test_mwh[b] == pytest.approx(ref_mwh[b], rel=1e-7), (
            f"{b}: anchor-only carry gave {test_mwh[b]} MWh, expected the anchor's carry inherited "
            f"(= {ref_mwh[b]} MWh, authored directly on every period)")
    # At least one branch (the fixture's max-demand branch) is actually capped by the carry-adjusted
    # budget -- otherwise the comparison above would hold trivially (both uncapped) regardless of the fix.
    assert max(test_mwh.values()) == pytest.approx(binding_q - realized, rel=1e-7)


def test_dispatch_sibling_parent_carries_no_realized_mwh_of_its_own() -> None:
    """``_extract_cumulative_commodity`` / ``_extract_cum_sim_hours`` must contribute nothing of their
    own when THIS solve is the parent of an auto-spawned realized-dispatch sibling (#59 fix 3): the
    sibling re-solves the same realized hours for real right after it and carries that forward instead
    (``_run_model``'s ``ROLL_FORWARD_FIELDS`` routes the sibling's prior from its own previous sibling
    roll, never from this parent) -- adding this parent's own (degenerate, weight-0) contribution on top
    would count those hours twice.

    Hand-computed oracle (not copied from the function under test): with step_duration 10h at each of
    two steps and only the first realized, ``this_roll = v * unitsize * (realized_hours / horizon_hours)
    = 5.0 * 1.0 * (10 / 20) = 2.5``; combined with a prior carry of 7.0 that is ``7.0 + 2.5 = 9.5``.
    """
    from flextool.engine.preprocess.flex_data import (
        _extract_cum_sim_hours,
        _extract_cumulative_commodity,
    )

    class _NoProvider:
        def has(self, name: str) -> bool:
            return False

        def get(self, name: str):  # pragma: no cover - never called, has() is always False
            raise AssertionError(f"unexpected provider.get({name!r})")

    class _FakeSol:
        def __init__(self, v_trade: pl.DataFrame) -> None:
            self._vars = {"v_trade": True}
            self._v_trade = v_trade

        def value(self, name: str):
            return self._v_trade if name == "v_trade" else None

    from types import SimpleNamespace

    from polar_high import Param

    v_trade = pl.DataFrame({"c": ["coal"], "i": [1], "d": ["period1"], "value": [5.0]})
    flex_data = SimpleNamespace(
        ci_ladder_cumulative=pl.DataFrame({"c": ["coal"], "i": [1]}),
        commodity__tier_ann=pl.DataFrame(schema={"c": pl.Utf8, "i": pl.Int64}),
        p_step_duration=Param(("d", "t"), pl.DataFrame(
            {"d": ["period1", "period1"], "t": ["t1", "t2"], "value": [10.0, 10.0]})),
        realized_dispatch=pl.DataFrame({"period": ["period1"], "step": ["t1"]}),
    )
    prior_handoff = SimpleNamespace(
        cumulative_commodity=pl.DataFrame(
            {"commodity": ["coal"], "tier": [1], "period": ["period1"], "p_ladder_cum_realized_mwh": [7.0]}),
        cum_sim_hours=pl.DataFrame({"period": ["period1"], "p_ladder_cum_sim_hours": [3.0]}),
    )
    sol = _FakeSol(v_trade)
    sd = Path("/nonexistent")

    def _carried_mwh(is_dispatch_sibling_parent: bool) -> float:
        df = _extract_cumulative_commodity(
            sol, sd, prior_handoff=prior_handoff, flex_data=flex_data, provider=_NoProvider(),
            is_dispatch_sibling_parent=is_dispatch_sibling_parent)
        row = df.filter((pl.col("commodity") == "coal") & (pl.col("period") == "period1"))
        return float(row["p_ladder_cum_realized_mwh"][0])

    def _carried_hrs(is_dispatch_sibling_parent: bool) -> float:
        df = _extract_cum_sim_hours(
            sd, prior_handoff=prior_handoff, flex_data=flex_data, provider=_NoProvider(),
            is_dispatch_sibling_parent=is_dispatch_sibling_parent)
        row = df.filter(pl.col("period") == "period1")
        return float(row["p_ladder_cum_sim_hours"][0])

    assert _carried_mwh(False) == pytest.approx(9.5, abs=0.0)
    assert _carried_hrs(False) == pytest.approx(13.0, abs=0.0)
    # The dispatch-sibling parent adds nothing of its own: the carry stays exactly the prior's.
    assert _carried_mwh(True) == pytest.approx(7.0, abs=0.0)
    assert _carried_hrs(True) == pytest.approx(3.0, abs=0.0)


# ---------------------------------------------------------------------------
# Unit tests of the helpers


def _pairs(rows) -> pl.DataFrame:
    return pl.DataFrame(rows, schema={"anchor": pl.Utf8, "br": pl.Utf8}, orient="row")


def test_inherit_anchor_period_helper() -> None:
    from flextool.engine.preprocess.params._branch import inherit_anchor_period

    frame = pl.DataFrame({"c": ["coal", "coal", "gas", "coal"], "i": ["1", "2", "1", "1"],
                          "d": ["p1", "p1", "p1", "p1_hi"], "price": [1.0, 2.0, 3.0, 9.0]})
    pairs = _pairs([("p1", "p1_hi"), ("p1", "p1_lo")])
    out = inherit_anchor_period(frame, pairs, group_keys=("c",))
    got = {(r["c"], r["i"], r["d"]): r["price"] for r in out.iter_rows(named=True)}
    assert got == {
        ("coal", "1", "p1"): 1.0, ("coal", "2", "p1"): 2.0, ("gas", "1", "p1"): 3.0,
        # p1_hi has its own coal curve: kept whole, no anchor tier 2 mixed in; gas is still inherited.
        ("coal", "1", "p1_hi"): 9.0, ("gas", "1", "p1_hi"): 3.0,
        ("coal", "1", "p1_lo"): 1.0, ("coal", "2", "p1_lo"): 2.0, ("gas", "1", "p1_lo"): 3.0,
    }
    assert out.height == len(got)
    assert out.head(frame.height).equals(frame), "authored rows come first, unchanged"
    # Idempotent: a second pass adds nothing.
    assert inherit_anchor_period(out, pairs, group_keys=("c",)).equals(out)
    # No pairs: unchanged.
    assert inherit_anchor_period(frame, None, group_keys=("c",)) is frame
    assert inherit_anchor_period(frame, _pairs([]), group_keys=("c",)) is frame
    # A non-Utf8 period column keeps its dtype.
    enum = pl.Enum(["p1", "p1_hi", "p1_lo"])
    out_enum = inherit_anchor_period(frame.with_columns(pl.col("d").cast(enum)), pairs, group_keys=("c",))
    assert out_enum.schema["d"] == enum and out_enum.height == len(got)


def test_branch_anchor_pairs_helper() -> None:
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
    from flextool.engine.preprocess.params._branch import branch_anchor_pairs

    def provider(rows):
        prov = FlexDataProvider()
        prov.put("solve_data/period__branch",
                 pl.DataFrame(rows, schema={"period": pl.Utf8, "branch": pl.Utf8}, orient="row"))
        return prov

    pb = [("p1", "p1"), ("p1", "p1_hi"), ("p1", "p1_lo"), ("p1", "p1_rlz"), ("p2", "p2")]
    pairs = branch_anchor_pairs(provider(pb), ["p1", "p1_hi", "p1_lo", "p2"])
    # Self rows dropped; a branch period outside the period set (realized / zero weight) gives no pair.
    assert pairs.rows() == [("p1", "p1_hi"), ("p1", "p1_lo")]
    # periods=None (explicit): every branch period, the caller restricts downstream.
    assert branch_anchor_pairs(provider(pb), None).rows() == [("p1", "p1_hi"), ("p1", "p1_lo"), ("p1", "p1_rlz")]
    assert branch_anchor_pairs(provider([("p1", "p1"), ("p2", "p2")]), None) is None
    assert branch_anchor_pairs(None, None) is None
    with pytest.raises(ValueError, match=r"more than one anchor"):
        branch_anchor_pairs(provider(pb + [("p2", "p1_hi")]), None)


def test_branch_weight_per_period() -> None:
    from types import SimpleNamespace

    from polar_high import Param

    from flextool.engine.model._commodity_ladder import _branch_weight

    cnd = pl.DataFrame({"c": ["coal"] * 3, "n": ["m"] * 3, "d": ["p1", "p1_hi", "p9"]})

    def pdt(rows):
        return Param(("d", "t"), pl.DataFrame(rows, schema={"d": pl.Utf8, "t": pl.Utf8, "value": pl.Float64},
                                              orient="row"))

    assert _branch_weight(SimpleNamespace(cnd_ladder=cnd, pdt_branch_weight=None)) is None
    bw = _branch_weight(SimpleNamespace(cnd_ladder=cnd, pdt_branch_weight=pdt(
        [("p1", "t1", 0.0), ("p1", "t2", 0.0), ("p1_hi", "t1", 0.3), ("p1_hi", "t2", 0.3)])))
    # p9 has no dt row (v_trade is 0 there) and gets 1.0.
    assert bw.frame.rows() == [("p1", 0.0), ("p1_hi", 0.3), ("p9", 1.0)]
    with pytest.raises(ValueError, match=r"varies within period\(s\) \['p1_hi'\]"):
        _branch_weight(SimpleNamespace(cnd_ladder=cnd, pdt_branch_weight=pdt(
            [("p1", "t1", 0.0), ("p1_hi", "t1", 0.3), ("p1_hi", "t2", 0.4)])))


# ---------------------------------------------------------------------------
# #59: rolling stochastic fan — a forecast's tier allocation counts its period's realized hours
#
# ``1_week_rolling_wind``: period1 = 168 one-hour steps, rolling horizon 96 h / jump 24 h -> stochastic rolls
# k = 0..6, each fanning into period1_lower / _mid / _upper, each followed by a realization roll that realizes
# its 24 h jump under the label period1.  Stochastic roll k starts with 24k realized hours, so
#     f_d_k[forecast] = (24k + min(96, 168 - 24k)) / 168 = min(24k + 96, 168) / 168
# Gas runs in every realization roll, so the gas ladder carries realized purchases (coal does not run there).
# With the forecast's window-only f_d_k = min(96, 168 - 24k) / 168 and a binding tier, the realized carry
# reaches Q·share·24k/168, so the cap Q·share·(min(96, 168 - 24k) - 24k)/168 went negative from k = 4.

ROLL_FORECASTS = ("period1_lower", "period1_mid", "period1_upper")
ROLL_SOLVE = "1week_rolling"
ROLL_SHARE = 168 / 8760


def _gas_ladder(method: str, quantity1: float) -> list:
    tiers = {k: v for k, v in _tiers(5.0, quantity1, 7.0).items() if k != "type"}
    return [["commodity", "gas", "price_method", b64(method, "str")],
            ["commodity", "gas", method, b64(tiers, "map")]]


def _stochastic_rolls(steps: dict) -> list:
    rolls = [steps[f"{ROLL_SOLVE}_roll_{k}"] for k in range(7)]
    assert len([n for n in steps if "__realized_dispatch" in n]) == 7
    return rolls


def _forecast_f(k: int) -> float:
    return min(24 * k + 96, 168) / 168


def _bought(step, tier: str = "1") -> float:
    """Tier MWh bought in a realization roll (commodity unitsize 1, the default): its whole horizon is realized."""
    rows = _v_trade(step).filter((pl.col("c") == "gas") & (pl.col("i") == tier))
    assert set(rows["d"]) == {"period1"}
    return float(rows["value"].sum())


def test_rolling_forecast_annual_cap_counts_realized_hours(tmp_path_factory) -> None:
    """Each forecast's annual tier cap is Q·share·f_d_k − realized[period1], with f_d_k counting the realized
    hours: the cap never goes negative, and the chain solves to the end of the week."""
    from tests.engine._lp_rows import row_bounds

    quantity = 1.0e6  # 2740 MWh per realized day, below the gas a realization roll burns: tier 1 binds
    steps = _steps(tmp_path_factory, "c59_roll_ann_cap", _gas_ladder("price_ladder_annual", quantity),
                   chain=CHAIN_ROLL)
    realization = [steps[f"{ROLL_SOLVE}__realized_dispatch_roll_{k}"] for k in range(7)]
    for k, step in enumerate(_stochastic_rolls(steps)):
        fd = step.flex_data
        fdk = dict(utf8(fd.p_f_d_k.frame).select("d", "value").iter_rows())
        # the twin (flex_data.p_f_d_k) and the emitted f_d_k.csv (p_co2_f_d_k) agree
        assert fdk == dict(utf8(fd.p_co2_f_d_k.frame).select("d", "value").iter_rows()), k
        share = dict(utf8(fd.p_period_share.frame).select("d", "value").iter_rows())
        realized = sum(_bought(s) for s in realization[:k])
        if k > 0:
            assert realized > 0.0, k
        rows = row_bounds(build(fd), "ladder_tier_cap_annual_roll", "gas,1,")
        assert {nm[nm.rindex(",") + 1:-1] for nm in rows} == set(ROLL_FORECASTS), k
        for b in ROLL_FORECASTS:
            assert fdk[b] == pytest.approx(_forecast_f(k), rel=1e-12), (k, b)
            assert share[b] == pytest.approx(ROLL_SHARE, rel=1e-12), (k, b)
            ub = rows[f"ladder_tier_cap_annual_roll[gas,1,{b}]"][1]
            assert ub == pytest.approx(quantity * ROLL_SHARE * _forecast_f(k) - realized, rel=1e-9), (k, b)
            assert ub >= 0.0, (k, b)


def test_cumulative_ladder_fallback_counts_realized_hours_once(tmp_path_factory) -> None:
    """Without ``p_ladder_cum_weight`` (the derived-param pass skipped) a scenario path's weight is Σ f_d_k over its
    periods with the fan label left out: f_d_k[forecast] alone, which already holds the realized hours.  Adding the
    label's f_d_k (24k/168) as well would count the realized hours twice.

    Hand values, roll k: with p_ladder_cum_weight (years_represented 1) a path weighs the label's realized hours
    plus the forecast's own, (24k + min(96, 168 - 24k)) / 8760; the f_d_k fallback weighs the same hours over the
    period's 168 h, min(24k + 96, 168) / 168 — i.e. 8760/168 times the first."""
    import dataclasses

    from tests.engine._lp_rows import row_bounds

    quantity = 5.0e7
    steps = _steps(tmp_path_factory, "c59_roll_cum_fallback", _gas_ladder("price_ladder_cumulative", quantity),
                   chain=CHAIN_ROLL)
    for k, step in list(enumerate(_stochastic_rolls(steps)))[1:]:
        fd = step.flex_data
        assert fd.p_ladder_cum_weight is not None
        realized = float(fd.p_ladder_cum_realized_mwh.frame.filter(
            (pl.col("c").cast(pl.Utf8) == "gas") & (pl.col("i").cast(pl.Utf8) == "1"))["value"].sum())
        assert realized > 0.0, k

        def weights(flex_data) -> dict[str, float]:
            rows = row_bounds(build(flex_data), "ladder_tier_cap_cumulative_roll_path", "gas,1,")
            assert len(rows) == len(ROLL_FORECASTS), (k, rows)
            return {nm[nm.rindex(",") + 1:-1]: (ub + realized) / quantity for nm, (_lb, ub) in rows.items()}

        with_weight = weights(fd)
        fallback = weights(dataclasses.replace(fd, p_ladder_cum_weight=None))
        assert set(with_weight) == set(fallback) == {b.removeprefix("period1_") for b in ROLL_FORECASTS}, k
        for b in (f.removeprefix("period1_") for f in ROLL_FORECASTS):
            assert with_weight[b] == pytest.approx((24 * k + min(96, 168 - 24 * k)) / 8760, rel=1e-12), (k, b)
            assert fallback[b] == pytest.approx(_forecast_f(k), rel=1e-12), (k, b)
            assert fallback[b] == pytest.approx(with_weight[b] * 8760 / 168, rel=1e-12), (k, b)


# ---------------------------------------------------------------------------
# A rolling fan over two periods: the second period's roll starts after the first period has finished.

TWO_PERIOD_JSON = REPO_ROOT / "tests" / "fixtures" / "stoch_two_period.json"
TWO_PERIOD_CHAIN = ["init", "base", "stoch"]
TWO_PERIOD_FORECASTS = ("high", "low")
FUEL_ENTITIES = [["commodity", "fuel", None], ["node", "fuel_node", None],
                 ["commodity__node", ["fuel", "fuel_node"], None], ["unit__inputNode", ["thermal", "fuel_node"], None]]


def _two_period_rolling_ladder(tmp_path_factory, alt: str, quantity: float) -> dict:
    """``stoch_two_period.json`` as a rolling solve (horizon = jump = 3 h, one roll per period) with a forecast fan
    revealed at the start of each period (realized designation + forecasts high/low), and the thermal unit burning
    ``fuel`` bought on a cumulative ladder (tier 1 at 5 up to ``quantity``, tier 2 at 7 unlimited)."""
    def at(step: str, realized: str, weight: float) -> dict:
        return {"type": "map", "index_type": "str",
                "data": [[step, {"type": "map", "index_type": "str", "data": [[realized, weight]]}]]}

    branches = {"index_type": "str", "data": [
        [period, {"type": "map", "index_type": "str", "data": [
            ["realized", at(step, "yes", 1.0)], ["high", at(step, "no", 1.0)], ["low", at(step, "no", 3.0)]]}]
        for period, step in (("p2035", "t0001"), ("p2040", "t0004"))]}
    tiers = {k: v for k, v in _tiers(5.0, quantity, 7.0).items() if k != "type"}
    values = [["solve", "stoch_2p", "solve_mode", b64("rolling_window", "str")],
              ["solve", "stoch_2p", "rolling_solve_horizon", b64(3.0, "float")],
              ["solve", "stoch_2p", "rolling_solve_jump", b64(3.0, "float")],
              ["solve", "stoch_2p", "stochastic_branches", b64(branches, "map")],
              ["node", "fuel_node", "node_type", b64("commodity", "str")],
              ["commodity", "fuel", "price_method", b64("price_ladder_cumulative", "str")],
              ["commodity", "fuel", "price_ladder_cumulative", b64(tiers, "map")]]
    url, scen = make_db(tmp_path_factory, alt=alt, values=values, entities=FUEL_ENTITIES,
                        base_chain=TWO_PERIOD_CHAIN, fixture_json=TWO_PERIOD_JSON)
    return run(tmp_path_factory, url, scen)


def _hand_d_leaf(flex_data, label: str) -> pl.DataFrame:
    """The scenario paths of a fan at ``label``, as a fan defines them: one leaf per forecast holding the forecast
    and the label."""
    d_dtype = flex_data.p_f_d_k.frame.schema["d"]
    return pl.DataFrame({"d": [d for f in TWO_PERIOD_FORECASTS for d in (f"{label}_{f}", label)],
                         "leaf": [f for f in TWO_PERIOD_FORECASTS for _ in range(2)]}).with_columns(
        pl.col("d").cast(d_dtype))


def test_rolling_path_tier_cap_counts_finished_periods(tmp_path_factory) -> None:
    """A scenario path's cumulative tier cap counts the periods earlier rolls finished: one realized past, shared by
    every path, the same past whose purchases ``realized`` subtracts.

    Hand values (q = 1e5; every period is 3 one-hour steps; years_represented 1, the fixture sets none).  The thermal
    unit burns at least 100 − 60 = 40 MW (demand 100, wind 100 MW × profile ≤ 0.6) for 3 h = 120 MWh of fuel per
    period, above any tier-1 cap below, so tier 1 binds.  Realization roll 0 (p2035 realized) has the cap
    q·3/8760 = 34.247 MWh and buys it all: R_0 = q·3/8760.  Roll 1 is the p2040 fan, with p2035 finished (outside its
    period_in_use).  A path weighs the finished p2035 (3 realized hours: 3/8760), the label p2040 (no hours yet: 0)
    and its forecast's own 3 hours (3/8760): RHS = q·6/8760 − R_0 = q·3/8760.  Without the history the RHS would be
    q·3/8760 − R_0 = 0.  The f_d_k fallback weighs the same periods by the share of their hours realized:
    f_d_k[p2035] = 3/3 = 1 and f_d_k[p2040_<f>] = 3/3 = 1, the label left out: RHS = q·2 − R_0."""
    import dataclasses

    from tests.engine._lp_rows import row_bounds

    quantity = 1.0e5
    steps = _two_period_rolling_ladder(tmp_path_factory, "ladder_two_period_roll", quantity)
    bought0 = _v_trade(steps["stoch_2p__realized_dispatch_roll_0"]).filter(
        (pl.col("c") == "fuel") & (pl.col("i") == "1"))
    assert set(bought0["d"]) == {"p2035"}
    r0 = float(bought0["value"].sum())
    assert r0 == pytest.approx(quantity * 3 / 8760, rel=1e-9)

    fd = steps["stoch_2p_roll_1"].flex_data
    assert "p2035" not in set(utf8(fd.period_in_use_set)["d"])
    assert set(utf8(fd.d_leaf).iter_rows()) == set(utf8(_hand_d_leaf(fd, "p2040")).iter_rows())

    def rhs(flex_data) -> dict[str, float]:
        rows = row_bounds(build(flex_data), "ladder_tier_cap_cumulative_roll_path", "fuel,1,")
        return {nm[nm.rindex(",") + 1:-1]: ub for nm, (_lb, ub) in rows.items()}

    with_weight = rhs(fd)
    fallback = rhs(dataclasses.replace(fd, p_ladder_cum_weight=None))
    assert set(with_weight) == set(fallback) == set(TWO_PERIOD_FORECASTS)
    for f in TWO_PERIOD_FORECASTS:
        assert with_weight[f] == pytest.approx(quantity * 6 / 8760 - r0, rel=1e-9), f
        assert fallback[f] == pytest.approx(quantity * 2 - r0, rel=1e-9), f


# ---------------------------------------------------------------------------
# The rolls of a rolling fan carry their scenario paths (``d_leaf``)


def test_rolling_fan_cumulative_tier_cap_per_forecast_path(tmp_path_factory) -> None:
    """Every stochastic roll of the rolling week keeps one cumulative tier cap per forecast path; before, a roll had
    no scenario paths and pooled the three forecasts into one row whose budget summed all their hours.

    Hand values (q = 1e6, years_represented 1): stochastic roll k starts after 24k realized hours (the realization
    rolls 0..k−1, each realizing its 24 h jump under ``period1``) and its forecasts cover min(96, 168 − 24k) more, so
    each path's RHS is q·(24k + min(96, 168 − 24k))/8760 − R_k, R_k = the tier-1 MWh the realization rolls 0..k−1
    bought (read from their solutions).  Roll 0: q·96/8760 = 10 958.904 MWh.  A realization roll buys more gas than its
    q·24/8760 = 2 739.726 MWh day cap, and a forecast more than its window's, so the rows bind."""
    from tests.engine._lp_rows import row_bounds, row_terms

    quantity = 1.0e6
    steps = _steps(tmp_path_factory, "ladder_roll_cum_paths", _gas_ladder("price_ladder_cumulative", quantity),
                   chain=CHAIN_ROLL)
    realization = [steps[f"{ROLL_SOLVE}__realized_dispatch_roll_{k}"] for k in range(7)]
    family = "ladder_tier_cap_cumulative_roll"
    bought = 0.0
    binding = 0
    for k, step in enumerate(_stochastic_rolls(steps)):
        pb = build(step.flex_data)
        assert row_bounds(pb, family, "") == {}, k
        rows = row_bounds(pb, f"{family}_path", "gas,1,")
        assert set(rows) == {f"{family}_path[gas,1,{f.removeprefix('period1_')}]" for f in ROLL_FORECASTS}, k
        want = quantity * (24 * k + min(96, 168 - 24 * k)) / 8760 - bought
        for nm, (_lb, ub) in rows.items():
            assert ub == pytest.approx(want, rel=1e-9), (k, nm)
        for nm, terms in row_terms(pb, f"{family}_path", "gas,1,").items():
            periods = {col.split(",")[2] for col in terms if col.startswith("v_trade[")}
            assert periods == {f"period1_{nm[nm.rindex(',') + 1:-1]}"}, (k, nm, periods)
        duals = step.solution.constraint_dual(f"{family}_path")["dual"].to_list()
        binding += sum(1 for v in duals if v != 0.0)
        bought += _bought(realization[k])
    assert binding > 0
    for k, step in enumerate(realization):
        pb = build(step.flex_data)
        assert row_bounds(pb, f"{family}_path", "") == {}, k
        assert set(row_bounds(pb, family, "")) == {f"{family}[gas,1]"}, k


def _single_reveal_branches() -> list:
    """The ``1week_rolling`` fan revealed once, at t0001: realized designation + forecasts upper/lower/mid."""
    def at(realized: str) -> dict:
        return {"type": "map", "index_type": "str",
                "data": [["t0001", {"type": "map", "index_type": "str", "data": [[realized, 1.0]]}]]}

    branches = {"index_type": "str", "data": [["period1", {"type": "map", "index_type": "str", "data": [
        ["realized", at("yes")], ["upper", at("no")], ["lower", at("no")], ["mid", at("no")]]}]]}
    return ["solve", ROLL_SOLVE, "stochastic_branches", b64(branches, "map")]


def test_one_roll_fan_equals_single_solve_fan(tmp_path_factory) -> None:
    """A rolling solve of one roll (horizon = jump = 168 h) is the single solve of the same week: the same scenario
    paths, and the same per-path cumulative tier cap and CO2 total cap rows."""
    from tests.engine._lp_rows import row_bounds
    from tests.engine._unit_db import last_stochastic, without_realization

    entities = [["group", "co2_limit", None], ["group__node", ["co2_limit", "gas_market"], None]]
    values = (_gas_ladder("price_ladder_cumulative", 1.0e6) + [_single_reveal_branches()]
              + [["commodity", "gas", "co2_content", b64(0.2, "float")],
                 ["group", "co2_limit", "co2_method", b64("total", "str")],
                 ["group", "co2_limit", "co2_max_total", b64(1.0e7, "float")]])
    one_roll = [["solve", ROLL_SOLVE, "rolling_solve_horizon", b64(168.0, "float")],
                ["solve", ROLL_SOLVE, "rolling_solve_jump", b64(168.0, "float")]]
    single = [["solve", ROLL_SOLVE, "solve_mode", b64("single_solve", "str")]]
    fds, names = {}, {}
    for alt, extra in (("fan_one_roll", one_roll), ("fan_single", single)):
        url, scen = make_db(tmp_path_factory, alt=alt, values=values + extra, entities=entities,
                            base_chain=CHAIN_ROLL, fixture_json=STOCHASTICS_JSON)
        steps = run(tmp_path_factory, url, scen)
        names[alt] = list(without_realization(steps))
        fds[alt] = last_stochastic(steps).flex_data
    assert names == {"fan_one_roll": [f"{ROLL_SOLVE}_roll_0"], "fan_single": [ROLL_SOLVE]}

    def leaves(fd) -> set:
        return set(utf8(fd.d_leaf).iter_rows())

    assert leaves(fds["fan_one_roll"]) == leaves(fds["fan_single"])
    assert {leaf for _d, leaf in leaves(fds["fan_single"])} == {"lower", "mid", "upper"}
    pbs = {alt: build(fd) for alt, fd in fds.items()}
    for family in ("ladder_tier_cap_cumulative_roll_path", "co2_max_total_path"):
        rows = {alt: row_bounds(pb, family, "") for alt, pb in pbs.items()}
        assert len(rows["fan_single"]) == 3, family
        assert set(rows["fan_one_roll"]) == set(rows["fan_single"]), family
        for nm, (_lb, ub) in rows["fan_single"].items():
            assert rows["fan_one_roll"][nm][1] == pytest.approx(ub, rel=1e-12), (family, nm)
