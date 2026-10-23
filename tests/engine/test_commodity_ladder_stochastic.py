"""Commodity price ladders in stochastic solves (#32).

#32: a ladder purchase is an operating cost, so in a stochastic solve it carries the probability of its
period's branch (``pdt_branch_weight``, the weight in ``op_factor``), like every other operating cost.
Before the fix every branch counted in full.

Vehicle: ``2_day_stochastic_dispatch`` of ``tests/fixtures/stochastics.json`` (one real period
``period1`` = the weight-0 phantom trunk, four forecast branch periods), coal moved to a ladder.  A
ladder whose cheap tier is the flat price (16) and unlimited must cost exactly what the flat price
costs.  Databases are built from the JSON fixture (never a checked-in ``.sqlite``); the stochastic
branch weights are rewritten in the ``stochastic_branches`` map for the skewed cases.
"""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import polars as pl
import pytest

from tests.engine._unit_db import b64, build, make_db, run, utf8

REPO_ROOT = Path(__file__).resolve().parents[2]
STOCHASTICS_JSON = REPO_ROOT / "tests" / "fixtures" / "stochastics.json"
CHAIN_2DAY = ["base", "init", "system", "2day"]
CHAIN_ROLL = ["init", "base", "system", "1week_rolling"]
BRANCHES = ("realized", "upper", "lower", "mid")
BRANCH_PERIODS = tuple(f"period1_{b}" for b in BRANCHES)
ALL_PERIODS = ("period1",) + BRANCH_PERIODS
FLAT_PRICE = 16.0
# Input weights 1/2/3/4: the four forecasts renormalise to 0.1/0.2/0.3/0.4; the trunk is a weight-0 phantom.
SKEWED = {"realized": 1.0, "upper": 2.0, "lower": 3.0, "mid": 4.0}
SKEWED_BW = {"period1": 0.0, "period1_realized": 0.1, "period1_upper": 0.2,
             "period1_lower": 0.3, "period1_mid": 0.4}
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
    """coal on the annual ladder: ``2d`` (no period index), ``3d_trunk`` (keyed by ``period1`` only) or
    ``3d_all`` (keyed by ``period1`` and every branch period name)."""
    if form == "2d":
        value = {k: v for k, v in tiers.items() if k != "type"}
    else:
        periods = ("period1",) if form == "3d_trunk" else ALL_PERIODS
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


# ---------------------------------------------------------------------------
# #32: a ladder at the flat price costs the flat price


@pytest.mark.parametrize("weights", list(WEIGHTS))
@pytest.mark.parametrize("form", ["3d_all"])
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
    assert seen == set(ALL_PERIODS)


# ---------------------------------------------------------------------------
# Unit tests of the helpers


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
