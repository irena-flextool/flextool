"""Stochastic branch periods inherit the period-keyed input values of their anchor period (#55).

A stochastic solve mints its forecast branch periods (``period1_upper``, ``p2035_high``, …) per solve, so
no input value can be keyed by them.  They are possible futures of their anchor period, so a value keyed
by the anchor period applies to them, unless the branch period authors its own value:

    value[k, br] = authored[k, br]            if k has an authored row in br
                 = authored[k, anchor(br)]    otherwise, (anchor, br) ∈ period__branch, anchor ≠ br

Before the fix a period map ``{anchor: v}`` reached only the anchor (a weight-0 phantom trunk in the
2-day vehicle), so the branch periods fell back to the default (efficiency 1, no startup cost, free fuel,
no variable cost, free slack, …).  The criterion here: a period map keyed by the anchor gives the same
parameters and objective as the same value given as a scalar (or keyed explicitly by every branch period).

Vehicles (databases built from JSON, never a checked-in ``.sqlite``):
- ``2_day_stochastic_dispatch`` (``tests/fixtures/stochastics.json``): ``period1`` (the period label: no
  dispatch, so no per-step parameter rows) and four forecast branch periods.
- ``stoch`` (``stoch_two_period.json``): two anchors, ``p2035 → p2035_{high,low}``, ``p2040 → p2040_{…}``.
- ``recourse`` (``stoch_two_period_invest.json``): recourse investment; branch periods are invest periods.
- ``1_week_rolling_wind`` (``stochastics.json``): 7 stochastic rolls.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from tests.engine._unit_db import b64, last_stochastic, make_db, run, utf8

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
STOCHASTICS_JSON = FIXTURES / "stochastics.json"
TWO_PERIOD_JSON = FIXTURES / "stoch_two_period.json"
RECOURSE_JSON = FIXTURES / "stoch_two_period_invest.json"

ANCHOR = "period1"
BRANCH_PERIODS = tuple(f"period1_{b}" for b in ("realized", "upper", "lower", "mid"))
ALL_PERIODS = (ANCHOR,) + BRANCH_PERIODS
T48 = [f"t{i:04d}" for i in range(1, 49)]
T6 = [f"t{i:04d}" for i in range(1, 7)]
TWO_P_BRANCHES = {"p2035": ("p2035_high", "p2035_low"), "p2040": ("p2040_high", "p2040_low")}


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


# ---------------------------------------------------------------------------
# Value / database helpers


def pmap(values: dict) -> list:
    """1-d period map."""
    return b64({"index_type": "str", "index_name": "period",
                "data": [[d, v] for d, v in values.items()]}, "map")


def ptmap(per: dict) -> list:
    """2-d map period -> time -> value."""
    return b64({"index_type": "str", "index_name": "period", "data": [
        [d, {"type": "map", "index_type": "str", "index_name": "time",
             "data": [[t, v] for t, v in tv.items()]}] for d, tv in per.items()]}, "map")


def tmap(tv: dict) -> list:
    return b64({"index_type": "str", "index_name": "time", "data": [[t, v] for t, v in tv.items()]}, "map")


def scalar(v: float) -> list:
    return b64(v, "float")


def _chain(fixture_json: Path, scenario: str) -> list[str]:
    rows = [r for r in json.loads(fixture_json.read_text(encoding="utf-8"))["scenario_alternatives"] if r[0] == scenario]
    nxt = {a: b for _, a, b in rows}
    alt = next(a for a in nxt if a not in set(nxt.values()))
    chain = []
    while alt is not None:
        chain.append(alt)
        alt = nxt.get(alt)
    return chain


def _steps(tmp_path_factory, alt: str, values: list, *, fixture_json: Path = STOCHASTICS_JSON,
           scenario: str = "2_day_stochastic_dispatch") -> dict:
    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=_chain(fixture_json, scenario),
                        fixture_json=fixture_json)
    return run(tmp_path_factory, url, scen)


def _steps_with_entities(tmp_path_factory, alt: str, values: list, entities: list, entity_alts: list, *,
                         fixture_json: Path, scenario: str) -> dict:
    """Like :func:`_steps`, with extra entities (and entity alternatives) written into the fixture JSON."""
    data = json.loads(fixture_json.read_text(encoding="utf-8"))
    data["entities"].extend(entities)
    data["entity_alternatives"].extend(entity_alts)
    path = tmp_path_factory.mktemp(f"{alt}_json") / fixture_json.name
    path.write_text(json.dumps(data), encoding="utf-8")
    return _steps(tmp_path_factory, alt, values, fixture_json=path, scenario=scenario)


def _last(steps: dict):
    """The stochastic solve: the branch periods live there, not in the realization solve after it."""
    return last_stochastic(steps)


def _frame(x) -> pl.DataFrame:
    assert x is not None, "parameter is None"
    f = getattr(x, "frame", x)
    if isinstance(f, pl.LazyFrame):
        f = f.collect()
    return utf8(f)


def _by_period(x, **keys) -> dict[str, set[float]]:
    """``d -> {values}`` of the rows matching ``keys``."""
    f = _frame(x)
    for col, val in keys.items():
        f = f.filter(pl.col(col) == val)
    out: dict[str, set[float]] = {}
    for d, v in f.select("d", "value").iter_rows():
        out.setdefault(d, set()).add(round(float(v), 12))
    return out


def _assert_every_period(x, value: float, periods=ALL_PERIODS, **keys) -> None:
    got = _by_period(x, **keys)
    for d in periods:
        assert got.get(d) == {value}, f"{keys} period {d}: expected {{{value}}}, got {got.get(d)} (all: {got})"


def _assert_dispatch_periods(x, value: float, periods=BRANCH_PERIODS, labels=(ANCHOR,), **keys) -> None:
    """An operational (per-step) parameter: the value in every forecast branch period, no row at a period label
    (the label has no dispatch)."""
    _assert_every_period(x, value, periods=periods, **keys)
    got = _by_period(x, **keys)
    for label in labels:
        assert label not in got, f"{keys}: period label {label} has rows {got[label]}"


def _periods_of(frame, **keys) -> set[str]:
    f = _frame(frame)
    for col, val in keys.items():
        f = f.filter(pl.col(col) == val)
    return set(f["d"].to_list())


# ---------------------------------------------------------------------------
# 2-day vehicle: every family as an anchor-keyed period map, and the same values as scalars


def _values_2day(as_map: bool) -> list:
    def v(x):
        return pmap({ANCHOR: x}) if as_map else scalar(x)

    gas_avail = ptmap({ANCHOR: {t: 0.7 for t in T48}}) if as_map else scalar(0.7)
    return [
        ["unit", "coal_plant", "startup_method", b64("linear", "str")],
        ["unit", "coal_plant", "efficiency", v(0.25)],
        ["unit", "coal_plant", "startup_cost", v(7000.0)],
        ["unit", "coal_plant", "availability", v(0.8)],
        ["unit", "coal_plant", "min_load", v(0.3)],
        ["unit", "coal_plant", "efficiency_at_min_load", v(0.2)],
        ["commodity", "coal", "price", v(17.0)],
        ["unit__outputNode", ["coal_plant", "demand_node"], "other_operational_cost", v(3.0)],
        ["unit__inputNode", ["gas_plant", "gas_market"], "other_operational_cost", v(2.0)],
        ["node", "hydro_reservoir", "availability", v(0.9)],
        ["node", "hydro_reservoir", "storage_state_reference_value", v(0.4)],
        ["unit", "gas_plant", "availability", gas_avail],
        ["group", "add_stochastics", "co2_price", v(50.0)],
        ["group", "add_stochastics", "co2_max_period", v(1e6)],
        ["group", "add_stochastics", "capacity_margin", v(10.0)],
        ["group", "add_stochastics", "inertia_limit", v(11.0)],
        ["group", "add_stochastics", "non_synchronous_limit", v(0.9)],
        ["group", "add_stochastics", "penalty_non_synchronous", v(77.0)],
    ]


@pytest.fixture(scope="module")
def maps_2day(tmp_path_factory):
    return _steps(tmp_path_factory, "c6_maps", _values_2day(as_map=True))


@pytest.fixture(scope="module")
def scalars_2day(tmp_path_factory):
    return _steps(tmp_path_factory, "c6_scalars", _values_2day(as_map=False))


def test_efficiency_map_reaches_branches(maps_2day) -> None:
    d = _last(maps_2day).flex_data
    _assert_dispatch_periods(d.p_slope, 4.0, p="coal_plant")


def test_startup_cost_map_reaches_branches_and_online_domain(maps_2day) -> None:
    d = _last(maps_2day).flex_data
    _assert_dispatch_periods(d.p_startup_cost, 7000.0, p="coal_plant")
    # The startup-cost objective term runs over pdt_online_linear, built in _load_online from the same
    # startup-cost read (before the DB-direct pass): every branch period must be in it.
    assert _periods_of(d.pdt_online_linear, p="coal_plant") == set(BRANCH_PERIODS)


def test_commodity_price_map_reaches_branches(maps_2day) -> None:
    d = _last(maps_2day).flex_data
    _assert_dispatch_periods(d.p_commodity_price, 17.0, c="coal")


def test_variable_cost_map_reaches_branches_and_lp_domain(maps_2day) -> None:
    d = _last(maps_2day).flex_data
    _assert_dispatch_periods(d.p_pdt_varCost_sink, 3.0, p="coal_plant", sink="demand_node")
    _assert_dispatch_periods(d.p_pdt_varCost_source, 2.0, p="gas_plant", source="gas_market")
    _assert_dispatch_periods(d.p_pssdt_varCost, 3.0, p="coal_plant", sink="demand_node")
    _assert_dispatch_periods(d.p_pssdt_varCost, 2.0, p="gas_plant", source="gas_market")
    # The LP domain of the variable-cost terms comes from the emit layer (pdtProcess_source / _sink).
    assert _periods_of(d.pssdt_varCost_eff_unit_sink, p="coal_plant") == set(BRANCH_PERIODS)
    assert _periods_of(d.pssdt_varCost_eff_unit_source, p="gas_plant") == set(BRANCH_PERIODS)


def test_availability_maps_reach_branches(maps_2day) -> None:
    d = _last(maps_2day).flex_data
    _assert_dispatch_periods(d.p_process_availability, 0.8, p="coal_plant")
    _assert_dispatch_periods(d.p_process_availability, 0.7, p="gas_plant")  # 2-d period x time map
    _assert_dispatch_periods(d.p_node_availability, 0.9, n="hydro_reservoir")


def test_min_load_and_reference_value_maps_reach_branches(maps_2day) -> None:
    d = _last(maps_2day).flex_data
    _assert_dispatch_periods(d.p_min_load, 0.3, p="coal_plant")  # consumed: coal_plant is an online unit
    _assert_dispatch_periods(d.p_storage_state_reference_value, 0.4, n="hydro_reservoir")


def test_group_period_maps_reach_branches(maps_2day) -> None:
    d = _last(maps_2day).flex_data
    g = {"g": "add_stochastics"}
    _assert_dispatch_periods(d.p_co2_price, 50.0, **g)
    _assert_dispatch_periods(d.p_co2_max_period, 1e6, **g)
    _assert_dispatch_periods(d.pdGroup_capacity_margin, 10.0, **g)
    _assert_dispatch_periods(d.pdtGroup_inertia_limit, 11.0, **g)
    _assert_dispatch_periods(d.pdtGroup_non_synchronous_limit, 0.9, **g)
    _assert_dispatch_periods(d.pdGroup_penalty_non_synchronous, 77.0, **g)


def test_period_maps_equal_scalars_single_anchor(maps_2day, scalars_2day) -> None:
    obj_map = _last(maps_2day).solution.obj
    obj_scalar = _last(scalars_2day).solution.obj
    assert obj_map == pytest.approx(obj_scalar, rel=1e-9)


# ---------------------------------------------------------------------------
# 2-day vehicle: the profile readers bypass _param_shapes (2-d period x time and 1-d period maps)

WIND_CURVE = {t: round(0.2 + 0.01 * i, 6) for i, t in enumerate(T48)}


@pytest.fixture(scope="module")
def curves_2day(tmp_path_factory):
    return _steps(tmp_path_factory, "c6_curves", [
        ["profile", "wind", "profile", ptmap({ANCHOR: WIND_CURVE})],
    ])


@pytest.fixture(scope="module")
def curves_2day_ref(tmp_path_factory):
    return _steps(tmp_path_factory, "c6_curves_ref", [
        ["profile", "wind", "profile", tmap(WIND_CURVE)],
    ])


def test_profile_period_map_reaches_branches(curves_2day) -> None:
    f = _frame(_last(curves_2day).flex_data.p_profile_value).filter(pl.col("f") == "wind")
    for d in BRANCH_PERIODS:
        got = dict(f.filter(pl.col("d") == d).select("t", "value").iter_rows())
        assert got == pytest.approx(WIND_CURVE), f"wind profile in {d}"
    assert f.filter(pl.col("d") == ANCHOR).height == 0, "the period label has no dispatch, so no profile rows"


def test_period_curves_equal_time_curves(curves_2day, curves_2day_ref) -> None:
    assert _last(curves_2day).solution.obj == pytest.approx(_last(curves_2day_ref).solution.obj, rel=1e-9)


@pytest.fixture(scope="module")
def profile_1d_2day(tmp_path_factory):
    return _steps(tmp_path_factory, "c6_prof1d", [["profile", "wind", "profile", pmap({ANCHOR: 0.35})]])


def test_profile_1d_period_map_reaches_branches(profile_1d_2day) -> None:
    f = _frame(_last(profile_1d_2day).flex_data.p_profile_value).filter(pl.col("f") == "wind")
    for d in BRANCH_PERIODS:
        vals = set(f.filter(pl.col("d") == d)["value"].to_list())
        assert vals == {0.35}, f"wind profile in {d}: {vals}"
    assert f.filter(pl.col("d") == ANCHOR).height == 0, "the period label has no dispatch, so no profile rows"


# ---------------------------------------------------------------------------
# Invest-side per-period readers inherit branch values unconditionally (#63): a period map on
# ``existing`` / ``fixed_cost`` keyed by the anchor must give the same ``p_entity_all_existing``
# in every branch period, and the same existing-fixed-cost objective constant, as the equivalent
# scalar.  Before the fix, only ``period1`` (the weight-0 phantom trunk) saw the map's value: every
# branch fell back to 0 (``p_entity_all_existing``) and the objective constant was lost entirely
# (the only ``p_ed_fixed_cost`` row left was the one with ``pd_branch_weight`` 0).


def _values_existing_fixed_cost(as_map: bool) -> list:
    def v(x):
        return pmap({ANCHOR: x}) if as_map else scalar(x)

    return [
        ["unit", "coal_plant", "existing", v(1500.0)],
        ["unit", "coal_plant", "fixed_cost", v(3.0)],
    ]


@pytest.fixture(scope="module")
def existing_fc_maps(tmp_path_factory):
    return _steps(tmp_path_factory, "c63_existing_maps", _values_existing_fixed_cost(as_map=True))


@pytest.fixture(scope="module")
def existing_fc_scalars(tmp_path_factory):
    return _steps(tmp_path_factory, "c63_existing_scalars", _values_existing_fixed_cost(as_map=False))


def test_existing_fixed_cost_period_map_equals_scalar(existing_fc_maps, existing_fc_scalars) -> None:
    from polar_high import Problem

    from flextool.engine import build_flextool

    d_map = _last(existing_fc_maps).flex_data
    d_scalar = _last(existing_fc_scalars).flex_data

    # p_entity_all_existing: the map must reach every branch period, exactly like the scalar.
    _assert_every_period(d_map.p_entity_all_existing, 1500.0, e="coal_plant")
    assert _by_period(d_map.p_entity_all_existing, e="coal_plant") == \
        _by_period(d_scalar.p_entity_all_existing, e="coal_plant")

    # The existing-fixed-cost objective constant (opt-in; off by default in the chain run) must
    # also match the scalar-authored run — it sums exist[e,d] * fixed_cost[e,d] * inflation[d] *
    # branch_weight[d] over every (e, d), so a branch missing its fixed_cost row changes it.
    def _obj_with_existing_fixed_cost(fd) -> float:
        pb = Problem()
        build_flextool(pb, fd, include_existing_fixed_cost=True)
        sol = pb.solve()
        assert sol.optimal
        return sol.obj

    obj_map = _obj_with_existing_fixed_cost(d_map)
    obj_scalar = _obj_with_existing_fixed_cost(d_scalar)
    assert obj_map == pytest.approx(obj_scalar, rel=1e-9)


# ---------------------------------------------------------------------------
# Two anchors (stoch_two_period): each branch period takes its own anchor's value


def _values_two_p(expand_branches: bool) -> list:
    def m(per: dict) -> dict:
        if not expand_branches:
            return per
        out = {}
        for a, v in per.items():
            out[a] = v
            for br in TWO_P_BRANCHES[a]:
                out[br] = v
        return out

    return [
        ["unit", "thermal", "efficiency", pmap(m({"p2035": 0.5, "p2040": 0.25}))],
        ["unit", "thermal", "startup_method", b64("linear", "str")],
        ["unit", "thermal", "min_load", pmap(m({"p2035": 0.2, "p2040": 0.3}))],
        ["unit", "thermal", "startup_cost", pmap(m({"p2035": 7.0, "p2040": 9.0}))],
        ["unit__outputNode", ["thermal", "city"], "other_operational_cost", pmap(m({"p2035": 40.0, "p2040": 60.0}))],
        ["node", "city", "penalty_up", pmap(m({"p2035": 8000.0, "p2040": 9000.0}))],
        ["unit", "thermal", "availability",
         ptmap(m({"p2035": {t: 0.9 for t in T6[:3]}, "p2040": {t: 0.8 for t in T6[3:]}}))],
    ]


@pytest.fixture(scope="module")
def two_p_maps(tmp_path_factory):
    return _steps(tmp_path_factory, "c6_2p", _values_two_p(False), fixture_json=TWO_PERIOD_JSON, scenario="stoch")


@pytest.fixture(scope="module")
def two_p_full(tmp_path_factory):
    return _steps(tmp_path_factory, "c6_2p_full", _values_two_p(True), fixture_json=TWO_PERIOD_JSON,
                  scenario="stoch")


def test_penalty_map_reaches_branches(two_p_maps) -> None:
    d = _last(two_p_maps).flex_data
    _assert_dispatch_periods(d.p_penalty_up, 8000.0, periods=TWO_P_BRANCHES["p2035"], labels=("p2035",), n="city")
    _assert_dispatch_periods(d.p_penalty_up, 9000.0, periods=TWO_P_BRANCHES["p2040"], labels=("p2040",), n="city")


def test_period_maps_two_anchors(two_p_maps, two_p_full) -> None:
    d = _last(two_p_maps).flex_data
    _assert_dispatch_periods(d.p_slope, 2.0, periods=TWO_P_BRANCHES["p2035"], labels=("p2035",), p="thermal")
    _assert_dispatch_periods(d.p_slope, 4.0, periods=TWO_P_BRANCHES["p2040"], labels=("p2040",), p="thermal")
    _assert_dispatch_periods(d.p_startup_cost, 9.0, periods=TWO_P_BRANCHES["p2040"], labels=("p2040",), p="thermal")
    assert _last(two_p_maps).solution.obj == pytest.approx(_last(two_p_full).solution.obj, rel=1e-9)


@pytest.fixture(scope="module")
def two_p_native(tmp_path_factory):
    return _steps(tmp_path_factory, "c6_2p_native", [
        ["unit", "thermal", "efficiency", pmap({"p2035": 0.5, "p2035_high": 0.8, "p2040": 0.25})],
        ["unit", "thermal", "startup_method", b64("linear", "str")],
        ["unit", "thermal", "min_load", scalar(0.2)],
        ["unit", "thermal", "startup_cost", pmap({"p2035": 7.0, "p2035_high": 0.0, "p2040": 9.0})],
    ], fixture_json=TWO_PERIOD_JSON, scenario="stoch")


def test_branch_native_value_wins(two_p_native) -> None:
    d = _last(two_p_native).flex_data
    slope = _by_period(d.p_slope, p="thermal")
    assert slope["p2035_high"] == {1.25}
    assert slope["p2035_low"] == {2.0}
    assert "p2035" not in slope  # the period label has no dispatch
    # An authored 0 in a branch period is that period's value (then dropped by the zero filter of the
    # startup-cost reader); it is not replaced by the anchor's 7.
    startup = _by_period(d.p_startup_cost, p="thermal")
    assert "p2035_high" not in startup
    assert startup["p2035_low"] == {7.0}
    assert startup["p2040_high"] == {9.0}


# ---------------------------------------------------------------------------
# Rolling (synthetic sub-solves; p_penalty_up, the conversion slope and the variable-cost seeds come from the emit
# layer; the profile is read DB-direct).

CHAIN_ROLL_SCENARIO = "1_week_rolling_wind"


def _values_roll(as_map: bool) -> list:
    def v(x):
        return pmap({ANCHOR: x}) if as_map else scalar(x)

    return [
        ["unit", "coal_plant", "startup_method", b64("linear", "str")],
        ["unit", "coal_plant", "startup_cost", v(7000.0)],
        ["unit", "coal_plant", "min_load", v(0.3)],
        ["commodity", "coal", "price", v(17.0)],
        ["unit__outputNode", ["coal_plant", "demand_node"], "other_operational_cost", v(3.0)],
        ["unit__inputNode", ["gas_plant", "gas_market"], "other_operational_cost", v(2.0)],
        ["node", "demand_node", "penalty_up", v(4000.0)],
        ["unit", "coal_plant", "availability", v(0.8)],
        ["unit", "coal_plant", "efficiency", v(0.5)],
        ["unit", "hydro_plant", "efficiency", v(1.5)],
    ]


def test_rolling_period_maps_equal_scalars(tmp_path_factory) -> None:
    maps = _steps(tmp_path_factory, "c6_roll_maps", _values_roll(True), scenario=CHAIN_ROLL_SCENARIO)
    scalars = _steps(tmp_path_factory, "c6_roll_scalars", _values_roll(False), scenario=CHAIN_ROLL_SCENARIO)
    assert len(maps) == len(scalars) > 1
    for (name, s_map), s_scalar in zip(maps.items(), scalars.values()):
        slope = _by_period(s_map.flex_data.p_slope, p="coal_plant")
        assert slope and all(v == {2.0} for v in slope.values()), (name, slope)
        assert s_map.solution.obj == pytest.approx(s_scalar.solution.obj, rel=1e-9), name


def test_rolling_profile_period_map_equals_scalar(tmp_path_factory) -> None:
    def values(as_map: bool) -> list:
        return [["profile", "wind", "profile", pmap({ANCHOR: 0.35}) if as_map else scalar(0.35)]]

    maps = _steps(tmp_path_factory, "c8_roll_profile_map", values(True), scenario=CHAIN_ROLL_SCENARIO)
    scalars = _steps(tmp_path_factory, "c8_roll_profile_scalar", values(False), scenario=CHAIN_ROLL_SCENARIO)
    assert len(maps) == len(scalars) > 1
    for (name, s_map), s_scalar in zip(maps.items(), scalars.values()):
        assert s_map.solution.obj == pytest.approx(s_scalar.solution.obj, rel=1e-9), name


# ---------------------------------------------------------------------------
# Recourse: branch periods are invest periods, so an invest-side period cap now applies there too.
# The reference keys the same cap explicitly by every branch period.  (A scalar is not a usable
# reference: a scalar invest_max_period gives no p_entity_max_units row, deterministic too, and the
# group cap's row domain gd_invest_period has no branch periods under recourse; both are separate
# issues in specs/github_issues.md.)

RECOURSE_INVEST = [
    ["unit", "peaker", "invest_method", b64("invest_period", "str")],
    ["group", "invg", "invest_method", b64("invest_period", "str")],
]
RECOURSE_ENTITIES = [["group", "invg", None], ["group__unit", ["invg", "peaker"], None]]
PEAKER_CAP = 20.0
GROUP_CAP = 15.0


def _recourse_steps(tmp_path_factory, alt: str, expand_branches: bool) -> dict:
    def v(x):
        periods = [p for a, brs in TWO_P_BRANCHES.items() for p in ((a, *brs) if expand_branches else (a,))]
        return pmap({p: x for p in periods})

    values = RECOURSE_INVEST + [
        ["unit", "peaker", "invest_max_period", v(PEAKER_CAP)],
        ["group", "invg", "invest_max_period", v(GROUP_CAP)],
    ]
    return _steps_with_entities(tmp_path_factory, alt, values, RECOURSE_ENTITIES, [],
                                fixture_json=RECOURSE_JSON, scenario="recourse")


def test_recourse_invest_period_caps_reach_branches(tmp_path_factory) -> None:
    maps = _recourse_steps(tmp_path_factory, "c6_rec_maps", False)
    explicit = _recourse_steps(tmp_path_factory, "c6_rec_explicit", True)
    d = _last(maps).flex_data
    for anchor, branches in TWO_P_BRANCHES.items():
        _assert_every_period(d.ed_invest_max_period, PEAKER_CAP, periods=(anchor,) + branches, e="peaker")
        _assert_every_period(d.p_group_invest_max_period, GROUP_CAP, periods=(anchor,) + branches, g="invg")
    assert _last(maps).solution.obj == pytest.approx(_last(explicit).solution.obj, rel=1e-9)


# ---------------------------------------------------------------------------
# #67 — ``gd_invest_period`` (the row domain of ``maxInvestGroup_entity_period_p``) lacked
# branch-period rows under recourse, so the group's own per-period invest cap (15, tighter than
# the entity's own cap of 20) never bound there: peaker invested up to its entity cap (20) in
# every branch period instead of the group's 15.  Demand is large enough, and ``invest_cost``
# cheap enough, that peaker wants to invest at its cap in every invest period.


def test_recourse_group_invest_period_cap_binds_in_branches(tmp_path_factory) -> None:
    from tests.engine._unit_db import unitsize

    step = _last(_recourse_steps(tmp_path_factory, "c67_group_cap", False))
    us = unitsize(step, "peaker")
    rows = utf8(step.solution.value("v_invest_p")).filter(pl.col("p") == "peaker")
    assert rows.height > 0, "no v_invest_p rows for peaker"
    invested = {r["d"]: r["value"] * us for r in rows.iter_rows(named=True)}
    for anchor, branches in TWO_P_BRANCHES.items():
        # Under recourse a period label invests nothing here-and-now: no column, every branch invests itself.
        assert anchor not in invested, f"recourse: period label {anchor} has a v_invest_p column"
        for period in branches:
            assert period in invested, f"no v_invest_p row for {period}"
            assert invested[period] <= GROUP_CAP + 1e-6, (
                f"{period}: peaker invested {invested[period]}, exceeding the group's "
                f"per-period cap of {GROUP_CAP} (entity cap alone is {PEAKER_CAP})")


# ---------------------------------------------------------------------------
# Helpers


def _pairs(rows) -> pl.DataFrame:
    return pl.DataFrame(rows, schema={"anchor": pl.Utf8, "br": pl.Utf8}, orient="row")


def _provider(frames: dict):
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider

    prov = FlexDataProvider()
    for key, df in frames.items():
        prov.put(key, df)
    return prov


def test_inherit_anchor_period_validates_pairs() -> None:
    from flextool.engine.preprocess.params._branch import inherit_anchor_period

    frame = pl.DataFrame({"e": ["a"], "d": ["p1"], "value": [1.0]})
    with pytest.raises(ValueError, match="more than one anchor"):
        inherit_anchor_period(frame, _pairs([("p1", "x"), ("p2", "x")]), group_keys=("e",))
    with pytest.raises(ValueError, match="lacks column"):
        inherit_anchor_period(frame, pl.DataFrame({"anchor": ["p1"]}), group_keys=("e",))
    with pytest.raises(ValueError, match="null"):
        inherit_anchor_period(frame, _pairs([("p1", None)]), group_keys=("e",))
    # Self rows only: nothing to do, same object.
    assert inherit_anchor_period(frame, _pairs([("p1", "p1")]), group_keys=("e",)) is frame


def test_inherit_anchor_period_accepts_enum_pairs() -> None:
    from flextool.engine.preprocess.params._branch import inherit_anchor_period

    enum = pl.Enum(["p1", "p1_hi"])
    pairs = _pairs([("p1", "p1_hi")]).with_columns(pl.col("anchor").cast(enum), pl.col("br").cast(enum))
    frame = pl.DataFrame({"e": ["a"], "d": ["p1"], "value": [1.0]})
    out = inherit_anchor_period(frame, pairs, group_keys=("e",))
    assert out.rows() == [("a", "p1", 1.0), ("a", "p1_hi", 1.0)]


def test_inherit_anchor_period_null_group_key_blocks() -> None:
    from flextool.engine.preprocess.params._branch import inherit_anchor_period

    frame = pl.DataFrame({"e": [None, None], "d": ["p1", "p1_hi"], "value": [1.0, 5.0]})
    out = inherit_anchor_period(frame, _pairs([("p1", "p1_hi")]), group_keys=("e",))
    assert out.rows() == [(None, "p1", 1.0), (None, "p1_hi", 5.0)]


def test_branch_anchor_pairs_requires_periods() -> None:
    from flextool.engine.preprocess.params._branch import branch_anchor_pairs

    prov = _provider({"solve_data/period__branch": pl.DataFrame(
        {"period": ["p1", "p1"], "branch": ["p1", "p1_hi"]})})
    with pytest.raises(TypeError):
        branch_anchor_pairs(prov)  # type: ignore[call-arg]
    assert branch_anchor_pairs(prov, None).rows() == [("p1", "p1_hi")]


def test_solve_branch_anchor_pairs_needs_period_in_use() -> None:
    from flextool.engine.preprocess.params._branch import solve_branch_anchor_pairs

    pb = pl.DataFrame({"period": ["p1", "p1", "p1"], "branch": ["p1", "p1_hi", "p1_rlz"]})
    with pytest.raises(ValueError, match="period_in_use_set"):
        solve_branch_anchor_pairs(_provider({"solve_data/period__branch": pb}))
    prov = _provider({"solve_data/period__branch": pb,
                      "solve_data/period_in_use_set": pl.DataFrame({"period": ["p1", "p1_hi"]})})
    # The realized-designated branch period is not in use: no pair.
    assert solve_branch_anchor_pairs(prov).rows() == [("p1", "p1_hi")]
    # Deterministic: only self rows.
    assert solve_branch_anchor_pairs(_provider({"solve_data/period__branch": pl.DataFrame(
        {"period": ["p1"], "branch": ["p1"]})})) is None


def test_solve_branch_anchor_pairs_checks_d_vocabulary() -> None:
    from flextool.engine.input._axis_enums import reset_global_axis_enums, set_global_axis_enums
    from flextool.engine.preprocess.params._branch import solve_branch_anchor_pairs

    prov = _provider({"solve_data/period__branch": pl.DataFrame({"period": ["p1", "p1"], "branch": ["p1", "p1_hi"]}),
                      "solve_data/period_in_use_set": pl.DataFrame({"period": ["p1", "p1_hi"]})})
    tok = set_global_axis_enums({"d": pl.Enum(["p1"])})
    try:
        with pytest.raises(ValueError, match="vocabulary"):
            solve_branch_anchor_pairs(prov)
    finally:
        reset_global_axis_enums(tok)
    tok = set_global_axis_enums({"d": pl.Enum(["p1", "p1_hi"])})
    try:
        assert solve_branch_anchor_pairs(prov).rows() == [("p1", "p1_hi")]
    finally:
        reset_global_axis_enums(tok)


def test_inherit_pd_branch_periods() -> None:
    from flextool.engine.preprocess.emit._pdt_lookup import inherit_pd_branch_periods

    pd = {("u", "eff", "d1"): 0.5, ("u", "oc", "d1"): 3.0, ("u", "oc", "d1_b"): 9.0}
    kw = {"key_cols": ["e", "param", "d"], "period_pos": 2}
    # Deterministic: self rows only -> same object.
    assert inherit_pd_branch_periods(pd, {"d1": ["d1"]}, periods=["d1"], **kw) is pd
    # A branch period outside ``periods`` gets nothing.
    assert inherit_pd_branch_periods(pd, {"d1_b": ["d1"]}, periods=["d1"], **kw) is pd
    out = inherit_pd_branch_periods(pd, {"d1": ["d1"], "d1_b": ["d1"]}, periods=["d1", "d1_b"], **kw)
    assert out == {**pd, ("u", "eff", "d1_b"): 0.5}  # branch-native oc kept
    assert list(out)[:3] == list(pd), "authored entries first"
    # Multi-parent pairs whose anchors have no pd rows are ignored ...
    assert inherit_pd_branch_periods(pd, {"dc": ["pA", "pB"]}, periods=["dc"], **kw) is pd
    # ... a relevant two-anchor conflict raises.
    pd2 = {**pd, ("u", "eff", "d2"): 0.7}
    with pytest.raises(ValueError, match="more than one anchor"):
        inherit_pd_branch_periods(pd2, {"dc": ["d1", "d2"]}, periods=["dc"], **kw)


def test_no_active_pairs_after_stochastic_load(maps_2day) -> None:
    from flextool.engine.preprocess.params._branch import active_branch_anchor_pairs

    assert maps_2day  # a stochastic load_flextool has run in this process
    assert active_branch_anchor_pairs() is None


def test_no_active_pairs_after_failed_stochastic_load(tmp_path_factory, monkeypatch) -> None:
    """A load that raises after the pairs are armed must still reset them (outer finally)."""
    from flextool.engine.preprocess import flex_data as fd_mod
    from flextool.engine.preprocess.params import _branch

    seen = {}

    def boom(*args, **kwargs):
        seen["pairs"] = _branch.active_branch_anchor_pairs()
        raise RuntimeError("c6 injected failure")

    monkeypatch.setattr(fd_mod._commodity_ladder, "inherit_branch_periods", boom)
    url, scen = make_db(tmp_path_factory, alt="c6_fail", values=[],
                        base_chain=_chain(STOCHASTICS_JSON, "2_day_stochastic_dispatch"),
                        fixture_json=STOCHASTICS_JSON)
    with pytest.raises(RuntimeError, match="c6 injected failure"):  # not an earlier, unrelated error
        run(tmp_path_factory, url, scen)
    assert seen.get("pairs") is not None and seen["pairs"].height == len(BRANCH_PERIODS), seen
    assert _branch.active_branch_anchor_pairs() is None
