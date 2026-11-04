"""Existing capacity reaches its end of life inside the horizon; retiring part of it before then.

Fixture ``tests.json`` (built from JSON) with the alternatives of ``test_chain_costs``: coal_plant has 500 MW
existing (one unit of 500 MW), fixed_cost 100 CUR/kW/yr, discount_rate 0.05; four periods p2020 … p2035 of 5
years each, so the run years are Y = 0, 5, 10, 15 and the horizon ends at 20.  SINGLE solves all four periods at
once, CHAIN (``invest_solveSequence_5weeks``) solves y2020 … y2035, each its own period and the next one.

End of life (``reinvest_choice`` / ``no_investment``): the run year of the run's first period plus ``lifetime``,
one value per entity, in every solve of the run.  Lifetime 15: EOL = 0 + 15 = 15, so the existing 500 MW operate
in p2020, p2025, p2030 (Y < 15) and are gone in p2035 (Y = 15 ≥ 15).
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._unit_db import FLEXTOOL_ROOT, b64, make_db, run, utf8, without_realization
from tests.engine.test_chain_costs import (
    F_OPS,
    FIXED,
    MEGA,
    PERIODS,
    _assert_same_results,
    _no_battery_invest,
    _results,
)
from tests.engine.test_user_constraint_capacity_terms import CHAIN, PLANT, SINGLE

FAN = FLEXTOOL_ROOT / "tests" / "fixtures" / "stoch_two_period_invest.json"


def _ed(param, entity: str = PLANT) -> dict[str, float]:
    frame = utf8(param.frame).filter(pl.col("e") == entity)
    return dict(zip(frame["d"].to_list(), frame["value"].to_list()))


def _lifetime(method: str, lifetime: float, entity: str = PLANT) -> list:
    return [["unit", entity, "lifetime", b64(lifetime, "float")],
            ["unit", entity, "lifetime_method", b64(method, "str")]]


NO_INVEST = [["unit", PLANT, "invest_method", b64("not_allowed", "str")]]


# ---------------------------------------------------------------------------
# End of life counted from the start of the run
# ---------------------------------------------------------------------------


@pytest.mark.solver
def test_existing_capacity_ends_in_the_same_period_in_a_chain(tmp_path_factory) -> None:
    """reinvest_choice, lifetime 15: the pre-existing capacity is 500 MW in p2020 … p2030 and 0 in p2035 in the
    single solve and in every solve of the chain (before: a later solve of the chain counted the lifetime from
    its own first period, so y2030 and y2035 kept 500 MW in p2035)."""
    values = _lifetime("reinvest_choice", 15.0) + NO_INVEST
    want = {"p2020": 500.0, "p2025": 500.0, "p2030": 500.0, "p2035": 0.0}
    for kind, chain in (("single", SINGLE), ("chain", CHAIN)):
        url, scen = make_db(tmp_path_factory, alt=f"eol_{kind}", values=values, base_chain=_no_battery_invest(chain))
        steps = run(tmp_path_factory, url, scen)
        for name, step in steps.items():
            got = _ed(step.flex_data.p_entity_pre_existing)
            assert got == {d: want[d] for d in got}, (kind, name)
            capacity = _ed(step.flex_data.p_entity_all_existing)
            assert capacity == {d: want[d] for d in capacity}, (kind, name)


@pytest.mark.solver
def test_chain_reports_fixed_cost_of_existing_capacity_until_its_end_of_life(tmp_path_factory) -> None:
    """The reported fixed cost of the existing 500 MW: 500 · 1e5 · F_ops(d) in p2020 … p2030, nothing in p2035;
    the chain reports what the single solve reports (before: the chain charged it in p2035 too)."""
    values = _lifetime("reinvest_choice", 15.0) + NO_INVEST
    single = _results(tmp_path_factory, "eol_fc_single", values, _no_battery_invest(SINGLE))
    chain = _results(tmp_path_factory, "eol_fc_chain", values, _no_battery_invest(CHAIN))
    _assert_same_results(chain, single)
    discounted = chain["cost_unit_discounted__d"]
    for d in PERIODS:
        rows = discounted[(discounted["period"] == d) & (discounted["unit"] == PLANT)]
        fixed = rows["fixed cost pre-existing"].sum() if len(rows) else 0.0
        want = 0.0 if d == "p2035" else 500.0 * FIXED * F_OPS[d]
        assert fixed * MEGA == pytest.approx(want, rel=1e-7, abs=1e-6), d


@pytest.mark.solver
@pytest.mark.parametrize("chain", [["init", "base", "stoch"], ["init", "base", "stoch", "recourse"]],
                         ids=["none", "recourse"])
def test_fan_in_the_first_period_counts_the_end_of_life_once(tmp_path_factory, chain) -> None:
    """``stoch_two_period_invest``: the fan (high, low) opens in the run's first period p2035 (year 0); p2040
    starts at year 1.  wind (existing 100 MW) with reinvest_choice and lifetime 1 ends at year 0 + 1 = 1, so it
    has 100 MW in p2035 and its branches and nothing in p2040 and its branches, as in the deterministic twin
    (before: the end of life was summed over the two branch periods of p2035, year 2, and wind ran in p2040)."""
    values = _lifetime("reinvest_choice", 1.0, "wind")
    url, scen = make_db(tmp_path_factory, alt=f"eol_fan_{len(chain)}", values=values, base_chain=chain,
                        fixture_json=FAN)
    step = list(without_realization(run(tmp_path_factory, url, scen)).values())[-1]
    pre = _ed(step.flex_data.p_entity_pre_existing, "wind")
    capacity = _ed(step.flex_data.p_entity_all_existing, "wind")
    for d in ("p2035_high", "p2035_low", "p2040_high", "p2040_low"):
        want = 100.0 if d.startswith("p2035") else 0.0
        assert pre[d] == want, d
        assert capacity[d] == want, d
    flow = utf8(step.solution.value("v_flow")).filter(pl.col("p") == "wind")
    assert flow.filter(pl.col("d").str.starts_with("p2040"))["value"].abs().max() == pytest.approx(0.0, abs=1e-9)
    assert flow.filter(pl.col("d").str.starts_with("p2035"))["value"].max() > 0.0


# ---------------------------------------------------------------------------
# Retiring existing capacity before its end of life
# ---------------------------------------------------------------------------
#
# coal_plant retires exactly 100 MW (retire_min_period = retire_max_period = 100 in p2025, 0 elsewhere,
# invest_method retire_period, salvage_value 200 so that it has a retirement option).  The retirement removes
# the 100 MW from p2025 until the end of life of the existing capacity; there the whole 500 MW are gone.

SALVAGE = 200.0


def _pins(period: str, mw: float) -> list:
    from tests.engine.test_chain_costs import _period_map
    return _period_map({d: (mw if d == period else 0.0) for d in PERIODS})


def _retire(mw: float = 100.0, period: str = "p2025", *, pinned: bool = True, entity: str = PLANT,
            cls: str = "unit", salvage: float = SALVAGE) -> list:
    from tests.engine.test_chain_costs import _period_map
    low = _pins(period, mw) if pinned else _period_map(dict.fromkeys(PERIODS, 0.0))
    return [[cls, entity, "invest_method", b64("retire_period", "str")],
            [cls, entity, "retire_min_period", low],
            [cls, entity, "retire_max_period", _pins(period, mw)],
            [cls, entity, "salvage_value", b64(salvage, "float")]]


def _divest_mw(step, entity: str = PLANT, var: str = "v_divest_p", col: str = "p") -> dict[str, float]:
    from tests.engine._unit_db import unitsize
    if col == "n":
        sizes = utf8(step.flex_data.p_state_unitsize.frame)
        size = float(sizes.filter(pl.col(sizes.columns[0]) == entity)["value"][0])
    else:
        size = unitsize(step, entity)
    if var not in step.solution._vars:
        return {}
    frame = utf8(step.solution.value(var)).filter(pl.col(col) == entity)
    return {d: v * size for d, v in zip(frame["d"].to_list(), frame["value"].to_list())}


def _divest_columns(terms: dict[str, float], entity: str = PLANT, var: str = "v_divest_p") -> list[str]:
    return sorted(c for c in terms if c.startswith(f"{var}[{entity},"))


def _rows_by_period(pb, family: str, entity: str = PLANT) -> dict[str, list[tuple[dict, tuple]]]:
    from tests.engine._lp_rows import row_bounds, row_terms
    terms, bounds = row_terms(pb, family, entity), row_bounds(pb, family, entity)
    out: dict[str, list] = {}
    for name, t in terms.items():
        period = next(d for d in PERIODS if f",{d}," in name or name.endswith(f",{d}]"))
        out.setdefault(period, []).append((t, bounds[name]))
    return out


def _single(tmp_path_factory, alt: str, values: list):
    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=_no_battery_invest(SINGLE))
    return list(run(tmp_path_factory, url, scen).values())[-1]


@pytest.mark.solver
def test_retirement_counts_until_the_end_of_life(tmp_path_factory) -> None:
    """reinvest_choice, lifetime 15 (EOL year 15 = p2035), 100 MW retired in p2025, with linear unit commitment.
    The capacity in service is 500 / 400 / 400 / 0 MW in p2020 … p2035 (before: −100 MW in p2035, infeasible):
    the retirement options are p2020, p2025 and p2030 (p2035 has no existing capacity left); every maxFlow and
    maxOnline row of p2030 carries the p2025 retirement, no row of p2035 carries one, and p2035's capacity is 0;
    maxRetired_existing_p caps the retired capacity at the 500 MW in p2025 and p2030 (and p2020)."""
    from tests.engine._unit_db import build, uc, unitsize
    values = _lifetime("reinvest_choice", 15.0) + _retire() + uc(PLANT, 0.3)
    step = _single(tmp_path_factory, "ret_eol", values)
    assert step.optimal
    retired = _divest_mw(step)
    assert sorted(retired) == ["p2020", "p2025", "p2030"]
    assert retired["p2025"] == pytest.approx(100.0, abs=1e-6)
    assert retired["p2020"] == pytest.approx(0.0, abs=1e-6) and retired["p2030"] == pytest.approx(0.0, abs=1e-6)
    pb = build(step.flex_data)
    size = unitsize(step, PLANT)
    for family in ("maxFlow", "maxOnline_linear"):
        rows = _rows_by_period(pb, family)
        assert rows["p2030"] and rows["p2035"], family
        for terms, _ in rows["p2030"]:
            assert _divest_columns(terms) == [f"v_divest_p[{PLANT},p2020]", f"v_divest_p[{PLANT},p2025]",
                                              f"v_divest_p[{PLANT},p2030]"], family
        for terms, (_, upper) in rows["p2035"]:
            assert _divest_columns(terms) == [], family
            assert upper == pytest.approx(0.0, abs=1e-9), family
    capped = _rows_by_period(pb, "maxRetired_existing_p")
    assert sorted(capped) == ["p2020", "p2025", "p2030"]
    for period, ((terms, (_, upper)),) in capped.items():
        assert upper == pytest.approx(500.0, rel=1e-12), period
        assert all(v == pytest.approx(size, rel=1e-12) for v in terms.values()), period


@pytest.mark.solver
def test_no_investment_lifetime_ends_the_retirement(tmp_path_factory) -> None:
    """no_investment, lifetime 10 (EOL year 10 = p2030): 100 MW retired in p2025 leave 400 MW in p2025 and nothing
    from p2030 on; (coal, p2030) and (coal, p2035) are no retirement options (before: infeasible)."""
    from tests.engine._unit_db import build
    step = _single(tmp_path_factory, "ret_noinv", _lifetime("no_investment", 10.0) + _retire())
    assert step.optimal
    retired = _divest_mw(step)
    assert sorted(retired) == ["p2020", "p2025"]
    assert retired["p2025"] == pytest.approx(100.0, abs=1e-6)
    rows = _rows_by_period(build(step.flex_data), "maxFlow")
    for period in ("p2030", "p2035"):
        for terms, (_, upper) in rows[period]:
            assert _divest_columns(terms) == [] and upper == pytest.approx(0.0, abs=1e-9), period
    assert all(_divest_columns(t) == [f"v_divest_p[{PLANT},p2020]", f"v_divest_p[{PLANT},p2025]"]
               for t, _ in rows["p2025"])


@pytest.mark.solver
def test_retired_capacity_stays_within_the_existing_capacity(tmp_path_factory) -> None:
    """reinvest_automatic, ``existing`` 500 / 500 / 200 / 200 MW, up to 400 MW retirable in p2025, availability
    0 (the plant never runs, so retiring it only earns the salvage value and saves fixed cost: the solve retires
    as much as it may).  The retirement counts in p2025 … p2035, so maxRetired_existing_p caps it at the smallest
    existing capacity there, 200 MW (the p2030 and p2035 rows bind); the solve retires exactly 200 MW (before:
    no such row, and with availability 0 no capacity row held the retirement back)."""
    from tests.engine._unit_db import build
    from tests.engine.test_chain_costs import _period_map
    existing = {"p2020": 500.0, "p2025": 500.0, "p2030": 200.0, "p2035": 200.0}
    values = _retire(400.0, pinned=False) + [["unit", PLANT, "existing", _period_map(existing)],
                                             ["unit", PLANT, "availability", b64(0.0, "float")]]
    step = _single(tmp_path_factory, "ret_limit", values)
    assert step.optimal
    assert _divest_mw(step)["p2025"] == pytest.approx(200.0, rel=1e-9)
    capped = _rows_by_period(build(step.flex_data), "maxRetired_existing_p")
    assert {d: rows[0][1][1] for d, rows in capped.items()} == pytest.approx(existing, rel=1e-12)


@pytest.mark.solver
def test_entity_without_existing_capacity_has_no_retirement(tmp_path_factory, capsys) -> None:
    """coal_plant with existing 0 that may invest and retire (invest_retire_no_limit): only existing capacity can
    be retired, so it has investment variables but no retirement variable, and a warning names it."""
    values = [["unit", PLANT, "existing", b64(0.0, "float")],
              ["unit", PLANT, "invest_method", b64("invest_retire_no_limit", "str")],
              ["unit", PLANT, "salvage_value", b64(SALVAGE, "float")]]
    step = _single(tmp_path_factory, "ret_none", values)
    assert step.optimal
    assert utf8(step.solution.value("v_invest_p")).filter(pl.col("p") == PLANT).height == len(PERIODS)
    fd = step.flex_data
    assert fd.pd_divest_set is None or utf8(fd.pd_divest_set).filter(pl.col("p") == PLANT).height == 0
    assert fd.edd_divest_active is None or utf8(fd.edd_divest_active).filter(pl.col("p") == PLANT).height == 0
    out = capsys.readouterr().out
    assert f"{PLANT} has a retirement option" in out and "Only existing capacity can be retired" in out


def _config_error(tmp_path_factory, alt: str, values: list) -> str:
    from flextool.engine.input._solve_state import FlexToolConfigError
    with pytest.raises(FlexToolConfigError) as err:
        _single(tmp_path_factory, alt, values)
    return str(err.value)


def test_minimum_retirement_after_the_end_of_life_is_an_error(tmp_path_factory) -> None:
    """reinvest_choice, lifetime 15: retiring 100 MW in p2035 is required, but the existing capacity has ended
    there (year 15) — an error naming the entity and the period (before: an infeasible LP)."""
    msg = _config_error(tmp_path_factory, "ret_err_eol", _lifetime("reinvest_choice", 15.0) + _retire(period="p2035"))
    assert PLANT in msg and "p2035" in msg and "retire_min_period" in msg and "year 15" in msg


def test_minimum_retirement_above_the_existing_capacity_is_an_error(tmp_path_factory) -> None:
    """600 MW required in p2025 of the existing 500 MW: an error naming the entity, the period and the 500 MW."""
    msg = _config_error(tmp_path_factory, "ret_err_600", _retire(600.0))
    assert PLANT in msg and "p2025" in msg and "600" in msg and "500" in msg


def test_minimum_total_retirement_above_what_the_window_allows_is_an_error(tmp_path_factory) -> None:
    """reinvest_automatic, ``existing`` 500 / 500 / 200 / 200 MW, retire_total with retire_min_total 300 MW: a
    retirement in any period counts until the end of the horizon, so at most min(…, 200, 200) = 200 MW can be
    retired (the largest existing capacity, 500 MW, is not the limit) — an error naming the entity."""
    from tests.engine.test_chain_costs import _period_map
    existing = {"p2020": 500.0, "p2025": 500.0, "p2030": 200.0, "p2035": 200.0}
    values = [["unit", PLANT, "existing", _period_map(existing)],
              ["unit", PLANT, "invest_method", b64("retire_total", "str")],
              ["unit", PLANT, "retire_min_total", b64(300.0, "float")],
              ["unit", PLANT, "retire_max_total", b64(1000.0, "float")],
              ["unit", PLANT, "salvage_value", b64(SALVAGE, "float")]]
    msg = _config_error(tmp_path_factory, "ret_err_total", values)
    assert PLANT in msg and "retire_min_total" in msg and "at most 200" in msg


@pytest.mark.solver
def test_group_minimum_retirement_above_the_members_existing_capacity_is_an_error(tmp_path_factory) -> None:
    """A group retirement minimum (off-schema ``retire_min_total`` of a group) of 600 MW over coal_plant's 500 MW
    existing capacity: building the LP raises an error naming the group."""
    from polar_high import Param

    from flextool.engine.input._solve_state import FlexToolConfigError
    from tests.engine._unit_db import build
    step = _single(tmp_path_factory, "ret_err_group", _retire(100.0, pinned=False))
    fd = step.flex_data
    e_dtype = fd.p_entity_pre_existing.frame.schema["e"]
    fd.group_entity = pl.DataFrame({"g": ["retire_grp"], "e": [PLANT]}).with_columns(pl.col("e").cast(e_dtype))
    fd.p_group_retire_min_total = Param(("g",), pl.DataFrame({"g": ["retire_grp"], "value": [600.0]}))
    with pytest.raises(FlexToolConfigError) as err:
        build(fd)
    assert "group retire_grp" in str(err.value) and "500" in str(err.value)


@pytest.mark.solver
def test_storage_retirement_counts_until_the_end_of_life(tmp_path_factory) -> None:
    """The battery node (existing 50 MWh) with reinvest_choice, lifetime 15, retires 10 MWh in p2025: feasible;
    the state capacity rows of p2030 carry the p2025 retirement, those of p2035 none (capacity 0 there, the
    existing storage has ended), and maxRetired_existing_n caps the retired storage at 50 MWh."""
    from tests.engine._unit_db import build
    values = _lifetime("reinvest_choice", 15.0, "battery") + _retire(10.0, entity="battery", cls="node")
    values = [[("node" if v[1] == "battery" else v[0]), *v[1:]] for v in values]
    step = _single(tmp_path_factory, "ret_storage", values)
    assert step.optimal
    retired = _divest_mw(step, "battery", "v_divest_n", "n")
    assert sorted(retired) == ["p2020", "p2025", "p2030"]
    assert retired["p2025"] == pytest.approx(10.0, abs=1e-6)
    pb = build(step.flex_data)
    rows = _rows_by_period(pb, "maxState", "battery")
    assert rows["p2030"] and rows["p2035"]
    for terms, _ in rows["p2030"]:
        assert _divest_columns(terms, "battery", "v_divest_n") == [
            "v_divest_n[battery,p2020]", "v_divest_n[battery,p2025]", "v_divest_n[battery,p2030]"]
    for terms, (_, upper) in rows["p2035"]:
        assert _divest_columns(terms, "battery", "v_divest_n") == [] and upper == pytest.approx(0.0, abs=1e-9)
    capped = _rows_by_period(pb, "maxRetired_existing_n", "battery")
    assert sorted(capped) == ["p2020", "p2025", "p2030"]
    assert all(rows[0][1][1] == pytest.approx(50.0, rel=1e-12) for rows in capped.values())


# ---------------------------------------------------------------------------
# Chains of solves carry the retirements per decision period
# ---------------------------------------------------------------------------


def _in_service(step, entity: str = PLANT) -> dict[str, float]:
    """Capacity in service per period of ``step``: its existing capacity (with the retirements of earlier solves
    still counting) minus its own retirements still counting (edd_divest_active)."""
    fd = step.flex_data
    capacity = dict(_ed(fd.p_entity_all_existing, entity))
    if fd.edd_divest_active is not None:
        retired = _divest_mw(step, entity)
        for d_r, d in utf8(fd.edd_divest_active).filter(pl.col("p") == entity).select("d_divest", "d").iter_rows():
            capacity[d] = capacity.get(d, 0.0) - retired.get(d_r, 0.0)
    return capacity


@pytest.mark.solver
def test_chain_carries_the_retirement_until_the_end_of_life(tmp_path_factory) -> None:
    """reinvest_choice, lifetime 15, 100 MW retired in p2025, solved as one solve and as the chain y2020 … y2035.
    Capacity in service 500 / 400 / 400 / 0 MW in p2020 … p2035 in every solve that holds the period (before: the
    chain subtracted the 100 MW from every period of the later solves, so y2030 had −100 MW in p2035 and was
    infeasible); y2030 carries the retirement into p2030 (500 − 100) but not into p2035 (the existing capacity has
    ended).  The reported costs of the chain equal the single solve's."""
    values = _lifetime("reinvest_choice", 15.0) + _retire()
    want = {"p2020": 500.0, "p2025": 400.0, "p2030": 400.0, "p2035": 0.0}
    for kind, chain in (("single", SINGLE), ("chain", CHAIN)):
        url, scen = make_db(tmp_path_factory, alt=f"ret_chain_{kind}", values=values,
                            base_chain=_no_battery_invest(chain))
        steps = run(tmp_path_factory, url, scen)
        for name, step in steps.items():
            assert step.optimal, (kind, name)
            got = _in_service(step)
            assert got == pytest.approx({d: want[d] for d in got}, abs=1e-6), (kind, name)
        if kind == "chain":
            y2030 = steps["y2030_5week"].flex_data
            assert _ed(y2030.p_entity_retired_carried) == pytest.approx({"p2030": 100.0})
            assert _ed(y2030.p_entity_all_existing) == pytest.approx({"p2030": 400.0, "p2035": 0.0})
    single = _results(tmp_path_factory, "ret_chain_costs_single", values, _no_battery_invest(SINGLE))
    chained = _results(tmp_path_factory, "ret_chain_costs_chain", values, _no_battery_invest(CHAIN))
    _assert_same_results(chained, single)


@pytest.mark.solver
def test_nested_dispatch_solve_carries_the_retirement(tmp_path_factory) -> None:
    """An investment solve decides the retirement for p2020 … p2035 (realized, no dispatch) and a dispatch solve
    runs on the capacity it hands over: 500 / 400 / 400 / 0 MW (before: the dispatch solve subtracted the 100 MW in
    p2035 too, −100 MW, infeasible); the reported costs equal the single solve's."""
    from tests.engine.test_chain_costs import _solve, _solves
    values = _lifetime("reinvest_choice", 15.0) + _retire()
    nested_values = (values + _solves(["cc_invest", "cc_dispatch"])
                     + _solve("cc_invest", PERIODS, invest_periods=PERIODS, realized_invest_periods=PERIODS)
                     + _solve("cc_dispatch", PERIODS, realized_periods=PERIODS))
    entities = [["solve", "cc_invest", None], ["solve", "cc_dispatch", None]]
    url, scen = make_db(tmp_path_factory, alt="ret_nested", values=nested_values,
                        base_chain=_no_battery_invest(CHAIN), entities=entities)
    steps = run(tmp_path_factory, url, scen)
    dispatch = steps["cc_dispatch"]
    assert dispatch.optimal
    assert _ed(dispatch.flex_data.p_entity_all_existing) == pytest.approx(
        {"p2020": 500.0, "p2025": 400.0, "p2030": 400.0, "p2035": 0.0})
    nested = _results(tmp_path_factory, "ret_nested_costs", nested_values, _no_battery_invest(CHAIN),
                      entities=entities)
    single = _results(tmp_path_factory, "ret_nested_single", values, _no_battery_invest(SINGLE))
    _assert_same_results(nested, single)


@pytest.mark.solver
@pytest.mark.parametrize("chain", [["init", "base", "stoch"], ["init", "base", "stoch", "recourse"]],
                         ids=["none", "recourse"])
def test_fan_retirement_ends_with_the_existing_capacity(tmp_path_factory, chain) -> None:
    """``stoch_two_period_invest``: wind (existing 100 MW, reinvest_choice, lifetime 1: it ends at year 1 = p2040)
    retires exactly 50 MW in p2035.  The retirement counts in the p2035 branch periods only: no maxFlow row of a
    p2040 branch carries a retirement and their capacity is 0 (before: the end of life was year 2, so the
    retirement and 50 MW stayed in p2040)."""
    from tests.engine._unit_db import build
    from tests.engine.test_chain_costs import _period_map
    pins = _period_map({"p2035": 50.0, "p2040": 0.0})
    values = (_lifetime("reinvest_choice", 1.0, "wind")
              + [["unit", "wind", "invest_method", b64("retire_period", "str")],
                 ["unit", "wind", "retire_min_period", pins], ["unit", "wind", "retire_max_period", pins],
                 ["unit", "wind", "salvage_value", b64(SALVAGE, "float")]])
    url, scen = make_db(tmp_path_factory, alt=f"ret_fan_{len(chain)}", values=values, base_chain=chain,
                        fixture_json=FAN)
    step = list(without_realization(run(tmp_path_factory, url, scen)).values())[-1]
    assert step.optimal
    active = utf8(step.flex_data.edd_divest_active).filter(pl.col("p") == "wind")
    assert {"p2035_high", "p2035_low"} <= set(active["d"].to_list())
    assert not [d for d in active["d"].to_list() if d.startswith("p2040")]
    pb = build(step.flex_data)
    from tests.engine._lp_rows import row_bounds, row_terms
    terms, bounds = row_terms(pb, "maxFlow", "wind"), row_bounds(pb, "maxFlow", "wind")
    p2040 = [n for n in terms if ",p2040_" in n]
    assert p2040
    for name in p2040:
        assert _divest_columns(terms[name], "wind") == [] and bounds[name][1] == pytest.approx(0.0, abs=1e-9)
    assert all(_divest_columns(terms[n], "wind") for n in terms if ",p2035_" in n)


# ---------------------------------------------------------------------------
# Costs of a retirement over its window
# ---------------------------------------------------------------------------


@pytest.mark.solver
def test_retirement_costs_stop_at_the_end_of_life(tmp_path_factory) -> None:
    """reinvest_choice, lifetime 15 (EOL year 15), 100 MW retired in p2025: the window is p2025 and p2030.  By hand
    (``test_chain_costs`` factors, r = 0.04 / offsets 0.5 and 0)::

        fixed cost reduction (p2025) = −100 · 1e5 · (F_ops(p2025) + F_ops(p2030))
        R = 5 + 5 = 10 years,  A = 200 · 1000 · 0.05 / (1 − 1.05^−10) = 25 900.9 CUR/MW/yr
        retirement (p2025)           = −100 · A · (F_inv(p2025) + F_inv(p2030))
        fixed cost pre-existing (d)  = 500 · 1e5 · F_ops(d) in p2020 … p2030, 0 in p2035

    Annualized: −100 · A and −100 · 1e5 per year in p2025 and p2030, nothing in p2035.  The chain reports what the
    single solve reports."""
    from tests.engine.test_chain_costs import F_INV
    values = _lifetime("reinvest_choice", 15.0) + _retire()
    single = _results(tmp_path_factory, "ret_cost_single", values, _no_battery_invest(SINGLE))
    chain = _results(tmp_path_factory, "ret_cost_chain", values, _no_battery_invest(CHAIN))
    _assert_same_results(chain, single)
    window = ["p2025", "p2030"]
    annuity = SALVAGE * 1000.0 * 0.05 / (1.0 - 1.05 ** -10.0)
    discounted = single["cost_unit_discounted__d"]
    rows = discounted[(discounted["period"] == "p2025") & (discounted["unit"] == PLANT)]
    assert rows["fixed cost reduction of divestments"].sum() * MEGA == pytest.approx(
        -100.0 * FIXED * sum(F_OPS[d] for d in window), rel=1e-7)
    assert rows["retirement"].sum() * MEGA == pytest.approx(-100.0 * annuity * sum(F_INV[d] for d in window),
                                                           rel=1e-7)
    for d in PERIODS:
        rows = discounted[(discounted["period"] == d) & (discounted["unit"] == PLANT)]
        want = 0.0 if d == "p2035" else 500.0 * FIXED * F_OPS[d]
        assert rows["fixed cost pre-existing"].sum() * MEGA == pytest.approx(want, rel=1e-7, abs=1e-6), d
    for annualized in (chain["cost_unit_annualized__d"], single["cost_unit_annualized__d"]):
        for d in PERIODS:
            rows = annualized[(annualized["period"] == d) & (annualized["unit"] == PLANT)]
            mw = 100.0 if d in window else 0.0
            assert rows["retirement"].sum() * MEGA == pytest.approx(-mw * annuity, rel=1e-7, abs=1e-6), d
            assert rows["fixed cost reduction of divestments"].sum() * MEGA == pytest.approx(
                -mw * FIXED, rel=1e-7, abs=1e-6), d


@pytest.mark.solver
def test_retirement_saves_exactly_the_fixed_cost_of_existing_capacity(tmp_path_factory) -> None:
    """Retiring all 500 MW in p2020 (reinvest_automatic): the fixed cost of the existing capacity
    (500 · 1e5 · F_ops(d) over p2020 … p2035) and the retirement's saving cancel exactly, so the reported total of
    the two is 0 (before: the saving used the investment factor, F_inv = 1.04^0.5 · F_ops, a leftover of 2 %)."""
    values = _retire(500.0, "p2020")
    single = _results(tmp_path_factory, "ret_all", values, _no_battery_invest(SINGLE))
    discounted = single["cost_unit_discounted__d"]
    unit = discounted[discounted["unit"] == PLANT]
    existing = unit["fixed cost pre-existing"].sum()
    assert existing * MEGA == pytest.approx(500.0 * FIXED * sum(F_OPS[d] for d in PERIODS), rel=1e-7)
    # The CSV rounds each figure (M CUR); the two cancel up to that rounding.
    assert (existing + unit["fixed cost reduction of divestments"].sum()) == pytest.approx(0.0, abs=1e-6 * existing)


# ---------------------------------------------------------------------------
# lifetime as a constant or as a period map; the behaviour-change notice
# ---------------------------------------------------------------------------


@pytest.mark.solver
@pytest.mark.parametrize("kind", ["single", "chain"])
def test_lifetime_constant_equals_map(tmp_path_factory, kind) -> None:
    """lifetime 15 as a constant and as the period map {p2020 … p2035: 15}: the same solves (objective and
    retirement in every solve; the end of life counts from the run's first period in both)."""
    from tests.engine.test_chain_costs import _period_map
    chain = _no_battery_invest(SINGLE if kind == "single" else CHAIN)
    outcomes = []
    for shape, life in (("const", b64(15.0, "float")), ("map", _period_map(dict.fromkeys(PERIODS, 15.0)))):
        values = ([["unit", PLANT, "lifetime", life], ["unit", PLANT, "lifetime_method", b64("reinvest_choice", "str")]]
                  + _retire())
        url, scen = make_db(tmp_path_factory, alt=f"life_{shape}_{kind}", values=values, base_chain=chain)
        steps = run(tmp_path_factory, url, scen)
        outcomes.append({name: (step.solution.obj, _divest_mw(step)) for name, step in steps.items()})
    const, mapped = outcomes
    assert sorted(const) == sorted(mapped)
    for name in const:
        assert mapped[name][0] == pytest.approx(const[name][0], rel=1e-12), name
        assert mapped[name][1] == pytest.approx(const[name][1], abs=1e-9), name


NOTICE_ID = "retirement_of_existing_capacity_only"
NOTICE_MARKER = f"flextool_behaviour_change:{NOTICE_ID}"


def _notice_lines(tmp_path_factory, capsys, alt: str, values: list, chain: list[str], marker: bool) -> list[str]:
    from spinedb_api import DatabaseMapping

    from flextool.engine import run_chain_from_db
    from flextool.engine._behaviour_changes import BEHAVIOUR_CHANGES

    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=chain)
    if marker:
        with DatabaseMapping(url) as db:
            db.add_item("metadata", name=NOTICE_MARKER, value="77")
            db.commit_session("behaviour-change marker")
    capsys.readouterr()
    run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"{alt}_w"))
    summary = BEHAVIOUR_CHANGES[NOTICE_ID].summary
    return [ln for ln in capsys.readouterr().out.splitlines() if summary in ln]


@pytest.mark.solver
def test_notice_names_the_retiring_entities_once(tmp_path_factory, capsys) -> None:
    """The chain y2020 … y2035 with a retiring coal_plant: one notice line naming it."""
    lines = _notice_lines(tmp_path_factory, capsys, "ret_notice", _lifetime("reinvest_choice", 15.0) + _retire(),
                          _no_battery_invest(CHAIN), marker=True)
    assert len(lines) == 1
    assert f"Affected entities: {PLANT}." in lines[0] and NOTICE_MARKER in lines[0]


@pytest.mark.solver
@pytest.mark.parametrize("chain", ["chain", "single"])
def test_notice_names_the_end_of_life_entities_of_a_chain(tmp_path_factory, capsys, chain) -> None:
    """reinvest_choice existing capacity without retirement: a chain names coal_plant once (its end of life now
    counts from the run's start in the later solves); a single solve prints nothing."""
    values = _lifetime("reinvest_choice", 15.0) + NO_INVEST
    lines = _notice_lines(tmp_path_factory, capsys, f"eol_notice_{chain}", values,
                          _no_battery_invest(CHAIN if chain == "chain" else SINGLE), marker=True)
    if chain == "chain":
        assert len(lines) == 1 and f"Affected entities: {PLANT}." in lines[0]
    else:
        assert lines == []


@pytest.mark.solver
@pytest.mark.parametrize("retiring, marker", [(True, False), (False, True)], ids=["no_marker", "no_retirement"])
def test_no_retirement_notice(tmp_path_factory, capsys, retiring, marker) -> None:
    values = _retire() if retiring else NO_INVEST
    alt = f"ret_no_notice_{retiring}_{marker}"
    assert _notice_lines(tmp_path_factory, capsys, alt, values, _no_battery_invest(SINGLE), marker) == []


def test_notice_is_registered_and_written_by_the_upgrade() -> None:
    from flextool.engine._behaviour_changes import BEHAVIOUR_CHANGES
    from flextool.update_flextool import db_migration

    assert NOTICE_ID in BEHAVIOUR_CHANGES
    assert NOTICE_ID in db_migration._V77_UNCONDITIONAL_MARKERS
