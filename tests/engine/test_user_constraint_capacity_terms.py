"""User-constraint capacity terms (database version 77).

* ``constraint_capacity_coeff`` puts the capacity in service in each period on the left side; without a
  construction lead time it equals the sum of the pre-built, invested and retired terms with the same
  coefficient (same row, single solve, chain of solves and recourse solve).
* With a construction lead time the invested term counts the capacity coming online, so the identity holds
  as well (:func:`test_lead_time_keeps_identity`).
* ``constraint_cumulative_pre_built_capacity_coeff`` subtracts the retirements of earlier periods, so a single
  solve and a chain of solves agree.
* A constraint without flow or state terms is written once per period; with a flow term once per step; the
  objective is the same.
* A user-constraint coefficient gives a zero-cost entity its investment / retirement variables.

Built from ``tests/fixtures/tests.json`` and ``stoch_two_period_invest.json`` (never a checked-in ``.sqlite``).
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._lp_rows import row_bounds, row_terms
from tests.engine._unit_db import FLEXTOOL_ROOT, b64, build, make_db, run, utf8

PLANT = "coal_plant"
SINGLE = ["init", "west", "wind", "coal", "coal_invest", "battery", "battery_invest", "5weeks",
          "invest_solveSequence_5weeks", "invest_allYears_5weeks"]
CHAIN = ["init", "west", "wind", "coal", "coal_invest", "battery", "battery_invest", "5weeks",
         "invest_solveSequence_5weeks"]
STOCH_JSON = FLEXTOOL_ROOT / "tests" / "fixtures" / "stoch_two_period_invest.json"
STOCH_CHAIN = ["init", "base", "stoch", "recourse"]
GE = "process_constraint_greater_than"


def _map(data: dict[str, float], index_name: str = "constraint") -> list:
    return b64({"index_type": "str", "index_name": index_name, "data": [[k, v] for k, v in data.items()]}, "map")


def _constraint(name: str, sense: str = "greater_than", constant: float | None = 0.0) -> list:
    out = [["constraint", name, "sense", b64(sense, "str")]]
    if constant is not None:
        out.append(["constraint", name, "constant", b64(constant, "float")])
    return out


def _twin_values(cls: str, entity: str) -> list:
    """``c_cap``: capacity coefficient 1; ``c_twin``: pre-built 1, invested 1, retired −1."""
    return [[cls, entity, "constraint_capacity_coeff", _map({"c_cap": 1.0})],
            [cls, entity, "constraint_cumulative_pre_built_capacity_coeff", _map({"c_twin": 1.0})],
            [cls, entity, "constraint_invested_capacity_coeff", _map({"c_twin": 1.0})],
            [cls, entity, "constraint_retired_capacity_coeff", _map({"c_twin": -1.0})]]


TWIN_ENTITIES = [["constraint", "c_cap", None], ["constraint", "c_twin", None]]
RETIRABLE = [["unit", PLANT, "invest_method", b64("invest_retire_no_limit", "str")],
             ["node", "battery", "invest_method", b64("invest_retire_no_limit", "str")]]
TWIN_VALUES = (RETIRABLE + _constraint("c_cap") + _constraint("c_twin")
               + _twin_values("unit", PLANT) + _twin_values("node", "battery"))


def _period_rows(pb, family: str, cn: str) -> dict[str, tuple[dict[str, float], tuple[float, float]]]:
    """``{period: (terms, bounds)}`` of the one-row-per-period rows of constraint ``cn``."""
    terms = row_terms(pb, family, f"{cn},")
    bounds = row_bounds(pb, family, f"{cn},")
    return {row[row.index("[") + 1:-1].split(",")[-1]: (terms[row], bounds[row]) for row in terms}


def _assert_twin(pb) -> None:
    cap = _period_rows(pb, GE + "_period", "c_cap")
    twin = _period_rows(pb, GE + "_period", "c_twin")
    assert cap, "no capacity rows"
    assert sorted(cap) == sorted(twin)
    for d in cap:
        assert cap[d][0] == pytest.approx(twin[d][0], rel=1e-12), d
        assert cap[d][1] == pytest.approx(twin[d][1], rel=1e-12), d
    # The capacity rows of a period carry the investments of that period and of the earlier ones.
    assert any(len([c for c in t if c.startswith("v_invest")]) > 1 for t, _ in cap.values())


@pytest.mark.parametrize("chain", [SINGLE, CHAIN], ids=["single_solve", "chain_of_solves"])
def test_capacity_equals_prebuilt_plus_invested_minus_retired(tmp_path_factory, chain) -> None:
    url, scen = make_db(tmp_path_factory, alt="uc_cap_twin", values=TWIN_VALUES, base_chain=chain,
                        entities=TWIN_ENTITIES)
    steps = run(tmp_path_factory, url, scen)
    for step in steps.values():
        _assert_twin(build(step.flex_data))


def _add_values(url: str, alt: str, values: list) -> None:
    """Add parameter values in ``alt`` to the migrated database at ``url`` (for parameters newer than the
    fixture's database version)."""
    from spinedb_api import DatabaseMapping, import_data

    from tests.db_utils import _unpack_value

    with DatabaseMapping(url) as db:
        _, errors = import_data(db, unparse_value=_unpack_value,
                                parameter_values=[[c, e, p, v, alt] for c, e, p, v in values])
        assert not errors, errors
        db.commit_session("v77 user-constraint coefficients")


def test_capacity_twin_under_recourse(tmp_path_factory) -> None:
    values = ([["unit", "peaker", "invest_method", b64("invest_retire_no_limit", "str")]]
              + _constraint("c_cap") + _constraint("c_twin"))
    url, scen = make_db(tmp_path_factory, alt="uc_cap_twin_recourse", values=values, base_chain=STOCH_CHAIN,
                        entities=TWIN_ENTITIES, fixture_json=STOCH_JSON)
    _add_values(url, "uc_cap_twin_recourse", _twin_values("unit", "peaker"))
    steps = run(tmp_path_factory, url, scen)
    step = list(steps.values())[0]
    pb = build(step.flex_data)
    _assert_twin(pb)
    # The branch periods of the stochastic fan get their own rows.
    cap = _period_rows(pb, GE + "_period", "c_cap")
    branch_rows = {d: t for d, (t, _) in cap.items() if d not in ("p2035", "p2040")}
    assert branch_rows, sorted(cap)


def test_lead_time_keeps_identity(tmp_path_factory) -> None:
    """coal_plant with a 5-year construction lead time over p2020 … p2035 (5 years apart): an investment decided
    in d comes online in the next period.  The invested term counts it there (v_invest[d − 1] in d), as capacity
    in service and pre-built do, so capacity = pre-built + invested − retired row by row, as without a lead time.
    The p2035 decision would start to operate at the end of the horizon and is not offered."""
    values = TWIN_VALUES + [["unit", PLANT, "construction_lead_time", b64(5.0, "float")]]
    url, scen = make_db(tmp_path_factory, alt="uc_cap_lead", values=values, base_chain=SINGLE,
                        entities=TWIN_ENTITIES)
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    us = float(utf8(step.flex_data.p_unitsize.frame).filter(pl.col("p") == PLANT)["value"][0])
    pb = build(step.flex_data)
    _assert_twin(pb)
    cap = _period_rows(pb, GE + "_period", "c_cap")
    assert sorted(cap) == ["p2020", "p2025", "p2030", "p2035"]
    # Capacity in service in p2030: the decisions of p2020 and p2025 (in service from p2025 and p2030).
    coal = {c: v for c, v in cap["p2030"][0].items() if c.startswith(f"v_invest_p[{PLANT},")}
    assert coal == pytest.approx({f"v_invest_p[{PLANT},p2020]": us, f"v_invest_p[{PLANT},p2025]": us}, rel=1e-12)
    assert not any(c == f"v_invest_p[{PLANT},p2035]" for t, _ in cap.values() for c in t)


GROW_ENTITIES = [["constraint", "c_grow", None]]
# Wind must be built to at least half of coal_plant's pre-built capacity in each period; wind investment is
# expensive, so the row binds.  coal_plant retires exactly 200 MW in p2020 and nothing later.
GROW_VALUES = (_constraint("c_grow")
               + [["unit", PLANT, "invest_method", b64("retire_period", "str")],
                  ["unit", PLANT, "retire_min_period",
                   _map({"p2020": 200.0, "p2025": 0.0, "p2030": 0.0, "p2035": 0.0}, "period")],
                  ["unit", PLANT, "retire_max_period",
                   _map({"p2020": 200.0, "p2025": 0.0, "p2030": 0.0, "p2035": 0.0}, "period")],
                  ["unit", PLANT, "constraint_cumulative_pre_built_capacity_coeff", _map({"c_grow": -0.5})],
                  ["unit", "wind_plant", "invest_cost", b64(1.0e6, "float")],
                  ["unit", "wind_plant", "constraint_invested_capacity_coeff", _map({"c_grow": 1.0})]])


def _wind_invest_mw(step, period: str) -> float:
    us = float(utf8(step.flex_data.p_unitsize.frame).filter(pl.col("p") == "wind_plant")["value"][0])
    rows = utf8(step.solution.value("v_invest_p")).filter((pl.col("p") == "wind_plant") & (pl.col("d") == period))
    assert rows.height == 1, rows
    return float(rows["value"][0]) * us


def test_prebuilt_nets_earlier_retirements_single_equals_chain(tmp_path_factory) -> None:
    """Hand value: coal_plant existing 500 MW, 200 MW retired in p2020 → 300 MW in service at the start of
    p2025, so wind investment in p2025 = 0.5 × 300 = 150 MW, in a single solve and in a chain alike (before,
    the single solve ignored the retirement and built 250 MW)."""
    single_wind = ["wind_invest" if a == "coal_invest" else a for a in SINGLE] + ["coal_invest"]
    chain_wind = ["wind_invest" if a == "coal_invest" else a for a in CHAIN] + ["coal_invest"]
    url, scen = make_db(tmp_path_factory, alt="uc_grow_single", values=GROW_VALUES, base_chain=single_wind,
                        entities=GROW_ENTITIES)
    single = list(run(tmp_path_factory, url, scen).values())[-1]
    url, scen = make_db(tmp_path_factory, alt="uc_grow_chain", values=GROW_VALUES, base_chain=chain_wind,
                        entities=GROW_ENTITIES)
    chain = {k: s for k, s in run(tmp_path_factory, url, scen).items()}
    in_2025 = next(s for k, s in chain.items() if "y2025" in k)
    assert _wind_invest_mw(single, "p2020") == pytest.approx(250.0, rel=1e-6)
    assert _wind_invest_mw(single, "p2025") == pytest.approx(150.0, rel=1e-6)
    assert _wind_invest_mw(in_2025, "p2025") == pytest.approx(_wind_invest_mw(single, "p2025"), rel=1e-6)


def test_prebuilt_retirement_notice(tmp_path_factory, capsys) -> None:
    """With the upgrade marker, a run prints the notice for a constraint whose pre-built entity can retire,
    and not for one whose entity cannot."""
    from flextool.engine import _behaviour_changes as bc

    values = GROW_VALUES + _constraint("c_fixed") + [
        ["unit", "wind_plant", "constraint_cumulative_pre_built_capacity_coeff", _map({"c_fixed": 1.0})]]
    single_wind = ["wind_invest" if a == "coal_invest" else a for a in SINGLE] + ["coal_invest"]
    url, scen = make_db(tmp_path_factory, alt="uc_grow_notice", values=values, base_chain=single_wind,
                        entities=GROW_ENTITIES + [["constraint", "c_fixed", None]])
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    capsys.readouterr()
    bc.start_run({"user_constraint_prebuilt_nets_retirements": "77"})
    try:
        build(step.flex_data)
        out = capsys.readouterr().out
        bc.start_run({})
        build(step.flex_data)
        silent = capsys.readouterr().out
    finally:
        bc.start_run({})
    lines = [ln for ln in out.splitlines() if "constraint_cumulative_pre_built_capacity_coeff" in ln]
    assert len(lines) == 1, out
    assert "Affected user constraints: c_grow." in lines[0]
    assert "constraint_cumulative_pre_built_capacity_coeff" not in silent


PERIOD_ENTITIES = [["constraint", "c_min_coal", None]]
# coal_plant capacity in service must reach 600 MW (existing 500): the row binds and forces investment.
PERIOD_VALUES = ([["constraint", "c_min_coal", "sense", b64("greater_than", "str")],
                  ["constraint", "c_min_coal", "constant", b64(600.0, "float")],
                  ["unit", PLANT, "constraint_capacity_coeff", _map({"c_min_coal": 1.0})]])
# The same constraint with a zero flow coefficient: one row per step.
STEP_VALUES = PERIOD_VALUES + [["unit__outputNode", [PLANT, "west"], "constraint_flow_coeff",
                                _map({"c_min_coal": 0.0})]]


def test_capacity_only_rows_per_period_same_objective(tmp_path_factory) -> None:
    url, scen = make_db(tmp_path_factory, alt="uc_rows_period", values=PERIOD_VALUES, base_chain=SINGLE,
                        entities=PERIOD_ENTITIES)
    per_period = list(run(tmp_path_factory, url, scen).values())[-1]
    url, scen = make_db(tmp_path_factory, alt="uc_rows_step", values=STEP_VALUES, base_chain=SINGLE,
                        entities=PERIOD_ENTITIES)
    per_step = list(run(tmp_path_factory, url, scen).values())[-1]
    pb_period, pb_step = build(per_period.flex_data), build(per_step.flex_data)
    n_periods = per_period.flex_data.dt.select("d").unique().height
    n_steps = per_period.flex_data.dt.height
    assert len(row_bounds(pb_period, GE + "_period", "c_min_coal,")) == n_periods
    assert not row_bounds(pb_period, GE, "c_min_coal,")
    assert len(row_bounds(pb_step, GE, "c_min_coal,")) == n_steps
    assert not row_bounds(pb_step, GE + "_period", "c_min_coal,")
    # The capacity-only battery_tie_kW_kWh (invested terms only) is per period in both.
    assert len(row_bounds(pb_step, "process_constraint_equal_period", "battery_tie_kW_kWh,")) == n_periods
    assert per_period.solution.obj == pytest.approx(per_step.solution.obj, rel=1e-9)
    inv = utf8(per_period.solution.value("v_invest_p")).filter(pl.col("p") == PLANT)
    assert inv["value"].sum() > 0


@pytest.mark.parametrize(("kind", "invest", "divest"), [
    ("constraint_cumulative_pre_built_capacity_coeff", True, True),
    ("constraint_capacity_coeff", True, True),
    ("constraint_retired_capacity_coeff", False, True),
])
def test_coefficient_gives_zero_cost_entity_its_variables(tmp_path_factory, kind, invest, divest) -> None:
    """A zero-cost entity (no annuity, no salvage) gets investment / retirement pairs only from a link."""
    values = (_constraint("c_link", constant=None)
              + [["unit", "wind_plant", "invest_method", b64("invest_retire_no_limit", "str")],
                 ["unit", "wind_plant", "invest_cost", b64(0.0, "float")],
                 ["unit", "wind_plant", "fixed_cost", b64(0.0, "float")],
                 ["constraint", "c_link", "constant", b64(-1.0e9, "float")],
                 ["unit", "wind_plant", kind, _map({"c_link": 1.0})]])
    single_wind = ["wind_invest" if a == "coal_invest" else a for a in SINGLE]
    url, scen = make_db(tmp_path_factory, alt=f"uc_link_{kind[11:20]}", values=values, base_chain=single_wind,
                        entities=[["constraint", "c_link", None]])
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    fd = step.flex_data

    def has(frame) -> bool:
        return frame is not None and utf8(frame).filter(pl.col("e") == "wind_plant").height > 0

    assert has(fd.ed_invest_set) is invest
    assert has(fd.ed_divest_set) is divest


@pytest.mark.parametrize("chain", [SINGLE, CHAIN], ids=["single_solve", "chain_of_solves"])
def test_constant_kept_beside_capacity_terms_of_other_constraints(tmp_path_factory, chain) -> None:
    """Constraints without existing-capacity terms keep their constant as the right-hand side when other
    constraints have such terms: the per-step ``flow ≤ 7`` and the once-per-period ``invested ≤ 5`` (hand
    values), in a single solve and in a chain of solves.  Before: the (cn,) constant summed with the (cn, d)
    existing-capacity constants of the other constraints lost its period, and the rows read ``≤ 0``."""
    unit_twin = [v for v in _twin_values("unit", PLANT) if v[2] != "constraint_invested_capacity_coeff"]
    values = (RETIRABLE + _constraint("c_cap") + _constraint("c_twin") + unit_twin
              + [["unit", PLANT, "constraint_invested_capacity_coeff", _map({"c_twin": 1.0, "c_inv": 1.0})]]
              + _twin_values("node", "battery")
              + _constraint("c_flow_cap", "less_than", 7.0) + _constraint("c_inv", "less_than", 5.0)
              + [["unit__outputNode", [PLANT, "west"], "constraint_flow_coeff", _map({"c_flow_cap": 1.0})]])
    url, scen = make_db(tmp_path_factory, alt="uc_flow_cap", values=values, base_chain=chain,
                        entities=TWIN_ENTITIES + [["constraint", "c_flow_cap", None], ["constraint", "c_inv", None]])
    for step in run(tmp_path_factory, url, scen).values():
        pb = build(step.flex_data)
        bounds = row_bounds(pb, "process_constraint_less_than", "c_flow_cap,")
        assert len(bounds) == step.flex_data.dt.height
        assert {upper for _, upper in bounds.values()} == {7.0}
        per_period = _period_rows(pb, "process_constraint_less_than_period", "c_inv")
        assert per_period, "no once-per-period rows"
        assert {upper for _, (_, upper) in per_period.values()} == {5.0}
        assert all(any(c.startswith("v_invest") for c in terms) for terms, _ in per_period.values())
        _assert_twin(pb)
