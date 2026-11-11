"""User constraints summed over a period (``constraint_period_method`` ``period_total``, database version 79).

The row (one per constraint and period) is the period's time-integrated, annualised left side:

    Σ_t w[d,t]·(flow and online terms)  +  H[d]·(capacity terms and their existing constants)  sense  constant
    w[d,t] = step_duration · timestep_weight / period_share,   H[d] = Σ_t w[d,t]

Hand-built model (on ``tests/fixtures/tests.json``, alternative ``init``: period p2020, timeset ``2day`` cut to
four 1-h steps, so period_share = 4/8760 and w = 2190 per step; never a checked-in ``.sqlite``): node ``elec`` with
100 MW demand; ``csp`` (no fuel, 100 MW, profile upper_limit [1, 1, 0.5, 0]), ``gas`` (fuel 50 €/MWh, 100 MW) and
``oil`` (fuel 100 €/MWh, 100 MW).  The residual after csp is [0, 0, 50, 100]; the gas split between steps 3 and 4
is degenerate, so the tests assert sums.

Hand values (design b7 "Oracle", derived independently of the code):

    case                                   binding row (raw MWh over the 4 h)        gas     oil
    A   gas − 0.15·csp ≤ 0, period_total     Σgas ≤ 0.15·250                         37.5    112.5
    A   same, per_timestep                   gas₃ ≤ 7.5, gas₄ ≤ 0                    7.5     142.5
    A-rp  rep weights → tw [4/3,4/3,2/3,2/3]  Σtw·gas ≤ 0.15·Σtw·csp = 45 (steps 3+4) 67.5    82.5
    B   gas − 0.3·capacity(gas) ≤ 0,          2190·Σgas ≤ 0.3·8760·50 → Σgas ≤ 60      60      340
        gas existing 50, no csp
    C   gas ≤ 87 600 (MWh/yr), no csp         2190·Σgas ≤ 87 600                       40      360
    B-inv  A's csp, the B row, gas existing   Σgas ≤ 1.2·cap (0.3 × 4 h)               150     0
           0 and investable (fixed_cost       cap* = 125 MW (100 without the row: the row forces overbuild)
           60 €/kW/yr, saving 131 400 €/MW/yr)
    B-inv with gas existing 50                 Σgas ≤ 1.2·(50 + x) → x* = 75 (≈ 125 if the existing constant
                                               were not × H)

Also: a period_total constraint with a ``constraint_state_coeff`` is a configuration error; a rolling solve whose
rolls do not realize the period whole leaves the row out with a warning, and with no other solve the run stops; a
rolling solve with a jump of one whole period writes the row in each roll.
"""
from __future__ import annotations

import logging

import polars as pl
import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from tests.engine._lp_rows import row_bounds, row_terms
from tests.engine._unit_db import RunCache, b64, build, make_db, utf8

pytestmark = pytest.mark.solver

SOLVE = "y2020_2day_dispatch"
LE = "process_constraint_less_than"
PT = LE + "_period_total"
TOL = 1e-6
#: The steps the period stands for in a year: 8760 h; one step of the four 1-h steps stands for 2190 h.
W = 8760.0 / 4.0

_RUNS = RunCache()


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _f(x: float) -> list:
    return b64(float(x), "float")


def _s(x: str) -> list:
    return b64(x, "str")


def _map(data: dict, index_name: str = "constraint") -> list:
    return b64({"index_type": "str", "index_name": index_name, "data": [[k, v] for k, v in data.items()]}, "map")


def _entities(*, csp: bool = True) -> list:
    out = [["node", "elec", None], ["node", "gas_n", None], ["node", "oil_n", None],
           ["commodity", "gasc", None], ["commodity", "oilc", None],
           ["commodity__node", ["gasc", "gas_n"], None], ["commodity__node", ["oilc", "oil_n"], None],
           ["group__node", ["electricity", "elec"], None],
           ["unit", "gas", None], ["unit", "oil", None],
           ["unit__inputNode", ["gas", "gas_n"], None], ["unit__inputNode", ["oil", "oil_n"], None],
           ["unit__outputNode", ["gas", "elec"], None], ["unit__outputNode", ["oil", "elec"], None],
           ["constraint", "c", None]]
    if csp:
        out += [["unit", "csp", None], ["unit__outputNode", ["csp", "elec"], None],
                ["profile", "csp_prof", None], ["unit__node__profile", ["csp", "elec", "csp_prof"], None]]
    return out


def _values(*, method: str = "period_total", csp: bool = True, gas_existing: float = 100.0,
            terms: dict | None = None, constant: float = 0.0, extra: list | None = None,
            durations: dict | None = None, sense: str = "less_than") -> list:
    """The base model with constraint ``c``: ``terms`` maps (class, entity tuple, coefficient parameter) to the
    coefficient of ``c`` (default: the 15 % share ``gas − 0.15·csp``)."""
    if terms is None:
        terms = {("unit__outputNode", ("gas", "elec"), "constraint_flow_coeff"): 1.0,
                 ("unit__outputNode", ("csp", "elec"), "constraint_flow_coeff"): -0.15}
    durations = durations or {"index_type": "str", "data": [["t0001", 4.0]]}
    vals = [["timeset", "2day", "timeset_duration", b64(durations, "map")],
            ["node", "elec", "node_type", _s("balance")], ["node", "elec", "inflow", _f(-100.0)],
            ["node", "gas_n", "node_type", _s("commodity")], ["node", "oil_n", "node_type", _s("commodity")],
            ["commodity", "gasc", "price", _f(50.0)], ["commodity", "oilc", "price", _f(100.0)],
            ["unit", "gas", "existing", _f(gas_existing)], ["unit", "oil", "existing", _f(100.0)],
            ["constraint", "c", "sense", _s(sense)], ["constraint", "c", "constant", _f(constant)],
            ["constraint", "c", "constraint_period_method", _s(method)]]
    if csp:
        vals += [["unit", "csp", "existing", _f(100.0)],
                 ["profile", "csp_prof", "profile",
                  _map({"t0001": 1.0, "t0002": 1.0, "t0003": 0.5, "t0004": 0.0}, "time")],
                 ["unit__node__profile", ["csp", "elec", "csp_prof"], "profile_method", _s("upper_limit")]]
    for (cls, ent, param), value in terms.items():
        vals.append([cls, list(ent) if len(ent) > 1 else ent[0], param, _map({"c": value})])
    return vals + list(extra or [])


def _steps(tmp_path_factory, alt: str, values: list, *, csp: bool = True) -> dict:
    steps, _wf, _scen = _RUNS.get(tmp_path_factory, alt=alt, values=values, base_chain=["init"],
                                  entities=_entities(csp=csp))
    for name, step in steps.items():
        assert step.solution is not None, f"{name}: no solution"
    return steps


def _step(tmp_path_factory, alt: str, values: list, *, csp: bool = True):
    steps = _steps(tmp_path_factory, alt, values, csp=csp)
    assert len(steps) == 1, list(steps)
    return next(iter(steps.values()))


def _energy(steps, unit: str) -> float:
    """Σ over the steps (and the solves) of the unit's output (MWh: 1-h steps)."""
    total = 0.0
    for step in (steps.values() if isinstance(steps, dict) else [steps]):
        flows = utf8(step.solution.value("v_flow")).filter(pl.col("p") == unit)
        us = utf8(step.flex_data.p_unitsize.frame).filter(pl.col("p") == unit)["value"][0]
        realized = utf8(step.flex_data.realized_dispatch) if isinstance(steps, dict) else None
        if realized is not None and realized.height:
            keys = realized.select(pl.col(realized.columns[0]).alias("d"), pl.col(realized.columns[1]).alias("t"))
            flows = flows.join(keys, on=["d", "t"], how="semi")
        total += float(flows["value"].sum()) * float(us)
    return total


def _hours(step) -> dict[str, float]:
    """H[d] = Σ_t step_duration · timestep_weight / period_share of the step's FlexData."""
    fd = step.flex_data
    w = (utf8(fd.p_step_duration.frame).rename({"value": "sd"})
         .join(utf8(fd.p_timestep_weight.frame).rename({"value": "tw"}), on=["d", "t"])
         .join(utf8(fd.p_period_share.frame).rename({"value": "ps"}), on="d"))
    out = w.group_by("d").agg((pl.col("sd") * pl.col("tw") / pl.col("ps")).sum().alias("h"))
    return dict(zip(out["d"].to_list(), out["h"].to_list()))


def _invested(step, unit: str) -> float:
    inv = utf8(step.solution.value("v_invest_p")).filter(pl.col("p") == unit)
    us = utf8(step.flex_data.p_unitsize.frame).filter(pl.col("p") == unit)["value"][0]
    return float(inv["value"].sum()) * float(us)


# ─── A: the 15 % share ──────────────────────────────────────────────────


def test_share_period_total(tmp_path_factory) -> None:
    step = _step(tmp_path_factory, "pt_a", _values())
    assert _energy(step, "csp") == pytest.approx(250.0, abs=TOL)
    assert _energy(step, "gas") == pytest.approx(37.5, abs=TOL)
    assert _energy(step, "oil") == pytest.approx(112.5, abs=TOL)
    pb = build(step.flex_data)
    assert list(row_terms(pb, PT, "c,")) == [f"{PT}[c,p2020]"]
    assert not row_terms(pb, LE, "c,")
    # The coefficients are w = 2190 on every step's flow (gas 1, csp −0.15).
    terms = row_terms(pb, PT, "c,")[f"{PT}[c,p2020]"]
    us = {u: float(utf8(step.flex_data.p_unitsize.frame).filter(pl.col("p") == u)["value"][0]) for u in ("gas",)}
    gas = sorted(v for k, v in terms.items() if "gas" in k)
    assert gas == pytest.approx([W * us["gas"]] * 4, rel=1e-12)


def test_share_per_timestep(tmp_path_factory) -> None:
    step = _step(tmp_path_factory, "pt_a_step", _values(method="per_timestep"))
    assert _energy(step, "gas") == pytest.approx(7.5, abs=TOL)
    assert _energy(step, "oil") == pytest.approx(142.5, abs=TOL)
    pb = build(step.flex_data)
    assert len(row_terms(pb, LE, "c,")) == 4
    assert not row_terms(pb, PT, "c,")


def test_share_with_representative_weights(tmp_path_factory) -> None:
    """Two representative blocks (t0001–t0002 weight 2, t0003–t0004 weight 1): tw = [4/3, 4/3, 2/3, 2/3], so the
    weighted csp is 300 and Σ_{3,4} tw·gas ≤ 45 → gas₃ + gas₄ = 67.5 (37.5 if tw were ignored), oil 82.5."""
    rpw = {"type": "map", "index_type": "str", "index_name": "base_period", "rank": 1,
           "data": [["t0001", {"type": "map", "index_type": "str", "index_name": "representative_period",
                               "rank": 1, "data": [["t0001", 2.0 / 3.0], ["t0003", 1.0 / 3.0]]}]]}
    durations = {"index_type": "str", "index_name": "time", "data": [["t0001", 2.0], ["t0003", 2.0]]}
    extra = [["timeset", "2day", "representative_period_weights", b64(rpw, "map")]]
    step = _step(tmp_path_factory, "pt_a_rp", _values(extra=extra, durations=durations))
    tw = utf8(step.flex_data.p_timestep_weight.frame).sort("t")["value"].to_list()
    assert tw == pytest.approx([4 / 3, 4 / 3, 2 / 3, 2 / 3], rel=1e-12)
    assert _hours(step) == {"p2020": pytest.approx(8760.0, rel=1e-12)}
    assert _energy(step, "gas") == pytest.approx(67.5, abs=TOL)
    assert _energy(step, "oil") == pytest.approx(82.5, abs=TOL)


# ─── B, C: capacity and constants are annual ────────────────────────────


GAS_FLOW = ("unit__outputNode", ("gas", "elec"), "constraint_flow_coeff")
GAS_CAP = ("unit", ("gas",), "constraint_capacity_coeff")


def test_capacity_factor(tmp_path_factory) -> None:
    """gas − 0.3·capacity(gas) ≤ 0 with 50 MW existing: the capacity (and its existing constant) count H = 8760 h,
    so Σgas ≤ 0.3·8760·50 / 2190 = 60 MWh (0.007 if capacity were raw MW)."""
    step = _step(tmp_path_factory, "pt_b",
                 _values(csp=False, gas_existing=50.0, terms={GAS_FLOW: 1.0, GAS_CAP: -0.3}), csp=False)
    assert _energy(step, "gas") == pytest.approx(60.0, abs=TOL)
    assert _energy(step, "oil") == pytest.approx(340.0, abs=TOL)
    # The existing capacity is a constant: on the right-hand side, 0.3·50·8760.
    (lower, upper), = row_bounds(build(step.flex_data), PT, "c,").values()
    assert upper == pytest.approx(0.3 * 50.0 * 8760.0, rel=1e-12)


def test_annual_constant(tmp_path_factory) -> None:
    """gas ≤ 87 600 MWh per year: 2190·Σgas ≤ 87 600 → Σgas = 40 MWh, oil 360."""
    step = _step(tmp_path_factory, "pt_c", _values(csp=False, terms={GAS_FLOW: 1.0}, constant=87600.0), csp=False)
    assert _energy(step, "gas") == pytest.approx(40.0, abs=TOL)
    assert _energy(step, "oil") == pytest.approx(360.0, abs=TOL)


def test_capacity_only_row_without_a_variable(tmp_path_factory) -> None:
    """A capacity-only period_total row on existing capacity has no variable: its constant 0.3·50 MW·8760 h =
    131 400 cannot be ≤ 100 000, a configuration error (the raw MW, 15, would hold and only warn)."""
    from flextool.engine import run_chain_from_db

    values = _values(csp=False, gas_existing=50.0, terms={GAS_CAP: 0.3}, constant=100000.0)
    url, scen = make_db(tmp_path_factory, alt="pt_cap_only", values=values, base_chain=["init"],
                        entities=_entities(csp=False))
    with pytest.raises(FlexToolConfigError, match=r"c \(period p2020: 131400 <= 100000\)"):
        run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp("pt_cap_only_w"))


def _invest(existing: float) -> list:
    """gas investable at a fixed cost of 60 €/kW/yr, the capacity-factor row on gas, A's csp.  The token 1 €/kW
    investment cost is for the control case (test_investment_without_the_row): without it and without a capacity
    term an entity is not offered investment (github_issues item 218); the annuity, ~0.09 €/kW/yr, does not move
    the oracle."""
    return _values(gas_existing=existing, terms={GAS_FLOW: 1.0, GAS_CAP: -0.3}, extra=[
        ["unit", "gas", "invest_method", _s("invest_no_limit")],
        ["unit", "gas", "fixed_cost", _f(60.0)],
        ["unit", "gas", "invest_cost", _f(1.0)],
        ["unit", "gas", "lifetime", _f(30.0)],
        ["solve", SOLVE, "invest_periods",
         b64({"value_type": "str", "data": ["p2020"], "index_name": "period"}, "array")]])


@pytest.mark.parametrize("existing, built", [(0.0, 125.0), (50.0, 75.0)], ids=["new", "with_existing"])
def test_capacity_factor_forces_investment(tmp_path_factory, existing: float, built: float) -> None:
    """Σgas ≤ 1.2·(existing + x): gas covers the whole residual 150 MWh, so x = 150 / 1.2 − existing (a gas MW
    saves 131 400 €/yr of oil at a fixed cost of 60 000 €/yr).  Without the row 100 MW would cover the peak."""
    step = _step(tmp_path_factory, f"pt_binv_{int(existing)}", _invest(existing))
    assert _invested(step, "gas") == pytest.approx(built, abs=1e-4)
    assert _energy(step, "gas") == pytest.approx(150.0, abs=1e-4)
    assert _energy(step, "oil") == pytest.approx(0.0, abs=1e-4)


def test_investment_without_the_row(tmp_path_factory) -> None:
    """Control for the investment cases: without the constraint the investment covers the peak residual of
    100 MW."""
    values = [v for v in _invest(0.0) if v[0] != "constraint"
              and v[2] not in ("constraint_flow_coeff", "constraint_capacity_coeff")]
    step = _step(tmp_path_factory, "pt_binv_none", values)
    assert _invested(step, "gas") == pytest.approx(100.0, abs=1e-4)


# ─── online and coarse terms ────────────────────────────────────────────


GAS_ONLINE = ("unit", ("gas",), "constraint_online_coeff")


@pytest.mark.parametrize("startup", ["binary", "linear"])
def test_online_unit_hours(tmp_path_factory, startup: str) -> None:
    """online(gas) ≥ 6570 unit-hours per year, gas with min_load 0.5 (one unit of 100 MW): each step's units
    online count w = 2190 h, so Σ_t online ≥ 3.  Without the row gas is online only in steps 3 and 4 (residual 50
    and 100); the third unit-step falls in steps 1–2, where gas must run ≥ 50 MW per unit online and csp is
    curtailed for it: gas 50 + 50 + 100 = 200 MWh, oil 0.  (Raw unit-steps, w ignored, would be infeasible.)"""
    extra = [["unit", "gas", "startup_method", _s(startup)], ["unit", "gas", "min_load", _f(0.5)],
             ["solve", SOLVE, "solver_mip_gap", _f(0.0)]]
    step = _step(tmp_path_factory, f"pt_online_{startup}",
                 _values(terms={GAS_ONLINE: 1.0}, constant=3 * W, sense="greater_than", extra=extra))
    var = "v_online_integer" if startup == "binary" else "v_online_linear"
    online = utf8(step.solution.value(var)).filter(pl.col("p") == "gas").sort("t")
    assert float(online["value"].sum()) == pytest.approx(3.0, abs=TOL)
    assert online.filter(pl.col("t").is_in(["t0003", "t0004"]))["value"].to_list() == pytest.approx([1.0, 1.0],
                                                                                                    abs=TOL)
    assert _energy(step, "gas") == pytest.approx(200.0, abs=TOL)
    assert _energy(step, "oil") == pytest.approx(0.0, abs=TOL)
    pb = build(step.flex_data)
    (row, terms), = row_terms(pb, "process_constraint_greater_than_period_total", "c,").items()
    assert row == "process_constraint_greater_than_period_total[c,p2020]"
    assert sorted(v for k, v in terms.items() if k.startswith(var)) == pytest.approx([W] * 4, rel=1e-12)
    (lower, _upper), = row_bounds(pb, "process_constraint_greater_than_period_total", "c,").values()
    assert lower == pytest.approx(3 * W, rel=1e-12)


def test_share_with_a_coarse_member(tmp_path_factory) -> None:
    """A's share with a coarse member: heat node ``h`` (10 MW demand) in group ``hgrp`` with new_stepduration 2 h,
    served by ``boiler`` (gas_n -> h, 50 €/MWh) and ``oboil`` (oil_n -> h, 100 €/MWh), both coarse (every balance
    node coarse); c is boiler − 0.15·csp ≤ 0.  The boiler's one flow per 2-h step counts on both of its fine steps
    (coefficient 2·w per column), so Σboiler ≤ 0.15·250 = 37.5 MWh and oboil makes the other 2.5 of the 40 MWh
    (with one fine step per coarse column the limit would be 75 and oboil 0); elec as without c: gas 150, oil 0."""
    ents = _entities() + [["node", "h", None], ["group", "hgrp", None], ["group__node", ["hgrp", "h"], None],
                          ["unit", "boiler", None], ["unit", "oboil", None],
                          ["unit__inputNode", ["boiler", "gas_n"], None], ["unit__outputNode", ["boiler", "h"], None],
                          ["unit__inputNode", ["oboil", "oil_n"], None], ["unit__outputNode", ["oboil", "h"], None]]
    extra = [["group", "hgrp", "new_stepduration", _f(2.0)],
             ["node", "h", "node_type", _s("balance")], ["node", "h", "inflow", _f(-10.0)],
             ["unit", "boiler", "existing", _f(100.0)], ["unit", "oboil", "existing", _f(100.0)],
             ["unit", "boiler", "efficiency", _f(1.0)], ["unit", "oboil", "efficiency", _f(1.0)]]
    terms = {("unit__outputNode", ("boiler", "h"), "constraint_flow_coeff"): 1.0,
             ("unit__outputNode", ("csp", "elec"), "constraint_flow_coeff"): -0.15}
    steps, _wf, _scen = _RUNS.get(tmp_path_factory, alt="pt_coarse", values=_values(terms=terms, extra=extra),
                                  base_chain=["init"], entities=ents)
    step, = steps.values()
    fd = step.flex_data
    assert set(utf8(fd.processCoarse)["p"].to_list()) == {"boiler", "oboil"}
    assert _coarse_energy(step, "boiler") == pytest.approx(37.5, abs=TOL)
    assert _coarse_energy(step, "oboil") == pytest.approx(2.5, abs=TOL)
    assert _energy(step, "gas") == pytest.approx(150.0, abs=TOL)
    assert _energy(step, "oil") == pytest.approx(0.0, abs=TOL)
    (_row, row), = row_terms(build(fd), PT, "c,").items()
    us = float(utf8(fd.p_unitsize.frame).filter(pl.col("p") == "boiler")["value"][0])
    assert sorted(v for k, v in row.items() if "boiler" in k) == pytest.approx([2 * W * us] * 2, rel=1e-12)


def _coarse_energy(step, unit: str) -> float:
    """The output in MWh of a unit on the 2-h coarse steps: one flow (a rate) per coarse step, at its first fine
    step (t0001, t0003), × 2 h."""
    flows = utf8(step.solution.value("v_flow")).filter(pl.col("p") == unit)
    assert sorted(flows["t"].to_list()) == ["t0001", "t0003"]
    us = utf8(step.flex_data.p_unitsize.frame).filter(pl.col("p") == unit)["value"][0]
    return float(flows["value"].sum()) * float(us) * 2.0


# ─── D: the state-term error ────────────────────────────────────────────


def test_state_coefficient_is_an_error(tmp_path_factory) -> None:
    from flextool.engine import run_chain_from_db

    values = _values(extra=[["node", "elec", "constraint_state_coeff", _map({"c": 1.0})]])
    url, scen = make_db(tmp_path_factory, alt="pt_state", values=values, base_chain=["init"], entities=_entities())
    with pytest.raises(FlexToolConfigError, match=r"period_total cannot have a constraint_state_coeff: c \(node elec\)"):
        run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp("pt_state_w"))


# ─── D: rolling solves ──────────────────────────────────────────────────


ROLL = "b7_roll"
TWO_PERIODS = b64({"index_type": "str", "index_name": "period", "data": [["p2020", "2day"], ["p2025", "2day"]]},
                  "map")


def _rolling(jump: float, horizon: float, *, periods: list | None = None, also_single: bool = False) -> list:
    """The rolling solve ``b7_roll`` over the four steps of each period in ``periods`` (default p2020); with
    ``also_single`` the run solves the single solve y2020_2day_dispatch (p2020) first."""
    periods = periods or ["p2020"]
    timesets = b64({"index_type": "str", "index_name": "period", "data": [[p, "2day"] for p in periods]}, "map")
    solves = [SOLVE, ROLL] if also_single else [ROLL]
    return [["solve", ROLL, "solve_mode", _s("rolling_window")],
            ["solve", ROLL, "rolling_solve_jump", _f(jump)],
            ["solve", ROLL, "rolling_solve_horizon", _f(horizon)],
            ["solve", ROLL, "period_timeset", timesets],
            ["solve", ROLL, "realized_periods", b64({"value_type": "str", "data": periods, "index_name": "period"},
                                                    "array")],
            ["model", "flexTool", "solves", b64({"value_type": "str", "data": solves, "index_name": "sequence_index"},
                                                "array")]]


def _roll_steps(tmp_path_factory, alt: str, extra: list) -> dict:
    steps, _wf, _scen = _RUNS.get(tmp_path_factory, alt=alt, values=_values(extra=extra), base_chain=["init"],
                                  entities=_entities() + [["solve", ROLL, None]])
    return steps


def test_rolling_whole_periods_write_the_row_per_roll(tmp_path_factory) -> None:
    """Rolls that jump one whole period (4 h) over p2020 and p2025, horizon 6 h: each roll writes the row of the
    period it realizes, not of the look-ahead tail (p2025 half in roll 0), and each period's dispatch is A's."""
    steps = _roll_steps(tmp_path_factory, "pt_roll_whole", _rolling(4.0, 6.0, periods=["p2020", "p2025"]))
    assert list(steps) == [f"{ROLL}_roll_0", f"{ROLL}_roll_1"]
    rows = {name: sorted(row_terms(build(step.flex_data), PT, "c,")) for name, step in steps.items()}
    assert rows == {f"{ROLL}_roll_0": [f"{PT}[c,p2020]"], f"{ROLL}_roll_1": [f"{PT}[c,p2025]"]}
    for step in steps.values():
        assert not step.flex_data.user_constraint_period_total_off
    assert _energy(steps, "gas") == pytest.approx(2 * 37.5, abs=TOL)
    assert _energy(steps, "oil") == pytest.approx(2 * 112.5, abs=TOL)


def test_rolling_split_period_is_left_out_with_a_warning(tmp_path_factory, caplog) -> None:
    """Rolls of 2 h over the 4-h period p2020 after the single solve of p2020: the rolling solve leaves the row out
    (warning naming the constraint, the solve and the period); the single solve enforces it."""
    caplog.set_level(logging.WARNING, logger="flextool.engine.solve._recursive_solve")
    from flextool.engine import run_chain_from_db

    url, scen = make_db(tmp_path_factory, alt="pt_roll_split", values=_values(extra=_rolling(2.0, 4.0,
                                                                                             also_single=True)),
                        base_chain=["init"], entities=_entities() + [["solve", ROLL, None]])
    steps = run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp("pt_roll_split_w"), keep_solutions=True)
    assert list(steps) == [SOLVE, f"{ROLL}_roll_0", f"{ROLL}_roll_1"]
    assert not steps[SOLVE].flex_data.user_constraint_period_total_off
    assert list(row_terms(build(steps[SOLVE].flex_data), PT, "c,")) == [f"{PT}[c,p2020]"]
    for name in (f"{ROLL}_roll_0", f"{ROLL}_roll_1"):
        assert steps[name].flex_data.user_constraint_period_total_off
        assert not row_terms(build(steps[name].flex_data), PT, "c,")
    warnings = [r.getMessage() for r in caplog.records if "period_total" in r.getMessage()]
    assert len(warnings) == 1, warnings
    assert "'c'" in warnings[0] and f"rolling solve '{ROLL}'" in warnings[0] and "period 'p2020'" in warnings[0]
    assert "rolling_solve_jump 2 h" in warnings[0]


def test_rolling_split_period_without_another_solve_is_an_error(tmp_path_factory) -> None:
    from flextool.engine import run_chain_from_db

    url, scen = make_db(tmp_path_factory, alt="pt_roll_only", values=_values(extra=_rolling(2.0, 4.0)),
                        base_chain=["init"], entities=_entities() + [["solve", ROLL, None]])
    with pytest.raises(FlexToolConfigError, match=r"'c' have constraint_period_method period_total, but no solve "
                                                   r"of the run enforces them: the rolls of 'b7_roll'"):
        run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp("pt_roll_only_w"))


def test_period_total_without_a_sense_is_not_checked(tmp_path_factory) -> None:
    """A period_total constraint without a ``sense`` has no rows anywhere: the split rolls neither warn nor stop
    the run."""
    values = [v for v in _values(extra=_rolling(2.0, 4.0)) if not (v[0] == "constraint" and v[2] == "sense")]
    steps = _roll_steps_values(tmp_path_factory, "pt_roll_nosense", values)
    assert list(steps) == [f"{ROLL}_roll_0", f"{ROLL}_roll_1"]
    for step in steps.values():
        assert not step.flex_data.user_constraint_period_total_off


def _roll_steps_values(tmp_path_factory, alt: str, values: list) -> dict:
    steps, _wf, _scen = _RUNS.get(tmp_path_factory, alt=alt, values=values, base_chain=["init"],
                                  entities=_entities() + [["solve", ROLL, None]])
    return steps


# ─── E: stochastic branches ─────────────────────────────────────────────


def _stochastic() -> list:
    """y2020_2day_dispatch with two forecast branches b1 / b2 from t0001; csp's profile is [1, 1, 0.5, 0] in b1 and
    [1, 0.5, 0.5, 0] in b2; the constraint's constant is a period map {p2020: 21 900} (MWh/yr, 10 MWh over the
    four steps), so a branch period that missed its anchor's value would read 0."""
    def level(name: str, data: list) -> dict:
        return {"type": "map", "index_type": "str", "index_name": name, "data": data}

    steps4 = ["t0001", "t0002", "t0003", "t0004"]
    profile = {"b1": [1.0, 1.0, 0.5, 0.0], "b2": [1.0, 0.5, 0.5, 0.0]}
    prof = level("branch", [[b, level("analysis_time", [["t0001", level("time", list(map(list, zip(steps4, v))))]])]
                            for b, v in profile.items()])
    prof.pop("type")
    branches = level("period", [["p2020", level("branch", [
        [b, level("analysis_time", [["t0001", level("realized", [["no", 1.0]])]])] for b in profile])]])
    branches.pop("type")
    values = [v for v in _values() if not (v[0] == "profile" or (v[1] == "c" and v[2] == "constant"))]
    return values + [["profile", "csp_prof", "profile", b64(prof, "map")],
                     ["solve", SOLVE, "stochastic_branches", b64(branches, "map")],
                     ["group", "b7_stoch", "include_stochastics", _s("yes")],
                     ["constraint", "c", "constant", _map({"p2020": 21900.0}, "period")]]


def test_stochastic_branch_periods(tmp_path_factory) -> None:
    """One row per forecast branch period, each with H = 8760 (not n_branch × 8760), and each branch's gas is 15 %
    of its own csp plus the constant's 10 MWh: b1 csp 250 → gas 47.5, oil 102.5; b2 csp 200 → gas 40, oil 160."""
    steps, _wf, _scen = _RUNS.get(tmp_path_factory, alt="pt_stoch", values=_stochastic(), base_chain=["init"],
                                  entities=_entities() + [["group", "b7_stoch", None],
                                                          ["group__unit", ["b7_stoch", "csp"], None]])
    step = steps[SOLVE]
    flows = utf8(step.solution.value("v_flow"))
    us = dict(zip(*utf8(step.flex_data.p_unitsize.frame).select("p", "value").to_dict(as_series=False).values()))
    periods = sorted(flows["d"].unique().to_list())
    hours = _hours(step)
    by_branch = {}
    for d in periods:
        sums = {u: float(flows.filter((pl.col("p") == u) & (pl.col("d") == d))["value"].sum()) * us[u]
                for u in ("csp", "gas", "oil")}
        by_branch[d] = sums
    assert hours == {d: pytest.approx(8760.0, rel=1e-12) for d in periods}
    rows = sorted(row_terms(build(step.flex_data), PT, "c,"))
    assert rows == [f"{PT}[c,{d}]" for d in periods] and len(periods) == 2
    b1, b2 = (by_branch[d] for d in periods)
    assert (b1["csp"], b1["gas"], b1["oil"]) == pytest.approx((250.0, 47.5, 102.5), abs=TOL)
    assert (b2["csp"], b2["gas"], b2["oil"]) == pytest.approx((200.0, 40.0, 160.0), abs=TOL)
