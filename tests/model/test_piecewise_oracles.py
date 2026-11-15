"""The piecewise_linear oracle cases solved (``specs/piecewise_efficiency_oracle.md``; design T5).

Each case is a small database written from scratch (``tests/model/_piecewise_oracle_db.py``: the schema of
``tests/fixtures/tests.json``, never a checked-in ``.sqlite``) and solved with HiGHS (MIP cases with
``solver_mip_gap`` 1e-9).  The expected values are the oracle's hand calculation (exact fractions), compared with
rtol 1e-7 (atol 1e-6 for values that are 0).  The ties the oracle names are not asserted: the start-ups of O1 /
O2c / O4, the units online of O2c at t02 and of O4 in p2035.  O3 checks the solution (the input above the curve);
its post-solve warning is part of the outputs.  Every run has ``FLEXTOOL_AUTOSCALE_STRICT=1``.

Weight of a modelled hour (oracle §0.1): 8760 / Σ step_duration — 2920 for three 1-h steps, 4380 for two.  With
``virtual_unitsize`` the unit's ``existing`` is a number of units (units.md): the oracle's 200 MW of 100-MW units is
``existing`` 2.
"""
from __future__ import annotations

from fractions import Fraction as Fr

import polars as pl
import pytest

from tests.engine._unit_db import build, cstr_over, utf8
from tests.model._piecewise_oracle_db import (
    array, balance_node, commodity_node, curve, curve_per_period, make_oracle_db, merge, scalar, str_map, text,
    time_settings, unit,
)

pytestmark = pytest.mark.solver

RTOL = 1e-7
ATOL = 1e-6
_RUNS: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch) -> None:
    """A missing autoscale registry entry raises instead of falling back to an unscaled LP."""
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _solve(tmp_path_factory, name: str, parts: tuple):
    """The one step of the case's single solve (cached per module)."""
    if name not in _RUNS:
        from flextool.engine import run_chain_from_db

        ents, vals = parts
        url, scen = make_oracle_db(tmp_path_factory, name, ents, vals)
        steps = run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"{name}_w"), keep_solutions=True)
        assert len(steps) == 1, list(steps)
        step = next(iter(steps.values()))
        assert step.solution is not None
        _RUNS[name] = step
    return _RUNS[name]


def _flow(step, p: str, source: str, sink: str) -> list[float]:
    """MW of the arc per (d, t), in (d, t) order (v_flow × unitsize)."""
    us = float(utf8(step.flex_data.p_unitsize.frame).filter(pl.col("p") == p)["value"][0])
    rows = (utf8(step.solution.value("v_flow"))
            .filter((pl.col("p") == p) & (pl.col("source") == source) & (pl.col("sink") == sink)).sort("d", "t"))
    assert rows.height > 0, f"no v_flow of {p} {source}->{sink}"
    return [v * us for v in rows["value"].to_list()]


def _var(step, name: str, p: str) -> list[float]:
    rows = utf8(step.solution.value(name)).filter(pl.col("p") == p).sort("d", "t")
    assert rows.height > 0, f"no {name} of {p}"
    return rows["value"].to_list()


def _slack(step, name: str, n: str) -> list[float]:
    """Balance slack ``vq_state_up`` / ``vq_state_down`` of node n in MW, per (d, t)."""
    rows = utf8(step.solution.value(name)).filter(pl.col("n") == n).sort("d", "t")
    assert rows.height > 0, f"no {name} of {n}"
    return rows["value"].to_list()


def _close(got: list[float], want: list, *, what: str) -> None:
    assert len(got) == len(want), f"{what}: {got} vs {want}"
    for g, w in zip(got, want):
        w = float(w)
        assert g == pytest.approx(w, rel=RTOL, abs=ATOL if w == 0.0 else 0.0), f"{what}: {got} vs {want}"


def _obj(step, want) -> None:
    w = float(want)
    assert float(step.solution.obj) == pytest.approx(w, rel=RTOL, abs=ATOL if w == 0.0 else 0.0)


def _pw(curve_value, **params) -> dict:
    out = {"conversion_method": text("piecewise_linear"), "piecewise_linear_efficiency": curve_value}
    out.update(params)
    return out


# ---------------------------------------------------------------------------
# The cases (oracle §2-§12)

O1_CURVE = curve([(0.4, 0.30), (0.7, 0.40), (1.0, 0.42)])
E_CURVE = curve([(0.2, 0.70), (0.6, 0.66), (1.0, 0.60)])


def _o1(startup: str, inflow: dict, *, startup_cost: float = 0.0, extra_solve: list | None = None) -> tuple:
    ents_c, vals_c = commodity_node("fuel", "fuel_node", 10.0, co2=0.2)
    co2 = ([["group", "co2grp", None], ["group__node", ["co2grp", "fuel_node"], None]],
           [["group", "co2grp", "co2_method", text("price")], ["group", "co2grp", "co2_price", scalar(50.0)]])
    return merge(time_settings(3, ["p2030"], extra_solve=extra_solve), (ents_c, vals_c), co2,
                 balance_node("elec", inflow),
                 unit("G", ["fuel_node"], ["elec"], _pw(
                     O1_CURVE, virtual_unitsize=scalar(100.0), existing=scalar(2.0), startup_method=text(startup),
                     startup_cost=scalar(startup_cost), min_load=scalar(0.4))))


def _o2(e_params: dict, *, inflow=-30.0, extra_solve: list | None = None, power_node: str = "power_node",
        power: tuple | None = None, h2: tuple | None = None) -> tuple:
    params = {"existing": scalar(100.0)}
    params.update(e_params)
    return merge(time_settings(3, ["p2030"], extra_solve=extra_solve),
                 power if power is not None else commodity_node("power", power_node, 50.0),
                 h2 if h2 is not None else balance_node("h2", inflow),
                 unit("E", [power_node], ["h2"], _pw(params.pop("curve", E_CURVE), **params)))


CASES = {
    "O1": lambda: _o1("binary", {"t01": -60.0, "t02": -130.0, "t03": -200.0}),
    "O1-L": lambda: _o1("linear", {"t01": -200.0, "t02": -60.0, "t03": -200.0}, startup_cost=10.0),
    "O1-R": lambda: _o1("linear", {"t01": -200.0, "t02": -60.0, "t03": -200.0}, startup_cost=10.0,
                        extra_solve=[["solve", "s1", "reduce_unit_detail_method", text("online")]]),
    "O2a": lambda: _o2({}),
    "O2b": lambda: _o2({"existing": scalar(0.0), "invest_method": text("invest_no_limit"), "lifetime": scalar(1.0),
                        "discount_rate": scalar(0.0), "invest_cost": scalar(33.652506)},
                       extra_solve=[["solve", "s1", "invest_periods", str_list(["p2030"])],
                                    ["solve", "s1", "realized_invest_periods", str_list(["p2030"])]]),
    "O2c": lambda: _o2({"virtual_unitsize": scalar(50.0), "existing": scalar(2.0), "startup_method": text("binary"),
                        "startup_cost": scalar(0.0)}, inflow={"t01": -30.0, "t02": -5.0, "t03": -50.0}),
    "O2d": lambda: _o2({"startup_method": text("linear")},
                       extra_solve=[["solve", "s1", "reduce_unit_detail_method", text("integers")]]),
    "O3": lambda: _o2({}, power_node="power_bal",
                      power=balance_node("power_bal", 60.0, penalty_down=1000.0, penalty_up=10000.0),
                      h2=balance_node("h2", -30.0, penalty_down=1000.0, penalty_up=10000.0)),
    "O3-c": lambda: _o2({"curve": curve([(0.2, 0.66), (1.0, 0.66)])}, power_node="power_bal",
                        power=balance_node("power_bal", 60.0, penalty_down=1000.0, penalty_up=10000.0),
                        h2=balance_node("h2", -30.0, penalty_down=1000.0, penalty_up=10000.0)),
}


def str_list(values: list[str]) -> list:
    return array(values, "period")


def _o4() -> tuple:
    return merge(time_settings(2, ["p2030", "p2035"]), commodity_node("fuel", "fuel_node", 10.0),
                 balance_node("elec", -40.0),
                 unit("G2", ["fuel_node"], ["elec"], _pw(
                     curve_per_period({"p2030": [(0.5, 0.40), (1.0, 0.50)], "p2035": [(0.5, 0.55), (1.0, 0.50)]}),
                     existing=scalar(100.0), startup_cost=scalar(0.0))))


def _o5() -> tuple:
    ents, vals = merge(time_settings(2, ["p2030"]), commodity_node("gas", "gas_node", 20.0),
                       balance_node("elec", {"t01": -50.0, "t02": -70.0}),
                       balance_node("heat", {"t01": -30.0, "t02": -25.0}),
                       unit("CHP", ["gas_node"], ["elec", "heat"], _pw(
                           curve([(0.3, 0.25), (0.6, 0.35), (1.0, 0.40)]), virtual_unitsize=scalar(100.0),
                           existing=scalar(1.0), startup_method=text("binary"), startup_cost=scalar(0.0),
                           min_load=scalar(0.3))))
    vals += [["unit__inputNode", ["CHP", "gas_node"], "conversion_flow_coeff", scalar(1.0)],
             ["unit__outputNode", ["CHP", "elec"], "conversion_flow_coeff", scalar(1.0)],
             ["unit__outputNode", ["CHP", "elec"], "capacity_max_coeff", scalar(1.0)],
             ["unit__outputNode", ["CHP", "heat"], "conversion_flow_coeff", scalar(0.2)],
             ["unit__outputNode", ["CHP", "heat"], "capacity_max_coeff", scalar(1.0)]]
    return ents, vals


CASES["O4"] = _o4
CASES["O5"] = _o5


def _case(tmp_path_factory, name: str):
    return _solve(tmp_path_factory, name.replace("-", "_"), CASES[name]())


# ---------------------------------------------------------------------------
# O1 — thermal, integer online (§2)


def test_O1_integer_online(tmp_path_factory) -> None:
    s = _case(tmp_path_factory, "O1")
    _close(_var(s, "v_online_integer", "G"), [1, 2, 2], what="units online")
    _close(_flow(s, "G", "fuel_node", "G"), [Fr(1450, 9), Fr(3025, 9), Fr(10000, 21)], what="fuel")
    _close(_flow(s, "G", "G", "elec"), [60, 130, 200], what="output")
    _obj(s, Fr(3581380000, 63))
    # The lines (§2): seg 1 α 25/18 β 7/9, seg 2 α 265/126 β 5/18; no origin line (the efficiency rises).
    fd = s.flex_data
    lines = {seg: (a, b) for seg, a, b in utf8(fd.p_piecewise_slope.frame).join(
        utf8(fd.p_piecewise_intercept.frame), on=["p", "seg", "d"]).select("seg", "value", "value_right").rows()}
    assert set(lines) == {"1", "2"}
    _close(list(lines["1"]) + list(lines["2"]), [Fr(25, 18), Fr(7, 9), Fr(265, 126), Fr(5, 18)], what="lines")
    # one conversion_piecewise row per line and step; no conversion_indirect row of G
    pb = build(fd)
    assert cstr_over(pb, "conversion_piecewise").height == 2 * 3
    ci = cstr_over(pb, "conversion_indirect")
    assert ci is None or ci.filter(pl.col("p") == "G").height == 0


def test_O1_L_linear_online_sticky_commitment(tmp_path_factory) -> None:
    s = _case(tmp_path_factory, "O1-L")
    _close(_var(s, "v_online_linear", "G"), [2, Fr(6, 7), 2], what="units online")
    _close(_flow(s, "G", "fuel_node", "G"), [Fr(10000, 21), 150, Fr(10000, 21)], what="fuel")
    _close(_var(s, "v_startup_linear", "G"), [0, 0, Fr(8, 7)], what="start-ups")
    _obj(s, Fr(1422040000, 21))


def test_O1_R_reduce_online_full_load_efficiency(tmp_path_factory) -> None:
    """No online variables: G runs as a constant_efficiency unit at its full-load efficiency 0.42 (slope 50/21 in
    conversion_indirect, no curve lines, no conversion_piecewise row)."""
    s = _case(tmp_path_factory, "O1-R")
    fd = s.flex_data
    assert fd.process_online is None or "G" not in set(utf8(fd.process_online)["p"].to_list())
    _close(_flow(s, "G", "fuel_node", "G"), [Fr(10000, 21), Fr(1000, 7), Fr(10000, 21)], what="fuel")
    _obj(s, Fr(1343200000, 21))
    assert fd.process_piecewise is None and fd.p_piecewise_slope is None
    _close(sorted(set(utf8(fd.p_slope.frame).filter(pl.col("p") == "G")["value"].to_list())), [Fr(50, 21)],
           what="full-load slope")
    pb = build(fd)
    assert cstr_over(pb, "conversion_piecewise") is None
    assert cstr_over(pb, "conversion_indirect").filter(pl.col("p") == "G").height == 3


# ---------------------------------------------------------------------------
# O2 — electrolyser (§5-§8)


def test_O2a_no_online_capacity_form(tmp_path_factory) -> None:
    s = _case(tmp_path_factory, "O2a")
    _close(_flow(s, "E", "power_node", "E"), [Fr(3400, 77)] * 3, what="power input")
    _close(_flow(s, "E", "E", "h2"), [30] * 3, what="h2 output")
    _obj(s, Fr(1489200000, 77))
    lines = (utf8(s.flex_data.p_piecewise_slope.frame).join(utf8(s.flex_data.p_piecewise_intercept.frame),
                                                            on=["p", "seg", "d"]).sort("seg"))
    assert lines["seg"].to_list() == ["0", "1", "2"]
    _close(lines["value"].to_list() + lines["value_right"].to_list(),
           [Fr(10, 7), Fr(120, 77), Fr(125, 66), 0, Fr(-2, 77), Fr(-5, 22)], what="lines")


def test_O2b_investment_at_the_breakpoint(tmp_path_factory) -> None:
    s = _case(tmp_path_factory, "O2b")
    us = float(utf8(s.flex_data.p_unitsize.frame).filter(pl.col("p") == "E")["value"][0])
    inv = utf8(s.solution.value("v_invest_p")).filter(pl.col("p") == "E")
    _close([float(inv["value"].sum()) * us], [50], what="invested MW")
    _close(_flow(s, "E", "power_node", "E"), [Fr(500, 11)] * 3, what="power input")
    _obj(s, Fr(2375088783, 110))


def test_O2c_integer_online_origin_line(tmp_path_factory) -> None:
    s = _case(tmp_path_factory, "O2c")
    on = _var(s, "v_online_integer", "E")
    _close([on[0], on[2]], [2, 2], what="units online t01, t03")      # t02 is a tie (oracle §7)
    _close(_flow(s, "E", "power_node", "E"), [Fr(3400, 77), Fr(50, 7), Fr(5800, 77)], what="power input")
    _obj(s, Fr(1423500000, 77))


def test_O2d_online_pruned_by_integers(tmp_path_factory) -> None:
    s = _case(tmp_path_factory, "O2d")
    _close(_flow(s, "E", "power_node", "E"), [Fr(3400, 77)] * 3, what="power input")
    _obj(s, Fr(1489200000, 77))
    pb = build(s.flex_data)
    names = pb.canonicalise().col_names
    assert not [n for n in names if n.startswith("v_online_") and "[E," in n]
    assert s.flex_data.process_online is None or "E" not in set(utf8(s.flex_data.process_online)["p"].to_list())


# ---------------------------------------------------------------------------
# O3 — disposal (§9, §10)


def test_O3_disposal_burns_above_the_curve(tmp_path_factory) -> None:
    """The input surplus of power_bal goes through E (cost 0): 60 MW in, 30 MW out, 1220/77 MWh above the curve's
    need 3400/77 in every step; all slacks 0."""
    s = _case(tmp_path_factory, "O3")
    q = _flow(s, "E", "power_bal", "E")
    x = _flow(s, "E", "E", "h2")
    _close(q, [60] * 3, what="power input")
    _close(x, [30] * 3, what="h2 output")
    _obj(s, 0)
    fd = s.flex_data
    lines = utf8(fd.p_piecewise_slope.frame).join(utf8(fd.p_piecewise_intercept.frame), on=["p", "seg", "d"])
    need = [max(a * xx + b * 100.0 for a, b in lines.select("value", "value_right").rows()) for xx in x]
    _close(need, [Fr(3400, 77)] * 3, what="curve need")
    _close([qq - nn for qq, nn in zip(q, need)], [Fr(1220, 77)] * 3, what="excess")
    for name in ("vq_state_up", "vq_state_down"):
        for n in ("power_bal", "h2"):
            _close(_slack(s, name, n), [0] * 3, what=f"{name} {n}")


def test_O3_c_flat_curve_is_an_equality(tmp_path_factory) -> None:
    """A flat two-point curve is a constant-efficiency direct unit: x = 0.66·q, q = 60, the surplus at h2."""
    s = _case(tmp_path_factory, "O3-c")
    fd = s.flex_data
    assert fd.process_piecewise is None or fd.process_piecewise.height == 0
    _close(_flow(s, "E", "power_bal", "h2"), [39.6] * 3, what="h2 output (direct flow)")
    _obj(s, 84096000)


# ---------------------------------------------------------------------------
# O4 — curve by period (§11); O5 — two outputs (§12)


def test_O4_curve_by_period(tmp_path_factory) -> None:
    s = _case(tmp_path_factory, "O4")
    q = _flow(s, "G2", "fuel_node", "G2")
    _close(q, [80, 80, Fr(800, 11), Fr(800, 11)], what="fuel")
    us = float(utf8(s.flex_data.p_unitsize.frame).filter(pl.col("p") == "G2")["value"][0])
    on = [v * us for v in _var(s, "v_online_linear", "G2")]
    _close(on[:2], [40, 40], what="online MW p2030")                    # p2035 is a tie (oracle §11)
    _obj(s, Fr(147168000, 11))


def test_O5_two_outputs_weighted_load_share(tmp_path_factory) -> None:
    s = _case(tmp_path_factory, "O5")
    _close(_var(s, "v_online_integer", "CHP"), [1, 1], what="units online")
    _close(_flow(s, "CHP", "gas_node", "CHP"), [Fr(1152, 7), Fr(5625, 28)], what="gas")
    _obj(s, Fr(224102700, 7))


# ---------------------------------------------------------------------------
# T6 — a one-input unit keeps the output meaning of its input-side items (design §3.4)

T6_INFLOW = {"t01": -60.0, "t02": -130.0, "t03": -200.0, "t04": -180.0, "t05": -90.0, "t06": -50.0}
T6_PROFILE = {"t01": 0.9, "t02": 0.8, "t03": 1.0, "t04": 0.7, "t05": 0.6, "t06": 0.95}


def _t6(side: str, rolling: bool) -> tuple:
    """G (fuel -> elec, the O1 curve, linear start-ups) with a ramp limit (both directions) and a ramp cost, a
    profile upper limit, a constraint_flow_coeff in ``cap_c`` (G ≤ 150 MW) and an input share / output cap:

    * ``side = "input"``: on unit__inputNode (G, fuel_node) — ramp speeds 0.01 / 0.01, ramp cost 2, profile, coefficient
      1, input_share_max 0.9 — plus on the output a ramp limit up 0.02 (no down limit), ramp cost 1, coefficient 0.5
      and capacity_max_coeff 0.95;
    * ``side = "output"``: all of it as authored on unit__outputNode (G, elec): speeds min(0.01, 0.02) = 0.01 up and
      0.01 down, ramp cost 2 + 1 = 3, the profile, coefficient 1 + 0.5 = 1.5, capacity_max_coeff min(0.95, 0.9) =
      0.9."""
    ents_t, vals_t = time_settings(6, ["p2030"], rolling=(4.0, 2.0) if rolling else None)
    cap = ([["constraint", "cap_c", None], ["profile", "prof", None]],
           [["constraint", "cap_c", "sense", text("less_than")], ["constraint", "cap_c", "constant", scalar(150.0)],
            ["profile", "prof", "profile", str_map(T6_PROFILE, "time")]])
    g = unit("G", ["fuel_node"], ["elec"], _pw(O1_CURVE, virtual_unitsize=scalar(100.0), existing=scalar(2.0),
                                               startup_method=text("linear"), startup_cost=scalar(1.0),
                                               min_load=scalar(0.4)))
    inp, out = ["unit__inputNode", ["G", "fuel_node"]], ["unit__outputNode", ["G", "elec"]]

    def coeff(v: float) -> list:
        return str_map({"cap_c": v}, "constraint")

    if side == "input":
        node = "fuel_node"
        vals = [[*inp, "ramp_method", text("both")], [*inp, "ramp_speed_up", scalar(0.01)],
                [*inp, "ramp_speed_down", scalar(0.01)], [*inp, "ramp_cost", scalar(2.0)],
                [*inp, "constraint_flow_coeff", coeff(1.0)], [*inp, "input_share_max", scalar(0.9)],
                [*out, "ramp_method", text("both")], [*out, "ramp_speed_up", scalar(0.02)],
                [*out, "ramp_cost", scalar(1.0)], [*out, "constraint_flow_coeff", coeff(0.5)],
                [*out, "capacity_max_coeff", scalar(0.95)]]
    else:
        node = "elec"
        vals = [[*out, "ramp_method", text("both")], [*out, "ramp_speed_up", scalar(0.01)],
                [*out, "ramp_speed_down", scalar(0.01)], [*out, "ramp_cost", scalar(3.0)],
                [*out, "constraint_flow_coeff", coeff(1.5)], [*out, "capacity_max_coeff", scalar(0.9)]]
    prof = ([["unit__node__profile", ["G", node, "prof"], None]],
            [["unit__node__profile", ["G", node, "prof"], "profile_method", text("upper_limit")]])
    return merge((ents_t, vals_t), commodity_node("fuel", "fuel_node", 10.0), balance_node("elec", T6_INFLOW),
                 cap, g, prof, ([], vals))


def _lp(pb) -> dict:
    """The built LP keyed by names: rows (bounds and coefficients) and columns (objective and bounds)."""
    m = pb.canonicalise()
    rows: dict = {nm: [float(m.row_lb[i]), float(m.row_ub[i]), {}] for i, nm in enumerate(m.row_names)}
    cols: dict = {}
    for c in range(m.n_cols):
        name = m.col_names[c]
        cols[name] = (float(m.col_obj[c]), float(m.col_lb[c]), float(m.col_ub[c]))
        for k in range(m.col_ptr[c], m.col_ptr[c + 1]):
            rows[m.row_names[m.row_idx[k]]][2][name] = float(m.val[k])
    return {"rows": rows, "cols": cols}


def _t6_steps(tmp_path_factory, side: str, rolling: bool) -> dict:
    from flextool.engine import run_chain_from_db

    name = f"t6_{side}_{'roll' if rolling else 'single'}"
    if name not in _RUNS:
        ents, vals = _t6(side, rolling)
        url, scen = make_oracle_db(tmp_path_factory, name, ents, vals)
        _RUNS[name] = run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"{name}_w"),
                                        keep_solutions=True)
    return _RUNS[name]


@pytest.mark.parametrize("rolling", [False, True], ids=["single", "rolling"])
def test_T6_input_side_items_act_on_the_output(tmp_path_factory, tmp_path, rolling) -> None:
    """The one-input multi-line unit with its ramp limit, ramp cost, profile, constraint_flow_coeff and
    input_share_max on the input gives the LP of the same unit with them authored on the output (every row,
    column, bound and coefficient; the written MPS byte for byte), in a single and in a rolling solve."""
    steps = _t6_steps(tmp_path_factory, "input", rolling)
    ref = _t6_steps(tmp_path_factory, "output", rolling)
    assert list(steps) == list(ref)
    if rolling:
        assert len([k for k in steps if "_roll_" in k]) >= 2, list(steps)
    for name in ref:
        fd, fd_ref = steps[name].flex_data, ref[name].flex_data
        assert utf8(fd.process_piecewise_remap)["p"].to_list() == ["G"]
        pb, pb_ref = build(fd), build(fd_ref)
        lp, lp_ref = _lp(pb), _lp(pb_ref)
        assert lp["rows"].keys() == lp_ref["rows"].keys(), \
            sorted(set(lp["rows"]) ^ set(lp_ref["rows"]))[:10]
        diff = [k for k in lp_ref["rows"] if lp["rows"][k] != lp_ref["rows"][k]]
        assert not diff, f"{name}: rows differ, e.g. {diff[:3]}: {[lp['rows'][k] for k in diff[:1]]} vs " \
                         f"{[lp_ref['rows'][k] for k in diff[:1]]}"
        assert lp["cols"] == lp_ref["cols"], name
        safe = name.replace("/", "_")
        pb.write_mps(tmp_path / f"{safe}_a.mps")
        pb_ref.write_mps(tmp_path / f"{safe}_b.mps")
        assert (tmp_path / f"{safe}_a.mps").read_bytes() == (tmp_path / f"{safe}_b.mps").read_bytes(), name
        assert float(steps[name].solution.obj) == float(ref[name].solution.obj), name
    # The rows the remap moved are there (the identity is not vacuous).
    pb = build(list(steps.values())[0].flex_data)
    for fam in ("ramp_sink_up_constraint", "ramp_sink_down_constraint", "ramp_change_eq", "profile_flow_upper_limit",
                "conversion_piecewise"):
        over = cstr_over(pb, fam)
        assert over is not None and over.height > 0, fam
    assert cstr_over(pb, "ramp_source_up_constraint") is None


def test_T6_warm_rolling_equals_cold(tmp_path_factory) -> None:
    """The remapped multi-line unit in a rolling solve under FLEXTOOL_WARM_START (``warm=True``): every roll has the
    objective and the written outputs of the cold run (the warm path compares process_piecewise and the curve
    fields like every other field before it reuses an LP)."""
    from flextool.engine import run_chain_from_db

    from tests.engine._unit_db import assert_outputs_equal

    ents, vals = _t6("input", True)
    url, scen = make_oracle_db(tmp_path_factory, "t6_warm", ents, vals)
    runs, wfs = {}, {}
    for warm in (False, True):
        wfs[warm] = tmp_path_factory.mktemp(f"t6_warm_{int(warm)}")
        runs[warm] = run_chain_from_db(url, scen, warm=warm, keep_solutions=True, work_folder=wfs[warm])
    cold, hot = runs[False], runs[True]
    rolls = [k for k in cold if "_roll_" in k]
    assert len(rolls) >= 2 and list(cold) == list(hot), list(cold)
    for k in rolls:
        c, w = float(cold[k].solution.obj), float(hot[k].solution.obj)
        assert abs(c - w) <= 1e-9 * max(1.0, abs(c)), f"{k}: warm={w}, cold={c}"
        assert_outputs_equal(wfs[True], wfs[False], k)


def _storage_reserve_case(e_params: dict, extra_solve: list | None = None) -> tuple:
    """E (the O2a curve, 100 MW) draws from the storage node ``st`` (500 MWh) and holds upward reserve ``rs`` at elec
    with reserve_duration 2 h."""
    grp, key = ["rs", "up", "elecg"], ["rs", "up", "E", "elec"]
    st = ([["node", "st", None]],
          [["node", "st", "node_type", text("storage")], ["node", "st", "existing", scalar(500.0)],
           ["node", "st", "storage_state_start", scalar(0.5)],
           ["node", "st", "storage_start_end_method", text("fix_start")]])
    res = ([["group", "elecg", None], ["group__node", ["elecg", "elec"], None], ["reserve", "rs", None],
            ["upDown", "up", None], ["reserve__upDown__group", grp, None], ["reserve__upDown__unit__node", key, None]],
           [["reserve__upDown__group", grp, "reserve_method", text("timeseries_only")],
            ["reserve__upDown__group", grp, "reservation", scalar(20.0)],
            ["reserve__upDown__group", grp, "penalty_reserve", scalar(1000.0)],
            ["reserve__upDown__group", grp, "reserve_duration", scalar(2.0)],
            ["reserve__upDown__unit__node", key, "is_enabled", text("yes")],
            ["reserve__upDown__unit__node", key, "max_share", scalar(1.0)],
            ["reserve__upDown__unit__node", key, "reliability", scalar(1.0)]])
    return merge(time_settings(3, ["p2030"], extra_solve=extra_solve), st, balance_node("elec", -30.0), res,
                 unit("E", ["st"], ["elec"], _pw(E_CURVE, existing=scalar(100.0), **e_params)))


def _reserve_energy_coefs(fd) -> list[float]:
    """The v_reserve coefficients of the reserve_storage_up_floor rows, one per step."""
    m = build(fd).canonicalise()
    coef = {}
    for c in range(m.n_cols):
        for k in range(m.col_ptr[c], m.col_ptr[c + 1]):
            row = m.row_names[m.row_idx[k]]
            if row.startswith("reserve_storage_up_floor[") and m.col_names[c].startswith("v_reserve["):
                coef[row] = float(m.val[k])
    return list(coef.values())


def test_T6_storage_backed_reserve_of_a_remapped_unit(tmp_path_factory) -> None:
    """E (no online variables) as the direct unit it would be without its curve is a storage-backed provider:
    reserve_storage_up_floor of st has its reserve with the energy drawn per MW of reserve at the largest slope of
    its lines, seg 2 α = 125/66 (oracle §5): coefficient = −unitsize · duration · α = −100 · 2 · 125/66."""
    s = _solve(tmp_path_factory, "t6_storage_reserve", _storage_reserve_case({}))
    assert utf8(s.flex_data.process_piecewise_remap)["p"].to_list() == ["E"]
    _close(_reserve_energy_coefs(s.flex_data), [-100 * 2 * Fr(125, 66)] * 3, what="reserve energy coefficient")


def test_T6_storage_backed_reserve_of_a_reduced_remapped_unit(tmp_path_factory) -> None:
    """E with linear start-ups under reduce_unit_detail_method online runs at its full-load efficiency 0.60 and
    leaves its curve lines, but stays paired as a storage-backed provider: the energy drawn per MW of reserve is its
    p_slope 1/η(1) = 5/3, coefficient = −100 · 2 · 5/3 = −1000/3."""
    s = _solve(tmp_path_factory, "t6_storage_reserve_reduced", _storage_reserve_case(
        {"startup_method": text("linear"), "startup_cost": scalar(0.0)},
        extra_solve=[["solve", "s1", "reduce_unit_detail_method", text("online")]]))
    fd = s.flex_data
    assert fd.process_piecewise is None and fd.p_piecewise_slope is None
    assert fd.process_online is None
    assert utf8(fd.process_piecewise_remap)["p"].to_list() == ["E"]
    _close(_reserve_energy_coefs(fd), [Fr(-1000, 3)] * 3, what="reserve energy coefficient")
