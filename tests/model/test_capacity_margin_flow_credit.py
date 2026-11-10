"""The capacity margin counts firm and VRE capacity, and the other flows of the group by their value.

    Σ_{F ∪ V} (counted capacity − flow) + vq ≥ M + Σ_{n ∈ N(g)} q_up      (the row after substituting the balance)
    F: firm unit arcs, capacity × availability × capacity_max_coeff;  V: VRE arcs, capacity × profile × availability
    storage-backed units, connections and consumption by their flow;  energy dumped with vq_state_down does not count

Vehicle (``specs/capacity_margin_design.md`` §8): ``tests.json`` chain ``init, west, coal`` (+ the named
alternatives), solve ``y2020_2day_dispatch`` (48 steps) with ``invest_periods [p2020]``, group ``cm`` with
``capacity_margin`` 100 MW and penalty 1000.  west's real demand peaks at 872 MW (t0043); ``coal_plant`` is 500 MW.
The shortfall is ``vq_capacity_margin × group_capacity_for_scaling`` [MW].  The expected values are worked by hand
in the design and its critique (a separate review), or computed here from the fixture JSON (E, E2), never read from
the model.  Databases are built from JSON (never a checked-in ``.sqlite``).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import polars as pl
import pytest

from tests.engine._unit_db import b64, make_db, utf8

REPO = Path(__file__).resolve().parents[2]
FIXTURE = json.loads((REPO / "tests" / "fixtures" / "tests.json").read_text())
STEPS = [f"t{i:04d}" for i in range(1, 49)]
BASE = ["init", "west", "coal"]


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _fixture_value(alt: str, entity: str, param: str):
    for row in FIXTURE["parameter_values"]:
        if row[4] == alt and row[1] == entity and row[2] == param:
            return json.loads(base64.b64decode(row[3][0]))
    raise KeyError((alt, entity, param))


DEMAND = {t: -v for t, v in _fixture_value("west", "west", "inflow")["data"]}
WIND = {t: v for t, v in _fixture_value("wind", "wind_profile", "profile")["data"]}


def sc(v: float) -> list:
    return b64(float(v), "float")


def st(v: str) -> list:
    return b64(v, "str")


_RUNS: dict = {}


def _shortfall(tmp_path_factory, name: str, chain: list[str], values: list, entities: list, nodes: list[str], *,
               margin: float | None = 100.0, method: str = "manual",
               penalty: float = 1000.0) -> tuple[float, object]:
    """``(shortfall MW of group cm in p2020, the solve's step)``."""
    from flextool.engine import run_chain_from_db

    if name in _RUNS:
        return _RUNS[name]
    ents = [["group", "cm", None]] + [["group__node", ["cm", n], None] for n in nodes] + entities
    vals = [["group", "cm", "capacity_margin_method", st(method)],
            ["group", "cm", "penalty_capacity_margin", sc(penalty)],
            ["solve", "y2020_2day_dispatch", "invest_periods",
             b64({"value_type": "str", "data": ["p2020"], "index_name": "period"}, "array")]]
    own = {tuple(v[:3]) if isinstance(v[1], str) else (v[0], tuple(v[1]), v[2]) for v in values}
    vals = [v for v in vals if (v[0], v[1], v[2]) not in own] + values
    if margin is not None:
        vals.append(["group", "cm", "capacity_margin", sc(margin)])
    url, scen = make_db(tmp_path_factory, alt=f"cmf_{name}", values=vals, base_chain=chain, entities=ents)
    steps = run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"cmf_{name}_w"), keep_solutions=True)
    step = list(steps.values())[-1]
    vq = utf8(step.solution.value("vq_capacity_margin"))
    gcs = utf8(step.flex_data.p_group_capacity_for_scaling.frame)
    short = vq.join(gcs.rename({"value": "gcs"}), on=["g", "d"]).filter((pl.col("g") == "cm") & (pl.col("d") == "p2020"))
    assert short.height == 1
    _RUNS[name] = (float(short["value"][0] * short["gcs"][0]), step)
    return _RUNS[name]


# ─── A, B: firm capacity, derated by availability ────────────────────────


def test_firm_capacity_A(tmp_path_factory) -> None:
    """A: coal 500 MW counts at capacity: 872 + 100 − 500 = 472."""
    assert _shortfall(tmp_path_factory, "A", BASE, [], [], ["west"])[0] == pytest.approx(472.0, abs=1e-4)


def test_availability_derates_firm_capacity_B(tmp_path_factory) -> None:
    """B: coal availability 0.5 counts 250 MW: 872 + 100 − 250 = 722 (before: 472, availability ignored)."""
    vals = [["unit", "coal_plant", "availability", sc(0.5)]]
    assert _shortfall(tmp_path_factory, "B", BASE, vals, [], ["west"])[0] == pytest.approx(722.0, abs=1e-4)


# ─── C, dual input: storage-backed units ─────────────────────────────────


BAT_ENTITIES = [["node", "bat", None], ["unit", "bat_out", None], ["unit__inputNode", ["bat_out", "bat"], None],
                ["unit__outputNode", ["bat_out", "west"], None]]
BAT_VALUES = [["node", "bat", "node_type", st("storage")], ["node", "bat", "existing", sc(100.0)],
              ["node", "bat", "storage_start_end_method", st("fix_start")],
              ["node", "bat", "storage_state_start", sc(0.0)],
              ["node", "bat", "penalty_method", st("off")],
              ["node", "bat", "storage_binding_method", st("bind_within_timeblock")],
              ["unit", "bat_out", "existing", sc(200.0)], ["unit", "bat_out", "efficiency", sc(1.0)]]


def test_empty_storage_unit_counts_by_flow_C(tmp_path_factory) -> None:
    """C: ``bat_out`` (200 MW) on an empty storage node (start 0, no charging, hard balance) delivers nothing, so it
    adds nothing: 472 (before: 272, its capacity counted)."""
    short, _ = _shortfall(tmp_path_factory, "C", BASE, BAT_VALUES, BAT_ENTITIES, ["west"])
    assert short == pytest.approx(472.0, abs=1e-4)


def test_storage_unit_with_a_full_fuel_input_is_firm(tmp_path_factory) -> None:
    """Dual input: ``bat_out`` with a coal input of ``input_share_max`` 1 can run at full output without the
    storage, so it counts at capacity: 472 − 200 = 272."""
    ents = BAT_ENTITIES + [["unit__inputNode", ["bat_out", "coal_market"], None]]
    vals = BAT_VALUES + [["unit__inputNode", ["bat_out", "coal_market"], "input_share_max", sc(1.0)]]
    short, _ = _shortfall(tmp_path_factory, "dual", BASE, vals, ents, ["west"])
    assert short == pytest.approx(272.0, abs=1e-4)


# ─── D: imports by flow; dumped energy does not count ────────────────────


def test_import_only_group_holds_no_margin_D(tmp_path_factory) -> None:
    """D: group east (demand 300 MW) is fed only by ``west_east`` (400 MW delivered); east's upward slack is made
    dear (1e5 CUR/MWh), so east does not make energy to send back to the short west.  The import covers the
    demand but cannot hold the margin: shortfall = M = 100 (before: 400, imports ignored)."""
    vals = [["node", "east", "penalty_up", sc(1e5)]]
    short, _ = _shortfall(tmp_path_factory, "D", BASE + ["network"], vals, [], ["east"])
    assert short == pytest.approx(100.0, abs=1e-4)


# ─── E, E2: VRE at capacity × profile × availability ─────────────────────


def test_vre_counts_at_its_potential_E(tmp_path_factory) -> None:
    """E: wind 1000 MW with an upper_limit profile counts 1000 · profile; margin 250:
    max_t(D_t + 250 − 500 − 1000 · profile_t) (from the fixture JSON)."""
    expected = max(DEMAND[t] + 250.0 - 500.0 - 1000.0 * WIND[t] for t in STEPS)
    short, _ = _shortfall(tmp_path_factory, "E", BASE + ["wind"], [], [], ["west"], margin=250.0)
    assert short == pytest.approx(expected, abs=1e-4)


def test_vre_derated_by_availability_E2(tmp_path_factory) -> None:
    """E2: wind availability 0.7: max_t(D_t + 250 − 500 − 700 · profile_t) (from the fixture JSON)."""
    expected = max(DEMAND[t] + 250.0 - 500.0 - 700.0 * WIND[t] for t in STEPS)
    vals = [["unit", "wind_plant", "availability", sc(0.7)]]
    short, _ = _shortfall(tmp_path_factory, "E2", BASE + ["wind"], vals, [], ["west"], margin=250.0)
    assert short == pytest.approx(expected, abs=1e-4)


# ─── F: a battery behind a connection ────────────────────────────────────


def test_battery_connection_counts_by_flow_F(tmp_path_factory) -> None:
    """F: ``battery_inverter`` (10 MW, delivered at the sink side) counts by its flow: 472 − 10 = 462 (before:
    472, connections ignored)."""
    short, _ = _shortfall(tmp_path_factory, "F", BASE + ["battery"], [], [], ["west"])
    assert short == pytest.approx(462.0, abs=1e-4)


# ─── P: share_of_peak ────────────────────────────────────────────────────


def test_share_of_peak_P(tmp_path_factory) -> None:
    """P: ``share_of_peak`` with ``capacity_margin_share`` 0.1: M = 0.1 · 872 = 87.2, so 872 + 87.2 − 500 = 459.2.
    The peak is over the period's original timeline (72 steps; its peak 872 is at t0043)."""
    vals = [["group", "cm", "capacity_margin_share", sc(0.1)]]
    short, step = _shortfall(tmp_path_factory, "P", BASE, vals, [], ["west"], margin=None, method="share_of_peak")
    assert short == pytest.approx(459.2, abs=1e-4)
    peak = utf8(step.flex_data.pdGroup_peak_demand.frame).filter(pl.col("g") == "cm")
    assert peak["value"].to_list() == [pytest.approx(max(DEMAND.values()))]


def test_share_of_peak_takes_the_peak_outside_the_solve(tmp_path_factory) -> None:
    """P with west's demand 1000 MW at t0060, a step of the timeline (72 steps) outside the solve's 48 steps: the
    peak is 1000, so M = 100, and the solve's own peak step (t0043) gives 872 + 100 − 500 = 472 (a peak over the
    solve's steps would give M = 87.2 and 459.2)."""
    data = [[t, -1000.0 if t == "t0060" else v] for t, v in _fixture_value("west", "west", "inflow")["data"]]
    inflow = b64({"index_type": "str", "rank": 1, "index_name": "time", "type": "map", "data": data}, "map")
    vals = [["group", "cm", "capacity_margin_share", sc(0.1)], ["node", "west", "inflow", inflow]]
    short, step = _shortfall(tmp_path_factory, "P_out", BASE, vals, [], ["west"], margin=None,
                             method="share_of_peak")
    peak = utf8(step.flex_data.pdGroup_peak_demand.frame).filter(pl.col("g") == "cm")
    assert peak["value"].to_list() == [pytest.approx(1000.0)]
    assert short == pytest.approx(472.0, abs=1e-4)


def test_share_of_peak_ignores_capacity_margin(tmp_path_factory) -> None:
    """A ``capacity_margin`` value on a share_of_peak group is not its margin: still 459.2."""
    vals = [["group", "cm", "capacity_margin_share", sc(0.1)]]
    short, _ = _shortfall(tmp_path_factory, "P_mw", BASE, vals, [], ["west"], margin=300.0, method="share_of_peak")
    assert short == pytest.approx(459.2, abs=1e-4)


def test_share_of_peak_constant_equals_period_map(tmp_path_factory) -> None:
    """A constant ``capacity_margin_share`` and the same value as a period map give the same LP result."""
    vals = [["group", "cm", "capacity_margin_share",
             b64({"index_type": "str", "index_name": "period", "data": [["p2020", 0.1]]}, "map")]]
    short, _ = _shortfall(tmp_path_factory, "P_map", BASE, vals, [], ["west"], margin=None, method="share_of_peak")
    assert short == pytest.approx(_shortfall(tmp_path_factory, "P", BASE, [["group", "cm", "capacity_margin_share",
                                                                             sc(0.1)]], [], ["west"], margin=None,
                                             method="share_of_peak")[0], abs=1e-9)


# ─── X, L: exports and internal losses ───────────────────────────────────


EX_ENTITIES = [["node", "ex", None], ["connection", "w_ex", None],
               ["connection__node__node", ["w_ex", "west", "ex"], None]]


def _ex_values(efficiency: float, inflow: float, transfer: str = "regular") -> list:
    return [["node", "ex", "node_type", st("balance")], ["node", "ex", "inflow", sc(inflow)],
            ["node", "ex", "penalty_method", st("off")],
            ["connection", "w_ex", "existing", sc(300.0)], ["connection", "w_ex", "efficiency", sc(efficiency)],
            ["connection", "w_ex", "transfer_method", st(transfer)]]


def test_exports_add_to_the_requirement_X(tmp_path_factory) -> None:
    """X: west exports 200 MW to ex (hard balance) through ``w_ex`` (efficiency 1): 872 + 200 + 100 − 500 = 672."""
    short, _ = _shortfall(tmp_path_factory, "X", BASE, _ex_values(1.0, -200.0), EX_ENTITIES, ["west"])
    assert short == pytest.approx(672.0, abs=1e-4)


def test_exports_on_a_one_variable_connection(tmp_path_factory) -> None:
    """X with ``w_ex`` a one-variable two-way connection (``no_losses_no_variable_cost``, reverse flow in
    ``v_flow_back``): the export still counts once, at west only: 672."""
    vals = _ex_values(1.0, -200.0, "no_losses_no_variable_cost")
    short, _ = _shortfall(tmp_path_factory, "X1var", BASE, vals, EX_ENTITIES, ["west"])
    assert short == pytest.approx(672.0, abs=1e-4)


def test_internal_lines_are_left_out_L(tmp_path_factory) -> None:
    """L: group {west, ex}; ``w_ex`` (both ends in the group) delivers 300 MW at efficiency 0.9.  The line is left
    out of the row, so its 33.33 MW loss is not part of the requirement and ex's demand counts one to one:
    872 + 300 + 100 − 500 = 772 (with the line in the row: 805.33)."""
    short, _ = _shortfall(tmp_path_factory, "L", BASE, _ex_values(0.9, -300.0), EX_ENTITIES, ["west", "ex"])
    assert short == pytest.approx(872.0 + 300.0 + 100.0 - 500.0, abs=1e-4)


CONV_ENTITIES = [["node", "ex", None], ["unit", "conv", None], ["unit__inputNode", ["conv", "west"], None],
                 ["unit__outputNode", ["conv", "ex"], None]]


@pytest.mark.parametrize("efficiency,expected", [(0.9, 872.0 + 300.0 + 100.0 - 500.0),
                                                 (3.0, 872.0 + 300.0 / 3.0 + 100.0 - 500.0)])
def test_converter_inside_the_group(tmp_path_factory, efficiency, expected) -> None:
    """Group {west, ex}; ``conv`` (400 MW) turns west's power into ex's demand of 300 MW (hard balance).

    * efficiency 0.9 (an electric boiler): a lossy converter inside the group is left out, as a line inside it is:
      872 + 300 + 100 − 500 = 772 (in the row it would be 805.33, and shedding ex would lower the requirement);
    * efficiency 3 (a heat pump): counted by its flow, output 300 − input 100, so ex's demand needs 100 MW of
      west's power: 872 + 100 + 100 − 500 = 572."""
    vals = [["node", "ex", "node_type", st("balance")], ["node", "ex", "inflow", sc(-300.0)],
            ["node", "ex", "penalty_method", st("off")],
            ["unit", "conv", "existing", sc(400.0)], ["unit", "conv", "efficiency", sc(efficiency)]]
    short, _ = _shortfall(tmp_path_factory, f"conv_{efficiency}", BASE, vals, CONV_ENTITIES, ["west", "ex"])
    assert short == pytest.approx(expected, abs=1e-4)


def test_dumped_energy_does_not_count(tmp_path_factory) -> None:
    """Group {ex} (demand 200 MW, dumping almost free: ``penalty_down`` 1, ``penalty_up`` 1e5) is fed by ``w_ex``
    (300 MW, efficiency 1) from west, which has surplus (``coal_plant`` 2000 MW).  Importing 100 MW more than ex
    needs and dumping it costs less than the margin penalty, so a row that let the dumped energy count would show
    0.  The dump (vq_state_down) is subtracted: an import-only group holds no margin, shortfall = M = 100."""
    vals = [["node", "ex", "node_type", st("balance")], ["node", "ex", "inflow", sc(-200.0)],
            ["node", "ex", "penalty_up", sc(1e5)], ["node", "ex", "penalty_down", sc(1.0)],
            ["connection", "w_ex", "existing", sc(300.0)], ["connection", "w_ex", "efficiency", sc(1.0)],
            ["connection", "w_ex", "transfer_method", st("regular")],
            ["unit", "coal_plant", "existing", sc(2000.0)]]
    short, _ = _shortfall(tmp_path_factory, "dump", BASE, vals, EX_ENTITIES, ["ex"])
    assert short == pytest.approx(100.0, abs=1e-4)


# ─── CHP into an elec + heat group ───────────────────────────────────────


def test_chp_in_elec_and_heat_group_counts_its_capacity_once(tmp_path_factory) -> None:
    """``coal_chp`` (1000 MW, outputs west and heat, west = 0.5 · heat) with ``coal_plant`` (500 MW) in group
    {west, heat} (heat demand 300 MW), margin 1500.  maxOutputSum caps the sum of the CHP's outputs at 1000 MW, so
    it counts once, at 1000 · min(1, 1 + 1) = 1000 MW: vq ≥ M + D_t − counted = 1500 + (872 + 300) − (500 + 1000)
    = 1172 at the peak (the model does not dump heat there: that would raise the shortfall).  The sum of the CHP's
    flows into the group stays within its counted capacity (headroom ≥ 0)."""
    short, step = _shortfall(tmp_path_factory, "CHP", BASE + ["heat", "coal_chp"], [], [], ["west", "heat"],
                             margin=1500.0)
    assert short == pytest.approx(1172.0, abs=1e-4)
    unitsize = utf8(step.flex_data.p_unitsize.frame).filter(pl.col("p") == "coal_chp")["value"][0]
    out = (utf8(step.solution.value("v_flow"))
           .filter((pl.col("p") == "coal_chp") & pl.col("sink").is_in(["west", "heat"])))
    assert set(out["sink"].to_list()) == {"west", "heat"}
    assert (out.group_by("d", "t").agg(pl.col("value").sum())["value"] * unitsize).max() <= 1000.0 + 1e-6


# ─── storage slack does not pose as storage delivery ─────────────────────


def test_storage_slack_energy_does_not_count(tmp_path_factory) -> None:
    """C with the storage node's balance slack on (``penalty_method`` regular, ``penalty_up`` 1e4 by default) and a
    margin penalty of 1e5 CUR/kW: a MW of margin (1e8) costs more than the slack energy behind it (1e4 · 182.5 per
    MWh), so a row that credited ``bat_out``'s flow would let the slack fill ``bat`` (shortfall 272).  The slack
    energy made at ``bat`` is taken back: 472."""
    vals = [v for v in BAT_VALUES if v[2] != "penalty_method"]
    short, step = _shortfall(tmp_path_factory, "C_slack", BASE, vals, BAT_ENTITIES, ["west"], penalty=1e5)
    assert short == pytest.approx(472.0, abs=1e-4)


# ─── share_of_peak in rolling and stochastic solves; the peak per period ──


def _share(group: str, value) -> list:
    return [["group", group, "capacity_margin_method", st("share_of_peak")],
            ["group", group, "capacity_margin_share", value]]


def _period_map(periods: list[str], value: float) -> list:
    return b64({"index_type": "str", "index_name": "period", "data": [[p, value] for p in periods]}, "map")


def _run_vehicle(tmp_path_factory, name: str, veh: str, values: list) -> dict:
    from flextool.engine import run_chain_from_db
    from tests.model import test_capacity_margin_invest_periods as cm

    if name in _RUNS:
        return _RUNS[name]
    if veh == "roll":
        fj, chain, ents = (cm._deterministic_rolling_json(tmp_path_factory),
                           cm._chain(cm.STOCHASTICS_JSON, "1_week_rolling_wind"), cm.ELEC_G)
        values = values + [["solve", "1week_rolling", "invest_periods",
                            b64({"value_type": "str", "data": ["period1"], "index_name": "period"}, "array")]]
    else:
        fj, chain, ents = cm.RECOURSE_JSON, cm._chain(cm.RECOURSE_JSON, "recourse"), cm.CM_G
    url, scen = make_db(tmp_path_factory, alt=f"cmf_{name}", values=values, base_chain=chain, fixture_json=fj,
                        entities=ents)
    _RUNS[name] = run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"cmf_{name}_w"),
                                    keep_solutions=True, warm=False)
    return _RUNS[name]


def _margins(step) -> dict[str, float]:
    m = step.flex_data.pdGroup_capacity_margin
    return {} if m is None else {d: v for d, v in utf8(m.frame).select("d", "value").iter_rows()}


@pytest.mark.parametrize("veh,group,periods", [("roll", "elec_g", ["period1"]),
                                               ("stoch", "cm_g", ["p2035", "p2040"])])
def test_share_constant_equals_period_map_in_rolling_and_stochastic_solves(tmp_path_factory, veh, group,
                                                                           periods) -> None:
    """A constant ``capacity_margin_share`` and the same value as a period map give the same margin and the same
    objective in every roll of a rolling investment solve and in a stochastic (recourse) fan.  The margin is the
    share of the peak over the period's original timeline, so every roll has the same margin, and each forecast
    branch period has a margin of its own peak."""
    const = _run_vehicle(tmp_path_factory, f"{veh}_const", veh, _share(group, sc(0.1)))
    as_map = _run_vehicle(tmp_path_factory, f"{veh}_map", veh, _share(group, _period_map(periods, 0.1)))
    assert len(const) == len(as_map)
    for a, b in zip(const.values(), as_map.values()):
        assert float(a.solution.obj) == pytest.approx(float(b.solution.obj), rel=1e-9)
        assert _margins(a) == _margins(b)
        peak = {d: v for d, v in utf8(a.flex_data.pdGroup_peak_demand.frame).select("d", "value").iter_rows()}
        assert _margins(a) == {d: pytest.approx(0.1 * peak[d]) for d in _margins(a)}
        assert all(v > 0 for v in _margins(a).values())
    if veh == "roll":
        assert len({tuple(sorted(_margins(s).items())) for s in const.values()}) == 1
    else:
        (step,) = const.values()
        assert sorted(_margins(step)) == ["p2035_high", "p2035_low", "p2040_high", "p2040_low"]


# ─── input checks ────────────────────────────────────────────────────────


def test_capacity_margin_on_a_share_of_peak_group_is_reported_ignored(tmp_path_factory, caplog) -> None:
    """``capacity_margin`` on a share_of_peak group is not its margin; the run says so."""
    import logging

    with caplog.at_level(logging.WARNING):
        _shortfall(tmp_path_factory, "P_warn", BASE, [["group", "cm", "capacity_margin_share", sc(0.1)]], [],
                   ["west"], margin=300.0, method="share_of_peak")
    assert ("Group 'cm' has capacity_margin but capacity_margin_method is not manual: the limit is ignored."
            in caplog.text)


def test_balance_within_period_node_in_a_capacity_margin_group_is_an_error(tmp_path_factory) -> None:
    """A balance_within_period node balances per period, the margin holds per time step: an error that names the
    group and the node."""
    from flextool.engine.input._solve_state import FlexToolConfigError

    ents = [["node", "bwp", None]]
    vals = [["node", "bwp", "node_type", st("balance_within_period")], ["node", "bwp", "inflow", sc(-10.0)]]
    with pytest.raises(FlexToolConfigError, match=r"Capacity margin group 'cm' contains balance_within_period "
                                                  r"nodes \(bwp\)"):
        _shortfall(tmp_path_factory, "bwp", BASE, vals, ents, ["west", "bwp"])


def test_storage_node_with_demand_in_a_capacity_margin_group_is_reported(tmp_path_factory, caplog) -> None:
    """A storage node of the group with demand has that demand left out of the margin; the run says so."""
    import logging

    vals = [v for v in BAT_VALUES if v[2] != "penalty_method"] + [["node", "bat", "inflow", sc(-10.0)]]
    with caplog.at_level(logging.WARNING):
        _shortfall(tmp_path_factory, "bat_demand", BASE, vals, BAT_ENTITIES, ["west", "bat"])
    assert "Capacity margin group 'cm': the storage node(s) bat have demand" in caplog.text


def test_share_of_peak_per_period_on_a_shared_timeline(tmp_path_factory) -> None:
    """Two investment periods take different blocks of one timeline (p2020 t0001-t0024, p2025 t0025-t0072), and
    west's demand is 1000 MW at t0060: each period's peak is over its own part of the timeline, so p2020 has the
    peak of t0001-t0024 (from the fixture JSON) and p2025 has 1000."""
    data = [[t, -1000.0 if t == "t0060" else v] for t, v in _fixture_value("west", "west", "inflow")["data"]]
    inflow = b64({"index_type": "str", "rank": 1, "index_name": "time", "type": "map", "data": data}, "map")
    periods = b64({"value_type": "str", "data": ["p2020", "p2025"], "index_name": "period"}, "array")
    ents = [["timeset", "first_day", None], ["timeset", "rest", None]]
    vals = [["group", "cm", "capacity_margin_share", sc(0.1)], ["node", "west", "inflow", inflow],
            ["timeset", "first_day", "timeline", st("y2020")], ["timeset", "rest", "timeline", st("y2020")],
            ["timeset", "first_day", "timeset_duration", b64({"index_type": "str", "index_name": "timestep",
                                                              "data": [["t0001", 24.0]]}, "map")],
            ["timeset", "rest", "timeset_duration", b64({"index_type": "str", "index_name": "timestep",
                                                         "data": [["t0025", 48.0]]}, "map")],
            ["solve", "y2020_2day_dispatch", "period_timeset",
             b64({"index_type": "str", "index_name": "period",
                  "data": [["p2020", "first_day"], ["p2025", "rest"]]}, "map")],
            ["solve", "y2020_2day_dispatch", "realized_periods", periods],
            ["solve", "y2020_2day_dispatch", "invest_periods", periods]]
    _, step = _shortfall(tmp_path_factory, "P_shared", BASE, vals, ents, ["west"], margin=None,
                         method="share_of_peak")
    peak = {d: v for g, d, v in utf8(step.flex_data.pdGroup_peak_demand.frame).iter_rows() if g == "cm"}
    first = max(DEMAND[f"t{i:04d}"] for i in range(1, 25))
    assert peak == {"p2020": pytest.approx(first), "p2025": pytest.approx(1000.0)}
