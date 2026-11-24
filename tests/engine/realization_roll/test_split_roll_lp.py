"""A split realization roll's LP, end to end (design rev 7.1 §2.2 S2/S3, §8.1 T-L1 / T-cw; design note G3 wraps).

Each test forces the live plan through the seam and suppresses the realization solve the spawn still adds
(``test_trunk_sites._live_plan``, owner decision D6), so every roll of a ``within_solve`` rolling solve is a
realization roll R_k: the label ``p1`` dispatches the jump on the realized branch, the forecasts ``p1_<f>`` the
rest of the horizon from the jump end on.  The period-cyclic placeholders at the first steps of ``p1`` and of
each ``p1_<f>`` are null (``_step_relations.null_split_placeholders``), so the build passes G1-G4 and solves.
These models have no cyclic storage, no blocks and no delays: a split roll with one of those is still refused by
G3 (the per-path rows are the next package's).

Every expected value is derived by hand in the test's docstring.  The objective of a roll is in EUR: the operating
cost of its steps (branch-weighted) times 8760 h / the period's timeset hours (``years_represented`` 1).
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine import _solve_copy_models as scm
from tests.engine.realization_roll import test_trunk_sites as ts

pack, pmap, nested = scm.pack, scm.pmap, scm.nested


def _fan(rows) -> list:
    """``stochastic_branches`` of p1: *rows* ``(branch, {analysis time: weight}, realized "yes" / "no")``."""
    return pack(pmap([("p1", nested([(b, nested([(at, nested([(yes, w)])) for at, w in times.items()]))
                                     for b, times, yes in rows]))]), "map")


def _series(branches: dict) -> list:
    """A branch map ``{branch: {analysis time: {step: value}}}``."""
    return pack(pmap([(b, nested([(at, nested(list(vals.items()))) for at, vals in made.items()]))
                      for b, made in branches.items()]), "map")


def _model(steps, *, jump: int, horizon: int, fan: list, entities: list, active: list, values: list) -> dict:
    """One period ``p1`` on 1 h *steps*, a rolling ``within_solve`` solve ``st`` with *fan*; group ``sg`` with
    ``include_stochastics``; *entities*, *active* ``(class, name)`` and parameter *values* of the model."""
    from flextool.update_flextool.fixture_schema import schema_sections

    spec = schema_sections("packed")
    spec["alternatives"] = [["init", ""]]
    spec["scenarios"] = [["base", False, ""]]
    spec["scenario_alternatives"] = [["base", "init", None]]
    spec["entities"] = [["model", "flextool", None], ["timeline", "tl", None], ["timeset", "ts", None],
                        ["solve", "st", None], ["group", "sg", None], *entities]
    spec["entity_alternatives"] = [[c, [n], "init", True] for c, n in active]
    solve = [("solve_mode", pack("rolling_window", "str")),
             ("period_timeset", pack(pmap([("p1", "ts")], index_name="period"), "map")),
             ("realized_periods", scm.array(["p1"])),
             ("years_represented", pack(pmap([("p1", 1.0)], index_name="period"), "map")),
             ("rolling_solve_jump", pack(float(jump), "float")),
             ("rolling_solve_horizon", pack(float(horizon), "float")),
             ("dispatch_realization_method", pack("within_solve", "str")),
             ("stochastic_branches", fan)]
    spec["parameter_values"] = [
        ["model", "flextool", "solves", scm.array(["st"]), "init"],
        ["timeline", "tl", "timestep_duration", pack(pmap([(t, 1.0) for t in steps], index_name="time"), "map"),
         "init"],
        ["timeset", "ts", "timeline", pack("tl", "str"), "init"],
        ["timeset", "ts", "timeset_duration",
         pack(pmap([(steps[0], float(len(steps)))], index_name="time"), "map"), "init"],
        ["group", "sg", "include_stochastics", pack("yes", "str"), "init"],
        *(["solve", "st", name, value, "init"] for name, value in solve),
        *values]
    return spec


def _unit(name: str, node: str, capacity: float, cost) -> list:
    cost = pack(cost, "float") if isinstance(cost, float) else pack(pmap(list(cost.items()), index_name="time"),
                                                                    "map")
    return [["unit", name, "efficiency", pack(1.0, "float"), "init"],
            ["unit", name, "existing", pack(capacity, "float"), "init"],
            ["unit__outputNode", [name, node], "other_operational_cost", cost, "init"]]


def _node(name: str, node_type: str, inflow) -> list:
    return [["node", name, "node_type", pack(node_type, "str"), "init"],
            ["node", name, "inflow", inflow, "init"],
            ["node", name, "penalty_up", pack(10000.0, "float"), "init"],
            ["node", name, "penalty_down", pack(10000.0, "float"), "init"]]


def _run(spec: dict, folder, monkeypatch):
    """Run *spec* with the live plan: ``(builds, steps, flex data of each build)``."""
    from flextool.engine.model import core

    ts._live_plan(monkeypatch)
    datas = []
    build = core.build_flextool

    def recording(m, d, *args, **kwargs):
        datas.append(d)
        return build(m, d, *args, **kwargs)

    monkeypatch.setattr(core, "build_flextool", recording)
    built, steps = scm.run_spec(spec, folder)
    return built, steps, datas


def _objectives(steps) -> list[float]:
    return [s.solution.obj for s in steps.values()]


def _value(frame: pl.DataFrame, **keys) -> float:
    rows = frame.filter(*(pl.col(k).cast(pl.Utf8) == v for k, v in keys.items()))
    assert rows.height == 1, (keys, rows)
    return rows["value"][0]


# ---------------------------------------------------------------------------
# The placeholders (design note G3 wraps): test_trunk_sites' model, with gen online (linear, no start cost).
# ---------------------------------------------------------------------------

def _trunk_online() -> dict:
    spec = ts._spec(real_at=ts.OVERLAP, forecast_at=ts.OVERLAP)
    spec["parameter_values"].append(["unit", "gen", "startup_method", pack("linear", "str"), "init"])
    return spec


def test_split_roll_nulls_the_placeholders_and_solves(tmp_path, monkeypatch):
    """Jump 2, horizon 4 on t01..t06 (test_trunk_sites' model): R_0 = p1 t01-t02 + p1_high / p1_low t03-t04, R_1 =
    p1 t03-t04 + tails t05-t06, R_2 = p1 t05-t06 (horizon cut at the timeline end: no tail, deterministic).

    Placeholders (``make_step_jump``: a period's first step has t_previous = t_previous_within_timeset = its own last
    step): null at (p1, t01), (p1_high, t03), (p1_low, t03) in R_0; the seam link (p1, t02) of each tail stays.
    ``dt_online_wrap`` has one row, the label's first step (a free start: one stretch, no rp), its predecessor
    null; so no ``online_change`` row at (p1, t01), and the tail's first ``online_change`` row reads the label's
    last jump step.

    Objectives (demand 100 MW, gen 10 EUR/MWh, wind 100 MW x CF; factor 8760 / 6 = 1460):
    * R_0: jump on the series made at t01 (0.3, 0.5): gen 70 + 50 -> 1200; tails on the forecasts made at t03:
      high 0.25 x 10 x (10 + 10) = 50, low 0.75 x 10 x (95 + 95) = 1425; 2675 x 1460 = 3 905 500.
    * R_1: jump on the series made at t03 (0.22, 0.62): gen 78 + 38 -> 1160; tails (made at t05) 50 + 1425;
      2635 x 1460 = 3 847 100.
    * R_2: the series made at t05 (0.44, 0.14): gen 56 + 86 -> 1420 x 1460 = 2 073 200.
    """
    built, steps, datas = _run(_trunk_online(), tmp_path, monkeypatch)
    assert len(built) == 3
    d = datas[0]
    assert ts._rows(d.dtttdt) == [
        ("p1", "t01", "None", "None", "p1_low", "t04"), ("p1", "t02", "t01", "t01", "p1", "t01"),
        ("p1_high", "t03", "None", "None", "p1", "t02"), ("p1_high", "t04", "t03", "t03", "p1_high", "t03"),
        ("p1_low", "t03", "None", "None", "p1", "t02"), ("p1_low", "t04", "t03", "t03", "p1_low", "t03")]
    assert ts._rows(d.dt_online_wrap, "d", "t", "d_previous", "t_previous") == [("p1", "t01", "None", "None")]
    lp = scm.lp_snapshot().parse_mps(built[0][1])
    online_rows = sorted(r for r in lp.rows if r.startswith("online_change_linear["))
    assert online_rows == [f"online_change_linear[gen,{d},{t}]" for d, t in (
        ("p1", "t02"), ("p1_high", "t03"), ("p1_high", "t04"), ("p1_low", "t03"), ("p1_low", "t04"))]
    for f in ("p1_high", "p1_low"):
        assert lp.coefs.get((f"online_change_linear[gen,{f},t03]", "v_online_linear[gen,p1,t02]"), 0.0) != 0.0
    assert _objectives(steps) == pytest.approx([3_905_500.0, 3_847_100.0, 2_073_200.0], rel=1e-9)


def test_g3_refuses_the_placeholders_left_in(tmp_path, monkeypatch):
    """With the null left out, G3 refuses the label's first step again: its t_previous is its own last jump step."""
    from flextool.engine.preprocess import _step_relations

    monkeypatch.setattr(_step_relations, "null_split_placeholders", lambda frame, firsts, columns: frame)
    with pytest.raises(RuntimeError, match=r"Internal error: dtttdt on period 'p1' of roll '\w+' is not "
                                           r"path-aware"):
        _run(_trunk_online(), tmp_path, monkeypatch)


# ---------------------------------------------------------------------------
# T-L1: the forecasts continue from the label's last jump step (T-w1's model).
# ---------------------------------------------------------------------------

def _tw1(w_b: float, *, storage: list | None = None, delay: bool = False) -> dict:
    """T-w1: node ``elec`` (demand 1 MW) with 1 MWh of lossless storage (fix_start 0, free end, the default
    bind_forward_only; in the stochastic group), t01-t02, jump 1, horizon 2.  ``gas`` 2 MW at 10 EUR/MWh in t01 and
    30 in t02, ``peak`` 10 MW at 30, ``wind`` 1 MW on profile ``wp``: realized ``R`` 0 (made at t01 and t02),
    forecasts made at t02 ``A`` 1 (weight 1 - *w_b*) and ``B`` 0 (weight *w_b*).  *storage*: extra node values;
    *delay*: ``peak`` takes its input from node ``fuel`` with a 1 h delay."""
    steps = ("t01", "t02")
    entities = [["node", "elec", None], ["unit", "gas", None], ["unit", "peak", None], ["unit", "wind", None],
                ["profile", "wp", None], ["unit__outputNode", ["gas", "elec"], None],
                ["unit__outputNode", ["peak", "elec"], None], ["unit__outputNode", ["wind", "elec"], None],
                ["group__unit", ["sg", "wind"], None], ["group__node", ["sg", "elec"], None],
                ["unit__node__profile", ["wind", "elec", "wp"], None]]
    active = [("node", "elec"), ("unit", "gas"), ("unit", "peak"), ("unit", "wind")]
    values = [*_node("elec", "storage", pack(-1.0, "float")),
              ["node", "elec", "existing", pack(1.0, "float"), "init"],
              ["node", "elec", "storage_start_end_method", pack("fix_start", "str"), "init"],
              ["node", "elec", "storage_state_start", pack(0.0, "float"), "init"],
              *_unit("gas", "elec", 2.0, {"t01": 10.0, "t02": 30.0}), *_unit("peak", "elec", 10.0, 30.0),
              *_unit("wind", "elec", 1.0, 0.0),
              ["unit__node__profile", ["wind", "elec", "wp"], "profile_method", pack("upper_limit", "str"), "init"],
              ["profile", "wp", "profile", _series({"A": {"t02": {"t02": 1.0}}, "B": {"t02": {"t02": 0.0}},
                                                    "R": {"t01": {"t01": 0.0, "t02": 0.0},
                                                          "t02": {"t02": 0.0}}}), "init"],
              *(storage or [])]
    if delay:
        entities += [["node", "fuel", None], ["unit__inputNode", ["peak", "fuel"], None]]
        active.append(("node", "fuel"))
        values += [*_node("fuel", "balance", pack(10.0, "float")),
                   ["unit", "peak", "delay", pack(1.0, "float"), "init"]]
    fan = _fan([("A", {"t02": 1.0 - w_b}, "no"), ("B", {"t02": w_b}, "no"), ("R", {"t01": 0.0, "t02": 0.0}, "yes")])
    return _model(steps, jump=1, horizon=2, fan=fan, entities=entities, active=active, values=values)


@pytest.mark.parametrize(("w_b", "state", "objectives"), [
    (0.5, 1.0, [87_600.0, 0.0]),
    (0.25, 0.0, [76_650.0, 131_400.0])])
def test_forecasts_continue_from_the_last_jump_step(tmp_path, monkeypatch, w_b, state, objectives):
    """T-L1 on T-w1 (2 builds: R_0 = p1 t01 + p1_A / p1_B t02; R_1 = p1 t02, deterministic).  Factor 8760 / 2 = 4380.

    The first step of each tail continues from (p1, t01), jump 1: the dtttdt seam link and the ramp link.  The
    storage level s at the end of t01 is what both tails start from.  R_0: gas covers t01 (1 MWh) and charges s at
    10; under B (wind 0) s saves 30 EUR/MWh at t02, under A nothing.  w_B = 0.5: 15 > 10, s = 1 (gas 2 MW), cost
    20 -> 87 600; R_1 starts from s = 1 and covers its 1 MWh from storage: 0.  w_B = 0.25: 7.5 < 10, s = 0, cost
    10 + 0.25 x 30 = 17.5 -> 76 650; R_1 buys 1 MWh at 30 -> 131 400.  Realized cost 20 or 10 + 30 = 40."""
    built, steps, datas = _run(_tw1(w_b), tmp_path, monkeypatch)
    assert len(built) == 2
    d = datas[0]
    links = {r[:2]: r[2:] for r in ts._rows(d.dtttdt, "d", "t", "d_previous", "t_previous_within_solve")}
    assert links[("p1_A", "t02")] == links[("p1_B", "t02")] == ("p1", "t01")
    assert {("p1_A", "t02", "p1", "t01"), ("p1_B", "t02", "p1", "t01")} <= set(
        ts._rows(d.dt_ramp_link, "d", "t", "d_previous", "t_previous"))
    first = list(steps.values())[0]
    assert _value(first.solution.value("v_state"), n="elec", d="p1", t="t01") == pytest.approx(state, abs=1e-9)
    assert _value(first.handoff.roll_end_state, node="elec") == pytest.approx(state, abs=1e-9)
    assert _objectives(steps) == pytest.approx(objectives, rel=1e-9, abs=1e-6)


def test_g4_refuses_a_tail_that_does_not_continue_from_the_jump(tmp_path, monkeypatch):
    """T-L1: ``make_step_jump`` as before S2 (a first-period fan's copy self-cycles on its own last step): G4 raises
    on the first tail."""
    from flextool.engine.solve import _run_model, _stochastic

    make = _stochastic.make_step_jump

    def before_s2(active_time_list, period__branch, sbtb):
        rows = make(active_time_list, period__branch, sbtb)
        labels = {a for a, b in period__branch if a != b}
        out = []
        for r in rows:
            if r[0] not in labels and r[4] in labels and r[4] != r[0]:
                own = active_time_list[r[0]]
                r = (*r[:4], r[0], own[-1].timestep, own[0].index - own[-1].index)
            out.append(r)
        return out

    monkeypatch.setattr(_stochastic, "make_step_jump", before_s2)
    monkeypatch.setattr(_run_model, "make_step_jump", before_s2)
    with pytest.raises(RuntimeError, match=r"Internal error: dtttdt on period 'p1_A' of roll '\w+' is not "
                                           r"path-aware"):
        _run(_tw1(0.5), tmp_path, monkeypatch)


# ---------------------------------------------------------------------------
# T-cw (uniform): the label's cells cost at weight 1.
# ---------------------------------------------------------------------------

def _tcw() -> dict:
    """T-cw: one period t01-t03 (factor 8760 / 3 = 2920), horizon 3, jump 1, ``gas`` 10 MW at 10 EUR/MWh only.
    Demand (stochastic, node ``elec`` in the group): ``R`` 1 at t01-t03 (made at t01); forecasts made at t02: ``A``
    (weight 0.25) 1 and ``B`` (0.75) 2 at t02, t03; made at t03 (read by R_1 only): ``A`` (0.5) 1, ``B`` (0.5) 1."""
    steps = ("t01", "t02", "t03")
    demand = _series({"R": {"t01": {t: -1.0 for t in steps}},
                      "A": {"t02": {"t02": -1.0, "t03": -1.0}, "t03": {"t03": -1.0}},
                      "B": {"t02": {"t02": -2.0, "t03": -2.0}, "t03": {"t03": -1.0}}})
    fan = _fan([("A", {"t02": 0.25, "t03": 0.5}, "no"), ("B", {"t02": 0.75, "t03": 0.5}, "no"),
                ("R", {"t01": 0.0}, "yes")])
    return _model(steps, jump=1, horizon=3, fan=fan,
                  entities=[["node", "elec", None], ["unit", "gas", None], ["unit__outputNode", ["gas", "elec"], None],
                            ["group__node", ["sg", "elec"], None]],
                  active=[("node", "elec"), ("unit", "gas")],
                  values=[*_node("elec", "balance", demand), *_unit("gas", "elec", 10.0, 10.0)])


def test_label_cells_cost_at_weight_one(tmp_path, monkeypatch):
    """T-cw uniform.  R_0 = p1 t01 + p1_A / p1_B t02-t03: 2920 x (1 x 10 + 0.25 x 20 + 0.75 x 40) = 131 400 (with
    the label at weight 0 it would be 102 200).  R_1 = p1 t02 + tails t03 on the forecasts made at t03:
    2920 x (10 + 0.5 x 10 + 0.5 x 10) = 58 400.  R_2 = p1 t03 (deterministic, R): 2920 x 10 = 29 200."""
    built, steps, datas = _run(_tcw(), tmp_path, monkeypatch)
    assert len(built) == 3
    assert ts._values(datas[0], "pdt_branch_weight", "d", "t") == {
        ("p1", "t01"): 1.0, ("p1_A", "t02"): 0.25, ("p1_A", "t03"): 0.25,
        ("p1_B", "t02"): 0.75, ("p1_B", "t03"): 0.75}
    assert _objectives(steps) == pytest.approx([131_400.0, 58_400.0, 29_200.0], rel=1e-9)


# ---------------------------------------------------------------------------
# Still refused: cyclic storage, blocks and delays need per-path rows (the next package's).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("storage", "delay", "frame"), [
    ([["node", "elec", "storage_binding_method", pack("bind_within_period", "str"), "init"]], False,
     "dtttdt_within_period"),
    ([["node", "elec", "storage_binding_method", pack("bind_intraperiod_blocks", "str"), "init"]], False,
     "period_block_succ"),
    (None, True, "dtt__delay_duration")], ids=["bind_within_period", "bind_intraperiod_blocks", "delay"])
def test_split_roll_with_a_cycle_or_delay_is_refused(tmp_path, monkeypatch, storage, delay, frame):
    """T-w1 with cyclic storage, intraperiod blocks or a 1 h unit delay: G3 refuses roll 0, not a different LP.  The
    within-period lag of the label's first step wraps to its own last step (t01 -> t01), the block of a tail
    succeeds itself, and the delay wraps the flow of p1's only step back into p1 (it must reach each tail)."""
    with pytest.raises(RuntimeError, match=rf"Internal error: {frame} on period '\w+' of roll '\w+' is not "
                                           r"path-aware"):
        _run(_tw1(0.5, storage=storage, delay=delay), tmp_path, monkeypatch)
