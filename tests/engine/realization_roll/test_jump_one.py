"""Jump = 1 step (design rev 7.1 §4 toy, §8.2 T-w2, §8.3 T-sd1 / T-b5 / T-b6 / T-a2): both realization methods.

Model (``_toy``): one period ``p1`` on t01-t04 (1 h steps, ``years_represented`` 1: factor 8760 / 4 = 2190 per
step), a rolling solve ``st`` with jump 1 and horizon 2: roll r = t_{r+1}-t_{r+2} (r = 0..2), roll 3 = t04 (the
horizon cut at the timeline end).  Node ``elec`` demand 10 MW.  ``wind`` 10 MW, upper-limit profile ``wp`` (in the
stochastic group): realized ``R`` 0.5 at every step (authored once, at t01); forecast ``upper`` (weight 1) made at
t_k is 0.80 + 0.01k at every step it covers (t_k..t04), ``lower`` (weight 3) 0.20 + 0.01k; made at t02, t03, t04
(``within_solve``) and also at t01 (``separate_dispatch``, whose solve (1) of roll 0 fans from t01).  ``coal``: two
integer units of 5 MW (``virtual_unitsize`` 5, ``existing`` 2), 20 EUR/MWh, ``startup_cost`` 1 EUR/MW (5 EUR per
unit start), no min load (except T-a2).

Hand rules used below (weights 1 / 3 -> 0.25 / 0.75):
* Jump step on R: wind 5, coal 5 MWh -> 100 EUR.
* A tail step on forecasts made at t_k: upper wind 8 + 0.1k, coal 2 - 0.1k MWh (1 unit); lower wind 2 + 0.1k, coal
  8 - 0.1k MWh (2 units); weighted 0.25 (40 - 2k) + 0.75 (160 - 2k) = 130 - 2k EUR.
* Units online: the lower forecast needs 2 units after the jump, so the jump keeps 2 online (1 online would cost a
  start, 0.75 x 5 EUR expected).  Roll 0 has a free start (no online_change row at its first step); every later
  roll continues from 2 units online (no start).  If the carry were lost, roll r+1 would pay 1.75 x 5 EUR more.
* Realization roll r < 3 (R_r): 100 + 130 - 2(r+2) = 226, 224, 222 EUR -> 494 940, 490 560, 486 180; roll 3 is its
  jump only (deterministic on R): 100 -> 219 000.
* ``separate_dispatch`` solve (1) of roll r (r < 3): both steps on the forecasts made at t_{r+1}, units online tied
  at t_{r+1} (2: the lower forecast needs 2): 2 x (130 - 2(r+1)) = 256, 252, 248 -> 560 640, 551 880, 543 120;
  roll 3: one step, 130 - 2 x 4 = 122 -> 267 180.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine import _solve_copy_models as scm
from tests.engine.realization_roll import test_split_roll_lp as srl

pack = scm.pack
pytestmark = pytest.mark.solver
STEPS = ("t01", "t02", "t03", "t04")
FACTOR = 8760 / 4


@pytest.fixture(autouse=True)
def _strict(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _made(method: str) -> list[int]:
    return [1, 2, 3, 4] if method == "separate_dispatch" else [2, 3, 4]


def _forecasts(method: str) -> dict:
    """``wp``: ``R`` 0.5 made at t01; ``upper`` / ``lower`` made at t_k, 0.80 / 0.20 + 0.01k on t_k..t04."""
    out = {"R": {"t01": {t: 0.5 for t in STEPS}}}
    for b, base in (("upper", 0.80), ("lower", 0.20)):
        out[b] = {f"t0{k}": {t: round(base + 0.01 * k, 10) for t in STEPS[k - 1:]} for k in _made(method)}
    return out


def _toy(method: str = "within_solve", *, outage: bool = False) -> dict:
    """The §4 toy (module docstring).  *outage* (T-a2): ``coal`` min load 0.5 and in the stochastic group, its
    availability 1 in every forecast and 0 at t02 on ``R``; ``peak`` 10 MW at 100 EUR/MWh covers the gap."""
    entities = [["node", "elec", None], ["unit", "coal", None], ["unit", "wind", None], ["profile", "wp", None],
                ["unit__outputNode", ["coal", "elec"], None], ["unit__outputNode", ["wind", "elec"], None],
                ["group__unit", ["sg", "wind"], None], ["group__node", ["sg", "elec"], None],
                ["unit__node__profile", ["wind", "elec", "wp"], None]]
    active = [("node", "elec"), ("unit", "coal"), ("unit", "wind")]
    startup = "integer_fix_in_uc" if method == "separate_dispatch" else "integer"
    values = [*srl._node("elec", "balance", pack(-10.0, "float")),
              *srl._unit("coal", "elec", 2.0, 20.0), *srl._unit("wind", "elec", 10.0, 0.0),
              ["unit", "coal", "virtual_unitsize", pack(5.0, "float"), "init"],
              ["unit", "coal", "startup_method", pack(startup, "str"), "init"],
              ["unit", "coal", "startup_cost", pack(1.0, "float"), "init"],
              ["unit__node__profile", ["wind", "elec", "wp"], "profile_method", pack("upper_limit", "str"), "init"],
              ["profile", "wp", "profile", srl._series(_forecasts(method)), "init"]]
    if outage:
        entities += [["unit", "peak", None], ["unit__outputNode", ["peak", "elec"], None],
                     ["group__unit", ["sg", "coal"], None]]
        active.append(("unit", "peak"))
        avail = {"R": {"t01": {"t01": 1.0, "t02": 0.0, "t03": 1.0, "t04": 1.0}}}
        for b in ("upper", "lower"):
            avail[b] = {f"t0{k}": {t: 1.0 for t in STEPS[k - 1:]} for k in _made(method)}
        values += [*srl._unit("peak", "elec", 10.0, 100.0),
                   ["unit", "coal", "min_load", pack(0.5, "float"), "init"],
                   ["unit", "coal", "availability", srl._series(avail), "init"]]
    fan = srl._fan([("upper", {f"t0{k}": 1.0 for k in _made(method)}, "no"),
                    ("lower", {f"t0{k}": 3.0 for k in _made(method)}, "no"),
                    ("R", {"t01": 0.0}, "yes")])
    spec = srl._model(STEPS, jump=1, horizon=2, fan=fan, entities=entities, active=active, values=values)
    if method != "within_solve":
        for row in spec["parameter_values"]:
            if row[:3] == ["solve", "st", "dispatch_realization_method"]:
                row[3] = pack(method, "str")
    return spec


def _cells(step, var: str, p: str) -> dict[tuple[str, str], float]:
    frame = step.solution.value(var).with_columns(pl.col(c).cast(pl.Utf8) for c in ("p", "d", "t"))
    return {(d, t): v for d, t, v in frame.filter(pl.col("p") == p).select("d", "t", "value").rows()}


def _wind_caps(mps) -> dict[tuple[str, str], float]:
    """``{(d, t): wind's flow ceiling in MW}`` of an LP: the ``profile_flow_upper_limit`` rows' rhs over their wind
    flow coefficient (the flow is per unit size, 10 MW)."""
    lp = scm.lp_snapshot().parse_mps(mps)
    out = {}
    for row in lp.rows:
        if not row.startswith("profile_flow_upper_limit[wind,"):
            continue
        d, t = row[:-1].split(",")[-2:]
        coef = lp.coefs[(row, f"v_flow[wind,wind,elec,{d},{t}]")]
        out[(d, t)] = round(lp.rhs.get(row, 0.0) / coef * 10.0, 9)
    return out


def _weights(step) -> dict[tuple[str, str], float]:
    frame = step.flex_data.pdt_branch_weight
    frame = frame if isinstance(frame, pl.DataFrame) else frame.frame
    return {(str(d), str(t)): v for d, t, v in frame.select("d", "t", "value").rows()}


@pytest.fixture(scope="module")
def within(tmp_path_factory):
    return scm.run_spec(_toy("within_solve"), tmp_path_factory.mktemp("jump1_within"))


@pytest.fixture(scope="module")
def separate(tmp_path_factory):
    return scm.run_spec(_toy("separate_dispatch"), tmp_path_factory.mktemp("jump1_separate"))


REALIZATION = [494_940.0, 490_560.0, 486_180.0, 219_000.0]
SOLVE_ONE = [560_640.0, 551_880.0, 543_120.0, 267_180.0]


def _tail_caps(r: int) -> dict:
    """R_r's wind ceilings: 5 MW at the jump step on ``p1``; the forecasts made at t_{r+2} at t_{r+2}."""
    if r == 3:
        return {("p1", "t04"): 5.0}
    k = r + 2
    tail = STEPS[k - 1]
    return {("p1", STEPS[r]): 5.0, ("p1_upper", tail): round(10 * (0.80 + 0.01 * k), 9),
            ("p1_lower", tail): round(10 * (0.20 + 0.01 * k), 9)}


def _check_realization_rolls(steps: list, mps: list) -> None:
    for r, (step, path) in enumerate(zip(steps, mps, strict=True)):
        assert _wind_caps(path) == pytest.approx(_tail_caps(r)), r
        weights = _weights(step)
        if r < 3:
            k_t = STEPS[r + 1]
            assert weights == {("p1", STEPS[r]): 1.0, ("p1_upper", k_t): 0.25, ("p1_lower", k_t): 0.75}, r
        else:
            assert set(weights) == {("p1", "t04")}, r
        # One realized step, under the period's own name; its units online integral (2, module docstring).
        # Roll 3 has no tail: 1 or 2 units online cost the same there (no start, a stop is free).
        online = _cells(step, "v_online_integer", "coal")
        if r < 3:
            assert online[("p1", STEPS[r])] == pytest.approx(2.0, abs=1e-9), r
        else:
            assert round(online[("p1", "t04")]) in (1, 2), r
        assert all(abs(v - round(v)) < 1e-9 for v in online.values()), (r, online)
        assert [c for c in online if c[0] == "p1"] == [("p1", STEPS[r])], r
        assert _cells(step, "v_flow", "coal")[("p1", STEPS[r])] == pytest.approx(1.0, abs=1e-9), r  # 5 MW / unit size 5


def test_within_solve_jump_one(within):
    """T-w2: 4 solves (today 8), one per roll, no realization solve.  R_r (r < 3) dispatches t_{r+1} under ``p1``
    on R (wind ceiling 0.5 x 10 = 5 MW), its tails ``p1_upper`` / ``p1_lower`` hold t_{r+2} on the forecasts made
    there (10 x (0.80 / 0.20 + 0.01(r+2)) MW) at weights 0.25 / 0.75; roll 3 is deterministic on R.  Units online
    are integral, 2 at each jump step; roll r+1 starts from roll r's 2 units (roll_end_online, and no start cost in
    its objective); objectives as in the module docstring."""
    built, steps = within
    names = [s for s, _, _ in built]
    assert len(names) == 4 and not any("realized_dispatch" in s for s in names), names
    rolls = list(steps.values())
    _check_realization_rolls(rolls, [p for _, p, _ in built])
    for r in range(3):
        carry = rolls[r].handoff.roll_end_online.with_columns(pl.col("process").cast(pl.Utf8))
        assert dict(carry.select("process", "value").rows()) == {"coal": pytest.approx(2.0, abs=1e-9)}, r
    assert srl._objectives(steps) == pytest.approx(REALIZATION, rel=1e-9)


def test_separate_dispatch_jump_one(separate):
    """T-sd1 / T-b5: 8 solves, solve (1) then the realization roll per roll.  Solve (1) of roll r reads the forecasts
    made at its first step t_{r+1}: 10 x (0.80 / 0.20 + 0.01(r+1)) MW at both its steps, no realized step; it pins
    coal's 2 units online at t_{r+1}.  The realization rolls are T-w2's R_r (same ceilings, weights, units online
    and objectives); solve (1) objectives as in the module docstring."""
    built, steps = separate
    names = [s for s, _, _ in built]
    assert len(names) == 8 and ["realized_dispatch" in s for s in names] == [False, True] * 4, names
    rolls = list(steps.values())
    for r in range(4):
        k = r + 1
        own = STEPS[r:r + 2]
        want = {(f"p1_{b}", t): round(10 * (base + 0.01 * k), 9)
                for b, base in (("upper", 0.80), ("lower", 0.20)) for t in own}
        assert _wind_caps(built[2 * r][1]) == pytest.approx(want), r
        assert _weights(rolls[2 * r]) == {(f"p1_{b}", t): w for b, w in (("upper", 0.25), ("lower", 0.75))
                                          for t in own}, r
        pins = rolls[2 * r + 1].flex_data.p_fix_online.frame.with_columns(pl.col(c).cast(pl.Utf8) for c in "pdt")
        assert pins.select("p", "d", "t", "value").rows() == [("coal", "p1", STEPS[r], pytest.approx(2.0))], r
    _check_realization_rolls(rolls[1::2], [p for _, p, _ in built[1::2]])
    objectives = srl._objectives(steps)
    assert objectives[0::2] == pytest.approx(SOLVE_ONE, rel=1e-9)
    assert objectives[1::2] == pytest.approx(REALIZATION, rel=1e-9)


def test_outage_in_the_jump_on_a_pinned_unit(tmp_path):
    """T-a2: ``separate_dispatch`` with the outage (``_toy(outage=True)``): coal is unavailable at t02 on R only.
    Solve (1) of roll 1 (forecasts made at t02, coal available) pins 2 units at t02 (the lower forecast needs 7.8 MW
    > one unit).  R_1's jump t02 cannot hold coal online (availability 0 < min load 0.5): the pin yields, coal is
    decided again (online 0 at t02, W-0592 names it), and ``peak`` covers 10 - 5 = 5 MWh at 100 = 500 EUR.  R_1's
    tails (made at t03, coal available, starts from 0 online, 5 EUR per unit): upper residual 10 - 8.3 = 1.7: one
    coal unit at its min load 2.5 MW (wind curtailed) 50 + 5 = 55 (peak: 170); lower residual 7.7: two units 154 +
    10 = 164 (one unit + peak 2.7: 100 + 5 + 270).  R_1 = 500 + 0.25 x 55 + 0.75 x 164 = 636.75 -> 1 394 482.5.
    Every solve is optimal."""
    from tests._support.coded_warnings import capture_coded

    with capture_coded() as got:
        built, steps = scm.run_spec(_toy("separate_dispatch", outage=True), tmp_path)
    rolls = list(steps.values())
    assert len(rolls) == 8 and all(s.solution.optimal for s in rolls)
    r1 = rolls[3]
    assert "realized_dispatch" in built[3][0]
    assert _cells(rolls[2], "v_online_integer", "coal")[("p1_lower", "t02")] == pytest.approx(2.0, abs=1e-9)
    assert _cells(r1, "v_online_integer", "coal")[("p1", "t02")] == pytest.approx(0.0, abs=1e-9)
    assert _cells(r1, "v_flow", "peak")[("p1", "t02")] == pytest.approx(0.5, abs=1e-9)  # 5 MW / 10
    assert r1.solution.obj == pytest.approx(636.75 * FACTOR, rel=1e-9)
    assert got.names("W-0592") == ["coal"]


def _min_up_toy() -> dict:
    """The toy's timeline, rolls and node with one ``coal`` unit of 10 MW (``virtual_unitsize`` 10, ``existing`` 1,
    ``integer``, min load 0.5, ``minimum_time_method`` min_uptime, ``min_uptime`` 2 h, 20 EUR/MWh, no start-up cost),
    ``peak`` 10 MW at 100 EUR/MWh and ``wind`` 10 MW: realized ``R`` 1, 0, 1, 1 at t01-t04 (made at t01);
    forecasts ``upper`` (weight 1) / ``lower`` (weight 3) made at t02, t03, t04, 1.0 at every step they cover."""
    entities = [["node", "elec", None], ["unit", "coal", None], ["unit", "peak", None], ["unit", "wind", None],
                ["profile", "wp", None], ["unit__outputNode", ["coal", "elec"], None],
                ["unit__outputNode", ["peak", "elec"], None], ["unit__outputNode", ["wind", "elec"], None],
                ["group__unit", ["sg", "wind"], None], ["group__node", ["sg", "elec"], None],
                ["unit__node__profile", ["wind", "elec", "wp"], None]]
    active = [("node", "elec"), ("unit", "coal"), ("unit", "peak"), ("unit", "wind")]
    series = {"R": {"t01": {"t01": 1.0, "t02": 0.0, "t03": 1.0, "t04": 1.0}}}
    for b in ("upper", "lower"):
        series[b] = {f"t0{k}": {t: 1.0 for t in STEPS[k - 1:]} for k in (2, 3, 4)}
    values = [*srl._node("elec", "balance", pack(-10.0, "float")),
              *srl._unit("coal", "elec", 1.0, 20.0), *srl._unit("peak", "elec", 10.0, 100.0),
              *srl._unit("wind", "elec", 10.0, 0.0),
              ["unit", "coal", "virtual_unitsize", pack(10.0, "float"), "init"],
              ["unit", "coal", "startup_method", pack("integer", "str"), "init"],
              ["unit", "coal", "min_load", pack(0.5, "float"), "init"],
              ["unit", "coal", "minimum_time_method", pack("min_uptime", "str"), "init"],
              ["unit", "coal", "min_uptime", pack(2.0, "float"), "init"],
              ["unit__node__profile", ["wind", "elec", "wp"], "profile_method", pack("upper_limit", "str"), "init"],
              ["profile", "wp", "profile", srl._series(series), "init"]]
    fan = srl._fan([("upper", {f"t0{k}": 1.0 for k in (2, 3, 4)}, "no"),
                    ("lower", {f"t0{k}": 3.0 for k in (2, 3, 4)}, "no"),
                    ("R", {"t01": 0.0}, "yes")])
    return srl._model(STEPS, jump=1, horizon=2, fan=fan, entities=entities, active=active, values=values)


def test_within_solve_min_uptime_across_the_seam_and_into_the_next_roll(tmp_path):
    """T-w2, minimum up-time (hand, factor 2190 per MWh-hour).  Roll 0: t01 on R (wind 10 MW covers demand), tails
    t02 on the forecasts (wind 10): coal stays offline (online would force its 5 MW min load): 0.  Roll 1: jump t02
    on R has no wind; coal (200 at 10 MWh) beats peak (1000), so coal starts at t02.  Its min up-time (2 h: a start
    at t02 counts at t03, elapsed 1 h < 2 h, not at t04) reaches across the label into both tails ``p1_upper`` /
    ``p1_lower`` at t03 (their predecessor is p1 t02): online 1 there at min load 5 MW (wind curtailed), 100 in each
    tail, weights 0.25 / 0.75: 300 -> 657 000.  Roll 2: t03 on R (wind 10), the start at t02 carried from roll 1's
    realized jump keeps coal online at t03 at min load: 100; tails t04: offline (2 h done): 219 000 (without the
    carry: 0).  Roll 3: t04 on R, offline: 0."""
    built, steps = scm.run_spec(_min_up_toy(), tmp_path)
    assert len(built) == 4 and not any("realized_dispatch" in s for s, _, _ in built)
    rolls = list(steps.values())
    online = [_cells(s, "v_online_integer", "coal") for s in rolls]
    assert online[1] == pytest.approx({("p1", "t02"): 1.0, ("p1_upper", "t03"): 1.0, ("p1_lower", "t03"): 1.0},
                                      abs=1e-9)
    assert online[2][("p1", "t03")] == pytest.approx(1.0, abs=1e-9)
    assert _cells(rolls[2], "v_flow", "coal")[("p1", "t03")] == pytest.approx(0.5, abs=1e-9)  # 5 MW / unit size 10
    assert srl._objectives(steps) == pytest.approx([0.0, 300 * FACTOR, 100 * FACTOR, 0.0], rel=1e-9, abs=1e-6)


def _prices(folder, roll: str) -> dict[tuple[str, str], float]:
    """``{(d, t): price of elec}`` of *roll* from its ``v_dual_node_balance`` output (EUR/MWh)."""
    frame = pl.read_parquet(folder / "w" / "output_raw" / f"v_dual_node_balance__{roll}.parquet")
    return {(d, t): v for d, t, v in frame.select("period", "time", "elec").rows()}


@pytest.mark.parametrize(("w_b", "state", "objectives"), [
    (0.5, 1.0, [87_600.0, 0.0]),
    (0.25, 0.0, [76_650.0, 131_400.0])])
def test_jump_price_is_the_base_periods_own_dual(tmp_path, w_b, state, objectives):
    """T-b6 (hand): T-w1 (``test_split_roll_lp._tw1``) with gas uncapped (10 MW).  R_0: storing 1 MWh in the jump at
    10 saves 30 under B only: w_B 0.5 -> 15 > 10, stored 1; w_B 0.25 -> 7.5 < 10, 0 (objectives as T-w1: gas was
    never short).  Gas is marginal at t01 in both cases, so the jump price at (p1, t01) is 10 EUR/MWh: p1's own
    nodeBalance dual at weight 1 (with the label at weight 0 there would be no price; a sum over the forecasts'
    duals would give 10 only by accident of the weights).  R_1 (t02, deterministic): gas is the only source with
    headroom (the store holds at most what t02 uses), so the price is 30."""
    spec = srl._tw1(w_b)
    for row in spec["parameter_values"]:
        if row[:3] == ["unit", "gas", "existing"]:
            row[3] = pack(10.0, "float")
    built, steps = scm.run_spec(spec, tmp_path)
    assert len(built) == 2
    first = list(steps.values())[0]
    assert srl._value(first.solution.value("v_state"), n="elec", d="p1", t="t01") == pytest.approx(state, abs=1e-9)
    assert srl._objectives(steps) == pytest.approx(objectives, rel=1e-9, abs=1e-6)
    rolls = list(steps)
    assert _prices(tmp_path, rolls[0]) == {("p1", "t01"): pytest.approx(10.0, rel=1e-9)}
    assert _prices(tmp_path, rolls[1]) == {("p1", "t02"): pytest.approx(30.0, rel=1e-9)}
