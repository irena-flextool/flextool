"""Parts B and C (c14c, item 99): startups/shutdowns equal the change in units online, and the ramp room of
a unit with online variables is split by unit state instead of flat per declared start/stop.

Part B: ``online_change`` replaces the two ``>=`` rows ``online__startup`` / ``online__shutdown`` with one
equality (``on - on_prev - su + sd = 0``); ``maxStartup`` becomes ``su <= on_t`` and ``maxShutdown`` becomes
``sd <= built - on_t`` (same-period predecessor only, owner question Q-c14c-A).  Together these make an
ORPHAN startup/shutdown (one with no matching change in units online) impossible.

Part C (``model/core.py:_eq_ramp``, design §3): a unit with online variables now ramps on ``v_online`` /
``v_startup`` / ``v_shutdown`` (continuing units change by at most rho-bar, a starter reaches at most SU, a
stopper floors at F) instead of on ``existing_count`` plus a flat ``+-v_startup``/``v_shutdown`` of
coefficient 1 -- the item-99 exploit (declaring a start/stop pair buys a full unit of ramp room for free) is
gone because ``v_startup`` / ``v_shutdown`` now enter the ramp row with THEIR OWN physical coefficients (SU,
F, SD'), not a flat 1.  C1-C9 below are design section 7's part-C tests; each docstring derives its expected
numbers from the fixture's data (ml, ramp speed, step duration), independently re-solved and verified before
being written here (never copied from the implementation's own console output).
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._lp_rows import coef, row_bounds, row_terms
from tests.engine._unit_db import FLEXTOOL_ROOT, PLANT, b64, build, make_db, no_ratio_constraint, run, utf8

pytestmark = pytest.mark.solver

TOL = 1e-7

TWO_PERIODS_ENTS = [["timeset", "ta", None], ["timeset", "tb", None]]


def _pmap(values: dict, index_name: str = "period") -> dict:
    return {"index_type": "str", "index_name": index_name, "data": [[k, v] for k, v in values.items()]}


def _tmap(values: dict[str, float]) -> list:
    return b64({"index_type": "str", "index_name": "time", "data": [[t, v] for t, v in values.items()]}, "map")


def _two_periods() -> list:
    return [
        ["timeset", "ta", "timeline", b64("y2020", "str")],
        ["timeset", "tb", "timeline", b64("y2020", "str")],
        ["timeset", "ta", "timeset_duration", b64(_pmap({"t0001": 24.0}, "time"), "map")],
        ["timeset", "tb", "timeset_duration", b64(_pmap({"t0025": 24.0}, "time"), "map")],
        ["solve", "y2020_2day_dispatch", "period_timeset", b64(_pmap({"p2020": "ta", "p2025": "tb"}), "map")],
        ["solve", "y2020_2day_dispatch", "realized_periods",
         b64({"value_type": "str", "data": ["p2020", "p2025"]}, "array")],
    ]


def _ramp(speed: float, rel: list) -> list:
    return [["unit__outputNode", rel, "ramp_method", b64("ramp_limit", "str")],
            ["unit__outputNode", rel, "ramp_speed_up", b64(speed, "float")],
            ["unit__outputNode", rel, "ramp_speed_down", b64(speed, "float")]]


def _reserve_entities_values(r: str, ud: str, group: str, providers: list,
                             method: str = "timeseries_only", reservation: float = 50.0,
                             penalty: float = 1000.0) -> tuple[list, list]:
    grp = [r, ud, group]
    ents = [["upDown", ud, None], ["reserve", r, None], ["reserve__upDown__group", grp, None]]
    vals = [["reserve__upDown__group", grp, "reserve_method", b64(method, "str")],
            ["reserve__upDown__group", grp, "penalty_reserve", b64(penalty, "float")]]
    if reservation:
        vals.append(["reserve__upDown__group", grp, "reservation", b64(reservation, "float")])
    for cls, p, n in providers:
        rel = f"reserve__upDown__{cls}__node"
        ents.append([rel, [r, ud, p, n], None])
        vals += [[rel, [r, ud, p, n], "is_enabled", b64("yes", "str")],
                 [rel, [r, ud, p, n], "max_share", b64(1.0, "float")],
                 [rel, [r, ud, p, n], "reliability", b64(1.0, "float")]]
    return ents, vals


_E_BOILER_ENTS = [["unit", "e_boiler", None], ["unit__inputNode", ["e_boiler", "west"], None],
                  ["unit__outputNode", ["e_boiler", "heat"], None]]
_E_BOILER_VALUES = [["unit", "e_boiler", "efficiency", b64(0.5, "float")],
                    ["unit", "e_boiler", "existing", b64(100.0, "float")]]


def test_startups_equal_online_increases(tmp_path_factory) -> None:
    """B1 (item 99 repro: the lp_snapshot ``ramp_reserve_online`` fixture's data -- coal_plant linear online
    with an output ramp (0.005/min) and up/down reserve, e_boiler linear online with an input ramp (0.002/min)
    and down reserve, startup costs 20 / 5).

    The ``online_change`` equality GUARANTEES the per-step identity ``su - sd = on - on_prev`` exactly, for
    EVERY process and step (checked for both coal_plant and e_boiler, relational, +-1e-7) -- an ORPHAN
    startup/shutdown (one with no matching change in units online) is impossible, and ``maxStartup`` /
    ``maxShutdown`` additionally bound su / sd individually (checked against the solved online count).

    For coal_plant specifically, Sigma(su) == Sigma(Delta on)+ EXACTLY (no excess at all, not even a matched
    pair): its ramp (0.005/min) leaves no room worth buying.  e_boiler is deliberately NOT asserted the same
    way: Sigma(su) stays ~3.78 against Sigma(Delta on)+ ~0.11 here (reproducing the design's own evidence,
    section 6.1's ``ramp_reserve_online`` "B (eq)" objective 113107597.9 -- UNCHANGED from the baseline, and
    design section 0.1's "Before: ... Sigma su 3.78" almost exactly) because the actual item-99 exploit for a
    unit with a RAMP limit lives in core._eq_ramp's ramp rows (a flat full unit of room per start/stop,
    Part C, not landed in this commit) -- online_change / maxStartup / maxShutdown alone make that pair
    EXACT and BOUNDED (su, sd both correctly track online, no orphan), but a genuine matched pair can still be
    cheaper than not pairing until the ramp rows themselves stop rewarding it.  The "no excess at all" claim
    for a ramp-limited process is C1's (design section 7), checked once Part C lands.
    """
    eu, vu = _reserve_entities_values("ru", "up", "electricity", [("unit", "coal_plant", "west")],
                                      penalty=5000.0, reservation=40.0)
    ed, vd = _reserve_entities_values("rd", "down", "electricity",
                                      [("unit", "coal_plant", "west"), ("unit", "e_boiler", "west")],
                                      penalty=5000.0, reservation=40.0)
    values = (_E_BOILER_VALUES + vu + vd
              + _ramp(0.005, ["coal_plant", "west"])
              + [["unit__inputNode", ["e_boiler", "west"], "ramp_method", b64("ramp_limit", "str")],
                 ["unit__inputNode", ["e_boiler", "west"], "ramp_speed_up", b64(0.002, "float")],
                 ["unit__inputNode", ["e_boiler", "west"], "ramp_speed_down", b64(0.002, "float")],
                 ["unit", "e_boiler", "conversion_method", b64("constant_efficiency", "str")],
                 ["constraint", "coal_chp_fix", "is_enabled", b64("no", "str")],
                 ["node", "west", "inflow", b64(-250.0, "float")],
                 ["unit", "coal_plant", "startup_method", b64("linear", "str")],
                 ["unit", "coal_plant", "min_load", b64(0.3, "float")],
                 ["unit", "e_boiler", "startup_method", b64("linear", "str")],
                 ["unit", "e_boiler", "min_load", b64(0.2, "float")],
                 ["unit", "coal_plant", "startup_cost", b64(20.0, "float")],
                 ["unit", "e_boiler", "startup_cost", b64(5.0, "float")]])
    entities = _E_BOILER_ENTS + eu + ed
    url, scen = make_db(tmp_path_factory, alt="b1_startup_eq", values=values,
                        base_chain=["init", "west", "coal", "coal_chp", "heat"], entities=entities)
    step = list(run(tmp_path_factory, url, scen).values())[-1]

    for p in ("coal_plant", "e_boiler"):
        on = utf8(step.solution.value("v_online_linear")).filter(pl.col("p") == p).sort("t")
        su = utf8(step.solution.value("v_startup_linear")).filter(pl.col("p") == p).sort("t")
        sd = utf8(step.solution.value("v_shutdown_linear")).filter(pl.col("p") == p).sort("t")
        on_v, su_v, sd_v = on["value"].to_list(), su["value"].to_list(), sd["value"].to_list()
        existing_count = float(utf8(step.flex_data.p_process_existing_count.frame)
                               .filter(pl.col("p") == p)["value"][0])   # unitsize falls back to existing -> 1
        # The equality identity: su - sd = on - on_prev at every interior step (cyclic wrap at t0001 means
        # on_prev there is on[-1], the last step -- also exact).
        for i in range(len(on_v)):
            prev = on_v[i - 1]   # Python negative indexing: i=0 -> on_v[-1], the cyclic wrap
            assert su_v[i] - sd_v[i] == pytest.approx(on_v[i] - prev, abs=TOL)
        # No orphans: su bounded by online_now, sd bounded by built - online_now (maxStartup / maxShutdown,
        # same-period predecessor -- Q-c14c-A). Both sides in units of unitsize (existing_count = 1 here).
        for i in range(len(on_v)):
            assert su_v[i] <= on_v[i] + TOL
            assert sd_v[i] <= existing_count - on_v[i] + TOL

    su_coal = utf8(step.solution.value("v_startup_linear")).filter(pl.col("p") == "coal_plant")["value"].sum()
    on_coal = (utf8(step.solution.value("v_online_linear")).filter(pl.col("p") == "coal_plant")
              .sort("t")["value"].to_list())
    delta_pos_coal = sum(max(0.0, on_coal[i] - on_coal[i - 1]) for i in range(len(on_coal)))
    assert su_coal == pytest.approx(delta_pos_coal, abs=TOL)


def test_online_change_rows_replace_the_two_inequalities(tmp_path_factory) -> None:
    """B2.  TWO_PERIODS + coal_plant linear online (min_load 0.2) + a ramp: no ``online__startup`` /
    ``online__shutdown`` row exists anywhere; ``online_change_linear`` carries +online(now), -startup,
    +shutdown and the (sign-flipped) predecessor term; ``maxStartup_linear`` is ``su - on <= 0`` (RHS 0, no
    invest/divest terms, unlike the old existing-count cap); ``maxShutdown_linear`` is
    ``sd + on - invest + divest <= existing`` but is ABSENT at p2025 t0025 (its predecessor, p2020 t0024, is in
    a different period -- owner question Q-c14c-A default)."""
    vals = _ramp(0.002, ["coal_plant", "west"]) + _two_periods() + [
        ["unit", "coal_plant", "startup_method", b64("linear", "str")],
        ["unit", "coal_plant", "min_load", b64(0.2, "float")]]
    url, scen = make_db(tmp_path_factory, alt="b2_rows", values=vals, base_chain=["init", "west", "coal"],
                        entities=TWO_PERIODS_ENTS)
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    from tests.engine._unit_db import build

    pb = build(step.flex_data)
    m = pb.canonicalise()
    assert not any(nm.startswith("online__startup") or nm.startswith("online__shutdown") for nm in m.row_names)

    oc = row_terms(pb, "online_change_linear")[f"online_change_linear[{PLANT},p2025,t0025]"]
    assert coef(oc, f"v_online_linear[{PLANT},p2025,t0025]") == pytest.approx(1.0)
    assert coef(oc, f"v_shutdown_linear[{PLANT},p2025,t0025]") == pytest.approx(1.0)
    assert coef(oc, f"v_startup_linear[{PLANT},p2025,t0025]") == pytest.approx(-1.0)
    assert coef(oc, f"v_online_linear[{PLANT},p2020,t0024]") == pytest.approx(-1.0)
    bounds = row_bounds(pb, "online_change_linear")[f"online_change_linear[{PLANT},p2025,t0025]"]
    assert bounds == (pytest.approx(0.0), pytest.approx(0.0))

    ms = row_terms(pb, "maxStartup_linear")[f"maxStartup_linear[{PLANT},p2025,t0025]"]
    assert coef(ms, f"v_startup_linear[{PLANT},p2025,t0025]") == pytest.approx(1.0)
    assert coef(ms, f"v_online_linear[{PLANT},p2025,t0025]") == pytest.approx(-1.0)
    assert set(ms) == {f"v_startup_linear[{PLANT},p2025,t0025]", f"v_online_linear[{PLANT},p2025,t0025]"}
    ms_bounds = row_bounds(pb, "maxStartup_linear")[f"maxStartup_linear[{PLANT},p2025,t0025]"]
    assert ms_bounds[1] == pytest.approx(0.0)

    mshut = row_terms(pb, "maxShutdown_linear")
    assert f"maxShutdown_linear[{PLANT},p2025,t0025]" not in mshut
    # An interior (same-period) step keeps the row.
    assert f"maxShutdown_linear[{PLANT},p2020,t0012]" in mshut
    interior = mshut[f"maxShutdown_linear[{PLANT},p2020,t0012]"]
    assert coef(interior, f"v_shutdown_linear[{PLANT},p2020,t0012]") == pytest.approx(1.0)
    assert coef(interior, f"v_online_linear[{PLANT},p2020,t0012]") == pytest.approx(1.0)


def test_boundary_capacity_drop_without_maxshutdown_stays_feasible(tmp_path_factory) -> None:
    """Critique item 9 / owner question Q-c14c-A.  TWO_PERIODS + coal_plant linear online (min_load 0.2,
    ramp 0.002/min -- same SU/SD' as A4), existing a period map 500 MW (p2020) / 300 MW (p2025) -- a
    retirement between periods, unitsize falling back to the MAX existing over periods (500, as the
    cascade's own per-period-MAX rule for a 1d_map ``existing`` dictates): existing_count 1.0 in p2020 drops
    to 0.6 in p2025.  Demand keeps the plant near built capacity through p2020 t0024 and within p2025's
    lower cap after, so the units online genuinely must drop across the boundary.

    ``maxShutdown_linear`` is absent at the cross-period step: using the NEW period's (lower) existing_count
    there would wrongly cap this step's shutdown by the SMALLER number, when the units actually online a
    moment before are the LARGER, pre-retirement count -- Q-c14c-A's default.  Nothing else blocks the drop
    either: the run SOLVES (a hard infeasibility, unlike a penalised node-balance deficit, would raise inside
    ``run()``'s solution check), and online_change's equality plus maxStartup (``su <= on_now``) together
    force ``sd = su - on_now + on_prev >= on_prev - on_now >= on_prev - existing_count[p2025]`` -- the
    shutdown is bounded by the units online BEFORE the boundary, never by the new, lower capacity.  The A4
    pre-shutdown reduction row (``maxFlow_online_shutdown_linear``) is present at p2020 t0024 -- this scenario
    also exercises that mechanism, and the solved schedule (a witness at min_load) satisfies it only because
    the solve is feasible at all."""
    vals = (_ramp(0.002, ["coal_plant", "west"]) + _two_periods() + [
        ["unit", PLANT, "existing", b64(_pmap({"p2020": 500.0, "p2025": 300.0}), "map")],
        ["unit", PLANT, "startup_method", b64("linear", "str")],
        ["unit", PLANT, "min_load", b64(0.2, "float")],
        ["node", "west", "inflow", _tmap({f"t{i:04d}": -480.0 for i in range(1, 25)}
                                         | {f"t{i:04d}": -250.0 for i in range(25, 49)})]])
    url, scen = make_db(tmp_path_factory, alt="b2b_boundary_drop", values=vals,
                        base_chain=["init", "west", "coal"], entities=TWO_PERIODS_ENTS)
    step = list(run(tmp_path_factory, url, scen).values())[-1]   # raises if hard-infeasible
    pb = build(step.flex_data)

    ec = utf8(step.flex_data.p_process_existing_count.frame).filter(pl.col("p") == PLANT).sort("d")
    ec_p2020 = float(ec.filter(pl.col("d") == "p2020")["value"][0])
    ec_p2025 = float(ec.filter(pl.col("d") == "p2025")["value"][0])
    assert ec_p2020 == pytest.approx(1.0)
    assert ec_p2025 == pytest.approx(0.6)

    on = utf8(step.solution.value("v_online_linear")).filter(pl.col("p") == PLANT)
    sd = utf8(step.solution.value("v_shutdown_linear")).filter(pl.col("p") == PLANT)
    on_prev = float(on.filter((pl.col("d") == "p2020") & (pl.col("t") == "t0024"))["value"][0])
    sd_now = float(sd.filter((pl.col("d") == "p2025") & (pl.col("t") == "t0025"))["value"][0])
    assert on_prev > ec_p2025 + TOL   # the premise this test needs: the boundary genuinely forces a drop
    assert sd_now >= on_prev - ec_p2025 - TOL   # bounded by the units online BEFORE the boundary

    mshut = row_terms(pb, "maxShutdown_linear")
    assert f"maxShutdown_linear[{PLANT},p2025,t0025]" not in mshut   # Q-c14c-A: no cross-period row

    mred = row_terms(pb, "maxFlow_online_shutdown_linear", f"{PLANT},coal_market,west,")
    assert f"maxFlow_online_shutdown_linear[{PLANT},coal_market,west,p2020,t0024]" in mred   # A4 engaged


def test_single_integer_unit_has_no_start_stop_pair(tmp_path_factory) -> None:
    """B3 (critique item 4): e_boiler as a single BINARY unit (existing 100 MW, unitsize falls back to
    existing itself when virtual_unitsize is unauthored -> existing_count = 1 exactly: one integer unit),
    linear ramp on its input (0.002/min) and a DOWN reserve requirement (40 MW, penalty 5000) it can provide
    while online -- the binding incentive critique item 4 asks for: startup_cost 0 makes toggling online
    status free, and before the fix, declaring startup / shutdown independently of the real state change let
    the solver swing the unit's committed status opportunistically to harvest reserve headroom cheaply
    (measured: Sigma(v_startup_integer) = 3.76, Sigma(v_shutdown_integer) = 1.0 -- excess far beyond any real
    transition -- at objective 125,560,000).  After this commit, online_change's equality plus maxStartup /
    maxShutdown tie su, sd exactly to the real change in committed units (and, for a single integer unit,
    make su and sd both positive at the same step impossible: on_t in {0, 1}, so one of su <= on_t /
    sd <= built - on_t is always 0): the toggle no longer helps, so the solver instead commits the unit online
    for the whole horizon (su = sd = 0 throughout, at the much higher true-cost objective 688,974,000).
    Relationally, Sigma(su) == Sigma(Delta on)+ (both 0, since the unit never actually starts or stops here)."""
    ed, vd = _reserve_entities_values("rd", "down", "electricity", [("unit", "e_boiler", "west")],
                                      penalty=5000.0, reservation=40.0)
    values = (_E_BOILER_VALUES + vd
              + [["unit__inputNode", ["e_boiler", "west"], "ramp_method", b64("ramp_limit", "str")],
                 ["unit__inputNode", ["e_boiler", "west"], "ramp_speed_up", b64(0.002, "float")],
                 ["unit__inputNode", ["e_boiler", "west"], "ramp_speed_down", b64(0.002, "float")],
                 ["unit", "e_boiler", "conversion_method", b64("constant_efficiency", "str")],
                 ["constraint", "coal_chp_fix", "is_enabled", b64("no", "str")],
                 ["node", "west", "inflow", b64(-250.0, "float")],
                 ["unit", "e_boiler", "startup_method", b64("binary", "str")],
                 ["unit", "e_boiler", "min_load", b64(0.2, "float")],
                 ["unit", "e_boiler", "startup_cost", b64(0.0, "float")]])
    entities = _E_BOILER_ENTS + ed
    url, scen = make_db(tmp_path_factory, alt="b3_single_int", values=values,
                        base_chain=["init", "west", "coal", "coal_chp", "heat"], entities=entities)
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    ec = utf8(step.flex_data.p_process_existing_count.frame).filter(pl.col("p") == "e_boiler")["value"][0]
    assert ec == pytest.approx(1.0)   # exactly one integer unit (the premise this test needs)

    su = utf8(step.solution.value("v_startup_integer")).filter(pl.col("p") == "e_boiler").sort("t")
    sd = utf8(step.solution.value("v_shutdown_integer")).filter(pl.col("p") == "e_boiler").sort("t")
    on = utf8(step.solution.value("v_online_integer")).filter(pl.col("p") == "e_boiler").sort("t")
    su_v, sd_v, on_v = su["value"].to_list(), sd["value"].to_list(), on["value"].to_list()
    for s, d in zip(su_v, sd_v):
        assert min(s, d) == pytest.approx(0.0, abs=TOL)
    delta_pos = sum(max(0.0, on_v[i] - on_v[i - 1]) for i in range(len(on_v)))
    assert sum(su_v) == pytest.approx(delta_pos, abs=1e-5)
    # The excess toggling this commit fixes (the fail-before this test needs): old code reached Sigma(su)
    # 3.76 here; after the fix the unit commits online for the whole horizon instead.
    assert sum(su_v) < 0.5
    assert sum(sd_v) < 0.5


# ---------------------------------------------------------------------------
# Part C (design section 3, c14c item 99): ramp rows of online units on online capacity, split by state.
# ---------------------------------------------------------------------------

_C1_VALUES = [
    ["unit__outputNode", [PLANT, "west"], "ramp_method", b64("ramp_limit", "str")],
    ["unit__outputNode", [PLANT, "west"], "ramp_speed_up", b64(0.005, "float")],
    ["unit__outputNode", [PLANT, "west"], "ramp_speed_down", b64(0.005, "float")],
    ["unit", PLANT, "min_load", b64(0.1, "float")],
    ["node", "west", "penalty_up", b64(1000.0, "float")],
    ["node", "west", "penalty_down", b64(1.0e9, "float")],
]


def _step_demand(low: float, high: float, switch: int = 25, last: int = 72) -> list:
    """``node west inflow``: ``-low`` MW for ``t0001..t{switch-1}``, ``-high`` from ``t{switch}`` on."""
    values = {f"t{i:04d}": (-low if i < switch else -high) for i in range(1, last + 1)}
    return [["node", "west", "inflow", b64(_pmap(values, "time"), "map")]]


def test_fake_starts_buy_no_ramp_room(tmp_path_factory) -> None:
    """C1 (``uc_ramp_free_start`` data, design 7/C1 + critique item 12 -- coal_plant ALONE at west, existing
    500 MW = unitsize fallback -> existing_count 1 exactly, no other supply at west): min_load 0.1, ramp
    0.005/min (rho-bar = 0.005*60*1h = 0.3 on an interior hourly step), demand 100 MW t0001-t0024 then 400 MW
    from t0025 on; ``penalty_down`` (surplus) 1e9 blocks pre-ramping ahead of the jump by dumping excess
    early, ``penalty_up`` (shortfall) 1000 lets the physical shortfall actually show up instead of being
    hidden by some other substitution.

    Hand derivation: being online costs nothing extra (no cost is tied to v_online itself) and MORE online
    capacity only ever helps later ramp room, so the solver pins v_online = 1 (= existing_count) throughout,
    in particular at t0024 (flow there is exactly the 100 MW demand, 0.2 of existing).  At t0025 the unit is
    continuing (no start helps: su = sd = 0 is optimal under the equality, see below), so the up row gives
    flow(t0025) <= flow(t0024) + rho-bar*(on - su) = 0.2 + 0.3*1 = 0.5 (250 MW) -- 150 MW of the 400 MW
    demanded (0.8 - 0.2 - 0.3 = 0.3, *500 MW = 150 MWh) is unserved.  Because every term of the ramp row is
    now an exact physical bound (no more flat +-1 per declared start/stop), paying for a free start/stop pair
    buys NOTHING here (design section 1.3: a single-output arc's pair gain is -1/2*rho*step < 0): the
    objective is identical whether startup_cost is 0 or 1000, and with cost > 0 the solver reports zero
    startups.
    """
    url0, scen0 = make_db(tmp_path_factory, alt="c1_cost0",
                          values=_C1_VALUES + _step_demand(100.0, 400.0) + [
                              ["unit", PLANT, "startup_method", b64("linear", "str")],
                              ["unit", PLANT, "startup_cost", b64(0.0, "float")]],
                          base_chain=["init", "west", "coal"])
    step0 = list(run(tmp_path_factory, url0, scen0).values())[-1]
    url1, scen1 = make_db(tmp_path_factory, alt="c1_cost1000",
                          values=_C1_VALUES + _step_demand(100.0, 400.0) + [
                              ["unit", PLANT, "startup_method", b64("linear", "str")],
                              ["unit", PLANT, "startup_cost", b64(1000.0, "float")]],
                          base_chain=["init", "west", "coal"])
    step1 = list(run(tmp_path_factory, url1, scen1).values())[-1]

    assert step0.solution.obj == pytest.approx(step1.solution.obj, rel=1e-6)
    su1 = utf8(step1.solution.value("v_startup_linear")).filter(pl.col("p") == PLANT)["value"].sum()
    assert su1 == pytest.approx(0.0, abs=TOL)

    flow25 = (utf8(step1.solution.value("v_flow")).filter((pl.col("p") == PLANT) & (pl.col("t") == "t0025"))
              ["value"][0])
    assert flow25 * 500.0 == pytest.approx(250.0, abs=1e-5)
    unserved25 = (utf8(step1.solution.value("vq_state_up")).filter(pl.col("t") == "t0025")["value"][0])
    assert unserved25 == pytest.approx(150.0, abs=1e-5)


def test_integer_fake_starts_buy_no_ramp_room(tmp_path_factory) -> None:
    """C9: C1 with ``startup_method`` ``binary`` (the ``_integer`` family's own rows) -- same scenario, same
    hand derivation (a single existing_count-1 integer unit has no room to be partially online, so the
    result is the same: v_online pinned at 1 throughout, flow(t0025) = 0.5*500 = 250 MW, unserved
    150 MWh, objective unchanged between startup_cost 0 and 1000, zero startups at cost 1000).

    This fixture happens to be solved identically before and after the fix (a single always-on unit never
    needs to declare a start here), so the objective/startup-sum checks above do not themselves fail before
    this commit; the row-level check below does: at an interior continuing step the up row's
    ``v_startup_integer`` coefficient is the ``_integer`` family's OWN c*(SU - rho-bar) = 1*(0.25 - 0.3) =
    -0.05 (canonical sign +0.05; SU = min(1, 0.1 + 0.5*0.3) = 0.25 with this fixture's min_load 0.1) -- before
    this commit every ramp row carried a flat coefficient-1 ``v_startup`` term regardless of family."""
    url0, scen0 = make_db(tmp_path_factory, alt="c9_cost0",
                          values=_C1_VALUES + _step_demand(100.0, 400.0) + [
                              ["unit", PLANT, "startup_method", b64("binary", "str")],
                              ["unit", PLANT, "startup_cost", b64(0.0, "float")]],
                          base_chain=["init", "west", "coal"])
    step0 = list(run(tmp_path_factory, url0, scen0).values())[-1]
    url1, scen1 = make_db(tmp_path_factory, alt="c9_cost1000",
                          values=_C1_VALUES + _step_demand(100.0, 400.0) + [
                              ["unit", PLANT, "startup_method", b64("binary", "str")],
                              ["unit", PLANT, "startup_cost", b64(1000.0, "float")]],
                          base_chain=["init", "west", "coal"])
    step1 = list(run(tmp_path_factory, url1, scen1).values())[-1]

    assert step0.solution.obj == pytest.approx(step1.solution.obj, rel=1e-6)
    su1 = utf8(step1.solution.value("v_startup_integer")).filter(pl.col("p") == PLANT)["value"].sum()
    assert su1 == pytest.approx(0.0, abs=TOL)
    flow25 = (utf8(step1.solution.value("v_flow")).filter((pl.col("p") == PLANT) & (pl.col("t") == "t0025"))
              ["value"][0])
    assert flow25 * 500.0 == pytest.approx(250.0, abs=1e-5)

    pb = build(step1.flex_data)
    row = row_terms(pb, "ramp_sink_up_constraint")[f"ramp_sink_up_constraint[{PLANT},coal_market,west,p2020,"
                                                   "t0012]"]
    assert coef(row, f"v_startup_integer[{PLANT}") == pytest.approx(0.05, abs=1e-6)


def test_room_on_online_capacity(tmp_path_factory) -> None:
    """C2: coal_plant split into 2 units of 250 MW (``virtual_unitsize`` 250, existing 500 -> existing_count
    2), binary online, ramp 0.005/min on its output (rho-bar = 0.3 at an interior hourly step, c = 1 -- no
    capacity_max_coeff authored); coal_chp left WITHOUT a startup_method (built, non-online) but given its
    own output ramp, 0.01/min (rho-bar_chp = 0.6), existing_count_chp = 1 (virtual_unitsize unauthored falls
    back to existing itself).  Demand 200 MW at west (needs at most one of coal_plant's 2 units).

    Hand derivation, canonical row sign (a term's canonical coefficient = -(its RHS-formula value), e.g. the
    c14a built row's ``+rho-bar*c`` RHS term shows as ``-rho-bar*c`` here): coal_plant's up row carries
    v_online_integer at -(rho-bar*c) = -0.3*1 = -0.3 and NO existing_count term at all (RHS bound 0: all of
    the built row's existing_count/invest/divest terms are online-row-only, i.e. dropped here).  coal_chp's
    up row keeps the untouched c14a shape: no v_online term, RHS bound = +rho-bar_chp*c_chp*existing_count_chp
    = 0.6*1*1 = 0.6."""
    values = no_ratio_constraint() + [
        ["unit", PLANT, "virtual_unitsize", b64(250.0, "float")],
        ["unit", PLANT, "startup_method", b64("binary", "str")],
        ["unit__outputNode", [PLANT, "west"], "ramp_method", b64("ramp_limit", "str")],
        ["unit__outputNode", [PLANT, "west"], "ramp_speed_up", b64(0.005, "float")],
        ["unit__outputNode", [PLANT, "west"], "ramp_speed_down", b64(0.005, "float")],
        ["unit__outputNode", ["coal_chp", "west"], "ramp_method", b64("ramp_limit", "str")],
        ["unit__outputNode", ["coal_chp", "west"], "ramp_speed_up", b64(0.01, "float")],
        ["unit__outputNode", ["coal_chp", "west"], "ramp_speed_down", b64(0.01, "float")],
        ["node", "west", "inflow", b64(-200.0, "float")],
    ]
    url, scen = make_db(tmp_path_factory, alt="c2_room", values=values,
                        base_chain=["init", "west", "coal", "coal_chp", "heat"])
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    pb = build(step.flex_data)

    plant_terms = row_terms(pb, "ramp_sink_up_constraint")[f"ramp_sink_up_constraint[{PLANT},coal_market,west,"
                                                           "p2020,t0012]"]
    assert coef(plant_terms, f"v_online_integer[{PLANT}") == pytest.approx(-0.3)
    plant_bounds = row_bounds(pb, "ramp_sink_up_constraint")[f"ramp_sink_up_constraint[{PLANT},coal_market,"
                                                             "west,p2020,t0012]"]
    assert plant_bounds[1] == pytest.approx(0.0)

    chp_terms = row_terms(pb, "ramp_sink_up_constraint")["ramp_sink_up_constraint[coal_chp,coal_chp,west,"
                                                         "p2020,t0012]"]
    assert "v_online" not in "".join(chp_terms)
    chp_bounds = row_bounds(pb, "ramp_sink_up_constraint")["ramp_sink_up_constraint[coal_chp,coal_chp,west,"
                                                           "p2020,t0012]"]
    assert chp_bounds[1] == pytest.approx(0.6)


def test_real_start_reaches_su(tmp_path_factory) -> None:
    """C3 (guard): coal_plant alone at west, min_load 0.3, ramp 0.005/min (rho-bar 0.3), startup_cost 10,
    huge ``penalty_down`` (no pre-ramping/overshoot allowed); demand 0 MW t0001-t0024 then 400 MW from
    t0025 (the unit is OFFLINE the whole first day: with demand 0 and penalty_down huge there's no reason to
    be online, no min-load floor to satisfy).

    Hand derivation: at t0025 the unit starts (su = 1): SU = min(1, ml + 1/2*ramp_up*60*1h) =
    min(1, 0.3 + 0.5*0.3) = 0.45 -- flow(t0025) = 0.45 (225 MW).  t0026 is a continuing step (no further
    start helps): flow <= flow(t0025) + rho-bar = 0.45 + 0.3 = 0.75 (375 MW), still short of the 400 MW (0.8)
    demanded.  t0027: flow <= 0.75 + 0.3 = 1.05, but demand is only 0.8 and surplus is forbidden, so flow
    settles at exactly 0.8 (400 MW) -- fully served from t0027 on."""
    values = [
        ["unit__outputNode", [PLANT, "west"], "ramp_method", b64("ramp_limit", "str")],
        ["unit__outputNode", [PLANT, "west"], "ramp_speed_up", b64(0.005, "float")],
        ["unit__outputNode", [PLANT, "west"], "ramp_speed_down", b64(0.005, "float")],
        ["unit", PLANT, "startup_method", b64("linear", "str")],
        ["unit", PLANT, "min_load", b64(0.3, "float")],
        ["unit", PLANT, "startup_cost", b64(10.0, "float")],
        ["node", "west", "penalty_up", b64(1000.0, "float")],
        ["node", "west", "penalty_down", b64(1.0e9, "float")],
    ] + _step_demand(0.0, 400.0)
    url, scen = make_db(tmp_path_factory, alt="c3_start_su", values=values, base_chain=["init", "west", "coal"])
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    flow = utf8(step.solution.value("v_flow")).filter(pl.col("p") == PLANT).sort("t")

    def _f(t: str) -> float:
        return flow.filter(pl.col("t") == t)["value"][0]

    assert _f("t0025") == pytest.approx(0.45, abs=1e-6)
    assert _f("t0026") == pytest.approx(0.75, abs=1e-6)
    assert _f("t0027") == pytest.approx(0.8, abs=1e-6)


def test_stopper_floor_and_sd(tmp_path_factory) -> None:
    """C4: coal_plant alone at west, min_load 0.3, ramp 0.005/min (rho-bar = 0.3, SD' = ml + 1/2*ramp_down*60*
    1h = 0.3 + 0.15 = 0.45), startup_cost 10, huge ``penalty_down``; demand 400 MW t0001-t0024 then 0 from
    t0025 (``_step_demand(400.0, 0.0)``: the switch is at t0025, the default).

    Hand derivation: continuing ramp-down alone only allows -0.3/step, so from the steady 0.8 the fastest a
    CONTINUING unit reaches is 0.8 - 0.3 = 0.5 by t0024; a one-step full stop instead needs flow(t0024) <=
    SD' = 0.45 for flow(t0025) = 0 to be feasible at all (maxFlow_online forces flow = 0 once on = 0).  The
    drop from the steady 0.8 to 0.45 is 0.35, more than one step's room (rho-bar = 0.3), so the shortfall is
    NOT confined to one step: since penalty_up (1000) << penalty_down (1e9), the solver sheds the smallest
    possible amount first -- t0023 drops only to 0.75 (375 of the 400 MW demanded served, a 25 MW shortfall),
    exactly as much as leaves -0.3/step of room into t0024 -- then lands flow(t0024) exactly at SD' = 0.45
    (175 of the 400 MW demanded short) and shuts down cleanly at t0025 (flow 0, su/sd row
    ``ramp_sink_down_constraint`` at t0025
    carries +rho-bar*c on v_online (0.3), -(rho-bar*c + F) = -(0.3+0.3) = -0.6 on v_startup (F = min_load 0.3,
    the ONLY arc of this direct unit's output measure), and +c*SD' = +0.45 on v_shutdown -- canonical signs,
    i.e. the negative of each RHS-formula term).  The last online (pre-shutdown) flow is therefore <= SD' =
    0.45, exactly (not merely bounded): a looser pre-shutdown value would make t0025 = 0 infeasible."""
    values = [
        ["unit__outputNode", [PLANT, "west"], "ramp_method", b64("ramp_limit", "str")],
        ["unit__outputNode", [PLANT, "west"], "ramp_speed_up", b64(0.005, "float")],
        ["unit__outputNode", [PLANT, "west"], "ramp_speed_down", b64(0.005, "float")],
        ["unit", PLANT, "startup_method", b64("linear", "str")],
        ["unit", PLANT, "min_load", b64(0.3, "float")],
        ["unit", PLANT, "startup_cost", b64(10.0, "float")],
        ["node", "west", "penalty_up", b64(1000.0, "float")],
        ["node", "west", "penalty_down", b64(1.0e9, "float")],
    ] + _step_demand(400.0, 0.0)
    url, scen = make_db(tmp_path_factory, alt="c4_stopper", values=values, base_chain=["init", "west", "coal"])
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    flow = utf8(step.solution.value("v_flow")).filter(pl.col("p") == PLANT).sort("t")
    last_online = flow.filter(pl.col("t") == "t0024")["value"][0]
    assert last_online == pytest.approx(0.45, abs=1e-6)
    assert flow.filter(pl.col("t") == "t0025")["value"][0] == pytest.approx(0.0, abs=1e-6)

    pb = build(step.flex_data)
    dn = row_terms(pb, "ramp_sink_down_constraint")[f"ramp_sink_down_constraint[{PLANT},coal_market,west,"
                                                     "p2020,t0025]"]
    assert coef(dn, f"v_online_linear[{PLANT}") == pytest.approx(0.3)
    assert coef(dn, f"v_startup_linear[{PLANT}") == pytest.approx(-0.6)
    assert coef(dn, f"v_shutdown_linear[{PLANT}") == pytest.approx(0.45)


def test_floor_by_arc_type(tmp_path_factory) -> None:
    """C5: coefficients of v_shutdown in the UP row (= F, the arc's own per-unit-online floor, design table
    section 1.2) by arc type, in one fixture -- coal_plant (direct unit, min_load 0.25): F = min_load = 0.25
    (its single output arc IS the unit's whole output measure).  coal_chp (indirect, min_load 0.4,
    ``capacity_min_coeff`` 0.5 on its west output): F = min_load*capacity_min_coeff = 0.4*0.5 = 0.2 (its own
    ``minFlow_output_floor`` row).  coal_chp's fuel input arc (coal_market, also ramp-limited): F = 0 -- no
    v_shutdown term in that row at all (input arcs are never the unit's output measure)."""
    values = no_ratio_constraint() + [
        ["unit__outputNode", [PLANT, "west"], "ramp_method", b64("ramp_limit", "str")],
        ["unit__outputNode", [PLANT, "west"], "ramp_speed_up", b64(0.005, "float")],
        ["unit__outputNode", [PLANT, "west"], "ramp_speed_down", b64(0.005, "float")],
        ["unit", PLANT, "startup_method", b64("linear", "str")],
        ["unit", PLANT, "min_load", b64(0.25, "float")],
        ["unit__outputNode", ["coal_chp", "west"], "capacity_min_coeff", b64(0.5, "float")],
        ["unit", "coal_chp", "startup_method", b64("linear", "str")],
        ["unit", "coal_chp", "min_load", b64(0.4, "float")],
        ["unit__outputNode", ["coal_chp", "west"], "ramp_method", b64("ramp_limit", "str")],
        ["unit__outputNode", ["coal_chp", "west"], "ramp_speed_up", b64(0.002, "float")],
        ["unit__outputNode", ["coal_chp", "west"], "ramp_speed_down", b64(0.002, "float")],
        ["unit__inputNode", ["coal_chp", "coal_market"], "ramp_method", b64("ramp_limit", "str")],
        ["unit__inputNode", ["coal_chp", "coal_market"], "ramp_speed_up", b64(0.002, "float")],
        ["unit__inputNode", ["coal_chp", "coal_market"], "ramp_speed_down", b64(0.002, "float")],
        ["node", "west", "inflow", b64(-500.0, "float")],
        ["node", "heat", "inflow", b64(-500.0, "float")],
    ]
    url, scen = make_db(tmp_path_factory, alt="c5_floor", values=values,
                        base_chain=["init", "west", "coal", "coal_chp", "heat"])
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    pb = build(step.flex_data)
    up = row_terms(pb, "ramp_sink_up_constraint")
    src_up = row_terms(pb, "ramp_source_up_constraint")

    plant_row = up[f"ramp_sink_up_constraint[{PLANT},coal_market,west,p2020,t0012]"]
    assert coef(plant_row, f"v_shutdown_linear[{PLANT}") == pytest.approx(0.25)

    chp_row = up["ramp_sink_up_constraint[coal_chp,coal_chp,west,p2020,t0012]"]
    assert coef(chp_row, "v_shutdown_linear[coal_chp") == pytest.approx(0.2)

    fuel_row = src_up["ramp_source_up_constraint[coal_chp,coal_market,coal_chp,p2020,t0012]"]
    assert "v_shutdown" not in "".join(fuel_row)


def test_reserve_only_row_on_online_capacity(tmp_path_factory) -> None:
    """C6: coal_plant online (linear), up reserve provider at west (reservation 40 MW), ramp 0.005/min on its
    output (rho = ramp_speed*60*1h = 0.3 < 1 at the very first step t0001, which has no ramp predecessor --
    design section 3.1's ``prundt`` row).  min_load 0.2.

    Hand derivation: at a step with no predecessor, the row is a pure reserve bound: R^up <= rho*c*cap, cap =
    v_online for an online process (c = 1, no capacity_max_coeff authored).  So the t0001 up row is exactly
    ``v_reserve - 0.3*v_online_linear <= 0`` -- no v_flow term at all (it is not a flow_rows / continuing row)."""
    grp = ["ru", "up", "electricity"]
    values = [
        ["unit__outputNode", [PLANT, "west"], "ramp_method", b64("ramp_limit", "str")],
        ["unit__outputNode", [PLANT, "west"], "ramp_speed_up", b64(0.005, "float")],
        ["unit__outputNode", [PLANT, "west"], "ramp_speed_down", b64(0.005, "float")],
        ["unit", PLANT, "startup_method", b64("linear", "str")],
        ["unit", PLANT, "min_load", b64(0.2, "float")],
        ["reserve__upDown__group", grp, "reserve_method", b64("timeseries_only", "str")],
        ["reserve__upDown__group", grp, "penalty_reserve", b64(5000.0, "float")],
        ["reserve__upDown__group", grp, "reservation", b64(40.0, "float")],
        ["reserve__upDown__unit__node", ["ru", "up", PLANT, "west"], "is_enabled", b64("yes", "str")],
        ["reserve__upDown__unit__node", ["ru", "up", PLANT, "west"], "max_share", b64(1.0, "float")],
        ["reserve__upDown__unit__node", ["ru", "up", PLANT, "west"], "reliability", b64(1.0, "float")],
        ["node", "west", "inflow", b64(-200.0, "float")],
    ]
    entities = [["upDown", "up", None], ["reserve", "ru", None], ["reserve__upDown__group", grp, None],
                ["reserve__upDown__unit__node", ["ru", "up", PLANT, "west"], None]]
    url, scen = make_db(tmp_path_factory, alt="c6_reserve_row", values=values, base_chain=["init", "west", "coal"],
                        entities=entities)
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    pb = build(step.flex_data)
    up = row_terms(pb, "ramp_sink_up_constraint")[f"ramp_sink_up_constraint[{PLANT},coal_market,west,p2020,"
                                                   "t0001]"]
    bounds = row_bounds(pb, "ramp_sink_up_constraint")[f"ramp_sink_up_constraint[{PLANT},coal_market,west,"
                                                        "p2020,t0001]"]
    assert "v_flow" not in "".join(up)
    assert coef(up, f"v_online_linear[{PLANT}") == pytest.approx(-0.3)
    assert coef(up, "v_reserve[") == pytest.approx(1.0)
    assert bounds[1] == pytest.approx(0.0)


def test_carry_row_survives_unforeseen_shutdown(tmp_path_factory) -> None:
    """C7 (guard; rolling horizon = jump): coal_plant alone at west, min_load unauthored (0), ramp 0.005/min;
    demand 500 MW (= existing, full output) for t0001-t0024, then 0; availability forced to 0 ONLY at t0025.
    Two 24h rolls (``rolling_solve_horizon`` = ``rolling_solve_jump`` = 24): roll 1 ends with the unit
    committed at v_online = 1, flow = 1 (full, f-hat = c*c-hat exactly); roll 2's first step (t0025) is
    unforeseen by roll 1 (its own availability = 0 forces it fully offline there).

    Hand derivation (critique item 3): on the carry row, SD'_carry = max(1, f-hat/(c*c-hat)) = max(1, 1.0 /
    (1*1.0)) = 1 exactly (the carried flow was already at the carried online count's full capacity), so the
    down row's v_shutdown coefficient is -c*SD'_carry = -1 (canonical sign: +1, the negative of the RHS-
    formula value) -- NOT the ml-based SD' of an interior link row.  The up row's carry-row shutdown term is
    F = 0 at the (unknown) carried predecessor step, so it carries NO v_shutdown column at all.  Both rolls
    must still solve to Optimal (the carry row remains feasible for any history, design section 3.3)."""
    inflow = {f"t{i:04d}": (-500.0 if i <= 24 else 0.0) for i in range(1, 73)}
    avail = {f"t{i:04d}": (0.0 if i == 25 else 1.0) for i in range(1, 73)}
    values = [
        ["unit__outputNode", [PLANT, "west"], "ramp_method", b64("ramp_limit", "str")],
        ["unit__outputNode", [PLANT, "west"], "ramp_speed_up", b64(0.005, "float")],
        ["unit__outputNode", [PLANT, "west"], "ramp_speed_down", b64(0.005, "float")],
        ["unit", PLANT, "startup_method", b64("linear", "str")],
        ["unit", PLANT, "availability", b64(_pmap(avail, "time"), "map")],
        ["node", "west", "inflow", b64(_pmap(inflow, "time"), "map")],
        ["solve", "y2020_2day_dispatch", "solve_mode", b64("rolling_window", "str")],
        ["solve", "y2020_2day_dispatch", "rolling_solve_horizon", b64(24.0, "float")],
        ["solve", "y2020_2day_dispatch", "rolling_solve_jump", b64(24.0, "float")],
    ]
    url, scen = make_db(tmp_path_factory, alt="c7_carry", values=values, base_chain=["init", "west", "coal"])
    steps = run(tmp_path_factory, url, scen)
    assert len(steps) == 2
    for name, step in steps.items():
        assert step.solution is not None, f"{name} did not solve"

    roll2 = list(steps.values())[-1]
    pb = build(roll2.flex_data)
    dn = row_terms(pb, "ramp_sink_down_constraint")[f"ramp_sink_down_constraint[{PLANT},coal_market,west,"
                                                     "p2020,t0025]"]
    up = row_terms(pb, "ramp_sink_up_constraint")[f"ramp_sink_up_constraint[{PLANT},coal_market,west,p2020,"
                                                   "t0025]"]
    assert coef(dn, f"v_shutdown_linear[{PLANT}") == pytest.approx(1.0)
    assert "v_shutdown" not in "".join(up)


def test_min_load_constant_vs_map_same_ramp_rows(tmp_path_factory) -> None:
    """C8 (CLAUDE.md invariant #2): ``min_load`` as a constant, an equivalent uniform time map and an
    equivalent uniform period map must give byte-identical ``ramp_*`` / ``online_change_linear`` rows --
    checked in a single solve, a 2-roll rolling solve and the 2-day stochastic dispatch
    (``tests/fixtures/stochastics.json``, coal_plant's existing output arc given its own ramp + online
    overlay).  A name-based broadcaster would silently cross-join the map's silent-default ``"x"`` row against
    the whole timeline (CLAUDE.md invariant #2); this test is the constant-vs-equivalent-map regression gate
    for the ramp row's ``ml`` (design section 3.2 notes it must go through ``promote_param_to_dt``, same as
    ``_min_load_share`` already does).  Guard, like C7: before this commit the ramp row never read
    ``min_load`` at all (a flat +-1 ``v_startup``/``v_shutdown``), so the shapes were trivially identical
    then too; this test guards the NEW ``ml`` usage against the exact class of bug CLAUDE.md invariant #2
    warns about, it does not fail on the parent commit."""

    def _row_shapes(pb) -> dict:
        return {fam: {nm: tuple(sorted(terms.items())) for nm, terms in row_terms(pb, fam).items()}
                for fam in ("ramp_sink_up_constraint", "ramp_sink_down_constraint", "online_change_linear")}

    def _ramp_values(const: float | None = None, tmap: dict | None = None, pmap_: dict | None = None,
                     sink: str = "west", unit: str = PLANT) -> list:
        vals = [
            ["unit__outputNode", [unit, sink], "ramp_method", b64("ramp_limit", "str")],
            ["unit__outputNode", [unit, sink], "ramp_speed_up", b64(0.005, "float")],
            ["unit__outputNode", [unit, sink], "ramp_speed_down", b64(0.005, "float")],
            ["unit", unit, "startup_method", b64("linear", "str")],
        ]
        if const is not None:
            vals.append(["unit", unit, "min_load", b64(const, "float")])
        if tmap is not None:
            vals.append(["unit", unit, "min_load", b64(_pmap(tmap, "time"), "map")])
        if pmap_ is not None:
            vals.append(["unit", unit, "min_load", b64(_pmap(pmap_, "period"), "map")])
        return vals

    # Single solve.
    shapes = {}
    for kind, extra in (("const", {"const": 0.2}), ("time", {"tmap": {f"t{i:04d}": 0.2 for i in range(1, 73)}}),
                        ("period", {"pmap_": {"p2020": 0.2}})):
        values = _ramp_values(**extra) + [["node", "west", "inflow", b64(-200.0, "float")]]
        url, scen = make_db(tmp_path_factory, alt=f"c8_single_{kind}", values=values,
                            base_chain=["init", "west", "coal"])
        step = list(run(tmp_path_factory, url, scen).values())[-1]
        shapes[kind] = _row_shapes(build(step.flex_data))
    assert shapes["const"] == shapes["time"] == shapes["period"]

    # 2-roll rolling solve.
    roll_shapes = {}
    for kind, extra in (("const", {"const": 0.2}), ("time", {"tmap": {f"t{i:04d}": 0.2 for i in range(1, 73)}})):
        values = _ramp_values(**extra) + [
            ["node", "west", "inflow", b64(-200.0, "float")],
            ["solve", "y2020_2day_dispatch", "solve_mode", b64("rolling_window", "str")],
            ["solve", "y2020_2day_dispatch", "rolling_solve_horizon", b64(36.0, "float")],
            ["solve", "y2020_2day_dispatch", "rolling_solve_jump", b64(24.0, "float")]]
        url, scen = make_db(tmp_path_factory, alt=f"c8_roll_{kind}", values=values,
                            base_chain=["init", "west", "coal"])
        steps = run(tmp_path_factory, url, scen)
        roll_shapes[kind] = {name: _row_shapes(build(s.flex_data)) for name, s in steps.items()}
    assert roll_shapes["const"] == roll_shapes["time"]

    # Stochastic 2-day dispatch.
    stoch_shapes = {}
    for kind, extra in (("const", {"const": 0.2, "sink": "demand_node"}),
                        ("time", {"tmap": {f"t{i:04d}": 0.2 for i in range(1, 49)}, "sink": "demand_node"})):
        url, scen = make_db(tmp_path_factory, alt=f"c8_stoch_{kind}", values=_ramp_values(**extra),
                            base_chain=["base", "init", "system", "2day"],
                            fixture_json=FLEXTOOL_ROOT / "tests" / "fixtures" / "stochastics.json")
        step = list(run(tmp_path_factory, url, scen).values())[-1]
        stoch_shapes[kind] = _row_shapes(build(step.flex_data))
    assert stoch_shapes["const"] == stoch_shapes["time"]


def test_branch_point_shutdown_share_counted_once(tmp_path_factory) -> None:
    """At a branch point (the mid-horizon fixture: trunk p2035 = t0001–t0003 continues into the branches p2040_high
    and p2040_low at t0004) each branch's down-ramp row reads its own shut-down with c·SD′ of the trunk's last step
    once.  base: linear online, min_load 0.4, ramp speed 1e-3 / min on base -> elec, c = 1, 1-h steps:
    SD′[p2035, t0003] = min(1, 0.4 + ½·1e-3·60·1) = 0.43.  Before: the trunk step was looked up once per branch and
    the share was multiplied by the repeats (1.72 with two branches)."""
    rel = ["base", "elec"]
    vals = [["unit", "base", "startup_method", b64("linear", "str")], ["unit", "base", "min_load", b64(0.4, "float")],
            *_ramp(1e-3, rel)]
    url, scen = make_db(tmp_path_factory, alt="ramp_branch", values=vals, base_chain=["init", "base", "stoch"],
                        fixture_json=FLEXTOOL_ROOT / "tests" / "fixtures" / "stoch_two_period_midhorizon_nreal0.json")
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    rows = {k: v for k, v in row_terms(build(step.flex_data), "ramp_sink_down_constraint", "base,base,elec,p2040_").items()
            if k.endswith(",t0004]")}
    assert sorted(rows) == [f"ramp_sink_down_constraint[base,base,elec,p2040_{b},t0004]" for b in ("high", "low")]
    for name, terms in rows.items():
        branch = name.split(",")[3]
        assert coef(terms, f"v_shutdown_linear[base,{branch},t0004]") == pytest.approx(0.43, abs=TOL), name
        assert coef(terms, "v_flow[base,base,elec,p2035,t0003]") == pytest.approx(-1.0, abs=TOL)
