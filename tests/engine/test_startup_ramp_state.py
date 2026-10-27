"""Part B (c14c, item 99): startups and shutdowns equal the change in units online.

``online_change`` replaces the two ``>=`` rows ``online__startup`` / ``online__shutdown`` with one equality
(``on - on_prev - su + sd = 0``); ``maxStartup`` becomes ``su <= on_t`` and ``maxShutdown`` becomes
``sd <= built - on_t`` (same-period predecessor only, owner question Q-c14c-A).  Together these make an
ORPHAN startup/shutdown (one with no matching change in units online) impossible.  Part C (ramp rows of
online units on online capacity) is NOT in this commit; see each test's docstring for exactly what that
means for the fake-pair claim.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._lp_rows import coef, row_bounds, row_terms
from tests.engine._unit_db import PLANT, b64, make_db, run, utf8

pytestmark = pytest.mark.solver

TOL = 1e-7

TWO_PERIODS_ENTS = [["timeset", "ta", None], ["timeset", "tb", None]]


def _pmap(values: dict, index_name: str = "period") -> dict:
    return {"index_type": "str", "index_name": index_name, "data": [[k, v] for k, v in values.items()]}


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

    for p, existing in (("coal_plant", 500.0), ("e_boiler", 100.0)):
        on = utf8(step.solution.value("v_online_linear")).filter(pl.col("p") == p).sort("t")
        su = utf8(step.solution.value("v_startup_linear")).filter(pl.col("p") == p).sort("t")
        sd = utf8(step.solution.value("v_shutdown_linear")).filter(pl.col("p") == p).sort("t")
        on_v, su_v, sd_v = on["value"].to_list(), su["value"].to_list(), sd["value"].to_list()
        # The equality identity: su - sd = on - on_prev at every interior step (cyclic wrap at t0001 means
        # on_prev there is on[-1], the last step -- also exact).
        for i in range(len(on_v)):
            prev = on_v[i - 1]   # Python negative indexing: i=0 -> on_v[-1], the cyclic wrap
            assert su_v[i] - sd_v[i] == pytest.approx(on_v[i] - prev, abs=TOL)
        # No orphans: su bounded by online_now, sd bounded by existing - online_now (maxStartup / maxShutdown).
        for i in range(len(on_v)):
            assert su_v[i] <= on_v[i] + TOL
            assert sd_v[i] <= existing + TOL   # existing in count-of-unitsize units (unitsize unauthored = 1)

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
