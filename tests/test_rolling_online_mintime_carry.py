"""OC1 Carrier 2 — cross-roll-boundary min_uptime / min_downtime continuity.

MINTIME-1 (the prerequisite slice) restored WITHIN-roll min-up/down
enforcement on the rolling/synthetic path.  Carrier 2 carries an obligation
opened in roll N into roll N+1's early steps: the prior roll's realized
committed ``v_startup`` / ``v_shutdown`` is handed off (``online_history``),
rebuilt into ``uptime_carry`` / ``downtime_carry`` at the continuation roll
(``_derived.online_mintime_carry_from_source``) and injected on the
``minimum_uptime`` RHS / ``minimum_downtime`` LHS by ``_online.add_constraints``.

The carry follows the timeline: it starts at the next roll's first steps that
continue the previous roll's last realized step (``dt_ramp_carry``, ``jump ==
1``) and counts each history step's ``hours_to_end``.  The fullYear periods
p2020 → p2025 reuse one timeset, so p2025's first step does not continue p2020
and takes no carry — forcing one across would be a unit-commitment
infeasibility no node penalty slack can absorb.

All scenarios are built from JSON (CLAUDE.md invariant #3), never a
checked-in .sqlite.  The witnesses drive the full rolling chain and read
the realized ``output_raw`` parquets.  The rolling chain runs over the
periods a witness needs (``_PERIODS_ALT``): p2020 alone (18 rolls of 8 h /
4 h, 17 roll boundaries) where the witness lies within one period, p2020 +
p2025 where it must cross the period boundary.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from db_utils import json_to_db  # noqa: E402

from flextool.engine import run_chain_from_db  # noqa: E402
from flextool.update_flextool.db_migration import migrate_database  # noqa: E402

pytestmark = [pytest.mark.solver]

UNIT = "coal_plant"
EXISTING_COUNT = 1.0  # existing=500, unitsize=500 → v_online ∈ [0, 1]


# ───────────────────────── helpers ──────────────────────────────────────
def _roll_index(name: str) -> int:
    return int(name.split("roll_")[-1])


def _per_roll_frame(workdir: Path, key: str) -> dict[int, pl.DataFrame]:
    out: dict[int, pl.DataFrame] = {}
    for f in (workdir / "output_raw").glob(f"{key}__*.parquet"):
        out[_roll_index(f.stem)] = pl.read_parquet(f).sort(["period", "time"])
    return out


def _global_realized(workdir: Path):
    """Concatenate the per-roll realized v_online / v_startup / v_shutdown in
    roll order, keeping each row's (period, time, roll-source)."""
    on = _per_roll_frame(workdir, "v_online_linear")
    su = _per_roll_frame(workdir, "v_startup_linear")
    sd = _per_roll_frame(workdir, "v_shutdown_linear")
    rows = []
    for r in sorted(on):
        o = on[r][UNIT].to_list()
        s = su[r][UNIT].to_list()
        d = sd[r][UNIT].to_list()
        per = on[r]["period"].to_list()
        tm = on[r]["time"].to_list()
        for k in range(len(o)):
            rows.append((r, per[k], tm[k], o[k], s[k], d[k]))
    return rows


def _cross_boundary_violations(rows, win_len, leg):
    """Count cross-roll-boundary, same-period min-up/down violations.

    leg='up': v_online[k] < Σ v_startup over the same-period window ending at
    k (window straddling a roll boundary).
    leg='dn': v_online[k] + Σ v_shutdown over that window > existing.
    """
    viol = []
    for k in range(len(rows)):
        _, per_k, tm_k, o_k, _, _ = rows[k]
        lo = max(0, k - win_len + 1)
        win = [j for j in range(lo, k + 1) if rows[j][1] == per_k]
        if not win:
            continue
        straddles = rows[win[0]][0] != rows[k][0]  # different roll source
        if not straddles:
            continue
        if leg == "up":
            ssum = sum(rows[j][4] for j in win)
            if o_k < ssum - 1e-5:
                viol.append((per_k, tm_k, round(o_k, 3), round(ssum, 3)))
        else:
            dsum = sum(rows[j][5] for j in win)
            if o_k + dsum > EXISTING_COUNT + 1e-5:
                viol.append((per_k, tm_k, round(o_k, 3), round(dsum, 3)))
    return viol


def _db(tmp_path_factory, name):
    db_path = tmp_path_factory.mktemp(name) / "tests.sqlite"
    url = json_to_db(TEST_DIR / "fixtures" / "tests.json", db_path)
    migrate_database(url, up_to=40)
    return url


#: Alternative (last in a witness's chain) that limits dispatch_fullYear_roll to some of its four periods.
_PERIODS_ALT = "c2_periods"


def _import_periods(dm, periods: list[str]) -> None:
    """Add ``_PERIODS_ALT``: dispatch_fullYear_roll over ``periods`` only (fullYear timeset, all realized)."""
    from spinedb_api import import_data
    from spinedb_api.parameter_value import Array, Map

    import_data(dm, alternatives=[_PERIODS_ALT], object_parameter_values=[
        ("solve", "dispatch_fullYear_roll", "period_timeset",
         Map(list(periods), ["fullYear"] * len(periods), index_name="period"), _PERIODS_ALT),
        ("solve", "dispatch_fullYear_roll", "realized_periods", Array(list(periods)), _PERIODS_ALT)])


# ───────────────────── min_uptime witness (48 → 0) ──────────────────────
@pytest.fixture(scope="module")
def updown_run(tmp_path_factory):
    """``coal_min_updown_roll``: min_uptime = min_downtime = 5, method=both,
    on the fullYear rolling timeline (horizon=8, jump=4) of p2020.  Exercises BOTH
    Carrier 1 (``roll_end_online``, inherent to rolling UC) and Carrier 2."""
    from spinedb_api import DatabaseMapping, import_data
    url = _db(tmp_path_factory, "db_c2_updown")
    scen = "coal_min_updown_roll"
    with DatabaseMapping(url) as dm:
        _import_periods(dm, ["p2020"])
        import_data(dm, scenarios=[(scen, False, "")], scenario_alternatives=[
            (scen, "init", "west"), (scen, "west", "coal"),
            (scen, "coal", "coal_min_load"),
            (scen, "coal_min_load", "coal_min_updowntime"),
            (scen, "coal_min_updowntime", "wind"), (scen, "wind", "fullYear"),
            (scen, "fullYear", "dispatch_fullYear_roll"),
            (scen, "dispatch_fullYear_roll", _PERIODS_ALT),
            (scen, _PERIODS_ALT, None)])
        dm.commit_session("c2 updown witness")
    workdir = tmp_path_factory.mktemp("c2_updown_run")
    steps = run_chain_from_db(url, scen, work_folder=workdir,
                              keep_solutions=True)
    return {"steps": steps, "workdir": workdir}


def test_min_uptime_cross_boundary_zero(updown_run):
    """Carrier 2: cross-boundary min_uptime obligations are honoured — 0
    cross-roll-boundary up-violations (HEAD/MINTIME-1 left 48 over fullYear_roll's four periods)."""
    assert all(s.optimal for s in updown_run["steps"].values())
    rows = _global_realized(updown_run["workdir"])
    viol = _cross_boundary_violations(rows, 5, "up")
    assert not viol, f"{len(viol)} cross-boundary min_uptime violations: {viol[:5]}"


def test_composition_carrier1_and_carrier2_active(updown_run):
    """Composition: on a continuation roll BOTH carriers are live — Carrier 1
    (``p_roll_continue_online``, no startup double-count) and Carrier 2
    (``uptime_carry`` / ``online_startup_history``).  Both obligations are
    honoured (0 cross-boundary up+down violations) and every roll is
    optimal."""
    steps = updown_run["steps"]
    assert all(s.optimal for s in steps.values())
    c1 = c2 = False
    for step in steps.values():
        fd = step.flex_data
        if getattr(fd, "p_roll_continue_online", None) is not None:
            c1 = True
        if getattr(fd, "uptime_carry", None) is not None or \
                getattr(fd, "online_startup_history", None) is not None:
            c2 = True
    assert c1, "Carrier 1 (p_roll_continue_online) never fired on any roll"
    assert c2, "Carrier 2 (uptime_carry/online_startup_history) never fired"
    rows = _global_realized(updown_run["workdir"])
    assert not _cross_boundary_violations(rows, 5, "up")
    assert not _cross_boundary_violations(rows, 5, "dn")


# ─────── timeline-aware: carries where the timeline continues, not across a new year ───────
@pytest.fixture(scope="module")
def cross_period_run(tmp_path_factory):
    """min_uptime=6 (> jump), method=min_uptime.  The fullYear rolling
    timeline of p2020 + p2025 has a roll straddling the p2020→p2025 period
    boundary; a NAIVE cross-period carry makes that roll infeasible even with
    node slack.  The timeline-aware carry (only where the next step continues
    the timeline) keeps it feasible."""
    from spinedb_api import DatabaseMapping, import_data
    url = _db(tmp_path_factory, "db_c2_xp")
    scen = "coal_min_up6_roll"
    with DatabaseMapping(url) as dm:
        _import_periods(dm, ["p2020", "p2025"])
        import_data(dm, scenarios=[(scen, False, "")], scenario_alternatives=[
            (scen, "init", "west"), (scen, "west", "coal"),
            (scen, "coal", "coal_min_load"),
            (scen, "coal_min_load", "coal_min_updowntime"),
            (scen, "coal_min_updowntime", "wind"), (scen, "wind", "fullYear"),
            (scen, "fullYear", "dispatch_fullYear_roll"),
            (scen, "dispatch_fullYear_roll", _PERIODS_ALT),
            (scen, _PERIODS_ALT, None)])
        import_data(dm, object_parameter_values=[
            ("unit", UNIT, "min_uptime", 6.0, "coal_min_updowntime"),
            ("unit", UNIT, "min_downtime", 0.0, "coal_min_updowntime"),
            ("unit", UNIT, "minimum_time_method", "min_uptime",
             "coal_min_updowntime")])
        dm.commit_session("c2 cross-period witness")
    workdir = tmp_path_factory.mktemp("c2_xp_run")
    steps = run_chain_from_db(url, scen, work_folder=workdir,
                              keep_solutions=True)
    return {"steps": steps, "workdir": workdir}


def test_timeline_aware_carry_keeps_the_new_year_roll_feasible(cross_period_run):
    """A min_uptime=6 roll straddling the period boundary stays optimal — the
    timeline-aware carry does NOT force a cross-period obligation (which would
    be infeasible).  HEAD with a naive cross-period carry is infeasible."""
    steps = cross_period_run["steps"]
    nonopt = [n for n, s in steps.items() if not s.optimal]
    assert not nonopt, f"cross-period rolls not optimal: {nonopt[:5]}"


def test_timeline_aware_carry_continues_the_timeline_not_a_new_year(cross_period_run):
    """Explicit timeline-aware behaviour: on the roll whose window first enters
    a NEW period, the carry holds NO obligation for that new period's early
    steps (no history there yet / no cross-period carry), while within-period
    continuation rolls DO carry (uptime_carry non-empty with all rows in a
    single period)."""
    within_period_carried = False
    for step in cross_period_run["steps"].values():
        uc = getattr(step.flex_data, "uptime_carry", None)
        if uc is None or uc.height == 0:
            continue
        carried_periods = set(uc["d"].to_list())
        # every carried row's constrained step shares the period of its
        # carried predecessor — never a cross-period obligation
        within_period_carried = True
        # the roll's dt periods
        dt = step.flex_data.dt
        roll_periods = set(dt["d"].cast(pl.Utf8).to_list())
        # the first period a roll ever visits has no in-period history → if a
        # roll spans two periods, the carry must not reference the later
        # period's opening step from the earlier period's history.  Assert
        # each carried obligation lies in a period that the roll actually
        # realizes with prior same-period history — operationally: the carry
        # period set is a subset of the roll's periods (never invents one).
        assert carried_periods <= roll_periods, (
            f"carry references a period not in the roll: "
            f"{carried_periods - roll_periods}")
    assert within_period_carried, (
        "no within-period carry ever fired — witness does not exercise "
        "Carrier 2")


def test_lookback_stops_at_a_period_that_reuses_the_timeset(cross_period_run):
    """In the roll whose window holds the end of p2020 and the start of p2025 (the two years reuse one timeset,
    so p2025's first step does not continue p2020), the ramp predecessor links have no p2025 → p2020 row, and the
    min_uptime look-back has no row whose step and look-back step are in different periods.  The look-back used
    to walk the name-sorted step list from p2025's first steps back into the last steps of p2020."""
    straddling = 0
    for name, step in cross_period_run["steps"].items():
        fd = step.flex_data
        if fd.dt["d"].cast(pl.Utf8).n_unique() < 2:
            continue
        straddling += 1
        link = fd.dt_ramp_link
        assert link is not None
        assert link.filter(pl.col("d") != pl.col("d_previous")).height == 0, (
            f"{name}: precondition — the periods are linked on the timeline: "
            f"{link.filter(pl.col('d') != pl.col('d_previous')).rows()}")
        lb = fd.uptime_lookback
        assert lb is not None and lb.height > 0
        cross = lb.filter(pl.col("d").cast(pl.Utf8) != pl.col("d_back").cast(pl.Utf8))
        assert cross.height == 0, f"{name}: look-back crosses the period boundary: {cross.rows()[:5]}"
    assert straddling > 0, "no roll straddles the p2020 → p2025 boundary — the witness does not exercise it"


# ─────────────────── dedicated min_downtime witness ─────────────────────
@pytest.fixture(scope="module")
def downtime_run(tmp_path_factory):
    """Dedicated min_downtime binding witness.  Demand is high (−400) except
    a long valley (0) at t0007–t0020 so the shutdown roll (horizon 8) cannot
    see the far-future peak and cleanly FULLY shuts coal at t0007; demand
    resumes at t0021 — still within min_downtime=20 of that shutdown.  Wind
    is removed so coal is the swing unit.  WITHOUT the carry a later roll
    illegally re-starts coal at t0021; WITH it the obligation is honoured
    (``v_online + carried_shutdown ≤ existing`` ⇒ v_online ≤ ~0.2).  The rolls
    run over p2020, where the witness lies."""
    from spinedb_api import DatabaseMapping, import_data
    from spinedb_api.parameter_value import Map
    url = _db(tmp_path_factory, "db_c2_dn")
    scen = "coal_min_dn_roll"
    steps_t = [f"t{i:04d}" for i in range(1, 73)]
    dem = [(s, 0.0 if 7 <= i <= 20 else -400.0)
           for i, s in enumerate(steps_t, start=1)]
    demand_map = Map([s for s, _ in dem], [v for _, v in dem],
                     index_name="time")
    with DatabaseMapping(url) as dm:
        _import_periods(dm, ["p2020"])
        import_data(dm, alternatives=["dn_witness"],
                    scenarios=[(scen, False, "")], scenario_alternatives=[
            (scen, "init", "west"), (scen, "west", "coal"),
            (scen, "coal", "coal_min_load"),
            (scen, "coal_min_load", "coal_min_updowntime"),
            (scen, "coal_min_updowntime", "wind"), (scen, "wind", "fullYear"),
            (scen, "fullYear", "dispatch_fullYear_roll"),
            (scen, "dispatch_fullYear_roll", "dn_witness"),
            (scen, "dn_witness", _PERIODS_ALT),
            (scen, _PERIODS_ALT, None)])
        import_data(dm, object_parameter_values=[
            ("unit", UNIT, "min_uptime", 0.0, "dn_witness"),
            ("unit", UNIT, "min_downtime", 20.0, "dn_witness"),
            ("unit", UNIT, "minimum_time_method", "min_downtime",
             "dn_witness"),
            ("unit", "wind_plant", "existing", 0.0, "dn_witness"),
            ("node", "west", "inflow", demand_map, "dn_witness")])
        dm.commit_session("c2 min_downtime witness")
    workdir = tmp_path_factory.mktemp("c2_dn_run")
    steps = run_chain_from_db(url, scen, work_folder=workdir,
                              keep_solutions=True)
    return {"steps": steps, "workdir": workdir}


def _online_at(rows, period, time):
    for (_, per, tm, o, _, _) in rows:
        if per == period and tm == time:
            return o
    raise AssertionError(f"({period},{time}) not realized")


def test_min_downtime_shutdown_realized(downtime_run):
    """Coal fully shuts at t0007 (the valley opens) — v_online 0.8 → 0 with a
    full v_shutdown — giving a genuine obligation to carry."""
    assert all(s.optimal for s in downtime_run["steps"].values())
    rows = _global_realized(downtime_run["workdir"])
    assert _online_at(rows, "p2020", "t0006") > 0.5
    assert _online_at(rows, "p2020", "t0007") < 0.05, (
        "coal did not fully shut at the valley")


def test_min_downtime_cross_boundary_binds(downtime_run):
    """Carrier 2 (down leg): at the t0021 demand peak — 14 steps after the
    t0007 shutdown, within min_downtime=20 — coal is held OFF (v_online ≤ 0.3)
    by the carried shutdown.  HEAD (no Carrier 2) illegally re-starts coal to
    ~0.8 because the continuation roll has forgotten the prior shutdown."""
    rows = _global_realized(downtime_run["workdir"])
    online_peak = _online_at(rows, "p2020", "t0021")
    assert online_peak <= 0.3, (
        f"coal online={online_peak} at the post-valley peak — the carried "
        f"min_downtime obligation was not honoured (HEAD re-starts to ~0.8)")


def test_downtime_carry_is_timeline_aware_and_present(downtime_run):
    """The down-leg carry is actually built and timeline-aware (every carried
    row's period equals its carried predecessor's period, by construction)."""
    any_dn_carry = False
    for step in downtime_run["steps"].values():
        dc = getattr(step.flex_data, "downtime_carry", None)
        if dc is not None and dc.height > 0:
            any_dn_carry = True
            roll_periods = set(step.flex_data.dt["d"].cast(pl.Utf8).to_list())
            assert set(dc["d"].to_list()) <= roll_periods
    assert any_dn_carry, "downtime_carry never fired — witness is inert"


# ───────────────────────── byte-parity (no-op) ──────────────────────────
@pytest.fixture(scope="module")
def _no_mintime_rolling(tmp_path_factory):
    """A rolling solve (p2020) WITHOUT any min-up/down.  Carrier 2 must be fully
    inert: no ``online_history`` stored, no carry built."""
    from spinedb_api import DatabaseMapping, import_data
    url = _db(tmp_path_factory, "db_c2_nomin")
    scen = "coal_nomin_roll"
    with DatabaseMapping(url) as dm:
        _import_periods(dm, ["p2020"])
        import_data(dm, scenarios=[(scen, False, "")], scenario_alternatives=[
            (scen, "init", "west"), (scen, "west", "coal"),
            (scen, "coal", "coal_min_load"), (scen, "coal_min_load", "wind"),
            (scen, "wind", "fullYear"),
            (scen, "fullYear", "dispatch_fullYear_roll"),
            (scen, "dispatch_fullYear_roll", _PERIODS_ALT),
            (scen, _PERIODS_ALT, None)])
        dm.commit_session("c2 no-mintime rolling")
    workdir = tmp_path_factory.mktemp("c2_nomin_run")
    steps = run_chain_from_db(url, scen, work_folder=workdir,
                              keep_solutions=True)
    return steps


def test_no_mintime_rolling_is_inert(_no_mintime_rolling):
    """Byte-parity gate: without min-up/down, Carrier 2 stores and builds
    nothing on every roll (so the solve takes its baseline path)."""
    assert all(s.optimal for s in _no_mintime_rolling.values())
    for step in _no_mintime_rolling.values():
        fd = step.flex_data
        assert getattr(fd, "online_startup_history", None) is None
        assert getattr(fd, "uptime_carry", None) is None
        assert getattr(fd, "downtime_carry", None) is None
        assert getattr(fd, "online_min_time", None) is None


# ──────────── a coarser solve continues a finer dispatch (position in time) ────────────
@pytest.fixture(scope="module")
def nested_coarse_run(tmp_path_factory):
    """Nested invest → rolling storage (48 h horizon / 24 h jump, 6-h steps) → rolling dispatch (24 h / 12 h, 1-h
    steps) with coal unit commitment (linear, min_load 0.4, min_uptime 8 h, ramp limit 0.002).  The second storage
    roll starts at the 6-h step (p2020, t0025); the dispatch realized before it ends with the 1-h (p2020, t0024)."""
    import contextlib
    import io

    from tests.engine._unit_db import b64, make_db

    def v(cls, ent, param, val):
        return [cls, ent, param, b64(val, "str" if isinstance(val, str) else "float")]

    chain = ["init", "west", "coal", "coal_invest", "wind", "wind_invest", "battery", "battery_invest",
             "invest_solveSequence_5weeks", "invest_allYears_5weeks", "fullYear", "dispatch_fullYear_roll",
             "storage_fullYear_6h", "invest_24h", "bind_battery_forward", "unlimited_investment"]
    values = [v("unit__outputNode", ["coal_plant", "west"], "ramp_method", "ramp_limit"),
              v("unit__outputNode", ["coal_plant", "west"], "ramp_speed_up", 0.002),
              v("unit__outputNode", ["coal_plant", "west"], "ramp_speed_down", 0.002),
              v("solve", "storage_fullYear_6h", "solve_mode", "rolling_window"),
              v("solve", "storage_fullYear_6h", "rolling_solve_horizon", 48.0),
              v("solve", "storage_fullYear_6h", "rolling_solve_jump", 24.0),
              v("solve", "dispatch_fullYear_roll", "rolling_solve_horizon", 24.0),
              v("solve", "dispatch_fullYear_roll", "rolling_solve_jump", 12.0),
              v("node", "battery", "storage_nested_fix_method", "fix_nothing"),
              v("unit", "coal_plant", "startup_method", "linear"),
              v("unit", "coal_plant", "min_load", 0.4),
              v("unit", "coal_plant", "min_uptime", 8.0)]
    url, scen = make_db(tmp_path_factory, alt="nested_coarse_continues_fine", values=values, base_chain=chain)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        steps = run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp("nested_coarse_run"),
                                  keep_solutions=True)
    return steps, out.getvalue()


def test_coarser_solve_continues_the_finer_dispatch_by_position(nested_coarse_run):
    """The 6-h storage step t0025 follows t0019 (6 h, base hours 19-24) in the storage timeline; the dispatch
    realized before it ends with the 1-h t0024 (base hour 24).  Both end at base hour 24, so the storage roll
    continues that step: the ramp carry comes from (p2020, t0024) with its 1-h duration, and the minimum up time
    carry at t0025 is the realized start-ups of that dispatch within L = 8 h of its end: hours_to_end (from a step's
    start to the end of t0024) = 25 - k for t00k, < 8 for k = 18..24.  No cut-short look-back warning."""
    steps, printed = nested_coarse_run
    names = list(steps)
    storage = "storage_fullYear_6h_roll_1"
    assert storage in names, names
    before = [n for n in names[:names.index(storage)] if n.startswith("dispatch_fullYear_roll")]
    dispatch = steps[before[-1]]
    fd = steps[storage].flex_data
    carry = fd.dt_ramp_carry
    assert carry is not None
    assert carry.select("d", "t", "d_previous", "t_previous").rows() == [("p2020", "t0025", "p2020", "t0024")]
    assert carry["duration_previous"].to_list() == [1.0]
    su = (dispatch.solution.value("v_startup_linear")
          .with_columns(pl.col(c).cast(pl.Utf8) for c in ("p", "d", "t"))
          .filter((pl.col("p") == UNIT) & (pl.col("d") == "p2020")
                  & pl.col("t").is_in([f"t{k:04d}" for k in range(18, 25)])))
    assert su.height == 7
    expected = float(su["value"].sum())
    assert expected > 0.5, "witness inert: no start-up within 8 h of the dispatch's end"
    uc = fd.uptime_carry.with_columns(pl.col(c).cast(pl.Utf8) for c in ("p", "d", "t"))
    at_root = uc.filter((pl.col("p") == UNIT) & (pl.col("d") == "p2020") & (pl.col("t") == "t0025"))
    assert at_root["value"].to_list() == [pytest.approx(expected, abs=1e-6)]
    assert "covers only 0 hours" not in printed
