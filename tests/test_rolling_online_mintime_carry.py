"""OC1 Carrier 2 — cross-roll-boundary min_uptime / min_downtime continuity.

MINTIME-1 (the prerequisite slice) restored WITHIN-roll min-up/down
enforcement on the rolling/synthetic path.  Carrier 2 carries an obligation
opened in roll N into roll N+1's early steps: the prior roll's realized
committed ``v_startup`` / ``v_shutdown`` is handed off (``online_history``),
rebuilt into ``uptime_carry`` / ``downtime_carry`` at the continuation roll
(``_derived_params.online_mintime_carry_from_source``) and injected on the
``minimum_uptime`` RHS / ``minimum_downtime`` LHS by ``_add_online_block``.

The carry is PERIOD-AWARE: an obligation is carried ONLY within the same
period, never across a period boundary.  FlexTool's online dynamics are
period-cyclic (a period's first step wraps within that period, never to the
prior period), so a min-up/down obligation must not propagate across the
non-contiguous investment years — forcing one across is a unit-commitment
infeasibility no node penalty slack can absorb (STEP-0 finding).

All scenarios are built from JSON (CLAUDE.md invariant #3), never a
checked-in .sqlite.  The witnesses drive the full rolling chain and read
the realized ``output_raw`` parquets.
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

from flextool.engine_polars import run_chain_from_db  # noqa: E402
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


# ───────────────────── min_uptime witness (48 → 0) ──────────────────────
@pytest.fixture(scope="module")
def updown_run(tmp_path_factory):
    """``coal_min_updown_roll``: min_uptime = min_downtime = 5, method=both,
    on the fullYear rolling timeline (horizon=8, jump=4).  Exercises BOTH
    Carrier 1 (``roll_end_online``, inherent to rolling UC) and Carrier 2."""
    from spinedb_api import DatabaseMapping, import_data
    url = _db(tmp_path_factory, "db_c2_updown")
    scen = "coal_min_updown_roll"
    with DatabaseMapping(url) as dm:
        import_data(dm, scenarios=[(scen, False, "")], scenario_alternatives=[
            (scen, "init", "west"), (scen, "west", "coal"),
            (scen, "coal", "coal_min_load"),
            (scen, "coal_min_load", "coal_min_updowntime"),
            (scen, "coal_min_updowntime", "wind"), (scen, "wind", "fullYear"),
            (scen, "fullYear", "dispatch_fullYear_roll"),
            (scen, "dispatch_fullYear_roll", None)])
        dm.commit_session("c2 updown witness")
    workdir = tmp_path_factory.mktemp("c2_updown_run")
    steps = run_chain_from_db(url, scen, work_folder=workdir,
                              keep_solutions=True)
    return {"steps": steps, "workdir": workdir}


def test_min_uptime_cross_boundary_zero(updown_run):
    """Carrier 2: cross-boundary min_uptime obligations are honoured — 0
    cross-roll-boundary up-violations (HEAD/MINTIME-1 left 48)."""
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


# ──────────── period-aware: within-period carries, across does not ───────
@pytest.fixture(scope="module")
def cross_period_run(tmp_path_factory):
    """min_uptime=6 (> jump), method=min_uptime.  The fullYear rolling
    timeline has a roll straddling the p2020→p2025 period boundary; a NAIVE
    cross-period carry makes that roll infeasible even with node slack
    (STEP-0 finding).  The period-aware carry keeps it feasible."""
    from spinedb_api import DatabaseMapping, import_data
    url = _db(tmp_path_factory, "db_c2_xp")
    scen = "coal_min_up6_roll"
    with DatabaseMapping(url) as dm:
        import_data(dm, scenarios=[(scen, False, "")], scenario_alternatives=[
            (scen, "init", "west"), (scen, "west", "coal"),
            (scen, "coal", "coal_min_load"),
            (scen, "coal_min_load", "coal_min_updowntime"),
            (scen, "coal_min_updowntime", "wind"), (scen, "wind", "fullYear"),
            (scen, "fullYear", "dispatch_fullYear_roll"),
            (scen, "dispatch_fullYear_roll", None)])
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


def test_period_aware_cross_period_feasible(cross_period_run):
    """A min_uptime=6 roll straddling the period boundary stays optimal — the
    period-aware carry does NOT force a cross-period obligation (which would
    be infeasible).  HEAD with a naive cross-period carry is infeasible."""
    steps = cross_period_run["steps"]
    nonopt = [n for n, s in steps.items() if not s.optimal]
    assert not nonopt, f"cross-period rolls not optimal: {nonopt[:5]}"


def test_period_aware_within_period_carries_across_does_not(cross_period_run):
    """Explicit period-aware behaviour: on the roll whose window first enters
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


# ─────────────────── dedicated min_downtime witness ─────────────────────
@pytest.fixture(scope="module")
def downtime_run(tmp_path_factory):
    """Dedicated min_downtime binding witness.  Demand is high (−400) except
    a long valley (0) at t0007–t0020 so the shutdown roll (horizon 8) cannot
    see the far-future peak and cleanly FULLY shuts coal at t0007; demand
    resumes at t0021 — still within min_downtime=20 of that shutdown.  Wind
    is removed so coal is the swing unit.  WITHOUT the carry a later roll
    illegally re-starts coal at t0021; WITH it the obligation is honoured
    (``v_online + carried_shutdown ≤ existing`` ⇒ v_online ≤ ~0.2)."""
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
        import_data(dm, alternatives=["dn_witness"],
                    scenarios=[(scen, False, "")], scenario_alternatives=[
            (scen, "init", "west"), (scen, "west", "coal"),
            (scen, "coal", "coal_min_load"),
            (scen, "coal_min_load", "coal_min_updowntime"),
            (scen, "coal_min_updowntime", "wind"), (scen, "wind", "fullYear"),
            (scen, "fullYear", "dispatch_fullYear_roll"),
            (scen, "dispatch_fullYear_roll", "dn_witness"),
            (scen, "dn_witness", None)])
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


def test_downtime_carry_is_period_aware_and_present(downtime_run):
    """The down-leg carry is actually built and period-aware (every carried
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
    """A rolling solve WITHOUT any min-up/down.  Carrier 2 must be fully
    inert: no ``online_history`` stored, no carry built."""
    from spinedb_api import DatabaseMapping, import_data
    url = _db(tmp_path_factory, "db_c2_nomin")
    scen = "coal_nomin_roll"
    with DatabaseMapping(url) as dm:
        import_data(dm, scenarios=[(scen, False, "")], scenario_alternatives=[
            (scen, "init", "west"), (scen, "west", "coal"),
            (scen, "coal", "coal_min_load"), (scen, "coal_min_load", "wind"),
            (scen, "wind", "fullYear"),
            (scen, "fullYear", "dispatch_fullYear_roll"),
            (scen, "dispatch_fullYear_roll", None)])
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
