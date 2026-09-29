"""Slice E — hedged two-stage stochastic investment (mid-horizon reveal).

Design: ``specs/sliceE_invest_na_design.md`` (§3 fan-out, §5 storage
linkage, §6 per-path caps, §7 invest-NA-by-construction, §8 hedging
hand-calc).

Fixture ``stoch_two_period_hedge.json`` (built via
``build_stoch_two_period_hedge.py``): two 3-step periods p2035 / p2040,
period-Map ``invest_cost`` {p2035: 1, p2040: 3} so the first stage is
cheaper per delivered p2040-MW; a free ``wind`` unit's branch-varying
``upper_limit`` profile carves the residual the investable ``base`` unit
must cover (p2035 = 40 shared; p2040 realized/high = 120, low = 60).

Three scenarios on identical scenario data:
  * ``hedge``         — mid-horizon reveal at p2040 (RP = 173 250).
  * ``hedge_ws``      — wait-and-see fan-at-p2035 control (WS = 157 500).
  * ``hedge_storage`` — mid-horizon + a stochastic-group storage node
    (``resv``) so the dispatch storage-NA + continuity linkage fire.

Headline (§8.3): WS = 157 500 < RP = 173 250 < EEV = 192 937.5, so
EVPI = RP − WS = 15 750 > 0 and VSS = EEV − RP = 19 687.5 > 0 — genuine
hedging value.  The RP ≠ WS gap is a direct behavioural proof that the
mid-horizon first stage ``v_invest[base, p2035]`` is genuinely SHARED
(Option B) — if E0 accidentally fanned p2035 the objective would collapse
to WS.
"""
from __future__ import annotations

import logging
import os
from collections import defaultdict
from types import SimpleNamespace

import polars as pl
import pytest

from flextool.engine_polars._solve_state import (
    ActiveTimeEntry,
    FlexToolConfigError,
)
from flextool.engine_polars._stochastic import StochasticSolver
from flextool.engine_polars._timeline import make_step_jump

SOLVE = "stoch_2p_hedge"

# Hand-calc headline numbers (design §8).
RP = 173_250.0    # hedged mid-horizon reveal
WS = 157_500.0    # wait-and-see (fan-at-p2035)
EEV = 192_937.5   # deterministic mean-value decision, evaluated
EVPI = RP - WS    # = 15 750
VSS = EEV - RP    # = 19 687.5


# ---------------------------------------------------------------------------
# Unit harness (solver-free) — create_stochastic_periods + make_step_jump
# ---------------------------------------------------------------------------


def _ats(pairs):
    return [ActiveTimeEntry(timestep=t, index=i, duration=1.0)
            for t, i in pairs]


P1 = _ats([("t0001", 0), ("t0002", 1), ("t0003", 2)])
P2 = _ats([("t0004", 3), ("t0005", 4), ("t0006", 5)])


def _make_solver() -> StochasticSolver:
    state = SimpleNamespace(
        logger=logging.getLogger("test_recourse_invest_hedge"),
        timeline=SimpleNamespace(stochastic_timesteps=defaultdict(list)),
    )
    return StochasticSolver(state)


def _run(solver, info):
    active = {"p2035": P1, "p2040": P2}
    realized = {"p2035": P1[0:], "p2040": P2[0:]}
    fix_storage = {"p2040": P2[0:]}
    return solver.create_stochastic_periods(
        {"s": info}, ["s"], {"s": "s"},
        {"s": active}, {"s": fix_storage}, {"s": realized},
    )


# --- T-F: mid-horizon fan-out is Option B (shared trunk) --------------------

# stochastic_branches rows: (period, branch, start_step, realized, weight).
MIDHORIZON_INFO = [
    ("p2040", "rlz", "t0004", "yes", "1"),
    ("p2040", "low", "t0004", "no", "3"),
]


def test_midhorizon_fan_period_branch_and_active_time() -> None:
    """T-F (§3.2): a reveal declared at p2040 keeps p2035 a single
    real-named trunk period and fans ONLY p2040 — exactly Option B."""
    solver = _make_solver()
    pb, sbtb, atl, _jumps, _fstl, _rtl, bstl = _run(solver, MIDHORIZON_INFO)

    # p2035 is a single trunk self-row (NOT fanned); p2040 fans.
    assert pb["s"] == [
        ("p2035", "p2035"),
        ("p2040", "p2040"),
        ("p2040", "p2040_rlz"),
        ("p2040", "p2040_low"),
    ]
    # Active-time keys: p2035 trunk, p2040 realized, p2040_low branch copy.
    assert list(atl["s"].keys()) == ["p2035", "p2040", "p2040_low"]
    assert atl["s"]["p2035"] == P1                # trunk unchanged
    assert atl["s"]["p2040_low"] == P2[0:]        # branch copy of p2040
    # Realized resolution: trunk p2035 -> rlz (via the after-branch path),
    # p2040 -> rlz, p2040_low -> low.
    assert ("p2035", "rlz") in sbtb["s"]
    assert ("p2040", "rlz") in sbtb["s"]
    assert ("p2040_low", "low") in sbtb["s"]
    # The reveal is recorded at p2040 / t0004.
    assert bstl["s"] == ("p2040", "t0004")


# --- T-V: validation (period-boundary reveal, one realized per period) ------


def test_validation_midperiod_start_raises() -> None:
    """T-V (§3.3 check 1): a branch start_step that is not the first step
    of its period is rejected (invest is period-granular)."""
    solver = _make_solver()
    bad = [
        ("p2040", "rlz", "t0004", "yes", "1"),
        ("p2040", "low", "t0005", "no", "3"),   # not p2040's first step
    ]
    with pytest.raises(FlexToolConfigError, match="first step"):
        _run(solver, bad)


def test_validation_two_realized_in_period_raises() -> None:
    """T-V (§3.3 check 2): a branching period with two realized:yes
    branches is rejected."""
    solver = _make_solver()
    bad = [
        ("p2040", "rlz", "t0004", "yes", "1"),
        ("p2040", "low", "t0004", "yes", "3"),  # two realized
    ]
    with pytest.raises(FlexToolConfigError, match="exactly one realized"):
        _run(solver, bad)


def test_validation_zero_realized_in_period_raises() -> None:
    """T-V (§3.3 check 2): a branching period with no realized branch is
    rejected (subsumes the old 'no realized start time' guard)."""
    solver = _make_solver()
    bad = [
        ("p2040", "rlz", "t0004", "no", "1"),
        ("p2040", "low", "t0004", "no", "3"),
    ]
    with pytest.raises(FlexToolConfigError, match="exactly one realized"):
        _run(solver, bad)


def test_validation_fan_at_first_still_passes() -> None:
    """T-V control: the fan-at-first-step shape (branch at the solve's
    first step) still validates and fans byte-identically."""
    solver = _make_solver()
    good = [
        ("p2035", "rlz", "t0001", "yes", "1"),
        ("p2035", "low", "t0001", "no", "3"),
    ]
    pb, _sbtb, atl, *_ = _run(solver, good)
    assert list(atl["s"].keys()) == [
        "p2035", "p2035_low", "p2040", "p2040_low",
    ]
    assert ("p2035", "p2035_low") in pb["s"]


# --- T-S (jump): storage-continuity predecessor + fan-at-first control ------

# make_step_jump tuple = (period, timestep, previous,
#   previous_within_timeset, previous_period, previous_within_solve, jump).
_PREV_PERIOD = 4
_PREV_WITHIN_SOLVE = 5


def _first_step_rows(mj):
    return {(r[0], r[1]): r for r in mj if r[1] in ("t0001", "t0004")}


def test_step_jump_midhorizon_links_branch_to_shared_trunk() -> None:
    """T-S (§5.2, F1): the mid-horizon branch copy p2040_low's first step
    links to the shared pre-reveal trunk p2035's LAST step (t0003) — NOT
    a self-cycle, and NOT its own realized sibling p2040."""
    atl = {"p2035": P1, "p2040": P2, "p2040_low": P2}
    pb = [("p2035", "p2035"), ("p2040", "p2040"),
          ("p2040", "p2040_rlz"), ("p2040", "p2040_low")]
    sbtb = [("p2040_low", "low"), ("p2035", "rlz"), ("p2040", "rlz")]
    rows = _first_step_rows(make_step_jump(atl, pb, sbtb))
    branch = rows[("p2040_low", "t0004")]
    assert branch[_PREV_PERIOD] == "p2035"
    assert branch[_PREV_WITHIN_SOLVE] == "t0003"
    # The realized continuation anchor p2040 also links to the trunk.
    assert rows[("p2040", "t0004")][_PREV_PERIOD] == "p2035"


def test_step_jump_fan_at_first_self_cycles_control() -> None:
    """T-S control (F1 (a) byte-parity): in the fan-at-first-step shape
    the branching-period copy p2035_low STILL self-cycles (its anchor IS
    the first period → empty anchor range → self-cycle)."""
    atl = {"p2035": P1, "p2035_low": P1, "p2040": P2, "p2040_low": P2}
    pb = [("p2035", "p2035"), ("p2035", "p2035_rlz"),
          ("p2035", "p2035_low"), ("p2040", "p2040"),
          ("p2040", "p2040_rlz"), ("p2040", "p2040_low")]
    sbtb = [("p2035_low", "low"), ("p2040_low", "low"),
            ("p2035", "rlz"), ("p2040", "rlz")]
    rows = _first_step_rows(make_step_jump(atl, pb, sbtb))
    # p2035_low self-cycles on its own last step.
    assert rows[("p2035_low", "t0001")][_PREV_PERIOD] == "p2035_low"
    # p2040_low continues from the earlier same-branch member p2035_low.
    assert rows[("p2040_low", "t0004")][_PREV_PERIOD] == "p2035_low"


# ---------------------------------------------------------------------------
# Workdir fixtures (full cascade) + load_flextool structural tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hedge_wf(scenario_workdir):
    prev = os.environ.get("FLEXTOOL_AUTOSCALE_STRICT")
    os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = "1"
    try:
        return scenario_workdir("hedge",
                                db_fixture="stoch_two_period_hedge")
    finally:
        if prev is None:
            os.environ.pop("FLEXTOOL_AUTOSCALE_STRICT", None)
        else:
            os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = prev


@pytest.fixture(scope="module")
def hedge_ws_wf(scenario_workdir):
    return scenario_workdir("hedge_ws", db_fixture="stoch_two_period_hedge")


@pytest.fixture(scope="module")
def hedge_storage_wf(scenario_workdir):
    prev = os.environ.get("FLEXTOOL_AUTOSCALE_STRICT")
    os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = "1"
    try:
        return scenario_workdir("hedge_storage",
                                db_fixture="stoch_two_period_hedge")
    finally:
        if prev is None:
            os.environ.pop("FLEXTOOL_AUTOSCALE_STRICT", None)
        else:
            os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = prev


@pytest.fixture(scope="module")
def hedge_resolved(hedge_wf):
    """In-process re-solve off the snapshot workdir (gives the branch
    v_invest columns the committed parquet omits)."""
    from polar_high import Problem
    from flextool.engine_polars import build_flextool, load_flextool

    data = load_flextool(hedge_wf)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal
    return data, sol, pb


# --- T-RP / T-WS / T-EEV: the hedging gate ---------------------------------


@pytest.mark.solver
def test_rp_objective_hand_calc(hedge_wf) -> None:
    """T-RP (§8.2): the hedged mid-horizon objective is exactly 173 250."""
    obj = float(
        pl.read_parquet(hedge_wf / "output_raw" / f"v_obj__{SOLVE}.parquet")
        ["objective"][0]
    )
    assert obj == pytest.approx(RP, rel=1e-9)


@pytest.mark.solver
def test_rp_invest_is_a_genuine_hedge(hedge_resolved) -> None:
    """T-RP (§8.2): the shared first stage x = v_invest[base, p2035] = 60
    (a hedge between D2_low = 60 and D2_high = 120, above D1 = 40), with
    recourse r_rlz = 60 at p2040 and r_low = 0 at p2040_low."""
    _data, sol, _pb = hedge_resolved
    vi = sol.value("v_invest_p")
    got = {r["d"]: round(float(r["value"]), 4)
           for r in vi.iter_rows(named=True)}
    assert got == pytest.approx({
        "p2035": 60.0, "p2040": 60.0, "p2040_low": 0.0,
    })


@pytest.mark.solver
def test_ws_objective_and_evpi(hedge_ws_wf, hedge_wf) -> None:
    """T-WS (§8.2): the wait-and-see control is 157 500, strictly below
    RP — EVPI = RP − WS = 15 750 > 0.  RP > WS is the behavioural proof
    that the mid-horizon first stage is genuinely SHARED (had E0 fanned
    p2035 the objective would collapse to WS)."""
    ws = float(
        pl.read_parquet(hedge_ws_wf / "output_raw" / f"v_obj__{SOLVE}.parquet")
        ["objective"][0]
    )
    rp = float(
        pl.read_parquet(hedge_wf / "output_raw" / f"v_obj__{SOLVE}.parquet")
        ["objective"][0]
    )
    assert ws == pytest.approx(WS, rel=1e-9)
    assert rp - ws == pytest.approx(EVPI, rel=1e-9)
    assert rp > ws


def test_eev_and_vss_hand_calc() -> None:
    """T-EEV (§8.2, hand-calc + arithmetic pin): the mean-value decision
    x_EV = 0.75·60 + 0.25·120 = 75; frozen and evaluated with recourse,
    EEV = 2100·75 + 0.25·3150·(120−75) = 192 937.5.  VSS = EEV − RP =
    19 687.5 > 0, and WS < RP < EEV strictly."""
    K1, K2 = 2100.0, 3150.0
    x_ev = 0.75 * 60 + 0.25 * 120
    assert x_ev == 75.0
    eev = K1 * x_ev + 0.25 * K2 * (120 - x_ev)
    assert eev == pytest.approx(EEV)
    assert EEV - RP == pytest.approx(VSS)
    assert WS < RP < EEV


# --- T-L / T-E2: lineage populated, invest-NA by construction ---------------


@pytest.mark.solver
def test_lineage_shared_trunk_on_every_leaf(hedge_resolved) -> None:
    """T-L (§4.1): dd_same_scenario carries (p2035, p2040_low) and its
    symmetric mate — the shared first-stage invest at p2035 is on the low
    scenario's information path (capacity alive on every leaf's edd walk)."""
    data, _sol, _pb = hedge_resolved
    dd = {(str(r["d"]), str(r["d_other"]))
          for r in data.dd_same_scenario.iter_rows(named=True)}
    assert ("p2035", "p2040_low") in dd
    assert ("p2040_low", "p2035") in dd


@pytest.mark.solver
def test_d_leaf_is_a_relation_not_a_partition(hedge_resolved) -> None:
    """T-C / T-L (§6.2): d_leaf is a RELATION — the shared pre-reveal
    anchor p2035 is on BOTH the __realized leaf and the low leaf, so each
    scenario path's total cap counts it.  A strict partition would put
    p2035 on __realized only (under-counting the low path)."""
    data, _sol, _pb = hedge_resolved
    dl = {(str(r["d"]), str(r["leaf"]))
          for r in data.d_leaf.iter_rows(named=True)}
    assert dl == {
        ("p2035", "__realized"), ("p2040", "__realized"),
        ("p2035", "low"), ("p2040_low", "low"),
    }


@pytest.mark.solver
def test_pd_non_anticipativity_empty_no_invest_na(hedge_resolved) -> None:
    """T-L / T-E2 (§7): under Option B the first stage is a single shared
    trunk variable, so pd_non_anticipativity is provably EMPTY and no
    non_anticipativity_invest_* constraint family is emitted (NA by
    construction — no new autoscale family, CLAUDE.md invariant 1)."""
    data, _sol, pb = hedge_resolved
    pdna = data.pd_non_anticipativity
    assert pdna is None or pdna.height == 0
    invest_na = [n for n in pb.cstr_names()
                 if "non_anticipativity_invest" in n]
    assert invest_na == []


@pytest.mark.solver
def test_per_path_cap_two_leaf_rows(hedge_resolved) -> None:
    """T-C (§6.2 / §8.3): the entity total cap fans into ONE row per
    scenario leaf (base × {__realized, low}) under the ``…_path`` name;
    the legacy single-row ``maxInvest_entity_total`` (which would sum all
    three periods) is gone."""
    _data, _sol, pb = hedge_resolved
    names = pb.cstr_names()
    assert "maxInvest_entity_total_path" in names
    assert "maxInvest_entity_total" not in names
    recs = pb.cstrs_named("maxInvest_entity_total_path")
    over = recs[0].over
    assert over.height == 2
    assert sorted(str(v) for v in over["leaf"].to_list()) == [
        "__realized", "low",
    ]
    assert {str(v) for v in over["p"].to_list()} == {"base"}


@pytest.mark.solver
def test_committed_output_realized_only(hedge_wf) -> None:
    """§9.4: the committed v_invest output carries only the realized
    (trunk) periods with real names — the shared first stage p2035 = 60
    and the realized recourse p2040 = 60; branch rows are dropped."""
    inv = pl.read_parquet(
        hedge_wf / "output_raw" / f"v_invest__{SOLVE}.parquet")
    got = {r["period"]: round(float(r["base"]), 6)
           for r in inv.iter_rows(named=True)}
    assert got == {"p2035": 60.0, "p2040": 60.0}


# --- T-S (storage): NA fires + branch copy links to shared trunk -----------


@pytest.mark.solver
def test_storage_na_fires_on_reveal_pair(hedge_storage_wf) -> None:
    """T-S (§5.1, §9.1): with a stochastic-group storage node the dispatch
    non-anticipativity net-charge pinning fires on the reveal pair
    (p2040, p2040_low) — the branch's storage state is pinned to the
    realized branch at the reveal window."""
    from polar_high import Problem
    from flextool.engine_polars import build_flextool, load_flextool

    data = load_flextool(hedge_storage_wf)
    pb = Problem()
    build_flextool(pb, data)
    names = pb.cstr_names()
    assert "non_anticipativity_storage_use" in names
    over = pb.cstrs_named("non_anticipativity_storage_use")[0].over
    pairs = {(str(r["d"]), str(r["b"])) for r in over.iter_rows(named=True)}
    assert pairs == {("p2040", "p2040_low")}


def test_storage_branch_copy_predecessor_is_shared_trunk(
        hedge_storage_wf) -> None:
    """T-S (§5.2): in the storage scenario, the branch copy p2040_low's
    first step still links to the shared trunk p2035's last step (the
    two-stage storage-state non-anticipativity), verified on the emitted
    step_previous.csv."""
    sp = pl.read_csv(hedge_storage_wf / "solve_data" / "step_previous.csv")
    row = sp.filter((pl.col("period") == "p2040_low")
                    & (pl.col("time") == "t0004")).to_dicts()[0]
    assert row["previous_period"] == "p2035"
    assert row["previous_within_solve"] == "t0003"


def test_make_step_jump_dtttdt_parity_midhorizon(hedge_wf) -> None:
    """T-S (§5.2, parity guard): the CSV step_previous and the live model
    lag frame ``dtttdt`` agree on d_previous / t_previous_within_solve for
    every first-of-period row (the mid-horizon predecessor linkage is the
    same in both derivations)."""
    from flextool.engine_polars import load_flextool

    data = load_flextool(hedge_wf)
    dtttdt = {(str(r["d"]), str(r["t"])): r
              for r in data.dtttdt.rows(named=True)}
    sp = pl.read_csv(hedge_wf / "solve_data" / "step_previous.csv")
    assert sp.height == len(dtttdt)
    first_steps = {("p2035", "t0001"), ("p2040", "t0004"),
                   ("p2040_low", "t0004")}
    for r in sp.rows(named=True):
        key = (r["period"], r["time"])
        m = dtttdt[key]
        assert str(m["d_previous"]) == r["previous_period"], key
        assert str(m["t_previous_within_solve"]) == (
            r["previous_within_solve"]), key
        if key in first_steps:
            continue
        assert str(m["t_previous"]) == r["previous"], key
