"""reserve_price_t = the sum of the duals of every requirement row of the group, per MW held for one hour.

A reserve group can have up to four kinds of requirement rows per step: the timeseries row, the dynamic row
and one large-failure (n-1) row per failing process.  They share the group's shortfall ``vq_reserve``, so
one more MW of every requirement at once costs the sum of their duals: ``penalty_reserve`` when the
shortfall is positive, and ``−dual[reserve_total_upper]`` (the value of one MW from a provider that is not
failing) when the group's total reserve is positive.  The price is reported for every group with a
requirement, in single, representative-period, rolling and stochastic solves:

    reserve_price[r,ud,g,d,t] = Σ_rows dual / s / (Δt[d,t] · w[d,t])
    w = timestep_weight · inflation_op / period_share · branch_weight

Every database is a JSON fixture plus an override alternative (``tests/engine/_unit_db.make_db``).
"""
from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest

from tests.engine._unit_db import PLANT, PLANT_CHAIN, b64, make_db, utf8

REL = "reserve__upDown__unit__node"
FAMILIES = ("reserveBalance_timeseries_eq", "reserveBalance_dynamic_eq",
            "reserveBalance_up_n_1_eq", "reserveBalance_down_n_1_eq")
PENALTY = 1000.0
STOCH_JSON = Path(__file__).resolve().parents[1] / "fixtures" / "stochastics.json"


def _run(tmp_path_factory, url: str, scenario: str) -> tuple[dict, Path]:
    from flextool.engine import run_chain_from_db

    wf = tmp_path_factory.mktemp(f"{scenario}_w")
    steps = run_chain_from_db(url, scenario, work_folder=wf, keep_solutions=True)
    assert steps
    for step in steps.values():
        assert step.solution is not None
    return steps, wf


def _summed_duals(step, r: str, ud: str, g: str) -> dict[tuple[str, str], float]:
    """Σ of the duals of the group's requirement rows per (d, t); the row key ends with (d, t)."""
    out: dict[tuple[str, str], float] = {}
    for fam in FAMILIES:
        try:
            du = step.solution.constraint_dual(fam)
        except KeyError:
            continue
        for key, dual in du.select("key", "dual").iter_rows():
            k = key.split(",")
            if (k[0], k[1], k[2]) == (r, ud, g):
                out[(k[-2], k[-1])] = out.get((k[-2], k[-1]), 0.0) + dual
    return out


def _price_column(wf: Path, step, r: str, ud: str, g: str) -> dict[tuple[str, str], float]:
    from flextool.lean_parquet import read_lean_parquet

    files = list(Path(wf).rglob(f"v_dual_reserve__upDown__group__period__t__{step.solve_name}.parquet"))
    assert len(files) == 1, files
    price = read_lean_parquet(files[0])
    cols = [c for c in price.columns if tuple(str(x) for x in c) == (r, ud, g)]
    assert len(cols) == 1, f"no reserve price column for {(r, ud, g)}: {list(price.columns)}"
    return {(str(idx[1]), str(idx[2])): float(v) for idx, v in price[cols[0]].items()}


def _weights(step) -> dict[tuple[str, str], float]:
    from flextool.engine.output._prices import price_weight_dt

    frame = price_weight_dt(step.flex_data, per_hour=True)
    return {(str(d), str(t)): w for d, t, w in frame.select("d", "t", "w").iter_rows()}


def _vq(step, r: str) -> dict[tuple[str, str], float]:
    vq = utf8(step.solution.value("vq_reserve")).filter(pl.col("r") == r)
    return {(d, t): v for d, t, v in vq.select("d", "t", "value").iter_rows()}


def _realized(step) -> set[tuple[str, str]]:
    rd = step.flex_data.realized_dispatch
    assert rd is not None and rd.height > 0
    return {(str(a), str(b)) for a, b in rd.select(rd.columns[0], rd.columns[1]).iter_rows()}


def _assert_price_is_summed_duals(step, wf: Path, r: str, ud: str, g: str) -> dict[tuple[str, str], float]:
    """The parquet holds the realized steps of the solve, each = Σ duals / s / (Δt · w) (empty where w = 0);
    returns the non-empty prices."""
    from flextool.engine.solve._orchestration import _LEGACY_DEFAULT_OBJECTIVE_SCALE as S

    got = _price_column(wf, step, r, ud, g)
    duals = _summed_duals(step, r, ud, g)
    w = _weights(step)
    assert set(got) == _realized(step)
    for dt, price in got.items():
        if w[dt] == 0.0:      # a step the objective does not weigh: no price
            assert math.isnan(price), dt
            continue
        assert price == pytest.approx(duals[dt] / S / w[dt], rel=1e-9, abs=1e-9), dt
    return {dt: price for dt, price in got.items() if not math.isnan(price)}


# ---------------------------------------------------------------------------
# the group's requirement methods (coal_plant + wind_plant at west, tests.json)


def _group_values(r: str, method: str, *, reservation: float | None, group_ratio: float | None,
                  wind_dynamic: float | None) -> tuple[list, list]:
    grp = [r, "up", "electricity"]
    ents = [["reserve", r, None], ["reserve__upDown__group", grp, None]]
    vals = [["reserve__upDown__group", grp, "reserve_method", b64(method, "str")],
            ["reserve__upDown__group", grp, "penalty_reserve", b64(PENALTY, "float")]]
    if reservation is not None:
        vals.append(["reserve__upDown__group", grp, "reservation", b64(reservation, "float")])
    if group_ratio is not None:
        vals.append(["reserve__upDown__group", grp, "increase_reserve_ratio", b64(group_ratio, "float")])
    for unit, prm in (("wind_plant", {"max_share": 0.05, "reliability": 0.6, "large_failure_ratio": 0.1}),
                      (PLANT, {"large_failure_ratio": 1.0, "max_share": 0.01, "reliability": 1.0})):
        key = [r, "up", unit, "west"]
        ents.append([REL, key, None])
        vals.append([REL, key, "is_enabled", b64("yes", "str")])
        vals += [[REL, key, n, b64(v, "float")] for n, v in prm.items()]
        if wind_dynamic is not None and unit == "wind_plant":
            vals.append([REL, key, "increase_reserve_ratio", b64(wind_dynamic, "float")])
    return ents, vals


def _method_case(method: str) -> tuple[list, list]:
    if method == "timeseries_and_dynamic":
        # the dynamic requirement (10 % of the west demand) exceeds the 5 MW reservation: the dynamic row
        # binds and the timeseries row does not
        return _group_values("rq", method, reservation=5.0, group_ratio=0.1, wind_dynamic=None)
    has_ts = "timeseries" in method or method == "all"
    has_dyn = method in ("dynamic_and_large_failure", "all")
    return _group_values("rq", method, reservation=30.0 if has_ts else None, group_ratio=None,
                         wind_dynamic=0.05 if has_dyn else None)


_RP_TIMESET = [["timeset", "rp2", None]]


def _rp_values() -> list:
    """Representative periods: two 24-step blocks with unequal weights (timestep_weight ≠ 1)."""
    rpw = {"index_type": "str", "index_name": "base_period", "rank": 1,
           "data": [["t0001", {"type": "map", "index_type": "str", "index_name": "representative_period",
                               "rank": 1, "data": [["t0001", 0.3], ["t0025", 0.7]]}]]}
    dur = {"index_type": "str", "index_name": "time", "data": [["t0001", 24.0], ["t0025", 24.0]]}
    return [["timeset", "rp2", "timeline", b64("y2020", "str")],
            ["timeset", "rp2", "timeset_duration", b64(dur, "map")],
            ["timeset", "rp2", "representative_period_weights", b64(rpw, "map")],
            ["solve", "y2020_2day_dispatch", "period_timeset",
             b64({"index_type": "str", "index_name": "period", "data": [["p2020", "rp2"]]}, "map")]]


@pytest.mark.parametrize("method", ["large_failure_only", "timeseries_and_large_failure",
                                    "dynamic_and_large_failure", "all", "timeseries_and_dynamic",
                                    "timeseries_and_large_failure_rp"])
def test_reserve_price_sums_requirement_duals(tmp_path_factory, method: str) -> None:
    """The price is Σ duals of the group's requirement rows / s / (Δt · w) at every step, and equals
    penalty_reserve wherever the group's shortfall is positive (the ``_rp`` case: timestep weights ≠ 1)."""
    rp = method.endswith("_rp")
    ents, vals = _method_case(method.removesuffix("_rp"))
    if rp:
        ents, vals = ents + _RP_TIMESET, vals + _rp_values()
    url, scen = make_db(tmp_path_factory, alt=f"rps_{method[:16]}", values=vals,
                        base_chain=PLANT_CHAIN + ["wind"], entities=ents)
    steps, wf = _run(tmp_path_factory, url, scen)
    step = list(steps.values())[-1]
    got = _assert_price_is_summed_duals(step, wf, "rq", "up", "electricity")
    vq = _vq(step, "rq")
    short = [dt for dt in got if vq.get(dt, 0.0) > 1e-6]
    assert short, "the requirement must be short somewhere for the penalty check to mean something"
    for dt in short:
        assert got[dt] == pytest.approx(PENALTY, rel=1e-7), dt
    if method == "timeseries_and_dynamic":
        assert max(got.values()) > 1.0
    if rp:
        assert any(abs(w - 1.0) > 1e-6 for w in _weights(step).values()), "timestep weights must not all be 1"


def test_reserve_price_n_1_without_shortfall(tmp_path_factory) -> None:
    """tests.json ``reserve_n_1`` (large_failure_only, met by the providers): the price is the value of one MW
    from the provider that does not fail, −dual[reserve_total_upper] / s / (Δt · w), positive somewhere."""
    from flextool.engine.solve._orchestration import _LEGACY_DEFAULT_OBJECTIVE_SCALE as S

    url, scen = make_db(tmp_path_factory, alt="rps_n1", values=[], base_chain=PLANT_CHAIN + ["wind", "reserve_n_1"])
    steps, wf = _run(tmp_path_factory, url, scen)
    step = list(steps.values())[-1]
    got = _assert_price_is_summed_duals(step, wf, "n_1", "up", "electricity")
    upper = {}
    for key, dual in step.solution.constraint_dual("reserve_total_upper").select("key", "dual").iter_rows():
        k = key.split(",")
        if (k[0], k[1], k[2]) == ("n_1", "up", "electricity"):
            upper[(k[-2], k[-1])] = dual
    w = _weights(step)
    for dt, price in got.items():
        assert price == pytest.approx(-upper[dt] / S / w[dt], rel=1e-7, abs=1e-7), dt
    assert max(got.values()) > 1.0


def test_reserve_price_rolling(tmp_path_factory) -> None:
    """Rolling dispatch (36 h horizon, 24 h jump) with ``reserve_n_1``: every roll writes the summed price of
    its realized steps."""
    vals = [["solve", "y2020_2day_dispatch", "solve_mode", b64("rolling_window", "str")],
            ["solve", "y2020_2day_dispatch", "rolling_solve_horizon", b64(36.0, "float")],
            ["solve", "y2020_2day_dispatch", "rolling_solve_jump", b64(24.0, "float")]]
    url, scen = make_db(tmp_path_factory, alt="rps_roll", values=vals,
                        base_chain=PLANT_CHAIN + ["wind", "reserve_n_1"])
    steps, wf = _run(tmp_path_factory, url, scen)
    assert len(steps) >= 2, list(steps)
    seen: set[tuple[str, str]] = set()
    for step in steps.values():
        got = _assert_price_is_summed_duals(step, wf, "n_1", "up", "electricity")
        assert not (set(got) & seen), "rolls realize disjoint steps"
        seen |= set(got)
    assert len(seen) == 48


def _unequal_branches() -> list:
    """2day_dispatch branch weights realized 1, upper 2, lower 0.5, mid 1 (unequal branch weights)."""
    def branch(w: float) -> dict:
        return {"type": "map", "index_type": "str",
                "data": [["t0001", {"type": "map", "index_type": "str", "data": [["no", w]]}]]}
    sb = {"index_type": "str", "data": [["period1", {"type": "map", "index_type": "str", "data": [
        ["realized", branch(1.0)], ["upper", branch(2.0)], ["lower", branch(0.5)], ["mid", branch(1.0)]]}]]}
    return [["solve", "2day_dispatch", "stochastic_branches", b64(sb, "map")]]


def test_reserve_price_stochastic(tmp_path_factory) -> None:
    """``2_day_stochastic_dispatch`` with a large_failure_only group on demand_node (coal fails; gas and coal
    provide) and unequal branch weights: the price at every (branch period, t) is Σ duals / s / (Δt · w) with
    the branch weight in w."""
    grp = ["rq", "up", "dn"]
    ents = [["upDown", "up", None], ["group", "dn", None], ["group__node", ["dn", "demand_node"], None],
            ["reserve", "rq", None], ["reserve__upDown__group", grp, None]]
    vals = [["reserve__upDown__group", grp, "reserve_method", b64("large_failure_only", "str")],
            ["reserve__upDown__group", grp, "penalty_reserve", b64(PENALTY, "float")]]
    for unit, prm in (("coal_plant", {"large_failure_ratio": 0.5, "max_share": 0.1, "reliability": 1.0}),
                      ("gas_plant", {"max_share": 0.2, "reliability": 1.0})):
        key = ["rq", "up", unit, "demand_node"]
        ents.append([REL, key, None])
        vals.append([REL, key, "is_enabled", b64("yes", "str")])
        vals += [[REL, key, n, b64(v, "float")] for n, v in prm.items()]
    url, scen = make_db(tmp_path_factory, alt="rps_stoch", values=vals + _unequal_branches(),
                        base_chain=["init", "system", "2day"], entities=ents, fixture_json=STOCH_JSON)
    steps, wf = _run(tmp_path_factory, url, scen)
    step = list(steps.values())[-1]
    assert step.flex_data.pdt_branch_weight is not None
    bw = {(d, t): v for d, t, v in utf8(step.flex_data.pdt_branch_weight.frame).select("d", "t", "value").iter_rows()}
    got = _assert_price_is_summed_duals(step, wf, "rq", "up", "dn")
    assert any(abs(bw.get(dt, 1.0) - 1.0) > 1e-6 for dt in got), "some priced step must have a branch weight ≠ 1"
    periods = {d for d, _t in got}
    assert len(periods) > 1, f"the price must cover the branch periods: {periods}"
    assert max(got.values()) > 1.0

