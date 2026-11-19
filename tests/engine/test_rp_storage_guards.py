"""Configuration guards and warnings of storage in solves on representative periods (owner decisions, Phase 4).

* A rolling_window solve on representative periods stops with a configuration error (rolling over representative
  periods is not supported; before, its *_blended_weights nodes bound forward only).
* A plain bind_within_solve / bind_within_period / bind_intraperiod_blocks node in a solve on representative
  periods runs with a warning (owner decision 2026-11-19; before a configuration error): the level moves from one
  representative period into the next without their weights while the flows count with them.  That is reasonable
  with equal weights or storage whose inflow timing does not matter; with unequal weights and a timed inflow, as in
  the reproduction below, energy charged in a period of low weight serves one of high weight.
* A storage node with balance slack linked to a node with penalty_method off is named in a warning.

The tiny tank model of ``test_rolling_storage_scopes`` (built from the schema JSON, never a checked-in ``.sqlite``),
on a representative-period timeset ``rp``: timeline t01..t16 (1 h), base periods t01, t05, t09, t13 (4 h each);
representative period A = t01-t04 (stands for base t01, weight 1) and B = t09-t12 (stands for bases t05, t09 and
t13, weight 3).  The charger ``gen`` (10 MW, 1 CUR/MWh) is available only in A, the load (5 MW) is drawn only in B.
"""
from __future__ import annotations

import contextlib
import io

import polars as pl
import pytest
import spinedb_api as api

from flextool.engine.input._solve_state import FlexToolConfigError
from tests.engine import test_rolling_storage_scopes as T
from tests.engine._unit_db import utf8

LOAD_B = {t: 5.0 for t in ("t09", "t10", "t11", "t12")}
GEN_A = {t: 1.0 for t in ("t01", "t02", "t03", "t04")}
WEIGHTS = {"t01": ("t01", 1.0), "t05": ("t09", 1.0), "t09": ("t09", 1.0), "t13": ("t09", 1.0)}


def _rp_solve(**extra) -> dict:
    return {"solve_mode": "single_solve", "period_timeset": T.tmap({"p1": "rp"}, "period"),
            "years_represented": T.tmap({"p1": 1.0}, "period"), "realized_periods": T.arr(["p1"]), **extra}


def _run_rp(tmp_path, method: str, solve: dict | None = None, out: list | None = None) -> dict:
    """The tank model on the representative periods A and B, the tank on ``method`` with a free start.  The
    printed output is appended to ``out``."""
    from flextool.engine import run_chain_from_db
    from flextool.update_flextool.db_handle import open_database_mapping

    tmp_path.mkdir(parents=True, exist_ok=True)
    url = T.build_db(tmp_path, {"rp_solve": solve or _rp_solve()}, top=["rp_solve"],
                     tank={"storage_binding_method": method, "storage_start_end_method": "fix_nothing"},
                     load=LOAD_B, gen=GEN_A, timesets={"rp": {"t01": 4.0, "t09": 4.0}})
    rpw = api.Map(list(WEIGHTS), [api.Map([rep], [w], index_name="representative_period")
                                  for rep, w in WEIGHTS.values()], index_name="base_period")
    with open_database_mapping(url) as db:
        _count, errors = api.import_data(db, parameter_values=[
            ("timeset", "rp", "representative_period_weights", rpw, T.ALT)])
        assert not errors, errors
        db.commit_session("rp weights")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        steps = run_chain_from_db(url, T.SCENARIO, work_folder=tmp_path / "work", keep_solutions=True)
    if out is not None:
        out.append(buf.getvalue())
    for step in steps.values():
        assert step.solution is not None and step.solution.optimal
    return steps


def _weighted_mwh(step, process: str) -> float:
    """Σ_t timestep weight × flow × step duration of ``process`` (MWh the solve counts for the whole period)."""
    fd = step.flex_data
    flows = utf8(step.solution.value("v_flow")).filter(pl.col("p") == process)
    weight = utf8(fd.p_timestep_weight.frame).select("d", "t", pl.col("value").alias("w"))
    unitsize = dict(zip(*[utf8(fd.p_unitsize.frame)[c].to_list() for c in ("p", "value")]))
    joined = flows.join(weight, on=["d", "t"], how="left").with_columns(pl.col("w").fill_null(1.0))
    return float((joined["value"] * joined["w"]).sum() * unitsize[process])   # 1 h steps


BLENDED = {"bind_within_solve": "bind_within_solve_blended_weights",
           "bind_within_period": "bind_within_period_blended_weights",
           "bind_intraperiod_blocks": "bind_within_period_blended_weights"}


@pytest.mark.parametrize("method", list(BLENDED))
def test_plain_cycle_on_representative_periods_warns_and_moves_energy_unweighted(tmp_path, method):
    """The documented warning case (battery investigation, variant C): the plain cycle runs v_state from A's last
    step into B's first step unweighted, so the 20 MWh charged once in A (weight 1) serve the load of B in all three
    base periods it stands for: weighted, the solve delivers three times what it charged (the timestep weights are
    the representative weights × blocks / base periods: 0.5 for A, 1.5 for B), and leaves no load unserved.  The
    run solves and prints one warning line naming the node and the blended variant.  The blended variant links A
    and B by their weights on the base-period chain: what the tank delivers is at most what it was charged with,
    and it prints no such warning."""
    out: list = []
    plain = _run_rp(tmp_path / "plain", method, out=out)["rp_solve"]
    charged, delivered = _weighted_mwh(plain, "gen"), _weighted_mwh(plain, "dis")
    assert charged == pytest.approx(0.5 * 20.0, abs=1e-6)
    assert delivered == pytest.approx(1.5 * 20.0, abs=1e-6)
    lines = [ln for ln in out[0].splitlines() if "without their weights" in ln]
    assert len(lines) == 1, out[0][-2000:]
    assert lines[0].startswith("Warning: solve 'rp_solve' is on representative periods (timeset(s) rp)")
    assert f"1 storage node (tank) use a storage_binding_method (e.g. tank: {method})" in lines[0]
    assert "equal weights" in lines[0] and f"{method} -> {BLENDED[method]}" in lines[0]
    blended = _run_rp(tmp_path / "blended", BLENDED[method], out=out)["rp_solve"]
    assert "without their weights" not in out[1]
    assert _weighted_mwh(blended, "dis") <= _weighted_mwh(blended, "gen") + 1e-6
    assert blended.solution.obj > plain.solution.obj


@pytest.mark.parametrize("method", ["bind_within_timeblock", "bind_within_solve_blended_weights",
                                    "bind_within_period_blended_weights"])
def test_weighted_or_per_period_cycles_run(tmp_path, method):
    """bind_within_timeblock cycles within each representative period (one timeset block each), so energy is not
    carried between them; the blended variants carry it by the weights."""
    step = _run_rp(tmp_path, method)["rp_solve"]
    assert _weighted_mwh(step, "dis") <= _weighted_mwh(step, "gen") + 1e-6


def test_rolling_solve_on_representative_periods_is_an_error(tmp_path):
    rolling = _rp_solve(solve_mode="rolling_window", rolling_solve_jump=8.0, rolling_solve_horizon=12.0)
    with pytest.raises(FlexToolConfigError) as exc:
        _run_rp(tmp_path, "bind_within_solve_blended_weights", rolling)
    assert ("Solve 'rp_solve' is a rolling_window solve on representative periods (timeset(s) rp have "
            "representative_period_weights): rolling over representative periods is not supported.") in str(exc.value)


@pytest.mark.parametrize("penalty_method", ["off", "regular"])
def test_storage_slack_beside_a_hard_balance_is_warned_once(tmp_path, capsys, penalty_method):
    """The tank (balance slack on) feeds ``load`` through ``dis``: with penalty_method off on ``load`` the tank's
    slack can serve the load at the tank's penalty, one warning line for the rolling solve (its four rolls) names
    how many storage nodes and the first pair."""
    from flextool.engine import run_chain_from_db
    from flextool.update_flextool.db_handle import open_database_mapping

    solve = {"solve_mode": "rolling_window", "rolling_solve_jump": 4.0, "rolling_solve_horizon": 8.0,
             "rolling_duration": 16.0, "period_timeset": T.tmap({"p1": "full"}, "period"),
             "years_represented": T.tmap({"p1": 1.0}, "period"), "realized_periods": T.arr(["p1"])}
    url = T.build_db(tmp_path, {"st": solve}, top=["st"], tank={}, load={t: 5.0 for t in ("t05", "t06")},
                     gen={t: 1.0 for t in ("t01", "t02")})
    with open_database_mapping(url) as db:
        _count, errors = api.import_data(db, parameter_values=[
            ("node", "load", "penalty_method", penalty_method, T.ALT)])
        assert not errors, errors
        db.commit_session("penalty method")
    run_chain_from_db(url, T.SCENARIO, work_folder=tmp_path / "work", keep_solutions=True)
    out = capsys.readouterr().out
    hits = [line for line in out.splitlines() if "linked by a unit or a connection to node(s) with penalty_method off"
            in line]
    if penalty_method == "regular":
        assert hits == []
        return
    assert len(hits) == 1, hits
    assert hits[0].startswith("Warning: in solve 'st' 1 storage node (tank)")
    assert "(e.g. tank to load)" in hits[0]
