"""Ramp envelope end to end: JSON fixture → solve → outputs (``node_ramp_envelope_dt_ep``).

The cases and every expected number are the hand-computed oracle of ``specs/ramp_envelope_oracle_v2.md`` (Cases 7,
9, 10; each LP optimum is unique there).  They prove the wiring of the per-solve inputs (capacity multiplier,
floor, ramp-limit sets, shut-downs, reserves at t⁺, availability at t⁺, per-step efficiency) from the solve to the
output table.  Built on ``tests.json`` ``init`` (one period p2020 of the ``2day`` timeset, cut to four one-hour
steps) plus one override alternative; never a checked-in ``.sqlite``.
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.engine._unit_db import b64, is_realization, make_db

pytestmark = pytest.mark.solver

REPO = Path(__file__).resolve().parents[2]
SOLVE = "y2020_2day_dispatch"
UP = ("units_up", "storage_up", "vre_up", "connections_up")
DOWN = ("units_down", "storage_down", "vre_down", "connections_down")


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _steps(n: int) -> list[str]:
    return [f"t{i:04d}" for i in range(1, n + 1)]


def _series(values: list[float]) -> list:
    return b64({"index_type": "str", "index_name": "time", "data": [[t, x] for t, x in zip(_steps(len(values)),
                                                                                         values)]}, "map")


def _float(x: float) -> list:
    return b64(x, "float")


def _str(x: str) -> list:
    return b64(x, "str")


def _n_steps(n: int) -> list:
    """The ``2day`` timeset cut to its first ``n`` one-hour steps."""
    return [["timeset", "2day", "timeset_duration", b64({"index_type": "str", "data": [["t0001", float(n)]]}, "map")]]


def _rolling(horizon: float, jump: float) -> list:
    return [["solve", SOLVE, "solve_mode", _str("rolling_window")],
            ["solve", SOLVE, "rolling_solve_horizon", _float(horizon)],
            ["solve", SOLVE, "rolling_solve_jump", _float(jump)]]


#: Solved cases of this module: ``alt -> (steps, work folder, scenario)``; freed at module teardown.
_RUNS: dict[str, tuple[dict, Path, str]] = {}


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()


def _solve(tmp_path_factory, alt: str, entities: list, values: list, n_steps: "int | None" = 4,
           fixture_json: "Path | None" = None, base_chain: "list[str] | None" = None) -> tuple[dict, Path, str]:
    """Solve the case once per module (``_RUNS``), the solutions kept; ``n_steps`` None keeps the timesets of the
    base chain."""
    from flextool.engine import run_chain_from_db

    if alt not in _RUNS:
        extra = {} if fixture_json is None else {"fixture_json": fixture_json}
        url, scen = make_db(tmp_path_factory, alt=alt, values=(_n_steps(n_steps) if n_steps else []) + values,
                            base_chain=base_chain or ["init"], entities=entities, **extra)
        work = tmp_path_factory.mktemp(f"{alt}_w")
        with contextlib.redirect_stdout(io.StringIO()):
            steps = run_chain_from_db(url, scen, work_folder=work, keep_solutions=True)
        _RUNS[alt] = (steps, work, scen)
    return _RUNS[alt]


def _envelope(tmp_path_factory, alt: str, entities: list, values: list, n_steps: int = 4,
              **kwargs) -> dict[str, pd.DataFrame]:
    """Solve the case and write the two envelope tables as parquet; ``{table: frame}`` with (period, time) rows and
    (entity, band) columns."""
    from flextool.lean_parquet import read_lean_parquet
    from flextool.process_outputs.out_group import nodeGroup_ramp_envelope
    from flextool.process_outputs.out_node import node_ramp_envelope
    from flextool.process_outputs.write_outputs import write_outputs

    steps, work, scen = _solve(tmp_path_factory, alt, entities, values, n_steps, **kwargs)
    last = next(reversed(steps.values()))
    out_dir = work / "out"
    if not (out_dir / "output_parquet" / scen).is_dir():
        with contextlib.redirect_stdout(io.StringIO()):
            write_outputs(scen, output_location=str(out_dir), subdir=scen,
                          output_config_path=str(REPO / "flextool" / "schemas" / "default_plots.yaml"),
                          output_funcs=[node_ramp_envelope, nodeGroup_ramp_envelope],
                          write_methods=["parquet"], fallback_output_location=str(work),
                          raw_output_dir=str(work / "output_raw"), solution=last.solution,
                          solve_name=last.solve_name, flex_data=last.flex_data,
                          flex_data_provider=last.flex_data_provider)
    out = {}
    for path in (out_dir / "output_parquet" / scen).glob("*.parquet"):
        frame = read_lean_parquet(path)
        if isinstance(frame.columns, pd.MultiIndex) and "scenario" in frame.columns.names:
            frame = frame.droplevel("scenario", axis=1)
        out[path.stem] = frame
    return out


def _results(tmp_path_factory, alt: str, entities: list, values: list, n_steps: int = 4, **kwargs):
    """``(par, s, r)`` of the solved case, read and post-processed as the output writer does."""
    from flextool.process_outputs.process_results import post_process_results
    from flextool.process_outputs.write_outputs import _read_outputs

    steps, work, _ = _solve(tmp_path_factory, alt, entities, values, n_steps, **kwargs)
    last = next(reversed(steps.values()))
    with contextlib.redirect_stdout(io.StringIO()):
        par, s, v = _read_outputs(str(work / "output_raw"), flex_data=last.flex_data, solution=last.solution,
                                  solve_name=last.solve_name, flex_data_provider=last.flex_data_provider)
        r = post_process_results(par, s, v)
    return par, s, r


def _rows(frame: pd.DataFrame, entity: str) -> pd.DataFrame:
    """The bands of ``entity`` with the rows indexed by time alone."""
    sub = frame[entity]
    sub.index = sub.index.get_level_values("time")
    return sub


def _check(frame: pd.DataFrame, entity: str, expected: dict[str, list[float]]) -> None:
    """Every row but the last equals ``expected`` (band -> values; bands not given are 0), the last row (no
    successor) is empty."""
    sub = _rows(frame, entity)
    n = len(next(iter(expected.values())))
    steps = _steps(n + 1)
    assert list(sub.index) == steps, (entity, list(sub.index))
    for band in ("ramp_net_load", "ramp_demand", *UP, *DOWN):
        want = expected.get(band, [0.0] * n)
        got = sub.loc[steps[:n], band].to_numpy(dtype=float)
        assert got == pytest.approx(want, abs=1e-6), (entity, band, list(got))
    assert sub.loc[steps[n]].isna().all(), (entity, "the last step has no successor")


# ── Case 9: a connection holding up and down reserve at the node (spec §2.5, CHG-13) ─────────────────────────


def _case9() -> tuple[list, list]:
    ents = [["node", "n", None], ["node", "m", None],
            ["unit", "A", None], ["unit", "Bm", None],
            ["unit__outputNode", ["A", "n"], None], ["unit__outputNode", ["Bm", "m"], None],
            ["connection", "c", None], ["connection__node__node", ["c", "n", "m"], None],
            ["upDown", "up", None], ["upDown", "down", None],
            ["reserve", "res_up", None], ["reserve", "res_dn", None],
            ["group", "g_n", None], ["group__node", ["g_n", "n"], None],
            ["reserve__upDown__group", ["res_up", "up", "g_n"], None],
            ["reserve__upDown__group", ["res_dn", "down", "g_n"], None],
            ["reserve__upDown__connection__node", ["res_up", "up", "c", "n"], None],
            ["reserve__upDown__connection__node", ["res_dn", "down", "c", "n"], None]]
    rel = "reserve__upDown__connection__node"
    vals = [["node", "n", "node_type", _str("balance")], ["node", "n", "inflow", _series([-30, -40, -70, -50])],
            ["node", "m", "node_type", _str("balance")], ["node", "m", "inflow", _series([-60, -30, -40, -80])],
            ["unit", "A", "existing", _float(100.0)], ["unit", "A", "conversion_method", _str("none")],
            ["unit__outputNode", ["A", "n"], "other_operational_cost", _float(10.0)],
            ["unit", "Bm", "existing", _float(200.0)], ["unit", "Bm", "conversion_method", _str("none")],
            ["unit__outputNode", ["Bm", "m"], "other_operational_cost", _float(50.0)],
            ["connection", "c", "existing", _float(50.0)],
            ["connection", "c", "transfer_method", _str("no_losses_no_variable_cost")],
            ["reserve__upDown__group", ["res_up", "up", "g_n"], "reserve_method", _str("timeseries_only")],
            ["reserve__upDown__group", ["res_up", "up", "g_n"], "reservation", _float(20.0)],
            ["reserve__upDown__group", ["res_dn", "down", "g_n"], "reserve_method", _str("timeseries_only")],
            ["reserve__upDown__group", ["res_dn", "down", "g_n"], "reservation", _series([0, 20, 20, 0])],
            [rel, ["res_up", "up", "c", "n"], "is_enabled", _str("yes")],
            [rel, ["res_up", "up", "c", "n"], "max_share", _float(0.4)],
            [rel, ["res_up", "up", "c", "n"], "reliability", _float(1.0)],
            [rel, ["res_dn", "down", "c", "n"], "is_enabled", _str("yes")],
            [rel, ["res_dn", "down", "c", "n"], "max_share", _float(1.0)],
            [rel, ["res_dn", "down", "c", "n"], "reliability", _float(1.0)]]
    return ents, vals


def test_case9_connection_reserve_at_the_node(tmp_path_factory) -> None:
    """Oracle Case 9: the connection band subtracts the reserve the connection holds at the node at t⁺ (up from
    up, down from down); the far node m does not see it.  n t1 conn_down raw −20 (the 20 MW down reserve at t2
    needs the export cut) → 0."""
    ents, vals = _case9()
    tables = _envelope(tmp_path_factory, "re_case9", ents, vals)
    env = tables["node_ramp_envelope_dt_ep"]
    _check(env, "n", {"ramp_net_load": [10, 30, -20], "ramp_demand": [10, 30, -20],
                      "units_up": [20, 30, 0], "connections_up": [80, 60, 60],
                      "units_down": [-80, -70, -100], "connections_down": [0, 0, -20]})
    _check(env, "m", {"ramp_net_load": [-30, 10, 40], "ramp_demand": [-30, 10, 40],
                      "units_up": [190, 200, 190], "connections_up": [0, 20, 20],
                      "units_down": [-10, 0, -10], "connections_down": [-100, -80, -80]})
    hours = tables["timestep_hours_dt"]
    assert hours.to_numpy(dtype=float).ravel() == pytest.approx(np.full(4, 1.0 / (4.0 / 8760.0)))
    # One-hour steps: each transition is (1 + 1)/2 = 1 h; the last step has none.
    trans = tables["transition_hours_dt"].to_numpy(dtype=float).ravel()
    assert trans[:3].tolist() == [1.0, 1.0, 1.0] and np.isnan(trans[3])


# ── Case 7: start-up and shut-down at t⁺, reserve at t⁺, availability at t⁺, input-side reserve (φ) ──────────


def _case7() -> tuple[list, list]:
    ents = [["node", "n", None], ["node", "h", None],
            ["unit", "G", None], ["unit", "B", None], ["unit", "HP", None],
            ["unit__outputNode", ["G", "n"], None], ["unit__outputNode", ["B", "n"], None],
            ["unit__inputNode", ["HP", "n"], None], ["unit__outputNode", ["HP", "h"], None],
            ["upDown", "up", None], ["reserve", "res_up", None],
            ["group", "g_n", None], ["group__node", ["g_n", "n"], None],
            ["reserve__upDown__group", ["res_up", "up", "g_n"], None],
            ["reserve__upDown__unit__node", ["res_up", "up", "HP", "n"], None]]
    rel = "reserve__upDown__unit__node"
    vals = [["node", "n", "node_type", _str("balance")], ["node", "n", "inflow", _series([-33, -70, -70, -40])],
            ["node", "h", "node_type", _str("balance")], ["node", "h", "inflow", _series([-20, -40, -30, -20])],
            ["unit", "G", "virtual_unitsize", _float(50.0)], ["unit", "G", "existing", _float(150.0)],
            ["unit", "G", "conversion_method", _str("none")], ["unit", "G", "startup_method", _str("integer_fix_in_uc")],
            ["unit", "G", "min_load", _float(0.8)], ["unit", "G", "startup_cost", _float(1.0)],
            ["unit__outputNode", ["G", "n"], "other_operational_cost", _float(20.0)],
            ["unit__outputNode", ["G", "n"], "ramp_method", _str("ramp_limit")],
            ["unit__outputNode", ["G", "n"], "ramp_speed_up", _float(0.002)],
            ["unit__outputNode", ["G", "n"], "ramp_speed_down", _float(0.002)],
            ["unit", "B", "existing", _float(100.0)], ["unit", "B", "conversion_method", _str("none")],
            ["unit", "B", "availability", _series([1.0, 1.0, 0.5, 0.5])],
            ["unit__outputNode", ["B", "n"], "other_operational_cost", _float(60.0)],
            ["unit", "HP", "existing", _float(100.0)], ["unit", "HP", "efficiency", _float(2.0)],
            ["reserve__upDown__group", ["res_up", "up", "g_n"], "reserve_method", _str("timeseries_only")],
            ["reserve__upDown__group", ["res_up", "up", "g_n"], "reservation", _series([10, 20, 15, 10])],
            [rel, ["res_up", "up", "HP", "n"], "is_enabled", _str("yes")],
            [rel, ["res_up", "up", "HP", "n"], "max_share", _float(1.0)],
            [rel, ["res_up", "up", "HP", "n"], "reliability", _float(1.0)]]
    return ents, vals


def test_case7_online_changes_reserve_and_availability_at_next_step(tmp_path_factory) -> None:
    """Oracle Case 7 (unique MIP optimum: online 1, 2, 2, 1; start-up at t2, shut-down at t4).  Units online in
    both steps (row t1: 2 − 1 = 1 unit); the stopping unit's output at t3 is forced movement (f₂ = 85 − 0.8·50 =
    45: G up 5 / down 5); B's availability at t⁺ (row t2: 50); HP's consumption-cut reserve at t⁺ through φ = 2
    (n row t1 up raw −10 → 0; h row t1 down raw −20 → 0)."""
    ents, vals = _case7()
    env = _envelope(tmp_path_factory, "re_case7", ents, vals)["node_ramp_envelope_dt_ep"]
    _check(env, "n", {"ramp_net_load": [37, 0, -30], "ramp_demand": [37, 0, -30],
                      "units_up": [106, 65, 60], "units_down": [-43, -40, -40]})
    _check(env, "h", {"ramp_net_load": [20, -10, -10], "ramp_demand": [20, -10, -10],
                      "units_up": [80, 60, 70], "units_down": [0, -10, -10]})


# ── Case 10: direct unit with a time-varying efficiency, input-side reserve and a ramp limit (CHG-12) ────────


def _case10() -> tuple[list, list]:
    ents = [["node", "e", None], ["node", "h", None],
            ["unit", "HP", None], ["unit", "G", None],
            ["unit__inputNode", ["HP", "e"], None], ["unit__outputNode", ["HP", "h"], None],
            ["unit__outputNode", ["G", "e"], None],
            ["upDown", "up", None], ["reserve", "res_up", None],
            ["group", "g_e", None], ["group__node", ["g_e", "e"], None],
            ["reserve__upDown__group", ["res_up", "up", "g_e"], None],
            ["reserve__upDown__unit__node", ["res_up", "up", "HP", "e"], None]]
    rel = "reserve__upDown__unit__node"
    vals = [["node", "e", "node_type", _str("balance")], ["node", "e", "inflow", _float(-20.0)],
            ["node", "h", "node_type", _str("balance")], ["node", "h", "inflow", _series([-40, -60, -50, -30])],
            ["unit", "HP", "existing", _float(100.0)], ["unit", "HP", "efficiency", _series([2.0, 4.0, 2.5, 2.0])],
            ["unit__outputNode", ["HP", "h"], "ramp_method", _str("ramp_limit")],
            ["unit__outputNode", ["HP", "h"], "ramp_speed_up", _float(0.005)],
            ["unit__outputNode", ["HP", "h"], "ramp_speed_down", _float(0.005)],
            ["unit", "G", "existing", _float(100.0)], ["unit", "G", "conversion_method", _str("none")],
            ["unit__outputNode", ["G", "e"], "other_operational_cost", _float(10.0)],
            ["reserve__upDown__group", ["res_up", "up", "g_e"], "reserve_method", _str("timeseries_only")],
            ["reserve__upDown__group", ["res_up", "up", "g_e"], "reservation", _float(10.0)],
            ["reserve__upDown__group", ["res_up", "up", "g_e"], "penalty_reserve", _float(1000.0)],
            [rel, ["res_up", "up", "HP", "e"], "is_enabled", _str("yes")],
            [rel, ["res_up", "up", "HP", "e"], "max_share", _float(0.2)],
            [rel, ["res_up", "up", "HP", "e"], "reliability", _float(1.0)]]
    return ents, vals


def test_case10_time_varying_efficiency_converts_each_step(tmp_path_factory) -> None:
    """Oracle Case 10: the input arc of a direct unit converts each step with its own slope (slope 0.5, 0.25,
    0.4, 0.5; reserve 10, 5, 8, 5 MW at e): e up 12.5, 0 (raw −5: the efficiency drop raises the draw), 5 and down
    0 (raw −2.5), 21, 20 from HP; h the sink arc with the ramp-with-reserve room."""
    ents, vals = _case10()
    env = _envelope(tmp_path_factory, "re_case10", ents, vals)["node_ramp_envelope_dt_ep"]
    _check(env, "e", {"units_up": [72.5, 65, 65], "units_down": [-40, -56, -60]})
    _check(env, "h", {"ramp_net_load": [20, -10, -20], "ramp_demand": [20, -10, -20],
                      "units_up": [30, 30, 30], "units_down": [-10, -10, -20]})


# ── Case 8a: a connection with efficiency < 1 and an availability drop at t⁺ (CHG-2, CHG-4) ─────────────────


def _case8a() -> tuple[list, list]:
    ents = [["node", "n", None], ["node", "m", None],
            ["unit", "A", None], ["unit", "E", None], ["unit", "Bm", None],
            ["unit__outputNode", ["A", "n"], None], ["unit__outputNode", ["E", "n"], None],
            ["unit__outputNode", ["Bm", "m"], None],
            ["connection", "c", None], ["connection__node__node", ["c", "n", "m"], None]]
    vals = [["node", "n", "node_type", _str("balance")], ["node", "n", "inflow", _series([-30, -80, -95, -60])],
            ["node", "m", "node_type", _str("balance")], ["node", "m", "inflow", _series([-60, -30, -10, -20])],
            ["unit", "A", "existing", _float(100.0)], ["unit", "A", "conversion_method", _str("none")],
            ["unit", "A", "availability", _series([1.0, 1.0, 0.5, 1.0])],
            ["unit__outputNode", ["A", "n"], "other_operational_cost", _float(10.0)],
            ["unit", "E", "existing", _float(100.0)], ["unit", "E", "conversion_method", _str("none")],
            ["unit__outputNode", ["E", "n"], "other_operational_cost", _float(100.0)],
            ["unit", "Bm", "existing", _float(200.0)], ["unit", "Bm", "conversion_method", _str("none")],
            ["unit__outputNode", ["Bm", "m"], "other_operational_cost", _float(50.0)],
            ["connection", "c", "existing", _float(40.0)], ["connection", "c", "efficiency", _float(0.8)]]
    return ents, vals


#: Oracle Case 8a, both nodes (rows t1..t3; t4 empty).
CASE_8A = {
    "n": {"ramp_net_load": [50, 15, -35], "ramp_demand": [50, 15, -35],
          "units_up": [120, 100, 145], "connections_up": [90, 60, 0],
          "units_down": [-80, -100, -55], "connections_down": [0, -30, -90]},
    "m": {"ramp_net_load": [-30, -20, 10], "ramp_demand": [-30, -20, 10],
          "units_up": [180, 186, 140], "connections_up": [0, 24, 90],
          "units_down": [-20, -14, -60], "connections_down": [-90, -66, 0]},
}


def test_case8a_lossy_connection_and_availability_drop(tmp_path_factory) -> None:
    """Oracle Case 8a: the connection's node-side range is [−cap·slope, +cap] = [−50, +40] (losses on the drawing
    side): n t1 exports 50 MW drawn → up 90, down 0; A's availability falls to 0.5 at t3, so n t2 A up raw
    50 − 100 = −50 → 0 (forced movement)."""
    ents, vals = _case8a()
    env = _envelope(tmp_path_factory, "re_case8a", ents, vals)["node_ramp_envelope_dt_ep"]
    for node, expected in CASE_8A.items():
        _check(env, node, expected)


def test_rolling_seam_row_reads_the_next_roll(tmp_path_factory) -> None:
    """Case 8a as a rolling solve (horizon 3 h, jump 2 h: roll 0 realizes t1, t2, roll 1 t3, t4).  Every step's
    optimum is independent of the others, so the dispatch is the single solve's, and the seam row t2 is filled from
    roll 1's t3 (A's availability 0.5 at t3: n t2 units_up 100, the connection's flows at t2 and the range at t3)
    with the Case 8a values; the last realized step t4 stays empty."""
    ents, vals = _case8a()
    steps, _, _ = _solve(tmp_path_factory, "re_case8a_roll", ents, vals + _rolling(3.0, 2.0))
    assert len(steps) == 2, list(steps)
    env = _envelope(tmp_path_factory, "re_case8a_roll", ents, vals + _rolling(3.0, 2.0))["node_ramp_envelope_dt_ep"]
    for node, expected in CASE_8A.items():
        _check(env, node, expected)


# ── Cases 8b / 8c: ramp-limited wind; heavy curtailment then a large potential drop (CHG-5, CHG-6) ──────────


def _case8b(reduce: "str | None" = None) -> tuple[list, list]:
    ents = [["node", "n", None], ["unit", "W", None], ["unit", "G", None],
            ["unit__outputNode", ["W", "n"], None], ["unit__outputNode", ["G", "n"], None],
            ["profile", "wp", None], ["unit__node__profile", ["W", "n", "wp"], None]]
    vals = [["node", "n", "node_type", _str("balance")], ["node", "n", "inflow", _float(-50.0)],
            ["unit", "W", "existing", _float(100.0)], ["unit", "W", "conversion_method", _str("none")],
            ["profile", "wp", "profile", _series([0.0, 0.6, 0.9, 0.2, 0.2])],
            ["unit__node__profile", ["W", "n", "wp"], "profile_method", _str("upper_limit")],
            ["unit__outputNode", ["W", "n"], "ramp_method", _str("ramp_limit")],
            ["unit__outputNode", ["W", "n"], "ramp_speed_up", _float(0.005)],
            ["unit__outputNode", ["W", "n"], "ramp_speed_down", _float(0.005)],
            ["unit", "G", "existing", _float(100.0)], ["unit", "G", "conversion_method", _str("none")],
            ["unit__outputNode", ["G", "n"], "other_operational_cost", _float(50.0)]]
    if reduce is not None:
        vals.append(["solve", SOLVE, "reduce_unit_detail_method", _str(reduce)])
    return ents, vals


def test_case8b_ramp_limited_wind(tmp_path_factory) -> None:
    """Oracle Case 8b (W = 0, 30, 50, 20, 20; potential 0, 60, 90, 20, 20): the wind cannot follow the sunrise-like
    rise at t1 (raw vre_up −30 → 0) and at t3 its 40 MW curtailment exceeds what the drop to 20 MW leaves (raw
    vre_down −40 → 0); t3 vre_up 40 = the curtailment released."""
    ents, vals = _case8b()
    env = _envelope(tmp_path_factory, "re_case8b", ents, vals, n_steps=5)["node_ramp_envelope_dt_ep"]
    _check(env, "n", {"ramp_net_load": [-60, -30, 70, 0], "ramp_demand": [0, 0, 0, 0],
                      "units_up": [50, 80, 100, 70], "vre_up": [0, 0, 40, 0],
                      "units_down": [-50, -20, 0, -30], "vre_down": [-60, -60, 0, -20]})


def test_case8c_integers_ramps_drops_the_ramp_cap(tmp_path_factory) -> None:
    """Oracle Case 8c: the same data in a solve with ``reduce_unit_detail_method = integers_ramps`` (the ramp limits
    left out): W = min(pot, 50) = 0, 50, 50, 20, 20 and no ramp cap on the wind's room (t2 vre_up 10 = the
    curtailment, vre_down 80)."""
    ents, vals = _case8b("integers_ramps")
    env = _envelope(tmp_path_factory, "re_case8c", ents, vals, n_steps=5)["node_ramp_envelope_dt_ep"]
    _check(env, "n", {"ramp_net_load": [-60, -30, 70, 0], "ramp_demand": [0, 0, 0, 0],
                      "units_up": [50, 100, 100, 70], "vre_up": [0, 10, 40, 0],
                      "units_down": [-50, 0, 0, -30], "vre_down": [-60, -80, 0, -20]})


# ── A time-varying input given as a constant and as an equivalent time map ─────────────────────────────────────


def _case10_stochastic() -> tuple[list, list]:
    """Case 10 in a stochastic solve (the scaffolding of Case 4b): trunk p2020 = {t1, t2}, fan p2025 = {t3, t4},
    forecasts hi and lo of h's demand, the realized designation ``real`` with hi's data."""
    ents, vals = _case10()
    ents = ents + [["group", "stoch", None], ["group__node", ["stoch", "h"], None],
                   ["timeset", "trunk", None], ["timeset", "fan", None]]
    branches = {"index_type": "str", "index_name": "period", "data": [["p2025", _map("branch", [
        [b, _map("analysis_time", [["t0003", _map("realized", [[yes_no, 1.0]])]])]
        for b, yes_no in (("real", "yes"), ("hi", "no"), ("lo", "no"))])]]}
    vals = [v for v in vals if not (v[1] == "h" and v[2] == "inflow")] + [
        ["node", "h", "inflow", _stochastic({"real": {"t0001": [-40, -60, -50, -30], "t0003": [-50, -30]},
                                             "hi": {"t0003": [-50, -30]}, "lo": {"t0003": [-45, -25]}})],
        ["group", "stoch", "include_stochastics", _str("yes")],
        ["timeset", "trunk", "timeline", _str("y2020")], ["timeset", "fan", "timeline", _str("y2020")],
        ["timeset", "trunk", "timeset_duration", b64({"index_type": "str", "data": [["t0001", 2.0]]}, "map")],
        ["timeset", "fan", "timeset_duration", b64({"index_type": "str", "data": [["t0003", 2.0]]}, "map")],
        ["solve", SOLVE, "period_timeset",
         b64({"index_type": "str", "index_name": "period", "data": [["p2020", "trunk"], ["p2025", "fan"]]}, "map")],
        ["solve", SOLVE, "realized_periods",
         b64({"value_type": "str", "index_name": "period", "data": ["p2020", "p2025"]}, "array")],
        ["solve", SOLVE, "stochastic_branches", b64(branches, "map")],
        # The forecasts are solved: deterministic, the default since 4.7.0, would ignore them.
        ["solve", SOLVE, "dispatch_realization_method", _str("within_solve")]]
    return ents, vals


@pytest.mark.parametrize("solve", ["single", "rolling", "stochastic"])
def test_constant_and_equivalent_map_efficiency_give_the_same_envelope(tmp_path_factory, solve: str) -> None:
    """Case 10 with HP's efficiency 2 given as a constant and as a time map of 2s: the same LP, so the same
    envelope tables (both nodes, every band and row), in a single solve, a rolling solve (3-step horizon, 2-step
    jump: a roll seam) and a stochastic solve."""
    if solve == "stochastic":
        ents, vals = _case10_stochastic()
    else:
        ents, vals = _case10()
        if solve == "rolling":
            vals = vals + _rolling(3, 2)
    const = [v for v in vals if v[2] != "efficiency"] + [["unit", "HP", "efficiency", _float(2.0)]]
    tmap = [v for v in vals if v[2] != "efficiency"] + [["unit", "HP", "efficiency", _series([2.0] * 4)]]
    suffix = "" if solve == "single" else f"_{solve}"
    steps, _, _ = _solve(tmp_path_factory, f"re_eff_const{suffix}", ents, const)
    if solve == "rolling":
        assert len(steps) > 1, list(steps)
    if solve == "stochastic":
        assert any(is_realization(name) for name in steps), list(steps)
    a = _envelope(tmp_path_factory, f"re_eff_const{suffix}", ents, const)["node_ramp_envelope_dt_ep"]
    b = _envelope(tmp_path_factory, f"re_eff_map{suffix}", ents, tmap)["node_ramp_envelope_dt_ep"]
    assert a.notna().to_numpy().any()
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12, atol=1e-9)


# ── Case 4b: the step before a forecast fan (branch point), realized-branch output ───────────────────────────


def _map(name: str, data: list) -> dict:
    return {"type": "map", "index_type": "str", "index_name": name, "data": data}


def _stochastic(series: dict[str, dict[str, list[float]]]) -> list:
    """A stochastic-time map branch → analysis_time → time; ``series`` = {branch: {analysis time: values from that
    step on}}."""
    top = _map("branch", [[b, _map("analysis_time", [
        [at, _map("time", [[f"t{int(at[1:]) + i:04d}", x] for i, x in enumerate(xs)])] for at, xs in ats.items()])]
        for b, ats in series.items()])
    top.pop("type")
    return b64(top, "map")


def _case4b() -> tuple[list, list]:
    """Oracle Case 4b in FlexTool's stochastic form: trunk period p2020 = {t1, t2} and the fan at p2025 = {t3, t4}
    (continuing p2020 on the same timeline), forecasts hi and lo, the realized designation ``real`` with hi's data.
    The results hold the realization dispatch (p2020, then p2025 with the realized data), never the forecasts: the
    oracle's variant B.  Variant A (both branches visible: two successors → empty) is not in the normal results
    and stays pure-calc."""
    ents = [["node", "n", None], ["unit", "G1", None], ["unit", "PV", None],
            ["unit__outputNode", ["G1", "n"], None], ["unit__outputNode", ["PV", "n"], None],
            ["profile", "pp", None], ["unit__node__profile", ["PV", "n", "pp"], None],
            ["group", "stoch", None], ["group__node", ["stoch", "n"], None], ["group__unit", ["stoch", "PV"], None],
            ["timeset", "trunk", None], ["timeset", "fan", None]]
    branches = {"index_type": "str", "index_name": "period", "data": [["p2025", _map("branch", [
        [b, _map("analysis_time", [["t0003", _map("realized", [[yes_no, 1.0]])]])]
        for b, yes_no in (("real", "yes"), ("hi", "no"), ("lo", "no"))])]]}
    # PV capacity 50 MW: potential trunk 0, 10; hi (= real) 20, 10; lo 0, 0.  Demand trunk 50, 60; hi 80, 90; lo 40, 50.
    vals = [["node", "n", "node_type", _str("balance")],
            ["node", "n", "inflow", _stochastic({"real": {"t0001": [-50, -60, -80, -90], "t0003": [-80, -90]},
                                                 "hi": {"t0003": [-80, -90]}, "lo": {"t0003": [-40, -50]}})],
            ["profile", "pp", "profile", _stochastic({"real": {"t0001": [0.0, 0.2, 0.4, 0.2], "t0003": [0.4, 0.2]},
                                                      "hi": {"t0003": [0.4, 0.2]}, "lo": {"t0003": [0.0, 0.0]}})],
            ["unit__node__profile", ["PV", "n", "pp"], "profile_method", _str("upper_limit")],
            ["unit", "PV", "existing", _float(50.0)], ["unit", "PV", "conversion_method", _str("none")],
            ["unit", "G1", "existing", _float(100.0)], ["unit", "G1", "conversion_method", _str("none")],
            ["unit__outputNode", ["G1", "n"], "other_operational_cost", _float(50.0)],
            ["unit__outputNode", ["G1", "n"], "ramp_method", _str("ramp_limit")],
            ["unit__outputNode", ["G1", "n"], "ramp_speed_up", _float(0.005)],
            ["unit__outputNode", ["G1", "n"], "ramp_speed_down", _float(0.005)],
            ["group", "stoch", "include_stochastics", _str("yes")],
            ["timeset", "trunk", "timeline", _str("y2020")], ["timeset", "fan", "timeline", _str("y2020")],
            ["timeset", "trunk", "timeset_duration", b64({"index_type": "str", "data": [["t0001", 2.0]]}, "map")],
            ["timeset", "fan", "timeset_duration", b64({"index_type": "str", "data": [["t0003", 2.0]]}, "map")],
            ["solve", SOLVE, "period_timeset",
             b64({"index_type": "str", "index_name": "period", "data": [["p2020", "trunk"], ["p2025", "fan"]]}, "map")],
            ["solve", SOLVE, "realized_periods",
             b64({"value_type": "str", "index_name": "period", "data": ["p2020", "p2025"]}, "array")],
            ["solve", SOLVE, "stochastic_branches", b64(branches, "map")],
            # The forecasts are solved: deterministic, the default since 4.7.0, would ignore them.
            ["solve", SOLVE, "dispatch_realization_method", _str("within_solve")]]
    return ents, vals


def test_case4b_branch_point_row_reads_the_realized_branch(tmp_path_factory) -> None:
    """Oracle Case 4b variant B: the last trunk step (p2020, t2) has one realized successor, the realization
    dispatch's (p2025, t3), so its row is filled with the realized (hi) branch's t⁺ (PV potential 10 → 20, demand
    60 → 80): rnl 10, rdem 20, vre_down −20; the fan's last step is empty."""
    ents, vals = _case4b()
    steps, _, _ = _solve(tmp_path_factory, "re_case4b", ents, vals)
    assert any(is_realization(name) for name in steps), list(steps)
    env = _envelope(tmp_path_factory, "re_case4b", ents, vals)["node_ramp_envelope_dt_ep"]
    assert list(env.index) == [("p2020", "t0001"), ("p2020", "t0002"), ("p2025", "t0003"), ("p2025", "t0004")]
    _check(env, "n", {"ramp_net_load": [0, 10, 20], "ramp_demand": [10, 20, 10],
                      "units_up": [30, 30, 30], "units_down": [-30, -30, -30], "vre_down": [-10, -20, -10]})


# ── The envelope holds the realized net-load ramp (spec §2.10, test 7) ─────────────────────────────────────────


def _invariant(par, s, r, tol: float = 1e-6) -> tuple[int, list, dict]:
    """Check −Σdown ≤ ramp_net_load − Δslack ≤ Σup (Δslack = the change of unserved − excess energy, MW) on every
    filled row of every node and dispatch group, except where spec §2.10 does not promise it: rows with a start-up or
    a shut-down at t⁺ of a process with an arc at the entity, entities with an arc to a storage node that has an
    inflow or a self-discharge, and groups with a process inside them that does not convert 1:1 (efficiency ≠ 1 at
    some step, or a min-load section): its arcs count 0, so the change of its losses is in no band
    (``network_coal_wind_reserve``'s group exceeds the envelope by exactly the change of the internal connections'
    losses).  A group whose internal processes are lossless is checked.  Returns ``(rows checked, violations,
    {group: rows checked})``; t⁺ is read from ``s.ramp_link`` here,
    not from the envelope code."""
    rows = r.flow_dt.index
    link = s.ramp_link.to_frame(index=False)[["period", "time", "period_previous", "time_previous"]]
    link = link[~link.duplicated(["period_previous", "time_previous"], keep=False)]
    successor = pd.MultiIndex.from_frame(link[["period", "time"]])
    predecessor = pd.MultiIndex.from_frame(link[["period_previous", "time_previous"]], names=["period", "time"])

    def at_next(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.reindex(successor)
        out.index = predecessor
        return out.reindex(rows)

    def per_node(frame, columns) -> pd.DataFrame:
        if frame is None or frame.shape[1] == 0:
            return pd.DataFrame(0.0, index=rows, columns=pd.Index([]) if columns is None else columns)
        out = frame.copy()
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = out.columns.get_level_values(0)
        out = out.T.groupby(level=0).sum().T
        return out.reindex(index=rows, columns=out.columns if columns is None else columns, fill_value=0.0).fillna(0.0)

    nodes = pd.Index(sorted(set(s.process_source_sink.get_level_values("source"))
                            | set(s.process_source_sink.get_level_values("sink"))
                            | set(s.node_balance) | set(s.node_balance_period)))
    duration = par.step_duration.reindex(rows).to_numpy(dtype=float)[:, None]
    slack = (per_node(r.upward_node_slack_dt, nodes) - per_node(r.downward_node_slack_dt, nodes)) / duration
    moves = (per_node(r.process_startup_dt, None).add(per_node(r.process_shutdown_dt, None), fill_value=0.0))
    moves_next = at_next(moves).fillna(0.0)
    pss = s.process_source_sink.to_frame(index=False)
    ends = pd.concat([pss[["process", "source"]].set_axis(["process", "node"], axis=1),
                      pss[["process", "sink"]].set_axis(["process", "node"], axis=1)]).drop_duplicates()
    ends = ends[ends["node"] != ends["process"]]
    storage = set(s.node_state)
    leaky = set()
    for name in ("node_inflow", "node_self_discharge_loss"):
        frame = getattr(par, name, None)
        if frame is not None and frame.shape[1]:
            cols = frame.columns.get_level_values(0)
            leaky |= {n for n, nonzero in zip(cols, (frame.fillna(0.0).abs() > tol).any().to_numpy()) if nonzero}
    leaky &= storage
    members = {n: [n] for n in r.node_ramp_envelope_dt_ep.columns.get_level_values(0).unique()}
    gn = s.group_node.to_frame(index=False)
    groups = {g: sorted(set(gn.loc[gn["group"] == g, "node"]) - storage) for g in s.nodeGroupDispatch}
    nodes_of = ends.groupby("process")["node"].agg(set)
    lossy = set()
    for name, neutral in (("process_slope", 1.0), ("process_section", 0.0)):
        frame = getattr(par, name, None)
        if frame is not None and frame.shape[1]:
            cols = frame.columns.get_level_values(0)
            off = ((frame - neutral).abs() > tol).any().to_numpy()
            lossy |= {p for p, flag in zip(cols, off) if flag}
    groups = {g: members_g for g, members_g in groups.items()
              if not any(len(ns) > 1 and ns <= set(members_g) and p in lossy for p, ns in nodes_of.items())}
    tables = [(r.node_ramp_envelope_dt_ep, members), (r.nodeGroup_ramp_envelope_dt_gp, groups)]
    checked, bad, group_rows = 0, [], {}
    for table, entities in tables:
        for entity, entity_nodes in entities.items():
            procs = set(ends.loc[ends["node"].isin(entity_nodes), "process"])
            touched = set(ends.loc[ends["process"].isin(procs), "node"])
            if touched & leaky:
                continue
            env = table[entity]
            line = env["ramp_net_load"].to_numpy(dtype=float)
            up = env[list(UP)].sum(axis=1).to_numpy(dtype=float)
            down = -env[list(DOWN)].sum(axis=1).to_numpy(dtype=float)
            sl = slack.reindex(columns=entity_nodes, fill_value=0.0).sum(axis=1)
            d_slack = (at_next(sl.to_frame()).iloc[:, 0] - sl).to_numpy(dtype=float)
            stops = moves_next.reindex(columns=sorted(procs), fill_value=0.0).abs().sum(axis=1).to_numpy() > tol
            keep = ~np.isnan(line) & ~stops
            checked += int(keep.sum())
            if entities is groups:
                group_rows[entity] = int(keep.sum())
            net = line - d_slack
            for i in np.flatnonzero(keep & ((net > up + tol) | (net < -down - tol))):
                bad.append((entity, rows[i], -down[i], net[i], up[i]))
    return checked, bad, group_rows


#: Engine cases of this module (``alt -> case builder, steps``) and existing ``tests.json`` scenarios.
_SWEEP_CASES = {"re_case7": (_case7, 4), "re_case8a": (_case8a, 4), "re_case8b": (_case8b, 5),
                "re_case9": (_case9, 4), "re_case10": (_case10, 4), "re_case4b": (_case4b, 4)}
#: ``scenario -> (alternative chain, override values)``; the battery's self-discharge is set to 0 (§2.10 leaves a
#: storage node with self-discharge out); ``network_lossless`` is ``network_coal_wind_reserve`` with its three
#: internal connections at efficiency 1, so its multi-node dispatch group is checked too.
_LOSSLESS_NETWORK = [["connection", c, "efficiency", _float(1.0)] for c in ("east_north", "west_east", "west_north")]
_SWEEP_SCENARIOS = {
    "coal_ramp_limit": (["init", "west", "coal", "coal_ramp"], []),
    "coal_min_load_wind": (["init", "west", "coal", "coal_min_load", "wind"], []),
    "network_coal_wind_reserve": (["init", "west", "coal", "wind", "network", "reserve"], []),
    "network_lossless": (["init", "west", "coal", "wind", "network", "reserve"], _LOSSLESS_NETWORK),
    "wind_battery": (["init", "west", "wind", "battery"], [["node", "battery", "self_discharge_loss", _float(0.0)]]),
    "coal_chp": (["init", "west", "coal_chp", "heat"], []),
}


@pytest.mark.parametrize("alt", sorted(_SWEEP_CASES))
def test_line_within_envelope_engine_cases(tmp_path_factory, alt: str) -> None:
    """Spec §2.10 on the oracle's engine cases: the realized net-load ramp lies inside the envelope on every row
    that meets the conditions (Case 7's start-up row t1 and shut-down row t3 are left out)."""
    build, n = _SWEEP_CASES[alt]
    ents, vals = build()
    par, s, r = _results(tmp_path_factory, alt, ents, vals, n)
    checked, bad, _ = _invariant(par, s, r)
    assert checked > 0
    assert not bad, bad


@pytest.mark.parametrize("scenario", sorted(_SWEEP_SCENARIOS))
def test_line_within_envelope_existing_fixtures(tmp_path_factory, scenario: str) -> None:
    """Spec §2.10 on existing ``tests.json`` scenarios (ramp limits, online units with min load and wind, a network
    with reserves, a battery, a CHP), the 48-hour dispatch of each; the network's group is left out where its
    internal connections have losses (see :func:`_invariant`) and checked in the lossless variant."""
    chain, overrides = _SWEEP_SCENARIOS[scenario]
    par, s, r = _results(tmp_path_factory, f"re_sweep_{scenario}", [], overrides, None, base_chain=chain)
    checked, bad, group_rows = _invariant(par, s, r)
    assert checked > 0
    assert not bad, bad[:10]
    if scenario == "network_lossless":
        # The multi-node dispatch group is checked, not skipped.
        multi = {g for g in group_rows
                 if len(set(s.group_node.to_frame(index=False).query("group == @g")["node"])) > 1}
        assert multi and all(group_rows[g] > 0 for g in multi), group_rows
