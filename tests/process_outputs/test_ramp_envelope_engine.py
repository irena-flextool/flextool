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

from tests.engine._unit_db import b64, make_db

pytestmark = pytest.mark.solver

REPO = Path(__file__).resolve().parents[2]
STEPS = ["t0001", "t0002", "t0003", "t0004"]
UP = ("units_up", "storage_up", "vre_up", "connections_up")
DOWN = ("units_down", "storage_down", "vre_down", "connections_down")


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _series(values: list[float]) -> list:
    return b64({"index_type": "str", "index_name": "time", "data": [[t, x] for t, x in zip(STEPS, values)]}, "map")


def _float(x: float) -> list:
    return b64(x, "float")


def _str(x: str) -> list:
    return b64(x, "str")


def _four_steps() -> list:
    return [["timeset", "2day", "timeset_duration", b64({"index_type": "str", "data": [["t0001", 4.0]]}, "map")]]


def _envelope(tmp_path_factory, alt: str, entities: list, values: list) -> dict[str, pd.DataFrame]:
    """Solve the case and write the two envelope tables as parquet; ``{table: frame}`` with (period, time) rows and
    (entity, band) columns."""
    from flextool.engine import run_chain_from_db
    from flextool.lean_parquet import read_lean_parquet
    from flextool.process_outputs.out_group import nodeGroup_ramp_envelope
    from flextool.process_outputs.out_node import node_ramp_envelope
    from flextool.process_outputs.write_outputs import write_outputs

    url, scen = make_db(tmp_path_factory, alt=alt, values=_four_steps() + values, base_chain=["init"],
                        entities=entities)
    work = tmp_path_factory.mktemp(f"{alt}_w")
    with contextlib.redirect_stdout(io.StringIO()):
        steps = run_chain_from_db(url, scen, work_folder=work, keep_solutions=True)
        last = next(reversed(steps.values()))
        write_outputs(scen, output_location=str(work / "out"), subdir=scen,
                      output_config_path=str(REPO / "flextool" / "schemas" / "default_plots.yaml"),
                      output_funcs=[node_ramp_envelope, nodeGroup_ramp_envelope],
                      write_methods=["parquet"], fallback_output_location=str(work),
                      raw_output_dir=str(work / "output_raw"), solution=last.solution,
                      solve_name=last.solve_name, flex_data=last.flex_data,
                      flex_data_provider=last.flex_data_provider)
    out = {}
    for path in (work / "out" / "output_parquet" / scen).glob("*.parquet"):
        frame = read_lean_parquet(path)
        if isinstance(frame.columns, pd.MultiIndex) and "scenario" in frame.columns.names:
            frame = frame.droplevel("scenario", axis=1)
        out[path.stem] = frame
    return out


def _rows(frame: pd.DataFrame, entity: str) -> pd.DataFrame:
    """The bands of ``entity`` with the rows indexed by time alone."""
    sub = frame[entity]
    sub.index = sub.index.get_level_values("time")
    return sub


def _check(frame: pd.DataFrame, entity: str, expected: dict[str, list[float]]) -> None:
    """Rows t0001..t0003 equal ``expected`` (band -> three values; bands not given are 0), row t0004 empty."""
    sub = _rows(frame, entity)
    for band in ("ramp_net_load", "ramp_demand", *UP, *DOWN):
        want = expected.get(band, [0.0, 0.0, 0.0])
        got = sub.loc[STEPS[:3], band].to_numpy(dtype=float)
        assert got == pytest.approx(want, abs=1e-6), (entity, band, list(got))
    assert sub.loc[STEPS[3]].isna().all(), (entity, "the last step has no successor")


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
            ["unit", "G", "conversion_method", _str("none")], ["unit", "G", "startup_method", _str("binary")],
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
