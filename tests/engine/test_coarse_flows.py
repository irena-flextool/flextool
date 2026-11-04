"""Coarse processes: a unit or connection whose every balance node is coarse takes the coarse step of its nodes.

Owner decision 2026-11-03 ("Coarse new_stepduration groups"): a process between coarse nodes has one flow per
coarse step (a rate constant over the step); a process with a fine balance node keeps the solve's step, and so
does a process with a feature whose rows are per fine step (online variables, reserves, a delay, DC power flow,
an exact connection, minimum load, a user-constraint flow term, a flow-group instant limit, a capacity-margin /
inertia / non-synchronous group).

Fixture: ``tests/fixtures/lh2_three_region.json`` (h2_* / lh2_* in the 24-h ``daily_group``, elec_* hourly; the
liquefiers h2 -> lh2 and the pipes lh2 <-> lh2 are between daily nodes, the electrolysers elec -> h2 are not),
with override alternatives built from JSON under the pytest tmp dir.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl
import pytest

from flextool.engine.preprocess.params import _coarse
from tests.engine._unit_db import FLEXTOOL_ROOT, b64, last_stochastic, make_db, run, utf8

pytestmark = pytest.mark.solver

FIX = FLEXTOOL_ROOT / "tests" / "fixtures"
LH2 = FIX / "lh2_three_region.json"
LH2_COARSE = {"liquefier_A", "liquefier_B", "liquefier_C", "pipe_AB", "pipe_BC"}


def _chain(fixture: Path, scenario: str) -> list[str]:
    data = json.loads(fixture.read_text())
    rows = [r for r in data["scenario_alternatives"] if r[0] == scenario]
    nxt = {a: b for _, a, b in rows}
    alt = next(a for a in nxt if a not in set(nxt.values()))
    out = []
    while alt is not None:
        out.append(alt)
        alt = nxt.get(alt)
    return out


def _f(x) -> list:
    return b64(float(x), "float")


def _s(x) -> list:
    return b64(x, "str")


def _lh2(tmp_path_factory, alt, values=(), entities=None) -> dict:
    url, scen = make_db(tmp_path_factory, alt=alt, values=list(values), base_chain=_chain(LH2, "lh2_three_region"),
                        entities=entities, fixture_json=LH2)
    return run(tmp_path_factory, url, scen)


def _coarse_procs(fd) -> set[str]:
    return set() if fd.processCoarse is None else set(utf8(fd.processCoarse)["p"].to_list())


# ---------------------------------------------------------------------------
# The rule (pure function)


def _frames(process_nodes, node_step, balance=None, excluded=None, steps_of=None):
    """``coarse_process_frames`` on one 48-h period of 1-h steps; ``steps_of`` node -> its group step (h)."""
    rows = []
    for n, step in (steps_of or node_step).items():
        for h in range(48):
            first = int(h // step * step)
            rows.append((n, "d1", f"t{first:02d}", f"t{h:02d}"))
    ncs = pl.DataFrame(rows, schema={"n": pl.Utf8, "d": pl.Utf8, "b_first": pl.Utf8, "t": pl.Utf8}, orient="row")
    sd = pl.DataFrame({"d": ["d1"] * 48, "t": [f"t{h:02d}" for h in range(48)], "value": [1.0] * 48})
    nodes = set().union(*process_nodes.values())
    return _coarse.coarse_process_frames(
        process_nodes=process_nodes, balance_nodes=nodes if balance is None else balance, node_step=node_step,
        node_coarse_step=ncs, step_duration=sd, excluded=excluded or {}, solve_step=1.0)


def test_rule_all_balance_nodes_coarse() -> None:
    """A process is coarse only when every balance node is coarse; a node without a balance is neutral; a
    process without a balance node keeps the solve's step."""
    fr = _frames({"pipe": {"a", "b"}, "elz": {"a", "fine"}, "fuel_unit": {"a", "fuel"}, "free": {"fuel"}},
                 {"a": 24.0, "b": 24.0}, balance={"a", "b", "fine"})
    assert utf8(fr.processCoarse)["p"].to_list() == ["fuel_unit", "pipe"]
    assert fr.step_hours == {"fuel_unit": 24.0, "pipe": 24.0}
    pipe = fr.process_coarse_step.filter(pl.col("p") == "pipe")
    assert pipe.height == 48 and pipe["b_first"].unique().sort().to_list() == ["t00", "t24"]
    sums = pipe.group_by("b_first").agg(pl.col("w").sum())["w"].to_list()
    assert sums == pytest.approx([1.0, 1.0])
    assert pipe["w"].to_list() == pytest.approx([1.0 / 24.0] * 48)


def test_rule_shortest_dividing_step() -> None:
    """24 h + 12 h nodes: the process takes the 12-h steps; 24 h + 36 h: the steps do not divide, it stays fine
    with the reason."""
    fr = _frames({"p12": {"a", "b"}, "p36": {"a", "c"}}, {"a": 24.0, "b": 12.0, "c": 36.0})
    assert utf8(fr.processCoarse)["p"].to_list() == ["p12"]
    assert fr.step_hours == {"p12": 12.0}
    assert fr.process_coarse_step["b_first"].unique().sort().to_list() == ["t00", "t12", "t24", "t36"]
    assert "do not divide" in fr.kept_fine["p36"]


def test_rule_excluded_stays_fine() -> None:
    """A process with a per-step feature keeps the solve's step even when all its nodes are coarse; the reason
    is kept for the info line."""
    fr = _frames({"liq": {"a", "b"}}, {"a": 24.0, "b": 24.0}, excluded={"liq": "it has online variables"})
    assert fr.processCoarse.height == 0
    assert fr.kept_fine == {"liq": "it has online variables"}


# ---------------------------------------------------------------------------
# lh2


def test_lh2_coarse_processes(tmp_path_factory) -> None:
    """lh2: the liquefiers (h2 -> lh2) and the pipes (lh2 <-> lh2) are coarse at 24 h; the electrolysers
    (elec hourly -> h2 daily) and every hourly process are not.  Each coarse step's weights sum to 1."""
    fd = last_stochastic(_lh2(tmp_path_factory, "cf_rule")).flex_data
    assert _coarse_procs(fd) == LH2_COARSE
    steps = utf8(fd.process_coarse_step.drop("w")).with_columns(w=fd.process_coarse_step["w"])
    assert set(steps["p"].unique().to_list()) == LH2_COARSE
    liq = steps.filter(pl.col("p") == "liquefier_A")
    assert liq.height == 168 and liq["b_first"].n_unique() == 7
    assert liq.group_by("d", "b_first").agg(pl.col("w").sum())["w"].to_list() == pytest.approx([1.0] * 7)
    node = utf8(fd.node_coarse_step).filter(pl.col("n") == "lh2_A").select("d", "b_first", "t")
    assert liq.select("d", "b_first", "t").equals(node)


def test_lh2_online_unit_stays_fine(tmp_path_factory, caplog, monkeypatch) -> None:
    """liquefier_A with online variables keeps the solve's step (owner Q2), with an info line naming the reason;
    the other liquefiers and the pipes stay coarse."""
    monkeypatch.setattr(_coarse, "_WARNED", set())
    caplog.set_level(logging.INFO, logger=_coarse.__name__)
    fd = last_stochastic(_lh2(tmp_path_factory, "cf_uc", [
        ["unit", "liquefier_A", "startup_method", _s("linear")], ["unit", "liquefier_A", "min_load", _f(0.3)],
        ["unit", "liquefier_A", "virtual_unitsize", _f(50.0)]])).flex_data
    assert _coarse_procs(fd) == LH2_COARSE - {"liquefier_A"}
    msgs = [r.message for r in caplog.records]
    assert any("'liquefier_A' connects only nodes of coarse" in m and "online variables" in m for m in msgs)
    assert any("use the 24 h time step of their nodes" in m and "'pipe_AB'" in m for m in msgs)


def test_lh2_user_constraint_flow_stays_fine(tmp_path_factory) -> None:
    """A user constraint with a flow coefficient on pipe_AB keeps pipe_AB at the solve's step: the constraint's
    rows are per fine step, and with a time-varying constant a constant flow could not meet them."""
    fd = last_stochastic(_lh2(tmp_path_factory, "cf_uc_flow", [
        ["constraint", "cap_ab", "sense", _s("less_than")], ["constraint", "cap_ab", "constant", _f(1000.0)],
        ["connection__node", ["pipe_AB", "lh2_B"], "constraint_flow_coeff",
         b64({"index_type": "str", "index_name": "constraint", "data": [["cap_ab", 1.0]]}, "map")]],
        [["constraint", "cap_ab", None], ["connection__node", ["pipe_AB", "lh2_B"], None]])).flex_data
    assert _coarse_procs(fd) == LH2_COARSE - {"pipe_AB"}
