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
import re
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


# ---------------------------------------------------------------------------
# Output expansion (flextool.engine.output._coarse_flows)


class _Steps:
    """A FlexData stand-in carrying only ``process_coarse_step``: pipe has 3-h coarse steps on 6 hourly steps."""

    process_coarse_step = pl.DataFrame({"p": ["pipe"] * 6, "d": ["d1"] * 6,
                                        "b_first": ["t1", "t1", "t1", "t4", "t4", "t4"],
                                        "t": [f"t{h}" for h in range(1, 7)], "w": [1.0 / 3.0] * 6})


def test_expand_coarse_flows_copies_to_missing_steps() -> None:
    """A coarse flow (one LP row per coarse step) is copied to the other fine steps of its step, right after its
    own row; a fine process is unchanged; a step that has its own LP row keeps it (an LP whose flows are fine
    comes back unchanged)."""
    from flextool.engine.output._coarse_flows import expand_coarse_flows

    lp = pl.DataFrame({"p": ["elz", "elz", "pipe", "pipe"], "source": ["a", "a", "x", "x"],
                       "sink": ["b", "b", "y", "y"], "d": ["d1"] * 4, "t": ["t1", "t2", "t1", "t4"],
                       "value": [5.0, 6.0, 2.0, 3.0]})
    out = expand_coarse_flows(lp, _Steps())
    assert out.columns == lp.columns and out.schema == lp.schema
    assert out.filter(pl.col("p") == "elz").equals(lp.filter(pl.col("p") == "elz"))
    pipe = out.filter(pl.col("p") == "pipe")
    assert pipe["t"].to_list() == [f"t{h}" for h in range(1, 7)]
    assert pipe["value"].to_list() == [2.0, 2.0, 2.0, 3.0, 3.0, 3.0]
    fine_lp = pl.DataFrame({"p": ["pipe"] * 6, "source": ["x"] * 6, "sink": ["y"] * 6, "d": ["d1"] * 6,
                            "t": [f"t{h}" for h in range(1, 7)], "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    assert expand_coarse_flows(fine_lp, _Steps()).equals(fine_lp)
    assert expand_coarse_flows(lp, None).equals(lp)


def test_expand_value_keys_copies_to_missing_steps() -> None:
    """The HiGHS-name reader's keys ``(d, t, p, source, sink)``: the same rule as expand_coarse_flows."""
    from flextool.engine.output._coarse_flows import coarse_steps, expand_value_keys

    keys = {("d1", "t1", "pipe", "x", "y"): 2.0, ("d1", "t4", "pipe", "x", "y"): 3.0,
            ("d1", "t2", "elz", "a", "b"): 6.0, ("d1", "t5", "pipe", "x", "y"): 9.0}
    expand_value_keys(keys, coarse_steps(_Steps()))
    pipe = {k[1]: v for k, v in keys.items() if k[2] == "pipe"}
    assert pipe == {"t1": 2.0, "t2": 2.0, "t3": 2.0, "t4": 3.0, "t5": 9.0, "t6": 3.0}
    assert keys[("d1", "t2", "elz", "a", "b")] == 6.0


_FLOW_READS = re.compile(r"""\.value\(\s*["']v_flow(_back)?["']|_vars\[\s*["']v_flow(_back)?["']\s*\]"""
                         r"""|col_id_of_var\(\s*["']v_flow|fix_cols\(\s*["']v_flow""")
# Files that may read the LP's flow columns as they are: the expansion helper itself, and Benders, which reads only
# the half-flows of cross-region arcs (a cross-region process is never coarse).
_RAW_FLOW_READERS = {"flextool/engine/output/_coarse_flows.py", "flextool/engine/solve/_benders.py"}


def test_flow_readers_expand_coarse_flows() -> None:
    """Every reader of v_flow / v_flow_back values outside the expansion helper goes through it: a raw read
    would see a coarse flow only at its coarse step's first fine step (outputs lose the other steps, the CO2
    carry under-counts)."""
    hits = []
    for path in sorted((FLEXTOOL_ROOT / "flextool").rglob("*.py")):
        rel = path.relative_to(FLEXTOOL_ROOT).as_posix()
        if rel in _RAW_FLOW_READERS:
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if _FLOW_READS.search(line):
                hits.append(f"{rel}:{i}: {line.strip()}")
    assert not hits, "raw v_flow reads (use flextool.engine.output._coarse_flows):\n" + "\n".join(hits)
