"""``bind_intraperiod_blocks`` storage nodes keep their block balance in every sub-solve that can have one (item 113).

A ``bind_intraperiod_blocks`` node (``nodeStateBlock``) is balanced once per timeset block by
``nodeBalanceBlock_eq`` (``nodeBalance_eq`` skips it) and its state is constant within each block
(``stateConstantWithinBlock_eq``).  Before, a sub-solve that is not a Spine solve (every roll, every nested anchor
``<solve>_<period>``) never got the block frames, so the node had no energy balance at all (energy from nothing).
Now: a rolling solve with such a node is an error (how a block cut by a roll window should behave is an open
modelling question, Q-c15c-A), a nested anchor sub-solve gets its blocks, and the model refuses a block node
without blocks.

Fixtures: ``tests/fixtures/tests.json`` (``5weeks`` = five 4-step blocks; ``bind_battery_intraperiod_blocks`` on the
battery) with override alternatives, built from JSON under the pytest tmp dir.
"""
from __future__ import annotations

import json

import polars as pl
import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from tests.engine._lp_rows import row_terms
from tests.engine._unit_db import FLEXTOOL_ROOT, b64, build, make_db, run, utf8

pytestmark = pytest.mark.solver

FIX = FLEXTOOL_ROOT / "tests" / "fixtures"


def _chain(scenario: str) -> list[str]:
    data = json.loads((FIX / "tests.json").read_text())
    rows = [r for r in data["scenario_alternatives"] if r[0] == scenario]
    nxt = {a: b for _, a, b in rows}
    alt = next(a for a in nxt if a not in set(nxt.values()))
    out = []
    while alt is not None:
        out.append(alt)
        alt = nxt.get(alt)
    return out


def test_rolling_intraperiod_blocks_raises(tmp_path_factory) -> None:
    """Rolling dispatch (48 h window / 24 h jump over the 72-h fullYear) with a bind_intraperiod_blocks battery
    stops with an error naming the solve, the node and the method.  Before: the run succeeded and every roll had
    no balance row of any family for the battery."""
    vals = [["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", b64(48.0, "float")],
            ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", b64(24.0, "float")],
            ["solve", "dispatch_fullYear_roll", "period_timeset",
             b64({"index_type": "str", "index_name": "period", "data": [["p2020", "fullYear"]]}, "map")],
            ["solve", "dispatch_fullYear_roll", "realized_periods",
             b64({"value_type": "str", "data": ["p2020"]}, "array")]]
    url, scen = make_db(tmp_path_factory, alt="ib_roll", values=vals,
                        base_chain=["init", "west", "wind", "coal", "battery", "fullYear", "dispatch_fullYear_roll",
                                    "bind_battery_intraperiod_blocks"])
    with pytest.raises(FlexToolConfigError, match="bind_intraperiod_blocks") as exc:
        run(tmp_path_factory, url, scen)
    assert "battery" in str(exc.value) and "dispatch_fullYear_roll" in str(exc.value)


def test_synthetic_subsolve_gets_blocks(tmp_path_factory) -> None:
    """Nested invest (5weeks, one solve per period) → storage (fullYear 6 h, single solve) with a
    bind_intraperiod_blocks battery: every sub-solve is a synthetic anchor ``<solve>_<period>`` and has the block
    frames of its own steps — by hand: 5weeks → blocks t0001 (4 steps), t0009 (8: t0009–t0016, the t0009 and
    t0013 weeks touch), t0025 (4), t0037 (4); fullYear_6h → one block of 12 steps from t0001 — and equal to what
    the same two solves get as Spine solves (``apply_derived_e``'s ``period_block_family_from_source``) when they
    run un-nested over all four periods.  The battery has one ``nodeBalanceBlock_eq`` row per block,
    ``stateConstantWithinBlock_eq`` rows and no ``nodeBalance_eq`` row.  Before: the anchors had no block
    frames, so the battery had 0 balance rows of any family."""
    data = json.loads((FIX / "tests.json").read_text())
    # The storage solve keeps no rolling child: the anchors are then the only sub-solves.
    data["parameter_values"] = [pv for pv in data["parameter_values"]
                                if not (pv[1] == "storage_fullYear_6h" and pv[2] == "contains_solves")]
    fixture = tmp_path_factory.mktemp("ib_fx") / "tests_no_child.json"
    fixture.write_text(json.dumps(data))
    url, scen = make_db(tmp_path_factory, alt="ib_syn", values=[], fixture_json=fixture,
                        base_chain=_chain("multi_fullYear_battery_nested_multi_invest")
                        + ["bind_battery_intraperiod_blocks"])
    steps = run(tmp_path_factory, url, scen)
    assert len(steps) == 8 and all(k.rsplit("_", 1)[-1].startswith("p20") for k in steps), list(steps)
    # The same solves as Spine solves (no per-period anchors): invest_5weeks contains storage_fullYear_6h.
    url_s, scen_s = make_db(tmp_path_factory, alt="ib_spine", values=[], fixture_json=fixture,
                            base_chain=["init", "west", "wind", "coal", "battery", "5weeks", "fullYear",
                                        "storage_fullYear_6h", "invest_oneYear4Times_5weeks",
                                        "bind_battery_intraperiod_blocks"])
    spine = run(tmp_path_factory, url_s, scen_s)
    assert set(spine) == {"invest_5weeks", "storage_fullYear_6h"}, list(spine)
    for name, step in steps.items():
        fd = step.flex_data
        assert fd.nodeStateBlock is not None and utf8(fd.nodeStateBlock)["n"].to_list() == ["battery"], name
        assert fd.period_block_time is not None, name
        got = utf8(fd.period_block_time).sort("d", "b_first", "t")
        periods = sorted(set(got["d"].to_list()))
        base = name.rsplit("_", 1)[0]
        expected = ({"t0001": 4, "t0009": 8, "t0025": 4, "t0037": 4} if base == "invest_5weeks"
                    else {"t0001": 12})
        for d in periods:
            sizes = dict(got.filter(pl.col("d") == d).group_by("b_first").agg(pl.len()).rows())
            assert sizes == expected, (name, d, sizes)
        ref = utf8(spine[base].flex_data.period_block_time).filter(pl.col("d").is_in(periods))
        assert got.equals(ref.sort("d", "b_first", "t")), name
        ref_succ = utf8(spine[base].flex_data.period_block_succ).filter(pl.col("d").is_in(periods))
        assert utf8(fd.period_block_succ).sort("d", "b_first").equals(ref_succ.sort("d", "b_first")), name
        n_blocks = utf8(fd.period_block).height
        assert n_blocks == len(expected) * len(periods)
        pb = build(fd)
        assert len(row_terms(pb, "nodeBalanceBlock_eq", "battery,")) == n_blocks, name
        assert len(row_terms(pb, "stateConstantWithinBlock_eq", "battery,")) > 0, name
        assert len(row_terms(pb, "nodeBalance_eq", "battery,")) == 0, name


# ---------------------------------------------------------------------------
# #117 — nodeBalanceBlock_eq is nodeBalance_eq summed over each block
# ---------------------------------------------------------------------------

BLOCK_CHAIN = ["init", "west", "wind", "battery", "5weeks", "bind_battery_intraperiod_blocks"]
# 5weeks blocks (the t0009 and t0013 weeks touch and merge): first step → steps of the block.
BLOCKS = {"t0001": [f"t{i:04d}" for i in range(1, 5)], "t0009": [f"t{i:04d}" for i in range(9, 17)],
          "t0025": [f"t{i:04d}" for i in range(25, 29)], "t0037": [f"t{i:04d}" for i in range(37, 41)]}
SU = 50.0          # battery existing 50 MWh → state_unitsize 50


def _no_loss() -> list:
    return [["node", "battery", "self_discharge_loss", b64(0.0, "float")]]


def _inner(name: str) -> list[str]:
    return name[name.index("[") + 1:-1].split(",")


def test_block_state_is_end_of_block(tmp_path_factory) -> None:
    """Inverter off, 10 MWh into the battery in t0003 and out in t0026, fix_start 0: the state of each block is its
    end state — p2020 blocks t0001 / t0009 / t0025 / t0037 hold 10 / 10 / 0 / 0 MWh — and the fix_start pin sits on
    the period's last step t0040 (the state the first block starts from).  Before: start-of-block labelling
    (0 / 10 / 10 / 0) and the pin on t0001."""
    from tests.engine._unit_db import time_map

    vals = _no_loss() + [["connection", "battery_inverter", "existing", b64(0.0, "float")],
                         ["node", "battery", "storage_state_start", b64(0.0, "float")],
                         ["node", "battery", "inflow", time_map({"t0003": 10.0, "t0026": -10.0}, 0.0)]]
    url, scen = make_db(tmp_path_factory, alt="ib117_end", values=vals, base_chain=BLOCK_CHAIN)
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    state = utf8(step.solution.value("v_state")).filter((pl.col("n") == "battery") & (pl.col("d") == "p2020"))
    got = {t: v * SU for t, v in state.select("t", "value").iter_rows()}
    for b_first, ts in BLOCKS.items():
        want = 10.0 if b_first in ("t0001", "t0009") else 0.0
        for t in ts:
            assert got[t] == pytest.approx(want, abs=1e-6), (b_first, t, got[t])
    pb = build(step.flex_data)
    assert row_terms(pb, "storage_state_start_binding", "battery,") == {}
    pins = row_terms(pb, "storage_state_start_binding_cyclic_period", "battery,")
    assert set(pins) == {"storage_state_start_binding_cyclic_period[battery,p2020,t0040]"}, sorted(pins)


def _overlay_values() -> tuple[list, list]:
    """A lossless two-way inverter (one flow variable: v_flow_back) and a min_load_efficiency unit ``bgen``
    drawing from the battery (section input v_online · section)."""
    entities = [["unit", "bgen", None], ["unit__inputNode", ["bgen", "battery"], None],
                ["unit__outputNode", ["bgen", "west"], None]]
    vals = _no_loss() + [
        ["connection", "battery_inverter", "transfer_method", b64("no_losses_no_variable_cost", "str")],
        ["unit", "bgen", "existing", b64(10.0, "float")],
        ["unit", "bgen", "conversion_method", b64("min_load_efficiency", "str")],
        ["unit", "bgen", "efficiency", b64(0.9, "float")],
        ["unit", "bgen", "efficiency_at_min_load", b64(0.8, "float")],
        ["unit", "bgen", "min_load", b64(0.3, "float")],
        ["unit", "bgen", "startup_method", b64("linear", "str")]]
    return entities, vals


def _built(tmp_path_factory, alt: str, extra: list):
    entities, vals = _overlay_values()
    url, scen = make_db(tmp_path_factory, alt=alt, values=vals + extra, base_chain=BLOCK_CHAIN, entities=entities)
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    return step.flex_data, build(step.flex_data)


def test_block_row_is_summed_node_balance(tmp_path_factory) -> None:
    """Every non-state column of a block row (flows, the inverter's v_flow_back, bgen's v_online section input,
    slacks) has the coefficient the same column has in the battery's nodeBalance_eq row when the battery is a
    plain bind_within_period node, the RHS is the summed −inflow, and the state terms are −su on the block's
    state and +su on the previous block's state (the first block of a period: the period's last block).  Before:
    no v_flow_back and no section terms, the row negated, the state change from the block to the next block."""
    fd, pb = _built(tmp_path_factory, "ib117_sum", [])
    _, pb_wp = _built(tmp_path_factory, "ib117_wp",
                      [["node", "battery", "storage_binding_method", b64("bind_within_period", "str")]])
    from tests.engine._lp_rows import row_bounds

    blocks = row_terms(pb, "nodeBalanceBlock_eq", "battery,")
    fine = row_terms(pb_wp, "nodeBalance_eq", "battery,")
    fine_bounds = row_bounds(pb_wp, "nodeBalance_eq", "battery,")
    bounds = row_bounds(pb, "nodeBalanceBlock_eq", "battery,")
    periods = sorted(set(utf8(fd.period_block)["d"].to_list()))
    assert len(blocks) == len(BLOCKS) * len(periods)
    order = list(BLOCKS)
    seen_back = seen_online = 0
    for d in periods:
        for i, (b_first, ts) in enumerate(BLOCKS.items()):
            row = blocks[f"nodeBalanceBlock_eq[battery,{d},{b_first}]"]
            want: dict[str, float] = {}
            for t in ts:
                for col, v in fine[f"nodeBalance_eq[battery,{d},{t}]"].items():
                    if not col.startswith("v_state["):
                        want[col] = v
            got = {c: v for c, v in row.items() if not c.startswith("v_state[")}
            assert set(got) == set(want), (d, b_first, sorted(set(got) ^ set(want)))
            for c, v in want.items():
                assert got[c] == pytest.approx(v, rel=1e-12, abs=1e-12), (d, b_first, c)
            seen_back += sum(c.startswith("v_flow_back[battery_inverter,") for c in got)
            seen_online += sum(c.startswith("v_online_linear[bgen,") for c in got)
            lo = sum(fine_bounds[f"nodeBalance_eq[battery,{d},{t}]"][0] for t in ts)
            assert bounds[f"nodeBalanceBlock_eq[battery,{d},{b_first}]"][0] == pytest.approx(lo, abs=1e-9)
            states = {c: v for c, v in row.items() if c.startswith("v_state[") and v != 0.0}
            prev_last = BLOCKS[order[i - 1]][-1]          # i = 0 → the period's last block (the loop)
            assert states == {f"v_state[battery,{d},{ts[-1]}]": pytest.approx(-SU, rel=1e-12),
                              f"v_state[battery,{d},{prev_last}]": pytest.approx(SU, rel=1e-12)}, (d, b_first)
    assert seen_back == sum(len(ts) for ts in BLOCKS.values()) * len(periods)
    assert seen_online == seen_back
