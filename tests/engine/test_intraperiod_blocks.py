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
