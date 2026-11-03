"""The routing rule of the storage carry between sub-solves (``flextool.engine.solve._carry_routing``), on synthetic
solve trees: no LP, no database.

A sub-solve continues from the latest state realised before it at its own nesting level or deeper; a level's first
appearance has none (fix_start); a realised-dispatch sibling starts where its parent starts.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import polars as pl
import pytest

from flextool.engine.solve._carry_routing import CarryRouter, make_position, make_time_index, with_end_position


def _handoff(value: float, step: tuple[str, str], node: str = "s"):
    return SimpleNamespace(
        roll_end_state=pl.DataFrame({"node": [node], "value": [value]}),
        roll_end_step=pl.DataFrame({"period": [step[0]], "step": [step[1]], "duration": [1.0]}))


def _pos(period: str, step: str):
    """Synthetic time: periods p1 < p2 < …, steps tNN."""
    return int(period[1:]), int(step[1:])


def _value(frame, node: str = "s"):
    return None if frame is None else frame.filter(pl.col("node") == node)["value"][0]


def _run(router, order, realised, firsts):
    """Run ``order``; each sub-solve records ``realised[solve]`` = (value, end step) after it; returns the carry
    each sub-solve got."""
    got = {}
    for solve in order:
        router.start(solve)
        got[solve] = _value(router.roll_end_state(solve, firsts[solve]))
        if solve in realised:
            router.record(solve, _handoff(*realised[solve]))
    return got


def test_top_level_sequence_continues():
    """y1 → y2 → y3, each realising its own period: every later solve takes the previous solve's end."""
    router = CarryRouter({"y1": None, "y2": None, "y3": None}, position=_pos)
    got = _run(router, ["y1", "y2", "y3"],
               {"y1": (10.0, ("p1", "t48")), "y2": (20.0, ("p2", "t48")), "y3": (30.0, ("p3", "t48"))},
               {"y1": ("p1", "t01"), "y2": ("p2", "t01"), "y3": ("p3", "t01")})
    assert got == {"y1": None, "y2": 10.0, "y3": 20.0}


def test_rolling_under_rolling():
    """Storage rolls S0, S1 (level 0, realising nothing) ⊃ dispatch rolls D0–D3 (level 1, two per storage roll):
    D0 is the first of its level; D1 takes D0; S1 takes D1 (upward feedback); D2 (the first child of S1) takes D1;
    D3 takes D2."""
    parent = {"S0": None, "S1": None, "D0": "S0", "D1": "S0", "D2": "S1", "D3": "S1"}
    router = CarryRouter(parent, position=_pos)
    assert [router.nesting_level(s) for s in ("S0", "D0", "D3")] == [0, 1, 1]
    got = _run(router, ["S0", "D0", "D1", "S1", "D2", "D3"],
               {"D0": (1.0, ("p1", "t06")), "D1": (2.0, ("p1", "t12")), "D2": (3.0, ("p1", "t18")),
                "D3": (4.0, ("p1", "t24"))},
               {"S0": ("p1", "t01"), "D0": ("p1", "t01"), "D1": ("p1", "t07"), "S1": ("p1", "t13"),
                "D2": ("p1", "t13"), "D3": ("p1", "t19")})
    assert got == {"S0": None, "D0": None, "D1": 1.0, "S1": 2.0, "D2": 2.0, "D3": 3.0}


def test_parent_realising_its_window_is_not_a_child_start():
    """A parent that realises its own window (level 0) does not start its child (level 1): the child's level appears
    for the first time; and the parent's realised end lies after the child's start anyway."""
    router = CarryRouter({"P": None, "C": "P"}, position=_pos)
    got = _run(router, ["P", "C"], {"P": (9.0, ("p1", "t48"))}, {"P": ("p1", "t01"), "C": ("p1", "t01")})
    assert got == {"P": None, "C": None}


def test_sibling_starts_where_its_parent_starts():
    """Rolling separate_dispatch: uc_k and its realised-dispatch sibling disp_k both continue from disp_{k−1}; disp_k
    never takes uc_k's end-of-window state (recorded after uc_k, before disp_k)."""
    parent = {"uc0": None, "d0": None, "uc1": None, "d1": None}
    router = CarryRouter(parent, siblings={"d0": "uc0", "d1": "uc1"}, position=_pos)
    got = _run(router, ["uc0", "d0", "uc1", "d1"],
               {"uc0": (100.0, ("p1", "t24")), "d0": (7.0, ("p1", "t24")), "uc1": (200.0, ("p1", "t48")),
                "d1": (8.0, ("p1", "t48"))},
               {"uc0": ("p1", "t01"), "d0": ("p1", "t01"), "uc1": ("p1", "t25"), "d1": ("p1", "t25")})
    assert got == {"uc0": None, "d0": None, "uc1": 7.0, "d1": 7.0}


def test_per_node_latest():
    """Per node, the value of the latest entry holding it."""
    router = CarryRouter({"a": None, "b": None, "c": None}, position=_pos)
    router.start("a")
    router.record("a", SimpleNamespace(roll_end_state=pl.DataFrame({"node": ["x", "y"], "value": [1.0, 2.0]}),
                                       roll_end_step=pl.DataFrame({"period": ["p1"], "step": ["t10"]})))
    router.start("b")
    router.record("b", SimpleNamespace(roll_end_state=pl.DataFrame({"node": ["x"], "value": [3.0]}),
                                       roll_end_step=pl.DataFrame({"period": ["p2"], "step": ["t10"]})))
    router.start("c")
    frame = router.roll_end_state("c", ("p3", "t01"))
    assert frame.rows() == [("x", 3.0), ("y", 2.0)]


def test_entry_not_before_the_start_is_not_carried(caplog):
    """Routing guard: an entry whose realised end is not before the consumer's first step (a re-solve of the same
    time) is not carried, with a warning naming both solves."""
    router = CarryRouter({"a": None, "b": None}, position=_pos)
    with caplog.at_level(logging.WARNING, logger="flextool.engine.solve._carry_routing"):
        got = _run(router, ["a", "b"], {"a": (5.0, ("p1", "t48"))}, {"a": ("p1", "t01"), "b": ("p1", "t01")})
    assert got == {"a": None, "b": None}
    assert any("'a'" in r.getMessage() and "'b'" in r.getMessage() for r in caplog.records)


def test_level_first_appearing_later_warns(caplog):
    """y1 (level 0, realised) then I2 (level 0) ⊃ D2 (level 1): I2 continues from y1, D2 is the first of its level
    and starts from fix_start (owner rule) — visible as a warning."""
    router = CarryRouter({"y1": None, "I2": None, "D2": "I2"}, position=_pos)
    with caplog.at_level(logging.WARNING, logger="flextool.engine.solve._carry_routing"):
        got = _run(router, ["y1", "I2", "D2"], {"y1": (4.0, ("p1", "t48"))},
                   {"y1": ("p1", "t01"), "I2": ("p2", "t01"), "D2": ("p2", "t01")})
    assert got == {"y1": None, "I2": 4.0, "D2": None}
    assert any("'D2'" in r.getMessage() and "'y1'" in r.getMessage() for r in caplog.records)


def test_registry_is_per_router():
    """Two routers (two runs) share nothing."""
    first = CarryRouter({"a": None}, position=_pos)
    first.start("a")
    first.record("a", _handoff(1.0, ("p1", "t10")))
    second = CarryRouter({"a": None}, position=_pos)
    second.start("a")
    assert second.roll_end_state("a", ("p2", "t01")) is None


def test_position_uses_the_base_timeline_and_branch_anchor():
    """Steps of an aggregated solve (6-h timeline) and of an hourly solve compare by their base-timeline index; a
    stochastic branch period ranks with the period it branches from."""
    E = SimpleNamespace
    hourly = [E(timestep=f"t{i:02d}", index=i - 1, duration="1") for i in range(1, 13)]
    six = [E(timestep="t01", index=0, duration="6"), E(timestep="t07", index=1, duration="6")]
    pos = make_position(
        ["agg", "hour"], {"agg": {"p1": six}, "hour": {"p1": hourly, "p1_hi": hourly[6:]}},
        {"hour": [("p1", "p1"), ("p1", "p1_hi")]},
        complete_solve={"agg": "agg", "hour": "hour"},
        timesets_used_by_solves={"agg": [("p1", "ts6")], "hour": [("p1", "ts1"), ("p1_hi", "ts1")]},
        timesets__timeline={"ts6": "tl6", "ts1": "tl"},
        timelines={"tl": [(f"t{i:02d}", 1.0) for i in range(1, 13)], "tl6": [("t01", 6.0), ("t07", 6.0)]},
        original_timeline={"tl6": "tl"})
    assert pos("p1", "t07") == (0, 6)
    assert pos("p1", "t06") < pos("p1", "t07") < pos("p1", "t08")
    assert pos("p1_hi", "t09") == pos("p1", "t09")
    assert pos("p9", "t01") is None


def test_end_of_places_a_step_end_on_the_base_timeline():
    """end_of: where a step ends on the base timeline.  The 6-h step t07 of the aggregated solve starts at base
    index 6 and ends at 6 + 6 = 12; the hourly t12 starts at 11 and ends at 12, so the two ends compare equal.  A
    duration that does not cover whole base steps (2.5 h of 1-h steps) has no end; a branch period carries its own
    name; a step two solves place on different base indices has no end."""
    E = SimpleNamespace
    hourly = [E(timestep=f"t{i:02d}", index=i - 1, duration="1") for i in range(1, 13)]
    six = [E(timestep="t01", index=0, duration="6"), E(timestep="t07", index=1, duration="6")]
    index = make_time_index(
        ["agg", "hour"], {"agg": {"p1": six}, "hour": {"p1": hourly, "p1_hi": hourly[6:]}},
        {"hour": [("p1", "p1"), ("p1", "p1_hi")]},
        complete_solve={"agg": "agg", "hour": "hour"},
        timesets_used_by_solves={"agg": [("p1", "ts6")], "hour": [("p1", "ts1"), ("p1_hi", "ts1")]},
        timesets__timeline={"ts6": "tl6", "ts1": "tl"},
        timelines={"tl": [(f"t{i:02d}", 1.0) for i in range(1, 13)], "tl6": [("t01", 6.0), ("t07", 6.0)]},
        original_timeline={"tl6": "tl"})
    assert index.end_of("p1", "t07", 6.0) == (0, "tl", 12, None)
    assert index.end_of("p1", "t12", "1") == (0, "tl", 12, None)
    assert index.end_of("p1", "t01", 2.5) is None
    assert index.end_of("p1_hi", "t12", 1.0) == (0, "tl", 12, "p1_hi")
    assert index.end_of("p9", "t01", 1.0) is None
    step = pl.DataFrame({"period": ["p1"], "step": ["t12"], "duration": [1.0]})
    routed = with_end_position(step, index.end_of)
    assert routed.select("end_rank", "end_root", "end_index", "end_branch").row(0) == ("0", "tl", "12", "")
    clash = make_time_index(
        ["a", "b"], {"a": {"p1": hourly[:2]}, "b": {"p1": hourly[:2]}}, {},
        complete_solve={"a": "a", "b": "b"},
        timesets_used_by_solves={"a": [("p1", "ts1")], "b": [("p1", "tsx")]},
        timesets__timeline={"ts1": "tl", "tsx": "tlx"},
        timelines={"tl": [("t01", 1.0), ("t02", 1.0)], "tlx": [("t00", 1.0), ("t01", 1.0), ("t02", 1.0)]},
        original_timeline={})
    assert clash.end_of("p1", "t01", 1.0) is None


@pytest.mark.parametrize("handoff", [None, SimpleNamespace(roll_end_state=None, roll_end_step=None),
                                     SimpleNamespace(roll_end_state=pl.DataFrame({"node": [], "value": []}),
                                                     roll_end_step=None)])
def test_no_realised_state_records_nothing(handoff):
    router = CarryRouter({"a": None, "b": None}, position=_pos)
    router.start("a")
    router.record("a", handoff)
    router.start("b")
    assert router.roll_end_state("b", ("p2", "t01")) is None


def test_warm_fingerprint_sees_the_carried_nodes():
    """The carried node set swaps start rows in or out (fix_start pin / carry terms): two sub-solves that differ only
    in it never share a warm LP (the fingerprint pre-filter already differs)."""
    from polar_high import Param

    from flextool.engine.solve._warm import _fingerprint

    def data(nodes):
        carry = None if nodes is None else Param(("n",), pl.DataFrame({"n": nodes, "value": [1.0] * len(nodes)}))
        return SimpleNamespace(p_roll_continue_state=carry)

    assert _fingerprint(data(None)) != _fingerprint(data(["battery"]))
    assert _fingerprint(data(["battery"])) != _fingerprint(data(["battery", "hydro"]))
    assert _fingerprint(data(["hydro", "battery"])) == _fingerprint(data(["battery", "hydro"]))
