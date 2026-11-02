"""The minimum up/down-time look-back follows the ramp predecessor links, not the order of the step names.

``params._mintime.mintime_lookback`` on hand-built frames.  The look-back of a step is the step itself plus its
ancestors along ``dt_ramp_link`` while the elapsed time (the summed durations of the chain from the ancestor up
to, not including, the step) stays below the minimum time; it stops at a step without a link (solve start, a
period that does not continue the previous one, a gap, a representative-block start, a forecast branch that starts
the solve) and never wraps.  Every expected set below is derived by hand from that rule.
"""
from __future__ import annotations

import polars as pl
import pytest

from flextool.engine.preprocess.params._mintime import cut_short_lookback_warnings, mintime_lookback
from flextool.engine.preprocess.params._ramp_links import ramp_link_from_jump


def _dt(steps: list[tuple[str, str]]) -> pl.DataFrame:
    return pl.DataFrame({"d": [d for d, _ in steps], "t": [t for _, t in steps]})


def _link(pairs: list[tuple[tuple[str, str], tuple[str, str]]]) -> pl.DataFrame:
    return pl.DataFrame({"d": [a[0] for a, _ in pairs], "t": [a[1] for a, _ in pairs],
                         "d_previous": [b[0] for _, b in pairs], "t_previous": [b[1] for _, b in pairs]},
                        schema={c: pl.Utf8 for c in ("d", "t", "d_previous", "t_previous")})


def _chain(d: str, ts: list[str]) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    return [((d, ts[i]), (d, ts[i - 1])) for i in range(1, len(ts))]


def _backs(lb: pl.DataFrame, p: str, d: str, t: str) -> set[tuple[str, str]]:
    sel = lb.filter((pl.col("p") == p) & (pl.col("d") == d) & (pl.col("t") == t))
    return set(sel.select("d_back", "t_back").rows())


T5 = ["t1", "t2", "t3", "t4", "t5"]


def test_contiguous_block_counts_elapsed_predecessor_durations():
    """5 × 1 h, L = 3: t3 reaches t2 (elapsed 1) and t1 (elapsed 2); t1 has only itself.
    Durations t1 = 1, t2 = 2, t3 = 1 h, L = 3: from t3, t2 is 2 h back (< 3), t1 is 2 + 1 = 3 h back (not < 3)."""
    dt = _dt([("p", t) for t in T5])
    lb = mintime_lookback(dt, None, _link(_chain("p", T5)), [("u", 3.0)])
    assert _backs(lb, "u", "p", "t3") == {("p", "t3"), ("p", "t2"), ("p", "t1")}
    assert _backs(lb, "u", "p", "t5") == {("p", "t5"), ("p", "t4"), ("p", "t3")}
    assert _backs(lb, "u", "p", "t1") == {("p", "t1")}
    assert lb.height == 1 + 2 + 3 + 3 + 3
    assert lb.rows() == sorted(lb.rows())

    dur = pl.DataFrame({"d": ["p"] * 3, "t": ["t1", "t2", "t3"], "value": [1.0, 2.0, 1.0]})
    lb3 = mintime_lookback(_dt([("p", t) for t in T5[:3]]), dur, _link(_chain("p", T5[:3])), [("u", 3.0)])
    assert _backs(lb3, "u", "p", "t3") == {("p", "t3"), ("p", "t2")}
    assert _backs(lb3, "u", "p", "t2") == {("p", "t2"), ("p", "t1")}


def test_each_process_uses_its_own_minimum_time():
    """L = 1 h on 1 h steps: the first predecessor is 1 h back, not < 1 → only the step itself."""
    dt = _dt([("p", t) for t in T5])
    lb = mintime_lookback(dt, None, _link(_chain("p", T5)), [("u", 3.0), ("g", 1.0)])
    for t in T5:
        assert _backs(lb, "g", "p", t) == {("p", t)}
    assert _backs(lb, "u", "p", "t4") == {("p", "t4"), ("p", "t3"), ("p", "t2")}


def test_reused_timeset_stops_and_continuing_period_crosses_whatever_the_names():
    """Two years on the same timeset: no link at p2025,t1 → only itself.  Periods b2020 → a2025 continuing one
    timeline (link a2025,t1 → b2020,t3), L = 3: a2025,t1 reaches b2020,t3 (1 h) and b2020,t2 (2 h); the name
    order (a2025 before b2020) does not matter."""
    ts = ["t1", "t2", "t3"]
    dt = _dt([("p2020", t) for t in ts] + [("p2025", t) for t in ts])
    lb = mintime_lookback(dt, None, _link(_chain("p2020", ts) + _chain("p2025", ts)), [("u", 3.0)])
    assert _backs(lb, "u", "p2025", "t1") == {("p2025", "t1")}
    assert _backs(lb, "u", "p2025", "t2") == {("p2025", "t2"), ("p2025", "t1")}

    dt2 = _dt([("b2020", t) for t in ts] + [("a2025", t) for t in ts])
    link2 = _link(_chain("b2020", ts) + [(("a2025", "t1"), ("b2020", "t3"))] + _chain("a2025", ts))
    lb2 = mintime_lookback(dt2, None, link2, [("u", 3.0)])
    assert _backs(lb2, "u", "a2025", "t1") == {("a2025", "t1"), ("b2020", "t3"), ("b2020", "t2")}
    assert _backs(lb2, "u", "a2025", "t2") == {("a2025", "t2"), ("a2025", "t1"), ("b2020", "t3")}
    assert _backs(lb2, "u", "b2020", "t1") == {("b2020", "t1")}


def test_gap_and_representative_block_start_stop_the_look_back():
    """Links built by ``ramp_link_from_jump``.  Steps t1 t2 t3 t5 t6 (t4 not active, jump 2 at t5; t1 wraps):
    L = 3 → t6 reaches t5 only, t5 only itself.  Steps t1..t4 with a representative block starting at t3:
    t4 reaches t3 only, t3 only itself."""
    gap_steps = pl.DataFrame({
        "d": ["p"] * 5, "t": ["t1", "t2", "t3", "t5", "t6"],
        "d_previous": ["p"] * 5, "t_previous_within_solve": ["t6", "t1", "t2", "t3", "t5"],
        "jump": [-5, 1, 1, 2, 1]})
    dt = gap_steps.select("d", "t")
    lb = mintime_lookback(dt, None, ramp_link_from_jump(gap_steps), [("u", 3.0)])
    assert _backs(lb, "u", "p", "t6") == {("p", "t6"), ("p", "t5")}
    assert _backs(lb, "u", "p", "t5") == {("p", "t5")}
    assert _backs(lb, "u", "p", "t1") == {("p", "t1")}
    assert _backs(lb, "u", "p", "t3") == {("p", "t3"), ("p", "t2"), ("p", "t1")}

    rp_steps = pl.DataFrame({
        "d": ["p"] * 4, "t": ["t1", "t2", "t3", "t4"],
        "d_previous": ["p"] * 4, "t_previous_within_solve": ["t4", "t1", "t2", "t3"],
        "jump": [-3, 1, 1, 1]})
    rp_first = pl.DataFrame({"d": ["p", "p"], "t": ["t1", "t3"]})
    lb_rp = mintime_lookback(rp_steps.select("d", "t"), None, ramp_link_from_jump(rp_steps, rp_first),
                             [("u", 3.0)])
    assert _backs(lb_rp, "u", "p", "t4") == {("p", "t4"), ("p", "t3")}
    assert _backs(lb_rp, "u", "p", "t3") == {("p", "t3")}


def test_forecast_branch_walks_into_its_pre_reveal_steps_never_a_sibling():
    """Period P (t1, t2) before the reveal; forecast branches A and B whose t3 links to P,t2.  L = 3:
    A,t4 → {A t4, A t3, P t2}; A,t3 → {A t3, P t2, P t1}; no row of A ever names B."""
    dt = _dt([("P", "t1"), ("P", "t2"), ("A", "t3"), ("A", "t4"), ("B", "t3"), ("B", "t4")])
    link = _link([(("P", "t2"), ("P", "t1")),
                  (("A", "t3"), ("P", "t2")), (("A", "t4"), ("A", "t3")),
                  (("B", "t3"), ("P", "t2")), (("B", "t4"), ("B", "t3"))])
    lb = mintime_lookback(dt, None, link, [("u", 3.0)])
    assert _backs(lb, "u", "A", "t4") == {("A", "t4"), ("A", "t3"), ("P", "t2")}
    assert _backs(lb, "u", "A", "t3") == {("A", "t3"), ("P", "t2"), ("P", "t1")}
    assert _backs(lb, "u", "B", "t4") == {("B", "t4"), ("B", "t3"), ("P", "t2")}
    assert lb.filter((pl.col("d") == "A") & (pl.col("d_back") == "B")).height == 0
    assert lb.filter((pl.col("d") == "B") & (pl.col("d_back") == "A")).height == 0


def test_predecessor_outside_the_solve_ends_the_chain():
    """A link to a step that is not in this solve (p0,t9) adds nothing: t1 has only itself."""
    dt = _dt([("p", "t1"), ("p", "t2")])
    link = _link([(("p", "t1"), ("p0", "t9")), (("p", "t2"), ("p", "t1"))])
    lb = mintime_lookback(dt, None, link, [("u", 5.0)])
    assert _backs(lb, "u", "p", "t1") == {("p", "t1")}
    assert _backs(lb, "u", "p", "t2") == {("p", "t2"), ("p", "t1")}


def test_missing_links_raise_and_no_process_gives_none():
    dt = _dt([("p", "t1"), ("p", "t2")])
    with pytest.raises(ValueError, match="no ramp predecessor links"):
        mintime_lookback(dt, None, None, [("u", 3.0)])
    one = mintime_lookback(_dt([("p", "t1")]), None, None, [("u", 3.0)])
    assert one.rows() == [("u", "p", "t1", "p", "t1")]
    assert mintime_lookback(dt, None, None, []) is None
    assert mintime_lookback(dt, None, None, [("u", 0.0)]) is None


def _first_step_link(rows: list[tuple[str, str, str, str, int]]) -> pl.DataFrame:
    return pl.DataFrame({"period": [r[0] for r in rows], "time": [r[1] for r in rows],
                         "previous_period": [r[2] for r in rows], "previous_within_solve": [r[3] for r in rows],
                         "jump": [str(r[4]) for r in rows]})


def test_cut_short_warning_only_when_no_history_reaches_a_continuing_first_step():
    """A roll p,t5..t6 whose first step continues p,t4 of the previous roll (jump 1, p,t4 not in this roll):
    warned for a process without handed-over on-line history; silent for a process with history, when the first
    step does not continue the timeline (jump ≠ 1), and when the predecessor is a step of this roll."""
    dt = _dt([("p", "t5"), ("p", "t6")])
    fsl = _first_step_link([("p", "t5", "p", "t4", 1)])
    mt = {"min_uptime": [("u", 3.0), ("v", 4.0)], "min_downtime": [("u", 2.0), ("g", 0.0)]}
    hist = pl.DataFrame({"p": ["v"], "d": ["p"], "t": ["t4"], "startup": [1.0], "shutdown": [0.0],
                         "duration": [1.0]})
    msgs = cut_short_lookback_warnings(mt, dt, fsl, None, "roll_1")
    assert len(msgs) == 3
    assert "'u' has min_uptime=3" in msgs[0] and "(p, t5)" in msgs[0] and "'roll_1'" in msgs[0]
    assert "'v' has min_uptime=4" in msgs[1]
    assert "'u' has min_downtime=2" in msgs[2]
    with_hist = cut_short_lookback_warnings(mt, dt, fsl, hist, "roll_1")
    assert len(with_hist) == 2 and not any("'v'" in m for m in with_hist)
    assert cut_short_lookback_warnings(mt, dt, _first_step_link([("p", "t5", "p", "t6", -1)]), None,
                                       "roll_0") == []
    assert cut_short_lookback_warnings(mt, _dt([("p", "t4"), ("p", "t5")]), fsl, None, "solve") == []
