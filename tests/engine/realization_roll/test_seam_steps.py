"""Storage cycles of a split realization roll across the seam (design rev 7.1 §2.3, §8.1 T-st; design note G3 wraps).

A split roll R_k holds the period ``p1`` as one path per forecast: ``p1``'s jump steps, then the steps of one branch
period ``p1_<f>``.  A cyclic storage method closes per path: each branch period's first step continues from the
label's last jump step, the label's first step wraps to the end of d* (the last branch period in name order), and
``storage_split_path_end`` ends every other path at d*'s level (bind_within_period, looping
bind_intraperiod_blocks; bind_within_solve has ``storage_within_solve_path_end``).  A roll that continues the period
from the carry closes or anchors each path's end (the scope end of each branch period).

The model (``_seam``): one period ``p1`` of six 1 h steps t01..t06 (factor 8760 / 6 = 1460), a rolling
``within_solve`` solve, jump 2, horizon 4: R_0 = p1 t01-t02 + p1_A / p1_B t03-t04 (forecasts made at t03), R_1 =
p1 t03-t04 + tails t05-t06 (made at t05), R_2 = p1 t05-t06 (horizon cut at the timeline end, deterministic).  Node
``elec``: demand 1 MW, 1 MWh of lossless storage, fix_start 0 (storage_state_start 0).  ``gas`` 2 MW at 10 EUR/MWh in
t01-t02, 25 in t03-t04, 60 in t05-t06; ``peak`` 10 MW at 100; ``wind`` 1 MW (upper limit): realized 0, forecast
``A`` 1 and ``B`` 0, weight 0.5 each.

Hand derivation (the same for the three methods; s = the stored energy):
* R_0, path f = t01..t04, closing at 0 (the cyclic start pin 0 at d*'s end, and the tie): charging 1 MWh at 10 in
  the jump saves 25 at t03 under B (0.5 x 25 = 12.5 > 10); under A the wind covers the tail and the store empties
  into the curtailed wind.  Cost 3 x 10 + 0.5 x 25 = 42.5 -> 62 050; realized s(t02) = 1.  With the period cycling
  on each name (no seam) the store could not cross t02 -> t03: 20 + 0.5 x 50 = 45 -> 65 700.
* R_1 starts from s = 1 and closes each path at the period's start level 0 at t06 (scope end per path; the
  closing roll of bind_within_solve): keeping s for t05 under B costs 2 x 25 + 0.5 x 60 = 80 (refilling at t04 is
  the same); using it in the jump 25 + 0.5 x 120 = 85.  So 80 -> 116 800; realized s(t04) = 1.  Closed at the label's
  last jump step t04 instead, the store would have to empty in the jump: 85 -> 124 100.
* R_2 = t05-t06 from s = 1, closing at 0: 60 -> 87 600.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine import _solve_copy_models as scm
from tests.engine.realization_roll import test_split_roll_lp as sr
from tests.engine.realization_roll import test_trunk_sites as ts

pack = scm.pack
STEPS = ("t01", "t02", "t03", "t04", "t05", "t06")
OBJECTIVES = [62_050.0, 116_800.0, 87_600.0]


def _seam(method: str, n_steps: int = 6) -> dict:
    steps = STEPS[:n_steps]
    gas = {t: c for t, c in zip(steps, (10.0, 10.0, 25.0, 25.0, 60.0, 60.0))}
    entities = [["node", "elec", None], ["unit", "gas", None], ["unit", "peak", None], ["unit", "wind", None],
                ["profile", "wp", None], ["unit__outputNode", ["gas", "elec"], None],
                ["unit__outputNode", ["peak", "elec"], None], ["unit__outputNode", ["wind", "elec"], None],
                ["group__unit", ["sg", "wind"], None], ["group__node", ["sg", "elec"], None],
                ["unit__node__profile", ["wind", "elec", "wp"], None]]
    active = [("node", "elec"), ("unit", "gas"), ("unit", "peak"), ("unit", "wind")]
    made = [("t03", 2), ("t05", 4)][:n_steps // 2 - 1]
    wind = sr._series({"A": {at: {t: 1.0 for t in steps[i:i + 2]} for at, i in made},
                       "B": {at: {t: 0.0 for t in steps[i:i + 2]} for at, i in made},
                       "R": {at: {t: 0.0 for t in steps[i:]} for at, i in (("t01", 0), *made)}})
    values = [*sr._node("elec", "storage", pack(-1.0, "float")),
              ["node", "elec", "existing", pack(1.0, "float"), "init"],
              ["node", "elec", "storage_start_end_method", pack("fix_start", "str"), "init"],
              ["node", "elec", "storage_state_start", pack(0.0, "float"), "init"],
              ["node", "elec", "storage_binding_method", pack(method, "str"), "init"],
              *sr._unit("gas", "elec", 2.0, gas), *sr._unit("peak", "elec", 10.0, 100.0),
              *sr._unit("wind", "elec", 1.0, 0.0),
              ["unit__node__profile", ["wind", "elec", "wp"], "profile_method", pack("upper_limit", "str"), "init"],
              ["profile", "wp", "profile", wind, "init"]]
    fan = sr._fan([("A", {at: 0.5 for at, _ in made}, "no"), ("B", {at: 0.5 for at, _ in made}, "no"),
                   ("R", {at: 0.0 for at, _ in (("t01", 0), *made)}, "yes")])
    return sr._model(steps, jump=2, horizon=4, fan=fan, entities=entities, active=active, values=values)


def _rows_of(lp, prefix: str) -> dict:
    """``{row: {column: coefficient}}`` of the LP rows whose name starts with ``prefix``."""
    out: dict = {}
    for (row, col), v in lp.coefs.items():
        if row.startswith(prefix):
            out.setdefault(row, {})[col] = v
    return out


def _realized_state(steps, k: int, t: str) -> float:
    return sr._value(list(steps.values())[k].solution.value("v_state"), n="elec", d="p1", t=t)


def _common(built, steps, datas):
    """What every method shares: three rolls, the hand objectives and realized levels, the closing rows of R_1."""
    assert len(built) == 3
    assert sr._objectives(steps) == pytest.approx(OBJECTIVES, rel=1e-9)
    assert _realized_state(steps, 0, "t02") == pytest.approx(1.0, abs=1e-9)
    assert _realized_state(steps, 1, "t04") == pytest.approx(1.0, abs=1e-9)
    return [scm.lp_snapshot().parse_mps(b[1]) for b in built]


def test_bind_within_period_closes_each_path(tmp_path, monkeypatch):
    """T-st (bind_within_period).  R_0: each tail's first step continues from (p1, t02), the label's first step
    wraps to (p1_B, t04), the start pin 0 sits there and p1_A's end is tied to it.  R_1 continues the period from
    the carry (no wrap at (p1, t03)), each tail starts from (p1, t04), and each path closes at its own end t06 at
    the period's start level 0 (storage_state_roll_close)."""
    built, steps, datas = sr._run(_seam("bind_within_period"), tmp_path, monkeypatch)
    lps = _common(built, steps, datas)
    assert ts._rows(datas[0].dtttdt_within_period) == [
        ("p1", "t01", "p1_B", "t04"), ("p1", "t02", "p1", "t01"),
        ("p1_A", "t03", "p1", "t02"), ("p1_A", "t04", "p1_A", "t03"),
        ("p1_B", "t03", "p1", "t02"), ("p1_B", "t04", "p1_B", "t03")]
    balance = _rows_of(lps[0], "nodeBalance_eq[elec,p1,t01]")["nodeBalance_eq[elec,p1,t01]"]
    assert balance["v_state[elec,p1_B,t04]"] == pytest.approx(1.0)
    for f in ("p1_A", "p1_B"):
        assert _rows_of(lps[0], f"nodeBalance_eq[elec,{f},t03]")[f"nodeBalance_eq[elec,{f},t03]"][
            "v_state[elec,p1,t02]"] == pytest.approx(1.0)
    assert set(_rows_of(lps[0], "storage_state_start_binding_cyclic_period")) == {
        "storage_state_start_binding_cyclic_period[elec,p1_B,t04]"}
    assert _rows_of(lps[0], "storage_split_path_end") == {
        "storage_split_path_end[elec,p1_A,t04,p1_B,t04]": {"v_state[elec,p1_A,t04]": 1.0,
                                                           "v_state[elec,p1_B,t04]": -1.0}}
    # R_1: the carried first step has no wrap; the scope closes per path.
    first = _rows_of(lps[1], "nodeBalance_eq[elec,p1,t03]")["nodeBalance_eq[elec,p1,t03]"]
    assert [c for c in first if c.startswith("v_state")] == ["v_state[elec,p1,t03]"]
    for f in ("p1_A", "p1_B"):
        assert _rows_of(lps[1], f"nodeBalance_eq[elec,{f},t05]")[f"nodeBalance_eq[elec,{f},t05]"][
            "v_state[elec,p1,t04]"] == pytest.approx(1.0)
    assert set(_rows_of(lps[1], "storage_state_roll_close")) == {
        "storage_state_roll_close[elec,p1_A,t06]", "storage_state_roll_close[elec,p1_B,t06]"}
    assert not _rows_of(lps[1], "storage_split_path_end")


def test_bind_within_solve_closes_each_path(tmp_path, monkeypatch):
    """T-st (bind_within_solve, roll 0 and later rolls).  R_0 cycles over its horizon: the label's first step wraps
    to the last step of the last leaf (p1_B, t04) and storage_within_solve_path_end ties p1_A's end to it (one loop
    per path; the G4 check of the label's wrap).  R_1 and R_2 continue from the carry and, reaching the last step of
    the solve, close each leaf at the level the first roll started from (0)."""
    built, steps, datas = sr._run(_seam("bind_within_solve"), tmp_path, monkeypatch)
    lps = _common(built, steps, datas)
    links = {r[:2]: r[2:] for r in ts._rows(datas[0].dtttdt, "d", "t", "d_previous", "t_previous_within_solve")}
    assert links[("p1", "t01")] == ("p1_B", "t04")
    assert _rows_of(lps[0], "storage_within_solve_path_end") == {
        "storage_within_solve_path_end[elec,p1_A,t04,p1_B,t04]": {"v_state[elec,p1_A,t04]": 1.0,
                                                                  "v_state[elec,p1_B,t04]": -1.0}}
    assert set(_rows_of(lps[1], "storage_state_roll_close")) == {
        "storage_state_roll_close[elec,p1_A,t06]", "storage_state_roll_close[elec,p1_B,t06]"}


def test_bind_intraperiod_blocks_chain_through_the_seam(tmp_path, monkeypatch):
    """T-st blocks, a roll holding p1 in part.  Whole is decided per path (the label's steps, then a tail's,
    against the complete solve's steps of p1): R_0's paths t01..t04 and R_1's t03..t06 hold p1 (t01..t06) in part,
    so the blocks chain forward through the seam: each tail's block continues from the label's last jump step, the
    roll's first step from fix_start (R_0) or the carry (R_1).  R_0 anchors its leaves at the fix_start share (0);
    R_1 holds the period's end on each path and closes each at its own end t06.  The blocks are cut at the
    realized end, so each roll's blocks are its jump and its tails: the hand values hold block by block."""
    built, steps, datas = sr._run(_seam("bind_intraperiod_blocks"), tmp_path, monkeypatch)
    lps = _common(built, steps, datas)
    for d, (j0, j1, t0, t1) in zip(datas[:2], (("t01", "t02", "t03", "t04"), ("t03", "t04", "t05", "t06"))):
        assert sorted(ts._rows(d.period_block_chain)) == [("p1",), ("p1_A",), ("p1_B",)]
        assert ts._rows(d.dtttdt_block_lag) == [
            ("p1", j1, "p1", j0), ("p1_A", t0, "p1", j1), ("p1_A", t1, "p1_A", t0),
            ("p1_B", t0, "p1", j1), ("p1_B", t1, "p1_B", t0)]
        assert ts._rows(d.period_block_succ) == [("p1", j0, "None"), ("p1_A", t0, "None"), ("p1_B", t0, "None")]
    assert not _rows_of(lps[0], "storage_split_path_end")
    assert set(_rows_of(lps[1], "storage_state_roll_close")) == {
        "storage_state_roll_close[elec,p1_A,t06]", "storage_state_roll_close[elec,p1_B,t06]"}


def test_bind_intraperiod_blocks_loop_per_path(tmp_path, monkeypatch):
    """T-st blocks, a roll holding p1 whole (rk_blocks): the model on t01..t04 (factor 8760 / 4 = 2190; gas 10 in
    the jump t01-t02, 25 in t03-t04; R_1 = p1 t03-t04, deterministic).  R_0's paths t01-t02 + t03-t04 hold p1 whole:
    the blocks loop, the label's block continues from (p1_B, t04), where the start pin 0 sits, and p1_A's end is tied
    to it.  R_0 = 42.5 x 2190 = 93 075 (as above), realized s(t02) = 1; R_1 continues p1 from s = 1 and closes it at
    t04 at the start level 0: t03 from the store, t04 gas at 25 -> 54 750."""
    built, steps, datas = sr._run(_seam("bind_intraperiod_blocks", n_steps=4), tmp_path, monkeypatch)
    assert len(built) == 2
    assert sr._objectives(steps) == pytest.approx([93_075.0, 54_750.0], rel=1e-9)
    assert _realized_state(steps, 0, "t02") == pytest.approx(1.0, abs=1e-9)
    d0 = datas[0]
    assert d0.period_block_chain is None
    assert ts._rows(d0.dtttdt_block_lag) == [
        ("p1", "t01", "p1_B", "t04"), ("p1", "t02", "p1", "t01"),
        ("p1_A", "t03", "p1", "t02"), ("p1_A", "t04", "p1_A", "t03"),
        ("p1_B", "t03", "p1", "t02"), ("p1_B", "t04", "p1_B", "t03")]
    lp = scm.lp_snapshot().parse_mps(built[0][1])
    assert set(_rows_of(lp, "storage_state_start_binding_cyclic_period")) == {
        "storage_state_start_binding_cyclic_period[elec,p1_B,t04]"}
    assert _rows_of(lp, "storage_split_path_end") == {
        "storage_split_path_end[elec,p1_A,t04,p1_B,t04]": {"v_state[elec,p1_A,t04]": 1.0,
                                                           "v_state[elec,p1_B,t04]": -1.0}}


# ---------------------------------------------------------------------------
# G4: the label's first step wraps to a leaf's last step (storage_within_solve_path_end closes the other paths).
# ---------------------------------------------------------------------------

def test_g4_label_wrap_must_land_on_a_leaf_end():
    """test_guards' synthetic split roll (p1 = t1, t2; p1_A / p1_B = t3, t4; the leaves, which no period continues
    from, are p1_A and p1_B).  The label's first step linking to the previous roll (t0) or wrapping to (p1_B, t4)
    passes; a wrap to (p1_B, t3), not the leaf's end, closes one path only: G4 raises."""
    from flextool.engine.preprocess import _step_relations as rel
    from tests.engine.realization_roll import test_guards as tg

    rel.check_step_relations(tg._data(), "s_roll_0")
    for target, ok in ((("p1_B", "t4"), True), (("p1_B", "t3"), False)):
        d = tg._data()
        first = (pl.col("d") == "p1") & (pl.col("t") == "t1")
        d.dtttdt = d.dtttdt.with_columns(
            d_previous=pl.when(first).then(pl.lit(target[0])).otherwise(pl.col("d_previous")),
            t_previous_within_solve=pl.when(first).then(pl.lit(target[1])).otherwise(
                pl.col("t_previous_within_solve")))
        if ok:
            rel.check_step_relations(d, "s_roll_0")
        else:
            with pytest.raises(RuntimeError, match="dtttdt on period 'p1' of roll 's_roll_0'"):
                rel.check_step_relations(d, "s_roll_0")
