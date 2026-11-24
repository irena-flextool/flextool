"""The realization-roll checks G1-G4 and the step-relation coverage check (T-g1-T-g4).

Every case is a synthetic split roll on four 1 h steps t1..t4: the base period ``p1`` (a period label with steps)
keeps the jump t1, t2; its branch periods ``p1_A`` and ``p1_B`` hold the steps after the jump, t3, t4.  A clean
case passes, the same case with one violating row raises the internal error naming the family or relation.  G2b
(a forecast without its own branch weight row) is ``_realization_tail.check_tail_weights``, tested in
``test_plan.py``.  The corpus case (every lp_snapshot fixture passes the checks, which are no-ops there: no
fixture has a label with steps) is the capture pin of ``tools/lp_snapshot.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import polars as pl
import pytest
from polar_high import Param, Problem

from flextool.engine.model import _period_rows as pr
from flextool.engine.preprocess import _step_relations as sr
from flextool.engine.preprocess.params._block_layout import DEFAULT_BLOCK, BlockLayout

STEPS = {"p1": ["t1", "t2"], "p1_A": ["t3", "t4"], "p1_B": ["t3", "t4"]}


def _frame(cols, rows):
    return pl.DataFrame(rows, schema={c: pl.Utf8 for c in cols}, orient="row")


def _data(*, label_steps=True, **fields):
    """The FlexData fields the checks read; ``label_steps=False``: today's fan (the label p1 has no dt rows)."""
    periods = STEPS if label_steps else {k: v for k, v in STEPS.items() if k != "p1"}
    layout = BlockLayout()
    layout.per_block_timeline = {DEFAULT_BLOCK: {p: [(t, 1.0) for t in ts] for p, ts in STEPS.items()}}
    # The clean links: p1_A / p1_B start from (p1, t2), the last jump step; p1's first step from a step of the
    # previous roll (t0, outside this build).
    dtttdt = _frame(("d", "t", "t_previous", "t_previous_within_timeset", "d_previous", "t_previous_within_solve"),
                    [("p1", "t1", "t0", "t0", "p1", "t0"), ("p1", "t2", "t1", "t1", "p1", "t1"),
                     *[(f, "t3", "t2", "t2", "p1", "t2") for f in ("p1_A", "p1_B")],
                     *[(f, "t4", "t3", "t3", f, "t3") for f in ("p1_A", "p1_B")]])
    base = dict(
        dt=_frame(("d", "t"), [(p, t) for p, ts in periods.items() for t in ts]),
        period_branch_full=_frame(("d", "b"), [("p1", "p1"), ("p1", "p1_A"), ("p1", "p1_B")]),
        block_layout=layout, dtttdt=dtttdt, nodeState_first_dt=None, nodeState_last_dt=None,
        pdt_branch_weight=None)
    base.update(fields)
    return SimpleNamespace(**base)


def _build(families, *, converted=(), **fields):
    """A ``_Build`` stand-in: per (family, over) of ``families`` a variable on ``over`` and the rows x <= 1."""
    m = Problem()
    for k, (name, over) in enumerate(families.items()):
        x = m.add_var(f"x{k}", tuple(over.columns), over)
        m.add_cstr(name, over=over, sense="<=", lhs_terms={"x": x}, rhs_terms={"one": 1.0})
    return SimpleNamespace(m=m, d=_data(**fields), path_converted=set(converted), solve_name="s_roll_0")


def _gd(*periods):
    return pl.DataFrame({"g": ["g1"] * len(periods), "d": list(periods)})


# ── T-g1: period rows ───────────────────────────────────────────────────────────────────────────────────────────

def test_g1_unclassified_period_family_on_the_label():
    # A (g, d) family that PERIOD_ROW_KINDS does not name is period-level by its columns: default-deny on p1.
    with pytest.raises(RuntimeError, match="Internal error: test_family on period 'p1' of roll 's_roll_0' is not "
                                           "path-aware. Please report this."):
        pr.check_period_rows(_build({"test_family": _gd("p1")}))
    # On the branch periods only it is no label row; with a step column it is no period row.
    pr.check_period_rows(_build({"test_family": _gd("p1_A", "p1_B")}))
    pr.check_period_rows(_build({"test_family": _gd("p1").with_columns(t=pl.lit("t1"))}))


def test_g1_path_family_on_a_branch_period_needs_the_path_helper():
    with pytest.raises(RuntimeError, match="co2_max_period on period 'p1_A'"):
        pr.check_period_rows(_build({"co2_max_period": _gd("p1_A", "p1_B")}))
    pr.check_period_rows(_build({"co2_max_period": _gd("p1_A", "p1_B")}, converted={"co2_max_period"}))
    # Converted rows sit on the branch periods; a row left on the label still raises.
    with pytest.raises(RuntimeError, match="co2_max_period on period 'p1'"):
        pr.check_period_rows(_build({"co2_max_period": _gd("p1", "p1_A")}, converted={"co2_max_period"}))


def test_g1_kinds_on_the_label():
    # path_at_step: registered with a step column; invest and rp (by prefix) on the label raise; fragment does not.
    usage = pl.DataFrame({"n": ["n1"], "d": ["p1_A"], "t": ["t4"]})
    with pytest.raises(RuntimeError, match="node_storage_usage_fix_le on period 'p1_A'"):
        pr.check_period_rows(_build({"node_storage_usage_fix_le": usage}))
    for name in ("maxInvest_entity_period_p", "rp_inter_period_some_new_row"):
        with pytest.raises(RuntimeError, match=f"{name} on period 'p1'"):
            pr.check_period_rows(_build({name: _gd("p1")}))
    pr.check_period_rows(_build({"commodity_ladder_balance": _gd("p1", "p1_A")}))


def test_g1_no_op_without_a_label_with_steps():
    # Today's fan: the label p1 has no dt rows, so even an unclassified label row passes (no-op).
    pr.check_period_rows(_build({"test_family": _gd("p1"), "co2_max_period": _gd("p1_A")}, label_steps=False))
    assert sr.labels_with_steps(_data(label_steps=False)) == {}
    assert sr.labels_with_steps(_data()) == {"p1": ["p1_A", "p1_B"]}


def test_unclassified_period_families_pin():
    m = _build({"test_family": _gd("p1"), "co2_max_period": _gd("p1"), "maxFlow_like": _gd("p1").with_columns(
        t=pl.lit("t1")), "coarse_like": _gd("p1").with_columns(b_first=pl.lit("t1"))}).m
    assert pr.unclassified_period_families(m) == ["test_family"]


# ── T-g2: branch weights ────────────────────────────────────────────────────────────────────────────────────────

def _weights(values):
    cells = [(p, t) for p, ts in STEPS.items() for t in ts]
    return Param(("d", "t"), pl.DataFrame({"d": [c[0] for c in cells][:len(values)],
                                           "t": [c[1] for c in cells][:len(values)],
                                           "value": [float(v) for v in values]}))


def test_g2_every_cell_has_a_positive_weight():
    pr.check_branch_weights(_build({}))   # no weight frame: every cell weighs 1
    # Label cells 1, forecasts 0.25 / 0.75.
    pr.check_branch_weights(_build({}, pdt_branch_weight=_weights([1, 1, .25, .25, .75, .75])))
    with pytest.raises(RuntimeError, match="pdt_branch_weight on period 'p1'"):
        pr.check_branch_weights(_build({}, pdt_branch_weight=_weights([0, 0, .25, .25, .75, .75])))
    with pytest.raises(RuntimeError, match="pdt_branch_weight on period 'p1_B'"):
        pr.check_branch_weights(_build({}, pdt_branch_weight=_weights([1, 1, .25, .25, .75, float("nan")])))
    with pytest.raises(RuntimeError, match="pdt_branch_weight on period 'p1_B'"):   # (p1_B, t4) has no row
        pr.check_branch_weights(_build({}, pdt_branch_weight=_weights([1, 1, .25, .25, .75])))


# ── T-g3 / T-g4: step relations ─────────────────────────────────────────────────────────────────────────────────

def test_g4_tail_starts_from_the_last_jump_step():
    sr.check_step_relations(_data(), "s_roll_0")
    d = _data()
    # Today's mid-horizon link: p1_A's first step continues from the previous anchor p0.
    d.dtttdt = d.dtttdt.with_columns(
        d_previous=pl.when((pl.col("d") == "p1_A") & (pl.col("t") == "t3")).then(pl.lit("p0"))
        .otherwise(pl.col("d_previous")))
    with pytest.raises(RuntimeError, match="dtttdt on period 'p1_A' of roll 's_roll_0'"):
        sr.check_step_relations(d, "s_roll_0")
    # No link row at all.
    d = _data()
    d.dtttdt = d.dtttdt.filter(~((pl.col("d") == "p1_B") & (pl.col("t") == "t3")))
    with pytest.raises(RuntimeError, match="dtttdt on period 'p1_B'"):
        sr.check_step_relations(d, "s_roll_0")


def test_g3_self_wrapping_tail_lag():
    # The within-period lag of p1_A's first step wraps to p1_A's own last step t4.
    lag = _frame(("d", "t", "t_previous_within_period"),
                 [("p1", "t1", "t0"), ("p1", "t2", "t1"), ("p1_A", "t3", "t4"), ("p1_A", "t4", "t3")])
    with pytest.raises(RuntimeError, match="dtttdt_within_period on period 'p1_A'"):
        sr.check_step_relations(_data(dtttdt_within_period=lag), "s_roll_0")
    # The same lag on the label's own first step (t1 -> t2).
    lag = _frame(("d", "t", "t_previous_within_period"), [("p1", "t1", "t2"), ("p1", "t2", "t1")])
    with pytest.raises(RuntimeError, match="dtttdt_within_period on period 'p1'"):
        sr.check_step_relations(_data(dtttdt_within_period=lag), "s_roll_0")
    # Not checked without a label with steps.
    sr.check_step_relations(_data(label_steps=False, dtttdt_within_period=lag), "s_roll_0")


def test_g3_delay_and_block_successors():
    # A 1-step delay from the label's last jump step t2 wraps to t1 (same name, before t2); a 0-step delay is the
    # own step, a sink in p1_A is no same-name link.
    delay = _frame(("d", "t_source", "t_sink", "td"), [("p1", "t2", "t1", "1")])
    with pytest.raises(RuntimeError, match="dtt__delay_duration on period 'p1'"):
        sr.check_step_relations(_data(dtt__delay_duration=delay), "s_roll_0")
    sr.check_step_relations(_data(dtt__delay_duration=_frame(("d", "t_source", "t_sink", "td"),
                                                            [("p1", "t2", "t2", "0"), ("p1", "t1", "t2", "1")])),
                            "s_roll_0")
    succ = _frame(("d", "b_first", "b_next"), [("p1_A", "t4", "t3")])   # the last block loops to the first
    with pytest.raises(RuntimeError, match="period_block_succ on period 'p1_A'"):
        sr.check_step_relations(_data(period_block_succ=succ), "s_roll_0")
    # Coarse-step membership: t3 and t4 in the coarse step starting at t3 is fine; starting at t4 is a wrap.
    member = _frame(("n", "d", "b_first", "t"), [("n1", "p1_A", "t3", "t3"), ("n1", "p1_A", "t3", "t4")])
    sr.check_step_relations(_data(node_coarse_step=member), "s_roll_0")
    with pytest.raises(RuntimeError, match="node_coarse_step on period 'p1_A'"):
        sr.check_step_relations(_data(node_coarse_step=_frame(("n", "d", "b_first", "t"),
                                                              [("n1", "p1_A", "t4", "t3")])), "s_roll_0")


def test_g3_scope_end_on_the_label():
    # A node's first step of the roll (p1, t1) wraps to the last step of its own period, (p1, t2).
    first = _frame(("n", "d", "t"), [("n1", "p1", "t1")])
    last = _frame(("n", "d", "t"), [("n1", "p1", "t2"), ("n1", "p1_A", "t4")])
    with pytest.raises(RuntimeError, match="scope_ends on period 'p1'"):
        sr.check_step_relations(_data(nodeState_first_dt=first, nodeState_last_dt=last), "s_roll_0")
    assert sr.scope_end_pairs(_data(nodeState_first_dt=first, nodeState_last_dt=last)).rows() == [
        ("n1", "p1", "t1", "t2")]


# ── coverage ───────────────────────────────────────────────────────────────────────────────────────────────────

# Every step-pair frame FlexData builds, with its columns (the lp_snapshot corpus census, plus dtttdt_within_solve
# which has the columns of dtttdt): min up/down look-backs, the delay frame, coarse and block frames, the
# timeline map.
FRAME_COLUMNS = {
    "dtttdt": ("d", "t", "t_previous", "t_previous_within_timeset", "d_previous", "t_previous_within_solve"),
    "dtttdt_forward_only": ("d", "t", "t_previous", "t_previous_within_timeset", "d_previous",
                            "t_previous_within_solve"),
    "dtttdt_within_solve": ("d", "t", "t_previous", "t_previous_within_timeset", "d_previous",
                            "t_previous_within_solve"),
    "dtttdt_within_period": ("d", "t", "t_previous_within_period"),
    "dtttdt_block_interior": ("d", "t", "t_previous"),
    "dtttdt_block_lag": ("d", "t", "d_previous", "t_previous_within_solve"),
    "dt_ramp_link": ("d", "t", "d_previous", "t_previous"),
    "dt_online_wrap": ("d", "t", "d_previous", "t_previous"),
    "dt_ramp_carry": ("d", "t", "d_previous", "t_previous", "duration_previous"),
    "uptime_lookback": ("p", "d", "t", "d_back", "t_back"),
    "downtime_lookback": ("p", "d", "t", "d_back", "t_back"),
    "dtt__delay_duration": ("d", "t_source", "t_sink", "td"),
    "dtt_timeline_matching": ("d", "t", "t_upper"),
    "node_coarse_interior": ("n", "d", "t", "t_previous"),
    "node_coarse_step": ("n", "d", "b_first", "t"),
    "process_coarse_step": ("p", "d", "b_first", "t", "w"),
    "period_block_time": ("d", "b_first", "t"),
    "period_block_succ": ("d", "b_first", "b_next"),
}


@dataclass
class _Fields:
    dtttdt: object = None
    uptime_lookback: object = None
    dtt__delay_duration: object = None
    p_flow: object = None
    block_chain: object = None
    prev_blocks: object = None
    lazy_link: object = None


def test_coverage_of_step_pair_frames():
    for name, cols in FRAME_COLUMNS.items():
        assert sum(map(sr.is_step_column, cols)) >= 2, name
        assert name in sr.STEP_RELATIONS, name
        rel = sr.STEP_RELATIONS[name]
        if rel.kind == sr.CHECKED and name != "scope_ends":
            used = {rel.own[1], *(c for _, c, _ in rel.links), *(c for c, _, _ in rel.links if c)}
            assert used <= set(cols), (name, used - set(cols))
        if rel.kind == sr.DERIVED:
            assert sr.STEP_RELATIONS[rel.note].kind == sr.CHECKED
    f = _Fields(dtttdt=pl.DataFrame(schema=dict.fromkeys(FRAME_COLUMNS["dtttdt"], pl.Utf8)),
                uptime_lookback=pl.DataFrame(schema=dict.fromkeys(FRAME_COLUMNS["uptime_lookback"], pl.Utf8)),
                dtt__delay_duration=pl.DataFrame(schema=dict.fromkeys(FRAME_COLUMNS["dtt__delay_duration"],
                                                                      pl.Utf8)))
    assert sr.unregistered_step_frames(f) == []
    # A Param with one step column, a frame with b_first / b_next, a b_prev* frame, a lazy frame: the last three
    # are not registered.
    f.p_flow = Param(("p", "d", "t"), pl.DataFrame({"p": ["u"], "d": ["p1"], "t": ["t1"], "value": [1.0]}))
    f.block_chain = pl.DataFrame(schema={"d": pl.Utf8, "b_first": pl.Utf8, "b_next": pl.Utf8})
    f.prev_blocks = pl.DataFrame(schema={"d": pl.Utf8, "b": pl.Utf8, "b_prev_block": pl.Utf8, "t": pl.Utf8})
    f.lazy_link = pl.LazyFrame(schema={"d": pl.Utf8, "t": pl.Utf8, "t_next": pl.Utf8})
    assert sr.unregistered_step_frames(f) == ["block_chain", "prev_blocks", "lazy_link"]
