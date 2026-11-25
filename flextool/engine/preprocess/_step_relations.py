"""The step-pair relations of a build and the realization-roll checks G3 (no self-wrap) and G4 (tail link).

A *step-pair relation* is a FlexData frame that links a step to another step (a predecessor, a successor, a coarse
or block neighbour).  Under a split realization roll the base period ``p`` keeps its jump steps and each branch
period ``p_f`` holds the steps after the jump; a relation built per period name then wraps inside ``p`` or inside
``p_f`` (a *self-wrap*) where it should cross the seam from ``p``'s last jump step to ``p_f``'s first step.  The
checks turn such a missed case into an internal error, not a different LP.  They do nothing unless the build has
a period label with steps (``labels_with_steps``), which only a split realization roll makes.

``STEP_RELATIONS`` classifies every relation (``checked`` / ``derived`` / ``exempt``);
``unregistered_step_frames`` is its coverage check: every FlexData frame with two step columns is registered.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import polars as pl

from flextool.engine.input._axis_enums import _AXIS_SYNONYMS
from flextool.engine.preprocess.params._block_layout import DEFAULT_BLOCK
from flextool.engine.preprocess.params._branch import fan_period_labels

CHECKED, DERIVED, EXEMPT = "checked", "derived", "exempt"
BEFORE, AT_OR_BEFORE, AFTER = "before", "at_or_before", "after"


@dataclass(frozen=True)
class StepRelation:
    """One step-pair relation.  ``own``: its (period, step) columns.  ``links``: (period column, step column, side)
    of each linked step; period column ``None`` = the own period; side ``before`` (a predecessor), ``at_or_before``
    (the first step of the coarse step or block holding the own step) or ``after`` (a successor).  ``zero_lag``: a column whose value 0 marks a link to the own step (not a wrap).  ``note``: for a
    ``derived`` relation the relation it follows, for an ``exempt`` one why it is not a link inside the solve."""
    kind: str
    links: tuple[tuple[str | None, str, str], ...] = ()
    own: tuple[str, str] = ("d", "t")
    zero_lag: str | None = None
    note: str = ""


_DTTTDT = ((None, "t_previous", BEFORE), (None, "t_previous_within_timeset", BEFORE),
           ("d_previous", "t_previous_within_solve", BEFORE))
_LINK = (("d_previous", "t_previous", BEFORE),)

STEP_RELATIONS: dict[str, StepRelation] = {
    "dtttdt": StepRelation(CHECKED, _DTTTDT),
    "dtttdt_forward_only": StepRelation(CHECKED, _DTTTDT),
    "dtttdt_within_solve": StepRelation(CHECKED, _DTTTDT),
    "dtttdt_block_interior": StepRelation(CHECKED, ((None, "t_previous", BEFORE),)),
    "dtttdt_within_period": StepRelation(CHECKED, (("d_previous", "t_previous_within_period", BEFORE),)),
    "dt_online_wrap": StepRelation(CHECKED, _LINK),
    "dt_ramp_link": StepRelation(CHECKED, _LINK),
    "dt_ramp_carry": StepRelation(CHECKED, _LINK),
    "dtt__delay_duration": StepRelation(CHECKED, ((None, "t_sink", AFTER),), own=("d", "t_source"),
                                        zero_lag="td"),
    "node_coarse_interior": StepRelation(CHECKED, ((None, "t_previous", BEFORE),)),
    "dtttdt_block_lag": StepRelation(CHECKED, (("d_previous", "t_previous_within_solve", BEFORE),)),
    "period_block_succ": StepRelation(CHECKED, ((None, "b_next", AFTER),), own=("d", "b_first")),
    "period_block_time": StepRelation(CHECKED, ((None, "b_first", AT_OR_BEFORE),)),
    "node_coarse_step": StepRelation(CHECKED, ((None, "b_first", AT_OR_BEFORE),)),
    "process_coarse_step": StepRelation(CHECKED, ((None, "b_first", AT_OR_BEFORE),)),
    # Not a FlexData field: the pairs of model.core._first_scope_ends (scope_end_pairs).
    "scope_ends": StepRelation(CHECKED, (("d_end", "t_end", BEFORE),)),
    "uptime_lookback": StepRelation(DERIVED, note="dt_ramp_link"),
    "downtime_lookback": StepRelation(DERIVED, note="dt_ramp_link"),
    "dtt_timeline_matching": StepRelation(EXEMPT, note="maps a lower solve's steps to the upper solve's steps"),
    # Model-internal step pairs (not FlexData fields), reviewed: each pairs two one-step frames, never a step with
    # its neighbour in the same period.
    "online_successor_steps": StepRelation(DERIVED, note="dt_ramp_link"),   # _online._successor_steps, its inverse
    "storage_state_end_match": StepRelation(
        EXEMPT, note="nodeState_first_dt x nodeState_last_dt of period_last: a start and the model end"),
    "storage_within_solve_path_end": StepRelation(
        EXEMPT, note="nodeState_last_dt x nodeState_last_dt: the last step of each last period and of d*"),
    "storage_split_path_end": StepRelation(
        EXEMPT, note="nodeState_last_dt x nodeState_last_dt: the last step of each branch period and of d*"),
    "storage_state_fan_start": StepRelation(
        EXEMPT, note="nodeState_first_dt x nodeState_first_dt: the first step of each root of a first-period fan"),
}

# The storage_roll_end kinds whose nodes read the scope ends (model.core._first_scope_ends).
SCOPE_END_KINDS = ("scope_anchor", "scope_close")

# Step columns: the synonyms of the step axis, the t_previous* / b_prev* families and the other step-valued columns.
_STEP_NAMES = frozenset({k for k, v in _AXIS_SYNONYMS.items() if v == "t"}
                        | {"t", "t_next", "t_end", "t_b", "b_first", "b_next"})


def is_step_column(name: str) -> bool:
    return name in _STEP_NAMES or name.startswith(("t_previous", "b_prev"))


def _columns(value) -> list[str]:
    if isinstance(value, pl.DataFrame):
        return value.columns
    if isinstance(value, pl.LazyFrame):
        return value.collect_schema().names()
    dims = getattr(value, "dims", None)  # a Param
    return list(dims) if isinstance(dims, tuple) else []


def unregistered_step_frames(d) -> list[str]:
    """The FlexData fields of ``d`` holding a frame with two or more step columns that ``STEP_RELATIONS`` does not
    classify (the coverage check; empty when complete)."""
    return [f.name for f in dataclasses.fields(d)
            if f.name not in STEP_RELATIONS
            and sum(is_step_column(c) for c in _columns(getattr(d, f.name))) >= 2]


def labels_with_steps(d) -> dict[str, list[str]]:
    """``{p: [p_f, ...]}``: the period labels of the build that have dt rows (L*) and their branch periods (B(L*)),
    in ``period__branch`` order.  Empty without a split realization roll.  A branch period is listed whether or not
    it has dt rows: a split roll's tail without steps is a plan error, which the checks then raise (the timeline,
    G4)."""
    pbf = d.period_branch_full
    if pbf is None or pbf.height == 0:
        return {}
    return _split_labels(pbf, d.dt)


def _split_labels(period_branch: pl.DataFrame, steps: pl.DataFrame) -> dict[str, list[str]]:
    """``labels_with_steps`` from ``period_branch`` (d, b) and the build's steps ``steps`` (d, t)."""
    pairs = list(zip(period_branch["d"].cast(pl.Utf8).to_list(), period_branch["b"].cast(pl.Utf8).to_list()))
    labels = fan_period_labels(set(pairs)) & set(steps["d"].cast(pl.Utf8).unique().to_list())
    return {p: [b for a, b in pairs if a == p and b != p] for p in sorted(labels)}


def split_first_steps(period_branch: pl.DataFrame | None, steps: pl.DataFrame | None) -> pl.DataFrame:
    """``(d, t)`` Utf8: the first step (in the row order of ``steps``, solve order) of each label with steps and
    of each of its branch periods (L* and B(L*)).  Empty without a split realization roll."""
    empty = pl.DataFrame(schema={"d": pl.Utf8, "t": pl.Utf8})
    if period_branch is None or period_branch.height == 0 or steps is None or steps.height == 0:
        return empty
    lab = _split_labels(period_branch, steps)
    periods = [*lab, *(f for fs in lab.values() for f in fs)]
    return (steps.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8)).with_row_index("__i")
            .filter(pl.col("d").is_in(periods))
            .group_by("d", maintain_order=True).agg(pl.col("t").sort_by("__i").first()))


def null_split_placeholders(frame: pl.DataFrame | None, firsts: pl.DataFrame,
                            columns: tuple[str, ...], step: str = "t") -> pl.DataFrame | None:
    """``frame`` with ``columns`` null on its rows whose (``d``, ``step``) is at ``firsts`` (``split_first_steps``).

    Under a split realization roll the label and its branch periods are no periods on a path, so the period-cyclic
    placeholders at their first steps (``dtttdt`` ``t_previous`` / ``t_previous_within_timeset``: the period's
    own last step; the ``dt_online_wrap`` free-start row) have no same-name predecessor.  No row reads them there
    (the seam link ``(d_previous, t_previous_within_solve)``, ``dt_ramp_link`` and the roll carry give the real
    predecessors), so they are null, and the frames derived from ``frame`` inherit that."""
    if frame is None or frame.height == 0 or firsts.height == 0:
        return frame
    hit = (frame.select(pl.col("d").cast(pl.Utf8), pl.col(step).cast(pl.Utf8).alias("t"))
           .join(firsts.with_columns(__hit=pl.lit(True)), on=["d", "t"], how="left", maintain_order="left")
           ["__hit"].fill_null(False))
    return frame.with_columns(pl.when(pl.lit(hit)).then(None).otherwise(pl.col(c)).alias(c) for c in columns)


def null_split_block_wraps(succ: pl.DataFrame | None, period_branch: pl.DataFrame | None,
                           steps: pl.DataFrame | None) -> pl.DataFrame | None:
    """``period_block_succ`` (d, b_first, b_next) with ``b_next`` null on the row of each split label's and branch
    period's last block, whose successor is the period's own first block (the loop of a period).  Under a split
    realization roll the label's last block continues in each branch period and a branch period's last block (where
    the path loops) in the label's first block: those successors lie in another period, in ``dtttdt_block_lag``
    (the state lag the model reads; nothing reads ``period_block_succ``).  Unchanged without a label with steps."""
    return null_split_placeholders(succ, split_first_steps(period_branch, steps), ("b_next",), step="b_next")


def not_path_aware(what: str, period: str, solve: str | None) -> RuntimeError:
    return RuntimeError(f"Internal error: {what} on period '{period}' of roll '{solve}' is not path-aware. "
                        "Please report this.")


def scope_end_pairs(d) -> pl.DataFrame | None:
    """``(n, d, t, d_end, t_end)``, Utf8: for each node with a ``SCOPE_END_KINDS`` row of storage_roll_end (the nodes
    whose scope end the model reads), its first step of the solve (nodeState_first_dt) and the step it wraps to, the
    last step of its period (nodeState_last_dt).  In a split realization roll a first step on a label with steps
    has one scope end per path: the last step of each of its branch periods.  None without rows."""
    first, last = d.nodeState_first_dt, d.nodeState_last_dt
    sre = getattr(d, "storage_roll_end", None)
    if first is None or last is None or sre is None or first.height == 0 or last.height == 0:
        return None
    nodes = sre.filter(pl.col("kind").cast(pl.Utf8).is_in(SCOPE_END_KINDS)).select(pl.col("n").cast(pl.Utf8))
    ends = last.select(pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8).alias("d_end"),
                       pl.col("t").cast(pl.Utf8).alias("t_end"))
    lab = labels_with_steps(d)
    route = pl.DataFrame([(p, f) for p in ends["d_end"].unique().sort().to_list() for f in lab.get(p) or [p]],
                         schema={"d": pl.Utf8, "d_end": pl.Utf8}, orient="row")
    out = (first.select(pl.col(c).cast(pl.Utf8) for c in ("n", "d", "t"))
           .join(nodes, on="n", how="semi")
           .join(route, on="d", how="inner")
           .join(ends, on=["n", "d_end"], how="inner"))
    return out if out.height > 0 else None


def _solve_timeline(d) -> dict:
    """``{period: [(step, duration), ...]}``: the solve's timeline (``d.block_layout``), periods in solve order."""
    return {} if d.block_layout is None else d.block_layout.per_block_timeline.get(DEFAULT_BLOCK, {})


def _period_steps(d, periods, solve: str | None) -> dict[str, list[str]]:
    """The steps of each of ``periods`` in timeline order (the solve's timeline in ``d.block_layout``)."""
    timeline = _solve_timeline(d)
    out = {}
    for p in periods:
        if not timeline.get(p):
            raise not_path_aware("the timeline", p, solve)
        out[p] = [s for s, _ in timeline[p]]
    return out


def check_step_relations(d, solve: str | None) -> None:
    """G4: the dtttdt row of each branch period's first step links to (p, p's last jump step), and the label's first
    step links to an earlier period or wraps to the last step of a leaf (one wrap that closes every path).  G3: no ``checked``
    relation links a step of a label with steps or of its branch periods to a step of the same period on the wrong
    side (a self-wrap; a linked step outside the period's steps of this build cannot be a wrap).  No-op without a
    label with steps."""
    lab = labels_with_steps(d)
    if not lab:
        return
    steps = _period_steps(d, [*lab, *(f for fs in lab.values() for f in fs)], solve)
    links = (pl.DataFrame(schema={"d": pl.Utf8, "t": pl.Utf8, "d_previous": pl.Utf8, "t_previous_within_solve": pl.Utf8})
             if d.dtttdt is None else d.dtttdt.select(pl.col(c).cast(pl.Utf8) for c in
                                                      ("d", "t", "d_previous", "t_previous_within_solve")))
    order = {k: i for i, k in enumerate(_solve_timeline(d))}   # the solve's period order (its timeline), not dtttdt's rows
    wraps = {(p, steps[p][0]) for p in lab}
    leaves = set(links["d"].to_list()) - {dp for dd, tt, dp, _ in links.rows() if dp != dd and (dd, tt) not in wraps}
    for p, branches in lab.items():
        for f in branches:
            row = links.filter((pl.col("d") == f) & (pl.col("t") == steps[f][0]))
            if row.rows() != [(f, steps[f][0], p, steps[p][-1])]:
                raise not_path_aware("dtttdt", f, solve)
        # The label's first step: its within-solve link is the step before it (an earlier period of the solve, or
        # the previous roll), or the bind_within_solve wrap of the solve's first step.  A wrap closes every path
        # only when it lands on the last step of a leaf (storage_within_solve_path_end ties the other leaves' ends
        # to it): a wrap to a later period that is no leaf, or to a step before a leaf's end, closes one path only.
        for _, _, dp, tp in links.filter((pl.col("d") == p) & (pl.col("t") == steps[p][0])).rows():
            later = dp in order and order[dp] > order[p]
            if (dp in leaves and tp != _period_steps(d, [dp], solve)[dp][-1]) or (later and dp not in leaves):
                raise not_path_aware("dtttdt", p, solve)
    pos = pl.DataFrame([(p, t, i) for p, ts in steps.items() for i, t in enumerate(ts)],
                       schema={"_d": pl.Utf8, "_t": pl.Utf8, "_i": pl.Int64}, orient="row")
    for name, rel in STEP_RELATIONS.items():
        if rel.kind != CHECKED:
            continue
        frame = scope_end_pairs(d) if name == "scope_ends" else getattr(d, name, None)
        if frame is None or frame.height == 0:
            continue
        od, ot = rel.own
        for pcol, scol, side in rel.links:
            lag = pl.lit(1.0) if rel.zero_lag is None else pl.col(rel.zero_lag).cast(pl.Float64)
            pairs = (frame.select(pl.col(od).cast(pl.Utf8).alias("_d"), pl.col(ot).cast(pl.Utf8).alias("_t"),
                                  pl.col(pcol or od).cast(pl.Utf8).alias("_od"),
                                  pl.col(scol).cast(pl.Utf8).alias("_ot"), lag.floor().alias("_lag"))
                     .filter(pl.col("_od") == pl.col("_d"))
                     .join(pos, on=["_d", "_t"], how="inner")
                     .join(pos.rename({"_d": "_od", "_t": "_ot", "_i": "_oi"}), on=["_od", "_ot"], how="inner"))
            wrong = {BEFORE: pl.col("_oi") >= pl.col("_i"), AT_OR_BEFORE: pl.col("_oi") > pl.col("_i"),
                     AFTER: (pl.col("_oi") <= pl.col("_i")) & (pl.col("_lag") != 0)}[side]
            bad = pairs.filter(wrong)
            if bad.height > 0:
                raise not_path_aware(name, bad["_d"][0], solve)


def null_split_dtttdt(dtttdt: pl.DataFrame | None, period_branch: pl.DataFrame | None) -> pl.DataFrame | None:
    """``dtttdt`` with ``t_previous`` / ``t_previous_within_timeset`` null at the first steps of a split
    realization roll's label and branch periods (``null_split_placeholders``); unchanged otherwise.  ``dtttdt``
    holds one row per step of the solve in solve order, so its own rows give the first steps."""
    cols = tuple(c for c in ("t_previous", "t_previous_within_timeset") if dtttdt is not None and c in dtttdt.columns)
    return null_split_placeholders(dtttdt, split_first_steps(period_branch, dtttdt), cols)
