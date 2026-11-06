"""Timeline breaks: the realized steps at which a time-series plot starts a new line.

A realized step is joined to the line before it when its ramp predecessor (``ramp_link`` = the model's step
adjacency: ``dt_ramp_link`` within a solve, ``dt_ramp_carry`` from the previous roll) is itself a realized step.
Every other step — the step after a gap in the timeline, the start of a representative block (also when two
blocks are adjacent on the timeline: they represent different base periods), the first step of a period that does
not continue the previous one (also a period that reuses the previous period's timeset), and the first step of a
roll that starts after a gap — is a break.  Roll seams that continue the timeline and periods that continue the
previous one are links, so they are drawn as one line.  The first output step is never a break.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

BREAK_LEVELS = ["period", "time"]
_PREV_LEVELS = ["period_previous", "time_previous"]

logger = logging.getLogger(__name__)


def empty_breaks() -> pd.MultiIndex:
    """An empty ``(period, time)`` break index."""
    return pd.MultiIndex.from_arrays([pd.Index([], dtype=object), pd.Index([], dtype=object)], names=BREAK_LEVELS)


def _as_str_steps(idx: pd.MultiIndex, levels: list[str]) -> pd.MultiIndex:
    """``idx`` restricted to ``levels`` (in that order) as plain-string ``(period, time)`` levels."""
    return pd.MultiIndex.from_arrays([idx.get_level_values(lv).astype(str) for lv in levels], names=BREAK_LEVELS)


def timeline_breaks(
    dt: pd.MultiIndex,
    ramp_link: pd.MultiIndex,
    *,
    block_first: pd.MultiIndex | None = None,
    scenario: str | None = None,
) -> pd.MultiIndex:
    """The ``(period, time)`` steps of ``dt`` at which a plot line starts again.

    ``dt``: the realized output steps ``(period, time)`` in output order.  ``ramp_link``: ``(period, time,
    period_previous, time_previous)``, the predecessor each linked step is adjacent to.  A step other than the first
    of ``dt`` is a break when it has no link or its link predecessor is not in ``dt``; the rule does not depend on
    the row order of ``dt``.

    An empty ``ramp_link`` gives no breaks: then the step adjacency is not known (or no step has an adjacent
    predecessor), and drawing every step on its own would make every plot unreadable.  That is logged at INFO,
    unless every step is a representative-block start (``block_first``), where no links are expected.
    """
    if len(dt) == 0:
        return empty_breaks()
    steps = _as_str_steps(dt, BREAK_LEVELS)
    if len(ramp_link) == 0:
        every_step_starts_a_block = (
            block_first is not None and len(block_first) > 0
            and bool(steps.isin(_as_str_steps(block_first, BREAK_LEVELS)).all()))
        if len(steps) > 1 and not every_step_starts_a_block:
            logger.info("Scenario %s: step adjacency unavailable; time-series plots are drawn without timeline "
                        "gaps.", scenario or "?")
        return empty_breaks()
    target = _as_str_steps(ramp_link, BREAK_LEVELS)
    pred = _as_str_steps(ramp_link, _PREV_LEVELS)
    keep = ~target.duplicated(keep="last")
    target, pred = target[keep], pred[keep]
    pos = target.get_indexer(steps)
    has_link = pos >= 0
    joined = np.zeros(len(steps), dtype=bool)
    joined[has_link] = pred.take(pos[has_link]).isin(steps)
    is_break = ~joined
    is_break[0] = False
    return steps[is_break]
