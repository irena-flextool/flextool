"""Coarse time steps must respect representative-period seams.

Representative periods are days sampled from across the year and concatenated
into one FlexTool period.  A coarse ``new_stepduration`` step of a node must
therefore restart at every representative-block start: a coarse step that
reached across the seam would balance energy (and keep one storage state) over
days that are months apart.  ``params._coarse.coarse_node_frames`` cuts the
coarse steps at every ``rp_block_first`` step and rejects, with a plain-English
``FlexToolConfigError``, a ``new_stepduration`` that does not divide the
representative-period length.

The blended-weights storage chain at coarse resolution (each representative
day's first coarse row starts from ``v_state_rp_start``) is pinned in
``tests/engine/test_coarse_groups.py::test_rp_coarse_rows_start_from_rp_state``.
"""
from __future__ import annotations

import polars as pl
import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.preprocess.params._coarse import coarse_node_frames

DAY0 = ["d0h0", "d0h1", "d0h2", "d0h3"]
DAY1 = ["d1h0", "d1h1", "d1h2", "d1h3"]


def _frames(group_step: float, rp: bool = True):
    """Two 4-hour representative days inside one FlexTool period ``y`` (the
    days touch on the step list: d1h0 follows d0h3 within the timeset block),
    node ``store`` in group ``g``."""
    steps = DAY0 + DAY1
    dt = pl.DataFrame({"d": ["y"] * 8, "t": steps})
    prev = [steps[-1]] + steps[:-1]          # one timeset block: d1h0's predecessor is d0h3
    dtttdt = pl.DataFrame({"d": ["y"] * 8, "t": steps, "t_previous_within_timeset": prev})
    return coarse_node_frames(
        node_group={"store": "g", "other": "default"}, group_step={"default": 1.0, "g": group_step},
        solve_step=1.0, dt=dt, step_duration=dt.with_columns(value=pl.lit(1.0)), dtttdt=dtttdt,
        rp_block_first=pl.DataFrame({"d": ["y", "y"], "t": ["d0h0", "d1h0"]}) if rp else None,
        balance_nodes=pl.DataFrame({"n": ["store", "other"]}), state_nodes=pl.DataFrame({"n": ["store"]}))


def _steps(frames) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for _n, _d, b, t in frames.node_coarse_step.iter_rows():
        out.setdefault(b, []).append(t)
    return out


def test_coarse_steps_restart_at_each_representative_day() -> None:
    """2-h coarse steps: two per representative day, none crossing the seam."""
    frames = _frames(2.0)
    assert _steps(frames) == {"d0h0": ["d0h0", "d0h1"], "d0h2": ["d0h2", "d0h3"],
                              "d1h0": ["d1h0", "d1h1"], "d1h2": ["d1h2", "d1h3"]}
    assert frames.nodeCoarse["n"].to_list() == ["store"]
    # The state is constant inside each coarse step only (no link d0h3 -> d1h0).
    assert sorted(frames.node_coarse_interior.select("t", "t_previous").iter_rows()) == [
        ("d0h1", "d0h0"), ("d0h3", "d0h2"), ("d1h1", "d1h0"), ("d1h3", "d1h2")]


def test_day_long_step_without_representative_periods_spans_the_block() -> None:
    """Control: without representative blocks the two days are one timeset block, so an 8-h step covers
    both."""
    assert _steps(_frames(8.0, rp=False)) == {"d0h0": DAY0 + DAY1}


def test_step_straddling_a_representative_seam_raises() -> None:
    """A 3-h step does not divide a 4-h representative day: a plain-English configuration error."""
    with pytest.raises(FlexToolConfigError) as exc:
        _frames(3.0)
    msg = str(exc.value)
    assert "representative period" in msg
    assert "new_stepduration" in msg
    assert "'g'" in msg
