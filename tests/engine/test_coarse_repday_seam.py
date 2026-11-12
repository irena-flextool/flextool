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


def _frames(group_step: float, rp: bool = True, *, steps: list[str] | None = None,
            realized: list[str] | None = None, rolling: bool = False):
    """Two 4-hour representative days inside one FlexTool period ``y`` (the
    days touch on the step list: d1h0 follows d0h3 within the timeset block),
    node ``store`` in group ``g``.  ``steps`` / ``realized`` / ``rolling``: a roll holding only ``steps`` and
    realizing ``realized``."""
    steps = DAY0 + DAY1 if steps is None else steps
    dt = pl.DataFrame({"d": ["y"] * len(steps), "t": steps})
    prev = [steps[-1]] + steps[:-1]          # one timeset block: d1h0's predecessor is d0h3
    dtttdt = pl.DataFrame({"d": ["y"] * len(steps), "t": steps, "t_previous_within_timeset": prev})
    return coarse_node_frames(
        node_group={"store": "g", "other": "default"}, group_step={"default": 1.0, "g": group_step},
        solve_step=1.0, dt=dt, step_duration=dt.with_columns(value=pl.lit(1.0)), dtttdt=dtttdt,
        rp_block_first=pl.DataFrame({"d": ["y", "y"], "t": ["d0h0", "d1h0"]}) if rp else None,
        realized=None if realized is None else pl.DataFrame({"d": ["y"] * len(realized), "t": realized}),
        balance_nodes=pl.DataFrame({"n": ["store", "other"]}), state_nodes=pl.DataFrame({"n": ["store"]}),
        rolling=rolling)


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
    assert "does not evenly divide the length of a representative period" in msg


@pytest.mark.parametrize("steps, realized, cause", [
    # The roll realizes up to d1h0, one hour into the second day: its realized end cuts the 2-h step d1h0-d1h1.
    (DAY0 + DAY1, DAY0 + ["d1h0"], "the realized part of this roll ends at step 'd1h0' (rolling_solve_jump)"),
    # The roll starts at d0h1 (the previous roll's realized end), one hour into the first day.
    (DAY0[1:] + DAY1, DAY0[1:] + DAY1, "this roll starts at step 'd0h1'"),
    # The roll's horizon ends at d1h2, three hours into the second day (it realizes the first day).
    (DAY0 + DAY1[:3], DAY0, "this roll's horizon ends at step 'd1h2' (rolling_solve_horizon)"),
], ids=["realized_end", "roll_start", "horizon_end"])
def test_roll_boundary_inside_a_representative_day_names_the_roll(steps, realized, cause) -> None:
    """A 2-h step divides the 4-h representative days, so a cut coarse step in a roll is the roll boundary's
    fault: the error names that boundary, not the representative-period length."""
    with pytest.raises(FlexToolConfigError) as exc:
        _frames(2.0, steps=steps, realized=realized, rolling=True)
    msg = str(exc.value)
    assert cause in msg
    assert "does not evenly divide" not in msg
    assert "'g'" in msg


def test_whole_days_in_a_roll_are_not_cut() -> None:
    """Control: a roll that holds and realizes whole representative days cuts no 2-h step."""
    assert _steps(_frames(2.0, steps=DAY0 + DAY1, realized=DAY0, rolling=True)) == {
        "d0h0": ["d0h0", "d0h1"], "d0h2": ["d0h2", "d0h3"], "d1h0": ["d1h0", "d1h1"], "d1h2": ["d1h2", "d1h3"]}
