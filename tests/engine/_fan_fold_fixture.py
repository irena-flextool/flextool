"""Shared synthetic fan for the vectorized-fold parity gates (item 228).

One fan opens at ``d1`` into forecasts ``b`` / ``c`` (copies ``d1_b`` / ``d1_c``, own first step ``s1``) and
continues into ``d2`` (copies ``d2_b`` / ``d2_c``, own first step ``s2``); ``solve_data/branch_time_start`` puts
the fan start ``s1`` on every one of them, so the ``d2`` copies have two analysis-time tiers (own ``s2``, then
fan ``s1``).  The anchors carry a realized branch ``r`` (the parent-period fold), and ``d2_b`` lists ``b`` twice
(the S3 duplicate-row multiplicity: both twins add it once per occurrence).

Entities: ``X`` is stochastic (in an ``include_stochastics`` group), ``Y`` is not.  Values (hand-derived from the
rule "own first step, else the fan start, per value cell"):

* ``X`` (branch 1, own tier ``s2`` where authored, else fan ``s1``):
  ``d1_b/t1`` 1, ``d1_c/t1`` 10, ``d2_b/t3`` 2·300 = 600 (own, ``b`` twice), ``d2_b/t4`` 2·4 = 8 (fan),
  ``d2_c/t3`` 3000 (own), ``d2_c/t4`` 40 (fan).
* ``Y`` (branch 2 over the anchor's ``r``): ``d2_*/t3`` 5 (fan; ``s2`` authors only ``t4``),
  ``d2_*/t4`` 60 (own); ``d1_*/t1`` no fold value.

Rolling look-ahead variant (``rolling=True``: ``read_first`` = ``fan``, a roll of a rolling solve reads the fan
start ``s1`` -- the roll's own analysis time -- and never a forecast made later, at the look-ahead period's own
first step ``s2``).  It reads :data:`PBT_ROWS_ROLLING`: :data:`PBT_ROWS` plus ``Y``'s ``s1`` value at ``t4``, so the
roll's forecast covers its whole horizon:

* ``X``: ``d1_*`` unchanged (own step = fan start); ``d2_b/t3`` 2·3 = 6 (fan, ``b`` twice), ``d2_b/t4`` 2·4 = 8,
  ``d2_c/t3`` 30, ``d2_c/t4`` 40 (all fan; the ``s2`` values 300 / 3000 are not read).
* ``Y``: ``d2_*/t3`` 5 (fan), ``d2_*/t4`` 6 (fan; the ``s2`` value 60 is not read).

Uncovered rolling variant (``rolling=True`` over :data:`PBT_ROWS`): ``s1`` authors no ``t4`` for ``Y`` but ``s2``
does, so ``Y`` at ``d2_b/t4`` and ``d2_c/t4`` would read the later forecast -- an input error naming ``Y``, the
parameter, roll :data:`ROLL`, step ``t4`` of periods ``d2_b`` / ``d2_c`` and the analysis times ``s1`` / ``s2``
(:func:`uncovered_error`).  ``X`` is covered (``s1`` authors ``t3`` and ``t4``) and is not named.
"""
from __future__ import annotations

import re

import polars as pl

# (entity, branch, time_start, time, value)
PBT_ROWS: list[tuple[str, str, str, str, float]] = [
    ("X", "b", "s1", "t1", 1.0), ("X", "b", "s1", "t3", 3.0), ("X", "b", "s1", "t4", 4.0),
    ("X", "c", "s1", "t1", 10.0), ("X", "c", "s1", "t3", 30.0), ("X", "c", "s1", "t4", 40.0),
    ("X", "b", "s2", "t3", 300.0), ("X", "c", "s2", "t3", 3000.0),
    ("Y", "r", "s1", "t3", 5.0), ("Y", "r", "s2", "t4", 60.0),
]

# The rolling variant: the roll's fan-start forecast ``s1`` also covers ``Y`` at ``t4``.
PBT_ROWS_ROLLING: list[tuple[str, str, str, str, float]] = [*PBT_ROWS, ("Y", "r", "s1", "t4", 6.0)]

# The roll (``solve_data/solve_current``) of the rolling variants.
ROLL = "sol_roll_1"

# (entity, period, time) -> the fold value; a cell not listed has no fold value.
EXPECTED_FOLD: dict[tuple[str, str, str], float] = {
    ("X", "d1_b", "t1"): 1.0, ("X", "d1_c", "t1"): 10.0,
    ("X", "d2_b", "t3"): 600.0, ("X", "d2_b", "t4"): 8.0,
    ("X", "d2_c", "t3"): 3000.0, ("X", "d2_c", "t4"): 40.0,
    ("Y", "d2_b", "t3"): 5.0, ("Y", "d2_b", "t4"): 60.0,
    ("Y", "d2_c", "t3"): 5.0, ("Y", "d2_c", "t4"): 60.0,
}

EXPECTED_FOLD_ROLLING: dict[tuple[str, str, str], float] = {
    ("X", "d1_b", "t1"): 1.0, ("X", "d1_c", "t1"): 10.0,
    ("X", "d2_b", "t3"): 6.0, ("X", "d2_b", "t4"): 8.0,
    ("X", "d2_c", "t3"): 30.0, ("X", "d2_c", "t4"): 40.0,
    ("Y", "d2_b", "t3"): 5.0, ("Y", "d2_b", "t4"): 6.0,
    ("Y", "d2_c", "t3"): 5.0, ("Y", "d2_c", "t4"): 6.0,
}

# The parity cases: a single solve, a roll whose forecast covers its horizon, and one whose forecast does not.
FAN_CASES = ["single", "rolling_lookahead", "rolling_uncovered"]


def fan_case(case: str) -> "tuple[bool, list[tuple[str, str, str, str, float]]]":
    """``(rolling, pbt rows)`` of a :data:`FAN_CASES` case."""
    return {"single": (False, PBT_ROWS), "rolling_lookahead": (True, PBT_ROWS_ROLLING),
            "rolling_uncovered": (True, PBT_ROWS)}[case]


def uncovered_error(subject: str, param: str) -> str:
    """The ``pytest.raises(match=...)`` pattern of the uncovered rolling variant for entity *subject* (as the
    message names it, e.g. ``node 'Y'``) and parameter *param*."""
    return (re.escape(f"{subject}, parameter '{param}': roll '{ROLL}' of a rolling solve starts at time step "
                      "'s1'") + r".*" + re.escape("no value at time step 't4' (periods 'd2_b', 'd2_c')")
            + r".*" + re.escape("made later, at 's2'"))


def expected_fold(rolling: bool) -> dict[tuple[str, str, str], float]:
    """The hand-derived fold values: single solve (own first) or a roll's look-ahead (fan first)."""
    return EXPECTED_FOLD_ROLLING if rolling else EXPECTED_FOLD


STEPS: list[tuple[str, str]] = [("d1_b", "t1"), ("d1_c", "t1"), ("d2_b", "t3"), ("d2_b", "t4"),
                                ("d2_c", "t3"), ("d2_c", "t4")]


def put_fan_scaffold(put, *, rolling: bool = False) -> None:
    """Register the fan's period / branch / step frames through ``put(parent, stem, frame)``; *rolling* marks
    every ``branch_time_start`` row ``read_first`` = ``fan`` and names the roll :data:`ROLL` (a roll of a
    rolling solve)."""
    put("solve_data", "first_timesteps", pl.DataFrame({
        "period": ["d1_b", "d1_c", "d2_b", "d2_c"], "step": ["s1", "s1", "s2", "s2"]}))
    put("solve_data", "branch_time_start", pl.DataFrame({
        "period": ["d1", "d1_b", "d1_c", "d2", "d2_b", "d2_c"], "step": ["s1"] * 6,
        "read_first": ["fan" if rolling else "own"] * 6}))
    put("solve_data", "solve_branch__time_branch", pl.DataFrame({
        "period": ["d1_b", "d1_c", "d2_b", "d2_b", "d2_c", "d1", "d2"],
        "branch": ["b", "c", "b", "b", "c", "r", "r"]}))
    put("solve_data", "period__branch", pl.DataFrame({
        "period": ["d1", "d1", "d1", "d2", "d2", "d2"],
        "branch": ["d1", "d1_b", "d1_c", "d2", "d2_b", "d2_c"]}))
    put("solve_data", "steps_in_use", pl.DataFrame({
        "period": [d for d, _t in STEPS], "time": [t for _d, t in STEPS]}))
    put("input", "groupIncludeStochastics", pl.DataFrame({"group": ["g"]}))
    if rolling:
        put("solve_data", "solve_current", pl.DataFrame({"solve": [ROLL]}))


def pbt_frame(entity_cols: dict[str, "list[str] | str"], *, rows=PBT_ROWS) -> pl.DataFrame:
    """The pbt frame of *rows*: ``entity_cols`` maps the frame's leading columns to either the column holding
    the row's entity name (``"entity"``) or a constant value; then branch / time_start / time / value."""
    data: dict[str, list] = {}
    for col, src in entity_cols.items():
        data[col] = [r[0] if src == "entity" else src for r in rows]
    data["branch"] = [r[1] for r in rows]
    data["time_start"] = [r[2] for r in rows]
    data["time"] = [r[3] for r in rows]
    data["value"] = [r[4] for r in rows]
    return pl.DataFrame(data)
