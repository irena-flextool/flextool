"""Solver-free unit tests for the two step-lag producers (design
``specs/step05_continuation_fanout_design.md`` §5.5).

``make_step_jump`` (``_timeline.py``) and ``dtttdt_from_source``
(``_derived_params.py``) must share the same cross-period wrap
semantics.  Byte-parity for pre-existing shapes is enforced with
hard-coded literals CAPTURED FROM PRE-CHANGE HEAD (main 643a8fb8):
the deterministic 2-period and single-period stochastic rows below are
verbatim pre-fix output, so any drift on those shapes fails loudly.
The multi-period stochastic rows are the §4 hand-derived expectations
(DEFECT PIN — pre-fix, branch periods self-cycled and realized
continuation periods took the positional previous, i.e. the last fan
member of the previous period).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars._derived_params import dtttdt_from_source
from flextool.engine_polars._solve_state import ActiveTimeEntry
from flextool.engine_polars._timeline import make_step_jump


def _ats(names_and_indices):
    return [ActiveTimeEntry(timestep=t, index=i, duration=1.0)
            for t, i in names_and_indices]


P1 = _ats([("t0001", 0), ("t0002", 1), ("t0003", 2)])
P2 = _ats([("t0004", 3), ("t0005", 4), ("t0006", 5)])

# ---------------------------------------------------------------------------
# Shared shapes
# ---------------------------------------------------------------------------

DET_ATL = {"p2035": P1, "p2040": P2}
DET_PB = [("p2035", "p2035"), ("p2040", "p2040")]
DET_SBTB: list[tuple[str, str]] = []

# Single-period stochastic: the 2day fan shape (base period + realized
# metadata row + three active branches), reduced to 3 steps.
SP_ATL = {
    "period1": P1,
    "period1_upper": P1[0:],
    "period1_lower": P1[0:],
    "period1_mid": P1[0:],
}
SP_PB = [
    ("period1", "period1"),
    ("period1", "period1_realized"),
    ("period1", "period1_upper"),
    ("period1", "period1_lower"),
    ("period1", "period1_mid"),
]
SP_SBTB = [
    ("period1_upper", "upper"),
    ("period1_lower", "lower"),
    ("period1_mid", "mid"),
    ("period1", "realized"),
]

# Multi-period stochastic (§4 shape).
MP_ATL = {"p2035": P1, "p2035_low": P1[0:],
          "p2040": P2, "p2040_low": P2[0:]}
MP_PB = [
    ("p2035", "p2035"), ("p2035", "p2035_rlz"), ("p2035", "p2035_low"),
    ("p2040", "p2040"), ("p2040", "p2040_rlz"), ("p2040", "p2040_low"),
]
MP_SBTB = [
    ("p2035_low", "low"), ("p2040_low", "low"),
    ("p2035", "rlz"), ("p2040", "rlz"),
]


# ---------------------------------------------------------------------------
# make_step_jump
# ---------------------------------------------------------------------------


def test_make_step_jump_deterministic_two_period_byte_parity() -> None:
    """Captured from PRE-change HEAD — must stay byte-identical."""
    assert make_step_jump(DET_ATL, DET_PB, DET_SBTB) == [
        ("p2035", "t0001", "t0003", "t0003", "p2040", "t0006", -5),
        ("p2035", "t0002", "t0001", "t0001", "p2035", "t0001", 1),
        ("p2035", "t0003", "t0002", "t0002", "p2035", "t0002", 1),
        ("p2040", "t0004", "t0006", "t0006", "p2035", "t0003", 1),
        ("p2040", "t0005", "t0004", "t0004", "p2040", "t0004", 1),
        ("p2040", "t0006", "t0005", "t0005", "p2040", "t0005", 1),
    ]


def test_make_step_jump_single_period_stochastic_byte_parity() -> None:
    """Captured from PRE-change HEAD (2day fan shape) — branch periods
    self-cycle, the base period wraps to the last dict key (retained
    quirk).  Must stay byte-identical."""
    assert make_step_jump(SP_ATL, SP_PB, SP_SBTB) == [
        ("period1", "t0001", "t0003", "t0003", "period1_mid", "t0003", -2),
        ("period1", "t0002", "t0001", "t0001", "period1", "t0001", 1),
        ("period1", "t0003", "t0002", "t0002", "period1", "t0002", 1),
        ("period1_upper", "t0001", "t0003", "t0003",
         "period1_upper", "t0003", -2),
        ("period1_upper", "t0002", "t0001", "t0001",
         "period1_upper", "t0001", 1),
        ("period1_upper", "t0003", "t0002", "t0002",
         "period1_upper", "t0002", 1),
        ("period1_lower", "t0001", "t0003", "t0003",
         "period1_lower", "t0003", -2),
        ("period1_lower", "t0002", "t0001", "t0001",
         "period1_lower", "t0001", 1),
        ("period1_lower", "t0003", "t0002", "t0002",
         "period1_lower", "t0002", 1),
        ("period1_mid", "t0001", "t0003", "t0003",
         "period1_mid", "t0003", -2),
        ("period1_mid", "t0002", "t0001", "t0001",
         "period1_mid", "t0001", 1),
        ("period1_mid", "t0003", "t0002", "t0002",
         "period1_mid", "t0002", 1),
    ]


def test_make_step_jump_multi_period_stochastic_continuation() -> None:
    """§4 hand-derived (DEFECT PIN): p2040 ← p2035 and p2040_low ←
    p2035_low; p2035_low self-cycles (branching-period fan); p2035
    keeps the first-period wrap quirk (last dict key = p2040_low)."""
    assert make_step_jump(MP_ATL, MP_PB, MP_SBTB) == [
        ("p2035", "t0001", "t0003", "t0003", "p2040_low", "t0006", -5),
        ("p2035", "t0002", "t0001", "t0001", "p2035", "t0001", 1),
        ("p2035", "t0003", "t0002", "t0002", "p2035", "t0002", 1),
        ("p2035_low", "t0001", "t0003", "t0003", "p2035_low", "t0003", -2),
        ("p2035_low", "t0002", "t0001", "t0001", "p2035_low", "t0001", 1),
        ("p2035_low", "t0003", "t0002", "t0002", "p2035_low", "t0002", 1),
        ("p2040", "t0004", "t0006", "t0006", "p2035", "t0003", 1),
        ("p2040", "t0005", "t0004", "t0004", "p2040", "t0004", 1),
        ("p2040", "t0006", "t0005", "t0005", "p2040", "t0005", 1),
        ("p2040_low", "t0004", "t0006", "t0006", "p2035_low", "t0003", 1),
        ("p2040_low", "t0005", "t0004", "t0004", "p2040_low", "t0004", 1),
        ("p2040_low", "t0006", "t0005", "t0005", "p2040_low", "t0005", 1),
    ]


# ---------------------------------------------------------------------------
# dtttdt_from_source (stub provider + stub source)
# ---------------------------------------------------------------------------


class StubProvider:
    """Minimal FlexDataProvider stand-in (has/get by canonical key)."""

    def __init__(self, frames: dict[str, pl.DataFrame]) -> None:
        self.frames = frames

    def has(self, key: str) -> bool:
        return key in self.frames

    def get(self, key: str) -> pl.DataFrame:
        return self.frames[key]


class StubSource:
    """InputSource stub — only timeline.timestep_duration is consulted."""

    def __init__(self, tl_frame: pl.DataFrame) -> None:
        self._tl = tl_frame

    def parameter(self, entity_class: str,
                  parameter_name: str) -> pl.DataFrame:
        if (entity_class, parameter_name) == (
            "timeline", "timestep_duration",
        ):
            return self._tl
        raise KeyError((entity_class, parameter_name))

    def entities(self, entity_class: str) -> pl.DataFrame:
        raise KeyError(entity_class)


TL = pl.DataFrame({
    "name": ["y6h"] * 6,
    "t": [f"t{i:04d}" for i in range(1, 7)],
    "value": [1.0] * 6,
})

# ``workdir`` only keys the provider lookups; no disk access happens.
WD = Path("/nonexistent_workdir_for_stub_provider")


def _frames(steps_rows, piu_rows, pb_rows, sbtb_rows):
    return {
        "solve_data/steps_in_use": pl.DataFrame(
            {"period": [r[0] for r in steps_rows],
             "step": [r[1] for r in steps_rows],
             "step_duration": [1.0] * len(steps_rows)}),
        "solve_data/period_in_use_set": pl.DataFrame(
            {"period": piu_rows}),
        "solve_data/period__branch": pl.DataFrame(
            {"period": [r[0] for r in pb_rows],
             "branch": [r[1] for r in pb_rows]},
            schema={"period": pl.Utf8, "branch": pl.Utf8}),
        "solve_data/solve_branch__time_branch": pl.DataFrame(
            {"period": [r[0] for r in sbtb_rows],
             "branch": [r[1] for r in sbtb_rows]},
            schema={"period": pl.Utf8, "branch": pl.Utf8}),
    }


def _steps(atl):
    return [(p, e.timestep) for p, entries in atl.items() for e in entries]


def test_dtttdt_deterministic_two_period_byte_parity() -> None:
    """Captured from PRE-change HEAD — must stay byte-identical.
    NB: first-of-period ``t_previous`` is the CROSS-period predecessor
    here (unlike step_previous.csv's within-period cyclic column —
    pre-existing divergence, design §3.3)."""
    frames = _frames(_steps(DET_ATL), ["p2035", "p2040"], DET_PB, DET_SBTB)
    df = dtttdt_from_source(StubSource(TL), "det", WD,
                            provider=StubProvider(frames))
    assert df.rows() == [
        ("p2035", "t0001", "t0006", "t0003", "p2040", "t0006"),
        ("p2035", "t0002", "t0001", "t0001", "p2035", "t0001"),
        ("p2035", "t0003", "t0002", "t0002", "p2035", "t0002"),
        ("p2040", "t0004", "t0003", "t0006", "p2035", "t0003"),
        ("p2040", "t0005", "t0004", "t0004", "p2040", "t0004"),
        ("p2040", "t0006", "t0005", "t0005", "p2040", "t0005"),
    ]


def test_dtttdt_single_period_stochastic_byte_parity() -> None:
    """Captured from PRE-change HEAD — branch periods self-wrap, the
    base period wraps to period_order[-1] (retained quirk).  Must stay
    byte-identical.  (Frame is sorted d, t — Utf8 lexicographic.)"""
    piu = ["period1", "period1_upper", "period1_lower", "period1_mid"]
    frames = _frames(_steps(SP_ATL), piu, SP_PB, SP_SBTB)
    df = dtttdt_from_source(StubSource(TL), "sp", WD,
                            provider=StubProvider(frames))
    assert df.rows() == [
        ("period1", "t0001", "t0003", "t0003", "period1_mid", "t0003"),
        ("period1", "t0002", "t0001", "t0001", "period1", "t0001"),
        ("period1", "t0003", "t0002", "t0002", "period1", "t0002"),
        ("period1_lower", "t0001", "t0003", "t0003",
         "period1_lower", "t0003"),
        ("period1_lower", "t0002", "t0001", "t0001",
         "period1_lower", "t0001"),
        ("period1_lower", "t0003", "t0002", "t0002",
         "period1_lower", "t0002"),
        ("period1_mid", "t0001", "t0003", "t0003",
         "period1_mid", "t0003"),
        ("period1_mid", "t0002", "t0001", "t0001",
         "period1_mid", "t0001"),
        ("period1_mid", "t0003", "t0002", "t0002",
         "period1_mid", "t0002"),
        ("period1_upper", "t0001", "t0003", "t0003",
         "period1_upper", "t0003"),
        ("period1_upper", "t0002", "t0001", "t0001",
         "period1_upper", "t0001"),
        ("period1_upper", "t0003", "t0002", "t0002",
         "period1_upper", "t0002"),
    ]


def test_dtttdt_multi_period_stochastic_continuation() -> None:
    """§4 hand-derived (DEFECT PIN): pre-fix, p2040_low self-wrapped
    (``wrap_period = period``) and p2040 wrapped to the positional
    previous ``p2035_low`` (the last fan member of p2035)."""
    piu = ["p2035", "p2035_low", "p2040", "p2040_low"]
    frames = _frames(_steps(MP_ATL), piu, MP_PB, MP_SBTB)
    df = dtttdt_from_source(StubSource(TL), "mp", WD,
                            provider=StubProvider(frames))
    assert df.rows() == [
        ("p2035", "t0001", "t0006", "t0003", "p2040_low", "t0006"),
        ("p2035", "t0002", "t0001", "t0001", "p2035", "t0001"),
        ("p2035", "t0003", "t0002", "t0002", "p2035", "t0002"),
        ("p2035_low", "t0001", "t0003", "t0003", "p2035_low", "t0003"),
        ("p2035_low", "t0002", "t0001", "t0001", "p2035_low", "t0001"),
        ("p2035_low", "t0003", "t0002", "t0002", "p2035_low", "t0002"),
        ("p2040", "t0004", "t0003", "t0006", "p2035", "t0003"),
        ("p2040", "t0005", "t0004", "t0004", "p2040", "t0004"),
        ("p2040", "t0006", "t0005", "t0005", "p2040", "t0005"),
        ("p2040_low", "t0004", "t0003", "t0006", "p2035_low", "t0003"),
        ("p2040_low", "t0005", "t0004", "t0004", "p2040_low", "t0004"),
        ("p2040_low", "t0006", "t0005", "t0005", "p2040_low", "t0005"),
    ]


def test_dtttdt_raises_when_time_branch_map_missing() -> None:
    """Failure semantics (round-2 review correction 3a): branch periods
    declared in period__branch but no solve_branch__time_branch frame →
    RAISE, never silently self-wrap (that would reintroduce the silent
    truncation bug)."""
    piu = ["p2035", "p2035_low", "p2040", "p2040_low"]
    frames = _frames(_steps(MP_ATL), piu, MP_PB, MP_SBTB)
    del frames["solve_data/solve_branch__time_branch"]
    with pytest.raises(ValueError, match="solve_branch__time_branch"):
        dtttdt_from_source(StubSource(TL), "mp", WD,
                           provider=StubProvider(frames))
