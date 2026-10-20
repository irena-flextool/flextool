"""Coherence / fracture-guard for the realized-dispatch (``separate_dispatch``)
feature.

The sibling's configuration is DEFINED TWICE, in two vocabularies (see the
feature map at ``_orchestration._spawn_realized_dispatch_siblings``):

* the RAW Spine-param half — ``SpineDbReader._REALIZED_DISPATCH_SIBLING_PARAMS``
  (cloned so the sibling takes the full override path), and
* the DERIVED-dict half — the copy/set in
  ``SolveConfig.spawn_realized_dispatch_sibling`` (for ``native_run_model``).

The four DEFINING facts must agree across both halves:
``period_timeset`` ↔ ``timesets_used_by_solves``,
``years_represented`` ↔ ``solve_period_years_represented``,
``solve_mode`` ↔ ``solve_modes``, ``realized_periods`` ↔ ``realized_periods``.

Nothing structurally forces the two lists to stay aligned (they live in
different layers / vocabularies), so this test is the fracture guard: adding a
defining fact to one half without the other, or drifting the reader allow-list,
fails here loudly.
"""
from __future__ import annotations

from flextool.engine_polars._solve_config import SolveConfig
from flextool.engine_polars._spinedb_reader import SpineDbReader

#: The four defining realized-dispatch facts, RAW Spine param -> DERIVED
#: SolveConfig dict.  This mapping IS the single declaration of the
#: dual-definition correspondence the two halves must honour.
_DEFINING_FACT_MAP = {
    "period_timeset": "timesets_used_by_solves",
    "years_represented": "solve_period_years_represented",
    "solve_mode": "solve_modes",
    "realized_periods": "realized_periods",
}


def test_reader_allow_list_matches_the_defining_facts():
    """The reader's raw-param allow-list is exactly the four defining facts.

    If this drifts, update ``_DEFINING_FACT_MAP`` AND the SolveConfig
    ``spawn_realized_dispatch_sibling`` copy/set in lockstep.
    """
    assert set(SpineDbReader._REALIZED_DISPATCH_SIBLING_PARAMS) == set(
        _DEFINING_FACT_MAP
    ), (
        "SpineDbReader._REALIZED_DISPATCH_SIBLING_PARAMS drifted from the four "
        "defining realized-dispatch facts — the reader half and the SolveConfig "
        "half of the sibling config are no longer in lockstep (feature map: "
        "_orchestration._spawn_realized_dispatch_siblings)."
    )


def _parent_solveconfig_stub() -> SolveConfig:
    """A bare SolveConfig carrying only what ``spawn_realized_dispatch_sibling``
    reads/writes, with a single parent solve ``P`` populated."""
    sc = object.__new__(SolveConfig)
    sc.model_solve = {"m": ["P"]}
    sc.timesets_used_by_solves = {"P": ["ts1"]}
    sc.solve_period_years_represented = {"P": {"p2035": 1.0}}
    sc.periods_available = {"P": ["p2035"]}
    sc.delay_durations = {}
    sc.hole_multipliers = {}
    sc.rolling_times = {}
    sc.solve_modes = {"P": "single_solve"}
    sc.realized_periods = {"P": ["p2035"]}
    sc.realized_dispatch_parent = {}
    return sc


def test_spawn_populates_every_defining_fact_derived_dict():
    """After spawn, every defining fact's DERIVED SolveConfig dict carries the
    sibling — i.e. the SolveConfig half covers the same facts the reader half
    clones.  A fact present in the reader allow-list but not populated here = a
    half-configured sibling (the fracture)."""
    sc = _parent_solveconfig_stub()
    sib = "P__realized_dispatch"
    sc.spawn_realized_dispatch_sibling("P", sib)

    for raw, derived in _DEFINING_FACT_MAP.items():
        d = getattr(sc, derived)
        assert sib in d, (
            f"defining fact {raw!r} (SolveConfig dict {derived!r}) is in the "
            f"reader allow-list but spawn_realized_dispatch_sibling did not "
            f"populate it for the sibling — the two halves have drifted."
        )


def test_spawn_sibling_config_is_consistent_with_parent():
    """The sibling's defining facts agree with the parent (solve_mode SET to
    single_solve; the rest copied), and the authoritative parent map + the
    position-after-parent insert hold."""
    sc = _parent_solveconfig_stub()
    sib = "P__realized_dispatch"
    sc.spawn_realized_dispatch_sibling("P", sib)

    assert sc.realized_dispatch_parent[sib] == "P"          # authoritative map
    assert sc.model_solve["m"] == ["P", sib]                # inserted after parent
    assert sc.solve_modes[sib] == "single_solve"            # SET
    assert sc.realized_periods[sib] == sc.realized_periods["P"]
    assert sc.timesets_used_by_solves[sib] == sc.timesets_used_by_solves["P"]
    assert (
        sc.solve_period_years_represented[sib]
        == sc.solve_period_years_represented["P"]
    )
