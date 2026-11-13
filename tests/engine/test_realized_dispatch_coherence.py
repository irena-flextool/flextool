"""Coherence / fracture-guard for the realized-dispatch (``separate_dispatch``)
feature.

The sibling's configuration is DEFINED TWICE, in two vocabularies (see the
feature map at ``_orchestration._spawn_realized_dispatch_siblings``): both halves
copy everything from the parent except a drop list,

* the RAW Spine-param half — ``SpineDbReader.REALIZATION_SOLVE_DROPPED_PARAMS``
  (the parameter rows ``register_derived_solve`` does not clone), and
* the DERIVED-dict half — ``_solve_config.REALIZATION_SOLVE_DROPS`` (the
  SolveConfig fields ``derive_realization_solve`` does not copy).

Nothing structurally forces the two lists to stay aligned (they live in
different layers / vocabularies), so this test is the fracture guard: dropping a
parameter in one half without the other fails here loudly.  The four defining
facts the sibling must carry (period_timeset ↔ timesets_used_by_solves,
years_represented ↔ solve_period_years_represented, solve_mode ↔ solve_modes,
realized_periods ↔ realized_periods) and ``new_stepduration`` ↔
``TimelineConfig.new_step_durations`` (set by ``_spawn_realized_dispatch_siblings``)
are in neither drop list.
"""
from __future__ import annotations

from flextool.engine.input._solve_config import NOT_COPIED_FIELDS, REALIZATION_SOLVE_DROPS, SolveConfig
from flextool.engine.input._spinedb_reader import SpineDbReader

#: The four defining realized-dispatch facts, RAW Spine param -> DERIVED
#: SolveConfig dict.
_DEFINING_FACT_MAP = {
    "period_timeset": "timesets_used_by_solves",
    "years_represented": "solve_period_years_represented",
    "solve_mode": "solve_modes",
    "realized_periods": "realized_periods",
}

#: Defining facts whose DERIVED half lives on TimelineConfig (set by
#: ``_spawn_realized_dispatch_siblings`` from the parent's value).
_TIMELINE_FACT_MAP = {
    "new_stepduration": "new_step_durations",
}

#: The dropped parameters, RAW Spine param -> DERIVED SolveConfig field.
_DROP_MAP = {
    "invest_periods": "invest_periods",
    "realized_invest_periods": "realized_invest_periods",
    "fix_storage_periods": "fix_storage_periods",
    "stochastic_branches": "stochastic_branches",
    "stochastic_invest_method": "stochastic_invest_method",
    "shared_operation_periods": "non_anticipativity_periods",
    "shared_invest_periods": "non_anticipativity_invest_periods",
    "decomposition": "decomposition",
    "benders_max_iter": "benders_max_iter",
    "benders_tolerance": "benders_tolerance",
    "benders_in_out_weight": "benders_in_out_weight",
    "contains_solves": "contains_solves",
    "dispatch_realization_method": "dispatch_realization_method",
}


def test_reader_and_solve_config_drop_the_same_parameters():
    """The reader's drop list is the SolveConfig drop list in schema names, plus ``solver_precommand``, which
    SolveConfig never copies to any solve.  If this drifts, update both halves in lockstep."""
    reader = set(SpineDbReader.REALIZATION_SOLVE_DROPPED_PARAMS)
    assert {_DROP_MAP[p] for p in reader if p in _DROP_MAP} == set(REALIZATION_SOLVE_DROPS)
    assert reader - set(_DROP_MAP) == {"solver_precommand"}
    assert "solver_settings.precommand" in NOT_COPIED_FIELDS


def test_defining_facts_are_not_dropped():
    """The sibling carries its parent's defining facts in both halves."""
    assert not (set(_DEFINING_FACT_MAP) | set(_TIMELINE_FACT_MAP)) & set(SpineDbReader.REALIZATION_SOLVE_DROPPED_PARAMS)
    assert not set(_DEFINING_FACT_MAP.values()) & set(REALIZATION_SOLVE_DROPS)


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
    clones (the fracture: a half-configured sibling)."""
    sc = _parent_solveconfig_stub()
    sib = "P__realized_dispatch"
    sc.spawn_realized_dispatch_sibling("P", sib)

    for raw, derived in _DEFINING_FACT_MAP.items():
        d = getattr(sc, derived)
        assert sib in d, (
            f"defining fact {raw!r} (SolveConfig dict {derived!r}): "
            f"spawn_realized_dispatch_sibling did not populate it for the "
            f"sibling — the two halves have drifted."
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
