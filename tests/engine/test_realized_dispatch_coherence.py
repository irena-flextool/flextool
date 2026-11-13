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
    sc.timesets_used_by_solves = {"P": [("p2035", "ts1")]}
    sc.solve_period_years_represented = {"P": [("p2035", 1.0)]}
    sc.periods_available = {"P": ["p2035"]}
    sc.delay_durations = {}
    sc.hole_multipliers = {}
    sc.rolling_times = {}
    sc.solve_modes = {"P": "single_solve"}
    sc.realized_periods = {"P": [("p2035", "p2035")]}
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


class _CapturingReader:
    """The reader half of the spawn: records each ``register_derived_solve`` call with its ``replace`` values."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def entities(self, _cls):
        import polars as pl
        return pl.DataFrame({"name": ["P"]})

    def register_derived_solve(self, parent, sibling, *, drop=(), replace=None):
        self.calls.append((parent, sibling, dict(replace or {})))


def _lookahead_parent(mode: str = "single_solve", realized=(("p2035", "p2035"),)) -> SolveConfig:
    """A ``separate_dispatch`` parent ``P`` over p2035 (realized) and a look-ahead p2040 (not realized)."""
    sc = object.__new__(SolveConfig)
    sc.model_solve = {"m": ["P"]}
    sc.contains_solves = {}
    sc.timesets_used_by_solves = {"P": [("p2035", "ts35"), ("p2040", "ts40")]}
    sc.solve_period_years_represented = {"P": [("p2035", 5.0), ("p2040", 10.0)]}
    sc.solve_modes = {"P": mode}
    sc.realized_periods = {"P": list(realized)}
    sc.dispatch_realization_method = {"P": "separate_dispatch"}
    sc.fix_storage_periods = {}
    sc.realized_dispatch_parent = {}
    return sc


def test_single_solve_realization_has_the_realized_periods_only_in_both_halves():
    """Item 255: the realization solve of a single solve dispatches its realized periods only, not the
    look-ahead.  Both halves agree: the SolveConfig dicts and the reader's ``period_timeset`` /
    ``years_represented`` rows (``replace=``) hold p2035 and not p2040."""
    from flextool.engine.solve._orchestration import _spawn_realized_dispatch_siblings

    sc = _lookahead_parent()
    reader = _CapturingReader()
    _spawn_realized_dispatch_siblings(sc, reader, logger=None)
    sib = "P__realized_dispatch"
    assert sc.realized_dispatch_parent == {sib: "P"}
    assert sc.timesets_used_by_solves[sib] == [("p2035", "ts35")]
    assert sc.solve_period_years_represented[sib] == [("p2035", 5.0)]
    assert sc.timesets_used_by_solves["P"] == [("p2035", "ts35"), ("p2040", "ts40")]
    [(parent, sibling, replace)] = reader.calls
    assert (parent, sibling) == ("P", sib)
    assert set(replace) == {"period_timeset", "years_represented"}
    assert list(replace["period_timeset"].indexes) == ["p2035"]
    assert list(replace["period_timeset"].values) == ["ts35"]
    assert replace["period_timeset"].index_name == "period"
    assert list(replace["years_represented"].indexes) == ["p2035"]
    assert list(replace["years_represented"].values) == [5.0]


def test_rolling_realization_keeps_the_parent_periods():
    """A rolling parent's realization rolls are trimmed to each roll's jump at run time (``_run_model`` OC1), so
    both halves copy the parent's periods unchanged."""
    from flextool.engine.solve._orchestration import _spawn_realized_dispatch_siblings

    sc = _lookahead_parent("rolling_window")
    sc.roll_counter = {"P": 0}
    reader = _CapturingReader()
    _spawn_realized_dispatch_siblings(sc, reader, logger=None)
    sib = "P__realized_dispatch"
    assert sc.timesets_used_by_solves[sib] == sc.timesets_used_by_solves["P"]
    assert sc.solve_period_years_represented[sib] == sc.solve_period_years_represented["P"]
    assert reader.calls == [("P", sib, {})]


def test_separate_dispatch_without_realized_periods_adds_no_realization_solve():
    """A ``separate_dispatch`` solve that realizes no period (an invest-only solve) has nothing to dispatch: no
    realization solve is added (owner decision, item 255 critique M2)."""
    from flextool.engine.solve._orchestration import _spawn_realized_dispatch_siblings

    sc = _lookahead_parent(realized=())
    reader = _CapturingReader()
    _spawn_realized_dispatch_siblings(sc, reader, logger=None)
    assert sc.realized_dispatch_parent == {}
    assert reader.calls == []
    assert sc.model_solve == {"m": ["P"]}


def test_realized_periods_with_a_gap_are_an_error():
    """A period the parent does not realize between two it realizes: the realization solve (realized periods
    only) would run its storage from one straight into the next -- a configuration error naming the gap."""
    import pytest

    from flextool.engine.input._solve_state import FlexToolConfigError

    sc = _lookahead_parent()
    sc.timesets_used_by_solves["P"] = [("p2035", "ts35"), ("p2040", "ts40"), ("p2045", "ts45")]
    sc.realized_periods["P"] = [("p2035", "p2035"), ("p2045", "p2045")]
    with pytest.raises(FlexToolConfigError, match=r"Solve 'P' realizes periods 'p2035', 'p2045' but not 'p2040'"):
        sc.realization_window("P")
