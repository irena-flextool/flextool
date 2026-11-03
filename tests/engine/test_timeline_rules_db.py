"""Time-structure fallback rules 1 and 5 (``TimelineConfig.create_assumptive_parts``) (#175).

* **Rule 1** -- the model has no timeset and exactly one timeline: FlexTool creates one timeset,
  ``SYNTHETIC_TIMESET`` (``"full_timeline"``), which Rule 2 gives the whole timeline.
* **Rule 5b** -- a solve of the model has no ``period_timeset`` and exactly one timeset is available
  (Rule 1's included): that timeset is used for each of the solve's periods (invest periods first,
  then realized periods; Rule 5a's ``model.periods_available`` when the solve lists none).

A model without a timeset therefore runs on the whole timeline:

    LP(no timeset)               == LP(timeset full_timeline = whole timeline, authored period_timeset)
    LP(sole timeset, no mapping) == LP(the same timeset, authored period_timeset)

Fixtures are built from ``tests/fixtures/tests.json`` (``base``: solve ``y2020_2day_dispatch``, period
``p2020``, timeline ``y2020`` of 72 steps, timeset ``2day``).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.input._timeline import TimelineConfig
from flextool.engine.input._timeline_rules import SYNTHETIC_TIMESET
from tests.engine.test_cyclic_fix_start import _lp_snapshot
from tests.engine.test_silent_label_params import _build_and_run
from tests.engine.test_years_represented_default import SOLVE, _solve_config

TESTS_JSON = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "tests.json"
ACTIVE_SOLVE = "y2020_2day_dispatch"
_RTOL_LP = 1e-12


# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------

def _drop_init_period_timeset(data: dict) -> None:
    data["parameter_values"] = [
        v for v in data["parameter_values"]
        if not (v[0] == "solve" and v[2] == "period_timeset" and v[4] == "init")]


def _keep_timesets(data: dict, keep: set[str]) -> None:
    """Remove every timeset entity (and its values / entity alternatives) not in *keep*."""
    def kept(name) -> bool:
        return (name[0] if isinstance(name, list) else name) in keep
    data["entities"] = [e for e in data["entities"] if e[0] != "timeset" or kept(e[1])]
    data["parameter_values"] = [v for v in data["parameter_values"] if v[0] != "timeset" or kept(v[1])]
    data["entity_alternatives"] = [ea for ea in data["entity_alternatives"] if ea[0] != "timeset" or kept(ea[1])]


def _rule1_data(data: dict) -> dict:
    """``base`` with no timeset at all and no ``period_timeset`` for its solve: Rules 1, 2 and 5b."""
    _keep_timesets(data, set())
    _drop_init_period_timeset(data)
    return _lp_snapshot()._overlay(data, name="rule1", base="base", values=[])


def _rule1_twin_data(data: dict) -> dict:
    """The Rule-1 data plus the timeset Rules 1, 2 and 5b must create, authored explicitly: ``full_timeline``
    on timeline ``y2020``, one block of all 72 steps, and ``period_timeset {p2020: full_timeline}``."""
    lps = _lp_snapshot()
    data = _rule1_data(data)
    return lps._overlay(data, name="rule1_twin", base="rule1", entities=[("timeset", SYNTHETIC_TIMESET)], values=[
        ["timeset", SYNTHETIC_TIMESET, "timeline", "y2020"],
        ["timeset", SYNTHETIC_TIMESET, "timeset_duration", lps._pmap({"t0001": 72.0}, index_name="t")],
        ["solve", ACTIVE_SOLVE, "period_timeset", lps._pmap({"p2020": SYNTHETIC_TIMESET})]])


# ---------------------------------------------------------------------------
# Unit: create_assumptive_parts
# ---------------------------------------------------------------------------

def _timeline_config(timelines: dict) -> TimelineConfig:
    tl: defaultdict = defaultdict(list)
    tl.update(timelines)
    return TimelineConfig(timelines=tl, timesets=[], timesets__timeline=defaultdict(str),
                          timeset_durations=defaultdict(list), new_step_durations={})


def test_no_timeset_one_timeline_creates_the_whole_timeline_timeset() -> None:
    """One timeline of two steps, no timeset, no ``period_timeset``: Rule 1 creates ``full_timeline``, Rule 2
    gives it one block of both steps, Rule 5b maps each of the solve's periods to it.  Before the fix Rule 1
    created one timeset per character of ``"full_timeline"`` and Rule 5b stopped with "More than one timeset"."""
    tc = _timeline_config({"tl": [("t0001", 1.0), ("t0002", 1.0)]})
    cfg = _solve_config(None)
    periods = [p for p, _ in cfg.realized_periods[SOLVE]]
    cfg.timesets_used_by_solves.pop(SOLVE)
    tc.create_assumptive_parts(cfg)
    assert tc.timesets == [SYNTHETIC_TIMESET]
    assert dict(tc.timeset_durations) == {SYNTHETIC_TIMESET: [("t0001", 2)]}
    assert dict(tc.timesets__timeline) == {SYNTHETIC_TIMESET: "tl"}
    assert cfg.timesets_used_by_solves[SOLVE] == [(p, SYNTHETIC_TIMESET) for p in periods]


def test_no_timeset_several_timelines_says_no_timeset() -> None:
    """Two timelines and no timeset: Rule 1 cannot choose a timeline, so the solve has no timeset to use.  The
    error says so (before: "More than one timeset available")."""
    tc = _timeline_config({"tl_a": [("t0001", 1.0)], "tl_b": [("t0001", 1.0)]})
    cfg = _solve_config(None)
    cfg.timesets_used_by_solves.pop(SOLVE)
    with pytest.raises(FlexToolConfigError, match="no timeset is available") as err:
        tc.create_assumptive_parts(cfg)
    assert "More than one timeset" not in str(err.value)
    assert "(it has 2)" in str(err.value)


# ---------------------------------------------------------------------------
# End to end: a model without a timeset builds the LP of its explicit twin
# ---------------------------------------------------------------------------

def _assert_same_lp(case, twin) -> None:
    lps = _lp_snapshot()
    (case_lp, case_obj), (twin_lp, twin_obj) = case, twin
    assert [s for s, _, _ in case_lp] == [s for s, _, _ in twin_lp]
    for (solve, pa, off_a), (_, pb, off_b) in zip(case_lp, twin_lp):
        msgs = lps.canonical_diff(lps.parse_mps(pa), lps.parse_mps(pb), _RTOL_LP, off_a, off_b)
        assert not msgs, f"{solve}: LP differs from the twin: {msgs}"
    assert case_obj == pytest.approx(twin_obj, rel=1e-9)


def test_model_without_timeset_builds_the_explicit_whole_timeline_lp(tmp_path_factory) -> None:
    """``base`` with no timeset at all == the same model with ``full_timeline`` (72 steps) authored and mapped to
    ``p2020``.  Before the fix the case stopped with "More than one timeset available"."""
    case = _build_and_run(tmp_path_factory, _rule1_data, "rule1")
    twin = _build_and_run(tmp_path_factory, _rule1_twin_data, "rule1_twin")
    _assert_same_lp(case, twin)


# ---------------------------------------------------------------------------
# DB-side readers: one shared mapping, pinned to create_assumptive_parts
# ---------------------------------------------------------------------------

_ABLATED_FIELDS = (
    "dt", "p_step_duration", "p_period_share", "p_inflation_op", "p_timestep_weight",
    "pd_branch_weight", "pdt_branch_weight", "p_process_existing_count", "p_profile_value",
)


def _rule5b_data(data: dict) -> dict:
    """``base`` with ``2day`` as the only timeset and no ``period_timeset`` for its solve: Rule 5b."""
    _keep_timesets(data, {"2day"})
    _drop_init_period_timeset(data)
    return _lp_snapshot()._overlay(data, name="rule5b", base="base", values=[])


def _with_twin(case_patch, case: str, twin_values: list, entities: list = ()):
    """*case_patch* plus scenario ``<case>_twin``: the case with *twin_values* authored on top."""
    def patch(data: dict) -> dict:
        return _lp_snapshot()._overlay(case_patch(data), name=f"{case}_twin", base=case, values=twin_values,
                                       entities=list(entities))
    return patch


def _rule5b_with_twin(data: dict) -> dict:
    return _with_twin(_rule5b_data, "rule5b", [
        ["solve", ACTIVE_SOLVE, "period_timeset", _lp_snapshot()._pmap({"p2020": "2day"})]])(data)


def _rule5b_multi_period(data: dict) -> dict:
    """Rule 5b over several periods: invest periods first (``p2030``, ``p2020``), then realized (``p2020``,
    ``p2025``); the mapping keeps that order and drops the repeat."""
    lps = _lp_snapshot()
    data = _rule5b_data(data)
    return lps._overlay(data, name="rule5b_multi", base="rule5b", values=[
        ["solve", ACTIVE_SOLVE, "invest_periods", {"type": "array", "value_type": "str", "data": ["p2030", "p2020"]}],
        ["solve", ACTIVE_SOLVE, "realized_periods", {"type": "array", "value_type": "str",
                                                     "data": ["p2020", "p2025"]}]])


def _rule5a(data: dict) -> dict:
    """Rule 5a then 5b: the solve lists no periods, its model's ``periods_available`` are used."""
    data = _rule5b_data(data)
    data["parameter_values"] = [v for v in data["parameter_values"]
                                if not (v[0] == "solve" and v[1] == ACTIVE_SOLVE and v[2] == "realized_periods")]
    return _lp_snapshot()._overlay(data, name="rule5a", base="rule5b", values=[
        ["model", "flexTool", "periods_available", {"type": "array", "value_type": "str",
                                                    "data": ["p2020", "p2025"]}]])


def _contained_child(data: dict) -> dict:
    """The Rule-5b solve contains a child solve without ``period_timeset``: Rule 5b covers the model's solves
    only, so the child gets no mapping on either path."""
    lps = _lp_snapshot()
    data = _rule5b_data(data)
    return lps._overlay(data, name="contained_child", base="rule5b", entities=[("solve", "child_solve")], values=[
        ["solve", ACTIVE_SOLVE, "contains_solves", {"type": "array", "value_type": "str", "data": ["child_solve"]}],
        ["solve", "child_solve", "realized_periods", {"type": "array", "value_type": "str", "data": ["p2020"]}]])


def _authored(data: dict) -> dict:
    return data


PIN_CASES = [
    (_authored, "base"),
    (_rule1_data, "rule1"),
    (_rule5b_data, "rule5b"),
    (_rule5b_multi_period, "rule5b_multi"),
    (_rule5a, "rule5a"),
    (_contained_child, "contained_child"),
]


def _build(tmp_path_factory, patch, name: str, json_path: Path = TESTS_JSON) -> str:
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    work = tmp_path_factory.mktemp(name)
    (work / "f.json").write_text(json.dumps(patch(json.loads(json_path.read_text()))))
    url = json_to_db(work / "f.json", work / "f.sqlite")
    migrate_database(url)
    return url


def _reader(url: str, scenario: str):
    from flextool.engine.input._spinedb_reader import SpineDbReader
    return SpineDbReader(url, scenario=scenario)


@pytest.mark.parametrize("patch, scenario", PIN_CASES, ids=[c[1] for c in PIN_CASES])
def test_db_side_mapping_equals_create_assumptive_parts(tmp_path_factory, patch, scenario) -> None:
    """For every solve of the model and every contained solve, the DB-side mapping (through the InputSource
    and through the SpineDB backend) equals ``timesets_used_by_solves`` after ``create_assumptive_parts``, in
    order (repeats dropped); a solve the CSV path gives no mapping gets ``None``."""
    from flextool.engine.input._solve_config import SolveConfig
    from flextool.engine.input._timeline_rules import solve_period_timesets, time_structure_from_backend
    from flextool.engine.preprocess.params._per_solve_sets import solve_period_timeset_frame
    from flextool.spinedb_backend import SpineDBBackend

    url = _build(tmp_path_factory, patch, scenario)
    sc = SolveConfig.load_from_db_url(url, scenario)
    tc = TimelineConfig.load_from_db_url(url, scenario)
    tc.create_assumptive_parts(sc)
    solves = list(dict.fromkeys(
        [s for v in sc.model_solve.values() for s in v] + [s for v in sc.contains_solves.values() for s in v]))
    assert ACTIVE_SOLVE in solves
    src = _reader(url, scenario)
    backend = SpineDBBackend(url, scenario)
    try:
        structure = time_structure_from_backend(backend)
    finally:
        backend.close()
    for solve in solves:
        csv = list(dict.fromkeys(tuple(x) for x in sc.timesets_used_by_solves.get(solve, []))) or None
        frame = solve_period_timeset_frame(src, solve)
        db = None if frame is None else [(str(d), str(ts)) for d, ts in frame.iter_rows()]
        assert db == csv, (solve, db, csv)
        assert solve_period_timesets(solve, structure) == csv, solve
    if scenario == "contained_child":
        assert sc.timesets_used_by_solves.get("child_solve") is None


def test_synthetic_sub_solve_names_get_no_mapping(tmp_path_factory) -> None:
    """A rolling / nested sub-solve name is not a solve entity: the helper returns ``None``, so its readers keep
    their workdir fallback."""
    from flextool.engine.preprocess.params._per_solve_sets import solve_period_timeset_frame

    url = _build(tmp_path_factory, _rule5b_data, "rule5b_synth")
    assert solve_period_timeset_frame(_reader(url, "rule5b"), f"{ACTIVE_SOLVE}_roll_0") is None


@pytest.mark.parametrize("patch, case", [(_rule1_twin_data, "rule1"), (_rule5b_with_twin, "rule5b")])
def test_db_producers_match_the_authored_twin(tmp_path_factory, patch, case) -> None:
    """``dt_and_step_duration_from_source``, ``_dt_period_active_steps`` (period order included),
    ``derive_per_solve_aggregates`` and ``p_timestep_weight_from_source`` give the twin's result.  Before the
    fix each returned ``None`` (or the 1.0 default) for the case."""
    from flextool.engine.preprocess.params._derived import (
        _dt_period_active_steps,
        dt_and_step_duration_from_source,
        p_timestep_weight_from_source,
        solve_has_rp_timeset,
    )
    from flextool.engine.preprocess.params._per_solve_sets import derive_per_solve_aggregates
    from tests._support.fields import field_diff

    url = _build(tmp_path_factory, patch, case)
    case_src, twin_src = _reader(url, case), _reader(url, f"{case}_twin")

    case_r = dt_and_step_duration_from_source(case_src, ACTIVE_SOLVE)
    twin_r = dt_and_step_duration_from_source(twin_src, ACTIVE_SOLVE)
    assert case_r is not None and twin_r is not None
    assert case_r[0].height == (72 if case == "rule1" else 48)
    assert case_r[0].equals(twin_r[0])
    assert field_diff(case_r[1], twin_r[1]) is None

    case_steps = _dt_period_active_steps(case_src, ACTIVE_SOLVE)
    twin_steps = _dt_period_active_steps(twin_src, ACTIVE_SOLVE)
    assert case_steps is not None and case_steps == twin_steps

    case_agg = derive_per_solve_aggregates(case_src, ACTIVE_SOLVE)
    twin_agg = derive_per_solve_aggregates(twin_src, ACTIVE_SOLVE)
    assert case_agg is not None and twin_agg is not None
    for field in ("period_in_use", "dt_complete", "period_timeline",
                  "p_timeline_duration_in_years", "complete_period_share_of_year"):
        assert field_diff(getattr(case_agg, field), getattr(twin_agg, field)) is None, field

    assert field_diff(p_timestep_weight_from_source(case_src, case_r[0], ACTIVE_SOLVE),
                      p_timestep_weight_from_source(twin_src, twin_r[0], ACTIVE_SOLVE)) is None
    assert solve_has_rp_timeset(case_src, ACTIVE_SOLVE) is solve_has_rp_timeset(twin_src, ACTIVE_SOLVE)


def _run_ablated(url: str, scenario: str, work_folder):
    """Run *scenario* with the nine ``apply_derived_a`` fields nulled right before every ``_apply_db_overrides``
    call, so they can only come from the DB-side readers (no CSV seed).  Returns (last FlexData, objectives)."""
    from flextool.engine.preprocess import flex_data as _fd
    from flextool.engine.solve._orchestration import run_chain_from_db

    original = _fd._apply_db_overrides
    captured: dict[str, object] = {}

    def wrapper(flex_data, db_reader, source, ctx=None, provider=None):
        for field in _ABLATED_FIELDS:
            setattr(flex_data, field, None)
        result = original(flex_data, db_reader, source, ctx=ctx, provider=provider)
        captured["flex_data"] = flex_data
        return result

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_fd, "_apply_db_overrides", wrapper)
        steps = run_chain_from_db(url, scenario, work_folder=work_folder, warm=False, keep_solutions=True)
    assert steps and all(s.solution is not None and s.solution.optimal for s in steps.values())
    return captured["flex_data"], [float(s.solution.obj) for s in steps.values()]


def _weighted(data: dict) -> dict:
    """Rule 5b on ``2day`` with its ``timeset_weights`` (alternative ``weighted_2day``), and the twin with
    ``period_timeset {p2020: 2day}`` authored: the sole timeset's weights must apply without the mapping."""
    lps = _lp_snapshot()
    data = _rule5b_data(data)
    data = lps._overlay(data, name="rule5b_weighted", base=["init", "west", "weighted_2day"], values=[])
    data["parameter_values"] = [v for v in data["parameter_values"]
                                if not (v[0] == "solve" and v[2] == "period_timeset" and v[4] == "weighted_2day")]
    return lps._overlay(data, name="rule5b_weighted_twin", base="rule5b_weighted", values=[
        ["solve", ACTIVE_SOLVE, "period_timeset", lps._pmap({"p2020": "2day"})]])


@pytest.mark.parametrize("patch, case, twin", [
    (_rule5b_with_twin, "rule5b", "rule5b_twin"),
    (_rule1_twin_data, "rule1", "rule1_twin"),
    (_weighted, "rule5b_weighted", "rule5b_weighted_twin"),
], ids=["rule5b", "rule1", "rule5b_timeset_weights"])
def test_flexdata_fields_match_the_twin_without_any_seed(tmp_path_factory, patch, case, twin) -> None:
    """With the nine fields nulled before ``_apply_db_overrides``, the case fills each of them like its twin
    with an authored ``period_timeset`` (the sole timeset's ``timeset_weights`` included), and solves to the
    same objective.  Before the fix the case's fields stayed ``None``."""
    from tests._support.fields import field_diff

    url = _build(tmp_path_factory, patch, case)
    case_fd, case_obj = _run_ablated(url, case, tmp_path_factory.mktemp(f"{case}_w"))
    twin_fd, twin_obj = _run_ablated(url, twin, tmp_path_factory.mktemp(f"{twin}_w"))
    for field in _ABLATED_FIELDS:
        assert getattr(case_fd, field) is not None or getattr(twin_fd, field) is None, field
        assert field_diff(getattr(case_fd, field), getattr(twin_fd, field)) is None, field
    assert case_obj == pytest.approx(twin_obj, rel=1e-9)
    if case == "rule5b_weighted":
        weights = set(case_fd.p_timestep_weight.frame["value"].round(9).to_list())
        assert weights != {1.0}, "the timeset_weights of 2day did not reach p_timestep_weight"


def _roll_case(data: dict) -> dict:
    """``fullYear_roll`` (4 periods, rolling) with ``fullYear`` as the only timeset and no ``period_timeset``."""
    _keep_timesets(data, {"fullYear"})
    data["parameter_values"] = [
        v for v in data["parameter_values"]
        if not (v[0] == "solve" and v[1] == "dispatch_fullYear_roll" and v[2] == "period_timeset")]
    return _lp_snapshot()._overlay(data, name="roll_rule5b", base="fullYear_roll", values=[])


@pytest.mark.slow
def test_rolling_solve_without_period_timeset_builds_the_same_lp(tmp_path_factory) -> None:
    """``fullYear_roll`` without ``period_timeset`` (sole timeset ``fullYear``) builds, roll by roll, the LP of
    plain ``fullYear_roll``: the rolls (synthetic names) keep their workdir path, the parent solve gets the
    Rule 5b mapping."""
    case = _build_and_run(tmp_path_factory, _roll_case, "roll_rule5b")
    twin = _build_and_run(tmp_path_factory, _authored, "fullYear_roll")
    assert len(case[0]) == len(twin[0]) > 1
    _assert_same_lp(case, twin)


# ---------------------------------------------------------------------------
# Readers outside the per-solve chain: axis vocabulary and commodity-ladder periods
# ---------------------------------------------------------------------------

STOCH_TWO_PERIOD_JSON = TESTS_JSON.parent / "stoch_two_period.json"


def _stoch_case(data: dict) -> dict:
    """``stoch`` (``stoch_2p``: periods ``p2035`` / ``p2040``, stochastic branches) with ``ts1`` as the only
    timeset and no ``period_timeset``, plus the twin authoring ``{p2035: ts1, p2040: ts1}``."""
    lps = _lp_snapshot()
    _keep_timesets(data, {"ts1"})
    data["parameter_values"] = [v for v in data["parameter_values"]
                                if not (v[0] == "solve" and v[2] == "period_timeset")]
    data = lps._overlay(data, name="stoch_rule5b", base="stoch", values=[])
    return lps._overlay(data, name="stoch_rule5b_twin", base="stoch_rule5b", values=[
        ["solve", "stoch_2p", "period_timeset", lps._pmap({"p2035": "ts1", "p2040": "ts1"})]])


def test_stochastic_period_vocabulary_without_period_timeset(tmp_path_factory) -> None:
    """The continuation tokens ``<later period>_<branch>`` of a multi-period stochastic solve come from its
    periods; without ``period_timeset`` they come from the Rule 5b mapping, the same as the twin's."""
    from flextool.spinedb_backend import SpineDBBackend
    from flextool.spinedb_backend._axis_enums import _collect_stochastic_branch_period_tokens

    url = _build(tmp_path_factory, _stoch_case, "stoch_vocab", json_path=STOCH_TWO_PERIOD_JSON)
    tokens = {}
    for scenario in ("stoch_rule5b", "stoch_rule5b_twin"):
        backend = SpineDBBackend(url, scenario)
        try:
            tokens[scenario] = _collect_stochastic_branch_period_tokens(backend)
        finally:
            backend.close()
    assert any(t.startswith("p2040_") for t in tokens["stoch_rule5b_twin"])
    assert tokens["stoch_rule5b"] == tokens["stoch_rule5b_twin"]


def test_stochastic_two_period_solve_without_period_timeset_builds_the_twin_lp(tmp_path_factory) -> None:
    """End to end: the multi-period stochastic solve without ``period_timeset`` builds the twin's LP."""
    case = _build_and_run(tmp_path_factory, _stoch_case, "stoch_rule5b", json_path=STOCH_TWO_PERIOD_JSON)
    twin = _build_and_run(tmp_path_factory, _stoch_case, "stoch_rule5b_twin", json_path=STOCH_TWO_PERIOD_JSON)
    _assert_same_lp(case, twin)


def test_commodity_ladder_periods_without_period_timeset(tmp_path_factory) -> None:
    """Without ``model.periods_available`` and ``period_timeset``, the periods a 2-D price ladder expands over
    are the Rule 5b periods of the solve, the same as the twin's authored mapping gives."""
    from flextool.input_derivation._commodity_ladder import _collect_periods
    from flextool.spinedb_backend import SpineDBBackend

    url = _build(tmp_path_factory, _rule5b_with_twin, "ladder_periods")
    periods = {}
    for scenario in ("rule5b", "rule5b_twin"):
        backend = SpineDBBackend(url, scenario)
        try:
            periods[scenario], from_solves = _collect_periods(backend, None, backend.api)
            assert from_solves is (scenario == "rule5b")
        finally:
            backend.close()
    assert periods["rule5b_twin"] == ["p2020"]
    assert periods["rule5b"] == periods["rule5b_twin"]


def _ladder_case(data: dict) -> dict:
    """``ladder_ann_depth2`` (coal's ``price_ladder_annual`` as tier -> {price, quantity}, no periods) with
    ``2day`` as the only timeset and no ``period_timeset``, plus the twin authoring ``{p2020: 2day}``."""
    lps = _lp_snapshot()
    data = lps.PATCHES["ladder_ann_depth2"](data)
    _keep_timesets(data, {"2day"})
    _drop_init_period_timeset(data)
    return lps._overlay(data, name="ladder_rule5b_twin", base="ladder_ann_depth2", values=[
        ["solve", ACTIVE_SOLVE, "period_timeset", lps._pmap({"p2020": "2day"})]])


def test_price_ladder_without_periods_applies_without_period_timeset(tmp_path_factory, capsys) -> None:
    """A period-less ``price_ladder_annual`` expands over the Rule 5b periods: the case builds the twin's LP
    (before the fix the ladder was skipped, with a warning, and the LP lacked it).  An upgraded database prints
    the behaviour-change notice naming the commodity."""
    from spinedb_api import DatabaseMapping

    case = _build_and_run(tmp_path_factory, _ladder_case, "ladder_ann_depth2")
    twin = _build_and_run(tmp_path_factory, _ladder_case, "ladder_rule5b_twin")
    _assert_same_lp(case, twin)

    url = _build(tmp_path_factory, _ladder_case, "ladder_notice")
    with DatabaseMapping(url) as db:
        db.add_item("metadata", name="flextool_behaviour_change:price_ladder_periods_from_solve_periods", value="77")
        db.commit_session("behaviour-change marker")
    from flextool.engine import run_chain_from_db

    capsys.readouterr()
    run_chain_from_db(url, "ladder_ann_depth2", work_folder=tmp_path_factory.mktemp("ladder_notice_w"))
    out = capsys.readouterr().out
    assert "price_ladder_annual given without periods" in out and "Affected commodities: coal." in out, out
