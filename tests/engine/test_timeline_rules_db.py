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

from collections import defaultdict

import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.input._timeline import TimelineConfig
from flextool.engine.input._timeline_rules import SYNTHETIC_TIMESET
from tests.engine.test_cyclic_fix_start import _lp_snapshot
from tests.engine.test_silent_label_params import _build_and_run
from tests.engine.test_years_represented_default import SOLVE, _solve_config

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
