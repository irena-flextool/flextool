"""Single-timeline defaults in the DB-side timeline readers (#171).

The CSV / ``TimelineConfig`` path (``flextool/engine/input/_timeline.py``,
``create_assumptive_parts``) has always applied two fallback rules when a scenario's time
structure is incomplete:

* **Rule 2** -- no ``timeset.timeset_duration`` value anywhere in the scenario, and exactly one
  timeline -- every timeset gets one block covering the whole timeline.
* **Rule 3** -- a timeset with no ``timeset.timeline`` row, and exactly one timeline -- that
  timeset uses the sole timeline.

The three DB-side timeline readers (``_derived.dt_and_step_duration_from_source``,
``_derived._dt_period_active_steps``, ``_per_solve_sets.derive_per_solve_aggregates``) had no
such defaults: each re-reads ``timeset.timeset_duration`` / ``timeset.timeline`` directly and
returns ``None`` the moment either is absent, so ``apply_derived_a`` bails before
``p_period_share``, ``p_inflation_op``, ``p_timestep_weight``, ``pd_branch_weight``,
``pdt_branch_weight``, ``p_process_existing_count`` and ``p_profile_value`` (and ``dt`` /
``p_step_duration`` themselves).  Every existing fixture is masked by the CSV seed those fields
still carry from the legacy path, so the gap was invisible until the CSV seed itself was
(correctly) made conditional on the DB override succeeding -- see ``3c7726a9``, which restored
the ``p_profile_value`` seed specifically because ``model_builder.build_model`` (used by
``tests/test_representative_periods.py``) never writes ``timeset_duration`` at all.

The fix is one shared helper pair in ``preprocess/params/_per_solve_sets.py``
(``timeset_duration_lf`` / ``timeset_timeline_lf``), used by all three readers, plus the matching
one-line fix to ``_timeline.py``'s own Rule 2 guard (Q-c25-A's default: the guard is evaluated
ONCE before the loop over timesets, not re-checked inside it -- otherwise only the first timeset
of a multi-timeset model gets the whole-timeline block).

Fixture notes (``tests/fixtures/tests.json``):

* ``base`` / solve ``y2020_2day_dispatch`` uses timeset ``2day``, whose *authored*
  ``timeset_duration`` (alternative ``init``) is a **partial** 48-step block (``{t0001: 48}``) --
  not the whole 72-step ``y2020`` timeline.  So ``base`` itself is NOT a valid twin for the
  Rule 2 default (which gives the WHOLE timeline, 72 steps): the twin used below is ``base`` with
  ``2day.timeset_duration`` replaced by an *explicit* full-timeline block (``{t0001: 72}``) in a
  fresh override alternative -- the authored equivalent of what Rule 2 must default to once the
  partial block is removed entirely.  This matches the design's own probe (``design.md`` S3.1),
  whose twin is the same explicit 72-step map, not ``base`` unmodified.
* ``fullYear_roll`` / solve ``dispatch_fullYear_roll`` (rolling) uses timeset ``fullYear``, whose
  authored ``timeset_duration`` (alternative ``fullYear``) IS already the whole 72-step block --
  so removing it (together with ``2day`` / ``5weeks``'s own rows in ``init``, which are also
  visible in this scenario's alternative chain and would otherwise keep ``timeset_durations``
  non-empty and suppress the scenario-wide Rule 2 guard) makes plain ``fullYear_roll`` itself the
  correct twin.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._support.fields import field_diff
from tests.engine.test_cyclic_fix_start import _lp_snapshot
from tests.engine.test_silent_label_params import _build_and_run

REPO = Path(__file__).resolve().parents[2]
TESTS_JSON = REPO / "tests" / "fixtures" / "tests.json"
ACTIVE_SOLVE = "y2020_2day_dispatch"
_RTOL_LP = 1e-12

# The 9 FlexData fields ``apply_derived_a`` writes from the DB-side timeline readers: ``dt`` /
# ``p_step_duration`` (from ``dt_and_step_duration_from_source``) plus the 7 fields that depend on
# them and are only reached once that call succeeds (design.md S3.1).
_ABLATED_FIELDS = (
    "dt", "p_step_duration", "p_period_share", "p_inflation_op", "p_timestep_weight",
    "pd_branch_weight", "pdt_branch_weight", "p_process_existing_count", "p_profile_value",
)


def _timeline_default_data() -> dict:
    """``base`` with every ``timeset_duration`` of ``init`` removed (the c25-1 ``timeline_default``
    fixture patch, reused verbatim), plus a fresh ``timeline_default_twin`` scenario on the SAME
    (already-patched) data: ``2day.timeset_duration`` authored explicitly as the full 72-step
    block.  Both scenarios live in one DB so a single build serves every test below."""
    lps = _lp_snapshot()
    data = lps.PATCHES["timeline_default"](json.loads(TESTS_JSON.read_text()))
    return lps._overlay(data, name="timeline_default_twin", base="base", values=[
        ["timeset", "2day", "timeset_duration", lps._pmap({"t0001": 72.0}, index_name="t")]])


def _build_db(tmp_path: Path, data: dict) -> str:
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    json_path = tmp_path / "f.json"
    json_path.write_text(json.dumps(data))
    url = json_to_db(json_path, tmp_path / "f.sqlite")
    migrate_database(url)
    return url


def _reader(url: str, scenario: str):
    from flextool.engine.input._spinedb_reader import SpineDbReader
    return SpineDbReader(url, scenario=scenario)


def _run_ablated(url: str, scenario: str, work_folder: Path):
    """Run ``scenario`` to completion with the 9 fields nulled right before every
    ``_apply_db_overrides`` call, so the only way they end up populated is through the DB-side
    timeline readers this commit fixes -- no leftover CSV seed can paper over a gap.  Returns the
    last sub-solve's ``FlexData``."""
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
        steps = run_chain_from_db(url, scenario, work_folder=work_folder, warm=False,
                                   keep_solutions=True)
    assert steps and all(s.solution is not None and s.solution.optimal for s in steps.values())
    assert "flex_data" in captured, f"{scenario}: _apply_db_overrides never ran"
    return captured["flex_data"]


# ---------------------------------------------------------------------------
# (a) Producer level, relational, fail-before.
# ---------------------------------------------------------------------------

def test_producers_match_the_explicit_full_timeline_twin(tmp_path):
    """``dt_and_step_duration_from_source``, ``_dt_period_active_steps`` and
    ``derive_per_solve_aggregates`` give the same result whether the whole-timeline block comes
    from Rule 2's default or from an equivalent explicit Map.  Before the fix every producer
    returns ``None`` for ``timeline_default`` (no ``timeset_duration`` anywhere, Rule 2 absent)."""
    from flextool.engine.preprocess.params._derived import (
        _dt_period_active_steps,
        dt_and_step_duration_from_source,
    )
    from flextool.engine.preprocess.params._per_solve_sets import derive_per_solve_aggregates

    url = _build_db(tmp_path, _timeline_default_data())
    case_src = _reader(url, "timeline_default")
    twin_src = _reader(url, "timeline_default_twin")

    case_r = dt_and_step_duration_from_source(case_src, ACTIVE_SOLVE)
    twin_r = dt_and_step_duration_from_source(twin_src, ACTIVE_SOLVE)
    assert case_r is not None, "Rule 2 default must give dt/p_step_duration, not None"
    assert twin_r is not None
    case_dt, case_step_dur = case_r
    twin_dt, twin_step_dur = twin_r
    assert field_diff(case_dt, twin_dt) is None
    assert field_diff(case_step_dur, twin_step_dur) is None

    case_steps = _dt_period_active_steps(case_src, ACTIVE_SOLVE)
    twin_steps = _dt_period_active_steps(twin_src, ACTIVE_SOLVE)
    assert case_steps is not None
    assert twin_steps is not None
    assert case_steps == twin_steps

    case_agg = derive_per_solve_aggregates(case_src, ACTIVE_SOLVE)
    twin_agg = derive_per_solve_aggregates(twin_src, ACTIVE_SOLVE)
    assert case_agg is not None
    assert twin_agg is not None
    for field in ("period_in_use", "dt_complete", "period_timeline",
                  "p_timeline_duration_in_years", "complete_period_share_of_year"):
        assert field_diff(getattr(case_agg, field), getattr(twin_agg, field)) is None, field


# ---------------------------------------------------------------------------
# (b) Rule 3.
# ---------------------------------------------------------------------------

def _patch_rule3_default(data: dict) -> dict:
    """``base`` with ``2day.timeline`` removed (``init``); ``2day.timeset_duration`` is kept, so
    only Rule 3 (the sole-timeline default for the timeset -> timeline mapping) is exercised."""
    data = dict(data)
    data["parameter_values"] = [
        v for v in data["parameter_values"]
        if not (v[0] == "timeset" and v[1] == "2day" and v[2] == "timeline" and v[4] == "init")]
    return _lp_snapshot()._overlay(data, name="rule3_default", base="base", values=[])


def test_rule3_sole_timeline_default_builds_the_same_lp(tmp_path_factory):
    """``base`` with ``2day.timeline`` missing must build the identical LP to ``base`` itself --
    before the fix this raises (``dt`` is ``None``, no override possible)."""
    lps = _lp_snapshot()
    case_lp, case_obj = _build_and_run(tmp_path_factory, _patch_rule3_default, "rule3_default")
    twin_lp, twin_obj = _build_and_run(tmp_path_factory, lambda d: d, "base")

    assert [s for s, _, _ in case_lp] == [s for s, _, _ in twin_lp]
    diffs = {}
    for (solve, pa, off_a), (_, pb, off_b) in zip(case_lp, twin_lp):
        msgs = lps.canonical_diff(lps.parse_mps(pa), lps.parse_mps(pb), _RTOL_LP, off_a, off_b)
        if msgs:
            diffs[solve] = msgs
    assert not diffs, f"Rule 3 default LP differs from base: {diffs}"
    assert case_obj == pytest.approx(twin_obj, rel=1e-9)


# ---------------------------------------------------------------------------
# (e) LP twin: Rule 2 default vs. the explicit equivalent map.
# ---------------------------------------------------------------------------

def test_rule2_default_builds_the_same_lp_as_the_explicit_twin(tmp_path):
    """``timeline_default`` (no ``timeset_duration`` anywhere) must build the identical LP to the
    explicit full-timeline twin.  Passes before the fix too (the CSV seed already applies the
    legacy Rule 2 default for this single-relevant-timeset scenario) -- it guards the DB-side
    producer once the CSV seed is no longer the only thing giving the right answer."""
    lps = _lp_snapshot()
    data = _timeline_default_data()
    json_path = tmp_path / "f.json"
    json_path.write_text(json.dumps(data))

    import warnings

    from flextool.engine import run_chain_from_db
    from flextool.engine.model import core
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    url = json_to_db(json_path, tmp_path / "f.sqlite")
    migrate_database(url)

    def _run(scenario: str, tag: str):
        built: list[tuple[str, Path, float]] = []
        original = core.build_flextool

        def capturing(m, d, *args, **kwargs):
            result = original(m, d, *args, **kwargs)
            solve = lps._solve_name_from_stack()
            path = tmp_path / f"{tag}_{len(built):03d}_{lps._safe(solve)}.mps"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                m.write_mps(path, emit_names=True)
            built.append((solve, path, float(m._obj_offset)))
            return result

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(core, "build_flextool", capturing)
            steps = run_chain_from_db(url, scenario, work_folder=tmp_path / f"w_{tag}", warm=False)
        assert built
        assert steps and all(s.solution is not None and s.solution.optimal for s in steps.values())
        return built, [float(s.solution.obj) for s in steps.values()]

    case_lp, case_obj = _run("timeline_default", "case")
    twin_lp, twin_obj = _run("timeline_default_twin", "twin")

    assert [s for s, _, _ in case_lp] == [s for s, _, _ in twin_lp]
    diffs = {}
    for (solve, pa, off_a), (_, pb, off_b) in zip(case_lp, twin_lp):
        msgs = lps.canonical_diff(lps.parse_mps(pa), lps.parse_mps(pb), _RTOL_LP, off_a, off_b)
        if msgs:
            diffs[solve] = msgs
    assert not diffs, f"Rule 2 default LP differs from the explicit twin: {diffs}"
    assert case_obj == pytest.approx(twin_obj, rel=1e-9)


# ---------------------------------------------------------------------------
# (c) FlexData level, fail-before: the 9 fields with no CSV seed to fall back on.
# ---------------------------------------------------------------------------

def test_flexdata_fields_match_the_twin_without_any_seed(tmp_path):
    """With the 9 fields nulled right before ``_apply_db_overrides`` runs, ``timeline_default``
    must end up with the same values as the explicit twin -- there is no CSV seed left to mask a
    gap.  Before the fix, ``dt_and_step_duration_from_source`` returns ``None`` for
    ``timeline_default`` and ``apply_derived_a`` bails before writing ANY of the 9 fields, so they
    stay ``None`` (the twin, whose ``timeset_duration`` is authored directly, is unaffected)."""
    url = _build_db(tmp_path, _timeline_default_data())
    case_fd = _run_ablated(url, "timeline_default", tmp_path / "w_case")
    twin_fd = _run_ablated(url, "timeline_default_twin", tmp_path / "w_twin")

    for field in _ABLATED_FIELDS:
        diff = field_diff(getattr(case_fd, field), getattr(twin_fd, field))
        assert diff is None, f"{field}: {diff}"


# ---------------------------------------------------------------------------
# (c), critique item 9 extension: the exact 3c7726a9 trigger shape.
# ---------------------------------------------------------------------------

_GENERATED_YAML = """\
profile_lengths: 48

node_locations:
  island:
    number: 1
    distribution:
      name: pert
      min: 50
      mode: 120
      max: 300

grids:
  elec:
    node_group: island
    demand:
      source: demand_elec
      distribution:
        name: pert
        min: 500
        mode: 1200
        max: 2500
      nodes_with_demand: 1
    connections_per_node:
      distribution:
        name: pert_integer
        min: 1
        mode: 1
        max: 1
    cost-per-kw-per-km: 20
    loss-per-100km: 0.01

fuels:
  natgas:
    price: 10

profiles:
  wind:
    auto_correlate:
      pattern_length_average: 12
      pattern_length_st_dev: 4
      distribution:
        name: pert
        min: 0.05
        mode: 0.4
        max: 0.7
  demand_elec:
    auto_correlate:
      pattern_length_average: 24
      pattern_length_st_dev: 0
      distribution:
        name: pert
        min: -1.0
        mode: -0.8
        max: -0.6

technologies:
  natgas:
    elec:
      open_cycle:
        conversion_method: constant_efficiency
        efficiency: 0.35
        invest_cost: 700
  wind:
    elec:
      wind_pp:
        conversion_method: constant_efficiency
        efficiency: 1.0
        invest_cost: 800
"""


def _build_generated_db(tmp_path: Path, tag: str, yaml_path: Path) -> str:
    from flextool._resources import package_data_path
    from flextool.update_flextool import initialize_database
    from flextool.model_builder.build_model import build_model

    db_path = tmp_path / f"{tag}.sqlite"
    initialize_database(str(package_data_path("schemas/spinedb_schema.json")), str(db_path))
    build_model(str(yaml_path), db_url=f"sqlite:///{db_path}", seed=42)
    return f"sqlite:///{db_path}"


def test_flexdata_fields_match_without_any_seed_on_the_generated_timeset_shape(tmp_path):
    """The exact shape that caused the ``3c7726a9`` regression: ``model_builder.build_model``
    writes a ``timeset`` (``generated_timeset``) mapped to its sole timeline
    (``generated_timeline``) but NEVER writes ``timeset.timeset_duration`` -- the model relies
    entirely on Rule 2.  It also authors a wind profile, so ``p_profile_value`` (the field whose
    CSV seed ``3c7726a9`` had to restore because this gap made the DB path bail before reaching
    it) is genuinely populated on both sides, not vacuously ``None``.

    Twin: the identical model (same YAML, same seed -> identical topology) with
    ``generated_timeset.timeset_duration`` authored explicitly as the whole 48-step timeline."""
    from flextool.model_builder.encoding import make_map
    from spinedb_api import DatabaseMapping, import_data

    yaml_path = tmp_path / "m.yaml"
    yaml_path.write_text(_GENERATED_YAML)

    case_url = _build_generated_db(tmp_path, "case", yaml_path)
    twin_url = _build_generated_db(tmp_path, "twin", yaml_path)

    with DatabaseMapping(twin_url) as db:
        duration = make_map(["t0001"], [48.0], index_name="t")
        _num, errors = import_data(db, parameter_values=[
            ("timeset", "generated_timeset", "timeset_duration", duration, "generated")])
        assert not errors, errors
        db.commit_session("twin: explicit whole-timeline timeset_duration")

    case_fd = _run_ablated(case_url, "generated_scenario", tmp_path / "w_case")
    twin_fd = _run_ablated(twin_url, "generated_scenario", tmp_path / "w_twin")

    for field in _ABLATED_FIELDS:
        diff = field_diff(getattr(case_fd, field), getattr(twin_fd, field))
        assert diff is None, f"{field}: {diff}"
    # The profile field must be genuinely populated on both sides -- otherwise this would only
    # prove None == None, not that the DB-side profile cascade actually ran.
    assert case_fd.p_profile_value is not None and case_fd.p_profile_value.frame.height > 0


# ---------------------------------------------------------------------------
# (d) Rolling.
# ---------------------------------------------------------------------------

def _patch_roll_default(data: dict) -> dict:
    """``fullYear_roll`` with every ``timeset_duration`` visible in its alternative chain removed
    (``init`` -- ``2day`` and ``5weeks``'s rows, which this scenario doesn't use but which DO keep
    the scenario-wide ``timeset_durations`` dict non-empty if left in place -- and ``fullYear``,
    the one it actually uses).  ``fullYear``'s own authored duration already covers the WHOLE
    72-step timeline, so plain ``fullYear_roll`` is the correct twin unmodified."""
    data = dict(data)
    data["parameter_values"] = [
        v for v in data["parameter_values"]
        if not (v[0] == "timeset" and v[2] == "timeset_duration" and v[4] in ("init", "fullYear"))]
    return _lp_snapshot()._overlay(data, name="roll_default", base="fullYear_roll", values=[])


@pytest.mark.slow
def test_rolling_whole_timeline_default_builds_the_same_lp(tmp_path_factory):
    """``fullYear_roll`` (72 rolls) with no ``timeset_duration`` anywhere must build the identical
    LP, roll by roll, to plain ``fullYear_roll``.  Guards the per-roll re-evaluation: the DB-side
    readers re-run every roll (critique.md item 6), and this is the only test that exercises the
    default across every one of them, not just the first."""
    lps = _lp_snapshot()
    case_lp, case_obj = _build_and_run(tmp_path_factory, _patch_roll_default, "roll_default")
    twin_lp, twin_obj = _build_and_run(tmp_path_factory, lambda d: d, "fullYear_roll")

    assert len(case_lp) == 72 and len(twin_lp) == 72
    assert [s for s, _, _ in case_lp] == [s for s, _, _ in twin_lp]
    diffs = {}
    for (solve, pa, off_a), (_, pb, off_b) in zip(case_lp, twin_lp):
        msgs = lps.canonical_diff(lps.parse_mps(pa), lps.parse_mps(pb), _RTOL_LP, off_a, off_b)
        if msgs:
            diffs[solve] = msgs
    assert not diffs, f"rolls differ from the twin: {diffs}"
    assert sum(case_obj) == pytest.approx(sum(twin_obj), rel=1e-9)
