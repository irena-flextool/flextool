"""The reload's field seeds (``flextool.engine._csv_reload._field_seeds``) round-trip a dump (c24 stage 4).

Stage 4 removes from ``load_flextool`` the seeds that the DB overlay always overwrites in production; a removed
seed whose field a reader-less reload still needs moves to :func:`apply_dump_field_seeds`, which the loader runs
through ``_dump_hooks.field_seeds`` where the overlay would run.

* The hook runs exactly once on a reader-less reload, on the FlexData the reload returns, with the dump's
  ``solve_data`` / ``input`` folders; it does not run when the reload has a DB reader (the overlay runs instead).
* The hook writes only the fields it declares (:data:`SEEDED_FIELDS`).
* Round trip: for every declared field, the reader-less ``load_from_dump(dump)`` equals the production FlexData
  (``load_scenario(run)``) of the run the dump came from.  :data:`SCENARIO_OF_FIELD` names, per field, a
  ``tests.json`` scenario in which the field is seeded; a field the emit and the DB legitimately disagree on
  goes on :data:`EXPECTED_DIFF` with the reason.

Each stage-4 commit that moves a seed here adds its fields to :data:`SCENARIO_OF_FIELD`.
"""
from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import pytest

from flextool.engine._csv_reload import CsvReloadWarning, copy_runner_meta, load_from_dump
from flextool.engine._csv_reload import load as reload_load
from flextool.engine._csv_reload._field_seeds import SEEDED_FIELDS
from flextool.engine.preprocess.flex_data import FlexData
from tests._support.fields import field_diff
from tests._support.load import load_scenario

#: field -> a ``tests.json`` scenario whose last solve seeds it.  Must name exactly :data:`SEEDED_FIELDS`.
SCENARIO_OF_FIELD: dict[str, str] = {}

#: Fields where the reload legitimately differs from production: field -> reason.  None yet.
EXPECTED_DIFF: dict[str, str] = {}

#: The scenario of the mechanism tests; it is also round-tripped (with no declared field it compares nothing).
_BASE_SCENARIO = "coal"

_ROUND_TRIP_SCENARIOS = sorted(set(SCENARIO_OF_FIELD.values()) | {_BASE_SCENARIO})


def _dump_of(run, out: Path) -> Path:
    """``dump_csvs`` of the run's last FlexData plus the run's metadata CSVs: a reloadable dump folder."""
    run.last_flex_data.dump_csvs(out, include_heavy=True)
    assert copy_runner_meta(run.work_folder, out) > 0
    return out


@pytest.fixture(scope="module")
def dump_of(tmp_path_factory, scenario_run):
    """Factory: scenario -> a stamped dump of its run (one per scenario per module; never change it)."""
    cache: dict[str, Path] = {}

    def factory(scenario: str) -> Path:
        if scenario not in cache:
            cache[scenario] = _dump_of(scenario_run(scenario), tmp_path_factory.mktemp(f"{scenario}_dump") / "dump")
        return cache[scenario]

    return factory


def _reload(dump: Path, **kwargs) -> FlexData:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CsvReloadWarning)
        return load_from_dump(dump, **kwargs)


class _Recorder:
    """Wraps the real field-seeds hook; records each call and which fields the call changed."""

    def __init__(self, inner):
        self.inner = inner
        self.calls: list[tuple[object, Path, Path, object]] = []
        self.changed: list[set[str]] = []

    def __call__(self, flex_data, sd, inp, provider) -> None:
        before = {f.name: getattr(flex_data, f.name) for f in dataclasses.fields(flex_data)}
        self.inner(flex_data, sd, inp, provider)
        self.calls.append((flex_data, Path(sd), Path(inp), provider))
        self.changed.append({name for name, value in before.items() if getattr(flex_data, name) is not value})


@pytest.fixture
def recorder(monkeypatch) -> _Recorder:
    rec = _Recorder(reload_load._DUMP_HOOKS.field_seeds)
    monkeypatch.setattr(reload_load, "_DUMP_HOOKS", dataclasses.replace(reload_load._DUMP_HOOKS, field_seeds=rec))
    return rec


def test_declared_fields_have_a_scenario_and_are_flexdata_fields():
    """Every seeded field names a FlexData field and a round-trip scenario; nothing else is listed."""
    names = {f.name for f in dataclasses.fields(FlexData)}
    assert set(SEEDED_FIELDS) <= names, sorted(set(SEEDED_FIELDS) - names)
    assert len(set(SEEDED_FIELDS)) == len(SEEDED_FIELDS)
    assert set(SCENARIO_OF_FIELD) == set(SEEDED_FIELDS)
    assert set(EXPECTED_DIFF) <= set(SEEDED_FIELDS)


def test_hook_runs_once_on_a_reader_less_reload(dump_of, recorder):
    """One call, on the returned FlexData, with the dump's folders; it changes only the declared fields."""
    dump = dump_of(_BASE_SCENARIO)
    fd = _reload(dump)
    assert len(recorder.calls) == 1
    flex_data, sd, inp, _provider = recorder.calls[0]
    assert flex_data is fd
    assert sd == dump / "solve_data"
    assert inp == dump / "input"
    assert recorder.changed[0] <= set(SEEDED_FIELDS), sorted(recorder.changed[0] - set(SEEDED_FIELDS))


def test_hook_does_not_run_with_a_reader(dump_of, scenario_run, recorder):
    """With ``db_url`` + ``scenario`` the DB overlay runs and the field seeds do not."""
    run = scenario_run(_BASE_SCENARIO)
    _reload(dump_of(_BASE_SCENARIO), db_url=run.db_url, scenario=run.scenario)
    assert recorder.calls == []


@pytest.mark.parametrize("scenario", _ROUND_TRIP_SCENARIOS)
def test_seeded_fields_round_trip(scenario, dump_of, scenario_run):
    """Every declared field seeded in ``scenario`` equals the production FlexData of the dumped run."""
    run = scenario_run(scenario)
    reloaded = _reload(dump_of(scenario))
    production = load_scenario(run)
    fields = [f for f, s in SCENARIO_OF_FIELD.items() if s == scenario]
    for name in fields:
        assert getattr(production, name) is not None, f"{name}: not seeded in {scenario!r}; pick another scenario"
    diffs = {name: field_diff(getattr(reloaded, name), getattr(production, name))
             for name in fields if name not in EXPECTED_DIFF}
    assert {name: d for name, d in diffs.items() if d is not None} == {}
