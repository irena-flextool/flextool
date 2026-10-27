"""The reload's field seeds (``flextool.engine._csv_reload._field_seeds``) round-trip a dump (c24 stage 4).

Stage 4 removes from ``load_flextool`` the seeds that the DB overlay always overwrites in production; a removed
seed whose field a reader-less reload still needs moves to :func:`apply_dump_field_seeds`, which the loader runs
through ``_dump_hooks.field_seeds`` where the overlay would run.

* The hook runs exactly once on a reader-less reload, on the FlexData the reload returns, with the dump's
  ``solve_data`` / ``input`` folders; it does not run when the reload has a DB reader (the overlay runs instead).
* The hook writes only the fields it declares (:data:`SEEDED_FIELDS`).
* Round trip: for every declared field, the reader-less ``load_from_dump(dump)`` equals the production FlexData
  (``load_scenario(run)``) of the run the dump came from.  :data:`SCENARIO_OF_FIELD` names, per field, a
  ``tests.json`` scenario in which the field is seeded, or a scenario of :data:`PATCHED_SCENARIOS` (``tests.json``
  plus one overlay alternative) when no ``tests.json`` scenario seeds it; a field the emit and the DB legitimately
  disagree on goes on :data:`EXPECTED_DIFF` with the reason.

Each stage-4 commit that moves a seed here adds its fields to :data:`SCENARIO_OF_FIELD`.
"""
from __future__ import annotations

import dataclasses
import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from flextool.engine._csv_reload import CsvReloadWarning, copy_runner_meta, load_from_dump
from flextool.engine._csv_reload import load as reload_load
from flextool.engine._csv_reload._field_seeds import SEEDED_FIELDS, apply_dump_field_seeds
from flextool.engine.preprocess.flex_data import FlexData
from tests._support.fields import field_diff
from tests._support.load import load_scenario, run_scenario
from tests._support.provider import provider_from_csv_dirs
from tests.engine.test_cyclic_fix_start import _lp_snapshot

_TESTS_JSON = Path(__file__).resolve().parents[1] / "fixtures" / "tests.json"

#: Scenarios built from ``tests.json`` plus one overlay alternative: name -> (base scenario, overlay values in
#: ``lp_snapshot._overlay`` form).  For fields no ``tests.json`` scenario seeds.
PATCHED_SCENARIOS: dict[str, tuple[str, list]] = {
    # coal_plant converts coal_market -> west with an efficiency, so its arc is in process_source_sink_eff: an
    # input-side and an output-side other_operational_cost seed pssdt_varCost_eff_unit_source / _sink.
    "coal_varcost_eff_unit": ("coal", [
        ["unit__inputNode", ["coal_plant", "coal_market"], "other_operational_cost", 2.0],
        ["unit__outputNode", ["coal_plant", "west"], "other_operational_cost", 3.0],
    ]),
}

#: field -> a scenario whose last solve seeds it (``tests.json`` or :data:`PATCHED_SCENARIOS`).  Must name exactly
#: :data:`SEEDED_FIELDS`.
SCENARIO_OF_FIELD: dict[str, str] = {
    # S4-2 variable cost.  fusion: fusion_plant -> west output cost on a unit without conversion (noEff);
    # hyphenated_entity_names: connection west-east-aux other_operational_cost.
    "pssdt_varCost_noEff": "fusion",
    "pssdt_varCost_eff_unit_source": "coal_varcost_eff_unit",
    "pssdt_varCost_eff_unit_sink": "coal_varcost_eff_unit",
    "pssdt_varCost_eff_connection": "hyphenated_entity_names",
    "p_pssdt_varCost": "fusion",
}

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
def run_of(tmp_path_factory, scenario_run, test_solver_config_dir):
    """Factory: scenario -> its run; a :data:`PATCHED_SCENARIOS` name runs on its own patched DB (once per module)."""
    cache: dict[str, object] = {}

    def factory(scenario: str):
        if scenario not in PATCHED_SCENARIOS:
            return scenario_run(scenario)
        if scenario not in cache:
            from flextool.update_flextool.db_migration import migrate_database
            from tests.db_utils import json_to_db

            base, values = PATCHED_SCENARIOS[scenario]
            root = tmp_path_factory.mktemp(f"_root_{scenario}")
            data = _lp_snapshot()._overlay(json.loads(_TESTS_JSON.read_text()), name=scenario, base=base,
                                           values=values)
            (root / "db.json").write_text(json.dumps(data))
            url = json_to_db(root / "db.json", root / "db.sqlite")
            migrate_database(url)
            cache[scenario] = run_scenario(url, scenario, root / f"work_{scenario}",
                                           solver_config_dir=test_solver_config_dir)
        return cache[scenario]

    return factory


@pytest.fixture(scope="module")
def dump_of(tmp_path_factory, run_of):
    """Factory: scenario -> a stamped dump of its run (one per scenario per module; never change it)."""
    cache: dict[str, Path] = {}

    def factory(scenario: str) -> Path:
        if scenario not in cache:
            cache[scenario] = _dump_of(run_of(scenario), tmp_path_factory.mktemp(f"{scenario}_dump") / "dump")
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
def test_seeded_fields_round_trip(scenario, dump_of, run_of):
    """Every declared field seeded in ``scenario`` equals the production FlexData of the dumped run."""
    run = run_of(scenario)
    reloaded = _reload(dump_of(scenario))
    production = load_scenario(run)
    fields = [f for f, s in SCENARIO_OF_FIELD.items() if s == scenario]
    for name in fields:
        assert getattr(production, name) is not None, f"{name}: not seeded in {scenario!r}; pick another scenario"
    diffs = {name: field_diff(getattr(reloaded, name), getattr(production, name))
             for name in fields if name not in EXPECTED_DIFF}
    assert {name: d for name, d in diffs.items() if d is not None} == {}


# --- the variable-cost seeds on hand-written dump CSVs (the A8 loader checks of the reads moved here in S4-2) ---

def _varcost_seeds(tmp_path: Path, files: dict[str, str], *, pss: bool = True) -> SimpleNamespace:
    sd = tmp_path / "solve_data"
    sd.mkdir(parents=True)
    for name, text in files.items():
        (sd / name).write_text(text)
    fd = SimpleNamespace(process_source_sink=pl.DataFrame({"p": ["p1"], "source": ["a"], "sink": ["b"]}) if pss
                         else None, **{name: "untouched" for name in SEEDED_FIELDS})
    apply_dump_field_seeds(fd, sd, tmp_path / "input", provider_from_csv_dirs(solve_data_dir=sd))
    return fd


def test_varcost_sets_route_to_their_own_field(tmp_path: Path):
    """Each ``pssdt_varCost_*`` set comes from its own CSV only; a set without a CSV is None."""
    fd = _varcost_seeds(tmp_path, {
        "pssdt_varCost_eff_unit_source.csv": "process,source,sink,period,time\np1,n_src,p1,d1,t1\n",
        "pssdt_varCost_eff_unit_sink.csv": "process,source,sink,period,time\np1,p1,n_snk,d1,t1\n",
        "pssdt_varCost_eff_connection.csv": "process,source,sink,period,time\nc1,n1,n2,d1,t1\n",
    })
    assert fd.pssdt_varCost_noEff is None
    assert fd.p_pssdt_varCost is None
    rows = {name: getattr(fd, name).rows() for name in
            ("pssdt_varCost_eff_unit_source", "pssdt_varCost_eff_unit_sink", "pssdt_varCost_eff_connection")}
    assert rows == {"pssdt_varCost_eff_unit_source": [("p1", "n_src", "p1", "d1", "t1")],
                    "pssdt_varCost_eff_unit_sink": [("p1", "p1", "n_snk", "d1", "t1")],
                    "pssdt_varCost_eff_connection": [("c1", "n1", "n2", "d1", "t1")]}
    assert fd.pssdt_varCost_eff_connection.columns == ["p", "source", "sink", "d", "t"]


def test_varcost_param_drops_zero_rows(tmp_path: Path):
    """``p_pssdt_varCost`` keeps the non-zero rows; all-zero rows give None."""
    head = "process,source,sink,period,time,value\n"
    fd = _varcost_seeds(tmp_path / "mixed", {
        "pdtProcess__source__sink__dt_varCost.csv": head + "p1,a,b,d1,t1,2.5\np1,a,b,d1,t2,0.0\n"})
    assert fd.p_pssdt_varCost.dims == ("p", "source", "sink", "d", "t")
    assert fd.p_pssdt_varCost.frame.rows() == [("p1", "a", "b", "d1", "t1", 2.5)]
    fd = _varcost_seeds(tmp_path / "zero", {"pdtProcess__source__sink__dt_varCost.csv": head + "p1,a,b,d1,t1,0.0\n"})
    assert fd.p_pssdt_varCost is None


def test_varcost_seeds_skipped_without_process_arcs(tmp_path: Path):
    """No ``process_source_sink`` -> the hook writes no variable-cost field, whatever the dump holds."""
    fd = _varcost_seeds(tmp_path, {
        "pssdt_varCost_noEff.csv": "process,source,sink,period,time\np1,a,b,d1,t1\n",
        "pdtProcess__source__sink__dt_varCost.csv": "process,source,sink,period,time,value\np1,a,b,d1,t1,9.0\n",
    }, pss=False)
    assert all(getattr(fd, name) == "untouched" for name in SEEDED_FIELDS)
