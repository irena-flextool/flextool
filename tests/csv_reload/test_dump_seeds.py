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

_PERIODS = ("p2020", "p2025", "p2030", "p2035")


def _invest_retire_limits(cls: str, ent: str) -> list:
    """invest_retire_period_total with non-zero min and max invest / retire limits, per period and in total (the
    values of ``tools/lp_snapshot.py``'s ``invest_limits`` fixture)."""
    def per(value: float) -> dict:
        return {"index_type": "str", "index_name": "period", "data": [[d, value] for d in _PERIODS]}
    return [
        [cls, ent, "invest_method", "invest_retire_period_total"],
        [cls, ent, "invest_max_period", per(400.0)],
        [cls, ent, "invest_min_period", per(1.0)],
        [cls, ent, "invest_max_total", 900.0],
        [cls, ent, "invest_min_total", 2.0],
        [cls, ent, "retire_max_period", per(20.0)],
        [cls, ent, "retire_min_period", per(0.5)],
        [cls, ent, "retire_max_total", 40.0],
        [cls, ent, "retire_min_total", 1.0],
        [cls, ent, "salvage_value", 100.0],  # without a salvage value the entity gets no divest variable
    ]


#: Scenarios built from ``tests.json`` plus one overlay alternative: name -> (base scenario or alternative chain,
#: overlay values, new entities), in ``lp_snapshot._overlay`` form.  For fields no ``tests.json`` scenario seeds.
PATCHED_SCENARIOS: dict[str, tuple[str | list[str], list, list]] = {
    # coal_plant converts coal_market -> west with an efficiency, so its arc is in process_source_sink_eff: an
    # input-side and an output-side other_operational_cost seed pssdt_varCost_eff_unit_source / _sink.
    "coal_varcost_eff_unit": ("coal", [
        ["unit__inputNode", ["coal_plant", "coal_market"], "other_operational_cost", 2.0],
        ["unit__outputNode", ["coal_plant", "west"], "other_operational_cost", 3.0],
    ], []),
    # Every invest entity of the base (coal_plant, battery, battery_inverter) gets non-zero invest / retire minimum
    # totals and a retire_max_period: the DB producers drop unauthored (zero) minimum rows and write per-period caps
    # for authored entities only, while the emit's CSVs carry a row for every invest / divest entity.
    "invest_retire_limits_all": ("network_coal_wind_battery_invest_cumulative", [
        *_invest_retire_limits("unit", "coal_plant"),
        *_invest_retire_limits("node", "battery"),
        *_invest_retire_limits("connection", "battery_inverter"),
    ], []),
    # lp_snapshot's nonsync_internal_dc without its is_DC line: a non-synchronous group {west, east} with flagged
    # wind; west_east runs inside the group.  No tests.json scenario has a non-synchronous group.  (With an is_DC
    # connection the projected ``connection_dc_power_flow`` has a row; the dump writes it with a ``connection``
    # column, and the reload's ``_dc_power_flow.load_data`` reads ``process``, so the reload fails.)
    "nonsync_internal": (["init", "west", "coal", "wind", "network"], [
        ["group", "gwe", "has_non_synchronous", "yes"],
        ["group", "gwe", "non_synchronous_limit", 0.3],
        ["group", "gwe", "penalty_non_synchronous", 4000.0],
        ["unit__outputNode", ["wind_plant", "west"], "is_non_synchronous", "yes"],
    ], [["group", "gwe"], ["group__node", ["gwe", "west"]], ["group__node", ["gwe", "east"]]]),
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
    # S4-3 direct parameters.  coal_co2_price: coal co2_content on a CO2-priced node; multi_year: coal_plant
    # invest_max_total / invest_max_period and a divest cap; invest_retire_limits_all: the minimum totals and
    # retire_max_period, which no tests.json scenario authors.
    "p_co2_content": "coal_co2_price",
    "e_invest_max_total": "multi_year",
    "e_divest_max_total": "multi_year",
    "ed_invest_max_period": "multi_year",
    "ed_divest_max_period": "invest_retire_limits_all",
    "e_invest_min_total": "invest_retire_limits_all",
    "e_divest_min_total": "invest_retire_limits_all",
    # S4-4 group sets and min-load units.  capacity_margin: a manual capacity-margin group; coal_wind_inertia: an
    # inertia group; nonsync_internal: the non-synchronous group, its flagged arcs and the internal line;
    # coal_min_load: coal_plant online with min_load.
    "groupCapacityMargin": "capacity_margin",
    "groupInertia": "coal_wind_inertia",
    "groupNonSync": "nonsync_internal",
    "process_sink_nonSync": "nonsync_internal",
    "process_group_inside_nonSync": "nonsync_internal",
    "process_minload": "coal_min_load",
    # S4-5 profile values and delay tables.  coal_wind_inertia: wind_plant upper_limit profile; water_pump_delayed:
    # the delayed water pump.
    "p_profile_value": "coal_wind_inertia",
    "dtt__delay_duration": "water_pump_delayed",
    "p_process_delay_weight": "water_pump_delayed",
    "process_delayed__duration": "water_pump_delayed",
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

            base, values, entities = PATCHED_SCENARIOS[scenario]
            root = tmp_path_factory.mktemp(f"_root_{scenario}")
            data = _lp_snapshot()._overlay(json.loads(_TESTS_JSON.read_text()), name=scenario, base=base,
                                           values=values, entities=entities)
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


# --- the hook on hand-written dump CSVs (the loader checks of the reads moved here in S4-2, S4-3 and S4-4) ---

_VARCOST_FIELDS = ("pssdt_varCost_noEff", "pssdt_varCost_eff_unit_source", "pssdt_varCost_eff_unit_sink",
                   "pssdt_varCost_eff_connection", "p_pssdt_varCost")


def _seeds(tmp_path: Path, files: dict[str, str], *, inputs: dict[str, str] | None = None, pss: bool = True,
           **attrs) -> SimpleNamespace:
    """Run the hook on a stub FlexData; ``files`` / ``inputs`` are the dump's solve_data / input CSVs."""
    sd, inp = tmp_path / "solve_data", tmp_path / "input"
    sd.mkdir(parents=True)
    inp.mkdir(parents=True)
    for folder, texts in ((sd, files), (inp, inputs or {})):
        for name, text in texts.items():
            (folder / name).write_text(text)
    fd = SimpleNamespace(process_source_sink=pl.DataFrame({"p": ["p1"], "source": ["a"], "sink": ["b"]}) if pss
                         else None, **{name: "untouched" for name in SEEDED_FIELDS}, **attrs)
    apply_dump_field_seeds(fd, sd, inp, provider_from_csv_dirs(inp, sd))
    return fd


def test_varcost_sets_route_to_their_own_field(tmp_path: Path):
    """Each ``pssdt_varCost_*`` set comes from its own CSV only; a set without a CSV is None."""
    fd = _seeds(tmp_path, {
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
    fd = _seeds(tmp_path / "mixed", {
        "pdtProcess__source__sink__dt_varCost.csv": head + "p1,a,b,d1,t1,2.5\np1,a,b,d1,t2,0.0\n"})
    assert fd.p_pssdt_varCost.dims == ("p", "source", "sink", "d", "t")
    assert fd.p_pssdt_varCost.frame.rows() == [("p1", "a", "b", "d1", "t1", 2.5)]
    fd = _seeds(tmp_path / "zero", {"pdtProcess__source__sink__dt_varCost.csv": head + "p1,a,b,d1,t1,0.0\n"})
    assert fd.p_pssdt_varCost is None


def test_varcost_seeds_skipped_without_process_arcs(tmp_path: Path):
    """No ``process_source_sink`` -> the hook writes no variable-cost field, whatever the dump holds."""
    fd = _seeds(tmp_path, {
        "pssdt_varCost_noEff.csv": "process,source,sink,period,time\np1,a,b,d1,t1\n",
        "pdtProcess__source__sink__dt_varCost.csv": "process,source,sink,period,time,value\np1,a,b,d1,t1,9.0\n",
    }, pss=False)
    assert all(getattr(fd, name) == "untouched" for name in _VARCOST_FIELDS)


_P_COMMODITY = "commodity,commodityParam,p_commodity\ncoal,co2_content,0.34\ncoal,price,20.0\ngas,co2_content,0.2\n"


@pytest.mark.parametrize("gate", ["flow_from_co2_priced", "p_co2_max_period", "p_co2_max_total"])
def test_co2_content_from_p_commodity_when_co2_priced_or_capped(tmp_path: Path, gate: str):
    """CO2-priced flows or either CO2 cap -> ``p_co2_content`` = the ``co2_content`` rows of p_commodity."""
    fd = _seeds(tmp_path, {}, inputs={"p_commodity.csv": _P_COMMODITY},
                **{"flow_from_co2_priced": None, "p_co2_max_period": None, "p_co2_max_total": None, gate: "set"})
    assert fd.p_co2_content.dims == ("c",)
    assert sorted(fd.p_co2_content.frame.rows()) == [("coal", 0.34), ("gas", 0.2)]


def test_co2_content_untouched_without_co2_price_or_cap(tmp_path: Path):
    """No CO2-priced flow and no CO2 cap -> the hook leaves ``p_co2_content`` alone (the loader's None)."""
    fd = _seeds(tmp_path, {}, inputs={"p_commodity.csv": _P_COMMODITY},
                flow_from_co2_priced=None, p_co2_max_period=None, p_co2_max_total=None)
    assert fd.p_co2_content == "untouched"


_INVEST_CAP_CSVS = {
    "e_invest_max_total.csv": "entity,value\ne1,700.0\ne2,\n",
    "e_divest_max_total.csv": "entity,value\ne1,40.0\n",
    "ed_invest_max_period.csv": "entity,period,value\ne1,p1,250.0\ne1,p2,0.0\n",
    "ed_divest_max_period.csv": "solve,period,e1,e2\ns,p1,20.0,5.0\n",
}


@pytest.mark.parametrize("gate_csv", ["ed_invest.csv", "ed_divest.csv"])
def test_invest_caps_when_an_invest_or_divest_set_has_rows(tmp_path: Path, gate_csv: str):
    """``ed_invest`` or ``ed_divest`` with a row -> the four max caps; an empty value reads 0, zero rows stay, the
    per-period caps read long or wide CSVs."""
    fd = _seeds(tmp_path, {**_INVEST_CAP_CSVS, gate_csv: "entity,period\ne1,p1\n"})
    assert fd.e_invest_max_total.dims == ("e",)
    assert sorted(fd.e_invest_max_total.frame.rows()) == [("e1", 700.0), ("e2", 0.0)]
    assert fd.e_divest_max_total.frame.rows() == [("e1", 40.0)]
    assert fd.ed_invest_max_period.dims == ("e", "d")
    assert sorted(fd.ed_invest_max_period.frame.rows()) == [("e1", "p1", 250.0), ("e1", "p2", 0.0)]
    assert sorted(fd.ed_divest_max_period.frame.rows()) == [("e1", "p1", 20.0), ("e2", "p1", 5.0)]


def test_invest_caps_untouched_without_invest_rows(tmp_path: Path):
    """Header-only ``ed_invest`` / ``ed_divest`` -> the hook leaves the four max caps alone (the loader's None)."""
    fd = _seeds(tmp_path, {**_INVEST_CAP_CSVS, "ed_invest.csv": "entity,period\n",
                           "ed_divest.csv": "entity,period\n"})
    assert all(getattr(fd, name) == "untouched" for name in
               ("e_invest_max_total", "e_divest_max_total", "ed_invest_max_period", "ed_divest_max_period"))


def test_invest_caps_none_when_their_csv_is_missing_or_empty(tmp_path: Path):
    """An invest row but no / a header-only cap CSV -> that cap is None."""
    fd = _seeds(tmp_path, {"ed_invest.csv": "entity,period\ne1,p1\n", "e_divest_max_total.csv": "entity,value\n",
                           "ed_divest_max_period.csv": "entity,period,value\n"})
    assert (fd.e_invest_max_total, fd.e_divest_max_total, fd.ed_invest_max_period, fd.ed_divest_max_period) == (
        None, None, None, None)


def test_min_invest_totals_always_read(tmp_path: Path):
    """``e_invest_min_total`` / ``e_divest_min_total`` need no invest gate: a value casts to float, a missing CSV is
    None (the A16 loader check of the read moved here in S4-3)."""
    fd = _seeds(tmp_path, {"e_invest_min_total.csv": "entity,value\ne1,3.0\n"})
    assert fd.e_invest_min_total.dims == ("e",)
    assert fd.e_invest_min_total.frame.rows() == [("e1", 3.0)]
    assert fd.e_divest_min_total is None


def test_group_sets_come_from_the_flag_files(tmp_path: Path):
    """The three group sets are the groups in the method / flag files (``input/groupCapacityMargin.csv``,
    ``groupInertia.csv``, ``groupNonSync.csv``); limit values do not make a group (#69).  The A14 loader check of
    the reads moved here in S4-4."""
    fd = _seeds(tmp_path, {
        "pdGroup_capacity_margin.csv": "period,g1,g3\nd1,0.0,0.5\n",
        "pdGroup.csv": "group,param,period,value\ng3,inertia_limit,d1,5.0\ng_other,non_synchronous_limit,d1,0.3\n",
    }, inputs={
        "groupCapacityMargin.csv": "groupCapacityMargin\ng1\ng2\ng1\n",
        "groupInertia.csv": "groupInertia\ng1\n",
        "groupNonSync.csv": "groupNonSync\ng_sync_only\n",
    })
    assert fd.groupCapacityMargin.columns == ["g"]
    assert sorted(fd.groupCapacityMargin["g"].to_list()) == ["g1", "g2"]
    assert fd.groupInertia["g"].to_list() == ["g1"]
    assert fd.groupNonSync["g"].to_list() == ["g_sync_only"]


def test_limits_without_flag_files_give_no_group_sets(tmp_path: Path):
    """Limit values without the flag files give None sets (#69); a header-only flag file is the same as none."""
    fd = _seeds(tmp_path, {
        "pdGroup_capacity_margin.csv": "period,g1,g2\nd1,0.0,0.5\n",
        "pdGroup_inertia_limit.csv": "period,g1\nd1,5.0\n",
        "pdGroup.csv": "group,param,period,value\ng_other,non_synchronous_limit,d1,0.3\n",
    }, inputs={"groupInertia.csv": "groupInertia\n"})
    assert (fd.groupCapacityMargin, fd.groupInertia, fd.groupNonSync) == (None, None, None)


def test_non_sync_arcs_and_internal_transfers(tmp_path: Path):
    """``process_sink_nonSync`` is the dump's ``process__sink_nonSync.csv`` as ``(p, sink)`` (duplicates dropped);
    ``process_group_inside_nonSync`` follows the production rule on the reloaded topology: p1 (a -> b) has a source
    and a different sink node in the non-synchronous group g = {a, b}, so it is inside g; a dumped
    ``process__group_inside_group_nonSync.csv`` is not read."""
    files = {"process__sink_nonSync.csv": "process,sink\nw1,a\nw1,a\nc1,b\n",
             "process__group_inside_group_nonSync.csv": "process,group\nstale,g\n"}
    fd = _seeds(tmp_path / "in", files, inputs={"groupNonSync.csv": "groupNonSync\ng\n"},
                group_node=pl.DataFrame({"g": ["g", "g"], "n": ["a", "b"]}))
    assert fd.process_sink_nonSync.columns == ["p", "sink"]
    assert sorted(fd.process_sink_nonSync.rows()) == [("c1", "b"), ("w1", "a")]
    assert fd.process_group_inside_nonSync.rows() == [("p1", "g")]
    # Only the source a is in g = {a}: p1 crosses the boundary, nothing is inside.
    fd = _seeds(tmp_path / "out", files, inputs={"groupNonSync.csv": "groupNonSync\ng\n"},
                group_node=pl.DataFrame({"g": ["g"], "n": ["a"]}))
    assert fd.process_group_inside_nonSync is None
    # No non-synchronous group: no internal transfers, and no arc file gives no arcs.
    fd = _seeds(tmp_path / "none", {}, group_node=pl.DataFrame({"g": ["g"], "n": ["a"]}))
    assert (fd.groupNonSync, fd.process_sink_nonSync, fd.process_group_inside_nonSync) == (None, None, None)


def test_minload_set_when_the_run_has_online_processes(tmp_path: Path):
    """With online processes, ``process_minload`` is the dumped ``process_minload.csv`` as a ``(p,)`` set, or an
    empty ``(p,)`` set without the file; without online processes the hook leaves it alone (the loader's None)."""
    online = pl.DataFrame({"p": ["coal_plant"]})
    fd = _seeds(tmp_path / "file", {"process_minload.csv": "process\ncoal_plant\n"}, process_online=online)
    assert fd.process_minload.columns == ["p"]
    assert fd.process_minload["p"].to_list() == ["coal_plant"]
    fd = _seeds(tmp_path / "missing", {}, process_online=online)
    assert fd.process_minload.height == 0
    assert fd.process_minload.schema == {"p": pl.Utf8}
    fd = _seeds(tmp_path / "offline", {"process_minload.csv": "process\ncoal_plant\n"}, process_online=None)
    assert fd.process_minload == "untouched"


_PDT_PROFILE = "solve,period,time,wind,ev\ns,d1,t1,0.5,0.25\ns,d1,t2,0.75,0.0\n"


def test_profile_values_when_the_run_has_profile_arcs(tmp_path: Path):
    """With the loader's profile-method sets (``process_profile_upper``), ``p_profile_value`` is the emitted wide
    ``pdtProfile.csv`` as ``(f, d, t)``; without the file it is None; without profile arcs the hook leaves it alone
    (the loader's None).  The ``_load_profiles`` read moved here in S4-5."""
    upper = pl.DataFrame({"p": ["w"], "source": ["w"], "sink": ["n"], "f": ["wind"]})
    fd = _seeds(tmp_path / "file", {"pdtProfile.csv": _PDT_PROFILE}, process_profile_upper=upper)
    assert fd.p_profile_value.dims == ("f", "d", "t")
    assert sorted(fd.p_profile_value.frame.rows()) == [
        ("ev", "d1", "t1", 0.25), ("ev", "d1", "t2", 0.0), ("wind", "d1", "t1", 0.5), ("wind", "d1", "t2", 0.75)]
    fd = _seeds(tmp_path / "missing", {}, process_profile_upper=upper)
    assert fd.p_profile_value is None
    fd = _seeds(tmp_path / "no_arcs", {"pdtProfile.csv": _PDT_PROFILE}, process_profile_upper=None)
    assert fd.p_profile_value == "untouched"


_DELAY_CSVS = {
    "dtt__delay_duration.csv": "period,time_source,time_sink,delay_duration\nd1,t1,t2,1\nd1,t2,t1,1.0\n",
    "p_process_delay_weight.csv": "process,delay_duration,value\npump,1,0.6\npump,2.0,0.4\n",
}


def test_delay_tables_when_a_process_is_delayed(tmp_path: Path):
    """With a delayed process, the shift table and the weights come from the emitted CSVs with Float64 ``td`` ("1"
    and "1.0" read the same), and ``process_delayed__duration`` is the weights' ``(p, td)`` keys; a missing CSV is
    None (and no weights give no durations).  The ``_delay.load_data`` reads moved here in S4-5."""
    delayed = pl.DataFrame({"p": ["pump"]})
    fd = _seeds(tmp_path / "file", _DELAY_CSVS, process_delayed=delayed)
    assert fd.dtt__delay_duration.columns == ["d", "t_source", "t_sink", "td"]
    assert fd.dtt__delay_duration.schema["td"] == pl.Float64
    assert sorted(fd.dtt__delay_duration.rows()) == [("d1", "t1", "t2", 1.0), ("d1", "t2", "t1", 1.0)]
    assert fd.p_process_delay_weight.dims == ("p", "td")
    assert sorted(fd.p_process_delay_weight.frame.rows()) == [("pump", 1.0, 0.6), ("pump", 2.0, 0.4)]
    assert fd.process_delayed__duration.schema["td"] == pl.Float64
    assert sorted(fd.process_delayed__duration.select("p", "td").rows()) == [("pump", 1.0), ("pump", 2.0)]
    fd = _seeds(tmp_path / "missing", {}, process_delayed=delayed)
    assert (fd.dtt__delay_duration, fd.p_process_delay_weight, fd.process_delayed__duration) == (None, None, None)


@pytest.mark.parametrize("delayed", [None, pl.DataFrame(schema={"p": pl.Utf8})], ids=["none", "empty"])
def test_delay_tables_untouched_without_delayed_processes(tmp_path: Path, delayed):
    """No delayed process -> the hook leaves the three delay tables alone (the loader's None)."""
    fd = _seeds(tmp_path, _DELAY_CSVS, process_delayed=delayed)
    assert all(getattr(fd, name) == "untouched" for name in
               ("dtt__delay_duration", "p_process_delay_weight", "process_delayed__duration"))
