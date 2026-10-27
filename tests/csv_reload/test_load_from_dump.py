"""``flextool.engine._csv_reload.load_from_dump``: stamp check, warning, arguments and parity (c24 stage 2, S2-2).

* Every call emits one :class:`CsvReloadWarning` that points at the caller's line.
* A dump without a readable stamp (A), with another dump format (B) or another schema version (C) is refused
  without ``force`` and loads with it.
* ``db_url`` and ``scenario`` come together.
* Parity while the loader's own CSV topology branch still exists: on a ``dump_csvs`` folder, ``load_from_dump``
  (topology from the package's ``pss_from_dump``) builds the same FlexData as a bare ``load_flextool`` (topology
  from the loader's inline branch).
* No discovery: a reload of a ``work_<scenario>`` folder that holds ``tests.sqlite`` builds no axis Enums, where a
  bare ``load_flextool`` of the same folder does.

Dumps come from the session-cached cascade runs (``scenario_run``), whose DBs are built from the JSON fixtures.
"""
from __future__ import annotations

import dataclasses
import json
import shutil
import warnings
from pathlib import Path

import polars as pl
import pytest

from polar_high import Param
from flextool.engine._csv_reload import (
    SUPPORTED_DUMP_FORMAT,
    CsvReloadWarning,
    copy_runner_meta,
    load_from_dump,
    load_from_dump_with_provider,
)
from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.preprocess._dump_stamp import STAMP_FILE, _engine_identity, read_dump_stamp
from flextool.engine.preprocess.flex_data import FlexData, load_flextool

_UNSUPPORTED = "CSV reload is unsupported"


def _dump_of(run, out: Path) -> Path:
    """``dump_csvs`` of the run's last FlexData plus the run's metadata CSVs: a reloadable dump folder."""
    run.last_flex_data.dump_csvs(out, include_heavy=True)
    assert copy_runner_meta(run.work_folder, out) > 0
    return out


@pytest.fixture(scope="module")
def coal_dump(tmp_path_factory, scenario_run) -> Path:
    """A stamped, reloadable dump of ``coal`` (shared by this module; copy it before changing it)."""
    return _dump_of(scenario_run("coal"), tmp_path_factory.mktemp("coal_dump") / "dump")


def _copy(dump: Path, tmp_path: Path) -> Path:
    target = tmp_path / "dump"
    shutil.copytree(dump, target)
    return target


def _rewrite_stamp(dump: Path, **changes) -> None:
    stamp = read_dump_stamp(dump)
    assert stamp is not None
    stamp.update(changes)
    (dump / STAMP_FILE).write_text(json.dumps(stamp), encoding="utf-8")


def _reload_warnings(record) -> list[warnings.WarningMessage]:
    return [w for w in record if issubclass(w.category, CsvReloadWarning)]


# ---------------------------------------------------------------------------
# Warning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", ["load_from_dump", "load_from_dump_with_provider"])
def test_every_call_warns_once_at_the_caller(coal_dump, entry):
    """One CsvReloadWarning per call (also on a repeat call), pointing at this file, with the documented text."""
    fn = {"load_from_dump": load_from_dump, "load_from_dump_with_provider": load_from_dump_with_provider}[entry]
    for _ in range(2):
        with pytest.warns(CsvReloadWarning, match=_UNSUPPORTED) as record:
            fn(coal_dump)
        reload = _reload_warnings(record)
        assert len(reload) == 1
        assert Path(reload[0].filename).resolve() == Path(__file__).resolve()
    text = str(reload[0].message)
    stamp = read_dump_stamp(coal_dump)
    engine = _engine_identity()
    git = "unknown" if stamp["git"] is None else stamp["git"]
    assert f"dump format {stamp['dump_format']}, flextool {stamp['flextool']} (git {git})" in text
    assert f"This engine: dump format {SUPPORTED_DUMP_FORMAT}, flextool {engine['flextool']}" in text
    assert "Reload last verified with flextool" in text
    assert "CSVs left in a reused folder by an earlier run are not detected" in text
    assert "earlier dump" not in text


def test_previous_stamp_adds_the_mixed_folder_sentence(coal_dump, tmp_path):
    dump = _copy(coal_dump, tmp_path)
    _rewrite_stamp(dump, previous={"dump_format": 1, "flextool": "0.0.1", "schema_version": 1, "git": None})
    with pytest.warns(CsvReloadWarning, match=_UNSUPPORTED) as record:
        load_from_dump(dump)
    text = str(_reload_warnings(record)[0].message)
    assert "The folder also holds files of an earlier dump (flextool 0.0.1, schema 1); stale CSVs may be read." in text


# ---------------------------------------------------------------------------
# Refusals (A), (B), (C) and force
# ---------------------------------------------------------------------------


def _no_stamp(dump: Path) -> None:
    (dump / STAMP_FILE).unlink()


def _garbage_stamp(dump: Path) -> None:
    (dump / STAMP_FILE).write_text("{not json", encoding="utf-8")


def _incomplete_stamp(dump: Path) -> None:
    (dump / STAMP_FILE).write_text(json.dumps({"git": None, "writers": ["dump_csvs"]}), encoding="utf-8")


def _other_format(dump: Path) -> None:
    _rewrite_stamp(dump, dump_format=SUPPORTED_DUMP_FORMAT + 1)


def _other_schema(dump: Path) -> None:
    _rewrite_stamp(dump, schema_version=_engine_identity()["schema_version"] + 1)


@pytest.mark.parametrize("spoil,match,forced_desc", [
    (_no_stamp, f"has no readable {STAMP_FILE}", f"no {STAMP_FILE}"),
    (_garbage_stamp, f"has no readable {STAMP_FILE}", f"no {STAMP_FILE}"),
    (_incomplete_stamp, f"has no readable {STAMP_FILE}", "dump format unknown, flextool unknown (git unknown)"),
    (_other_format, f"is dump format {SUPPORTED_DUMP_FORMAT + 1}; this CSV reload reads dump format "
                    f"{SUPPORTED_DUMP_FORMAT}", f"dump format {SUPPORTED_DUMP_FORMAT + 1}"),
    (_other_schema, "was written with database schema", "schema "),
], ids=["A_absent", "A_unparsable", "A_missing_fields", "B_format", "C_schema"])
def test_incompatible_dump_is_refused_unless_forced(coal_dump, tmp_path, spoil, match, forced_desc):
    dump = _copy(coal_dump, tmp_path)
    spoil(dump)
    with pytest.raises(FlexToolConfigError, match=match) as excinfo:
        load_from_dump(dump)
    assert "pass force=True to load it anyway" in str(excinfo.value)
    with pytest.warns(CsvReloadWarning, match=_UNSUPPORTED) as record:
        fd = load_from_dump(dump, force=True)
    assert isinstance(fd, FlexData)
    assert fd.process_source_sink is not None and fd.process_source_sink.height > 0
    assert forced_desc in str(_reload_warnings(record)[0].message)


def test_missing_folder_is_not_found_even_when_forced(tmp_path):
    for force in (False, True):
        with pytest.raises(FileNotFoundError):
            load_from_dump(tmp_path / "absent", force=force)


def test_other_flextool_version_and_git_only_warn(coal_dump, tmp_path):
    dump = _copy(coal_dump, tmp_path)
    _rewrite_stamp(dump, flextool="0.0.1", git="deadbeef")
    with pytest.warns(CsvReloadWarning, match=_UNSUPPORTED) as record:
        load_from_dump(dump)
    assert "flextool 0.0.1 (git deadbeef)" in str(_reload_warnings(record)[0].message)


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------


def test_db_url_and_scenario_come_together(coal_dump, scenario_run):
    run = scenario_run("coal")
    with pytest.raises(ValueError, match="db_url and scenario= must be given together"):
        load_from_dump(coal_dump, db_url=run.db_url)
    with pytest.raises(ValueError, match="db_url and scenario= must be given together"):
        load_from_dump(coal_dump, scenario=run.scenario)


# ---------------------------------------------------------------------------
# Parity: pss_from_dump ≡ the loader's inline CSV topology branch
# ---------------------------------------------------------------------------


def _frame(value) -> pl.DataFrame | None:
    if isinstance(value, Param):
        return value.frame
    if isinstance(value, pl.DataFrame):
        return value
    return None


def _same_frame(a: pl.DataFrame, b: pl.DataFrame) -> bool:
    if a.schema != b.schema or a.shape != b.shape:
        return False
    sortable = [c for c, t in a.schema.items() if not isinstance(t, (pl.List, pl.Struct, pl.Object))]
    if sortable:
        a, b = a.sort(sortable, nulls_last=True), b.sort(sortable, nulls_last=True)
    return a.equals(b)


@pytest.mark.parametrize("scenario,db_fixture", [("coal", "main"), ("lh2_three_region", "lh2")])
def test_reload_matches_bare_load_flextool(tmp_path, scenario_run, scenario, db_fixture):
    """Every frame field of ``load_from_dump(dump)`` equals that of a bare ``load_flextool(dump)``.

    The two loads differ only in where the process topology comes from (package vs loader CSV branch), so this
    pins ``pss_from_dump`` to the inline branch while both exist.  The dump folder is not a ``work_<scenario>``
    folder and has no ``tests.sqlite``, so the bare load is CSV-only too.
    """
    dump = _dump_of(scenario_run(scenario, db_fixture), tmp_path / "dump")
    with pytest.warns(CsvReloadWarning, match=_UNSUPPORTED):
        reloaded = load_from_dump(dump)
    bare = load_flextool(dump)
    assert reloaded._axis_enums is None and bare._axis_enums is None

    compared = 0
    mismatches: list[str] = []
    for field in dataclasses.fields(FlexData):
        a, b = _frame(getattr(reloaded, field.name, None)), _frame(getattr(bare, field.name, None))
        if a is None and b is None:
            continue
        if a is None or b is None:
            mismatches.append(f"{field.name}: load_from_dump populated={a is not None}, bare={b is not None}")
            continue
        compared += 1
        if not _same_frame(a, b):
            mismatches.append(f"{field.name}: load_from_dump {a.shape} {a.schema}\nbare {b.shape} {b.schema}")
    assert not mismatches, "\n".join(mismatches)
    for field in ("process_source_sink", "process_source_sink_eff"):
        frame = _frame(getattr(reloaded, field))
        assert frame is not None and frame.height > 0, f"{field} must be populated for the parity to bite"
    assert compared > 30  # coal populates 43 frame fields


# ---------------------------------------------------------------------------
# No discovery: no tests.sqlite axis Enums, DB only when named
# ---------------------------------------------------------------------------


def test_work_folder_reload_builds_no_axis_enums(scenario_run):
    """``run.work_folder`` is ``work_coal`` with a ``tests.sqlite``: the bare loader discovers it, the reload not."""
    run = scenario_run("coal")
    assert (run.work_folder / "tests.sqlite").exists()
    assert load_flextool(run.work_folder)._axis_enums is not None  # precondition: discovery would fire

    with pytest.warns(CsvReloadWarning, match=_UNSUPPORTED):
        csv_only, provider = load_from_dump_with_provider(run.work_folder)
    assert csv_only._axis_enums is None and provider.axis_enums is None

    with pytest.warns(CsvReloadWarning, match=_UNSUPPORTED):
        with_db, provider = load_from_dump_with_provider(run.work_folder, db_url=run.db_url,
                                                         scenario=run.scenario)
    assert with_db._axis_enums is None and provider.axis_enums is None
    assert not isinstance(with_db.process_source_sink.schema["p"], pl.Enum)
