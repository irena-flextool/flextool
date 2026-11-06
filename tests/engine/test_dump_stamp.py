"""The CSV dump format stamp ``_dump_meta.json`` (``flextool/engine/preprocess/_dump_stamp.py``).

Both supported dump writers stamp the folder they write: ``FlexData.dump_csvs`` (through
``output/_dump_csvs.dump_csvs``) and ``FlexDataProvider.snapshot_processed_inputs``.  The region-input CLI export
(``write_input_for_region``) is not a dump and is not stamped.  Expected values come from the running engine
(``flextool_version()``, ``FLEXTOOL_DB_VERSION``, ``DUMP_FORMAT``), not from the writer's own output.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import polars as pl
import pytest

from flextool.engine.preprocess import _dump_stamp
from flextool.engine.preprocess._dump_stamp import DUMP_FORMAT, STAMP_FILE, read_dump_stamp, write_dump_stamp
from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool.install_info import flextool_version

TEST_DIR = Path(__file__).resolve().parents[1]
if str(TEST_DIR / "fixtures") not in sys.path:
    sys.path.insert(0, str(TEST_DIR / "fixtures"))

_IDENTITY = ("dump_format", "flextool", "schema_version", "git")


def _small_provider() -> FlexDataProvider:
    p = FlexDataProvider()
    p.put("input/node", pl.DataFrame({"node": ["n1", "n2"]}))
    p.put("solve_data/solve_current", pl.DataFrame({"solve": ["s1"]}))
    return p


def _raw_stamp(folder: Path) -> dict:
    return json.loads((folder / STAMP_FILE).read_text(encoding="utf-8"))


def _assert_engine_identity(stamp: dict) -> None:
    assert stamp["dump_format"] == DUMP_FORMAT
    assert isinstance(stamp["dump_format"], int)
    assert stamp["flextool"] == flextool_version()
    assert stamp["schema_version"] == FLEXTOOL_DB_VERSION
    assert isinstance(stamp["schema_version"], int)
    # A git worktree (``.git`` is a file) or a wheel install has no checkout root: git is null there.
    assert stamp["git"] is None or (isinstance(stamp["git"], str) and stamp["git"])


@pytest.fixture
def base_run(scenario_run):
    return scenario_run("base")


def test_snapshot_processed_inputs_stamps_the_folder(tmp_path: Path) -> None:
    _small_provider().snapshot_processed_inputs(tmp_path)
    assert (tmp_path / "input" / "node.csv").is_file()
    stamp = _raw_stamp(tmp_path)
    assert set(stamp) == {*_IDENTITY, "writers"}
    _assert_engine_identity(stamp)
    assert stamp["writers"] == ["snapshot_processed_inputs"]
    assert read_dump_stamp(tmp_path) == stamp


def test_stamp_file_is_deterministic_json(tmp_path: Path) -> None:
    """``indent=1, sort_keys=True``, no timestamp: rewriting the same dump gives the same bytes."""
    p = _small_provider()
    p.snapshot_processed_inputs(tmp_path)
    first = (tmp_path / STAMP_FILE).read_bytes()
    p.snapshot_processed_inputs(tmp_path)
    assert (tmp_path / STAMP_FILE).read_bytes() == first
    assert first.decode("utf-8") == json.dumps(_raw_stamp(tmp_path), indent=1, sort_keys=True) + "\n"
    assert not (tmp_path / (STAMP_FILE + ".tmp")).exists()


def test_dump_csvs_stamps_the_folder(base_run, tmp_path: Path) -> None:
    out = tmp_path / "dump"
    returned = base_run.last_flex_data.dump_csvs(out)
    assert Path(returned) == out
    assert any((out / "solve_data").glob("*.csv"))
    stamp = _raw_stamp(out)
    assert set(stamp) == {*_IDENTITY, "writers"}
    _assert_engine_identity(stamp)
    assert stamp["writers"] == ["dump_csvs"]


def test_scenario_run_work_folder_is_a_stamped_dump(base_run) -> None:
    """``run_scenario`` snapshots the last provider, so every ``ScenarioRun.work_folder`` carries a stamp."""
    stamp = read_dump_stamp(base_run.work_folder)
    assert stamp is not None
    _assert_engine_identity(stamp)
    assert stamp["writers"] == ["snapshot_processed_inputs"]


def test_same_engine_merge_unions_writers(base_run, tmp_path: Path) -> None:
    _small_provider().snapshot_processed_inputs(tmp_path)
    base_run.last_flex_data.dump_csvs(tmp_path)
    stamp = _raw_stamp(tmp_path)
    assert stamp["writers"] == ["dump_csvs", "snapshot_processed_inputs"]
    assert "previous" not in stamp
    _assert_engine_identity(stamp)
    # Writing again with a writer already listed changes nothing.
    _small_provider().snapshot_processed_inputs(tmp_path)
    assert _raw_stamp(tmp_path) == stamp


@pytest.mark.parametrize("field, other", [
    ("flextool", "0.0.1"),
    ("schema_version", 1),
    ("dump_format", 0),
    ("git", "deadbeef"),
])
def test_stamp_of_another_engine_is_kept_under_previous(tmp_path: Path, field: str, other) -> None:
    write_dump_stamp(tmp_path, writer="dump_csvs")
    old = _raw_stamp(tmp_path)
    old[field] = other
    old["previous"] = {"dump_format": 1, "flextool": "0.0.0", "schema_version": 0, "git": None,
                       "writers": ["dump_csvs"]}
    (tmp_path / STAMP_FILE).write_text(json.dumps(old), encoding="utf-8")

    _small_provider().snapshot_processed_inputs(tmp_path)

    stamp = _raw_stamp(tmp_path)
    _assert_engine_identity(stamp)
    assert stamp["writers"] == ["snapshot_processed_inputs"]
    assert stamp["previous"] == {k: v for k, v in old.items() if k != "previous"}
    assert stamp["previous"][field] == other


def test_same_engine_merge_keeps_previous(tmp_path: Path) -> None:
    """A folder that once held another engine's dump stays marked as mixed after later same-engine writes."""
    write_dump_stamp(tmp_path, writer="dump_csvs")
    stamp = _raw_stamp(tmp_path)
    earlier = {"dump_format": 1, "flextool": "0.0.1", "schema_version": 1, "git": None, "writers": ["dump_csvs"]}
    stamp["previous"] = earlier
    (tmp_path / STAMP_FILE).write_text(json.dumps(stamp), encoding="utf-8")

    _small_provider().snapshot_processed_inputs(tmp_path)

    merged = _raw_stamp(tmp_path)
    assert merged["writers"] == ["dump_csvs", "snapshot_processed_inputs"]
    assert merged["previous"] == earlier


def test_unparsable_existing_stamp_is_overwritten(tmp_path: Path) -> None:
    (tmp_path / STAMP_FILE).write_text("{not json", encoding="utf-8")
    _small_provider().snapshot_processed_inputs(tmp_path)
    stamp = _raw_stamp(tmp_path)
    _assert_engine_identity(stamp)
    assert stamp["writers"] == ["snapshot_processed_inputs"]
    assert "previous" not in stamp


def _raise_on_write(path: Path, text: str) -> None:
    raise OSError(f"simulated write failure for {path}")


def test_snapshot_stamp_write_failure_only_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(_dump_stamp, "_write_stamp_file", _raise_on_write)
    with caplog.at_level(logging.WARNING, logger=_dump_stamp.__name__):
        result = _small_provider().snapshot_processed_inputs(tmp_path)
    assert result is None
    assert (tmp_path / "input" / "node.csv").is_file()
    assert (tmp_path / "solve_data" / "solve_current.csv").is_file()
    assert not (tmp_path / STAMP_FILE).exists()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and r.name == _dump_stamp.__name__]
    assert len(warnings) == 1
    assert "simulated write failure" in warnings[0].getMessage()
    assert "snapshot_processed_inputs" in warnings[0].getMessage()


def test_dump_csvs_stamp_write_failure_only_logs(
    base_run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(_dump_stamp, "_write_stamp_file", _raise_on_write)
    out = tmp_path / "dump"
    with caplog.at_level(logging.WARNING, logger=_dump_stamp.__name__):
        returned = base_run.last_flex_data.dump_csvs(out)
    assert Path(returned) == out
    assert any((out / "solve_data").glob("*.csv"))
    assert not (out / STAMP_FILE).exists()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and r.name == _dump_stamp.__name__]
    assert len(warnings) == 1
    assert "dump_csvs" in warnings[0].getMessage()


def test_identity_failure_only_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """Any error while building the stamp (not only the file write) is logged, never raised."""
    def _boom() -> dict:
        raise RuntimeError("simulated identity failure")

    monkeypatch.setattr(_dump_stamp, "_engine_identity", _boom)
    with caplog.at_level(logging.WARNING, logger=_dump_stamp.__name__):
        path = write_dump_stamp(tmp_path, writer="dump_csvs")
    assert path == tmp_path / STAMP_FILE
    assert not path.exists()
    assert any("simulated identity failure" in r.getMessage() for r in caplog.records)


def test_engine_identity_is_computed_once_per_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The git subprocess runs once per process, not once per writer call."""
    from flextool.update_flextool import install_info

    calls: list[object] = []

    def _counting_commit(root):
        calls.append(root)
        return "abcdef12"

    monkeypatch.setattr(install_info, "current_commit", _counting_commit)
    monkeypatch.setattr(install_info, "git_checkout_root", lambda: tmp_path)
    _dump_stamp._engine_identity.cache_clear()
    try:
        for i in range(3):
            (tmp_path / f"d{i}").mkdir()
            write_dump_stamp(tmp_path / f"d{i}", writer="dump_csvs")
        assert calls == [tmp_path]
        assert all(_raw_stamp(tmp_path / f"d{i}")["git"] == "abcdef12" for i in range(3))
    finally:
        _dump_stamp._engine_identity.cache_clear()


def test_snapshot_without_stamp(tmp_path: Path) -> None:
    _small_provider().snapshot_processed_inputs(tmp_path, stamp=False)
    assert (tmp_path / "input" / "node.csv").is_file()
    assert not (tmp_path / STAMP_FILE).exists()


def test_region_export_has_no_stamp(lh2_db_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``write_input_for_region`` writes a CLI export (and one manifest into the work folder), not a dump."""
    from build_lh2_three_region import SCENARIO

    from flextool.decomposition.region_decomposition import write_input_for_region

    monkeypatch.chdir(tmp_path)
    out = tmp_path / "input_region_region_B"
    write_input_for_region(
        input_db_url=lh2_db_url,
        scenario_name=SCENARIO,
        logger=logging.getLogger("test_dump_stamp"),
        region_group="region_B",
        output_dir=out,
        work_folder=tmp_path,
    )
    assert any(out.glob("*.csv"))
    assert (tmp_path / "solve_data" / "region_coupling.csv").is_file()
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob(STAMP_FILE)) == []


def test_read_dump_stamp_none_on_absent_or_garbage(tmp_path: Path) -> None:
    assert read_dump_stamp(tmp_path) is None
    assert read_dump_stamp(tmp_path / "missing_folder") is None
    for text in ("{not json", "[1, 2, 3]", "\"a string\"", "", "\xff\xfe"):
        (tmp_path / STAMP_FILE).write_text(text, encoding="latin-1")
        assert read_dump_stamp(tmp_path) is None, text
