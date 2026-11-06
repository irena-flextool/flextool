"""A run reads only its in-memory provider, never CSV files left in a reused work folder (item 160).

A ``--csv-dump`` (or any earlier run) leaves ``input/*.csv`` and ``solve_data/*.csv`` in the work folder. The
loader's provider helpers in ``preprocess/flex_data.py`` used to fall back to such a file whenever the provider
lacked the key, so a stale ``solve_data/process_exact.csv`` from an ``exact``-transfer model turned a
``regular`` connection of the next model into an exact one (extra ``v_exact_forward`` columns and
``maxFlow_exact_*`` rows).

* ``test_stale_dump_does_not_change_the_model``: model A (``west_east`` transfer_method ``exact``) is run and
  dumped into W; model B (the same connection with linear startup, the regular method) is run in W and in a
  fresh folder F. Both B runs must build the same LP.
* The helper tests pin the provider-only contract directly: a CSV at the path is ignored when the provider
  lacks the key, and ``provider=None`` raises.
* A database file in the work folder is ignored (item 166). ``load_flextool`` used to build the axis-Enum
  vocabulary from a ``<work folder>/tests.sqlite``, which switched the run to Enum dtypes: a reordered LP, and a
  ``SchemaError`` crash in ``rolling_separate_dispatch``. ``test_tests_sqlite_in_work_folder_is_ignored`` (unit DB)
  and ``test_tests_sqlite_does_not_crash_rolling_separate_dispatch`` run the model with its own DB copied there and
  compare with a fresh folder; ``test_load_flextool_requires_provider_and_reader`` pins that the loader has no
  discovery mode left.

Databases are built from JSON: ``tests/fixtures/tests.json`` (``tests/engine/_unit_db.make_db``) and
``tests/fixtures/rolling_separate_dispatch.json``.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl
import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
from flextool.engine.preprocess.emit._blocks import _read_input_rows
from flextool.engine.preprocess.flex_data import (
    _provider_has,
    _provider_pick,
    _provider_read,
    _read_process_exact,
    load_flextool,
)
from tests._support.load import sqlite_path_from_url
from tests.engine._unit_db import b64, make_db

NET_CHAIN = ["init", "west", "coal", "wind", "network"]
CONN = "west_east"


def _run_steps(url: str, scenario: str, work, **kwargs) -> list:
    from flextool.engine import run_chain_from_db

    steps = run_chain_from_db(url, scenario, work_folder=work, keep_solutions=True, **kwargs)
    assert steps, "run_chain_from_db produced no orchestration steps"
    for step in steps.values():
        assert step.solution is not None, "solve produced no solution"
    return list(steps.values())


def _run(url: str, scenario: str, work):
    return _run_steps(url, scenario, work)[-1]


def _var_names(step) -> set[str]:
    return set(step.solution._vars)


def test_stale_dump_does_not_change_the_model(tmp_path_factory: pytest.TempPathFactory) -> None:
    url_a, scen_a = make_db(
        tmp_path_factory, alt="iso_exact", base_chain=NET_CHAIN,
        values=[["connection", CONN, "transfer_method", b64("exact", "str")]],
    )
    url_b, scen_b = make_db(
        tmp_path_factory, alt="iso_regular", base_chain=NET_CHAIN,
        values=[["connection", CONN, "startup_method", b64("linear", "str")],
                ["connection", CONN, "startup_cost", b64(3.0, "float")]],
    )
    reused = tmp_path_factory.mktemp("reused_work")
    fresh = tmp_path_factory.mktemp("fresh_work")

    a = _run(url_a, scen_a, reused)
    # Positive control: model A really is an exact-transfer model.
    assert a.flex_data.process_exact is not None
    assert "v_exact_forward" in _var_names(a)
    a.flex_data.dump_csvs(reused)
    # Precondition: the dirt the old disk fallback picked up is on disk.
    assert (reused / "solve_data" / "process_exact.csv").exists()

    b_fresh = _run(url_b, scen_b, fresh)
    b_reused = _run(url_b, scen_b, reused)

    assert b_fresh.flex_data.process_exact is None
    assert b_reused.flex_data.process_exact is None
    assert "v_exact_forward" not in _var_names(b_reused)
    assert _var_names(b_reused) == _var_names(b_fresh)
    assert sorted(b_reused.solution.col_names) == sorted(b_fresh.solution.col_names)
    assert sorted(b_reused.solution.row_names) == sorted(b_fresh.solution.row_names)
    assert b_reused.obj == pytest.approx(b_fresh.obj, rel=1e-12)


# ---------------------------------------------------------------------------
# Helper contract: a CSV at the path is ignored when the provider lacks the key.

KEY = "solve_data/stale_frame"


@pytest.fixture
def stale_csv(tmp_path):
    path = tmp_path / "solve_data" / "stale_frame.csv"
    path.parent.mkdir()
    pl.DataFrame({"process": ["p1"], "value": [1.0]}).write_csv(path)
    return path


def test_provider_has_ignores_csv_on_disk(stale_csv) -> None:
    assert _provider_has(FlexDataProvider(), KEY, stale_csv) is False


def test_provider_read_ignores_csv_on_disk(stale_csv) -> None:
    with pytest.raises(FileNotFoundError, match="stale_frame"):
        _provider_read(FlexDataProvider(), KEY, stale_csv)


def test_provider_pick_ignores_csv_on_disk(stale_csv) -> None:
    assert _provider_pick(FlexDataProvider(), (KEY, stale_csv)) is None


def test_provider_helpers_serve_the_provider_frame(stale_csv) -> None:
    frame = pl.DataFrame({"process": ["p2"], "value": [2.0]})
    provider = FlexDataProvider()
    provider.put(KEY, frame)
    assert _provider_has(provider, KEY, stale_csv) is True
    assert _provider_read(provider, KEY, stale_csv).equals(frame)
    assert _provider_pick(provider, ("solve_data/absent", stale_csv.parent / "absent.csv"),
                          (KEY, stale_csv)) == stale_csv


@pytest.mark.parametrize("call", [
    lambda path: _provider_has(None, KEY, path),
    lambda path: _provider_read(None, KEY, path),
    lambda path: _provider_pick(None, (KEY, path)),
], ids=["has", "read", "pick"])
def test_provider_helpers_require_a_provider(stale_csv, call) -> None:
    with pytest.raises(ValueError, match="requires a FlexDataProvider"):
        call(stale_csv)


def test_process_exact_ignores_a_dumped_process_exact_csv(tmp_path) -> None:
    """The dump's ``solve_data/process_exact.csv`` no longer wins over ``input/process_method``."""
    sd = tmp_path / "solve_data"
    sd.mkdir()
    pl.DataFrame({"process": ["stale_conn"]}).write_csv(sd / "process_exact.csv")
    provider = FlexDataProvider()
    provider.put("input/process_method", pl.DataFrame({
        "process": ["conn_regular"], "method": ["method_2way_1var_off"],
    }))
    assert _read_process_exact(tmp_path / "input", provider=provider) is None
    provider.put("input/process_method", pl.DataFrame({
        "process": ["conn_exact"], "method": ["method_2way_2var_MIP_exclude"],
    }))
    got = _read_process_exact(tmp_path / "input", provider=provider)
    assert got is not None and got["p"].cast(pl.Utf8).to_list() == ["conn_exact"]


def test_block_input_rows_ignore_csv_on_disk(tmp_path) -> None:
    inp = tmp_path / "input"
    inp.mkdir()
    pl.DataFrame({"node": ["stale_node"]}).write_csv(inp / "node.csv")
    assert _read_input_rows(inp, "node.csv", FlexDataProvider()) == []
    provider = FlexDataProvider()
    provider.put("input/node", pl.DataFrame({"node": ["live_node"]}))
    assert _read_input_rows(inp, "node.csv", provider) == [["live_node"]]
    with pytest.raises(ValueError, match="requires a FlexDataProvider"):
        _read_input_rows(inp, "node.csv", None)


# ---------------------------------------------------------------------------
# Item 166: a database file in the work folder is ignored.


def _assert_same_lp(dirty: list, fresh: list) -> None:
    """Same steps, no axis Enums, and per step the same LP (column and row names in order) and objective."""
    assert [s.solve_name for s in dirty] == [s.solve_name for s in fresh]
    for d, f in zip(dirty, fresh):
        assert f.flex_data._axis_enums is None, f"{f.solve_name}: a fresh run builds no axis Enums"
        assert d.flex_data._axis_enums is None, f"{d.solve_name}: tests.sqlite in the work folder built axis Enums"
        assert d.solution.col_names == f.solution.col_names, f"{d.solve_name}: column order differs"
        assert d.solution.row_names == f.solution.row_names, f"{d.solve_name}: row order differs"
        assert d.obj == pytest.approx(f.obj, rel=1e-12), d.solve_name


def test_tests_sqlite_in_work_folder_is_ignored(tmp_path_factory: pytest.TempPathFactory) -> None:
    url, scen = make_db(
        tmp_path_factory, alt="iso_sqlite", base_chain=NET_CHAIN,
        values=[["connection", CONN, "startup_method", b64("linear", "str")],
                ["connection", CONN, "startup_cost", b64(3.0, "float")]],
    )
    dirty = tmp_path_factory.mktemp("sqlite_work")
    shutil.copy(sqlite_path_from_url(url), dirty / "tests.sqlite")
    fresh = tmp_path_factory.mktemp("fresh_work")

    _assert_same_lp(_run_steps(url, scen, dirty), _run_steps(url, scen, fresh))


ROLLING_SEPARATE_DISPATCH = Path(__file__).resolve().parents[1] / "fixtures" / "rolling_separate_dispatch.json"


def test_tests_sqlite_does_not_crash_rolling_separate_dispatch(tmp_path_factory: pytest.TempPathFactory) -> None:
    """The Enum switch made this model crash (``SchemaError: join keys b: enum vs enum``); now it runs as fresh."""
    from db_utils import json_to_db
    from flextool.update_flextool.db_migration import migrate_database

    db_dir = tmp_path_factory.mktemp("rsd_db")
    url = json_to_db(ROLLING_SEPARATE_DISPATCH, db_dir / "rolling_separate_dispatch.sqlite")
    migrate_database(url)
    dirty = tmp_path_factory.mktemp("rsd_sqlite_work")
    shutil.copy(sqlite_path_from_url(url), dirty / "tests.sqlite")
    fresh = tmp_path_factory.mktemp("rsd_fresh_work")

    scenario = "rolling_separate_dispatch"
    _assert_same_lp(_run_steps(url, scenario, dirty, warm=False), _run_steps(url, scenario, fresh, warm=False))


def test_load_flextool_requires_provider_and_reader(tmp_path) -> None:
    """No provider: the error points to the CSV reload package; a provider but no reader: it names ``db_reader``."""
    with pytest.raises(FlexToolConfigError, match=r"FlexDataProvider.*_csv_reload"):
        load_flextool(tmp_path)
    with pytest.raises(FlexToolConfigError, match=r"needs db_reader .*_csv_reload"):
        load_flextool(tmp_path, provider=FlexDataProvider())
