"""A run whose input database cannot be run records the reason, and the run header names the resolved output folder.

``main`` is driven in-process with ``run_chain_from_db`` replaced by a stub that constructs the
:class:`FlexToolRunner` the real cascade constructs, against a database stand-in with no scenario: the loader
raises ``FlexToolConfigError("No scenario found")``.
"""
from __future__ import annotations

import contextlib
import os
import sys

import pytest

import flextool.engine
from flextool import run_record
from flextool.cli import cmd_run_flextool
from flextool.engine.input import _db_loader


@pytest.fixture
def restore_env():
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


class _DatabaseWithoutScenarios:
    def get_scenario_items(self):
        return []


def test_a_config_error_is_recorded_with_its_message_and_exits_1(restore_env, monkeypatch, tmp_path, caplog):
    def stub(input_db_url, scenario_name, **kwargs):
        _db_loader.FlexToolRunner(input_db_url=input_db_url, scenario_name=None, work_folder=tmp_path / "work")
        pytest.fail("the loader accepted a database without a scenario")

    monkeypatch.setattr(flextool.engine, "run_chain_from_db", stub)
    monkeypatch.setattr(_db_loader, "open_database_mapping",
                        lambda url: contextlib.nullcontext(_DatabaseWithoutScenarios()))
    db = tmp_path / "input.sqlite"
    db.write_bytes(b"")
    (tmp_path / "work").mkdir()
    # A relative --output-location: the header shows the absolute folder the record and the outputs use.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "cmd_run_flextool", f"sqlite:///{db}", f"sqlite:///{tmp_path / 'run_info.sqlite'}",
        "--scenario-name", "base", "--work-folder", str(tmp_path / "work"), "--output-location", "out",
        "--output-subdir", "base_1",
    ])
    with pytest.raises(SystemExit) as exc:
        cmd_run_flextool.main()
    assert exc.value.code == 1

    folder = tmp_path / "out" / "output_parquet" / "base_1"
    status = run_record.read_status_file(folder)
    assert status["state"] == "failed" and status["exit_code"] == 1
    assert status["error"] == {"type": "FlexToolConfigError", "message": "No scenario found"}
    log = (folder / run_record.RUN_LOG_FILE).read_text(encoding="utf-8").splitlines()
    [output_line] = [line for line in log if line.startswith("Output:")]
    assert output_line.split(":", 1)[1].strip() == str(tmp_path / "out")
    # The reason is logged once, without a traceback (a configuration problem, not a bug).
    errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert errors == ["No scenario found"]
