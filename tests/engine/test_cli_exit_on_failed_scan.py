"""A run whose final optimality scan fails exits 1 and does not write the output-info database.

``main`` is driven in-process with ``run_chain_from_db`` replaced by a stub that returns one Benders step without a
feasible incumbent (``obj=None``): ``_scan_cascade_optimality`` rejects it, so no outputs are written.  The
output-info database named on the command line does not exist beforehand, and only the output-info write would
create it, so its absence shows the write was skipped.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

import flextool.engine
from flextool.cli import cmd_run_flextool


@pytest.fixture
def restore_env():
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def _benders_step_without_incumbent() -> SimpleNamespace:
    return SimpleNamespace(
        optimal=False, near_optimal=False, is_benders=True, obj=None, solve_name="y2030",
        benders_gap=None, benders_tol=1e-4, benders_iterations=50,
        flex_data=None, solution=None, flex_data_provider=None,
    )


def test_failed_scan_exits_1_without_output_info_write(restore_env, monkeypatch, tmp_path):
    def stub(input_db_url, scenario_name, **kwargs):
        return {"y2030": _benders_step_without_incumbent()}

    monkeypatch.setattr(flextool.engine, "run_chain_from_db", stub)
    db = tmp_path / "input.sqlite"
    db.write_bytes(b"")
    info_db = tmp_path / "run_info.sqlite"
    monkeypatch.setattr(sys, "argv", [
        "cmd_run_flextool", f"sqlite:///{db}", f"sqlite:///{info_db}", "--scenario-name", "base",
        "--work-folder", str(tmp_path / "work"), "--output-location", str(tmp_path / "out"),
    ])
    with pytest.raises(SystemExit) as exc:
        cmd_run_flextool.main()
    assert exc.value.code == 1
    assert not info_db.exists()
    assert not list((tmp_path / "out").glob("output_parquet/base/*.parquet"))
