"""A reader-less reload of a dump keeps the start-up input draws (``startup_input_method``, database version 79).

``dump_csvs`` writes ``p_startup_energy`` (``solve_data/process_node_startup_energy.csv``); the reload reads it in
``flex_data._load_startup_inputs``, so the reloaded FlexData has the start-up nodes, energies and derived sets of the
run.  Model: the coal-on-oil model of ``tests/engine/test_startup_input.py`` (built from ``tests.json``).
"""
from __future__ import annotations

import polars as pl
import pytest

from flextool.engine._csv_reload import copy_runner_meta, load_from_dump
from tests.engine._unit_db import make_db, utf8
from tests.engine.test_startup_input import _entities, _values

pytestmark = pytest.mark.solver


def _rows(value) -> list:
    frame = value.frame if hasattr(value, "frame") else value
    return sorted(utf8(frame).rows())


def test_reload_keeps_the_start_up_input(tmp_path_factory) -> None:
    url, scen = make_db(tmp_path_factory, alt="su_reload", values=_values(), base_chain=["init"],
                        entities=_entities())
    wf = tmp_path_factory.mktemp("su_reload_w")
    from flextool.engine import run_chain_from_db

    steps = run_chain_from_db(url, scen, work_folder=wf, keep_solutions=True)
    last = list(steps.values())[-1]
    original = last.flex_data
    # The run's provider snapshot (input/ + solve_data/ CSVs), as tests._support.load.run_scenario writes it.
    last.flex_data_provider.snapshot_processed_inputs(wf)
    out = tmp_path_factory.mktemp("su_reload_dump")
    original.dump_csvs(out, include_heavy=True)
    rows = pl.read_csv(out / "solve_data" / "process_node_startup_energy.csv")
    assert rows.select("process", "node", "period", "value").rows() == [("cp", "oil_n", "p2020", 2.0)]
    copy_runner_meta(wf, out)
    redo = load_from_dump(out)
    for field in ("process_startup_node", "p_startup_energy", "startup_from_commodity"):
        assert getattr(redo, field) is not None, field
        assert _rows(getattr(redo, field)) == _rows(getattr(original, field)), field
    assert _rows(redo.process_startup_node) == [("cp", "oil_n")]
