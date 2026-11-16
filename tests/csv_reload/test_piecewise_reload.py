"""A dump of a run with a ``piecewise_linear`` unit cannot be reloaded without the database
(``specs/piecewise_efficiency_final.md``): the part-load curves are read from the database at the load boundary,
so the reader-less CSV reload names the unit instead of loading a model without its curves.

Moved here from ``tests/model/test_piecewise_two_point_parity.py``: only ``tests/csv_reload/`` may import the
unsupported reload package ``flextool.engine._csv_reload``.  The run is built from ``tests/fixtures/tests.json``.
"""
from __future__ import annotations

import shutil

import pytest

from flextool.engine._csv_reload import copy_runner_meta, load_from_dump
from flextool.engine.input._solve_state import FlexToolConfigError
from tests.engine._unit_db import RunCache
from tests.model.test_piecewise_two_point_parity import CASES, TESTS_JSON, _twin_values

pytestmark = pytest.mark.solver

_RUNS = RunCache()


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch) -> None:
    """A missing autoscale registry entry raises instead of falling back to an unscaled LP."""
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def test_reload_without_database_refuses_piecewise_units(tmp_path_factory, tmp_path) -> None:
    """A dump of a run with a piecewise_linear unit cannot be reloaded without the database (the curves are read
    from it at the load boundary): the reader-less reload names the unit."""
    c = CASES["direct_linear"]
    steps, wf, _ = _RUNS.get(tmp_path_factory, alt="t1_direct_linear_pw", values=_twin_values(c, "pw"),
                             base_chain=c["chain"], fixture_json=TESTS_JSON)
    work = tmp_path / "work"
    shutil.copytree(wf, work)  # the cached run's folder is shared: snapshot into a copy
    last = list(steps.values())[-1]
    last.flex_data_provider.snapshot_processed_inputs(work)
    out = tmp_path / "dumped"
    last.flex_data.dump_csvs(out, include_heavy=True)
    copy_runner_meta(work, out)
    with pytest.raises(FlexToolConfigError, match=r"the dump has piecewise_linear unit\(s\) 'coal_plant': their "
                                                  r"part-load curves are read from the database"):
        load_from_dump(out)
