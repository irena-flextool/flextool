"""Restoring an empty method parameter default changes no result (the repair tool's part B claim).

The repair tool says that restoring an empty method parameter default "does not change results: FlexTool already
uses these defaults when none is set" (``list_default_restore.block_lines``, ``RESTORE_REASON``).  Each case builds
one fixture twice: the database as shipped, and the same database with the method parameter defaults of
``list_default_restore.reference()`` emptied (``NULL``, the state the repair finds).  Both are solved and every
solve's objective, column names and column values must be equal (twin == case; one process, so exactly equal).
"""
from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from flextool.update_flextool import list_default_restore as ldr
from tests.update_flextool._value_list_cases import clear_defaults

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
ALL = sorted(".".join(k) for k in ldr.reference())


def _build(tmp_path: Path, fixture: str, stem: str, emptied) -> str:
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    path = tmp_path / f"{stem}.sqlite"
    url = json_to_db(FIXTURES / fixture, path)
    migrate_database(url)
    if emptied:
        clear_defaults(path, emptied, None)
        with contextlib.closing(sqlite3.connect(path)) as con:
            assert ldr.empty_shipped_defaults(con) == sorted(emptied)  # the empty set the repair restores
    return url


def _run(tmp_path_factory, url: str, scenario: str) -> dict:
    from flextool.engine import run_chain_from_db

    steps = run_chain_from_db(url, scenario, work_folder=tmp_path_factory.mktemp("work"), keep_solutions=True)
    assert steps and all(s.solution is not None for s in steps.values())
    return steps


def _assert_same_results(shipped: dict, emptied: dict) -> None:
    assert list(shipped) == list(emptied)
    for name, a in shipped.items():
        b = emptied[name]
        assert a.solution.obj == b.solution.obj, name
        assert a.solution.col_names == b.solution.col_names, name
        assert np.array_equal(a.solution.col_value, b.solution.col_value), name


@pytest.mark.solver
@pytest.mark.parametrize(("fixture", "scenario", "emptied"), [
    # groups invest_stochastic (node town) and rolling_stochastic have no include_stochastics in this scenario;
    # the storage nodes rely on the storage method defaults; every method parameter default emptied
    ("stochastics.json", "2_day_stochastic_dispatch", ALL),
    # DC power flow: nodes commodity_gen1/2 have no node_type of their own.  node.node_type stays set here: an
    # empty node_type default drops such nodes from the energy balance
    ("case14_dc_power_flow.json", "dc_opf_test", [n for n in ALL if n != "node.node_type"]),
], ids=["stochastic", "dc_power_flow"])
def test_emptied_method_parameter_defaults_give_the_shipped_results(tmp_path, tmp_path_factory, fixture, scenario,
                                                                    emptied) -> None:
    shipped = _run(tmp_path_factory, _build(tmp_path, fixture, "shipped", ()), scenario)
    _assert_same_results(shipped, _run(tmp_path_factory, _build(tmp_path, fixture, "emptied", emptied), scenario))
