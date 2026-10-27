"""The ``exact`` transfer method's ``process_exact`` set survives a ``--csv-dump`` reload.

Moved from ``tests/engine/test_exact_connection.py`` (c24 S1-8): the CSV
reader (provider-less ``load_flextool``) derives ``process_exact`` from
``input/process_method`` of a snapshot work folder, and reads back the
``solve_data/process_exact.csv`` that ``dump_csvs`` writes.  The writer half
stays there as ``test_process_exact_dump_writes_csv``.
"""
from __future__ import annotations

import pytest

from tests.engine import test_exact_connection as _exact
from tests.engine._unit_db import make_db, utf8

pytestmark = pytest.mark.solver

# The fixture data of the source module (shared, not copied).
LL = _exact.LL
_dissipate = _exact._dissipate


def test_process_exact_csv_round_trip(tmp_path_factory) -> None:
    """Critique 8f: the CSV reader derives process_exact from input/process_method (a --csv-dump work folder), and
    dump_csvs writes solve_data/process_exact.csv, which load_flextool reads back."""
    from flextool.engine import load_flextool, run_chain_from_db

    url, scen = make_db(tmp_path_factory, alt="ex_dump", values=_dissipate("exact"), base_chain=LL)
    wf = tmp_path_factory.mktemp("ex_dump_wf")
    steps = run_chain_from_db(url, scen, work_folder=wf, csv_dump=True, keep_solutions=True)
    list(steps.values())[-1].flex_data_provider.snapshot_processed_inputs(wf)
    original = load_flextool(wf)
    assert set(utf8(original.process_exact)["p"].to_list()) == {"ll"}
    out = tmp_path_factory.mktemp("ex_dump_out")
    original.dump_csvs(out, copy_meta_from=wf, include_heavy=True)
    assert (out / "solve_data" / "process_exact.csv").exists()
    redo = load_flextool(out)
    assert set(utf8(redo.process_exact)["p"].to_list()) == {"ll"}
