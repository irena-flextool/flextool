"""The ``--csv-dump`` reload keeps the real (un-margined) demand of the group rows.

Moved from ``tests/engine/test_group_limits_real_demand.py`` (c24 S1-8): the
FlexData reloaded from a ``dump_csvs`` tree with
``_csv_reload.load_from_dump(out, db_url=…, scenario=…)`` has the same real inflow and builds the same
non-sync and capacity-margin rows as the cascade's.  The writer half
(``dump_csvs`` writes ``pdtNodeInflow_unmargined.csv``) stays there as
``test_csv_dump_writes_real_demand``.
"""
from __future__ import annotations

import pytest

from tests.engine import test_group_limits_real_demand as _gl
from tests.engine._unit_db import make_db, utf8

# The helpers of the source module (shared, not copied).
GROUP_FAMILIES = _gl.GROUP_FAMILIES
INVEST_CHAIN = _gl.INVEST_CHAIN
_group = _gl._group
_multiplier = _gl._multiplier
_rows = _gl._rows


def test_csv_dump_keeps_real_demand(tmp_path_factory) -> None:
    """With a margin, ``dump_csvs`` writes ``pdtNodeInflow_unmargined.csv``: the FlexData reloaded from the
    dump (with the DB reader, which runs the derived passes) has the same real inflow and builds the same
    non-sync and capacity-margin rows as the original."""
    from flextool.engine import run_chain_from_db
    from flextool.engine._csv_reload import copy_runner_meta, load_from_dump

    ents, vals = _group("gm", ["west"], flagged=[("wind_plant", "west")])
    url, scen = make_db(tmp_path_factory, alt="rd_dump", values=vals + _multiplier("west", 1.1),
                        base_chain=INVEST_CHAIN, entities=ents)
    wf = tmp_path_factory.mktemp("rd_dump_w")
    steps = run_chain_from_db(url, scen, work_folder=wf, csv_dump=True, keep_solutions=True)
    step = list(steps.values())[-1]
    step.flex_data_provider.snapshot_processed_inputs(wf)
    original = step.flex_data
    assert original.p_inflow_unmargined is not None
    out = tmp_path_factory.mktemp("rd_dumped")
    original.dump_csvs(out, include_heavy=True)      # the FlexData fields only
    assert (out / "solve_data" / "pdtNodeInflow_unmargined.csv").exists(), "dump_csvs must write the real inflow"
    copy_runner_meta(wf, out)                           # + the runner metadata the CSV reader needs
    redo = load_from_dump(out, db_url=url, scenario=scen)
    assert redo.p_inflow_unmargined is not None, "pdtNodeInflow_unmargined.csv not dumped"
    key = ["n", "d", "t"]
    assert utf8(redo.p_inflow_unmargined.frame).sort(key).equals(utf8(original.p_inflow_unmargined.frame).sort(key))
    a, b = _rows(redo), _rows(original)
    for fam in GROUP_FAMILIES:
        assert a[fam]["bounds"], fam
        assert a[fam]["bounds"].keys() == b[fam]["bounds"].keys(), fam
        for row, (lo, hi) in a[fam]["bounds"].items():
            assert (lo, hi) == pytest.approx(b[fam]["bounds"][row], rel=1e-12, abs=1e-9), row
