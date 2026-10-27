"""T12 of ``tests/engine/test_derived_branch_lineage.py``: stochastic dump → CSV reload.

Moved from ``tests/engine/test_derived_branch_lineage.py`` (c24 S1-8).  The
writer half (the dump keeps every lineage rebuild input) stays there as
``test_t12_stoch_dump_keeps_lineage_inputs``, built from the cascade's own
FlexData; this copy keeps the whole original test, whose reload half is a
``_csv_reload.load_from_dump`` of the dumped tree (after
``copy_runner_meta`` adds the run's metadata CSVs).
"""
from __future__ import annotations

import polars as pl

from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
from flextool.engine.preprocess.params._branch import (
    dd_same_scenario_df,
    pd_non_anticipativity_df,
)
from tests.engine import test_derived_branch_lineage as _lineage

# The module fixtures and helper of the source module (shared, not copied).
_rows = _lineage._rows
stoch_run = _lineage.stoch_run
stoch_wf = _lineage.stoch_wf
stoch_data = _lineage.stoch_data


def test_t12_stoch_dump_reload_roundtrip(stoch_data, stoch_wf, tmp_path):
    """T12 (F2, r3 — regression pin of the ``copy_runner_meta`` glob):
    dump the stochastic FlexData, copy the run's metadata with
    ``copy_runner_meta``, assert ``solve_branch__time_branch.csv`` is in
    the dumped tree (glob copy), reload without a raise, and assert the lineage frames
    REBUILD identically from the dumped workdir's CSVs (no §3.1.1
    inconsistency raise).  First stochastic dump/reload round-trip in
    the suite — existing round-trip tests cover deterministic fixtures
    only.

    Path note (design §5 implementation note): a CSV-only reload
    (dumped dir carries no Spine DB) skips the ENTIRE derived cascade
    (``_apply_db_overrides`` gates on ``db_reader is not None``,
    ``flex_data.py:4679``), so every cascade-produced field — the lineage
    frames, like their cousins ``period_branch_full`` /
    ``dt_non_anticipativity`` — is None after ``load_from_dump`` of a
    dumped tree.  The F2 substance ("the rebuild inputs all survive a
    dump") is therefore pinned by running the builders directly
    against the dumped CSVs."""
    from flextool.engine._csv_reload import copy_runner_meta, load_from_dump

    out_dir = tmp_path / "dumped"
    stoch_data.dump_csvs(out_dir)
    copy_runner_meta(stoch_wf, out_dir)
    assert (out_dir / "solve_data"
            / "solve_branch__time_branch.csv").exists(), (
        "copy_runner_meta glob no longer copies solve_branch__time_branch."
        "csv — the always-on lineage builders would raise when the "
        "cascade runs against the dumped tree"
    )
    # Reload succeeds — no raise introduced by Slice A anywhere on the
    # CSV-only load path.
    redo = load_from_dump(out_dir)
    # Current-architecture pin: CSV-only reloads skip the derived
    # cascade, so cascade-produced fields stay None (same as
    # period_branch_full).  If this flips, the cascade now runs on
    # reload — re-point the equality leg below at ``redo`` directly.
    assert redo.dd_same_scenario is None
    assert redo.period_branch_full is None
    # F2 substance: the builders, fed the DUMPED workdir's CSVs,
    # rebuild both frames identically — i.e. every rebuild input
    # survived the dump (period__branch.csv, period_in_use_set.csv,
    # solve_branch__time_branch.csv).
    prov = FlexDataProvider()
    for stem in ("period__branch", "period_in_use_set",
                 "solve_branch__time_branch"):
        p = out_dir / "solve_data" / f"{stem}.csv"
        assert p.exists(), f"rebuild input {stem}.csv missing from dump"
        prov.put(f"solve_data/{stem}", pl.read_csv(p))
    dd_rebuilt = dd_same_scenario_df(out_dir, provider=prov)
    pd_rebuilt = pd_non_anticipativity_df(out_dir, provider=prov)
    assert dd_rebuilt.columns == ["d", "d_other"]
    assert _rows(dd_rebuilt) == _rows(stoch_data.dd_same_scenario)
    assert pd_rebuilt.columns == ["d", "b"]
    assert _rows(pd_rebuilt) == _rows(stoch_data.pd_non_anticipativity)
