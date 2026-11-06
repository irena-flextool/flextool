"""``FlexData.dump_csvs`` writer checks on the solved FlexData.

Each test dumps the FlexData the cascade built for the last sub-solve
(``run.last_flex_data``) and asserts the files written.  The dump → reload
round-trip tests are in ``tests/csv_reload/test_dump_csvs_roundtrip.py``
(c24 S1-8).
"""
from __future__ import annotations

import pandas as pd


def test_dump_csvs_creates_expected_layout(tmp_path, scenario_run):
    """``dump_csvs`` creates ``input/`` + ``solve_data/`` subdirs."""
    run = scenario_run("coal")
    out_dir = tmp_path / "dumped_layout"
    run.last_flex_data.dump_csvs(out_dir)
    assert (out_dir / "input").is_dir()
    assert (out_dir / "solve_data").is_dir()
    # At least the foundational time-keeping CSV is always written.
    assert (out_dir / "solve_data" / "steps_in_use.csv").exists()


def test_dump_csvs_writes_input_entity_set_csvs(tmp_path, scenario_run):
    """Δ.30 — ``dump_csvs`` produces the four entity-class set CSVs and
    the wide-format unitsize CSV that handoff_writers consume.

    Every fast-path workdir must carry these for
    ``handoff_writers._load_entity_class_set`` and
    ``handoff_writers._load_unitsize`` to succeed.
    """
    run = scenario_run("base")
    out_dir = tmp_path / "dumped_input_sets"
    run.last_flex_data.dump_csvs(out_dir)

    # Single-column entity-class set CSVs.
    expected_files = ("entity.csv", "node.csv", "process_unit.csv",
                       "process_connection.csv")
    for name in expected_files:
        path = out_dir / "input" / name
        assert path.exists(), f"missing input/{name}"

    # node.csv contains the nodeBalance set.
    node_df = pd.read_csv(out_dir / "input" / "node.csv")
    assert "node" in node_df.columns
    assert "west" in set(node_df["node"].astype(str)), (
        f"node.csv missing 'west': {node_df}"
    )

    # entity.csv = nodes ∪ processes.
    entity_df = pd.read_csv(out_dir / "input" / "entity.csv")
    assert "entity" in entity_df.columns
    assert "west" in set(entity_df["entity"].astype(str))

    # p_entity_unitsize.csv: wide-transposed (entity row + value row).
    unitsize_path = out_dir / "input" / "p_entity_unitsize.csv"
    assert unitsize_path.exists()
    df = pd.read_csv(unitsize_path, index_col=0)
    assert "value" in df.index, f"value row missing: {df}"
    # base has no explicit p_unitsize / p_state_unitsize → default 1000.
    assert float(df.loc["value", "west"]) == 1000.0


def test_dump_csvs_writes_solve_data_pre_existing_stub(tmp_path, scenario_run):
    """Δ.30 — ``dump_csvs`` writes a header-only stub for
    ``solve_data/solve__p_entity_pre_existing.csv`` so
    ``handoff_writers.write_p_entity_period_existing_capacity`` reads
    an empty frame instead of crashing on the fast path.
    """
    run = scenario_run("base")
    out_dir = tmp_path / "dumped_pre_existing"
    run.last_flex_data.dump_csvs(out_dir)

    stub = out_dir / "solve_data" / "solve__p_entity_pre_existing.csv"
    assert stub.exists()
    # Empty (0, 0) when read with index_col=[0, 1].
    df = pd.read_csv(stub, index_col=[0, 1])
    assert df.empty
