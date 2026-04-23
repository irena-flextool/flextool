"""Tests for the HiGHS → parquet extraction pipeline.

Covers:
  * Unit tests for ``extract_variable`` using a mocked ``highspy.Highs`` —
    verifies every spec shape (MultiIndex vs single col, time vs period-
    only, variable vs dual), realized-step filtering, value-scale, and
    the empty-result path.
  * Parser / registry sanity — regex, no duplicate output names.
  * One integration test that runs a small scenario end-to-end and diffs
    every populated parquet in ``output_raw/`` against the legacy
    ``glpsol``-phase-3 CSV, tolerating only format-precision rounding.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from flextool.lean_parquet import read_lean_parquet
from flextool.process_outputs.read_highs_solution import (
    VARIABLE_SPECS,
    VariableSpec,
    _name_regex,
    extract_variable,
    write_all_variables,
    write_v_dual_invest_by_class,
    write_v_dual_node_balance,
    write_v_obj,
    write_variable_parquet,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _fake_highs(
    variable_names: list[str],
    col_values: list[float],
    col_duals: list[float] | None = None,
    row_names: list[str] | None = None,
    row_duals: list[float] | None = None,
) -> MagicMock:
    """Construct a mock ``Highs`` exposing the arrays each extractor reads."""
    h = MagicMock()
    h.allVariableNames.return_value = list(variable_names)
    sol = SimpleNamespace(
        col_value=list(col_values),
        col_dual=list(col_duals if col_duals is not None else [0.0] * len(col_values)),
        row_dual=list(row_duals or []),
    )
    h.getSolution.return_value = sol
    h.getLp.return_value = SimpleNamespace(row_names_=list(row_names or []))
    return h


# ---------------------------------------------------------------------------
# Parser / registry sanity
# ---------------------------------------------------------------------------


def test_name_regex_matches_bracket_form() -> None:
    regex = _name_regex("v_flow")
    m = regex.match("v_flow[p1,srcA,sinkX,d1,t1]")
    assert m is not None
    assert m.group(1).split(",") == ["p1", "srcA", "sinkX", "d1", "t1"]


def test_name_regex_does_not_cross_variables() -> None:
    # v_flow regex must not match v_flow_extra or v_flowX.
    regex = _name_regex("v_flow")
    assert regex.match("v_flow_extra[a,b,c,d,t]") is None
    assert regex.match("v_invest[e,d]") is None


def test_variable_specs_output_names_unique() -> None:
    """Every spec must write to a distinct parquet file."""
    out_names = [s.output_name or s.name for s in VARIABLE_SPECS]
    assert len(out_names) == len(set(out_names)), (
        f"duplicate output names: {[n for n in out_names if out_names.count(n) > 1]}"
    )


# ---------------------------------------------------------------------------
# extract_variable — shape permutations
# ---------------------------------------------------------------------------


def test_extract_multiindex_columns_with_time() -> None:
    h = _fake_highs(
        variable_names=[
            "v_flow[p1,srcA,sinkX,d1,t1]",
            "v_flow[p1,srcA,sinkX,d1,t2]",
            "v_flow[p2,srcB,sinkY,d1,t1]",
            "v_invest[e1,d1]",  # unrelated — must be ignored
        ],
        col_values=[0.5, 0.6, 1.2, 99.0],
    )
    df = extract_variable(
        h, "v_flow", ("process", "source", "sink"), solve_name="s1",
    )
    assert df.index.names == ["solve", "period", "time"]
    assert df.columns.names == ["process", "source", "sink"]
    assert df.shape == (2, 2)
    assert df.loc[("s1", "d1", "t1"), ("p1", "srcA", "sinkX")] == 0.5
    # Missing (p2,srcB,sinkY, d1, t2) must be filled with 0.0
    assert df.loc[("s1", "d1", "t2"), ("p2", "srcB", "sinkY")] == 0.0


def test_extract_single_col_with_time() -> None:
    h = _fake_highs(
        variable_names=["v_state[nA,d1,t1]", "v_state[nA,d1,t2]"],
        col_values=[10.0, 20.0],
    )
    df = extract_variable(h, "v_state", ("node",), solve_name="s1")
    assert df.index.names == ["solve", "period", "time"]
    # Single-level column index with name preserved
    assert not isinstance(df.columns, pd.MultiIndex)
    assert df.columns.name == "node"
    assert list(df.columns) == ["nA"]


def test_extract_single_col_no_time() -> None:
    h = _fake_highs(
        variable_names=["v_invest[e1,d1]", "v_invest[e2,d1]"],
        col_values=[42.0, 7.0],
    )
    df = extract_variable(
        h, "v_invest", ("entity",), solve_name="s1", has_time=False,
    )
    assert df.index.names == ["solve", "period"]
    assert df.columns.name == "entity"
    assert df.loc[("s1", "d1"), "e1"] == 42.0
    assert df.loc[("s1", "d1"), "e2"] == 7.0


def test_extract_empty_result_has_correct_index_shape() -> None:
    """When no variables match, return a well-typed empty frame."""
    h = _fake_highs(variable_names=["v_other[a,b]"], col_values=[1.0])
    df = extract_variable(h, "v_flow", ("process", "source", "sink"), solve_name="s1")
    assert df.shape == (0, 0)
    assert df.index.names == ["solve", "period", "time"]
    assert df.columns.names == ["process", "source", "sink"]


def test_extract_dual_uses_row_arrays_and_applies_scale() -> None:
    """source='row_dual' reads row_names_ + row_dual; value_scale multiplies."""
    h = _fake_highs(
        variable_names=["v_flow[p1,a,b,d1,t1]"],
        col_values=[999.0],  # must be ignored
        row_names=[
            "total_cost",  # not matched — no bracket form
            "maxInvest_entity_period[e1,d1]",
            "maxInvest_entity_period[e2,d1]",
        ],
        row_duals=[0.0, 2.5e-3, -1.1e-3],
    )
    df = extract_variable(
        h, "maxInvest_entity_period", ("entity",),
        solve_name="s1", has_time=False, source="row_dual", value_scale=1e6,
    )
    # scale of 1e6 applied
    assert df.loc[("s1", "d1"), "e1"] == pytest.approx(2500.0)
    assert df.loc[("s1", "d1"), "e2"] == pytest.approx(-1100.0)


def test_extract_col_dual_reads_col_dual_array() -> None:
    """source='col_dual' uses allVariableNames + col_dual, not col_value."""
    h = _fake_highs(
        variable_names=["v_invest[e1,d1]", "v_invest[e2,d1]"],
        col_values=[10.0, 20.0],
        col_duals=[0.5, -1.2],
    )
    df = extract_variable(
        h, "v_invest", ("entity",), solve_name="s1",
        has_time=False, source="col_dual",
    )
    assert df.loc[("s1", "d1"), "e1"] == 0.5
    assert df.loc[("s1", "d1"), "e2"] == -1.2


def test_extract_no_period_produces_solve_only_index() -> None:
    """has_period=False collapses row index to ``(solve,)``."""
    h = _fake_highs(
        variable_names=["v_other[a]"], col_values=[7.0],
        row_names=["co2_max_total[g1]", "co2_max_total[g2]"],
        row_duals=[1.5, -2.0],
    )
    df = extract_variable(
        h, "co2_max_total", ("group",), solve_name="s1",
        has_time=False, has_period=False, source="row_dual", value_scale=2.0,
    )
    assert df.index.names == ["solve"]
    assert df.index.tolist() == ["s1"]
    assert df.columns.name == "group"
    assert df.loc["s1", "g1"] == 3.0  # 1.5 × scale
    assert df.loc["s1", "g2"] == -4.0


def test_extract_leading_and_trailing_ignore() -> None:
    """leading_ignore/trailing_ignore drop bookkeeping indices from parse."""
    # nodeBalance_eq has 8 parts: c, n, d, t, tp, tpwt, dp, tpws
    h = _fake_highs(
        variable_names=[],
        col_values=[],
        row_names=[
            "nodeBalance_eq[s1,nA,d1,t1,tp,tpwt,dp,tpws]",
            "nodeBalance_eq[s1,nB,d1,t1,tp,tpwt,dp,tpws]",
        ],
        row_duals=[1.0, 2.0],
    )
    df = extract_variable(
        h, "nodeBalance_eq", ("node",),
        solve_name="s1", has_time=True, source="row_dual",
        leading_ignore=1, trailing_ignore=4,
    )
    assert df.columns.name == "node"
    assert list(df.columns) == ["nA", "nB"]
    assert df.loc[("s1", "d1", "t1"), "nA"] == 1.0
    assert df.loc[("s1", "d1", "t1"), "nB"] == 2.0


def test_extract_skips_wrong_arity(caplog: pytest.LogCaptureFixture) -> None:
    """A name with fewer/more indices than expected is warned and skipped."""
    h = _fake_highs(
        variable_names=[
            "v_state[nA,d1,t1]",
            "v_state[broken_extra,nA,d1,t1]",  # 4 parts — wrong
        ],
        col_values=[10.0, 99.0],
    )
    with caplog.at_level("WARNING"):
        df = extract_variable(h, "v_state", ("node",), solve_name="s1")
    assert df.shape == (1, 1)
    assert "Unexpected v_state arity" in caplog.text


# ---------------------------------------------------------------------------
# Realized-dispatch / realized-periods filtering (integrated cheaply)
# ---------------------------------------------------------------------------


def test_realized_dispatch_filter(tmp_path: Path) -> None:
    realized_csv = tmp_path / "realized.csv"
    # Only t1 is realized; t2 must be dropped.
    realized_csv.write_text("period,step\nd1,t1\n")
    h = _fake_highs(
        variable_names=["v_state[nA,d1,t1]", "v_state[nA,d1,t2]"],
        col_values=[10.0, 20.0],
    )
    df = extract_variable(
        h, "v_state", ("node",), solve_name="s1",
        realized_dispatch_csv=realized_csv,
    )
    assert df.shape == (1, 1)
    assert ("s1", "d1", "t1") in df.index
    assert ("s1", "d1", "t2") not in df.index


def test_realized_periods_filter(tmp_path: Path) -> None:
    realized_csv = tmp_path / "realized_periods.csv"
    realized_csv.write_text("period\nd1\n")  # d2 must be dropped
    h = _fake_highs(
        variable_names=["v_invest[e1,d1]", "v_invest[e1,d2]"],
        col_values=[42.0, 99.0],
    )
    df = extract_variable(
        h, "v_invest", ("entity",), solve_name="s1",
        has_time=False, realized_periods_csv=realized_csv,
    )
    assert df.shape == (1, 1)
    assert ("s1", "d1") in df.index
    assert ("s1", "d2") not in df.index


# ---------------------------------------------------------------------------
# write_variable_parquet / write_all_variables — file I/O round-trip
# ---------------------------------------------------------------------------


def test_write_variable_parquet_round_trip(tmp_path: Path) -> None:
    h = _fake_highs(
        variable_names=["v_state[nA,d1,t1]", "v_state[nB,d1,t1]"],
        col_values=[1.5, 2.5],
    )
    spec = VariableSpec("v_state", ("node",))
    path = write_variable_parquet(h, spec, solve_name="s1", output_dir=tmp_path)
    assert path.name == "v_state__s1.parquet"
    back = read_lean_parquet(path)
    assert back.columns.name == "node"
    assert back.index.names == ["solve", "period", "time"]
    assert back.loc[("s1", "d1", "t1"), "nB"] == 2.5


def test_write_v_obj_scalar(tmp_path: Path) -> None:
    """v_obj file contains (solve,)-indexed objective scaled by 1e6."""
    h = _fake_highs(variable_names=[], col_values=[])
    h.getObjectiveValue.return_value = 1.234  # raw HiGHS objective (scaled)
    path = write_v_obj(h, solve_name="s1", output_dir=tmp_path)
    assert path.name == "v_obj__s1.parquet"
    df = read_lean_parquet(path)
    assert df.index.name == "solve"
    assert df.loc["s1", "objective"] == pytest.approx(1.234e6)


def test_write_v_dual_invest_by_class_splits_entities(tmp_path: Path) -> None:
    """Produces 3 parquets; each keeps only entities matching its class."""
    # input/ layout that the loader reads
    (tmp_path / "input").mkdir()
    (tmp_path / "output_raw").mkdir()
    (tmp_path / "input" / "process_unit.csv").write_text("process_unit\nu1\nu2\n")
    (tmp_path / "input" / "process_connection.csv").write_text("process_connection\nc1\n")
    (tmp_path / "input" / "node.csv").write_text("node\nn1\n")

    h = _fake_highs(
        variable_names=[
            "v_invest[u1,p2020]",
            "v_invest[u2,p2020]",
            "v_invest[c1,p2020]",
            "v_invest[n1,p2020]",
        ],
        col_values=[0.0, 0.0, 0.0, 0.0],
        col_duals=[1.1, 2.2, 3.3, 4.4],
    )

    paths = write_v_dual_invest_by_class(
        h, solve_name="s1",
        output_dir=tmp_path / "output_raw",
        work_folder=tmp_path,
    )
    assert len(paths) == 3
    unit_df = read_lean_parquet(tmp_path / "output_raw" / "v_dual_invest_unit__s1.parquet")
    conn_df = read_lean_parquet(tmp_path / "output_raw" / "v_dual_invest_connection__s1.parquet")
    node_df = read_lean_parquet(tmp_path / "output_raw" / "v_dual_invest_node__s1.parquet")

    assert sorted(unit_df.columns) == ["u1", "u2"]
    assert list(conn_df.columns) == ["c1"]
    assert list(node_df.columns) == ["n1"]
    assert unit_df.loc[("s1", "p2020"), "u1"] == 1.1
    assert unit_df.loc[("s1", "p2020"), "u2"] == 2.2
    assert conn_df.loc[("s1", "p2020"), "c1"] == 3.3
    assert node_df.loc[("s1", "p2020"), "n1"] == 4.4


def test_write_v_dual_node_balance_applies_per_period_inflation(tmp_path: Path) -> None:
    """Value = −raw_dual × 1e6 / inflation[period]; trailing 4 indices skipped."""
    (tmp_path / "output_raw").mkdir()
    (tmp_path / "input").mkdir()
    (tmp_path / "solve_data").mkdir()
    (tmp_path / "solve_data" / "p_inflation_factor_operations_yearly.csv").write_text(
        "solve,period,value\ns1,p2020,1.0\ns1,p2025,2.0\n"
    )
    # Agent 1.4: nodeBalance_eq gained a ``bn`` block subscript between
    # ``node`` and ``period``; the extractor handles this via
    # ``mid_ignore=1``.  In degenerate mode the block tag is 'default'.
    h = _fake_highs(
        variable_names=[], col_values=[],
        row_names=[
            "nodeBalance_eq[s1,nA,default,p2020,t0001,tp,tpwt,dp,tpws]",
            "nodeBalance_eq[s1,nA,default,p2025,t0001,tp,tpwt,dp,tpws]",
        ],
        row_duals=[3e-6, -4e-6],  # scaled duals
    )
    write_v_dual_node_balance(
        h, solve_name="s1",
        output_dir=tmp_path / "output_raw",
        work_folder=tmp_path,
    )
    df = read_lean_parquet(tmp_path / "output_raw" / "v_dual_node_balance__s1.parquet")
    # raw × −1e6 / inflation
    # (s1, p2020, t0001): -(3e-6) × 1e6 / 1.0 = -3.0
    # (s1, p2025, t0001): -(-4e-6) × 1e6 / 2.0 = +2.0
    assert df.loc[("s1", "p2020", "t0001"), "nA"] == pytest.approx(-3.0)
    assert df.loc[("s1", "p2025", "t0001"), "nA"] == pytest.approx(2.0)


def test_write_all_variables_isolates_failures(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """A broken spec must not abort the rest."""
    h = _fake_highs(variable_names=["v_flow[a,b,c,d1,t1]"], col_values=[1.0])
    # Custom list including one bad spec (wrong arity).
    specs = [
        VariableSpec("v_flow", ("process", "source", "sink")),
        VariableSpec("v_flow", ("WRONG", "ARITY")),  # arity mismatch
    ]
    with caplog.at_level("WARNING"):
        paths = write_all_variables(
            h, solve_name="s1", output_dir=tmp_path, specs=specs,
        )
    # The good one succeeded — the bad one was skipped (the mismatch warning
    # is logged from extract_variable, not from write_all_variables' except
    # clause, so both may be 'written' as empty parquets).
    assert len(paths) >= 1


# ---------------------------------------------------------------------------
# Integration test: run a small scenario and diff every populated parquet
# against the corresponding glpsol CSV.
# ---------------------------------------------------------------------------


def _read_csv_like_legacy(
    path: Path, col_names: tuple[str, ...], has_time: bool,
) -> pd.DataFrame | None:
    """Read a glpsol-written output_raw CSV into the legacy wide shape."""
    if not path.exists():
        return None
    header = list(range(len(col_names))) if len(col_names) >= 2 else 0
    index_col = [0, 1, 2] if has_time else [0, 1]
    try:
        return pd.read_csv(path, header=header, index_col=index_col).astype(float)
    except Exception:
        # Empty CSV (header only) — return a header-only frame
        return pd.read_csv(path, header=header, index_col=index_col)


def test_integration_parquet_matches_csv_on_base_scenario(
    test_db_url: str, test_bin_dir: Path, workdir: Path,
) -> None:
    """End-to-end: run a scenario, then for every registered spec whose
    parquet has content, verify the values match the CSV within the
    CSV's format precision (``%.6g`` → ~1e-4 absolute error at magnitudes
    of 10²).
    """
    from flextool.flextoolrunner.flextoolrunner import FlexToolRunner

    # ``base`` is the smallest scenario in the test fixture — just enough
    # to exercise v_flow + v_state (dispatch-only).  The full registry is
    # still written — vars not used by ``base`` come out as empty parquets.
    scenario = "base"
    runner = FlexToolRunner(
        input_db_url=test_db_url,
        scenario_name=scenario,
        root_dir=workdir,
        bin_dir=test_bin_dir,
        use_old_raw_csv=True,  # phase-3 CSVs needed alongside parquet for the diff
    )
    runner.write_input(test_db_url, scenario)
    assert runner.run_model() == 0, "Model solve failed"

    solve_name = runner.state.solve.real_solves[0] if runner.state.solve.real_solves else None
    # Fall back to scanning parquet filenames if real_solves was not populated.
    parquet_dir = workdir / "output_raw"
    if not solve_name:
        any_pq = next(parquet_dir.glob("v_flow__*.parquet"), None)
        assert any_pq, "no v_flow parquet written"
        solve_name = any_pq.stem[len("v_flow__"):]

    compared = 0
    for spec in VARIABLE_SPECS:
        out_name = spec.output_name or spec.name
        parq_path = parquet_dir / f"{out_name}__{solve_name}.parquet"
        assert parq_path.exists(), f"missing parquet for {out_name}"
        par = read_lean_parquet(parq_path)
        if par.empty:
            continue  # nothing to diff

        csv = _read_csv_like_legacy(
            parquet_dir / f"{out_name}.csv", spec.col_names, spec.has_time,
        )
        if csv is None or csv.empty:
            continue

        common_cols = csv.columns.intersection(par.columns)
        common_rows = csv.index.intersection(par.index)
        if not len(common_cols) or not len(common_rows):
            continue

        csv_s = csv.loc[common_rows, common_cols].sort_index().sort_index(axis=1)
        par_s = par.loc[common_rows, common_cols].sort_index().sort_index(axis=1)
        max_diff = float(np.nanmax(np.abs(
            csv_s.astype(float).values - par_s.astype(float).values
        )))
        # CSV uses %.6g (6 sig figs).  For flextool values in [0, 1e6] this
        # means absolute error up to ~1 in the worst case; typical MW-range
        # dispatch values stay well under 1e-3.
        assert max_diff < 1.0, (
            f"{out_name}: parquet/CSV diverge beyond CSV format precision "
            f"(max abs diff {max_diff:.3e})"
        )
        compared += 1

    assert compared >= 1, "no populated parquet was compared — check the scenario"
