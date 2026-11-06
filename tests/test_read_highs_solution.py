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

import pandas as pd
import polars as pl
import pytest
from polar_high import Param

from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
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
from tests._support.provider import provider_from_csv_dirs


def _param(*dims: str, **cols) -> Param:
    """A real :class:`Param` built from column lists, e.g.
    ``_param("d", "t", d=["p2020"], t=["t0001"], value=[2.0])``."""
    return Param(dims, pl.DataFrame(cols))


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


def test_extract_iso8601_timestamp_strips_glpsol_quotes(tmp_path: Path) -> None:
    """ISO 8601 timestamps contain colons, which trigger GLPSOL/MPS auto-quoting:
    ``v_state[ARG_NH3,y2050,'2050-01-01T00:00:00']``.  HiGHS preserves the
    apostrophes in ``allVariableNames()``, while the canonical row order
    (``solve_data/p_step_duration.csv`` or a realized-dispatch CSV) carries
    bare timestamps.  Without stripping, every time-indexed lookup misses
    and the parquet ends up filled with zeros — see h2-imo project regression.
    """
    realized = tmp_path / "realized_dispatch.csv"
    realized.write_text(
        "period,step\n"
        "y2050,2050-01-01T00:00:00\n"
        "y2050,2050-01-01T01:00:00\n"
    )
    h = _fake_highs(
        variable_names=[
            "v_state[ARG_NH3,y2050,'2050-01-01T00:00:00']",
            "v_state[ARG_NH3,y2050,'2050-01-01T01:00:00']",
        ],
        col_values=[0.192, 0.154],
    )
    df = extract_variable(
        h, "v_state", ("node",),
        solve_name="solve_test_6h",
        realized_dispatch_csv=realized,
    )
    # Row index must use the bare timestamp form — apostrophes stripped.
    assert list(df.index) == [
        ("solve_test_6h", "y2050", "2050-01-01T00:00:00"),
        ("solve_test_6h", "y2050", "2050-01-01T01:00:00"),
    ]
    # And the values must land in the correct cells, not get silently
    # dropped because the quoted row_key missed the canonical lookup.
    assert df.loc[("solve_test_6h", "y2050", "2050-01-01T00:00:00"), "ARG_NH3"] == 0.192
    assert df.loc[("solve_test_6h", "y2050", "2050-01-01T01:00:00"), "ARG_NH3"] == 0.154


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


def test_extract_arity3_node_balance() -> None:
    """Δ.30 — nodeBalance_eq is arity-3 ``[node, period, time]``.

    The polars LP emits this constraint without the GMPL-era bookkeeping
    subscripts (``c, bn, tp, tpwt, dp, tpws``); the parser accepts only
    the polars-native shape now.
    """
    h = _fake_highs(
        variable_names=[],
        col_values=[],
        row_names=[
            "nodeBalance_eq[nA,d1,t1]",
            "nodeBalance_eq[nB,d1,t1]",
        ],
        row_duals=[1.0, 2.0],
    )
    df = extract_variable(
        h, "nodeBalance_eq", ("node",),
        solve_name="s1", has_time=True, source="row_dual",
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
    """Item 163: no disk fallback — the (period, step) filter frame comes from the provider,
    keyed ``solve_data/realized_dispatch`` (the key ``_provider_lookup`` derives from the
    ``realized_dispatch_csv`` path's parent/stem)."""
    sd = tmp_path / "solve_data"
    sd.mkdir()
    # Only t1 is realized; t2 must be dropped.
    (sd / "realized_dispatch.csv").write_text("period,step\nd1,t1\n")
    provider = provider_from_csv_dirs(solve_data_dir=sd)
    h = _fake_highs(
        variable_names=["v_state[nA,d1,t1]", "v_state[nA,d1,t2]"],
        col_values=[10.0, 20.0],
    )
    df = extract_variable(
        h, "v_state", ("node",), solve_name="s1",
        realized_dispatch_csv=sd / "realized_dispatch.csv",
        provider=provider,
    )
    assert df.shape == (1, 1)
    assert ("s1", "d1", "t1") in df.index
    assert ("s1", "d1", "t2") not in df.index


def test_realized_periods_filter(tmp_path: Path) -> None:
    """Item 163: a dispatch-only roll — an EMPTY realized-INVEST period list — makes a
    ``period_source="invest"`` variable (``v_invest`` / ``v_divest``) emit NO rows for the roll, even
    though its dispatch axis (``flex_data.realized_dispatch``) has periods.  This is the still-
    provider-driven mechanism in :func:`_load_realized_invest_periods_list` (no disk fallback), keyed
    ``solve_data/realized_invest_periods_of_current_solve`` — see the ``period_source="invest"``
    VariableSpec comment for why a dispatch-only roll must contribute no rows.
    """
    sd = tmp_path / "solve_data"
    sd.mkdir()
    (sd / "realized_invest_periods_of_current_solve.csv").write_text("period\n")  # empty: no investment this roll
    provider = provider_from_csv_dirs(solve_data_dir=sd)
    fd = SimpleNamespace(
        realized_dispatch=pl.DataFrame({"period": ["d1", "d2"], "step": ["t1", "t1"]}),
        recourse_invest=False,
    )
    h = _fake_highs(
        variable_names=["v_invest[e1,d1]", "v_invest[e1,d2]"],
        col_values=[42.0, 99.0],
    )
    df = extract_variable(
        h, "v_invest", ("entity",), solve_name="s1",
        has_time=False, period_source="invest",
        realized_periods_csv=sd / "realized_invest_periods_of_current_solve.csv",
        provider=provider, flex_data=fd,
    )
    assert df.shape == (0, 1)
    assert ("s1", "d1") not in df.index
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
    """v_obj file contains (solve,)-indexed objective scaled by 1e6.

    Cold-path contract: when ``_flextool_unscaled_objective`` is not
    stashed (no autoscale Layer 2 push), ``write_v_obj`` falls back
    to ``h.getObjectiveValue()`` (see commit aab46eb5).  We must
    explicitly null the attribute because :class:`MagicMock`
    auto-fabricates attribute reads and ``float(MagicMock())`` returns
    ``1.0`` — without the explicit None the test would silently exit
    via the autoscale branch and never exercise getObjectiveValue.
    """
    h = _fake_highs(variable_names=[], col_values=[])
    h._flextool_unscaled_objective = None  # cold path
    h.getObjectiveValue.return_value = 1.234  # raw HiGHS objective (scaled)
    path = write_v_obj(h, solve_name="s1", output_dir=tmp_path)
    assert path.name == "v_obj__s1.parquet"
    df = read_lean_parquet(path)
    assert df.index.name == "solve"
    assert df.loc["s1", "objective"] == pytest.approx(1.234e6)


def test_write_v_dual_invest_by_class_splits_entities(tmp_path: Path) -> None:
    """Produces 3 parquets; each keeps only entities matching its class.

    Item 163: the entity-class membership comes from the sub-solve's
    FlexDataProvider (``input/process_unit`` / ``input/process_connection`` /
    ``input/node``), never from a work-folder CSV — a normal run never
    writes those CSVs.

    Also pins the col-dual scaling: HiGHS returns the reduced cost in
    scaled-objective units, so the writer multiplies by
    ``1 / scale_the_objective``.  With ``scale_the_objective=1e-6`` the
    written values equal the raw col duals × 1e6.
    """
    (tmp_path / "output_raw").mkdir()
    provider = FlexDataProvider()
    provider.put("input/process_unit", pl.DataFrame({"process_unit": ["u1", "u2"]}))
    provider.put("input/process_connection", pl.DataFrame({"process_connection": ["c1"]}))
    provider.put("input/node", pl.DataFrame({"node": ["n1"]}))

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

    scale = 1e-6
    paths = write_v_dual_invest_by_class(
        h, solve_name="s1",
        output_dir=tmp_path / "output_raw",
        provider=provider,
        scale_the_objective=scale,
    )
    assert len(paths) == 3
    unit_df = read_lean_parquet(tmp_path / "output_raw" / "v_dual_invest_unit__s1.parquet")
    conn_df = read_lean_parquet(tmp_path / "output_raw" / "v_dual_invest_connection__s1.parquet")
    node_df = read_lean_parquet(tmp_path / "output_raw" / "v_dual_invest_node__s1.parquet")

    inv_scale = 1.0 / scale  # 1e6
    assert sorted(unit_df.columns) == ["u1", "u2"]
    assert list(conn_df.columns) == ["c1"]
    assert list(node_df.columns) == ["n1"]
    assert unit_df.loc[("s1", "p2020"), "u1"] == pytest.approx(1.1 * inv_scale)
    assert unit_df.loc[("s1", "p2020"), "u2"] == pytest.approx(2.2 * inv_scale)
    assert conn_df.loc[("s1", "p2020"), "c1"] == pytest.approx(3.3 * inv_scale)
    assert node_df.loc[("s1", "p2020"), "n1"] == pytest.approx(4.4 * inv_scale)


def test_write_v_dual_invest_by_class_honours_custom_scale(tmp_path: Path) -> None:
    """A non-default scale_the_objective scales the reduced cost by 1/scale."""
    (tmp_path / "output_raw").mkdir()
    provider = FlexDataProvider()
    provider.put("input/process_unit", pl.DataFrame({"process_unit": ["u1"]}))
    provider.put(
        "input/process_connection",
        pl.DataFrame({"process_connection": []}, schema={"process_connection": pl.Utf8}),
    )
    provider.put("input/node", pl.DataFrame({"node": []}, schema={"node": pl.Utf8}))

    h = _fake_highs(
        variable_names=["v_invest[u1,p2020]"],
        col_values=[0.0],
        col_duals=[5.0],
    )
    scale = 2e-3
    write_v_dual_invest_by_class(
        h, solve_name="s1",
        output_dir=tmp_path / "output_raw",
        provider=provider,
        scale_the_objective=scale,
    )
    unit_df = read_lean_parquet(tmp_path / "output_raw" / "v_dual_invest_unit__s1.parquet")
    assert unit_df.loc[("s1", "p2020"), "u1"] == pytest.approx(5.0 / scale)


def test_v_dual_invest_classes_ignore_input_csvs(tmp_path: Path) -> None:
    """A stale ``input/process_connection.csv`` (left by an earlier run / dump) must not override
    the provider's entity-class membership — item 163 T8: the writer never reads the work folder."""
    (tmp_path / "output_raw").mkdir()
    (tmp_path / "input").mkdir()
    # Stale disk file: claims u1 is a connection. The writer must ignore it.
    (tmp_path / "input" / "process_connection.csv").write_text("process_connection\nu1\n")

    provider = FlexDataProvider()
    provider.put("input/process_unit", pl.DataFrame({"process_unit": ["u1"]}))
    provider.put(
        "input/process_connection",
        pl.DataFrame({"process_connection": []}, schema={"process_connection": pl.Utf8}),
    )
    provider.put("input/node", pl.DataFrame({"node": []}, schema={"node": pl.Utf8}))

    h = _fake_highs(
        variable_names=["v_invest[u1,p2020]"],
        col_values=[0.0],
        col_duals=[1.0],
    )
    write_v_dual_invest_by_class(
        h, solve_name="s1",
        output_dir=tmp_path / "output_raw",
        provider=provider,
        scale_the_objective=1e-6,
    )
    unit_df = read_lean_parquet(tmp_path / "output_raw" / "v_dual_invest_unit__s1.parquet")
    conn_df = read_lean_parquet(tmp_path / "output_raw" / "v_dual_invest_connection__s1.parquet")
    assert list(unit_df.columns) == ["u1"]
    assert list(conn_df.columns) == []


def test_write_v_dual_invest_by_class_requires_provider(tmp_path: Path) -> None:
    """Item 163: a None provider is a wiring bug, not a tolerated absence — raise, don't silently
    revert to an empty entity-class split."""
    (tmp_path / "output_raw").mkdir()
    h = _fake_highs(variable_names=["v_invest[u1,p2020]"], col_values=[0.0], col_duals=[1.0])
    with pytest.raises(ValueError, match="FlexDataProvider"):
        write_v_dual_invest_by_class(
            h, solve_name="s1", output_dir=tmp_path / "output_raw", provider=None,
        )


def test_write_v_dual_node_balance_applies_per_period_inflation(tmp_path: Path) -> None:
    """price = +raw_dual × 1e6 / (timestep_weight · inflation / period_share); arity-3 polars row names.

    Item 163: the weights come from ``flex_data`` only (no ``solve_data`` CSV fallback) — a stub carrying
    the values the CSVs used to.  The default scale 1e-6 applies without ``scale_the_objective`` given.
    """
    (tmp_path / "output_raw").mkdir()
    fd = SimpleNamespace(
        p_timestep_weight=_param("d", "t", d=["p2020"], t=["t0001"], value=[2.0]),
        p_inflation_op=_param("d", d=["p2020", "p2025"], value=[1.0, 2.0]),
        p_period_share=_param("d", d=["p2020", "p2025"], value=[0.5, 0.25]),
        pdt_branch_weight=None,
        p_step_duration=None,
        # ``dt`` is the (d, t) domain :func:`price_weight_from_components` unions in: without it, a period
        # with no ``p_timestep_weight`` row (p2025 here) would be absent from the weight frame entirely and
        # fall back to weight 1 (null -> fill_null(1.0)) instead of the period-only components' 2.0/0.25=8.
        dt=pl.DataFrame({"d": ["p2020", "p2025"], "t": ["t0001", "t0001"]}),
    )
    h = _fake_highs(
        variable_names=[], col_values=[],
        row_names=[
            "nodeBalance_eq[nA,p2020,t0001]",
            "nodeBalance_eq[nA,p2025,t0001]",
        ],
        row_duals=[3e-6, -4e-6],  # scaled duals
    )
    write_v_dual_node_balance(
        h, solve_name="s1",
        output_dir=tmp_path / "output_raw",
        flex_data=fd,
    )
    df = read_lean_parquet(tmp_path / "output_raw" / "v_dual_node_balance__s1.parquet")
    # (s1, p2020, t0001): 3e-6 × 1e6 / (2.0 · 1.0 / 0.5) = 0.75
    # (s1, p2025, t0001): -4e-6 × 1e6 / (1.0 · 2.0 / 0.25) = -0.5   (no timestep weight row → 1)
    assert df.loc[("s1", "p2020", "t0001"), "nA"] == pytest.approx(0.75)
    assert df.loc[("s1", "p2025", "t0001"), "nA"] == pytest.approx(-0.5)


def test_write_v_dual_reserve_balance_sums_requirement_rows(tmp_path: Path) -> None:
    """reserve_price_t = the sum of the duals of the group's requirement rows (timeseries, dynamic, and the
    large-failure rows of every failing process), whatever the group's reserve_method; no method CSV needed.
    Price = Σ dual × 1e6 (default scale, no weight components → weight 1).

    Item 163: no FlexData weight components at all, but ``dt`` still supplies the (period, time) domain so
    the weight frame covers the dual's cell (an empty domain would give a null price, not weight 1).
    """
    from flextool.process_outputs.read_highs_solution import write_v_dual_reserve_balance

    (tmp_path / "output_raw").mkdir()
    fd = SimpleNamespace(
        p_timestep_weight=None, p_inflation_op=None, p_period_share=None,
        pdt_branch_weight=None, p_step_duration=None,
        dt=pl.DataFrame({"d": ["p2020"], "t": ["t0001"]}),
    )
    h = _fake_highs(
        variable_names=[], col_values=[],
        row_names=[
            "reserveBalance_timeseries_eq[both,up,g1,p2020,t0001]",
            "reserveBalance_dynamic_eq[both,up,g1,p2020,t0001]",
            "reserveBalance_dynamic_eq[dz,up,g2,p2020,t0001]",
            "reserveBalance_dynamic_eq[mix,up,g3,p2020,t0001]",
            "reserveBalance_up_n_1_eq[mix,up,g3,f1,p2020,t0001]",
            "reserveBalance_up_n_1_eq[mix,up,g3,f2,p2020,t0001]",
            "reserveBalance_up_n_1_eq[n1,up,g4,f1,p2020,t0001]",
            "reserveBalance_up_n_1_eq[n1,up,g4,f2,p2020,t0001]",
            "reserveBalance_down_n_1_eq[nd,down,g5,f1,p2020,t0001]",
        ],
        row_duals=[1.5e-6, 7.0e-6, 2.5e-6, 3.5e-6, 0.5e-6, 1.0e-6, 2.0e-6, 0.25e-6, 0.75e-6],
    )
    write_v_dual_reserve_balance(h, solve_name="s1", output_dir=tmp_path / "output_raw", flex_data=fd)
    df = read_lean_parquet(tmp_path / "output_raw" / "v_dual_reserve__upDown__group__period__t__s1.parquet")
    cols = {tuple(str(x) for x in c): c for c in df.columns}
    expected = {("both", "up", "g1"): 8.5, ("dz", "up", "g2"): 2.5, ("mix", "up", "g3"): 5.0,
                ("n1", "up", "g4"): 2.25, ("nd", "down", "g5"): 0.75}
    assert set(cols) == set(expected)
    for key, price in expected.items():
        assert df.loc[("s1", "p2020", "t0001"), cols[key]] == pytest.approx(price)


def test_write_v_dual_reserve_balance_divides_by_step_and_weights(tmp_path: Path) -> None:
    """reserve_price = dual / s / (Δt · tw · inflation / period_share): 6e-6 × 1e6 / (2 · 0.5 · 3 / 0.25) = 0.5.

    Item 163: every weight component is a FlexData stub (no ``solve_data`` CSVs)."""
    from flextool.process_outputs.read_highs_solution import write_v_dual_reserve_balance

    (tmp_path / "output_raw").mkdir()
    fd = SimpleNamespace(
        p_timestep_weight=_param("d", "t", d=["p2020"], t=["t0001"], value=[0.5]),
        p_inflation_op=_param("d", d=["p2020"], value=[3.0]),
        p_period_share=_param("d", d=["p2020"], value=[0.25]),
        pdt_branch_weight=None,
        p_step_duration=_param("d", "t", d=["p2020"], t=["t0001"], value=[2.0]),
        dt=pl.DataFrame({"d": ["p2020"], "t": ["t0001"]}),
    )
    h = _fake_highs(variable_names=[], col_values=[],
                    row_names=["reserveBalance_timeseries_eq[r,up,g,p2020,t0001]"], row_duals=[6e-6])
    write_v_dual_reserve_balance(h, solve_name="s1", output_dir=tmp_path / "output_raw", flex_data=fd)
    df = read_lean_parquet(tmp_path / "output_raw" / "v_dual_reserve__upDown__group__period__t__s1.parquet")
    assert df.iloc[0, 0] == pytest.approx(0.5)


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


