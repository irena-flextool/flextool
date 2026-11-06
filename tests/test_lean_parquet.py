"""Reading lean parquets without pandas: the footer layout, the wide read of chosen columns and the long view.

Every table is written here by ``write_lean_parquet`` (as the engine writes its results) or by
``DataFrame.to_parquet`` (an older file); the expected layouts and values are the frames the test wrote.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from flextool import lean_parquet
from flextool.lean_parquet import lean_parquet_layout, read_lean_parquet_wide, to_long, write_lean_parquet


def _result_frame() -> pd.DataFrame:
    """A result table as write_outputs writes it: rows (period, time), columns (scenario, unit, node)."""
    index = pd.MultiIndex.from_tuples([("p2020", "t01"), ("p2020", "t02"), ("p2025", "t01")],
                                      names=["period", "time"])
    columns = pd.MultiIndex.from_tuples([("coal", "plant_a", "west"), ("coal", "o'neil", "east")],
                                        names=["scenario", "unit", "node"])
    return pd.DataFrame([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], index=index, columns=columns)


def test_layout_of_a_result_table(tmp_path: Path) -> None:
    path = tmp_path / "unit_outputNode_dt_ee.parquet"
    write_lean_parquet(_result_frame(), path, column_metadata={"west": {"unit": "MW"}})
    layout = lean_parquet_layout(path)
    assert layout.lean and layout.num_rows == 3
    assert (layout.idx, layout.index_physical) == (("period", "time"), ("period", "time"))
    assert layout.col == ("scenario", "unit", "node")
    # A name with a quote survives arrow's tuple text.
    assert layout.columns == (("coal", "plant_a", "west"), ("coal", "o'neil", "east"))
    assert layout.physical == ("('coal', 'plant_a', 'west')", '(\'coal\', "o\'neil", \'east\')')
    assert layout.column_meta == {"west": {"unit": "MW"}}
    assert layout.level_values("unit") == ["plant_a", "o'neil"]
    assert layout.select({"node": lambda v: v == "east"}) == [layout.physical[1]]


def test_wide_read_keeps_only_the_chosen_columns_and_long_view_matches_pandas(tmp_path: Path) -> None:
    path = tmp_path / "t.parquet"
    frame = _result_frame()
    write_lean_parquet(frame, path)
    layout = lean_parquet_layout(path)
    wide = read_lean_parquet_wide(path, keep=[layout.physical[1]], layout=layout)
    assert wide.columns == ["period", "time", layout.physical[1]]
    assert wide[layout.physical[1]].to_list() == [10.0, 20.0, 30.0]
    long = to_long(read_lean_parquet_wide(path), layout)
    assert long.columns == ["period", "time", "scenario", "unit", "node", "value"]
    expected = (frame.stack(level=[0, 1, 2], future_stack=True).rename("value").reset_index()
                .sort_values(["period", "time", "unit"]).reset_index(drop=True))
    got = long.sort(["period", "time", "unit"]).to_pandas()
    pd.testing.assert_frame_equal(got, expected, check_dtype=False)


def test_long_view_of_a_reduced_frame(tmp_path: Path) -> None:
    """A frame reduced on the wide side (a mean per period) keeps its column keys in the long view."""
    path = tmp_path / "t.parquet"
    write_lean_parquet(_result_frame(), path)
    layout = lean_parquet_layout(path)
    wide = read_lean_parquet_wide(path, layout=layout)
    reduced = wide.group_by("period", maintain_order=True).agg(pl.col(list(layout.physical)).mean())
    long = to_long(reduced, layout)
    # p2020: plant_a (1 + 2) / 2 = 1.5, o'neil (10 + 20) / 2 = 15; p2025: 3 and 30.
    assert long.sort(["period", "unit"]).rows() == [
        ("p2020", "coal", "o'neil", "east", 15.0), ("p2020", "coal", "plant_a", "west", 1.5),
        ("p2025", "coal", "o'neil", "east", 30.0), ("p2025", "coal", "plant_a", "west", 3.0)]


def test_one_column_level_flat_tables_and_unnamed_levels(tmp_path: Path) -> None:
    single = pd.DataFrame({"coal": [5.0, 6.0]}, index=pd.Index(["investment", "fuel"], name="category"))
    single.columns.name = "scenario"
    write_lean_parquet(single, tmp_path / "single.parquet")
    layout = lean_parquet_layout(tmp_path / "single.parquet")
    assert (layout.idx, layout.col, layout.columns, layout.physical) == (
        ("category",), ("scenario",), (("coal",),), ("coal",))
    assert to_long(read_lean_parquet_wide(tmp_path / "single.parquet"), layout).rows() == [
        ("investment", "coal", 5.0), ("fuel", "coal", 6.0)]

    flat = pd.DataFrame({"scenario": ["coal", "coal"], "group": ["g1", "g2"], "node": ["west", "east"]})
    write_lean_parquet(flat, tmp_path / "group_node.parquet")
    layout = lean_parquet_layout(tmp_path / "group_node.parquet")
    assert (layout.idx, layout.col, layout.columns) == (("scenario", "group", "node"), (), ())
    assert read_lean_parquet_wide(tmp_path / "group_node.parquet").rows() == [
        ("coal", "g1", "west"), ("coal", "g2", "east")]

    unnamed = pd.DataFrame({"a": [1.0]}, index=pd.MultiIndex.from_tuples([("p1", "t1")], names=["period", None]))
    write_lean_parquet(unnamed, tmp_path / "unnamed.parquet")
    layout = lean_parquet_layout(tmp_path / "unnamed.parquet")
    assert (layout.idx, layout.index_physical, layout.col) == (
        ("period", "level_1"), ("period", "__index_level_1__"), ("column",))
    assert read_lean_parquet_wide(tmp_path / "unnamed.parquet").columns == ["period", "level_1", "a"]


def test_a_file_written_by_pandas(tmp_path: Path) -> None:
    path = tmp_path / "old.parquet"
    _result_frame().to_parquet(path)
    layout = lean_parquet_layout(path)
    assert not layout.lean
    assert (layout.idx, layout.col) == (("period", "time"), ("scenario", "unit", "node"))
    assert layout.columns == (("coal", "plant_a", "west"), ("coal", "o'neil", "east"))
    assert read_lean_parquet_wide(path, layout=layout)[layout.physical[0]].to_list() == [1.0, 2.0, 3.0]


def test_large_files_are_read_from_the_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "t.parquet"
    write_lean_parquet(_result_frame(), path)
    monkeypatch.setattr(lean_parquet, "IN_MEMORY_MAX_BYTES", 0)
    assert isinstance(lean_parquet._open_source(path), str)
    layout = lean_parquet_layout(path)
    assert read_lean_parquet_wide(path, layout=layout).height == 3


def test_a_partly_written_file_raises_arrow_invalid(tmp_path: Path) -> None:
    path = tmp_path / "t.parquet"
    write_lean_parquet(_result_frame(), path)
    path.write_bytes(path.read_bytes()[:100])
    with pytest.raises(pa.ArrowInvalid):
        lean_parquet_layout(path)
