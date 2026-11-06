"""Join / membership helpers across axis columns of different Enum vocabularies.

``join_aligned`` and ``filter_isin`` (``flextool.engine.input._axis_enums``) act only on ``pl.Enum`` keys:
an Enum key on the right is cast to the left key's Enum, a mix of Enum and Utf8 raises, and every other
dtype pair is handed to polars unchanged.  The expected results are the same operation on Utf8 columns
(production axis dtype), so each assertion is relational.
"""

import polars as pl
import pytest

from flextool.engine.input._axis_enums import filter_isin, join_aligned

ENTITY = pl.Enum(["coal", "gas", "north", "south", "wind"])     # entity union: processes and nodes
NODE = pl.Enum(["east", "north", "south"])                      # node axis: "east" is no entity of ENTITY


def _arcs(dtype) -> pl.DataFrame:
    return pl.DataFrame({"p": ["coal", "gas", "wind"], "source": ["north", "south", "wind"]},
                        schema={"p": pl.Utf8, "source": dtype})


def _group_nodes(dtype) -> pl.DataFrame:
    return pl.DataFrame({"g": ["g1", "g1", "g2"], "n": ["north", "east", "south"]},
                        schema={"g": pl.Utf8, "n": dtype})


def _utf8(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(pl.col(c).cast(pl.Utf8) for c, t in frame.schema.items() if isinstance(t, pl.Enum))


def _rows(frame: pl.DataFrame) -> list[tuple]:
    return sorted(frame.rows(), key=lambda r: tuple("" if v is None else str(v) for v in r))


def test_join_utf8_keys_is_the_plain_join():
    left, right = _arcs(pl.Utf8).rename({"source": "n"}), _group_nodes(pl.Utf8)
    assert join_aligned(left, right, on="n", how="inner").equals(left.join(right, on="n", how="inner"))


def test_join_non_enum_dtype_mismatch_is_left_to_polars():
    left = pl.DataFrame({"k": [1, 2, 3]}, schema={"k": pl.Int64})
    right = pl.DataFrame({"k": [2, 3], "v": [0.5, 1.5]}, schema={"k": pl.Int32, "v": pl.Float64})
    assert join_aligned(left, right, on="k").equals(left.join(right, on="k"))


def test_join_enum_vs_enum_inner_matches_the_utf8_join():
    left, right = _arcs(ENTITY).rename({"source": "n"}), _group_nodes(NODE)
    out = join_aligned(left, right, on="n", how="inner")
    expected = _arcs(pl.Utf8).rename({"source": "n"}).join(_group_nodes(pl.Utf8), on="n", how="inner")
    assert out.schema["n"] == ENTITY
    assert _rows(_utf8(out)) == _rows(expected)


def test_join_enum_vs_enum_left_keeps_every_left_row_and_key():
    left, right = _arcs(ENTITY).rename({"source": "n"}), _group_nodes(NODE)
    out = join_aligned(left, right, on="n", how="left")
    assert out.schema["n"] == ENTITY
    assert out.height == left.height
    assert out["n"].null_count() == 0
    assert _rows(out.select("p", "n")) == _rows(left)


def test_join_enum_vs_enum_left_on_right_on_matches_the_utf8_join():
    out = join_aligned(_arcs(ENTITY), _group_nodes(NODE), left_on="source", right_on="n", how="inner")
    expected = _arcs(pl.Utf8).join(_group_nodes(pl.Utf8), left_on="source", right_on="n", how="inner")
    assert out.schema["source"] == ENTITY
    assert _rows(_utf8(out)) == _rows(expected)


def test_join_enum_vs_enum_lazy_frames():
    out = join_aligned(_arcs(ENTITY).rename({"source": "n"}).lazy(), _group_nodes(NODE).lazy(), on="n")
    assert isinstance(out, pl.LazyFrame)
    expected = _arcs(pl.Utf8).rename({"source": "n"}).join(_group_nodes(pl.Utf8), on="n")
    assert _rows(_utf8(out.collect())) == _rows(expected)


@pytest.mark.parametrize("how", ["full", "cross"])
def test_join_full_and_cross_raise(how):
    with pytest.raises(ValueError, match=how):
        join_aligned(_arcs(ENTITY).rename({"source": "n"}), _group_nodes(NODE), on="n", how=how)


@pytest.mark.parametrize("left_dtype, right_dtype", [(pl.Utf8, NODE), (ENTITY, pl.Utf8)])
def test_join_enum_vs_utf8_raises(left_dtype, right_dtype):
    with pytest.raises(TypeError, match="n"):
        join_aligned(_arcs(left_dtype).rename({"source": "n"}), _group_nodes(right_dtype), on="n", how="left")


def test_isin_enum_vs_enum_matches_the_utf8_membership():
    frame = _arcs(ENTITY)
    values = _group_nodes(NODE)["n"]
    expected = _arcs(pl.Utf8).filter(pl.col("source").is_in(_group_nodes(pl.Utf8)["n"].implode()))
    out = filter_isin(frame, "source", values)
    assert out.schema["source"] == ENTITY
    assert _rows(_utf8(out)) == _rows(expected)


def test_isin_utf8_is_the_plain_filter():
    frame, values = _arcs(pl.Utf8), _group_nodes(pl.Utf8)["n"]
    assert filter_isin(frame, "source", values).equals(frame.filter(pl.col("source").is_in(values.implode())))


@pytest.mark.parametrize("frame_dtype, values_dtype", [(pl.Utf8, NODE), (ENTITY, pl.Utf8)])
def test_isin_enum_vs_utf8_raises(frame_dtype, values_dtype):
    with pytest.raises(TypeError, match="source"):
        filter_isin(_arcs(frame_dtype), "source", _group_nodes(values_dtype)["n"])
