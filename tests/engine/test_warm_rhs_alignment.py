"""Rolling rows: the ``(n, d, t)`` cross-join helpers are left-major, and a roll never reuses an LP.

``compute_nodeBalance_dt`` & co. build the ``over`` frames of ``model/core.py``; their row order must
follow the sorted inputs.  Under polars 2 a lazy cross join with a small left side emitted right-major
order unless ``maintain_order`` was given.  That once misplaced the positional ``p_inflow`` push of the
warm path onto the wrong node.  The warm path no longer pushes anything: a rolled sub-solve (shifted
``t`` labels) always builds its own LP, which the second test pins at the unit level.
"""
from __future__ import annotations

from types import SimpleNamespace

import polars as pl
import pytest
from polar_high import Param

from flextool.engine.model._pdt_join import (
    compute_nodeBalance_dt,
    compute_nodeState_dt,
    compute_process_indirect_dt,
    compute_pss_dt,
)
from flextool.engine.solve._warm import _apply_warm_updates, _IncompatibleUpdate

NODES = ["heat", "west"]


def _dt(first: int, n: int = 24) -> pl.DataFrame:
    return pl.DataFrame({"d": ["p2020"] * n,
                         "t": [f"t{first + i:04d}" for i in range(n)]})


def _inflow(dt: pl.DataFrame) -> Param:
    """heat 0, west −1000 − hour index (distinct per row)."""
    frame = (pl.DataFrame({"n": NODES})
             .join(dt.with_row_index("i"), how="cross")
             .with_columns(pl.when(pl.col("n") == "west")
                           .then(-1000.0 - pl.col("i"))
                           .otherwise(0.0).alias("value"))
             .select("n", "d", "t", "value"))
    return Param(("n", "d", "t"), frame)


def test_cross_join_helpers_are_left_major() -> None:
    """Sorted inputs give a product sorted by (left cols, right cols)."""
    dt = _dt(1, 30)
    d = SimpleNamespace(
        dt=dt,
        nodeBalance=pl.DataFrame({"n": NODES}),
        nodeState=pl.DataFrame({"n": NODES}),
        process_indirect=pl.DataFrame({"p": ["a", "b"]}),
        process_source_sink=pl.DataFrame(
            {"p": ["a", "b"], "source": ["x", "y"], "sink": ["a", "b"]}))
    for fn in (compute_nodeBalance_dt, compute_nodeState_dt,
               compute_process_indirect_dt, compute_pss_dt):
        out = fn(d)
        assert out.equals(out.sort(out.columns)), fn.__name__


def test_rolled_transition_is_not_reused() -> None:
    """A roll shifts the ``t`` labels of ``dt``: the reuse decision names ``dt`` and the LP is rebuilt."""
    prior = SimpleNamespace(dt=_dt(1), nodeBalance=pl.DataFrame({"n": NODES}),
                            p_inflow=_inflow(_dt(1)))
    nxt_dt = _dt(13)
    nxt = SimpleNamespace(dt=nxt_dt, nodeBalance=pl.DataFrame({"n": NODES}),
                          p_inflow=_inflow(nxt_dt))
    with pytest.raises(_IncompatibleUpdate, match="'dt'"):
        _apply_warm_updates(object(), prior, nxt)
    same = SimpleNamespace(dt=_dt(1), nodeBalance=pl.DataFrame({"n": NODES}),
                           p_inflow=_inflow(_dt(1)))
    assert _apply_warm_updates(object(), prior, same) == 0
