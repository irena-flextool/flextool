"""Warm rolling reuse: the positional RHS push must land on the right rows.

A rolled sub-solve with the same LP shape reuses the previous roll's
:class:`polar_high.WarmProblem` and pushes the new ``p_inflow`` into
``nodeBalance_eq`` as a positional vector.  The constraint's rows are in
``over`` sorted by its columns (the add_cstr determinism wrapper), so the
vector must follow that order whatever order ``nodeBalance × dt`` comes
out in.  Under polars 2 a lazy cross join with a small left side emits
right-major order unless ``maintain_order`` is given: every other
``(n, t)`` inflow went onto the wrong node (west demand 0 in every other
hour of roll 1 of ``test_delayed_input_share_rolling_csv_chain``).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import polars as pl
from polar_high import Param

from flextool.engine_polars._pdt_join import (
    compute_nodeBalance_dt,
    compute_nodeState_dt,
    compute_process_indirect_dt,
    compute_pss_dt,
)
from flextool.engine_polars._warm import _apply_warm_updates

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


class _RecordingWarm:
    """Captures the vector ``_apply_warm_updates`` pushes."""

    def __init__(self) -> None:
        self.pushed: dict[str, np.ndarray] = {}

    def update_rhs(self, name: str, vec) -> None:
        self.pushed[name] = np.asarray(vec)


def test_warm_inflow_push_follows_sorted_over_rows() -> None:
    """The pushed vector is ``−p_inflow`` in ``(n, d, t)``-sorted order,
    even when ``nodeBalance`` itself is not sorted."""
    prior = SimpleNamespace(dt=_dt(1), nodeBalance=pl.DataFrame({"n": NODES}),
                            p_inflow=_inflow(_dt(1)))
    nxt_dt = _dt(13)
    nxt = SimpleNamespace(dt=nxt_dt,
                          nodeBalance=pl.DataFrame({"n": NODES[::-1]}),
                          p_inflow=_inflow(nxt_dt))
    warm = _RecordingWarm()
    assert _apply_warm_updates(warm, prior, nxt) == 1
    expected = -(nxt.p_inflow.frame.sort("n", "d", "t")["value"].to_numpy())
    np.testing.assert_array_equal(warm.pushed["nodeBalance_eq"], expected)
