"""Regression — per-entity scalar value mixed with period-Map values in
the same entity class.

When one unit carries a *constant* ``existing`` capacity while a sibling
unit in the same class carries a *period-Map* ``existing``, the Spine
reader returns a single frame with one shared index column (named ``x``
by Spine's silent default) where the constant unit's row has a NULL
index::

    name         x      value
    wind_scalar  null   31.5     <- scalar, broadcast across periods
    wind_map     p2024  10.8     <- period-Map row
    wind_map     p2025  22.8

``_per_entity_param_lf`` must classify the null-index row as a SCALAR
(``is_scalar=True``) so ``_resolve_per_period_lf`` broadcasts it across
the period universe.  The historical bug classified the whole frame by
the *presence* of the index column, so the null-index scalar became an
explicit ``(e, d=null)`` row that never joined the period grid and was
silently filled with 0 — surfacing downstream as a zeroed-out existing
capacity (e.g. negative VRE curtailment = ``potential - flow`` with
``potential`` computed from a 0 capacity).

Both mirror implementations of the helper are covered.
"""
from __future__ import annotations

import polars as pl
import pytest

from flextool.engine_polars import _derived_existing as de
from flextool.engine_polars import _derived_npv as dn


def _mixed_existing_source():
    """InMemoryReader whose ``unit.existing`` mixes a scalar (null index)
    with a period-Map, mirroring ``SpineDbReader.parameter`` output."""
    from flextool.engine_polars._inmemory_reader import InMemoryReader

    frame = pl.DataFrame(
        {
            "name": ["wind_scalar", "wind_map", "wind_map", "wind_map"],
            "x": [None, "p2024", "p2025", "p2026"],
            "value": [31.5, 10.8, 22.8, 22.8],
        }
    )
    entities = {"unit": pl.DataFrame({"name": ["wind_scalar", "wind_map"]})}
    return InMemoryReader(entities, {("unit", "existing"): frame})


@pytest.mark.parametrize("module", [de, dn], ids=["derived_existing", "derived_npv"])
def test_null_index_row_is_scalar(module):
    """A null-index row must be flagged ``is_scalar=True``; the Map rows
    must stay ``is_scalar=False`` with their explicit period."""
    src = _mixed_existing_source()
    out = module._per_entity_param_lf(src, "existing").collect()

    scalar_row = out.filter(pl.col("e") == "wind_scalar")
    assert scalar_row.height == 1
    assert scalar_row["is_scalar"][0] is True
    assert scalar_row["d"][0] is None
    assert scalar_row["value"][0] == pytest.approx(31.5)

    map_rows = out.filter(pl.col("e") == "wind_map")
    assert map_rows.height == 3
    assert not any(map_rows["is_scalar"].to_list())
    assert set(map_rows["d"].to_list()) == {"p2024", "p2025", "p2026"}


@pytest.mark.parametrize("module", [de, dn], ids=["derived_existing", "derived_npv"])
def test_scalar_broadcasts_across_period_grid(module):
    """The scalar must resolve to its constant value at EVERY period in
    the grid, not be zero-filled.  This is the load-bearing assertion:
    the bug zeroed ``wind_scalar`` at every period."""
    src = _mixed_existing_source()
    per = module._per_entity_param_lf(src, "existing")

    grid = pl.LazyFrame(
        {
            "e": ["wind_scalar", "wind_scalar", "wind_map", "wind_map"],
            "d": ["p2025", "p2026", "p2025", "p2026"],
        }
    )
    resolved = module._resolve_per_period_lf(per, grid, fill=0.0).collect()

    scalar_res = resolved.filter(pl.col("e") == "wind_scalar").sort("d")
    assert scalar_res["value"].to_list() == pytest.approx([31.5, 31.5])

    map_res = resolved.filter(pl.col("e") == "wind_map").sort("d")
    assert map_res["value"].to_list() == pytest.approx([22.8, 22.8])


def test_pre_existing_broadcasts_scalar_to_all_periods():
    """End-to-end through ``p_entity_pre_existing_lf``: the scalar unit's
    pre-existing capacity must be present (non-zero) at the active
    period, matching the period-Map unit's treatment."""
    src = _mixed_existing_source()
    pre = de.p_entity_pre_existing_lf(
        src, active_solve="y2025", period_in_use=["p2025"]
    ).collect()

    scalar_res = pre.filter(pl.col("e") == "wind_scalar")
    assert scalar_res.height == 1
    assert scalar_res["value"][0] == pytest.approx(31.5)

    map_res = pre.filter(pl.col("e") == "wind_map")
    assert map_res.height == 1
    assert map_res["value"][0] == pytest.approx(22.8)
