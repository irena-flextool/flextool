"""Warm reuse: a change of the stochastic branch weights forces a cold rebuild.

``pdt_branch_weight`` / ``pd_branch_weight`` enter the objective only through composite factors
(``op_factor``, the startup and storage reference-price terms, the commodity ladder, and the invest
costs under recourse), which the warm path cannot update in place.  Two rolls with different weights
have the same LP shape, so both Params are in ``_WARM_PARAMS_DEFERRED``: a difference raises
``_IncompatibleUpdate`` (cold rebuild), while the same values on shifted ``t`` labels do not.
"""
from __future__ import annotations

from types import SimpleNamespace

import polars as pl
import pytest
from polar_high import Param

from flextool.engine.solve._warm import _apply_warm_updates, _IncompatibleUpdate


def _pdt(first: int, weights: list[float]) -> Param:
    rows = [(f"p1_{b}", f"t{first + k:04d}", w) for b, w in zip("ab", weights) for k in range(3)]
    return Param(("d", "t"), pl.DataFrame(rows, schema={"d": pl.Utf8, "t": pl.Utf8, "value": pl.Float64},
                                          orient="row"))


def _pd(weights: list[float]) -> Param:
    return Param(("d",), pl.DataFrame({"d": ["p1_a", "p1_b"], "value": weights}))


@pytest.mark.parametrize("field", ["pdt_branch_weight", "pd_branch_weight"])
def test_branch_weight_change_forces_cold_rebuild(field) -> None:
    prior = SimpleNamespace(pdt_branch_weight=_pdt(1, [0.4, 0.6]), pd_branch_weight=_pd([0.4, 0.6]))
    nxt = SimpleNamespace(pdt_branch_weight=_pdt(25, [0.4, 0.6]), pd_branch_weight=_pd([0.4, 0.6]))
    # Same weights, t labels shifted by the roll: warm reuse stays possible.
    assert _apply_warm_updates(object(), prior, nxt) == 0
    setattr(nxt, field, _pdt(25, [0.7, 0.3]) if field == "pdt_branch_weight" else _pd([0.7, 0.3]))
    with pytest.raises(_IncompatibleUpdate, match=field):
        _apply_warm_updates(object(), prior, nxt)
