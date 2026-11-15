"""Load-boundary step of ``piecewise_linear`` units (``specs/piecewise_efficiency_final.md`` §3.2).

:func:`apply_piecewise_curves` is called once per load in ``load_flextool`` (single, rolling, synthetic / nested
and stochastic sub-solves), right after ``_apply_db_overrides`` and its completion block.  The model has no rows
for a part-load curve yet, so a scenario with a ``piecewise_linear`` unit stops here with an error naming the units
and whether their curve is one line or several (two or more lines in some period); the input derivation has
already checked the curves and classified the units.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from flextool.engine.input._solve_state import FlexToolConfigError

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource


def piecewise_units(db_reader: "InputSource") -> list[str]:
    """The units whose ``conversion_method`` is ``piecewise_linear`` in the scenario, sorted."""
    from flextool.engine.preprocess.params._derived import _try_param

    cm = _try_param(db_reader, "unit", "conversion_method")
    if cm is None:
        return []
    return sorted(cm.filter(pl.col("value").cast(pl.Utf8) == "piecewise_linear")
                    .get_column("name").cast(pl.Utf8).unique().to_list())


def apply_piecewise_curves(flex_data, db_reader: "InputSource") -> None:
    """Stop a load that has ``piecewise_linear`` units: their model rows are not implemented yet.  The message
    names the units by curve class (:func:`flextool.engine.input._piecewise_curve.piecewise_unit_flags`): two or
    more lines in some period, or one line in every period.  No-op without such units."""
    del flex_data
    units = piecewise_units(db_reader)
    if not units:
        return
    from flextool.engine.preprocess.params._derived import _piecewise_flags

    flags = _piecewise_flags(db_reader)
    multi = set(flags.filter(pl.col("multi_line")).get_column("p").to_list())
    parts = []
    several = [u for u in units if u in multi]
    one = [u for u in units if u not in multi]
    if several:
        parts.append(f"piecewise_linear with more than one line is not implemented yet (unit "
                     f"{', '.join(repr(u) for u in several)})")
    if one:
        parts.append(f"piecewise_linear with one line is not implemented yet (unit "
                     f"{', '.join(repr(u) for u in one)})")
    raise FlexToolConfigError("; ".join(parts) + ".")
