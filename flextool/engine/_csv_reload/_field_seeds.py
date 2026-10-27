"""Field seeds of a dump reloaded without a DB reader.

In production ``load_flextool`` always has a DB reader, and the DB overlay (``_apply_db_overrides``) writes some
fields that the loader's own CSV-shaped read used to seed only for a reload.  c24 stage 4 removes those
reload-only seeds from the loader; each removed seed whose field the dump carries moves here.  The loader calls
:func:`apply_dump_field_seeds` through its private ``_dump_hooks.field_seeds`` hook, at the point where the overlay
would run, and only when there is no reader (a reload with ``db_url`` runs the overlay instead).

:data:`SEEDED_FIELDS` names every FlexData field :func:`apply_dump_field_seeds` writes;
``tests/csv_reload/test_dump_seeds.py`` checks that it writes no other field and compares each one with the
production FlexData of the dumped run.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
from polar_high import Param

from flextool.engine.input._axis_enums import rename_to_axis

#: The variable-cost index sets and ``p_pssdt_varCost`` (S4-2): production builds them with
#: ``_direct.apply_varcost_params`` from the DB-direct per-side Params.
_VARCOST_SETS: tuple[str, ...] = (
    "pssdt_varCost_noEff",
    "pssdt_varCost_eff_unit_source",
    "pssdt_varCost_eff_unit_sink",
    "pssdt_varCost_eff_connection",
)

#: The FlexData fields :func:`apply_dump_field_seeds` writes (each stage-4 deletion adds its fields).
SEEDED_FIELDS: tuple[str, ...] = (*_VARCOST_SETS, "p_pssdt_varCost")

_PSSDT_RENAME = {"process": "p", "period": "d", "time": "t"}


def _varcost_set(provider: object, name: str) -> pl.DataFrame | None:
    """``solve_data/<name>`` as ``[p, source, sink, d, t]``; None when the dump has no rows."""
    key = f"solve_data/{name}"
    if not provider.has(key):
        return None
    df = provider.get(key)
    if df.height == 0:
        return None
    return df.pipe(rename_to_axis, _PSSDT_RENAME).select("p", "source", "sink", "d", "t")


def _varcost_param(provider: object) -> Param | None:
    """``solve_data/pdtProcess__source__sink__dt_varCost`` as ``p_pssdt_varCost`` (rows with value ≠ 0)."""
    key = "solve_data/pdtProcess__source__sink__dt_varCost"
    if not provider.has(key):
        return None
    df = provider.get(key)
    if df.height == 0:
        return None
    sliced = (df.pipe(rename_to_axis, _PSSDT_RENAME)
                .with_columns(value=pl.col("value").cast(pl.Float64, strict=False).fill_null(0.0))
                .filter(pl.col("value") != 0.0))
    if sliced.height == 0:
        return None
    return Param(("p", "source", "sink", "d", "t"), sliced.select("p", "source", "sink", "d", "t", "value"))


def apply_dump_field_seeds(flex_data: object, sd: Path, inp: Path, provider: object) -> None:
    """Seed :data:`SEEDED_FIELDS` on ``flex_data`` from the seeded ``provider``.

    ``sd`` / ``inp`` are the dump's ``solve_data`` / ``input`` folders (the path keys of the provider frames).

    * Variable cost (the loader's ``_load_varcost`` seed before S4-2, moved here): the four index sets from the
      run's emitted ``pssdt_varCost_*.csv`` (copied into the dump by ``copy_runner_meta``) and
      ``p_pssdt_varCost`` from the dumped ``pdtProcess__source__sink__dt_varCost.csv``.  Rebuilding them with
      ``_direct.apply_varcost_params`` from the per-side seeds is not possible: the dump's ``pdtProcess.csv``
      carries no ``other_operational_cost`` slice, so ``p_pdt_varCost_process`` (connection costs) is None on a
      reload.  Like the emit, the sets have no period × time tier (#74).  Written only when the run has process
      arcs (``process_source_sink``), as the loader's seed was.
    """
    del sd, inp
    if getattr(flex_data, "process_source_sink", None) is None:
        return
    for name in _VARCOST_SETS:
        setattr(flex_data, name, _varcost_set(provider, name))
    flex_data.p_pssdt_varCost = _varcost_param(provider)


__all__ = ["SEEDED_FIELDS", "apply_dump_field_seeds"]
