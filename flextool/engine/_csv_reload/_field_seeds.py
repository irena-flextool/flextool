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
from flextool.engine.preprocess._invest_seeds import read_invest_set
from flextool.engine.preprocess.flex_data import _read_wide_e_d

#: The variable-cost index sets and ``p_pssdt_varCost`` (S4-2): production builds them with
#: ``_direct.apply_varcost_params`` from the DB-direct per-side Params.
_VARCOST_SETS: tuple[str, ...] = (
    "pssdt_varCost_noEff",
    "pssdt_varCost_eff_unit_source",
    "pssdt_varCost_eff_unit_sink",
    "pssdt_varCost_eff_connection",
)

#: The per-entity invest / divest caps (S4-3): production writes them in ``_direct.apply_direct_params_a`` (totals)
#: and ``apply_direct_params_b`` (per period).
_INVEST_CAP_TOTALS: tuple[str, ...] = ("e_invest_max_total", "e_divest_max_total")
_INVEST_CAP_PERIODS: tuple[str, ...] = ("ed_invest_max_period", "ed_divest_max_period")
_INVEST_MIN_TOTALS: tuple[str, ...] = ("e_invest_min_total", "e_divest_min_total")

#: The FlexData fields :func:`apply_dump_field_seeds` writes (each stage-4 deletion adds its fields).
SEEDED_FIELDS: tuple[str, ...] = (
    *_VARCOST_SETS, "p_pssdt_varCost",
    "p_co2_content", *_INVEST_CAP_TOTALS, *_INVEST_CAP_PERIODS, *_INVEST_MIN_TOTALS,
)

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


def _co2_content(provider: object) -> Param:
    """The ``co2_content`` rows of ``input/p_commodity`` as ``p_co2_content`` ``(c,)``."""
    p_comm = provider.get("input/p_commodity")
    return Param(("c",),
        p_comm.filter(pl.col("commodityParam") == "co2_content")
              .pipe(rename_to_axis, {"commodity": "c", "p_commodity": "value"})
              .select("c", "value"))


def _e_param(provider: object, name: str) -> Param | None:
    """``solve_data/<name>`` (``entity, value``) as an ``(e,)`` Param; None when absent, empty or malformed."""
    key = f"solve_data/{name}"
    if not provider.has(key):
        return None
    df = provider.get(key)
    if df.height == 0 or "entity" not in df.columns or "value" not in df.columns:
        return None
    return Param(("e",),
                 df.pipe(rename_to_axis, {"entity": "e"})
                   .with_columns(value=pl.col("value").cast(pl.Float64, strict=False).fill_null(0.0))
                   .select("e", "value"))


def _e_d_param(sd: Path, provider: object, name: str) -> Param | None:
    """``solve_data/<name>`` (long or wide ``(e, d)`` CSV) as an ``(e, d)`` Param; None when absent or empty."""
    if not provider.has(f"solve_data/{name}"):
        return None
    df = _read_wide_e_d(sd / f"{name}.csv", provider=provider)
    if df.height == 0:
        return None
    return Param(("e", "d"), df.select("e", "d", "value"))


def _has_invest_rows(sd: Path, provider: object) -> bool:
    """True when ``ed_invest`` or ``ed_divest`` has a row (the loader's invest gate on a reader-less load)."""
    return any(read_invest_set(sd, name, "e", provider=provider).height > 0 for name in ("ed_invest", "ed_divest"))


def _apply_co2_and_invest_seeds(flex_data: object, sd: Path, provider: object) -> None:
    """The CO2-content and invest / divest cap seeds (the loader's reads before S4-3, moved here unchanged)."""
    if (getattr(flex_data, "flow_from_co2_priced", None) is not None
            or getattr(flex_data, "p_co2_max_period", None) is not None
            or getattr(flex_data, "p_co2_max_total", None) is not None):
        flex_data.p_co2_content = _co2_content(provider)
    if _has_invest_rows(sd, provider):
        for name in _INVEST_CAP_TOTALS:
            setattr(flex_data, name, _e_param(provider, name))
        for name in _INVEST_CAP_PERIODS:
            setattr(flex_data, name, _e_d_param(sd, provider, name))
    for name in _INVEST_MIN_TOTALS:
        setattr(flex_data, name, _e_param(provider, name))


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
    * CO2 content and the invest / divest caps (``_load_co2_price``'s ``input/p_commodity`` read and its
      CO2-cap fallback in ``load_flextool``, ``_load_invest``'s ``e_*_max_total`` / ``ed_*_max_period`` reads,
      ``_load_cumulative_invest``'s ``e_*_min_total`` reads before S4-3, moved here): ``p_co2_content`` from the
      dumped ``input/p_commodity.csv`` when the run has CO2-priced flows or a CO2 cap; the max caps from the run's
      ``solve_data`` CSVs when ``ed_invest`` or ``ed_divest`` has a row; the min totals always.
    """
    del inp
    _apply_co2_and_invest_seeds(flex_data, sd, provider)
    if getattr(flex_data, "process_source_sink", None) is None:
        return
    for name in _VARCOST_SETS:
        setattr(flex_data, name, _varcost_set(provider, name))
    flex_data.p_pssdt_varCost = _varcost_param(provider)


__all__ = ["SEEDED_FIELDS", "apply_dump_field_seeds"]
