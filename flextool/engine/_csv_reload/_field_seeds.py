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
from flextool.engine.preprocess.params._derived import (
    process_delayed__duration_from_weight,
    process_group_inside_nonSync_from_flex,
)

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

#: The group sets of the capacity margin, inertia and non-synchronous limits (S4-4): production writes them in
#: ``_projection.apply_projection_params`` (``_AUTHORITATIVE_SETS``).
_GROUP_SETS: tuple[str, ...] = ("groupCapacityMargin", "groupInertia", "groupNonSync")

#: The delay tables (S4-5): production writes the three together in ``_derived.apply_delay_params``.
_DELAY_TABLES: tuple[str, ...] = ("dtt__delay_duration", "p_process_delay_weight", "process_delayed__duration")

#: The FlexData fields :func:`apply_dump_field_seeds` writes (each stage-4 deletion adds its fields).
#: ``p_profile_value`` is not here: the c24 stage-4 follow-up restored its seed to the production loader's
#: ``_load_profiles`` (``flex_data.py``), which runs whether or not there is a DB reader, so the reload needs
#: no separate copy.
SEEDED_FIELDS: tuple[str, ...] = (
    *_VARCOST_SETS, "p_pssdt_varCost",
    "p_co2_content", *_INVEST_CAP_TOTALS, *_INVEST_CAP_PERIODS, *_INVEST_MIN_TOTALS,
    *_GROUP_SETS, "process_sink_nonSync", "process_group_inside_nonSync", "process_minload",
    *_DELAY_TABLES,
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
            or getattr(flex_data, "flow_to_co2_priced", None) is not None
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


def _frame_or_none(provider: object, key: str) -> pl.DataFrame | None:
    """The provider's frame under ``key``; None when absent or without rows."""
    if not provider.has(key):
        return None
    df = provider.get(key)
    return df if df.height > 0 else None


def _group_set(provider: object, name: str) -> pl.DataFrame | None:
    """``input/<name>`` (one column of group names) as a ``(g,)`` set; None when absent or empty."""
    df = _frame_or_none(provider, f"input/{name}")
    return None if df is None else df.rename({df.columns[0]: "g"}).select("g").unique()


def _process_sink_nonsync(provider: object) -> pl.DataFrame | None:
    """``solve_data/process__sink_nonSync`` as ``(p, sink)``; None when absent or empty."""
    df = _frame_or_none(provider, "solve_data/process__sink_nonSync")
    return None if df is None else df.rename({"process": "p"}).select("p", "sink").unique()


def _process_minload(provider: object) -> pl.DataFrame:
    """``solve_data/process_minload`` as a ``(p,)`` set; an empty ``(p,)`` frame when the dump has no file."""
    key = "solve_data/process_minload"
    if not provider.has(key):
        return pl.DataFrame(schema={"p": pl.Utf8})
    return provider.get(key).pipe(rename_to_axis, {"process": "p"})


def _apply_group_and_minload_seeds(flex_data: object, provider: object) -> None:
    """The group sets, the non-synchronous arcs and groups' internal transfers, and the min-load units (S4-4).

    The three group sets and ``process_sink_nonSync`` are ``_group_slack.load_data``'s reads before S4-4 and
    ``process_minload`` ``_load_online``'s read (only when the run has online processes), moved here unchanged.
    ``process_group_inside_nonSync`` is the production rule (``process_group_inside_nonSync_from_flex``) on the
    reloaded ``process_source_sink`` / ``group_node`` and the seeded ``groupNonSync``.
    """
    for name in _GROUP_SETS:
        setattr(flex_data, name, _group_set(provider, name))
    flex_data.process_sink_nonSync = _process_sink_nonsync(provider)
    flex_data.process_group_inside_nonSync = process_group_inside_nonSync_from_flex(flex_data)
    if getattr(flex_data, "process_online", None) is not None:
        flex_data.process_minload = _process_minload(provider)


def _delay_shift_table(provider: object) -> pl.DataFrame | None:
    """``solve_data/dtt__delay_duration`` as ``(d, t_source, t_sink, td)`` with Float64 ``td`` (the text column may
    read "1" or "1.0"); None when absent or empty."""
    raw = _frame_or_none(provider, "solve_data/dtt__delay_duration")
    if raw is None:
        return None
    rename_map = {old: new for old, new in (("period", "d"), ("time_source", "t_source"),
                                           ("time_sink", "t_sink"), ("delay_duration", "td"))
                  if old in raw.columns}
    return raw.rename(rename_map).select("d", "t_source", "t_sink", pl.col("td").cast(pl.Float64))


def _delay_weight(provider: object) -> Param | None:
    """``solve_data/p_process_delay_weight`` as a ``(p, td)`` Param with Float64 ``td`` and value; None when absent
    or empty."""
    raw = _frame_or_none(provider, "solve_data/p_process_delay_weight")
    if raw is None:
        return None
    rename_map = {old: new for old, new in (("process", "p"), ("delay_duration", "td")) if old in raw.columns}
    return Param(("p", "td"), raw.rename(rename_map).select(
        "p", pl.col("td").cast(pl.Float64), pl.col("value").cast(pl.Float64)))


def _apply_delay_seeds(flex_data: object, provider: object) -> None:
    """The delay tables (S4-5).

    ``_delay.load_data``'s reads before S4-5, moved here unchanged with their gate: only when a process is
    delayed.  ``process_delayed__duration`` is the production rule (``process_delayed__duration_from_weight``) on
    the seeded weights, so its ``td`` labels match the weights exactly.

    ``p_profile_value`` used to be seeded alongside these (S4-5); the c24 stage-4 follow-up restored its seed to
    the production loader instead (``flex_data.py::_load_profiles``), so it no longer needs a copy here.
    """
    delayed = getattr(flex_data, "process_delayed", None)
    if delayed is not None and delayed.height > 0:
        flex_data.dtt__delay_duration = _delay_shift_table(provider)
        flex_data.p_process_delay_weight = _delay_weight(provider)
        flex_data.process_delayed__duration = process_delayed__duration_from_weight(flex_data.p_process_delay_weight)


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
    * Group sets and min-load units (S4-4, :func:`_apply_group_and_minload_seeds`): ``groupCapacityMargin`` /
      ``groupInertia`` / ``groupNonSync`` from the dump's ``input/<field>.csv``, ``process_sink_nonSync`` from
      ``solve_data/process__sink_nonSync.csv``, ``process_minload`` from the dumped ``process_minload.csv`` (an
      empty set without the file) when the run has online processes, and ``process_group_inside_nonSync`` derived
      from the reloaded topology by the production rule.
    * Delay tables (S4-5, :func:`_apply_delay_seeds`): ``dtt__delay_duration`` / ``p_process_delay_weight`` from
      the emitted CSVs and ``process_delayed__duration`` derived from the weights, when a process is delayed.
      ``p_profile_value`` is not seeded here: the production loader's own ``pdtProfile.csv`` read
      (``flex_data.py::_load_profiles``) covers the reload too.
    """
    del inp
    _apply_co2_and_invest_seeds(flex_data, sd, provider)
    _apply_group_and_minload_seeds(flex_data, provider)
    _apply_delay_seeds(flex_data, provider)
    if getattr(flex_data, "process_source_sink", None) is None:
        return
    for name in _VARCOST_SETS:
        setattr(flex_data, name, _varcost_set(provider, name))
    flex_data.p_pssdt_varCost = _varcost_param(provider)


__all__ = ["SEEDED_FIELDS", "apply_dump_field_seeds"]
