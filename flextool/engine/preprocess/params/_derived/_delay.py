"""§3.15 — the delay of units: the delay distributions, ``p_process_delay_weight``,
``process_delayed__duration``, ``dtt__delay_duration`` and :func:`apply_delay_params`."""
from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from polar_high import Param

from flextool.engine.input._axis_enums import schema_dtype

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource

from flextool.engine.preprocess.params._derived._scope import _enums, _try_param


# ---------------------------------------------------------------------------
# §3.15 — Delay
# ---------------------------------------------------------------------------


def _delay_distributions_from_source(
    source: "InputSource",
) -> tuple[dict[tuple[str, str], float],
            set[tuple[str, str]],
            set[str]] | None:
    """Read ``unit.delay`` from the source and classify into:

    * ``weighted`` — (p, td) → weight when value is a 1d_map.  These are
      multi-duration distributions.
    * ``single`` — (p, td) when value is a scalar.  Single delay duration.
    * ``processes`` — set of all delayed processes (union of both).

    Returns ``None`` if no delays are defined.

    ``connection.delay`` is not read: connection delay is not implemented
    and is ignored (``input_derivation`` warns about it), so a connection
    is modelled exactly as without a delay.

    flextool's input writer (input_writer.py:638-648) emits the same
    distinction via ``filter_in_type``: 1d_map → ``p_process_delay_weighted``,
    scalar → ``process_delay_single``.
    """
    weighted: dict[tuple[str, str], float] = {}
    single: set[tuple[str, str]] = set()
    processes: set[str] = set()
    found_any = False
    for ec in ("unit",):
        df = _try_param(source, ec, "delay")
        if df is None or df.height == 0:
            continue
        found_any = True
        cols = df.columns
        # Distinguish 1d_map shape (entity, key, value — three cols) from
        # scalar shape (entity, value — two cols).  The SpineDbReader
        # uses generic key names for 1d_map (typically `constraint` or
        # similar from the shape registry); we detect by column count.
        if len(cols) >= 3:
            # 1d_map: name, <key>, value.
            key_col = next((c for c in cols
                              if c not in ("name", "value")),
                            None)
            if key_col is None:
                continue
            for row in df.iter_rows(named=True):
                p = row["name"]
                kv = row[key_col]
                v = row["value"]
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                # The CSV writer formats the delay duration via
                # repr(float(x)), so the string key matches when we
                # round-trip via float.
                try:
                    td = repr(float(kv))
                except (TypeError, ValueError):
                    td = str(kv)
                weighted[(p, td)] = fv
                processes.add(p)
        else:
            # Scalar entry — value IS the delay duration.
            for row in df.iter_rows(named=True):
                p = row["name"]
                v = row["value"]
                if v is None:
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                # 0 and negative durations don't activate delay.
                if fv <= 0.0:
                    continue
                td = repr(fv)
                single.add((p, td))
                processes.add(p)
    if not found_any:
        return None
    return weighted, single, processes


def p_process_delay_weight_from_source(
    source: "InputSource",
) -> "Param | None":
    """Per-(p, td) delay weight — audit §3.15.2.

    Algorithm — mirror of
    ``preprocessing/process_arc_unions.py:write_p_process_delay_weight:2105-2141``::

        p_process_delay_weight[p, td] = 1.0 if (p, td) ∈ process_delay_single
                                         else p_process_delay_weighted[p, td]

    The set ``process_delayed__duration`` is the union of the two sources;
    every row in that set produces an output value.

    Returns ``None`` when no delays are defined (no fixture rows on
    either source).
    """
    info = _delay_distributions_from_source(source)
    if info is None:
        return None
    weighted, single, _ = info
    rows: list[tuple[str, float, float]] = []
    # Order: weighted entries first, then single-only (mirrors flextool's
    # union which preserves first-occurrence order across the two sources).
    seen: set[tuple[str, str]] = set()
    for (p, td), v in weighted.items():
        rows.append((p, float(td), 1.0 if (p, td) in single else v))
        seen.add((p, td))
    for (p, td) in single:
        if (p, td) not in seen:
            rows.append((p, float(td), 1.0))
    if not rows:
        return None
    out = pl.DataFrame(rows, schema={"p": schema_dtype(_enums, "p"),
                                       "td": pl.Float64,
                                       "value": pl.Float64},
                       orient="row")
    return Param(("p", "td"), out)


def process_delayed__duration_from_weight(
    p_process_delay_weight: "Param | None",
) -> pl.DataFrame | None:
    """``process_delayed__duration`` ``(p, td)`` — the delay durations of
    each delayed process (flextool.mod ``process_delay_weighted ∪
    process_delay_single``).

    Built from the keys of ``p_process_delay_weight`` (non-zero weights),
    i.e. from the same ``_delay_distributions_from_source`` read as the
    weights and ``dtt__delay_duration``: a scalar ``delay`` gives
    ``(p, delay)`` and a map gives one row per duration.  ``td`` therefore
    has exactly the dtype and values of ``p_process_delay_weight.td`` and
    ``dtt__delay_duration.td`` (Float64), which ``_delay.delayed_input_expr``
    joins on.  (A separate map-only reader used to produce this set: a
    scalar ``delay`` got no rows, the delayed unit lost all its input
    terms and was forced to zero output.)

    Returns ``None`` when no process is delayed.
    """
    if p_process_delay_weight is None or p_process_delay_weight.frame.height == 0:
        return None
    out = (p_process_delay_weight.frame
           .filter(pl.col("value") != 0.0)
           .select("p", "td")
           .sort("p", "td"))
    return out if out.height > 0 else None


def apply_delay_params(flex_data: object, source: "InputSource") -> None:
    """Set the three delay frames — ``dtt__delay_duration``,
    ``p_process_delay_weight`` and ``process_delayed__duration`` — from one
    read of ``unit.delay`` over ``flex_data.dt`` (a connection ``delay``
    is ignored).

    Solve-agnostic: :func:`apply_derived_g` calls it, and so does the
    synthetic (rolling) sub-solve path in ``flex_data._db_overrides._apply_db_overrides``,
    which skips the derived passes.  Producing all three together keeps
    the ``td`` labels identical across them.
    """
    flex_data.dtt__delay_duration = dtt__delay_duration_from_source(
        source, getattr(flex_data, "dt", None))
    flex_data.p_process_delay_weight = p_process_delay_weight_from_source(
        source)
    flex_data.process_delayed__duration = (
        process_delayed__duration_from_weight(flex_data.p_process_delay_weight))


def dtt__delay_duration_from_source(
    source: "InputSource",
    dt: pl.DataFrame | None,
) -> pl.DataFrame | None:
    """Per-(d, t_source, t_sink, td) delay shift map — audit §3.15.1.

    For each delay duration ``td`` and each (d, t_source) in dt, produce
    one row (d, t_source, t_sink, td) where ``t_sink = t_source + td``
    timesteps within the period.  Wraps at period boundaries::

        if k + td_int < len(time_steps):
            t_sink = time_steps[k + td_int]
        else:
            t_sink = time_steps[k - len(time_steps) + td_int]

    flextool casts ``td`` through ``int(float(delay_duration))`` so the
    csv-string "4.0" becomes shift index 4.

    Inputs: ``dt`` (period-time index) + the union of all delay durations
    from ``unit.delay`` (a connection ``delay`` is ignored).

    Returns ``None`` when no delays are defined or dt is empty.
    """
    if dt is None or dt.height == 0:
        return None
    info = _delay_distributions_from_source(source)
    if info is None:
        return None
    weighted, single, _ = info
    # Distinct delay duration set (string keys, as written by the CSV).
    durations: set[str] = set()
    for (_, td) in weighted:
        durations.add(td)
    for (_, td) in single:
        durations.add(td)
    if not durations:
        return None
    # Build per-period ordered timestep lists from dt.
    dt_sorted = (dt.lazy().sort("d", "t")
                    .group_by("d", maintain_order=True)
                    .agg(pl.col("t").alias("steps"))
                    .collect())
    rows: list[tuple[str, str, str, float]] = []
    for row in dt_sorted.iter_rows(named=True):
        d = row["d"]
        steps: list[str] = list(row["steps"])
        n = len(steps)
        if n == 0:
            continue
        for td in sorted(durations):  # a set: its order varies with the hash seed
            try:
                td_int = int(float(td))
                td_float = float(td)
            except (TypeError, ValueError):
                continue
            for k, t_src in enumerate(steps):
                if k + td_int < n:
                    t_sink = steps[k + td_int]
                else:
                    t_sink = steps[k - n + td_int]
                rows.append((d, t_src, t_sink, td_float))
    if not rows:
        return None
    out = pl.DataFrame(rows, schema={"d": schema_dtype(_enums, "d"),
                                       "t_source": schema_dtype(_enums, "t_source"),
                                       "t_sink": schema_dtype(_enums, "t_sink"),
                                       "td": pl.Float64},
                       orient="row")
    return out
