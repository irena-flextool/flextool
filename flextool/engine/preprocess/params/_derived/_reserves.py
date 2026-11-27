"""§3.13 — reserves: the active reserve relationships, ``process_reserve_upDown_node_active`` and
``prundt``."""
from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from flextool.engine.input._axis_enums import alias_to_axis, rename_to_axis
from flextool.engine.preprocess.params._direct import (
    pdtReserve_upDown_group_reservation_from_source,
)

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource

from flextool.engine.preprocess.params._derived._scope import _try_entities, _try_param


# ---------------------------------------------------------------------------
# §3.13 reserve helpers (narrow gate-on-equality overlays)
# ---------------------------------------------------------------------------


def _reserve_active_relationships(source: "InputSource"
                                       ) -> pl.DataFrame | None:
    """Union the unit-side and connection-side reserve relationships
    into a single ``(p, r, ud, n)`` frame.  Returns None when no reserve
    relationships exist on the source.
    """
    parts: list[pl.LazyFrame] = []
    for ec in ("reserve__upDown__unit__node",
               "reserve__upDown__connection__node"):
        ents = _try_entities(source, ec)
        if ents is None or ents.height == 0:
            continue
        cols = ents.columns
        # Schema: [reserve, upDown, unit_or_connection, node]
        # Column names from SpineDbReader use the dim class names.
        rename = {}
        for c in cols:
            if c == "reserve":
                rename[c] = "r"
            elif c == "upDown":
                rename[c] = "ud"
            elif c == "node":
                rename[c] = "n"
            elif c in ("unit", "connection"):
                rename[c] = "p"
        if "r" not in [rename.get(c, c) for c in cols] \
                or "ud" not in [rename.get(c, c) for c in cols]:
            continue
        parts.append(ents.lazy().pipe(rename_to_axis, rename).select("p", "r", "ud", "n"))
    if not parts:
        return None
    return pl.concat(parts).unique().sort("p", "r", "ud", "n").collect()


def process_reserve_upDown_node_active_from_source(source: "InputSource",
                                                       ) -> pl.DataFrame | None:
    """§3.13.x — Projection of (p, r, ud, n) tuples that are active.

    The simple case is the union of the unit-side / connection-side
    reserve relationships filtered to those with non-zero
    ``reliability``.  When ``reliability`` is missing on the source,
    fall back to the relationship membership (default reliability=1.0
    per the Spine schema).
    """
    # Build a typed empty frame for the "reserve feature active but no
    # active (p, r, ud, n) tuples" fall-through.  ``_reserve._check``
    # requires a DataFrame (not None) when ``reserve_upDown_group`` is
    # populated; downstream uses guard on ``height > 0`` before joining.
    def _empty_pruna() -> pl.DataFrame:
        return pl.DataFrame(schema={
            "p": pl.Utf8, "r": pl.Utf8, "ud": pl.Utf8, "n": pl.Utf8,
        })

    rel = _reserve_active_relationships(source)
    if rel is None or rel.height == 0:
        return _empty_pruna()
    # If reliability is parametrised, keep rows with reliability > 0.
    rel_dfs: list[pl.LazyFrame] = []
    for ec in ("reserve__upDown__unit__node",
               "reserve__upDown__connection__node"):
        df = _try_param(source, ec, "reliability")
        if df is None or df.height == 0:
            continue
        cols = df.columns
        rename: dict[str, str] = {}
        for c in cols:
            if c == "reserve":
                rename[c] = "r"
            elif c == "upDown":
                rename[c] = "ud"
            elif c == "node":
                rename[c] = "n"
            elif c in ("unit", "connection"):
                rename[c] = "p"
        rel_dfs.append(df.lazy().pipe(rename_to_axis, rename).select(
            "p", "r", "ud", "n",
            pl.col("value").cast(pl.Float64).alias("rel"),
        ))
    if not rel_dfs:
        return rel  # Defaults to "all relationships active".
    rel_lf = (pl.concat(rel_dfs)
                  .filter(pl.col("rel") > 0)
                  .select("p", "r", "ud", "n"))
    out = rel_lf.unique().sort("p", "r", "ud", "n").collect()
    return out if out.height > 0 else _empty_pruna()


# ---------------------------------------------------------------------------
# §3.13 — prundt (residual reserve coverage)
# ---------------------------------------------------------------------------


def prundt_from_source(
    source: "InputSource",
    active_solve: str | None,
    dt: pl.DataFrame | None,
) -> pl.DataFrame | None:
    """``prundt`` (p, r, ud, n, d, t): the reserve relationships of the reserves with a requirement × dt.

    A relationship ``(p, r, ud, n)`` (``process_reserve_upDown_node_active_from_source``) is kept when its
    ``(r, ud)`` has a requirement (:func:`~flextool.engine.preprocess._reserve_methods.demanded_reserves`):
    a ``reservation`` whose sum over the groups of ``(r, ud)`` and the solve's (d, t) is not 0, or a
    dynamic / n-1 group driven by an ``increase_reserve_ratio`` (of a relationship at a node of the group, or of the group itself) /
    ``large_failure_ratio``.  The solve_data producer
    (``emit/_reserve_sets._compute_process_reserve_active``) applies the same rule, so ``prundt`` and the
    ratio sets the model reads agree.  A reserve without a requirement gets no ``v_reserve`` columns
    (no all-zero columns in ``process__reserve__upDown__node__dt``).

    Returns an empty typed frame when dt, the relationships or every requirement are absent.
    """
    from flextool.engine.preprocess._reserve_methods import demanded_reserves
    from flextool.engine.preprocess.params import _projection as _proj

    # Empty typed prundt for the "feature active but no requirement" fall-through: ``_reserve._check``
    # expects a DataFrame (not None) when the reserve feature is active.  Downstream guards on
    # ``height == 0`` before touching values.
    def _empty_prundt() -> pl.DataFrame:
        return pl.DataFrame(schema={
            "p": pl.Utf8,
            "r": pl.Utf8,
            "ud": pl.Utf8,
            "n": pl.Utf8,
            "d": pl.Utf8,
            "t": pl.Utf8,
        })

    if dt is None or dt.height == 0:
        return _empty_prundt()
    pruna = process_reserve_upDown_node_active_from_source(source)
    if pruna is None or pruna.height == 0:
        return _empty_prundt()

    def _labels(df: pl.DataFrame, cols: tuple[str, ...]) -> list[tuple[str, ...]]:
        return [tuple(str(x) for x in row)
                for row in df.select(pl.col(c).cast(pl.Utf8) for c in cols).iter_rows()]

    # 1. Reservation: (r, ud) whose reservation, summed over its groups and this solve's (d, t), is not 0
    #    (the solve-resolved values; a stochastic-time reservation contributes its fold) — the emit's rule.
    reservation_pairs: set[tuple[str, str]] = set()
    res = pdtReserve_upDown_group_reservation_from_source(source, period_filter=dt)
    if res is not None and res.frame.height > 0:
        # A value without a d / t axis (a constant) holds on every (d, t) of the solve.
        axes = [c for c in ("d", "t") if c in res.frame.columns]
        dt_keys = dt.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8)).unique().lazy()
        vals = res.frame.lazy().select(*(pl.col(c).cast(pl.Utf8) for c in ("r", "ud", *axes)),
                                       pl.col("value").cast(pl.Float64, strict=False).fill_null(0.0))
        on_dt = vals.join(dt_keys, on=axes, how="inner") if axes else vals.join(dt_keys, how="cross")
        sums = (on_dt
                .group_by("r", "ud").agg(pl.col("value").sum())
                .filter(pl.col("value") != 0.0)
                .collect())
        reservation_pairs = set(_labels(sums, ("r", "ud")))
    # 2./3. Dynamic and n-1 groups and the ratios that drive them.
    group_methods: list[tuple[str, ...]] = []
    for part in ("dynamic", "n_1"):
        group_methods += _labels(_proj.reserve_upDown_group_method(source, part), ("r", "ud", "g", "method"))
    group_nodes = _labels(_proj.group_node(source), ("g", "n"))
    rel = _reserve_active_relationships(source)
    rel_keys = set(_labels(rel, ("p", "r", "ud", "n"))) if rel is not None else set()
    ratios: dict[tuple[str, ...], dict[str, float]] = {}
    for ec, ent in (("reserve__upDown__unit__node", "unit"), ("reserve__upDown__connection__node", "connection")):
        for name in ("increase_reserve_ratio", "large_failure_ratio"):
            df = _try_param(source, ec, name)
            if df is None:
                continue
            rows = df.select(*(pl.col(c).cast(pl.Utf8) for c in (ent, "reserve", "upDown", "node")),
                             pl.col("value").cast(pl.Float64, strict=False).fill_null(0.0))
            for p, r, ud, n, v in rows.iter_rows():
                if (p, r, ud, n) in rel_keys:
                    ratios.setdefault((p, r, ud, n), {})[name] = float(v)
    group_ratios: dict[tuple[str, ...], float] = {}
    gr = _try_param(source, "reserve__upDown__group", "increase_reserve_ratio")
    if gr is not None:
        rows = gr.select(*(pl.col(c).cast(pl.Utf8) for c in ("reserve", "upDown", "group")),
                         pl.col("value").cast(pl.Float64, strict=False).fill_null(0.0))
        group_ratios = {(r, ud, g): float(v) for r, ud, g, v in rows.iter_rows()}
    demanded = demanded_reserves(
        reservation_pairs=reservation_pairs,
        group_methods=group_methods,
        group_nodes=group_nodes,
        relationship_ratios=ratios,
        group_ratios=group_ratios,
    )
    if not demanded:
        return _empty_prundt()
    active_ru = pl.DataFrame({"__r": [r for r, _ in sorted(demanded)], "__ud": [u for _, u in sorted(demanded)]},
                             schema={"__r": pl.Utf8, "__ud": pl.Utf8})
    pruna_lf = (pruna.lazy()
                .with_columns(pl.col("r").cast(pl.Utf8).alias("__r"), pl.col("ud").cast(pl.Utf8).alias("__ud"))
                .join(active_ru.lazy(), on=["__r", "__ud"], how="semi")
                .drop("__r", "__ud"))
    # Defensive re-cast: ensure d/t are canonical Enum on the cross-join
    # output (pruna_lf carries Enum p / r / ud / n).
    out = (pruna_lf
              .join(dt.lazy()
                       .select(alias_to_axis("d", "d"),
                               alias_to_axis("t", "t")),
                    how="cross")
              .select("p", "r", "ud", "n", "d", "t")
              .sort("p", "r", "ud", "n", "d", "t")
              .collect())
    if out.height == 0:
        return _empty_prundt()
    return out
