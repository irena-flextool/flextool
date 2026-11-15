"""Part-load curve of a ``piecewise_linear`` unit -> the lines of its fuel use (no FlexData dependency).

A curve is the points ``(ℓ_i, η_i)``, ``i = 0..N−1``, of ``unit.piecewise_linear_efficiency`` (load share ->
efficiency, sorted by load, ``ℓ_{N−1} = 1``).  The fuel per unit of capacity at load ``ℓ`` is ``F(ℓ) = ℓ/η(ℓ)``,
linear between the points; each segment ``k = 1..N−1`` is the line ``F = α_k·ℓ + β_k`` through
``(ℓ_{k−1}, F_{k−1})`` and ``(ℓ_k, F_k)``::

    β_k = ℓ_{k−1} · ℓ_k · (1/η_{k−1} − 1/η_k) · (1/(ℓ_k − ℓ_{k−1}))      input per unit at zero output
    α_k = 1/η_k − β_k · (1/ℓ_k)                                           incremental input per output

(``ℓ_k = 1``: ``β = ℓ_{k−1} · (1/η_{k−1} − 1/η_k) · (1/(1 − ℓ_{k−1}))``, ``α = 1/η_k − β``: the float operations
of the ``min_load_efficiency`` section and slope, ``params/_derived._mle_section`` / ``_mle_slope``.)  Only
multiplications and ``1/x``, so a curve gives the same bits whatever way it was authored; ``β = 0`` exactly when
``η_{k−1} = η_k``.  ``sign(β_k) = sign(η_k − η_{k−1})``.

Line sets per curve (design ``specs/piecewise_efficiency_final.md`` §2.1), after merging the inner points where
two segments have the same slope (relative 1e-9):

* online form (the unit has online variables): the segments, plus the origin line ``α = 1/η₀, β = 0`` when the
  efficiency falls from the first point to the second (``η₀ > η₁``);
* capacity form (no online variables): the origin line through the best point ``α = 1/max η, β = 0``, plus the
  segments with ``β < 0``.

The functions take the long frame ``(p, d, load, eff)`` (``d`` null for a curve without a period level) that the
input derivation and the engine reader both build, so both classify a unit the same way.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import polars as pl

#: A load key above 1 by at most this is read as 1.
LOAD_ONE_TOLERANCE = 1e-9
#: Two segments whose slopes differ by at most this (relative) are one line: the inner point is dropped.
MERGE_RTOL = 1e-9
#: The slope of a segment may fall from the one before by at most this (relative) and the curve is still convex.
CONVEX_RTOL = 1e-9
#: Origin line token of the ``seg`` axis.
ORIGIN = "0"

Point = tuple[float, float]


def normalise_load(load: float) -> float:
    """``load``, with a value above 1 by at most :data:`LOAD_ONE_TOLERANCE` read as 1."""
    return 1.0 if 1.0 < load <= 1.0 + LOAD_ONE_TOLERANCE else load


def segment(p0: Point, p1: Point) -> tuple[float, float]:
    """``(α, β)`` of the segment from ``p0 = (ℓ_{k−1}, η_{k−1})`` to ``p1 = (ℓ_k, η_k)`` (module docstring)."""
    l0, e0 = p0
    l1, e1 = p1
    if l1 == 1.0:
        beta = l0 * (1.0 / e0 - 1.0 / e1) * (1.0 / (1.0 - l0))
        return 1.0 / e1 - beta, beta
    beta = l0 * l1 * (1.0 / e0 - 1.0 / e1) * (1.0 / (l1 - l0))
    return 1.0 / e1 - beta * (1.0 / l1), beta


def segments(points: Sequence[Point]) -> list[tuple[float, float]]:
    """``[(α_k, β_k)]`` for ``k = 1..N−1`` of the sorted *points*."""
    return [segment(points[k - 1], points[k]) for k in range(1, len(points))]


def _same_slope(a: float, b: float) -> bool:
    return abs(b - a) <= MERGE_RTOL * max(abs(a), abs(b))


def merge_points(points: Sequence[Point]) -> list[Point]:
    """*points* without the inner points where the two segments around them have the same slope (relative
    :data:`MERGE_RTOL`); the slopes are recomputed from the kept points after each drop."""
    kept = list(points)
    while len(kept) > 2:
        lines = segments(kept)
        drop = next((k for k in range(1, len(lines)) if _same_slope(lines[k - 1][0], lines[k][0])), None)
        if drop is None:
            break
        del kept[drop]
    return kept


@dataclass(frozen=True)
class Line:
    """One line ``input ≥ slope · output + intercept · capacity`` of a curve; ``seg`` is its ``seg`` axis token."""

    seg: str
    slope: float
    intercept: float


def online_lines(points: Sequence[Point]) -> list[Line]:
    """The online-form lines of a sorted curve (module docstring), after merging."""
    kept = merge_points(points)
    if len(kept) == 1:
        return [Line(ORIGIN, 1.0 / kept[0][1], 0.0)]
    out = [Line(str(k + 1), a, b) for k, (a, b) in enumerate(segments(kept))]
    if kept[0][1] > kept[1][1]:
        out.insert(0, Line(ORIGIN, 1.0 / kept[0][1], 0.0))
    return out


def capacity_lines(points: Sequence[Point]) -> list[Line]:
    """The capacity-form lines of a sorted curve (module docstring), after merging."""
    kept = merge_points(points)
    best = max(e for _l, e in kept)
    out = [Line(ORIGIN, 1.0 / best, 0.0)]
    out += [Line(str(k + 1), a, b) for k, (a, b) in enumerate(segments(kept)) if b < 0.0]
    return out


def n_lines(points: Sequence[Point]) -> int:
    """The number of online-form lines of a sorted curve."""
    return len(online_lines(points))


def rises_first(points: Sequence[Point]) -> bool:
    """The efficiency rises from the first point to the second (the curve burns fuel at zero output)."""
    return len(points) > 1 and points[0][1] < points[1][1]


@dataclass(frozen=True)
class PiecewiseFlags:
    """How a unit's whole curve (all periods) is modelled (design §2.2).

    * ``multi_line``: some period's curve has two or more lines (``process_piecewise``);
    * ``needs_online``: multi-line and the efficiency rises from the first point to the second in some period;
    * ``single_line_rising``: one line in every period, with a positive intercept in some (a
      ``min_load_efficiency``-like unit);
    * ``single_line_flat``: one line with zero intercept in every period (a constant-efficiency unit).
    """

    multi_line: bool
    needs_online: bool
    single_line_rising: bool
    single_line_flat: bool

    @property
    def auto_startup(self) -> bool:
        """The curve burns fuel at zero output in some period: the unit needs online variables."""
        return self.needs_online or self.single_line_rising


def piecewise_unit_flags(curves: Iterable[Sequence[Point]]) -> PiecewiseFlags:
    """The :class:`PiecewiseFlags` of a unit from its sorted curves (one per period, or the one curve)."""
    curves = list(curves)
    lines = [online_lines(c) for c in curves]
    multi = any(len(ls) >= 2 for ls in lines)
    rising = any(ls[0].intercept > 0.0 for ls in lines if len(ls) == 1)
    return PiecewiseFlags(
        multi_line=multi,
        needs_online=multi and any(rises_first(c) for c in curves),
        single_line_rising=not multi and rising,
        single_line_flat=not multi and not rising,
    )


def lower_hull(points: Sequence[Point]) -> list[Point]:
    """The points of a sorted curve on the lower convex hull of ``(ℓ_i, F_i)``, ``F = ℓ/η`` (always the first and
    the last point): the nearest convex curve, with efficiencies at or above the given ones."""
    hull: list[tuple[float, float, Point]] = []
    for point in points:
        x, f = point[0], point[0] / point[1]
        while len(hull) >= 2:
            (x1, f1, _), (x2, f2, _) = hull[-2], hull[-1]
            if (x2 - x1) * (f - f1) - (f2 - f1) * (x - x1) <= 0.0:
                hull.pop()
            else:
                break
        hull.append((x, f, point))
    return [p for _x, _f, p in hull]


def hull_efficiency(hull: Sequence[Point], load: float) -> float:
    """The efficiency of the curve *hull* at *load* (fuel linear between its points)."""
    for (l0, e0), (l1, e1) in zip(hull, hull[1:]):
        if l0 <= load <= l1:
            f0, f1 = l0 / e0, l1 / e1
            return load / (f0 + (f1 - f0) * (load - l0) / (l1 - l0))
    raise ValueError(f"load {load} outside the curve {list(hull)}")


# ---------------------------------------------------------------------------
# Frame API: the long frame (p, d, load, eff)
# ---------------------------------------------------------------------------

#: Schema of the long curve frame both sides build.
CURVE_SCHEMA = {"p": pl.Utf8, "d": pl.Utf8, "load": pl.Float64, "eff": pl.Float64}


def curves_by_unit(frame: pl.DataFrame) -> dict[str, dict[str | None, list[Point]]]:
    """``{p: {d: sorted points}}`` of a long curve frame (``d`` None for a curve without a period level)."""
    out: dict[str, dict[str | None, list[Point]]] = {}
    for p, d, load, eff in frame.select(pl.col("p").cast(pl.Utf8), pl.col("d").cast(pl.Utf8), "load",
                                        "eff").iter_rows():
        out.setdefault(p, {}).setdefault(d, []).append((load, eff))
    return {p: {d: sorted(pts) for d, pts in by_d.items()} for p, by_d in out.items()}


def unit_flags(frame: pl.DataFrame) -> dict[str, PiecewiseFlags]:
    """``{p: PiecewiseFlags}`` of every unit of a long curve frame (all its periods)."""
    return {p: piecewise_unit_flags(by_d.values()) for p, by_d in curves_by_unit(frame).items()}


def unit_flags_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """:func:`unit_flags` as a frame ``(p, multi_line, needs_online, single_line_rising, single_line_flat,
    auto_startup)``, sorted by ``p``."""
    flags = unit_flags(frame)
    names = sorted(flags)
    return pl.DataFrame({
        "p": names,
        "multi_line": [flags[p].multi_line for p in names],
        "needs_online": [flags[p].needs_online for p in names],
        "single_line_rising": [flags[p].single_line_rising for p in names],
        "single_line_flat": [flags[p].single_line_flat for p in names],
        "auto_startup": [flags[p].auto_startup for p in names],
    }, schema={"p": pl.Utf8, "multi_line": pl.Boolean, "needs_online": pl.Boolean,
               "single_line_rising": pl.Boolean, "single_line_flat": pl.Boolean, "auto_startup": pl.Boolean})


def curve_lines(frame: pl.DataFrame, online: Iterable[str]) -> pl.DataFrame:
    """The lines ``(p, d, seg, slope, intercept)`` of every curve of a long curve frame: the online form for the
    units in *online*, the capacity form for the others; in ``(p, d)`` order of the frame's sorted keys, each
    curve's lines in ``seg`` order of :func:`online_lines` / :func:`capacity_lines`."""
    online = set(online)
    rows: list[tuple[str, str | None, str, float, float]] = []
    for p, by_d in sorted(curves_by_unit(frame).items()):
        form = online_lines if p in online else capacity_lines
        for d in sorted(by_d, key=lambda x: (x is not None, x or "")):
            rows.extend((p, d, ln.seg, ln.slope, ln.intercept) for ln in form(by_d[d]))
    return pl.DataFrame(rows, schema={"p": pl.Utf8, "d": pl.Utf8, "seg": pl.Utf8, "slope": pl.Float64,
                                      "intercept": pl.Float64}, orient="row")
