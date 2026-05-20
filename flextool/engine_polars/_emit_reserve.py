"""Writer-port Phase 2 (sub-dispatch 5) — reserve calculated params.

Native port of
``flextool.flextoolrunner.preprocessing.reserve_calc_params`` (legacy
337 LOC).  Fired per-solve from
``flextool.flextoolrunner.preprocessing.solve_time.run`` at batches 43,
44 and 49.

Three public entry points (mirroring the legacy module):

* :func:`write_pdtReserve_upDown_group` — mod L1319.  4-branch hourly
  resolution for reserve groups, emitting ``pdtReserve_upDown_group.csv``
  ``(reserve, upDown, group, param, period, time, value)``.
* :func:`write_process_reserve_upDown_node_active_and_prundt` — mod
  L1321-1322.  Filters ``process_reserve_upDown_node`` by nonzero summed
  reservation across the matching ``reserve__upDown__group`` × ``dt``
  cross product, then emits ``prundt = process_reserve_upDown_node_active
  × dt``.
* :func:`write_process_reserve_filters_and_reliability` — mod L1655 /
  L1660-1668.  Emits the reliability fallback (default 1) and two
  ``> 0`` filter sets (``increase_reserve_ratio`` /
  ``large_failure_ratio``) plus the ``process_large_failure`` projection.

Output CSVs (7 total):

* ``pdtReserve_upDown_group.csv``
* ``process_reserve_upDown_node_active.csv``
* ``prundt.csv``
* ``p_process_reserve_upDown_node_reliability.csv``
* ``process_reserve_upDown_node_increase_reserve_ratio.csv``
* ``process_reserve_upDown_node_large_failure_ratio.csv``
* ``process_large_failure.csv``

Reuse note
----------

``flextool.engine_polars._reserve`` consumes these CSVs as RHS / domain
inputs (``prundt`` for v_reserve, ``pdtReserve_upDown_group_reservation``
for the balance constraint, etc.) but does not derive them — it is a
constraint-builder that runs *after* preprocessing.  No code is shared
across the port boundary; this module mirrors the legacy emitter's
shape verbatim for byte-identical parity.

Float values formatted with ``repr(float(v))`` for byte-identical
parity with the legacy emitter.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from flextool.engine_polars._emit_provider_io import (
    _emit,
    _provider_key,
    _provider_open,
)


# ---------------------------------------------------------------------------
# Canonical emitter — mirrors the ``_write(df, path)`` idiom in the other
# patched writer modules.  All public ``write_*`` entry points in this
# module funnel their derived frames through this helper so
# :mod:`._flex_data_accumulator` can capture them via its monkey-patch.
#
# I/O contract: legacy emitters here use ``path.open("w") + fh.write(...)``
# with plain ``\n`` line terminators (no CRLF), so the default polars
# ``write_csv`` line ending matches byte-for-byte.  Float cells are
# pre-rendered with ``repr(float(v))`` by the ``derive_*`` builders, so
# the underlying frame is all-``Utf8`` and polars does no extra numeric
# formatting at write time.
# ---------------------------------------------------------------------------


def _to_utf8_frame(
    headers: tuple[str, ...],
    rows: list[tuple],
) -> pl.DataFrame:
    """Build an all-``Utf8`` polars frame from a header tuple + row list.

    Each cell is taken verbatim when already a string, otherwise via
    ``str(v)``.  All ``derive_*`` builders in this module pre-render
    float cells through ``repr(float(v))`` so byte parity with the
    legacy emit holds.
    """
    cols: dict[str, list[str]] = {h: [] for h in headers}
    for row in rows:
        for h, v in zip(headers, row):
            cols[h].append(v if isinstance(v, str) else str(v))
    return pl.DataFrame(cols, schema={h: pl.Utf8 for h in headers})


# ---------------------------------------------------------------------------
# Tiny CSV I/O — same shape as _emit_inflow_scaling helpers.  The legacy
# emitter writes plain text with ``repr(v)`` and the parity surface is
# small enough that a dict-of-rows pass is the simplest correct path.
# ---------------------------------------------------------------------------


def _read_singles(path: Path,
                  *, provider: "object | None" = None) -> list[str]:
    """First-column header-less reader (skip header row)."""
    seeded = _provider_open(provider, _provider_key(path), path)
    if seeded is None:
        return []
    out: list[str] = []
    with seeded as fh:
        next(fh, None)
        for line in fh:
            parts = line.rstrip("\n").split(",")
            if parts and parts[0]:
                out.append(parts[0])
    return out


def _read_pairs(path: Path,
                *, provider: "object | None" = None) -> list[tuple[str, str]]:
    """First-two-column header-less reader (skip header row)."""
    seeded = _provider_open(provider, _provider_key(path), path)
    if seeded is None:
        return []
    out: list[tuple[str, str]] = []
    with seeded as fh:
        next(fh, None)
        for line in fh:
            parts = line.rstrip("\n").split(",")
            if len(parts) >= 2 and parts[0] and parts[1]:
                out.append((parts[0], parts[1]))
    return out


def _read_n_col(path: Path, n: int,
                *, provider: "object | None" = None) -> list[tuple[str, ...]]:
    """First-n-column header-less reader; rows with any empty key skipped."""
    seeded = _provider_open(provider, _provider_key(path), path)
    if seeded is None:
        return []
    out: list[tuple[str, ...]] = []
    with seeded as fh:
        next(fh, None)
        for line in fh:
            parts = line.rstrip("\n").split(",")
            if len(parts) >= n and all(parts[i] for i in range(n)):
                out.append(tuple(parts[:n]))
    return out


def _read_pairs_to_dict(path: Path, key_col: int,
                        *, provider: "object | None" = None,
                        ) -> dict[str, list[str]]:
    """Two-col CSV → {row[key_col]: [row[other_col], ...]} preserving order."""
    out: dict[str, list[str]] = {}
    seeded = _provider_open(provider, _provider_key(path), path)
    if seeded is None:
        return out
    other_col = 1 - key_col
    with seeded as fh:
        next(fh, None)
        for line in fh:
            parts = line.rstrip("\n").split(",")
            if len(parts) >= 2 and parts[0] and parts[1]:
                out.setdefault(parts[key_col], []).append(parts[other_col])
    return out


def _read_pbt_reserve(
    path: Path,
    *, provider: "object | None" = None,
) -> dict[tuple[str, str, str, str, str, str, str], float]:
    """``pbt_reserve__upDown__group.csv`` →
    {(r, ud, g, param, branch, ts, t): float}.  Malformed / non-numeric
    rows silently skipped (matches legacy)."""
    out: dict[tuple[str, str, str, str, str, str, str], float] = {}
    seeded = _provider_open(provider, _provider_key(path), path)
    if seeded is None:
        return out
    with seeded as fh:
        next(fh, None)
        for line in fh:
            parts = line.rstrip("\n").split(",")
            if len(parts) >= 8 and all(parts[i] for i in range(7)):
                try:
                    out[(parts[0], parts[1], parts[2], parts[3],
                         parts[4], parts[5], parts[6])] = float(parts[7])
                except ValueError:
                    continue
    return out


def _read_pt_reserve(
    path: Path,
    *, provider: "object | None" = None,
) -> dict[tuple[str, str, str, str, str], float]:
    """``pt_reserve__upDown__group.csv`` →
    {(r, ud, g, param, t): float}."""
    out: dict[tuple[str, str, str, str, str], float] = {}
    seeded = _provider_open(provider, _provider_key(path), path)
    if seeded is None:
        return out
    with seeded as fh:
        next(fh, None)
        for line in fh:
            parts = line.rstrip("\n").split(",")
            if len(parts) >= 6 and all(parts[i] for i in range(5)):
                try:
                    out[(parts[0], parts[1], parts[2],
                         parts[3], parts[4])] = float(parts[5])
                except ValueError:
                    continue
    return out


def _read_p_reserve(
    path: Path,
    *, provider: "object | None" = None,
) -> dict[tuple[str, str, str, str], float]:
    """``p_reserve__upDown__group.csv`` → {(r, ud, g, param): float}."""
    out: dict[tuple[str, str, str, str], float] = {}
    seeded = _provider_open(provider, _provider_key(path), path)
    if seeded is None:
        return out
    with seeded as fh:
        next(fh, None)
        for line in fh:
            parts = line.rstrip("\n").split(",")
            if len(parts) >= 5 and all(parts[i] for i in range(4)):
                try:
                    out[(parts[0], parts[1], parts[2],
                         parts[3])] = float(parts[4])
                except ValueError:
                    continue
    return out


# Reserve param table fixed across the writers — flextool_base.dat L183
# defines only the ``reservation`` reserveTimeParam.  Default per
# reserveParam_defaults (mod L493) is 0 for reservation.
_RESERVE_TIME_PARAMS: tuple[str, ...] = ("reservation",)
_RESERVATION_DEFAULT: float = 0.0


# ---------------------------------------------------------------------------
# write_pdtReserve_upDown_group  (mod L1319)
# ---------------------------------------------------------------------------


def _resolve_pdtReserve(
    r: str, ud: str, g: str, param: str, d: str, t: str,
    *,
    pbt: dict[tuple[str, str, str, str, str, str, str], float],
    pt: dict[tuple[str, str, str, str, str], float],
    p: dict[tuple[str, str, str, str], float],
    ts_for_d: dict[str, list[str]],
    tb_for_d: dict[str, list[str]],
    pe_for_d: dict[str, list[str]],
    g_stoch: bool,
) -> float:
    """4-branch fallback mirroring ``pdtReserve_upDown_group``'s mod body.

    Branches mirror ``pdtProcess``'s branches 1, 2, 4, 5 (no period axis,
    no def1 — and the scalar branch returns the table's default of 0
    for ``reservation`` when the (r, ud, g) row is missing).
    """
    # Branch 1: stochastic + outer-d's ts/tb.
    if g_stoch:
        ts_list = ts_for_d.get(d, ())
        tb_list = tb_for_d.get(d, ())
        total = 0.0
        hit = False
        for tb in tb_list:
            for ts in ts_list:
                v = pbt.get((r, ud, g, param, tb, ts, t))
                if v is not None:
                    total += v
                    hit = True
        if hit:
            return total
    # Branch 2: parent period pe of d, tb from solve_branch[pe], ts from
    # period__time_first[d].
    ts_list = ts_for_d.get(d, ())
    pe_list = pe_for_d.get(d, ())
    if pe_list and ts_list:
        total = 0.0
        hit = False
        for pe in pe_list:
            for tb in tb_for_d.get(pe, ()):
                for ts in ts_list:
                    v = pbt.get((r, ud, g, param, tb, ts, t))
                    if v is not None:
                        total += v
                        hit = True
        if hit:
            return total
    # Branch 3: time axis.
    v = pt.get((r, ud, g, param, t))
    if v is not None:
        return v
    # Branch 4: scalar (with table default 0 for ``reservation``).
    v = p.get((r, ud, g, param))
    if v is not None:
        return v
    return _RESERVATION_DEFAULT


def derive_pdtReserve_upDown_group(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    """Build the canonical ``pdtReserve_upDown_group`` frame
    ``(reserve, upDown, group, param, period, time, value)``.

    Value column is pre-rendered with ``repr(float(v))`` for byte
    parity with the legacy emitter.
    """
    pbt = _read_pbt_reserve(
        input_dir / "pbt_reserve__upDown__group.csv", provider=provider,
    )
    pt = _read_pt_reserve(
        input_dir / "pt_reserve__upDown__group.csv", provider=provider,
    )
    p = _read_p_reserve(
        input_dir / "p_reserve__upDown__group.csv", provider=provider,
    )

    ts_for_d = _read_pairs_to_dict(
        solve_data_dir / "first_timesteps.csv", 0, provider=provider,
    )
    tb_for_d = _read_pairs_to_dict(
        solve_data_dir / "solve_branch__time_branch.csv", 0, provider=provider,
    )
    pe_for_d = _read_pairs_to_dict(
        solve_data_dir / "period__branch.csv", 1, provider=provider,
    )
    groups_stoch = frozenset(
        _read_singles(
            input_dir / "groupIncludeStochastics.csv", provider=provider,
        )
    )

    rug = _read_n_col(
        solve_data_dir / "reserve__upDown__group.csv", 3, provider=provider,
    )
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv", provider=provider)

    rows: list[tuple] = []
    for (r, ud, g) in rug:
        g_stoch = g in groups_stoch
        for param in _RESERVE_TIME_PARAMS:
            for (d, t) in dt:
                val = _resolve_pdtReserve(
                    r, ud, g, param, d, t,
                    pbt=pbt, pt=pt, p=p,
                    ts_for_d=ts_for_d,
                    tb_for_d=tb_for_d,
                    pe_for_d=pe_for_d,
                    g_stoch=g_stoch,
                )
                rows.append((r, ud, g, param, d, t, repr(float(val))))
    return _to_utf8_frame(
        ("reserve", "upDown", "group", "param", "period", "time", "value"),
        rows,
    )


def emit_pdtReserve_upDown_group(
    input_dir: Path, solve_data_dir: Path,
    *, provider,
) -> None:
    """Provider-emitting twin of :func:`write_pdtReserve_upDown_group`."""
    _emit(
        provider, "solve_data/pdtReserve_upDown_group.csv",
        derive_pdtReserve_upDown_group(
            input_dir, solve_data_dir, provider=provider,
        ),
    )


# ---------------------------------------------------------------------------
# write_process_reserve_upDown_node_active_and_prundt  (mod L1321-1322)
# ---------------------------------------------------------------------------


def _compute_process_reserve_active(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> tuple[list[tuple[str, str, str, str]], list[tuple[str, str]]]:
    """Shared computation for the active/prundt pair — returns the
    active ``(p, r, ud, n)`` list and the ``dt`` pair list."""
    pdt_reserve: dict[tuple[str, str, str, str, str, str], float] = {}
    pdt_path = solve_data_dir / "pdtReserve_upDown_group.csv"
    pdt_seeded = _provider_open(provider, _provider_key(pdt_path), pdt_path)
    if pdt_seeded is not None:
        with pdt_seeded as fh:
            next(fh, None)
            for line in fh:
                parts = line.rstrip("\n").split(",")
                if len(parts) >= 7 and all(parts[i] for i in range(6)):
                    try:
                        pdt_reserve[(
                            parts[0], parts[1], parts[2],
                            parts[3], parts[4], parts[5],
                        )] = float(parts[6])
                    except ValueError:
                        continue

    rug_by_ru: dict[tuple[str, str], list[str]] = {}
    for r, ud, g in _read_n_col(
        solve_data_dir / "reserve__upDown__group.csv", 3, provider=provider,
    ):
        rug_by_ru.setdefault((r, ud), []).append(g)

    dt = _read_pairs(solve_data_dir / "steps_in_use.csv", provider=provider)
    prun = _read_n_col(
        input_dir / "process__reserve__upDown__node.csv", 4, provider=provider,
    )

    active_rows: list[tuple[str, str, str, str]] = []
    for (p, r, ud, n) in prun:
        groups = rug_by_ru.get((r, ud), ())
        total = 0.0
        for g in groups:
            for (d, t) in dt:
                total += pdt_reserve.get(
                    (r, ud, g, "reservation", d, t), 0.0,
                )
        if total != 0.0:
            active_rows.append((p, r, ud, n))
    return active_rows, dt


def derive_process_reserve_upDown_node_active(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    """Build the ``process_reserve_upDown_node_active`` frame
    ``(process, reserve, upDown, node)`` — entries from
    ``process_reserve_upDown_node`` whose summed reservation across
    the matching ``reserve__upDown__group × dt`` cross product is
    nonzero.
    """
    active_rows, _dt = _compute_process_reserve_active(
        input_dir, solve_data_dir, provider=provider,
    )
    return _to_utf8_frame(
        ("process", "reserve", "upDown", "node"), active_rows,
    )


def derive_prundt(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    """Build the ``prundt`` frame —
    ``process_reserve_upDown_node_active × dt``.
    """
    active_rows, dt = _compute_process_reserve_active(
        input_dir, solve_data_dir, provider=provider,
    )
    rows: list[tuple[str, str, str, str, str, str]] = []
    for (p, r, ud, n) in active_rows:
        for (d, t) in dt:
            rows.append((p, r, ud, n, d, t))
    return _to_utf8_frame(
        ("process", "reserve", "upDown", "node", "period", "time"), rows,
    )


def emit_process_reserve_upDown_node_active_and_prundt(
    input_dir: Path, solve_data_dir: Path,
    *, provider,
) -> None:
    """Provider-emitting twin of
    :func:`write_process_reserve_upDown_node_active_and_prundt`."""
    active_rows, dt = _compute_process_reserve_active(
        input_dir, solve_data_dir, provider=provider,
    )
    _emit(
        provider,
        "solve_data/process_reserve_upDown_node_active.csv",
        _to_utf8_frame(
            ("process", "reserve", "upDown", "node"), active_rows,
        ),
    )
    prundt_rows: list[tuple[str, str, str, str, str, str]] = []
    for (p, r, ud, n) in active_rows:
        for (d, t) in dt:
            prundt_rows.append((p, r, ud, n, d, t))
    _emit(
        provider,
        "solve_data/prundt.csv",
        _to_utf8_frame(
            ("process", "reserve", "upDown", "node", "period", "time"),
            prundt_rows,
        ),
    )


# ---------------------------------------------------------------------------
# write_process_reserve_filters_and_reliability  (mod L1655, L1660-1668)
# ---------------------------------------------------------------------------


def _read_p_process_reserve(
    path: Path,
    *, provider: "object | None" = None,
) -> dict[tuple[str, str, str, str, str], float]:
    """``p_process__reserve__upDown__node.csv`` →
    {(process, reserve, upDown, node, param): float}."""
    out: dict[tuple[str, str, str, str, str], float] = {}
    seeded = _provider_open(provider, _provider_key(path), path)
    if seeded is None:
        return out
    with seeded as fh:
        next(fh, None)
        for line in fh:
            parts = line.rstrip("\n").split(",")
            if len(parts) >= 6 and all(parts[i] for i in range(5)):
                try:
                    out[(parts[0], parts[1], parts[2],
                         parts[3], parts[4])] = float(parts[5])
                except ValueError:
                    continue
    return out


def _compute_reserve_filters(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> tuple[
    list[tuple[str, str, str, str, str]],  # reliability rows (with value)
    list[tuple[str, str, str, str]],       # increase_reserve_ratio rows
    list[tuple[str, str, str, str]],       # large_failure_ratio rows
    list[tuple[str]],                       # process_large_failure rows
]:
    """Shared scan of the four reserve-filter frames.  All value cells
    pre-stringified via ``repr(float(v))``.
    """
    p_prn = _read_p_process_reserve(
        input_dir / "p_process__reserve__upDown__node.csv", provider=provider,
    )
    active = _read_n_col(
        solve_data_dir / "process_reserve_upDown_node_active.csv", 4,
        provider=provider,
    )

    rel_rows: list[tuple[str, str, str, str, str]] = []
    incr_rows: list[tuple[str, str, str, str]] = []
    lf_rows: list[tuple[str, str, str, str]] = []
    for (p, r, ud, n) in active:
        v = p_prn.get((p, r, ud, n, "reliability"), 1.0)
        if v == 0.0:
            v = 1.0
        rel_rows.append((p, r, ud, n, repr(float(v))))
        if p_prn.get((p, r, ud, n, "increase_reserve_ratio"), 0.0) > 0:
            incr_rows.append((p, r, ud, n))
        if p_prn.get((p, r, ud, n, "large_failure_ratio"), 0.0) > 0:
            lf_rows.append((p, r, ud, n))

    process_lf = [
        (p,) for p in dict.fromkeys(p for (p, _, _, _) in lf_rows)
    ]
    return rel_rows, incr_rows, lf_rows, process_lf


def derive_p_process_reserve_upDown_node_reliability(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    """Reliability fallback frame (process, reserve, upDown, node, value).
    Default 1 with the legacy zero→1 collapse already applied."""
    rel_rows, _i, _l, _p = _compute_reserve_filters(
        input_dir, solve_data_dir, provider=provider,
    )
    return _to_utf8_frame(
        ("process", "reserve", "upDown", "node", "value"), rel_rows,
    )


def derive_process_reserve_upDown_node_increase_reserve_ratio(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    """Frame of ``(p, r, ud, n)`` where ``increase_reserve_ratio > 0``."""
    _r, incr_rows, _l, _p = _compute_reserve_filters(
        input_dir, solve_data_dir, provider=provider,
    )
    return _to_utf8_frame(
        ("process", "reserve", "upDown", "node"), incr_rows,
    )


def derive_process_reserve_upDown_node_large_failure_ratio(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    """Frame of ``(p, r, ud, n)`` where ``large_failure_ratio > 0``."""
    _r, _i, lf_rows, _p = _compute_reserve_filters(
        input_dir, solve_data_dir, provider=provider,
    )
    return _to_utf8_frame(
        ("process", "reserve", "upDown", "node"), lf_rows,
    )


def derive_process_large_failure(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    """First-column setof projection of the large_failure_ratio frame."""
    _r, _i, _l, process_lf = _compute_reserve_filters(
        input_dir, solve_data_dir, provider=provider,
    )
    return _to_utf8_frame(("process",), process_lf)


def emit_process_reserve_filters_and_reliability(
    input_dir: Path, solve_data_dir: Path,
    *, provider,
) -> None:
    """Provider-emitting twin of
    :func:`write_process_reserve_filters_and_reliability`."""
    rel_rows, incr_rows, lf_rows, process_lf = _compute_reserve_filters(
        input_dir, solve_data_dir, provider=provider,
    )
    _emit(
        provider,
        "solve_data/p_process_reserve_upDown_node_reliability.csv",
        _to_utf8_frame(
            ("process", "reserve", "upDown", "node", "value"), rel_rows,
        ),
    )
    _emit(
        provider,
        "solve_data/process_reserve_upDown_node_increase_reserve_ratio.csv",
        _to_utf8_frame(
            ("process", "reserve", "upDown", "node"), incr_rows,
        ),
    )
    _emit(
        provider,
        "solve_data/process_reserve_upDown_node_large_failure_ratio.csv",
        _to_utf8_frame(
            ("process", "reserve", "upDown", "node"), lf_rows,
        ),
    )
    _emit(
        provider,
        "solve_data/process_large_failure.csv",
        _to_utf8_frame(("process",), process_lf),
    )
