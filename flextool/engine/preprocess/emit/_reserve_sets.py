"""Reserve calculated params — per-solve emitters.

Fired per-solve from ``_solve_time.run`` at batches 43, 44 and 49.

Three public entry points:

* :func:`emit_pdtReserve_upDown_group` — mod L1319.  4-branch hourly
  resolution for reserve groups, emitting ``pdtReserve_upDown_group.csv``
  ``(reserve, upDown, group, param, period, time, value)``.
* :func:`emit_process_reserve_upDown_node_active_and_prundt` — keeps the
  ``process_reserve_upDown_node`` rows of the reserves with a requirement
  (:func:`~flextool.engine.preprocess._reserve_methods.demanded_reserves`:
  a nonzero summed reservation across the matching
  ``reserve__upDown__group`` × ``dt`` cross product, or a dynamic / n-1
  group driven by a ratio, the group's own ``increase_reserve_ratio``
  included), then emits ``prundt =
  process_reserve_upDown_node_active × dt``.
* :func:`emit_process_reserve_filters_and_reliability` — mod L1655 /
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

``flextool.engine.model._reserve`` consumes these CSVs as RHS / domain
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

from flextool.engine.preprocess._provider_io import (
    _emit,
    _provider_key,
)
from flextool.engine.preprocess._reserve_methods import demanded_reserves
from flextool.engine.preprocess.emit._pdt_lookup import (
    _anchor_fan_members,
    _check_pbt_branch_coverage,
    _check_pre_reveal_reads_have_forecast,
    _check_roll_forecast_covers_lookahead,
    _check_ungrouped_reads_have_realized_branch,
    fold_over_tiers,
    read_fan_for_d,
    ts_tiers,
)


# ---------------------------------------------------------------------------
# I/O contract: legacy emitters in this family used
# ``path.open("w") + fh.write(...)`` with plain ``\n`` line terminators
# (no CRLF), so the default polars ``write_csv`` line ending matches
# byte-for-byte.  Float cells are pre-rendered with ``repr(float(v))``
# by the ``derive_*`` builders, so the underlying frame is all-``Utf8``
# and polars does no extra numeric formatting at emission time.
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
# Native-frame readers — same shape as _inflow_scaling / _pdt_lookup
# helpers.  The legacy emitter wrote plain text with ``repr(v)``; these
# read the in-memory polars frame directly via
# ``provider.get(_provider_key(path))`` instead of round-tripping through
# CSV text.
#
# Type-fidelity contract — reproduce *exactly* what ``line.split(",")``
# over a ``DataFrame.write_csv`` serialisation would have yielded:
#
#   * Key columns (dict-key / structural string positions) were string
#     cells.  ``write_csv`` serialises ``null`` → ``""`` and any scalar
#     to its textual form; we coerce each key cell with :func:`_cell_str`
#     and apply the original truthiness guard to the *string* form so a
#     null cell is skipped while a literal ``"0"`` is kept.
#   * Value columns were re-coerced with ``float(...)``.  We apply the same
#     coercion to the native cell and widen the legacy ``except
#     ValueError`` to ``except (ValueError, TypeError)`` so a null value
#     cell is skipped (matching the legacy empty-string ``ValueError``).
#
# ``provider.get`` returns data rows only (no header), so there is no
# header row to skip; an empty / missing frame yields the same empty
# output the legacy loop produced.
# ---------------------------------------------------------------------------


def _cell_str(value: "object | None") -> str:
    """Reproduce a CSV cell string for a native frame value.

    ``DataFrame.write_csv`` renders ``null`` as the empty string and every
    other scalar as its textual form; the legacy ``line.split(",")`` then
    read those strings back.  Mirror that here so dict keys / structural
    strings stay byte-identical to the legacy CSV round-trip.
    """
    return "" if value is None else str(value)


def _read_singles(path: Path,
                  *, provider: "object | None" = None) -> list[str]:
    """First-column reader (data rows only)."""
    df = provider.get(_provider_key(path))
    if df is None:
        return []
    out: list[str] = []
    for row in df.iter_rows():
        if not row:
            continue
        c0 = _cell_str(row[0])
        if c0:
            out.append(c0)
    return out


def _read_pairs(path: Path,
                *, provider: "object | None" = None) -> list[tuple[str, str]]:
    """First-two-column reader (data rows only)."""
    df = provider.get(_provider_key(path))
    if df is None:
        return []
    out: list[tuple[str, str]] = []
    for row in df.iter_rows():
        if len(row) < 2:
            continue
        c0, c1 = _cell_str(row[0]), _cell_str(row[1])
        if c0 and c1:
            out.append((c0, c1))
    return out


def _read_n_col(path: Path, n: int,
                *, provider: "object | None" = None) -> list[tuple[str, ...]]:
    """First-n-column reader; rows with any empty key skipped."""
    df = provider.get(_provider_key(path))
    if df is None:
        return []
    out: list[tuple[str, ...]] = []
    for row in df.iter_rows():
        if len(row) < n:
            continue
        cells = tuple(_cell_str(row[i]) for i in range(n))
        if all(cells):
            out.append(cells)
    return out


def _read_pairs_to_dict(path: Path, key_col: int,
                        *, provider: "object | None" = None,
                        ) -> dict[str, list[str]]:
    """Two-col frame → {row[key_col]: [row[other_col], ...]} preserving order."""
    out: dict[str, list[str]] = {}
    df = provider.get(_provider_key(path))
    if df is None:
        return out
    other_col = 1 - key_col
    for row in df.iter_rows():
        if len(row) < 2:
            continue
        c0, c1 = _cell_str(row[0]), _cell_str(row[1])
        if c0 and c1:
            cells = (c0, c1)
            out.setdefault(cells[key_col], []).append(cells[other_col])
    return out


def _read_pbt_reserve(
    path: Path,
    *, provider: "object | None" = None,
) -> dict[tuple[str, str, str, str, str, str, str], float]:
    """``pbt_reserve__upDown__group.csv`` →
    {(r, ud, g, param, branch, ts, t): float}.  Malformed / non-numeric
    rows silently skipped (matches legacy)."""
    out: dict[tuple[str, str, str, str, str, str, str], float] = {}
    df = provider.get(_provider_key(path))
    if df is None:
        return out
    for row in df.iter_rows():
        if len(row) < 8:
            continue
        c = [_cell_str(row[i]) for i in range(7)]
        if all(c):
            try:
                out[(c[0], c[1], c[2], c[3], c[4], c[5], c[6])] = float(row[7])
            except (ValueError, TypeError):
                continue
    return out


def _read_pt_reserve(
    path: Path,
    *, provider: "object | None" = None,
) -> dict[tuple[str, str, str, str, str], float]:
    """``pt_reserve__upDown__group.csv`` →
    {(r, ud, g, param, t): float}."""
    out: dict[tuple[str, str, str, str, str], float] = {}
    df = provider.get(_provider_key(path))
    if df is None:
        return out
    for row in df.iter_rows():
        if len(row) < 6:
            continue
        c = [_cell_str(row[i]) for i in range(5)]
        if all(c):
            try:
                out[(c[0], c[1], c[2], c[3], c[4])] = float(row[5])
            except (ValueError, TypeError):
                continue
    return out


def _read_p_reserve(
    path: Path,
    *, provider: "object | None" = None,
) -> dict[tuple[str, str, str, str], float]:
    """``p_reserve__upDown__group.csv`` → {(r, ud, g, param): float}."""
    out: dict[tuple[str, str, str, str], float] = {}
    df = provider.get(_provider_key(path))
    if df is None:
        return out
    for row in df.iter_rows():
        if len(row) < 5:
            continue
        c = [_cell_str(row[i]) for i in range(4)]
        if all(c):
            try:
                out[(c[0], c[1], c[2], c[3])] = float(row[4])
            except (ValueError, TypeError):
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
    fan_for_d: "dict[str, list[str]] | None" = None,
    tiers_by_d: "dict[str, list[list[str]]] | None" = None,
) -> float:
    """4-branch fallback mirroring ``pdtReserve_upDown_group``'s mod body.

    Branches mirror ``pdtProcess``'s branches 1, 2, 4, 5 (no period axis,
    no def1 — and the scalar branch returns the table's default of 0
    for ``reservation`` when the (r, ud, g) row is missing).  *fan_for_d*
    (``branch_time_start``) is the fan start of a fold cell's second (in a roll: first) tier; *tiers_by_d* (period ->
    :func:`ts_tiers`) lets a caller compute the tiers once per period instead of once per cell.
    """
    # Branch 1: stochastic + outer-d's tb; ts = d's own first step and the fan start in ts_tiers order
    # (per value cell).
    tiers = tiers_by_d.get(d) if tiers_by_d is not None else None
    if tiers is None:
        tiers = ts_tiers(d, ts_for_d, fan_for_d or {})
    if g_stoch:
        v = fold_over_tiers(pbt, (r, ud, g, param), [tb_for_d.get(d, ())], tiers, t)
        if v is not None:
            return v
    # Branch 2: parent period pe of d, tb from solve_branch[pe], same ts tiers.
    pe_list = pe_for_d.get(d, ())
    if pe_list:
        v = fold_over_tiers(pbt, (r, ud, g, param), [tb_for_d.get(pe, ()) for pe in pe_list], tiers, t)
        if v is not None:
            return v
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

    Consumer besides the seeds: the DB-direct readers take the values of entities authoring a
    stochastic-time map from this frame (``_param_shapes.STOCHASTIC_TIME_PARAMS``), so it stays emitted
    while that registry names it, even once the seeds are retired.
    """
    pbt = _read_pbt_reserve(
        solve_data_dir / "pbt_reserve__upDown__group.csv", provider=provider,
    )
    pt = _read_pt_reserve(
        solve_data_dir / "pt_reserve__upDown__group.csv", provider=provider,
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

    # A reserve group outside every include_stochastics group reads the realized branch's value
    # (branch 2 below); a forecasts-only fan has none, so its stochastic-time reservation is an
    # input error rather than a silent fall-through to the default 0.
    # A reservation that authors only some of a fan's forecasts is an input error as well.
    rug_set = set(rug)
    fan_for_d = read_fan_for_d(solve_data_dir, provider=provider)
    anchor_members = _anchor_fan_members(solve_data_dir / "period__branch.csv", provider=provider)
    pbt_in_use = {k: v for k, v in pbt.items() if k[:3] in rug_set}
    _check_ungrouped_reads_have_realized_branch(
        pbt_in_use, 3, groups_stoch, anchor_members,
        tb_for_d, ts_for_d, member_pos=2, fan_for_d=fan_for_d, kind="reserve",
    )
    _check_pbt_branch_coverage(pbt_in_use, 3, anchor_members, tb_for_d, ts_for_d, fan_for_d, kind="reserve")
    # In a roll of a rolling solve the roll's own forecast must cover the roll's look-ahead periods.
    # Before the reveal, a period reads the forecast made when the solve or roll starts, and it must cover it.
    _check_pre_reveal_reads_have_forecast(pbt_in_use, 3, groups_stoch, tb_for_d, ts_for_d, pe_for_d, fan_for_d,
                                          solve_data_dir, provider=provider, member_pos=2, kind="reserve")
    _check_roll_forecast_covers_lookahead(pbt_in_use, 3, groups_stoch, tb_for_d, ts_for_d, pe_for_d, fan_for_d,
                                          solve_data_dir, provider=provider, member_pos=2, kind="reserve")

    tiers_by_d = {d: ts_tiers(d, ts_for_d, fan_for_d) for d in dict.fromkeys(d for d, _t in dt)}
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
                    fan_for_d=fan_for_d,
                    tiers_by_d=tiers_by_d,
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
    """Emit ``pdtReserve_upDown_group`` to the Provider."""
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
    active ``(p, r, ud, n)`` list and the ``dt`` pair list.

    A relationship is active when its ``(r, ud)`` has a requirement
    (:func:`~flextool.engine.preprocess._reserve_methods.demanded_reserves`);
    ``params/_derived.prundt_from_source`` applies the same rule to the DB.
    """
    pdt_reserve: dict[tuple[str, str, str, str, str, str], float] = {}
    pdt_path = solve_data_dir / "pdtReserve_upDown_group.csv"
    pdt_df = provider.get(_provider_key(pdt_path))
    if pdt_df is not None:
        for row in pdt_df.iter_rows():
            if len(row) < 7:
                continue
            c = [_cell_str(row[i]) for i in range(6)]
            if all(c):
                try:
                    pdt_reserve[(c[0], c[1], c[2], c[3], c[4], c[5])] = float(row[6])
                except (ValueError, TypeError):
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

    reservation_pairs: set[tuple[str, str]] = set()
    for (p, r, ud, n) in prun:
        groups = rug_by_ru.get((r, ud), ())
        total = 0.0
        for g in groups:
            for (d, t) in dt:
                total += pdt_reserve.get(
                    (r, ud, g, "reservation", d, t), 0.0,
                )
        if total != 0.0:
            reservation_pairs.add((r, ud))

    # Dynamic / n-1 requirements are reserve demand by themselves
    # (``demanded_reserves``).
    group_methods: list[tuple[str, ...]] = []
    for method in ("dynamic", "n_1"):
        group_methods += _read_n_col(
            solve_data_dir / f"reserve__upDown__group__method_{method}.csv", 4,
            provider=provider,
        )
    # solve_data/group_node.csv, else input/group__node.csv (as
    # ``model._reserve.load_data``).
    group_nodes = (
        _read_n_col(solve_data_dir / "group_node.csv", 2, provider=provider)
        or _read_n_col(input_dir / "group__node.csv", 2, provider=provider)
    )
    prun_set = set(prun)
    ratios: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for (p, r, ud, n, param), v in _read_p_process_reserve(
        input_dir / "p_process__reserve__upDown__node.csv", provider=provider,
    ).items():
        if param in ("increase_reserve_ratio", "large_failure_ratio") \
                and (p, r, ud, n) in prun_set:
            ratios.setdefault((p, r, ud, n), {})[param] = v
    demanded = demanded_reserves(
        reservation_pairs=reservation_pairs,
        group_methods=group_methods,
        group_nodes=group_nodes,
        relationship_ratios=ratios,
        group_ratios={(r, ud, g): v for (r, ud, g, param), v in _read_p_reserve(
            input_dir / "p_reserve__upDown__group.csv", provider=provider,
        ).items() if param == "increase_reserve_ratio"},
    )

    # ``prun`` order (byte parity of the emitted CSVs).
    active_rows = [(p, r, ud, n) for (p, r, ud, n) in prun
                   if (r, ud) in demanded]
    return active_rows, dt


def emit_process_reserve_upDown_node_active_and_prundt(
    input_dir: Path, solve_data_dir: Path,
    *, provider,
) -> None:
    """Emit ``process_reserve_upDown_node_active_and_prundt`` to the Provider."""
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
    df = provider.get(_provider_key(path))
    if df is None:
        return out
    for row in df.iter_rows():
        if len(row) < 6:
            continue
        c = [_cell_str(row[i]) for i in range(5)]
        if all(c):
            try:
                out[(c[0], c[1], c[2], c[3], c[4])] = float(row[5])
            except (ValueError, TypeError):
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


def emit_process_reserve_filters_and_reliability(
    input_dir: Path, solve_data_dir: Path,
    *, provider,
) -> None:
    """Emit ``process_reserve_filters_and_reliability`` to the Provider."""
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
