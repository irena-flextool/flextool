"""Writer-port Phase 2 (sub-dispatch 2) — LP row-scaling capacity proxies.

Native polars port of
``flextool.flextoolrunner.preprocessing.lp_scaling_params``
(legacy ~245 LOC).  Called per-solve from
``flextool.flextoolrunner.preprocessing.solve_time.run`` via
``write_lp_scaling_params``.

The 9 CSVs compute a power-of-10 capacity proxy per (node, period) and
(group, period) so the LP solver sees coefficient ranges compressed
near O(1).  When ``p_use_row_scaling[solve] < 0.5`` the scalers collapse
to 1 (Mode A, pre-Agent-5 behaviour).

All output values are floats, pre-stringified with ``repr(float(v))``
for byte-identical parity with the legacy emitter.
"""
from __future__ import annotations

import math
from pathlib import Path

import polars as pl


# ---------------------------------------------------------------------------
# CSV I/O helpers — same conventions as ``_writer_per_solve``.
# ---------------------------------------------------------------------------


def _read_csv(path: Path, columns: list[str]) -> pl.DataFrame:
    # Phase E-d — seed-aware: prefer in-memory accumulator frame.
    from flextool.engine_polars._input_source import _seed_lookup_positional
    seeded = _seed_lookup_positional(path, columns)
    if seeded is not None:
        return seeded
    if not path.exists() or path.stat().st_size == 0:
        return pl.DataFrame(
            {c: [] for c in columns},
            schema={c: pl.Utf8 for c in columns},
        )
    df = pl.read_csv(
        path,
        has_header=True,
        infer_schema_length=0,
        truncate_ragged_lines=True,
    )
    keep = df.columns[: len(columns)]
    df = df.select(keep)
    df.columns = columns
    return df


def _read_singles(path: Path) -> list[str]:
    df = _read_csv(path, ["v"])
    return [v for v in df["v"].to_list() if v]


def _read_pairs(path: Path) -> list[tuple[str, str]]:
    df = _read_csv(path, ["a", "b"])
    return [(a, b) for a, b in zip(df["a"].to_list(), df["b"].to_list())
            if a and b]


def _read_keyed_value(path: Path) -> dict[str, float]:
    df = _read_csv(path, ["key", "value"])
    out: dict[str, float] = {}
    for k, v in zip(df["key"].to_list(), df["value"].to_list()):
        if not k or v is None or v == "":
            continue
        try:
            out[k] = float(v)
        except ValueError:
            continue
    return out


def _read_triples(path: Path) -> list[tuple[str, str, str]]:
    df = _read_csv(path, ["a", "b", "c"])
    return [(a, b, c) for a, b, c in zip(df["a"].to_list(),
                                          df["b"].to_list(),
                                          df["c"].to_list())
            if a and b and c]


def _read_node_period_value(path: Path) -> dict[tuple[str, str], float]:
    df = _read_csv(path, ["node", "period", "value"])
    out: dict[tuple[str, str], float] = {}
    for n, d, v in zip(df["node"].to_list(),
                       df["period"].to_list(),
                       df["value"].to_list()):
        if not n or not d:
            continue
        try:
            out[(n, d)] = float(v)
        except (ValueError, TypeError):
            continue
    return out


def _write_keyed_2(path: Path, header: tuple[str, str, str],
                   rows: list[tuple[str, str, object]]) -> None:
    """Emit a 3-col CSV with ``repr(v)`` (legacy-faithful).

    Legacy emits ``f"{a},{b},{repr(v)}"`` — so an empty ``sum(())`` lands
    as ``"0"`` (int) while a float arithmetic result lands as ``"1.0"``.
    We do NOT pre-cast values to float here; callers must hand us values
    with the *same* runtime type the legacy emitter would have produced.

    Retained for backward compatibility; new emissions in this module
    flow through :func:`_write` (which feeds the Phase E-b accumulator).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        ",".join(header) + "\n"
        + "".join(f"{a},{b},{repr(v)}\n" for a, b, v in rows)
    )


def _write(df: pl.DataFrame, path: Path) -> None:
    """Polars-frame emission funnel — patched by Phase E-b accumulator.

    Identical I/O contract to the dispatcher / entity_annual ``_write``:
    the patched variant in
    :func:`._flex_data_accumulator.capture_frames` rebinds this name
    to also stash ``(path.name -> df)`` into the accumulator.

    Phase E-c — disk emission gated behind ``emit_csvs_enabled()``.
    """
    from flextool.engine_polars._flex_data_accumulator import emit_csvs_enabled
    if not emit_csvs_enabled():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)


def _rows_to_frame(
    header: tuple[str, str, str],
    rows: list[tuple[str, str, object]],
) -> pl.DataFrame:
    """Materialise (k1, k2, repr(value)) rows as an all-Utf8 frame.

    Pre-stringifies values via ``repr(v)`` (NOT ``repr(float(v))``) so
    that an empty ``sum(())`` -> int 0 round-trips as ``"0"`` and a
    float arithmetic result round-trips as ``"1.0"`` — matching the
    legacy ``_write_keyed_2`` byte shape exactly.
    """
    return pl.DataFrame(
        {
            header[0]: [r[0] for r in rows],
            header[1]: [r[1] for r in rows],
            header[2]: [repr(r[2]) for r in rows],
        },
        schema={h: pl.Utf8 for h in header},
    )


# ---------------------------------------------------------------------------
# Numeric primitive.
# ---------------------------------------------------------------------------


def _pow10_round_clamped(v: float) -> float:
    """``max(1e-6, min(1e9, 10 ** round(log10(v))))`` with v ≤ 0 → 1."""
    if v <= 0:
        return 1.0
    return max(1e-6, min(1e9, 10.0 ** round(math.log10(v))))


# ---------------------------------------------------------------------------
# Top-level writer.
# ---------------------------------------------------------------------------


def _compute_lp_scaling_frames(
    input_dir: Path, solve_data_dir: Path,
) -> dict[str, pl.DataFrame]:
    """Compute every LP-scaling CSV in one pass, returning a dict keyed
    by output basename.

    Used by both the wrapper (each frame fed to ``_write``) and the
    standalone ``derive_*`` functions (which index this dict).  The 9
    CSVs share heavy cross-CSV state (``raw_dict`` -> ``pow10_dict`` ->
    ``ncfs_dict`` -> ``grp_raw`` -> ``grp_pow10`` -> group_capacity);
    the dict-of-frames adapter pattern from the audit doc is the
    appropriate refactor here.
    """
    out: dict[str, pl.DataFrame] = {}

    nodes = _read_singles(input_dir / "node.csv")
    groups = _read_singles(input_dir / "group.csv")
    period_in_use = _read_singles(solve_data_dir / "period_in_use_set.csv")

    # solve_current → scaling_active.  Mod-faithful test:
    #     sum_{c in solve_current} p_use_row_scaling[c] >= 0.5
    p_use_row_scaling = _read_keyed_value(
        solve_data_dir / "p_use_row_scaling.csv"
    )
    solve_current = _read_singles(solve_data_dir / "solve_current.csv")
    scaling_active = sum(
        p_use_row_scaling.get(c, 0.0) for c in solve_current
    ) >= 0.5

    # process_source_sink + p_entity_unitsize → _node_cap_unitsize_sum.
    pss = _read_triples(solve_data_dir / "process_source_sink.csv")
    p_entity_unitsize = _read_keyed_value(
        solve_data_dir / "p_entity_unitsize.csv"
    )
    inflow_fallback = _read_node_period_value(
        solve_data_dir / "_node_cap_inflow_fallback.csv"
    )
    group_node = _read_pairs(input_dir / "group__node.csv")
    nodes_for_group: dict[str, list[str]] = {}
    for g, n in group_node:
        nodes_for_group.setdefault(g, []).append(n)

    node_set = frozenset(nodes)

    # ── _node_cap_unitsize_sum ───────────────────────────────────────────
    # Σ p_entity_unitsize[p] for each arc end ∈ nodes.
    cap_unitsize: dict[str, float] = {n: 0.0 for n in nodes}
    for p, source, sink in pss:
        usz = p_entity_unitsize.get(p, 0.0)
        if sink in node_set:
            cap_unitsize[sink] += usz
        if source in node_set:
            cap_unitsize[source] += usz

    rows_unitsize: list[tuple[str, str, float]] = [
        (n, d, cap_unitsize.get(n, 0.0)) for n in nodes for d in period_in_use
    ]
    out["_node_cap_unitsize_sum.csv"] = _rows_to_frame(
        ("node", "period", "value"), rows_unitsize,
    )

    # ── _node_cap_raw ────────────────────────────────────────────────────
    # unitsize_sum if > 0 else inflow_fallback if > 0 else 1.
    raw_dict: dict[tuple[str, str], float] = {}
    rows_raw: list[tuple[str, str, float]] = []
    for n in nodes:
        usz = cap_unitsize.get(n, 0.0)
        for d in period_in_use:
            if usz > 0:
                v = usz
            else:
                fb = inflow_fallback.get((n, d), 0.0)
                v = fb if fb > 0 else 1.0
            raw_dict[(n, d)] = v
            rows_raw.append((n, d, v))
    out["_node_cap_raw.csv"] = _rows_to_frame(
        ("node", "period", "value"), rows_raw,
    )

    # ── _node_cap_pow10 ──────────────────────────────────────────────────
    pow10_dict: dict[tuple[str, str], float] = {}
    rows_pow10: list[tuple[str, str, float]] = []
    for (n, d), v in raw_dict.items():
        p10 = _pow10_round_clamped(v)
        pow10_dict[(n, d)] = p10
        rows_pow10.append((n, d, p10))
    out["_node_cap_pow10.csv"] = _rows_to_frame(
        ("node", "period", "value"), rows_pow10,
    )

    # ── node_capacity_for_scaling + inv_node_cap ─────────────────────────
    ncfs_dict: dict[tuple[str, str], float] = {}
    rows_ncfs: list[tuple[str, str, float]] = []
    rows_inc: list[tuple[str, str, float]] = []
    for n in nodes:
        for d in period_in_use:
            v = pow10_dict.get((n, d), 1.0) if scaling_active else 1.0
            ncfs_dict[(n, d)] = v
            rows_ncfs.append((n, d, v))
            rows_inc.append((n, d, 1.0 / v if v != 0 else 0.0))
    out["node_capacity_for_scaling.csv"] = _rows_to_frame(
        ("node", "period", "value"), rows_ncfs,
    )
    out["inv_node_cap.csv"] = _rows_to_frame(
        ("node", "period", "value"), rows_inc,
    )

    # ── _group_cap_raw ───────────────────────────────────────────────────
    # NOTE: Python's ``sum(())`` returns int ``0`` (not ``0.0``) for empty
    # groups; the legacy emitter preserves this int/float inconsistency
    # via ``repr(v)``.  The byte-parity gate requires we preserve it too.
    # The Phase C accumulator parity test (which round-trips through
    # ``pl.read_csv`` -> Float64 cast) skips this basename for the same
    # reason — see ``_flex_data_accumulator.expected_basenames()`` for
    # the exclusion list.
    grp_raw: dict[tuple[str, str], float] = {}
    rows_graw: list[tuple[str, str, float]] = []
    for g in groups:
        for d in period_in_use:
            v = sum(ncfs_dict.get((n, d), 1.0)
                    for n in nodes_for_group.get(g, ()))
            grp_raw[(g, d)] = v
            rows_graw.append((g, d, v))
    out["_group_cap_raw.csv"] = _rows_to_frame(
        ("group", "period", "value"), rows_graw,
    )

    # ── _group_cap_pow10 ─────────────────────────────────────────────────
    grp_pow10: dict[tuple[str, str], float] = {}
    rows_gpow10: list[tuple[str, str, float]] = []
    for (g, d), v in grp_raw.items():
        p10 = _pow10_round_clamped(v) if v > 0 else 1.0
        grp_pow10[(g, d)] = p10
        rows_gpow10.append((g, d, p10))
    out["_group_cap_pow10.csv"] = _rows_to_frame(
        ("group", "period", "value"), rows_gpow10,
    )

    # ── group_capacity_for_scaling + inv_group_cap ───────────────────────
    rows_gcfs: list[tuple[str, str, float]] = []
    rows_igc: list[tuple[str, str, float]] = []
    for g in groups:
        for d in period_in_use:
            v = grp_pow10.get((g, d), 1.0) if scaling_active else 1.0
            rows_gcfs.append((g, d, v))
            rows_igc.append((g, d, 1.0 / v if v != 0 else 0.0))
    out["group_capacity_for_scaling.csv"] = _rows_to_frame(
        ("group", "period", "value"), rows_gcfs,
    )
    out["inv_group_cap.csv"] = _rows_to_frame(
        ("group", "period", "value"), rows_igc,
    )

    return out


# ---- Phase E-b — derive_X family for each emitted CSV --------------------
#
# Each derive_* delegates to the shared :func:`_compute_lp_scaling_frames`
# pass and indexes the resulting dict.  The shared compute is the path
# of least re-walking — splitting into 9 standalone derive_* would
# re-scan ``process_source_sink`` and the (n, d) cross-product per call.


def _derive(input_dir: Path, solve_data_dir: Path,
            basename: str) -> pl.DataFrame:
    return _compute_lp_scaling_frames(input_dir, solve_data_dir)[basename]


def derive__node_cap_unitsize_sum(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``_node_cap_unitsize_sum.csv`` — Σ p_entity_unitsize for each
    arc-end ∈ nodes, broadcast across period_in_use."""
    return _derive(input_dir, solve_data_dir, "_node_cap_unitsize_sum.csv")


def derive__node_cap_raw(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``_node_cap_raw.csv`` — unitsize_sum if > 0, else inflow_fallback,
    else 1."""
    return _derive(input_dir, solve_data_dir, "_node_cap_raw.csv")


def derive__node_cap_pow10(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``_node_cap_pow10.csv`` — pow-10 round-clamp of _node_cap_raw."""
    return _derive(input_dir, solve_data_dir, "_node_cap_pow10.csv")


def derive_node_capacity_for_scaling(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``node_capacity_for_scaling.csv`` — _node_cap_pow10 gated by
    scaling_active (Mode A collapses to 1)."""
    return _derive(input_dir, solve_data_dir, "node_capacity_for_scaling.csv")


def derive_inv_node_cap(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``inv_node_cap.csv`` — 1 / node_capacity_for_scaling."""
    return _derive(input_dir, solve_data_dir, "inv_node_cap.csv")


def derive__group_cap_raw(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``_group_cap_raw.csv`` — Σ over nodes_for_group of
    node_capacity_for_scaling."""
    return _derive(input_dir, solve_data_dir, "_group_cap_raw.csv")


def derive__group_cap_pow10(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``_group_cap_pow10.csv`` — pow-10 round-clamp of _group_cap_raw."""
    return _derive(input_dir, solve_data_dir, "_group_cap_pow10.csv")


def derive_group_capacity_for_scaling(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``group_capacity_for_scaling.csv`` — _group_cap_pow10 gated by
    scaling_active."""
    return _derive(input_dir, solve_data_dir, "group_capacity_for_scaling.csv")


def derive_inv_group_cap(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``inv_group_cap.csv`` — 1 / group_capacity_for_scaling."""
    return _derive(input_dir, solve_data_dir, "inv_group_cap.csv")


def write_lp_scaling_params(
    input_dir: Path, solve_data_dir: Path,
) -> None:
    """Native port of ``lp_scaling_params.write_lp_scaling_params``.

    Reads:
      * ``input/node.csv`` / ``group.csv`` / ``group__node.csv``
      * ``solve_data/`` — period_in_use_set, p_use_row_scaling,
        solve_current, process_source_sink, p_entity_unitsize,
        _node_cap_inflow_fallback

    Emits 9 ``solve_data/`` CSVs covering node-level + group-level
    capacity proxies and their reciprocals.  Each output flows through
    ``_write(frame, path)`` so the Phase E-b accumulator captures every
    emitted frame.
    """
    frames = _compute_lp_scaling_frames(input_dir, solve_data_dir)
    for basename, df in frames.items():
        _write(df, solve_data_dir / basename)
