"""LP row-scaling capacity proxies — per-solve emitters.

Called per-solve from ``_emit_solve_time.run`` via
:func:`emit_lp_scaling_params`.

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

from flextool.engine_polars._emit_provider_io import _emit
from flextool.engine_polars._vectorize import _render_value_column


# ---------------------------------------------------------------------------
# CSV I/O helpers — same conventions as ``_emit_per_solve``.
# ---------------------------------------------------------------------------


def _read_csv(path: Path, columns: list[str],
              *, provider: "object | None" = None) -> pl.DataFrame:
    """Provider-only — Step 2.5 Phase C dropped the disk-fallback arm."""
    from flextool.engine_polars._emit_provider_io import (
        _provider_key,
        _provider_lookup_positional,
    )
    seeded = _provider_lookup_positional(
        provider, _provider_key(path), path, columns,
    )
    if seeded is not None:
        return seeded
    return pl.DataFrame(
        {c: [] for c in columns},
        schema={c: pl.Utf8 for c in columns},
    )


def _read_singles(path: Path,
                  *, provider: "object | None" = None) -> list[str]:
    df = _read_csv(path, ["v"], provider=provider)
    return [v for v in df["v"].to_list() if v]


def _read_pairs(path: Path,
                *, provider: "object | None" = None) -> list[tuple[str, str]]:
    df = _read_csv(path, ["a", "b"], provider=provider)
    return [(a, b) for a, b in zip(df["a"].to_list(), df["b"].to_list())
            if a and b]


def _read_keyed_value(path: Path,
                      *, provider: "object | None" = None) -> dict[str, float]:
    df = _read_csv(path, ["key", "value"], provider=provider)
    out: dict[str, float] = {}
    for k, v in zip(df["key"].to_list(), df["value"].to_list()):
        if not k or v is None or v == "":
            continue
        try:
            out[k] = float(v)
        except ValueError:
            continue
    return out


def _read_triples(path: Path,
                  *, provider: "object | None" = None) -> list[tuple[str, str, str]]:
    df = _read_csv(path, ["a", "b", "c"], provider=provider)
    return [(a, b, c) for a, b, c in zip(df["a"].to_list(),
                                          df["b"].to_list(),
                                          df["c"].to_list())
            if a and b and c]


def _read_node_period_value(path: Path,
                             *, provider: "object | None" = None,
                             ) -> dict[tuple[str, str], float]:
    df = _read_csv(path, ["node", "period", "value"], provider=provider)
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


def _rows_to_frame(
    header: tuple[str, str, str],
    rows: list[tuple[str, str, object]],
) -> pl.DataFrame:
    """Materialise (k1, k2, repr(value)) rows as an all-Utf8 frame.

    Pre-stringifies values via ``repr(v)`` (NOT ``repr(float(v))``) so
    that an empty ``sum(())`` -> int 0 round-trips as ``"0"`` and a
    float arithmetic result round-trips as ``"1.0"`` — matching the
    legacy 3-col emitter's byte shape exactly.
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
    *, provider: "object | None" = None,
) -> dict[str, pl.DataFrame]:
    """Compute every LP-scaling CSV in one pass, returning a dict keyed
    by output basename.

    Retained as the parity ORACLE for
    :func:`_compute_lp_scaling_frames_vectorized` (see
    ``tests/engine_polars/test_vectorize_lp_scaling_parity.py``).  The 9
    CSVs share heavy cross-CSV state (``raw_dict`` -> ``pow10_dict`` ->
    ``ncfs_dict`` -> ``grp_raw`` -> ``grp_pow10`` -> group_capacity);
    the dict-of-frames adapter pattern from the audit doc is the
    appropriate refactor here.
    """
    out: dict[str, pl.DataFrame] = {}

    nodes = _read_singles(input_dir / "node.csv", provider=provider)
    groups = _read_singles(input_dir / "group.csv", provider=provider)
    period_in_use = _read_singles(solve_data_dir / "period_in_use_set.csv",
                                   provider=provider)

    # solve_current → scaling_active.  Mod-faithful test:
    #     sum_{c in solve_current} p_use_row_scaling[c] >= 0.5
    p_use_row_scaling = _read_keyed_value(
        solve_data_dir / "p_use_row_scaling.csv",
        provider=provider,
    )
    solve_current = _read_singles(solve_data_dir / "solve_current.csv",
                                   provider=provider)
    scaling_active = sum(
        p_use_row_scaling.get(c, 0.0) for c in solve_current
    ) >= 0.5

    # process_source_sink + p_entity_unitsize → _node_cap_unitsize_sum.
    pss = _read_triples(solve_data_dir / "process_source_sink.csv",
                         provider=provider)
    p_entity_unitsize = _read_keyed_value(
        solve_data_dir / "p_entity_unitsize.csv",
        provider=provider,
    )
    inflow_fallback = _read_node_period_value(
        solve_data_dir / "_node_cap_inflow_fallback.csv",
        provider=provider,
    )
    group_node = _read_pairs(input_dir / "group__node.csv",
                              provider=provider)
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


# ---------------------------------------------------------------------------
# Vectorized twin of _compute_lp_scaling_frames (vectorize-per-roll).
#
# Built as a FULL COPY of the legacy body, with each stage's COMPUTE
# replaced by vectorized polars incrementally (node chain → node
# capacity-for-scaling → group chain).  The in-memory dicts the legacy
# stages consume (``cap_unitsize``, ``raw_dict``, ``pow10_dict``,
# ``ncfs_dict``, ``grp_raw``, ``grp_pow10``) are reconstructed from each
# vectorized frame right after it is built, so any not-yet-vectorized
# downstream stage still works at every commit (full-9-key parity-gateable
# per commit).  The legacy :func:`_compute_lp_scaling_frames` is KEPT as
# the parity oracle.
#
# Tier policy (critique Defect X): NO lp key is hard-asserted Tier A.  On
# real fixtures the unitsize / group sums are exact-integer so no ULP drift
# occurs and the LP-coefficient keys land Tier A byte-exact in practice,
# but a pathological sum landing on a half-decade ``10^(k+0.5)`` could flip
# a ``_pow10_round_clamped`` decade bucket → factor-10 gap → the parity
# gate RAISES (loud, correct — never masked).  The pow10 stages reuse the
# SAME scalar :func:`_pow10_round_clamped` UDF for byte-fidelity.
# ---------------------------------------------------------------------------


def _empty_value_frame(header: tuple[str, str, str]) -> pl.DataFrame:
    """An explicit all-Utf8 empty 3-col frame (key1, key2, value)."""
    return pl.DataFrame(
        {h: [] for h in header},
        schema={h: pl.Utf8 for h in header},
    )


def _ordered_value_frame(
    df: pl.DataFrame,
    header: tuple[str, str, str],
    order_cols: list[str],
) -> pl.DataFrame:
    """Sort *df* by *order_cols*, render its ``value_f`` Float64 column via
    ``repr`` and project to the all-Utf8 ``(key1, key2, value)`` shape.

    *df* must carry the two key columns named ``header[0]``/``header[1]``,
    a Float64 ``value_f`` column, and the integer *order_cols*.  An empty
    *df* yields the explicit empty schema.
    """
    if df.height == 0:
        return _empty_value_frame(header)
    df = df.sort(order_cols)
    value = _render_value_column(df["value_f"])
    return df.select([header[0], header[1]]).with_columns(
        value.alias(header[2]),
    )


def _pow10_value_column(df: pl.DataFrame) -> pl.DataFrame:
    """Add a ``value_f`` column = ``_pow10_round_clamped(raw)`` per row.

    Applies the EXACT legacy scalar UDF per cell (critique-corrected: the
    same Python fn guarantees byte-identical bits — it re-derives the
    clamp / ``v<=0`` / NaN edge cases identically; a re-implemented polars
    ``log10().round()`` would risk diverging on those edges).  The Float64
    ``raw`` lifted from the dict is bit-identical to the legacy Python
    float, so ``round(math.log10(v))`` matches.
    """
    raw = df["raw"].to_list()
    return df.with_columns(
        pl.Series("value_f", [_pow10_round_clamped(v) for v in raw],
                  dtype=pl.Float64),
    )


def _compute_lp_scaling_frames_vectorized(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> dict[str, pl.DataFrame]:
    """Vectorized twin of :func:`_compute_lp_scaling_frames`.

    Same reader block, same in-memory dicts, same stage order, same 9
    LIVE ``out[...]=`` assignments.  Each vectorized stage reconstructs
    the in-memory dict its legacy downstream consumer reads, so the
    function is internally consistent (full-9-key parity-gateable) at
    every commit.
    """
    out: dict[str, pl.DataFrame] = {}

    # ── Sources (copied verbatim from the legacy reader block) ─────────
    nodes = _read_singles(input_dir / "node.csv", provider=provider)
    groups = _read_singles(input_dir / "group.csv", provider=provider)
    period_in_use = _read_singles(solve_data_dir / "period_in_use_set.csv",
                                   provider=provider)

    p_use_row_scaling = _read_keyed_value(
        solve_data_dir / "p_use_row_scaling.csv",
        provider=provider,
    )
    solve_current = _read_singles(solve_data_dir / "solve_current.csv",
                                   provider=provider)
    scaling_active = sum(
        p_use_row_scaling.get(c, 0.0) for c in solve_current
    ) >= 0.5

    pss = _read_triples(solve_data_dir / "process_source_sink.csv",
                         provider=provider)
    p_entity_unitsize = _read_keyed_value(
        solve_data_dir / "p_entity_unitsize.csv",
        provider=provider,
    )
    inflow_fallback = _read_node_period_value(
        solve_data_dir / "_node_cap_inflow_fallback.csv",
        provider=provider,
    )
    group_node = _read_pairs(input_dir / "group__node.csv",
                              provider=provider)
    # (the legacy ``nodes_for_group`` per-group node list is NOT built: the
    # vectorized group chain joins the ``group_node`` pairs directly.)

    node_set = frozenset(nodes)

    # ── Shared order frames (node list order, group order, period order) ─
    node_eo = pl.DataFrame(
        {"node": list(nodes), "__eo": list(range(len(nodes)))},
        schema={"node": pl.Utf8, "__eo": pl.Int64},
    )
    period_po = pl.DataFrame(
        {"period": list(period_in_use),
         "__po": list(range(len(period_in_use)))},
        schema={"period": pl.Utf8, "__po": pl.Int64},
    )

    # ── Stage U: _node_cap_unitsize_sum (Tier B) ───────────────────────
    # cap_unitsize pre-seeded {n: 0.0}; for each pss (p, source, sink) with
    # usz = p_entity_unitsize.get(p, 0.0): TWO INDEPENDENT ifs —
    #   if sink   in node_set: cap_unitsize[sink]   += usz
    #   if source in node_set: cap_unitsize[source] += usz
    # (a self-loop source==sink adds TWICE).  Vectorize as TWO concatenated
    # contributions so the double-count survives (do NOT struct-union /
    # .unique()), group-by-sum, then densify over node × period.
    pss_df = pl.DataFrame(
        {"process": [p for p, _s, _k in pss],
         "source": [s for _p, s, _k in pss],
         "sink": [k for _p, _s, k in pss]},
        schema={"process": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8},
    )
    usz_lk = pl.DataFrame(
        {"process": list(p_entity_unitsize.keys()),
         "v_usz": list(p_entity_unitsize.values())},
        schema={"process": pl.Utf8, "v_usz": pl.Float64},
    )
    node_list = list(node_set)
    pss_usz = pss_df.join(usz_lk, on="process", how="left").with_columns(
        pl.col("v_usz").fill_null(0.0),
    )
    sink_contrib = (
        pss_usz.filter(pl.col("sink").is_in(node_list))
        .select(pl.col("sink").alias("node"), pl.col("v_usz"))
    )
    source_contrib = (
        pss_usz.filter(pl.col("source").is_in(node_list))
        .select(pl.col("source").alias("node"), pl.col("v_usz"))
    )
    contrib = pl.concat([sink_contrib, source_contrib], how="vertical")
    if contrib.height:
        unitsize_sum_df = (
            contrib.group_by("node")
            .agg(pl.col("v_usz").sum().alias("v_unitsize"))
        )
    else:
        unitsize_sum_df = pl.DataFrame(
            {"node": [], "v_unitsize": []},
            schema={"node": pl.Utf8, "v_unitsize": pl.Float64},
        )
    # Densify over node × period; pre-seeded 0.0 → arc-less node emits
    # "0.0" (a float, NOT int-0).
    unitsize_grid = (
        node_eo.join(period_po, how="cross")
        .join(unitsize_sum_df, on="node", how="left")
        .with_columns(pl.col("v_unitsize").fill_null(0.0).alias("value_f"))
    )
    out["_node_cap_unitsize_sum.csv"] = _ordered_value_frame(
        unitsize_grid, ("node", "period", "value"), ["__eo", "__po"],
    )
    # Reconstruct cap_unitsize {n: float} (period-independent) for legacy
    # downstream stages still in scalar form.
    cap_unitsize: dict[str, float] = {n: 0.0 for n in nodes}
    for n, v in unitsize_sum_df.select(["node", "v_unitsize"]).iter_rows():
        cap_unitsize[n] = v

    # ── Stage R: _node_cap_raw (Tier A on real fixtures) ───────────────
    # usz = cap_unitsize.get(n, 0.0); per (n, d):
    #   v = usz if usz > 0 else (fb if (fb := inflow_fallback[(n,d)]) > 0
    #                            else 1.0)
    # Dense over node × period, no drop.
    usz_lk2 = pl.DataFrame(
        {"node": list(nodes),
         "v_usz": [cap_unitsize.get(n, 0.0) for n in nodes]},
        schema={"node": pl.Utf8, "v_usz": pl.Float64},
    )
    fb_lk = pl.DataFrame(
        {"node": [k[0] for k in inflow_fallback],
         "period": [k[1] for k in inflow_fallback],
         "v_fb": list(inflow_fallback.values())},
        schema={"node": pl.Utf8, "period": pl.Utf8, "v_fb": pl.Float64},
    )
    raw_grid = (
        node_eo.join(period_po, how="cross")
        .join(usz_lk2, on="node", how="left")
        .join(fb_lk, on=["node", "period"], how="left")
        .with_columns(
            pl.col("v_usz").fill_null(0.0),
            pl.col("v_fb").fill_null(0.0),
        )
        .with_columns(
            pl.when(pl.col("v_usz") > 0.0)
            .then(pl.col("v_usz"))
            .otherwise(
                pl.when(pl.col("v_fb") > 0.0)
                .then(pl.col("v_fb"))
                .otherwise(pl.lit(1.0)))
            .alias("value_f"),
        )
    )
    out["_node_cap_raw.csv"] = _ordered_value_frame(
        raw_grid, ("node", "period", "value"), ["__eo", "__po"],
    )
    # (raw_dict is not reconstructed: the vectorized P10 stage reads
    # ``raw_grid`` directly rather than iterating a scalar dict.)

    # ── Stage P10: _node_cap_pow10 ─────────────────────────────────────
    # _pow10_round_clamped(raw) per cell — reuse the SAME scalar UDF for
    # byte-fidelity (the Float64 raw lifted above is bit-identical to the
    # legacy Python float).
    pow10_src = raw_grid.select(
        ["node", "period", "__eo", "__po",
         pl.col("value_f").alias("raw")])
    pow10_grid = _pow10_value_column(pow10_src)
    out["_node_cap_pow10.csv"] = _ordered_value_frame(
        pow10_grid, ("node", "period", "value"), ["__eo", "__po"],
    )
    # (pow10_dict is not reconstructed: the vectorized NCFS stage reads
    # ``pow10_grid`` directly rather than iterating a scalar dict.)

    # ── node_capacity_for_scaling + inv_node_cap (Tier B) ──────────────
    # ncfs = pow10 if scaling_active else 1.0 (scaling_active is a Python
    # bool — branch in Python); inc = 1/ncfs if ncfs != 0 else 0.0.  Dense
    # over node × period.  When scaling_active the value frame is the
    # already-computed pow10_grid (every (n, d) is present, no .get default
    # needed); otherwise a constant-1.0 dense grid.
    if scaling_active:
        ncfs_df = pow10_grid.select(
            ["node", "period", "__eo", "__po",
             pl.col("value_f")])
    else:
        ncfs_df = node_eo.join(period_po, how="cross").with_columns(
            pl.lit(1.0, dtype=pl.Float64).alias("value_f"),
        )
    out["node_capacity_for_scaling.csv"] = _ordered_value_frame(
        ncfs_df, ("node", "period", "value"), ["__eo", "__po"],
    )
    inc_df = ncfs_df.with_columns(
        pl.when(pl.col("value_f") != 0.0)
        .then(1.0 / pl.col("value_f"))
        .otherwise(pl.lit(0.0))
        .alias("value_f"),
    )
    out["inv_node_cap.csv"] = _ordered_value_frame(
        inc_df, ("node", "period", "value"), ["__eo", "__po"],
    )
    # (ncfs_dict is not reconstructed: the vectorized group chain reads the
    # ncfs FRAME directly when summing ncfs over a group's member nodes.)

    # Group order frame (group list position → __eo).
    group_eo = pl.DataFrame(
        {"group": list(groups), "__eo": list(range(len(groups)))},
        schema={"group": pl.Utf8, "__eo": pl.Int64},
    )

    # ── _group_cap_raw (Tier B + S5 int-0) — UNCONDITIONAL (Defect Y) ───
    # v = sum(ncfs_dict.get((n, d), 1.0) for n in nodes_for_group.get(g,()))
    # Join group_node→ncfs on node (left-join + fill_null(1.0) reproduces
    # the .get((n, d), 1.0) default for a member node absent from ncfs),
    # group_by([group, period]).sum, densify over group × period.
    # S5 int-0: a group with NO member nodes → sum(()) == int 0 → emit the
    # literal "0" (NOT "0.0").  GR is NOT gated by scaling_active.
    ncfs_lk = ncfs_df.select(
        ["node", "period", pl.col("value_f").alias("v_ncfs")])
    gn_df = pl.DataFrame(
        {"group": [g for g, _n in group_node],
         "node": [n for _g, n in group_node]},
        schema={"group": pl.Utf8, "node": pl.Utf8},
    )
    # group × member-node × period, then sum the ncfs default-1.0 value.
    gn_member_sum = (
        gn_df
        .join(period_po, how="cross")
        .join(ncfs_lk, on=["node", "period"], how="left")
        .with_columns(pl.col("v_ncfs").fill_null(1.0))
        .group_by(["group", "period"])
        .agg(pl.col("v_ncfs").sum().alias("v_graw"))
    )
    graw_grid = (
        group_eo.join(period_po, how="cross")
        .join(gn_member_sum, on=["group", "period"], how="left")
    )
    # Split member-less (null group-sum → int-0 "0") from member-bearing
    # (render the Float64 sum via repr); concat + re-sort by [__eo, __po].
    if graw_grid.height:
        member_rows = graw_grid.filter(pl.col("v_graw").is_not_null())
        memberless_rows = graw_grid.filter(pl.col("v_graw").is_null())
        parts: list[pl.DataFrame] = []
        if member_rows.height:
            parts.append(
                member_rows.with_columns(
                    _render_value_column(member_rows["v_graw"]).alias(
                        "value"))
                .select(["group", "period", "value", "__eo", "__po"]))
        if memberless_rows.height:
            parts.append(
                memberless_rows.with_columns(
                    pl.lit("0", dtype=pl.Utf8).alias("value"))
                .select(["group", "period", "value", "__eo", "__po"]))
        graw_out = (
            pl.concat(parts, how="vertical")
            .sort(["__eo", "__po"])
            .select(["group", "period", "value"])
        )
    else:
        graw_out = _empty_value_frame(("group", "period", "value"))
    out["_group_cap_raw.csv"] = graw_out
    # Reconstruct grp_raw {(g, d): float} (member-less → 0.0; the int/float
    # distinction is irrelevant to the downstream ``v > 0`` GP10 test).
    grp_raw: dict[tuple[str, str], float] = {
        (r[0], r[1]): (r[2] if r[2] is not None else 0.0)
        for r in graw_grid.select(["group", "period", "v_graw"]).iter_rows()
    }

    # ── _group_cap_pow10 — UNCONDITIONAL (Defect Y), scalar UDF ─────────
    # _pow10_round_clamped(graw) if graw > 0 else 1.0.  Reuse the SAME
    # scalar UDF (note: it already returns 1.0 for v <= 0, so the explicit
    # ``if graw > 0 else 1.0`` is subsumed by the UDF — apply it directly).
    gpow10_grid = (
        group_eo.join(period_po, how="cross")
        .join(
            pl.DataFrame(
                {"group": [k[0] for k in grp_raw],
                 "period": [k[1] for k in grp_raw],
                 "raw": list(grp_raw.values())},
                schema={"group": pl.Utf8, "period": pl.Utf8,
                        "raw": pl.Float64}),
            on=["group", "period"], how="left")
    )
    gpow10_grid = _pow10_value_column(gpow10_grid)
    out["_group_cap_pow10.csv"] = _ordered_value_frame(
        gpow10_grid, ("group", "period", "value"), ["__eo", "__po"],
    )
    # (grp_pow10 is not reconstructed: the vectorized GCFS stage reads the
    # gpow10 FRAME directly.)

    # ── group_capacity_for_scaling + inv_group_cap (Tier B) ────────────
    # gcfs = grp_pow10 if scaling_active else 1.0 (scaling_active is a
    # Python bool); igc = 1/gcfs if gcfs != 0 else 0.0.  GCFS IS gated.
    if scaling_active:
        gcfs_df = gpow10_grid.select(
            ["group", "period", "__eo", "__po",
             pl.col("value_f")])
    else:
        gcfs_df = group_eo.join(period_po, how="cross").with_columns(
            pl.lit(1.0, dtype=pl.Float64).alias("value_f"),
        )
    out["group_capacity_for_scaling.csv"] = _ordered_value_frame(
        gcfs_df, ("group", "period", "value"), ["__eo", "__po"],
    )
    igc_df = gcfs_df.with_columns(
        pl.when(pl.col("value_f") != 0.0)
        .then(1.0 / pl.col("value_f"))
        .otherwise(pl.lit(0.0))
        .alias("value_f"),
    )
    out["inv_group_cap.csv"] = _ordered_value_frame(
        igc_df, ("group", "period", "value"), ["__eo", "__po"],
    )

    return out


def emit_lp_scaling_params(
    input_dir: Path, solve_data_dir: Path,
    *, provider,
) -> None:
    """Emit ``lp_scaling_params`` to the Provider.
    Emits the same 9 frames under ``solve_data/<basename>`` keys via
    :func:`_emit` (dual-key registration).
    """
    frames = _compute_lp_scaling_frames_vectorized(
        input_dir, solve_data_dir, provider=provider)
    for basename, df in frames.items():
        _emit(provider, f"solve_data/{basename}", df)
