"""Node inflow scaling params — per-solve emitters.

Called per-solve from ``_emit_solve_time.run`` (batch 17).

Output CSVs (6 emitted — the 6 with a downstream consumer):

* ``ptNode_inflow.csv``                            — (n, t) merged inflow
* ``_node_cap_inflow_fallback.csv``                — (n, d) abs(max)
* ``period_flow_annual_multiplier.csv``            — (n, d) cpsoy / psaf
* ``period_flow_proportional_multiplier.csv``      — (n, d) af / (abs(sum_t)/tdy)
* ``new_old_slope.csv``                            — (n, d) npop * (1 + nom)
* ``new_old_section.csv``                          — (n, d) peak * nom

These feed ``derive_pdtNodeInflow`` (5) and the lp-scaling emitter (1).

The 9 internal middle parameters that used to be emitted as scratch CSVs
— ``orig_flow_sum``, ``period_share_of_annual_flow``, ``new_peak_sign``,
``old_peak_max``, ``old_peak_min``, ``old_peak_sign``,
``new_peak_divided_by_old_peak``, ``new_peak_inflow_sum``,
``new_old_multiplier`` — have NO external consumer (all internal middle
parameters); they are computed in-memory where still needed to feed the 6
consumed outputs, but no longer written.  The legacy oracle
``_compute_inflow_scaling_frames`` still materialises all 15 as the
independent parity reference for the 6 consumed outputs.

Reuse note (Phase 2 sub-dispatch 4 brief)
-----------------------------------------

``flextool.engine_polars._inflow_scaling`` already implements every
per-(n, d) formula here (``_compute_period_share_of_annual_flow``,
``_compute_period_flow_annual_multiplier``,
``_compute_period_flow_proportional_multiplier``, ``_compute_peak_scaling``).
But that helper operates on :class:`InputSource` + per-solve aggregates
and folds all the intermediates into a single ``p_inflow`` Param —
the per-CSV intermediates are never materialised.  Sharing it would
require either threading intermediate accessors through the helper
(cross-cutting) or running a different data path (legacy CSV reads
here, InputSource there).  Instead this writer mirrors the legacy
preprocessing's CSV-in / CSV-out shape verbatim; the formulas match
``_inflow_scaling`` line-by-line so the two paths agree numerically.

Float values formatted with ``repr(float(v))`` for byte-identical
parity with the legacy emitter.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from flextool.engine_polars._emit_provider_io import (
    _emit,
    _provider_key,
)
from flextool.engine_polars._vectorize import _render_value_column


# ---------------------------------------------------------------------------
# Native-frame row helpers.
#
# These read the in-memory polars frame directly from the Provider via
# ``provider.get(_provider_key(path))`` instead of round-tripping through
# CSV text (the legacy ``_provider_open`` + manual ``line.split(",")``
# path, which had no quoted-field handling).
#
# Type-fidelity contract — reproduce *exactly* what the legacy
# ``line.rstrip("\n").split(",")`` over a ``DataFrame.write_csv``
# serialisation would have yielded:
#
#   * Key columns (single-value entries and dict-key positions) were
#     split strings.  ``write_csv`` serialises ``null`` → ``""`` and any
#     scalar (Enum / Int / Float / Utf8) → its string form.  We coerce
#     each key cell with :func:`_cell_str` (``None`` → ``""``, else
#     ``str``) and apply the original truthiness guard to the *string*
#     form so a null cell is skipped (matching the legacy ``if parts[i]``
#     test) while a literal ``"0"`` is kept.
#   * Value columns were re-coerced with ``float(...)``.  We apply
#     ``float(...)`` to the native cell — harmless on an already-float
#     frame, necessary on an int frame, and identical to the legacy
#     ``float(str_cell)`` on a stringified-value frame.  A value that
#     cannot be parsed as a float is skipped: the legacy ``except
#     ValueError`` is widened to ``except (ValueError, TypeError)`` so a
#     native ``None`` value cell (``float(None)`` raises ``TypeError``)
#     is skipped exactly as the legacy ``float("")`` ``ValueError`` was.
#
# ``provider.get`` returns data rows only (no header), so there is no
# header row to skip; an empty / missing frame yields the same empty
# list / dict the legacy loop produced.
# ---------------------------------------------------------------------------


def _cell_str(value: "object | None") -> str:
    """Reproduce a split CSV cell string for a native frame value.

    ``DataFrame.write_csv`` renders ``null`` as the empty string and every
    other scalar as its textual form; the legacy ``line.split(",")`` then
    read those strings back.  Mirror that here so dict keys / single
    values stay byte-identical to the legacy CSV round-trip.
    """
    return "" if value is None else str(value)


def _read_singles(path: Path,
                  *, provider: "object | None" = None) -> list[str]:
    """First-column reader → list of non-empty first-column strings."""
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
    """First-two-column reader → list of (c0, c1) with both non-empty."""
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


def _read_keyed2_float(path: Path,
                       *, provider: "object | None" = None,
                       ) -> dict[tuple[str, str], float]:
    """Three-col frame (key1, key2, value) → {(k1, k2): float}.

    Mirrors legacy ``_read_p`` / ``_read_pt_node_inflow``: malformed
    or non-numeric rows silently skipped.
    """
    out: dict[tuple[str, str], float] = {}
    df = provider.get(_provider_key(path))
    if df is None:
        return out
    for row in df.iter_rows():
        if len(row) < 3:
            continue
        c0, c1 = _cell_str(row[0]), _cell_str(row[1])
        if c0 and c1:
            try:
                out[(c0, c1)] = float(row[2])
            except (ValueError, TypeError):
                continue
    return out


def _read_keyed3_float(path: Path,
                       *, provider: "object | None" = None,
                       ) -> dict[tuple[str, str, str], float]:
    """Four-col frame (k1, k2, k3, value) → {(k1, k2, k3): float}."""
    out: dict[tuple[str, str, str], float] = {}
    df = provider.get(_provider_key(path))
    if df is None:
        return out
    for row in df.iter_rows():
        if len(row) < 4:
            continue
        c0, c1, c2 = _cell_str(row[0]), _cell_str(row[1]), _cell_str(row[2])
        if c0 and c1 and c2:
            try:
                out[(c0, c1, c2)] = float(row[3])
            except (ValueError, TypeError):
                continue
    return out


def _read_keyed_float(path: Path,
                      *, provider: "object | None" = None,
                      ) -> dict[str, float]:
    """Two-col frame (key, value) → {key: float}."""
    out: dict[str, float] = {}
    df = provider.get(_provider_key(path))
    if df is None:
        return out
    for row in df.iter_rows():
        if len(row) < 2:
            continue
        c0 = _cell_str(row[0])
        if c0:
            try:
                out[c0] = float(row[1])
            except (ValueError, TypeError):
                continue
    return out


def _rows_to_frame(
    header: tuple[str, str, str],
    rows: list[tuple[str, str, float]],
) -> pl.DataFrame:
    """Materialise (k1, k2, repr(value)) rows as an all-Utf8 frame.

    Pre-stringifies values via ``repr(v)`` so the polars ``write_csv``
    output is byte-identical to the legacy ``f"{a},{b},{repr(v)}\\n"``
    text emitter.  Note this uses ``repr(v)`` NOT ``repr(float(v))`` —
    the legacy emitter assumes the caller already produced floats.
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
# Family — write_node_inflow_scaling_params
# ---------------------------------------------------------------------------


def _compute_inflow_scaling_frames(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> dict[str, pl.DataFrame]:
    """Compute every inflow-scaling CSV in one pass, returning a dict
    keyed by output basename.

    Retained as the parity oracle for
    :func:`_compute_inflow_scaling_frames_vectorized`
    (``tests/engine_polars/test_vectorize_inflow_scaling_parity.py``); the
    live emit path is the vectorized twin.  The cross-CSV state (peak
    family, npop / nom / etc) is heavy enough that splitting into
    independent per-CSV passes would re-walk the time axis O(N) times — the
    dict-of-frames pattern from the audit doc is the appropriate adapter
    here.
    """
    out: dict[str, pl.DataFrame] = {}

    # ── Sources ────────────────────────────────────────────────────────
    nodes = _read_singles(input_dir / "node.csv", provider=provider)
    period_in_use = _read_singles(solve_data_dir / "period_in_use_set.csv",
                                   provider=provider)
    time_set = _read_singles(solve_data_dir / "time.csv", provider=provider)
    p_node = _read_keyed2_float(input_dir / "p_node.csv", provider=provider)
    pt_node_inflow = _read_keyed2_float(
        solve_data_dir / "pt_node_inflow.csv", provider=provider,
    )
    # The "(n, t) is explicitly set" predicate uses ONLY (n, t) presence
    # in pt_node_inflow.csv — values are read separately above.
    node_time_inflow = frozenset(_read_pairs(
        solve_data_dir / "pt_node_inflow.csv", provider=provider,
    ))

    inflow_method = _read_pairs(solve_data_dir / "node__inflow_method.csv",
                                provider=provider)
    methods_for_node: dict[str, set[str]] = {}
    for n, m in inflow_method:
        methods_for_node.setdefault(n, set()).add(m)

    pdNode = _read_keyed3_float(solve_data_dir / "pdNode.csv", provider=provider)
    cpsoy = _read_keyed_float(
        solve_data_dir / "complete_period_share_of_year_calc.csv",
        provider=provider,
    )
    p_tdy = _read_keyed_float(
        solve_data_dir / "p_timeline_duration_in_years.csv",
        provider=provider,
    )

    # period__timeline (Python output): list of (period, timeline) pairs.
    period_timeline = _read_pairs(solve_data_dir / "period__timeline_set.csv",
                                   provider=provider)
    timelines_for_d: dict[str, list[str]] = {}
    for d, tl in period_timeline:
        timelines_for_d.setdefault(d, []).append(tl)

    # complete_time_in_use (Python output, _set suffix).
    complete_time_in_use = _read_singles(
        solve_data_dir / "complete_time_in_use_set.csv",
        provider=provider,
    )

    # dt_complete from steps_complete_solve.csv — (period, step) pairs.
    dt_complete_pairs = _read_pairs(
        solve_data_dir / "steps_complete_solve.csv",
        provider=provider,
    )
    dt_complete_for_d: dict[str, list[str]] = {}
    for d, t in dt_complete_pairs:
        dt_complete_for_d.setdefault(d, []).append(t)

    # ── ptNode_inflow{n in node, t in time} ───────────────────────────
    # Per legacy mod L1237: pass-through pt_node_inflow when (n, t) is
    # in the explicit set, else the per-node scalar default from
    # p_node[(n, "inflow")] (typically 0).
    p_node_inflow_default = {
        n: p_node.get((n, "inflow"), 0.0) for n in nodes
    }
    pti: dict[tuple[str, str], float] = {}
    for n in nodes:
        default = p_node_inflow_default[n]
        for t in time_set:
            if (n, t) in node_time_inflow:
                pti[(n, t)] = pt_node_inflow.get((n, t), 0.0)
            else:
                pti[(n, t)] = default

    rows_pt = [(n, t, pti[(n, t)]) for n in nodes for t in time_set]
    out["ptNode_inflow.csv"] = _rows_to_frame(
        ("node", "time", "value"), rows_pt,
    )

    # ── _node_cap_inflow_fallback{n in node, d in period_in_use} ──────
    # value = max_{t in time} abs(ptNode_inflow[n, t]); 0 if no time.
    fallback_rows: list[tuple[str, str, float]] = []
    if not time_set:
        for n in nodes:
            for d in period_in_use:
                fallback_rows.append((n, d, 0.0))
    else:
        for n in nodes:
            max_abs = max(abs(pti[(n, t)]) for t in time_set)
            for d in period_in_use:
                fallback_rows.append((n, d, max_abs))
    out["_node_cap_inflow_fallback.csv"] = _rows_to_frame(
        ("node", "period", "value"), fallback_rows,
    )

    # Helper: does node n have inflow method m?
    def _has_method(n: str, m: str) -> bool:
        return m in methods_for_node.get(n, ())

    # Domain predicates -- legacy guards each writer with:
    #   methods AND pdNode[..annual_flow..] != 0 [AND peak_inflow != 0]
    def _annual_eligible(n: str) -> bool:
        return (_has_method(n, "scale_to_annual_flow")
                or _has_method(n, "scale_to_annual_and_peak_flow"))

    # ── orig_flow_sum ─────────────────────────────────────────────────
    # value = sum_{t in complete_time_in_use} ptNode_inflow[n, t].
    # Domain: (n, d) where annual_eligible AND pdNode annual_flow != 0.
    rows_orig: list[tuple[str, str, float]] = []
    # Cache the t-sum per node (it's t-axis only — period-independent).
    sum_complete_inflow: dict[str, float] = {}
    for n in nodes:
        if not _annual_eligible(n):
            continue
        sum_complete_inflow[n] = sum(
            pti[(n, t)] for t in complete_time_in_use
        )
    for n in nodes:
        if not _annual_eligible(n):
            continue
        s = sum_complete_inflow[n]
        for d in period_in_use:
            if pdNode.get((n, "annual_flow", d), 0.0) == 0.0:
                continue
            rows_orig.append((n, d, s))
    out["orig_flow_sum.csv"] = _rows_to_frame(
        ("node", "period", "value"), rows_orig,
    )
    orig_flow_sum = {(n, d): v for n, d, v in rows_orig}

    # ── period_share_of_annual_flow ───────────────────────────────────
    # value = abs(sum_{t in dt_complete[d]} ptNode_inflow[n, t])
    #         / pdNode[n, 'annual_flow', d].
    rows_psaf: list[tuple[str, str, float]] = []
    for n in nodes:
        if not _annual_eligible(n):
            continue
        for d in period_in_use:
            af = pdNode.get((n, "annual_flow", d), 0.0)
            if af == 0.0:
                continue
            s = sum(pti[(n, t)] for t in dt_complete_for_d.get(d, ()))
            rows_psaf.append((n, d, abs(s) / af))
    out["period_share_of_annual_flow.csv"] = _rows_to_frame(
        ("node", "period", "value"), rows_psaf,
    )
    psaf = {(n, d): v for n, d, v in rows_psaf}

    # ── period_flow_annual_multiplier ─────────────────────────────────
    # value = complete_period_share_of_year[d] / period_share_of_annual_flow[n, d].
    # Domain: (n, d) where scale_to_annual_flow AND pdNode annual_flow.
    rows_pfam: list[tuple[str, str, float]] = []
    for n in nodes:
        if not _has_method(n, "scale_to_annual_flow"):
            continue
        for d in period_in_use:
            if pdNode.get((n, "annual_flow", d), 0.0) == 0.0:
                continue
            denom = psaf.get((n, d), 0.0)
            if denom == 0.0:
                continue
            rows_pfam.append((n, d, cpsoy.get(d, 0.0) / denom))
    out["period_flow_annual_multiplier.csv"] = _rows_to_frame(
        ("node", "period", "value"), rows_pfam,
    )

    # ── period_flow_proportional_multiplier ───────────────────────────
    # value = pdNode[n, 'annual_flow', d] /
    #         (abs(sum_{t in time} ptNode_inflow[n, t]) /
    #          sum_{tl in period__timeline[d]} p_timeline_duration_in_years[tl]).
    rows_pfpm: list[tuple[str, str, float]] = []
    # time_sum is t-axis only.
    sum_time_inflow_by_n: dict[str, float] = {}
    for n in nodes:
        if not _has_method(n, "scale_in_proportion"):
            continue
        sum_time_inflow_by_n[n] = sum(pti[(n, t)] for t in time_set)
    for n in nodes:
        if not _has_method(n, "scale_in_proportion"):
            continue
        time_sum = sum_time_inflow_by_n[n]
        for d in period_in_use:
            af = pdNode.get((n, "annual_flow", d), 0.0)
            if af == 0.0:
                continue
            tdy_sum = sum(p_tdy.get(tl, 0.0)
                          for tl in timelines_for_d.get(d, ()))
            if tdy_sum == 0.0 or time_sum == 0.0:
                continue
            rows_pfpm.append((n, d, af / (abs(time_sum) / tdy_sum)))
    out["period_flow_proportional_multiplier.csv"] = _rows_to_frame(
        ("node", "period", "value"), rows_pfpm,
    )

    # ── Peak-flow family (annual_and_peak_flow) ────────────────────────
    # Domain: (n, d) where scale_to_annual_and_peak_flow AND pdNode
    # annual_flow != 0 AND pdNode peak_inflow != 0.
    # Per-node availability of explicit (n, t) inflow rows — this drives
    # the op_max / op_min "use scalar default" fallback path.
    has_node_time_inflow: dict[str, bool] = {
        n: any(nn == n for (nn, _t) in node_time_inflow) for n in nodes
    }
    # Per-node max/min across t (independent of d).
    op_max_by_n: dict[str, float] = {}
    op_min_by_n: dict[str, float] = {}
    op_sign_by_n: dict[str, float] = {}
    old_peak_by_n: dict[str, float] = {}
    for n in nodes:
        if has_node_time_inflow[n]:
            inflow_vals = [pti[(n, t)] for t in time_set]
            op_max = max(inflow_vals) if inflow_vals else 0.0
            op_min = min(inflow_vals) if inflow_vals else 0.0
        else:
            scalar = p_node_inflow_default[n]
            op_max = scalar
            op_min = scalar
        op_max_by_n[n] = op_max
        op_min_by_n[n] = op_min
        if has_node_time_inflow[n]:
            op_sign = 1.0 if abs(op_max) >= abs(op_min) else -1.0
        else:
            op_sign = 1.0 if p_node_inflow_default[n] >= 0 else -1.0
        op_sign_by_n[n] = op_sign
        old_peak_by_n[n] = op_max if op_sign >= 0 else op_min

    rows_nps: list[tuple[str, str, float]] = []
    rows_opmax: list[tuple[str, str, float]] = []
    rows_opmin: list[tuple[str, str, float]] = []
    rows_ops: list[tuple[str, str, float]] = []
    rows_npop: list[tuple[str, str, float]] = []
    rows_npopinflow: list[tuple[str, str, float]] = []

    def _peak_domain(n: str, d: str) -> bool:
        return (
            _has_method(n, "scale_to_annual_and_peak_flow")
            and pdNode.get((n, "annual_flow", d), 0.0) != 0.0
            and pdNode.get((n, "peak_inflow", d), 0.0) != 0.0
        )

    for n in nodes:
        for d in period_in_use:
            if not _peak_domain(n, d):
                continue
            peak = pdNode.get((n, "peak_inflow", d), 0.0)
            rows_nps.append((n, d, 1.0 if peak >= 0 else -1.0))
            rows_opmax.append((n, d, op_max_by_n[n]))
            rows_opmin.append((n, d, op_min_by_n[n]))
            rows_ops.append((n, d, op_sign_by_n[n]))
            old_peak_val = old_peak_by_n[n]
            if old_peak_val == 0.0:
                # Legacy skips downstream rows when old_peak is 0 (avoids
                # division by zero) — npop / npopinflow remain absent.
                continue
            npop = peak / old_peak_val
            rows_npop.append((n, d, npop))

            ofs = orig_flow_sum.get((n, d), 0.0)
            cps = cpsoy.get(d, 0.0)
            npopis = (npop * ofs / cps) if cps != 0.0 else 0.0
            rows_npopinflow.append((n, d, npopis))

    out["new_peak_sign.csv"] = _rows_to_frame(
        ("node", "period", "value"), rows_nps,
    )
    out["old_peak_max.csv"] = _rows_to_frame(
        ("node", "period", "value"), rows_opmax,
    )
    out["old_peak_min.csv"] = _rows_to_frame(
        ("node", "period", "value"), rows_opmin,
    )
    out["old_peak_sign.csv"] = _rows_to_frame(
        ("node", "period", "value"), rows_ops,
    )
    out["new_peak_divided_by_old_peak.csv"] = _rows_to_frame(
        ("node", "period", "value"), rows_npop,
    )

    # ── new_peak_inflow_sum, new_old_multiplier/slope/section ─────────
    # Same domain as rows_nps; values derived from peak / npop / npopis.
    npis_dict = {
        (n, d): pdNode.get((n, "peak_inflow", d), 0.0) * 8760.0
        for n, d, _ in rows_nps
    }
    rows_npis = [(n, d, npis_dict.get((n, d), 0.0)) for n, d, _ in rows_nps]
    out["new_peak_inflow_sum.csv"] = _rows_to_frame(
        ("node", "period", "value"), rows_npis,
    )

    op_sign_dict = {(n, d): v for n, d, v in rows_ops}
    npopinflow_dict = {(n, d): v for n, d, v in rows_npopinflow}
    rows_nom: list[tuple[str, str, float]] = []
    for n, d, _ in rows_nps:
        npis = npis_dict.get((n, d), 0.0)
        npopis = npopinflow_dict.get((n, d), 0.0)
        os_sign = op_sign_dict.get((n, d), 0.0)
        af = pdNode.get((n, "annual_flow", d), 0.0)
        denom = npis - npopis
        if denom == 0.0:
            v = 0.0
        else:
            v = os_sign * (os_sign * npopis - af) / denom
        rows_nom.append((n, d, v))
    out["new_old_multiplier.csv"] = _rows_to_frame(
        ("node", "period", "value"), rows_nom,
    )

    nom_dict = {(n, d): v for n, d, v in rows_nom}
    npop_dict = {(n, d): v for n, d, v in rows_npop}
    rows_nos = [
        (n, d, npop_dict.get((n, d), 0.0)
                * (1.0 + nom_dict.get((n, d), 0.0)))
        for n, d, _ in rows_nps
    ]
    out["new_old_slope.csv"] = _rows_to_frame(
        ("node", "period", "value"), rows_nos,
    )

    rows_nosec = [
        (n, d, pdNode.get((n, "peak_inflow", d), 0.0)
                * nom_dict.get((n, d), 0.0))
        for n, d, _ in rows_nps
    ]
    out["new_old_section.csv"] = _rows_to_frame(
        ("node", "period", "value"), rows_nosec,
    )

    return out


# ---------------------------------------------------------------------------
# Vectorized twin of _compute_inflow_scaling_frames (vectorize-per-roll).
#
# Built as a FULL COPY of the legacy body, with each stage's COMPUTE
# replaced by vectorized polars.  Only the 6 CONSUMED outputs are emitted;
# the 9 internal middle parameters are computed in-memory where still needed
# (orig_flow_sum / psaf dicts, inline fused912 npop/npis/nom) but never
# written.  The legacy _compute_inflow_scaling_frames is KEPT as the parity
# oracle (it still materialises all 15; the test compares the 6).
#
# Tier policy: ptNode_inflow + _node_cap_inflow_fallback are sum-free
# (coalesce / max-abs) and MUST stay byte-identical (Tier A — read by the
# already-vectorized pdtNodeInflow and by lp-scaling).  The sum-bearing
# consumed stages (period_flow_annual_multiplier,
# period_flow_proportional_multiplier, new_old_slope, new_old_section) are
# Tier B (last-ULP drift tolerated).
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


def _compute_inflow_scaling_frames_vectorized(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> dict[str, pl.DataFrame]:
    """Vectorized twin of :func:`_compute_inflow_scaling_frames`.

    Same reader block, same in-memory dicts, same stage order.  Only the
    6 CONSUMED outputs are assigned to ``out`` (``ptNode_inflow``,
    ``_node_cap_inflow_fallback``, ``period_flow_annual_multiplier``,
    ``period_flow_proportional_multiplier``, ``new_old_slope``,
    ``new_old_section``).  The 9 internal middle parameters
    (``orig_flow_sum``, ``period_share_of_annual_flow``, ``new_peak_sign``,
    ``old_peak_max``, ``old_peak_min``, ``old_peak_sign``,
    ``new_peak_divided_by_old_peak``, ``new_peak_inflow_sum``,
    ``new_old_multiplier``) have no external consumer and are no longer
    emitted; the values still needed downstream live in-memory only
    (``orig_flow_sum`` dict, ``psaf`` dict, and the inline fused912
    ``v_npop`` / ``v_npis`` / ``v_nom``).  The legacy oracle
    materialises all 15 and gates the 6 consumed.
    """
    out: dict[str, pl.DataFrame] = {}

    # ── Sources (copied verbatim from the legacy reader block) ─────────
    nodes = _read_singles(input_dir / "node.csv", provider=provider)
    period_in_use = _read_singles(solve_data_dir / "period_in_use_set.csv",
                                   provider=provider)
    time_set = _read_singles(solve_data_dir / "time.csv", provider=provider)
    p_node = _read_keyed2_float(input_dir / "p_node.csv", provider=provider)
    pt_node_inflow = _read_keyed2_float(
        solve_data_dir / "pt_node_inflow.csv", provider=provider,
    )
    node_time_inflow = frozenset(_read_pairs(
        solve_data_dir / "pt_node_inflow.csv", provider=provider,
    ))

    inflow_method = _read_pairs(solve_data_dir / "node__inflow_method.csv",
                                provider=provider)
    methods_for_node: dict[str, set[str]] = {}
    for n, m in inflow_method:
        methods_for_node.setdefault(n, set()).add(m)

    pdNode = _read_keyed3_float(solve_data_dir / "pdNode.csv", provider=provider)
    cpsoy = _read_keyed_float(
        solve_data_dir / "complete_period_share_of_year_calc.csv",
        provider=provider,
    )
    p_tdy = _read_keyed_float(
        solve_data_dir / "p_timeline_duration_in_years.csv",
        provider=provider,
    )

    period_timeline = _read_pairs(solve_data_dir / "period__timeline_set.csv",
                                   provider=provider)
    timelines_for_d: dict[str, list[str]] = {}
    for d, tl in period_timeline:
        timelines_for_d.setdefault(d, []).append(tl)

    complete_time_in_use = _read_singles(
        solve_data_dir / "complete_time_in_use_set.csv",
        provider=provider,
    )

    dt_complete_pairs = _read_pairs(
        solve_data_dir / "steps_complete_solve.csv",
        provider=provider,
    )
    dt_complete_for_d: dict[str, list[str]] = {}
    for d, t in dt_complete_pairs:
        dt_complete_for_d.setdefault(d, []).append(t)

    # ── Shared order frames (node list order, period order, time order) ─
    node_eo = pl.DataFrame(
        {"node": list(nodes), "__eo": list(range(len(nodes)))},
        schema={"node": pl.Utf8, "__eo": pl.Int64},
    )
    period_po = pl.DataFrame(
        {"period": list(period_in_use),
         "__po": list(range(len(period_in_use)))},
        schema={"period": pl.Utf8, "__po": pl.Int64},
    )
    time_to = pl.DataFrame(
        {"time": list(time_set), "__to": list(range(len(time_set)))},
        schema={"time": pl.Utf8, "__to": pl.Int64},
    )

    # ── Stage 1: ptNode_inflow{n in node, t in time} (Tier A) ──────────
    # value = pt_node_inflow[(n, t)] if (n, t) in node_time_inflow
    #         else p_node[(n, "inflow")] (per-node scalar default).
    p_node_inflow_default = {
        n: p_node.get((n, "inflow"), 0.0) for n in nodes
    }
    nt_grid = node_eo.join(time_to, how="cross")
    # pt_node_inflow lookup frame (value present when (n, t) explicit).
    pti_lk = pl.DataFrame(
        {"node": [k[0] for k in pt_node_inflow],
         "time": [k[1] for k in pt_node_inflow],
         "v_pti": list(pt_node_inflow.values())},
        schema={"node": pl.Utf8, "time": pl.Utf8, "v_pti": pl.Float64},
    )
    # explicit-set membership frame (presence in node_time_inflow).
    nti_lk = pl.DataFrame(
        {"node": [k[0] for k in node_time_inflow],
         "time": [k[1] for k in node_time_inflow],
         "__nti": [True] * len(node_time_inflow)},
        schema={"node": pl.Utf8, "time": pl.Utf8, "__nti": pl.Boolean},
    )
    # per-node scalar default frame.
    dflt_lk = pl.DataFrame(
        {"node": list(nodes),
         "v_dflt": [p_node_inflow_default[n] for n in nodes]},
        schema={"node": pl.Utf8, "v_dflt": pl.Float64},
    )
    pti_df = (
        nt_grid
        .join(nti_lk, on=["node", "time"], how="left")
        .join(pti_lk, on=["node", "time"], how="left")
        .join(dflt_lk, on="node", how="left")
        .with_columns(
            pl.when(pl.col("__nti").fill_null(False))  # noqa: FBT003
            # explicit: pt_node_inflow.get((n,t), 0.0)
            .then(pl.col("v_pti").fill_null(0.0))
            # else: per-node scalar default
            .otherwise(pl.col("v_dflt").fill_null(0.0))
            .alias("value_f"),
        )
    )
    out["ptNode_inflow.csv"] = _ordered_value_frame(
        pti_df, ("node", "time", "value"), ["__eo", "__to"],
    )
    # NOTE: every downstream consumer of the per-(n, t) inflow series is now
    # vectorized (it reads ``pti_df`` / ``pti_series`` directly), so the
    # scalar ``pti`` dict is no longer reconstructed here.

    # ── Stage 2: _node_cap_inflow_fallback{n, d} (Tier A) ──────────────
    # value = max_t abs(pti[(n, t)]); 0.0 if no time.
    if not time_set:
        nd_grid = node_eo.join(period_po, how="cross").with_columns(
            pl.lit(0.0, dtype=pl.Float64).alias("value_f"),
        )
        out["_node_cap_inflow_fallback.csv"] = _ordered_value_frame(
            nd_grid, ("node", "period", "value"), ["__eo", "__po"],
        )
    else:
        max_abs_df = (
            pti_df.group_by("node")
            .agg(pl.col("value_f").abs().max().alias("v_maxabs"))
        )
        nd_grid = (
            node_eo.join(period_po, how="cross")
            .join(max_abs_df, on="node", how="left")
            .with_columns(
                pl.col("v_maxabs").fill_null(0.0).alias("value_f"),
            )
        )
        out["_node_cap_inflow_fallback.csv"] = _ordered_value_frame(
            nd_grid, ("node", "period", "value"), ["__eo", "__po"],
        )

    # Helper: does node n have inflow method m? (legacy parity)
    def _has_method(n: str, m: str) -> bool:
        return m in methods_for_node.get(n, ())

    def _annual_eligible(n: str) -> bool:
        return (_has_method(n, "scale_to_annual_flow")
                or _has_method(n, "scale_to_annual_and_peak_flow"))

    # ── Shared C2 building blocks ──────────────────────────────────────
    # FOUR DISTINCT method masks (Defect A — do NOT collapse onto one):
    #   annual_eligible        → stages 3, 4
    #   scale_to_annual_flow   → stage 5 (a peak-only node is EXCLUDED)
    #   scale_in_proportion    → stage 6 (disjoint from the annual methods)
    #   scale_to_annual_and_peak_flow → stages 7, 8 (C3 peak family)
    # Each as an ordered ``(node, __eo)`` frame so the entity-major
    # emission order survives the joins.
    _node_idx = {n: i for i, n in enumerate(nodes)}

    def _ordered_node_frame(pred) -> pl.DataFrame:
        sel = [n for n in nodes if pred(n)]
        return pl.DataFrame(
            {"node": sel, "__eo": [_node_idx[n] for n in sel]},
            schema={"node": pl.Utf8, "__eo": pl.Int64},
        )

    annual_eligible_eo = _ordered_node_frame(_annual_eligible)
    annual_flow_only_eo = _ordered_node_frame(
        lambda n: _has_method(n, "scale_to_annual_flow"))
    proportion_eo = _ordered_node_frame(
        lambda n: _has_method(n, "scale_in_proportion"))

    # pdNode annual_flow lookup keyed (node, period) → af.
    af_keys = [(k[0], k[2]) for k in pdNode if k[1] == "annual_flow"]
    af_vals = [v for k, v in pdNode.items() if k[1] == "annual_flow"]
    af_lk = pl.DataFrame(
        {"node": [k[0] for k in af_keys],
         "period": [k[1] for k in af_keys],
         "v_af": af_vals},
        schema={"node": pl.Utf8, "period": pl.Utf8, "v_af": pl.Float64},
    )

    # Per-node complete-timeline t-sum (period-independent): the legacy
    # ``sum(pti[(n,t)] for t in complete_time_in_use)``.  Lift the t-list
    # to a frame and inner-join the per-node inflow series, then group-sum.
    cti_lk = pl.DataFrame(
        {"time": list(complete_time_in_use)},
        schema={"time": pl.Utf8},
    )
    # pti as a (node, time, v_pti) frame restricted to the per-node series.
    pti_series = pti_df.select(["node", "time", "value_f"]).rename(
        {"value_f": "v_pti"})
    complete_sum_df = (
        pti_series.join(cti_lk, on="time", how="inner")
        .group_by("node")
        .agg(pl.col("v_pti").sum().alias("v_complete"))
    )

    # ── Stage 3: orig_flow_sum (in-memory only — feeds npopis) ─────────
    # Domain: annual_eligible AND pdNode annual_flow != 0 (DROP on 0/miss).
    # value = per-node complete-timeline sum (period-independent).  NOT
    # emitted (no external consumer); only the dict below is used (by the
    # fused912 npopis).  Because the CSV is no longer written there is no
    # byte-parity contract — the empty-timeline ``sum(())`` int-0 ``"0"``
    # vs ``"0.0"`` special-case (which existed ONLY to byte-match the
    # dropped CSV) is gone; value_f is a plain Float64 sum.
    orig_df = (
        annual_eligible_eo
        .join(period_po, how="cross")
        .join(af_lk, on=["node", "period"], how="left")
        .filter(pl.col("v_af").fill_null(0.0) != 0.0)
        .join(complete_sum_df, on="node", how="left")
        # A node with NO complete-timeline rows contributes 0: the legacy
        # ``sum(())`` over an empty complete_time_in_use is 0.
        .with_columns(pl.col("v_complete").fill_null(0.0).alias("value_f"))
    )
    # Reconstruct orig_flow_sum dict (float values) for downstream stages.
    orig_flow_sum = {
        (r[0], r[1]): r[2]
        for r in orig_df.select(["node", "period", "value_f"]).iter_rows()
    }

    # ── Stage 4: period_share_of_annual_flow (Tier B) ──────────────────
    # Domain: annual_eligible AND af != 0 (DROP on 0/miss).
    # value = abs(sum_{t in dt_complete[d]} pti[(n,t)]) / af.
    # Per-(node, period) dt_complete sum: lift the (period, time) pairs.
    dtc_pairs = pl.DataFrame(
        {"period": [d for d, _t in dt_complete_pairs],
         "time": [t for _d, t in dt_complete_pairs]},
        schema={"period": pl.Utf8, "time": pl.Utf8},
    )
    # node × (period, time) restricted to annual_eligible nodes only, then
    # group-sum the inflow series per (node, period).
    dtc_sum_df = (
        annual_eligible_eo.select("node")
        .join(pti_series, on="node", how="inner")
        .join(dtc_pairs, on="time", how="inner")
        .group_by(["node", "period"])
        .agg(pl.col("v_pti").sum().alias("v_dtc"))
    )
    psaf_df = (
        annual_eligible_eo
        .join(period_po, how="cross")
        .join(af_lk, on=["node", "period"], how="left")
        .filter(pl.col("v_af").fill_null(0.0) != 0.0)
        .join(dtc_sum_df, on=["node", "period"], how="left")
        # A (node, period) with no dt_complete rows: legacy sum(()) == 0.
        .with_columns(
            (pl.col("v_dtc").fill_null(0.0).abs() / pl.col("v_af"))
            .alias("value_f"),
        )
    )
    # NOT emitted (no external consumer); only the psaf dict below is used
    # (by stage 5 = period_flow_annual_multiplier).
    psaf = {
        (r[0], r[1]): r[2]
        for r in psaf_df.select(["node", "period", "value_f"]).iter_rows()
    }

    # ── Stage 6: period_flow_proportional_multiplier (Tier B) ──────────
    # Domain: scale_in_proportion AND af != 0; DROP if tdy_sum==0 OR
    # time_sum==0.  value = af / (abs(time_sum) / tdy_sum).
    # Per-node time-axis sum over the WHOLE time_set.
    time_sum_df = (
        proportion_eo.select("node")
        .join(pti_series, on="node", how="inner")
        .group_by("node")
        .agg(pl.col("v_pti").sum().alias("v_timesum"))
    )
    # Per-period tdy sum: sum p_tdy over timelines_for_d[d].
    pt_pairs = pl.DataFrame(
        {"period": [d for d, _tl in period_timeline],
         "timeline": [tl for _d, tl in period_timeline]},
        schema={"period": pl.Utf8, "timeline": pl.Utf8},
    )
    tdy_lk = pl.DataFrame(
        {"timeline": list(p_tdy.keys()), "v_tdy": list(p_tdy.values())},
        schema={"timeline": pl.Utf8, "v_tdy": pl.Float64},
    )
    tdy_sum_df = (
        pt_pairs
        .join(tdy_lk, on="timeline", how="left")
        .group_by("period")
        .agg(pl.col("v_tdy").fill_null(0.0).sum().alias("v_tdysum"))
    )
    pfpm_df = (
        proportion_eo
        .join(period_po, how="cross")
        .join(af_lk, on=["node", "period"], how="left")
        .filter(pl.col("v_af").fill_null(0.0) != 0.0)
        .join(time_sum_df, on="node", how="left")
        .join(tdy_sum_df, on="period", how="left")
        # A node missing the time-sum join means no inflow rows ⇒ legacy
        # sum(()) == 0.0; a period missing tdy_sum ⇒ legacy sum(()) == 0.0.
        .with_columns(
            pl.col("v_timesum").fill_null(0.0).alias("v_timesum"),
            pl.col("v_tdysum").fill_null(0.0).alias("v_tdysum"),
        )
        .filter(
            (pl.col("v_tdysum") != 0.0) & (pl.col("v_timesum") != 0.0))
        .with_columns(
            (pl.col("v_af")
             / (pl.col("v_timesum").abs() / pl.col("v_tdysum")))
            .alias("value_f"),
        )
    )
    out["period_flow_proportional_multiplier.csv"] = _ordered_value_frame(
        pfpm_df, ("node", "period", "value"), ["__eo", "__po"],
    )

    # ── Stage 5: period_flow_annual_multiplier (Tier B) ────────────────
    # Domain: scale_to_annual_flow ONLY (Defect A — a peak-only node is
    # annual_eligible but EXCLUDED here).  THREE filters, TWO semantics
    # (Defect B):
    #   (a) af==0 → DROP (inner-join af_lk + filter af!=0),
    #   (b) psaf.get((n,d),0.0)==0 → DROP (inner-join psaf + filter !=0 →
    #       drops BOTH a miss AND an exact-0),
    #   (c) cpsoy.get(d,0.0) numerator → SURVIVE-with-0 (left-join cpsoy +
    #       fill_null(0.0)).
    # value = cpsoy_filled / psaf.
    psaf_lk = pl.DataFrame(
        {"node": [k[0] for k in psaf],
         "period": [k[1] for k in psaf],
         "v_psaf": list(psaf.values())},
        schema={"node": pl.Utf8, "period": pl.Utf8, "v_psaf": pl.Float64},
    )
    cpsoy_lk = pl.DataFrame(
        {"period": list(cpsoy.keys()), "v_cpsoy": list(cpsoy.values())},
        schema={"period": pl.Utf8, "v_cpsoy": pl.Float64},
    )
    pfam_df = (
        annual_flow_only_eo
        .join(period_po, how="cross")
        # (a) af==0/miss → DROP.
        .join(af_lk, on=["node", "period"], how="left")
        .filter(pl.col("v_af").fill_null(0.0) != 0.0)
        # (b) psaf miss OR exact-0 → DROP.
        .join(psaf_lk, on=["node", "period"], how="left")
        .filter(pl.col("v_psaf").fill_null(0.0) != 0.0)
        # (c) cpsoy numerator → survive-with-0.
        .join(cpsoy_lk, on="period", how="left")
        .with_columns(
            (pl.col("v_cpsoy").fill_null(0.0) / pl.col("v_psaf"))
            .alias("value_f"),
        )
    )
    out["period_flow_annual_multiplier.csv"] = _ordered_value_frame(
        pfam_df, ("node", "period", "value"), ["__eo", "__po"],
    )

    # ── Stage 7: per-node peak precompute (Tier A: max/min/sign) ───────
    # has_node_time_inflow[n] = node has ANY explicit (n, t) inflow row.
    peak_eo = _ordered_node_frame(
        lambda n: _has_method(n, "scale_to_annual_and_peak_flow"))
    hnti_nodes = {nn for (nn, _t) in node_time_inflow}
    # Per-node max/min over the WHOLE time series (dense grid).  When
    # time_set is empty the group has no rows → null → fill 0.0 (matches
    # the legacy ``max(inflow_vals) if inflow_vals else 0.0``).
    minmax_df = (
        pti_df.group_by("node")
        .agg(
            pl.col("value_f").max().alias("v_tmax"),
            pl.col("value_f").min().alias("v_tmin"),
        )
    )
    # Per-node precompute over ALL nodes (the legacy loop iterates nodes).
    precomp = (
        node_eo
        .join(minmax_df, on="node", how="left")
        .with_columns(
            pl.col("node").is_in(list(hnti_nodes)).alias("__hnti"),
            pl.col("node").replace_strict(
                p_node_inflow_default, default=0.0,
                return_dtype=pl.Float64).alias("v_dflt"),
        )
        .with_columns(
            # op_max / op_min: time-series bound when has_node_time_inflow
            # (null → 0.0 for an empty time_set), else the scalar default.
            pl.when(pl.col("__hnti"))
            .then(pl.col("v_tmax").fill_null(0.0))
            .otherwise(pl.col("v_dflt")).alias("op_max"),
            pl.when(pl.col("__hnti"))
            .then(pl.col("v_tmin").fill_null(0.0))
            .otherwise(pl.col("v_dflt")).alias("op_min"),
        )
        .with_columns(
            # op_sign: has_node_time_inflow → 1 if |max|>=|min| else -1;
            # else 1 if default >= 0 else -1.
            pl.when(pl.col("__hnti"))
            .then(
                pl.when(pl.col("op_max").abs() >= pl.col("op_min").abs())
                .then(pl.lit(1.0)).otherwise(pl.lit(-1.0)))
            .otherwise(
                pl.when(pl.col("v_dflt") >= 0.0)
                .then(pl.lit(1.0)).otherwise(pl.lit(-1.0)))
            .alias("op_sign"),
        )
        .with_columns(
            # old_peak = op_max if op_sign >= 0 else op_min.
            pl.when(pl.col("op_sign") >= 0.0)
            .then(pl.col("op_max")).otherwise(pl.col("op_min"))
            .alias("old_peak"),
        )
        # op_max / op_min were only needed to DERIVE op_sign / old_peak
        # above; nothing downstream reads them, so drop them here.
        .select(["node", "op_sign", "old_peak"])
    )

    # ── Stage 8: peak-domain family ────────────────────────────────────
    # peak_domain = scale_to_annual_and_peak_flow AND af!=0 AND peak!=0.
    peak_lk_keys = [(k[0], k[2]) for k in pdNode if k[1] == "peak_inflow"]
    peak_lk = pl.DataFrame(
        {"node": [k[0] for k in peak_lk_keys],
         "period": [k[1] for k in peak_lk_keys],
         "v_peak": [v for k, v in pdNode.items() if k[1] == "peak_inflow"]},
        schema={"node": pl.Utf8, "period": pl.Utf8, "v_peak": pl.Float64},
    )
    peak_domain_df = (
        peak_eo
        .join(period_po, how="cross")
        .join(af_lk, on=["node", "period"], how="left")
        .filter(pl.col("v_af").fill_null(0.0) != 0.0)
        .join(peak_lk, on=["node", "period"], how="left")
        .filter(pl.col("v_peak").fill_null(0.0) != 0.0)
        .join(precomp, on="node", how="left")
        .sort(["__eo", "__po"])
    )
    # The per-cell peak ingredients new_peak_sign / old_peak_max /
    # old_peak_min / old_peak_sign and new_peak_divided_by_old_peak (npop)
    # are NOT emitted (no external consumer).  op_sign / old_peak are
    # carried on ``peak_domain_df`` from stage-7 ``precomp``;
    # npop is computed INLINE in the fused912 graph below (as ``v_npop``,
    # fill-0 on old_peak==0) feeding slope/section.  No separate scratch
    # frame is built.
    # ── Stages 9-12 (Tier-B): ONE fused closed-form graph ──────────────
    # The whole sub-pipeline runs over the FULL peak-domain keyset
    # (peak_domain_df == new_peak_sign keyset).  npop and npopis are
    # computed INLINE as ``when(old_peak == 0.0)`` expressions — an
    # old_peak==0 row is a FILL-to-0 (npop=0, npopis=0), NOT a row drop
    # (constraint 1): the row survives into slope/section/nom exactly as
    # the legacy left-join + fill_null(0.0) did, with the smaller
    # new_peak_divided_by_old_peak emit (npop_df, old_peak!=0 only) already
    # written above untouched (constraint 2).  Collapsing ``npopis_df`` +
    # ``base912`` + the two npop/npopis re-joins into this single graph
    # removes the round-trip joins while preserving the exact float op
    # order / parenthesization (constraint 3) and the ``== 0.0`` guards
    # (constraint 4).  op_sign / old_peak come from stage-7 ``precomp``,
    # carried on peak_domain_df, unchanged (constraint 5).
    ofs_lk = pl.DataFrame(
        {"node": [k[0] for k in orig_flow_sum],
         "period": [k[1] for k in orig_flow_sum],
         "v_ofs": list(orig_flow_sum.values())},
        schema={"node": pl.Utf8, "period": pl.Utf8, "v_ofs": pl.Float64},
    )
    fused912 = (
        peak_domain_df
        # Per-cell ingredients: orig_flow_sum (.get(k, 0.0)) + cpsoy
        # (.get(d, 0.0)) — left-join + fill_null(0.0) reproduces the dict
        # defaults the legacy stages read.
        .join(ofs_lk, on=["node", "period"], how="left")
        .join(cpsoy_lk, on="period", how="left")
        .with_columns(
            pl.col("v_ofs").fill_null(0.0).alias("v_ofs"),
            pl.col("v_cpsoy").fill_null(0.0).alias("v_cpsoy"),
        )
        .with_columns(
            # npis = peak * 8760.0 (constraint 3).
            (pl.col("v_peak") * 8760.0).alias("v_npis"),
            # npop = 0.0 if old_peak == 0.0 else peak / old_peak.  FILL,
            # not drop (constraint 1); exact ``== 0.0`` guard (constraint 4).
            pl.when(pl.col("old_peak") == 0.0)
            .then(pl.lit(0.0))
            .otherwise(pl.col("v_peak") / pl.col("old_peak"))
            .alias("v_npop"),
        )
        .with_columns(
            # npopis = 0.0 if (cps == 0.0 or old_peak == 0.0) else
            #          (npop * ofs) / cps.  Left-to-right multiply-then-
            #          divide (constraint 3).  old_peak==0 ⇒ npop==0 above,
            #          but the legacy code never reached the cps branch on
            #          old_peak==0 (the row was skipped), defaulting npopis
            #          to 0 — so guard old_peak==0 here too (constraint 1).
            pl.when((pl.col("v_cpsoy") != 0.0)
                    & (pl.col("old_peak") != 0.0))
            .then(pl.col("v_npop") * pl.col("v_ofs") / pl.col("v_cpsoy"))
            .otherwise(pl.lit(0.0))
            .alias("v_npopis"),
        )
        .with_columns(
            # denom = npis - npopis; exact ``== 0.0`` guard (constraint 4).
            (pl.col("v_npis") - pl.col("v_npopis")).alias("v_denom"))
        .with_columns(
            # nom = 0.0 if denom == 0.0 else
            #       op_sign * (op_sign * npopis - af) / denom.  Inner parens
            #       + multiply-before-divide preserved (constraint 3).
            pl.when(pl.col("v_denom") == 0.0)
            .then(pl.lit(0.0))
            .otherwise(
                pl.col("op_sign")
                * (pl.col("op_sign") * pl.col("v_npopis") - pl.col("v_af"))
                / pl.col("v_denom"))
            .alias("v_nom"),
        )
        .with_columns(
            # slope = npop * (1.0 + nom); section = peak * nom (constraint 3).
            (pl.col("v_npop") * (1.0 + pl.col("v_nom"))).alias("v_slope"),
            (pl.col("v_peak") * pl.col("v_nom")).alias("v_section"),
        )
    )

    # Stage 9 — new_peak_inflow_sum (npis = peak * 8760) and Stage 10 —
    # new_old_multiplier (nom) are NOT emitted (no external consumer); both
    # remain inline on ``fused912`` (v_npis / v_nom) feeding slope/section.
    # Stage 11 — new_old_slope = npop * (1 + nom).
    out["new_old_slope.csv"] = _ordered_value_frame(
        fused912.with_columns(pl.col("v_slope").alias("value_f")),
        ("node", "period", "value"), ["__eo", "__po"],
    )
    # Stage 12 — new_old_section = peak * nom.
    out["new_old_section.csv"] = _ordered_value_frame(
        fused912.with_columns(pl.col("v_section").alias("value_f")),
        ("node", "period", "value"), ["__eo", "__po"],
    )

    return out


def emit_node_inflow_scaling_params(
    input_dir: Path, solve_data_dir: Path,
    *, provider,
) -> None:
    """Emit ``node_inflow_scaling_params`` to the Provider.

    Emits the 6 consumed frames under ``solve_data/<basename>`` keys via
    :func:`_emit` (dual-key registration).  Uses the vectorized compute
    (:func:`_compute_inflow_scaling_frames_vectorized`); the legacy
    :func:`_compute_inflow_scaling_frames` is retained as the parity
    oracle (``tests/engine_polars/test_vectorize_inflow_scaling_parity.py``).
    """
    frames = _compute_inflow_scaling_frames_vectorized(
        input_dir, solve_data_dir, provider=provider)
    for basename, df in frames.items():
        _emit(provider, f"solve_data/{basename}", df)
