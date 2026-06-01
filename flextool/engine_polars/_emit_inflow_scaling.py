"""Node inflow scaling params — per-solve emitters.

Called per-solve from ``_emit_solve_time.run`` (batch 17).

Output CSVs (17 total):

* ``ptNode_inflow.csv``                            — (n, t) merged inflow
* ``_node_cap_inflow_fallback.csv``                — (n, d) abs(max)
* ``orig_flow_sum.csv``                            — (n, d) sum over complete time
* ``period_share_of_annual_flow.csv``              — (n, d) abs(sum/dt) / annual_flow
* ``period_flow_annual_multiplier.csv``            — (n, d) cpsoy / psaf
* ``period_flow_proportional_multiplier.csv``      — (n, d) af / (abs(sum_t)/tdy)
* ``new_peak_sign.csv``                            — (n, d) sign(peak_inflow)
* ``old_peak_max.csv`` / ``old_peak_min.csv``      — (n, d) inflow series bounds
* ``old_peak_sign.csv``                            — (n, d) sign per dominant peak
* ``old_peak.csv``                                 — (n, d) signed dominant peak
* ``new_peak_divided_by_old_peak.csv``             — (n, d) peak / old_peak
* ``new_peak_divide_by_old_peak_sum_inflow.csv``   — (n, d) npop * ofs / cpsoy
* ``new_peak_inflow_sum.csv``                      — (n, d) peak * 8760
* ``new_old_multiplier.csv``                       — (n, d) affine coefficient
* ``new_old_slope.csv``                            — (n, d) npop * (1 + nom)
* ``new_old_section.csv``                          — (n, d) peak * nom

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

    Used by both the wrapper (each frame is fed to ``_write``) and the
    standalone ``derive_*`` functions (which index this dict).  The
    cross-CSV state (peak family, npop / nom / etc) is heavy enough that
    splitting into independent derive_* calls would re-walk the time
    axis O(N) times — the dict-of-frames pattern from the audit doc is
    the appropriate adapter here.
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
    rows_op: list[tuple[str, str, float]] = []
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
            rows_op.append((n, d, old_peak_val))
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
    out["old_peak.csv"] = _rows_to_frame(
        ("node", "period", "value"), rows_op,
    )
    out["new_peak_divided_by_old_peak.csv"] = _rows_to_frame(
        ("node", "period", "value"), rows_npop,
    )
    out["new_peak_divide_by_old_peak_sum_inflow.csv"] = _rows_to_frame(
        ("node", "period", "value"), rows_npopinflow,
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


# ---- Phase E-b — derive_X family for each emitted CSV --------------------
#
# Each derive_* delegates to the shared :func:`_compute_inflow_scaling_frames`
# pass and indexes the resulting dict.  The shared compute is the path of
# least re-walking — splitting into 17 standalone derive_* would re-scan
# the t-axis O(N) times per call.  Public derive_* are thin lookups for
# Phase D / E-a seed consumers.


def _derive(input_dir: Path, solve_data_dir: Path,
            basename: str) -> pl.DataFrame:
    return _compute_inflow_scaling_frames(input_dir, solve_data_dir)[basename]


def derive_ptNode_inflow(input_dir: Path, solve_data_dir: Path) -> pl.DataFrame:
    """``ptNode_inflow.csv`` — (n, t) merged inflow with scalar fallback."""
    return _derive(input_dir, solve_data_dir, "ptNode_inflow.csv")


def derive_node_cap_inflow_fallback(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``_node_cap_inflow_fallback.csv`` — per-(n, d) max abs inflow."""
    return _derive(input_dir, solve_data_dir, "_node_cap_inflow_fallback.csv")


def derive_orig_flow_sum(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``orig_flow_sum.csv`` — Σ ptNode_inflow over complete_time_in_use."""
    return _derive(input_dir, solve_data_dir, "orig_flow_sum.csv")


def derive_period_share_of_annual_flow(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``period_share_of_annual_flow.csv`` — abs(Σ_t inflow) / annual_flow."""
    return _derive(input_dir, solve_data_dir, "period_share_of_annual_flow.csv")


def derive_period_flow_annual_multiplier(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``period_flow_annual_multiplier.csv`` — cpsoy / psaf."""
    return _derive(input_dir, solve_data_dir, "period_flow_annual_multiplier.csv")


def derive_period_flow_proportional_multiplier(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``period_flow_proportional_multiplier.csv`` — annual_flow / (abs(Σ_t)/tdy)."""
    return _derive(
        input_dir, solve_data_dir, "period_flow_proportional_multiplier.csv",
    )


def derive_new_peak_sign(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``new_peak_sign.csv`` — sign(peak_inflow)."""
    return _derive(input_dir, solve_data_dir, "new_peak_sign.csv")


def derive_old_peak_max(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``old_peak_max.csv`` — max_t inflow per (n, d)."""
    return _derive(input_dir, solve_data_dir, "old_peak_max.csv")


def derive_old_peak_min(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``old_peak_min.csv`` — min_t inflow per (n, d)."""
    return _derive(input_dir, solve_data_dir, "old_peak_min.csv")


def derive_old_peak_sign(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``old_peak_sign.csv`` — +1 if max-abs is positive, else -1."""
    return _derive(input_dir, solve_data_dir, "old_peak_sign.csv")


def derive_old_peak(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``old_peak.csv`` — signed dominant peak."""
    return _derive(input_dir, solve_data_dir, "old_peak.csv")


def derive_new_peak_divided_by_old_peak(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``new_peak_divided_by_old_peak.csv`` — peak / old_peak."""
    return _derive(input_dir, solve_data_dir, "new_peak_divided_by_old_peak.csv")


def derive_new_peak_divide_by_old_peak_sum_inflow(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``new_peak_divide_by_old_peak_sum_inflow.csv`` — npop * ofs / cpsoy."""
    return _derive(
        input_dir, solve_data_dir,
        "new_peak_divide_by_old_peak_sum_inflow.csv",
    )


def derive_new_peak_inflow_sum(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``new_peak_inflow_sum.csv`` — peak_inflow * 8760."""
    return _derive(input_dir, solve_data_dir, "new_peak_inflow_sum.csv")


def derive_new_old_multiplier(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``new_old_multiplier.csv`` — affine coefficient (npis - npopis basis)."""
    return _derive(input_dir, solve_data_dir, "new_old_multiplier.csv")


def derive_new_old_slope(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``new_old_slope.csv`` — npop * (1 + nom)."""
    return _derive(input_dir, solve_data_dir, "new_old_slope.csv")


def derive_new_old_section(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``new_old_section.csv`` — peak * nom."""
    return _derive(input_dir, solve_data_dir, "new_old_section.csv")


def emit_node_inflow_scaling_params(
    input_dir: Path, solve_data_dir: Path,
    *, provider,
) -> None:
    """Emit ``node_inflow_scaling_params`` to the Provider.
    Emits the same 17 frames under ``solve_data/<basename>`` keys via
    :func:`_emit` (dual-key registration).
    """
    frames = _compute_inflow_scaling_frames(input_dir, solve_data_dir,
                                              provider=provider)
    for basename, df in frames.items():
        _emit(provider, f"solve_data/{basename}", df)
