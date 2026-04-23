"""Precision cleanup and near-duplicate detection for input CSV parameters.

This module implements the Agent 7 precision cleanup pass for the
LP-scaling project (see
``~/.claude/projects/-home-jkiviluo-sources-flextool/memory/project_lp_scaling_2026-04.md``).

Two user-facing entry points:

* :func:`round_to_sigfigs` — round a float to N significant figures.
  Used by :mod:`flextool.flextoolrunner.input_writer` to collapse
  accumulated floating-point noise (e.g. ``1/3`` stored as
  ``0.3333333333333333``) so HiGHS' ``mip_detect_symmetry`` presolver
  collapses structurally-identical coefficients.

* :func:`find_near_duplicates` / :func:`report_near_duplicates` — a
  diagnostic that clusters nearly-equal values within each parameter
  CSV.  Purpose: point users at cases where symmetry detection could
  benefit if the values were made *exactly* equal.  Silent by default;
  only runs when ``--report-near-duplicates`` (or the corresponding
  env var) is set.

Design constraints:

* Stdlib only (``math``, ``os``, ``csv``, ``pathlib``).  No new deps.
* Default behaviour must be unchanged — a run with no CLI flag, no
  env var, and ``precision_digits=0`` must round-trip every CSV
  byte-for-byte with the pre-Agent-7 state.
"""

from __future__ import annotations

import csv
import math
import os
import warnings
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Precision env-var resolution
# ---------------------------------------------------------------------------

PRECISION_ENV_VAR = "FLEXTOOL_PRECISION_DIGITS"
"""Environment variable fallback for the ``--precision-digits`` CLI flag."""

REPORT_ENV_VAR = "FLEXTOOL_REPORT_NEAR_DUPS"
"""Environment variable fallback for ``--report-near-duplicates``."""

DEFAULT_PRECISION_DIGITS = 10
"""Default significant-figure count when the user enables rounding by
setting the CLI flag (or env var) to a non-zero value but gives no explicit
digit count.  The CLI passes ``10`` explicitly, so this is only the
fallback used by library callers / tests."""

_WARNED_HIGH_PRECISION = False


def resolve_precision_digits(cli_value: Optional[int]) -> int:
    """Resolve the effective precision setting.

    Precedence: explicit CLI value (``--precision-digits N`` or None when
    the flag is not passed) overrides the env var.  ``0`` or a negative
    value disables rounding (passthrough).  Values > 15 exceed
    ``float64`` precision — emits a one-shot warning and returns 0 so
    rounding is a no-op.

    Returns the effective digit count, or 0 when rounding is disabled.
    """
    global _WARNED_HIGH_PRECISION
    if cli_value is None:
        raw = os.environ.get(PRECISION_ENV_VAR, "").strip()
        if not raw:
            return 0
        try:
            value = int(raw)
        except ValueError:
            warnings.warn(
                f"{PRECISION_ENV_VAR}={raw!r} is not an integer; "
                f"ignoring and disabling precision rounding.",
                RuntimeWarning,
                stacklevel=2,
            )
            return 0
    else:
        value = int(cli_value)
    if value <= 0:
        return 0
    if value > 15:
        if not _WARNED_HIGH_PRECISION:
            print(
                f"WARNING: precision_digits={value} is higher than "
                f"float64 precision, no-op (passthrough)."
            )
            _WARNED_HIGH_PRECISION = True
        return 0
    return value


def resolve_report_near_duplicates(cli_flag: bool) -> bool:
    """True if the CLI flag is set OR the env var resolves truthy."""
    if cli_flag:
        return True
    raw = os.environ.get(REPORT_ENV_VAR, "").strip().lower()
    return raw in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# round_to_sigfigs
# ---------------------------------------------------------------------------


def round_to_sigfigs(value: float, digits: int) -> float:
    """Round *value* to *digits* significant figures.

    Edge cases:

    * ``0.0`` stays ``0.0`` (sign preserved).
    * ``nan`` / ``inf`` / ``-inf`` pass through unchanged.
    * Very small values (``abs(value) < 1e-300``) round to ``0.0`` —
      they would overflow the exponent arithmetic.
    * ``digits <= 0`` returns *value* unchanged (pass-through sentinel).
    * Negative values round symmetrically around 0.

    The classical formula is
    ``scale = 10 ** (floor(log10(|value|)) + 1 - digits)``;
    rounding via Python's builtin ``round(value, ndigits)`` avoids the
    textbook ``round(v/scale)*scale`` double-rounding artefact (e.g.
    ``0.3333... @ 4 sig figs`` would otherwise come back as
    ``0.33330000000000004`` because ``10**-4`` is not exactly
    representable).  ``ndigits`` is negative when rounding above the
    decimal point, which ``round`` handles natively.
    """
    if digits <= 0:
        return value
    if value == 0.0 or not math.isfinite(value):
        return value
    abs_value = abs(value)
    if abs_value < 1e-300:
        # Sub-denormal — would blow up the exponent below.
        return 0.0
    # Position of the leading significant digit (base 10).
    exponent = math.floor(math.log10(abs_value))
    # Python's builtin ``round(x, ndigits)`` uses decimal-aware rounding
    # that avoids the double-rounding artefact of the textbook
    # ``round(x / scale) * scale`` (where 10**-4 cannot be represented
    # exactly).  We compute ``ndigits`` so that after rounding there are
    # ``digits`` significant figures.  For 0.3333... with digits=4, that
    # is ``ndigits = digits - exponent - 1 = 4 - (-1) - 1 = 4`` → 0.3333.
    ndigits = digits - exponent - 1
    # ``round`` accepts negative ndigits for rounding to tens, hundreds, …
    return round(value, ndigits)


# ---------------------------------------------------------------------------
# CSV-writer hook: format a scalar for the input/ CSV stream
# ---------------------------------------------------------------------------


def format_scalar_for_csv(value, precision_digits: int) -> str:
    """Return the string form of *value* that should land in the CSV.

    * When *value* is a float / int (or a string that parses as one)
      and ``precision_digits > 0``, the numeric is rounded to that many
      significant figures first.
    * Integer values pass through untouched (``.is_integer()`` heuristic).
    * Non-numeric strings (entity names, method names, ``"yes"``/``"no"``)
      pass through ``str()`` unchanged.

    Default behaviour (``precision_digits == 0``) is identical to
    ``str(value)`` — the pre-Agent-7 writer convention — so benchmarks
    round-trip byte-for-byte.
    """
    # Fast path: rounding disabled → exactly the old behaviour.
    if precision_digits <= 0:
        return str(value)

    # True bools: "True"/"False" — pass through.
    if isinstance(value, bool):
        return str(value)

    # Native numeric types.
    if isinstance(value, (int, float)):
        f = float(value)
        if not math.isfinite(f):
            return str(value)
        # Integer values sidestep rounding so period indices, year
        # counts, tier numbers etc. are never perturbed.
        if isinstance(value, int) or f.is_integer():
            return str(value)
        rounded = round_to_sigfigs(f, precision_digits)
        return _format_float(rounded)

    # String that might encode a float (common path for map values
    # flattened via convert_map_to_table).
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return value
        try:
            f = float(s)
        except ValueError:
            return value  # entity name, method string, etc.
        if not math.isfinite(f):
            return value
        if f.is_integer() and s.lstrip("+-").isdigit():
            # Pure integer literal — keep original form (no ".0" drift).
            return value
        rounded = round_to_sigfigs(f, precision_digits)
        return _format_float(rounded)

    return str(value)


def _format_float(value: float) -> str:
    """Render a rounded float compactly.

    Python's ``repr(float)`` uses the shortest round-trippable form,
    which for a value that has been truncated to N sig figs produces a
    clean N-digit representation (no trailing garbage).  We prefer
    ``repr`` over ``str`` for Python 2/3 consistency — they are
    identical in Python 3.
    """
    return repr(value)


# ---------------------------------------------------------------------------
# Near-duplicate detection
# ---------------------------------------------------------------------------


def find_near_duplicates(
    values: Iterable[float],
    rel_tol: float = 1e-6,
) -> list[list[float]]:
    """Cluster *values* by within-cluster relative spread ≤ *rel_tol*.

    Algorithm:

    1. Drop NaN / non-finite entries; take absolute magnitude only for
       the clustering distance (sign kept on each value for reporting).
    2. Sort by magnitude.
    3. Sweep: extend the current cluster as long as
       ``(v - cluster_min) / max(|cluster_min|, |v|, 1e-300) <= rel_tol``.
    4. Drop singleton clusters; return clusters of size ≥ 2, sorted by
       cluster size descending, then by representative magnitude.

    The per-parameter scope means we look at e.g. all
    ``p_entity_unitsize`` values across entities in one pool; VOM
    values across units in a separate pool.  Cross-parameter matching
    (where symmetry actually bites) is left for a future, deeper tool.
    """
    if rel_tol < 0:
        raise ValueError("rel_tol must be non-negative")
    buckets: list[list[float]] = []
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return buckets
    # Sort so sign-flips never collapse into a single cluster.  Sort by
    # (sign, abs) — negatives together ascending by magnitude, then
    # positives, and a cluster boundary is forced at every sign change.
    def _sk(x: float) -> tuple[int, float]:
        return (1 if x >= 0 else -1, abs(x))
    finite.sort(key=_sk)
    current: list[float] = [finite[0]]
    cluster_min_abs = abs(finite[0])
    cluster_sign = _sk(finite[0])[0]
    for v in finite[1:]:
        va = abs(v)
        vs = _sk(v)[0]
        denom = max(cluster_min_abs, va, 1e-300)
        same_sign = vs == cluster_sign
        if same_sign and (va - cluster_min_abs) / denom <= rel_tol:
            current.append(v)
        else:
            if len(current) >= 2:
                buckets.append(current)
            current = [v]
            cluster_min_abs = va
            cluster_sign = vs
    if len(current) >= 2:
        buckets.append(current)
    buckets.sort(key=lambda b: (-len(b), abs(b[0])))
    return buckets


# ---------------------------------------------------------------------------
# Near-duplicate report
# ---------------------------------------------------------------------------


def _iter_numeric_columns(path: Path) -> Iterator[tuple[str, list[tuple[str, float]]]]:
    """Yield ``(column_name, [(entity_label, value), ...])`` per numeric
    column in *path*.

    Numeric detection: a column is numeric iff every non-empty row in
    it parses as a finite float.  Columns that look like period /
    timestep / tier indices (pure integer literals throughout) are
    skipped — symmetry is a numerical-coefficient phenomenon, and
    structural indices would produce noisy "clusters" with zero
    signal.
    """
    try:
        with path.open(newline="") as fh:
            reader = csv.reader(fh)
            try:
                header = next(reader)
            except StopIteration:
                return
            rows = list(reader)
    except OSError:
        return
    if not header or not rows:
        return

    n_cols = len(header)
    # Per-column parse attempt.
    parsed: list[list[Optional[float]]] = [[] for _ in range(n_cols)]
    for row in rows:
        for i in range(n_cols):
            cell = row[i].strip() if i < len(row) else ""
            if not cell:
                parsed[i].append(None)
                continue
            try:
                parsed[i].append(float(cell))
            except ValueError:
                parsed[i].append(None)

    for i, col_values in enumerate(parsed):
        non_null = [v for v in col_values if v is not None]
        if len(non_null) < 2:
            continue
        # Column is numeric iff *every* non-empty cell parsed.
        n_non_empty = sum(
            1 for row in rows
            if i < len(row) and row[i].strip() != ""
        )
        if len(non_null) != n_non_empty:
            continue
        # Skip pure-integer columns (period ids, tier numbers, etc.).
        if all(float(v).is_integer() for v in non_null):
            continue
        # Build entity label = non-numeric columns joined.
        labelled: list[tuple[str, float]] = []
        for r_idx, row in enumerate(rows):
            if col_values[r_idx] is None:
                continue
            label_parts = []
            for j in range(min(len(row), n_cols)):
                if j == i:
                    continue
                cell = row[j].strip()
                try:
                    float(cell)
                    continue  # skip other numeric columns from the label
                except ValueError:
                    if cell:
                        label_parts.append(cell)
            labelled.append((" | ".join(label_parts) or f"row{r_idx}", col_values[r_idx]))
        yield header[i], labelled


def report_near_duplicates(
    input_dir: Path,
    top: int = 10,
    rel_tol: float = 1e-6,
) -> None:
    """Scan every CSV in *input_dir* and print the top *top* clusters.

    Output format (one block per cluster)::

        near-dup [<size>×] <csv>:<column>  value=<repr>  spread=<Δ>  rel=<r>
                 entities: <ent1>, <ent2>, <ent3>, ...

    Never fails the run: any per-file error is caught and logged.  A
    final summary line reports the total number of clusters found.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        print(
            f"[precision] report_near_duplicates: {input_dir} is not a "
            f"directory; nothing to scan."
        )
        return

    all_clusters: list[tuple[int, str, str, list[tuple[str, float]]]] = []
    # (size, csv_file, column, [(entity, value), ...])

    for csv_path in sorted(input_dir.glob("*.csv")):
        try:
            for column, labelled_values in _iter_numeric_columns(csv_path):
                raw_values = [v for _, v in labelled_values]
                clusters = find_near_duplicates(raw_values, rel_tol=rel_tol)
                if not clusters:
                    continue
                # Back-map each cluster value to its (label, value) tuple.
                by_value: dict[float, list[str]] = {}
                for label, v in labelled_values:
                    by_value.setdefault(v, []).append(label)
                for cluster in clusters:
                    lv: list[tuple[str, float]] = []
                    used: dict[float, int] = {}
                    for v in cluster:
                        labels = by_value.get(v, [])
                        idx = used.get(v, 0)
                        label = labels[idx] if idx < len(labels) else f"value={v}"
                        used[v] = idx + 1
                        lv.append((label, v))
                    all_clusters.append(
                        (len(cluster), csv_path.name, column, lv)
                    )
        except Exception as exc:  # diagnostic only — never fail the run
            print(
                f"[precision] skipping {csv_path.name}: {exc}"
            )
            continue

    # Sort: largest cluster first, then by CSV/column for stable output.
    all_clusters.sort(key=lambda c: (-c[0], c[1], c[2]))
    if not all_clusters:
        print(
            f"[precision] no near-duplicate clusters found in {input_dir} "
            f"(rel_tol={rel_tol:g})."
        )
        return

    shown = all_clusters[:top]
    for size, csv_file, column, entries in shown:
        values = [v for _, v in entries]
        lo = min(values)
        hi = max(values)
        spread = hi - lo
        mag = max(abs(lo), abs(hi), 1e-300)
        rel = spread / mag
        rep = entries[0][1]
        sample_names = [label for label, _ in entries[:3]]
        print(
            f"near-dup [{size}x] {csv_file}:{column}  "
            f"value={rep!r}  spread={spread:g}  rel={rel:g}"
        )
        print(f"         entities: {', '.join(sample_names)}")

    total_values_in_clusters = sum(size for size, *_ in all_clusters)
    print(
        f"[precision] {len(all_clusters)} near-duplicate cluster(s) "
        f"covering {total_values_in_clusters} value(s) "
        f"(showing top {len(shown)}, rel_tol={rel_tol:g})."
    )
