"""MPS structural-parity harness.

Two MPS files are considered "structurally equivalent" iff:
- Same set of rows (type + name), regardless of file order
- Same set of columns, each with the same (row, coefficient) bag
- Same RHS values per row
- Same RANGES values per row
- Same bounds (type + value) per column

Sort order in the file does NOT matter (MathProg's iteration order over
sets is allowed to change as long as the *contents* of the matrix do
not). Coefficient values DO matter — bit-exact float equality.

This is the validation gate for the python-preprocessing migration:
each step must produce an MPS that is structurally equivalent to the
baseline, or the step is rejected.

CLI:
    python -m migration.mps_parity baseline <mps_file>      # store canonical hash + JSON
    python -m migration.mps_parity check    <mps_file>      # compare against stored baseline
    python -m migration.mps_parity diff <mps_a> <mps_b>     # show structured diff

Library:
    canon = parse_mps(path)
    h = canonical_hash(canon)
    diff = diff_canonical(canon_a, canon_b)   # None == identical
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from dataclasses import dataclass, asdict
from pathlib import Path


_OBJSENSE_TOKENS = ("MAX", "MAXIMIZE", "MIN", "MINIMIZE")


@dataclass(frozen=True)
class CanonicalMPS:
    """Order-independent canonical representation of an MPS file.

    Tuples (not lists) so the dataclass is hashable.
    Floats stored as 8-byte little-endian hex strings for bit-exact comparison.
    """
    name: str
    objsense: str
    rows: tuple[tuple[str, str], ...]                    # (type, name)
    columns: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
                                                          # (col, ((row, coef_hex), ...))
    rhs: tuple[tuple[str, str], ...]                     # (row, rhs_hex)
    ranges: tuple[tuple[str, str], ...]                  # (row, range_hex)
    bounds: tuple[tuple[str, str, str], ...]             # (col, type, value_hex)


def _f64_hex(value: float) -> str:
    """Bit-exact 16-char hex of a float64."""
    return struct.pack("<d", float(value)).hex()


def parse_mps(path: str | Path) -> CanonicalMPS:
    """Parse a Free MPS file into a canonical, order-independent form."""
    path = Path(path)
    name = ""
    objsense = "MIN"  # MPS default
    rows: list[tuple[str, str]] = []
    columns: dict[str, list[tuple[str, str]]] = {}
    rhs: dict[str, str] = {}
    ranges: dict[str, str] = {}
    # MPS bounds are per (col, type) pairs but a single col may have multiple
    # bound rows (e.g. UP + LO). Key by (col, type) to avoid collisions.
    bounds: list[tuple[str, str, str]] = []

    section: str | None = None
    in_marker_block = False  # MARKER lines inside COLUMNS toggle integer flag

    with path.open() as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line or line.startswith("*"):
                continue
            stripped = line.strip()
            head = stripped.split()[0] if stripped else ""

            # Section headers begin in column 1 (no leading whitespace).
            if not line[0].isspace():
                if head == "NAME":
                    parts = stripped.split(None, 1)
                    name = parts[1] if len(parts) > 1 else ""
                    continue
                if head in ("ROWS", "COLUMNS", "RHS", "RANGES", "BOUNDS", "ENDATA",
                            "SOS", "OBJSENSE"):
                    section = head
                    if head == "ENDATA":
                        break
                    continue

            tokens = stripped.split()
            if not tokens:
                continue

            if section == "OBJSENSE":
                t = tokens[0].upper()
                if t in _OBJSENSE_TOKENS:
                    objsense = "MAX" if t.startswith("MAX") else "MIN"
                continue

            if section == "ROWS":
                # <type> <name>
                rtype, rname = tokens[0], tokens[1]
                rows.append((rtype, rname))
                continue

            if section == "COLUMNS":
                # MARKER lines toggle integer-block context — we ignore the
                # context for canonical comparison (var integrality is a model
                # property; for the migration we expect identical .mod, so
                # MARKER blocks should match anyway and need no special
                # handling beyond passing through).
                if "'MARKER'" in tokens:
                    in_marker_block = "'INTORG'" in tokens
                    continue
                # Either: col row coef [row coef]
                # Free MPS may pack 2 (row, coef) pairs per line.
                col = tokens[0]
                pairs = tokens[1:]
                if len(pairs) % 2 != 0:
                    raise ValueError(f"Malformed COLUMNS line in {path}: {line!r}")
                lst = columns.setdefault(col, [])
                for i in range(0, len(pairs), 2):
                    row = pairs[i]
                    coef = float(pairs[i + 1])
                    lst.append((row, _f64_hex(coef)))
                continue

            if section == "RHS":
                # <rhs_name> <row> <value> [<row> <value>]
                pairs = tokens[1:]
                if len(pairs) % 2 != 0:
                    raise ValueError(f"Malformed RHS line in {path}: {line!r}")
                for i in range(0, len(pairs), 2):
                    row = pairs[i]
                    val = float(pairs[i + 1])
                    if row in rhs:
                        raise ValueError(f"Duplicate RHS for {row} in {path}")
                    rhs[row] = _f64_hex(val)
                continue

            if section == "RANGES":
                pairs = tokens[1:]
                if len(pairs) % 2 != 0:
                    raise ValueError(f"Malformed RANGES line in {path}: {line!r}")
                for i in range(0, len(pairs), 2):
                    row = pairs[i]
                    val = float(pairs[i + 1])
                    if row in ranges:
                        raise ValueError(f"Duplicate RANGE for {row} in {path}")
                    ranges[row] = _f64_hex(val)
                continue

            if section == "BOUNDS":
                # <type> <bnd_name> <col> [<value>]
                # Types: UP, LO, FX, FR, MI, PL, BV, LI, UI
                btype = tokens[0]
                col = tokens[2]
                if btype in ("FR", "MI", "PL", "BV"):
                    # No numeric value
                    bounds.append((col, btype, _f64_hex(0.0)))
                else:
                    val = float(tokens[3])
                    bounds.append((col, btype, _f64_hex(val)))
                continue

    # Canonicalize: sort everything for order-independence
    rows_sorted = tuple(sorted(rows))
    columns_sorted = tuple(
        (col, tuple(sorted(pairs))) for col, pairs in sorted(columns.items())
    )
    rhs_sorted = tuple(sorted(rhs.items()))
    ranges_sorted = tuple(sorted(ranges.items()))
    bounds_sorted = tuple(sorted(bounds))

    return CanonicalMPS(
        name=name,
        objsense=objsense,
        rows=rows_sorted,
        columns=columns_sorted,
        rhs=rhs_sorted,
        ranges=ranges_sorted,
        bounds=bounds_sorted,
    )


def canonical_hash(canon: CanonicalMPS) -> str:
    """SHA-256 over the canonical form. Stable across file orderings."""
    h = hashlib.sha256()
    h.update(canon.name.encode())
    h.update(b"|")
    h.update(canon.objsense.encode())
    h.update(b"|")
    for rtype, rname in canon.rows:
        h.update(f"{rtype} {rname}\n".encode())
    h.update(b"||")
    for col, pairs in canon.columns:
        h.update(f"{col}".encode())
        for row, coef in pairs:
            h.update(f" {row} {coef}".encode())
        h.update(b"\n")
    h.update(b"||")
    for row, val in canon.rhs:
        h.update(f"{row} {val}\n".encode())
    h.update(b"||")
    for row, val in canon.ranges:
        h.update(f"{row} {val}\n".encode())
    h.update(b"||")
    for col, btype, val in canon.bounds:
        h.update(f"{col} {btype} {val}\n".encode())
    return h.hexdigest()


def diff_canonical(a: CanonicalMPS, b: CanonicalMPS) -> str | None:
    """Return a human-readable diff, or None if structurally identical."""
    if a == b:
        return None
    out: list[str] = []
    if a.name != b.name:
        out.append(f"NAME: {a.name!r} vs {b.name!r}")
    if a.objsense != b.objsense:
        out.append(f"OBJSENSE: {a.objsense} vs {b.objsense}")

    a_rows, b_rows = set(a.rows), set(b.rows)
    if a_rows != b_rows:
        only_a = sorted(a_rows - b_rows)
        only_b = sorted(b_rows - a_rows)
        if only_a:
            out.append(f"ROWS only in A ({len(only_a)}): {only_a[:5]}...")
        if only_b:
            out.append(f"ROWS only in B ({len(only_b)}): {only_b[:5]}...")

    a_cols = {c: dict(p) for c, p in a.columns}
    b_cols = {c: dict(p) for c, p in b.columns}
    only_a_cols = sorted(set(a_cols) - set(b_cols))
    only_b_cols = sorted(set(b_cols) - set(a_cols))
    if only_a_cols:
        out.append(f"COLUMNS only in A ({len(only_a_cols)}): {only_a_cols[:5]}...")
    if only_b_cols:
        out.append(f"COLUMNS only in B ({len(only_b_cols)}): {only_b_cols[:5]}...")
    coef_diffs = 0
    for col in sorted(set(a_cols) & set(b_cols)):
        ap, bp = a_cols[col], b_cols[col]
        for row in sorted(set(ap) | set(bp)):
            if ap.get(row) != bp.get(row):
                coef_diffs += 1
                if coef_diffs <= 5:
                    out.append(f"COEF[{col}, {row}]: {ap.get(row)} vs {bp.get(row)}")
    if coef_diffs > 5:
        out.append(f"... and {coef_diffs - 5} more coefficient differences")

    a_rhs, b_rhs = dict(a.rhs), dict(b.rhs)
    rhs_diffs = [r for r in sorted(set(a_rhs) | set(b_rhs)) if a_rhs.get(r) != b_rhs.get(r)]
    if rhs_diffs:
        out.append(f"RHS differs on {len(rhs_diffs)} rows: {rhs_diffs[:5]}...")

    a_rng, b_rng = dict(a.ranges), dict(b.ranges)
    rng_diffs = [r for r in sorted(set(a_rng) | set(b_rng)) if a_rng.get(r) != b_rng.get(r)]
    if rng_diffs:
        out.append(f"RANGES differs on {len(rng_diffs)} rows: {rng_diffs[:5]}...")

    a_bnd, b_bnd = set(a.bounds), set(b.bounds)
    if a_bnd != b_bnd:
        only_a_b = sorted(a_bnd - b_bnd)[:5]
        only_b_b = sorted(b_bnd - a_bnd)[:5]
        out.append(f"BOUNDS differ: {len(a_bnd - b_bnd)} only-A, {len(b_bnd - a_bnd)} only-B")
        if only_a_b:
            out.append(f"  only-A sample: {only_a_b}")
        if only_b_b:
            out.append(f"  only-B sample: {only_b_b}")

    return "\n".join(out) if out else "(differences in dataclass fields not surfaced — investigate manually)"


def write_canonical_summary(canon: CanonicalMPS, out_path: str | Path) -> None:
    """Write a compact JSON summary (NOT the full canonical form — too big)."""
    summary = {
        "name": canon.name,
        "objsense": canon.objsense,
        "n_rows": len(canon.rows),
        "n_cols": len(canon.columns),
        "n_coefficients": sum(len(p) for _, p in canon.columns),
        "n_rhs": len(canon.rhs),
        "n_ranges": len(canon.ranges),
        "n_bounds": len(canon.bounds),
        "hash": canonical_hash(canon),
    }
    Path(out_path).write_text(json.dumps(summary, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MPS structural-parity tool")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_base = sub.add_parser("baseline", help="Store canonical hash + summary")
    p_base.add_argument("mps", type=Path)
    p_base.add_argument("--out", type=Path,
                        default=Path("migration/baselines/mps_baseline.json"))

    p_check = sub.add_parser("check", help="Compare against stored baseline")
    p_check.add_argument("mps", type=Path)
    p_check.add_argument("--baseline", type=Path,
                         default=Path("migration/baselines/mps_baseline.json"))

    p_diff = sub.add_parser("diff", help="Diff two MPS files structurally")
    p_diff.add_argument("a", type=Path)
    p_diff.add_argument("b", type=Path)

    args = parser.parse_args(argv)

    if args.cmd == "baseline":
        canon = parse_mps(args.mps)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_canonical_summary(canon, args.out)
        print(f"Baseline stored: {args.out}")
        print(f"  hash={canonical_hash(canon)}")
        print(f"  rows={len(canon.rows)} cols={len(canon.columns)} "
              f"coefs={sum(len(p) for _, p in canon.columns)}")
        return 0

    if args.cmd == "check":
        if not args.baseline.exists():
            print(f"FAIL: baseline file missing: {args.baseline}", file=sys.stderr)
            return 2
        baseline = json.loads(args.baseline.read_text())
        canon = parse_mps(args.mps)
        h = canonical_hash(canon)
        if h == baseline["hash"]:
            print(f"OK: MPS structurally identical to baseline (hash={h[:12]}...)")
            return 0
        print(f"FAIL: MPS differs from baseline", file=sys.stderr)
        print(f"  baseline hash: {baseline['hash']}", file=sys.stderr)
        print(f"  current hash:  {h}", file=sys.stderr)
        print(f"  baseline shape: rows={baseline['n_rows']} cols={baseline['n_cols']} "
              f"coefs={baseline['n_coefficients']}", file=sys.stderr)
        print(f"  current shape:  rows={len(canon.rows)} cols={len(canon.columns)} "
              f"coefs={sum(len(p) for _, p in canon.columns)}", file=sys.stderr)
        return 1

    if args.cmd == "diff":
        a = parse_mps(args.a)
        b = parse_mps(args.b)
        d = diff_canonical(a, b)
        if d is None:
            print("MPS files are structurally identical")
            return 0
        print(d)
        return 1

    return 2


if __name__ == "__main__":
    sys.exit(main())
