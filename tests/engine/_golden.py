"""Γ.4 — Golden-objective oracle helpers.

The flextool-parity test pattern was::

    sol = pb.solve()
    flextool_obj = pl.read_parquet(work / "output_raw" / "v_obj_*.parquet")["objective"][0]
    assert abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj)) < 1e-6

After Γ.4 the reference comes from a per-fixture
``golden_obj.json`` pinned in the repository::

    {
        "obj": <captured_value>,
        "rel_tolerance": 1e-6,
        "captured_at": "<YYYY-MM-DD>",
        "captured_via": "db_direct" | "csv"
    }

This decouples polar_high's parity oracle from flextool's
``output_raw/`` parquets — even if flextool drifts, polar_high's parity
tests stay green.

Migration plan: tests use :func:`solve_and_check` which prefers
``golden_obj.json`` if present and falls back to ``output_raw/``
otherwise.  As fixtures gain a golden file, the dependency on
flextool's recorded outputs fades.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl


GOLDEN_FILENAME = "golden_obj.json"
DEFAULT_REL_TOL = 1e-6


def golden_path(fixture_dir: Path) -> Path:
    return Path(fixture_dir) / GOLDEN_FILENAME


def has_golden(fixture_dir: Path) -> bool:
    return golden_path(Path(fixture_dir)).is_file()


def read_golden(fixture_dir: Path) -> dict[str, Any]:
    """Read ``golden_obj.json`` from a fixture dir.  Raises FileNotFoundError
    if missing — callers should gate via :func:`has_golden`.
    """
    path = golden_path(Path(fixture_dir))
    with path.open() as fh:
        return json.load(fh)


def write_golden(fixture_dir: Path, obj: float,
                  *, captured_via: str = "db_direct",
                  rel_tolerance: float = DEFAULT_REL_TOL,
                  captured_at: str | None = None) -> Path:
    """Write a ``golden_obj.json`` next to the fixture's other artefacts."""
    if captured_at is None:
        from datetime import date
        captured_at = date.today().isoformat()
    payload = {
        "obj": float(obj),
        "rel_tolerance": float(rel_tolerance),
        "captured_at": captured_at,
        "captured_via": captured_via,
    }
    path = golden_path(Path(fixture_dir))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def lookup_v_obj_parquet(fixture_dir: Path,
                           preferred: str | None = None) -> Path | None:
    """Find the flextool-recorded ``v_obj__*.parquet`` for a fixture.

    Falls back to glob if no exact filename is preferred.  Returns
    ``None`` when no parquet exists.
    """
    out = Path(fixture_dir) / "output_raw"
    if not out.is_dir():
        return None
    if preferred:
        p = out / preferred
        if p.exists():
            return p
    cands = sorted(out.glob("v_obj__*.parquet"))
    return cands[0] if cands else None


def expected_obj(fixture_dir: Path,
                  *, parquet_glob: str | None = None,
                  ) -> tuple[float, float, str]:
    """Resolve the (reference_obj, rel_tolerance, source) triple.

    Lookup order:

    1. If ``golden_obj.json`` is present, return that.
    2. Otherwise find a ``v_obj__*.parquet`` in ``output_raw/`` and
       return ``(parquet_obj, DEFAULT_REL_TOL, "flextool_parquet")``.

    Raises ``FileNotFoundError`` if neither exists.
    """
    fixture_dir = Path(fixture_dir)
    if has_golden(fixture_dir):
        g = read_golden(fixture_dir)
        return float(g["obj"]), float(g.get("rel_tolerance", DEFAULT_REL_TOL)), \
                f"golden ({g.get('captured_via', 'unknown')})"
    parquet = lookup_v_obj_parquet(fixture_dir, parquet_glob)
    if parquet is None:
        raise FileNotFoundError(
            f"no golden_obj.json and no v_obj__*.parquet under {fixture_dir}"
        )
    obj = float(pl.read_parquet(parquet)["objective"][0])
    return obj, DEFAULT_REL_TOL, f"flextool_parquet({parquet.name})"


def assert_obj_within(actual: float, fixture_dir: Path,
                       *, parquet_glob: str | None = None) -> None:
    """Assert ``actual`` matches the expected obj for ``fixture_dir``
    within the (golden or default) relative tolerance.
    """
    expected, tol, source = expected_obj(fixture_dir, parquet_glob=parquet_glob)
    rel = abs(actual - expected) / max(1.0, abs(expected))
    assert rel < tol, (
        f"objective parity failed: actual={actual}, expected={expected}, "
        f"rel={rel}, tol={tol}, source={source}"
    )
