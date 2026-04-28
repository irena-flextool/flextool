"""DC power flow angle bounds — conditional float params per DC node.

Migrated from flextool.mod:2262-2263:

    param p_angle_lower{n in node_dc_power_flow} :=
        if n in node_reference_angle then 0 else -3.14159265;
    param p_angle_upper{n in node_dc_power_flow} :=
        if n in node_reference_angle then 0 else 3.14159265;

The literal ``3.14159265`` is what mod uses (an 8-digit truncation of
π). We preserve that exact representation so MathProg parses the same
float bit-pattern as the original derivation produced — bit-exact MPS
parity requires this.
"""
from __future__ import annotations

import csv
from pathlib import Path


# Literal from flextool.mod:2262 — 8 digits of π. Do NOT replace with
# math.pi; that's a different float value and would break MPS parity.
_PI_LITERAL = "3.14159265"


def _read_singles(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        return [r[0] for r in reader if r and r[0]]


def write_dc_angle_bounds(input_dir: Path, solve_data_dir: Path) -> None:
    """Write per-node angle lower/upper bounds for nodes participating
    in DC power flow.
    """
    dc_nodes = _read_singles(input_dir / "node_dc_power_flow.csv")
    ref_nodes = frozenset(_read_singles(input_dir / "node_reference_angle.csv"))

    lower_lines: list[str] = ["node,value"]
    upper_lines: list[str] = ["node,value"]
    for n in dc_nodes:
        if n in ref_nodes:
            lower_lines.append(f"{n},0")
            upper_lines.append(f"{n},0")
        else:
            lower_lines.append(f"{n},-{_PI_LITERAL}")
            upper_lines.append(f"{n},{_PI_LITERAL}")
    (solve_data_dir / "p_angle_lower.csv").write_text("\n".join(lower_lines) + "\n")
    (solve_data_dir / "p_angle_upper.csv").write_text("\n".join(upper_lines) + "\n")
