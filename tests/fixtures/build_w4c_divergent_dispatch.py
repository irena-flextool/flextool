"""Build ``tests/fixtures/w4c_divergent_dispatch.json`` — the W4c
"divergent-forecast realized dispatch" fixture.

W4b (``build_w4b_auto_dispatch.py``) kept the realized operational data
DETERMINISTIC (a single 2d ``0.40`` wind profile shared by every forecast
branch), because the deterministic realized-dispatch sibling had no branch
scaffolding and would DROP a branch-distinct 3d_map (wind -> 0 -> VOLL).

W4c removes that limitation: the engine now PROJECTS each stochastic
parameter onto the parent's REALIZED branch for the auto-spawned sibling.
This fixture is W4b with the wind ``profile`` authored as a
branch-DISTINCT 3-deep Map ``branch -> time_start -> time -> value``:

    rlz  -> 0.50   (the REALIZED branch the sibling must read)
    high -> 0.80
    low  -> 0.10

The three values are mutually distinct AND distinct from W4b's old 0.40,
so the proof is unambiguous: the sibling reads 0.50 (not 0.40, not any
forecast, not the collapsed 0 fallback).  Branch weights are inherited
from the W4b/W8 builder: the realized ``rlz`` branch is a WEIGHT-0
designation and the two forecasts carry all the mass (high 0.5 / low 0.5,
SUM 1.0 on their own) — the corrected forecasts-only encoding.

Hand calculation (sibling dispatch solve)
-----------------------------------------
Realized data (rlz branch): demand 100 MW/step, wind cap 0.50 * 100 = 50
MW/step (cost 0), peaker fills residual 50 MW/step (cost 10).  The 3-step
timeset ANNUALIZES (timeset_duration=3 over years_represented=1 -> each
step represents 8760/3 h), so
    operational objective = 50 MW * 8760 h * 10 CUR/MWh = 4 380 000 CUR.

Negative controls (NOT reached by the sibling): 0.80/0.20 (high branch),
0.10/0.90 (low branch), 0/1.00-with-VOLL (the pre-W4c collapsed fallback
where the 3d_map is dropped and wind -> 0 -> 50 MW unserved at VOLL).

Run:  ~/venv-spi/bin/python tests/fixtures/build_w4c_divergent_dispatch.py
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent

STEPS = [f"t{i:04d}" for i in range(1, 4)]

# Branch-distinct realized/forecast wind-profile fractions.
RLZ_PROFILE = 0.50
HIGH_PROFILE = 0.80
LOW_PROFILE = 0.10


def _enc(obj: object) -> str:
    return base64.b64encode(json.dumps(obj).encode()).decode("ascii")


def _leaf(val: float) -> dict:
    """Nested (typed) Map: time -> value, for every step."""
    return {"type": "map", "index_type": "str",
            "data": [[t, val] for t in STEPS]}


def _branch_profile() -> dict:
    """3-deep ``branch -> time_start -> time -> value`` Map (time_start is
    the fan-at-first branching step ``t0001``)."""
    return {
        "index_type": "str",
        "data": [
            ["rlz", {"type": "map", "index_type": "str",
                     "data": [["t0001", _leaf(RLZ_PROFILE)]]}],
            ["high", {"type": "map", "index_type": "str",
                      "data": [["t0001", _leaf(HIGH_PROFILE)]]}],
            ["low", {"type": "map", "index_type": "str",
                     "data": [["t0001", _leaf(LOW_PROFILE)]]}],
        ],
    }


def build() -> dict:
    # Import the W4b builder so this fixture tracks it.
    if str(FIXTURES) not in sys.path:
        sys.path.insert(0, str(FIXTURES))
    from build_w4b_auto_dispatch import build as build_w4b

    spec = build_w4b()

    # Override the wind profile with the branch-DISTINCT 3d_map, authored
    # on the ``stoch`` alternative (wins over the base 0.40 and is foldable
    # only by the genuinely stochastic parent / the W4c-projected sibling).
    bp = _branch_profile()
    found = False
    for pv in spec["parameter_values"]:
        if pv[0] == "profile" and pv[1] == "wprof" and pv[2] == "profile":
            pv[3] = [_enc(bp), "map"]
            pv[4] = "stoch"
            found = True
    if not found:
        raise SystemExit("wprof profile value not found in W4b spec")
    return spec


def main() -> None:
    spec = build()
    out = FIXTURES / "w4c_divergent_dispatch.json"
    out.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"Wrote {out} "
          f"({len(spec['parameter_values'])} parameter values)")


if __name__ == "__main__":
    main()
