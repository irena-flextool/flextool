"""Build ``tests/fixtures/w4b_reserve_dispatch.json`` — the W4b
RESERVE full-path proof fixture.

This is the W4b ``w4b_auto_dispatch`` single-solve ``separate_dispatch``
fixture PLUS an up-reserve requirement on the ``elec`` node, provided by
the peaker.  Its sole purpose is to lock in the critique's core concern:
the engine-automatic realized-dispatch sibling must take the FULL DB
override path (``_solve_in_spine`` → True), NOT the synthetic
``<base>_<anchor>`` fallback.  The synthetic path returns ``None`` for
every solve-keyed helper and would silently DROP the reserve subsystem,
so a reserve variable / constraint PRESENT in the sibling's LP proves the
full path was taken.

Reserve subsystem (all reserve/upDown/group classes are
active_by_default, so no entity_alternatives are needed):

  * reserve ``r1``, direction ``up``;
  * group ``res_g`` spanning node ``elec`` (``group__node``);
  * ``reserve__upDown__group[r1, up, res_g]`` — ``reserve_method =
    timeseries_only``, ``reservation`` = 10 MW/step, a penalty so an
    unmet requirement is a priced slack rather than infeasibility;
  * ``reserve__upDown__unit__node[r1, up, peaker, elec]`` — the peaker is
    a reliable provider (reliability 1, max_share 1), so the invest solve
    sizes the peaker for 60 MW energy + 10 MW up-reserve headroom and the
    deterministic dispatch sibling holds that reserve on the fixed
    capacity.

Run:  ~/venv-spi/bin/python tests/fixtures/build_w4b_reserve_dispatch.py
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent

STEPS = [f"t{i:04d}" for i in range(1, 4)]
RESERVE_MW = 10.0


def _pack(value: object, value_type: str) -> list:
    return [base64.b64encode(json.dumps(value).encode()).decode("ascii"),
            value_type]


def _map(pairs: list[tuple[str, object]]) -> dict:
    return {"index_type": "str", "data": [[k, v] for k, v in pairs]}


def build() -> dict:
    import sys

    if str(FIXTURES) not in sys.path:
        sys.path.insert(0, str(FIXTURES))
    from build_w4b_auto_dispatch import build as build_w4b

    spec = build_w4b()

    # ---- reserve entities (active_by_default classes) ------------------
    spec["entities"] += [
        ["reserve", "r1", None],
        ["upDown", "up", None],
        ["group", "res_g", None],
        ["group__node", ["res_g", "elec"], None],
        ["reserve__upDown__group", ["r1", "up", "res_g"], None],
        ["reserve__upDown__unit__node", ["r1", "up", "peaker", "elec"], None],
    ]

    # ---- reserve parameter values (on the 'base' alternative) ----------
    reservation = _map([(t, RESERVE_MW) for t in STEPS])
    spec["parameter_values"] += [
        ["reserve__upDown__group", ["r1", "up", "res_g"], "reserve_method",
         _pack("timeseries_only", "str"), "base"],
        ["reserve__upDown__group", ["r1", "up", "res_g"], "reservation",
         _pack(reservation, "map"), "base"],
        ["reserve__upDown__group", ["r1", "up", "res_g"], "penalty_reserve",
         _pack(5000.0, "float"), "base"],
        ["reserve__upDown__unit__node", ["r1", "up", "peaker", "elec"],
         "reliability", _pack(1.0, "float"), "base"],
        ["reserve__upDown__unit__node", ["r1", "up", "peaker", "elec"],
         "max_share", _pack(1.0, "float"), "base"],
    ]
    return spec


def main() -> None:
    spec = build()
    out = FIXTURES / "w4b_reserve_dispatch.json"
    out.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"Wrote {out} "
          f"({len(spec['parameter_values'])} parameter values)")


if __name__ == "__main__":
    main()
