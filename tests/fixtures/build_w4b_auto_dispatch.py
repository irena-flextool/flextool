"""Build ``tests/fixtures/w4b_auto_dispatch.json`` — the W4b
"engine-automatic realized dispatch" fixture.

This is W8.1's ``stoch_realized_dispatch`` fixture collapsed to a SINGLE
top-level solve that authors ``dispatch_realization_method =
separate_dispatch``.  Where W8.1 hand-declares a second
``stoch_rd_realized_dispatch`` solve, W4b removes that author-declared
solve entirely and lets the engine AUTOMATICALLY register the
deterministic realized-dispatch sibling ``stoch_rd__realized_dispatch``.

Transforms applied to the W8.1 spec (so the two never drift):

  1. ``model.solves`` = ``[stoch_rd]`` (drop the author-declared
     ``stoch_rd_realized_dispatch``);
  2. remove the ``stoch_rd_realized_dispatch`` solve entity + its
     parameter values;
  3. set ``solve.stoch_rd.dispatch_realization_method =
     separate_dispatch`` (W4a schema).

The engine then spawns ``stoch_rd__realized_dispatch`` in memory and it
reproduces the four W8.1 oracles.

Run:  ~/venv-spi/bin/python tests/fixtures/build_w4b_auto_dispatch.py
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent

PARENT = "stoch_rd"
AUTHOR_DISPATCH = "stoch_rd_realized_dispatch"


def _enc(obj: object) -> str:
    return base64.b64encode(json.dumps(obj).encode()).decode("ascii")


def build() -> dict:
    # Import the W8.1 builder so this fixture tracks it.
    import sys

    if str(FIXTURES) not in sys.path:
        sys.path.insert(0, str(FIXTURES))
    from build_stoch_realized_dispatch import build as build_w8

    spec = build_w8()

    # 1. model.solves -> [stoch_rd] only.
    for pv in spec["parameter_values"]:
        if pv[0] == "model" and pv[2] == "solves":
            pv[3] = [_enc({"value_type": "str", "data": [PARENT]}), "array"]

    # 2. drop the author-declared dispatch solve entity + its params.
    spec["entities"] = [
        e for e in spec["entities"]
        if not (e[0] == "solve" and e[1] == AUTHOR_DISPATCH)
    ]
    spec["parameter_values"] = [
        pv for pv in spec["parameter_values"]
        if not (pv[0] == "solve" and pv[1] == AUTHOR_DISPATCH)
    ]

    # 3. author separate_dispatch on the surviving single top-level solve,
    #    on the same alternative its other params use ("init").
    spec["parameter_values"].append(
        ["solve", PARENT, "dispatch_realization_method",
         [_enc("separate_dispatch"), "str"], "init"]
    )
    return spec


def main() -> None:
    spec = build()
    out = FIXTURES / "w4b_auto_dispatch.json"
    out.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"Wrote {out} "
          f"({len(spec['parameter_values'])} parameter values)")


if __name__ == "__main__":
    main()
