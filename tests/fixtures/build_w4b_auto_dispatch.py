"""Build ``tests/fixtures/w4b_auto_dispatch.json`` — the W4b
"engine-automatic realized dispatch" fixture.

This is W8.1's ``stoch_realized_dispatch`` fixture collapsed to a SINGLE
top-level investment solve.  Where W8.1 hand-declares a second
``stoch_rd_realized_dispatch`` solve, W4b removes that author-declared
solve entirely and lets the engine AUTOMATICALLY register the
deterministic realized-dispatch sibling ``stoch_rd__realized_dispatch``:
an investment solve whose ``stochastic_branches`` has a ``realized: yes``
branch reaching its realized periods gets a realization solve without any
``dispatch_realization_method`` (an investment solve does not read it; a
``separate_dispatch`` there is ignored with warning W-0640).

Transforms applied to the W8.1 spec (so the two never drift):

  1. ``model.solves`` = ``[stoch_rd]`` (drop the author-declared
     ``stoch_rd_realized_dispatch``);
  2. remove the ``stoch_rd_realized_dispatch`` solve entity + its
     parameter values, and the parent's ``dispatch_realization_method``.

The engine then spawns ``stoch_rd__realized_dispatch`` in memory and it
reproduces the four W8.1 oracles.

Run:  ~/venv-spi/bin/python tests/fixtures/build_w4b_auto_dispatch.py
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

from flextool.update_flextool.test_fixtures import write_built_fixture

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

    # 2. drop the author-declared dispatch solve entity + its params, and the
    #    parent's within_solve (an investment solve does not read the method).
    spec["entities"] = [
        e for e in spec["entities"]
        if not (e[0] == "solve" and e[1] == AUTHOR_DISPATCH)
    ]
    spec["parameter_values"] = [
        pv for pv in spec["parameter_values"]
        if not (pv[0] == "solve" and pv[1] == AUTHOR_DISPATCH)
        and not (pv[0] == "solve" and pv[1] == PARENT and pv[2] == "dispatch_realization_method")
    ]

    return spec


def main() -> None:
    spec = build()
    out = FIXTURES / "w4b_auto_dispatch.json"
    write_built_fixture(spec, out)
    print(f"Wrote {out} "
          f"({len(spec['parameter_values'])} parameter values)")


if __name__ == "__main__":
    main()
