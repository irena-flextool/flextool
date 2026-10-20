"""Build ``tests/fixtures/stoch_two_period_nreal0.json`` — the n_real=0
forecast-to-forecast non-anticipativity witness (R-NA re-grounding).

Corrected stochastic model: the stochastic fan is FORECASTS ONLY — there is
NO realised member in the weighted cohort, and non-anticipativity ties the
forecasts to EACH OTHER (via a deterministic reference forecast), never to a
realised anchor.  This fixture is the capability witness for ``n_real = 0``:
ZERO ``realized: yes`` rows, yet the storage NA family still fires, re-grounds
to a per-anchor-group forecast star, and ties the forecasts' storage states
equal over the committed window — all with no realised/trunk period tied.

Structure (reuses the Slice G ``stoch_two_period_free_ops`` topology — two
3-step periods ``p2035`` / ``p2040``, branch-asymmetric wind timing, a
stochastic-group storage node ``resv``, a shared investable ``base`` unit),
with two changes:

* ``stochastic_branches`` — all three branches ``mid`` / ``up`` / ``low`` are
  FORECASTS (``realized: no``), each with nonzero weight 1.0 (probabilities
  1/3 each).  No ``realized: yes`` row ⟹ ``n_real = 0``.
* ``shared_operation_periods`` — a CURATED window ``[p2035, p2040]`` (the
  reveal periods) so ``dt_non_anticipativity`` populates from the window
  independently of any ``realized_dispatch`` (there is no realised branch to
  seed the legacy union at ``n_real = 0``; design B1).

Behaviour: the storage NA family re-grounds each anchor group to a forecast
star keyed on the lexicographically-smallest forecast (``p2035_low`` /
``p2040_low``), ties ``v_state[resv]`` equal across the forecast copies over
the windowed steps, and leaves the base/phantom period ``p2035`` / ``p2040``
untied.  Solve is Optimal.

Run:  ~/venv-spi/bin/python tests/fixtures/build_stoch_two_period_nreal0.py
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import build_stoch_two_period_free_ops as _free_ops

FIXTURES = Path(__file__).resolve().parent


def _pack(value: object, value_type: object) -> list:
    return [
        base64.b64encode(json.dumps(value).encode()).decode("ascii"),
        value_type,
    ]


def _map(pairs: list[tuple[str, object]]) -> dict:
    return {"index_type": "str", "data": [[k, v] for k, v in pairs]}


def _nested(pairs: list[tuple[str, object]]) -> dict:
    return {"type": "map", "index_type": "str",
            "data": [[k, v] for k, v in pairs]}


def build() -> dict:
    spec = _free_ops.build()

    # n_real = 0: all three branches are FORECASTS (realized: no), weight 1.0.
    stochastic_branches = _map([
        ("p2035", _nested([
            ("mid", _nested([("t0001", _nested([("no", 1.0)]))])),
            ("up",  _nested([("t0001", _nested([("no", 1.0)]))])),
            ("low", _nested([("t0001", _nested([("no", 1.0)]))])),
        ])),
    ])

    # Curated NA window = the reveal periods, so dt_non_anticipativity
    # populates from the window (no realized_dispatch at n_real=0).
    window = _pack({"value_type": "str", "data": ["p2035", "p2040"]}, "array")

    new_pv = []
    for row in spec["parameter_values"]:
        # Swap the stochastic_branches payload for the all-forecast fan.
        if row[0] == "solve" and row[2] == "stochastic_branches":
            row = [row[0], row[1], row[2], _pack(stochastic_branches, "map"),
                   row[4]]
        # Drop the free_ops empty-window row (replaced by the curated window).
        if row[0] == "solve" and row[2] == "shared_operation_periods":
            continue
        new_pv.append(row)
    # Curated window lives in the ``stoch`` alternative (seen by the scenario).
    new_pv.append(["solve", "stoch_2p_free", "shared_operation_periods",
                   window, "stoch"])
    spec["parameter_values"] = new_pv

    # Single n_real=0 scenario (drop the free_ops / pinned_ops pair and the
    # ``freeops`` alternative — unused here).
    spec["alternatives"] = [
        a for a in spec["alternatives"] if a[0] != "freeops"
    ]
    spec["scenarios"] = [
        ["nreal0", False,
         "n_real=0: three forecasts (realized:no), curated NA window — "
         "forecast-to-forecast storage NA with no realised branch"],
    ]
    spec["scenario_alternatives"] = [
        ["nreal0", "init", "base"],
        ["nreal0", "base", "stoch"],
        ["nreal0", "stoch", None],
    ]
    return spec


def main() -> None:
    spec = build()
    out = FIXTURES / "stoch_two_period_nreal0.json"
    out.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"Wrote {out} "
          f"({len(spec['parameter_values'])} parameter values)")


if __name__ == "__main__":
    main()
