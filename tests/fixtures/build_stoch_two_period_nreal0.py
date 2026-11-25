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
stochastic-group storage node ``resv``), with two changes:

* ``stochastic_branches`` — all three branches ``mid`` / ``up`` / ``low`` are
  FORECASTS (``realized: no``), each with nonzero weight 1.0 (probabilities
  1/3 each).  No ``realized: yes`` row ⟹ ``n_real = 0``.
* a DISPATCH solve: no ``invest_periods``; ``base`` has an existing 12 MW
  instead of being investable (an investment solve ties no operations).  A
  dispatch solve ties its forecasts over its committed window, the steps of
  its ``realized_periods`` p2035 / p2040.

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

from flextool.update_flextool.test_fixtures import write_built_fixture

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

    new_pv = []
    for row in spec["parameter_values"]:
        # A dispatch solve: no invest periods, base existing 12 MW (the shared capacity of the free_ops optimum
        # is 2 MW; 12 MW covers the demand in every step without the storage).
        if row[0] == "solve" and row[2] in ("invest_periods", "realized_invest_periods"):
            continue
        if row[0] == "unit" and row[1] == "base" and row[2] in (
                "invest_method", "invest_cost", "discount_rate", "lifetime_technical", "invest_max_total"):
            continue
        if row[0] == "unit" and row[1] == "base" and row[2] == "existing":
            row = [row[0], row[1], row[2], _pack(12.0, "float"), row[4]]
        # Swap the stochastic_branches payload for the all-forecast fan.
        if row[0] == "solve" and row[2] == "stochastic_branches":
            row = [row[0], row[1], row[2], _pack(stochastic_branches, "map"),
                   row[4]]
        new_pv.append(row)
    spec["parameter_values"] = new_pv

    # Single n_real=0 scenario (the free_ops scenario is unused here).
    spec["scenarios"] = [
        ["nreal0", False,
         "n_real=0: three forecasts (realized:no) in a dispatch solve — "
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
    write_built_fixture(spec, out)
    print(f"Wrote {out} "
          f"({len(spec['parameter_values'])} parameter values)")


if __name__ == "__main__":
    main()
