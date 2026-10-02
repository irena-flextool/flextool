"""Build ``tests/fixtures/stoch_mid_horizon.json`` — the mid-horizon
(Option B "realized-not-past-reveal") continuation fan-out fixture.

Schema sections are copied from the committed ``stochastics.json`` so the
fixture always carries the current schema version; only the data
sections are authored here.

Model: one 9-step timeline split into THREE 3-step periods
(``p2030`` = t0001-t0003 COMMITTED, ``p2035`` = t0004-t0006 REVEAL,
``p2040`` = t0007-t0009 CONTINUATION); a demand node ``city``
(100 / 100 / 110 MW), a free wind unit capped by a branch-varying
profile (via ``group__unit`` so no node enters the stochastic group),
and a 50 EUR/MWh thermal backstop.

Branches are declared at ``p2035``/t0004 (a MID-HORIZON reveal — the
committed period ``p2030`` precedes it): ``rlz`` (realized, weight 1.0),
``low`` (weight 3.0) and ``high`` (weight 1.0).  Under Option B the
realized trunk stops at the reveal: ``p2030`` + ``p2035`` are
real-named (committed + reveal), while ``p2040`` carries ONLY the
synthetic forecast copies ``p2040_low`` / ``p2040_high``.

Weight expectations (Stage C): committed ``p2030`` = 1.0 (cohort-of-one);
reveal cohort {p2035, p2035_low, p2035_high} sums to 1 (0.2 / 0.6 / 0.2);
post-reveal forecasts {p2040_low, p2040_high} renormalize to sum 1
(0.75 / 0.25) with NO realized term.

Run:  ~/venv-spi/bin/python tests/fixtures/build_stoch_mid_horizon.py
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent

STEPS = [f"t{i:04d}" for i in range(1, 10)]
P0_STEPS = STEPS[0:3]   # p2030 committed
P1_STEPS = STEPS[3:6]   # p2035 reveal
P2_STEPS = STEPS[6:9]   # p2040 continuation


def _pack(value: object, value_type: str) -> list:
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
    src = json.loads((FIXTURES / "stochastics.json").read_text())

    spec: dict = {
        key: src[key]
        for key in (
            "entity_classes",
            "parameter_value_lists",
            "parameter_groups",
            "parameter_definitions",
            "parameter_types",
        )
    }

    spec["alternatives"] = [
        ["init", "Model, solve and time structure"],
        ["base", "System data (deterministic profile)"],
        ["stoch", "Mid-horizon stochastic branches + branch profile"],
        ["horizon", "output_horizon = yes"],
    ]
    spec["scenarios"] = [
        ["det", False, "Deterministic control"],
        ["mid", False, "Mid-horizon stochastic, realized-only outputs"],
        ["mid_horizon", False, "Mid-horizon stochastic, horizon outputs"],
    ]
    spec["scenario_alternatives"] = [
        ["det", "init", "base"],
        ["det", "base", None],
        ["mid", "init", "base"],
        ["mid", "base", "stoch"],
        ["mid", "stoch", None],
        ["mid_horizon", "init", "base"],
        ["mid_horizon", "base", "stoch"],
        ["mid_horizon", "stoch", "horizon"],
        ["mid_horizon", "horizon", None],
    ]

    spec["entities"] = [
        ["group", "stoch_g", None],
        ["model", "flextool", None],
        ["node", "city", None],
        ["profile", "wprof", None],
        ["solve", "stoch_mid", None],
        ["timeline", "y9h", None],
        ["timeset", "ts0", None],
        ["timeset", "ts1", None],
        ["timeset", "ts2", None],
        ["unit", "thermal", None],
        ["unit", "wind", None],
        ["group__unit", ["stoch_g", "wind"], None],
        ["unit__outputNode", ["thermal", "city"], None],
        ["unit__outputNode", ["wind", "city"], None],
        ["unit__node__profile", ["wind", "city", "wprof"], None],
    ]
    spec["entity_alternatives"] = [
        ["node", ["city"], "init", True],
        ["unit", ["thermal"], "init", True],
        ["unit", ["wind"], "init", True],
    ]

    # ---- init: model / solve / time structure --------------------------
    # Mid-horizon reveal: branches START at p2035/t0004, so p2030 is a
    # pre-reveal COMMITTED period.
    stochastic_branches = _map([
        ("p2035", _nested([
            ("rlz", _nested([("t0004", _nested([("yes", 1.0)]))])),
            ("low", _nested([("t0004", _nested([("no", 3.0)]))])),
            ("high", _nested([("t0004", _nested([("no", 1.0)]))])),
        ])),
    ])
    profile_base = _map(
        [(t, 0.6) for t in P0_STEPS]
        + [(t, 0.6) for t in P1_STEPS]
        + [(t, 0.5) for t in P2_STEPS]
    )
    # Branch-varying profile (3d_map).  rlz covers committed t0001 +
    # reveal t0004 (realized stops at reveal); forecasts cover reveal
    # t0004 + continuation t0007.
    profile_stoch = _map([
        ("rlz", _nested([
            ("t0001", _nested([(t, 0.6) for t in P0_STEPS])),
            ("t0004", _nested([(t, 0.6) for t in P1_STEPS])),
        ])),
        ("low", _nested([
            ("t0004", _nested([(t, 0.2) for t in P1_STEPS])),
            ("t0007", _nested([(t, 0.1) for t in P2_STEPS])),
        ])),
        ("high", _nested([
            ("t0004", _nested([(t, 0.4) for t in P1_STEPS])),
            ("t0007", _nested([(t, 0.3) for t in P2_STEPS])),
        ])),
    ])
    inflow = _map(
        [(t, -100.0) for t in P0_STEPS]
        + [(t, -100.0) for t in P1_STEPS]
        + [(t, -110.0) for t in P2_STEPS]
    )

    spec["parameter_values"] = [
        ["model", "flextool", "solves",
         _pack({"value_type": "str", "data": ["stoch_mid"]}, "array"),
         "init"],
        ["model", "flextool", "output_horizon", _pack("yes", "str"),
         "horizon"],
        ["timeline", "y9h", "timestep_duration",
         _pack(_map([(t, 1.0) for t in STEPS]), "map"), "init"],
        ["timeset", "ts0", "timeline", _pack("y9h", "str"), "init"],
        ["timeset", "ts0", "timeset_duration",
         _pack(_map([("t0001", 3.0)]), "map"), "init"],
        ["timeset", "ts1", "timeline", _pack("y9h", "str"), "init"],
        ["timeset", "ts1", "timeset_duration",
         _pack(_map([("t0004", 3.0)]), "map"), "init"],
        ["timeset", "ts2", "timeline", _pack("y9h", "str"), "init"],
        ["timeset", "ts2", "timeset_duration",
         _pack(_map([("t0007", 3.0)]), "map"), "init"],
        ["solve", "stoch_mid", "period_timeset",
         _pack(_map([("p2030", "ts0"), ("p2035", "ts1"),
                     ("p2040", "ts2")]), "map"), "init"],
        ["solve", "stoch_mid", "realized_periods",
         _pack({"value_type": "str",
                "data": ["p2030", "p2035", "p2040"]}, "array"),
         "init"],
        ["solve", "stoch_mid", "solve_mode",
         _pack("single_solve", "str"), "init"],
        # ---- base: system data ----------------------------------------
        ["node", "city", "node_type", _pack("balance", "str"), "base"],
        ["node", "city", "penalty_up", _pack(10000.0, "float"), "base"],
        ["node", "city", "penalty_down", _pack(10000.0, "float"), "base"],
        ["node", "city", "inflow", _pack(inflow, "map"), "base"],
        ["unit", "wind", "existing", _pack(100.0, "float"), "base"],
        ["unit", "wind", "efficiency", _pack(1.0, "float"), "base"],
        ["unit", "thermal", "existing", _pack(200.0, "float"), "base"],
        ["unit", "thermal", "efficiency", _pack(1.0, "float"), "base"],
        ["unit__outputNode", ["thermal", "city"], "other_operational_cost",
         _pack(50.0, "float"), "base"],
        ["unit__node__profile", ["wind", "city", "wprof"], "profile_method",
         _pack("upper_limit", "str"), "base"],
        ["profile", "wprof", "profile", _pack(profile_base, "map"), "base"],
        # ---- stoch: branches + branch-varying profile ------------------
        ["group", "stoch_g", "include_stochastics", _pack("yes", "str"),
         "stoch"],
        ["solve", "stoch_mid", "stochastic_branches",
         _pack(stochastic_branches, "map"), "stoch"],
        ["profile", "wprof", "profile", _pack(profile_stoch, "map"),
         "stoch"],
    ]
    return spec


def main() -> None:
    spec = build()
    out = FIXTURES / "stoch_mid_horizon.json"
    out.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"Wrote {out} "
          f"({len(spec['parameter_values'])} parameter values)")


if __name__ == "__main__":
    main()
