"""Build ``tests/fixtures/stoch_two_period.json`` — the multi-period
stochastic continuation fan-out fixture (design
``specs/step05_continuation_fanout_design.md`` §5.1).

Schema sections (entity_classes, parameter_value_lists,
parameter_groups, parameter_definitions, parameter_types) are copied
from the committed ``stochastics.json`` so the fixture always carries
the current schema version; only the data sections are authored here.

Model: one 6-step timeline split into two 3-step periods
(p2035 = t0001-t0003, p2040 = t0004-t0006); a demand node ``city``
(100 MW then 110 MW), a free wind unit capped by a branch-varying
profile (via ``group__unit`` so no node enters the stochastic group),
and a 50 EUR/MWh thermal backstop.  Branches ``rlz`` (realized,
weight 1.0) and ``low`` (weight 3.0) are declared at p2035/t0001, so
p2040 is a CONTINUATION period — the surface the fan-out fix repairs.

Hand calculation (design §5.2): stochastic objective 70 080 000,
deterministic control 43 800 000.

Run:  ~/venv-spi/bin/python tests/fixtures/build_stoch_two_period.py
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent

STEPS = [f"t{i:04d}" for i in range(1, 7)]
P1_STEPS = STEPS[:3]
P2_STEPS = STEPS[3:]


def _pack(value: object, value_type: str) -> list:
    """Serialize a parsed value into the fixture's [base64, type] pair."""
    return [
        base64.b64encode(json.dumps(value).encode()).decode("ascii"),
        value_type,
    ]


def _map(pairs: list[tuple[str, object]]) -> dict:
    """Top-level Spine Map payload."""
    return {"index_type": "str", "data": [[k, v] for k, v in pairs]}


def _nested(pairs: list[tuple[str, object]]) -> dict:
    """Nested (typed) Spine Map payload."""
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
        ["stoch", "Stochastic branches + branch-varying wind profile"],
        ["horizon", "output_horizon = yes"],
    ]
    spec["scenarios"] = [
        ["det", False, "Deterministic control"],
        ["stoch", False, "Two-period stochastic, realized-only outputs"],
        ["stoch_horizon", False, "Two-period stochastic, horizon outputs"],
    ]
    spec["scenario_alternatives"] = [
        ["det", "init", "base"],
        ["det", "base", None],
        ["stoch", "init", "base"],
        ["stoch", "base", "stoch"],
        ["stoch", "stoch", None],
        ["stoch_horizon", "init", "base"],
        ["stoch_horizon", "base", "stoch"],
        ["stoch_horizon", "stoch", "horizon"],
        ["stoch_horizon", "horizon", None],
    ]

    spec["entities"] = [
        ["group", "stoch_g", None],
        ["model", "flextool", None],
        ["node", "city", None],
        ["profile", "wprof", None],
        ["solve", "stoch_2p", None],
        ["timeline", "y6h", None],
        ["timeset", "ts1", None],
        ["timeset", "ts2", None],
        ["unit", "thermal", None],
        ["unit", "wind", None],
        ["group__unit", ["stoch_g", "wind"], None],
        ["unit__outputNode", ["thermal", "city"], None],
        ["unit__outputNode", ["wind", "city"], None],
        ["unit__node__profile", ["wind", "city", "wprof"], None],
    ]
    # node / unit classes are active_by_default=False.
    spec["entity_alternatives"] = [
        ["node", ["city"], "init", True],
        ["unit", ["thermal"], "init", True],
        ["unit", ["wind"], "init", True],
    ]

    # ---- init: model / solve / time structure --------------------------
    stochastic_branches = _map([
        ("p2035", _nested([
            ("rlz", _nested([("t0001", _nested([("yes", 1.0)]))])),
            ("low", _nested([("t0001", _nested([("no", 3.0)]))])),
        ])),
    ])
    profile_base = _map(
        [(t, 0.6) for t in P1_STEPS] + [(t, 0.5) for t in P2_STEPS]
    )
    profile_stoch = _map([
        ("rlz", _nested([
            ("t0001", _nested([(t, 0.6) for t in P1_STEPS])),
            ("t0004", _nested([(t, 0.5) for t in P2_STEPS])),
        ])),
        ("low", _nested([
            ("t0001", _nested([(t, 0.2) for t in P1_STEPS])),
            ("t0004", _nested([(t, 0.1) for t in P2_STEPS])),
        ])),
    ])
    inflow = _map(
        [(t, -100.0) for t in P1_STEPS] + [(t, -110.0) for t in P2_STEPS]
    )

    spec["parameter_values"] = [
        ["model", "flextool", "solves",
         _pack({"value_type": "str", "data": ["stoch_2p"]}, "array"),
         "init"],
        ["model", "flextool", "output_horizon", _pack("yes", "str"),
         "horizon"],
        ["timeline", "y6h", "timestep_duration",
         _pack(_map([(t, 1.0) for t in STEPS]), "map"), "init"],
        ["timeset", "ts1", "timeline", _pack("y6h", "str"), "init"],
        ["timeset", "ts1", "timeset_duration",
         _pack(_map([("t0001", 3.0)]), "map"), "init"],
        ["timeset", "ts2", "timeline", _pack("y6h", "str"), "init"],
        ["timeset", "ts2", "timeset_duration",
         _pack(_map([("t0004", 3.0)]), "map"), "init"],
        ["solve", "stoch_2p", "period_timeset",
         _pack(_map([("p2035", "ts1"), ("p2040", "ts2")]), "map"), "init"],
        ["solve", "stoch_2p", "realized_periods",
         _pack({"value_type": "str", "data": ["p2035", "p2040"]}, "array"),
         "init"],
        ["solve", "stoch_2p", "solve_mode",
         _pack("single_solve", "str"), "init"],
        # Explicit 1.0/period — numerically identical to the intended
        # "default to one year per period" fallback.  The fallback
        # itself is unusable for multi-period solves at HEAD: the
        # Rule-6 default in ``_timeline.py`` (``:557-569``) checks
        # membership inside its per-period loop, so it seeds only ONE
        # period and the years_represented completeness check then
        # fails for the rest.  (Pre-existing, deterministic-path bug —
        # out of scope for the fan-out fix.)
        ["solve", "stoch_2p", "years_represented",
         _pack(_map([("p2035", 1.0), ("p2040", 1.0)]), "map"), "init"],
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
        ["solve", "stoch_2p", "stochastic_branches",
         _pack(stochastic_branches, "map"), "stoch"],
        ["profile", "wprof", "profile", _pack(profile_stoch, "map"),
         "stoch"],
    ]
    return spec


def main() -> None:
    spec = build()
    out = FIXTURES / "stoch_two_period.json"
    out.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"Wrote {out} "
          f"({len(spec['parameter_values'])} parameter values)")


if __name__ == "__main__":
    main()
