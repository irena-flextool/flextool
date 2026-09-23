"""Build ``tests/fixtures/stoch_two_period_invest.json`` — the Slice D
per-scenario (wait-and-see) stochastic-investment gate fixture
(design ``specs/sliceD_recourse_invest_design.md`` §15.1, variant A).

Extends the ``stoch_two_period`` shape (two 3-step periods p2035 / p2040,
branches ``rlz`` w=1.0 / ``low`` w=3.0 declared at p2035/t0001 so p2040 is
a CONTINUATION period) with an INVEST entity.  The branch variation rides
a UNIT profile through ``group__unit`` (the base fixture's mechanism) — no
NODE enters the stochastic group, so the node-side NA net-charge pinning
(``model.py`` ``non_anticipativity_storage_use``, gated on ``group_node``
× ``groupStochastic``) never fires and the LP stays separable per leaf
(hand-calc-able).  A node-authored branch-keyed ``inflow`` would NOT work
without the node in the group — the stochastic inflow fold-in
(``_emit_period_params``) gates on ``group__node`` × include_stochastics —
which is why the demand variation is supply-side here:

  * node ``elec`` — balance, high penalty (VOLL), DETERMINISTIC demand
    100 MW (p2035) / 130 MW (p2040).
  * unit ``wind`` → ``elec`` — existing 100 MW, cost 0, ``upper_limit``
    profile branch-varying via the stochastic group: rlz 0.40/0.40,
    low 0.20/0.10 → wind supply rlz 40/40, low 20/10 → peaker residual
    rlz 60/90, low 80/120 (the §15.1 demand table).
  * unit ``peaker`` → ``elec`` — variable cost 0, virtual_unitsize 1 MW,
    invest_method=invest_total, invest_cost 1 CUR/kW, discount_rate 0.05,
    lifetime 1 (→ annuity 1000·CRF(0.05,1) = 1050 exactly),
    invest_max_total 150 (the design §15.1 per-path demonstration value:
    each leaf's own total — realized 60+30 = 90, low 80+40 = 120 — is
    ≤ 150 so the cap is non-binding PER LEAF, while the legacy all-``d``
    sum 210 would spuriously bind.  With D6 per-path caps
    ``maxInvest_entity_total_path`` splits the cap by scenario leaf so
    the objective stays 196 875).

Hand calculation (design §15.1): with the invest axis fanned, per-scenario
optimal invest {p2035:60, p2040:30, p2035_low:80, p2040_low:40}; annuity
windows 2100 (branching) / 1050 (continuation); weighted objective

    0.25·(60·2100 + 30·1050) + 0.75·(80·2100 + 40·1050) = 196 875.

Run:  ~/venv-spi/bin/python tests/fixtures/build_stoch_two_period_invest.py
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

    # ``stochastics.json`` still carries the pre-v70 schema, so the
    # ``solve.stochastic_invest_method`` opt-in (Slice C, v70) is not yet
    # defined — but ``json_to_db`` validates authored values against the
    # embedded schema BEFORE ``migrate_database`` runs.  Inject the v70
    # parameter definition + value list here so the ``recourse`` value
    # imports cleanly; the migration's idempotent ``add_update_item`` then
    # reconciles it to the canonical v70 shape.
    if not any(vl[0] == "stochastic_invest_methods"
               for vl in spec["parameter_value_lists"]):
        spec["parameter_value_lists"] += [
            ["stochastic_invest_methods", _pack("none", "str")],
            ["stochastic_invest_methods", _pack("recourse", "str")],
        ]
    if not any(pd[0] == "solve" and pd[1] == "stochastic_invest_method"
               for pd in spec["parameter_definitions"]):
        spec["parameter_definitions"].append([
            "solve", "stochastic_invest_method",
            _pack("none", "str"), "stochastic_invest_methods",
            "How stochastic branches participate in investment "
            "(none/recourse).", "solve_advanced",
        ])

    spec["alternatives"] = [
        ["init", "Model, solve and time structure"],
        ["base", "System data (peaker invest + wind + elec demand)"],
        ["stoch", "Stochastic branches + branch-varying wind profile"],
        ["recourse", "stochastic_invest_method = recourse"],
        ["horizon", "model.output_horizon = yes (branch invest rows)"],
    ]
    spec["scenarios"] = [
        ["recourse", False, "Per-scenario stochastic investment (flag on)"],
        ["recourse_horizon", False,
         "Recourse + output_horizon (non-realized branch invest visible)"],
    ]
    spec["scenario_alternatives"] = [
        ["recourse", "init", "base"],
        ["recourse", "base", "stoch"],
        ["recourse", "stoch", "recourse"],
        ["recourse", "recourse", None],
        ["recourse_horizon", "init", "base"],
        ["recourse_horizon", "base", "stoch"],
        ["recourse_horizon", "stoch", "recourse"],
        ["recourse_horizon", "recourse", "horizon"],
        ["recourse_horizon", "horizon", None],
    ]

    spec["entities"] = [
        ["group", "stoch_g", None],
        ["model", "flextool", None],
        ["node", "elec", None],
        ["profile", "wprof", None],
        ["solve", "stoch_2p_inv", None],
        ["timeline", "y6h", None],
        ["timeset", "ts1", None],
        ["timeset", "ts2", None],
        ["unit", "peaker", None],
        ["unit", "wind", None],
        ["group__unit", ["stoch_g", "wind"], None],
        ["unit__outputNode", ["peaker", "elec"], None],
        ["unit__outputNode", ["wind", "elec"], None],
        ["unit__node__profile", ["wind", "elec", "wprof"], None],
    ]
    spec["entity_alternatives"] = [
        ["node", ["elec"], "init", True],
        ["unit", ["peaker"], "init", True],
        ["unit", ["wind"], "init", True],
    ]

    # ---- init: model / solve / time structure --------------------------
    stochastic_branches = _map([
        ("p2035", _nested([
            ("rlz", _nested([("t0001", _nested([("yes", 1.0)]))])),
            ("low", _nested([("t0001", _nested([("no", 3.0)]))])),
        ])),
    ])
    # Deterministic profile (base alternative) — matches the rlz branch.
    profile_base = _map(
        [(t, 0.40) for t in P1_STEPS] + [(t, 0.40) for t in P2_STEPS]
    )
    # Branch-varying wind profile (stoch alternative): wind supply
    # rlz 40/40 MW, low 20/10 MW → peaker residual rlz 60/90, low 80/120.
    profile_stoch = _map([
        ("rlz", _nested([
            ("t0001", _nested([(t, 0.40) for t in P1_STEPS])),
            ("t0004", _nested([(t, 0.40) for t in P2_STEPS])),
        ])),
        ("low", _nested([
            ("t0001", _nested([(t, 0.20) for t in P1_STEPS])),
            ("t0004", _nested([(t, 0.10) for t in P2_STEPS])),
        ])),
    ])
    # Deterministic demand: 100 MW at p2035 steps, 130 MW at p2040 steps.
    inflow = _map(
        [(t, -100.0) for t in P1_STEPS] + [(t, -130.0) for t in P2_STEPS]
    )

    spec["parameter_values"] = [
        ["model", "flextool", "solves",
         _pack({"value_type": "str", "data": ["stoch_2p_inv"]}, "array"),
         "init"],
        ["timeline", "y6h", "timestep_duration",
         _pack(_map([(t, 1.0) for t in STEPS]), "map"), "init"],
        ["timeset", "ts1", "timeline", _pack("y6h", "str"), "init"],
        ["timeset", "ts1", "timeset_duration",
         _pack(_map([("t0001", 3.0)]), "map"), "init"],
        ["timeset", "ts2", "timeline", _pack("y6h", "str"), "init"],
        ["timeset", "ts2", "timeset_duration",
         _pack(_map([("t0004", 3.0)]), "map"), "init"],
        ["solve", "stoch_2p_inv", "period_timeset",
         _pack(_map([("p2035", "ts1"), ("p2040", "ts2")]), "map"), "init"],
        ["solve", "stoch_2p_inv", "realized_periods",
         _pack({"value_type": "str", "data": ["p2035", "p2040"]}, "array"),
         "init"],
        ["solve", "stoch_2p_inv", "invest_periods",
         _pack({"value_type": "str", "data": ["p2035", "p2040"]}, "array"),
         "init"],
        ["solve", "stoch_2p_inv", "realized_invest_periods",
         _pack({"value_type": "str", "data": ["p2035", "p2040"]}, "array"),
         "init"],
        ["solve", "stoch_2p_inv", "solve_mode",
         _pack("single_solve", "str"), "init"],
        ["solve", "stoch_2p_inv", "years_represented",
         _pack(_map([("p2035", 1.0), ("p2040", 1.0)]), "map"), "init"],
        # ---- base: system data ----------------------------------------
        ["node", "elec", "node_type", _pack("balance", "str"), "base"],
        ["node", "elec", "penalty_up", _pack(10000.0, "float"), "base"],
        ["node", "elec", "penalty_down", _pack(10000.0, "float"), "base"],
        ["node", "elec", "inflow", _pack(inflow, "map"), "base"],
        ["unit", "wind", "existing", _pack(100.0, "float"), "base"],
        ["unit", "wind", "efficiency", _pack(1.0, "float"), "base"],
        ["unit__node__profile", ["wind", "elec", "wprof"], "profile_method",
         _pack("upper_limit", "str"), "base"],
        ["profile", "wprof", "profile", _pack(profile_base, "map"), "base"],
        ["unit", "peaker", "efficiency", _pack(1.0, "float"), "base"],
        ["unit", "peaker", "existing", _pack(0.0, "float"), "base"],
        ["unit", "peaker", "virtual_unitsize", _pack(1.0, "float"), "base"],
        ["unit", "peaker", "invest_method",
         _pack("invest_total", "str"), "base"],
        ["unit", "peaker", "invest_cost", _pack(1.0, "float"), "base"],
        ["unit", "peaker", "discount_rate", _pack(0.05, "float"), "base"],
        ["unit", "peaker", "lifetime", _pack(1.0, "float"), "base"],
        ["unit", "peaker", "invest_max_total", _pack(150.0, "float"),
         "base"],
        ["unit__outputNode", ["peaker", "elec"], "other_operational_cost",
         _pack(0.0, "float"), "base"],
        # ---- stoch: branches + branch-varying profile ------------------
        ["group", "stoch_g", "include_stochastics", _pack("yes", "str"),
         "stoch"],
        ["solve", "stoch_2p_inv", "stochastic_branches",
         _pack(stochastic_branches, "map"), "stoch"],
        ["profile", "wprof", "profile", _pack(profile_stoch, "map"),
         "stoch"],
        # ---- recourse: opt-in flag -------------------------------------
        ["solve", "stoch_2p_inv", "stochastic_invest_method",
         _pack("recourse", "str"), "recourse"],
        # ---- horizon: surface non-realized branch invest (§11.2) -------
        ["model", "flextool", "output_horizon", _pack("yes", "str"),
         "horizon"],
    ]
    return spec


def main() -> None:
    spec = build()
    out = FIXTURES / "stoch_two_period_invest.json"
    out.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"Wrote {out} "
          f"({len(spec['parameter_values'])} parameter values)")


if __name__ == "__main__":
    main()
