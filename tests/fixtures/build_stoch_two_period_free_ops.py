"""Build ``tests/fixtures/stoch_two_period_free_ops.json`` — the Slice G
configurable-non-anticipativity fixture (design
``specs/sliceG_configurable_nonanticipativity_design.md`` §7).

Goal: a two-period, t0-branching stochastic model that exercises the
operational non-anticipativity WINDOW knob (``non_anticipativity_periods``)
on a stochastic-group storage node, with a UNIQUE per-branch storage
optimum so the "operations differ across branches" assertion is
deterministic (design §7 / F6).

Structure
---------
Two 3-step periods ``p2035`` (t0001-t0003) / ``p2040`` (t0004-t0006),
step duration 1 h.  Three branches ``mid`` / ``up`` / ``low`` declared at
**t0001** (the first step of the first period) — a wait-and-see-from-start
(t0-branching) topology, so EVERY period is per-branch.  ``mid`` is the
realized branch (weight 2); ``up`` / ``low`` carry weight 1 each
(probabilities 0.5 / 0.25 / 0.25).

* node ``elec`` — balance, constant demand 12 MW/step.
* unit ``wind`` — free (cost 0), existing 100 MW, in ``stoch_g`` (so its
  branch-varying profile is stochastic).  The profile share of the 100 MW
  cap is BRANCH-VARYING and asymmetric in TIMING, engineered so each
  branch's cost-optimal storage trajectory is strictly distinct:
    - ``up``  — surplus early (steps 1-3 share 0.20 -> wind 20, +8 over
      demand), deficit late (steps 4-6 share 0.00 -> deficit 12).
    - ``low`` — the mirror: deficit early, surplus late.
    - ``mid`` — flat 0.10 -> wind 10, a steady 2 MW deficit every step.
  Total wind per branch = 60 MWh < demand 72 MWh, so a net 12 MWh must
  come from the investable ``base`` unit in EVERY branch (a shared,
  deterministic capacity floor).
* node ``resv`` — storage in ``stoch_g`` (via ``group__node``), existing
  1000 MWh, start fixed at 0.5 (500 MWh), reference end 0.5.  NO inflow
  (deterministic) — so the ONLY driver of the storage state is the
  controllable charge/discharge, which is exactly what the storage NA
  family pins.  ``charge`` (elec->resv) and ``discharge`` (resv->elec) are
  free existing 100 MW units; a tiny 0.01 CUR/MWh discharge cost breaks
  degenerate cycling so the per-branch optimum is unique.
* unit ``base`` — investable peaker -> elec, ``stochastic_invest_method =
  none`` so there is exactly ONE shared ``v_invest`` (R-O6: invest stays
  realized-only, never fans to the branch copies).  ``other_operational_
  cost`` 5 CUR/MWh so wind + stored energy is preferred to fresh base.

Behaviour under the two scenarios (identical data, differing ONLY in the
``non_anticipativity_periods`` value):

* ``pinned_ops`` — ``non_anticipativity_periods`` UNSET -> legacy window
  = realized_dispatch u fix_storage = the whole horizon.  The storage NA
  family ties the branches' charge/discharge net-charge equal at every
  step, so with no branch inflow the ``v_state[resv]`` trajectory is
  IDENTICAL across ``mid`` / ``up`` / ``low`` — the branches cannot follow
  their opposite surplus/deficit timing and pay for it.  (Byte-parity
  guard: this is the flag-off behaviour.)
* ``free_ops`` — ``non_anticipativity_periods = []`` -> empty window ->
  ZERO ``non_anticipativity_*`` rows -> each branch charges/discharges to
  its own optimum, so ``v_state[resv]`` DIVERGES across branches, and the
  objective is <= the pinned objective (freeing operations only relaxes
  the feasible set).

Run:  ~/venv-spi/bin/python tests/fixtures/build_stoch_two_period_free_ops.py
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent

STEPS = [f"t{i:04d}" for i in range(1, 7)]
P1_STEPS = STEPS[:3]
P2_STEPS = STEPS[3:]


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

    # ``stochastics.json`` carries the pre-v70 schema; inject the v70
    # ``solve.stochastic_invest_method`` opt-in and the v71
    # ``solve.non_anticipativity_periods`` window knob before import
    # (json_to_db validates against the embedded schema BEFORE the
    # idempotent v70/v71 migrations reconcile it).
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
    if not any(pd[0] == "solve" and pd[1] == "non_anticipativity_periods"
               for pd in spec["parameter_definitions"]):
        # Array param (default null, no value list) — mirrors
        # realized_periods / fix_storage_periods (v71, Slice G).
        spec["parameter_definitions"].append([
            "solve", "non_anticipativity_periods",
            _pack(None, None), None,
            "Array of periods over which operational non-anticipativity "
            "is enforced (unset = legacy window; [] = free ops).",
            "solve_advanced",
        ])

    spec["alternatives"] = [
        ["init", "Model, solve and time structure"],
        ["base", "System data (elec demand + wind + investable base)"],
        ["stoch", "include_stochastics + branch-varying wind + t0 branches "
                  "+ stochastic-group storage (stochastic_invest_method=none)"],
        ["freeops", "non_anticipativity_periods = [] (operations free "
                    "from t0)"],
    ]
    spec["scenarios"] = [
        ["free_ops", False,
         "non_anticipativity_periods = [] -> operations branch freely; "
         "shared invest, divergent v_state"],
        ["pinned_ops", False,
         "non_anticipativity_periods UNSET -> legacy window; storage NA "
         "pins v_state equal across branches (byte-parity guard)"],
    ]
    spec["scenario_alternatives"] = [
        ["free_ops", "init", "base"],
        ["free_ops", "base", "stoch"],
        ["free_ops", "stoch", "freeops"],
        ["free_ops", "freeops", None],
        ["pinned_ops", "init", "base"],
        ["pinned_ops", "base", "stoch"],
        ["pinned_ops", "stoch", None],
    ]

    spec["entities"] = [
        ["group", "stoch_g", None],
        ["model", "flextool", None],
        ["node", "elec", None],
        ["node", "resv", None],
        ["profile", "wprof", None],
        ["solve", "stoch_2p_free", None],
        ["timeline", "y6h", None],
        ["timeset", "ts1", None],
        ["timeset", "ts2", None],
        ["unit", "base", None],
        ["unit", "wind", None],
        ["unit", "charge", None],
        ["unit", "discharge", None],
        ["group__unit", ["stoch_g", "wind"], None],
        ["group__node", ["stoch_g", "resv"], None],
        ["unit__outputNode", ["base", "elec"], None],
        ["unit__outputNode", ["wind", "elec"], None],
        ["unit__inputNode", ["charge", "elec"], None],
        ["unit__outputNode", ["charge", "resv"], None],
        ["unit__inputNode", ["discharge", "resv"], None],
        ["unit__outputNode", ["discharge", "elec"], None],
        ["unit__node__profile", ["wind", "elec", "wprof"], None],
    ]
    spec["entity_alternatives"] = [
        ["node", ["elec"], "base", True],
        ["unit", ["base"], "base", True],
        ["unit", ["wind"], "base", True],
        ["node", ["resv"], "stoch", True],
        ["unit", ["charge"], "stoch", True],
        ["unit", ["discharge"], "stoch", True],
    ]

    # ---- init: model / solve / time structure --------------------------
    # t0 branching: all three branches declared at p2035 / t0001.  ``mid``
    # is realized (yes, weight 2); ``up`` / ``low`` weight 1 each.
    stochastic_branches = _map([
        ("p2035", _nested([
            ("mid", _nested([("t0001", _nested([("yes", 2.0)]))])),
            ("up", _nested([("t0001", _nested([("no", 1.0)]))])),
            ("low", _nested([("t0001", _nested([("no", 1.0)]))])),
        ])),
    ])

    # Deterministic profile (base alternative) — matches the realized
    # ``mid`` branch: flat 0.10 (wind 10, a steady 2 MW deficit / step).
    profile_base = _map([(t, 0.10) for t in STEPS])

    # Branch-varying wind profile (stoch alternative).  Keyed
    # branch -> period-first-step -> per-step share of the 100 MW cap.
    # t0-branching, so BOTH periods are per-branch — declare shares for
    # every period's first step.  ``up`` = surplus early / deficit late;
    # ``low`` = the mirror; ``mid`` = flat (byte-parity trunk).
    def _branch_profile(early: float, late: float) -> dict:
        return _nested([
            ("t0001", _nested([(t, early) for t in P1_STEPS])),
            ("t0004", _nested([(t, late) for t in P2_STEPS])),
        ])

    profile_stoch = _map([
        ("mid", _branch_profile(0.10, 0.10)),
        ("up", _branch_profile(0.20, 0.00)),
        ("low", _branch_profile(0.00, 0.20)),
    ])

    # Deterministic elec demand: 12 MW at every step.
    inflow = _map([(t, -12.0) for t in STEPS])

    spec["parameter_values"] = [
        ["model", "flextool", "solves",
         _pack({"value_type": "str", "data": ["stoch_2p_free"]}, "array"),
         "init"],
        ["timeline", "y6h", "timestep_duration",
         _pack(_map([(t, 1.0) for t in STEPS]), "map"), "init"],
        ["timeset", "ts1", "timeline", _pack("y6h", "str"), "init"],
        ["timeset", "ts1", "timeset_duration",
         _pack(_map([(t, 1.0) for t in P1_STEPS]), "map"), "init"],
        ["timeset", "ts2", "timeline", _pack("y6h", "str"), "init"],
        ["timeset", "ts2", "timeset_duration",
         _pack(_map([(t, 1.0) for t in P2_STEPS]), "map"), "init"],
        ["solve", "stoch_2p_free", "period_timeset",
         _pack(_map([("p2035", "ts1"), ("p2040", "ts2")]), "map"), "init"],
        ["solve", "stoch_2p_free", "realized_periods",
         _pack({"value_type": "str", "data": ["p2035", "p2040"]}, "array"),
         "init"],
        ["solve", "stoch_2p_free", "invest_periods",
         _pack({"value_type": "str", "data": ["p2035"]}, "array"),
         "init"],
        ["solve", "stoch_2p_free", "solve_mode",
         _pack("single_solve", "str"), "init"],
        ["solve", "stoch_2p_free", "years_represented",
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
        ["unit", "base", "efficiency", _pack(1.0, "float"), "base"],
        ["unit", "base", "existing", _pack(0.0, "float"), "base"],
        ["unit", "base", "virtual_unitsize", _pack(1.0, "float"), "base"],
        ["unit", "base", "invest_method",
         _pack("invest_total", "str"), "base"],
        ["unit", "base", "invest_cost", _pack(20.0, "float"), "base"],
        ["unit", "base", "discount_rate", _pack(0.05, "float"), "base"],
        ["unit", "base", "lifetime", _pack(1.0, "float"), "base"],
        ["unit", "base", "invest_max_total", _pack(150.0, "float"),
         "base"],
        ["unit__outputNode", ["base", "elec"], "other_operational_cost",
         _pack(5.0, "float"), "base"],
        # ---- stoch: branches + branch-varying profile + storage --------
        ["group", "stoch_g", "include_stochastics", _pack("yes", "str"),
         "stoch"],
        ["profile", "wprof", "profile", _pack(profile_stoch, "map"),
         "stoch"],
        ["solve", "stoch_2p_free", "stochastic_branches",
         _pack(stochastic_branches, "map"), "stoch"],
        ["solve", "stoch_2p_free", "stochastic_invest_method",
         _pack("none", "str"), "stoch"],
        ["node", "resv", "node_type", _pack("storage", "str"), "stoch"],
        ["node", "resv", "existing", _pack(1000.0, "float"), "stoch"],
        ["node", "resv", "storage_state_start",
         _pack(0.5, "float"), "stoch"],
        ["node", "resv", "storage_start_end_method",
         _pack("fix_start", "str"), "stoch"],
        ["node", "resv", "storage_solve_horizon_method",
         _pack("use_reference_value", "str"), "stoch"],
        ["node", "resv", "storage_state_reference_value",
         _pack(0.5, "float"), "stoch"],
        ["unit", "charge", "efficiency", _pack(1.0, "float"), "stoch"],
        ["unit", "charge", "existing", _pack(100.0, "float"), "stoch"],
        ["unit", "discharge", "efficiency", _pack(1.0, "float"), "stoch"],
        ["unit", "discharge", "existing", _pack(100.0, "float"), "stoch"],
        ["unit__outputNode", ["discharge", "elec"], "other_operational_cost",
         _pack(0.01, "float"), "stoch"],
        # ---- freeops: empty NA window (operations free from t0) --------
        ["solve", "stoch_2p_free", "non_anticipativity_periods",
         _pack({"value_type": "str", "data": []}, "array"), "freeops"],
    ]
    return spec


def main() -> None:
    spec = build()
    out = FIXTURES / "stoch_two_period_free_ops.json"
    out.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"Wrote {out} "
          f"({len(spec['parameter_values'])} parameter values)")


if __name__ == "__main__":
    main()
