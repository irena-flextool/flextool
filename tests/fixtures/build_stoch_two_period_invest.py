"""Build ``tests/fixtures/stoch_two_period_invest.json`` — the Slice D
per-scenario (recourse) stochastic-investment gate fixture, rebuilt to a
MID-HORIZON (Option B "realized-not-past-reveal") structure.

Option B forbids fan-at-first-with-realized: a realized branch needs a
COMMITTED (pre-reveal) region.  This fixture therefore has THREE 3-step
periods — ``p2030`` (committed), ``p2035`` (reveal), ``p2040``
(continuation) — and declares the branches at ``p2035``/t0004 (a
mid-horizon reveal), so:

  * ``p2030`` is committed (realized only, weight 1.0 — a cohort-of-one);
  * ``p2035`` is the reveal — the realized branch ``rlz`` keeps the real
    name and fans into forecasts ``low`` / ``high`` (synthetic
    ``p2035_low`` / ``p2035_high``).  Reveal cohort {p2035, p2035_low,
    p2035_high} weights {0.2, 0.6, 0.2} (inputs rlz 1 / low 3 / high 1);
  * ``p2040`` is a continuation — under Option B the realized trunk has
    STOPPED, so p2040 carries ONLY the forecast copies ``p2040_low`` /
    ``p2040_high`` (the realized copy ``p2040_rlz`` is metadata-only).
    The post-reveal cohort {p2040_low, p2040_high} renormalizes to
    {0.75, 0.25} with NO realized term.

System (supply-side branch variation through ``group__unit`` — no NODE in
the stochastic group, so the LP stays separable per leaf and hand-calc-
able, mirroring the original §15.1 mechanism note):

  * node ``elec`` — balance, high penalty (VOLL), DETERMINISTIC demand
    100 MW (p2030, p2035) / 130 MW (p2040).
  * unit ``wind`` → ``elec`` — existing 100 MW, cost 0, ``upper_limit``
    profile branch-varying via the stochastic group.  Wind supply (share
    × 100) carves the residual the peaker must cover:
      - committed p2030: 0.50 → wind 50 → residual 50 (both periods, all
        branches share the committed value).
      - reveal p2035:   rlz 0.40 → residual 60;  low 0.20 → residual 80;
        high 0.30 → residual 70.
      - cont.  p2040:                            low 0.0833… → residual
        120; high 0.2308 → residual 100.  (Authored directly as residual-
        implied shares; see the per-step shares below.)
  * unit ``peaker`` → ``elec`` — variable cost 0, virtual_unitsize 1 MW,
    invest_method=invest_total (cumulative capacity, reinvest-alive
    forward), invest_cost 1 CUR/kW, discount_rate 0.05, lifetime 1
    (→ annuity per MW-period = 1000·CRF(0.05,1) = 1050 exactly),
    invest_max_total 500 (non-binding per path).

Because capacity is cumulative and alive forward on each scenario leaf,
each period's INCREMENTAL invest is residual(this period) −
capacity-so-far on the leaf:

    committed  I[p2030]      = 50             (from 0)
    reveal     I[p2035]      = 60 − 50 = 10   (rlz continuation of trunk)
               I[p2035_low]  = 80 − 50 = 30
               I[p2035_high] = 70 − 50 = 20
    cont.      I[p2040_low]  = 120 − (50+30) = 40
               I[p2040_high] = 100 − (50+20) = 30

Annuity (NPV) forward-reachability windows per scenario leaf (× 1050):
leaves are __realized = [p2030, p2035] (committed + reveal; realized
stops at the reveal), low = [p2030, p2035_low, p2040_low], high =
[p2030, p2035_high, p2040_high].  The shared committed trunk p2030 is
charged its annuity ONCE per CALENDAR period it reaches across leaves
(the ``dd_same_scenario_annuity`` dedup) = {p2030, p2035, p2040} →
window 3 = 3150.  Every other invest reaches forward only to the end of
its own leaf:

    K[p2030]      = 3·1050 = 3150   (committed, all calendar periods)
    K[p2035]      = 1·1050 = 1050   (reveal; realized leaf ends here)
    K[p2035_low]  = 2·1050 = 2100   (p2035_low → p2040_low)
    K[p2035_high] = 2·1050 = 2100
    K[p2040_low]  = 1·1050 = 1050
    K[p2040_high] = 1·1050 = 1050

Expected EXPECTATION objective (per-period branch weight w_d):

    0.2·? no — the objective weights each invest by ITS period's weight:
      p2030       : 50 ·3150·1.00 = 157 500
      p2035 (rlz) : 10 ·1050·0.20 =   2 100
      p2035_low   : 30 ·2100·0.60 =  37 800
      p2035_high  : 20 ·2100·0.20 =   8 400
      p2040_low   : 40 ·1050·0.75 =  31 500
      p2040_high  : 30 ·1050·0.25 =   7 875
    objective = 245 175.

Committed (realized-period) invest cost — I-a, probability-weighted by
``pd_branch_weight`` over the realized periods {p2030, p2035}:

    0.20 reveal weight on p2035, 1.00 on committed p2030:
      p2030 : 50·3150·1.00 = 157 500
      p2035 : 10·1050·0.20 =   2 100
    invest_realized = 159 600  (0.1596 M CUR).

These are DERIVED values; the asserts in
``tests/test_recourse_cost_reconciliation.py`` are confirmed against the
actual solve.

Run:  ~/venv-spi/bin/python tests/fixtures/build_stoch_two_period_invest.py
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
        ["stoch", "Mid-horizon stochastic branches + branch wind profile"],
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
        ["timeline", "y9h", None],
        ["timeset", "ts0", None],
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
    # Mid-horizon reveal: branches START at p2035/t0004, so p2030 is a
    # pre-reveal COMMITTED period.
    stochastic_branches = _map([
        ("p2035", _nested([
            ("rlz", _nested([("t0004", _nested([("yes", 1.0)]))])),
            ("low", _nested([("t0004", _nested([("no", 3.0)]))])),
            ("high", _nested([("t0004", _nested([("no", 1.0)]))])),
        ])),
    ])
    # Deterministic profile (base alternative) — matches the rlz branch on
    # the committed + reveal periods (0.50 committed, 0.40 reveal); p2040
    # uses the rlz-implied 0.0 (no realized continuation, so this value is
    # only a placeholder for the deterministic control).
    profile_base = _map(
        [(t, 0.50) for t in P0_STEPS]
        + [(t, 0.40) for t in P1_STEPS]
        + [(t, 0.0) for t in P2_STEPS]
    )
    # Branch-varying wind profile (stoch alternative) — 3d map
    # branch → period-first-step → per-step share of the 100 MW wind cap.
    # Committed p2030 (t0001) is SHARED (0.50 in every branch; pre-reveal
    # data must be identical — only the realized/trunk value is read).
    # Residual (= 100/130 demand − wind) drives the per-period invest.
    #   rlz : p2030 0.50 (res 50), p2035 0.40 (res 60)  [stops at reveal]
    #   low : p2035 0.20 (res 80), p2040 0.0833… → wind 8.33 res 120? use
    #         0.10 → wind 10, res 120.
    #   high: p2035 0.30 (res 70), p2040 0.30 → wind 30, res 100.
    profile_stoch = _map([
        ("rlz", _nested([
            ("t0001", _nested([(t, 0.50) for t in P0_STEPS])),
            ("t0004", _nested([(t, 0.40) for t in P1_STEPS])),
        ])),
        ("low", _nested([
            ("t0001", _nested([(t, 0.50) for t in P0_STEPS])),
            ("t0004", _nested([(t, 0.20) for t in P1_STEPS])),
            ("t0007", _nested([(t, 0.10) for t in P2_STEPS])),
        ])),
        ("high", _nested([
            ("t0001", _nested([(t, 0.50) for t in P0_STEPS])),
            ("t0004", _nested([(t, 0.30) for t in P1_STEPS])),
            ("t0007", _nested([(t, 0.30) for t in P2_STEPS])),
        ])),
    ])
    # Deterministic demand: 100 MW at p2030/p2035 steps, 130 MW at p2040.
    inflow = _map(
        [(t, -100.0) for t in P0_STEPS]
        + [(t, -100.0) for t in P1_STEPS]
        + [(t, -130.0) for t in P2_STEPS]
    )

    spec["parameter_values"] = [
        ["model", "flextool", "solves",
         _pack({"value_type": "str", "data": ["stoch_2p_inv"]}, "array"),
         "init"],
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
        ["solve", "stoch_2p_inv", "period_timeset",
         _pack(_map([("p2030", "ts0"), ("p2035", "ts1"),
                     ("p2040", "ts2")]), "map"), "init"],
        ["solve", "stoch_2p_inv", "realized_periods",
         _pack({"value_type": "str",
                "data": ["p2030", "p2035", "p2040"]}, "array"),
         "init"],
        ["solve", "stoch_2p_inv", "invest_periods",
         _pack({"value_type": "str",
                "data": ["p2030", "p2035", "p2040"]}, "array"),
         "init"],
        ["solve", "stoch_2p_inv", "realized_invest_periods",
         _pack({"value_type": "str",
                "data": ["p2030", "p2035", "p2040"]}, "array"),
         "init"],
        ["solve", "stoch_2p_inv", "solve_mode",
         _pack("single_solve", "str"), "init"],
        ["solve", "stoch_2p_inv", "years_represented",
         _pack(_map([("p2030", 1.0), ("p2035", 1.0), ("p2040", 1.0)]),
               "map"), "init"],
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
        ["unit", "peaker", "invest_max_total", _pack(500.0, "float"),
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
