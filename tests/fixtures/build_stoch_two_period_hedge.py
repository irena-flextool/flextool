"""Build ``tests/fixtures/stoch_two_period_hedge.json`` — the Slice E
hedged two-stage (mid-horizon reveal) fixture (design
``specs/sliceE_invest_na_design.md`` §8).

Two 3-step periods ``p2035`` (t0001-t0003) / ``p2040`` (t0004-t0006).  The
lever that makes hedging valuable: a period-Map ``invest_cost``
``{p2035: 1, p2040: 3}`` so the FIRST-STAGE (p2035) capacity is cheaper per
delivered p2040-MW than the recourse (p2040) capacity — under
``reinvest_automatic`` p2035 capacity is alive at p2040 (unbounded-forward
windows), so pre-positioning under uncertainty pays off.

Two scenarios drive the WS < RP < EEV proof on IDENTICAL scenario data,
differing ONLY in WHERE the branches are declared:

  * ``hedge``     — reveal at **p2040** (mid-horizon).  ``v_invest[base,
    p2035]`` is a single SHARED first-stage decision (Option B: the shared
    trunk, NA by construction).  Hand-calc objective **173 250** (RP).
  * ``hedge_ws``  — reveal at **p2035** (fan-at-first-step, the Slice-D
    wait-and-see topology).  ``v_invest`` is scenario-specific; each
    scenario builds all-first-stage.  Hand-calc objective **157 500** (WS).

``RP − WS = 15 750 = EVPI > 0`` — a direct behavioural proof that the
mid-horizon first stage is genuinely shared (§8.3).

Effective demand seen by the investable ``base`` unit (design §8.1: p2035 =
40 shared; p2040 realized/high = 120, low = 60) is produced supply-side —
NO stochastic NODE, so the LP stays separable / storage-free and the
hand-calc is tractable (mirrors ``build_stoch_two_period_invest`` §8's
mechanism note).  A free existing ``wind`` unit with a branch-varying
``upper_limit`` profile carves the residual that ``base`` must cover:

  * node ``elec`` demand = 120 MW at every step (deterministic inflow).
  * wind existing 100 MW, cost 0; profile (share of 100 MW):
      - p2035 (pre-reveal, SHARED across branches): 0.8 → wind 80 →
        base residual 40 (= D1).
      - p2040 realized (rlz):  0.0 → wind  0 → base residual 120 (= D2 high).
      - p2040 low:             0.6 → wind 60 → base residual  60 (= D2 low).
  * unit ``base`` → ``elec`` — the investable peaker (virtual_unitsize 1,
    invest_total, discount_rate 0.05, lifetime 1 → annuity per MW-period
    = cost·1000·CRF(0.05,1) = cost·1050; unbounded-forward windows so the
    p2035 anchor's window is {p2035, p2040} → K1 = 2·1050 = 2100 and the
    p2040 anchor's window is {p2040} → K2 = 1·1050 = 3150),
    ``invest_max_total`` 150 (non-binding; the per-path cap witness §8.3).

**No inflation / no solve-level discount** (design §8.1 F3): the exact
173 250 golden requires ``inv_factor ≡ 1.0``, i.e. the SOLVE-level
inflation scalar = 0.  This builder authors NO ``model.inflation_rate``,
NO ``inflation_offset_investment/operations``, and NO solve-level
``discount_rate`` — all left at their 0.0 default.  The entity
``discount_rate = 0.05`` on ``base`` drives ONLY the CRF (the annuity), a
distinct knob.

Hand calculation (design §8.2), ``x = v_invest[base, p2035]``,
``r_s = v_invest[base, p2040_s]``, weights ``w_rlz = 0.25`` (D2 = 120),
``w_low = 0.75`` (D2 = 60):

    RP  = min_x 2100·x + Σ_s w_s·3150·(D2_s − x)^+
        → x* = 60 (hedge between 60 and 120, above D1 = 40)
        → 2100·60 + 0.25·3150·60 = 126 000 + 47 250 = 173 250.
    WS  = 0.75·(2100·60) + 0.25·(2100·120) = 94 500 + 63 000 = 157 500.

Run:  ~/venv-spi/bin/python tests/fixtures/build_stoch_two_period_hedge.py
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

    # ``stochastics.json`` still carries the pre-v70 schema, so inject the
    # ``solve.stochastic_invest_method`` opt-in (Slice C, v70) before
    # import (json_to_db validates against the embedded schema BEFORE the
    # idempotent v70 migration reconciles it).
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
        ["base", "System data (base invest + wind + elec demand)"],
        ["stoch", "include_stochastics + branch-varying wind + recourse"],
        ["rp", "stochastic_branches at p2040 (mid-horizon reveal)"],
        ["ws", "stochastic_branches at p2035 (wait-and-see / fan-at-first)"],
    ]
    spec["scenarios"] = [
        ["hedge", False, "Mid-horizon reveal at p2040 (RP = 173 250)"],
        ["hedge_ws", False,
         "Wait-and-see fan-at-p2035 control (WS = 157 500)"],
    ]
    spec["scenario_alternatives"] = [
        ["hedge", "init", "base"],
        ["hedge", "base", "stoch"],
        ["hedge", "stoch", "rp"],
        ["hedge", "rp", None],
        ["hedge_ws", "init", "base"],
        ["hedge_ws", "base", "stoch"],
        ["hedge_ws", "stoch", "ws"],
        ["hedge_ws", "ws", None],
    ]

    spec["entities"] = [
        ["group", "stoch_g", None],
        ["model", "flextool", None],
        ["node", "elec", None],
        ["profile", "wprof", None],
        ["solve", "stoch_2p_hedge", None],
        ["timeline", "y6h", None],
        ["timeset", "ts1", None],
        ["timeset", "ts2", None],
        ["unit", "base", None],
        ["unit", "wind", None],
        ["group__unit", ["stoch_g", "wind"], None],
        ["unit__outputNode", ["base", "elec"], None],
        ["unit__outputNode", ["wind", "elec"], None],
        ["unit__node__profile", ["wind", "elec", "wprof"], None],
    ]
    spec["entity_alternatives"] = [
        ["node", ["elec"], "init", True],
        ["unit", ["base"], "init", True],
        ["unit", ["wind"], "init", True],
    ]

    # ---- init: model / solve / time structure --------------------------
    # Mid-horizon reveal (scenario ``hedge``): branches declared at p2040.
    stochastic_branches_rp = _map([
        ("p2040", _nested([
            ("rlz", _nested([("t0004", _nested([("yes", 1.0)]))])),
            ("low", _nested([("t0004", _nested([("no", 3.0)]))])),
        ])),
    ])
    # Wait-and-see (scenario ``hedge_ws``): branches declared at p2035.
    stochastic_branches_ws = _map([
        ("p2035", _nested([
            ("rlz", _nested([("t0001", _nested([("yes", 1.0)]))])),
            ("low", _nested([("t0001", _nested([("no", 3.0)]))])),
        ])),
    ])
    # Deterministic profile (base alternative) — matches the rlz branch:
    # 0.8 at p2035 (wind 80 → residual 40), 0.0 at p2040 (residual 120).
    profile_base = _map(
        [(t, 0.80) for t in P1_STEPS] + [(t, 0.0) for t in P2_STEPS]
    )
    # Branch-varying wind profile (stoch alternative).  Keyed
    # branch → period-first-step → per-step share of the 100 MW wind cap.
    # p2035 (t0001) is SHARED (0.8 in BOTH branches — pre-reveal data must
    # be identical; under mid-horizon only the realized/trunk value is
    # ever read).  p2040 (t0004): rlz 0.0 (residual 120), low 0.6
    # (residual 60).
    profile_stoch = _map([
        ("rlz", _nested([
            ("t0001", _nested([(t, 0.80) for t in P1_STEPS])),
            ("t0004", _nested([(t, 0.0) for t in P2_STEPS])),
        ])),
        ("low", _nested([
            ("t0001", _nested([(t, 0.80) for t in P1_STEPS])),
            ("t0004", _nested([(t, 0.60) for t in P2_STEPS])),
        ])),
    ])
    # Deterministic demand: 120 MW at every step (both periods).
    inflow = _map([(t, -120.0) for t in STEPS])
    # Period-Map invest_cost: first stage cheap (1), recourse dear (3).
    invest_cost = _map([("p2035", 1.0), ("p2040", 3.0)])

    spec["parameter_values"] = [
        ["model", "flextool", "solves",
         _pack({"value_type": "str", "data": ["stoch_2p_hedge"]}, "array"),
         "init"],
        ["timeline", "y6h", "timestep_duration",
         _pack(_map([(t, 1.0) for t in STEPS]), "map"), "init"],
        ["timeset", "ts1", "timeline", _pack("y6h", "str"), "init"],
        ["timeset", "ts1", "timeset_duration",
         _pack(_map([("t0001", 3.0)]), "map"), "init"],
        ["timeset", "ts2", "timeline", _pack("y6h", "str"), "init"],
        ["timeset", "ts2", "timeset_duration",
         _pack(_map([("t0004", 3.0)]), "map"), "init"],
        ["solve", "stoch_2p_hedge", "period_timeset",
         _pack(_map([("p2035", "ts1"), ("p2040", "ts2")]), "map"), "init"],
        ["solve", "stoch_2p_hedge", "realized_periods",
         _pack({"value_type": "str", "data": ["p2035", "p2040"]}, "array"),
         "init"],
        ["solve", "stoch_2p_hedge", "invest_periods",
         _pack({"value_type": "str", "data": ["p2035", "p2040"]}, "array"),
         "init"],
        ["solve", "stoch_2p_hedge", "realized_invest_periods",
         _pack({"value_type": "str", "data": ["p2035", "p2040"]}, "array"),
         "init"],
        ["solve", "stoch_2p_hedge", "solve_mode",
         _pack("single_solve", "str"), "init"],
        ["solve", "stoch_2p_hedge", "years_represented",
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
        ["unit", "base", "invest_cost", _pack(invest_cost, "map"), "base"],
        ["unit", "base", "discount_rate", _pack(0.05, "float"), "base"],
        ["unit", "base", "lifetime", _pack(1.0, "float"), "base"],
        ["unit", "base", "invest_max_total", _pack(150.0, "float"),
         "base"],
        ["unit__outputNode", ["base", "elec"], "other_operational_cost",
         _pack(0.0, "float"), "base"],
        # ---- stoch: branches + branch-varying profile + recourse -------
        ["group", "stoch_g", "include_stochastics", _pack("yes", "str"),
         "stoch"],
        ["profile", "wprof", "profile", _pack(profile_stoch, "map"),
         "stoch"],
        ["solve", "stoch_2p_hedge", "stochastic_invest_method",
         _pack("recourse", "str"), "stoch"],
        # ---- rp: mid-horizon reveal at p2040 ---------------------------
        ["solve", "stoch_2p_hedge", "stochastic_branches",
         _pack(stochastic_branches_rp, "map"), "rp"],
        # ---- ws: wait-and-see reveal at p2035 --------------------------
        ["solve", "stoch_2p_hedge", "stochastic_branches",
         _pack(stochastic_branches_ws, "map"), "ws"],
    ]
    return spec


def main() -> None:
    spec = build()
    out = FIXTURES / "stoch_two_period_hedge.json"
    out.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"Wrote {out} "
          f"({len(spec['parameter_values'])} parameter values)")


if __name__ == "__main__":
    main()
