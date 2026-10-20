"""Build ``tests/fixtures/stoch_two_period_midhorizon_nreal0.json`` — the
R-NAME witness: a mid-horizon reveal with a genuine forecasts-only fan
(n_real=0) on the DEFAULT ``stochastic_invest_method = none`` path.

Topology (one 3-step pre-reveal trunk + one 3-step fanned period):

  * Two 3-step periods ``p2035`` (t0001-t0003) / ``p2040`` (t0004-t0006).
  * Reveal at **p2040** (mid-horizon), so ``p2035`` is a LONE PRE-REVEAL
    TRUNK (not fanned) whose real (un-suffixed) names must source their
    branch-keyed wind profile from SOME branch.
  * The fan is a genuine ``{high, low}`` pair — BOTH ``realized: no`` with
    nonzero weight, so **n_real = 0**.  No branch owns the real period
    names.  Before R-NAME the pre-reveal trunk's branch-keyed wind profile
    silently collapsed to the zero fallback (no ``solve_branch__time_branch``
    owner for the real names), inflating the pre-reveal residual and forcing
    the here-and-now invest up.  R-NAME binds the pre-reveal trunk to the
    reference forecast (lexically-smallest token = ``high``) so p2035's wind
    binds nonzero.

DEFAULT invest path: ``stochastic_invest_method = none`` (single invest var
per period; operations fan stochastically across the forecasts).  This keeps
the witness clear of the recourse / lineage machinery so it exercises ONLY
the pre-reveal data-binding guard.

Numbers are chosen so the guard's effect is DIRECTLY observable on BOTH the
pre-reveal dispatch and the here-and-now investment (not just structurally):

  * The investable ``base`` carries a small positive ``other_operational_cost``
    so the free ``wind`` is STRICTLY preferred up to its profile cap (no
    dispatch degeneracy) — then ``wind``'s pre-reveal flow equals its bound
    profile iff the profile binds, and collapses to 0 under the zero fallback.
  * The pre-reveal residual is the BINDING period for the single ``none``
    invest var (``invest_cost`` cheaper at p2035; p2035's residual exceeds the
    post-reveal forecasts'), so the here-and-now invest tracks the pre-reveal
    residual — 60 when p2035's profile binds, 120 under the zero fallback.

Branch-varying wind profile (share of the 100 MW free ``wind`` cap):
  * p2035 (t0001, pre-reveal / shared): 0.6 in BOTH branches → wind 60 →
    pre-reveal residual the investable ``base`` must cover = 120 − 60 = 60.
    Under the zero fallback (guard removed) this collapses to 0 → residual 120.
  * p2040 (t0004), ``high``: 1.0 → wind 100 → residual 20.
  * p2040 (t0004), ``low`` : 0.8 → wind 80  → residual 40.
  Both post-reveal residuals (20, 40) are below the pre-reveal 60, so a single
  p2035 invest of 60 (persisting forward under the unbounded invest window)
  covers every period; p2040 invests nothing.

Elec demand = 120 MW at every step (deterministic inflow).

Run:  ~/venv-spi/bin/python \
        tests/fixtures/build_stoch_two_period_midhorizon_nreal0.py
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
        ["stoch", "include_stochastics + branch-varying wind + "
                  "mid-horizon reveal (none invest path)"],
    ]
    spec["scenarios"] = [
        ["midhorizon", False,
         "Mid-horizon reveal at p2040, genuine n_real=0 {high,low} fan, "
         "stochastic_invest_method=none (R-NAME witness)"],
    ]
    spec["scenario_alternatives"] = [
        ["midhorizon", "init", "base"],
        ["midhorizon", "base", "stoch"],
        ["midhorizon", "stoch", None],
    ]

    spec["entities"] = [
        ["group", "stoch_g", None],
        ["model", "flextool", None],
        ["node", "elec", None],
        ["profile", "wprof", None],
        ["solve", "stoch_2p_mid", None],
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
    # Mid-horizon reveal: a GENUINE forecasts-only fan declared at p2040 —
    # both ``high`` and ``low`` are ``realized: no`` with nonzero weight, so
    # n_real = 0.
    stochastic_branches = _map([
        ("p2040", _nested([
            ("high", _nested([("t0004", _nested([("no", 1.0)]))])),
            ("low", _nested([("t0004", _nested([("no", 3.0)]))])),
        ])),
    ])
    # Deterministic profile (base alternative): 0.6 at p2035 (wind 60 →
    # residual 60), 0.0 at p2040.
    profile_base = _map(
        [(t, 0.60) for t in P1_STEPS] + [(t, 0.0) for t in P2_STEPS]
    )
    # Branch-varying wind profile (stoch alternative).  Keyed
    # branch → period-first-step → per-step share of the 100 MW wind cap.
    # p2035 (t0001) is SHARED (0.6 in BOTH branches — pre-reveal data is
    # identical).  p2040 (t0004): high 1.0 (residual 20), low 0.8
    # (residual 40).
    profile_stoch = _map([
        ("high", _nested([
            ("t0001", _nested([(t, 0.60) for t in P1_STEPS])),
            ("t0004", _nested([(t, 1.0) for t in P2_STEPS])),
        ])),
        ("low", _nested([
            ("t0001", _nested([(t, 0.60) for t in P1_STEPS])),
            ("t0004", _nested([(t, 0.80) for t in P2_STEPS])),
        ])),
    ])
    # Deterministic demand: 120 MW at every step (both periods).
    inflow = _map([(t, -120.0) for t in STEPS])
    # Period-Map invest_cost: first stage cheap (1), later period dear (3).
    invest_cost = _map([("p2035", 1.0), ("p2040", 3.0)])

    spec["parameter_values"] = [
        ["model", "flextool", "solves",
         _pack({"value_type": "str", "data": ["stoch_2p_mid"]}, "array"),
         "init"],
        ["timeline", "y6h", "timestep_duration",
         _pack(_map([(t, 1.0) for t in STEPS]), "map"), "init"],
        ["timeset", "ts1", "timeline", _pack("y6h", "str"), "init"],
        ["timeset", "ts1", "timeset_duration",
         _pack(_map([("t0001", 3.0)]), "map"), "init"],
        ["timeset", "ts2", "timeline", _pack("y6h", "str"), "init"],
        ["timeset", "ts2", "timeset_duration",
         _pack(_map([("t0004", 3.0)]), "map"), "init"],
        ["solve", "stoch_2p_mid", "period_timeset",
         _pack(_map([("p2035", "ts1"), ("p2040", "ts2")]), "map"), "init"],
        ["solve", "stoch_2p_mid", "realized_periods",
         _pack({"value_type": "str", "data": ["p2035", "p2040"]}, "array"),
         "init"],
        ["solve", "stoch_2p_mid", "invest_periods",
         _pack({"value_type": "str", "data": ["p2035", "p2040"]}, "array"),
         "init"],
        ["solve", "stoch_2p_mid", "realized_invest_periods",
         _pack({"value_type": "str", "data": ["p2035", "p2040"]}, "array"),
         "init"],
        ["solve", "stoch_2p_mid", "solve_mode",
         _pack("single_solve", "str"), "init"],
        ["solve", "stoch_2p_mid", "years_represented",
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
        # Small positive dispatch cost so free wind is STRICTLY preferred up
        # to its profile cap (removes dispatch degeneracy → wind's pre-reveal
        # flow cleanly witnesses the bound profile).
        ["unit__outputNode", ["base", "elec"], "other_operational_cost",
         _pack(1.0, "float"), "base"],
        # ---- stoch: branches + branch-varying profile -----------------
        # Default invest path: stochastic_invest_method = none (single
        # invest var per period; ops fan across the forecasts).
        ["group", "stoch_g", "include_stochastics", _pack("yes", "str"),
         "stoch"],
        ["profile", "wprof", "profile", _pack(profile_stoch, "map"),
         "stoch"],
        ["solve", "stoch_2p_mid", "stochastic_invest_method",
         _pack("none", "str"), "stoch"],
        ["solve", "stoch_2p_mid", "stochastic_branches",
         _pack(stochastic_branches, "map"), "stoch"],
    ]
    return spec


def main() -> None:
    spec = build()
    out = FIXTURES / "stoch_two_period_midhorizon_nreal0.json"
    out.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"Wrote {out} "
          f"({len(spec['parameter_values'])} parameter values)")


if __name__ == "__main__":
    main()
