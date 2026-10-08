"""Build ``tests/fixtures/stoch_realized_dispatch.json`` — the W8.1
"realized dispatch" fixture: a stochastic (fan-at-first) INVEST solve
followed by a SEPARATE deterministic realized-dispatch solve that
consumes the stochastic solve's committed capacity as FIXED existing
capacity via the invest→dispatch handoff.

This pins the W8 premise — that realized OPERATIONS can come from a
separate deterministic solve whose investments are fixed from the
stochastic solve — using ONLY the existing engine machinery (no engine
code change).  Establishing that realized ops have a home is what later
makes W4 (hedge forecasts only) safe.  Whether the realized-dispatch
pairing is author-declared (as here) or engine-automatic is a deferred
W4-era decision.

Why the realized operational data is DETERMINISTIC (the W4 boundary)
--------------------------------------------------------------------
Both solves share ONE scenario, so a parameter resolves to the SAME
value for both.  A branch-DISTINCT (3-deep branch×time_start×time Map)
profile is only foldable by a genuinely stochastic solve; the
deterministic dispatch solve has no branch scaffolding, so it drops the
3-deep Map and wind collapses to a 0 upper_limit (empirically verified:
dispatch then maxes peaker and leaves 20 MW unserved at VOLL).  Letting
the hedge forecasts diverge from the realized data the dispatch reads is
exactly the deferred W4 change ("hedge forecasts only").  W8.1 therefore
keeps the realized operational data deterministic (readable by both
solves) and proves the handoff + last-wins plumbing across a genuinely
stochastic solve and a deterministic one — which is what gives realized
ops a home and makes W4 safe later.

Shape (small, single period, hand-calc-able)
--------------------------------------------
One 3-step period ``p2035`` on an hourly timeline.

  * node ``elec`` — balance, high penalty (VOLL 10000), DETERMINISTIC
    demand 100 MW/step (``inflow`` = -100).
  * unit ``wind`` → ``elec`` — existing 100 MW, cost 0, DETERMINISTIC
    ``upper_limit`` profile 0.40 → wind supply 40 → residual 60 (read by
    both solves; shared by every forecast branch).
  * unit ``peaker`` → ``elec`` — invest_total, virtual_unitsize 1 MW,
    invest_cost 1 CUR/kW, discount_rate 0.05, lifetime 1
    (annuity 1000·CRF(0.05,1) = 1050), invest_max_total 150,
    ``other_operational_cost`` 10 CUR/MWh on the output flow.

Two solves (``model.solves`` ORDER matters — dispatch listed AFTER)
------------------------------------------------------------------
  1. ``stoch_rd`` — GENUINELY stochastic, ``solve_mode=single_solve``
     (fan-at-first).  ``stochastic_invest_method`` left at the DEFAULT
     ``none`` (shared first-stage invest — one unambiguous ``v_invest``).
     The fan is FORECASTS-ONLY: the realized (``rlz``) branch is a
     WEIGHT-0 designation (dispatch owned by the separate realized solve);
     the two forecasts carry all the mass and sum to 1 (high 0.5 /
     low 0.5).  The fan creates synthetic p2035_high / p2035_low
     periods that own LP variables; every branch carries the SAME
     (deterministic) realized operational data, so the shared first-stage
     invest covers the per-step residual 60 MW → ``v_invest[peaker] = 60``.
  2. ``stoch_rd_realized_dispatch`` — DETERMINISTIC (no
     ``stochastic_branches``, not in ``groupIncludeStochastics``), listed
     AFTER, covering the SAME realized period p2035.  No invest of its
     own; it consumes ``stoch_rd``'s invested 60 MW peaker as FIXED
     existing capacity through the handoff overlay (mirrors the proven
     ``lh2_invest → lh2_dispatch`` precedent).  Reading the deterministic
     realized data (wind 0.40 → 40, demand 100 → residual 60) it
     dispatches peaker at 60 MW.

Last-wins: both solves realize p2035/t0001-t0003; ``stoch_rd`` is
emptied of its realized window (truncated in ``_native_run_model``) so
the deterministic dispatch solve OWNS the reported realized operations.

Hand calculation (dispatch solve, assertion (d))
------------------------------------------------
Realized data: demand 100 MW/step, wind cap 0.40·100 = 40 MW/step (cost
0), peaker fills residual 60 MW/step (cost 10).  The 3-step timeset
ANNUALIZES (``timeset_duration=3`` over ``years_represented=1`` → each
step represents 8760/3 h), so peaker energy = 60 MW · 8760 h and
    operational objective = 60 MW · 8760 h · 10 CUR/MWh = 5 256 000 CUR
(the peaker's fixed existing capacity carries NO annuity in the dispatch
solve, so the dispatch objective is pure fuel).  The test asserts this
exact value at rel=1e-9; the scaling-free physical split (peaker 60 MW,
wind 40 MW per step) is asserted separately.

Run:  ~/venv-spi/bin/python tests/fixtures/build_stoch_realized_dispatch.py
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent

STEPS = [f"t{i:04d}" for i in range(1, 4)]


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
        ["init", "Model, solves and time structure (two-solve chain)"],
        ["base", "System data (peaker invest + wind + elec demand)"],
        ["stoch", "Stochastic branches + branch-varying wind profile"],
    ]
    spec["scenarios"] = [
        ["w8_realized_dispatch", False,
         "Stochastic invest → deterministic realized-dispatch handoff"],
    ]
    spec["scenario_alternatives"] = [
        ["w8_realized_dispatch", "init", "base"],
        ["w8_realized_dispatch", "base", "stoch"],
        ["w8_realized_dispatch", "stoch", None],
    ]

    spec["entities"] = [
        ["group", "stoch_g", None],
        ["model", "flextool", None],
        ["node", "elec", None],
        ["profile", "wprof", None],
        ["solve", "stoch_rd", None],
        ["solve", "stoch_rd_realized_dispatch", None],
        ["timeline", "y3h", None],
        ["timeset", "ts1", None],
        ["unit", "peaker", None],
        ["unit", "wind", None],
        ["group__unit", ["stoch_g", "wind"], None],
        ["unit__outputNode", ["peaker", "elec"], None],
        ["unit__outputNode", ["wind", "elec"], None],
        ["unit__node__profile", ["wind", "elec", "wprof"], None],
    ]
    # node / unit classes are active_by_default=False.
    spec["entity_alternatives"] = [
        ["node", ["elec"], "init", True],
        ["unit", ["peaker"], "init", True],
        ["unit", ["wind"], "init", True],
    ]

    # ---- init: model / solves / time structure ------------------------
    # Single-period stochastic fan declared at p2035/t0001.  Under the
    # corrected forecasts-only model the realized (``rlz``) branch is a
    # WEIGHT-0 designation pulled OUT of the weighted fan (its dispatch is
    # owned by the separate realized-dispatch solve); the two FORECASTS
    # carry all the probability mass and sum to 1 ON THEIR OWN
    # (high 0.5 / low 0.5).  This matches the weights the engine actually
    # applies: the phantom rule already zeroes the realized trunk and
    # renormalises the forecasts among themselves, so declaring rlz=0 and
    # high=low=0.5 is behaviour-preserving (the applied pd/pdt weights are
    # identical to the old rlz 0.25 / high 0.375 / low 0.375 encoding — the
    # forecast RATIO 1:1 and the zeroed realized trunk are unchanged) while
    # self-documenting the corrected model.
    stochastic_branches = _map([
        ("p2035", _nested([
            ("rlz", _nested([("t0001", _nested([("yes", 0.0)]))])),
            ("high", _nested([("t0001", _nested([("no", 0.5)]))])),
            ("low", _nested([("t0001", _nested([("no", 0.5)]))])),
        ])),
    ])
    # DETERMINISTIC realized operational data (base alternative), read by
    # BOTH solves: wind upper_limit 0.40 → wind supply 40 → peaker
    # residual 60.  The forecast branches below carry the SAME realized
    # operational data — they are a structurally genuine fan (synthetic
    # p2035_high / p2035_low periods owning LP variables) but share the
    # realized operational data, which is exactly the pre-W4 state.
    # Authoring branch-DISTINCT operational data (a 3-deep
    # branch×time_start×time Map) is NOT readable by the deterministic
    # dispatch solve: it shares the scenario, is not genuinely
    # stochastic, has no branch scaffolding, so the 3-deep profile is
    # dropped and wind collapses to a 0 upper_limit (verified).  Letting
    # the hedge forecasts diverge from the realized data that the
    # dispatch reads is precisely the deferred W4 change.
    profile_base = _map([(t, 0.40) for t in STEPS])
    # Deterministic demand: 100 MW every step.
    inflow = _map([(t, -100.0) for t in STEPS])

    spec["parameter_values"] = [
        ["model", "flextool", "solves",
         _pack({"value_type": "str",
                "data": ["stoch_rd", "stoch_rd_realized_dispatch"]}, "array"),
         "init"],
        ["timeline", "y3h", "timestep_duration",
         _pack(_map([(t, 1.0) for t in STEPS]), "map"), "init"],
        ["timeset", "ts1", "timeline", _pack("y3h", "str"), "init"],
        ["timeset", "ts1", "timeset_duration",
         _pack(_map([("t0001", 3.0)]), "map"), "init"],
        # ---- solve 1: stoch_rd (genuinely stochastic invest) ----------
        ["solve", "stoch_rd", "period_timeset",
         _pack(_map([("p2035", "ts1")]), "map"), "init"],
        ["solve", "stoch_rd", "realized_periods",
         _pack({"value_type": "str", "data": ["p2035"]}, "array"), "init"],
        ["solve", "stoch_rd", "invest_periods",
         _pack({"value_type": "str", "data": ["p2035"]}, "array"), "init"],
        ["solve", "stoch_rd", "realized_invest_periods",
         _pack({"value_type": "str", "data": ["p2035"]}, "array"), "init"],
        ["solve", "stoch_rd", "solve_mode",
         _pack("single_solve", "str"), "init"],
        ["solve", "stoch_rd", "years_represented",
         _pack(_map([("p2035", 1.0)]), "map"), "init"],
        # ---- solve 2: stoch_rd_realized_dispatch (deterministic) ------
        # No invest_periods ⇒ monolithic dispatch, no v_invest of its own.
        # No stochastic_branches ⇒ is_genuinely_stochastic False ⇒ reads
        # the realized (base) data.  Listed AFTER ⇒ owns realized p2035.
        ["solve", "stoch_rd_realized_dispatch", "period_timeset",
         _pack(_map([("p2035", "ts1")]), "map"), "init"],
        ["solve", "stoch_rd_realized_dispatch", "realized_periods",
         _pack({"value_type": "str", "data": ["p2035"]}, "array"), "init"],
        ["solve", "stoch_rd_realized_dispatch", "solve_mode",
         _pack("single_solve", "str"), "init"],
        ["solve", "stoch_rd_realized_dispatch", "years_represented",
         _pack(_map([("p2035", 1.0)]), "map"), "init"],
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
         _pack(10.0, "float"), "base"],
        # ---- stoch: genuine stochastic fan (shared realized data) ------
        # include_stochastics + stochastic_branches make ``stoch_rd``
        # genuinely stochastic (synthetic p2035_high / p2035_low periods
        # own LP variables).  The branch-varying operational data is
        # deliberately NOT overridden here (see profile_base) so the
        # realized operational data stays deterministic and readable by
        # the deterministic realized-dispatch solve.
        ["group", "stoch_g", "include_stochastics", _pack("yes", "str"),
         "stoch"],
        ["solve", "stoch_rd", "stochastic_branches",
         _pack(stochastic_branches, "map"), "stoch"],
    ]
    return spec


def main() -> None:
    spec = build()
    out = FIXTURES / "stoch_realized_dispatch.json"
    out.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"Wrote {out} "
          f"({len(spec['parameter_values'])} parameter values)")


if __name__ == "__main__":
    main()
