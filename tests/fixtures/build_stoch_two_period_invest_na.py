"""Build ``tests/fixtures/stoch_two_period_invest_na.json`` — the Slice D
variant-B (cross-scenario coupling) fixture (design
``specs/sliceD_recourse_invest_design.md`` §15.6).

Variant A (``stoch_two_period_invest``) authors NO stochastic *node*, so
its LP is separable per leaf (which makes the hand-calc tractable).
Variant B exercises the coupling path the feature must NOT be gated away
from: a stochastic-group NODE fires the dispatch non-anticipativity
net-charge pinning (``non_anticipativity_storage_use``, gated on
``group_node`` × ``groupStochastic``), so the flag-on LP is genuinely
NOT separable.

Rather than author a storage node + branch machinery from scratch, this
builder EXTENDS the proven ``stochastics.json`` shape (the
``2_day_stochastic_dispatch`` scenario, which already fires
``non_anticipativity_storage_use`` with a populated
``dt_non_anticipativity`` — see
``tests/engine_polars/emission/test_non_anticipativity_storage_use_emits.py``)
with a new ``recourse_na`` scenario that:

  * turns on ``solve.stochastic_invest_method = recourse`` for
    ``2day_dispatch``;
  * gives the solve ``invest_periods`` / ``realized_invest_periods`` =
    ``[period1]`` so the invest axis fans across the period-1 branches;
  * makes the stochastic-group storage node ``hydro_reservoir``
    INVESTABLE (``invest_method = invest_total``), so its node-side
    objective annuity ``annu_n`` (the §9.1 #5 regression hook) is
    exercised and — under recourse — probability-weighted by
    ``pd_branch_weight[d]``.

Assertions (structural + directional, no second hand-calc) live in
``tests/engine_polars/test_recourse_invest_na.py``:
  * ``dt_non_anticipativity`` non-empty; ``non_anticipativity_storage_use``
    row count > 0 — the coupling is live.
  * branch-leaf ``v_state`` at the NA-tied timesteps equals the realized
    value (the flag-on LP is not separable on this fixture).
  * the objective coefficient of ``v_invest_n[hydro_reservoir, d]`` equals
    ``unitsize × annu_n[d] × pd_branch_weight[d]`` (the D4 analog for the
    node-side Param).
  * feasible / optimal; ``check_recourse_npv_preconditions`` → ``[]``.

Run:  ~/venv-spi/bin/python tests/fixtures/build_stoch_two_period_invest_na.py
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent


def _pack(value: object, value_type: str) -> list:
    return [
        base64.b64encode(json.dumps(value).encode()).decode("ascii"),
        value_type,
    ]


def build() -> dict:
    spec = json.loads((FIXTURES / "stochastics.json").read_text())

    # ``stochastics.json`` predates v70, so inject the
    # ``solve.stochastic_invest_method`` opt-in (value list + definition)
    # before import — ``json_to_db`` validates authored values against the
    # embedded schema BEFORE the idempotent v70 migration reconciles it.
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

    # New alternative + scenario layered on the 2_day_stochastic_dispatch
    # chain (base → init → system → 2day → recourse_na).
    spec["alternatives"].append(
        ["recourse_na", "Recourse invest on the stochastic storage node"])
    spec["scenarios"].append(
        ["recourse_na", False,
         "Variant B — cross-scenario coupling (stochastic storage node)"])
    spec["scenario_alternatives"] += [
        ["recourse_na", "base", "init"],
        ["recourse_na", "init", "system"],
        ["recourse_na", "system", "2day"],
        ["recourse_na", "2day", "recourse_na"],
        ["recourse_na", "recourse_na", None],
    ]

    # Recourse opt-in + invest axis + an investable stochastic storage
    # node.  ``hydro_reservoir`` is already a ``storage`` node in the
    # ``add_stochastics`` group (system alternative), so making it
    # investable exercises the node-side ``annu_n`` under the fanned
    # invest axis while the storage NA keeps the leaves coupled.
    spec["parameter_values"] += [
        ["solve", "2day_dispatch", "stochastic_invest_method",
         _pack("recourse", "str"), "recourse_na"],
        ["solve", "2day_dispatch", "invest_periods",
         _pack({"value_type": "str", "data": ["period1"]}, "array"),
         "recourse_na"],
        ["solve", "2day_dispatch", "realized_invest_periods",
         _pack({"value_type": "str", "data": ["period1"]}, "array"),
         "recourse_na"],
        ["solve", "2day_dispatch", "years_represented",
         _pack({"index_type": "str", "data": [["period1", 1.0]]}, "map"),
         "recourse_na"],
        ["node", "hydro_reservoir", "invest_method",
         _pack("invest_total", "str"), "recourse_na"],
        ["node", "hydro_reservoir", "invest_cost",
         _pack(1.0, "float"), "recourse_na"],
        ["node", "hydro_reservoir", "discount_rate",
         _pack(0.05, "float"), "recourse_na"],
        ["node", "hydro_reservoir", "lifetime",
         _pack(1.0, "float"), "recourse_na"],
        ["node", "hydro_reservoir", "invest_max_total",
         _pack(100000.0, "float"), "recourse_na"],
    ]
    return spec


def main() -> None:
    spec = build()
    out = FIXTURES / "stoch_two_period_invest_na.json"
    out.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"Wrote {out} "
          f"({len(spec['parameter_values'])} parameter values)")


if __name__ == "__main__":
    main()
