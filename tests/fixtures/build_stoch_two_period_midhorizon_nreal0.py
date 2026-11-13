"""Build ``tests/fixtures/stoch_two_period_midhorizon_nreal0.json`` -- a
mid-horizon reveal with a genuine forecasts-only fan (n_real=0) on the
DEFAULT ``stochastic_invest_method = none`` path, plus two variants that
witness how a period before the reveal reads stochastic data (item 250).

Topology (one 3-step pre-reveal trunk + one 3-step fanned period):

  * Two 3-step periods ``p2035`` (t0001-t0003) / ``p2040`` (t0004-t0006).
  * Reveal at **p2040** (mid-horizon), so ``p2035`` is a LONE PRE-REVEAL
    TRUNK (not fanned).
  * The fan is a genuine ``{high, low}`` pair -- BOTH ``realized: no`` with
    nonzero weight (1 : 3), so **n_real = 0** in ``midhorizon``.

Scenarios:

  * ``midhorizon`` (init, base, stoch): the forecasts-only fan with a plain
    (non-stochastic) wind profile -- 0.6 at p2035, 0.85 at p2040 in both
    forecasts.  A forecasts-only fan has no series for the periods before it,
    so its stochastic-typed data could not be read there; plain data is.
  * ``midhorizon_stoch_profile`` (+ stoch_profile): the same fan with a
    branch-varying wind profile (high / low, given at t0001 and t0004).  p2035
    would have to read a stochastic value but no realized branch provides it:
    a configuration error (FlexTool never picks one of the forecasts).
  * ``midhorizon_realized`` (+ realized): the fan gains a ``realized: yes``
    branch ``real`` whose series (0.6 at t0001, 0.9 at t0004) p2035 reads
    before the reveal; high / low as above.

Numbers (share of the 100 MW free ``wind`` cap; elec demand 120 MW at every
step, deterministic inflow):

  * The investable ``base`` carries a small positive ``other_operational_cost``
    so the free ``wind`` is STRICTLY preferred up to its profile cap (no
    dispatch degeneracy).
  * p2035: 0.6 -> wind 60 -> residual 60 for ``base``.
  * p2040: plain 0.85 -> residual 35; high 1.0 -> 20; low 0.8 -> 40.
  * ``invest_cost`` is cheaper at p2035 and every later residual is below 60,
    so the single ``none`` invest var invests 60 at p2035 and nothing at p2040.

Run:  ~/venv-spi/bin/python \
        tests/fixtures/build_stoch_two_period_midhorizon_nreal0.py
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

from flextool.update_flextool.fixture_schema import schema_sections
from flextool.update_flextool.test_fixtures import write_built_fixture

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
    spec: dict = schema_sections("packed")

    spec["alternatives"] = [
        ["init", "Model, solve and time structure"],
        ["base", "System data (base invest + wind + elec demand)"],
        ["stoch", "include_stochastics + forecasts-only mid-horizon reveal "
                  "(none invest path), plain wind profile"],
        ["stoch_profile", "Branch-varying wind profile without a realized branch "
                          "(an error before the reveal)"],
        ["realized", "Realized branch real + branch-varying wind profile"],
    ]
    spec["scenarios"] = [
        ["midhorizon", False,
         "Mid-horizon reveal at p2040, genuine n_real=0 {high,low} fan, "
         "stochastic_invest_method=none, plain wind profile"],
        ["midhorizon_stoch_profile", False,
         "midhorizon with a stochastic wind profile and no realized branch "
         "(configuration error)"],
        ["midhorizon_realized", False,
         "midhorizon with a realized branch real and a stochastic wind profile"],
    ]
    spec["scenario_alternatives"] = [
        ["midhorizon", "init", "base"],
        ["midhorizon", "base", "stoch"],
        ["midhorizon", "stoch", None],
        ["midhorizon_stoch_profile", "init", "base"],
        ["midhorizon_stoch_profile", "base", "stoch"],
        ["midhorizon_stoch_profile", "stoch", "stoch_profile"],
        ["midhorizon_stoch_profile", "stoch_profile", None],
        ["midhorizon_realized", "init", "base"],
        ["midhorizon_realized", "base", "stoch"],
        ["midhorizon_realized", "stoch", "realized"],
        ["midhorizon_realized", "realized", None],
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
    forecasts = [
        ("high", _nested([("t0004", _nested([("no", 1.0)]))])),
        ("low", _nested([("t0004", _nested([("no", 3.0)]))])),
    ]
    stochastic_branches = _map([("p2040", _nested(forecasts))])
    # The realized variant: ``real`` (realized: yes) provides the data before
    # the reveal and the realization dispatch.
    stochastic_branches_realized = _map([("p2040", _nested([
        *forecasts,
        ("real", _nested([("t0004", _nested([("yes", 0.0)]))])),
    ]))])
    # Deterministic profile (base alternative): 0.6 at p2035 (wind 60 →
    # residual 60), 0.0 at p2040.
    profile_base = _map(
        [(t, 0.60) for t in P1_STEPS] + [(t, 0.0) for t in P2_STEPS]
    )
    # Plain wind profile of the forecasts-only fan (stoch alternative): 0.6 at
    # p2035, 0.85 at p2040 in both forecasts (residuals 60 and 35).
    profile_plain = _map(
        [(t, 0.60) for t in P1_STEPS] + [(t, 0.85) for t in P2_STEPS]
    )
    # Branch-varying wind profile.  Keyed branch → analysis time → per-step
    # share of the 100 MW wind cap.  p2040 (t0004): high 1.0 (residual 20),
    # low 0.8 (residual 40), real 0.9 (residual 30).  p2035 (t0001): 0.6.
    def _branch(at_t0001: bool, p2040: float) -> dict:
        ats = [("t0001", _nested([(t, 0.60) for t in P1_STEPS]))] if at_t0001 else []
        return _nested([*ats, ("t0004", _nested([(t, p2040) for t in P2_STEPS]))])

    # Without a realized branch the forecasts carry the p2035 series too (an
    # error: no realized branch provides the data before the reveal).
    profile_stoch = _map([("high", _branch(True, 1.0)), ("low", _branch(True, 0.80))])
    # With the realized branch only ``real`` gives the p2035 series.
    profile_realized = _map([("high", _branch(False, 1.0)), ("low", _branch(False, 0.80)),
                             ("real", _branch(True, 0.90))])
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
        # ---- stoch: branches + plain profile ---------------------------
        # Default invest path: stochastic_invest_method = none (single
        # invest var per period; ops fan across the forecasts).
        ["group", "stoch_g", "include_stochastics", _pack("yes", "str"),
         "stoch"],
        ["profile", "wprof", "profile", _pack(profile_plain, "map"),
         "stoch"],
        ["solve", "stoch_2p_mid", "stochastic_invest_method",
         _pack("none", "str"), "stoch"],
        ["solve", "stoch_2p_mid", "stochastic_branches",
         _pack(stochastic_branches, "map"), "stoch"],
        # ---- stoch_profile: stochastic profile, no realized branch -----
        ["profile", "wprof", "profile", _pack(profile_stoch, "map"),
         "stoch_profile"],
        # ---- realized: realized branch + stochastic profile ------------
        ["profile", "wprof", "profile", _pack(profile_realized, "map"),
         "realized"],
        ["solve", "stoch_2p_mid", "stochastic_branches",
         _pack(stochastic_branches_realized, "map"), "realized"],
    ]
    return spec


def main() -> None:
    spec = build()
    out = FIXTURES / "stoch_two_period_midhorizon_nreal0.json"
    write_built_fixture(spec, out)
    print(f"Wrote {out} "
          f"({len(spec['parameter_values'])} parameter values)")


if __name__ == "__main__":
    main()
