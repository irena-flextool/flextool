"""A two-region Benders toy for the region-split tests (an overlay on ``tests.json``, base chain ``["init"]``).

Nodes ``rA`` and ``rB`` (balance, VOLL ``penalty_up`` / ``penalty_down`` 1000): rA needs 200 MW in hours 1-24 and
nothing after, rB nothing in hours 1-24 and 200 MW (``uom``: 100 t) in hours 25-48.  ``genA`` -> rA has 100 MW
existing (no cost); ``genB`` -> rB has none and invests (``invest_period``, ``invest_cost`` 100 per kW, 20 years,
``invest_max_period`` 1000).  Connection ``link`` rA -> rB, 200 MW existing.  Groups ``region_A`` {rA, rA_aux} and
``region_B`` {rB} list nodes only (``group__node``), as the docs say; the units are not listed.  ``rA_aux`` is an
inert balance node inside region_A: it keeps the master buildable (a master with no non-terminal balance node
fails to build) without moving the test onto the master-hosted path.

``master_hosted_rB``: rB is in no region (hosted in the master), region_B gets no ``decomposition_method`` and a
third region ``region_C`` {rC} has ``genC`` -> rC (10 MW existing) serving rC's constant 10 MW (a region needs a
balance node with content).  genB is then master-local.

Solve ``y2020_2day_dispatch`` (48 x 1 h), invests in p2020, ``benders_max_iter`` 400 and ``benders_tolerance``
1e-6 (the default 50 iterations do not converge on this toy).

Hand oracle (regular link, efficiency 1): hours 1-24 rA needs 200 = genA's 100 + the reverse flow, which genB's
capacity C limits, so C >= 100; hours 25-48 rB needs 200 = C + the forward flow, which genA's 100 limits, so
C >= 100.  VOLL (1000 per MWh, 24 h, scaled to a year) is far above the annuity, so C = 100 MW, the only cost is
genB's investment: :func:`hand_objective`.
"""
from __future__ import annotations

from tests.engine._unit_db import TIMESTEPS, b64, make_db

SOLVE = "y2020_2day_dispatch"
HOURS = 48
INVEST_COST = 100.0       # per kW
LIFETIME = 20.0           # years
DISCOUNT_RATE = 0.05      # the default discount_rate


def hand_objective(capacity_mw: float) -> float:
    """genB's yearly investment cost for *capacity_mw*: with the defaults (``discount_offset_investment`` 0, the
    payments at the start of the year sized to be worth ``invest_cost``; ``social_discount_rate`` 0;
    ``years_represented`` 1) the payment is ``invest_cost * CRF / (1 + r)``, CRF = r / (1 - (1 + r)^-n)."""
    r = DISCOUNT_RATE
    crf = r / (1.0 - (1.0 + r) ** -LIFETIME)
    return capacity_mw * 1000.0 * INVEST_COST * crf / (1.0 + r)


def _time_map(f) -> list:
    return b64({"index_type": "str", "index_name": "time",
                "data": [[t, f(i)] for i, t in enumerate(TIMESTEPS[:HOURS], 1)]}, "map")


def build(tmp_path_factory, *, transfer_method: str, efficiency: float = 1.0, uom: bool = False,
          master_hosted_rB: bool = False, decomposition: str = "benders", alt: str | None = None,
          extra_entities=(), extra_values=()) -> tuple[str, str]:
    """Build the toy database in a tmp folder; returns ``(url, scenario)``."""
    demand_b = 100.0 if uom else 200.0
    ents = [["node", "rA", None], ["node", "rA_aux", None], ["node", "rB", None],
            ["unit", "genA", None], ["unit", "genB", None],
            ["unit__outputNode", ["genA", "rA"], None], ["unit__outputNode", ["genB", "rB"], None],
            ["connection", "link", None], ["connection__node__node", ["link", "rA", "rB"], None],
            ["group", "region_A", None], ["group", "region_B", None],
            ["group__node", ["region_A", "rA"], None], ["group__node", ["region_A", "rA_aux"], None]]
    benders_regional = b64("benders_regional", "str")
    vals = [["node", "rA", "node_type", b64("balance", "str")],
            ["node", "rA", "inflow", _time_map(lambda i: -200.0 if i <= 24 else 0.0)],
            ["node", "rA", "penalty_up", b64(1000.0, "float")], ["node", "rA", "penalty_down", b64(1000.0, "float")],
            ["node", "rA_aux", "node_type", b64("balance", "str")],
            ["node", "rB", "node_type", b64("balance", "str")],
            ["node", "rB", "inflow", _time_map(lambda i: 0.0 if i <= 24 else -demand_b)],
            ["node", "rB", "penalty_up", b64(1000.0, "float")], ["node", "rB", "penalty_down", b64(1000.0, "float")],
            ["unit", "genA", "existing", b64(100.0, "float")],
            ["unit", "genB", "existing", b64(0.0, "float")],
            ["unit", "genB", "invest_method", b64("invest_period", "str")],
            ["unit", "genB", "invest_cost", b64(INVEST_COST, "float")],
            ["unit", "genB", "lifetime_technical", b64(LIFETIME, "float")],
            ["unit", "genB", "invest_max_period",
             b64({"index_type": "str", "index_name": "period", "data": [["p2020", 1000.0]]}, "map")],
            ["connection", "link", "transfer_method", b64(transfer_method, "str")],
            ["connection", "link", "existing", b64(200.0, "float")],
            ["connection", "link", "efficiency", b64(efficiency, "float")],
            ["group", "region_A", "decomposition_method", benders_regional],
            ["solve", SOLVE, "invest_periods", b64({"value_type": "str", "data": ["p2020"]}, "array")],
            ["solve", SOLVE, "realized_invest_periods", b64({"value_type": "str", "data": ["p2020"]}, "array")],
            ["solve", SOLVE, "decomposition", b64(decomposition, "str")],
            ["solve", SOLVE, "benders_max_iter", b64(400.0, "float")],
            ["solve", SOLVE, "benders_tolerance", b64(1e-6, "float")]]
    if master_hosted_rB:
        ents += [["node", "rC", None], ["unit", "genC", None], ["unit__outputNode", ["genC", "rC"], None],
                 ["group", "region_C", None], ["group__node", ["region_C", "rC"], None]]
        vals += [["group", "region_C", "decomposition_method", benders_regional],
                 ["node", "rC", "node_type", b64("balance", "str")], ["node", "rC", "inflow", b64(-10.0, "float")],
                 ["unit", "genC", "existing", b64(10.0, "float")]]
    else:
        ents.append(["group__node", ["region_B", "rB"], None])
        vals.append(["group", "region_B", "decomposition_method", benders_regional])
    if uom:
        vals += [["node", "rA", "unit_of_measure", b64("MWh", "str")],
                 ["node", "rB", "unit_of_measure", b64("t", "str")]]
    if alt is None:
        alt = (f"two_region_{transfer_method}_{decomposition}{'_uom' if uom else ''}"
               f"{'_mh' if master_hosted_rB else ''}")
    return make_db(tmp_path_factory, alt=alt, values=vals + list(extra_values), base_chain=["init"],
                   entities=ents + list(extra_entities))
