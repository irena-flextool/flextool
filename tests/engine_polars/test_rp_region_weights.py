"""Switch S — per-region representative-period weights, storage coupling.

Exercises the ``rp_base__rep`` (n, b, r) path added for per-region
representative-period weight matrices, covering:

1. **Byte-parity fallback** — ``_compute_rp_frames`` / ``_rp_weights_frame``
   with no group weights emit the historical three-column
   ``(base_start, rep_start, weight)`` frame, byte-identical to prior
   behaviour (the primary regression guard).
2. **Per-node fan-out shape** — with region weights + storage nodes the
   frame gains a ``node`` column, each node carrying its region's matrix.
3. **Per-region inter-period drift diverges** — a 2-node toy where the two
   nodes carry deliberately different weight matrices: their
   ``v_state_inter`` inter-period drifts differ by exactly the weight ratio,
   proving the per-region weight reaches the ``rp_inter_period_balance``
   constraint per node instead of cross-joining a single shared matrix.
4. **RHS stays unit-free at unitsize != 1** — the same toy runs at
   ``p_state_unitsize == 1000`` and the drift equals ``w · D`` in var-units
   (no ``× unitsize`` factor): the previously-fixed unitsize bug must not
   resurface on the ``n``-keyed path.
"""
from __future__ import annotations

import polars as pl

from polar_high import Param, Problem
from flextool.engine_polars import build_flextool
from flextool.engine_polars.input import FlexData
from flextool.engine_polars._emit_solve_writers import (
    _compute_rp_frames,
    _rp_weights_frame,
)

UNITSIZE = 1000.0
# Var-unit cap on v_state / v_state_inter.  Physical cap = 1000 energy
# units, far above the +2 physical surplus (var drift 0.002), so the store
# absorbs the whole surplus at zero slack and v_state_inter is free to carry
# the weighted drift without overflow.
STATE_CAP = 1.0

# Shared base->rep weight rows (b1, b2 both map to the single rep t01).
_SHARED_W = 1.0
# Per-region: Ireland keeps weight 1.0; Scotland gets 0.5 on the chained
# edge (b2) so its inter-period drift is exactly half Ireland's.
_SCOT_W = 0.5


# ---------------------------------------------------------------------------
# 1 + 2 — emit-layer frame shape
# ---------------------------------------------------------------------------
def _shared_rp_weights() -> dict[str, dict[str, float]]:
    return {"b1": {"t01": 1.0}, "b2": {"t01": 1.0}}


def test_compute_rp_frames_byte_parity_fallback() -> None:
    """No region weights => rp_weights.csv is the node-less shared frame."""
    rp_w = _shared_rp_weights()
    ts_dur = [("t01", 2.0)]
    steps = ["t01", "t02"]
    base = _compute_rp_frames(rp_w, ts_dur, "p_rp", steps)
    # Explicitly passing empty region context must be identical to omitting.
    with_empty = _compute_rp_frames(
        rp_w, ts_dur, "p_rp", steps,
        region_rp_weights={}, node_to_region={}, storage_nodes=[],
    )
    assert base["rp_weights.csv"].columns == [
        "base_start", "rep_start", "weight",
    ]
    assert base["rp_weights.csv"].equals(with_empty["rp_weights.csv"])
    # timestep_weight.csv (annualisation) stays shared / node-less too.
    assert base["timestep_weight.csv"].equals(
        with_empty["timestep_weight.csv"])


def test_rp_weights_frame_fans_node_column() -> None:
    """Per-region weights fan a node column, region matrix per node."""
    rp_w = _shared_rp_weights()
    base_starts = ["b1", "b2"]
    region_rp = {"Scotland_region": {"b1": {"t01": 1.0}, "b2": {"t01": 0.5}}}
    node_to_region = {"bat_scot": "Scotland_region"}
    frame = _rp_weights_frame(
        rp_w, base_starts, region_rp, node_to_region,
        ["bat_ie", "bat_scot"],
    )
    assert frame.columns == ["base_start", "rep_start", "node", "weight"]
    # Ireland (no region) uses the shared matrix; Scotland uses its own.
    ie = frame.filter(pl.col("node") == "bat_ie")
    scot = frame.filter(pl.col("node") == "bat_scot")
    ie_b2 = ie.filter(pl.col("base_start") == "b2")["weight"].to_list()
    scot_b2 = scot.filter(pl.col("base_start") == "b2")["weight"].to_list()
    assert ie_b2 == ["1.0"]
    assert scot_b2 == ["0.5"]


# ---------------------------------------------------------------------------
# 3 + 4 — engine: per-region drift diverges, RHS unit-free
# ---------------------------------------------------------------------------
def _build_toy_two_node(*, per_region: bool) -> FlexData:
    """Two forward_only storage nodes at unitsize=1000, +2 physical surplus.

    ``per_region`` toggles ``rp_base__rep`` between the shared ``(b, r)``
    matrix (both nodes weight 1.0) and the region-fanned ``(n, b, r)``
    matrix (bat_ie weight 1.0, bat_scot weight 0.5 on the chained edge b2).
    """
    nodes = ["bat_ie", "bat_scot"]
    dt = pl.DataFrame({"d": ["p_rp", "p_rp"], "t": ["t01", "t02"]})
    p_step = Param(("d", "t"), dt.with_columns(value=pl.lit(1.0)))
    p_rpcw = Param(("d", "t"), dt.with_columns(value=pl.lit(1.0)))
    p_infl = Param(("d",), pl.DataFrame({"d": ["p_rp"], "value": [1.0]}))
    p_psh = Param(("d",), pl.DataFrame({"d": ["p_rp"], "value": [1.0]}))

    nb = pl.DataFrame({"n": nodes})
    nb_dt = nb.join(dt, how="cross")
    p_inflow = Param(("n", "d", "t"), pl.DataFrame({
        "n": [n for n in nodes for _ in range(2)],
        "d": ["p_rp"] * 4,
        "t": ["t01", "t02"] * 2,
        "value": [5.0, -3.0, 5.0, -3.0]}))
    p_pup = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(1.0)).select("n", "d", "t", "value"))
    p_pdn = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(1.0)).select("n", "d", "t", "value"))

    nodeState = pl.DataFrame({"n": nodes})
    nodeState_dt = nodeState.join(dt, how="cross")
    nodeState_first_dt = (nodeState_dt.sort(["n", "d", "t"])
        .group_by(["n", "d"], maintain_order=True).first()
        .select("n", "d", "t"))
    nodeState_last_dt = (nodeState_dt.sort(["n", "d", "t"])
        .group_by(["n", "d"], maintain_order=True).last()
        .select("n", "d", "t"))
    p_state_unitsize = Param(("n",),
        pl.DataFrame({"n": nodes, "value": [UNITSIZE, UNITSIZE]}))
    p_state_upper = Param(("n", "d"),
        pl.DataFrame({"n": nodes, "d": ["p_rp"] * 2,
                      "value": [STATE_CAP, STATE_CAP]}))
    p_state_sd = Param(("n",),
        pl.DataFrame({"n": nodes, "value": [0.0, 0.0]}))
    p_state_exi_cap = Param(("n", "d"),
        pl.DataFrame({"n": nodes, "d": ["p_rp"] * 2,
                      "value": [STATE_CAP * UNITSIZE, STATE_CAP * UNITSIZE]}))

    dtttdt = pl.DataFrame({
        "d": ["p_rp", "p_rp"], "t": ["t01", "t02"],
        "t_previous": ["t02", "t01"],
        "t_previous_within_timeset": ["t02", "t01"],
        "d_previous": ["p_rp", "p_rp"],
        "t_previous_within_solve": ["t02", "t01"],
    })

    both = pl.DataFrame({"n": nodes})
    rp_base_period_set = pl.DataFrame({"b": ["b1", "b2"]})
    rp_base_chain = pl.DataFrame({"b": ["b2"], "b_prev": ["b1"]})
    rp_base_first = pl.DataFrame({"b": ["b1"]})
    rp_base_last = pl.DataFrame({"b": ["b2"]})
    rp_block_first = pl.DataFrame({"d": ["p_rp"], "t": ["t01"]})
    p_rp_last_step = pl.DataFrame({"r": ["t01"], "last_step": ["t02"]})

    if per_region:
        rp_base__rep = Param(("n", "b", "r"), pl.DataFrame({
            "n": ["bat_ie", "bat_ie", "bat_scot", "bat_scot"],
            "b": ["b1", "b2", "b1", "b2"],
            "r": ["t01", "t01", "t01", "t01"],
            "value": [1.0, _SHARED_W, 1.0, _SCOT_W]}))
    else:
        rp_base__rep = Param(("b", "r"), pl.DataFrame({
            "b": ["b1", "b2"], "r": ["t01", "t01"],
            "value": [_SHARED_W, _SHARED_W]}))

    return FlexData(
        dt=dt, p_step_duration=p_step, p_timestep_weight=p_rpcw,
        p_inflation_op=p_infl, p_period_share=p_psh,
        nodeBalance=nb, nodeBalance_dt=nb_dt,
        p_inflow=p_inflow, p_penalty_up=p_pup, p_penalty_down=p_pdn,
        nodeState=nodeState, nodeState_dt=nodeState_dt,
        nodeState_first_dt=nodeState_first_dt,
        nodeState_last_dt=nodeState_last_dt,
        p_state_unitsize=p_state_unitsize,
        p_state_upper=p_state_upper,
        p_state_self_discharge=p_state_sd,
        p_state_existing_capacity=p_state_exi_cap,
        dtttdt=dtttdt,
        storage_bind_forward_only_blended_weights=both,
        nodeState_rp=both,
        rp_base_period_set=rp_base_period_set,
        rp_base_chain=rp_base_chain,
        rp_base_first=rp_base_first,
        rp_base_last=rp_base_last,
        rp_block_first=rp_block_first,
        p_rp_last_step=p_rp_last_step,
        rp_base__rep=rp_base__rep,
    )


def _inter_drift(sol, node: str) -> float:
    """v_state_inter[node, b2] - v_state_inter[node, b1] from a Solution."""
    frame = sol.value("v_state_inter")
    ncol = "n" if "n" in frame.columns else frame.columns[0]
    sub = frame.filter(pl.col(ncol) == node)
    by_b = {row["b"]: row["value"] for row in sub.iter_rows(named=True)}
    return float(by_b["b2"] - by_b["b1"])


def test_per_region_drift_diverges_and_is_unit_free() -> None:
    """Per-region weights make the two nodes' inter-period drift differ.

    Expected var-unit drift D = +2 physical / unitsize = 0.002.

    * shared matrix (both weight 1.0): both nodes drift 1.0 · D = 0.002.
    * per-region (bat_ie 1.0, bat_scot 0.5): bat_ie drifts 0.002,
      bat_scot drifts 0.5 · 0.002 = 0.001.

    obj == 0 in both runs (the store absorbs the surplus at zero slack),
    and the drift equals ``w · D`` WITHOUT any ``× unitsize`` factor at
    unitsize == 1000 — the unit-free RHS guard on the n-keyed path.
    """
    D = 2.0 / UNITSIZE

    shared = _build_toy_two_node(per_region=False)
    pb_s = Problem()
    build_flextool(pb_s, shared)
    sol_s = pb_s.solve(options={"random_seed": 42, "parallel": "off"})
    assert sol_s.optimal
    assert abs(sol_s.obj) < 1e-9, f"shared obj {sol_s.obj!r} != 0"
    ie_s = _inter_drift(sol_s, "bat_ie")
    scot_s = _inter_drift(sol_s, "bat_scot")
    assert abs(ie_s - D) < 1e-9, f"shared bat_ie drift {ie_s!r} != {D}"
    assert abs(scot_s - D) < 1e-9, f"shared bat_scot drift {scot_s!r} != {D}"

    region = _build_toy_two_node(per_region=True)
    pb_r = Problem()
    build_flextool(pb_r, region)
    sol_r = pb_r.solve(options={"random_seed": 42, "parallel": "off"})
    assert sol_r.optimal
    assert abs(sol_r.obj) < 1e-9, f"region obj {sol_r.obj!r} != 0"
    ie_r = _inter_drift(sol_r, "bat_ie")
    scot_r = _inter_drift(sol_r, "bat_scot")
    # Ireland keeps the full-weight drift; Scotland's halves — the exact
    # weight ratio, proving the per-region matrix reaches the constraint
    # per node.  A × unitsize factor would inflate these ~1000x (and
    # overflow the cap, changing obj), so the equalities also guard the
    # unit-free RHS.
    assert abs(ie_r - D) < 1e-9, f"region bat_ie drift {ie_r!r} != {D}"
    assert abs(scot_r - _SCOT_W * D) < 1e-9, (
        f"region bat_scot drift {scot_r!r} != {_SCOT_W * D} "
        f"(expected half of Ireland's — per-region weight not applied)"
    )
    assert abs(scot_r - scot_s) > 1e-6, (
        "per-region and shared Scotland drift must differ — the region "
        "weight is not diverging from the shared matrix"
    )
