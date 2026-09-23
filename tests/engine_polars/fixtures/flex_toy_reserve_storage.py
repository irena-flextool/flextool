"""Synthetic minimal-feature toy: storage-backed reserve with
``reserve_duration`` energy adequacy (issue #322).

Extends ``flex_toy_reserve.py`` with a storage node ``bat`` (nodeState)
and a storage-coupled reserve provider, so the two new constraints in
``_reserve._add_storage_reserve_constraints`` bind:

  * ``reserve_storage_up_floor``      — UP  / discharge floor
  * ``reserve_storage_down_headroom`` — DOWN / charge headroom

Topology (``direction="up"``)
-----------------------------
* nodeBalance nodes ``elec`` (reserve node) and ``bat`` (storage node),
  both with zero inflow (dispatch is trivial — the only driver is the
  reserve requirement).
* Storage node ``bat``: nodeState, ``p_state_unitsize = 1``,
  ``p_state_upper = cap`` (var-units), no binding method (``v_state``
  free in ``[0, cap]``, uncosted).
* Provider process ``dis``: source=``bat``, sink=``elec``, eff partition,
  ``p_unitsize = 100`` MW, slope ``= 1/eta``.  Provides UP reserve at
  ``elec`` → ``bat`` is its storage source, so the up-floor couples
  ``v_reserve`` to ``v_state``.
* Reserve ``(r1, up, g)``, method ``timeseries``, reservation 50 MW,
  ``penalty_reserve = 1000``, ``reserve_duration = duration`` (h).

Closed form (``direction="up"``, ``eta=1``, ``cap=20``, ``duration=0.5``)
------------------------------------------------------------------------
Energy floor:  v_reserve·100·duration·slope ≤ v_state·1 ≤ cap·1 = 20.
  ⇒ v_reserve ≤ cap / (100·duration·slope) = 20 / (100·0.5·1) = 0.4.
reserveBalance (>=):  v_reserve·100·1 + vq·50 ≥ 50
  ⇒ 40 + vq·50 ≥ 50  ⇒ vq = 0.2.
Raising duration 0.5→1.0 halves the energy allowance:
  v_reserve ≤ 0.2, vq = 0.6 — reserve tightens, slack rises.

Topology (``direction="down"``)
-------------------------------
* Provider ``chg``: source=``elec``, sink=``bat`` (charge).  Provides
  DOWN reserve at ``elec``; ``bat`` is its storage sink, so the
  down-headroom couples committed charge power to free capacity.
* Efficiency divides (×η): E_dn = v_reserve·100·duration·(1/slope) =
  v_reserve·100·duration·eta.  With ``eta=0.9``, ``duration=0.5``,
  ``cap=20`` and ``v_state → 0`` (LP maximises headroom):
    v_reserve·100·0.5·0.9 ≤ 20  ⇒ v_reserve ≤ 0.4444,  vq = 0.1111.
"""
from __future__ import annotations

import polars as pl
from polar_high import Param
from flextool.engine_polars.input import FlexData


_RESERVATION = 50.0
_UNITSIZE = 100.0        # provider p_unitsize (MW)
_US_BAT = 1.0            # storage p_state_unitsize (MWh / var-unit)
_PENALTY = 1000.0


def data(*, direction: str = "up", duration: float | None = 0.5,
         eta: float = 1.0, cap: float = 20.0,
         provider: str = "prov") -> FlexData:
    """Build the storage-backed reserve toy.

    Parameters
    ----------
    direction : {"up", "down"}
        UP → discharge provider (source=bat); DOWN → charge provider
        (source=elec, sink=bat).
    duration : float | None
        ``reserve_duration`` in hours.  ``None`` omits the parameter
        entirely (no-op guard — the two coupling constraints are not
        emitted and the LP is byte-identical to the pre-feature build).
    eta : float
        Provider efficiency; slope = 1/eta.
    cap : float
        Storage ``p_state_upper`` in var-units (max stored energy = cap ·
        p_state_unitsize MWh).
    """
    ud = "up" if direction == "up" else "down"
    slope = 1.0 / eta

    # ── Time ──────────────────────────────────────────────────────────
    dt = pl.DataFrame({"d": ["p2020", "p2020"], "t": ["t01", "t02"]})
    p_step_duration = Param(("d", "t"),
        pl.DataFrame({"d": ["p2020"] * 2, "t": ["t01", "t02"], "value": [1.0, 1.0]}))
    p_timestep_weight = Param(("d", "t"),
        pl.DataFrame({"d": ["p2020"] * 2, "t": ["t01", "t02"], "value": [1.0, 1.0]}))
    p_inflation_op = Param(("d",), pl.DataFrame({"d": ["p2020"], "value": [1.0]}))
    p_period_share = Param(("d",), pl.DataFrame({"d": ["p2020"], "value": [1.0]}))

    # ── Nodes (both zero-inflow; reserve is the only driver) ──────────
    nodeBalance = pl.DataFrame({"n": ["elec", "bat"]})
    nodeBalance_dt = nodeBalance.join(dt, how="cross")
    p_inflow = Param(("n", "d", "t"),
        nodeBalance_dt.with_columns(value=pl.lit(0.0)).select("n", "d", "t", "value"))
    p_penalty_up = Param(("n", "d", "t"),
        nodeBalance_dt.with_columns(value=pl.lit(1e6)).select("n", "d", "t", "value"))
    p_penalty_down = Param(("n", "d", "t"),
        nodeBalance_dt.with_columns(value=pl.lit(1e6)).select("n", "d", "t", "value"))

    # ── Process topology ──────────────────────────────────────────────
    if direction == "up":
        src, snk = "bat", "elec"      # discharge: bat → elec
    else:
        src, snk = "elec", "bat"      # charge: elec → bat
    pss = pl.DataFrame({"p": [provider], "source": [src], "sink": [snk]})
    pss_eff = pss.clone()
    pss_noEff = pl.DataFrame(schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    pss_dt = pss.join(dt, how="cross")
    flow_to_n = pss.with_columns(n=pl.col("sink"))
    # Source draws from a nodeBalance node (bat for up, elec for down).
    flow_from_nodeBalance_eff = pss.clone()
    flow_from_nodeBalance_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    # No commodity flows.
    flow_from_commodity_eff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8, "c": pl.Utf8})
    flow_from_commodity_noEff = flow_from_commodity_eff.clone()

    p_unitsize = Param(("p",), pl.DataFrame({"p": [provider], "value": [_UNITSIZE]}))
    p_flow_upper = Param(("p", "source", "sink", "d", "t"),
        pss_dt.with_columns(value=pl.lit(1.0))
              .select("p", "source", "sink", "d", "t", "value"))
    p_slope = Param(("p", "d", "t"),
        dt.with_columns(p=pl.lit(provider), value=pl.lit(slope))
          .select("p", "d", "t", "value"))
    p_commodity_price = Param(("c", "d", "t"),
        pl.DataFrame(schema={"c": pl.Utf8, "d": pl.Utf8, "t": pl.Utf8,
                             "value": pl.Float64}))

    # ── Storage node ``bat`` ──────────────────────────────────────────
    nodeState = pl.DataFrame({"n": ["bat"]})
    nodeState_dt = nodeState.join(dt, how="cross")
    nodeState_first_dt = (nodeState_dt.sort(["n", "d", "t"])
                          .group_by(["n", "d"], maintain_order=True)
                          .first().select("n", "d", "t"))
    nodeState_last_dt = (nodeState_dt.sort(["n", "d", "t"])
                         .group_by(["n", "d"], maintain_order=True)
                         .last().select("n", "d", "t"))
    p_state_unitsize = Param(("n",),
        pl.DataFrame({"n": ["bat"], "value": [_US_BAT]}))
    p_state_upper = Param(("n", "d"),
        pl.DataFrame({"n": ["bat"], "d": ["p2020"], "value": [cap]}))
    p_state_self_discharge = Param(("n",),
        pl.DataFrame({"n": ["bat"], "value": [0.0]}))
    p_state_existing_capacity = Param(("n", "d"),
        pl.DataFrame({"n": ["bat"], "d": ["p2020"], "value": [cap * _US_BAT]}))
    # Cyclic step_previous over the 2 timesteps within p2020.
    dtttdt = pl.DataFrame({
        "d": ["p2020", "p2020"],
        "t": ["t01", "t02"],
        "t_previous": ["t02", "t01"],
        "t_previous_within_timeset": ["t02", "t01"],
        "d_previous": ["p2020", "p2020"],
        "t_previous_within_solve": ["t02", "t01"],
    })

    # ── Reserve data ──────────────────────────────────────────────────
    reserve_upDown_group = pl.DataFrame({"r": ["r1"], "ud": [ud], "g": ["g"]})
    reserve_upDown_group_method_timeseries = pl.DataFrame(
        {"r": ["r1"], "ud": [ud], "g": ["g"], "method": ["timeseries"]})
    reserve_upDown_group_method_dynamic = pl.DataFrame(
        schema={"r": pl.Utf8, "ud": pl.Utf8, "g": pl.Utf8, "method": pl.Utf8})
    reserve_upDown_group_method_n_1 = pl.DataFrame(
        schema={"r": pl.Utf8, "ud": pl.Utf8, "g": pl.Utf8, "method": pl.Utf8})

    prundt = pl.DataFrame({
        "p": [provider] * 2, "r": ["r1"] * 2, "ud": [ud] * 2,
        "n": ["elec"] * 2, "d": ["p2020"] * 2, "t": ["t01", "t02"],
    })
    process_reserve_upDown_node_active = pl.DataFrame(
        {"p": [provider], "r": ["r1"], "ud": [ud], "n": ["elec"]})
    group_node = pl.DataFrame({"g": ["g"], "n": ["elec"]})

    p_process_reserve_upDown_node_reliability = Param(("p", "r", "ud", "n"),
        pl.DataFrame({"p": [provider], "r": ["r1"], "ud": [ud], "n": ["elec"],
                      "value": [1.0]}))
    pdtReserve_upDown_group_reservation = Param(("r", "ud", "g", "d", "t"),
        pl.DataFrame({"r": ["r1"] * 2, "ud": [ud] * 2, "g": ["g"] * 2,
                      "d": ["p2020"] * 2, "t": ["t01", "t02"],
                      "value": [_RESERVATION, _RESERVATION]}))
    p_reserve_upDown_group_penalty_reserve = Param(("r", "ud", "g"),
        pl.DataFrame({"r": ["r1"], "ud": [ud], "g": ["g"], "value": [_PENALTY]}))
    p_process_reserve_upDown_node_max_share = Param(("p", "r", "ud", "n"),
        pl.DataFrame({"p": [provider], "r": ["r1"], "ud": [ud], "n": ["elec"],
                      "value": [1.0]}))
    p_process_existing_count = Param(("p", "d"),
        pl.DataFrame({"p": [provider], "d": ["p2020"], "value": [1.0]}))

    kwargs = dict(
        dt=dt,
        p_step_duration=p_step_duration,
        p_timestep_weight=p_timestep_weight,
        p_inflation_op=p_inflation_op,
        p_period_share=p_period_share,
        nodeBalance=nodeBalance,
        nodeBalance_dt=nodeBalance_dt,
        p_inflow=p_inflow,
        p_penalty_up=p_penalty_up,
        p_penalty_down=p_penalty_down,
        process_source_sink=pss,
        process_source_sink_eff=pss_eff,
        process_source_sink_noEff=pss_noEff,
        pss_dt=pss_dt,
        flow_to_n=flow_to_n,
        flow_from_nodeBalance_eff=flow_from_nodeBalance_eff,
        flow_from_nodeBalance_noEff=flow_from_nodeBalance_noEff,
        flow_from_commodity_eff=flow_from_commodity_eff,
        flow_from_commodity_noEff=flow_from_commodity_noEff,
        p_unitsize=p_unitsize,
        p_flow_upper=p_flow_upper,
        p_slope=p_slope,
        p_commodity_price=p_commodity_price,
        # Storage
        nodeState=nodeState,
        nodeState_dt=nodeState_dt,
        nodeState_first_dt=nodeState_first_dt,
        nodeState_last_dt=nodeState_last_dt,
        p_state_unitsize=p_state_unitsize,
        p_state_upper=p_state_upper,
        p_state_self_discharge=p_state_self_discharge,
        p_state_existing_capacity=p_state_existing_capacity,
        dtttdt=dtttdt,
        # Reserve
        reserve_upDown_group=reserve_upDown_group,
        reserve_upDown_group_method_timeseries=reserve_upDown_group_method_timeseries,
        reserve_upDown_group_method_dynamic=reserve_upDown_group_method_dynamic,
        reserve_upDown_group_method_n_1=reserve_upDown_group_method_n_1,
        prundt=prundt,
        process_reserve_upDown_node_active=process_reserve_upDown_node_active,
        group_node=group_node,
        p_process_reserve_upDown_node_reliability=p_process_reserve_upDown_node_reliability,
        pdtReserve_upDown_group_reservation=pdtReserve_upDown_group_reservation,
        p_reserve_upDown_group_penalty_reserve=p_reserve_upDown_group_penalty_reserve,
        p_process_reserve_upDown_node_max_share=p_process_reserve_upDown_node_max_share,
        p_process_existing_count=p_process_existing_count,
    )
    if duration is not None:
        kwargs["p_reserve_upDown_group_reserve_duration"] = Param(
            ("r", "ud", "g"),
            pl.DataFrame({"r": ["r1"], "ud": [ud], "g": ["g"],
                          "value": [float(duration)]}))
    return FlexData(**kwargs)
