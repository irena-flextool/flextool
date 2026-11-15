"""Ramp envelope: the room each node / dispatch node group has to change its supply toward the next step.

Row t of the tables is the transition t → t⁺ (t⁺ the unique realized successor of t, the inverse of
``s.ramp_link`` incl. the roll carry).  Per row and entity, in MW (``specs/ramp_envelope_final_spec.md`` §2):

* ``ramp_net_load`` = N(t⁺) − N(t), N = demand − Σ VRE potential (+ = more supply needed); ``ramp_demand`` the
  same for the demand alone (demand = −inflow);
* ``units_up`` / ``units_down``, ``storage_up`` / ``storage_down``, ``vre_up`` / ``vre_down``,
  ``connections_up`` / ``connections_down``: the room toward t⁺ of the arcs at the entity in entity-supply MW,
  every band ≥ 0 and the down bands stored negative.

Every arc's room is computed raw first (may be negative: a forced movement), converted to the node's MW, and
clipped at 0 per arc last, before the arcs are summed into a band (§2.0): a forced movement of one arc never eats
another arc's room.  Rows without exactly one realized successor (the last step, a block end before a gap, a
branch point with several visible successors) are empty (NaN).

Unit arcs (§2.1-§2.2), in v_flow space then node MW::

    U   = (online(t⁺) − startup(t⁺)) · unitsize      units online in both steps (online processes)
        = capacity(d⁺)                                otherwise (incl. a solve that left online out), floor 0
    C   = κ(t⁺) · U    (κ the arc's maxFlow multiplier, capacity coefficient × availability)
    L   = F(t⁺) · U    (F the floor per unit online)
    f₂  = f(t) − F(t) · unitsize · shutdown(t⁺)      (the stopping units' output is forced movement)
    up_a   = min(C − f₂, ρ̄↑) − R↑(t⁺),   down_a = min(f₂ − L, ρ̄↓) − R↓(t⁺)
    ρ̄   = speed · 60 · (Δ + Δ⁺)/2 · c(t⁺) · U        for the arcs of the ramp-limit sets of the solve that
                                                     realized t⁺; in one period only where speed·60·(Δ+Δ⁺)/2 < 1
    R↑  = Σ_r reserve_up(sink) + φ · Σ_r reserve_down(source),  R↓ the mirror,  φ = 1/slope(t⁺) on a direct arc

Node conversion: an output arc counts ×1; an indirect unit's input arc flips direction (up_n = down_a, down_n =
up_a); a direct unit's input arc converts each step with its own slope and section (CHG-12)::

    up_n   = (slope(t) − slope(t⁺)) · f₂ + slope(t⁺) · down_a + (section(t) − section(t⁺)) · U
    down_n = (slope(t⁺) − slope(t)) · f₂ + slope(t⁺) · up_a  + (section(t⁺) − section(t)) · U

VRE arcs (an output with an ``upper_limit`` profile, §2.3) take C = pot(t⁺) and are measured against the
potential: vre_up = clip0(up_a − ΔP), vre_down = clip0(down_a + ΔP), ΔP = pot(t⁺) − pot(t).  Storage arcs (§2.4,
processes between a storage node S and the entity) are capped per S by the state of charge::

    up ≤ clip0(state(t)/Δ⁺ − D(t)) · η_out,   down ≤ clip0((capacity_S(d⁺) − state(t))/Δ⁺ + D(t)) / η_in

(D the net draw from S at t, η = 1/slope(t⁺) of the processes delivering from / to S).  Connections (§2.5) use the
node-side range of their flow at t⁺, losses on the drawing side, with the reserve they hold at the node::

    up = cap·avail(t⁺) − f_in(t) − R↑_c,n(t⁺),   down = f_in(t) + cap·avail(t⁺)·slope(t⁺) − R↓_c,n(t⁺)

A coarse process (``processCoarse``) has one flow per coarse step: its room is 0 on the rows inside a coarse step,
and at a coarse step's last fine row ρ̄ uses the coarse step durations.  Node groups (``print_dispatch``) sum the
arcs at their non-storage members; a process with every node in the group is internal and counts 0; member storage
nodes count as outside (§2.7).  The arcs are summed into the entities with one membership-matrix product per band
and direction.

The arcs with ``conversion_flow_coeff = 0`` (``process_source_sink_uncapped``: outside every per-arc capacity and
ramp row) have no room of their own and are left out; a process that touches more than one storage node is in the
storage band without a state-of-charge cap (the cap is per storage node).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: The bands of the envelope tables, in table order.
UP_BANDS = ("units_up", "storage_up", "vre_up", "connections_up")
DOWN_BANDS = ("units_down", "storage_down", "vre_down", "connections_down")
BANDS = ("ramp_net_load", "ramp_demand", *UP_BANDS, *DOWN_BANDS)
#: The ramp-limit sets (``_ramp_inputs.RAMP_LIMITS`` labels) per direction.
_LIMITS = {"up": ("sink_up", "source_up"), "down": ("sink_down", "source_down")}
_SENTINEL = "\x00no-successor"


class _Next:
    """The realized successor t⁺ of every row (spec §2.8): the inverse of ``s.ramp_link``, rows with no or with
    several successors have none."""

    def __init__(self, rows: pd.MultiIndex, ramp_link: pd.MultiIndex, ramp_carry: "pd.MultiIndex | None"):
        self.rows = rows
        later_p = ramp_link.get_level_values("period").astype(object)
        later_t = ramp_link.get_level_values("time").astype(object)
        prev = pd.MultiIndex.from_arrays(
            [ramp_link.get_level_values("period_previous"), ramp_link.get_level_values("time_previous")],
            names=["period", "time"])
        n_succ = pd.Series(1, index=prev).groupby(level=["period", "time"]).sum()
        single = ~prev.isin(n_succ.index[n_succ != 1])
        succ = pd.DataFrame({"period": np.asarray(later_p)[single], "time": np.asarray(later_t)[single]},
                            index=prev[single])
        succ = succ.reindex(rows)
        self.has = succ["period"].notna().to_numpy()
        self.period_next = np.where(self.has, succ["period"].to_numpy(dtype=object), _SENTINEL)
        time_next = np.where(self.has, succ["time"].to_numpy(dtype=object), _SENTINEL)
        self.index = pd.MultiIndex.from_arrays([self.period_next, time_next], names=["period", "time"])
        periods = rows.get_level_values("period").to_numpy(dtype=object)
        carried = np.zeros(len(rows), dtype=bool)
        if ramp_carry is not None and len(ramp_carry):
            carry_pairs = pd.MultiIndex.from_arrays(
                [ramp_carry.get_level_values("period"), ramp_carry.get_level_values("time"),
                 ramp_carry.get_level_values("period_previous"), ramp_carry.get_level_values("time_previous")])
            pairs = pd.MultiIndex.from_arrays(
                [self.period_next, time_next, periods, rows.get_level_values("time").to_numpy(dtype=object)])
            carried = np.asarray(pairs.isin(carry_pairs), dtype=bool)
        #: Rows whose link stays inside one period and is not carried from a previous roll: the ramp limit
        #: applies there only where the ramp room per unit of capacity is below 1.
        self.in_period = self.has & (self.period_next == periods) & ~carried

    def frame(self, frame: pd.DataFrame, columns=None) -> pd.DataFrame:
        """``frame`` (indexed by ``(period, time)``) read at t⁺ of every row, NaN without t⁺."""
        if columns is not None:
            frame = frame.reindex(columns=columns)
        out = frame.reindex(self.index)
        out.index = self.rows
        return out

    def series(self, series: pd.Series) -> pd.Series:
        out = series.reindex(self.index)
        out.index = self.rows
        return out

    def by_period(self, frame: pd.DataFrame, columns) -> pd.DataFrame:
        """``frame`` (indexed by ``period``) read at the period d⁺ of t⁺."""
        out = frame.reindex(index=pd.Index(self.period_next, name="period"), columns=columns)
        out.index = self.rows
        return out


def _wide(values, rows, columns) -> pd.DataFrame:
    return pd.DataFrame(values, index=rows, columns=columns, dtype=float)


def _level_frame(frame: "pd.DataFrame | None", label: str, columns: "pd.MultiIndex | None") -> "pd.DataFrame | None":
    """The ``limit == label`` slice of a ``(process, source, sink, limit)`` frame, over ``columns`` (NaN-filled)
    when given; ``None`` (or all-NaN over ``columns``) when the frame has no such slice."""
    if frame is None or frame.shape[1] == 0 or label not in frame.columns.get_level_values("limit"):
        if columns is None:
            return None
        return pd.DataFrame(np.nan, index=frame.index if frame is not None else None, columns=columns)
    out = frame.xs(label, level="limit", axis=1)
    return out if columns is None else out.reindex(columns=columns)


def _reduced_online(s, index: pd.MultiIndex, procs: pd.Index) -> np.ndarray:
    """``(rows, processes)`` True where the solve that realized the step of ``index`` left the units online out
    (``reduce_unit_detail_method``, ``s.reduced_cells``: the whole feature, or the feature for one process)."""
    out = np.zeros((len(index), len(procs)), dtype=bool)
    cells = getattr(s, "reduced_cells", None)
    if not cells:
        return out
    out |= cells.step_mask("online", index)[:, None]
    # One mask per marker key of a process left without online variables (the markers, not the data).
    per_process = {k[1]: cells.step_mask(k, index) for k in cells.steps if isinstance(k, tuple) and k[0] == "online"}
    if per_process:
        masks = pd.DataFrame(per_process).reindex(columns=procs, fill_value=False)
        out |= masks.to_numpy(dtype=bool)
    return out


def _process_nodes(s) -> pd.DataFrame:
    """``(process, node, side)`` of every arc end that is a node (the process itself is not); side ``source`` /
    ``sink``.  A node on both sides of a process (a connection's ends) has a row per side."""
    pss = s.process_source_sink.to_frame(index=False)
    ends = pd.concat([pss[["process", "source"]].rename(columns={"source": "node"}).assign(side="source"),
                      pss[["process", "sink"]].rename(columns={"sink": "node"}).assign(side="sink")],
                     ignore_index=True)
    return ends[ends["node"] != ends["process"]].drop_duplicates(ignore_index=True)


def compute_ramp_envelope(par, s, v, r) -> None:
    """``r.node_ramp_envelope_dt_ep`` (node, band), ``r.nodeGroup_ramp_envelope_dt_gp`` (group, band) and
    ``r.timestep_hours_dt`` (the hours a year each step stands for: Δt · timestep_weight / period share)."""
    rows = r.flow_dt.index
    nx = _Next(rows, s.ramp_link, getattr(s, "ramp_carry", None))
    dur = par.step_duration.reindex(rows).to_numpy(dtype=float)
    dur_next = nx.series(par.step_duration).to_numpy(dtype=float)
    storage_nodes = set(s.node_state)
    proc_nodes = _process_nodes(s)

    unit_up, unit_down, unit_meta = _unit_contributions(par, s, v, r, nx, dur, dur_next)
    conn_up, conn_down, conn_meta = _connection_contributions(par, s, r, nx)
    up = pd.concat([unit_up, conn_up], axis=1)
    down = pd.concat([unit_down, conn_down], axis=1)
    meta = pd.concat([unit_meta, conn_meta], ignore_index=True)
    del unit_up, unit_down, conn_up, conn_down
    # Storage nodes have no envelope of their own: the arc ends at them are dropped.
    keep = ~meta["node"].isin(storage_nodes).to_numpy()
    up, down, meta = up.iloc[:, keep], down.iloc[:, keep], meta[keep].reset_index(drop=True)
    # Clip per arc, last (spec §2.0); an arc missing from a solve has no room.
    up = up.clip(lower=0.0).fillna(0.0)
    down = down.clip(lower=0.0).fillna(0.0)

    env_nodes = s.node_balance.union(s.node_balance_period).difference(pd.Index(sorted(storage_nodes)))
    groups = s.nodeGroupDispatch
    gn = s.group_node.to_frame(index=False)
    gn = gn[gn["group"].isin(groups) & ~gn["node"].isin(storage_nodes)]
    membership = _membership(meta, env_nodes, gn, proc_nodes)
    storage_cap = _storage_capacity(par, s, v, r, nx, rows, dur_next)

    lines = _lines(par, r, nx, rows, dur, env_nodes, gn)
    for kind, entities, level in (("node", list(env_nodes), "node"), ("group", list(groups), "group")):
        cols = pd.MultiIndex.from_product([entities, BANDS], names=[level, "band"])
        values = np.full((len(rows), len(entities), len(BANDS)), np.nan)
        if entities:
            bands = _band_sums(up, down, meta, membership[membership["kind"] == kind], entities, storage_cap, rows)
            bands["ramp_net_load"] = lines[kind]["net"]
            bands["ramp_demand"] = lines[kind]["demand"]
            for position, band in enumerate(BANDS):
                values[:, :, position] = bands[band].reindex(columns=entities).to_numpy(dtype=float)
            values[~nx.has] = np.nan
        out = pd.DataFrame(values.reshape(len(rows), len(entities) * len(BANDS)), index=rows, columns=cols)
        if kind == "node":
            r.node_ramp_envelope_dt_ep = out
        else:
            r.nodeGroup_ramp_envelope_dt_gp = out

    weight = par.timestep_weight.reindex(rows).to_numpy(dtype=float)
    share = par.complete_period_share_of_year.reindex(rows.get_level_values("period")).to_numpy(dtype=float)
    hours = pd.DataFrame({"timestep_hours": dur * weight / share}, index=rows)
    hours.columns.name = "param"
    r.timestep_hours_dt = hours


def _arc_columns(frame: "pd.DataFrame | None", arcs: pd.MultiIndex) -> "tuple[np.ndarray, pd.DataFrame | None]":
    """``(positions in arcs, frame restricted to those columns)`` of the columns of ``frame`` that are arcs."""
    if frame is None or frame.shape[1] == 0:
        return np.zeros(0, dtype=int), None
    position = arcs.get_indexer(frame.columns)
    keep = position >= 0
    return position[keep], frame.loc[:, keep]


def _unit_contributions(par, s, v, r, nx: _Next, dur, dur_next):
    """Raw per-arc node rooms of the unit arcs: ``(up, down, meta)``, up / down ``rows × contributions``,
    meta one row per contribution (process, node, band, storage).  Full ``rows × arcs`` arrays only for the flow,
    the capacity and the two rooms; every other attribute on the columns that have it."""
    rows = r.flow_dt.index
    pss = s.process_source_sink
    pss = pss[pss.get_level_values("process").isin(s.process_unit)]
    unc = getattr(s, "process_source_sink_uncapped", None)
    if unc is not None and len(unc):
        pss = pss[~pss.isin(unc)]
    empty = (pd.DataFrame(index=rows), pd.DataFrame(index=rows),
             pd.DataFrame(columns=["process", "node", "band", "storage"]))
    if len(pss) == 0:
        return empty
    arcs = pd.MultiIndex.from_arrays([pss.get_level_values(k) for k in ("process", "source", "sink")],
                                     names=["process", "source", "sink"])
    p = arcs.get_level_values("process")
    src = arcs.get_level_values("source")
    snk = arcs.get_level_values("sink")
    procs = pd.Index(p)
    is_in = np.asarray(snk == p)
    is_direct = np.asarray((src != p) & (snk != p))
    n_rows, n_arcs = len(rows), len(arcs)

    # f: the arc's v_flow in MW — a direct unit's single flow is its output (flow_dt at the sink).
    flow_keys = pd.MultiIndex.from_arrays([p, np.where(is_in, src, p), np.where(is_in, p, snk)])
    f2 = np.nan_to_num(r.flow_dt.reindex(columns=flow_keys).to_numpy(dtype=float), nan=0.0)
    unitsize = par.entity_unitsize.reindex(p).to_numpy(dtype=float)

    # U: units online in both steps (online processes not left out by the solve that realized t⁺), else the
    # built capacity at d⁺.
    cap_u = nx.by_period(r.entity_all_capacity, procs).to_numpy(dtype=float, copy=True)
    online = np.zeros((n_rows, n_arcs), dtype=bool)
    on_pos = np.flatnonzero(procs.isin(s.process_online))
    units_both = np.zeros((n_rows, len(on_pos)))
    if len(on_pos):
        on_procs = procs[on_pos]
        online_next = nx.frame(r.process_online_dt, columns=on_procs).to_numpy(dtype=float)
        startup_next = np.nan_to_num(nx.frame(r.process_startup_dt, columns=on_procs).to_numpy(dtype=float))
        shutdown_next = np.nan_to_num(nx.frame(r.process_shutdown_dt, columns=on_procs).to_numpy(dtype=float))
        on = ~np.isnan(online_next) & ~_reduced_online(s, nx.index, on_procs)
        online[:, on_pos] = on
        units_both = np.where(on, online_next - startup_next, 0.0)
        cap_u[:, on_pos] = np.where(on, units_both * unitsize[on_pos][None, :], cap_u[:, on_pos])
        shutdown = np.zeros((n_rows, n_arcs))
        shutdown[:, on_pos] = np.where(on, shutdown_next, 0.0)
        del online_next, startup_next, shutdown_next

    # Floor F (online only): the stopping units are taken at F(t) (f₂), the floor toward t⁺ is F(t⁺)·U.
    fl_pos, floor = _arc_columns(par.process_arc_floor_share_dt, arcs)
    low = None
    if len(fl_pos):
        fl_on = online[:, fl_pos]
        floor_now = np.where(fl_on, np.nan_to_num(floor.reindex(rows).to_numpy(dtype=float)), 0.0)
        floor_next = np.where(fl_on, np.nan_to_num(nx.frame(floor).to_numpy(dtype=float)), 0.0)
        f2[:, fl_pos] -= floor_now * unitsize[fl_pos][None, :] * shutdown[:, fl_pos]
        low = floor_next * cap_u[:, fl_pos]
        del floor_now, floor_next

    # C = κ(t⁺)·U; a VRE arc's capacity toward t⁺ is its potential at t⁺.
    cap = cap_u.copy()
    k_pos, kappa = _arc_columns(par.process_arc_cap_coef_dt, arcs)
    if len(k_pos):
        cap[:, k_pos] *= np.nan_to_num(nx.frame(kappa).to_numpy(dtype=float), nan=1.0)
    pot = r.potentialVREgen_dt
    vre_cols = pd.MultiIndex.from_arrays([p, snk], names=["process", "node"])
    vre_pos = np.flatnonzero(np.asarray(vre_cols.isin(pot.columns)) & ~is_in)
    delta_p = None
    if len(vre_pos):
        pot_next = np.nan_to_num(nx.frame(pot, columns=vre_cols[vre_pos]).to_numpy(dtype=float))
        pot_now = np.nan_to_num(pot.reindex(index=rows, columns=vre_cols[vre_pos]).to_numpy(dtype=float))
        cap[:, vre_pos] = pot_next
        delta_p = pot_next - pot_now
        del pot_next, pot_now

    up_a = cap - f2
    down_a = f2.copy()
    if low is not None:
        down_a[:, fl_pos] -= low
    del cap, low

    # Coarse processes: the coarse step durations in ρ̄; no room on the rows inside a coarse step.
    rate_row = 60.0 * (dur + dur_next) / 2.0
    co_pos, inside = np.zeros(0, dtype=int), None
    coarse_dur = par.process_coarse_step_duration_dt
    if coarse_dur.shape[1] > 0:
        co_pos = np.flatnonzero(procs.isin(coarse_dur.columns))
        co_procs = procs[co_pos]
        cdur_now = coarse_dur.reindex(index=rows, columns=co_procs).to_numpy(dtype=float)
        cdur_next = nx.frame(coarse_dur, columns=co_procs).to_numpy(dtype=float)
        last = par.process_coarse_step_last_dt.reindex(index=rows, columns=co_procs).to_numpy(dtype=float)
        coarse_cell = ~np.isnan(cdur_now)
        coarse_rate = np.where(coarse_cell, 60.0 * (cdur_now + np.nan_to_num(cdur_next)) / 2.0, rate_row[:, None])
        inside = coarse_cell & (last == 0.0)

    # Ramp caps of the ramp-limit sets of the solve that realized t⁺: min with the capacity room, per direction.
    in_period = nx.in_period[:, None]
    for direction, labels in _LIMITS.items():
        target = up_a if direction == "up" else down_a
        for label in labels:
            sp_pos, speed = _arc_columns(_level_frame(par.process_ramp_limit_dt, label, None), arcs)
            if not len(sp_pos):
                continue
            speed = nx.frame(speed).to_numpy(dtype=float)
            coef = _level_frame(par.process_ramp_cap_coef_dt, label, arcs[sp_pos])
            coef = np.nan_to_num(nx.frame(coef).to_numpy(dtype=float), nan=1.0)
            rate = np.repeat(rate_row[:, None], len(sp_pos), axis=1)
            if len(co_pos):
                at = pd.Index(co_pos).get_indexer(sp_pos)
                hit = at >= 0
                rate[:, hit] = coarse_rate[:, at[hit]]
            room = speed * rate
            applies = ~np.isnan(speed) & (~in_period | (room < 1.0))
            limit = np.where(applies, room * coef * cap_u[:, sp_pos], np.nan)
            target[:, sp_pos] = np.fmin(target[:, sp_pos], limit)
    del cap_u

    # Reserve held at t⁺ toward each direction of the arc: R↑ = up at the sink + φ · down at the source, R↓ the
    # mirror; φ = 1/slope(t⁺) on a direct arc.
    res = _reserve_by_node(r, nx)
    direct_pos = np.flatnonzero(is_direct)
    slope_next_d = np.nan_to_num(nx.frame(par.process_slope, columns=procs[direct_pos]).to_numpy(dtype=float),
                                 nan=1.0)
    for target, sink_dir, source_dir in ((up_a, "up", "down"), (down_a, "down", "up")):
        for node, ud, at_source in ((snk, sink_dir, False), (src, source_dir, True)):
            keys = pd.MultiIndex.from_arrays([p, np.full(n_arcs, ud, dtype=object), node])
            pos = np.flatnonzero(np.asarray(keys.isin(res.columns)))
            if not len(pos):
                continue
            held = res.reindex(columns=keys[pos]).to_numpy(dtype=float)
            held = np.nan_to_num(held)
            if at_source:
                at = pd.Index(direct_pos).get_indexer(pos)
                hit = at >= 0
                held[:, hit] = held[:, hit] / slope_next_d[:, at[hit]]
            target[:, pos] -= held

    # Node rooms per arc end: an output ×1 (a VRE output measured against its potential), an indirect unit's
    # input flipped, a direct unit's input converted with the slope and section of each step.
    sink_idx = np.flatnonzero(~is_in)
    in_idx = np.flatnonzero(is_in)
    sink_up, sink_down = up_a[:, sink_idx], down_a[:, sink_idx]
    if delta_p is not None:
        at = pd.Index(sink_idx).get_indexer(vre_pos)
        sink_up[:, at] -= delta_p
        sink_down[:, at] += delta_p
    slope_now_d = np.nan_to_num(par.process_slope.reindex(index=rows, columns=procs[direct_pos])
                                .to_numpy(dtype=float), nan=1.0)
    f2_d, up_d, down_d = f2[:, direct_pos], up_a[:, direct_pos], down_a[:, direct_pos]
    direct_up = (slope_now_d - slope_next_d) * f2_d + slope_next_d * down_d
    direct_down = (slope_next_d - slope_now_d) * f2_d + slope_next_d * up_d
    ct = s.process__ct_method
    mle = ct.get_level_values("process")[ct.get_level_values("method") == "min_load_efficiency"] if len(ct) else []
    sec_pos = np.flatnonzero(procs[direct_pos].isin(mle))
    if len(sec_pos) and len(on_pos):
        arc_pos = direct_pos[sec_pos]
        at = pd.Index(on_pos).get_indexer(arc_pos)
        hit = at >= 0
        sec_units = np.zeros((n_rows, len(arc_pos)))
        sec_units[:, hit] = units_both[:, at[hit]] * unitsize[arc_pos[hit]][None, :]
        sec_procs = procs[arc_pos]
        sec_now = np.nan_to_num(par.process_section.reindex(index=rows, columns=sec_procs).to_numpy(dtype=float))
        sec_next = np.nan_to_num(nx.frame(par.process_section, columns=sec_procs).to_numpy(dtype=float))
        direct_up[:, sec_pos] += (sec_now - sec_next) * sec_units
        direct_down[:, sec_pos] += (sec_next - sec_now) * sec_units
    del f2, f2_d, up_d, down_d

    up_v = np.concatenate([sink_up, down_a[:, in_idx], direct_up], axis=1)
    down_v = np.concatenate([sink_down, up_a[:, in_idx], direct_down], axis=1)
    del up_a, down_a, sink_up, sink_down, direct_up, direct_down
    end_arc = np.concatenate([sink_idx, in_idx, direct_pos])
    if inside is not None:
        at = pd.Index(co_pos).get_indexer(end_arc)
        hit = np.flatnonzero(at >= 0)
        cells = inside[:, at[hit]]
        up_v[:, hit] = np.where(cells, 0.0, up_v[:, hit])
        down_v[:, hit] = np.where(cells, 0.0, down_v[:, hit])
    nodes = np.concatenate([np.asarray(snk)[sink_idx], np.asarray(src)[in_idx], np.asarray(src)[direct_pos]])
    owner = np.asarray(p)[end_arc]
    vre_end = np.zeros(len(end_arc), dtype=bool)
    vre_end[pd.Index(sink_idx).get_indexer(vre_pos)] = True
    storage = _storage_of(s, owner)
    band = np.where(storage != "", "storage", np.where(vre_end, "vre", "units"))
    meta = pd.DataFrame({"process": owner, "node": nodes, "band": band, "storage": storage})
    cols = pd.RangeIndex(len(meta))
    return _wide(up_v, rows, cols), _wide(down_v, rows, cols), meta


def _storage_of(s, procs) -> np.ndarray:
    """The storage node of each process in ``procs`` (``""`` for none; ``"\\x00multi"`` for a process that
    touches more than one storage node: storage band without a state-of-charge cap)."""
    ends = _process_nodes(s).drop_duplicates(["process", "node"])
    ends = ends[ends["node"].isin(s.node_state)]
    counts = ends.groupby("process")["node"].agg(["first", "nunique"])
    key = counts["first"].where(counts["nunique"] == 1, "\x00multi")
    return pd.Series(np.asarray(procs, dtype=object)).map(key).fillna("").to_numpy(dtype=object)


def _reserve_by_node(r, nx: _Next) -> pd.DataFrame:
    """Reserve held at t⁺ per ``(process, updown, node)`` (MW, summed over the reserves)."""
    res = r.reserves_dt
    if res is None or res.shape[1] == 0:
        return pd.DataFrame(index=nx.rows, columns=pd.MultiIndex.from_tuples([], names=["process", "updown", "node"]))
    summed = res.T.groupby(level=["process", "updown", "node"]).sum().T
    return nx.frame(summed)


def _connection_contributions(par, s, r, nx: _Next):
    """Raw node rooms of the connection ends: ``(up, down, meta)`` as :func:`_unit_contributions`."""
    rows = r.flow_dt.index
    fin = pd.concat([r.connection_to_left_node__dt, r.connection_to_right_node__dt], axis=1)
    ends = _process_nodes(s).drop_duplicates(["process", "node"])
    ends = ends[ends["process"].isin(s.process_connection)].reset_index(drop=True)[["process", "node"]]
    if ends.empty:
        return (pd.DataFrame(index=rows), pd.DataFrame(index=rows),
                pd.DataFrame(columns=["process", "node", "band", "storage"]))
    conn = pd.Index(ends["process"])
    keys = pd.MultiIndex.from_arrays([ends["process"], ends["node"]], names=["process", "node"])
    fin = fin.loc[:, ~fin.columns.duplicated()] if fin.shape[1] else fin
    f_in = np.nan_to_num(fin.reindex(index=rows, columns=keys).to_numpy(dtype=float))

    unitsize = par.entity_unitsize.reindex(conn).to_numpy(dtype=float)
    online_next = nx.frame(r.process_online_dt, columns=conn).to_numpy(dtype=float)
    startup_next = np.nan_to_num(nx.frame(r.process_startup_dt, columns=conn).to_numpy(dtype=float))
    online = (np.asarray(conn.isin(s.process_online))[None, :] & ~np.isnan(online_next)
              & ~_reduced_online(s, nx.index, conn))
    built = nx.by_period(r.entity_all_capacity, conn).to_numpy(dtype=float)
    cap = np.where(online, (online_next - startup_next) * unitsize[None, :], built)
    avail = np.nan_to_num(nx.frame(par.process_availability, columns=conn).to_numpy(dtype=float), nan=1.0)
    slope = np.nan_to_num(nx.frame(par.process_slope, columns=conn).to_numpy(dtype=float), nan=1.0)
    res = _reserve_by_node(r, nx)
    r_up = np.nan_to_num(res.reindex(columns=pd.MultiIndex.from_arrays(
        [ends["process"], ["up"] * len(ends), ends["node"]])).to_numpy(dtype=float))
    r_down = np.nan_to_num(res.reindex(columns=pd.MultiIndex.from_arrays(
        [ends["process"], ["down"] * len(ends), ends["node"]])).to_numpy(dtype=float))
    up = cap * avail - f_in - r_up
    down = f_in + cap * avail * slope - r_down
    coarse_dur = par.process_coarse_step_duration_dt
    if coarse_dur.shape[1] > 0:
        last = par.process_coarse_step_last_dt.reindex(index=rows, columns=conn).to_numpy(dtype=float)
        inside = last == 0.0
        up = np.where(inside, 0.0, up)
        down = np.where(inside, 0.0, down)

    # The far end: a storage node puts the connection in the storage band.
    other = ends.merge(ends, on="process", suffixes=("", "_other"))
    other = other[other["node"] != other["node_other"]]
    far_storage = other[other["node_other"].isin(s.node_state)].drop_duplicates(["process", "node"])
    storage = (ends.merge(far_storage[["process", "node", "node_other"]], on=["process", "node"], how="left")
               ["node_other"].fillna("").to_numpy(dtype=object))
    meta = pd.DataFrame({"process": ends["process"].to_numpy(dtype=object), "node": ends["node"].to_numpy(dtype=object),
                         "band": np.where(storage != "", "storage", "connections"), "storage": storage})
    cols = pd.RangeIndex(len(meta))
    return _wide(up, rows, cols), _wide(down, rows, cols), meta


def _membership(meta: pd.DataFrame, env_nodes: pd.Index, gn: pd.DataFrame, proc_nodes: pd.DataFrame) -> pd.DataFrame:
    """``(contribution, kind, entity)``: each arc end counts for its node (an envelope node) and for every
    dispatch group that has the node as a non-storage member, unless the process is internal to the group: it
    has an input node and an output node, and every node of it is a non-storage member (a unit with an output
    alone, or with a node outside the group, is not internal)."""
    contrib = meta.reset_index().rename(columns={"index": "contribution"})
    at_node = contrib[contrib["node"].isin(env_nodes)][["contribution"]].assign(kind="node")
    at_node["entity"] = contrib.loc[at_node["contribution"], "node"].to_numpy()
    in_group = contrib.merge(gn, on="node")[["contribution", "process", "group"]]
    nodes = proc_nodes.drop_duplicates(["process", "node"])
    n_nodes = nodes.groupby("process")["node"].nunique()
    n_inside = nodes.merge(gn, on="node").groupby(["process", "group"])["node"].nunique()
    both_sides = proc_nodes[proc_nodes["side"] == "source"]["process"].unique()
    both_sides = pd.Index(both_sides).intersection(proc_nodes[proc_nodes["side"] == "sink"]["process"].unique())
    inside_procs = n_inside.index.get_level_values("process")
    whole = (n_inside.to_numpy() == n_nodes.reindex(inside_procs).to_numpy()) & np.asarray(inside_procs.isin(both_sides))
    internal = n_inside.index[whole]
    keys = pd.MultiIndex.from_arrays([in_group["process"], in_group["group"]])
    in_group = in_group[~keys.isin(internal)]
    at_group = in_group[["contribution"]].assign(kind="group", entity=in_group["group"].to_numpy())
    return pd.concat([at_node, at_group], ignore_index=True)


def _matrix(mem: pd.DataFrame, n_contrib: int, columns: pd.Index) -> np.ndarray:
    """0/1 matrix ``(contribution, column)`` of ``mem`` rows (``contribution``, ``column``)."""
    out = np.zeros((n_contrib, len(columns)))
    if len(mem):
        out[mem["contribution"].to_numpy(dtype=int), columns.get_indexer(mem["column"])] = 1.0
    return out


def _band_sums(up, down, meta, mem, entities, storage_cap, rows) -> dict:
    """The band frames (rows × entities; down bands negative) of one entity kind."""
    out = {}
    entity_index = pd.Index(entities)
    mem = mem.merge(meta[["band", "storage", "process"]], left_on="contribution", right_index=True)
    up_v, down_v = up.to_numpy(), down.to_numpy()
    for band, up_name, down_name in (("units", "units_up", "units_down"), ("vre", "vre_up", "vre_down"),
                                     ("connections", "connections_up", "connections_down")):
        sel = mem[mem["band"] == band].assign(column=lambda m: m["entity"])
        matrix = _matrix(sel, up.shape[1], entity_index)
        out[up_name] = _wide(up_v @ matrix, rows, entity_index)
        out[down_name] = _wide(-(down_v @ matrix), rows, entity_index)
    # Storage: the arcs of each (storage node, entity) pair, capped by the state of charge of the node.
    sel = mem[mem["band"] == "storage"]
    pairs = pd.MultiIndex.from_frame(sel[["storage", "entity"]].drop_duplicates()) if len(sel) else \
        pd.MultiIndex.from_tuples([], names=["storage", "entity"])
    sel = sel.assign(column=list(zip(sel["storage"], sel["entity"])))
    matrix = _matrix(sel, up.shape[1], pairs)
    pair_up = up_v @ matrix
    pair_down = down_v @ matrix
    if len(pairs):
        room_up, room_down = storage_cap(pairs, sel)
        pair_up = np.fmin(pair_up, room_up)
        pair_down = np.fmin(pair_down, room_down)
        entity_of = entity_index.get_indexer(pairs.get_level_values("entity"))
        to_entity = np.zeros((len(pairs), len(entity_index)))
        to_entity[np.arange(len(pairs)), entity_of] = 1.0
        store_up, store_down = pair_up @ to_entity, pair_down @ to_entity
    else:
        store_up = store_down = np.zeros((len(rows), len(entity_index)))
    out["storage_up"] = _wide(store_up, rows, entity_index)
    out["storage_down"] = _wide(-store_down, rows, entity_index)
    return out


def _storage_capacity(par, s, v, r, nx: _Next, rows, dur_next):
    """A function ``(pairs, sel) -> (room_up, room_down)`` (rows × pairs, entity MW): the state-of-charge cap of
    each (storage node, entity) pair, clip0(state(t)/Δ⁺ − D(t))·η_out and clip0((cap(d⁺) − state(t))/Δ⁺ +
    D(t))/η_in; ``inf`` for the processes that touch several storage nodes (no cap)."""
    stores = pd.Index(sorted(s.node_state), name="node")
    flow = r.flow_dt
    if len(stores):
        src = flow.columns.get_level_values("source")
        snk = flow.columns.get_level_values("sink")
        out_of = flow.loc[:, src.isin(stores)].T.groupby(level="source").sum().T.reindex(columns=stores)
        into = flow.loc[:, snk.isin(stores)].T.groupby(level="sink").sum().T.reindex(columns=stores)
        draw = (out_of.fillna(0.0) - into.fillna(0.0)).reindex(rows).to_numpy(dtype=float)
        unitsize = par.entity_unitsize.reindex(stores).to_numpy(dtype=float)
        state = v.state.reindex(index=rows, columns=stores).to_numpy(dtype=float) * unitsize[None, :]
        capacity = nx.by_period(r.entity_all_capacity, stores).to_numpy(dtype=float)
        room_up_s = np.clip(state / dur_next[:, None] - draw, 0.0, None)
        room_down_s = np.clip((capacity - state) / dur_next[:, None] + draw, 0.0, None)
    else:
        room_up_s = room_down_s = np.zeros((len(rows), 0))
    pss = s.process_source_sink
    delivers = pd.MultiIndex.from_arrays([pss.get_level_values("process"), pss.get_level_values("source")])
    receives = pd.MultiIndex.from_arrays([pss.get_level_values("process"), pss.get_level_values("sink")])
    connections = s.process_connection
    slope_next = nx.frame(par.process_slope)

    def caps(pairs: pd.MultiIndex, sel):
        n = len(pairs)
        room_up = np.full((len(rows), n), np.inf)
        room_down = np.full((len(rows), n), np.inf)
        store = pairs.get_level_values("storage")
        known = np.asarray(store.isin(stores))
        position = stores.get_indexer(store[known])
        room_up[:, known] = room_up_s[:, position]
        room_down[:, known] = room_down_s[:, position]
        # η of the processes of each pair: delivering from S (η_out) and into S (η_in); a pair without one of them
        # takes the efficiency of its other processes.
        procs = sel.drop_duplicates(["storage", "entity", "process"])
        eta = 1.0 / slope_next.reindex(columns=pd.Index(procs["process"])).fillna(1.0).to_numpy(dtype=float)
        col = pairs.get_indexer(pd.MultiIndex.from_arrays([procs["storage"], procs["entity"]]))
        proc_store = pd.MultiIndex.from_arrays([procs["process"], procs["storage"]])
        is_conn = np.asarray(procs["process"].isin(connections))
        out_ok = np.asarray(proc_store.isin(delivers)) | is_conn
        in_ok = np.asarray(proc_store.isin(receives)) | is_conn
        eta_out = _group_reduce(np.where(out_ok[None, :], eta, np.nan), col, n, "max")
        eta_out = np.where(np.isnan(eta_out), _group_reduce(eta, col, n, "max"), eta_out)
        eta_in = _group_reduce(np.where(in_ok[None, :], eta, np.nan), col, n, "min")
        eta_in = np.where(np.isnan(eta_in), _group_reduce(eta, col, n, "min"), eta_in)
        return room_up * eta_out, room_down / eta_in

    return caps


def _group_reduce(values: np.ndarray, col: np.ndarray, n: int, how: str) -> np.ndarray:
    """``rows × n``: the ``how`` (``"max"`` / ``"min"``, NaN skipped) over the columns of ``values`` that map to
    each of ``n`` groups (``col``: the group of each column); NaN for a group without a value."""
    frame = pd.DataFrame(values.T)
    reduced = frame.groupby(col).agg(how).reindex(range(n))
    return reduced.to_numpy(dtype=float).T


def _lines(par, r, nx: _Next, rows, dur, env_nodes: pd.Index, gn: pd.DataFrame) -> dict:
    """``{"node": {"net", "demand"}, "group": {...}}``: ramp_net_load and ramp_demand per entity (rows × entity)."""
    nodes = pd.Index(sorted(set(env_nodes) | set(gn["node"])), name="node")
    demand = -par.node_inflow.reindex(index=rows, columns=nodes).fillna(0.0).div(dur, axis=0)
    pot = r.potentialVREgen_dt
    if pot.shape[1]:
        pot_n = pot.reindex(rows).T.groupby(level="node").sum().T.reindex(columns=nodes).fillna(0.0)
    else:
        pot_n = pd.DataFrame(0.0, index=rows, columns=nodes)
    net = demand - pot_n
    group_matrix = pd.crosstab(gn["node"], gn["group"]).reindex(index=nodes, fill_value=0)
    out = {}
    for kind in ("node", "group"):
        frames = {}
        for name, level in (("net", net), ("demand", demand)):
            if kind == "group":
                level = pd.DataFrame(level.to_numpy() @ group_matrix.to_numpy(dtype=float), index=rows,
                                     columns=group_matrix.columns)
            frames[name] = nx.frame(level) - level
        out[kind] = frames
    return out
