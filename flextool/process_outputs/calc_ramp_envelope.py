"""Ramp envelope: the room each node / dispatch node group has to change its supply toward the next step.

Row t of the tables is the transition t → t⁺ (t⁺ the unique realized successor of t, the inverse of
``s.ramp_link`` incl. the roll carry).  Per row and entity, in MW (``specs/ramp_envelope_final_spec.md`` §2):

* ``ramp_net_load`` = N(t⁺) − N(t), N = demand − Σ VRE potential (+ = more supply needed; the potential of a VRE
  unit that touches a storage node is left out, as its arcs are storage arcs); ``ramp_demand`` the same for the
  demand alone (demand = −inflow);
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
    f₂  = f(t) − F(t) · unitsize · shutdown(t⁺)      (the stopping units' output is forced movement; on a link
                                                     carried from the previous roll f₂ = f: the model's carried
                                                     ramp row has no shut-down floor term)
    up_a   = min(C − f₂, ρ̄↑) − R↑(t⁺),   down_a = min(f₂ − L, ρ̄↓) − R↓(t⁺)
    ρ̄   = speed · 60 · (Δ + Δ⁺)/2 · c(t⁺) · U        for the arcs of the ramp-limit sets of the solve that
                                                     realized t⁺; in one period only where speed·60·(Δ+Δ⁺)/2 < 1
    R↑  = Σ_r reserve_up(sink) + φ · Σ_r reserve_down(source),  R↓ the mirror,  φ = 1/slope(t⁺) on a direct arc

Node conversion: an output arc counts ×1; an indirect unit's input arc flips direction (up_n = down_a, down_n =
up_a); a direct unit's input arc converts each step with its own slope and section (CHG-12)::

    up_n   = (slope(t) − slope(t⁺)) · f₂ + slope(t⁺) · down_a + (section(t) − section(t⁺)) · U
    down_n = (slope(t⁺) − slope(t)) · f₂ + slope(t⁺) · up_a  + (section(t⁺) − section(t)) · U

VRE arcs (an output with an ``upper_limit`` profile, §2.3) take C = pot(t⁺) and are measured against the
potential: vre_up = clip0(up_a − ΔP), vre_down = clip0(down_a + ΔP), ΔP = pot(t⁺) − pot(t); a VRE unit that touches
a storage node is in the storage band, without the potential shift.  Storage arcs (§2.4, processes between a
storage node S and the entity) are capped per S by the state of charge::

    up ≤ clip0(state⁻(t⁺)/Δ⁺ − D(t)) · η_out,   down ≤ clip0((capacity_S(d⁺) − state⁻(t⁺))/Δ⁺ + D(t)) / η_in

(state⁻(t⁺) the state that the storage balance starts t⁺ from, ``s.node_state_prev``: the end-of-step state of t,
or the step a binding method relinks t⁺ to, e.g. bind_within_timeblock at a timeblock's first step; the state of t
where t⁺ has no such link; D the net draw from S at t, η = 1/slope(t⁺) of the processes delivering from / to S).
Connections (§2.5) use the node-side range of their flow at t⁺, losses on the drawing side, with the reserve they
hold at the node::

    up = cap·avail(t⁺) − f_in(t) − R↑_c,n(t⁺),   down = f_in(t) + cap·avail(t⁺)·slope(t⁺) − R↓_c,n(t⁺)

A coarse process (``processCoarse``) has one flow per coarse step: its room is 0 on the rows inside a coarse step,
and at a coarse step's last fine row ρ̄ uses the coarse step durations, and κ, F and c are the coarse steps' means
(``_ramp_inputs`` writes them to every fine step of the coarse step).  Node groups (``print_dispatch``) sum the arcs
at their non-storage members; a process with every node in the group is internal and counts 0; member storage
nodes count as outside (§2.7).  The arcs are summed into the entities with one membership-map sum (``np.add.at``)
per band and direction.

The arcs with ``conversion_flow_coeff = 0`` (``process_source_sink_uncapped``: outside every per-arc capacity and
ramp row) have no room of their own and are left out; a process that touches more than one storage node is in the
storage band without a state-of-charge cap (the cap is per storage node).

The rows are computed in blocks (the rows of a period, at most :data:`_BLOCK_ROWS` at a time): every row reads only
its own step and t⁺, so the blocks are independent and only block-size arrays of rows × arcs are held.
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
#: The most rows computed at a time.
_BLOCK_ROWS = 2048


class _Next:
    """The realized successor t⁺ of every row (spec §2.8): the inverse of ``s.ramp_link``, rows with no or with
    several successors have none; and the solve that realized t⁺ (``s.solve_period_time``)."""

    def __init__(self, rows: pd.MultiIndex, ramp_link: pd.MultiIndex, ramp_carry: "pd.MultiIndex | None",
                 solve_period_time: "pd.MultiIndex | None" = None):
        self.rows = rows
        #: The positions of the rows in the full row set (a block of :meth:`take` holds a part of them).
        self.positions = np.arange(len(rows))
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
        self.time_next = np.where(self.has, succ["time"].to_numpy(dtype=object), _SENTINEL)
        self.index = pd.MultiIndex.from_arrays([self.period_next, self.time_next], names=["period", "time"])
        periods = rows.get_level_values("period").to_numpy(dtype=object)
        carried = np.zeros(len(rows), dtype=bool)
        if ramp_carry is not None and len(ramp_carry):
            carry_pairs = pd.MultiIndex.from_arrays(
                [ramp_carry.get_level_values("period"), ramp_carry.get_level_values("time"),
                 ramp_carry.get_level_values("period_previous"), ramp_carry.get_level_values("time_previous")])
            pairs = pd.MultiIndex.from_arrays(
                [self.period_next, self.time_next, periods, rows.get_level_values("time").to_numpy(dtype=object)])
            carried = np.asarray(pairs.isin(carry_pairs), dtype=bool)
        #: Rows whose link is carried from the previous roll's last realized step (the model's carried ramp row).
        self.carried = carried
        #: Rows whose link stays inside one period and is not carried from a previous roll: the ramp limit
        #: applies there only where the ramp room per unit of capacity is below 1.
        self.in_period = self.has & (self.period_next == periods) & ~carried
        #: The solve that realized t⁺ (``_SENTINEL`` without t⁺ or without a realizing solve).
        self.solve_next = np.full(len(rows), _SENTINEL, dtype=object)
        if solve_period_time is not None and len(solve_period_time):
            pt = solve_period_time.droplevel("solve")
            at = pt.get_indexer(self.index)
            hit = at >= 0
            self.solve_next[hit] = solve_period_time.get_level_values("solve").to_numpy(dtype=object)[at[hit]]

    def take(self, positions: np.ndarray) -> "_Next":
        """The rows at ``positions`` (a block)."""
        out = object.__new__(_Next)
        out.rows = self.rows[positions]
        for name in ("positions", "has", "period_next", "time_next", "carried", "in_period", "solve_next"):
            setattr(out, name, getattr(self, name)[positions])
        out.index = self.index[positions]
        return out

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


def _blocks(rows: pd.MultiIndex) -> list[np.ndarray]:
    """Row positions in blocks: the rows of each period (in row order), at most :data:`_BLOCK_ROWS` each."""
    if not len(rows):
        return []
    codes = pd.factorize(rows.get_level_values("period"))[0]
    order = np.argsort(codes, kind="stable")
    out = []
    for positions in np.split(order, np.flatnonzero(np.diff(codes[order])) + 1):
        out += [positions[i:i + _BLOCK_ROWS] for i in range(0, len(positions), _BLOCK_ROWS)]
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


def _clip0(values: np.ndarray) -> np.ndarray:
    """Per arc clip at 0, NaN (an arc missing from a solve: no room) as 0."""
    return np.where(values >= 0.0, values, 0.0)


def compute_ramp_envelope(par, s, v, r) -> None:
    """``r.node_ramp_envelope_dt_ep`` (node, band), ``r.nodeGroup_ramp_envelope_dt_gp`` (group, band),
    ``r.timestep_hours_dt`` (the hours a year each step stands for: Δt · timestep_weight / period share) and
    ``r.transition_hours_dt`` (the length (Δt + Δt⁺)/2 of each row's transition, h; empty without t⁺)."""
    rows = r.flow_dt.index
    nx_all = _Next(rows, s.ramp_link, getattr(s, "ramp_carry", None), s.solve_period_time)
    dur_all = par.step_duration.reindex(rows).to_numpy(dtype=float)
    dur_next_all = nx_all.series(par.step_duration).to_numpy(dtype=float)
    storage_nodes = set(s.node_state)
    proc_nodes = _process_nodes(s)
    env_nodes = s.node_balance.union(s.node_balance_period).difference(pd.Index(sorted(storage_nodes)))
    groups = s.nodeGroupDispatch
    gn = s.group_node.to_frame(index=False)
    gn = gn[gn["group"].isin(groups) & ~gn["node"].isin(storage_nodes)]
    kinds = (("node", list(env_nodes), "node"), ("group", list(groups), "group"))
    values = {kind: np.full((len(rows), len(entities), len(BANDS)), np.nan) for kind, entities, _ in kinds}

    shared = _Shared(par, s, v, r, nx_all, dur_next_all)
    matrices = None
    for positions in _blocks(rows):
        nx = nx_all.take(positions)
        dur, dur_next = dur_all[positions], dur_next_all[positions]
        unit_up, unit_down, unit_meta = _unit_contributions(par, s, r, nx, dur, dur_next, shared)
        conn_up, conn_down, conn_meta = _connection_contributions(par, s, r, nx, shared)
        up = np.concatenate([unit_up, conn_up], axis=1)
        down = np.concatenate([unit_down, conn_down], axis=1)
        del unit_up, unit_down, conn_up, conn_down
        meta = pd.concat([unit_meta, conn_meta], ignore_index=True)
        # Storage nodes have no envelope of their own: the arc ends at them are dropped.
        keep = ~meta["node"].isin(storage_nodes).to_numpy()
        up, down, meta = up[:, keep], down[:, keep], meta[keep].reset_index(drop=True)
        # Clip per arc, last (spec §2.0); an arc missing from a solve has no room.
        up, down = _clip0(up), _clip0(down)
        if matrices is None:
            membership = _membership(meta, env_nodes, gn, proc_nodes)
            matrices = {kind: _band_matrices(meta, membership[membership["kind"] == kind], entities)
                        for kind, entities, _ in kinds if entities}
        storage_cap = _storage_capacity(par, s, nx, shared, positions)
        for kind, entities, _ in kinds:
            if not entities:
                continue
            bands = _band_sums(up, down, matrices[kind], storage_cap)
            for position, band in enumerate(BANDS[2:], start=2):
                values[kind][positions, :, position] = bands[band]
        del up, down

    lines = _lines(par, s, r, nx_all, rows, dur_all, env_nodes, gn)
    for kind, entities, level in kinds:
        cols = pd.MultiIndex.from_product([entities, BANDS], names=[level, "band"])
        out_values = values[kind]
        if entities:
            for position, name in ((0, "net"), (1, "demand")):
                out_values[:, :, position] = lines[kind][name].reindex(columns=entities).to_numpy(dtype=float)
            out_values[~nx_all.has] = np.nan
        out = pd.DataFrame(out_values.reshape(len(rows), len(entities) * len(BANDS)), index=rows, columns=cols)
        if kind == "node":
            r.node_ramp_envelope_dt_ep = out
        else:
            r.nodeGroup_ramp_envelope_dt_gp = out
    del values

    weight = par.timestep_weight.reindex(rows).to_numpy(dtype=float)
    share = par.complete_period_share_of_year.reindex(rows.get_level_values("period")).to_numpy(dtype=float)
    hours = pd.DataFrame({"timestep_hours": dur_all * weight / share}, index=rows)
    hours.columns.name = "param"
    r.timestep_hours_dt = hours
    # The length of each row's transition, (Δt + Δt⁺)/2 (the ramp histogram's rate is the line over it).
    transition = pd.DataFrame(
        {"transition_hours": np.where(nx_all.has, (dur_all + dur_next_all) / 2.0, np.nan)}, index=rows)
    transition.columns.name = "param"
    r.transition_hours_dt = transition


class _Shared:
    """The inputs every block reads, prepared once: the reserves at each ``(process, updown, node)``, the ramp
    speeds per ramp-limit set (rows of ``par.process_ramp_limit`` at the solve that realized each row's t⁺), and the
    state-of-charge terms of the storage nodes."""

    def __init__(self, par, s, v, r, nx: _Next, dur_next: np.ndarray):
        res = r.reserves_dt
        if res is None or res.shape[1] == 0:
            self.reserves = pd.DataFrame(
                index=r.flow_dt.index, columns=pd.MultiIndex.from_tuples([], names=["process", "updown", "node"]))
        else:
            self.reserves = res.T.groupby(level=["process", "updown", "node"]).sum().T
        # Ramp speeds: per label, (rows of the frame, arc columns); each row's t⁺ reads the row of its solve.
        limits = par.process_ramp_limit
        self.limit_columns: dict = {}
        self.limit_values: dict = {}
        if limits is not None and limits.shape[1] > 0 and len(limits):
            key = pd.MultiIndex.from_arrays([nx.solve_next, nx.period_next])
            self.limit_row = limits.index.get_indexer(key)
            for label in limits.columns.get_level_values("limit").unique():
                part = limits.xs(label, level="limit", axis=1)
                self.limit_columns[label] = part.columns
                self.limit_values[label] = part.to_numpy(dtype=float)
        else:
            self.limit_row = np.full(len(nx.rows), -1)
        self.storage = _storage_state(par, s, v, r, nx, dur_next)

    def speeds(self, label: str, arcs: pd.MultiIndex, positions: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
        """``(positions in arcs, rows × those arcs)``: the ramp speed of the ramp-limit set ``label`` toward t⁺ of
        the rows at ``positions``, NaN where the solve that realized t⁺ has no such limit (or no t⁺)."""
        columns = self.limit_columns.get(label)
        if columns is None:
            return np.zeros(0, dtype=int), np.zeros((len(positions), 0))
        at = arcs.get_indexer(columns)
        keep = np.flatnonzero(at >= 0)
        if not len(keep):
            return np.zeros(0, dtype=int), np.zeros((len(positions), 0))
        row = self.limit_row[positions]
        out = np.full((len(positions), len(keep)), np.nan)
        hit = row >= 0
        out[hit] = self.limit_values[label][row[hit]][:, keep]
        return at[keep], out


def _arc_columns(frame: "pd.DataFrame | None", arcs: pd.MultiIndex) -> "tuple[np.ndarray, pd.DataFrame | None]":
    """``(positions in arcs, frame restricted to those columns)`` of the columns of ``frame`` that are arcs."""
    if frame is None or frame.shape[1] == 0:
        return np.zeros(0, dtype=int), None
    position = arcs.get_indexer(frame.columns)
    keep = position >= 0
    return position[keep], frame.loc[:, keep]


def _unit_contributions(par, s, r, nx: _Next, dur, dur_next, shared: _Shared):
    """Raw per-arc node rooms of the unit arcs of the block ``nx``: ``(up, down, meta)``, up / down ``rows ×
    contributions`` arrays, meta one row per contribution (process, node, band, storage).  Full ``rows × arcs``
    arrays only for the flow, the capacity and the two rooms; every other attribute on the columns that have it."""
    rows = nx.rows
    n_rows = len(rows)
    pss = s.process_source_sink
    pss = pss[pss.get_level_values("process").isin(s.process_unit)]
    unc = getattr(s, "process_source_sink_uncapped", None)
    if unc is not None and len(unc):
        pss = pss[~pss.isin(unc)]
    empty = (np.zeros((n_rows, 0)), np.zeros((n_rows, 0)),
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
    n_arcs = len(arcs)
    arc_storage = _storage_of(s, p)

    # f: the arc's v_flow in MW — a direct unit's single flow is its output (flow_dt at the sink).
    flow_keys = pd.MultiIndex.from_arrays([p, np.where(is_in, src, p), np.where(is_in, p, snk)])
    f2 = np.nan_to_num(r.flow_dt.reindex(index=rows, columns=flow_keys).to_numpy(dtype=float), nan=0.0)
    unitsize = par.entity_unitsize.reindex(p).to_numpy(dtype=float)

    # U: units online in both steps (online processes not left out by the solve that realized t⁺), else the
    # built capacity at d⁺.
    cap_u = nx.by_period(r.entity_all_capacity, procs).to_numpy(dtype=float, copy=True)
    online = np.zeros((n_rows, n_arcs), dtype=bool)
    shutdown = np.zeros((n_rows, n_arcs))
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
        # A link carried from the previous roll: the model's carried ramp row has no shut-down floor term
        # (core._eq_ramp: F[d′,t′] → 0), so the stopping units' output stays in f₂ there.
        shutdown[:, on_pos] = np.where(on & ~nx.carried[:, None], shutdown_next, 0.0)
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
    del online, shutdown

    # C = κ(t⁺)·U; a VRE arc's capacity toward t⁺ is its potential at t⁺.
    cap = cap_u.copy()
    k_pos, kappa = _arc_columns(par.process_arc_cap_coef_dt, arcs)
    if len(k_pos):
        cap[:, k_pos] *= np.nan_to_num(nx.frame(kappa).to_numpy(dtype=float), nan=1.0)
    pot = r.potentialVREgen_dt
    vre_cols = pd.MultiIndex.from_arrays([p, snk], names=["process", "node"])
    vre_pos = np.flatnonzero(np.asarray(vre_cols.isin(pot.columns)) & ~is_in)
    # The potential frame applies to the VRE arcs of the VRE band only: a VRE unit that touches a storage node is
    # in the storage band.
    shift_pos = vre_pos[arc_storage[vre_pos] == ""]
    delta_p = None
    if len(vre_pos):
        pot_next = np.nan_to_num(nx.frame(pot, columns=vre_cols[vre_pos]).to_numpy(dtype=float))
        cap[:, vre_pos] = pot_next
        if len(shift_pos):
            at = pd.Index(vre_pos).get_indexer(shift_pos)
            pot_now = np.nan_to_num(pot.reindex(index=rows, columns=vre_cols[shift_pos]).to_numpy(dtype=float))
            delta_p = pot_next[:, at] - pot_now
            del pot_now
        del pot_next

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
        del cdur_now, cdur_next, last, coarse_cell

    # Ramp caps of the ramp-limit sets of the solve that realized t⁺: min with the capacity room, per direction.
    in_period = nx.in_period[:, None]
    for direction, labels in _LIMITS.items():
        target = up_a if direction == "up" else down_a
        for label in labels:
            sp_pos, speed = shared.speeds(label, arcs, nx.positions)
            if not len(sp_pos):
                continue
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
            del speed, coef, rate, room, applies, limit
    del cap_u

    # Reserve held at t⁺ toward each direction of the arc: R↑ = up at the sink + φ · down at the source, R↓ the
    # mirror; φ = 1/slope(t⁺) on a direct arc.
    res = nx.frame(shared.reserves)
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
    del res

    # Node rooms per arc end: an output ×1 (a VRE output measured against its potential), an indirect unit's
    # input flipped, a direct unit's input converted with the slope and section of each step.
    sink_idx = np.flatnonzero(~is_in)
    in_idx = np.flatnonzero(is_in)
    sink_up, sink_down = up_a[:, sink_idx], down_a[:, sink_idx]
    if delta_p is not None:
        at = pd.Index(sink_idx).get_indexer(shift_pos)
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
    del f2, f2_d, up_d, down_d, slope_now_d, slope_next_d

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
    vre_end[pd.Index(sink_idx).get_indexer(shift_pos)] = True
    storage = arc_storage[end_arc]
    band = np.where(storage != "", "storage", np.where(vre_end, "vre", "units"))
    meta = pd.DataFrame({"process": owner, "node": nodes, "band": band, "storage": storage})
    return up_v, down_v, meta


def _storage_of(s, procs) -> np.ndarray:
    """The storage node of each process in ``procs`` (``""`` for none; ``"\\x00multi"`` for a process that
    touches more than one storage node: storage band without a state-of-charge cap)."""
    ends = _process_nodes(s).drop_duplicates(["process", "node"])
    ends = ends[ends["node"].isin(s.node_state)]
    counts = ends.groupby("process")["node"].agg(["first", "nunique"])
    key = counts["first"].where(counts["nunique"] == 1, "\x00multi")
    return pd.Series(np.asarray(procs, dtype=object)).map(key).fillna("").to_numpy(dtype=object)


def _connection_contributions(par, s, r, nx: _Next, shared: _Shared):
    """Raw node rooms of the connection ends of the block ``nx``: ``(up, down, meta)`` as
    :func:`_unit_contributions`."""
    rows = nx.rows
    fin = pd.concat([r.connection_to_left_node__dt, r.connection_to_right_node__dt], axis=1)
    ends = _process_nodes(s).drop_duplicates(["process", "node"])
    ends = ends[ends["process"].isin(s.process_connection)].reset_index(drop=True)[["process", "node"]]
    if ends.empty:
        return (np.zeros((len(rows), 0)), np.zeros((len(rows), 0)),
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
    res = nx.frame(shared.reserves)
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
    return up, down, meta


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


class _Map:
    """A 0/1 map ``contribution → column`` (each pair once): ``values @ map`` sums the contributions of every
    column.  ``source`` / ``target`` hold the pairs in contribution order, so each column adds its contributions
    one after the other in that order, from 0."""

    def __init__(self, source: np.ndarray, target: np.ndarray, n_columns: int):
        self.source = source
        self.target = target
        self.n_columns = n_columns


def _matrix(mem: pd.DataFrame, columns: pd.Index) -> _Map:
    """The map ``(contribution, column)`` of ``mem`` rows (``contribution``, ``column``)."""
    if not len(mem):
        return _Map(np.zeros(0, dtype=int), np.zeros(0, dtype=int), len(columns))
    pairs = pd.DataFrame({"source": mem["contribution"].to_numpy(dtype=int),
                          "target": columns.get_indexer(mem["column"])})
    pairs = pairs.drop_duplicates().sort_values(["source", "target"], kind="stable")
    return _Map(pairs["source"].to_numpy(dtype=int), pairs["target"].to_numpy(dtype=int), len(columns))


def _product(values: np.ndarray, matrix: _Map) -> np.ndarray:
    """``values @ matrix`` (rows × contributions summed into rows × columns)."""
    out = np.zeros((matrix.n_columns, values.shape[0]))
    if len(matrix.source):
        np.add.at(out, matrix.target, values.T[matrix.source])
    return out.T


def _band_matrices(meta: pd.DataFrame, mem: pd.DataFrame, entities) -> dict:
    """The membership maps of one entity kind: per band (units, vre, connections) ``contribution ×
    entity``; for the storage band ``contribution × (storage node, entity)`` pairs, ``pair × entity`` and the
    pair selection (``storage``, ``entity``, ``process``) of the state-of-charge cap."""
    entity_index = pd.Index(entities)
    mem = mem.merge(meta[["band", "storage", "process"]], left_on="contribution", right_index=True)
    out = {"entities": entity_index}
    for band in ("units", "vre", "connections"):
        sel = mem[mem["band"] == band].assign(column=lambda m: m["entity"])
        out[band] = _matrix(sel, entity_index)
    sel = mem[mem["band"] == "storage"]
    pairs = pd.MultiIndex.from_frame(sel[["storage", "entity"]].drop_duplicates()) if len(sel) else \
        pd.MultiIndex.from_tuples([], names=["storage", "entity"])
    sel = sel.assign(column=list(zip(sel["storage"], sel["entity"])))
    out["storage"] = _matrix(sel, pairs)
    out["pairs"] = pairs
    out["pair_sel"] = sel
    if len(pairs):
        entity_of = entity_index.get_indexer(pairs.get_level_values("entity"))
        out["to_entity"] = _Map(np.arange(len(pairs)), entity_of, len(entity_index))
    return out


def _band_sums(up: np.ndarray, down: np.ndarray, matrices: dict, storage_cap) -> dict:
    """The band arrays (rows × entities; down bands negative) of one entity kind for a block of rows."""
    out = {}
    for band, up_name, down_name in (("units", "units_up", "units_down"), ("vre", "vre_up", "vre_down"),
                                     ("connections", "connections_up", "connections_down")):
        out[up_name] = _product(up, matrices[band])
        out[down_name] = -_product(down, matrices[band])
    # Storage: the arcs of each (storage node, entity) pair, capped by the state of charge of the node.
    pairs = matrices["pairs"]
    if len(pairs):
        pair_up = _product(up, matrices["storage"])
        pair_down = _product(down, matrices["storage"])
        room_up, room_down = storage_cap(pairs, matrices["pair_sel"])
        pair_up = np.fmin(pair_up, room_up)
        pair_down = np.fmin(pair_down, room_down)
        store_up = _product(pair_up, matrices["to_entity"])
        store_down = _product(pair_down, matrices["to_entity"])
    else:
        store_up = store_down = np.zeros((up.shape[0], len(matrices["entities"])))
    out["storage_up"] = store_up
    out["storage_down"] = -store_down
    return out


def _storage_state(par, s, v, r, nx: _Next, dur_next: np.ndarray) -> "tuple[pd.Index, np.ndarray, np.ndarray]":
    """``(stores, room_up_s, room_down_s)``: the state-of-charge rooms of the storage nodes (rows × stores, S
    units), clip0(state⁻(t⁺)/Δ⁺ − D(t)) and clip0((cap(d⁺) − state⁻(t⁺))/Δ⁺ + D(t)).  state⁻(t⁺) is the state that
    the storage balance starts t⁺ from (``s.node_state_prev``: the end-of-step state of the step it links t⁺ to,
    e.g. a timeblock's last step at the timeblock's first step under bind_within_timeblock), the end-of-step state
    of t where t⁺ has no link (a fixed / carried start) or its linked state is not in the outputs."""
    stores = pd.Index(sorted(s.node_state), name="node")
    rows = nx.rows
    if not len(stores):
        return stores, np.zeros((len(rows), 0)), np.zeros((len(rows), 0))
    flow = r.flow_dt
    src = flow.columns.get_level_values("source")
    snk = flow.columns.get_level_values("sink")
    out_of = flow.loc[:, src.isin(stores)].T.groupby(level="source").sum().T.reindex(columns=stores)
    into = flow.loc[:, snk.isin(stores)].T.groupby(level="sink").sum().T.reindex(columns=stores)
    draw = (out_of.fillna(0.0) - into.fillna(0.0)).reindex(rows).to_numpy(dtype=float)
    unitsize = par.entity_unitsize.reindex(stores).to_numpy(dtype=float)
    state_tab = v.state.reindex(columns=stores)
    state_now = state_tab.reindex(index=rows).to_numpy(dtype=float)
    state = state_now
    prev = getattr(s, "node_state_prev", None)
    if prev is not None and len(prev):
        # One lookup for every (row, store): the step that t⁺ of the row starts the store's state from.
        n_rows, n_stores = len(rows), len(stores)
        key = pd.MultiIndex.from_arrays([np.tile(stores.to_numpy(dtype=object), n_rows),
                                         np.repeat(nx.period_next, n_stores), np.repeat(nx.time_next, n_stores)])
        at = prev.droplevel(["period_prev", "time_prev"]).get_indexer(key)
        hit = at >= 0
        prev_pt = pd.MultiIndex.from_arrays([prev.get_level_values("period_prev").to_numpy(dtype=object)[at[hit]],
                                             prev.get_level_values("time_prev").to_numpy(dtype=object)[at[hit]]])
        state_row = state_tab.index.get_indexer(prev_pt)
        linked = np.full(n_rows * n_stores, np.nan)
        flat_store = np.tile(np.arange(n_stores), n_rows)[hit]
        found = state_row >= 0
        values = state_tab.to_numpy(dtype=float)
        linked_hit = np.full(int(hit.sum()), np.nan)
        linked_hit[found] = values[state_row[found], flat_store[found]]
        linked[hit] = linked_hit
        linked = linked.reshape(n_rows, n_stores)
        state = np.where(np.isnan(linked), state_now, linked)
    state = state * unitsize[None, :]
    capacity = nx.by_period(r.entity_all_capacity, stores).to_numpy(dtype=float)
    room_up_s = np.clip(state / dur_next[:, None] - draw, 0.0, None)
    room_down_s = np.clip((capacity - state) / dur_next[:, None] + draw, 0.0, None)
    return stores, room_up_s, room_down_s


def _storage_capacity(par, s, nx: _Next, shared: _Shared, positions: np.ndarray):
    """A function ``(pairs, sel) -> (room_up, room_down)`` (rows of the block × pairs, entity MW): the
    state-of-charge cap of each (storage node, entity) pair, the rooms of :func:`_storage_state` × η_out and /
    η_in; ``inf`` for the processes that touch several storage nodes (no cap)."""
    stores, room_up_all, room_down_all = shared.storage
    room_up_s, room_down_s = room_up_all[positions], room_down_all[positions]
    n_rows = len(positions)
    pss = s.process_source_sink
    delivers = pd.MultiIndex.from_arrays([pss.get_level_values("process"), pss.get_level_values("source")])
    receives = pd.MultiIndex.from_arrays([pss.get_level_values("process"), pss.get_level_values("sink")])
    connections = s.process_connection

    def caps(pairs: pd.MultiIndex, sel):
        n = len(pairs)
        room_up = np.full((n_rows, n), np.inf)
        room_down = np.full((n_rows, n), np.inf)
        store = pairs.get_level_values("storage")
        known = np.asarray(store.isin(stores))
        position = stores.get_indexer(store[known])
        room_up[:, known] = room_up_s[:, position]
        room_down[:, known] = room_down_s[:, position]
        # η of the processes of each pair: delivering from S (η_out) and into S (η_in); a pair without one of them
        # takes the efficiency of its other processes.
        procs = sel.drop_duplicates(["storage", "entity", "process"])
        slope_next = nx.frame(par.process_slope, columns=pd.Index(procs["process"]))
        eta = 1.0 / slope_next.fillna(1.0).to_numpy(dtype=float)
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


def _lines(par, s, r, nx: _Next, rows, dur, env_nodes: pd.Index, gn: pd.DataFrame) -> dict:
    """``{"node": {"net", "demand"}, "group": {...}}``: ramp_net_load and ramp_demand per entity (rows × entity).
    N = demand − Σ potential of the VRE arcs of the VRE band: a VRE unit that touches a storage node is in the
    storage band (absolute frame), so its potential is left out of N too (line and bands in one frame)."""
    nodes = pd.Index(sorted(set(env_nodes) | set(gn["node"])), name="node")
    demand = -par.node_inflow.reindex(index=rows, columns=nodes).fillna(0.0).div(dur, axis=0)
    pot = r.potentialVREgen_dt
    if pot.shape[1]:
        pot = pot.loc[:, _storage_of(s, pot.columns.get_level_values("process")) == ""]
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
