"""Ramp envelope calculation on hand-built frames (``calc_ramp_envelope.compute_ramp_envelope``).

The dispatches of these cases are hand-chosen (LP-feasible, not LP optima), so the frames are built directly in
the post-``drop_levels`` shapes the calculation reads.  Every expected number is the hand-computed oracle of
``specs/ramp_envelope_oracle_v2.md`` (Cases 1, 2, 3, 4, 4b, 6, 7c, 8c-seam, 11-14), never the code's output; each case also
checks the oracle's realized decomposition (line = Σ of the realized changes, each inside its band).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from flextool.process_outputs.calc_ramp_envelope import BANDS, _Next, compute_ramp_envelope

UP = ("units_up", "storage_up", "vre_up", "connections_up")
DOWN = ("units_down", "storage_down", "vre_down", "connections_down")


def _mi(tuples, names) -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(list(tuples), names=names) if tuples else \
        pd.MultiIndex.from_arrays([[]] * len(names), names=names)


class Frames:
    """The inputs of ``compute_ramp_envelope`` for a hand-built case, in their post-``drop_levels`` shapes."""

    def __init__(self, rows: list[tuple[str, str]], links: list[tuple[tuple[str, str], tuple[str, str]]], *,
                 carry=(), dur: "dict | None" = None, share: float = 1.0, weight: "dict | None" = None,
                 solves: "dict | None" = None):
        self.rows = _mi(rows, ["period", "time"])
        #: The solve that realized each step (default: one solve).
        self.solves = {row: "solve" for row in rows} | (solves or {})
        self.weight = {row: 1.0 for row in rows} | (weight or {})
        self.links = links
        self.carry = carry
        self.dur = {row: 1.0 for row in rows} | (dur or {})
        self.share = share
        self.periods = list(dict.fromkeys(d for d, _ in rows))
        self.flow: dict = {}
        self.pss: list = []
        self.units: list = []
        self.connections: list = []
        self.capacity: dict = {}
        self.unitsize: dict = {}
        self.online: dict = {}
        self.startup: dict = {}
        self.shutdown: dict = {}
        self.floor: dict = {}
        self.speed: dict = {}
        self.kappa: dict = {}
        self.slope: dict = {}
        self.section: dict = {}
        self.methods: list = []
        self.state_prev: list = []
        self.reserve: dict = {}
        self.pot: dict = {}
        self.fin: dict = {}
        self.stores: list = []
        self.state: dict = {}
        self.inflow: dict = {}
        self.balance: list = []
        self.groups: list = []
        self.coarse_dur: dict = {}
        self.coarse_last: dict = {}

    def _series(self, values) -> list:
        values = list(values) if np.ndim(values) else [values] * len(self.rows)
        assert len(values) == len(self.rows)
        return [float(x) for x in values]

    def unit(self, name, *, sink, flow, capacity, source=None, slope=None, speed=None, speed_in=None, online=None,
             startup=None, shutdown=None, unitsize=None, floor=None, reserve=None, pot=None, kappa=None,
             section=None):
        """A unit with one output ``sink`` (and one input ``source``: a direct unit, input draw slope · flow +
        section · online · unitsize; ``section`` makes it a min_load_efficiency unit).  ``speed``: the sink-side
        ramp speed (up and down) in the ramp-limit sets of the solves ``speed_in`` (default every solve);
        ``kappa``: the capacity multiplier of the arc's maxFlow row per step (capacity coefficient ×
        availability)."""
        self.units.append(name)
        flow = self._series(flow)
        arc = (name, source or name, sink)
        self.pss.append(arc)
        self.flow[(name, name, sink)] = flow
        if source is not None:
            sl = self._series(slope)
            self.slope[name] = sl
            draw = [a * b for a, b in zip(flow, sl)]
            if section is not None:
                sec = self._series(section)
                self.section[name] = sec
                self.methods.append((name, "min_load_efficiency"))
                on = self._series(online)
                draw = [x + c * n * (unitsize or 1000.0) for x, c, n in zip(draw, sec, on)]
            self.flow[(name, source, name)] = draw
        self.capacity[name] = capacity
        self.unitsize[name] = unitsize or 1000.0
        if speed is not None:
            solves = set(self.solves.values()) if speed_in is None else set(speed_in)
            for label in ("sink_up", "sink_down"):
                self.speed[(*arc, label)] = (float(speed), solves)
        if kappa is not None:
            self.kappa[arc] = self._series(kappa)
        if online is not None:
            self.online[name] = self._series(online)
            self.startup[name] = self._series(startup if startup is not None else 0.0)
            self.shutdown[name] = self._series(shutdown if shutdown is not None else 0.0)
        if floor is not None:
            self.floor[arc] = self._series(floor)
        for (ud, node), values in (reserve or {}).items():
            self.reserve[(name, "r", ud, node)] = self._series(values)
        if pot is not None:
            self.pot[(name, sink)] = self._series(pot)

    def connection(self, name, a, b, *, f_in_a, capacity):
        """A lossless connection between ``a`` and ``b`` with the signed flow into ``a`` (both directions)."""
        self.connections.append(name)
        self.pss += [(name, a, b), (name, b, a)]
        f = self._series(f_in_a)
        self.fin[(name, a)] = f
        self.fin[(name, b)] = [-x for x in f]
        self.flow[(name, b, name)] = [max(x, 0.0) for x in f]
        self.flow[(name, name, a)] = [max(x, 0.0) for x in f]
        self.flow[(name, a, name)] = [max(-x, 0.0) for x in f]
        self.flow[(name, name, b)] = [max(-x, 0.0) for x in f]
        self.capacity[name] = capacity
        self.unitsize[name] = 1000.0

    def storage(self, name, *, capacity, state, prev=None):
        """A storage node with end-of-step ``state``; ``prev``: ``{step: previous step}``, the step whose state each
        step starts from in the storage balance (``s.node_state_prev``; a step without one starts from a fixed or
        carried state)."""
        self.stores.append(name)
        self.capacity[name] = capacity
        self.unitsize[name] = 1.0
        self.state[name] = self._series(state)
        self.state_prev += [(name, *now, *before) for now, before in (prev or {}).items()]

    def demand(self, node, values):
        self.balance.append(node)
        dur = [self.dur[row] for row in self.rows]
        self.inflow[node] = [-x * d for x, d in zip(self._series(values), dur)]

    def group(self, name, members):
        self.groups += [(name, m) for m in members]

    def coarse(self, name, *, duration, last):
        """``name`` is a coarse process: the duration of the coarse step that holds each step, 1 at its last fine
        step (``process_coarse_step_duration_dt`` / ``process_coarse_step_last_dt``)."""
        self.coarse_dur[name] = self._series(duration)
        self.coarse_last[name] = self._series(last)

    def _ramp_limit(self) -> pd.DataFrame:
        """``process_ramp_limit``: (solve, period) × (process, source, sink, limit), the speed where the solve has
        the arc in its ramp-limit set."""
        index = _mi(list(dict.fromkeys((self.solves[row], row[0]) for row in self.rows)), ["solve", "period"])
        keys = list(self.speed)
        cols = _mi(keys, ["process", "source", "sink", "limit"])
        data = np.array([[speed if solve in solves else np.nan for speed, solves in (self.speed[k] for k in keys)]
                         for solve, _period in index]).reshape(len(index), len(keys))
        return pd.DataFrame(data, index=index, columns=cols, dtype=float)

    def _wide(self, data: dict, names) -> pd.DataFrame:
        if not data:
            cols = pd.Index([], name=names[0]) if len(names) == 1 else _mi([], names)
            return pd.DataFrame(index=self.rows, columns=cols, dtype=float)
        keys = list(data)
        cols = pd.Index(keys, name=names[0]) if len(names) == 1 else _mi(keys, names)
        return pd.DataFrame(np.array([data[k] for k in keys]).T, index=self.rows, columns=cols, dtype=float)

    def build(self):
        arc_names = ["process", "source", "sink"]
        empty_arc = pd.DataFrame(index=self.rows, columns=_mi([], arc_names), dtype=float)
        par = SimpleNamespace(
            step_duration=pd.Series([self.dur[row] for row in self.rows], index=self.rows),
            timestep_weight=pd.Series([self.weight[row] for row in self.rows], index=self.rows),
            complete_period_share_of_year=pd.Series(self.share, index=pd.Index(self.periods, name="period")),
            node_inflow=self._wide(self.inflow, ["node"]),
            process_slope=self._wide(self.slope, ["process"]),
            process_section=self._wide(self.section, ["process"]),
            process_availability=self._wide({}, ["process"]),
            entity_unitsize=pd.Series(self.unitsize).rename_axis("entity"),
            process_arc_cap_coef_dt=self._wide(self.kappa, arc_names) if self.kappa else empty_arc,
            process_arc_floor_share_dt=self._wide(self.floor, arc_names) if self.floor else empty_arc,
            process_ramp_limit=self._ramp_limit(),
            process_ramp_cap_coef_dt=self._wide({}, [*arc_names, "limit"]),
            process_coarse_step_duration_dt=self._wide(self.coarse_dur, ["process"]),
            process_coarse_step_last_dt=self._wide(self.coarse_last, ["process"]),
        )
        link_names = ["period", "time", "period_previous", "time_previous"]
        s = SimpleNamespace(
            dt_realize_dispatch=self.rows,
            solve_period_time=_mi([(self.solves[row], *row) for row in self.rows], ["solve", "period", "time"]),
            node_state_prev=_mi(self.state_prev, ["node", "period", "time", "period_prev", "time_prev"]),
            ramp_link=_mi([(*later, *earlier) for later, earlier in self.links], link_names),
            ramp_carry=_mi([(*later, *earlier) for later, earlier in self.carry], link_names),
            process_source_sink=_mi(self.pss, arc_names),
            process_source_sink_uncapped=_mi([], arc_names),
            process_unit=pd.Index(self.units, name="process"),
            process_connection=pd.Index(self.connections, name="process"),
            process_online=pd.Index(list(self.online), name="process"),
            process__ct_method=_mi(self.methods, ["process", "method"]),
            node_balance=pd.Index(self.balance + self.stores, name="node"),
            node_balance_period=pd.Index([], name="node"),
            node_state=pd.Index(self.stores, name="node"),
            nodeGroupDispatch=pd.Index(list(dict.fromkeys(g for g, _ in self.groups)), name="group"),
            group_node=_mi(self.groups, ["group", "node"]),
        )
        v = SimpleNamespace(state=self._wide(self.state, ["node"]))
        capacity = pd.DataFrame({e: [c] * len(self.periods) for e, c in self.capacity.items()},
                                index=pd.Index(self.periods, name="period"))
        capacity.columns.name = "process"
        fin = self._wide(self.fin, ["process", "node"])
        r = SimpleNamespace(
            flow_dt=self._wide(self.flow, arc_names),
            process_online_dt=self._wide(self.online, ["process"]),
            process_startup_dt=self._wide(self.startup, ["process"]),
            process_shutdown_dt=self._wide(self.shutdown, ["process"]),
            reserves_dt=self._wide(self.reserve, ["process", "reserve", "updown", "node"]),
            potentialVREgen_dt=self._wide(self.pot, ["process", "node"]),
            entity_all_capacity=capacity,
            connection_to_left_node__dt=fin,
            connection_to_right_node__dt=fin.iloc[:, :0],
        )
        return par, s, v, r


def _envelope(frames: Frames) -> SimpleNamespace:
    par, s, v, r = frames.build()
    compute_ramp_envelope(par, s, v, r)
    return r


def _assert_row(table: pd.DataFrame, entity: str, row, expected: dict) -> None:
    """One row of ``entity``: ``expected`` band -> value (bands not given are 0)."""
    got = table[entity].loc[row]
    for band in BANDS:
        assert got[band] == pytest.approx(expected.get(band, 0.0), abs=1e-9), (entity, row, band, got[band])


def _assert_empty(table: pd.DataFrame, entity: str, row) -> None:
    assert table[entity].loc[row].isna().all(), (entity, row)


def _row(ramp_net_load, ramp_demand, units_up, storage_up, vre_up, connections_up, units_down, storage_down,
         vre_down, connections_down) -> dict:
    return dict(zip(BANDS, (ramp_net_load, ramp_demand, units_up, storage_up, vre_up, connections_up, units_down,
                            storage_down, vre_down, connections_down)))


# ── Case 1 ───────────────────────────────────────────────────────────────────────────────────────────────────

T4 = [("p", f"t{i}") for i in range(1, 5)]
LINKS4 = [(T4[1], T4[0]), (T4[2], T4[1]), (T4[3], T4[2])]


def _case1(*, links=LINKS4, carry=(), g2_online=2.0) -> Frames:
    f = Frames(T4, links, carry=carry)
    f.unit("G1", sink="n", flow=[40, 50, 70, 60], capacity=100.0, speed=0.005)
    f.unit("G2", sink="n", flow=[40, 45, 60, 50], capacity=150.0, speed=0.0075, online=g2_online, unitsize=50.0,
           floor=0.4, reserve={("up", "n"): 10.0})
    f.unit("PV", sink="n", flow=[20, 15, 0, 30], capacity=60.0, pot=[30, 15, 0, 30])
    f.storage("S", capacity=50.0, state=[33, 33, 18, 18])
    f.connection("bat", "n", "S", f_in_a=[-10, 0, 15, 0], capacity=20.0)
    f.connection("c", "n", "m", f_in_a=[10, 10, 5, -10], capacity=30.0)
    f.demand("n", [100, 120, 150, 130])
    return f


CASE1 = {
    T4[0]: _row(35, 20, 65, 30, 10, 20, -30, -7, -5, -40),
    T4[1]: _row(45, 30, 65, 20, 0, 20, -35, -17, 0, -40),
    T4[2]: _row(-50, -20, 60, 3, 0, 25, -50, -35, -30, -35),
}


def test_case1_every_band_and_the_empty_last_row() -> None:
    """Oracle Case 1: every column of rows t1..t3; t4 has no successor (empty)."""
    env = _envelope(_case1()).node_ramp_envelope_dt_ep
    assert list(env.columns.names) == ["node", "band"]
    assert list(env["n"].columns) == list(BANDS)
    for row, expected in CASE1.items():
        _assert_row(env, "n", row, expected)
    _assert_empty(env, "n", T4[3])
    # Σup 125, 105, 88; Σdown 82, 92, 150 (down bands stored negative).
    sub = env["n"]
    assert sub.loc[T4[:3], list(UP)].sum(axis=1).tolist() == pytest.approx([125, 105, 88])
    assert (-sub.loc[T4[:3], list(DOWN)].sum(axis=1)).tolist() == pytest.approx([82, 92, 150])


def test_case1_realized_decomposition_lies_in_the_bands() -> None:
    """Oracle Case 1 realized decomposition: line = Σ realized Δ per band, each Δ inside [−down, up]."""
    env = _envelope(_case1()).node_ramp_envelope_dt_ep["n"]
    realized = {  # (units, storage, VRE: Δf − ΔP, connections)
        T4[0]: (15, 10, 10, 0), T4[1]: (35, 15, 0, -5), T4[2]: (-20, -15, 0, -15)}
    for row, parts in realized.items():
        assert sum(parts) == pytest.approx(env.loc[row, "ramp_net_load"])
        for delta, up, down in zip(parts, UP, DOWN):
            assert env.loc[row, down] - 1e-9 <= delta <= env.loc[row, up] + 1e-9, (row, up, delta)


def test_case2_lp_relaxed_online() -> None:
    """Oracle Case 2: G2 online 1.5 (LP-relaxed): U = 75, C 75, L 30, ramp 33.75."""
    env = _envelope(_case1(g2_online=1.5)).node_ramp_envelope_dt_ep
    for row, units in zip(T4[:3], ((53.75, -40), (50, -45), (35, -60))):
        expected = dict(CASE1[row], units_up=units[0], units_down=units[1])
        _assert_row(env, "n", row, expected)
    _assert_empty(env, "n", T4[3])


def test_case1_timestep_hours() -> None:
    """timestep_hours_dt = Δt · timestep_weight / complete_period_share_of_year (h/year)."""
    f = _case1()
    f.share = 4.0 / 8760.0
    hours = _envelope(f).timestep_hours_dt
    assert list(hours.columns) == ["timestep_hours"]
    assert hours["timestep_hours"].tolist() == pytest.approx([8760.0 / 4.0] * 4)


def test_case6_timestep_hours_with_block_weights() -> None:
    """Oracle Case 6 (the part of chunk A): blocks A = {t1, t2} weight 100 and B = {t3, t4} weight 50, period
    share 1: timestep_hours 100, 100, 50, 50 h/year; rows t2, t4 empty (block ends); the rates of rows t1, t3 are
    Case 1's (demand 20, −20; net load 35, −50; one-hour steps)."""
    f = _case1(links=[(T4[1], T4[0]), (T4[3], T4[2])])
    f.weight = {T4[0]: 100.0, T4[1]: 100.0, T4[2]: 50.0, T4[3]: 50.0}
    r = _envelope(f)
    assert r.timestep_hours_dt["timestep_hours"].tolist() == pytest.approx([100.0, 100.0, 50.0, 50.0])
    env = r.node_ramp_envelope_dt_ep["n"]
    assert env.loc[[T4[0], T4[2]], "ramp_demand"].tolist() == pytest.approx([20.0, -20.0])
    assert env.loc[[T4[0], T4[2]], "ramp_net_load"].tolist() == pytest.approx([35.0, -50.0])
    assert env.loc[[T4[1], T4[3]]].isna().all().all()


# ── Case 3: node group ───────────────────────────────────────────────────────────────────────────────────────


def test_case3_group_internal_connection_and_outside_storage() -> None:
    """Oracle Case 3: G = {n1, n2}; the internal connection n1–n2 counts 0 for G, the external m–n1 counts; the
    storage node S (outside G) is G's storage band.  Row t2 is empty for all three entities."""
    rows = [("p", "t1"), ("p", "t2")]
    f = Frames(rows, [(rows[1], rows[0])])
    f.unit("G1", sink="n1", flow=40, capacity=100.0, speed=0.005)
    f.unit("PV", sink="n2", flow=20, capacity=60.0, pot=[30, 15])
    f.storage("S", capacity=50.0, state=33)
    f.connection("bat", "n2", "S", f_in_a=-10, capacity=20.0)
    f.connection("ext", "n1", "m", f_in_a=10, capacity=30.0)
    f.connection("int", "n2", "n1", f_in_a=10, capacity=40.0)
    f.demand("n1", [40, 50])
    f.demand("n2", [20, 30])
    f.group("G", ["n1", "n2"])
    r = _envelope(f)
    env, grp = r.node_ramp_envelope_dt_ep, r.nodeGroup_ramp_envelope_dt_gp
    _assert_row(env, "n1", rows[0], _row(10, 10, 30, 0, 0, 70, -30, 0, 0, -70))
    _assert_row(env, "n2", rows[0], _row(25, 10, 0, 30, 10, 30, 0, -7, -5, -50))
    assert list(grp.columns.names) == ["group", "band"]
    _assert_row(grp, "G", rows[0], _row(35, 20, 30, 30, 10, 20, -30, -7, -5, -40))
    for table, entity in ((env, "n1"), (env, "n2"), (grp, "G")):
        _assert_empty(table, entity, rows[1])


# ── Case 4: rolling seam, rp / gap, unequal steps ────────────────────────────────────────────────────────────


def test_case4_rolling_seam_is_filled_from_the_next_roll() -> None:
    """Oracle Case 4 (rolling): roll 1 realizes t1, t2, roll 2 t3, t4 with the carry t3 ← t2: every row as Case 1,
    the seam row t2 filled."""
    env = _envelope(_case1(carry=[(T4[2], T4[1])])).node_ramp_envelope_dt_ep
    for row, expected in CASE1.items():
        _assert_row(env, "n", row, expected)
    _assert_empty(env, "n", T4[3])


def test_case4_block_end_before_a_gap_is_empty() -> None:
    """Oracle Case 4 (rp / gap): t3 starts a new block (no link t3 ← t2): row t2 empty, t1 and t3 as Case 1."""
    env = _envelope(_case1(links=[(T4[1], T4[0]), (T4[3], T4[2])])).node_ramp_envelope_dt_ep
    _assert_row(env, "n", T4[0], CASE1[T4[0]])
    _assert_row(env, "n", T4[2], CASE1[T4[2]])
    _assert_empty(env, "n", T4[1])
    _assert_empty(env, "n", T4[3])


def _g1_only(speed: float, dur: dict) -> pd.Series:
    rows = [("p", "t1"), ("p", "t2")]
    f = Frames(rows, [(rows[1], rows[0])], dur=dur)
    f.unit("G1", sink="n", flow=[40, 50], capacity=100.0, speed=speed)
    f.demand("n", [40, 50])
    return _envelope(f).node_ramp_envelope_dt_ep["n"].loc[rows[0]]


def test_case4_unequal_steps_and_ramp_room_at_least_one() -> None:
    """Oracle Case 4 side check: Δt = 1, Δt⁺ = 3 → ρ̄ = 0.005·60·(1 + 3)/2 = 0.6 → cap 60, up = min(60, 60);
    with Δt⁺ = 1 the cap is 30 (up 30, down min(40, 30) = 30); a speed with ρ̄ ≥ 1 inside a period has no cap
    (0.02·60·1 = 1.2: up 100 − 40 = 60, down 40)."""
    row = _g1_only(0.005, {("p", "t2"): 3.0})
    assert (row["units_up"], row["units_down"]) == pytest.approx((60.0, -40.0))
    row = _g1_only(0.005, {})
    assert (row["units_up"], row["units_down"]) == pytest.approx((30.0, -30.0))
    row = _g1_only(0.02, {})
    assert (row["units_up"], row["units_down"]) == pytest.approx((60.0, -40.0))


def test_carried_link_is_not_an_in_period_link() -> None:
    """A link carried from the previous roll is built whatever the ramp room (the model's carry rows), so it is
    not subject to the in-period ρ̄ < 1 rule; an adjacent link inside one period is, a cross-period one is not."""
    rows = _mi([("p", "t1"), ("p", "t2"), ("p", "t3"), ("q", "t4")], ["period", "time"])
    names = ["period", "time", "period_previous", "time_previous"]
    link = _mi([("p", "t2", "p", "t1"), ("p", "t3", "p", "t2"), ("q", "t4", "p", "t3")], names)
    nx = _Next(rows, link, _mi([("p", "t3", "p", "t2")], names))
    assert nx.has.tolist() == [True, True, True, False]
    assert nx.in_period.tolist() == [True, False, False, False]


# ── Case 4b: branch point ────────────────────────────────────────────────────────────────────────────────────


def _case4b(variant: str) -> Frames:
    rows = [("d0", "t1"), ("d0", "t2"), ("b_hi", "t3"), ("b_hi", "t4")]
    if variant == "A":
        rows += [("b_lo", "t3"), ("b_lo", "t4")]
    links = [(("d0", "t2"), ("d0", "t1")), (("b_hi", "t3"), ("d0", "t2")), (("b_hi", "t4"), ("b_hi", "t3"))]
    if variant == "A":
        links += [(("b_lo", "t3"), ("d0", "t2")), (("b_lo", "t4"), ("b_lo", "t3"))]
    f = Frames(rows, links)
    pot = {("d0", "t1"): 0, ("d0", "t2"): 10, ("b_hi", "t3"): 20, ("b_hi", "t4"): 10, ("b_lo", "t3"): 0,
           ("b_lo", "t4"): 0}
    dem = {("d0", "t1"): 50, ("d0", "t2"): 60, ("b_hi", "t3"): 80, ("b_hi", "t4"): 90, ("b_lo", "t3"): 40,
           ("b_lo", "t4"): 50}
    f.unit("G1", sink="n", flow=[dem[x] - pot[x] for x in rows], capacity=100.0, speed=0.005)
    f.unit("PV", sink="n", flow=[pot[x] for x in rows], capacity=50.0, pot=[pot[x] for x in rows])
    f.demand("n", [dem[x] for x in rows])
    return f


def test_case4b_branch_point_with_two_visible_successors_is_empty() -> None:
    """Oracle Case 4b variant A: both branches realized in the output namespace: (d0, t2) has two successors →
    empty; the rows inside each branch are filled."""
    env = _envelope(_case4b("A")).node_ramp_envelope_dt_ep
    _assert_row(env, "n", ("d0", "t1"), _row(0, 10, 30, 0, 0, 0, -30, 0, -10, 0))
    _assert_empty(env, "n", ("d0", "t2"))
    _assert_row(env, "n", ("b_hi", "t3"), _row(20, 10, 30, 0, 0, 0, -30, 0, -10, 0))
    _assert_row(env, "n", ("b_lo", "t3"), _row(10, 10, 30, 0, 0, 0, -30, 0, 0, 0))
    _assert_empty(env, "n", ("b_hi", "t4"))
    _assert_empty(env, "n", ("b_lo", "t4"))


def test_case4b_realized_branch_fills_the_branch_point() -> None:
    """Oracle Case 4b variant B: only d0 + b_hi realized: (d0, t2) continues into (b_hi, t3)."""
    env = _envelope(_case4b("B")).node_ramp_envelope_dt_ep
    _assert_row(env, "n", ("d0", "t1"), _row(0, 10, 30, 0, 0, 0, -30, 0, -10, 0))
    _assert_row(env, "n", ("d0", "t2"), _row(10, 20, 30, 0, 0, 0, -30, 0, -20, 0))
    _assert_row(env, "n", ("b_hi", "t3"), _row(20, 10, 30, 0, 0, 0, -30, 0, -10, 0))
    _assert_empty(env, "n", ("b_hi", "t4"))


# ── Case 8c-seam: the ramp-limit set of the solve that realized t⁺ ───────────────────────────────────────────


def test_case8c_seam_uses_the_ramp_set_of_the_realizing_solve() -> None:
    """Oracle Case 8c-seam: solve A (ramp limits kept) realizes t1, t2; solve B (integers_ramps: no ramp limits)
    realizes t3..t5 and continues from A's t2.  The ramp-limit set is per solve: W is in A's only.  Row t2's t⁺ is
    B's t3 → no ramp cap: vre_up 30 (A's set would give 0)."""
    rows = [("p", f"t{i}") for i in range(1, 6)]
    links = [(rows[i + 1], rows[i]) for i in range(4)]
    f = Frames(rows, links, carry=[(rows[2], rows[1])], solves={row: "A" if row in rows[:2] else "B" for row in rows})
    f.unit("W", sink="n", flow=[0, 30, 50, 20, 20], capacity=100.0, pot=[0, 60, 90, 20, 20], speed=0.005,
           speed_in={"A"})
    f.unit("G", sink="n", flow=[50, 20, 0, 30, 30], capacity=100.0)
    f.demand("n", [50] * 5)
    env = _envelope(f).node_ramp_envelope_dt_ep
    _assert_row(env, "n", rows[0], _row(-60, 0, 50, 0, 0, 0, -50, 0, -60, 0))
    _assert_row(env, "n", rows[1], _row(-30, 0, 80, 0, 30, 0, -20, 0, -60, 0))
    _assert_row(env, "n", rows[2], _row(70, 0, 100, 0, 40, 0, 0, 0, 0, 0))
    _assert_row(env, "n", rows[3], _row(0, 0, 70, 0, 0, 0, -30, 0, -20, 0))
    _assert_empty(env, "n", rows[4])
    # Realized t2: ΔG −20, VRE (50 − 30) − 30 = −10 → −30, inside the bands.
    sub = env["n"]
    assert -20 + -10 == pytest.approx(sub.loc[rows[1], "ramp_net_load"])
    assert sub.loc[rows[1], "vre_down"] <= -10 <= sub.loc[rows[1], "vre_up"]


# ── Third round (oracle Cases 11, 12, 7c, 13, 14): the chunk A review fixes ─────────────────────────────────────


def _case11(*, relinked: bool) -> Frames:
    f = Frames(T4, LINKS4)
    f.unit("G", sink="n", flow=[40, 60, 55, 45], capacity=100.0)
    prev = {T4[0]: T4[1], T4[1]: T4[0], T4[2]: T4[3], T4[3]: T4[2]} if relinked else None
    f.storage("S", capacity=50.0, state=[20, 30, 10, 5], prev=prev)
    f.connection("bat", "n", "S", f_in_a=[10, -10, -5, 5], capacity=20.0)
    f.demand("n", [50] * 4)
    return f


def test_case11_storage_cap_reads_the_state_the_balance_links_the_next_step_to() -> None:
    """Oracle Case 11: S binds within its timeblocks A = {t1, t2}, B = {t3, t4} (prev(t3) = t4: block B wraps).
    The storage room toward t⁺ is capped with the state t⁺ starts from, state(prev(t⁺)):
    row t1 (t⁺ t2, prev t1, state 20): arc 20 − 10 = 10 / 10 + 20 = 30, SoC 20 − 10 = 10 / (50 − 20) + 10 = 40 → 10 / 30;
    row t2 (t⁺ t3, prev t4, state 5): arc 30 / 10, SoC 5 + 10 = 15 / 45 − 10 = 35 → 15 / 10;
    row t3 (t⁺ t4, prev t3, state 10): arc 25 / 15, SoC 10 + 5 = 15 / 40 − 5 = 35 → 15 / 15.
    G (capacity 100): up 100 − f = 60, 40, 45, down f = 40, 60, 55.  Without the relink map the cap falls back to
    the state of t: row t2 SoC up 30 + 10 = 40 → storage_up 30 (the arc)."""
    env = _envelope(_case11(relinked=True)).node_ramp_envelope_dt_ep
    _assert_row(env, "n", T4[0], _row(0, 0, 60, 10, 0, 0, -40, -30, 0, 0))
    _assert_row(env, "n", T4[1], _row(0, 0, 40, 15, 0, 0, -60, -10, 0, 0))
    _assert_row(env, "n", T4[2], _row(0, 0, 45, 15, 0, 0, -55, -15, 0, 0))
    _assert_empty(env, "n", T4[3])
    # Realized: ΔG + Δf_in = line, each inside its band.
    sub = env["n"]
    for row, (dg, ds) in zip(T4[:3], ((20, -20), (-5, 5), (-10, 10))):
        assert dg + ds == pytest.approx(sub.loc[row, "ramp_net_load"])
        assert sub.loc[row, "storage_down"] <= ds <= sub.loc[row, "storage_up"]
    chrono = _envelope(_case11(relinked=False)).node_ramp_envelope_dt_ep["n"]
    assert chrono.loc[T4[1], "storage_up"] == pytest.approx(30.0)


def test_case12a_coarse_process_has_room_only_between_its_coarse_steps() -> None:
    """Oracle Case 12a: a coarse process (one flow per coarse step, spec §2.8) has room 0 on the rows inside a
    coarse step; at a coarse step's last fine row ρ̄ uses the coarse step durations.  G1 coarse steps {t1, t2},
    {t3, t4} of 2 h, flow 40, 70, capacity 100, speed 0.005 /min: row t2 ρ̄ = 0.005·60·(2 + 2)/2 = 0.6 → 60,
    up = min(100 − 40, 60) = 60, down = min(40, 60) = 40, lines 70 − 40 = 30; rows t1, t3 0; row t4 empty."""
    f = Frames(T4, LINKS4)
    f.unit("G1", sink="n", flow=[40, 40, 70, 70], capacity=100.0, speed=0.005)
    f.coarse("G1", duration=2.0, last=[0, 1, 0, 1])
    f.demand("n", [40, 40, 70, 70])
    env = _envelope(f).node_ramp_envelope_dt_ep
    _assert_row(env, "n", T4[0], _row(0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    _assert_row(env, "n", T4[1], _row(30, 30, 60, 0, 0, 0, -40, 0, 0, 0))
    _assert_row(env, "n", T4[2], _row(0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    _assert_empty(env, "n", T4[3])


def test_case12b_coarse_process_takes_the_coarse_step_means_at_the_next_step() -> None:
    """Oracle Case 12b: the inputs carry a coarse process's κ and F as their coarse-step means at every fine step
    (fine κ 1, 1, 0.8, 0.6 → 1, 1, 0.7, 0.7; fine F 0.2, 0.4, 0.2, 0.4 → 0.3 everywhere).  G1 online 1 unit of 100,
    flow 40 on b1: row t2 C = 0.7·100 = 70, L = 0.3·100 = 30, ramp 60: up min(70 − 40, 60) = 30, down
    min(40 − 30, 60) = 10 (the fine-step values at t3 would give 40 / 20)."""
    f = Frames(T4, LINKS4)
    f.unit("G1", sink="n", flow=[40, 40, 40, 40], capacity=100.0, speed=0.005, online=1.0, unitsize=100.0,
           floor=0.3, kappa=[1.0, 1.0, 0.7, 0.7])
    f.coarse("G1", duration=2.0, last=[0, 1, 0, 1])
    f.demand("n", [40] * 4)
    env = _envelope(f).node_ramp_envelope_dt_ep
    _assert_row(env, "n", T4[0], _row(0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    _assert_row(env, "n", T4[1], _row(0, 0, 30, 0, 0, 0, -10, 0, 0, 0))
    _assert_row(env, "n", T4[2], _row(0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    _assert_empty(env, "n", T4[3])


def test_case12b_inputs_write_the_coarse_step_means_to_every_fine_step() -> None:
    """Oracle Case 12b, the input side: ``_ramp_inputs._coarse_mean`` replaces a coarse process's rows by the
    Δt-weighted mean (w = ½) over each coarse step, the fine steps without a row at the default: κ rows (≠ 1)
    t3 0.8, t4 0.6 → 0.7 at t3, t4 and 1 at t1, t2; F 0.2, 0.4, 0.2, 0.4 → 0.3 everywhere.  A process that is not
    coarse keeps its rows."""
    import polars as pl

    from flextool.process_outputs._ramp_inputs import _coarse_mean

    steps = pl.DataFrame({"p": ["G1"] * 4, "d": ["p"] * 4, "b_first": ["t1", "t1", "t3", "t3"],
                          "t": ["t1", "t2", "t3", "t4"], "w": [0.5] * 4})
    d = SimpleNamespace(process_coarse_step=steps)
    keys = ["p", "source", "sink"]

    def frame(rows):
        return pl.DataFrame(rows, schema=["p", "source", "sink", "d", "t", "value"], orient="row")

    kappa = _coarse_mean(frame([("G1", "G1", "n", "p", "t3", 0.8), ("G1", "G1", "n", "p", "t4", 0.6),
                                ("G2", "G2", "n", "p", "t3", 0.5)]), d, keys, default=1.0)
    got = {(r["p"], r["t"]): r["value"] for r in kappa.iter_rows(named=True)}
    assert got == pytest.approx({("G1", "t1"): 1.0, ("G1", "t2"): 1.0, ("G1", "t3"): 0.7, ("G1", "t4"): 0.7,
                                 ("G2", "t3"): 0.5})
    floor = _coarse_mean(frame([("G1", "G1", "n", "p", t, x) for t, x in zip(("t1", "t2", "t3", "t4"),
                                                                             (0.2, 0.4, 0.2, 0.4))]),
                         d, keys, default=0.0)
    assert sorted(floor["value"].to_list()) == pytest.approx([0.3] * 4)


def _case7c(*, carried: bool) -> Frames:
    rows = [("p", "t1"), ("p", "t2")]
    f = Frames(rows, [(rows[1], rows[0])], carry=[(rows[1], rows[0])] if carried else (),
               solves={rows[0]: "roll1", rows[1]: "roll2" if carried else "roll1"})
    f.unit("G", sink="n", flow=[85, 50], capacity=150.0, unitsize=50.0, online=[2, 1], shutdown=[0, 1], floor=0.8,
           speed=0.002)
    f.demand("n", [85, 50])
    return f


def test_case7c_carried_link_keeps_the_stopping_units_output() -> None:
    """Oracle Case 7c: G (unitsize 50, F 0.8, ramp 0.12 → 6 MW per unit) goes 2 → 1 unit at t2 (shutdown 1),
    flow 85: U₂ = 1, C 50, L 40.  Adjacent link: f₂ = 85 − 0.8·50 = 45 → up min(5, 6) = 5, down min(5, 6) = 5.  A link
    carried from the previous roll: the model's carried ramp row has no shut-down floor term, f₂ = 85 → up
    min(50 − 85, 6) = −35 → 0, down min(85 − 40, 6) = 6.  Lines −35; row t2 empty."""
    rows = [("p", "t1"), ("p", "t2")]
    env = _envelope(_case7c(carried=False)).node_ramp_envelope_dt_ep
    _assert_row(env, "n", rows[0], _row(-35, -35, 5, 0, 0, 0, -5, 0, 0, 0))
    _assert_empty(env, "n", rows[1])
    env = _envelope(_case7c(carried=True)).node_ramp_envelope_dt_ep
    _assert_row(env, "n", rows[0], _row(-35, -35, 0, 0, 0, 0, -6, 0, 0, 0))
    _assert_empty(env, "n", rows[1])


def test_case13_vre_unit_touching_a_storage_node_has_no_potential_shift() -> None:
    """Oracle Case 13: PV with outputs n (potential 30, 15) and storage S; its arc to n is in the storage band,
    absolute frame: up = 15 − 20 = −5 → 0, down = 20; SoC (state 33 of 50, D 0) 33 / 17 → storage 0 / 17 (with the
    potential shift ΔP = −15 it would be 10 / 5).  Lines: the unit's potential is left out of the net load too,
    N = demand 20 → 15: −5 (with the potential, −10 → 0: +10); demand −5.  Realized: line −5 = the arc's Δf 15 − 20,
    inside the storage band [−17, 0]."""
    rows = [("p", "t1"), ("p", "t2")]
    f = Frames(rows, [(rows[1], rows[0])])
    f.unit("PV", sink="n", flow=[20, 15], capacity=60.0, pot=[30, 15])
    f.pss.append(("PV", "PV", "S"))
    f.flow[("PV", "PV", "S")] = [0.0, 0.0]
    f.storage("S", capacity=50.0, state=[33, 33])
    f.demand("n", [20, 15])
    env = _envelope(f).node_ramp_envelope_dt_ep
    _assert_row(env, "n", rows[0], _row(-5, -5, 0, 0, 0, 0, 0, -17, 0, 0))
    _assert_empty(env, "n", rows[1])


def test_case14_min_load_efficiency_section_and_kappa_at_the_next_step() -> None:
    """Oracle Case 14: direct min_load_efficiency unit HP e → h (unitsize 100, online 1, F 0.2, κ 1, 0.8, 0.7,
    slope 0.5, 0.4, 0.5, section 0.1, 0.2, 0.1, v 40, 50, 30; draw 30, 40, 25 = G).
    v-space: C = κ(t⁺)·100 = 80, 70; up_a 40, 20; down_a 40 − 20 = 20, 50 − 20 = 30 → h up 40, 20, down 20, 30.
    e: up_n = (slope − slope⁺)·v + slope⁺·down_a + (section − section⁺)·100: t1 4 + 8 − 10 = 2, t2 −5 + 15 + 10 = 20;
    down_n = (slope⁺ − slope)·v + slope⁺·up_a + (section⁺ − section)·100: t1 −4 + 16 + 10 = 22, t2 5 + 10 − 10 = 5.
    G (capacity 100): up 70, 60, down 30, 40 → e units 72, 80 / 52, 45.  Lines h +10, −20; e 0."""
    rows = [("p", f"t{i}") for i in range(1, 4)]
    f = Frames(rows, [(rows[1], rows[0]), (rows[2], rows[1])])
    f.unit("HP", sink="h", source="e", flow=[40, 50, 30], slope=[0.5, 0.4, 0.5], section=[0.1, 0.2, 0.1],
           online=1.0, unitsize=100.0, floor=0.2, kappa=[1.0, 0.8, 0.7], capacity=100.0)
    f.unit("G", sink="e", flow=[30, 40, 25], capacity=100.0)
    f.demand("h", [40, 50, 30])
    f.demand("e", [0, 0, 0])
    env = _envelope(f).node_ramp_envelope_dt_ep
    _assert_row(env, "e", rows[0], _row(0, 0, 72, 0, 0, 0, -52, 0, 0, 0))
    _assert_row(env, "e", rows[1], _row(0, 0, 80, 0, 0, 0, -45, 0, 0, 0))
    _assert_row(env, "h", rows[0], _row(10, 10, 40, 0, 0, 0, -20, 0, 0, 0))
    _assert_row(env, "h", rows[1], _row(-20, -20, 20, 0, 0, 0, -30, 0, 0, 0))
    for node in ("e", "h"):
        _assert_empty(env, node, rows[2])
    # Realized at e: ΔG + (draw(t) − draw(t⁺)) = 0, the HP term inside its band.
    sub = env["e"]
    for row, (dg, dhp) in zip(rows[:2], ((10, -10), (-15, 15))):
        assert dg + dhp == pytest.approx(sub.loc[row, "ramp_net_load"])
