# Unit floors, input shares and reserve shortfall

Engine notes for `_unit_floors.py`, `model/core.py::_eq_min_input_share` and
`_reserve.reserve_shortfall_scale`. User-facing semantics are in
`docs/reference.md` (`min_load`, `capacity_min_coeff`, `input_share_min`,
Reserves).

## Output measure

`unit_measured_arcs(d)` — the arcs that define a unit's capacity:
the `conversion_flow_coeff ≠ 0` output arcs of an indirect unit (the
`maxOutputSum` arc set, `_indirect_sum_arcs`), the single capped arc of a
direct unit (including the `(p, src, p)` arc of a sink-less unit). Inputs of
indirect units are never in it.

## Rows

| name | over | row |
|---|---|---|
| `minFlow_minload_<kind>` | `(p,d,t) ∈ p_idx`, online ∩ min_load units | `Σ_A v_flow ≥ v_online·min_load + Rdn_p` |
| `minFlow_output_floor_<kind>` | `(p,p,k,d,t)`, units with ≥ 2 measured outputs, `capacity_min_coeff_k > 0` | `v_flow ≥ v_online·min_load·cmin_k + Rdn_k` |
| `minFlow_reserve` | `(p,d,t)`, units with a reducing reserve on A, not covered above | `Σ_A v_flow ≥ Rdn_p` |
| `minFlow_reserve_arc` | `(p,s,k,d,t)`, reducing reserve on an arc of a multi-output unit or outside A | `v_flow ≥ Rdn_a` |
| `minInputShare` | `(p,s,p,d,t)`, indirect units with ≥ 2 conv≠0 inputs, `share_s > 0` | `conv_s·in_s ≥ share_s·Σ conv·in` |

`Rdn` = reserve that reduces the arc: `down` at the sink node, `up` at the
source node. Factor 1 when `v_flow` is the flow at that node (sink side; an
indirect unit's input; a sink-less unit); `1/slope` for a source-side reserve
on a direct input→output arc. All rows are dimensionless (unit-count flow
units) in the autoscale registry.

Decisions: the floor is not scaled by availability; connections are excluded;
an indirect unit's input-side `up` reserve gets only the zero floor on that
input arc (expressing "the input cut takes the outputs below min load" needs
the conversion mix), so combined with output-side `down` reserve such a unit
can end up below `min_load`.

## Upward reserve and online capacity

Upward reserve held at a node a process delivers into (`b.reserve_up_to_sink`,
`b.reserve_up_sink_parts` from `core._reserve_coupling`) sits on the LHS of
the capacity rows, with coefficient 1 (`v_flow` and `v_reserve` are both in
output units of unitsize):

* built capacity, every process: `maxFlow` (per arc) and `maxOutputSum`
  (sum of an indirect unit's outputs);
* online capacity, processes with online variables (`_online.py`):
  `maxFlow_online_<kind>`: `v_flow + Σ_r Rup ≤ v_online·avail·cap_max_coeff`
  per capped arc, and `maxOutputSum_online_<kind>`:
  `Σ_k v_flow + Σ_k Σ_r Rup ≤ v_online·avail`.

So an offline unit holds no upward reserve (spinning reserve). The
built-capacity rows keep their term for online processes too (redundant
while online ≤ built; they carry the invest/divest terms).
`reserve_process_upward` (`max_share`) still uses the built capacity.
Together with the floors the online band is
`v_online·min_load + Rdn ≤ Σ out ≤ v_online·cap·avail − Rup`.

## Source-side reserve and capacity

The mirror of the floors: reserve that raises a unit arc's flow.
`_unit_floors.increasing_source_selectors(d, arcs)` gives `(down_src_flow,
down_src_eff)` keyed `(p, r, ud, source)` over all unit arcs
(`all_unit_arcs`): `down` reserve at the source node of an arc whose
`v_flow` is the input (`sink == p`: an indirect unit's input, a sink-less
unit; φ = 1) or of a direct input→output arc (`source ≠ p ≠ sink`; φ =
`1/slope`, `_inverse_slope`). `core._reserve_coupling` turns them into
`b.reserve_down_at_source` (per arc `(p, source, sink, d, t)`, converted
by φ) and `b.reserve_down_input_arcs`; the reducing `up` source selectors
give `b.reserve_up_at_source` for `profile_flow_lower_limit`.

| row | term |
|---|---|
| `maxFlow` | `+ Σ_r v_reserve_down[p,r,s]·φ`; a bound input arc carrying the reserve gets a row too (the v_flow bound stays, redundant) |
| `maxFlow_online_<kind>` | same term against online capacity |
| `maxInputFlow_online_<kind>` (new) | an online indirect unit's input arc with the reserve: `v_flow + Σ_r v_reserve_down ≤ v_online·avail·p_indirect_input_cap` |
| `profile_flow_upper_limit` / `_lower_limit` | `+ down at the source` / `− up at the source`, × φ |
| `reserve_process_{upward,downward}` | `max_share · c`, c = input capacity per unit at an input node (`_reserve._input_capacity_factor`: `p_indirect_input_cap`, `slope + section`, 1) |

Per-arc simplifications (as the floors): a direct unit with several outputs
has one arc per output from the same input, and the reserve term joins each
of them on `(p, source)`, so the reserve needs headroom on every output arc
(conservative; `_input_capacity_factor` de-duplicates `(p, n)`). An indirect
unit's up reserve at an output (`maxOutputSum`) and down reserve at an input
(`maxFlow` on the input arc) are the same "burn more" action but are
checked in separate rows, so the headroom can be counted twice; a
multi-input indirect unit is only kept within each input's own capacity.
The reverse arc of a two-way unit (2way_2var) is a direct arc in
`process_source_sink_eff`: down reserve at its node is headroom on the arc
that draws from the node and a floor on the arc that feeds it. Connections
are not coupled (Follow-up 2).

## Reserve shortfall multiplier

`vq_reserve ∈ [0, 1]` (bounded for numerics) is multiplied by
`reserve_shortfall_scale(d)` in every `reserveBalance_*` row, in the
objective penalty and in `process_outputs` (`calc_slacks`):

* timeseries: `reservation` — returned as the very same Param when no
  dynamic / n-1 group exists (byte-identical LP);
* dynamic: `Σ p_flow_upper·unitsize·increase_reserve_ratio` over the flows
  of the dynamic RHS (sink and source side);
* n-1: `max_f Σ p_flow_upper·unitsize·large_failure_ratio`, the sum over
  the failing flows of one process f, maximised over the failing processes
  (`_flow_requirement_bound(per_process_max=True)`): the largest single
  n-1 row;
* source side (a flow out of the group's node, dynamic and n-1 down): the
  bound of an arc in `process_source_sink_eff` is multiplied by the
  full-load draw `slope + section` (`_with_full_load_draw`; section only for
  `min_load_efficiency` units with online variables), because the rows count
  the flow drawn from the node as `nodeBalance_eq` does
  (`_reserve._drawn_terms`: `v_flow·unitsize·slope + v_online·section·unitsize`
  on an efficiency arc, `v_flow·unitsize` elsewhere); `v_online ≤
  p_flow_upper` makes it a valid bound;
* dynamic group demand: `group_demand_requirement(d)` = `ratio_g · Σ_{n∈g}
  max(−inflow, 0) / Δt` (exact, a constant), the same Param as the RHS term
  of `reserveBalance_dynamic_eq`. The inflow is `p_negative_inflow_unmargined`:
  the energy-margin emitters write `solve_data/pdtNodeInflow_unmargined`
  (the inflow before the margin) whenever a solve applies a margin, and keep
  it equal to `pdtNodeInflow` in later solves sharing the Provider; without
  the key the field is `p_negative_inflow`. Benders: the field is not
  region-filtered (like `penalty_reserve`); a subproblem joins it with its
  filtered `group_node`, so each region gets its own nodes' demand;
* authored `reservation` is added for any group; methods add up.

`p_flow_upper` is the structural bound (existing + max invest by that
period, per unit size); unlimited invest and `conversion_flow_coeff = 0`
arcs carry `max_flow_for_unconstrained_variables`, so the multiplier stays
finite. Rows with a zero multiplier are dropped (no coefficient).

Activation (`_reserve_methods.demanded_reserves`, shared by
`_derived.prundt_from_source` and `emit/_reserve_sets`): a reserve's
relationships are active when the (reserve, up/down) has a requirement — a
non-zero `reservation` on some group, a dynamic group driven by an
`increase_reserve_ratio` at one of its nodes (or by its own group ratio), or an n-1 group with a
`large_failure_ratio` at one of its nodes (owner Q5, 2026-10-24). A dynamic
/ n-1 group without a driving ratio stays inactive. An n-1 ratio on a
process without a flow in the failing direction activates the reserve with
no n-1 row (inert columns).

## n-1 rows

`reserveBalance_{up,down}_n_1_eq[r, ud, g, p_fail, d, t]`: one row per
failing process `p_fail` (a process with `large_failure_ratio` at a node of
g and a flow into, for up, or out of, for down, that node). LHS:
`v_reserve_total[r, ud, g, d, t]` minus the reserve of `p_fail` at every
node of g (× unitsize × reliability) plus the group's shared
`vq_reserve · scale`; RHS: `p_fail`'s failing flows × unitsize × ratio. The
shared slack prices the largest deficit once (.mod parity).

`v_reserve_total` (POWER column, no cost) exists for every group with an
n-1 row and at least one provider; `reserve_total_upper` bounds it by the
reserve of all providers: `v_reserve_total − reserve_sum ≤ 0`. Eliminating
the total gives back the per-provider form (each n-1 row counts every
provider except `p_fail`), so the reformulation is an exact projection.
The inequality suffices because the total enters only `≥` requirement rows,
with a positive sign, and has no cost. It needs `reserve_sum ≥ 0`, which
holds because providers have reliability > 0
(`process_reserve_upDown_node_active`) and unitsize > 0; otherwise
`v_reserve_total ≥ 0` would add a constraint. A group without providers
(or without reliability data) has no total: its rows are
`vq_reserve · scale ≥ failure`.

The `reserveBalance_timeseries_eq` / `reserveBalance_dynamic_eq` rows of a
group that has a total use `v_reserve_total` in place of `reserve_sum`
(exact by the same projection; groups without n-1 rows keep `reserve_sum`).
This roughly halves the reserve block of groups combining n-1 with another
method. The value function in the timeseries right-hand side is unchanged,
so the reported reserve price (the timeseries row duals) is the same
wherever it is unique.

Nonzeros per `(r, ud, g, d, t)`, with P provider pairs `(p, n)`, and per
failing process f its own provider pairs `own_f` and failing arcs `a_f`:
`(P + 1) + Σ_f (2 + own_f + a_f)` (the per-provider form had
`Σ_f (P − own_f + 1 + a_f)`), so they grow with providers plus failing
processes. HiGHS presolve keeps the total columns (measured with HiGHS 1.14:
presolve neither removes the total columns nor re-densifies the rows).

Results: the objective and every value that is unique at the optimum are
those of the per-provider form; `v_reserve` is costless, so its split among
providers is often not unique and may differ between the two forms. The
n-1 row duals keep their meaning; when the total is positive, the magnitude
of the `reserve_total_upper` dual is the sum of the duals of the group-step's
rows that use the total (HiGHS reports the `≤` row with the opposite sign to
the `≥` rows). The total is not written to the outputs.

## Follow-ups (not implemented)

1. (Done: "Source-side reserve and capacity" above.)
2. Connections (owner question Q-c13-A, planned with the connection
   online/capacity work, c14): a net-position rule (`down at n ≤ f_into_n + (cap·avail −
   f_out_of_n)`), interacting with `v_flow_back` and the other end.
3. `reserveBalance` lacks the .mod's slope factor for source-side reserves
   of direct units; `_unit_floors._inverse_slope` follows the current
   convention and must change with it (tripwire:
   `test_down_reserve_floor.py::test_direct_unit_input_side_up_reserve`).
4. Connection `delay` is not implemented. Connections are never in
   `process_indirect`, and the delayed input term exists only in
   `conversion_indirect`; the .mod had the same gap (its nodeBalance delay
   term is commented out). The `delay` used to mark the connection delayed
   anyway, which made it `fork_yes`, moved its arcs from
   `process_source_sink_eff` to `_noEff` (efficiency losses dropped) and
   turned a `regular` connection one-way. Now every reader of `delay`
   reads `unit.delay` only (`input_derivation/_specs.py` delay tables,
   `input_derivation/_process_method.py`, `_derived`
   `_classify_process_method` / `_delay_distributions_from_source`,
   `_projection.process_delayed`), so a connection is modelled as
   if no delay were set, and `_validators.validate_connection_delay` warns
   once per connection with a non-zero delay
   (`tests/engine/test_connection_delay_ignored.py`). Applying a
   connection delay needs a delayed term in the node balance; when it is
   added, drop the filters and the warning.
5. The dynamic and n-1 rows ignore `v_flow_back` (2way_1var / DC
   connections), which `nodeBalance_eq` counts
   (`nodeBalance_back_flow_terms`): a one-variable connection flowing
   backwards into the group, or failing while doing so, is not counted.
   Planned with the connection online/capacity work (c14).
