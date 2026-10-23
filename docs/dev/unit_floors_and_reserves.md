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
* authored `reservation` is added for any group; methods add up.

`p_flow_upper` is the structural bound (existing + max invest by that
period, per unit size); unlimited invest and `conversion_flow_coeff = 0`
arcs carry `max_flow_for_unconstrained_variables`, so the multiplier stays
finite. Rows with a zero multiplier are dropped (no coefficient).

Activation (unchanged, .mod parity): a reserve's relationships are active
only if some group of the same (reserve, up/down) authors a non-zero
`reservation` (`prundt_from_source`, `_reserve_sets`). A dynamic / n-1-only
reserve without any `reservation` is therefore inactive.

## n-1 rows

`reserveBalance_{up,down}_n_1_eq[r, ud, g, p_fail, d, t]`: one row per
failing process `p_fail` (a process with `large_failure_ratio` at a node of
g and a flow into, for up, or out of, for down, that node). LHS: the reserve
of every provider of g except `p_fail` (× unitsize × reliability) plus the
group's shared `vq_reserve · scale`; RHS: `p_fail`'s failing flows ×
unitsize × ratio. The shared slack prices the largest deficit once (.mod
parity). Rows grow with failing processes, nonzeros with failing processes
× providers per group (each row repeats the other providers); an
exact-equivalent aggregate form (a total-reserve variable per
`(r, ud, g, d, t)`) is the remedy if a large model shows the growth.

## Follow-ups (not implemented)

1. Increasing reserves on the source side (a consumer offering `down` by
   consuming more) are not bounded by input headroom.
2. Connections: a net-position rule (`down at n ≤ f_into_n + (cap·avail −
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
5. Dynamic and n-1 reserves activate only if some group of the same
   (reserve, up/down) has a non-zero `reservation` (`prundt_from_source`,
   `_reserve_sets`; see "Activation" above). A dynamic / n-1-only reserve
   is silently inactive — e.g. `network_coal_wind_reserve_n_1` does not
   test any n-1 requirement. Activation should follow the group's
   `reserve_method` (dynamic: any `increase_reserve_ratio`; n-1: any
   `large_failure_ratio`), not the presence of a timeseries reservation.
   Open owner question: the reservation filter was introduced deliberately
   ("reserve provision is active only when there is demand for the
   reserve").
