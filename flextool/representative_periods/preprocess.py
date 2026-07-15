"""Main orchestrator for representative periods pre-processing.

Reads time series from a FlexTool Spine database, runs greedy convex hull
clustering, computes convex weights, and writes results back to the database.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import spinedb_api as api
from spinedb_api import DatabaseMapping, import_data, Map

from flextool.engine_polars._db_reader import DictMode, params_to_dict
from flextool.representative_periods import force_include
from flextool.representative_periods.clustering import greedy_convex_hull_clustering
from flextool.representative_periods.weights import compute_weight_matrix


def _read_time_series(
    db: DatabaseMapping,
) -> tuple[
    dict[str, list[tuple[str, float]]],
    dict[str, list[tuple[str, float]]],
    dict[str, float],
]:
    """Read profile and inflow time series from the database.

    Returns:
        Tuple of ``(profiles, inflows, demand_scalars)``. ``profiles`` and
        ``inflows`` each map entity name to a list of ``(timestep_key, value)``
        pairs (the time-varying series that feed clustering). ``demand_scalars``
        maps node name to its *scalar* (constant) inflow value as a float — the
        constant demand levels dropped from clustering but needed by
        force-include as per-region demand weights ``D_r``.

    Time-varying series are the ``isinstance(v, list)`` entries.
    ``params_to_dict`` returns a scalar float as a *string* (see
    ``db_reader.py`` ``params_to_dict`` line ~117), and the downstream
    clustering matrix builder iterates each entry's value as
    ``for k, v in ts_data:`` which crashes on the per-character unpack, so
    scalars are excluded from ``inflows``. A constant-inflow node has zero
    variance so it would be skipped by the constant-series filter further down
    anyway; dropping it here makes the skip explicit and matches the existing
    "no matching timesteps" semantics. The scalar inflows are instead collected
    into ``demand_scalars`` (coerced from the string via ``float(...)``, skipping
    any value that does not coerce).
    """
    raw_profiles = params_to_dict(
        db=db, cl="profile", par="profile", mode=DictMode.DICT
    )
    raw_inflows = params_to_dict(
        db=db, cl="node", par="inflow", mode=DictMode.DICT
    )
    profiles: dict[str, list[tuple[str, float]]] = {
        k: v for k, v in raw_profiles.items() if isinstance(v, list)
    }
    inflows: dict[str, list[tuple[str, float]]] = {
        k: v for k, v in raw_inflows.items() if isinstance(v, list)
    }
    # Collect the dropped scalar inflows as demand levels. A scalar comes back
    # as a string from params_to_dict; coerce and skip anything non-numeric.
    demand_scalars: dict[str, float] = {}
    for k, v in raw_inflows.items():
        if isinstance(v, list):
            continue
        try:
            demand_scalars[k] = float(v)
        except (TypeError, ValueError):
            continue
    dropped_profiles = len(raw_profiles) - len(profiles)
    dropped_inflows = len(raw_inflows) - len(inflows)
    if dropped_profiles or dropped_inflows:
        print(
            f"  Skipping {dropped_profiles} scalar profile(s) and "
            f"{dropped_inflows} scalar inflow(s) — clustering only "
            f"considers time-varying series."
        )
    return profiles, inflows, demand_scalars


def _read_region_maps(
    db: DatabaseMapping,
    region_groups: list[str],
    demand_scalars: dict[str, float],
) -> tuple[dict[str, list[str]], dict[str, float], dict[str, list[str]]]:
    """Build node-group demand-weighting maps for force-include (structural).

    For each named region group, resolve its member nodes via ``group__node``
    membership, then:

    * ``region_profiles[r]`` = profile names attached (via ``unit__node__profile``)
      to a node that is a member of region ``r``.
    * ``region_demand[r]`` = ``Σ|demand_scalars[node]|`` over region ``r``'s
      member nodes (the scalar demand already collected by ``_read_time_series``).
    * ``region_nodes[r]`` = the member node names of region ``r`` (used by the
      per-region force-include mode to filter inflow / demand series per region).

    Purely structural — no name parsing. ``group__node`` ``element_name_list`` is
    ``(group, node)`` (dimension order group=0, node=1); ``unit__node__profile``
    is ``(unit, node, profile)`` (node=1, profile=2).

    Args:
        db: Scenario-filtered database mapping.
        region_groups: Names of the region node-groups to weight by.
        demand_scalars: Node name -> scalar demand level (from ``_read_time_series``).

    Returns:
        Tuple of ``(region_profiles, region_demand, region_nodes)``.
    """
    wanted = set(region_groups)

    # region -> set of member node names
    region_node_sets: dict[str, set[str]] = {r: set() for r in region_groups}
    for item in db.get_entity_items(entity_class_name="group__node"):
        group_name, node_name = item["element_name_list"]
        if group_name in wanted:
            region_node_sets[group_name].add(node_name)

    # node -> profiles attached to it
    node_profiles: dict[str, list[str]] = {}
    for item in db.get_entity_items(entity_class_name="unit__node__profile"):
        _unit, node_name, profile_name = item["element_name_list"]
        node_profiles.setdefault(node_name, []).append(profile_name)

    region_profiles: dict[str, list[str]] = {}
    region_demand: dict[str, float] = {}
    region_nodes: dict[str, list[str]] = {}
    for region in region_groups:
        nodes = region_node_sets[region]
        profs: list[str] = []
        for node_name in nodes:
            profs.extend(node_profiles.get(node_name, []))
        region_profiles[region] = profs
        region_nodes[region] = sorted(nodes)
        region_demand[region] = sum(
            abs(demand_scalars[n]) for n in nodes if n in demand_scalars
        )
        print(
            f"  Region '{region}': {len(nodes)} member node(s), "
            f"{len(profs)} profile(s), D_r = {region_demand[region]:.3f}"
        )

    return region_profiles, region_demand, region_nodes


def _get_timeline_keys(db: DatabaseMapping) -> list[str]:
    """Determine the main timeline and return ordered timestep keys.

    Reads all timelines and returns the timestep keys from the first one found.
    """
    timelines = params_to_dict(
        db=db, cl="timeline", par="timestep_duration", mode=DictMode.DICT
    )
    if not timelines:
        raise ValueError("No timeline found in the database.")

    # Use the first timeline available
    timeline_name = next(iter(timelines))
    timeline_data = timelines[timeline_name]
    print(f"Using timeline: '{timeline_name}' with {len(timeline_data)} timesteps")

    # timeline_data is a list of (timestep_key, duration) pairs
    timestep_keys = [entry[0] for entry in timeline_data]
    return timestep_keys


def _build_clustering_matrix(
    profiles: dict[str, list[tuple[str, float]]],
    inflows: dict[str, list[tuple[str, float]]],
    timestep_keys: list[str],
    period_length: int,
) -> tuple[np.ndarray, int]:
    """Build the clustering matrix from time series data.

    Each time series is split into periods, normalized to [0,1],
    and stacked as features.

    Args:
        profiles: Profile time series (name -> [(key, value), ...]).
        inflows: Inflow time series (name -> [(key, value), ...]).
        timestep_keys: Ordered list of timestep keys from the timeline.
        period_length: Number of timesteps per period.

    Returns:
        Tuple of (C, n_base_periods) where C has shape
        (n_features * period_length, n_base_periods).
    """
    n_total = len(timestep_keys)
    n_base_periods = n_total // period_length
    n_dropped = n_total - n_base_periods * period_length

    if n_base_periods == 0:
        raise ValueError(
            f"Timeline has {n_total} timesteps but period_length is {period_length}. "
            f"Need at least {period_length} timesteps."
        )

    if n_dropped > 0:
        print(f"Warning: Dropping {n_dropped} timesteps from end of timeline")

    n_used = n_base_periods * period_length
    key_set = set(timestep_keys[:n_used])

    # Collect all time series into a list of 1D arrays
    all_series: list[np.ndarray] = []
    series_names: list[str] = []

    combined_ts: dict[str, list[tuple[str, float]]] = {}
    for name, data in profiles.items():
        combined_ts[f"profile:{name}"] = data
    for name, data in inflows.items():
        combined_ts[f"inflow:{name}"] = data

    for ts_name, ts_data in combined_ts.items():
        # Build a lookup from key to value
        ts_dict = {k: v for k, v in ts_data}

        # Check if this time series covers the timeline
        available_keys = key_set & ts_dict.keys()
        if len(available_keys) < n_used:
            if len(available_keys) == 0:
                print(f"  Skipping '{ts_name}': no matching timesteps")
                continue
            print(
                f"  Warning: Skipping '{ts_name}': only {len(available_keys)}/{n_used} "
                f"timesteps available"
            )
            continue

        # Extract values in timeline order
        values = np.array([ts_dict[k] for k in timestep_keys[:n_used]], dtype=np.float64)

        # Skip constant time series (no variation, adds no information)
        val_min = values.min()
        val_max = values.max()
        if val_max == val_min:
            print(f"  Skipping '{ts_name}': constant value ({val_min})")
            continue

        # Normalize to [0, 1]
        values = (values - val_min) / (val_max - val_min)

        all_series.append(values)
        series_names.append(ts_name)

    if not all_series:
        raise ValueError("No valid time series found for clustering.")

    print(f"Using {len(all_series)} time series features for clustering")
    for name in series_names:
        print(f"  - {name}")

    # Stack features: each base period d has a feature vector of length
    # (n_features * period_length)
    # Reshape each series to (n_base_periods, period_length) then stack
    feature_blocks: list[np.ndarray] = []
    for values in all_series:
        # Reshape to (n_base_periods, period_length)
        reshaped = values[:n_used].reshape(n_base_periods, period_length)
        feature_blocks.append(reshaped)

    # Stack along feature dimension: (n_base_periods, n_features * period_length)
    feature_matrix = np.hstack(feature_blocks)  # (n_base_periods, n_features * period_length)

    # Transpose to get C of shape (n_features * period_length, n_base_periods)
    C = feature_matrix.T

    print(
        f"Clustering matrix shape: {C.shape} "
        f"({len(all_series)} features x {period_length} timesteps, {n_base_periods} periods)"
    )

    return C, n_base_periods


def _build_timeset_duration_map(
    rep_indices: list[int],
    timestep_keys: list[str],
    period_length: int,
) -> Map:
    """Build a Map of representative period start keys to period durations."""
    keys = [timestep_keys[idx * period_length] for idx in rep_indices]
    values = [float(period_length)] * len(rep_indices)
    return Map(keys, values)


def _build_weights_map(
    W: np.ndarray,
    rep_indices: list[int],
    timestep_keys: list[str],
    period_length: int,
    n_base_periods: int,
) -> Map:
    """Build a nested Map of representative_period_weights.

    Outer keys = base period starting timesteps.
    Inner keys = representative period starting timesteps.
    Inner values = weights (only non-zero entries are included).

    Both levels carry an ``index_name`` (``base_period`` / ``representative_period``)
    so the self-describing-xlsx export labels the two index columns
    meaningfully instead of falling back to the stochastic-sheet defaults
    (``forecast`` / ``branch_time``); the names round-trip with the Map.
    """
    rep_start_keys = [timestep_keys[idx * period_length] for idx in rep_indices]

    outer_keys: list[str] = []
    outer_values: list[Map] = []

    for d in range(n_base_periods):
        base_start_key = timestep_keys[d * period_length]

        # Only include non-zero weights (sparse)
        inner_keys: list[str] = []
        inner_values: list[float] = []
        for r_idx, rep_key in enumerate(rep_start_keys):
            weight = float(W[d, r_idx])
            if weight > 1e-10:
                inner_keys.append(rep_key)
                inner_values.append(weight)

        if inner_keys:
            outer_keys.append(base_start_key)
            outer_values.append(
                Map(inner_keys, inner_values, index_name="representative_period")
            )

    return Map(outer_keys, outer_values, index_name="base_period")


def _write_results_to_db(
    db_url: str,
    timeset_name: str,
    alternative_name: str,
    timeline_name: str,
    timeset_duration_map: Map,
    weights_map: Map,
    solve_period_timesets: dict,
) -> None:
    """Write clustering results to the database in a new alternative.

    Also updates each solve's period_timeset to point to the new RP timeset.
    Opens a NEW connection WITHOUT scenario filter.
    """
    with DatabaseMapping(db_url) as db:
        # Ensure parameter definition exists for representative_period_weights
        parameter_definitions = [
            ("timeset", "representative_period_weights"),
        ]

        # Create alternative
        alternatives = [(alternative_name, f"Representative periods: {timeset_name}")]

        # Create timeset entity
        entities = [("timeset", timeset_name)]

        # Set parameter values (5-tuple: class, entity, param, value, alternative)
        parameter_values = [
            ("timeset", timeset_name, "timeline", timeline_name, alternative_name),
            ("timeset", timeset_name, "timeset_duration", timeset_duration_map, alternative_name),
            ("timeset", timeset_name, "representative_period_weights", weights_map, alternative_name),
        ]

        # Update each solve's period_timeset to use the new RP timeset
        entity_alternatives = [
            ("timeset", timeset_name, alternative_name, True),
        ]
        for solve_name, pts_data in solve_period_timesets.items():
            # pts_data is a list of (period, timeset) tuples or a Map
            if isinstance(pts_data, list):
                periods = [entry[0] for entry in pts_data]
            elif isinstance(pts_data, api.Map):
                periods = list(pts_data.indexes)
            else:
                continue
            # Replace all timesets with the new RP timeset
            new_map = Map(
                [str(p) for p in periods],
                [timeset_name] * len(periods),
            )
            parameter_values.append(
                ("solve", solve_name, "period_timeset", new_map, alternative_name)
            )
            entity_alternatives.append(
                ("solve", solve_name, alternative_name, True)
            )
            print(f"  Updated solve '{solve_name}' period_timeset -> '{timeset_name}' for periods {[str(p) for p in periods]}")

        count, errors = import_data(
            db,
            parameter_definitions=parameter_definitions,
            alternatives=alternatives,
            entities=entities,
            parameter_values=parameter_values,
            entity_alternatives=entity_alternatives,
        )

        if errors:
            for err in errors:
                print(f"  DB import error: {err}")
            raise RuntimeError(f"Failed to write results: {len(errors)} errors")

        # Tag the lazily-created parameter so it lands under the
        # solve_advanced group in group-filtered exports.  No-op on
        # pre-v44 DBs that don't have parameter_groups yet.
        if db.item(db.mapped_table("parameter_group"), name="solve_advanced") is not None:
            db.add_update_item(
                "parameter_definition",
                entity_class_name="timeset",
                name="representative_period_weights",
                parameter_group_name="solve_advanced",
            )

        db.commit_session(f"Add representative periods: {timeset_name}")
        print(f"Wrote {count} items to database")


def preprocess_representative_periods(
    db_url: str,
    scenario_name: str,
    n_rp: int,
    period_length: int,
    *,
    force_peak_load: bool = False,
    force_highest_net_load: bool = False,
    force_window: int | None = None,
    force_count_mode: str = "grow",
    vg_weight: float = 0.5,
    region_groups: list[str] | None = None,
    force_region_scope: bool = False,
    force_region_budget: int | None = None,
) -> str:
    """Select representative periods and write results to database.

    Args:
        db_url: Spine database URL (e.g., 'sqlite:///path.sqlite')
        scenario_name: Name of the scenario to read time series from
        n_rp: Number of representative periods to select
        period_length: Length of each period in timesteps (typically hours)
        force_peak_load: Force-include the peak-net-load base period (Flag A).
        force_highest_net_load: Force-include the sustained-net-load base
            period (Flag B — the energy-adequacy fix).
        force_window: Sub-window length (timesteps) for Flag B; ``None`` means
            the whole-period mean.
        force_count_mode: ``"grow"`` (default) appends forced periods on top of
            the hull picks; ``"fixed"`` keeps the total at ``n_rp`` by dropping
            the most-marginal hull tail picks to make room.
        vg_weight: Convex blend weight in [0, 1] on the VG-shortfall term of the
            net-load signal; the inflow-demand term gets ``1 - vg_weight``.
        region_groups: Optional list of node-group names for demand-weighting
            the VG term (``Σ_r D_r · mean_{p∈r}(1 - avail)``). ``None`` (default)
            keeps the unweighted system aggregate — the byte-parity path.
        force_region_scope: Opt in to the per-region budgeted force-include mode
            (requires ``region_groups``). ``False`` (default) keeps the
            single system-coincident signal — the byte-parity path. When ``True``,
            net load is scored independently per region group and the forced set
            is chosen by greedy budgeted coverage of each region's worst lull.
        force_region_budget: Cap on the number of forced periods under region
            scope. ``None`` (default) derives a sane cap from ``n_rp`` as
            ``max(1, n_rp // 2)`` so forced periods never dominate the
            representative set; ignored when ``force_region_scope`` is ``False``.

    Returns:
        Name of the created timeset entity.

    With no force flag set (the default) the forced-index set is empty and both
    the selected ``rep_indices`` and the emitted Maps are byte-identical to the
    pure-hull path — this is the opt-in byte-parity contract.
    """
    # ------------------------------------------------------------------
    # 1. Read from DB with scenario filter
    # ------------------------------------------------------------------
    print(f"Reading time series from database (scenario: '{scenario_name}')...")
    scen_config = api.filters.scenario_filter.scenario_filter_config(scenario_name)
    with DatabaseMapping(db_url) as db:
        api.filters.scenario_filter.scenario_filter_from_dict(db, scen_config)
        db.fetch_all("parameter_value")

        profiles, inflows, demand_scalars = _read_time_series(db)
        print(
            f"  Found {len(profiles)} profiles, {len(inflows)} node inflows, "
            f"{len(demand_scalars)} scalar demand level(s)"
        )

        # ------------------------------------------------------------------
        # 2. Determine timeline
        # ------------------------------------------------------------------
        timestep_keys = _get_timeline_keys(db)

        # Also read the timeline name for later use
        timelines = params_to_dict(
            db=db, cl="timeline", par="timestep_duration", mode=DictMode.DICT
        )
        timeline_name = next(iter(timelines))

        # Read solve period_timeset mappings so we can update them
        solve_period_timesets: dict = params_to_dict(
            db=db, cl="solve", par="period_timeset", mode=DictMode.DICT
        )

        # Node-group demand-weighting / per-region maps (only when region
        # groups requested).
        region_profiles: dict[str, list[str]] | None = None
        region_demand: dict[str, float] | None = None
        region_nodes: dict[str, list[str]] | None = None
        if region_groups:
            region_profiles, region_demand, region_nodes = _read_region_maps(
                db, region_groups, demand_scalars
            )

    # ------------------------------------------------------------------
    # 3. Build clustering matrix
    # ------------------------------------------------------------------
    print("Building clustering matrix...")
    C, n_base_periods = _build_clustering_matrix(
        profiles, inflows, timestep_keys, period_length
    )

    # ------------------------------------------------------------------
    # 4. Run clustering
    # ------------------------------------------------------------------
    print(f"Running greedy convex hull clustering (selecting {n_rp} from {n_base_periods} periods)...")
    # Keep the UNSORTED greedy selection order: greedy appends the most-marginal
    # pick last, which the "fixed" count mode needs in order to drop from the
    # tail. Sort a copy for the (order-insensitive) hull-index set/logging.
    hull_order = list(greedy_convex_hull_clustering(C, n_rp))
    hull_indices = sorted(hull_order)
    print(f"Selected representative period indices: {hull_indices}")

    # ------------------------------------------------------------------
    # 4b. Force-include adequacy-critical base periods (opt-in)
    # ------------------------------------------------------------------
    # Augment-not-substitute (design §3): add the forced extremes as extra hull
    # vertices, then re-fit ALL weights over the union. Empty forced set (no
    # flag) leaves rep_indices == hull_indices → byte-parity default path.
    # Under region scope, resolve the None budget to an n_rp-derived cap so
    # forced periods never dominate the representative set (documented default).
    effective_region_budget = force_region_budget
    if force_region_scope and effective_region_budget is None:
        effective_region_budget = max(1, n_rp // 2)
    forced_indices = force_include.compute_forced_indices(
        profiles,
        inflows,
        demand_scalars,
        timestep_keys,
        period_length,
        n_base_periods,
        force_peak_load=force_peak_load,
        force_highest_net_load=force_highest_net_load,
        force_window=force_window,
        vg_weight=vg_weight,
        region_profiles=region_profiles,
        region_demand=region_demand,
        force_region_scope=force_region_scope,
        force_region_budget=effective_region_budget,
        region_nodes=region_nodes,
    )

    if not forced_indices:
        # No forcing requested (or nothing scored) → unchanged, byte-identical.
        rep_indices = hull_indices
        n_forced = 0
    elif force_count_mode == "fixed":
        # Keep the total at n_rp. Add the forced indices, then drop the
        # most-marginal hull picks (the TAIL of the greedy selection order) to
        # compensate — but never drop a forced index, and never drop below the
        # forced set. Walk the greedy order from the tail (most-marginal first),
        # dropping hull picks that are not themselves forced, until the union is
        # back down to n_rp.
        forced_set = set(forced_indices)
        keep = set(hull_indices) | forced_set
        # candidates to drop: hull picks not in the forced set, most-marginal
        # (greedy tail) first.
        droppable = [idx for idx in reversed(hull_order) if idx not in forced_set]
        di = 0
        while len(keep) > n_rp and di < len(droppable):
            keep.discard(droppable[di])
            di += 1
        rep_indices = sorted(keep)
        # Forced periods that displaced hull picks = forced periods newly added
        # (i.e. not already hull picks). This is what names the timeset.
        n_forced = len(forced_set - set(hull_indices))
    else:
        # "grow" (default): rep = hull ∪ forced, dedup. n_forced counts only
        # the periods that ACTUALLY entered the set after dedup (a forced index
        # already in the hull adds 0).
        rep_indices = sorted(set(hull_indices) | set(forced_indices))
        n_forced = len(rep_indices) - len(hull_indices)

    if forced_indices:
        forced_start_keys = [
            timestep_keys[idx * period_length] for idx in forced_indices
        ]
        print(
            f"Force-include ({force_count_mode}): forced base periods "
            f"{forced_indices} (starts {forced_start_keys}); "
            f"{n_forced} entered the representative set."
        )
        print(f"Final representative period indices: {rep_indices}")

    # ------------------------------------------------------------------
    # 5. Compute weights
    # ------------------------------------------------------------------
    print("Computing convex weights...")
    W = compute_weight_matrix(C, rep_indices)

    # Compute projection errors for summary
    R = C[:, rep_indices]
    errors = np.array([
        np.linalg.norm(R @ W[d, :] - C[:, d]) for d in range(n_base_periods)
    ])
    mean_error = float(errors.mean())
    max_error = float(errors.max())

    # ------------------------------------------------------------------
    # 6. Build output
    # ------------------------------------------------------------------
    # Naming rule: base is the pure-hull name ``hull_{n_rp}rp_{PL}h``. Append a
    # ``+f{n_forced}`` suffix whenever forced periods actually entered/displaced
    # the set (``n_forced > 0``) — in "grow" mode that is the post-dedup count
    # appended, in "fixed" mode the count of forced periods that displaced hull
    # picks. When ``n_forced == 0`` (default, or a forced index that was already
    # a hull pick) the name is unchanged, preserving byte-parity of the default
    # path and not overwriting the pure-hull timeset.
    suffix = f"+f{n_forced}" if n_forced > 0 else ""
    timeset_name = f"hull_{n_rp}rp_{period_length}h{suffix}"
    alternative_name = timeset_name

    timeset_duration_map = _build_timeset_duration_map(
        rep_indices, timestep_keys, period_length
    )
    weights_map = _build_weights_map(
        W, rep_indices, timestep_keys, period_length, n_base_periods
    )

    # ------------------------------------------------------------------
    # 7. Write to DB (new connection, no scenario filter)
    # ------------------------------------------------------------------
    print("Writing results to database...")
    _write_results_to_db(
        db_url,
        timeset_name,
        alternative_name,
        timeline_name,
        timeset_duration_map,
        weights_map,
        solve_period_timesets,
    )

    # ------------------------------------------------------------------
    # 8. Print summary
    # ------------------------------------------------------------------
    print("\n--- Representative Periods Summary ---")
    print(f"  Representative periods selected: {n_rp}")
    print(f"  Mean projection error: {mean_error:.6f}")
    print(f"  Max projection error:  {max_error:.6f}")
    print(f"  Timeset name:     '{timeset_name}'")
    print(f"  Alternative name: '{alternative_name}'")

    rep_start_keys = [timestep_keys[idx * period_length] for idx in rep_indices]
    print(f"  Representative period starts: {rep_start_keys}")

    return timeset_name


def main() -> None:
    """CLI entry point for representative periods preprocessing."""
    parser = argparse.ArgumentParser(
        description="Select representative periods using greedy convex hull clustering "
        "and write results to a FlexTool Spine database.",
    )
    parser.add_argument(
        "db_url",
        help="Spine database URL (e.g., 'sqlite:///path/to/db.sqlite')",
    )
    parser.add_argument(
        "scenario",
        help="Name of the scenario to read time series from",
    )
    parser.add_argument(
        "n_rp",
        type=int,
        help="Number of representative periods to select",
    )
    parser.add_argument(
        "period_length",
        type=int,
        help="Length of each period in timesteps (e.g., 24 for daily, 168 for weekly)",
    )
    parser.add_argument(
        "--force-peak-load",
        action="store_true",
        help="Force-include the base period with the highest instantaneous "
        "net load (Flag A, capacity-adequacy).",
    )
    parser.add_argument(
        "--force-highest-net-load",
        action="store_true",
        help="Force-include the base period with the greatest sustained net "
        "load (Flag B, energy-adequacy — the multi-carrier fix).",
    )
    parser.add_argument(
        "--force-window",
        type=int,
        default=None,
        help="Sub-window length in timesteps for the sustained net-load score "
        "(default: whole period).",
    )
    parser.add_argument(
        "--force-count-mode",
        choices=("grow", "fixed"),
        default="grow",
        help="'grow' (default) appends forced periods on top of the hull picks; "
        "'fixed' keeps the total at n_rp by dropping the most-marginal hull picks.",
    )
    parser.add_argument(
        "--vg-weight",
        type=float,
        default=0.5,
        help="Convex blend weight in [0,1] on the VG-shortfall term of the "
        "net-load signal; the inflow-demand term gets 1 - vg_weight (default: 0.5).",
    )
    parser.add_argument(
        "--region-groups",
        type=str,
        default=None,
        help="Comma-separated node-group names to demand-weight the net-load "
        "VG term (e.g. 'decomp_AUS,decomp_JAP,decomp_KOR'). Omitted → unweighted "
        "system aggregate (byte-parity default).",
    )
    parser.add_argument(
        "--force-region-scope",
        action="store_true",
        help="Score net load PER region group and greedily force each region's "
        "worst lull under a budget cap, instead of forcing the single "
        "system-coincident worst period. Requires --region-groups.",
    )
    parser.add_argument(
        "--force-region-budget",
        type=int,
        default=None,
        help="Max forced periods under --force-region-scope (default: derived "
        "from n_rp as max(1, n_rp // 2)).",
    )

    args = parser.parse_args()
    region_groups = (
        [g.strip() for g in args.region_groups.split(",") if g.strip()]
        if args.region_groups
        else None
    )

    try:
        preprocess_representative_periods(
            db_url=args.db_url,
            scenario_name=args.scenario,
            n_rp=args.n_rp,
            period_length=args.period_length,
            force_peak_load=args.force_peak_load,
            force_highest_net_load=args.force_highest_net_load,
            force_window=args.force_window,
            force_count_mode=args.force_count_mode,
            vg_weight=args.vg_weight,
            region_groups=region_groups,
            force_region_scope=args.force_region_scope,
            force_region_budget=args.force_region_budget,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
