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

from flextool.flextoolrunner.db_reader import DictMode, params_to_dict
from flextool.representative_periods.clustering import greedy_convex_hull_clustering
from flextool.representative_periods.weights import compute_weight_matrix


def _read_time_series(
    db: DatabaseMapping,
) -> tuple[dict[str, list[tuple[str, float]]], dict[str, list[tuple[str, float]]]]:
    """Read profile and inflow time series from the database.

    Returns:
        Tuple of (profiles, inflows) where each is a dict mapping
        entity name to a list of (timestep_key, value) pairs.
    """
    profiles: dict[str, list[tuple[str, float]]] = params_to_dict(
        db=db, cl="profile", par="profile", mode=DictMode.DICT
    )
    inflows: dict[str, list[tuple[str, float]]] = params_to_dict(
        db=db, cl="node", par="inflow", mode=DictMode.DICT
    )
    return profiles, inflows


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
            outer_values.append(Map(inner_keys, inner_values))

    return Map(outer_keys, outer_values)


def _write_results_to_db(
    db_url: str,
    timeset_name: str,
    alternative_name: str,
    timeline_name: str,
    timeset_duration_map: Map,
    weights_map: Map,
) -> None:
    """Write clustering results to the database in a new alternative.

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

        # Entity alternatives
        entity_alternatives = [
            ("timeset", timeset_name, alternative_name, True),
        ]

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

        db.commit_session(f"Add representative periods: {timeset_name}")
        print(f"Wrote {count} items to database")


def preprocess_representative_periods(
    db_url: str,
    scenario_name: str,
    n_rp: int,
    period_length: int,
) -> str:
    """Select representative periods and write results to database.

    Args:
        db_url: Spine database URL (e.g., 'sqlite:///path.sqlite')
        scenario_name: Name of the scenario to read time series from
        n_rp: Number of representative periods to select
        period_length: Length of each period in timesteps (typically hours)

    Returns:
        Name of the created timeset entity.
    """
    # ------------------------------------------------------------------
    # 1. Read from DB with scenario filter
    # ------------------------------------------------------------------
    print(f"Reading time series from database (scenario: '{scenario_name}')...")
    scen_config = api.filters.scenario_filter.scenario_filter_config(scenario_name)
    with DatabaseMapping(db_url) as db:
        api.filters.scenario_filter.scenario_filter_from_dict(db, scen_config)
        db.fetch_all("parameter_value")

        profiles, inflows = _read_time_series(db)
        print(f"  Found {len(profiles)} profiles, {len(inflows)} node inflows")

        # ------------------------------------------------------------------
        # 2. Determine timeline
        # ------------------------------------------------------------------
        timestep_keys = _get_timeline_keys(db)

        # Also read the timeline name for later use
        timelines = params_to_dict(
            db=db, cl="timeline", par="timestep_duration", mode=DictMode.DICT
        )
        timeline_name = next(iter(timelines))

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
    rep_indices = greedy_convex_hull_clustering(C, n_rp)
    print(f"Selected representative period indices: {rep_indices}")

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
    timeset_name = f"hull_{n_rp}rp_{period_length}h"
    alternative_name = f"hull_{n_rp}rp_{period_length}h"

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

    args = parser.parse_args()

    try:
        preprocess_representative_periods(
            db_url=args.db_url,
            scenario_name=args.scenario,
            n_rp=args.n_rp,
            period_length=args.period_length,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
