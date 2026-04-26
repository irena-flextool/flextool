import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import time
import yaml

from flextool.lean_parquet import read_lean_parquet, write_lean_parquet


def _parse_rename_entry(entry) -> tuple[str, bool]:
    """Parse a rename map value: ``[display_name, export_to_excel]``.

    Returns (display_name, export_to_excel).
    """
    if isinstance(entry, list) and len(entry) >= 2:
        return str(entry[0]), bool(entry[1])
    # Bare string (legacy) — treat as name with export=True
    return str(entry), True

from datetime import datetime, timezone
from flextool.process_outputs.read_variables import read_variables
from flextool.process_outputs.read_parameters import read_parameters
from flextool.process_outputs.read_sets import read_sets
from flextool.process_outputs.process_results import post_process_results
from flextool.process_outputs.out_capacity import unit_capacity, connection_capacity, node_capacity
from flextool.process_outputs.out_flows import (
    unit_outputNode, unit_inputNode,
    unit_cf_outputNode, unit_cf_inputNode,
    unit_VRE_curtailment_and_potential, unit_ramps,
    unit_online_and_startup,
)
from flextool.process_outputs.out_node import node_summary, node_additional_results
from flextool.process_outputs.out_costs import generic, cost_summaries, CO2
from flextool.process_outputs.out_ancillary import (
    connection, connection_wards, connection_cf,
    reserves, investment_duals, inertia_results,
    slack_variables, input_sets, dc_power_flow,
    co2_duals,
)
from flextool.process_outputs.out_group import (
    nodeGroup_indicators, nodeGroup_VRE_share,
    nodeGroup_total_inflow, nodeGroup_flows,
)
from flextool.process_outputs.out_flowgroup import flowGroup_indicators
from flextool.plot_outputs.plot_functions import plot_dict_of_dataframes
import logging
from spinedb_api import DatabaseMapping, from_database, Array
import warnings


def _read_outputs(output_dir):
    """Read solver output files and return (par, s, v)."""
    p = read_parameters(output_dir)
    s = read_sets(output_dir)
    v = read_variables(output_dir)
    return p, s, v


def _resolve_comparison_config_path(output_config_path):
    """Resolve the comparison plot config path for write_outputs.

    Mirrors the GUI's resolution order (see ``result_viewer._resolve_config_path``
    and ``scenario_comparison.orchestrator``): if the single-mode config
    sits beside a ``default_comparison_plots.yaml`` (i.e. the user has
    overridden both in a project-local ``templates/``), use that.
    Otherwise fall back to ``templates/default_comparison_plots.yaml``
    in the flextool root.
    """
    candidate_dir = os.path.dirname(output_config_path) if output_config_path else None
    if candidate_dir:
        candidate = os.path.join(candidate_dir, 'default_comparison_plots.yaml')
        if os.path.isfile(candidate):
            return candidate
    _flextool_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(_flextool_root, 'templates', 'default_comparison_plots.yaml')


def _snapshot_plan_dir(plan_dir):
    """Capture all files currently in ``plan_dir`` as ``{name: bytes}``.

    Used to overlay the single-config plans back after the comparison
    pass wipes the directory.  Returns ``{}`` if the dir doesn't exist.
    """
    if not os.path.isdir(plan_dir):
        return {}
    snapshot: dict[str, bytes] = {}
    for name in os.listdir(plan_dir):
        path = os.path.join(plan_dir, name)
        if os.path.isfile(path):
            try:
                with open(path, 'rb') as f:
                    snapshot[name] = f.read()
            except OSError:
                continue
    return snapshot


def _restore_plan_files(plan_dir, snapshot, prefer_existing_files=None):
    """Write *snapshot* files back into ``plan_dir``.

    Files in *prefer_existing_files* (a set of basenames) are skipped —
    used for the single ``_availability.json`` we synthesise as a union
    of both passes' availability lists.
    """
    if not snapshot:
        return
    os.makedirs(plan_dir, exist_ok=True)
    skip = prefer_existing_files or set()
    for name, blob in snapshot.items():
        if name in skip:
            continue
        try:
            with open(os.path.join(plan_dir, name), 'wb') as f:
                f.write(blob)
        except OSError as exc:
            logging.warning(
                "Failed to restore plan file %s: %s", name, exc,
            )


def _merge_availability_manifests(plan_dir, snapshot):
    """Merge the snapshot's ``_availability.json`` with the current one.

    The current file (just written by the comparison pass) lists
    comparison-only ``(result_key, sub_config)`` pairs.  The snapshot's
    file (from the single pass) lists single-config pairs.  We union
    them and rewrite ``_availability.json`` so the viewer's manifest
    reflects every plan that survives on disk.
    """
    import json as _json
    avail_name = "_availability.json"
    current_pairs: list[list[str]] = []
    avail_path = os.path.join(plan_dir, avail_name)
    try:
        if os.path.isfile(avail_path):
            with open(avail_path, "r", encoding="utf-8") as f:
                current_pairs = list(_json.load(f).get("available", []))
    except (OSError, ValueError) as exc:
        logging.warning(
            "Failed to read current availability manifest %s: %s",
            avail_path, exc,
        )
        current_pairs = []
    snapshot_pairs: list[list[str]] = []
    if avail_name in snapshot:
        try:
            snapshot_pairs = list(_json.loads(snapshot[avail_name]).get("available", []))
        except ValueError as exc:
            logging.warning(
                "Failed to parse snapshot availability manifest: %s", exc,
            )
    # Union, preserving order: snapshot (single) first, then comparison-only.
    seen: set[tuple[str, str]] = set()
    merged: list[list[str]] = []
    for pair in list(snapshot_pairs) + list(current_pairs):
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        key = (str(pair[0]), str(pair[1]))
        if key in seen:
            continue
        seen.add(key)
        merged.append([key[0], key[1]])
    try:
        with open(avail_path, "w", encoding="utf-8") as f:
            _json.dump({"available": merged}, f)
    except OSError as exc:
        logging.warning(
            "Failed to write merged availability manifest %s: %s",
            avail_path, exc,
        )


def _compute_comparison_only_plot_plans(
    *, plan_results, single_plot_settings, parquet_dir,
    output_config_path, active_configs, plot_rows, plan_break_times,
):
    """Compute per-scenario plans for comparison-only result_keys.

    Strategy:
      1. Snapshot the existing ``plot_plans/`` dir (single-config plans).
      2. Call ``compute_all_plot_plans`` with the comparison config —
         this wipes the dir and writes plans for whatever result_keys
         the comparison config can resolve.
      3. Overlay the snapshot back, so single-config plans win on
         overlapping ``(result_key, sub_config)`` pairs.
      4. Merge the two ``_availability.json`` files into a union.

    Logs WARNING for comparison-only result_keys we expected to fill
    but had no in-memory data for.
    """
    from flextool.plot_outputs.orchestrator import compute_all_plot_plans
    from flextool.plot_outputs.config import flatten_new_format

    comparison_config_path = _resolve_comparison_config_path(output_config_path)
    if not os.path.isfile(comparison_config_path):
        logging.debug(
            "Comparison config not found at %s — skipping second-pass plan compute.",
            comparison_config_path,
        )
        return

    with open(comparison_config_path, 'r', encoding='utf-8') as f:
        comparison_settings = yaml.safe_load(f) or {}
    comparison_plot_settings = comparison_settings.get('plots', {}) or {}

    # Find result_keys present in comparison but NOT in single (so we
    # can warn if they have no in-memory data).
    single_keys = set(flatten_new_format(single_plot_settings).keys())
    comparison_keys = set(flatten_new_format(comparison_plot_settings).keys())
    comparison_only = comparison_keys - single_keys
    missing_data = [k for k in comparison_only if k not in plan_results or plan_results[k].empty]
    if missing_data:
        logging.warning(
            "Comparison-only plot result_keys with no available data "
            "(plans will be missing for these in the viewer): %s",
            sorted(missing_data),
        )

    plan_dir = os.path.join(parquet_dir, "plot_plans")
    snapshot = _snapshot_plan_dir(plan_dir)
    avail_name = "_availability.json"

    # Run the comparison-config pass.  This wipes plan_dir and writes
    # only what the comparison config resolves.
    compute_all_plot_plans(
        plan_results, comparison_plot_settings, parquet_dir,
        active_settings=active_configs, plot_rows=plot_rows,
        break_times=plan_break_times,
    )

    # Restore single-config plan files (parquet + json) over the
    # comparison-pass output.  Skip _availability.json — that gets
    # merged separately so the union of pairs is preserved.
    _restore_plan_files(plan_dir, snapshot, prefer_existing_files={avail_name})
    _merge_availability_manifests(plan_dir, snapshot)


def log_time(log_string, start):
    print(f"---{log_string}: {time.perf_counter() - start:.4f} seconds")
    os.makedirs('output', exist_ok=True)
    with open("output/solve_progress.csv", "a") as solve_progress:
        solve_progress.write(log_string + ',' + str(round(time.perf_counter() - start, 4)) + '\n')
    return time.perf_counter()


def print_namespace_structure(namespace, name='r', max_items=3, output_file='namespace_structure.txt'):
    import pandas as pd
    import sys

    def format_list(items, max_n=max_items):
        items_list = list(items)
        if len(items_list) <= max_n:
            return items_list
        return items_list[:max_n] + [f'... ({len(items_list)} total)']

    with open(output_file, 'a') as f:
        original_stdout = sys.stdout
        sys.stdout = f

        for attr_name in dir(namespace):
            if attr_name.startswith('_'):
                continue

            obj = getattr(namespace, attr_name)
            print(f"\n{name}.{attr_name}")

            if isinstance(obj, pd.DataFrame):
                print("Type: DataFrame")
                print(f"Shape: {obj.shape}")
                print(f"Index: {obj.index.names if hasattr(obj.index, 'names') else 'default'}")
                print(f"Columns: {format_list(obj.columns)}")
                print(f"Dtypes:\n{obj.dtypes}")

            elif isinstance(obj, pd.Series):
                print("Type: Series")
                print(f"Shape: {obj.shape}")
                print(f"Index: {obj.index.name or 'default'}")
                print(f"Dtype: {obj.dtype}")

            elif isinstance(obj, pd.Index):
                print("Type: Index")
                print(f"Name: {obj.name}")
                print(f"Values: {format_list(obj)}")
                print(f"Dtype: {obj.dtype}")

        sys.stdout = original_stdout


def write_summary_csv(par, s, v, r, csv_dir):
    """Write summary CSV file matching the GNU MathProg format"""
    import os

    # Output file path
    fn_summary = os.path.join(csv_dir, 'summary_solve.csv')

    # Get common parameters - these are Series indexed by period
    p_inflation_factor_operations_yearly = par.inflation_factor_operations_yearly
    p_inflation_factor_investment_yearly = par.inflation_factor_investment_yearly
    complete_period_share_of_year = par.complete_period_share_of_year

    # Get period sets
    period_in_use = complete_period_share_of_year.index
    d_realized_period = s.d_realized_period
    d_realize_invest = s.d_realize_invest

    # Open file and write all content
    with open(fn_summary, 'w') as f:
        # Header with timestamp
        timestamp = datetime.now(timezone.utc)
        f.write(f'"Diagnostic results from all solves. Output at (UTC): {timestamp}"\n\n')

        # Total cost from solver (M CUR)
        f.write('\n')
        f.write('"Solve","Objective","Total cost from solver, includes all penalty costs"\n')
        for row_idx in v.obj.index:
            f.write(f'{row_idx},{v.obj.loc[row_idx, "objective"] / 1000000:.12g}\n')

        # Total cost (calculated) full horizon (M CUR)
        total_cost_full = (
            r.costOper_and_penalty_d
                .add(r.costInvest_d, fill_value=0.0)
                .add(r.costDivest_d, fill_value=0.0)
        ).sum(axis=0) / 1000000

        f.write(f'"Total cost (calculated) full horizon (M CUR)",{total_cost_full:.12g},"Annualized operational, penalty and investment costs"\n')
        f.write(f'"Total cost (calculated) realized periods (M CUR)",{total_cost_full:.12g}\n')

        # Operational costs for realized periods (M CUR)
        operational_costs = r.costOper_d.sum(axis=0) / 1000000
        f.write(f'"Operational costs for realized periods (M CUR)",{operational_costs:.12g}\n')

        # Investment costs for realized periods (M CUR)
        investment_costs = r.costInvest_d.sum(axis=0) / 1000000
        f.write(f'"Investment costs for realized periods (M CUR)",{investment_costs:.12g}\n')

        # Retirement costs / salvage revenue for realized periods (M CUR).
        # Sign follows the user's salvage_value input: positive salvage
        # (scrap revenue) shows as a negative cost; negative salvage
        # (decommissioning cost) shows as a positive cost.  Included in
        # the solver objective since v_divest is a decision variable.
        retirement_costs = r.costDivest_d.sum(axis=0) / 1000000
        f.write(f'"Retirement costs / salvage revenue for realized periods (M CUR)",{retirement_costs:.12g}\n')

        # Fixed costs for existing entities (M CUR)
        fixed_costs_pre_existing = r.costFixedPreExisting_d.sum(axis=0) / 1000000
        fixed_costs_invested = r.costFixedInvested_d.sum(axis=0) / 1000000
        fixed_costs_divested = r.costFixedDivested_d.sum(axis=0) / 1000000

        f.write(f'"Fixed costs for pre-existing entities (M CUR)",{fixed_costs_pre_existing:.12g}\n')
        f.write(f'"Fixed costs for invested entities (M CUR)",{fixed_costs_invested:.12g}\n')
        f.write(f'"Fixed cost removal due to divested entities (M CUR)",{fixed_costs_divested:.12g}\n')

        # Penalty (slack) costs for realized periods (M CUR)
        penalty_costs = r.costPenalty_d.sum(axis=0) / 1000000
        f.write(f'"Penalty (slack) costs for realized periods (M CUR)",{penalty_costs:.12g}\n')

        # Period information table
        f.write('\nPeriod')
        for d in period_in_use:
            f.write(f',{d}')
        f.write('\n')

        # Time in use in years
        f.write('"Time in use in years"')
        for d in period_in_use:
            f.write(f',{complete_period_share_of_year[d]:.12g}')
        f.write('\n')

        # Operational discount factor
        f.write('"Operational discount factor"')
        for d in period_in_use:
            f.write(f',{p_inflation_factor_operations_yearly[d]:.12g}')
        f.write('\n')

        # Investment discount factor
        f.write('"Investment discount factor"')
        for d in s.d_realize_invest:
            f.write(f',{p_inflation_factor_investment_yearly[d]:.12g}')
        f.write('\n\n')

        # Emissions section
        f.write('Emissions\n')
        co2_total = r.emissions_co2_d.sum(axis=0) / 1000000
        f.write(f'"CO2 [Mt]",{co2_total:.6g},"System-wide annualized CO2 emissions for realized periods"\n')

        # Slack variables section
        f.write('\n"Slack variables multiplied by timestep duration (creating or removing energy/matter, ')
        f.write('creating inertia, adding synchronous generation, decreasing capacity margin, creating reserve)"\n')

        # Node state slack - upward (creating energy)
        for node in r.upward_node_slack_d_not_annualized.columns:
            for period in d_realized_period:
                if period in r.upward_node_slack_d_not_annualized.index and r.upward_node_slack_d_not_annualized.loc[period, node] > 0:
                    f.write(f'Created, {node}, {period}, {r.upward_node_slack_d_not_annualized.loc[period, node]:.5g}\n')

        # Node state slack - downward (removing energy)
        for node in r.downward_node_slack_d_not_annualized.columns:
            for period in d_realized_period:
                if period in r.downward_node_slack_d_not_annualized.index and r.downward_node_slack_d_not_annualized.loc[period, node] > 0:
                    f.write(f'Removed, {node}, {period}, {r.downward_node_slack_d_not_annualized.loc[period, node]:.5g}\n')

        # Inertia slack
        for group in r.q_inertia_d_not_annualized.columns:
            for period in d_realized_period:
                if period in r.q_inertia_d_not_annualized.index and r.q_inertia_d_not_annualized.loc[period, group] > 0:
                    f.write(f'Inertia, {group}, {period}, {r.q_inertia_d_not_annualized.loc[period, group]:.5g}\n')

        # Non-synchronous slack
        for group in r.q_non_synchronous_d_not_annualized.columns:
            for period in d_realized_period:
                if period in r.q_non_synchronous_d_not_annualized.index and r.q_non_synchronous_d_not_annualized.loc[period, group] > 0:
                    f.write(f'NonSync, {group}, {period}, {r.q_non_synchronous_d_not_annualized.loc[period, group]:.5g}\n')

        # Capacity margin slack
        for group in r.q_capacity_margin_d_not_annualized.columns:
            for period in d_realize_invest:
                if period in r.q_capacity_margin_d_not_annualized.index and r.q_capacity_margin_d_not_annualized.loc[period, group] > 0:
                    f.write(f'CapMargin, {group}, {period}, {r.q_capacity_margin_d_not_annualized.loc[period, group]:.5g}\n')

        # Reserve slack
        for group in r.q_reserves_d_not_annualized.columns:
            for period in d_realized_period:
                if period in r.q_reserves_d_not_annualized.index and r.q_reserves_d_not_annualized.loc[period, group] > 0:
                    f.write(f'Reserve, {group}, {period}, {r.q_reserves_d_not_annualized.loc[period, group]:.5g}\n')


# List of all output functions
ALL_OUTPUTS = [
    input_sets,
    generic,
    cost_summaries,
    reserves,
    unit_online_and_startup,
    node_summary,
    node_additional_results,
    investment_duals,
    inertia_results,
    slack_variables,
    unit_capacity,
    connection_capacity,
    node_capacity,
    nodeGroup_indicators,
    nodeGroup_VRE_share,
    flowGroup_indicators,
    CO2,
    co2_duals,
    nodeGroup_flows,
    unit_outputNode,
    unit_inputNode,
    connection,
    connection_wards,
    nodeGroup_total_inflow,
    connection_cf,
    unit_cf_outputNode,
    unit_cf_inputNode,
    unit_VRE_curtailment_and_potential,
    unit_ramps,
    dc_power_flow,
]


def _resolve_settings(write_methods, output_config_path, active_configs, plot_rows,
                      output_location, plot_file_format, settings_db_url,
                      fallback_output_location):
    """Resolve output settings: explicit args > settings DB > hardcoded defaults."""
    db_reachable = False
    if settings_db_url:
        if settings_db_url.startswith('sqlite:///'):
            db_reachable = os.path.exists(settings_db_url.replace('sqlite:///', ''))
        else:
            db_reachable = True  # HTTP or other remote URLs — let DatabaseMapping handle errors
    if db_reachable:
        with DatabaseMapping(settings_db_url) as settings_db:
            settings_entities = settings_db.get_entity_items(entity_class_name="settings")
            if len(settings_entities) == 1:
                settings_name = settings_entities[0]["name"]
                settings_params: dict = {}
                for pv in settings_db.get_parameter_value_items(entity_class_name="settings"):
                    if pv["entity_byname"] == (settings_name,):
                        settings_params[pv["parameter_definition_name"]] = from_database(pv["value"], pv["type"])
                logging.debug(f"Settings DB parameters: {settings_params}")

                if write_methods is None:
                    method_keys = [f'output-{m}' for m in ['plot', 'parquet', 'csv', 'excel']]
                    if any(k in settings_params for k in method_keys):
                        write_methods = [m for m in ['plot', 'parquet', 'csv', 'excel']
                                         if settings_params.get(f'output-{m}', False)]

                if output_config_path is None and 'output-config-path' in settings_params:
                    output_config_path = str(settings_params['output-config-path'])

                if active_configs is None and 'active-output-configs' in settings_params:
                    val = settings_params['active-output-configs']
                    if isinstance(val, str):
                        active_configs = [val]
                    elif isinstance(val, Array):
                        active_configs = list(val.values)
                    else:
                        active_configs = list(val)

                if plot_rows is None:
                    first = settings_params.get('plot_first_timestep')
                    duration = settings_params.get('plot_duration')
                    if first is not None and duration is not None:
                        plot_rows = (int(first), int(first) + int(duration))

                if output_location is None and 'output-location' in settings_params:
                    output_location = str(settings_params['output-location'])

                if plot_file_format is None and 'plot-file-format' in settings_params:
                    plot_file_format = str(settings_params['plot-file-format'])

    # Apply hardcoded defaults for anything still unset
    _flextool_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if write_methods is None:
        write_methods = ['plot', 'parquet', 'excel']
    if output_config_path is None:
        output_config_path = os.path.join(_flextool_root, 'templates', 'default_plots.yaml')
    elif not os.path.isabs(output_config_path):
        output_config_path = os.path.join(_flextool_root, output_config_path)
    if active_configs is None:
        active_configs = ['default']
    if plot_rows is None:
        plot_rows = (0, 167)
    if output_location is None:
        output_location = fallback_output_location or ''
    if plot_file_format is None:
        plot_file_format = 'png'

    return write_methods, output_config_path, active_configs, plot_rows, output_location, plot_file_format


def write_outputs(scenario_name, output_config_path=None, active_configs=None, output_funcs=None, output_location=None, subdir=None, read_parquet_dir=False, write_methods=None, plot_rows=None, debug=False, single_result=None, settings_db_url=None, fallback_output_location=None, plot_file_format=None, raw_output_dir=None, only_first_file=False):
    """
    Write FlexTool outputs to various formats.

    Args:
        scenario_name: Name of the scenario
        output_config_path: Path to YAML configuration file defining outputs
        active_configs: output_config yaml can contain multiple plot configurations for same data, choose which ones to use. Defaults to 'default' only.
        output_funcs: list of functions to run, or None for ALL_OUTPUTS
        subdir: Subdirectory for outputs
        read_parquet_dir: Directory to read existing parquet files from
        write_methods: List of output methods ('plot', 'parquet', 'excel', 'csv')
        plot_rows: Tuple of first and last row to plot in a time series plots. Default is (0, 167).
        debug: Enable debug output
        single_result: Tuple of (key, csv_name, plot_name, plot_type, subplots_per_row, legend_position)
                       for processing a single result. Overrides config file.
        settings_db_url: URL of the settings database (optional, fills in unset params)
        fallback_output_location: Used as output_location if not set by caller or settings DB
        raw_output_dir: Path to the directory containing solver raw output CSV files
            (default: 'output_raw' relative to CWD)
    """
    write_methods, output_config_path, active_configs, plot_rows, output_location, plot_file_format = _resolve_settings(
        write_methods, output_config_path, active_configs, plot_rows,
        output_location, plot_file_format, settings_db_url, fallback_output_location,
    )

    logging.debug(
        f"Resolved output settings: write_methods={write_methods}, "
        f"output_config_path={output_config_path}, active_configs={active_configs}, "
        f"plot_rows={plot_rows}, output_location={output_location}"
    )

    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    start = time.perf_counter()

    # Load output configuration from YAML or create from single_result
    if single_result:
        # Parse single_result tuple
        key, csv_name, plot_name, plot_type, subplots_per_row, legend_position = single_result

        # Convert string "null" to None
        def parse_value(val):
            if val == "null" or val == "None":
                return None
            # Try to convert to int if it's a numeric string
            try:
                return int(val)
            except (ValueError, TypeError):
                return val

        csv_name = parse_value(csv_name)
        plot_name = parse_value(plot_name)
        plot_type = parse_value(plot_type)
        subplots_per_row = parse_value(subplots_per_row)
        legend_position = parse_value(legend_position)

        # Create single-entry settings dict
        settings = {
            "plots": {
                key: [plot_name, plot_type, subplots_per_row, legend_position]
            },
            "rename": {
                key: csv_name
            }
        }
    else:
        # Load output configuration from YAML
        with open(output_config_path, 'r', encoding='utf-8') as f:
            settings = yaml.safe_load(f)

    if subdir:
        parquet_dir = os.path.join(output_location, 'output_parquet', subdir)
        csv_dir = os.path.join(output_location, 'output_csv', subdir)
        plot_dir = os.path.join(output_location, 'output_plots', subdir)
    else:
        parquet_dir = os.path.join(output_location, 'output_parquet')
        csv_dir = os.path.join(output_location, 'output_csv')
        plot_dir = os.path.join(output_location, 'output_plots')

    # Read and process data
    start = log_time("Read configuration files", start)

    # If results already exist as parquet files, read them
    if read_parquet_dir:
        results = {}
        for filename in os.listdir(parquet_dir):
            if filename.endswith('.parquet') and filename != 'timeline_breaks.parquet':
                key = filename[:-8]  # Remove '.parquet' extension
                filepath = os.path.join(parquet_dir, filename)
                results[key] = read_lean_parquet(filepath)
                if len(results[key].columns.names) == 1:
                    results[key] = results[key].squeeze()
                else:
                    results[key] = results[key].droplevel('scenario', axis=1)
        start = log_time("Read parquet files", start)

    # Read original raw outputs from FlexTool
    else:
        par, s, v = _read_outputs(raw_output_dir or 'output_raw')
        start = log_time("Read flextool outputs", start)

        # Pre-process results to be closer to what needed for output writing
        r = post_process_results(par, s, v)
        start = log_time("Post-processed outputs", start)

        # Call the final processing functions for each category of outputs
        # and make a dict of dataframes to hold final results
        output_funcs = output_funcs or ALL_OUTPUTS

        all_results = {}
        for func in output_funcs:
            func_results = func(par, s, v, r, debug)
            if not func_results:
                continue

            # Handle both single result (wrapped in list) and multiple results
            if not isinstance(func_results, list):
                func_results = [func_results]

            for result_df, table_name in func_results:
                # Use excel_sheet as the key to allow multiple outputs per function
                all_results[table_name] = result_df

        results = all_results
        start = log_time("Formatted for output", start)

    # Write files for debugging purposes
    if debug:
        open('namespace_structure.txt', 'w').close()
        print_namespace_structure(r, 'r')
        print_namespace_structure(s, 's')
        print_namespace_structure(v, 'v')
        print_namespace_structure(par, 'par')
        start = log_time("Wrote debugging files", start)

    # Write to parquet
    if 'parquet' in write_methods and not read_parquet_dir:
        os.makedirs(parquet_dir, exist_ok=True)
        if not os.path.exists(parquet_dir):
            os.makedirs(parquet_dir)
        for name, df in results.items():
            if isinstance(df, (pd.MultiIndex, pd.Index)):
                df = df.to_frame(index=False)
                df.insert(0, 'scenario', scenario_name)
                df.set_index(list(df.columns)).index
            else:
                df = pd.concat({scenario_name: df}, axis=1, names=['scenario'])
            write_lean_parquet(df, f'{parquet_dir}/{name}.parquet')

        # Copy timeline_breaks from solve_data to parquet dir.  It moved
        # out of output_raw with the rest of the derived-parameter printfs.
        raw_dir = raw_output_dir or 'output_raw'
        work_dir = os.path.dirname(raw_dir) or '.'
        breaks_csv = os.path.join(work_dir, 'solve_data', 'timeline_breaks.csv')
        if os.path.exists(breaks_csv):
            breaks_df = pd.read_csv(breaks_csv)
            write_lean_parquet(breaks_df, f'{parquet_dir}/timeline_breaks.parquet', index=False)

        start = log_time("Wrote to parquet", start)

    # Compute plot plans for the viewer (always, when parquets exist)
    if 'parquet' in write_methods or read_parquet_dir:
        try:
            from flextool.plot_outputs.orchestrator import compute_all_plot_plans
            from flextool.plot_outputs.format_helpers import load_timeline_breaks
            plan_break_times = load_timeline_breaks(parquet_dir)
            plan_results = {k: v.to_frame() if isinstance(v, pd.Series) else v for k, v in results.items()}
            single_plot_settings = settings.get('plots', {})
            compute_all_plot_plans(
                plan_results, single_plot_settings, parquet_dir,
                active_settings=active_configs, plot_rows=plot_rows,
                break_times=plan_break_times,
            )
            start = log_time("Computed plot plans", start)
        except Exception as exc:
            logging.warning("Plot plan computation failed (non-fatal): %s", exc)

        # Also compute per-scenario plans for any result_keys that only
        # appear in the comparison config (e.g. ``costs_discounted_p_``,
        # which is referenced from default_comparison_plots.yaml but not
        # from default_plots.yaml).  Without this second pass the
        # comparison view's lazy plan-union path fails for those keys
        # ("No per-scenario plan for <rk> in viewer scenarios").
        #
        # ``compute_all_plot_plans`` wipes its plan_dir at entry, so
        # we snapshot the single-config plans first and overlay them
        # back afterwards; on overlapping (result_key, sub_config)
        # pairs the *single* config's layout wins (it's the
        # authoritative one for single-mode rendering, and the
        # comparison view tolerates the same parquet shape).
        if 'parquet' in write_methods or read_parquet_dir:
            try:
                _compute_comparison_only_plot_plans(
                    plan_results=plan_results,
                    single_plot_settings=single_plot_settings,
                    parquet_dir=parquet_dir,
                    output_config_path=output_config_path,
                    active_configs=active_configs,
                    plot_rows=plot_rows,
                    plan_break_times=plan_break_times,
                )
                start = log_time("Computed comparison-only plot plans", start)
            except Exception as exc:
                logging.warning(
                    "Comparison-only plot plan computation failed (non-fatal): %s",
                    exc,
                )

        # Phase D: write per-scenario dispatch metadata so the viewer can
        # union ylims across scenarios on demand (without waiting for the
        # cross-scenario combine).  Loads the just-written parquets back
        # via the existing readers, restricted to this single scenario.
        try:
            import json as _json
            from flextool.scenario_comparison.db_reader import (
                build_scenario_folders_from_dir,
                collect_parquet_files,
                combine_parquet_files,
            )
            from flextool.scenario_comparison.dispatch_mappings import (
                combine_dispatch_mappings,
            )
            from flextool.scenario_comparison.data_models import TimeSeriesResults
            from flextool.scenario_comparison.dispatch_plots import (
                compute_dispatch_metadata_for_scenario,
            )

            # parquet_dir == <output_location>/output_parquet/<scenario>/
            scenario_pq_dir = os.path.dirname(parquet_dir) if subdir else parquet_dir
            scenario_name_for_meta = subdir or scenario_name

            sc_folders = build_scenario_folders_from_dir(
                scenario_pq_dir, [scenario_name_for_meta],
            )
            if sc_folders:
                files_by_name = collect_parquet_files(sc_folders, output_subdir="")
                combined = combine_parquet_files(files_by_name, num_scenarios=1)
                ts_results = TimeSeriesResults.from_dict(combined)
                dispatch_mappings = combine_dispatch_mappings(sc_folders, "")

                if plot_rows and len(plot_rows) >= 2:
                    meta_timeline = (int(plot_rows[0]), int(plot_rows[1]) + 1)
                else:
                    meta_timeline = (0, 168)

                disp_meta = compute_dispatch_metadata_for_scenario(
                    ts_results, dispatch_mappings,
                    scenario_name_for_meta, meta_timeline,
                )
            else:
                disp_meta = {"nodeGroups": {}}

            meta_path = os.path.join(parquet_dir, "_dispatch_metadata.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                _json.dump(disp_meta, f, indent=2)
            start = log_time("Wrote dispatch metadata", start)
        except Exception as exc:
            logging.warning(
                "Per-scenario dispatch metadata computation failed (non-fatal): %s",
                exc,
            )

    # Plot results
    if 'plot' in write_methods:
        os.makedirs(plot_dir, exist_ok=True)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        # Don't delete existing plots when processing single result
        delete_plots = not bool(single_result)
        results = {k: v.to_frame() if isinstance(v, pd.Series) else v for k, v in results.items()}
        # Load timeline breaks for visual gaps in time-series plots
        from flextool.plot_outputs.format_helpers import load_timeline_breaks
        break_times = load_timeline_breaks(parquet_dir)
        plot_dict_of_dataframes(results, plot_dir, settings['plots'], active_settings=active_configs, plot_rows=plot_rows, delete_existing_plots=delete_plots, plot_file_format=plot_file_format, only_first_file=only_first_file, break_times=break_times)

        start = log_time('Plotted figures', start)

    # Write to csv
    if 'csv' in write_methods:
        os.makedirs(csv_dir, exist_ok=True)
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)

        # Only empty csv dir when not processing single result
        if not single_result:
            for filename in os.listdir(csv_dir):
                file_path = os.path.join(csv_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        # Different CSV writing logic depending on data source
        if read_parquet_dir:
            # Simplified CSV writing from parquet (no par,s,v,r available)
            rename_raw = settings.get('rename', {})
            for table_name, df in results.items():
                display_name, _ = _parse_rename_entry(rename_raw.get(table_name, table_name))
                csv_filename = display_name + '.csv'
                csv_path = os.path.join(csv_dir, csv_filename)
                df_copy = df.reset_index()
                df_copy.columns.names = [None] * df_copy.columns.nlevels
                df_copy.to_csv(csv_path, index=False, float_format='%.8g')
        else:
            # Full CSV writing from output_raw (par,s,v,r available)
            write_summary_csv(par, s, v, r, csv_dir)

            rename_raw = settings.get('rename', {})
            for table_name, df in results.items():
                if isinstance(df, (pd.MultiIndex, pd.Index)):
                    df = df.to_frame(index=False)
                if 'solve' not in df.index.names and 'period' in df.index.names:
                    if 'time' in df.index.names:
                        # Use per-timestep mapping for dispatch data (correct per-roll solve names)
                        # Build a dict from (period, time) → solve to avoid sort-merge reordering
                        spt = s.solve_period_time
                        if spt.droplevel('solve').duplicated().any():
                            spt = spt[~spt.droplevel('solve').duplicated(keep='last')]
                        pt_to_solve = dict(zip(spt.droplevel('solve'), spt.get_level_values('solve')))
                        solve_vals = df.index.map(lambda x: pt_to_solve.get((x[0], x[1]) if isinstance(x, tuple) else x, ''))
                        df.index = pd.MultiIndex.from_arrays(
                            [solve_vals] + [df.index.get_level_values(n) for n in df.index.names],
                            names=['solve'] + list(df.index.names)
                        )
                    else:
                        # For period-only data, use solve_period to add solve column
                        # Use dict lookup to preserve index order (join reorders alphabetically)
                        unique_sp = s.solve_period[~s.solve_period.droplevel('solve').duplicated(keep='last')]
                        period_to_solve = dict(zip(unique_sp.get_level_values('period'), unique_sp.get_level_values('solve')))
                        if df.index.nlevels == 1:
                            solve_vals = df.index.map(lambda p: period_to_solve.get(p, ''))
                            df.index = pd.MultiIndex.from_arrays(
                                [solve_vals, df.index],
                                names=['solve', df.index.name]
                            )
                        else:
                            period_level = df.index.get_level_values('period')
                            solve_vals = period_level.map(lambda p: period_to_solve.get(p, ''))
                            df.index = pd.MultiIndex.from_arrays(
                                [solve_vals] + [df.index.get_level_values(n) for n in df.index.names],
                                names=['solve'] + list(df.index.names)
                            )
                    names = list(df.index.names)
                    solve_pos = names.index('solve')
                    period_pos = names.index('period')
                    names.pop(solve_pos)
                    if solve_pos < period_pos:
                        period_pos -= 1
                    names.insert(period_pos, 'solve')
                    df.index = df.index.reorder_levels(order=names)

                display_name, _ = _parse_rename_entry(rename_raw.get(table_name, table_name))
                csv_filename = display_name + '.csv'
                csv_path = os.path.join(csv_dir, csv_filename)
                df = df.reset_index()
                df.columns.names = [None] * df.columns.nlevels
                df.to_csv(csv_path, index=False, float_format='%.8g')

        start = log_time('Wrote to csv', start)

    # Write to excel
    if 'excel' in write_methods:
        rename_raw = settings.get('rename', {})
        excel_dir = os.path.join(output_location, 'output_excel')
        os.makedirs(excel_dir, exist_ok=True)
        excel_path = os.path.join(excel_dir, 'output_' + scenario_name + '.xlsx')
        # Build list of (sheet_name, df) sorted alphabetically
        sheets: list[tuple[str, pd.DataFrame]] = []
        used_names: set[str] = set()
        for name, df in results.items():
            display_name, export = _parse_rename_entry(
                rename_raw.get(name, name)
            )
            if not export:
                continue
            if isinstance(df, (pd.MultiIndex, pd.Index)):
                df = df.to_frame(index=False)
            if (not df.empty) & (len(df) > 0):
                sheet_name = display_name[:31]
                if sheet_name in used_names:
                    suffix = 1
                    while f"{sheet_name[:28]}_{suffix}" in used_names:
                        suffix += 1
                    sheet_name = f"{sheet_name[:28]}_{suffix}"
                used_names.add(sheet_name)
                sheets.append((sheet_name, df))
        sheets.sort(key=lambda x: x[0].lower())

        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            for sheet_name, df in sheets:
                df.to_excel(writer, sheet_name=sheet_name)

        start = log_time('Wrote to Excel', start)
