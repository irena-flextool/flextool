import os
import re
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
from flextool.process_outputs.read_parameters import (
    read_parameters,
    read_parameters_multi,
)
from flextool.process_outputs.read_sets import (
    read_sets,
    read_sets_multi,
)
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


def _backfill_group_indicator_sets(s, output_dir):
    """Populate ``nodeGroupDispatch`` / ``*Indicators`` from input CSVs.

    These three sets are derived from per-group user parameters
    (``output_nodeGroup_dispatch`` / ``output_nodeGroup_indicators`` /
    ``output_flowGroup_indicators``).  No FlexData field carries them
    today; ``read_sets`` therefore hardcodes them empty.  The legacy
    input-writer (``flextool/flextoolrunner/input_writer.py``) still
    emits the CSVs to ``<workdir>/input/`` ahead of every solve, so we
    backfill from there.  Without this, every ``nodeGroup_flows`` /
    ``flowGroup_indicators`` writer short-circuits to "empty" and the
    group_flows__dt / group_flows__d / flowGroup CSVs go missing.

    Also backfills the 12 ``nodeGroupDispatch__*`` arc-union MultiIndex
    sets from the polars-LP writer's ``solve_data/*.csv`` artefacts
    (see ``flextool/engine_polars/_writer_arc_unions.py``).  Without
    these, ``calc_group_flows`` finds zero rows for the unit / connection
    aggregator joins, and ``out_group.nodeGroup_flows`` emits only the
    slack/inflow/loss column families — group_flows__dt loses its
    ``from_unitGroup`` / ``from_unit`` / ``from_connectionGroup`` /
    ``to_connectionGroup`` / per-connection ``internal_losses`` columns.
    """
    raw_dir = output_dir or 'output_raw'
    work_dir = os.path.dirname(raw_dir) or '.'
    input_dir = os.path.join(work_dir, 'input')
    for attr, fname in (
        ('nodeGroupDispatch', 'nodeGroupDispatch.csv'),
        ('nodeGroupIndicators', 'nodeGroupIndicators.csv'),
        ('flowGroupIndicators', 'flowGroupIndicators.csv'),
    ):
        path = os.path.join(input_dir, fname)
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if df.empty or df.shape[1] == 0:
            continue
        setattr(s, attr, pd.Index(df.iloc[:, 0].dropna(), name='group'))

    # --- 12 nodeGroupDispatch__* arc-union sets (from solve_data/) -----------
    # Each entry: (attr-on-s, filename, csv→multi-index column map).
    # The column map is ``[(csv_col, level_name), ...]`` — same CSV column may
    # appear under multiple level names (e.g. ``unit`` populates both
    # ``process`` and ``unit`` in the unit-aggregator sets where process==unit).
    solve_data_dir = os.path.join(work_dir, 'solve_data')
    dispatch_specs = (
        # 4-col / 2-col Not-in-aggregate sets — CSV columns map 1:1 to levels.
        ('nodeGroupDispatch__process_fully_inside',
         'nodeGroupDispatch__process_fully_inside.csv',
         (('group', 'group'), ('process', 'process'))),
        ('nodeGroupDispatch__process__unit__to_node_Not_in_aggregate',
         'nodeGroupDispatch__process__unit__to_node_Not_in_aggregate.csv',
         (('group', 'group'), ('process', 'process'),
          ('unit', 'unit'), ('node', 'node'))),
        ('nodeGroupDispatch__process__node__to_unit_Not_in_aggregate',
         'nodeGroupDispatch__process__node__to_unit_Not_in_aggregate.csv',
         (('group', 'group'), ('process', 'process'),
          ('node', 'node'), ('unit', 'unit'))),
        ('nodeGroupDispatch__process__connection__to_node_Not_in_aggregate',
         'nodeGroupDispatch__process__connection__to_node_Not_in_aggregate.csv',
         (('group', 'group'), ('process', 'process'),
          ('connection', 'connection'), ('node', 'node'))),
        ('nodeGroupDispatch__process__node__to_connection_Not_in_aggregate',
         'nodeGroupDispatch__process__node__to_connection_Not_in_aggregate.csv',
         (('group', 'group'), ('process', 'process'),
          ('node', 'node'), ('connection', 'connection'))),
        ('nodeGroupDispatch__connection_Not_in_aggregate',
         'nodeGroupDispatch__connection_Not_in_aggregate.csv',
         (('group', 'group'), ('connection', 'connection'))),
        # Unit aggregator sets: CSV stores (group, group_aggregate, unit,
        # source, sink); a unit's process name equals its unit name, and the
        # node is the sink for to-node side / source for to-unit side.
        ('nodeGroupDispatch__processGroup__process__unit__to_node',
         'nodeGroupDispatch__group_aggregate__process__unit__to_node.csv',
         (('group', 'group'), ('group_aggregate', 'group_aggregate'),
          ('unit', 'process'), ('unit', 'unit'), ('sink', 'node'))),
        ('nodeGroupDispatch__processGroup__process__node__to_unit',
         'nodeGroupDispatch__group_aggregate__process__node__to_unit.csv',
         (('group', 'group'), ('group_aggregate', 'group_aggregate'),
          ('unit', 'process'), ('source', 'node'), ('unit', 'unit'))),
        # Connection aggregator sets: CSV stores (group, group_aggregate,
        # connection, source, sink); process==connection, node side depends.
        ('nodeGroupDispatch__processGroup__process__connection__to_node',
         'nodeGroupDispatch__group_aggregate__process__connection__to_node.csv',
         (('group', 'group'), ('group_aggregate', 'group_aggregate'),
          ('connection', 'process'), ('connection', 'connection'),
          ('sink', 'node'))),
        ('nodeGroupDispatch__processGroup__process__node__to_connection',
         'nodeGroupDispatch__group_aggregate__process__node__to_connection.csv',
         (('group', 'group'), ('group_aggregate', 'group_aggregate'),
          ('connection', 'process'), ('source', 'node'),
          ('connection', 'connection'))),
        # 2-col projection sets — (group, group_aggregate).
        ('nodeGroupDispatch__processGroup_Unit_to_group',
         'nodeGroupDispatch__group_aggregate_Unit_to_group.csv',
         (('group', 'group'), ('group_aggregate', 'group_aggregate'))),
        ('nodeGroupDispatch__processGroup_Group_to_unit',
         'nodeGroupDispatch__group_aggregate_Group_to_unit.csv',
         (('group', 'group'), ('group_aggregate', 'group_aggregate'))),
        ('nodeGroupDispatch__processGroup_Connection',
         'nodeGroupDispatch__group_aggregate_Connection.csv',
         (('group', 'group'), ('group_aggregate', 'group_aggregate'))),
    )
    for attr, fname, col_map in dispatch_specs:
        path = os.path.join(solve_data_dir, fname)
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        # Skip if CSV is missing any required source column.
        csv_cols = [c for c, _ in col_map]
        if not all(c in df.columns for c in csv_cols):
            continue
        level_names = [lvl for _, lvl in col_map]
        arrays = [df[c].tolist() for c, _ in col_map]
        mi = pd.MultiIndex.from_arrays(arrays, names=level_names)
        setattr(s, attr, mi)


def _read_outputs(
    output_dir,
    *,
    flex_data=None,
    solution=None,
    solve_name=None,
    solve_steps=None,
):
    """Read solver output files and return (par, s, v).

    Δ.31: ``flex_data`` + ``solution`` are required for the in-memory
    parameter / set reading.  Variables (parquets) are still read from
    ``output_raw/`` since the polars-LP write path emits them there.

    ``solve_steps`` (post-Δ.31 multi-solve fix): an ordered list of
    ``(solve_name, FlexData)`` pairs covering every sub-solve (roll)
    of the orchestration cascade.  When supplied, ``par`` / ``s`` are
    built as the union over every sub-solve's dt axis — matching the
    ``solve``-axis union ``v`` carries from per-sub-solve parquet
    aggregation.  Falls back to the single-``flex_data`` path when
    not supplied (single-solve scenarios).
    """
    if solve_steps is not None:
        if solution is None:
            raise ValueError(
                "_read_outputs requires solution alongside solve_steps."
            )
        p = read_parameters_multi(solve_steps, solution)
        s = read_sets_multi(solve_steps, solution)
        _backfill_group_indicator_sets(s, output_dir)
        v = read_variables(output_dir)
        return p, s, v
    if flex_data is None or solution is None:
        raise ValueError(
            "_read_outputs requires flex_data and solution after Δ.31; "
            "the CSV-based read_parameters / read_sets path is gone."
        )
    p = read_parameters(flex_data, solution, solve_name=solve_name or "solve")
    s = read_sets(flex_data, solution, solve_name=solve_name or "solve")
    _backfill_group_indicator_sets(s, output_dir)
    v = read_variables(output_dir)
    return p, s, v


def log_time(log_string, start, timing_recorder=None):
    """Print and record one ``write_outputs`` sub-phase.

    Refactored to feed the new TimingRecorder when supplied; the legacy
    append to ``output/solve_progress.csv`` (CWD-relative, append-mode,
    never cleared) has been removed.  The ``print(...)`` line is kept
    for human-readable stdout — that's the file most operators eyeball.
    """
    elapsed = time.perf_counter() - start
    print(f"---{log_string}: {elapsed:.4f} seconds")
    if timing_recorder is not None:
        timing_recorder.record(
            'write_outputs',
            subphase=log_string,
            seconds=elapsed,
            t_start=start,
        )
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

        # Total cost from solver (M CUR).  Iterate solve names in
        # natural-sort order (so ``roll_0, roll_1, ..., roll_10, ...``
        # not ``roll_1, roll_10, ...``).  ``v.obj.index`` carries the
        # parquet-concat order, which depends on
        # ``solve_data/solve__p_entity_pre_existing.csv`` — written as a
        # header-only stub by the engine cascade for some rolling-solve
        # paths, leaving the sort key unstable.  Natural-sort here is
        # the writer-side guard that matches v3.32.0's numeric solve
        # ordering.
        f.write('\n')
        f.write('"Solve","Objective","Total cost from solver, includes all penalty costs"\n')
        sorted_solves = sorted(
            v.obj.index,
            key=lambda s: [int(t) if t.isdigit() else t
                           for t in re.split(r"(\d+)", str(s))],
        )
        for row_idx in sorted_solves:
            f.write(f'{row_idx},{v.obj.loc[row_idx, "objective"] / 1000000:.12g}\n')

        # Total cost (calculated) full horizon (M CUR).  Mirrors the LP
        # objective: oper + penalty + invest + divest + fixed (pre-existing
        # + invested + divested).  Fixed-cost terms are in the LP obj at
        # flextool.mod:2107-2126; omitting them here would under-report
        # the objective for any scenario with non-zero existing capacity
        # or a non-zero lifetime_fixed_cost on invest/divest entities.
        total_cost_full = (
            r.costOper_and_penalty_d
                .add(r.costInvest_d, fill_value=0.0)
                .add(r.costDivest_d, fill_value=0.0)
                .add(r.costFixedPreExisting_d, fill_value=0.0)
                .add(r.costFixedInvested_d, fill_value=0.0)
                .add(r.costFixedDivested_d, fill_value=0.0)
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

        # Investment discount factor — iterate ``period_in_use`` (the
        # same axis as the "Operational discount factor" / "Time in use
        # in years" rows above) so the per-period values align under
        # the period header.  Earlier HEAD iterated ``s.d_realize_invest``
        # which (a) emits no values for scenarios without realized
        # invest decisions (e.g. ``capacity_margin``, where v3.32.0
        # still wrote ``,1`` against the single p2020 period) and
        # (b) carries a different period order on multi-period invest
        # scenarios than the surrounding rows.  v3.32.0 wrote one
        # value per period in ``period_in_use``.
        f.write('"Investment discount factor"')
        for d in period_in_use:
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

        # Capacity margin slack — iterate realized periods rather than
        # d_realize_invest.  The capacity-margin constraint is defined
        # per realized period and is independent of invest/divest
        # entities; scenarios such as ``capacity_margin`` have no
        # invest entities, so d_realize_invest is empty and this loop
        # would silently drop legitimate slack rows.
        for group in r.q_capacity_margin_d_not_annualized.columns:
            for period in d_realized_period:
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
    from flextool._resources import package_data_path
    _default_plots = str(package_data_path("textual_templates/default_plots.yaml"))
    if write_methods is None:
        write_methods = ['plot', 'parquet', 'excel']
    if output_config_path is None:
        output_config_path = _default_plots
    elif not os.path.isabs(output_config_path):
        # Legacy: a settings DB may carry a repo-relative override like
        # ``templates/default_plots.yaml``.  Treat such relative paths as
        # references to the bundled default — installing from a wheel no
        # longer has a repo root to resolve against.
        if os.path.basename(output_config_path) in {
            'default_plots.yaml', 'default_comparison_plots.yaml'
        }:
            output_config_path = _default_plots
        else:
            output_config_path = os.path.abspath(output_config_path)
    # Self-heal for stale settings DBs that still point at the deleted
    # ``default_comparison_plots.yaml`` (comparison rules now live inside
    # default_plots.yaml via per-leaf ``scenario_rule``).
    if (os.path.basename(output_config_path) == 'default_comparison_plots.yaml'
            or not os.path.isfile(output_config_path)):
        if output_config_path != _default_plots:
            logging.info(
                "output-config-path %s not found or superseded — using "
                "bundled %s instead.", output_config_path, _default_plots,
            )
        output_config_path = _default_plots
    if active_configs is None:
        active_configs = ['default']
    if plot_rows is None:
        plot_rows = (0, 167)
    if output_location is None:
        output_location = fallback_output_location or ''
    if plot_file_format is None:
        plot_file_format = 'png'

    return write_methods, output_config_path, active_configs, plot_rows, output_location, plot_file_format


def write_outputs(scenario_name, output_config_path=None, active_configs=None, output_funcs=None, output_location=None, subdir=None, read_parquet_dir=False, write_methods=None, plot_rows=None, debug=False, single_result=None, settings_db_url=None, fallback_output_location=None, plot_file_format=None, raw_output_dir=None, only_first_file=False, timing_recorder=None, flex_data=None, solution=None, solve_name=None, solve_steps=None):
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
        flex_data: FlexData (polars input bundle) — required after Δ.31 for
            the in-memory parameter / set reading.  Pass-through from the
            orchestration step's last solve.
        solution: polar_high.Solution — required after Δ.31; used by the
            in-memory read_parameters for entity_all_capacity (and friends)
            derivation.
        solve_name: complete solve identifier (e.g. ``"y2020_2day_dispatch"``)
            used as the leading ``solve`` index level on the wide-format
            params / sets.  Defaults to ``scenario_name``.
        solve_steps: optional ordered list of ``(solve_name, FlexData)``
            pairs covering every sub-solve (roll) of the orchestration
            cascade.  Multi-solve / rolling scenarios MUST supply this
            so ``par`` and ``s`` are unioned across every sub-solve's
            dt axis — otherwise they would only carry the last roll's
            (d, t) rows while ``v`` carries the union from per-sub-
            solve parquet aggregation, and pandas joins/muls would
            mismatch.  Single-solve scenarios can ignore.
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
    start = log_time("Read configuration files", start, timing_recorder)

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
        start = log_time("Read parquet files", start, timing_recorder)

    # Read original raw outputs from FlexTool
    else:
        par, s, v = _read_outputs(
            raw_output_dir or 'output_raw',
            flex_data=flex_data,
            solution=solution,
            solve_name=solve_name or scenario_name,
            solve_steps=solve_steps,
        )
        start = log_time("Read flextool outputs", start, timing_recorder)

        # Pre-process results to be closer to what needed for output writing
        r = post_process_results(par, s, v)
        start = log_time("Post-processed outputs", start, timing_recorder)

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
        start = log_time("Formatted for output", start, timing_recorder)

    # Write files for debugging purposes
    if debug:
        open('namespace_structure.txt', 'w').close()
        print_namespace_structure(r, 'r')
        print_namespace_structure(s, 's')
        print_namespace_structure(v, 'v')
        print_namespace_structure(par, 'par')
        start = log_time("Wrote debugging files", start, timing_recorder)

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

        start = log_time("Wrote to parquet", start, timing_recorder)

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
            start = log_time("Computed plot plans", start, timing_recorder)
        except Exception as exc:
            logging.warning("Plot plan computation failed (non-fatal): %s", exc)

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
            start = log_time("Wrote dispatch metadata", start, timing_recorder)
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

        start = log_time('Plotted figures', start, timing_recorder)

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

        start = log_time('Wrote to csv', start, timing_recorder)

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

        start = log_time('Wrote to Excel', start, timing_recorder)

    # Phase F — write manifest.json describing the output bundle.
    # Idempotent (safe to call multiple times); rooted at the work
    # folder (parent of output_raw/).  Failure is non-fatal so it can't
    # block a successful cascade.
    try:
        from flextool.engine_polars._parquet_bundle import write_manifest
        raw_dir = raw_output_dir or 'output_raw'
        bundle_root = os.path.dirname(raw_dir) or '.'
        write_manifest(bundle_root)
        start = log_time('Wrote manifest.json', start, timing_recorder)
    except Exception as exc:  # noqa: BLE001
        logging.warning("Manifest write failed (non-fatal): %s", exc)
