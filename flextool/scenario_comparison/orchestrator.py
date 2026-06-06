"""Top-level orchestration: ties together data loading, config, and plotting."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yaml

from flextool.lean_parquet import write_lean_parquet

from flextool.scenario_comparison.config_builder import (
    assign_palette_colors,
    create_or_update_dispatch_config,
    discover_dispatch_entities,
    get_scenarios_from_config,
)
from flextool.scenario_comparison.data_models import DispatchMappings
from flextool.scenario_comparison.db_reader import get_scenario_results
from flextool.scenario_comparison.dispatch_mappings import (
    combine_dispatch_mappings,
    data_scenario_tags,
)
from flextool.scenario_comparison.dispatch_plots import create_dispatch_plots
from flextool.scenario_comparison.plan_union import (
    derive_comparison_config,
    has_comparison_view,
)
from flextool.plot_outputs.config import _is_single_config, flatten_new_format
from flextool.plot_outputs.orchestrator import plot_dict_of_dataframes
from flextool.plot_outputs.color_template import resolve_plot_settings_path


def _seed_dispatch_colors_into_plot_settings(
    project_path: Path,
    color_path: Path,
    mappings: DispatchMappings,
    dispatch_scenarios: list[str],
) -> None:
    """Auto-seed discovered dispatch entity / scenario colors additively.

    Relocates ``config_builder``'s color-seeding intent from ``config.yaml``
    to the project ``plot_settings.yaml``: discover the run's group / unit /
    connection / scenario names, assign each a default palette color, and
    splice the genuinely-new ones into the project settings file without
    disturbing existing content.

    When *color_path* is the bundled fallback (the project has no own
    ``plot_settings.yaml`` yet), seed the project copy first via
    :func:`flextool.gui.project_utils.seed_plot_settings` so we never write
    the packaged default.  Non-fatal: any failure is logged and the run
    continues.
    """
    from flextool.plot_outputs.color_template import (
        _clear_cache,
        _default_path,
    )
    from flextool.scenario_comparison.plot_settings_seed import (
        seed_colors_into_plot_settings,
    )

    try:
        discovered = discover_dispatch_entities(mappings, dispatch_scenarios)
        entity_colors: dict[str, dict[str, str]] = {}
        for cls in ("group", "unit", "connection"):
            names = discovered.get(cls) or []
            if names:
                entity_colors[cls] = assign_palette_colors(names)
        scenario_colors = assign_palette_colors(
            discovered.get("scenarios") or [], use_tab10=True
        )
        if not any(entity_colors.values()) and not scenario_colors:
            return

        # Resolve to the project's own file; if the resolver handed back the
        # bundled default, seed the project copy first (never write the
        # package file).
        target = Path(color_path)
        if target == Path(_default_path()):
            from flextool.gui.project_utils import seed_plot_settings
            target = seed_plot_settings(project_path)

        changed = seed_colors_into_plot_settings(
            target, entity_colors, scenario_colors
        )
        if changed:
            # The settings file changed on disk — drop the mtime-keyed cache
            # so downstream readers in this same process see the new colors.
            _clear_cache()
            print(f"Seeded dispatch colors into {target}")
    except Exception as exc:  # noqa: BLE001 - seeding must never abort a run
        import logging as _logging
        _logging.warning(
            "Auto-seeding plot_settings.yaml failed (non-fatal): %s", exc
        )


def _derive_comparison_settings(plots_flat: dict) -> dict:
    """Map each leaf config dict through ``derive_comparison_config``.

    Drops entries with no comparison view — those lacking ``scenario_rule``
    *and* an explicit ``comparison_overrides.map_dimensions_for_plots`` (see
    :func:`~flextool.scenario_comparison.plan_union.has_comparison_view`).
    Produces the same flat-dict shape that ``plot_dict_of_dataframes`` and
    ``compute_all_plot_plans`` expect.
    """
    out: dict = {}
    for rk, cfg_dict in plots_flat.items():
        if not isinstance(cfg_dict, dict):
            continue
        if _is_single_config(cfg_dict):
            # Direct config (no sub_configs)
            if not has_comparison_view(cfg_dict):
                continue
            try:
                out[rk] = derive_comparison_config(cfg_dict)
            except (ValueError, TypeError):
                continue
            continue
        # Named sub-configs
        new_sub: dict = {}
        for sub, leaf in cfg_dict.items():
            if not isinstance(leaf, dict):
                continue
            if not has_comparison_view(leaf):
                continue
            try:
                new_sub[sub] = derive_comparison_config(leaf)
            except (ValueError, TypeError):
                continue
        if new_sub:
            out[rk] = new_sub
    return out


def run(
    db_url: str | None,
    parquet_subdir: str,
    plot_dir: str,
    output_config_path: str,
    active_configs: list[str],
    plot_rows: list[int],
    write_to_xlsx: bool,
    write_dispatch_xlsx: bool,
    write_to_ods: bool,
    show_plots: bool,
    dispatch_plots: bool,
    plot_file_format: str = 'png',
    scenario_folders: dict[str, str] | None = None,
    excel_dir: str | None = None,
    shared_legend: bool = True,
    only_first_file: bool = False,
    comparison_parquet_dir: str | None = None,
) -> None:
    """Run the full scenario-comparison pipeline.

    Takes already-resolved parameters (CLI > settings DB > defaults)
    and orchestrates: load data → build config → generate plots → write Excel.

    When *scenario_folders* is provided the database is not queried and
    *db_url* may be ``None``.
    """
    with open(output_config_path, 'r', encoding='utf-8') as f:
        settings = yaml.safe_load(f)

    scenario_folders, results = get_scenario_results(
        db_url=db_url, parquet_subdir=parquet_subdir,
        scenario_folders=scenario_folders,
    )
    combined_dfs = results.to_dict()

    os.makedirs(plot_dir, exist_ok=True)

    # The plot color template comes from the project's ``plot_settings.yaml``
    # when present, else the bundled default.  ``plot_dir`` is always a
    # per-project subdir (``<project>/output_plot_comparisons``), so its
    # parent is the project root.
    color_path = resolve_plot_settings_path(Path(plot_dir).parent)

    scenarios = list(scenario_folders.keys())

    # Load and combine dispatch mappings across all scenarios
    if scenario_folders:
        mappings = combine_dispatch_mappings(scenario_folders, parquet_subdir)
        combined_mapping_dfs = {
            k: v for k, v in vars(mappings).items() if v is not None
        }
    else:
        mappings = DispatchMappings()
        combined_mapping_dfs = {}

    # Folder names (``scenarios``) drive paths, _metadata.json and the
    # viewer-set match, but dispatch slicing keys off the scenario tag
    # embedded in the data — which differs when a folder carries the GUI
    # run-index suffix (folder S2_Dry_1 → tag S2_Dry).  Resolve the tags
    # from the data and use them for every dispatch slice.
    dispatch_scenarios = data_scenario_tags(mappings) or scenarios

    # Derive group_node_df for summary plots (needs 'scenario' as column)
    group_node_combined = combined_mapping_dfs.get('group_node')
    if group_node_combined is not None and not group_node_combined.empty:
        group_node_combined.reset_index()

    # Create or update dispatch config
    dispatch_config = None
    if dispatch_plots:
        dispatch_config = create_or_update_dispatch_config(
            plot_dir, results, dispatch_scenarios, mappings
        )
        # Additively seed discovered entity / scenario colors into the
        # project ``plot_settings.yaml`` (the durable, comment-preserving
        # colors file the renderers read).  ``config.yaml`` is still written
        # above for now (4.1 made the renderers ignore its colors).
        _seed_dispatch_colors_into_plot_settings(
            Path(plot_dir).parent, color_path, mappings, dispatch_scenarios
        )
        # Seeding may have created the project ``plot_settings.yaml`` (when
        # only the bundled default existed); re-resolve so the renderers and
        # the viewer plan use the freshly-seeded project file.
        color_path = resolve_plot_settings_path(Path(plot_dir).parent)

    # Flatten new-format entries to flat result_key mapping for downstream use.
    # Then derive comparison-mode configs from each leaf (single rules +
    # scenario_rule), since the merged ``default_plots.yaml`` carries
    # single-mode rules.  Entries without ``scenario_rule`` have no
    # comparison rendering and are dropped here.
    settings['plots'] = _derive_comparison_settings(
        flatten_new_format(settings.get('plots', {}))
    )

    # If shared_legend is disabled, replace 'shared' legend with 'right' in all plot configs
    if not shared_legend:
        for result_name, config_dict in settings['plots'].items():
            if not isinstance(config_dict, dict):
                continue
            if _is_single_config(config_dict):
                # Single config: the dict itself is the plot settings
                if config_dict.get('legend') == 'shared':
                    config_dict['legend'] = 'right'
            else:
                # Named configs: iterate sub-dicts
                for config_name, plot_cfg in config_dict.items():
                    if isinstance(plot_cfg, dict) and plot_cfg.get('legend') == 'shared':
                        plot_cfg['legend'] = 'right'

    # Load timeline breaks from all scenario parquet dirs.
    # For --parquet-base-dir mode (subdir=''): parquets at base_dir/name/
    # For DB mode (subdir='output_parquet'): parquets at folder/subdir/name/
    from flextool.plot_outputs.format_helpers import load_timeline_breaks
    break_dirs = []
    for name, folder in scenario_folders.items():
        if parquet_subdir:
            break_dirs.append(os.path.join(folder, parquet_subdir, name))
        else:
            break_dirs.append(os.path.join(folder, name))
    break_times = load_timeline_breaks(*break_dirs)

    # Write combined comparison parquets if output directory specified
    if comparison_parquet_dir:
        os.makedirs(comparison_parquet_dir, exist_ok=True)
        for name, df in combined_dfs.items():
            if not df.empty:
                write_lean_parquet(df, os.path.join(comparison_parquet_dir, f"{name}.parquet"))
        # Also write a metadata file with the scenario list
        import json
        meta = {"scenarios": scenarios}
        with open(os.path.join(comparison_parquet_dir, "_metadata.json"), "w") as f:
            json.dump(meta, f)
        if break_times:
            # Save break times so the viewer can load them
            bt_df = pd.DataFrame({"break_time": list(break_times)})
            write_lean_parquet(bt_df, os.path.join(comparison_parquet_dir, "timeline_breaks.parquet"), index=False)
        print(f"Wrote comparison parquets to: {comparison_parquet_dir}")

    # Compute and save dispatch metadata (cross-scenario ylims) for the viewer
    if combined_mapping_dfs and comparison_parquet_dir:
        try:
            from flextool.scenario_comparison.dispatch_plots import compute_dispatch_metadata
            # Use same timeline as dispatch plots
            if plot_rows and len(plot_rows) >= 2:
                meta_timeline = (int(plot_rows[0]), int(plot_rows[1]) + 1)
            else:
                meta_timeline = (0, 168)
            dispatch_meta = compute_dispatch_metadata(
                results, mappings, dispatch_scenarios, meta_timeline,
            )
            import json as _json
            meta_path = os.path.join(comparison_parquet_dir, "_dispatch_metadata.json")
            with open(meta_path, "w") as f:
                _json.dump(dispatch_meta, f, indent=2)
            print(f"Wrote dispatch metadata to: {meta_path}")
        except Exception as exc:
            import logging as _logging
            _logging.warning("Dispatch metadata computation failed (non-fatal): %s", exc)

    # Compute plot plans for the viewer (always in comparison parquet dir)
    if comparison_parquet_dir:
        try:
            from flextool.plot_outputs.orchestrator import compute_all_plot_plans
            compute_all_plot_plans(
                combined_dfs, settings.get('plots', {}), comparison_parquet_dir,
                active_settings=active_configs, plot_rows=plot_rows,
                break_times=break_times,
                strip_scenario_level=False,
                color_path=color_path,
            )
            print("Computed plot plans for viewer")
        except Exception as exc:
            import logging as _logging
            _logging.warning("Plot plan computation failed (non-fatal): %s", exc)

    # Generate comparison plots from the merged settings derived above.
    plot_dict_of_dataframes(
        combined_dfs, plot_dir, settings['plots'],
        active_settings=active_configs, plot_rows=plot_rows,
        delete_existing_plots=True, plot_file_format=plot_file_format,
        only_first_file=only_first_file,
        break_times=break_times,
        color_path=color_path,
    )
    print(f'\nPlotted comparison of {len(scenario_folders)} scenarios to folder: {plot_dir}')

    # Generate dispatch plots
    if dispatch_plots:
        if dispatch_config and combined_mapping_dfs:
            print("\nGenerating dispatch plots...")
            create_dispatch_plots(
                results, mappings, dispatch_config, plot_dir,
                scenarios=get_scenarios_from_config(dispatch_config),
                show_plot=show_plots,
                write_xlsx=write_dispatch_xlsx,
                break_times=break_times,
                plot_rows=plot_rows,
            )
        else:
            print("Warning: Cannot generate dispatch plots - missing dispatch mappings")

    # Write to Excel (combined results)
    if write_to_xlsx:
        # Load rename/export config: use the comparison YAML's rename section
        # if present, otherwise fall back to default_plots.yaml.
        rename_raw = settings.get('rename')
        if not rename_raw:
            try:
                default_plots_path = os.path.join(
                    os.path.dirname(output_config_path), 'default_plots.yaml',
                )
                with open(default_plots_path, 'r', encoding='utf-8') as f_dp:
                    rename_raw = yaml.safe_load(f_dp).get('rename', {})
            except Exception:
                rename_raw = {}

        def _parse_rename(entry):
            if isinstance(entry, list) and len(entry) >= 2:
                return str(entry[0]), bool(entry[1])
            return str(entry), True

        filename = 'compare_' + str(len(scenario_folders)) + '_scens.xlsx'
        target_dir = excel_dir if excel_dir is not None else plot_dir
        excel_path = os.path.join(target_dir, filename)
        # Build list of (sheet_name, df) sorted alphabetically
        sheets: list[tuple[str, pd.DataFrame]] = []
        used_names: set[str] = set()
        for name, df in combined_dfs.items():
            display_name, export = _parse_rename(rename_raw.get(name, name))
            if not export:
                continue
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
        print(f'\nWrote comparison of {len(scenario_folders)} scenarios to xlsx file: {excel_path}')

    print('\nDone!')
