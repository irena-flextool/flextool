"""The scenario comparison keeps every processed result table.

``TimeSeriesResults.from_dict`` drops a parquet table that has no field, so a new output table without a field
silently vanishes from the comparison plots (the horizon tables ``nodeGroup_g_p`` / ``nodeGroup_VRE_share_g`` did).
"""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pandas as pd

from flextool.engine.output._parquet_bundle import REGISTRY
from flextool.lean_parquet import write_lean_parquet
from flextool.scenario_comparison.data_models import _DISPATCH_MAPPING_FILENAMES, TimeSeriesResults

# Processed tables the comparison reads separately (not result variables).
_NOT_RESULTS = {"timeline_breaks"}


def test_every_processed_table_has_a_field() -> None:
    known = {f.name for f in fields(TimeSeriesResults)}
    processed = {key for key, spec in REGISTRY.items() if spec.category == "processed"}
    missing = sorted(processed - known - _DISPATCH_MAPPING_FILENAMES - _NOT_RESULTS)
    assert missing == [], f"add fields to scenario_comparison/data_models.py: {missing}"


def _write_scenario(base: Path, scenario: str, share: float) -> None:
    """The two horizon tables as write_outputs writes them (scenario level prepended to the columns)."""
    d = base / scenario
    d.mkdir(parents=True)
    g_p = pd.DataFrame(
        {"1. Loss of load share": [0.05], "2. VRE share of demand": [share], "3. Excess load share": [0.0],
         "4. Curtailed VRE of demand": [0.07], "5. Inflow (horizon total)": [5.0e6],
         "6. Curtailed VRE of potential VRE": [0.07]},
        index=pd.Index(["electricity"], name="group"))
    g_p.columns.name = "parameter"
    vre = pd.DataFrame({"electricity": [share]}, index=pd.Index(["total"], name="horizon"))
    vre.columns.name = "group"
    for name, df in (("nodeGroup_g_p", g_p), ("nodeGroup_VRE_share_g", vre)):
        write_lean_parquet(pd.concat({scenario: df}, axis=1, names=["scenario"]), d / f"{name}.parquet")


def test_comparison_keeps_and_plots_the_horizon_tables(tmp_path: Path, capsys) -> None:
    """The comparison run (the CLI path) reads both tables and draws their 'total' plots."""
    from flextool._resources import package_data_path
    from flextool.scenario_comparison.db_reader import build_scenario_folders_from_dir, get_scenario_results
    from flextool.scenario_comparison.orchestrator import run

    base = tmp_path / "output_parquet"
    _write_scenario(base, "A", 0.93)
    _write_scenario(base, "B", 0.95)
    folders = build_scenario_folders_from_dir(base, ["A", "B"])
    _folders, results = get_scenario_results(parquet_subdir="", scenario_folders=folders)
    assert "not registered" not in capsys.readouterr().out
    assert results.nodeGroup_g_p is not None and results.nodeGroup_VRE_share_g is not None
    assert set(results.nodeGroup_VRE_share_g.columns.get_level_values("scenario")) == {"A", "B"}

    plot_dir = tmp_path / "plots"
    run(db_url=None, parquet_subdir="", plot_dir=str(plot_dir),
        output_config_path=str(package_data_path("schemas/default_plots.yaml")), active_configs=["sum_periods"],
        plot_rows=[0, 167], write_to_xlsx=False, write_dispatch_xlsx=False, write_to_ods=False, show_plots=False,
        dispatch_plots=False, scenario_folders=folders)
    assert "not registered" not in capsys.readouterr().out
    names = {p.name for p in plot_dir.iterdir()}
    assert "NodeGroup indicators total.png" in names
    assert "NodeGroup VRE share total.png" in names
