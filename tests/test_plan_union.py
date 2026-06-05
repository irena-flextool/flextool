"""Tests for the merged-config comparison helpers in plan_union.py.

The legacy ``normalize_*`` and per-scenario plan-parquet helpers are gone;
comparison rendering is now driven by ``scenario_rule`` on the single
config (see ``derive_comparison_config``) and unioned raw parquets (see
``union_raw_data``).
"""
from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from flextool.lean_parquet import write_lean_parquet
from flextool.plot_outputs.config import PlotConfig
from flextool.scenario_comparison.plan_union import (
    derive_comparison_config,
    has_comparison_view,
    union_raw_data,
)


# ---------------------------------------------------------------------------
#  derive_comparison_config
# ---------------------------------------------------------------------------


def test_derive_pure_inserts_scenario_into_col_part():
    cfg = PlotConfig(
        plot_name="bar",
        map_dimensions_for_plots=["d_e", "t_l"],
        scenario_rule="g",
    )
    out = derive_comparison_config(cfg)
    assert out.map_dimensions_for_plots == ["d_se", "t_gl"]


def test_derive_with_two_col_dims():
    cfg = PlotConfig(
        plot_name="bar",
        map_dimensions_for_plots=["ed_p", "bg_u"],
        scenario_rule="g",
    )
    out = derive_comparison_config(cfg)
    assert out.map_dimensions_for_plots == ["ed_sp", "bg_gu"]


def test_derive_applies_simple_overrides():
    cfg = PlotConfig(
        plot_name="bar",
        map_dimensions_for_plots=["d_e", "t_l"],
        max_subplots_per_file=8,
        legend="right",
        scenario_rule="l",
        comparison_overrides={"max_subplots_per_file": 2, "legend": "shared"},
    )
    out = derive_comparison_config(cfg)
    assert out.max_subplots_per_file == 2
    assert out.legend == "shared"
    # Auto-derived map_dims still wins when override doesn't include it
    assert out.map_dimensions_for_plots == ["d_se", "t_ll"]


def test_derive_lets_override_replace_map_dimensions():
    """When the comparison view wants different rules than 'single + scenario_rule'.

    Used by configs where the chart author chose a different visual
    treatment for an existing dim in comparison mode (e.g. period as bar
    in single, but as expand-axis in comparison).
    """
    cfg = PlotConfig(
        plot_name="cap",
        map_dimensions_for_plots=["ed_p", "bg_u"],
        scenario_rule="g",
        comparison_overrides={"map_dimensions_for_plots": ["ed_sp", "eb_gu"]},
    )
    out = derive_comparison_config(cfg)
    # Override should win, not the auto-derived ['ed_sp', 'bg_gu']
    assert out.map_dimensions_for_plots == ["ed_sp", "eb_gu"]


def test_derive_works_on_dict_input():
    cfg = {
        "plot_name": "bar",
        "map_dimensions_for_plots": ["d_e", "t_l"],
        "scenario_rule": "g",
        "comparison_overrides": {"legend": "shared"},
    }
    out = derive_comparison_config(cfg)
    assert out["map_dimensions_for_plots"] == ["d_se", "t_gl"]
    assert out["legend"] == "shared"
    # original unchanged
    assert cfg["map_dimensions_for_plots"] == ["d_e", "t_l"]


def test_derive_raises_when_scenario_rule_missing():
    cfg = PlotConfig(
        plot_name="bar", map_dimensions_for_plots=["d_e", "t_l"],
    )
    with pytest.raises(ValueError, match="scenario_rule"):
        derive_comparison_config(cfg)


def test_derive_without_scenario_rule_uses_override_map_dimensions():
    """No scenario_rule, but comparison_overrides give the full layout.

    scenario_rule is irrelevant here: the override fully specifies
    map_dimensions_for_plots (including the scenario dim), so no
    auto-derivation is needed and the config still renders.
    """
    cfg = PlotConfig(
        plot_name="bar",
        map_dimensions_for_plots=["d_e", "t_l"],
        comparison_overrides={"map_dimensions_for_plots": ["d_se", "t_gl"]},
    )
    out = derive_comparison_config(cfg)
    assert out.map_dimensions_for_plots == ["d_se", "t_gl"]


def test_derive_without_scenario_rule_override_md_on_dict():
    cfg = {
        "plot_name": "bar",
        "map_dimensions_for_plots": ["d_e", "t_l"],
        "comparison_overrides": {"map_dimensions_for_plots": ["d_se", "t_gl"]},
    }
    out = derive_comparison_config(cfg)
    assert out["map_dimensions_for_plots"] == ["d_se", "t_gl"]


def test_has_comparison_view():
    # scenario_rule set → renderable
    assert has_comparison_view(
        PlotConfig(plot_name="x", map_dimensions_for_plots=["d_e", "t_l"],
                   scenario_rule="g")
    )
    # override map_dimensions_for_plots set, no scenario_rule → renderable
    assert has_comparison_view(
        PlotConfig(plot_name="x", map_dimensions_for_plots=["d_e", "t_l"],
                   comparison_overrides={"map_dimensions_for_plots": ["d_se", "t_gl"]})
    )
    # neither → not renderable
    assert not has_comparison_view(
        PlotConfig(plot_name="x", map_dimensions_for_plots=["d_e", "t_l"])
    )
    # comparison_overrides without map_dimensions_for_plots → not renderable
    assert not has_comparison_view(
        PlotConfig(plot_name="x", map_dimensions_for_plots=["d_e", "t_l"],
                   comparison_overrides={"legend": "shared"})
    )
    # dict form works too
    assert has_comparison_view({"scenario_rule": "g"})
    assert not has_comparison_view({"legend": "shared"})


def test_derive_raises_on_malformed_index_types():
    cfg = PlotConfig(
        plot_name="bar",
        map_dimensions_for_plots=["dewithoutunderscore", "rules_x"],
        scenario_rule="g",
    )
    with pytest.raises(ValueError, match="index_types"):
        derive_comparison_config(cfg)


def test_derive_raises_on_malformed_rules():
    cfg = PlotConfig(
        plot_name="bar",
        map_dimensions_for_plots=["d_e", "rules_no_underscore"[:5]],
        scenario_rule="g",
    )
    # 'rules' (5 chars, no underscore) — should fail
    cfg.map_dimensions_for_plots = ["d_e", "rules"]
    with pytest.raises(ValueError, match="rules"):
        derive_comparison_config(cfg)


# ---------------------------------------------------------------------------
#  union_raw_data
# ---------------------------------------------------------------------------


def _write_per_scenario_raw(
    project: pathlib.Path, scenario: str, result_key: str, df: pd.DataFrame,
) -> None:
    out_dir = project / "output_parquet" / scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    write_lean_parquet(df, out_dir / f"{result_key}.parquet")


def test_union_raw_concats_with_scenario_at_outermost(tmp_path: pathlib.Path):
    df_a = pd.DataFrame(
        [[1.0, 2.0]],
        index=pd.Index(["x"], name="entity"),
        columns=pd.MultiIndex.from_tuples(
            [("scenA", "p1"), ("scenA", "p2")], names=["scenario", "param"],
        ),
    )
    df_b = pd.DataFrame(
        [[10.0, 20.0]],
        index=pd.Index(["x"], name="entity"),
        columns=pd.MultiIndex.from_tuples(
            [("scenB", "p1"), ("scenB", "p2")], names=["scenario", "param"],
        ),
    )
    _write_per_scenario_raw(tmp_path, "scenA", "rk", df_a)
    _write_per_scenario_raw(tmp_path, "scenB", "rk", df_b)

    out = union_raw_data(tmp_path, ["scenA", "scenB"], "rk")
    assert out is not None
    assert out.shape == (1, 4)
    assert out.columns.names == ["scenario", "param"]
    assert sorted(out.columns.get_level_values("scenario").unique()) == ["scenA", "scenB"]


def test_union_raw_uses_viewer_scenario_name_not_embedded(tmp_path: pathlib.Path):
    """Embedded scenario name in the parquet may differ from folder name.

    union_raw_data must use the folder/viewer name (the keys= arg) and not
    leak the embedded one — handles sensitivity replicas where the folder
    is e.g. ``scenA_2`` but the parquet's embedded scenario is ``scenA``.
    """
    df = pd.DataFrame(
        [[1.0]],
        index=pd.Index(["x"], name="entity"),
        columns=pd.MultiIndex.from_tuples(
            [("scenA", "p1")], names=["scenario", "param"],  # embedded = scenA
        ),
    )
    _write_per_scenario_raw(tmp_path, "scenA_2", "rk", df)  # folder = scenA_2

    out = union_raw_data(tmp_path, ["scenA_2"], "rk")
    assert out is not None
    assert list(out.columns.get_level_values("scenario").unique()) == ["scenA_2"]


def test_union_raw_returns_none_when_no_files(tmp_path: pathlib.Path):
    out = union_raw_data(tmp_path, ["scenA"], "rk")
    assert out is None


def test_union_raw_skips_missing_scenarios(tmp_path: pathlib.Path):
    df = pd.DataFrame(
        [[1.0]],
        index=pd.Index(["x"], name="entity"),
        columns=pd.MultiIndex.from_tuples(
            [("scenA", "p1")], names=["scenario", "param"],
        ),
    )
    _write_per_scenario_raw(tmp_path, "scenA", "rk", df)

    out = union_raw_data(tmp_path, ["scenA", "missing_scenario"], "rk")
    assert out is not None
    assert list(out.columns.get_level_values("scenario").unique()) == ["scenA"]
