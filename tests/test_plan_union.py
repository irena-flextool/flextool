"""Tests for the lazy plan-parquet union path used by the comparison view.

Phase E (refined) drops the legacy combined-parquet fallback: every
comparison config is served from per-scenario plan parquets unioned
along a top-level ``scenario`` column index, with the comparison
config optionally normalised by ``normalize_config_for_plan_union``
(currently only the 4 ``sdt_*`` configs need adjustment — the row-part
``s`` is FlexTool's ``solve`` dimension which the per-scenario compute
step has already collapsed).

These tests prove three things:

1. ``is_scenario_pivot_config`` is now a no-op (always ``False``) — no
   shipped comparison config is routed to the legacy fallback.
2. ``normalize_config_for_plan_union`` strips the row-part ``solve``
   dim only for configs that need it; everything else is returned
   unchanged.
3. End-to-end: per-scenario plan parquets unioned with
   ``union_plan_data`` and run through ``compute_live_plan`` +
   ``build_figure_from_plan`` produce a Figure for both a typical
   line config (``dt_se / tt_lu``) and a typical bar config
   (``d_se / b_ge``).
"""
from __future__ import annotations

import pathlib
import tempfile

import matplotlib
matplotlib.use("Agg")  # headless tests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import yaml

from flextool.lean_parquet import write_lean_parquet
from flextool.plot_outputs.config import PlotConfig
from flextool.plot_outputs.plan import (
    build_figure_from_plan, compute_live_plan,
)
from flextool.scenario_comparison.plan_union import (
    is_scenario_pivot_config,
    normalize_config_for_per_scenario_compute,
    normalize_config_for_plan_union,
    per_scenario_plan_path,
    union_plan_data,
)


# ---------------------------------------------------------------------------
# is_scenario_pivot_config — pure no-op for forward compat
# ---------------------------------------------------------------------------

def test_is_scenario_pivot_config_always_false():
    """The function exists for backward import-compat; never flags anything."""
    assert is_scenario_pivot_config(None) is False
    assert is_scenario_pivot_config({}) is False
    assert is_scenario_pivot_config(
        {"map_dimensions_for_plots": ["dt_se", "tt_lu"]}
    ) is False
    assert is_scenario_pivot_config(
        {"map_dimensions_for_plots": ["sdt_se", "mtt_lu"]}
    ) is False


def test_no_shipped_comparison_config_is_pivot():
    """Real configs from the shipped YAML — none should be pivot."""
    yaml_path = (
        pathlib.Path(__file__).parent.parent
        / "templates" / "default_comparison_plots.yaml"
    )
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    plots = cfg.get("plots", {}) or {}
    leaves: list[dict] = []
    for entry in plots.values():
        if not isinstance(entry, dict):
            continue
        for rk, sub in entry.items():
            if not isinstance(sub, dict) or rk in ("group", "order"):
                continue
            for scfg in sub.values():
                if isinstance(scfg, dict):
                    leaves.append(scfg)
    assert leaves, "expected at least one leaf config"
    assert all(not is_scenario_pivot_config(c) for c in leaves)


# ---------------------------------------------------------------------------
# normalize_config_for_plan_union
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "map_dims",
    [
        ["dt_se", "tt_lu"],          # most common: no row-part s
        ["d_se", "b_ge"],            # bar config
        ["d_seeg", "b_essu"],        # complex: no row-part s
        ["_se", "x_bx"],             # CO2: empty row part
    ],
)
def test_normalize_passthrough_for_non_solve_configs(map_dims):
    """Configs without a row-part ``s`` are returned unchanged."""
    cfg = {"map_dimensions_for_plots": list(map_dims)}
    out = normalize_config_for_plan_union(cfg)
    assert out is cfg  # exact identity — no copy made


@pytest.mark.parametrize(
    "map_dims, expected",
    [
        # All four shipped sdt_* configs:
        (["sdt_se", "mtt_lu"], ["dt_se", "tt_lu"]),
        (["sdt_se", "mti_lu"], ["dt_se", "ti_lu"]),
        (["sdt_sppg", "mtt_gllu"], ["dt_sppg", "tt_gllu"]),
        (["sdt_sppg", "mui_gllu"], ["dt_sppg", "ui_gllu"]),
        # Weighted-sum and weighted-avg variants — also collapsing rules:
        (["sdt_se", "ytt_lu"], ["dt_se", "tt_lu"]),
        (["sdt_se", "ztt_lu"], ["dt_se", "tt_lu"]),
    ],
)
def test_normalize_strips_solve_row_dim(map_dims, expected):
    """Leading row-part ``s`` with collapsing rule is stripped."""
    cfg_dict = {"map_dimensions_for_plots": list(map_dims)}
    out = normalize_config_for_plan_union(cfg_dict)
    assert out is not cfg_dict
    assert out["map_dimensions_for_plots"] == expected
    # Original dict is not mutated.
    assert cfg_dict["map_dimensions_for_plots"] == list(map_dims)


def test_normalize_does_not_strip_non_collapsing_solve_row():
    """If the row-part ``s`` rule is not collapsing, leave it alone.

    No shipped config currently uses such a shape, but the helper must
    stay conservative — non-collapsing rules (e.g. 't' for time-axis)
    on a row-part ``s`` would change semantics if stripped.
    """
    cfg = {"map_dimensions_for_plots": ["sdt_e", "ttt_l"]}  # row-rule for s = 't'
    out = normalize_config_for_plan_union(cfg)
    assert out is cfg


def test_normalize_handles_plotconfig_dataclass():
    """``PlotConfig`` dataclass instances also normalise via dataclasses.replace."""
    cfg = PlotConfig(
        plot_name="test",
        map_dimensions_for_plots=["sdt_se", "mtt_lu"],
    )
    out = normalize_config_for_plan_union(cfg)
    assert out is not cfg
    assert out.plot_name == "test"  # other fields preserved
    assert out.map_dimensions_for_plots == ["dt_se", "tt_lu"]
    # Original unchanged.
    assert cfg.map_dimensions_for_plots == ["sdt_se", "mtt_lu"]


def test_normalize_passes_plotconfig_through_unchanged():
    cfg = PlotConfig(
        plot_name="test",
        map_dimensions_for_plots=["dt_se", "tt_lu"],
    )
    out = normalize_config_for_plan_union(cfg)
    assert out is cfg


# ---------------------------------------------------------------------------
# normalize_config_for_per_scenario_compute  (Issue A: comparison-only plans)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "map_dims, expected",
    [
        # Common time-series shapes — strip leading col-part ``s`` + its rule.
        (["dt_se", "tt_lu"], ["dt_e", "tt_u"]),
        (["dt_see", "tt_luu"], ["dt_ee", "tt_uu"]),
        (["dt_seeg", "tt_luuu"], ["dt_eeg", "tt_uuu"]),
        # Bar configs — same mechanic on a non-time row.
        (["d_se", "b_ge"], ["d_e", "b_e"]),
        (["d_see", "b_gee"], ["d_ee", "b_ee"]),
        (["d_see", "y_gbb"], ["d_ee", "y_bb"]),
        # Complex bar with multiple col rules — leading col-part ``s`` and
        # its matching col-rule char (the *first* col-rule char) are
        # both stripped.  Row rules are preserved.  ``y_gxbxe``: row="y",
        # col rules="gxbxe" → drop col-rule[0]="g" → col rules="xbxe".
        (["d_seeee", "y_gxbxe"], ["d_eeee", "y_xbxe"]),
        # Row-part also has ``s`` (sdt_*); unaffected — we only strip col-part.
        (["sdt_se", "mtt_lu"], ["sdt_e", "mtt_u"]),
        (["sdt_sppg", "mtt_gllu"], ["sdt_ppg", "mtt_llu"]),
    ],
)
def test_normalize_per_scenario_strips_col_scenario_dim(map_dims, expected):
    """Leading column-part ``s`` and its matching col-rule char are stripped."""
    cfg_dict = {"map_dimensions_for_plots": list(map_dims)}
    out = normalize_config_for_per_scenario_compute(cfg_dict)
    assert out is not None
    assert out is not cfg_dict
    assert out["map_dimensions_for_plots"] == expected
    # Original dict not mutated.
    assert cfg_dict["map_dimensions_for_plots"] == list(map_dims)


@pytest.mark.parametrize(
    "map_dims",
    [
        # No column-part ``s`` at all (post-normalisation shape) — pass-through.
        ["dt_e", "tt_u"],
        ["d_eg", "b_ge"],
        ["_e", "x_x"],
    ],
)
def test_normalize_per_scenario_passthrough_when_no_col_scenario(map_dims):
    """Configs with no column-part ``s`` are returned unchanged."""
    cfg = {"map_dimensions_for_plots": list(map_dims)}
    out = normalize_config_for_per_scenario_compute(cfg)
    assert out is cfg  # exact identity — no copy made


def test_normalize_per_scenario_skips_column_only_scenario_config():
    """``[d_s, s_b]`` has no plotted column dim left after stripping ``s``.

    The function returns ``None`` so the caller can skip the config
    entirely — there is no meaningful single-scenario plan for a chart
    whose only column dim is scenario.
    """
    cfg = {"map_dimensions_for_plots": ["d_s", "s_b"]}
    out = normalize_config_for_per_scenario_compute(cfg)
    assert out is None


def test_normalize_per_scenario_handles_plotconfig_dataclass():
    """``PlotConfig`` dataclass instances also normalise via dataclasses.replace."""
    cfg = PlotConfig(
        plot_name="test",
        map_dimensions_for_plots=["dt_se", "tt_lu"],
    )
    out = normalize_config_for_per_scenario_compute(cfg)
    assert out is not None
    assert out is not cfg
    assert out.plot_name == "test"  # other fields preserved
    assert out.map_dimensions_for_plots == ["dt_e", "tt_u"]
    # Original unchanged.
    assert cfg.map_dimensions_for_plots == ["dt_se", "tt_lu"]


def test_normalize_per_scenario_passes_clean_plotconfig_through():
    """A PlotConfig with no col-part ``s`` is returned identity-equal."""
    cfg = PlotConfig(
        plot_name="test",
        map_dimensions_for_plots=["dt_e", "tt_u"],
    )
    out = normalize_config_for_per_scenario_compute(cfg)
    assert out is cfg


def test_normalize_per_scenario_real_yaml_sweep():
    """Sweep every shipped comparison config — none should crash, results sane.

    Asserts: every leaf config either yields a normalised result whose
    column part has no leading ``s``, or returns ``None`` (column-only
    scenario configs).  This is a smoke test against the full YAML
    catalogue.
    """
    yaml_path = (
        pathlib.Path(__file__).parent.parent
        / "templates" / "default_comparison_plots.yaml"
    )
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg_yaml = yaml.safe_load(f)
    plots = cfg_yaml.get("plots", {}) or {}
    sweep_count = 0
    for entry in plots.values():
        if not isinstance(entry, dict):
            continue
        for rk, sub in entry.items():
            if not isinstance(sub, dict) or rk in ("group", "order"):
                continue
            for scfg in sub.values():
                if not isinstance(scfg, dict):
                    continue
                map_dims = scfg.get("map_dimensions_for_plots")
                if not isinstance(map_dims, list) or len(map_dims) < 2:
                    continue
                sweep_count += 1
                out = normalize_config_for_per_scenario_compute(scfg)
                if out is None:
                    # Verify column part really was column-only-``s``.
                    idx_str = map_dims[0]
                    assert "_" in idx_str
                    _, col_idx = idx_str.split("_", 1)
                    assert col_idx == "s"
                    continue
                # Returned a normalised (or unchanged) config — col part
                # must not start with ``s`` after the call.
                new_idx = out["map_dimensions_for_plots"][0]
                if "_" in new_idx:
                    _, new_col = new_idx.split("_", 1)
                    assert not new_col.startswith("s"), (
                        f"col part still begins with s after normalisation: "
                        f"{map_dims} -> {out['map_dimensions_for_plots']}"
                    )
    assert sweep_count > 50, f"only {sweep_count} configs swept — YAML changed?"


# ---------------------------------------------------------------------------
# End-to-end: union → compute_live_plan → build_figure_from_plan
# ---------------------------------------------------------------------------

def _write_per_scenario_plan(
    project_path: pathlib.Path, scenario: str, result_key: str,
    sub_config: str, df: pd.DataFrame,
) -> pathlib.Path:
    """Write a per-scenario plan parquet at the expected layout."""
    p = per_scenario_plan_path(project_path, scenario, result_key, sub_config)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_lean_parquet(df, p)
    return p


def _make_time_plan_df(rng: np.random.Generator, n_t: int = 24, n_e: int = 3) -> pd.DataFrame:
    """Per-scenario plan parquet for a ``dt_e`` (time) result.

    Mirrors what ``compute_all_plot_plans`` writes for
    ``node_slack_up_dt_e__default`` once the scenario level has been
    stripped: rows = ``(period, time)``, cols = single-level ``node``.
    """
    idx = pd.MultiIndex.from_product(
        [["2030"], [f"t{i:03d}" for i in range(n_t)]],
        names=["period", "time"],
    )
    cols = pd.Index([f"node{i}" for i in range(n_e)], name="node")
    return pd.DataFrame(rng.standard_normal((n_t, n_e)), index=idx, columns=cols)


def _make_bar_plan_df(rng: np.random.Generator, n_e: int = 4) -> pd.DataFrame:
    """Per-scenario plan parquet for a ``d_e`` (bar) result.

    Rows = ``period`` (single level), cols = single-level ``node``.
    """
    idx = pd.Index(["2030", "2040"], name="period")
    cols = pd.Index([f"node{i}" for i in range(n_e)], name="node")
    return pd.DataFrame(rng.standard_normal((2, n_e)), index=idx, columns=cols)


def test_union_path_e2e_line_config(tmp_path: pathlib.Path):
    """End-to-end: time-series config (``dt_se / tt_lu``) renders a Figure.

    This is the most common comparison-config shape (used by node-flow,
    slack-up, slack-down, curtailment, unit-output, connection-flow,
    NodeGroup-dispatch hourly variants, etc.).
    """
    rng = np.random.default_rng(seed=0)
    scenarios = ["scenA", "scenB"]
    rk, sub = "node_slack_up_dt_e", "default"
    for s in scenarios:
        _write_per_scenario_plan(
            tmp_path, s, rk, sub, _make_time_plan_df(rng),
        )

    df = union_plan_data(tmp_path, scenarios, rk, sub)
    assert df is not None
    assert isinstance(df.columns, pd.MultiIndex)
    assert df.columns.names == ["scenario", "node"]
    assert df.index.names == ["period", "time"]

    cfg = PlotConfig(
        plot_name="loss_load",
        map_dimensions_for_plots=["dt_se", "tt_lu"],
        legend="shared",
    )
    cfg = normalize_config_for_plan_union(cfg)
    plan = compute_live_plan(df, cfg, "loss_load")
    assert plan is not None
    assert plan.chart_type in ("lines", "stack")

    fig = build_figure_from_plan(plan, file_index=0)
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_union_path_e2e_bar_config(tmp_path: pathlib.Path):
    """End-to-end: bar config (``d_se / b_ge``) renders a Figure.

    Used by the period-bar variants (``slack_up_d_e``,
    ``slack_down_d_e``, ``connection_d_eee``, etc.).
    """
    rng = np.random.default_rng(seed=1)
    scenarios = ["scenA", "scenB", "scenC"]
    rk, sub = "node_slack_up_d_e", "default"
    for s in scenarios:
        _write_per_scenario_plan(
            tmp_path, s, rk, sub, _make_bar_plan_df(rng),
        )

    df = union_plan_data(tmp_path, scenarios, rk, sub)
    assert df is not None
    assert df.columns.names == ["scenario", "node"]

    cfg = PlotConfig(
        plot_name="loss_load_bars",
        map_dimensions_for_plots=["d_se", "b_ge"],
        legend="shared",
    )
    cfg = normalize_config_for_plan_union(cfg)
    plan = compute_live_plan(df, cfg, "loss_load_bars")
    assert plan is not None
    assert plan.chart_type == "bar"

    fig = build_figure_from_plan(plan, file_index=0)
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_union_plan_data_dedupes_duplicate_row_index(tmp_path: pathlib.Path):
    """Plan parquets with duplicate row entries shouldn't crash the union.

    ``pd.concat(..., axis=1, keys=...)`` reindexes each piece against
    the union of indices and raises ``InvalidIndexError`` when a piece
    has a non-unique index.  ``union_plan_data`` defensively dedupes
    via groupby-first and logs a warning.  Verify the function returns
    a sane combined frame instead of raising.
    """
    rng = np.random.default_rng(seed=42)
    rk, sub = "rk", "sc"

    # Scenario A: clean, unique index.
    df_a = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=pd.Index([0, 1, 2], name="row"))
    _write_per_scenario_plan(tmp_path, "A", rk, sub, df_a)

    # Scenario B: duplicate index entries (row 0 appears twice).
    df_b = pd.DataFrame(
        {"x": [10.0, 20.0, 30.0]},
        index=pd.Index([0, 0, 1], name="row"),
    )
    _write_per_scenario_plan(tmp_path, "B", rk, sub, df_b)

    combined = union_plan_data(tmp_path, ["A", "B"], rk, sub)
    assert combined is not None
    # Combined index must be unique (groupby-first deduped scenario B).
    assert combined.index.is_unique, combined.index.tolist()
    # Both scenarios appear as the top column-MultiIndex level.
    assert set(combined.columns.get_level_values("scenario")) == {"A", "B"}
    # B's first occurrence at index 0 was kept (value 10.0), not summed
    # to 30.0 — confirms .first() semantics.
    val_b_at_0 = combined.loc[0, ("B", "x")]
    assert val_b_at_0 == 10.0


def test_union_path_e2e_normalises_sdt_config(tmp_path: pathlib.Path):
    """``sdt_se / mtt_lu`` (Node prices) — normaliser strips the solve dim.

    The per-scenario plan parquet has no ``solve`` row level (already
    summed by the single-mode compute step); the comparison config
    must therefore be normalised before it can re-run dim rules.
    Without normalisation, ``_apply_dimension_rules`` would raise on
    the rules-vs-levels length mismatch.
    """
    rng = np.random.default_rng(seed=2)
    scenarios = ["scenA", "scenB"]
    rk, sub = "node_prices_dt_e", "default"
    for s in scenarios:
        _write_per_scenario_plan(
            tmp_path, s, rk, sub, _make_time_plan_df(rng),
        )

    df = union_plan_data(tmp_path, scenarios, rk, sub)
    assert df is not None

    # The shipped comparison config carries the row-part ``s`` (solve).
    raw_cfg = PlotConfig(
        plot_name="node_prices",
        map_dimensions_for_plots=["sdt_se", "mtt_lu"],
        legend="shared",
    )

    # Without normalisation, dim rules would refuse the unioned frame.
    with pytest.raises(ValueError):
        compute_live_plan(df, raw_cfg, "node_prices")

    cfg = normalize_config_for_plan_union(raw_cfg)
    assert cfg.map_dimensions_for_plots == ["dt_se", "tt_lu"]
    plan = compute_live_plan(df, cfg, "node_prices")
    assert plan is not None
    fig = build_figure_from_plan(plan, file_index=0)
    assert fig is not None
    plt.close(fig)
