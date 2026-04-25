"""Lazy union of per-scenario plot plan parquets for comparison view.

Comparison-mode rendering doesn't pre-combine raw data.  Instead, the
result viewer reads each viewer scenario's already-plot-shaped plan
parquet for the requested ``(result_key, sub_config)`` and concats them
with ``scenario`` as the top column-MultiIndex level — a small,
per-plot operation.

Per-scenario plan files live at::

    <project>/output_parquet/<scenario>/plot_plans/{result_key}__{sub_config}_plan.parquet
    <project>/output_parquet/<scenario>/plot_plans/{result_key}__{sub_config}_plan.json

(see :func:`flextool.plot_outputs.plan.save_plot_plan`).  The
``_plan`` suffix is part of the layout — kept for compatibility with
existing scenario-runs.

Some plot configs declare ``scenario`` as one of their input dimensions
(``s`` in the first half of ``map_dimensions_for_plots``) and therefore
need scenario as a level *inside* the dataframe — not as a top
column-MultiIndex level. ``compute_all_plot_plans`` strips the
scenario level when generating per-scenario plans, so those configs
can't be served from the union path; the viewer routes them through
the legacy combined-parquet pipeline instead.
:func:`is_scenario_pivot_config` flags such configs.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from flextool.lean_parquet import read_lean_parquet
except ImportError:  # pragma: no cover — defensive fallback
    read_lean_parquet = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


def _read_parquet(path: Path) -> pd.DataFrame:
    """Read a plan parquet, preferring lean reader when available."""
    if read_lean_parquet is not None:
        return read_lean_parquet(path)
    return pd.read_parquet(path)


def per_scenario_plan_path(
    project_path: Path, scenario: str, result_key: str, sub_config: str,
) -> Path:
    """Return the on-disk path of a single-scenario plan parquet.

    The trailing ``_plan.parquet`` suffix matches
    :func:`flextool.plot_outputs.plan.save_plot_plan`.  ``scenario`` is
    used verbatim as the directory name — underscores or other
    filesystem-safe characters round-trip; the layer that originally
    chose those names already vetted them.
    """
    return (
        Path(project_path)
        / "output_parquet"
        / scenario
        / "plot_plans"
        / f"{result_key}__{sub_config}_plan.parquet"
    )


def per_scenario_plan_json_path(
    project_path: Path, scenario: str, result_key: str, sub_config: str,
) -> Path:
    """Return the on-disk path of a single-scenario plan-JSON metadata file."""
    return (
        Path(project_path)
        / "output_parquet"
        / scenario
        / "plot_plans"
        / f"{result_key}__{sub_config}_plan.json"
    )


def load_per_scenario_plan_jsons(
    project_path: Path,
    scenarios: list[str],
    result_key: str,
    sub_config: str,
) -> list[dict]:
    """Load the per-scenario plan-JSON metadata for one ``(result_key, sub_config)``.

    Missing files contribute nothing (silent skip — matches Phase C/D).
    """
    out: list[dict] = []
    for s in scenarios:
        p = per_scenario_plan_json_path(project_path, s, result_key, sub_config)
        if not p.is_file():
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                out.append(json.load(f))
        except Exception:
            continue
    return out


def union_plan_data(
    project_path: Path,
    scenarios: list[str],
    result_key: str,
    sub_config: str,
) -> pd.DataFrame | None:
    """Concat per-scenario plan parquets with scenario at top column level.

    Returns ``None`` when no per-scenario file exists for any of
    *scenarios*.  Missing files for a subset are silently skipped — the
    union proceeds with whatever is present (matches Phase C/D
    fail-open behaviour).
    """
    pieces: list[pd.DataFrame] = []
    found_scenarios: list[str] = []
    for s in scenarios:
        path = per_scenario_plan_path(project_path, s, result_key, sub_config)
        if not path.is_file():
            continue
        try:
            df = _read_parquet(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "plan_union: failed to read %s: %s", path, exc,
            )
            continue
        pieces.append(df)
        found_scenarios.append(s)
    if not pieces:
        return None
    combined = pd.concat(pieces, axis=1, keys=found_scenarios, names=["scenario"])
    return combined


# ---------------------------------------------------------------------------
#  Scenario-pivot detection
# ---------------------------------------------------------------------------

# Roles that, when populated with "scenario", indicate the plot pivots on
# scenario as a *dataframe* dimension (not a top column level appended at
# render time).  Per-scenario plan parquets have the scenario level
# stripped at compute time; configs that pivot on scenario therefore
# can't be served from the union path.
_SCENARIO_ROLE_KEYS = (
    "stack",
    "stack_levels",
    "stack_level_names",
    "expand_axis_levels",
    "expand_axis_level_names",
    "grouped_bar_levels",
    "grouped_bar_level_names",
    "subplot",
    "subplot_levels",
    "sub_levels",
    "item_level_names",
)


def _has_scenario_role(value: Any) -> bool:
    """Return True if *value* is a list/tuple/string containing 'scenario'."""
    if isinstance(value, str):
        return value == "scenario"
    if isinstance(value, (list, tuple)):
        return any(
            (isinstance(v, str) and v == "scenario") or _has_scenario_role(v)
            for v in value
        )
    return False


def is_scenario_pivot_config(config: Any) -> bool:
    """Return True if a plot config pivots on the ``scenario`` dimension.

    A "scenario pivot" plot needs ``scenario`` as one of its
    *input-dataframe* levels (not as a top column-MultiIndex level
    appended at render time).  Per-scenario plan parquets strip the
    scenario level, so the union path can't satisfy these configs —
    callers should fall through to the legacy combined-parquet
    pipeline.

    Two signals are checked:

    1. The PlotConfig-style raw-YAML signal: the first element of
       ``map_dimensions_for_plots`` is an ``index_types`` string
       (e.g. ``"dt_seeg"``) that includes ``s`` as a dimension.  This is
       what real-world FlexTool comparison configs use.
    2. A more general dict-shape signal where role keys
       (``stack``, ``subplot``, ``expand_axis_levels``,
       ``grouped_bar_levels``, ...) are populated with the literal
       string ``"scenario"`` — useful for tests and callers that pass
       resolved level-role data.

    Returns ``False`` for ``None`` or non-dict / non-PlotConfig
    arguments.
    """
    if config is None:
        return False
    # PlotConfig dataclass instance — pull its dict-friendly attrs.
    cfg_dict: dict[str, Any]
    if hasattr(config, "__dataclass_fields__"):
        cfg_dict = {
            name: getattr(config, name)
            for name in config.__dataclass_fields__
        }
    elif isinstance(config, dict):
        cfg_dict = config
    else:
        return False

    # Signal 1: map_dimensions_for_plots string contains 's' (scenario)
    map_dims = cfg_dict.get("map_dimensions_for_plots")
    if isinstance(map_dims, (list, tuple)) and len(map_dims) >= 1:
        index_types = map_dims[0]
        if isinstance(index_types, str) and "s" in index_types:
            return True

    # Signal 2: any role key carries the literal string "scenario".
    for key in _SCENARIO_ROLE_KEYS:
        if key in cfg_dict and _has_scenario_role(cfg_dict[key]):
            return True
    return False
