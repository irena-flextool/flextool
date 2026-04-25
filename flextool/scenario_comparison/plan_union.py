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

Every comparison plot config can be served from the union path.  The
unioned DataFrame has ``scenario`` as the **top** column-MultiIndex
level, with whatever per-scenario columns the dimension-rule pass
produced nested underneath — structurally what
``_apply_dimension_rules`` would compute on a combined raw frame for
any role that names ``scenario`` (line / subplot / stack / grouped-bar
/ expand-axis).  The column index simply needs ``scenario`` at the
position the comparison config expects (always level 0 in the
``index_types`` column part — verified across every shipped config in
``templates/default_comparison_plots.yaml``).

Two subtleties — both handled by :func:`normalize_config_for_plan_union`:

1. The dim-rule character ``s`` is overloaded in
   ``map_dimensions_for_plots[0]``: in the **column** part it means
   ``scenario``; in the **row** part it means ``solve`` (the FlexTool
   solve dimension).  The per-scenario plan compute step already
   collapses ``solve`` (rule ``m`` / ``y`` / ``z`` in the row part), so
   the unioned plan parquet has no ``solve`` row level.  The
   comparison config's ``s`` row entry must therefore be stripped out
   before we re-run the dim rules on the unioned frame; otherwise the
   length check in ``_apply_dimension_rules`` fails (rules count
   exceeds level count by one).
2. Nothing else needs reordering.  ``pd.concat(axis=1,
   keys=found_scenarios, names=['scenario'])`` already places
   ``scenario`` as level 0 of the column MultiIndex, matching every
   shipped config's expected ``s`` position in its column-part
   ``index_types``.
"""
from __future__ import annotations

import dataclasses
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
#  Config normalisation for the plan-union path
# ---------------------------------------------------------------------------

# Row-part rules that mean "collapse this row level entirely".  When the
# row-part of ``index_types`` carries ``s`` (the FlexTool ``solve``
# dimension, *not* ``scenario``) and the matching rule is one of these,
# the per-scenario plan compute step already collapsed ``solve`` away.
# The unioned plan parquet therefore has no ``solve`` row level and the
# comparison config's row-part ``s`` + collapsing rule must be stripped
# before re-running ``_apply_dimension_rules`` on the union — otherwise
# the rule-length vs level-count check fails by one.
_ROW_COLLAPSE_RULES = frozenset({"m", "y", "z"})


def _strip_row_solve_dim(map_dims: Any) -> tuple[str, str] | None:
    """If row-part has a leading collapsing ``s``, return adjusted (idx, rules).

    Returns ``None`` when no adjustment is needed.  The function is
    deliberately conservative: it only strips when the **first** row
    dimension is ``s`` AND the matching rule collapses (sum / weighted
    sum / weighted average).  Any other shape is left untouched so this
    helper is a no-op for the 110/114 shipped configs that have no
    row-part ``s`` at all.
    """
    if not isinstance(map_dims, (list, tuple)) or len(map_dims) < 2:
        return None
    idx, rules = map_dims[0], map_dims[1]
    if not (isinstance(idx, str) and isinstance(rules, str)):
        return None
    if "_" not in idx:
        return None
    row_idx, col_idx = idx.split("_", 1)
    # Rules string carries an underscore separating row/col rules in the
    # YAML; ``_apply_dimension_rules`` strips it.  We mirror that here so
    # we can reason positionally about row-rule chars.
    rules_no_us = rules.replace("_", "")
    if not row_idx or not row_idx.startswith("s"):
        return None
    if len(rules_no_us) < len(row_idx) + len(col_idx):
        return None
    row_rule_for_s = rules_no_us[0]
    if row_rule_for_s not in _ROW_COLLAPSE_RULES:
        return None
    new_row_idx = row_idx[1:]
    # Drop the first row-rule char and rebuild the rules string with the
    # original underscore position (between row and col rules).
    new_row_rules = rules_no_us[1: len(row_idx)]
    new_col_rules = rules_no_us[len(row_idx):]
    new_idx = f"{new_row_idx}_{col_idx}"
    new_rules = f"{new_row_rules}_{new_col_rules}"
    return new_idx, new_rules


def normalize_config_for_plan_union(config: Any) -> Any:
    """Return a config adjusted for re-applying dim rules to a unioned plan.

    The unioned plan parquet shape is ``(per-scenario plan rows) ×
    (scenario, per-scenario plan cols)`` — i.e. the per-scenario row
    dims are unchanged but ``scenario`` has been added as the outermost
    column level.

    For most configs (110/114 shipped comparison configs), the
    comparison ``map_dimensions_for_plots`` already matches that shape
    1:1, so this returns *config* unchanged.

    For the ``sdt_*`` configs (4 shipped: ``Node prices``, ``Reserve
    price in NodeGroups``), the comparison config still names the
    FlexTool ``solve`` row dim (``s`` in row-part of ``index_types``)
    even though the per-scenario plan compute step already collapsed
    it.  We strip the row-part ``s`` + matching collapse rule so the
    rules length matches the unioned frame's level count.

    Accepts ``PlotConfig`` dataclass instances or plain dicts.  Returns
    a new instance / dict; never mutates *config* in place.
    """
    if config is None:
        return config

    if hasattr(config, "__dataclass_fields__"):
        map_dims = getattr(config, "map_dimensions_for_plots", None)
        adjusted = _strip_row_solve_dim(map_dims)
        if adjusted is None:
            return config
        new_idx, new_rules = adjusted
        return dataclasses.replace(
            config, map_dimensions_for_plots=[new_idx, new_rules],
        )

    if isinstance(config, dict):
        map_dims = config.get("map_dimensions_for_plots")
        adjusted = _strip_row_solve_dim(map_dims)
        if adjusted is None:
            return config
        new_idx, new_rules = adjusted
        new_cfg = dict(config)
        new_cfg["map_dimensions_for_plots"] = [new_idx, new_rules]
        return new_cfg

    return config


def is_scenario_pivot_config(config: Any) -> bool:
    """Pure no-op kept for forward compatibility — always returns ``False``.

    Phase E originally routed configs whose ``map_dimensions_for_plots``
    contained ``s`` to a legacy combined-parquet fallback.  That routing
    was overconservative: ``s`` in the column part of ``index_types``
    means ``scenario`` (which the union path handles natively), and
    ``s`` in the row part means ``solve`` (which the per-scenario plan
    compute step has already collapsed — see
    :func:`normalize_config_for_plan_union`).  Every shipped comparison
    config is therefore servable from the union path.

    The function is kept (returning ``False`` for any input) so older
    callers that import it don't break; new code should use
    :func:`normalize_config_for_plan_union` instead.
    """
    return False
