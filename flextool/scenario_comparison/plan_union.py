"""Comparison-mode helpers built on the merged ``default_plots.yaml``.

Each leaf config in ``default_plots.yaml`` carries optional comparison-mode
add-ons:

* ``scenario_rule`` — single character that says how the ``scenario`` dim is
  folded into the comparison view (``g``=grouped/coloured bars, ``l``=lines,
  ``u``=subplots, ``f``=files, ``s``=stacked, ``e``=expand-axis).
* ``comparison_overrides`` — optional dict of plot-setting overrides applied
  on top of the single config in comparison mode.  When it includes
  ``map_dimensions_for_plots``, that explicit value wins over the value
  auto-derived from ``single + scenario_rule``.

Two helpers live here:

* :func:`derive_comparison_config` — builds a comparison-mode ``PlotConfig``
  from a single-mode one by inserting ``s`` + ``scenario_rule`` into the
  column part of ``index_types``/``rules`` and merging in
  ``comparison_overrides``.
* :func:`union_raw_data` — reads each viewer scenario's raw result parquet
  and unions them with ``scenario`` as the outermost column-MultiIndex
  level, so the derived config can be applied via the standard
  ``compute_live_plan`` pipeline.
"""
from __future__ import annotations

import dataclasses
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
    """Read a parquet file, preferring the lean reader when available."""
    if read_lean_parquet is not None:
        return read_lean_parquet(path)
    return pd.read_parquet(path)


def derive_comparison_config(single_cfg: Any) -> Any:
    """Build a comparison-mode PlotConfig from a single-mode one.

    The unioned **raw** comparison frame has shape::

        row = (single's row dims)
        col = ('scenario', *single's col dims)

    so we prepend ``s`` to ``index_types``'s column part and prepend
    ``scenario_rule`` to the rules string's column part.  The single-mode
    rules string is otherwise unchanged, which means
    ``_apply_dimension_rules`` will reshape the data exactly as it does in
    single mode, with one extra column dim representing scenario.

    ``comparison_overrides`` (also on the single config) provides per-mode
    overrides for layout knobs.  Overrides win — including
    ``map_dimensions_for_plots`` if explicitly given (used by the rare
    configs whose comparison view wants a different visual treatment than
    "single + scenario_rule prepended").

    Raises ``ValueError`` when ``scenario_rule`` is not set or
    ``map_dimensions_for_plots`` is malformed.
    """
    scenario_rule = getattr(single_cfg, "scenario_rule", None)
    if scenario_rule is None and isinstance(single_cfg, dict):
        scenario_rule = single_cfg.get("scenario_rule")
    if scenario_rule is None:
        raise ValueError(
            "derive_comparison_config: the single config has no "
            "scenario_rule — define one (e.g. 'g' for grouped bars, "
            "'l' for lines, 'u' for subplots) to enable comparison view."
        )

    if hasattr(single_cfg, "map_dimensions_for_plots"):
        md = single_cfg.map_dimensions_for_plots
    else:
        md = single_cfg.get("map_dimensions_for_plots")
    if not isinstance(md, (list, tuple)) or len(md) < 2:
        raise ValueError(
            f"derive_comparison_config: map_dimensions_for_plots must be a "
            f"2-element [index_types, rules] list, got {md!r}"
        )
    idx, rules = md[0], md[1]
    if not isinstance(idx, str) or "_" not in idx:
        raise ValueError(
            f"derive_comparison_config: index_types must contain '_' "
            f"separating row and column parts, got {idx!r}"
        )
    if not isinstance(rules, str) or "_" not in rules:
        raise ValueError(
            f"derive_comparison_config: rules must contain '_' separating "
            f"row and column parts, got {rules!r}"
        )
    row_idx, col_idx = idx.split("_", 1)
    row_rules, col_rules = rules.split("_", 1)

    # If the single rules already name 'scenario' in the col part (e.g.
    # ``[d_s, s_b]`` for a costs-by-scenario chart that's the same shape
    # in both modes), don't auto-prepend another ``s`` — the existing rule
    # for scenario stands.  ``scenario_rule`` is required to mark the
    # config as comparison-renderable, but its value is unused on this
    # branch (and may be overridden via ``comparison_overrides``).
    if "s" in col_idx:
        new_idx, new_rules = idx, rules
    else:
        new_idx = f"{row_idx}_s{col_idx}"
        new_rules = f"{row_rules}_{scenario_rule}{col_rules}"

    overrides = getattr(single_cfg, "comparison_overrides", None)
    if overrides is None and isinstance(single_cfg, dict):
        overrides = single_cfg.get("comparison_overrides")
    overrides = dict(overrides or {})

    if hasattr(single_cfg, "__dataclass_fields__"):
        valid_fields = set(single_cfg.__dataclass_fields__.keys())
        replace_kwargs = {k: v for k, v in overrides.items() if k in valid_fields}
        # Two-step replace: derive first, then apply overrides — so a config
        # whose comparison rules differ beyond just adding scenario can carry
        # ``map_dimensions_for_plots`` in ``comparison_overrides`` and have it
        # take precedence over the derived value.
        derived = dataclasses.replace(
            single_cfg, map_dimensions_for_plots=[new_idx, new_rules],
        )
        return dataclasses.replace(derived, **replace_kwargs)

    if isinstance(single_cfg, dict):
        new_cfg = dict(single_cfg)
        new_cfg["map_dimensions_for_plots"] = [new_idx, new_rules]
        new_cfg.update(overrides)  # overrides win, including map_dimensions_for_plots
        return new_cfg

    raise TypeError(
        f"derive_comparison_config: unsupported config type {type(single_cfg)!r}"
    )


def union_raw_data(
    project_path: Path,
    scenarios: list[str],
    result_key: str,
) -> pd.DataFrame | None:
    """Read each scenario's raw result parquet and union them.

    Each per-scenario raw parquet at ``output_parquet/<scenario>/<rk>.parquet``
    already carries a ``scenario`` column-MultiIndex level (added at write
    time); we drop that local level and re-concat across scenarios with the
    **viewer-supplied** scenario name as the new outermost ``scenario``
    level.  This handles the case where a scenario folder name differs from
    the embedded scenario name (e.g. sensitivity replicas where the folder
    is ``scenario_test_6h_2`` but the embedded name is ``scenario_test_6h``).

    Returns ``None`` when no raw parquet exists for any of *scenarios*.
    Missing files for a subset are silently skipped — the union proceeds
    with whatever is present.
    """
    pieces: list[pd.DataFrame] = []
    found: list[str] = []
    for s in scenarios:
        path = Path(project_path) / "output_parquet" / s / f"{result_key}.parquet"
        if not path.is_file():
            continue
        try:
            df = _read_parquet(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("union_raw_data: failed to read %s: %s", path, exc)
            continue
        # Strip the embedded scenario level so the keys=... we set below is
        # the only authoritative scenario label.  Two cases:
        #  * MultiIndex columns with a 'scenario' level → droplevel.
        #  * Single-level Index named 'scenario' (e.g. costs_discounted_p_,
        #    a category × scenario table) → squeeze to a Series so concat
        #    below adds 'scenario' as the only column level (otherwise the
        #    inner unnamed level lingers and breaks rules like ``s_b``).
        if isinstance(df.columns, pd.MultiIndex) and "scenario" in df.columns.names:
            df = df.droplevel("scenario", axis=1)
        elif (not isinstance(df.columns, pd.MultiIndex)
              and df.columns.name == "scenario"
              and len(df.columns) == 1):
            df = df.iloc[:, 0]  # Series with the row index preserved
        pieces.append(df)
        found.append(s)
    if not pieces:
        return None
    return pd.concat(pieces, axis=1, keys=found, names=["scenario"])
