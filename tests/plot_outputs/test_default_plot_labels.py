"""Value-axis labels of ``flextool/schemas/default_plots.yaml`` say what the numbers are.

* A plot that sums the periods with ``y`` (years represented × the annual value) shows a horizon total: its value
  label says "(horizon total".
* A per-period plot of an annualized table (``OUTPUT_TRANSFORM`` semantics ``annualized``) says "(annualized)" or
  "/a".
* No table is a plain sample sum any more (``ENERGY_PERIOD``): a ``y`` total of such a table would multiply a sample
  sum by years. The root rule: a ``y`` total is only drawn of an annualized table (or an annualized count).
* A per-subplot label list has one label per subplot (the length of the per-subplot ``axis_tick_format`` /
  ``axis_bounds`` lists), and every label has balanced parentheses: an unquoted comma inside a YAML ``[...]`` item
  splits the label in two, which these catch.

Each config is checked in single mode and in comparison mode (``comparison_overrides`` merged over the config).
Some configs put the value label on the axis the comparison view swaps to, so a check passes when either axis
label says it.
"""
from __future__ import annotations

import yaml

from flextool._resources import package_data_path
from flextool.process_outputs._output_meta import ENERGY_PERIOD, OUTPUT_TRANSFORM, Semantics


def _configs():
    with open(package_data_path("schemas/default_plots.yaml"), encoding="utf-8") as f:
        plots = yaml.safe_load(f)["plots"]
    for heading, tables in plots.items():
        for table, variants in tables.items():
            if not isinstance(variants, dict) or table in ("group", "order"):
                continue
            for variant, cfg in variants.items():
                if not isinstance(cfg, dict) or "map_dimensions_for_plots" not in cfg:
                    continue
                yield f"{heading}/{table}/{variant}", table, cfg
                overrides = cfg.get("comparison_overrides")
                if overrides:
                    yield f"{heading}/{table}/{variant} (comparison)", table, {**cfg, **overrides}


def _period_rule(cfg) -> str | None:
    """The rule of the period level (``d``), in the row or the column part."""
    index_types, rules = cfg["map_dimensions_for_plots"]
    types, rules = index_types.replace("_", ""), rules.replace("_", "")
    return rules[types.index("d")] if "d" in types else None


def _axis_labels(cfg) -> list[str]:
    """The x and y labels of a config (a per-subplot list counts element by element)."""
    out: list[str] = []
    for key in ("xlabel", "ylabel"):
        label = cfg.get(key)
        if label is not None:
            out.extend(str(x) for x in label) if isinstance(label, list) else out.append(str(label))
    return out


def test_horizon_totals_are_labelled() -> None:
    bad = [f"{where}: {labels}" for where, _t, cfg in _configs() if _period_rule(cfg) == "y"
           for labels in [_axis_labels(cfg)] if not any("(horizon total" in x for x in labels)]
    assert not bad, "\n".join(bad)


def test_annualized_tables_are_labelled_per_year() -> None:
    bad = []
    for where, table, cfg in _configs():
        transform = OUTPUT_TRANSFORM.get(table)
        if getattr(transform, "semantics", None) != Semantics.ANNUALIZED or _period_rule(cfg) in ("y", "z", "m"):
            continue
        labels = _axis_labels(cfg)
        if labels and not any("(annualized)" in x or "/a" in x for x in labels):
            bad.append(f"{where}: {labels}")
    assert not bad, "\n".join(bad)


def test_no_table_is_a_plain_sample_sum() -> None:
    users = [key for key, t in OUTPUT_TRANSFORM.items()
             if t is ENERGY_PERIOD or (isinstance(t, dict) and any(v is ENERGY_PERIOD for v in t.values()))]
    assert users == []


def _semantics(table: str) -> set:
    transform = OUTPUT_TRANSFORM.get(table)
    transforms = transform.values() if isinstance(transform, dict) else [transform]
    return {getattr(t, "semantics", None) for t in transforms if t is not None}


def test_horizon_totals_are_of_annualized_tables() -> None:
    allowed = {Semantics.ANNUALIZED, Semantics.COUNT}
    bad = [f"{where}: {sorted(map(str, sem))}" for where, table, cfg in _configs() if _period_rule(cfg) == "y"
           for sem in [_semantics(table)] if not sem or not sem <= allowed]
    assert not bad, "\n".join(bad)


def _per_subplot_lengths(cfg) -> set[int]:
    return {len(cfg[k]) for k in ("axis_tick_format", "axis_bounds") if isinstance(cfg.get(k), list)}


def test_label_lists_have_one_label_per_subplot() -> None:
    bad = []
    for where, _t, cfg in _configs():
        counts = _per_subplot_lengths(cfg)
        for key in ("xlabel", "ylabel"):
            label = cfg.get(key)
            if isinstance(label, list) and (len(counts) != 1 or len(label) not in counts):
                bad.append(f"{where}: {key} has {len(label)} labels, per-subplot lists have {sorted(counts)}")
    assert not bad, "\n".join(bad)


def test_labels_have_balanced_parentheses() -> None:
    bad = [f"{where}: {x!r}" for where, _t, cfg in _configs() for x in _axis_labels(cfg)
           if x.count("(") != x.count(")")]
    assert not bad, "\n".join(bad)
