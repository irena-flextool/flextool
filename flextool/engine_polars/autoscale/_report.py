"""YAML audit report for the autoscaler.

Layer 1 writes the four ranges + trigger flag.  Layer 2 / Layer 3 will
extend the same file with their decisions (column scalers per quantity,
``user_bound_scale`` value applied, etc.) — the structure is kept flat
and self-documenting so the operator can read it without consulting
schema docs.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from ._layer2 import Layer2Plan
from ._layer3 import Layer3Plan
from ._ranges import RangeReport


def _coerce_range(span: tuple[float, float]) -> dict[str, Any]:
    """Render a ``(lo, hi)`` tuple as a small YAML dict.

    NaN entries (the "no finite non-zero entries" sentinel) become the
    string ``"empty"`` so the YAML is still valid (YAML disallows bare
    NaN in most loaders) and the meaning is unambiguous in a manual read.
    """
    lo, hi = span
    if math.isnan(lo) or math.isnan(hi):
        return {"status": "empty"}
    return {
        "min": float(lo),
        "max": float(hi),
        "ratio": float(hi) / float(lo) if lo != 0 else None,
    }


def _coerce_cross_ratio(value: float) -> Any:
    """Render the cross-group ratio, mapping NaN → ``None``."""
    if math.isnan(value):
        return None
    return float(value)


def _render_layer1(report: RangeReport) -> dict[str, Any]:
    """Build the ``layer1`` section of the autoscaler audit YAML.

    Kept as a small pure function so Layer 2 / Layer 3 sections can be
    composed alongside it without :func:`write_report` growing branches.
    """
    return {
        "ranges": {
            "matrix": _coerce_range(report.matrix),
            "cost": _coerce_range(report.cost),
            "bound": _coerce_range(report.bound),
            "rhs": _coerce_range(report.rhs),
        },
        "cross_group_max_ratio": _coerce_cross_ratio(report.cross_group_max_ratio),
        "trigger": bool(report.trigger),
    }


def _dump_simple_yaml(obj: Any, indent: int = 0) -> str:
    """Tiny dependency-free YAML emitter.

    Avoids a hard PyYAML dependency for this stub — the autoscaler only
    ever writes scalar / mapping / list trees with no anchors, references
    or tags, so the surface stays trivial.  Keys are emitted in insertion
    order (Python 3.7+ dict semantics) so the operator-visible layout is
    stable.  Floats are rendered with ``repr`` so round-tripping into a
    real YAML loader recovers the bit-pattern.

    If a future phase needs richer YAML (e.g. multi-line strings),
    swap to :mod:`yaml`; the API of :func:`write_report` does not change.
    """
    pad = " " * indent
    if isinstance(obj, Mapping):
        if not obj:
            return "{}"
        lines: list[str] = []
        for k, v in obj.items():
            if isinstance(v, (Mapping, list)):
                rendered = _dump_simple_yaml(v, indent + 2)
                if isinstance(v, Mapping) and v:
                    lines.append(f"{pad}{k}:")
                    lines.append(rendered)
                elif isinstance(v, list) and v:
                    lines.append(f"{pad}{k}:")
                    lines.append(rendered)
                else:
                    lines.append(f"{pad}{k}: {rendered}")
            else:
                lines.append(f"{pad}{k}: {_scalar(v)}")
        return "\n".join(lines)
    if isinstance(obj, list):
        if not obj:
            return "[]"
        return "\n".join(
            f"{pad}- {_dump_simple_yaml(item, indent + 2).lstrip() if isinstance(item, (Mapping, list)) else _scalar(item)}"
            for item in obj
        )
    return _scalar(obj)


def _scalar(v: Any) -> str:
    """Render a scalar leaf for the YAML emitter."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if math.isnan(v):
            return ".nan"
        if math.isinf(v):
            return ".inf" if v > 0 else "-.inf"
        return repr(v)
    if isinstance(v, int):
        return str(v)
    s = str(v)
    if any(c in s for c in (":", "#", "\n", "{", "}", "[", "]", "&", "*")) or s.strip() != s:
        # Escape via single-quoted scalar — YAML's safest form.
        return "'" + s.replace("'", "''") + "'"
    return s


def render_layer2(plan: Layer2Plan) -> dict[str, Any]:
    """Build the ``layer2`` section of the autoscaler audit YAML.

    Captures the per-type exponents, the count of skipped rows /
    integer columns, and the per-type before/after magnitude ranges.
    Also surfaces the post-scale RHS range of the
    ``ladder_tier_cap_annual_roll`` family for the H2_trade sanity
    check — the family the regression handoff identifies as the
    Layer-2 trigger.
    """
    exponents = {t.value: int(e) for t, e in plan.type_exponents.items()}
    type_ranges_before = {
        t.value: _coerce_range(r) for t, r in plan.type_buckets_before.items()
    }
    type_ranges_after = {
        t.value: _coerce_range(r) for t, r in plan.type_buckets_after.items()
    }
    return {
        "type_exponents": exponents,
        "skipped_rows_count": len(plan.skipped_rows),
        "skipped_rows": list(plan.skipped_rows),
        "skipped_integer_cols_count": len(plan.skipped_integer_cols),
        "n_cols": int(plan.col_factors.shape[0]),
        "n_rows": int(plan.row_factors.shape[0]),
        "type_ranges_before": type_ranges_before,
        "type_ranges_after": type_ranges_after,
    }


def render_layer3(plan: Layer3Plan) -> dict[str, Any]:
    """Build the ``layer3`` section of the autoscaler audit YAML.

    Surfaces the three HiGHS options Layer 3 set and the reasoning
    string so the operator can correlate the YAML entry with the
    one-line log emitted at apply time.
    """
    return {
        "user_objective_scale": int(plan.user_objective_scale),
        "user_bound_scale": int(plan.user_bound_scale),
        "simplex_scale_strategy": int(plan.simplex_scale_strategy),
        "reasoning": str(plan.reasoning),
    }


def write_report(result: Mapping[str, Any], path: Path | str) -> Path:
    """Serialise the autoscaler result tree to ``path`` as YAML.

    ``result`` is a mapping whose top-level keys are layer names
    (``"layer1"``, later ``"layer2"`` / ``"layer3"``) and whose values
    are the per-layer payloads.  Layer 1 hands a :class:`RangeReport`;
    we render it via :func:`_render_layer1`.  Unknown payload types pass
    through to the generic dumper — keeps the door open for later
    phases without coupling the layers via shared schema.

    Returns the resolved :class:`Path` that was written, for caller
    bookkeeping (the wire-in logs it).
    """
    rendered: dict[str, Any] = {}
    for key, payload in result.items():
        if isinstance(payload, RangeReport):
            rendered[key] = _render_layer1(payload)
        else:
            rendered[key] = payload

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = _dump_simple_yaml(rendered)
    path.write_text(text + "\n", encoding="utf-8")
    return path


__all__ = ["render_layer2", "render_layer3", "write_report"]
