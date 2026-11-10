"""Excel sheet names: at most 31 characters, distinct without regard to case, the time-resolution suffix kept.

Excel cuts a longer sheet name and compares names case-insensitively.  Cutting the end of a name loses what tells a
per-step table from its per-period one (``…__dt`` / ``…__d``), so two tables can land on one name or on a name that
says the wrong resolution.  The shipped long names have hand-made sheet names (:data:`SHEET_NAMES`);
:func:`excel_sheet_name` shortens any other long name's stem instead and keeps the suffix.
"""
from __future__ import annotations

import re

#: Excel's limit on the length of a sheet name.
MAX_SHEET_NAME = 31
#: The time-resolution suffix starts at the last ``__d`` / ``__dt`` token that ends the name or is followed by
#: ``_<word>`` (``__d``, ``__dt``, ``__d_average``, ``__dt_…``) and runs to the end.
_SUFFIX_START = re.compile(r"__dt?(?=_[^_]|$)")


#: Hand-made sheet names of the shipped output names (``default_plots.yaml`` ``rename``) that are longer than
#: :data:`MAX_SHEET_NAME`, and of their pairs, so that a sheet name reads as its table.  Shortening such a name
#: automatically cuts words (``…__upDown__n__dt``) or hides them behind a counter (``…_curtai_1__d``).  Keyed by the
#: display name; the CSV file names are not changed.  The curtailment table (a VRE output) is named like its
#: ``unit_VRE_*`` neighbours although its display name would fit.  :func:`excel_sheet_name` shortens any other long name.
SHEET_NAMES: dict[str, str] = {
    "process__reserve__upDown__node__dt": "process__reserve__node__dt",
    "process__reserve__upDown__node__d_average": "process__reserve__node__d_avg",
    "reserve_price__upDown__group__dt": "reserve_price__group__dt",
    "slack__reserve__upDown__group__d": "slack__reserve__group__d",
    "slack__reserve__upDown__group__dt": "slack__reserve__group__dt",
    "unit_curtailment__outputNode__d": "unit_VRE_curtailment__d",
    "unit_curtailment__outputNode__dt": "unit_VRE_curtailment__dt",
    "unit_outputNode_VRE_potential__d": "unit_VRE_potential__d",
    "unit_outputNode_VRE_potential__dt": "unit_VRE_potential__dt",
    "unit_outputNode_VRE_curtailment_share__d": "unit_VRE_curtailment_share__d",
    "unit_outputNode_VRE_curtailment_share__dt": "unit_VRE_curtailment_share__dt",
}


def _split(name: str) -> tuple[str, str]:
    """``(stem, suffix)`` of ``name`` (the suffix from the last ``__d…`` token, empty without one)."""
    hits = list(_SUFFIX_START.finditer(name))
    if not hits:
        return name, ""
    start = hits[-1].start()
    return name[:start], name[start:]


def excel_sheet_name(name: str, used: set[str]) -> str:
    """A sheet name for the table ``name``: its :data:`SHEET_NAMES` entry, or ``name`` itself when it fits in
    :data:`MAX_SHEET_NAME` characters, as long as no name of ``used`` is the same without regard to case; else the
    stem shortened in front of the time-resolution suffix, with ``_<n>`` (n = 1, 2, …) before the suffix when that
    is taken.  The result is added to ``used`` (lower-case)."""
    name = SHEET_NAMES.get(name, name)
    stem, suffix = _split(name)
    if len(suffix) > MAX_SHEET_NAME - 1:
        stem, suffix = name, ""
    candidate = name if len(name) <= MAX_SHEET_NAME else stem[:MAX_SHEET_NAME - len(suffix)] + suffix
    n = 0
    while candidate.lower() in used:
        n += 1
        tag = f"_{n}"
        keep = MAX_SHEET_NAME - len(suffix) - len(tag)
        # A suffix too long to leave room for a stem and the tag is dropped (the name still fits).
        candidate = (stem[:keep] + tag + suffix if keep > 0 else name[:MAX_SHEET_NAME - len(tag)] + tag)
    used.add(candidate.lower())
    return candidate
