"""Row-level views of a built polar_high Problem (coefficients and bounds by row name), for LP-shape tests."""
from __future__ import annotations

import collections


def row_terms(pb, family: str, prefix: str = "") -> dict[str, dict[str, float]]:
    """``{row name: {column name: coefficient}}`` of the rows ``family[prefix…``."""
    m = pb.canonicalise()
    out: dict[str, dict[str, float]] = collections.defaultdict(dict)
    rows = {i: nm for i, nm in enumerate(m.row_names) if nm.startswith(f"{family}[{prefix}")}
    for nm in rows.values():
        out[nm]
    for c in range(m.n_cols):
        for k in range(m.col_ptr[c], m.col_ptr[c + 1]):
            if m.row_idx[k] in rows:
                out[rows[m.row_idx[k]]][m.col_names[c]] = m.val[k]
    return dict(out)


def row_bounds(pb, family: str, prefix: str = "") -> dict[str, tuple[float, float]]:
    """``{row name: (lower, upper)}`` of the rows ``family[prefix…``."""
    m = pb.canonicalise()
    return {nm: (float(m.row_lb[i]), float(m.row_ub[i])) for i, nm in enumerate(m.row_names)
            if nm.startswith(f"{family}[{prefix}")}


def coef(terms: dict[str, float], prefix: str) -> float:
    """The one coefficient whose column name starts with ``prefix``."""
    hit = [v for c, v in terms.items() if c.startswith(prefix)]
    assert len(hit) == 1, (prefix, terms)
    return hit[0]


def step_of(row: str) -> tuple[str, str]:
    """``(d, t)`` — the last two indices of a row or column name."""
    inner = row[row.index("[") + 1:-1].split(",")
    return inner[-2], inner[-1]
