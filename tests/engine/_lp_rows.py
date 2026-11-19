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


def expand_divested_built(pb, terms: dict[str, float]) -> dict[str, float]:
    """``terms`` with each ``v_divested_built_{p,n}[e,d]`` (the model-built capacity retired so far that counts in
    d, one aggregate column) replaced by what its defining row ``divestedBuilt_{p,n}[e,d]`` sets it to, the
    retirement columns ``v_divest_built_{p,n}[e,c,d′]`` it sums: two rows that differ only in using the aggregate or
    its parts are the same constraint."""
    out: dict[str, float] = {}
    for col, value in terms.items():
        if not col.startswith("v_divested_built_"):
            out[col] = out.get(col, 0.0) + value
            continue
        axis = col[len("v_divested_built_")]
        key = col[col.index("["):]
        (row,) = row_terms(pb, f"divestedBuilt_{axis}", key[1:-1]).values()
        own = row[col]
        for part, coefficient in row.items():
            if part != col:
                out[part] = out.get(part, 0.0) - value * coefficient / own
    return out
