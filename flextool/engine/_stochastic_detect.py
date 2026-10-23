"""Shared stochastic-structure detectors for the recourse guards (Slice C).

This module is deliberately dependency-light (polars only) so it can be
imported at module level by BOTH ``_orchestration`` and ``_benders``
without reintroducing the ``_orchestration`` <-> ``_benders`` import
cycle (``_orchestration`` imports ``_benders`` locally to avoid it).

Two guards read stochastic structure off ``period_branch_full``:

* Guard 2 (Benders x genuine stochastics) needs the *genuinely
  stochastic* synthetic branch periods only — the ``d != b`` branch
  tokens that also have LP variables (are in ``period_in_use``).  This
  excludes metadata-only fan members (realized-named / zero-weight rows
  that carry no active time).
* Guard 3 (handoff synthetic-name assertion) needs the *broader* bare
  ``d != b`` b-token set: committing invest at a rolling metadata mirror
  (e.g. ``period1_realized``) must ALSO be forbidden, and those rolling
  metadata tokens never legitimately appear in ``realized_invest``.

Both guards extract the ``d != b`` b-token set through the single shared
idiom :func:`synthetic_branch_tokens`; Guard 2 additionally intersects
it with ``period_in_use``.
"""

from __future__ import annotations

import polars as pl


def synthetic_branch_tokens(pbf) -> set[str]:
    """Return the set of ``b`` tokens of ``period_branch_full`` ``d != b`` rows.

    ``pbf`` is a FlexData ``period_branch_full`` frame (cols ``d``, ``b``)
    or ``None``.  Returns an empty set when the frame is ``None`` or has
    no ``d != b`` row.  These are the synthetic stochastic-branch fan
    members (the anchor self-rows ``(p_j, p_j)`` and continuation
    self-rows are excluded by the ``d != b`` filter).

    Both columns are cast to ``Utf8`` BEFORE the comparison: at runtime
    ``d`` and ``b`` are polars ``Enum`` columns with DIFFERENT category
    sets (``b`` carries the synthetic-branch names ``d`` never does), and
    comparing two enums with disjoint categories raises in polars.  Null
    ``b`` rows (the deterministic ``(p_j, null)`` shape) are dropped —
    a null branch is not a synthetic token.
    """
    if pbf is None:
        return set()
    return set(
        pbf.select(
            pl.col("d").cast(pl.Utf8).alias("d"),
            pl.col("b").cast(pl.Utf8).alias("b"),
        )
        .filter(pl.col("b").is_not_null() & (pl.col("d") != pl.col("b")))
        .get_column("b")
        .to_list()
    )


def is_genuinely_stochastic(data) -> bool:
    """True iff *data* has a synthetic branch period that owns LP variables.

    A solve is genuinely stochastic iff ``period_branch_full`` has a
    ``d != b`` row whose ``b`` is ALSO in ``period_in_use_set``.  The
    intersection with ``period_in_use`` excludes metadata-only fan
    members (realized-named / zero-weight rows in ``period__branch`` that
    carry no active time), so the plain rolling-bookkeeping shape
    (``(period1, period1_realized)``, a ``d != b`` metadata row) does
    NOT trip this detector.

    Guard 2 is load-bearing only when ``period_in_use_set`` is populated
    (the chain runner writes it).  When it is ``None`` this degrades to
    ``False`` — that is today's behaviour (Benders x stochastics is
    unguarded today), not a regression; the flag-independent Guard 2 only
    tightens over the populated-PIU case.
    """
    piu = getattr(data, "period_in_use_set", None)
    if piu is None:
        return False
    synth_b = synthetic_branch_tokens(getattr(data, "period_branch_full", None))
    if not synth_b:
        return False
    piu_set = set(
        piu.select(pl.col("d").cast(pl.Utf8)).to_series().to_list()
    )
    return bool(synth_b & piu_set)
