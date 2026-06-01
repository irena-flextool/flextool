"""full_dt slicing helper — "solve as picker" infrastructure.

This module is installment 1 of the ``(d,t)``-centric / "solve as picker"
preprocessing refactor (see ``specs/preprocessing_csv_to_db_migration.md``
§9 / §10.1).  The single public function :func:`slice_full_dt` realises
the picker step of the architecture: a parameter family is computed ONCE
over the COMPLETE ``(d,t)`` timeline (``full_dt``), cached per
``complete_solve``, and each rolling sub-solve obtains its values by
*slicing* that full-domain frame down to the roll's ``(period, time)``
window — never by re-deriving per roll.

The correctness invariant (§9.5) that makes the slice value-preserving:
for a pure ``(d,t)`` parameter ``P``,

    broadcast(P, full_dt).filter((d,t) ∈ dt_solve) ≡ broadcast(P, dt_solve)

because a value at ``(c, "price", d, t)`` is a coordinate lookup with ZERO
window aggregation — restricting the ``(d,t)`` grid before vs. after the
derivation yields identical rows.  This helper is therefore only valid for
families *proven* coordinate-stable; aggregating families must compute
their aggregate over the complete timeline and only slice the coordinate
(§9.5 amendment 1).

Laziness is load-bearing (§9.8 Attack 2): the full-domain frame is scanned
lazily and the ``(d,t)`` predicate is pushed down via a semi-join, so a
roll materialises only its slice rather than the whole-timeline frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import polars as pl

__all__ = [
    "slice_full_dt",
    "FullDtFamily",
    "FULL_DT_FAMILIES",
    "register_family",
    "emit_via_full_dt",
    "build_branch_expanded_complete_frames",
    "BRANCHED_STEPS_KEY",
    "BRANCHED_PERIOD_BRANCH_KEY",
    "BRANCHED_SOLVE_BRANCH_KEY",
    "BRANCHED_FIRST_TIMESTEPS_KEY",
]


# ---------------------------------------------------------------------------
# Branch-expanded COMPLETE timeline (R4) — separate artifact for the
# branch-dependent ``PdtLookup`` families (pdtProcess / pdtNode / …).
# ---------------------------------------------------------------------------
#
# Branch-dependent families resolve DISTINCT per-branch values at the
# stochastic branch-periods (``period1_lower`` / ``period1_mid`` / …).  The
# per-roll ``steps_in_use.csv`` carries those branch-periods; the COMPLETE
# ``steps_complete_solve.csv`` carries the BASE periods only.  Deriving a
# branch-dependent family over ``steps_complete_solve`` would therefore drop
# every branch-period row.
#
# The fix is a SEPARATE branch-expanded complete timeline: the base complete
# timeline with the stochastic branch-periods folded in, mirroring exactly
# what :meth:`StochasticSolver.create_stochastic_periods` does per-roll.  We
# REUSE that method (rather than re-implementing branch folding) by running
# it over the complete timeline as a synthetic single solve, then derive the
# branch maps (``period__branch`` / ``solve_branch__time_branch`` /
# ``first_timesteps``) the ``PdtLookup`` cascade reads, so branch values
# resolve over the complete timeline.
#
# These frames are written into the per-roll Provider under DISTINCT keys
# (the ``BRANCHED_*`` constants below) so they NEVER disturb the existing
# ``steps_complete_solve`` consumers (inflow / peak / lp scaling) which keep
# reading the un-branched ``solve_data/steps_complete_solve.csv``.

# Provider keys for the branch-expanded complete artifacts.  They live under
# a distinct ``_full_dt_branched/`` parent so they NEVER collide with the
# manifest-tracked ``solve_data/*.csv`` keys — the Provider matches keys
# EXACTLY (no bare↔qualified fallback), so a sibling parent fully isolates
# them.  Each key is the canonical ``_provider_key`` form (``parent/stem``,
# no ``.csv``) of the path the branch-dependent ``derive_*_full_dt`` reads
# (``solve_data_dir / "_full_dt_branched" / "<stem>.csv"``).
_BRANCHED_PARENT = "_full_dt_branched"
BRANCHED_STEPS_KEY = f"{_BRANCHED_PARENT}/steps_complete_solve_branched"
BRANCHED_PERIOD_BRANCH_KEY = f"{_BRANCHED_PARENT}/period__branch"
BRANCHED_SOLVE_BRANCH_KEY = f"{_BRANCHED_PARENT}/solve_branch__time_branch"
BRANCHED_FIRST_TIMESTEPS_KEY = f"{_BRANCHED_PARENT}/first_timesteps"


def branched_path(solve_data_dir: "Path | str", stem: str) -> Path:
    """Path whose :func:`_provider_key` is the matching ``BRANCHED_*`` key.

    The branch-dependent ``derive_*_full_dt`` functions read the branched
    artifacts through the same ``provider.get(_provider_key(path))`` contract
    every reader uses; this builds the path so its key resolves to the
    ``BRANCHED_*`` slot the runner ``put`` it under.
    """
    return Path(solve_data_dir) / _BRANCHED_PARENT / f"{stem}.csv"


def build_branch_expanded_complete_frames(
    state: "object",
    cs: str,
    complete_active_time_list: "dict",
) -> "dict[str, pl.DataFrame]":
    """Build the branch-expanded COMPLETE timeline + branch maps for ``cs``.

    Runs :meth:`StochasticSolver.create_stochastic_periods` over the complete
    base timeline (``complete_active_time_list``) as a synthetic single-solve
    expansion — REUSING the exact per-roll branch-folding logic rather than
    re-implementing it — and returns the four frames the branch-dependent
    ``derive_*_full_dt`` functions read:

    * ``steps`` — the branch-expanded complete ``(period, step, step_duration)``
      grid (branch-periods folded in; base-only when the solve is
      non-stochastic).
    * ``period_branch`` — ``(period, branch)`` over the complete timeline.
    * ``solve_branch`` — ``(period, branch)`` realized-branch map.
    * ``first_timesteps`` — ``(period, step)`` first step per (branch-)period.

    The returned dict is keyed ``"steps" / "period_branch" / "solve_branch" /
    "first_timesteps"``.  ``create_stochastic_periods`` mutates
    ``state.timeline.stochastic_timesteps[cs]``; we snapshot and restore that
    slot so the synthetic expansion leaves no residue on the runner state.
    """
    from flextool.engine_polars._stochastic import StochasticSolver
    from flextool.engine_polars import _emit_solve_writers as _sw

    # Snapshot the stochastic_timesteps slot that create_stochastic_periods
    # appends to under the synthetic ``cs`` key, so the synthetic expansion
    # leaves the runner state untouched.
    _st = state.timeline.stochastic_timesteps
    _had_slot = cs in _st
    _saved = list(_st[cs]) if _had_slot else None

    solver = StochasticSolver(state)
    (
        period__branch_lists,
        solve_branch__time_branch_lists,
        active_time_lists_out,
        _jump_lists,
        _fix_storage,
        _realized,
        _branch_start,
    ) = solver.create_stochastic_periods(
        state.solve.stochastic_branches,
        [cs],
        {cs: cs},
        {cs: complete_active_time_list},
        {cs: {}},
        {cs: {}},
    )

    # Restore the stochastic_timesteps slot to its pre-call value.
    if _had_slot:
        _st[cs] = _saved
    else:
        _st.pop(cs, None)

    expanded_atl = active_time_lists_out[cs]
    steps = _sw.derive_active_timelines(expanded_atl, complete=False)
    period_branch = _sw.derive_branch__period_relationship(
        period__branch_lists[cs]
    )
    solve_branch = _sw.derive_solve_branch__time_branch(
        solve_branch__time_branch_lists[cs]
    )
    first_timesteps = _sw.derive_period_boundary_step(expanded_atl, last=False)
    return {
        "steps": steps,
        "period_branch": period_branch,
        "solve_branch": solve_branch,
        "first_timesteps": first_timesteps,
    }


def slice_full_dt(
    full_frame: "pl.DataFrame | pl.LazyFrame",
    roll_dt: "pl.DataFrame | pl.LazyFrame",
    *,
    dt_cols: tuple[str, str] = ("period", "time"),
    roll_dt_cols: tuple[str, str] = ("period", "step"),
) -> pl.DataFrame:
    """Slice a full-domain ``(d,t)`` frame down to a roll's window.

    Parameters
    ----------
    full_frame:
        The parameter frame computed once over the COMPLETE timeline.
        May be eager or lazy; ``.lazy()`` is called defensively.
    roll_dt:
        The roll's ``(period, step)`` window — typically the frame read
        from ``solve_data/steps_in_use.csv``.  Only ``roll_dt_cols`` are
        used; duplicates are de-duplicated.  May be eager or lazy.
    dt_cols:
        The ``(period, time)`` column names on *full_frame*.
    roll_dt_cols:
        The ``(period, step)`` column names on *roll_dt* (mapped
        positionally onto *dt_cols* for the join).

    Returns
    -------
    pl.DataFrame
        * **strict_static** — if *full_frame* carries neither ``dt_cols``
          column, it is returned unchanged (identity): a frame with no
          ``(d,t)`` axis is solve-invariant and needs no slice.
        * **slice_static** — otherwise, the rows of *full_frame* whose
          ``(period, time)`` coordinate appears in the roll window,
          preserving *full_frame*'s row order (a semi-join keeps the
          left side's order).  An empty *roll_dt* yields an empty frame
          with *full_frame*'s schema.

    Notes
    -----
    The slice is a semi-join, not an inner join: it filters *full_frame*
    by coordinate membership without duplicating rows or reordering, and
    pushes the ``(d,t)`` predicate down through the lazy plan so only the
    matching slice is materialised.
    """
    full_lazy = full_frame.lazy()
    schema_names = full_lazy.collect_schema().names()

    period_col, time_col = dt_cols
    # strict_static: no (d,t) axis at all → solve-invariant, identity.
    if period_col not in schema_names and time_col not in schema_names:
        return full_lazy.collect()

    # slice_static: semi-join on the (d,t) coordinate.  Select + unique
    # the roll window down to just its join keys so the semi-join filters
    # purely by coordinate membership and never widens the result.
    roll_keys = (
        roll_dt.lazy()
        .select(list(roll_dt_cols))
        .unique()
    )
    return (
        full_lazy.join(
            roll_keys,
            left_on=list(dt_cols),
            right_on=list(roll_dt_cols),
            how="semi",
        )
        .collect()
    )


# ---------------------------------------------------------------------------
# Family registry — "compute-once-over-full_dt + slice" generalised.
# ---------------------------------------------------------------------------
#
# Installment 1 wired ONE family (``pdtCommodity``) by hand: a bespoke
# per-``cs`` compute in ``_native_run_model.py``, a state attribute, and a
# bespoke ``full_dt_frame=`` keyword threaded into ``emit_pdtCommodity``.
# This registry generalises that wiring so converting a NEW family is purely
# additive (see the recipe below) and never touches the per-``cs`` loop in
# ``_native_run_model.py`` again.
#
# A family registers three things:
#   * ``name``        — the family key (e.g. ``"pdtCommodity"``); also the
#                       cache slot key on ``cached_full_dt_frames[cs]``.
#   * ``derive_full_dt`` — computes the family ONCE over the COMPLETE
#                       timeline (``steps_complete_solve``).  Signature:
#                       ``(input_dir, solve_data_dir, *, provider) -> frame``.
#   * ``legacy_derive`` — the per-roll fallback (over ``steps_in_use``), used
#                       when the cache slot is absent.  Same signature.
#   * ``emit_key``    — the Provider key the per-roll emit writes the sliced
#                       (or legacy) frame to (e.g.
#                       ``"solve_data/pdtCommodity.csv"``).
#   * ``dt_cols`` / ``roll_dt_cols`` — the slice axes (defaults match
#                       :func:`slice_full_dt`).
#
# To keep this module import-safe (it is imported by ``_native_run_model``
# and the emit modules), the ``derive_*`` callables are stored as *lazy
# resolvers* — zero-arg callables returning the real function — so the heavy
# ``_emit_*`` modules are imported only when a family is actually computed,
# never at registry-build time.

_LazyFn = Callable[[], Callable[..., "pl.DataFrame"]]


@dataclass(frozen=True)
class FullDtFamily:
    """A parameter family that is computed once over full_dt and sliced.

    See the module-level registry comment for the role of each field and
    the conversion recipe.
    """

    name: str
    _derive_full_dt_resolver: _LazyFn
    _legacy_derive_resolver: _LazyFn
    emit_key: str
    dt_cols: tuple[str, str] = ("period", "time")
    roll_dt_cols: tuple[str, str] = ("period", "step")

    def derive_full_dt(
        self, input_dir: "Path | str", solve_data_dir: "Path | str",
        *, provider: "object | None" = None,
    ) -> "pl.DataFrame":
        """Compute the family over the COMPLETE timeline (one per ``cs``)."""
        return self._derive_full_dt_resolver()(
            input_dir, solve_data_dir, provider=provider,
        )

    def legacy_derive(
        self, input_dir: "Path | str", solve_data_dir: "Path | str",
        *, provider: "object | None" = None,
    ) -> "pl.DataFrame":
        """Per-roll fallback derivation (over the roll's ``steps_in_use``)."""
        return self._legacy_derive_resolver()(
            input_dir, solve_data_dir, provider=provider,
        )


FULL_DT_FAMILIES: list[FullDtFamily] = []


def register_family(
    name: str,
    *,
    derive_full_dt: _LazyFn,
    legacy_derive: _LazyFn,
    emit_key: str,
    dt_cols: tuple[str, str] = ("period", "time"),
    roll_dt_cols: tuple[str, str] = ("period", "step"),
) -> FullDtFamily:
    """Register a family into :data:`FULL_DT_FAMILIES` and return its spec.

    *derive_full_dt* / *legacy_derive* are **lazy resolvers** (zero-arg
    callables returning the real ``derive_*`` function) so the heavy
    ``_emit_*`` modules import lazily, never at registry-build time.  A
    duplicate *name* replaces the prior entry (idempotent re-registration).
    """
    spec = FullDtFamily(
        name=name,
        _derive_full_dt_resolver=derive_full_dt,
        _legacy_derive_resolver=legacy_derive,
        emit_key=emit_key,
        dt_cols=dt_cols,
        roll_dt_cols=roll_dt_cols,
    )
    for i, existing in enumerate(FULL_DT_FAMILIES):
        if existing.name == name:
            FULL_DT_FAMILIES[i] = spec
            return spec
    FULL_DT_FAMILIES.append(spec)
    return spec


def emit_via_full_dt(
    family: FullDtFamily,
    input_dir: "Path | str",
    solve_data_dir: "Path | str",
    *,
    provider,
    full_dt_frame: "pl.DataFrame | None" = None,
) -> None:
    """Emit *family*'s roll frame: slice the cached full-domain frame, or
    fall back to the legacy per-roll derivation when it is absent.

    When *full_dt_frame* is supplied (the complete-timeline frame computed
    once per ``complete_solve`` and cached on the runner state) the roll's
    frame is obtained by slicing it to the roll's ``steps_in_use`` window
    via :func:`slice_full_dt`.  When it is absent the legacy per-roll
    derivation runs.  Either way the emitted key and frame content are
    byte-identical (the slice is value-preserving for coordinate-stable
    families, specs §9.5).
    """
    # Local imports keep this module free of import cycles with the emit /
    # provider-IO modules (which themselves import ``_full_dt``).
    from flextool.engine_polars._emit_provider_io import _emit, _provider_key

    if full_dt_frame is not None:
        roll_dt = provider.get(
            _provider_key(Path(solve_data_dir) / "steps_in_use.csv")
        )
        if roll_dt is None:
            period_col, step_col = family.roll_dt_cols
            roll_dt = pl.DataFrame(
                {period_col: [], step_col: []},
                schema={period_col: pl.Utf8, step_col: pl.Utf8},
            )
        frame = slice_full_dt(
            full_dt_frame, roll_dt,
            dt_cols=family.dt_cols, roll_dt_cols=family.roll_dt_cols,
        )
    else:
        frame = family.legacy_derive(
            input_dir, solve_data_dir, provider=provider,
        )
    _emit(provider, family.emit_key, frame)


# ---------------------------------------------------------------------------
# Registered families.
# ---------------------------------------------------------------------------
# pdtCommodity — the first family on the registry (was the installment-1
# proof-of-concept).  Both callables already exist in ``_emit_period_params``
# (kept by design as the registry entry's compute fns).
register_family(
    "pdtCommodity",
    derive_full_dt=lambda: __import__(
        "flextool.engine_polars._emit_period_params",
        fromlist=["derive_pdtCommodity_full_dt"],
    ).derive_pdtCommodity_full_dt,
    legacy_derive=lambda: __import__(
        "flextool.engine_polars._emit_period_params",
        fromlist=["derive_pdtCommodity"],
    ).derive_pdtCommodity,
    emit_key="solve_data/pdtCommodity.csv",
)

# pdtProcess — first BRANCH-DEPENDENT family on the registry (R4).  Its
# ``derive_*_full_dt`` reads the branch-expanded complete timeline + complete
# branch maps (``build_branch_expanded_complete_frames`` →
# ``_full_dt_branched/`` Provider keys), so the per-roll slice carries the
# branch-period rows with correct DISTINCT per-branch values.
register_family(
    "pdtProcess",
    derive_full_dt=lambda: __import__(
        "flextool.engine_polars._emit_pdt_params",
        fromlist=["derive_pdtProcess_full_dt"],
    ).derive_pdtProcess_full_dt,
    legacy_derive=lambda: __import__(
        "flextool.engine_polars._emit_pdt_params",
        fromlist=["derive_pdtProcess"],
    ).derive_pdtProcess,
    emit_key="solve_data/pdtProcess.csv",
)

# pdtProcess_source — BRANCH-DEPENDENT per-side family (R4); same
# ``PdtLookupPerSide`` cascade as pdtProcess.  Its domain
# (``process_source_sourceSinkTimeParam_in_use``) is solve-invariant and
# pre-seeded by ``input_derivation`` (``emit_param_in_use_sets``), so it is
# present at full_dt compute time.  Its ``derive_*_full_dt`` reads the
# branch-expanded complete timeline + complete branch maps, so the per-roll
# slice carries the branch-period rows with correct DISTINCT per-branch
# values.
register_family(
    "pdtProcess_source",
    derive_full_dt=lambda: __import__(
        "flextool.engine_polars._emit_pdt_params",
        fromlist=["derive_pdtProcess_source_full_dt"],
    ).derive_pdtProcess_source_full_dt,
    legacy_derive=lambda: __import__(
        "flextool.engine_polars._emit_pdt_params",
        fromlist=["derive_pdtProcess_source"],
    ).derive_pdtProcess_source,
    emit_key="solve_data/pdtProcess_source.csv",
)
