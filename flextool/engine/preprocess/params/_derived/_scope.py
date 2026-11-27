"""Readers the derived-parameter families share: Provider access, the ``source.parameter`` /
``source.entities`` wrappers, the active solve, the recourse flag and the axis-enum handle."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
from flextool.engine.preprocess._provider_io import _provider_key
from flextool.engine.preprocess.params._branch import fan_period_labels

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource
    from flextool.engine.preprocess._solve_context import SolveContext


def _provider_has_key(provider, path: "Path | str") -> bool:
    """Provider-only existence check.  Returns ``True`` iff the live
    :class:`FlexDataProvider` carries the artefact for *path* (under
    its canonical name); ``False`` when *provider* is ``None`` or the
    key is missing.  The disk-fallback arm was removed in Step 2.5 —
    cascade modules consume frames exclusively from the Provider.
    """
    if provider is None:
        return False
    return provider.has(_provider_key(path))


def _provider_read(provider, path: "Path | str") -> "pl.DataFrame":
    """Provider-only read.  Returns the frame for *path* (under its
    canonical key) from the Provider.  Caller must guard with
    :func:`_provider_has_key` first; calling with ``provider=None``
    raises :class:`ValueError`.
    """
    if provider is None:
        raise ValueError(
            "_provider_read requires a FlexDataProvider; guard with "
            "_provider_has_key first."
        )
    return provider.get(_provider_key(path))


def _recourse_invest_active(
    workdir: "Path | None",
    *,
    ctx: "object | None" = None,
    provider: "object | None" = None,
) -> bool:
    """True iff ``solve_data/stochastic_invest_method.csv`` == ``'recourse'``.

    Slice D's single gate.  False when the CSV is absent (deterministic /
    flag-off / pre-Slice-D dumps) — the byte-parity-safe default.

    Provider-only read (canonical key), matching every other reader in
    this module (``_build_recourse_anchor_pairs`` et al.) and the post-
    Step-2 cascade invariant (``test_meta_provider_invariants``): the
    flag CSV reaches the cascade exclusively through the Provider, never
    from disk.  On the snapshot-reload path (``load_flextool`` off a
    dumped workdir) the Provider is seeded from ``solve_data/`` before
    the cascade runs (``_csv_reload.load_from_dump``), so it carries
    this key by construction;
    cascade entry points always thread an explicit Provider.  Tests that
    exercise the flag seed the Provider from the workdir the same way.
    ``workdir`` is retained in the signature only to key the canonical
    Provider path (``_provider_key`` ignores the prefix, so the key is
    stable whether or not a workdir is supplied).
    """
    path = (
        Path(workdir) / "solve_data" / "stochastic_invest_method.csv"
        if workdir is not None
        else Path("solve_data") / "stochastic_invest_method.csv"
    )
    if not _provider_has_key(provider, path):
        return False
    df = _provider_read(provider, path)
    if df is None or df.height == 0 or "method" not in df.columns:
        return False
    val = df["method"].cast(pl.Utf8, strict=False).to_list()[0]
    return str(val).strip().lower() == "recourse" if val is not None else False


# Substrate handle for the cascade-wide axis enum vocabulary.
# Bare ``None`` here; ``cast_dim`` / ``schema_dtype`` in
# ``_axis_enums`` fall back to ``_LIVE_AXIS_ENUMS_CTX`` (the live
# ContextVar) when this is ``None``, so substrate sites pick up
# activation set by ``load_flextool`` automatically.
_enums: "dict | None" = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _try_param(source: "InputSource", entity_class: str,
                parameter_name: str) -> pl.DataFrame | None:
    """Return ``source.parameter(...)`` or ``None`` if class unknown /
    parameter unknown / empty.  Mirrors ``_projection._try_param``.
    """
    try:
        df = source.parameter(entity_class, parameter_name)
    except KeyError:
        return None
    if df.height == 0:
        return None
    return df


def _min_time_methods(source: "InputSource") -> dict[str, str]:
    """``{unit: minimum_time_method}``: the explicit value, else the definition default (``source.parameter``
    broadcasts a non-null default).  A unit absent from the result reads ``none``; a ``none`` default reads the same
    as no default.  The one reading rule of the ``startup_method`` promotion (Step 5 of
    :func:`_classify_process_method`) and the min up / down gate (:func:`_process_min_time`)."""
    df = _try_param(source, "unit", "minimum_time_method")
    if df is None:
        return {}
    return {str(n): str(v) for n, v in df.select("name", "value").iter_rows() if v is not None}


def _try_param_explicit(source: "InputSource", entity_class: str,
                         parameter_name: str) -> pl.DataFrame | None:
    """Like :func:`_try_param` but suppresses default-broadcast rows.

    Mirrors flextool's ``p_unit.get(name, None)`` semantic: returns only
    entities with a parameter_value row in the active scenario.  Falls
    back to the public :meth:`InputSource.parameter` when the source
    has no ``parameter_explicit`` method (legacy custom InputSource
    implementations).
    """
    fn = getattr(source, "parameter_explicit", None) or source.parameter
    try:
        df = fn(entity_class, parameter_name)
    except KeyError:
        return None
    if df.height == 0:
        return None
    return df


def _try_entities(source: "InputSource", entity_class: str) -> pl.DataFrame | None:
    try:
        df = source.entities(entity_class)
    except KeyError:
        return None
    if df.height == 0:
        return None
    return df


def _scalar_default(source: "InputSource", entity_class: str,
                     parameter_name: str, fallback: Any) -> Any:
    """Look up the scalar Spine default; fall back to *fallback* when
    the source returns ``None`` (None-default policy).
    """
    try:
        d = source.parameter_default(entity_class, parameter_name)
    except KeyError:
        return fallback
    return fallback if d is None else d


def _read_active_solve(workdir: Path, *, provider) -> str | None:
    """Read ``solve_data/solve_current.csv`` and return the active solve
    name, or ``None`` when the Provider doesn't carry it / it's empty.
    """
    if provider is None:
        raise ValueError(
            "_read_active_solve requires a FlexDataProvider; the disk-"
            "fallback arm was removed in Step 2.5."
        )
    p = Path(workdir) / "solve_data" / "solve_current.csv"
    if not provider.has(_provider_key(p)):
        return None
    df = provider.get(_provider_key(p))
    if df is None or df.height == 0:
        return None
    col = df.columns[0]
    return df[col][0]


def _solve_in_spine(source: "InputSource",
                      active_solve: str | None) -> bool:
    """Return True iff ``active_solve`` appears as a row in Spine's
    ``solve`` entity class.

    Δ.18 — synthetic per-sub-solve names (e.g. ``invest_5weeks_p2020``
    for nested-multi-invest fixtures) don't exist in Spine; the per-solve
    override chain (``apply_derived_a`` etc.) returns None for every
    helper that takes ``active_solve`` as a key.  Detecting this upfront
    lets the loader keep the snapshot-CSV-loaded values as authoritative.

    Probes the ``solve`` entity table directly — relying on a per-solve
    parameter (e.g. ``invest_periods``) was incorrect because some solves
    have empty ``invest_periods`` (dispatch-only) yet are valid solves.
    """
    if active_solve is None:
        return False
    try:
        ents = source.entities("solve")
    except KeyError:
        return False
    if ents is None or ents.height == 0:
        return False
    name_col = "name" if "name" in ents.columns else ents.columns[0]
    return ents.filter(pl.col(name_col) == active_solve).height > 0


def _ctx_read(
    ctx: "SolveContext | None",
    workdir: Path | str | None,
    name: str,
    *,
    kind: str = "solve_data",
    provider: "object | None" = None,
) -> "pl.DataFrame | None":
    """Read the per-solve frame *name* (``<kind>/<stem>``) from *provider*.

    *provider* is required: ``None`` raises ``ValueError`` (no work-folder
    read).  Key present → the frame (an empty body → ``pl.DataFrame()``);
    key absent, or no *workdir* to form the key from → ``None`` (the
    callers' "no frame" branch).  *ctx* is unused; it stays in the
    signature for call-site stability.
    """
    if provider is None:
        raise ValueError(
            f"_ctx_read({name!r}) requires a FlexDataProvider; it never reads the work folder on disk."
        )
    if workdir is None:
        return None
    if kind == "solve_data":
        path = Path(workdir) / "solve_data" / name
    elif kind == "input":
        path = Path(workdir) / "input" / name
    else:
        path = Path(workdir) / name
    if not _provider_has_key(provider, path):
        return None
    try:
        return _provider_read(provider, path)
    except pl.exceptions.NoDataError:
        return pl.DataFrame()


def _labels_of_pairs(periods: list, branches: list) -> set[str]:
    """The period labels (:func:`._branch.fan_period_labels`) of ``period__branch`` given as two columns."""
    return fan_period_labels({(str(a), str(b)) for a, b in zip(periods, branches)})


def _entity_classes_lookup(source: "InputSource"
                              ) -> tuple[set[str], set[str], set[str]]:
    """Return ``(process_set, node_set, connection_set)`` of entity names.

    Mirrors flextool's input/process.csv = unit ∪ connection (see
    ``input_writer.py:42``); the distinct ``unit`` and ``connection``
    classes both feed into the shared ``process_set`` consumed by
    ``entity_period_calc_params._ed_value`` and the annuity sums.
    The ``connection_set`` is kept separate so callers can resolve
    connection-only attributes (e.g. ``is_DC``) when needed.
    """
    def _set(ec: str) -> set[str]:
        df = _try_entities(source, ec)
        if df is None:
            return set()
        return set(df["name"].to_list())
    unit_s = _set("unit")
    conn_s = _set("connection")
    process_s = unit_s | conn_s
    return process_s, _set("node"), conn_s
