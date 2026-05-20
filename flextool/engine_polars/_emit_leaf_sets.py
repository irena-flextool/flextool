"""Writer-port Phase 1 (L0-L2) — leaf-level set projections.

Native polars port of the four "trivial setof" preprocessing families
invoked from :func:`flextool.flextoolrunner.input_writer.write_input`
at lines 1882-1939.  Each helper reads one or two already-written
``input/*.csv`` (or ``solve_data/*.csv``) files and emits a small
single- or multi-column set CSV under ``solve_data/``.

Ported legacy modules (preprocessing/):

* ``period_param_sets.py``  — 4 period projections from pd_*.csv
* ``invest_method_sets.py`` — 4 method-filter projections
* ``co2_method_sets.py``    — 3 co2-method projections
* ``simple_projections.py`` — 11 trivial setof projections

Each public ``derive_*`` returns a fresh ``pl.DataFrame`` in-memory
(the primary contract).  ``write_*`` wrappers materialise that frame
to the legacy CSV path so downstream consumers (``load_flextool``,
``flextool.mod`` via ``table data IN``) still see the same file
layout during the transition.

Style: read tiny CSVs eagerly with polars, project with native polars
expressions, deduplicate via ``.unique(maintain_order=True)``.  No
abstraction beyond a per-family ``write_all`` orchestrator.  Match
:mod:`._derived_existing` / :mod:`._projection_params` style.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from ._axis_enums import alias_to_axis
from ._emit_provider_io import _emit


# ---------------------------------------------------------------------------
# Method-enum constants — mirror flextool/flextool_base.dat and the legacy
# constants in preprocessing/{invest_method_sets,co2_method_sets}.py.
# ---------------------------------------------------------------------------

_INVEST_METHOD_NOT_ALLOWED: frozenset[str] = frozenset((
    "not_allowed", "retire_period", "retire_total", "retire_no_limit",
))
_DIVEST_METHOD_NOT_ALLOWED: frozenset[str] = frozenset((
    "not_allowed", "invest_period", "invest_total", "invest_no_limit",
))

_CO2_PRICE_METHOD: frozenset[str] = frozenset((
    "price", "price_period", "price_total", "price_period_total",
))
_CO2_MAX_PERIOD_METHOD: frozenset[str] = frozenset((
    "period", "price_period", "period_total", "price_period_total",
))
_CO2_MAX_TOTAL_METHOD: frozenset[str] = frozenset((
    "total", "price_total", "period_total", "price_period_total",
))


# ---------------------------------------------------------------------------
# Internal CSV reader.  All inputs are tiny set/parameter CSVs (≤ thousands
# of rows in any realistic fixture) so eager read is fine.  Missing source
# means "empty set" — flextool's legacy code returns ``[]`` in that case.
# ---------------------------------------------------------------------------

def _read_csv(path: Path, columns: list[str],
              *, provider: "object | None" = None) -> pl.DataFrame:
    """Provider-only — returns an empty all-Utf8 frame on Provider miss.

    Step 2.5 Phase C dropped the disk-fallback arm.  The
    canonical-schema empty frame is the documented behaviour when an
    upstream writer hasn't populated *path*'s key (legacy behaviour
    for a missing on-disk CSV).
    """
    from flextool.engine_polars._emit_provider_io import (
        _provider_key,
        _provider_lookup_positional,
    )
    seeded = _provider_lookup_positional(
        provider, _provider_key(path), path, columns,
    )
    if seeded is not None:
        return seeded
    return pl.DataFrame(
        {c: [] for c in columns}, schema={c: pl.Utf8 for c in columns},
    )


def _write(df: pl.DataFrame, path: Path) -> None:
    """Write a tiny set CSV.  Empty frame still writes the header line.

    Phase E-c — disk emission is gated behind the module-level
    flag.  When disabled, the helper returns without touching disk; the
    accumulator hook (installed by ``capture_frames``) still captures
    the frame in-memory because capture happens BEFORE the wrapped
    real ``_write`` is invoked.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)


# ---------------------------------------------------------------------------
# Family 1 — period_param_sets (legacy: preprocessing/period_param_sets.py)
# ---------------------------------------------------------------------------

# (source pd_*.csv, target solve_data csv)
_PERIOD_PARAM_SOURCES: list[tuple[str, str]] = [
    ("pd_group.csv",     "period_group.csv"),
    ("pd_node.csv",      "period_node.csv"),
    ("pd_commodity.csv", "period_commodity.csv"),
    ("pd_process.csv",   "period_process.csv"),
]


def derive_period_param_set(input_dir: Path, source_csv: str,
                             *, provider: "object | None" = None,
                             ) -> pl.DataFrame:
    """Project the ``period`` column out of a ``pd_*.csv`` file.

    Legacy: setof {(e, param, d, value) in entity__param__period} (d).
    Order = first occurrence in the source CSV.
    """
    df = _read_csv(input_dir / source_csv,
                   ["entity", "param", "period", "value"],
                   provider=provider)
    return (
        df.filter(pl.col("period") != "")
          .select("period")
          .unique(maintain_order=True)
    )


def write_period_param_sets(input_dir: Path, solve_data_dir: Path,
                             *, provider: "object | None" = None,
                             ) -> None:
    for source_csv, target_name in _PERIOD_PARAM_SOURCES:
        _write(
            derive_period_param_set(input_dir, source_csv, provider=provider),
            solve_data_dir / target_name,
        )


def emit_period_param_sets(input_dir: Path, solve_data_dir: Path,
                            *, provider) -> None:
    """Provider-emitting twin of :func:`write_period_param_sets`."""
    del solve_data_dir  # legacy signature parity; keys are static
    for source_csv, target_name in _PERIOD_PARAM_SOURCES:
        _emit(provider, f"solve_data/{target_name}",
              derive_period_param_set(input_dir, source_csv, provider=provider))


# ---------------------------------------------------------------------------
# Family 2 — invest_method_sets (legacy: preprocessing/invest_method_sets.py)
# ---------------------------------------------------------------------------

def _project_entity_by_method(
    df: pl.DataFrame,
    disallowed_methods: frozenset[str],
    out_column: str,
) -> pl.DataFrame:
    """``setof {(e, m) : m not in disallowed} (e)`` — order preserved."""
    return (
        df.filter(
            (pl.col("entity") != "") & (~pl.col("method").is_in(list(disallowed_methods)))
        )
        .select(alias_to_axis("entity", out_column))
        .unique(maintain_order=True)
    )


def derive_entity_invest(input_dir: Path,
                          *, provider: "object | None" = None,
                          ) -> pl.DataFrame:
    df = _read_csv(input_dir / "entity__invest_method.csv",
                   ["entity", "method"], provider=provider)
    return _project_entity_by_method(df, _INVEST_METHOD_NOT_ALLOWED, "entity")


def derive_entity_divest(input_dir: Path,
                          *, provider: "object | None" = None,
                          ) -> pl.DataFrame:
    df = _read_csv(input_dir / "entity__invest_method.csv",
                   ["entity", "method"], provider=provider)
    return _project_entity_by_method(df, _DIVEST_METHOD_NOT_ALLOWED, "entity")


def derive_group_invest(input_dir: Path,
                         *, provider: "object | None" = None,
                         ) -> pl.DataFrame:
    df = _read_csv(input_dir / "group__invest_method.csv",
                   ["entity", "method"], provider=provider)
    return _project_entity_by_method(df, _INVEST_METHOD_NOT_ALLOWED, "group")


def derive_group_divest(input_dir: Path,
                         *, provider: "object | None" = None,
                         ) -> pl.DataFrame:
    df = _read_csv(input_dir / "group__invest_method.csv",
                   ["entity", "method"], provider=provider)
    return _project_entity_by_method(df, _DIVEST_METHOD_NOT_ALLOWED, "group")


def write_invest_method_sets(input_dir: Path, solve_data_dir: Path,
                              *, provider: "object | None" = None,
                              ) -> None:
    _write(derive_entity_invest(input_dir, provider=provider),
           solve_data_dir / "entityInvest.csv")
    _write(derive_entity_divest(input_dir, provider=provider),
           solve_data_dir / "entityDivest.csv")
    _write(derive_group_invest(input_dir, provider=provider),
           solve_data_dir / "group_invest.csv")
    _write(derive_group_divest(input_dir, provider=provider),
           solve_data_dir / "group_divest.csv")


def emit_invest_method_sets(input_dir: Path, solve_data_dir: Path,
                             *, provider) -> None:
    """Provider-emitting twin of :func:`write_invest_method_sets`."""
    del solve_data_dir
    _emit(provider, "solve_data/entityInvest.csv",
          derive_entity_invest(input_dir, provider=provider))
    _emit(provider, "solve_data/entityDivest.csv",
          derive_entity_divest(input_dir, provider=provider))
    _emit(provider, "solve_data/group_invest.csv",
          derive_group_invest(input_dir, provider=provider))
    _emit(provider, "solve_data/group_divest.csv",
          derive_group_divest(input_dir, provider=provider))


# ---------------------------------------------------------------------------
# Family 3 — co2_method_sets (legacy: preprocessing/co2_method_sets.py)
# ---------------------------------------------------------------------------

def _project_group_by_method_in(
    df: pl.DataFrame, allowed_methods: frozenset[str],
) -> pl.DataFrame:
    return (
        df.filter(
            (pl.col("group") != "") & pl.col("method").is_in(list(allowed_methods))
        )
        .select("group")
        .unique(maintain_order=True)
    )


def derive_group_co2(input_dir: Path, kind: str,
                      *, provider: "object | None" = None,
                      ) -> pl.DataFrame:
    """Project groups whose co2_method ∈ allowed set.

    ``kind`` is one of ``"price"``, ``"max_period"``, ``"max_total"``.
    """
    allowed = {
        "price":      _CO2_PRICE_METHOD,
        "max_period": _CO2_MAX_PERIOD_METHOD,
        "max_total":  _CO2_MAX_TOTAL_METHOD,
    }[kind]
    df = _read_csv(input_dir / "group__co2_method.csv",
                   ["group", "method"], provider=provider)
    return _project_group_by_method_in(df, allowed)


def write_co2_method_sets(input_dir: Path, solve_data_dir: Path,
                           *, provider: "object | None" = None,
                           ) -> None:
    for kind, target in (
        ("price",      "group_co2_price.csv"),
        ("max_period", "group_co2_max_period.csv"),
        ("max_total",  "group_co2_max_total.csv"),
    ):
        _write(derive_group_co2(input_dir, kind, provider=provider),
               solve_data_dir / target)


def emit_co2_method_sets(input_dir: Path, solve_data_dir: Path,
                          *, provider) -> None:
    """Provider-emitting twin of :func:`write_co2_method_sets`."""
    del solve_data_dir
    for kind, target in (
        ("price",      "group_co2_price.csv"),
        ("max_period", "group_co2_max_period.csv"),
        ("max_total",  "group_co2_max_total.csv"),
    ):
        _emit(provider, f"solve_data/{target}",
              derive_group_co2(input_dir, kind, provider=provider))


# ---------------------------------------------------------------------------
# Family 4 — simple_projections (legacy: preprocessing/simple_projections.py)
# ---------------------------------------------------------------------------

def derive_optional_yes(input_dir: Path,
                         *, provider: "object | None" = None,
                         ) -> pl.DataFrame:
    """optional_outputs filtered to value == 'yes'."""
    df = _read_csv(input_dir / "optional_outputs.csv",
                   ["output", "value"], provider=provider)
    return (
        df.filter(pl.col("value") == "yes")
          .select("output")
          .unique(maintain_order=True)
    )


def derive_reserve_upDown_group(input_dir: Path,
                                 *, provider: "object | None" = None,
                                 ) -> pl.DataFrame:
    """3-tuple (reserve, upDown, group) for method != 'no_reserve'."""
    df = _read_csv(
        input_dir / "reserve__upDown__group__method.csv",
        ["reserve", "upDown", "group", "method"],
        provider=provider,
    )
    return (
        df.filter(pl.col("method") != "no_reserve")
          .select("reserve", "upDown", "group")
          .unique(maintain_order=True)
    )


def derive_group_loss_share(input_dir: Path,
                             *, provider: "object | None" = None,
                             ) -> pl.DataFrame:
    df = _read_csv(input_dir / "group__loss_share_type.csv",
                   ["group", "type"], provider=provider)
    return (
        df.filter(pl.col("group") != "")
          .select("group")
          .unique(maintain_order=True)
    )


def derive_def_optional_yes(input_dir: Path,
                             *, provider: "object | None" = None,
                             ) -> pl.DataFrame:
    """def_optional_outputs filtered to 'yes' and not overridden 'no'."""
    explicit = _read_csv(input_dir / "optional_outputs.csv",
                         ["output", "value"], provider=provider)
    explicit_no = explicit.filter(pl.col("value") == "no").select("output")
    defaults = _read_csv(input_dir / "def_optional_outputs.csv",
                          ["output", "value"], provider=provider)
    return (
        defaults.filter(pl.col("value") == "yes")
                .join(explicit_no, on="output", how="anti")
                .select("output")
                .unique(maintain_order=True)
    )


def derive_process_delayed(solve_data_dir: Path,
                            *, provider: "object | None" = None,
                            ) -> pl.DataFrame:
    """Project ``process`` out of solve_data/process_delayed__duration.csv."""
    df = _read_csv(
        solve_data_dir / "process_delayed__duration.csv",
        ["process", "duration"],
        provider=provider,
    )
    return (
        df.filter(pl.col("process") != "")
          .select("process")
          .unique(maintain_order=True)
    )


def derive_process_side() -> pl.DataFrame:
    """Literal 2-element constant set."""
    return pl.DataFrame({"side": ["source", "sink"]})


def derive_period_solve(solve_data_dir: Path,
                         *, provider: "object | None" = None,
                         ) -> pl.DataFrame:
    """Project ``period`` out of solve_data/solve_period.csv."""
    df = _read_csv(solve_data_dir / "solve_period.csv",
                   ["solve", "period"], provider=provider)
    return (
        df.filter(pl.col("period") != "")
          .select("period")
          .unique(maintain_order=True)
    )


def derive_time_set(input_dir: Path,
                     *, provider: "object | None" = None,
                     ) -> pl.DataFrame:
    """Project ``time`` out of input/timeline.csv (cols: timeline, step, ...)."""
    df = _read_csv(input_dir / "timeline.csv",
                   ["timeline", "step"], provider=provider)
    return (
        df.filter(pl.col("step") != "")
          .select(pl.col("step").alias("time"))
          .unique(maintain_order=True)
    )


def derive_enable_optional_outputs(solve_data_dir: Path,
                                    *, provider: "object | None" = None,
                                    ) -> pl.DataFrame:
    """Union of optional_yes and def_optional_yes (order: optional first)."""
    a = _read_csv(solve_data_dir / "optional_yes.csv", ["output"],
                  provider=provider)
    b = _read_csv(solve_data_dir / "def_optional_yes.csv", ["output"],
                  provider=provider)
    return (
        pl.concat([a, b], how="vertical")
          .filter(pl.col("output") != "")
          .select("output")
          .unique(maintain_order=True)
    )


def derive_node_state_subset(
    solve_data_dir: Path, binding_method: str,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    """Filter nodeState by a specific storage_binding_method.

    ``binding_method`` is one of ``"bind_using_blended_weights"`` (→
    nodeState_rp) or ``"bind_intraperiod_blocks"`` (→ nodeStateBlock).
    """
    state = _read_csv(solve_data_dir / "nodeState.csv", ["node"],
                     provider=provider)
    binding = _read_csv(
        solve_data_dir / "node__storage_binding_method.csv",
        ["node", "method"], provider=provider,
    )
    matching = binding.filter(pl.col("method") == binding_method).select("node")
    return (
        state.join(matching, on="node", how="inner")
             .select("node")
             .unique(maintain_order=True)
    )


def derive_commodity_tier(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    """commodity__tier = commodity__tier_cum ∪ commodity__tier_ann."""
    cum = _read_csv(
        input_dir / "commodity_ladder_cumulative.csv",
        ["commodity", "tier"],
        provider=provider,
    )
    ann = _read_csv(
        solve_data_dir / "commodity__tier_ann.csv",
        ["commodity", "tier"],
        provider=provider,
    )
    return (
        pl.concat([cum, ann], how="vertical")
          .filter((pl.col("commodity") != "") & (pl.col("tier") != ""))
          .select("commodity", "tier")
          .unique(maintain_order=True)
    )


def derive_tier(commodity_tier: pl.DataFrame) -> pl.DataFrame:
    return commodity_tier.select("tier").unique(maintain_order=True)


# --- simple_setof_projections: 4 trivial single-column projections ---------

def derive_solve_period(input_dir: Path,
                         *, provider: "object | None" = None,
                         ) -> pl.DataFrame:
    """(solve, period) projected from input/timesets_in_use.csv."""
    df = _read_csv(
        input_dir / "timesets_in_use.csv",
        ["solve", "period", "tb"],
        provider=provider,
    )
    return (
        df.filter((pl.col("solve") != "") & (pl.col("period") != ""))
          .select("solve", "period")
          .unique(maintain_order=True)
    )


def derive_timeline(input_dir: Path,
                     *, provider: "object | None" = None,
                     ) -> pl.DataFrame:
    """``timeline`` projected from input/timesets__timeline.csv (col 1)."""
    df = _read_csv(
        input_dir / "timesets__timeline.csv", ["tb", "timeline"],
        provider=provider,
    )
    return (
        df.filter(pl.col("timeline") != "")
          .select("timeline")
          .unique(maintain_order=True)
    )


def derive_timeline_steps(input_dir: Path,
                           *, provider: "object | None" = None,
                           ) -> pl.DataFrame:
    """(timeline, step) projected from input/timeline.csv."""
    df = _read_csv(input_dir / "timeline.csv",
                   ["timeline", "step"], provider=provider)
    return (
        df.filter((pl.col("timeline") != "") & (pl.col("step") != ""))
          .select("timeline", "step")
          .unique(maintain_order=True)
    )


def derive_commodity_tier_ann(input_dir: Path,
                               *, provider: "object | None" = None,
                               ) -> pl.DataFrame:
    """(commodity, tier) projected from input/commodity_ladder_annual.csv.

    Header order: commodity, period, tier, price, quantity — tier is col 2.
    """
    df = _read_csv(
        input_dir / "commodity_ladder_annual.csv",
        ["commodity", "period", "tier", "price", "quantity"],
        provider=provider,
    )
    return (
        df.filter((pl.col("commodity") != "") & (pl.col("tier") != ""))
          .select("commodity", "tier")
          .unique(maintain_order=True)
    )


# --- orchestrators for simple_projections (preserves legacy call order) ----

def write_optional_yes(input_dir: Path, solve_data_dir: Path,
                        *, provider: "object | None" = None,
                        ) -> None:
    _write(derive_optional_yes(input_dir, provider=provider),
           solve_data_dir / "optional_yes.csv")


def write_reserve_upDown_group(input_dir: Path, solve_data_dir: Path,
                                *, provider: "object | None" = None,
                                ) -> None:
    _write(
        derive_reserve_upDown_group(input_dir, provider=provider),
        solve_data_dir / "reserve__upDown__group.csv",
    )


def write_group_loss_share(input_dir: Path, solve_data_dir: Path,
                            *, provider: "object | None" = None,
                            ) -> None:
    _write(derive_group_loss_share(input_dir, provider=provider),
           solve_data_dir / "group_loss_share.csv")


def write_def_optional_yes(input_dir: Path, solve_data_dir: Path,
                            *, provider: "object | None" = None,
                            ) -> None:
    _write(derive_def_optional_yes(input_dir, provider=provider),
           solve_data_dir / "def_optional_yes.csv")


def write_process_delayed(input_dir: Path, solve_data_dir: Path,
                           *, provider: "object | None" = None,
                           ) -> None:
    # input_dir is unused — kept for legacy signature parity.
    del input_dir
    _write(derive_process_delayed(solve_data_dir, provider=provider),
           solve_data_dir / "process_delayed.csv")


def write_process_side(solve_data_dir: Path,
                        *, provider: "object | None" = None,
                        ) -> None:
    # provider unused for the constant-set derivation; accepted for
    # late-binding override-dispatch parity.
    del provider
    _write(derive_process_side(), solve_data_dir / "process_side.csv")


def write_period_solve(solve_data_dir: Path,
                        *, provider: "object | None" = None,
                        ) -> None:
    _write(derive_period_solve(solve_data_dir, provider=provider),
           solve_data_dir / "period_solve.csv")


def write_time_set(input_dir: Path, solve_data_dir: Path,
                    *, provider: "object | None" = None,
                    ) -> None:
    _write(derive_time_set(input_dir, provider=provider),
           solve_data_dir / "time.csv")


def write_enable_optional_outputs(solve_data_dir: Path,
                                   *, provider: "object | None" = None,
                                   ) -> None:
    _write(
        derive_enable_optional_outputs(solve_data_dir, provider=provider),
        solve_data_dir / "enable_optional_outputs.csv",
    )


def write_node_state_subsets(solve_data_dir: Path,
                              *, provider: "object | None" = None,
                              ) -> None:
    rp = derive_node_state_subset(solve_data_dir, "bind_using_blended_weights",
                                  provider=provider)
    block = derive_node_state_subset(solve_data_dir, "bind_intraperiod_blocks",
                                     provider=provider)
    _write(rp, solve_data_dir / "nodeState_rp.csv")
    _write(block, solve_data_dir / "nodeStateBlock.csv")


def write_commodity_tier_sets(input_dir: Path, solve_data_dir: Path,
                                *, provider: "object | None" = None,
                                ) -> None:
    ct = derive_commodity_tier(input_dir, solve_data_dir, provider=provider)
    _write(ct, solve_data_dir / "commodity__tier.csv")
    _write(derive_tier(ct), solve_data_dir / "tier.csv")


def write_simple_setof_projections(input_dir: Path, solve_data_dir: Path,
                                     *, provider: "object | None" = None,
                                     ) -> None:
    _write(derive_solve_period(input_dir, provider=provider),
           solve_data_dir / "solve_period.csv")
    _write(derive_timeline(input_dir, provider=provider),
           solve_data_dir / "timeline.csv")
    _write(derive_timeline_steps(input_dir, provider=provider),
           solve_data_dir / "timeline_steps.csv")
    _write(derive_commodity_tier_ann(input_dir, provider=provider),
           solve_data_dir / "commodity__tier_ann.csv")


# --- emit_* twins for the simple_projections orchestrators ---------------

def emit_optional_yes(input_dir: Path, solve_data_dir: Path,
                       *, provider) -> None:
    """Provider-emitting twin of :func:`write_optional_yes`."""
    del solve_data_dir
    _emit(provider, "solve_data/optional_yes.csv",
          derive_optional_yes(input_dir, provider=provider))


def emit_reserve_upDown_group(input_dir: Path, solve_data_dir: Path,
                               *, provider) -> None:
    """Provider-emitting twin of :func:`write_reserve_upDown_group`."""
    del solve_data_dir
    _emit(provider, "solve_data/reserve__upDown__group.csv",
          derive_reserve_upDown_group(input_dir, provider=provider))


def emit_group_loss_share(input_dir: Path, solve_data_dir: Path,
                           *, provider) -> None:
    """Provider-emitting twin of :func:`write_group_loss_share`."""
    del solve_data_dir
    _emit(provider, "solve_data/group_loss_share.csv",
          derive_group_loss_share(input_dir, provider=provider))


def emit_def_optional_yes(input_dir: Path, solve_data_dir: Path,
                           *, provider) -> None:
    """Provider-emitting twin of :func:`write_def_optional_yes`."""
    del solve_data_dir
    _emit(provider, "solve_data/def_optional_yes.csv",
          derive_def_optional_yes(input_dir, provider=provider))


def emit_process_delayed(input_dir: Path, solve_data_dir: Path,
                          *, provider) -> None:
    """Provider-emitting twin of :func:`write_process_delayed`."""
    del input_dir
    _emit(provider, "solve_data/process_delayed.csv",
          derive_process_delayed(solve_data_dir, provider=provider))


def emit_process_side(solve_data_dir: Path, *, provider) -> None:
    """Provider-emitting twin of :func:`write_process_side`."""
    del solve_data_dir
    _emit(provider, "solve_data/process_side.csv", derive_process_side())


def emit_period_solve(solve_data_dir: Path, *, provider) -> None:
    """Provider-emitting twin of :func:`write_period_solve`."""
    _emit(provider, "solve_data/period_solve.csv",
          derive_period_solve(solve_data_dir, provider=provider))


def emit_time_set(input_dir: Path, solve_data_dir: Path,
                   *, provider) -> None:
    """Provider-emitting twin of :func:`write_time_set`."""
    del solve_data_dir
    _emit(provider, "solve_data/time.csv",
          derive_time_set(input_dir, provider=provider))


def emit_enable_optional_outputs(solve_data_dir: Path,
                                  *, provider) -> None:
    """Provider-emitting twin of :func:`write_enable_optional_outputs`."""
    _emit(provider, "solve_data/enable_optional_outputs.csv",
          derive_enable_optional_outputs(solve_data_dir, provider=provider))


def emit_node_state_subsets(solve_data_dir: Path,
                             *, provider) -> None:
    """Provider-emitting twin of :func:`write_node_state_subsets`."""
    rp = derive_node_state_subset(solve_data_dir, "bind_using_blended_weights",
                                  provider=provider)
    block = derive_node_state_subset(solve_data_dir, "bind_intraperiod_blocks",
                                     provider=provider)
    _emit(provider, "solve_data/nodeState_rp.csv", rp)
    _emit(provider, "solve_data/nodeStateBlock.csv", block)


def emit_commodity_tier_sets(input_dir: Path, solve_data_dir: Path,
                              *, provider) -> None:
    """Provider-emitting twin of :func:`write_commodity_tier_sets`."""
    ct = derive_commodity_tier(input_dir, solve_data_dir, provider=provider)
    _emit(provider, "solve_data/commodity__tier.csv", ct)
    _emit(provider, "solve_data/tier.csv", derive_tier(ct))


def emit_simple_setof_projections(input_dir: Path, solve_data_dir: Path,
                                    *, provider) -> None:
    """Provider-emitting twin of :func:`write_simple_setof_projections`."""
    del solve_data_dir
    _emit(provider, "solve_data/solve_period.csv",
          derive_solve_period(input_dir, provider=provider))
    _emit(provider, "solve_data/timeline.csv",
          derive_timeline(input_dir, provider=provider))
    _emit(provider, "solve_data/timeline_steps.csv",
          derive_timeline_steps(input_dir, provider=provider))
    _emit(provider, "solve_data/commodity__tier_ann.csv",
          derive_commodity_tier_ann(input_dir, provider=provider))
