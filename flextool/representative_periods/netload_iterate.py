"""Solve-iteration driver for net-load representative-period selection (Phase 4).

This composes the single-shot net-load selector
(:func:`flextool.representative_periods.preprocess.preprocess_representative_periods`
with ``netload_clustering=True``) into an *iterate-until-stable* loop that
feeds each iteration's solved investment capacities back into the next
selection.  The net-load signal is ``demand − Σ VRE·avail``; sizing the VRE
fleet by a solve's realised investments (instead of the iteration-0
demand-match default) sharpens which periods are net-load-critical, so the
representative set converges on the ones the *invested* system actually
stresses.

Loop shape (option C — cheap by default, keep-best opt-in)
----------------------------------------------------------
Iteration ``k`` in ``range(iterations + 1)``; ``k = 0`` is the BOOTSTRAP:

* **k = 0 (bootstrap).** Select on the demand-match default caps
  (``solved_caps=None``), run the INVEST-ONLY solve, read the invested caps.
  NO dispatch is run on the bootstrap — default-cap investments are known-bad,
  so their full-year cost is not worth measuring.
* **k = 1..n.** Select with ``solved_caps = caps_{k-1}`` (the prior
  iteration's invested fleet).  If the newly-selected representative set is
  UNCHANGED from the previous iteration → converged, break before solving.
  Otherwise run the INVEST-ONLY solve and read the caps.  When ``keep_best``
  is set, ALSO run the full-year DISPATCH for this mature iteration, read its
  total system cost, and track the lowest-cost representative set.
* **After the loop.** The chosen representative set is the keep-best winner
  (if any dispatch ran and one was cheaper) else the last-selected set.  When
  keep-best picked an earlier iteration, its set is RE-MATERIALISED by
  re-running the selector with that iteration's *input* caps — the net-load
  selection is a deterministic function of the fed-back caps (greedy convex
  hull over a fixed matrix, no randomness), so the same input caps reproduce
  the exact same representative set into the pinned alternative.  Finally ONE
  DISPATCH is run on the chosen set for output.

Invariants owned here
----------------------
* **Stable alternative name.**  The full ``alternative_name`` is pinned ONCE
  in the config and passed as the explicit override to the selector EVERY
  iteration, so the selector never derives a per-iteration name (whose
  force-suffix could change as the forced count varies) and orphans timesets.
  A stable name lets the selector's purge+rewrite cleanly REPLACE the single
  timeset each iteration — running N iterations leaves exactly ONE net-load
  timeset alternative, never an accumulation.
* **No shared warm-start basis.**  The LP columns change whenever the
  representative set changes, so a basis cached from a prior iteration would be
  structurally stale.  Each solve gets its OWN fresh (cold) basis-cache dir by
  default (``warm_start=False``); ``warm_start=True`` opts into ONE shared dir
  for the whole run (faster, but only safe if the grid is stable — use with
  care).  This deliberately contrasts the adequacy calibrator, whose grid is
  fixed across iterations so it shares one cache.

Solve subset mechanism
----------------------
``cmd_run_flextool`` has NO ``--solves`` flag; the authoritative, DB-driven
way to run a subset of a scenario's solves is the ``model.solves`` Array,
resolved under the scenario's alternative stack
(:meth:`flextool.engine_polars._solve_config.SolveConfig.load_from_db`).  The
driver therefore writes a dedicated ``model.solves`` OVERRIDE alternative at
the TOP of the scenario's stack (mirroring
:mod:`flextool.calibrate._db_alt`): set to the invest solve(s) for an
invest-only pass, and to the dispatch solves (or the scenario's full original
solve list) for a dispatch pass.  This is a genuine subset run, not a silent
full-chain fallback.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import spinedb_api as api
from spinedb_api import Array, DatabaseMapping, import_data
from spinedb_api.exception import NothingToCommit

from flextool.calibrate._readers import (
    read_total_system_cost,
    read_unit_capacity_total,
)
from flextool.calibrate._solve import run_solve
from flextool.calibrate._solve_status import assess_solve
from flextool.engine_polars._db_reader import get_single_entities
from flextool.gui.solve_reader import read_scenario_solves
from flextool.representative_periods.preprocess import (
    preprocess_representative_periods,
)
from flextool.representative_periods.scenario_stack import (
    add_alternative_to_scenario,
)

# Required-output gates for the fail-closed solve assessment, per pass.
_INVEST_REQUIRED = ("unit_capacity_ed_p", "node_slack_up_d_e")
_DISPATCH_REQUIRED = ("costs_discounted_p_", "node_slack_up_d_e")


class NetloadIterError(RuntimeError):
    """Raised when a net-load iteration cannot proceed.

    Covers both a fail-closed solve (an untrusted iteration solve) and the
    mutual-exclusion guard against a timed per-cell ``energy_margin_adder``.
    """


@dataclass
class NetloadIterConfig:
    """Configuration for one net-load iteration run.

    Attributes:
        n_rp: Number of representative periods to select.
        period_length: Timesteps per representative period.
        iterations: Number of ADJUSTMENT iterations AFTER the bootstrap; the
            loop runs at most ``iterations + 1`` selections (``iterations = 0``
            is bootstrap-only).
        scenario: The scenario (model instance) to select + solve.
        invest_solves: Solve name(s) to run for the INVEST-ONLY pass (written
            into the ``model.solves`` override alternative).
        dispatch_solves: Solve name(s) for the full-year DISPATCH pass;
            ``None`` means "the scenario's original ``model.solves`` chain"
            (invest + dispatch, so the full chain re-invests on the chosen
            representative set then dispatches over the full year).
        work_dir: Working directory for the per-iteration solve subprocesses.
        out_root: Output-location root; solve outputs land under
            ``out_root/output_parquet/<scenario>/``.
        alternative_name: STABLE alternative name pinned by the driver and
            passed to the selector every iteration (see the module docstring —
            the #1 correctness guard against orphaned timesets).
        vre_penetration: Energy-share target threaded into the demand-match
            default caps for the bootstrap selection.
        keep_best: When set, dispatch every MATURE iteration (k >= 1) and keep
            the lowest full-year-cost representative set.
        warm_start: When ``False`` (default) every solve gets a fresh cold
            basis cache (safe — the grid changes across iterations); ``True``
            shares one cache for the whole run (faster, only safe on a stable
            grid).
    """

    n_rp: int
    period_length: int
    iterations: int
    scenario: str
    invest_solves: list[str]
    dispatch_solves: list[str] | None
    work_dir: Path
    out_root: Path
    alternative_name: str
    vre_penetration: float = 1.0
    keep_best: bool = False
    warm_start: bool = False


@dataclass
class NetloadIterRecord:
    """One iteration's observed state (the iteration trajectory element).

    Attributes:
        iteration: The iteration index ``k`` (0 = bootstrap).
        input_caps: The ``solved_caps`` fed INTO this iteration's selection
            (``None`` for the bootstrap).  Kept so the keep-best winner can be
            deterministically re-materialised by re-running the selector with
            the same input caps.
        rep_starts: The representative-period start keys selected this
            iteration — the rp-set identity used for convergence.
        solved_caps: The invested capacities READ from this iteration's
            invest-only solve.
        dispatch_cost: The full-year total system cost when this iteration was
            dispatched (keep-best, mature iterations only), else ``None``.
    """

    iteration: int
    input_caps: dict[str, float] | None
    rep_starts: tuple[str, ...]
    solved_caps: dict[str, float]
    dispatch_cost: float | None = None


@dataclass
class NetloadIterResult:
    """Outcome of a net-load iteration run.

    Attributes:
        iterations_run: Number of invest-only solves actually performed.
        converged: Whether the representative set stabilised (an iteration
            re-selected the previous set) within the budget.
        stop_reason: ``"converged"`` (rp set stabilised), ``"bootstrap_only"``
            (``iterations == 0``), or ``"budget_exhausted"`` (ran the full
            adjustment budget without stabilising).
        final_rep_starts: The representative set written to the pinned
            alternative and dispatched for output.
        final_caps: The invested capacities associated with the final set.
        best_cost: The lowest full-year dispatch cost seen (keep-best), else
            ``None``.
        final_cost: The full-year total system cost of the final output
            dispatch.
        alternative_name: The pinned alternative the final set lives in.
        trajectory: Per-iteration :class:`NetloadIterRecord` list.
    """

    iterations_run: int
    converged: bool
    stop_reason: str
    final_rep_starts: tuple[str, ...]
    final_caps: dict[str, float]
    best_cost: float | None
    final_cost: float
    alternative_name: str
    trajectory: list[NetloadIterRecord] = field(default_factory=list)


def _normalise_url(url: str) -> str:
    """Accept either a bare filesystem path or a full SQLAlchemy URL."""
    return url if "://" in url else f"sqlite:///{url}"


def _guard_no_timed_energy_margin(url: str, scenario: str) -> None:
    """Fail closed when the scenario has an ACTIVE timed per-cell adder.

    The ``energy_margin`` calibrator's ``pdt_energy_margin_adder`` places
    demand margin at specific ``(period, time)`` representative CELLS.
    Re-selecting the representative periods between iterations moves those
    cells, stranding the adder, so net-load iteration is mutually exclusive
    with an active timed adder.  A node whose ``energy_margin_method`` resolves
    to ``inflow_adder`` AND whose ``energy_margin_adder`` is a 2-D
    (``period -> time -> value``) :class:`spinedb_api.Map` triggers a clear
    error.  A scalar (uniform) or 1-D (period-only) adder is grid-independent
    and allowed; an adder whose method is not ``inflow_adder`` is inert and
    ignored.  Read under the scenario filter so the *effective* (top-rank)
    method and adder value are seen.
    """
    scen_config = api.filters.scenario_filter.scenario_filter_config(scenario)
    with DatabaseMapping(_normalise_url(url)) as db:
        api.filters.scenario_filter.scenario_filter_from_dict(db, scen_config)
        db.fetch_all("parameter_value")

        methods: dict[str, object] = {}
        for pv in db.find_parameter_values(
            entity_class_name="node",
            parameter_definition_name="energy_margin_method",
        ):
            methods[pv["entity_name"]] = api.from_database(
                pv["value"], pv["type"]
            )

        for pv in db.find_parameter_values(
            entity_class_name="node",
            parameter_definition_name="energy_margin_adder",
        ):
            node = pv["entity_name"]
            if methods.get(node) != "inflow_adder":
                continue
            value = api.from_database(pv["value"], pv["type"])
            if isinstance(value, api.Map) and any(
                isinstance(inner, api.Map) for inner in value.values
            ):
                raise NetloadIterError(
                    f"Scenario '{scenario}' has an ACTIVE timed per-cell "
                    f"energy_margin_adder on node '{node}' "
                    f"(period -> time Map with method 'inflow_adder'). Net-load "
                    f"iteration re-selects representative periods each round, "
                    f"which would strand the per-cell adder — the two features "
                    f"are mutually exclusive. Use a scalar (uniform) adder, or "
                    f"run the net-load iteration without the timed margin."
                )


def _full_chain_solves(url: str, scenario: str) -> list[str]:
    """Return the scenario's ordered ``model.solves`` names (the full chain)."""
    return [si.name for si in read_scenario_solves(_normalise_url(url), scenario)]


def _write_solves_override_alt(
    url: str, scenario: str, solves: list[str], alt_name: str
) -> None:
    """Set ``model.solves`` = *solves* in a top-of-stack override alternative.

    Writes (idempotently, via ``import_data`` merge) the ``model.solves`` Array
    onto every ``model`` entity under a dedicated override alternative, then
    appends that alternative to *scenario*'s stack at the top rank so its value
    WINS over the baseline solve list.  Re-writing a changed subset UPDATEs the
    single row in place.  When the database has no ``model`` entity (the
    auto-wired single-solve case) there is nothing to override, so the natural
    single solve runs and the override is skipped.
    """
    with DatabaseMapping(_normalise_url(url)) as db:
        models = get_single_entities(db=db, entity_class_name="model")
        if not models:
            return
        entities = [("model", m) for m in models]
        pvs = [
            ("model", m, "solves", Array(list(solves)), alt_name)
            for m in models
        ]
        _count, errors = import_data(
            db,
            alternatives=[alt_name],
            entities=entities,
            parameter_values=pvs,
        )
        assert not errors, errors
        try:
            db.commit_session(f"netload solves override: {solves}")
        except NothingToCommit:
            pass

        existing = db.get_scenario_alternative_items(scenario_name=scenario)
        if not any(sa["alternative_name"] == alt_name for sa in existing):
            next_rank = max((sa["rank"] for sa in existing), default=0) + 1
            db.add_scenario_alternative(
                scenario_name=scenario, alternative_name=alt_name, rank=next_rank
            )
            try:
                db.commit_session("netload solves override link")
            except NothingToCommit:
                pass


def _read_rep_starts(url: str, timeset_name: str, alternative_name: str) -> tuple[str, ...]:
    """Read the selected representative-period start keys (the rp-set identity).

    The ``timeset_duration`` Map the selector writes is keyed by representative
    period START timestep; its ordered index IS the representative set.  Read
    it back from the pinned ``(timeset, alternative)`` so the driver can detect
    when a re-selection reproduces the previous set (convergence).
    """
    with DatabaseMapping(_normalise_url(url)) as db:
        db.fetch_all("parameter_value")
        for pv in db.find_parameter_values(
            entity_class_name="timeset",
            parameter_definition_name="timeset_duration",
        ):
            if (
                pv["entity_name"] == timeset_name
                and pv["alternative_name"] == alternative_name
            ):
                value = api.from_database(pv["value"], pv["type"])
                return tuple(str(idx) for idx in value.indexes)
    return ()


def _select(
    url: str, config: NetloadIterConfig, solved_caps: dict[str, float] | None
) -> tuple[str, ...]:
    """Run one net-load selection into the pinned alternative; return its set.

    Always passes the PINNED ``alternative_name`` and restricts the
    ``period_timeset`` repoint to the invest solves (so only the invest pass
    runs on the representative timeset; the dispatch keeps its full-year
    timeset).  Returns the selected representative-period start keys.
    """
    timeset_name = preprocess_representative_periods(
        url,
        scenario_name=config.scenario,
        n_rp=config.n_rp,
        period_length=config.period_length,
        netload_clustering=True,
        vre_penetration=config.vre_penetration,
        solved_caps=solved_caps,
        alternative_name=config.alternative_name,
        solves=list(config.invest_solves),
    )
    add_alternative_to_scenario(url, config.scenario, config.alternative_name)
    return _read_rep_starts(url, timeset_name, config.alternative_name)


def _solve_once(
    url: str,
    config: NetloadIterConfig,
    *,
    solves: list[str],
    required: tuple[str, ...],
    cache_tag: str,
) -> Path:
    """Write the solves override, run one fail-closed solve, return assess_dir.

    *cache_tag* names the per-solve basis-cache dir; a fresh (cold) dir per
    solve is used unless ``warm_start`` opts into one shared dir.
    """
    _write_solves_override_alt(
        url, config.scenario, solves, f"{config.alternative_name}_solves"
    )
    cache_dir = config.work_dir / (
        "warm_shared" if config.warm_start else f"cache_{cache_tag}"
    )
    run = run_solve(
        url,
        config.scenario,
        work_dir=config.work_dir,
        out_root=config.out_root,
        cache_dir=cache_dir,
    )
    outcome = assess_solve(
        run.assess_dir,
        exit_code=run.returncode,
        required_outputs=list(required),
        started_at=run.started_at,
    )
    if not outcome.succeeded:
        raise NetloadIterError(
            f"solve of {solves} for scenario '{config.scenario}' not "
            f"trusted: {outcome.reason}"
        )
    return run.assess_dir


def run_netload_iteration(
    url: str, config: NetloadIterConfig
) -> NetloadIterResult:
    """Run the net-load representative-period iteration for *config.scenario*.

    See the module docstring for the loop shape.  Raises
    :class:`NetloadIterError` on an active timed energy-margin adder (the
    mutual-exclusion guard) or any untrusted iteration solve (fail-closed).
    """
    # Fail closed BEFORE any solve if a timed per-cell adder is active.
    _guard_no_timed_energy_margin(url, config.scenario)

    dispatch_solves = (
        config.dispatch_solves
        if config.dispatch_solves is not None
        else _full_chain_solves(url, config.scenario)
    )

    trajectory: list[NetloadIterRecord] = []
    prev_starts: tuple[str, ...] | None = None
    caps: dict[str, float] = {}
    best_cost: float | None = None
    best_input_caps: dict[str, float] | None = None
    best_starts: tuple[str, ...] | None = None
    best_caps: dict[str, float] | None = None
    converged = False
    stop_reason = "bootstrap_only" if config.iterations == 0 else "budget_exhausted"
    iterations_run = 0
    last_starts: tuple[str, ...] = ()

    for k in range(config.iterations + 1):
        input_caps = None if k == 0 else dict(caps)
        print(
            f"====== netload[{config.scenario}] iteration "
            f"{k}/{config.iterations} — "
            f"{'bootstrap (demand-match caps)' if k == 0 else 'fed-back caps'} "
            f"======",
            flush=True,
        )

        rep_starts = _select(url, config, input_caps)
        last_starts = rep_starts

        # Convergence: a re-selection reproducing the previous set means the
        # fixed point is reached — stop before spending another solve.
        if prev_starts is not None and rep_starts == prev_starts:
            converged = True
            stop_reason = "converged"
            print(
                f"------ iteration {k}: representative set unchanged "
                f"({len(rep_starts)} periods) — converged ------",
                flush=True,
            )
            break

        # Invest-only pass: read the fleet this representative set invests in.
        assess_dir = _solve_once(
            url,
            config,
            solves=list(config.invest_solves),
            required=_INVEST_REQUIRED,
            cache_tag=f"invest_{k}",
        )
        caps = read_unit_capacity_total(assess_dir)
        iterations_run += 1

        dispatch_cost: float | None = None
        # Keep-best dispatches only MATURE iterations (k >= 1); the bootstrap's
        # default-cap investments are known-bad and never dispatched.
        if config.keep_best and k >= 1:
            d_assess = _solve_once(
                url,
                config,
                solves=list(dispatch_solves),
                required=_DISPATCH_REQUIRED,
                cache_tag=f"dispatch_{k}",
            )
            dispatch_cost = read_total_system_cost(d_assess)
            if best_cost is None or dispatch_cost < best_cost:
                best_cost = dispatch_cost
                best_input_caps = input_caps
                best_starts = rep_starts
                best_caps = dict(caps)
            print(
                f"------ iteration {k}: full-year cost {dispatch_cost:.4f} M€ "
                f"(best {best_cost:.4f} M€) ------",
                flush=True,
            )

        trajectory.append(
            NetloadIterRecord(
                iteration=k,
                input_caps=input_caps,
                rep_starts=rep_starts,
                solved_caps=dict(caps),
                dispatch_cost=dispatch_cost,
            )
        )
        prev_starts = rep_starts

    # Choose the final representative set: the keep-best winner if one was
    # dispatched cheaper, else the last-selected set.
    if config.keep_best and best_starts is not None:
        final_starts = best_starts
        final_caps = dict(best_caps) if best_caps is not None else {}
        # Re-materialise the winning set deterministically by re-selecting with
        # the same input caps that produced it (idempotent when it is already
        # the current pinned set).
        reproduced = _select(url, config, best_input_caps)
        if reproduced != final_starts:
            raise NetloadIterError(
                "keep-best re-selection did not reproduce the winning "
                f"representative set (expected {final_starts}, got "
                f"{reproduced}); net-load selection is expected to be a "
                "deterministic function of the fed-back caps."
            )
    else:
        final_starts = last_starts
        final_caps = dict(caps)

    # One final DISPATCH on the chosen set for output.
    print(
        f"====== netload[{config.scenario}] final dispatch on "
        f"{len(final_starts)} representative periods ======",
        flush=True,
    )
    final_assess = _solve_once(
        url,
        config,
        solves=list(dispatch_solves),
        required=_DISPATCH_REQUIRED,
        cache_tag="final",
    )
    final_cost = read_total_system_cost(final_assess)

    print(
        f"====== netload[{config.scenario}] finished: {stop_reason} "
        f"(converged={converged}), {iterations_run} invest solve(s), "
        f"final cost {final_cost:.4f} M€ ======",
        flush=True,
    )

    return NetloadIterResult(
        iterations_run=iterations_run,
        converged=converged,
        stop_reason=stop_reason,
        final_rep_starts=final_starts,
        final_caps=final_caps,
        best_cost=best_cost,
        final_cost=final_cost,
        alternative_name=config.alternative_name,
        trajectory=trajectory,
    )


def _resolve_solves(
    url: str,
    scenario: str,
    invest_solves: list[str] | None,
    dispatch_solves: list[str] | None,
) -> tuple[list[str], list[str] | None]:
    """Resolve invest / dispatch solve lists, auto-detecting when not given.

    When ``invest_solves`` is not supplied, auto-detect it as the scenario's
    solves that carry a non-empty ``invest_periods`` Array (an investment
    solve — the exact test :mod:`flextool.gui.solve_reader` uses).  An explicit
    list is used verbatim.  ``dispatch_solves`` is passed through unchanged
    (``None`` = the scenario's full original ``model.solves`` chain).  Raises
    when no invest solve can be resolved.
    """
    resolved_invest = invest_solves
    if not resolved_invest:
        infos = read_scenario_solves(_normalise_url(url), scenario)
        resolved_invest = [si.name for si in infos if si.has_invest_periods]
        if not resolved_invest:
            raise NetloadIterError(
                f"No invest solve found for scenario '{scenario}': none of its "
                f"solves {[si.name for si in infos]} carry a non-empty "
                f"invest_periods, and --invest-solves was not given."
            )
    return resolved_invest, dispatch_solves


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the net-load iteration CLI."""
    parser = argparse.ArgumentParser(
        prog="flextool.representative_periods.netload_iterate",
        description=(
            "Iterate net-load representative-period selection, feeding each "
            "iteration's solved investment capacities back into the next "
            "selection until the representative set stabilises."
        ),
    )
    parser.add_argument("db", help="SpineDB URL (or path) of the model.")
    parser.add_argument("scenario", help="Scenario (model instance) to run.")
    parser.add_argument("--n-rp", type=int, required=True, dest="n_rp",
                        help="Number of representative periods to select.")
    parser.add_argument("--period-length", type=int, required=True,
                        dest="period_length",
                        help="Timesteps per representative period.")
    parser.add_argument("--iterations", type=int, required=True,
                        help="Adjustment iterations after the bootstrap "
                             "(0 = bootstrap-only).")
    parser.add_argument("--invest-solves", type=str, default=None,
                        dest="invest_solves",
                        help="Comma-separated invest solve name(s) for the "
                             "invest-only pass. Omitted → auto-detect solves "
                             "with a non-empty invest_periods.")
    parser.add_argument("--dispatch-solves", type=str, default=None,
                        dest="dispatch_solves",
                        help="Comma-separated dispatch solve name(s) for the "
                             "full-year pass. Omitted → the scenario's full "
                             "original model.solves chain.")
    parser.add_argument("--vre-penetration", type=float, default=1.0,
                        dest="vre_penetration",
                        help="Energy-share target for the bootstrap "
                             "demand-match caps (default 1.0).")
    parser.add_argument("--keep-best", action="store_true", dest="keep_best",
                        help="Dispatch each mature iteration and keep the "
                             "lowest full-year-cost representative set.")
    parser.add_argument("--warm-start", action="store_true", dest="warm_start",
                        help="Share one basis cache across all solves (faster "
                             "but only safe on a stable grid; OFF by default "
                             "because the grid changes each iteration).")
    parser.add_argument("--work-dir", type=Path, default=Path("netload_work"),
                        dest="work_dir",
                        help="Working directory for the per-iteration solves.")
    parser.add_argument("--output-location", type=Path,
                        default=Path("netload_out"), dest="out_root",
                        help="Output-location root for the solve outputs.")
    parser.add_argument("--alternative-name", type=str, default=None,
                        dest="alternative_name",
                        help="Override the pinned output alternative name "
                             "(default: netload_<n_rp>rp_<PL>h).")
    return parser


def _split(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [s.strip() for s in value.split(",") if s.strip()]


def main(argv: list[str] | None = None) -> int:
    """Run the net-load iteration from the command line.

    Returns 0 on success; on a :class:`NetloadIterError` (fail-closed solve or
    the timed-adder guard) prints the reason to stderr and returns 1.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        invest_solves, dispatch_solves = _resolve_solves(
            args.db,
            args.scenario,
            _split(args.invest_solves),
            _split(args.dispatch_solves),
        )
    except NetloadIterError as exc:
        print(f"net-load iteration failed: {exc}", file=sys.stderr)
        return 1

    alternative_name = args.alternative_name or (
        f"netload_{args.n_rp}rp_{args.period_length}h"
    )
    config = NetloadIterConfig(
        n_rp=args.n_rp,
        period_length=args.period_length,
        iterations=args.iterations,
        scenario=args.scenario,
        invest_solves=invest_solves,
        dispatch_solves=dispatch_solves,
        work_dir=Path(args.work_dir),
        out_root=Path(args.out_root),
        alternative_name=alternative_name,
        vre_penetration=args.vre_penetration,
        keep_best=args.keep_best,
        warm_start=args.warm_start,
    )

    try:
        result = run_netload_iteration(args.db, config)
    except NetloadIterError as exc:
        print(f"net-load iteration failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"\nNet-load iteration finished: {result.stop_reason} "
        f"(converged={result.converged}), "
        f"{result.iterations_run} invest solve(s).\n"
        f"  Representative set ({len(result.final_rep_starts)}): "
        f"{list(result.final_rep_starts)}\n"
        f"  Alternative: '{result.alternative_name}'\n"
        f"  Final full-year cost: {result.final_cost:.4f} M€"
        + (
            f"  (best iterated: {result.best_cost:.4f} M€)"
            if result.best_cost is not None
            else ""
        )
    )
    return 0


__all__ = [
    "NetloadIterConfig",
    "NetloadIterError",
    "NetloadIterRecord",
    "NetloadIterResult",
    "build_parser",
    "main",
    "run_netload_iteration",
]


if __name__ == "__main__":
    sys.exit(main())
