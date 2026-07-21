"""Resilient solve-success detection for the energy-margin calibrator.

The calibrator runs an investment+dispatch solve each iteration by
*shelling out* to :mod:`flextool.cli.cmd_run_flextool` and then reads the
per-node unserved-energy slack from the produced outputs.  Before it can
trust those numbers it must answer one question: **did this solve actually
succeed?**  This module is that answer.

Why the subprocess exit code is not enough
------------------------------------------
Success is ``f(solve-status signals, output completeness)`` with the exit
code as only *one weak input*, because the exit code lies in both
directions:

* **False failure (nonzero exit, good solve).**  A known *model-specific*
  post-solve writer bug — ``Shared-alternative write failed: '<REG>'``
  ``KeyError`` in the separate PLEXOS→FlexTool writer, **not** in this
  engine — can raise *after* the cascade has solved and written every
  output, bubbling to a nonzero exit.  The results on disk are complete
  and usable; the run must be treated as a success.

* **False success (zero exit, missing results).**  A run that never
  reached, or aborted inside, output writing can still exit cleanly in
  some paths; if the calibrator's required result files are absent it must
  be treated as a failure regardless of the exit code.

What FlexTool actually leaves on disk
-------------------------------------
FlexTool does **not** persist a per-sub-solve optimality/acceptance status
file.  The authoritative "was this solve acceptable" decision
(:func:`flextool.engine_polars._solve_acceptance.classify_acceptance`, run
at the solve site, and the cascade exit-scan
``flextool.cli.cmd_run_flextool._scan_cascade_optimality`` that consumes
it) lives *in memory* and is surfaced only via:

* the process **exit code** (0 iff every sub-solve was ``kOptimal``,
  accepted near-optimal, or a Benders solve with a feasible incumbent;
  1 on a genuine failure), and
* **log lines** on stdout/stderr.

Crucially, ``cmd_run_flextool`` calls ``write_outputs`` **only when the
cascade returned success** — a genuinely failed / infeasible / unaccepted
solve short-circuits with ``return_code == 1`` and writes *no* output
files at all.  Therefore, on a fresh output directory, the **presence and
non-emptiness of the required result parquets is itself the on-disk
signal that every sub-solve was accepted**: a failed sub-solve manifests
as *missing outputs*, not as a status flag.

The detector's rule
-------------------
``outputs_complete`` = every required output parquet is present, is a
readable parquet with at least one row, and (when ``started_at`` is given)
was written by *this* run rather than left over from a previous one:

* not complete                      → **failed** (name the offending files);
* complete + exit 0/``None``        → **succeeded** (``started_at`` optional);
* complete + nonzero exit + fresh   → **succeeded**, the nonzero exit is
  recorded as *overridden* (the post-solve-writer-crash case);
* complete + nonzero exit + freshness UNVERIFIABLE (no ``started_at``)
  → **failed** — the override is refused because it cannot be made safely.

Stale-output caveat (why ``started_at`` matters)
------------------------------------------------
When a solve genuinely fails, ``write_outputs`` is skipped and the parquet
directory is **not** emptied, so a *previous* successful run's files can
linger.  "outputs complete + nonzero exit" would then wrongly look like
the writer-crash case.  So the nonzero-exit → success override is allowed
ONLY when ``started_at`` (the wall-clock time the calibrator launched the
subprocess) was supplied AND every required output is at least that new; a
stale file fails the completeness check, and a nonzero exit with no
``started_at`` at all is treated as a **failure** (freshness unverifiable —
the override cannot be made safely).  **C1 must always pass ``started_at``.**
The exit-0 / no-exit path does not need it: a successful ``write_outputs``
empties then rewrites the parquet dir, so its files are this run's product
by construction.

This module never solves and never touches the network: it is a pure
post-hoc reader of a solve's output directory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

# The calibrator's load-bearing signals: per-period node up-slack (unserved
# energy) and the discounted per-entity node cost table (its 'upward slack
# penalty' category is the monetised slack).  Held as REGISTRY keys — the
# on-disk filenames are resolved through the parquet-bundle registry so a
# schema/rename breaks loudly *here* rather than silently missing a file.
# Only ``node_slack_up_d_e`` is a robust success gate: it is *dense* — one
# row per period for every balance node, emitted unconditionally
# (out_node.py, ``v.q_state_up`` clipped ≥0) — so a valid solve NEVER omits
# it, even at zero slack.  ``cost_node_discounted_d_ec`` is deliberately NOT
# a default requirement: out_costs.py skips a node cost category with an
# ``if not pieces: continue`` guard, so a legitimate solve can omit that
# table → it would cause a false FAIL.  The calibrator READS it for the
# penalty M€, but presence of the slack table is the success signal.
_DEFAULT_REQUIRED_KEYS: tuple[str, ...] = (
    "node_slack_up_d_e",
)


def _registry_filename(key: str) -> str:
    """Resolve a processed-output *key* to its on-disk parquet basename.

    Validated against
    :data:`flextool.process_outputs._output_meta.OUTPUT_TRANSFORM` — the
    single-source registry of every processed output table name (the same
    keys ``write_outputs`` uses when it writes ``<key>.parquet``).  We use
    this rather than
    :data:`flextool.engine_polars._parquet_bundle.REGISTRY`, whose
    processed-output coverage is documented as REPRESENTATIVE / incomplete
    by design (``cost_node_discounted_d_ec`` is absent there).  A key not in
    the registry raises loudly here instead of silently looking for a file
    that can never exist — so a schema rename that updates ``OUTPUT_TRANSFORM``
    surfaces as a clear error at the calibrator boundary.
    """
    from flextool.process_outputs._output_meta import OUTPUT_TRANSFORM

    if key not in OUTPUT_TRANSFORM:
        raise KeyError(
            f"{key!r} is not a registered FlexTool output "
            "(flextool.process_outputs._output_meta.OUTPUT_TRANSFORM). "
            "The calibrator's required-output default is stale; update "
            "_DEFAULT_REQUIRED_KEYS or pass required_outputs explicitly."
        )
    return f"{key}.parquet"


def default_required_outputs() -> tuple[str, ...]:
    """The default required-output filenames, resolved via the registry.

    Returns the ``*.parquet`` basenames the calibrator minimally needs to
    trust a solve: the node up-slack and the discounted node-cost table.
    """
    return tuple(_registry_filename(k) for k in _DEFAULT_REQUIRED_KEYS)


def _normalise_required(
    required_outputs: Sequence[str] | None,
) -> list[str]:
    """Normalise the caller's required-output list to ``*.parquet`` basenames.

    Accepts either REGISTRY keys (e.g. ``"node_slack_up_d_e"``) or explicit
    filenames (e.g. ``"node_slack_up_d_e.parquet"``); a bare key is resolved
    through the registry, a ``*.parquet`` name is taken verbatim.  ``None``
    yields :func:`default_required_outputs`.
    """
    if required_outputs is None:
        return list(default_required_outputs())
    resolved: list[str] = []
    for item in required_outputs:
        name = str(item)
        if name.endswith(".parquet"):
            resolved.append(name)
        else:
            resolved.append(_registry_filename(name))
    return resolved


def _as_epoch(started_at: float | int | datetime | None) -> float | None:
    """Coerce a ``started_at`` marker to a POSIX timestamp, or ``None``.

    The recommended input (what C1 passes) is a plain epoch ``float`` from
    :func:`time.time`, which compares directly against ``st_mtime``.  A
    tz-aware :class:`datetime` also compares correctly.  A *naive* datetime
    is interpreted as LOCAL time via :meth:`datetime.timestamp` — the same
    convention ``st_mtime`` follows on POSIX — so it is safe (not silently
    skewed); prefer the epoch float to avoid any ambiguity.
    """
    if started_at is None:
        return None
    if isinstance(started_at, datetime):
        # ``.timestamp()`` assumes LOCAL time for a naive datetime, matching
        # how ``st_mtime`` (also epoch) relates to local wall-clock; a
        # tz-aware datetime converts exactly.  No skew either way.
        return started_at.timestamp()
    return float(started_at)


@dataclass
class OutputCheck:
    """Per-required-output evidence gathered from the output directory.

    FlexTool persists no per-*sub-solve* status to disk (see the module
    docstring), so this per-*output* record is the finest-grained on-disk
    success evidence available.  It is what :attr:`SolveOutcome.per_solve`
    carries.
    """

    filename: str
    present: bool
    num_rows: int | None
    fresh: bool
    detail: str

    @property
    def ok(self) -> bool:
        """Whether this output counts toward completeness.

        Requires the file to be present, a readable parquet with at least
        one row, and (subject to ``started_at``) fresh.
        """
        return self.present and (self.num_rows or 0) > 0 and self.fresh


@dataclass
class SolveOutcome:
    """Verdict on a completed (or crashed) FlexTool solve run.

    ``succeeded``        — whether the calibrator may consume this run's
                           results.
    ``reason``           — human-readable justification for the verdict.
    ``exit_code``        — the subprocess exit code, if the caller supplied
                           it (a weak input only).
    ``per_solve``        — per-required-output evidence (:class:`OutputCheck`
                           list).  FlexTool exposes no on-disk per-sub-solve
                           optimality status, so this is per-output, not
                           per-LP-subsolve.
    ``outputs_complete`` — whether every required output was present,
                           non-empty and (if checked) fresh.
    """

    succeeded: bool
    reason: str
    exit_code: int | None
    per_solve: list[OutputCheck] = field(default_factory=list)
    outputs_complete: bool = False


def _parquet_num_rows(path: Path) -> int | None:
    """Row count of a parquet file via footer metadata, or ``None``.

    Reads only the parquet footer (no column data), so it is cheap even for
    large tables.  A missing/truncated/corrupt file (e.g. a partial write
    from an interrupted run) returns ``None`` — the caller treats that as an
    incomplete output, i.e. a failure signal, not a crash.
    """
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(str(path)).metadata.num_rows)
    except Exception as exc:  # noqa: BLE001 - any read error ⇒ "not usable"
        logger.debug("Could not read parquet row count for %s: %s", path, exc)
        return None


def _check_output(
    output_dir: Path, filename: str, started_epoch: float | None,
) -> OutputCheck:
    """Assess a single required output file inside *output_dir*."""
    path = output_dir / filename
    if not path.is_file():
        return OutputCheck(
            filename=filename,
            present=False,
            num_rows=None,
            fresh=False,
            detail="missing",
        )

    num_rows = _parquet_num_rows(path)
    if num_rows is None:
        return OutputCheck(
            filename=filename,
            present=True,
            num_rows=None,
            fresh=False,
            detail="present but unreadable/corrupt as parquet",
        )
    if num_rows == 0:
        # An empty table fails on the row count regardless of freshness;
        # report ``fresh`` honestly (mtime vs started_at) rather than
        # conflating "empty" with "stale".
        empty_fresh = True
        if started_epoch is not None:
            try:
                empty_fresh = path.stat().st_mtime + 1.0 >= started_epoch
            except OSError:
                empty_fresh = False
        return OutputCheck(
            filename=filename,
            present=True,
            num_rows=0,
            fresh=empty_fresh,
            detail="present but empty (0 rows)",
        )

    fresh = True
    detail = "present, non-empty"
    if started_epoch is not None:
        try:
            mtime = path.stat().st_mtime
        except OSError as exc:
            fresh = False
            detail = f"present, non-empty, but mtime unavailable ({exc})"
        else:
            # 1s slack absorbs coarse filesystem mtime granularity so a file
            # written in the same second the subprocess launched is not
            # wrongly judged stale.
            if mtime + 1.0 < started_epoch:
                fresh = False
                detail = (
                    "present, non-empty, but STALE "
                    "(older than this run's start — left over from a "
                    "previous run)"
                )
            else:
                detail = "present, non-empty, fresh"

    return OutputCheck(
        filename=filename,
        present=True,
        num_rows=num_rows,
        fresh=fresh,
        detail=detail,
    )


def assess_solve(
    output_dir: Path | str,
    *,
    exit_code: int | None = None,
    required_outputs: Sequence[str] | None = None,
    started_at: float | int | datetime | None = None,
) -> SolveOutcome:
    """Decide whether a FlexTool solve run succeeded, from its outputs.

    Parameters
    ----------
    output_dir:
        The directory that directly holds the run's ``*.parquet`` result
        files — i.e. ``<output_location>/output_parquet/<subdir>/``.
    exit_code:
        The subprocess exit code, if known.  A *weak* input: it is a
        warning that can be overridden (see the success rule below), never
        the sole determinant.  ``None`` means "not supplied".
    required_outputs:
        Output keys or ``*.parquet`` filenames that must be present and
        non-empty for the run to count as successful.  Defaults to the
        calibrator's robust success gate, ``node_slack_up_d_e``
        (:func:`default_required_outputs`); pass more if a caller wants
        stricter completeness.
    started_at:
        POSIX timestamp (recommended: ``time.time()``) / :class:`datetime`
        of when the subprocess was launched.  A required output older than
        this is treated as a stale leftover (not this run's product) and
        fails completeness.  **Required to make the nonzero-exit override
        safe** — see the success rule.  C1 must always pass it.

    Returns
    -------
    SolveOutcome
        The verdict, its reason, the exit code echoed back, the per-output
        evidence, and the ``outputs_complete`` flag.

    Success rule
    ------------
    Let ``complete`` = every required output present, a readable parquet
    with ≥1 row, and (if ``started_at`` given) fresh.

    * ``not complete``                   → **failed**.
    * ``complete`` and exit 0 / ``None`` → **succeeded** (``started_at``
      optional: a successful run empties + rewrites the dir, so files are
      fresh by construction).
    * ``complete`` and nonzero exit and ``started_at`` given and fresh
      → **succeeded**; the nonzero exit is *overridden* (post-solve writer
      crash with complete, fresh results).
    * ``complete`` and nonzero exit and NO ``started_at``
      → **failed**; freshness is unverifiable, so a genuine failure that
      left a prior run's outputs in place cannot be ruled out — the
      override is refused.
    """
    out_dir = Path(output_dir)
    started_epoch = _as_epoch(started_at)
    required = _normalise_required(required_outputs)

    checks = [_check_output(out_dir, fn, started_epoch) for fn in required]
    outputs_complete = bool(checks) and all(c.ok for c in checks)

    if not out_dir.is_dir():
        return SolveOutcome(
            succeeded=False,
            reason=(
                f"output directory {out_dir} does not exist; the solve wrote "
                "no results (a genuinely failed / infeasible / unaccepted "
                "solve skips output writing entirely)."
            ),
            exit_code=exit_code,
            per_solve=checks,
            outputs_complete=False,
        )

    if not outputs_complete:
        bad = [c for c in checks if not c.ok]
        detail = "; ".join(f"{c.filename}: {c.detail}" for c in bad)
        return SolveOutcome(
            succeeded=False,
            reason=(
                "required output(s) missing, empty or stale — the solve did "
                f"not produce usable results [{detail}]. A genuinely failed "
                "sub-solve is surfaced this way: FlexTool skips output "
                "writing on a non-accepted cascade, so absent outputs ARE "
                "the failure signal."
            ),
            exit_code=exit_code,
            per_solve=checks,
            outputs_complete=False,
        )

    # Every required output is present, non-empty and (subject to
    # started_at) fresh.  Because FlexTool writes outputs only for an
    # accepted cascade, this state means no failed/unaccepted sub-solve is
    # detectable.
    #
    # Exit 0 / None: succeed.  On success write_outputs empties then
    # rewrites the parquet dir, so the files are this run's product by
    # construction — no stale-masking hole, and started_at is optional here.
    if exit_code in (None, 0):
        fresh_note = (
            " (verified fresh against this run's start time)"
            if started_epoch is not None
            else ""
        )
        return SolveOutcome(
            succeeded=True,
            reason=(
                "all required outputs present and non-empty"
                f"{fresh_note}; "
                + ("exit code 0." if exit_code == 0 else "no exit code "
                   "supplied.")
            ),
            exit_code=exit_code,
            per_solve=checks,
            outputs_complete=True,
        )

    # Nonzero exit but complete outputs.  This is EITHER the known
    # post-solve writer-crash case (the cascade solved and wrote complete
    # results; the nonzero exit is a writer failure) OR a genuinely failed
    # solve that skipped write_outputs — which does NOT empty the parquet
    # dir — leaving a PRIOR iteration's complete outputs lingering.  The
    # only thing that tells these apart is freshness.  So the override to
    # success is allowed ONLY when started_at was supplied AND every output
    # verified fresh; otherwise we must NOT override.
    if started_epoch is None:
        return SolveOutcome(
            succeeded=False,
            reason=(
                f"nonzero exit code {exit_code} and freshness is unverifiable "
                "(no started_at supplied) — the required outputs are complete "
                "but we CANNOT distinguish a post-solve writer crash (which "
                "leaves this run's fresh results) from a genuinely failed "
                "solve that skipped output writing and left a PRIOR run's "
                "outputs in place. Pass started_at (the subprocess launch "
                "time) so the override can be made safely."
            ),
            exit_code=exit_code,
            per_solve=checks,
            outputs_complete=True,
        )

    # started_at supplied and all outputs are fresh → safe to override the
    # nonzero exit to success (the post-solve writer-crash case, e.g. the
    # PLEXOS→FlexTool writer's "Shared-alternative write failed" KeyError,
    # which lives outside this engine).
    return SolveOutcome(
        succeeded=True,
        reason=(
            "all required outputs present, non-empty and verified fresh "
            f"against this run's start time; the nonzero exit code {exit_code} "
            "is OVERRIDDEN to success — the cascade solved and wrote complete, "
            "fresh results, and the nonzero exit reflects a post-solve writer "
            "failure (e.g. the model-specific 'Shared-alternative write "
            "failed' KeyError in the PLEXOS→FlexTool writer, which is outside "
            "the solve engine), not a solve failure."
        ),
        exit_code=exit_code,
        per_solve=checks,
        outputs_complete=True,
    )


__all__ = [
    "OutputCheck",
    "SolveOutcome",
    "assess_solve",
    "default_required_outputs",
]
