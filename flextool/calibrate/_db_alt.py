"""Idempotent write of the per-scenario adequacy-calibration alternative.

Each calibrator iteration needs the current per-node
``energy_margin_adder`` (and its enabling ``energy_margin_method =
inflow_adder``) applied to the model *without editing the fixture's own
alternatives*.  We do that by writing a dedicated calibration alternative
named ``<scenario>_adeq_calib`` and appending it to the scenario's
alternative stack at the TOP rank, so its values WIN over any baseline the
scenario already sets while leaving that baseline untouched.

The write is idempotent by construction: :func:`spinedb_api.import_data`
defaults to ``on_conflict='merge'`` (an UPDATE in place), so re-writing a
changed adder updates the single existing row rather than creating a
duplicate, and appending an already-present ``(scenario, alt)`` link is a
no-op.  Re-writing an *identical* state leaves nothing to commit, which
:meth:`DatabaseMapping.commit_session` signals by raising
:class:`NothingToCommit` — caught and treated as success.

This module never solves and never touches the network beyond the target
SpineDB.
"""

from __future__ import annotations

from collections import defaultdict

from spinedb_api import DatabaseMapping, import_data
from spinedb_api.exception import NothingToCommit
from spinedb_api.parameter_value import Map

# Suffix appended to a scenario name to form its calibration alternative.
_CALIB_ALT_SUFFIX = "_adeq_calib"


def _adder_map(cells: dict[tuple[str, str], float]) -> Map:
    """Build a 2-D ``(period → time → float)`` Map from per-cell adders.

    *cells* is ``{(period, time): adder}`` (the ``timed`` sizer's output for
    one node).  The nested ``Map`` round-trips through ``import_data`` into the
    ``pdt_energy_margin_adder.csv`` spec the emitter reads, placing each cell's
    value at exactly that invest ``(period, time)``.  Periods and times are
    emitted in sorted order for a deterministic, idempotent write.
    """
    by_period: dict[str, dict[str, float]] = defaultdict(dict)
    for (period, time), value in cells.items():
        by_period[str(period)][str(time)] = float(value)
    periods = sorted(by_period)
    inner_maps = []
    for period in periods:
        times = sorted(by_period[period])
        inner_maps.append(
            Map(times, [by_period[period][t] for t in times], index_name="time")
        )
    return Map(periods, inner_maps, index_name="period")


def calib_alt_name(scenario: str) -> str:
    """Return the calibration alternative name for *scenario*.

    A pure naming helper (``f"{scenario}{_CALIB_ALT_SUFFIX}"``) so callers
    and tests agree on the alternative the calibrator writes into without
    duplicating the string literal.
    """
    return f"{scenario}{_CALIB_ALT_SUFFIX}"


def _normalise_url(url: str) -> str:
    """Accept either a bare filesystem path or a full SQLAlchemy URL.

    A bare path (no ``"://"`` scheme) is promoted to a ``sqlite:///`` URL;
    anything already carrying a scheme is passed through verbatim.
    """
    return url if "://" in url else f"sqlite:///{url}"


def write_calib_alt(
    url: str,
    scenario: str,
    per_node_adder: "dict[str, float | dict[tuple[str, str], float]]",
) -> None:
    """Write (or update) the calibration alternative for *scenario*.

    For every ``node -> adder`` in *per_node_adder*, set both
    ``energy_margin_method = inflow_adder`` and ``energy_margin_adder`` on that
    node under the ``<scenario>_adeq_calib`` alternative, and append that
    alternative to *scenario*'s stack at the top rank (higher rank wins; the
    existing stack is left intact).

    The adder value is EITHER a scalar float (the ``uniform`` sizer — a
    constant per-timestep margin) OR a ``{(period, time): float}`` map (the
    ``timed`` sizer — per-cell margin), written as a 2-D
    ``period → time → float`` :class:`spinedb_api.parameter_value.Map` that
    the emitter ingests as ``pdt_energy_margin_adder.csv``.  Both share the
    idempotent-overwrite path.

    Idempotent: re-writing a changed adder UPDATEs the single row in place
    (``import_data`` merges on conflict); re-writing an identical state
    commits nothing (:class:`NothingToCommit` is swallowed).  An empty
    *per_node_adder* still materialises the alternative and its scenario
    link — so a baseline (k=0) iteration is solved *through* the calibration
    alternative from the start, keeping the alternative stack constant
    across every iteration.

    Parameters
    ----------
    url:
        Target SpineDB — a bare path (promoted to ``sqlite:///``) or a full
        SQLAlchemy URL.
    scenario:
        The model scenario whose stack the calibration alternative joins.
    per_node_adder:
        ``{node_name: adder_MWh}``.  May be empty (baseline iteration).
    """
    alt = calib_alt_name(scenario)

    pvs: list[tuple] = []
    for node, adder in per_node_adder.items():
        pvs.append(("node", node, "energy_margin_method", "inflow_adder", alt))
        value = _adder_map(adder) if isinstance(adder, dict) else float(adder)
        pvs.append(("node", node, "energy_margin_adder", value, alt))

    with DatabaseMapping(_normalise_url(url)) as db:
        _count, errors = import_data(
            db,
            alternatives=[alt],
            scenario_alternatives=[(scenario, alt)],
            parameter_values=pvs,
        )
        assert not errors, errors
        try:
            db.commit_session("adeq_calib iteration")
        except NothingToCommit:
            # Re-writing an identical state changes nothing to persist.
            pass


__all__ = ["calib_alt_name", "write_calib_alt"]
