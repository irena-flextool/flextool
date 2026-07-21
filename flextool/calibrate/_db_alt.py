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

from spinedb_api import DatabaseMapping, import_data
from spinedb_api.exception import NothingToCommit

# Suffix appended to a scenario name to form its calibration alternative.
_CALIB_ALT_SUFFIX = "_adeq_calib"


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
    url: str, scenario: str, per_node_adder: dict[str, float],
) -> None:
    """Write (or update) the calibration alternative for *scenario*.

    For every ``node -> adder`` in *per_node_adder*, set both
    ``energy_margin_method = inflow_adder`` and the float
    ``energy_margin_adder`` on that node under the ``<scenario>_adeq_calib``
    alternative, and append that alternative to *scenario*'s stack at the
    top rank (higher rank wins; the existing stack is left intact).

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
        pvs.append(("node", node, "energy_margin_adder", float(adder), alt))

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
