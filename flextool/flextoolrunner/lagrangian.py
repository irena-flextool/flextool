"""Spatial Lagrangian coordinator — RETIRED in Δ.22.

The legacy implementation built per-region work folders, drove
``glpsol --wfreemps`` per region, loaded the resulting MPS into
``highspy`` via :class:`HighsModelHandle`, and ran a damped
sub-gradient outer loop with primal averaging on the coupling
flows.  Every step depended on the GMPL/.mod/.dat infrastructure
that Δ.22 deleted (``flextool/flextool.mod``,
``flextool/flextool_base.dat``, and ``bin/glpsol*``).

The native (engine_polars) port of this scheme already exists in
:mod:`flextool.engine_polars._lagrangian` and is exercised by
``tests/engine_polars/test_lagrangian.py`` — that path is the
forward direction.  The CLI ``--decomposition lagrangian`` entry
point in :mod:`flextool.cli.cmd_run_flextool` still imports
:func:`run_lagrangian` from this module; until the CLI is rewired
onto :func:`flextool.engine_polars._lagrangian.solve_lagrangian`
(deferred dispatch — see ``specs/lagrangian_port_handoff.md``),
calling ``--decomposition lagrangian`` raises
:class:`NotImplementedError` with the retirement banner.

Δ.22 scope was bounded to **delete + skip-mark**, not port; the
user explicitly accepted that the Lagrangian path is broken until
the next dispatch picks the port up.
"""
from __future__ import annotations


_LAGRANGIAN_RETIRED_MESSAGE = (
    "Lagrangian retired in Δ.22; the GMPL coordinator that drove "
    "glpsol per region was deleted with flextool.mod / flextool_base.dat / "
    "bin/glpsol*.  The native port lives in "
    "flextool.engine_polars._lagrangian.solve_lagrangian; rewiring the CLI "
    "``--decomposition lagrangian`` dispatch onto it is a deferred port "
    "(see specs/lagrangian_port_handoff.md)."
)


def run_lagrangian(*args, **kwargs):
    """Stub — the GMPL Lagrangian coordinator was deleted in Δ.22.

    Use :func:`flextool.engine_polars._lagrangian.solve_lagrangian`
    via the engine_polars cascade for the working native port.  The
    CLI ``--decomposition lagrangian`` entry point still imports this
    function; calling it raises :class:`NotImplementedError` with the
    retirement banner so the failure is loud and actionable.
    """
    raise NotImplementedError(_LAGRANGIAN_RETIRED_MESSAGE)


__all__ = ["run_lagrangian"]
