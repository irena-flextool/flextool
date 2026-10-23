"""The HiGHS options carried by the data (``FlexData.solver_options``) reach the built Problem.

``build_flextool`` builds model content only; the solve layer stores ``data.solver_options`` on the
Problem right after the build (``_solver_dispatch.apply_data_solver_options``).  These tests pin that
for every production builder that previously received the options from inside ``build_flextool``:
the warm-start builder, the Benders master and the default Benders region builder.

The orchestration cold/warm paths also apply them, but replace them on the next line with the
resolved ``highs_options`` (``set_solver_options`` replaces, it does not merge), as before.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from polar_high import Problem

from flex_toy_existing_fixed_cost import data as toy_data

from flextool.engine.solve import _benders
from flextool.engine.solve._solver_dispatch import apply_data_solver_options

OPTS = {"random_seed": 7, "primal_feasibility_tolerance": 1e-8}


def _toy_with_options():
    d = toy_data()
    d.solver_options = dict(OPTS)
    return d


def test_apply_data_solver_options_stores_options() -> None:
    pb = Problem()
    apply_data_solver_options(pb, SimpleNamespace(solver_options=dict(OPTS)))
    assert pb._solver_options == OPTS


@pytest.mark.parametrize("value", [None, {}])
def test_apply_data_solver_options_keeps_existing_when_absent(value) -> None:
    pb = Problem()
    pb.set_solver_options({"threads": 1})
    apply_data_solver_options(pb, SimpleNamespace(solver_options=value))
    apply_data_solver_options(pb, SimpleNamespace())
    assert pb._solver_options == {"threads": 1}


def test_build_flextool_does_not_set_solver_options() -> None:
    from flextool.engine import build_flextool

    pb = Problem()
    build_flextool(pb, _toy_with_options())
    assert pb._solver_options is None


def test_warm_problem_builder_applies_data_options() -> None:
    from flextool.engine.solve._warm import _build_warm_problem

    warm = _build_warm_problem(_toy_with_options())
    assert warm.problem._solver_options == OPTS
    assert warm.solve().optimal


def test_warm_problem_builder_explicit_options_win() -> None:
    from flextool.engine.solve._warm import _build_warm_problem

    warm = _build_warm_problem(_toy_with_options(), solver_options={"threads": 2})
    assert warm.problem._solver_options == {"threads": 2}


def test_benders_region_builder_applies_data_options(monkeypatch) -> None:
    captured: dict = {}

    def fake_inner(data, regions, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(_benders, "_is_genuinely_stochastic", lambda data: False)
    monkeypatch.setattr(_benders, "_solve_benders_inner", fake_inner)
    d = _toy_with_options()
    _benders.solve_benders(d, ["r"])

    pb = Problem()
    captured["build_problem"](pb, d)
    assert pb._solver_options == OPTS


def test_benders_master_applies_data_options(monkeypatch) -> None:
    class _Stop(Exception):
        pass

    built: dict = {}

    def fake_warm_problem(m):
        built["m"] = m
        raise _Stop

    monkeypatch.setattr(_benders._region_filter, "master_network_data", lambda data, *a, **k: data)
    monkeypatch.setattr(_benders, "WarmProblem", fake_warm_problem)
    master = object.__new__(_benders._BendersMaster)
    master.regions = ["r"]
    master._region_membership = None
    master._master_hosted_nodes = None
    master._obj_scale = 1.0
    with pytest.raises(_Stop):
        master._build_flextool_master(_toy_with_options())
    assert built["m"]._solver_options == OPTS
