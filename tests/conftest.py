"""pytest fixtures for FlexTool scenario tests.

CWD notes (Task 1 findings)
---------------------------
FlexToolRunner writes several files relative to CWD:
  solve_data/          — auto-created by __init__; holds intermediate
                         per-solve CSVs and timings.csv (the unified
                         phase-timing log; replaces the legacy
                         solve_progress.csv files)
  output_raw/          — created by glpsol Phase 3; holds raw solver output CSVs
  HiGHS.log            — written by HiGHS

Intermediate solver files go to root_dir (not CWD):
  flextool.mps, flextool.sol, glpsol_solution.txt

The workdir fixture uses monkeypatch.chdir() to isolate each test's CWD.
root_dir is set to workdir so intermediate files also stay in the temp dir.

bin_dir must contain highs.opt (+ solver binaries).
A session-scoped test_bin_dir fixture creates a temp dir with the test
highs.opt and symlinks to the actual solver binaries.
"""
from __future__ import annotations

import os
# Skip the per-cascade solver license probe in tests.  The probe calls
# xpress.problem() (among others), which emits a FICO Community
# LicenseWarning; the production solve uses HiGHS anyway.  Must be set
# before any flextool / polar_high import.
os.environ.setdefault("FLEXTOOL_SKIP_SOLVER_PROBE", "1")

import importlib.util
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

TEST_DIR = Path(__file__).parent
FIXTURES_DIR = TEST_DIR / "fixtures"

# ---------------------------------------------------------------------------
# Re-export the engine_polars constraint conftest as a top-level plugin so
# the engine_polars/objective conftest can reuse its fixtures.  pytest 8.x
# forbids ``pytest_plugins`` in non-top-level conftests, so the registration
# lives here.  The constraint conftest itself is loaded by file path because
# ``tests/`` is not a package.  pytest also tries to import the plugin by
# the same name during its plugin-discovery phase; pre-registering in
# ``sys.modules`` short-circuits that lookup.  (The "Module already imported
# so cannot be rewritten" warning is benign — the constraints conftest is
# fixtures, not assertion-heavy test code.)
# ---------------------------------------------------------------------------
_CONSTRAINTS_CONFTEST = (
    TEST_DIR / "engine_polars" / "constraints" / "conftest.py"
)
if _CONSTRAINTS_CONFTEST.exists() and (
    "_engine_polars_constraints_conftest" not in sys.modules
):
    _spec = importlib.util.spec_from_file_location(
        "_engine_polars_constraints_conftest", _CONSTRAINTS_CONFTEST,
    )
    _constraints_module = importlib.util.module_from_spec(_spec)
    sys.modules["_engine_polars_constraints_conftest"] = _constraints_module
    _spec.loader.exec_module(_constraints_module)

pytest_plugins = ["_engine_polars_constraints_conftest"]
REPO_ROOT = TEST_DIR.parent

# Make db_utils importable from this directory
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from db_utils import json_to_db, round_for_comparison  # noqa: E402

__all__ = ["round_for_comparison"]  # re-export for test modules


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--regenerate",
        metavar="SCENARIO",
        default=None,
        help=(
            "Regenerate expected CSVs for the given scenario name instead of comparing. "
            "Example: pytest test/ --regenerate coal"
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    # Materialize canonical example/template SQLites (templates/examples.sqlite,
    # how to example databases/*.sqlite, ...) from their JSON sources so tests
    # that reference these paths directly work on a fresh clone.  Idempotent —
    # skips files already present in the working tree.
    from flextool.update_flextool.canonical_databases import materialize
    materialize(overwrite=False)

    config.addinivalue_line(
        "markers", "smoke: fast Layer-1 scenarios for the per-commit gate"
    )
    config.addinivalue_line(
        "markers", "solver: tests that invoke a real solver (glpsol/HiGHS)"
    )
    config.addinivalue_line(
        "markers", "slow: tests that take more than ~30 seconds"
    )
    config.addinivalue_line(
        "markers", "decomposition: Tier 8 obj-decomposition parity tests"
    )
    config.addinivalue_line(
        "markers", "perturbation: Tier 6 single-multiplier perturbation tests"
    )
    config.addinivalue_line(
        "markers", "emission: Tier 7 MPS row-count emission tests"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """When --regenerate is given, run only the matching scenario test."""
    regenerate = config.getoption("--regenerate", default=None)
    if not regenerate:
        return
    selected, deselected = [], []
    for item in items:
        params = getattr(item, "callspec", None) and item.callspec.params
        if params and params.get("scenario") == regenerate:
            selected.append(item)
        else:
            deselected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected


@pytest.fixture(scope="session")
def test_db_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Import JSON fixture → fresh SQLite DB once per test session."""
    db_path = tmp_path_factory.mktemp("db") / "tests.sqlite"
    return json_to_db(FIXTURES_DIR / "tests.json", db_path)


@pytest.fixture(scope="session")
def stochastic_db_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Stochastic-feature scenarios DB.

    Built from ``tests/fixtures/stochastics.json`` (a JSON dump of the
    user-facing ``how to example databases/stochastics.sqlite``).  The
    JSON was exported at FlexTool DB v25 — this fixture migrates to the
    current ``FLEXTOOL_DB_VERSION`` so scenarios resolve against the
    same schema as the main test DB.
    """
    from flextool.update_flextool.db_migration import migrate_database

    db_path = tmp_path_factory.mktemp("db_stoch") / "stochastics.sqlite"
    url = json_to_db(FIXTURES_DIR / "stochastics.json", db_path)
    migrate_database(url)
    return url


# Map scenarios.yaml ``db_fixture`` values to fixture names — kept in
# conftest so adding a new fixture requires touching exactly two
# places: the fixture definition above and this map.
_DB_FIXTURE_NAMES: dict[str, str] = {
    "main": "test_db_url",
    "stochastic": "stochastic_db_url",
}


@pytest.fixture
def scenario_db_url(request: pytest.FixtureRequest, db_fixture: str) -> str:
    """Resolve the per-scenario DB url according to ``db_fixture`` field
    in scenarios.yaml.  Defaults to ``main`` (== ``test_db_url``)."""
    fixture_name = _DB_FIXTURE_NAMES.get(db_fixture)
    if fixture_name is None:
        raise ValueError(
            f"Unknown db_fixture {db_fixture!r}; "
            f"valid: {sorted(_DB_FIXTURE_NAMES)}"
        )
    return request.getfixturevalue(fixture_name)


@pytest.fixture(scope="session")
def test_bin_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Temp bin dir with test highs.opt and symlinked solver binaries.

    Using test/highs.opt instead of bin/highs.opt ensures deterministic,
    tight-precision solves for golden file comparisons.
    """
    repo_bin = REPO_ROOT / "bin"
    bin_dir = tmp_path_factory.mktemp("bin")

    # Select platform-appropriate glpsol binary
    import platform
    if sys.platform == "darwin" and platform.machine() == "arm64":
        glpsol_candidates = ["glpsol_macos15_arm64", "glpsol"]
    elif sys.platform.startswith("win"):
        glpsol_candidates = ["glpsol.exe"]
    else:
        glpsol_candidates = ["glpsol"]

    for candidate in glpsol_candidates:
        src = repo_bin / candidate
        if src.exists():
            (bin_dir / "glpsol").symlink_to(src)
            break

    for binary in ["highs", "highs.exe", "glpsol.exe"]:
        src = repo_bin / binary
        if src.exists() and not (bin_dir / binary).exists():
            (bin_dir / binary).symlink_to(src)

    shutil.copy(TEST_DIR / "highs.opt", bin_dir / "highs.opt")
    return bin_dir


@pytest.fixture(autouse=True)
def _reset_global_axis_enums() -> None:
    """Reset the cascade-wide axis_enums ContextVar before every test.

    ``flextool.engine_polars.input.load_flextool`` sets the global
    ``axis_enums`` ContextVar on success (input.py finally block
    ~line 4216).  The ContextVar persists into the next test in the
    same pytest worker, pinning Enum dtypes that may differ from the
    next test's fixture vocabulary — surfacing as "enum on left does
    not match enum on right" SchemaError in joins, OR (more subtly)
    as test-order-dependent flakes where a downstream test inherits
    a polluted vocabulary and silently produces wrong-shape output.

    The ``tests/engine_polars/conftest.py`` already has a local
    ``autouse=True`` version of this fixture covering tests in that
    subtree.  SCEN-3 (``test_examples_scenario_solves[test_a_lot]``
    passes alone, fails in full sweep) hypothesised that polluters
    from OUTSIDE ``tests/engine_polars/`` (e.g. ``test_scenarios.py``,
    ``test_xlsx_workflow.py``, ``model/test_examples_e2e.py``) leak
    axis_enums state into subsequent tests because the local fixture
    didn't cover them.  Lifting the reset to the top-level conftest
    closes that gap for ALL tests, not just the engine_polars
    subtree.

    Cheap (a single ContextVar assignment) and harmless to run
    between every test.
    """
    from flextool.engine_polars._axis_enums import set_global_axis_enums
    set_global_axis_enums(None)
    yield
    set_global_axis_enums(None)


@pytest.fixture(autouse=True)
def _reset_flextool_module_caches() -> None:
    """Clear FlexTool module-level caches before every test.

    Agent 15 (LP-scaling): ``flextool.flextoolrunner.scaling`` keeps a
    module-level ``_scale_cache`` keyed by solve name.  Scenario tests
    frequently share solve names (e.g. multiple scenarios with
    ``y2020_5week``) but feed the analyser different input CSVs.  Left
    to its own devices the cache hands the second scenario the first
    scenario's ``scale_the_objective`` and ``use_row_scaling``
    recommendation, which then writes a wrong-by-10x objective scalar
    into ``solve_data/scale_the_objective.csv`` and cascades into an
    MPS with the wrong penalty coefficients.  Clearing per-test makes
    the full-suite ordering irrelevant.

    Cheap: the cache is a plain dict, and scenarios that don't share
    solve names were never caching-sensitive anyway.

    Δ.41 (test-order flake): ``flextool.engine_polars.scaling`` has its
    own ``_scale_cache`` (the polars/in-memory port that the engine
    actually consults post-cascade-refactor).  The legacy
    ``flextoolrunner.scaling._scale_cache`` is still cleared above to
    cover any code paths that still touch it, but the engine-side cache
    is what fed ``capacity_margin``'s ``user_bound_scale=-19`` into the
    subsequent ``coal_co2_limit`` solve (both share the
    ``dispatch_y2020_5week`` solve name → same cache key → same
    recommended scalar applied to a problem with a different LP range,
    yielding alt-optima column residuals against the golden).
    """
    from flextool.flextoolrunner import scaling as _scaling
    from flextool.engine_polars import scaling as _engine_scaling
    _scaling.clear_cache()
    _engine_scaling.clear_cache()
    yield
    _scaling.clear_cache()
    _engine_scaling.clear_cache()


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Per-test isolated working directory.

    FlexToolRunner writes to CWD (solve_data/, output_raw/, HiGHS.log).
    monkeypatch.chdir ensures each test gets a clean CWD and leaves no
    files in the repo root.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


