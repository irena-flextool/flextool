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

solver_config_dir must contain highs.opt (+ solver binaries).
A session-scoped test_solver_config_dir fixture creates a temp dir with the test
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
from urllib.parse import urlparse

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
    # skips files already present in the working tree.  Tests that need a
    # guaranteed-current SQLite (e.g. after a ``FLEXTOOL_DB_VERSION`` bump
    # invalidates the on-disk copy) should build their own from the JSON
    # source under ``tmp_path`` rather than mutating these user-facing files.
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
def schema_db_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Build a fresh SQLite from ``schemas/spinedb_schema.json`` once per
    session.

    The schema JSON is the source of truth for the parameter / entity-class
    / value-list contract.  Tests that need to enumerate "every parameter
    FlexTool declares" must read it from a DB built off this JSON rather
    than a checked-in template (``templates/input_data_template.sqlite``),
    which lags the schema between regenerations and silently under-covers
    (the v56 ``is_enabled`` add was missing from the on-disk template while
    present in the schema).  Building here also exercises the real
    schema→DB ``import_data`` path.  Session-scoped: the ~50 ms
    ``initialize_database`` call runs once for the whole suite.
    """
    from flextool.update_flextool.initialize_database import initialize_database

    schema_json = (
        TEST_DIR.parent / "flextool" / "schemas" / "spinedb_schema.json"
    )
    db_path = tmp_path_factory.mktemp("schema_db") / "schema.sqlite"
    initialize_database(str(schema_json), str(db_path))
    return f"sqlite:///{db_path}"


@pytest.fixture(scope="session")
def test_db_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Import JSON fixture → fresh SQLite DB once per test session.

    The ``migrate_database`` call is defensive — idempotent at the
    current schema and only does work if the committed JSON lags the
    code's ``FLEXTOOL_DB_VERSION``.  Without it, a code-side rename
    (e.g. commit 109bd689) goes silent until someone reruns
    ``test_fixtures migrate-all``.  Mirrors the pattern used by all
    other DB fixtures below.
    """
    from flextool.update_flextool.db_migration import migrate_database

    db_path = tmp_path_factory.mktemp("db") / "tests.sqlite"
    url = json_to_db(FIXTURES_DIR / "tests.json", db_path)
    migrate_database(url)
    return url


@pytest.fixture(scope="session")
def stochastic_db_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Stochastic-feature scenarios DB.

    Built from ``tests/fixtures/stochastics.json`` (a JSON dump of the
    user-facing ``how to example databases/stochastics.sqlite``).  The
    committed JSON is kept current with ``FLEXTOOL_DB_VERSION`` via
    ``python -m flextool.update_flextool.test_fixtures migrate-all``
    (see CONTRIBUTING.md), so the ``migrate_database`` call below is
    defensive — it is idempotent at the current schema and only does
    work if someone bumps the schema without re-running migrate-all.
    Kept in place to future-proof the fixture against that window.
    """
    from flextool.update_flextool.db_migration import migrate_database

    db_path = tmp_path_factory.mktemp("db_stoch") / "stochastics.sqlite"
    url = json_to_db(FIXTURES_DIR / "stochastics.json", db_path)
    migrate_database(url)
    return url


@pytest.fixture(scope="session")
def lh2_db_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Three-region LH2 fixture DB.

    Built from ``tests/fixtures/lh2_three_region.json``.  Like
    :func:`stochastic_db_url`, the migration is defensive — idempotent
    at the current schema, only doing work if the schema is bumped
    without re-running ``migrate-all``.
    """
    from flextool.update_flextool.db_migration import migrate_database

    db_path = tmp_path_factory.mktemp("db_lh2") / "lh2.sqlite"
    url = json_to_db(FIXTURES_DIR / "lh2_three_region.json", db_path)
    migrate_database(url)
    return url


@pytest.fixture(scope="session")
def lh2_trade_invest_db_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Benders Phase-0 greenfield-trade LH2 fixture DB.

    Built from ``tests/fixtures/lh2_three_region_trade_invest.json`` — a
    2-day / 48h sibling of the LH2 fixture whose pipes are greenfield
    investable, exercising the greenfield-trade decomposition bug.
    Defensive migration as in :func:`stochastic_db_url`.
    """
    from flextool.update_flextool.db_migration import migrate_database

    db_path = tmp_path_factory.mktemp("db_lh2ti") / "lh2_trade_invest.sqlite"
    url = json_to_db(
        FIXTURES_DIR / "lh2_three_region_trade_invest.json", db_path
    )
    migrate_database(url)
    return url


@pytest.fixture(scope="session")
def h2_trade_parity_db_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    """``h2_trade_parity`` fixture DB.

    Built from ``tests/fixtures/h2_trade_parity.json``.  Defensive
    migration as in :func:`stochastic_db_url`.
    """
    from flextool.update_flextool.db_migration import migrate_database

    db_path = tmp_path_factory.mktemp("db_h2tp") / "h2_trade_parity.sqlite"
    url = json_to_db(FIXTURES_DIR / "h2_trade_parity.json", db_path)
    migrate_database(url)
    return url


@pytest.fixture(scope="session")
def stochastics_pbt_inflow_db_url(
    tmp_path_factory: pytest.TempPathFactory,
) -> str:
    """``stochastics_pbt_inflow`` fixture DB.

    Built from ``tests/fixtures/stochastics_pbt_inflow.json``.
    Defensive migration as in :func:`stochastic_db_url`.
    """
    from flextool.update_flextool.db_migration import migrate_database

    db_path = (
        tmp_path_factory.mktemp("db_spbti") / "stochastics_pbt_inflow.sqlite"
    )
    url = json_to_db(FIXTURES_DIR / "stochastics_pbt_inflow.json", db_path)
    migrate_database(url)
    return url


@pytest.fixture(scope="session")
def branch2_parent_period_db_url(
    tmp_path_factory: pytest.TempPathFactory,
) -> str:
    """``branch2_parent_period`` fixture DB.

    Built from ``tests/fixtures/branch2_parent_period.json``.  Defensive
    migration as in :func:`stochastic_db_url`.
    """
    from flextool.update_flextool.db_migration import migrate_database

    db_path = (
        tmp_path_factory.mktemp("db_b2pp") / "branch2_parent_period.sqlite"
    )
    url = json_to_db(FIXTURES_DIR / "branch2_parent_period.json", db_path)
    migrate_database(url)
    return url


@pytest.fixture(scope="session")
def case14_db_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    """PGLib IEEE case-14 DC OPF fixture DB.

    Built from ``tests/fixtures/case14_dc_power_flow.json``, which is a
    one-shot export of the MATPOWER-derived test DB used by
    ``test_flex_dc_power_flow.py``.  Defensive migration as in
    :func:`stochastic_db_url`.
    """
    from flextool.update_flextool.db_migration import migrate_database

    db_path = tmp_path_factory.mktemp("db_case14") / "case14_dc_power_flow.sqlite"
    url = json_to_db(FIXTURES_DIR / "case14_dc_power_flow.json", db_path)
    migrate_database(url)
    return url


# Map scenarios.yaml ``db_fixture`` values to fixture names — kept in
# conftest so adding a new fixture requires touching exactly two
# places: the fixture definition above and this map.
_DB_FIXTURE_NAMES: dict[str, str] = {
    "main": "test_db_url",
    "stochastic": "stochastic_db_url",
    "lh2": "lh2_db_url",
    "lh2_trade_invest": "lh2_trade_invest_db_url",
    "h2_trade_parity": "h2_trade_parity_db_url",
    "stochastics_pbt_inflow": "stochastics_pbt_inflow_db_url",
    "branch2_parent_period": "branch2_parent_period_db_url",
    "case14": "case14_db_url",
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
def test_solver_config_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Temp dir containing the deterministic test highs.opt.

    Using tests/highs.opt instead of the project-root solver_config/highs.opt
    ensures tight-precision, deterministic solves for golden file comparisons.
    """
    solver_config_dir = tmp_path_factory.mktemp("solver_config")
    shutil.copy(TEST_DIR / "highs.opt", solver_config_dir / "highs.opt")
    return solver_config_dir


@pytest.fixture(scope="session")
def scenario_workdir(
    test_db_url,
    stochastic_db_url,
    lh2_db_url,
    lh2_trade_invest_db_url,
    h2_trade_parity_db_url,
    stochastics_pbt_inflow_db_url,
    branch2_parent_period_db_url,
    case14_db_url,
    test_solver_config_dir,
    tmp_path_factory,
):
    """Factory: build a fully-preprocessed work folder for a scenario.

    Returns a callable ``factory(scenario_name, db_fixture="main")`` that
    runs the full cascade with ``csv_dump=True`` and then snapshots the
    last sub-solve's :class:`FlexDataProvider` to disk, materialising
    every ``_emit_*`` output as a CSV under ``<wf>/input/*`` and
    ``<wf>/solve_data/*``.  Results are cached per ``(scenario,
    db_fixture)`` pair for the lifetime of the test session — calling
    ``factory("base")`` twice returns the same ``Path``.

    This mirrors the canonical CLI ``--csv-dump`` flow
    (``flextool/cli/cmd_run_flextool.py``): the cascade itself runs
    in-memory; ``csv_dump=True`` flushes the in-memory FlexData via
    ``data.dump_csvs`` per sub-solve, and the post-cascade
    ``provider.snapshot_processed_inputs`` materialises the
    parent-qualified Provider key layout.  The combination reproduces
    what the disk-staged generator used to seed
    ``tests/engine_polars/data/`` produced.

    The full cascade (HiGHS solve included) is required because the bulk
    of the per-solve emitters run *inside* the cascade loop and populate
    the live Provider; ``keep_solutions=True`` is required because that
    Provider is otherwise slimmed away by the post-cascade memory-saver.

    The ``db_fixture`` keyword selects the JSON fixture backing the DB.
    Valid values are the keys of :data:`_DB_FIXTURE_NAMES` — currently
    ``"main"`` (tests.json), ``"stochastic"`` (stochastics.json),
    ``"lh2"`` (lh2_three_region.json), ``"h2_trade_parity"``,
    ``"stochastics_pbt_inflow"``,
    ``"branch2_parent_period"``, ``"case14"``.

    Tests use it like:

        def test_foo(scenario_workdir):
            work = scenario_workdir("base")
            data = load_flextool(work)
    """
    from flextool.engine_polars._orchestration import run_chain_from_db

    url_by_fixture = {
        "main": test_db_url,
        "stochastic": stochastic_db_url,
        "lh2": lh2_db_url,
        "lh2_trade_invest": lh2_trade_invest_db_url,
        "h2_trade_parity": h2_trade_parity_db_url,
        "stochastics_pbt_inflow": stochastics_pbt_inflow_db_url,
        "branch2_parent_period": branch2_parent_period_db_url,
        "case14": case14_db_url,
    }

    cache: dict[tuple[str, str], Path] = {}

    def factory(scenario_name: str, db_fixture: str = "main") -> Path:
        key = (scenario_name, db_fixture)
        if key in cache:
            return cache[key]
        try:
            url = url_by_fixture[db_fixture]
        except KeyError:
            raise ValueError(
                f"Unknown db_fixture {db_fixture!r}; valid: "
                f"{sorted(url_by_fixture)}"
            ) from None
        # ``_find_scenario`` (input.py) auto-constructs a SpineDbReader
        # for ``load_flextool`` if and only if the workdir's basename
        # matches ``work_<scenario_name>`` EXACTLY (no numeric suffix)
        # and the workdir contains ``tests.sqlite``.  ``tmp_path_factory.
        # mktemp`` always appends a numeric suffix (``work_base0`` etc.),
        # so build a uniquely-named parent and nest the precisely-named
        # work folder inside.
        parent = tmp_path_factory.mktemp(f"_root_{scenario_name}_{db_fixture}")
        wf = parent / f"work_{scenario_name}"
        wf.mkdir()
        steps = run_chain_from_db(
            input_db_url=url,
            scenario_name=scenario_name,
            work_folder=wf,
            solver_config_dir=test_solver_config_dir,
            csv_dump=True,
            keep_solutions=True,
        )
        # Copy the SQLite the cascade was driven from into the workdir
        # so ``_find_scenario`` can auto-construct a SpineDbReader for
        # the re-solve in tests that call ``load_flextool(wf)``.  The
        # url is ``sqlite:///<absolute-path>``; urlparse returns the
        # leading slash inside ``.path``.
        sqlite_src = urlparse(url).path
        shutil.copy(sqlite_src, wf / "tests.sqlite")
        # Mirror the CLI ``--csv-dump`` post-cascade snapshot: only the
        # last sub-solve's Provider holds the union of every cascade
        # emitter's frames; snapshot it to disk so ``_emit_*`` outputs
        # appear under ``input/`` + ``solve_data/``.
        if steps:
            last_step = next(reversed(list(steps.values())))
            provider = getattr(last_step, "flex_data_provider", None)
            if provider is not None:
                provider.snapshot_processed_inputs(wf)
        cache[key] = wf
        return wf

    return factory


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


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Per-test isolated working directory.

    FlexToolRunner writes to CWD (solve_data/, output_raw/, HiGHS.log).
    monkeypatch.chdir ensures each test gets a clean CWD and leaves no
    files in the repo root.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


