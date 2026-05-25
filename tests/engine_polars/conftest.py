"""Pytest sys.path setup for engine_polars integration tests.

Layers:
  * polar-high is installed as a real dependency (``pip install -e``).
  * ``flextool.engine_polars`` is in the flextool-engine repo and importable
    via the repo's package layout (no path injection needed).
  * ``tests/engine_polars/fixtures/`` — synthetic flextool-flavoured fixtures
    that import flextool, kept on sys.path so test files can do
    ``from flex_toy_<feature> import ...``.

Also resets the global axis-enums :class:`contextvars.ContextVar` between
tests so a prior test's ``load_flextool`` doesn't pollute the next test's
joins with stale Enum vocabularies (cross-test ContextVar leak).
"""

import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse

import pytest

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
if str(FIXTURES) not in sys.path:
    sys.path.insert(0, str(FIXTURES))
# Under pytest --import-mode=importlib, the directory of a test file is not
# automatically prepended to sys.path.  We host ``_golden.py`` and a couple
# of other private helpers next to the test files, and they're imported by
# bare name (``from _golden import ...``).  Make those resolvable.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DATA_DIR = HERE / "data"


# Map ``db_fixture`` keyword values to the session-scoped fixture
# providing the DB URL.  Adding a new fixture requires:
#   1. Define a session-scoped ``<name>_db_url`` fixture in
#      ``tests/conftest.py`` (parallel to ``stochastic_db_url``).
#   2. Add the entry below.
#   3. Add the fixture as an arg to ``scenario_workdir`` so pytest
#      instantiates it.
_DB_FIXTURE_URL_NAMES: dict[str, str] = {
    "main": "test_db_url",
    "stochastic": "stochastic_db_url",
    "lh2": "lh2_db_url",
    "h2_trade_parity": "h2_trade_parity_db_url",
    "multi_ts_branch1": "multi_ts_branch1_db_url",
    "stochastics_pbt_inflow": "stochastics_pbt_inflow_db_url",
    "branch2_parent_period": "branch2_parent_period_db_url",
    "case14": "case14_db_url",
}


@pytest.fixture(scope="session")
def scenario_workdir(
    test_db_url,
    stochastic_db_url,
    lh2_db_url,
    h2_trade_parity_db_url,
    multi_ts_branch1_db_url,
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
    Valid values are the keys of :data:`_DB_FIXTURE_URL_NAMES` —
    currently ``"main"`` (tests.json), ``"stochastic"``
    (stochastics.json), ``"lh2"`` (lh2_three_region.json),
    ``"h2_trade_parity"``, ``"multi_ts_branch1"``,
    ``"stochastics_pbt_inflow"``, ``"branch2_parent_period"``.

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
        "h2_trade_parity": h2_trade_parity_db_url,
        "multi_ts_branch1": multi_ts_branch1_db_url,
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


# Phase 3d: canonical cross-cluster sweep list is exported from
# ``_parity_sweep.py`` (a sibling module, not conftest) so test files
# can import it without pytest's conftest discovery firing.
from _parity_sweep import PARITY_SWEEP_CASES  # noqa: F401,E402


@pytest.fixture(autouse=True)
def _reset_global_axis_enums():
    """Reset the cascade-wide axis_enums ContextVar before each test.

    ``load_flextool`` sets the global ContextVar on success (input.py
    finally block ~line 4089) so ``build_flextool`` and other
    post-load consumers see the live vocabulary.  But the
    ContextVar persists into the next test in the same pytest worker,
    pinning Enum dtypes that may differ from the next test's
    fixture's vocabulary — surfacing as "enum on left does not match
    enum on right" SchemaError in joins.

    Resetting before each test guarantees a clean slate.
    """
    from flextool.engine_polars._axis_enums import set_global_axis_enums
    set_global_axis_enums(None)
    yield
    set_global_axis_enums(None)
