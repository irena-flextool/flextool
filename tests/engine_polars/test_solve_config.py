"""polar_high port of flextool's ``test_solve_config.py``.

flextool's test exercises ``SolveConfig.duplicate_solve`` — a
flextool-runner-internal class that propagates per-solve period
maps when a solve gets duplicated for rolling/per-sub-solve modes.
polar_high doesn't reproduce ``SolveConfig`` (chain.py manages
sub-solves directly via FlexData snapshots); the port instead
covers the closest analogue polar_high ships: the
``solve_mode.csv`` parser in :func:`flextool.input._load_solver_options`.

That parser is the parsing-and-application layer flextool's
test framing alludes to ("solve_mode.csv parsing & application"):

* Reads ``input/solve_mode.csv`` (or falls back to ``solve_data/``).
* Selects the row matching ``solve_data/solve_current.csv`` when
  multiple solves are present.
* Translates the forward-compatibility numeric flextool param names
  (``highs_time_limit`` / ``highs_threads`` / ``highs_mip_rel_gap`` /
  ``highs_random_seed`` / ``highs_output_flag`` / …) to HiGHS
  canonical option names (``time_limit``, ``threads``, …).  Batches
  C.3-C.5 retired the three string shortcuts ``highs_method`` /
  ``highs_parallel`` / ``highs_presolve`` — those overrides are now
  authored on ``solver_arguments`` (keys ``solver`` / ``parallel`` /
  ``presolve``) and resolved through
  ``_resolve_effective_highs_options`` instead.
* Coerces the ``value`` column to the right Python type per
  ``_HIGHS_PARAM_MAP``.
* Returns ``None`` when the CSV is missing / empty / has no
  applicable rows.

Each test pins one of these contracts.

flextool's per-solve period-map duplication has no polar_high analogue
to test (polar_high carries period maps in FlexData, not a separate
``SolveConfig`` object), so those cases are intentionally not
ported — see the gap-B6 task brief: "If a test fundamentally
can't be ported … document it in the report and skip — don't
paper over with a fake assertion."  Documented here for the record.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars.input import _load_solver_options


def _write_solve_mode(p: Path, rows: list[tuple[str, str, str | int | float]]) -> None:
    """Write ``input/solve_mode.csv`` with header ``param,solve,value``."""
    p.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=["param", "solve", "value"], orient="row").write_csv(p)


def _write_solve_current(sd: Path, solve: str) -> None:
    sd.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"solve": [solve]}).write_csv(sd / "solve_current.csv")


class TestSolveModeParsing:
    """Parsing layer: ``solve_mode.csv`` → flextool params."""

    def test_forward_compat_numeric_param(self, tmp_path: Path) -> None:
        """The forward-compat numeric flextool params still translate
        to their HiGHS option names with coerced values.  Batches
        C.3-C.5 removed all three string shortcuts (``highs_method`` /
        ``highs_parallel`` / ``highs_presolve``) — those overrides are
        now authored on ``solver_arguments`` and read through the
        effective-options resolver instead.  The forward-compat
        numeric set in ``_HIGHS_PARAM_MAP`` remains in case flextool
        ever emits these rows into ``solve_mode.csv``.
        """
        sd = tmp_path / "solve_data"
        _write_solve_mode(tmp_path / "input" / "solve_mode.csv", [
            ("highs_time_limit", "S", "120.5"),
        ])
        _write_solve_current(sd, "S")
        opts = _load_solver_options(sd)
        assert opts == {"time_limit": pytest.approx(120.5)}

    def test_solve_mode_param_is_ignored(self, tmp_path: Path) -> None:
        """flextool's solve framework param ``solve_mode`` (single_solve /
        rolling / nested) is *not* a HiGHS option — the parser must drop it
        rather than forwarding it to HiGHS (which would crash on the
        unknown option)."""
        sd = tmp_path / "solve_data"
        _write_solve_mode(tmp_path / "input" / "solve_mode.csv", [
            ("highs_time_limit", "S", "60"),
            ("solve_mode", "S", "single_solve"),
            ("solve_mode", "S", "rolling_window"),
        ])
        _write_solve_current(sd, "S")
        opts = _load_solver_options(sd)
        assert opts == {"time_limit": pytest.approx(60.0)}, (
            f"solve_mode rows must be filtered out; got {opts}"
        )

    def test_unknown_param_is_dropped(self, tmp_path: Path) -> None:
        """Random unrecognized flextool params are silently dropped (so
        the loader stays forward-compatible with future writer additions
        without crashing on legacy fixtures)."""
        sd = tmp_path / "solve_data"
        _write_solve_mode(tmp_path / "input" / "solve_mode.csv", [
            ("highs_time_limit", "S", "60"),
            ("future_unknown_param", "S", "whatever"),
        ])
        _write_solve_current(sd, "S")
        opts = _load_solver_options(sd)
        assert opts == {"time_limit": pytest.approx(60.0)}

    def test_numeric_options_get_coerced(self, tmp_path: Path) -> None:
        """The ``_HIGHS_PARAM_MAP`` declares a forward-compatible set of
        numeric options.  When present, they must coerce to the declared
        Python type (float / int / bool), tolerating string
        representations that flextool's CSV writer might emit."""
        sd = tmp_path / "solve_data"
        _write_solve_mode(tmp_path / "input" / "solve_mode.csv", [
            ("highs_time_limit", "S", "120.5"),
            ("highs_threads", "S", "4"),
            ("highs_mip_rel_gap", "S", "1e-4"),
            ("highs_random_seed", "S", "1.0"),  # tolerate "1.0" → 1
            ("highs_output_flag", "S", "true"),
        ])
        _write_solve_current(sd, "S")
        opts = _load_solver_options(sd)
        assert opts["time_limit"] == pytest.approx(120.5)
        assert isinstance(opts["time_limit"], float)
        assert opts["threads"] == 4
        assert isinstance(opts["threads"], int)
        assert opts["mip_rel_gap"] == pytest.approx(1e-4)
        assert opts["random_seed"] == 1
        assert isinstance(opts["random_seed"], int)
        assert opts["output_flag"] is True

    def test_bool_coercion_handles_aliases(self, tmp_path: Path) -> None:
        """bool coercion accepts true/yes/on/1 (and false/no/off/0)
        synonyms (input.py:2617-2621)."""
        for raw, expected in [("true", True), ("yes", True), ("on", True),
                              ("1", True), ("false", False), ("off", False),
                              ("no", False), ("0", False)]:
            sd = tmp_path / f"sd_{raw}" / "solve_data"
            _write_solve_mode(
                tmp_path / f"sd_{raw}" / "input" / "solve_mode.csv",
                [("highs_output_flag", "S", raw)],
            )
            _write_solve_current(sd, "S")
            opts = _load_solver_options(sd)
            assert opts == {"output_flag": expected}, (
                f"bool coercion of {raw!r} → got {opts}"
            )

    def test_malformed_value_does_not_crash(self, tmp_path: Path) -> None:
        """Malformed numeric values are silently dropped — let the HiGHS
        defaults stand rather than crashing the whole loader on one bad
        cell (input.py:2670-2672)."""
        sd = tmp_path / "solve_data"
        _write_solve_mode(tmp_path / "input" / "solve_mode.csv", [
            ("highs_time_limit", "S", "not_a_float"),
            ("highs_threads", "S", "4"),
        ])
        _write_solve_current(sd, "S")
        opts = _load_solver_options(sd)
        assert opts == {"threads": 4}

    def test_missing_csv_returns_none(self, tmp_path: Path) -> None:
        """No ``input/solve_mode.csv`` AND no ``solve_data/solve_mode.csv``
        ⇒ ``solver_options=None`` (HiGHS defaults stand)."""
        sd = tmp_path / "solve_data"
        sd.mkdir()
        assert _load_solver_options(sd) is None

    def test_empty_csv_returns_none(self, tmp_path: Path) -> None:
        """Header-only solve_mode.csv ⇒ None.

        The loader explicitly rejects empty / malformed files so that
        downstream HiGHS runs with default options (input.py:2632-2633).
        """
        sd = tmp_path / "solve_data"
        sd.mkdir()
        (tmp_path / "input").mkdir()
        # Header only, no rows.
        pl.DataFrame(
            schema={"param": pl.Utf8, "solve": pl.Utf8, "value": pl.Utf8}
        ).write_csv(tmp_path / "input" / "solve_mode.csv")
        assert _load_solver_options(sd) is None

    def test_solve_data_fallback(self, tmp_path: Path) -> None:
        """Some fixtures drop ``solve_mode.csv`` under ``solve_data/``
        instead of ``input/`` — the loader must fall back (input.py:2628)."""
        sd = tmp_path / "solve_data"
        sd.mkdir()
        # No input/solve_mode.csv; only solve_data/solve_mode.csv.
        pl.DataFrame(
            [("highs_time_limit", "S", "60")],
            schema=["param", "solve", "value"], orient="row",
        ).write_csv(sd / "solve_mode.csv")
        _write_solve_current(sd, "S")
        opts = _load_solver_options(sd)
        assert opts == {"time_limit": pytest.approx(60.0)}


class TestSolveModeApplication:
    """Application layer: per-solve disambiguation via solve_current.csv."""

    def test_active_solve_row_is_picked(self, tmp_path: Path) -> None:
        """When solve_mode.csv has rows for multiple solves, the active
        one (per ``solve_data/solve_current.csv``) is selected."""
        sd = tmp_path / "solve_data"
        _write_solve_mode(tmp_path / "input" / "solve_mode.csv", [
            ("highs_time_limit", "S1", "60"),
            ("highs_time_limit", "S2", "120"),
            ("highs_time_limit", "S3", "300"),
        ])
        _write_solve_current(sd, "S2")
        opts = _load_solver_options(sd)
        assert opts == {"time_limit": pytest.approx(120.0)}, (
            f"expected S2 row to win; got {opts}"
        )

    def test_other_solves_filtered_out(self, tmp_path: Path) -> None:
        """Rows for other solves don't bleed into the active solve's
        config — even if they declare a different value for the same
        param."""
        sd = tmp_path / "solve_data"
        _write_solve_mode(tmp_path / "input" / "solve_mode.csv", [
            ("highs_time_limit", "S1", "60"),
            ("highs_threads", "S1", "8"),
            ("highs_time_limit", "S2", "120"),
            # No highs_threads for S2 — must NOT inherit S1's "8".
        ])
        _write_solve_current(sd, "S2")
        opts = _load_solver_options(sd)
        assert opts == {"time_limit": pytest.approx(120.0)}, (
            f"S2 must not inherit S1.threads; got {opts}"
        )

    def test_missing_solve_current_picks_any_row(self, tmp_path: Path) -> None:
        """No ``solve_current.csv`` ⇒ single-solve fixture; the parser
        takes whatever rows are there (input.py:2640-2652)."""
        sd = tmp_path / "solve_data"
        sd.mkdir()
        _write_solve_mode(tmp_path / "input" / "solve_mode.csv", [
            ("highs_time_limit", "ANY", "60"),
        ])
        # No solve_current.csv.
        opts = _load_solver_options(sd)
        assert opts == {"time_limit": pytest.approx(60.0)}

    def test_solve_not_in_csv_falls_back(self, tmp_path: Path) -> None:
        """solve_current names a solve absent from solve_mode.csv — the
        parser falls back to the full table (input.py:2649-2650).
        Behaviorally equivalent to "single-solve mode, take whatever
        rows exist" so the loader doesn't wedge on a typo."""
        sd = tmp_path / "solve_data"
        _write_solve_mode(tmp_path / "input" / "solve_mode.csv", [
            ("highs_time_limit", "S1", "60"),
        ])
        _write_solve_current(sd, "TYPO_SOLVE")
        opts = _load_solver_options(sd)
        assert opts == {"time_limit": pytest.approx(60.0)}


class TestRealFixture:
    """End-to-end against the real ``work_base_weighted`` fixture flextool
    emits.  After Batches C.3-C.5 the regenerated ``solve_mode.csv``
    carries only the (ignored) ``solve_mode`` row, so the loader
    returns ``None`` and HiGHS-side defaults stand."""

    def test_work_base_weighted(self, scenario_workdir) -> None:
        work = scenario_workdir("base_weighted", db_fixture="main")
        sd = work / "solve_data"
        opts = _load_solver_options(sd)
        # solve_mode.csv has only the ``solve_mode`` framework rows
        # (ignored by the HiGHS-option loader) — Batches C.3-C.5
        # retired ``highs_method`` / ``highs_parallel`` /
        # ``highs_presolve``; those overrides are now authored on
        # ``solver_arguments`` and read through the engine-side
        # ``_resolve_effective_highs_options``.
        assert opts is None
