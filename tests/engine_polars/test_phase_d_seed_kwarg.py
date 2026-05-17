"""Phase D — ``load_flextool(seed=…)`` parity test.

Phase C built the ``FlexDataAccumulator`` that captures the 37
``OK_thin_wrapper`` writers' derived frames during a sub-solve's
preprocessing pass.  Phase D wires an optional ``seed`` keyword into
:func:`load_flextool` so the accumulator can short-circuit the
disk-read for those CSVs.

This test runs the cascade once on the smallest fixture
(``work_base``) to obtain a populated accumulator, then loads
:class:`FlexData` twice off the resulting workdir:

  1. ``load_flextool(work)`` — the existing disk-read path.
  2. ``load_flextool(work, seed=accum)`` — the new seed path.

The two ``FlexData`` instances must be field-by-field equal.  The
seeded basenames (covered by the accumulator) take the in-memory
path; everything else falls back to disk-read, so the only delta
between the two loads is the source of the 37+ thin-wrapper CSV
frames.  Phase C asserted those are byte-equivalent to the on-disk
CSV, so the resulting FlexData must match exactly.

Phase D does NOT change the cascade itself — ``run_chain_from_db``
continues to call ``load_flextool`` without the ``seed`` kwarg.
Phase E will flip the cascade.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars import load_flextool, run_chain_from_db
from flextool.engine_polars._flex_data_accumulator import (
    FlexDataAccumulator,
)


# Step 1-f — the per-sub-solve accumulator is no longer populated, so
# ``last_step.flex_data_accumulator`` is always ``None`` and the
# ``seed=`` kwarg goes via the Provider instead.  These tests document
# the legacy seed-kwarg pathway which Step 2 deletes outright.
pytestmark = [
    pytest.mark.solver,
    pytest.mark.skip(
        reason="Step 1-f — seed kwarg fed via Provider; Step 2 deletes the seed kwarg.",
    ),
]


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
WORK_NAME = "work_base"
SCENARIO_NAME = "base"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _frames_equal(a: object, b: object) -> bool:
    """Order-insensitive frame comparison tolerant to dtype drift.

    Returns ``True`` when both inputs are ``None``, or both are
    DataFrames with the same column set and identical content after
    sorting on all columns.
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if not isinstance(a, pl.DataFrame) or not isinstance(b, pl.DataFrame):
        return a == b
    if set(a.columns) != set(b.columns):
        return False
    a_aligned = a.select(sorted(a.columns))
    b_aligned = b.select(sorted(a.columns))
    if a_aligned.is_empty() and b_aligned.is_empty():
        return True
    return a_aligned.sort(a_aligned.columns).equals(
        b_aligned.sort(b_aligned.columns))


def _param_frames_equal(a: object, b: object) -> bool:
    """``Param`` field comparison: unwrap to underlying frame, then
    delegate to :func:`_frames_equal`.

    :class:`polar_high.Param` exposes its underlying frame via
    ``.frame`` (which may be a :class:`pl.LazyFrame` — collect when
    needed).  ``None`` on both sides is accepted.
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    a_frame = getattr(a, "frame", None)
    b_frame = getattr(b, "frame", None)
    if a_frame is None or b_frame is None:
        return a == b
    if isinstance(a_frame, pl.LazyFrame):
        a_frame = a_frame.collect()
    if isinstance(b_frame, pl.LazyFrame):
        b_frame = b_frame.collect()
    return _frames_equal(a_frame, b_frame)


def _fielded_equal(name: str, a: object, b: object) -> tuple[bool, str]:
    """Per-field comparator — returns ``(ok, message)``.

    Handles ``pl.DataFrame`` / ``pl.LazyFrame``, ``Param`` (via
    duck-typed ``.frame``), and scalar / None / sequence fields by
    direct equality.  Defensive against ambiguous-truth errors from
    polars frames embedded inside container fields.
    """
    if a is None and b is None:
        return True, ""
    if a is None or b is None:
        return False, f"{name}: one side None ({a!r} vs {b!r})"
    # DataFrame / LazyFrame branch.
    if isinstance(a, (pl.DataFrame, pl.LazyFrame)) or isinstance(
        b, (pl.DataFrame, pl.LazyFrame)
    ):
        a2 = a.collect() if isinstance(a, pl.LazyFrame) else a
        b2 = b.collect() if isinstance(b, pl.LazyFrame) else b
        ok = _frames_equal(a2, b2)
        return ok, "" if ok else f"{name}: frame mismatch"
    # Param branch (duck-typed: has ``.frame`` that is a polars frame).
    def _is_param(x: object) -> bool:
        f = getattr(x, "frame", None)
        return isinstance(f, (pl.DataFrame, pl.LazyFrame))
    if _is_param(a) or _is_param(b):
        ok = _param_frames_equal(a, b)
        return ok, "" if ok else f"{name}: Param mismatch"
    # Scalar / None / other — guard against ambiguous-truth.
    try:
        ok = bool(a == b)
    except (ValueError, TypeError):
        # polars frame embedded in tuple/dict — fall back to repr eq.
        ok = repr(a) == repr(b)
    return ok, "" if ok else f"{name}: scalar mismatch ({a!r} vs {b!r})"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _run_cascade(tmp_path: Path) -> tuple[Path, FlexDataAccumulator]:
    fixture = DATA / WORK_NAME
    db = fixture / "tests.sqlite"
    if not db.exists():
        pytest.skip(f"fixture sqlite missing: {db}")
    work = tmp_path / WORK_NAME
    sols = run_chain_from_db(
        db, scenario_name=SCENARIO_NAME, work_folder=work,
    )
    assert sols, "cascade produced no sub-solves"
    last_step = next(reversed(sols.values()))
    accum = last_step.flex_data_accumulator
    assert accum is not None and isinstance(accum, FlexDataAccumulator), (
        "Phase C accumulator not present on the last step — wiring "
        "regression."
    )
    assert accum.frames, "accumulator captured zero frames"
    return work, accum


def test_seed_kwarg_field_by_field_parity(tmp_path: Path) -> None:
    """``load_flextool(work, seed=accum)`` must produce a FlexData
    field-by-field equal to ``load_flextool(work)`` (disk-read path)."""
    work, accum = _run_cascade(tmp_path)

    fd_disk = load_flextool(work)
    fd_seed = load_flextool(work, seed=accum)

    # Both are dataclasses with identical field names.  Walk the fields
    # and compare each one.  Any mismatch fails fast with a per-field
    # report so the parity-break is localised.
    fields_disk = {f.name for f in dataclasses.fields(fd_disk)}
    fields_seed = {f.name for f in dataclasses.fields(fd_seed)}
    assert fields_disk == fields_seed, (
        f"FlexData field set diverged: only-disk={fields_disk - fields_seed} "
        f"only-seed={fields_seed - fields_disk}"
    )

    mismatches: list[str] = []
    for name in sorted(fields_disk):
        a = getattr(fd_disk, name)
        b = getattr(fd_seed, name)
        ok, msg = _fielded_equal(name, a, b)
        if not ok:
            mismatches.append(msg)

    assert not mismatches, (
        "Phase D seed-vs-disk FlexData mismatch:\n  "
        + "\n  ".join(mismatches)
    )


def test_seed_lookup_basename_and_suffix(tmp_path: Path) -> None:
    """``FlexDataAccumulator.lookup`` accepts ``Path``, basename, and
    suffix-less names — the API contract for the seed hook.

    Phase E-d — the accumulator now stores BOTH a bare-basename key
    AND a ``"<parent>/<basename>"`` key per capture (to disambiguate
    the ``input/X.csv`` vs ``solve_data/X.csv`` basename collision).
    When the caller passes a qualified path, ``lookup`` returns ONLY
    the matching parent-qualified frame.
    """
    work, accum = _run_cascade(tmp_path)
    # Pick one key the work_base fixture is guaranteed to cover.
    coverage_keys = [
        k for k in accum.frames
        if k.endswith(".csv") and "/" not in k
    ]
    assert coverage_keys, "no .csv keys captured"
    key = coverage_keys[0]
    base = key[:-len(".csv")]

    # Find which parent this key lives under by checking the qualified
    # keys.  Phase E-d stores both bare and "parent/basename".
    parent_keys = [
        k for k in accum.frames
        if k.endswith("/" + key)
    ]
    assert parent_keys, f"no parent-qualified key for {key}"
    parent_dir = parent_keys[0].split("/", 1)[0]

    # 1. Bare basename round-trips.
    assert accum.lookup(key) is not None
    # 2. Suffix-less basename works.
    assert accum.lookup(base) is not None
    # 3. Full path (with the correct directory prefix) works.
    assert accum.lookup(work / parent_dir / key) is not None
    # 4. Full path with the WRONG parent dir returns None (Phase E-d
    #    disambiguation prevents wrong-frame lookups).
    wrong_parent = "input" if parent_dir != "input" else "solve_data"
    if not any(k.startswith(wrong_parent + "/") and k.endswith("/" + key)
               for k in accum.frames):
        assert accum.lookup(work / wrong_parent / key) is None
    # 5. Unknown basename → None.
    assert accum.lookup("definitely_not_a_writer.csv") is None


def test_seed_kwarg_default_is_none_backwards_compatible(tmp_path: Path) -> None:
    """``load_flextool(work)`` with no ``seed`` argument must work
    exactly as before — the kwarg is opt-in only."""
    work, _ = _run_cascade(tmp_path)
    # Two no-seed loads must themselves be field-by-field equal (proves
    # disk-read path is deterministic and the seed kwarg's default
    # ``None`` is fully transparent).
    a = load_flextool(work)
    b = load_flextool(work)
    fields = {f.name for f in dataclasses.fields(a)}
    for name in fields:
        va = getattr(a, name)
        vb = getattr(b, name)
        ok, msg = _fielded_equal(name, va, vb)
        assert ok, f"disk-read path non-deterministic on field {name}: {msg}"
