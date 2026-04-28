"""Unit tests for migration.mps_parity.

Exercises the canonical-form parser on tiny synthetic MPS files. Real
fixture comparison is exercised separately by the migration's per-step
validation gate.
"""
from __future__ import annotations

from textwrap import dedent

import pytest

from migration.mps_parity import canonical_hash, diff_canonical, parse_mps


# A 2-row, 2-col LP in Free MPS. Standard form is intentional — the
# fixture exercises ROWS, COLUMNS, RHS, BOUNDS at the minimum.
_BASE = dedent("""\
    NAME tiny
    ROWS
     N obj
     L c1
     G c2
    COLUMNS
     x obj 1.0 c1 2.0
     x c2 1.0
     y obj 3.0 c1 4.0
     y c2 1.0
    RHS
     RHS1 c1 10.0 c2 1.0
    BOUNDS
     UP BND1 x 5.0
     LO BND1 y 0.0
    ENDATA
""")


def _write(tmp_path, name: str, body: str):
    p = tmp_path / name
    p.write_text(body)
    return p


def test_parses_basic_structure(tmp_path):
    canon = parse_mps(_write(tmp_path, "a.mps", _BASE))
    assert canon.name == "tiny"
    assert canon.objsense == "MIN"
    assert ("N", "obj") in canon.rows
    assert ("L", "c1") in canon.rows
    assert ("G", "c2") in canon.rows
    cols = dict(canon.columns)
    assert "x" in cols and "y" in cols
    rhs = dict(canon.rhs)
    assert set(rhs) == {"c1", "c2"}


def test_row_reorder_does_not_change_hash(tmp_path):
    reordered_rows = dedent("""\
        NAME tiny
        ROWS
         G c2
         N obj
         L c1
        COLUMNS
         x obj 1.0 c1 2.0
         x c2 1.0
         y obj 3.0 c1 4.0
         y c2 1.0
        RHS
         RHS1 c1 10.0 c2 1.0
        BOUNDS
         UP BND1 x 5.0
         LO BND1 y 0.0
        ENDATA
    """)
    a = parse_mps(_write(tmp_path, "a.mps", _BASE))
    b = parse_mps(_write(tmp_path, "b.mps", reordered_rows))
    assert canonical_hash(a) == canonical_hash(b)
    assert diff_canonical(a, b) is None


def test_column_reorder_does_not_change_hash(tmp_path):
    swapped = dedent("""\
        NAME tiny
        ROWS
         N obj
         L c1
         G c2
        COLUMNS
         y obj 3.0 c1 4.0
         y c2 1.0
         x obj 1.0 c1 2.0
         x c2 1.0
        RHS
         RHS1 c1 10.0 c2 1.0
        BOUNDS
         UP BND1 x 5.0
         LO BND1 y 0.0
        ENDATA
    """)
    a = parse_mps(_write(tmp_path, "a.mps", _BASE))
    b = parse_mps(_write(tmp_path, "b.mps", swapped))
    assert canonical_hash(a) == canonical_hash(b)


def test_within_column_pair_reorder_does_not_change_hash(tmp_path):
    # x's two rows expressed in opposite order
    inner_swap = _BASE.replace(
        " x obj 1.0 c1 2.0\n x c2 1.0",
        " x c2 1.0\n x obj 1.0 c1 2.0",
    )
    a = parse_mps(_write(tmp_path, "a.mps", _BASE))
    b = parse_mps(_write(tmp_path, "b.mps", inner_swap))
    assert canonical_hash(a) == canonical_hash(b)


def test_coefficient_change_changes_hash(tmp_path):
    flipped = _BASE.replace("x obj 1.0", "x obj 1.5")
    a = parse_mps(_write(tmp_path, "a.mps", _BASE))
    b = parse_mps(_write(tmp_path, "b.mps", flipped))
    assert canonical_hash(a) != canonical_hash(b)
    diff = diff_canonical(a, b)
    assert diff is not None
    assert "COEF" in diff and "x" in diff and "obj" in diff


def test_rhs_change_changes_hash(tmp_path):
    flipped = _BASE.replace("c1 10.0", "c1 11.0")
    a = parse_mps(_write(tmp_path, "a.mps", _BASE))
    b = parse_mps(_write(tmp_path, "b.mps", flipped))
    assert canonical_hash(a) != canonical_hash(b)
    diff = diff_canonical(a, b)
    assert diff is not None and "RHS" in diff


def test_bound_change_changes_hash(tmp_path):
    flipped = _BASE.replace("UP BND1 x 5.0", "UP BND1 x 7.0")
    a = parse_mps(_write(tmp_path, "a.mps", _BASE))
    b = parse_mps(_write(tmp_path, "b.mps", flipped))
    assert canonical_hash(a) != canonical_hash(b)
    diff = diff_canonical(a, b)
    assert diff is not None and "BOUNDS" in diff


def test_extra_row_detected(tmp_path):
    extra = _BASE.replace(" G c2\n", " G c2\n L c3\n")
    a = parse_mps(_write(tmp_path, "a.mps", _BASE))
    b = parse_mps(_write(tmp_path, "b.mps", extra))
    diff = diff_canonical(a, b)
    assert diff is not None and "ROWS only in B" in diff


def test_objsense_max(tmp_path):
    max_form = _BASE.replace("ROWS\n", "OBJSENSE\n MAX\nROWS\n")
    canon = parse_mps(_write(tmp_path, "a.mps", max_form))
    assert canon.objsense == "MAX"


def test_packed_rhs_pairs_parsed(tmp_path):
    # Two rhs pairs on one line — already exercised by _BASE, but assert
    # the values are what we expect.
    canon = parse_mps(_write(tmp_path, "a.mps", _BASE))
    rhs = dict(canon.rhs)
    assert len(rhs) == 2  # c1 and c2
