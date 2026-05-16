"""Phase E-j meta-guard — flag any newly introduced bare ``path.exists()``
guard in ``_derived_params.py`` (or sibling modules) where the path
target is a ``solve_data/`` or ``input/`` CSV literal.

Why this exists
---------------
Phase E-h fixed nine latent-bug sites where a bare ``path.exists()``
short-circuited under ``csv_emission_disabled()`` even though the
active in-memory FlexData accumulator held the frame.  Phase E-j
migrated the remaining 22 sites in ``_derived_params.py`` to the
seed-aware :func:`_seed_or_exists` helper.

A future contributor adding a fresh ``Path(workdir) / "solve_data" /
"X.csv" / "...if p.exists()..."`` site would silently reintroduce the
class of bug.  This test scans the file and fails when such a pattern
appears, pointing the writer at :func:`_seed_or_exists` instead.

Allowed patterns (NOT flagged)
------------------------------
* ``_seed_or_exists(p)`` — the canonical seed-aware replacement.
* The legacy ``seeded is None and not p.exists()`` shape — the caller
  manually consults the seed before checking the disk; semantically
  equivalent to ``_seed_or_exists`` and not a latent bug.
* ``.exists()`` calls on non-``solve_data``/``input`` paths (sqlite,
  logs, output dirs).
"""
from __future__ import annotations

import re
from pathlib import Path

import flextool.engine_polars._derived_params as _derived_params_module


_TARGET = Path(_derived_params_module.__file__)


# Any ``.exists()`` call.  We capture an optional preceding identifier
# (``p.exists()`` → ``p``) so we can walk back to its assignment.  When
# the call is on a parenthesised inline expression
# (``(Path(...) / "solve_data" / ...).exists()``) the identifier group
# is empty and the inline-literal path inspects the call site directly.
_EXISTS_RE = re.compile(
    r"(?:(?P<var>[A-Za-z_][A-Za-z0-9_]*)|\))\.exists\(\)"
)


_CSV_LIT_RE = re.compile(r'["\'][A-Za-z0-9_\-./]+\.csv["\']')
_DIR_LIT_RE = re.compile(r'["\'](?:solve_data|input)["\']')


def _csv_path_root(
    lines: list[str], current_idx: int, var: str, hops: int = 3,
) -> bool:
    """Walk upward from ``current_idx`` to determine whether ``var``
    eventually traces back to a ``solve_data`` / ``input`` workdir
    construction terminating in a ``.csv`` segment.

    Returns True when the chain shows both a ``solve_data``/``input``
    literal and a ``.csv`` literal (possibly across one indirection
    hop).  Bails out at function boundaries or after ``hops`` levels.
    """
    cur_var = var
    saw_dir = False
    saw_csv = False
    for _ in range(hops):
        found_assign = False
        for j in range(current_idx - 1, max(-1, current_idx - 14), -1):
            prior = _strip_comment(lines[j])
            if re.match(r"\s*(def|class)\s+", prior):
                return False
            assign_re = re.compile(
                r"^\s*" + re.escape(cur_var) + r"\s*=\s*(.+)$"
            )
            am = assign_re.search(prior)
            if am is None:
                continue
            rhs = am.group(1)
            if _DIR_LIT_RE.search(rhs):
                saw_dir = True
            if _CSV_LIT_RE.search(rhs):
                saw_csv = True
            if saw_dir and saw_csv:
                return True
            # Hop through the leftmost identifier in the RHS — typically
            # ``<base> / "..."`` shape.
            next_var_m = re.match(
                r"\s*([A-Za-z_][A-Za-z0-9_]*)", rhs
            )
            if next_var_m is None:
                return False
            nxt = next_var_m.group(1)
            if nxt in ("Path", "str"):
                # Path(workdir) / "..." — no further indirection on Path.
                return False
            cur_var = nxt
            current_idx = j
            found_assign = True
            break
        if not found_assign:
            return False
    return False


def _has_workdir_csv_literal(code: str) -> bool:
    """Return True iff ``code`` contains both a ``"solve_data"`` /
    ``"input"`` literal and a ``"...csv"`` literal — the flextool
    construction shape used to assemble per-solve workdir paths.
    """
    has_dir = bool(re.search(r'["\'](?:solve_data|input)["\']', code))
    has_csv = bool(re.search(r'["\'][A-Za-z0-9_\-./]+\.csv["\']', code))
    return has_dir and has_csv


def _strip_comment(line: str) -> str:
    """Return the code part of ``line`` (everything before a ``#``).

    Conservative: ignores ``#`` inside string literals only by checking
    for an opening quote on the same line.  Sufficient for this codebase
    where ``.exists()`` never appears inside a string literal.
    """
    # If there's a quote before the first ``#``, treat as code.
    hash_idx = line.find("#")
    if hash_idx == -1:
        return line
    pre = line[:hash_idx]
    # Heuristic: if the quote/paren counts are balanced before the ``#``,
    # everything after ``#`` is a comment.  Good enough for this file.
    return pre


def _flag_sites(text: str) -> list[tuple[int, str]]:
    """Return ``[(lineno, snippet), ...]`` for every bare ``.exists()``
    site whose path target is a ``solve_data/``/``input/`` CSV literal
    AND the site is NOT already seed-aware.
    """
    lines = text.splitlines()
    flagged: list[tuple[int, str]] = []
    for i, raw in enumerate(lines):
        code = _strip_comment(raw)
        m = _EXISTS_RE.search(code)
        if m is None:
            continue
        # Skip explicit ``_seed_or_exists(...)`` (no ``.exists()`` there
        # anyway, but defensive).
        if "_seed_or_exists" in code:
            continue
        # Skip the legacy explicit-seed pattern:
        # ``if seeded is None and not p.exists():``
        if re.search(r"seeded\s+is\s+None\s+and", code):
            continue
        var = m.group("var")
        # Inline path-literal — e.g. ``(Path(workdir) / "solve_data" / "x.csv").exists()``
        # The flextool layout uses separate ``Path / "solve_data" / "x.csv"``
        # segments, so detect a same-line pairing of one of the directory
        # tokens and a ``.csv`` literal.
        if _has_workdir_csv_literal(code):
            flagged.append((i + 1, raw.strip()))
            continue
        if var is None:
            # Inline parenthesised receiver without a CSV literal on the
            # same line — not our concern.
            continue
        # Walk upward looking for an assignment to ``var`` that builds
        # a workdir CSV path.  Chain through intermediate references
        # (e.g. ``sd = workdir / "solve_data"`` then
        # ``siu_path = sd / "steps_in_use.csv"``) by following the
        # leftmost identifier in the RHS up to two hops.
        if _csv_path_root(lines, i, var, hops=3):
            flagged.append((i + 1, raw.strip()))
    return flagged


def test_no_bare_path_exists_on_workdir_csv() -> None:
    text = _TARGET.read_text()
    flagged = _flag_sites(text)
    if flagged:
        msg_lines = [
            f"{len(flagged)} bare path.exists() site(s) in "
            f"{_TARGET.name} target a solve_data/ or input/ CSV literal. "
            f"Use _seed_or_exists(path) so csv_emission_disabled() runs do "
            f"not fall through to disk-only logic when the seed holds the "
            f"frame. See specs/phase_e_h_post_mortem.md.",
        ]
        for lineno, snippet in flagged:
            msg_lines.append(f"  L{lineno}: {snippet}")
        raise AssertionError("\n".join(msg_lines))
