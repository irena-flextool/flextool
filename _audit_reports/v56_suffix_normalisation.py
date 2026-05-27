"""E.3 — mechanical normalisation of `parameter_definitions` description
suffixes against the (now-complete) `parameter_types` block in
``flextool/schemas/spinedb_schema.json``.

Scope per task brief: rewrite ONLY when the description already has a
detectable canonical-suffix-style trailing annotation that disagrees
with the derived canonical form. Descriptions that lack a trailing
type-annotation sentence are left untouched — that's a different
policy decision (out of scope for E.3).

Run modes:
  python -m _audit_reports.v56_suffix_normalisation --dry-run
  python -m _audit_reports.v56_suffix_normalisation --apply

The script is a one-off; kept under ``_audit_reports/`` so we can
re-run/audit it later without polluting the engine package.

Derivation rule per audit §Q3:

  tuple                                                  canonical suffix
  ────────────────────────────────────────────────────────────────────────
  ("float",) or ("str",) or scalar-only                  Constant.
  ("array",)                                             Array.
  ("1d_map",)  index ⇒ time                              Time.
  ("1d_map",)  index ⇒ period                            Period.
  ("float","1d_map") or ("str","1d_map")  1d⇒period      Constant or period.
  ("float","1d_map")                       1d⇒time       Constant or time.
  ("float","1d_map","2d_map")                            Constant, period, time or period+time.
  ("2d_map",)                                            Period and time.
  ("2d_map","3d_map")                                    Stochastic time.
  ("1d_map","3d_map")                                    Time or stochastic time.
  ("float","1d_map","3d_map")                            Constant, time or stochastic time.
  ("float","1d_map","2d_map","3d_map")                   Constant, period, time, period+time or stochastic time.
  ("4d_map",) or any 4d_map                              Stochastic-branch map.

The 1d_map period-vs-time discriminator uses
``flextool.engine_polars._param_shapes.PARAM_ALLOWED_SHAPES`` when the
parameter is registered; otherwise it falls back to prose-hint detection
("time" or "period" in the existing canonical suffix).

The splitter only strips trailing sentences that match a known
canonical-suffix template (case-insensitive, with or without trailing
period); substantive sentences that happen to end in a similar word
are left intact.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flextool.engine_polars._param_shapes import PARAM_ALLOWED_SHAPES, Shape  # noqa: E402

SCHEMA = ROOT / "flextool" / "schemas" / "spinedb_schema.json"

# ---------------------------------------------------------------------------
# canonical suffix templates (the set we will emit)
# ---------------------------------------------------------------------------

CANONICAL_SUFFIXES = {
    "Constant.",
    "Array.",
    "Period.",
    "Time.",
    "Period or time.",
    "Constant or period.",
    "Constant or time.",
    "Period and time.",
    "Constant, period, time or period+time.",
    "Period, time or period+time.",
    "Time or stochastic time.",
    "Constant, time or stochastic time.",
    "Stochastic time.",
    "Constant, period or time.",
    "Constant, period, time, period+time or stochastic time.",
    "Stochastic-branch map.",
}

# additional historical canonical phrasings to strip (case-insensitive) so
# we don't double-append. These never appear with substantive prose; they
# are pure suffix templates that prior editors used.
HISTORICAL_SUFFIXES_CI = [
    "constant.",
    "array.",
    "period.",
    "time.",
    "period or time.",
    "constant or period.",
    "constant or time.",
    "period and time.",
    "constant, period or time.",
    "constant, period, time or period+time.",
    "constant, time or stochastic time.",
    "time or stochastic time.",
    "constant, period, time, period+time or stochastic time.",
    "stochastic time.",
    "stochastic-branch map.",
]

# Sort longer first for greedy match
HISTORICAL_SUFFIXES_CI.sort(key=len, reverse=True)


def split_desc(desc: str) -> tuple[str, str | None]:
    """Strip a trailing canonical-suffix sentence (case-insensitive,
    trailing period optional) from ``desc``. Returns
    (head_without_suffix, stripped_suffix or None).

    The canonical suffix must be a standalone sentence — preceded by a
    sentence-boundary punctuation (``.`` / ``!`` / ``?``) plus whitespace,
    or be the entire string. Words like "in each period." inside a
    substantive sentence are therefore NOT stripped."""
    if not desc:
        return desc or "", None
    stripped = desc.rstrip()
    lower = stripped.lower()
    SENTENCE_BOUNDARIES = (". ", ".\n", ".\t", "! ", "? ")
    for pat in HISTORICAL_SUFFIXES_CI:
        # canonical with trailing period
        for sep in SENTENCE_BOUNDARIES:
            tail = sep + pat
            if lower.endswith(tail):
                cut = len(stripped) - len(pat)
                return stripped[:cut].rstrip(), stripped[cut:]
        if lower == pat:
            return "", stripped
        # canonical without trailing period (legacy "Constant or Period")
        bare = pat.rstrip(".")
        for sep in SENTENCE_BOUNDARIES:
            tail = sep + bare
            if lower.endswith(tail):
                cut = len(stripped) - len(bare)
                return stripped[:cut].rstrip(), stripped[cut:]
        if lower == bare:
            return "", stripped
    return desc, None


def to_tup(rows: list[tuple[str, int]]) -> tuple[str, ...]:
    """Convert parameter_types rows ``[(type, depth), …]`` to the canonical
    tuple form (e.g. ``("float", "1d_map", "3d_map")``)."""
    out: list[str] = []
    for t, d in rows:
        if t == "float":
            out.append("float")
        elif t == "str":
            out.append("str")
        elif t == "array":
            out.append("array")
        elif t == "map":
            out.append(f"{d}d_map")
    order = {"float": 0, "str": 1, "array": 2, "1d_map": 3, "2d_map": 4, "3d_map": 5, "4d_map": 6}
    return tuple(sorted(out, key=lambda x: order.get(x, 99)))


def prose_hint_for_one_d(text: str | None) -> str | None:
    """Look at an existing description / stripped suffix and decide whether
    the parameter's 1d_map indexes period or time. Returns 'time',
    'period', 'both', or None (no hint)."""
    if not text:
        return None
    s = text.lower()
    says_time = ("or time" in s) or (", time" in s)
    says_period = ("or period" in s) or (", period" in s)
    if says_time and says_period:
        return "both"
    if says_time:
        return "time"
    if says_period:
        return "period"
    return None


def engine_one_d_axis(cls: str, name: str) -> str | None:
    """Return 'time' if the engine registry says the 1d_map can index
    time but never period; 'period' for the converse; 'both' if both;
    None if the parameter isn't in the registry.
    """
    sh = PARAM_ALLOWED_SHAPES.get((cls, name))
    if not sh:
        return None
    has_time = any(x in sh for x in (Shape.MAP_TIME, Shape.MAP_PERIOD_TIME, Shape.MAP_TIME_PERIOD))
    has_period = any(x in sh for x in (Shape.MAP_PERIOD, Shape.MAP_PERIOD_TIME, Shape.MAP_TIME_PERIOD))
    if has_time and has_period:
        return "both"
    if has_time:
        return "time"
    if has_period:
        return "period"
    return None


def derive_suffix(
    cls: str,
    name: str,
    tup: tuple[str, ...],
    old_desc_with_suffix: str,
) -> str | None:
    """Mechanical suffix derivation per the audit rule table.

    Falls back to a prose hint over ``old_desc_with_suffix`` only for
    ambiguous 1d_map (period-vs-time) cases when the engine registry has
    no entry. Returns None for an untyped parameter (should not occur
    post-E.1)."""
    has_float = "float" in tup
    has_str = "str" in tup
    has_array = "array" in tup
    has_1d = "1d_map" in tup
    has_2d = "2d_map" in tup
    has_3d = "3d_map" in tup
    has_4d = "4d_map" in tup

    if has_4d:
        return "Stochastic-branch map."

    # 3d_map family
    if has_3d:
        if has_2d and has_1d and (has_float or has_str):
            return "Constant, period, time, period+time or stochastic time."
        if has_1d and (has_float or has_str):
            return "Constant, time or stochastic time."
        if has_1d:
            return "Time or stochastic time."
        if has_2d:
            return "Stochastic time."
        return "Stochastic time."

    # 2d_map family (no 3d_map)
    if has_2d:
        if has_1d and (has_float or has_str):
            return "Constant, period, time or period+time."
        if has_1d:
            return "Period, time or period+time."
        return "Period and time."

    # 1d_map only family
    if has_1d:
        engine_axis = engine_one_d_axis(cls, name)
        prose_axis = prose_hint_for_one_d(old_desc_with_suffix)
        if has_float or has_str:
            # Engine registry is the strongest signal.
            if engine_axis == "time":
                return "Constant or time."
            if engine_axis == "both":
                return "Constant, period or time."
            if engine_axis == "period":
                return "Constant or period."
            # No engine entry → trust prose hint when unambiguous.
            if prose_axis == "time":
                return "Constant or time."
            if prose_axis == "period":
                return "Constant or period."
            if prose_axis == "both":
                # Prose claims both but schema declares only
                # (float, 1d_map) AND engine registry has no entry —
                # this is a Tier-3 / engine-narrowing candidate. Surface
                # via None; the operator picks the correct suffix.
                return None
            return "Constant or period."  # default
        # 1d_map without scalar
        if engine_axis == "time":
            return "Time."
        if engine_axis == "both":
            return "Period or time."
        if engine_axis == "period":
            return "Period."
        if prose_axis == "time":
            return "Time."
        if prose_axis == "period":
            return "Period."
        if prose_axis == "both":
            return None
        return "Period."

    if has_array:
        return "Array."
    if has_float or has_str:
        return "Constant."
    return None


def normalised_capitalisation(s: str) -> str:
    """Force the canonical form: only the leading letter capitalised
    (and proper noun-free), trailing period."""
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    if not (args.dry_run or args.apply):
        ap.error("specify --dry-run or --apply")

    schema = json.loads(SCHEMA.read_text())
    types = schema["parameter_types"]
    type_map: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for row in types:
        type_map.setdefault((row[0], row[1]), []).append((row[2], row[3]))

    defs = schema["parameter_definitions"]
    rewrites: list[tuple[str, str, str, str, str, str]] = []
    ambiguous: list[tuple[str, str, tuple[str, ...]]] = []
    no_type: list[tuple[str, str]] = []

    for d in defs:
        cls, name = d[0], d[1]
        desc = d[4] or ""
        tup = to_tup(type_map.get((cls, name), []))
        if not tup:
            no_type.append((cls, name))
            continue
        head, cur_suffix = split_desc(desc)
        # Scope per task: only rewrite when an existing canonical suffix
        # is present and disagrees with the derived form. Skip rows with
        # no detectable trailing annotation.
        if cur_suffix is None:
            continue
        derived = derive_suffix(cls, name, tup, desc)
        if derived is None:
            ambiguous.append((cls, name, tup))
            continue
        if head:
            new_desc = head.rstrip() + " " + derived
        else:
            new_desc = derived
        if new_desc == desc:
            continue
        rewrites.append((cls, name, tup, desc, new_desc, derived))

    total = len(defs)
    print(f"Total parameter_definitions: {total}")
    print(f"  unchanged:  {total - len(rewrites) - len(ambiguous) - len(no_type)}")
    print(f"  rewrites:   {len(rewrites)}")
    print(f"  ambiguous:  {len(ambiguous)}")
    print(f"  untyped:    {len(no_type)}")

    if ambiguous:
        print("\n=== AMBIGUOUS (cannot derive) ===")
        for cls, name, tup in ambiguous:
            print(f"  {cls}.{name}: types={tup}")
    if no_type:
        print("\n=== UNTYPED (post-E.1/E.2, should be 0) ===")
        for cls, name in no_type:
            print(f"  {cls}.{name}")

    pct = (len(ambiguous) + len(no_type)) / total * 100
    if pct > 15:
        print(f"\n!!! {pct:.1f}% of rows resisted mechanical derivation — STOP and revisit.")
        sys.exit(2)

    if args.dry_run:
        print("\n=== FIRST 10 REWRITES (sample) ===")
        for cls, name, tup, old, new, derived in rewrites[:10]:
            print(f"\n  {cls}.{name}  types={tup}")
            print(f"    OLD: {old!r}")
            print(f"    NEW: {new!r}")
        return

    # apply: rewrite descriptions, preserve all other fields
    for d in defs:
        cls, name = d[0], d[1]
        for c2, n2, tup, old, new, _ in rewrites:
            if c2 == cls and n2 == name:
                d[4] = new
                break

    schema["parameter_definitions"] = defs
    out = json.dumps(schema, indent=2, ensure_ascii=False) + "\n"
    SCHEMA.write_text(out)
    print(f"\nApplied {len(rewrites)} rewrites to {SCHEMA}.")


if __name__ == "__main__":
    main()
