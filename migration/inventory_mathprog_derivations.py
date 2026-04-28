"""Inventory of derived sets and calculated parameters in flextool.mod.

Walks flextool.mod, classifies every top-level ``set`` and ``param``
declaration, and emits ``migration/inventory.csv``. Used by Phase 0 to
size the python-preprocessing migration and by Phase 0's DAG builder to
order steps.

Classification fields:
- kind: 'set' or 'param'
- name: identifier
- line: 1-based line number of the declaration
- dimen: dimension (sets) or index expression (params) — best-effort string
- within: optional ``within ...`` constraint (sets only) or 'default ...' clause (params)
- has_derivation: True if body contains ':=' (derived/computed) vs. data-loaded
- already_loaded: True if a matching ``table data IN ... <- [..., name, ...];`` exists
- references_variables: True if body mentions a ``v_*`` or ``vq_*`` identifier — these MUST stay in the model
- references: comma-joined list of *other* set/param identifiers the body references (best-effort)
- complexity: heuristic — 'simple', 'conditional', 'joining', 'multi_clause'
- body_chars: length of the derivation body (rough size signal)

CLI:
    python -m migration.inventory_mathprog_derivations
        [--mod flextool/flextool.mod]
        [--out migration/inventory.csv]
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


# Identifiers used as variables — body referencing these means the
# derivation depends on solver decisions and CANNOT move to Python.
_VARIABLE_PREFIXES = ("v_", "vq_")


# Comments are line-prefixed with '#' or '/*...*/'. We strip them before
# tokenizing to avoid false matches on identifiers in commentary.
_LINE_COMMENT = re.compile(r"#[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def _strip_comments(src: str) -> str:
    src = _BLOCK_COMMENT.sub(" ", src)
    src = _LINE_COMMENT.sub("", src)
    return src


def _split_top_level_statements(src: str) -> list[tuple[int, str]]:
    """Split source into ``;``-terminated top-level statements.

    Returns list of (1-based line number where the statement starts, body).
    Quoted strings and bracket nesting are respected so we don't split on
    semicolons that appear inside expressions.
    """
    out: list[tuple[int, str]] = []
    depth = 0
    in_squote = False
    in_dquote = False
    buf: list[str] = []
    line = 1
    start_line = 1
    i = 0
    n = len(src)
    while i < n:
        c = src[i]
        if not buf:
            start_line = line
        if c == "\n":
            line += 1
        if in_squote:
            buf.append(c)
            if c == "'":
                in_squote = False
            i += 1
            continue
        if in_dquote:
            buf.append(c)
            if c == '"':
                in_dquote = False
            i += 1
            continue
        if c == "'":
            in_squote = True
            buf.append(c)
            i += 1
            continue
        if c == '"':
            in_dquote = True
            buf.append(c)
            i += 1
            continue
        if c in "({[":
            depth += 1
            buf.append(c)
            i += 1
            continue
        if c in ")}]":
            depth -= 1
            buf.append(c)
            i += 1
            continue
        if c == ";" and depth == 0:
            stmt = "".join(buf).strip()
            if stmt:
                out.append((start_line, stmt))
            buf = []
            i += 1
            continue
        buf.append(c)
        i += 1
    return out


# MathProg ``table data IN`` reads use the form
#   <set_or_param_name> <- [col1, col2, ...]
#   <param_alias> ~ <db_column>
# The identifier we want is the one immediately preceding ``<-``.
# Param aliases (``foo ~ bar``) also pull values into ``foo``.
_TABLE_LOADED_VIA_ARROW = re.compile(r"\b([A-Za-z_][A-Za-z_0-9]*)\s*<-")
_TABLE_LOADED_VIA_TILDE = re.compile(r"\b([A-Za-z_][A-Za-z_0-9]*)\s*~\s*[A-Za-z_]")


def _find_table_loaded_names(src: str) -> set[str]:
    """Identifiers populated by a ``table data IN`` block."""
    names: set[str] = set()
    # Restrict to the body of ``table data IN ... ;`` blocks to avoid
    # false positives (e.g. ``check`` expressions also use ``<-`` in
    # comments). Cheap approach: scan all matches but rely on the regex
    # being narrow enough — `<-` is rare in MathProg outside table reads.
    for m in _TABLE_LOADED_VIA_ARROW.finditer(src):
        names.add(m.group(1))
    for m in _TABLE_LOADED_VIA_TILDE.finditer(src):
        names.add(m.group(1))
    return names


_SET_HEADER = re.compile(
    r"""^\s*set\s+
        (?P<name>[A-Za-z_][A-Za-z_0-9]*)
        (?P<rest>.*)$""",
    re.DOTALL | re.VERBOSE,
)

_PARAM_HEADER = re.compile(
    r"""^\s*param\s+
        (?P<name>[A-Za-z_][A-Za-z_0-9]*)
        (?P<rest>.*)$""",
    re.DOTALL | re.VERBOSE,
)


def _extract_assignment(rest: str) -> tuple[str, str]:
    """Return (header, body) where header is everything before ``:=`` and
    body is everything after. If no ``:=``, body is empty.
    """
    pos = rest.find(":=")
    if pos < 0:
        return rest, ""
    return rest[:pos], rest[pos + 2:]


_DIMEN = re.compile(r"\bdimen\s+(\d+)")
_WITHIN = re.compile(r"\bwithin\s+(\{[^}]*\}|[A-Za-z_][A-Za-z_0-9]*(?:\s*,\s*[A-Za-z_][A-Za-z_0-9]*)*)")
_DEFAULT = re.compile(r"\bdefault\s+([^\s;]+)")
_PARAM_INDEX = re.compile(r"^\s*(\{[^}]*\})")


def _summarize_set_header(rest: str) -> tuple[str, str]:
    dim = _DIMEN.search(rest)
    within = _WITHIN.search(rest)
    return (dim.group(1) if dim else ""), (within.group(1).strip() if within else "")


def _summarize_param_header(rest: str) -> tuple[str, str]:
    idx = _PARAM_INDEX.match(rest.strip())
    default = _DEFAULT.search(rest)
    return (idx.group(1) if idx else ""), (default.group(1) if default else "")


_IDENT = re.compile(r"\b([A-Za-z_][A-Za-z_0-9]*)\b")


# MathProg keywords / built-ins that aren't user identifiers
_KEYWORDS = frozenset({
    "set", "param", "var", "subject", "to", "minimize", "maximize", "in",
    "within", "dimen", "default", "setof", "if", "then", "else", "end",
    "for", "while", "do", "check", "display", "printf", "let", "table",
    "data", "IN", "OUT", "and", "or", "not", "mod", "div", "by",
    "sum", "prod", "min", "max", "card", "exists", "forall",
    "true", "false", "Infinity", "abs", "ceil", "floor", "round", "sqrt",
    "log", "log10", "exp", "cos", "sin", "tan", "atan", "atan2",
    "Uniform", "Normal", "gmtime", "time2str", "str2time", "length", "substr",
    "Boolean", "Symbolic", "binary", "integer", "symbolic",
    "BV", "FX", "MI", "PL", "FR", "UP", "LO",
    "first", "last", "next", "nextw", "prev", "prevw", "ord",
    "tr", "Real", "tab",
})


def _referenced_idents(body: str, definitions: set[str]) -> tuple[set[str], bool]:
    """Identifiers referenced in body that match known set/param names.

    Returns (referenced names, references_variables flag).
    """
    refs: set[str] = set()
    refs_var = False
    for m in _IDENT.finditer(body):
        ident = m.group(1)
        if ident in _KEYWORDS:
            continue
        if any(ident.startswith(p) for p in _VARIABLE_PREFIXES):
            refs_var = True
            continue
        if ident in definitions:
            refs.add(ident)
    return refs, refs_var


def _classify_complexity(body: str) -> str:
    if not body.strip():
        return "data_loaded"
    has_if = bool(re.search(r"\bif\s+.*\bthen\b", body, re.DOTALL))
    if has_if:
        return "conditional"
    # Count membership tests in filters — proxy for "joining" complexity
    membership_count = len(re.findall(r"\b\w+\s+in\s+", body))
    if membership_count >= 3:
        return "joining"
    if "union" in body or "inter" in body or "diff" in body or "symdiff" in body:
        return "multi_clause"
    return "simple"


def build_inventory(mod_path: Path) -> list[dict]:
    raw = mod_path.read_text()
    src = _strip_comments(raw)

    # Pass 1 — collect every set/param name so we can cross-ref dependencies
    set_names: set[str] = set()
    param_names: set[str] = set()
    for _, stmt in _split_top_level_statements(src):
        m = _SET_HEADER.match(stmt)
        if m:
            set_names.add(m.group("name"))
            continue
        m = _PARAM_HEADER.match(stmt)
        if m:
            param_names.add(m.group("name"))
    all_names = set_names | param_names

    table_loaded = _find_table_loaded_names(raw)

    # Pass 2 — classify each declaration
    rows: list[dict] = []
    for line, stmt in _split_top_level_statements(src):
        m_set = _SET_HEADER.match(stmt)
        m_par = _PARAM_HEADER.match(stmt)
        if not (m_set or m_par):
            continue
        kind = "set" if m_set else "param"
        m = m_set or m_par
        name = m.group("name")
        rest = m.group("rest")
        header, body = _extract_assignment(rest)
        if kind == "set":
            dim, within_or_default = _summarize_set_header(header)
        else:
            dim, within_or_default = _summarize_param_header(header)
        refs, refs_var = _referenced_idents(body, all_names)
        # Self-references are uninteresting
        refs.discard(name)
        rows.append({
            "kind": kind,
            "name": name,
            "line": line,
            "dimen_or_index": dim,
            "within_or_default": within_or_default,
            "has_derivation": bool(body.strip()),
            "already_loaded": name in table_loaded,
            "references_variables": refs_var,
            "references": ",".join(sorted(refs)),
            "complexity": _classify_complexity(body),
            "body_chars": len(body.strip()),
        })

    # Sort by file order for stable diffing
    rows.sort(key=lambda r: r["line"])
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mod", type=Path,
                        default=Path("flextool/flextool.mod"))
    parser.add_argument("--out", type=Path,
                        default=Path("migration/inventory.csv"))
    args = parser.parse_args(argv)

    rows = build_inventory(args.mod)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Summary printed to stdout for convenience
    n_total = len(rows)
    n_sets = sum(1 for r in rows if r["kind"] == "set")
    n_params = sum(1 for r in rows if r["kind"] == "param")
    n_derived_sets = sum(1 for r in rows if r["kind"] == "set" and r["has_derivation"])
    n_derived_params = sum(1 for r in rows if r["kind"] == "param" and r["has_derivation"])
    n_var_dependent = sum(1 for r in rows if r["references_variables"])
    n_already = sum(1 for r in rows if r["already_loaded"])
    in_scope = sum(1 for r in rows
                   if r["has_derivation"]
                   and not r["references_variables"]
                   and not r["already_loaded"])

    print(f"Wrote {args.out} ({n_total} rows: {n_sets} sets, {n_params} params)")
    print(f"  derived sets:        {n_derived_sets}")
    print(f"  derived params:      {n_derived_params}")
    print(f"  reference variables: {n_var_dependent} (out of scope — must stay in .mod)")
    print(f"  already CSV-loaded:  {n_already} (out of scope — already done)")
    print(f"  IN SCOPE:            {in_scope}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
