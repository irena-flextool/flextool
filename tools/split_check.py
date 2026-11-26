"""Prove that splitting a module into several moved code and changed none of it.

The old modules are read from git (``REV:PATH``), the new ones from the working tree (files or folders)::

    python tools/split_check.py --old 503b8270:flextool/update_flextool/db_migration.py \\
        flextool/update_flextool/db_migration [--import-check]

Checked, each failure printed and counted (exit code 1 when any):

1. Every top-level statement of the old modules (definitions, assignments, ``if`` blocks) is in the new modules
   exactly once with the same ``ast.dump`` (decorators included, positions excluded).  The new modules add nothing
   else but imports, module docstrings and ``__all__``.  An import, at the top level or inside a body, is compared
   by the names it binds, not by the module it reads them from.  Module docstrings are not compared.
2. The statements a new module takes from one old module keep their old relative order (import-time references
   between module-level statements keep working).
3. Extracted bodies: a new top-level function *g* that is not an old statement must replace a statement list of an
   old function exactly: the new function holds the call ``g(a, b)`` (a whole statement list, *g*'s parameters
   ``a, b``) where the old one held *g*'s body, statement for statement.  The body must read only its parameters and
   module-level names (no local of the old function), assign nothing the rest of the old function reads, and hold no
   ``return`` / ``yield`` / ``break`` / ``continue`` / ``global`` that would act on the old function.
4. Every comment line of the old modules is in the new ones (extra comments are listed, not failed); a comment
   reworded on purpose is named with ``--allow-lost-comment REGEX`` (listed, not failed).
5. The top-level imports between the new modules have no cycle; with ``--import-check`` each new module imports.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import copy
import importlib
import io
import re
import subprocess
import sys
import tokenize
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_BUILTINS = set(dir(builtins))
_LIST_FIELDS = ("body", "orelse", "finalbody")


class _NormImports(ast.NodeTransformer):
    """Imports compared by the names they bind."""

    def visit_Import(self, node: ast.Import) -> ast.AST:
        return ast.Import(names=[ast.alias(name=_bound(a, plain=True)) for a in node.names])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
        return ast.Import(names=[ast.alias(name=_bound(a)) for a in node.names])


def _bound(alias: ast.alias, plain: bool = False) -> str:
    if alias.asname:
        return alias.asname
    return alias.name.split(".")[0] if plain else alias.name


def _dump(node: ast.AST) -> str:
    return ast.dump(_NormImports().visit(copy.deepcopy(node)))


def _is_docstring(node: ast.stmt) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def _units(tree: ast.Module) -> list[ast.stmt]:
    """Top-level statements other than imports and the module docstring."""
    body = tree.body[1:] if tree.body and _is_docstring(tree.body[0]) else tree.body
    return [n for n in body if not isinstance(n, (ast.Import, ast.ImportFrom))]


def _comments(text: str) -> Counter:
    out: Counter = Counter()
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT:
            out[tok.string.strip()] += 1
    return out


def _git_show(spec: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), "show", spec], check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout


def _new_files(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        path = (ROOT / p) if not Path(p).is_absolute() else Path(p)
        out.extend(sorted(path.rglob("*.py")) if path.is_dir() else [path])
    return out


def _call_target(stmts: list[ast.stmt], extracted: dict[str, ast.FunctionDef]) -> str | None:
    """The extracted function *stmts* is exactly a call of (with its parameter names as arguments), else None."""
    if len(stmts) != 1 or not isinstance(stmts[0], ast.Expr) or not isinstance(stmts[0].value, ast.Call):
        return None
    call = stmts[0].value
    if not isinstance(call.func, ast.Name) or call.func.id not in extracted or call.keywords:
        return None
    params = [a.arg for a in extracted[call.func.id].args.args]
    if [a.id if isinstance(a, ast.Name) else None for a in call.args] != params:
        return None
    return call.func.id


def _substitute(node: ast.AST, extracted: dict[str, ast.FunctionDef], used: list[str]) -> ast.AST:
    """*node* with each statement list that calls an extracted function replaced by that function's body."""
    node = copy.deepcopy(node)
    for sub in ast.walk(node):
        for field in _LIST_FIELDS:
            stmts = getattr(sub, field, None)
            if isinstance(stmts, list):
                name = _call_target(stmts, extracted)
                if name is not None:
                    used.append(name)
                    setattr(sub, field, copy.deepcopy(extracted[name].body))
    return node


def _names(nodes: list[ast.AST], ctx: type) -> set[str]:
    out = set()
    for n in nodes:
        for x in ast.walk(n):
            if isinstance(x, ast.Name) and isinstance(x.ctx, ctx):
                out.add(x.id)
            if ctx is ast.Store:
                if isinstance(x, ast.arg):
                    out.add(x.arg)
                elif isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    out.add(x.name)
                elif isinstance(x, (ast.Import, ast.ImportFrom)):
                    out.update(_bound(a, plain=isinstance(x, ast.Import)) for a in x.names)
                elif isinstance(x, ast.ExceptHandler) and x.name:
                    out.add(x.name)
    return out


def _escapes(stmts: list[ast.stmt]) -> list[str]:
    """``return`` / ``yield`` / ``break`` / ``continue`` / ``global`` of *stmts* that act outside them."""
    found: list[str] = []

    def walk(node: ast.AST, in_loop: bool) -> None:
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                continue
            if isinstance(ch, (ast.Return, ast.Yield, ast.YieldFrom, ast.Await, ast.Global, ast.Nonlocal)):
                found.append(f"{type(ch).__name__} at line {ch.lineno}")
            if isinstance(ch, (ast.Break, ast.Continue)) and not in_loop:
                found.append(f"{type(ch).__name__} at line {ch.lineno}")
            walk(ch, in_loop or isinstance(ch, (ast.For, ast.AsyncFor, ast.While)))
    walk(ast.Module(body=stmts, type_ignores=[]), False)
    return found


def _check_extraction(old: ast.stmt, new: ast.stmt, names: list[str], extracted: dict[str, ast.FunctionDef],
                      errors: list[str]) -> None:
    old_locals = _names([old], ast.Store)
    if isinstance(old, (ast.FunctionDef, ast.AsyncFunctionDef)):
        old_locals |= {a.arg for a in old.args.args}
    rest_loads = _names([new], ast.Load)
    for name in names:
        g = extracted[name]
        params = {a.arg for a in g.args.args}
        stores = _names(g.body, ast.Store)
        free = _names(g.body, ast.Load) - stores - params - _BUILTINS
        if free & old_locals:
            errors.append(f"{name}: reads locals of {old.name}: {sorted(free & old_locals)}")
        if (stores - params) & (rest_loads | old_locals - stores):
            clash = sorted((stores - params) & (rest_loads | old_locals - stores))
            errors.append(f"{name}: assigns names the rest of {old.name} uses: {clash}")
        if stores & params:
            errors.append(f"{name}: assigns its parameters {sorted(stores & params)}")
        for esc in _escapes(g.body):
            errors.append(f"{name}: {esc} would act on {old.name}")
        if g.decorator_list or g.args.defaults or g.args.kwonlyargs or g.args.vararg or g.args.kwarg:
            errors.append(f"{name}: not a plain function of positional parameters")


def _import_graph(files: list[Path], trees: dict[Path, ast.Module]) -> dict[str, set[str]]:
    mods = {_dotted(f): f for f in files}
    graph: dict[str, set[str]] = defaultdict(set)
    for mod, f in mods.items():
        for n in trees[f].body:
            targets: list[str] = []
            if isinstance(n, ast.Import):
                targets = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module:
                targets = [n.module] + [f"{n.module}.{a.name}" for a in n.names]
            graph[mod].update(t for t in targets if t in mods and t != mod and not mod.startswith(t + "."))
    return graph


def _dotted(f: Path) -> str:
    """The dotted module name of *f* (under the repository root; elsewhere the parent folder and the file)."""
    f = f.resolve()
    rel = f.relative_to(ROOT) if f.is_relative_to(ROOT) else f.relative_to(f.parents[1])
    parts = list(rel.with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    found, state = [], {}

    def visit(node: str, path: list[str]) -> None:
        state[node] = 1
        for nxt in sorted(graph.get(node, ())):
            if state.get(nxt) == 1:
                found.append(path[path.index(nxt):] + [nxt] if nxt in path else [node, nxt])
            elif nxt not in state:
                visit(nxt, path + [nxt])
        state[node] = 2
    for node in sorted(graph):
        if node not in state:
            visit(node, [node])
    return found


def check(old_specs: list[str], new_paths: list[str], import_check: bool = False,
          allow_lost: tuple[str, ...] = ()) -> list[str]:
    """The failures of the checks of this module (empty: the split only moved code)."""
    files = _new_files(new_paths)
    return check_texts({spec: _git_show(spec) for spec in old_specs},
                       {f: f.read_text(encoding="utf-8") for f in files}, import_check, allow_lost)


def check_texts(old_texts: dict[str, str], new_texts: dict[Path, str], import_check: bool = False,
                allow_lost: tuple[str, ...] = ()) -> list[str]:
    """:func:`check` of the old module texts ``{name: text}`` against the new ones ``{path: text}``."""
    errors: list[str] = []
    files = list(new_texts)
    trees = {f: ast.parse(t) for f, t in new_texts.items()}

    old_pool: dict[str, list[tuple[str, int, ast.stmt]]] = defaultdict(list)
    for spec, text in old_texts.items():
        for i, n in enumerate(_units(ast.parse(text))):
            old_pool[_dump(n)].append((spec, i, n))
    match: dict[int, tuple[str, int]] = {}
    unmatched_new: list[tuple[Path, ast.stmt]] = []
    for f, tree in trees.items():
        for n in _units(tree):
            hits = old_pool.get(_dump(n))
            if hits:
                spec, i, _ = hits.pop(0)
                match[id(n)] = (spec, i)
            else:
                unmatched_new.append((f, n))
    unmatched_old = [(spec, i, n) for hits in old_pool.values() for spec, i, n in hits]

    old_names = {n.name for _, _, n in unmatched_old if hasattr(n, "name")}
    extracted = {n.name: n for _, n in unmatched_new
                 if isinstance(n, ast.FunctionDef) and n.name not in old_names}
    used_all: list[str] = []
    for spec, i, old in sorted(unmatched_old, key=lambda x: (x[0], x[1])):
        partner = next(((f, n) for f, n in unmatched_new if getattr(n, "name", None) == getattr(old, "name", 0)
                        and type(n) is type(old)), None)
        if partner is None:
            errors.append(f"missing from the new modules: {spec} statement {i} "
                          f"({getattr(old, 'name', type(old).__name__)}, old line {old.lineno})")
            continue
        f, new = partner
        used: list[str] = []
        if _dump(_substitute(new, extracted, used)) != _dump(old) or not used:
            errors.append(f"changed: {getattr(old, 'name', '?')} (old line {old.lineno}, now {f}:{new.lineno})")
            continue
        unmatched_new.remove(partner)
        match[id(new)] = (spec, i)
        _check_extraction(old, new, used, extracted, errors)
        used_all += used
    for name, count in Counter(used_all).items():
        if count > 1:
            errors.append(f"{name}: replaces {count} bodies")
    for f, n in unmatched_new:
        name = getattr(n, "name", None)
        if name in extracted and name in used_all:
            continue
        if isinstance(n, ast.Assign) and [ast.unparse(t) for t in n.targets] == ["__all__"]:
            continue
        errors.append(f"new statement: {f}:{n.lineno} {name or ast.unparse(n)[:80]}")

    for f, tree in trees.items():
        by_spec: dict[str, list[int]] = defaultdict(list)
        for n in _units(tree):
            if id(n) in match:
                spec, i = match[id(n)]
                by_spec[spec].append(i)
        for spec, idx in by_spec.items():
            if idx != sorted(idx):
                errors.append(f"{f}: statements from {spec} out of their old order")

    old_comments = sum((_comments(t) for t in old_texts.values()), Counter())
    new_comments = sum((_comments(t) for t in new_texts.values()), Counter())
    allowed = [re.compile(r) for r in allow_lost]
    for c, k in (old_comments - new_comments).items():
        if any(r.search(c) for r in allowed):
            print(f"comment reworded on purpose ({k}x): {c[:100]}")
        else:
            errors.append(f"comment lost ({k}x): {c[:100]}")
    extra = new_comments - old_comments
    for cyc in _cycles(_import_graph(files, trees)):
        errors.append(f"import cycle: {' -> '.join(cyc)}")
    if import_check:
        sys.path.insert(0, str(ROOT))
        for f in files:
            try:
                importlib.import_module(_dotted(f))
            except Exception as exc:   # noqa: BLE001 — every failure is reported
                errors.append(f"import of {_dotted(f)} fails: {exc!r}")

    print(f"old: {sum(len(_units(ast.parse(t))) for t in old_texts.values())} top-level statements in "
          f"{len(old_texts)} module(s); new: {len(files)} module(s); extracted functions: {len(set(used_all))}")
    for f in files:
        print(f"  {_dotted(f)}: {new_texts[f].count(chr(10))} lines")
    if extra:
        print(f"comments added: {sum(extra.values())}")
        for c in sorted(extra):
            print(f"  + {c[:110]}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--old", action="append", required=True, help="REV:PATH of an old module (repeatable)")
    parser.add_argument("new", nargs="+", help="new module files or folders")
    parser.add_argument("--import-check", action="store_true", help="also import every new module")
    parser.add_argument("--allow-lost-comment", action="append", default=[], metavar="REGEX",
                        help="an old comment matching REGEX may be missing (reworded on purpose; repeatable)")
    args = parser.parse_args(argv)
    errors = check(args.old, args.new, args.import_check, tuple(args.allow_lost_comment))
    for e in errors:
        print(f"FAIL {e}")
    print("split check: " + ("OK" if not errors else f"{len(errors)} failure(s)"))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
