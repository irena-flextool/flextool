"""Prove that splitting a module into several moved code and changed none of it.

The old modules are read from git (``REV:PATH``), the new ones from the working tree (files or folders)::

    python tools/split_check.py --old BASE:flextool/engine/model/core.py flextool/engine/model/core [--import-check]

(``--old`` once per old module, e.g. SPLIT-1 moved ``db_migration.py`` and ``v84_branching.py`` from 9605e9dd.)

Checked, each failure printed and counted (exit code 1 when any):

1. Every top-level statement of the old modules (definitions, assignments, ``if`` blocks) is in the new modules
   exactly once with the same ``ast.dump`` (decorators included, positions excluded).  The new modules add nothing
   else but imports, module docstrings and ``__all__``.  An import, at the top level or inside a body, is compared
   by the name it binds and the absolute target it reads (relative imports resolved against the module's package);
   an import of one of the old (new) modules themselves is compared by what it resolves to (check 2).  Module
   docstrings are not compared.
2. Every global name a moved statement reads resolves in its new module to what it resolved to in the old one:
   the same old statement (followed through imports between the new modules), the same outside object (a
   different dotted path passes when both import to the identical object), the same builtin.  Module-level code
   (assignments, decorators, default values, class bodies) sees the bindings made before it; function bodies see
   the module's last binding.  ``__name__`` / ``__file__`` / … are the module's own, so a moved reader of them
   fails (``pkg/__main__.py`` counts as ``pkg`` for ``__name__``: ``python -m pkg`` runs it as it ran the old
   ``if __name__ == '__main__'`` block).  A new module stands for the old one of its dotted name, or for the old
   module it holds all statements of.  Annotations are not reads.  A name that an old function rebinds with ``global`` is used (read, written,
   defined) in one new module only: an import elsewhere would hold a stale copy.
3. Imports and names kept: every top-level import binding of an old module is bound to the same object in some new
   module; every name of an old module's ``__all__``, and every name the other tracked modules of the repository
   take from it (``from D import x``, ``D.x``; submodules of the new package aside), resolves in the new module
   with the old module's dotted name to the same thing; a new module holding statements of an old one has its
   ``__future__`` features.
4. Order: the statements a new module takes from one old module keep their old relative order; and, simulating
   the import of the new modules (from the one with the old module's name, each in-set import executed where it
   stands, then the unreached modules in path order), the statements with import-time effects of each old module
   run in their old order.  Effect-free (order free): undecorated ``def`` with plain defaults, assignments of
   literals / names / containers / operators / builtin constructors, undecorated classes of such statements.
5. Extracted bodies: a new top-level function *g* that is not an old statement must replace a statement list of an
   old function exactly: the new function holds the call ``g(a, b)`` (a whole statement list, *g*'s parameters
   ``a, b``) where the old one held *g*'s body, statement for statement.  The body must read only its parameters and
   module-level names (no local of the old function, by Python's scope rules: comprehension variables and nested
   functions' locals are not *g*'s), assign nothing the rest of the old function reads, and hold no ``return`` /
   ``yield`` / ``break`` / ``continue`` / ``global`` that would act on the old function.  Its global reads are
   checked as in 2.
6. Every comment line of the old modules is in the new ones (extra comments are listed, not failed); a comment
   reworded on purpose is named with ``--allow-lost-comment REGEX`` (listed, not failed).
7. The top-level imports between the new modules have no cycle; with ``--import-check`` each new module imports.

Not covered: dynamic access (``globals()[...]``, ``getattr(module, name)``), names a star import provides beyond
following the named module, callers outside the new modules.
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
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_BUILTINS = set(dir(builtins))
_LIST_FIELDS = ("body", "orelse", "finalbody")
_DUNDERS = {"__name__", "__file__", "__package__", "__spec__", "__loader__", "__path__", "__doc__", "__cached__"}
_COMPS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)
_UNRESOLVED = object()


# --- modules and their top-level bindings ------------------------------------------------------------------------

@dataclass
class _Binding:
    pos: int            # index of the binding statement in the module body
    name: str
    kind: str           # "def" (a statement of the module), "mod" (import a.b), "from" (from m import x), "star"
    target: str         # "mod": the module; "from" / "star": the absolute module
    attr: str           # "from": the imported name
    node: ast.stmt


@dataclass
class _Mod:
    key: str | Path     # the old spec or the new file
    dotted: str
    is_pkg: bool
    file: str           # the path a moved reader of __file__ sees
    tree: ast.Module
    bindings: list[_Binding] = field(default_factory=list)
    future: frozenset[str] = frozenset()
    pos: dict[int, int] = field(default_factory=dict)

    @property
    def package(self) -> str:
        return self.dotted if self.is_pkg else self.dotted.rpartition(".")[0]


def _absolute(node: ast.ImportFrom, package: str) -> str:
    if not node.level:
        return node.module or ""
    parts = package.split(".") if package else []
    base = parts[:len(parts) - (node.level - 1)] if node.level - 1 <= len(parts) else []
    return ".".join(base + ([node.module] if node.module else []))


def _make_mod(key: str | Path, dotted: str, is_pkg: bool, file: str, tree: ast.Module) -> _Mod:
    mod = _Mod(key, dotted, is_pkg, file, tree)
    future: set[str] = set()
    for pos, n in enumerate(tree.body):
        mod.pos[id(n)] = pos
        if isinstance(n, ast.Import):
            for a in n.names:
                target = a.name if a.asname else a.name.split(".")[0]
                mod.bindings.append(_Binding(pos, _bound(a, plain=True), "mod", target, "", n))
        elif isinstance(n, ast.ImportFrom):
            absmod = _absolute(n, mod.package)
            if absmod == "__future__":
                future.update(a.name for a in n.names)
                continue
            for a in n.names:
                if a.name == "*":
                    mod.bindings.append(_Binding(pos, "*", "star", absmod, "", n))
                else:
                    mod.bindings.append(_Binding(pos, _bound(a), "from", absmod, a.name, n))
        else:
            for name in sorted(_binds(_scope_nodes([n])[0])):
                mod.bindings.append(_Binding(pos, name, "def", "", "", n))
    mod.future = frozenset(future)
    return mod


def _old_mod(spec: str, text: str) -> _Mod:
    path = Path(spec.split(":", 1)[-1])
    parts = list(path.with_suffix("").parts)
    is_pkg = parts[-1] == "__init__"
    return _make_mod(spec, ".".join(parts[:-1] if is_pkg else parts), is_pkg, path.as_posix(), ast.parse(text))


def _new_mod(f: Path, text: str, root: Path) -> _Mod:
    rf = f.resolve()
    file = rf.relative_to(ROOT).as_posix() if rf.is_relative_to(ROOT) else rf.as_posix()
    return _make_mod(f, _dotted(f, root), f.name == "__init__.py", file, ast.parse(text))


# --- statement comparison ----------------------------------------------------------------------------------------

class _NormImports(ast.NodeTransformer):
    """Imports compared by the name they bind and the absolute target they read.  A target among the modules of
    the same side (*inset*: old or new) is written ``<in-set>``: what it binds is compared by resolution (check 2),
    as the module paths differ by the split itself."""

    def __init__(self, package: str, inset: set[str]) -> None:
        self.package, self.inset = package, inset

    def visit_Import(self, node: ast.Import) -> ast.AST:
        return ast.Import(names=[ast.alias(name=f"{_bound(a, plain=True)}=import "
                                                f"{'<in-set>' if a.name in self.inset else a.name}")
                                 for a in node.names])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
        absmod = _absolute(node, self.package)
        return ast.Import(names=[ast.alias(name=f"{_bound(a)}=from "
                                                f"{'<in-set>' if _from_in_set(absmod, a.name, self.inset) else absmod}"
                                                f" import {a.name}")
                                 for a in node.names])


def _from_in_set(absmod: str, attr: str, inset: set[str]) -> bool:
    return absmod in inset or f"{absmod}.{attr}" in inset


def _bound(alias: ast.alias, plain: bool = False) -> str:
    if alias.asname:
        return alias.asname
    return alias.name.split(".")[0] if plain else alias.name


def _dump(node: ast.AST, package: str, inset: set[str]) -> str:
    return ast.dump(_NormImports(package, inset).visit(copy.deepcopy(node)))


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


# --- scopes (Python's rules, from the AST: symtable merges inlined comprehensions from 3.12 on) -----------------

def _scope_nodes(nodes: list[ast.AST]) -> tuple[list[ast.AST], list[ast.AST]]:
    """The nodes of *nodes* evaluated in their own scope, and the nested scopes (functions, lambdas, classes,
    comprehensions) whose defaults, decorators, bases and first iterable belong to it.  Annotations are left out."""
    own: list[ast.AST] = []
    subs: list[ast.AST] = []
    stack = list(reversed(nodes))
    while stack:
        n = stack.pop()
        if isinstance(n, (*_FUNCS, ast.Lambda)):
            if not isinstance(n, ast.Lambda):
                own.append(n)
                stack.extend(n.decorator_list)
            stack.extend(n.args.defaults + [d for d in n.args.kw_defaults if d is not None])
            subs.append(n)
        elif isinstance(n, ast.ClassDef):
            own.append(n)
            stack.extend(n.decorator_list + n.bases + [k.value for k in n.keywords])
            subs.append(n)
        elif isinstance(n, _COMPS):
            stack.append(n.generators[0].iter)
            subs.append(n)
        elif isinstance(n, ast.AnnAssign):
            own.append(n)
            stack.extend([n.target] + ([n.value] if n.value else []))
        else:
            own.append(n)
            stack.extend(reversed(list(ast.iter_child_nodes(n))))
    return own, subs


def _binds(own: list[ast.AST]) -> set[str]:
    out: set[str] = set()
    for x in own:
        if isinstance(x, ast.Name) and isinstance(x.ctx, (ast.Store, ast.Del)):
            out.add(x.id)
        elif isinstance(x, (*_FUNCS, ast.ClassDef)):
            out.add(x.name)
        elif isinstance(x, (ast.Import, ast.ImportFrom)):
            out.update(_bound(a, plain=isinstance(x, ast.Import)) for a in x.names if a.name != "*")
        elif isinstance(x, ast.ExceptHandler) and x.name:
            out.add(x.name)
        elif isinstance(x, (ast.MatchAs, ast.MatchStar)) and x.name:
            out.add(x.name)
        elif isinstance(x, ast.MatchMapping) and x.rest:
            out.add(x.rest)
    return out


@dataclass
class _Reads:
    reads: dict[str, bool] = field(default_factory=dict)    # global name -> read at import time (else call time)
    global_writes: set[str] = field(default_factory=set)
    func_locals: set[str] = field(default_factory=set)

    def add(self, name: str, import_time: bool) -> None:
        self.reads[name] = self.reads.get(name, False) or import_time


def _comp_parts(n: ast.AST) -> list[ast.AST]:
    gens = n.generators
    elts = [n.key, n.value] if isinstance(n, ast.DictComp) else [n.elt]
    return elts + [g.target for g in gens] + [i for g in gens for i in g.ifs] + [g.iter for g in gens[1:]]


def _scope(n: ast.AST, enclosing: set[str], import_time: bool, out: _Reads) -> None:
    """Record the global reads and ``global`` writes of the nested scope *n*."""
    if isinstance(n, (*_FUNCS, ast.Lambda)):
        a = n.args
        params = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs + [v for v in (a.vararg, a.kwarg) if v]}
        own, subs = _scope_nodes(n.body if isinstance(n.body, list) else [n.body])
        gdecl = {x for s in own if isinstance(s, ast.Global) for x in s.names}
        nldecl = {x for s in own if isinstance(s, ast.Nonlocal) for x in s.names}
        local = (_binds(own) | params) - gdecl - nldecl
        out.func_locals |= local
        visible, import_time = enclosing | local | nldecl, False
        for x in own:
            if isinstance(x, ast.Name):
                if isinstance(x.ctx, ast.Load) and (x.id in gdecl or x.id not in visible):
                    out.add(x.id, False)
                elif x.id in gdecl:
                    out.global_writes.add(x.id)
                    out.add(x.id, False)
    else:
        own, subs = _scope_nodes(n.body if isinstance(n, ast.ClassDef) else _comp_parts(n))
        local = _binds(own)
        for x in own:
            if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load) and x.id not in local | enclosing:
                out.add(x.id, import_time)
        visible = enclosing | local if not isinstance(n, ast.ClassDef) else enclosing
    for s in subs:
        _scope(s, visible, import_time and not isinstance(s, (*_FUNCS, ast.Lambda)), out)


def _reads(stmt: ast.stmt) -> _Reads:
    """The global names the top-level statement *stmt* reads."""
    out = _Reads()
    own, subs = _scope_nodes([stmt])
    for x in own:
        if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load):
            out.add(x.id, True)
    for s in subs:
        _scope(s, set(), not isinstance(s, (*_FUNCS, ast.Lambda)), out)
    return out


# --- resolution --------------------------------------------------------------------------------------------------

class _Resolver:
    """Canonical values of global names: ``("def", spec, i)`` for an old statement, ``("new", file, line)`` for a
    new statement that is not an old one, ``("ext", dotted)``, ``("module", dotted)`` for a module of the set,
    ``("builtin", name)``, ``("dunder", name, module)``, ``("missing", name)``."""

    def __init__(self, mods: list[_Mod], ident: dict[int, tuple[str, int]],
                 module_id: dict[str, str] | None = None) -> None:
        self.mods = {m.dotted: m for m in mods}
        self.ident = ident
        self.module_id = module_id or {d: d for d in self.mods}

    def resolve(self, mod: _Mod, name: str, before: int | None = None, seen: frozenset = frozenset()) -> tuple:
        cands = [b for b in mod.bindings if before is None or b.pos < before]
        for b in reversed(cands):
            if b.name == name:
                return self.binding(mod, b, seen)
        if name in _DUNDERS:   # ``python -m pkg`` runs pkg/__main__.py as the old module ran: one __name__ test
            return ("dunder", name, mod.file if name == "__file__" else mod.dotted.removesuffix(".__main__"))
        for b in reversed(cands):
            if b.kind == "star":
                t = self.mods.get(b.target)
                if t is None:
                    return ("ext", f"{b.target}.{name}")
                if any(x.name == name for x in t.bindings) and (t.dotted, name) not in seen:
                    return self.resolve(t, name, None, seen | {(t.dotted, name)})
        return ("builtin", name) if name in _BUILTINS else ("missing", name)

    def binding(self, mod: _Mod, b: _Binding, seen: frozenset = frozenset()) -> tuple:
        if b.kind == "def":
            if id(b.node) in self.ident:
                return ("def", *self.ident[id(b.node)])
            return ("new", str(mod.key), b.node.lineno)
        if b.kind == "mod":
            return ("module", self.module_id[b.target]) if b.target in self.mods else ("ext", b.target)
        full = f"{b.target}.{b.attr}"
        t = self.mods.get(b.target)
        if t is not None and any(x.name == b.attr for x in t.bindings) and (t.dotted, b.attr) not in seen:
            return self.resolve(t, b.attr, None, seen | {(t.dotted, b.attr)})
        if full in self.mods:
            return ("module", self.module_id[full])
        return ("ext", full) if t is None else ("missing", full)

    def nested_imports(self, node: ast.AST, mod: _Mod) -> Counter:
        """``(bound name, canonical value)`` of the imports inside *node* that read a module of the set."""
        out: Counter = Counter()
        for x in ast.walk(node):
            if isinstance(x, ast.Import):
                for a in x.names:
                    if a.name in self.mods:
                        target = a.name if a.asname else a.name.split(".")[0]
                        out[(_bound(a, plain=True), self.binding(mod, _Binding(0, "", "mod", target, "", x)))] += 1
            elif isinstance(x, ast.ImportFrom):
                absmod = _absolute(x, mod.package)
                for a in x.names:
                    if _from_in_set(absmod, a.name, set(self.mods)):
                        out[(_bound(a), self.binding(mod, _Binding(0, "", "from", absmod, a.name, x)))] += 1
        return out


_EXT_CACHE: dict[str, object] = {}


def _ext_obj(dotted: str) -> object:
    if dotted not in _EXT_CACHE:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        try:
            obj = importlib.import_module(dotted)
        except Exception:   # noqa: BLE001 — not a module: an attribute of one
            mod, _, attr = dotted.rpartition(".")
            try:
                obj = getattr(importlib.import_module(mod), attr) if mod else _UNRESOLVED
            except Exception:   # noqa: BLE001 — unresolvable: never identical
                obj = _UNRESOLVED
        _EXT_CACHE[dotted] = obj
    return _EXT_CACHE[dotted]


def _same(a: tuple, b: tuple) -> bool:
    if a == b:
        return True
    if a[0] == "ext" and b[0] == "ext":
        oa, ob = _ext_obj(a[1]), _ext_obj(b[1])
        return oa is ob and oa is not _UNRESOLVED
    return False


def _fmt(c: tuple) -> str:
    if c[0] == "def":
        return f"{c[1]} statement {c[2]}"
    if c[0] == "new":
        return f"the new statement {c[1]}:{c[2]}"
    if c[0] == "dunder":
        return f"{c[1]} of {c[2]}"
    if c[0] in ("missing", "module"):
        return f"{c[0]} {c[1]}"
    return c[1]


# --- extraction --------------------------------------------------------------------------------------------------

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
        for fld in _LIST_FIELDS:
            stmts = getattr(sub, fld, None)
            if isinstance(stmts, list):
                name = _call_target(stmts, extracted)
                if name is not None:
                    used.append(name)
                    setattr(sub, fld, copy.deepcopy(extracted[name].body))
    return node


def _loads(nodes: list[ast.AST]) -> set[str]:
    return {x.id for n in nodes for x in ast.walk(n) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)}


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
    if not isinstance(old, _FUNCS):
        errors.append(f"{', '.join(names)}: extracted from {type(old).__name__} at old line {old.lineno}, "
                      "not from a function")
        return
    old_reads = _Reads()
    _scope(old, set(), False, old_reads)
    old_locals = old_reads.func_locals
    rest_loads = _loads([new])
    for name in names:
        g = extracted[name]
        params = {a.arg for a in g.args.args}
        own, _ = _scope_nodes(g.body)
        gdecl = {x for s in own if isinstance(s, ast.Global) for x in s.names}
        stores = _binds(own) - gdecl
        free = set(_reads(g).reads) - {name}
        if free & old_locals:
            errors.append(f"{name}: reads locals of {old.name}: {sorted(free & old_locals)}")
        if (stores - params) & (rest_loads | old_locals - stores):
            clash = sorted((stores - params) & (rest_loads | old_locals - stores))
            errors.append(f"{name}: assigns names the rest of {old.name} uses: {clash}")
        if stores & params:
            errors.append(f"{name}: assigns its parameters {sorted(stores & params)}")
        for esc in _escapes(g.body):
            errors.append(f"{name}: {esc} would act on {old.name}")
        if g.decorator_list or g.args.defaults or g.args.kwonlyargs or g.args.vararg or g.args.kwarg \
                or g.args.posonlyargs:
            errors.append(f"{name}: not a plain function of positional parameters")


# --- imports between the new modules -----------------------------------------------------------------------------

def _in_set_imports(n: ast.stmt, mod: _Mod, dotted: set[str]) -> list[str]:
    """The modules of *dotted* the top-level import *n* of *mod* executes (parent packages first)."""
    if isinstance(n, ast.Import):
        targets = [a.name for a in n.names]
    elif isinstance(n, ast.ImportFrom):
        absmod = _absolute(n, mod.package)
        targets = [absmod] + [f"{absmod}.{a.name}" for a in n.names]
    else:
        return []
    out: list[str] = []
    for t in targets:
        parts = t.split(".")
        for k in range(1, len(parts) + 1):
            p = ".".join(parts[:k])
            if p in dotted and p not in out:
                out.append(p)
    return out


def _import_graph(mods: list[_Mod]) -> dict[str, set[str]]:
    dotted = {m.dotted for m in mods}
    graph: dict[str, set[str]] = defaultdict(set)
    for m in mods:
        for n in m.tree.body:
            graph[m.dotted].update(t for t in _in_set_imports(n, m, dotted)
                                   if t != m.dotted and not m.dotted.startswith(t + "."))
    return graph


def _dotted(f: Path, root: Path = ROOT) -> str:
    """The dotted module name of *f* (under *root*; elsewhere the parent folder and the file)."""
    f, root = f.resolve(), root.resolve()
    rel = f.relative_to(root) if f.is_relative_to(root) else f.relative_to(f.parents[1])
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


def _execution_order(mods: list[_Mod], first: set[str]) -> list[ast.stmt]:
    """The top-level statements of *mods* in the order an import of the package runs them."""
    by = {m.dotted: m for m in mods}
    started: set[str] = set()
    order: list[ast.stmt] = []

    def run(d: str) -> None:
        parts = d.split(".")
        for k in range(1, len(parts)):
            parent = ".".join(parts[:k])
            if parent in by and parent not in started:
                run(parent)
        if d in started:
            return
        started.add(d)
        m = by[d]
        for n in m.tree.body:
            for t in _in_set_imports(n, m, set(by)):
                if t not in started:
                    run(t)
            order.append(n)
    for m in sorted(mods, key=lambda m: (m.dotted not in first, str(m.key))):
        if m.dotted not in started:
            run(m.dotted)
    return order


# --- the check ---------------------------------------------------------------------------------------------------

def check(old_specs: list[str], new_paths: list[str], import_check: bool = False,
          allow_lost: tuple[str, ...] = ()) -> list[str]:
    """The failures of the checks of this module (empty: the split only moved code)."""
    files = _new_files(new_paths)
    olds = {spec: _git_show(spec) for spec in old_specs}
    dotteds = {_old_mod(spec, "").dotted for spec in olds}
    return check_texts(olds, {f: f.read_text(encoding="utf-8") for f in files}, import_check, allow_lost,
                       callers=_callers(dotteds, files))


def _callers(dotteds: set[str], exclude: list[Path]) -> dict[str, set[str]]:
    """The names the tracked modules of the repository other than *exclude* take from each module of *dotteds*
    (``from D import x``, ``D.x``, ``alias.x`` after ``import D as alias`` / ``from P import last``)."""
    out: dict[str, set[str]] = defaultdict(set)
    skip = {f.resolve() for f in exclude}
    listed = subprocess.run(["git", "-C", str(ROOT), "ls-files", "*.py"], check=True, capture_output=True,
                            text=True, encoding="utf-8").stdout.split()
    for rel in listed:
        f = (ROOT / rel).resolve()
        if f in skip or not f.is_file():
            continue
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        dotted = _dotted(f)
        package = dotted if f.name == "__init__.py" else dotted.rpartition(".")[0]
        alias: dict[str, str] = {}
        for x in ast.walk(tree):
            if isinstance(x, ast.ImportFrom):
                absmod = _absolute(x, package)
                for a in x.names:
                    if absmod in dotteds and a.name != "*":
                        out[absmod].add(a.name)
                    elif f"{absmod}.{a.name}" in dotteds:
                        alias[_bound(a)] = f"{absmod}.{a.name}"
            elif isinstance(x, ast.Import):
                for a in x.names:
                    if a.name in dotteds and a.asname:
                        alias[a.asname] = a.name
        for x in ast.walk(tree):
            if isinstance(x, ast.Attribute):
                if isinstance(x.value, ast.Name) and x.value.id in alias:
                    out[alias[x.value.id]].add(x.attr)
                elif isinstance(x.value, (ast.Attribute, ast.Name)) and ast.unparse(x.value) in dotteds:
                    out[ast.unparse(x.value)].add(x.attr)
    return out


def check_texts(old_texts: dict[str, str], new_texts: dict[Path, str], import_check: bool = False,
                allow_lost: tuple[str, ...] = (), root: Path = ROOT,
                callers: dict[str, set[str]] | None = None) -> list[str]:
    """:func:`check` of the old module texts ``{name: text}`` against the new ones ``{path: text}``; the new
    modules' dotted names are taken relative to *root*; *callers* ``{old dotted name: names}`` are the names other
    modules take from an old module (:func:`_callers`)."""
    errors: list[str] = []
    files = list(new_texts)
    olds = {spec: _old_mod(spec, text) for spec, text in old_texts.items()}
    news = {f: _new_mod(f, t, root) for f, t in new_texts.items()}
    old_set, new_set = {m.dotted for m in olds.values()}, {m.dotted for m in news.values()}

    old_pool: dict[str, list[tuple[str, int, ast.stmt]]] = defaultdict(list)
    old_ident: dict[int, tuple[str, int]] = {}
    old_node: dict[tuple[str, int], ast.stmt] = {}
    for spec, om in olds.items():
        for i, n in enumerate(_units(om.tree)):
            old_pool[_dump(n, om.package, old_set)].append((spec, i, n))
            old_ident[id(n)] = (spec, i)
            old_node[(spec, i)] = n
    match: dict[int, tuple[str, int]] = {}
    unmatched_new: list[tuple[Path, ast.stmt]] = []
    for f, nm in news.items():
        for n in _units(nm.tree):
            hits = old_pool.get(_dump(n, nm.package, new_set))
            if hits:
                spec, i, _ = hits.pop(0)
                match[id(n)] = (spec, i)
            else:
                unmatched_new.append((f, n))
    unmatched_old = [(spec, i, n) for hits in old_pool.values() for spec, i, n in hits]

    old_names = {n.name for _, _, n in unmatched_old if hasattr(n, "name")}
    extracted = {n.name: n for _, n in unmatched_new
                 if isinstance(n, ast.FunctionDef) and n.name not in old_names}
    extracted_file = {n.name: f for f, n in unmatched_new if getattr(n, "name", None) in extracted}
    used_all: list[str] = []
    used_by: dict[int, list[str]] = {}
    for spec, i, old in sorted(unmatched_old, key=lambda x: (x[0], x[1])):
        partner = next(((f, n) for f, n in unmatched_new if getattr(n, "name", None) == getattr(old, "name", 0)
                        and type(n) is type(old)), None)
        if partner is None:
            errors.append(f"missing from the new modules: {spec} statement {i} "
                          f"({getattr(old, 'name', type(old).__name__)}, old line {old.lineno})")
            continue
        f, new = partner
        used: list[str] = []
        if (_dump(_substitute(new, extracted, used), news[f].package, new_set)
                != _dump(old, olds[spec].package, old_set) or not used):
            errors.append(f"changed: {getattr(old, 'name', '?')} (old line {old.lineno}, now {f}:{new.lineno})")
            continue
        unmatched_new.remove(partner)
        match[id(new)] = (spec, i)
        used_by[id(new)] = used
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

    old_res = _Resolver(list(olds.values()), old_ident)
    new_res = _Resolver(list(news.values()), match, _module_ids(olds, news, match))
    _check_resolution(olds, news, match, old_node, used_by, extracted, extracted_file, old_res, new_res, errors)
    _check_kept(olds, news, match, old_res, new_res, callers or {}, errors)
    _check_order(olds, news, match, errors)

    old_comments = sum((_comments(t) for t in old_texts.values()), Counter())
    new_comments = sum((_comments(t) for t in new_texts.values()), Counter())
    allowed = [re.compile(r) for r in allow_lost]
    for c, k in (old_comments - new_comments).items():
        if any(r.search(c) for r in allowed):
            print(f"comment reworded on purpose ({k}x): {c[:100]}")
        else:
            errors.append(f"comment lost ({k}x): {c[:100]}")
    extra = new_comments - old_comments
    for cyc in _cycles(_import_graph(list(news.values()))):
        errors.append(f"import cycle: {' -> '.join(cyc)}")
    if import_check:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        for f in files:
            try:
                importlib.import_module(_dotted(f, root))
            except Exception as exc:   # noqa: BLE001 — every failure is reported
                errors.append(f"import of {_dotted(f, root)} fails: {exc!r}")

    print(f"old: {sum(len(_units(m.tree)) for m in olds.values())} top-level statements in "
          f"{len(old_texts)} module(s); new: {len(files)} module(s); extracted functions: {len(set(used_all))}")
    for f in files:
        print(f"  {_dotted(f, root)}: {new_texts[f].count(chr(10))} lines")
    if extra:
        print(f"comments added: {sum(extra.values())}")
        for c in sorted(extra):
            print(f"  + {c[:110]}")
    return errors


def _module_ids(olds: dict[str, _Mod], news: dict[Path, _Mod], match: dict[int, tuple[str, int]]) -> dict[str, str]:
    """The old module each new module stands for: the one of its dotted name, or the one it holds all of."""
    old_by_dotted = {m.dotted: m for m in olds.values()}
    out: dict[str, str] = {}
    for nm in news.values():
        specs = [match[id(n)][0] for n in _units(nm.tree) if id(n) in match]
        out[nm.dotted] = nm.dotted
        if nm.dotted not in old_by_dotted and len(set(specs)) == 1 and len(specs) == len(_units(olds[specs[0]].tree)):
            out[nm.dotted] = olds[specs[0]].dotted
    return out


def _check_resolution(olds: dict[str, _Mod], news: dict[Path, _Mod], match: dict[int, tuple[str, int]],
                      old_node: dict[tuple[str, int], ast.stmt], used_by: dict[int, list[str]],
                      extracted: dict[str, ast.FunctionDef], extracted_file: dict[str, Path],
                      old_res: _Resolver, new_res: _Resolver, errors: list[str]) -> None:
    """Check 2: the global reads of every moved statement and extracted body resolve as they did."""
    rebound: dict[str, set[str]] = defaultdict(set)     # name rebound by `global` -> new modules using it
    old_writes: set[str] = set()
    for om in olds.values():
        for n in _units(om.tree):
            old_writes |= _reads(n).global_writes

    def compare(where: str, nm: _Mod, om: _Mod, name: str, new_before: int | None,
                old_before: int | None) -> None:
        nc, oc = new_res.resolve(nm, name, new_before), old_res.resolve(om, name, old_before)
        if not _same(nc, oc):
            errors.append(f"{where}: {name} resolves to {_fmt(nc)}, was {_fmt(oc)}")

    for f, nm in news.items():
        for n in _units(nm.tree):
            if id(n) not in match:
                continue
            spec, i = match[id(n)]
            om, old = olds[spec], old_node[(spec, i)]
            where = f"{f}:{n.lineno} {getattr(n, 'name', None) or ast.unparse(n)[:40]}"
            used = used_by.get(id(n), [])
            r = _reads(n)
            for name in sorted(r.reads):
                if name in used:
                    g = extracted[name]
                    got = new_res.resolve(nm, name, None)
                    if got != ("new", str(extracted_file[name]), g.lineno):
                        errors.append(f"{where}: {name} resolves to {_fmt(got)}, not the extracted function")
                    continue
                compare(where, nm, om, name, nm.pos[id(n)] if r.reads[name] else None,
                        om.pos[id(old)] if r.reads[name] else None)
            got = new_res.nested_imports(n, nm)
            for name in used:
                got += new_res.nested_imports(extracted[name], news[extracted_file[name]])
            want = old_res.nested_imports(old, om)
            if got != want:
                errors.append(f"{where}: imports inside it bind {sorted((b, _fmt(c)) for b, c in got.elements())}, "
                              f"was {sorted((b, _fmt(c)) for b, c in want.elements())}")
            for name in (set(r.reads) | _binds(_scope_nodes([n])[0])) & old_writes:
                rebound[name].add(nm.dotted)
            for name in used:
                gf = extracted_file[name]
                for read in sorted(set(_reads(extracted[name]).reads) - {name}):
                    compare(f"{gf}:{extracted[name].lineno} {name}", news[gf], om, read, None, None)
                    if read in old_writes:
                        rebound[read].add(news[gf].dotted)
    for name, where in sorted(rebound.items()):
        if len(where) > 1:
            errors.append(f"{name} is rebound by `global` but used in several new modules: {', '.join(sorted(where))}")


def _check_kept(olds: dict[str, _Mod], news: dict[Path, _Mod], match: dict[int, tuple[str, int]],
                old_res: _Resolver, new_res: _Resolver, callers: dict[str, set[str]], errors: list[str]) -> None:
    """Check 3: old imports, the names callers and ``__all__`` take from an old module, ``__future__`` kept."""
    new_by_dotted = {m.dotted: m for m in news.values()}
    for spec, om in olds.items():
        for b in om.bindings:
            if b.kind not in ("mod", "from") or (b.kind == "from" and b.target in old_res.mods):
                continue
            want = old_res.binding(om, b)
            if not any(_same(new_res.binding(nm, x), want) for nm in news.values() for x in nm.bindings
                       if x.name == b.name and x.kind in ("mod", "from")):
                alias = next(a for a in b.node.names if _bound(a, plain=b.kind == "mod") == b.name)
                stmt = (ast.Import(names=[alias]) if b.kind == "mod" else
                        ast.ImportFrom(module=b.node.module, names=[alias], level=b.node.level))
                errors.append(f"dropped import: {spec}: {ast.unparse(stmt)}")
        nm = new_by_dotted.get(om.dotted)
        if nm is not None:
            public = set(callers.get(om.dotted, ()))
            for n in om.tree.body:
                if isinstance(n, ast.Assign) and [ast.unparse(t) for t in n.targets] == ["__all__"]:
                    try:
                        public |= set(ast.literal_eval(n.value))
                    except ValueError:
                        errors.append(f"{spec}: __all__ is not a literal")
            for name in sorted(public - _DUNDERS):
                if f"{om.dotted}.{name}" in new_res.mods:   # a submodule of the new package
                    continue
                nc, oc = new_res.resolve(nm, name), old_res.resolve(om, name)
                if not _same(nc, oc):
                    errors.append(f"kept name {name} of {om.dotted} resolves to {_fmt(nc)}, was {_fmt(oc)}")
    for f, nm in news.items():
        for spec in sorted({match[id(n)][0] for n in _units(nm.tree) if id(n) in match}):
            if nm.future != olds[spec].future:
                errors.append(f"{f}: __future__ features {sorted(nm.future)} differ from {spec}'s "
                              f"{sorted(olds[spec].future)}")


_PURE_CALLS = {"frozenset", "tuple", "dict", "set", "list", "str", "int", "float", "bool"}


def _pure(e: ast.AST | None) -> bool:
    """*e* evaluates without effects (literals, names, containers, operators, f-strings, builtin constructors)."""
    if e is None or isinstance(e, (ast.Constant, ast.Name)):
        return True
    if isinstance(e, ast.Call):
        return (isinstance(e.func, ast.Name) and e.func.id in _PURE_CALLS
                and all(_pure(a) for a in e.args) and all(_pure(k.value) for k in e.keywords))
    if isinstance(e, (ast.Attribute, ast.Subscript, ast.Tuple, ast.List, ast.Set, ast.Dict, ast.BinOp, ast.UnaryOp,
                      ast.BoolOp, ast.Compare, ast.JoinedStr, ast.FormattedValue, ast.Slice, ast.Starred,
                      ast.IfExp)):
        return all(_pure(c) for c in ast.iter_child_nodes(e) if isinstance(c, ast.expr))
    return False


def _effect_free(n: ast.stmt) -> bool:
    """*n* has no import-time effect beyond binding its names: its order against other modules is free."""
    if isinstance(n, _FUNCS):
        return not n.decorator_list and all(_pure(d) for d in n.args.defaults + n.args.kw_defaults)
    if isinstance(n, ast.Assign):
        return all(isinstance(t, ast.Name) for t in n.targets) and _pure(n.value)
    if isinstance(n, ast.AnnAssign):
        return isinstance(n.target, ast.Name) and _pure(n.value)
    if isinstance(n, ast.ClassDef):
        return (not n.decorator_list and not n.keywords and all(_pure(b) for b in n.bases)
                and all(_effect_free(x) or _is_docstring(x) or isinstance(x, ast.Pass) for x in n.body))
    return False


def _check_order(olds: dict[str, _Mod], news: dict[Path, _Mod], match: dict[int, tuple[str, int]],
                 errors: list[str]) -> None:
    """Check 4: old order within each new module, and of import-time effects along the package import."""
    for f, nm in news.items():
        by_spec: dict[str, list[int]] = defaultdict(list)
        for n in _units(nm.tree):
            if id(n) in match:
                spec, i = match[id(n)]
                by_spec[spec].append(i)
        for spec, idx in by_spec.items():
            if idx != sorted(idx):
                errors.append(f"{f}: statements from {spec} out of their old order")
    seq: dict[str, list[int]] = defaultdict(list)
    for n in _execution_order(list(news.values()), {m.dotted for m in olds.values()}):
        if id(n) in match and not _effect_free(n):
            spec, i = match[id(n)]
            seq[spec].append(i)
    for spec, idx in sorted(seq.items()):
        bad = next(((a, b) for a, b in zip(idx, idx[1:]) if b < a), None)
        if bad:
            errors.append(f"statements with import-time effects of {spec} run out of their old order: "
                          f"statement {bad[0]} before {bad[1]}")


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
