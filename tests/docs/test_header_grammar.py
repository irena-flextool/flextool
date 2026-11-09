"""The EQ. / VAR. / OBJ. header grammar (docs/dev/engine.md "Equation headers") and its ratchets.

The odd header forms parse to the names they stand for; every LP row and column the model emits sits under a
header (or in a row helper whose callers have one); every autoscale family key is a header name; every
objective step has an OBJ. header; the formulation page renders.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from flextool.docs_build import formulation, render_page, scope
from flextool.docs_build.formulation import HeaderGrammarError, expand_names, parse_headers
from flextool.engine.autoscale._layer2_types import (
    CONSTRAINT_FAMILIES,
    VARIABLE_FAMILIES,
    lookup_cstr,
    lookup_var,
)

REPO = Path(__file__).resolve().parents[2]

# Generic row helpers: they carry no header of their own; every call site of them must have one.
ROW_HELPERS: frozenset[tuple[str, str]] = frozenset({
    ("_coarse_flows", "add_cstr"),
    ("_coarse_flows", "add_twin"),
    ("_coarse_flows", "add_row_rule_cstr"),
    ("_lead_time", "period_limit"),
    ("core", "_ramp_family"),
    ("_cumulative_invest", "_group_period_limit_online"),
    ("_online", "_add_max_input_flow_online"),
})

# CONSTRAINT_FAMILIES keys that are no header name: lookup_cstr reaches them from a header name by suffix
# (_linear / _integer, _p / _n) or by prefix (ramp_<side>_<dir>_constraint).
PREFIX_REGISTERED: frozenset[str] = frozenset({
    "connection_reserve_down_online", "connection_reserve_up_online",
    "maxCumulative_capacity", "minCumulative_capacity",
    "maxDivestGroup_entity_period", "maxDivestGroup_entity_total", "maxDivestGroup_entity_total_path",
    "maxInvestGroup_entity_period", "maxInvestGroup_entity_total", "maxInvestGroup_entity_total_path",
    "minDivestGroup_entity_period", "minDivestGroup_entity_total_path",
    "minInvestGroup_entity_period", "minInvestGroup_entity_total_path",
    "minDivest_entity_period", "minDivest_entity_total_path",
    "minInvest_entity_period", "minInvest_entity_total_path",
    "maxFlow_back_online", "maxFlow_online", "maxFlow_online_shutdown", "maxFlow_online_shutdown_branch",
    "maxInputFlow_online", "maxInputFlow_online_shutdown", "maxInputFlow_online_shutdown_branch",
    "maxOnline", "maxOutputSum_online", "maxShutdown", "maxStartup",
    "minFlow_minload", "minFlow_output_floor",
    "minimum_downtime", "minimum_uptime", "online_change", "online_fix_eq",
    "ramp",
})


def _headers_of(snippet: str) -> list:
    return parse_headers(snippet, "snippet.py", "snippet")


# ---------------------------------------------------------------------------
# The odd forms


def test_continuation_label_semicolon_and_placeholder() -> None:
    """The coarse-family form: continuation lines, a `<family>_coarse:` label, `;`, `{sfx}` and a qualifier
    holding ` / ` (`_coarse_flows.py`)."""
    snippet = (
        "    # ─── EQ. ─── <family>_coarse: maxFlow_coarse, maxFlow_online{sfx}_coarse,\n"
        "    #             profile_flow_{upper_limit,fixed}_coarse; by the row rule\n"
        "    #             (add_row_rule_cstr): dc_flow_eq_coarse   ({sfx} = _linear / _integer) ──────\n"
        "    # Σ_{t∈b} w · ( LHS_t − RHS_t )  {≤,=,≥}  E\n"
        "    #   ∀ coarse row key K\n"
        "    # The fine row's Δt-weighted average over the coarse step.\n"
        "x = 1\n"
    )
    (h,) = _headers_of(snippet)
    assert h.kind == "EQ"
    assert h.names == ("maxFlow_coarse", "maxFlow_online_linear_coarse", "maxFlow_online_integer_coarse",
                       "profile_flow_upper_limit_coarse", "profile_flow_fixed_coarse", "dc_flow_eq_coarse")
    assert h.formula == "Σ_{t∈b} w · ( LHS_t − RHS_t )  {≤,=,≥}  E\n  ∀ coarse row key K"
    assert h.notes == "The fine row's Δt-weighted average over the coarse step."


def test_the_real_coarse_header() -> None:
    """The header of `_coarse_flows.add_cstr` names the row-rule families too, and nothing that is no name."""
    (h,) = [h for h in formulation.all_headers() if h.module == "_coarse_flows"]
    for name in ("maxFlow_coarse", "maxInputFlow_online_shutdown_branch_integer_coarse",
                 "connection_reserve_down_online_linear_coarse", "dc_flow_eq_coarse",
                 "non_anticipativity_storage_use_coarse", "process_constraint_greater_than_coarse"):
        assert name in h.names
    assert h.function == "add_cstr"


def test_qualifier_is_dropped() -> None:
    (h,) = _headers_of("# ─── VAR. ─── v_flow (coarse processes) ─────\n# v_flow[p,s,k,d,b] ≥ 0\n")
    assert h.names == ("v_flow",)
    assert h.title == "v_flow (coarse processes)"


def test_list_with_trailing_qualifier() -> None:
    (h,) = _headers_of("# ─── VAR. ─── vq_inertia, vq_non_synchronous (coarse rows) ───\n# vq[g,d,b] ≥ 0\n")
    assert h.names == ("vq_inertia", "vq_non_synchronous")


def test_free_text_header_takes_the_names_line() -> None:
    snippet = (
        "# ─── OBJ. ─── reserve shortfall ───────\n"
        "# names: reserve_shortfall_penalty\n"
        "# ─────────────────────────────────────\n"
        "# + Σ vq_reserve · scale · penalty_reserve · w_op\n"
        "# Prices the shortfall in MW.\n"
    )
    (h,) = _headers_of(snippet)
    assert (h.kind, h.title, h.names) == ("OBJ", "reserve shortfall", ("reserve_shortfall_penalty",))
    assert h.formula == "+ Σ vq_reserve · scale · penalty_reserve · w_op"


def test_brackets_and_lists_expand() -> None:
    assert expand_names("max{Invest,Divest}_entity_total[_path][_n]") == (
        "maxInvest_entity_total", "maxInvest_entity_total_n", "maxInvest_entity_total_path",
        "maxInvest_entity_total_path_n", "maxDivest_entity_total", "maxDivest_entity_total_n",
        "maxDivest_entity_total_path", "maxDivest_entity_total_path_n")


@pytest.mark.parametrize("snippet, message", [
    ("# ─── OBJ. ─── reserve shortfall ───\n# + Σ x\n", "names: a, b"),
    ("# ─── EQ. ─── maxFlow,\n# formula\n", "do not end in '─'"),
    ("# ─── EQ. ─── maxFlow{tail} ───\n", "declared placeholder"),
    ("# ─── EQN. ─── maxFlow ───\n", "unknown header kind"),
])
def test_grammar_errors_name_the_line(snippet: str, message: str) -> None:
    with pytest.raises(HeaderGrammarError, match="snippet.py:1") as err:
        _headers_of(snippet)
    assert message in str(err.value)


# ---------------------------------------------------------------------------
# Ratchets over the installed code


def _sources() -> list[tuple[str, Path]]:
    files = [(p.stem, p) for p in sorted(formulation.MODEL_DIR.glob("*.py")) if p.stem != "__init__"]
    return files + [("_benders", formulation.BENDERS_FILE)]


def _functions(tree: ast.AST) -> list[ast.FunctionDef]:
    return [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _innermost(funcs: list[ast.FunctionDef], line: int) -> ast.FunctionDef | None:
    best = None
    for f in funcs:
        if f.lineno <= line <= (f.end_lineno or f.lineno) and (best is None or f.lineno > best.lineno):
            best = f
    return best


def _row_calls(module: str, tree: ast.AST):
    """(call, kind) for every LP row / column call: direct add_cstr / add_var, and row-helper calls."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and (fn.value.id, fn.attr) in ROW_HELPERS:
            yield node, "EQ", (fn.value.id, fn.attr)
        elif isinstance(fn, ast.Name) and (module, fn.id) in ROW_HELPERS:
            yield node, "EQ", (module, fn.id)
        elif isinstance(fn, ast.Attribute) and fn.attr in ("add_cstr", "add_var"):
            yield node, ("EQ" if fn.attr == "add_cstr" else "VAR"), None


def _literal_name(call: ast.Call, helper: tuple[str, str] | None) -> str | None:
    """The row / column name a call passes as a string literal (``add_cstr`` / ``add_var``: the first positional
    argument; a row helper: the one after the builder; either: ``name=``), ``None`` when it is computed."""
    position = 0 if helper is None else 1
    node = call.args[position] if len(call.args) > position else next(
        (k.value for k in call.keywords if k.arg == "name"), None)
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _row_issues(module: str, file: str, source: str) -> tuple[list[str], list[str], dict[tuple[str, str], int]]:
    """(rows without a header, literal names that are no header name, row-helper call counts) of one source.

    A row needs an EQ. / VAR. header earlier in its function (or sits in a row helper).  Its literal name must be
    one of the names of those headers (an ``add_twin`` name, written without ``_coarse``, as is)."""
    lines = source.splitlines()
    tree = ast.parse(source)
    funcs = _functions(tree)
    headers = parse_headers(source, file, module)
    missing: list[str] = []
    unnamed: list[str] = []
    helper_calls: dict[tuple[str, str], int] = {h: 0 for h in ROW_HELPERS}
    for call, kind, helper in _row_calls(module, tree):
        if helper is not None:
            helper_calls[helper] += 1
        fn = _innermost(funcs, call.lineno)
        if fn is not None and (module, fn.name) in ROW_HELPERS:
            continue
        start = fn.lineno if fn is not None else 1
        above = [h for h in headers if h.kind == kind and start <= h.line < call.lineno]
        where = f"{file}:{call.lineno} {lines[call.lineno - 1].strip()[:80]}"
        if not above:
            missing.append(where)
            continue
        name = _literal_name(call, helper)
        if name is not None and not any(name in h.names for h in above):
            unnamed.append(f"{where}: {name!r} is not named by the {kind}. header(s) above it in the function")
    return missing, unnamed, helper_calls


def test_every_row_and_column_has_a_header() -> None:
    """Ratchet (a): an add_cstr / add_var has an EQ. / VAR. header earlier in its function, or sits in a row
    helper whose every call site has one; every literal row / column name is a name of such a header (review
    C2-7: one header used to cover every later call of its function, whatever the call's name)."""
    missing: list[str] = []
    unnamed: list[str] = []
    helper_calls: dict[tuple[str, str], int] = {h: 0 for h in ROW_HELPERS}
    for module, path in _sources():
        m, u, calls = _row_issues(module, path.name, path.read_text(encoding="utf-8"))
        missing += m
        unnamed += u
        for helper, n in calls.items():
            helper_calls[helper] += n
    assert not missing, "LP rows / columns without an EQ. / VAR. header in their function:\n" + "\n".join(missing)
    assert not unnamed, "LP row / column names that their header does not name:\n" + "\n".join(unnamed)
    unused = sorted(f"{m}.{f}" for (m, f), n in helper_calls.items() if n == 0)
    assert not unused, f"ROW_HELPERS entries with no call site (stale): {unused}"


def test_a_row_named_by_no_header_above_it_is_reported() -> None:
    """A new family added to a function that already has a header, without a header of its own (C2-7)."""
    snippet = (
        "def eqs(m):\n"
        "    # ─── EQ. ─── maxFlow ───\n"
        "    # v_flow ≤ cap\n"
        "    m.add_cstr('maxFlow', over=None)\n"
        "    m.add_cstr(name='newFamily', over=None)\n"
        "    _coarse_flows.add_cstr(b, 'otherFamily', over=None)\n"
        "    m.add_cstr(f'maxFlow{sfx}', over=None)\n"
    )
    missing, unnamed, _ = _row_issues("snippet", "snippet.py", snippet)
    assert missing == []
    assert [u.split(" ", 1)[0] for u in unnamed] == ["snippet.py:5", "snippet.py:6"]
    assert "'newFamily'" in unnamed[0] and "'otherFamily'" in unnamed[1]


# A docstring that starts with a names list (``name, {a,b}_c[_d]: what it does``).
_DOC_NAMES = re.compile(r"^([\w{}\[\],]+(?:\s*/\s*[\w{}\[\],]+)*):\s")


def test_a_docstring_names_only_its_functions_header_names() -> None:
    """A function with EQ. / VAR. headers whose docstring starts with a names list names nothing its headers do
    not (review C2-11: two docstrings still named minInvestGroup_entity_total_{p,n})."""
    stale: list[str] = []
    for module, path in _sources():
        source = path.read_text(encoding="utf-8")
        headers = parse_headers(source, path.name, module)
        for fn in _functions(ast.parse(source)):
            names = {n for h in headers if h.kind in ("EQ", "VAR") and fn.lineno <= h.line <= (fn.end_lineno or 0)
                     for n in h.names}
            match = _DOC_NAMES.match(ast.get_docstring(fn) or "")
            if names and match:
                extra = sorted(set(expand_names(match.group(1))) - names)
                if extra:
                    stale.append(f"{path.name}:{fn.lineno} {fn.name}: {extra}")
    assert not stale, "docstrings naming what no header of their function names:\n" + "\n".join(stale)


def test_every_autoscale_family_is_a_header_name() -> None:
    """Ratchet (b): every CONSTRAINT_FAMILIES / VARIABLE_FAMILIES key is a header name, the `_coarse` twin of
    one, or in PREFIX_REGISTERED; every header name resolves in the registry."""
    headers = formulation.all_headers()
    eq = {n for h in headers if h.kind == "EQ" for n in h.names}
    var = {n for h in headers if h.kind == "VAR" for n in h.names}

    def named(key: str, names: set[str]) -> bool:
        return key in names or (key.endswith("_coarse") and key[: -len("_coarse")] in names)

    unnamed = sorted(k for k in CONSTRAINT_FAMILIES if not named(k, eq) and k not in PREFIX_REGISTERED)
    assert not unnamed, f"CONSTRAINT_FAMILIES keys that no EQ. header names: {unnamed}"
    stale = sorted(k for k in PREFIX_REGISTERED if k not in CONSTRAINT_FAMILIES or named(k, eq))
    assert not stale, f"PREFIX_REGISTERED entries that are no registry key or are a header name: {stale}"
    unreached = sorted(k for k in PREFIX_REGISTERED
                       if not any(n != k and lookup_cstr(n) is CONSTRAINT_FAMILIES[k] for n in eq))
    assert not unreached, f"PREFIX_REGISTERED keys that lookup_cstr reaches from no header name: {unreached}"
    unnamed_vars = sorted(k for k in VARIABLE_FAMILIES if not named(k, var))
    assert not unnamed_vars, f"VARIABLE_FAMILIES keys that no VAR. header names: {unnamed_vars}"

    unregistered: list[str] = []
    for n in sorted(eq):
        try:
            lookup_cstr(n)
        except KeyError:
            unregistered.append(n)
    for n in sorted(var):
        try:
            lookup_var(n)
        except KeyError:
            unregistered.append(n)
    assert not unregistered, f"header names without an autoscale family: {unregistered}"


def test_every_objective_step_has_an_obj_header() -> None:
    """Ratchet (c): every `_obj_*` step and every `objective_terms` has an OBJ. header in its body."""
    missing: list[str] = []
    for module, path in _sources():
        if module == "_benders":
            continue
        source = path.read_text(encoding="utf-8")
        headers = parse_headers(source, path.name, module)
        for fn in _functions(ast.parse(source)):
            if fn.name.startswith("_obj_") or fn.name == "objective_terms":
                end = fn.end_lineno or fn.lineno
                if not any(h.kind == "OBJ" and fn.lineno <= h.line <= end for h in headers):
                    missing.append(f"{path.name}:{fn.lineno} {fn.name}")
    assert not missing, "objective steps without an OBJ. header:\n" + "\n".join(missing)


def test_the_formulation_page_renders() -> None:
    """Ratchet (d): the generator runs over the installed code; every module with headers has its section and
    every header its source line."""
    page = render_page(scope.FORMULATION_PAGE, REPO / "docs")
    assert page.startswith("# Model formulation\n")
    assert f"## {scope.FORMULATION_PREAMBLE_SECTION}\n" in page
    assert f"## {formulation.BENDERS_TITLE}\n" in page
    for h in formulation.all_headers():
        assert f"`{h.file}:{h.line}`" in page
    order = [m for m in formulation.module_order()]
    assert order[0] == "core" and order.index("_online") < order.index("_non_anticipativity")
