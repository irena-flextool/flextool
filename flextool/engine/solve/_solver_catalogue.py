"""FlexTool's list of each solver's parameters (item 240 B1).

``solver_catalogue/<solver>.json`` (package data) lists every parameter of HiGHS, Gurobi, CPLEX, Xpress and COPT
that FlexTool knows, keyed as the solver reads the name (:func:`catalogue_key`), with its type and a class that
says what the input database may do with it:

* ``plain``: taken (after the general check of :func:`._solver_dispatch.validate_solver_option`);
* ``file``: a number, a switch or a plain file name, placed in the solve's work folder;
* ``local_only``: refused from the database; ``reason`` says why: ``remote`` (another computer, a licence service
  or its credentials), ``unreviewed`` (a text parameter nobody has classified) or ``command`` (a command of the
  solver's console, not a parameter);
* ``flextool``: FlexTool's own (Gurobi ``ResultFile``), dropped from every layer.

The files are written by ``tools/solver_catalogue.py`` from the solver packages, with a reviewed class for every
name (``tools/solver_catalogue_review.json``); names are only ever added, so a database written for an older
solver keeps passing.  The name rules below are shared with the generator: a name that matches one is never
``plain`` unless a review says why.

No solver package is imported here, except ``highspy`` for :func:`highs_live_type` (the running HiGHS is the solver
FlexTool uses, so a name it knows that the list lacks can be looked up in it).
"""
from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping

SOLVERS = ("highs", "gurobi", "cplex", "xpress", "copt")
#: The solver's name as its makers write it, for messages.
SOLVER_LABEL = {"highs": "HiGHS", "gurobi": "Gurobi", "cplex": "CPLEX", "xpress": "Xpress", "copt": "COPT"}
CLASSES = ("plain", "file", "local_only", "flextool")
REASONS = ("remote", "unreviewed", "command")
TYPES = ("bool", "int", "double", "string", "none", "unknown")
#: The catalogue files, inside the ``flextool`` package.
CATALOGUE_FOLDER = "engine/solve/solver_catalogue"
FORMAT = 1

# ---------------------------------------------------------------------------
# Name rules (every type: an integer can switch on remote solving, Xpress ``COMPUTE``)
# ---------------------------------------------------------------------------

#: Words in an option name that mark a file, folder or path option (``LogFile``, ``workdir``, ``TUNEROUTPUTPATH``).
PATH_NAME_WORDS = ("file", "dir", "path")
#: CPLEX: a name whose first word is one of these is a file option too (the ``read`` / ``write`` commands).
CPLEX_PATH_FIRST_WORDS = frozenset({"read", "write"})
#: HiGHS: a name with one of these prefixes reads or writes a file.
HIGHS_PATH_PREFIXES = ("read_", "write_")
#: A name with one of these prefixes connects the solver to another machine (Gurobi: Compute Server, Cluster
#: Manager, Instant Cloud, Web License Service, token server, distributed workers; Xpress: remote solving through
#: the Insight Compute interface).
SERVER_PREFIXES: dict[str, tuple[str, ...]] = {
    "gurobi": ("computeserver", "server", "cs", "cloud", "wls", "token", "worker"),
    "xpress": ("compute",),
}


def commercial_option_key(solver: str, name: str) -> str:
    """The comparison key of a commercial parameter name: one key for every spelling the solver reads as the
    same parameter, so that the file-path filter and the merge of the layers match what the solver does.

    Every solver: case-folded, runs of whitespace collapsed.  CPLEX: ``.`` read as a space.  Gurobi: ``_``
    after the first character removed (the Gurobi library (``GRBsetparam``, 13.0.2) reads ``Log_File``,
    ``LOG_FILE`` and ``M_I_P_Gap`` as ``LogFile`` / ``MIPGap``, but refuses a leading ``_``; it ignores no
    other character).  Xpress: one leading ``xprs_`` removed (the library's control lookup, xpress 46.01.02,
    reads ``XPRS_MAXTIME`` / ``xprs_maxtime`` as ``MAXTIME``, but not ``XPRS_XPRS_MAXTIME`` or ``XPRSMAXTIME``).
    COPT (coptpy 8.0.4) only ignores case; HiGHS names are exact.
    """
    key = str(name).strip()
    if solver == "cplex":
        key = key.replace(".", " ")
    key = " ".join(key.split()).casefold()
    if solver == "gurobi":
        key = key[:1] + key[1:].replace("_", "")
    elif solver == "xpress" and key.startswith("xprs_"):
        key = key[len("xprs_"):]
    return key


def catalogue_key(solver: str, name: object) -> str:
    """The key of *name* in *solver*'s catalogue: :func:`commercial_option_key` for the commercial solvers (one
    key for every spelling the solver reads as one parameter), the exact name for HiGHS (HiGHS refuses another
    case)."""
    if solver == "highs":
        return str(name).strip()
    return commercial_option_key(solver, str(name))


def rule_class(solver: str, key: str) -> tuple[str, str | None] | None:
    """The class the name rules give *key* (a catalogue key): ``("local_only", "remote")`` for a server prefix,
    ``("file", None)`` for a file / folder / path name, else ``None``.  Applied to every type."""
    folded = key.casefold()
    if folded.startswith(SERVER_PREFIXES.get(solver, ())):
        return ("local_only", "remote")
    if any(word in folded for word in PATH_NAME_WORDS):
        return ("file", None)
    if solver == "cplex" and folded.split(" ", 1)[0] in CPLEX_PATH_FIRST_WORDS:
        return ("file", None)
    if solver == "highs" and folded.startswith(HIGHS_PATH_PREFIXES):
        return ("file", None)
    return None


# ---------------------------------------------------------------------------
# The catalogue files
# ---------------------------------------------------------------------------


class CatalogueError(RuntimeError):
    """FlexTool's solver parameter list is missing or damaged (an installation problem)."""


@dataclass(frozen=True)
class Entry:
    """One parameter: its key, the solver's own spelling, type, class and (``local_only``) reason."""

    key: str
    name: str
    type: str
    cls: str
    reason: str | None = None


@dataclass(frozen=True)
class Catalogue:
    solver: str
    versions: tuple[str, ...]
    entries: Mapping[str, Entry]

    @property
    def label(self) -> str:
        return SOLVER_LABEL[self.solver]

    def versions_text(self) -> str:
        """``made from Gurobi 13.0.2`` (every version merged into the list)."""
        return f"made from {self.label} {', '.join(self.versions)}"

    def entry(self, name: object) -> Entry | None:
        return self.entries.get(catalogue_key(self.solver, name))


def parse_catalogue(solver: str, data: Mapping) -> Catalogue:
    """A :class:`Catalogue` from the JSON *data* of *solver*'s file; raises :class:`CatalogueError` when it is not
    one."""
    if not isinstance(data, Mapping) or data.get("solver") != solver or data.get("format") != FORMAT:
        raise CatalogueError(f"not a format-{FORMAT} catalogue of {solver}")
    versions = tuple(dict.fromkeys(str(g["library"]) for g in data.get("generated_from", [])))
    entries: dict[str, Entry] = {}
    for key, item in data["parameters"].items():
        entry = Entry(key, item["name"], item["type"], item["class"], item.get("reason"))
        if entry.cls not in CLASSES or entry.type not in TYPES or (entry.cls == "local_only") != (
                entry.reason in REASONS):
            raise CatalogueError(f"{solver} parameter {key!r}: class {entry.cls!r} / reason {entry.reason!r}")
        entries[key] = entry
    return Catalogue(solver, versions, entries)


@lru_cache(maxsize=None)
def load(solver: str) -> Catalogue:
    """*solver*'s catalogue, read once per process.  A missing or damaged file raises :class:`CatalogueError`:
    FlexTool never falls back to taking every name."""
    from flextool._resources import package_data_path

    path = package_data_path(f"{CATALOGUE_FOLDER}/{solver}.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return parse_catalogue(solver, data)
    except (OSError, ValueError, KeyError, TypeError, CatalogueError) as exc:
        raise CatalogueError(
            f"FlexTool's list of {SOLVER_LABEL.get(solver, solver)} parameters ({path}) cannot be read ({exc}). "
            "The FlexTool installation is incomplete: reinstall FlexTool (pip install --force-reinstall flextool)."
        ) from exc


# ---------------------------------------------------------------------------
# Lookups for messages
# ---------------------------------------------------------------------------


def cplex_full_names(catalogue: Catalogue, name: object) -> list[str]:
    """The catalogued CPLEX names that *name* abbreviates word by word (``mip tol mipgap`` ->
    ``mip tolerances mipgap``), suggestible ones only, sorted; empty when *name* is listed or abbreviates none."""
    key = catalogue_key("cplex", name)
    if not key or key in catalogue.entries:
        return []
    words = key.split(" ")
    return sorted(
        e.key for e in catalogue.entries.values()
        if e.cls in ("plain", "file")
        and len(e.key.split(" ")) == len(words) and all(f.startswith(w) for w, f in zip(words, e.key.split(" ")))
    )


def suggest(catalogue: Catalogue, name: object) -> str | None:
    """The listed name closest to *name* (only ``plain`` / ``file`` entries: never point a user at a server
    parameter), in the solver's spelling, or ``None``."""
    key = catalogue_key(catalogue.solver, name)
    candidates = {e.key: e.name for e in catalogue.entries.values() if e.cls in ("plain", "file")}
    folded = {k.casefold(): k for k in candidates}
    if key.casefold() in folded:  # HiGHS: another case of a listed name
        return candidates[folded[key.casefold()]]
    if catalogue.solver == "cplex":
        full = cplex_full_names(catalogue, name)
        if full:
            return full[0]
    close = difflib.get_close_matches(key.casefold(), list(folded), n=1, cutoff=0.75)
    return candidates[folded[close[0]]] if close else None


@lru_cache(maxsize=256)
def highs_live_type(name: str) -> str | None:
    """The type the installed HiGHS gives option *name* (``bool`` / ``int`` / ``double`` / ``string``), or
    ``None`` when it does not know the name, the lookup fails or highspy is missing.  Only a positive answer
    (status ``kOk`` and one of the four known types) counts."""
    try:
        import highspy
    except ImportError:
        return None
    try:
        h = highspy.Highs()
        h.silent()
        status, kind = h.getOptionType(name)
    except Exception:  # noqa: BLE001 - any failure of the lookup means "not known"
        return None
    if status != highspy.HighsStatus.kOk:
        return None
    known = {
        highspy.HighsOptionType.kBool: "bool", highspy.HighsOptionType.kInt: "int",
        highspy.HighsOptionType.kDouble: "double", highspy.HighsOptionType.kString: "string",
    }
    return known.get(kind)
