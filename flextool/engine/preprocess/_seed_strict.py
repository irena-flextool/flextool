"""Env-gated strict check that the seeds pruned in c24 stage 4 are always overwritten (temporary).

With ``FLEXTOOL_SEED_STRICT=1``, :func:`load_flextool` replaces the provider-built seed of every field in
:data:`SEED_PRUNE_CANDIDATES` with a poison object just before ``_apply_db_overrides`` (:func:`arm`) and, just
before the end-of-load axis cast, raises :class:`SeedPoisonError` (:func:`check`) when

* any code read a poisoned seed (a touch), even if an ``except Exception`` swallowed the poison's error: every
  touch is recorded on the poison before it raises;
* an armed field still holds its poison (nothing replaced the seed); or
* a poison was copied into another field.

A field that passes on every load is replaced by the DB overlay before anything reads it, so deleting its seed
cannot change a result.  Deletion commits drop their fields from :data:`SEED_PRUNE_CANDIDATES`; the stage-4 clean-up
commit deletes this module, its two call sites and its unit test.

Blind spot: ``isinstance`` and ``type()`` are not touches (the poison defines ``__getattr__``, not
``__getattribute__``, so the class lookups stay safe).  A branch on ``isinstance(field, ...)`` of a poisoned field
passes silently; the strict run's "LP equals the plain LP" requirement covers it.

Counting: :func:`arm` logs ``seed_strict armed: <solve>: <fields>`` at INFO.  When ``FLEXTOOL_SEED_STRICT_LOG``
names a file, :func:`arm`, every touch and :func:`check` also append one JSON line each to it (one line per
``write``, so parallel workers can share the file).

:func:`arm` arms nothing while the ``_cascade_gate()`` test gate is active: the gate drops reassignment of non-None
fields, so a poison could never be replaced there.
"""
from __future__ import annotations

import json
import logging
import os
import traceback

__all__ = ["SEED_PRUNE_CANDIDATES", "SeedPoisonError", "arm", "check", "strict_enabled"]

_LOG = logging.getLogger(__name__)

#: The fields of design_stage4_rev.md §1 whose DB producer assigns the field on every DB path and whose loader
#: seed is not yet deleted (each deletion commit drops its fields; the five varcost fields went in S4-2, the seven
#: direct-parameter fields in S4-3).
SEED_PRUNE_CANDIDATES: tuple[str, ...] = (
    # authoritative projection sets (an empty projection assigns None).
    "groupCapacityMargin",
    "groupInertia",
    "groupNonSync",
    "process_sink_nonSync",
    "process_minload",
    # derived.
    "process_group_inside_nonSync",
    "p_profile_value",
    "dtt__delay_duration",
    "p_process_delay_weight",
    "process_delayed__duration",
)

_STACK_FRAMES = 8


def strict_enabled() -> bool:
    """``True`` when ``FLEXTOOL_SEED_STRICT`` is ``"1"``."""
    return os.environ.get("FLEXTOOL_SEED_STRICT") == "1"


class SeedPoisonError(RuntimeError):
    """A pruned-seed candidate was read, or was not replaced, during the DB overlay."""


def _record(event: dict) -> None:
    path = os.environ.get("FLEXTOOL_SEED_STRICT_LOG")
    if not path:
        return
    line = json.dumps(event, sort_keys=True) + "\n"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line)


def _stack() -> list[str]:
    frames = traceback.extract_stack()[:-3][-_STACK_FRAMES:]
    return [f"{f.filename.rsplit('/', 2)[-1]}:{f.lineno} {f.name}" for f in frames]


class _Poison:
    """Stands in for a seed; any use other than ``is`` / ``isinstance`` / ``type()`` is recorded and raises."""

    __slots__ = ("_name", "_solve", "_touches")

    def __init__(self, name: str, solve: str | None) -> None:
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_solve", solve)
        object.__setattr__(self, "_touches", [])

    def _touch(self, what: str) -> SeedPoisonError:
        stack = _stack()
        self._touches.append((what, stack))
        _record({"event": "touch", "field": self._name, "solve": self._solve, "what": what, "stack": stack})
        return SeedPoisonError(
            f"seed_strict: the seed of {self._name!r} (solve {self._solve!r}) was used ({what}) before the "
            f"DB overlay replaced it")

    # Only attributes the class lacks reach ``__getattr__`` (``.height``, ``.frame``, ``.filter``, ``__array__``
    # …); ``__class__`` and the slots do not, so ``isinstance`` / ``type()`` stay safe.
    def __getattr__(self, attr: str):
        raise self._touch(f".{attr}")

    def __setattr__(self, attr: str, value) -> None:
        raise self._touch(f"set .{attr}")

    def __delattr__(self, attr: str) -> None:
        raise self._touch(f"del .{attr}")

    def __repr__(self) -> str:
        # Never raises, so an error message that formats the value stays intact; still a touch.
        self._touch("repr")
        return f"<seed poison {self._name}>"

    __str__ = __repr__

    def __format__(self, spec: str) -> str:
        self._touch("format")
        return f"<seed poison {self._name}>"

    def __hash__(self) -> int:
        raise self._touch("hash")


def _make_dunder(dunder: str):
    def _dunder(self, *args, **kwargs):
        raise self._touch(dunder)
    _dunder.__name__ = dunder
    return _dunder


# Special methods are looked up on the type, so ``__getattr__`` never sees them: define each one.
for _d in (
    "__bool__", "__len__", "__iter__", "__reversed__", "__getitem__", "__setitem__", "__delitem__",
    "__contains__", "__call__",
    "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
    "__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__", "__truediv__", "__rtruediv__",
    "__floordiv__", "__rfloordiv__", "__mod__", "__rmod__", "__pow__", "__rpow__", "__matmul__", "__rmatmul__",
    "__and__", "__rand__", "__or__", "__ror__", "__xor__", "__rxor__", "__neg__", "__pos__", "__abs__",
    "__invert__", "__float__", "__int__", "__index__",
    "__copy__", "__deepcopy__", "__reduce__", "__reduce_ex__", "__getstate__",
):
    setattr(_Poison, _d, _make_dunder(_d))
del _d


def arm(flex_data, solve: str | None) -> dict[str, _Poison]:
    """Replace every non-None candidate seed with a poison; return ``{field: poison}``.

    Bypasses ``FlexData.__setattr__``.  Arms nothing (returns ``{}``) while ``_cascade_gate()`` is active.
    """
    from flextool.engine.preprocess import flex_data as _fd

    if _fd._CASCADE_GATE_ACTIVE:
        return {}
    armed: dict[str, _Poison] = {}
    for name in SEED_PRUNE_CANDIDATES:
        if getattr(flex_data, name) is None:
            continue
        poison = _Poison(name, solve)
        object.__setattr__(flex_data, name, poison)
        armed[name] = poison
    _LOG.info("seed_strict armed: %s: %s", solve, ", ".join(armed) or "-")
    _record({"event": "arm", "solve": solve, "armed": list(armed)})
    return armed


def check(flex_data, armed: dict[str, _Poison], solve: str | None) -> None:
    """Raise :class:`SeedPoisonError` if an armed seed was touched, survived, or was copied to another field.

    A no-op for an empty ``armed`` (nothing armed, or the cascade gate was active).
    """
    if not armed:
        return
    touched = {name: p._touches for name, p in armed.items() if p._touches}
    survived = [name for name, p in armed.items() if getattr(flex_data, name) is p]
    poisons = {id(p): name for name, p in armed.items()}
    leaked = {f: poisons[id(v)] for f, v in vars(flex_data).items()
              if isinstance(v, _Poison) and id(v) in poisons and f != poisons[id(v)]}
    _record({"event": "check", "solve": solve, "armed": list(armed), "touched": sorted(touched),
             "survived": survived, "leaked": leaked})
    if not (touched or survived or leaked):
        return
    lines = [f"seed_strict: solve {solve!r}: pruned-seed candidates were used or not replaced"]
    for name, touches in touched.items():
        for what, stack in touches:
            lines.append(f"  touched {name} ({what}):")
            lines.extend(f"      {s}" for s in stack)
    lines.extend(f"  not replaced: {name}" for name in survived)
    lines.extend(f"  copied: {src} -> field {dst}" for dst, src in leaked.items())
    raise SeedPoisonError("\n".join(lines))
