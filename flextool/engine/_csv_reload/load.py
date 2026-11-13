"""Rebuild a :class:`FlexData` from a CSV dump folder (unsupported; see the package docstring)."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from flextool.engine._csv_reload._field_seeds import apply_dump_field_seeds
from flextool.engine._csv_reload._seed import seed_provider_from_dump
from flextool.engine._csv_reload._stamp import check_dump, warn_reload
from flextool.engine._csv_reload._topology import pss_from_dump
from flextool.engine.input._spinedb_reader import SpineDbReader
from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
from flextool.engine.preprocess.flex_data import FlexData, load_flextool


@dataclass(frozen=True)
class _DumpHooks:
    """The private ``load_flextool(_dump_hooks=)`` argument: the reload's process topology and field seeds."""

    topology: Callable[[Path, object], dict | None]
    field_seeds: Callable[[object, Path, Path, object], None]


_DUMP_HOOKS = _DumpHooks(topology=pss_from_dump, field_seeds=apply_dump_field_seeds)


def _load(dump_dir: Path | str, *, db_url: str | None, scenario: str | None, force: bool,
          stacklevel: int) -> tuple[FlexData, FlexDataProvider]:
    """Shared body of the public loaders; ``stacklevel`` counts from this function's caller (1 = that caller)."""
    if (db_url is None) != (scenario is None):
        raise ValueError("load_from_dump: db_url and scenario= must be given together")
    stamp = check_dump(dump_dir, force)
    warn_reload(dump_dir, stamp, stacklevel=stacklevel + 1)
    provider = FlexDataProvider()
    seed_provider_from_dump(provider, dump_dir)
    reader = SpineDbReader(db_url, scenario=scenario) if db_url is not None else None
    if reader is not None:
        # The solves of a 2-D period map are reader solves in the run; give the reload the same view.
        from flextool.engine.input._solve_config import SolveConfig, register_solve_copies
        register_solve_copies(SolveConfig.load_from_db_url(db_url, scenario), reader)
    fd = load_flextool(Path(dump_dir), provider=provider, db_reader=reader, _dump_hooks=_DUMP_HOOKS)
    return fd, provider


def load_from_dump_with_provider(dump_dir: Path | str, *, db_url: str | None = None, scenario: str | None = None,
                                 force: bool = False) -> tuple[FlexData, FlexDataProvider]:
    """Rebuild a :class:`FlexData` from ``dump_dir``; return it with the provider it was loaded from.

    Steps: check the stamp (:func:`check_dump`; refuses an unstamped or incompatible dump unless ``force``), emit
    :class:`CsvReloadWarning` (every call), seed a fresh provider from ``dump_dir/{input,solve_data}/*.csv``, then
    ``load_flextool`` with that provider.  With ``db_url`` + ``scenario`` the loader reads the DB-direct parameters
    from that database (no axis Enums, as in production); without them it is a CSV-only load.  Nothing is
    discovered: no ``tests.sqlite`` next to the dump, no scenario from the folder name.  The reload has no
    solve-to-solve carry (``handoff/`` folders are not read), so it can differ from the solved model.  Nor does it
    apply the solve's ``reduce_unit_detail_method`` / ``reduce_system_detail_method`` (the cascade's
    ``preprocess._reduce_detail.apply_reduce_detail``): a reduced solve's dump reloads as the unreduced model.

    Raises ``ValueError`` when only one of ``db_url`` / ``scenario`` is given.
    """
    return _load(dump_dir, db_url=db_url, scenario=scenario, force=force, stacklevel=2)


def load_from_dump(dump_dir: Path | str, *, db_url: str | None = None, scenario: str | None = None,
                   force: bool = False) -> FlexData:
    """:func:`load_from_dump_with_provider` without the provider."""
    return _load(dump_dir, db_url=db_url, scenario=scenario, force=force, stacklevel=2)[0]


__all__ = ["load_from_dump", "load_from_dump_with_provider"]
