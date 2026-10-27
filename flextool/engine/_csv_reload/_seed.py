"""Seed a :class:`FlexDataProvider` from a dump folder's ``input/`` and ``solve_data/`` CSVs."""
from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
import polars.exceptions as pl_exc

logger = logging.getLogger(__name__)

# Sub-folders of a dump that are seeded, in order, keyed ``"{kind}/{stem}"``.  ``handoff/``,
# ``produced_handoff/`` and ``derived/`` of a provider snapshot are not seeded: a reload has no solve-to-solve
# carry.
_SEEDED_KINDS: tuple[str, ...] = ("input", "solve_data")


def seed_provider_from_dump(provider, dump_dir: Path | str) -> int:
    """Put every ``dump_dir/{input,solve_data}/*.csv`` into ``provider`` as ``"{kind}/{stem}"``; return the count.

    Files are read in sorted order.  A malformed or unreadable CSV is logged at WARNING and skipped.  A missing
    sub-folder seeds nothing.  The frames are what ``pl.read_csv`` returns, the same as the loader's own
    provider-less seeding (which runs before its CSV cache is active).
    """
    seeded = 0
    for kind in _SEEDED_KINDS:
        folder = Path(dump_dir) / kind
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.csv")):
            try:
                df = pl.read_csv(path)
            except (pl_exc.ComputeError, pl_exc.NoDataError) as exc:
                logger.warning("seed_provider_from_dump: skipping malformed CSV %s (%s: %s)",
                               path, type(exc).__name__, exc)
                continue
            except Exception as exc:  # noqa: BLE001 — log + continue for stray files
                logger.warning("seed_provider_from_dump: skipping unreadable CSV %s (%s: %s)",
                               path, type(exc).__name__, exc)
                continue
            provider.put(f"{kind}/{path.stem}", df)
            seeded += 1
    return seeded


__all__ = ["seed_provider_from_dump"]
