"""Build a :class:`FlexDataProvider` from hand-written ``input/`` and ``solve_data/`` CSV folders.

Loader unit tests write a few CSVs and call one loader helper directly (``_load_storage``,
``_group_slack.load_data``, …).  The loaders read only the provider they are given (production passes
the sub-solve's provider; the disk fallback for ``provider=None`` is being removed), so such a test
puts its CSVs into a provider first.  Each file ``<dir>/<stem>.csv`` becomes the frame
``pl.read_csv(<file>)`` under the key ``"input/<stem>"`` or ``"solve_data/<stem>"`` — the key the
loaders look up for that path, and the same read the loader's disk fallback made.

This is a test helper: main code never builds a provider from a folder.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from flextool.engine.preprocess._flex_data_provider import FlexDataProvider

__all__ = ["provider_from_csv_dirs"]


def provider_from_csv_dirs(
    input_dir: Path | str | None = None,
    solve_data_dir: Path | str | None = None,
    *,
    names: tuple[str, ...] | None = None,
) -> FlexDataProvider:
    """A new provider holding the CSVs of ``input_dir`` (keys ``input/<stem>``) and ``solve_data_dir``
    (keys ``solve_data/<stem>``).

    * A folder given as ``None`` is skipped; a folder that is given but missing raises
      ``FileNotFoundError``.
    * ``names``: only these stems (without ``.csv``) are read, in both folders; a listed stem without a
      file is skipped.  ``None`` reads every ``*.csv``.
    * A file ``pl.read_csv`` cannot read (for example a 0-byte file) raises: a test fixture never
      silently loses a file.
    """
    provider = FlexDataProvider()
    for kind, folder in (("input", input_dir), ("solve_data", solve_data_dir)):
        if folder is None:
            continue
        folder = Path(folder)
        if not folder.is_dir():
            raise FileNotFoundError(f"provider_from_csv_dirs: {kind} folder {folder} is missing")
        if names is None:
            files = sorted(folder.glob("*.csv"))
        else:
            files = [folder / f"{n}.csv" for n in names if (folder / f"{n}.csv").is_file()]
        for path in files:
            try:
                frame = pl.read_csv(path)
            except Exception as exc:
                raise ValueError(f"provider_from_csv_dirs: cannot read {path}: {exc}") from exc
            provider.put(f"{kind}/{path.stem}", frame)
    return provider
