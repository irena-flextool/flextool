"""Prepare a ``dump_csvs`` folder for reload: copy the run's metadata CSVs it lacks."""
from __future__ import annotations

import shutil
from pathlib import Path


def copy_runner_meta(src_workdir: Path | str, dump_dir: Path | str) -> int:
    """Copy every ``src_workdir/{input,solve_data}/*.csv`` that ``dump_dir`` lacks; return the number copied.

    ``FlexData.dump_csvs`` writes the FlexData fields only.  A reload also needs the run's metadata and structural
    CSVs (timeline, period markers, method discriminators, header-only filter sets, ...) that are not FlexData
    fields; this copies them from the run's work folder (a provider snapshot) without overwriting what the dump
    wrote.  Typical use: ``fd.dump_csvs(out)``, ``copy_runner_meta(work_folder, out)``, ``load_from_dump(out)``.
    """
    src_workdir, dump_dir = Path(src_workdir), Path(dump_dir)
    copied = 0
    for kind in ("input", "solve_data"):
        src_dir = src_workdir / kind
        dst_dir = dump_dir / kind
        if not src_dir.is_dir():
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(src_dir.glob("*.csv")):
            dst = dst_dir / src.name
            if dst.exists():
                continue
            shutil.copy(src, dst)
            copied += 1
    return copied


__all__ = ["copy_runner_meta"]
