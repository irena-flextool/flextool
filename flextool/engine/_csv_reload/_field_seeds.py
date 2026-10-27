"""Field seeds of a dump reloaded without a DB reader.

In production ``load_flextool`` always has a DB reader, and the DB overlay (``_apply_db_overrides``) writes some
fields that the loader's own CSV-shaped read used to seed only for a reload.  c24 stage 4 removes those
reload-only seeds from the loader; each removed seed whose field the dump carries moves here.  The loader calls
:func:`apply_dump_field_seeds` through its private ``_dump_hooks.field_seeds`` hook, at the point where the overlay
would run, and only when there is no reader (a reload with ``db_url`` runs the overlay instead).

:data:`SEEDED_FIELDS` names every FlexData field :func:`apply_dump_field_seeds` writes;
``tests/csv_reload/test_dump_seeds.py`` checks that it writes no other field and compares each one with the
production FlexData of the dumped run.
"""
from __future__ import annotations

from pathlib import Path

#: The FlexData fields :func:`apply_dump_field_seeds` writes (none yet; each stage-4 deletion adds its fields).
SEEDED_FIELDS: tuple[str, ...] = ()


def apply_dump_field_seeds(flex_data: object, sd: Path, inp: Path, provider: object) -> None:
    """Seed :data:`SEEDED_FIELDS` on ``flex_data`` from the seeded ``provider``.

    ``sd`` / ``inp`` are the dump's ``solve_data`` / ``input`` folders (the path keys of the provider frames).
    No field has moved here yet, so it writes nothing.
    """
    del flex_data, sd, inp, provider


__all__ = ["SEEDED_FIELDS", "apply_dump_field_seeds"]
