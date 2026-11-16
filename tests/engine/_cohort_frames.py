"""Hand-built cohort frames (``FlexData.cohort`` / ``cohort_alive`` / ``dd_forward``) for tests that build a
``FlexData`` by hand.

A test states the capacity in service as the sets the model reads (``_cohorts.in_service``):

* ``invest`` — ``[e, d_invest, d]``: the investments in service in ``d``;
* ``prebuilt`` — ``[e, d_invest, d]`` (a subset of ``invest``): those in service from the start of ``d``;
* ``retired_p`` / ``retired_n`` — ``[p | n, d_divest, d]``: the retirements of existing capacity counting in ``d``,
  with the retirement options ``divest_p`` / ``divest_n`` (``pd_divest_set`` / ``nd_divest_set``, ``[p | n, d]``).

:func:`cohort_fields` returns the cohort frames that give exactly these sets: an investment cohort per
``(e, d_invest)`` in service in its ``invest`` periods (start of operation 0; year 1 in a pre-built period, 0
otherwise), the existing cohort of each retiring entity in service in the periods its retirements count in, and
``dd_forward`` the ``(d_divest, d)`` pairs of the retirements (an empty frame without retirements).  ``base``
keeps the sets of a hand-built ``FlexData`` that a test does not restate.  A set the frames do not reproduce
(a retirement set not of the form ``pairs ⋈ dd_forward ⋈ alive``) raises: the test must state one that is.
"""
from __future__ import annotations

from types import SimpleNamespace

import polars as pl

from flextool.engine.input._axis_enums import rename_to_axis
from flextool.engine.preprocess.params._cohorts import (
    KIND_EXISTING,
    in_service,
    seed_cohorts,
)


def _triples(rows: "pl.DataFrame | list[tuple[str, str, str]] | None", cols: tuple[str, str, str]) -> pl.DataFrame:
    if rows is None:
        return pl.DataFrame(schema={c: pl.Utf8 for c in cols})
    if isinstance(rows, pl.DataFrame):
        return rows.select(*(pl.col(c).cast(pl.Utf8) for c in cols))
    return pl.DataFrame(rows, schema={c: pl.Utf8 for c in cols}, orient="row")


def _pairs(rows: "pl.DataFrame | None", col: str) -> pl.DataFrame:
    if rows is None:
        return pl.DataFrame(schema={col: pl.Utf8, "d": pl.Utf8})
    return rows.select(pl.col(col).cast(pl.Utf8), pl.col("d").cast(pl.Utf8))


def cohort_fields(*,
                  base: object | None = None,
                  invest: "pl.DataFrame | list[tuple[str, str, str]] | None" = None,
                  prebuilt: "pl.DataFrame | list[tuple[str, str, str]] | None" = None,
                  retired_p: "pl.DataFrame | list[tuple[str, str, str]] | None" = None,
                  retired_n: "pl.DataFrame | list[tuple[str, str, str]] | None" = None,
                  divest_p: "pl.DataFrame | None" = None,
                  divest_n: "pl.DataFrame | None" = None,
                  ) -> dict:
    """``{"cohort", "cohort_alive", "dd_forward"}`` giving the stated sets (module docstring).  With ``base`` (a
    ``FlexData``), a set not stated is the one ``base`` has."""
    if base is not None:
        have = in_service(base)
        invest = have.alive if invest is None else invest
        prebuilt = have.prebuilt if prebuilt is None else prebuilt
        if retired_p is None and have.retired_p.height > 0:
            retired_p, divest_p = have.retired_p, base.pd_divest_set
        if retired_n is None and have.retired_n.height > 0:
            retired_n, divest_n = have.retired_n, base.nd_divest_set
    inv = _triples(invest, ("e", "d_invest", "d"))
    pre = _triples(prebuilt, ("e", "d_invest", "d"))
    if pre.join(inv, on=["e", "d_invest", "d"], how="anti").height > 0:
        raise ValueError("prebuilt must be a subset of invest")
    cohort, alive = seed_cohorts(inv)
    cohort = cohort.with_columns(y_on=pl.lit(0.0))
    pre_keys = pre.with_columns(_pre=pl.lit(True))
    years = (alive.select(pl.col("e").cast(pl.Utf8), pl.col("d_build").cast(pl.Utf8).alias("d_invest"),
                          pl.col("d").cast(pl.Utf8))
             .join(pre_keys, on=["e", "d_invest", "d"], how="left")
             .select(y=pl.when(pl.col("_pre")).then(1.0).otherwise(0.0)))
    alive = alive.with_columns(y=years["y"])

    ret = {"p": _triples(retired_p, ("p", "d_divest", "d")), "n": _triples(retired_n, ("n", "d_divest", "d"))}
    existing = pl.concat([ret[a].select(pl.col(a).alias("e"), "d") for a in ("p", "n")]).unique(maintain_order=True)
    existing_alive = (existing.with_columns(cohort=pl.lit(f"{KIND_EXISTING}:0"), kind=pl.lit(KIND_EXISTING),
                                            d_build=pl.lit(None, dtype=pl.Utf8), y=pl.lit(0.0))
                      .pipe(rename_to_axis, {"e": "e", "d_build": "d_build", "d": "d"})
                      .select(alive.columns))
    forward = (pl.concat([ret[a].select(pl.col("d_divest").alias("d_retire"), "d") for a in ("p", "n")])
               .unique(maintain_order=True)
               .pipe(rename_to_axis, {"d_retire": "d_retire", "d": "d"}))
    fields = {"cohort": cohort,
              "cohort_alive": pl.concat([existing_alive.cast(alive.schema), alive]),
              "dd_forward": forward}
    data = SimpleNamespace(**fields, pd_divest_set=_pairs(divest_p, "p"), nd_divest_set=_pairs(divest_n, "n"))
    built = in_service(data)
    for name, frame, want in (("invest", built.alive, inv), ("prebuilt", built.prebuilt, pre),
                              ("retired_p", built.retired_p, ret["p"]), ("retired_n", built.retired_n, ret["n"])):
        got = frame.select(pl.all().cast(pl.Utf8)).sort(frame.columns)
        if not got.equals(want.sort(want.columns)):
            raise ValueError(f"{name}: the cohort frames give {got.rows()}, not the stated {want.rows()}")
    return fields
