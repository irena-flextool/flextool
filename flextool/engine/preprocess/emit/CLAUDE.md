# Vectorized emit rules (`flextool/engine/preprocess/emit/`)

The vectorized per-roll emit must stay byte-identical to the per-roll writer.  Every heavy derive here has a
`tests/**/test_vectorize_<family>_parity.py` gate (Tier A `.equals`, Tier B `rtol≈1e-12`); run the one for the family
you touch.

- Render value columns with `_render_value_column` (`_vectorize.py`, the `repr()` loop), never `.cast(pl.Utf8)`: a
  cast diverges on sci-notation padding and `NaN` (a static guard checks `col("value").cast(...)`).
- Lift cascade lookups from the de-duplicated dict (`lift_dict_to_lookup`), never from the raw CSV: duplicate keys
  explode the join.
- Keep domain and period order: don't `.unique()` the source lists; keep the `__eo` / `__to` sort keys.
