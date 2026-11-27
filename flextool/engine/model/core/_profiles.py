"""Profile limits of process flows."""

import polars as pl
from polar_high import Where
from flextool.engine.model import _coarse_flows
from flextool.engine.input._axis_enums import rename_to_axis
from flextool.engine.model._common import _availability_factor, _entity_rows, _has_rows
from flextool.engine.model.core._context import _Build


def _eq_profiles(b: _Build) -> None:
    """Profile limits of process flows."""
    if not b.has_profile:
        return
    d = b.d

    # ─── EQ. ─── profile_flow_{upper_limit,lower_limit,fixed} ────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_flow[p,s,k,d,t] ± (Σ_r v_reserve[p,r,k,d,t] + Σ_r v_reserve[p,r,s,d,t] · φ[p,s,k,d,t])
    #   − profile[f,d,t]·av[p,d,t]·invested[p,d]
    #   + profile[f,d,t]·av[p,d,t]·retired[p,d]
    #   ≤ / ≥ / =  profile[f,d,t] · av[p,d,t] · existing_count[p,d]
    #   ∀ (p,s,k,f) ∈ process_profile_{upper,lower,fixed}, (d,t) ∈ dt
    # Bounds a flow by a profile times the built capacity, in units of unitsize.  av = availability, 1 for
    # processes without availability data.  The upper limit adds the reserve that raises the arc's flow
    # (up at the sink; down at a unit's source), the lower limit subtracts the reserve that lowers it (down
    # at the sink; up at a unit's source); a source-side reserve is converted to the arc's flow by φ (1
    # where v_flow is the input, 1/slope on a direct unit's input→output arc).  The fixed profile has no
    # reserve term.  The invest/divest terms are restricted to the (p,f) pairs of the method's set.
    # Capacity in service: invested[e,d], retired[e,d] (and prebuilt[e,d]) of _capacity, the retirements of
    # existing capacity counting until its end of life (none for reinvest_automatic).
    # The cached sums of _capacity, present only where a period is in service (no term over an empty domain).
    invested = divested = None
    if b.has_invest_p and _entity_rows(b.cap.alive, "e", d.process_source_sink, "p").height > 0:
        invested = b.cap.invested_p
    if b.has_divest_p or b.has_divest_built_p or b.has_extend_p:
        # The retirements and process_source_sink carry different p vocabularies: cast to the former.
        retired_p = b.cap.retired_frame("p")
        procs = d.process_source_sink.select(pl.col("p").cast(retired_p.schema["p"], strict=False)).unique()
        if (retired_p.join(procs, on="p", how="semi").height > 0 or b.has_divest_built_p
                or b.has_extend_p):
            divested = b.cap.retired_p
    reserve_down = -b.reserve_down_to_sink if b.reserve_down_to_sink is not None else None
    source_terms = {
        "profile_flow_upper_limit": dict(b.reserve_down_at_source),
        "profile_flow_lower_limit": {f"{k}_neg": -v for k, v in b.reserve_up_at_source.items()},
    }
    for idx, name, sense, reserve_label, reserve in (
        (d.process_profile_upper, "profile_flow_upper_limit", "<=", "reserve",     b.reserve_up_to_sink),
        (d.process_profile_lower, "profile_flow_lower_limit", ">=", "reserve_neg", reserve_down),
        (d.process_profile_fixed, "profile_flow_fixed",       "==", None,          None),
    ):
        if not _has_rows(idx):
            continue
        over = idx.join(d.dt, how="cross")
        rhs = d.p_profile_value * b.cap.existing_count_p
        avail = _availability_factor(d, idx) if d.p_process_availability is not None else None
        if avail is not None:
            rhs = rhs * avail
        lhs: dict = {"flow": b.v_flow}
        if invested is not None or divested is not None:
            # Hand-built data may name the profile column "profile" instead of f.
            process_profile_pairs = idx.select("p", "f" if "f" in idx.columns else "profile").unique()
            if "profile" in process_profile_pairs.columns and "f" not in process_profile_pairs.columns:
                process_profile_pairs = process_profile_pairs.pipe(rename_to_axis, {"profile": "f"})
        if invested is not None:
            term = invested * d.p_profile_value
            if avail is not None:
                term = term * avail
            lhs["invest_neg"] = -Where(term, process_profile_pairs)
        if divested is not None:
            term = divested * d.p_profile_value
            if avail is not None:
                term = term * avail
            lhs["divest"] = Where(term, process_profile_pairs)
        if reserve is not None:
            lhs[reserve_label] = reserve
        lhs.update(source_terms.get(name, {}))
        _coarse_flows.add_cstr(
            b, name,
            over      = over,
            sense     = sense,
            lhs_terms = lhs,
            rhs_terms = {"limit": rhs},
        )
