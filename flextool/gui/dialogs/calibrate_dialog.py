"""Non-modal "Calibrate investments" dialog.

The dialog gathers the representative-period build knobs and the calibration
controls for one or more scenarios and turns them into auxiliary jobs via
:mod:`flextool.gui.calibrate_jobs` — it never renders a command line itself
(the single argv renderers in :mod:`flextool.gui.calibrate_commands` do that,
for both the launcher and the live CLI previews, so the previewed text can
never drift from what runs).

Why non-modal
-------------
The dialog launches long-running solves that stream into the Execution window;
the user must be able to watch those while the dialog stays open. It is
therefore ``transient`` to its parent (stacks with it, no taskbar entry on
some WMs) but takes NO ``grab_set`` — matching the repo rule that ``grab_set``
in a Toplevel constructor can abort "not viewable", and the non-modal picker
precedent (``PlotSettingsPicker``).

Two independent tools
---------------------
The dialog hosts two separate tools stacked in the same window: a
representative-periods builder (top) and the adequacy calibrator (bottom).
They share the scenario set and the solve checklist but are otherwise
independent — each has its own explanation, its own action button, and its own
live CLI preview. Both write their result into a NEW alternative, so any change
made here is undone by deleting that alternative.

Dependency injection
--------------------
Every external input — the scenarios, the DB-url resolver, the execution
manager, the python executable, the settings object and its save callback — is
injected through the constructor, never reached through globals, so the dialog
can be built headlessly with fakes in a test (see
``tests/gui/test_calibrate_dialog.py``).

xlsx exclusion
--------------
RP and calibration both WRITE into the input database; for an xlsx-backed
scenario the GUI regenerates an intermediate sqlite from the xlsx on every
run, discarding those writes. So when ANY provided scenario is xlsx-backed the
action buttons are disabled with an explanatory reason, and the run/RP/copy
paths defensively skip xlsx scenarios too.
"""
from __future__ import annotations

import logging
import tkinter as tk
from collections.abc import Callable, Sequence
from pathlib import Path
from tkinter import ttk
from typing import Any

from flextool.gui.calibrate_commands import (
    build_calibrate_command,
    build_rp_command,
    command_to_display_string,
    final_write_methods_from_settings,
    overshoot_pct_to_multiplier,
)
from flextool.gui.calibrate_jobs import (
    CalibJobSpec,
    RpJobSpec,
    _calib_dirs,
    launch_calibration_jobs,
    launch_rp_jobs,
)
from flextool.gui.hover_tooltip import attach_tooltip
from flextool.gui.solve_reader import read_scenario_solves
from flextool.representative_periods.scenario_stack import (
    add_alternative_to_scenario,
    dedup_alternative_name,
    existing_alternative_names,
)

logger = logging.getLogger(__name__)

# How long (ms) the action buttons stay disabled after a launch, to swallow
# an accidental double-click without permanently locking the button (the
# launcher itself serialises, but a duplicate submit would spawn a duplicate
# job with the same action_key).
_GUARD_MS = 2500

# Pixel width used to wrap the (verbose) hover tooltips so they never run off
# the screen edge.
_TIP_WRAP = 380

# ── Above-tool explanations (shown as a paragraph at the top of each tool) ──
_RP_EXPLANATION = (
    "Representative periods compress a long timeline into a handful of short, "
    "weighted periods that stand in for the whole span. Solving over these "
    "instead of every time step makes investment runs far faster while keeping "
    "the demand and weather patterns that drive the result. This tool clusters "
    "the scenario's profiles and inflows, writes the periods into a NEW "
    "alternative, and — if the box below is ticked — adds that alternative to "
    "the selected scenario(s) so their solve uses the new periods. Nothing "
    "existing is overwritten: the change lives entirely in the new alternative, "
    "so you can undo it any time by deleting that alternative."
)
_CALIB_EXPLANATION = (
    "Calibration repeatedly solves each scenario and nudges its energy-margin "
    "adder until the system reaches an adequate capacity margin, so the "
    "resulting investments are neither over- nor under-built. Each iteration "
    "reads the previous solve's shortfall and resizes the margin accordingly. "
    "If you built representative periods above and added them to the scenario, "
    "calibration automatically solves over those periods too — so build them "
    "first for a much faster calibration. The margins are written into a NEW "
    "per-scenario alternative ('<scenario>_adeq_calib') and your original data "
    "is left untouched — so, as with the periods above, you can undo everything "
    "by deleting that alternative."
)

# ── Per-control hover tooltips (plain-English, verbose) ─────────────────────
_TIPS = {
    "n_rp": (
        "How many representative periods to keep. More periods reproduce the "
        "original timeline more faithfully but make every solve slower. A "
        "typical starting point is 5–20 (use more periods for better seasonal "
        "representation, but with shorter periods to maintain solve speed)."
    ),
    "period_length": (
        "How many time steps each representative period spans (e.g. 24 for a "
        "day, 168 for a week). Longer periods capture within-period storage "
        "and ramping behaviour but leave fewer distinct periods to choose "
        "from."
    ),
    "force_sustained": (
        "Also keep the period with the greatest SUSTAINED net load — a long "
        "stretch where demand stays high while wind/solar stay low. Clustering "
        "alone can miss such energy-adequacy stress periods; ticking this "
        "guarantees the worst multi-hour lull is represented, which matters "
        "when sizing storage and firm capacity."
    ),
    "force_peak": (
        "Also keep the period containing the single highest INSTANTANEOUS net "
        "load (the peak hour). This protects capacity adequacy — the moment "
        "the system is most likely to fall short — even if that hour would "
        "otherwise be averaged away by clustering."
    ),
    "force_window": (
        "The length (in time steps) of the rolling window used to score the "
        "'sustained net load' above. A smaller window reacts to short sharp "
        "lulls; a larger one favours long multi-day droughts. Only relevant "
        "when 'Force highest sustained net load' is ticked."
    ),
    "solves": (
        "Which solves get their periods rebuilt. Only these solves have their "
        "'period_timeset' repointed at the new representative periods; other "
        "solves are left untouched. Investment solves are ticked by default "
        "because that is where representative periods usually matter."
    ),
    "add_to_scenario": (
        "When ticked, the new representative-period alternative is appended to "
        "each selected scenario's alternative stack (at the bottom, so it "
        "wins — the bottommost alternative overrides whatever the ones above "
        "it set), meaning the scenario immediately uses the new periods on its "
        "next run. When unticked, the alternative is still created but stays "
        "detached — you can add it to a scenario yourself later. Either way, "
        "deleting the alternative fully reverts the change."
    ),
    "max_iterations": (
        "The maximum number of solve-and-resize rounds the calibrator runs per "
        "scenario. It stops early once the margin is adequate; this is just "
        "the ceiling so a hard-to-satisfy case cannot loop forever."
    ),
    "sizing": (
        "How the energy-margin adder is shaped. 'timed' places the extra "
        "margin only in the specific periods/time steps that showed a "
        "shortfall (targeted, usually cheaper). 'uniform' adds the same "
        "constant margin to every time step (simpler, more conservative, might "
        "not converge)."
    ),
    "overshoot": (
        "A planning safety margin, in percent, applied on top of the sizing "
        "each iteration. For example 20 means aim 20% above the bare adequate "
        "level, trading a little extra cost for robustness. 0 turns the "
        "overshoot off."
    ),
    "damping_first": (
        "How strongly the FIRST iteration moves toward the newly computed "
        "margin (1.0 = take the full step). The first step often has the "
        "furthest to travel, so it is exposed separately from later steps."
    ),
    "damping_remaining": (
        "How strongly every iteration AFTER the first moves toward the newly "
        "computed margin (e.g. 0.5 = move halfway each round). Lower values "
        "converge more smoothly and avoid overshoot oscillation; higher values "
        "converge faster but can bounce."
    ),
    "stall_fraction": (
        "How sensitive the run is to a resource cap. If an iteration's "
        "shortfall barely improves — changing by less than this fraction — the "
        "calibrator treats the case as capped (e.g. a hard resource or "
        "transmission limit) and stops rather than spending more iterations "
        "chasing a target it cannot reach."
    ),
    "keep_artifacts": (
        "Keep the intermediate files (per-iteration solves, logs, warm-start "
        "caches) instead of cleaning them up. Useful for debugging a "
        "calibration that behaves unexpectedly; leave off for normal runs to "
        "save disk space."
    ),
}


class CalibrateDialog(tk.Toplevel):
    """Non-modal dialog to build representative periods and run calibrations.

    Parameters
    ----------
    parent:
        Owning widget; the window is ``transient`` to it (and centred on it,
        so it opens on the parent's monitor) but NOT modal.
    scenarios:
        The scenarios to operate on. Each must expose a ``name`` attribute and
        an ``is_xlsx`` attribute (``bool``; xlsx-backed scenarios are excluded
        from running — see the module docstring).
    project_path:
        Project root; passed to the launchers, which derive per-scenario work /
        output directories under it.
    settings:
        The :class:`~flextool.gui.data_models.ProjectSettings` whose ``calib_*``
        fields seed every widget and receive every change.
    execution_mgr:
        The ``ExecutionManager`` the launchers register aux jobs on.
    python_exe:
        Interpreter used to spawn the RP / calibrate subprocesses.
    resolve_db_url:
        ``scenario -> db_url`` — resolves a (non-xlsx) scenario object to the
        sqlite URL RP / calibrate read and write.
    save_settings:
        Zero-arg callback persisting ``settings`` to disk; invoked after every
        change and on close.
    show_execution_window:
        Optional zero-arg callback that opens the Execution-jobs window (or
        raises it if already open). Invoked right after an RP / calibration
        launch so the user sees the job stream immediately. ``None`` (the
        headless-test default) skips it.
    """

    def __init__(
        self,
        parent: tk.Misc,
        *,
        scenarios: Sequence[Any],
        project_path: Path,
        settings: Any,
        execution_mgr: Any,
        python_exe: str,
        resolve_db_url: Callable[[Any], str],
        save_settings: Callable[[], None],
        show_execution_window: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.title("Calibrate investments")

        self._scenarios = list(scenarios)
        self._project_path = Path(project_path)
        self._settings = settings
        self._execution_mgr = execution_mgr
        self._python_exe = python_exe
        self._resolve_db_url = resolve_db_url
        self._save_settings = save_settings
        self._show_execution_window = show_execution_window

        # Non-modal: transient (stacks with the parent) but NO grab_set.
        self.transient(parent)

        # In-flight guards: once a launch fires the matching button is disabled
        # for _GUARD_MS to swallow a double-click, then re-enabled.
        self._rp_guarded = False
        self._calib_guarded = False

        # Lazily-seeded cache of alternative names already present per db_url,
        # used to de-duplicate the derived RP alternative name (append _2/_3…).
        # An RP launch optimistically records the names it reserves here so the
        # next preview / launch sees them without another DB read.
        self._existing_alts: dict[str, set[str]] = {}

        # Tk variables (created before traces are wired).
        self._var_n_rp = tk.StringVar()
        self._var_period_length = tk.StringVar()
        self._var_force_sustained = tk.BooleanVar()
        self._var_force_peak = tk.BooleanVar()
        self._var_force_window = tk.StringVar()
        self._var_add_to_scenario = tk.BooleanVar()
        self._var_max_iterations = tk.StringVar()
        self._var_sizing = tk.StringVar()
        self._var_overshoot_pct = tk.StringVar()
        self._var_damping_first = tk.StringVar()
        self._var_damping_remaining = tk.StringVar()
        self._var_stall_fraction = tk.StringVar()
        self._var_keep_artifacts = tk.BooleanVar()
        self._reason_var = tk.StringVar(value="")

        # Solve-checklist BooleanVars, keyed by solve name (union across
        # scenarios), and the union order for stable display.
        self._solve_vars: dict[str, tk.BooleanVar] = {}
        self._solve_order: list[str] = []

        self._seed_from_settings()
        self._build_solve_selection()

        self._advanced_visible = tk.BooleanVar(value=False)

        self._build_widgets()
        self._wire_traces()
        self._refresh_run_state()
        self._refresh_cli_preview()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Escape>", lambda _e: self._on_close())
        # One wheel handler on the toplevel catches events bubbling up from any
        # descendant (X11 Button-4/5 + Windows/macOS MouseWheel), routed to the
        # body or the solve list by _on_wheel.
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.bind(seq, self._on_wheel)

        # Cap the height to the screen (small-screen guard) and centre on the
        # parent. Guarded: a headless / detached parent may not report geometry.
        self.update_idletasks()
        self._size_and_center(parent)

    # ── Seed / model helpers ──────────────────────────────────────────
    def _seed_from_settings(self) -> None:
        """Initialise every scalar widget var from the settings object."""
        s = self._settings
        self._var_n_rp.set(str(s.calib_rp_n_rp))
        self._var_period_length.set(str(s.calib_rp_period_length))
        self._var_force_sustained.set(bool(s.calib_rp_force_sustained))
        self._var_force_peak.set(bool(s.calib_rp_force_peak))
        self._var_force_window.set(str(s.calib_rp_force_window))
        self._var_add_to_scenario.set(bool(s.calib_rp_add_to_scenario))
        self._var_max_iterations.set(str(s.calib_max_iterations))
        self._var_sizing.set(s.calib_sizing or "timed")
        self._var_overshoot_pct.set(str(s.calib_overshoot_pct))
        self._var_damping_first.set(str(s.calib_damping_first))
        self._var_damping_remaining.set(str(s.calib_damping_remaining))
        self._var_stall_fraction.set(str(s.calib_stall_fraction))
        self._var_keep_artifacts.set(bool(s.calib_keep_artifacts))

    def _build_solve_selection(self) -> None:
        """Union the solves across scenarios; seed each check state.

        Same-named solves collapse to one row (first appearance keeps the
        order; ``has_invest_periods`` is OR-ed so an invest solve in any
        scenario is treated as invest). Initial check state: if the solve name
        is already in ``settings.calib_selected_solves`` use that; otherwise
        default-check iff it is an investment solve.
        """
        prior = set(self._settings.calib_selected_solves or [])
        invest_flag: dict[str, bool] = {}
        # Per-scenario solve names, so the RP launch/preview passes each
        # scenario ONLY the selected solves that actually belong to it — a
        # union-checklist solve from another scenario must never reach a
        # scenario that lacks it (the preprocess errors on such a solve).
        self._scenario_solves: dict[str, list[str]] = {}
        for sc in self._scenarios:
            if getattr(sc, "is_xlsx", False):
                continue
            name = self._scenario_name(sc)
            try:
                url = self._resolve_db_url(sc)
                solves = read_scenario_solves(url, name)
            except Exception as exc:  # a bad DB must not break the dialog
                logger.warning(
                    "Could not read solves for scenario %r: %s", name, exc,
                )
                continue
            self._scenario_solves[name] = [info.name for info in solves]
            for info in solves:
                if info.name not in invest_flag:
                    self._solve_order.append(info.name)
                    invest_flag[info.name] = info.has_invest_periods
                else:
                    invest_flag[info.name] |= info.has_invest_periods

        for name in self._solve_order:
            # calib_selected_solves lists only the CHECKED solves, so presence
            # means checked; a never-seen solve defaults to checked iff invest.
            checked = True if name in prior else invest_flag[name]
            self._solve_vars[name] = tk.BooleanVar(value=checked)

    @staticmethod
    def _scenario_name(scenario: Any) -> str:
        return getattr(scenario, "name", str(scenario))

    def _runnable_scenarios(self) -> list[Any]:
        """Non-xlsx scenarios — the ones RP / calibrate may write into."""
        return [
            sc for sc in self._scenarios if not getattr(sc, "is_xlsx", False)
        ]

    def _selected_solves(self) -> list[str]:
        return [n for n in self._solve_order if self._solve_vars[n].get()]

    # ── RP alternative naming ──────────────────────────────────────────
    def _base_alt_name(self, scenario_name: str) -> str:
        """Derive the RP alternative base name (before de-duplication).

        Keyed on the SCENARIO name, plus the representative-period count and
        the period length, e.g. ``coal_rp_40rp_54h`` for scenario ``coal``,
        40 periods of 54 steps. The scenario — not the solve — is the
        informative discriminator: each scenario is clustered separately from
        its own VRE/demand series, so its representative periods are scenario-
        specific. The solve(s) the periods were built for are recorded in the
        description instead. Distinct scenarios never collide (names are
        unique); a repeat build of the SAME scenario/count/length de-dups to
        ``_2``/``_3``.
        """
        s = self._settings
        return (
            f"{scenario_name}_rp_{s.calib_rp_n_rp}rp_"
            f"{s.calib_rp_period_length}h"
        )

    def _alt_description(self, scenario_name: str, solves: Sequence[str]) -> str:
        """Verbose description written onto the RP alternative."""
        s = self._settings
        if solves:
            solve_txt = "solve(s) " + ", ".join(f"'{x}'" for x in solves)
        else:
            solve_txt = "all solves carrying a period_timeset"
        return (
            "Representative periods generated with the Calibrate-investments "
            f"tool for scenario '{scenario_name}': {s.calib_rp_n_rp} "
            f"representative period(s) of {s.calib_rp_period_length} time steps "
            f"each, built for {solve_txt}. This alternative can be deleted to "
            "undo the change."
        )

    def _taken_alt_names(self, db_url: str) -> set[str]:
        """Cached set of alternative names already in *db_url* (lazy-seeded)."""
        if db_url not in self._existing_alts:
            try:
                self._existing_alts[db_url] = existing_alternative_names(db_url)
            except Exception as exc:  # a bad DB must not break the preview
                logger.warning(
                    "Could not read alternatives for %s: %s", db_url, exc
                )
                self._existing_alts[db_url] = set()
        return self._existing_alts[db_url]

    def _applicable_solves(self, scenario_name: str) -> list[str]:
        """Selected solves that actually belong to *scenario_name* (in order).

        The solve checklist is the UNION across scenarios; a scenario only
        gets the selected solves it actually runs. A solve selected but absent
        from this scenario is dropped, so the RP preprocess never receives a
        solve it cannot repoint.
        """
        selected = set(self._selected_solves())
        scen = set(self._scenario_solves.get(scenario_name, []))
        return [n for n in self._solve_order if n in selected and n in scen]

    def _rp_plan(
        self, *, commit: bool
    ) -> list[tuple[Any, str, str, list[str], str, str]]:
        """Plan one RP job per runnable scenario that has an applicable solve.

        Returns ``[(scenario, name, db_url, applicable_solves, alt_name,
        description), …]``. A scenario contributes an entry ONLY when at least
        one selected solve belongs to it — so with no matching solve the
        scenario is skipped rather than launched with a solve it cannot use.

        The alternative name encodes that scenario's applicable solve(s) and is
        de-duplicated (``_2``/``_3``) against the per-db cache; names are
        allocated sequentially through a per-db overlay so two scenarios that
        share a database and a base name get ``base`` then ``base_2``. With
        ``commit`` the reserved names are folded into the persistent cache so
        the next preview / launch sees them (the launch path); the preview
        passes ``commit=False`` so merely looking never reserves.
        """
        overlay: dict[str, set[str]] = {}
        plan: list[tuple[Any, str, str, list[str], str, str]] = []
        for sc in self._runnable_scenarios():
            name = self._scenario_name(sc)
            applicable = self._applicable_solves(name)
            if not applicable:
                continue
            db_url = self._resolve_db_url(sc)
            base = self._base_alt_name(name)
            taken = set(self._taken_alt_names(db_url)) | overlay.get(db_url, set())
            alt = dedup_alternative_name(base, taken)
            overlay.setdefault(db_url, set()).add(alt)
            desc = self._alt_description(name, applicable)
            plan.append((sc, name, db_url, applicable, alt, desc))
        if commit:
            for db_url, names in overlay.items():
                self._existing_alts.setdefault(db_url, set()).update(names)
        return plan

    # ── Layout ────────────────────────────────────────────────────────
    def _build_widgets(self) -> None:
        pad = dict(padx=10, pady=(6, 0))

        # Fixed footer (reason + Close) pinned to the bottom OUTSIDE the scroll
        # region, so the Close button is always reachable however tall the body
        # grows. Packed first (side=bottom) to reserve its space.
        self._footer_area = ttk.Frame(self)
        self._footer_area.pack(side="bottom", fill="x")
        ttk.Label(
            self._footer_area, textvariable=self._reason_var,
            foreground="gray", wraplength=560, justify="left",
        ).pack(fill="x", padx=10, pady=(6, 4))
        close_row = ttk.Frame(self._footer_area)
        close_row.pack(fill="x", padx=10, pady=(0, 6))
        ttk.Button(close_row, text="Close", command=self._on_close).pack(
            side="right"
        )

        # The body scrolls vertically inside a height-capped window — the fixed
        # content (explanations + previews) alone can exceed a 1024-tall screen.
        outer = self._build_scroll_body()

        # ── 1) Representative periods ─────────────────────────────────
        rp = ttk.LabelFrame(
            outer, text="1 · Build representative periods (optional)"
        )
        rp.pack(fill="x", **pad)

        # No explicit foreground — a hardcoded gray vanishes against a dark
        # theme; the default label colour tracks the theme in both modes.
        ttk.Label(
            rp, text=_RP_EXPLANATION, wraplength=560, justify="left",
        ).pack(fill="x", padx=8, pady=(8, 4))

        row = ttk.Frame(rp)
        row.pack(fill="x", padx=8, pady=(4, 4))
        n_lbl = ttk.Label(row, text="Periods (n_rp):")
        n_lbl.pack(side="left")
        n_ent = ttk.Entry(row, textvariable=self._var_n_rp, width=8)
        n_ent.pack(side="left", padx=(4, 16))
        pl_lbl = ttk.Label(row, text="Period length (steps):")
        pl_lbl.pack(side="left")
        pl_ent = ttk.Entry(row, textvariable=self._var_period_length, width=8)
        pl_ent.pack(side="left", padx=(4, 0))
        for w in (n_lbl, n_ent):
            attach_tooltip(w, _TIPS["n_rp"], wraplength=_TIP_WRAP)
        for w in (pl_lbl, pl_ent):
            attach_tooltip(w, _TIPS["period_length"], wraplength=_TIP_WRAP)

        frow = ttk.Frame(rp)
        frow.pack(fill="x", padx=8, pady=4)
        cb_sust = ttk.Checkbutton(
            frow, text="Force highest sustained net load",
            variable=self._var_force_sustained,
        )
        cb_sust.pack(side="left")
        cb_peak = ttk.Checkbutton(
            frow, text="Force instantaneous peak",
            variable=self._var_force_peak,
        )
        cb_peak.pack(side="left", padx=(16, 0))
        attach_tooltip(cb_sust, _TIPS["force_sustained"], wraplength=_TIP_WRAP)
        attach_tooltip(cb_peak, _TIPS["force_peak"], wraplength=_TIP_WRAP)

        wrow = ttk.Frame(rp)
        wrow.pack(fill="x", padx=8, pady=4)
        w_lbl = ttk.Label(wrow, text="Window:")
        w_lbl.pack(side="left")
        w_ent = ttk.Entry(wrow, textvariable=self._var_force_window, width=8)
        w_ent.pack(side="left", padx=(4, 16))
        for w in (w_lbl, w_ent):
            attach_tooltip(w, _TIPS["force_window"], wraplength=_TIP_WRAP)

        # Solve checklist (bounded, scrollable — a long solve list must not
        # push the dialog past small-screen height).
        solves_lbl = ttk.Label(rp, text="Solves to (re)build periods for:")
        solves_lbl.pack(anchor="w", padx=8, pady=(6, 0))
        attach_tooltip(solves_lbl, _TIPS["solves"], wraplength=_TIP_WRAP)
        self._build_solve_checklist(rp)

        cb_add = ttk.Checkbutton(
            rp,
            text="Add the new alternative to the selected scenario(s)",
            variable=self._var_add_to_scenario,
        )
        cb_add.pack(anchor="w", padx=8, pady=(2, 2))
        attach_tooltip(cb_add, _TIPS["add_to_scenario"], wraplength=_TIP_WRAP)

        self._rp_button = ttk.Button(
            rp, text="Create new representative periods",
            command=self._on_create_rp,
        )
        self._rp_button.pack(anchor="w", padx=8, pady=(4, 4))

        # RP CLI preview (live). The commands are typically far wider than the
        # dialog, so the Text is wrap="none" with a horizontal scrollbar.
        self._rp_cli_text = self._make_cli_preview(
            rp, "Representative-periods CLI command:", self._on_copy_rp
        )

        # ── separator between the two tools ───────────────────────────
        ttk.Separator(outer, orient="horizontal").pack(
            fill="x", padx=10, pady=10
        )

        # ── 2) Calibration settings ───────────────────────────────────
        cal = ttk.LabelFrame(outer, text="2 · Calibrate investments")
        cal.pack(fill="x", **pad)

        ttk.Label(
            cal, text=_CALIB_EXPLANATION, wraplength=560, justify="left",
        ).pack(fill="x", padx=8, pady=(8, 4))

        crow = ttk.Frame(cal)
        crow.pack(fill="x", padx=8, pady=(4, 4))
        it_lbl = ttk.Label(crow, text="Max. iterations:")
        it_lbl.pack(side="left")
        it_ent = ttk.Entry(crow, textvariable=self._var_max_iterations, width=8)
        it_ent.pack(side="left", padx=(4, 16))
        for w in (it_lbl, it_ent):
            attach_tooltip(w, _TIPS["max_iterations"], wraplength=_TIP_WRAP)
        sz_lbl = ttk.Label(crow, text="Sizing:")
        sz_lbl.pack(side="left")
        rb_timed = ttk.Radiobutton(
            crow, text="timed", value="timed", variable=self._var_sizing,
        )
        rb_timed.pack(side="left", padx=(4, 0))
        rb_uniform = ttk.Radiobutton(
            crow, text="uniform", value="uniform", variable=self._var_sizing,
        )
        rb_uniform.pack(side="left", padx=(4, 0))
        for w in (sz_lbl, rb_timed, rb_uniform):
            attach_tooltip(w, _TIPS["sizing"], wraplength=_TIP_WRAP)

        self._advanced_button = ttk.Button(
            cal, text="Advanced ▸", width=14,
            command=self._toggle_advanced,
        )
        self._advanced_button.pack(anchor="w", padx=8, pady=(4, 2))

        self._advanced_frame = ttk.Frame(cal)
        # Not packed yet — shown by _toggle_advanced.
        self._build_advanced(self._advanced_frame)

        cb_keep = ttk.Checkbutton(
            cal, text="Keep per-iteration artifacts",
            variable=self._var_keep_artifacts,
        )
        cb_keep.pack(anchor="w", padx=8, pady=(4, 2))
        attach_tooltip(cb_keep, _TIPS["keep_artifacts"], wraplength=_TIP_WRAP)

        self._run_button = ttk.Button(
            cal, text="Run calibrations", command=self._on_run,
        )
        self._run_button.pack(anchor="w", padx=8, pady=(4, 4))

        # Calibration CLI preview (live), same wide-scrolling treatment.
        self._calib_cli_text = self._make_cli_preview(
            cal, "Calibration CLI command:", self._on_copy_calib
        )
        # (The reason label + Close button live in the fixed footer, built at
        # the top of this method outside the scroll region.)

    def _build_scroll_body(self) -> ttk.Frame:
        """Create the vertically-scrollable body and return its inner frame.

        A canvas hosts an inner ``ttk.Frame`` (returned) that every section is
        packed into; a vertical scrollbar and the shared wheel handler
        (``_on_wheel``) scroll it. The inner frame is kept exactly as wide as
        the canvas so nothing needs horizontal scrolling at the dialog level
        (the CLI previews scroll horizontally within themselves).
        """
        holder = ttk.Frame(self)
        holder.pack(side="top", fill="both", expand=True)
        canvas = tk.Canvas(holder, highlightthickness=0)
        vsb = ttk.Scrollbar(holder, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = ttk.Frame(canvas, padding=10)
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>", lambda e: canvas.itemconfigure(win, width=e.width)
        )
        self._body_canvas = canvas
        self._scroll_inner = inner
        return inner

    def _on_wheel(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """Route a wheel event to the solve list if over it, else the body.

        Bound once on the toplevel (so it catches wheel events bubbling up from
        any descendant) — no per-widget Enter/Leave juggling. Handles both the
        X11 ``Button-4/5`` events and the ``MouseWheel`` ``delta`` of
        Windows/macOS.
        """
        num = getattr(event, "num", 0)
        if num == 4:
            step = -1
        elif num == 5:
            step = 1
        else:
            step = -int(event.delta / 120)
        target = self._body_canvas
        node = getattr(event, "widget", None)
        solve_canvas = getattr(self, "_solve_canvas", None)
        while node is not None:
            if node is solve_canvas:
                target = solve_canvas
                break
            node = getattr(node, "master", None)
        if target is not None:
            target.yview_scroll(step, "units")

    def _build_advanced(self, frame: ttk.Frame) -> None:
        """Build the (initially hidden) advanced-knobs subframe."""
        def _num_row(label: str, var: tk.StringVar, tip: str) -> None:
            r = ttk.Frame(frame)
            r.pack(fill="x", padx=8, pady=2)
            lbl = ttk.Label(r, text=label, width=26, anchor="w")
            lbl.pack(side="left")
            ent = ttk.Entry(r, textvariable=var, width=10)
            ent.pack(side="left")
            for w in (lbl, ent):
                attach_tooltip(w, tip, wraplength=_TIP_WRAP)

        _num_row(
            "Planning safety margin (%):", self._var_overshoot_pct,
            _TIPS["overshoot"],
        )
        _num_row("Damping first:", self._var_damping_first, _TIPS["damping_first"])
        _num_row(
            "Damping remaining:", self._var_damping_remaining,
            _TIPS["damping_remaining"],
        )
        _num_row(
            "Resource-cap sensitivity:", self._var_stall_fraction,
            _TIPS["stall_fraction"],
        )

    def _toggle_advanced(self) -> None:
        if self._advanced_visible.get():
            self._advanced_frame.pack_forget()
            self._advanced_visible.set(False)
            self._advanced_button.configure(text="Advanced ▸")
        else:
            self._advanced_frame.pack(
                fill="x", after=self._advanced_button, padx=0, pady=(0, 4)
            )
            self._advanced_visible.set(True)
            self._advanced_button.configure(text="Advanced ▾")

    # Max solve rows shown before the checklist starts scrolling.
    _MAX_SOLVE_ROWS = 5

    def _build_solve_checklist(self, parent: tk.Misc) -> None:
        """Render the solve checklist inside a height-bounded, scrollable box.

        The list shows at most ``_MAX_SOLVE_ROWS`` solves; beyond that a
        vertical scrollbar appears and the box stops growing, so a scenario
        with many solves can never stretch the dialog past a small screen.
        """
        if not self._solve_order:
            ttk.Label(
                parent, text="(no solves found for the selected scenarios)",
                foreground="gray",
            ).pack(anchor="w", padx=16, pady=(2, 4))
            return

        from flextool.gui.ui_metrics import get_metrics

        m = get_metrics(self)
        visible = min(len(self._solve_order), self._MAX_SOLVE_ROWS)
        overflow = len(self._solve_order) > self._MAX_SOLVE_ROWS

        container = ttk.Frame(parent)
        container.pack(fill="x", padx=16, pady=(2, 4))
        canvas = tk.Canvas(
            container, highlightthickness=0, height=m.row_height * visible,
        )
        canvas.pack(side="left", fill="x", expand=True)
        sb = ttk.Scrollbar(
            container, orient="vertical", command=canvas.yview
        )
        canvas.configure(yscrollcommand=sb.set)

        rows = ttk.Frame(canvas)
        win = canvas.create_window((0, 0), window=rows, anchor="nw")
        # Keep the scrollregion current and the inner frame as wide as the
        # canvas (so the checkbuttons fill the row and hover reads naturally).
        rows.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfigure(win, width=e.width),
        )
        # The shared toplevel wheel handler (_on_wheel) scrolls this canvas
        # when the pointer is over it — no per-widget wheel binding needed.
        self._solve_canvas = canvas

        for name in self._solve_order:
            cb = ttk.Checkbutton(
                rows, text=name, variable=self._solve_vars[name],
            )
            cb.pack(anchor="w", fill="x")
            attach_tooltip(cb, _TIPS["solves"], wraplength=_TIP_WRAP)

        # The scrollbar is only useful (and only shown) when the list overflows.
        if overflow:
            sb.pack(side="right", fill="y")

    def _make_cli_preview(
        self, parent: tk.Misc, label: str, copy_cmd: Callable[[], None]
    ) -> tk.Text:
        """Build a read-only, horizontally-scrollable CLI-preview + Copy button.

        Returns the disabled :class:`tk.Text` widget so the caller can keep a
        handle for live refreshes. ``wrap="none"`` plus the horizontal
        scrollbar lets the (usually very long) command lines scroll instead of
        being clipped by the dialog width.
        """
        ttk.Label(parent, text=label).pack(anchor="w", padx=8)
        row = ttk.Frame(parent)
        row.pack(fill="both", expand=True, padx=8, pady=(2, 8))
        text_frame = ttk.Frame(row)
        text_frame.pack(side="left", fill="both", expand=True)
        text = tk.Text(text_frame, height=3, wrap="none")
        hsb = ttk.Scrollbar(
            text_frame, orient="horizontal", command=text.xview
        )
        text.configure(xscrollcommand=hsb.set)
        text.pack(side="top", fill="both", expand=True)
        hsb.pack(side="bottom", fill="x")
        text.configure(state="disabled")
        ttk.Button(row, text="Copy", width=8, command=copy_cmd).pack(
            side="left", padx=(6, 0), anchor="n"
        )
        return text

    # Vertical margin left free below/above the window on a small screen
    # (title bar + panel/taskbar) when the content would otherwise fill it.
    _SCREEN_V_MARGIN = 96

    def _size_and_center(self, parent: tk.Misc) -> None:
        """Size the window to content but cap the height to the screen.

        The content width is used as-is; the height is ``min(content, screen -
        margin)`` so on a short screen (≈1024 px) the body scrolls instead of
        the window running off the top and bottom. The window is then centred
        on the parent (placing it on the parent's monitor).

        The body lives inside a Canvas, which does NOT propagate its scrolled
        content's requested size to the toplevel — so the natural size is
        measured from the inner content frame plus the fixed footer, not from
        ``self.winfo_reqheight()`` (which would report only the canvas default).
        """
        try:
            # +18 px leaves room for the vertical scrollbar beside the canvas.
            dw = self._scroll_inner.winfo_reqwidth() + 18
            content_h = (
                self._scroll_inner.winfo_reqheight()
                + self._footer_area.winfo_reqheight()
            )
            screen_h = self.winfo_screenheight()
            cap_h = min(content_h, max(400, screen_h - self._SCREEN_V_MARGIN))
            self.geometry(f"{dw}x{cap_h}")
        except tk.TclError:
            return
        try:
            px = parent.winfo_rootx()
            py = parent.winfo_rooty()
            pw = parent.winfo_width()
            ph = parent.winfo_height()
            x = px + (pw - dw) // 2
            y = max(0, py + (ph - cap_h) // 2)
            self.geometry(f"+{x}+{y}")
        except tk.TclError:
            pass

    # ── Persistence ───────────────────────────────────────────────────
    def _wire_traces(self) -> None:
        """Flush to settings on every widget change (and, later, on close)."""
        every = [
            self._var_n_rp, self._var_period_length, self._var_force_sustained,
            self._var_force_peak, self._var_force_window,
            self._var_add_to_scenario,
            self._var_max_iterations, self._var_sizing, self._var_overshoot_pct,
            self._var_damping_first, self._var_damping_remaining,
            self._var_stall_fraction, self._var_keep_artifacts,
        ]
        for var in (*every, *self._solve_vars.values()):
            var.trace_add("write", lambda *_a: self._flush())

    @staticmethod
    def _as_int(var: tk.StringVar, fallback: int) -> int:
        """Parse *var* as int, keeping *fallback* on partial / bad input."""
        try:
            return int(float(var.get().strip()))
        except (ValueError, tk.TclError):
            return fallback

    @staticmethod
    def _as_float(var: tk.StringVar, fallback: float) -> float:
        try:
            return float(var.get().strip())
        except (ValueError, tk.TclError):
            return fallback

    def _flush(self) -> None:
        """Read every widget into ``settings``, persist, and refresh previews.

        Numeric fields tolerate mid-edit / empty text by keeping the last good
        value, so a partially typed entry never wipes a setting.
        """
        s = self._settings
        s.calib_rp_n_rp = self._as_int(self._var_n_rp, s.calib_rp_n_rp)
        s.calib_rp_period_length = self._as_int(
            self._var_period_length, s.calib_rp_period_length
        )
        s.calib_rp_force_sustained = bool(self._var_force_sustained.get())
        s.calib_rp_force_peak = bool(self._var_force_peak.get())
        s.calib_rp_force_window = self._as_int(
            self._var_force_window, s.calib_rp_force_window
        )
        s.calib_rp_add_to_scenario = bool(self._var_add_to_scenario.get())
        s.calib_max_iterations = self._as_int(
            self._var_max_iterations, s.calib_max_iterations
        )
        sizing = self._var_sizing.get()
        if sizing in ("timed", "uniform"):
            s.calib_sizing = sizing
        s.calib_overshoot_pct = self._as_float(
            self._var_overshoot_pct, s.calib_overshoot_pct
        )
        s.calib_damping_first = self._as_float(
            self._var_damping_first, s.calib_damping_first
        )
        s.calib_damping_remaining = self._as_float(
            self._var_damping_remaining, s.calib_damping_remaining
        )
        s.calib_stall_fraction = self._as_float(
            self._var_stall_fraction, s.calib_stall_fraction
        )
        s.calib_keep_artifacts = bool(self._var_keep_artifacts.get())
        s.calib_selected_solves = self._selected_solves()

        self._refresh_cli_preview()

        try:
            self._save_settings()
        except Exception:  # persistence must never break the UI
            logger.exception("Saving calibrate settings failed")

    # ── Enable / disable ──────────────────────────────────────────────
    def _refresh_run_state(self) -> None:
        """Set the disabled reason and the button states.

        Both action buttons are disabled when there are no scenarios or ANY
        scenario is xlsx-backed (its DB writes would be discarded). The
        per-action in-flight guard disables only the just-clicked button.
        """
        xlsx = [
            self._scenario_name(sc)
            for sc in self._scenarios
            if getattr(sc, "is_xlsx", False)
        ]
        if not self._scenarios:
            reason = "No scenarios selected — nothing to calibrate."
        elif xlsx:
            reason = (
                "Disabled: xlsx-backed scenario(s) "
                f"{', '.join(xlsx)} regenerate their database on each run, "
                "which would discard the representative-period / calibration "
                "writes. Convert them to a Spine database first."
            )
        else:
            reason = ""
        self._reason_var.set(reason)

        base_enabled = reason == ""
        self._rp_button.configure(
            state="normal" if base_enabled and not self._rp_guarded
            else "disabled"
        )
        self._run_button.configure(
            state="normal" if base_enabled and not self._calib_guarded
            else "disabled"
        )

    def _guard_rp(self) -> None:
        self._rp_guarded = True
        self._refresh_run_state()
        self.after(_GUARD_MS, self._unguard_rp)

    def _unguard_rp(self) -> None:
        self._rp_guarded = False
        if self.winfo_exists():
            self._refresh_run_state()

    def _guard_calib(self) -> None:
        self._calib_guarded = True
        self._refresh_run_state()
        self.after(_GUARD_MS, self._unguard_calib)

    def _unguard_calib(self) -> None:
        self._calib_guarded = False
        if self.winfo_exists():
            self._refresh_run_state()

    def _reveal_execution_window(self) -> None:
        """Open / raise the Execution-jobs window so the launch is visible."""
        if self._show_execution_window is None:
            return
        try:
            self._show_execution_window()
        except Exception:  # never let a UI-raise failure break the launch
            logger.exception("Opening the execution window failed")

    # ── RP flow ───────────────────────────────────────────────────────
    def _on_create_rp(self) -> None:
        """Launch one RP-preprocess job per runnable scenario.

        Each job carries an ``on_success(scenario)`` hook — enabled only when
        "Add the new alternative to the selected scenario(s)" is ticked — that,
        AFTER the RP subprocess finishes successfully, on the worker thread,
        appends the freshly written RP alternative onto that scenario's stack
        so a subsequent calibration sees the new periods.
        """
        self._flush()
        s = self._settings
        add_to_scenario = bool(self._var_add_to_scenario.get())
        # Reserve (commit) the per-scenario names so the preview immediately
        # advances (base → base_2) and a second launch cannot collide.
        plan = self._rp_plan(commit=True)
        if not plan:
            return
        specs: list[RpJobSpec] = []
        for _sc, name, db_url, applicable, alt, desc in plan:
            on_success = (
                (lambda scenario, u=db_url, a=alt:
                 add_alternative_to_scenario(u, scenario, a))
                if add_to_scenario else None
            )
            specs.append(
                RpJobSpec(
                    db_url=db_url,
                    scenario=name,
                    n_rp=s.calib_rp_n_rp,
                    period_length=s.calib_rp_period_length,
                    force_sustained=s.calib_rp_force_sustained,
                    force_peak=s.calib_rp_force_peak,
                    force_window=s.calib_rp_force_window,
                    solves=applicable,
                    alternative_name=alt,
                    alternative_description=desc,
                    on_success=on_success,
                )
            )
        launch_rp_jobs(
            self._execution_mgr,
            python_exe=self._python_exe,
            project_path=self._project_path,
            jobs=specs,
        )
        self._guard_rp()
        self._reveal_execution_window()
        self._refresh_cli_preview()

    # ── Run flow ──────────────────────────────────────────────────────
    def _on_run(self) -> None:
        """Launch one calibration job per runnable scenario."""
        self._flush()
        runnable = self._runnable_scenarios()
        if not runnable:
            return
        s = self._settings
        specs: list[CalibJobSpec] = []
        for sc in runnable:
            specs.append(
                CalibJobSpec(
                    db_url=self._resolve_db_url(sc),
                    scenario=self._scenario_name(sc),
                    iterations=s.calib_max_iterations,
                    sizing=s.calib_sizing,
                    overshoot_pct=s.calib_overshoot_pct,
                    damping_first=s.calib_damping_first,
                    damping_remaining=s.calib_damping_remaining,
                    stall_fraction=s.calib_stall_fraction,
                    debug=s.calib_keep_artifacts,
                    # Regenerate the same formats a regular run would (the
                    # project's "File outputs" choices), from the final parquet.
                    final_write_methods=final_write_methods_from_settings(s),
                )
            )
        launch_calibration_jobs(
            self._execution_mgr,
            python_exe=self._python_exe,
            project_path=self._project_path,
            jobs=specs,
        )
        self._guard_calib()
        self._reveal_execution_window()

    # ── CLI previews ──────────────────────────────────────────────────
    def _render_rp_cli(self) -> str:
        """Render the RP command(s) exactly as "Create…" would run them."""
        s = self._settings
        plan = self._rp_plan(commit=False)
        if not plan:
            # Distinguish "no runnable scenario" from "no selected solve
            # applies" so the empty preview is not mysterious.
            if self._runnable_scenarios():
                return (
                    "# No selected solve belongs to the selected scenario(s) — "
                    "tick a solve that the scenario actually runs."
                )
            return ""
        lines: list[str] = []
        for _sc, name, db_url, applicable, alt, desc in plan:
            argv = build_rp_command(
                self._python_exe, db_url, name,
                n_rp=s.calib_rp_n_rp,
                period_length=s.calib_rp_period_length,
                force_sustained=s.calib_rp_force_sustained,
                force_peak=s.calib_rp_force_peak,
                force_window=s.calib_rp_force_window,
                solves=applicable,
                alternative_name=alt,
                alternative_description=desc,
            )
            lines.append(command_to_display_string(argv))
        return "\n".join(lines)

    def _render_calib_cli(self) -> str:
        """Render the calibrate command(s) exactly as "Run…" would run them."""
        s = self._settings
        lines: list[str] = []
        for sc in self._runnable_scenarios():
            name = self._scenario_name(sc)
            db_url = self._resolve_db_url(sc)
            # create=False: previewing must not litter empty work directories.
            warm, work, out = _calib_dirs(
                self._project_path, name, create=False
            )
            argv = build_calibrate_command(
                self._python_exe, db_url, name,
                iterations=s.calib_max_iterations,
                sizing=s.calib_sizing,
                overshoot=overshoot_pct_to_multiplier(s.calib_overshoot_pct),
                damping_first=s.calib_damping_first,
                damping_remaining=s.calib_damping_remaining,
                stall_fraction=s.calib_stall_fraction,
                warm_start_cache_dir=warm,
                work_dir=work,
                output_location=out,
                debug=s.calib_keep_artifacts,
                final_write_methods=final_write_methods_from_settings(s),
            )
            lines.append(command_to_display_string(argv))
        return "\n".join(lines)

    @staticmethod
    def _set_text(widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _refresh_cli_preview(self) -> None:
        """Re-render both live CLI previews from the current settings."""
        # Guard: called from _flush, which can fire via a trace before the text
        # widgets are built.
        if not hasattr(self, "_rp_cli_text"):
            return
        try:
            self._set_text(self._rp_cli_text, self._render_rp_cli())
            self._set_text(self._calib_cli_text, self._render_calib_cli())
        except tk.TclError:
            pass

    def _on_copy_rp(self) -> None:
        self._flush()
        self.clipboard_clear()
        self.clipboard_append(self._render_rp_cli())

    def _on_copy_calib(self) -> None:
        self._flush()
        self.clipboard_clear()
        self.clipboard_append(self._render_calib_cli())

    # ── Close ─────────────────────────────────────────────────────────
    def _on_close(self) -> None:
        self._flush()
        self.destroy()
