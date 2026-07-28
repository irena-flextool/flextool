"""Non-modal "Calibrate investments" dialog.

The dialog gathers the representative-period build knobs and the calibration
controls for one or more scenarios and turns them into auxiliary jobs via
:mod:`flextool.gui.calibrate_jobs` — it never renders a command line itself
(the single argv renderers in :mod:`flextool.gui.calibrate_commands` do that,
for both the launcher and the "Copy CLI command" preview, so the copied text
can never drift from what runs).

Why non-modal
-------------
The dialog launches long-running solves that stream into the Execution window;
the user must be able to watch those while the dialog stays open. It is
therefore ``transient`` to its parent (stacks with it, no taskbar entry on
some WMs) but takes NO ``grab_set`` — matching the repo rule that ``grab_set``
in a Toplevel constructor can abort "not viewable", and the non-modal picker
precedent (``PlotSettingsPicker``).

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
    overshoot_pct_to_multiplier,
)
from flextool.gui.calibrate_jobs import (
    CalibJobSpec,
    RpJobSpec,
    _calib_dirs,
    launch_calibration_jobs,
    launch_rp_jobs,
)
from flextool.gui.solve_reader import read_scenario_solves
from flextool.representative_periods.scenario_stack import (
    add_alternative_to_scenario,
)

logger = logging.getLogger(__name__)

# How long (ms) the action buttons stay disabled after a launch, to swallow
# an accidental double-click without permanently locking the button (the
# launcher itself serialises, but a duplicate submit would spawn a duplicate
# job with the same action_key).
_GUARD_MS = 2500

_COUNT_MODES = ("grow", "fixed")


def _rp_alt_name(scenario: str) -> str:
    """Per-scenario RP alternative name (must match the RP flow + Copy CLI)."""
    return f"{scenario}_rp"


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

        # Non-modal: transient (stacks with the parent) but NO grab_set.
        self.transient(parent)

        # In-flight guards: once a launch fires the matching button is disabled
        # for _GUARD_MS to swallow a double-click, then re-enabled.
        self._rp_guarded = False
        self._calib_guarded = False

        # Tk variables (created before traces are wired).
        self._var_n_rp = tk.StringVar()
        self._var_period_length = tk.StringVar()
        self._var_force_sustained = tk.BooleanVar()
        self._var_force_peak = tk.BooleanVar()
        self._var_force_window = tk.StringVar()
        self._var_count_mode = tk.StringVar()
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

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Escape>", lambda _e: self._on_close())

        # Size to content and centre on the parent (which places the window on
        # the parent's monitor). Guarded: a headless / detached parent may not
        # report a geometry.
        self.update_idletasks()
        self._center_on_parent(parent)

    # ── Seed / model helpers ──────────────────────────────────────────
    def _seed_from_settings(self) -> None:
        """Initialise every scalar widget var from the settings object."""
        s = self._settings
        self._var_n_rp.set(str(s.calib_rp_n_rp))
        self._var_period_length.set(str(s.calib_rp_period_length))
        self._var_force_sustained.set(bool(s.calib_rp_force_sustained))
        self._var_force_peak.set(bool(s.calib_rp_force_peak))
        self._var_force_window.set(str(s.calib_rp_force_window))
        self._var_count_mode.set(s.calib_rp_count_mode or "grow")
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
        for sc in self._scenarios:
            if getattr(sc, "is_xlsx", False):
                continue
            try:
                url = self._resolve_db_url(sc)
                solves = read_scenario_solves(url, self._scenario_name(sc))
            except Exception as exc:  # a bad DB must not break the dialog
                logger.warning(
                    "Could not read solves for scenario %r: %s",
                    self._scenario_name(sc), exc,
                )
                continue
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

    # ── Layout ────────────────────────────────────────────────────────
    def _build_widgets(self) -> None:
        pad = dict(padx=10, pady=6)

        outer = ttk.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)

        # ── 1) Representative periods ─────────────────────────────────
        rp = ttk.LabelFrame(outer, text="Representative periods (optional)")
        rp.pack(fill="x", **pad)

        row = ttk.Frame(rp)
        row.pack(fill="x", padx=8, pady=(8, 4))
        ttk.Label(row, text="Periods (n_rp):").pack(side="left")
        ttk.Entry(row, textvariable=self._var_n_rp, width=8).pack(
            side="left", padx=(4, 16)
        )
        ttk.Label(row, text="Period length (steps):").pack(side="left")
        ttk.Entry(row, textvariable=self._var_period_length, width=8).pack(
            side="left", padx=(4, 0)
        )

        frow = ttk.Frame(rp)
        frow.pack(fill="x", padx=8, pady=4)
        ttk.Checkbutton(
            frow, text="Force highest sustained net load",
            variable=self._var_force_sustained,
        ).pack(side="left")
        ttk.Checkbutton(
            frow, text="Force instantaneous peak",
            variable=self._var_force_peak,
        ).pack(side="left", padx=(16, 0))

        wrow = ttk.Frame(rp)
        wrow.pack(fill="x", padx=8, pady=4)
        ttk.Label(wrow, text="Window:").pack(side="left")
        ttk.Entry(wrow, textvariable=self._var_force_window, width=8).pack(
            side="left", padx=(4, 16)
        )
        ttk.Label(wrow, text="Count mode:").pack(side="left")
        ttk.Combobox(
            wrow, textvariable=self._var_count_mode,
            values=list(_COUNT_MODES), width=8, state="readonly",
        ).pack(side="left", padx=(4, 0))

        # Solve checklist.
        ttk.Label(rp, text="Solves to (re)build periods for:").pack(
            anchor="w", padx=8, pady=(6, 0)
        )
        checks = ttk.Frame(rp)
        checks.pack(fill="x", padx=16, pady=(2, 4))
        if self._solve_order:
            for name in self._solve_order:
                ttk.Checkbutton(
                    checks, text=name, variable=self._solve_vars[name],
                ).pack(anchor="w")
        else:
            ttk.Label(
                checks, text="(no solves found for the selected scenarios)",
                foreground="gray",
            ).pack(anchor="w")

        self._rp_button = ttk.Button(
            rp, text="Create new representative periods",
            command=self._on_create_rp,
        )
        self._rp_button.pack(anchor="w", padx=8, pady=(4, 8))

        # ── 2) Calibration settings ───────────────────────────────────
        cal = ttk.LabelFrame(outer, text="Calibration settings")
        cal.pack(fill="x", **pad)

        crow = ttk.Frame(cal)
        crow.pack(fill="x", padx=8, pady=(8, 4))
        ttk.Label(crow, text="Max. iterations:").pack(side="left")
        ttk.Entry(crow, textvariable=self._var_max_iterations, width=8).pack(
            side="left", padx=(4, 16)
        )
        ttk.Label(crow, text="Sizing:").pack(side="left")
        ttk.Radiobutton(
            crow, text="timed", value="timed", variable=self._var_sizing,
        ).pack(side="left", padx=(4, 0))
        ttk.Radiobutton(
            crow, text="uniform", value="uniform", variable=self._var_sizing,
        ).pack(side="left", padx=(4, 0))

        self._advanced_button = ttk.Button(
            cal, text="Advanced ▸", width=14,
            command=self._toggle_advanced,
        )
        self._advanced_button.pack(anchor="w", padx=8, pady=(4, 2))

        self._advanced_frame = ttk.Frame(cal)
        # Not packed yet — shown by _toggle_advanced.
        self._build_advanced(self._advanced_frame)

        # ── 3) Footer ─────────────────────────────────────────────────
        footer = ttk.Frame(outer)
        footer.pack(fill="x", **pad)

        ttk.Checkbutton(
            footer, text="Keep per-iteration artifacts",
            variable=self._var_keep_artifacts,
        ).pack(side="left")

        self._run_button = ttk.Button(
            footer, text="Run calibrations", command=self._on_run,
        )
        self._run_button.pack(side="right")
        ttk.Button(footer, text="Cancel", command=self._on_close).pack(
            side="right", padx=(0, 6)
        )
        ttk.Button(
            footer, text="Copy CLI command", command=self._on_copy,
        ).pack(side="right", padx=(0, 6))

        # Reason label (why the action buttons are disabled) + CLI preview.
        ttk.Label(
            outer, textvariable=self._reason_var, foreground="gray",
            wraplength=520, justify="left",
        ).pack(fill="x", padx=10, pady=(0, 4))

        ttk.Label(outer, text="CLI preview (copied to clipboard):").pack(
            anchor="w", padx=10
        )
        self._cli_text = tk.Text(outer, height=6, wrap="none")
        self._cli_text.pack(fill="both", expand=True, padx=10, pady=(2, 8))
        self._cli_text.configure(state="disabled")

    def _build_advanced(self, frame: ttk.Frame) -> None:
        """Build the (initially hidden) advanced-knobs subframe."""
        def _num_row(label: str, var: tk.StringVar) -> None:
            r = ttk.Frame(frame)
            r.pack(fill="x", padx=8, pady=2)
            ttk.Label(r, text=label, width=26, anchor="w").pack(side="left")
            ttk.Entry(r, textvariable=var, width=10).pack(side="left")

        _num_row("Planning safety margin (%):", self._var_overshoot_pct)
        _num_row("Damping first:", self._var_damping_first)
        _num_row("Damping remaining:", self._var_damping_remaining)
        _num_row("Resource-cap sensitivity:", self._var_stall_fraction)

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

    def _center_on_parent(self, parent: tk.Misc) -> None:
        try:
            px = parent.winfo_rootx()
            py = parent.winfo_rooty()
            pw = parent.winfo_width()
            ph = parent.winfo_height()
            dw = self.winfo_reqwidth()
            dh = self.winfo_reqheight()
            self.geometry(f"+{px + (pw - dw) // 2}+{py + (ph - dh) // 2}")
        except tk.TclError:
            pass

    # ── Persistence ───────────────────────────────────────────────────
    def _wire_traces(self) -> None:
        """Flush to settings on every widget change (and, later, on close)."""
        every = [
            self._var_n_rp, self._var_period_length, self._var_force_sustained,
            self._var_force_peak, self._var_force_window, self._var_count_mode,
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
        """Read every widget into ``settings`` and persist.

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
        mode = self._var_count_mode.get()
        if mode in _COUNT_MODES:
            s.calib_rp_count_mode = mode
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

    # ── RP flow ───────────────────────────────────────────────────────
    def _on_create_rp(self) -> None:
        """Launch one RP-preprocess job per runnable scenario.

        Each job carries an ``on_success(scenario)`` hook that — AFTER the RP
        subprocess finishes successfully, on the worker thread — appends the
        freshly written RP alternative onto that scenario's stack, so a
        subsequent calibration sees the new periods.
        """
        self._flush()
        runnable = self._runnable_scenarios()
        if not runnable:
            return
        s = self._settings
        solves = self._selected_solves()
        specs: list[RpJobSpec] = []
        for sc in runnable:
            name = self._scenario_name(sc)
            db_url = self._resolve_db_url(sc)
            alt = _rp_alt_name(name)
            specs.append(
                RpJobSpec(
                    db_url=db_url,
                    scenario=name,
                    n_rp=s.calib_rp_n_rp,
                    period_length=s.calib_rp_period_length,
                    force_sustained=s.calib_rp_force_sustained,
                    force_peak=s.calib_rp_force_peak,
                    force_window=s.calib_rp_force_window,
                    count_mode=s.calib_rp_count_mode,
                    solves=solves,
                    alternative_name=alt,
                    on_success=(
                        lambda scenario, u=db_url, a=alt:
                        add_alternative_to_scenario(u, scenario, a)
                    ),
                )
            )
        launch_rp_jobs(
            self._execution_mgr,
            python_exe=self._python_exe,
            project_path=self._project_path,
            jobs=specs,
        )
        self._guard_rp()

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
                )
            )
        launch_calibration_jobs(
            self._execution_mgr,
            python_exe=self._python_exe,
            project_path=self._project_path,
            jobs=specs,
        )
        self._guard_calib()

    # ── Copy CLI ──────────────────────────────────────────────────────
    def _on_copy(self) -> None:
        """Render the RP + calibrate commands per scenario, copy + show them.

        Uses the SAME argv builders and directory derivation as the launchers,
        so the copied text is exactly what "Create…" / "Run…" execute.
        """
        self._flush()
        text = self._render_cli()
        self.clipboard_clear()
        self.clipboard_append(text)
        self._cli_text.configure(state="normal")
        self._cli_text.delete("1.0", "end")
        self._cli_text.insert("1.0", text)
        self._cli_text.configure(state="disabled")

    def _render_cli(self) -> str:
        s = self._settings
        solves = self._selected_solves()
        lines: list[str] = []
        for sc in self._runnable_scenarios():
            name = self._scenario_name(sc)
            db_url = self._resolve_db_url(sc)
            alt = _rp_alt_name(name)
            rp_argv = build_rp_command(
                self._python_exe, db_url, name,
                n_rp=s.calib_rp_n_rp,
                period_length=s.calib_rp_period_length,
                force_sustained=s.calib_rp_force_sustained,
                force_peak=s.calib_rp_force_peak,
                force_window=s.calib_rp_force_window,
                count_mode=s.calib_rp_count_mode,
                solves=solves,
                alternative_name=alt,
            )
            lines.append(command_to_display_string(rp_argv))

            warm, work, out = _calib_dirs(self._project_path, name)
            cal_argv = build_calibrate_command(
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
            )
            lines.append(command_to_display_string(cal_argv))
        return "\n".join(lines)

    # ── Close ─────────────────────────────────────────────────────────
    def _on_close(self) -> None:
        self._flush()
        self.destroy()
