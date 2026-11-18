#!/usr/bin/env python3
"""
GelMA DS reverse model - GUI
============================
Desktop interface for the Gaussian Process model in
gelma_ds_reverse_model.py (Shirahama et al. 2016, Sci. Rep. 6:31036).

Enter a target degree of substitution (DS %) and gelatin mass, and the
app returns the recommended synthesis conditions, model prediction,
and extrapolation warnings. A forward mode (ratio -> DS) is included.

Requires only tkinter (bundled with python.org Python) plus the
existing model file. Run with:  .venv/bin/python gelma_ds_gui.py
"""

import csv
import tkinter as tk
from datetime import datetime
from tkinter import ttk, messagebox, filedialog

import numpy as np
from matplotlib import ticker
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from gelma_ds_reverse_model import (
    reverse_ds, predict_ds, OPTIMAL, SENSITIVITY,
    DS_MIN, DS_MAX, RATIO_MIN, RATIO_MAX,
    GP, FEED_RATIO, DS_OBSERVED,
)

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
BG = "#1e1f24"
PANEL = "#26272e"
FG = "#e8e8ea"
MUTED = "#9a9ba3"
ACCENT = "#4f8ef7"
WARN = "#e0a13e"
FONT = ("SF Pro Text", 13)
FONT_B = ("SF Pro Text", 13, "bold")
MONO = ("Menlo", 12)


def _fmt_conditions(ratio, gelatin_g, ds_hat, ds_std):
    """Build the results text for a reverse prediction."""
    mmol_maa = ratio * 1.035 / 154.16 * 1000  # per g gelatin
    lines = [
        ("RECOMMENDED CONDITIONS", True),
        (f"  MAA/gelatin feed ratio : {ratio:.4f} mL/g", False),
        (f"  MAA volume           : {ratio * gelatin_g:.2f} mL "
         f"for {gelatin_g:g} g gelatin", False),
        (f"  Gelatin concentration: {OPTIMAL['gelatin_wv_pct']:g} w/v % "
         f"({gelatin_g:g} g in {gelatin_g / OPTIMAL['gelatin_wv_pct'] * 100:.0f} mL "
         f"of 0.25 M CB buffer)", False),
        (f"  CB buffer            : {OPTIMAL['CB_buffer_M']} M, adjust to "
         f"pH {OPTIMAL['initial_pH']:g} before adding MAA", False),
        (f"  Temperature          : {OPTIMAL['temperature_C']:g} °C "
         f"(35–50 °C all equivalent)", False),
        (f"  Reaction time        : {OPTIMAL['reaction_time_min']:g} min "
         f"(quench by dilution/cooling, then dialyze)", False),
        ("", False),
        (f"  Predicted DS at this ratio: {ds_hat:.1f} ± {ds_std:.1f} %", False),
        (f"  ≈{mmol_maa / 0.286:.2f}× molar excess of MAA over "
         f"gelatin free –NH₂ groups", False),
    ]
    return lines


class GelMAApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GelMA DS Reverse Model")
        self.configure(bg=BG)
        self.resizable(True, True)

        self._style()
        self._build_inputs()
        self._build_output()
        self._build_plot()
        self._build_history()
        self._plot_curve()

    # ------------------------------------------------------------------ UI
    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TFrame", background=BG)
        s.configure("Panel.TFrame", background=PANEL)
        s.configure("TLabel", background=BG, foreground=FG, font=FONT)
        s.configure("Muted.TLabel", background=BG, foreground=MUTED,
                    font=("SF Pro Text", 11))
        s.configure("Title.TLabel", background=BG, foreground=FG,
                    font=("SF Pro Text", 17, "bold"))
        s.configure("TEntry", fieldbackground=PANEL, foreground=FG,
                    insertcolor=FG)
        s.configure("Accent.TButton", font=FONT_B)
        s.map("Accent.TButton",
              background=[("active", "#3a6fd8")], foreground=[("active", "#ffffff")])
        s.configure("Treeview", background=PANEL, fieldbackground=PANEL,
                    foreground=FG, font=("SF Pro Text", 11), rowheight=22,
                    borderwidth=0)
        s.configure("Treeview.Heading", background=BG, foreground=MUTED,
                    font=("SF Pro Text", 11, "bold"))
        s.map("Treeview", background=[("selected", "#3a4a66")],
              foreground=[("selected", FG)])

    def _build_inputs(self):
        top = ttk.Frame(self, padding=(20, 18, 20, 6))
        top.grid(row=0, column=0, sticky="ew")

        ttk.Label(top, text="GelMA one-pot synthesis — GP model",
                  style="Title.TLabel").grid(row=0, column=0, columnspan=4,
                                             sticky="w")
        ttk.Label(top, text="Shirahama et al. 2016, Sci. Rep. 6:31036",
                  style="Muted.TLabel").grid(row=1, column=0, columnspan=4,
                                             sticky="w", pady=(0, 12))

        ttk.Label(top, text="Target DS (%):").grid(row=2, column=0, sticky="w")
        self.ds_var = tk.StringVar(value="80")
        ttk.Entry(top, textvariable=self.ds_var, width=8).grid(
            row=2, column=1, sticky="w", padx=(8, 24))

        ttk.Label(top, text="Gelatin (g):").grid(row=2, column=2, sticky="w")
        self.gel_var = tk.StringVar(value="10")
        ttk.Entry(top, textvariable=self.gel_var, width=8).grid(
            row=2, column=3, sticky="w", padx=(8, 0))

        btns = ttk.Frame(top)
        btns.grid(row=3, column=0, columnspan=4, sticky="w", pady=(14, 2))
        rev = ttk.Button(btns, text="Calculate conditions",
                         style="Accent.TButton", command=self.on_reverse)
        rev.grid(row=0, column=0)
        ttk.Button(btns, text="Clear", command=self.on_clear).grid(
            row=0, column=1, padx=(10, 0))

        # Forward mode row
        fwd = ttk.Frame(top)
        fwd.grid(row=4, column=0, columnspan=4, sticky="w", pady=(14, 0))
        ttk.Label(fwd, text="Forward check — feed ratio (mL/g):",
                  style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        self.ratio_var = tk.StringVar(value="0.05")
        ttk.Entry(fwd, textvariable=self.ratio_var, width=8).grid(
            row=0, column=1, padx=(8, 10))
        ttk.Button(fwd, text="Predict DS", command=self.on_forward).grid(
            row=0, column=2)

        self.bind("<Return>", lambda _e: self.on_reverse())

        # Full-conditions prediction (Tier 1 GP + Tier 2 sensitivity)
        full = ttk.Frame(top)
        full.grid(row=5, column=0, columnspan=4, sticky="w", pady=(14, 0))
        ttk.Label(full, text="Full conditions → predicted DS "
                             "(enter X for unknown):",
                  style="Muted.TLabel").grid(row=0, column=0, columnspan=9,
                                             sticky="w", pady=(0, 4))

        self.f_vars = {}
        defaults = {"ratio": "0.05", "cb": "0.25", "ph": "9",
                    "wv": "10", "temp": "45", "time": "60"}
        labels = [("ratio", "Ratio (mL/g)"), ("cb", "CB (M)"),
                  ("ph", "pH"), ("wv", "Gelatin (w/v %)"),
                  ("temp", "Temp (°C)"), ("time", "Time (min)")]
        for i, (key, lab) in enumerate(labels):
            r, c = divmod(i, 3)
            ttk.Label(full, text=lab, style="Muted.TLabel").grid(
                row=1 + r, column=c * 2, sticky="w", padx=(0, 6),
                pady=(0, 4) if r == 0 else (0, 0))
            var = tk.StringVar(value=defaults[key])
            self.f_vars[key] = var
            ttk.Entry(full, textvariable=var, width=7).grid(
                row=1 + r, column=c * 2 + 1, sticky="w",
                padx=(0, 18), pady=(0, 4) if r == 0 else (0, 0))
        ttk.Button(full, text="Predict from conditions",
                   command=self.on_full).grid(row=2, column=4, columnspan=2,
                                              sticky="w", pady=(4, 0))

    def _build_output(self):
        box = ttk.Frame(self, padding=(20, 8, 20, 18))
        box.grid(row=1, column=0, sticky="nsew")

        self.out = tk.Text(box, width=78, height=20, bg=PANEL, fg=FG,
                           font=MONO, relief="flat", padx=14, pady=12,
                           insertbackground=FG, state="disabled",
                           spacing1=1, spacing3=1)
        self.out.grid(row=0, column=0, sticky="nsew")
        self.out.tag_configure("head", foreground=ACCENT, font=FONT_B)
        self.out.tag_configure("warn", foreground=WARN)
        self.out.tag_configure("muted", foreground=MUTED)

        sb = ttk.Scrollbar(box, orient="vertical", command=self.out.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.out.configure(yscrollcommand=sb.set)

        self._write([("Enter a target DS above and press "
                      "'Calculate conditions'.", "muted")])

    def _build_plot(self):
        box = ttk.Frame(self, padding=(0, 8, 18, 18))
        box.grid(row=1, column=1, sticky="nsew")
        self.rowconfigure(1, weight=1)
        self.columnconfigure(1, weight=1)
        box.rowconfigure(0, weight=1)
        box.columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(5.2, 4.6), dpi=100, facecolor=BG)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=box)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        ttk.Button(box, text="Save figure…", command=self.on_save_figure).grid(
            row=1, column=0, sticky="e", pady=(8, 0))

    # ------------------------------------------------------------ save figure
    def on_save_figure(self):
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Save figure",
            defaultextension=".png",
            initialfile="gelma_ds_curve.png",
            filetypes=[("PNG image", "*.png"),
                       ("PDF document", "*.pdf"),
                       ("SVG image", "*.svg")],
        )
        if not path:
            return
        try:
            self.fig.savefig(path, dpi=200,
                             facecolor=self.fig.get_facecolor())
        except OSError as exc:
            messagebox.showerror("Save failed", f"Could not save the figure:\n{exc}")
            return
        messagebox.showinfo("Figure saved", f"Saved to:\n{path}")

    # -------------------------------------------------------------- history
    def _build_history(self):
        box = ttk.Frame(self, padding=(20, 0, 18, 18))
        box.grid(row=2, column=0, columnspan=2, sticky="nsew")
        self.rowconfigure(2, weight=1)
        box.columnconfigure(0, weight=1)
        box.rowconfigure(1, weight=1)

        ttk.Label(box, text="Calculation history",
                  style="Muted.TLabel").grid(row=0, column=0, sticky="w",
                                             pady=(0, 6))

        cols = ("time", "mode", "input", "ratio", "maa", "pred")
        heads = ("Time", "Mode", "Input", "Ratio (mL/g)", "MAA (mL)",
                 "Predicted DS (%)")
        widths = (70, 80, 170, 110, 90, 130)
        self.history = ttk.Treeview(box, columns=cols, show="headings",
                                    height=6)
        for c, h, w in zip(cols, heads, widths):
            self.history.heading(c, text=h)
            self.history.column(c, width=w, anchor="w", stretch=(c == "input"))
        self.history.grid(row=1, column=0, sticky="nsew")

        sb = ttk.Scrollbar(box, orient="vertical", command=self.history.yview)
        sb.grid(row=1, column=1, sticky="ns")
        self.history.configure(yscrollcommand=sb.set)

        btns = ttk.Frame(box)
        btns.grid(row=2, column=0, columnspan=2, sticky="e", pady=(8, 0))
        ttk.Button(btns, text="Copy selected",
                   command=self.on_copy_history).grid(row=0, column=0)
        ttk.Button(btns, text="Export CSV…",
                   command=self.on_export_history).grid(row=0, column=1,
                                                        padx=(10, 0))
        ttk.Button(btns, text="Clear history",
                   command=self.on_clear_history).grid(row=0, column=2,
                                                       padx=(10, 0))

    HISTORY_COLS = ("Time", "Mode", "Input", "Ratio (mL/g)", "MAA (mL)",
                    "Predicted DS (%)")

    def _log_history(self, mode, input_str, ratio_str, maa_str, pred_str):
        row = (datetime.now().strftime("%H:%M:%S"), mode, input_str,
               ratio_str, maa_str, pred_str)
        iid = self.history.insert("", "end", values=row)
        self.history.see(iid)

    def _history_rows(self, only_selected=False):
        items = (self.history.selection() if only_selected
                 else self.history.get_children())
        return [self.history.item(i, "values") for i in items]

    def on_copy_history(self):
        rows = self._history_rows(only_selected=True)
        if not rows:
            messagebox.showinfo("Nothing selected",
                                "Select one or more history rows first.")
            return
        text = "\t".join(self.HISTORY_COLS) + "\n"
        text += "\n".join("\t".join(r) for r in rows)
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update()  # keep clipboard after window closes

    def on_export_history(self):
        rows = self._history_rows()
        if not rows:
            messagebox.showinfo("History empty", "No calculations to export yet.")
            return
        path = filedialog.asksaveasfilename(
            parent=self, title="Export history", defaultextension=".csv",
            initialfile="gelma_ds_history.csv",
            filetypes=[("CSV file", "*.csv")])
        if not path:
            return
        try:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(self.HISTORY_COLS)
                w.writerows(rows)
        except OSError as exc:
            messagebox.showerror("Export failed", f"Could not write the file:\n{exc}")
            return
        messagebox.showinfo("History exported", f"Saved to:\n{path}")

    def on_clear_history(self):
        for i in self.history.get_children():
            self.history.delete(i)

    # ------------------------------------------------------------- plotting
    def _plot_curve(self, marker_ratio=None, marker_ds=None,
                    marker_label=None):
        ax = self.ax
        ax.clear()
        ax.set_facecolor(PANEL)

        grid = np.logspace(np.log10(RATIO_MIN), np.log10(RATIO_MAX), 400)
        mu, std = GP.predict(np.log10(grid).reshape(-1, 1), return_std=True)

        ax.fill_between(grid, mu - std, mu + std,
                        color=ACCENT, alpha=0.25, lw=0, label="±1σ uncertainty")
        ax.plot(grid, mu, color=ACCENT, lw=2, label="GP model")
        ax.plot(FEED_RATIO, DS_OBSERVED, "o", color=FG, ms=7,
                label="Paper data (Series 3)")

        if marker_ratio is not None and marker_ds is not None:
            ax.plot([marker_ratio], [marker_ds], marker="*", color=WARN,
                    ms=18, mec=BG, mew=1.2, zorder=5,
                    label=marker_label or "Your point")
            ax.axhline(marker_ds, color=WARN, lw=0.8, ls=":", alpha=0.7)
            ax.axvline(marker_ratio, color=WARN, lw=0.8, ls=":", alpha=0.7)

        ax.set_xscale("log")
        ax.set_xlim(RATIO_MIN, RATIO_MAX)
        # Clean x axis: ticks exactly at the paper's five measured ratios,
        # plain decimal labels, no crowded minor-tick labels.
        ax.set_xticks(list(FEED_RATIO))
        ax.set_xticklabels([f"{v:g}" for v in FEED_RATIO])
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_ylim(0, 105)
        ax.set_xlabel("MAA/gelatin feed ratio (mL/g)", color=FG)
        ax.set_ylabel("Degree of substitution (%)", color=FG)
        ax.set_title("DS tuning curve", color=FG, pad=10)
        ax.tick_params(colors=MUTED, which="both")
        for spine in ax.spines.values():
            spine.set_color(MUTED)
        ax.grid(True, which="major", color=FG, alpha=0.12)
        leg = ax.legend(facecolor=PANEL, edgecolor=MUTED, fontsize=9,
                        loc="lower right")
        for t in leg.get_texts():
            t.set_color(FG)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    # -------------------------------------------------------------- helpers
    def _write(self, tagged_lines):
        """tagged_lines: list of (text, tag_or_None)."""
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        for i, (text, tag) in enumerate(tagged_lines):
            self.out.insert("end", text + ("\n" if i < len(tagged_lines) - 1 else ""),
                            (tag,) if tag else ())
        self.out.configure(state="disabled")

    def _parse(self, var, name, lo=None, hi=None):
        try:
            v = float(var.get())
        except ValueError:
            messagebox.showerror("Invalid input", f"{name} must be a number.")
            return None
        if lo is not None and hi is not None and not (lo <= v <= hi):
            messagebox.showerror("Out of range", f"{name} must be between {lo} and {hi}.")
            return None
        return v

    def _parse_or_x(self, var, name, lo, hi, fallback):
        """Like _parse, but 'X'/'x'/empty means 'unknown' -> use fallback.
        Returns (value, was_unknown); (None, None) on invalid input."""
        raw = var.get().strip()
        if raw == "" or raw.lower() == "x":
            return fallback, True
        return self._parse(var, name, lo, hi), False

    # --------------------------------------------------------------- events
    def on_reverse(self):
        ds = self._parse(self.ds_var, "Target DS", 0.0, 100.0)
        gel = self._parse(self.gel_var, "Gelatin mass", 0.1, 1000.0)
        if ds is None or gel is None:
            return

        ratio, ds_hat, ds_std = reverse_ds(ds)
        self._plot_curve(marker_ratio=ratio, marker_ds=ds_hat,
                         marker_label=f"Target {ds:g}% → {ratio:.4f} mL/g")
        lines = [(f"TARGET DS: {ds:.1f} %", "head"), ("", None)]
        lines += [(t, "head" if b else None) for t, b in _fmt_conditions(ratio, gel, ds_hat, ds_std)]

        if ds < DS_MIN or ds > DS_MAX:
            lines += [("", None),
                      (f"⚠ WARNING: {ds:g}% is outside the paper's observed range "
                       f"({DS_MIN:.0f}–{DS_MAX:.0f}%).", "warn"),
                      ("  The result above is an extrapolation — validate "
                       "experimentally.", "warn")]
        elif ds_std > 6:
            lines += [("", None),
                      ("⚠ NOTE: model uncertainty is elevated here (>6%); this DS "
                       "region is sparsely sampled in the paper.", "warn")]

        lines += [("", None),
                  ("Feed ratio is the ONLY freely tuned variable. Deviating from the "
                   "fixed", "muted"),
                  ("conditions shifts DS per the paper's one-factor sensitivity table "
                   "(see", "muted"),
                  ("SENSITIVITY in gelma_ds_reverse_model.py).", "muted")]
        self._write(lines)
        self._log_history("Reverse", f"DS {ds:g}% / {gel:g} g",
                          f"{ratio:.4f}", f"{ratio * gel:.2f}",
                          f"{ds_hat:.1f} ± {ds_std:.1f}")

    def on_forward(self):
        ratio = self._parse(self.ratio_var, "Feed ratio", 1e-4, 10.0)
        if ratio is None:
            return
        ds, std = predict_ds(ratio)
        self._plot_curve(marker_ratio=ratio, marker_ds=ds,
                         marker_label=f"{ratio:g} mL/g → {ds:.1f}%")
        lines = [(f"FORWARD: MAA/gelatin = {ratio:.4f} mL/g", "head"), ("", None),
                 (f"  Predicted DS = {ds:.1f} ± {std:.1f} %", None)]
        if not (RATIO_MIN <= ratio <= RATIO_MAX):
            lines += [("", None),
                      (f"⚠ WARNING: outside the paper's tested range "
                       f"({RATIO_MIN}–{RATIO_MAX} mL/g). Extrapolation.", "warn")]
        self._write(lines)
        self._log_history("Forward", f"{ratio:g} mL/g",
                          f"{ratio:.4f}", "—", f"{ds:.1f} ± {std:.1f}")

    def on_clear(self):
        self._write([("Enter a target DS above and press "
                      "'Calculate conditions'.", "muted")])

    # --------------------------------------- full-conditions (Tier 1 + 2)
    def _full_prediction(self, ratio, cb, ph, wv, temp, time, skip=()):
        """GP base (Tier 1) + one-factor sensitivity shifts (Tier 2).

        Shifts are linearly interpolated between the paper's tabulated
        points and clamped at the table edges. Parameters named in `skip`
        (unknown inputs) contribute no shift. Returns
        (base_ds, final_ds, gp_std, shifts, clamped)."""
        base, std = predict_ds(ratio)
        shifts, clamped = {}, []
        for param, val in (("CB_buffer_M", cb), ("initial_pH", ph),
                           ("gelatin_wv_pct", wv), ("temperature_C", temp),
                           ("reaction_time_min", time)):
            if param in skip:
                shifts[param] = 0.0
                continue
            table = SENSITIVITY[param]
            xs = sorted(table)
            ys = [table[x] for x in xs]
            if val < xs[0] or val > xs[-1]:
                clamped.append((param, xs[0], xs[-1]))
            shifts[param] = float(np.interp(val, xs, ys))
        ds = base + sum(shifts.values())
        return base, ds, std, shifts, clamped

    def on_full(self):
        ratio_raw = self.f_vars["ratio"].get().strip()
        if ratio_raw == "" or ratio_raw.lower() == "x":
            messagebox.showerror("Required input",
                                 "Feed ratio is required — it drives the GP "
                                 "model and has no default to assume.")
            return
        ratio = self._parse(self.f_vars["ratio"], "Feed ratio", 1e-4, 10.0)
        if ratio is None:
            return

        cb, u_cb = self._parse_or_x(self.f_vars["cb"], "CB buffer",
                                    0.01, 5.0, OPTIMAL["CB_buffer_M"])
        ph, u_ph = self._parse_or_x(self.f_vars["ph"], "pH",
                                    5.0, 13.0, OPTIMAL["initial_pH"])
        wv, u_wv = self._parse_or_x(self.f_vars["wv"], "Gelatin w/v %",
                                    0.5, 40.0, OPTIMAL["gelatin_wv_pct"])
        temp, u_temp = self._parse_or_x(self.f_vars["temp"], "Temperature",
                                        4.0, 80.0, OPTIMAL["temperature_C"])
        time, u_time = self._parse_or_x(self.f_vars["time"], "Reaction time",
                                        1.0, 1440.0, OPTIMAL["reaction_time_min"])
        if None in (cb, ph, wv, temp, time):
            return

        unknown = {p for p, u in (("CB_buffer_M", u_cb), ("initial_pH", u_ph),
                                  ("gelatin_wv_pct", u_wv),
                                  ("temperature_C", u_temp),
                                  ("reaction_time_min", u_time)) if u}
        base, ds, std, shifts, clamped = self._full_prediction(
            ratio, cb, ph, wv, temp, time, skip=unknown)
        ds_clamped = min(max(ds, 0.0), 100.0)

        names = {"CB_buffer_M": f"CB buffer {cb:g} M",
                 "initial_pH": f"pH {ph:g}",
                 "gelatin_wv_pct": f"gelatin {wv:g} w/v %",
                 "temperature_C": f"{temp:g} °C",
                 "reaction_time_min": f"{time:g} min"}
        plain = {"CB_buffer_M": "CB buffer", "initial_pH": "pH",
                 "gelatin_wv_pct": "gelatin w/v %", "temperature_C": "temperature",
                 "reaction_time_min": "reaction time"}
        vals = {"CB_buffer_M": cb, "initial_pH": ph, "gelatin_wv_pct": wv,
                "temperature_C": temp, "reaction_time_min": time}
        n_dev = sum(1 for p, v in vals.items()
                    if p not in unknown and abs(v - OPTIMAL[p]) > 1e-9)

        lines = [("FULL-CONDITIONS PREDICTION (Tier 1 + Tier 2)", "head"),
                 ("", None),
                 (f"  Tier 1 — GP at {ratio:.4f} mL/g:        "
                  f"base DS = {base:.1f} ± {std:.1f} %", None)]
        for param, shift in shifts.items():
            if param in unknown:
                lines.append((f"  Tier 2 — {plain[param]:<28}unknown (X), "
                              f"optimum assumed, no shift", "muted"))
            elif abs(shift) > 1e-9:
                lines.append((f"  Tier 2 — {names[param]:<28}"
                              f"{shift:+.1f} pp", None))
        if n_dev == 0:
            lines.append(("  Tier 2 — all conditions at optima: no shift", None))
        lines += [("  " + "-" * 44, None),
                  (f"  Predicted DS = {ds_clamped:.1f} % "
                   f"(GP ±{std:.1f} %; Tier-2 shifts are estimates)", None)]
        if abs(ds - ds_clamped) > 1e-9:
            lines.append((f"  (raw value {ds:.1f} % clipped to the physical "
                          f"0–100 % range)", "muted"))

        if unknown:
            unk_names = ", ".join(plain[p] for p in unknown)
            lines += [("", None),
                      (f"⚠ {len(unknown)} condition(s) unknown (X): {unk_names}.",
                       "warn"),
                      ("  Assumed at the validated optimum — if the real value "
                       "differed,", "warn"),
                      ("  the true DS will shift per the sensitivity table.", "warn")]
        if n_dev > 0:
            lines += [("", None),
                      (f"⚠ NOTE: {n_dev} condition(s) deviate from the validated "
                       f"optimum.", "warn"),
                      ("  Tier-2 shifts are one-factor-at-a-time estimates; "
                       "combined", "warn"),
                      ("  effects are NOT validated and may not add linearly.", "warn")]
        for param, lo, hi in clamped:
            lines.append((f"⚠ {param} is outside the sensitivity table "
                          f"({lo}–{hi}); edge value used.", "warn"))
        if not (RATIO_MIN <= ratio <= RATIO_MAX):
            lines.append((f"⚠ Ratio outside the tested range "
                          f"({RATIO_MIN}–{RATIO_MAX} mL/g). Extrapolation.", "warn"))
        self._write(lines)

        self._plot_curve(marker_ratio=ratio, marker_ds=ds_clamped,
                         marker_label=f"Full conditions → {ds_clamped:.1f}%")
        parts = [f"r={ratio:g}",
                 "X" if u_cb else f"{cb:g}M",
                 "X" if u_ph else f"pH{ph:g}",
                 "X" if u_wv else f"{wv:g}%",
                 "X" if u_temp else f"{temp:g}°C",
                 "X" if u_time else f"{time:g}min"]
        cond = ", ".join(parts)
        self._log_history("Full", cond, f"{ratio:.4f}", "—",
                          f"{ds_clamped:.1f} ± {std:.1f}")


if __name__ == "__main__":
    GelMAApp().mainloop()
