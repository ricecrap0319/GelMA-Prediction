#!/usr/bin/env python3
"""
GelMA DS reverse model (Gaussian Process Regression)
====================================================
Data source: Shirahama H, Lee BH, Tan LP, Cho NJ (2016)
"Precise Tuning of Facile One-Pot Gelatin Methacryloyl (GelMA) Synthesis"
Scientific Reports 6:31036 - https://www.nature.com/articles/srep31036

Structure ("safe" design):
  Tier 1 - GP model: DS = f(MAA/gelatin feed ratio), with the other five
           parameters fixed at the paper's validated optima.
  Tier 2 - Sensitivity corrections from the paper's one-factor sweeps,
           applied only as flagged estimates (NOT validated combinations).

The paper measured one parameter at a time, so off-axis combinations of
conditions are extrapolations. This code tells you when that happens.

Usage:
  python3 gelma_ds_reverse_model.py --ds 80              # target DS = 80%
  python3 gelma_ds_reverse_model.py --ds 80 --gelatin 20 # scale to 20 g gelatin
  python3 gelma_ds_reverse_model.py --forward 0.05       # predict DS at 0.05 mL/g

Requires: numpy, scikit-learn (pip install numpy scikit-learn)
"""

import argparse
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

# ---------------------------------------------------------------------------
# Paper's validated optimal conditions (bold rows, Fig. 1C of the paper)
# ---------------------------------------------------------------------------
OPTIMAL = {
    "CB_buffer_M": 0.25,        # carbonate-bicarbonate buffer molarity
    "initial_pH": 9.0,          # adjust before MAA addition
    "gelatin_wv_pct": 10.0,     # w/v % (10-20% both fine)
    "temperature_C": 45.0,      # 35-50 C all equivalent (~96% DS)
    "reaction_time_min": 60.0,  # reaction complete within 1 h
}

# ---------------------------------------------------------------------------
# Training data: Series 3 (Fig. 4B) - the primary tuning curve.
# Fixed: 0.25 M CB, initial pH 9, 10 w/v% gelatin, 50 C, 3 h.
# Values digitized from the figure (approx. +/-2-3%); see Excel workbook.
# ---------------------------------------------------------------------------
FEED_RATIO = np.array([0.0125, 0.025, 0.05, 0.1, 0.2])   # mL MAA per g gelatin
DS_OBSERVED = np.array([22.0, 48.0, 73.5, 96.0, 98.5])   # degree of substitution (%)

# Valid interpolation domain (paper's tested range)
RATIO_MIN, RATIO_MAX = FEED_RATIO.min(), FEED_RATIO.max()
DS_MIN, DS_MAX = DS_OBSERVED.min(), DS_OBSERVED.max()

# ---------------------------------------------------------------------------
# Tier 2: sensitivity corrections (one-factor-at-a-time, from the paper).
# Each entry: (parameter, value) -> approximate DS shift in percentage points
# relative to the optimum. Use ONLY as rough adjustments, never as validated
# combinations.
# ---------------------------------------------------------------------------
SENSITIVITY = {
    "CB_buffer_M": {0.10: -2.0, 0.25: 0.0, 0.50: -3.5, 0.75: -11.0, 1.00: -26.0},
    "initial_pH": {8: -4.5, 9: 0.0, 10: -7.0, 11: -22.0},
    "gelatin_wv_pct": {1: -13.0, 2.5: -2.5, 5: -2.0, 10: 0.0, 20: +1.5},
    "temperature_C": {35: -0.5, 40: 0.0, 45: +0.5, 50: 0.0},   # ~no effect
    "reaction_time_min": {15: -3.0, 30: -1.0, 60: 0.0, 120: 0.0, 180: 0.0},
}


def build_gp():
    """GP on log10(feed ratio): the ratio spans 16x, so a log transform
    makes the RBF length-scale meaningful across the whole range."""
    X = np.log10(FEED_RATIO).reshape(-1, 1)
    y = DS_OBSERVED
    # alpha = 2.5 % digitization/measurement noise on each observation.
    # With normalize_y=True, alpha must be expressed in normalized units,
    # i.e. (sigma / std(y))^2.
    alpha_norm = (2.5 / y.std()) ** 2
    kernel = (ConstantKernel(50.0, (1e-1, 1e4))
              * RBF(length_scale=0.3, length_scale_bounds=(0.05, 5.0)))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha_norm, normalize_y=True,
                                  n_restarts_optimizer=10, random_state=42)
    gp.fit(X, y)
    return gp


GP = build_gp()


def predict_ds(feed_ratio):
    """Forward prediction: feed ratio (mL/g) -> (DS mean, DS std) in %."""
    x = np.log10(np.atleast_1d(float(feed_ratio))).reshape(-1, 1)
    mu, std = GP.predict(x, return_std=True)
    return float(mu[0]), float(std[0])


def reverse_ds(target_ds):
    """Reverse prediction: target DS (%) -> feed ratio with the highest
    posterior probability of hitting that DS.

    Objective: minimize |mu(r) - target| / (std(r) + 1), i.e. prefer a ratio
    whose predicted DS is close to the target AND whose uncertainty is low
    (close to measured data). Solved on a dense grid + local refinement.
    """
    grid = np.linspace(np.log10(RATIO_MIN), np.log10(RATIO_MAX), 4000)
    mu, std = GP.predict(grid.reshape(-1, 1), return_std=True)
    score = np.abs(mu - target_ds) / (std + 1.0)
    best = grid[int(np.argmin(score))]

    # Local refinement around the grid winner
    fine = np.linspace(best - 0.01, best + 0.01, 400)
    mu2, std2 = GP.predict(fine.reshape(-1, 1), return_std=True)
    score2 = np.abs(mu2 - target_ds) / (std2 + 1.0)
    ratio = 10 ** float(fine[int(np.argmin(score2))])
    ds_hat, ds_std = predict_ds(ratio)
    return ratio, ds_hat, ds_std


def report(target_ds=None, gelatin_g=10.0, forward_ratio=None):
    line = "=" * 64
    print(line)
    print("GelMA one-pot synthesis - GP model (Shirahama et al. 2016)")
    print(line)

    if forward_ratio is not None:
        ds, std = predict_ds(forward_ratio)
        print(f"FORWARD: MAA/gelatin = {forward_ratio:.4f} mL/g")
        print(f"  predicted DS = {ds:.1f} +/- {std:.1f} %")
        if not (RATIO_MIN <= forward_ratio <= RATIO_MAX):
            print("  WARNING: outside the paper's tested range "
                  f"({RATIO_MIN}-{RATIO_MAX} mL/g). Extrapolation.")
        return

    ratio, ds_hat, ds_std = reverse_ds(target_ds)

    print(f"TARGET DS: {target_ds:.1f} %")
    print()
    print("RECOMMENDED CONDITIONS (single most likely set)")
    print(f"  MAA/gelatin feed ratio : {ratio:.4f} mL/g")
    print(f"  -> MAA volume          : {ratio * gelatin_g:.2f} mL "
          f"for {gelatin_g:g} g gelatin")
    print(f"  Gelatin concentration  : {OPTIMAL['gelatin_wv_pct']:g} w/v % "
          f"({gelatin_g:g} g in {gelatin_g / OPTIMAL['gelatin_wv_pct'] * 100:.0f} mL "
          f"of 0.25 M CB buffer)")
    print(f"  CB buffer              : {OPTIMAL['CB_buffer_M']} M, "
          f"adjust to pH {OPTIMAL['initial_pH']:g} before adding MAA")
    print(f"  Temperature            : {OPTIMAL['temperature_C']:g} deg C "
          f"(35-50 deg C all equivalent)")
    print(f"  Reaction time          : {OPTIMAL['reaction_time_min']:g} min "
          f"(quench by dilution/cooling, then dialyze)")
    print()
    print(f"  Model prediction at this ratio: DS = {ds_hat:.1f} +/- {ds_std:.1f} %")

    # ---- honest warnings ----
    if target_ds < DS_MIN or target_ds > DS_MAX:
        print()
        print(f"  WARNING: target DS {target_ds}% is outside the paper's observed "
              f"range ({DS_MIN:.0f}-{DS_MAX:.0f}%). The answer above is an "
              f"extrapolation; validate experimentally.")
    elif ds_std > 6:
        print()
        print("  NOTE: model uncertainty is elevated here (>6%); this DS region "
              "is sparsely sampled in the paper.")

    # Molar-excess context (0.286 mmol -NH2 per g gelatin; MAA d=1.035 g/mL, MW=154.16)
    mmol_maa = ratio * 1.035 / 154.16 * 1000  # per g gelatin
    print()
    print(f"  Context: this ratio is a ~{mmol_maa / 0.286:.2f}x molar excess of "
          f"MAA over gelatin free amino groups.")
    print()
    print("  Reminder: feed ratio is the ONLY freely tuned variable in this")
    print("  model. Deviating from the fixed conditions above shifts DS per")
    print("  the sensitivity table (one-factor estimates, not validated")
    print("  combinations) - see SENSITIVITY in this file.")
    print(line)


def main():
    p = argparse.ArgumentParser(description="GelMA DS reverse GP model")
    p.add_argument("--ds", type=float, help="target degree of substitution (%%)")
    p.add_argument("--gelatin", type=float, default=10.0,
                   help="gelatin mass in grams (default 10)")
    p.add_argument("--forward", type=float, default=None,
                   help="instead: predict DS for a given MAA/gelatin ratio (mL/g)")
    a = p.parse_args()
    if a.forward is not None:
        report(forward_ratio=a.forward, gelatin_g=a.gelatin)
    elif a.ds is not None:
        report(target_ds=a.ds, gelatin_g=a.gelatin)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
