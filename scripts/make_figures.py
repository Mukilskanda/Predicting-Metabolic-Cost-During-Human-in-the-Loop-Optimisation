"""Generate every figure in results/figures from the tables written by run_experiments.py.

    python scripts/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from metacost.data import ALL_FEATURES, FEATURE_SETS, NOISE_FLOOR_MSE, load_all

RES, FIG = ROOT / "results", ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
T = lambda n: pd.read_csv(RES / "tables" / n)

# ---- palette (validated categorical order) & style ------------------------------------
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
SUBJ = {"S1": BLUE, "S2": ORANGE}
GROUPC = {"EMG": BLUE, "Step": AQUA, "Control": YELLOW}

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", name)


def noise_line(ax, label=True):
    ax.axhline(NOISE_FLOOR_MSE, color=MUTED, ls=":", lw=1.5)
    if label:
        ax.annotate("measurement-noise floor", (1, NOISE_FLOOR_MSE), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=8, color=INK2)


data = load_all()
res = json.loads((RES / "results.json").read_text())
oofs = joblib.load(RES / "oof_predictions.joblib")

# ---- F1: dataset overview -------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
for ax, (s, d) in zip(axes, data.items()):
    t = d.day * 36 + d.condition
    ax.plot(t, d.y, color=SUBJ[s], lw=1.4, marker="o", ms=3)
    for k in range(1, 5):
        ax.axvline(k * 36 - 0.5, color=MUTED, lw=1, ls="--")
    for k in range(5):
        m = d.y[d.day == k].mean()
        ax.hlines(m, k * 36, k * 36 + 35, color=INK, lw=1.2)
        ax.text(k * 36 + 1, ax.get_ylim()[1] if False else d.y.max() * 1.02, f"Day {k+1}", fontsize=8, color=INK2)
    ax.set_ylabel("Normalised\nmetabolic cost")
    ax.set_title(f"Subject {s[-1]}  (n = {len(d.y)}, variance = {d.y.var():.4f})")
axes[-1].set_xlabel("Control condition (36 per day; black bar = day mean)")
save(fig, "F1_dataset_overview")

# ---- F2: feature-target correlation --------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 6.2), sharey=True)
for ax, (s, d) in zip(axes, data.items()):
    r = np.array([np.corrcoef(d.X[f], d.y)[0, 1] for f in ALL_FEATURES])
    cols = [GROUPC["EMG"] if f.startswith("EMG") else GROUPC["Control"] if f.startswith("ctrl") else GROUPC["Step"]
            for f in ALL_FEATURES]
    ax.barh(range(len(r)), r, color=cols, height=0.75)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(r)), ALL_FEATURES, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Pearson r with metabolic cost")
    ax.set_title(f"Subject {s[-1]}")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUPC.values()]
axes[1].legend(handles, ["EMG", "Step / gait", "Exo control"], loc="lower right")
save(fig, "F2_feature_correlation")

# ---- F3: neuron sweep (paper Fig. 1) --------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, fs in zip(axes.ravel(), FEATURE_SETS):
    for s in ["S1", "S2"]:
        sw = res["E2_neuron_sweep"][s][fs]
        n = sorted(sw, key=int)
        ax.plot([int(k) for k in n], [sw[k] for k in n], marker="o", color=SUBJ[s], label=f"Subject {s[-1]}")
    noise_line(ax)
    ax.set_ylim(0, None)
    ax.set_title(fs)
    ax.set_xlabel("Hidden neurons")
    ax.set_ylabel("CV MSE")
axes[0, 0].legend()
save(fig, "F3_neuron_sweep")

# ---- F4: model zoo (random vs leave-one-day-out) --------------------------------------
e3 = T("E3_model_zoo.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=False)
for ax, s in zip(axes, ["S1", "S2"]):
    sub = e3[(e3.subject == s) & (e3.feature_set == "All features")]
    piv = sub.pivot(index="model", columns="protocol", values="MSE").sort_values("random", ascending=False)
    piv = piv[piv.index != "Constant (mean)"]
    yy = np.arange(len(piv))
    ax.barh(yy + 0.2, piv["random"], height=0.38, color=SUBJ[s], label="Repeated nested 5-fold CV")
    ax.barh(yy - 0.2, piv["day"], height=0.38, color=SUBJ[s], alpha=0.45, label="Leave-one-day-out")
    const = sub[(sub.model == "Constant (mean)") & (sub.protocol == "random")].MSE.item()
    ax.axvline(const, color=RED, ls="--", lw=1.4)
    ax.text(const, len(piv) - 0.3, " predict-the-mean", color=INK2, fontsize=8)
    ax.axvline(NOISE_FLOOR_MSE, color=MUTED, ls=":", lw=1.5)
    ax.set_yticks(yy, piv.index)
    ax.set_xlabel("Cross-validated MSE (lower is better)")
    ax.set_title(f"Subject {s[-1]} - all 29 features")
    best = piv["random"].idxmin()
    ax.get_yticklabels()[list(piv.index).index(best)].set_fontweight("bold")
axes[0].legend(loc="center right", fontsize=8)
save(fig, "F4_model_zoo")

# ---- F5: prediction timeline for an unseen day (LODO) --------------------------------
best_model = {}
for s in ["S1", "S2"]:
    sub = e3[(e3.subject == s) & (e3.feature_set == "All features") & (e3.protocol == "day")]
    best_model[s] = sub.sort_values("MSE").iloc[0].model
best_random = {s: e3[(e3.subject == s) & (e3.feature_set == "All features") & (e3.protocol == "random")]
               .sort_values("MSE").iloc[0].model for s in ["S1", "S2"]}
fig, axes = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True)
gp = res["E8_gp_lodo"]
for ax, (s, d) in zip(axes, data.items()):
    t = d.day * 36 + d.condition
    mu, sd = np.array(gp[s]["mu"]), np.array(gp[s]["sd"])
    ax.fill_between(t, mu - 1.96 * sd, mu + 1.96 * sd, color=SUBJ[s], alpha=0.15, lw=0,
                    label="GP 95 % interval")
    ax.plot(t, d.y, color=INK, lw=1.3, label="Measured (indirect calorimetry)")
    oof = mu
    ax.plot(t, oof, color=SUBJ[s], lw=1.6, ls="--", label="Predicted - Gaussian Process")
    for k in range(1, 5):
        ax.axvline(k * 36 - 0.5, color=MUTED, lw=1, ls="--")
    r = np.corrcoef(d.y, oof)[0, 1]
    ax.set_title(f"Subject {s[-1]}: every day predicted by a model that never saw that day   (r = {r:.2f})")
    ax.set_ylabel("Metabolic cost")
axes[0].legend(loc="upper right", fontsize=8, ncol=3)
axes[-1].set_xlabel("Control condition (Day 1 ... Day 5)")
save(fig, "F5_unseen_day_timeline")

# ---- F6: scatter predicted vs measured -----------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
for ax, (s, d) in zip(axes, data.items()):
    oof = oofs[f"{s}|All features|{best_random[s]}|random"]
    ax.scatter(d.y, oof, s=22, color=SUBJ[s], edgecolor=SURF, lw=0.8)
    lo, hi = min(d.y.min(), oof.min()) - .05, max(d.y.max(), oof.max()) + .05
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls="--")
    r2 = 1 - np.sum((d.y - oof) ** 2) / np.sum((d.y - d.y.mean()) ** 2)
    ax.set_title(f"Subject {s[-1]} - {best_random[s]}  (R² = {r2:.2f})")
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted (out-of-fold)")
    ax.set_aspect("equal")
save(fig, "F6_pred_vs_actual")

# ---- F7: feature-set ablation (paper Table I) ----------------------------------------
e1 = T("E1_reproduction.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
models = ["LASSO", "Bayesian-reg. NN", "Gaussian Process"]
mcol = {"LASSO": BLUE, "Bayesian-reg. NN": ORANGE, "Gaussian Process": AQUA}
for ax, s in zip(axes, ["S1", "S2"]):
    x = np.arange(4)
    for i, m in enumerate(models):
        v = [e1[(e1.subject == s) & (e1.feature_set == fs) & (e1.model == m)].MSE.item() for fs in FEATURE_SETS]
        ax.bar(x + (i - 1) * 0.26, v, width=0.24, color=mcol[m], label=m)
    const = e1[(e1.subject == s) & (e1.model == "Constant (mean)")].MSE.mean()
    ax.axhline(const, color=RED, ls="--", lw=1.4)
    ax.text(3.45, const, "mean", va="bottom", ha="right", fontsize=8, color=INK2)
    noise_line(ax, label=False)
    ax.set_xticks(x, list(FEATURE_SETS))
    ax.set_title(f"Subject {s[-1]}")
    ax.set_ylabel("CV MSE")
axes[0].legend(fontsize=8)
save(fig, "F7_feature_set_ablation")

# ---- F8: importance ------------------------------------------------------------------
perm, grp, stab = T("E5_permutation_importance.csv"), T("E5_group_importance.csv"), T("E5_lasso_stability.csv")
fig = plt.figure(figsize=(11, 5.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.8])
for i, s in enumerate(["S1", "S2"]):
    ax = fig.add_subplot(gs[i])
    if i == 0:
        ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUPC.values()],
                  ["EMG", "Step / gait", "Exo control"], loc="lower right", fontsize=8)
    p = perm[perm.subject == s].sort_values("importance_mse", ascending=False).head(12)
    ax.barh(range(len(p)), p.importance_mse, color=[GROUPC[g] for g in p.group], height=0.7)
    ax.set_yticks(range(len(p)), p.feature, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("MSE increase when shuffled")
    ax.set_title(f"Subject {s[-1]}: top-12 features")
ax = fig.add_subplot(gs[2])
x = np.arange(3)
for j, s in enumerate(["S1", "S2"]):
    v = [grp[(grp.subject == s) & (grp.group == g)].importance_mse.item() for g in ["EMG", "Step", "Control"]]
    ax.bar(x + (j - .5) * .38, v, width=.36, color=SUBJ[s], label=f"Subject {s[-1]}")
ax.set_xticks(x, ["EMG", "Step", "Control"])
ax.set_title("Grouped importance")
ax.set_ylabel("MSE increase")
ax.legend(fontsize=8)
save(fig, "F8_feature_importance")

# ---- F9: dimensionality reduction ----------------------------------------------------
e6 = T("E6_dimensionality_reduction.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
mc = {"PCA + Ridge": YELLOW, "PLS": BLUE, "Forward selection": AQUA}
for ax, s in zip(axes, ["S1", "S2"]):
    for m, c in mc.items():
        sub = e6[(e6.subject == s) & (e6.method == m)]
        ax.plot(sub.k, sub.MSE, marker="o", color=c, label=m)
    noise_line(ax, label=False)
    ax.set_xlabel("Number of components / features")
    ax.set_ylabel("CV MSE")
    ax.set_title(f"Subject {s[-1]}")
axes[0].legend()
save(fig, "F9_dimensionality_reduction")

# ---- F10: learning curve -------------------------------------------------------------
e7 = T("E7_learning_curve.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
lc = {"LASSO": BLUE, "Gaussian Process": AQUA, "Ensemble (avg)": VIOLET}
for ax, s in zip(axes, ["S1", "S2"]):
    for m, c in lc.items():
        sub = e7[(e7.subject == s) & (e7.model == m)]
        ax.plot(sub.n_train, sub["mean"], marker="o", color=c, label=m)
        ax.fill_between(sub.n_train, sub["mean"] - sub["std"] / np.sqrt(15), sub["mean"] + sub["std"] / np.sqrt(15),
                        color=c, alpha=0.12, lw=0)
    noise_line(ax, label=False)
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Test MSE")
    ax.set_title(f"Subject {s[-1]}")
axes[0].legend()
save(fig, "F10_learning_curve")

# ---- F11: calibration ----------------------------------------------------------------
e8 = T("E8_calibration.csv")
fig, ax = plt.subplots(figsize=(4.6, 4.3))
ax.plot([0.4, 1], [0.4, 1], color=INK2, ls="--", lw=1, label="perfect calibration")
for s in ["S1", "S2"]:
    sub = e8[e8.subject == s]
    ax.plot(sub.nominal, sub.empirical, marker="o", color=SUBJ[s], label=f"Subject {s[-1]}")
ax.set_xlabel("Nominal interval coverage")
ax.set_ylabel("Empirical coverage (unseen days)")
ax.set_title("GP uncertainty calibration")
ax.legend(fontsize=8)
save(fig, "F11_calibration")

# ---- F12: cross-subject --------------------------------------------------------------
e4 = T("E4_cross_subject.csv")
within = {s: e3[(e3.subject == s) & (e3.feature_set == "All features") & (e3.protocol == "random")].MSE.min()
          for s in ["S1", "S2"]}
fig, ax = plt.subplots(figsize=(8, 4))
sub = e4[(e4.feature_set == "All features") & (e4.model == "Gaussian Process")]
labels, vals, cols = [], [], []
for te in ["S2", "S1"]:
    tr = "S1" if te == "S2" else "S2"
    labels += [f"Within {te}\n(best CV)", f"{tr}→{te}\nglobal", f"{tr}→{te}\nper-subject z", f"{te}\npredict mean"]
    vals += [within[te],
             sub[(sub.test == te) & (sub.normalisation == "global")].MSE.item(),
             sub[(sub.test == te) & (sub.normalisation != "global")].MSE.item(),
             sub[sub.test == te].constant_mse.iloc[0]]
    cols += [SUBJ[te], MUTED, AQUA, RED]
ax.bar(range(len(vals)), vals, color=cols, width=0.7)
for i, v in enumerate(vals):
    ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=INK2)
ax.set_xticks(range(len(vals)), labels, fontsize=8)
ax.set_ylabel("Test MSE")
ax.set_yscale("log")
ax.set_title("Does a model trained on one person work on another?")
save(fig, "F12_cross_subject")

# ---- F13: pipeline diagram -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 3.0))
ax.set_xlim(0, 12)
ax.set_ylim(0, 3)
ax.axis("off")
steps = [("Sensors", "EMG x16\nTreadmill force\nAnkle angle\nExo controls"),
         ("Pre-processing", "EMG band-pass,\nrectify, RMS\nper-step medians\nday normalisation"),
         ("29 features", "9 step\n16 EMG\n4 control\n(5 days x 36 cond.)"),
         ("Models", "LASSO / PLS / SVR\nGaussian Process\nBayesian-reg. NN\nEnsemble"),
         ("Validation", "Nested 5-fold x5\nLeave-one-day-out\nCross-subject"),
         ("Output", "Metabolic cost\n± uncertainty\nwithin-day ranking\nfor CMA-ES")]
cs = [MUTED, VIOLET, AQUA, BLUE, ORANGE, GREEN]
for i, ((h, b), c) in enumerate(zip(steps, cs)):
    x = 0.1 + i * 2
    ax.add_patch(FancyBboxPatch((x, 0.2), 1.75, 2.5, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=SURF, ec=c, lw=2))
    ax.add_patch(FancyBboxPatch((x, 2.2), 1.75, 0.5, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=c, ec=c, lw=2))
    ax.text(x + 0.875, 2.45, h, ha="center", va="center", color="white", fontweight="bold", fontsize=10)
    ax.text(x + 0.875, 1.2, b, ha="center", va="center", fontsize=8.5, color=INK, linespacing=1.5)
    if i < len(steps) - 1:
        ax.annotate("", (x + 2.0, 1.45), (x + 1.8, 1.45), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.5))
save(fig, "F13_pipeline")
print("best models (LODO):", best_model)

# ==== follow-up experiments (scripts/run_improvements.py) ===============================
if (RES / "tables" / "E10_day_calibration.csv").exists():
    e10 = T("E10_day_calibration.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, s in zip(axes, ["S1", "S2"]):
        for m, c in [("Gaussian Process", AQUA), ("Ridge", BLUE)]:
            for strat, ls in [("Spot-checks", "-"), ("Start of session", "--")]:
                sub = e10[(e10.subject == s) & (e10.model == m) & (e10.strategy == strat)]
                ax.plot(sub.k, sub.MSE, ls=ls, marker="o", color=c, label=f"{m} - {strat.lower()}")
        noise_line(ax, label=False)
        ax.set_ylim(0, None)
        ax.set_xlabel("Mask measurements on the new day (k)")
        ax.set_ylabel("Unseen-day MSE")
        ax.set_title(f"Subject {s[-1]}")
    axes[1].legend(fontsize=8)
    save(fig, "F14_day_calibration")

if (RES / "tables" / "E12_sensor_reduction.csv").exists():
    e12 = T("E12_sensor_reduction.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, s in zip(axes, ["S1", "S2"]):
        for m, c in [("Gaussian Process", AQUA), ("Ridge", BLUE)]:
            for proto, ls in [("random", "-"), ("day", "--")]:
                sub = e12[(e12.subject == s) & (e12.model == m) & (e12.protocol == proto)]
                ax.plot(sub.n_emg, sub.MSE, ls=ls, marker="o", color=c,
                        label=f"{m} - {'random CV' if proto == 'random' else 'unseen day'}")
        noise_line(ax, label=False)
        ax.set_ylim(0, None)
        ax.set_xticks([0, 2, 4, 6, 8, 12, 16])
        ax.set_xlabel("EMG electrodes kept (chosen inside each training fold)")
        ax.set_ylabel("CV MSE")
        ax.set_title(f"Subject {s[-1]}")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    axes[1].legend(fontsize=7.5, ncol=2, loc="upper left")
    save(fig, "F15_sensor_reduction")

if (RES / "tables" / "E11_cross_subject_calibration.csv").exists():
    e11 = T("E11_cross_subject_calibration.csv")
    e11 = e11[e11.model == "Gaussian Process"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sc = {"offset": MUTED, "offset+scale": VIOLET, "own model on k only": ORANGE, "mean of k": RED}
    for ax, te in zip(axes, ["S2", "S1"]):
        tr = "S1" if te == "S2" else "S2"
        for strat, c in sc.items():
            sub = e11[(e11.test == te) & (e11.strategy == strat)]
            lab = {"offset": f"{tr} model + offset", "offset+scale": f"{tr} model + offset & scale",
                   "own model on k only": f"{te}'s own model (k samples)", "mean of k": "mean of k samples"}[strat]
            ax.plot(sub.k, sub.MSE, marker="o", color=c, label=lab)
        ax.set_xlabel(f"Calibration samples from {te} (k)")
        ax.set_ylabel(f"MSE on the rest of {te}")
        ax.set_ylim(0, None)
        ax.set_title(f"Transfer {tr} → {te}")
        ax.legend(fontsize=8)
    save(fig, "F16_cross_subject_calibration")
