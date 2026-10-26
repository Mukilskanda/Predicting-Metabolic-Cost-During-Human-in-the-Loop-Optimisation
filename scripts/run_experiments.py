"""Run every experiment in the study and write tables + raw results to results/.

    python scripts/run_experiments.py            # full run (~10-15 min on 8+ cores)
    python scripts/run_experiments.py --quick    # fewer repeats, for a smoke test

Experiments
  E1  Reproduction of the paper's Table I (Constant / LASSO / Bayesian-reg. NN x 4 feature sets)
  E2  Hidden-neuron sweep for the Bayesian-regularised NN (paper Fig. 1)
  E3  Model zoo (13 models) under two protocols: random 5-fold (paper) and leave-one-day-out
  E4  Cross-subject generalisation (train on one person, test on the other)
  E5  Feature importance: grouped + per-feature permutation importance, LASSO stability selection
  E6  Dimensionality reduction: PCA vs PLS vs forward stepwise selection
  E7  Learning curve (MSE vs number of training samples)
  E8  Uncertainty calibration of the Gaussian Process (95 % interval coverage)
  E9  HILO usefulness: within-day rank correlation of predicted vs measured cost
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingRegressor
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from metacost.data import (ALL_FEATURES, EMG_FEATURES, FEATURE_SETS, GROUP_OF, NOISE_FLOOR_MSE,
                           load_all, to_frame)
from metacost.evaluation import cross_validate, metrics, summarise
from metacost.models import MODEL_NAMES, make_model

RES = ROOT / "results"
(RES / "tables").mkdir(parents=True, exist_ok=True)
(RES / "models").mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ensemble():
    """Simple average of four diverse, individually strong learners."""
    return VotingRegressor([("gp", make_model("Gaussian Process")), ("svr", make_model("SVR (RBF)")),
                            ("lasso", make_model("LASSO")), ("et", make_model("Extra Trees"))])


def get_model(name, n_hidden=2):
    return ensemble() if name == "Ensemble (avg)" else make_model(name, n_hidden)


def _cv_job(key, model_name, X, y, groups, protocol, n_repeats, n_hidden=2):
    fold_mse, oof = cross_validate(get_model(model_name, n_hidden), X, y, groups, protocol, n_repeats)
    return key, fold_mse, oof


def main(quick=False):
    t0 = time.time()
    R = 2 if quick else 5
    data = load_all()
    out = {"noise_floor_mse": NOISE_FLOOR_MSE, "subjects": {}}
    pd.concat([to_frame(d) for d in data.values()]).to_csv(ROOT / "data" / "hilo_metabolic_dataset.csv",
                                                          index=False)
    for s, d in data.items():
        out["subjects"][s] = {"n": int(len(d.y)), "target_var": float(d.y.var()),
                              "target_mean": float(d.y.mean()), "days": int(d.day.max() + 1)}

    # ---------------- E2: neuron sweep (needed first: E1 uses the best n) -------------
    log("E2 neuron sweep")
    neurons = list(range(1, 7))
    jobs = [delayed(_cv_job)((s, fs, n), "Bayesian-reg. NN", d.features(cols), d.y, None, "random",
                             2 if quick else 3, n)
            for s, d in data.items() for fs, cols in FEATURE_SETS.items() for n in neurons]
    sweep = {}
    for (s, fs, n), fm, _ in Parallel(n_jobs=-1)(jobs):
        sweep.setdefault(s, {}).setdefault(fs, {})[n] = float(fm.mean())
    out["E2_neuron_sweep"] = sweep
    best_n = {s: {fs: min(v, key=v.get) for fs, v in sweep[s].items()} for s in sweep}

    # ---------------- E1: reproduction of Table I -------------------------------------
    log("E1 reproduction table")
    jobs = []
    for s, d in data.items():
        for fs, cols in FEATURE_SETS.items():
            for m in ["Constant (mean)", "LASSO", "Bayesian-reg. NN", "Gaussian Process"]:
                jobs.append(delayed(_cv_job)((s, fs, m), m, d.features(cols), d.y, None, "random", R,
                                             best_n[s][fs]))
    e1 = []
    for (s, fs, m), fm, oof in Parallel(n_jobs=-1)(jobs):
        e1.append({"subject": s, "feature_set": fs, "model": m, **summarise(fm, data[s].y, oof)})
    e1 = pd.DataFrame(e1)
    e1.to_csv(RES / "tables" / "E1_reproduction.csv", index=False)
    out["E1_best_neurons"] = best_n

    # ---------------- E3: model zoo, two protocols ------------------------------------
    log("E3 model zoo")
    zoo = MODEL_NAMES + ["Ensemble (avg)"]
    jobs = []
    for s, d in data.items():
        for fs in ["All features", "Step + EMG"]:
            X = d.features(FEATURE_SETS[fs])
            for m in zoo:
                for proto in ["random", "day"]:
                    jobs.append(delayed(_cv_job)((s, fs, m, proto), m, X, d.y, d.day, proto, R,
                                                 best_n[s][fs]))
    e3, oofs = [], {}
    for (s, fs, m, proto), fm, oof in Parallel(n_jobs=-1)(jobs):
        d = data[s]
        row = {"subject": s, "feature_set": fs, "model": m, "protocol": proto,
               **summarise(fm, d.y, oof)}
        if proto == "day":   # E9: does the surrogate rank conditions correctly within a day?
            rhos = [spearmanr(d.y[d.day == k], oof[d.day == k])[0] for k in np.unique(d.day)]
            row["within_day_spearman"] = float(np.nanmean(rhos))
        e3.append(row)
        oofs[f"{s}|{fs}|{m}|{proto}"] = oof
    e3 = pd.DataFrame(e3)
    e3.to_csv(RES / "tables" / "E3_model_zoo.csv", index=False)
    joblib.dump(oofs, RES / "oof_predictions.joblib")

    # ---------------- E4: cross-subject ------------------------------------------------
    log("E4 cross-subject")
    e4 = []
    for tr, te in [("S1", "S2"), ("S2", "S1")]:
        for fs in ["All features", "Step + EMG", "EMG only"]:
            cols = FEATURE_SETS[fs]
            for norm in ["global", "per-subject z-score"]:
                Xtr, Xte = data[tr].features(cols), data[te].features(cols)
                ytr, yte = data[tr].y, data[te].y
                if norm != "global":   # calibrate each person to their own feature distribution
                    Xtr = StandardScaler().fit_transform(Xtr)
                    Xte = StandardScaler().fit_transform(Xte)
                for m in ["LASSO", "Gaussian Process", "Ensemble (avg)"]:
                    p = get_model(m).fit(Xtr, ytr).predict(Xte)
                    e4.append({"train": tr, "test": te, "feature_set": fs, "normalisation": norm,
                               "model": m, **metrics(yte, p),
                               "constant_mse": float(np.mean((yte - ytr.mean()) ** 2))})
    e4 = pd.DataFrame(e4)
    e4.to_csv(RES / "tables" / "E4_cross_subject.csv", index=False)

    # ---------------- E5: feature importance -------------------------------------------
    log("E5 feature importance")
    e5_perm, e5_group, e5_stab = [], [], []
    rng = np.random.RandomState(0)
    for s, d in data.items():
        X, y = d.features(ALL_FEATURES), d.y
        # permutation importance on held-out folds of the GP
        imp = np.zeros(len(ALL_FEATURES))
        group_imp = {g: [] for g in ["Step", "EMG", "Control"]}
        for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(X)):
            m = make_model("Gaussian Process").fit(X[tr], y[tr])
            pi = permutation_importance(m, X[te], y[te], scoring="neg_mean_squared_error",
                                        n_repeats=20, random_state=k)
            imp += pi.importances_mean / 5
            base = np.mean((m.predict(X[te]) - y[te]) ** 2)
            for g in group_imp:
                idx = [i for i, f in enumerate(ALL_FEATURES) if GROUP_OF[f] == g]
                deltas = []
                for _ in range(20):
                    Xp = X[te].copy()
                    Xp[:, idx] = Xp[rng.permutation(len(te))][:, idx]   # permute group jointly
                    deltas.append(np.mean((m.predict(Xp) - y[te]) ** 2) - base)
                group_imp[g].append(np.mean(deltas))
        for f, v in zip(ALL_FEATURES, imp):
            e5_perm.append({"subject": s, "feature": f, "group": GROUP_OF[f], "importance_mse": v})
        for g, v in group_imp.items():
            e5_group.append({"subject": s, "group": g, "importance_mse": float(np.mean(v))})
        # LASSO stability selection: how often is each feature selected on bootstrap samples?
        Xs = StandardScaler().fit_transform(X)
        alpha = make_model("LASSO").fit(X, y)[-1].alpha_
        counts = np.zeros(len(ALL_FEATURES))
        for b in range(200):
            idx = rng.choice(len(y), len(y), replace=True)
            counts += Lasso(alpha=alpha, max_iter=50000).fit(Xs[idx], y[idx]).coef_ != 0
        for f, c in zip(ALL_FEATURES, counts / 200):
            e5_stab.append({"subject": s, "feature": f, "group": GROUP_OF[f], "selection_freq": c})
    pd.DataFrame(e5_perm).to_csv(RES / "tables" / "E5_permutation_importance.csv", index=False)
    pd.DataFrame(e5_group).to_csv(RES / "tables" / "E5_group_importance.csv", index=False)
    pd.DataFrame(e5_stab).to_csv(RES / "tables" / "E5_lasso_stability.csv", index=False)

    # ---------------- E6: dimensionality reduction -------------------------------------
    log("E6 dimensionality reduction")

    def _dimred_job(s, method, k):
        d = data[s]
        X = d.features(ALL_FEATURES)
        if method == "PCA + Ridge":
            mdl = make_pipeline(StandardScaler(), PCA(k), RidgeCV(np.logspace(-3, 3, 30)))
        elif method == "PLS":
            mdl = make_pipeline(StandardScaler(), PLSRegression(k, scale=False))
        else:   # forward stepwise selection (selection done inside each training fold)
            mdl = make_pipeline(StandardScaler(),
                                SequentialFeatureSelector(RidgeCV(np.logspace(-3, 3, 13)), n_features_to_select=k,
                                                          direction="forward", cv=5,
                                                          scoring="neg_mean_squared_error"),
                                RidgeCV(np.logspace(-3, 3, 30)))
        fm, _ = cross_validate(mdl, X, d.y, None, "random", 2 if quick else 3)
        return s, method, k, float(fm.mean())

    ks = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25]
    jobs = [delayed(_dimred_job)(s, m, k) for s in data for m in ["PCA + Ridge", "PLS", "Forward selection"]
            for k in ks]
    e6 = pd.DataFrame(Parallel(n_jobs=-1)(jobs), columns=["subject", "method", "k", "MSE"])
    pca_var = {s: np.cumsum(PCA().fit(StandardScaler().fit_transform(d.features(ALL_FEATURES)))
                            .explained_variance_ratio_).tolist() for s, d in data.items()}
    out["E6_pca_cumvar"] = pca_var
    e6.to_csv(RES / "tables" / "E6_dimensionality_reduction.csv", index=False)

    # ---------------- E7: learning curve -----------------------------------------------
    log("E7 learning curve")

    def _lc_job(s, m, frac, rep):
        d = data[s]
        X, y = d.features(ALL_FEATURES), d.y
        r = np.random.RandomState(rep)
        te = r.choice(len(y), len(y) // 5, replace=False)
        pool = np.setdiff1d(np.arange(len(y)), te)
        tr = r.choice(pool, int(frac * len(pool)), replace=False)
        p = get_model(m).fit(X[tr], y[tr]).predict(X[te])
        return s, m, len(tr), float(np.mean((p - y[te]) ** 2))

    jobs = [delayed(_lc_job)(s, m, f, r) for s in data for m in ["LASSO", "Gaussian Process", "Ensemble (avg)"]
            for f in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] for r in range(6 if quick else 15)]
    e7 = pd.DataFrame(Parallel(n_jobs=-1)(jobs), columns=["subject", "model", "n_train", "MSE"])
    e7 = e7.groupby(["subject", "model", "n_train"]).MSE.agg(["mean", "std"]).reset_index()
    e7.to_csv(RES / "tables" / "E7_learning_curve.csv", index=False)

    # ---------------- E8: GP uncertainty calibration (leave-one-day-out) ---------------
    log("E8 uncertainty calibration")
    e8 = []
    for s, d in data.items():
        X, y = d.features(ALL_FEATURES), d.y
        mu, sd = np.zeros(len(y)), np.zeros(len(y))
        for k in np.unique(d.day):
            tr, te = d.day != k, d.day == k
            m = make_model("Gaussian Process").fit(X[tr], y[tr])
            mu[te], sd[te] = m.predict(X[te], return_std=True)
        z = np.abs(y - mu) / sd
        for level, zc in [(0.5, 0.674), (0.8, 1.282), (0.9, 1.645), (0.95, 1.96)]:
            e8.append({"subject": s, "nominal": level, "empirical": float(np.mean(z <= zc))})
        out.setdefault("E8_gp_lodo", {})[s] = {"mu": mu.tolist(), "sd": sd.tolist()}
    pd.DataFrame(e8).to_csv(RES / "tables" / "E8_calibration.csv", index=False)

    # ---------------- Final models for the demo app ------------------------------------
    log("Training final models for the app")
    for s, d in data.items():
        for fs in ["All features", "Step + EMG", "EMG only"]:
            cols = FEATURE_SETS[fs]
            for m in ["LASSO", "Gaussian Process", "Bayesian-reg. NN", "Ensemble (avg)"]:
                mdl = get_model(m, best_n[s].get(fs, 2)).fit(d.features(cols), d.y)
                joblib.dump({"model": mdl, "features": cols, "subject": s, "name": m, "feature_set": fs},
                            RES / "models" / f"{s}__{fs.replace(' ', '').replace('+', '_')}__{m.split(' ')[0].rstrip('.')}.joblib")

    out["runtime_min"] = (time.time() - t0) / 60
    (RES / "results.json").write_text(json.dumps(out, indent=1))
    log(f"done in {out['runtime_min']:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
