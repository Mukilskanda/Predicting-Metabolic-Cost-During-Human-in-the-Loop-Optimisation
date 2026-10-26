"""Follow-up experiments that address the weaknesses found by run_experiments.py.

    python scripts/run_improvements.py          (~5 min on 16 cores)

  E10  Day calibration: a few mask measurements on the new day, combined with an empirical-Bayes
       shrunken offset. Two strategies: (a) the first k conditions of the session (a calibration
       walk at the start), (b) k spot-checks spread over the session.
  E11  Cross-subject transfer with k calibration samples from the new person.
  E12  Sensor reduction: how many EMG electrodes are needed? (selection nested inside CV)
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from metacost.calibration import TopKSensors, estimate_offset_prior, shrunk_offset
from metacost.data import ALL_FEATURES, EMG_FEATURES, load_all
from metacost.evaluation import cross_validate
from metacost.models import make_model

TABLES = ROOT / "results" / "tables"
KS = [0, 1, 2, 3, 5, 8]
N_DRAWS = 40
MODELS = ["Ridge", "LASSO", "Gaussian Process"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- E10
def _calib_job(s, d, m, day):
    X, y = d.features(ALL_FEATURES), d.y
    tr, te = d.day != day, np.where(d.day == day)[0]
    model = make_model(m)
    sigma2, tau2 = estimate_offset_prior(model, X[tr], y[tr], d.day[tr])
    p = np.ravel(clone(model).fit(X[tr], y[tr]).predict(X[te]))
    yt = y[te]
    rng = np.random.RandomState(day)
    rows = []
    for k in KS:
        # (a) start-of-session calibration: first k conditions; scored on conditions 9..36 for all k
        off, w = shrunk_offset(yt[:k] - p[:k], sigma2, tau2)
        ev = slice(max(KS), None)
        rows.append(dict(subject=s, model=m, day=day, k=k, strategy="Start of session",
                         MSE=np.mean((yt[ev] - p[ev] - off) ** 2), weight=w))
        off_raw = np.mean(yt[:k] - p[:k]) if k else 0.0
        rows.append(dict(subject=s, model=m, day=day, k=k, strategy="Start of session (no shrinkage)",
                         MSE=np.mean((yt[ev] - p[ev] - off_raw) ** 2), weight=1.0 if k else 0.0))
        # (b) k spot-checks at random conditions; scored on the remaining conditions
        mses, mses_raw = [], []
        for _ in range(N_DRAWS):
            cal = rng.choice(len(te), k, replace=False)
            rest = np.setdiff1d(np.arange(len(te)), cal)
            off, w2 = shrunk_offset(yt[cal] - p[cal], sigma2, tau2)
            mses.append(np.mean((yt[rest] - p[rest] - off) ** 2))
            off_raw = np.mean(yt[cal] - p[cal]) if k else 0.0
            mses_raw.append(np.mean((yt[rest] - p[rest] - off_raw) ** 2))
        rows.append(dict(subject=s, model=m, day=day, k=k, strategy="Spot-checks", MSE=np.mean(mses), weight=w2))
        rows.append(dict(subject=s, model=m, day=day, k=k, strategy="Spot-checks (no shrinkage)",
                         MSE=np.mean(mses_raw), weight=1.0 if k else 0.0))
    return rows


# ---------------------------------------------------------------- E11
def _transfer_job(tr_s, te_s, data, m):
    """Train on one person (per-subject z-scored features), calibrate on k samples of the other."""
    A, B = data[tr_s], data[te_s]
    Xa = StandardScaler().fit_transform(A.features(ALL_FEATURES))
    Xb = StandardScaler().fit_transform(B.features(ALL_FEATURES))
    p = np.ravel(make_model(m).fit(Xa, A.y).predict(Xb))
    rng = np.random.RandomState(0)
    rows = []
    for k in [0, 3, 5, 10, 20, 36]:
        res = {"offset": [], "offset+scale": [], "own model on k only": [], "mean of k": []}
        for _ in range(N_DRAWS):
            cal = rng.choice(len(B.y), k, replace=False) if k else np.array([], int)
            rest = np.setdiff1d(np.arange(len(B.y)), cal)
            yr = B.y[rest]
            off = np.mean(B.y[cal] - p[cal]) if k else 0.0
            res["offset"].append(np.mean((yr - p[rest] - off) ** 2))
            if k >= 3:
                a, b = np.polyfit(p[cal], B.y[cal], 1)
                a = np.clip(a, 0, 2)
                b = np.mean(B.y[cal]) - a * np.mean(p[cal])
                res["offset+scale"].append(np.mean((yr - (a * p[rest] + b)) ** 2))
                own = make_model("Ridge").fit(B.features(ALL_FEATURES)[cal], B.y[cal])
                res["own model on k only"].append(np.mean((yr - own.predict(B.features(ALL_FEATURES)[rest])) ** 2))
                res["mean of k"].append(np.mean((yr - B.y[cal].mean()) ** 2))
        for strat, v in res.items():
            if v:
                rows.append(dict(train=tr_s, test=te_s, model=m, k=k, strategy=strat, MSE=float(np.mean(v))))
    return rows


# ---------------------------------------------------------------- E12
def _sensor_job(s, d, m, k, protocol):
    X, y = d.features(ALL_FEATURES), d.y
    emg_idx = tuple(ALL_FEATURES.index(f) for f in EMG_FEATURES)
    pipe = make_pipeline(TopKSensors(emg_idx, k), make_model(m))
    fm, _ = cross_validate(pipe, X, y, d.day, protocol, n_repeats=3)
    return dict(subject=s, model=m, n_emg=k, protocol=protocol, MSE=float(fm.mean()))


def main():
    data = load_all()
    t0 = time.time()

    log("E10 day calibration")
    out = Parallel(n_jobs=-1)(delayed(_calib_job)(s, d, m, day) for s, d in data.items() for m in MODELS
                              for day in range(5))
    e10 = pd.DataFrame([r for rows in out for r in rows])
    e10 = e10.groupby(["subject", "model", "strategy", "k"]).agg(MSE=("MSE", "mean"),
                                                                 weight=("weight", "mean")).reset_index()
    e10.to_csv(TABLES / "E10_day_calibration.csv", index=False)

    log("E11 cross-subject calibration")
    out = Parallel(n_jobs=-1)(delayed(_transfer_job)(a, b, data, m) for a, b in [("S1", "S2"), ("S2", "S1")]
                              for m in ["Ridge", "Gaussian Process"])
    pd.DataFrame([r for rows in out for r in rows]).to_csv(TABLES / "E11_cross_subject_calibration.csv", index=False)

    log("E12 sensor reduction")
    out = Parallel(n_jobs=-1)(delayed(_sensor_job)(s, d, m, k, p) for s, d in data.items()
                              for m in ["Ridge", "Gaussian Process"] for k in [0, 1, 2, 3, 4, 6, 8, 12, 16]
                              for p in ["random", "day"])
    pd.DataFrame(out).to_csv(TABLES / "E12_sensor_reduction.csv", index=False)
    # full-data channel ranking (for reporting which electrodes to keep)
    rank_rows = []
    for s, d in data.items():
        emg_idx = tuple(ALL_FEATURES.index(f) for f in EMG_FEATURES)
        r = TopKSensors(emg_idx, 16).fit(d.features(ALL_FEATURES), d.y).ranking_
        for pos, i in enumerate(r):
            rank_rows.append(dict(subject=s, rank=pos + 1, channel=ALL_FEATURES[i]))
    pd.DataFrame(rank_rows).to_csv(TABLES / "E12_channel_ranking.csv", index=False)
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
