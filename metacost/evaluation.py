"""Cross-validation protocols and metrics."""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut, RepeatedKFold

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def metrics(y, p) -> dict:
    mse = mean_squared_error(y, p)
    return {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": mean_absolute_error(y, p),
        "R2": r2_score(y, p),
        # error relative to the normalised walking metabolic rate (=1.0), as in the paper
        "pct_error": float(np.sqrt(mse)) * 100,
        "pearson_r": float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 else 0.0,
    }


def splitter(protocol: str, n_repeats: int = 5, seed: int = 0):
    if protocol == "random":           # pooled days, shuffled K-fold (the paper used 10-fold)
        return RepeatedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed)
    if protocol == "day":              # stricter: predict a whole unseen experiment day
        return LeaveOneGroupOut()
    raise KeyError(protocol)


def cross_validate(model, X, y, groups=None, protocol="random", n_repeats=5, seed=0):
    """Returns per-fold MSE list and out-of-fold predictions (averaged over repeats)."""
    cv = splitter(protocol, n_repeats, seed)
    fold_mse, oof_sum, oof_cnt = [], np.zeros(len(y)), np.zeros(len(y))
    for tr, te in cv.split(X, y, groups if protocol == "day" else None):
        m = clone(model).fit(X[tr], y[tr])
        p = np.ravel(m.predict(X[te]))
        fold_mse.append(mean_squared_error(y[te], p))
        oof_sum[te] += p
        oof_cnt[te] += 1
    return np.array(fold_mse), oof_sum / np.maximum(oof_cnt, 1)


def summarise(fold_mse, y, oof):
    out = metrics(y, oof)
    out["MSE"] = float(np.mean(fold_mse))          # mean of fold MSEs, as in the paper
    out["MSE_std"] = float(np.std(fold_mse))
    out["RMSE"] = float(np.sqrt(out["MSE"]))
    out["pct_error"] = out["RMSE"] * 100
    return out
