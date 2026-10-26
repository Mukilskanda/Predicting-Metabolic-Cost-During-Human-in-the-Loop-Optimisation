"""Session calibration and sensor reduction.

Empirical-Bayes day-offset calibration
--------------------------------------
On a new day, a model trained on earlier days is often off by a roughly constant amount
(day-to-day drift: electrode placement, fatigue, motor learning). If k conditions of the new
day are also measured with the mask, the offset can be estimated from their residuals.
A plain mean of k noisy residuals is itself noisy, so we shrink it towards 0:

    offset = w * mean(y_cal - f(x_cal)),     w = k / (k + sigma^2 / tau^2)

    sigma^2  within-day residual variance  (measurement + model noise)
    tau^2    variance of true day offsets

Both are estimated *from the training days only*, by an inner leave-one-day-out loop. A subject
whose days are already well predicted (small tau^2) gets little correction, and a subject with
large day shifts gets a lot. This is the classic random-effects / James-Stein estimator.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import lars_path
from sklearn.preprocessing import StandardScaler


def estimate_offset_prior(model, X, y, days):
    """Inner leave-one-day-out on the training days -> (sigma2, tau2)."""
    means, within, sizes = [], [], []
    for d in np.unique(days):
        tr, te = days != d, days == d
        r = y[te] - np.ravel(clone(model).fit(X[tr], y[tr]).predict(X[te]))
        means.append(r.mean())
        within.append(r.var(ddof=1))
        sizes.append(te.sum())
    sigma2 = float(np.mean(within))
    # observed spread of day means = true offset variance + sampling noise of each mean
    tau2 = max(float(np.var(means, ddof=1)) - sigma2 / float(np.mean(sizes)), 1e-6)
    return sigma2, tau2


def shrunk_offset(residuals, sigma2, tau2):
    k = len(residuals)
    if k == 0:
        return 0.0, 0.0
    w = k / (k + sigma2 / tau2)
    return float(w * np.mean(residuals)), float(w)


class TopKSensors(BaseEstimator, TransformerMixin):
    """Keep all non-EMG columns plus the k EMG channels that enter the LASSO path first.

    Fitted inside each training fold, so the sensor choice never sees test data."""

    def __init__(self, emg_idx=(), k=16):
        self.emg_idx = emg_idx
        self.k = k

    def fit(self, X, y):
        emg = list(self.emg_idx)
        Xs = StandardScaler().fit_transform(X)
        _, active, coefs = lars_path(Xs, y - y.mean(), method="lasso", max_iter=500)
        order = [i for i in dict.fromkeys(active) if i in emg]
        # channels that never entered the path: rank by final |coef|
        rest = sorted(set(emg) - set(order), key=lambda i: -abs(coefs[i, -1]))
        self.ranking_ = order + rest
        keep = sorted(set(range(X.shape[1])) - set(emg)) + sorted(self.ranking_[: self.k])
        self.keep_ = np.array(keep)
        return self

    def transform(self, X):
        return X[:, self.keep_]
