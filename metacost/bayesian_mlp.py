"""Bayesian-regularised one-hidden-layer neural network (MacKay 1992; Foresee & Hagan 1997).

A from-scratch NumPy re-implementation of MATLAB's ``trainbr`` - the training algorithm
used in the original study - wrapped as a scikit-learn regressor.

Network:      f(x) = w2 . tanh(W1 x + b1) + b2          (tanh hidden, linear output)
Objective:    F(w) = beta * E_D + alpha * E_W,   E_D = 1/2 sum e^2,  E_W = 1/2 sum w^2
Optimiser:    Levenberg-Marquardt   dw = -(beta J'J + alpha I + mu I)^-1 (beta J'e' + alpha w)
Evidence:     gamma = P - alpha tr(H^-1),  alpha = gamma / 2E_W,  beta = (N - gamma) / 2E_D
              (re-estimated after every accepted step, as trainbr does)

alpha/beta are learned from the training data via the Bayesian evidence, so no separate
validation split is needed - which suits ~180 noisy samples. ``gamma_`` (the effective
number of parameters) is exposed so you can see how much the network was regularised.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class BayesianRegularizedMLP(BaseEstimator, RegressorMixin):
    def __init__(self, n_hidden: int = 2, max_epochs: int = 150, n_restarts: int = 3,
                 random_state: int | None = 0):
        self.n_hidden = n_hidden
        self.max_epochs = max_epochs
        self.n_restarts = n_restarts
        self.random_state = random_state

    # ---- network maths ---------------------------------------------------------------
    def _unpack(self, w, d):
        H = self.n_hidden
        W1 = w[: H * d].reshape(H, d)
        b1 = w[H * d: H * d + H]
        w2 = w[H * d + H: H * d + 2 * H]
        b2 = w[-1]
        return W1, b1, w2, b2

    def _forward(self, w, X):
        W1, b1, w2, b2 = self._unpack(w, X.shape[1])
        h = np.tanh(X @ W1.T + b1)
        return h @ w2 + b2, h, w2

    def _jacobian(self, w, X):
        """d f / d w  (N x P), closed form."""
        f, h, w2 = self._forward(w, X)
        dh = w2 * (1 - h ** 2)                                   # N x H
        dW1 = (dh[:, :, None] * X[:, None, :]).reshape(len(X), -1)
        return f, np.hstack([dW1, dh, h, np.ones((len(X), 1))])

    def _fit_once(self, X, y, rng):
        n, d = X.shape
        P = self.n_hidden * (d + 2) + 1
        w = rng.normal(0, 1 / np.sqrt(d), P)
        w[-self.n_hidden - 1:] = rng.normal(0, 1 / np.sqrt(self.n_hidden), self.n_hidden + 1)
        alpha, beta, mu = 0.0, 1.0, 0.005
        I = np.eye(P)

        f, J = self._jacobian(w, X)
        e = y - f
        Ed, Ew = 0.5 * e @ e, 0.5 * w @ w
        gamma = P
        for _ in range(self.max_epochs):
            F = beta * Ed + alpha * Ew
            JtJ = J.T @ J
            g = -beta * J.T @ e + alpha * w
            while True:
                dw = -np.linalg.solve(beta * JtJ + (alpha + mu) * I, g)
                w_new = w + dw
                e_new = y - self._forward(w_new, X)[0]
                Ed_new, Ew_new = 0.5 * e_new @ e_new, 0.5 * w_new @ w_new
                if beta * Ed_new + alpha * Ew_new < F:
                    mu = max(mu * 0.1, 1e-20)
                    break
                mu *= 10
                if mu > 1e10:
                    break
            if mu > 1e10:
                break
            w, Ed, Ew = w_new, Ed_new, Ew_new
            f, J = self._jacobian(w, X)
            e = y - f
            # evidence re-estimation of the hyper-parameters
            Hm = beta * J.T @ J + alpha * I
            gamma = P - alpha * np.trace(np.linalg.inv(Hm)) if alpha > 0 else P
            gamma = float(np.clip(gamma, 1e-3, min(P, n - 1)))
            alpha = gamma / (2 * Ew + 1e-12)
            beta = (n - gamma) / (2 * Ed + 1e-12)
            if np.linalg.norm(g) < 1e-7:
                break

        Hm = beta * J.T @ J + alpha * I
        log_ev = (-beta * Ed - alpha * Ew - 0.5 * np.linalg.slogdet(Hm)[1]
                  + 0.5 * P * np.log(alpha) + 0.5 * n * np.log(beta))
        return w, log_ev, alpha, beta, gamma

    # ---- sklearn API -----------------------------------------------------------------
    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=np.float64)
        self.x_mean_, self.x_std_ = X.mean(0), X.std(0) + 1e-12
        self.y_mean_, self.y_std_ = y.mean(), y.std() + 1e-12
        Xs, ys = (X - self.x_mean_) / self.x_std_, (y - self.y_mean_) / self.y_std_
        rng = np.random.RandomState(self.random_state)
        best = None
        for _ in range(self.n_restarts):
            res = self._fit_once(Xs, ys, rng)
            if np.isfinite(res[1]) and (best is None or res[1] > best[1]):
                best = res
        self.coef_, self.log_evidence_, self.alpha_, self.beta_, self.gamma_ = best
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        X = check_array(X, dtype=np.float64)
        f = self._forward(self.coef_, (X - self.x_mean_) / self.x_std_)[0]
        return f * self.y_std_ + self.y_mean_
