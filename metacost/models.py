"""Model zoo. Every model is a scikit-learn Pipeline that standardises inputs itself, so the
scaler is fitted on training folds only (no test-fold leakage), and every hyper-parameter
is tuned by an *inner* CV - the outer CV score is therefore an honest (nested) estimate."""
from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel, DotProduct
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .bayesian_mlp import BayesianRegularizedMLP

INNER = KFold(5, shuffle=True, random_state=1)


def gp_kernel():
    # linear trend + smooth non-linear part + learned observation noise
    return (ConstantKernel(1.0) * DotProduct(sigma_0=1.0)
            + ConstantKernel(1.0) * RBF(length_scale=5.0, length_scale_bounds=(1e-1, 1e3))
            + WhiteKernel(0.5, noise_level_bounds=(1e-4, 10)))


def make_model(name: str, n_hidden: int = 2):
    s = StandardScaler()
    if name == "Constant (mean)":
        return DummyRegressor()
    if name == "Linear (OLS)":
        return make_pipeline(s, LinearRegression())
    if name == "LASSO":
        return make_pipeline(s, LassoCV(alphas=np.logspace(-5, -1, 25), cv=INNER, max_iter=50000))
    if name == "Ridge":
        return make_pipeline(s, RidgeCV(alphas=np.logspace(-3, 3, 30)))
    if name == "Elastic Net":
        return make_pipeline(s, ElasticNetCV(l1_ratio=[.1, .5, .9], alphas=np.logspace(-5, 0, 25),
                                             cv=INNER, max_iter=50000))
    if name == "PLS":
        return GridSearchCV(make_pipeline(s, PLSRegression(scale=False)),
                            {"plsregression__n_components": [1, 2, 3, 4, 6, 8]},
                            cv=INNER, scoring="neg_mean_squared_error")
    if name == "SVR (RBF)":
        return GridSearchCV(make_pipeline(s, SVR()),
                            {"svr__C": [0.3, 1, 3, 10], "svr__epsilon": [0.02, 0.05, 0.1],
                             "svr__gamma": ["scale", 0.01]},
                            cv=INNER, scoring="neg_mean_squared_error")
    if name == "kNN":
        return GridSearchCV(make_pipeline(s, KNeighborsRegressor(weights="distance")),
                            {"kneighborsregressor__n_neighbors": [3, 5, 8, 12, 20]},
                            cv=INNER, scoring="neg_mean_squared_error")
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=400, min_samples_leaf=2, max_features=0.5,
                                     random_state=0, n_jobs=1)
    if name == "Extra Trees":
        return ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2, max_features=0.5,
                                   random_state=0, n_jobs=1)
    if name == "Gradient Boosting":
        return GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=2,
                                         subsample=0.8, random_state=0)
    if name == "Gaussian Process":
        return make_pipeline(s, GaussianProcessRegressor(gp_kernel(), normalize_y=True,
                                                         n_restarts_optimizer=2, random_state=0))
    if name == "Bayesian-reg. NN":
        return BayesianRegularizedMLP(n_hidden=n_hidden)
    raise KeyError(name)


MODEL_NAMES = ["Constant (mean)", "Linear (OLS)", "LASSO", "Ridge", "Elastic Net", "PLS",
               "kNN", "SVR (RBF)", "Random Forest", "Extra Trees", "Gradient Boosting",
               "Gaussian Process", "Bayesian-reg. NN"]
