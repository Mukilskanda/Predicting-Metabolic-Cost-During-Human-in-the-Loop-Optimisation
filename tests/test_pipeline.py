"""Run with:  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metacost.bayesian_mlp import BayesianRegularizedMLP
from metacost.data import ALL_FEATURES, FEATURE_SETS, load_subject
from metacost.evaluation import cross_validate
from metacost.models import make_model


@pytest.mark.parametrize("s,n", [("S1", 180), ("S2", 179)])
def test_data_shapes(s, n):
    d = load_subject(s)
    assert d.X.shape == (n, 29) and list(d.X.columns) == ALL_FEATURES
    assert (d.y >= 0).all() and set(np.unique(d.day)) == set(range(5))


def test_target_variance_matches_paper():
    # paper Table I "constant prediction MSE": 0.0657 (S1) / 0.0465 (S2, before dropping the artefact)
    assert abs(load_subject("S1").y.var() - 0.0657) < 1e-3
    assert abs(load_subject("S2", drop_invalid=False).y.var() - 0.0465) < 1e-3


def test_jacobian_matches_finite_differences():
    rng = np.random.RandomState(0)
    X = rng.randn(6, 4)
    m = BayesianRegularizedMLP(n_hidden=3)
    w = rng.randn(3 * 6 + 1)
    f, J = m._jacobian(w, X)
    eps = 1e-6
    for j in range(len(w)):
        w2 = w.copy(); w2[j] += eps
        assert np.allclose((m._forward(w2, X)[0] - f) / eps, J[:, j], atol=1e-4)


def test_br_mlp_learns_a_sine():
    rng = np.random.RandomState(0)
    X = rng.uniform(-3, 3, (120, 1))
    y = np.sin(X[:, 0]) + 0.05 * rng.randn(120)
    m = BayesianRegularizedMLP(n_hidden=4).fit(X, y)
    assert np.mean((m.predict(X) - y) ** 2) < 0.01
    assert 0 < m.gamma_ <= 4 * 3 + 1


def test_models_beat_constant_baseline():
    d = load_subject("S1")
    X = d.features(FEATURE_SETS["All features"])
    base, _ = cross_validate(make_model("Constant (mean)"), X, d.y, n_repeats=1)
    for name in ["LASSO", "Gaussian Process"]:
        mse, _ = cross_validate(make_model(name), X, d.y, n_repeats=1)
        assert mse.mean() < 0.3 * base.mean()


def test_leave_one_day_out_has_5_folds():
    d = load_subject("S2")
    mse, oof = cross_validate(make_model("LASSO"), d.features(ALL_FEATURES), d.y, d.day, "day")
    assert len(mse) == 5 and oof.shape == d.y.shape


def test_shrinkage_weight_behaviour():
    from metacost.calibration import shrunk_offset
    r = np.array([0.1, 0.1, 0.1])
    off_noisy, w_noisy = shrunk_offset(r, sigma2=1.0, tau2=0.01)    # tiny day effect -> trust little
    off_real, w_real = shrunk_offset(r, sigma2=0.01, tau2=1.0)      # big day effect -> trust a lot
    assert w_noisy < 0.1 < 0.9 < w_real and abs(off_real - 0.1) < 0.01
    assert shrunk_offset(np.array([]), 1, 1) == (0.0, 0.0)


def test_top_k_sensors_keeps_non_emg_and_k_channels():
    from metacost.calibration import TopKSensors
    from metacost.data import EMG_FEATURES
    d = load_subject("S1")
    X = d.features(ALL_FEATURES)
    emg_idx = tuple(ALL_FEATURES.index(f) for f in EMG_FEATURES)
    t = TopKSensors(emg_idx, 3).fit(X, d.y)
    assert t.transform(X).shape[1] == 29 - 16 + 3
    assert sorted(t.ranking_) == sorted(emg_idx)


def test_paper_reproduction_loader_drops_artefact():
    import importlib.util
    path = Path(__file__).resolve().parent.parent / "paper_reproduction" / "data_processing.py"
    spec = importlib.util.spec_from_file_location("pr_data", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    d = mod.load_processed_data()
    assert d["S1"]["X"].shape == (180, 29) and d["S2"]["X"].shape == (179, 29)
    assert (d["S2"]["y"] >= 0).all()
