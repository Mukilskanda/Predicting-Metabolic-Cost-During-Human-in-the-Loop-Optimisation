from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.io import savemat

from data_processing import load_processed_data
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
RESULT_FILE = HERE / "results" / "k_fold_results.mat"

N_SPLITS = 5
MAX_NEURONS = 14
RANDOM_STATE = 42


def clean_data(X, y):
    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    return X[valid], y[valid]


def evaluate_configuration(X, y):
    kf = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    nn_mse = []

    for neurons in range(1, MAX_NEURONS + 1):
        fold_mse = []

        for train_idx, test_idx in kf.split(X):
            # MATLAB used trainbr. MLPRegressor does not have an exact
            # Bayesian-regularization equivalent, so L2 regularization
            # (alpha) is used as the practical Python replacement.
            # Scaler is fitted on the training fold only (no test-fold leakage).
            model = make_pipeline(StandardScaler(), MLPRegressor(
                hidden_layer_sizes=(neurons,),
                activation="tanh",
                solver="lbfgs",
                alpha=1e-4,
                max_iter=5000,
                random_state=RANDOM_STATE,
            ))

            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            fold_mse.append(mean_squared_error(y[test_idx], pred))

        nn_mse.append(float(np.mean(fold_mse)))

    lr_fold_mse = []

    for train_idx, test_idx in kf.split(X):
        model = make_pipeline(StandardScaler(), LinearRegression())
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        lr_fold_mse.append(mean_squared_error(y[test_idx], pred))

    lr_mse = float(np.mean(lr_fold_mse))
    return np.asarray(nn_mse), lr_mse


def main():
    data = load_processed_data()

    results = {}

    for subject in ("S1", "S2"):
        X, y = clean_data(data[subject]["X"], data[subject]["y"])

        # Original processed data has 9 step features, 16 EMG features,
        # and 4 control parameters.
        if X.shape[1] < 20:
            raise ValueError(
                f"{subject} has only {X.shape[1]} features. "
                "The expected project layout needs at least 20."
            )

        all_features = X
        step_emg = X[:, :-4]
        step_only = X[:, :9]
        emg_only = X[:, 9:25]

        configs = {
            "all": all_features,
            "step_emg": step_emg,
            "step_only": step_only,
            "emg_only": emg_only,
        }

        results[subject] = {}

        for name, features in configs.items():
            nn_mse, lr_mse = evaluate_configuration(features, y)
            results[subject][name] = {
                "nn_mse": nn_mse,
                "lr_mse": lr_mse,
            }

            best_neuron = int(np.argmin(nn_mse) + 1)
            print(
                f"{subject:>2} | {name:<10} | "
                f"NN MSE={nn_mse[best_neuron-1]:.6f} "
                f"({best_neuron} neurons) | "
                f"LR MSE={lr_mse:.6f}"
            )

    # Save a MATLAB-readable result file too.
    mat = {}
    for subject, configs in results.items():
        for name, values in configs.items():
            prefix = "michael" if subject == "S1" else "eley"
            mat[f"{prefix}_{name}_nn_mse"] = values["nn_mse"]
            mat[f"{prefix}_{name}_lr_mse"] = values["lr_mse"]

    savemat(RESULT_FILE, mat)
    print(f"\nSaved: {RESULT_FILE}")


if __name__ == "__main__":
    main()
