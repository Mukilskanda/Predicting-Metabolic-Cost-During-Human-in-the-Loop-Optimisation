
from pathlib import Path
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_processing import load_processed_data

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

# Replace these after running model_selection.py.
SUBJECT = "S1"
BEST_CONFIGURATION = "all"  # all | step_emg | step_only | emg_only
BEST_NEURONS = 1


def main():
    data = load_processed_data()
    X = data[SUBJECT]["X"]
    y = data[SUBJECT]["y"]

    if BEST_CONFIGURATION == "all":
        feature_idx = np.arange(X.shape[1])
    elif BEST_CONFIGURATION == "step_emg":
        feature_idx = np.arange(25)
    elif BEST_CONFIGURATION == "step_only":
        feature_idx = np.arange(9)
    elif BEST_CONFIGURATION == "emg_only":
        feature_idx = np.arange(9, 25)
    else:
        raise ValueError("Unknown BEST_CONFIGURATION.")

    X_final = X[:, feature_idx]

    model = make_pipeline(StandardScaler(), MLPRegressor(
        hidden_layer_sizes=(BEST_NEURONS,),
        activation="tanh",
        solver="lbfgs",
        alpha=1e-4,
        max_iter=3000,
        random_state=42,
    ))

    # Out-of-fold predictions: every point is predicted by a model that did not see it.
    # (Plotting predictions of a model fitted on all the data would look far better than reality.)
    oof = cross_val_predict(model, X_final, y, cv=KFold(5, shuffle=True, random_state=42))
    cv_mse = np.mean((y - oof) ** 2)

    model.fit(X_final, y)
    train_mse = np.mean((y - model.predict(X_final)) ** 2)

    print(f"Subject: {SUBJECT}  Configuration: {BEST_CONFIGURATION}  Hidden neurons: {BEST_NEURONS}")
    print(f"5-fold out-of-fold MSE: {cv_mse:.6f}   (training MSE {train_mse:.6f} - not a fair score)")

    joblib.dump(
        {
            "model": model,  # pipeline: scaler + network
            "feature_idx": feature_idx,
            "configuration": BEST_CONFIGURATION,
            "neurons": BEST_NEURONS,
        },
        RESULTS / "final_model.joblib",
    )

    plt.figure(figsize=(10, 5))
    plt.plot(y, label="Actual")
    plt.plot(oof, "--", label="Predicted (out-of-fold)")
    plt.xlabel("Sample")
    plt.ylabel("Metabolic Cost")
    plt.title(f"Metabolic Cost Prediction - {SUBJECT}, 5-fold out-of-fold (MSE {cv_mse:.4f})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS / "final_prediction.png", dpi=200)
    plt.close()

    print(f"Saved model: {RESULTS / 'final_model.joblib'}")


if __name__ == "__main__":
    main()
