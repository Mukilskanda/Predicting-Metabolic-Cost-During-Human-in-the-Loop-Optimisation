
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
DATA = HERE / "results" / "k_fold_results.mat"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)


def make_plot(s1_nn, s1_lr, s2_nn, s2_lr, title, filename):
    neurons = np.arange(1, len(s1_nn) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(neurons, s1_nn, "o-", label="S1")
    plt.axhline(s1_lr, linestyle="--", label="S1 Linear Regression")
    plt.plot(neurons, s2_nn, "o-", label="S2")
    plt.axhline(s2_lr, linestyle="--", label="S2 Linear Regression")

    plt.xlabel("Number of Neurons")
    plt.ylabel("MSE")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS / filename, dpi=300)
    plt.close()


def main():
    if not DATA.exists():
        raise FileNotFoundError("Run k_fold_cv.py first.")

    d = loadmat(DATA)

    make_plot(
        d["michael_all_nn_mse"].ravel(),
        float(d["michael_all_lr_mse"].ravel()[0]),
        d["eley_all_nn_mse"].ravel(),
        float(d["eley_all_lr_mse"].ravel()[0]),
        "All Features",
        "cv_all_features_python.png",
    )

    make_plot(
        d["michael_step_emg_nn_mse"].ravel(),
        float(d["michael_step_emg_lr_mse"].ravel()[0]),
        d["eley_step_emg_nn_mse"].ravel(),
        float(d["eley_step_emg_lr_mse"].ravel()[0]),
        "Step + EMG",
        "cv_no_control_python.png",
    )

    make_plot(
        d["michael_step_only_nn_mse"].ravel(),
        float(d["michael_step_only_lr_mse"].ravel()[0]),
        d["eley_step_only_nn_mse"].ravel(),
        float(d["eley_step_only_lr_mse"].ravel()[0]),
        "Step Only",
        "cv_step_only_python.png",
    )

    make_plot(
        d["michael_emg_only_nn_mse"].ravel(),
        float(d["michael_emg_only_lr_mse"].ravel()[0]),
        d["eley_emg_only_nn_mse"].ravel(),
        float(d["eley_emg_only_lr_mse"].ravel()[0]),
        "EMG Only",
        "cv_emg_only_python.png",
    )

    print("Plots generated in results/.")


if __name__ == "__main__":
    main()
